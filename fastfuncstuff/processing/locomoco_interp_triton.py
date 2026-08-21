"""Fused CUDA interpolation for locomoco's active warp dimensions.

The tensor rank is not the interpolation dimensionality: a single-PE field in
a 3-D EPI is still a 1-D resample, while dual PE is a 2-D resample embedded in
that volume.  These kernels sample only the axes the displacement can move.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _lanczos_weight(frac, k: tl.constexpr, RADIUS: tl.constexpr):
    x = frac - k
    pix = 3.141592653589793 * tl.where(tl.abs(x) < 1.0e-7, 1.0, x)
    sinc = tl.where(tl.abs(x) < 1.0e-7, 1.0, tl.sin(pix) / pix)
    xr = x / RADIUS
    pixr = 3.141592653589793 * tl.where(tl.abs(xr) < 1.0e-7, 1.0, xr)
    window = tl.where(tl.abs(xr) < 1.0e-7, 1.0, tl.sin(pixr) / pixr)
    return sinc * window


@triton.jit
def _keys_weight(frac, k: tl.constexpr):
    # PyTorch grid_sample bicubic uses Keys cubic convolution with alpha=-0.75.
    a = -0.75
    if k == -1:
        x = frac + 1.0
        return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
    elif k == 0:
        x = frac
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    elif k == 1:
        x = 1.0 - frac
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    else:
        x = 2.0 - frac
        return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a


@triton.jit
def _shift1d_kernel(
    inp,
    shift,
    out,
    n_values,
    axis_size,
    axis_stride,
    start,
    n_chunk,
    RADIUS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    local = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    idx = start + local
    mask = (local < n_chunk) & (idx < n_values)
    pos = (idx // axis_stride) % axis_size
    row = idx - pos * axis_stride
    coord = pos + tl.load(shift + idx, mask=mask, other=0.0)
    base = tl.floor(coord).to(tl.int32)
    frac = coord - base
    num = tl.zeros([BLOCK], tl.float32)
    den = tl.zeros([BLOCK], tl.float32)
    for k in tl.static_range(-(RADIUS - 1), RADIUS + 1):
        tap = tl.minimum(tl.maximum(base + k, 0), axis_size - 1)
        w = _lanczos_weight(frac, k, RADIUS)
        val = tl.load(inp + row + tap * axis_stride, mask=mask, other=0.0)
        num += w * val
        den += w
    tl.store(out + idx, num / den, mask=mask)


@triton.jit
def _shift2d_kernel(
    inp,
    shift0,
    shift1,
    out,
    n_values,
    size0,
    size1,
    stride0,
    stride1,
    start,
    n_chunk,
    MODE: tl.constexpr,
    RADIUS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    local = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    idx = start + local
    mask = (local < n_chunk) & (idx < n_values)
    p0 = (idx // stride0) % size0
    p1 = (idx // stride1) % size1
    row = idx - p0 * stride0 - p1 * stride1
    c0 = p0 + tl.load(shift0 + idx, mask=mask, other=0.0)
    c1 = p1 + tl.load(shift1 + idx, mask=mask, other=0.0)
    b0, b1 = tl.floor(c0).to(tl.int32), tl.floor(c1).to(tl.int32)
    f0, f1 = c0 - b0, c1 - b1
    num = tl.zeros([BLOCK], tl.float32)
    den0 = tl.zeros([BLOCK], tl.float32)
    den1 = tl.zeros([BLOCK], tl.float32)
    if MODE == 1:
        for k0 in tl.static_range(-(RADIUS - 1), RADIUS + 1):
            t0 = tl.minimum(tl.maximum(b0 + k0, 0), size0 - 1)
            w0 = _lanczos_weight(f0, k0, RADIUS)
            den0 += w0
            for k1 in tl.static_range(-(RADIUS - 1), RADIUS + 1):
                t1 = tl.minimum(tl.maximum(b1 + k1, 0), size1 - 1)
                w1 = _lanczos_weight(f1, k1, RADIUS)
                if k0 == -(RADIUS - 1):
                    den1 += w1
                val = tl.load(inp + row + t0 * stride0 + t1 * stride1, mask=mask, other=0.0)
                num += val * w0 * w1
        num /= den0 * den1
    else:
        for k0 in tl.static_range(-1, 3):
            t0 = tl.minimum(tl.maximum(b0 + k0, 0), size0 - 1)
            w0 = _keys_weight(f0, k0)
            for k1 in tl.static_range(-1, 3):
                t1 = tl.minimum(tl.maximum(b1 + k1, 0), size1 - 1)
                w1 = _keys_weight(f1, k1)
                val = tl.load(inp + row + t0 * stride0 + t1 * stride1, mask=mask, other=0.0)
                num += val * w0 * w1
    tl.store(out + idx, num, mask=mask)


def shift1d_lanczos_triton(vol: Tensor, shift: Tensor, dim: int, radius: int) -> Tensor:
    """Lanczos shift along one tensor dimension; CUDA float32, no autograd."""
    if not vol.is_contiguous():
        vol = vol.contiguous()
    shift = shift.expand(vol.shape).contiguous()
    out = torch.empty_like(vol)
    from ..memory import bytes_per_voxel_locomoco_interp, get_available_memory

    avail = get_available_memory(vol.device, empty_cache=False)
    chunk = max(1, min(vol.numel(), avail // bytes_per_voxel_locomoco_interp(1)))
    block = 256
    for start in range(0, vol.numel(), chunk):
        n_chunk = min(chunk, vol.numel() - start)
        _shift1d_kernel[(triton.cdiv(n_chunk, block),)](
            vol,
            shift,
            out,
            vol.numel(),
            vol.shape[dim],
            vol.stride(dim),
            start,
            n_chunk,
            RADIUS=radius,
            BLOCK=block,
        )
    return out


def shift2d_triton(
    vol: Tensor,
    shift0: Tensor,
    shift1: Tensor,
    dim0: int,
    dim1: int,
    mode: str,
    radius: int = 3,
) -> Tensor:
    """Cubic-convolution or Lanczos shift along two tensor dimensions."""
    if mode not in ("bicubic", "lanczos"):
        raise ValueError(f"unsupported 2-D fused mode {mode!r}")
    if not vol.is_contiguous():
        vol = vol.contiguous()
    shift0 = shift0.expand(vol.shape).contiguous()
    shift1 = shift1.expand(vol.shape).contiguous()
    out = torch.empty_like(vol)
    from ..memory import bytes_per_voxel_locomoco_interp, get_available_memory

    avail = get_available_memory(vol.device, empty_cache=False)
    chunk = max(1, min(vol.numel(), avail // bytes_per_voxel_locomoco_interp(2)))
    block = 128
    for start in range(0, vol.numel(), chunk):
        n_chunk = min(chunk, vol.numel() - start)
        _shift2d_kernel[(triton.cdiv(n_chunk, block),)](
            vol,
            shift0,
            shift1,
            out,
            vol.numel(),
            vol.shape[dim0],
            vol.shape[dim1],
            vol.stride(dim0),
            vol.stride(dim1),
            start,
            n_chunk,
            MODE=1 if mode == "lanczos" else 2,
            RADIUS=radius,
            BLOCK=block,
        )
    return out
