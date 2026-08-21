"""Forward-only fused CUDA kernels for high-order 3-D interpolation.

The portable implementation in :mod:`fastfuncstuff.processing.interp` builds
per-axis weight and int64 index tables before its compiled gather.  This kernel
keeps coordinates, weights, indices, and the tensor-product contraction inside
one launch.  It is deliberately forward-only: callers requiring coordinate
gradients continue through the PyTorch implementation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

_MODE_ID = {"cubic": 2, "quintic": 4, "heptic": 5, "wsinc5": 72}


@triton.jit
def _weight(x, k: tl.constexpr, MODE: tl.constexpr):
    x2 = x * x
    x2m1 = x2 - 1.0
    x2m4 = x2 - 4.0
    x2m9 = x2 - 9.0
    if MODE == 2:
        if k == -1:
            return -x * (x - 1.0) * (x - 2.0) * (1.0 / 6.0)
        elif k == 0:
            return 3.0 * (x + 1.0) * (x - 1.0) * (x - 2.0) * (1.0 / 6.0)
        elif k == 1:
            return -3.0 * x * (x + 1.0) * (x - 2.0) * (1.0 / 6.0)
        else:
            return x * (x + 1.0) * (x - 1.0) * (1.0 / 6.0)
    elif MODE == 4:
        if k == -2:
            return x * x2m1 * (2.0 - x) * (x - 3.0) * 0.008333333
        elif k == -1:
            return x * x2m4 * (x - 1.0) * (x - 3.0) * 0.041666667
        elif k == 0:
            return x2m4 * x2m1 * (3.0 - x) * 0.083333333
        elif k == 1:
            return x * x2m4 * (x + 1.0) * (x - 3.0) * 0.083333333
        elif k == 2:
            return x * x2m1 * (x + 2.0) * (3.0 - x) * 0.041666667
        else:
            return x * x2m1 * x2m4 * 0.008333333
    elif MODE == 5:
        if k == -3:
            return x * x2m1 * x2m4 * (x - 3.0) * (4.0 - x) * 0.0001984126984
        elif k == -2:
            return x * x2m1 * (x - 2.0) * x2m9 * (x - 4.0) * 0.001388888889
        elif k == -1:
            return x * (x - 1.0) * x2m4 * x2m9 * (4.0 - x) * 0.004166666667
        elif k == 0:
            return x2m1 * x2m4 * x2m9 * (x - 4.0) * 0.006944444444
        elif k == 1:
            return x * (x + 1.0) * x2m4 * x2m9 * (4.0 - x) * 0.006944444444
        elif k == 2:
            return x * x2m1 * (x + 2.0) * x2m9 * (x - 4.0) * 0.004166666667
        elif k == 3:
            return x * x2m1 * x2m4 * (x + 3.0) * (4.0 - x) * 0.001388888889
        else:
            return x * x2m1 * x2m4 * x2m9 * 0.0001984126984
    else:
        d = tl.abs(x - k)
        pid = 3.141592653589793 * tl.where(d < 1.0e-7, 1.0, d)
        sinc = tl.where(d < 1.0e-7, 1.0, tl.sin(pid) / pid)
        arg = 3.141592653589793 * d / 5.001
        win = 0.4243801 + 0.4973406 * tl.cos(arg) + 0.0782793 * tl.cos(2.0 * arg)
        return sinc * win


@triton.jit
def _resample_kernel(
    source,
    xp,
    yp,
    zp,
    out,
    n_points,
    nx: tl.constexpr,
    ny: tl.constexpr,
    nz: tl.constexpr,
    MODE: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pmask = p < n_points
    x = tl.load(xp + p, mask=pmask, other=-1.0)
    y = tl.load(yp + p, mask=pmask, other=-1.0)
    z = tl.load(zp + p, mask=pmask, other=-1.0)
    inside = (
        pmask
        & (x >= -0.5)
        & (x <= nx - 0.5)
        & (y >= -0.5)
        & (y <= ny - 0.5)
        & (z >= -0.5)
        & (z <= nz - 0.5)
    )
    xb = tl.floor(x).to(tl.int32)
    yb = tl.floor(y).to(tl.int32)
    zb = tl.floor(z).to(tl.int32)
    fx, fy, fz = x - xb, y - yb, z - zb

    # AFNI wsinc5 normalizes each axis. Polynomial kernels already sum to one.
    sx = tl.zeros([BLOCK], tl.float32)
    sy = tl.zeros([BLOCK], tl.float32)
    sz = tl.zeros([BLOCK], tl.float32)
    for k in tl.static_range(-(H - 1), H + 1):
        sx += _weight(fx, k, MODE)
        sy += _weight(fy, k, MODE)
        sz += _weight(fz, k, MODE)
    norm = tl.where(
        MODE == 72,
        1.0 / (tl.maximum(sx, 1.0e-10) * tl.maximum(sy, 1.0e-10) * tl.maximum(sz, 1.0e-10)),
        1.0,
    )

    acc = tl.zeros([BLOCK], tl.float32)
    for kz in tl.static_range(-(H - 1), H + 1):
        iz = tl.minimum(tl.maximum(zb + kz, 0), nz - 1)
        wz = _weight(fz, kz, MODE)
        for ky in tl.static_range(-(H - 1), H + 1):
            iy = tl.minimum(tl.maximum(yb + ky, 0), ny - 1)
            wyz = _weight(fy, ky, MODE) * wz
            for kx in tl.static_range(-(H - 1), H + 1):
                ix = tl.minimum(tl.maximum(xb + kx, 0), nx - 1)
                val = tl.load(source + (iz * ny + iy) * nx + ix, mask=inside, other=0.0)
                acc += val * _weight(fx, kx, MODE) * wyz
    tl.store(out + p, acc * norm, mask=pmask)


@triton.jit
def _resample_xy_kernel(
    source,
    xp,
    yp,
    zp,
    planes,
    n_points,
    nx: tl.constexpr,
    ny: tl.constexpr,
    nz: tl.constexpr,
    MODE: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Contract X/Y for one Z tap per program-row."""
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    kt = tl.program_id(1)
    kz = kt - (H - 1)
    pmask = p < n_points
    x = tl.load(xp + p, mask=pmask, other=-1.0)
    y = tl.load(yp + p, mask=pmask, other=-1.0)
    z = tl.load(zp + p, mask=pmask, other=-1.0)
    inside = (
        pmask
        & (x >= -0.5)
        & (x <= nx - 0.5)
        & (y >= -0.5)
        & (y <= ny - 0.5)
        & (z >= -0.5)
        & (z <= nz - 0.5)
    )
    xb, yb, zb = tl.floor(x).to(tl.int32), tl.floor(y).to(tl.int32), tl.floor(z).to(tl.int32)
    fx, fy = x - xb, y - yb
    iz = tl.minimum(tl.maximum(zb + kz, 0), nz - 1)
    sx = tl.zeros([BLOCK], tl.float32)
    sy = tl.zeros([BLOCK], tl.float32)
    acc = tl.zeros([BLOCK], tl.float32)
    for ky in tl.static_range(-(H - 1), H + 1):
        iy = tl.minimum(tl.maximum(yb + ky, 0), ny - 1)
        wy = _weight(fy, ky, MODE)
        sy += wy
        for kx in tl.static_range(-(H - 1), H + 1):
            ix = tl.minimum(tl.maximum(xb + kx, 0), nx - 1)
            wx = _weight(fx, kx, MODE)
            if ky == -(H - 1):
                sx += wx
            val = tl.load(source + (iz * ny + iy) * nx + ix, mask=inside, other=0.0)
            acc += val * wx * wy
    norm = tl.where(MODE == 72, 1.0 / (tl.maximum(sx, 1.0e-10) * tl.maximum(sy, 1.0e-10)), 1.0)
    tl.store(planes + kt * n_points + p, acc * norm, mask=pmask)


@triton.jit
def _resample_z_kernel(
    planes, zp, out, n_points, MODE: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr
):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pmask = p < n_points
    z = tl.load(zp + p, mask=pmask, other=0.0)
    fz = z - tl.floor(z)
    acc = tl.zeros([BLOCK], tl.float32)
    sz = tl.zeros([BLOCK], tl.float32)
    kt = 0
    for kz in tl.static_range(-(H - 1), H + 1):
        wz = _weight(fz, kz, MODE)
        sz += wz
        acc += tl.load(planes + kt * n_points + p, mask=pmask, other=0.0) * wz
        kt += 1
    norm = tl.where(MODE == 72, 1.0 / tl.maximum(sz, 1.0e-10), 1.0)
    tl.store(out + p, acc * norm, mask=pmask)


def separable_resample_3d_triton(
    source: Tensor, x: Tensor, y: Tensor, z: Tensor, kernel: str
) -> Tensor:
    """Resample one contiguous float32 CUDA volume without autograd."""
    if source.ndim != 3 or source.dtype != torch.float32 or source.device.type != "cuda":
        raise ValueError("fused interpolation requires a 3-D CUDA float32 source")
    if kernel not in _MODE_ID:
        raise ValueError(f"unsupported fused interpolation kernel {kernel!r}")
    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError("coordinate arrays must have identical shapes")
    from ..memory import bytes_per_voxel_interp_triton, get_available_memory

    out_shape = x.shape
    x, y, z = (v.reshape(-1).contiguous() for v in (x, y, z))
    out = torch.empty(x.numel(), dtype=source.dtype, device=source.device)
    h = {"cubic": 2, "quintic": 3, "heptic": 4, "wsinc5": 5}[kernel]
    block = 128
    if kernel in ("cubic", "quintic"):
        _resample_kernel[(triton.cdiv(x.numel(), block),)](
            source.contiguous(),
            x,
            y,
            z,
            out,
            x.numel(),
            source.shape[2],
            source.shape[1],
            source.shape[0],
            MODE=_MODE_ID[kernel],
            H=h,
            BLOCK=block,
        )
    else:
        ntaps = 2 * h
        avail = get_available_memory(source.device, empty_cache=False)
        per_point = bytes_per_voxel_interp_triton(ntaps)
        chunk = max(1, min(x.numel(), int(avail // per_point)))
        for start in range(0, x.numel(), chunk):
            end = min(start + chunk, x.numel())
            xc, yc, zc = x[start:end], y[start:end], z[start:end]
            planes = torch.empty((ntaps, end - start), dtype=source.dtype, device=source.device)
            _resample_xy_kernel[(triton.cdiv(end - start, block), ntaps)](
                source.contiguous(),
                xc,
                yc,
                zc,
                planes,
                end - start,
                source.shape[2],
                source.shape[1],
                source.shape[0],
                MODE=_MODE_ID[kernel],
                H=h,
                BLOCK=block,
            )
            _resample_z_kernel[(triton.cdiv(end - start, block),)](
                planes,
                zc,
                out[start:end],
                end - start,
                MODE=_MODE_ID[kernel],
                H=h,
                BLOCK=block,
            )
    return out.reshape(out_shape)
