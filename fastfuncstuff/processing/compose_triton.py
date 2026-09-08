"""Fused CUDA kernel for qwarp's compose-and-sample step.

:func:`fastfuncstuff.processing.interp.batched_compose_and_interpolate` is the
hot path of a qwarp level: compose the patch displacement with the running global
warp, then sample the source at the composed position. Profiled on a 0.7 mm pair
it was 35% of the whole run -- of which the two ``grid_sample`` calls it exists
to make were 6.7%.

The rest is coordinate bookkeeping. ``grid_sample`` wants normalised coordinates
in an interleaved ``(..., 3)`` grid, so the portable path spends roughly forty
full ``(B, V)`` passes turning voxel positions into that layout and back: six to
clamp the query point, nine to normalise it, a three-wide stack to interleave it,
three to compose, six for the source position, then the normalise-and-stack again
for the second sample.

None of it is needed by anything but ``grid_sample``'s calling convention. This
kernel walks the same trilinear gathers directly in voxel space and writes only
the four ``(B, V)`` tensors the caller actually consumes. Same load count, about
a seventh of the traffic, and eight launches become one -- which matters twice
over now that qwarp is drifting off being purely compute-bound.

Semantics are pinned to the path it replaces: ``align_corners=True`` with
``padding_mode="border"``, which under that normalisation is exactly voxel-space
trilinear interpolation with the sample position clamped to the volume.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from fastfuncstuff.triton_key import install_triton_key_cache

install_triton_key_cache()


@triton.jit
def _axis_setup(x, n):
    """Border-clamped voxel coordinate -> (lower index, upper index, fraction).

    The clamp is on the continuous coordinate, matching ``padding_mode="border"``,
    which clips before it floors. Clamping the upper index too is safe rather than
    approximate: it can only bind when the coordinate sits exactly on the last
    voxel, where its weight is zero.
    """
    xc = tl.minimum(tl.maximum(x, 0.0), n - 1.0)
    x0f = tl.floor(xc)
    fx = xc - x0f
    x0 = tl.minimum(tl.maximum(x0f.to(tl.int32), 0), n - 1)
    x1 = tl.minimum(x0 + 1, n - 1)
    return x0, x1, fx


@triton.jit
def _trilinear(ptr, x0, x1, fx, y0, y1, fy, z0, z1, fz, nx, plane, mask):
    """One trilinear sample from a volume based at ``ptr``, laid out (nz, ny, nx)."""
    z0p, z1p = z0 * plane, z1 * plane
    y0n, y1n = y0 * nx, y1 * nx
    c000 = tl.load(ptr + z0p + y0n + x0, mask=mask, other=0.0)
    c001 = tl.load(ptr + z0p + y0n + x1, mask=mask, other=0.0)
    c010 = tl.load(ptr + z0p + y1n + x0, mask=mask, other=0.0)
    c011 = tl.load(ptr + z0p + y1n + x1, mask=mask, other=0.0)
    c100 = tl.load(ptr + z1p + y0n + x0, mask=mask, other=0.0)
    c101 = tl.load(ptr + z1p + y0n + x1, mask=mask, other=0.0)
    c110 = tl.load(ptr + z1p + y1n + x0, mask=mask, other=0.0)
    c111 = tl.load(ptr + z1p + y1n + x1, mask=mask, other=0.0)
    c00 = c000 + (c001 - c000) * fx
    c01 = c010 + (c011 - c010) * fx
    c10 = c100 + (c101 - c100) * fx
    c11 = c110 + (c111 - c110) * fx
    c0 = c00 + (c01 - c00) * fy
    c1 = c10 + (c11 - c10) * fy
    return c0 + (c1 - c0) * fz


@triton.jit
def _compose_interp_kernel(
    src_ptr,
    warp_ptr,  # (3, nz, ny, nx)
    pxd_ptr,
    pyd_ptr,
    pzd_ptr,
    bi_ptr,
    bj_ptr,
    bk_ptr,
    out_w_ptr,
    out_ax_ptr,
    out_ay_ptr,
    out_az_ptr,
    n_elem,
    nx,
    ny,
    nz,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem

    pxd = tl.load(pxd_ptr + offs, mask=mask, other=0.0)
    pyd = tl.load(pyd_ptr + offs, mask=mask, other=0.0)
    pzd = tl.load(pzd_ptr + offs, mask=mask, other=0.0)
    bi = tl.load(bi_ptr + offs, mask=mask, other=0.0)
    bj = tl.load(bj_ptr + offs, mask=mask, other=0.0)
    bk = tl.load(bk_ptr + offs, mask=mask, other=0.0)

    plane = nx * ny
    vol = nz * plane

    # Where this voxel lands after the patch's own displacement.
    x0, x1, fx = _axis_setup(bi + pxd, nx)
    y0, y1, fy = _axis_setup(bj + pyd, ny)
    z0, z1, fz = _axis_setup(bk + pzd, nz)

    # The running global warp there, all three channels off one set of corners.
    axd = _trilinear(warp_ptr, x0, x1, fx, y0, y1, fy, z0, z1, fz, nx, plane, mask)
    ayd = _trilinear(warp_ptr + vol, x0, x1, fx, y0, y1, fy, z0, z1, fz, nx, plane, mask)
    azd = _trilinear(warp_ptr + 2 * vol, x0, x1, fx, y0, y1, fy, z0, z1, fz, nx, plane, mask)

    ah_x = pxd + axd
    ah_y = pyd + ayd
    ah_z = pzd + azd

    # The source at the composed position. The portable path clamps to
    # [-0.499, n - 0.501] before handing this to grid_sample, which then clamps
    # again to the volume; the border clamp inside _axis_setup subsumes both.
    sx0, sx1, sfx = _axis_setup(ah_x + bi, nx)
    sy0, sy1, sfy = _axis_setup(ah_y + bj, ny)
    sz0, sz1, sfz = _axis_setup(ah_z + bk, nz)
    warped = _trilinear(src_ptr, sx0, sx1, sfx, sy0, sy1, sfy, sz0, sz1, sfz, nx, plane, mask)

    tl.store(out_w_ptr + offs, warped, mask=mask)
    tl.store(out_ax_ptr + offs, ah_x, mask=mask)
    tl.store(out_ay_ptr + offs, ah_y, mask=mask)
    tl.store(out_az_ptr + offs, ah_z, mask=mask)


def compose_and_interpolate_triton(
    source: Tensor,
    warp_3ch: Tensor,
    patch_xd: Tensor,
    patch_yd: Tensor,
    patch_zd: Tensor,
    base_i: Tensor,
    base_j: Tensor,
    base_k: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """``(warped_vals, ah_xd, ah_yd, ah_zd)``, each (B, V).

    ``source`` is (nz, ny, nx), ``warp_3ch`` is (3, nz, ny, nx), and the patch
    displacements and base coordinates are all (B, V).
    """
    nz, ny, nx = source.shape
    shape = patch_xd.shape
    n_elem = patch_xd.numel()

    pxd = patch_xd.contiguous()
    pyd = patch_yd.contiguous()
    pzd = patch_zd.contiguous()
    bi = base_i.expand(shape).contiguous()
    bj = base_j.expand(shape).contiguous()
    bk = base_k.expand(shape).contiguous()

    warped = torch.empty(shape, device=source.device, dtype=torch.float32)
    ah_x = torch.empty(shape, device=source.device, dtype=torch.float32)
    ah_y = torch.empty(shape, device=source.device, dtype=torch.float32)
    ah_z = torch.empty(shape, device=source.device, dtype=torch.float32)

    block = 256
    _compose_interp_kernel[(triton.cdiv(n_elem, block),)](
        source.contiguous(),
        warp_3ch.contiguous(),
        pxd,
        pyd,
        pzd,
        bi,
        bj,
        bk,
        warped,
        ah_x,
        ah_y,
        ah_z,
        n_elem,
        nx,
        ny,
        nz,
        BLOCK=block,
    )
    return warped, ah_x, ah_y, ah_z
