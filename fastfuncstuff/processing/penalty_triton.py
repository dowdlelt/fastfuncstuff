"""Fused CUDA kernel for the batched qwarp deformation penalty.

The portable path builds nine central-difference fields, then some forty
elementwise expressions over them, then reduces -- roughly eighty full
``(B, nz, ny, nx)` passes to produce one number per patch. It measured 15% of a
0.7 mm run, more than the interpolation it guards.

Nothing between the displacement fields and the per-patch sum needs to be
written down: the stencil, the Jacobian determinant, the strain and vorticity
terms, and the deadband all live in registers, and only the sum leaves. As with
the Gauss-Newton kernel, voxels are split into a fixed number of ranges whose
partial sums torch adds, so the result does not depend on atomic ordering and
qwarp stays bit-reproducible run to run.

Reference: :mod:`fastfuncstuff.processing.penalty`, which this must match
exactly -- including AFNI's forward/backward differences at the patch faces and
the ``HPEN_CUT`` deadband (mri_nwarp.c:2241).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from fastfuncstuff.triton_key import install_triton_key_cache

install_triton_key_cache()

_TARGET_PROGRAMS = 1024


@triton.jit
def _axis_diff(ptr, off_p, off_m, sc, mask):
    """One central (or one-sided, at a face) difference along a prepared axis."""
    vp = tl.load(ptr + off_p, mask=mask, other=0.0)
    vm = tl.load(ptr + off_m, mask=mask, other=0.0)
    return sc * (vp - vm)


@triton.jit
def _penalty_kernel(
    xd_ptr,
    yd_ptr,
    zd_ptr,
    out_ptr,  # (B, NSPLIT)
    V,
    nx,
    ny,
    nz,
    n_split,
    CUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    per_split = tl.cdiv(tl.cdiv(V, BLOCK), n_split) * BLOCK
    v_start = pid_s * per_split
    v_end = tl.minimum(v_start + per_split, V)

    plane = nx * ny
    base = pid_b * V
    acc = tl.zeros((BLOCK,), dtype=tl.float32)

    for f0 in tl.range(v_start, v_end, BLOCK):
        offs = f0 + tl.arange(0, BLOCK)
        mask = offs < v_end
        i = offs % nx
        j = (offs // nx) % ny
        k = offs // plane

        # AFNI takes a central difference inside and a one-sided difference on
        # each face; both are two loads and a scale, so they are expressed the
        # same way and only the pair of indices and the scale differ.
        ip = tl.where(i == 0, 1, tl.where(i == nx - 1, nx - 1, i + 1))
        im = tl.where(i == 0, 0, tl.where(i == nx - 1, nx - 2, i - 1))
        sx = tl.where((i == 0) | (i == nx - 1), 1.0, 0.5) * (nx > 1)
        jp = tl.where(j == 0, 1, tl.where(j == ny - 1, ny - 1, j + 1))
        jm = tl.where(j == 0, 0, tl.where(j == ny - 1, ny - 2, j - 1))
        sy = tl.where((j == 0) | (j == ny - 1), 1.0, 0.5) * (ny > 1)
        kp = tl.where(k == 0, 1, tl.where(k == nz - 1, nz - 1, k + 1))
        km = tl.where(k == 0, 0, tl.where(k == nz - 1, nz - 2, k - 1))
        sz = tl.where((k == 0) | (k == nz - 1), 1.0, 0.5) * (nz > 1)

        row = base + k * plane + j * nx
        x_p, x_m = row + ip, row + im
        y_p = base + k * plane + jp * nx + i
        y_m = base + k * plane + jm * nx + i
        z_p = base + kp * plane + j * nx + i
        z_m = base + km * plane + j * nx + i

        a11 = 1.0 + _axis_diff(xd_ptr, x_p, x_m, sx, mask)
        a12 = _axis_diff(xd_ptr, y_p, y_m, sy, mask)
        a13 = _axis_diff(xd_ptr, z_p, z_m, sz, mask)
        a21 = _axis_diff(yd_ptr, x_p, x_m, sx, mask)
        a22 = 1.0 + _axis_diff(yd_ptr, y_p, y_m, sy, mask)
        a23 = _axis_diff(yd_ptr, z_p, z_m, sz, mask)
        a31 = _axis_diff(zd_ptr, x_p, x_m, sx, mask)
        a32 = _axis_diff(zd_ptr, y_p, y_m, sy, mask)
        a33 = 1.0 + _axis_diff(zd_ptr, z_p, z_m, sz, mask)

        det = (
            a11 * (a22 * a33 - a23 * a32)
            - a12 * (a21 * a33 - a23 * a31)
            + a13 * (a21 * a32 - a22 * a31)
        )
        je = (det - 1.0) * (det - 1.0)

        e12 = 0.5 * (a12 + a21)
        e13 = 0.5 * (a13 + a31)
        e23 = 0.5 * (a23 + a32)
        w12 = 0.5 * (a12 - a21)
        w13 = 0.5 * (a13 - a31)
        w23 = 0.5 * (a23 - a32)
        e11 = a11 - 1.0
        e22 = a22 - 1.0
        e33 = a33 - 1.0
        se = (
            e12 * e12
            + e13 * e13
            + e23 * e23
            + w12 * w12
            + w13 * w13
            + w23 * w23
            + 0.5 * (e11 * e11 + e22 * e22 + e33 * e33)
        )

        ej = tl.maximum(je - CUT, 0.0)
        es = tl.maximum(se - CUT, 0.0)
        contrib = ej * ej * ej * ej + es * es * es * es
        acc += tl.where(mask, contrib, 0.0)

    tl.store(out_ptr + pid_b * n_split + pid_s, tl.sum(acc, axis=0))


def penalty_sums_triton(xd: Tensor, yd: Tensor, zd: Tensor, cut: float) -> Tensor:
    """Per-patch summed penalty energy, ``(B,)``, without materialising the fields.

    ``xd``, ``yd``, ``zd`` are (B, nz, ny, nx) composed displacements.
    """
    b, nz, ny, nx = xd.shape
    v = nz * ny * nx
    block = 128 if v >= 128 else max(16, triton.next_power_of_2(v))
    n_split = max(1, min(triton.cdiv(v, block), triton.cdiv(_TARGET_PROGRAMS, b)))

    out = torch.zeros((b, n_split), device=xd.device, dtype=torch.float32)
    _penalty_kernel[(b, n_split)](
        xd.contiguous(),
        yd.contiguous(),
        zd.contiguous(),
        out,
        v,
        nx,
        ny,
        nz,
        n_split,
        CUT=cut,
        BLOCK=block,
    )
    return out.sum(1)
