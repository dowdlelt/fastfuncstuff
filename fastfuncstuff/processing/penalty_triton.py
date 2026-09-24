"""Fused CUDA kernel for the batched qwarp deformation penalty.

The portable path builds eight hexahedron corners per displacement component,
then the bulk/shear expressions, then reduces -- many full
``(B, nz, ny, nx)` passes to produce one number per patch. It measured 15% of a
0.7 mm run, more than the interpolation it guards.

Nothing between the displacement fields and the per-patch sum needs to be
written down: the stencil, the Jacobian determinant, the strain and vorticity
terms, and the deadband all live in registers, and only the sum leaves. As with
the Gauss-Newton kernel, voxels are split into a fixed number of ranges whose
partial sums torch adds, so the result does not depend on atomic ordering and
qwarp stays bit-reproducible run to run.

Reference: :mod:`fastfuncstuff.processing.penalty`, which this must match
exactly -- including AFNI's clamped hexahedron corners and ``HPEN_CUT`` deadband
(``mri_nwarp.c:2164-2284``).
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

        ip = tl.minimum(i + 1, nx - 1)
        jp = tl.minimum(j + 1, ny - 1)
        kp = tl.minimum(k + 1, nz - 1)

        o0 = base + k * plane + j * nx + i
        o1 = base + k * plane + j * nx + ip
        o2 = base + k * plane + jp * nx + i
        o3 = base + k * plane + jp * nx + ip
        o4 = base + kp * plane + j * nx + i
        o5 = base + kp * plane + j * nx + ip
        o6 = base + kp * plane + jp * nx + i
        o7 = base + kp * plane + jp * nx + ip

        x0 = tl.load(xd_ptr + o0, mask=mask, other=0.0)
        x1 = tl.load(xd_ptr + o1, mask=mask, other=0.0)
        x2 = tl.load(xd_ptr + o2, mask=mask, other=0.0)
        x3 = tl.load(xd_ptr + o3, mask=mask, other=0.0)
        x4 = tl.load(xd_ptr + o4, mask=mask, other=0.0)
        x5 = tl.load(xd_ptr + o5, mask=mask, other=0.0)
        x6 = tl.load(xd_ptr + o6, mask=mask, other=0.0)
        x7 = tl.load(xd_ptr + o7, mask=mask, other=0.0)
        y0 = tl.load(yd_ptr + o0, mask=mask, other=0.0)
        y1 = tl.load(yd_ptr + o1, mask=mask, other=0.0)
        y2 = tl.load(yd_ptr + o2, mask=mask, other=0.0)
        y3 = tl.load(yd_ptr + o3, mask=mask, other=0.0)
        y4 = tl.load(yd_ptr + o4, mask=mask, other=0.0)
        y5 = tl.load(yd_ptr + o5, mask=mask, other=0.0)
        y6 = tl.load(yd_ptr + o6, mask=mask, other=0.0)
        y7 = tl.load(yd_ptr + o7, mask=mask, other=0.0)
        z0 = tl.load(zd_ptr + o0, mask=mask, other=0.0)
        z1 = tl.load(zd_ptr + o1, mask=mask, other=0.0)
        z2 = tl.load(zd_ptr + o2, mask=mask, other=0.0)
        z3 = tl.load(zd_ptr + o3, mask=mask, other=0.0)
        z4 = tl.load(zd_ptr + o4, mask=mask, other=0.0)
        z5 = tl.load(zd_ptr + o5, mask=mask, other=0.0)
        z6 = tl.load(zd_ptr + o6, mask=mask, other=0.0)
        z7 = tl.load(zd_ptr + o7, mask=mask, other=0.0)

        fxx = 0.5 * ((x1 - x0) + (x7 - x6)) + 1.0
        fxy = 0.5 * ((y1 - y0) + (y7 - y6))
        fxz = 0.5 * ((z1 - z0) + (z7 - z6))
        fyx = 0.5 * ((x2 - x0) + (x7 - x5))
        fyy = 0.5 * ((y2 - y0) + (y7 - y5)) + 1.0
        fyz = 0.5 * ((z2 - z0) + (z7 - z5))
        fzx = 0.5 * ((x4 - x0) + (x7 - x3))
        fzy = 0.5 * ((y4 - y0) + (y7 - y3))
        fzz = 0.5 * ((z4 - z0) + (z7 - z3)) + 1.0

        det = (
            fxx * (fyy * fzz - fyz * fzy)
            - fxy * (fyx * fzz - fyz * fzx)
            + fxz * (fyx * fzy - fyy * fzx)
        )
        det = tl.maximum(0.1, tl.minimum(det, 10.0))
        bulk = det - 1.0 / det
        je = (1.0 / 3.0) * bulk * bulk

        matrix_norm = (
            fxx * fxx
            + fxy * fxy
            + fxz * fxz
            + fyx * fyx
            + fyy * fyy
            + fyz * fyz
            + fzx * fzx
            + fzy * fzy
            + fzz * fzz
        )
        vorticity = 2.0 * (
            (fyz - fzy) * (fyz - fzy) + (fxz - fzx) * (fxz - fzx) + (fxy - fyx) * (fxy - fyx)
        )
        se = tl.maximum((matrix_norm + vorticity) / tl.exp((2.0 / 3.0) * tl.log(det)) - 3.0, 0.0)

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
