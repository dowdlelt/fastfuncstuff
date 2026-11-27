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


# ---------------------------------------------------------------------------
# Gradient: d(penalty)/d(displacement), for the hybrid polish's analytic step.
#
# Two passes, both deterministic (no atomics, so qwarp stays bit-reproducible):
#   A. per hexahedron, the energy's derivative with respect to its nine
#      deformation-gradient entries F (stored as nine fields), and the per-patch
#      energy sums, in one sweep;
#   B. per voxel, gather from every hexahedron that has it as a corner. A corner
#      offset d along an axis comes from the hexahedron at u - d -- plus, on the
#      last plane, the hexahedron at u itself, whose clamped +1 corner is u. Per
#      axis that is (offset, corner) in {(0,0), (1,1) if u >= 1, (0,1) if u is
#      last}: 27 combinations, statically unrolled.
# ---------------------------------------------------------------------------


@triton.jit
def _penalty_dF_kernel(
    xd_ptr,
    yd_ptr,
    zd_ptr,
    g_ptr,  # (9, B, V): dE/dF per hexahedron, F row-major xx xy xz yx yy yz zx zy zz
    out_ptr,  # (B, NSPLIT) energy partial sums
    V,
    BV,
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

        # cofactors = d det / d F
        cxx = fyy * fzz - fyz * fzy
        cxy = -(fyx * fzz - fyz * fzx)
        cxz = fyx * fzy - fyy * fzx
        cyx = -(fxy * fzz - fxz * fzy)
        cyy = fxx * fzz - fxz * fzx
        cyz = -(fxx * fzy - fxy * fzx)
        czx = fxy * fyz - fxz * fyy
        czy = -(fxx * fyz - fxz * fyx)
        czz = fxx * fyy - fxy * fyx
        det_raw = fxx * cxx + fxy * cxy + fxz * cxz
        live = (det_raw > 0.1) & (det_raw < 10.0)  # clamp passes no gradient outside
        det = tl.maximum(0.1, tl.minimum(det_raw, 10.0))
        bulk = det - 1.0 / det
        je = (1.0 / 3.0) * bulk * bulk
        matrix_norm = (
            fxx * fxx + fxy * fxy + fxz * fxz + fyx * fyx + fyy * fyy
            + fyz * fyz + fzx * fzx + fzy * fzy + fzz * fzz
        )  # fmt: skip
        vyz = fyz - fzy
        vxz = fxz - fzx
        vxy = fxy - fyx
        vorticity = 2.0 * (vyz * vyz + vxz * vxz + vxy * vxy)
        d23 = tl.exp((2.0 / 3.0) * tl.log(det))
        s = (matrix_norm + vorticity) / d23 - 3.0
        se = tl.maximum(s, 0.0)
        ej = tl.maximum(je - CUT, 0.0)
        es = tl.maximum(se - CUT, 0.0)
        acc += tl.where(mask, ej * ej * ej * ej + es * es * es * es, 0.0)

        dj = 4.0 * ej * ej * ej  # zero unless je > CUT
        ds = 4.0 * es * es * es  # zero unless se > CUT (> 0)
        # d contribution / d det, through je and through se's det^(2/3)
        ddet = dj * (2.0 / 3.0) * bulk * (1.0 + 1.0 / (det * det)) - ds * (2.0 / 3.0) * (
            matrix_norm + vorticity
        ) / (d23 * det)
        ddet = tl.where(live, ddet, 0.0)
        a = ds / d23  # d contribution / d (matrix_norm + vorticity)
        a = tl.where(mask, a, 0.0)
        ddet = tl.where(mask, ddet, 0.0)

        gb = pid_b * V + offs
        tl.store(g_ptr + 0 * BV + gb, ddet * cxx + a * 2.0 * fxx, mask=mask)
        tl.store(g_ptr + 1 * BV + gb, ddet * cxy + a * (2.0 * fxy + 4.0 * vxy), mask=mask)
        tl.store(g_ptr + 2 * BV + gb, ddet * cxz + a * (2.0 * fxz + 4.0 * vxz), mask=mask)
        tl.store(g_ptr + 3 * BV + gb, ddet * cyx + a * (2.0 * fyx - 4.0 * vxy), mask=mask)
        tl.store(g_ptr + 4 * BV + gb, ddet * cyy + a * 2.0 * fyy, mask=mask)
        tl.store(g_ptr + 5 * BV + gb, ddet * cyz + a * (2.0 * fyz + 4.0 * vyz), mask=mask)
        tl.store(g_ptr + 6 * BV + gb, ddet * czx + a * (2.0 * fzx - 4.0 * vxz), mask=mask)
        tl.store(g_ptr + 7 * BV + gb, ddet * czy + a * (2.0 * fzy - 4.0 * vyz), mask=mask)
        tl.store(g_ptr + 8 * BV + gb, ddet * czz + a * 2.0 * fzz, mask=mask)

    tl.store(out_ptr + pid_b * n_split + pid_s, tl.sum(acc, axis=0))


@triton.jit
def _penalty_gather_kernel(
    g_ptr,  # (9, B, V)
    coef_ptr,  # (B,)
    out_ptr,  # (3, B, V)
    V,
    BV,
    nx,
    ny,
    nz,
    BLOCK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_v = tl.program_id(1)
    offs = pid_v * BLOCK + tl.arange(0, BLOCK)
    mask = offs < V
    plane = nx * ny
    i = offs % nx
    j = (offs // nx) % ny
    k = offs // plane
    coef = tl.load(coef_ptr + pid_b)
    gx = tl.zeros((BLOCK,), dtype=tl.float32)
    gy = tl.zeros((BLOCK,), dtype=tl.float32)
    gz = tl.zeros((BLOCK,), dtype=tl.float32)
    # option per axis: 0 -> (offset 0, corner 0); 1 -> (1, 1); 2 -> (0, 1).
    # Corner weights: d F[row, comp] / d corner of that comp. Row x pairs corners
    # (1,0),(7,6); row y (2,0),(7,5); row z (4,0),(7,3) -- see the F expressions.
    for qk in tl.static_range(3):
        for qj in tl.static_range(3):
            for qi in tl.static_range(3):
                c = (0 if qi == 0 else 1) + 2 * (0 if qj == 0 else 1) + 4 * (0 if qk == 0 else 1)
                wx = 0.5 if (c == 1 or c == 7) else (-0.5 if (c == 0 or c == 6) else 0.0)
                wy = 0.5 if (c == 2 or c == 7) else (-0.5 if (c == 0 or c == 5) else 0.0)
                wz = 0.5 if (c == 4 or c == 7) else (-0.5 if (c == 0 or c == 3) else 0.0)
                ok = mask
                if qi == 1:
                    ok = ok & (i >= 1)
                if qi == 2:
                    ok = ok & (i == nx - 1)
                if qj == 1:
                    ok = ok & (j >= 1)
                if qj == 2:
                    ok = ok & (j == ny - 1)
                if qk == 1:
                    ok = ok & (k >= 1)
                if qk == 2:
                    ok = ok & (k == nz - 1)
                h = (
                    pid_b * V
                    + (k - (1 if qk == 1 else 0)) * plane
                    + (j - (1 if qj == 1 else 0)) * nx
                    + (i - (1 if qi == 1 else 0))
                )
                if wx != 0.0:
                    gx += wx * tl.load(g_ptr + 0 * BV + h, mask=ok, other=0.0)
                    gy += wx * tl.load(g_ptr + 1 * BV + h, mask=ok, other=0.0)
                    gz += wx * tl.load(g_ptr + 2 * BV + h, mask=ok, other=0.0)
                if wy != 0.0:
                    gx += wy * tl.load(g_ptr + 3 * BV + h, mask=ok, other=0.0)
                    gy += wy * tl.load(g_ptr + 4 * BV + h, mask=ok, other=0.0)
                    gz += wy * tl.load(g_ptr + 5 * BV + h, mask=ok, other=0.0)
                if wz != 0.0:
                    gx += wz * tl.load(g_ptr + 6 * BV + h, mask=ok, other=0.0)
                    gy += wz * tl.load(g_ptr + 7 * BV + h, mask=ok, other=0.0)
                    gz += wz * tl.load(g_ptr + 8 * BV + h, mask=ok, other=0.0)
    o = pid_b * V + offs
    tl.store(out_ptr + 0 * BV + o, coef * gx, mask=mask)
    tl.store(out_ptr + 1 * BV + o, coef * gy, mask=mask)
    tl.store(out_ptr + 2 * BV + o, coef * gz, mask=mask)


def penalty_value_and_grad_triton(
    xd: Tensor, yd: Tensor, zd: Tensor, cut: float, pen_fac: float, external_sums: Tensor
) -> tuple[Tensor, Tensor]:
    """``pen_fac * (external + sum E)^0.25`` per patch, ``(B,)``, and its gradient with
    respect to the composed displacements, ``(3, B, nz, ny, nx)`` (x, y, z).

    Matches autograd through :func:`fastfuncstuff.processing.penalty.compute_penalty_batched`
    (the clamps pass no gradient outside their range, as torch's do).
    """
    b, nz, ny, nx = xd.shape
    v = nz * ny * nx
    block = 128 if v >= 128 else max(16, triton.next_power_of_2(v))
    n_split = max(1, min(triton.cdiv(v, block), triton.cdiv(_TARGET_PROGRAMS, b)))
    g9 = torch.empty((9, b, v), device=xd.device, dtype=torch.float32)
    part = torch.zeros((b, n_split), device=xd.device, dtype=torch.float32)
    _penalty_dF_kernel[(b, n_split)](
        xd.contiguous(), yd.contiguous(), zd.contiguous(), g9, part,
        v, b * v, nx, ny, nz, n_split, CUT=cut, BLOCK=block,
    )  # fmt: skip
    hsum = (external_sums + part.sum(1)).clamp(min=0)
    value = pen_fac * hsum.pow(0.25)
    # d value / d sum = pen_fac / 4 * hsum^-3/4; zero where nothing bites (as autograd
    # would give inf * 0 there, the patch has no offending voxel to push anyway).
    coef = torch.where(hsum > 0, 0.25 * pen_fac * hsum.clamp_min(1e-30).pow(-0.75), 0.0)
    grad = torch.empty((3, b, v), device=xd.device, dtype=torch.float32)
    _penalty_gather_kernel[(b, triton.cdiv(v, block))](
        g9, coef.contiguous(), grad, v, b * v, nx, ny, nz, BLOCK=block
    )
    return value, grad.view(3, b, nz, ny, nx)
