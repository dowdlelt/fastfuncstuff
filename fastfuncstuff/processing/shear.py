"""Shear-based rigid resampling — AFNI's THD_rota_vol method on the GPU.

A rigid rotation+shift is applied as an optional 180-degree flip plus four
successive 1D fractional shears (Cox's xzyx decomposition), instead of one
scattered 3D interpolation per voxel. For heptic that is 4x8 = 32 taps/voxel
along contiguous rows versus 8^3 = 512 scattered taps — far less work and far
better memory locality. This is exactly what 3dvolreg does, so it is both
faster and *more* faithful to the reference than a general affine resample.

The closed-form ``shear_xzyx`` factorization and the ``shear_best`` 6-way
axis-permutation search are ported from AFNI's ``thd_shear3d.c``, vectorized so
they factor a whole batch of per-volume matrices at once. The 180-degree flip
branch (large rotations only) is not implemented; the returned ``valid`` mask
is False there and the caller must fall back to a general resample.

See ``../afni/src/fastreg/fr_gpu_rota.cu`` and ``thd_shear3d.c``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .interp import _cubic_kernel, _heptic_kernel, _quintic_kernel

_BIG_NORM = 1.0e38

# floor-convention kernels shared with interp._separable_resample_3d.
_FLOOR_KERNELS = {
    "cubic": (_cubic_kernel, 2),
    "quintic": (_quintic_kernel, 3),
    "heptic": (_heptic_kernel, 4),
}


# ---------------------------------------------------------------------------
# xzyx shear factorization (ported from thd_shear3d.c:shear_xzyx, vectorized)
# ---------------------------------------------------------------------------


def _shear_xzyx(q: Tensor, d: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Factor q (B,3,3) + shift d (B,3) into 4 shears Q = Sx2 Sz Sy Sx1.

    Returns (ax, scl, sft, valid):
      ax  : (B,4) long, shear axis per step — fixed [0,1,2,0] (x,y,z,x).
      scl : (B,4,3) shear matrix rows (diagonal entry is the f stretch).
      sft : (B,4) per-shear shift.
      valid: (B,) bool, False where a denominator vanished.
    The struct layout matches AFNI's MCW_3shear.
    """
    q11, q12, q13 = q[:, 0, 0], q[:, 0, 1], q[:, 0, 2]
    q21, q22, q23 = q[:, 1, 0], q[:, 1, 1], q[:, 1, 2]
    q31, q32, q33 = q[:, 2, 0], q[:, 2, 1], q[:, 2, 2]
    xdel, ydel, zdel = d[:, 0], d[:, 1], d[:, 2]

    valid = torch.ones_like(q11, dtype=torch.bool)

    ay = q21
    dy = ydel
    t1 = q21 * q12
    t3 = q13 * q22
    t4 = t3 * q31
    t5 = q21 * q13
    t6 = t5 * q32
    t7 = q23 * q11
    t8 = t7 * q32
    t9 = q12 * q23
    t10 = t9 * q31
    t11 = q22 * q11
    t12 = t11 * q33
    t13 = t1 * q33 + t4 - t6 + t8 - t10 - t12
    t15 = q32 * q32
    t16 = t15 * q32
    t17 = q21 * q21
    t18 = t17 * q21
    t19 = t16 * t18
    t20 = q22 * q22
    t22 = q31 * q31
    t23 = t22 * q32
    t24 = q21 * t20 * t23
    t25 = t20 * q22
    t26 = t22 * q31
    t27 = t25 * t26
    t28 = t15 * t17
    t29 = q22 * q31
    t30 = t28 * t29
    t32 = t13 * t13
    t34 = (-t19 - 3.0 * t24 + t27 + 3.0 * t30) * t32
    t34 = torch.sign(t34) * t34.abs().pow(1.0 / 3.0)
    valid = valid & (t13 != 0.0)
    t35 = 1 / t13 * t34
    t36 = t35 + q31
    t37 = t36 * q21
    t38 = q12 * q33
    t44 = t36 * q23
    t45 = q11 * q32
    t47 = t36 * q12
    t50 = t36 * q22
    t51 = q11 * q33
    t53 = q32 * t17
    t54 = t53 * q12
    t55 = q32 * q21
    t57 = q32 * q31
    t61 = q32 * q23 * q11 * q31
    t62 = q31 * q21
    t64 = q22 * q12
    t66 = t22 * q23
    t67 = t66 * q12
    t68 = t22 * q13
    t69 = t68 * q22
    t70 = t29 * t51
    t73 = (
        -t37 * t38
        - t36 * q13 * t29
        + t37 * q13 * q32
        - t44 * t45
        + t47 * q23 * q31
        + t50 * t51
        + t54
        - t55 * t11
        - t57 * t5
        + t61
        + t62 * t38
        - t62 * t64
        - t67
        + t69
        - t70
        + q31 * t20 * q11
    )
    t75 = t20 * t22
    t77 = t28 - 2.0 * t29 * t55 + t75
    valid = valid & (t77 != 0.0)
    t77 = 1 / t77
    cx2 = t73 * t77
    t78 = t44 * q31
    t79 = t62 * q22
    t80 = t62 * q33
    t81 = t37 * q33
    t84 = t34 * t34
    valid = valid & (t84 != 0.0)
    t85 = 1 / t84
    cy = (-t78 + t79 - t80 + t81 - t53 + t66) * t32 * t85
    t86 = q21 * t22
    t87 = t64 * t36
    t89 = t17 * q12
    t90 = t89 * t36
    t92 = t51 * t22
    t94 = t68 * q32
    t96 = t36 * t36
    t102 = t51 * q31
    t107 = t11 * t36
    t109 = t38 * t22
    t113 = (
        t86 * t87
        - t57 * t90
        + 2.0 * t50 * t92
        + 2.0 * t37 * t94
        + t96 * t22 * t3
        + t96 * q31 * t8
        - 3.0 * t24
        - t96 * q22 * t102
        + 3.0 * t30
        + t27
        - 2.0 * t36 * t26 * t3
        - t19
        - t62 * q32 * t107
        - 2.0 * t37 * t109
        - t26 * q21 * t64
    )
    t118 = q32 * q22
    t119 = t118 * q11
    t121 = q11 * t36
    t123 = t26 * q13
    t125 = t26 * q23
    t127 = q33 * t26
    t129 = t96 * q12
    t131 = t96 * q21
    t132 = t38 * q31
    t134 = t22 * t22
    t141 = q31 * q13 * q32
    t145 = (
        -q11 * t15 * t17 * q31
        + t23 * t89
        + t86 * t119
        + t121 * t28
        - t123 * t55
        + t125 * t45
        + t1 * t127
        - t129 * t66
        + t131 * t132
        + t134 * q13 * q22
        - t9 * t134
        - 2.0 * t36 * t22 * t8
        - t131 * t141
        - t11 * t127
        + 2.0 * t47 * t125
    )
    valid = valid & (t34 != 0.0)
    t148 = 1 / t34
    valid = valid & (q21 != 0.0)
    t150 = 1 / q21
    t151 = t148 * t77 * t150
    bx2 = (t113 + t145) * t13 * t151
    az = -t35
    f = (-t29 + t55) * t13 * t148
    t157 = ydel * q12
    t160 = zdel * t17
    t163 = ydel * t22
    t164 = t163 * q21
    t167 = ydel * t26
    t185 = xdel * q21
    t190 = ydel * q11
    t193 = (
        -ydel * q22 * t51 * t26
        - t157 * q23 * t134
        - t160 * t129 * q33
        + t164 * t119
        + t163 * t54
        - t167 * q21 * q22 * q12
        + ydel * t134 * t3
        + t157 * q21 * q33 * t26
        - t167 * t6
        - 3.0 * ydel * q21 * t75 * q32
        + t167 * t8
        + 3.0 * ydel * t15 * t17 * q22 * q31
        + t185 * t20 * t26
        - ydel * t16 * t18
        - t190 * t28 * q31
    )
    t194 = zdel * q21
    t195 = t125 * q12
    t203 = xdel * t18
    t206 = xdel * t17
    t207 = q22 * t22
    t210 = t160 * q32
    t220 = zdel * t18
    t221 = q32 * q12
    t224 = t123 * q22
    t230 = t194 * t96
    t233 = (
        t194 * t195
        + t194 * t22 * t12
        + t160 * t94
        - t194 * q32 * t7 * t22
        + t203 * q31 * t15
        - 2.0 * t206 * t207 * q32
        - t210 * t107
        - t194 * t75 * q11
        + t194 * q31 * t20 * q11 * t36
        + t210 * t11 * q31
        - t220 * t221 * q31
        - t194 * t224
        + t160 * t207 * q12
        + ydel * t25 * t26
        - t230 * t8
        - t160 * t109
    )
    t238 = t194 * t36
    t240 = ydel * t96
    t241 = t240 * q21
    t252 = ydel * t36
    t264 = (
        -t203 * t36 * t15
        + t230 * t12
        + 2.0 * t238 * t61
        - t241 * t141
        - t240 * q22 * t102
        + t220 * t221 * t36
        - t160 * q31 * t87
        + 2.0 * t206 * t36 * t29 * q32
        - 2.0 * t252 * t22 * t8
        + t164 * t87
        + t190 * t36 * t17 * t15
        - t240 * t67
        - 2.0 * t252 * t224
        + 2.0 * t252 * q22 * t92
        + t241 * t132
    )
    t267 = t160 * t36
    t269 = ydel * q31
    t275 = t252 * q21
    t292 = (
        -2.0 * t238 * t67
        + t240 * t69
        + 2.0 * t267 * t132
        - t269 * q32 * t90
        - t185 * t36 * t20 * t22
        + 2.0 * t275 * t94
        + t230 * t10
        + 2.0 * t238 * t69
        + 2.0 * t252 * t195
        - t230 * t4
        - 2.0 * t275 * t109
        - 2.0 * t267 * t141
        - 2.0 * t238 * t70
        + t240 * q31 * t8
        - t269 * q21 * t118 * t121
        + t160 * t96 * q13 * q32
    )
    dx = -(t193 + t233 + t264 + t292) * t13 * t151
    bz = t36 * t150
    cx1 = -(t78 + t79 - t80 - t96 * q23 + t81 - t53) * t150 * t32 * t85
    dz = (-t252 + t194) * t150
    bx1 = -(-t50 + t55) * t150 * t13 * t148

    B = q11.shape[0]
    one = torch.ones_like(q11)
    zero = torch.zeros_like(q11)
    ax = torch.tensor([0, 1, 2, 0], device=q.device).expand(B, 4).clone()
    scl = torch.stack(
        [
            torch.stack([one, bx1, cx1], dim=-1),  # x1: [1, bx1, cx1]
            torch.stack([ay, f, cy], dim=-1),  # y : [ay, f, cy]
            torch.stack([az, bz, f], dim=-1),  # z : [az, bz, f]
            torch.stack([f, bx2, cx2], dim=-1),  # x2: [f, bx2, cx2]
        ],
        dim=1,
    )  # (B,4,3)
    sft = torch.stack([zero, dy, dz, dx], dim=-1)  # (B,4)

    valid = valid & torch.isfinite(scl).all(dim=(1, 2)) & torch.isfinite(sft).all(dim=1)
    return ax, scl, sft, valid


def _norm_3shear(ax: Tensor, scl: Tensor, valid: Tensor) -> Tensor:
    """Max shear (stretch) magnitude over the first 3 shears; BIG if invalid."""
    B = ax.shape[0]
    top = torch.zeros(B, device=ax.device, dtype=scl.dtype)
    for ii in range(3):
        jj = ax[:, ii]
        o1 = (jj + 1) % 3
        o2 = (jj + 2) % 3
        v1 = scl[torch.arange(B), ii, o1].abs()
        v2 = scl[torch.arange(B), ii, o2].abs()
        top = torch.maximum(top, torch.maximum(v1, v2))
    return torch.where(valid, top, torch.full_like(top, _BIG_NORM))


def _permute_back(ax: Tensor, scl: Tensor, sft: Tensor, pi: list[int]):
    """permute_3shear: ax->pi[ax]; scl[:, ii, pi[j]] = scl_in[:, ii, j]."""
    aout = torch.tensor(pi, device=ax.device)[ax]  # (B,4)
    scl_out = torch.empty_like(scl)
    for j in range(3):
        scl_out[:, :, pi[j]] = scl[:, :, j]
    return aout, scl_out, sft


_PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


def _shear_best(q: Tensor, d: Tensor):
    """6-way axis-permutation search; returns the min-shear-norm plan per lane.

    Returns (ax (B,4), scl (B,4,3), sft (B,4), valid (B,)).
    """
    B = q.shape[0]
    device = q.device
    plans = []
    norms = []
    for pi in _PERMS:
        pi = list(pi)
        # permute_dmat33 / permute_dfvec3: qq[i,j]=q[pi[i],pi[j]], xx[i]=d[pi[i]]
        idx = torch.tensor(pi, device=device)
        qq = q[:, idx][:, :, idx]
        xx = d[:, idx]
        ax, scl, sft, valid = _shear_xzyx(qq, xx)
        ax, scl, sft = _permute_back(ax, scl, sft, pi)
        plans.append((ax, scl, sft, valid))
        norms.append(_norm_3shear(ax, scl, valid))

    norms = torch.stack(norms, dim=1)  # (B,6)
    best = norms.argmin(dim=1)
    any_valid = norms.min(dim=1).values < _BIG_NORM
    ar = torch.arange(B, device=device)
    ax = torch.stack([p[0] for p in plans], dim=1)[ar, best]
    scl = torch.stack([p[1] for p in plans], dim=1)[ar, best]
    sft = torch.stack([p[2] for p in plans], dim=1)[ar, best]
    return ax, scl, sft, any_valid


def _reconstruct_pull(
    ax: Tensor, scl: Tensor, sft: Tensor, shape: tuple[int, int, int]
) -> tuple[Tensor, Tensor]:
    """Compose the pull transform the 4-shear plan *actually* applies.

    Mirrors ``_apply_one_shear`` exactly: each step is a unit-diagonal shear
    along ``ax`` whose offset is ``a*c_p1 + b*c_p2 + s`` over the two driving
    axes' centred coordinates. Returns the net (B,3,3) linear part and (B,3)
    shift in x,y,z index order, so it can be compared against the requested
    pull matrix to detect a degenerate factorization (see ``rigid_matrix_to_shears``).
    """
    B = ax.shape[0]
    device, dtype = scl.device, scl.dtype
    nz, ny, nx = shape
    ctr = torch.tensor([(nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0], device=device, dtype=dtype)
    # per-step shear coeffs (fr_gpu_unpack): a on driving axis p1, b on p2.
    a_co = torch.where(ax == 0, scl[..., 1], scl[..., 0])  # (B,4)
    b_co = torch.where(ax == 0, scl[..., 2], torch.where(ax == 1, scl[..., 2], scl[..., 1]))
    # driving axes per ax: 0->(y,z), 1->(x,z), 2->(x,y)
    p1 = torch.where(ax == 0, 1, 0)  # (B,4)
    p2 = torch.where(ax == 2, 1, 2)

    ar = torch.arange(B, device=device)
    Lt = torch.eye(3, device=device, dtype=dtype)[None].repeat(B, 1, 1)
    bt = torch.zeros(B, 3, device=device, dtype=dtype)
    for step in range(4):
        a_s, p1_s, p2_s = ax[:, step], p1[:, step], p2[:, step]
        ac, bc, s_s = a_co[:, step], b_co[:, step], sft[:, step]
        Li = torch.eye(3, device=device, dtype=dtype)[None].repeat(B, 1, 1)
        Li[ar, a_s, p1_s] -= ac
        Li[ar, a_s, p2_s] -= bc
        bi = torch.zeros(B, 3, device=device, dtype=dtype)
        bi[ar, a_s] = ac * ctr[p1_s] + bc * ctr[p2_s] - s_s
        bt = (Lt @ bi.unsqueeze(-1)).squeeze(-1) + bt
        Lt = Lt @ Li
    return Lt, bt


def rigid_matrix_to_shears(matrix: Tensor, shape: tuple[int, int, int]):
    """Decompose a voxel-index pull matrix into a 4-shear plan (shear_best).

    ``matrix`` (4,4) or (B,4,4) maps output index (i,j,k) to source sample
    location: ``out(p) = source(A p + t)``. Returns (ax, scl, sft, valid),
    each batched over B, with axes x=nx, y=ny, z=nz.
    """
    if matrix.dim() == 2:
        matrix = matrix[None]
    B = matrix.shape[0]
    out_dtype = matrix.dtype
    # The closed-form xzyx factorization (cube roots + chained divisions) loses
    # ~7 digits, so in float32 a near-single-axis rotation (pitch with tiny
    # roll/yaw, as real GN fits of pitch-dominated motion produce) decomposes
    # into a catastrophically wrong plan — a spurious rotation 10-100% off the
    # request. AFNI's thd_shear3d.c is robust only because it is entirely double;
    # the math is identical. So we decompose in float64 (cheap: per-volume 3x3)
    # and cast the plan back for the float32 shear apply. The reconstruction
    # guard below still catches the genuinely degenerate exact-single-axis case.
    dtype = torch.float64
    matrix = matrix.to(dtype)
    device = matrix.device
    A = matrix[:, :3, :3]
    t = matrix[:, :3, 3]
    nz, ny, nx = shape
    c = torch.tensor([(nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0], device=device, dtype=dtype)

    # Re-express the pull about the centre, then map to the centred forward
    # map the shear apply realises: q = A^T, d = -A^T (A c + t - c).
    d_c = (A @ c) + t - c
    AT = A.transpose(-1, -2)
    q = AT
    d = -(AT @ d_c.unsqueeze(-1)).squeeze(-1)

    # near-identity special case (shear_best): trivial shears, shifts only.
    trace = q[:, 0, 0] + q[:, 1, 1] + q[:, 2, 2]
    esum = (
        q[:, 0, 1].abs()
        + q[:, 0, 2].abs()
        + q[:, 1, 0].abs()
        + q[:, 1, 2].abs()
        + q[:, 2, 0].abs()
        + q[:, 2, 1].abs()
    )
    near_id = (trace >= 2.99999) & (esum / trace.clamp(min=1e-12) < 1.0e-6)

    ax, scl, sft, any_valid = _shear_best(q, d)

    # overwrite near-identity lanes with a trivial plan (shifts d only)
    if bool(near_id.any()):
        triv_ax = torch.tensor([0, 1, 2, 0], device=device).expand(B, 4)
        triv_scl = torch.zeros(B, 4, 3, device=device, dtype=dtype)
        triv_scl[:, 0, 0] = 1.0
        triv_scl[:, 1, 1] = 1.0
        triv_scl[:, 2, 2] = 1.0
        triv_scl[:, 3, 0] = 1.0
        triv_sft = torch.stack([torch.zeros_like(d[:, 0]), d[:, 1], d[:, 2], d[:, 0]], dim=-1)
        m = near_id
        ax = torch.where(m[:, None], triv_ax, ax)
        scl = torch.where(m[:, None, None], triv_scl, scl)
        sft = torch.where(m[:, None], triv_sft, sft)
        any_valid = any_valid | near_id

    # Safety net: the plan must reproduce the requested pull. In float64 this
    # rejects only the genuinely degenerate exact-single-axis rotation (where the
    # factorization is NaN regardless of precision); those lanes fall back to a
    # general resample. The linear-part error is scale-free; honest motion is
    # ~1e-11 here, so the 1e-2 cutoff has enormous margin.
    Lt, _bt = _reconstruct_pull(ax, scl, sft, shape)
    recon_err = (Lt - A).abs().amax(dim=(1, 2))
    any_valid = any_valid & (recon_err <= 1.0e-2)

    return ax.to(torch.int64), scl.to(out_dtype), sft.to(out_dtype), any_valid


# ---------------------------------------------------------------------------
# 1D fractional shear application (matches fr_gpu_rota.cu:fr_shear_kernel)
# ---------------------------------------------------------------------------


def _interp_1d_along(vol: Tensor, dim: int, af: Tensor, mode: str) -> Tensor:
    """Shift each row of ``vol`` along ``dim``: out[..,p,..] = vol[..,p-af,..].

    AFNI pull convention: ia = floor(-af), aa = -af - ia in [0,1), zero-fill
    outside [0,n). ``af`` has size 1 along ``dim`` (constant within a row) and
    broadcasts over the other two axes. Gathers directly along ``dim`` — no
    transpose/copy — so the shear stays cheap.
    """
    n = vol.shape[dim]
    device = vol.device
    nshift = -af  # shape S, with S[dim] == 1
    ia = torch.floor(nshift)
    aa = nshift - ia  # in [0,1), shape S
    ia_l = ia.to(torch.long)
    aa_flat = aa.reshape(-1)  # (K,) over the two non-dim axes

    if mode == "linear":
        offs = (0, 1)
        wlist = [1.0 - aa, aa]
    elif mode == "nn":
        offs = (0,)
        wlist = [torch.ones_like(aa)]
    elif mode == "wsinc5":
        offs = tuple(range(-4, 6))  # AFNI floor wsinc5: 10 taps, raw
        wlist = []
        for k in offs:
            dist = (aa - k).abs()
            pid = torch.pi * dist
            s = torch.where(dist > 0.01, torch.sin(pid) / pid, 1.0 - 1.6449341 * dist * dist)
            mx = 0.19999 * dist
            m3 = (
                0.4243801
                + 0.4973406 * torch.cos(torch.pi * mx)
                + 0.0782793 * torch.cos(torch.pi * mx * 2.0)
            )
            wlist.append(s * m3)
    else:
        kfn, H = _FLOOR_KERNELS[mode]
        offs = tuple(range(-(H - 1), H + 1))
        w = kfn(aa_flat)  # (K, ntaps)
        wlist = [w[:, j].reshape(aa.shape) for j in range(w.shape[1])]

    p_shape = [1, 1, 1]
    p_shape[dim] = n
    p = torch.arange(n, device=device).reshape(p_shape)

    acc = torch.zeros_like(vol)
    for w, off in zip(wlist, offs, strict=True):
        idx = (p + ia_l + off).expand(vol.shape)  # full-shape gather index
        inb = (idx >= 0) & (idx < n)
        g = torch.gather(vol, dim, idx.clamp(0, n - 1))
        acc = acc + w * torch.where(inb, g, torch.zeros((), device=device, dtype=vol.dtype))

    return acc


# fr-axis (0=x,1=y,2=z) -> tensor dim (vol is (nz,ny,nx)), plus the two driving
# axes (p1,p2) with their fr-axis id and tensor dim.
_AXIS_INFO = {
    0: (2, (1, 1), (2, 0)),  # x: tdim2; p1=y(fr1,dim1), p2=z(fr2,dim0)
    1: (1, (0, 2), (2, 0)),  # y: tdim1; p1=x(fr0,dim2), p2=z(fr2,dim0)
    2: (0, (0, 2), (1, 1)),  # z: tdim0; p1=x(fr0,dim2), p2=y(fr1,dim1)
}


def _apply_one_shear(vol: Tensor, fr_axis: int, a: float, b: float, s: float, mode: str) -> Tensor:
    nz, ny, nx = vol.shape
    sizes = {0: nx, 1: ny, 2: nz}
    centers = {0: (nx - 1) / 2.0, 1: (ny - 1) / 2.0, 2: (nz - 1) / 2.0}
    tdim, (p1fr, p1dim), (p2fr, p2dim) = _AXIS_INFO[fr_axis]

    st = abs(a) * centers[p1fr] + abs(b) * centers[p2fr] + abs(s)
    if st < 1.0e-3:
        return vol  # identity shear

    device, dtype = vol.device, vol.dtype
    coord1 = torch.arange(sizes[p1fr], device=device, dtype=dtype) - centers[p1fr]
    coord2 = torch.arange(sizes[p2fr], device=device, dtype=dtype) - centers[p2fr]
    shp1 = [1, 1, 1]
    shp1[p1dim] = sizes[p1fr]
    shp2 = [1, 1, 1]
    shp2[p2dim] = sizes[p2fr]
    af = a * coord1.reshape(shp1) + b * coord2.reshape(shp2) + s
    return _interp_1d_along(vol, tdim, af, mode)


# unpack MCW_3shear (ax, scl) -> per-shear (a, b) (matches fr_gpu_unpack)
def _unpack(ax: int, scl3) -> tuple[float, float]:
    if ax == 0:  # x: a=scl[1], b=scl[2]
        return scl3[1], scl3[2]
    elif ax == 1:  # y: a=scl[0], b=scl[2]
        return scl3[0], scl3[2]
    else:  # z: a=scl[0], b=scl[1]
        return scl3[0], scl3[1]


def shear_resample(
    source: Tensor, matrix: Tensor, shape: tuple[int, int, int], mode: str
) -> Tensor | None:
    """Resample ``source`` by a rigid pull ``matrix`` via 4 shears.

    Drop-in for ``resample_affine_fast`` on rigid transforms. Returns the
    resampled volume, or ``None`` if the decomposition is invalid (caller falls
    back to a general resample).
    """
    ax_t, scl_t, sft_t, valid = rigid_matrix_to_shears(matrix, shape)
    if not bool(valid.all()):
        return None
    ax_l = ax_t[0].tolist()
    scl_l = scl_t[0].tolist()
    sft_l = sft_t[0].tolist()

    vol = source
    for step in range(4):
        ax = ax_l[step]
        a, b = _unpack(ax, scl_l[step])
        vol = _apply_one_shear(vol, ax, a, b, sft_l[step], mode)
    return vol
