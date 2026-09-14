"""Tests for the blip-up/down (FSL topup-style) distortion estimator.

Synthetic data only: we verify the field solver recovers a known off-resonance
field, that the two blips agree after correction, and a few primitive invariants.
"""

from __future__ import annotations

import math

import torch

from fastfuncstuff.processing import topup as T


def _make_synthetic(nz=20, ny=32, nx=32, readout=0.5, device="cpu"):
    """A smooth 'true' image + known Hz field -> blip-up/down pair (PE = y/j)."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz).float(),
        torch.arange(ny).float(),
        torch.arange(nx).float(),
        indexing="ij",
    )
    true = torch.full((nz, ny, nx), 20.0)
    for cz, cy, cx, r, a in [(10, 12, 16, 6, 100), (8, 22, 20, 4, 70)]:
        true += a * torch.exp(-(((zz - cz) / r) ** 2 + ((yy - cy) / r) ** 2 + ((xx - cx) / r) ** 2))
    field = 7.0 * torch.sin(2 * math.pi * yy / ny) * torch.exp(-(((xx - nx / 2) / 10) ** 2))
    true, field = true.to(device), field.to(device)

    pe_tdim = 1  # y

    def observed(sign):
        disp = field * (readout * sign)
        o = T._resample_pe(true, -disp, pe_tdim)
        return o / T._jacobian_pe(disp, pe_tdim).clamp(min=0.1)

    up = T.ScanSpec(observed(+1.0), pe_axis=1, sign=+1.0, readout=readout)
    down = T.ScanSpec(observed(-1.0), pe_axis=1, sign=-1.0, readout=readout)
    return true, field, [up, down]


def test_field_recovery():
    true, field, scans = _make_synthetic()
    cfg = T.TopupConfig(
        warpres=[16, 10], fwhm=[5, 2], lam=[1e-3, 1e-4], miter=[8, 8], subsamp=[1, 1]
    )
    res = T.run_topup(scans, (3.0, 2.5, 2.5), cfg, progress=False)

    m = torch.zeros_like(field, dtype=torch.bool)
    m[3:-3, 4:-4, 4:-4] = True
    ft, fr = field[m], res.field_hz[m]
    corr = torch.corrcoef(torch.stack([ft, fr]))[0, 1].item()
    assert corr > 0.9, f"field correlation too low: {corr}"


def test_blips_agree_after_correction():
    _, _, scans = _make_synthetic()
    cfg = T.TopupConfig(
        warpres=[16, 10], fwhm=[5, 2], lam=[1e-3, 1e-4], miter=[8, 8], subsamp=[1, 1]
    )
    # caller's tensors must be untouched by run_topup
    before_data = scans[0].data.clone()
    res = T.run_topup(scans, (3.0, 2.5, 2.5), cfg, progress=False)
    assert torch.allclose(scans[0].data, before_data), "run_topup mutated caller scans"

    m = torch.zeros_like(scans[0].data, dtype=torch.bool)
    m[3:-3, 4:-4, 4:-4] = True
    before = ((scans[0].data - scans[1].data)[m]).pow(2).mean().sqrt()
    after = ((res.unwarped[0] - res.unwarped[1])[m]).pow(2).mean().sqrt()
    assert after < 0.4 * before, f"blips not reconciled: before={before} after={after}"


def test_pe_shift_recovery():
    # Clean translation (no field): scan0 = true shifted +1 vox, scan1 = -1 vox along PE.
    true, _, _ = _make_synthetic()
    pe_tdim = 1
    a = T._resample_pe(true, torch.full_like(true, 1.0), pe_tdim)
    b = T._resample_pe(true, torch.full_like(true, -1.0), pe_tdim)
    scans = [T.ScanSpec(a, 1, 1.0, 0.5), T.ScanSpec(b, 1, -1.0, 0.5)]

    m = torch.zeros_like(true, dtype=torch.bool)
    m[3:-3, 4:-4, 4:-4] = True
    before = ((a - b)[m]).pow(2).mean()
    # widen the range and disable smoothing for this clean-translation mechanism test
    s = T.estimate_pe_shift(scans, max_shift_vox=3.0, smooth_sigma_vox=0.0)
    # The 2-vox differential is realigned by a ~2-vox correction.
    assert abs(abs(s) - 2.0) < 0.4, f"PE shift estimate off: {s}"
    T.apply_pe_shift(scans, s)
    after = ((scans[0].data - scans[1].data)[m]).pow(2).mean()
    assert after < 0.3 * before, f"shift did not realign: before={before} after={after}"


def test_reg_residual_invariants():
    # constant field -> zero bending and membrane energy
    f = torch.full((8, 8, 8), 3.0)
    assert T.reg_residual(f, "bending").abs().max() < 1e-6
    assert T.reg_residual(f, "membrane").abs().max() < 1e-6
    # linear ramp -> zero bending (2nd deriv), nonzero membrane (1st deriv)
    ramp = torch.arange(8).float()[None, None, :].expand(8, 8, 8).contiguous()
    assert T.reg_residual(ramp, "bending").abs().max() < 1e-5
    assert T.reg_residual(ramp, "membrane").abs().max() > 0.5


def _small_gn_setup(dtype=torch.float64):
    """Small field problem shared by the analytic-GN correctness tests."""
    _, _, scans = _make_synthetic(nz=12, ny=16, nx=16)
    scans = [T.ScanSpec(s.data.to(dtype), s.pe_axis, s.sign, s.readout) for s in scans]
    shape = tuple(scans[0].data.shape)
    basis = T.build_spline_basis(shape, (3.0, 2.5, 2.5), 12.0, torch.device("cpu"), dtype)
    mask = T.compute_mask(scans)
    mask_idx = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(-1)
    coeff = 0.02 * torch.randn(basis.coeff_shape, dtype=dtype)
    return basis, scans, mask_idx, coeff


def test_field_adjoint_is_transpose():
    # <field(v), g> == <v, field_adjoint(g)> for the separable B-spline expansion.
    torch.manual_seed(1)
    basis = T.build_spline_basis((10, 12, 14), (3.0, 2.5, 2.5), 12.0, torch.device("cpu"))
    v = torch.randn(basis.coeff_shape, dtype=torch.float64)
    g = torch.randn((10, 12, 14), dtype=torch.float64)
    lhs = torch.dot(basis.field(v).reshape(-1), g.reshape(-1))
    rhs = torch.dot(v.reshape(-1), basis.field_adjoint(g).reshape(-1))
    assert torch.allclose(lhs, rhs, rtol=1e-9, atol=1e-9), (lhs.item(), rhs.item())


def test_analytic_jacobian_adjoint_identity():
    # The analytic J and J^T must satisfy <J v, u> == <v, J^T u> exactly (dot test).
    torch.manual_seed(2)
    basis, scans, mask_idx, coeff = _small_gn_setup()
    lin = T._linearize(coeff, basis, scans, mask_idx, 1, 1e-3, "bending")
    v = torch.randn(basis.coeff_shape, dtype=torch.float64)
    jv = T._lin_jv(lin, v)
    u = torch.randn_like(jv)
    jtu = T._lin_jtu(lin, u)
    lhs = torch.dot(jv.reshape(-1), u.reshape(-1))
    rhs = torch.dot(v.reshape(-1), jtu.reshape(-1))
    assert torch.allclose(lhs, rhs, rtol=1e-8, atol=1e-8), (lhs.item(), rhs.item())


def test_antifold_barrier_operator_matches_autograd():
    # With the anti-fold barrier ON, the analytic J / J^T / J^T r must still equal the
    # reverse-mode-autodiff operator to ~machine precision, and satisfy the adjoint identity.
    # A large penalty makes the relu active so the barrier rows are exercised.
    torch.manual_seed(5)
    basis, scans, mask_idx, coeff = _small_gn_setup()
    be, jf = 5.0, 0.1

    lin = T._linearize(coeff, basis, scans, mask_idx, 1, 1e-3, "bending", be, jf)
    assert lin.barrier_b, "barrier linearisation should be populated"
    v = torch.randn(basis.coeff_shape, dtype=torch.float64)
    jv = T._lin_jv(lin, v)
    u = torch.randn_like(jv)
    lhs = torch.dot(jv.reshape(-1), u.reshape(-1))
    rhs = torch.dot(v.reshape(-1), T._lin_jtu(lin, u).reshape(-1))
    assert torch.allclose(lhs, rhs, rtol=1e-9, atol=1e-9), "barrier adjoint identity"

    c = coeff.detach().requires_grad_(True)
    r0 = T._residual_vector(c, basis, scans, mask_idx, 1, 1e-3, "bending", be, jf)
    w = torch.zeros_like(r0, requires_grad=True)
    (jtw,) = torch.autograd.grad(r0, c, grad_outputs=w, create_graph=True, retain_graph=True)
    jv_auto = torch.autograd.grad(jtw, w, grad_outputs=v, retain_graph=True)[0]
    assert (jv - jv_auto).norm() < 1e-9 * jv_auto.norm(), "J v mismatch (barrier)"
    g_auto = torch.autograd.grad(r0, c, grad_outputs=r0.detach(), retain_graph=True)[0].reshape(-1)
    g_an = T._lin_jtu(lin, r0)
    assert (g_an - g_auto).norm() < 1e-9 * g_auto.norm(), "J^T r mismatch (barrier)"


def _make_folding_pair():
    """Textured volume + a steep localized field that makes the unconstrained fit fold."""
    from fastfuncstuff.processing.cost import _separable_smooth_3d

    torch.manual_seed(0)
    nz, ny, nx = 20, 40, 40
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz).float(), torch.arange(ny).float(), torch.arange(nx).float(), indexing="ij"
    )
    mask = (((zz - nz / 2) / 8) ** 2 + ((yy - ny / 2) / 16) ** 2 + ((xx - nx / 2) / 16) ** 2) < 1.0
    t = _separable_smooth_3d(torch.randn(nz, ny, nx), 1.0)
    t = t - t.min()
    true = (t / t.max() * 100.0 + 10.0) * mask.float()
    field = 11.0 * torch.exp(
        -(((yy - 14) / 2.5) ** 2 + ((xx - nx / 2) / 7) ** 2 + ((zz - nz / 2) / 5) ** 2)
    )
    ro, pe_tdim = 0.5, 1

    def observed(sign):
        disp = field * (ro * sign)
        return T._resample_pe(true, -disp, pe_tdim) / T._jacobian_pe(disp, pe_tdim).clamp(min=0.05)

    scans = [
        T.ScanSpec(observed(+1.0), 1, +1.0, ro),
        T.ScanSpec(observed(-1.0), 1, -1.0, ro),
    ]
    return scans, ro, pe_tdim


def test_antifold_barrier_removes_fold():
    # A field steep enough to fold one polarity: the barrier must drive the min Jacobian
    # back positive, while leaving the field almost unchanged where it was already safe.
    scans, ro, pe_tdim = _make_folding_pair()
    kw = dict(
        warpres=[16, 8, 5],
        fwhm=[4, 1, 0],
        lam=[1e-3, 1e-6, 1e-9],
        miter=[8, 12, 15],
        subsamp=[1, 1, 1],
    )

    off = T.run_topup(scans, (3.0, 2.5, 2.5), T.TopupConfig(jac_penalty=0.0, **kw), progress=False)
    jac_off = T._jacobian_pe(off.field_hz * ro, pe_tdim)
    assert float(jac_off.min()) < 0.0, f"test setup did not fold (min {jac_off.min()})"

    on = T.run_topup(scans, (3.0, 2.5, 2.5), T.TopupConfig(jac_penalty=5e-2, **kw), progress=False)
    jac_on = T._jacobian_pe(on.field_hz * ro, pe_tdim)
    assert float(jac_on.min()) > 0.0, f"barrier failed to remove fold (min {jac_on.min()})"

    # Local: where the unconstrained warp was comfortably safe, the field barely moves.
    safe = jac_off > 0.5
    rms_safe = float((on.field_hz - off.field_hz)[safe].pow(2).mean().sqrt())
    assert rms_safe < 0.2 * float(off.field_hz.std()), f"barrier perturbed safe region: {rms_safe}"


def test_analytic_gn_operator_matches_autograd():
    # The analytic J / J^T J / g must equal the reverse-mode-autodiff operator to
    # ~machine precision. (We compare the *operator*, not the CG solution: with the
    # tiny Levenberg damping the normal equations are ill-conditioned, so bit-level
    # rounding differences between two exact-to-1e-16 matvecs drift over the hundreds
    # of CG iterations — the operator identity is the correctness invariant.)
    torch.manual_seed(3)
    basis, scans, mask_idx, coeff = _small_gn_setup()
    lam = 1e-3

    c = coeff.detach().requires_grad_(True)
    r0 = T._residual_vector(c, basis, scans, mask_idx, 1, lam, "bending")
    w = torch.zeros_like(r0, requires_grad=True)
    (jtw,) = torch.autograd.grad(r0, c, grad_outputs=w, create_graph=True, retain_graph=True)

    def vjp(u):
        return torch.autograd.grad(r0, c, grad_outputs=u, retain_graph=True)[0]

    v = torch.randn(basis.coeff_shape, dtype=torch.float64)
    jv_auto = torch.autograd.grad(jtw, w, grad_outputs=v, retain_graph=True)[0]

    lin = T._linearize(coeff, basis, scans, mask_idx, 1, lam, "bending")
    jv_an = T._lin_jv(lin, v)
    assert (jv_an - jv_auto).norm() < 1e-10 * jv_auto.norm(), "J v mismatch"

    jtjv_auto = vjp(jv_auto).reshape(-1)
    jtjv_an = T._lin_jtu(lin, jv_an).reshape(-1)
    assert (jtjv_an - jtjv_auto).norm() < 1e-10 * jtjv_auto.norm(), "J^T J v mismatch"

    g_auto = vjp(r0.detach()).reshape(-1)
    g_an = T._lin_jtu(lin, r0)
    assert (g_an - g_auto).norm() < 1e-10 * g_auto.norm(), "J^T r mismatch"


def test_analytic_and_autograd_recover_same_field():
    # End-to-end: the two matvecs recover the same field (very high correlation; CG
    # drift on the ill-conditioned normal equations forbids bit-identity).
    _, field, scans = _make_synthetic()
    kw = dict(warpres=[16, 10], fwhm=[5, 2], lam=[1e-3, 1e-4], miter=[6, 6], subsamp=[1, 1])
    res_a = T.run_topup(
        scans, (3.0, 2.5, 2.5), T.TopupConfig(analytic_gn=True, **kw), progress=False
    )
    res_b = T.run_topup(
        scans, (3.0, 2.5, 2.5), T.TopupConfig(analytic_gn=False, **kw), progress=False
    )
    m = torch.zeros_like(field, dtype=torch.bool)
    m[3:-3, 4:-4, 4:-4] = True
    corr = torch.corrcoef(torch.stack([res_a.field_hz[m], res_b.field_hz[m]]))[0, 1].item()
    assert corr > 0.999, f"analytic vs autograd field correlation too low: {corr}"


def test_spline_field_shape_and_smoothness():
    basis = T.build_spline_basis((10, 12, 14), (3.0, 2.5, 2.5), 12.0, torch.device("cpu"))
    coeff = torch.randn(basis.coeff_shape, dtype=torch.float64)
    fld = basis.field(coeff)
    assert fld.shape == (10, 12, 14)
    # refit round-trip: fitting a field's own coeffs reproduces it closely
    fld2 = basis.field(T.refit_coeff(fld, basis))
    assert (fld - fld2).abs().max() < 1e-3 * fld.abs().max().clamp(min=1.0)


# ---------------------------------------------------------------------------
# Rigid movement estimation (joint with the field)
# ---------------------------------------------------------------------------
def test_default_estmov_is_coarse_levels_only():
    # Motion is estimated while the field is too coarse to mimic it (>=10 mm knots) and
    # frozen once it can, so the flags must be a run of True followed by a run of False
    # -- never re-enabled after being switched off.
    for lad in (T.make_ladder(9), T.make_ladder(12, res_start=30.0), T.make_ladder(3)):
        flags = T._default_estmov(lad.warpres)
        assert flags == sorted(flags, reverse=True), flags
        assert flags[0] is True, "the coarsest level must estimate motion"
        assert flags[-1] is False, "the finest level must not"


def test_ladder_rule_is_monotone_and_terminates_unsmoothed():
    for kw in ({}, dict(n_levels=12, res_start=30.0), dict(n_levels=14, res_final=2.0)):
        lad = T.make_ladder(**kw)
        n = len(lad.warpres)
        assert len(lad.fwhm) == len(lad.lam) == len(lad.miter) == n
        # Knots refine (never coarsen), smoothing releases, penalty releases, effort grows.
        assert lad.warpres == sorted(lad.warpres, reverse=True)
        assert lad.fwhm == sorted(lad.fwhm, reverse=True)
        assert lad.lam == sorted(lad.lam, reverse=True)
        assert lad.miter == sorted(lad.miter)
        # The last level fits real data at the field's final resolution.
        assert lad.fwhm[-1] == 0.0
        assert lad.warpres[-1] == min(lad.warpres)


def test_drop_pe_center_translation():
    from fastfuncstuff.processing.affine import identity_params, params_to_matrix

    shape = (20, 32, 32)  # (nz, ny, nx); PE = y/j (axis 1)
    pe_axis = 1
    p = identity_params(dtype=torch.float64)
    p[3] = 3.0  # rz (deg) — a real rotation to preserve
    p[1] = 1.5  # dy — the PE-axis translation to remove
    mat = params_to_matrix(p)
    mat2 = T._drop_pe_center_translation(mat, shape, pe_axis)

    nz, ny, nx = shape
    c = torch.tensor([(nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2, 1.0], dtype=torch.float64)
    d = mat2 @ c - c
    assert abs(d[pe_axis].item()) < 1e-8, "PE displacement at centre not removed"
    # Non-PE rows (and the whole rotation block) are untouched.
    assert torch.allclose(mat2[0], mat[0])
    assert torch.allclose(mat2[2], mat[2])
    assert torch.allclose(mat2[pe_axis, :3], mat[pe_axis, :3])


def _apply_rigid(vol, params):
    """Resample vol under a rigid transform given by 6 ffs_moco params."""
    from fastfuncstuff.processing.affine import (
        _build_homo_coords,
        identity_params,
        params_to_matrix,
        resample_affine_fast,
    )

    p = identity_params(device=vol.device, dtype=vol.dtype)
    p[:6] = torch.tensor(params, device=vol.device, dtype=vol.dtype)
    coords = _build_homo_coords(tuple(vol.shape), vol.device, vol.dtype)
    return resample_affine_fast(vol, params_to_matrix(p), coords, "cubic", tuple(vol.shape))


def _make_textured(nz=24, ny=40, nx=40, readout=0.5):
    """A textured (brain-like) volume + smooth Hz field -> blip-up/down pair (PE=y/j).

    The two-blob :func:`_make_synthetic` volume is too feature-poor for rigid
    registration to converge; a smoothed random field inside an ellipsoidal mask gives
    gradients in all three axes, which is what the Gauss-Newton rigid solver needs.
    """
    from fastfuncstuff.processing.cost import _separable_smooth_3d

    torch.manual_seed(0)
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz).float(), torch.arange(ny).float(), torch.arange(nx).float(), indexing="ij"
    )
    mask = (((zz - nz / 2) / 9) ** 2 + ((yy - ny / 2) / 16) ** 2 + ((xx - nx / 2) / 16) ** 2) < 1.0
    t = _separable_smooth_3d(torch.randn(nz, ny, nx), 1.2)
    t = t - t.min()
    true = (t / t.max() * 100.0 + 10.0) * mask.float()
    field = 6.0 * torch.sin(2 * math.pi * yy / ny) * torch.exp(-(((xx - nx / 2) / 12) ** 2))
    return true, field, readout


def test_motion_estimation_reconciles_moved_pair():
    # Blip-down acquired after a small rigid head move dominated by a THROUGH-PLANE (z)
    # translation, which a phase-encode (y) field physically cannot represent -- so unlike
    # an in-plane move (which the smooth field can partly absorb) only rigid motion
    # estimation can reconcile the pair. Estimation should cut the corrected-pair residual
    # substantially and recover a non-trivial transform for the moved scan.
    true, field, readout = _make_textured()
    pe_tdim = 1

    def observed(src, sign):
        disp = field * (readout * sign)
        return T._resample_pe(src, -disp, pe_tdim) / T._jacobian_pe(disp, pe_tdim).clamp(min=0.1)

    # dz=1.3 vox (through-plane, non-PE), rz=1.0 deg; no PE-axis (dy) translation.
    true_moved = _apply_rigid(true, [0.0, 0.0, 1.3, 1.0, 0.0, 0.0])

    def make_scans():
        up = T.ScanSpec(observed(true, +1.0), pe_axis=1, sign=+1.0, readout=readout)
        down = T.ScanSpec(observed(true_moved, -1.0), pe_axis=1, sign=-1.0, readout=readout)
        return [up, down]

    cfg = T.TopupConfig(
        warpres=[16, 10], fwhm=[4, 2], lam=[1e-3, 1e-4], miter=[8, 8], subsamp=[1, 1]
    )
    m = torch.zeros_like(true, dtype=torch.bool)
    m[3:-3, 6:-6, 6:-6] = True

    res_off = T.run_topup(make_scans(), (3.0, 2.5, 2.5), cfg, progress=False)
    res_on = T.run_topup(make_scans(), (3.0, 2.5, 2.5), cfg, progress=False, estimate_motion=True)

    resid_off = ((res_off.unwarped[0] - res_off.unwarped[1])[m]).pow(2).mean().sqrt()
    resid_on = ((res_on.unwarped[0] - res_on.unwarped[1])[m]).pow(2).mean().sqrt()
    assert resid_on < 0.7 * resid_off, f"motion did not help: off={resid_off} on={resid_on}"

    # Reference scan stays put; the moved scan recovers ~ -1.3 vox of z-translation.
    eye = torch.eye(4, dtype=res_on.motion_matrices[0].dtype)
    assert torch.allclose(res_on.motion_matrices[0], eye), "reference scan should not move"
    dz = res_on.motion_matrices[1][2, 3].item()
    assert -1.7 < dz < -0.9, f"z-translation not recovered: {dz}"


def test_wrap_pad_pe_takes_the_opposite_end():
    vol = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)
    out = T.wrap_pad_pe(vol, [1], 2)
    assert out.shape == (2, 10, 3)
    assert torch.equal(out[:, :2], vol[:, -2:])
    assert torch.equal(out[:, 2:8], vol)
    assert torch.equal(out[:, 8:], vol[:, :2])


def _aliasing_pair(nz=16, ny=32, nx=28, readout=0.5):
    """Tissue against both PE edges, pushed across them: the acquisition aliases."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz).float(), torch.arange(ny).float(), torch.arange(nx).float(), indexing="ij"
    )
    true = torch.full((nz, ny, nx), 5.0)
    for cy, a in [(1.5, 120.0), (30.0, 90.0), (16, 60.0)]:
        true += a * torch.exp(
            -(((zz - 8) / 5) ** 2 + ((yy - cy) / 2.5) ** 2 + ((xx - 14) / 6) ** 2)
        )
    field = 6.0 * torch.cos(2 * math.pi * (yy + 0.5) / ny) * torch.exp(-(((xx - 14) / 10) ** 2))

    def observed(sign):
        disp = field * readout * sign
        coord = (torch.arange(ny).float()[None, :, None] - disp) % ny  # periodic PE sampling
        lo = coord.floor().long() % ny
        hi = (lo + 1) % ny
        fr = coord - coord.floor()
        o = torch.gather(true, 1, lo) * (1 - fr) + torch.gather(true, 1, hi) * fr
        return o / T._jacobian_pe(disp, 1).clamp(min=0.1)

    return field, [
        T.ScanSpec(observed(+1.0), 1, +1.0, readout),
        T.ScanSpec(observed(-1.0), 1, -1.0, readout),
    ]


def test_wrap_padding_reconciles_blips_at_the_pe_edges():
    """Without the wrap, signal aliased across the FOV edge is matched where it landed."""
    field, scans = _aliasing_pair()
    band = torch.zeros_like(field, dtype=torch.bool)
    band[3:-3, :5, 4:-4] = True
    band[3:-3, -5:, 4:-4] = True

    def edge_disagreement(pad):
        cfg = T.TopupConfig(
            warpres=[16, 10], fwhm=[5, 2], lam=[1e-3, 1e-4], miter=[8, 8], subsamp=[1, 1]
        )
        r = T.run_topup(scans, (3.0, 2.5, 2.5), cfg, progress=False, wrap_pad=pad)
        assert r.field_hz.shape == field.shape and r.unwarped[0].shape == field.shape
        return ((r.unwarped[0] - r.unwarped[1])[band]).pow(2).mean().sqrt().item()

    # Measured 4.41 -> 2.56.
    assert edge_disagreement(4) < 0.75 * edge_disagreement(0)


def test_unwrap_pe_moves_the_slab_and_zero_fills():
    v = torch.arange(1, 7, dtype=torch.float32).reshape(1, 6, 1)  # y = 1..6
    up = T.unwrap_pe(v, 1, -2, front=1, back=2)[0, :, 0].tolist()
    assert up == [0, 0, 0, 3, 4, 5, 6, 1, 2]  # first 2 moved past the end, zeros behind
    down = T.unwrap_pe(v, 1, +1, front=1, back=2)[0, :, 0].tolist()
    assert down == [6, 1, 2, 3, 4, 5, 0, 0, 0]  # last 1 moved before the start
    still = T.unwrap_pe(v, 1, 0, front=1, back=2)[0, :, 0].tolist()
    assert still == [0, 1, 2, 3, 4, 5, 6, 0, 0]


def test_unwrap_recovers_the_field_where_only_one_blip_aliased():
    """Only blip_up's edge tissue left the FOV; saying so should beat guessing symmetric."""
    nz, big, nx, ro, lo = 16, 40, 28, 0.5, 4
    n = big - 2 * lo
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz).float(), torch.arange(big).float(), torch.arange(nx).float(), indexing="ij"
    )
    true = 5.0 * ((yy > lo) & (yy < big - lo - 1)).float()
    for cy, a in [(lo + 2.5, 120.0), (20, 60.0), (big - lo - 4, 80.0)]:
        true = true + a * torch.exp(
            -(((zz - 8) / 5) ** 2 + ((yy - cy) / 2.5) ** 2 + ((xx - 14) / 6) ** 2)
        )
    field = -8.0 * torch.exp(-(((yy - lo - 3) / 5) ** 2)) * torch.exp(-(((xx - 14) / 10) ** 2))

    def acquire(sign):
        disp = field * ro * sign
        return T._resample_pe(true, -disp, 1) / T._jacobian_pe(disp, 1).clamp(min=0.1)

    def fov(v):  # crop to the FOV; what fell outside aliases in at the other end
        out = v[:, lo : lo + n].clone()
        out[:, -lo:] += v[:, :lo]
        out[:, :lo] += v[:, lo + n :]
        return out

    def cfg():
        return T.TopupConfig(
            warpres=[16, 10], fwhm=[5, 2], lam=[1e-3, 1e-4], miter=[8, 8], subsamp=[1, 1]
        )

    up, down = acquire(+1.0), acquire(-1.0)
    ref = T.run_topup(
        [T.ScanSpec(up, 1, 1.0, ro), T.ScanSpec(down, 1, -1.0, ro)],
        (3, 2.5, 2.5),
        cfg(),
        progress=False,
    )
    scans = [T.ScanSpec(fov(up), 1, 1.0, ro), T.ScanSpec(fov(down), 1, -1.0, ro)]
    band = torch.zeros(nz, n, nx, dtype=torch.bool)
    band[3:-3, :8, 4:-4] = True

    def err(**kw):
        r = T.run_topup(scans, (3, 2.5, 2.5), cfg(), progress=False, **kw)
        assert r.field_hz.shape == (nz, n, nx)
        return (r.field_hz - ref.field_hz[:, lo : lo + n])[band].abs().mean().item()

    none, right, wrong = err(), err(unwrap=[4, 0]), err(unwrap=[-4, 0])
    # Measured 0.60 / 0.26 / 5.16 Hz against an 8 Hz peak.
    assert right < 0.6 * none
    assert wrong > 3 * none


def test_pad_phase_wrap_sign_is_anatomical_not_index_order():
    """+K must move the posterior slab forward however the file is stored.

    Read as index order, +9 on a scan indexed posterior-to-anterior put the air in
    front of the head behind the occipital pole.
    """
    import numpy as np

    from fastfuncstuff.cli.blipflip import _anatomical_moves_to_index_shifts

    ras = np.diag([2.0, 2.0, 2.0, 1.0])  # j increases toward anterior
    lpi = np.diag([-2.0, -2.0, 2.0, 1.0])  # j increases toward posterior
    vol = torch.arange(1, 11, dtype=torch.float32).reshape(1, 10, 1)
    for affine, posterior_first in ((ras, True), (lpi, False)):
        (shift,) = _anatomical_moves_to_index_shifts([+3], [1], affine)
        out = T.unwrap_pe(vol, 1, shift, front=max(0, shift), back=max(0, -shift))[0, :, 0]
        # The three most posterior slices must now sit beyond the anterior end.
        posterior = [1.0, 2.0, 3.0] if posterior_first else [10.0, 9.0, 8.0]
        beyond_anterior = out[-3:].tolist() if posterior_first else out[:3].flip(0).tolist()
        assert beyond_anterior == posterior


def test_padding_skips_all_zero_edge_planes():
    """A reversed-PE recon's empty edge line must not be dragged into the image."""
    v = torch.tensor([1, 2, 3, 4, 5, 0], dtype=torch.float32).reshape(1, 6, 1)  # empty last plane
    # Last 2 DATA slices (4, 5) move before the start; the empty plane stays at the edge.
    out = T.unwrap_pe(v, 1, +2, front=2, back=0)[0, :, 0].tolist()
    assert out == [4, 5, 1, 2, 3, 0, 0, 0]
    # First 2 slices move past the end, against the data (not after the empty plane).
    out = T.unwrap_pe(v, 1, -2, front=0, back=2)[0, :, 0].tolist()
    assert out == [0, 0, 3, 4, 5, 1, 2, 0]
    # Symmetric wrap: no zero plane between the volume and its wrapped neighbours.
    out = T.wrap_pad_pe(v, [1], 2)[0, :, 0].tolist()
    assert out == [4, 5, 1, 2, 3, 4, 5, 1, 2, 0]
    assert out[2:7] == [1, 2, 3, 4, 5]  # data keeps its own voxels, so the crop is unchanged
