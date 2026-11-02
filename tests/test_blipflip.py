"""Tests for the blip-up/down (FSL topup-style) distortion estimator.

Synthetic data only: we verify the field solver recovers a known off-resonance
field, that the two blips agree after correction, and a few primitive invariants.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from fastfuncstuff.processing import blipflip as T


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
    cfg = T.BlipflipConfig(
        warpres=[16, 10], fwhm=[5, 2], lam=[1e-3, 1e-4], miter=[8, 8], subsamp=[1, 1]
    )
    res = T.run_blipflip(scans, (3.0, 2.5, 2.5), cfg, progress=False)

    m = torch.zeros_like(field, dtype=torch.bool)
    m[3:-3, 4:-4, 4:-4] = True
    ft, fr = field[m], res.field_hz[m]
    corr = torch.corrcoef(torch.stack([ft, fr]))[0, 1].item()
    assert corr > 0.9, f"field correlation too low: {corr}"


def test_blips_agree_after_correction():
    _, _, scans = _make_synthetic()
    cfg = T.BlipflipConfig(
        warpres=[16, 10], fwhm=[5, 2], lam=[1e-3, 1e-4], miter=[8, 8], subsamp=[1, 1]
    )
    # caller's tensors must be untouched by run_blipflip
    before_data = scans[0].data.clone()
    res = T.run_blipflip(scans, (3.0, 2.5, 2.5), cfg, progress=False)
    assert torch.allclose(scans[0].data, before_data), "run_blipflip mutated caller scans"

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


@pytest.mark.slow  # 11s
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

    off = T.run_blipflip(
        scans, (3.0, 2.5, 2.5), T.BlipflipConfig(jac_penalty=0.0, **kw), progress=False
    )
    jac_off = T._jacobian_pe(off.field_hz * ro, pe_tdim)
    assert float(jac_off.min()) < 0.0, f"test setup did not fold (min {jac_off.min()})"

    on = T.run_blipflip(
        scans, (3.0, 2.5, 2.5), T.BlipflipConfig(jac_penalty=5e-2, **kw), progress=False
    )
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


def test_air_rows_operator_matches_autograd():
    # The air rows ride the barrier's B * D_pe blocks; with both on, the analytic J, J^T J
    # and the fused normal operator must still be the autodiff operator.
    torch.manual_seed(5)
    basis, scans, mask_idx, coeff = _small_gn_setup()
    w = T.air_stretch_weight(scans, 0.5)
    assert 0 < float(w.mean()) < 1, "the fixture needs some dark voxels"
    air = [(w * a, td) for td, a in T._fold_barrier_specs(scans)]
    lam, bar, air_eff = 1e-3, 5.0, 0.3

    c = coeff.detach().requires_grad_(True)
    r0 = T._residual_vector(
        c, basis, scans, mask_idx, 1, lam, "bending", bar, 0.1, air=air, air_eff=air_eff
    )
    wv = torch.zeros_like(r0, requires_grad=True)
    (jtw,) = torch.autograd.grad(r0, c, grad_outputs=wv, create_graph=True, retain_graph=True)
    v = torch.randn(basis.coeff_shape, dtype=torch.float64)
    jv_auto = torch.autograd.grad(jtw, wv, grad_outputs=v, retain_graph=True)[0]
    jtjv_auto = torch.autograd.grad(r0, c, grad_outputs=jv_auto, retain_graph=True)[0].reshape(-1)

    lin = T._linearize(
        coeff, basis, scans, mask_idx, 1, lam, "bending", bar, 0.1, air=air, air_eff=air_eff
    )
    jv_an = T._lin_jv(lin, v)
    assert (jv_an - jv_auto).norm() < 1e-10 * jv_auto.norm(), "J v mismatch"
    jtjv_an = T._lin_jtu(lin, jv_an).reshape(-1)
    assert (jtjv_an - jtjv_auto).norm() < 1e-10 * jtjv_auto.norm(), "J^T J v mismatch"
    fused = T._NormalOperator(lin, T.reg_gram_terms(basis, "bending"))(v.reshape(-1))
    ref = jtjv_an + 1e-8 * v.reshape(-1)
    assert (fused - ref).norm() < 1e-10 * ref.norm(), "fused normal operator mismatch"


def test_air_term_flattens_the_field_where_there_is_no_signal():
    # The data term does not see the dark corners, so a field gradient started there
    # survives a fit at a tiny bending weight -- unless the air rows ask for a unit
    # Jacobian. Inside the head the data still decide.
    torch.manual_seed(9)
    _, _, scans = _make_synthetic()
    shape = tuple(scans[0].data.shape)
    for s in scans:  # dark bands at both PE ends: noise only, like air beside a head
        noise = 0.01 * s.data.mean() * torch.rand_like(s.data)
        s.data[:, :6], s.data[:, -6:] = noise[:, :6], noise[:, -6:]
    w = T.air_stretch_weight(scans, 0.3)
    assert float((w > 0.5).float().mean()) > 0.05, "fixture needs air"
    y = torch.linspace(-1, 1, shape[1], dtype=scans[0].data.dtype)
    start = 40.0 * torch.sin(3 * math.pi * y).view(1, -1, 1).expand(shape).contiguous()

    def stretch_in_air(air_weight):
        cfg = T.BlipflipConfig(
            warpres=[8.0],
            fwhm=[0.0],
            lam=[1e-9],
            miter=[6],
            subsamp=[1],
            air_weight=air_weight,
            air_max_warpres=8.0,
        )
        res = T.run_blipflip(
            [T.ScanSpec(s.data.clone(), s.pe_axis, s.sign, s.readout) for s in scans],
            (3.0, 2.5, 2.5),
            cfg,
            progress=False,
            init_field=start,
            mask_field=False,
        )
        g = (T._central_diff_pe(res.field_hz, 1) * scans[0].readout).abs()
        return float(g[w > 0.5].mean())

    assert stretch_in_air(1e-2) < 0.5 * stretch_in_air(0.0)


def test_air_weight_spares_dark_gaps_inside_the_head():
    # The occipital fissure is dark but enclosed; a brightness-only weight flattened the
    # field there (ses-fine fissure gap 1.22 -> 1.83 vox). Only exterior dark counts.
    _, _, scans = _make_synthetic()
    for s in scans:
        noise = 0.01 * s.data.mean() * torch.rand_like(s.data)
        s.data[:, :6], s.data[:, -6:] = noise[:, :6], noise[:, -6:]
        mid = s.data.shape[2] // 2
        s.data[:, 12:-12, mid - 1 : mid + 1] = noise[:, 12:-12, mid - 1 : mid + 1]  # a fissure
    w = T.air_stretch_weight(scans, 0.3)
    mid = w.shape[2] // 2
    assert float(w[:, 14:-14, mid - 1 : mid + 1].max()) < 0.05, "the enclosed gap got weight"
    assert float(w[:, :3].mean()) > 0.5, "the outside air lost its weight"


def test_analytic_and_autograd_recover_same_field():
    # End-to-end: the two matvecs recover the same field (very high correlation; CG
    # drift on the ill-conditioned normal equations forbids bit-identity).
    _, field, scans = _make_synthetic()
    kw = dict(warpres=[16, 10], fwhm=[5, 2], lam=[1e-3, 1e-4], miter=[6, 6], subsamp=[1, 1])
    res_a = T.run_blipflip(
        scans, (3.0, 2.5, 2.5), T.BlipflipConfig(analytic_gn=True, **kw), progress=False
    )
    res_b = T.run_blipflip(
        scans, (3.0, 2.5, 2.5), T.BlipflipConfig(analytic_gn=False, **kw), progress=False
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


def test_subsampled_schedule_is_rejected_until_handoffs_are_unit_correct():
    cfg = T.BlipflipConfig(
        warpres=[16, 10], fwhm=[5, 2], lam=[1e-3, 1e-4], miter=[8, 8], subsamp=[2, 1]
    )
    with pytest.raises(ValueError, match="supports only 1"):
        cfg.validate()


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


@pytest.mark.slow  # 16s
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

    cfg = T.BlipflipConfig(
        warpres=[16, 10], fwhm=[4, 2], lam=[1e-3, 1e-4], miter=[8, 8], subsamp=[1, 1]
    )
    m = torch.zeros_like(true, dtype=torch.bool)
    m[3:-3, 6:-6, 6:-6] = True

    res_off = T.run_blipflip(make_scans(), (3.0, 2.5, 2.5), cfg, progress=False)
    res_on = T.run_blipflip(
        make_scans(), (3.0, 2.5, 2.5), cfg, progress=False, estimate_motion=True
    )

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
        cfg = T.BlipflipConfig(
            warpres=[16, 10], fwhm=[5, 2], lam=[1e-3, 1e-4], miter=[8, 8], subsamp=[1, 1]
        )
        r = T.run_blipflip(scans, (3.0, 2.5, 2.5), cfg, progress=False, wrap_pad=pad)
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
        return T.BlipflipConfig(
            warpres=[16, 10], fwhm=[5, 2], lam=[1e-3, 1e-4], miter=[8, 8], subsamp=[1, 1]
        )

    up, down = acquire(+1.0), acquire(-1.0)
    ref = T.run_blipflip(
        [T.ScanSpec(up, 1, 1.0, ro), T.ScanSpec(down, 1, -1.0, ro)],
        (3, 2.5, 2.5),
        cfg(),
        progress=False,
    )
    scans = [T.ScanSpec(fov(up), 1, 1.0, ro), T.ScanSpec(fov(down), 1, -1.0, ro)]
    band = torch.zeros(nz, n, nx, dtype=torch.bool)
    band[3:-3, :8, 4:-4] = True

    def err(**kw):
        r = T.run_blipflip(scans, (3, 2.5, 2.5), cfg(), progress=False, **kw)
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


def test_mask_frames_only_the_phase_encode_axis():
    # The frame keeps the clamped PE resample out of the cost. Framing the slice axis
    # too left a thin slab's end slices with no data term, and their field ran away.
    _, _, scans = _make_synthetic()  # PE = j (tensor dim 1)
    m = T.compute_mask(scans)
    assert not m[:, 0, :].any() and not m[:, -1, :].any()
    assert m[0].any() and m[-1].any(), "end slices must stay in the cost"
    assert m[:, :, 0].any() and m[:, :, -1].any(), "readout edges must stay in the cost"


@pytest.mark.slow  # 14s
def test_estmov_takes_moved_out_of_slab_voxels_out_of_the_cost():
    # A through-plane move leaves one end of the moved scan with no measured data; the
    # border-clamped resample fills it with a copy of the edge slice, which must not
    # count as data (a thin slab's end slice broke on HiHiFaces ses-fine).
    true, _, readout = _make_textured()
    moved = _apply_rigid(true, [0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    work = [
        T.ScanSpec(true.clone(), pe_axis=1, sign=+1.0, readout=readout),
        T.ScanSpec(moved, pe_axis=1, sign=-1.0, readout=readout),
    ]
    mats = [torch.eye(4) for _ in work]
    valid = [torch.ones(true.shape, dtype=torch.bool) for _ in work]
    T.estimate_motion_pass(work, torch.zeros_like(true), 0, mats, valid=valid)
    first = valid[1].clone()
    # The next level's pass finds ~no further motion; the mask must not erode with it
    # (moving the mask itself each pass lost a voxel per level).
    T.estimate_motion_pass(work, torch.zeros_like(true), 0, mats, valid=valid)

    assert valid[0].all(), "the reference scan never moves"
    lost = ~valid[1]
    assert lost[:1].all() or lost[-1:].all(), "the vacated end slice must be invalid"
    assert not lost[3:-3].any(), "interior slices keep their data"
    assert (first ^ valid[1]).float().mean() < 0.01
    m = T.compute_mask(work, valid=valid)
    assert not (m & lost).any()


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


def test_declared_empty_positions_are_scored_as_zeros():
    """Vacated slab and padding count; a scan's own empty edge plane does not."""
    v = torch.tensor([1, 2, 3, 4, 5, 0], dtype=torch.float32).reshape(1, 6, 1)
    scans = [T.ScanSpec(v, 1, 1.0, 0.05)]
    data, tdims, front, empty = T.pad_scans_pe(scans, unwrap=[+2])
    assert data[0][0, :, 0].tolist() == [4, 5, 1, 2, 3, 0, 0, 0]
    # 5, 6: where 4 and 5 came from (declared empty); 7: the recon's empty line (unmeasured).
    assert empty is not None and empty[0].tolist() == [False] * 5 + [True, True, False]

    big = data[0].expand(3, 8, 3).contiguous()
    padded = [T.ScanSpec(big, 1, 1.0, 0.05)]
    blind = T.compute_mask(padded)[1, :, 1].tolist()
    scored = T.compute_mask(padded, empty, pe_tdim=1)[1, :, 1].tolist()
    # compute_mask also drops the outer frame plane on every axis (index 0 and 7 here).
    assert blind[5:7] == [False, False]
    assert scored[5:7] == [True, True]
    assert scored[7] is False


@pytest.mark.slow  # 14s
def test_level_stops_when_the_field_stops_moving():
    # The displacement stop must end a level before miter and report why. Mechanism only:
    # on this phantom the sub-0.1-vox steps it cuts still added up to ~0.6 vox, so how
    # early to stop is a real-data question, not something to assert here.
    _, _, scans = _make_synthetic()

    def run(min_update):
        cfg = T.BlipflipConfig(
            warpres=[16, 10], fwhm=[5, 2], lam=[1e-3, 1e-4], miter=[40, 40], subsamp=[1, 1]
        )
        cfg.min_update_vox = min_update
        return T.run_blipflip(scans, (3.0, 2.5, 2.5), cfg, progress=False)

    early, full = run(0.1), run(0.0)
    # The displacement stop fires when enabled, and only then.
    assert [lv.stop for lv in early.levels] == ["converged", "converged"]
    assert all(lv.last_update_vox < 0.1 for lv in early.levels)
    assert all(lv.stop != "converged" for lv in full.levels)
    # ...and ending early means fewer Gauss-Newton steps. Not asserted on the
    # exact reason min_update=0.0 ends with: "miter" vs the cost-plateau stop
    # turns on float noise at ~1e-7, so it moves with thread count.
    assert all(
        lv_full.iters >= lv_early.iters
        for lv_early, lv_full in zip(early.levels, full.levels, strict=True)
    )


def test_help_schedule_lines_round_trip_to_the_preset():
    # The -help preset lines are meant to be pasted back on the command line.
    import shlex

    from fastfuncstuff.cli import blipflip as B

    for name in B._PRESETS:
        cfg = B._build_config(name)
        argv = ["-config", name, "-pe_dir", "j", "-prefix", "x"]
        argv += shlex.split(B._schedule_flags(cfg, "").replace("\n", " "))
        args = B.create_parser().parse_args(argv)
        assert args.warpres == cfg.warpres and args.fwhm == cfg.fwhm
        assert args.miter == cfg.miter
        assert all(math.isclose(a, b, rel_tol=0.01) for a, b in zip(args.lam, cfg.lam, strict=True))


@pytest.mark.parametrize("tier", ["workhard", "superhard"])
def test_high_tiers_are_unsmoothed_down_to_two_voxel_knots(tier):
    # Smoothing under these tiers' weak penalty set the falx field from the bulk tissue at
    # the coarse levels, and the sharp fine levels could not undo it (HiHiFaces ses-fine,
    # 2026-10-29). The knot floor follows the data: two voxels, not a fixed 2 mm, which
    # was a 1-voxel floor on 2 mm data and overfit there.
    from fastfuncstuff.cli import blipflip as B

    for vox in (0.8, 2.0):
        cfg = B._build_config(tier, vox)
        assert set(cfg.fwhm) == {0.0}
        assert cfg.warpres[-1] == pytest.approx(2 * vox, abs=0.05)
        assert cfg.warpres[0] == 30.0
        assert cfg.warpres == sorted(cfg.warpres, reverse=True)


def test_fused_normal_matvec_equals_jt_j():
    # The CG matvec skips the masked gather/scatter and applies the penalty through
    # coefficient-space Kronecker Grams; it must still be exactly J^T J (+ the 1e-8 ridge),
    # with the barrier active, two scans, and both penalty models.
    torch.manual_seed(7)
    basis, scans, mask_idx, coeff = _small_gn_setup()
    for reg_mode in ("bending", "membrane"):
        lin = T._linearize(coeff, basis, scans, mask_idx, 1, 3e-3, reg_mode, 5.0, 0.1)
        assert lin.barrier_b, "barrier rows should be active"
        matvec = T._NormalOperator(lin, T.reg_gram_terms(basis, reg_mode))
        v = torch.randn(basis.coeff_shape, dtype=torch.float64)
        ref = T._lin_jtu(lin, T._lin_jv(lin, v)) + 1e-8 * v.reshape(-1)
        got = matvec(v.reshape(-1))
        assert (got - ref).norm() < 1e-10 * ref.norm(), reg_mode


def _anat_term_for(basis, scans, mask_idx, coeff):
    """An anat term whose target is refit to the scans undistorted by ``coeff``."""
    torch.manual_seed(11)
    shape = tuple(scans[0].data.shape)
    anat = T._AnatTerm.build(
        200.0 - scans[0].data + torch.randn(shape, dtype=scans[0].data.dtype), 2.0, 1
    )
    field = basis.field(coeff)
    und = [T._undistort_scan(s.data, field, s.readout, s.sign, 1) for s in scans]
    mask = torch.zeros(shape, dtype=torch.bool).reshape(-1)
    mask[mask_idx] = True
    anat.update(torch.stack(und).mean(0), mask.reshape(shape))
    return anat


def test_anat_rows_match_autograd_and_the_fused_matvec():
    # The anat rows' target is held fixed within a step, so their analytic J is the
    # uncentred per-scan map; J v, J^T r and the fused CG matvec must all agree with
    # autograd / J^T J to machine precision, with the barrier on as well.
    torch.manual_seed(13)
    basis, scans, mask_idx, coeff = _small_gn_setup()
    anat = _anat_term_for(basis, scans, mask_idx, coeff)
    lam, be, ae = 1e-3, 5.0, 0.7

    c = coeff.detach().requires_grad_(True)
    r0 = T._residual_vector(c, basis, scans, mask_idx, 1, lam, "bending", be, 0.1, anat, ae)
    w = torch.zeros_like(r0, requires_grad=True)
    (jtw,) = torch.autograd.grad(r0, c, grad_outputs=w, create_graph=True, retain_graph=True)
    v = torch.randn(basis.coeff_shape, dtype=torch.float64)
    jv_auto = torch.autograd.grad(jtw, w, grad_outputs=v, retain_graph=True)[0]

    lin = T._linearize(coeff, basis, scans, mask_idx, 1, lam, "bending", be, 0.1, ae)
    jv = T._lin_jv(lin, v)
    assert jv.shape == r0.shape
    assert (jv - jv_auto).norm() < 1e-10 * jv_auto.norm(), "J v mismatch (anat)"
    g_auto = torch.autograd.grad(r0, c, grad_outputs=r0.detach(), retain_graph=True)[0]
    g_an = T._lin_jtu(lin, r0)
    assert (g_an - g_auto.reshape(-1)).norm() < 1e-10 * g_auto.norm(), "J^T r mismatch (anat)"

    matvec = T._NormalOperator(lin, T.reg_gram_terms(basis, "bending"))
    ref = T._lin_jtu(lin, jv) + 1e-8 * v.reshape(-1)
    assert (matvec(v.reshape(-1)) - ref).norm() < 1e-10 * ref.norm(), "fused matvec (anat)"


# The synthetic schedule is coarser than the real-data anat threshold; switch it on throughout.
_ANAT_KW = dict(
    warpres=[16, 10],
    fwhm=[5, 2],
    lam=[1e-3, 1e-4],
    miter=[8, 8],
    subsamp=[1, 1],
    anat_max_warpres=100.0,
)


def _field_rms_error(res, field):
    m = torch.zeros_like(field, dtype=torch.bool)
    m[3:-3, 4:-4, 4:-4] = True
    return float((res.field_hz - field)[m].pow(2).mean().sqrt())


def _make_exact_pair(readout=0.5):
    """:func:`_make_synthetic`'s anatomy and field, but each blip made with the exact
    inverse of the correction, so the true field undistorts both back to ``true``.

    ``_make_synthetic`` resamples ``true`` at ``-d``, which only approximately inverts
    ``resample(., +d)`` when ``d`` varies: with 3.5-voxel shifts on 4-6-voxel blobs the
    true field leaves the pair visibly apart, so an anat of ``true`` is not what the
    corrected blips look like and cannot referee them."""
    from fastfuncstuff.processing.medic import invert_displacement_pe

    true, field, _ = _make_synthetic(readout=readout)
    scans = []
    for sign in (+1.0, -1.0):
        d = field * (readout * sign)
        h = invert_displacement_pe(d, 1, iters=200)
        obs = T._resample_pe(true / T._jacobian_pe(d, 1), h, 1)
        scans.append(T.ScanSpec(obs, pe_axis=1, sign=sign, readout=readout))
    return true, field, scans


def test_exact_pair_is_undistorted_by_its_true_field():
    # Measured: RMS 1.0 and 0.16 against a std of 19.3 (the approximate generator: 17.6
    # and 10.8). Not a max bound: two linear interpolations leave single-voxel errors of
    # ~9 at the steepest edge.
    true, field, scans = _make_exact_pair()
    m = torch.zeros_like(true, dtype=torch.bool)
    m[3:-3, 4:-4, 4:-4] = True
    for sc in scans:
        und = T._undistort_scan(sc.data, field, sc.readout, sc.sign, 1)
        assert float((und - true)[m].pow(2).mean().sqrt()) < 0.1 * float(true[m].std())


def test_inverted_contrast_anat_does_not_hurt_recovery():
    # A T1 against a T2*-weighted EPI is roughly an inverted contrast: the local linear
    # map must absorb that (a negative slope), not fight the pair.
    true, field, scans = _make_exact_pair()
    plain = T.run_blipflip(scans, (3.0, 2.5, 2.5), T.BlipflipConfig(**_ANAT_KW), progress=False)
    cfg = T.BlipflipConfig(**_ANAT_KW, anat_weight=1.0, anat_sigma_mm=8.0)
    with_anat = T.run_blipflip(scans, (3.0, 2.5, 2.5), cfg, progress=False, anat=300.0 - true)
    assert _field_rms_error(with_anat, field) <= 1.05 * _field_rms_error(plain, field)


def test_robust_weighted_rows_match_autograd_and_the_fused_matvec():
    # With per-voxel weights on the data rows, the analytic J v / J^T r and the fused CG
    # operator (which folds the weights into the data mask) must still equal autograd / J^T J.
    torch.manual_seed(17)
    basis, scans, mask_idx, coeff = _small_gn_setup()
    shape = tuple(scans[0].data.shape)
    sqrt_w = torch.rand(shape, dtype=torch.float64) * 0.9 + 0.1
    lam, be = 1e-3, 5.0

    c = coeff.detach().requires_grad_(True)
    r0 = T._residual_vector(
        c, basis, scans, mask_idx, 1, lam, "bending", be, 0.1, None, 0.0, sqrt_w
    )
    w = torch.zeros_like(r0, requires_grad=True)
    (jtw,) = torch.autograd.grad(r0, c, grad_outputs=w, create_graph=True, retain_graph=True)
    v = torch.randn(basis.coeff_shape, dtype=torch.float64)
    jv_auto = torch.autograd.grad(jtw, w, grad_outputs=v, retain_graph=True)[0]

    lin = T._linearize(coeff, basis, scans, mask_idx, 1, lam, "bending", be, 0.1, 0.0, sqrt_w)
    jv = T._lin_jv(lin, v)
    assert (jv - jv_auto).norm() < 1e-10 * jv_auto.norm(), "J v mismatch (robust)"
    g_auto = torch.autograd.grad(r0, c, grad_outputs=r0.detach(), retain_graph=True)[0]
    assert (T._lin_jtu(lin, r0) - g_auto.reshape(-1)).norm() < 1e-10 * g_auto.norm()

    matvec = T._NormalOperator(lin, T.reg_gram_terms(basis, "bending"))
    ref = T._lin_jtu(lin, jv) + 1e-8 * v.reshape(-1)
    assert (matvec(v.reshape(-1)) - ref).norm() < 1e-10 * ref.norm(), "fused matvec (robust)"


def test_robust_weights_zero_one_sided_signal_and_spare_the_rest():
    # At the true field the exact pair agrees everywhere except where only one blip has
    # signal: the weights must be ~1 on the agreeing tissue and ~0 on the one-sided blob,
    # or the reweighting would either do nothing or throw away good data.
    true, field, scans = _make_exact_pair()
    zz, yy, xx = torch.meshgrid(*(torch.arange(n).float() for n in true.shape), indexing="ij")
    blob = 120.0 * torch.exp(-(((zz - 10) / 2) ** 2 + ((yy - 20) / 2) ** 2 + ((xx - 12) / 2) ** 2))
    scans = [T.ScanSpec(scans[0].data + blob, 1, scans[0].sign, scans[0].readout), scans[1]]
    mask = T.compute_mask(scans)
    w = T.robust_weights(scans, field, mask)
    core = blob > 60.0
    far = (blob < 1e-3) & mask
    assert float(w[core].max()) < 0.05, "one-sided signal kept weight"
    assert float(w[far].median()) > 0.8, "agreeing tissue was down-weighted"


def test_init_field_starts_the_first_level_from_it():
    # One Gauss-Newton step: from zero it cannot get far, from the true field it is
    # already there. (Run to convergence both end near the pair's own optimum, which on
    # this pair is ~0.3 Hz off the truth either way.)
    _, field, scans = _make_exact_pair()
    kw = dict(warpres=[8], fwhm=[0], lam=[1e-4], miter=[1], subsamp=[1])
    cold = T.run_blipflip(scans, (3.0, 2.5, 2.5), T.BlipflipConfig(**kw), progress=False)
    warm = T.run_blipflip(
        scans, (3.0, 2.5, 2.5), T.BlipflipConfig(**kw), progress=False, init_field=field
    )
    assert _field_rms_error(warm, field) < 0.5 * _field_rms_error(cold, field)


def test_anat_misaligned_along_pe_does_not_drag_the_field():
    # A shift common to both blips (here the anat is 1.5 voxels off along PE, as a rigid
    # alignment error or a scanner-to-scanner scaling would be) needs both undistorted
    # blips to move the same way. A field moves them oppositely, so it must stay put.
    true, field, scans = _make_exact_pair()
    plain = T.run_blipflip(scans, (3.0, 2.5, 2.5), T.BlipflipConfig(**_ANAT_KW), progress=False)
    off = T._resample_pe(300.0 - true, torch.full_like(true, 1.5), 1)
    cfg = T.BlipflipConfig(**_ANAT_KW, anat_weight=1.0, anat_sigma_mm=8.0)
    shifted = T.run_blipflip(scans, (3.0, 2.5, 2.5), cfg, progress=False, anat=off)
    m = torch.zeros_like(field, dtype=torch.bool)
    m[3:-3, 4:-4, 4:-4] = True
    moved = float((shifted.field_hz - plain.field_hz)[m].pow(2).mean().sqrt())
    # Measured 0.37 Hz (frame on every axis) and 0.40 Hz (PE-only frame) of a 3.9 Hz std;
    # the failure this guards against -- no PE-gradient regressor -- moved it 3.2 Hz.
    assert moved < 0.15 * float(field[m].std()), f"common-mode anat offset moved the field {moved}"


@pytest.mark.gpu
@pytest.mark.parametrize("with_anat", [False, True])
def test_cuda_graphed_cg_matches_eager(with_anat):
    # The per-level CG replays one captured iteration against buffers the linearisation is
    # copied into; across several Gauss-Newton steps it must give exactly the eager result.
    # With an anat, the residual graphs also read a target rewritten in place every step.
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    true, _, scans = _make_exact_pair()
    dev = torch.device("cuda")
    scans = [T.ScanSpec(s.data.to(dev), s.pe_axis, s.sign, s.readout) for s in scans]
    anat = (300.0 - true).to(dev) if with_anat else None

    def run(graphs):
        cfg = T.BlipflipConfig(
            warpres=[16, 10], fwhm=[5, 2], lam=[1e-3, 1e-4], miter=[6, 6], subsamp=[1, 1]
        )
        cfg.cuda_graphs = graphs
        cfg.anat_max_warpres = 100.0
        return T.run_blipflip(scans, (3.0, 2.5, 2.5), cfg, progress=False, anat=anat)

    eager, graphed = run(False), run(True)
    assert [lv.iters for lv in eager.levels] == [lv.iters for lv in graphed.levels]
    assert torch.equal(eager.field_hz, graphed.field_hz)


def test_blip_down_on_a_shifted_fov_is_regridded_onto_blip_up():
    # HiHiFaces ses-fine: the reverse-PE sbref was acquired with its FOV 4.3 mm off along
    # readout, headers correct. Comparing raw voxel arrays fed the fit a 5-voxel shift the
    # PE-only field cannot explain, and the "corrected" EPI came out worse than the input.
    from fastfuncstuff.cli import blipflip as B

    torch.manual_seed(0)
    world = torch.rand(4, 12, 10)  # (nz, ny, nx)
    ref_aff = np.diag([0.8, 0.8, 0.8, 1.0])
    shifted_aff = ref_aff.copy()
    shifted_aff[1, 3] += 3 * 0.8  # origin 3 voxels further along j
    shifted = torch.zeros_like(world)
    shifted[:, :-3] = world[:, 3:]  # the same anatomy, seen through the shifted FOV

    out = B._onto_first_grid(shifted, shifted_aff, tuple(world.shape), ref_aff, "down", 0)
    assert torch.equal(out[:, 3:], world[:, 3:])
    assert not out[:, :3].any()  # outside the shifted FOV: no data, not wrapped tissue

    same = B._onto_first_grid(world, ref_aff, tuple(world.shape), ref_aff, "down", 0)
    assert same is world

    rotated = ref_aff.copy()
    rotated[:3, :3] = np.array([[0, -0.8, 0], [0.8, 0, 0], [0, 0, 0.8]])
    with pytest.raises(SystemExit, match="rotated grid"):
        B._onto_first_grid(world, rotated, tuple(world.shape), ref_aff, "down", 0)


def test_fine_tiers_estimate_movement_unless_told_not_to():
    from fastfuncstuff.cli import blipflip as B

    base = ["-pe_dir", "j", "-prefix", "x"]
    for argv, want in (
        (["-superhard"], True),
        (["-workhard"], True),
        ([], False),
        (["-superhard", "-no_estmov"], False),
        (["-estmov"], True),
    ):
        args = B.create_parser().parse_args(argv + base)
        assert B._estmov_for(args.config, args.estmov) is want, argv
