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
    res_a = T.run_topup(scans, (3.0, 2.5, 2.5), T.TopupConfig(analytic_gn=True, **kw), progress=False)
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
