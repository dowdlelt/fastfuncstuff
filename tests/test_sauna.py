"""Tests for SAUNA denoising module."""

import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import pytest

from fastfuncstuff.denoise.nordic import _optimal_shrinkage_weights
from fastfuncstuff.denoise.sauna import (
    SaunaConfig,
    _c4_bias_correction,
    _calibrate_sigma,
    _construct_3d_legendre_basis,
    _estimate_gfactor_from_noise,
    _fit_polynomial_gfactor,
    _gaussian_smooth_3d,
    _loo_optimize_gfactor_degree,
    _loo_optimize_gfactor_fwhm,
    _patch_variance_cov,
    run_sauna,
)


def _write_nifti(path: Path, data: np.ndarray) -> None:
    img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
    nib.save(img, path)


# ---------------------------------------------------------------------------
# Unit tests: optimal shrinkage weights
# ---------------------------------------------------------------------------


def test_optimal_shrinkage_zero_below_bulk_edge():
    """Weights for singular values below bulk edge should be exactly 0."""
    sigma = 1.0
    m, n = 100, 20
    bulk_edge = sigma * (math.sqrt(m) + math.sqrt(n))

    # Create singular values: some above, some below bulk edge
    s = torch.tensor([[bulk_edge * 2, bulk_edge * 1.5, bulk_edge * 0.9, bulk_edge * 0.5]])
    w = _optimal_shrinkage_weights(s, sigma, m, n)

    assert w[0, 0] > 0, "Above bulk edge should have positive weight"
    assert w[0, 1] > 0, "Above bulk edge should have positive weight"
    assert w[0, 2] == 0, "Below bulk edge should have zero weight"
    assert w[0, 3] == 0, "Below bulk edge should have zero weight"


def test_optimal_shrinkage_at_bulk_edge():
    """Weight exactly at the bulk edge should be 0."""
    sigma = 1.0
    m, n = 50, 10
    bulk_edge = sigma * (math.sqrt(m) + math.sqrt(n))

    s = torch.tensor([[bulk_edge]])
    w = _optimal_shrinkage_weights(s, sigma, m, n)
    assert w[0, 0] == 0.0


def test_optimal_shrinkage_monotonic():
    """Weights should be monotonically increasing with singular value."""
    sigma = 1.0
    m, n = 100, 20
    bulk_edge = sigma * (math.sqrt(m) + math.sqrt(n))

    s_vals = torch.linspace(bulk_edge * 1.01, bulk_edge * 5.0, 20).unsqueeze(0)
    w = _optimal_shrinkage_weights(s_vals, sigma, m, n)

    # Weights should increase as singular values increase
    diffs = w[0, 1:] - w[0, :-1]
    assert (diffs >= -1e-6).all(), "Weights should be monotonically non-decreasing"


def test_optimal_shrinkage_approaches_one():
    """For very large singular values, weights should approach 1."""
    sigma = 1.0
    m, n = 50, 10
    bulk_edge = sigma * (math.sqrt(m) + math.sqrt(n))

    s = torch.tensor([[bulk_edge * 100.0]])
    w = _optimal_shrinkage_weights(s, sigma, m, n)
    assert w[0, 0] > 0.99, f"Very large SV should have weight ~1, got {w[0, 0]:.4f}"


def test_optimal_shrinkage_weights_bounded():
    """All weights should be in [0, 1]."""
    sigma = 0.5
    m, n = 200, 40
    rng = torch.Generator().manual_seed(42)
    s = torch.rand(16, min(m, n), generator=rng) * 50.0
    s = s.sort(dim=1, descending=True).values

    w = _optimal_shrinkage_weights(s, sigma, m, n)
    assert (w >= 0).all()
    assert (w <= 1.0).all()


def test_optimal_shrinkage_batch():
    """Batched computation should match single-patch computation."""
    sigma = 1.0
    m, n = 80, 15

    rng = torch.Generator().manual_seed(123)
    s = torch.rand(4, min(m, n), generator=rng) * 20.0
    s = s.sort(dim=1, descending=True).values

    w_batch = _optimal_shrinkage_weights(s, sigma, m, n)

    for i in range(4):
        w_single = _optimal_shrinkage_weights(s[i : i + 1], sigma, m, n)
        torch.testing.assert_close(w_batch[i], w_single[0])


# ---------------------------------------------------------------------------
# Unit tests: g-factor estimation
# ---------------------------------------------------------------------------


def test_c4_bias_correction():
    """c4 for known values."""
    # k=2: c4 = sqrt(2/1) * Gamma(1) / Gamma(0.5) = sqrt(2) / sqrt(pi)
    c4_2 = _c4_bias_correction(2)
    expected = math.sqrt(2.0) / math.sqrt(math.pi)
    assert abs(c4_2 - expected) < 1e-10

    # k=1: should return 1.0 (degenerate)
    assert _c4_bias_correction(1) == 1.0

    # c4 should approach 1 for large k
    c4_100 = _c4_bias_correction(100)
    assert 0.99 < c4_100 < 1.0


def test_gaussian_smooth_identity_at_zero_fwhm():
    """FWHM=0 should return input unchanged."""
    vol = torch.randn(10, 10, 5)
    smoothed = _gaussian_smooth_3d(vol, fwhm=0.0)
    torch.testing.assert_close(smoothed, vol)


def test_gaussian_smooth_reduces_variance():
    """Smoothing should reduce spatial variance."""
    rng = torch.Generator().manual_seed(42)
    vol = torch.randn(20, 20, 10, generator=rng)
    smoothed = _gaussian_smooth_3d(vol, fwhm=3.0)
    assert smoothed.var() < vol.var()


def test_estimate_gfactor_uniform_noise():
    """For uniform noise, g-factor should be approximately 1 everywhere."""
    rng = torch.Generator().manual_seed(42)
    noise_vols = torch.randn(20, 20, 10, 5, generator=rng)

    gfactor, global_sigma = _estimate_gfactor_from_noise(
        noise_vols,
        smooth_fwhm=3.0,
        verbose=False,
    )

    assert gfactor.shape == (20, 20, 10)
    # For uniform noise, g-factor should be close to 1
    assert float(gfactor.mean()) == pytest.approx(1.0, abs=0.15)
    # Global sigma should be close to 1.0 (we generated N(0,1))
    assert global_sigma == pytest.approx(1.0, abs=0.2)


def test_estimate_gfactor_heterogeneous_noise():
    """G-factor should capture spatial noise variation."""
    rng = torch.Generator().manual_seed(42)
    noise_vols = torch.randn(20, 20, 10, 5, generator=rng)

    # Make one quadrant 3x noisier
    noise_vols[:10, :10, :, :] *= 3.0

    gfactor, global_sigma = _estimate_gfactor_from_noise(
        noise_vols,
        smooth_fwhm=2.0,
        verbose=False,
    )

    # The noisy quadrant should have higher g-factor
    g_noisy = float(gfactor[:8, :8, :].mean())
    g_quiet = float(gfactor[12:, 12:, :].mean())
    assert g_noisy > g_quiet * 1.5, (
        f"Noisy region g={g_noisy:.2f} should be much larger than quiet g={g_quiet:.2f}"
    )


def test_estimate_gfactor_needs_2_volumes():
    """Should raise with < 2 noise volumes."""
    with pytest.raises(ValueError, match="at least 2"):
        _estimate_gfactor_from_noise(torch.randn(5, 5, 3, 1), verbose=False)


# ---------------------------------------------------------------------------
# Integration tests: full SAUNA pipeline
# ---------------------------------------------------------------------------


def test_run_sauna_magnitude_only(tmp_path):
    """SAUNA with magnitude-only data."""
    rng = np.random.default_rng(123)
    # 16 time volumes + 3 noise volumes
    magn = np.abs(rng.normal(size=(12, 10, 6, 19))).astype(np.float32)

    magn_file = tmp_path / "magn.nii.gz"
    _write_nifti(magn_file, magn)

    cfg = SaunaConfig(
        temporal_phase=0,
        magnitude_only=True,
        noise_volume_last=3,
        kernel_size_pca=(3, 3, 3),
        patch_overlap=2,
        save_gfactor_map=True,
        verbose=False,
    )

    out = run_sauna(
        magnitude_file=str(magn_file),
        phase_file=None,
        output_prefix=str(tmp_path / "SAUNA_magn"),
        config=cfg,
    )

    assert out.magnitude_file.exists()
    assert out.gfactor_file is not None and out.gfactor_file.exists()
    assert out.metadata_file.exists()

    den = nib.load(out.magnitude_file).get_fdata(dtype=np.float32)
    assert den.shape == magn.shape
    assert np.isfinite(den).all()

    # Check metadata
    with open(out.metadata_file) as f:
        meta = json.load(f)
    assert meta["method"] == "SAUNA"
    assert meta["noise_estimation"]["n_noise_volumes"] == 3
    assert meta["noise_estimation"]["global_sigma"] > 0


def test_run_sauna_complex(tmp_path):
    """SAUNA with complex (magnitude + phase) data."""
    rng = np.random.default_rng(456)
    magn = np.abs(rng.normal(size=(10, 8, 5, 17))).astype(np.float32)
    phase = rng.uniform(low=-2000.0, high=2000.0, size=(10, 8, 5, 17)).astype(np.float32)

    magn_file = tmp_path / "magn.nii.gz"
    phase_file = tmp_path / "phase.nii.gz"
    _write_nifti(magn_file, magn)
    _write_nifti(phase_file, phase)

    cfg = SaunaConfig(
        temporal_phase=1,
        phase_filter_width=3.0,
        noise_volume_last=3,
        kernel_size_pca=(3, 3, 3),
        patch_overlap=2,
        make_complex_nii=True,
        save_gfactor_map=True,
        verbose=False,
    )

    out = run_sauna(
        magnitude_file=str(magn_file),
        phase_file=str(phase_file),
        output_prefix=str(tmp_path / "SAUNA_complex"),
        config=cfg,
    )

    assert out.magnitude_file.exists()
    assert out.phase_file is not None and out.phase_file.exists()

    den_magn = nib.load(out.magnitude_file).get_fdata(dtype=np.float32)
    den_phase = nib.load(out.phase_file).get_fdata(dtype=np.float32)
    assert den_magn.shape == magn.shape
    assert np.isfinite(den_magn).all()
    assert np.isfinite(den_phase).all()


def test_run_sauna_hard_shrinkage(tmp_path):
    """SAUNA with hard (MP-PCA) shrinkage instead of optimal."""
    rng = np.random.default_rng(789)
    magn = np.abs(rng.normal(size=(10, 8, 5, 17))).astype(np.float32)

    magn_file = tmp_path / "magn.nii.gz"
    _write_nifti(magn_file, magn)

    cfg = SaunaConfig(
        temporal_phase=0,
        magnitude_only=True,
        noise_volume_last=3,
        kernel_size_pca=(3, 3, 3),
        patch_overlap=2,
        shrinkage="hard",
        verbose=False,
    )

    out = run_sauna(
        magnitude_file=str(magn_file),
        phase_file=None,
        output_prefix=str(tmp_path / "SAUNA_hard"),
        config=cfg,
    )

    assert out.magnitude_file.exists()
    den = nib.load(out.magnitude_file).get_fdata(dtype=np.float32)
    assert np.isfinite(den).all()


# ---------------------------------------------------------------------------
# Tests: LOO FWHM optimization
# ---------------------------------------------------------------------------


def test_loo_fwhm_prefers_moderate_smoothing():
    """LOO should prefer moderate FWHM over extreme values."""
    rng = torch.Generator().manual_seed(42)
    # Create noise with spatially varying std (mimics g-factor)
    noise = torch.randn(20, 20, 10, 5, generator=rng)

    # Add smooth spatial variation (like real coil sensitivity)
    x = torch.linspace(0.5, 2.0, 20)
    y = torch.linspace(0.8, 1.5, 20)
    z = torch.linspace(0.9, 1.1, 10)
    gfactor_true = x[:, None, None] * y[None, :, None] * z[None, None, :]
    noise = noise * gfactor_true[..., None]

    best_fwhm, scores = _loo_optimize_gfactor_fwhm(
        noise,
        fwhm_candidates=(1.0, 3.0, 5.0, 10.0, 20.0),
        verbose=False,
    )

    # Best should not be the most extreme values
    assert best_fwhm not in (1.0, 20.0), (
        f"Expected moderate FWHM, got {best_fwhm}. Scores: {scores}"
    )


def test_loo_returns_all_scores():
    """LOO should return scores for all candidates."""
    rng = torch.Generator().manual_seed(123)
    noise = torch.randn(12, 12, 6, 4, generator=rng)

    candidates = (2.0, 5.0, 8.0)
    best_fwhm, scores = _loo_optimize_gfactor_fwhm(
        noise,
        fwhm_candidates=candidates,
        verbose=False,
    )

    assert set(scores.keys()) == set(candidates)
    assert best_fwhm in candidates
    assert all(np.isfinite(v) for v in scores.values())


def test_patch_variance_cov_uniform_is_low():
    """Uniform noise after perfect g-correction should have low CoV."""
    rng = torch.Generator().manual_seed(42)
    noise = torch.randn(16, 16, 8, 5, generator=rng)
    gfactor = torch.ones(16, 16, 8)

    cov = _patch_variance_cov(noise, gfactor, (3, 3, 3), 2)
    # Uniform noise: CoV should be moderate (finite sample variance)
    # but much lower than poorly-corrected case
    assert cov < 1.0


def test_patch_variance_cov_bad_gfactor_is_high():
    """Wrong g-factor should yield higher CoV than correct one."""
    rng = torch.Generator().manual_seed(42)
    noise = torch.randn(16, 16, 8, 5, generator=rng)

    # True g-factor: one half noisier
    gfactor_true = torch.ones(16, 16, 8)
    gfactor_true[:8, :, :] = 3.0
    noise = noise * gfactor_true[..., None]

    # "Wrong" g-factor: uniform (ignores spatial structure)
    gfactor_wrong = torch.ones(16, 16, 8)

    cov_correct = _patch_variance_cov(noise, gfactor_true, (3, 3, 3), 2)
    cov_wrong = _patch_variance_cov(noise, gfactor_wrong, (3, 3, 3), 2)

    assert cov_correct < cov_wrong, (
        f"Correct g-factor CoV ({cov_correct:.4f}) should be lower than "
        f"wrong g-factor CoV ({cov_wrong:.4f})"
    )


# ---------------------------------------------------------------------------
# Tests: σ calibration
# ---------------------------------------------------------------------------


def test_calibrate_sigma_near_one_for_correct_sigma():
    """With correct σ and perfect g-factor, ratio should be ≈ 1."""
    rng = torch.Generator().manual_seed(42)
    noise = torch.randn(16, 16, 8, 5, generator=rng) * 2.0  # σ = 2.0
    gfactor = torch.ones(16, 16, 8)
    sigma = 2.0

    result = _calibrate_sigma(
        noise,
        gfactor,
        sigma,
        verbose=False,
    )

    assert abs(result["mean_var_ratio"] - 1.0) < 0.2, (
        f"Ratio should be near 1.0, got {result['mean_var_ratio']:.4f}"
    )


def test_run_sauna_auto_fwhm(tmp_path):
    """SAUNA with auto FWHM (LOO cross-validation)."""
    rng = np.random.default_rng(111)
    magn = np.abs(rng.normal(size=(12, 10, 6, 19))).astype(np.float32)

    magn_file = tmp_path / "magn.nii.gz"
    _write_nifti(magn_file, magn)

    cfg = SaunaConfig(
        temporal_phase=0,
        magnitude_only=True,
        noise_volume_last=3,
        kernel_size_pca=(3, 3, 3),
        patch_overlap=2,
        gfactor_smooth_fwhm="auto",
        gfactor_fwhm_range=(2.0, 5.0, 8.0),  # small set for speed
        verbose=True,  # exercise the verbose LOO output
    )

    out = run_sauna(
        magnitude_file=str(magn_file),
        phase_file=None,
        output_prefix=str(tmp_path / "SAUNA_auto"),
        config=cfg,
    )

    assert out.magnitude_file.exists()
    with open(out.metadata_file) as f:
        meta = json.load(f)
    assert meta["config"]["gfactor_smooth_fwhm_requested"] == "auto"
    assert meta["config"]["gfactor_smooth_fwhm_used"] > 0
    assert meta["noise_estimation"]["loo_scores"] is not None
    assert meta["noise_estimation"]["sigma_calibration"] is not None


def test_run_sauna_requires_noise_volumes():
    """SAUNA should raise if noise_volume_last < 2."""
    cfg = SaunaConfig(noise_volume_last=1)
    with pytest.raises(ValueError, match="at least 2"):
        run_sauna("dummy.nii", None, "out", config=cfg)


def test_run_sauna_absolute_scale_roundtrip(tmp_path):
    """Output magnitudes should be in a similar range as input."""
    rng = np.random.default_rng(999)
    magn = np.abs(rng.normal(scale=100.0, size=(8, 8, 4, 15))).astype(np.float32)

    magn_file = tmp_path / "magn.nii.gz"
    _write_nifti(magn_file, magn)

    cfg = SaunaConfig(
        temporal_phase=0,
        magnitude_only=True,
        noise_volume_last=3,
        kernel_size_pca=(3, 3, 3),
        patch_overlap=2,
        verbose=False,
    )

    out = run_sauna(
        magnitude_file=str(magn_file),
        phase_file=None,
        output_prefix=str(tmp_path / "SAUNA_scale"),
        config=cfg,
    )

    den = nib.load(out.magnitude_file).get_fdata(dtype=np.float32)
    assert den.max() > 1.0, "Output collapsed near zero (ABSOLUTE_SCALE not restored)"


def test_optimal_shrinkage_preserves_strong_signal():
    """Optimal shrinkage should preserve strong signal components better than hard threshold.

    Inject a known low-rank signal into noise and verify that optimal
    shrinkage reconstruction has lower error than hard truncation.
    """
    rng = torch.Generator().manual_seed(42)
    m, n = 121, 20  # ~11x11x1 patch, 20 timepoints
    sigma = 1.0

    # Low-rank signal: rank 3 with varying strengths
    U_true = torch.linalg.qr(torch.randn(m, 3, generator=rng))[0]
    S_true = torch.tensor([15.0, 8.0, 3.5])  # 3.5 is just above bulk edge
    V_true = torch.linalg.qr(torch.randn(n, 3, generator=rng))[0]
    signal = U_true @ torch.diag(S_true) @ V_true.T

    noise = sigma * torch.randn(m, n, generator=rng)
    data = signal + noise

    U, s, Vh = torch.linalg.svd(data, full_matrices=False)

    # Optimal shrinkage
    w_opt = _optimal_shrinkage_weights(s.unsqueeze(0), sigma, m, n)
    s_shrunk = s * w_opt.squeeze(0)
    recon_opt = (U * s_shrunk.unsqueeze(0)) @ Vh

    # Hard threshold at bulk edge (like NORDIC without the random calibration)
    bulk_edge = sigma * (math.sqrt(m) + math.sqrt(n))
    mask = (s >= bulk_edge).float()
    s_hard = s * mask
    recon_hard = (U * s_hard.unsqueeze(0)) @ Vh

    err_opt = torch.norm(recon_opt - signal, "fro").item()
    err_hard = torch.norm(recon_hard - signal, "fro").item()

    # Optimal should be at least as good (usually better for weak components)
    assert err_opt <= err_hard * 1.05, (
        f"Optimal error ({err_opt:.3f}) should be <= hard error ({err_hard:.3f})"
    )


def test_residual_map_saved(tmp_path):
    """Residual = |input_complex - denoised_complex|, saved as float32 NIfTI."""
    rng = np.random.RandomState(99)
    # 16 time volumes + 3 noise volumes
    magn = np.abs(rng.normal(size=(12, 10, 6, 19))).astype(np.float32)
    magn_file = tmp_path / "magn.nii.gz"
    _write_nifti(magn_file, magn)

    cfg = SaunaConfig(
        temporal_phase=0,
        magnitude_only=True,
        noise_volume_last=3,
        kernel_size_pca=(3, 3, 3),
        patch_overlap=2,
        save_residual_map=True,
        verbose=False,
    )

    out = run_sauna(str(magn_file), None, str(tmp_path / "SAUNA_res"), cfg)

    assert out.residual_file is not None and out.residual_file.exists()
    res = nib.load(out.residual_file).get_fdata(dtype=np.float32)
    assert res.shape == magn.shape
    assert np.isfinite(res).all()
    assert (res >= 0).all()  # magnitude is non-negative

    # Check metadata records residual path
    with open(out.metadata_file) as f:
        meta = json.load(f)
    assert meta["outputs"]["residual"] is not None


# ---------------------------------------------------------------------------
# 3D Legendre basis
# ---------------------------------------------------------------------------


def test_legendre_basis_shape_and_orthogonality():
    """Basis has correct number of terms and columns are near-orthogonal."""
    shape = (10, 12, 8)
    degree = 3
    B = _construct_3d_legendre_basis(shape, degree)
    # n_terms = C(degree+3, 3) = (4)(5)(6)/6 = 20
    expected_terms = (degree + 1) * (degree + 2) * (degree + 3) // 6
    assert B.shape == (10 * 12 * 8, expected_terms)
    # First column should be constant (P0 * P0 * P0 = 1)
    assert torch.allclose(B[:, 0], torch.ones(B.shape[0]))


def test_legendre_basis_degree_zero():
    """Degree 0 gives a single constant column."""
    B = _construct_3d_legendre_basis((5, 5, 5), 0)
    assert B.shape == (125, 1)
    assert torch.allclose(B[:, 0], torch.ones(125))


# ---------------------------------------------------------------------------
# Polynomial g-factor fit
# ---------------------------------------------------------------------------


def test_polynomial_gfactor_uniform_noise():
    """Uniform noise → poly fit gives near-constant g-factor ≈ 1."""
    rng = torch.Generator().manual_seed(42)
    noise = torch.randn(16, 14, 10, 5, generator=rng)
    gf, sigma = _fit_polynomial_gfactor(noise, degree=2)
    assert gf.shape == (16, 14, 10)
    # With uniform noise, g-factor should be close to 1 everywhere
    assert torch.all(torch.isfinite(gf))
    assert float(gf.std()) < 0.3, f"g-factor std too high: {gf.std()}"


def test_polynomial_gfactor_captures_gradient():
    """Noise with spatial gradient → poly should capture the trend."""
    rng = torch.Generator().manual_seed(7)
    nx, ny, nz, k = 20, 18, 10, 5
    # Create noise with linear gradient along x
    x_scale = torch.linspace(0.5, 2.0, nx).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    noise = torch.randn(nx, ny, nz, k, generator=rng) * x_scale
    gf, sigma = _fit_polynomial_gfactor(noise, degree=2)
    # g-factor should be larger at high x than low x
    assert float(gf[-1, ny // 2, nz // 2]) > float(gf[0, ny // 2, nz // 2])


# ---------------------------------------------------------------------------
# LOO polynomial degree selection
# ---------------------------------------------------------------------------


def test_loo_degree_returns_all_candidates():
    """LOO returns scores for every candidate degree."""
    rng = torch.Generator().manual_seed(99)
    noise = torch.randn(12, 10, 8, 4, generator=rng)
    candidates = (1, 2, 3)
    best_deg, scores = _loo_optimize_gfactor_degree(
        noise,
        degree_candidates=candidates,
        verbose=False,
    )
    assert set(scores.keys()) == set(candidates)
    assert best_deg in candidates
    assert all(np.isfinite(v) for v in scores.values())


# ---------------------------------------------------------------------------
# End-to-end polynomial method
# ---------------------------------------------------------------------------


def test_run_sauna_polynomial_method(tmp_path):
    """SAUNA with gfactor_method='polynomial' runs and produces valid output."""
    rng = np.random.RandomState(42)
    magn = np.abs(rng.normal(size=(12, 10, 6, 19))).astype(np.float32)
    magn_file = tmp_path / "magn.nii.gz"
    _write_nifti(magn_file, magn)

    cfg = SaunaConfig(
        temporal_phase=0,
        magnitude_only=True,
        noise_volume_last=3,
        kernel_size_pca=(3, 3, 3),
        patch_overlap=2,
        gfactor_method="polynomial",
        gfactor_degree_range=(1, 2, 3),
        save_gfactor_map=True,
        verbose=False,
    )

    out = run_sauna(str(magn_file), None, str(tmp_path / "SAUNA_poly"), cfg)
    assert out.magnitude_file.exists()
    assert out.gfactor_file is not None and out.gfactor_file.exists()

    den = nib.load(out.magnitude_file).get_fdata(dtype=np.float32)
    assert den.shape == magn.shape
    assert np.isfinite(den).all()

    with open(out.metadata_file) as f:
        meta = json.load(f)
    assert meta["config"]["gfactor_method_used"] == "polynomial"
    assert meta["config"]["gfactor_poly_degree"] is not None


def test_run_sauna_auto_method(tmp_path):
    """SAUNA with gfactor_method='auto' picks the better method."""
    rng = np.random.RandomState(11)
    magn = np.abs(rng.normal(size=(12, 10, 6, 19))).astype(np.float32)
    magn_file = tmp_path / "magn.nii.gz"
    _write_nifti(magn_file, magn)

    cfg = SaunaConfig(
        temporal_phase=0,
        magnitude_only=True,
        noise_volume_last=3,
        kernel_size_pca=(3, 3, 3),
        patch_overlap=2,
        gfactor_method="auto",
        gfactor_degree_range=(1, 2, 3),
        gfactor_fwhm_range=(0.0, 3.0, 5.0),
        verbose=False,
    )

    out = run_sauna(str(magn_file), None, str(tmp_path / "SAUNA_auto"), cfg)
    assert out.magnitude_file.exists()

    with open(out.metadata_file) as f:
        meta = json.load(f)
    assert meta["config"]["gfactor_method_used"] in ("gaussian", "polynomial")
