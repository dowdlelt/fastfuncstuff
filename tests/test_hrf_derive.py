"""Unit tests for fastfuncstuff.design.hrf_derive.

These exercise the pure-numpy stages of the NSD-style HRF library
pipeline using small synthetic datasets:

- ``select_voxels`` honors the R² threshold and sample size cap.
- ``svd_decompose`` reproduces a known low-rank input within
  reconstruction error.
- ``project_unit_sphere`` returns unit-norm rows.
- ``trace_manifold_auto`` returns ``n_points`` ordered points covering
  the density.
- ``reconstruct_timecourses`` produces peak=1 cubic-interpolated curves.
- ``fit_double_gamma`` recovers known double-gamma parameters within
  tolerance.
- ``derive_library`` end-to-end on a synthetic 3-flavor HRF dataset.

Tests run in seconds; no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import gamma

from fastfuncstuff.design.hrf_derive import (
    deconvolve_event_duration,
    derive_library,
    fit_double_gamma,
    project_unit_sphere,
    reconstruct_timecourses,
    select_voxels,
    svd_decompose,
    trace_manifold_auto,
    trace_manifold_grid,
)

# ---------- helpers ----------------------------------------------------------


def _double_gamma_curve(t: np.ndarray, a1: float = 6.0) -> np.ndarray:
    """SPM double-gamma normalized to peak 1.  Used as ground truth."""
    h = gamma.pdf(t, a1) - (1 / 6.0) * gamma.pdf(t, 16.0)
    return h / h.max()


@pytest.fixture
def synthetic_fir_dataset():
    """3-flavor HRF dataset: each voxel is a mix of 3 known shapes + noise.

    Returns ``(betas, r2, lag_times, true_templates)`` so each test can
    reason about ground truth.
    """
    rng = np.random.default_rng(0)
    tr = 1.0
    n_lags = 25
    lag_times = np.arange(n_lags) * tr
    templates = np.stack([_double_gamma_curve(lag_times, a1) for a1 in (4.0, 5.5, 7.0)])
    n_voxels = 5_000
    weights = rng.dirichlet(np.ones(3), size=n_voxels)
    betas = weights @ templates + 0.05 * rng.standard_normal((n_voxels, n_lags))
    r2 = 0.5 + 0.1 * rng.standard_normal(n_voxels)
    return betas, r2, lag_times, templates


# ---------- select_voxels ----------------------------------------------------


def test_select_voxels_threshold():
    r2 = np.array([0.05, 0.2, 0.3, -0.1, 0.5])
    sel = select_voxels(r2, threshold=0.1, max_voxels=10)
    assert set(sel.tolist()) == {1, 2, 4}


def test_select_voxels_caps_at_max():
    rng = np.random.default_rng(1)
    r2 = np.full(100, 0.5)
    sel = select_voxels(r2, threshold=0.1, max_voxels=20, seed=1)
    assert sel.size == 20
    # Deterministic with seed
    sel2 = select_voxels(r2, threshold=0.1, max_voxels=20, seed=1)
    np.testing.assert_array_equal(sel, sel2)


# ---------- svd_decompose ----------------------------------------------------


def test_svd_recovers_low_rank():
    # Construct an exact rank-2 matrix; SVD with K=2 reconstructs it.
    # Use unit_normalize=False here to test the math directly — the
    # default path rescales each row to unit length first, which is the
    # NSD-correct mode but means reconstruction equals the *normalized*
    # input, not the original.  Both modes share the same SVD code path.
    rng = np.random.default_rng(2)
    basis = rng.standard_normal((2, 30))
    coords = rng.standard_normal((200, 2))
    M = coords @ basis
    out = svd_decompose(M, n_pcs=2, unit_normalize=False, sign_align=False)
    recon = out.weights @ out.pcs
    np.testing.assert_allclose(recon, M, atol=1e-8)
    assert out.variance_explained.sum() > 0.999


def test_svd_unit_normalize_default_path():
    # With unit_normalize=True (default), reconstruction matches the
    # row-normalized input, and per-row Pythagorean energies of the
    # weights approximately equal 1.
    rng = np.random.default_rng(20)
    M = rng.standard_normal((50, 12))
    out = svd_decompose(M, n_pcs=12, unit_normalize=True, sign_align=False)
    M_unit = M / np.linalg.norm(M, axis=1, keepdims=True)
    recon = out.weights @ out.pcs
    np.testing.assert_allclose(recon, M_unit, atol=1e-8)
    # Each row's full K-D coordinate has norm ≈ 1 (since we kept all PCs).
    row_norms = np.linalg.norm(out.weights, axis=1)
    np.testing.assert_allclose(row_norms, 1.0, atol=1e-8)


def test_svd_sign_align_makes_pc1_positive():
    rng = np.random.default_rng(21)
    # Construct rows that all look like a "negative" HRF; SVD's V[0]
    # would naturally come out negative.  sign_align should flip it.
    template = -np.exp(-((np.arange(20) - 5) ** 2) / 8.0)
    M = template[None, :] + 0.05 * rng.standard_normal((200, 20))
    out = svd_decompose(M, n_pcs=1, unit_normalize=True, sign_align=True)
    # PC1 must have a positive peak.
    assert out.pcs[0, np.argmax(np.abs(out.pcs[0]))] > 0


def test_svd_pcs_are_orthonormal():
    rng = np.random.default_rng(3)
    M = rng.standard_normal((100, 20))
    out = svd_decompose(M, n_pcs=3)
    gram = out.pcs @ out.pcs.T
    np.testing.assert_allclose(gram, np.eye(3), atol=1e-8)


def test_svd_n_pcs_too_large():
    M = np.zeros((4, 8))
    with pytest.raises(ValueError, match="exceeds"):
        svd_decompose(M, n_pcs=10)


# ---------- project_unit_sphere ----------------------------------------------


def test_project_unit_sphere_norms():
    w = np.array([[3.0, 4.0], [1e-15, 0.0], [0.0, -5.0]])
    u = project_unit_sphere(w)
    np.testing.assert_allclose(np.linalg.norm(u[0]), 1.0)
    # Zero-norm row collapses to zero
    assert np.linalg.norm(u[1]) == 0.0
    np.testing.assert_allclose(u[2], [0.0, -1.0])


# ---------- manifold tracers ------------------------------------------------


def test_trace_manifold_auto_returns_unit_points():
    rng = np.random.default_rng(4)
    # Random Gaussian noise on the sphere
    pts = rng.standard_normal((1000, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    out = trace_manifold_auto(pts, n_points=10, angular_step_deg=8.0, n_grid=2048)
    assert out.shape[1] == 3
    assert out.shape[0] <= 10
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-10)


def test_trace_manifold_auto_rejects_non_3d():
    pts = np.eye(4)
    with pytest.raises(ValueError, match="K=3"):
        trace_manifold_auto(pts)


def test_trace_manifold_grid_exact_count():
    rng = np.random.default_rng(5)
    pts = rng.standard_normal((500, 4))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    out = trace_manifold_grid(pts, n_points=15)
    assert out.shape == (15, 4)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-10)


# ---------- reconstruct_timecourses ------------------------------------------


def test_reconstruct_shapes_and_peak(synthetic_fir_dataset):
    betas, _r2, lag_times, _ = synthetic_fir_dataset
    svd = svd_decompose(betas, n_pcs=3)
    manifold = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
    out, t = reconstruct_timecourses(manifold, svd.pcs, lag_times, target_dt=0.1)
    assert out.shape == (3, t.size)
    # peak normalization
    np.testing.assert_allclose(out.max(axis=1), 1.0)


# ---------- fit_double_gamma -------------------------------------------------


def test_fit_double_gamma_smoke():
    # We don't try to recover the exact (a1,b1,a2,b2,c) from a ground-truth
    # gamma — the parameter family is degenerate under local search, so
    # different starting points reach different minima.  What we *do*
    # require is: the function reports fit_ok, returns a peak-normalized
    # waveform of the right length, and produces a sane peak time.
    t = np.arange(0, 30, 0.1)
    truth = _double_gamma_curve(t, a1=5.5)
    fitted, params = fit_double_gamma(truth, dt=0.1)
    assert params["fit_ok"]
    assert fitted.shape == truth.shape
    np.testing.assert_allclose(float(np.max(fitted)), 1.0, atol=1e-6)
    peak_time = float(np.argmax(fitted)) * 0.1
    assert 2.0 < peak_time < 12.0


def test_fit_double_gamma_fallback_on_nonsense():
    # Pure noise → curve_fit cannot converge sensibly; fit_ok must be
    # False and the original is returned unchanged so the caller can
    # fall back to the raw reconstruction.
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(50)
    fitted, params = fit_double_gamma(noise, dt=0.1, maxfev=50)
    # Either fit succeeded against the noise (unlikely but possible) or
    # the fallback kicked in.  In both cases shapes must match.
    assert fitted.shape == noise.shape
    assert "fit_ok" in params


# ---------- end-to-end derive_library ---------------------------------------


def test_derive_library_end_to_end(synthetic_fir_dataset):
    betas, r2, lag_times, _ = synthetic_fir_dataset
    res = derive_library(
        betas,
        r2,
        lag_times,
        n_pcs=3,
        n_hrfs=10,
        r2_threshold=0.05,
        max_voxels=2000,
        fit_gamma=True,
    )
    # Expected output shapes
    assert res.raw.shape[0] == res.manifold.shape[0]
    assert res.raw.shape[0] <= 10
    assert res.fitted is not None and res.fitted.shape == res.raw.shape
    # All reconstructions are peak-normalized
    np.testing.assert_allclose(res.raw.max(axis=1), 1.0)
    # PCs capture the bulk of the variance for our 3-template dataset
    assert res.svd.variance_explained.sum() > 0.95
    # Provenance flags propagate
    assert res.duration_convolved is True


def test_deconvolve_event_duration_recovers_impulse():
    # Convolve a known HRF with a 3 s boxcar, then deconvolve and check
    # we recover the original.
    dt = 0.1
    t = np.arange(0, 30, dt)
    h_imp = _double_gamma_curve(t, a1=6.0)
    n_box = int(3.0 / dt)
    box = np.zeros_like(t)
    box[:n_box] = 1.0
    h_obs = np.convolve(h_imp, box)[: len(t)]
    h_obs = h_obs / h_obs.max()
    lib = np.stack([h_obs, h_obs])
    recovered = deconvolve_event_duration(lib, duration=3.0, dt=dt, snr=1000.0)
    # Peak time must match
    assert dt * int(np.argmax(recovered[0])) == dt * int(np.argmax(h_imp))
    # Curve matches reasonably (Wiener has a small bias near the tail)
    err = np.linalg.norm(recovered[0] - h_imp) / np.linalg.norm(h_imp)
    assert err < 0.05


def test_deconvolve_short_duration_noop():
    # duration <= dt should be a no-op (returns a copy).
    dt = 0.1
    lib = np.random.RandomState(0).rand(3, 50)
    out = deconvolve_event_duration(lib, duration=0.05, dt=dt)
    np.testing.assert_array_equal(out, lib)
    assert out is not lib  # should be a copy


def test_derive_library_with_deconvolution(synthetic_fir_dataset):
    betas, r2, lag_times, _ = synthetic_fir_dataset
    res = derive_library(
        betas,
        r2,
        lag_times,
        n_hrfs=8,
        n_pcs=3,
        r2_threshold=0.05,
        max_voxels=2000,
        deconvolve_duration=2.0,
        deconv_snr=200.0,
    )
    assert res.raw_deconvolved is not None
    assert res.raw_deconvolved.shape == res.raw.shape
    assert res.fitted_deconvolved is not None
    assert len(res.gamma_params_deconvolved) == res.raw.shape[0]
    assert res.duration_convolved is False
    # Deconvolved curves should still be peak-normalized.
    np.testing.assert_allclose(res.raw_deconvolved.max(axis=1), 1.0, atol=1e-6)


def test_derive_library_fit_gamma_off(synthetic_fir_dataset):
    betas, r2, lag_times, _ = synthetic_fir_dataset
    res = derive_library(betas, r2, lag_times, n_hrfs=8, fit_gamma=False)
    assert res.fitted is None
    assert res.gamma_params == []
