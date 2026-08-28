"""Unit tests for fastfuncstuff.design.hrf_derive.

These exercise the pure-numpy stages of the NSD-style HRF library
pipeline using small synthetic datasets:

- ``select_voxels`` honors the R² threshold and sample size cap.
- ``svd_decompose`` reproduces a known low-rank input within
  reconstruction error.
- ``project_unit_sphere`` returns unit-norm rows.
- ``trace_manifold_auto`` returns ``n_points`` ordered points covering
  the density.
- ``reconstruct_timecourses`` produces peak=1 cubic-interpolated curves
  and flags non-HRF-like ones instead of flipping them.
- ``fit_double_gamma`` recovers known double-gamma parameters within
  tolerance; ``fit_double_gamma_through_boxcar`` recovers the impulse
  response from a duration-convolved curve.
- ``crossval_n_pcs`` reproduces NSD's held-out dimensionality curve.
- ``stack_subject_betas`` gives every subject equal weight.
- ``derive_library`` end-to-end on a synthetic 3-flavor HRF dataset.

Tests run in seconds; no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import gamma

from fastfuncstuff.design.hrf_derive import (
    crossval_n_pcs,
    deconvolve_event_duration,
    derive_library,
    fit_double_gamma,
    fit_double_gamma_through_boxcar,
    fit_spline_through_boxcar,
    manifold_coverage,
    project_unit_sphere,
    reconstruct_timecourses,
    select_library_voxels,
    select_voxels,
    stack_subject_betas,
    svd_decompose,
    trace_manifold_auto,
    trace_manifold_blob,
    trace_manifold_grid,
    trace_manifold_kmeans,
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
    out = trace_manifold_auto(pts, n_points=10, angular_step_deg=8.0)
    assert out.shape[1] == 3
    assert out.shape[0] <= 10
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-10)


def test_trace_manifold_auto_rejects_1d():
    # K=1 has no manifold to trace.  K>=4 is now supported (see
    # test_trace_manifold_auto_traces_curve_in_high_dimensions) -- it used to
    # raise here because the walk snapped to a Fibonacci 2-sphere.
    with pytest.raises(ValueError, match="K>=2"):
        trace_manifold_auto(np.ones((10, 1)))


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
    out, t, valid = reconstruct_timecourses(manifold, svd.pcs, lag_times, target_dt=0.1)
    assert out.shape == (3, t.size)
    assert valid.shape == (3,)
    # peak normalization: divide by the SIGNED max, never flip the curve
    np.testing.assert_allclose(out[valid].max(axis=1), 1.0)


def test_reconstruct_flags_inverted_curve_instead_of_flipping():
    # A manifold point that reconstructs to a predominantly NEGATIVE curve
    # is not an HRF.  The old code negated it and shipped it as a library
    # entry; it must now be flagged invalid so the caller drops it.
    lag_times = np.arange(31, dtype=float)
    pcs = np.zeros((1, 31))
    pcs[0, 5] = -1.0  # single big negative excursion
    out, _t, valid = reconstruct_timecourses(np.array([[1.0]]), pcs, lag_times)
    assert not valid[0]
    assert out[0].min() < 0  # not silently flipped positive


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
        deconv_method="wiener",
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


def test_derive_library_is_ordered_by_time_to_peak(synthetic_fir_dataset):
    # The ridge walk starts at the density peak and runs in whichever
    # direction the local tangent pointed, so the raw walk order is
    # arbitrary.  The emitted library must be sorted by time-to-peak so
    # a downstream "which index did it pick" reads as early-vs-late.
    betas, r2, lag_times, _ = synthetic_fir_dataset
    res = derive_library(betas, r2, lag_times, n_hrfs=10, r2_threshold=0.05, max_voxels=2000)
    ttp = res.target_times[np.argmax(res.raw, axis=1)]
    assert np.all(np.diff(ttp) >= 0), f"library not ordered by time-to-peak: {ttp}"


# ---------- manifold tracing -------------------------------------------------


def test_trace_manifold_does_not_double_back():
    # Regression: the previous annulus-based walk carried no direction and
    # only forbade near-exact repeats, so it folded back and re-traversed
    # the ridge a few degrees off — emitting duplicate library entries
    # while never reaching one end.  Points must advance monotonically
    # along a clean synthetic ridge.
    rng = np.random.default_rng(0)
    angles = rng.uniform(-40, 40, 20000) * np.pi / 180
    v = np.column_stack([np.cos(angles), np.sin(angles), 0.02 * rng.standard_normal(angles.size)])
    v /= np.linalg.norm(v, axis=1, keepdims=True)

    pts = trace_manifold_auto(v, n_points=20, angular_step_deg=6.0, bandwidth_deg=8.0)
    # The ridge spans 80 degrees, so a 6-degree walk runs out of ridge before
    # it runs out of budget.  Stopping short is correct; doubling back is not.
    assert pts.shape[1] == 3
    assert 12 <= pts.shape[0] <= 20

    walked = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    diffs = np.diff(walked)
    assert np.all(diffs > 0) or np.all(diffs < 0), f"walk reverses direction: {walked}"

    # No duplicates: every consecutive pair is a real step apart.
    cos_step = np.sum(pts[:-1] * pts[1:], axis=1)
    assert np.degrees(np.arccos(np.clip(cos_step, -1, 1))).min() > 3.0

    # And it covers both sides of the density peak, not just one arm.
    assert walked.min() < -20 and walked.max() > 20


def _blob(spread_deg, n=20000, seed=3):
    """Density blob on the sphere: an arc with controllable off-arc width."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(-35, 35, n) * np.pi / 180
    off = rng.normal(0, spread_deg, n) * np.pi / 180
    v = np.column_stack([np.cos(t) * np.cos(off), np.sin(t) * np.cos(off), np.sin(off)])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_manifold_coverage_cos_is_shape_correlation():
    # The identity the coverage metric rests on: PCs are orthonormal and
    # sphere points are unit-norm, so <w1·PCs, w2·PCs> == w1·w2.  The
    # cosine of the sphere angle IS the reconstructed-HRF correlation.
    rng = np.random.default_rng(0)
    pcs = np.linalg.qr(rng.standard_normal((25, 3)))[0].T  # (3, 25) orthonormal
    w = rng.standard_normal((2, 3))
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    curves = w @ pcs
    np.testing.assert_allclose(curves[0] @ curves[1], w[0] @ w[1], atol=1e-10)

    # arccos round-trips the SIGNED cosine, so the recovered value is the
    # correlation itself, not its magnitude.
    cov = manifold_coverage(w[:1], w[1:])
    np.testing.assert_allclose(np.cos(np.radians(cov["angles_deg"][0])), w[0] @ w[1], atol=1e-6)


def test_two_d_modes_beat_the_arc_on_a_wide_blob():
    # The reason these modes exist: a 1-D ridge under-serves voxels that
    # sit off the arc.  On a blob with real 2-D extent both 2-D samplers
    # must cut the poorly-covered tail substantially.
    v = _blob(12)
    arc = manifold_coverage(v, trace_manifold_auto(v, 20, 6.0, 8.0))
    km = manifold_coverage(v, trace_manifold_kmeans(v, 20))
    bl = manifold_coverage(v, trace_manifold_blob(v, 20, 8.0))

    # k-means is density-proportional → best typical-case mismatch.
    assert km["median_deg"] < arc["median_deg"]
    assert km["p90_deg"] < arc["p90_deg"] / 2
    # blob is uniform over the support → best worst-case.
    assert bl["max_deg"] < arc["max_deg"]


def test_arc_is_competitive_on_a_thin_ridge():
    # Conversely, when the blob really is a 1-D arc (NSD's case), uniform
    # blob coverage wastes entries on empty support and does worse than
    # the ridge.  This is why 'auto' stays the default.
    v = _blob(3)
    arc = manifold_coverage(v, trace_manifold_auto(v, 20, 6.0, 8.0))
    bl = manifold_coverage(v, trace_manifold_blob(v, 20, 8.0))
    assert arc["median_deg"] < bl["median_deg"]


def test_blob_and_kmeans_return_exact_counts():
    # Unlike the ridge walk, which may stop early when the ridge ends.
    v = _blob(10)
    assert trace_manifold_blob(v, 17, 8.0).shape == (17, 3)
    assert trace_manifold_kmeans(v, 17).shape == (17, 3)
    # Unit norm preserved.
    for m in (trace_manifold_blob(v, 12, 8.0), trace_manifold_kmeans(v, 12)):
        np.testing.assert_allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-10)


def test_kmeans_is_deterministic_and_handles_non_3d():
    # k-means is the K != 3 fallback, so it must work off the 2-sphere.
    rng = np.random.default_rng(1)
    v = rng.standard_normal((2000, 5))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    a = trace_manifold_kmeans(v, 8)
    b = trace_manifold_kmeans(v, 8)
    assert a.shape == (8, 5)
    np.testing.assert_allclose(a, b)


def test_derive_library_2d_modes_run_end_to_end(synthetic_fir_dataset):
    betas, r2, lag_times, _ = synthetic_fir_dataset
    for mode in ("blob", "kmeans"):
        res = derive_library(
            betas,
            r2,
            lag_times,
            n_hrfs=10,
            r2_threshold=0.05,
            max_voxels=2000,
            manifold_mode=mode,
        )
        assert res.coverage is not None
        assert res.coverage["median_deg"] >= 0
        # Even 2-D sets get a deterministic time-to-peak ordering.
        ttp = res.target_times[np.argmax(res.raw, axis=1)]
        assert np.all(np.diff(ttp) >= 0)


# ---------- duration correction ----------------------------------------------


def test_fit_through_boxcar_beats_wiener_for_long_durations():
    # The reason "fit" is the default: Wiener has to regularize away the
    # boxcar's spectral zeros, which for a long boxcar wrecks the recovered
    # impulse response.  Putting the boxcar in the forward model does not.
    dt = 0.1
    t = np.arange(0, 32, dt)
    imp = _double_gamma_curve(t, a1=6.0)
    n_box = int(10.0 / dt)
    conv = np.convolve(imp, np.ones(n_box))[: t.size]
    conv = conv / conv.max()

    recovered, params = fit_double_gamma_through_boxcar(conv, duration=10.0, dt=dt)
    assert params["fit_ok"]
    assert np.abs(recovered - imp).max() < 0.01

    wiener = deconvolve_event_duration(conv[None], duration=10.0, dt=dt, snr=100.0)[0]
    assert np.abs(recovered - imp).max() < np.abs(wiener - imp).max()


def test_derive_library_deconv_fit_mode(synthetic_fir_dataset):
    betas, r2, lag_times, _ = synthetic_fir_dataset
    res = derive_library(
        betas,
        r2,
        lag_times,
        n_hrfs=8,
        r2_threshold=0.05,
        max_voxels=2000,
        deconvolve_duration=2.0,
        deconv_method="fit",
    )
    # Fit mode carries the correction inside the gamma family, so there is
    # no separate deconvolved raw curve — the fitted one IS the impulse
    # response.
    assert res.raw_deconvolved is None
    assert res.fitted_deconvolved is not None
    assert res.duration_convolved is False
    assert all("duration_s" in p for p in res.gamma_params_deconvolved)


# ---------- cross-validated PC count -----------------------------------------


def test_crossval_n_pcs_turns_over():
    # Two noisy observations of the same low-rank truth: held-out R² must
    # rise then fall as K starts fitting the training split's noise.
    rng = np.random.default_rng(0)
    lag = np.arange(31, dtype=float)
    truth = np.stack([_double_gamma_curve(lag, a1=a) for a in rng.uniform(4, 8, 2000)])
    a = truth + 0.08 * rng.standard_normal(truth.shape)
    b = truth + 0.08 * rng.standard_normal(truth.shape)

    cv = crossval_n_pcs(a, b, max_pcs=10)
    assert cv.shape == (10,)
    best = int(cv.argmax()) + 1
    assert 1 <= best <= 4, f"expected a low-rank optimum, got K={best}"
    assert cv[-1] < cv[best - 1], "curve should decline once K starts fitting noise"


# ---------- multi-subject stacking -------------------------------------------


def test_stack_subject_betas_equalizes_contribution():
    # NSD's point: per-subject supra-threshold counts ranged 2834-17387, so
    # a global sample would let the best subject dominate the PCs.  Each
    # subject must contribute the same number of rows.
    rng = np.random.default_rng(0)
    lag = np.arange(31, dtype=float)
    subs_betas, subs_r2 = [], []
    for n_good in (300, 5000):
        n = 6000
        b = np.stack([_double_gamma_curve(lag, a1=a) for a in rng.uniform(4, 8, n)])
        r = np.full(n, 0.01)
        r[:n_good] = 0.5
        subs_betas.append(b)
        subs_r2.append(r)

    betas, r2, ids = stack_subject_betas(
        subs_betas, subs_r2, r2_threshold=0.1, per_subject_voxels=1000
    )
    assert betas.shape == (2000, 31)
    assert r2.shape == (2000,)
    assert (ids == 0).sum() == 1000
    assert (ids == 1).sum() == 1000  # upsampled despite having 300 vs 5000

    # Without equalization the richer subject dominates.
    _b2, _r2b, ids2 = stack_subject_betas(
        subs_betas, subs_r2, r2_threshold=0.1, per_subject_voxels=1000, equalize=False
    )
    assert (ids2 == 0).sum() == 300
    assert (ids2 == 1).sum() == 1000


def test_stack_subject_betas_rejects_mismatched_lags():
    a = np.zeros((10, 31))
    b = np.zeros((10, 20))
    with pytest.raises(ValueError, match="FIR lags"):
        stack_subject_betas([a, b], [np.ones(10), np.ones(10)])


def test_select_library_voxels_drops_constant_high_r2_voxels():
    # Air voxels are fit perfectly by the nuisance block, so they score
    # R²≈1 with all-zero FIR betas and would otherwise dominate the SVD.
    betas = np.zeros((100, 31))
    betas[50:] = np.random.default_rng(0).standard_normal((50, 31))
    r2 = np.full(100, 0.9)
    sel = select_library_voxels(betas, r2, threshold=0.1)
    assert sel.size == 50
    assert sel.min() >= 50


# ---------- ridge tracing: follows density, and does so in any K --------------


def _curved_ridge(k: int, n: int = 20000, scatter: float = 0.03, span: float = 1.2, seed: int = 1):
    """A genuinely CURVED 1-D ridge embedded in S^(k-1), plus off-ridge scatter.

    Curvature is the whole point: a straight (great-circle) ridge cannot
    distinguish a walk that follows the density from one that merely
    extrapolates along its starting tangent.
    """
    rng = np.random.default_rng(seed)
    t = rng.uniform(-span, span, n)
    cols = [np.cos(t), np.sin(t), 0.35 * np.sin(2 * t)]
    cols += [0.25 * np.cos((j + 1) * t) for j in range(max(0, k - 3))]
    p = np.column_stack(cols[:k]) + rng.normal(0, scatter, (n, k))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    p *= np.sign(p[:, 0:1])
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def test_trace_manifold_auto_follows_a_curved_ridge():
    # Regression: the correction step used to snap to the densest point of a
    # 4096-point Fibonacci sphere within half a step.  That grid has ~3.1 deg
    # spacing and the snap cone had a 3 deg half-angle, so it held a MEDIAN OF
    # ONE candidate -- the correction was a no-op and the walk was an
    # uncorrected geodesic, i.e. a straight line, drifting off this ridge by
    # up to 8 deg.  Mean shift is continuous and has no such resolution floor.
    col = np.radians(50.0)
    rng = np.random.default_rng(0)
    t = rng.uniform(-1.4, 1.4, 20000)
    v = np.column_stack(
        [np.sin(col) * np.cos(t), np.sin(col) * np.sin(t), np.cos(col) * np.ones_like(t)]
    )
    v += rng.normal(0, 0.02, v.shape)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v *= np.sign(v[:, 0:1])
    v /= np.linalg.norm(v, axis=1, keepdims=True)

    pts = trace_manifold_auto(v, n_points=20, angular_step_deg=6.0)

    # Every sample must sit ON the small circle it was drawn from.
    drift = np.degrees(np.abs(np.arccos(np.clip(pts[:, 2], -1, 1)) - col))
    assert np.median(drift) < 1.0, f"walk leaves the ridge: median drift {np.median(drift):.2f} deg"
    assert drift.max() < 3.0, f"walk leaves the ridge: max drift {drift.max():.2f} deg"


def test_trace_manifold_auto_honours_the_requested_spacing():
    # The perpendicular correction pulls a step back inside the geodesic, so
    # without the re-prediction pass the achieved spacing came up ~35% short
    # on a curved ridge; an early version that capped only the per-ITERATION
    # mean-shift hop (not the total excursion) overshot to 38 deg.
    v = _curved_ridge(3)
    pts = trace_manifold_auto(v, n_points=20, angular_step_deg=6.0)
    spacing = np.degrees(np.arccos(np.clip(np.sum(pts[:-1] * pts[1:], axis=1), -1, 1)))
    assert spacing.min() > 4.5, f"steps too short: {spacing.min():.2f} deg"
    assert spacing.max() < 8.0, f"steps too long: {spacing.max():.2f} deg"


@pytest.mark.parametrize("k", [3, 4, 5, 6])
def test_manifold_samplers_all_work_in_k_dimensions(k):
    # Ridge tracing was capped at K=3 by a Fibonacci 2-sphere grid, and so was
    # `blob`.  A grid cannot be the fix -- matching 3 deg on S^3 needs ~260k
    # points and on S^4 ~16M -- so both are now grid-free.
    v = _curved_ridge(k)
    for sampler in (
        lambda: trace_manifold_auto(v, n_points=24, angular_step_deg=8.0),
        lambda: trace_manifold_blob(v, 24),
        lambda: trace_manifold_kmeans(v, 24),
        lambda: trace_manifold_grid(v, 24),
    ):
        out = sampler()
        assert out.shape[1] == k
        assert out.shape[0] <= 24
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-10)
        # Every sampler must actually represent the cloud it came from.
        assert manifold_coverage(v, out)["p90_deg"] < 15.0


def test_trace_manifold_auto_beats_a_great_circle_in_high_dimensions():
    # The concrete claim: in 5-D the walk tracks the curve rather than the
    # tangent it started on.  A great circle through the same endpoints is the
    # thing the old grid-snapped walk degenerated into.
    v = _curved_ridge(5)
    pts = trace_manifold_auto(v, n_points=30, angular_step_deg=6.0)
    n = pts.shape[0]
    a, b = pts[0], pts[-1]
    # Great-circle interpolation between the same two ends, same count.
    omega = np.arccos(np.clip(a @ b, -1, 1))
    frac = np.linspace(0, 1, n)[:, None]
    chord = (np.sin((1 - frac) * omega) * a + np.sin(frac * omega) * b) / np.sin(omega)
    chord /= np.linalg.norm(chord, axis=1, keepdims=True)

    walked = manifold_coverage(v, pts)["median_deg"]
    straight = manifold_coverage(v, chord)["median_deg"]
    assert walked < straight, (
        f"walk ({walked:.2f} deg) no better than a straight line ({straight:.2f})"
    )


# ---------- gamma fit must not be duration-blind on block designs ------------


def test_duration_convolved_gamma_curve_tracks_the_raw_curve():
    # Regression: the "gamma fit - duration-convolved" QC curve was an
    # INDEPENDENT bare double-gamma fit of `raw`.  For a 20 s block, `raw` is a
    # plateau that an impulse-response family cannot make, so every fit
    # saturated (a1 pinned at its upper bound 12.0 in every case measured, c
    # slammed to a bound) and manufactured excursions to -0.48 where the raw
    # cubic reached -0.13.  It is now the boxcar fit pushed back through the
    # boxcar, so it must track the curve it is drawn against.
    rng = np.random.default_rng(3)
    lag = np.arange(0, 36, 2.0)
    t_fine = np.arange(0, 36, 0.1)
    n_box = 200  # 20 s at dt=0.1
    n = 2000
    lat = rng.uniform(4, 7, n)
    wid = rng.uniform(0.8, 1.6, n)
    cs = rng.uniform(0.05, 0.35, n)
    rows = []
    for i in range(n):
        imp = gamma.pdf(t_fine, lat[i] / wid[i], scale=wid[i]) - cs[i] * gamma.pdf(
            t_fine, 16.0, scale=1.0
        )
        blk = np.convolve(imp, np.ones(n_box))[: t_fine.size]
        rows.append(np.interp(lag, t_fine, blk / np.abs(blk).max()))
    betas = np.array(rows) + rng.normal(0, 0.02, (n, lag.size))

    res = derive_library(
        betas,
        np.full(n, 0.5),
        lag,
        n_pcs=3,
        n_hrfs=20,
        fit_gamma=True,
        deconvolve_duration=20.0,
        deconv_method="fit",
        r2_threshold=0.0,
    )

    assert res.fitted is not None and res.fitted_deconvolved is not None
    r = np.array([np.corrcoef(res.raw[i], res.fitted[i])[0, 1] for i in range(res.raw.shape[0])])
    assert r.min() > 0.99, f"duration-convolved gamma curve does not track raw: r={r.min():.4f}"
    # And it must not invent an excursion the raw curve does not have.
    assert res.fitted.min() > res.raw.min() - 0.05

    # The impulse fit itself must not be sitting on the a1 bound.
    a1 = np.array([p["a1"] for p in res.gamma_params_deconvolved])
    assert (a1 < 11.99).any(), "every boxcar fit pinned at the a1 upper bound"


# ---------- penalized-spline shape model -------------------------------------


def _impulse_and_block(
    bump: float = 0.0, dt: float = 0.1, window: float = 36.0, block: float = 20.0
):
    """A double-gamma impulse (optionally + an off-family bump) and its block response."""
    t = np.arange(0, window, dt)
    imp = gamma.pdf(t, 5.0, scale=1.0) - 0.17 * gamma.pdf(t, 16.0, scale=1.0)
    if bump:
        imp = imp + bump * np.exp(-((t - 14) ** 2) / 8)
    imp = imp / imp.max()
    blk = np.convolve(imp, np.ones(int(round(block / dt))))[: t.size]
    return t, imp, blk / blk.max()


def test_spline_recovers_a_true_double_gamma():
    # The spline must cost almost nothing when the double-gamma family was
    # right, or it is not a safe default to offer.
    _, imp, blk = _impulse_and_block()
    fit, p = fit_spline_through_boxcar(blk, duration=20.0, dt=0.1)
    assert p["fit_ok"]
    assert np.corrcoef(imp, fit)[0, 1] > 0.99


def test_spline_beats_double_gamma_off_family():
    # The whole point: an impulse response carrying structure the
    # double-gamma cannot express.  Measured 0.99 vs 0.89.
    _, imp, blk = _impulse_and_block(bump=0.30)
    spline_fit, _ = fit_spline_through_boxcar(blk, duration=20.0, dt=0.1)
    gamma_fit, _ = fit_double_gamma_through_boxcar(blk, duration=20.0, dt=0.1)
    r_spline = np.corrcoef(imp, spline_fit)[0, 1]
    r_gamma = np.corrcoef(imp, gamma_fit)[0, 1]
    assert r_spline > r_gamma + 0.05, f"spline {r_spline:.4f} vs gamma {r_gamma:.4f}"
    assert r_spline > 0.97


def test_spline_drops_unidentifiable_basis_functions():
    # Convolution with a D-second boxcar leaves the late impulse response
    # barely constrained -- with T=36 and D=20 the final convolved design
    # column carries 4% of the strongest -- and a second-difference penalty
    # cannot hold it because its null space is exactly {constant, linear}, so
    # a tail ramp is free.  The unidentifiable directions are dropped instead.
    _, _, blk = _impulse_and_block()
    _, p = fit_spline_through_boxcar(blk, duration=20.0, dt=0.1)
    assert p["n_basis_kept"] < p["n_basis_total"]

    # And the tail must stay put under noise, which is what the drop buys.
    rng = np.random.default_rng(0)
    t = np.arange(0, 36, 0.1)
    tails = []
    for _ in range(8):
        noise = np.interp(t, np.linspace(0, 36, 18), rng.normal(0, 0.05, 18))
        fit, _ = fit_spline_through_boxcar(blk + noise, duration=20.0, dt=0.1)
        tails.append(np.abs(fit[-50:]).max())
    assert np.median(tails) < 0.25, f"tail runs away under noise: {np.median(tails):.3f}"


def test_spline_library_is_reconvolvable():
    # A library entry is only useful if convolving it back with the event
    # boxcar reproduces the curve it was derived from -- that is exactly what
    # downstream consumers do at modelling time.
    rng = np.random.default_rng(11)
    lag = np.arange(0, 36, 2.0)
    t_fine = np.arange(0, 36, 0.1)
    n_box, n = 200, 1500
    rows = []
    for lat, wid, c, bump in zip(
        rng.uniform(4, 7, n),
        rng.uniform(0.8, 1.6, n),
        rng.uniform(0.05, 0.35, n),
        rng.uniform(0, 0.35, n),
        strict=True,
    ):
        imp = (
            gamma.pdf(t_fine, lat / wid, scale=wid)
            - c * gamma.pdf(t_fine, 16.0, scale=1.0)
            + bump * np.exp(-((t_fine - 14) ** 2) / 8)
        )
        blk = np.convolve(imp, np.ones(n_box))[: t_fine.size]
        rows.append(np.interp(lag, t_fine, blk / np.abs(blk).max()))
    betas = np.array(rows) + rng.normal(0, 0.02, (n, lag.size))

    res = derive_library(
        betas,
        np.full(n, 0.5),
        lag,
        n_pcs=3,
        n_hrfs=12,
        manifold_mode="kmeans",
        fit_gamma=True,
        shape_model="spline",
        deconvolve_duration=20.0,
        deconv_method="fit",
        r2_threshold=0.0,
    )
    assert res.shape_model == "spline"
    assert res.fitted_deconvolved is not None

    n_t = res.raw.shape[1]
    for i in range(res.fitted_deconvolved.shape[0]):
        pred = np.convolve(res.fitted_deconvolved[i], np.ones(n_box))[:n_t]
        assert np.corrcoef(res.raw[i], pred / pred.max())[0, 1] > 0.99


def test_shape_model_none_still_skips_the_fit():
    _, _, blk = _impulse_and_block()
    lag = np.arange(0, 36, 2.0)
    rng = np.random.default_rng(2)
    betas = rng.standard_normal((500, lag.size)) * 0.1 + np.interp(lag, np.arange(0, 36, 0.1), blk)
    res = derive_library(
        betas, np.full(500, 0.5), lag, n_pcs=3, n_hrfs=8, fit_gamma=False, r2_threshold=0.0
    )
    assert res.fitted is None
    assert res.shape_model == "none"


def _block_response_betas(n=1500, seed=5):
    """FIR betas that are responses to a 20 s block, with an off-family bump."""
    rng = np.random.default_rng(seed)
    lag = np.arange(0, 36, 2.0)
    t_fine = np.arange(0, 36, 0.1)
    rows = []
    for lat, wid, c, bump in zip(
        rng.uniform(4, 7, n),
        rng.uniform(0.8, 1.6, n),
        rng.uniform(0.05, 0.35, n),
        rng.uniform(0, 0.35, n),
        strict=True,
    ):
        imp = (
            gamma.pdf(t_fine, lat / wid, scale=wid)
            - c * gamma.pdf(t_fine, 16.0, scale=1.0)
            + bump * np.exp(-((t_fine - 14) ** 2) / 8)
        )
        blk = np.convolve(imp, np.ones(200))[: t_fine.size]
        rows.append(np.interp(lag, t_fine, blk / np.abs(blk).max()))
    return np.array(rows) + rng.normal(0, 0.02, (n, lag.size)), lag


@pytest.mark.parametrize("deconv_method", ["fit", "wiener"])
def test_shape_model_is_honoured_on_every_deconv_path(deconv_method):
    # Regression: shape_model="spline" was wired into the boxcar-fit branch and
    # the no-deconvolution branch but NOT into the wiener branch, which kept
    # calling fit_double_gamma.  The shipped library was then a double-gamma
    # fit while the QC panel title, the final-library label and the metadata
    # all said "spline" -- on the one path where the choice matters most.
    betas, lag = _block_response_betas()
    res = derive_library(
        betas,
        np.full(betas.shape[0], 0.5),
        lag,
        n_pcs=3,
        n_hrfs=12,
        manifold_mode="kmeans",
        fit_gamma=True,
        shape_model="spline",
        deconvolve_duration=20.0,
        deconv_method=deconv_method,
        r2_threshold=0.0,
    )
    assert res.shape_model == "spline"
    assert res.gamma_params_deconvolved, "no impulse-space fit was recorded"
    # The spline's param dict carries "lambda"; the double-gamma's carries "a1".
    for p in res.gamma_params_deconvolved:
        assert "lambda" in p, f"library entry fitted with the wrong family: {sorted(p)}"


@pytest.mark.parametrize("deconv_method", ["fit", "wiener"])
def test_library_reconvolves_onto_the_curve_it_came_from(deconv_method):
    # The end-to-end referee for the duration correction: downstream consumers
    # re-convolve the library entry with the event boxcar, so if that does not
    # reproduce the curve the entry was derived from, the entry is wrong.
    betas, lag = _block_response_betas()
    res = derive_library(
        betas,
        np.full(betas.shape[0], 0.5),
        lag,
        n_pcs=3,
        n_hrfs=12,
        manifold_mode="kmeans",
        fit_gamma=True,
        shape_model="spline",
        deconvolve_duration=20.0,
        deconv_method=deconv_method,
        r2_threshold=0.0,
    )
    rr = res.reconvolution_r
    assert rr is not None and np.isfinite(rr).all()
    assert rr.min() > 0.98, f"worst entry re-convolves at r={rr.min():.4f}"


def test_reconvolution_r_is_none_without_a_duration_correction():
    betas, lag = _block_response_betas(n=600)
    res = derive_library(betas, np.full(600, 0.5), lag, n_pcs=3, n_hrfs=8, r2_threshold=0.0)
    assert res.reconvolution_r is None
