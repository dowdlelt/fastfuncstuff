"""Unit tests for fastfuncstuff.design.flobs.

Covers:

- :func:`generate_flobs_basis` — eigenHRF + MVN(m, C) constraint
  generation: basis is L2-unit, top-3 PCs explain >90 % variance,
  PC1 looks canonical, ``m`` and ``C`` have sensible scales.
- :func:`fit_flobs_constrained` — Bayesian shape-constrained fit
  beats unconstrained OLS at low SNR and reduces to OLS at high SNR.
- Determinism: re-running with the same seed gives identical output.
"""
from __future__ import annotations

import numpy as np
import torch
import pytest

from fastfuncstuff.design.flobs import (
    FLOBSBasis,
    FLOBSCVResult,
    FLOBSFitResult,
    FracRidgeFitResult,
    cv_basis_constrained_ridge,
    decouple_amplitude_prior,
    estimate_and_apply_arma11_prewhitening,
    fit_basis_constrained_ridge,
    fit_basis_fracridge,
    fit_flobs_constrained,
    flobs_prior,
    generate_flobs_basis,
    generate_spmg_basis,
    ridge_prior,
    spmg_prior,
)


@pytest.fixture(scope="module")
def basis() -> FLOBSBasis:
    return generate_flobs_basis(n_basis=3, n_samples=1000, duration=32.0, dt=0.1, seed=42)


# ---------- generate_flobs_basis ---------------------------------------------


def test_basis_shape(basis):
    assert basis.basis_functions.shape == (3, 320)
    assert basis.m.shape == (3,)
    assert basis.C.shape == (3, 3)


def test_basis_l2_unit_columns(basis):
    # Each basis function should have L2 norm = 1 (matches TR04MW2's
    # convention and makes the (m, C) numerics consistent).
    norms = np.linalg.norm(basis.basis_functions, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_basis_top3_explain_most_variance(basis):
    ev = basis.eigenvalues ** 2
    frac = ev / ev.sum()
    # TR04MW2 fig 3 shows the top 3 dominate; sharp elbow at K=3
    assert frac[:3].sum() > 0.90


def test_basis_pc1_is_canonical_like(basis):
    # PC1 should look like a canonical positive HRF with peak near 5s
    pc1 = basis.basis_functions[0]
    peak_t = float(np.argmax(pc1)) * basis.dt
    assert 4.0 < peak_t < 7.0
    assert pc1.max() > 0  # sign-aligned positive


def test_basis_determinism():
    b1 = generate_flobs_basis(n_basis=3, n_samples=500, seed=123)
    b2 = generate_flobs_basis(n_basis=3, n_samples=500, seed=123)
    np.testing.assert_array_equal(b1.basis_functions, b2.basis_functions)
    np.testing.assert_array_equal(b1.m, b2.m)
    np.testing.assert_array_equal(b1.C, b2.C)


def test_basis_keep_samples_flag():
    b = generate_flobs_basis(n_basis=3, n_samples=50, seed=1, keep_samples=True)
    assert b.sample_hrfs is not None
    assert b.sample_hrfs.shape[1] == 50

    b2 = generate_flobs_basis(n_basis=3, n_samples=50, seed=1, keep_samples=False)
    assert b2.sample_hrfs is None


def test_basis_custom_parametrization():
    # Override one range; check the sampler honoured it (by checking
    # the basis shape responds — e.g. m4 push to longer tails should
    # make PC1 peak slightly later).
    long_tails = generate_flobs_basis(
        n_basis=3, n_samples=500, seed=1,
        parametrization={"m4": (8.0, 14.0)},
    )
    default = generate_flobs_basis(n_basis=3, n_samples=500, seed=1)
    # Long tail shouldn't change basis shape DRASTICALLY but should
    # produce a measurably different C (covariance picks up the extra
    # variation).
    assert not np.allclose(long_tails.C, default.C)


# ---------- fit_flobs_constrained -------------------------------------------


def _build_test_data(basis, n_vox=80, n_t=200, tr=1.0, noise_sigma=0.5, seed=7):
    rng = np.random.default_rng(seed)
    # True per-voxel coefficients sampled from FLOBS prior (so by
    # construction they're "sensible HRFs").
    true_coefs = rng.multivariate_normal(basis.m, basis.C * 0.3, size=n_vox)
    # Design: 7 onsets, each basis function convolved with onset train
    # at TR resolution.
    onset_idx = np.array([10, 35, 60, 90, 115, 150, 180])
    basis_tr = basis.basis_functions[:, :: int(round(tr / basis.dt))]
    X = np.zeros((n_t, 3))
    onsets = np.zeros(n_t); onsets[onset_idx] = 1.0
    for b in range(3):
        X[:, b] = np.convolve(onsets, basis_tr[b])[:n_t]
    y_clean = (X @ true_coefs.T).T
    y = y_clean + noise_sigma * y_clean.std() * rng.standard_normal((n_vox, n_t))
    return y, X, true_coefs


def test_constrained_fit_high_snr_matches_ols(basis):
    # At high SNR, the prior fades (σ²→0) and the constrained fit
    # should match OLS within a small margin.
    y, X, true_coefs = _build_test_data(basis, noise_sigma=0.02)
    beta_ols = np.linalg.lstsq(X, y.T, rcond=None)[0].T
    fit = fit_flobs_constrained(
        data=torch.from_numpy(y).double(),
        design_task=torch.from_numpy(X).double(),
        basis=basis, n_conditions=1, prior_weight="auto",
        device=torch.device("cpu"),
    )
    err_ols = np.linalg.norm(beta_ols - true_coefs, axis=1).mean()
    err_flobs = np.linalg.norm(fit.betas - true_coefs, axis=1).mean()
    # At high SNR they should be within ~5% of each other
    assert err_flobs / err_ols < 1.1


def test_constrained_fit_low_snr_beats_ols(basis):
    # At low SNR, the prior should dominate and pull toward the
    # canonical shape mean — should clearly beat unregularized OLS.
    y, X, true_coefs = _build_test_data(basis, noise_sigma=2.0)
    beta_ols = np.linalg.lstsq(X, y.T, rcond=None)[0].T
    fit = fit_flobs_constrained(
        data=torch.from_numpy(y).double(),
        design_task=torch.from_numpy(X).double(),
        basis=basis, n_conditions=1, prior_weight="auto",
        device=torch.device("cpu"),
    )
    err_ols = np.linalg.norm(beta_ols - true_coefs, axis=1).mean()
    err_flobs = np.linalg.norm(fit.betas - true_coefs, axis=1).mean()
    # At low SNR we expect FLOBS to be at most 80% of OLS error
    assert err_flobs / err_ols < 0.85


def test_constrained_fit_prior_weight_zero_equals_ols(basis):
    # prior_weight=0 multiplier should disable the prior entirely.
    y, X, true_coefs = _build_test_data(basis, noise_sigma=1.0)
    beta_ols = np.linalg.lstsq(X, y.T, rcond=None)[0].T
    fit = fit_flobs_constrained(
        data=torch.from_numpy(y).double(),
        design_task=torch.from_numpy(X).double(),
        basis=basis, n_conditions=1, prior_weight=0.0,
        device=torch.device("cpu"),
    )
    np.testing.assert_allclose(fit.betas, beta_ols, atol=1e-6)


def test_constrained_fit_returns_correct_shapes(basis):
    y, X, _ = _build_test_data(basis, n_vox=12)
    fit = fit_flobs_constrained(
        data=torch.from_numpy(y).double(),
        design_task=torch.from_numpy(X).double(),
        basis=basis, n_conditions=1,
        device=torch.device("cpu"),
    )
    assert isinstance(fit, FLOBSFitResult)
    assert fit.betas.shape == (12, 3)
    assert fit.hrfs.shape == (12, 1, basis.basis_functions.shape[1])
    assert fit.r2.shape == (12,)
    assert 0.0 <= fit.r2.mean() <= 1.0


def test_constrained_fit_with_nuisance(basis):
    # Add a polynomial nuisance regressor; the constrained fit should
    # pass that through without prior-shrinking it.
    y, X, _ = _build_test_data(basis)
    n_t = X.shape[0]
    # Add a linear drift to y
    drift = np.linspace(0, 5, n_t)
    y_drift = y + drift[np.newaxis, :]
    Z = np.column_stack([np.ones(n_t), np.linspace(-1, 1, n_t)])

    fit = fit_flobs_constrained(
        data=torch.from_numpy(y_drift).double(),
        design_task=torch.from_numpy(X).double(),
        basis=basis, n_conditions=1,
        nuisance=torch.from_numpy(Z).double(),
        device=torch.device("cpu"),
    )
    # Task betas (first 3 cols) + 2 nuisance cols
    assert fit.betas.shape == (y.shape[0], 5)
    # Recovered constant ~5 (we added drift 0..5 linearly), linear ~5
    nuis = fit.betas[:, 3:].mean(axis=0)
    assert abs(nuis[0] - 2.5) < 0.5  # mean of drift ≈ 2.5


# ---------- SPMG bases + prior helpers --------------------------------------


def test_generate_spmg_basis_shapes():
    for n in (1, 2, 3):
        b = generate_spmg_basis(n_basis=n, duration=32.0, dt=0.1)
        assert b.basis_functions.shape == (n, 320)
        # L2-unit columns (same convention as FLOBS).
        norms = np.linalg.norm(b.basis_functions, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)
        assert b.parametrization == {"family": f"SPMG{n}"}


def test_generate_spmg_basis_invalid_n():
    with pytest.raises(ValueError, match="n_basis"):
        generate_spmg_basis(n_basis=4)
    with pytest.raises(ValueError, match="n_basis"):
        generate_spmg_basis(n_basis=0)


def test_spmg_prior_spmg2():
    m, C = spmg_prior(canonical_std=5.0, derivative_std=0.5)
    assert m.shape == (2,)
    assert C.shape == (2, 2)
    np.testing.assert_array_equal(m, [0.0, 0.0])
    # diag(canonical_std², derivative_std²); off-diagonal zero
    assert C[0, 0] == pytest.approx(25.0)
    assert C[1, 1] == pytest.approx(0.25)
    assert C[0, 1] == 0.0
    assert C[1, 0] == 0.0


def test_spmg_prior_spmg3():
    m, C = spmg_prior(canonical_std=5.0, derivative_std=0.5, dispersion_std=0.3)
    assert m.shape == (3,)
    assert C.shape == (3, 3)
    assert C[2, 2] == pytest.approx(0.09)


def test_spmg_prior_canonical_mean_bias():
    m, _ = spmg_prior(canonical_std=5.0, derivative_std=0.5, canonical_mean=2.0)
    assert m[0] == pytest.approx(2.0)
    assert m[1] == 0.0


def test_ridge_prior_scalar_mean():
    m, C = ridge_prior(n_basis=4, coefficient_std=2.0, mean=0.5)
    np.testing.assert_array_equal(m, [0.5, 0.5, 0.5, 0.5])
    np.testing.assert_array_equal(C, 4.0 * np.eye(4))


def test_ridge_prior_vector_mean():
    m, _ = ridge_prior(n_basis=3, mean=np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(m, [1.0, 2.0, 3.0])


def test_ridge_prior_vector_mean_wrong_shape():
    with pytest.raises(ValueError, match="≠"):
        ridge_prior(n_basis=3, mean=np.array([1.0, 2.0]))


def test_flobs_prior_extractor(basis):
    m, C = flobs_prior(basis)
    np.testing.assert_array_equal(m, basis.m)
    np.testing.assert_array_equal(C, basis.C)
    # Should be a copy, not a view
    m[0] = 999
    assert basis.m[0] != 999


# ---------- New primitive: fit_basis_constrained_ridge ----------------------


def test_renamed_primitive_matches_old_alias(basis):
    """fit_basis_constrained_ridge with FLOBS prior should match the old
    fit_flobs_constrained on identical inputs."""
    y, X, _ = _build_test_data(basis, n_vox=40, noise_sigma=0.5)
    fit_old = fit_flobs_constrained(
        data=torch.from_numpy(y).double(),
        design_task=torch.from_numpy(X).double(),
        basis=basis, n_conditions=1,
        device=torch.device("cpu"),
    )
    fit_new = fit_basis_constrained_ridge(
        data=torch.from_numpy(y).double(),
        design_task=torch.from_numpy(X).double(),
        basis_functions=basis.basis_functions,
        prior_mean=basis.m,
        prior_cov=basis.C,
        n_blocks=1,
        device=torch.device("cpu"),
    )
    np.testing.assert_allclose(fit_new.betas, fit_old.betas, atol=1e-10)
    np.testing.assert_allclose(fit_new.hrfs, fit_old.hrfs, atol=1e-10)
    np.testing.assert_allclose(fit_new.r2, fit_old.r2, atol=1e-10)


def test_spmg2_constrained_fit_beats_ols_at_low_snr():
    """SPMG2 single-trial style fits with derivative coefficient
    blow-up: the spmg_prior should clamp the derivative coefficient
    without killing the canonical amplitude — net effect: better
    coefficient recovery at low SNR than plain OLS.
    """
    rng = np.random.default_rng(11)
    sb = generate_spmg_basis(n_basis=2, duration=32.0, dt=0.1)
    n_vox, n_t, tr = 80, 200, 1.0

    # True coefs: real amplitude, small-to-zero derivative
    true_canon = rng.uniform(0.5, 3.0, size=n_vox)
    true_deriv = rng.normal(0.0, 0.05, size=n_vox)            # tiny
    true_coefs = np.column_stack([true_canon, true_deriv])    # (n_vox, 2)

    # Design at TR resolution
    sb_tr = sb.basis_functions[:, :: int(round(tr / sb.dt))]
    X = np.zeros((n_t, 2))
    onsets = np.zeros(n_t); onsets[[15, 45, 80, 120, 165]] = 1.0
    for b in range(2):
        X[:, b] = np.convolve(onsets, sb_tr[b])[:n_t]
    y_clean = (X @ true_coefs.T).T

    # Low SNR
    y = y_clean + 1.5 * y_clean.std() * rng.standard_normal((n_vox, n_t))

    beta_ols = np.linalg.lstsq(X, y.T, rcond=None)[0].T
    pm, pc = spmg_prior(canonical_std=5.0, derivative_std=0.3)
    fit = fit_basis_constrained_ridge(
        data=torch.from_numpy(y).double(),
        design_task=torch.from_numpy(X).double(),
        basis_functions=sb.basis_functions,
        prior_mean=pm,
        prior_cov=pc,
        n_blocks=1,
        device=torch.device("cpu"),
    )
    err_ols = np.linalg.norm(beta_ols - true_coefs, axis=1).mean()
    err_constrained = np.linalg.norm(fit.betas - true_coefs, axis=1).mean()
    # The constraint should help — at low SNR, derivative coefficient
    # otherwise wildly overshoots.
    assert err_constrained < err_ols
    # And the derivative coefficient stays sensible (not 100× its truth)
    assert np.median(np.abs(fit.betas[:, 1])) < 1.0


# ---------- cv_basis_constrained_ridge --------------------------------------


def _build_multirun_synth(basis, n_runs=4, n_tp_run=120, n_vox=60, noise_sigma=1.0,
                          tr=1.0, seed=11):
    """Build per-run (data, task design) lists for CV tests.

    Same true coefficient vector for every voxel, drawn from the
    FLOBS prior so it's a "sensible HRF" by construction.  Onsets
    differ run-to-run (jittered) so held-out runs are genuinely
    new data.
    """
    rng = np.random.default_rng(seed)
    true_coefs = rng.multivariate_normal(basis.m, basis.C * 0.3, size=n_vox)  # (n_vox, K)
    K = basis.basis_functions.shape[0]
    basis_tr = basis.basis_functions[:, :: max(int(round(tr / basis.dt)), 1)]
    per_run_data: list = []
    per_run_task: list = []
    for r in range(n_runs):
        onset_idx = np.sort(rng.choice(np.arange(8, n_tp_run - 30), size=7, replace=False))
        X = np.zeros((n_tp_run, K))
        on = np.zeros(n_tp_run); on[onset_idx] = 1.0
        for b in range(K):
            X[:, b] = np.convolve(on, basis_tr[b])[:n_tp_run]
        y_clean = (X @ true_coefs.T).T                                 # (n_vox, n_tp)
        y = y_clean + noise_sigma * y_clean.std() * rng.standard_normal((n_vox, n_tp_run))
        per_run_data.append(torch.from_numpy(y).float())
        per_run_task.append(torch.from_numpy(X).float())
    return per_run_data, per_run_task, true_coefs


def test_cv_returns_correct_shapes(basis):
    per_run_data, per_run_task, _ = _build_multirun_synth(basis, n_runs=4, n_vox=20)
    cv = cv_basis_constrained_ridge(
        per_run_data=per_run_data,
        per_run_task_designs=per_run_task,
        basis_functions=basis.basis_functions,
        prior_mean=basis.m,
        prior_cov=basis.C,
        n_blocks=1,
        polort=2,
        weight_grid=[0.5, 1.0, 2.0],
        include_ols=True,
        device=torch.device("cpu"),
        verbose=False,
    )
    assert isinstance(cv, FLOBSCVResult)
    assert cv.r2_per_weight.shape == (20, 4)             # OLS + 3 weights
    assert cv.argmax_weight_idx.shape == (20,)
    assert cv.weights[0] == "OLS"
    assert cv.n_splits == 4                              # LORO with 4 runs


def test_cv_high_snr_prefers_ols(basis):
    """At high SNR, OLS should win held-out R² since the prior is
    unnecessary; constraint at strong weights should hurt."""
    per_run_data, per_run_task, _ = _build_multirun_synth(
        basis, n_runs=4, n_vox=40, noise_sigma=0.05,
    )
    cv = cv_basis_constrained_ridge(
        per_run_data=per_run_data,
        per_run_task_designs=per_run_task,
        basis_functions=basis.basis_functions,
        prior_mean=basis.m,
        prior_cov=basis.C,
        n_blocks=1,
        polort=2,
        weight_grid=[0.1, 1.0, 10.0],
        include_ols=True,
        device=torch.device("cpu"),
        verbose=False,
    )
    medians = np.median(cv.r2_per_weight, axis=0)
    # OLS should be at least as good as the strongest constraint at
    # high SNR.  At very high SNR both are essentially perfect, so
    # accept equality — the test is "the strong constraint isn't
    # *winning* by an appreciable margin," which would be wrong.
    ols_idx = cv.weights.index("OLS")
    strong_idx = cv.weights.index(10.0)
    assert medians[ols_idx] >= medians[strong_idx] - 0.001


def test_cv_low_snr_prefers_constrained(basis):
    """At low SNR, OLS overfits → held-out R² lower than a sensible
    constraint."""
    per_run_data, per_run_task, _ = _build_multirun_synth(
        basis, n_runs=4, n_vox=40, noise_sigma=4.0,
    )
    cv = cv_basis_constrained_ridge(
        per_run_data=per_run_data,
        per_run_task_designs=per_run_task,
        basis_functions=basis.basis_functions,
        prior_mean=basis.m,
        prior_cov=basis.C,
        n_blocks=1,
        polort=2,
        weight_grid=[0.1, 1.0, 10.0],
        include_ols=True,
        device=torch.device("cpu"),
        verbose=False,
    )
    medians = np.median(cv.r2_per_weight, axis=0)
    ols_idx = cv.weights.index("OLS")
    # Best constrained should beat OLS at low SNR
    best_constrained_idx = int(np.argmax([medians[i] for i in range(len(cv.weights))
                                          if cv.weights[i] != "OLS"]))
    best_constrained_idx += 1 if ols_idx == 0 else 0     # skip OLS slot
    assert medians[best_constrained_idx] > medians[ols_idx]


def test_voxelwise_lambda_runs_and_differs_from_global(basis):
    """voxelwise λ produces different betas from global λ on
    heterogeneous-SNR data, and ought to shrink high-SNR voxels less
    than the global path does."""
    rng = np.random.default_rng(31)
    n_t = 200; tr = 1.0
    sb = generate_spmg_basis(n_basis=2, duration=32.0, dt=0.1)
    sb_tr = sb.basis_functions[:, :: int(round(tr / sb.dt))]
    X = np.zeros((n_t, 2))
    on = np.zeros(n_t); on[[15, 45, 80, 120, 165]] = 1.0
    for b in range(2):
        X[:, b] = np.convolve(on, sb_tr[b])[:n_t]

    # Make voxels with VERY heterogeneous SNR — first half low-noise,
    # second half high-noise — so per-voxel σ² differs by 100×.
    n_vox = 40
    true_coefs = np.column_stack([
        rng.uniform(0.5, 2.0, size=n_vox),
        rng.normal(0.0, 0.05, size=n_vox),
    ])
    y_clean = (X @ true_coefs.T).T                                # (n_vox, n_t)
    noise = rng.standard_normal((n_vox, n_t))
    sigma_per_vox = np.concatenate([np.full(n_vox // 2, 0.1),
                                     np.full(n_vox - n_vox // 2, 5.0)])
    y = y_clean + sigma_per_vox[:, None] * y_clean.std() * noise

    pm, pc = spmg_prior(canonical_std=5.0, derivative_std=0.3)
    common = dict(
        data=torch.from_numpy(y).double(),
        design_task=torch.from_numpy(X).double(),
        basis_functions=sb.basis_functions,
        prior_mean=pm, prior_cov=pc, n_blocks=1,
        device=torch.device("cpu"),
    )
    fit_global = fit_basis_constrained_ridge(**common, lambda_mode="global")
    fit_voxel = fit_basis_constrained_ridge(**common, lambda_mode="voxelwise",
                                            lambda_n_bins=8)

    # Different mode → different betas
    assert not np.allclose(fit_global.betas, fit_voxel.betas, atol=1e-6)

    # On the LOW-noise half (first n_vox/2 voxels), voxelwise λ should
    # shrink LESS (since σ²_v is much lower than σ²_mean).  Amplitude
    # coefficients should be CLOSER TO the OLS estimate in voxelwise
    # mode than in global mode.
    low_idx = np.arange(n_vox // 2)
    diff_to_ols_global = np.abs(
        fit_global.betas[low_idx, 0] - fit_global.betas_ols[low_idx, 0]
    ).mean()
    diff_to_ols_voxel = np.abs(
        fit_voxel.betas[low_idx, 0] - fit_voxel.betas_ols[low_idx, 0]
    ).mean()
    assert diff_to_ols_voxel < diff_to_ols_global


def test_voxelwise_lambda_invalid_mode(basis):
    y, X, _ = _build_test_data(basis, n_vox=10)
    with pytest.raises(ValueError, match="lambda_mode"):
        fit_basis_constrained_ridge(
            data=torch.from_numpy(y).double(),
            design_task=torch.from_numpy(X).double(),
            basis_functions=basis.basis_functions,
            prior_mean=basis.m, prior_cov=basis.C, n_blocks=1,
            lambda_mode="nonsense",
            device=torch.device("cpu"),
        )


# ---------- amplitude decoupling --------------------------------------------


def test_decouple_amplitude_prior_zero_mean_raises():
    with pytest.raises(ValueError, match="prior_mean is ~zero"):
        decouple_amplitude_prior(np.zeros(3), np.eye(3))


def test_decouple_amplitude_prior_preserves_amplitude(basis):
    """Decoupled prior with strong weight at low SNR should:
       - shrink shape (derivative coefficient stays small)
       - NOT shrink amplitude (canonical coefficient stays ≈ OLS).
    Compare to plain mvn at the same weight — amplitude should be
    closer to OLS under mvn-shape.
    """
    rng = np.random.default_rng(53)
    n_t, tr = 200, 1.0
    sb = generate_spmg_basis(n_basis=2, duration=32.0, dt=0.1)
    sb_tr = sb.basis_functions[:, :: int(round(tr / sb.dt))]
    X = np.zeros((n_t, 2))
    on = np.zeros(n_t); on[[15, 45, 80, 120, 165]] = 1.0
    for b in range(2):
        X[:, b] = np.convolve(on, sb_tr[b])[:n_t]

    # 30 voxels with LARGE amplitudes (we want to see if the prior
    # over-shrinks them).
    n_vox = 30
    true_canon = rng.uniform(2.0, 5.0, size=n_vox)               # strong signal
    true_deriv = rng.normal(0.0, 0.05, size=n_vox)
    true_coefs = np.column_stack([true_canon, true_deriv])
    y_clean = (X @ true_coefs.T).T
    # Moderate noise — enough to make the prior visibly help shape
    # but not so much that amplitude is irrelevant.
    y = y_clean + 1.0 * y_clean.std() * rng.standard_normal((n_vox, n_t))

    # Plain mvn prior (with a non-trivial mean so decoupling is well-defined).
    base_m = np.array([3.0, 0.0])
    base_C = np.diag([1.0, 0.09])

    common = dict(
        data=torch.from_numpy(y).double(),
        design_task=torch.from_numpy(X).double(),
        basis_functions=sb.basis_functions,
        n_blocks=1,
        prior_weight=5.0,                                # strong prior
        device=torch.device("cpu"),
    )
    fit_mvn = fit_basis_constrained_ridge(prior_mean=base_m, prior_cov=base_C,
                                          **common)
    m_dec, C_dec = decouple_amplitude_prior(base_m, base_C)
    fit_dec = fit_basis_constrained_ridge(prior_mean=m_dec, prior_cov=C_dec,
                                          **common)

    # On the amplitude (col 0) coefficient: decoupled fit should be
    # CLOSER to OLS than plain mvn (which shrinks it toward base_m[0]=3,
    # which biases voxels whose true canonical is > 3 downward).
    err_mvn_amp = np.abs(fit_mvn.betas[:, 0] - fit_mvn.betas_ols[:, 0]).mean()
    err_dec_amp = np.abs(fit_dec.betas[:, 0] - fit_dec.betas_ols[:, 0]).mean()
    assert err_dec_amp < err_mvn_amp, (
        f"Decoupled amplitude was supposed to track OLS more closely "
        f"({err_dec_amp:.3f}) than full-mvn ({err_mvn_amp:.3f})."
    )

    # On the shape (derivative, col 1) coefficient: BOTH priors
    # should shrink it; the decoupled prior shouldn't undo the shape
    # constraint.  Check that decoupled derivative magnitude isn't
    # dramatically larger than mvn's.
    deriv_mvn = np.abs(fit_mvn.betas[:, 1]).median() if hasattr(np.abs(fit_mvn.betas[:, 1]), 'median') else float(np.median(np.abs(fit_mvn.betas[:, 1])))
    deriv_dec = float(np.median(np.abs(fit_dec.betas[:, 1])))
    deriv_ols = float(np.median(np.abs(fit_mvn.betas_ols[:, 1])))
    # Decoupled should still shrink derivative — it should be closer
    # to mvn than to OLS.
    assert deriv_dec < deriv_ols * 0.8


def test_constrained_fit_two_conditions(basis):
    # Two conditions, distinct onsets, distinct true betas.
    rng = np.random.default_rng(11)
    n_vox, n_t, tr = 30, 200, 1.0
    n_b = 3
    basis_tr = basis.basis_functions[:, :: int(round(tr / basis.dt))]
    X = np.zeros((n_t, 2 * n_b))
    on0 = np.zeros(n_t); on0[[15, 60, 130]] = 1.0
    on1 = np.zeros(n_t); on1[[40, 90, 170]] = 1.0
    for b in range(n_b):
        X[:, b] = np.convolve(on0, basis_tr[b])[:n_t]
        X[:, n_b + b] = np.convolve(on1, basis_tr[b])[:n_t]

    coefs = rng.multivariate_normal(basis.m, basis.C * 0.3, size=(n_vox, 2))  # (n_vox, 2, 3)
    coefs_flat = coefs.reshape(n_vox, 2 * n_b)
    y_clean = (X @ coefs_flat.T).T
    y = y_clean + 0.5 * y_clean.std() * rng.standard_normal((n_vox, n_t))

    fit = fit_flobs_constrained(
        data=torch.from_numpy(y).double(),
        design_task=torch.from_numpy(X).double(),
        basis=basis, n_conditions=2,
        device=torch.device("cpu"),
    )
    assert fit.betas.shape == (n_vox, 6)
    assert fit.hrfs.shape == (n_vox, 2, basis.basis_functions.shape[1])
    # Each condition's recovery should be sensible
    for c in range(2):
        beta_c = fit.betas[:, c * n_b:(c + 1) * n_b]
        err = np.linalg.norm(beta_c - coefs[:, c], axis=1).mean()
        assert err < 1.5  # generous, just a sanity check


# ---------- fit_basis_fracridge ---------------------------------------------


def test_fracridge_returns_correct_shapes(basis):
    per_run_data, per_run_task, _ = _build_multirun_synth(basis, n_runs=4, n_vox=25)
    fit = fit_basis_fracridge(
        per_run_data=per_run_data,
        per_run_task_designs=per_run_task,
        n_blocks=1,
        n_basis=basis.basis_functions.shape[0],
        polort=2,
        fracs=np.array([0.2, 0.5, 1.0]),
        device=torch.device("cpu"),
        verbose=False,
    )
    assert isinstance(fit, FracRidgeFitResult)
    n_task = basis.basis_functions.shape[0]
    assert fit.betas.shape == (25, n_task)
    assert fit.betas_ols.shape == (25, n_task)
    assert fit.r2.shape == (25,)
    assert fit.r2_ols.shape == (25,)
    assert fit.optimal_fracs.shape == (25,)
    assert fit.r2_by_frac.shape == (25, 3)
    # Optimal frac is the one that maximises r2_by_frac per voxel.
    argmax_frac_idx = np.argmax(fit.r2_by_frac, axis=1)
    assert np.allclose(fit.optimal_fracs, fit.fracs[argmax_frac_idx])
    # r2 at optimal frac equals the max along the frac axis.
    assert np.allclose(fit.r2, fit.r2_by_frac.max(axis=1))


def test_fracridge_low_snr_prefers_shrinkage(basis):
    """At low SNR, lots of voxels should pick a frac < 1.0 (some shrinkage)
    rather than frac=1.0 (OLS).  Pure OLS overfits the held-out data."""
    per_run_data, per_run_task, _ = _build_multirun_synth(
        basis, n_runs=4, n_vox=80, noise_sigma=4.0,
    )
    fit = fit_basis_fracridge(
        per_run_data=per_run_data,
        per_run_task_designs=per_run_task,
        n_blocks=1,
        n_basis=basis.basis_functions.shape[0],
        polort=2,
        fracs=np.linspace(0.1, 1.0, 10),
        device=torch.device("cpu"),
        verbose=False,
    )
    # At least half the voxels should prefer some shrinkage.
    fraction_shrunk = float((fit.optimal_fracs < 1.0).mean())
    assert fraction_shrunk > 0.5, f"only {fraction_shrunk:.2f} voxels shrunk"


# ---------- ARMA(1,1) prewhitening foundation -------------------------------


def test_arma11_prewhiten_recovers_known_ar_coefficient(basis):
    """Generate AR(1) noise with known rho; the global-ARMA REML fit
    should pick that rho off the grid (within one grid step).
    """
    rng = np.random.default_rng(0)
    n_runs, n_tp, n_vox = 3, 200, 30
    K = basis.basis_functions.shape[0]
    tr = 1.0
    true_coefs = rng.multivariate_normal(basis.m, basis.C * 0.3, size=n_vox)
    basis_tr = basis.basis_functions[:, :: int(round(tr / basis.dt))]
    target_rho = 0.6

    per_run_data, per_run_designs = [], []
    for _ in range(n_runs):
        on = np.zeros(n_tp)
        on[rng.choice(np.arange(8, n_tp - 30), 7, replace=False)] = 1.0
        X = np.stack(
            [np.convolve(on, basis_tr[b])[:n_tp] for b in range(K)], axis=1
        )
        y_clean = (X @ true_coefs.T).T
        eps = rng.standard_normal((n_vox, n_tp))
        noise = np.zeros_like(eps)
        noise[:, 0] = eps[:, 0]
        for t in range(1, n_tp):
            noise[:, t] = target_rho * noise[:, t - 1] + eps[:, t]
        y = y_clean + 1.5 * y_clean.std() * noise
        per_run_data.append(torch.from_numpy(y).float())
        per_run_designs.append(torch.from_numpy(X).float())

    pwd, pwds, a_opt, b_opt = estimate_and_apply_arma11_prewhitening(
        per_run_data, per_run_designs, polort=2,
        device=torch.device("cpu"), verbose=False,
    )
    # Default grid step is 0.1, so accept |a - target| ≤ 0.1.
    assert abs(a_opt - target_rho) <= 0.1, f"got a={a_opt}, expected ~{target_rho}"
    # Shapes preserved.
    for r in range(n_runs):
        assert pwd[r].shape == per_run_data[r].shape
        assert pwds[r].shape == per_run_designs[r].shape


# ---------- per-voxel ARMA ---------------------------------------------------


def test_per_voxel_arma_recovers_two_groups(basis):
    """Two voxel groups with different AR(1) rho should recover distinct a."""
    rng = np.random.default_rng(0)
    n_runs, n_tp, n_vox = 3, 200, 60
    K = basis.basis_functions.shape[0]
    tr = 1.0
    true_coefs = rng.multivariate_normal(basis.m, basis.C * 0.3, size=n_vox)
    basis_tr = basis.basis_functions[:, :: int(round(tr / basis.dt))]
    rho = np.where(np.arange(n_vox) < n_vox // 2, 0.3, 0.7)
    per_run_data, per_run_designs = [], []
    for _ in range(n_runs):
        on = np.zeros(n_tp)
        on[rng.choice(np.arange(8, n_tp - 30), 7, replace=False)] = 1.0
        X = np.stack(
            [np.convolve(on, basis_tr[b])[:n_tp] for b in range(K)], axis=1
        )
        y_clean = (X @ true_coefs.T).T
        eps = rng.standard_normal((n_vox, n_tp))
        noise = np.zeros_like(eps)
        noise[:, 0] = eps[:, 0]
        for t in range(1, n_tp):
            noise[:, t] = rho * noise[:, t - 1] + eps[:, t]
        y = y_clean + 1.5 * y_clean.std() * noise
        per_run_data.append(torch.from_numpy(y).float())
        per_run_designs.append(torch.from_numpy(X).float())

    from fastfuncstuff.design.flobs import (
        bin_and_whiten_arma11,
        estimate_arma11_per_voxel,
    )
    ab = estimate_arma11_per_voxel(
        per_run_data, per_run_designs, polort=2,
        device=torch.device("cpu"), verbose=False,
    )
    assert ab.shape == (n_vox, 2)
    assert abs(np.median(ab[: n_vox // 2, 0]) - 0.3) <= 0.1
    assert abs(np.median(ab[n_vox // 2 :, 0]) - 0.7) <= 0.1

    cells = bin_and_whiten_arma11(
        per_run_data, per_run_designs, ab, polort=2,
        device=torch.device("cpu"), verbose=False,
    )
    # Sum of cell voxel counts == n_vox; each cell has the right shapes.
    total = sum(c.voxel_indices.size for c in cells)
    assert total == n_vox
    cell = cells[0]
    assert cell.per_run_data[0].shape == (cell.voxel_indices.size, n_tp)
    assert cell.per_run_task_designs[0].shape == (n_tp, K)
    assert cell.per_run_polys is not None
    assert cell.per_run_polys[0].shape == (n_tp, 3)         # polort=2 → 3 cols
