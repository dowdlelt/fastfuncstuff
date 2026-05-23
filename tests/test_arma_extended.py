"""
Extended tests for glm/arma.py - covering uncovered functions:
- compute_reml_likelihood
- reml_grid_search (single voxel)
- build_arma11_covariance_batch
- prewhiten_with_arma11
- compute_ljung_box_statistic
- precompute_reml_grid
- ARMA11Results container
- compute_arma_lambda
- ensure_zero_in_grid
- estimate_valid_grid_pairs
- calculate_grid_memory_footprint
"""

import numpy as np
import pytest
import torch

from fastfuncstuff.glm.arma import (
    ARMA11Results,
    build_arma11_covariance,
    build_arma11_covariance_batch,
    calculate_grid_memory_footprint,
    compute_arma_lambda,
    compute_ljung_box_statistic,
    compute_reml_likelihood,
    ensure_zero_in_grid,
    estimate_valid_grid_pairs,
    get_default_arma_grids,
    precompute_reml_grid,
    prewhiten_with_arma11,
    reml_grid_search,
)

DEVICE = torch.device("cpu")


class TestComputeArmaLambda:
    def test_known_value(self):
        """lambda for a=0.5, b=0.0 should be a (pure AR1)."""
        lam = compute_arma_lambda(0.5, 0.0)
        assert abs(lam - 0.5) < 1e-6

    def test_zero_params(self):
        """a=0, b=0 => lambda=0 (white noise)."""
        assert compute_arma_lambda(0.0, 0.0) == 0.0

    def test_symmetric_formula(self):
        """Verify formula: lambda = (a+b)(1+ab)/(1+b^2+2ab)."""
        a, b = 0.6, 0.2
        expected = (a + b) * (1 + a * b) / (1 + b**2 + 2 * a * b)
        assert abs(compute_arma_lambda(a, b) - expected) < 1e-10


class TestEnsureZeroInGrid:
    def test_adds_zero(self):
        a = torch.tensor([0.1, 0.2, 0.3])
        b = torch.tensor([0.1, 0.2])
        a_new, b_new = ensure_zero_in_grid(a, b)
        assert torch.any(torch.abs(a_new) < 1e-9)
        assert torch.any(torch.abs(b_new) < 1e-9)

    def test_no_duplicate_if_present(self):
        a = torch.tensor([0.0, 0.1, 0.2])
        b = torch.tensor([-0.1, 0.0, 0.1])
        a_new, b_new = ensure_zero_in_grid(a, b)
        assert len(a_new) == 3
        assert len(b_new) == 3

    def test_sorted_output(self):
        a = torch.tensor([0.3, 0.1, 0.2])
        b = torch.tensor([0.2, 0.1])
        a_new, b_new = ensure_zero_in_grid(a, b)
        assert (a_new[:-1] <= a_new[1:]).all()


class TestEstimateValidGridPairs:
    def test_basic(self):
        a = torch.linspace(0.1, 0.5, 5)
        b = torch.linspace(-0.3, 0.3, 7)
        n = estimate_valid_grid_pairs(a, b)
        assert isinstance(n, int)
        assert n > 0

    def test_no_valid(self):
        """All params >= 1 should give 0 valid."""
        a = torch.tensor([1.0, 1.5])
        b = torch.tensor([1.0, 1.5])
        n = estimate_valid_grid_pairs(a, b)
        assert n == 0


class TestCalculateGridMemoryFootprint:
    def test_scales_with_pairs(self):
        mem1 = calculate_grid_memory_footprint(10, 100, 5)
        mem2 = calculate_grid_memory_footprint(20, 100, 5)
        assert mem2 == 2 * mem1

    def test_double_precision(self):
        mem_f32 = calculate_grid_memory_footprint(10, 100, 5, use_double=False)
        mem_f64 = calculate_grid_memory_footprint(10, 100, 5, use_double=True)
        assert mem_f64 == 2 * mem_f32


class TestComputeRemlLikelihood:
    def test_basic_computation(self):
        torch.manual_seed(42)
        T, p = 50, 3
        X = torch.randn(T, p, device=DEVICE)
        Y = torch.randn(T, device=DEVICE)  # 1D input (bug fix handles this)
        R = build_arma11_covariance(0.5, 0.1, T, DEVICE)
        ll = compute_reml_likelihood(X, Y, R)
        assert np.isfinite(ll)
        assert ll < 1e9  # Not a penalty value

    def test_white_noise_baseline(self):
        """Identity covariance (a=0,b=0) should give valid likelihood."""
        torch.manual_seed(0)
        T, p = 40, 2
        X = torch.randn(T, p, device=DEVICE)
        Y = X @ torch.tensor([1.0, 2.0]) + 0.1 * torch.randn(T)
        R = torch.eye(T, device=DEVICE)
        ll = compute_reml_likelihood(X, Y, R)
        assert np.isfinite(ll)
        assert ll < 1e9

    def test_better_model_lower_likelihood(self):
        """Data generated from AR(0.5) should prefer a≈0.5 over a=0."""
        torch.manual_seed(42)
        T, p = 100, 2
        X = torch.randn(T, p, device=DEVICE)
        # Generate AR(1) correlated noise
        noise = torch.zeros(T)
        noise[0] = torch.randn(1)
        for t in range(1, T):
            noise[t] = 0.5 * noise[t - 1] + torch.randn(1)
        Y = X @ torch.tensor([1.0, 2.0]) + noise
        R_good = build_arma11_covariance(0.5, 0.0, T, DEVICE)
        R_bad = torch.eye(T, device=DEVICE)
        ll_good = compute_reml_likelihood(X, Y, R_good)
        ll_bad = compute_reml_likelihood(X, Y, R_bad)
        assert ll_good < ll_bad


class TestRemlGridSearch:
    def test_basic_search(self):
        torch.manual_seed(42)
        T, p = 60, 2
        X = torch.randn(T, p, device=DEVICE)
        Y = torch.randn(T, device=DEVICE)
        a_opt, b_opt, ll_opt = reml_grid_search(X, Y, device=DEVICE)
        assert 0.0 <= a_opt <= 0.9
        assert -0.9 <= b_opt <= 0.9
        assert np.isfinite(ll_opt)

    def test_custom_grids(self):
        torch.manual_seed(0)
        T, p = 40, 2
        X = torch.randn(T, p, device=DEVICE)
        Y = torch.randn(T, device=DEVICE)
        a_grid = torch.tensor([0.1, 0.3, 0.5], device=DEVICE)
        b_grid = torch.tensor([-0.1, 0.0, 0.1], device=DEVICE)
        a_opt, b_opt, ll = reml_grid_search(X, Y, a_grid=a_grid, b_grid=b_grid, device=DEVICE)
        assert any(abs(a_opt - v) < 1e-5 for v in [0.1, 0.3, 0.5])
        assert any(abs(b_opt - v) < 1e-5 for v in [-0.1, 0.0, 0.1])

    def test_returns_finite_likelihood(self):
        """Grid search should return finite optimal likelihood."""
        torch.manual_seed(42)
        T = 60
        X = torch.randn(T, 2, device=DEVICE)
        Y = torch.randn(T, device=DEVICE)
        _, _, ll = reml_grid_search(X, Y, device=DEVICE)
        assert np.isfinite(ll)


class TestBuildArma11CovarianceBatch:
    def test_basic(self):
        a_grid = torch.tensor([0.2, 0.5], device=DEVICE)
        b_grid = torch.tensor([0.0, 0.1], device=DEVICE)
        R_batch, params, param_list = build_arma11_covariance_batch(
            a_grid, b_grid, n=30, device=DEVICE
        )
        assert R_batch.ndim == 3
        assert R_batch.shape[1] == 30
        assert R_batch.shape[2] == 30
        assert params.shape[1] == 2
        assert len(param_list) == params.shape[0]

    def test_matches_scalar(self):
        """Batch version should match scalar build_arma11_covariance."""
        a_grid = torch.tensor([0.3], device=DEVICE)
        b_grid = torch.tensor([0.1], device=DEVICE)
        R_batch, params, _ = build_arma11_covariance_batch(
            a_grid, b_grid, n=20, device=DEVICE
        )
        R_scalar = build_arma11_covariance(0.3, 0.1, 20, DEVICE)
        torch.testing.assert_close(R_batch[0], R_scalar, atol=1e-5, rtol=1e-5)

    def test_symmetry(self):
        a_grid = torch.tensor([0.4], device=DEVICE)
        b_grid = torch.tensor([0.2], device=DEVICE)
        R_batch, _, _ = build_arma11_covariance_batch(
            a_grid, b_grid, n=15, device=DEVICE
        )
        torch.testing.assert_close(R_batch[0], R_batch[0].T)

    def test_invalid_all(self):
        """All invalid params should return empty."""
        a_grid = torch.tensor([1.5], device=DEVICE)
        b_grid = torch.tensor([1.5], device=DEVICE)
        R_batch, params, param_list = build_arma11_covariance_batch(
            a_grid, b_grid, n=10, device=DEVICE
        )
        assert R_batch.shape[0] == 0
        assert len(param_list) == 0

    def test_filters_negative_lambda(self):
        """Params giving lambda<0 should be filtered out."""
        # a=0.1, b=-0.8 gives negative lambda
        a_grid = torch.tensor([0.1], device=DEVICE)
        b_grid = torch.tensor([-0.8], device=DEVICE)
        R_batch, params, param_list = build_arma11_covariance_batch(
            a_grid, b_grid, n=10, device=DEVICE
        )
        assert R_batch.shape[0] == 0


class TestPrewhitenWithArma11:
    def test_basic(self):
        torch.manual_seed(42)
        T, p = 50, 3
        X = torch.randn(T, p, device=DEVICE)
        Y = torch.randn(T, device=DEVICE)
        X_w, Y_w, L = prewhiten_with_arma11(X, Y, 0.5, 0.1)
        assert X_w.shape == X.shape
        assert Y_w.shape == Y.shape
        assert L.shape == (T, T)

    def test_2d_data(self):
        """Should work with (T, n_voxels) data."""
        T, p, V = 40, 2, 10
        X = torch.randn(T, p, device=DEVICE)
        Y = torch.randn(T, V, device=DEVICE)
        X_w, Y_w, L = prewhiten_with_arma11(X, Y, 0.3, 0.0)
        assert X_w.shape == (T, p)
        assert Y_w.shape == (T, V)

    def test_invalid_params_raises(self):
        X = torch.randn(20, 2, device=DEVICE)
        Y = torch.randn(20, device=DEVICE)
        with pytest.raises(ValueError, match="Invalid ARMA"):
            prewhiten_with_arma11(X, Y, 1.5, 0.0)

    def test_identity_for_white_noise(self):
        """a=0, b=0 should give identity-like transform (no change)."""
        T, p = 30, 2
        X = torch.randn(T, p, device=DEVICE)
        Y = torch.randn(T, device=DEVICE)
        X_w, Y_w, L = prewhiten_with_arma11(X, Y, 0.0, 0.0)
        # L should be identity, so X_w ≈ X
        torch.testing.assert_close(X_w, X, atol=1e-5, rtol=1e-5)


class TestComputeLjungBoxStatistic:
    def test_white_noise(self):
        """White noise should have low LB statistic."""
        rng = np.random.default_rng(42)
        residuals = rng.standard_normal((5, 200))
        lb = compute_ljung_box_statistic(residuals, max_lag=10)
        assert lb.shape == (5,)
        # White noise: LB should be moderate (chi2 with df=8)
        assert (lb < 100).all()

    def test_correlated_residuals(self):
        """Highly correlated residuals should have large LB."""
        rng = np.random.default_rng(42)
        n_vox, T = 3, 200
        residuals = np.zeros((n_vox, T))
        for v in range(n_vox):
            residuals[v, 0] = rng.standard_normal()
            for t in range(1, T):
                residuals[v, t] = 0.8 * residuals[v, t - 1] + 0.2 * rng.standard_normal()
        lb = compute_ljung_box_statistic(residuals, max_lag=10)
        assert (lb > 10).all()  # Should be clearly significant

    def test_zero_residuals(self):
        """Zero residuals should give LB=0."""
        residuals = np.zeros((2, 50))
        lb = compute_ljung_box_statistic(residuals)
        assert (lb == 0.0).all()

    def test_torch_input(self):
        """Should accept torch tensors."""
        residuals = torch.randn(3, 100)
        lb = compute_ljung_box_statistic(residuals, max_lag=5)
        assert isinstance(lb, np.ndarray)
        assert lb.shape == (3,)


class TestPrecomputeRemlGrid:
    def test_basic(self):
        torch.manual_seed(42)
        T, p = 40, 3
        X = torch.randn(T, p, device=DEVICE)
        a_grid = torch.tensor([0.2, 0.5], device=DEVICE)
        b_grid = torch.tensor([0.0, 0.1], device=DEVICE)
        grid = precompute_reml_grid(
            X, T, a_grid, b_grid, device=DEVICE, verbose=False
        )
        # Grid is a dict keyed by (a,b) tuples
        assert isinstance(grid, dict)
        assert len(grid) > 0
        # Each entry should have the whitening inverse and whitened design
        first_key = next(iter(grid))
        assert "L_inv" in grid[first_key]
        assert "X_w" in grid[first_key]

    def test_with_qr(self):
        torch.manual_seed(0)
        T, p = 30, 2
        X = torch.randn(T, p, device=DEVICE)
        a_grid = torch.tensor([0.3], device=DEVICE)
        b_grid = torch.tensor([0.0], device=DEVICE)
        grid = precompute_reml_grid(
            X, T, a_grid, b_grid, device=DEVICE, use_qr=True
        )
        assert len(grid) > 0


class TestARMA11Results:
    def test_init(self):
        r = ARMA11Results()
        assert r.betas is None
        assert r.tstats is None
        assert r.r2 is None
        assert r.arma_params is None
        assert r.dof is None
        assert r.contrast_labels is None

    def test_assignment(self):
        r = ARMA11Results()
        r.betas = torch.randn(100, 5)
        r.r2 = torch.randn(100)
        r.arma_params = torch.randn(100, 2)
        assert r.betas.shape == (100, 5)
        assert r.r2.shape == (100,)


class TestGetDefaultArmaGrids:
    def test_grids(self):
        a, b = get_default_arma_grids(DEVICE)
        assert a.device == DEVICE
        assert b.device == DEVICE
        assert len(a) >= 5
        assert len(b) >= 5
        # a should be non-negative
        assert (a >= 0).all()
