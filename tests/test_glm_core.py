"""
Comprehensive tests for GLM core functionality.

Tests cover:
- Basic GLM fitting (OLS)
- Design matrix construction
- Statistical tests (t-stats, F-stats)
- R² computation
- Edge cases and numerical stability
"""

import numpy as np
import pytest
import torch

from fastfuncsim.glm_core import fit_glm
from fastfuncsim.utils import get_device


class TestBasicGLM:
    """Test basic GLM fitting functionality."""

    @pytest.fixture
    def device(self):
        return get_device()

    def test_simple_glm_fit(self, device):
        """Test basic GLM fit with known signal."""
        torch.manual_seed(42)

        n_timepoints = 100
        n_voxels = 10
        n_regressors = 2

        # Create design matrix
        X = torch.randn(n_timepoints, n_regressors, device=device)

        # True betas
        true_betas = torch.tensor([[1.0, 2.0]] * n_voxels, device=device)

        # Generate signal with small noise
        signal = X @ true_betas.T  # (n_timepoints, n_voxels)
        noise = 0.1 * torch.randn_like(signal)
        data = (signal + noise).T  # (n_voxels, n_timepoints)

        # Fit GLM
        results = fit_glm(data, X, tr=1.0, verbose=False, device=device)

        # Check beta recovery
        assert results.betas.shape == (n_voxels, n_regressors)
        assert torch.allclose(
            results.betas, true_betas, atol=0.3
        )  # Allow some noise tolerance

        # Check R² is high (low noise)
        assert torch.all(results.r2 > 0.9)

    def test_glm_perfect_fit(self, device):
        """Test GLM with perfect fit (no noise)."""
        torch.manual_seed(42)

        n_timepoints = 100
        n_voxels = 5
        n_regressors = 3

        X = torch.randn(n_timepoints, n_regressors, device=device)
        true_betas = torch.randn(n_voxels, n_regressors, device=device)

        # Perfect signal (no noise)
        signal = X @ true_betas.T
        data = signal.T

        results = fit_glm(data, X, tr=1.0, verbose=False, device=device)

        # Perfect recovery
        assert torch.allclose(results.betas, true_betas, atol=1e-5)

        # R² should be 1 (or very close)
        assert torch.all(results.r2 > 0.9999)

    def test_glm_single_voxel(self, device):
        """Test GLM with single voxel."""
        torch.manual_seed(42)

        n_timepoints = 50
        n_regressors = 2

        X = torch.randn(n_timepoints, n_regressors, device=device)
        true_beta = torch.tensor([[2.5, -1.5]], device=device)

        signal = X @ true_beta.T
        noise = 0.5 * torch.randn(n_timepoints, 1, device=device)
        data = (signal + noise).T  # (1, n_timepoints)

        results = fit_glm(data, X, tr=1.0, verbose=False, device=device)

        assert results.betas.shape == (1, 2)
        assert results.r2.shape == (1,)
        assert torch.allclose(results.betas, true_beta, atol=0.5)

    def test_glm_single_regressor(self, device):
        """Test GLM with single regressor."""
        torch.manual_seed(42)

        n_timepoints = 100
        n_voxels = 10

        X = torch.randn(n_timepoints, 1, device=device)
        true_betas = torch.randn(n_voxels, 1, device=device) * 2

        signal = X @ true_betas.T
        noise = 0.3 * torch.randn_like(signal)
        data = (signal + noise).T

        results = fit_glm(data, X, tr=1.0, verbose=False, device=device)

        assert results.betas.shape == (n_voxels, 1)
        assert torch.allclose(results.betas, true_betas, atol=0.5)


class TestGLMStatistics:
    """Test statistical outputs from GLM."""

    @pytest.fixture
    def device(self):
        return get_device()

    def test_tstat_computation(self, device):
        """Test t-statistic computation."""
        torch.manual_seed(42)

        n_timepoints = 200
        n_voxels = 20
        n_regressors = 3

        X = torch.randn(n_timepoints, n_regressors, device=device)
        true_betas = torch.randn(n_voxels, n_regressors, device=device)

        signal = X @ true_betas.T
        noise = 0.5 * torch.randn_like(signal)
        data = (signal + noise).T

        results = fit_glm(data, X, tr=1.0, verbose=False, device=device)

        # Check t-stats shape
        assert results.tstats.shape == (n_voxels, n_regressors)

        # t-stats should be finite
        assert torch.all(torch.isfinite(results.tstats))

        # For true signal, t-stats should be large
        # (|t| > 2 is roughly p < 0.05 for large df)
        mean_abs_tstat = torch.abs(results.tstats).mean()
        assert mean_abs_tstat > 2.0

    def test_fstat_computation(self, device):
        """Test F-statistic computation."""
        torch.manual_seed(42)

        n_timepoints = 150
        n_voxels = 15
        n_regressors = 4

        X = torch.randn(n_timepoints, n_regressors, device=device)
        true_betas = torch.randn(n_voxels, n_regressors, device=device)

        signal = X @ true_betas.T
        noise = 0.3 * torch.randn_like(signal)
        data = (signal + noise).T

        results = fit_glm(data, X, tr=1.0, verbose=False, device=device)

        # F-stat is overall model significance
        assert results.fstats.shape == (n_voxels,)
        assert torch.all(torch.isfinite(results.fstats))
        assert torch.all(results.fstats >= 0)  # F-stats are non-negative

        # For signal + noise, F should be significant
        assert torch.all(results.fstats > 1.0)

    def test_r2_range(self, device):
        """Test that R² is in valid range [0, 1]."""
        torch.manual_seed(42)

        n_timepoints = 100
        n_voxels = 20
        n_regressors = 2

        X = torch.randn(n_timepoints, n_regressors, device=device)
        true_betas = torch.randn(n_voxels, n_regressors, device=device)

        signal = X @ true_betas.T

        # Test different noise levels
        for noise_std in [0.1, 0.5, 1.0, 2.0]:
            noise = noise_std * torch.randn_like(signal)
            data = (signal + noise).T

            results = fit_glm(data, X, tr=1.0, verbose=False, device=device)

            # R² should be in [0, 1]
            assert torch.all(results.r2 >= 0)
            assert torch.all(results.r2 <= 1)

            # Higher noise should decrease R²
            if noise_std == 0.1:
                high_r2 = results.r2.mean().item()
            elif noise_std == 2.0:
                low_r2 = results.r2.mean().item()

        assert high_r2 > low_r2

    def test_residuals_orthogonal(self, device):
        """Test that residuals are orthogonal to design matrix."""
        torch.manual_seed(42)

        n_timepoints = 100
        n_voxels = 10
        n_regressors = 3

        X = torch.randn(n_timepoints, n_regressors, device=device)
        true_betas = torch.randn(n_voxels, n_regressors, device=device)

        signal = X @ true_betas.T
        noise = 0.5 * torch.randn_like(signal)
        data = (signal + noise).T

        # Disable polynomial detrending completely (use -1 to get no polynomial terms)
        results = fit_glm(data, X, tr=1.0, max_poly_degree=-1, verbose=False, device=device)

        # Residuals = data - predictions
        predictions = (X @ results.betas.T).T  # (n_voxels, n_timepoints)
        residuals = data - predictions

        # Check X^T @ residuals ≈ 0
        orth_check = X.T @ residuals.T  # (n_regressors, n_voxels)

        # Should be close to zero
        assert torch.abs(orth_check).max() < 1e-4


class TestGLMEdgeCases:
    """Test edge cases and numerical stability."""

    @pytest.fixture
    def device(self):
        return get_device()

    def test_glm_with_constant(self, device):
        """Test GLM with constant regressor (intercept)."""
        torch.manual_seed(42)

        n_timepoints = 100
        n_voxels = 5

        # Design with constant
        X = torch.ones(n_timepoints, 1, device=device)
        true_mean = torch.tensor([[100.0]] * n_voxels, device=device)

        signal = X @ true_mean.T
        noise = 2.0 * torch.randn_like(signal)
        data = (signal + noise).T

        # Disable polynomial detrending completely (use -1 to avoid adding constant)
        results = fit_glm(data, X, tr=1.0, max_poly_degree=-1, verbose=False, device=device)

        # Beta should estimate mean
        assert torch.allclose(results.betas, true_mean, atol=1.0)

    def test_glm_large_data(self, device):
        """Test GLM with large data (stress test)."""
        torch.manual_seed(42)

        n_timepoints = 500
        n_voxels = 1000
        n_regressors = 10

        X = torch.randn(n_timepoints, n_regressors, device=device)
        true_betas = torch.randn(n_voxels, n_regressors, device=device)

        signal = X @ true_betas.T
        noise = 0.5 * torch.randn_like(signal)
        data = (signal + noise).T

        results = fit_glm(data, X, tr=1.0, verbose=False, device=device)

        # Should complete without error
        assert results.betas.shape == (n_voxels, n_regressors)
        assert torch.all(torch.isfinite(results.betas))
        assert torch.all(torch.isfinite(results.tstats))
        assert torch.all(torch.isfinite(results.fstats))

    def test_glm_collinear_regressors(self, device):
        """Test GLM with highly collinear regressors."""
        torch.manual_seed(42)

        n_timepoints = 100
        n_voxels = 5

        # Create collinear design
        x1 = torch.randn(n_timepoints, 1, device=device)
        x2 = x1 + 0.01 * torch.randn(n_timepoints, 1, device=device)  # Almost identical
        X = torch.cat([x1, x2], dim=1)

        data = torch.randn(n_voxels, n_timepoints, device=device)

        # Should still run (ridge regularization handles this)
        results = fit_glm(data, X, tr=1.0, verbose=False, device=device)

        # Results should be finite (no NaN/Inf)
        assert torch.all(torch.isfinite(results.betas))

    def test_glm_zero_variance_data(self, device):
        """Test GLM with zero-variance data (constant signal)."""
        n_timepoints = 100
        n_voxels = 5
        n_regressors = 2

        X = torch.randn(n_timepoints, n_regressors, device=device)

        # Constant data (no variance)
        data = torch.ones(n_voxels, n_timepoints, device=device) * 50.0

        results = fit_glm(data, X, tr=1.0, verbose=False, device=device)

        # R² should be 0 or very small (no variance to explain)
        assert torch.all(results.r2 < 0.01)

    def test_glm_different_tr_values(self, device):
        """Test that TR parameter is handled correctly."""
        torch.manual_seed(42)

        n_timepoints = 100
        n_voxels = 10
        n_regressors = 2

        X = torch.randn(n_timepoints, n_regressors, device=device)
        true_betas = torch.randn(n_voxels, n_regressors, device=device)
        signal = X @ true_betas.T
        data = signal.T

        # Test different TRs
        for tr in [0.5, 1.0, 2.0, 3.0]:
            results = fit_glm(data, X, tr=tr, verbose=False, device=device)

            # Beta estimates shouldn't depend on TR
            assert torch.allclose(results.betas, true_betas, atol=1e-4)


class TestGLMBatchProcessing:
    """Test batch processing capabilities."""

    @pytest.fixture
    def device(self):
        return get_device()

    def test_multiple_voxel_batches(self, device):
        """Test GLM processes all voxels correctly."""
        torch.manual_seed(42)

        n_timepoints = 100
        n_regressors = 3

        X = torch.randn(n_timepoints, n_regressors, device=device)

        # Process different numbers of voxels
        for n_voxels in [1, 5, 50, 500]:
            true_betas = torch.randn(n_voxels, n_regressors, device=device)
            signal = X @ true_betas.T
            data = signal.T

            results = fit_glm(data, X, tr=1.0, verbose=False, device=device)

            assert results.betas.shape == (n_voxels, n_regressors)
            assert results.r2.shape == (n_voxels,)
            assert results.tstats.shape == (n_voxels, n_regressors)
            assert results.fstats.shape == (n_voxels,)

    def test_batched_vs_sequential(self, device):
        """Test that batched processing gives same results as sequential."""
        torch.manual_seed(42)

        n_timepoints = 80
        n_voxels = 10
        n_regressors = 2

        X = torch.randn(n_timepoints, n_regressors, device=device)
        data = torch.randn(n_voxels, n_timepoints, device=device)

        # Batch process
        results_batch = fit_glm(data, X, tr=1.0, verbose=False, device=device)

        # Sequential process
        betas_seq = []
        for v in range(n_voxels):
            results_single = fit_glm(
                data[v : v + 1], X, tr=1.0, verbose=False, device=device
            )
            betas_seq.append(results_single.betas)

        betas_seq = torch.cat(betas_seq, dim=0)

        # Should match
        assert torch.allclose(results_batch.betas, betas_seq, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
