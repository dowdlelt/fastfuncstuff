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


class TestGLMResultsToSpatial:
    """Test GLMResults.to_spatial() spatial reshaping."""

    @pytest.fixture
    def device(self):
        return get_device()

    def test_to_spatial_3d(self, device):
        """Test reshaping results back to 3D spatial format."""
        from fastfuncsim.glm_core import GLMResults
        
        torch.manual_seed(42)
        nx, ny, nz = 5, 6, 4
        n_voxels = nx * ny * nz
        n_timepoints = 100
        n_regressors = 2

        X = torch.randn(n_timepoints, n_regressors, device=device)
        true_betas = torch.randn(n_voxels, n_regressors, device=device)
        signal = X @ true_betas.T
        data = signal.T

        results = fit_glm(data, X, tr=1.0, verbose=False, device=device)
        
        # Store original shape and call to_spatial
        results.original_shape = (nx, ny, nz)
        results = results.to_spatial()

        # Check shapes are now spatial
        assert results.betas.shape == (nx, ny, nz, n_regressors)
        assert results.r2.shape == (nx, ny, nz)
        assert results.tstats.shape == (nx, ny, nz, n_regressors)
        assert results.fstats.shape == (nx, ny, nz)

    def test_to_spatial_no_original_shape_warns(self, device):
        """Test that to_spatial warns when original_shape is None."""
        from fastfuncsim.glm_core import GLMResults
        import warnings
        
        results = GLMResults()
        results.betas = torch.randn(100, 3)
        results.r2 = torch.randn(100)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results.to_spatial()
            assert len(w) == 1
            assert "Original shape not stored" in str(w[0].message)


class TestGLM4DInput:
    """Test GLM with 4D spatial data input."""

    @pytest.fixture
    def device(self):
        return get_device()

    def test_4d_data_input(self, device):
        """Test GLM with 4D (nx, ny, nz, nt) data input."""
        torch.manual_seed(42)

        nx, ny, nz, n_timepoints = 5, 5, 3, 80
        n_regressors = 2

        # 4D data
        data_4d = torch.randn(nx, ny, nz, n_timepoints, device=device)
        X = torch.randn(n_timepoints, n_regressors, device=device)

        results = fit_glm(data_4d, X, tr=1.0, verbose=False, device=device)

        # Should be flattened
        n_voxels = nx * ny * nz
        assert results.betas.shape == (n_voxels, n_regressors)
        assert results.original_shape == (nx, ny, nz)

    def test_invalid_data_dimension(self, device):
        """Test that invalid data dimensions raise error."""
        X = torch.randn(100, 2, device=device)
        data_3d = torch.randn(10, 10, 100, device=device)  # Wrong 3D

        with pytest.raises(ValueError, match="must be 2D.*or 4D"):
            fit_glm(data_3d, X, tr=1.0, verbose=False, device=device)


class TestGLMMultiRun:
    """Test GLM with multiple runs."""

    @pytest.fixture
    def device(self):
        return get_device()

    def test_multi_run_fit(self, device):
        """Test GLM with list of runs."""
        torch.manual_seed(42)

        n_voxels = 50
        n_timepoints_per_run = 100
        n_regressors = 3
        n_runs = 2

        # Create data and design for each run
        data_list = [torch.randn(n_voxels, n_timepoints_per_run, device=device)
                     for _ in range(n_runs)]
        design_list = [torch.randn(n_timepoints_per_run, n_regressors, device=device)
                       for _ in range(n_runs)]

        results = fit_glm(
            data_list, design_list, tr=1.0,
            verbose=False, device=device
        )

        # Betas should have task regressors from all runs
        # Default behavior: n_runs * n_regressors task betas
        assert results.betas.shape[0] == n_voxels
        assert torch.all(torch.isfinite(results.betas))

    def test_multi_run_with_extra_regressors(self, device):
        """Test multi-run GLM with extra nuisance regressors."""
        torch.manual_seed(42)

        n_voxels = 30
        n_timepoints = 80
        n_task = 2
        n_extra = 3
        n_runs = 2

        data_list = [torch.randn(n_voxels, n_timepoints, device=device)
                     for _ in range(n_runs)]
        design_list = [torch.randn(n_timepoints, n_task, device=device)
                       for _ in range(n_runs)]
        extra_list = [torch.randn(n_timepoints, n_extra, device=device)
                      for _ in range(n_runs)]

        results = fit_glm(
            data_list, design_list, tr=1.0,
            extra_regressors=extra_list,
            verbose=False, device=device
        )

        assert results.betas.shape[0] == n_voxels
        assert torch.all(torch.isfinite(results.r2))


class TestGLMAdvancedFeatures:
    """Test advanced GLM features."""

    @pytest.fixture
    def device(self):
        return get_device()

    def test_use_double_precision(self, device):
        """Test GLM with double precision."""
        torch.manual_seed(42)

        n_timepoints = 100
        n_voxels = 10
        n_regressors = 2

        X = torch.randn(n_timepoints, n_regressors, device=device)
        data = torch.randn(n_voxels, n_timepoints, device=device)

        results = fit_glm(
            data, X, tr=1.0,
            use_double=True,
            verbose=False, device=device
        )

        # Results should be in float64
        assert results.betas.dtype == torch.float64

    def test_streaming_mode(self, device):
        """Test GLM with preload_data_to_device=False (streaming)."""
        torch.manual_seed(42)

        n_timepoints = 100
        n_voxels = 100
        n_regressors = 2

        X = torch.randn(n_timepoints, n_regressors, device=device)
        data = torch.randn(n_voxels, n_timepoints, device=device)

        # Force data to CPU for streaming test
        data_cpu = data.cpu()

        results = fit_glm(
            data_cpu, X, tr=1.0,
            preload_data_to_device=False,
            verbose=False, device=device
        )

        assert results.betas.shape == (n_voxels, n_regressors)
        assert torch.all(torch.isfinite(results.betas))

    def test_want_residuals_and_predicted(self, device):
        """Test GLM returning residuals and predicted values."""
        torch.manual_seed(42)

        n_timepoints = 80
        n_voxels = 20
        n_regressors = 2

        X = torch.randn(n_timepoints, n_regressors, device=device)
        true_betas = torch.randn(n_voxels, n_regressors, device=device)
        signal = X @ true_betas.T
        data = signal.T + 0.1 * torch.randn(n_voxels, n_timepoints, device=device)

        results = fit_glm(
            data, X, tr=1.0,
            want_residuals=True,
            want_predicted=True,
            verbose=False, device=device
        )

        # Should have residuals and predicted
        assert results.residuals is not None
        assert results.predicted is not None
        assert results.residuals.shape == (n_voxels, n_timepoints)
        assert results.predicted.shape == (n_voxels, n_timepoints)

    def test_partial_r2_computation(self, device):
        """Test partial R² computation."""
        torch.manual_seed(42)

        n_timepoints = 100
        n_voxels = 30
        n_regressors = 3

        X = torch.randn(n_timepoints, n_regressors, device=device)
        data = torch.randn(n_voxels, n_timepoints, device=device)

        results = fit_glm(
            data, X, tr=1.0,
            want_r2_partial=True,
            verbose=False, device=device
        )

        # Should have partial R²
        assert results.r2_partial is not None
        # Partial R² should be in [0, 1]
        assert torch.all(results.r2_partial >= 0)
        assert torch.all(results.r2_partial <= 1)

    def test_semipartial_r2_computation(self, device):
        """Test semi-partial R² computation."""
        torch.manual_seed(42)

        n_timepoints = 100
        n_voxels = 30
        n_regressors = 3

        X = torch.randn(n_timepoints, n_regressors, device=device)
        data = torch.randn(n_voxels, n_timepoints, device=device)

        results = fit_glm(
            data, X, tr=1.0,
            want_r2_semipartial=True,
            verbose=False, device=device
        )

        # Should have semi-partial R²
        assert results.r2_semipartial is not None
        assert torch.all(results.r2_semipartial >= 0)


class TestGLMContrasts:
    """Test GLT contrast computation."""

    @pytest.fixture
    def device(self):
        return get_device()

    def test_basic_contrast(self, device):
        """Test basic single-row contrast (t-test)."""
        torch.manual_seed(42)

        n_timepoints = 100
        n_voxels = 20
        n_regressors = 3

        X = torch.randn(n_timepoints, n_regressors, device=device)
        data = torch.randn(n_voxels, n_timepoints, device=device)

        # Simple contrast: first regressor vs zero
        glt_labels = ["reg1_vs_zero"]
        glt_matrices = [np.array([1.0, 0.0, 0.0])]

        results = fit_glm(
            data, X, tr=1.0,
            max_poly_degree=-1,  # No polynomial to keep regressor count simple
            glt_labels=glt_labels,
            glt_matrices=glt_matrices,
            verbose=False, device=device
        )

        assert results.contrast_labels == glt_labels
        assert results.contrast_betas is not None
        assert results.contrast_tstats is not None
        assert results.contrast_betas.shape == (n_voxels, 1)

    def test_difference_contrast(self, device):
        """Test difference contrast between two regressors."""
        torch.manual_seed(42)

        n_timepoints = 100
        n_voxels = 20
        n_regressors = 2

        X = torch.randn(n_timepoints, n_regressors, device=device)
        true_betas = torch.tensor([[5.0, 1.0]] * n_voxels, device=device)
        signal = X @ true_betas.T
        data = signal.T + 0.1 * torch.randn(n_voxels, n_timepoints, device=device)

        # Contrast: regressor 1 - regressor 2
        glt_labels = ["reg1_minus_reg2"]
        glt_matrices = [np.array([1.0, -1.0])]

        results = fit_glm(
            data, X, tr=1.0,
            max_poly_degree=-1,
            glt_labels=glt_labels,
            glt_matrices=glt_matrices,
            verbose=False, device=device
        )

        # Contrast beta should be approximately 5 - 1 = 4
        mean_contrast_beta = results.contrast_betas.mean().item()
        assert abs(mean_contrast_beta - 4.0) < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

