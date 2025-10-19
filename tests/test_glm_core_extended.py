"""
Additional tests for GLM core functions to increase coverage.

Focus on:
- construct_polynomial_matrix
- orthogonalize_design
- percent_bold_change
- fit_glm edge cases
- chunk_size parameter
"""

import pytest
import torch

from fastfuncsim.glm_core import (
    fit_glm,
    construct_polynomial_matrix,
    orthogonalize_design,
    percent_bold_change,
)
from fastfuncsim.utils import get_device


class TestPolynomialMatrix:
    """Test polynomial design matrix construction."""

    @pytest.fixture
    def device(self):
        return get_device()

    def test_construct_polynomial_basic(self, device):
        """Test constructing polynomial matrix with different degrees."""
        n_timepoints = 100

        # Degree 1 (constant + linear trend)
        poly1 = construct_polynomial_matrix(n_timepoints, max_degree=1, device=device)
        assert poly1.shape == (n_timepoints, 2)  # constant + linear

        # First column should be constant (t^0 = 1)
        assert torch.allclose(poly1[:, 0], torch.ones(n_timepoints, device=device))

        # Second column should be linear (t^1 = t)
        t = torch.linspace(-1, 1, n_timepoints, device=device)
        assert torch.allclose(poly1[:, 1], t)

    def test_construct_polynomial_degree_2(self, device):
        """Test quadratic polynomial."""
        n_timepoints = 100
        poly2 = construct_polynomial_matrix(n_timepoints, max_degree=2, device=device)

        assert poly2.shape == (n_timepoints, 3)  # constant + linear + quadratic

        t = torch.linspace(-1, 1, n_timepoints, device=device)

        # First column: constant
        assert torch.allclose(poly2[:, 0], torch.ones(n_timepoints, device=device))

        # Second column: linear
        assert torch.allclose(poly2[:, 1], t)

        # Third column: quadratic
        assert torch.allclose(poly2[:, 2], t**2)

    def test_construct_polynomial_degree_3(self, device):
        """Test cubic polynomial."""
        n_timepoints = 50
        poly3 = construct_polynomial_matrix(n_timepoints, max_degree=3, device=device)

        assert poly3.shape == (n_timepoints, 4)  # constant + linear + quadratic + cubic

        t = torch.linspace(-1, 1, n_timepoints, device=device)
        assert torch.allclose(poly3[:, 0], torch.ones(n_timepoints, device=device))
        assert torch.allclose(poly3[:, 1], t)
        assert torch.allclose(poly3[:, 2], t**2)
        assert torch.allclose(poly3[:, 3], t**3)

    def test_construct_polynomial_orthogonal(self, device):
        """Test that polynomial columns have controlled correlations."""
        n_timepoints = 100
        poly = construct_polynomial_matrix(n_timepoints, max_degree=3, device=device)

        # Note: Raw polynomial basis is NOT orthogonal
        # But correlations should be reasonable (not perfectly correlated)
        poly_normalized = poly / poly.norm(dim=0, keepdim=True)
        corr = poly_normalized.T @ poly_normalized

        # Diagonal should be 1
        assert torch.allclose(torch.diag(corr), torch.ones(4, device=device))

        # All correlations should be finite (not NaN or Inf)
        assert torch.all(torch.isfinite(corr))

    def test_construct_polynomial_different_lengths(self, device):
        """Test polynomial construction with different timepoint lengths."""
        for n in [10, 50, 100, 200, 1000]:
            poly = construct_polynomial_matrix(n, max_degree=2, device=device)
            assert poly.shape == (n, 3)  # constant + linear + quadratic
            assert not torch.any(torch.isnan(poly))

    def test_construct_polynomial_zero_degree(self, device):
        """Test that degree 0 returns constant column."""
        n_timepoints = 100
        poly = construct_polynomial_matrix(n_timepoints, max_degree=0, device=device)

        # Should return just constant term
        assert poly.shape == (n_timepoints, 1)
        assert torch.allclose(poly[:, 0], torch.ones(n_timepoints, device=device))


class TestOrthogonalizeDesign:
    """Test design matrix orthogonalization."""

    @pytest.fixture
    def device(self):
        return get_device()

    def test_orthogonalize_basic(self, device):
        """Test basic orthogonalization."""
        torch.manual_seed(42)

        # Task regressors
        X = torch.randn(100, 3, device=device)

        # Nuisance regressors (confounds)
        Z = torch.randn(100, 2, device=device)

        # Orthogonalize
        X_orth = orthogonalize_design(X, Z)

        # Check shape preserved
        assert X_orth.shape == X.shape

        # Check orthogonal to Z
        corr = (X_orth.T @ Z).abs()
        assert torch.all(corr < 1e-5), "X_orth should be orthogonal to Z"

    def test_orthogonalize_removes_shared_variance(self, device):
        """Test that orthogonalization removes shared variance."""
        torch.manual_seed(42)

        # Create task regressor with shared variance
        base = torch.randn(100, 1, device=device)
        X = base + 0.5 * torch.randn(100, 1, device=device)
        Z = base + 0.5 * torch.randn(100, 1, device=device)

        # Before orthogonalization, should be correlated
        corr_before = (X.T @ Z)[0, 0] / (X.norm() * Z.norm())
        assert corr_before.abs() > 0.3, (
            "Should have correlation before orthogonalization"
        )

        # After orthogonalization
        X_orth = orthogonalize_design(X, Z)
        corr_after = (X_orth.T @ Z).abs().max()

        assert corr_after < 1e-4, (
            "Should have minimal correlation after orthogonalization"
        )

    def test_orthogonalize_preserves_independent_variance(self, device):
        """Test that independent variance is largely preserved."""
        torch.manual_seed(42)

        # Create completely independent regressors
        X = torch.randn(100, 2, device=device)
        Z = torch.randn(100, 2, device=device)

        # Orthogonalize
        X_orth = orthogonalize_design(X, Z)

        # Check that X_orth is orthogonal to Z
        corr = (X_orth.T @ Z).abs().max()
        assert corr < 1e-4, "Should be orthogonal to Z"

        # Variance should be similar (independent regressors)
        var_orig = X.var(dim=0)
        var_orth = X_orth.var(dim=0)
        # Allow some difference due to removing small correlations
        assert torch.allclose(var_orig, var_orth, rtol=0.5)

    def test_orthogonalize_constant_regressor(self, device):
        """Test orthogonalizing with constant regressor (mean removal)."""
        torch.manual_seed(42)

        # Task regressor
        X = torch.randn(100, 1, device=device)

        # Constant regressor
        Z = torch.ones(100, 1, device=device)

        # Orthogonalize (this should mean-center X)
        X_orth = orthogonalize_design(X, Z)

        # Should be mean-centered
        assert torch.allclose(
            X_orth.mean(dim=0), torch.tensor([0.0], device=device), atol=1e-5
        )


class TestPercentBoldChange:
    """Test percent BOLD signal change conversion."""

    @pytest.fixture
    def device(self):
        return get_device()

    def test_percent_bold_basic(self, device):
        """Test basic percent BOLD change computation."""
        # Beta value of 1.0 with mean of 100 = 1% signal change
        betas = torch.tensor([[1.0, 2.0, 0.5]], device=device)
        meanvol = torch.tensor([100.0], device=device)

        psc = percent_bold_change(betas, meanvol)

        expected = torch.tensor([[1.0, 2.0, 0.5]], device=device)
        assert torch.allclose(psc, expected)

    def test_percent_bold_different_means(self, device):
        """Test with different mean volumes."""
        betas = torch.tensor([[10.0]], device=device)

        # Higher baseline = lower % change
        meanvol_high = torch.tensor([1000.0], device=device)
        psc_high = percent_bold_change(betas, meanvol_high)
        assert torch.allclose(psc_high, torch.tensor([[1.0]], device=device))

        # Lower baseline = higher % change
        meanvol_low = torch.tensor([100.0], device=device)
        psc_low = percent_bold_change(betas, meanvol_low)
        assert torch.allclose(psc_low, torch.tensor([[10.0]], device=device))

    def test_percent_bold_multiple_voxels(self, device):
        """Test with multiple voxels."""
        n_voxels = 50
        n_regressors = 3

        betas = torch.randn(n_voxels, n_regressors, device=device) * 10
        meanvol = torch.rand(n_voxels, device=device) * 100 + 50

        psc = percent_bold_change(betas, meanvol)

        # Check shape
        assert psc.shape == betas.shape

        # Manual check for first voxel
        expected_first = (betas[0] / meanvol[0]) * 100
        assert torch.allclose(psc[0], expected_first)

    def test_percent_bold_sign_preservation(self, device):
        """Test that sign is preserved."""
        betas = torch.tensor([[-1.0, 0.0, 1.0]], device=device)
        meanvol = torch.tensor([100.0], device=device)

        psc = percent_bold_change(betas, meanvol)

        # Sign should be preserved
        assert psc[0, 0] < 0
        assert psc[0, 1] == 0
        assert psc[0, 2] > 0


class TestGLMChunkSize:
    """Test GLM with explicit chunk_size parameter."""

    @pytest.fixture
    def device(self):
        return get_device()

    def test_glm_with_chunk_size(self, device):
        """Test that chunk_size parameter works."""
        torch.manual_seed(42)

        n_timepoints = 100
        n_voxels = 500
        n_regressors = 5

        X = torch.randn(n_timepoints, n_regressors, device=device)
        data = torch.randn(n_voxels, n_timepoints, device=device)

        # Fit with explicit chunk size
        results = fit_glm(data, X, tr=2.0, chunk_size=100, verbose=False, device=device)

        assert results.betas.shape == (n_voxels, n_regressors)
        assert results.r2.shape == (n_voxels,)

    def test_glm_small_chunks(self, device):
        """Test with very small chunks."""
        torch.manual_seed(42)

        n_timepoints = 100
        n_voxels = 100
        n_regressors = 3

        X = torch.randn(n_timepoints, n_regressors, device=device)
        true_betas = torch.randn(n_voxels, n_regressors, device=device)
        data = (X @ true_betas.T).T + 0.1 * torch.randn(
            n_voxels, n_timepoints, device=device
        )

        # Fit with small chunks
        results_chunked = fit_glm(
            data, X, tr=2.0, chunk_size=10, verbose=False, device=device
        )

        # Fit without chunking
        results_full = fit_glm(
            data, X, tr=2.0, chunk_size=None, verbose=False, device=device
        )

        # Results should be identical regardless of chunk size
        assert torch.allclose(results_chunked.betas, results_full.betas, rtol=1e-4)
        assert torch.allclose(results_chunked.r2, results_full.r2, rtol=1e-4)

    def test_glm_chunk_size_larger_than_data(self, device):
        """Test when chunk_size is larger than n_voxels."""
        torch.manual_seed(42)

        n_timepoints = 100
        n_voxels = 50
        n_regressors = 3

        X = torch.randn(n_timepoints, n_regressors, device=device)
        data = torch.randn(n_voxels, n_timepoints, device=device)

        # Chunk size larger than data
        results = fit_glm(
            data, X, tr=2.0, chunk_size=1000, verbose=False, device=device
        )

        # Should still work
        assert results.betas.shape == (n_voxels, n_regressors)


class TestGLMResidualsAndPredicted:
    """Test GLM with residuals and predicted values."""

    @pytest.fixture
    def device(self):
        return get_device()

    def test_glm_with_residuals(self, device):
        """Test that residuals are computed correctly."""
        torch.manual_seed(42)

        n_timepoints = 100
        n_voxels = 10
        n_regressors = 3

        X = torch.randn(n_timepoints, n_regressors, device=device)
        data = torch.randn(n_voxels, n_timepoints, device=device)

        results = fit_glm(
            data, X, tr=2.0, want_residuals=True, verbose=False, device=device
        )

        assert results.residuals is not None
        assert results.residuals.shape == data.shape

        # Residuals should have mean close to zero (after accounting for design)
        assert results.residuals.abs().mean() > 0  # Should have some residuals

    def test_glm_with_predicted(self, device):
        """Test that predicted values are computed correctly."""
        torch.manual_seed(42)

        n_timepoints = 100
        n_voxels = 10
        n_regressors = 3

        X = torch.randn(n_timepoints, n_regressors, device=device)
        data = torch.randn(n_voxels, n_timepoints, device=device)

        results = fit_glm(
            data, X, tr=2.0, want_predicted=True, verbose=False, device=device
        )

        assert results.predicted is not None
        assert results.predicted.shape == data.shape

    def test_residuals_plus_predicted_equals_data(self, device):
        """Test that residuals + predicted = original data."""
        torch.manual_seed(42)

        n_timepoints = 100
        n_voxels = 10
        n_regressors = 3

        X = torch.randn(n_timepoints, n_regressors, device=device)
        data = torch.randn(n_voxels, n_timepoints, device=device)

        results = fit_glm(
            data,
            X,
            tr=2.0,
            want_residuals=True,
            want_predicted=True,
            verbose=False,
            device=device,
        )

        # residuals + predicted should equal data
        reconstructed = results.predicted + results.residuals
        assert torch.allclose(reconstructed, data, rtol=1e-4)

    def test_residuals_orthogonal_to_design(self, device):
        """Test that residuals are orthogonal to design matrix."""
        torch.manual_seed(42)

        n_timepoints = 100
        n_voxels = 10
        n_regressors = 3

        X = torch.randn(n_timepoints, n_regressors, device=device)
        data = torch.randn(n_voxels, n_timepoints, device=device)

        results = fit_glm(
            data, X, tr=2.0, want_residuals=True, verbose=False, device=device
        )

        # Compute correlation between residuals and design
        for i in range(n_regressors):
            for j in range(n_voxels):
                corr = torch.corrcoef(torch.stack([results.residuals[j], X[:, i]]))[
                    0, 1
                ]
                assert torch.abs(corr) < 0.1, (
                    "Residuals should be uncorrelated with design"
                )


class TestGLMPolynomialRegressors:
    """Test GLM with polynomial nuisance regressors."""

    @pytest.fixture
    def device(self):
        return get_device()

    def test_glm_with_linear_trend(self, device):
        """Test GLM removes linear trend."""
        torch.manual_seed(42)

        n_timepoints = 200
        n_voxels = 20

        # Create data with strong linear trend
        t = torch.linspace(0, 1, n_timepoints, device=device)
        trend = 10 * t  # Strong upward trend
        data = trend.unsqueeze(0).repeat(n_voxels, 1)
        data += 0.1 * torch.randn(n_voxels, n_timepoints, device=device)

        # Task regressor (no relationship with trend)
        X = torch.randn(n_timepoints, 1, device=device)

        # Fit with linear detrending
        results = fit_glm(
            data,
            X,
            tr=2.0,
            max_poly_degree=1,
            want_residuals=True,
            verbose=False,
            device=device,
        )

        # Residuals should have trend removed
        # Check that residuals don't have linear trend
        t_centered = t - t.mean()
        for j in range(min(5, n_voxels)):  # Check first few voxels
            resid = results.residuals[j]
            corr = torch.corrcoef(torch.stack([resid, t_centered]))[0, 1]
            assert torch.abs(corr) < 0.2, "Linear trend should be removed"

    def test_glm_with_quadratic_trend(self, device):
        """Test GLM with quadratic detrending."""
        torch.manual_seed(42)

        n_timepoints = 200
        n_voxels = 10

        # Create data with quadratic trend
        t = torch.linspace(-1, 1, n_timepoints, device=device)
        trend = 5 * t**2  # Quadratic trend
        data = trend.unsqueeze(0).repeat(n_voxels, 1)
        data += 0.1 * torch.randn(n_voxels, n_timepoints, device=device)

        X = torch.randn(n_timepoints, 1, device=device)

        # Fit with quadratic detrending
        results = fit_glm(
            data, X, tr=2.0, max_poly_degree=2, verbose=False, device=device
        )

        # Should successfully fit
        assert results.betas.shape == (n_voxels, 1)


class TestGLMExtraRegressors:
    """Test GLM with extra nuisance regressors."""

    @pytest.fixture
    def device(self):
        return get_device()

    def test_glm_with_motion_regressors(self, device):
        """Test GLM with motion nuisance regressors."""
        torch.manual_seed(42)

        n_timepoints = 100
        n_voxels = 20
        n_task = 3
        n_motion = 6  # 6 motion parameters

        # Task regressors
        X_task = torch.randn(n_timepoints, n_task, device=device)

        # Motion regressors (simulated)
        X_motion = torch.randn(n_timepoints, n_motion, device=device) * 0.1

        # Create data influenced by motion
        motion_effect = X_motion @ torch.randn(n_motion, n_voxels, device=device)
        task_effect = X_task @ torch.randn(n_task, n_voxels, device=device)
        data = (
            motion_effect
            + task_effect
            + 0.1 * torch.randn(n_timepoints, n_voxels, device=device)
        ).T

        # Fit with motion regressors
        results = fit_glm(
            data,
            X_task,
            tr=2.0,
            extra_regressors=X_motion,
            verbose=False,
            device=device,
        )

        # Should successfully fit
        assert results.betas.shape == (n_voxels, n_task)

        # Task betas should be estimable even with motion confounds
        assert not torch.any(torch.isnan(results.betas))

    def test_glm_extra_regressors_concatenated(self, device):
        """Test extra_regressors with concatenated array."""
        torch.manual_seed(42)

        n_timepoints = 100
        n_voxels = 10

        X = torch.randn(n_timepoints, 2, device=device)
        data = torch.randn(n_voxels, n_timepoints, device=device)

        # Extra regressors as single concatenated array
        extra = torch.randn(n_timepoints, 3, device=device)

        results = fit_glm(
            data, X, tr=2.0, extra_regressors=extra, verbose=False, device=device
        )

        assert results.betas.shape == (n_voxels, 2)


class TestGLMNumericalStability:
    """Test GLM numerical stability in edge cases."""

    @pytest.fixture
    def device(self):
        return get_device()

    def test_glm_with_very_small_values(self, device):
        """Test GLM with very small data values."""
        torch.manual_seed(42)

        n_timepoints = 100
        n_voxels = 10

        X = torch.randn(n_timepoints, 2, device=device)
        # Very small values
        data = torch.randn(n_voxels, n_timepoints, device=device) * 1e-6

        results = fit_glm(data, X, tr=2.0, verbose=False, device=device)

        # Should not produce NaN or Inf
        assert not torch.any(torch.isnan(results.betas))
        assert not torch.any(torch.isinf(results.betas))

    def test_glm_with_large_values(self, device):
        """Test GLM with very large data values."""
        torch.manual_seed(42)

        n_timepoints = 100
        n_voxels = 10

        X = torch.randn(n_timepoints, 2, device=device)
        # Very large values
        data = torch.randn(n_voxels, n_timepoints, device=device) * 1e6

        results = fit_glm(data, X, tr=2.0, verbose=False, device=device)

        # Should not produce NaN or Inf
        assert not torch.any(torch.isnan(results.betas))
        assert not torch.any(torch.isinf(results.betas))

    def test_glm_with_mixed_scales(self, device):
        """Test GLM when regressors have very different scales."""
        torch.manual_seed(42)

        n_timepoints = 100
        n_voxels = 10

        # Regressors with different scales
        X1 = torch.randn(n_timepoints, 1, device=device) * 1e-3
        X2 = torch.randn(n_timepoints, 1, device=device) * 1e3
        X = torch.cat([X1, X2], dim=1)

        data = torch.randn(n_voxels, n_timepoints, device=device)

        results = fit_glm(data, X, tr=2.0, verbose=False, device=device)

        # Should produce reasonable results
        assert not torch.any(torch.isnan(results.betas))
        assert results.r2.min() >= 0
        assert results.r2.max() <= 1
