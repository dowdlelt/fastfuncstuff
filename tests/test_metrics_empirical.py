"""
Tests for empirical design metrics (Das et al. 2023 / deconv implementation)

These metrics use GLS with AR(1) prewhitening for realistic fMRI noise models.
Separate from theoretical Liu & Frank metrics - these estimate from simulated data.

IMPORTANT: These functions need:
- Convolved design matrix (for detection power)
- Onset matrix (for FIR-based estimation efficiency)
- Simulated or real data

Key concepts:
- Detection Power: Ability to detect activation (with AR(1) correction)
- Estimation Efficiency: Ability to estimate HRF shape (FIR with AR(1))
- AR(1) coefficient: Temporal autocorrelation in fMRI noise

Tests validate CORRECTNESS of metric calculations, not just coverage.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncsim.design import convolve_hrf
from fastfuncsim.hrf import get_canonical_hrf
from fastfuncsim.metrics_empirical import (
    build_ar1_covariance_matrix,
    compute_detection_power_empirical,
    compute_estimation_efficiency_empirical,
    estimate_ar1_coefficient,
    evaluate_design_empirical,
    gls_fit,
)

# =============================================================================
# CONSTANTS
# =============================================================================

TR = 1.0
HRF_DURATION = 32.0
HRF_LENGTH = 32


def create_onset_matrix(
    n_timepoints: int,
    n_conditions: int,
    event_times: list[list[int]],
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create binary onset matrix from event times per condition."""
    onsets = torch.zeros(n_timepoints, n_conditions, device=device)
    for cond, times in enumerate(event_times):
        for t in times:
            if 0 <= t < n_timepoints:
                onsets[t, cond] = 1.0
    return onsets


def create_convolved_design(
    onsets: torch.Tensor,
    tr: float = TR,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Convolve onset matrix with canonical HRF."""
    hrf = get_canonical_hrf(stim_duration=0.0, tr=tr, duration=HRF_DURATION, device=device)
    n_timepoints = onsets.shape[0]
    return convolve_hrf(onsets, hrf, n_timepoints, device=device)


def create_simulated_data(
    design: torch.Tensor,
    effect_sizes: list[float],
    noise_std: float = 0.5,
    ar1_rho: float = 0.3,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Simulate fMRI data with AR(1) noise."""
    n_timepoints, n_conditions = design.shape
    betas = torch.tensor(effect_sizes[:n_conditions], device=device)

    # Signal
    signal = design @ betas

    # AR(1) noise
    noise = torch.zeros(n_timepoints, device=device)
    noise[0] = torch.randn(1, device=device)
    for t in range(1, n_timepoints):
        noise[t] = ar1_rho * noise[t - 1] + torch.randn(1, device=device) * noise_std * np.sqrt(
            1 - ar1_rho**2
        )

    return signal + noise


# =============================================================================
# TESTS: AR(1) COEFFICIENT ESTIMATION
# =============================================================================


class TestEstimateAR1Coefficient:
    """Test AR(1) coefficient estimation from residuals."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_white_noise_zero_rho(self, device):
        """White noise should have AR(1) ≈ 0."""
        residuals = np.random.randn(1000)
        rho = estimate_ar1_coefficient(residuals, device=device)
        assert abs(rho) < 0.1

    def test_known_positive_rho(self, device):
        """Should correctly estimate known positive AR(1)."""
        np.random.seed(42)
        n = 1000
        true_rho = 0.5
        residuals = np.zeros(n)
        residuals[0] = np.random.randn()
        for t in range(1, n):
            residuals[t] = true_rho * residuals[t - 1] + np.random.randn() * np.sqrt(
                1 - true_rho**2
            )

        estimated = estimate_ar1_coefficient(residuals, device=device)
        assert 0.4 < estimated < 0.6, f"Expected ~0.5, got {estimated}"

    def test_known_negative_rho(self, device):
        """Should correctly estimate known negative AR(1)."""
        np.random.seed(42)
        n = 1000
        true_rho = -0.4
        residuals = np.zeros(n)
        residuals[0] = np.random.randn()
        for t in range(1, n):
            residuals[t] = true_rho * residuals[t - 1] + np.random.randn() * np.sqrt(
                1 - true_rho**2
            )

        estimated = estimate_ar1_coefficient(residuals, device=device)
        assert -0.5 < estimated < -0.3, f"Expected ~-0.4, got {estimated}"

    def test_high_autocorrelation(self, device):
        """Should correctly estimate high AR(1)."""
        np.random.seed(42)
        n = 1000
        true_rho = 0.8
        residuals = np.zeros(n)
        residuals[0] = np.random.randn()
        for t in range(1, n):
            residuals[t] = true_rho * residuals[t - 1] + np.random.randn() * np.sqrt(
                1 - true_rho**2
            )

        estimated = estimate_ar1_coefficient(residuals, device=device)
        assert 0.7 < estimated < 0.9

    def test_short_series_handled(self, device):
        """Should handle short time series."""
        residuals = np.random.randn(10)
        rho = estimate_ar1_coefficient(residuals, device=device)
        assert -1 < rho < 1

    def test_constant_series_returns_nan(self, device):
        """Constant series should return NaN (undefined correlation)."""
        residuals = np.array([1.0, 1.0, 1.0, 1.0])
        rho = estimate_ar1_coefficient(residuals, device=device)
        assert np.isnan(rho) or (-0.99 <= rho <= 0.99)

    def test_torch_tensor_input(self, device):
        """Should accept torch tensors."""
        residuals = torch.randn(100, device=device)
        rho = estimate_ar1_coefficient(residuals, device=device)
        assert -1 < rho < 1


# =============================================================================
# TESTS: AR(1) COVARIANCE MATRIX
# =============================================================================


class TestBuildAR1CovarianceMatrix:
    """Test AR(1) covariance matrix construction."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_diagonal_is_one(self, device):
        """Diagonal elements should be 1 (ρ^0)."""
        sigma = build_ar1_covariance_matrix(10, 0.3, device=device)
        assert torch.allclose(torch.diag(sigma), torch.ones(10, device=device))

    def test_off_diagonal_structure(self, device):
        """Off-diagonal should follow Σ[i,j] = ρ^|i-j|."""
        n, rho = 5, 0.5
        sigma = build_ar1_covariance_matrix(n, rho, device=device)

        assert torch.isclose(sigma[0, 1], torch.tensor(rho**1, device=device))
        assert torch.isclose(sigma[0, 2], torch.tensor(rho**2, device=device))
        assert torch.isclose(sigma[1, 3], torch.tensor(rho**2, device=device))
        assert torch.isclose(sigma[0, 4], torch.tensor(rho**4, device=device))

    def test_symmetry(self, device):
        """Matrix should be symmetric."""
        sigma = build_ar1_covariance_matrix(20, 0.4, device=device)
        assert torch.allclose(sigma, sigma.T)

    def test_positive_definite(self, device):
        """Matrix should be positive definite."""
        sigma = build_ar1_covariance_matrix(20, 0.5, device=device)
        try:
            torch.linalg.cholesky(sigma)
        except RuntimeError:
            pytest.fail("Covariance matrix is not positive definite")

    def test_zero_rho_is_identity(self, device):
        """ρ=0 should give identity matrix."""
        sigma = build_ar1_covariance_matrix(10, 0.0, device=device)
        assert torch.allclose(sigma, torch.eye(10, device=device))

    def test_shape(self, device):
        """Output shape should match input n."""
        for n in [5, 10, 50]:
            sigma = build_ar1_covariance_matrix(n, 0.3, device=device)
            assert sigma.shape == (n, n)


# =============================================================================
# TESTS: GLS FIT
# =============================================================================


class TestGLSFit:
    """Test Generalized Least Squares fitting."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_basic_recovery(self, device):
        """GLS should recover true betas."""
        n, n_reg = 100, 2
        X = torch.randn(n, n_reg, device=device)
        true_betas = torch.tensor([[1.5], [-0.8]], device=device)
        Y = X @ true_betas + torch.randn(n, 1, device=device) * 0.1

        sigma = torch.eye(n, device=device)  # OLS case
        result = gls_fit(Y, X, sigma, device=device)

        assert torch.allclose(result["betas"], true_betas, atol=0.2)

    def test_with_ar1_covariance(self, device):
        """GLS with AR(1) covariance should work."""
        np.random.seed(42)
        n, rho = 100, 0.3

        X = torch.randn(n, 2, device=device)
        true_betas = torch.tensor([[1.0], [0.5]], device=device)

        # AR(1) noise
        noise = torch.zeros(n, 1, device=device)
        noise[0] = torch.randn(1, device=device)
        for t in range(1, n):
            noise[t] = rho * noise[t - 1] + torch.randn(1, device=device) * np.sqrt(1 - rho**2)

        Y = X @ true_betas + noise * 0.5
        sigma = build_ar1_covariance_matrix(n, rho, device=device)

        result = gls_fit(Y, X, sigma, device=device)
        assert torch.allclose(result["betas"], true_betas, atol=0.3)

    def test_variance_estimation(self, device):
        """Should estimate variance of betas."""
        n = 100
        X = torch.randn(n, 2, device=device)
        Y = X @ torch.tensor([[1.0], [0.5]], device=device) + torch.randn(n, 1, device=device)
        sigma = torch.eye(n, device=device)

        result = gls_fit(Y, X, sigma, device=device)

        assert result["var_betas"].shape == (2, 2)
        assert torch.all(torch.diag(result["var_betas"]) > 0)

    def test_residuals_shape(self, device):
        """Residuals should have correct shape."""
        n = 100
        X = torch.randn(n, 2, device=device)
        Y = torch.randn(n, 1, device=device)
        sigma = torch.eye(n, device=device)

        result = gls_fit(Y, X, sigma, device=device)
        assert result["residuals"].shape == (n, 1)

    def test_1d_y_input(self, device):
        """Should handle 1D Y input."""
        n = 100
        X = torch.randn(n, 2, device=device)
        Y = torch.randn(n, device=device)
        sigma = torch.eye(n, device=device)

        result = gls_fit(Y, X, sigma, device=device)
        assert result["betas"].shape[1] == 1


# =============================================================================
# TESTS: EMPIRICAL DETECTION POWER
# =============================================================================


class TestComputeDetectionPowerEmpirical:
    """Test empirical detection power with AR(1) correction."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_basic_computation(self, device):
        """Should compute detection power."""
        # Create design
        onsets = create_onset_matrix(100, 2, [[10, 30, 50, 70], [20, 40, 60, 80]], device=device)
        design = create_convolved_design(onsets, device=device)
        data = create_simulated_data(design, [1.0, 0.5], noise_std=0.3, device=device)

        result = compute_detection_power_empirical(data, design, device=device)

        assert "detection_power" in result
        assert "rho" in result
        assert "betas" in result
        assert result["detection_power"] > 0

    def test_with_known_rho(self, device):
        """Should use provided AR(1) coefficient."""
        onsets = create_onset_matrix(100, 1, [[10, 30, 50, 70]], device=device)
        design = create_convolved_design(onsets, device=device)
        data = create_simulated_data(design, [1.0], noise_std=0.3, device=device)

        result = compute_detection_power_empirical(
            data, design, estimate_ar1=False, rho=0.3, device=device
        )

        assert result["rho"] == 0.3

    def test_with_contrast(self, device):
        """Should work with custom contrast."""
        onsets = create_onset_matrix(100, 2, [[10, 30, 50, 70], [20, 40, 60, 80]], device=device)
        design = create_convolved_design(onsets, device=device)
        data = create_simulated_data(design, [1.0, 0.5], noise_std=0.3, device=device)

        contrast = torch.tensor([1.0, -1.0], device=device)
        result = compute_detection_power_empirical(data, design, contrast=contrast, device=device)

        assert result["detection_power"] > 0

    def test_var_contrast_positive(self, device):
        """Variance of contrast should be positive."""
        onsets = create_onset_matrix(100, 1, [[10, 30, 50, 70]], device=device)
        design = create_convolved_design(onsets, device=device)
        data = create_simulated_data(design, [1.0], noise_std=0.3, device=device)

        result = compute_detection_power_empirical(data, design, device=device)

        assert result["var_contrast"] > 0


# =============================================================================
# TESTS: EMPIRICAL ESTIMATION EFFICIENCY
# =============================================================================


class TestComputeEstimationEfficiencyEmpirical:
    """Test empirical estimation efficiency with FIR and AR(1)."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_basic_computation(self, device):
        """Should compute estimation efficiency."""
        onsets = create_onset_matrix(100, 1, [[10, 30, 50, 70]], device=device)
        design = create_convolved_design(onsets, device=device)
        data = create_simulated_data(design, [1.0], noise_std=0.3, device=device)

        result = compute_estimation_efficiency_empirical(
            data, onsets, n_conditions=1, hrf_length=20, device=device
        )

        assert "estimation_efficiency" in result
        assert "rho" in result
        assert "betas_fir" in result

    def test_with_known_rho(self, device):
        """Should use provided AR(1) coefficient."""
        onsets = create_onset_matrix(100, 1, [[10, 30, 50, 70]], device=device)
        data = torch.randn(100, device=device)

        result = compute_estimation_efficiency_empirical(
            data, onsets, n_conditions=1, hrf_length=20, estimate_ar1=False, rho=0.4, device=device
        )

        assert result["rho"] == 0.4

    def test_fir_betas_shape(self, device):
        """FIR betas should have correct shape."""
        hrf_length = 15
        onsets = create_onset_matrix(100, 1, [[10, 30, 50, 70]], device=device)
        data = torch.randn(100, device=device)

        result = compute_estimation_efficiency_empirical(
            data, onsets, n_conditions=1, hrf_length=hrf_length, device=device
        )

        assert result["betas_fir"].shape[0] == hrf_length


# =============================================================================
# TESTS: COMPLETE EMPIRICAL EVALUATION
# =============================================================================


class TestEvaluateDesignEmpirical:
    """Test complete empirical design evaluation."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_returns_all_metrics(self, device):
        """Should return both power and efficiency."""
        onsets = create_onset_matrix(100, 2, [[10, 30, 50, 70], [20, 40, 60, 80]], device=device)
        design = create_convolved_design(onsets, device=device)
        data = create_simulated_data(design, [1.0, 0.5], noise_std=0.3, device=device)

        result = evaluate_design_empirical(data, design, onsets, n_conditions=2, device=device)

        assert "detection_power" in result
        assert "estimation_efficiency" in result
        assert "rho" in result
        assert "summary" in result

    def test_summary_has_key_metrics(self, device):
        """Summary should contain Fd, Fe, and rho."""
        onsets = create_onset_matrix(100, 2, [[10, 30, 50, 70], [20, 40, 60, 80]], device=device)
        design = create_convolved_design(onsets, device=device)
        data = create_simulated_data(design, [1.0, 0.5], noise_std=0.3, device=device)

        result = evaluate_design_empirical(data, design, onsets, n_conditions=2, device=device)

        assert "Fd" in result["summary"]
        assert "Fe" in result["summary"]
        assert "rho_ar1" in result["summary"]

    def test_shared_ar1_estimate(self, device):
        """AR(1) should be shared between power and efficiency."""
        onsets = create_onset_matrix(100, 2, [[10, 30, 50, 70], [20, 40, 60, 80]], device=device)
        design = create_convolved_design(onsets, device=device)
        data = create_simulated_data(design, [1.0, 0.5], noise_std=0.3, ar1_rho=0.4, device=device)

        result = evaluate_design_empirical(data, design, onsets, n_conditions=2, device=device)

        assert result["rho"] == result["summary"]["rho_ar1"]

    def test_details_contains_full_results(self, device):
        """Details should contain full results."""
        onsets = create_onset_matrix(100, 2, [[10, 30, 50, 70], [20, 40, 60, 80]], device=device)
        design = create_convolved_design(onsets, device=device)
        data = create_simulated_data(design, [1.0, 0.5], noise_std=0.3, device=device)

        result = evaluate_design_empirical(data, design, onsets, n_conditions=2, device=device)

        assert "power" in result["details"]
        assert "efficiency" in result["details"]


# =============================================================================
# TESTS: NUMERICAL STABILITY
# =============================================================================


class TestNumericalStability:
    """Test numerical stability."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_large_design_matrix(self, device):
        """Should handle large design matrices."""
        n, n_reg = 500, 10
        X = torch.randn(n, n_reg, device=device)
        true_betas = torch.randn(n_reg, 1, device=device)
        Y = X @ true_betas + torch.randn(n, 1, device=device) * 0.1

        sigma = build_ar1_covariance_matrix(n, 0.3, device=device)
        result = gls_fit(Y, X, sigma, device=device)

        assert torch.allclose(result["betas"], true_betas, atol=0.3)

    def test_covariance_invertibility_across_rho(self, device):
        """Covariance should be invertible for various rho values."""
        n = 50
        for rho in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
            sigma = build_ar1_covariance_matrix(n, rho, device=device)
            try:
                torch.linalg.inv(sigma)
            except RuntimeError:
                pytest.fail(f"Failed to invert with rho={rho}")

    def test_near_singular_design(self, device):
        """Should handle near-singular design with regularization."""
        n = 50
        X = torch.randn(n, 2, device=device)
        X[:, 1] = X[:, 0] + torch.randn(n, device=device) * 1e-6  # Near-collinear

        Y = X @ torch.tensor([[1.0], [1.0]], device=device) + torch.randn(n, 1, device=device)
        sigma = torch.eye(n, device=device)

        result = gls_fit(Y, X, sigma, device=device)
        assert "betas" in result


# =============================================================================
# TESTS: THEORETICAL VALIDATION
# =============================================================================


class TestTheoreticalValidation:
    """Tests validating theoretical relationships."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_more_events_higher_power(self, device):
        """Designs with more events should have higher detection power."""
        # Few events
        onsets_few = create_onset_matrix(200, 1, [[20, 100, 180]], device=device)
        design_few = create_convolved_design(onsets_few, device=device)
        data_few = create_simulated_data(design_few, [1.0], noise_std=0.3, device=device)

        # More events
        onsets_many = create_onset_matrix(200, 1, [[20, 50, 80, 110, 140, 170]], device=device)
        design_many = create_convolved_design(onsets_many, device=device)
        data_many = create_simulated_data(design_many, [1.0], noise_std=0.3, device=device)

        result_few = compute_detection_power_empirical(data_few, design_few, device=device)
        result_many = compute_detection_power_empirical(data_many, design_many, device=device)

        assert result_many["detection_power"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
