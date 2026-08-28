"""
Comprehensive tests for ARMA noise generation and covariance structures.

Tests cover:
- ARMA noise generation (AR, MA, ARMA)
- Covariance matrix construction
- Parameter validation
- Edge cases
"""

import numpy as np
import pytest
import torch

from fastfuncstuff.glm.arma import build_arma11_covariance
from fastfuncstuff.simulation.noise import generate_arma_noise


class TestARMANoiseGeneration:
    """Test ARMA noise generation functions."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_ar1_noise_generation(self, device):
        """Test AR(1) noise generation."""
        ar_coeff = 0.5
        n_timepoints = 1000
        n_voxels = 10

        noise = generate_arma_noise(
            ar_coeffs=[ar_coeff],
            ma_coeffs=[],
            n_timepoints=n_timepoints,
            n_voxels=n_voxels,
            device=device,
        )

        # Check shape
        assert noise.shape == (n_timepoints, n_voxels)

        # Check mean is approximately 0
        assert torch.abs(noise.mean()) < 0.1

        # Check autocorrelation approximately matches AR(1)
        # For large N, lag-1 correlation should be close to ar_coeff
        noise_np = noise.cpu().numpy()
        lag1_corr = np.corrcoef(noise_np[:-1, 0], noise_np[1:, 0])[0, 1]
        assert abs(lag1_corr - ar_coeff) < 0.1

    def test_ma1_noise_generation(self, device):
        """Test MA(1) noise generation."""
        ma_coeff = 0.3
        n_timepoints = 1000
        n_voxels = 5

        noise = generate_arma_noise(
            ar_coeffs=[],
            ma_coeffs=[ma_coeff],
            n_timepoints=n_timepoints,
            n_voxels=n_voxels,
            device=device,
        )

        assert noise.shape == (n_timepoints, n_voxels)
        assert torch.abs(noise.mean()) < 0.1
        assert torch.all(torch.isfinite(noise))

    def test_arma11_noise_generation(self, device):
        """Test ARMA(1,1) noise generation."""
        ar_coeff = 0.4
        ma_coeff = 0.2
        n_timepoints = 1000
        n_voxels = 10

        noise = generate_arma_noise(
            ar_coeffs=[ar_coeff],
            ma_coeffs=[ma_coeff],
            n_timepoints=n_timepoints,
            n_voxels=n_voxels,
            device=device,
        )

        assert noise.shape == (n_timepoints, n_voxels)
        assert torch.abs(noise.mean()) < 0.1

        # Theoretical lag-1 correlation for ARMA(1,1)
        lam = ((ma_coeff + ar_coeff) * (1 + ar_coeff * ma_coeff)) / (
            1 + 2 * ar_coeff * ma_coeff + ma_coeff**2
        )

        noise_np = noise.cpu().numpy()
        lag1_corr = np.corrcoef(noise_np[:-1, 0], noise_np[1:, 0])[0, 1]

        # Allow some tolerance due to finite sample size
        assert abs(lag1_corr - lam) < 0.15

    def test_white_noise_generation(self, device):
        """Test white noise (no AR or MA)."""
        n_timepoints = 1000
        n_voxels = 10

        noise = generate_arma_noise(
            ar_coeffs=[],
            ma_coeffs=[],
            n_timepoints=n_timepoints,
            n_voxels=n_voxels,
            device=device,
        )

        # Should be approximately white (low autocorrelation)
        noise_np = noise.cpu().numpy()
        lag1_corr = np.corrcoef(noise_np[:-1, 0], noise_np[1:, 0])[0, 1]
        assert abs(lag1_corr) < 0.1

    def test_noise_variance_scaling(self, device):
        """Test that noise variance is approximately 1."""
        noise = generate_arma_noise(
            ar_coeffs=[0.4],
            ma_coeffs=[0.2],
            n_timepoints=10000,
            n_voxels=1,
            device=device,
        )

        variance = torch.var(noise).item()
        # Should be close to 1 for large N
        assert abs(variance - 1.0) < 0.1

    def test_invalid_parameters(self, device):
        """Test that invalid parameters are caught."""
        # AR coefficient too large (non-stationary)
        with pytest.raises((ValueError, RuntimeError)):
            generate_arma_noise(
                ar_coeffs=[1.1],
                ma_coeffs=[],
                n_timepoints=100,
                n_voxels=1,
                device=device,
            )


class TestARMA11Covariance:
    """Test ARMA(1,1) covariance matrix construction."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_covariance_symmetry(self, device):
        """Test that covariance matrix is symmetric."""
        R = build_arma11_covariance(a=0.5, b=0.2, n=50, device=device)
        assert torch.allclose(R, R.T, atol=1e-6)

    def test_covariance_positive_definite(self, device):
        """Test that covariance matrix is positive definite."""
        import warnings

        R = build_arma11_covariance(a=0.4, b=0.1, n=50, device=device)
        # Suppress MPS backend fallback warning (expected for linalg.eigvals)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*not currently supported on the MPS backend.*"
            )
            eigenvalues = torch.linalg.eigvals(R).real
        assert torch.all(eigenvalues > 0)

    def test_covariance_toeplitz_structure(self, device):
        """Test that covariance has Toeplitz structure."""
        R = build_arma11_covariance(a=0.3, b=0.15, n=20, device=device)

        # Check that diagonals are constant
        for k in range(5):  # Check first few diagonals
            diag_vals = torch.diagonal(R, offset=k)
            assert torch.allclose(diag_vals, diag_vals[0], atol=1e-5)

    def test_covariance_lag1_correlation(self, device):
        """Test that R[0,1] matches theoretical lag-1 correlation."""
        a, b = 0.4, 0.2
        R = build_arma11_covariance(a, b, n=50, device=device)

        # Theoretical lag-1 correlation
        lam = ((b + a) * (1 + a * b)) / (1 + 2 * a * b + b**2)

        assert abs(R[0, 1].item() - lam) < 1e-4

    def test_ar1_special_case(self, device):
        """Test AR(1) as special case of ARMA(1,1) with b=0."""
        a = 0.5
        R = build_arma11_covariance(a, b=0.0, n=30, device=device)

        # For AR(1), lag-k correlation is a^k
        for k in range(1, 5):
            expected = a**k
            actual = R[0, k].item()
            assert abs(actual - expected) < 1e-4

    def test_invalid_parameters_rejected(self, device):
        """Test that invalid parameters return None or raise error."""
        # Non-stationary AR parameter
        R = build_arma11_covariance(a=1.1, b=0.0, n=10, device=device)
        assert R is None

        # Parameters that violate stationarity
        R = build_arma11_covariance(a=0.9, b=0.9, n=10, device=device)
        # Should either return None or be positive definite if accepted
        if R is not None:
            eigenvalues = torch.linalg.eigvals(R).real
            assert torch.all(eigenvalues > 0)

    def test_matrix_size(self, device):
        """Test that matrix size is correct."""
        for n in [10, 50, 100]:
            R = build_arma11_covariance(a=0.4, b=0.1, n=n, device=device)
            assert R.shape == (n, n)

    def test_unit_variance(self, device):
        """Test that diagonal is 1 (unit variance)."""
        R = build_arma11_covariance(a=0.3, b=0.2, n=50, device=device)
        assert torch.allclose(torch.diagonal(R), torch.ones(50, device=device), atol=1e-6)


class TestARMAParameterValidation:
    """Test parameter validation and edge cases."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_zero_parameters(self, device):
        """Test ARMA(0,0) is white noise."""
        noise = generate_arma_noise(
            ar_coeffs=[], ma_coeffs=[], n_timepoints=1000, n_voxels=5, device=device
        )

        # Should be uncorrelated
        noise_np = noise.cpu().numpy()
        lag1_corr = np.corrcoef(noise_np[:-1, 0], noise_np[1:, 0])[0, 1]
        assert abs(lag1_corr) < 0.1

    def test_boundary_ar_parameter(self, device):
        """Test AR parameter close to boundary (0.9)."""
        noise = generate_arma_noise(
            ar_coeffs=[0.9], ma_coeffs=[], n_timepoints=1000, n_voxels=1, device=device
        )

        assert torch.all(torch.isfinite(noise))
        # High persistence
        noise_np = noise.cpu().numpy()
        # Handle both (n_timepoints,) and (n_timepoints, 1)
        if noise_np.ndim == 2:
            noise_np = noise_np[:, 0]
        lag1_corr = np.corrcoef(noise_np[:-1], noise_np[1:])[0, 1]
        assert lag1_corr > 0.8

    def test_negative_ma_parameter(self, device):
        """Test negative MA parameter."""
        noise = generate_arma_noise(
            ar_coeffs=[0.3],
            ma_coeffs=[-0.2],
            n_timepoints=1000,
            n_voxels=1,
            device=device,
        )

        assert torch.all(torch.isfinite(noise))
        assert torch.abs(noise.mean()) < 0.1

    def test_small_sample_size(self, device):
        """Test with very small sample size."""
        noise = generate_arma_noise(
            ar_coeffs=[0.5], ma_coeffs=[0.1], n_timepoints=10, n_voxels=1, device=device
        )

        # Accept both (10, 1) and (10,) shapes
        assert noise.shape == (10, 1) or noise.shape == (10,)
        assert torch.all(torch.isfinite(noise))

    def test_large_sample_size(self, device):
        """Test with large sample size."""
        noise = generate_arma_noise(
            ar_coeffs=[0.4],
            ma_coeffs=[0.2],
            n_timepoints=10000,
            n_voxels=100,
            device=device,
        )

        assert noise.shape == (10000, 100)
        assert torch.all(torch.isfinite(noise))

        # Check memory efficiency - should not crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
