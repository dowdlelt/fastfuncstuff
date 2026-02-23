"""
Comprehensive tests for ICA utility functions in ica_tools.py.

Tests cover:
- Component count parsing
- Polynomial projection
- High-pass filtering
- MELODIC variance normalization
- Effective rank estimation
- Marchenko-Pastur spike detection
"""

import numpy as np
import pytest
import torch

from fastfuncsim.glm_core import construct_polynomial_matrix
from fastfuncsim.ica_tools import (
    apply_high_pass_fft,
    apply_melodic_voxel_varnorm,
    apply_polort_projection,
    effective_rank_from_spectrum,
    mp_spikes_from_spectrum,
    parse_num_comps_spec,
)


class TestParseNumCompsSpec:
    """Test component count specification parsing."""

    def test_parse_integer(self):
        """Test parsing integer component count."""
        result = parse_num_comps_spec("20")
        assert result == 20

    def test_parse_float(self):
        """Test parsing float (fraction) component count."""
        result = parse_num_comps_spec("0.5")
        assert result == 0.5

    def test_parse_scientific_notation(self):
        """Test parsing scientific notation."""
        result = parse_num_comps_spec("1e-1")
        assert result == 1e-1

    def test_parse_auto(self):
        """Test parsing 'auto' keyword."""
        result = parse_num_comps_spec("auto")
        assert result == "auto"

    def test_parse_melodic(self):
        """Test parsing 'melodic' keyword."""
        result = parse_num_comps_spec("melodic")
        assert result == "melodic"

    def test_parse_hybrid(self):
        """Test parsing 'hybrid' keyword."""
        result = parse_num_comps_spec("hybrid")
        assert result == "hybrid"

    def test_parse_current(self):
        """Test parsing 'current' keyword."""
        result = parse_num_comps_spec("current")
        assert result == "current"

    def test_parse_erank(self):
        """Test parsing 'erank' keyword."""
        result = parse_num_comps_spec("erank")
        assert result == "erank"

    def test_parse_mp(self):
        """Test parsing 'mp' keyword."""
        result = parse_num_comps_spec("mp")
        assert result == "mp"

    def test_parse_case_insensitive(self):
        """Test case-insensitive parsing."""
        result = parse_num_comps_spec("AUTO")
        assert result == "auto"

        result = parse_num_comps_spec("MeLoDiC")
        assert result == "melodic"

        result = parse_num_comps_spec("  Auto  ")
        assert result == "auto"

    def test_parse_invalid_spec(self):
        """Test invalid specification raises ValueError."""
        with pytest.raises(ValueError, match="Invalid -num_comps"):
            parse_num_comps_spec("invalid")

        with pytest.raises(ValueError, match="Invalid -num_comps"):
            parse_num_comps_spec("abc")


class TestApplyPolortProjection:
    """Test polynomial trend projection."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_no_projection_when_polort_negative(self, device):
        """Test that polort < 0 returns data unchanged."""
        data = torch.randn(10, 100, device=device)
        result = apply_polort_projection(data, polort=-1, device=device)
        assert torch.equal(result, data)

    def test_no_projection_when_polort_none(self, device):
        """Test that polort=None returns data unchanged."""
        data = torch.randn(10, 100, device=device)
        result = apply_polort_projection(data, polort=None, device=device)
        assert torch.equal(result, data)

    def test_removes_linear_trend(self, device):
        """Test that polort=1 removes linear trend."""
        # Create data with linear trend
        n_vox = 5
        n_time = 50
        trend = torch.linspace(0, 10, n_time, device=device)
        data = trend.unsqueeze(0).expand(n_vox, -1) + 0.1 * torch.randn(n_vox, n_time, device=device)

        result = apply_polort_projection(data, polort=1, device=device)

        # Check that trend is removed (low correlation with original trend)
        for i in range(n_vox):
            corr = torch.corrcoef(torch.stack([trend, result[i]]))[0, 1]
            assert abs(corr) < 0.3, f"Voxel {i} still has trend: corr={corr}"

    def test_removes_quadratic_trend(self, device):
        """Test that polort=2 removes quadratic trend."""
        n_vox = 3
        n_time = 50
        t = torch.linspace(-1, 1, n_time, device=device)
        quadratic = 5 * t**2 + 2 * t + 1

        data = quadratic.unsqueeze(0).expand(n_vox, -1) + 0.01 * torch.randn(n_vox, n_time, device=device)

        # Compute correlation with quadratic trend before projection
        data_normalized = data[0] - data[0].mean()
        quad_normalized = quadratic - quadratic.mean()
        corr_before = (data_normalized * quad_normalized).sum() / (data_normalized.norm() * quad_normalized.norm())

        result = apply_polort_projection(data, polort=2, device=device)

        # Compute correlation after projection
        result_normalized = result[0] - result[0].mean()
        corr_after = (result_normalized * quad_normalized).sum() / (result_normalized.norm() * quad_normalized.norm() + 1e-10)

        # Correlation with quadratic trend should be greatly reduced
        assert abs(corr_after) < 0.3, \
            f"Quadratic trend not removed: corr_before={corr_before:.3f}, corr_after={corr_after:.3f}"

    def test_preserves_data_shape(self, device):
        """Test that output shape matches input shape."""
        data = torch.randn(10, 100, device=device)
        result = apply_polort_projection(data, polort=2, device=device)
        assert result.shape == data.shape

    def test_device_consistency(self, device):
        """Test that result is on correct device."""
        data = torch.randn(5, 50, device=device)
        result = apply_polort_projection(data, polort=1, device=device)
        assert result.device == device


class TestApplyHighPassFFT:
    """Test Fourier-based high-pass filtering."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_no_filter_when_cutoff_zero(self, device):
        """Test that high_pass_hz=0 returns data unchanged."""
        data = torch.randn(10, 100, device=device)
        result = apply_high_pass_fft(data, tr=1.0, high_pass_hz=0.0)
        assert torch.allclose(result, data, atol=1e-5)

    def test_removes_low_frequency(self, device):
        """Test that low-frequency signal is removed."""
        n_time = 200
        tr = 1.0

        # Create low-frequency signal (0.5 Hz, should be removed by 0.1 Hz cutoff)
        t = torch.linspace(0, n_time * tr, n_time, device=device)
        low_freq = torch.sin(2 * np.pi * 0.02 * t)  # 0.02 Hz
        data = low_freq.unsqueeze(0)

        result = apply_high_pass_fft(data, tr=tr, high_pass_hz=0.1)

        # Low frequency should be attenuated
        assert result.std() < data.std() * 0.5, "Low frequency not sufficiently attenuated"

    def test_preserves_high_frequency(self, device):
        """Test that high-frequency signal is preserved."""
        n_time = 200
        tr = 1.0

        # Create high-frequency signal (1 Hz, should pass 0.1 Hz cutoff)
        t = torch.linspace(0, n_time * tr, n_time, device=device)
        high_freq = torch.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz
        data = high_freq.unsqueeze(0)

        result = apply_high_pass_fft(data, tr=tr, high_pass_hz=0.1)

        # High frequency should be preserved
        corr = torch.corrcoef(torch.stack([data.flatten(), result.flatten()]))[0, 1]
        assert corr > 0.9, f"High frequency not preserved: corr={corr}"

    def test_preserves_data_shape(self, device):
        """Test that output shape matches input shape."""
        data = torch.randn(10, 100, device=device)
        result = apply_high_pass_fft(data, tr=1.0, high_pass_hz=0.05)
        assert result.shape == data.shape

    def test_transition_width_parameter(self, device):
        """Test that transition_width affects filter behavior."""
        n_time = 100
        data = torch.randn(2, n_time, device=device)

        # Different transition widths should give different results
        result1 = apply_high_pass_fft(data, tr=1.0, high_pass_hz=0.1, transition_width=0.1)
        result2 = apply_high_pass_fft(data, tr=1.0, high_pass_hz=0.1, transition_width=0.5)
        # Results should differ (transition width matters)
        assert not torch.allclose(result1, result2)


class TestApplyMelodicVoxelVarNorm:
    """Test MELODIC-style voxel variance normalization."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_normalizes_variance(self, device):
        """Test that output has approximately normalized variance."""
        n_vox = 10
        n_time = 100

        # Create data with varying variances
        data = torch.randn(n_vox, n_time, device=device)
        scales = torch.linspace(0.5, 5.0, n_vox, device=device)
        data = data * scales.unsqueeze(1)

        result, n_const = apply_melodic_voxel_varnorm(data)

        # Result shape should match input
        assert result.shape == data.shape

        # Variance should be roughly normalized (different voxels brought closer)
        # Compute coefficient of variation across voxels
        result_vars = result.var(dim=1)
        input_vars = data.var(dim=1)
        # Normalization should reduce variance disparity
        result_cv = result_vars.std().item() / (result_vars.mean().item() + 1e-10)
        input_cv = input_vars.std().item() / (input_vars.mean().item() + 1e-10)
        assert result_cv < input_cv, "Variance not normalized"

    def test_handles_constant_voxels(self, device):
        """Test that constant voxels are handled."""
        n_vox = 5
        n_time = 50

        data = torch.randn(n_vox, n_time, device=device)
        # Make one voxel constant
        data[2, :] = 1.0

        result, n_const = apply_melodic_voxel_varnorm(data)

        # Should report at least one constant voxel
        assert n_const >= 1
        # Constant voxel should be zeroed
        assert torch.all(result[2, :] == 0)

    def test_returns_correct_shape(self, device):
        """Test that output shape matches input."""
        n_vox = 5
        n_time = 50
        data = torch.randn(n_vox, n_time, device=device)

        result, _ = apply_melodic_voxel_varnorm(data)

        assert result.shape == data.shape

    def test_device_consistency(self, device):
        """Test that result is on correct device."""
        data = torch.randn(5, 50, device=device)
        result, _ = apply_melodic_voxel_varnorm(data)
        assert result.device == device


class TestEffectiveRankFromSpectrum:
    """Test effective rank estimation from eigenvalue spectrum."""

    def test_uniform_spectrum(self):
        """Test uniform spectrum gives full rank."""
        n = 10
        evals = np.ones(n)
        erank = effective_rank_from_spectrum(evals)
        assert erank == n

    def test_single_dominant_eigenvalue(self):
        """Test single dominant eigenvalue gives reduced but not minimal rank."""
        # Entropy-based rank is sensitive to distribution shape
        # One dominant + 99 small values still gives intermediate rank due to entropy
        evals = np.array([100] + [1] * 99)
        erank = effective_rank_from_spectrum(evals)
        # Should be less than full rank (100) due to concentration
        assert erank < 100, f"Expected reduced rank, got {erank}"
        # But entropy spreads it out, so won't be tiny
        assert erank >= 10, f"Expected at least moderate rank, got {erank}"

    def test_exponential_decay(self):
        """Test exponential decay gives reasonable rank."""
        evals = np.exp(-np.arange(100) / 20)
        erank = effective_rank_from_spectrum(evals)
        # Exponential decay gives a range of values
        assert 1 <= erank <= 100, f"Expected valid rank, got {erank}"

    def test_two_eigenvalues(self):
        """Test with just two eigenvalues."""
        evals = np.array([3, 1])
        erank = effective_rank_from_spectrum(evals)
        # Entropy-based rank should be between 1 and 2
        assert 1 <= erank <= 2

    def test_minimum_rank_is_one(self):
        """Test that minimum effective rank is 1."""
        evals = np.array([100] + [1e-10] * 99)
        erank = effective_rank_from_spectrum(evals)
        assert erank >= 1

    def test_handles_zeros(self):
        """Test handling of zero eigenvalues."""
        evals = np.array([10, 5, 2, 0, 0, 0])
        erank = effective_rank_from_spectrum(evals)
        assert 1 <= erank <= 6

    def test_extreme_skew(self):
        """Test extremely skewed spectrum."""
        # One huge, rest tiny
        evals = np.array([1000] + [0.001] * 50)
        erank = effective_rank_from_spectrum(evals)
        # Should give low rank due to extreme concentration
        assert erank < 20, f"Expected low rank for extreme skew, got {erank}"


class TestMPSpikesFromSpectrum:
    """Test Marchenko-Pastur spike detection."""

    def test_no_signal_when_beta_below_one(self):
        """Test that beta < 1 gives zero spikes."""
        # n_features < n_samples
        evals = np.array([10, 5, 3, 2, 1])
        spikes = mp_spikes_from_spectrum(evals, n_samples=100, n_features=50)
        assert spikes == 0

    def test_detects_signal_spikes(self):
        """Test detection of signal eigenvalues above MP bulk."""
        # Create spectrum with clear signal spikes
        # With beta > 1 (n_features > n_samples)
        n_features = 100
        n_samples = 50
        beta = n_features / n_samples

        # Signal eigenvalues above MP edge
        sigma2 = 1.0
        lambda_plus = sigma2 * (1 + np.sqrt(beta))**2

        evals = np.concatenate([
            np.array([lambda_plus * 2, lambda_plus * 1.5]),  # Signal spikes
            np.random.randn(98)**2  # Noise floor
        ])
        evals = np.sort(evals)[::-1]

        spikes = mp_spikes_from_spectrum(evals, n_samples=n_samples, n_features=n_features)
        assert spikes >= 2, f"Expected at least 2 spikes, got {spikes}"

    def test_counts_above_threshold(self):
        """Test that function counts eigenvalues above MP threshold."""
        # With beta > 1 (n_features > n_samples)
        n_features = 100
        n_samples = 50
        beta = n_features / n_samples

        # Calculate MP upper edge
        sigma2 = 1.0
        lambda_plus = sigma2 * (1 + np.sqrt(beta))**2

        # Create eigenvalues: some clearly above threshold
        evals = np.concatenate([
            np.array([lambda_plus * 3, lambda_plus * 2, lambda_plus * 1.5]),  # 3 signal
            np.random.uniform(0, lambda_plus * 0.9, 97)  # Noise below threshold
        ])
        evals = np.sort(evals)[::-1]

        spikes = mp_spikes_from_spectrum(evals, n_samples=n_samples, n_features=n_features)
        # Should detect at least the 3 signal eigenvalues
        assert spikes >= 3, f"Expected at least 3 signal spikes, got {spikes}"

    def test_short_spectrum(self):
        """Test with very short spectrum."""
        evals = np.array([5, 3, 1])
        spikes = mp_spikes_from_spectrum(evals, n_samples=10, n_features=20)
        assert spikes == 0  # Too short for reliable detection

    def test_empty_spectrum(self):
        """Test edge case with minimal spectrum."""
        evals = np.array([1])
        spikes = mp_spikes_from_spectrum(evals, n_samples=10, n_features=20)
        assert spikes == 0

    def test_signal_plus_noise_floor(self):
        """Test realistic case with signal above noise floor."""
        n_features = 60
        n_samples = 40

        # Signal eigenvalues (first 5)
        signal_evals = np.array([100, 80, 60, 40, 30])

        # Noise eigenvalues (smaller, from chi-squared-ish distribution)
        np.random.seed(123)
        noise_evals = np.random.chisquare(2, n_features - 5) * 2

        evals = np.concatenate([signal_evals, noise_evals])
        evals = np.sort(evals)[::-1]

        spikes = mp_spikes_from_spectrum(evals, n_samples=n_samples, n_features=n_features)
        # Should detect at least some of the signal eigenvalues
        assert spikes >= 2, f"Expected at least 2 signal spikes, got {spikes}"
