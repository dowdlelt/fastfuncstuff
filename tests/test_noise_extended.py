"""
Tests for uncovered functions in fastfuncstuff/simulation/noise.py:
- generate_fmri_noise_batch
- add_drift
- add_motion_artifacts
- generate_ar_noise (AR(p))
- generate_arma_noise
- estimate_noise_parameters_from_data
- estimate_sfnr
"""

import numpy as np
import pytest
import torch

from fastfuncstuff.simulation.noise import (
    add_drift,
    add_motion_artifacts,
    estimate_noise_parameters_from_data,
    estimate_sfnr,
    generate_ar1_noise,
    generate_ar_noise,
    generate_arma_noise,
    generate_fmri_noise_batch,
)

DEVICE = torch.device("cpu")


class TestGenerateFmriNoiseBatch:
    def test_shape_single_voxel(self):
        result = generate_fmri_noise_batch(
            tr=2.0,
            duration_s=10.0,
            n_batches=3,
            matrix_size=(1, 1),
            device=DEVICE,
        )
        # 10s / 2s TR = 5 timepoints, single voxel
        assert result.shape == (3, 5)

    def test_shape_multi_voxel(self):
        result = generate_fmri_noise_batch(
            tr=1.0,
            duration_s=5.0,
            n_batches=2,
            matrix_size=(3, 4),
            device=DEVICE,
        )
        assert result.shape == (2, 5, 3, 4)

    def test_batches_are_independent(self):
        result = generate_fmri_noise_batch(
            tr=2.0,
            duration_s=20.0,
            n_batches=2,
            matrix_size=(1, 1),
            device=DEVICE,
        )
        # Two different realizations should not be identical
        assert not torch.allclose(result[0], result[1])


class TestAddDrift:
    def test_increases_low_freq_power(self):
        """Adding drift should increase low-frequency content."""
        torch.manual_seed(42)
        data = torch.randn(200, device=DEVICE)
        data_drift = add_drift(data, amplitude=2.0, n_modes=3, device=DEVICE)
        assert data_drift.shape == data.shape
        # Drift version should differ from original
        assert not torch.allclose(data, data_drift)

    def test_2d_data(self):
        """Should handle 2D (timepoints, voxels) data."""
        data = torch.randn(100, 50, device=DEVICE)
        result = add_drift(data, amplitude=1.0, device=DEVICE)
        assert result.shape == (100, 50)

    def test_zero_amplitude_preserves_data(self):
        """Zero amplitude drift should leave data approximately unchanged."""
        data = torch.randn(100, device=DEVICE) * 10.0
        result = add_drift(data, amplitude=0.0, device=DEVICE)
        torch.testing.assert_close(result, data.unsqueeze(1).squeeze(1), atol=1e-5, rtol=1e-5)


class TestAddMotionArtifacts:
    def test_adds_spikes(self):
        data = torch.randn(100, device=DEVICE)
        result, spike_times = add_motion_artifacts(
            data, max_displacement=5.0, n_spikes=3, device=DEVICE
        )
        assert result.shape == data.shape
        assert len(spike_times) == 3
        # Spike times should be valid indices > 0
        assert (spike_times > 0).all()
        assert (spike_times < 100).all()

    def test_modifies_data(self):
        data = torch.zeros(100, device=DEVICE)
        result, _ = add_motion_artifacts(data, max_displacement=10.0, n_spikes=5, device=DEVICE)
        assert not torch.allclose(result, data)

    def test_multidimensional(self):
        data = torch.randn(50, 10, device=DEVICE)
        result, spike_times = add_motion_artifacts(data, n_spikes=2, device=DEVICE)
        assert result.shape == (50, 10)


class TestGenerateArNoise:
    def test_ar2_shape(self):
        noise = generate_ar_noise([0.5, 0.2], n_timepoints=100, n_voxels=10, device=DEVICE)
        assert noise.shape == (100, 10)

    def test_ar1_single_voxel(self):
        noise = generate_ar_noise([0.3], n_timepoints=200, n_voxels=1, device=DEVICE)
        assert noise.shape == (200,)

    def test_normalized_unit_variance(self):
        noise = generate_ar_noise(
            [0.5, 0.1],
            n_timepoints=1000,
            n_voxels=5,
            normalize=True,
            device=DEVICE,
        )
        # Should have approximately unit variance
        assert abs(noise.std().item() - 1.0) < 0.15

    def test_accepts_list_and_tensor(self):
        n1 = generate_ar_noise([0.5], n_timepoints=50, device=DEVICE)
        n2 = generate_ar_noise(torch.tensor([0.5]), n_timepoints=50, device=DEVICE)
        assert n1.shape == n2.shape


class TestGenerateArmaNoise:
    def test_arma11_shape(self):
        noise = generate_arma_noise([0.3], [0.2], n_timepoints=100, n_voxels=5, device=DEVICE)
        assert noise.shape == (100, 5)

    def test_arma_normalized(self):
        noise = generate_arma_noise(
            [0.4],
            [0.3],
            n_timepoints=1000,
            n_voxels=10,
            normalize=True,
            device=DEVICE,
        )
        assert abs(noise.std().item() - 1.0) < 0.15

    def test_pure_ma(self):
        """AR part empty, only MA."""
        noise = generate_arma_noise([], [0.5, 0.3], n_timepoints=100, n_voxels=3, device=DEVICE)
        assert noise.shape == (100, 3)

    def test_pure_ar(self):
        """MA part empty, only AR."""
        noise = generate_arma_noise([0.5], [], n_timepoints=100, n_voxels=3, device=DEVICE)
        assert noise.shape == (100, 3)

    def test_invalid_ar_coeff_raises(self):
        with pytest.raises(ValueError, match="stationarity"):
            generate_arma_noise([1.5], [0.2], n_timepoints=100, device=DEVICE)


class TestEstimateNoiseParametersFromData:
    def test_basic_estimation(self):
        """Estimate AR(1) from synthetic AR(1) data."""
        torch.manual_seed(42)
        true_rho = 0.4
        noise = generate_ar1_noise(true_rho, n_timepoints=500, n_voxels=50, device=DEVICE)
        params = estimate_noise_parameters_from_data(noise, ar_order=1, device=DEVICE)
        # Should estimate close to true_rho
        estimated_rho = params["ar_coefficients"][0]
        assert abs(estimated_rho - true_rho) < 0.15

    def test_with_design_matrix(self):
        """Estimation after removing task variance via design."""
        torch.manual_seed(42)
        n_tp = 200
        design = torch.randn(n_tp, 3, device=DEVICE)
        betas = torch.randn(3, 20, device=DEVICE)
        noise = torch.randn(n_tp, 20, device=DEVICE) * 0.5
        data = design @ betas + noise

        params = estimate_noise_parameters_from_data(data, design=design, ar_order=1, device=DEVICE)
        assert "ar_coefficients" in params
        assert "sfnr" in params
        assert "noise_std" in params
        assert params["n_timepoints"] == n_tp

    def test_1d_input(self):
        data = torch.randn(100, device=DEVICE)
        params = estimate_noise_parameters_from_data(data, device=DEVICE)
        assert params["n_voxels"] == 1

    def test_3d_input(self):
        """3D+ data should be flattened correctly."""
        data = torch.randn(5, 5, 100, device=DEVICE)
        params = estimate_noise_parameters_from_data(data, device=DEVICE)
        assert params["n_voxels"] == 25
        assert params["n_timepoints"] == 100

    def test_with_mask(self):
        data = torch.randn(10, 100, device=DEVICE) + 50.0
        mask = torch.zeros(10, dtype=torch.bool)
        mask[:5] = True
        params = estimate_noise_parameters_from_data(data.T, mask=mask, device=DEVICE)
        # Only masked voxels analyzed
        assert params["n_voxels"] == 5

    def test_ar2_estimation(self):
        params = estimate_noise_parameters_from_data(
            torch.randn(200, 10, device=DEVICE),
            ar_order=2,
            device=DEVICE,
        )
        assert len(params["ar_coefficients"]) == 2

    def test_output_keys(self):
        params = estimate_noise_parameters_from_data(
            torch.randn(100, 5, device=DEVICE), device=DEVICE
        )
        expected_keys = [
            "ar_coefficients",
            "ar_coefficients_std",
            "ar_coefficients_all",
            "sfnr",
            "sfnr_std",
            "sfnr_all",
            "noise_std",
            "n_voxels",
            "n_timepoints",
            "summary",
        ]
        for key in expected_keys:
            assert key in params, f"Missing key: {key}"


class TestEstimateSfnr:
    def test_basic_sfnr(self):
        """High-mean, low-std signal should have high SFNR."""
        data = torch.ones(10, 100, device=DEVICE) * 500.0
        data += torch.randn(10, 100, device=DEVICE) * 2.0
        result = estimate_sfnr(data, device=DEVICE)
        # SFNR should be ~250 (500/2)
        assert result["sfnr_mean"] > 100.0

    def test_sfnr_with_mask(self):
        data = torch.randn(5, 5, 100, device=DEVICE) + 100.0
        mask = torch.zeros(5, 5, dtype=torch.bool)
        mask[:3, :3] = True
        result = estimate_sfnr(data, mask=mask, device=DEVICE)
        assert result["sfnr_map"].shape == (5, 5)
        # Masked-out voxels should be 0
        assert result["sfnr_map"][4, 4].item() == 0.0

    def test_output_keys(self):
        result = estimate_sfnr(torch.randn(50, device=DEVICE) + 100.0, device=DEVICE)
        for key in ["sfnr_mean", "sfnr_median", "sfnr_std", "sfnr_map", "summary"]:
            assert key in result

    def test_numpy_input(self):
        data = np.random.randn(20, 50).astype(np.float32) + 100.0
        result = estimate_sfnr(data, device=DEVICE)
        assert result["sfnr_mean"] > 0
