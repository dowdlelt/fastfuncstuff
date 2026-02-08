"""
Integration tests with realistic fMRI simulation.

These tests create structured fMRI data (not random noise) to verify that:
1. Analysis pipelines work end-to-end
2. Functions return expected types and values
3. Cross-validation produces sensible results
4. Bugs like missing return statements are caught

Uses:
- simulate_fmri_run() for realistic signal + noise + drift
- generate_fmri_noise() for physiological noise structure
- Actual GLM fitting with HRF convolution (not just random data)
"""

import pytest
import torch
import numpy as np
from pathlib import Path

import fastfuncsim as ffs
from fastfuncsim.simulation import simulate_fmri_run, simulate_fmri_experiment
from fastfuncsim.hrf import get_canonical_hrf
from fastfuncsim.glm_core import fit_glm, construct_polynomial_matrix
from fastfuncsim.utils import get_device, gaussian_blur_3d, to_tensor


@pytest.fixture
def device():
    return get_device()


class TestSimulationBasedGLM:
    """Test GLM with realistic simulated data."""

    def test_glm_recovers_known_betas(self, device):
        """Test that GLM can recover known beta coefficients from simulated data."""
        tr = 2.0
        n_timepoints = 200
        n_runs = 3
        matrix_size = (10, 10, 5)  # Small for speed

        # Create experimental design
        # 2 conditions, randomized inter-trial intervals
        onsets_list = []
        for run_idx in range(n_runs):
            onsets = torch.zeros(n_timepoints, 2, device=device)
            # Place 10 events per condition at random times
            for cond in [0, 1]:
                event_times = torch.randperm(n_timepoints, device=device)[:10]
                onsets[event_times, cond] = 1.0
            onsets_list.append(onsets)

        # Known betas
        true_betas = torch.tensor([3.0, 5.0], device=device)

        # Get HRF
        hrf = get_canonical_hrf(stim_duration=0.0, tr=tr, duration=30.0, device=device)

        # Simulate data with LOW noise for clear signal
        data_list = []
        for onsets in onsets_list:
            data = simulate_fmri_run(
                onsets=onsets,
                betas=true_betas.tolist(),
                hrf=hrf,
                tr=tr,
                n_timepoints=n_timepoints,
                matrix_size=matrix_size,
                noise_level=0.5,  # Low noise for recovery test
                baseline=100.0,
                add_scanner_drift=True,
                drift_amplitude=0.3,
                device=device
            )
            data_list.append(data)

        # Concatenate runs
        data_all = torch.cat([d.reshape(-1, n_timepoints) for d in data_list], dim=0)  # (n_voxels, total_tp)
        design_all = torch.cat([onsets for onsets in onsets_list], dim=0)  # (total_tp, n_cond)

        # Convolve design with HRF
        from fastfuncsim.design import build_glm_design
        design_conv = build_glm_design(design_all, hrf, n_timepoints * n_runs, mode='assumed', device=device)

        # Fit GLM
        results = fit_glm(
            data=data_all,
            design=design_conv,
            max_poly_degree=0,  # No polynomials for this simple test
        )

        # Check that we recovered the betas (mean across voxels should be close)
        recovered_betas = results.betas.mean(dim=0)  # (n_cond,)

        # Should be close to true values (within 20% due to noise and drift)
        assert torch.allclose(recovered_betas, true_betas, rtol=0.20, atol=0.5), \
            f"Failed to recover betas: true={true_betas}, recovered={recovered_betas}"

        # Check R² is reasonable (not negative, not >1)
        assert torch.all(results.r2 >= 0), "R² should be non-negative"
        assert torch.all(results.r2 <= 1), "R² should be ≤ 1"

        # Mean R² should be decent with low noise
        mean_r2 = results.r2.mean().item()
        assert mean_r2 > 0.1, f"R² too low: {mean_r2}"

    def test_glm_with_polynomial_drift(self, device):
        """Test that polynomial drift modeling works correctly."""
        tr = 2.0
        n_timepoints = 150
        matrix_size = (8, 8, 4)

        # Single event at beginning
        onsets = torch.zeros(n_timepoints, 1, device=device)
        onsets[20] = 1.0

        hrf = get_canonical_hrf(stim_duration=0.0, tr=tr, duration=30.0, device=device)

        # Simulate with STRONG drift
        data = simulate_fmri_run(
            onsets=onsets,
            betas=[5.0],
            hrf=hrf,
            tr=tr,
            n_timepoints=n_timepoints,
            matrix_size=matrix_size,
            noise_level=0.5,
            baseline=100.0,
            add_scanner_drift=True,
            drift_amplitude=2.0,  # Strong drift
            device=device
        )

        data_flat = data.reshape(-1, n_timepoints)

        # Build design
        from fastfuncsim.design import build_glm_design
        design = build_glm_design(onsets, hrf, n_timepoints, mode='assumed', device=device)

        # Fit WITH polynomials (correct)
        poly = construct_polynomial_matrix(n_timepoints, max_degree=3, device=device)
        results_with_poly = fit_glm(
            data=data_flat,
            design=design,
            extra_regressors=poly,
            max_poly_degree=-1,  # Don't add more polynomials
        )

        # Fit WITHOUT polynomials (should be worse)
        results_no_poly = fit_glm(
            data=data_flat,
            design=design,
            max_poly_degree=0,
        )

        # Version with polynomials should have higher R²
        mean_r2_with = results_with_poly.r2.mean().item()
        mean_r2_without = results_no_poly.r2.mean().item()

        assert mean_r2_with > mean_r2_without, \
            f"Polynomials should improve fit: with={mean_r2_with}, without={mean_r2_without}"

    def test_glm_returns_correct_attributes(self, device):
        """Test that fit_glm returns all expected attributes with correct shapes."""
        tr = 1.0
        n_timepoints = 100
        matrix_size = (5, 5, 2)

        onsets = torch.zeros(n_timepoints, 1, device=device)
        onsets[20] = 1.0

        hrf = get_canonical_hrf(stim_duration=0.0, tr=tr, duration=20.0, device=device)

        data = simulate_fmri_run(
            onsets=onsets,
            betas=[3.0],
            hrf=hrf,
            tr=tr,
            n_timepoints=n_timepoints,
            matrix_size=matrix_size,
            noise_level=1.0,
            baseline=100.0,
            device=device
        )

        data_flat = data.reshape(-1, n_timepoints)
        n_voxels = data_flat.shape[0]

        from fastfuncsim.design import build_glm_design
        design = build_glm_design(onsets, hrf, n_timepoints, mode='assumed', device=device)

        results = fit_glm(data=data_flat, design=design, max_poly_degree=0)

        # Check required attributes exist
        assert hasattr(results, 'betas'), "Missing betas attribute"
        assert hasattr(results, 'r2'), "Missing r2 attribute"
        assert hasattr(results, 'residuals'), "Missing residuals attribute"
        assert hasattr(results, 'predicted'), "Missing predicted attribute"

        # Check shapes
        assert results.betas.shape == (n_voxels, 1), f"Wrong betas shape: {results.betas.shape}"
        assert results.r2.shape == (n_voxels,), f"Wrong r2 shape: {results.r2.shape}"
        assert results.residuals.shape == (n_voxels, n_timepoints), f"Wrong residuals shape: {results.residuals.shape}"
        assert results.predicted.shape == (n_voxels, n_timepoints), f"Wrong predicted shape: {results.predicted.shape}"

        # Check values are finite
        assert torch.all(torch.isfinite(results.betas))
        assert torch.all(torch.isfinite(results.r2))


class TestGaussianBlur:
    """Test gaussian_blur_3d function (caught a bug here!)."""

    def test_blur_returns_correct_shape(self, device):
        """Test that gaussian_blur_3d returns data with correct shape."""
        # Create small 4D dataset
        nx, ny, nz, nt = 10, 10, 5, 20
        data = np.random.randn(nx, ny, nz, nt).astype(np.float32) * 10 + 100

        voxel_sizes = (2.0, 2.0, 2.0)
        fwhm_mm = 6.0

        # Apply blur
        data_blurred = gaussian_blur_3d(
            data=data,
            fwhm_mm=fwhm_mm,
            voxel_sizes=voxel_sizes,
            device=device,
            verbose=False
        )

        # CRITICAL: Check that function actually returns something!
        assert data_blurred is not None, "gaussian_blur_3d returned None (missing return statement!)"

        # Check shape is preserved
        assert data_blurred.shape == data.shape, \
            f"Shape changed: {data.shape} -> {data_blurred.shape}"

        # Check values are finite
        assert np.all(np.isfinite(data_blurred)), "Blurred data contains inf/nan"

        # Check that blurring actually changed the data
        assert not np.allclose(data, data_blurred), "Blurring had no effect"

    def test_blur_with_zero_fwhm(self, device):
        """Test that zero FWHM (no blur) returns near-identical data."""
        data = np.random.randn(8, 8, 4, 15).astype(np.float32) * 10 + 100
        voxel_sizes = (2.0, 2.0, 2.0)

        # Very small FWHM should have minimal effect
        data_blurred = gaussian_blur_3d(
            data=data,
            fwhm_mm=0.1,
            voxel_sizes=voxel_sizes,
            device=device,
            verbose=False
        )

        assert data_blurred is not None
        assert np.allclose(data, data_blurred, rtol=0.01), \
            "Zero/near-zero FWHM should not change data much"

    def test_blur_preserves_mean(self, device):
        """Test that blurring preserves mean signal."""
        data = np.random.randn(10, 10, 5, 20).astype(np.float32) * 10 + 100
        original_mean = data.mean()

        data_blurred = gaussian_blur_3d(
            data=data,
            fwhm_mm=6.0,
            voxel_sizes=(2.0, 2.0, 2.0),
            device=device,
            verbose=False
        )

        blurred_mean = data_blurred.mean()

        # Mean should be preserved (within numerical precision)
        assert abs(original_mean - blurred_mean) < 0.1, \
            f"Mean changed: {original_mean} -> {blurred_mean}"


class TestCrossValidation:
    """Test cross-validation with realistic data."""

    def test_loro_cv_produces_sensible_r2(self, device):
        """Test that leave-one-run-out CV produces sensible R² values."""
        tr = 2.0
        n_timepoints = 100
        n_runs = 4
        matrix_size = (8, 8, 4)

        # Create multi-run experiment
        onsets_list = []
        for run_idx in range(n_runs):
            onsets = torch.zeros(n_timepoints, 2, device=device)
            # Different event timing per run
            event_times = torch.randperm(n_timepoints, device=device)[:8]
            onsets[event_times, 0] = 1.0
            event_times = torch.randperm(n_timepoints, device=device)[:8]
            onsets[event_times, 1] = 1.0
            onsets_list.append(onsets)

        hrf = get_canonical_hrf(stim_duration=0.0, tr=tr, duration=30.0, device=device)

        # Simulate with moderate noise
        data_list = []
        for onsets in onsets_list:
            data = simulate_fmri_run(
                onsets=onsets,
                betas=[4.0, 6.0],
                hrf=hrf,
                tr=tr,
                n_timepoints=n_timepoints,
                matrix_size=matrix_size,
                noise_level=1.5,  # Moderate noise
                baseline=100.0,
                add_scanner_drift=True,
                device=device
            )
            data_list.append(data)

        # Flatten for CV
        data_all = torch.cat([d.reshape(-1, n_timepoints) for d in data_list], dim=0)

        # Build run-wise design
        from fastfuncsim.design import build_glm_design
        design_list = []
        for onsets in onsets_list:
            design_conv = build_glm_design(onsets, hrf, n_timepoints, mode='assumed', device=device)
            design_list.append(design_conv)

        # Run LORO CV
        from fastfuncsim.xval import generate_cv_splits, compute_xval_r2
        cv_splits = generate_cv_splits(n_runs, cv_type='loro', n_repeats=1)

        # Simple CV (no nuisance for now)
        r2_scores = []
        for train_idx, test_idx in cv_splits:
            # Train data
            train_data = torch.cat([data_all.reshape(-1, n_timepoints)[i] for i in train_idx], dim=1)
            train_design = torch.cat([design_list[i] for i in train_idx], dim=0)

            # Fit
            train_results = fit_glm(data=train_data, design=train_design, max_poly_degree=0)

            # Test design
            test_design = design_list[test_idx[0]]
            test_data = data_all.reshape(-1, n_timepoints)[test_idx[0]]

            # Predict
            predicted = test_design @ train_results.betas.T

            # R²
            ss_total = ((test_data - test_data.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
            ss_res = ((test_data - predicted) ** 2).sum(dim=1)
            r2 = 1 - ss_res / ss_total
            r2_scores.append(r2)

        # Check CV R² is reasonable
        all_r2 = torch.cat(r2_scores)
        assert torch.all(all_r2 >= -1), "R² should not be very negative (CV broken)"
        assert torch.all(all_r2 <= 1), "R² should not exceed 1"

        # Mean CV R² should be positive (signal is recoverable)
        mean_cv_r2 = all_r2.mean().item()
        assert mean_cv_r2 > 0.0, f"CV R² too low: {mean_cv_r2}"


class TestDenoise:
    """Test denoising with realistic data."""

    def test_noise_pool_selection(self, device):
        """Test that noise pool selection works."""
        tr = 2.0
        n_timepoints = 100
        matrix_size = (10, 10, 5)

        # Simulate data with varying signal levels
        onsets = torch.zeros(n_timepoints, 1, device=device)
        onsets[20::30] = 1.0  # Regular events

        hrf = get_canonical_hrf(stim_duration=0.0, tr=tr, duration=30.0, device=device)

        # Use heterogeneous betas (some voxels strong signal, some weak)
        n_voxels = 10 * 10 * 5
        betas = torch.zeros(n_voxels, 1, device=device)
        betas[:100] = 10.0  # Strong signal in first 100 voxels
        betas[100:] = 0.5  # Weak signal in rest

        data = simulate_fmri_run(
            onsets=onsets,
            betas=betas,
            hrf=hrf,
            tr=tr,
            n_timepoints=n_timepoints,
            matrix_size=matrix_size,
            noise_level=1.0,
            baseline=100.0,
            device=device
        )

        data_flat = data.reshape(-1, n_timepoints)

        # Select noise pool (voxels with low variance)
        from fastfuncsim.denoise import select_noise_pool_voxels
        noise_pool = select_noise_pool_voxels(
            data_flat,
            percentile=20,  # Bottom 20%
            n_voxels_max=200
        )

        # Check noise pool
        assert noise_pool.shape == (n_voxels,), f"Wrong shape: {noise_pool.shape}"
        assert noise_pool.dtype == torch.bool, "Should be boolean mask"

        # Should select some voxels
        n_selected = noise_pool.sum().item()
        assert 0 < n_selected <= n_voxels, f"Invalid selection: {n_selected}"

        # Selected voxels should have lower variance than non-selected
        var_all = data_flat.var(dim=1)
        var_noise = var_all[noise_pool]
        var_signal = var_all[~noise_pool]

        assert var_noise.mean() < var_signal.mean(), \
            "Noise pool should have lower variance than signal voxels"
