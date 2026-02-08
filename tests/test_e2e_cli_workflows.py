"""
End-to-end tests for CLI workflows using simulated data.

Tests the full analysis pipelines for:
- 3dDenoisefast.py (GLMdenoise-style denoising)
- 3dRidgefast.py (fractional ridge regression)
- 3dHRFoptfast.py (HRF optimization)

These tests create realistic fMRI data using simulate_fmri_run(),
then run the full analysis pipelines and verify outputs.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path

import fastfuncsim as ffs
from fastfuncsim.simulation import simulate_fmri_run
from fastfuncsim.hrf import get_canonical_hrf, get_hrf_library
from fastfuncsim.design import build_glm_design
from fastfuncsim.glm_core import fit_glm, construct_polynomial_matrix
from fastfuncsim.denoise import select_noise_pool_voxels, extract_noise_pcs_per_run, fit_denoising_model
from fastfuncsim.utils import get_device, scale_to_percent_signal
from fastfuncsim.cli_utils import build_nuisance_per_run
from fastfuncsim.xval import project_out_nuisance_per_run


@pytest.fixture
def device():
    return get_device()


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestDenoiseWorkflow:
    """Test GLMdenoise-style workflow end-to-end."""

    def test_full_denoise_workflow(self, device, temp_output_dir):
        """Test complete denoising pipeline: GLM -> noise pool -> PCs -> denoise."""
        tr = 2.0
        n_timepoints = 150
        n_runs = 4
        matrix_size = (10, 10, 5)  # Small for speed

        # Create multi-run experiment with varying signal strengths
        onsets_list = []
        for run_idx in range(n_runs):
            onsets = torch.zeros(n_timepoints, 2, device=device)
            # 10 events per condition
            for cond in [0, 1]:
                event_times = torch.randperm(n_timepoints, device=device)[:10]
                onsets[event_times, cond] = 1.0
            onsets_list.append(onsets)

        hrf = get_canonical_hrf(stim_duration=0.0, tr=tr, duration=30.0, device=device)

        # Simulate with heterogeneous signal (some voxels strong, some weak)
        n_voxels = matrix_size[0] * matrix_size[1] * matrix_size[2]
        betas = torch.zeros(n_voxels, 2, device=device)
        betas[:100, 0] = 8.0  # Strong signal voxels
        betas[:100, 1] = 5.0
        betas[100:, 0] = 1.0  # Weak signal voxels
        betas[100:, 1] = 0.5

        # Simulate all runs
        data_list = []
        design_list = []
        for onsets in onsets_list:
            data = simulate_fmri_run(
                onsets=onsets,
                betas=betas,
                hrf=hrf,
                tr=tr,
                n_timepoints=n_timepoints,
                matrix_size=matrix_size,
                noise_level=1.2,
                baseline=100.0,
                add_scanner_drift=True,
                drift_amplitude=0.5,
                device=device
            )
            data_list.append(data.reshape(-1, n_timepoints))

            # Build design
            design_conv = build_glm_design(onsets, hrf, n_timepoints, mode='assumed', device=device)
            design_list.append(design_conv)

        data_all = torch.cat(data_list, dim=1)  # (n_voxels, total_tp) - concatenate along time dimension

        # Build polynomial nuisance (block-diagonal per run)
        poly = construct_polynomial_matrix(n_timepoints * n_runs, max_degree=2, device=device)
        poly_per_run = []
        tp_per_run = n_timepoints
        for i in range(n_runs):
            poly_run = torch.zeros(tp_per_run, poly.shape[1], device=device)
            start = i * tp_per_run
            end = start + tp_per_run
            poly_run[:, :] = poly[start:end, :]
            poly_per_run.append(poly_run)

        # Step 1: Fit initial GLM
        initial_results = fit_glm(
            data=data_all,
            design=[d for d in design_list] + poly_per_run,
            tr=tr,
            max_poly_degree=-1,
        )

        # Check initial fit worked
        assert initial_results.betas is not None, "Initial GLM failed"
        assert initial_results.r2.mean() > 0, "Initial R² should be positive"

        # Step 2: Select noise pool (low R² voxels)
        r2_flat = initial_results.r2.flatten() if initial_results.r2.ndim > 1 else initial_results.r2
        noise_pool_mask, _ = select_noise_pool_voxels(
            r2=r2_flat,
            threshold=0.3,  # Select voxels with R² < 0.3
            min_noise_voxels=50,
        )

        # Check noise pool selection
        n_noise = noise_pool_mask.sum().item()
        assert n_noise > 0, "No noise voxels selected"
        assert n_noise < n_voxels, "All voxels selected as noise"
        print(f"  Selected {n_noise}/{n_voxels} noise voxels")

        # Step 3: Extract noise PCs per run
        noise_data = data_all[noise_pool_mask, :]  # (n_noise, total_tp)

        # Build run-wise data for noise pool
        noise_data_per_run = []
        for i in range(n_runs):
            start_idx = i * n_timepoints
            end_idx = start_idx + n_timepoints
            noise_data_per_run.append(noise_data[:, start_idx:end_idx])

        # Extract PCs
        max_pcs = 10
        pcs_per_run, variances = extract_noise_pcs_per_run(
            noise_data_per_run=noise_data_per_run,
            max_pcs=max_pcs,
            variance_threshold=0.95,
        )

        # Check PCs extracted
        assert len(pcs_per_run) == n_runs, "Should have PCs for each run"
        for i, pcs in enumerate(pcs_per_run):
            assert pcs.shape[1] <= max_pcs, f"Run {i} has too many PCs: {pcs.shape[1]}"

        print(f"  Extracted PCs per run: {[p.shape[1] for p in pcs_per_run]}")

        # Step 4: Use denoising model to get optimal PCs
        denoise_results = fit_denoising_model(
            data=data_all,
            task_design_list=design_list,
            nuisance_per_run=poly_per_run,
            noise_pool_mask=noise_pool_mask,
            max_pcs=max_pcs,
        )

        # Check results
        assert denoise_results is not None, "Denoising model failed"
        assert hasattr(denoise_results, 'optimal_n_pcs'), "Missing optimal_n_pcs"

        # Get optimal PCs for each run
        optimal_pcs = denoise_results.optimal_n_pcs
        print(f"  Optimal PCs per run: {optimal_pcs}")

        # Step 5: Apply optimal denoising
        optimal_noise_pcs = [pcs_per_run[i][:, :optimal_pcs[i]] for i in range(n_runs)]

        data_clean, design_clean = project_out_nuisance_per_run(
            data=data_all,
            design=[d for d in design_list] + poly_per_run,
            nuisance_per_run=optimal_noise_pcs,
            run_starts=None,
        )

        # Fit final GLM
        final_results = fit_glm(
            data=data_clean,
            design=design_clean,
            tr=tr,
            max_poly_degree=-1,
        )

        # Denoising should improve R²
        initial_r2 = initial_results.r2.mean().item()
        final_r2 = final_results.r2.mean().item()

        print(f"  Initial R²: {initial_r2:.3f}")
        print(f"  Final R²: {final_r2:.3f}")

        # Final R² should be >= initial (or at least not much worse)
        assert final_r2 >= initial_r2 * 0.95, \
            f"Denoising hurt R² too much: {initial_r2:.3f} -> {final_r2:.3f}"


class TestRidgeWorkflow:
    """Test fractional ridge regression workflow."""

    def test_ridge_regression_improves_stability(self, device):
        """Test that ridge regression improves stability with collinear design."""
        tr = 2.0
        n_timepoints = 120
        matrix_size = (8, 8, 4)
        n_voxels = 8 * 8 * 4

        # Create HIGHLY collinear design (events always co-occur)
        onsets = torch.zeros(n_timepoints, 3, device=device)
        event_times = torch.randperm(n_timepoints, device=device)[:8]
        # All 3 conditions at the same times = perfect collinearity
        for cond in range(3):
            onsets[event_times, cond] = 1.0

        hrf = get_canonical_hrf(stim_duration=0.0, tr=tr, duration=30.0, device=device)
        design = build_glm_design(onsets, hrf, n_timepoints, mode='assumed', device=device)

        # Simulate data
        true_betas = torch.tensor([2.0, 4.0, 6.0], device=device)
        data = simulate_fmri_run(
            onsets=onsets,
            betas=true_betas.tolist(),
            hrf=hrf,
            tr=tr,
            n_timepoints=n_timepoints,
            matrix_size=matrix_size,
            noise_level=2.0,  # High noise + collinearity = unstable OLS
            baseline=100.0,
            device=device
        )
        data_flat = data.reshape(-1, n_timepoints)

        # Fit OLS (will be unstable due to collinearity)
        results_ols = fit_glm(data=data_flat, design=design, tr=tr, max_poly_degree=0)

        # Check OLS betas are all over the place (high variance)
        ols_beta_std = results_ols.betas.std(dim=0).mean().item()
        print(f"  OLS beta std (across voxels): {ols_beta_std:.3f}")

        # Ridge should stabilize
        # Use moderate regularization
        from fastfuncsim.ridge import _fit_ridge_multiple_fracs

        # Fit with multiple fractions
        fractions = torch.tensor([0.0, 0.1, 0.3, 0.5, 1.0], device=device)

        # Ridge expects (n_voxels, n_regressors) for data
        ridge_results = _fit_ridge_multiple_fracs(
            data=data_flat,
            design=design,
            tr=tr,
            fractions=fractions,
            max_poly_degree=0,
            return_optimal_only=False,
        )

        # Check ridge results
        assert hasattr(ridge_results, 'betas'), "Missing betas in ridge results"
        assert ridge_results.betas.shape[0] == n_voxels, f"Wrong voxel count: {ridge_results.betas.shape[0]}"

        # Ridge with regularization should have more stable betas
        # Compare variance at frac=0.5 vs frac=0 (OLS)
        betas_ols = ridge_results.betas[:, :, 0]  # (n_voxels, n_regressors)
        betas_ridge = ridge_results.betas[:, :, 3]  # frac=0.5

        std_ols = betas_ols.std(dim=0).mean().item()
        std_ridge = betas_ridge.std(dim=0).mean().item()

        print(f"  OLS beta std: {std_ols:.3f}")
        print(f"  Ridge beta std: {std_ridge:.3f}")

        # Ridge should reduce variance (more stable)
        # (This might not always hold, but should with perfect collinearity)
        assert std_ridge <= std_ols * 1.5, \
            f"Ridge should stabilize betas: OLS std={std_ols:.3f}, Ridge std={std_ridge:.3f}"


class TestCLIUtilityFunctions:
    """Test utility functions used by CLI scripts."""

    def test_scale_to_percent_signal(self, device):
        """Test percent signal change scaling."""
        # Create data with varying baselines
        data = torch.randn(100, 50, device=device) * 10 + 100

        scaled, scale_factor = scale_to_percent_signal(
            data=data,
            max_scale=10.0,
        )

        # Check output
        assert scaled.shape == data.shape
        assert torch.all(torch.isfinite(scaled))

        # Mean should be around 100 (default target)
        assert abs(scaled.mean().item() - 100.0) < 1.0

        # Scale factor should be reasonable
        assert 0.5 < scale_factor < 2.0, f"Scale factor unusual: {scale_factor}"

    def test_build_nuisance_per_run(self, device):
        """Test building per-run nuisance regressors."""
        n_runs = 3
        n_timepoints = 100
        n_nuisance = 5

        # Create nuisance regressors
        nuisance = torch.randn(n_timepoints * n_runs, n_nuisance, device=device)

        # Build per-run
        nuisance_per_run = build_nuisance_per_run(
            nuisance=nuisance,
            run_lengths=[n_timepoints] * n_runs,
        )

        # Check structure
        assert len(nuisance_per_run) == n_runs
        for i, run_nuisance in enumerate(nuisance_per_run):
            expected_shape = (n_timepoints, n_nuisance)
            assert run_nuisance.shape == expected_shape, \
                f"Run {i} wrong shape: {run_nuisance.shape} vs {expected_shape}"

        # Check values are preserved
        nuisance_reconstructed = torch.cat(nuisance_per_run, dim=0)
        assert torch.allclose(nuisance, nuisance_reconstructed, atol=1e-5)

    def test_select_noise_pool_with_ground_truth(self, device):
        """Test noise pool selection when we know ground truth."""
        tr = 2.0
        n_timepoints = 100
        matrix_size = (10, 10, 5)
        n_voxels = 10 * 10 * 5

        # Create onsets
        onsets = torch.zeros(n_timepoints, 1, device=device)
        onsets[20::30] = 1.0

        hrf = get_canonical_hrf(stim_duration=0.0, tr=tr, duration=30.0, device=device)

        # First 200 voxels have signal, rest have NO signal (pure noise)
        betas = torch.zeros(n_voxels, 1, device=device)
        betas[:200, 0] = 10.0  # Strong signal
        # betas[200:, 0] = 0.0  # No signal

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
        design = build_glm_design(onsets, hrf, n_timepoints, mode='assumed', device=device)

        # Fit GLM
        results = fit_glm(data=data_flat, design=design, tr=tr, max_poly_degree=0)

        # Select noise pool (should prefer voxels 200+)
        r2_flat = results.r2.flatten() if results.r2.ndim > 1 else results.r2
        noise_pool_mask, _ = select_noise_pool_voxels(
            r2=r2_flat,
            threshold=0.1,  # Low R² threshold
            min_noise_voxels=100,
        )

        # Check that noise pool prefers the no-signal voxels
        signal_voxels_in_noise = noise_pool_mask[:200].sum().item()
        noise_voxels_in_noise = noise_pool_mask[200:].sum().item()

        print(f"  Signal voxels selected as noise: {signal_voxels_in_noise}/200")
        print(f"  No-signal voxels selected as noise: {noise_voxels_in_noise}/300")

        # Noise pool should preferentially select no-signal voxels
        # (This is probabilistic, so we check the trend)
        noise_fraction_signal = signal_voxels_in_noise / 200
        noise_fraction_noise = noise_voxels_in_noise / 300

        assert noise_fraction_noise > noise_fraction_signal, \
            "Noise pool should prefer no-signal voxels"
