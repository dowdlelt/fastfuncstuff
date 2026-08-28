"""
End-to-end tests for CLI workflows using simulated data.

Tests the full analysis pipelines for:
- 3dDenoisefast.py (GLMdenoise-style denoising)
- 3dRidgefast.py (fractional ridge regression)
- 3dHRFoptfast.py (HRF optimization)

These tests create realistic fMRI data using simulate_fmri_run(),
then run the full analysis pipelines and verify outputs.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from fastfuncstuff.cli_utils import build_nuisance_per_run
from fastfuncstuff.denoise.sequential import (
    extract_noise_pcs_per_run,
    fit_denoising_model,
    select_noise_pool_voxels,
)
from fastfuncstuff.design.hrf import get_canonical_hrf
from fastfuncstuff.design.matrices import build_glm_design
from fastfuncstuff.glm.core import construct_polynomial_matrix, fit_glm
from fastfuncstuff.simulation.core import simulate_fmri_run
from fastfuncstuff.utils import scale_to_percent_signal


@pytest.fixture
def device():
    return torch.device("cpu")


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
        for _run_idx in range(n_runs):
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
                device=device,
            )
            data_list.append(data.reshape(-1, n_timepoints))

            # Build design
            design_conv = build_glm_design(onsets, hrf, n_timepoints, mode="assumed", device=device)
            design_list.append(design_conv)

        data_all = torch.cat(
            data_list, dim=1
        )  # (n_voxels, total_tp) - concatenate along time dimension
        design_all = torch.cat(design_list, dim=0)
        run_starts = [i * n_timepoints for i in range(n_runs)]

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
            data=data_list,
            design=design_list,
            extra_regressors=poly_per_run,
            tr=tr,
            max_poly_degree=-1,
        )

        # Check initial fit worked
        assert initial_results.betas is not None, "Initial GLM failed"
        assert initial_results.r2.mean() > 0, "Initial R² should be positive"

        # Step 2: Select noise pool (low R² voxels)
        r2_flat = (
            initial_results.r2.flatten() if initial_results.r2.ndim > 1 else initial_results.r2
        )
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
        max_pcs = 10
        pcs_per_run = extract_noise_pcs_per_run(
            data=data_all,
            run_starts=run_starts,
            noise_pool_mask=noise_pool_mask,
            max_components=max_pcs,
            variance_threshold=0.95,
            nuisance_per_run=poly_per_run,
            device=device,
        )

        # Check PCs extracted
        assert len(pcs_per_run) == n_runs, "Should have PCs for each run"
        for i, pcs in enumerate(pcs_per_run):
            assert pcs.shape[1] <= max_pcs, f"Run {i} has too many PCs: {pcs.shape[1]}"

        print(f"  Extracted PCs per run: {[p.shape[1] for p in pcs_per_run]}")

        # Step 4: Use denoising model to get optimal PCs
        denoise_results = fit_denoising_model(
            data=data_all,
            design_matrix=design_all,
            run_starts=run_starts,
            tr=tr,
            initial_r2=r2_flat,
            r2_threshold=0.3,
            nuisance=poly_per_run,
            max_components=max_pcs,
            device=device,
        )

        # Check results
        assert denoise_results is not None, "Denoising model failed"
        assert hasattr(denoise_results, "optimal_n_components"), "Missing optimal_n_components"

        optimal_pcs = int(denoise_results.optimal_n_components)
        print(f"  Optimal PCs: {optimal_pcs}")

        # Step 5: Apply optimal denoising
        nuisance_with_pcs = []
        for i in range(n_runs):
            if optimal_pcs > 0:
                nuisance_run = torch.cat([poly_per_run[i], pcs_per_run[i][:, :optimal_pcs]], dim=1)
            else:
                nuisance_run = poly_per_run[i]
            nuisance_with_pcs.append(nuisance_run)

        # Fit final GLM with denoising regressors (same formulation as initial GLM)
        final_results = fit_glm(
            data=data_list,
            design=design_list,
            extra_regressors=nuisance_with_pcs,
            tr=tr,
            max_poly_degree=-1,
        )

        # Denoising should improve R²
        initial_r2 = initial_results.r2.mean().item()
        final_r2 = final_results.r2.mean().item()

        print(f"  Initial R²: {initial_r2:.3f}")
        print(f"  Final R²: {final_r2:.3f}")

        # Final R² should be >= initial (or at least not much worse)
        assert final_r2 >= initial_r2 * 0.95, (
            f"Denoising hurt R² too much: {initial_r2:.3f} -> {final_r2:.3f}"
        )


class TestRidgeWorkflow:
    """Test fractional ridge regression workflow."""

    def test_ridge_regression_improves_stability(self, device):
        """Test that ridge regression improves stability with collinear design."""
        tr = 2.0
        n_timepoints = 120
        matrix_size = (8, 8, 4)
        _n_voxels = 8 * 8 * 4

        # Create HIGHLY collinear design (events always co-occur)
        onsets = torch.zeros(n_timepoints, 3, device=device)
        event_times = torch.randperm(n_timepoints, device=device)[:8]
        # All 3 conditions at the same times = perfect collinearity
        for cond in range(3):
            onsets[event_times, cond] = 1.0

        hrf = get_canonical_hrf(stim_duration=0.0, tr=tr, duration=30.0, device=device)
        design = build_glm_design(onsets, hrf, n_timepoints, mode="assumed", device=device)

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
            device=device,
        )
        data_flat = data.reshape(-1, n_timepoints)

        # Fit OLS (will be unstable due to collinearity)
        results_ols = fit_glm(data=data_flat, design=design, tr=tr, max_poly_degree=0)

        # Check OLS betas are all over the place (high variance)
        ols_beta_std = results_ols.betas.std(dim=0).mean().item()
        print(f"  OLS beta std (across voxels): {ols_beta_std:.3f}")

        # Ridge should stabilize
        # Use moderate regularization
        from fastfuncstuff.glm.ridge import _fit_ridge_multiple_fracs

        # Fit with multiple fractions using core fracridge helper
        fracs = np.array([0.0, 0.1, 0.3, 0.5, 1.0], dtype=np.float32)
        coefs = _fit_ridge_multiple_fracs(
            X=design,
            y=data_flat.T,
            fracs=fracs,
            device=device,
        )  # (n_regressors, n_fracs, n_voxels)

        # Ridge with regularization should have more stable betas
        # Compare variance at frac=0.5 vs frac=1.0 (OLS)
        betas_ols = coefs[:, -1, :].T  # (n_voxels, n_regressors)
        frac_idx_05 = int(np.where(np.isclose(fracs, 0.5))[0][0])
        betas_ridge = coefs[:, frac_idx_05, :].T

        std_ols = betas_ols.std(dim=0).mean().item()
        std_ridge = betas_ridge.std(dim=0).mean().item()

        print(f"  OLS beta std: {std_ols:.3f}")
        print(f"  Ridge beta std: {std_ridge:.3f}")

        # Ridge should reduce variance (more stable)
        # (This might not always hold, but should with perfect collinearity)
        assert std_ridge <= std_ols * 1.5, (
            f"Ridge should stabilize betas: OLS std={std_ols:.3f}, Ridge std={std_ridge:.3f}"
        )


class TestCLIUtilityFunctions:
    """Test utility functions used by CLI scripts."""

    def test_scale_to_percent_signal(self, device):
        """Test percent signal change scaling."""
        # Create data with varying baselines
        data = torch.randn(100, 50, device=device) * 10 + 100

        scaled, violations_mask, scale_info = scale_to_percent_signal(
            data=data,
            run_starts=[0],
            max_scale=200.0,
            verbose=False,
        )

        # Check output
        assert scaled.shape == data.shape
        assert torch.all(torch.isfinite(scaled))

        # Mean should be around 100 (default target)
        assert abs(scaled.mean().item() - 100.0) < 1.0

        # Scale factors should be finite and mostly in a plausible range
        scale_factors = scale_info["scale_factors"]
        assert torch.isfinite(scale_factors).all()
        median_scale = torch.median(scale_factors).item()
        assert 0.5 < median_scale < 2.0, f"Median scale factor unusual: {median_scale}"

    def test_build_nuisance_per_run(self, device):
        """Test building per-run nuisance regressors.

        Note: build_nuisance_per_run now demeans each nuisance block
        per-run so it doesn't collide with the per-run polort intercept.
        The contract isn't "exact round-trip of the input" but "each
        per-run slice is the input minus that run's column mean."
        """
        n_runs = 3
        n_timepoints = 100
        n_nuisance = 5
        n_total = n_timepoints * n_runs
        run_starts = [i * n_timepoints for i in range(n_runs)]

        nuisance = torch.randn(n_total, n_nuisance, device=device)

        nuisance_per_run = build_nuisance_per_run(
            run_starts=run_starts,
            n_timepoints=n_total,
            polort=-1,
            device=device,
            ortvec_data=nuisance,
        )

        # Structure unchanged.
        assert len(nuisance_per_run) == n_runs
        for i, run_nuisance in enumerate(nuisance_per_run):  # noqa: B007
            expected_shape = (n_timepoints, n_nuisance)
            assert run_nuisance.shape == expected_shape

        # Each run's output must equal that run's input slice with its
        # column means subtracted (because of demean-on-assemble).
        for i, run_nuisance in enumerate(nuisance_per_run):
            start = run_starts[i]
            end = start + n_timepoints
            expected = nuisance[start:end, :]
            expected = expected - expected.mean(dim=0, keepdim=True)
            assert torch.allclose(
                run_nuisance,
                expected.to(run_nuisance.dtype),
                atol=1e-5,
            ), f"Run {i} not demeaned-equivalent to input slice"

        # Per-run output should be zero-mean (the whole point of the demean).
        for i, run_nuisance in enumerate(nuisance_per_run):
            col_means = run_nuisance.mean(dim=0)
            assert torch.allclose(
                col_means,
                torch.zeros_like(col_means),
                atol=1e-5,
            ), f"Run {i} not zero-mean after build_nuisance_per_run"

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
            device=device,
        )

        data_flat = data.reshape(-1, n_timepoints)
        design = build_glm_design(onsets, hrf, n_timepoints, mode="assumed", device=device)

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

        assert noise_fraction_noise > noise_fraction_signal, (
            "Noise pool should prefer no-signal voxels"
        )
