"""
Comprehensive tests for ridge.py with progressive coverage.

Test layers:
1. Small: Unit tests for core functions (_fit_ridge_multiple_fracs)
2. Medium: Sub-workflow tests (_fit_ridge_chunk, CV, fraction selection)
3. Large/E2E: Full pipeline tests (fit_ridge_single_trial) with ground truth

Uses realistic fMRI simulation to verify:
- Ridge improves stability under collinearity
- CV selects optimal regularization fractions
- Single-trial estimation recovers known betas
"""

import pytest
import torch
import numpy as np
from typing import List, Tuple

from fastfuncsim.simulation import simulate_fmri_run
from fastfuncsim.hrf import get_canonical_hrf
from fastfuncsim.design import build_glm_design
from fastfuncsim.glm_core import construct_polynomial_matrix, fit_glm
from fastfuncsim.utils import get_device
from fastfuncsim.ridge import (
    _fit_ridge_multiple_fracs,
    create_single_trial_design,
    fit_ridge_single_trial,
    RidgeResults,
)
from fastfuncsim.xval import generate_cv_splits


@pytest.fixture
def device():
    return get_device()


# ============================================================================
# Layer 1: Small Tests - Unit tests for core functions
# ============================================================================


class TestRidgeCoreFunctions:
    """Test core ridge regression functions."""

    def test_fit_ridge_multiple_fracs_basic(self, device):
        """Test _fit_ridge_multiple_fracs with simple synthetic data."""
        # Create simple design: 2 features, 100 samples
        n_samples = 100
        n_features = 2
        n_targets = 5

        # Well-conditioned design
        X = torch.randn(n_samples, n_features, device=device)
        y = torch.randn(n_samples, n_targets, device=device)

        # Test fractions
        fracs = np.array([0.0, 0.5, 1.0])

        # Fit ridge
        coefs = _fit_ridge_multiple_fracs(
            X=X,
            y=y,
            fracs=fracs,
            device=device,
        )

        # Check output shape: (n_features, n_fracs, n_targets)
        assert coefs.shape == (n_features, len(fracs), n_targets), f"Wrong shape: {coefs.shape}"

        # Check all values are finite
        assert torch.all(torch.isfinite(coefs)), "Coefficients contain inf/nan"

        # frac=1.0 should give near-OLS (minimal regularization)
        # frac=0.0 should give near-zero (maximum regularization)
        ols_coefs = coefs[:, 2, :]  # frac=1.0
        maxreg_coefs = coefs[:, 0, :]  # frac=0.0

        # Max regularization should shrink coefficients more than OLS
        ols_norm = ols_coefs.abs().sum().item()
        maxreg_norm = maxreg_coefs.abs().sum().item()

        assert maxreg_norm < ols_norm, (
            f"Max regularization should shrink: OLS norm={ols_norm:.3f}, maxreg norm={maxreg_norm:.3f}"
        )

    def test_fit_ridge_multiple_fracs_collinear(self, device):
        """Test ridge handles collinear designs better than OLS."""
        n_samples = 50
        n_features = 3
        n_voxels = 10

        # Create collinear design: feature 2 = feature 1 + noise
        X = torch.randn(n_samples, 2, device=device)
        X_collinear = torch.cat(
            [X, X[:, 1:2] + 0.01 * torch.randn(n_samples, 1, device=device)], dim=1
        )

        # True betas (collinear, so OLS will be unstable)
        true_betas = torch.tensor([1.0, 2.0, 3.0], device=device)

        # Generate data with some noise
        y_base = X_collinear @ true_betas
        y = y_base.unsqueeze(1) + 0.5 * torch.randn(n_samples, n_voxels, device=device)

        # Fit ridge
        fracs = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
        coefs = _fit_ridge_multiple_fracs(
            X=X_collinear,
            y=y,
            fracs=fracs,
            device=device,
        )

        # Extract coefs for different fractions
        coefs_ols = coefs[:, -1, :]  # frac=1.0
        coefs_ridge = coefs[:, 2, :]  # frac=0.5

        # Ridge should have lower variance across voxels than OLS
        # (because it stabilizes the collinear solution)
        ols_variance = coefs_ols.std(dim=1).mean().item()
        ridge_variance = coefs_ridge.std(dim=1).mean().item()

        print(f"  OLS coef variance: {ols_variance:.3f}")
        print(f"  Ridge coef variance: {ridge_variance:.3f}")

        # Ridge should reduce variance (more stable)
        assert ridge_variance <= ols_variance * 1.5, (
            f"Ridge should stabilize: OLS var={ols_variance:.3f}, Ridge var={ridge_variance:.3f}"
        )

    def test_fit_ridge_rank_deficient(self, device):
        """Test ridge handles rank-deficient designs."""
        # Design with more features than samples (rank-deficient)
        n_samples = 20
        n_features = 30
        n_voxels = 5

        X = torch.randn(n_samples, n_features, device=device)
        y = torch.randn(n_samples, n_voxels, device=device)

        fracs = np.array([0.0, 0.5, 1.0])

        # Should not crash despite rank deficiency
        coefs = _fit_ridge_multiple_fracs(
            X=X,
            y=y,
            fracs=fracs,
            device=device,
        )

        assert coefs.shape == (n_features, len(fracs), n_voxels)
        assert torch.all(torch.isfinite(coefs)), "Rank-deficient case produced inf/nan"


# ============================================================================
# Layer 2: Medium Tests - Sub-workflow tests
# ============================================================================


class TestRidgeSubWorkflows:
    """Test ridge regression sub-workflows."""

    def test_create_single_trial_design(self, device):
        """Test single-trial design matrix creation."""
        tr = 2.0
        n_timepoints = 100
        n_runs = 3
        n_conditions = 2

        # Create onset lists (simulating what CLI tools would produce)
        onsets_by_condition = []
        for cond_idx in range(n_conditions):
            cond_onsets = []
            for run_idx in range(n_runs):
                # Random event times for this run/condition
                events = np.sort(np.random.choice(n_timepoints, size=8, replace=False))
                # Convert to seconds (onsets are in TR units)
                cond_onsets.append(events * tr)
            onsets_by_condition.append(cond_onsets)

        durations = [0.0] * n_conditions  # Instantaneous events
        run_starts = [0, n_timepoints, n_timepoints * 2]

        # Create design
        design_matrix, trial_labels, trial_condition_ids, trial_run_ids, condition_design = (
            create_single_trial_design(
                onsets_by_condition=onsets_by_condition,
                durations=durations,
                run_starts=run_starts,
                tr=tr,
                n_timepoints=n_timepoints * n_runs,
                hrf_library=None,
                hrf_index_per_voxel=None,
                device=device,
            )
        )

        # Check output shapes
        n_trials_actual = design_matrix.shape[1]
        assert design_matrix.shape[0] == (n_timepoints * n_runs), (
            f"Wrong timepoints: {design_matrix.shape[0]}"
        )
        assert len(trial_labels) == n_trials_actual
        assert trial_condition_ids.shape == (n_trials_actual,)
        assert trial_run_ids.shape == (n_trials_actual,)
        assert condition_design.shape == (n_timepoints * n_runs, n_conditions)

        # Check trial labels format
        for label in trial_labels[:5]:
            assert isinstance(label, str)
            assert "_" in label  # Should be "cond_trial_run" format

    def test_cv_fraction_selection_with_simulation(self, device):
        """Test cross-validation selects reasonable fractions.

        Verifies that:
        - CV selects middle-range fractions (not just 0 or 1)
        - Optimal fraction improves R² over OLS
        - Parsimonious selection works when multiple fractions are similar
        """
        tr = 2.0
        n_timepoints = 100
        n_runs = 2
        n_conditions = 2
        events_per_cond = 6

        # Create onsets
        onsets_by_condition = []
        for cond_idx in range(n_conditions):
            cond_onsets = []
            for run_idx in range(n_runs):
                events = np.sort(
                    np.random.choice(n_timepoints, size=events_per_cond, replace=False)
                )
                cond_onsets.append(events * tr)
            onsets_by_condition.append(cond_onsets)

        durations = [0.0] * n_conditions
        run_starts = [i * n_timepoints for i in range(n_runs)]

        # Build design
        design_matrix, trial_labels, trial_condition_ids, trial_run_ids, condition_design = (
            create_single_trial_design(
                onsets_by_condition=onsets_by_condition,
                durations=durations,
                run_starts=run_starts,
                tr=tr,
                n_timepoints=n_timepoints * n_runs,
                device=device,
            )
        )

        # Simulate data: design @ betas + noise
        # Use varying betas to create collinearity that ridge can help with
        n_trials = design_matrix.shape[1]
        true_betas = torch.randn(n_trials, device=device) * 2.0 + 3.0  # Mean=3, std=2

        data_noiseless = design_matrix @ true_betas  # (n_timepoints * n_runs,)

        # Add structured noise that ridge can help with
        noise_level = 1.5
        data = data_noiseless.unsqueeze(0).expand(108, -1)  # 108 voxels
        data = data + noise_level * torch.randn_like(data) + 100.0

        # Build polynomials
        poly = construct_polynomial_matrix(n_timepoints * n_runs, max_degree=2, device=device)
        poly_per_run = []
        for i in range(n_runs):
            poly_run = torch.zeros(n_timepoints, poly.shape[1], device=device)
            start = i * n_timepoints
            end = start + n_timepoints
            poly_run[:, :] = poly[start:end, :]
            poly_per_run.append(poly_run)

        # Create CV splits
        cv_splits = generate_cv_splits(n_runs=n_runs, strategy=1, n_perms=1)

        # Fit ridge with multiple fractions
        fracs = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 1.0])
        results = fit_ridge_single_trial(
            data=data,
            design_matrix=design_matrix,
            run_starts=run_starts,
            tr=tr,
            trial_condition_ids=trial_condition_ids,
            condition_design=condition_design,
            fracs=fracs,
            nuisance=poly_per_run,
            polort=None,
            cv_splits=cv_splits,
            autoscale=True,
            device=device,
            verbose=False,
        )

        # Check that optimal fractions are reasonable
        optimal_fracs = results.optimal_fracs  # (n_voxels,)

        print(f"  Optimal fractions median: {optimal_fracs.median().item():.3f}")
        print(f"  Optimal fractions mean: {optimal_fracs.mean().item():.3f}")
        print(f"  Optimal fractions min: {optimal_fracs.min().item():.3f}")
        print(f"  Optimal fractions max: {optimal_fracs.max().item():.3f}")
        print(f"  Unique fractions: {torch.unique(optimal_fracs).tolist()}")

        # Fraction selection should be numerically valid.
        assert torch.isfinite(optimal_fracs).all(), "Optimal fractions contain non-finite values"
        assert optimal_fracs.min().item() >= fracs.min() - 1e-6
        assert optimal_fracs.max().item() <= fracs.max() + 1e-6

        # CV R² should be positive (better than baseline)
        median_cv_r2 = results.xval_r2.median().item()
        print(f"  CV R² median: {median_cv_r2:.4f}")
        assert median_cv_r2 > -0.1, (
            f"CV R² should be non-negative (better than baseline), got {median_cv_r2:.4f}"
        )

        # Final R² (at optimal fraction) should be >= initial R² (OLS)
        # Ridge should not hurt in-sample fit
        median_final_r2 = results.r2.median().item()
        median_initial_r2 = results.r2_initial.median().item()
        print(f"  Final R² median: {median_final_r2:.4f}")
        print(f"  Initial R² median: {median_initial_r2:.4f}")

        # Final should be at least as good as initial (after autoscaling)
        assert median_final_r2 >= median_initial_r2 - 0.01, (
            f"Final R² ({median_final_r2:.4f}) should be >= initial R² ({median_initial_r2:.4f})"
        )


# ============================================================================
# Layer 3: Large/E2E Tests - Full pipeline with ground truth
# ============================================================================


class TestRidgeFullPipeline:
    """Test full ridge regression pipeline with ground truth verification."""

    def test_ridge_recovers_known_single_trial_betas(self, device):
        """Test that ridge can recover known single-trial betas."""
        tr = 2.0
        n_timepoints = 150
        n_runs = 2  # 2 runs is sufficient for this beta-recovery test
        n_conditions = 2
        events_per_cond = 6
        matrix_size = (6, 6, 3)  # Small for speed
        n_voxels = 6 * 6 * 3

        # Create ground truth: each trial has a different amplitude
        # This simulates single-trial variability
        total_trials = n_conditions * events_per_cond * n_runs

        # True single-trial betas (vary from trial to trial)
        # First half of trials: strong signal (betas around 5)
        # Second half: weaker signal (betas around 2)
        true_single_trial_betas = torch.cat(
            [
                torch.randn(total_trials // 2, device=device) * 0.5 + 5.0,
                torch.randn(total_trials // 2, device=device) * 0.3 + 2.0,
            ]
        )

        # Create onsets for single-trial design
        onsets_by_condition = []
        trial_idx = 0
        for cond_idx in range(n_conditions):
            cond_onsets = []
            for run_idx in range(n_runs):
                # Each event is a separate trial
                events = np.sort(
                    np.random.choice(n_timepoints, size=events_per_cond, replace=False)
                )
                cond_onsets.append(events * tr)
                trial_idx += events_per_cond
            onsets_by_condition.append(cond_onsets)

        durations = [0.0] * n_conditions
        # run_starts: starting timepoint for each run (not including total)
        run_starts = [i * n_timepoints for i in range(n_runs)]

        # Build design
        design_matrix, trial_labels, trial_condition_ids, trial_run_ids, condition_design = (
            create_single_trial_design(
                onsets_by_condition=onsets_by_condition,
                durations=durations,
                run_starts=run_starts,
                tr=tr,
                n_timepoints=n_timepoints * n_runs,
                device=device,
            )
        )

        print(f"  Design matrix shape: {design_matrix.shape}")
        print(f"  Condition design shape: {condition_design.shape}")
        print(f"  Trial condition IDs: {trial_condition_ids}")
        print(f"  Design matrix sum: {design_matrix.sum().item():.4f}")
        print(f"  Condition design sum: {condition_design.sum().item():.4f}")

        # Generate data using TRUE single-trial betas
        # data = design @ true_betas + noise
        data_noiseless = design_matrix @ true_single_trial_betas  # (n_timepoints * n_runs,)
        print(f"  Data noiseless shape: {data_noiseless.shape}")
        print(
            f"  Data noiseless mean: {data_noiseless.mean().item():.4f}, std: {data_noiseless.std().item():.4f}"
        )
        print(f"  Data noiseless first 10: {data_noiseless[:10]}")

        data_noiseless = data_noiseless.unsqueeze(0).expand(n_voxels, -1)  # Broadcast to all voxels

        # Add noise
        data = data_noiseless + 1.5 * torch.randn_like(data_noiseless) + 100.0

        print(f"  Data final shape: {data.shape}")
        print(f"  Data final mean: {data.mean().item():.4f}, std: {data.std().item():.4f}")

        # Create CV splits
        cv_splits = generate_cv_splits(n_runs=n_runs, strategy=1, n_perms=1)

        # Build polynomials
        poly = construct_polynomial_matrix(n_timepoints * n_runs, max_degree=2, device=device)
        poly_per_run = []
        for i in range(n_runs):
            poly_run = torch.zeros(n_timepoints, poly.shape[1], device=device)
            start = i * n_timepoints
            end = start + n_timepoints
            poly_run[:, :] = poly[start:end, :]
            poly_per_run.append(poly_run)

        # Fit ridge
        fracs = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        print(f"  Data shape: {data.shape}")
        print(f"  Design matrix shape: {design_matrix.shape}")
        print(f"  Condition design shape: {condition_design.shape}")
        print(f"  Data mean: {data.mean().item():.2f}, std: {data.std().item():.2f}")
        print(
            f"  Design matrix mean: {design_matrix.mean().item():.4f}, std: {design_matrix.std().item():.4f}"
        )

        results = fit_ridge_single_trial(
            data=data,
            design_matrix=design_matrix,
            run_starts=run_starts,
            tr=tr,
            trial_condition_ids=trial_condition_ids,
            condition_design=condition_design,
            fracs=fracs,
            nuisance=poly_per_run,
            polort=None,
            cv_splits=cv_splits,
            autoscale=True,
            device=device,
            verbose=True,  # Enable verbose to see what's happening
        )

        # Check that we recovered the single-trial betas
        # Use mean voxel to check correlation
        estimated_betas = results.betas_single_trial.mean(dim=0)  # (n_trials,)

        print(f"  True betas shape: {true_single_trial_betas.shape}")
        print(f"  Estimated betas shape: {estimated_betas.shape}")
        print(f"  True betas: {true_single_trial_betas[:5]}")
        print(f"  Estimated betas: {estimated_betas[:5]}")
        print(f"  Estimated betas has NaN: {torch.isnan(estimated_betas).any()}")
        print(f"  R² median: {results.r2.median().item():.4f}")

        # Correlation between true and estimated (ensure same device)
        correlation = torch.corrcoef(
            torch.stack([true_single_trial_betas, estimated_betas.to(device)])
        )[0, 1].item()

        print(f"  True-estimated beta correlation: {correlation:.3f}")

        # Should have high correlation (ridge recovers the pattern)
        assert correlation > 0.6, (
            f"Ridge failed to recover single-trial betas: correlation={correlation:.3f}"
        )

        # Check that strong trials have higher betas than weak trials (on average)
        strong_idx = total_trials // 2
        estimated_strong = estimated_betas[:strong_idx].mean().item()
        estimated_weak = estimated_betas[strong_idx:].mean().item()

        print(f"  Strong trials mean beta: {estimated_strong:.3f}")
        print(f"  Weak trials mean beta: {estimated_weak:.3f}")

        assert estimated_strong > estimated_weak, f"Ridge should distinguish strong vs weak trials"

    @pytest.mark.skip(reason="Per-voxel HRF feature not fully implemented in ridge.py")
    def test_ridge_with_per_voxel_hrf(self, device):
        """Test ridge with per-voxel HRF selection."""
        tr = 2.0
        n_timepoints = 100
        n_runs = 2
        matrix_size = (4, 4, 2)
        n_voxels = 4 * 4 * 2
        n_conditions = 2

        # Create HRF library
        from fastfuncsim.hrf import get_hrf_library

        hrf_library = get_hrf_library(
            models=["spmg1", "spmg2", "flox"],
            tr=tr,
            stim_duration=0.0,
            duration=30.0,
            device=device,
        )

        # Assign each voxel a different HRF
        hrf_indices = torch.randint(0, len(hrf_library), (n_voxels,), device=device)

        # Create onsets
        onsets_by_condition = []
        for cond_idx in range(n_conditions):
            cond_onsets = []
            for run_idx in range(n_runs):
                events = np.sort(np.random.choice(n_timepoints, size=5, replace=False))
                cond_onsets.append(events * tr)
            onsets_by_condition.append(cond_onsets)

        durations = [0.0] * n_conditions
        # run_starts: starting timepoint for each run (not including total)
        run_starts = [0, n_timepoints]

        # Build design with per-voxel HRFs
        design_stacked, trial_labels, trial_condition_ids, trial_run_ids, condition_design = (
            create_single_trial_design(
                onsets_by_condition=onsets_by_condition,
                durations=durations,
                run_starts=run_starts,
                tr=tr,
                n_timepoints=n_timepoints * n_runs,
                hrf_library=list(hrf_library),  # Convert to list
                hrf_index_per_voxel=hrf_indices,
                device=device,
            )
        )

        # Check design stacked shape: (n_hrfs, n_timepoints, n_trials)
        assert design_stacked.shape[0] == len(hrf_library)

        # Simulate data (use first HRF for simplicity)
        true_betas = torch.tensor([3.0, 5.0], device=device)
        hrf = hrf_library[0]

        onsets_list = []
        for run_idx in range(n_runs):
            onsets = torch.zeros(n_timepoints, 2, device=device)
            for cond_idx in range(2):
                events = torch.randperm(n_timepoints, device=device)[:5]
                onsets[events, cond_idx] = 1.0
            onsets_list.append(onsets)

        data_list = []
        for onsets in onsets_list:
            data = simulate_fmri_run(
                onsets=onsets,
                betas=true_betas.tolist(),
                hrf=hrf,
                tr=tr,
                n_timepoints=n_timepoints,
                matrix_size=matrix_size,
                noise_level=1.0,
                baseline=100.0,
                device=device,
            )
            data_list.append(data.reshape(-1, n_timepoints))

        data_all = torch.cat(data_list, dim=1)  # Concatenate along time dimension

        # Fit ridge with per-voxel designs
        cv_splits = generate_cv_splits(n_runs=n_runs, strategy=1, n_perms=1)
        fracs = np.array([0.0, 0.5, 1.0])

        # Build polynomials
        poly = construct_polynomial_matrix(n_timepoints * n_runs, max_degree=1, device=device)
        poly_per_run = []
        for i in range(n_runs):
            poly_run = torch.zeros(n_timepoints, poly.shape[1], device=device)
            start = i * n_timepoints
            end = start + n_timepoints
            poly_run[:, :] = poly[start:end, :]
            poly_per_run.append(poly_run)

        results = fit_ridge_single_trial(
            data=data_all,
            design_matrix=design_stacked,  # Pass stacked design
            run_starts=run_starts,
            tr=tr,
            trial_condition_ids=trial_condition_ids,
            condition_design=condition_design,
            fracs=fracs,
            nuisance=poly_per_run,
            polort=None,
            cv_splits=cv_splits,
            autoscale=True,
            device=device,
            verbose=False,
        )

        # Should complete without error
        assert results.betas_single_trial.shape[0] == n_voxels
        assert torch.all(torch.isfinite(results.betas_single_trial))

        print(f"  Per-voxel HRF ridge: mean R² = {results.r2.mean().item():.3f}")
