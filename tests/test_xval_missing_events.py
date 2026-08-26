"""
Test cross-validation with missing stimulus events.

Uses synthetic data with known structure to test both "zero" and "nuisance"
strategies for handling events that are missing in train or test sets.

Test cases:
1. Baseline: All events present in all runs (sanity check)
2. Train missing events: Events present in test but not train
3. Test missing events: Events present in train but not test
4. Both missing same events: Event missing from both (no real impact)
5. Both missing different events: Complex case
6. Multiple runs (4 runs): Missing events across subsets of runs
"""

import numpy as np
import pytest
import torch

from fastfuncstuff.glm.xval import compute_xval_r2, generate_cv_splits


def create_true_betas(
    n_voxels: int,
    n_stim: int,
    n_nuisance: int,
    stim_snr: float = 2.0,
    nuisance_snr: float = 1.0,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create true beta values that will be shared across runs.

    Returns
    -------
    true_stim_betas : torch.Tensor
        True stimulus betas (n_voxels, n_stim)
    true_nuisance_betas : torch.Tensor
        True nuisance betas (n_voxels, n_nuisance)
    """
    rng = np.random.RandomState(seed)
    true_stim_betas = rng.randn(n_voxels, n_stim).astype(np.float32) * stim_snr
    true_nuisance_betas = rng.randn(n_voxels, n_nuisance).astype(np.float32) * nuisance_snr

    return torch.from_numpy(true_stim_betas), torch.from_numpy(true_nuisance_betas)


def create_synthetic_fmri_data(
    n_voxels: int,
    n_timepoints: int,
    n_stim: int,
    n_nuisance: int,
    true_stim_betas: torch.Tensor,
    true_nuisance_betas: torch.Tensor,
    noise_std: float = 1.0,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create synthetic fMRI data with known structure using provided true betas.

    Parameters
    ----------
    n_voxels : int
        Number of voxels
    n_timepoints : int
        Number of timepoints
    n_stim : int
        Number of stimulus regressors
    n_nuisance : int
        Number of nuisance regressors
    true_stim_betas : torch.Tensor
        True stimulus betas (n_voxels, n_stim) - SHARED across runs
    true_nuisance_betas : torch.Tensor
        True nuisance betas (n_voxels, n_nuisance) - SHARED across runs
    noise_std : float
        Standard deviation of noise
    seed : int
        Random seed for design matrices and noise (different per run)

    Returns
    -------
    data : torch.Tensor
        Synthetic fMRI data (n_voxels, n_timepoints)
    stim_design : torch.Tensor
        Stimulus design matrix (n_timepoints, n_stim)
    nuisance_design : torch.Tensor
        Nuisance design matrix (n_timepoints, n_nuisance)
    """
    rng = np.random.RandomState(seed)

    # Create design matrices (uncorrelated regressors)
    stim_design = rng.randn(n_timepoints, n_stim).astype(np.float32)
    nuisance_design = rng.randn(n_timepoints, n_nuisance).astype(np.float32)

    # Normalize columns to unit variance
    stim_design = stim_design / (stim_design.std(axis=0, keepdims=True) + 1e-8)
    nuisance_design = nuisance_design / (nuisance_design.std(axis=0, keepdims=True) + 1e-8)

    # Convert to torch
    stim_design = torch.from_numpy(stim_design)
    nuisance_design = torch.from_numpy(nuisance_design)

    # Generate data: Y = X_stim @ beta_stim + X_nuisance @ beta_nuisance + noise
    signal_stim = true_stim_betas @ stim_design.T  # (n_voxels, n_timepoints)
    signal_nuisance = true_nuisance_betas @ nuisance_design.T
    noise = rng.randn(n_voxels, n_timepoints).astype(np.float32) * noise_std
    noise = torch.from_numpy(noise)

    data = signal_stim + signal_nuisance + noise

    return data, stim_design, nuisance_design


def zero_out_events(
    stim_design: torch.Tensor,
    event_indices: list[int],
    timepoint_ranges: list[tuple[int, int]],
) -> torch.Tensor:
    """
    Zero out specific events in specific timepoint ranges.

    Parameters
    ----------
    stim_design : torch.Tensor
        Stimulus design matrix (n_timepoints, n_stim)
    event_indices : List[int]
        Which events to zero out
    timepoint_ranges : List[Tuple[int, int]]
        List of (start, end) timepoint ranges to zero out

    Returns
    -------
    modified_design : torch.Tensor
        Design matrix with events zeroed out
    """
    design = stim_design.clone()
    for start, end in timepoint_ranges:
        design[start:end, event_indices] = 0.0
    return design


def test_xval_baseline_all_events():
    """
    Test 1: Baseline - all events present in all runs.

    With 2 runs, 4 events, LORO CV should give good R² since we can
    predict all events.
    """
    n_voxels = 100
    n_timepoints_per_run = 100
    n_stim = 4
    n_nuisance = 2

    # Create SHARED true betas (same for all runs!)
    true_stim_betas, true_nuisance_betas = create_true_betas(
        n_voxels, n_stim, n_nuisance, stim_snr=3.0, nuisance_snr=1.0, seed=42
    )

    # Create data for 2 runs (different design/noise, same betas)
    data1, stim1, nuisance1 = create_synthetic_fmri_data(
        n_voxels,
        n_timepoints_per_run,
        n_stim,
        n_nuisance,
        true_stim_betas,
        true_nuisance_betas,
        noise_std=1.0,
        seed=100,
    )
    data2, stim2, nuisance2 = create_synthetic_fmri_data(
        n_voxels,
        n_timepoints_per_run,
        n_stim,
        n_nuisance,
        true_stim_betas,
        true_nuisance_betas,
        noise_std=1.0,
        seed=101,
    )

    # Concatenate runs
    data = torch.cat([data1, data2], dim=1)
    stim_design = torch.cat([stim1, stim2], dim=0)
    nuisance_design = torch.cat([nuisance1, nuisance2], dim=0)

    # Full design matrix
    design_matrix = torch.cat([stim_design, nuisance_design], dim=1)

    # Run info
    run_starts = [0, n_timepoints_per_run]
    stim_indices = list(range(n_stim))
    nuisance_indices = list(range(n_stim, n_stim + n_nuisance))

    # LORO CV
    cv_splits = generate_cv_splits(n_runs=2, strategy=1, n_perms=100)

    # Test both strategies (should give similar results when all events present)
    for strategy in ["zero", "nuisance"]:
        results = compute_xval_r2(
            data=data,
            design_matrix=design_matrix,
            run_starts=run_starts,
            stim_indices=stim_indices,
            nuisance_indices=nuisance_indices,
            cv_splits=cv_splits,
            metric="cod",
            zero_event_strategy=strategy,
            device=torch.device("cpu"),
            verbose=False,
        )

        # Check results
        assert results["r2_median"].shape == (n_voxels,)
        assert results["n_splits"] == 2  # 2 LORO splits

        # With good SNR and all events present, R² should be positive
        median_r2 = results["r2_median"].median().item()
        mean_r2 = results["r2_median"].mean().item()

        print(f"\nBaseline (strategy={strategy}):")
        print(f"  Median R²: {median_r2:.4f}")
        print(f"  Mean R²: {mean_r2:.4f}")

        # Should get decent R² with SNR=3.0
        assert mean_r2 > 0.2, f"Expected positive R² with all events, got {mean_r2:.4f}"


def test_xval_train_missing_events():
    """
    Test 2: Train missing events - events present in test but not train.

    Event 2 is present in run 1 but missing (zeroed) in run 2.
    When we train on run 2 and test on run 1, we have no beta for event 2.

    Expected:
    - "zero" strategy: Should give lower R² (missing predictions)
    - "nuisance" strategy: N/A (can't project out what's not in train)
    - Both should handle gracefully without errors
    """
    n_voxels = 100
    n_timepoints_per_run = 100
    n_stim = 4
    n_nuisance = 2
    missing_event = 2  # Event index to make missing

    # Create SHARED true betas
    true_stim_betas, true_nuisance_betas = create_true_betas(
        n_voxels, n_stim, n_nuisance, stim_snr=3.0, nuisance_snr=1.0, seed=42
    )

    # Create data for 2 runs
    data1, stim1, nuisance1 = create_synthetic_fmri_data(
        n_voxels,
        n_timepoints_per_run,
        n_stim,
        n_nuisance,
        true_stim_betas,
        true_nuisance_betas,
        noise_std=1.0,
        seed=100,
    )
    data2, stim2, nuisance2 = create_synthetic_fmri_data(
        n_voxels,
        n_timepoints_per_run,
        n_stim,
        n_nuisance,
        true_stim_betas,
        true_nuisance_betas,
        noise_std=1.0,
        seed=101,
    )

    # Zero out event 2 in run 2 (train will be missing this event)
    stim2 = zero_out_events(stim2, [missing_event], [(0, n_timepoints_per_run)])

    # Concatenate runs
    data = torch.cat([data1, data2], dim=1)
    stim_design = torch.cat([stim1, stim2], dim=0)
    nuisance_design = torch.cat([nuisance1, nuisance2], dim=0)
    design_matrix = torch.cat([stim_design, nuisance_design], dim=1)

    # Run info
    run_starts = [0, n_timepoints_per_run]
    stim_indices = list(range(n_stim))
    nuisance_indices = list(range(n_stim, n_stim + n_nuisance))

    # LORO CV
    cv_splits = generate_cv_splits(n_runs=2, strategy=1, n_perms=100)

    # Test both strategies
    for strategy in ["zero", "nuisance"]:
        results = compute_xval_r2(
            data=data,
            design_matrix=design_matrix,
            run_starts=run_starts,
            stim_indices=stim_indices,
            nuisance_indices=nuisance_indices,
            cv_splits=cv_splits,
            metric="cod",
            zero_event_strategy=strategy,
            device=torch.device("cpu"),
            verbose=False,
        )

        mean_r2 = results["r2_median"].mean().item()
        print(f"\nTrain missing event (strategy={strategy}):")
        print(f"  Mean R²: {mean_r2:.4f}")

        # Should still get reasonable R² for other events
        # But lower than baseline since we're missing predictions for 1/4 events
        assert torch.isfinite(results["r2_median"]).all()
        assert mean_r2 > -0.5, f"R² too negative: {mean_r2:.4f}"


def test_xval_test_missing_events():
    """
    Test 3: Test missing events - events present in train but not test.

    Event 2 is present in run 2 but missing (zeroed) in run 1.
    When we train on run 2 and test on run 1, we have beta but no test signal.

    Expected:
    - "zero" strategy: Beta exists but no test signal to predict
    - "nuisance" strategy: Should project out event 2 from test data
    - Nuisance strategy should give better R² than zero strategy
    """
    n_voxels = 100
    n_timepoints_per_run = 100
    n_stim = 4
    n_nuisance = 2
    missing_event = 2

    # Create SHARED true betas
    true_stim_betas, true_nuisance_betas = create_true_betas(
        n_voxels, n_stim, n_nuisance, stim_snr=3.0, nuisance_snr=1.0, seed=42
    )

    # Create data for 2 runs
    data1, stim1, nuisance1 = create_synthetic_fmri_data(
        n_voxels,
        n_timepoints_per_run,
        n_stim,
        n_nuisance,
        true_stim_betas,
        true_nuisance_betas,
        noise_std=1.0,
        seed=100,
    )
    data2, stim2, nuisance2 = create_synthetic_fmri_data(
        n_voxels,
        n_timepoints_per_run,
        n_stim,
        n_nuisance,
        true_stim_betas,
        true_nuisance_betas,
        noise_std=1.0,
        seed=101,
    )

    # Zero out event 2 in run 1 (test will be missing this event)
    stim1 = zero_out_events(stim1, [missing_event], [(0, n_timepoints_per_run)])

    # Also zero out in the DATA for run 1 (no signal for this event)
    # This simulates the event truly not happening in this run
    data1_signal = (
        true_stim_betas[:, missing_event : missing_event + 1]
        @ stim1[:, missing_event : missing_event + 1].T
    )
    data1 = data1 - data1_signal  # Remove signal for event 2

    # Concatenate runs
    data = torch.cat([data1, data2], dim=1)
    stim_design = torch.cat([stim1, stim2], dim=0)
    nuisance_design = torch.cat([nuisance1, nuisance2], dim=0)
    design_matrix = torch.cat([stim_design, nuisance_design], dim=1)

    # Run info
    run_starts = [0, n_timepoints_per_run]
    stim_indices = list(range(n_stim))
    nuisance_indices = list(range(n_stim, n_stim + n_nuisance))

    # LORO CV
    cv_splits = generate_cv_splits(n_runs=2, strategy=1, n_perms=100)

    # Test both strategies
    results_by_strategy = {}
    for strategy in ["zero", "nuisance"]:
        results = compute_xval_r2(
            data=data,
            design_matrix=design_matrix,
            run_starts=run_starts,
            stim_indices=stim_indices,
            nuisance_indices=nuisance_indices,
            cv_splits=cv_splits,
            metric="cod",
            zero_event_strategy=strategy,
            device=torch.device("cpu"),
            verbose=False,
        )

        mean_r2 = results["r2_median"].mean().item()
        results_by_strategy[strategy] = mean_r2

        print(f"\nTest missing event (strategy={strategy}):")
        print(f"  Mean R²: {mean_r2:.4f}")

        assert torch.isfinite(results["r2_median"]).all()

    # Nuisance strategy should perform better (projects out unpredictable event)
    # Zero strategy has beta for missing event which adds unexplained variance
    print("\nComparison:")
    print(
        f"  Nuisance improvement: {results_by_strategy['nuisance'] - results_by_strategy['zero']:.4f}"
    )

    # Both should be reasonable
    assert results_by_strategy["zero"] > -1.0
    assert results_by_strategy["nuisance"] > -1.0


def test_xval_both_missing_same_events():
    """
    Test 4: Both train and test missing same events.

    Event 2 is missing from both runs. This is effectively like having
    only 3 events instead of 4. Should work fine.

    Expected:
    - Both strategies should give similar results
    - R² should be reasonable (based on 3 events not 4)
    """
    n_voxels = 100
    n_timepoints_per_run = 100
    n_stim = 4
    n_nuisance = 2
    missing_event = 2

    # Create SHARED true betas
    true_stim_betas, true_nuisance_betas = create_true_betas(
        n_voxels, n_stim, n_nuisance, stim_snr=3.0, nuisance_snr=1.0, seed=42
    )

    # Create data for 2 runs
    data1, stim1, nuisance1 = create_synthetic_fmri_data(
        n_voxels,
        n_timepoints_per_run,
        n_stim,
        n_nuisance,
        true_stim_betas,
        true_nuisance_betas,
        noise_std=1.0,
        seed=100,
    )
    data2, stim2, nuisance2 = create_synthetic_fmri_data(
        n_voxels,
        n_timepoints_per_run,
        n_stim,
        n_nuisance,
        true_stim_betas,
        true_nuisance_betas,
        noise_std=1.0,
        seed=101,
    )

    # Zero out event 2 in both runs
    stim1 = zero_out_events(stim1, [missing_event], [(0, n_timepoints_per_run)])
    stim2 = zero_out_events(stim2, [missing_event], [(0, n_timepoints_per_run)])

    # Remove signal from data
    data1 = data1 - (
        true_stim_betas[:, missing_event : missing_event + 1]
        @ stim1[:, missing_event : missing_event + 1].T
    )
    data2 = data2 - (
        true_stim_betas[:, missing_event : missing_event + 1]
        @ stim2[:, missing_event : missing_event + 1].T
    )

    # Concatenate runs
    data = torch.cat([data1, data2], dim=1)
    stim_design = torch.cat([stim1, stim2], dim=0)
    nuisance_design = torch.cat([nuisance1, nuisance2], dim=0)
    design_matrix = torch.cat([stim_design, nuisance_design], dim=1)

    # Run info
    run_starts = [0, n_timepoints_per_run]
    stim_indices = list(range(n_stim))
    nuisance_indices = list(range(n_stim, n_stim + n_nuisance))

    # LORO CV
    cv_splits = generate_cv_splits(n_runs=2, strategy=1, n_perms=100)

    # Test both strategies
    for strategy in ["zero", "nuisance"]:
        results = compute_xval_r2(
            data=data,
            design_matrix=design_matrix,
            run_starts=run_starts,
            stim_indices=stim_indices,
            nuisance_indices=nuisance_indices,
            cv_splits=cv_splits,
            metric="cod",
            zero_event_strategy=strategy,
            device=torch.device("cpu"),
            verbose=False,
        )

        mean_r2 = results["r2_median"].mean().item()
        print(f"\nBoth missing same event (strategy={strategy}):")
        print(f"  Mean R²: {mean_r2:.4f}")

        # Should be similar to baseline (just with 3 effective events)
        assert torch.isfinite(results["r2_median"]).all()
        assert mean_r2 > 0.2, f"Expected good R², got {mean_r2:.4f}"


def test_xval_both_missing_different_events():
    """
    Test 5: Train and test missing different events.

    Event 1 is missing from run 1 (test).
    Event 2 is missing from run 2 (train).

    When training on run 2, we can't learn event 2.
    When testing on run 1, event 1 is missing.

    This is the most complex case - combines cases 2 and 3.

    Expected:
    - "zero" strategy: Missing betas for some events, missing test signal for others
    - "nuisance" strategy: Should handle both by projecting out unpredictable
    """
    n_voxels = 100
    n_timepoints_per_run = 100
    n_stim = 4
    n_nuisance = 2

    # Create SHARED true betas
    true_stim_betas, true_nuisance_betas = create_true_betas(
        n_voxels, n_stim, n_nuisance, stim_snr=3.0, nuisance_snr=1.0, seed=42
    )

    # Create data for 2 runs
    data1, stim1, nuisance1 = create_synthetic_fmri_data(
        n_voxels,
        n_timepoints_per_run,
        n_stim,
        n_nuisance,
        true_stim_betas,
        true_nuisance_betas,
        noise_std=1.0,
        seed=100,
    )
    data2, stim2, nuisance2 = create_synthetic_fmri_data(
        n_voxels,
        n_timepoints_per_run,
        n_stim,
        n_nuisance,
        true_stim_betas,
        true_nuisance_betas,
        noise_std=1.0,
        seed=101,
    )

    # Zero out event 1 in run 1 (test missing)
    stim1 = zero_out_events(stim1, [1], [(0, n_timepoints_per_run)])
    data1 = data1 - (true_stim_betas[:, 1:2] @ stim1[:, 1:2].T)

    # Zero out event 2 in run 2 (train missing)
    stim2 = zero_out_events(stim2, [2], [(0, n_timepoints_per_run)])
    data2 = data2 - (true_stim_betas[:, 2:3] @ stim2[:, 2:3].T)

    # Concatenate runs
    data = torch.cat([data1, data2], dim=1)
    stim_design = torch.cat([stim1, stim2], dim=0)
    nuisance_design = torch.cat([nuisance1, nuisance2], dim=0)
    design_matrix = torch.cat([stim_design, nuisance_design], dim=1)

    # Run info
    run_starts = [0, n_timepoints_per_run]
    stim_indices = list(range(n_stim))
    nuisance_indices = list(range(n_stim, n_stim + n_nuisance))

    # LORO CV
    cv_splits = generate_cv_splits(n_runs=2, strategy=1, n_perms=100)

    # Test both strategies
    for strategy in ["zero", "nuisance"]:
        results = compute_xval_r2(
            data=data,
            design_matrix=design_matrix,
            run_starts=run_starts,
            stim_indices=stim_indices,
            nuisance_indices=nuisance_indices,
            cv_splits=cv_splits,
            metric="cod",
            zero_event_strategy=strategy,
            device=torch.device("cpu"),
            verbose=False,
        )

        mean_r2 = results["r2_median"].mean().item()
        print(f"\nBoth missing different events (strategy={strategy}):")
        print(f"  Mean R²: {mean_r2:.4f}")

        # Complex case but should still be reasonable
        assert torch.isfinite(results["r2_median"]).all()
        assert mean_r2 > -1.0, f"R² too negative: {mean_r2:.4f}"


def test_xval_multiple_runs_missing_events():
    """
    Test 6: Multiple runs (4 runs) with missing events across subsets.

    Setup:
    - 4 runs, 6 events
    - Event 2 missing from runs 1 and 2 (but present in 3 and 4)
    - Event 4 missing from runs 3 and 4 (but present in 1 and 2)
    - LORO CV with 4 runs = 4 splits

    Important: An event is considered "missing" from train/test only if it's
    missing from ALL runs in that set, not just one run.

    Expected:
    - When training on runs {1,2,4} and testing on run 3:
      * Event 4 is missing from run 3 (test) but present in train (runs 1,2)
      * Nuisance strategy should help
    - When training on runs {1,3,4} and testing on run 2:
      * Event 2 is missing from run 2 (test) but present in train (runs 3,4)
      * Similar situation
    """
    n_voxels = 100
    n_timepoints_per_run = 100
    n_stim = 6
    n_nuisance = 2

    # Create SHARED true betas
    true_stim_betas, true_nuisance_betas = create_true_betas(
        n_voxels, n_stim, n_nuisance, stim_snr=3.0, nuisance_snr=1.0, seed=42
    )

    # Create 4 runs
    runs_data = []
    runs_stim = []
    runs_nuisance = []

    for i in range(4):
        data, stim, nuisance = create_synthetic_fmri_data(
            n_voxels,
            n_timepoints_per_run,
            n_stim,
            n_nuisance,
            true_stim_betas,
            true_nuisance_betas,
            noise_std=1.0,
            seed=100 + i,
        )
        runs_data.append(data)
        runs_stim.append(stim)
        runs_nuisance.append(nuisance)

    # Zero out event 2 from runs 0 and 1
    for i in [0, 1]:
        runs_stim[i] = zero_out_events(runs_stim[i], [2], [(0, n_timepoints_per_run)])
        runs_data[i] = runs_data[i] - (true_stim_betas[:, 2:3] @ runs_stim[i][:, 2:3].T)

    # Zero out event 4 from runs 2 and 3
    for i in [2, 3]:
        runs_stim[i] = zero_out_events(runs_stim[i], [4], [(0, n_timepoints_per_run)])
        runs_data[i] = runs_data[i] - (true_stim_betas[:, 4:5] @ runs_stim[i][:, 4:5].T)

    # Concatenate all runs
    data = torch.cat(runs_data, dim=1)
    stim_design = torch.cat(runs_stim, dim=0)
    nuisance_design = torch.cat(runs_nuisance, dim=0)
    design_matrix = torch.cat([stim_design, nuisance_design], dim=1)

    # Run info
    run_starts = [i * n_timepoints_per_run for i in range(4)]
    stim_indices = list(range(n_stim))
    nuisance_indices = list(range(n_stim, n_stim + n_nuisance))

    # LORO CV with 4 runs = 4 splits
    cv_splits = generate_cv_splits(n_runs=4, strategy=1, n_perms=100)

    print(f"\nGenerated {len(cv_splits)} CV splits for 4 runs")

    # Test both strategies
    for strategy in ["zero", "nuisance"]:
        results = compute_xval_r2(
            data=data,
            design_matrix=design_matrix,
            run_starts=run_starts,
            stim_indices=stim_indices,
            nuisance_indices=nuisance_indices,
            cv_splits=cv_splits,
            metric="cod",
            zero_event_strategy=strategy,
            device=torch.device("cpu"),
            verbose=False,
        )

        mean_r2 = results["r2_median"].mean().item()
        n_splits = results["n_splits"]

        print(f"\nMultiple runs (4 runs, strategy={strategy}):")
        print(f"  Splits: {n_splits}")
        print(f"  Mean R²: {mean_r2:.4f}")

        # Should work with 4 runs
        assert n_splits == 4
        assert torch.isfinite(results["r2_median"]).all()
        assert mean_r2 > 0.0, f"Expected positive R² with 4 runs, got {mean_r2:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


def _collinear_missing_event_dataset(
    overlap: bool, n_voxels: int = 60, n_tps: int = 60, seed: int = 7
):
    """Two runs; run 2 carries a condition that run 1 never sees.

    With ``overlap=True`` that unpredictable condition's boxcar sits directly on
    top of a predictable one, which is the case the FWL fix exists for. Data is
    noiseless and exactly linear in the design, so a correct estimator recovers
    R2 = 1 in the subspace it is allowed to be scored in.
    """
    rng = np.random.RandomState(seed)
    n_stim = 3  # cols 0,1 predictable; col 2 present only in run 2

    stim = torch.zeros(2 * n_tps, n_stim)
    for start in range(0, n_tps, 20):  # col 0 in both runs
        stim[start : start + 6, 0] = 1.0
        stim[n_tps + start : n_tps + start + 6, 0] = 1.0
    for start in range(8, n_tps, 20):  # col 1 in both runs
        stim[start : start + 6, 1] = 1.0
        stim[n_tps + start : n_tps + start + 6, 1] = 1.0
    # col 2: run 2 only, half-overlapping col 0's blocks or clear of everything.
    # PARTIAL overlap on purpose: a col 2 laid exactly on col 0 would make the two
    # indistinguishable inside run 2, so the fold that trains there could not
    # separate them and R2 < 1 would be the correct answer rather than a bug.
    offset = 3 if overlap else 14
    for start in range(offset, n_tps, 20):
        stim[n_tps + start : n_tps + start + 6, 2] = 1.0

    betas = torch.from_numpy(rng.randn(n_voxels, n_stim).astype(np.float32))
    data = betas @ stim.T  # noiseless

    return data, stim, [0, n_tps], list(range(n_stim))


@pytest.mark.parametrize("overlap", [True, False])
def test_nuisance_strategy_is_not_penalised_by_collinear_missing_events(overlap):
    """'nuisance' must project the prediction design, not just the test data.

    Bug of record: the projector was applied to the held-out data while the
    design used to predict it was left alone. Whatever a predictable condition
    shared with an unpredictable one was then removed from one side only and
    landed in the residual as fabricated error -- so R2 fell precisely when an
    unmodellable event sat near a real one, which is the opposite of what the
    strategy is for. Noiseless data makes the correct answer exactly 1.0.
    """
    data, stim, run_starts, stim_indices = _collinear_missing_event_dataset(overlap)

    results = compute_xval_r2(
        data=data,
        design_matrix=stim,
        run_starts=run_starts,
        stim_indices=stim_indices,
        nuisance_indices=[],
        cv_splits=generate_cv_splits(n_runs=2, strategy=1),
        metric="cod",
        zero_event_strategy="nuisance",
        device=torch.device("cpu"),
        verbose=False,
    )
    r2 = results["r2"].median().item()
    assert r2 > 0.99, f"overlap={overlap}: 'nuisance' should recover R2~1, got {r2:.4f}"


def test_nuisance_beats_zero_when_an_event_is_unpredictable():
    """The whole point of the strategy: don't score a fold on what it couldn't know.

    'zero' has no beta for the run-2-only condition, so its variance is charged
    to the residual and R2 collapses. 'nuisance' declines to score that subspace.
    """
    data, stim, run_starts, stim_indices = _collinear_missing_event_dataset(overlap=True)

    scores = {}
    for strategy in ("zero", "nuisance"):
        results = compute_xval_r2(
            data=data,
            design_matrix=stim,
            run_starts=run_starts,
            stim_indices=stim_indices,
            nuisance_indices=[],
            cv_splits=generate_cv_splits(n_runs=2, strategy=1),
            metric="cod",
            zero_event_strategy=strategy,
            device=torch.device("cpu"),
            verbose=False,
        )
        scores[strategy] = results["r2"].median().item()

    assert scores["nuisance"] > scores["zero"], scores


def test_unpredictable_basis_ignores_train_only_columns():
    """Train-only columns are zero across the test set and must not enter the basis.

    They used to, as all-zero columns, which is what made the ridge-stabilised
    inverse necessary in the first place.
    """
    from fastfuncstuff.glm.xval import _plan_fold_designs

    n_tps, n_stim = 40, 3
    train = torch.zeros(n_tps, n_stim)
    test = torch.zeros(n_tps, n_stim)
    train[:10, 0] = 1.0
    test[:10, 0] = 1.0  # predictable
    train[10:20, 1] = 1.0  # train-only: absent from test
    test[20:30, 2] = 1.0  # test-only: the genuinely unpredictable one

    plan = _plan_fold_designs(
        train_stim_design=train,
        test_stim_design=test,
        zero_event_strategy="nuisance",
        train_runs=[0],
        test_runs=[1],
        device=torch.device("cpu"),
        announce=False,
    )

    assert plan["unpredictable_mask"].tolist() == [False, True, False]
    assert plan["test_only_mask"].tolist() == [False, False, True]
    basis = plan["unpred_basis"]
    assert basis is not None
    assert basis.shape == (n_tps, 1), "only the test-only column spans the removed subspace"
    # Orthonormal, and orthogonal to the predictable design it was applied to.
    assert torch.allclose(basis.T @ basis, torch.eye(1), atol=1e-5)
    assert torch.allclose(basis.T @ plan["test_stim_predictable"], torch.zeros(1, 2), atol=1e-5)
