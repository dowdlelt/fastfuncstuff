"""
Cross-validation utilities for GLM analysis.

This module provides functions for computing cross-validated R² metrics
using run-based train/test splits. The main use case is testing denoising
methods and model selection (e.g., HRF choice).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from itertools import combinations


def project_out_nuisance_per_run(
    data: torch.Tensor,
    design: torch.Tensor,
    nuisance_per_run: List[torch.Tensor],
    run_starts: List[int],
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project out nuisance regressors from data and design matrix, per run.

    This is the GLMdenoise/GLMsingle approach:
    1. For each run, compute projection matrix from that run's nuisance
    2. Apply projection to both data and design
    3. Concatenate projected runs

    This avoids numerical issues with block-diagonal nuisance matrices
    during cross-validation (zero-padded columns cause singular matrices).

    Parameters
    ----------
    data : torch.Tensor
        (n_voxels, n_timepoints) fMRI data
    design : torch.Tensor
        (n_timepoints, n_task_cols) Task design matrix (NO nuisance columns)
    nuisance_per_run : List[torch.Tensor]
        List of (run_length, n_nuisance_cols) nuisance matrices per run
    run_starts : List[int]
        Starting timepoint for each run
    device : torch.device, optional
        Compute device

    Returns
    -------
    projected_data : torch.Tensor
        (n_voxels, n_timepoints) Data with nuisance projected out per run
    projected_design : torch.Tensor
        (n_timepoints, n_task_cols) Design with nuisance projected out per run
    """
    if device is None or device != data.device:
        device = data.device

    n_runs = len(run_starts)
    n_timepoints = data.shape[1]

    # CRITICAL: Keep data on its original device (may be CPU for memory efficiency)
    # Only the projection computation happens on the compute device
    data_device = data.device

    projected_data_runs = []
    projected_design_runs = []

    for run_idx in range(n_runs):
        start_tp = run_starts[run_idx]
        end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints

        # Get this run's data and design (keep on original device)
        run_data = data[:, start_tp:end_tp]  # (n_voxels, run_length)
        run_design = design[start_tp:end_tp, :]  # (run_length, n_task)
        run_length = end_tp - start_tp

        # Get this run's nuisance (on original device)
        run_nuisance = nuisance_per_run[run_idx].to(data_device)  # (run_length, n_nuisance)

        # CRITICAL: Remove zero columns from nuisance before projection
        # Zero columns (from padding) can cause numerical issues with ridge regularization
        col_norms = run_nuisance.abs().sum(dim=0)
        nonzero_mask = col_norms > 1e-10
        run_nuisance_clean = run_nuisance[:, nonzero_mask]

        if run_nuisance_clean.shape[1] > 0:
            # Compute projection matrix: P_perp = I - X(X'X)^-1 X'
            XtX = run_nuisance_clean.T @ run_nuisance_clean
            XtX_inv = torch.linalg.inv(XtX + 1e-6 * torch.eye(XtX.shape[0], device=data_device))
            P_nuisance = (
                run_nuisance_clean @ XtX_inv @ run_nuisance_clean.T
            )  # Projects ONTO nuisance
            projection = (
                torch.eye(run_length, device=data_device) - P_nuisance
            )  # Projects OUT nuisance

            # Project data: (n_voxels, run_length)
            run_data_proj = (projection @ run_data.T).T

            # Project design: (run_length, n_task)
            run_design_proj = projection @ run_design
        else:
            # No nuisance to project
            run_data_proj = run_data
            run_design_proj = run_design

        projected_data_runs.append(run_data_proj)
        projected_design_runs.append(run_design_proj)

    # Concatenate back
    projected_data = torch.cat(projected_data_runs, dim=1)  # (n_voxels, n_timepoints)
    projected_design = torch.cat(projected_design_runs, dim=0)  # (n_timepoints, n_task)

    return projected_data, projected_design


def generate_cv_splits(
    n_runs: int,
    strategy: Union[float, int],
    n_perms: int = 100,
) -> List[Tuple[List[int], List[int]]]:
    """
    Generate cross-validation splits for run-based CV.

    Parameters
    ----------
    n_runs : int
        Total number of runs
    strategy : float or int
        - float (0.0-1.0): Fraction for training (e.g., 0.5 = split halves)
        - int: Number of runs to leave out (e.g., 1 = LORO)
    n_perms : int, default=100
        Number of permutations to generate (for random split strategies)

    Returns
    -------
    splits : list of (train_runs, test_runs)
        Each element is (train_run_indices, test_run_indices)

    Examples
    --------
    >>> # Split halves (50/50)
    >>> generate_cv_splits(4, strategy=0.5, n_perms=6)
    [([0, 1], [2, 3]), ([0, 2], [1, 3]), ([0, 3], [1, 2]),
     ([1, 2], [0, 3]), ([1, 3], [0, 2]), ([2, 3], [0, 1])]

    >>> # Leave-one-run-out
    >>> generate_cv_splits(4, strategy=1)
    [([1,2,3], [0]), ([0,2,3], [1]), ([0,1,3], [2]), ([0,1,2], [3])]

    >>> # Leave-two-runs-out
    >>> generate_cv_splits(4, strategy=2)
    [([2,3], [0,1]), ([1,3], [0,2]), ([1,2], [0,3]),
     ([0,3], [1,2]), ([0,2], [1,3]), ([0,1], [2,3])]
    """
    run_indices = list(range(n_runs))

    if isinstance(strategy, float):
        # Split by fraction (e.g., 0.5 = half)
        if not 0.0 < strategy < 1.0:
            raise ValueError(f"Float strategy must be in (0.0, 1.0), got {strategy}")

        n_train = int(n_runs * strategy)
        if n_train == 0 or n_train == n_runs:
            raise ValueError(
                f"strategy={strategy} with n_runs={n_runs} results in "
                f"n_train={n_train} (must be > 0 and < n_runs)"
            )

        # Generate all possible combinations
        all_splits = []
        for train_runs in combinations(run_indices, n_train):
            train_runs = list(train_runs)
            test_runs = [r for r in run_indices if r not in train_runs]
            all_splits.append((train_runs, test_runs))

        # If we have more combinations than requested, sample randomly
        if len(all_splits) > n_perms:
            import random

            random.seed(42)  # Reproducible
            all_splits = random.sample(all_splits, n_perms)

        return all_splits

    elif isinstance(strategy, int):
        # Leave-N-out
        n_test = strategy
        if n_test <= 0 or n_test >= n_runs:
            raise ValueError(f"strategy={strategy} must be > 0 and < n_runs={n_runs}")

        # Generate all possible test sets
        all_splits = []
        for test_runs in combinations(run_indices, n_test):
            test_runs = list(test_runs)
            train_runs = [r for r in run_indices if r not in test_runs]
            all_splits.append((train_runs, test_runs))

        # If we have more combinations than requested, sample randomly
        if len(all_splits) > n_perms:
            import random

            random.seed(42)  # Reproducible
            all_splits = random.sample(all_splits, n_perms)

        return all_splits

    else:
        raise ValueError(f"strategy must be float or int, got {type(strategy)}")


def slice_by_runs(
    data: torch.Tensor,
    design_matrix: torch.Tensor,
    run_starts: List[int],
    run_indices: List[int],
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Extract data and design matrix for specific runs.

    Parameters
    ----------
    data : torch.Tensor
        fMRI data (n_voxels, n_timepoints)
    design_matrix : torch.Tensor
        Design matrix (n_timepoints, n_regressors)
    run_starts : list of int
        Starting timepoint for each run (from design_info["run_starts"])
    run_indices : list of int
        Which runs to include (0-indexed)

    Returns
    -------
    data_sliced : torch.Tensor
        Data for selected runs (n_voxels, n_timepoints_subset)
    design_sliced : torch.Tensor
        Design for selected runs (n_timepoints_subset, n_regressors)
    timepoint_indices : list of int
        Which timepoints were selected (for debugging)

    Examples
    --------
    >>> data = torch.randn(100, 800)  # 100 voxels, 800 timepoints
    >>> design = torch.randn(800, 50)
    >>> run_starts = [0, 200, 400, 600]  # 4 runs of 200 TRs each
    >>> # Select runs 0 and 2 (first and third)
    >>> data_sub, design_sub, tps = slice_by_runs(data, design, run_starts, [0, 2])
    >>> data_sub.shape
    torch.Size([100, 400])  # 2 runs × 200 TRs
    """
    # Compute run lengths
    n_timepoints_total = design_matrix.shape[0]
    run_lengths = np.diff(run_starts + [n_timepoints_total])

    # Collect timepoint indices for selected runs
    timepoint_indices = []
    for run_idx in run_indices:
        start = run_starts[run_idx]
        length = run_lengths[run_idx]
        timepoint_indices.extend(range(start, start + length))

    # Slice data and design
    data_sliced = data[:, timepoint_indices]
    design_sliced = design_matrix[timepoint_indices, :]

    return data_sliced, design_sliced, timepoint_indices


def project_out_nuisance(
    data: torch.Tensor,
    design_matrix: torch.Tensor,
    nuisance_indices: List[int],
    ridge: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project out nuisance regressors from data and design matrix.

    This removes variance explained by nuisance (motion, polynomials, etc.)
    from both the data and the full design matrix. Critical for cross-validation
    because we need to clean both train and test sets.

    Parameters
    ----------
    data : torch.Tensor
        Data to clean (n_voxels, n_timepoints)
    design_matrix : torch.Tensor
        Full design matrix (n_timepoints, n_regressors)
    nuisance_indices : list of int
        Which columns are nuisance regressors
    ridge : float, default=1e-6
        Ridge regularization for numerical stability

    Returns
    -------
    data_cleaned : torch.Tensor
        Data with nuisance projected out (n_voxels, n_timepoints)
    design_cleaned : torch.Tensor
        Design with nuisance projected out (n_timepoints, n_regressors)

    Notes
    -----
    Key steps:
    1. Extract nuisance design: X_nuis = design[:, nuisance_indices]
    2. **CRITICAL**: Remove all-zero columns (run-specific regressors!)
    3. Compute projection matrix: P = X @ (X.T @ X)^-1 @ X.T
    4. Project out: data_clean = data - P @ data
    5. Project out: design_clean = design - P @ design

    Why remove zero columns?
    ------------------------
    When we split by runs, some nuisance regressors (like run 3's polynomials)
    will be all-zero in splits that don't include run 3. We must remove these
    before computing the projection, or the matrix will be singular.

    Examples
    --------
    >>> data = torch.randn(100, 200)
    >>> design = torch.randn(200, 50)
    >>> nuisance_indices = list(range(40, 50))  # Last 10 columns are nuisance
    >>> data_clean, design_clean = project_out_nuisance(data, design, nuisance_indices)
    >>> # Verify nuisance is removed
    >>> # If we fit OLS on data_clean with design_clean[:, nuisance_indices],
    >>> # the betas should be ~0
    """
    if not nuisance_indices:
        # No nuisance regressors
        return data, design_matrix

    # Extract nuisance design
    X_nuis = design_matrix[:, nuisance_indices]

    # CRITICAL: Remove all-zero columns (run-specific regressors not in this split!)
    # Check column-wise variance to detect zero columns
    col_norms = X_nuis.abs().sum(dim=0)
    nonzero_mask = col_norms > 1e-10

    if not nonzero_mask.any():
        # All nuisance columns are zero (unusual but possible)
        return data, design_matrix

    X_nuis = X_nuis[:, nonzero_mask]

    # Compute projection matrix: P = X @ (X.T @ X)^-1 @ X.T
    # Using ridge regularization for numerical stability
    XtX = X_nuis.T @ X_nuis
    XtX_reg = XtX + ridge * torch.eye(XtX.shape[0], device=XtX.device, dtype=XtX.dtype)
    XtX_inv = torch.linalg.inv(XtX_reg)
    P = X_nuis @ XtX_inv @ X_nuis.T

    # Project out from data: data_clean = (I - P) @ data
    # data is (n_voxels, n_timepoints), P is (n_timepoints, n_timepoints)
    data_cleaned = data - (P @ data.T).T

    # Project out from design: design_clean = (I - P) @ design
    design_cleaned = design_matrix - P @ design_matrix

    return data_cleaned, design_cleaned


def compute_r2_metric(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    metric: str = "cod",
) -> torch.Tensor:
    """
    Compute R² metric between true and predicted data.

    Parameters
    ----------
    y_true : torch.Tensor
        True data (n_voxels, n_timepoints)
    y_pred : torch.Tensor
        Predicted data (n_voxels, n_timepoints)
    metric : str, default='cod'
        'cod': Coefficient of determination (1 - SS_res/SS_tot)
               Traditional R², can be negative if prediction is worse than mean
        'corr': Pearson correlation coefficient (range: -1 to 1)
        'corr2': Pearson correlation squared (range: 0 to 1)

    Returns
    -------
    r2 : torch.Tensor
        R² metric for each voxel (n_voxels,), dtype=float32

    Notes
    -----
    CoD vs. Correlation:
    - CoD measures prediction accuracy (can be negative)
    - Correlation measures linear relationship (always positive when squared)
    - They are similar but not identical for good predictions

    Examples
    --------
    >>> y_true = torch.randn(100, 200)
    >>> y_pred = y_true + 0.1 * torch.randn(100, 200)  # Add noise
    >>> r2_cod = compute_r2_metric(y_true, y_pred, 'cod')
    >>> r2_corr2 = compute_r2_metric(y_true, y_pred, 'corr2')
    >>> # Both should be high (~0.99) and similar
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    if metric == "cod":
        # Coefficient of determination: R² = 1 - SS_res/SS_tot
        ss_res = ((y_true - y_pred) ** 2).sum(dim=1)
        y_mean = y_true.mean(dim=1, keepdim=True)
        ss_tot = ((y_true - y_mean) ** 2).sum(dim=1)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-10))
        # Clamp max to 1.0 (values > 1 indicate numerical issues, e.g. zero variance)
        # Note: R² CAN be negative (model worse than mean) - that's valid information
        r2 = torch.clamp(r2, max=1.0)

    elif metric == "corr":
        # Pearson correlation coefficient
        y_true_centered = y_true - y_true.mean(dim=1, keepdim=True)
        y_pred_centered = y_pred - y_pred.mean(dim=1, keepdim=True)

        numerator = (y_true_centered * y_pred_centered).sum(dim=1)
        denom_true = torch.sqrt((y_true_centered**2).sum(dim=1))
        denom_pred = torch.sqrt((y_pred_centered**2).sum(dim=1))
        denominator = denom_true * denom_pred

        r2 = numerator / (denominator + 1e-10)

    elif metric == "corr2":
        # Pearson correlation squared
        r = compute_r2_metric(y_true, y_pred, metric="corr")
        r2 = r**2

    else:
        raise ValueError(f"Unknown metric '{metric}'. Choose from: 'cod', 'corr', 'corr2'")

    return r2.float()  # Cast to float32 to save space


def _compute_projection_matrix(
    design_matrix: torch.Tensor,
    nuisance_indices: List[int],
    ridge: float = 1e-6,
) -> Optional[torch.Tensor]:
    """
    Compute projection matrix P for nuisance regressors.

    Returns P such that: cleaned_data = data - P @ data

    This is a helper to avoid recomputing P for every voxel batch.
    """
    if not nuisance_indices:
        return None

    # Extract nuisance design
    X_nuis = design_matrix[:, nuisance_indices]

    # Remove all-zero columns (run-specific regressors not in this split)
    col_norms = X_nuis.abs().sum(dim=0)
    nonzero_mask = col_norms > 1e-10

    if not nonzero_mask.any():
        # All nuisance columns are zero
        return None

    X_nuis = X_nuis[:, nonzero_mask]

    # Compute projection matrix: P = X @ (X.T @ X)^-1 @ X.T
    XtX = X_nuis.T @ X_nuis
    XtX_reg = XtX + ridge * torch.eye(XtX.shape[0], device=XtX.device, dtype=XtX.dtype)
    XtX_inv = torch.linalg.inv(XtX_reg)
    P = X_nuis @ XtX_inv @ X_nuis.T

    return P


def compute_xval_r2(
    data: torch.Tensor,
    design_matrix: Union[np.ndarray, torch.Tensor],
    run_starts: List[int],
    stim_indices: List[int],
    nuisance_indices: List[int],
    cv_splits: List[Tuple[List[int], List[int]]],
    metric: str = "cod",
    zero_event_strategy: str = "zero",
    device: Optional[torch.device] = None,
    batch_size: Optional[int] = None,
    data_chunk_size: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Compute cross-validated R² using run-based train/test splits (GLMdenoise-style).

    Strategy (GLMdenoise/GLMsingle approach):
    -----------------------------------------
    Instead of computing R² per fold and averaging, we:
    1. For each CV split, predict the held-out test data
    2. Concatenate ALL predictions across all folds (rebuilding full timeseries)
    3. Concatenate ALL actual data across all folds
    4. Compute ONE R² from the concatenated prediction vs actual

    This gives a single R² per voxel (not one per fold that gets aggregated).
    The concatenated prediction covers the entire timeseries exactly once
    (each timepoint appears in exactly one test fold).

    For each CV split (train_runs, test_runs):
        1. Slice data and design by runs
        2. Project out nuisance from train data & design
        3. Project out nuisance from test data & design
        4. Extract stimulus design (after projection)
        5. Fit OLS on cleaned train data with cleaned train stim design
        6. Predict cleaned test data using train betas and cleaned test stim design
        7. Store predictions (don't compute R² yet)

    After all splits:
        - Concatenate predictions in original timepoint order
        - Compute single R² between full concatenated prediction and actual

    Parameters
    ----------
    data : torch.Tensor
        fMRI data (n_voxels, n_timepoints)
    design_matrix : np.ndarray or torch.Tensor
        Full design matrix (n_timepoints, n_regressors)
    run_starts : list of int
        Starting timepoint for each run
    stim_indices : list of int
        Column indices for stimulus regressors
    nuisance_indices : list of int
        Column indices for nuisance regressors (motion, polynomials, etc.)
    cv_splits : list of (train_runs, test_runs)
        Cross-validation splits from generate_cv_splits()
    metric : str, default='cod'
        R² metric: 'cod', 'corr', or 'corr2'
    zero_event_strategy : str, default='zero'
        How to handle events (stimulus columns) that are missing in train or test:
        - 'zero': Insert zero betas for missing events. Predictions only use present
          events. This reduces R² but keeps predictions valid.
        - 'nuisance': Move unpredictable events (missing in train) to nuisance in test.
          This removes variance from those events before prediction, allowing prediction
          of the remainder. More sophisticated but assumes missing events are noise.
    device : torch.device, optional
        Compute device (auto-detected if None)
    batch_size : int, optional
        Voxels per batch for computation (auto-detected if None).
        Used for batching the projection operation to avoid OOM.
    data_chunk_size : int, optional
        Number of voxels to load to GPU at once (auto-detected if None).
        If data is too large to fit on GPU, it will be processed in chunks.
        Each chunk runs all CV splits before moving to the next chunk.
    verbose : bool, default=True
        Print progress

    Returns
    -------
    results : dict
        'r2': (n_voxels,) single R² computed from concatenated predictions
        'n_splits': int, number of CV splits performed

    Examples
    --------
    >>> # Setup
    >>> data = torch.randn(1000, 800)
    >>> design = np.random.randn(800, 50)
    >>> run_starts = [0, 200, 400, 600]
    >>> stim_indices = list(range(40))  # First 40 are stimulus
    >>> nuisance_indices = list(range(40, 50))  # Last 10 are nuisance
    >>>
    >>> # Generate splits
    >>> cv_splits = generate_cv_splits(n_runs=4, strategy=0.5, n_perms=6)
    >>>
    >>> # Compute xval R²
    >>> results = compute_xval_r2(
    ...     data, design, run_starts, stim_indices, nuisance_indices, cv_splits
    ... )
    >>> print(f"Xval R²: {results['r2'].mean():.3f}")
    """
    from .utils import get_device, to_tensor

    if device is None:
        device = get_device()

    # Design matrix is small - keep on device
    design_matrix = to_tensor(design_matrix, device=device, dtype=torch.float32)

    n_voxels = data.shape[0]
    n_timepoints = data.shape[1]
    n_splits = len(cv_splits)
    n_runs = len(run_starts)

    # Auto-detect batch size if not provided
    if batch_size is None:
        # Conservative: 5000 voxels per batch for projection
        batch_size = min(5000, n_voxels)

    # Auto-detect data chunk size if not provided
    if data_chunk_size is None:
        # Try to estimate based on available GPU memory
        # Assume we need ~40 bytes per voxel per timepoint (including temporaries)
        # For 16 GB GPU, with 3000 timepoints: ~130k voxels max
        # Be conservative: 100k voxels
        data_chunk_size = min(100_000, n_voxels)

    # GLMdenoise-style: We accumulate predictions for each run
    # Each run's test data is predicted exactly once across all CV folds
    # We need to track which runs have been predicted
    # For LORO (leave-one-run-out), each run is test exactly once
    # For other strategies, we need to handle overlapping test sets

    if verbose:
        print("Cross-validation R² computation (GLMdenoise-style concatenation)")
        print(f"  Voxels: {n_voxels:,}")
        print(f"  Timepoints: {n_timepoints}")
        print(f"  Runs: {n_runs}")
        print(f"  Splits: {n_splits}")
        print(f"  Metric: {metric}")
        print(f"  Data chunk size: {data_chunk_size:,}")
        print(f"  Batch size: {batch_size:,}")
        print()

    # =========================================================================
    # GLMdenoise-style accumulation: Store predictions/actuals per run
    # =========================================================================
    # MEMORY STRATEGY:
    # - Keep data on CPU, stream voxel batches to GPU
    # - Accumulate predictions/actuals on CPU
    # - Design matrices are small - keep on GPU
    # - Only small batch of voxels on GPU at any time

    # Initialize storage on CPU
    pred_accumulator = torch.zeros(n_voxels, n_timepoints, dtype=torch.float32, device="cpu")
    actual_accumulator = torch.zeros(n_voxels, n_timepoints, dtype=torch.float32, device="cpu")
    count_per_timepoint = torch.zeros(n_timepoints, dtype=torch.float32, device="cpu")

    # Ensure data is on CPU for streaming
    if data.device.type != "cpu":
        if verbose:
            print("  Moving data to CPU for memory-efficient streaming...")
        data = data.cpu()

    # Pre-compute timepoint indices for each split (cheap, do once)
    split_info = []
    for train_runs, test_runs in cv_splits:
        # Train timepoints
        train_tps = []
        for run_idx in train_runs:
            start = run_starts[run_idx]
            end = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
            train_tps.extend(range(start, end))

        # Test timepoints
        test_tps = []
        for run_idx in test_runs:
            start = run_starts[run_idx]
            end = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
            test_tps.extend(range(start, end))

        split_info.append((train_tps, test_tps))

    # Process CV splits
    for split_idx, (train_runs, test_runs) in enumerate(cv_splits):
        if verbose:
            print(f"  Split {split_idx + 1}/{n_splits}: Train {train_runs} | Test {test_runs}")

        train_tps, test_tps = split_info[split_idx]

        # Slice DESIGN only (small - fits on GPU)
        train_design = design_matrix[train_tps, :]
        test_design = design_matrix[test_tps, :]

        # 2. Precompute projection matrix P (tiny - stays on GPU)
        train_P = _compute_projection_matrix(train_design, nuisance_indices)
        test_P = _compute_projection_matrix(test_design, nuisance_indices)

        # 3. Project design matrices (small - do once per split)
        if train_P is not None:
            train_design_clean = train_design - train_P @ train_design
        else:
            train_design_clean = train_design

        if test_P is not None:
            test_design_clean = test_design - test_P @ test_design
        else:
            test_design_clean = test_design

        # 4. Extract stimulus design (already on device!)
        train_stim_design = train_design_clean[:, stim_indices]
        test_stim_design = test_design_clean[:, stim_indices]

        # 4b. Detect zero columns in stimulus indices (missing events across runs)
        train_stim_norms = train_stim_design.abs().sum(dim=0)
        test_stim_norms = test_stim_design.abs().sum(dim=0)

        train_zero_mask = train_stim_norms < 1e-10
        test_zero_mask = test_stim_norms < 1e-10

        train_present_mask = ~train_zero_mask
        test_present_mask = ~test_zero_mask
        unpredictable_mask = train_present_mask & test_zero_mask
        test_only_mask = train_zero_mask & test_present_mask
        predictable_mask = train_present_mask & test_present_mask

        # Handle missing events
        if train_zero_mask.any() or test_zero_mask.any():
            unpredictable_cols = [i for i, unpred in enumerate(unpredictable_mask) if unpred]
            test_only_cols = [i for i, test_only in enumerate(test_only_mask) if test_only]

            if split_idx == 0 and verbose:
                print(f"\n{'=' * 80}")
                print("INFO: Handling missing events across train/test splits")
                print(f"{'=' * 80}")
                print(f"Train-only events: {len(unpredictable_cols)} - {unpredictable_cols}")
                print(f"Test-only events: {len(test_only_cols)} - {test_only_cols}")
                print(f"Predictable events: {predictable_mask.sum().item()}")
                print(f"Strategy: '{zero_event_strategy}'")
                print(f"{'=' * 80}\n")

            if not predictable_mask.any():
                raise ValueError(
                    f"No overlapping events between train {train_runs} and test {test_runs}!"
                )

            if zero_event_strategy == "zero":
                train_stim_design_fit = train_stim_design[:, train_present_mask]
            elif zero_event_strategy == "nuisance":
                train_stim_design_fit = train_stim_design[:, predictable_mask]
            else:
                raise ValueError(f"Unknown zero_event_strategy: '{zero_event_strategy}'")
        else:
            train_stim_design_fit = train_stim_design
            train_present_mask = torch.ones(
                train_stim_design.shape[1], dtype=torch.bool, device=device
            )
            test_present_mask = torch.ones(
                test_stim_design.shape[1], dtype=torch.bool, device=device
            )
            predictable_mask = train_present_mask
            unpredictable_mask = torch.zeros(
                train_stim_design.shape[1], dtype=torch.bool, device=device
            )
            test_only_mask = torch.zeros(
                train_stim_design.shape[1], dtype=torch.bool, device=device
            )

        # 5. Fit OLS and predict in VOXEL BATCHES (stream from CPU)
        for batch_start in range(0, n_voxels, batch_size):
            batch_end = min(batch_start + batch_size, n_voxels)
            batch_slice = slice(batch_start, batch_end)

            # Stream this batch's data to GPU (only train and test timepoints!)
            train_data_batch = data[batch_slice][:, train_tps].to(device)
            test_data_batch = data[batch_slice][:, test_tps].to(device)

            # Project out nuisance from data
            if train_P is not None:
                train_data_batch = train_data_batch - (train_P @ train_data_batch.T).T
            if test_P is not None:
                test_data_batch = test_data_batch - (test_P @ test_data_batch.T).T

            # Additional projection for 'nuisance' strategy
            if zero_event_strategy == "nuisance":
                events_to_project = unpredictable_mask | test_only_mask
                if events_to_project.any():
                    test_to_project = test_stim_design[:, events_to_project]
                    XuXu = test_to_project.T @ test_to_project
                    XuXu_inv = torch.linalg.inv(
                        XuXu + 1e-6 * torch.eye(XuXu.shape[0], device=device)
                    )
                    P_unpred = test_to_project @ XuXu_inv @ test_to_project.T
                    test_data_batch = test_data_batch - (P_unpred @ test_data_batch.T).T

            # OLS fit
            XtX = train_stim_design_fit.T @ train_stim_design_fit
            XtX_inv = torch.linalg.inv(XtX + 1e-6 * torch.eye(XtX.shape[0], device=device))
            betas_fit = XtX_inv @ train_stim_design_fit.T @ train_data_batch.T

            # Predict test data
            if zero_event_strategy == "zero":
                n_stim = len(stim_indices)
                betas_full = torch.zeros(n_stim, betas_fit.shape[1], device=device)
                betas_full[train_present_mask, :] = betas_fit
                test_stim_present = test_stim_design[:, test_present_mask]
                betas_test_present = betas_full[test_present_mask, :]
                predictions_batch = (test_stim_present @ betas_test_present).T
            elif zero_event_strategy == "nuisance":
                test_stim_predictable = test_stim_design[:, predictable_mask]
                predictions_batch = (test_stim_predictable @ betas_fit).T
            else:
                raise ValueError(f"Invalid strategy: {zero_event_strategy}")

            # ============================================================
            # ACCUMULATE predictions and actuals for test timepoints (on CPU)
            # ============================================================
            pred_accumulator[batch_slice, test_tps] += predictions_batch.cpu()
            actual_accumulator[batch_slice, test_tps] += test_data_batch.cpu()

            # Free GPU memory for this batch
            del train_data_batch, test_data_batch, predictions_batch, betas_fit
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Update count for these test timepoints (once per split)
        count_per_timepoint[test_tps] += 1.0

    # =========================================================================
    # Average accumulated predictions where timepoints appeared in multiple folds
    # =========================================================================
    # For LORO, each timepoint appears exactly once (count=1)
    # For split-halves with permutations, timepoints may appear multiple times
    count_per_timepoint = count_per_timepoint.clamp(min=1)  # Avoid division by zero
    pred_accumulator = pred_accumulator / count_per_timepoint.unsqueeze(0)
    actual_accumulator = actual_accumulator / count_per_timepoint.unsqueeze(0)

    # =========================================================================
    # Compute single R² from concatenated predictions vs actuals
    # =========================================================================
    if verbose:
        print()
        print("Computing R² from concatenated predictions...")

    # Compute R² in voxel batches (stream to GPU)
    r2_final = torch.zeros(n_voxels, dtype=torch.float32, device="cpu")

    for r2_batch_start in range(0, n_voxels, batch_size):
        r2_batch_end = min(r2_batch_start + batch_size, n_voxels)
        r2_batch_slice = slice(r2_batch_start, r2_batch_end)

        pred_batch = pred_accumulator[r2_batch_slice].to(device)
        actual_batch = actual_accumulator[r2_batch_slice].to(device)

        r2_batch = compute_r2_metric(actual_batch, pred_batch, metric=metric)
        r2_final[r2_batch_slice] = r2_batch.cpu()

        del pred_batch, actual_batch
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if verbose:
        print(f"  Mean R²: {r2_final.mean():.4f}, Std: {r2_final.std():.4f}")
        print()
        print("✓ Cross-validation complete (GLMdenoise-style)")
        print()

    # Return single R² (not per-fold statistics)
    results = {
        "r2": r2_final,
        "n_splits": n_splits,
    }

    return results
