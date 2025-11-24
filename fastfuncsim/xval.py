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
            raise ValueError(
                f"strategy={strategy} must be > 0 and < n_runs={n_runs}"
            )

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
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    if metric == "cod":
        # Coefficient of determination: R² = 1 - SS_res/SS_tot
        ss_res = ((y_true - y_pred) ** 2).sum(dim=1)
        y_mean = y_true.mean(dim=1, keepdim=True)
        ss_tot = ((y_true - y_mean) ** 2).sum(dim=1)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-10))

    elif metric == "corr":
        # Pearson correlation coefficient
        y_true_centered = y_true - y_true.mean(dim=1, keepdim=True)
        y_pred_centered = y_pred - y_pred.mean(dim=1, keepdim=True)

        numerator = (y_true_centered * y_pred_centered).sum(dim=1)
        denom_true = torch.sqrt((y_true_centered ** 2).sum(dim=1))
        denom_pred = torch.sqrt((y_pred_centered ** 2).sum(dim=1))
        denominator = denom_true * denom_pred

        r2 = numerator / (denominator + 1e-10)

    elif metric == "corr2":
        # Pearson correlation squared
        r = compute_r2_metric(y_true, y_pred, metric="corr")
        r2 = r ** 2

    else:
        raise ValueError(
            f"Unknown metric '{metric}'. Choose from: 'cod', 'corr', 'corr2'"
        )

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
    device: Optional[torch.device] = None,
    batch_size: Optional[int] = None,
    data_chunk_size: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Compute cross-validated R² using run-based train/test splits.

    Strategy:
    ---------
    For each CV split (train_runs, test_runs):
        1. Slice data and design by runs
        2. Project out nuisance from train data & design
        3. Project out nuisance from test data & design
        4. Extract stimulus design (after projection)
        5. Fit OLS on cleaned train data with cleaned train stim design
        6. Predict cleaned test data using train betas and cleaned test stim design
        7. Compute R² between cleaned test data and predictions

    Then aggregate across splits: median, std, min, max

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
        'r2_median': (n_voxels,) median R² across splits
        'r2_std': (n_voxels,) standard deviation across splits
        'r2_min': (n_voxels,) minimum R² across splits
        'r2_max': (n_voxels,) maximum R² across splits
        'r2_splits': (n_splits, n_voxels) R² for each split
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
    >>> print(f"Median xval R²: {results['r2_median'].mean():.3f}")
    """
    from .utils import get_device, to_tensor

    if device is None:
        device = get_device()

    # Design matrix is small - keep on device
    design_matrix = to_tensor(design_matrix, device=device, dtype=torch.float32)

    n_voxels = data.shape[0]
    n_splits = len(cv_splits)

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

    # Storage for R² across splits (float32 to save memory)
    r2_all_splits = torch.zeros(n_splits, n_voxels, dtype=torch.float32, device='cpu')

    if verbose:
        print(f"Cross-validation R² computation")
        print(f"  Voxels: {n_voxels:,}")
        print(f"  Splits: {n_splits}")
        print(f"  Metric: {metric}")
        print(f"  Data chunk size: {data_chunk_size:,}")
        print(f"  Batch size: {batch_size:,}")
        print()

    # Process data in chunks (for very large datasets that don't fit on GPU)
    n_chunks = (n_voxels + data_chunk_size - 1) // data_chunk_size

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * data_chunk_size
        chunk_end = min(chunk_start + data_chunk_size, n_voxels)
        chunk_slice = slice(chunk_start, chunk_end)
        n_voxels_chunk = chunk_end - chunk_start

        if verbose and n_chunks > 1:
            print(f"\n📦 Data chunk {chunk_idx + 1}/{n_chunks}: voxels {chunk_start:,}-{chunk_end:,}")

        # Load this chunk to GPU
        data_chunk = data[chunk_slice]
        if data_chunk.device != device:
            data_chunk = data_chunk.to(device)

        # Run all CV splits on this chunk
        for split_idx, (train_runs, test_runs) in enumerate(cv_splits):
            if verbose:
                prefix = "  " if n_chunks > 1 else ""
                print(f"{prefix}Split {split_idx+1}/{n_splits}: Train {train_runs} | Test {test_runs}")

            # 1. Slice data and design by runs (on GPU - free indexing!)
            train_data, train_design, _ = slice_by_runs(
                data_chunk, design_matrix, run_starts, train_runs
            )
            test_data, test_design, _ = slice_by_runs(
                data_chunk, design_matrix, run_starts, test_runs
            )

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

            # 4b. Detect zero columns in stimulus indices (run-specific regressors)
            # These are stimulus columns that are zero in this particular split
            train_stim_norms = train_stim_design.abs().sum(dim=0)
            test_stim_norms = test_stim_design.abs().sum(dim=0)

            train_zero_mask = train_stim_norms < 1e-10
            test_zero_mask = test_stim_norms < 1e-10

            if train_zero_mask.any() or test_zero_mask.any():
                # We have zero columns in stimulus - this is problematic!
                zero_cols_train = [i for i, is_zero in enumerate(train_zero_mask) if is_zero]
                zero_cols_test = [i for i, is_zero in enumerate(test_zero_mask) if is_zero]

                # Only warn once (on first split, first chunk)
                if split_idx == 0 and chunk_idx == 0:
                    import warnings
                    msg = (
                        f"\n{'='*80}\n"
                        f"WARNING: Zero columns detected in STIMULUS indices!\n"
                        f"{'='*80}\n"
                        f"Train split has {len(zero_cols_train)} zero stimulus columns: {zero_cols_train}\n"
                        f"Test split has {len(zero_cols_test)} zero stimulus columns: {zero_cols_test}\n"
                        f"\n"
                        f"This means you have RUN-SPECIFIC stimulus regressors (e.g., different\n"
                        f"stimuli in different runs) that are zero when not in that run.\n"
                        f"\n"
                        f"These columns will be REMOVED from the design before fitting to avoid\n"
                        f"singular matrices. However, this may not be the intended behavior.\n"
                        f"\n"
                        f"Recommendations:\n"
                        f"  1. If these are truly nuisance (e.g., run-specific polynomials),\n"
                        f"     they should be in nuisance_indices, not stim_indices\n"
                        f"  2. If these are legitimate stimuli that vary by run, you may need\n"
                        f"     special handling (future feature)\n"
                        f"{'='*80}\n"
                    )
                    warnings.warn(msg, UserWarning, stacklevel=2)

                # Remove zero columns from stimulus design to avoid singular matrices
                # Keep track of which columns are valid
                train_nonzero_mask = ~train_zero_mask
                test_nonzero_mask = ~test_zero_mask

                if train_nonzero_mask.any():
                    train_stim_design = train_stim_design[:, train_nonzero_mask]
                else:
                    raise ValueError(
                        f"All stimulus columns are zero in train split {train_runs}! "
                        f"Cannot fit model."
                    )

                if test_nonzero_mask.any():
                    test_stim_design = test_stim_design[:, test_nonzero_mask]
                else:
                    raise ValueError(
                        f"All stimulus columns are zero in test split {test_runs}! "
                        f"Cannot compute predictions."
                    )

            # 5. Fit OLS in batches (to avoid OOM when projecting ALL voxels at once)
            r2_split_chunk = torch.zeros(n_voxels_chunk, dtype=torch.float32, device='cpu')

            for batch_start in range(0, n_voxels_chunk, batch_size):
                batch_end = min(batch_start + batch_size, n_voxels_chunk)
                batch_slice = slice(batch_start, batch_end)

                # Slice batches (free on GPU - just indexing!)
                train_data_batch = train_data[batch_slice]  # (batch_size, n_train_timepoints)
                test_data_batch = test_data[batch_slice]    # (batch_size, n_test_timepoints)

                # Project out nuisance from data batches (fast on GPU!)
                if train_P is not None:
                    train_data_batch = train_data_batch - (train_P @ train_data_batch.T).T

                if test_P is not None:
                    test_data_batch = test_data_batch - (test_P @ test_data_batch.T).T

                # OLS: beta = (X.T @ X)^-1 @ X.T @ Y (fast on GPU!)
                # X: (n_train_timepoints, n_stim), Y: (n_train_timepoints, batch_size)
                XtX = train_stim_design.T @ train_stim_design
                XtX_inv = torch.linalg.inv(XtX + 1e-6 * torch.eye(XtX.shape[0], device=device))
                betas_batch = XtX_inv @ train_stim_design.T @ train_data_batch.T  # (n_stim, batch_size)

                # 6. Predict test data (fast on GPU!)
                predictions_batch = test_stim_design @ betas_batch
                predictions_batch = predictions_batch.T  # (batch_size, n_test_timepoints)

                # 7. Compute R² (fast on GPU!)
                r2_batch = compute_r2_metric(test_data_batch, predictions_batch, metric=metric)
                r2_split_chunk[batch_slice] = r2_batch.cpu()

            # Store results for this chunk
            r2_all_splits[split_idx, chunk_slice] = r2_split_chunk

            if verbose:
                prefix = "    " if n_chunks > 1 else "  "
                print(f"{prefix}Mean R²: {r2_split_chunk.mean():.4f}, Std: {r2_split_chunk.std():.4f}")

    if verbose:
        print()
        print("✓ Cross-validation complete")
        print()

    # 8. Aggregate across splits
    results = {
        "r2_median": torch.median(r2_all_splits, dim=0).values,
        "r2_std": torch.std(r2_all_splits, dim=0),
        "r2_min": torch.min(r2_all_splits, dim=0).values,
        "r2_max": torch.max(r2_all_splits, dim=0).values,
        "r2_splits": r2_all_splits,
        "n_splits": n_splits,
    }

    return results
