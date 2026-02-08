"""
Cross-validation utilities for GLM analysis.

This module provides functions for computing cross-validated R² metrics
using run-based train/test splits. The main use case is testing denoising
methods and model selection (e.g., HRF choice).
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .memory import estimate_chunk_size


def compute_qr_projectors(
    nuisance_per_run: List[torch.Tensor],
    run_starts: List[int],
    device: Optional[torch.device] = None,
) -> List[Optional[torch.Tensor]]:
    """
    Compute QR-based projection factors for per-run nuisance regressors.

    This is the single canonical way to get Q factors for nuisance projection.
    Used by project_out_nuisance_per_run() and directly by hrf_selection.py
    when separate data/design projection is needed.

    Parameters
    ----------
    nuisance_per_run : list of torch.Tensor
        Per-run nuisance matrices, each (run_length, n_nuisance_cols)
    run_starts : list of int
        Starting timepoint for each run (used to determine number of runs)
    device : torch.device, optional
        Device to compute QR on. If None, uses the device of the nuisance tensors.

    Returns
    -------
    q_factors : list of (torch.Tensor or None)
        Per-run Q factors from QR decomposition. None if a run has no
        non-zero nuisance columns. Each Q is (run_length, n_nuisance)
        with orthonormal columns, on `device`.

    Notes
    -----
    Apply projection as: y_proj = y - Q @ (Q.T @ y) per run.
    Zero columns are automatically removed before QR to avoid singularity.
    """
    n_runs = len(run_starts)

    if device is None:
        device = nuisance_per_run[0].device if nuisance_per_run else torch.device("cpu")

    q_factors = []
    for run_idx in range(n_runs):
        run_nuisance = nuisance_per_run[run_idx].to(device)

        # Remove zero columns (run-specific regressors not present in CV subsets)
        col_norms = run_nuisance.abs().sum(dim=0)
        nonzero_mask = col_norms > 1e-10
        run_nuisance_clean = run_nuisance[:, nonzero_mask]

        if run_nuisance_clean.shape[1] > 0:
            Q, _ = torch.linalg.qr(run_nuisance_clean)
            q_factors.append(Q)
        else:
            q_factors.append(None)

    return q_factors


def project_out_nuisance_per_run(
    data: torch.Tensor,
    design: torch.Tensor,
    nuisance_per_run: List[torch.Tensor],
    run_starts: List[int],
    device: Optional[torch.device] = None,
    chunk_size: Optional[int] = None,
    verbose: bool = False,
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
    chunk_size : int, optional
        Number of voxels to process at once. If None, auto-estimated based on
        available memory and data size.
    verbose : bool, default=False
        Print memory strategy information

    Returns
    -------
    projected_data : torch.Tensor
        (n_voxels, n_timepoints) Data with nuisance projected out per run
    projected_design : torch.Tensor
        (n_timepoints, n_task_cols) Design with nuisance projected out per run
    """
    if device is None:
        device = data.device

    n_voxels, n_timepoints = data.shape
    n_runs = len(run_starts)
    n_design_cols = design.shape[1]

    # Auto-estimate chunk size if not provided
    # Use memory-aware estimation that considers downstream GLM operations
    effective_chunk_size: int
    if chunk_size is None:
        effective_chunk_size = estimate_chunk_size(
            n_voxels=n_voxels,
            n_timepoints=n_timepoints,
            n_regressors=n_design_cols,
            device=device,
            operation="xval",
        )
    else:
        effective_chunk_size = chunk_size

    # Determine if we need chunking
    # Threshold: if data would exceed ~1GB OR we have lots of voxels, use chunking
    data_size_bytes = n_voxels * n_timepoints * 4
    needs_chunking = data_size_bytes > 1e9 or n_voxels > effective_chunk_size

    if verbose:
        data_size_gb = data_size_bytes / 1e9
        print(
            f"  Data size: {data_size_gb:.2f} GB ({n_voxels:,} voxels × {n_timepoints} timepoints)"
        )
        print(
            f"  Chunking: {'Yes' if needs_chunking else 'No'} (chunk_size={effective_chunk_size:,})"
        )

    # CRITICAL: Keep data on its original device (may be CPU for memory efficiency)
    # Projection matrices are computed on compute device, then applied
    data_device = data.device

    # Pre-compute QR factorizations using shared helper
    q_factors = compute_qr_projectors(nuisance_per_run, run_starts, device=device)

    # Compute run lengths using the same pattern as slice_by_runs
    # run_starts contains starting timepoints for each run
    run_lengths = np.diff(run_starts + [n_timepoints])

    # Project design matrix (small, no chunking needed)
    projected_design_runs = []
    for run_idx in range(n_runs):
        start_tp = run_starts[run_idx]
        run_length = run_lengths[run_idx]
        run_design = design[start_tp:start_tp + run_length, :].to(device)  # (run_length, n_task)

        if q_factors[run_idx] is not None:
            # Apply projection using Q: design_proj = design - Q @ (Q.T @ design)
            Q = q_factors[run_idx]
            run_design_proj = run_design - Q @ (Q.T @ run_design)
        else:
            run_design_proj = run_design

        projected_design_runs.append(run_design_proj.to(design.device))

    projected_design = torch.cat(projected_design_runs, dim=0)

    # Project data - use chunking for memory efficiency
    if not needs_chunking:
        # Small enough to process all at once
        projected_data_runs = []
        for run_idx in range(n_runs):
            start_tp = run_starts[run_idx]
            run_length = run_lengths[run_idx]
            run_data = data[:, start_tp:start_tp + run_length]  # (n_voxels, run_length)

            if q_factors[run_idx] is not None:
                # Move to compute device, project using Q, move back
                # data_proj = data - Q @ (Q.T @ data.T).T = data - (Q @ Q.T) @ data.T
                # Simplified: data_proj.T = data.T - Q @ (Q.T @ data.T)
                run_data_dev = run_data.to(device)
                Q = q_factors[run_idx]
                # Project: (Q @ Q.T) @ run_data_dev.T, then transpose back
                QQt_data = (Q @ (Q.T @ run_data_dev.T)).T
                run_data_proj = run_data_dev - QQt_data
                projected_data_runs.append(run_data_proj.to(data_device))
            else:
                projected_data_runs.append(run_data)

        projected_data = torch.cat(projected_data_runs, dim=1)
    else:
        # Large data: process voxels in chunks
        # Allocate output on CPU to avoid GPU memory pressure
        projected_data = torch.zeros(n_voxels, n_timepoints, dtype=data.dtype, device="cpu")

        n_chunks = (n_voxels + effective_chunk_size - 1) // effective_chunk_size
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * effective_chunk_size
            chunk_end = min(chunk_start + effective_chunk_size, n_voxels)

            # Process each run for this voxel chunk
            for run_idx in range(n_runs):
                start_tp = run_starts[run_idx]
                run_length = run_lengths[run_idx]

                # Get chunk data for this run
                run_chunk_data = data[chunk_start:chunk_end, start_tp:start_tp + run_length]

                if q_factors[run_idx] is not None:
                    # Move chunk to compute device, project using Q, store result
                    run_chunk_dev = run_chunk_data.to(device)
                    Q = q_factors[run_idx]
                    # Project: QQt_data = (Q @ (Q.T @ run_chunk_dev.T)).T
                    QQt_data = (Q @ (Q.T @ run_chunk_dev.T)).T
                    run_chunk_proj = run_chunk_dev - QQt_data
                    projected_data[chunk_start:chunk_end, start_tp:start_tp + run_length] = run_chunk_proj.cpu()

                    # Free GPU memory
                    del run_chunk_dev, run_chunk_proj
                else:
                    projected_data[chunk_start:chunk_end, start_tp:start_tp + run_length] = run_chunk_data.cpu()

            # Clear GPU cache occasionally (not too frequently - empty_cache is expensive ~10-50ms)
            # Only at the midpoint of processing to avoid memory buildup
            if device.type == "cuda" and n_chunks > 10:
                mid_point = n_chunks // 2
                if chunk_idx == mid_point:
                    torch.cuda.empty_cache()

        # CRITICAL: When chunking, keep result on CPU even if input was on GPU
        # This avoids OOM from moving large result back to GPU.
        # Caller will stream chunks to GPU as needed for computation.
        if needs_chunking:
            # Result already on CPU from line 174 - keep it there
            pass
        else:
            # Small data: move back to original device if needed
            projected_data = projected_data.to(data_device)

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

    Uses QR decomposition for numerical stability (superior to (X'X)^-1 approach).

    Parameters
    ----------
    data : torch.Tensor
        Data to clean (n_voxels, n_timepoints)
    design_matrix : torch.Tensor
        Full design matrix (n_timepoints, n_regressors)
    nuisance_indices : list of int
        Which columns are nuisance regressors
    ridge : float, default=1e-6
        Ridge regularization for numerical stability (only used if QR fails)

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
    3. Compute QR decomposition: Q, R = qr(X_nuis)
    4. Project out: data_clean = data - Q @ (Q.T @ data)
    5. Project out: design_clean = design - Q @ (Q.T @ design)

    Why remove zero columns?
    ------------------------
    When we split by runs, some nuisance regressors (like run 3's polynomials)
    will be all-zero in splits that don't include run 3. We must remove these
    before computing the projection, or the matrix will be singular.

    Why QR decomposition?
    ---------------------
    QR decomposition is numerically more stable than matrix inversion.
    Q is (n_timepoints, n_nuisance) where n_nuisance << n_timepoints, making it
    more memory-efficient than a full (n_timepoints, n_timepoints) projection matrix.

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

    # Use QR decomposition for numerical stability
    # Q is (n_timepoints, n_nuisance) with orthonormal columns
    # Projection: y_proj = y - Q @ (Q.T @ y)
    try:
        Q, _ = torch.linalg.qr(X_nuis)

        # Project out from data: data_clean = data - Q @ (Q.T @ data.T).T
        # Equivalent to: data_clean = data - (Q @ Q.T) @ data.T, then transpose
        data_cleaned = data - (Q @ (Q.T @ data.T)).T

        # Project out from design: design_clean = design - Q @ (Q.T @ design)
        design_cleaned = design_matrix - Q @ (Q.T @ design_matrix)

    except RuntimeError:
        # Fallback to (X'X)^-1 approach if QR fails (should be rare)
        XtX = X_nuis.T @ X_nuis
        XtX_reg = XtX + ridge * torch.eye(XtX.shape[0], device=XtX.device, dtype=XtX.dtype)
        XtX_inv = torch.linalg.inv(XtX_reg)
        P = X_nuis @ XtX_inv @ X_nuis.T

        data_cleaned = data - (P @ data.T).T
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


def compute_r2_from_sufficient_stats(
    ss_res: torch.Tensor,
    sum_actual: torch.Tensor,
    sum_sq_actual: torch.Tensor,
    n_timepoints: int,
) -> torch.Tensor:
    """
    Compute R² from sufficient statistics (streaming accumulators).

    This is for cases where full data arrays are not available, only
    pre-accumulated statistics from streaming/online computation.

    Parameters
    ----------
    ss_res : torch.Tensor
        Residual sum of squares (n_voxels,)
    sum_actual : torch.Tensor
        Sum of actual data values (n_voxels,)
    sum_sq_actual : torch.Tensor
        Sum of squared actual data values (n_voxels,)
    n_timepoints : int
        Number of timepoints (to compute variance from sums)

    Returns
    -------
    r2 : torch.Tensor
        R² values (n_voxels,)
        R² = 1 - ss_res / ss_tot
        where ss_tot = sum_sq_actual - sum_actual² / n_timepoints

    Notes
    -----
    This computes the Coefficient of Determination (CoD) metric only.
    For correlation-based metrics, full data arrays are needed.

    Derivation:
    - mean = sum / n
    - ss_tot = sum((y - mean)²) = sum(y²) - sum(y)² / n

    Examples
    --------
    >>> ss_res = torch.tensor([10.0, 20.0])
    >>> sum_act = torch.tensor([100.0, 200.0])
    >>> sum_sq_act = torch.tensor([1200.0, 2500.0])
    >>> n = 100
    >>> r2 = compute_r2_from_sufficient_stats(ss_res, sum_act, sum_sq_act, n)
    """
    # Compute total sum of squares from streaming statistics
    # ss_tot = sum((y - mean)²) = sum(y²) - sum(y)² / n
    mean_actual = sum_actual / n_timepoints
    ss_tot = sum_sq_actual - n_timepoints * (mean_actual**2)

    # Coefficient of determination
    r2 = 1.0 - ss_res / (ss_tot + 1e-10)

    # Clamp max to 1.0 (values > 1 indicate numerical issues)
    # Note: R² CAN be negative (model worse than mean) - that's valid information
    r2 = torch.clamp(r2, max=1.0)

    return r2.float()


def _compute_projection_matrix(
    design_matrix: torch.Tensor,
    nuisance_indices: List[int],
    ridge: float = 1e-6,
) -> Optional[torch.Tensor]:
    """
    Compute Q factor from QR decomposition of nuisance regressors.

    Returns Q such that: cleaned_data = data - Q @ (Q.T @ data)

    This is a helper to avoid recomputing Q for every voxel batch.
    Uses QR decomposition for numerical stability (superior to (X'X)^-1).

    Note: Function name kept for backwards compatibility, but now returns Q
    instead of full projection matrix P. Q is (n, p) where p << n, making it
    more memory-efficient than the old (n, n) projection matrix.
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

    # Use QR decomposition for numerical stability
    # Q is (n_timepoints, n_nuisance) with orthonormal columns
    # Projection: y_proj = y - Q @ (Q.T @ y)
    try:
        Q, _ = torch.linalg.qr(X_nuis)
        return Q
    except RuntimeError:
        # Fallback to (X'X)^-1 approach if QR fails (should be rare)
        XtX = X_nuis.T @ X_nuis
        XtX_reg = XtX + ridge * torch.eye(XtX.shape[0], device=XtX.device, dtype=XtX.dtype)
        XtX_inv = torch.linalg.inv(XtX_reg)
        P = X_nuis @ XtX_inv @ X_nuis.T
        # Return P for fallback (legacy behavior)
        # Note: P is (n, n) while Q is (n, p), but callers handle both via same API
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
    r2_method: str = "auto",
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
    r2_method : str, default='auto'
        Method for computing R²:
        - 'auto': Use 'fast' for LORO (each timepoint tested once), 'slow' otherwise
        - 'fast': Streaming stats - accumulates SS_res, sum, sum_sq instead of full
          timeseries. Uses ~3MB instead of ~8GB for accumulators. Requires LORO.
        - 'slow': Full accumulator - stores complete predicted/actual timeseries.
          Required when timepoints appear in multiple test sets (split-half CV).
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
        # Use memory-aware estimation for projection operations
        n_regressors = len(stim_indices) + len(nuisance_indices)
        batch_size = estimate_chunk_size(
            n_voxels=n_voxels,
            n_timepoints=n_timepoints,
            n_regressors=n_regressors,
            device=device,
            operation="xval",
        )

    # =========================================================================
    # Determine R² computation method: 'fast' (streaming) or 'slow' (full accumulator)
    # =========================================================================
    # Fast method: Only works for LORO where each timepoint is tested exactly once
    # - Accumulates SS_res, sum, sum_sq instead of full timeseries
    # - Memory: ~3MB instead of ~8GB for 271k voxels × 3609 timepoints
    # Slow method: Works for any CV strategy (overlapping test sets)
    # - Stores full predicted and actual timeseries
    # - Required when we need to average predictions before computing R²

    # Compute run lengths using the same pattern as slice_by_runs
    run_lengths = np.diff(run_starts + [n_timepoints])

    # Check if this is LORO (each timepoint tested exactly once)
    # by verifying test sets are disjoint and cover all timepoints
    all_test_tps = set()
    is_loro = True
    for _, test_runs in cv_splits:
        for run_idx in test_runs:
            start = run_starts[run_idx]
            run_length = run_lengths[run_idx]
            test_tps_run = set(range(start, start + run_length))
            if all_test_tps & test_tps_run:  # Overlap detected
                is_loro = False
                break
            all_test_tps |= test_tps_run
        if not is_loro:
            break

    # Determine effective r2_method
    if r2_method == "auto":
        use_fast_r2 = is_loro
    elif r2_method == "fast":
        if not is_loro:
            raise ValueError("r2_method='fast' requires LORO (each timepoint tested exactly once)")
        use_fast_r2 = True
    elif r2_method == "slow":
        use_fast_r2 = False
    else:
        raise ValueError(f"r2_method must be 'auto', 'fast', or 'slow', got '{r2_method}'")

    if verbose:
        print("Cross-validation R² computation (GLMdenoise-style)")
        print(f"  Voxels: {n_voxels:,}")
        print(f"  Timepoints: {n_timepoints}")
        print(f"  Runs: {n_runs}")
        print(f"  Splits: {n_splits}")
        print(f"  Metric: {metric}")
        print(
            f"  R² method: {'fast (streaming stats)' if use_fast_r2 else 'slow (full accumulator)'}"
        )
        if use_fast_r2:
            fast_mem_mb = n_voxels * 3 * 8 / 1e6  # 3 stats × float64
            slow_mem_gb = n_voxels * n_timepoints * 2 * 4 / 1e9  # 2 accumulators × float32
            print(
                f"  Memory savings: {fast_mem_mb:.1f}MB vs {slow_mem_gb:.2f}GB ({slow_mem_gb * 1000 / fast_mem_mb:.0f}x)"
            )
        print()

    # =========================================================================
    # Setup based on R² method
    # =========================================================================
    data_size_bytes = n_voxels * n_timepoints * 4  # float32
    data_on_gpu = data.device.type == device.type and device.type != "cpu"

    if use_fast_r2:
        # FAST MODE: Streaming stats - tiny accumulators that fit easily on GPU
        # Use float64 for precision (summing many values)
        # CRITICAL: For LORO, accumulators are tiny (~24 bytes/voxel = 45MB for 1.9M voxels)
        # Keep on GPU even when data is on CPU to avoid constant CPU<->GPU transfers
        use_gpu_accumulators = device.type == "cuda"
        accumulator_device = device if use_gpu_accumulators else torch.device("cpu")

        # Larger batches for CPU→GPU streaming (amortize transfer overhead)
        if data_on_gpu:
            effective_batch_size = min(batch_size * 4, n_voxels)
        elif use_gpu_accumulators:
            # Data on CPU but accumulators on GPU: use much larger batches
            # Target: 100k-200k voxels (amortize kernel launch overhead)
            effective_batch_size = min(batch_size * 20, n_voxels, 200_000)
        else:
            effective_batch_size = batch_size

        # Streaming stats accumulators (tiny: ~24 bytes per voxel)
        ss_res_accumulator = torch.zeros(n_voxels, dtype=torch.float64, device=accumulator_device)
        sum_actual = torch.zeros(n_voxels, dtype=torch.float64, device=accumulator_device)
        sum_sq_actual = torch.zeros(n_voxels, dtype=torch.float64, device=accumulator_device)
        count_timepoints = torch.zeros(n_voxels, dtype=torch.float64, device=accumulator_device)

        if verbose:
            if data_on_gpu:
                print("  Data on GPU - streaming stats on GPU (fast path)")
            elif use_gpu_accumulators:
                accum_mem_mb = n_voxels * 24 / 1e6
                print(f"  Data on CPU - streaming {effective_batch_size:,} voxel batches to GPU")
                print(f"  GPU accumulators: {accum_mem_mb:.1f}MB (tiny - avoids transfers)")
            else:
                print("  Data on CPU - CPU accumulators")
    else:
        # SLOW MODE: Full accumulators - need to check memory carefully
        # Check if we can fit accumulators on GPU
        use_gpu_accumulators = False
        if data_on_gpu and device.type == "cuda":
            free_mem = torch.cuda.get_device_properties(
                device
            ).total_memory - torch.cuda.memory_allocated(device)
            # Need 2x for accumulators + 1x for R² temps + 30% headroom
            needed_mem = data_size_bytes * 3.3
            use_gpu_accumulators = free_mem > needed_mem
            if verbose:
                free_gb = free_mem / 1e9
                needed_gb = needed_mem / 1e9
                if use_gpu_accumulators:
                    print(
                        f"  Data on GPU, {free_gb:.1f}GB free (need {needed_gb:.1f}GB) - GPU accumulators"
                    )
                else:
                    print(f"  Data on GPU but only {free_gb:.1f}GB free (need {needed_gb:.1f}GB)")
                    print("  Using CPU accumulators with GPU compute (hybrid mode)")
        # TODO - here is another section of the code not using the memory modules that we have.
        # This effective batch size, and hard coded values below mean we are thrashing a lot when we _might_ not need to .
        # TODO - basically any memory check stuff should be in the main memory module, for that specific opeartion.
        # I think xval already had one - does it include this stuff?
        if use_gpu_accumulators:
            accumulator_device = device
            effective_batch_size = min(batch_size * 4, n_voxels)
        else:
            accumulator_device = torch.device("cpu")
            effective_batch_size = batch_size
            if not data_on_gpu:
                if data.device.type != "cpu":
                    if verbose:
                        print("  Moving data to CPU for memory-efficient streaming...")
                    data = data.cpu()
                elif verbose:
                    print("  Data on CPU - streaming batches to GPU")

        # Full timeseries accumulators
        pred_accumulator = torch.zeros(
            n_voxels, n_timepoints, dtype=torch.float32, device=accumulator_device
        )
        actual_accumulator = torch.zeros(
            n_voxels, n_timepoints, dtype=torch.float32, device=accumulator_device
        )
        count_per_timepoint = torch.zeros(
            n_timepoints, dtype=torch.float32, device=accumulator_device
        )

    # Compute run lengths using the same pattern as slice_by_runs
    run_lengths = np.diff(run_starts + [n_timepoints])

    # Pre-compute timepoint indices for each split (cheap, do once)
    split_info = []
    for train_runs, test_runs in cv_splits:
        # Train timepoints
        train_tps = []
        for run_idx in train_runs:
            start = run_starts[run_idx]
            run_length = run_lengths[run_idx]
            train_tps.extend(range(start, start + run_length))

        # Test timepoints
        test_tps = []
        for run_idx in test_runs:
            start = run_starts[run_idx]
            run_length = run_lengths[run_idx]
            test_tps.extend(range(start, start + run_length))

        split_info.append((train_tps, test_tps))

    # Process CV splits
    for split_idx, (train_runs, test_runs) in enumerate(cv_splits):
        if verbose:
            print(f"  Split {split_idx + 1}/{n_splits}: Train {train_runs} | Test {test_runs}")

        train_tps, test_tps = split_info[split_idx]

        # Slice DESIGN only (small - fits on GPU)
        # Note: if design has block-diagonal nuisance columns, some will be zero
        # after slicing by run. _compute_projection_matrix handles zero columns.
        train_design = design_matrix[train_tps, :]
        test_design = design_matrix[test_tps, :]

        # Precompute projection Q factor (tiny - stays on GPU)
        # When nuisance_indices is empty, this is a no-op (returns None) —
        # safe for pre-projected data where nuisance was already removed.
        Q_train = _compute_projection_matrix(train_design, nuisance_indices)
        Q_test = _compute_projection_matrix(test_design, nuisance_indices)

        # Project design matrices (small - do once per split)
        if Q_train is not None:
            train_design_clean = train_design - Q_train @ (Q_train.T @ train_design)
        else:
            train_design_clean = train_design

        if Q_test is not None:
            test_design_clean = test_design - Q_test @ (Q_test.T @ test_design)
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
                    f"No overlapping events between train {train_runs} and test {test_runs}! "
                    f"All {len(stim_indices)} stimulus columns are zero in train or test."
                )

            if zero_event_strategy == "zero":
                train_stim_design_fit = train_stim_design[:, train_present_mask]
            elif zero_event_strategy == "nuisance":
                train_stim_design_fit = train_stim_design[:, predictable_mask]
            else:
                raise ValueError(f"Unknown zero_event_strategy: '{zero_event_strategy}'")
        else:
            if split_idx == 0 and verbose:
                print(f"    No missing events - full overlap ({len(stim_indices)} conditions)")
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

        # =========================================================================
        # OPTIMIZATION 1: Pre-compute OLS pseudoinverse ONCE per split
        # =========================================================================
        # train_stim_design_fit is identical for all voxel batches in this split
        # Computing (X'X)^-1 X' once saves N_batches matrix inversions
        # Dimensions: (n_stim_fit, n_train_tps) - tiny matrix, stays on device
        XtX = train_stim_design_fit.T @ train_stim_design_fit
        XtX_inv = torch.linalg.inv(XtX + 1e-6 * torch.eye(XtX.shape[0], device=device))
        ols_pseudoinverse = XtX_inv @ train_stim_design_fit.T  # (n_stim_fit, n_train_tps)

        # =========================================================================
        # OPTIMIZATION 2: CPU-only path when accumulating on CPU
        # =========================================================================
        # When accumulator is CPU, we have two options:
        # A) CPU→GPU→CPU: Fast GPU compute, but transfer overhead + uses GPU memory
        # B) CPU-only: Slower compute, no transfers, no GPU memory used
        #
        # We choose CPU-only when:
        # - Data is on CPU (not GPU), AND
        # - Batch size is small enough that transfer overhead dominates
        #   OR GPU memory is constrained
        #
        # For large batches (>= 20k voxels), GPU speed advantage typically wins
        # For small batches, avoiding transfers is faster
        use_cpu_only_path = (
            accumulator_device.type == "cpu"
            and not data_on_gpu
            and effective_batch_size < 20_000  # GPU wins for larger batches
        )

        if use_cpu_only_path:
            # Move small matrices to CPU for CPU-only computation
            # This saves GPU memory for large datasets that don't fit on GPU
            compute_device = torch.device("cpu")
            ols_pseudoinverse_cpu = ols_pseudoinverse.cpu()
            Q_train_cpu = Q_train.cpu() if Q_train is not None else None
            Q_test_cpu = Q_test.cpu() if Q_test is not None else None
            test_stim_design_cpu = test_stim_design.cpu()
            # Move mask tensors to CPU to match compute device
            train_present_mask = train_present_mask.cpu()
            test_present_mask = test_present_mask.cpu()
            predictable_mask = predictable_mask.cpu()
            unpredictable_mask = unpredictable_mask.cpu()
            test_only_mask = test_only_mask.cpu()
            if verbose:
                print(
                    f"    Using CPU-only path (batch size {effective_batch_size} < 20k threshold)"
                )
        else:
            compute_device = device
            if verbose and not data_on_gpu:
                print(f"    Using GPU path (batch size {effective_batch_size:,})")

        # 5. Fit OLS and predict in VOXEL BATCHES
        for batch_start in range(0, n_voxels, effective_batch_size):
            batch_end = min(batch_start + effective_batch_size, n_voxels)
            batch_slice = slice(batch_start, batch_end)

            # Get batch data - use CPU-only path when it's faster
            if use_cpu_only_path:
                train_data_batch = data[batch_slice][:, train_tps]  # Stays on CPU
                test_data_batch = data[batch_slice][:, test_tps]
                Q_train_batch = Q_train_cpu
                Q_test_batch = Q_test_cpu
                stim_design_test = test_stim_design_cpu
            elif data_on_gpu:
                train_data_batch = data[batch_slice][:, train_tps]
                test_data_batch = data[batch_slice][:, test_tps]
                Q_train_batch = Q_train
                Q_test_batch = Q_test
                stim_design_test = test_stim_design
            else:
                # Data on CPU, compute on GPU: stream batch to GPU
                train_data_batch = data[batch_slice][:, train_tps].to(device)
                test_data_batch = data[batch_slice][:, test_tps].to(device)
                Q_train_batch = Q_train
                Q_test_batch = Q_test
                stim_design_test = test_stim_design

            # Project out nuisance from data (QR-based projection)
            # When nuisance_indices is empty, Q is None → no-op (safe for pre-projected data)
            if Q_train_batch is not None:
                train_data_batch = train_data_batch - (Q_train_batch @ (Q_train_batch.T @ train_data_batch.T)).T
            if Q_test_batch is not None:
                test_data_batch = test_data_batch - (Q_test_batch @ (Q_test_batch.T @ test_data_batch.T)).T

            # Additional projection for 'nuisance' strategy
            if zero_event_strategy == "nuisance":
                events_to_project = unpredictable_mask | test_only_mask
                if events_to_project.any():
                    test_to_project = stim_design_test[:, events_to_project]
                    XuXu = test_to_project.T @ test_to_project
                    XuXu_inv = torch.linalg.inv(
                        XuXu + 1e-6 * torch.eye(XuXu.shape[0], device=compute_device)
                    )
                    P_unpred = test_to_project @ XuXu_inv @ test_to_project.T
                    test_data_batch = test_data_batch - (P_unpred @ test_data_batch.T).T

            # OLS fit using precomputed pseudoinverse (OPTIMIZATION 1)
            if use_cpu_only_path:
                betas_fit = ols_pseudoinverse_cpu @ train_data_batch.T
            else:
                betas_fit = ols_pseudoinverse @ train_data_batch.T

            # Predict test data
            if zero_event_strategy == "zero":
                n_stim = len(stim_indices)
                betas_full = torch.zeros(n_stim, betas_fit.shape[1], device=compute_device)
                betas_full[train_present_mask, :] = betas_fit
                test_stim_present = stim_design_test[:, test_present_mask]
                betas_test_present = betas_full[test_present_mask, :]
                predictions_batch = (test_stim_present @ betas_test_present).T
            elif zero_event_strategy == "nuisance":
                test_stim_predictable = stim_design_test[:, predictable_mask]
                predictions_batch = (test_stim_predictable @ betas_fit).T
            else:
                raise ValueError(f"Invalid strategy: {zero_event_strategy}")

            # ============================================================
            # ACCUMULATE: Fast mode (streaming stats) vs Slow mode (full timeseries)
            # ============================================================
            if use_fast_r2:
                # FAST MODE: Accumulate streaming stats (SS_res, sum, sum_sq)
                # Compute residuals and stats on GPU, accumulate to accumulators
                n_test_tps = len(test_tps)

                # Compute stats in float64 for precision
                # TODO - we make these float64 for a subtraction, does that matter? wouldn't it make more sense
                # If they were float64 for the matrix multiplcation up above? or am I overhtinking it, and risking vram OOM.
                # Wouldn't it be better to make it double to start with rather than a duplication here?
                test_data_f64 = test_data_batch.double()
                pred_f64 = predictions_batch.double()

                # SS_res = Σ(actual - pred)²
                residuals = test_data_f64 - pred_f64
                ss_res_batch = (residuals**2).sum(dim=1)

                # For SS_tot later: need Σ actual and Σ actual²
                sum_actual_batch = test_data_f64.sum(dim=1)
                sum_sq_actual_batch = (test_data_f64**2).sum(dim=1)

                # Accumulate (may need to move to CPU if accumulators are there)
                if accumulator_device.type == "cpu":
                    ss_res_accumulator[batch_slice] += ss_res_batch.cpu()
                    sum_actual[batch_slice] += sum_actual_batch.cpu()
                    sum_sq_actual[batch_slice] += sum_sq_actual_batch.cpu()
                    count_timepoints[batch_slice] += n_test_tps
                else:
                    ss_res_accumulator[batch_slice] += ss_res_batch
                    sum_actual[batch_slice] += sum_actual_batch
                    sum_sq_actual[batch_slice] += sum_sq_actual_batch
                    count_timepoints[batch_slice] += n_test_tps

                del (
                    test_data_f64,
                    pred_f64,
                    residuals,
                    ss_res_batch,
                    sum_actual_batch,
                    sum_sq_actual_batch,
                )
            else:
                # SLOW MODE: Accumulate full timeseries
                if accumulator_device.type != "cpu":
                    # Full GPU path: accumulate directly on GPU
                    pred_accumulator[batch_slice, test_tps] += predictions_batch
                    actual_accumulator[batch_slice, test_tps] += test_data_batch
                else:
                    # Hybrid/CPU path: move results to CPU for accumulation
                    pred_accumulator[batch_slice, test_tps] += predictions_batch.cpu()
                    actual_accumulator[batch_slice, test_tps] += test_data_batch.cpu()

            # Free intermediate tensors
            del train_data_batch, test_data_batch, predictions_batch, betas_fit

        # Clear GPU cache once per split when streaming from CPU
        if not data_on_gpu and device.type == "cuda":
            torch.cuda.empty_cache()

        # Update count for these test timepoints (slow mode only)
        if not use_fast_r2:
            count_per_timepoint[test_tps] += 1.0

    # =========================================================================
    # Compute final R²
    # =========================================================================
    if verbose:
        print()
        print(
            "Computing R² from accumulated stats..."
            if use_fast_r2
            else "Computing R² from concatenated predictions..."
        )

    if use_fast_r2:
        # FAST MODE: Compute R² from streaming stats
        # R² = 1 - SS_res / SS_tot
        # SS_tot = Σ(actual - mean)² = Σ actual² - n * mean²
        #        = sum_sq_actual - (sum_actual)² / n

        # Move to CPU for final computation (tiny tensors)
        ss_res = (
            ss_res_accumulator.cpu() if accumulator_device.type != "cpu" else ss_res_accumulator
        )
        sum_act = sum_actual.cpu() if accumulator_device.type != "cpu" else sum_actual
        sum_sq_act = sum_sq_actual.cpu() if accumulator_device.type != "cpu" else sum_sq_actual
        n_pts = count_timepoints.cpu() if accumulator_device.type != "cpu" else count_timepoints

        # Compute SS_tot using the variance formula: Var = E[X²] - E[X]²
        # SS_tot = n * Var = sum_sq - sum² / n
        mean_actual = sum_act / n_pts
        ss_tot = sum_sq_act - n_pts * (mean_actual**2)

        # R² = 1 - SS_res / SS_tot
        r2_final = (1.0 - ss_res / (ss_tot + 1e-10)).float()

        # Free accumulators
        del ss_res_accumulator, sum_actual, sum_sq_actual, count_timepoints

    else:
        # SLOW MODE: Average predictions if needed, then compute R²
        count_per_timepoint = count_per_timepoint.clamp(min=1)  # Avoid division by zero
        pred_accumulator = pred_accumulator / count_per_timepoint.unsqueeze(0)
        actual_accumulator = actual_accumulator / count_per_timepoint.unsqueeze(0)

        if accumulator_device.type != "cpu":
            # Full GPU path: accumulators on GPU, compute R² directly
            r2_final = compute_r2_metric(actual_accumulator, pred_accumulator, metric=metric)
            r2_final = r2_final.cpu()
        else:
            # Hybrid/CPU path: compute R² in voxel batches
            r2_final = torch.zeros(n_voxels, dtype=torch.float32, device="cpu")

            for r2_batch_start in range(0, n_voxels, effective_batch_size):
                r2_batch_end = min(r2_batch_start + effective_batch_size, n_voxels)
                r2_batch_slice = slice(r2_batch_start, r2_batch_end)

                pred_batch = pred_accumulator[r2_batch_slice].to(device)
                actual_batch = actual_accumulator[r2_batch_slice].to(device)

                r2_batch = compute_r2_metric(actual_batch, pred_batch, metric=metric)
                r2_final[r2_batch_slice] = r2_batch.cpu()

                del pred_batch, actual_batch

            if device.type == "cuda":
                torch.cuda.empty_cache()

    if verbose:
        q25, q50, q75 = torch.quantile(
            r2_final.float(), torch.tensor([0.25, 0.50, 0.75])
        )
        print(f"  R² summary: mean={r2_final.mean():.4f}, std={r2_final.std():.4f}")
        print(f"  Quartiles:  Q25={q25:.4f}, Q50={q50:.4f}, Q75={q75:.4f}")
        print(f"  Range:      [{r2_final.min():.4f}, {r2_final.max():.4f}]")
        print()

    # Return single R² (not per-fold statistics)
    # GLMdenoise-style concatenation produces single per-voxel R² across all folds.
    results = {
        "r2": r2_final,
        # Backward compat: "r2_median" is a misnomer — it's the per-voxel R² tensor.
        # Kept for ffs_pathfinder compatibility.
        "r2_median": r2_final,
        "r2_mean": r2_final.mean(),
        "r2_std": r2_final.std(),
        "r2_min": r2_final.min(),
        "r2_max": r2_final.max(),
        "n_splits": n_splits,
    }

    return results


def compute_xval_r2_single_trials(
    single_trial_betas: torch.Tensor,
    trial_condition_ids: torch.Tensor,
    trial_run_ids: torch.Tensor,
    cv_splits: List[Tuple[List[int], List[int]]],
    metric: str = "cod",
    device: Optional[torch.device] = None,
    chunk_size: Optional[int] = None,
    verbose: bool = True,
) -> dict[str, torch.Tensor | int]:
    """
    Cross-validated R² in single-trial beta space (GLMsingle-style fit-once).

    For each LORO fold:
      1. Partition trials by run membership (using trial_run_ids)
      2. Average train-run trial betas by condition → (n_voxels, n_conditions)
      3. For each test trial: predicted = condition_avg[trial_condition]
      4. Stack predicted vs actual test betas
    After all folds: compute single R² from concatenated predictions vs actuals.

    Parameters
    ----------
    single_trial_betas : (n_voxels, n_trials) betas from full-data fit
    trial_condition_ids : (n_trials,) which condition each trial belongs to
    trial_run_ids : (n_trials,) which run each trial came from
    cv_splits : list of (train_runs, test_runs) tuples
    metric : 'cod', 'corr', or 'corr2'
    device : compute device (auto if None)
    chunk_size : voxel chunk size for GPU memory (None = all at once)
    verbose : print progress

    Returns
    -------
    dict with:
      'r2': (n_voxels,) beta-space CV R²
      'n_splits': int
      'n_test_trials_total': int
    """
    from .utils import get_device

    if device is None:
        device = get_device()

    n_voxels, n_trials = single_trial_betas.shape
    n_conditions = int(trial_condition_ids.max().item()) + 1
    n_splits = len(cv_splits)

    # Move small tensors to device
    trial_condition_ids = trial_condition_ids.to(device)
    trial_run_ids = trial_run_ids.to(device)

    # Determine chunking
    if chunk_size is None:
        chunk_size = n_voxels  # No chunking by default (betas are small)

    # Accumulate predicted and actual test betas across folds
    all_predicted = []
    all_actual = []

    for split_idx, (train_runs, test_runs) in enumerate(cv_splits):
        if verbose:
            print(f"  Split {split_idx + 1}/{n_splits}: Train {train_runs} | Test {test_runs}")

        # Build run membership masks
        train_mask = torch.zeros(n_trials, dtype=torch.bool, device=device)
        for r in train_runs:
            train_mask |= trial_run_ids == r

        test_mask = torch.zeros(n_trials, dtype=torch.bool, device=device)
        for r in test_runs:
            test_mask |= trial_run_ids == r

        test_indices = torch.where(test_mask)[0]
        n_test_trials = len(test_indices)

        if n_test_trials == 0:
            continue

        # Average train betas by condition: (n_voxels, n_conditions)
        # Process in voxel chunks
        test_conditions = trial_condition_ids[test_indices]  # (n_test_trials,)

        for chunk_start in range(0, n_voxels, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_voxels)
            betas_chunk = single_trial_betas[chunk_start:chunk_end].to(device)

            # Condition averages from train trials
            condition_avg = torch.zeros(chunk_end - chunk_start, n_conditions, device=device)
            for c in range(n_conditions):
                cond_train_mask = train_mask & (trial_condition_ids == c)
                n_cond_train = cond_train_mask.sum().item()
                if n_cond_train > 0:
                    condition_avg[:, c] = betas_chunk[:, cond_train_mask].mean(dim=1)

            # Predicted test betas: for each test trial, use its condition average
            predicted = condition_avg[:, test_conditions]  # (chunk, n_test_trials)
            actual = betas_chunk[:, test_indices]  # (chunk, n_test_trials)

            if chunk_start == 0:
                fold_predicted = predicted.cpu()
                fold_actual = actual.cpu()
            else:
                fold_predicted = torch.cat([fold_predicted, predicted.cpu()], dim=0)  # ty: ignore[unresolved-reference]
                fold_actual = torch.cat([fold_actual, actual.cpu()], dim=0)  # ty: ignore[unresolved-reference]

        all_predicted.append(fold_predicted)
        all_actual.append(fold_actual)

    # Concatenate across folds: (n_voxels, total_test_trials)
    all_predicted = torch.cat(all_predicted, dim=1)
    all_actual = torch.cat(all_actual, dim=1)

    # Compute R²
    r2 = compute_r2_metric(all_actual, all_predicted, metric=metric)

    total_test_trials = all_predicted.shape[1]
    if verbose:
        print(
            f"  Beta-space CV R²: mean={r2.mean():.4f}, "
            f"median={r2.median():.4f} "
            f"({total_test_trials} test trials across {n_splits} folds)"
        )

    return {
        "r2": r2,
        "r2_median": r2,  # Backward compat: misleading name, actually per-voxel R² tensor
        "r2_mean": r2.mean(),  # Scalar mean for convenience
        "n_splits": n_splits,
        "n_test_trials_total": total_test_trials,
    }
