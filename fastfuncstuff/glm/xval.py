"""
Cross-validation utilities for GLM analysis.

This module provides functions for computing cross-validated R² metrics
using run-based train/test splits. The main use case is testing denoising
methods and model selection (e.g., HRF choice).
"""

from __future__ import annotations

import os
from itertools import combinations
from math import comb

import numpy as np
import torch

from fastfuncstuff._compile import safe_compile
from fastfuncstuff.memory import estimate_chunk_size, get_available_memory
from fastfuncstuff.utils import factor_device


def compute_qr_projectors(
    nuisance_per_run: list[torch.Tensor],
    run_starts: list[int],
    device: torch.device | None = None,
) -> list[torch.Tensor | None]:
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
            # MPS routes QR through an implicit CPU fallback. This matrix is small;
            # factor it explicitly on CPU and copy the reusable projector once.
            qr_device = torch.device("cpu") if device.type == "mps" else device
            Q, _ = torch.linalg.qr(run_nuisance_clean.to(qr_device))
            Q = Q.to(device)
            q_factors.append(Q)
        else:
            q_factors.append(None)

    return q_factors


def project_out_nuisance_per_run(
    data: torch.Tensor,
    design: torch.Tensor,
    nuisance_per_run: list[torch.Tensor],
    run_starts: list[int],
    device: torch.device | None = None,
    chunk_size: int | None = None,
    verbose: bool = False,
    precomputed_q_factors: list[torch.Tensor | None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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

    # Pre-compute QR factorizations using shared helper — or skip if the
    # caller already computed them for this set of runs (e.g. denoise CV
    # reuses the same per-run nuisance across many folds).
    if precomputed_q_factors is not None:
        if len(precomputed_q_factors) != len(nuisance_per_run):
            raise ValueError(
                f"precomputed_q_factors has {len(precomputed_q_factors)} entries "
                f"but nuisance_per_run has {len(nuisance_per_run)}"
            )
        q_factors = precomputed_q_factors
    else:
        q_factors = compute_qr_projectors(nuisance_per_run, run_starts, device=device)

    # Compute run lengths using the same pattern as slice_by_runs
    # run_starts contains starting timepoints for each run
    run_lengths = np.diff(run_starts + [n_timepoints])

    # Project design matrix (small, no chunking needed)
    projected_design_runs = []
    for run_idx in range(n_runs):
        start_tp = run_starts[run_idx]
        run_length = run_lengths[run_idx]
        run_design = design[start_tp : start_tp + run_length, :].to(device)  # (run_length, n_task)

        Q = q_factors[run_idx]
        if Q is not None:
            # Apply projection using Q: design_proj = design - Q @ (Q.T @ design)
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
            run_data = data[:, start_tp : start_tp + run_length]  # (n_voxels, run_length)

            Q = q_factors[run_idx]
            if Q is not None:
                # Move to compute device, project using Q, move back
                # data_proj = data - Q @ (Q.T @ data.T).T = data - (Q @ Q.T) @ data.T
                # Simplified: data_proj.T = data.T - Q @ (Q.T @ data.T)
                run_data_dev = run_data.to(device)
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
                run_chunk_data = data[chunk_start:chunk_end, start_tp : start_tp + run_length]

                Q = q_factors[run_idx]
                if Q is not None:
                    # Move chunk to compute device, project using Q, store result
                    run_chunk_dev = run_chunk_data.to(device)
                    # Project: QQt_data = (Q @ (Q.T @ run_chunk_dev.T)).T
                    QQt_data = (Q @ (Q.T @ run_chunk_dev.T)).T
                    run_chunk_proj = run_chunk_dev - QQt_data
                    projected_data[chunk_start:chunk_end, start_tp : start_tp + run_length] = (
                        run_chunk_proj.cpu()
                    )

                    # Free GPU memory
                    del run_chunk_dev, run_chunk_proj
                else:
                    projected_data[chunk_start:chunk_end, start_tp : start_tp + run_length] = (
                        run_chunk_data.cpu()
                    )

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
    strategy: float | int,
    n_perms: int = 100,
) -> list[tuple[list[int], list[int]]]:
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

        return _sample_run_splits(run_indices, n_train, n_perms)

    elif isinstance(strategy, int):
        # Leave-N-out
        n_test = strategy
        if n_test <= 0 or n_test >= n_runs:
            raise ValueError(f"strategy={strategy} must be > 0 and < n_runs={n_runs}")

        # The helper samples training sets, so use the complementary size here.
        return _sample_run_splits(run_indices, n_runs - n_test, n_perms)

    else:
        raise ValueError(f"strategy must be float or int, got {type(strategy)}")


def _sample_run_splits(
    run_indices: list[int],
    n_train: int,
    n_perms: int,
) -> list[tuple[list[int], list[int]]]:
    """Enumerate small split spaces and sample large ones without materialising them."""
    import random

    n_possible = comb(len(run_indices), n_train)
    if n_possible <= n_perms:
        train_sets = list(combinations(run_indices, n_train))
    else:
        # Rejection sampling is effectively collision-free for the large spaces
        # that motivated this branch (e.g. C(100, 50) ~= 1e29). It also remains
        # cheap near the boundary: at worst n_perms unique draws from a space of
        # n_perms + 1 possibilities.
        rng = random.Random(42)
        sampled: set[tuple[int, ...]] = set()
        n_runs = len(run_indices)
        n_test = n_runs - n_train
        coverage_possible = n_perms * n_train >= n_runs and n_perms * n_test >= n_runs
        if coverage_possible:
            # Seed the sample with the minimum number of balanced cyclic splits.
            # For a 50/50 split and n_perms=2 these are exact complements, so
            # every run is tested once rather than relying on an astronomically
            # unlikely random complementary draw.
            shuffled = list(run_indices)
            rng.shuffle(shuffled)
            n_coverage = max(
                (n_runs + n_train - 1) // n_train,
                (n_runs + n_test - 1) // n_test,
            )
            for split_idx in range(n_coverage):
                start = split_idx * n_train
                sampled.add(
                    tuple(sorted(shuffled[(start + offset) % n_runs] for offset in range(n_train)))
                )
        while len(sampled) < n_perms:
            sampled.add(tuple(sorted(rng.sample(run_indices, n_train))))
        train_sets = sorted(sampled)

    all_runs = set(run_indices)
    return [(list(train), sorted(all_runs.difference(train))) for train in train_sets]


def slice_by_runs(
    data: torch.Tensor,
    design_matrix: torch.Tensor,
    run_starts: list[int],
    run_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
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
    nuisance_indices: list[int],
    ridge: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
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


@torch.inference_mode()
def _cod_kernel(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Pure-tensor coefficient-of-determination kernel.

    Extracted as a standalone function so torch.compile can fuse the four
    elementwise+reduction passes (residual², mean, total², clamp) into a
    single kernel. Called inside CV inner loops at ~20k invocations per
    end-to-end denoise/hrfopt/ridge run; cumulative win is meaningful even
    if each call is modest.
    """
    ss_res = ((y_true - y_pred) ** 2).sum(dim=1)
    y_mean = y_true.mean(dim=1, keepdim=True)
    ss_tot = ((y_true - y_mean) ** 2).sum(dim=1)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-10))
    return torch.clamp(r2, max=1.0)


@torch.inference_mode()
def _cod_from_ss_res_kernel(y_true: torch.Tensor, ss_res: torch.Tensor) -> torch.Tensor:
    """CoD for callers that already hold the residual sum of squares.

    Same arithmetic as :func:`_cod_kernel`, minus the terms it can't avoid
    recomputing: a caller with residuals in hand would otherwise have to
    reconstruct ``y_pred`` (a full (V, T) allocation) only for this kernel to
    subtract it back off. See :func:`fastfuncstuff.glm.core.fit_glm_chunk`.
    """
    y_mean = y_true.mean(dim=1, keepdim=True)
    ss_tot = ((y_true - y_mean) ** 2).sum(dim=1)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-10))
    return torch.clamp(r2, max=1.0)


# Compile through the central policy: PCH disabled (no stale-cache crashes) plus a
# permanent eager fallback if compilation ever fails for another reason. See _compile.py.
_cod_kernel_compiled = safe_compile(_cod_kernel, dynamic=True, fullgraph=True)
_cod_from_ss_res_compiled = safe_compile(_cod_from_ss_res_kernel, dynamic=True, fullgraph=True)


def cod_from_ss_residual(y_true: torch.Tensor, ss_residual: torch.Tensor) -> torch.Tensor:
    """Coefficient of determination from precomputed residual sum of squares.

    Equivalent to ``compute_r2_metric(y_true, y_pred, metric="cod")`` when
    ``ss_residual == ((y_true - y_pred) ** 2).sum(dim=1)``.
    """
    return _cod_from_ss_res_compiled(y_true, ss_residual).float()


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
        'sse': Sum of squared errors (lower = better). Equivalent to
               GLMsingle's "badness" metric. For hyperparameter selection,
               minimize SSE rather than maximize R².

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

    if metric == "sse":
        # Sum of squared errors (lower = better).
        # Equivalent to GLMsingle's "badness" metric (calcbadness in
        # GLMestimatesingletrial.m). Used for hyperparameter selection in
        # beta-space CV: predicted condition-average betas vs held-out trial
        # betas. Selection minimizes SSE rather than maximizing R².
        r2 = ((y_true - y_pred) ** 2).sum(dim=1)
        return r2.float()

    elif metric == "cod":
        # Coefficient of determination: R² = 1 - SS_res/SS_tot. Fused via
        # torch.compile in _cod_kernel_compiled so the four element-wise
        # passes (residual², mean, total², clamp) become a single kernel.
        # R² can be negative (model worse than mean) — valid information.
        r2 = _cod_kernel_compiled(y_true, y_pred)

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
        raise ValueError(f"Unknown metric '{metric}'. Choose from: 'cod', 'corr', 'corr2', 'sse'")

    return r2.float()  # Cast to float32 to save space


def metric_higher_is_better(metric: str) -> bool:
    """Whether higher values indicate better fit for a given metric.

    Returns True for cod/corr/corr2 (maximize), False for sse (minimize).
    Used by hyperparameter selection logic (argmax vs argmin).
    """
    if metric == "sse":
        return False
    if metric in ("cod", "corr", "corr2"):
        return True
    raise ValueError(f"Unknown metric '{metric}'")


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
    nuisance_indices: list[int],
    ridge: float = 1e-6,
) -> torch.Tensor | None:
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


def _cap_batch_to_free_vram(
    effective_batch_size: int,
    *,
    cv_splits: list[tuple[list[int], list[int]]],
    run_lengths: np.ndarray,
    device: torch.device,
    use_fast_r2: bool,
    verbose: bool = False,
) -> int:
    """Shrink a voxel batch so one fold's working set fits in free VRAM.

    The batch sizes above this call are ``estimate_chunk_size`` multiplied by
    4 or 20 to amortize transfer and launch overhead, with the multipliers
    tuned for LORO — one run held out, so train/test slices are bounded by the
    dataset length. They stop being safe as soon as a fold is *big*: a split
    testing 10 runs, or an inner CV whose training set is 9 of 10 runs, scales
    the per-batch arrays by the same factor and the allocation fails. (On CUDA
    the failure surfaces as an NVML assert from the caching allocator, not an
    OOM, which is a long way from the cause.)

    Sizing uses the widest fold actually in *cv_splits* and the free VRAM at
    this moment, so it adapts to whatever else is already resident (the data
    itself, when it lives on the GPU).
    """
    if device.type != "cuda":
        return effective_batch_size

    max_train_tps = 0
    max_test_tps = 0
    for train_runs, test_runs in cv_splits:
        max_train_tps = max(max_train_tps, sum(int(run_lengths[r]) for r in train_runs))
        max_test_tps = max(max_test_tps, sum(int(run_lengths[r]) for r in test_runs))

    # Per voxel, live at the same moment inside the batch loop:
    #   train side: the batch, its transpose and the projection product (3 x f32)
    #   test side:  the same 3, plus the prediction (4 x f32)
    #   fast R²:    data, prediction and residual promoted to f64 (3 x f64)
    bytes_per_voxel = 12 * max_train_tps + 16 * max_test_tps
    if use_fast_r2:
        bytes_per_voxel += 24 * max_test_tps

    budget = get_available_memory(device)
    cap = max(1_000, int(budget // max(bytes_per_voxel, 1)))
    if cap >= effective_batch_size:
        return effective_batch_size

    if verbose:
        print(
            f"  Capping batch to {cap:,} voxels for a {max_train_tps:,}-train / "
            f"{max_test_tps:,}-test timepoint fold ({budget / 1e9:.1f}GB usable)"
        )
    return cap


@torch.inference_mode()
def _unpredictable_basis(
    test_stim_design: torch.Tensor,
    test_only_mask: torch.Tensor,
    rcond: float = 1e-6,
) -> torch.Tensor | None:
    """Orthonormal basis for the events this fold had no chance of predicting.

    A condition whose events all fall inside the held-out runs has no beta from
    training, so under ``zero_event_strategy="nuisance"`` its contribution is
    removed from the held-out data rather than scored as error. Only
    ``test_only_mask`` belongs here: the train-only columns are zero across the
    whole test set by definition, so including them added zero columns to the
    basis, which is what forced the old ridge-stabilised inverse.

    Returns an orthonormal (n_test_tps, k) basis rather than the (T, T)
    projector the old code materialised -- k is the number of unpredictable
    conditions, so applying it twice is O(T*k) instead of O(T^2), and it is an
    exact orthogonal projector rather than the shrunken one that
    ``inv(X'X + 1e-6 I)`` produces. ``None`` when there is nothing to remove.
    """
    cols = test_stim_design[:, test_only_mask]
    if cols.shape[1] == 0:
        return None
    work = factor_device(cols.device)
    u, sv, _ = torch.linalg.svd(cols.to(device=work, dtype=torch.float64), full_matrices=False)
    keep = sv > rcond * sv[0]
    if not bool(keep.any()):
        return None
    return u[:, keep].to(device=cols.device, dtype=cols.dtype)


def _project_out_basis(values: torch.Tensor, basis: torch.Tensor, time_dim: int) -> torch.Tensor:
    """Remove ``basis``'s span from ``values`` along its timepoint axis."""
    if time_dim == 0:  # (n_timepoints, n_cols), a design
        return values - basis @ (basis.T @ values)
    return values - (values @ basis) @ basis.T  # (n_voxels, n_timepoints), data


def _plan_fold_designs(
    train_stim_design: torch.Tensor,
    test_stim_design: torch.Tensor,
    zero_event_strategy: str,
    train_runs: list[int],
    test_runs: list[int],
    device: torch.device,
    announce: bool = False,
) -> dict:
    """Work out which stimulus columns are usable in one CV fold.

    A condition can be absent from the training runs, the test runs, or both
    (rare designs, or any leave-N-out split of an unbalanced study). This is
    purely a function of the design, not of the data, so it is shared by both
    the streaming and the per-run-accumulation paths — keeping the two from
    drifting apart on the awkward cases.

    Returns the column masks plus ``fit_mask``, the columns actually entering
    the OLS fit under the requested strategy.
    """
    n_stim = train_stim_design.shape[1]
    train_zero_mask = train_stim_design.abs().sum(dim=0) < 1e-10
    test_zero_mask = test_stim_design.abs().sum(dim=0) < 1e-10

    train_present_mask = ~train_zero_mask
    test_present_mask = ~test_zero_mask
    unpredictable_mask = train_present_mask & test_zero_mask
    test_only_mask = train_zero_mask & test_present_mask
    predictable_mask = train_present_mask & test_present_mask

    if train_zero_mask.any() or test_zero_mask.any():
        if announce:
            unpredictable_cols = [i for i, u in enumerate(unpredictable_mask) if u]
            test_only_cols = [i for i, t in enumerate(test_only_mask) if t]
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
                f"All {n_stim} stimulus columns are zero in train or test."
            )

        if zero_event_strategy not in ("zero", "nuisance"):
            raise ValueError(f"Unknown zero_event_strategy: '{zero_event_strategy}'")
        # Both strategies fit everything the training runs can see. 'nuisance'
        # used to fit only the predictable columns, which left a train-only
        # column's variance in the training data to bias the betas that were
        # kept -- omitted-variable bias, visible only when the dropped column is
        # collinear with a kept one, and buying nothing even when it is not
        # (a train-only column is zero across the test set, so it could never
        # have entered the prediction). The strategies differ on the test side.
        fit_mask = train_present_mask
    else:
        if announce:
            print(f"    No missing events - full overlap ({n_stim} conditions)")
        train_present_mask = torch.ones(n_stim, dtype=torch.bool, device=device)
        test_present_mask = torch.ones(n_stim, dtype=torch.bool, device=device)
        predictable_mask = train_present_mask
        unpredictable_mask = torch.zeros(n_stim, dtype=torch.bool, device=device)
        test_only_mask = torch.zeros(n_stim, dtype=torch.bool, device=device)
        fit_mask = train_present_mask

    # Under 'nuisance' the unpredictable events are removed from the held-out
    # DATA, so they must be removed from the design used to predict it as well:
    # anything a predictable condition shares with them is then dropped from both
    # sides instead of appearing in the data-side residual as fabricated error.
    # This is Frisch-Waugh-Lovell -- R2 is evaluated in the subspace orthogonal to
    # the unpredictable events. A temporally adjacent unpredictable event costs
    # power (a smaller subspace to be scored in), not accuracy.
    unpred_basis: torch.Tensor | None = None
    if zero_event_strategy == "nuisance":
        unpred_basis = _unpredictable_basis(test_stim_design, test_only_mask)
    test_stim_predictable = test_stim_design[:, predictable_mask]
    if unpred_basis is not None:
        test_stim_predictable = _project_out_basis(test_stim_predictable, unpred_basis, time_dim=0)
    # betas are indexed by the fit columns, so selecting the predictable ones out
    # of them needs the mask re-expressed in that basis, not the full-design one.
    predictable_within_fit = predictable_mask[fit_mask]

    return {
        "train_present_mask": train_present_mask,
        "test_present_mask": test_present_mask,
        "predictable_mask": predictable_mask,
        "unpredictable_mask": unpredictable_mask,
        "test_only_mask": test_only_mask,
        "fit_mask": fit_mask,
        "unpred_basis": unpred_basis,
        "test_stim_predictable": test_stim_predictable,
        "predictable_within_fit": predictable_within_fit,
    }


def compute_xval_r2(
    data: torch.Tensor,
    design_matrix: np.ndarray | torch.Tensor,
    run_starts: list[int],
    stim_indices: list[int],
    nuisance_indices: list[int],
    cv_splits: list[tuple[list[int], list[int]]],
    metric: str = "cod",
    zero_event_strategy: str = "zero",
    device: torch.device | None = None,
    batch_size: int | None = None,
    r2_method: str = "auto",
    verbose: bool = True,
) -> dict[str, torch.Tensor | int]:
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
    from fastfuncstuff.utils import get_device, to_tensor

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

        effective_batch_size = _cap_batch_to_free_vram(
            effective_batch_size,
            cv_splits=cv_splits,
            run_lengths=run_lengths,
            device=device,
            use_fast_r2=True,
            verbose=verbose,
        )

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

        effective_batch_size = _cap_batch_to_free_vram(
            effective_batch_size,
            cv_splits=cv_splits,
            run_lengths=run_lengths,
            device=device,
            use_fast_r2=False,
            verbose=verbose,
        )

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

    # =========================================================================
    # Per-run accumulation fast path
    # =========================================================================
    # When the caller has already removed nuisance (nuisance_indices empty --
    # the project-first house style), no per-fold projection touches the data,
    # and the only things a fold needs from the training set are X'X and X'y.
    # Both are sums over runs, and generate_cv_splits always emits
    # train == complement(test), so each fold is "all-run totals minus the
    # held-out runs". That replaces a full-length gather of the training data
    # per fold with two contiguous passes over the batch.
    #
    # FFS_XVAL_LEGACY=1 forces the original streaming loop, as an escape hatch
    # if this path is ever suspected in a result.
    use_per_run_path = len(nuisance_indices) == 0 and os.environ.get("FFS_XVAL_LEGACY", "") != "1"

    if use_per_run_path:
        stim_design_all = design_matrix[:, stim_indices]  # (n_timepoints, n_stim)
        run_bounds = [(run_starts[r], run_starts[r] + int(run_lengths[r])) for r in range(n_runs)]
        xtx_all = stim_design_all.T @ stim_design_all

        # Per-fold work that depends only on the design, done once for all batches
        fold_plans = []
        for split_idx, (train_runs, test_runs) in enumerate(cv_splits):
            train_tps, test_tps = split_info[split_idx]
            train_stim_design = stim_design_all[train_tps, :]
            test_stim_design = stim_design_all[test_tps, :]

            plan = _plan_fold_designs(
                train_stim_design=train_stim_design,
                test_stim_design=test_stim_design,
                zero_event_strategy=zero_event_strategy,
                train_runs=train_runs,
                test_runs=test_runs,
                device=device,
                announce=bool(split_idx == 0 and verbose),
            )
            fit_mask = plan["fit_mask"]

            xtx_train = xtx_all - test_stim_design.T @ test_stim_design
            xtx_fit = xtx_train[fit_mask][:, fit_mask]
            xtx_fit_inv = torch.linalg.inv(
                xtx_fit + 1e-6 * torch.eye(xtx_fit.shape[0], device=device)
            )

            plan.update(
                {
                    "test_runs": test_runs,
                    "test_tps": test_tps,
                    "test_stim_design": test_stim_design,
                    "xtx_fit_inv": xtx_fit_inv,
                }
            )
            fold_plans.append(plan)

            if not use_fast_r2:
                count_per_timepoint[test_tps] += 1.0

        n_stim_all = stim_design_all.shape[1]

        for batch_start in range(0, n_voxels, effective_batch_size):
            batch_end = min(batch_start + effective_batch_size, n_voxels)
            batch_slice = slice(batch_start, batch_end)

            # Pass 1: X'y over every run, so no fold has to touch the training data
            xty_all = torch.zeros(
                n_stim_all, batch_end - batch_start, dtype=torch.float32, device=device
            )
            for r in range(n_runs):
                start_tp, end_tp = run_bounds[r]
                y_run = data[batch_slice, start_tp:end_tp].to(device)
                xty_all += stim_design_all[start_tp:end_tp, :].T @ y_run.T
                del y_run

            # Pass 2: folds, touching only their held-out runs
            for plan in fold_plans:
                test_tps = plan["test_tps"]
                fit_mask = plan["fit_mask"]
                test_data_batch = torch.cat(
                    [
                        data[batch_slice, run_bounds[r][0] : run_bounds[r][1]]
                        for r in plan["test_runs"]
                    ],
                    dim=1,
                ).to(device)

                xty_train = xty_all - plan["test_stim_design"].T @ test_data_batch.T
                betas_fit = plan["xtx_fit_inv"] @ xty_train[fit_mask]

                # Data and prediction design get the SAME projection; the design
                # side was done once per fold in _plan_fold_designs.
                if plan["unpred_basis"] is not None:
                    test_data_batch = _project_out_basis(
                        test_data_batch, plan["unpred_basis"], time_dim=1
                    )

                if zero_event_strategy == "zero":
                    betas_full = torch.zeros(n_stim_all, betas_fit.shape[1], device=device)
                    betas_full[plan["train_present_mask"], :] = betas_fit
                    test_stim_present = plan["test_stim_design"][:, plan["test_present_mask"]]
                    predictions_batch = (
                        test_stim_present @ betas_full[plan["test_present_mask"], :]
                    ).T
                else:
                    predictions_batch = (
                        plan["test_stim_predictable"] @ betas_fit[plan["predictable_within_fit"]]
                    ).T

                if use_fast_r2:
                    # float64 accumulation without materialising float64 copies
                    # (MPS has no float64, so reduce in float32 there).
                    if test_data_batch.device.type == "mps":
                        red_dtype = torch.float32
                    else:
                        red_dtype = torch.float64
                    residuals = test_data_batch - predictions_batch
                    ss_res_batch = (residuals * residuals).sum(dim=1, dtype=red_dtype)
                    sum_actual_batch = test_data_batch.sum(dim=1, dtype=red_dtype)
                    sum_sq_actual_batch = (test_data_batch * test_data_batch).sum(
                        dim=1, dtype=red_dtype
                    )
                    ss_res_accumulator[batch_slice] += ss_res_batch.to(accumulator_device).double()
                    sum_actual[batch_slice] += sum_actual_batch.to(accumulator_device).double()
                    sum_sq_actual[batch_slice] += sum_sq_actual_batch.to(
                        accumulator_device
                    ).double()
                    count_timepoints[batch_slice] += len(test_tps)
                else:
                    pred_accumulator[batch_slice, test_tps] += predictions_batch.to(
                        accumulator_device
                    )
                    actual_accumulator[batch_slice, test_tps] += test_data_batch.to(
                        accumulator_device
                    )

                del test_data_batch, predictions_batch, betas_fit, xty_train

            del xty_all

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Process CV splits (empty when the per-run fast path above already ran)
    legacy_splits = [] if use_per_run_path else list(enumerate(cv_splits))
    for split_idx, (train_runs, test_runs) in legacy_splits:
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

        # 4b. Which stimulus columns are usable in this fold (design-only decision)
        plan = _plan_fold_designs(
            train_stim_design=train_stim_design,
            test_stim_design=test_stim_design,
            zero_event_strategy=zero_event_strategy,
            train_runs=train_runs,
            test_runs=test_runs,
            device=device,
            announce=bool(split_idx == 0 and verbose),
        )
        train_present_mask = plan["train_present_mask"]
        test_present_mask = plan["test_present_mask"]
        predictable_mask = plan["predictable_mask"]
        unpredictable_mask = plan["unpredictable_mask"]
        test_only_mask = plan["test_only_mask"]
        unpred_basis = plan["unpred_basis"]
        test_stim_predictable = plan["test_stim_predictable"]
        predictable_within_fit = plan["predictable_within_fit"]
        train_stim_design_fit = train_stim_design[:, plan["fit_mask"]]

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
            if unpred_basis is not None:
                unpred_basis = unpred_basis.cpu()
            test_stim_predictable = test_stim_predictable.cpu()
            predictable_within_fit = predictable_within_fit.cpu()
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
                train_data_batch = (
                    train_data_batch - (Q_train_batch @ (Q_train_batch.T @ train_data_batch.T)).T
                )
            if Q_test_batch is not None:
                test_data_batch = (
                    test_data_batch - (Q_test_batch @ (Q_test_batch.T @ test_data_batch.T)).T
                )

            # Additional projection for 'nuisance' strategy. Data and prediction
            # design get the SAME projection; the design side was done once per
            # fold in _plan_fold_designs.
            if unpred_basis is not None:
                test_data_batch = _project_out_basis(test_data_batch, unpred_basis, time_dim=1)

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
                predictions_batch = (test_stim_predictable @ betas_fit[predictable_within_fit]).T
            else:
                raise ValueError(f"Invalid strategy: {zero_event_strategy}")

            # ============================================================
            # ACCUMULATE: Fast mode (streaming stats) vs Slow mode (full timeseries)
            # ============================================================
            if use_fast_r2:
                # FAST MODE: Accumulate streaming stats (SS_res, sum, sum_sq)
                # Compute residuals and stats on GPU, accumulate to accumulators
                n_test_tps = len(test_tps)

                # Compute per-batch stats. On CUDA/CPU we promote to float64 on
                # the compute device for the subtraction. MPS has no float64, so
                # there we reduce in float32 on-device (SS_res is a difference, so
                # no catastrophic cancellation) and promote to float64 only after
                # moving onto the CPU float64 accumulators below.
                if test_data_batch.device.type == "mps":
                    residuals = test_data_batch - predictions_batch
                    ss_res_batch = (residuals**2).sum(dim=1)
                    sum_actual_batch = test_data_batch.sum(dim=1)
                    sum_sq_actual_batch = (test_data_batch**2).sum(dim=1)
                else:
                    test_data_f64 = test_data_batch.double()
                    pred_f64 = predictions_batch.double()
                    residuals = test_data_f64 - pred_f64
                    ss_res_batch = (residuals**2).sum(dim=1)
                    sum_actual_batch = test_data_f64.sum(dim=1)
                    sum_sq_actual_batch = (test_data_f64**2).sum(dim=1)

                # Accumulate (may need to move to CPU if accumulators are there).
                # .cpu().double() is a no-op cast when already float64 (CUDA), and
                # the float32→float64 promotion on MPS.
                if accumulator_device.type == "cpu":
                    ss_res_accumulator[batch_slice] += ss_res_batch.cpu().double()
                    sum_actual[batch_slice] += sum_actual_batch.cpu().double()
                    sum_sq_actual[batch_slice] += sum_sq_actual_batch.cpu().double()
                    count_timepoints[batch_slice] += n_test_tps
                else:
                    ss_res_accumulator[batch_slice] += ss_res_batch
                    sum_actual[batch_slice] += sum_actual_batch
                    sum_sq_actual[batch_slice] += sum_sq_actual_batch
                    count_timepoints[batch_slice] += n_test_tps

                del residuals, ss_res_batch, sum_actual_batch, sum_sq_actual_batch
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
        q25, q50, q75 = torch.quantile(r2_final.float(), torch.tensor([0.25, 0.50, 0.75]))
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


def single_trial_cv_helper(
    beta_variants: torch.Tensor,
    trial_condition_ids: torch.Tensor,
    trial_run_ids: torch.Tensor,
    cv_splits: list[tuple[list[int], list[int]]],
    metric: str = "cod",
    zscore_by_run: bool = False,
    reference_variant_idx: int = 0,
    test_variant_idx: int | None = None,
    device: torch.device | None = None,
    chunk_size: int | None = None,
    verbose: bool = True,
) -> dict:
    """
    Batch beta-series cross-validation over multiple hyperparameter variants.

    Evaluates multiple beta sets (e.g. different ridge fractions or PC counts)
    simultaneously using LORO CV in beta space. For each fold, condition-average
    train betas predict held-out test trial betas.

    Parameters
    ----------
    beta_variants : torch.Tensor
        (n_variants, n_voxels, n_trials) betas for each hyperparameter variant.
    trial_condition_ids : torch.Tensor
        (n_trials,) integer condition label for each trial.
    trial_run_ids : torch.Tensor
        (n_trials,) integer run index for each trial.
    cv_splits : list of (train_runs, test_runs)
        LORO or other CV splits.
    metric : str, default='cod'
        'cod', 'corr', 'corr2', or 'sse'.
        When 'sse', uses GLMsingle's ``calcbadness`` algorithm: each test trial
        is compared against ALL matching-condition individual training trial
        betas (not condition averages), test betas always come from variant 0
        (unregularized), and z-scoring is per session (all runs pooled).
    zscore_by_run : bool, default=False
        If True, z-score betas before CV. Normalization stats (mu, sigma)
        are computed from ``reference_variant_idx`` and applied to ALL variants
        (GLMsingle pattern: unregularized variant sets the scale).
        When ``metric='sse'``, z-scoring is done per session (all runs pooled
        together) matching GLMsingle's ``calcbadness``. For other metrics,
        z-scoring is done per run.
    reference_variant_idx : int, default=0
        Which variant supplies the z-scoring normalization stats. Only used when
        ``zscore_by_run=True``.
    test_variant_idx : int, optional
        If set, the test ("actual") targets for ALL variants come from this
        variant's betas. This is the GLMsingle fracridge pattern: compare every
        regularised variant's condition-average train betas against the OLS
        (frac=1) test betas, so the absolute amplitude differences are the
        discriminating signal.  Leave as None for the standard mode where each
        variant is compared against its own test betas.
        When ``metric='sse'``, this is forced to 0 (matching GLMsingle).
    device : torch.device, optional
        Compute device (auto-detected if None).
    chunk_size : int, optional
        Voxel chunk size. None = process all at once.
    verbose : bool, default=True
        Print progress.

    Returns
    -------
    dict with:
        'r2': (n_variants, n_voxels) per-voxel CV R² for each variant
        'r2_mean': (n_variants,) mean R² across voxels
        'n_splits': int
        'n_test_trials_total': int
    """
    from fastfuncstuff.utils import get_device

    if device is None:
        device = get_device()

    n_variants, n_voxels, n_trials = beta_variants.shape
    n_conditions = int(trial_condition_ids.max().item()) + 1
    n_splits = len(cv_splits)

    # Move small tensors to device
    trial_condition_ids = trial_condition_ids.to(device)
    trial_run_ids = trial_run_ids.to(device)

    if chunk_size is None:
        if device is not None and device.type == "cuda":
            # Auto-size chunks to fit in ~2GB GPU working memory.
            # Per chunk: betas (n_variants * chunk * n_trials * 4B)
            #          + condition_avg (n_variants * chunk * n_conditions * 4B)
            #          + predicted (n_variants * chunk * n_test * 4B)
            # Rough: 3 * n_variants * chunk * n_trials * 4
            bytes_per_voxel = 3 * n_variants * n_trials * 4
            target_bytes = 2 * 1024**3  # 2 GB
            chunk_size = max(1000, int(target_bytes / max(bytes_per_voxel, 1)))
            chunk_size = min(chunk_size, n_voxels)
        else:
            chunk_size = n_voxels

    # =========================================================================
    # Optional z-scoring (GLMsingle normalization)
    # =========================================================================
    # For SSE metric, default test_variant_idx to 0 (GLMsingle's calcbadness
    # convention). But if caller explicitly provides a variant index (e.g.,
    # ridge with frac=1.0 at the last index), honor that.
    if metric == "sse" and test_variant_idx is None:
        test_variant_idx = 0

    if metric == "sse":
        # GLMsingle calcbadness: ALWAYS z-score per session (all runs pooled).
        # This is unconditional in GLMsingle — part of the calcbadness algorithm,
        # not a user option. mu/sigma from variant 0 (unregularized).
        # Clone to avoid inplace update on inference-mode tensors.
        beta_variants = beta_variants.clone()
        ref_betas = beta_variants[reference_variant_idx]  # (n_vox, n_trials)
        mu = ref_betas.mean(dim=1, keepdim=True)  # (n_vox, 1)
        sigma = ref_betas.std(dim=1, keepdim=True)  # (n_vox, 1)
        sigma = sigma.clamp(min=1e-10)
        for v in range(n_variants):
            beta_variants[v] = (beta_variants[v] - mu) / sigma
    elif zscore_by_run:
        # Optional per-run z-scoring for non-SSE metrics.
        # Clone to avoid inplace update on inference-mode tensors.
        beta_variants = beta_variants.clone()
        beta_device = beta_variants.device
        unique_runs = torch.unique(trial_run_ids).tolist()
        for run_id in unique_runs:
            run_mask = (trial_run_ids == run_id).to(beta_device)
            ref_betas = beta_variants[reference_variant_idx, :, run_mask]  # (n_vox, n_run_trials)
            mu = ref_betas.mean(dim=1, keepdim=True)  # (n_vox, 1)
            sigma = ref_betas.std(dim=1, keepdim=True)  # (n_vox, 1)
            sigma = sigma.clamp(min=1e-10)
            for v in range(n_variants):
                beta_variants[v, :, run_mask] = (beta_variants[v, :, run_mask] - mu) / sigma

    # =========================================================================
    # Chunk-outer fast path
    # =========================================================================
    # The fold loop only ever needs, per condition, the sum (and for SSE the sum
    # of squares) over the *training* trials. Those are sums over runs, and the
    # CV splits partition the runs, so each fold is the all-trial total minus its
    # held-out trials — no per-fold gather over the training set, and no reason
    # to re-upload the same betas once per fold.
    #
    # Only conditions that actually appear in the test set can contribute, so the
    # per-fold work is proportional to the held-out trials rather than to
    # n_conditions.
    #
    # FFS_RIDGE_CV_LEGACY=1 forces the original fold-outer loop.
    all_run_ids = set(torch.unique(trial_run_ids).tolist())
    splits_partition_runs = all(
        set(tr) | set(te) == all_run_ids and not (set(tr) & set(te)) for tr, te in cv_splits
    )
    use_chunk_outer = splits_partition_runs and os.environ.get("FFS_RIDGE_CV_LEGACY", "") != "1"

    if use_chunk_outer:
        cond_ids_dev = trial_condition_ids.to(device)
        total_count = torch.bincount(cond_ids_dev, minlength=n_conditions).to(
            device=device, dtype=torch.float32
        )

        # Per-fold trial bookkeeping (voxel-independent, done once)
        folds = []
        for train_runs, test_runs in cv_splits:  # noqa: B007
            test_mask = torch.zeros(n_trials, dtype=torch.bool, device=device)
            for r in test_runs:
                test_mask |= trial_run_ids == r
            test_indices = torch.where(test_mask)[0]
            if len(test_indices) == 0:
                continue
            test_conditions = cond_ids_dev[test_indices]
            uniq_conds, inv = torch.unique(test_conditions, return_inverse=True)
            test_counts = torch.bincount(inv, minlength=uniq_conds.numel()).to(torch.float32)
            folds.append((test_indices, uniq_conds, inv, test_counts))

        total_test_trials = sum(len(f[0]) for f in folds)

        is_sse = metric == "sse"
        if is_sse:
            sse_accum = torch.zeros(n_variants, n_voxels, device="cpu")
        else:
            # Sufficient statistics for cod/corr/corr2 over the concatenated test
            # trials — replaces materialising the full predicted/actual matrices,
            # which at production scale is several GB of host memory.
            stat_shape = (n_variants, n_voxels)
            acc_sum_y = torch.zeros(stat_shape, dtype=torch.float64, device="cpu")
            acc_sum_p = torch.zeros(stat_shape, dtype=torch.float64, device="cpu")
            acc_sum_yy = torch.zeros(stat_shape, dtype=torch.float64, device="cpu")
            acc_sum_pp = torch.zeros(stat_shape, dtype=torch.float64, device="cpu")
            acc_sum_yp = torch.zeros(stat_shape, dtype=torch.float64, device="cpu")
            acc_ss_res = torch.zeros(stat_shape, dtype=torch.float64, device="cpu")

        for chunk_start in range(0, n_voxels, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_voxels)
            chunk_slice = slice(chunk_start, chunk_end)
            n_chunk = chunk_end - chunk_start

            # Uploaded once per chunk instead of once per (fold, chunk)
            betas_chunk = beta_variants[:, chunk_slice, :].to(device)

            total_sum = torch.zeros(n_variants, n_chunk, n_conditions, device=device).index_add_(
                2, cond_ids_dev, betas_chunk
            )
            total_sum_sq = None
            if is_sse:
                total_sum_sq = torch.zeros(
                    n_variants, n_chunk, n_conditions, device=device
                ).index_add_(2, cond_ids_dev, betas_chunk * betas_chunk)

            for test_indices, uniq_conds, inv, test_counts in folds:
                test_betas = betas_chunk[:, :, test_indices]  # (n_var, n_chunk, n_test)

                # Train totals for the conditions present in this test set
                test_sum = torch.zeros(
                    n_variants, n_chunk, uniq_conds.numel(), device=device
                ).index_add_(2, inv, test_betas)
                train_sum = total_sum[:, :, uniq_conds] - test_sum
                train_count = total_count[uniq_conds] - test_counts

                if is_sse:
                    test_sum_sq = torch.zeros(
                        n_variants, n_chunk, uniq_conds.numel(), device=device
                    ).index_add_(2, inv, test_betas * test_betas)
                    train_sum_sq = total_sum_sq[:, :, uniq_conds] - test_sum_sq

                    # GLMsingle calcbadness: each test trial against every
                    # matching-condition training trial, expanded algebraically.
                    ref_test = betas_chunk[int(test_variant_idx), :, test_indices]
                    n_train_per_test = train_count[inv]  # (n_test,)
                    sum_train_per_test = train_sum[:, :, inv]  # (n_var, n_chunk, n_test)
                    sum_sq_train_per_test = train_sum_sq[:, :, inv]
                    sse_per_test = (
                        n_train_per_test * ref_test.unsqueeze(0) ** 2
                        - 2 * ref_test.unsqueeze(0) * sum_train_per_test
                        + sum_sq_train_per_test
                    )
                    sse_accum[:, chunk_slice] += sse_per_test.sum(dim=-1).cpu()
                    del test_sum_sq, train_sum_sq, sse_per_test
                else:
                    cond_avg = train_sum / train_count.clamp(min=1.0)
                    predicted = cond_avg[:, :, inv]  # (n_var, n_chunk, n_test)
                    if test_variant_idx is not None:
                        ref_test = betas_chunk[int(test_variant_idx), :, test_indices]
                        actual = ref_test.unsqueeze(0).expand(n_variants, -1, -1)
                    else:
                        actual = test_betas

                    resid = actual - predicted
                    acc_sum_y[:, chunk_slice] += actual.sum(dim=-1, dtype=torch.float64).cpu()
                    acc_sum_p[:, chunk_slice] += predicted.sum(dim=-1, dtype=torch.float64).cpu()
                    acc_sum_yy[:, chunk_slice] += (
                        (actual * actual).sum(dim=-1, dtype=torch.float64).cpu()
                    )
                    acc_sum_pp[:, chunk_slice] += (
                        (predicted * predicted).sum(dim=-1, dtype=torch.float64).cpu()
                    )
                    acc_sum_yp[:, chunk_slice] += (
                        (actual * predicted).sum(dim=-1, dtype=torch.float64).cpu()
                    )
                    acc_ss_res[:, chunk_slice] += (
                        (resid * resid).sum(dim=-1, dtype=torch.float64).cpu()
                    )
                    del cond_avg, predicted, actual, resid

                del test_betas, test_sum, train_sum

            del betas_chunk, total_sum, total_sum_sq
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if is_sse:
            r2 = sse_accum
        else:
            n_pts = float(total_test_trials)
            if metric == "cod":
                ss_tot = acc_sum_yy - acc_sum_y**2 / n_pts
                r2 = (1.0 - acc_ss_res / (ss_tot + 1e-10)).float()
            else:
                # Pearson from sufficient statistics
                cov = acc_sum_yp - acc_sum_y * acc_sum_p / n_pts
                var_y = acc_sum_yy - acc_sum_y**2 / n_pts
                var_p = acc_sum_pp - acc_sum_p**2 / n_pts
                corr = cov / (
                    torch.sqrt(var_y.clamp(min=0)) * torch.sqrt(var_p.clamp(min=0)) + 1e-10
                )
                r2 = (corr**2).float() if metric == "corr2" else corr.float()

        if verbose:
            r2_mean = r2.mean(dim=1)
            label = "mean SSE" if is_sse else "mean R²"
            for v in range(min(n_variants, 5)):
                val = f"{r2_mean[v]:.1f}" if is_sse else f"{r2_mean[v]:.4f}"
                print(f"  Variant {v}: {label}={val}")
            if n_variants > 5:
                print(f"  ... ({n_variants - 5} more variants)")
            print(f"  ({total_test_trials} test trials across {n_splits} folds)")

        return {
            "r2": r2,
            "r2_mean": r2.mean(dim=1),
            "n_splits": n_splits,
            "n_test_trials_total": total_test_trials,
        }

    # =========================================================================
    # Build fold masks once
    # =========================================================================
    fold_info = []
    for train_runs, test_runs in cv_splits:
        train_mask = torch.zeros(n_trials, dtype=torch.bool, device=device)
        for r in train_runs:
            train_mask |= trial_run_ids == r
        test_mask = torch.zeros(n_trials, dtype=torch.bool, device=device)
        for r in test_runs:
            test_mask |= trial_run_ids == r
        test_indices = torch.where(test_mask)[0]
        if len(test_indices) == 0:
            continue
        test_conditions = trial_condition_ids[test_indices]

        # Pre-compute per-condition train masks for vectorized averaging
        cond_train_masks = []
        for c in range(n_conditions):
            cond_train_masks.append(train_mask & (trial_condition_ids == c))

        fold_info.append((train_mask, test_indices, test_conditions, cond_train_masks))

    # =========================================================================
    # SSE (GLMsingle calcbadness) vs standard (condition-average) path
    # =========================================================================
    if metric == "sse":
        # GLMsingle calcbadness: for each test trial, compare against ALL
        # matching-condition individual training trial betas. Accumulate SSE
        # directly per variant per voxel across folds.
        # Test betas come from test_variant_idx (default 0 for GLMsingle,
        # but callers may override, e.g. ridge where frac=1.0 is the OLS reference).
        sse_accum = torch.zeros(n_variants, n_voxels, device="cpu")
        total_test_trials = 0

        for fold_idx, (train_mask, test_indices, test_conditions, cond_train_masks) in enumerate(  # noqa: B007
            fold_info
        ):
            if verbose:
                train_runs, test_runs = cv_splits[fold_idx]
                print(f"  Split {fold_idx + 1}/{n_splits}: Train {train_runs} | Test {test_runs}")

            n_test = len(test_indices)
            total_test_trials += n_test

            for chunk_start in range(0, n_voxels, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_voxels)
                n_chunk = chunk_end - chunk_start

                # (n_variants, n_chunk, n_trials)
                betas_chunk = beta_variants[:, chunk_start:chunk_end, :].to(device)

                # Test betas from selected reference variant: (n_chunk, n_test)
                # test_variant_idx is guaranteed non-None by the block above.
                test_betas_ref = betas_chunk[int(test_variant_idx), :, test_indices]

                # Vectorized SSE per condition — eliminates per-trial Python loop.
                # Algebraic expansion: Σ_k(train_k - test_t)²
                #   = n_train·test_t² - 2·test_t·Σ_k(train_k) + Σ_k(train_k²)
                # This lets us sum over both train and test trials on GPU simultaneously,
                # replacing O(n_test) GPU syncs+transfers with O(n_conditions) per chunk.
                sse_chunk = torch.zeros(n_variants, n_chunk, device=device)
                for cond in range(n_conditions):
                    cm = cond_train_masks[cond]
                    if not cm.any():
                        continue
                    test_cond_mask = test_conditions == cond  # (n_test,) on device
                    if not test_cond_mask.any():
                        continue

                    # (n_variants, n_chunk, n_train_cond)
                    train_betas = betas_chunk[:, :, cm]
                    n_train = train_betas.shape[-1]
                    sum_train = train_betas.sum(dim=-1)  # (n_variants, n_chunk)
                    sum_sq_train = (train_betas**2).sum(dim=-1)  # (n_variants, n_chunk)

                    # (n_chunk, n_cond_test) — test betas for this condition
                    test_betas = test_betas_ref[:, test_cond_mask]

                    # (n_variants, n_chunk, n_cond_test)
                    sse_per_test = (
                        n_train * test_betas.unsqueeze(0) ** 2
                        - 2 * test_betas.unsqueeze(0) * sum_train.unsqueeze(-1)
                        + sum_sq_train.unsqueeze(-1)
                    )
                    sse_chunk += sse_per_test.sum(dim=-1)

                sse_accum[:, chunk_start:chunk_end] += sse_chunk.cpu()

        r2 = sse_accum  # "r2" is SSE here (lower = better)

        if verbose:
            r2_mean = r2.mean(dim=1)
            for v in range(min(n_variants, 5)):
                print(f"  Variant {v}: mean SSE={r2_mean[v]:.1f}")
            if n_variants > 5:
                print(f"  ... ({n_variants - 5} more variants)")
            print(f"  ({total_test_trials} test trials across {n_splits} folds)")

        return {
            "r2": r2,  # (n_variants, n_voxels) — SSE values
            "r2_mean": r2.mean(dim=1),  # (n_variants,)
            "n_splits": n_splits,
            "n_test_trials_total": total_test_trials,
        }

    # =========================================================================
    # Standard path: condition-average prediction (cod, corr, corr2)
    # =========================================================================
    all_predicted = []  # list of (n_variants * n_voxels, n_test_this_fold)
    all_actual = []

    for fold_idx, (_train_mask, test_indices, test_conditions, cond_train_masks) in enumerate(
        fold_info
    ):
        if verbose:
            train_runs, test_runs = cv_splits[fold_idx]
            print(f"  Split {fold_idx + 1}/{n_splits}: Train {train_runs} | Test {test_runs}")

        n_test = len(test_indices)

        fold_pred_chunks = []
        fold_actual_chunks = []

        for chunk_start in range(0, n_voxels, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_voxels)
            n_chunk = chunk_end - chunk_start

            # (n_variants, n_chunk, n_trials)
            betas_chunk = beta_variants[:, chunk_start:chunk_end, :].to(device)

            # Condition averages from train trials: (n_variants, n_chunk, n_conditions)
            condition_avg = torch.zeros(n_variants, n_chunk, n_conditions, device=device)
            for c in range(n_conditions):
                cm = cond_train_masks[c]
                n_train_c = cm.sum().item()
                if n_train_c > 0:
                    condition_avg[:, :, c] = betas_chunk[:, :, cm].mean(dim=2)

            # Predicted: index condition_avg by test_conditions
            predicted = condition_avg[:, :, test_conditions]
            if test_variant_idx is not None:
                ref_test = betas_chunk[test_variant_idx, :, test_indices]
                actual = ref_test.unsqueeze(0).expand(n_variants, -1, -1)
            else:
                actual = betas_chunk[:, :, test_indices]

            fold_pred_chunks.append(predicted.cpu())
            fold_actual_chunks.append(actual.cpu())

        fold_pred = torch.cat(fold_pred_chunks, dim=1).reshape(n_variants * n_voxels, n_test)
        fold_actual = torch.cat(fold_actual_chunks, dim=1).reshape(n_variants * n_voxels, n_test)

        all_predicted.append(fold_pred)
        all_actual.append(fold_actual)

    # Concatenate across folds: (n_variants * n_voxels, total_test_trials)
    all_predicted_cat = torch.cat(all_predicted, dim=1)
    all_actual_cat = torch.cat(all_actual, dim=1)
    total_test_trials = all_predicted_cat.shape[1]

    # Compute R²: (n_variants * n_voxels,)
    r2_flat = compute_r2_metric(all_actual_cat, all_predicted_cat, metric=metric)

    # Reshape to (n_variants, n_voxels)
    r2 = r2_flat.reshape(n_variants, n_voxels)

    if verbose:
        r2_mean = r2.mean(dim=1)
        for v in range(min(n_variants, 5)):
            print(f"  Variant {v}: mean R²={r2_mean[v]:.4f}")
        if n_variants > 5:
            print(f"  ... ({n_variants - 5} more variants)")
        print(f"  ({total_test_trials} test trials across {n_splits} folds)")

    return {
        "r2": r2,  # (n_variants, n_voxels)
        "r2_mean": r2.mean(dim=1),  # (n_variants,)
        "n_splits": n_splits,
        "n_test_trials_total": total_test_trials,
    }


def compute_xval_r2_single_trials(
    single_trial_betas: torch.Tensor,
    trial_condition_ids: torch.Tensor,
    trial_run_ids: torch.Tensor,
    cv_splits: list[tuple[list[int], list[int]]],
    metric: str = "cod",
    device: torch.device | None = None,
    chunk_size: int | None = None,
    verbose: bool = True,
) -> dict[str, torch.Tensor | int]:
    """
    Cross-validated R² in single-trial beta space (GLMsingle-style fit-once).

    Thin wrapper around :func:`single_trial_cv_helper` for backward compatibility.
    Accepts a single (n_voxels, n_trials) beta matrix and returns the same dict
    structure as before.

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
    # Wrap single beta set as 1-variant batch
    result = single_trial_cv_helper(
        beta_variants=single_trial_betas.unsqueeze(0),
        trial_condition_ids=trial_condition_ids,
        trial_run_ids=trial_run_ids,
        cv_splits=cv_splits,
        metric=metric,
        zscore_by_run=False,
        device=device,
        chunk_size=chunk_size,
        verbose=verbose,
    )

    r2 = result["r2"].squeeze(0)  # (n_voxels,)
    return {
        "r2": r2,
        "r2_median": r2,  # Backward compat: misleading name, actually per-voxel R² tensor
        "r2_mean": r2.mean(),  # Scalar mean for convenience
        "n_splits": result["n_splits"],
        "n_test_trials_total": result["n_test_trials_total"],
    }
