"""
Ridge regression for fMRI with single-trial estimation

This module implements GPU-accelerated ridge regression using fracridge,
with support for:
- Single-trial beta estimation (one beta per event)
- Per-voxel HRF selection (from HRFoptfast output)
- Noise regressor integration (from Denoisefast output)
- Cross-validated ridge fraction selection
- Non-TR-locked onsets and variable durations

Design philosophy:
- Reproduces GLMsingle functionality but GPU-accelerated
- Compatible with existing fastfuncsim HRF and denoising pipelines
- Supports flexible timing (non-TR-locked, variable durations)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# Import for memory-aware chunking
try:
    from fastfuncsim.memory import estimate_chunk_size
except ImportError:
    # Fallback if memory module not available
    estimate_chunk_size = None

# Import for R² metric computation
from fastfuncsim.xval import compute_r2_metric


def _fit_ridge_multiple_fracs(
    X: torch.Tensor,
    y: torch.Tensor,
    fracs: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    Fit ridge regression for multiple fractions using fracridge algorithm

    Implements proper fractional ridge where frac represents the fraction
    of unregularized (OLS) coefficient norm to retain:
    - frac=1.0 → keep 100% of OLS norm → alpha≈0 → no regularization
    - frac=0.0 → keep 0% of OLS norm → alpha→∞ → maximum regularization

    This matches the fracridge library implementation.

    Parameters
    ----------
    X : torch.Tensor, shape (n_samples, n_features)
        Design matrix
    y : torch.Tensor, shape (n_samples, n_targets)
        Target data (multiple targets)
    fracs : np.ndarray, shape (n_fracs,)
        Ridge fractions (1 = OLS, 0 = maximum regularization)
    device : torch.device
        Device for computation

    Returns
    -------
    coefs : torch.Tensor, shape (n_features, n_fracs, n_targets)
        Coefficients for each fraction and target
    """
    X = X.to(device)
    y = y.to(device)

    n_samples, n_features = X.shape
    n_targets = y.shape[1]
    n_fracs = len(fracs)

    # Compute SVD of X: X = U @ S @ Vt
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)

    # Handle rank-deficiency: Filter out zero/tiny singular values
    # This prevents 0/0 = NaN when computing sclg with alpha=0
    tol = 1e-10 * S[0]  # Tolerance relative to largest singular value
    valid_mask = S > tol
    n_valid = valid_mask.sum().item()

    if n_valid < len(S):
        # Rank deficient - truncate to valid singular values
        S = S[valid_mask]
        U = U[:, valid_mask]
        Vt = Vt[valid_mask, :]

    # Compute U'y
    Uty = U.T @ y  # (n_valid, n_targets)

    # OLS solution in rotated space: S^{-1} U'y
    ols_coef_rotated = Uty / S.unsqueeze(1)  # (n_valid, n_targets)

    # Squared singular values
    S_sq = S ** 2

    # Keep track of valid rank (may be less than n_features if rank-deficient)
    n_valid_rank = len(S)

    # Edge case: completely rank-deficient (no valid singular values)
    if n_valid_rank == 0:
        # Return zero coefficients for all fractions
        n_fracs = len(fracs)
        coefs = torch.zeros(n_features, n_fracs, n_targets, device=device)
        return coefs

    # Create alpha grid for interpolation (log-spaced from small to large)
    # Match fracridge exactly: BIG_BIAS = 10e3 = 10000, SMALL_BIAS = 10e-3 = 0.01
    SMALL_BIAS = 10e-3  # 0.01
    BIG_BIAS = 10e3     # 10000

    # Match fracridge logic: if smallest singular value squared is zero, use SMALL_BIAS
    S_min_sq = S[-1].item() ** 2
    if S_min_sq == 0:
        val2 = SMALL_BIAS
    else:
        val2 = SMALL_BIAS * S_min_sq

    val1 = BIG_BIAS * S[0].item() ** 2

    # Log-spaced grid with step 0.2 (like fracridge BIAS_STEP = 0.2)
    log_min = np.floor(np.log10(val2))
    log_max = np.ceil(np.log10(val1))

    # Limit grid size to avoid memory issues (max ~125 points with step 0.2)
    if (log_max - log_min) > 25:  # 25 / 0.2 = 125 points
        log_min = log_max - 25

    # Include alpha=0 at the start (for pure OLS case)
    alphagrid = torch.cat([
        torch.tensor([0.0], device=device, dtype=X.dtype),
        torch.tensor(10 ** np.arange(log_min, log_max, 0.2), device=device, dtype=X.dtype)
    ])

    # Compute coefficient scaling for each alpha in grid
    # sclg = S^2 / (S^2 + alpha), shape: (n_alphas, n_valid_rank)
    sclg = S_sq.unsqueeze(0) / (S_sq.unsqueeze(0) + alphagrid.unsqueeze(1))
    sclg_sq = sclg ** 2

    # Allocate output (will be filled after unrotation)
    coefs = torch.zeros(n_features, n_fracs, n_targets, device=device)

    # ========================================================================
    # Vectorized interpolation for all targets at once
    # ========================================================================
    # Compute coefficient length for each alpha in grid, for all targets
    # newlen: (n_alphas, n_targets)
    newlen = torch.sqrt(torch.einsum('aj,jt->at', sclg_sq, ols_coef_rotated ** 2))

    # Normalize to OLS length (alphagrid[0] = 0)
    # Add epsilon to avoid division by zero for zero-variance voxels
    ols_len = newlen[0:1, :]  # (1, n_targets)
    newlen = newlen / (ols_len + 1e-10)  # (n_alphas, n_targets)

    # Handle edge case: if OLS length is zero/tiny, set to 1.0 (will pick OLS = frac 1.0)
    zero_variance = ols_len.squeeze() < 1e-8
    if zero_variance.any():
        # For zero-variance voxels, set newlen to 1.0 everywhere (picks frac=1.0 which is fine)
        newlen[:, zero_variance] = 1.0

    # Move to CPU for interpolation
    log_alphagrid_np = np.log(1 + alphagrid.flip(0).cpu().numpy())  # (n_alphas,)
    newlen_np = newlen.flip(0).cpu().numpy()  # (n_alphas, n_targets)

    # Per-target interpolation (each voxel has its own coefficient length curve)
    log_target_alphas_all = np.zeros((n_fracs, n_targets), dtype=np.float32)
    for target_idx in range(n_targets):
        log_target_alphas_all[:, target_idx] = np.interp(
            fracs, newlen_np[:, target_idx], log_alphagrid_np
        )

    # Convert back to torch
    targetalphas_all = torch.tensor(
        np.exp(log_target_alphas_all) - 1, device=device, dtype=X.dtype
    )  # (n_fracs, n_targets)

    # Replace any NaN/inf with zero (will give OLS solution for those voxels)
    if torch.isnan(targetalphas_all).any() or torch.isinf(targetalphas_all).any():
        targetalphas_all = torch.nan_to_num(targetalphas_all, nan=0.0, posinf=1e10, neginf=0.0)

    # Compute scaling factors for all targets: (n_valid_rank, n_fracs, n_targets)
    # S_sq: (n_valid_rank,), targetalphas_all: (n_fracs, n_targets)
    sc_all = S_sq.unsqueeze(1).unsqueeze(2) / (
        S_sq.unsqueeze(1).unsqueeze(2) + targetalphas_all.unsqueeze(0)
    )  # (n_valid_rank, n_fracs, n_targets)

    # Apply scaling to OLS coefficients in rotated space
    # ols_coef_rotated: (n_valid_rank, n_targets)
    # sc_all: (n_valid_rank, n_fracs, n_targets)
    ridge_coef_rotated_all = sc_all * ols_coef_rotated.unsqueeze(1)

    # Unrotate: beta = V @ ridge_coef_rotated for all targets/fracs at once
    # Vt.T: (n_features, n_valid_rank), ridge_coef_rotated_all: (n_valid_rank, n_fracs, n_targets)
    # Reshape to (n_valid_rank, n_fracs * n_targets) for batch matmul
    ridge_flat = ridge_coef_rotated_all.reshape(n_valid_rank, n_fracs * n_targets)
    coefs_flat = Vt.T @ ridge_flat  # (n_features, n_fracs * n_targets)
    coefs = coefs_flat.reshape(n_features, n_fracs, n_targets)

    return coefs


@dataclass
class RidgeResults:
    """Results from ridge regression GLM fitting

    Attributes
    ----------
    betas_single_trial : torch.Tensor, shape (n_voxels, n_trials)
        Single-trial beta estimates at optimal ridge fraction
    r2 : torch.Tensor, shape (n_voxels,)
        R² for each voxel at optimal fraction (in-sample)
    r2_initial : torch.Tensor, shape (n_voxels,)
        Initial R² without ridge regularization (OLS)
    xval_r2 : torch.Tensor, shape (n_voxels,)
        Cross-validated R² at optimal fraction
    optimal_fracs : torch.Tensor, shape (n_voxels,)
        Optimal ridge fraction per voxel (selected via CV)
    r2_by_frac : torch.Tensor, shape (n_voxels, n_fracs)
        Cross-validated R² for each ridge fraction
    trial_labels : List[str]
        Labels for each trial (condition + trial number)
    metadata : Dict
        Processing metadata
    """

    betas_single_trial: torch.Tensor
    r2: torch.Tensor
    r2_initial: torch.Tensor
    xval_r2: torch.Tensor
    optimal_fracs: torch.Tensor
    r2_by_frac: torch.Tensor
    trial_labels: List[str]
    metadata: Dict


def create_single_trial_design(
    onsets_by_condition: List[List[np.ndarray]],
    durations: List[float],
    run_starts: List[int],
    tr: float,
    n_timepoints: int,
    hrf_library: Optional[List[torch.Tensor]] = None,
    hrf_index_per_voxel: Optional[torch.Tensor] = None,
    microtime_dt: float = 0.1,
    condition_labels: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    hrf_model_name: str = "spmg1",
    n_basis: int = 1,
) -> Tuple[torch.Tensor, List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create single-trial design matrix with optional per-voxel HRFs

    Each trial (event) gets its own regressor, allowing estimation of
    trial-specific beta weights. This is the core of GLMsingle-style analysis.

    Parameters
    ----------
    onsets_by_condition : list of list of np.ndarray
        Onsets organized as [condition][run] -> np.ndarray of onset times (seconds)
    durations : list of float
        Duration in seconds for each condition
    run_starts : list of int
        Starting timepoint index for each run
    tr : float
        Repetition time in seconds
    n_timepoints : int
        Total number of timepoints
    hrf_library : list of torch.Tensor, optional
        Library of HRF shapes. Each tensor has shape (hrf_length,).
        If None, uses canonical HRF.
    hrf_index_per_voxel : torch.Tensor, optional
        HRF index for each voxel (0-indexed). Shape (n_voxels,).
        If provided, creates per-voxel design matrices.
    microtime_dt : float, default=0.1
        Microtime resolution in seconds for non-TR-locked onsets
    condition_labels : list of str, optional
        Labels for each condition (for trial naming)
    device : torch.device, optional
        Device for computation
    hrf_model_name : str, default="spmg1"
        HRF model name (e.g., "spmg1", "spmg2", "spmg3", "glmsingle")
        Used to determine which HRF to use
    n_basis : int, default=1
        Number of basis functions per trial:
        - 1: Single HRF (SPMG1, glmsingle)
        - 2: Canonical + temporal derivative (SPMG2)
        - 3: Canonical + time + dispersion derivatives (SPMG3)

    Returns
    -------
    design_matrix : torch.Tensor
        If hrf_index_per_voxel is None: (n_timepoints, n_trials * n_basis)
        If hrf_index_per_voxel provided: (n_unique_hrfs, n_timepoints, n_trials * n_basis)
        For SPMG2/3, columns are interleaved: trial1_canonical, trial1_timederiv, ..., trial2_canonical, ...
    trial_labels : list of str
        Label for each column (e.g., "face_001_canonical", "face_001_timederiv")
        Length: n_trials * n_basis
    trial_condition_ids : torch.Tensor, shape (n_trials * n_basis,)
        Condition index (0-indexed) for each column
    trial_run_ids : torch.Tensor, shape (n_trials * n_basis,)
        Run index (0-indexed) for each column
    condition_design : torch.Tensor, shape (n_timepoints, n_conditions * n_basis)
        Condition-level design (sum of trials per condition, with basis functions)
    """
    from .design import convolve_hrf_microtime
    from .hrf import get_canonical_hrf

    device = device or torch.device("cpu")
    n_conditions = len(onsets_by_condition)

    if condition_labels is None:
        condition_labels = [f"cond{i+1:02d}" for i in range(n_conditions)]

    # Count total trials across all conditions and runs
    total_trials = 0
    trial_info = []  # (condition_idx, run_idx, trial_idx_in_run, onset_time)

    for cond_idx, runs_onsets in enumerate(onsets_by_condition):
        for run_idx, run_onsets in enumerate(runs_onsets):
            for trial_idx, onset_time in enumerate(run_onsets):
                trial_info.append((cond_idx, run_idx, trial_idx, onset_time))
                total_trials += 1

    # Generate trial labels and condition IDs
    # Track repeat number per condition (how many times we've seen each condition)
    trial_labels = []
    trial_condition_ids = []
    repeat_counter = {}  # condition_idx -> current repeat number

    for cond_idx, run_idx, trial_idx, onset_time in trial_info:
        # Get the repeat number for this condition (starting from 1)
        if cond_idx not in repeat_counter:
            repeat_counter[cond_idx] = 0
        repeat_counter[cond_idx] += 1
        repeat_num = repeat_counter[cond_idx]

        label = f"{condition_labels[cond_idx]}_{repeat_num:03d}"
        trial_labels.append(label)
        trial_condition_ids.append(cond_idx)
    trial_condition_ids = torch.tensor(trial_condition_ids, dtype=torch.long, device=device)

    # Extract run IDs for each trial
    trial_run_ids = torch.tensor(
        [run_idx for _, run_idx, _, _ in trial_info],
        dtype=torch.long, device=device,
    )

    # Create onset matrix at microtime resolution
    bins_per_tr = int(round(tr / microtime_dt))
    n_microtime = n_timepoints * bins_per_tr

    # Build single-trial onset matrix (microtime)
    onset_matrix_micro = torch.zeros(
        n_microtime, total_trials, dtype=torch.float32, device=device
    )

    for trial_idx, (cond_idx, run_idx, trial_in_run, onset_time) in enumerate(trial_info):
        # Convert onset time to microtime bin
        run_start_time = run_starts[run_idx] * tr
        onset_relative = onset_time + run_start_time
        onset_bin = int(round(onset_relative / microtime_dt))

        # Duration in microtime bins
        duration_bins = int(round(durations[cond_idx] / microtime_dt))

        # Set boxcar (handle edge cases)
        start_bin = max(0, onset_bin)
        if duration_bins == 0:
            # Instantaneous event: set single bin
            onset_matrix_micro[start_bin, trial_idx] = 1.0
        else:
            # Boxcar event: set range
            end_bin = min(n_microtime, onset_bin + duration_bins)
            if end_bin > start_bin:
                onset_matrix_micro[start_bin:end_bin, trial_idx] = 1.0

    # Apply HRF convolution (with optional derivatives for SPMG2/SPMG3)
    if hrf_index_per_voxel is None:
        # Single design matrix for all voxels
        is_spm_deriv = hrf_model_name.upper() in ("SPMG2", "SPMG3")

        if is_spm_deriv:
            # SPMG2/SPMG3: Use SPM canonical with derivatives
            from .hrf import get_spm_hrf_with_derivatives

            hrf_set = get_spm_hrf_with_derivatives(
                microtime_dt=microtime_dt,
                hrf_duration=32.0,
                n_basis=n_basis,
                device=device,
            )  # Shape: (n_basis, hrf_length)

            # Convolve each basis function with onset matrix
            designs_per_basis = []
            for basis_idx in range(n_basis):
                hrf_basis = hrf_set[basis_idx]
                design_basis = convolve_hrf_microtime(
                    onset_matrix_micro,
                    hrf_basis,
                    n_timepoints=n_timepoints,
                    tr=tr,
                    microtime_dt=microtime_dt,
                    device=device,
                    return_single_trials=False,
                )  # (n_timepoints, n_trials)
                designs_per_basis.append(design_basis)

            # Interleave columns: trial1_canonical, trial1_timederiv, ..., trial2_canonical, ...
            design_columns = []
            for trial_idx in range(total_trials):
                for basis_idx in range(n_basis):
                    design_columns.append(designs_per_basis[basis_idx][:, trial_idx:trial_idx+1])

            design_matrix = torch.cat(design_columns, dim=1)  # (n_timepoints, n_trials * n_basis)

            # Expand trial labels with basis suffixes
            basis_suffixes = {
                2: ["_canonical", "_timederiv"],
                3: ["_canonical", "_timederiv", "_dispderiv"],
            }
            suffixes = basis_suffixes[n_basis]

            expanded_trial_labels = []
            expanded_condition_ids = []
            expanded_run_ids = []
            for trial_idx, label in enumerate(trial_labels):
                for suffix in suffixes:
                    expanded_trial_labels.append(label + suffix)
                    expanded_condition_ids.append(trial_condition_ids[trial_idx].item())
                    expanded_run_ids.append(trial_run_ids[trial_idx].item())

            trial_labels = expanded_trial_labels
            trial_condition_ids = torch.tensor(expanded_condition_ids, dtype=torch.long, device=device)
            trial_run_ids = torch.tensor(expanded_run_ids, dtype=torch.long, device=device)

            # Build condition_design: sum trials within each condition, for each basis
            condition_design = torch.zeros(n_timepoints, n_conditions * n_basis, dtype=torch.float32, device=device)
            for cond_idx in range(n_conditions):
                for basis_idx in range(n_basis):
                    # Find columns for this condition and basis
                    # Columns are interleaved, so condition X basis Y is at positions: X*n_basis*trials_per_cond + trial*n_basis + Y
                    # Actually simpler: check trial_condition_ids and trial label suffix
                    col_idx_out = cond_idx * n_basis + basis_idx
                    suffix = suffixes[basis_idx]

                    # Sum all columns that match this condition and suffix
                    for col_idx, label in enumerate(trial_labels):
                        if label.endswith(suffix):
                            # Extract condition from label (before the _XXX suffix)
                            cond_label = condition_labels[cond_idx]
                            if label.startswith(cond_label + "_") and label.endswith(suffix):
                                condition_design[:, col_idx_out] += design_matrix[:, col_idx]

        else:
            # Standard single-basis (SPMG1, glmsingle)
            if hrf_library is None or len(hrf_library) == 0:
                # Use canonical HRF at microtime resolution (NOT TR resolution!)
                # convolve_hrf_microtime expects HRF at the same microtime_dt as the onset matrix
                hrf = get_canonical_hrf(
                    stim_duration=0.0, tr=microtime_dt, duration=32.0, device=device
                )
            else:
                # Use first HRF from library (should already be at microtime resolution)
                hrf = hrf_library[0].to(device)

            # Convolve - returns (n_timepoints, n_trials)
            design_matrix = convolve_hrf_microtime(
                onset_matrix_micro,
                hrf,
                n_timepoints=n_timepoints,
                tr=tr,
                microtime_dt=microtime_dt,
                device=device,
                return_single_trials=False,
            )

            # Build condition_design by summing trials within each condition
            condition_design = torch.zeros(n_timepoints, n_conditions, dtype=torch.float32, device=device)
            for cond_idx in range(n_conditions):
                cond_mask = trial_condition_ids == cond_idx
                if cond_mask.sum() > 0:
                    condition_design[:, cond_idx] = design_matrix[:, cond_mask].sum(dim=1)

        return design_matrix, trial_labels, trial_condition_ids, trial_run_ids, condition_design

    else:
        # Per-voxel design matrices
        n_voxels = hrf_index_per_voxel.shape[0]
        n_hrfs = len(hrf_library) if hrf_library is not None else 0

        if n_hrfs == 0:
            raise ValueError("hrf_library must be provided when using per-voxel HRFs")

        # Check incompatibility
        if n_basis > 1:
            raise ValueError(
                f"SPMG2/SPMG3 (n_basis={n_basis}) is incompatible with per-voxel HRFs. "
                "Use either SPM derivatives OR per-voxel HRF library, not both."
            )

        # Pre-compute convolved designs for each HRF
        designs_by_hrf = []
        for hrf_idx in range(n_hrfs):
            hrf = hrf_library[hrf_idx].to(device)
            design_tr = convolve_hrf_microtime(
                onset_matrix_micro,
                hrf,
                n_timepoints=n_timepoints,
                tr=tr,
                microtime_dt=microtime_dt,
                device=device,
                return_single_trials=False,
            )
            designs_by_hrf.append(design_tr)

        # Stack: (n_hrfs, n_timepoints, n_trials)
        designs_stacked = torch.stack(designs_by_hrf, dim=0)

        # Return per-HRF designs — NOT expanded to per-voxel!
        # Downstream code groups voxels by hrf_index and uses designs_stacked[hrf_idx]

        # Build condition_design using first HRF design (for CV prediction)
        first_design = designs_by_hrf[0]  # (n_timepoints, n_trials)
        condition_design = torch.zeros(n_timepoints, n_conditions, dtype=torch.float32, device=device)
        for cond_idx in range(n_conditions):
            cond_mask = trial_condition_ids == cond_idx
            if cond_mask.sum() > 0:
                condition_design[:, cond_idx] = first_design[:, cond_mask].sum(dim=1)

        return designs_stacked, trial_labels, trial_condition_ids, trial_run_ids, condition_design


def _fit_ridge_chunk(
    data_chunk: torch.Tensor,
    design_matrix: torch.Tensor,
    run_starts: List[int],
    nuisance_per_run: List[torch.Tensor],
    fracs: np.ndarray,
    cv_splits: List[Tuple[List[int], List[int]]],
    trial_condition_ids: torch.Tensor,
    condition_design: torch.Tensor,
    autoscale: bool,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Fit ridge regression for a chunk of voxels with a SINGLE design matrix

    Implements GLMsingle-style cross-validation:
    1. For each CV fold:
       - Project polynomials from train data and design
       - Fit single-trial betas on cleaned train data
       - Average betas within conditions to get condition-average betas
       - Project polynomials from test data
       - Predict test using CONDITION design × condition-average betas
       - Compute R² on cleaned test data
    2. Accumulate predictions across folds
    3. Compute R² for each fraction
    4. Select optimal fraction per voxel
    5. Refit on full cleaned data with optimal fraction

    Parameters
    ----------
    data_chunk : torch.Tensor, shape (chunk_voxels, n_timepoints)
        Data for this chunk
    design_matrix : torch.Tensor, shape (n_timepoints, n_trials)
        Single-trial design matrix (task only, no nuisance) - shared by all voxels
    run_starts : list of int
        Starting timepoint for each run
    nuisance_per_run : list of torch.Tensor
        Nuisance regressors per run (e.g., Legendre polynomials)
    fracs : np.ndarray
        Ridge fractions to test (1 = OLS, 0 = maximum regularization).
        Represents fraction of unregularized coefficient norm to retain.
    cv_splits : list of tuples
        CV splits as (train_runs, test_runs)
    trial_condition_ids : torch.Tensor, shape (n_trials,)
        Condition index (0-indexed) for each trial
    condition_design : torch.Tensor, shape (n_timepoints, n_conditions)
        Condition-level design matrix (sum of all trials per condition)
    autoscale : bool
        If True, apply GLMsingle-style post-hoc scaling to undo shrinkage bias.
        Fits scale+offset to match unregularized (OLS) beta distribution.
    device : torch.device
        Device for computation

    Returns
    -------
    results : dict
        - betas: (chunk_voxels, n_trials) final betas at optimal fraction
        - r2_initial: (chunk_voxels,) OLS R² (no ridge)
        - r2_final: (chunk_voxels,) R² at optimal fraction
        - xval_r2: (chunk_voxels,) cross-validated R²
        - optimal_fracs: (chunk_voxels,) optimal fraction per voxel
        - r2_by_frac: (chunk_voxels, n_fracs) R² for each fraction
        - r2_by_frac: (chunk_voxels, n_fracs) CV R² for each fraction
    """
    from .xval import project_out_nuisance_per_run

    chunk_voxels, n_timepoints = data_chunk.shape
    n_trials = design_matrix.shape[1]
    n_fracs = len(fracs)
    n_runs = len(run_starts)

    # Validate run_starts
    if len(run_starts) != n_runs:
        raise ValueError(
            f"run_starts has {len(run_starts)} elements but n_runs={n_runs}. "
            f"run_starts should have exactly n_runs elements (starting TR for each run), "
            f"not including the total timepoints."
        )

    # Compute run lengths using pattern from xval.py
    # run_starts contains starting timepoints for each run, e.g., [0, 120, 240] for 3 runs of 120 TRs
    run_lengths = np.diff(run_starts + [n_timepoints])

    # Cross-validation loop with proper polynomial projection
    predictions_by_frac = [
        torch.zeros(chunk_voxels, n_timepoints, device=device) for _ in range(n_fracs)
    ]
    actual_test_clean = torch.zeros(chunk_voxels, n_timepoints, device=device)

    fold_idx = 0
    for train_runs, test_runs in cv_splits:
        fold_idx += 1
        # ========================================================================
        # 1. Extract train data and build run_starts for train subset
        # ========================================================================
        train_tps = []
        train_run_starts_local = [0]  # Run starts relative to concatenated train data
        for run_idx in train_runs:
            start_tp = run_starts[run_idx]
            run_length = run_lengths[run_idx]
            train_tps.extend(range(start_tp, start_tp + run_length))
            train_run_starts_local.append(len(train_tps))
        train_run_starts_local = train_run_starts_local[:-1]  # Remove last (it's the total length)

        train_data = data_chunk[:, train_tps]  # (chunk_voxels, n_train_tps)
        train_design_raw = design_matrix[train_tps, :]  # (n_train_tps, n_trials)
        train_nuisance = [nuisance_per_run[i] for i in train_runs]

        # ========================================================================
        # 2. Project polynomials from train data and design
        # ========================================================================
        train_data_clean, train_design_clean = project_out_nuisance_per_run(
            train_data,
            train_design_raw,
            train_nuisance,
            train_run_starts_local,
            device=device,
        )

        # ========================================================================
        # 3. Fit ridge on cleaned train data
        # ========================================================================
        train_data_clean_t = train_data_clean.T  # (n_train_tps, chunk_voxels)
        coefs = _fit_ridge_multiple_fracs(train_design_clean, train_data_clean_t, fracs, device)

        # ========================================================================
        # 4. Extract test data and project polynomials
        # ========================================================================
        test_tps = []
        test_run_starts_local = [0]
        for run_idx in test_runs:
            start_tp = run_starts[run_idx]
            run_length = run_lengths[run_idx]
            test_tps.extend(range(start_tp, start_tp + run_length))
            test_run_starts_local.append(len(test_tps))
        test_run_starts_local = test_run_starts_local[:-1]

        test_data = data_chunk[:, test_tps]  # (chunk_voxels, n_test_tps)
        test_design_raw = design_matrix[test_tps, :]  # (n_test_tps, n_trials)
        test_nuisance = [nuisance_per_run[i] for i in test_runs]

        test_data_clean, test_design_clean = project_out_nuisance_per_run(
            test_data,
            test_design_raw,
            test_nuisance,
            test_run_starts_local,
            device=device,
        )

        # ========================================================================
        # 5. Average single-trial betas within conditions, predict with condition design
        # ========================================================================
        n_conditions = int(trial_condition_ids.max().item()) + 1

        # Get cleaned condition design for test timepoints
        test_cond_design_raw = condition_design[test_tps, :]  # (n_test_tps, n_conditions)
        # Project nuisance from condition design
        _, test_cond_design_clean = project_out_nuisance_per_run(
            test_data,  # dummy, we only need the design
            test_cond_design_raw,
            test_nuisance,
            test_run_starts_local,
            device=device,
        )

        # Process all fractions at once (vectorized condition averaging)
        # coefs: (n_trials, n_fracs, chunk_voxels)
        # Compute condition-average betas for all fractions simultaneously
        cond_betas_all = torch.zeros(n_conditions, n_fracs, chunk_voxels, device=device)

        for cond_idx in range(n_conditions):
            cond_mask = trial_condition_ids == cond_idx
            if cond_mask.sum() > 0:
                # Average across trials for this condition, for all fractions
                cond_betas_all[cond_idx, :, :] = coefs[cond_mask, :, :].mean(dim=0)

        # Predict for all fractions at once
        # test_cond_design_clean: (n_test_tps, n_conditions)
        # cond_betas_all: (n_conditions, n_fracs, chunk_voxels)
        # Result: (n_test_tps, n_fracs, chunk_voxels)
        y_pred_all = torch.einsum('tc,cfv->tfv', test_cond_design_clean, cond_betas_all)

        # Scatter predictions to output accumulators (vectorized)
        test_tps_tensor = torch.tensor(test_tps, device=device)
        for frac_idx in range(n_fracs):
            predictions_by_frac[frac_idx][:, test_tps_tensor] = y_pred_all[:, frac_idx, :].T

        # Store cleaned test data (vectorized)
        actual_test_clean[:, test_tps_tensor] = test_data_clean

    # Compute R² for each fraction (comparing cleaned data to predictions)
    r2_by_frac = torch.zeros(chunk_voxels, n_fracs, device=device)

    for frac_idx in range(n_fracs):
        pred = predictions_by_frac[frac_idx]
        r2_by_frac[:, frac_idx] = compute_r2_metric(actual_test_clean, pred, metric="cod")

    # Select optimal fraction per voxel (highest CV R²)
    xval_r2, best_frac_idx = r2_by_frac.max(dim=1)
    optimal_fracs = torch.from_numpy(fracs[best_frac_idx.cpu().numpy()]).to(device)

    # ========================================================================
    # Refit on full data (all runs) with optimal fraction per voxel
    # ========================================================================
    # Project polynomials from full data and design
    data_clean, design_clean = project_out_nuisance_per_run(
        data_chunk,
        design_matrix,
        nuisance_per_run,
        run_starts,
        device=device,
    )

    # Fit ridge for all fractions
    data_clean_t = data_clean.T  # (n_timepoints, chunk_voxels)
    coefs_final = _fit_ridge_multiple_fracs(design_clean, data_clean_t, fracs, device)

    # Extract trial betas at optimal fraction for each voxel (vectorized)
    # coefs_final: (n_trials, n_fracs, chunk_voxels)
    # best_frac_idx: (chunk_voxels,) indices into n_fracs dimension
    # Use advanced indexing to extract optimal betas for all voxels at once
    voxel_indices = torch.arange(chunk_voxels, device=device)
    betas_final = coefs_final[:, best_frac_idx, voxel_indices].T  # (chunk_voxels, n_trials)

    # ========================================================================
    # Apply GLMsingle-style autoscaling to undo shrinkage bias (vectorized)
    # ========================================================================
    if autoscale:
        # Find the index for frac=1.0 (pure OLS case)
        frac_dists = np.abs(fracs - 1.0)
        frac_ols_idx = int(np.argmin(frac_dists))

        # Extract unregularized betas for all voxels
        betas_ols = coefs_final[:, frac_ols_idx, :].T  # (chunk_voxels, n_trials)

        # Find voxels that need scaling (optimal frac < 0.99)
        needs_scaling = optimal_fracs < 0.99
        n_scale = int(needs_scaling.sum().item())

        if n_scale > 0:
            # Batch process all voxels that need scaling
            regularized_batch = betas_final[needs_scaling, :]  # (n_scale, n_trials)
            ols_batch = betas_ols[needs_scaling, :]  # (n_scale, n_trials)

            # Build design matrix for all voxels: [regularized, ones]
            # X_batch: (n_scale, n_trials, 2)
            ones = torch.ones(n_scale, n_trials, 1, dtype=regularized_batch.dtype, device=device)
            X_batch = torch.cat([regularized_batch.unsqueeze(2), ones], dim=2)

            # Solve batched OLS: [scale, offset] = (X'X)^{-1} X'y
            # XtX: (n_scale, 2, 2), Xty: (n_scale, 2)
            XtX = torch.bmm(X_batch.transpose(1, 2), X_batch)  # (n_scale, 2, 2)
            Xty = torch.bmm(X_batch.transpose(1, 2), ols_batch.unsqueeze(2)).squeeze(2)  # (n_scale, 2)

            # Add small regularization for stability
            XtX = XtX + 1e-10 * torch.eye(2, device=device).unsqueeze(0)

            # Solve all systems at once
            scale_offset_batch = torch.linalg.solve(XtX, Xty)  # (n_scale, 2)
            scales = scale_offset_batch[:, 0]  # (n_scale,)
            offsets = scale_offset_batch[:, 1]  # (n_scale,)

            # Revert to identity if scale is negative
            invalid_mask = scales < 0
            scales[invalid_mask] = 1.0
            offsets[invalid_mask] = 0.0

            # Apply transformations in batch
            betas_final[needs_scaling, :] = (
                scales.unsqueeze(1) * regularized_batch + offsets.unsqueeze(1)
            )

    # Compute final R² on cleaned data (using scaled betas)
    # Vectorized: y_pred_final[v, :] = design_clean @ betas_final[v, :]
    y_pred_final = betas_final @ design_clean.T  # (chunk_voxels, n_timepoints)

    r2_final = compute_r2_metric(data_clean, y_pred_final, metric="cod")

    # Compute initial R² (OLS or closest to it)
    # In fractional ridge: frac=1.0 is pure OLS, frac=0 is max regularization
    # Use largest fraction in array (if it's 1.0, this is actual OLS)
    frac_ols_idx = n_fracs - 1  # largest fraction
    r2_initial = r2_by_frac[:, frac_ols_idx]

    return {
        "betas": betas_final,
        "r2_initial": r2_initial,
        "r2_final": r2_final,
        "xval_r2": xval_r2,
        "optimal_fracs": optimal_fracs,
        "r2_by_frac": r2_by_frac,
    }


def _fit_ridge_chunk_with_per_voxel_designs(
    data_chunk: torch.Tensor,
    design_per_voxel: List[torch.Tensor],
    run_starts: List[int],
    nuisance_per_run: List[torch.Tensor],
    fracs: np.ndarray,
    cv_splits: List[Tuple[List[int], List[int]]],
    trial_condition_ids: torch.Tensor,
    condition_design: torch.Tensor,
    autoscale: bool,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Fit ridge regression for a chunk with per-voxel design matrices

    Groups voxels by unique design matrices and processes each group in batch.
    This is efficient when many voxels share the same HRF/design.

    Parameters: same as _fit_ridge_chunk, except design_per_voxel is a list
    """
    chunk_voxels = data_chunk.shape[0]
    n_trials = design_per_voxel[0].shape[1]
    n_fracs = len(fracs)

    # Group voxels by unique designs
    # Hash each design to find duplicates (vectorized - single GPU→CPU transfer)
    # Stack designs to compute hashes all at once
    # design_per_voxel is a list of (n_timepoints, n_trials) tensors
    designs_stacked = torch.stack(design_per_voxel, dim=0)  # (chunk_voxels, n_timepoints, n_trials)
    design_hashes = designs_stacked.sum(dim=(1, 2)).tolist()  # (chunk_voxels,) → list

    # Find unique design hashes and group voxels
    unique_hashes = list(set(design_hashes))
    groups = {}
    for hash_val in unique_hashes:
        groups[hash_val] = [i for i, h in enumerate(design_hashes) if h == hash_val]

    # Allocate output arrays
    betas_all = torch.zeros(chunk_voxels, n_trials, device=device)
    r2_initial_all = torch.zeros(chunk_voxels, device=device)
    r2_final_all = torch.zeros(chunk_voxels, device=device)
    xval_r2_all = torch.zeros(chunk_voxels, device=device)
    optimal_fracs_all = torch.zeros(chunk_voxels, device=device)
    r2_by_frac_all = torch.zeros(chunk_voxels, n_fracs, device=device)

    # Process each group
    for hash_val, voxel_indices in groups.items():
        # Get data and design for this group
        group_data = data_chunk[voxel_indices, :]
        group_design = design_per_voxel[voxel_indices[0]]  # All have same design

        # Call single-design function
        group_results = _fit_ridge_chunk(
            group_data,
            group_design,
            run_starts,
            nuisance_per_run,
            fracs,
            cv_splits,
            trial_condition_ids,
            condition_design,
            autoscale,
            device,
        )

        # Scatter results back to output arrays (vectorized)
        idx = torch.tensor(voxel_indices, device=betas_all.device)
        betas_all[idx] = group_results["betas"]
        r2_initial_all[idx] = group_results["r2_initial"]
        r2_final_all[idx] = group_results["r2_final"]
        xval_r2_all[idx] = group_results["xval_r2"]
        optimal_fracs_all[idx] = group_results["optimal_fracs"]
        r2_by_frac_all[idx] = group_results["r2_by_frac"]

    return {
        "betas": betas_all,
        "r2_initial": r2_initial_all,
        "r2_final": r2_final_all,
        "xval_r2": xval_r2_all,
        "optimal_fracs": optimal_fracs_all,
        "r2_by_frac": r2_by_frac_all,
    }


def fit_ridge_single_trial(
    data: torch.Tensor,
    design_matrix: Union[torch.Tensor, List[torch.Tensor]],
    run_starts: List[int],
    tr: float,
    trial_condition_ids: torch.Tensor,
    condition_design: torch.Tensor,
    fracs: Optional[np.ndarray] = None,
    nuisance: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    polort: Optional[int] = None,
    cv_splits: Optional[List[Tuple[List[int], List[int]]]] = None,
    trial_labels: Optional[List[str]] = None,
    autoscale: bool = True,
    chunk_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> RidgeResults:
    """
    Fit ridge regression with single-trial design using fracridge

    Uses cross-validation to select optimal ridge fraction per voxel.
    Supports per-voxel design matrices for HRF-specific regressors.

    Parameters
    ----------
    data : torch.Tensor, shape (n_voxels, n_timepoints)
        fMRI data
    design_matrix : torch.Tensor or list of torch.Tensor
        Single design: (n_timepoints, n_trials)
        Per-voxel designs: (n_voxels, n_timepoints, n_trials) for per-voxel HRFs
    run_starts : list of int
        Starting timepoint for each run
    tr : float
        Repetition time in seconds (for auto-determining polort)
    trial_condition_ids : torch.Tensor, shape (n_trials,)
        Condition index (0-indexed) for each trial. Used for CV prediction
        where single-trial betas are averaged within conditions.
    condition_design : torch.Tensor, shape (n_timepoints, n_conditions)
        Condition-level design matrix (sum of all trials per condition).
        Used for CV prediction of held-out runs.
    fracs : np.ndarray, optional
        Ridge fractions to test (1 = OLS, 0 = maximum regularization).
        Represents fraction of unregularized coefficient norm to retain.
        Default: np.arange(0.05, 1.05, 0.05) tests from heavy to light regularization.
    nuisance : list of torch.Tensor, optional
        Nuisance regressors per run (e.g., motion, noise PCs).
        Per-run format: list of (n_timepoints_run, n_nuisance)
        If None and polort is None, Legendre polynomials are added automatically.
    polort : int, optional
        Polynomial order for drift modeling. If None, auto-determined.
        Set to -1 to disable polynomial drift.
    cv_splits : list of tuples, optional
        Cross-validation splits as (train_runs, test_runs).
        Default: leave-one-run-out
    trial_labels : list of str, optional
        Labels for each trial
    autoscale : bool, default=True
        Apply GLMsingle-style post-hoc scaling to undo shrinkage bias.
        Fits a scale and offset transformation to match the unregularized (OLS)
        beta distribution. Recommended to keep True.
    chunk_size : int, default=10000
        Number of voxels to process at once
    device : torch.device, optional
        Device for computation
    verbose : bool, default=False
        Print progress

    Returns
    -------
    results : RidgeResults
        Ridge regression results with per-trial betas

    Notes
    -----
    This function implements the GLMsingle ridge regression approach:
    1. Build design matrix with trials + nuisance per run
    2. Cross-validate ridge fractions using LORO
    3. Select optimal fraction per voxel via CV R²
    4. Refit with optimal fraction for final beta estimates
    """
    from .glm_core import construct_polynomial_matrix

    device = device or torch.device("cpu")
    n_voxels, n_timepoints = data.shape
    n_runs = len(run_starts)

    # Default ridge fractions (similar to GLMsingle)
    if fracs is None:
        fracs = np.arange(0.05, 1.05, 0.05)
    n_fracs = len(fracs)

    # Generate CV splits if not provided (LORO by default)
    if cv_splits is None:
        from .xval import generate_cv_splits
        cv_splits = generate_cv_splits(n_runs, strategy=1, n_perms=n_runs)

    # Determine if we have per-voxel designs
    per_voxel_design = isinstance(design_matrix, torch.Tensor) and design_matrix.ndim == 3

    if per_voxel_design:
        # Check if this is per-voxel (n_voxels, n_tp, n_trials) or per-HRF (n_hrfs, n_tp, n_trials)
        # Per-HRF designs are returned by create_single_trial_design when hrf_library is provided
        if design_matrix.shape[0] == n_voxels:
            # Per-voxel designs (each voxel has its own design matrix)
            n_trials = design_matrix.shape[2]
            design_per_voxel_list = [design_matrix[i, :, :] for i in range(n_voxels)]
            if verbose:
                n_unique_hrfs = len(set(int(design.sum().item()) for design in design_per_voxel_list[:100]))
                print(f"  Per-voxel HRF designs detected (~{n_unique_hrfs} unique designs in first 100 voxels)")
        else:
            # Per-HRF designs (n_hrfs, n_tp, n_trials) - not yet supported directly
            # Users should expand per-HRF designs to per-voxel before calling fit_ridge_single_trial
            raise ValueError(
                f"Per-HRF design matrix detected with shape {design_matrix.shape}, but "
                f"fit_ridge_single_trial expects either a single design matrix (n_timepoints, n_trials) "
                f"or per-voxel designs (n_voxels, n_timepoints, n_trials). "
                f"When using create_single_trial_design with hrf_library, you must provide "
                f"hrf_index_per_voxel and expand the designs to per-voxel before calling this function. "
                f"See create_single_trial_design documentation for per-voxel HRF handling."
            )
    else:
        n_trials = design_matrix.shape[1]
        design_per_voxel_list = None  # Will use single design for all voxels

    # Calculate run lengths (needed for nuisance splitting regardless of polort)
    run_lengths = np.diff(run_starts + [n_timepoints])

    # Auto-determine polort if needed
    if polort is None and nuisance is None:
        # Calculate median run length using pattern from xval.py
        median_run_length = int(np.median(run_lengths))
        run_duration = median_run_length * tr
        polort = int(np.floor(1 + run_duration / 150.0))

    # Build nuisance regressors per run
    # Following GLMsingle: combine [polynomials, extra_regressors] like combinedmatrix
    nuisance_per_run = []

    # First, build polynomial regressors if needed
    poly_per_run = []
    if polort is not None and polort >= 0:
        for run_idx in range(n_runs):
            run_length = run_lengths[run_idx]
            poly = construct_polynomial_matrix(run_length, polort, device=device)
            poly_per_run.append(poly)

    # Second, get user-provided nuisance (e.g., noise PCs, motion, etc.)
    extra_per_run = []
    if nuisance is not None:
        if isinstance(nuisance, list):
            extra_per_run = nuisance
        else:
            # Split by runs
            for run_idx in range(n_runs):
                start_tp = run_starts[run_idx]
                run_length = run_lengths[run_idx]
                extra_per_run.append(nuisance[start_tp:start_tp + run_length, :])

    # Combine polynomials + extra regressors (like GLMsingle's combinedmatrix)
    for run_idx in range(n_runs):
        run_length = run_lengths[run_idx]

        regressors_for_run = []
        if poly_per_run:
            regressors_for_run.append(poly_per_run[run_idx])
        if extra_per_run:
            regressors_for_run.append(extra_per_run[run_idx])

        if regressors_for_run:
            combined = torch.cat(regressors_for_run, dim=1)
            nuisance_per_run.append(combined)
        else:
            # No nuisance at all
            nuisance_per_run.append(torch.zeros((run_length, 0), device=device))

    n_nuisance = nuisance_per_run[0].shape[1] if len(nuisance_per_run) > 0 else 0
    n_polys = poly_per_run[0].shape[1] if poly_per_run else 0
    n_extra = extra_per_run[0].shape[1] if extra_per_run else 0

    if verbose:
        print("\nRidge regression single-trial estimation")
        print(f"  Voxels: {n_voxels:,}")
        print(f"  Timepoints: {n_timepoints}")
        print(f"  Trials: {n_trials}")
        nuisance_desc = []
        if n_polys > 0:
            nuisance_desc.append(f"{n_polys} polys (polort={polort})")
        if n_extra > 0:
            nuisance_desc.append(f"{n_extra} extra")
        if nuisance_desc:
            print(f"  Nuisance per run: {n_nuisance} total ({', '.join(nuisance_desc)})")
        else:
            print("  Nuisance per run: none")
        print(f"  Ridge fractions: {n_fracs} ({fracs[0]:.2f} to {fracs[-1]:.2f})")
        print(f"  CV strategy: {len(cv_splits)} folds")
        print()

    # Determine chunk size (memory-aware if not specified)
    if chunk_size is None:
        if estimate_chunk_size is not None:
            # Determine n_regressors based on design type
            if per_voxel_design:
                # For per-voxel designs, use max trials across voxels
                n_regressors = n_trials
            else:
                n_regressors = design_matrix.shape[1]

            chunk_size = estimate_chunk_size(
                n_voxels=n_voxels,
                n_timepoints=n_timepoints,
                n_regressors=n_regressors,
                device=device,
                operation="ridge",
            )
            if verbose:
                print(f"  Auto-determined chunk size: {chunk_size:,} voxels")
        else:
            # Fallback to conservative default
            chunk_size = 10000
            if verbose:
                print(f"  Using default chunk size: {chunk_size:,} voxels")

    # Process in chunks
    n_chunks = (n_voxels + chunk_size - 1) // chunk_size

    # Output accumulators
    betas_single_trial = torch.zeros(n_voxels, n_trials, device="cpu")
    r2_initial = torch.zeros(n_voxels, device="cpu")
    r2_final = torch.zeros(n_voxels, device="cpu")
    xval_r2 = torch.zeros(n_voxels, device="cpu")
    optimal_fracs_per_voxel = torch.zeros(n_voxels, device="cpu")
    r2_by_frac = torch.zeros(n_voxels, n_fracs, device="cpu")

    if verbose:
        try:
            from tqdm import tqdm
            chunk_iter = tqdm(range(n_chunks), desc="Ridge fitting", unit="chunk")
        except ImportError:
            chunk_iter = range(n_chunks)
            print(f"Processing {n_chunks} chunks...")
    else:
        chunk_iter = range(n_chunks)

    for chunk_idx in chunk_iter:
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, n_voxels)

        # Get chunk data
        data_chunk = data[chunk_start:chunk_end, :].to(device)

        # Fit ridge for this chunk
        if per_voxel_design:
            # Per-voxel designs: extract designs for this chunk
            chunk_designs = design_per_voxel_list[chunk_start:chunk_end]
            chunk_results = _fit_ridge_chunk_with_per_voxel_designs(
                data_chunk=data_chunk,
                design_per_voxel=chunk_designs,
                run_starts=run_starts,
                nuisance_per_run=nuisance_per_run,
                fracs=fracs,
                cv_splits=cv_splits,
                trial_condition_ids=trial_condition_ids.to(device),
                condition_design=condition_design.to(device),
                autoscale=autoscale,
                device=device,
            )
        else:
            # Single design: use directly
            chunk_results = _fit_ridge_chunk(
                data_chunk=data_chunk,
                design_matrix=design_matrix,
                run_starts=run_starts,
                nuisance_per_run=nuisance_per_run,
                fracs=fracs,
                cv_splits=cv_splits,
                trial_condition_ids=trial_condition_ids.to(device),
                condition_design=condition_design.to(device),
                autoscale=autoscale,
                device=device,
            )

        # Store results
        betas_single_trial[chunk_start:chunk_end, :] = chunk_results["betas"].cpu()
        r2_initial[chunk_start:chunk_end] = chunk_results["r2_initial"].cpu()
        r2_final[chunk_start:chunk_end] = chunk_results["r2_final"].cpu()
        xval_r2[chunk_start:chunk_end] = chunk_results["xval_r2"].cpu()
        optimal_fracs_per_voxel[chunk_start:chunk_end] = chunk_results["optimal_fracs"].cpu()
        r2_by_frac[chunk_start:chunk_end, :] = chunk_results["r2_by_frac"].cpu()

    if verbose:
        print()
        print("Ridge regression complete")
        print(f"  Initial R² (OLS): {r2_initial.median():.4f} (median)")
        print(f"  Final R² (ridge): {r2_final.median():.4f} (median)")
        print(f"  CV R²: {xval_r2.median():.4f} (median)")
        print(f"  Improvement: {(r2_final.median() - r2_initial.median()):.4f}")
        print()

    # Build metadata
    metadata = {
        "n_voxels": n_voxels,
        "n_timepoints": n_timepoints,
        "n_trials": n_trials,
        "n_nuisance": n_nuisance,
        "polort": polort,
        "n_fracs": n_fracs,
        "fracs": fracs.tolist(),
        "cv_folds": len(cv_splits),
        "tr": tr,
    }

    return RidgeResults(
        betas_single_trial=betas_single_trial,
        r2=r2_final,
        r2_initial=r2_initial,
        xval_r2=xval_r2,
        optimal_fracs=optimal_fracs_per_voxel,
        r2_by_frac=r2_by_frac,
        trial_labels=trial_labels or [f"trial_{i:03d}" for i in range(n_trials)],
        metadata=metadata,
    )


def load_hrf_indices(hrf_index_file: str, mask: Optional[np.ndarray] = None) -> torch.Tensor:
    """
    Load HRF indices from HRFoptfast output

    Parameters
    ----------
    hrf_index_file : str
        Path to {prefix}_hrf_index.nii.gz from HRFoptfast
    mask : np.ndarray, optional
        Boolean mask to apply (if data was masked)

    Returns
    -------
    hrf_indices : torch.Tensor, shape (n_voxels,)
        HRF index per voxel (0-indexed, converted from 1-indexed NIFTI)
    """
    import nibabel as nib

    img = nib.load(hrf_index_file)
    hrf_data = img.get_fdata()

    # Convert from 1-indexed (NIFTI) to 0-indexed (Python)
    hrf_data = hrf_data - 1.0

    if mask is not None:
        hrf_data = hrf_data[mask]
    else:
        hrf_data = hrf_data.flatten()

    return torch.from_numpy(hrf_data).long()


def load_noise_pcs(noise_pc_file: str, run_starts: List[int], n_timepoints: int) -> List[torch.Tensor]:
    """
    Load noise PCs from Denoisefast output

    Parameters
    ----------
    noise_pc_file : str
        Path to {prefix}_noise_pcs.xmat.1D from Denoisefast
    run_starts : list of int
        Starting timepoint for each run
    n_timepoints : int
        Total number of timepoints

    Returns
    -------
    noise_pcs_per_run : list of torch.Tensor
        List of per-run noise PC matrices, each (n_timepoints_run, n_pcs)
    """
    # Load PC timecourses
    pcs = np.loadtxt(noise_pc_file)  # (n_timepoints, n_pcs)
    pcs_tensor = torch.from_numpy(pcs).float()

    # Split by run
    n_runs = len(run_starts)
    pcs_per_run = []

    # Compute run lengths using the same pattern as slice_by_runs
    run_lengths = np.diff(run_starts + [n_timepoints])

    for run_idx in range(n_runs):
        start_tp = run_starts[run_idx]
        run_length = run_lengths[run_idx]
        pcs_per_run.append(pcs_tensor[start_tp:start_tp + run_length, :])

    return pcs_per_run
