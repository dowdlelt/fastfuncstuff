"""
Cross-validated HRF selection per voxel

This module provides the core engine for selecting the optimal HRF per voxel
using cross-validation across runs. Unlike in-sample selection which can overfit,
CV-based selection provides a more reliable estimate of which HRF shape best
captures the true hemodynamic response for each voxel.

Key function: fit_glm_hrf_library_with_xval()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .design import convolve_hrf_microtime
from .glm_core import GLMResults, construct_polynomial_matrix, fit_glm
from .hrf import get_hrf_library
from .memory import dyn_chunk_estimator, estimate_chunk_size, estimate_keep_on_cpu
from .utils import get_device, to_tensor
from .xval import (
    compute_xval_r2,
    generate_cv_splits,
    project_out_nuisance_per_run,
)


def load_nuisance_file(
    filepath: str | Path,
    expected_rows: int | None = None,
) -> np.ndarray:
    """
    Load a nuisance regressor file (motion parameters, physio, etc.)

    Handles various text formats:
    - AFNI 1D files (whitespace-separated, may have # comment headers)
    - CSV files
    - Tab-separated files
    - Space-separated files

    Parameters
    ----------
    filepath : str or Path
        Path to the nuisance file
    expected_rows : int, optional
        Expected number of rows (timepoints). If provided, validates length.

    Returns
    -------
    data : np.ndarray
        (n_timepoints, n_columns) nuisance regressors
        Single-column files are reshaped to (n_timepoints, 1)

    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    ValueError
        If file is empty or has wrong number of rows

    Examples
    --------
    >>> # Load 6 motion parameters
    >>> motion = load_nuisance_file('motion.1D')
    >>> motion.shape
    (200, 6)

    >>> # Load with validation
    >>> motion = load_nuisance_file('motion.1D', expected_rows=200)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Nuisance file not found: {filepath}")

    # Read file content
    with open(filepath) as f:
        lines = f.readlines()

    # Filter out comment lines (AFNI 1D format uses # for comments/headers)
    data_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        data_lines.append(line)

    if not data_lines:
        raise ValueError(f"Nuisance file {filepath} is empty or contains only comments")

    # Detect delimiter: try comma first, then whitespace
    first_line = data_lines[0]
    if "," in first_line:
        delimiter = ","
    elif "\t" in first_line:
        delimiter = None  # np.loadtxt handles tabs with default
    else:
        delimiter = None  # whitespace

    # Parse data
    try:
        if delimiter == ",":
            data = np.array([[float(x.strip()) for x in line.split(",")] for line in data_lines])
        else:
            data = np.array([[float(x) for x in line.split()] for line in data_lines])
    except ValueError as e:
        raise ValueError(f"Error parsing nuisance file {filepath}: {e}")

    # Ensure 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Validate row count if expected
    if expected_rows is not None and data.shape[0] != expected_rows:
        raise ValueError(
            f"Nuisance file {filepath} has {data.shape[0]} rows, "
            f"expected {expected_rows} (total timepoints)"
        )

    return data.astype(np.float32)


@dataclass
class HRFSelectionResults:
    """Container for voxel-wise HRF selection results.

    Attributes
    ----------
    hrf_index : torch.Tensor
        (n_voxels,) Index of best HRF for each voxel (0 to n_hrfs-1)
    xval_r2_best : torch.Tensor
        (n_voxels,) Median cross-validated R² for the selected HRF
    xval_r2_std : torch.Tensor
        (n_voxels,) Std of cross-validated R² across CV splits for selected HRF
    xval_r2_all_hrfs : torch.Tensor
        (n_voxels, n_hrfs) Median CV R² for each HRF (for diagnostics)
    xval_r2_canonical : torch.Tensor
        (n_voxels,) Median CV R² using single canonical HRF (baseline comparison)
    final_results : GLMResults
        Results from final full-data fit with voxel-wise optimal HRFs
    canonical_results : GLMResults
        Results from full-data fit with single canonical HRF (baseline comparison)
        Contains betas, t-stats, etc. for the "what if we hadn't optimized" case.
    hrf_library : torch.Tensor
        (n_hrfs, n_hrf_timepoints) The HRF library used (stored for ARMA reuse)
    hrf_metadata : dict
        Metadata about HRF selection: mode, tr, stim_durations, cv_strategy, etc.
    """

    hrf_index: torch.Tensor = None
    xval_r2_best: torch.Tensor = None
    xval_r2_std: torch.Tensor = None
    xval_r2_all_hrfs: torch.Tensor = None
    xval_r2_canonical: torch.Tensor = None  # Baseline with single canonical HRF
    canonical_full_r2: torch.Tensor | None = None  # In-sample R², canonical HRF
    hrfopt_full_r2: torch.Tensor | None = None  # In-sample R², best HRF per voxel
    canonical_xval_r2: torch.Tensor | None = None  # Beta-series CV R², canonical HRF
    hrfopt_xval_r2: torch.Tensor | None = None  # Beta-series CV R², best HRF per voxel
    final_results: GLMResults = None
    canonical_results: GLMResults = None  # Full GLM with canonical HRF for comparison
    hrf_library: torch.Tensor = None
    hrf_metadata: dict = field(default_factory=dict)

    # For ARMA integration: store the convolved design per HRF group
    # This allows reloading without reconvolving
    hrf_group_indices: dict[int, torch.Tensor] = field(default_factory=dict)

    # Store design matrix (convolved with middle HRF) for debugging
    design_matrix: torch.Tensor | None = None

    # Store canonical HRF design matrix for comparison/saving
    canonical_design_matrix: torch.Tensor | None = None


def _project_design_with_q_factors(
    design: torch.Tensor,
    q_factors: list,
    run_starts: list[int],
    n_timepoints: int,
    n_runs: int,
    device: torch.device,
) -> torch.Tensor:
    """Project nuisance from design using pre-computed QR projectors.

    Tiny per-run matrix operation — no chunking needed.
    Returns projected design on CPU.
    """
    projected_runs = []
    for run_idx in range(n_runs):
        start_tp = run_starts[run_idx]
        end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
        run_design = design[start_tp:end_tp, :].to(device)
        if q_factors[run_idx] is not None:
            Q = q_factors[run_idx]
            run_design = run_design - Q @ (Q.T @ run_design)
        projected_runs.append(run_design.cpu())
    return torch.cat(projected_runs, dim=0)


def _evaluate_hrfs_batched(
    projected_data: torch.Tensor,
    projected_designs: list[torch.Tensor],
    run_starts: list[int],
    cv_splits: list[tuple[list[int], list[int]]],
    device: torch.device,
    metric: str = "cod",
    chunk_size: int | None = None,
    verbose: bool = True,
) -> torch.Tensor:
    """Evaluate multiple designs via CV in a single batched pass.

    Key optimization: data is moved to GPU once per voxel chunk per CV split,
    and ALL designs are evaluated on that same data. This reduces CPU-to-GPU
    transfers by len(designs)x compared to per-design evaluation.

    Two R² computation strategies (chosen automatically):

    **LORO** (each timepoint in test exactly once):
      Accumulate SS_res per fold → R² = 1 - SS_res_total / SS_tot_global.
      Exact and cheap (O(n_voxels × n_designs) memory).

    **Split-half / k-fold** (timepoints in multiple test folds):
      Accumulate predicted timeseries (sum_yhat) across folds → divide by
      per-timepoint count → compute R² from averaged predictions.
      Matches GLMdenoise/compute_xval_r2 "slow" path, avoids the
      prediction-variance bias that makes all HRFs look equally bad.
      Memory: O(n_designs × chunk × n_timepoints) per voxel chunk.

    Parameters
    ----------
    projected_data : torch.Tensor
        (n_voxels, n_timepoints) Projected fMRI data, on CPU.
    projected_designs : list of torch.Tensor
        Each (n_timepoints, n_stim_cols), on CPU.
    run_starts : list of int
        Starting timepoint for each run.
    cv_splits : list of (train_runs, test_runs)
        Any CV splits (LORO, split-half, k-fold, etc.).
    device : torch.device
        GPU device for computation.
    metric : str
        R² metric: 'cod', 'corr', or 'corr2'.
    chunk_size : int, optional
        Voxels per GPU chunk (auto if None).
    verbose : bool
        Print progress.

    Returns
    -------
    r2 : torch.Tensor
        (n_voxels, n_designs) CV R² for each design, on CPU.
    """
    n_voxels, n_timepoints = projected_data.shape
    n_designs = len(projected_designs)
    n_splits = len(cv_splits)
    data_on_device = projected_data.device == device or projected_data.device.type == device.type

    if not data_on_device and chunk_size is None:
        chunk_size = estimate_chunk_size(
            n_voxels=n_voxels,
            n_timepoints=n_timepoints,
            n_regressors=projected_designs[0].shape[1],
            device=device,
            operation="xval",
        )

    # Pre-slice data by runs (views, no copy)
    run_ends = run_starts[1:] + [n_timepoints]
    data_by_run = [projected_data[:, s:e] for s, e in zip(run_starts, run_ends)]

    # Pre-slice designs by runs
    designs_by_run: list[list[torch.Tensor]] = []
    for d_idx in range(n_designs):
        d = projected_designs[d_idx]
        designs_by_run.append([d[s:e, :] for s, e in zip(run_starts, run_ends)])

    # Detect LORO: each timepoint appears in test exactly once across all folds.
    # For LORO we use streaming SS_res accumulation (fast, O(n_voxels) accumulators).
    # For split-half / k-fold we use prediction averaging (correct but needs
    # O(n_designs x chunk x n_tp) accumulators per voxel chunk).
    n_test_per_tp = torch.zeros(n_timepoints, dtype=torch.long)
    for _, test_runs in cv_splits:
        for r in test_runs:
            n_test_per_tp[run_starts[r] : run_ends[r]] += 1
    is_loro_style = bool((n_test_per_tp == 1).all())

    # Global SS_tot from ALL timepoints -- used in both paths.
    if data_on_device:
        sum_y_all = projected_data.sum(dim=1).cpu()
        sum_y2_all = (projected_data**2).sum(dim=1).cpu()
    else:
        sum_y_all = projected_data.sum(dim=1)
        sum_y2_all = (projected_data**2).sum(dim=1)
    ss_tot_global = (sum_y2_all - sum_y_all**2 / n_timepoints).clamp(min=1e-10)

    # =========================================================================
    # Path A: LORO -- fold-outer, SS_res accumulation.
    # Works because each TP is tested exactly once, so sum_ss_res = SS_res_total.
    # =========================================================================
    if is_loro_style and metric == "cod":
        sum_ss_res = torch.zeros(n_voxels, n_designs)

        split_iter = tqdm(cv_splits, desc="CV splits") if verbose else cv_splits
        for train_runs, test_runs in split_iter:
            train_designs_gpu = [
                torch.cat([designs_by_run[d][r] for r in train_runs], dim=0).to(device)
                for d in range(n_designs)
            ]
            test_designs_gpu = [
                torch.cat([designs_by_run[d][r] for r in test_runs], dim=0).to(device)
                for d in range(n_designs)
            ]
            train_data_split = torch.cat([data_by_run[r] for r in train_runs], dim=1)
            test_data_split = torch.cat([data_by_run[r] for r in test_runs], dim=1)

            # pinv: SVD-based, handles rank-deficient designs (zero betas for
            # missing events), avoids NaN/Inf from CUDA gels driver.
            pinv_trains_gpu = [torch.linalg.pinv(td) for td in train_designs_gpu]

            if data_on_device:
                sum_y2_test_fold = (test_data_split**2).sum(dim=1)
                for d_idx in range(n_designs):
                    betas = pinv_trains_gpu[d_idx] @ train_data_split.T
                    yhat = (test_designs_gpu[d_idx] @ betas).T
                    ss_res = (
                        sum_y2_test_fold
                        - 2 * (test_data_split * yhat).sum(dim=1)
                        + (yhat**2).sum(dim=1)
                    )
                    sum_ss_res[:, d_idx] += ss_res.cpu()
            else:
                for cs in range(0, n_voxels, chunk_size):
                    ce = min(cs + chunk_size, n_voxels)
                    train_chunk = train_data_split[cs:ce].to(device)
                    test_chunk = test_data_split[cs:ce].to(device)
                    sum_y2_chunk = (test_chunk**2).sum(dim=1)
                    for d_idx in range(n_designs):
                        betas = pinv_trains_gpu[d_idx] @ train_chunk.T
                        yhat = (test_designs_gpu[d_idx] @ betas).T
                        ss_res = (
                            sum_y2_chunk - 2 * (test_chunk * yhat).sum(dim=1) + (yhat**2).sum(dim=1)
                        )
                        sum_ss_res[cs:ce, d_idx] += ss_res.cpu()
                    del train_chunk, test_chunk

            del (
                train_data_split,
                test_data_split,
                train_designs_gpu,
                test_designs_gpu,
                pinv_trains_gpu,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

        return 1.0 - sum_ss_res / ss_tot_global.unsqueeze(1)

    # --- Path B follows (guard clause above returned for LORO) ---------------
    # This is an intentional early-return guard rather than a giant else-block.
    # =========================================================================
    # Path B: Split-half / k-fold -- prediction averaging (chunk-outer).
    #
    # For split-half each timepoint appears in ~n_splits/2 test folds.
    # Averaging predictions across folds before computing R² removes the
    # between-fold beta variance, which otherwise swamps the small R² differences
    # between neighbouring HRFs and makes all curves look flat/negative.
    #
    # Precompute all fold x design pseudoinverses (tiny matrices, ~30-100 MB)
    # so the inner chunk loop only does cheap matmuls.
    # =========================================================================

    # Per-timepoint test count (for averaging).
    count_per_tp = n_test_per_tp.float()

    # Warn when n_train < n_test (training set smaller than test set).
    # Having more test data than training data is an unusual split that increases
    # beta variance per fold.  For HRF shapes far from the true BOLD response (e.g.
    # slow purely-positive HRFs at the library extremes whose design columns lose
    # energy to polynomial projection, or early-peak HRFs mismatched to the true
    # response), a handful of folds can produce sign-flipped betas that dominate the
    # prediction average → very negative R² for those specific HRFs.  LORO is immune
    # because each TP is tested once (Path A: SS_res accumulation).
    #
    # TODO: investigate further — empirically observed with 9 runs / strategy=0.5
    # (HRFs 4 and 17-19, 0-indexed) but NOT for HRFs 0-3 (even earlier peaks).
    # Mechanism: fewer training runs → lower effective SNR for wrong-HRF designs →
    # sign-flipping OLS betas in specific folds → averaged prediction anti-correlated
    # with data.  Worth testing with other datasets to characterise the threshold.
    # Rule of thumb that has worked: cv_strategy >= 5/n_runs (e.g. 0.6 for 9 runs).
    n_train_per_fold = len(cv_splits[0][0])
    n_test_per_fold = len(cv_splits[0][1])
    n_total_runs = n_train_per_fold + n_test_per_fold
    if n_train_per_fold < n_test_per_fold:
        import warnings

        rec_frac = n_test_per_fold / n_total_runs
        msg = (
            f"Split-half CV: {n_train_per_fold} training runs < {n_test_per_fold} test runs per fold. "
            f"This can produce unstable beta estimates for HRF shapes far from the "
            f"true BOLD response, causing very negative R² for those HRFs. "
            f"Consider using cv_strategy >= {rec_frac:.2f} so training >= test runs."
        )
        if verbose:
            print(f"  Warning: {msg}")
        warnings.warn(msg, stacklevel=4)

    if verbose:
        print(f"  Precomputing {n_splits} x {n_designs} pseudoinverses for split-half CV...")
    all_pinvs: list[list[torch.Tensor]] = []
    for fold_idx, (train_runs, _) in enumerate(cv_splits):
        pinvs_fold = []
        for d_idx in range(n_designs):
            X_train = torch.cat([designs_by_run[d_idx][r] for r in train_runs], dim=0).to(device)
            pinv_X = torch.linalg.pinv(X_train)
            pinvs_fold.append(pinv_X)
        all_pinvs.append(pinvs_fold)

    # Compute chunk_size that keeps sum_yhat (n_designs x chunk x n_tp) in GPU memory.
    if device.type == "cuda":
        try:
            free_mem = torch.cuda.mem_get_info()[0]
            bytes_per_vox = n_designs * n_timepoints * 4
            split_chunk = max(100, int(free_mem * 0.4 / bytes_per_vox))
        except Exception:
            split_chunk = chunk_size or n_voxels
        chunk_size = min(chunk_size or n_voxels, split_chunk)
    elif chunk_size is None:
        chunk_size = n_voxels

    r2_out = torch.zeros(n_voxels, n_designs)

    chunk_iter = range(0, n_voxels, chunk_size)
    if verbose:
        chunk_iter = tqdm(list(chunk_iter), desc="Voxel chunks")

    for cs in chunk_iter:
        ce = min(cs + chunk_size, n_voxels)
        chunk_len = ce - cs

        # Prediction accumulator: (n_designs, chunk, n_tp) on device.
        sum_yhat = torch.zeros(n_designs, chunk_len, n_timepoints, device=device)

        for fold_idx, (train_runs, test_runs) in enumerate(cv_splits):
            train_chunk = torch.cat([data_by_run[r][cs:ce] for r in train_runs], dim=1)
            if not data_on_device:
                train_chunk = train_chunk.to(device)

            for d_idx in range(n_designs):
                betas = all_pinvs[fold_idx][d_idx] @ train_chunk.T  # (n_stim, chunk)

                # Accumulate per test run using contiguous slices.
                for r in test_runs:
                    r_start, r_end = run_starts[r], run_ends[r]
                    test_design_r = designs_by_run[d_idx][r].to(device)
                    yhat_r = (test_design_r @ betas).T  # (chunk, T_r)
                    sum_yhat[d_idx, :, r_start:r_end] += yhat_r

            del train_chunk

        # R2 from averaged predictions.
        data_chunk = projected_data[cs:ce]
        if not data_on_device:
            data_chunk = data_chunk.to(device)
        count_d = count_per_tp.to(device)
        ss_tot_chunk = ss_tot_global[cs:ce].to(device)

        for d_idx in range(n_designs):
            avg_yhat_d = sum_yhat[d_idx] / count_d  # (chunk, n_tp)
            ss_res = ((data_chunk - avg_yhat_d) ** 2).sum(dim=1)
            r2_out[cs:ce, d_idx] = (1.0 - ss_res / ss_tot_chunk).cpu()

        del sum_yhat, data_chunk
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del all_pinvs
    return r2_out


def _evaluate_hrfs_batched_loro(
    projected_data: torch.Tensor,
    projected_designs: list[torch.Tensor],
    run_starts: list[int],
    cv_splits: list[tuple[list[int], list[int]]],
    device: torch.device,
    metric: str = "cod",
    chunk_size: int | None = None,
    verbose: bool = True,
) -> torch.Tensor:
    """Backward-compatible alias for _evaluate_hrfs_batched."""
    return _evaluate_hrfs_batched(
        projected_data=projected_data,
        projected_designs=projected_designs,
        run_starts=run_starts,
        cv_splits=cv_splits,
        device=device,
        metric=metric,
        chunk_size=chunk_size,
        verbose=verbose,
    )


def fit_glm_hrf_library_with_xval(
    data: torch.Tensor,
    onsets: torch.Tensor,
    hrf_library: torch.Tensor,
    tr: float,
    run_starts: list[int],
    stim_durations: list[float] | None = None,
    cv_strategy: float | int = 1,
    n_perms: int = 100,
    metric: str = "cod",
    microtime_dt: float = 0.1,
    microtime_onset: int = 0,
    polort: int | None = None,
    ortvec_files: list[tuple[str | Path, str]] | None = None,
    extra_regressors: np.ndarray | torch.Tensor | None = None,
    canonical_mode: str = "spmg1",
    device: torch.device | None = None,
    verbose: bool = True,
    chunk_size: int | None = None,
    r2_method: str = "auto",
    debug: bool = False,
    debug_prefix: str | None = None,
    condition_labels: list[str] | None = None,
) -> HRFSelectionResults:
    """
    Select best HRF per voxel using cross-validated R².

    This function:
    1. Loops through each HRF in the library
    2. For each HRF, computes cross-validated R² across CV splits
    3. Selects the HRF with highest median CV R² per voxel
    4. Refits the full dataset using voxel-wise optimal HRFs

    Parameters
    ----------
    data : torch.Tensor
        (n_voxels, n_timepoints) fMRI data
    onsets : torch.Tensor
        (n_microtime_points, n_conditions) binary onset matrix at microtime_dt resolution.
        n_microtime_points = n_timepoints * (tr / microtime_dt)
    hrf_library : torch.Tensor
        (n_hrfs, n_hrf_timepoints) Library of HRF candidates at TR resolution
    tr : float
        Repetition time in seconds
    run_starts : list of int
        Starting timepoint index for each run (required for CV splits)
    stim_durations : list of float, optional
        Duration in seconds for each condition. If None, assumes impulse (0s).
        Single value applies to all conditions.
    cv_strategy : float or int, default=1
        Cross-validation strategy:
        - int: Leave-N-out (1 = LORO, 2 = leave-2-out)
        - float: Split fraction (0.5 = split halves)
    n_perms : int, default=100
        Number of permutations for random split strategies
    metric : str, default='cod'
        R² metric: 'cod' (coefficient of determination), 'corr', or 'corr2'
    microtime_dt : float, default=0.1
        Microtime resolution in seconds. Default 0.1s is the standard throughout
        the pipeline. Both onsets and HRF library should be at this resolution.
    microtime_onset : int, default=0
        Which microtime bin within each TR to sample (0-indexed).
        0 = start of TR, bins_per_tr/2 = middle of TR.
    polort : int, optional
        Polynomial order for detrending (None = auto)
    ortvec_files : list of (filepath, label) tuples, optional
        Additional nuisance regressors to project out (like AFNI's -ortvec).
        Each file should contain already-concatenated regressors spanning
        all runs (same length as total timepoints). Files can be:
        - AFNI 1D format (whitespace-separated, # comments allowed)
        - CSV files
        - Tab or space-separated text files
        Example: [('motion_all.1D', 'motion'), ('physio.txt', 'physio')]
    extra_regressors : np.ndarray or torch.Tensor, optional
        Additional nuisance regressors as a matrix (n_timepoints, n_columns).
        Alternative to ortvec_files for passing already-loaded data.
        Must span all runs (already concatenated).
    canonical_mode : str, default='spmg1'
        Which canonical HRF to use for baseline comparison:
        - 'spmg1' or 'SPMG1': AFNI's SPMG1 formula (recommended default)
        - 'glmsingle': GLMsingle/nilearn-style double-gamma (scipy.stats.gamma)
        The baseline comparison shows how much HRF optimization improves over
        using a single canonical HRF for all voxels.
    device : torch.device, optional
        Compute device (auto-detected if None)
    verbose : bool, default=True
        Print progress information
    chunk_size : int, optional
        Number of voxels to process at once (auto if None)

    Returns
    -------
    HRFSelectionResults
        Container with:
        - hrf_index: (n_voxels,) best HRF per voxel
        - xval_r2_best: (n_voxels,) CV R² for selected HRF
        - xval_r2_std: (n_voxels,) std across CV splits
        - xval_r2_all_hrfs: (n_voxels, n_hrfs) CV R² for all HRFs
        - final_results: GLMResults from full refit
        - hrf_library: The HRF library used (for ARMA reuse)
        - hrf_metadata: Selection parameters

    Notes
    -----
    The final fit groups voxels by their selected HRF to maintain GPU efficiency.
    This is similar to how 3dREMLfast handles voxel-wise ARMA parameters.
    """
    if device is None:
        device = get_device()

    # Data always stays on CPU — CLI ensures this via force_cpu=True.
    # GPU is used only for computation, with data streamed in chunks.
    data = to_tensor(data, device="cpu")
    onsets = to_tensor(onsets, device=device)  # Small - can go to GPU
    hrf_library = to_tensor(hrf_library, device=device)  # Small - can go to GPU

    n_voxels, n_timepoints_data = data.shape
    n_hrfs = hrf_library.shape[0]
    n_runs = len(run_starts)

    # Calculate bins per TR for microtime
    bins_per_tr = int(round(tr / microtime_dt))

    # Determine n_timepoints at TR resolution
    n_timepoints = onsets.shape[0] // bins_per_tr

    # Validate data/design alignment
    if n_timepoints_data != n_timepoints:
        raise ValueError(
            f"Data has {n_timepoints_data} timepoints but design implies {n_timepoints}. "
            f"Check microtime_dt and tr settings (bins_per_tr={bins_per_tr})."
        )

    if verbose:
        print("=" * 70)
        print("CROSS-VALIDATED HRF SELECTION")
        print("=" * 70)
        print(f"  Voxels: {n_voxels:,}")
        print(f"  Timepoints: {n_timepoints}")
        print(f"  Runs: {n_runs}")
        print(f"  HRF candidates: {n_hrfs}")
        print(
            f"  CV strategy: {cv_strategy} ({'LORO' if cv_strategy == 1 else 'split-halves' if cv_strategy == 0.5 else cv_strategy})"
        )
        print(
            f"  Microtime: dt={microtime_dt}s ({bins_per_tr} bins/TR, onset bin {microtime_onset})"
        )
        data_size_gb = (n_voxels * n_timepoints * 4) / (1024**3)
        print(f"  Data: {data_size_gb:.2f} GB on CPU, compute on {device}")
        print()

    # Generate CV splits
    cv_splits = generate_cv_splits(n_runs, strategy=cv_strategy, n_perms=n_perms)
    n_splits = len(cv_splits)

    if verbose:
        print(f"  CV splits: {n_splits}")
        print()

    # =========================================================================
    # Build per-run nuisance blocks using shared utility
    # =========================================================================
    from .cli_utils import (
        auto_polort,
        build_nuisance_per_run,
        compute_run_lengths,
        get_average_run_duration,
    )

    # Auto-compute polort if not specified (AFNI formula)
    if polort is None:
        run_lengths = compute_run_lengths(run_starts, n_timepoints)
        avg_run_duration_sec = get_average_run_duration(run_lengths, tr)
        polort = auto_polort(avg_run_duration_sec, formula="afni")
        if verbose:
            print(f"  Auto polort: {polort} (AFNI formula, {avg_run_duration_sec:.0f}s avg run)")

    # Convert extra_regressors to tensor for ortvec_data parameter
    ortvec_data = None
    extra_nuisance_labels: list[str] = []
    if extra_regressors is not None:
        if isinstance(extra_regressors, np.ndarray):
            ortvec_data = torch.tensor(extra_regressors, dtype=torch.float32, device=device)
        else:
            ortvec_data = extra_regressors.to(device=device, dtype=torch.float32)
        if ortvec_data.ndim == 1:
            ortvec_data = ortvec_data.unsqueeze(1)
        extra_nuisance_labels = [f"extra_{i}" for i in range(ortvec_data.shape[1])]

    nuisance_blocks_per_run = build_nuisance_per_run(
        run_starts=run_starts,
        n_timepoints=n_timepoints,
        polort=polort,
        device=device,
        ortvec_files=ortvec_files,
        ortvec_data=ortvec_data,
        verbose=verbose,
    )

    n_poly_cols = polort + 1
    n_nuisance_cols_per_run = nuisance_blocks_per_run[0].shape[1]

    if verbose:
        print(f"  Nuisance per run: {n_nuisance_cols_per_run} columns")
        print()

    # =========================================================================
    # Compute QR projectors once, project data once, project each design cheaply
    # =========================================================================
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if verbose:
        print("Computing QR projectors and projecting data (once for all HRFs)...")

    from .xval import compute_qr_projectors

    q_factors = compute_qr_projectors(nuisance_blocks_per_run, run_starts, device=device)

    # Project data using Q factors (with chunking for large data)
    # Data is on CPU; project per-run, streaming chunks to GPU
    effective_chunk_size = chunk_size or estimate_chunk_size(
        n_voxels=n_voxels,
        n_timepoints=n_timepoints,
        n_regressors=n_nuisance_cols_per_run,
        device=device,
        operation="xval",
    )
    data_size_bytes = n_voxels * n_timepoints * 4
    needs_chunking = data_size_bytes > 1e9 or n_voxels > effective_chunk_size

    if not needs_chunking:
        projected_data = data.clone()
        for run_idx in range(n_runs):
            if q_factors[run_idx] is None:
                continue
            start_tp = run_starts[run_idx]
            end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
            Q = q_factors[run_idx]
            run_data = data[:, start_tp:end_tp].to(device)
            run_data_proj = run_data - (Q @ (Q.T @ run_data.T)).T
            projected_data[:, start_tp:end_tp] = run_data_proj.cpu()
    else:
        projected_data = torch.zeros_like(data)
        n_chunks = (n_voxels + effective_chunk_size - 1) // effective_chunk_size
        for chunk_idx in range(n_chunks):
            cs = chunk_idx * effective_chunk_size
            ce = min(cs + effective_chunk_size, n_voxels)
            for run_idx in range(n_runs):
                start_tp = run_starts[run_idx]
                end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
                chunk_run = data[cs:ce, start_tp:end_tp]
                if q_factors[run_idx] is not None:
                    Q = q_factors[run_idx]
                    chunk_dev = chunk_run.to(device)
                    chunk_proj = chunk_dev - (Q @ (Q.T @ chunk_dev.T)).T
                    projected_data[cs:ce, start_tp:end_tp] = chunk_proj.cpu()
                else:
                    projected_data[cs:ce, start_tp:end_tp] = chunk_run

    # If data fits on GPU, move it there to eliminate CPU→GPU transfers entirely.
    # This is the fast path for small/medium datasets (e.g., masked data, ROIs).
    # LORO CV needs ~3x data size for working memory (train-split copies + lstsq
    # workspace), so use a conservative 0.25 safety fraction.
    keep_on_cpu = estimate_keep_on_cpu(
        n_voxels,
        n_timepoints,
        device,
        gpu_safety_fraction=0.25,
    )
    if not keep_on_cpu and device.type != "cpu":
        projected_data = projected_data.to(device)

    if verbose:
        print(f"  Projected data: {projected_data.shape} on {projected_data.device}")
        print()

    # =========================================================================
    # Pre-compute projected stimulus designs for all HRFs + canonical
    # =========================================================================
    reference_hrf_idx = n_hrfs // 2
    design_matrix_ref = None

    if verbose:
        print("Pre-computing projected designs for all HRFs...")

    all_projected_designs = []
    for hrf_idx in range(n_hrfs):
        hrf = hrf_library[hrf_idx]
        stim_design = convolve_hrf_microtime(
            onsets,
            hrf,
            n_timepoints,
            tr=tr,
            microtime_dt=microtime_dt,
            microtime_onset=microtime_onset,
            run_starts=run_starts,
            device=device,
        )
        projected = _project_design_with_q_factors(
            stim_design, q_factors, run_starts, n_timepoints, n_runs, device
        )
        all_projected_designs.append(projected)
        if hrf_idx == reference_hrf_idx:
            design_matrix_ref = projected.clone()

    n_stim_cols = all_projected_designs[0].shape[1]

    # Canonical HRF baseline
    if verbose:
        print()
        print("Computing canonical HRF baseline for comparison...")

    from .hrf import get_spmg1_hrf

    canonical_mode_lower = canonical_mode.lower()
    if canonical_mode_lower == "spmg1":
        canonical_hrf = get_spmg1_hrf(
            microtime_dt=microtime_dt,
            stim_duration=0.0,
            normalize_peak=True,
            device=device,
        )
        canonical_label = "SPMG1"
    elif canonical_mode_lower in ("glmsingle", "single"):
        canonical_hrf = get_hrf_library(
            mode="single",
            stim_duration=0.0,
            microtime_dt=microtime_dt,
            device=device,
        )
        canonical_label = "GLMsingle"
    else:
        raise ValueError(
            f"Unknown canonical_mode: {canonical_mode}. "
            f"Choose 'spmg1' (AFNI) or 'glmsingle' (scipy/nilearn)."
        )

    if verbose:
        print(f"  Using {canonical_label} canonical HRF for baseline comparison")

    canonical_design = convolve_hrf_microtime(
        onsets,
        canonical_hrf,
        n_timepoints,
        tr=tr,
        microtime_dt=microtime_dt,
        microtime_onset=microtime_onset,
        run_starts=run_starts,
        device=device,
    )
    projected_canonical_design = _project_design_with_q_factors(
        canonical_design, q_factors, run_starts, n_timepoints, n_runs, device
    )

    # =========================================================================
    # Debug: save design diagnostic figures + print detailed stats
    # =========================================================================
    if debug:
        _prefix = debug_prefix or "debug"
        from pathlib import Path as _Path

        _debug_dir = f"{_prefix}_debug"
        _Path(_debug_dir).mkdir(parents=True, exist_ok=True)

        print()
        print("=" * 70)
        print("DEBUG MODE: Detailed Diagnostics")
        print("=" * 70)

        # Per-run data statistics
        run_ends = run_starts[1:] + [n_timepoints]
        print("\nPer-run data statistics (raw):")
        for ri, (rs, re) in enumerate(zip(run_starts, run_ends)):
            rd = data[:, rs:re]
            print(
                f"  Run {ri}: TRs [{rs}:{re}] mean={rd.mean():.2f} std={rd.std():.2f} "
                f"min={rd.min():.2f} max={rd.max():.2f}"
            )

        print("\nPer-run projected data statistics:")
        for ri, (rs, re) in enumerate(zip(run_starts, run_ends)):
            rd = projected_data[:, rs:re]
            print(
                f"  Run {ri}: mean={rd.mean():.4f} std={rd.std():.2f} "
                f"min={rd.min():.2f} max={rd.max():.2f}"
            )

        # Canonical design statistics
        print(f"\nCanonical design (unprojected): {canonical_design.shape}")
        for c in range(canonical_design.shape[1]):
            col = canonical_design[:, c]
            lbl = (
                condition_labels[c]
                if condition_labels and c < len(condition_labels)
                else f"cond{c}"
            )
            print(
                f"  {lbl}: max={col.max():.4f} min={col.min():.4f} "
                f"sum_abs={col.abs().sum():.2f} nonzero_frac={(col.abs() > 0.01).float().mean():.3f}"
            )

        print(f"\nCanonical design (projected): {projected_canonical_design.shape}")
        for c in range(projected_canonical_design.shape[1]):
            col = projected_canonical_design[:, c]
            lbl = (
                condition_labels[c]
                if condition_labels and c < len(condition_labels)
                else f"cond{c}"
            )
            print(
                f"  {lbl}: max={col.max():.4f} min={col.min():.4f} "
                f"sum_abs={col.abs().sum():.2f} nonzero_frac={(col.abs() > 0.01).float().mean():.3f}"
            )

        # Nuisance statistics
        print(
            f"\nNuisance per run: {n_nuisance_cols_per_run} columns/run, "
            f"{n_poly_cols} poly + {n_nuisance_cols_per_run - n_poly_cols} extra"
        )
        for ri, nb in enumerate(nuisance_blocks_per_run):
            print(f"  Run {ri}: shape={nb.shape}, col_norms={nb.norm(dim=0).tolist()}")

        # Q factor diagnostics
        print("\nQR projectors:")
        for ri, qf in enumerate(q_factors):
            if qf is not None:
                print(f"  Run {ri}: Q shape={qf.shape}, Q columns={qf.shape[1]}")
            else:
                print(f"  Run {ri}: None (no nuisance)")

        # Quick OLS fit on projected data+design to show what R² looks like
        print("\nQuick OLS diagnostic (canonical design on projected data):")
        _X = projected_canonical_design.to(projected_data.device)
        _b = torch.linalg.lstsq(_X, projected_data[: min(1000, n_voxels), :].T).solution
        _pred = (_X @ _b).T
        _y = projected_data[: min(1000, n_voxels), :]
        _ss_res = ((_y - _pred) ** 2).sum(dim=1)
        _ss_tot = ((_y - _y.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
        _r2_quick = 1 - _ss_res / _ss_tot.clamp(min=1e-10)
        print(
            f"  In-sample R² (first {min(1000, n_voxels)} voxels): "
            f"mean={_r2_quick.mean():.4f} median={_r2_quick.median():.4f} "
            f"Q25={_r2_quick.quantile(0.25):.4f} Q75={_r2_quick.quantile(0.75):.4f}"
        )
        del _X, _b, _pred, _y, _ss_res, _ss_tot, _r2_quick

        # Save design diagnostic figure
        save_design_diagnostic_figure(
            canonical_design=canonical_design,
            nuisance_per_run=nuisance_blocks_per_run,
            run_starts=run_starts,
            n_timepoints=n_timepoints,
            tr=tr,
            output_path=f"{_debug_dir}/design_diagnostic.png",
            condition_labels=condition_labels,
            projected_design=projected_canonical_design,
        )

        print("=" * 70)
        print()

    # =========================================================================
    # Evaluate all HRFs + canonical via cross-validation
    # =========================================================================
    is_loro = all(len(test) == 1 for _, test in cv_splits)

    if n_hrfs > 1:
        # Batched CV: move data to GPU once per split, evaluate ALL designs.
        # Reduces CPU→GPU transfers by (n_hrfs+1)× vs per-design evaluation.
        # Uses per-fold R² (mean R² across folds) — works for any CV strategy.
        cv_label = "LORO" if is_loro else f"split-half ({n_splits} perms)"
        if verbose:
            print()
            print(f"  Batched {cv_label} evaluation: {n_hrfs + 1} designs x {n_splits} splits")

        all_designs = all_projected_designs + [projected_canonical_design]
        r2_all = _evaluate_hrfs_batched(
            projected_data=projected_data,
            projected_designs=all_designs,
            run_starts=run_starts,
            cv_splits=cv_splits,
            device=device,
            metric=metric,
            chunk_size=effective_chunk_size,
            verbose=verbose,
        )
        xval_r2_median_all = r2_all[:, :n_hrfs].to(device)
        xval_r2_canonical = r2_all[:, n_hrfs].to(device)

        # Diagnostic: verify batched evaluation matches compute_xval_r2 (LORO only)
        if verbose and is_loro:
            # Path A: compute_xval_r2 on our QR-projected data+design
            canonical_check = compute_xval_r2(
                data=projected_data,
                design_matrix=projected_canonical_design,
                run_starts=run_starts,
                stim_indices=list(range(n_stim_cols)),
                nuisance_indices=[],
                cv_splits=cv_splits,
                metric=metric,
                zero_event_strategy="zero",
                device=device,
                batch_size=chunk_size,
                r2_method=r2_method,
                verbose=False,
            )
            r2_path_a = canonical_check["r2"].to(device)

            # Path B: full 3dDenoisefast-style (project_out_nuisance_per_run + compute_xval_r2)
            proj_data_b, proj_design_b = project_out_nuisance_per_run(
                data=data,
                design=canonical_design,
                nuisance_per_run=nuisance_blocks_per_run,
                run_starts=run_starts,
                device=device,
            )
            canonical_check_b = compute_xval_r2(
                data=proj_data_b,
                design_matrix=proj_design_b,
                run_starts=run_starts,
                stim_indices=list(range(n_stim_cols)),
                nuisance_indices=[],
                cv_splits=cv_splits,
                metric=metric,
                zero_event_strategy="zero",
                device=device,
                batch_size=chunk_size,
                r2_method=r2_method,
                verbose=False,
            )
            r2_path_b = canonical_check_b["r2"].to(device)

            print(f"  [diagnostic] Batched LORO canonical R²:      {xval_r2_canonical.mean():.4f}")
            print(f"  [diagnostic] compute_xval_r2 (QR-proj) R²:   {r2_path_a.mean():.4f}")
            print(f"  [diagnostic] 3dDenoisefast-style R²:          {r2_path_b.mean():.4f}")
            diff_ab = (r2_path_a - r2_path_b).abs().mean().item()
            diff_loro = (xval_r2_canonical - r2_path_b).abs().mean().item()
            print(f"  [diagnostic] QR-proj vs Denoisefast diff:     {diff_ab:.6f}")
            print(f"  [diagnostic] Batched LORO vs Denoisefast diff: {diff_loro:.6f}")
            if diff_ab > 0.01:
                print("  *** WARNING: QR projection diverges from project_out_nuisance! ***")
            if diff_loro > 0.01:
                print("  *** WARNING: Batched LORO diverges from Denoisefast path! ***")
            del canonical_check, r2_path_a, proj_data_b, proj_design_b
            del canonical_check_b, r2_path_b
    else:
        # Single-HRF fallback
        if verbose:
            print()
            print(f"  Per-design evaluation: {n_hrfs} HRFs x {n_splits} splits")

        xval_r2_median_all = torch.zeros(n_voxels, n_hrfs, device=device)
        for hrf_idx in tqdm(range(n_hrfs), desc="Evaluating HRFs", disable=not verbose):
            xval_results = compute_xval_r2(
                data=projected_data,
                design_matrix=all_projected_designs[hrf_idx],
                run_starts=run_starts,
                stim_indices=list(range(n_stim_cols)),
                nuisance_indices=[],
                cv_splits=cv_splits,
                metric=metric,
                zero_event_strategy="zero",
                device=device,
                batch_size=chunk_size,
                r2_method=r2_method,
                verbose=False,
            )
            xval_r2_median_all[:, hrf_idx] = xval_results["r2"].to(device)

        canonical_xval = compute_xval_r2(
            data=projected_data,
            design_matrix=projected_canonical_design,
            run_starts=run_starts,
            stim_indices=list(range(canonical_design.shape[1])),
            nuisance_indices=[],
            cv_splits=cv_splits,
            metric=metric,
            zero_event_strategy="zero",
            device=device,
            batch_size=chunk_size,
            r2_method=r2_method,
            verbose=False,
        )
        xval_r2_canonical = canonical_xval["r2"].to(device)

    xval_r2_std_all = torch.zeros(n_voxels, n_hrfs, device=device)

    # Clear GPU cache after evaluation
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if verbose:
        print(f"  Canonical HRF mean xval R²: {xval_r2_canonical.mean().item():.4f}")

    # Fit full dataset with canonical HRF to get betas/tstats for comparison
    # NOTE: For final fit, we need the full (unprojected) data and design with nuisance
    if verbose:
        print("  Fitting full dataset with canonical HRF...")

    # Build block-diagonal nuisance for final fit (used by fit_glm and _fit_voxelwise_hrf)
    nuisance_design = torch.block_diag(*nuisance_blocks_per_run)

    # Follow 3dDenoisefast pattern: pass task design + nuisance separately.
    # fit_glm concatenates them internally and knows which columns are task vs nuisance.
    # max_poly_degree=-1 prevents adding duplicate polynomials (already in nuisance_design).
    canonical_glm_results = fit_glm(
        data=data,
        design=canonical_design,  # Task-only design
        tr=tr,
        max_poly_degree=-1,  # No additional polynomials — already in extra_regressors
        extra_regressors=nuisance_design,  # Nuisance passed separately
        device=device,
        verbose=False,
        preload_data_to_device=(data.device == device),  # Stream chunks if data on CPU
    )

    # Store the canonical design matrix (task + nuisance) for saving
    canonical_design_matrix = torch.cat([canonical_design, nuisance_design], dim=1)

    if verbose:
        print(f"  Canonical HRF full-data R²: {canonical_glm_results.r2.mean().item():.4f}")

    # Select best HRF per voxel based on median CV R²
    hrf_index = xval_r2_median_all.argmax(dim=1)  # (n_voxels,)

    # Extract R² for selected HRF
    xval_r2_best = xval_r2_median_all[torch.arange(n_voxels, device=device), hrf_index]
    xval_r2_std = xval_r2_std_all[torch.arange(n_voxels, device=device), hrf_index]

    if verbose:
        print()
        print("HRF Selection Summary:")
        hrf_counts = torch.bincount(hrf_index, minlength=n_hrfs)
        print(f"  HRF usage distribution: {hrf_counts.cpu().tolist()}")
        print(f"  Mean xval R²: {xval_r2_best.mean().item():.4f}")
        print(f"  Median xval R²: {xval_r2_best.median().item():.4f}")
        r2_improvement = xval_r2_best.mean().item() - xval_r2_canonical.mean().item()
        print(f"  Improvement over canonical: {r2_improvement:+.4f}")
        print()

    # Clear GPU cache before final fit to free fragmented memory
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Final fit: refit entire dataset with voxel-wise optimal HRFs
    if verbose:
        print("Refitting full dataset with voxel-wise optimal HRFs...")

    final_results = _fit_voxelwise_hrf(
        data=data,
        onsets=onsets,
        hrf_library=hrf_library,
        hrf_index=hrf_index,
        nuisance_design=nuisance_design,
        run_starts=run_starts,
        tr=tr,
        microtime_dt=microtime_dt,
        microtime_onset=microtime_onset,
        device=device,
        verbose=verbose,
        chunk_size=chunk_size,
    )

    # Build metadata for ARMA reuse
    hrf_metadata = {
        "hrf_mode": "library",  # Will be set by caller if known
        "n_hrfs": n_hrfs,
        "tr": tr,
        "stim_durations": stim_durations,
        "cv_strategy": cv_strategy,
        "n_splits": n_splits,
        "metric": metric,
        "microtime_dt": microtime_dt,
        "microtime_onset": microtime_onset,
        "polort": polort,
        "n_poly_cols": n_poly_cols,
        "n_extra_nuisance": n_nuisance_cols_per_run - n_poly_cols,
        "n_nuisance_total": nuisance_design.shape[1],
        "extra_nuisance_labels": extra_nuisance_labels,
        "n_voxels": n_voxels,
        "n_timepoints": n_timepoints,
        "n_runs": n_runs,
        "hrf_usage_counts": torch.bincount(hrf_index, minlength=n_hrfs).cpu().tolist(),
    }

    # Build HRF group indices for efficient ARMA reuse
    hrf_group_indices = {}
    for h in range(n_hrfs):
        mask = hrf_index == h
        if mask.any():
            hrf_group_indices[h] = torch.where(mask)[0]

    # Create results container
    results = HRFSelectionResults(
        hrf_index=hrf_index.cpu(),
        xval_r2_best=xval_r2_best.cpu(),
        xval_r2_std=xval_r2_std.cpu(),
        xval_r2_all_hrfs=xval_r2_median_all.cpu(),
        xval_r2_canonical=xval_r2_canonical.cpu(),
        final_results=final_results,
        canonical_results=canonical_glm_results,
        hrf_library=hrf_library.cpu(),
        hrf_metadata=hrf_metadata,
        hrf_group_indices={k: v.cpu() for k, v in hrf_group_indices.items()},
        design_matrix=design_matrix_ref,
        canonical_design_matrix=canonical_design_matrix.cpu(),
    )

    if verbose:
        print()
        print("=" * 70)
        print("HRF SELECTION COMPLETE")
        print("=" * 70)
        print("  Best HRF per voxel stored in hrf_index")
        print(f"  Final betas shape: {final_results.betas.shape}")
        print(f"  Final R² mean: {final_results.r2.mean().item():.4f}")
        print()

    return results


def _fit_voxelwise_hrf(
    data: torch.Tensor,
    onsets: torch.Tensor,
    hrf_library: torch.Tensor,
    hrf_index: torch.Tensor,
    nuisance_design: torch.Tensor,
    run_starts: list[int],
    tr: float,
    microtime_dt: float,
    microtime_onset: int,
    device: torch.device,
    verbose: bool,
    chunk_size: int | None,
) -> GLMResults:
    """
    Fit GLM with voxel-wise HRFs by grouping voxels with same HRF.

    This is the key efficiency trick: instead of fitting each voxel separately,
    we group voxels by their selected HRF and fit each group together.
    Similar to how 3dREMLfast handles voxel-wise ARMA parameters.

    Parameters
    ----------
    nuisance_design : torch.Tensor
        (n_timepoints, n_nuisance_cols) Pre-built nuisance design matrix
        containing polynomials and any extra nuisance regressors.
    """
    n_voxels = data.shape[0]

    # Calculate bins per TR for microtime
    bins_per_tr = int(round(tr / microtime_dt))
    n_timepoints = onsets.shape[0] // bins_per_tr

    n_conditions = onsets.shape[1]
    n_nuisance_cols = nuisance_design.shape[1]

    # Initialize output tensors
    all_betas = torch.zeros(n_voxels, n_conditions, device=device)
    all_r2 = torch.zeros(n_voxels, device=device)
    all_tstats = torch.zeros(n_voxels, n_conditions, device=device)
    all_sigma2 = torch.zeros(n_voxels, device=device)

    # Group voxels by HRF
    unique_hrfs = torch.unique(hrf_index)

    if verbose:
        print(f"  Fitting {len(unique_hrfs)} HRF groups...")

    hrf_iterator = tqdm(unique_hrfs, desc="Fitting HRF groups")

    # Store dof from any group (should be same for all as design has same structure)
    stored_dof = None

    for hrf_idx in hrf_iterator:
        hrf_idx_int = hrf_idx.item()

        # Get voxels using this HRF
        voxel_mask = hrf_index == hrf_idx
        voxel_indices = torch.where(voxel_mask)[0]
        n_group_voxels = len(voxel_indices)

        if n_group_voxels == 0:
            continue

        # Convolve with this HRF (do this once for the group)
        hrf = hrf_library[hrf_idx_int]

        stim_design = convolve_hrf_microtime(
            onsets,
            hrf,
            n_timepoints,
            tr=tr,
            microtime_dt=microtime_dt,
            microtime_onset=microtime_onset,
            run_starts=run_starts,
            device=device,
        )

        # Chunk within HRF group if too large (to avoid OOM)
        # Use smaller chunks for GPU to prevent memory issues
        max_voxels_per_chunk = 50000 if device.type == "cuda" else 100000
        n_chunks = (n_group_voxels + max_voxels_per_chunk - 1) // max_voxels_per_chunk

        def _fit_and_store(voxel_idx: torch.Tensor, label: str) -> None:
            """Fit GLM for a subset of voxels and store results."""
            nonlocal stored_dof

            # Get data for this subset
            if data.device.type == "cpu" and voxel_idx.device.type != "cpu":
                idx_cpu = voxel_idx.cpu()
                subset_data = data[idx_cpu, :]
            else:
                subset_data = data[voxel_idx, :]

            # Follow 3dDenoisefast pattern: task design + nuisance separately
            # max_poly_degree=-1 prevents duplicate polynomials
            subset_results = fit_glm(
                subset_data,
                stim_design,  # Task-only design
                tr=tr,
                max_poly_degree=-1,  # No extra polynomials — already in nuisance_design
                extra_regressors=nuisance_design,  # Nuisance passed separately
                device=device,
                verbose=False,
                preload_data_to_device=(subset_data.device == device),
            )

            if stored_dof is None and subset_results.dof is not None:
                stored_dof = subset_results.dof

            # Store results (handle device mismatches)
            if subset_results.betas is not None:
                betas_to_store = subset_results.betas
                if betas_to_store.device != all_betas.device:
                    betas_to_store = betas_to_store.to(all_betas.device)
                all_betas[voxel_idx, :] = betas_to_store

            if subset_results.r2 is not None:
                r2_to_store = subset_results.r2
                if r2_to_store.device != all_r2.device:
                    r2_to_store = r2_to_store.to(all_r2.device)
                all_r2[voxel_idx] = r2_to_store

            if subset_results.tstats is not None:
                tstats_to_store = subset_results.tstats
                if tstats_to_store.device != all_tstats.device:
                    tstats_to_store = tstats_to_store.to(all_tstats.device)
                all_tstats[voxel_idx, :] = tstats_to_store

            if subset_results.sigma2 is not None:
                sigma2_to_store = subset_results.sigma2
                if sigma2_to_store.device != all_sigma2.device:
                    sigma2_to_store = sigma2_to_store.to(all_sigma2.device)
                all_sigma2[voxel_idx] = sigma2_to_store

            del subset_data, subset_results
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if n_chunks > 1:
            hrf_iterator.set_postfix_str(f"{n_group_voxels:,} voxels in {n_chunks} chunks")
            for chunk_idx in range(n_chunks):
                chunk_start = chunk_idx * max_voxels_per_chunk
                chunk_end = min(chunk_start + max_voxels_per_chunk, n_group_voxels)
                _fit_and_store(voxel_indices[chunk_start:chunk_end], f"chunk {chunk_idx}")
        else:
            hrf_iterator.set_postfix_str(f"{n_group_voxels:,} voxels")
            _fit_and_store(voxel_indices, "group")

    # Build GLMResults
    results = GLMResults()
    results.betas = all_betas.cpu()
    results.r2 = all_r2.cpu()
    results.tstats = all_tstats.cpu()
    results.sigma2 = all_sigma2.cpu()
    results.meanvol = data.mean(dim=1).cpu()
    results.dof = stored_dof  # Propagate dof from fit_glm for 3drefit labeling

    return results


def _fit_voxelwise_hrf_canonical(
    data: torch.Tensor,
    onsets: torch.Tensor,
    canonical_hrf: torch.Tensor,
    nuisance_design: torch.Tensor,
    tr: float,
    microtime_dt: float,
    microtime_onset: int,
    device: torch.device,
    verbose: bool = False,
) -> GLMResults:
    """
    Fit GLM with canonical HRF for all voxels (for comparison with per-voxel optimal HRFs).

    This creates ONE design matrix for all voxels (not per-voxel) and processes
    in chunks to avoid OOM. Used for comparison with the per-voxel optimal HRF results.

    Parameters
    ----------
    data : torch.Tensor
        (n_voxels, n_timepoints) fMRI data
    onsets : torch.Tensor
        Onset matrix (n_timepoints, n_conditions)
    canonical_hrf : torch.Tensor
        (hrf_length,) Canonical HRF to use for all voxels
    nuisance_design : torch.Tensor
        (n_timepoints, n_nuisance_cols) Pre-built nuisance design
    tr : float
        Repetition time in seconds
    microtime_dt : float
        Microtime resolution in seconds
    microtime_onset : int
        Microtime onset bin
    device : torch.device
        Compute device
    verbose : bool
        Print progress messages

    Returns
    -------
    GLMResults
        Results containing betas, R², tstats, etc. from canonical HRF fit
    """
    n_voxels = data.shape[0]
    n_timepoints = onsets.shape[0] // int(round(tr / microtime_dt))

    # Get number of conditions
    n_conditions = onsets.shape[1]

    if verbose:
        print("  Fitting canonical HRF (all voxels, one design matrix)...")

    # Create single design matrix with canonical HRF
    stim_design = convolve_hrf_microtime(
        onsets,
        canonical_hrf,
        n_timepoints=n_timepoints,
        tr=tr,
        microtime_dt=microtime_dt,
        microtime_onset=microtime_onset,
        device=device,
    )

    # Follow 3dDenoisefast pattern: pass task design + nuisance separately.
    # max_poly_degree=-1 prevents duplicate polynomials (already in nuisance_design).
    # fit_glm handles chunking internally.
    results_glm = fit_glm(
        data=data,
        design=stim_design,  # Task-only design
        tr=tr,
        max_poly_degree=-1,  # No extra polynomials — already in nuisance_design
        extra_regressors=nuisance_design,  # Nuisance passed separately
        device=device,
        verbose=False,
        preload_data_to_device=(data.device == device),
    )

    # Build results object
    results = GLMResults()
    results.betas = results_glm.betas.cpu() if results_glm.betas is not None else None
    results.r2 = results_glm.r2.cpu() if results_glm.r2 is not None else None
    results.tstats = results_glm.tstats.cpu() if results_glm.tstats is not None else None
    results.sigma2 = results_glm.sigma2.cpu() if results_glm.sigma2 is not None else None
    results.meanvol = data.mean(dim=1).cpu()
    results.dof = results_glm.dof

    if verbose:
        print(f"  Canonical fit complete. Mean R²: {results.r2.mean().item():.4f}")

    return results


def _fit_voxelwise_hrf_single_trial(
    data: torch.Tensor,
    onsets_by_condition: list[list[np.ndarray]],
    hrf_library: list[torch.Tensor],
    hrf_index: torch.Tensor,
    nuisance_design: torch.Tensor,
    durations: list[float],
    run_starts: list[int],
    tr: float,
    n_timepoints: int,
    microtime_dt: float,
    condition_labels: list[str],
    device: torch.device,
    verbose: bool = False,
) -> GLMResults:
    """
    Fit single-trial GLM with per-voxel optimal HRFs, grouped by HRF for efficiency.

    This is the key optimization: instead of creating a per-voxel design matrix (which OOMs),
    we group voxels by their optimal HRF and process each group with one design matrix.
    Within each HRF group, we use sub-chunking if needed to avoid OOM.

    Follows the same pattern as _fit_voxelwise_hrf() but for single-trial designs.

    Parameters
    ----------
    data : torch.Tensor
        (n_voxels, n_timepoints) fMRI data
    onsets_by_condition : list of list of np.ndarray
        Onsets organized as [condition][run] -> np.ndarray of onset times (seconds)
    hrf_library : list of torch.Tensor
        List of HRF functions, each (hrf_length,)
    hrf_index : torch.Tensor
        (n_voxels,) HRF index for each voxel
    nuisance_design : torch.Tensor
        (n_timepoints, n_nuisance_cols) Pre-built nuisance design
    durations : list of float
        Duration in seconds for each condition
    run_starts : list of int
        Starting timepoint for each run
    tr : float
        Repetition time in seconds
    n_timepoints : int
        Total number of timepoints
    microtime_dt : float
        Microtime resolution
    condition_labels : list of str
        Condition names
    device : torch.device
        Compute device
    verbose : bool
        Print progress

    Returns
    -------
    GLMResults
        Single-trial betas and statistics
    """
    from .xval import compute_r2_metric

    n_voxels = data.shape[0]
    n_timepoints = data.shape[1]
    n_conditions = len(condition_labels)
    n_hrfs = len(hrf_library)
    n_runs = len(run_starts)

    if verbose:
        print(f"  Refitting single-trial betas with optimal HRFs ({n_hrfs} HRF groups)...")

    # Group voxels by HRF
    unique_hrfs = torch.unique(hrf_index)

    # First, get trial info using canonical HRF (just to get n_trials)
    from .ridge import create_single_trial_design

    st_design_canonical, trial_labels, trial_cond_ids, trial_run_ids, condition_design = (
        create_single_trial_design(
            onsets_by_condition=onsets_by_condition,
            durations=durations,
            run_starts=run_starts,
            tr=tr,
            n_timepoints=n_timepoints,
            hrf_library=None,  # Canonical HRF
            microtime_dt=microtime_dt,
            condition_labels=condition_labels,
            device="cpu",  # Create on CPU to save GPU memory
        )
    )
    n_trials = len(trial_labels)

    if verbose:
        print(f"    Single-trial design: {n_trials} trials")
        print(f"    HRF groups: {len(unique_hrfs)}")

    # Initialize output for single-trial betas
    all_single_trial_betas = torch.zeros(n_voxels, n_trials, device="cpu")
    all_single_trial_r2 = torch.zeros(n_voxels, device="cpu")

    # Process each HRF group
    hrf_iterator = (
        tqdm(unique_hrfs, desc="Refitting HRF groups") if verbose else unique_hrfs.tolist()
    )
    for hrf_idx in hrf_iterator:
        hrf_idx_int = hrf_idx.item()

        # Get voxels using this HRF
        voxel_mask = hrf_index == hrf_idx
        voxel_indices = torch.where(voxel_mask)[0]
        n_group_voxels = len(voxel_indices)

        if n_group_voxels == 0:
            continue

        if verbose:
            hrf_iterator.set_postfix_str(f"{n_group_voxels:,} voxels")

        # Create single-trial design for this HRF
        hrf = hrf_library[hrf_idx_int]
        st_design, _, trial_cond_ids_hrf, trial_run_ids_hrf, _ = create_single_trial_design(
            onsets_by_condition=onsets_by_condition,
            durations=durations,
            run_starts=run_starts,
            tr=tr,
            n_timepoints=n_timepoints,
            microtime_dt=microtime_dt,
            condition_labels=condition_labels,
            hrf_library=[hrf],  # Convolve with this HRF
            device=device,
        )
        # st_design is (n_timepoints, n_trials) for this HRF

        # Build full design: [single_trial | nuisance] so drift is modeled
        full_design = torch.cat([st_design, nuisance_design.to(st_design.device)], dim=1)
        n_full_regressors = full_design.shape[1]

        # Determine chunk size using dynamic estimator
        # Single-trial GLM (many trial regressors)
        voxel_chunk_size = dyn_chunk_estimator(
            n_voxels=n_group_voxels,
            n_timepoints=n_timepoints,
            n_task_regressors=st_design.shape[1],  # Number of trials
            n_nuisance_regressors=nuisance_design.shape[1],
            device=device,
            operation="glm",
            cv_strategy=None,  # This is final fit, not CV
            n_runs=len(run_starts),
            data_location="auto",
            min_chunk_size=10000,
            max_chunk_size=None,  # Use default
            safety_factor=0.5,
            verbose=False,
        )

        n_chunks = (n_group_voxels + voxel_chunk_size - 1) // voxel_chunk_size

        if verbose and n_chunks > 1:
            print(f"      {n_group_voxels:,} voxels in {n_chunks} chunks of ~{voxel_chunk_size:,}")

        # Process this HRF group in chunks
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * voxel_chunk_size
            chunk_end = min(chunk_start + voxel_chunk_size, n_group_voxels)
            chunk_voxel_indices = voxel_indices[chunk_start:chunk_end]

            # Get data for this chunk
            if data.device.type == "cpu":
                chunk_data = data[chunk_voxel_indices.cpu(), :]
            else:
                chunk_data = data[chunk_voxel_indices, :]

            # Move design and data to same device
            full_design_device = full_design.to(device)
            chunk_data_device = chunk_data.to(device)

            # OLS via lstsq (numerically stable for ill-conditioned single-trial designs)
            # full_design: (n_timepoints, n_trials + n_nuisance), data.T: (n_timepoints, n_voxels)
            all_betas = torch.linalg.lstsq(
                full_design_device, chunk_data_device.T
            ).solution  # (n_full, n_voxels)
            all_betas = all_betas.T  # (n_voxels, n_full)

            # Extract only single-trial betas (first n_trials columns)
            chunk_betas = all_betas[:, :n_trials]

            # Compute predictions from full model for R²
            chunk_predictions = (full_design_device @ all_betas.T).T  # (n_voxels, n_timepoints)
            chunk_r2 = compute_r2_metric(chunk_data_device, chunk_predictions)  # (n_voxels,)

            # Accumulate results
            all_single_trial_betas[chunk_voxel_indices, :] = chunk_betas.cpu()
            all_single_trial_r2[chunk_voxel_indices] = chunk_r2.cpu()

            # Clean up
            del (
                chunk_data,
                chunk_data_device,
                full_design_device,
                all_betas,
                chunk_betas,
                chunk_predictions,
                chunk_r2,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Build results object
    results = GLMResults()
    results.betas = all_single_trial_betas
    results.r2 = all_single_trial_r2
    results.tstats = None  # tstats not computed for ridge
    results.sigma2 = None  # Not computed for ridge
    results.meanvol = data.mean(dim=1).cpu()
    results.dof = n_timepoints - n_trials  # Approximate dof
    results.trial_labels = trial_labels  # Store trial labels for saving

    if verbose:
        print(f"    Single-trial refit complete. Mean CV R²: {results.r2.mean().item():.4f}")

    return results


def _write_afni_xmat(
    design_matrix: np.ndarray,
    output_file: str,
    n_stim_cols: int,
    condition_labels: list[str] | None,
    run_starts: list[int],
    tr: float,
    polort: int | None = None,
    extra_nuisance_labels: list[str] | None = None,
) -> None:
    """Write design matrix in AFNI xmat.1D format.

    Parameters
    ----------
    design_matrix : np.ndarray
        (n_timepoints, n_columns) design matrix
    output_file : str
        Path to output file
    n_stim_cols : int
        Number of stimulus columns (rest are nuisance)
    condition_labels : list of str, optional
        Labels for stimulus conditions
    run_starts : list of int
        Starting timepoint for each run
    tr : float
        Repetition time in seconds
    polort : int, optional
        Polynomial order (for generating meaningful poly labels per run)
    extra_nuisance_labels : list of str, optional
        Labels for extra nuisance regressors (e.g., ['motion_0', 'motion_1', ...])
    """
    n_timepoints, n_cols = design_matrix.shape
    n_runs = len(run_starts)

    # Build column labels
    if condition_labels is not None and len(condition_labels) == n_stim_cols:
        stim_labels = list(condition_labels)
    else:
        stim_labels = [f"stim{i:02d}" for i in range(n_stim_cols)]

    # Build nuisance labels with meaningful names
    n_nuisance = n_cols - n_stim_cols
    nuisance_labels = []

    # If polort is known, build per-run polynomial labels
    if polort is not None:
        n_poly_per_run = polort + 1
        n_poly_total = n_runs * n_poly_per_run
        for r in range(n_runs):
            for p in range(n_poly_per_run):
                nuisance_labels.append(f"r{r + 1:02d}_poly{p}")
    else:
        # Fall back to auto-detecting poly columns
        n_poly_total = n_nuisance - (len(extra_nuisance_labels) if extra_nuisance_labels else 0)
        for i in range(n_poly_total):
            nuisance_labels.append(f"poly{i:02d}")

    # Add extra nuisance labels (motion, physio, etc.)
    if extra_nuisance_labels:
        nuisance_labels.extend(extra_nuisance_labels)

    # Pad if we don't have enough labels
    while len(nuisance_labels) < n_nuisance:
        nuisance_labels.append(f"nuisance{len(nuisance_labels):02d}")

    all_labels = stim_labels + nuisance_labels[:n_nuisance]

    # Write simplified AFNI xmat format
    with open(output_file, "w") as f:
        # Header
        f.write("# <matrix\n")
        f.write(f'#  ni_type = "{n_cols}*double"\n')
        f.write(f'#  ni_dimen = "{n_timepoints}"\n')
        f.write(f'#  ColumnLabels = "{" ; ".join(all_labels)}"\n')
        f.write(f'#  RowTR = "{tr}"\n')
        f.write(f'#  GoodList = "0..{n_timepoints - 1}"\n')
        f.write(f'#  NRowFull = "{n_timepoints}"\n')
        run_starts_str = ",".join(map(str, run_starts))
        f.write(f'#  RunStart = "{run_starts_str}"\n')
        f.write(f'#  Nstim = "{n_stim_cols}"\n')
        if n_stim_cols > 0:
            stim_bots = ",".join(map(str, range(n_stim_cols)))
            stim_tops = ",".join(map(str, range(n_stim_cols)))
            f.write(f'#  StimBots = "{stim_bots}"\n')
            f.write(f'#  StimTops = "{stim_tops}"\n')
            f.write(f'#  StimLabels = "{" ; ".join(stim_labels)}"\n')
        f.write("# >\n")

        # Data matrix
        for row in design_matrix:
            f.write(" ".join(f"{v:.6f}" for v in row) + "\n")


def save_design_diagnostic_figure(
    canonical_design: torch.Tensor,
    nuisance_per_run: list[torch.Tensor],
    run_starts: list[int],
    n_timepoints: int,
    tr: float,
    output_path: str,
    condition_labels: list[str] | None = None,
    projected_design: torch.Tensor | None = None,
) -> None:
    """Save diagnostic figure showing design matrix as the code interprets it.

    Creates a multi-panel figure showing:
    - Top: Canonical task design columns (labeled by condition)
    - Middle: Per-run polynomial nuisance (block-diagonal structure)
    - Bottom: If projected_design is given, the projected (cleaned) design
    - Run boundaries marked with vertical lines

    Parameters
    ----------
    canonical_design : torch.Tensor
        (n_timepoints, n_conditions) Unprojected canonical task design
    nuisance_per_run : list of torch.Tensor
        Per-run nuisance blocks
    run_starts : list of int
        Starting timepoint for each run
    n_timepoints : int
        Total number of timepoints
    tr : float
        Repetition time
    output_path : str
        Where to save the PNG
    condition_labels : list of str, optional
        Labels for each condition column
    projected_design : torch.Tensor, optional
        (n_timepoints, n_conditions) Projected (nuisance-removed) design for comparison
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not available, skipping design diagnostic figure")
        return

    n_conditions = canonical_design.shape[1]
    n_runs = len(run_starts)
    run_ends = run_starts[1:] + [n_timepoints]
    time_axis = np.arange(n_timepoints) * tr

    n_panels = 3 if projected_design is not None else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 4 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    # Panel 1: Task design (canonical)
    ax = axes[0]
    design_np = canonical_design.cpu().numpy()
    for c in range(n_conditions):
        label = (
            condition_labels[c] if condition_labels and c < len(condition_labels) else f"cond{c}"
        )
        ax.plot(time_axis, design_np[:, c], label=label, alpha=0.8)
    for rs in run_starts[1:]:
        ax.axvline(rs * tr, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Canonical Task Design ({n_conditions} conditions, {n_runs} runs)")
    ax.legend(loc="upper right", fontsize=8, ncol=min(n_conditions, 6))

    # Panel 2: Nuisance (block-diagonal polynomials)
    ax = axes[1]
    nuisance_full = torch.block_diag(*nuisance_per_run).cpu().numpy()
    n_nuis_cols = nuisance_full.shape[1]
    # Show as image
    im = ax.imshow(
        nuisance_full.T,
        aspect="auto",
        cmap="RdBu_r",
        extent=[0, n_timepoints * tr, n_nuis_cols - 0.5, -0.5],
        interpolation="nearest",
    )
    for rs in run_starts[1:]:
        ax.axvline(rs * tr, color="black", linestyle="-", alpha=0.7)
    ax.set_ylabel("Nuisance column")
    n_poly = nuisance_per_run[0].shape[1]
    n_extra = n_nuis_cols // n_runs - n_poly if n_runs > 0 else 0
    ax.set_title(
        f"Block-Diagonal Nuisance ({n_poly} poly + {n_extra} extra per run, "
        f"{n_nuis_cols} total columns)"
    )
    fig.colorbar(im, ax=ax, shrink=0.6)

    # Panel 3: Projected design (if available)
    if projected_design is not None:
        ax = axes[2]
        proj_np = projected_design.cpu().numpy()
        for c in range(n_conditions):
            label = (
                condition_labels[c]
                if condition_labels and c < len(condition_labels)
                else f"cond{c}"
            )
            ax.plot(time_axis, proj_np[:, c], label=label, alpha=0.8)
        for rs in run_starts[1:]:
            ax.axvline(rs * tr, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("Amplitude")
        ax.set_title("Projected Task Design (nuisance removed)")
        ax.legend(loc="upper right", fontsize=8, ncol=min(n_conditions, 6))

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Design diagnostic figure saved: {output_path}")


def save_hrf_selection_results(
    results: HRFSelectionResults,
    output_prefix: str,
    volume_shape: tuple[int, int, int] | None = None,
    affine: np.ndarray | None = None,
    voxel_mask: torch.Tensor | None = None,
    condition_labels: list[str] | None = None,
    run_starts: list[int] | None = None,
    save_all_hrf_designs: bool = False,
    onsets: torch.Tensor | None = None,
    save_plots: bool = False,
) -> dict[str, str]:
    """
    Save HRF selection results to disk.

    Parameters
    ----------
    results : HRFSelectionResults
        Results from fit_glm_hrf_library_with_xval
    output_prefix : str
        Output file prefix (e.g., 'output/subject01')
    volume_shape : tuple, optional
        (x, y, z) shape for NIfTI output
    affine : np.ndarray, optional
        4x4 affine transformation for NIfTI
    voxel_mask : torch.Tensor, optional
        Boolean mask for voxels (if data was masked)
    condition_labels : list of str, optional
        Labels for each condition
    run_starts : list of int, optional
        Starting timepoint for each run (for AFNI xmat format)
    save_all_hrf_designs : bool, default=False
        If True, save individual design matrices for each HRF in the library.
        Each file is named {prefix}_design_hrf{idx:02d}.xmat.1D and can be
        used to run the GLM externally (e.g., with AFNI's 3dREMLfit).
    onsets : torch.Tensor, optional
        Required if save_all_hrf_designs=True. The onset matrix used for
        design matrix construction (at microtime resolution if applicable).
    save_plots : bool, default=False
        If True, save design matrix plots as PNG images.

    Returns
    -------
    output_files : dict
        Mapping of output type to file path
    """
    import json
    from pathlib import Path

    from .glm_outputs import write_glm_bucket_as_nifti

    output_prefix = Path(output_prefix)
    output_dir = output_prefix.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = {}

    # Get TR for plotting
    tr = results.hrf_metadata.get("tr", 1.0)

    # 1. Save HRF index map (1-based: 1 to N, not 0 to N-1, since 0 = background)
    hrf_index_file = f"{output_prefix}_hrf_index.nii.gz"
    # Add 1 to convert from 0-indexed to 1-indexed for AFNI compatibility
    hrf_index_1based = results.hrf_index.float() + 1.0
    _save_volume(hrf_index_1based, hrf_index_file, volume_shape, affine, voxel_mask)
    output_files["hrf_index"] = hrf_index_file

    # 2. Save CV R² for best HRF
    xval_r2_file = f"{output_prefix}_xval_r2.nii.gz"
    _save_volume(results.xval_r2_best, xval_r2_file, volume_shape, affine, voxel_mask)
    output_files["xval_r2"] = xval_r2_file

    # 3. Save CV R² std
    xval_std_file = f"{output_prefix}_xval_r2_std.nii.gz"
    _save_volume(results.xval_r2_std, xval_std_file, volume_shape, affine, voxel_mask)
    output_files["xval_r2_std"] = xval_std_file

    # 3b. Save canonical HRF baseline R² for comparison
    if results.xval_r2_canonical is not None:
        canonical_r2_file = f"{output_prefix}_xval_r2_canonical.nii.gz"
        _save_volume(
            results.xval_r2_canonical,
            canonical_r2_file,
            volume_shape,
            affine,
            voxel_mask,
        )
        output_files["xval_r2_canonical"] = canonical_r2_file

    # 3c. Save CV R² for ALL HRFs as 4D volume (n_voxels, n_hrfs)
    if results.xval_r2_all_hrfs is not None:
        xval_r2_all_file = f"{output_prefix}_xval_r2_all_hrfs.nii.gz"
        _save_volume_4d(
            results.xval_r2_all_hrfs,
            xval_r2_all_file,
            volume_shape,
            affine,
            voxel_mask,
        )
        output_files["xval_r2_all_hrfs"] = xval_r2_all_file

    # 3d. Save clearly-named R² maps (single-trial path)
    for field_name in [
        "canonical_full_r2",
        "hrfopt_full_r2",
        "canonical_xval_r2",
        "hrfopt_xval_r2",
    ]:
        val = getattr(results, field_name, None)
        if val is not None:
            fpath = f"{output_prefix}_{field_name}.nii.gz"
            _save_volume(val, fpath, volume_shape, affine, voxel_mask)
            output_files[field_name] = fpath

    # 4. Save final betas
    if results.final_results is not None:
        results.final_results.original_shape = volume_shape
        results.final_results.affine = affine
        if voxel_mask is not None:
            results.final_results.voxel_mask = voxel_mask

        betas_file = f"{output_prefix}_stats.nii.gz"
        write_glm_bucket_as_nifti(
            results.final_results,
            betas_file,
            condition_names=condition_labels,
            volume_shape=volume_shape,
            affine=affine,
        )
        output_files["stats"] = betas_file

    # 4b. Save canonical HRF stats (betas, t-stats) for comparison
    if results.canonical_results is not None:
        results.canonical_results.original_shape = volume_shape
        results.canonical_results.affine = affine
        if voxel_mask is not None:
            results.canonical_results.voxel_mask = voxel_mask

        canonical_stats_file = f"{output_prefix}_canonical_stats.nii.gz"
        write_glm_bucket_as_nifti(
            results.canonical_results,
            canonical_stats_file,
            condition_names=condition_labels,
            volume_shape=volume_shape,
            affine=affine,
        )
        output_files["canonical_stats"] = canonical_stats_file

    # 5. Save HRF library for ARMA reuse
    hrf_lib_file = f"{output_prefix}_hrf_library.pt"
    torch.save(
        {
            "hrf_library": results.hrf_library,
            "hrf_index": results.hrf_index,
            "hrf_group_indices": results.hrf_group_indices,
            "metadata": results.hrf_metadata,
        },
        hrf_lib_file,
    )
    output_files["hrf_library"] = hrf_lib_file

    # 6. Save design matrix in AFNI xmat.1D format
    if results.design_matrix is not None:
        design_np = results.design_matrix.cpu().numpy()
        n_timepoints = design_np.shape[0]

        # Determine column structure: stimulus columns + nuisance columns
        n_conditions = results.hrf_metadata.get("stim_durations")
        if n_conditions is not None:
            n_stim_cols = len(n_conditions) if isinstance(n_conditions, list) else 1
        else:
            # Estimate from final_results if available
            if results.final_results is not None and results.final_results.betas is not None:
                n_stim_cols = results.final_results.betas.shape[1]
            else:
                n_stim_cols = design_np.shape[1]  # Assume all are stimulus

        # Get run_starts from metadata or parameter
        if run_starts is None:
            n_runs = results.hrf_metadata.get("n_runs", 1)
            run_length = n_timepoints // n_runs
            run_starts = [i * run_length for i in range(n_runs)]

        # Get TR and nuisance info from metadata
        tr = results.hrf_metadata.get("tr", 2.0)
        polort = results.hrf_metadata.get("polort")
        extra_nuisance_labels = results.hrf_metadata.get("extra_nuisance_labels", [])

        # Write optimized HRF design matrix
        design_file = f"{output_prefix}_design.xmat.1D"
        _write_afni_xmat(
            design_np,
            design_file,
            n_stim_cols,
            condition_labels,
            run_starts,
            tr,
            polort=polort,
            extra_nuisance_labels=extra_nuisance_labels,
        )
        output_files["design"] = design_file

    # 6b. Save canonical HRF design matrix in AFNI xmat.1D format
    if results.canonical_design_matrix is not None:
        canonical_design_np = results.canonical_design_matrix.cpu().numpy()
        n_timepoints = canonical_design_np.shape[0]

        # Determine column structure
        n_conditions = results.hrf_metadata.get("stim_durations")
        if n_conditions is not None:
            n_stim_cols = len(n_conditions) if isinstance(n_conditions, list) else 1
        else:
            if (
                results.canonical_results is not None
                and results.canonical_results.betas is not None
            ):
                n_stim_cols = results.canonical_results.betas.shape[1]
            else:
                n_stim_cols = canonical_design_np.shape[1]

        # Get run_starts/TR/nuisance info from metadata or parameter
        if run_starts is None:
            n_runs = results.hrf_metadata.get("n_runs", 1)
            run_length = n_timepoints // n_runs
            run_starts = [i * run_length for i in range(n_runs)]
        tr = results.hrf_metadata.get("tr", 2.0)
        polort = results.hrf_metadata.get("polort")
        extra_nuisance_labels = results.hrf_metadata.get("extra_nuisance_labels", [])

        # Write canonical HRF design matrix
        canonical_design_file = f"{output_prefix}_canonical_design.xmat.1D"
        _write_afni_xmat(
            canonical_design_np,
            canonical_design_file,
            n_stim_cols,
            condition_labels,
            run_starts,
            tr,
            polort=polort,
            extra_nuisance_labels=extra_nuisance_labels,
        )
        output_files["canonical_design"] = canonical_design_file

    # 6c. Save individual design matrices for each HRF in the library
    # These can be used to run external GLMs (e.g., with AFNI's 3dREMLfit)
    if save_all_hrf_designs:
        if onsets is None:
            import warnings

            warnings.warn(
                "save_all_hrf_designs=True but onsets not provided. "
                "Cannot generate individual HRF design matrices."
            )
        else:
            # Get parameters from metadata
            tr = results.hrf_metadata.get("tr", 2.0)
            microtime_dt = results.hrf_metadata.get("microtime_dt", 0.1)
            microtime_onset = results.hrf_metadata.get("microtime_onset", 0)
            n_hrfs = results.hrf_library.shape[0]
            hrf_mode = results.hrf_metadata.get("hrf_mode", "library")

            # Calculate bins per TR
            bins_per_tr = int(round(tr / microtime_dt))

            # Determine n_timepoints from design matrix or onsets
            if results.design_matrix is not None:
                n_timepoints = results.design_matrix.shape[0]
            else:
                n_timepoints = onsets.shape[0] // bins_per_tr

            # Get run_starts/condition labels
            if run_starts is None:
                n_runs = results.hrf_metadata.get("n_runs", 1)
                run_length = n_timepoints // n_runs
                run_starts_local = [i * run_length for i in range(n_runs)]
            else:
                run_starts_local = run_starts

            # Build polynomial design for nuisance columns (block-diagonal for runs)
            polort_val = results.hrf_metadata.get("polort")
            if polort_val is None or not isinstance(polort_val, int):
                polort_val = min(1 + int(n_timepoints * tr / 150), 3)

            # Build block-diagonal polynomial matrix
            n_runs = len(run_starts_local)
            poly_blocks = []
            for i in range(n_runs):
                if i < n_runs - 1:
                    run_len = run_starts_local[i + 1] - run_starts_local[i]
                else:
                    run_len = n_timepoints - run_starts_local[i]
                poly_block = construct_polynomial_matrix(run_len, polort_val, onsets.device)
                poly_blocks.append(poly_block)
            poly_design = torch.block_diag(*poly_blocks)

            # Create output directory for HRF designs
            hrf_designs_dir = Path(f"{output_prefix}_hrf_designs")
            hrf_designs_dir.mkdir(parents=True, exist_ok=True)

            hrf_design_files = []
            for hrf_idx in range(n_hrfs):
                hrf = results.hrf_library[hrf_idx]

                # Convolve onsets with this HRF
                stim_design = convolve_hrf_microtime(
                    onsets,
                    hrf,
                    n_timepoints,
                    tr=tr,
                    microtime_dt=microtime_dt,
                    microtime_onset=microtime_onset,
                )

                # Build full design: [stimulus | polynomials]
                full_design = torch.cat([stim_design, poly_design], dim=1)
                design_np = full_design.cpu().numpy()
                n_stim_cols = stim_design.shape[1]

                # Create descriptive filename (1-based: hrf01, hrf02, ..., hrf20)
                # Use hrf_idx + 1 for 1-based naming (0 = background in AFNI)
                hrf_num = hrf_idx + 1
                hrf_design_file = hrf_designs_dir / f"hrf{hrf_num:02d}_{hrf_mode}.xmat.1D"
                # Get extra nuisance labels from metadata
                extra_nuisance_labels = results.hrf_metadata.get("extra_nuisance_labels", [])
                _write_afni_xmat(
                    design_np,
                    str(hrf_design_file),
                    n_stim_cols,
                    condition_labels,
                    run_starts_local,
                    tr,
                    polort=polort_val,
                    extra_nuisance_labels=extra_nuisance_labels,
                )
                hrf_design_files.append(str(hrf_design_file))

            output_files["hrf_designs_dir"] = str(hrf_designs_dir)
            output_files["hrf_design_files"] = hrf_design_files

    # 6d. Save design matrix plots if requested
    if save_plots:
        # Create figures directory
        figs_dir = f"{output_prefix}_figures"
        Path(figs_dir).mkdir(parents=True, exist_ok=True)

        tr = results.hrf_metadata.get("tr", 1.0)

        # Determine n_stim_cols for labeling
        n_conditions_meta = results.hrf_metadata.get("stim_durations")
        if n_conditions_meta is not None:
            n_stim_cols = len(n_conditions_meta) if isinstance(n_conditions_meta, list) else 1
        elif results.final_results is not None and results.final_results.betas is not None:
            n_stim_cols = results.final_results.betas.shape[1]
        else:
            n_stim_cols = 0

        # Build column labels
        if condition_labels is not None and len(condition_labels) >= n_stim_cols:
            stim_labels = list(condition_labels[:n_stim_cols])
        else:
            stim_labels = [f"stim{i:02d}" for i in range(n_stim_cols)]

        # Plot optimized design matrix
        if results.design_matrix is not None:
            n_timepoints = results.design_matrix.shape[0]
            n_cols = results.design_matrix.shape[1]
            n_nuisance = n_cols - n_stim_cols
            nuisance_labels = [f"poly{i:02d}" for i in range(n_nuisance)]
            all_labels = stim_labels + nuisance_labels

            design_plot_file = f"{figs_dir}/design.png"
            plot_design_matrix(
                results.design_matrix,
                output_file=design_plot_file,
                column_labels=all_labels,
                tr=tr,
                title="Optimized HRF Design Matrix",
                run_starts=run_starts,
            )
            output_files["design_plot"] = design_plot_file

        # Plot canonical design matrix
        if results.canonical_design_matrix is not None:
            n_cols = results.canonical_design_matrix.shape[1]
            n_nuisance = n_cols - n_stim_cols
            nuisance_labels = [f"poly{i:02d}" for i in range(n_nuisance)]
            all_labels = stim_labels + nuisance_labels

            canonical_plot_file = f"{figs_dir}/canonical_design.png"
            plot_design_matrix(
                results.canonical_design_matrix,
                output_file=canonical_plot_file,
                column_labels=all_labels,
                tr=tr,
                title="Canonical HRF Design Matrix",
                run_starts=run_starts,
            )
            output_files["canonical_design_plot"] = canonical_plot_file

        # Plot HRF library
        if results.hrf_library is not None:
            hrf_plot_file = f"{figs_dir}/hrf_library.png"
            plot_hrf_library(
                results.hrf_library,
                output_file=hrf_plot_file,
                tr=tr,
                title="HRF Library",
            )
            output_files["hrf_library_plot"] = hrf_plot_file

    # 7. Save metadata JSON
    metadata_file = f"{output_prefix}_metadata.json"
    metadata = results.hrf_metadata.copy()
    metadata["output_files"] = {k: str(v) for k, v in output_files.items()}
    metadata["hrf_library_shape"] = list(results.hrf_library.shape)
    if results.design_matrix is not None:
        metadata["design_matrix_shape"] = list(results.design_matrix.shape)
        metadata["reference_hrf_idx"] = results.hrf_metadata.get("n_hrfs", 0) // 2

    # Add canonical baseline comparison statistics
    if results.xval_r2_canonical is not None:
        metadata["xval_r2_canonical_mean"] = float(results.xval_r2_canonical.mean().item())
        metadata["xval_r2_best_mean"] = float(results.xval_r2_best.mean().item())
        metadata["r2_improvement_over_canonical"] = float(
            results.xval_r2_best.mean().item() - results.xval_r2_canonical.mean().item()
        )

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    output_files["metadata"] = metadata_file

    return output_files


def load_hrf_selection_for_arma(hrf_library_file: str) -> dict:
    """
    Load HRF selection results for ARMA analysis.

    This allows running ARMA/REML with the previously selected HRFs,
    without re-running the CV selection process.

    Parameters
    ----------
    hrf_library_file : str
        Path to {prefix}_hrf_library.pt file

    Returns
    -------
    hrf_data : dict
        Contains:
        - hrf_library: (n_hrfs, n_timepoints) HRF shapes
        - hrf_index: (n_voxels,) selected HRF per voxel
        - hrf_group_indices: dict mapping HRF index to voxel indices
        - metadata: selection parameters
    """
    return torch.load(hrf_library_file, weights_only=False)


def _save_volume(
    data: torch.Tensor,
    filepath: str,
    volume_shape: tuple[int, int, int] | None,
    affine: np.ndarray | None,
    voxel_mask: torch.Tensor | None,
):
    """Helper to save a 1D tensor as a 3D NIfTI volume."""
    import nibabel as nib

    data_np = data.cpu().numpy()

    if volume_shape is not None:
        if voxel_mask is not None:
            # Unmask data
            full_volume = np.zeros(np.prod(volume_shape), dtype=np.float32)
            full_volume[voxel_mask.cpu().numpy()] = data_np
            volume_data = full_volume.reshape(volume_shape)
        else:
            volume_data = data_np.reshape(volume_shape)
    else:
        # Save as 1D
        volume_data = data_np

    if affine is None:
        affine = np.eye(4)

    img = nib.Nifti1Image(volume_data.astype(np.float32), affine)
    nib.save(img, filepath)


def _save_volume_4d(
    data: torch.Tensor,
    filepath: str,
    volume_shape: tuple[int, int, int] | None,
    affine: np.ndarray | None,
    voxel_mask: torch.Tensor | None,
):
    """Helper to save a 2D tensor (n_voxels, n_volumes) as a 4D NIfTI volume."""
    import nibabel as nib

    data_np = data.cpu().numpy()  # (n_voxels, n_volumes)
    n_volumes = data_np.shape[1]

    if volume_shape is not None:
        if voxel_mask is not None:
            # Unmask data: create (x, y, z, n_volumes) array
            mask_np = voxel_mask.cpu().numpy()
            full_volume = np.zeros((np.prod(volume_shape), n_volumes), dtype=np.float32)
            full_volume[mask_np, :] = data_np
            volume_data = full_volume.reshape((*volume_shape, n_volumes))
        else:
            volume_data = data_np.reshape((*volume_shape, n_volumes))
    else:
        # Save as 2D (voxels x volumes)
        volume_data = data_np

    if affine is None:
        affine = np.eye(4)

    img = nib.Nifti1Image(volume_data.astype(np.float32), affine)
    nib.save(img, filepath)


def plot_design_matrix(
    design_matrix: torch.Tensor | np.ndarray,
    output_file: str | None = None,
    column_labels: list[str] | None = None,
    tr: float = 1.0,
    title: str = "Design Matrix",
    figsize: tuple[float, float] = (10, 8),
    cmap: str = "RdBu_r",
    show_colorbar: bool = True,
    run_starts: list[int] | None = None,
) -> None:
    """
    Plot a design matrix in imagesc style.

    Parameters
    ----------
    design_matrix : torch.Tensor or np.ndarray
        (n_timepoints, n_columns) design matrix
    output_file : str, optional
        Path to save the figure. If None, displays interactively.
    column_labels : list of str, optional
        Labels for each column
    tr : float, default=1.0
        TR in seconds (for y-axis time labels)
    title : str, default="Design Matrix"
        Plot title
    figsize : tuple, default=(10, 8)
        Figure size in inches
    cmap : str, default="RdBu_r"
        Colormap name
    show_colorbar : bool, default=True
        Whether to show colorbar
    run_starts : list of int, optional
        Starting timepoint for each run (draws horizontal lines)
    """
    import matplotlib.pyplot as plt

    # Convert to numpy
    if isinstance(design_matrix, torch.Tensor):
        design_np = design_matrix.cpu().numpy()
    else:
        design_np = np.asarray(design_matrix)

    n_timepoints, n_cols = design_np.shape

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Determine symmetric color limits for diverging colormap
    vmax = np.abs(design_np).max()
    vmin = -vmax

    # Plot
    im = ax.imshow(
        design_np,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # Add colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Regressor Value")

    # Set labels
    ax.set_xlabel("Regressor")
    ax.set_ylabel("Time (s)")
    ax.set_title(title)

    # X-axis: column labels
    if column_labels is not None and len(column_labels) == n_cols:
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(column_labels, rotation=45, ha="right", fontsize=8)
    else:
        # Just show column indices
        if n_cols <= 20:
            ax.set_xticks(range(n_cols))
        else:
            ax.set_xticks(np.linspace(0, n_cols - 1, min(10, n_cols)).astype(int))

    # Y-axis: time in seconds
    n_yticks = min(10, n_timepoints)
    ytick_indices = np.linspace(0, n_timepoints - 1, n_yticks).astype(int)
    ytick_labels = [f"{idx * tr:.0f}" for idx in ytick_indices]
    ax.set_yticks(ytick_indices)
    ax.set_yticklabels(ytick_labels)

    # Draw horizontal lines at run boundaries
    if run_starts is not None and len(run_starts) > 1:
        for run_start in run_starts[1:]:  # Skip first (0)
            ax.axhline(y=run_start - 0.5, color="black", linewidth=1.5, linestyle="--")

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_hrf_library(
    hrf_library: torch.Tensor | np.ndarray,
    output_file: str | None = None,
    tr: float = 1.0,
    title: str = "HRF Library",
    figsize: tuple[float, float] = (10, 6),
    highlight_idx: int | None = None,
) -> None:
    """
    Plot all HRFs in a library as overlaid curves.

    Parameters
    ----------
    hrf_library : torch.Tensor or np.ndarray
        (n_hrfs, n_timepoints) HRF library
    output_file : str, optional
        Path to save the figure. If None, displays interactively.
    tr : float, default=1.0
        TR in seconds (for x-axis time labels)
    title : str, default="HRF Library"
        Plot title
    figsize : tuple, default=(10, 6)
        Figure size in inches
    highlight_idx : int, optional
        Index of HRF to highlight (thicker line)
    """
    import matplotlib.pyplot as plt

    # Convert to numpy
    if isinstance(hrf_library, torch.Tensor):
        hrf_np = hrf_library.cpu().numpy()
    else:
        hrf_np = np.asarray(hrf_library)

    n_hrfs, n_timepoints = hrf_np.shape
    time = np.arange(n_timepoints) * tr

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each HRF (labels use 1-based indexing for AFNI compatibility)
    cmap = plt.cm.viridis
    for i in range(n_hrfs):
        color = cmap(i / (n_hrfs - 1)) if n_hrfs > 1 else cmap(0.5)
        linewidth = 2.5 if i == highlight_idx else 1.0
        alpha = 1.0 if i == highlight_idx else 0.6
        ax.plot(
            time,
            hrf_np[i],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=f"HRF {i + 1}",
        )

    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Response")
    ax.set_title(title)

    # Add legend if not too many HRFs
    if n_hrfs <= 10:
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
