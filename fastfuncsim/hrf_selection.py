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
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm.auto import tqdm

from .design import convolve_hrf_microtime
from .glm_core import GLMResults, construct_polynomial_matrix, fit_glm
from .hrf import get_hrf_library
from .memory import estimate_chunk_size
from .utils import get_device, to_tensor
from .xval import (
    compute_xval_r2,
    generate_cv_splits,
    project_out_nuisance_per_run,
)


def load_nuisance_file(
    filepath: Union[str, Path],
    expected_rows: Optional[int] = None,
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
    final_results: GLMResults = None
    canonical_results: GLMResults = None  # Full GLM with canonical HRF for comparison
    hrf_library: torch.Tensor = None
    hrf_metadata: Dict = field(default_factory=dict)

    # For ARMA integration: store the convolved design per HRF group
    # This allows reloading without reconvolving
    hrf_group_indices: Dict[int, torch.Tensor] = field(default_factory=dict)

    # Store design matrix (convolved with middle HRF) for debugging
    design_matrix: Optional[torch.Tensor] = None

    # Store canonical HRF design matrix for comparison/saving
    canonical_design_matrix: Optional[torch.Tensor] = None


def fit_glm_hrf_library_with_xval(
    data: torch.Tensor,
    onsets: torch.Tensor,
    hrf_library: torch.Tensor,
    tr: float,
    run_starts: List[int],
    stim_durations: Optional[List[float]] = None,
    cv_strategy: Union[float, int] = 1,
    n_perms: int = 100,
    metric: str = "cod",
    microtime_dt: float = 0.1,
    microtime_onset: int = 0,
    polort: Optional[int] = None,
    ortvec_files: Optional[List[Tuple[Union[str, Path], str]]] = None,
    extra_regressors: Optional[Union[np.ndarray, torch.Tensor]] = None,
    canonical_mode: str = "spmg1",
    device: Optional[torch.device] = None,
    verbose: bool = True,
    chunk_size: Optional[int] = None,
    r2_method: str = "auto",
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

    # Convert data to tensor but decide whether to keep on CPU based on size
    # For large datasets, keep data on CPU and stream chunks to GPU
    # This prevents OOM when data exceeds GPU memory
    data = to_tensor(data, device="cpu")  # Always start on CPU
    onsets = to_tensor(onsets, device=device)  # Small - can go to GPU
    hrf_library = to_tensor(hrf_library, device=device)  # Small - can go to GPU

    # Estimate memory requirement for data
    n_voxels_check, n_timepoints_check = data.shape
    data_size_gb = (n_voxels_check * n_timepoints_check * 4) / (1024**3)

    # GPU memory threshold - keep on CPU if data exceeds this
    gpu_memory_threshold_gb = 4.0
    keep_data_on_cpu = device.type == "cuda" and data_size_gb > gpu_memory_threshold_gb

    if not keep_data_on_cpu:
        # Small enough to fit on GPU - move it there
        data = data.to(device)

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
        if keep_data_on_cpu:
            print(
                f"  Memory mode: CPU streaming (data {data_size_gb:.1f}GB > {gpu_memory_threshold_gb}GB threshold)"
            )
        print()

    # Generate CV splits
    cv_splits = generate_cv_splits(n_runs, strategy=cv_strategy, n_perms=n_perms)
    n_splits = len(cv_splits)

    if verbose:
        print(f"  CV splits: {n_splits}")
        print()

    # =========================================================================
    # Build per-run nuisance blocks (polynomials + extra regressors)
    # Using PROJECT-FIRST approach (GLMdenoise-style)
    # =========================================================================

    # Auto-compute polort if not specified (AFNI formula)
    if polort is None:
        from .cli_utils import auto_polort, compute_run_lengths, get_average_run_duration
        run_lengths = compute_run_lengths(run_starts, n_timepoints)
        avg_run_duration_sec = get_average_run_duration(run_lengths, tr)
        polort = auto_polort(avg_run_duration_sec, formula="afni")
        if verbose:
            print(f"  Auto polort: {polort} (AFNI formula, {avg_run_duration_sec:.0f}s avg run)")

    # Build nuisance blocks per run (for project-first approach)
    nuisance_blocks_per_run = []
    run_lengths = []
    for i in range(n_runs):
        if i < n_runs - 1:
            run_len = run_starts[i + 1] - run_starts[i]
        else:
            run_len = n_timepoints - run_starts[i]
        run_lengths.append(run_len)

        # Start with polynomials
        poly_block = construct_polynomial_matrix(run_len, polort, device)
        nuisance_blocks_per_run.append(poly_block)

    n_poly_cols = nuisance_blocks_per_run[0].shape[1]

    # =========================================================================
    # Load and add additional nuisance regressors (motion, physio, etc.)
    # Split by run and concatenate with polynomial blocks
    # =========================================================================
    n_extra_nuisance = 0
    extra_nuisance_labels = []

    # Load ortvec files and split by run
    if ortvec_files is not None:
        for filepath, label in ortvec_files:
            nuisance_data = load_nuisance_file(filepath, expected_rows=n_timepoints)
            n_cols_loaded = nuisance_data.shape[1]
            n_extra_nuisance += n_cols_loaded
            extra_nuisance_labels.extend([f"{label}_{i}" for i in range(n_cols_loaded)])

            if verbose:
                print(f"  Loaded nuisance: {filepath} ({n_cols_loaded} columns, label={label})")

            # Split by run and add to nuisance blocks
            for run_idx in range(n_runs):
                start_tp = run_starts[run_idx]
                end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
                run_nuisance = torch.tensor(
                    nuisance_data[start_tp:end_tp, :], dtype=torch.float32, device=device
                )
                nuisance_blocks_per_run[run_idx] = torch.cat(
                    [nuisance_blocks_per_run[run_idx], run_nuisance], dim=1
                )

    # Add extra_regressors if provided directly as array
    if extra_regressors is not None:
        if isinstance(extra_regressors, np.ndarray):
            extra_tensor = torch.tensor(extra_regressors, dtype=torch.float32, device=device)
        else:
            extra_tensor = extra_regressors.to(device=device, dtype=torch.float32)

        # Validate shape
        if extra_tensor.shape[0] != n_timepoints:
            raise ValueError(
                f"extra_regressors has {extra_tensor.shape[0]} rows, "
                f"expected {n_timepoints} (total timepoints)"
            )

        # Ensure 2D
        if extra_tensor.ndim == 1:
            extra_tensor = extra_tensor.unsqueeze(1)

        n_cols_extra = extra_tensor.shape[1]
        n_extra_nuisance += n_cols_extra
        extra_nuisance_labels.extend([f"extra_{i}" for i in range(n_cols_extra)])

        if verbose:
            print(f"  Added extra_regressors: {n_cols_extra} columns")

        # Split by run and add to nuisance blocks
        for run_idx in range(n_runs):
            start_tp = run_starts[run_idx]
            end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
            run_extra = extra_tensor[start_tp:end_tp, :]
            nuisance_blocks_per_run[run_idx] = torch.cat(
                [nuisance_blocks_per_run[run_idx], run_extra], dim=1
            )

    n_nuisance_cols_per_run = nuisance_blocks_per_run[0].shape[1]

    if verbose:
        if n_extra_nuisance > 0:
            print(
                f"  Nuisance per run: {n_nuisance_cols_per_run} columns "
                f"({n_poly_cols} polort + {n_extra_nuisance} extra)"
            )
        else:
            print(f"  Polynomial nuisance per run: {n_nuisance_cols_per_run} columns")
        print()

    # =========================================================================
    # Project out nuisance from DATA once (same for all HRFs)
    # This is the GLMdenoise PROJECT-FIRST approach
    # =========================================================================
    # Clear GPU cache before major allocation to reduce fragmentation
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if verbose:
        print("Projecting out nuisance from data (once for all HRFs)...")

    # Create a dummy design (we'll project actual stim designs per-HRF)
    # For now just project data
    dummy_design = torch.zeros((n_timepoints, 1), device=device)
    projected_data, _ = project_out_nuisance_per_run(
        data=data,
        design=dummy_design,
        nuisance_per_run=nuisance_blocks_per_run,
        run_starts=run_starts,
        device=device,
        chunk_size=chunk_size,
        verbose=verbose,
    )

    if verbose:
        print(f"  Projected data shape: {projected_data.shape}")
        print()

    # Storage for CV results: (n_voxels, n_hrfs)
    xval_r2_median_all = torch.zeros(n_voxels, n_hrfs, device=device)
    xval_r2_std_all = torch.zeros(n_voxels, n_hrfs, device=device)

    # Store design matrix for debugging (using middle HRF as reference)
    reference_hrf_idx = n_hrfs // 2
    design_matrix_ref = None

    # =========================================================================
    # Main loop: evaluate each HRF via cross-validation
    # Using PROJECT-FIRST approach: project stim design, then run CV
    # =========================================================================
    hrf_iterator = tqdm(range(n_hrfs), desc="Evaluating HRF candidates")

    for hrf_idx in hrf_iterator:
        hrf = hrf_library[hrf_idx]

        # Convolve onsets with this HRF to get stimulus design
        # HRF library and onsets are both at microtime_dt resolution
        stim_design = convolve_hrf_microtime(
            onsets,
            hrf,
            n_timepoints,
            tr=tr,
            microtime_dt=microtime_dt,
            microtime_onset=microtime_onset,
            device=device,
        )

        n_stim_cols = stim_design.shape[1]

        # Project out nuisance from stim design (per run)
        _, projected_stim_design = project_out_nuisance_per_run(
            data=torch.zeros((1, n_timepoints), device=device),  # Dummy data
            design=stim_design,
            nuisance_per_run=nuisance_blocks_per_run,
            run_starts=run_starts,
            device=device,
        )

        # Store reference design matrix for debugging
        if hrf_idx == reference_hrf_idx:
            design_matrix_ref = projected_stim_design.cpu().clone()

        # Run CV on PROJECTED data with PROJECTED stim design
        # No nuisance columns - already projected out!
        xval_results = compute_xval_r2(
            data=projected_data,
            design_matrix=projected_stim_design,
            run_starts=run_starts,
            stim_indices=list(range(n_stim_cols)),  # All columns are stim
            nuisance_indices=[],  # No nuisance - already projected out!
            cv_splits=cv_splits,
            metric=metric,
            zero_event_strategy="zero",
            device=device,
            batch_size=chunk_size,
            r2_method=r2_method,
            verbose=False,  # Don't spam per-HRF output
        )

        # Store results for this HRF (GLMdenoise-style: single R² per voxel)
        xval_r2_median_all[:, hrf_idx] = xval_results["r2"].to(device)

    # Clear GPU cache after HRF evaluation to free fragmented memory
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # =========================================================================
    # Compute canonical HRF baseline for comparison
    # This uses a single SPM-style canonical HRF as baseline
    # =========================================================================
    if verbose:
        print()
        print("Computing canonical HRF baseline for comparison...")

    # Get single canonical HRF as impulse response (stim_duration=0)
    # The onset matrix already encodes stimulus duration via boxcars at microtime
    # resolution, so the HRF should be an impulse response to avoid double-counting
    # the duration (convolving duration-encoded onsets with duration-encoded HRF)
    from .hrf import get_spmg1_hrf

    canonical_mode_lower = canonical_mode.lower()
    if canonical_mode_lower == "spmg1":
        # AFNI's SPMG1 canonical HRF (recommended default)
        canonical_hrf = get_spmg1_hrf(
            microtime_dt=microtime_dt,
            stim_duration=0.0,  # Impulse response - duration is in onset matrix
            normalize_peak=True,
            device=device,
        )
        canonical_label = "SPMG1"
    elif canonical_mode_lower in ("glmsingle", "single"):
        # GLMsingle/nilearn-style double-gamma
        canonical_hrf = get_hrf_library(
            mode="single",
            stim_duration=0.0,  # Impulse response - duration is in onset matrix
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

    # Convolve with canonical HRF
    canonical_design = convolve_hrf_microtime(
        onsets,
        canonical_hrf,
        n_timepoints,
        tr=tr,
        microtime_dt=microtime_dt,
        microtime_onset=microtime_onset,
        device=device,
    )

    # Project out nuisance from canonical design (same as HRF library)
    _, projected_canonical_design = project_out_nuisance_per_run(
        data=torch.zeros((1, n_timepoints), device=device),  # Dummy data
        design=canonical_design,
        nuisance_per_run=nuisance_blocks_per_run,
        run_starts=run_starts,
        device=device,
    )

    # Compute xval R² for canonical HRF (on projected data/design)
    canonical_xval_results = compute_xval_r2(
        data=projected_data,
        design_matrix=projected_canonical_design,
        run_starts=run_starts,
        stim_indices=list(range(canonical_design.shape[1])),  # All columns are stim
        nuisance_indices=[],  # No nuisance - already projected out!
        cv_splits=cv_splits,
        metric=metric,
        zero_event_strategy="zero",
        device=device,
        batch_size=chunk_size,
        r2_method=r2_method,
        verbose=False,
    )
    xval_r2_canonical = canonical_xval_results["r2"].to(device)

    if verbose:
        print(f"  Canonical HRF mean xval R²: {xval_r2_canonical.mean().item():.4f}")

    # Fit full dataset with canonical HRF to get betas/tstats for comparison
    # NOTE: For final fit, we need the full (unprojected) data and design with nuisance
    if verbose:
        print("  Fitting full dataset with canonical HRF...")

    # Build block-diagonal nuisance for final fit
    nuisance_design = torch.block_diag(*nuisance_blocks_per_run)
    full_design_canonical = torch.cat([canonical_design, nuisance_design], dim=1)

    # Use fit_glm with the proper signature: (data, design, tr, ...)
    # full_design_canonical already has poly columns, so set max_poly_degree=0 (no extra polys)
    # If data is on CPU, use chunk-based streaming to avoid OOM
    n_stim_cols_canonical = canonical_design.shape[1]
    canonical_glm_results = fit_glm(
        data=data,
        design=full_design_canonical,  # Already includes stimulus + polynomial columns
        tr=tr,
        max_poly_degree=0,  # No additional polynomials - already in design
        device=device,
        verbose=False,
        task_indices=list(range(n_stim_cols_canonical)),  # Mark which are task regressors
        preload_data_to_device=(data.device == device),  # Stream chunks if data on CPU
    )

    # Store the canonical design matrix for saving
    canonical_design_matrix = full_design_canonical

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
        "n_extra_nuisance": n_extra_nuisance,
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
    tr: float,
    microtime_dt: float,
    microtime_onset: int,
    device: torch.device,
    verbose: bool,
    chunk_size: Optional[int],
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
            device=device,
        )

        # Build full design: [stimulus | nuisance]
        full_design = torch.cat([stim_design, nuisance_design], dim=1)
        stim_indices = list(range(n_conditions))

        # Chunk within HRF group if too large (to avoid OOM)
        # Use smaller chunks for GPU to prevent memory issues
        max_voxels_per_chunk = 50000 if device.type == "cuda" else 100000
        n_chunks = (n_group_voxels + max_voxels_per_chunk - 1) // max_voxels_per_chunk

        if n_chunks > 1:
            # Update progress bar with chunking info
            hrf_iterator.set_postfix_str(f"{n_group_voxels:,} voxels in {n_chunks} chunks")

            # Process HRF group in chunks
            for chunk_idx in range(n_chunks):
                chunk_start = chunk_idx * max_voxels_per_chunk
                chunk_end = min(chunk_start + max_voxels_per_chunk, n_group_voxels)

                # Get voxel indices for this chunk
                chunk_voxel_indices = voxel_indices[chunk_start:chunk_end]

                # Get data for this chunk
                if data.device.type == "cpu" and chunk_voxel_indices.device.type != "cpu":
                    chunk_voxel_indices_for_data = chunk_voxel_indices.cpu()
                    chunk_data = data[chunk_voxel_indices_for_data, :]
                else:
                    chunk_data = data[chunk_voxel_indices, :]

                # Fit GLM for this chunk
                chunk_results = fit_glm(
                    chunk_data,
                    full_design,
                    tr=tr,
                    max_poly_degree=0,  # Nuisance already included
                    device=device,
                    verbose=False,
                    task_indices=stim_indices,
                    preload_data_to_device=(chunk_data.device == device),
                )

                # Store dof from first chunk
                if stored_dof is None and chunk_results.dof is not None:
                    stored_dof = chunk_results.dof

                # Store results
                if chunk_results.betas is not None:
                    betas_to_store = chunk_results.betas
                    if betas_to_store.device != all_betas.device:
                        betas_to_store = betas_to_store.to(all_betas.device)
                    all_betas[chunk_voxel_indices, :] = betas_to_store

                if chunk_results.r2 is not None:
                    r2_to_store = chunk_results.r2
                    if r2_to_store.device != all_r2.device:
                        r2_to_store = r2_to_store.to(all_r2.device)
                    all_r2[chunk_voxel_indices] = r2_to_store

                if chunk_results.tstats is not None:
                    tstats_to_store = chunk_results.tstats
                    if tstats_to_store.device != all_tstats.device:
                        tstats_to_store = tstats_to_store.to(all_tstats.device)
                    all_tstats[chunk_voxel_indices, :] = tstats_to_store

                if chunk_results.sigma2 is not None:
                    sigma2_to_store = chunk_results.sigma2
                    if sigma2_to_store.device != all_sigma2.device:
                        sigma2_to_store = sigma2_to_store.to(all_sigma2.device)
                    all_sigma2[chunk_voxel_indices] = sigma2_to_store

                # Clean up chunk data
                del chunk_data, chunk_results
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        else:
            # Small group - process all at once
            hrf_iterator.set_postfix_str(f"{n_group_voxels:,} voxels")

            # Get data for this group
            if data.device.type == "cpu" and voxel_indices.device.type != "cpu":
                voxel_indices_for_data = voxel_indices.cpu()
                group_data = data[voxel_indices_for_data, :]
            else:
                group_data = data[voxel_indices, :]

            # Fit GLM for this group
            group_results = fit_glm(
                group_data,
                full_design,
                tr=tr,
                max_poly_degree=0,  # Nuisance already included
                device=device,
                verbose=False,
                task_indices=stim_indices,
                preload_data_to_device=(group_data.device == device),
            )

            # Store dof from first group
            if stored_dof is None and group_results.dof is not None:
                stored_dof = group_results.dof

            # Store results
            if group_results.betas is not None:
                betas_to_store = group_results.betas
                if betas_to_store.device != all_betas.device:
                    betas_to_store = betas_to_store.to(all_betas.device)
                all_betas[voxel_indices, :] = betas_to_store

            if group_results.r2 is not None:
                r2_to_store = group_results.r2
                if r2_to_store.device != all_r2.device:
                    r2_to_store = r2_to_store.to(all_r2.device)
                all_r2[voxel_indices] = r2_to_store

            if group_results.tstats is not None:
                tstats_to_store = group_results.tstats
                if tstats_to_store.device != all_tstats.device:
                    tstats_to_store = tstats_to_store.to(all_tstats.device)
                all_tstats[voxel_indices, :] = tstats_to_store

            if group_results.sigma2 is not None:
                sigma2_to_store = group_results.sigma2
                if sigma2_to_store.device != all_sigma2.device:
                    sigma2_to_store = sigma2_to_store.to(all_sigma2.device)
                all_sigma2[voxel_indices] = sigma2_to_store

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

    # Build full design: [stimulus | nuisance]
    full_design = torch.cat([stim_design, nuisance_design], dim=1)
    stim_indices = list(range(n_conditions))

    # Determine chunk size using memory estimation
    # This is the simpler case: one design matrix, condition-level (not single-trial)
    voxel_chunk_size = estimate_chunk_size(
        n_voxels=n_voxels,
        n_timepoints=n_timepoints,
        n_regressors=full_design.shape[1],
        device=device,
        operation="glm",
        min_chunk_size=10000,
        max_chunk_size=50000 if device.type == "cuda" else 100000,
        safety_factor=0.4 if device.type == "cuda" else 0.6,
    )

    n_chunks = (n_voxels + voxel_chunk_size - 1) // voxel_chunk_size
    if verbose and n_chunks > 1:
        print(f"  Processing {n_voxels:,} voxels in {n_chunks} chunks of ~{voxel_chunk_size:,}")

    # Initialize output tensors
    all_betas = torch.zeros(n_voxels, n_conditions, device=device)
    all_r2 = torch.zeros(n_voxels, device=device)
    all_tstats = torch.zeros(n_voxels, n_conditions, device=device)
    all_sigma2 = torch.zeros(n_voxels, device=device)

    # Process in chunks
    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * voxel_chunk_size
        chunk_end = min(chunk_start + voxel_chunk_size, n_voxels)

        chunk_data = data[chunk_start:chunk_end, :]

        # Fit GLM for this chunk
        chunk_results = fit_glm(
            chunk_data,
            full_design,
            tr=tr,
            max_poly_degree=0,  # Nuisance already included
            device=device,
            verbose=False,
            task_indices=stim_indices,
        )

        # Accumulate results
        if chunk_results.betas is not None:
            all_betas[chunk_start:chunk_end, :] = chunk_results.betas

        if chunk_results.r2 is not None:
            all_r2[chunk_start:chunk_end] = chunk_results.r2

        if chunk_results.tstats is not None:
            all_tstats[chunk_start:chunk_end, :] = chunk_results.tstats

        if chunk_results.sigma2 is not None:
            all_sigma2[chunk_start:chunk_end] = chunk_results.sigma2

        # Clean up
        del chunk_data, chunk_results
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Build results object
    results = GLMResults()
    results.betas = all_betas.cpu()
    results.r2 = all_r2.cpu()
    results.tstats = all_tstats.cpu()
    results.sigma2 = all_sigma2.cpu()
    results.meanvol = data.mean(dim=1).cpu()
    results.dof = n_timepoints - full_design.shape[1]  # dof = n - p

    if verbose:
        print(f"  Canonical fit complete. Mean R²: {results.r2.mean().item():.4f}")

    return results


def _fit_voxelwise_hrf_single_trial(
    data: torch.Tensor,
    onsets_by_condition: List[List[np.ndarray]],
    hrf_library: List[torch.Tensor],
    hrf_index: torch.Tensor,
    nuisance_design: torch.Tensor,
    durations: List[float],
    run_starts: List[int],
    tr: float,
    n_timepoints: int,
    microtime_dt: float,
    condition_labels: List[str],
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

    # Build nuisance per-run list for ridge
    nuisance_per_run = []
    for run_idx in range(n_runs):
        run_start = run_starts[run_idx]
        run_end = run_starts[run_idx + 1] if run_idx + 1 < n_runs else n_timepoints
        nuisance_run = nuisance_design[run_start:run_end, :]
        nuisance_per_run.append(nuisance_run)

    # Create CV splits (use simple split-half: [0] vs [1:])
    cv_splits = [([0], list(range(1, n_runs)))]

    # First, get trial info using canonical HRF (just to get n_trials)
    from .ridge import create_single_trial_design
    st_design_canonical, trial_labels, trial_cond_ids, trial_run_ids, condition_design = create_single_trial_design(
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
    n_trials = len(trial_labels)

    if verbose:
        print(f"    Single-trial design: {n_trials} trials")
        print(f"    HRF groups: {len(unique_hrfs)}")

    # Initialize output for single-trial betas
    all_single_trial_betas = torch.zeros(n_voxels, n_trials, device="cpu")
    all_single_trial_r2 = torch.zeros(n_voxels, device="cpu")

    # Process each HRF group
    hrf_iterator = tqdm(unique_hrfs, desc="Refitting HRF groups") if verbose else unique_hrfs.tolist()
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

        # Determine chunk size using memory estimation
        voxel_chunk_size = estimate_chunk_size(
            n_voxels=n_group_voxels,
            n_timepoints=n_timepoints,
            n_regressors=n_trials,
            device=device,
            operation="glm",
            min_chunk_size=10000,
            max_chunk_size=50000 if device.type == "cuda" else 100000,
            safety_factor=0.4 if device.type == "cuda" else 0.6,
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
            st_design_device = st_design.to(device)
            chunk_data_device = chunk_data.to(device)

            # Simple OLS fit: betas = (X'X)^-1 X'y
            # X is (n_timepoints, n_trials), y is (n_voxels, n_timepoints)
            # Solution: betas = y @ X @ (X'X)^-1
            XtX = st_design_device.T @ st_design_device  # (n_trials, n_trials)
            XtX_inv = torch.inverse(XtX)  # (n_trials, n_trials)
            Xty = chunk_data_device @ st_design_device  # (n_voxels, n_trials)
            chunk_betas = Xty @ XtX_inv  # (n_voxels, n_trials)

            # Compute predictions and R²
            # predictions = X @ betas.T = (n_timepoints, n_trials) @ (n_trials, n_voxels)
            # Then transpose to get (n_voxels, n_timepoints)
            chunk_predictions = (st_design_device @ chunk_betas.T).T  # (n_voxels, n_timepoints)
            chunk_r2 = compute_r2_metric(chunk_data_device, chunk_predictions)  # (n_voxels,)

            # Accumulate results
            all_single_trial_betas[chunk_voxel_indices, :] = chunk_betas.cpu()
            all_single_trial_r2[chunk_voxel_indices] = chunk_r2.cpu()

            # Clean up
            del chunk_data, chunk_data_device, st_design_device, chunk_betas, chunk_predictions, chunk_r2
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
    condition_labels: Optional[List[str]],
    run_starts: List[int],
    tr: float,
    polort: Optional[int] = None,
    extra_nuisance_labels: Optional[List[str]] = None,
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


def save_hrf_selection_results(
    results: HRFSelectionResults,
    output_prefix: str,
    volume_shape: Optional[Tuple[int, int, int]] = None,
    affine: Optional[np.ndarray] = None,
    voxel_mask: Optional[torch.Tensor] = None,
    condition_labels: Optional[List[str]] = None,
    run_starts: Optional[List[int]] = None,
    save_all_hrf_designs: bool = False,
    onsets: Optional[torch.Tensor] = None,
    save_plots: bool = False,
) -> Dict[str, str]:
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

            design_plot_file = f"{output_prefix}_design.png"
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

            canonical_plot_file = f"{output_prefix}_canonical_design.png"
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
            hrf_plot_file = f"{output_prefix}_hrf_library.png"
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


def load_hrf_selection_for_arma(hrf_library_file: str) -> Dict:
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
    volume_shape: Optional[Tuple[int, int, int]],
    affine: Optional[np.ndarray],
    voxel_mask: Optional[torch.Tensor],
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
    volume_shape: Optional[Tuple[int, int, int]],
    affine: Optional[np.ndarray],
    voxel_mask: Optional[torch.Tensor],
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
    design_matrix: Union[torch.Tensor, np.ndarray],
    output_file: Optional[str] = None,
    column_labels: Optional[List[str]] = None,
    tr: float = 1.0,
    title: str = "Design Matrix",
    figsize: Tuple[float, float] = (10, 8),
    cmap: str = "RdBu_r",
    show_colorbar: bool = True,
    run_starts: Optional[List[int]] = None,
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
    hrf_library: Union[torch.Tensor, np.ndarray],
    output_file: Optional[str] = None,
    tr: float = 1.0,
    title: str = "HRF Library",
    figsize: Tuple[float, float] = (10, 6),
    highlight_idx: Optional[int] = None,
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
