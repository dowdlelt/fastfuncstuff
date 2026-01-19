"""
Cross-validated HRF selection per voxel

This module provides the core engine for selecting the optimal HRF per voxel
using cross-validation across runs. Unlike in-sample selection which can overfit,
CV-based selection provides a more reliable estimate of which HRF shape best
captures the true hemodynamic response for each voxel.

Key function: fit_glm_hrf_library_with_xval()
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm.auto import tqdm

from .design import convolve_hrf, convolve_hrf_microtime
from .glm_core import GLMResults, construct_polynomial_matrix, fit_glm
from .hrf import get_hrf_library
from .utils import get_device, to_tensor
from .xval import compute_xval_r2, generate_cv_splits


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
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Filter out comment lines (AFNI 1D format uses # for comments/headers)
    data_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        data_lines.append(line)

    if not data_lines:
        raise ValueError(f"Nuisance file {filepath} is empty or contains only comments")

    # Detect delimiter: try comma first, then whitespace
    first_line = data_lines[0]
    if ',' in first_line:
        delimiter = ','
    elif '\t' in first_line:
        delimiter = None  # np.loadtxt handles tabs with default
    else:
        delimiter = None  # whitespace

    # Parse data
    try:
        if delimiter == ',':
            data = np.array([
                [float(x.strip()) for x in line.split(',')]
                for line in data_lines
            ])
        else:
            data = np.array([
                [float(x) for x in line.split()]
                for line in data_lines
            ])
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
    microtime_resolution: int = 20,
    microtime_onset: Optional[int] = None,
    polort: Optional[int] = None,
    ortvec_files: Optional[List[Tuple[Union[str, Path], str]]] = None,
    extra_regressors: Optional[Union[np.ndarray, torch.Tensor]] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    chunk_size: Optional[int] = None,
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
        (n_timepoints, n_conditions) or (n_microtime, n_conditions) binary onset matrix
        If microtime_resolution > 1, should be at microtime resolution
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
    microtime_resolution : int, default=20
        Sub-TR resolution for onset timing
    microtime_onset : int, optional
        Which microtime bin to sample (default: 1, no shift)
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

    # Set microtime_onset default: first bin (no shift, events at actual times)
    if microtime_onset is None:
        microtime_onset = 1

    # Move data to device
    data = to_tensor(data, device=device)
    onsets = to_tensor(onsets, device=device)
    hrf_library = to_tensor(hrf_library, device=device)

    n_voxels, n_timepoints_data = data.shape
    n_hrfs = hrf_library.shape[0]
    n_runs = len(run_starts)

    # Determine n_timepoints at TR resolution
    if microtime_resolution > 1:
        n_timepoints = onsets.shape[0] // microtime_resolution
    else:
        n_timepoints = onsets.shape[0]

    # Validate data/design alignment
    if n_timepoints_data != n_timepoints:
        raise ValueError(
            f"Data has {n_timepoints_data} timepoints but design implies {n_timepoints}. "
            f"Check microtime_resolution setting."
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
        print(f"  Microtime: {microtime_resolution}x (onset bin {microtime_onset})")
        print()

    # Generate CV splits
    cv_splits = generate_cv_splits(n_runs, strategy=cv_strategy, n_perms=n_perms)
    n_splits = len(cv_splits)

    if verbose:
        print(f"  CV splits: {n_splits}")
        print()

    # =========================================================================
    # Build polynomial (nuisance) design matrix - same for all HRFs
    # Uses block-diagonal structure: each run gets its own polynomials
    # This is the SAME approach as fit_glm and compute_xval_r2 expect
    # =========================================================================

    # Auto-compute polort if not specified (AFNI convention: duration_minutes / 2)
    if polort is None:
        run_lengths = []
        for i in range(n_runs):
            if i < n_runs - 1:
                run_lengths.append(run_starts[i + 1] - run_starts[i])
            else:
                run_lengths.append(n_timepoints - run_starts[i])
        avg_run_duration_min = (sum(run_lengths) / n_runs * tr) / 60.0
        polort = max(1, round(avg_run_duration_min / 2))
        if verbose:
            print(
                f"  Auto polort: {polort} (based on {avg_run_duration_min:.1f} min avg run)"
            )

    # Build block-diagonal polynomial matrix for all runs
    poly_blocks = []
    run_lengths = []
    for i in range(n_runs):
        if i < n_runs - 1:
            run_len = run_starts[i + 1] - run_starts[i]
        else:
            run_len = n_timepoints - run_starts[i]
        run_lengths.append(run_len)
        poly_block = construct_polynomial_matrix(run_len, polort, device)
        poly_blocks.append(poly_block)

    # Create block-diagonal polynomial design
    poly_design = torch.block_diag(*poly_blocks)  # (n_timepoints, n_runs * (polort+1))
    n_poly_cols = poly_design.shape[1]

    # =========================================================================
    # Load and concatenate additional nuisance regressors (motion, physio, etc.)
    # These are passed via ortvec_files (file paths) or extra_regressors (arrays)
    # =========================================================================
    nuisance_design = poly_design  # Start with polynomials
    n_extra_nuisance = 0
    extra_nuisance_labels = []

    # Load ortvec files (AFNI-style nuisance regressors)
    if ortvec_files is not None:
        for filepath, label in ortvec_files:
            nuisance_data = load_nuisance_file(filepath, expected_rows=n_timepoints)
            n_cols_loaded = nuisance_data.shape[1]
            n_extra_nuisance += n_cols_loaded
            extra_nuisance_labels.extend([f"{label}_{i}" for i in range(n_cols_loaded)])

            # Convert to tensor and concatenate
            nuisance_tensor = torch.tensor(nuisance_data, dtype=torch.float32, device=device)
            nuisance_design = torch.cat([nuisance_design, nuisance_tensor], dim=1)

            if verbose:
                print(f"  Loaded nuisance: {filepath} ({n_cols_loaded} columns, label={label})")

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

        nuisance_design = torch.cat([nuisance_design, extra_tensor], dim=1)

        if verbose:
            print(f"  Added extra_regressors: {n_cols_extra} columns")

    n_nuisance_cols = nuisance_design.shape[1]

    if verbose:
        if n_extra_nuisance > 0:
            print(
                f"  Total nuisance design: {nuisance_design.shape} "
                f"({n_poly_cols} polort + {n_extra_nuisance} extra)"
            )
        else:
            print(
                f"  Polynomial design: {nuisance_design.shape} ({n_poly_cols} nuisance columns)"
            )
        print()

    # Storage for CV results: (n_voxels, n_hrfs)
    xval_r2_median_all = torch.zeros(n_voxels, n_hrfs, device=device)
    xval_r2_std_all = torch.zeros(n_voxels, n_hrfs, device=device)

    # Store design matrix for debugging (using middle HRF as reference)
    reference_hrf_idx = n_hrfs // 2
    design_matrix_ref = None

    # =========================================================================
    # Main loop: evaluate each HRF via cross-validation using compute_xval_r2
    # This reuses the SOLID xval code from 3dXvalR2fast
    # =========================================================================
    hrf_iterator = tqdm(range(n_hrfs), desc="Evaluating HRF candidates")

    for hrf_idx in hrf_iterator:
        hrf = hrf_library[hrf_idx]

        # Convolve onsets with this HRF to get stimulus design
        if microtime_resolution > 1:
            stim_design = convolve_hrf_microtime(
                onsets,
                hrf,
                n_timepoints,
                microtime_resolution,
                tr=tr,
                microtime_onset=microtime_onset,
                device=device,
            )
        else:
            stim_design = convolve_hrf(onsets, hrf, n_timepoints, device=device)

        n_stim_cols = stim_design.shape[1]

        # Build full design: [stimulus | nuisance (polort + extra)]
        # Stimulus columns come first (stim_indices = 0:n_stim)
        # Nuisance columns come after (nuisance_indices = n_stim:end)
        full_design = torch.cat([stim_design, nuisance_design], dim=1)

        stim_indices = list(range(n_stim_cols))
        nuisance_indices = list(range(n_stim_cols, n_stim_cols + n_nuisance_cols))

        # Store reference design matrix for debugging
        if hrf_idx == reference_hrf_idx:
            design_matrix_ref = full_design.cpu().clone()

        # Use the SOLID xval code that already handles all edge cases
        xval_results = compute_xval_r2(
            data=data,
            design_matrix=full_design,
            run_starts=run_starts,
            stim_indices=stim_indices,
            nuisance_indices=nuisance_indices,
            cv_splits=cv_splits,
            metric=metric,
            zero_event_strategy="zero",
            device=device,
            batch_size=chunk_size,
            verbose=False,  # Don't spam per-HRF output
        )

        # Store results for this HRF
        xval_r2_median_all[:, hrf_idx] = xval_results["r2_median"].to(device)
        xval_r2_std_all[:, hrf_idx] = xval_results["r2_std"].to(device)

    # =========================================================================
    # Compute canonical HRF baseline for comparison
    # This uses a single SPM-style canonical HRF as baseline
    # =========================================================================
    if verbose:
        print()
        print("Computing canonical HRF baseline for comparison...")

    # Get single canonical HRF
    mean_duration = float(np.mean(stim_durations)) if stim_durations else 0.0
    canonical_hrf = get_hrf_library(
        mode="single",
        stim_duration=mean_duration,
        tr=tr,
        device=device,
    )

    # Convolve with canonical HRF
    if microtime_resolution > 1:
        canonical_design = convolve_hrf_microtime(
            onsets,
            canonical_hrf,
            n_timepoints,
            microtime_resolution,
            tr=tr,
            microtime_onset=microtime_onset,
            device=device,
        )
    else:
        canonical_design = convolve_hrf(
            onsets, canonical_hrf, n_timepoints, device=device
        )

    # Build full design with nuisance (polort + extra)
    full_design_canonical = torch.cat([canonical_design, nuisance_design], dim=1)
    stim_indices_canonical = list(range(canonical_design.shape[1]))
    nuisance_indices_canonical = list(
        range(canonical_design.shape[1], full_design_canonical.shape[1])
    )

    # Compute xval R² for canonical HRF
    canonical_xval_results = compute_xval_r2(
        data=data,
        design_matrix=full_design_canonical,
        run_starts=run_starts,
        stim_indices=stim_indices_canonical,
        nuisance_indices=nuisance_indices_canonical,
        cv_splits=cv_splits,
        metric=metric,
        zero_event_strategy="zero",
        device=device,
        batch_size=chunk_size,
        verbose=False,
    )
    xval_r2_canonical = canonical_xval_results["r2_median"].to(device)

    if verbose:
        print(f"  Canonical HRF mean xval R²: {xval_r2_canonical.mean().item():.4f}")

    # Fit full dataset with canonical HRF to get betas/tstats for comparison
    if verbose:
        print("  Fitting full dataset with canonical HRF...")

    # Use fit_glm with the proper signature: (data, design, tr, ...)
    # full_design_canonical already has poly columns, so set max_poly_degree=0 (no extra polys)
    canonical_glm_results = fit_glm(
        data=data,
        design=full_design_canonical,  # Already includes stimulus + polynomial columns
        tr=tr,
        max_poly_degree=0,  # No additional polynomials - already in design
        device=device,
        verbose=False,
        task_indices=stim_indices_canonical,  # Mark which are task regressors
    )

    # Store the canonical design matrix for saving
    canonical_design_matrix = full_design_canonical

    if verbose:
        print(
            f"  Canonical HRF full-data R²: {canonical_glm_results.r2.mean().item():.4f}"
        )

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
        microtime_resolution=microtime_resolution,
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
        "microtime_resolution": microtime_resolution,
        "microtime_onset": microtime_onset,
        "polort": polort,
        "n_poly_cols": n_poly_cols,
        "n_extra_nuisance": n_extra_nuisance,
        "n_nuisance_total": n_nuisance_cols,
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
    microtime_resolution: int,
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

    if microtime_resolution > 1:
        n_timepoints = onsets.shape[0] // microtime_resolution
    else:
        n_timepoints = onsets.shape[0]

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

        # Get data for this group
        group_data = data[voxel_indices, :]  # (n_group_voxels, n_timepoints)

        # Convolve with this HRF
        hrf = hrf_library[hrf_idx_int]

        if microtime_resolution > 1:
            stim_design = convolve_hrf_microtime(
                onsets,
                hrf,
                n_timepoints,
                microtime_resolution,
                tr=tr,
                microtime_onset=microtime_onset,
                device=device,
            )
        else:
            stim_design = convolve_hrf(onsets, hrf, n_timepoints, device=device)

        # Build full design: [stimulus | nuisance]
        full_design = torch.cat([stim_design, nuisance_design], dim=1)
        stim_indices = list(range(n_conditions))

        # Fit GLM for this group (no additional polys - already in nuisance_design)
        group_results = fit_glm(
            group_data,
            full_design,
            tr=tr,
            max_poly_degree=0,  # Nuisance already included
            device=device,
            verbose=False,
            task_indices=stim_indices,
        )

        # Store dof from first group (same for all groups with same design structure)
        if stored_dof is None and group_results.dof is not None:
            stored_dof = group_results.dof

        # Store results (with None checks for type safety)
        if group_results.betas is not None:
            all_betas[voxel_indices, :] = group_results.betas
        if group_results.r2 is not None:
            all_r2[voxel_indices] = group_results.r2

        if group_results.tstats is not None:
            all_tstats[voxel_indices, :] = group_results.tstats
        if group_results.sigma2 is not None:
            all_sigma2[voxel_indices] = group_results.sigma2

    # Build GLMResults
    results = GLMResults()
    results.betas = all_betas.cpu()
    results.r2 = all_r2.cpu()
    results.tstats = all_tstats.cpu()
    results.sigma2 = all_sigma2.cpu()
    results.meanvol = data.mean(dim=1).cpu()
    results.dof = stored_dof  # Propagate dof from fit_glm for 3drefit labeling

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
            if (
                results.final_results is not None
                and results.final_results.betas is not None
            ):
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
            microtime_resolution = results.hrf_metadata.get("microtime_resolution", 1)
            microtime_onset = results.hrf_metadata.get("microtime_onset", 1)
            n_hrfs = results.hrf_library.shape[0]
            hrf_mode = results.hrf_metadata.get("hrf_mode", "library")

            # Determine n_timepoints from design matrix or onsets
            if results.design_matrix is not None:
                n_timepoints = results.design_matrix.shape[0]
            else:
                n_timepoints = (
                    onsets.shape[0] // microtime_resolution
                    if microtime_resolution > 1
                    else onsets.shape[0]
                )

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
                poly_block = construct_polynomial_matrix(
                    run_len, polort_val, onsets.device
                )
                poly_blocks.append(poly_block)
            poly_design = torch.block_diag(*poly_blocks)

            # Create output directory for HRF designs
            hrf_designs_dir = Path(f"{output_prefix}_hrf_designs")
            hrf_designs_dir.mkdir(parents=True, exist_ok=True)

            hrf_design_files = []
            for hrf_idx in range(n_hrfs):
                hrf = results.hrf_library[hrf_idx]

                # Convolve onsets with this HRF
                if microtime_resolution > 1:
                    stim_design = convolve_hrf_microtime(
                        onsets,
                        hrf,
                        n_timepoints,
                        microtime_resolution,
                        tr=tr,
                        microtime_onset=microtime_onset,
                    )
                else:
                    stim_design = convolve_hrf(onsets, hrf, n_timepoints)

                # Build full design: [stimulus | polynomials]
                full_design = torch.cat([stim_design, poly_design], dim=1)
                design_np = full_design.cpu().numpy()
                n_stim_cols = stim_design.shape[1]

                # Create descriptive filename (1-based: hrf01, hrf02, ..., hrf20)
                # Use hrf_idx + 1 for 1-based naming (0 = background in AFNI)
                hrf_num = hrf_idx + 1
                hrf_design_file = (
                    hrf_designs_dir / f"hrf{hrf_num:02d}_{hrf_mode}.xmat.1D"
                )
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
            n_stim_cols = (
                len(n_conditions_meta) if isinstance(n_conditions_meta, list) else 1
            )
        elif (
            results.final_results is not None
            and results.final_results.betas is not None
        ):
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
        metadata["xval_r2_canonical_mean"] = float(
            results.xval_r2_canonical.mean().item()
        )
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
