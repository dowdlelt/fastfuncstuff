"""
Design matrix construction from onset files

This module provides functions to build GLM design matrices from:
- Onset timing files (AFNI format)
- HRF models (SPM canonical, AFNI SPMG1, custom)
- Nuisance regressors (Legendre polynomials, motion parameters)

The goal is to replicate AFNI's 3dDeconvolve design matrix construction
while being easier to use and integrate with our fast GLM fitting.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict
from scipy import special
from scipy.stats import gamma as scipy_gamma


def spm_canonical_hrf(tr: float = 1.0, duration: float = 32.0) -> np.ndarray:
    """
    Create SPM canonical HRF (double gamma function)

    This is the standard HRF used in SPM and AFNI's SPMG1 basis function.

    Parameters
    ----------
    tr : float
        Repetition time in seconds
    duration : float
        Duration of HRF in seconds (default: 32s covers full response)

    Returns
    -------
    hrf : np.ndarray
        HRF sampled at TR intervals, shape (n_timepoints,)

    Notes
    -----
    The canonical HRF is a difference of two gamma functions:
    - Positive peak at ~6s (main response)
    - Negative undershoot at ~16s (post-stimulus undershoot)

    AFNI's SPMG1(d) uses this HRF with duration d for the stimulus.

    References
    ----------
    Glover, G. H. (1999). Deconvolution of impulse response in event-related BOLD fMRI.
    NeuroImage, 9(4), 416-429.
    """
    # Time vector
    t = np.arange(0, duration, tr)

    # Parameters for positive gamma (main response)
    # Peak at ~6s with dispersion
    peak1 = 6.0
    scale1 = 1.0

    # Parameters for negative gamma (undershoot)
    # Peak at ~16s
    peak2 = 16.0
    scale2 = 1.0

    # Compute gamma functions
    # Using scipy.stats.gamma for consistency with SPM
    pos_gamma = scipy_gamma.pdf(t, peak1, scale=scale1)
    neg_gamma = scipy_gamma.pdf(t, peak2, scale=scale2)

    # Combine: positive - (negative / 6)
    # The division by 6 controls the relative size of the undershoot
    hrf = pos_gamma - neg_gamma / 6.0

    # Normalize to unit peak
    hrf = hrf / hrf.max()

    return hrf


def legendre_polynomials(n_timepoints: int, order: int, normalize: bool = False) -> np.ndarray:
    """
    Generate Legendre polynomials for drift modeling

    Creates orthogonal polynomial basis for modeling slow drift in fMRI data.
    This replicates AFNI's -polort option.

    Parameters
    ----------
    n_timepoints : int
        Number of timepoints in the run
    order : int
        Polynomial order (AFNI's -polort value)
        - polort 0: constant (mean)
        - polort 1: constant + linear
        - polort 2: constant + linear + quadratic
        - polort 3: constant + linear + quadratic + cubic (default)
        - polort -1: no polynomials (rarely used)
    normalize : bool, default=False
        If True, normalize each polynomial to unit norm (orthonormal basis)
        If False, return raw Legendre polynomials (orthogonal but not normalized)
        AFNI uses normalize=False

    Returns
    -------
    polynomials : np.ndarray
        Legendre polynomials, shape (n_timepoints, order+1)
        Each column is one polynomial basis function
        If normalize=False (AFNI default), columns are orthogonal but NOT unit norm
        If normalize=True, columns are orthonormal

    Notes
    -----
    AFNI uses shifted Legendre polynomials on interval [-1, 1].
    Unlike many implementations, AFNI does NOT normalize the polynomials.
    This means the Gram matrix X'X has non-unit diagonal values.

    Common usage:
    - polort 3 for runs < 5 minutes
    - polort 4-5 for longer runs

    AFNI auto-determines polort based on run length if not specified:
    polort = 1 + floor(run_length_minutes / 2.5)
    """
    if order < 0:
        # No polynomials
        return np.zeros((n_timepoints, 0))

    # Map timepoints to [-1, 1] interval
    t = np.linspace(-1, 1, n_timepoints)

    # Pre-allocate
    polynomials = np.zeros((n_timepoints, order + 1))

    # Generate each polynomial using scipy.special.eval_legendre
    for p in range(order + 1):
        poly = special.eval_legendre(p, t)

        if normalize:
            # Normalize to unit norm (orthonormal basis)
            poly = poly / np.linalg.norm(poly)

        polynomials[:, p] = poly

    return polynomials


def parse_afni_timing_file(filepath: Union[str, Path]) -> List[np.ndarray]:
    """
    Parse AFNI timing file format

    AFNI timing files have one row per run, with onset times in seconds
    separated by spaces. Empty rows indicate no events in that run.

    Parameters
    ----------
    filepath : str or Path
        Path to timing file

    Returns
    -------
    onsets_by_run : list of np.ndarray
        List with one array per run, containing onset times in seconds
        Empty runs return empty array

    Examples
    --------
    File content:
        12 30 46 63 80
        14 32 49 71 92

    Returns: [array([12, 30, 46, 63, 80]), array([14, 32, 49, 71, 92])]

    File with empty run:
        12 30 46

        14 32 49

    Returns: [array([12, 30, 46]), array([]), array([14, 32, 49])]
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Timing file not found: {filepath}")

    onsets_by_run = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            if not line:
                # Empty line = no events in this run
                onsets_by_run.append(np.array([]))
            else:
                # Parse onset times
                try:
                    onsets = np.array([float(x) for x in line.split()])
                    onsets_by_run.append(onsets)
                except ValueError as e:
                    raise ValueError(f"Could not parse line '{line}' in {filepath}: {e}")

    return onsets_by_run


def create_onset_regressors(
    onset_times: np.ndarray,
    n_timepoints: int,
    tr: float,
    duration: float = 0.0,
    hrf: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create stimulus regressors from onset times

    Converts onset times (in seconds) to a regressor sampled at TR,
    optionally convolved with an HRF.

    Parameters
    ----------
    onset_times : np.ndarray
        Onset times in seconds, shape (n_events,)
    n_timepoints : int
        Total number of timepoints in the run
    tr : float
        Repetition time in seconds
    duration : float, default=0
        Duration of each stimulus in seconds
        - 0: impulse (single TR)
        - >0: boxcar of specified duration
    hrf : np.ndarray, optional
        HRF to convolve with (if None, returns unconfolved regressor)
        Should be sampled at TR intervals

    Returns
    -------
    regressor : np.ndarray
        Stimulus regressor, shape (n_timepoints,)
    """
    # Create stick/boxcar function
    regressor = np.zeros(n_timepoints)

    for onset in onset_times:
        # Convert onset time to TR index
        onset_tr = int(np.round(onset / tr))

        if onset_tr < 0 or onset_tr >= n_timepoints:
            # Skip onsets outside the run
            continue

        if duration <= 0:
            # Impulse (stick function)
            regressor[onset_tr] = 1.0
        else:
            # Boxcar with duration
            duration_tr = int(np.round(duration / tr))
            end_tr = min(onset_tr + duration_tr, n_timepoints)
            regressor[onset_tr:end_tr] = 1.0

    # Convolve with HRF if provided
    if hrf is not None:
        # Use 'full' mode and truncate to match input length
        regressor = np.convolve(regressor, hrf, mode='full')[:n_timepoints]

    return regressor


def build_design_matrix(
    timing_files: List[Union[str, Path]],
    stim_labels: List[str],
    n_timepoints_per_run: List[int],
    tr: float,
    polort: int = 3,
    hrf_model: str = 'SPMG1',
    stim_durations: Optional[List[float]] = None,
    motion_files: Optional[List[Union[str, Path]]] = None,
    extra_regressors: Optional[List[np.ndarray]] = None,
    extra_regressor_labels: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], List[int], Dict]:
    """
    Build complete GLM design matrix from onset files and nuisance regressors

    This function creates a design matrix similar to AFNI's 3dDeconvolve,
    combining stimulus regressors (with HRF convolution) and nuisance regressors
    (polynomials, motion, etc.).

    Parameters
    ----------
    timing_files : list of str/Path
        Paths to AFNI timing files, one per stimulus condition
    stim_labels : list of str
        Labels for each stimulus condition (must match timing_files length)
    n_timepoints_per_run : list of int
        Number of timepoints in each run
    tr : float
        Repetition time in seconds
    polort : int, default=3
        Polynomial order for drift modeling (AFNI -polort)
        Set to -1 for no polynomials
    hrf_model : str, default='SPMG1'
        HRF model to use:
        - 'SPMG1': SPM canonical HRF (default)
        - 'BLOCK': Boxcar (no HRF convolution)
        - 'TENT': Tent function (not yet implemented)
    stim_durations : list of float, optional
        Duration in seconds for each stimulus type
        If None, uses impulse (duration=0) for all
    motion_files : list of str/Path, optional
        Motion parameter files, one per run (AFNI format: 6 columns)
    extra_regressors : list of np.ndarray, optional
        Additional regressors to include (e.g., physio, custom nuisance)
        Each array should be shape (total_timepoints,) or (total_timepoints, n_cols)
    extra_regressor_labels : list of str, optional
        Labels for extra regressors (must match extra_regressors length)

    Returns
    -------
    design_matrix : np.ndarray
        Complete design matrix, shape (total_timepoints, n_regressors)
        Concatenated across runs
    regressor_labels : list of str
        Labels for each column in design matrix
    run_starts : list of int
        Starting indices for each run (for run-based CV)
    metadata : dict
        Additional metadata about the design:
        - 'stim_indices': Indices of stimulus columns
        - 'nuisance_indices': Indices of nuisance columns
        - 'polort_indices': Indices of polynomial columns
        - 'motion_indices': Indices of motion columns (if provided)
        - 'n_runs': Number of runs
        - 'n_timepoints_per_run': Timepoints per run
        - 'tr': TR in seconds

    Examples
    --------
    >>> # Simple 2-run, 2-condition design
    >>> design, labels, run_starts, meta = build_design_matrix(
    ...     timing_files=['movie.txt', 'prompt.txt'],
    ...     stim_labels=['movie', 'prompt'],
    ...     n_timepoints_per_run=[180, 180],
    ...     tr=2.0,
    ...     polort=3
    ... )
    >>> design.shape
    (360, 10)  # 2 stim + 8 polort (4 per run)
    """
    n_runs = len(n_timepoints_per_run)
    n_stim = len(timing_files)

    # Validate inputs
    if len(stim_labels) != n_stim:
        raise ValueError(f"stim_labels length ({len(stim_labels)}) must match timing_files ({n_stim})")

    if stim_durations is None:
        stim_durations = [0.0] * n_stim
    elif len(stim_durations) != n_stim:
        raise ValueError(f"stim_durations length ({len(stim_durations)}) must match n_stim ({n_stim})")

    # Parse timing files
    all_onsets = []
    for tf in timing_files:
        onsets_by_run = parse_afni_timing_file(tf)
        if len(onsets_by_run) != n_runs:
            raise ValueError(
                f"Timing file {tf} has {len(onsets_by_run)} runs, "
                f"but expected {n_runs} (from n_timepoints_per_run)"
            )
        all_onsets.append(onsets_by_run)

    # Create HRF
    if hrf_model == 'SPMG1':
        hrf = spm_canonical_hrf(tr=tr, duration=32.0)
    elif hrf_model == 'BLOCK':
        hrf = None  # No HRF convolution
    else:
        raise ValueError(f"Unknown HRF model: {hrf_model}")

    # Build design matrix in AFNI column order:
    # 1. Polynomial regressors (per-run, in run order)
    # 2. Stimulus regressors (spanning all runs)

    total_timepoints = sum(n_timepoints_per_run)
    run_starts = []
    current_timepoint = 0

    # Pre-allocate full design matrix
    n_polort_cols = (polort + 1) * n_runs if polort >= 0 else 0
    n_total_cols = n_polort_cols + n_stim
    design_matrix = np.zeros((total_timepoints, n_total_cols))

    # Track column indices
    col_idx = 0
    polort_indices = []
    regressor_labels = []

    # 1. Add polynomial regressors (per-run)
    if polort >= 0:
        for run_idx in range(n_runs):
            n_tp = n_timepoints_per_run[run_idx]
            run_start = sum(n_timepoints_per_run[:run_idx])
            run_end = run_start + n_tp
            run_starts.append(run_start)

            # Generate polynomials for this run
            polys = legendre_polynomials(n_tp, polort, normalize=False)

            # Insert into design matrix
            for p in range(polort + 1):
                design_matrix[run_start:run_end, col_idx] = polys[:, p]
                polort_indices.append(col_idx)
                regressor_labels.append(f'Run{run_idx+1}_Poly{p}')
                col_idx += 1
    else:
        # No polynomials, still need run_starts
        for run_idx in range(n_runs):
            run_starts.append(sum(n_timepoints_per_run[:run_idx]))

    # 2. Add stimulus regressors (spanning all runs)
    stim_indices = []
    for stim_idx in range(n_stim):
        # Combine all runs for this stimulus
        stim_regressor = np.zeros(total_timepoints)

        for run_idx in range(n_runs):
            n_tp = n_timepoints_per_run[run_idx]
            run_start = sum(n_timepoints_per_run[:run_idx])
            run_end = run_start + n_tp

            onsets = all_onsets[stim_idx][run_idx]
            duration = stim_durations[stim_idx]

            regressor = create_onset_regressors(
                onset_times=onsets,
                n_timepoints=n_tp,
                tr=tr,
                duration=duration,
                hrf=hrf,
            )

            stim_regressor[run_start:run_end] = regressor

        # Add to design matrix
        design_matrix[:, col_idx] = stim_regressor
        stim_indices.append(col_idx)
        regressor_labels.append(stim_labels[stim_idx])
        col_idx += 1

    # Create metadata
    metadata = {
        'stim_indices': stim_indices,
        'nuisance_indices': polort_indices,
        'polort_indices': polort_indices,
        'n_runs': n_runs,
        'n_timepoints_per_run': n_timepoints_per_run,
        'tr': tr,
        'hrf_model': hrf_model,
        'polort': polort,
    }

    return design_matrix, regressor_labels, run_starts, metadata
