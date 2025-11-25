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
import re


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
    separated by spaces. Special cases:
    - Empty row: no events in that run
    - '* *' (two asterisks): condition not present in that run (AFNI convention)

    Parameters
    ----------
    filepath : str or Path
        Path to timing file

    Returns
    -------
    onsets_by_run : list of np.ndarray
        List with one array per run, containing onset times in seconds
        Empty runs (no events or '* *') return empty array

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

    File with missing condition marker (* *):
        12 30 46
        * *
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
                # Check for '* *' marker (condition not present)
                tokens = line.split()
                if len(tokens) == 2 and tokens[0] == '*' and tokens[1] == '*':
                    # Condition not present in this run
                    onsets_by_run.append(np.array([]))
                else:
                    # Parse onset times
                    try:
                        onsets = np.array([float(x) for x in tokens])
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

        # Normalize by max to maintain percent signal change scaling
        # This ensures that a single boxcar+HRF response has amplitude 1.0
        # (matching AFNI's behavior for proper PSC interpretation)
        if duration > 0:
            # For boxcar stimuli, need to normalize by the response to a single boxcar
            # Create reference boxcar and convolve
            ref_boxcar = np.zeros(len(hrf) * 2)
            duration_tr = int(np.round(duration / tr))
            ref_boxcar[:duration_tr] = 1.0
            ref_response = np.convolve(ref_boxcar, hrf, mode='full')
            max_response = ref_response.max()

            if max_response > 0:
                regressor = regressor / max_response

    return regressor


def parse_glt_string(glt_string: str) -> Tuple[Dict[str, float], bool]:
    """
    Parse AFNI GLT (General Linear Test) contrast string

    Extracts weights for each regressor label from symbolic contrast strings.
    Also validates that weights sum to 0 (difference) or 1 (average).

    Parameters
    ----------
    glt_string : str
        GLT contrast in AFNI SYM format, e.g.:
        - 'SYM: +1*labelA -1*labelB' (difference)
        - 'SYM: +0.5*labelA +0.5*labelB' (average)
        - 'SYM: +1*cond1 +2*cond2 -3*cond3'

    Returns
    -------
    weights : dict
        Dictionary mapping label to weight, e.g., {'labelA': 1.0, 'labelB': -1.0}
    is_valid : bool
        True if weights sum to 0 or 1, False otherwise (with warning)

    Examples
    --------
    >>> parse_glt_string('SYM: +1*movie -1*prompt')
    ({'movie': 1.0, 'prompt': -1.0}, True)

    >>> parse_glt_string('SYM: +0.5*cond1 +0.5*cond2')
    ({'cond1': 0.5, 'cond2': 0.5}, True)
    """
    # Remove 'SYM:' prefix if present
    glt_string = glt_string.strip()
    if glt_string.upper().startswith('SYM:'):
        glt_string = glt_string[4:].strip()

    # Parse weights and labels: pattern is [+/-]weight*label
    # Match: optional sign, number (int or float), *, label
    pattern = r'([+-]?\s*\d+\.?\d*)\s*\*\s*([A-Za-z_][\w\-]*)'
    matches = re.findall(pattern, glt_string)

    if not matches:
        raise ValueError(f"Could not parse GLT string: '{glt_string}'. Expected format: 'SYM: +1*label1 -1*label2'")

    weights = {}
    for weight_str, label in matches:
        # Remove spaces and convert to float
        weight_str = weight_str.replace(' ', '')
        weight = float(weight_str)
        weights[label] = weight

    # Validate sum
    weight_sum = sum(weights.values())
    is_valid = abs(weight_sum) < 1e-6 or abs(weight_sum - 1.0) < 1e-6

    if not is_valid:
        import warnings
        warnings.warn(
            f"GLT weights sum to {weight_sum:.6f}, expected 0 (difference) or 1 (average). "
            f"GLT: '{glt_string}'"
        )

    return weights, is_valid


def glt_weights_to_vector(
    weights: Dict[str, float],
    regressor_labels: List[str],
) -> np.ndarray:
    """
    Convert GLT weights dict to contrast vector

    Maps label weights to column indices in design matrix.

    Parameters
    ----------
    weights : dict
        Dictionary mapping label to weight, e.g., {'movie': 1.0, 'prompt': -1.0}
    regressor_labels : list of str
        Labels for all regressors in design matrix (in column order)

    Returns
    -------
    contrast_vector : np.ndarray
        Vector of weights, shape (n_regressors,)
        Zero for regressors not in weights dict

    Raises
    ------
    ValueError
        If a label in weights is not found in regressor_labels

    Examples
    --------
    >>> weights = {'movie': 1.0, 'prompt': -1.0}
    >>> labels = ['Run1_Poly0', 'Run1_Poly1', 'movie', 'prompt']
    >>> glt_weights_to_vector(weights, labels)
    array([0., 0., 1., -1.])
    """
    n_regressors = len(regressor_labels)
    contrast_vector = np.zeros(n_regressors)

    for label, weight in weights.items():
        # Try exact match first
        if label in regressor_labels:
            idx = regressor_labels.index(label)
            contrast_vector[idx] = weight
        else:
            # Try matching base label (e.g., 'movie' matches 'movie#0')
            # For standard mode, this finds the single column
            # For IM mode, this would sum across all events (movie#0, movie#1, ...)
            matches = [i for i, l in enumerate(regressor_labels)
                      if l.split('#')[0] == label and '#' in l]

            if matches:
                # Standard mode: single match (movie#0)
                # IM mode: multiple matches (movie#0, movie#1, ...) - weight each equally
                for idx in matches:
                    contrast_vector[idx] = weight
            else:
                raise ValueError(
                    f"GLT label '{label}' not found in regressor labels. "
                    f"Available labels: {regressor_labels}"
                )

    return contrast_vector


def parse_hrf_model(hrf_string: str) -> Tuple[str, float]:
    """
    Parse AFNI HRF model string

    Extracts model name and duration from strings like 'SPMG1(5)', 'BLOCK(10)', etc.

    Parameters
    ----------
    hrf_string : str
        AFNI HRF model specification, e.g.:
        - 'SPMG1(5)': SPM canonical with 5s stimulus duration
        - 'SPMG1(30)': SPM canonical with 30s stimulus duration
        - 'BLOCK(10)': Boxcar with 10s duration
        - 'TENT(0,15,6)': Tent function (not yet implemented)

    Returns
    -------
    model_name : str
        HRF model name (e.g., 'SPMG1', 'BLOCK')
    duration : float
        Stimulus duration in seconds

    Raises
    ------
    ValueError
        If HRF string format is invalid

    Examples
    --------
    >>> parse_hrf_model('SPMG1(5)')
    ('SPMG1', 5.0)
    >>> parse_hrf_model('BLOCK(30)')
    ('BLOCK', 30.0)
    """
    # Match pattern: MODEL(duration) or MODEL(p1,p2,p3)
    # Model name can contain letters and digits (e.g., SPMG1, BLOCK, TENT)
    match = re.match(r'^([A-Z][A-Z0-9]*)\(([^)]+)\)$', hrf_string)

    if not match:
        raise ValueError(f"Invalid HRF model string: '{hrf_string}'. Expected format like 'SPMG1(5)'")

    model_name = match.group(1)
    params_str = match.group(2)

    # For now, we only support single-parameter models (SPMG1, BLOCK)
    # TENT and others with multiple parameters will come later
    try:
        duration = float(params_str)
    except ValueError:
        raise ValueError(
            f"Invalid duration in HRF model '{hrf_string}'. "
            f"For now, only single-parameter models are supported (e.g., 'SPMG1(5)')"
        )

    return model_name, duration


def load_and_pad_ortvec(
    filepath: Union[str, Path],
    run_number: int,
    n_timepoints_per_run: List[int],
) -> np.ndarray:
    """
    Load nuisance regressor file and zero-pad for specific run

    This implements -padortvec functionality: loads a file containing
    nuisance regressors for one run and pads with zeros for other runs.

    Parameters
    ----------
    filepath : str or Path
        Path to nuisance regressor file (e.g., motion parameters)
        File should have n_timepoints rows and n_regressors columns
    run_number : int
        Which run this file belongs to (1-indexed, like AFNI)
    n_timepoints_per_run : list of int
        Number of timepoints in each run

    Returns
    -------
    padded_regressors : np.ndarray
        Zero-padded regressors, shape (total_timepoints, n_regressors)
        Non-zero only for the specified run

    Examples
    --------
    >>> # 3 runs: 100, 100, 100 TRs
    >>> # Motion file has 100 rows (for run 2)
    >>> padded = load_and_pad_ortvec('motion_r02.1D', run_number=2,
    ...                                n_timepoints_per_run=[100, 100, 100])
    >>> padded.shape
    (300, 6)  # 6 motion parameters
    >>> # Rows 0-99 are zero, rows 100-199 are from file, rows 200-299 are zero
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Ortvec file not found: {filepath}")

    # Load regressor file
    data = np.loadtxt(filepath)

    # Ensure 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_rows, n_cols = data.shape

    # Validate run number
    if run_number < 1 or run_number > len(n_timepoints_per_run):
        raise ValueError(
            f"Invalid run_number={run_number}. "
            f"Must be between 1 and {len(n_timepoints_per_run)}"
        )

    # Validate file length matches run length
    expected_rows = n_timepoints_per_run[run_number - 1]  # Convert to 0-indexed
    if n_rows != expected_rows:
        raise ValueError(
            f"Ortvec file {filepath} has {n_rows} rows, "
            f"but run {run_number} has {expected_rows} timepoints"
        )

    # Create zero-padded array
    total_timepoints = sum(n_timepoints_per_run)
    padded = np.zeros((total_timepoints, n_cols))

    # Insert data at correct position
    run_start = sum(n_timepoints_per_run[:run_number - 1])
    run_end = run_start + expected_rows
    padded[run_start:run_end, :] = data

    return padded


def build_design_matrix(
    timing_files: List[Union[str, Path]],
    stim_labels: List[str],
    n_timepoints_per_run: List[int],
    tr: float,
    polort: int = 3,
    hrf_models: Optional[Union[str, List[str]]] = None,
    im_mode: Optional[Union[bool, List[bool]]] = None,
    padortvec_files: Optional[List[Tuple[Union[str, Path], str, int]]] = None,
    ortvec_files: Optional[List[Tuple[Union[str, Path], str]]] = None,
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
        Each file has one row per run with onset times in seconds
        Use '* *' to indicate condition not present in a run
    stim_labels : list of str
        Labels for each stimulus condition (must match timing_files length)
    n_timepoints_per_run : list of int
        Number of timepoints in each run
    tr : float
        Repetition time in seconds
    polort : int, default=3
        Polynomial order for drift modeling (AFNI -polort)
        Set to -1 for no polynomials
    hrf_models : str or list of str, optional
        HRF model(s) with duration, e.g., 'SPMG1(5)', 'BLOCK(30)'
        Can be single string (applied to all) or list (one per stimulus)
        Supported models: SPMG1 (SPM canonical), BLOCK (boxcar)
        Default: 'SPMG1(0)' (impulse with HRF)
    im_mode : bool or list of bool, optional
        Individual modulation mode (AFNI's -stim_times_IM)
        If True: each event gets its own column (for amplitude modulation)
        If False: all events for condition share one column (default)
        Can be single bool (all stimuli) or list (per stimulus)
        Example: im_mode=[False, False, True] - only 3rd stimulus uses IM
    padortvec_files : list of tuple, optional
        List of (filepath, label, run_number) for per-run nuisance regressors
        Files are zero-padded to span all runs
        Example: [('motion_r01.1D', 'motion_r01', 1), ...]
    ortvec_files : list of tuple, optional
        List of (filepath, label) for full-length nuisance regressors
        Files must have total_timepoints rows
        Example: [('physio.1D', 'physio')]
    extra_regressors : list of np.ndarray, optional
        Additional regressors to include
        Each array: (total_timepoints,) or (total_timepoints, n_cols)
    extra_regressor_labels : list of str, optional
        Labels for extra regressors

    Returns
    -------
    design_matrix : np.ndarray
        Complete design matrix, shape (total_timepoints, n_regressors)
        Column order: polynomials, padortvec, ortvec, stimuli, extra
    regressor_labels : list of str
        Labels for each column in design matrix
    run_starts : list of int
        Starting indices for each run (for run-based CV)
    metadata : dict
        Metadata about the design:
        - 'stim_indices': Stimulus column indices
        - 'nuisance_indices': All nuisance column indices
        - 'polort_indices': Polynomial column indices
        - 'padortvec_indices': Padded ortvec column indices
        - 'ortvec_indices': Standard ortvec column indices
        - 'extra_indices': Extra regressor column indices
        - 'n_runs': Number of runs
        - 'n_timepoints_per_run': Timepoints per run
        - 'tr': TR in seconds
        - 'hrf_models': HRF model strings used
        - 'hrf_types': HRF types extracted
        - 'stim_durations': Stimulus durations in seconds

    Examples
    --------
    >>> # Simple design with per-stimulus HRF models
    >>> design, labels, run_starts, meta = build_design_matrix(
    ...     timing_files=['instruct.txt', 'task.txt'],
    ...     stim_labels=['instruct', 'task'],
    ...     n_timepoints_per_run=[360, 360],
    ...     tr=1.0,
    ...     polort=3,
    ...     hrf_models=['SPMG1(5)', 'SPMG1(30)']
    ... )

    >>> # Complex design with motion parameters
    >>> design, labels, run_starts, meta = build_design_matrix(
    ...     timing_files=['cond1.txt', 'cond2.txt'],
    ...     stim_labels=['cond1', 'cond2'],
    ...     n_timepoints_per_run=[200, 200, 200],
    ...     tr=2.0,
    ...     polort=3,
    ...     hrf_models='SPMG1(10)',
    ...     padortvec_files=[
    ...         ('motion_r01.1D', 'motion_r01', 1),
    ...         ('motion_r02.1D', 'motion_r02', 2),
    ...         ('motion_r03.1D', 'motion_r03', 3),
    ...     ]
    ... )
    """
    n_runs = len(n_timepoints_per_run)
    n_stim = len(timing_files)

    # Validate inputs
    if len(stim_labels) != n_stim:
        raise ValueError(f"stim_labels length ({len(stim_labels)}) must match timing_files ({n_stim})")

    # Handle HRF models - can be single string or list
    if hrf_models is None:
        hrf_models = ['SPMG1(0)'] * n_stim  # Default: impulse with SPM HRF
    elif isinstance(hrf_models, str):
        hrf_models = [hrf_models] * n_stim  # Broadcast to all stimuli
    elif len(hrf_models) != n_stim:
        raise ValueError(f"hrf_models length ({len(hrf_models)}) must match n_stim ({n_stim})")

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

    # Parse HRF models and create stimulus durations
    stim_durations = []
    hrf_types = []
    for hrf_spec in hrf_models:
        model_name, duration = parse_hrf_model(hrf_spec)
        stim_durations.append(duration)
        hrf_types.append(model_name)

    # Handle IM mode - can be single bool or list
    if im_mode is None:
        im_mode = [False] * n_stim  # Default: no IM
    elif isinstance(im_mode, bool):
        im_mode = [im_mode] * n_stim  # Broadcast to all stimuli
    elif len(im_mode) != n_stim:
        raise ValueError(f"im_mode length ({len(im_mode)}) must match n_stim ({n_stim})")

    # Count total stimulus columns (depends on IM mode)
    # Non-IM: 1 column per stimulus
    # IM: 1 column per event (summed across runs)
    n_stim_cols = 0
    stim_n_events = []  # Number of events for each stimulus (for IM mode)
    for stim_idx in range(n_stim):
        if im_mode[stim_idx]:
            # Count total events across all runs
            n_events = sum(len(all_onsets[stim_idx][run_idx]) for run_idx in range(n_runs))
            stim_n_events.append(n_events)
            n_stim_cols += n_events
        else:
            stim_n_events.append(1)  # One column for all events
            n_stim_cols += 1

    # Build design matrix in AFNI column order:
    # 1. Polynomial regressors (per-run)
    # 2. Ortvec regressors (padortvec, then ortvec)
    # 3. Stimulus regressors (spanning all runs)
    #    - Non-IM: one column per condition
    #    - IM: one column per event
    # 4. Extra regressors (if any)

    total_timepoints = sum(n_timepoints_per_run)
    run_starts = []

    # Count columns
    n_polort_cols = (polort + 1) * n_runs if polort >= 0 else 0
    n_padortvec_cols = 0
    n_ortvec_cols = 0
    n_extra_cols = 0

    # Load and count padortvec files
    padortvec_data = []
    padortvec_labels_list = []
    if padortvec_files:
        for filepath, label, run_num in padortvec_files:
            padded = load_and_pad_ortvec(filepath, run_num, n_timepoints_per_run)
            padortvec_data.append(padded)
            # Create labels for each column
            n_cols = padded.shape[1]
            n_padortvec_cols += n_cols
            for col_idx_local in range(n_cols):
                if n_cols == 1:
                    padortvec_labels_list.append(label)
                else:
                    padortvec_labels_list.append(f'{label}[{col_idx_local}]')

    # Load and count ortvec files
    ortvec_data = []
    ortvec_labels_list = []
    if ortvec_files:
        for filepath, label in ortvec_files:
            data = np.loadtxt(filepath)
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            # Validate length
            if data.shape[0] != total_timepoints:
                raise ValueError(
                    f"Ortvec file {filepath} has {data.shape[0]} rows, "
                    f"but total timepoints is {total_timepoints}"
                )

            ortvec_data.append(data)
            n_cols = data.shape[1]
            n_ortvec_cols += n_cols
            for col_idx_local in range(n_cols):
                if n_cols == 1:
                    ortvec_labels_list.append(label)
                else:
                    ortvec_labels_list.append(f'{label}[{col_idx_local}]')

    # Count extra regressors
    if extra_regressors:
        for reg in extra_regressors:
            if reg.ndim == 1:
                n_extra_cols += 1
            else:
                n_extra_cols += reg.shape[1]

    # Total columns (use n_stim_cols which accounts for IM mode)
    n_total_cols = n_polort_cols + n_padortvec_cols + n_ortvec_cols + n_stim_cols + n_extra_cols
    design_matrix = np.zeros((total_timepoints, n_total_cols))

    # Track column indices
    col_idx = 0
    polort_indices = []
    padortvec_indices = []
    ortvec_indices = []
    stim_indices = []
    extra_indices = []
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
                regressor_labels.append(f'Run#{run_idx+1}Pol#{p}')
                col_idx += 1
    else:
        # No polynomials, still need run_starts
        for run_idx in range(n_runs):
            run_starts.append(sum(n_timepoints_per_run[:run_idx]))

    # 2a. Add padortvec regressors
    for padded in padortvec_data:
        for c in range(padded.shape[1]):
            design_matrix[:, col_idx] = padded[:, c]
            padortvec_indices.append(col_idx)
            col_idx += 1

    # Add padortvec labels
    regressor_labels.extend(padortvec_labels_list)

    # 2b. Add ortvec regressors
    for ort_data in ortvec_data:
        for c in range(ort_data.shape[1]):
            design_matrix[:, col_idx] = ort_data[:, c]
            ortvec_indices.append(col_idx)
            col_idx += 1

    # Add ortvec labels
    regressor_labels.extend(ortvec_labels_list)

    # 3. Add stimulus regressors (spanning all runs)
    for stim_idx in range(n_stim):
        # Get HRF for this stimulus
        hrf_type = hrf_types[stim_idx]

        if hrf_type == 'SPMG1':
            hrf = spm_canonical_hrf(tr=tr, duration=32.0)
        elif hrf_type == 'BLOCK':
            hrf = None  # No HRF convolution
        else:
            raise ValueError(f"Unknown HRF type: {hrf_type}. Supported: SPMG1, BLOCK")

        if im_mode[stim_idx]:
            # Individual modulation mode: one column per event
            # Collect all onsets across all runs and create one regressor per event
            event_idx = 0
            for run_idx in range(n_runs):
                n_tp = n_timepoints_per_run[run_idx]
                run_start = sum(n_timepoints_per_run[:run_idx])
                run_end = run_start + n_tp

                onsets = all_onsets[stim_idx][run_idx]
                duration = stim_durations[stim_idx]

                # Create one column per event in this run
                for onset in onsets:
                    event_regressor = create_onset_regressors(
                        onset_times=np.array([onset]),  # Single onset
                        n_timepoints=n_tp,
                        tr=tr,
                        duration=duration,
                        hrf=hrf,
                    )

                    # Place in full time series
                    full_regressor = np.zeros(total_timepoints)
                    full_regressor[run_start:run_end] = event_regressor

                    # Add to design matrix
                    design_matrix[:, col_idx] = full_regressor
                    stim_indices.append(col_idx)
                    # IM mode: label#0, label#1, label#2, etc.
                    regressor_labels.append(f'{stim_labels[stim_idx]}#{event_idx}')
                    col_idx += 1
                    event_idx += 1

        else:
            # Standard mode: one column for all events
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
            # Standard mode: label#0 (only one column per stimulus)
            regressor_labels.append(f'{stim_labels[stim_idx]}#0')
            col_idx += 1

    # 4. Add extra regressors (if any)
    if extra_regressors:
        for idx, reg in enumerate(extra_regressors):
            if reg.ndim == 1:
                design_matrix[:, col_idx] = reg
                extra_indices.append(col_idx)
                col_idx += 1
            else:
                for c in range(reg.shape[1]):
                    design_matrix[:, col_idx] = reg[:, c]
                    extra_indices.append(col_idx)
                    col_idx += 1

        # Add extra labels
        if extra_regressor_labels:
            if len(extra_regressor_labels) != len(extra_regressors):
                raise ValueError(
                    f"extra_regressor_labels length ({len(extra_regressor_labels)}) "
                    f"must match extra_regressors ({len(extra_regressors)})"
                )
            regressor_labels.extend(extra_regressor_labels)
        else:
            # Generate default labels
            for idx in range(len(extra_indices)):
                regressor_labels.append(f'extra_{idx}')

    # Create metadata
    nuisance_indices = polort_indices + padortvec_indices + ortvec_indices + extra_indices

    metadata = {
        'stim_indices': stim_indices,
        'nuisance_indices': nuisance_indices,
        'polort_indices': polort_indices,
        'padortvec_indices': padortvec_indices,
        'ortvec_indices': ortvec_indices,
        'extra_indices': extra_indices,
        'n_runs': n_runs,
        'n_timepoints_per_run': n_timepoints_per_run,
        'tr': tr,
        'hrf_models': hrf_models,
        'hrf_types': hrf_types,
        'stim_durations': stim_durations,
        'polort': polort,
    }

    return design_matrix, regressor_labels, run_starts, metadata


def write_afni_xmat(
    filepath: Union[str, Path],
    design_matrix: np.ndarray,
    regressor_labels: List[str],
    run_starts: List[int],
    metadata: Dict,
    glt_contrasts: Optional[List[Tuple[str, str]]] = None,
    command_line: Optional[str] = None,
) -> None:
    """
    Write design matrix in AFNI .xmat.1D format

    Parameters
    ----------
    filepath : str or Path
        Output file path
    design_matrix : np.ndarray
        Design matrix (n_timepoints, n_regressors)
    regressor_labels : list of str
        Column labels
    run_starts : list of int
        Starting timepoint index for each run
    metadata : dict
        Metadata dictionary from build_design_matrix()
        Must contain: 'n_runs', 'tr', 'stim_indices', 'hrf_models',
                      'hrf_types', 'stim_durations'
    glt_contrasts : list of (contrast_string, label) tuples, optional
        GLT contrasts in format [('SYM: +1*A -1*B', 'AvsB'), ...]
    command_line : str, optional
        Command line string to include in header

    Notes
    -----
    AFNI .xmat.1D format includes:
    - Header with matrix metadata (ni_type, ni_dimen, etc.)
    - Column labels, groups, and stimulus info
    - Run starts and TR information
    - GLT contrast matrices
    - Basis function formulas
    - Data matrix (space-delimited, one row per timepoint)

    Examples
    --------
    >>> design, labels, runs, meta = build_design_matrix(...)
    >>> glt = [('SYM: +1*movie -1*prompt', 'movieVprompt')]
    >>> write_afni_xmat('X.xmat.1D', design, labels, runs, meta, glt)
    """
    n_timepoints, n_regressors = design_matrix.shape
    n_runs = metadata['n_runs']
    tr = metadata['tr']
    stim_indices = metadata['stim_indices']

    # Extract unique stimulus labels and their column ranges
    stim_labels_list = []
    stim_bots = []
    stim_tops = []
    seen_stim = set()

    if len(stim_indices) > 0:
        for idx in stim_indices:
            label = regressor_labels[idx]
            # Remove #N suffix to get base label (e.g., movie#0 -> movie)
            base_label = label.split('#')[0] if '#' in label else label
            if base_label not in seen_stim:
                stim_labels_list.append(base_label)
                seen_stim.add(base_label)

        # For each unique stimulus, find its bottom and top column indices
        for base_label in stim_labels_list:
            cols = [i for i in stim_indices
                   if regressor_labels[i].split('#')[0] == base_label]
            stim_bots.append(min(cols))
            stim_tops.append(max(cols))

        n_stim = len(stim_labels_list)
    else:
        n_stim = 0

    # Build column groups
    # Format: "N@-1,M,K" means N columns in group -1 (nuisance), then groups M, K for stim
    # Group numbering: -1=polort, 0=motion/baseline, 1,2,3,...=stimuli of interest
    n_nuisance = len(metadata.get('nuisance_indices', []))
    if n_stim > 0:
        col_groups = f"{n_nuisance}@-1"
        for stim_idx in range(n_stim):
            col_groups += f",{stim_idx + 1}"  # Stimuli start at 1, not 0
    else:
        col_groups = f"{n_nuisance}@-1"

    # Create header
    with open(filepath, 'w') as f:
        # Matrix metadata
        f.write("# <matrix\n")
        f.write(f'#  ni_type = "{n_regressors}*double"\n')
        f.write(f'#  ni_dimen = "{n_timepoints}"\n')

        # Column labels
        labels_str = " ; ".join(regressor_labels)
        f.write(f'#  ColumnLabels = "{labels_str}"\n')

        # Column groups
        f.write(f'#  ColumnGroups = "{col_groups}"\n')

        # TR and timepoint info
        f.write(f'#  RowTR = "{tr}"\n')
        f.write(f'#  GoodList = "0..{n_timepoints-1}"\n')
        f.write(f'#  NRowFull = "{n_timepoints}"\n')

        # Run starts
        run_starts_str = ",".join(map(str, run_starts))
        f.write(f'#  RunStart = "{run_starts_str}"\n')

        # Stimulus info
        f.write(f'#  Nstim = "{n_stim}"\n')
        if n_stim > 0:
            # StimBots and StimTops are comma-separated lists of bottom/top columns for each stimulus
            stim_bots_str = ",".join(map(str, stim_bots))
            stim_tops_str = ",".join(map(str, stim_tops))
            f.write(f'#  StimBots = "{stim_bots_str}"\n')
            f.write(f'#  StimTops = "{stim_tops_str}"\n')

            stim_labels_str = " ; ".join(stim_labels_list)
            f.write(f'#  StimLabels = "{stim_labels_str}"\n')

        # GLT contrasts
        if glt_contrasts:
            f.write(f'#  Nglt = "{len(glt_contrasts)}"\n')
            glt_labels = [label for _, label in glt_contrasts]
            f.write(f'#  GltLabels = "{" ; ".join(glt_labels)}"\n')

            for glt_idx, (contrast_str, label) in enumerate(glt_contrasts):
                # Parse contrast and create matrix representation
                weights, _ = parse_glt_string(contrast_str)
                contrast_vec = glt_weights_to_vector(weights, regressor_labels)

                # Format: "1,n_regressors,values"
                # Only include non-zero weights
                nonzero_indices = np.where(contrast_vec != 0)[0]
                matrix_str = f"1,{n_regressors},"

                # Build compact representation
                parts = []
                for i, val in enumerate(contrast_vec):
                    if i == 0:
                        if val == 0:
                            # Count leading zeros
                            next_nonzero = nonzero_indices[0] if len(nonzero_indices) > 0 else n_regressors
                            if next_nonzero > 0:
                                parts.append(f"{next_nonzero}@0")
                    else:
                        if val != 0:
                            parts.append(str(val))
                        else:
                            # Check if we have contiguous zeros
                            pass  # Handle in the simplest way

                # Simple format: just list all values with compression for leading zeros
                if len(nonzero_indices) > 0 and nonzero_indices[0] > 0:
                    matrix_str += f"{nonzero_indices[0]}@0"
                    for i in range(nonzero_indices[0], n_regressors):
                        if contrast_vec[i] != 0:
                            matrix_str += f",{contrast_vec[i]:.0f}"
                else:
                    # No leading zeros, just list values
                    matrix_str += ",".join([str(int(v)) if v != 0 else "0" for v in contrast_vec])

                f.write(f'#  GltMatrix_{glt_idx:06d} = "{matrix_str}"\n')

        # Basis function info
        if n_stim > 0:
            f.write(f'#  BasisNstim = "{n_stim}"\n')

            # Get unique stimuli
            stim_info = {}
            for idx in stim_indices:
                label = regressor_labels[idx]
                # Remove #N suffix to get base label (e.g., movie#0 -> movie)
                base_label = label.split('#')[0] if '#' in label else label

                if base_label not in stim_info:
                    # Find which stimulus this is
                    stim_idx_in_list = None
                    for i, sl in enumerate(stim_labels_list):
                        if sl == base_label:
                            stim_idx_in_list = i
                            break

                    if stim_idx_in_list is not None:
                        # Get HRF model and duration
                        hrf_type = metadata.get('hrf_types', [])[stim_idx_in_list] if stim_idx_in_list < len(metadata.get('hrf_types', [])) else 'SPMG1'
                        duration = metadata.get('stim_durations', [])[stim_idx_in_list] if stim_idx_in_list < len(metadata.get('stim_durations', [])) else 0.0

                        # Find column range for this stimulus
                        # Match labels that are exactly base_label#N
                        cols_for_stim = [i for i, l in enumerate(regressor_labels)
                                        if l.split('#')[0] == base_label and '#' in l]

                        stim_info[base_label] = {
                            'idx': stim_idx_in_list + 1,
                            'hrf': hrf_type,
                            'duration': duration,
                            'cols': cols_for_stim,
                        }

            # Write basis info for each stimulus
            for stim_label in stim_labels_list:
                if stim_label in stim_info:
                    info = stim_info[stim_label]
                    idx = info['idx']

                    # Determine stim option (IM vs regular)
                    # IM mode: multiple columns (label#0, label#1, ...)
                    # Standard mode: single column (label#0)
                    is_im = len(info['cols']) > 1
                    option = "-stim_times_IM" if is_im else "-stim_times"

                    f.write(f'#  BasisOption_{idx:06d} = "{option}"\n')
                    f.write(f'#  BasisName_{idx:06d} = "{stim_label}"\n')
                    f.write(f'#  BasisFormula_{idx:06d} = "{info["hrf"]}({info["duration"]:.0f})"\n')

                    # Column range
                    col_start = min(info['cols'])
                    col_end = max(info['cols'])
                    f.write(f'#  BasisColumns_{idx:06d} = "{col_start}:{col_end}"\n')

        # Command line (optional)
        if command_line:
            f.write(f'#  CommandLine = "{command_line}"\n')

        # End header
        f.write("# >\n")

        # Write data matrix
        for row_idx in range(n_timepoints):
            row_str = " ".join([f"{val:.17g}" for val in design_matrix[row_idx, :]])
            f.write(f" {row_str}\n")
