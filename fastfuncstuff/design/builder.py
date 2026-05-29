"""
Design matrix construction from onset files

This module provides functions to build GLM design matrices from:
- Onset timing files (AFNI format)
- HRF models (SPM canonical, AFNI SPMG1, custom)
- Nuisance regressors (Legendre polynomials, motion parameters)

The goal is to replicate AFNI's 3dDeconvolve design matrix construction
while being easier to use and integrate with our fast GLM fitting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from scipy import special
from scipy.stats import gamma as scipy_gamma

from fastfuncstuff.design.matrices import (
    convolve_hrf,
    is_tr_locked,
    make_csplin_design,
    make_fir_design,
    make_tent_design,
)


def spm_canonical_hrf(tr: float = 1.0, duration: float = 32.0) -> np.ndarray:
    """
    Create SPM canonical HRF (double gamma function)

    This is the standard HRF used in SPM and AFNI's SPMG1 basis function.

    .. deprecated::
        For AFNI-exact HRF generation, use `hrf.get_spmg1_hrf()` instead,
        which implements the exact AFNI SPMG1 formula.

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

    For exact AFNI compatibility, prefer `hrf.get_spmg1_hrf()` which uses
    the precise AFNI formula: h(t) = exp(-t) * (A1*t^P1 - A2*t^P2)

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


def parse_afni_timing_file(filepath: str | Path) -> list[np.ndarray]:
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

    with open(filepath) as f:
        for line in f:
            line = line.strip()

            if not line:
                # Empty line = no events in this run
                onsets_by_run.append(np.array([]))
            else:
                # Check for '* *' marker (condition not present)
                tokens = line.split()
                if len(tokens) == 2 and tokens[0] == "*" and tokens[1] == "*":
                    # Condition not present in this run
                    onsets_by_run.append(np.array([]))
                else:
                    # Parse onset times
                    try:
                        onsets = np.array([float(x) for x in tokens])
                        onsets_by_run.append(onsets)
                    except ValueError as e:
                        raise ValueError(f"Could not parse line '{line}' in {filepath}: {e}") from e

    return onsets_by_run


def create_onset_regressors(
    onset_times: np.ndarray,
    n_timepoints: int,
    tr: float,
    duration: float = 0.0,
    hrf: np.ndarray | None = None,
) -> np.ndarray:
    """
    Legacy TR-resolution onset regressor builder. New code should drive
    ``build_design_matrix`` (which routes through the proper microtime
    pipeline: :func:`create_onset_matrix_microtime` →
    :func:`fastfuncstuff.design.hrf.get_spmg1_hrf` →
    :func:`fastfuncstuff.design.matrices.convolve_hrf_microtime`). This
    function is preserved for tests that exercise the TR-only path.
    """
    regressor = np.zeros(n_timepoints)
    for onset in onset_times:
        onset_tr = int(np.round(onset / tr))
        if onset_tr < 0 or onset_tr >= n_timepoints:
            continue
        if duration <= 0:
            regressor[onset_tr] = 1.0
        else:
            duration_tr = max(1, int(np.round(duration / tr)))
            end_tr = min(onset_tr + duration_tr, n_timepoints)
            regressor[onset_tr:end_tr] = 1.0

    if hrf is not None:
        regressor = np.convolve(regressor, hrf, mode="full")[:n_timepoints]
        if duration > 0:
            duration_tr = max(1, int(np.round(duration / tr)))
            ref = np.zeros(len(hrf) + duration_tr)
            ref[:duration_tr] = 1.0
            ref_response = np.convolve(ref, hrf, mode="full")
            max_response = ref_response.max()
            if max_response > 0:
                regressor = regressor / max_response
    return regressor


def parse_glt_string(
    glt_string: str | list[str],
) -> tuple[list[dict[str, _LabelWeight]], bool]:
    """
    Parse AFNI GLT (General Linear Test) contrast string(s).

    Single-row strings produce a t-test (one weight dict). Multi-row strings —
    separated by ``\\``, ``|``, or newline inside the SYM: payload — or a list
    of SYM: strings produce an F-test (one weight dict per row).

    Each label may optionally carry a sub-range ``label[a..b]`` selecting basis
    columns within a multi-column stim (AFNI ``-gltsym`` syntax). The range is
    returned as part of the value, not by mangling the label key.

    Parameters
    ----------
    glt_string : str or list of str
        GLT contrast in AFNI SYM format, e.g.:
        - 'SYM: +1*labelA -1*labelB'                       (t-test, difference)
        - 'SYM: +0.5*labelA +0.5*labelB'                   (t-test, average)
        - 'SYM: +1*A | +1*B | +1*C'                        (F-test, 3 rows)
        - 'SYM: +1*TaskA[2..4]'                            (sub-range)
        - ['SYM: +1*A', 'SYM: +1*B']                       (F-test, list form)

    Returns
    -------
    rows : list of dict
        One dict per contrast row mapping label to a (weight, range_or_None)
        tuple. ``range_or_None`` is ``(a, b)`` inclusive when ``[a..b]`` is
        present, else ``None``.
    is_valid : bool
        True if every row's weights sum to 0 or 1 (difference / average).

    Examples
    --------
    >>> rows, valid = parse_glt_string('SYM: +1*movie -1*prompt')
    >>> rows
    [{'movie': (1.0, None), 'prompt': (-1.0, None)}]
    >>> valid
    True

    >>> rows, _ = parse_glt_string('SYM: +1*A | +1*B')
    >>> len(rows)
    2
    """
    # Normalize input into a list of row strings.
    if isinstance(glt_string, list):
        row_strings = list(glt_string)
    else:
        s = glt_string.strip()
        if s.upper().startswith("SYM:"):
            s = s[4:].strip()
        # Split on backslash, pipe, or newline. Empty fragments dropped.
        row_strings = [r.strip() for r in re.split(r"[\\|\n]", s) if r.strip()]

    if not row_strings:
        raise ValueError(
            f"Could not parse GLT string: '{glt_string}'. "
            "Expected format: 'SYM: +1*label1 -1*label2' (rows separated by \\, |, or newline)"
        )

    # Per-token pattern: optional sign, number, *, label, optional [a..b]
    token_pattern = re.compile(
        r"([+-]?\s*\d+\.?\d*)\s*\*\s*([A-Za-z_][\w\-]*)"
        r"(?:\[\s*(\d+)\s*\.\.\s*(\d+)\s*\])?"
    )

    rows: list[dict[str, _LabelWeight]] = []
    overall_valid = True

    for row_str in row_strings:
        # Strip optional 'SYM:' prefix on a per-row basis too (allows list-of-SYM-strings).
        row_clean = row_str
        if row_clean.upper().startswith("SYM:"):
            row_clean = row_clean[4:].strip()

        matches = token_pattern.findall(row_clean)
        if not matches:
            raise ValueError(
                f"Could not parse GLT row: '{row_str}'. "
                "Expected format: '+1*label1 -1*label2'"
            )

        row_weights: dict[str, _LabelWeight] = {}
        for weight_str, label, range_lo, range_hi in matches:
            weight = float(weight_str.replace(" ", ""))
            rng: tuple[int, int] | None = None
            if range_lo and range_hi:
                lo, hi = int(range_lo), int(range_hi)
                if hi < lo:
                    raise ValueError(
                        f"GLT sub-range for '{label}' has hi < lo: [{lo}..{hi}]"
                    )
                rng = (lo, hi)
            row_weights[label] = (weight, rng)

        # Validate sum per row (sub-range broadcasting doesn't change the count
        # of "explicit" weights — we check the user-written coefficients, not
        # the eventual broadcast).
        weight_sum = sum(w for w, _ in row_weights.values())
        row_valid = abs(weight_sum) < 1e-6 or abs(weight_sum - 1.0) < 1e-6
        if not row_valid:
            import warnings

            warnings.warn(
                f"GLT row weights sum to {weight_sum:.6f}, expected 0 or 1. "
                f"Row: '{row_str}'",
                stacklevel=2,
            )
            overall_valid = False

        rows.append(row_weights)

    return rows, overall_valid


# Internal alias used in annotations above. Kept here (not exposed) to avoid
# spreading a tuple type across module-level imports.
_LabelWeight = tuple[float, "tuple[int, int] | None"]


def glt_weights_to_vector(
    weights: dict[str, float] | dict[str, _LabelWeight],
    regressor_labels: list[str],
    stim_ranges: dict[str, tuple[int, int]] | None = None,
) -> np.ndarray:
    """
    Convert GLT weights dict to a single contrast row vector.

    Maps label weights to column indices in the design matrix. Three label
    resolution modes are tried in order:

    1. **Exact match** against ``regressor_labels``.
    2. **Multi-column stim by name** — if ``stim_ranges`` is provided and the
       label appears there as ``(col_start, col_end)`` inclusive, the weight is
       broadcast full-width across that range (AFNI ``-gltsym`` default). If
       the value carries a sub-range ``(a, b)``, only those *basis-relative*
       columns (``col_start + a`` … ``col_start + b``) are weighted.
    3. **Base-label match** against ``label#N`` columns (IM mode).

    Parameters
    ----------
    weights : dict
        Either ``{label: weight}`` (back-compat scalar form) or
        ``{label: (weight, range_or_None)}`` as produced by parse_glt_string.
    regressor_labels : list of str
        Labels for all regressors in design matrix (in column order).
    stim_ranges : dict, optional
        Mapping ``base_label -> (col_start, col_end)`` inclusive. Required for
        full-width broadcast across multi-column bases (TENT, FIR, SPMG2/3).
        If omitted, multi-column broadcast falls back to the IM-style ``label#N``
        scan (which only finds suffixed columns).

    Returns
    -------
    contrast_vector : np.ndarray
        Vector of weights, shape (n_regressors,).

    Raises
    ------
    ValueError
        If a label resolves to no columns or a sub-range exceeds the stim's
        column count.

    Examples
    --------
    >>> weights = {'movie': 1.0, 'prompt': -1.0}
    >>> labels = ['Run1_Poly0', 'Run1_Poly1', 'movie', 'prompt']
    >>> glt_weights_to_vector(weights, labels)
    array([ 0.,  0.,  1., -1.])

    >>> # TENT(0,20,6) on TaskA spanning cols 12..17 — sub-range [2..4]
    >>> w = {'TaskA': (1.0, (2, 4))}
    >>> labels = [f'baseline#{i}' for i in range(12)] + [f'TaskA#{i}' for i in range(6)]
    >>> v = glt_weights_to_vector(w, labels, stim_ranges={'TaskA': (12, 17)})
    >>> list(v[14:17])
    [1.0, 1.0, 1.0]
    """
    n_regressors = len(regressor_labels)
    contrast_vector = np.zeros(n_regressors)
    stim_ranges = stim_ranges or {}

    for label, value in weights.items():
        # Normalize value: scalar (back-compat) vs (weight, range) tuple.
        if isinstance(value, tuple):
            weight, sub_range = value
        else:
            weight, sub_range = float(value), None

        # 1) Exact match — single column.
        if label in regressor_labels:
            if sub_range is not None:
                raise ValueError(
                    f"GLT label '{label}' is a single column but a sub-range "
                    f"{sub_range} was given."
                )
            idx = regressor_labels.index(label)
            contrast_vector[idx] = weight
            continue

        # 2) Multi-column stim by name (from StimBots/StimTops metadata).
        if label in stim_ranges:
            col_start, col_end = stim_ranges[label]
            n_basis = col_end - col_start + 1
            if sub_range is not None:
                lo, hi = sub_range
                if hi >= n_basis:
                    raise ValueError(
                        f"GLT sub-range [{lo}..{hi}] for '{label}' exceeds "
                        f"basis count {n_basis} (cols {col_start}..{col_end})."
                    )
                contrast_vector[col_start + lo : col_start + hi + 1] = weight
            else:
                # Full-width broadcast — weight applied to every basis column.
                contrast_vector[col_start : col_end + 1] = weight
            continue

        # 3) Base-label scan for IM-style `label#N` columns.
        matches = [
            i for i, l in enumerate(regressor_labels)
            if l.split("#")[0] == label and "#" in l
        ]
        if matches:
            if sub_range is not None:
                lo, hi = sub_range
                if hi >= len(matches):
                    raise ValueError(
                        f"GLT sub-range [{lo}..{hi}] for '{label}' exceeds "
                        f"match count {len(matches)}."
                    )
                for idx in matches[lo : hi + 1]:
                    contrast_vector[idx] = weight
            else:
                for idx in matches:
                    contrast_vector[idx] = weight
            continue

        raise ValueError(
            f"GLT label '{label}' not found in regressor labels. "
            f"Available labels: {regressor_labels}"
        )

    return contrast_vector


def glt_rows_to_matrix(
    rows: list[dict[str, _LabelWeight]],
    regressor_labels: list[str],
    stim_ranges: dict[str, tuple[int, int]] | None = None,
) -> np.ndarray:
    """
    Resolve a list of contrast rows (as produced by parse_glt_string) to a
    GLT matrix of shape ``(n_rows, n_regressors)``. ``n_rows == 1`` is a
    t-test; ``n_rows > 1`` is an F-test.
    """
    n_regressors = len(regressor_labels)
    matrix = np.zeros((len(rows), n_regressors))
    for i, row in enumerate(rows):
        matrix[i] = glt_weights_to_vector(row, regressor_labels, stim_ranges)
    return matrix


def _compress_int_sequence(values: list[int]) -> str:
    """Compress a sequence of integers using AFNI's two-rule notation:

    - ``a..b`` for ascending runs of consecutive values (step 1).
    - ``N@v`` for ``N`` repetitions of the same value.
    - Otherwise the literal value, comma-separated.

    AFNI applies both rules in the same string. Run-of-same wins over
    range when both could apply (i.e. a single repeated value is never
    rendered as ``a..a``).

    Examples:
      ``[1, 2, 3]``               → ``"1..3"``
      ``[-1, -1, -1, 0, 0]``     → ``"3@-1,2@0"``
      ``[-1]*12 + [1, 2, 3] + [0]*4`` → ``"12@-1,1..3,4@0"``
    """
    if not values:
        return ""
    parts: list[str] = []
    i = 0
    n = len(values)
    while i < n:
        v = values[i]
        # 1) Run of identical values.
        j = i
        while j < n and values[j] == v:
            j += 1
        run_len = j - i
        if run_len >= 2:
            parts.append(f"{run_len}@{v}")
            i = j
            continue
        # 2) Ascending consecutive run starting at v.
        j = i
        while j + 1 < n and values[j + 1] == values[j] + 1:
            j += 1
        seq_len = j - i + 1
        if seq_len >= 3:
            parts.append(f"{values[i]}..{values[j]}")
            i = j + 1
            continue
        parts.append(f"{v}")
        i += 1
    return ",".join(parts)


def _compress_index_runs(indices: list[int]) -> str:
    """
    Render a sorted list of TR indices in AFNI's GoodList shorthand:
    contiguous runs become ``a..b``, isolated indices stay as ``n``, all
    joined by commas. Example: ``[0,1,2,4,5,7]`` → ``"0..2,4..5,7"``.
    """
    if not indices:
        return ""
    parts: list[str] = []
    start = prev = indices[0]
    for v in indices[1:]:
        if v == prev + 1:
            prev = v
            continue
        parts.append(f"{start}..{prev}" if prev > start else f"{start}")
        start = prev = v
    parts.append(f"{start}..{prev}" if prev > start else f"{start}")
    return ",".join(parts)


def good_list_from_censor(
    keep_mask: np.ndarray | list[int] | list[bool],
) -> tuple[list[int], int]:
    """
    Convert a per-TR keep mask (1=keep, 0=censor, e.g. AFNI ``outcount.1D``)
    into a (good_list, n_row_full) pair suitable for write_afni_xmat.

    Accepts either a boolean/integer array of length ``n_row_full`` or an
    already-resolved list of kept TR indices (returned unchanged).
    """
    arr = np.asarray(keep_mask)
    if arr.ndim != 1:
        raise ValueError(f"keep_mask must be 1-D, got shape {arr.shape}")
    # Heuristic: if values are only 0/1 (or bool), treat as mask. Otherwise
    # treat as an already-resolved list of indices.
    if arr.dtype == bool or set(np.unique(arr).tolist()).issubset({0, 1}):
        n_row_full = int(arr.size)
        good = [i for i, v in enumerate(arr.tolist()) if v]
        return good, n_row_full
    indices = [int(i) for i in arr.tolist()]
    return indices, max(indices) + 1


def _format_glt_value(v: float) -> str:
    """Format a GLT weight for the AFNI compact notation. Integer-valued
    floats render without a decimal; fractional values use general format."""
    if float(v).is_integer():
        return f"{int(v)}"
    return f"{v:g}"


def _afni_compact_vector(values: np.ndarray, leading_count: int = 0) -> str:
    """
    Serialize a 1-D array in AFNI's compact notation: leading and trailing
    runs of zeros are compressed to ``N@0``. Interior zeros are left verbatim
    so the total token count equals the array length once expanded.

    Example: ``[0,0,0,1,-1,0,0,0,0,0,0]`` → ``"3@0,1,-1,6@0"``.

    The ``leading_count`` parameter is unused but reserved so callers can
    tune behaviour without touching the call sites.
    """
    del leading_count  # reserved
    n = len(values)
    if n == 0:
        return ""

    nonzero = np.flatnonzero(values)
    if len(nonzero) == 0:
        return f"{n}@0"

    first_nz = int(nonzero[0])
    last_nz = int(nonzero[-1])

    parts: list[str] = []
    if first_nz > 0:
        parts.append(f"{first_nz}@0")
    parts.extend(_format_glt_value(v) for v in values[first_nz : last_nz + 1])
    trailing = n - last_nz - 1
    if trailing > 0:
        parts.append(f"{trailing}@0")
    return ",".join(parts)


def parse_hrf_model(hrf_string: str) -> tuple[str, float | dict]:
    """
    Parse AFNI HRF model string

    Extracts model name and parameters from strings like 'SPMG1(5)', 'BLOCK(10)', 'TENT(0,15,6)', etc.

    Parameters
    ----------
    hrf_string : str
        AFNI HRF model specification, e.g.:
        - 'SPMG1(5)': SPM canonical with 5s stimulus duration
        - 'SPMG1(30)': SPM canonical with 30s stimulus duration
        - 'BLOCK(10)': Boxcar with 10s duration
        - 'TENT(0,15,6)': Tent function from 0-15s with 6 basis functions
        - 'TENTzero(0,20,8)': TENTzero from 0-20s with 8 basis functions

    Returns
    -------
    model_name : str
        HRF model name (e.g., 'SPMG1', 'BLOCK', 'TENT', 'TENTzero')
    params : float or dict
        For simple models (SPMG1, BLOCK): stimulus duration in seconds
        For TENT/TENTzero: dict with keys 'bot', 'top', 'n_basis'

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
    >>> parse_hrf_model('TENT(0,15,6)')
    ('TENT', {'bot': 0.0, 'top': 15.0, 'n_basis': 6})
    >>> parse_hrf_model('TENTzero(0,20,8)')
    ('TENTzero', {'bot': 0.0, 'top': 20.0, 'n_basis': 8})
    """
    # Match pattern: MODEL(duration) or MODEL(p1,p2,p3)
    # Model name can contain letters and digits (e.g., SPMG1, BLOCK, TENT)
    match = re.match(r"^([A-Z][A-Z0-9]*)\(([^)]+)\)$", hrf_string, re.IGNORECASE)

    if not match:
        raise ValueError(
            f"Invalid HRF model string: '{hrf_string}'. Expected format like 'SPMG1(5)' or 'TENT(0,15,6)'"
        )

    model_name = match.group(1).upper()  # Normalize to uppercase
    params_str = match.group(2)

    # Handle TENT/TENTzero models with 3 parameters: TENT(bot,top,n)
    if model_name in ("TENT", "TENTZERO"):
        params_parts = params_str.split(",")
        if len(params_parts) != 3:
            raise ValueError(
                f"Invalid {model_name} model '{hrf_string}'. "
                f"Expected format: '{model_name}(bot,top,n)' with 3 parameters"
            )
        try:
            bot = float(params_parts[0].strip())
            top = float(params_parts[1].strip())
            n_basis = int(params_parts[2].strip())
        except ValueError as e:
            raise ValueError(f"Invalid parameters in '{hrf_string}'. Expected numeric values: {e}") from e

        if bot >= top:
            raise ValueError(f"In '{hrf_string}': bot ({bot}) must be < top ({top})")

        min_n = 3 if model_name == "TENTZERO" else 2
        if n_basis < min_n:
            raise ValueError(f"In '{hrf_string}': n_basis must be >= {min_n}, got {n_basis}")

        return model_name, {"bot": bot, "top": top, "n_basis": n_basis}

    # Handle simple single-parameter models (SPMG1, BLOCK, etc.)
    else:
        try:
            duration = float(params_str)
        except ValueError as err:
            raise ValueError(
                f"Invalid duration in HRF model '{hrf_string}'. "
                f"Expected numeric value or use TENT(bot,top,n) for multi-parameter models"
            ) from err

        return model_name, duration


def load_and_pad_ortvec(
    filepath: str | Path,
    run_number: int,
    n_timepoints_per_run: list[int],
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
            f"Invalid run_number={run_number}. Must be between 1 and {len(n_timepoints_per_run)}"
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
    run_start = sum(n_timepoints_per_run[: run_number - 1])
    run_end = run_start + expected_rows
    padded[run_start:run_end, :] = data

    return padded


def build_design_matrix(
    timing_files: list[str | Path],
    stim_labels: list[str],
    n_timepoints_per_run: list[int],
    tr: float,
    polort: int = 3,
    hrf_models: str | list[str] | None = None,
    im_mode: bool | list[bool] | None = None,
    padortvec_files: list[tuple[str | Path, str, int]] | None = None,
    ortvec_files: list[tuple[str | Path, str]] | None = None,
    extra_regressors: list[np.ndarray] | None = None,
    extra_regressor_labels: list[str] | None = None,
) -> tuple[np.ndarray, list[str], list[int], dict]:
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
        raise ValueError(
            f"stim_labels length ({len(stim_labels)}) must match timing_files ({n_stim})"
        )

    # Handle HRF models - can be single string or list
    if hrf_models is None:
        hrf_models = ["SPMG1(0)"] * n_stim  # Default: impulse with SPM HRF
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
                    padortvec_labels_list.append(f"{label}[{col_idx_local}]")

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
                    ortvec_labels_list.append(f"{label}[{col_idx_local}]")

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
                regressor_labels.append(f"Run#{run_idx + 1}Pol#{p}")
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

    # 3. Add stimulus regressors via the microtime pipeline.
    # ------------------------------------------------------------------
    # All stim regressors are built at microtime_dt = 0.1 s (the codebase
    # convention) via:
    #   1) get_spmg1_hrf(microtime_dt)               — proper AFNI SPMG1
    #   2) create_onset_matrix_microtime(...)         — boxcar at microtime
    #   3) convolve_hrf_microtime(...)                — convolve, peak-norm,
    #                                                   downsample to TR
    # Sub-TR events (duration < TR/2) survive correctly because everything
    # below the TR grid lives on the fine grid until the final downsample.
    # ------------------------------------------------------------------
    from fastfuncstuff.design.hrf import get_spmg1_hrf
    from fastfuncstuff.design.matrices import convolve_hrf_microtime
    from fastfuncstuff.utils import get_device as _get_device

    microtime_dt = 0.1
    stim_device = _get_device()

    # Split stim conditions by HRF type. Each condition is either:
    #   - regular (one regressor),
    #   - IM (one regressor per event — each event becomes its own "condition"
    #     in the onset matrix, then is convolved like any other).
    # We process SPMG1 and BLOCK groups separately because BLOCK skips
    # convolution entirely.
    expanded_onsets: list[list[np.ndarray]] = []
    expanded_durations: list[float] = []
    expanded_labels: list[str] = []
    expanded_hrf_types: list[str] = []

    for stim_idx in range(n_stim):
        hrf_type = hrf_types[stim_idx]
        if hrf_type not in ("SPMG1", "BLOCK"):
            raise ValueError(
                f"Unknown HRF type: {hrf_type}. Supported: SPMG1, BLOCK"
            )

        duration = stim_durations[stim_idx]
        if im_mode[stim_idx]:
            # IM: each event becomes its own condition with onsets per run.
            event_idx = 0
            for run_idx in range(n_runs):
                for onset in all_onsets[stim_idx][run_idx]:
                    per_run = [
                        np.array([onset], dtype=np.float64) if r == run_idx
                        else np.array([], dtype=np.float64)
                        for r in range(n_runs)
                    ]
                    expanded_onsets.append(per_run)
                    expanded_durations.append(duration)
                    expanded_labels.append(f"{stim_labels[stim_idx]}#{event_idx}")
                    expanded_hrf_types.append(hrf_type)
                    event_idx += 1
        else:
            expanded_onsets.append(
                [np.asarray(all_onsets[stim_idx][r], dtype=np.float64)
                 for r in range(n_runs)]
            )
            expanded_durations.append(duration)
            expanded_labels.append(f"{stim_labels[stim_idx]}#0")
            expanded_hrf_types.append(hrf_type)

    if expanded_onsets:
        onset_matrix = create_onset_matrix_microtime(
            all_onsets=expanded_onsets,
            run_starts=run_starts,
            tr=tr,
            n_timepoints=total_timepoints,
            microtime_dt=microtime_dt,
            stim_durations=expanded_durations,
            device=stim_device,
        )

        # SPMG1 columns get HRF convolution; BLOCK columns are downsampled
        # without convolution.
        spmg_mask = np.array([h == "SPMG1" for h in expanded_hrf_types])

        # SPMG1 path.
        if spmg_mask.any():
            hrf_micro = get_spmg1_hrf(
                microtime_dt=microtime_dt,
                stim_duration=0.0,
                hrf_duration=32.0,
                normalize_peak=True,
                device=stim_device,
            )
            convolved = convolve_hrf_microtime(
                onsets_microtime=onset_matrix[:, spmg_mask],
                hrf=hrf_micro,
                n_timepoints=total_timepoints,
                tr=tr,
                microtime_dt=microtime_dt,
                run_starts=run_starts,
                device=stim_device,
            )
            spmg_cols = convolved.cpu().numpy()
        else:
            spmg_cols = None

        # BLOCK path — no HRF convolution. Sample the boxcar at TR grid.
        if (~spmg_mask).any():
            bins_per_tr = int(round(tr / microtime_dt))
            block_cols = (
                onset_matrix[:, ~spmg_mask][::bins_per_tr][:total_timepoints]
                .cpu().numpy()
            )
        else:
            block_cols = None

        # Splice columns back in the original expanded order.
        spmg_iter = iter(range(spmg_cols.shape[1])) if spmg_cols is not None else iter(())
        block_iter = iter(range(block_cols.shape[1])) if block_cols is not None else iter(())
        for k, label in enumerate(expanded_labels):
            if spmg_mask[k]:
                design_matrix[:, col_idx] = spmg_cols[:, next(spmg_iter)]
            else:
                design_matrix[:, col_idx] = block_cols[:, next(block_iter)]
            stim_indices.append(col_idx)
            regressor_labels.append(label)
            col_idx += 1

    # 4. Add extra regressors (if any)
    if extra_regressors:
        for _idx, reg in enumerate(extra_regressors):
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
                regressor_labels.append(f"extra_{idx}")

    # Create metadata
    nuisance_indices = polort_indices + padortvec_indices + ortvec_indices + extra_indices

    metadata = {
        "stim_indices": stim_indices,
        "nuisance_indices": nuisance_indices,
        "polort_indices": polort_indices,
        "padortvec_indices": padortvec_indices,
        "ortvec_indices": ortvec_indices,
        "extra_indices": extra_indices,
        "n_runs": n_runs,
        "n_timepoints_per_run": n_timepoints_per_run,
        "tr": tr,
        "hrf_models": hrf_models,
        "hrf_types": hrf_types,
        "stim_durations": stim_durations,
        "polort": polort,
    }

    return design_matrix, regressor_labels, run_starts, metadata


def write_afni_xmat(
    filepath: str | Path,
    design_matrix: np.ndarray,
    regressor_labels: list[str],
    run_starts: list[int],
    metadata: dict,
    glt_contrasts: list[tuple[str | list[str], str]] | None = None,
    command_line: str | None = None,
    good_list: list[int] | np.ndarray | None = None,
    n_row_full: int | None = None,
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
    _n_runs = metadata["n_runs"]
    tr = metadata["tr"]
    stim_indices = metadata["stim_indices"]

    # Extract unique stimulus labels and their column ranges
    stim_labels_list = []
    stim_bots = []
    stim_tops = []
    seen_stim = set()

    if len(stim_indices) > 0:
        for idx in stim_indices:
            label = regressor_labels[idx]
            # Remove #N suffix to get base label (e.g., movie#0 -> movie)
            base_label = label.split("#")[0] if "#" in label else label
            if base_label not in seen_stim:
                stim_labels_list.append(base_label)
                seen_stim.add(base_label)

        # For each unique stimulus, find its bottom and top column indices
        for base_label in stim_labels_list:
            cols = [i for i in stim_indices if regressor_labels[i].split("#")[0] == base_label]
            stim_bots.append(min(cols))
            stim_tops.append(max(cols))

        n_stim = len(stim_labels_list)
    else:
        n_stim = 0

    # Build column groups in column order (AFNI semantics):
    #   -1     polynomial drift (polort_indices)
    #    0     baseline / motion / generic nuisance (padortvec, ortvec, extra)
    #    1..N  one group per stim, numbered by appearance order
    # The string is N@v compressed for run-length and a..b compressed for
    # sequential integers so the output matches 3dDeconvolve byte-for-byte
    # on AFNI-style designs.
    polort_set = set(metadata.get("polort_indices", []))
    nuisance0_set = set(
        metadata.get("padortvec_indices", [])
        + metadata.get("ortvec_indices", [])
        + metadata.get("extra_indices", [])
    )
    # Map each stim column to its 1-indexed stim group (by stim_labels_list order).
    stim_col_to_group: dict[int, int] = {}
    for stim_idx, base_label in enumerate(stim_labels_list):
        cols = [
            i for i in stim_indices
            if regressor_labels[i].split("#")[0] == base_label
        ]
        for c in cols:
            stim_col_to_group[c] = stim_idx + 1

    per_col_group: list[int] = []
    for col in range(n_regressors):
        if col in polort_set:
            per_col_group.append(-1)
        elif col in stim_col_to_group:
            per_col_group.append(stim_col_to_group[col])
        elif col in nuisance0_set:
            per_col_group.append(0)
        else:
            # Unaccounted column — fall back to nuisance baseline.
            per_col_group.append(0)
    col_groups = _compress_int_sequence(per_col_group)

    # Create header
    with open(filepath, "w") as f:
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

        # GoodList: indices in the *full* (uncensored) dataset that survived
        # into this matrix. Length must equal n_timepoints. Without censoring
        # this is just 0..n_timepoints-1. AFNI compresses contiguous runs.
        if good_list is None:
            good_indices = list(range(n_timepoints))
            full_len = n_timepoints
        else:
            good_indices = [int(i) for i in good_list]
            if len(good_indices) != n_timepoints:
                raise ValueError(
                    f"good_list length ({len(good_indices)}) does not match "
                    f"design matrix rows ({n_timepoints})."
                )
            full_len = n_row_full if n_row_full is not None else (max(good_indices) + 1)
            if full_len < max(good_indices) + 1:
                raise ValueError(
                    f"n_row_full ({full_len}) is smaller than max(good_list)+1 "
                    f"({max(good_indices) + 1})."
                )
        f.write(f'#  GoodList = "{_compress_index_runs(good_indices)}"\n')
        f.write(f'#  NRowFull = "{full_len}"\n')

        # Run starts
        run_starts_str = ",".join(map(str, run_starts))
        f.write(f'#  RunStart = "{run_starts_str}"\n')

        # Stimulus info
        f.write(f'#  Nstim = "{n_stim}"\n')
        if n_stim > 0:
            # AFNI emits StimBots/StimTops as a possibly-compressed list
            # (sequential integers fold into "a..b" form). One bot per stim,
            # one top per stim, in StimLabels order.
            f.write(f'#  StimBots = "{_compress_int_sequence(stim_bots)}"\n')
            f.write(f'#  StimTops = "{_compress_int_sequence(stim_tops)}"\n')

            stim_labels_str = " ; ".join(stim_labels_list)
            f.write(f'#  StimLabels = "{stim_labels_str}"\n')

        # GLT contrasts. Each entry of glt_contrasts is (sym_str_or_list, label):
        #   - str  -> rows parsed by parse_glt_string (1 row = t-test, N = F)
        #   - list -> already a list of SYM: row strings (F-test convenience)
        if glt_contrasts:
            f.write(f'#  Nglt = "{len(glt_contrasts)}"\n')
            glt_labels = [label for _, label in glt_contrasts]
            f.write(f'#  GltLabels = "{" ; ".join(glt_labels)}"\n')

            stim_ranges = dict(
                zip(stim_labels_list, zip(stim_bots, stim_tops, strict=True), strict=True)
            )

            for glt_idx, (contrast_str, _label) in enumerate(glt_contrasts):
                rows, _valid = parse_glt_string(contrast_str)
                glt_matrix = glt_rows_to_matrix(rows, regressor_labels, stim_ranges)
                n_rows = glt_matrix.shape[0]

                # AFNI compact notation: "r,nreg,v0,v1,...,v_{r*nreg-1}" with
                # leading and trailing zero runs compressed via N@0. We flatten
                # row-major (AFNI's convention per remlfit.html).
                flat = glt_matrix.reshape(-1)
                matrix_str = _afni_compact_vector(flat, leading_count=2_000_000)
                # Prefix with r,nreg
                header = f"{n_rows},{n_regressors}"
                if matrix_str:
                    matrix_str = f"{header},{matrix_str}"
                else:
                    matrix_str = header

                f.write(f'#  GltMatrix_{glt_idx:06d} = "{matrix_str}"\n')

        # Basis function info
        if n_stim > 0:
            f.write(f'#  BasisNstim = "{n_stim}"\n')

            # Get unique stimuli
            stim_info = {}
            for idx in stim_indices:
                label = regressor_labels[idx]
                # Remove #N suffix to get base label (e.g., movie#0 -> movie)
                base_label = label.split("#")[0] if "#" in label else label

                if base_label not in stim_info:
                    # Find which stimulus this is
                    stim_idx_in_list = None
                    for i, sl in enumerate(stim_labels_list):
                        if sl == base_label:
                            stim_idx_in_list = i
                            break

                    if stim_idx_in_list is not None:
                        # Get HRF model and duration
                        hrf_type = (
                            metadata.get("hrf_types", [])[stim_idx_in_list]
                            if stim_idx_in_list < len(metadata.get("hrf_types", []))
                            else "SPMG1"
                        )
                        duration = (
                            metadata.get("stim_durations", [])[stim_idx_in_list]
                            if stim_idx_in_list < len(metadata.get("stim_durations", []))
                            else 0.0
                        )

                        # Find column range for this stimulus
                        # Match labels that are exactly base_label#N
                        cols_for_stim = [
                            i
                            for i, l in enumerate(regressor_labels)
                            if l.split("#")[0] == base_label and "#" in l
                        ]

                        stim_info[base_label] = {
                            "idx": stim_idx_in_list + 1,
                            "hrf": hrf_type,
                            "duration": duration,
                            "cols": cols_for_stim,
                        }

            # Write basis info for each stimulus
            for stim_label in stim_labels_list:
                if stim_label in stim_info:
                    info = stim_info[stim_label]
                    idx = info["idx"]

                    # Determine stim option (IM vs regular)
                    # IM mode: multiple columns (label#0, label#1, ...)
                    # Standard mode: single column (label#0)
                    is_im = len(info["cols"]) > 1
                    option = "-stim_times_IM" if is_im else "-stim_times"

                    f.write(f'#  BasisOption_{idx:06d} = "{option}"\n')
                    f.write(f'#  BasisName_{idx:06d} = "{stim_label}"\n')
                    f.write(
                        f'#  BasisFormula_{idx:06d} = "{info["hrf"]}({info["duration"]:.0f})"\n'
                    )

                    # Column range
                    col_start = min(info["cols"])
                    col_end = max(info["cols"])
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


def create_onset_matrix_microtime(
    all_onsets: list[list[np.ndarray]],
    run_starts: list[int],
    tr: float,
    n_timepoints: int,
    microtime_dt: float,
    stim_durations: list[float],
    device: torch.device,
) -> torch.Tensor:
    """
    Create binary onset matrix at microtime resolution.

    Parameters
    ----------
    all_onsets : list of list of np.ndarray
        Onsets organized as [condition][run] -> np.ndarray of onset times
    run_starts : list of int
        Starting timepoint for each run (in TRs)
    tr : float
        Repetition time in seconds
    n_timepoints : int
        Total number of TR timepoints
    microtime_dt : float
        Microtime resolution in seconds (e.g., 0.1 = 100ms resolution)
    stim_durations : list of float
        Duration in seconds for each condition
    device : torch.device
        Device for output tensor

    Returns
    -------
    onset_matrix : torch.Tensor
        (n_microtime, n_conditions) matrix with boxcar values
    """
    bins_per_tr = int(round(tr / microtime_dt))
    n_microtime = n_timepoints * bins_per_tr
    n_conditions = len(all_onsets)
    n_runs = len(run_starts)

    # Initialize onset matrix
    onset_matrix = torch.zeros((n_microtime, n_conditions), dtype=torch.float32, device=device)

    for cond_idx in range(n_conditions):
        duration = stim_durations[cond_idx]
        duration_bins = max(1, int(np.round(duration / microtime_dt)))

        for run_idx in range(n_runs):
            onsets = all_onsets[cond_idx][run_idx]
            # Use bins_per_tr-based offset to stay consistent with convolve_hrf_microtime's
            # sampling grid. convolve_hrf_microtime downsamples at every bins_per_tr bins,
            # so run boundaries must align to that grid, not to real-time seconds.
            run_start_micro = run_starts[run_idx] * bins_per_tr

            for onset_time in onsets:
                # onset_time is relative to run start, in seconds
                onset_bin = run_start_micro + int(np.round(onset_time / microtime_dt))

                if 0 <= onset_bin < n_microtime:
                    end_bin = min(onset_bin + duration_bins, n_microtime)
                    onset_matrix[onset_bin:end_bin, cond_idx] = 1.0

    return onset_matrix


def round_onsets(
    all_onsets: list[list[np.ndarray]],
    tr: float,
    threshold: float = 0.7,
) -> list[list[np.ndarray]]:
    """
    Snap onset times to TR boundaries.

    For each onset *t*:

    - ``remainder = t % tr``
    - if ``remainder / tr >= threshold`` → ceil to start of next TR
    - else → floor to start of current TR

    ``threshold=0.5`` is equivalent to standard nearest-TR rounding.
    ``threshold=0.7`` (default) biases toward floor: only rounds up when
    the onset is 70 %+ through a TR interval.  This is useful for designs
    where events are nominally TR-locked but have small positive jitter.

    Parameters
    ----------
    all_onsets : list[list[ndarray]]
        Nested structure ``all_onsets[condition_idx][run_idx]``.
    tr : float
        Repetition time in seconds.
    threshold : float, default=0.7
        Fractional position within a TR above which an onset rounds up.

    Returns
    -------
    list[list[ndarray]]
        Same nested structure with rounded onset times (float64).
    """
    result: list[list[np.ndarray]] = []
    for cond_onsets in all_onsets:
        cond_result: list[np.ndarray] = []
        for run_onsets in cond_onsets:
            arr = np.asarray(run_onsets, dtype=np.float64)
            if arr.size == 0:
                cond_result.append(arr)
                continue
            remainder = arr % tr
            rounded = np.where(
                remainder / tr >= threshold,
                np.ceil(arr / tr) * tr,
                np.floor(arr / tr) * tr,
            )
            cond_result.append(rounded)
        result.append(cond_result)
    return result


def parse_durations(
    durations_arg: list[str],
    n_conditions: int,
    condition_labels: list[str],
) -> list[float]:
    """
    Parse durations argument.

    Supports formats:
    - "2.0" -> single value
    - "3,20" -> 20 repeats of 3.0 (value,count)
    - Single value: applies to all conditions
    - Multiple values: one per condition (in same order as onsets)
    """
    import sys

    durations_parsed = []

    # Parse each duration spec (can be "value" or "value,count")
    for d in durations_arg:
        if "," in d:
            # Parse "value,count" format (e.g., "3,20" -> [3, 3, 3, ...])
            try:
                value_str, count_str = d.split(",")
                value = float(value_str)
                count = int(count_str)
                durations_parsed.extend([value] * count)
            except (ValueError, IndexError):
                print(
                    f"ERROR: Invalid duration format '{d}'. Use 'value' or 'value,count' (e.g., '3,20')"
                )
                sys.exit(1)
        else:
            # Single value
            try:
                durations_parsed.append(float(d))
            except ValueError:
                print(f"ERROR: Could not parse duration '{d}' as float")
                sys.exit(1)

    # Check if parsed durations match conditions
    if len(durations_parsed) == 1:
        # Single duration for all conditions
        return durations_parsed * n_conditions
    elif len(durations_parsed) == n_conditions:
        # One duration per condition
        return durations_parsed
    else:
        print(
            f"ERROR: Number of durations ({len(durations_parsed)}) must be 1 or match "
            f"number of conditions ({n_conditions})"
        )
        print(f"  Conditions: {condition_labels}")
        sys.exit(1)


# ============================================================================
# Shared per-run task-design builder (FIR / TENT / CSPLIN / assumed-HRF).
# ============================================================================
#
# This is the single source of truth for "given per-condition / per-run
# onset times, build the per-run task-only design tensors."  Both
# ``ffs_librarian`` and ``ffs_deconvolve`` (and any future CLI doing
# FIR/TENT modelling) call this — duplicating the loop has bitten us at
# least twice (axis-flipped slice; externally-built polynomials competing
# with fit_glm's automatic ones).
#
# Invariants:
#
#   1. Returns a LIST of per-run task-only tensors.  Callers always pass
#      that list to :func:`fit_glm` and let ``fit_glm`` build
#      block-diagonal polynomials via its ``max_poly_degree`` argument.
#      Polynomials are NEVER built externally — that path is the
#      historical foot-gun and is explicitly out of scope here.
#   2. Column ordering is deterministic: condition-major, basis-minor
#      (``cond0#0, cond0#1, …, cond0#K, cond1#0, …``).  Column labels
#      track the same ordering.
#   3. "auto" basis resolves via :func:`is_tr_locked` — FIR if every
#      onset is within ``tr_locked_threshold`` of a TR boundary, TENT
#      otherwise.  Auto-window from event durations + a configurable
#      floor.  These bells & whistles live here once, not duplicated
#      across CLIs.
#   4. Pure numpy/torch — no CLI argument parsing or I/O.  Tests can
#      drive it directly.


@dataclass
class TaskDesignResult:
    """Container returned by :func:`build_per_run_task_designs`.

    Attributes
    ----------
    per_run : list[torch.Tensor]
        One ``(n_tp_run, sum(n_basis_per_condition))`` tensor per run,
        in run order.  Columns are condition-major, basis-minor.  This
        is the list to pass as ``design=`` to
        :func:`fastfuncstuff.glm.core.fit_glm` (along with per-run data
        as a list).
    column_labels : list[str]
        Length ``sum(n_basis_per_condition)``.  Format
        ``"<condition_label>#<basis_idx>"`` so that downstream tools
        can recover the (condition, basis) of each beta.
    n_basis_per_condition : list[int]
        Number of regressors per condition (after any TENTzero edge
        drop).  Indices into ``column_labels`` for condition *c* span
        ``sum(n_basis_per_condition[:c])`` to
        ``sum(n_basis_per_condition[:c+1])``.
    basis_resolved : str
        The basis actually used after ``"auto"`` resolution.  One of
        ``"FIR"``, ``"TENT"``, ``"TENTzero"``, ``"CSPLIN"``,
        ``"CSPLINzero"``, ``"assumed"``.
    fir_window_s : list[tuple[float, float]] | None
        Per-condition ``(bot, top)`` window (seconds) that was used to
        build the basis.  ``None`` for ``"assumed"``.
    lag_times_s : list[np.ndarray] | None
        Per-condition lag-time grid.  For FIR these are TR multiples
        ``[0, tr, 2tr, …]``; for TENT/CSPLIN they are the knot
        positions in ``[bot, top]``.  ``None`` for ``"assumed"``.
    notes : list[str]
        Human-readable diagnostics ("Auto-resolved basis to TENT
        because onsets are not TR-locked.", etc.).  Caller logs these.
    """

    per_run: list[torch.Tensor]
    column_labels: list[str]
    n_basis_per_condition: list[int]
    basis_resolved: str
    fir_window_s: list[tuple[float, float]] | None = None
    lag_times_s: list[np.ndarray] | None = None
    notes: list[str] = field(default_factory=list)


_BASIS_CHOICES = (
    "auto",
    "FIR",
    "TENT",
    "TENTzero",
    "CSPLIN",
    "CSPLINzero",
    "assumed",
)


def _resolve_basis_auto(
    onsets_per_cond_per_run: list[list[np.ndarray]],
    tr: float,
    threshold: float,
) -> str:
    """Pick FIR (TR-locked) or TENT (not TR-locked) for ``basis='auto'``.

    Concatenates every onset across all conditions × runs and asks
    :func:`is_tr_locked`.  All-locked → FIR, otherwise → TENT.  A
    degenerate empty input falls back to FIR (the no-op case).
    """
    all_t = [
        float(t)
        for cond_runs in onsets_per_cond_per_run
        for run_onsets in cond_runs
        for t in (run_onsets.tolist() if run_onsets.size else [])
    ]
    if not all_t:
        return "FIR"
    return "FIR" if is_tr_locked(all_t, tr, threshold=threshold) else "TENT"


def _resolve_windows(
    durations_per_condition: list[float] | None,
    fir_window_s: float | list[float] | list[tuple[float, float]] | None,
    fir_window_min_s: float,
    tr: float,
    n_conditions: int,
) -> list[tuple[float, float]]:
    """Compute per-condition ``(bot, top)`` windows in seconds.

    Resolution order:

    1. If ``fir_window_s`` is a list of ``(bot, top)`` tuples with the
       right length → use as-is (advanced caller specified exact
       windows, e.g. for AFNI-style TENT(bot,top,n)).
    2. If ``fir_window_s`` is a list of scalars → ``(0, top_i)`` per
       condition.
    3. If ``fir_window_s`` is a scalar → ``(0, top)`` for every cond.
    4. Else, auto-derive from ``durations_per_condition`` via
       :func:`estimate_hrf_window` (in seconds), with a global floor
       of ``fir_window_min_s`` applied to the maximum.

    The ``bot`` value is currently always 0 for the auto path; only
    explicit ``(bot, top)`` tuples support non-zero starts.
    """
    from fastfuncstuff.design.hrf import estimate_hrf_window  # local: avoid cycle

    # Form 1: user supplied per-cond (bot, top) tuples
    if (
        isinstance(fir_window_s, list)
        and fir_window_s
        and isinstance(fir_window_s[0], tuple)
    ):
        if len(fir_window_s) != n_conditions:
            raise ValueError(
                f"fir_window_s as list of tuples must have {n_conditions} "
                f"entries; got {len(fir_window_s)}"
            )
        return [(float(b), float(t)) for b, t in fir_window_s]

    # Form 2: user supplied per-cond scalar tops
    if isinstance(fir_window_s, list):
        if len(fir_window_s) != n_conditions:
            raise ValueError(
                f"fir_window_s as list of scalars must have {n_conditions} "
                f"entries; got {len(fir_window_s)}"
            )
        return [(0.0, float(t)) for t in fir_window_s]

    # Form 3: scalar applies to all
    if fir_window_s is not None:
        return [(0.0, float(fir_window_s))] * n_conditions

    # Form 4: auto from durations
    if durations_per_condition is None:
        # No duration info & no override: fall back to a single
        # conservative window of fir_window_min_s.
        top = max(fir_window_min_s, 20.0)
        return [(0.0, float(top))] * n_conditions

    per_cond_top = [
        max(1, estimate_hrf_window(d, tr, threshold=0.10)) * tr
        for d in durations_per_condition
    ]
    # Apply the floor at the *maximum* across conditions — we want a
    # consistent FIR/TENT length to keep the design rectangular, and
    # the floor matches NSD's 30 s default (configurable).
    top_max = max(fir_window_min_s, max(per_cond_top))
    return [(0.0, top_max)] * n_conditions


def _lag_times_for(basis: str, window: tuple[float, float], n_basis: int) -> np.ndarray:
    """Return the time grid corresponding to a basis' knots/lags."""
    bot, top = window
    if basis == "FIR":
        # FIR uses TR multiples starting at bot.  Caller has already
        # determined n_basis from window/tr so this is just a grid.
        return bot + np.arange(n_basis) * ((top - bot) / max(n_basis, 1))
    # TENT / CSPLIN: knots equally spaced from bot to top.
    if basis in ("TENTzero", "CSPLINzero"):
        # Edge knots are dropped from the design.  The visible knots
        # are at indices 1..n_basis (the kept regressors), so we
        # generate the interior positions.
        full = np.linspace(bot, top, n_basis + 2)
        return full[1:-1]
    return np.linspace(bot, top, n_basis)


def build_per_run_task_designs(
    onsets_per_cond_per_run: list[list[np.ndarray]],
    n_timepoints_per_run: list[int],
    tr: float,
    *,
    basis: Literal[
        "auto", "FIR", "TENT", "TENTzero", "CSPLIN", "CSPLINzero", "assumed"
    ] = "auto",
    condition_labels: list[str] | None = None,
    durations_per_condition: list[float] | None = None,
    fir_window_s: float | list[float] | list[tuple[float, float]] | None = None,
    fir_window_min_s: float = 0.0,
    tent_n_basis: int | list[int] | None = None,
    hrf: torch.Tensor | np.ndarray | None = None,
    tr_locked_threshold: float = 0.1,
    device: torch.device | None = None,
) -> TaskDesignResult:
    """Build per-run task-only design tensors for FIR / TENT / CSPLIN / assumed.

    **Single source of truth** — both ``ffs_librarian`` and
    ``ffs_deconvolve`` route through here so axis-flip and
    polynomial-double-counting bugs can't lurk in one tool but not the
    other.

    Parameters
    ----------
    onsets_per_cond_per_run : list[list[np.ndarray]]
        Nested ``[condition][run] → np.ndarray of onset times (s)``,
        where times are relative to the START of that run (not the
        global concatenated timeline).
    n_timepoints_per_run : list[int]
        Number of TR samples in each run.  Must match the outer
        run-dimension of ``onsets_per_cond_per_run``.
    tr : float
        Repetition time in seconds.
    basis : str, default ``"auto"``
        Basis selection.  ``"auto"`` chooses FIR if all onsets lie
        within ``tr_locked_threshold`` of a TR boundary, otherwise
        TENT — see :func:`_resolve_basis_auto`.
    condition_labels : list[str], optional
        Per-condition display names used in ``column_labels``.
        Defaults to ``["cond0", "cond1", …]``.
    durations_per_condition : list[float], optional
        Event durations (s).  Used only by the auto-window path to
        derive an appropriate FIR/TENT window via
        :func:`estimate_hrf_window`.
    fir_window_s : float or list, optional
        Override the window.  Three accepted forms (see
        :func:`_resolve_windows`):

        - scalar → ``(0, top)`` for every condition;
        - list of scalars → per-condition ``(0, top_i)``;
        - list of ``(bot, top)`` tuples → per-condition explicit
          window (the only form that supports non-zero ``bot``).
    fir_window_min_s : float, default 0.0
        Floor applied to the auto-estimated window length.  NSD-style
        callers pass 30.0; ``ffs_deconvolve`` typically passes 0.0.
    tent_n_basis : int or list[int], optional
        Override the number of TENT/CSPLIN basis functions.  Defaults
        to "one knot per TR + 1" within the window.  For TENTzero /
        CSPLINzero this is the *full* knot count; the actual
        regressor count is ``tent_n_basis - 2``.
    hrf : tensor/ndarray, optional
        Required when ``basis="assumed"``.  1-D HRF kernel sampled at
        TR resolution (or microtime if you've upsampled the onsets).
        Each condition gets convolved with this kernel.
    tr_locked_threshold : float, default 0.1
        Tolerance (fraction of TR) used by :func:`is_tr_locked` when
        ``basis="auto"``.
    device : torch.device, optional
        Where to materialize the design tensors.  ``None`` → CPU.

    Returns
    -------
    TaskDesignResult
        See dataclass.  ``result.per_run`` is the per-run task-only
        list to pass to ``fit_glm``; *let fit_glm add polynomials*.

    Notes
    -----
    Anti-pattern this function exists to prevent::

        # DO NOT do this:
        task = build_my_task_design(...)
        polys = build_block_diagonal_polys(...)
        design = torch.cat([task, polys], dim=1)
        fit_glm(data=concat_data, design=design, max_poly_degree=-1)

    Instead::

        result = build_per_run_task_designs(...)
        fit_glm(data=per_run_data_list, design=result.per_run,
                max_poly_degree=polort)  # fit_glm owns the polynomials.

    The first form *does* work but has historically diverged between
    callers in subtle ways (axis flips, off-by-one polort, block
    misalignment after a run is excluded).  The second form is the
    "one right way".
    """
    if basis not in _BASIS_CHOICES:
        raise ValueError(
            f"basis must be one of {_BASIS_CHOICES}; got {basis!r}"
        )

    n_conditions = len(onsets_per_cond_per_run)
    if n_conditions == 0:
        raise ValueError("onsets_per_cond_per_run is empty (no conditions).")
    n_runs = len(n_timepoints_per_run)
    if n_runs == 0:
        raise ValueError("n_timepoints_per_run is empty (no runs).")
    for c, cond_runs in enumerate(onsets_per_cond_per_run):
        if len(cond_runs) != n_runs:
            raise ValueError(
                f"Condition {c} has {len(cond_runs)} runs but "
                f"n_timepoints_per_run has {n_runs}."
            )

    if condition_labels is None:
        condition_labels = [f"cond{i}" for i in range(n_conditions)]
    elif len(condition_labels) != n_conditions:
        raise ValueError(
            f"condition_labels has {len(condition_labels)} entries but "
            f"there are {n_conditions} conditions."
        )

    if device is None:
        device = torch.device("cpu")

    notes: list[str] = []

    # ----- Resolve basis ('auto' → FIR/TENT) ----------------------------
    resolved = basis
    if basis == "auto":
        resolved = _resolve_basis_auto(
            onsets_per_cond_per_run, tr, tr_locked_threshold
        )
        notes.append(
            f"Auto-resolved basis to {resolved} "
            f"(TR-lock tolerance {tr_locked_threshold:.2f} of TR={tr}s)."
        )

    if resolved == "assumed":
        if hrf is None:
            raise ValueError("basis='assumed' requires the `hrf` argument.")
        return _build_assumed_designs(
            onsets_per_cond_per_run, n_timepoints_per_run, tr,
            condition_labels=condition_labels, hrf=hrf, device=device,
            notes=notes,
        )

    # ----- Per-condition windows (bot, top) -----------------------------
    windows = _resolve_windows(
        durations_per_condition=durations_per_condition,
        fir_window_s=fir_window_s,
        fir_window_min_s=fir_window_min_s,
        tr=tr,
        n_conditions=n_conditions,
    )

    # ----- Per-condition basis counts -----------------------------------
    n_basis_per_cond: list[int] = []
    if tent_n_basis is not None:
        if isinstance(tent_n_basis, int):
            tent_n_basis_list = [tent_n_basis] * n_conditions
        else:
            if len(tent_n_basis) != n_conditions:
                raise ValueError(
                    f"tent_n_basis as list must have {n_conditions} entries"
                )
            tent_n_basis_list = list(tent_n_basis)
    else:
        tent_n_basis_list = [None] * n_conditions  # auto

    for c in range(n_conditions):
        bot, top = windows[c]
        if resolved == "FIR":
            # FIR samples at integer TR offsets.
            n = max(1, int(np.ceil((top - bot) / tr)))
        else:
            # TENT/CSPLIN: one knot per TR + 1 edge knot, unless overridden.
            if tent_n_basis_list[c] is not None:
                n_knots = int(tent_n_basis_list[c])
            else:
                n_knots = max(2, int(round((top - bot) / tr)) + 1)
            if resolved in ("TENTzero", "CSPLINzero"):
                n = n_knots - 2  # edges dropped
            else:
                n = n_knots
        n_basis_per_cond.append(n)

    # ----- Build per-run designs ----------------------------------------
    per_run: list[torch.Tensor] = []
    for r, n_run_tp in enumerate(n_timepoints_per_run):
        cond_blocks: list[torch.Tensor] = []
        for c in range(n_conditions):
            onset_times = np.asarray(
                onsets_per_cond_per_run[c][r], dtype=np.float64
            )
            bot, top = windows[c]
            if resolved == "FIR":
                # Quantize to nearest TR; build (n_run_tp, 1) onset
                # vector; expand to (n_run_tp, n_lags) via shifts.
                onset_vec = torch.zeros(n_run_tp, 1, device=device)
                if onset_times.size > 0:
                    idx = np.round(onset_times / tr).astype(int)
                    idx = idx[(idx >= 0) & (idx < n_run_tp)]
                    onset_vec[idx, 0] = 1.0
                block = make_fir_design(
                    onset_vec, n_basis_per_cond[c], n_run_tp, device=device
                )
            elif resolved in ("TENT", "TENTzero"):
                n_knots = (
                    n_basis_per_cond[c] + 2
                    if resolved == "TENTzero"
                    else n_basis_per_cond[c]
                )
                block = make_tent_design(
                    [onset_times],
                    bot=bot, top=top, tr=tr,
                    n_timepoints=n_run_tp,
                    n_basis=n_knots,
                    zero_edges=(resolved == "TENTzero"),
                    device=device,
                )
            elif resolved in ("CSPLIN", "CSPLINzero"):
                n_knots = (
                    n_basis_per_cond[c] + 2
                    if resolved == "CSPLINzero"
                    else n_basis_per_cond[c]
                )
                block = make_csplin_design(
                    [onset_times],
                    bot=bot, top=top, tr=tr,
                    n_timepoints=n_run_tp,
                    n_basis=n_knots,
                    zero_edges=(resolved == "CSPLINzero"),
                    device=device,
                )
            else:
                raise ValueError(
                    f"unreachable: basis '{resolved}' not handled "
                    "(should have been caught earlier)"
                )
            cond_blocks.append(block)
        # Condition-major concatenation along regressor axis.
        per_run.append(torch.cat(cond_blocks, dim=1))

    # ----- Column labels, lag times -------------------------------------
    column_labels: list[str] = []
    lag_times_s: list[np.ndarray] = []
    for c in range(n_conditions):
        n = n_basis_per_cond[c]
        for k in range(n):
            column_labels.append(f"{condition_labels[c]}#{k}")
        lag_times_s.append(_lag_times_for(resolved, windows[c], n))

    return TaskDesignResult(
        per_run=per_run,
        column_labels=column_labels,
        n_basis_per_condition=n_basis_per_cond,
        basis_resolved=resolved,
        fir_window_s=list(windows),
        lag_times_s=lag_times_s,
        notes=notes,
    )


@dataclass
class PackedSharedTaskDesign:
    """Concatenated design ready for the canonical multi-run GLM.

    Returned by :func:`pack_for_shared_task_glm` so callers can hand
    ``data_concat`` + ``design_concat`` to :func:`fit_glm` with
    ``max_poly_degree=-1`` and get **shared task betas across runs +
    block-diagonal polynomial nuisance**, which is the canonical
    fMRI multi-run GLM:

    .. code-block:: text

           [  task block  | run0_poly | run1_poly | … ]
           [    (shared    |   ↑↑↑    |    0      |   ]
           [   across runs)|   run0    |          |   ]
           [               |    0     |   run1   |   ]
           [               |          |  poly     |   ]

    Attributes
    ----------
    data_concat : torch.Tensor, (n_voxels, sum n_tp_run)
        Row-concatenated per-run data.
    design_concat : torch.Tensor, (sum n_tp_run, n_task_cols + n_runs * (polort+1))
        Shared task (row-stacked across runs) followed by
        block-diagonal polynomial nuisance.  Pass to ``fit_glm`` with
        ``max_poly_degree=-1`` to prevent double-adding polynomials.
    n_task_cols : int
        Number of task columns at the start of ``design_concat``.
        Extract task betas via ``results.betas[:, :n_task_cols]``.
    column_labels : list[str]
        Length ``design_concat.shape[1]``.  Task labels from
        :class:`TaskDesignResult`, then ``run{r}_poly{k}`` for each
        polynomial column.
    n_tp_per_run : list[int]
        Run lengths in TR, in run order.  Useful for downstream code
        that needs to split predictions/residuals back per run.
    polort : int
        Polynomial degree used.  ``-1`` means no polynomial nuisance
        was added.
    """

    data_concat: torch.Tensor
    design_concat: torch.Tensor
    n_task_cols: int
    column_labels: list[str]
    n_tp_per_run: list[int]
    polort: int


def pack_for_shared_task_glm(
    per_run_data: list[torch.Tensor],
    per_run_task_designs: list[torch.Tensor],
    polort: int,
    *,
    task_column_labels: list[str] | None = None,
    extra_regressors_per_run: list[torch.Tensor] | None = None,
    device: torch.device | None = None,
) -> PackedSharedTaskDesign:
    """Pack per-run task designs into the canonical "shared-task + block-diagonal-polys" GLM form.

    **Why this exists** — :func:`fit_glm`, when handed per-run lists,
    block-diagonalizes the TASK matrix too, which estimates separate
    task betas per run.  For typical fMRI analyses we want a **single
    set of task betas** estimated jointly across all runs (more data
    per parameter, cleaner HRF estimates), so we row-concatenate the
    task block while keeping polynomials per-run on a block diagonal.
    That is the form ``ffs_deconvolve`` historically built by hand;
    this helper makes it the one right way.

    The resulting tensors are designed to be fed to ``fit_glm`` as
    **single tensors** (not lists) with ``max_poly_degree=-1`` to
    suppress the auto-polynomial path:

    .. code-block:: python

        packed = pack_for_shared_task_glm(
            per_run_data, per_run_task_designs, polort=4,
            task_column_labels=task_design_result.column_labels,
        )
        result = fit_glm(
            data=packed.data_concat,
            design=packed.design_concat,
            max_poly_degree=-1,        # polys already in design_concat
        )
        task_betas = result.betas[:, :packed.n_task_cols]

    Parameters
    ----------
    per_run_data : list[torch.Tensor], each shape (n_voxels, n_tp_run)
        Voxel data per run, in run order.  Must all share the same
        ``n_voxels`` (i.e. the same brain mask).
    per_run_task_designs : list[torch.Tensor], each shape (n_tp_run, n_task)
        Per-run task design (output of
        :func:`build_per_run_task_designs`).  All runs must have the
        same ``n_task`` (the task model is shared by definition).
    polort : int
        Polynomial nuisance degree.  ``-1`` disables.  ``0`` adds a
        run-specific constant; ``4`` adds Legendre degrees 0–4 per
        run (5 columns × n_runs total).
    task_column_labels : list[str], optional
        Labels for the task columns.  Defaults to
        ``["task0", "task1", …]``.
    extra_regressors_per_run : list[torch.Tensor], optional
        Per-run external nuisance (e.g. motion, GLMdenoise PCs).  Each
        tensor must have shape ``(n_tp_run, n_extra)`` with the same
        ``n_extra`` across runs.  These are appended to each run's
        polynomial block in the block-diagonal section, so they
        remain run-specific (no shared external nuisance — that's the
        canonical convention).
    device : torch.device, optional
        Device for the output tensors.  ``None`` → CPU.

    Returns
    -------
    PackedSharedTaskDesign
        Ready to feed to ``fit_glm`` (see class docstring).

    Notes
    -----
    Equivalence with the old hand-built deconvolve path is
    bit-for-bit when both use the same Legendre polynomial
    construction (``construct_polynomial_matrix`` in glm/core.py
    delegates to ``legendre_polynomials`` in this module, so they
    agree).
    """
    if not per_run_data:
        raise ValueError("per_run_data is empty.")
    if len(per_run_data) != len(per_run_task_designs):
        raise ValueError(
            f"per_run_data has {len(per_run_data)} runs but "
            f"per_run_task_designs has {len(per_run_task_designs)}."
        )
    n_runs = len(per_run_data)
    n_voxels = per_run_data[0].shape[0]
    n_task = per_run_task_designs[0].shape[1]
    for r in range(1, n_runs):
        if per_run_data[r].shape[0] != n_voxels:
            raise ValueError(
                f"per_run_data[{r}] has {per_run_data[r].shape[0]} voxels "
                f"but run 0 has {n_voxels}."
            )
        if per_run_task_designs[r].shape[1] != n_task:
            raise ValueError(
                f"per_run_task_designs[{r}] has {per_run_task_designs[r].shape[1]} "
                f"task columns but run 0 has {n_task} — task model must be shared."
            )

    if extra_regressors_per_run is not None:
        if len(extra_regressors_per_run) != n_runs:
            raise ValueError(
                f"extra_regressors_per_run has {len(extra_regressors_per_run)} "
                f"runs but data has {n_runs}."
            )
        n_extra = extra_regressors_per_run[0].shape[1] if extra_regressors_per_run[0].ndim > 1 else 1
        for r in range(1, n_runs):
            x = extra_regressors_per_run[r]
            n_e_r = x.shape[1] if x.ndim > 1 else 1
            if n_e_r != n_extra:
                raise ValueError(
                    f"extra_regressors_per_run[{r}] has {n_e_r} columns; "
                    f"run 0 has {n_extra}."
                )
    else:
        n_extra = 0

    if device is None:
        device = per_run_data[0].device if per_run_data[0].is_floating_point() else torch.device("cpu")

    # --- Row-concat task across runs (this is the "shared task" trick) -----
    task_concat = torch.cat(
        [d.to(device).to(torch.float32) for d in per_run_task_designs], dim=0
    )

    # --- Row-concat data --------------------------------------------------
    data_concat = torch.cat(
        [d.to(device).to(torch.float32) for d in per_run_data], dim=1
    )

    # --- Build block-diagonal nuisance (poly + optional extras) per run ---
    n_tp_per_run = [d.shape[1] for d in per_run_data]
    total_tp = sum(n_tp_per_run)
    n_nuisance_per_run = (polort + 1 if polort >= 0 else 0) + n_extra
    if n_nuisance_per_run > 0:
        nuisance_full = torch.zeros(
            (total_tp, n_runs * n_nuisance_per_run), dtype=torch.float32, device=device,
        )
        tr_start = 0
        col_start = 0
        for r, n_tp in enumerate(n_tp_per_run):
            run_blocks: list[torch.Tensor] = []
            if polort >= 0:
                poly_np = legendre_polynomials(n_tp, polort)
                run_blocks.append(torch.from_numpy(poly_np).to(torch.float32).to(device))
            if extra_regressors_per_run is not None:
                x = extra_regressors_per_run[r].to(device).to(torch.float32)
                if x.ndim == 1:
                    x = x.unsqueeze(1)
                run_blocks.append(x)
            run_block = torch.cat(run_blocks, dim=1)
            nuisance_full[tr_start:tr_start + n_tp, col_start:col_start + run_block.shape[1]] = run_block
            tr_start += n_tp
            col_start += run_block.shape[1]
        design_concat = torch.cat([task_concat, nuisance_full], dim=1)
    else:
        design_concat = task_concat

    # --- Column labels ----------------------------------------------------
    if task_column_labels is None:
        labels = [f"task{i}" for i in range(n_task)]
    else:
        if len(task_column_labels) != n_task:
            raise ValueError(
                f"task_column_labels has {len(task_column_labels)} entries; "
                f"per_run_task_designs has {n_task} task columns."
            )
        labels = list(task_column_labels)
    for r in range(n_runs):
        if polort >= 0:
            for k in range(polort + 1):
                labels.append(f"run{r + 1}_poly{k}")
        for k in range(n_extra):
            labels.append(f"run{r + 1}_extra{k}")

    return PackedSharedTaskDesign(
        data_concat=data_concat,
        design_concat=design_concat,
        n_task_cols=n_task,
        column_labels=labels,
        n_tp_per_run=n_tp_per_run,
        polort=polort,
    )


def _build_assumed_designs(
    onsets_per_cond_per_run: list[list[np.ndarray]],
    n_timepoints_per_run: list[int],
    tr: float,
    *,
    condition_labels: list[str],
    hrf: torch.Tensor | np.ndarray,
    device: torch.device,
    notes: list[str],
) -> TaskDesignResult:
    """Assumed-HRF design: one regressor per condition = onsets ⊛ HRF.

    Implements the trivial "convolve onset train with HRF kernel per
    condition per run" path so that :func:`build_per_run_task_designs`
    has a single uniform interface across basis modes.  The HRF is
    sampled at TR resolution; sub-TR-onset support is currently
    handled by the caller upstream (e.g. via microtime upsampling).
    """
    hrf_t = torch.as_tensor(hrf, dtype=torch.float32, device=device).flatten()
    n_conditions = len(onsets_per_cond_per_run)
    per_run: list[torch.Tensor] = []
    for r, n_run_tp in enumerate(n_timepoints_per_run):
        cond_cols: list[torch.Tensor] = []
        for c in range(n_conditions):
            onset_times = np.asarray(
                onsets_per_cond_per_run[c][r], dtype=np.float64
            )
            onset_vec = torch.zeros(n_run_tp, 1, device=device)
            if onset_times.size > 0:
                idx = np.round(onset_times / tr).astype(int)
                idx = idx[(idx >= 0) & (idx < n_run_tp)]
                onset_vec[idx, 0] = 1.0
            conv = convolve_hrf(onset_vec, hrf_t, n_run_tp, device=device)
            cond_cols.append(conv)
        per_run.append(torch.cat(cond_cols, dim=1))

    return TaskDesignResult(
        per_run=per_run,
        column_labels=list(condition_labels),
        n_basis_per_condition=[1] * n_conditions,
        basis_resolved="assumed",
        fir_window_s=None,
        lag_times_s=None,
        notes=notes,
    )
