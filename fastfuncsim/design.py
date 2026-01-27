"""
Design matrix construction for GLM
Handles FIR, assumed HRF, and convolution operations
"""

from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from .utils import get_device, to_tensor


def convolve_hrf(
    onsets: torch.Tensor,
    hrf: torch.Tensor,
    n_timepoints: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Convolve onset vector with HRF using fast FFT convolution

    Parameters
    ----------
    onsets : torch.Tensor
        (n_timepoints, n_conditions) binary onset matrix (1=onset, 0=no onset)
    hrf : torch.Tensor
        (n_hrf_timepoints,) hemodynamic response function
    n_timepoints : int
        Total number of timepoints in output
    device : torch.device, optional
        Device for computation

    Returns
    -------
    design : torch.Tensor
        (n_timepoints, n_conditions) convolved design matrix
    """
    if device is None:
        device = get_device()

    onsets = to_tensor(onsets, device=device)
    hrf = to_tensor(hrf, device=device)

    n_conditions = onsets.shape[1] if onsets.ndim > 1 else 1
    if onsets.ndim == 1:
        onsets = onsets.unsqueeze(1)

    # Note: We do NOT normalize the HRF here. The HRF should already be
    # normalized at source resolution so that a single event peaks at 1.0.
    # Multiple events will sum to >1.0, which is correct (linear superposition).

    # Convolve each condition with HRF
    design = torch.zeros(n_timepoints, n_conditions, device=device)

    for cond_idx in range(n_conditions):
        # Use torch.nn.functional.conv1d for GPU acceleration
        # Need to reshape for conv1d: (batch, channels, length)
        onset_vec = onsets[:, cond_idx].unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        hrf_kernel = hrf.flip(0).unsqueeze(0).unsqueeze(0)  # (1, 1, H)

        # Causal convolution: response follows stimulus
        # padding=len(hrf)-1 ensures output[t] only depends on input[<=t]
        convolved = F.conv1d(onset_vec, hrf_kernel, padding=len(hrf) - 1)
        # Take first n_timepoints (trim the right side padding)
        design[:, cond_idx] = convolved.squeeze()[:n_timepoints]

    return design


def convolve_hrf_microtime(
    onsets_microtime: torch.Tensor,
    hrf: torch.Tensor,
    n_timepoints: int,
    tr: float = 1.0,
    microtime_dt: float = 0.1,
    microtime_onset: int = 0,
    device: Optional[torch.device] = None,
    return_single_trials: bool = False,
) -> Union[torch.Tensor, tuple]:
    """
    Convolve onset matrix with HRF using microtime (sub-TR) resolution.

    Uses single-trial convolution approach: each event is convolved separately,
    scaled to peak=1.0, then summed within condition. This guarantees:
    - Each event peaks at 1.0 regardless of stimulus duration
    - Overlapping events sum linearly (peak > 1.0 if they overlap)
    - Duration affects response *shape* (wider) but not *height*

    Parameters
    ----------
    onsets_microtime : torch.Tensor
        (n_microtime_points, n_conditions) onset matrix at microtime_dt resolution.
        Values can be binary (0/1) or boxcar (constant value during stimulus).
        Each contiguous non-zero region is treated as a separate event.
        n_microtime_points should equal n_timepoints * (tr / microtime_dt).
    hrf : torch.Tensor
        (n_hrf_microtime_timepoints,) hemodynamic response function at microtime_dt
        resolution. Should be pre-generated at the same microtime_dt as onsets.
    n_timepoints : int
        Number of TR timepoints in output
    tr : float, default=1.0
        Repetition time in seconds.
    microtime_dt : float, default=0.1
        Microtime resolution in seconds. Default 0.1s is the standard throughout
        the pipeline. Both onsets and HRF should be at this resolution.
    microtime_onset : int, default=0
        Which microtime bin within each TR to sample (0-indexed).
        0 = start of TR, bins_per_tr/2 = middle of TR.
    device : torch.device, optional
        Device for computation
    return_single_trials : bool, default=False
        If True, also return the single-trial design matrices (useful for LSS/LSA).

    Returns
    -------
    design : torch.Tensor
        (n_timepoints, n_conditions) convolved design matrix at TR resolution
    single_trial_designs : list of torch.Tensor, optional
        If return_single_trials=True, returns list of (n_timepoints, n_trials)
        tensors, one per condition, with each column being a single trial regressor.

    Notes
    -----
    Algorithm:
    1. For each condition:
       a. Identify individual events (contiguous non-zero regions)
       b. Convolve each event separately with HRF (both at microtime_dt)
       c. Scale each convolved event to peak=1.0
       d. Sum all scaled events to get condition regressor
    2. Downsample to TR resolution by sampling every tr/microtime_dt bins
    """
    if device is None:
        device = get_device()

    onsets_microtime = to_tensor(onsets_microtime, device=device)
    hrf = to_tensor(hrf, device=device)

    n_microtime_points = onsets_microtime.shape[0]
    n_conditions = onsets_microtime.shape[1] if onsets_microtime.ndim > 1 else 1
    if onsets_microtime.ndim == 1:
        onsets_microtime = onsets_microtime.unsqueeze(1)

    # Calculate bins per TR
    bins_per_tr = int(round(tr / microtime_dt))

    # Validate dimensions
    expected_microtime = n_timepoints * bins_per_tr
    if n_microtime_points != expected_microtime:
        raise ValueError(
            f"onsets_microtime has {n_microtime_points} points, "
            f"expected {expected_microtime} (n_timepoints={n_timepoints} * "
            f"bins_per_tr={bins_per_tr} where tr={tr}, microtime_dt={microtime_dt})"
        )

    # HRF is already at microtime_dt resolution - use directly
    hrf_microtime = hrf

    # Prepare HRF kernel for conv1d (flipped for causal convolution)
    hrf_kernel = hrf_microtime.flip(0).unsqueeze(0).unsqueeze(0)  # (1, 1, H)
    hrf_len = len(hrf_microtime)

    # Downsampling indices: sample every bins_per_tr, starting at microtime_onset
    sample_indices = torch.arange(microtime_onset, n_microtime_points, bins_per_tr, device=device)
    if len(sample_indices) > n_timepoints:
        sample_indices = sample_indices[:n_timepoints]
    elif len(sample_indices) < n_timepoints:
        raise ValueError(
            f"Downsampling produced {len(sample_indices)} samples, expected {n_timepoints}"
        )

    # 2. Process each condition using single-trial approach
    design = torch.zeros(n_timepoints, n_conditions, device=device)
    single_trial_designs = [] if return_single_trials else None

    for cond_idx in range(n_conditions):
        cond_onsets = onsets_microtime[:, cond_idx]

        # Find individual events (contiguous non-zero regions)
        # An event starts when we go from 0 to non-zero, ends when we go back to 0
        is_active = (cond_onsets != 0).float()

        # Detect event boundaries
        padded = torch.cat(
            [torch.zeros(1, device=device), is_active, torch.zeros(1, device=device)]
        )
        diff = padded[1:] - padded[:-1]
        event_starts = torch.where(diff == 1)[0]
        event_ends = torch.where(diff == -1)[0]

        n_events = len(event_starts)

        if n_events == 0:
            # No events for this condition
            if return_single_trials:
                single_trial_designs.append(torch.zeros(n_timepoints, 0, device=device))
            continue

        # Storage for single-trial regressors (at microtime resolution)
        trial_regressors_micro = torch.zeros(n_microtime_points, n_events, device=device)

        for event_idx, (start, end) in enumerate(zip(event_starts, event_ends)):
            # Create single-event onset vector
            single_event = torch.zeros(n_microtime_points, device=device)
            single_event[start:end] = cond_onsets[start:end]

            # Convolve with HRF
            event_vec = single_event.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
            convolved = F.conv1d(event_vec, hrf_kernel, padding=hrf_len - 1)
            convolved = convolved.squeeze()[:n_microtime_points]

            # Scale to peak = 1.0
            peak_val = convolved.abs().max()
            if peak_val > 0:
                convolved = convolved / peak_val

            trial_regressors_micro[:, event_idx] = convolved

        # Sum across trials to get condition regressor (at microtime resolution)
        cond_regressor_micro = trial_regressors_micro.sum(dim=1)

        # Downsample to TR resolution
        design[:, cond_idx] = cond_regressor_micro[sample_indices]

        if return_single_trials:
            # Downsample single-trial regressors too
            trial_regressors_tr = trial_regressors_micro[sample_indices, :]
            single_trial_designs.append(trial_regressors_tr)

    if return_single_trials:
        return design, single_trial_designs
    return design


def make_fir_design(
    onsets: torch.Tensor,
    n_lags: int,
    n_timepoints: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create FIR (Finite Impulse Response) design matrix

    For each condition, creates n_lags regressors representing the response
    at each lag after stimulus onset. This is the most flexible approach
    as it makes no assumptions about HRF shape.

    Parameters
    ----------
    onsets : torch.Tensor
        (n_timepoints, n_conditions) binary onset matrix
    n_lags : int
        Number of time lags to model (e.g., 30 TRs for ~30s response)
    n_timepoints : int
        Total number of timepoints
    device : torch.device, optional
        Device for computation

    Returns
    -------
    design : torch.Tensor
        (n_timepoints, n_conditions * n_lags) FIR design matrix
    """
    if device is None:
        device = get_device()

    onsets = to_tensor(onsets, device=device)

    n_conditions = onsets.shape[1] if onsets.ndim > 1 else 1
    if onsets.ndim == 1:
        onsets = onsets.unsqueeze(1)

    # Pre-allocate design matrix
    design = torch.zeros(n_timepoints, n_conditions * n_lags, device=device)

    # For each condition and lag, shift onset vector
    for cond_idx in range(n_conditions):
        for lag in range(n_lags):
            col_idx = cond_idx * n_lags + lag

            if lag == 0:
                design[:, col_idx] = onsets[:, cond_idx]
            else:
                # Shift onset vector by lag
                shifted = torch.zeros(n_timepoints, device=device)
                shifted[lag:] = onsets[: n_timepoints - lag, cond_idx]
                design[:, col_idx] = shifted

    return design


def basis_tent(x: torch.Tensor, bot: float, mid: float, top: float) -> torch.Tensor:
    """
    Tent basis function: piecewise linear interpolation

    Returns a tent function that:
    - equals 0 for x outside [bot, top]
    - equals 1 at x = mid
    - linear interpolation between bot-mid and mid-top

    Parameters
    ----------
    x : torch.Tensor
        Time points to evaluate (in seconds or TRs)
    bot : float
        Bottom/start of tent (left edge)
    mid : float
        Peak of tent (value = 1.0)
    top : float
        Top/end of tent (right edge)

    Returns
    -------
    values : torch.Tensor
        Tent function values at each x

    Notes
    -----
    This replicates AFNI's basis_tent function from 3dDeconvolve.c
    """
    val = torch.zeros_like(x)

    # Left ramp: bot to mid
    left_mask = (x > bot) & (x <= mid)
    val[left_mask] = (x[left_mask] - bot) / (mid - bot)

    # Right ramp: mid to top
    right_mask = (x > mid) & (x < top)
    val[right_mask] = (top - x[right_mask]) / (top - mid)

    return val


def make_tent_design(
    onsets: torch.Tensor,
    bot: float,
    top: float,
    tr: float,
    n_timepoints: int,
    n_basis: Optional[int] = None,
    zero_edges: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create TENT basis function design matrix for non-TR-locked onsets

    TENT uses piecewise linear splines (tent functions) to model the HRF.
    Unlike FIR which assumes TR-locked onsets, TENT can handle arbitrary
    onset times by creating tent basis functions on a microtime grid.

    Parameters
    ----------
    onsets : torch.Tensor
        (n_timepoints, n_conditions) onset matrix (can be fractional for sub-TR onsets)
        OR list of onset times in seconds for each condition
    bot : float
        Start time of HRF window (seconds after stimulus)
    top : float
        End time of HRF window (seconds after stimulus)
    tr : float
        Repetition time in seconds
    n_timepoints : int
        Total number of timepoints (TRs)
    n_basis : int, optional
        Number of tent basis functions (knots)
        If None, automatically calculated to give one knot per TR:
            n_basis = round((top - bot) / tr) + 1
        For TENT: n_basis determines resolution
        For TENTzero: actual basis functions = n_basis - 2
    zero_edges : bool
        If True, use TENTzero (force HRF to start and end at zero)
        If False, use TENT (standard tent functions)
    device : torch.device, optional
        Device for computation

    Returns
    -------
    design : torch.Tensor
        (n_timepoints, n_conditions * n_actual_basis) TENT design matrix
        where n_actual_basis = n_basis for TENT, n_basis-2 for TENTzero

    Notes
    -----
    TENT basis functions are 'cardinal interpolation' functions - their
    parameters are the HRF values at the knot points:
        bot, bot+dt, bot+2*dt, ..., top
    where dt = (top - bot) / (n_basis - 1)

    TENTzero eliminates the first and last basis functions to force the
    HRF to be zero at t=bot and t=top (continuous start/end at zero).

    This replicates AFNI's TENT and TENTzero basis functions.

    Examples
    --------
    Auto-calculate n_basis for TR spacing:
    >>> design = make_tent_design(onsets, bot=0, top=20, tr=1.0, n_timepoints=300)
    >>> # Creates 21 knots at 0, 1, 2, ..., 20 seconds

    Explicit n_basis:
    >>> design = make_tent_design(onsets, bot=0, top=15, tr=1.0, n_basis=6, n_timepoints=300)
    >>> # Creates 6 knots with dt = 15/5 = 3 seconds spacing

    References
    ----------
    AFNI 3dDeconvolve documentation
    https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dDeconvolve.html
    """
    if device is None:
        device = get_device()

    onsets = to_tensor(onsets, device=device)

    if bot >= top:
        raise ValueError(f"bot ({bot}) must be < top ({top})")

    # Auto-calculate n_basis if not provided
    if n_basis is None:
        # Default: one knot per TR
        n_basis = round((top - bot) / tr) + 1

    if n_basis < 3 if zero_edges else n_basis < 2:
        raise ValueError(f"n_basis must be >= {3 if zero_edges else 2}, got {n_basis}")

    n_conditions = onsets.shape[1] if onsets.ndim > 1 else 1
    if onsets.ndim == 1:
        onsets = onsets.unsqueeze(1)

    # Calculate knot spacing
    dt = (top - bot) / (n_basis - 1)

    # Determine actual number of basis functions
    if zero_edges:
        n_actual_basis = n_basis - 2
        first_basis_idx = 1  # Skip first tent
        last_basis_idx = n_basis - 1  # Skip last tent
    else:
        n_actual_basis = n_basis
        first_basis_idx = 0
        last_basis_idx = n_basis

    # Pre-allocate design matrix
    design = torch.zeros(
        n_timepoints, n_conditions * n_actual_basis, device=device
    )

    # Create time vector for each TR (in seconds)
    tr_times = torch.arange(n_timepoints, device=device) * tr

    # For each condition
    for cond_idx in range(n_conditions):
        # Get onset times for this condition (non-zero entries)
        onset_trs = torch.where(onsets[:, cond_idx] != 0)[0]

        if len(onset_trs) == 0:
            continue

        # Convert onset TRs to onset times in seconds
        onset_times = onset_trs.float() * tr

        # For each basis function
        basis_col_idx = 0
        for basis_idx in range(first_basis_idx, last_basis_idx):
            # Knot position
            knot_time = bot + basis_idx * dt

            # Tent edges
            if basis_idx == 0:
                # First tent: slightly before bot to bot+dt
                tent_bot = bot - 0.00111 * dt
                tent_mid = bot
                tent_top = bot + dt
            elif basis_idx == n_basis - 1:
                # Last tent: (top-dt) to slightly after top
                tent_bot = top - dt
                tent_mid = top
                tent_top = top + 0.00111 * dt
            else:
                # Middle tents: standard spacing
                tent_bot = bot + (basis_idx - 1) * dt
                tent_mid = bot + basis_idx * dt
                tent_top = bot + (basis_idx + 1) * dt

            # For each TR, sum contributions from all onsets
            for onset_t in onset_times:
                # Time relative to this onset
                rel_times = tr_times - onset_t

                # Evaluate tent basis at these relative times
                tent_vals = basis_tent(rel_times, tent_bot, tent_mid, tent_top)

                # Add to design matrix column
                col_idx = cond_idx * n_actual_basis + basis_col_idx
                design[:, col_idx] += tent_vals

            basis_col_idx += 1

    return design


def is_tr_locked(
    onset_times: Union[list, np.ndarray],
    tr: float,
    threshold: float = 0.1,
) -> bool:
    """
    Check if onset times are approximately TR-locked

    Determines if onsets align with TR boundaries within a tolerance.
    If onsets are TR-locked, standard FIR can be used efficiently.
    If not, TENT/TENTzero should be used for sub-TR accuracy.

    Parameters
    ----------
    onset_times : list or array
        Onset times in seconds
    tr : float
        Repetition time in seconds
    threshold : float
        Maximum remainder as fraction of TR (default: 0.1 = 10%)

    Returns
    -------
    is_locked : bool
        True if all onsets are within threshold of a TR boundary

    Examples
    --------
    >>> is_tr_locked([0, 2.0, 4.0], tr=2.0)  # Perfect TR locking
    True
    >>> is_tr_locked([0, 2.1, 4.0], tr=2.0, threshold=0.1)  # Within 10%
    True
    >>> is_tr_locked([0.5, 2.5, 4.5], tr=2.0)  # 0.5s offset (25% of 2s TR)
    False
    """
    if isinstance(onset_times, list):
        onset_times = np.array(onset_times)

    # Calculate remainder when dividing by TR
    remainders = np.mod(onset_times, tr)

    # Normalize remainders (could be close to 0 or close to TR)
    normalized_remainders = np.minimum(remainders, tr - remainders)

    # Check if all remainders are within threshold
    max_remainder_fraction = np.max(normalized_remainders) / tr

    return max_remainder_fraction < threshold


def make_singletrialdesign(
    onsets: torch.Tensor, device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Create single-trial design matrix (one regressor per trial)

    This creates a design where each trial gets its own regressor,
    allowing estimation of single-trial response amplitudes.

    Parameters
    ----------
    onsets : torch.Tensor
        (n_timepoints, n_conditions) binary onset matrix where each '1'
        represents a trial onset
    device : torch.device, optional
        Device for computation

    Returns
    -------
    design_single : torch.Tensor
        (n_timepoints, n_trials) single-trial design matrix
    trial_conditions : torch.Tensor
        (n_trials,) condition index for each trial
    """
    if device is None:
        device = get_device()

    onsets = to_tensor(onsets, device=device)

    if onsets.ndim == 1:
        onsets = onsets.unsqueeze(1)

    n_timepoints, n_conditions = onsets.shape

    # Find all trial onsets
    trial_times = []
    trial_conditions = []

    for cond_idx in range(n_conditions):
        cond_onsets = torch.where(onsets[:, cond_idx] > 0)[0]
        for onset_time in cond_onsets:
            trial_times.append(onset_time.item())
            trial_conditions.append(cond_idx)

    # Sort trials by time
    sort_idx = sorted(range(len(trial_times)), key=lambda i: trial_times[i])
    trial_times = [trial_times[i] for i in sort_idx]
    trial_conditions = [trial_conditions[i] for i in sort_idx]

    n_trials = len(trial_times)

    # Create design matrix
    design_single = torch.zeros(n_timepoints, n_trials, device=device)
    for trial_idx, onset_time in enumerate(trial_times):
        design_single[onset_time, trial_idx] = 1

    trial_conditions = torch.tensor(trial_conditions, device=device)

    return design_single, trial_conditions


def convolve_design_hrf(
    design: torch.Tensor, hrf: torch.Tensor, device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Convolve an existing design matrix with HRF

    This is useful when you have a design matrix (e.g., single-trial)
    and want to convolve all columns with an HRF.

    Parameters
    ----------
    design : torch.Tensor
        (n_timepoints, n_regressors) design matrix
    hrf : torch.Tensor
        (n_hrf_timepoints,) hemodynamic response function
    device : torch.device, optional
        Device for computation

    Returns
    -------
    convolved : torch.Tensor
        (n_timepoints, n_regressors) convolved design matrix
    """
    if device is None:
        device = get_device()

    design = to_tensor(design, device=device)
    hrf = to_tensor(hrf, device=device)

    n_timepoints, n_regressors = design.shape

    # Normalize HRF
    hrf = hrf / hrf.max()

    # Convolve each regressor
    convolved = torch.zeros_like(design)

    # Batch process with conv1d
    # Shape: (1, n_regressors, n_timepoints)
    design_batch = design.T.unsqueeze(0)
    hrf_kernel = hrf.flip(0).unsqueeze(0).unsqueeze(0)

    # Convolve
    conv_result = F.conv1d(design_batch, hrf_kernel, padding=len(hrf) // 2)
    convolved = conv_result.squeeze(0).T[:n_timepoints]

    return convolved


def build_glm_design(
    onsets: Union[torch.Tensor, list[torch.Tensor]],
    hrf: Optional[torch.Tensor] = None,
    n_timepoints: Optional[Union[int, list[int]]] = None,
    mode: str = "assumed",
    n_fir_lags: int = 30,
    tr: float = 1.0,
    tent_bot: float = 0.0,
    tent_top: float = 15.0,
    tent_n_basis: Optional[int] = None,
    single_trial: bool = False,
    device: Optional[torch.device] = None,
) -> Union[torch.Tensor, list[torch.Tensor]]:
    """
    Build design matrix for GLM fitting

    This is a high-level function that constructs design matrices for
    different modeling approaches.

    Parameters
    ----------
    onsets : torch.Tensor or list of torch.Tensor
        Onset vectors. Can be:
        - Single run: (n_timepoints, n_conditions)
        - Multiple runs: list of (n_timepoints, n_conditions)
    hrf : torch.Tensor, optional
        HRF to convolve with (required for 'assumed' mode)
    n_timepoints : int or list of int, optional
        Number of timepoints (inferred from onsets if not provided)
    mode : str
        Design mode:
        - 'assumed': Convolve with assumed HRF
        - 'fir': FIR design (no HRF assumption, TR-locked onsets)
        - 'tent': TENT basis (piecewise linear, for non-TR-locked onsets)
        - 'tentzero': TENTzero basis (forces HRF to start/end at zero)
        - 'onoff': Simple boxcar (summed onsets)
    n_fir_lags : int
        Number of lags for FIR design (default: 30)
    tr : float
        Repetition time in seconds (default: 1.0, required for tent/tentzero)
    tent_bot : float
        TENT start time in seconds after stimulus (default: 0.0)
    tent_top : float
        TENT end time in seconds after stimulus (default: 15.0)
    tent_n_basis : int, optional
        Number of TENT basis functions (default: None, auto-calculated for TR spacing)
    single_trial : bool
        If True, create single-trial design (one regressor per trial)
    device : torch.device, optional
        Device for computation

    Returns
    -------
    design : torch.Tensor or list of torch.Tensor
        Design matrix or list of design matrices (one per run)
    """
    if device is None:
        device = get_device()

    # Handle single vs multiple runs
    is_single_run = not isinstance(onsets, list)
    if is_single_run:
        onsets = [onsets]

    if n_timepoints is None:
        n_timepoints = [o.shape[0] for o in onsets]
    elif isinstance(n_timepoints, int):
        n_timepoints = [n_timepoints] * len(onsets)

    designs = []

    for run_idx, (onset, n_tp) in enumerate(zip(onsets, n_timepoints)):
        onset = to_tensor(onset, device=device)

        if single_trial:
            # Create single-trial design
            design, _ = make_singletrialdesign(onset, device=device)
            if mode == "assumed" and hrf is not None:
                design = convolve_design_hrf(design, hrf, device=device)
            elif mode == "fir":
                # For FIR with single trial, need to expand each trial
                raise NotImplementedError("FIR single-trial design not yet implemented")
        else:
            if mode == "onoff":
                # Simple boxcar - just sum across conditions
                design = onset.sum(dim=1, keepdim=True) if onset.ndim > 1 else onset.unsqueeze(1)
                if hrf is not None:
                    design = convolve_hrf(design, hrf, n_tp, device=device)

            elif mode == "assumed":
                if hrf is None:
                    raise ValueError("HRF must be provided for 'assumed' mode")
                design = convolve_hrf(onset, hrf, n_tp, device=device)

            elif mode == "fir":
                design = make_fir_design(onset, n_fir_lags, n_tp, device=device)

            elif mode == "tent":
                design = make_tent_design(
                    onset, tent_bot, tent_top, tr, n_tp,
                    n_basis=tent_n_basis, zero_edges=False, device=device
                )

            elif mode == "tentzero":
                design = make_tent_design(
                    onset, tent_bot, tent_top, tr, n_tp,
                    n_basis=tent_n_basis, zero_edges=True, device=device
                )

            else:
                raise ValueError(f"Unknown mode: {mode}. Valid modes: assumed, fir, tent, tentzero, onoff")

        designs.append(design)

    if is_single_run:
        return designs[0]
    else:
        return designs


def generate_random_onsets(
    n_timepoints: int,
    n_conditions: int,
    isi_mean: float,
    isi_range: tuple = (2, 8),
    tr: float = 1.0,
    alternate_conditions: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate random onset times with truncated Poisson ISI distribution

    This mimics the MATLAB code's approach to generating event timing.

    Parameters
    ----------
    n_timepoints : int
        Total number of timepoints
    n_conditions : int
        Number of conditions
    isi_mean : float
        Mean inter-stimulus interval in seconds
    isi_range : tuple
        (min, max) ISI in seconds
    tr : float
        TR in seconds
    alternate_conditions : bool
        If True, conditions alternate (A-B-A-B...)
        If False, conditions are randomized
    device : torch.device, optional
        Device for computation

    Returns
    -------
    onsets : torch.Tensor
        (n_timepoints, n_conditions) binary onset matrix
    """
    if device is None:
        device = get_device()

    # Convert to TRs
    isi_mean_tr = isi_mean / tr
    isi_min_tr = isi_range[0] / tr
    isi_max_tr = isi_range[1] / tr

    # Use truncated Poisson to generate ISIs (matching MATLAB code)
    # This is approximate - for exact match would need bisection search for lambda
    lambda_poisson = isi_mean_tr

    # Generate ISIs
    isis = []
    while len(isis) < 1000:  # Generate plenty
        isi = torch.poisson(torch.tensor([lambda_poisson]))[0].item()
        if isi_min_tr <= isi <= isi_max_tr:
            isis.append(int(isi))

    # Calculate number of trials that fit
    n_trials_total = 0
    cumulative_time = 0
    while cumulative_time < n_timepoints and n_trials_total < len(isis):
        cumulative_time += isis[n_trials_total]
        n_trials_total += 1

    # Round down to multiple of n_conditions
    n_trials_total = (n_trials_total // n_conditions) * n_conditions

    # Shuffle ISIs
    import random

    random.shuffle(isis)
    isis = isis[:n_trials_total]

    # Calculate onset times
    onset_times = [0]
    for isi in isis[:-1]:
        onset_times.append(onset_times[-1] + isi)

    # Assign conditions
    if alternate_conditions:
        conditions = [i % n_conditions for i in range(n_trials_total)]
    else:
        conditions = [random.randint(0, n_conditions - 1) for _ in range(n_trials_total)]

    # Build onset matrix
    onsets = torch.zeros(n_timepoints, n_conditions, device=device)
    for onset_time, condition in zip(onset_times, conditions):
        if onset_time < n_timepoints:
            onsets[onset_time, condition] = 1

    return onsets


def extract_hrf_estimates(
    betas: Union[torch.Tensor, np.ndarray],
    n_conditions: int,
    n_lags: int,
    brain_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    output_shape: Optional[tuple] = None,
) -> np.ndarray:
    """
    Extract HRF estimates from GLM betas for FIR/TENT models

    Reshapes flat beta coefficients into condition-specific HRF estimates.
    This creates the impulse response (iresp) volumes used in AFNI.

    Parameters
    ----------
    betas : torch.Tensor or np.ndarray
        Beta coefficients from GLM fit
        Shape: (n_voxels, n_conditions * n_lags) or (nx, ny, nz, n_conditions * n_lags)
    n_conditions : int
        Number of conditions/regressors
    n_lags : int
        Number of time lags (FIR) or basis functions (TENT)
    brain_mask : torch.Tensor or np.ndarray, optional
        3D brain mask to reshape from 1D voxel array to 3D volume
        Shape: (nx, ny, nz)
    output_shape : tuple, optional
        Alternative to brain_mask: explicit output shape (nx, ny, nz)
        If both are None, assumes betas are already 4D

    Returns
    -------
    iresp : np.ndarray
        Impulse response estimates
        Shape: (nx, ny, nz, n_conditions, n_lags)
        - First 3 dims: brain space
        - 4th dim: conditions
        - 5th dim: time lags (HRF time course)

    Examples
    --------
    >>> # After GLM fit with FIR design
    >>> results = fit_glm(data, fir_design, tr=1.0)
    >>> iresp = extract_hrf_estimates(results.betas, n_conditions=2, n_lags=30, brain_mask=mask)
    >>> # iresp.shape = (64, 64, 30, 2, 30)  # spatial + conditions + time
    """
    # Convert to numpy if needed
    if isinstance(betas, torch.Tensor):
        betas = betas.cpu().numpy()

    # Handle 1D voxel array vs 4D volume
    if betas.ndim == 2:
        # Shape: (n_voxels, n_conditions * n_lags)
        n_voxels = betas.shape[0]

        # Need mask or output_shape to reconstruct 3D
        if brain_mask is not None:
            if isinstance(brain_mask, torch.Tensor):
                brain_mask = brain_mask.cpu().numpy()
            output_shape = brain_mask.shape
            mask_flat = brain_mask.ravel() > 0
        elif output_shape is not None:
            n_voxels_expected = np.prod(output_shape)
            if n_voxels != n_voxels_expected:
                raise ValueError(
                    f"output_shape {output_shape} implies {n_voxels_expected} voxels, "
                    f"but betas has {n_voxels} voxels"
                )
            mask_flat = None
        else:
            raise ValueError("Either brain_mask or output_shape must be provided for 2D betas")

        # Reshape to (nx, ny, nz, n_conditions * n_lags)
        betas_4d = np.zeros((*output_shape, n_conditions * n_lags))
        if mask_flat is not None and brain_mask is not None:
            betas_4d[brain_mask > 0, :] = betas
        else:
            betas_4d = betas.reshape(*output_shape, n_conditions * n_lags)

    elif betas.ndim == 4:
        # Already 4D: (nx, ny, nz, n_conditions * n_lags)
        betas_4d = betas
        output_shape = betas.shape[:3]
    else:
        raise ValueError(f"betas must be 2D or 4D, got shape {betas.shape}")

    # Reshape to (nx, ny, nz, n_conditions, n_lags)
    nx, ny, nz = output_shape
    iresp = betas_4d.reshape(nx, ny, nz, n_conditions, n_lags)

    return iresp


def save_iresp(
    iresp: np.ndarray,
    output_prefix: str,
    condition_labels: Optional[list[str]] = None,
    tr: float = 1.0,
    bot: float = 0.0,
    top: Optional[float] = None,
    affine: Optional[np.ndarray] = None,
    reference_img: Optional[str] = None,
):
    """
    Save HRF estimates as 4D NIfTI files (AFNI-style iresp)

    Creates one 4D file per condition showing the estimated HRF time course.

    Parameters
    ----------
    iresp : np.ndarray
        Impulse response estimates from extract_hrf_estimates()
        Shape: (nx, ny, nz, n_conditions, n_lags)
    output_prefix : str
        Output file prefix (e.g., 'output_dir/GLM')
        Files will be named: {prefix}_iresp_{label}.nii.gz
    condition_labels : list of str, optional
        Labels for each condition (default: ['cond1', 'cond2', ...])
    tr : float
        Repetition time in seconds (for metadata)
    bot : float
        HRF window start time in seconds (for metadata)
    top : float, optional
        HRF window end time in seconds (for metadata)
        If None, calculated from n_lags and tr
    affine : np.ndarray, optional
        4x4 affine transformation matrix
        If None, uses identity (or from reference_img)
    reference_img : str, optional
        Path to reference NIfTI to copy affine/header from

    Returns
    -------
    output_files : list of str
        Paths to created NIfTI files

    Examples
    --------
    >>> iresp = extract_hrf_estimates(betas, n_conditions=2, n_lags=20, brain_mask=mask)
    >>> files = save_iresp(
    ...     iresp,
    ...     output_prefix='results/GLM',
    ...     condition_labels=['faces', 'scenes'],
    ...     tr=2.0,
    ...     bot=0,
    ...     top=30,
    ...     reference_img='data/func.nii.gz'
    ... )
    >>> # Creates: results/GLM_iresp_faces.nii.gz, results/GLM_iresp_scenes.nii.gz
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel is required to save NIfTI files. Install with: pip install nibabel")

    if iresp.ndim != 5:
        raise ValueError(f"iresp must be 5D (nx, ny, nz, n_conditions, n_lags), got shape {iresp.shape}")

    nx, ny, nz, n_conditions, n_lags = iresp.shape

    # Get affine matrix
    if reference_img is not None:
        ref_img = nib.load(reference_img)
        affine = ref_img.affine
    elif affine is None:
        # Default identity affine
        affine = np.eye(4)

    # Calculate top if not provided
    if top is None:
        top = bot + (n_lags - 1) * tr

    # Default condition labels
    if condition_labels is None:
        condition_labels = [f"cond{i+1}" for i in range(n_conditions)]

    if len(condition_labels) != n_conditions:
        raise ValueError(
            f"Number of condition_labels ({len(condition_labels)}) must match "
            f"n_conditions ({n_conditions})"
        )

    # Save one file per condition
    output_files = []
    for cond_idx, label in enumerate(condition_labels):
        # Extract this condition's HRF: (nx, ny, nz, n_lags)
        cond_hrf = iresp[:, :, :, cond_idx, :]

        # Create NIfTI image with TR in header
        img = nib.Nifti1Image(cond_hrf, affine)
        img.header.set_xyzt_units(xyz='mm', t='sec')
        img.header['pixdim'][4] = tr  # Set TR

        # Add description
        description = f"HRF estimate: {label} (bot={bot}s, top={top}s, n={n_lags})"
        img.header['descrip'] = description[:80]  # Max 80 chars

        # Save file
        output_file = f"{output_prefix}_iresp_{label}.nii.gz"
        nib.save(img, output_file)
        output_files.append(output_file)

    return output_files
