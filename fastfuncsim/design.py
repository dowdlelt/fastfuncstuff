"""
Design matrix construction for GLM
Handles FIR, assumed HRF, and convolution operations
"""

import torch
import torch.nn.functional as F
from typing import Union, List, Optional
from .utils import to_tensor, get_device


def convolve_hrf(onsets: torch.Tensor, hrf: torch.Tensor,
                 n_timepoints: int,
                 device: Optional[torch.device] = None) -> torch.Tensor:
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

    # Normalize HRF to unit peak
    hrf = hrf / hrf.max()

    # Convolve each condition with HRF
    design = torch.zeros(n_timepoints, n_conditions, device=device)

    for cond_idx in range(n_conditions):
        # Use torch.nn.functional.conv1d for GPU acceleration
        # Need to reshape for conv1d: (batch, channels, length)
        onset_vec = onsets[:, cond_idx].unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        hrf_kernel = hrf.flip(0).unsqueeze(0).unsqueeze(0)  # (1, 1, H)

        # Convolve with 'same' padding
        convolved = F.conv1d(onset_vec, hrf_kernel, padding=len(hrf) // 2)
        design[:, cond_idx] = convolved.squeeze()[:n_timepoints]

    return design


def make_fir_design(onsets: torch.Tensor, n_lags: int,
                    n_timepoints: int,
                    device: Optional[torch.device] = None) -> torch.Tensor:
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
                shifted[lag:] = onsets[:n_timepoints-lag, cond_idx]
                design[:, col_idx] = shifted

    return design


def make_singletrialdesign(onsets: torch.Tensor,
                           device: Optional[torch.device] = None) -> torch.Tensor:
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


def convolve_design_hrf(design: torch.Tensor, hrf: torch.Tensor,
                       device: Optional[torch.device] = None) -> torch.Tensor:
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


def build_glm_design(onsets: Union[torch.Tensor, List[torch.Tensor]],
                    hrf: Optional[torch.Tensor] = None,
                    n_timepoints: Optional[Union[int, List[int]]] = None,
                    mode: str = 'assumed',
                    n_fir_lags: int = 30,
                    single_trial: bool = False,
                    device: Optional[torch.device] = None) -> Union[torch.Tensor, List[torch.Tensor]]:
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
        - 'fir': FIR design (no HRF assumption)
        - 'onoff': Simple boxcar (summed onsets)
    n_fir_lags : int
        Number of lags for FIR design (default: 30)
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
            if mode == 'assumed' and hrf is not None:
                design = convolve_design_hrf(design, hrf, device=device)
            elif mode == 'fir':
                # For FIR with single trial, need to expand each trial
                raise NotImplementedError("FIR single-trial design not yet implemented")
        else:
            if mode == 'onoff':
                # Simple boxcar - just sum across conditions
                design = onset.sum(dim=1, keepdim=True) if onset.ndim > 1 else onset.unsqueeze(1)
                if hrf is not None:
                    design = convolve_hrf(design, hrf, n_tp, device=device)

            elif mode == 'assumed':
                if hrf is None:
                    raise ValueError("HRF must be provided for 'assumed' mode")
                design = convolve_hrf(onset, hrf, n_tp, device=device)

            elif mode == 'fir':
                design = make_fir_design(onset, n_fir_lags, n_tp, device=device)

            else:
                raise ValueError(f"Unknown mode: {mode}")

        designs.append(design)

    if is_single_run:
        return designs[0]
    else:
        return designs


def generate_random_onsets(n_timepoints: int,
                           n_conditions: int,
                           isi_mean: float,
                           isi_range: tuple = (2, 8),
                           tr: float = 1.0,
                           alternate_conditions: bool = True,
                           device: Optional[torch.device] = None) -> torch.Tensor:
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
        conditions = [random.randint(0, n_conditions-1) for _ in range(n_trials_total)]

    # Build onset matrix
    onsets = torch.zeros(n_timepoints, n_conditions, device=device)
    for onset_time, condition in zip(onset_times, conditions):
        if onset_time < n_timepoints:
            onsets[onset_time, condition] = 1

    return onsets
