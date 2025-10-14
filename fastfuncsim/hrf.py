"""
HRF (Hemodynamic Response Function) generation
Supports canonical HRFs, FLOBS, and custom HRF libraries
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict
from scipy import stats
from .utils import to_tensor, get_device


def get_canonical_hrf(stim_duration: float,
                     tr: float,
                     duration: float = 32.0,
                     device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Generate canonical double-gamma HRF (SPM-style)

    Parameters
    ----------
    stim_duration : float
        Stimulus duration in seconds
    tr : float
        TR in seconds
    duration : float
        Total HRF duration in seconds (default: 32s)
    device : torch.device, optional
        Device for tensor

    Returns
    -------
    hrf : torch.Tensor
        HRF sampled at TR intervals, normalized to peak=1
    """
    if device is None:
        device = get_device()

    # Time vector
    dt = 0.1  # High-res sampling
    t_highres = np.arange(0, duration, dt)

    # SPM canonical HRF parameters (from Glover 1999 / Friston 1998)
    # Response function (peak at ~5s)
    a1 = 6.0  # Time to peak
    b1 = 1.0  # Dispersion

    # Undershoot function (peak at ~15s)
    a2 = 16.0
    b2 = 1.0
    c = 1/6.0  # Undershoot ratio

    # Double gamma HRF using gamma PDF directly
    # First gamma (response): peak at ~5s
    hrf = stats.gamma.pdf(t_highres, a1, scale=b1)

    # Second gamma (undershoot): peak at ~15s, negative
    hrf -= c * stats.gamma.pdf(t_highres, a2, scale=b2)

    # Convolve with stimulus duration
    if stim_duration > 0:
        boxcar = np.zeros_like(t_highres)
        stim_samples = int(stim_duration / dt)
        boxcar[:stim_samples] = 1
        hrf = np.convolve(hrf, boxcar, mode='full')[:len(t_highres)]

    # Normalize to peak
    hrf = hrf / np.max(hrf)

    # Downsample to TR
    sample_indices = np.arange(0, len(t_highres), int(tr / dt))
    hrf_downsampled = hrf[sample_indices]

    return to_tensor(hrf_downsampled, device=device, dtype=torch.float32)


def get_canonical_hrf_library(stim_duration: float,
                              tr: float,
                              n_hrfs: int = 20,
                              device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Generate library of canonical HRFs with parameter variations

    This creates a library similar to GLMsingle's default library,
    with variations in timing parameters.

    Parameters
    ----------
    stim_duration : float
        Stimulus duration in seconds
    tr : float
        TR in seconds
    n_hrfs : int
        Number of HRFs in library (default: 20)
    device : torch.device, optional
        Device for tensors

    Returns
    -------
    hrf_library : torch.Tensor
        (n_hrfs, n_timepoints) library of HRFs
    """
    if device is None:
        device = get_device()

    duration = 32.0
    dt = 0.1
    t_highres = np.arange(0, duration, dt)

    # Parameter ranges (based on Natural Scenes Dataset / GLMsingle)
    # These create HRFs with peaks ranging from ~4-7s and varying undershoots
    param_variations = []

    for i in range(n_hrfs):
        # Vary time-to-peak and undershoot parameters
        a1 = 5.0 + (i / n_hrfs) * 3.0  # Time to peak: 5-8
        a2 = 15.0 + (i / n_hrfs) * 5.0  # Time to undershoot: 15-20
        c = 0.15 + (i / n_hrfs) * 0.15  # Undershoot ratio: 0.15-0.30

        param_variations.append((a1, a2, c))

    # Generate HRFs
    hrfs = []
    for a1, a2, c in param_variations:
        b1, b2 = 1.0, 1.0

        # Double gamma HRF using gamma PDF directly
        hrf = stats.gamma.pdf(t_highres, a1, scale=b1)
        hrf -= c * stats.gamma.pdf(t_highres, a2, scale=b2)

        # Convolve with stimulus
        if stim_duration > 0:
            boxcar = np.zeros_like(t_highres)
            stim_samples = int(stim_duration / dt)
            boxcar[:stim_samples] = 1
            hrf = np.convolve(hrf, boxcar, mode='full')[:len(t_highres)]

        # Normalize
        hrf = hrf / np.max(hrf)

        # Downsample
        sample_indices = np.arange(0, len(t_highres), int(tr / dt))
        hrf_downsampled = hrf[sample_indices]

        hrfs.append(hrf_downsampled)

    hrf_library = np.stack(hrfs, axis=0)

    return to_tensor(hrf_library, device=device, dtype=torch.float32)


def flobs_halfcos(m1: float, m2: float, m3: float, m4: float,
                  c2: float,
                  duration: float = 32.0,
                  sample_rate: float = 0.05,
                  device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Generate HRF using half-cosine basis (FLOBS method)

    This implements the flexible basis approach from FSL's FLOBS.

    Parameters
    ----------
    m1 : float
        Delay before rise starts (seconds)
    m2 : float
        Time from start to peak (seconds)
    m3 : float
        Time from peak to undershoot (seconds)
    m4 : float
        Time from undershoot back to baseline (seconds)
    c2 : float
        Undershoot magnitude (negative value)
    duration : float
        Total HRF duration in seconds
    sample_rate : float
        Sampling rate for generation (seconds)
    device : torch.device, optional
        Device for tensor

    Returns
    -------
    hrf : torch.Tensor
        Generated HRF
    """
    if device is None:
        device = get_device()

    t = np.arange(0, duration, sample_rate)
    hrf = np.zeros_like(t)

    # Segment 1: Flat at 0 until m1
    # (already zeros)

    # Segment 2: Rise to peak (half cosine, inverted)
    mask2 = (t > m1) & (t <= m1 + m2)
    if np.any(mask2):
        t_rel = t[mask2] - m1
        hrf[mask2] = 0.5 * (-1) * np.cos(2 * np.pi * t_rel / (m2 * 2)) + 0.5

    # Segment 3: Fall to undershoot (half cosine)
    mask3 = (t > m1 + m2) & (t <= m1 + m2 + m3)
    if np.any(mask3):
        t_rel = t[mask3] - (m1 + m2)
        hrf[mask3] = (1 + c2) / 2 * np.cos(2 * np.pi * t_rel / (m3 * 2)) + (1 - c2) / 2

    # Segment 4: Recovery from undershoot (half cosine, inverted)
    mask4 = (t > m1 + m2 + m3) & (t <= m1 + m2 + m3 + m4)
    if np.any(mask4):
        t_rel = t[mask4] - (m1 + m2 + m3)
        hrf[mask4] = (-c2) / 2 * (-1) * np.cos(2 * np.pi * t_rel / (m4 * 2)) + (-c2) / 2

    # Normalize
    if np.max(np.abs(hrf)) > 0:
        hrf = hrf / np.max(hrf)

    return to_tensor(hrf, device=device, dtype=torch.float32)


def create_flobs_library(n_hrfs: int = 20,
                        m1_range: Tuple[float, float] = (0, 2),
                        m2_range: Tuple[float, float] = (3, 8),
                        m3_range: Tuple[float, float] = (3, 10),
                        m4_range: Tuple[float, float] = (3, 12),
                        c2_range: Tuple[float, float] = (0, 0.35),
                        duration: float = 32.0,
                        sample_rate: float = 0.05,
                        tr: float = 1.0,
                        stim_duration: float = 0.0,
                        device: Optional[torch.device] = None) -> Tuple[torch.Tensor, Dict]:
    """
    Create HRF library using FLOBS half-cosine method with parameter sampling

    Parameters
    ----------
    n_hrfs : int
        Number of HRFs to generate
    m1_range, m2_range, m3_range, m4_range : tuple
        (min, max) ranges for timing parameters
    c2_range : tuple
        (min, max) range for undershoot magnitude
    duration : float
        HRF duration in seconds
    sample_rate : float
        High-resolution sampling rate
    tr : float
        TR for downsampling
    stim_duration : float
        Stimulus duration to convolve with
    device : torch.device, optional
        Device for tensors

    Returns
    -------
    hrf_library : torch.Tensor
        (n_hrfs, n_timepoints) library of HRFs at TR resolution
    params : dict
        Dictionary of parameters used for each HRF
    """
    if device is None:
        device = get_device()

    # Latin Hypercube Sampling for parameter values
    from scipy.stats import qmc

    sampler = qmc.LatinHypercube(d=5)
    samples = sampler.random(n=n_hrfs)

    # Scale to parameter ranges
    m1_vals = m1_range[0] + samples[:, 0] * (m1_range[1] - m1_range[0])
    m2_vals = m2_range[0] + samples[:, 1] * (m2_range[1] - m2_range[0])
    m3_vals = m3_range[0] + samples[:, 2] * (m3_range[1] - m3_range[0])
    m4_vals = m4_range[0] + samples[:, 3] * (m4_range[1] - m4_range[0])
    c2_vals = c2_range[0] + samples[:, 4] * (c2_range[1] - c2_range[0])

    # Generate HRFs at high resolution
    t_highres = np.arange(0, duration, sample_rate)
    hrfs_highres = []

    for i in range(n_hrfs):
        hrf = flobs_halfcos(m1_vals[i], m2_vals[i], m3_vals[i], m4_vals[i],
                           c2_vals[i], duration, sample_rate, device='cpu')

        hrfs_highres.append(hrf.numpy())

    hrfs_highres = np.stack(hrfs_highres, axis=0)

    # Convolve with stimulus if needed
    if stim_duration > 0:
        boxcar = np.zeros(int(30 / sample_rate))  # 30s window
        boxcar[:int(stim_duration / sample_rate)] = 1

        hrfs_convolved = np.zeros_like(hrfs_highres)
        for i in range(n_hrfs):
            convolved = np.convolve(hrfs_highres[i], boxcar, mode='full')
            hrfs_convolved[i] = convolved[:len(t_highres)]
            # Renormalize
            if np.max(hrfs_convolved[i]) > 0:
                hrfs_convolved[i] = hrfs_convolved[i] / np.max(hrfs_convolved[i])

        hrfs_highres = hrfs_convolved

    # Downsample to TR
    downsample_factor = int(tr / sample_rate)
    hrfs_tr = hrfs_highres[:, ::downsample_factor]

    # Truncate to reasonable length (e.g., 60 TRs)
    max_len = min(int(60 * tr / tr), hrfs_tr.shape[1])
    hrfs_tr = hrfs_tr[:, :max_len]

    hrf_library = to_tensor(hrfs_tr, device=device, dtype=torch.float32)

    params = {
        'm1': m1_vals,
        'm2': m2_vals,
        'm3': m3_vals,
        'm4': m4_vals,
        'c2': c2_vals
    }

    return hrf_library, params


def get_hrf_library(mode: str = 'canonical',
                   stim_duration: float = 5.0,
                   tr: float = 1.0,
                   n_hrfs: int = 20,
                   device: Optional[torch.device] = None,
                   **kwargs) -> torch.Tensor:
    """
    Convenience function to get HRF library

    Parameters
    ----------
    mode : str
        'canonical' - Double-gamma HRF variations
        'flobs' - FLOBS half-cosine HRFs
        'single' - Single canonical HRF (for assumed HRF approach)
    stim_duration : float
        Stimulus duration in seconds
    tr : float
        TR in seconds
    n_hrfs : int
        Number of HRFs (ignored for 'single' mode)
    device : torch.device, optional
        Device for tensors
    **kwargs : dict
        Additional arguments passed to specific generators

    Returns
    -------
    hrf_library : torch.Tensor
        (n_hrfs, n_timepoints) or (n_timepoints,) for single mode
    """
    if device is None:
        device = get_device()

    if mode == 'single':
        return get_canonical_hrf(stim_duration, tr, device=device)

    elif mode == 'canonical':
        return get_canonical_hrf_library(stim_duration, tr, n_hrfs, device=device)

    elif mode == 'flobs':
        library, _ = create_flobs_library(n_hrfs=n_hrfs,
                                         tr=tr,
                                         stim_duration=stim_duration,
                                         device=device,
                                         **kwargs)
        return library

    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'canonical', 'flobs', or 'single'")
