"""
HRF (Hemodynamic Response Function) generation
Supports canonical HRFs, PIGHS (Parametric Individually Generated HRFs), and custom HRF libraries

The canonical HRF library is loaded from pre-computed TSV files:
- getcanonicalbasic.tsv: Single canonical HRF (SPM-style double-gamma)
- getcanonicalhrflibrary.tsv: 20 HRF variants with different timing parameters

Both files are at 0.1s temporal resolution and must be:
1. Normalized to peak amplitude = 1.0
2. Resampled to the target microtime/TR grid
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
from scipy import stats
from scipy.interpolate import interp1d
from .utils import to_tensor, get_device

# Constants for pre-computed HRF files
_HRF_FILE_RESOLUTION = 0.1  # seconds per sample in TSV files
_HRF_LIBRARY_FILE = "getcanonicalhrflibrary.tsv"
_HRF_BASIC_FILE = "getcanonicalbasic.tsv"


def _get_hrf_file_path(filename: str) -> Path:
    """Get path to HRF data file in the package directory."""
    return Path(__file__).parent / filename


def _load_hrf_from_file(filename: str) -> np.ndarray:
    """
    Load HRF data from TSV file.

    Parameters
    ----------
    filename : str
        Name of the TSV file (in the package directory)

    Returns
    -------
    hrf_data : np.ndarray
        Raw HRF data at 0.1s resolution
        Shape: (n_timepoints,) for basic, (n_timepoints, n_hrfs) for library
    """
    filepath = _get_hrf_file_path(filename)
    if not filepath.exists():
        raise FileNotFoundError(f"HRF file not found: {filepath}")
    return np.loadtxt(filepath)


def _resample_hrf(
    hrf: np.ndarray,
    source_dt: float,
    target_dt: float,
    target_duration: Optional[float] = None,
) -> np.ndarray:
    """
    Resample HRF from source to target temporal resolution.

    Uses linear interpolation to resample the HRF to a new time grid.

    Parameters
    ----------
    hrf : np.ndarray
        HRF at source resolution, shape (n_source_timepoints,)
    source_dt : float
        Source temporal resolution in seconds (e.g., 0.1s for TSV files)
    target_dt : float
        Target temporal resolution in seconds (e.g., TR/16 for microtime)
    target_duration : float, optional
        Target duration in seconds. If None, uses source duration.

    Returns
    -------
    hrf_resampled : np.ndarray
        HRF at target resolution
    """
    n_source = len(hrf)
    source_duration = n_source * source_dt

    if target_duration is None:
        target_duration = source_duration

    # Create time vectors
    source_times = np.arange(n_source) * source_dt
    n_target = int(np.ceil(target_duration / target_dt))
    target_times = np.arange(n_target) * target_dt

    # Interpolate (clip to source duration to avoid extrapolation issues)
    target_times_clipped = np.clip(target_times, 0, source_times[-1])
    interpolator = interp1d(
        source_times, hrf, kind="linear", fill_value=0, bounds_error=False
    )
    hrf_resampled = interpolator(target_times_clipped)

    return hrf_resampled


def _normalize_hrf_to_unit_peak(hrf: np.ndarray) -> np.ndarray:
    """
    Normalize HRF so that peak amplitude is 1.0.

    This ensures that a single standalone event produces a response with max height 1.
    When multiple events are convolved and summed, the response can exceed 1.

    Parameters
    ----------
    hrf : np.ndarray
        HRF to normalize

    Returns
    -------
    hrf_normalized : np.ndarray
        HRF with peak = 1.0
    """
    peak = np.max(np.abs(hrf))  # Use abs to handle any sign issues
    if peak > 0:
        # Normalize so positive peak = 1.0
        return hrf / np.max(hrf)
    return hrf


def load_canonical_hrf_library(
    microtime_dt: float = 0.1,
    hrf_duration: float = 32.0,
    stim_duration: float = 0.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Load the pre-computed canonical HRF library from file.

    The library contains 20 HRFs with varying timing parameters (peak times
    ranging from ~2.7s to ~5.7s). Each HRF is normalized to peak=1.0.

    Parameters
    ----------
    microtime_dt : float, default=0.1
        Temporal resolution in seconds. The native file resolution is 0.1s.
        If microtime_dt == 0.1, no resampling is needed (preserves precision).
        If different, HRFs are resampled via linear interpolation.
    hrf_duration : float, default=32.0
        Duration of HRF in seconds (truncates or zero-pads as needed)
    stim_duration : float, default=0.0
        Stimulus duration in seconds. If > 0, HRFs are convolved with a boxcar
        of this duration. This creates HRFs appropriate for block designs.
        NOTE: Typically you want stim_duration=0 (impulse response) and handle
        duration in the onset matrix instead.
    device : torch.device, optional
        Device for output tensor

    Returns
    -------
    hrf_library : torch.Tensor
        Shape (n_hrfs, n_timepoints) where n_timepoints = ceil(hrf_duration / microtime_dt)

    Notes
    -----
    The HRF library file is at 0.1s resolution (501 timepoints = 50.1s).
    Each HRF is:
    1. Optionally convolved with stimulus duration boxcar (at 0.1s resolution)
    2. Normalized to peak amplitude = 1.0
    3. Resampled to microtime_dt if different from 0.1s
    4. Truncated to hrf_duration

    For maximum precision, use microtime_dt=0.1 (the native resolution).
    """
    if device is None:
        device = get_device()

    # Load raw library (501 timepoints × 20 HRFs at 0.1s resolution)
    raw_library = _load_hrf_from_file(_HRF_LIBRARY_FILE)
    n_file_timepoints, n_hrfs = raw_library.shape

    # Time vector at source resolution (for convolution)
    t_source = np.arange(n_file_timepoints) * _HRF_FILE_RESOLUTION

    # Process each HRF
    hrfs_processed = []
    for i in range(n_hrfs):
        hrf_raw = raw_library[:, i]

        # Convolve with stimulus duration if specified
        if stim_duration > 0:
            # Create boxcar at source resolution
            boxcar = np.zeros_like(t_source)
            stim_samples = int(stim_duration / _HRF_FILE_RESOLUTION)
            boxcar[:stim_samples] = 1
            # Convolve and truncate to original length
            hrf_convolved = np.convolve(hrf_raw, boxcar, mode="full")[: len(t_source)]
            hrf_raw = hrf_convolved

        # Normalize at source resolution (0.1s)
        hrf_normalized = _normalize_hrf_to_unit_peak(hrf_raw)

        # Resample only if target resolution differs from native 0.1s
        if abs(microtime_dt - _HRF_FILE_RESOLUTION) > 1e-6:
            hrf_final = _resample_hrf(
                hrf_normalized,
                source_dt=_HRF_FILE_RESOLUTION,
                target_dt=microtime_dt,
                target_duration=hrf_duration,
            )
        else:
            # Native resolution - just truncate to duration
            n_samples = int(np.ceil(hrf_duration / _HRF_FILE_RESOLUTION))
            hrf_final = hrf_normalized[:n_samples]

        hrfs_processed.append(hrf_final)

    # Stack into library (n_hrfs, n_timepoints)
    hrf_library = np.stack(hrfs_processed, axis=0)

    return to_tensor(hrf_library, device=device, dtype=torch.float32)


def load_canonical_hrf_basic(
    microtime_dt: float = 0.1,
    hrf_duration: float = 32.0,
    stim_duration: float = 0.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Load the single canonical (basic) HRF from file.

    This is the standard SPM-style double-gamma HRF (GLMsingle/nilearn style),
    used as a baseline for comparison against HRF library optimization.

    Parameters
    ----------
    microtime_dt : float, default=0.1
        Temporal resolution in seconds. The native file resolution is 0.1s.
        If microtime_dt == 0.1, no resampling is needed (preserves precision).
    hrf_duration : float, default=32.0
        Duration of HRF in seconds
    stim_duration : float, default=0.0
        Stimulus duration in seconds. If > 0, HRF is convolved with a boxcar
        of this duration. NOTE: Typically use stim_duration=0 and handle
        duration in the onset matrix instead.
    device : torch.device, optional
        Device for output tensor

    Returns
    -------
    hrf : torch.Tensor
        Shape (n_timepoints,) where n_timepoints = ceil(hrf_duration / microtime_dt)
        Normalized so peak=1.0.

    Notes
    -----
    For maximum precision, use microtime_dt=0.1 (the native resolution).
    """
    if device is None:
        device = get_device()

    # Load raw HRF (490 timepoints at 0.1s = 49s)
    hrf_raw = _load_hrf_from_file(_HRF_BASIC_FILE)
    n_file_timepoints = len(hrf_raw)

    # Convolve with stimulus duration if specified
    if stim_duration > 0:
        t_source = np.arange(n_file_timepoints) * _HRF_FILE_RESOLUTION
        boxcar = np.zeros_like(t_source)
        stim_samples = int(stim_duration / _HRF_FILE_RESOLUTION)
        boxcar[:stim_samples] = 1
        hrf_raw = np.convolve(hrf_raw, boxcar, mode="full")[:n_file_timepoints]

    # Normalize at source resolution (0.1s)
    hrf_normalized = _normalize_hrf_to_unit_peak(hrf_raw)

    # Resample only if target resolution differs from native 0.1s
    if abs(microtime_dt - _HRF_FILE_RESOLUTION) > 1e-6:
        hrf_final = _resample_hrf(
            hrf_normalized,
            source_dt=_HRF_FILE_RESOLUTION,
            target_dt=microtime_dt,
            target_duration=hrf_duration,
        )
    else:
        # Native resolution - just truncate to duration
        n_samples = int(np.ceil(hrf_duration / _HRF_FILE_RESOLUTION))
        hrf_final = hrf_normalized[:n_samples]

    return to_tensor(hrf_final, device=device, dtype=torch.float32)


def get_canonical_hrf(
    stim_duration: float,
    tr: float,
    duration: float = 32.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
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
    c = 1 / 6.0  # Undershoot ratio

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
        hrf = np.convolve(hrf, boxcar, mode="full")[: len(t_highres)]

    # Normalize to peak
    hrf = hrf / np.max(hrf)

    # Downsample to TR
    sample_indices = np.arange(0, len(t_highres), int(tr / dt))
    hrf_downsampled = hrf[sample_indices]

    return to_tensor(hrf_downsampled, device=device, dtype=torch.float32)


def get_spmg1_hrf(
    microtime_dt: float = 0.1,
    stim_duration: float = 0.0,
    hrf_duration: float = 32.0,
    normalize_peak: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate AFNI's SPMG1 canonical HRF.

    This exactly matches AFNI's SPMG1 basis function:
        h(t) = exp(-t) * (A1*t^P1 - A2*t^P2)
    where:
        A1 = 0.0083333333, P1 = 5  (main positive lobe, peak ~5s)
        A2 = 1.274527e-13, P2 = 15 (undershoot, peak ~15s)

    Parameters
    ----------
    microtime_dt : float, default=0.1
        Temporal resolution in seconds. Default 0.1s matches the standard
        microtime resolution used throughout the pipeline.
    stim_duration : float, default=0.0
        Stimulus duration in seconds. If > 0, HRF is convolved with a boxcar
        of this duration (like AFNI's SPMG1(duration)). If 0, returns the
        impulse response HRF. NOTE: Typically use stim_duration=0 and handle
        duration in the onset matrix instead.
    hrf_duration : float, default=32.0
        Total duration of HRF in seconds
    normalize_peak : bool, default=True
        If True, normalize so peak absolute value = 1.0 (like AFNI's SPMG1(0)).
        If False, return raw unnormalized HRF (AFNI's default SPMG1 without duration).
    device : torch.device, optional
        Device for output tensor

    Returns
    -------
    hrf : torch.Tensor
        Shape (n_timepoints,) where n_timepoints = ceil(hrf_duration / microtime_dt)

    Notes
    -----
    AFNI documentation:
    - SPMG1 = exp(-t)*(A1*t^P1-A2*t^P2) where
      A1 = 0.0083333333  P1 = 5  (main positive lobe)
      A2 = 1.274527e-13  P2 = 15 (undershoot part)
    - SPMG1(duration) convolves with square wave and normalizes to peak=1
    - SPMG1(0) produces usual shape but normalized to peak=1

    The function starts at t=0 with h(0)=0 and rises to peak around t=5s.
    This ensures causal convolution: the response begins at stimulus onset.
    """
    if device is None:
        device = get_device()

    # AFNI SPMG1 parameters (exact values from AFNI source)
    A1 = 0.0083333333  # = 1/120
    P1 = 5
    A2 = 1.274527e-13
    P2 = 15

    # Generate at high resolution (0.01s) for accurate computation, then resample
    # This ensures precision even if microtime_dt is coarser
    compute_dt = min(0.01, microtime_dt)  # At least as fine as target
    t_highres = np.arange(0, hrf_duration + compute_dt, compute_dt)

    # AFNI SPMG1 formula: exp(-t) * (A1*t^P1 - A2*t^P2)
    # Note: t=0 gives 0, which is correct (HRF starts at 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        hrf_highres = np.exp(-t_highres) * (A1 * np.power(t_highres, P1) - A2 * np.power(t_highres, P2))
    hrf_highres[0] = 0  # Ensure h(0) = 0

    # Convolve with stimulus duration if specified
    if stim_duration > 0:
        # Create boxcar at compute resolution
        stim_samples = max(1, int(stim_duration / compute_dt))
        boxcar = np.ones(stim_samples)
        # Convolve (this extends the signal, then truncate)
        hrf_convolved = np.convolve(hrf_highres, boxcar, mode='full')
        # Scale by dt to get proper integral (like AFNI does)
        hrf_convolved = hrf_convolved * compute_dt
        # Truncate to original duration
        hrf_highres = hrf_convolved[:len(t_highres)]

    # Normalize to peak = 1.0 if requested
    if normalize_peak:
        peak_val = np.max(np.abs(hrf_highres))
        if peak_val > 0:
            hrf_highres = hrf_highres / peak_val

    # Resample to target microtime_dt if different from compute resolution
    if abs(microtime_dt - compute_dt) > 1e-6:
        n_target = int(np.ceil(hrf_duration / microtime_dt))
        target_times = np.arange(n_target) * microtime_dt
        target_times_clipped = np.clip(target_times, 0, t_highres[-1])
        interpolator = interp1d(t_highres, hrf_highres, kind='linear', fill_value=0, bounds_error=False)
        hrf_final = interpolator(target_times_clipped)
    else:
        hrf_final = hrf_highres

    return to_tensor(hrf_final, device=device, dtype=torch.float32)


def get_canonical_hrf_library(
    stim_duration: float,
    tr: float,
    n_hrfs: int = 20,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
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
            hrf = np.convolve(hrf, boxcar, mode="full")[: len(t_highres)]

        # Normalize
        hrf = hrf / np.max(hrf)

        # Downsample
        sample_indices = np.arange(0, len(t_highres), int(tr / dt))
        hrf_downsampled = hrf[sample_indices]

        hrfs.append(hrf_downsampled)

    hrf_library = np.stack(hrfs, axis=0)

    return to_tensor(hrf_library, device=device, dtype=torch.float32)


def pighs_halfcos(
    m1: float,
    m2: float,
    m3: float,
    m4: float,
    c2: float,
    duration: float = 32.0,
    sample_rate: float = 0.05,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate HRF using half-cosine segments (PIGHS method).

    PIGHS = Parametric Individually Generated HRFs

    Creates smooth HRF shapes using connected half-cosine segments:
    - Segment 1 (0 to m1): Flat at baseline (delay)
    - Segment 2 (m1 to m1+m2): Rise from 0 to peak (1.0)
    - Segment 3 (m1+m2 to m1+m2+m3): Fall from peak to undershoot (-c2)
    - Segment 4 (m1+m2+m3 to m1+m2+m3+m4): Recovery from undershoot to baseline (0)

    Parameters
    ----------
    m1 : float
        Delay before rise starts (seconds)
    m2 : float
        Time from start to peak (seconds)
    m3 : float
        Time from peak to undershoot minimum (seconds)
    m4 : float
        Time from undershoot minimum back to baseline (seconds)
    c2 : float
        Undershoot magnitude as fraction of peak (e.g., 0.2 = 20% undershoot)
    duration : float
        Total HRF duration in seconds
    sample_rate : float
        Sampling rate for generation (seconds)
    device : torch.device, optional
        Device for tensor

    Returns
    -------
    hrf : torch.Tensor
        Generated HRF, normalized to peak = 1.0
    """
    if device is None:
        device = get_device()

    t = np.arange(0, duration, sample_rate)
    hrf = np.zeros_like(t)

    # Segment 1: Flat at 0 until m1 (delay)
    # (already zeros)

    # Segment 2: Rise from 0 to 1 (half cosine going up)
    # Formula: 0.5 * (1 - cos(pi * t_rel / m2))
    # At t_rel=0: 0.5 * (1 - 1) = 0
    # At t_rel=m2: 0.5 * (1 - (-1)) = 1
    mask2 = (t > m1) & (t <= m1 + m2)
    if np.any(mask2):
        t_rel = t[mask2] - m1
        hrf[mask2] = 0.5 * (1 - np.cos(np.pi * t_rel / m2))

    # Segment 3: Fall from 1 to -c2 (half cosine going down)
    # Formula: ((1 + c2) / 2) * cos(pi * t_rel / m3) + ((1 - c2) / 2)
    # At t_rel=0: ((1+c2)/2) * 1 + ((1-c2)/2) = 1
    # At t_rel=m3: ((1+c2)/2) * (-1) + ((1-c2)/2) = -c2
    mask3 = (t > m1 + m2) & (t <= m1 + m2 + m3)
    if np.any(mask3):
        t_rel = t[mask3] - (m1 + m2)
        hrf[mask3] = ((1 + c2) / 2) * np.cos(np.pi * t_rel / m3) + ((1 - c2) / 2)

    # Segment 4: Recovery from -c2 to 0 (half cosine going up)
    # Formula: (-c2 / 2) * cos(pi * t_rel / m4) + (-c2 / 2)
    # At t_rel=0: (-c2/2) * 1 + (-c2/2) = -c2
    # At t_rel=m4: (-c2/2) * (-1) + (-c2/2) = 0
    mask4 = (t > m1 + m2 + m3) & (t <= m1 + m2 + m3 + m4)
    if np.any(mask4):
        t_rel = t[mask4] - (m1 + m2 + m3)
        hrf[mask4] = (-c2 / 2) * np.cos(np.pi * t_rel / m4) + (-c2 / 2)

    # Normalize to peak = 1.0
    if np.max(np.abs(hrf)) > 0:
        hrf = hrf / np.max(hrf)

    return to_tensor(hrf, device=device, dtype=torch.float32)


# Backwards compatibility alias
flobs_halfcos = pighs_halfcos


def create_pighs_library(
    n_hrfs: int = 20,
    peak_time_range: Tuple[float, float] = (3, 10),
    rise_fraction_range: Tuple[float, float] = (0.3, 0.7),
    fall_time_range: Tuple[float, float] = (3, 10),
    recovery_time_range: Tuple[float, float] = (3, 12),
    undershoot_range: Tuple[float, float] = (0, 0.35),
    duration: float = 32.0,
    microtime_dt: float = 0.1,
    stim_duration: float = 0.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Create HRF library using PIGHS (Parametric Individually Generated HRFs).

    Uses stratified sampling for uniform coverage of the parameter space:
    - Peak times are evenly spaced (grid) to guarantee full range coverage
    - Secondary parameters use Latin Hypercube Sampling for variety

    Parameters
    ----------
    n_hrfs : int
        Number of HRFs to generate
    peak_time_range : tuple
        (min, max) for time to peak in seconds (e.g., (3, 10) for 3-10s peaks)
    rise_fraction_range : tuple
        (min, max) fraction of peak_time spent on delay vs rise.
        E.g., 0.3 means 30% delay, 70% rise; 0.7 means 70% delay, 30% rise.
    fall_time_range : tuple
        (min, max) for time from peak to undershoot minimum (seconds)
    recovery_time_range : tuple
        (min, max) for time from undershoot back to baseline (seconds)
    undershoot_range : tuple
        (min, max) for undershoot magnitude as fraction of peak
    duration : float
        HRF duration in seconds
    microtime_dt : float, default=0.1
        Temporal resolution in seconds. Default 0.1s matches the native resolution
        used throughout the pipeline for maximum precision.
    stim_duration : float
        Stimulus duration to convolve with (0 = impulse response)
    device : torch.device, optional
        Device for tensors

    Returns
    -------
    hrf_library : torch.Tensor
        (n_hrfs, n_timepoints) library of HRFs at microtime_dt resolution
    params : dict
        Dictionary of parameters used for each HRF
    """
    if device is None:
        device = get_device()

    # High-resolution generation (0.01s for smooth half-cosines)
    gen_dt = 0.01

    # Stratified sampling strategy:
    # 1. Grid sampling for peak_time (most important - guarantees coverage)
    # 2. LHS for secondary parameters within each peak_time stratum

    # Create evenly-spaced peak times that span the full range
    # This guarantees we get HRFs with peaks at 3s, 4s, 5s, etc.
    peak_times = np.linspace(peak_time_range[0], peak_time_range[1], n_hrfs)

    # Use LHS for the other 4 parameters (rise_fraction, fall, recovery, undershoot)
    from scipy.stats import qmc

    sampler = qmc.LatinHypercube(d=4)
    samples = sampler.random(n=n_hrfs)

    # Sample rise_fraction to split peak_time into delay (m1) and rise (m2)
    rise_fractions = rise_fraction_range[0] + samples[:, 0] * (
        rise_fraction_range[1] - rise_fraction_range[0]
    )

    # Compute m1 (delay) and m2 (rise) from peak_time and rise_fraction
    # rise_fraction = m2 / peak_time, so m2 = peak_time * rise_fraction
    m2_vals = peak_times * rise_fractions
    m1_vals = peak_times - m2_vals  # delay = peak_time - rise_time

    # Ensure minimum rise time (at least 1 second)
    m2_vals = np.maximum(m2_vals, 1.0)
    m1_vals = peak_times - m2_vals
    m1_vals = np.maximum(m1_vals, 0.0)  # delay can't be negative

    # Sample other parameters from LHS (indices 1, 2, 3 since rise_fraction is 0)
    m3_vals = fall_time_range[0] + samples[:, 1] * (
        fall_time_range[1] - fall_time_range[0]
    )
    m4_vals = recovery_time_range[0] + samples[:, 2] * (
        recovery_time_range[1] - recovery_time_range[0]
    )
    c2_vals = undershoot_range[0] + samples[:, 3] * (
        undershoot_range[1] - undershoot_range[0]
    )

    # Generate HRFs at high resolution
    t_highres = np.arange(0, duration, gen_dt)
    hrfs_highres = []

    for i in range(n_hrfs):
        hrf = pighs_halfcos(
            m1_vals[i],
            m2_vals[i],
            m3_vals[i],
            m4_vals[i],
            c2_vals[i],
            duration,
            gen_dt,
            device="cpu",
        )
        hrfs_highres.append(hrf.numpy())

    hrfs_highres = np.stack(hrfs_highres, axis=0)

    # Convolve with stimulus if needed
    if stim_duration > 0:
        boxcar = np.zeros(int(30 / gen_dt))  # 30s window
        boxcar[: int(stim_duration / gen_dt)] = 1

        hrfs_convolved = np.zeros_like(hrfs_highres)
        for i in range(n_hrfs):
            convolved = np.convolve(hrfs_highres[i], boxcar, mode="full")
            hrfs_convolved[i] = convolved[: len(t_highres)]
            # Renormalize
            if np.max(hrfs_convolved[i]) > 0:
                hrfs_convolved[i] = hrfs_convolved[i] / np.max(hrfs_convolved[i])

        hrfs_highres = hrfs_convolved

    # Normalize at high resolution before resampling
    for i in range(n_hrfs):
        peak = np.max(hrfs_highres[i])
        if peak > 0:
            hrfs_highres[i] = hrfs_highres[i] / peak

    # Resample to target microtime_dt resolution
    n_target = int(np.ceil(duration / microtime_dt))
    target_times = np.arange(n_target) * microtime_dt
    source_times = np.arange(len(t_highres)) * gen_dt

    hrfs_resampled = np.zeros((n_hrfs, n_target))
    for i in range(n_hrfs):
        interpolator = interp1d(
            source_times, hrfs_highres[i], kind="linear", fill_value=0, bounds_error=False
        )
        hrfs_resampled[i] = interpolator(np.clip(target_times, 0, source_times[-1]))

    hrf_library = to_tensor(hrfs_resampled, device=device, dtype=torch.float32)

    params = {
        "peak_time": peak_times,
        "m1": m1_vals,
        "m2": m2_vals,
        "m3": m3_vals,
        "m4": m4_vals,
        "c2": c2_vals,
    }

    return hrf_library, params


# Backwards compatibility alias
def create_flobs_library(
    n_hrfs: int = 20,
    m1_range: Tuple[float, float] = (0, 2),
    m2_range: Tuple[float, float] = (3, 8),
    m3_range: Tuple[float, float] = (3, 10),
    m4_range: Tuple[float, float] = (3, 12),
    c2_range: Tuple[float, float] = (0, 0.35),
    **kwargs,
) -> Tuple[torch.Tensor, Dict]:
    """Deprecated: Use create_pighs_library instead."""
    # Convert old m1/m2 ranges to new peak_time/rise_fraction
    peak_time_range = (m1_range[0] + m2_range[0], m1_range[1] + m2_range[1])
    return create_pighs_library(
        n_hrfs=n_hrfs,
        peak_time_range=peak_time_range,
        fall_time_range=m3_range,
        recovery_time_range=m4_range,
        undershoot_range=c2_range,
        **kwargs,
    )


def get_hrf_library(
    mode: str = "library",
    stim_duration: float = 0.0,
    microtime_dt: float = 0.1,
    n_hrfs: int = 20,
    hrf_duration: float = 32.0,
    device: Optional[torch.device] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Get HRF library for GLM fitting.

    For 'library' and 'single' modes, HRFs are loaded from pre-computed
    TSV files, normalized to peak=1.0. For 'spmg1', the AFNI SPMG1 formula
    is used. All HRFs are returned at the specified microtime resolution.

    Parameters
    ----------
    mode : str
        HRF source selection:
        - 'library' - Load 20-HRF library from file (recommended for HRF optimization)
        - 'pighs' - Generate PIGHS (Parametric Individually Generated HRFs)
        - 'single' or 'glmsingle' - Load single canonical HRF from file (GLMsingle-style)
        - 'spmg1' - Generate AFNI's SPMG1 canonical HRF
    stim_duration : float, default=0.0
        Stimulus duration in seconds. If > 0, HRF is convolved with a boxcar.
        NOTE: Typically use stim_duration=0 (impulse response) and encode
        duration in the onset matrix instead.
    microtime_dt : float, default=0.1
        Temporal resolution in seconds. Default 0.1s is the native resolution
        of the HRF library files, avoiding any resampling for maximum precision.
    n_hrfs : int, default=20
        Number of HRFs (only used for 'pighs' mode; library always returns 20)
    hrf_duration : float, default=32.0
        Duration of HRF in seconds
    device : torch.device, optional
        Device for tensors
    **kwargs : dict
        Additional arguments passed to PIGHS generator

    Returns
    -------
    hrf_library : torch.Tensor
        Shape depends on mode:
        - 'library': (20, n_timepoints)
        - 'pighs': (n_hrfs, n_timepoints)
        - 'single', 'glmsingle', 'spmg1': (n_timepoints,)

        Where n_timepoints = ceil(hrf_duration / microtime_dt)

    Notes
    -----
    HRFs are normalized so that a single standalone event produces a response
    with peak amplitude = 1.0. When multiple events are convolved and summed,
    the response can exceed 1.0.

    For maximum precision, use microtime_dt=0.1 (the native resolution of the
    library files). This avoids any resampling.
    """
    if device is None:
        device = get_device()

    mode_lower = mode.lower()

    if mode_lower in ("single", "glmsingle"):
        # Load basic canonical HRF from file (GLMsingle/nilearn-style)
        return load_canonical_hrf_basic(
            microtime_dt=microtime_dt,
            hrf_duration=hrf_duration,
            stim_duration=stim_duration,
            device=device,
        )

    elif mode_lower == "spmg1":
        # Generate AFNI's SPMG1 canonical HRF
        return get_spmg1_hrf(
            microtime_dt=microtime_dt,
            hrf_duration=hrf_duration,
            stim_duration=stim_duration,
            normalize_peak=True,
            device=device,
        )

    elif mode_lower in ("library", "canonical"):
        # Load 20-HRF library from file
        return load_canonical_hrf_library(
            microtime_dt=microtime_dt,
            hrf_duration=hrf_duration,
            stim_duration=stim_duration,
            device=device,
        )

    elif mode_lower in ("pighs", "flobs"):
        # Generate PIGHS HRFs programmatically
        # Note: PIGHS needs to be updated to use microtime_dt
        library, _ = create_pighs_library(
            n_hrfs=n_hrfs,
            microtime_dt=microtime_dt,
            stim_duration=stim_duration,
            duration=hrf_duration,
            device=device,
            **kwargs,
        )
        return library

    else:
        raise ValueError(
            f"Unknown mode: {mode}. Choose 'library', 'pighs', 'single', 'glmsingle', or 'spmg1'"
        )
