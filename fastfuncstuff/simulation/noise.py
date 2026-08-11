"""
Fast GPU-accelerated fMRI noise generation

This module provides two complementary uses of temporal autocorrelation models:

1. NOISE GENERATION (Simulation):
   - generate_ar1_noise(), generate_ar_noise(), generate_arma_noise()
   - Purpose: Create realistic autocorrelated noise for fMRI simulations
   - Use case: Simulate data with temporal structure matching real fMRI

2. GLM PREWHITENING (Analysis):
   - See metrics_empirical.py: estimate_ar1_coefficient(), gls_fit()
   - Purpose: Account for autocorrelation when fitting GLMs to real/simulated data
   - Use case: Correct standard errors and improve parameter estimates
   - Similar to AFNI's 3dREMLfit (Restricted Maximum Likelihood + prewhitening)

The same AR/ARMA models serve both purposes:
- In simulation: Generate noise with known autocorrelation
- In analysis: Estimate autocorrelation from residuals, then prewhiten

GPU-accelerated implementations provide massive speedup over CPU-based methods
like AFNI's 3dREMLfit, while maintaining mathematical equivalence.

References:
- AFNI 3dREMLfit: https://github.com/afni/afni (src/3dREMLfit.c)
- Worsley & Friston (1995): Analysis of fMRI time-series revisited
- Woolrich et al. (2001): Temporal autocorrelation in SPM
"""

from __future__ import annotations

import numpy as np
import torch

from fastfuncstuff.utils import get_device, to_tensor


def _infer_time_axis(
    data: torch.Tensor,
    n_design_rows: int | None = None,
    n_mask_elements: int | None = None,
    max_lines: int = 256,
) -> int:
    """Which axis of a 2D array is time? Ask the caller, then the data -- never the shape.

    Shape cannot answer this. "The longer axis is time" is wrong for fMRI, where
    voxels outnumber timepoints; "the shorter axis is time" then fails on the
    small-ROI-long-run case.

    Two better sources, in order. A design matrix states the number of
    timepoints outright, and a mask states the number of voxels, so when either
    is present and matches exactly one axis there is nothing to infer. Failing
    that, the defining property of the time axis is the very thing being
    measured: a timeseries is autocorrelated at lag 1, a row of unrelated voxels
    is not.

    Only genuinely white data reaches the final fallback, and there both
    readings give the same near-zero answer -- so it defers to the documented
    (n_timepoints, n_voxels) layout rather than guessing.
    """
    n_rows, n_cols = data.shape

    if n_design_rows is not None:
        if n_rows == n_design_rows and n_cols != n_design_rows:
            return 0
        if n_cols == n_design_rows and n_rows != n_design_rows:
            return 1
    if n_mask_elements is not None:
        if n_cols == n_mask_elements and n_rows != n_mask_elements:
            return 0
        if n_rows == n_mask_elements and n_cols != n_mask_elements:
            return 1

    def mean_abs_lag1(x: torch.Tensor) -> float:
        # x: (n_lines, length) -- autocorrelation measured along dim 1
        if x.shape[0] > max_lines:
            idx = torch.linspace(0, x.shape[0] - 1, max_lines, device=x.device).long()
            x = x[idx]
        x = x.to(torch.float32)
        x = x - x.mean(dim=1, keepdim=True)
        denom = (x * x).sum(dim=1)
        num = (x[:, :-1] * x[:, 1:]).sum(dim=1)
        valid = denom > 1e-20
        if not bool(valid.any()):
            return 0.0
        return float((num[valid] / denom[valid]).abs().mean().item())

    along_0 = mean_abs_lag1(data.T)  # treat axis 0 as time
    along_1 = mean_abs_lag1(data)  # treat axis 1 as time

    if max(along_0, along_1) < 0.02:
        return 0  # white data: both readings agree, so honour the documented layout
    return 0 if along_0 >= along_1 else 1


def _synthesise_from_spectrum(
    power_spectrum: torch.Tensor,
    n_samples: int,
    dt: float,
    n_voxels: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Synthesise a Gaussian timeseries with a prescribed one-sided PSD.

    Returns ``(n_samples, n_voxels)`` whose variance equals the integral of
    ``power_spectrum`` over frequency. That scaling is what lets components
    synthesised on *different* frequency grids be added together on one common
    scale -- without it, a component's amplitude would depend on the grid it
    happened to be built on rather than on the strength the caller asked for.
    """
    amplitude = torch.sqrt(power_spectrum)
    n_freqs = power_spectrum.shape[0]

    # Complex-Gaussian coefficients, not unit-amplitude random phases. Randomising
    # only the phase makes |X_k| deterministic, so every voxel gets a byte-identical
    # periodogram and the noise has no spectral variability at all -- a random-phase
    # surrogate rather than a Gaussian process. Real fMRI noise has chi-squared
    # distributed periodogram ordinates, and any statistic sensitive to that (an
    # estimated AR coefficient, a spectral fit) would otherwise be unrealistically
    # stable across voxels. Var of each part is 1/2 so that E|X_k|^2 = power.
    amp = amplitude.unsqueeze(1)
    real = torch.randn(n_freqs, n_voxels, device=device, generator=generator)
    imag = torch.randn(n_freqs, n_voxels, device=device, generator=generator)
    spectrum = amp * torch.complex(real, imag) / np.sqrt(2.0)

    # DC (and Nyquist, when n_samples is even) have no conjugate partner, so they
    # must be real or irfft silently discards their imaginary part -- which would
    # make the realised variance disagree with the requested spectrum. Rebuilt out
    # of place and kept complex: MPS cannot cat a real tensor with a complex one.
    def _force_real(col: torch.Tensor) -> torch.Tensor:
        # sqrt(2) restores the variance dropped with the imaginary part, so these
        # bins carry E[X^2] = power like every other bin.
        real_part = col.real * np.sqrt(2.0)
        return torch.complex(real_part, torch.zeros_like(real_part)).unsqueeze(0)

    spectrum = torch.cat([_force_real(spectrum[0]), spectrum[1:]], dim=0)
    if n_samples % 2 == 0:
        spectrum = torch.cat([spectrum[:-1], _force_real(spectrum[-1])], dim=0)

    series = torch.fft.irfft(spectrum, n=n_samples, dim=0).real

    # Variance of the synthesis above, analytically: bins with a conjugate
    # partner contribute twice. Computed rather than measured so a single-voxel
    # call keeps its natural sampling variability.
    weights = torch.full((n_freqs,), 2.0, device=device, dtype=amplitude.dtype)
    weights[0] = 1.0
    if n_samples % 2 == 0:
        weights[-1] = 1.0
    synth_var = (weights * amplitude**2).sum() / (n_samples**2)

    target_var = power_spectrum[1:].sum() / (n_samples * dt)  # PSD integral, DC excluded
    scale = torch.sqrt(target_var / synth_var.clamp_min(1e-30))
    return series * scale


def generate_fmri_noise(
    tr: float,
    duration_s: float,
    matrix_size: tuple[int, int] = (1, 1),
    fs_high: float = 10.0,
    resp_freq: float = 0.35,
    resp_width: float = 0.1,
    resp_strength: float = 3.0,
    cardiac_freq: float = 1.0,
    cardiac_width: float = 0.05,
    cardiac_strength: float = 5.0,
    pink_exp: float = 1.0,
    normalize: bool = True,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Generate realistic fMRI noise with 1/f spectrum and physiological components

    This creates noise that mimics real fMRI data characteristics:
    - 1/f (pink) noise spectrum
    - Respiratory peak (~0.3 Hz)
    - Cardiac peak (~1 Hz)
    - Independent noise per voxel

    Parameters
    ----------
    tr : float
        Repetition time in seconds
    duration_s : float
        Total duration in seconds
    matrix_size : tuple
        (rows, cols) for 2D slice, or (1,1) for single voxel
    fs_high : float
        High sampling rate for generation (Hz), default: 10
    resp_freq : float
        Respiratory frequency in Hz (default: 0.35)
    resp_width : float
        Width of respiratory peak (default: 0.1)
    resp_strength : float
        Strength of respiratory component (default: 3.0)
    cardiac_freq : float
        Cardiac frequency in Hz (default: 1.0)
    cardiac_width : float
        Width of cardiac peak (default: 0.05)
    cardiac_strength : float
        Strength of cardiac component (default: 5.0)
    pink_exp : float
        Exponent for 1/f noise (default: 1.0, range: 0.5-1.5). This is the
        realised spectral slope of the output, not merely a request: see Notes.
    normalize : bool
        Normalize to unit variance (default: True)
    device : torch.device, optional
        Device for computation
    generator : torch.Generator, optional
        Seeded generator for reproducible noise without touching global RNG state.

    Returns
    -------
    noise : torch.Tensor
        (n_trs, rows, cols) noise time series
        If matrix_size=(1,1), returns (n_trs,)

    Notes
    -----
    The background and the physiological peaks are synthesised on *different*
    frequency grids, on purpose:

    - **1/f background** is built directly on the TR grid. It describes the
      spectrum of the sampled timeseries, so there is nothing above Nyquist for
      it to fold down from. Building it at ``fs_high`` and decimating (as this
      function used to) folded 0.5-5 Hz back as a near-flat pedestal and
      delivered roughly half the requested slope -- ``pink_exp=1.0`` measured
      0.48. The realised slope now matches ``pink_exp`` above the 0.01 Hz knee
      that ``1/(f + 0.01)`` introduces.
    - **Respiratory and cardiac peaks** are built at ``fs_high`` and decimated,
      so they alias when they sit above Nyquist. That is deliberate and
      physical: cardiac pulsation near 1 Hz is genuinely undersampled at any
      ordinary TR, and real data carries it folded down. At TR=2 s a 1 Hz
      cardiac peak lands near DC, which is where it belongs -- not a bug.

    ``fs_high`` is rounded to the nearest exact integer multiple of ``1/tr``, so
    the output TR is exactly ``tr``. Requesting a sub-second TR used to truncate
    the ratio and shift every frequency (TR=0.35 became 0.30).
    """
    if device is None:
        device = get_device()

    n_voxels = matrix_size[0] * matrix_size[1]
    n_trs = int(duration_s / tr)

    # The decimation ratio must be an exact integer or the series carries a
    # different TR than the caller asked for: int(fs_high / (1 / tr)) truncates,
    # so TR=0.35 silently became 0.30 (-14%) and TR=0.25 became 0.20 (-20%),
    # putting every requested frequency in the wrong place. Round to the nearest
    # integer ratio and derive the generation rate from it instead.
    decimation = max(1, int(round(fs_high * tr)))
    fs_gen = decimation / tr

    # The 1/f background describes the spectrum of the *sampled* timeseries, so it
    # is synthesised directly on the TR grid. Building it at fs_high and decimating
    # without an anti-alias filter folds 0.5-5 Hz back as a near-flat pedestal,
    # which measurably flattens the result: pink_exp=1.0 delivered a realised
    # slope of ~0.48 before this split.
    freqs_tr = torch.fft.rfftfreq(n_trs, d=tr).to(device)
    background = 1 / (freqs_tr + 0.01) ** pink_exp
    noise_ts = _synthesise_from_spectrum(
        background, n_trs, tr, n_voxels, device, generator=generator
    )

    # Physiological peaks are a different matter: cardiac pulsation near 1 Hz is
    # genuinely undersampled at any ordinary TR, and real data really does carry
    # it folded down. So these are synthesised at the high rate and decimated,
    # which aliases them exactly as the scanner would.
    if resp_strength > 0 or cardiac_strength > 0:
        n_samples = n_trs * decimation
        dt = 1 / fs_gen
        freqs_hi = torch.fft.rfftfreq(n_samples, d=dt).to(device)

        physio = torch.zeros_like(freqs_hi)
        if resp_strength > 0:
            physio = physio + resp_strength * torch.exp(
                -((freqs_hi - resp_freq) ** 2) / (2 * resp_width**2)
            )
        if cardiac_strength > 0:
            physio = physio + cardiac_strength * torch.exp(
                -((freqs_hi - cardiac_freq) ** 2) / (2 * cardiac_width**2)
            )

        physio_hi = _synthesise_from_spectrum(
            physio, n_samples, dt, n_voxels, device, generator=generator
        )
        noise_ts = noise_ts + physio_hi[::decimation][:n_trs, :]

    # Normalize per voxel if requested
    if normalize:
        noise_ts = (noise_ts - noise_ts.mean(dim=0, keepdim=True)) / (
            noise_ts.std(dim=0, keepdim=True) + 1e-10
        )

    # Reshape to spatial dimensions
    if n_voxels > 1:
        noise_ts = noise_ts.reshape(n_trs, matrix_size[0], matrix_size[1])
    else:
        noise_ts = noise_ts.squeeze(1)  # (n_trs,)

    return noise_ts


def generate_fmri_noise_batch(
    tr: float,
    duration_s: float,
    n_batches: int,
    matrix_size: tuple[int, int] = (1, 1),
    device: torch.device | None = None,
    **kwargs,
) -> torch.Tensor:
    """
    Generate multiple noise realizations in batch (for simulation studies)

    Parameters
    ----------
    tr : float
        Repetition time in seconds
    duration_s : float
        Duration in seconds
    n_batches : int
        Number of independent noise realizations
    matrix_size : tuple
        (rows, cols) spatial dimensions
    device : torch.device, optional
        Device for computation
    **kwargs : dict
        Additional arguments passed to generate_fmri_noise

    Returns
    -------
    noise_batch : torch.Tensor
        (n_batches, n_trs, rows, cols) or (n_batches, n_trs) for single voxel
    """
    if device is None:
        device = get_device()

    noise_list = []
    for _i in range(n_batches):
        noise = generate_fmri_noise(tr, duration_s, matrix_size, device=device, **kwargs)
        noise_list.append(noise)

    return torch.stack(noise_list, dim=0)


def add_drift(
    data: torch.Tensor,
    amplitude: float = 0.5,
    n_modes: int = 3,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Add low-frequency drift to fMRI data (simulates scanner drift)

    Parameters
    ----------
    data : torch.Tensor
        (n_timepoints, ...) data to add drift to
    amplitude : float
        Amplitude of drift relative to signal std
    n_modes : int
        Number of low-frequency drift modes (default: 3)
    device : torch.device, optional
        Device for computation

    Returns
    -------
    data_with_drift : torch.Tensor
        Data with drift added
    """
    if device is None:
        device = get_device()

    data = to_tensor(data, device=device)
    n_timepoints = data.shape[0]
    original_shape = data.shape

    # Flatten spatial dimensions
    data_flat = data.reshape(n_timepoints, -1)
    n_voxels = data_flat.shape[1]

    # Create drift basis (low-frequency sines/cosines)
    t = torch.linspace(0, 1, n_timepoints, device=device)
    drift_basis = []

    for mode in range(1, n_modes + 1):
        drift_basis.append(torch.sin(2 * np.pi * mode * t))
        drift_basis.append(torch.cos(2 * np.pi * mode * t))

    drift_basis = torch.stack(drift_basis, dim=1)  # (n_timepoints, 2*n_modes)

    # Random weights per voxel
    weights = torch.randn(drift_basis.shape[1], n_voxels, device=device, generator=generator)

    # Generate drift
    drift = drift_basis @ weights  # (n_timepoints, n_voxels)

    # Scale drift
    data_std = data_flat.std(dim=0, keepdim=True)
    drift = drift * (amplitude * data_std / (drift.std(dim=0, keepdim=True) + 1e-10))

    # Add to data
    data_with_drift = data_flat + drift

    # Reshape back
    return data_with_drift.reshape(original_shape)


def add_motion_artifacts(
    data: torch.Tensor,
    max_displacement: float = 2.0,
    n_spikes: int = 3,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Add motion spikes to fMRI data

    Parameters
    ----------
    data : torch.Tensor
        (n_timepoints, ...) fMRI data
    max_displacement : float
        Maximum displacement amplitude (in signal units)
    n_spikes : int
        Number of motion spikes to add
    device : torch.device, optional
        Device for computation

    Returns
    -------
    data_with_motion : torch.Tensor
        Data with motion artifacts
    spike_times : torch.Tensor
        Time indices of motion spikes
    """
    if device is None:
        device = get_device()

    data = to_tensor(data, device=device)
    n_timepoints = data.shape[0]

    # Random spike times
    spike_times = torch.randperm(n_timepoints - 1)[:n_spikes] + 1  # Avoid first TR

    # Create spike template (impulse response)
    spike_template = torch.tensor([0.0, 1.0, 0.5, 0.2, 0.1], device=device)

    # Add spikes
    data_with_motion = data.clone()

    for spike_time in spike_times:
        # Random amplitude and sign
        amplitude = torch.rand(1, device=device).item() * max_displacement
        sign = torch.randint(0, 2, (1,), device=device).item() * 2 - 1

        # Apply spike with decay
        for i, decay in enumerate(spike_template):
            t = spike_time + i
            if t < n_timepoints:
                data_with_motion[t] += sign * amplitude * decay

    return data_with_motion, spike_times


def generate_ar1_noise(
    rho: float,
    n_timepoints: int,
    n_voxels: int = 1,
    normalize: bool = True,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Generate AR(1) temporally autocorrelated noise

    AR(1) model: y_t = ρ * y_{t-1} + ε_t
    where ε_t ~ N(0, σ²) and ρ is the autocorrelation coefficient

    This is the most important missing feature for realistic fMRI noise!
    Real fMRI data has temporal autocorrelation with ρ ≈ 0.2-0.4

    Parameters
    ----------
    rho : float
        AR(1) coefficient (typically 0.2-0.4 for fMRI)
        Must be in (-1, 1) for stationarity
    n_timepoints : int
        Number of time points
    n_voxels : int, default=1
        Number of independent voxel time series
    normalize : bool, default=True
        Normalize to unit variance
    device : torch.device, optional
        Device for computation

    Returns
    -------
    noise : torch.Tensor
        (n_timepoints, n_voxels) AR(1) noise
        If n_voxels=1, returns (n_timepoints,)

    Notes
    -----
    Implementation uses sequential generation which is vectorized across voxels.
    For n_timepoints=300 and n_voxels=10000, takes ~100ms on GPU.

    Theory: AR(1) variance is σ² / (1 - ρ²), where σ² is innovation variance.
    We normalize innovations to achieve unit variance in the AR(1) process.

    References
    ----------
    - Purdon & Weisskoff (1998): Effect of temporal autocorrelation
    - Woolrich et al. (2001): Prewhitening in fMRI analysis
    """
    if device is None:
        device = get_device()

    # Validate rho
    if not (-1 < rho < 1):
        raise ValueError(f"rho must be in (-1, 1) for stationarity, got {rho}")

    # Initialize output
    y = torch.zeros(n_timepoints, n_voxels, device=device)

    # Innovation variance to achieve unit variance AR(1) process
    # Var(y_t) = σ²_ε / (1 - ρ²) = 1  →  σ²_ε = (1 - ρ²)
    innovation_std = np.sqrt(1 - rho**2)

    # Generate innovations
    epsilon = (
        torch.randn(n_timepoints, n_voxels, device=device, generator=generator) * innovation_std
    )

    # Seed from the stationary distribution BEFORE the recursion. Assigning y[0]
    # afterwards (as this once did) left the loop starting from zeros, so the
    # series ramped up to its stationary variance instead of beginning at it --
    # at rho=0.9, var(y[1]) measured 0.19 against a stationary 1.0 and took ~50
    # samples to recover. Short runs are exactly where that bias bites.
    y[0] = epsilon[0] / np.sqrt(1 - rho**2)

    # Sequential generation (vectorized across voxels)
    # This is fast enough for typical use cases
    for t in range(1, n_timepoints):
        y[t] = rho * y[t - 1] + epsilon[t]

    # Normalize if requested
    if normalize:
        y = (y - y.mean(dim=0, keepdim=True)) / (y.std(dim=0, keepdim=True) + 1e-10)

    # Return shape
    if n_voxels == 1:
        return y.squeeze(1)
    else:
        return y


def generate_ar_noise(
    rho_coeffs: torch.Tensor | np.ndarray | list,
    n_timepoints: int,
    n_voxels: int = 1,
    normalize: bool = True,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
    burn_in: int | None = None,
) -> torch.Tensor:
    """
    Generate AR(p) temporally autocorrelated noise

    AR(p) model: y_t = ρ_1*y_{t-1} + ρ_2*y_{t-2} + ... + ρ_p*y_{t-p} + ε_t

    This generalizes AR(1) to higher orders. AR(2) or AR(3) is often sufficient
    to capture fMRI temporal autocorrelation more accurately than AR(1).

    Parameters
    ----------
    rho_coeffs : array-like, shape (p,)
        AR coefficients [ρ_1, ρ_2, ..., ρ_p]
        Example: [0.6, 0.2] for AR(2)
    n_timepoints : int
        Number of time points
    n_voxels : int, default=1
        Number of independent voxel time series
    normalize : bool, default=True
        Normalize to unit variance
    device : torch.device, optional
        Device for computation

    Returns
    -------
    noise : torch.Tensor
        (n_timepoints, n_voxels) AR(p) noise
        If n_voxels=1, returns (n_timepoints,)

    Notes
    -----
    No explicit stationarity check is performed. User should ensure
    the AR polynomial roots are outside the unit circle.

    Examples
    --------
    >>> # AR(2) with coefficients 0.6 and 0.2
    >>> noise = generate_ar_noise([0.6, 0.2], n_timepoints=300, n_voxels=100)

    References
    ----------
    - Box & Jenkins (1976): Time Series Analysis
    - Worsley & Friston (1995): AR models in fMRI
    """
    if device is None:
        device = get_device()

    # Convert coefficients to tensor
    if not torch.is_tensor(rho_coeffs):
        rho_coeffs = torch.tensor(rho_coeffs, dtype=torch.float32, device=device)
    else:
        rho_coeffs = rho_coeffs.to(device)

    p = len(rho_coeffs)

    # Burn-in, then discard. There is no one-line stationary seed for AR(p) as
    # there is for AR(1), and the alternative -- starting from zeros and writing
    # the first p samples afterwards -- is worse than it looks: those samples get
    # the innovation variance (1.0) while the stationary variance of e.g.
    # [0.5, 0.2] is 1.71, so every run opened with a variance step.
    n_burn = max(p, burn_in if burn_in is not None else 200)
    n_total = n_timepoints + n_burn

    # Initialize output
    y = torch.zeros(n_total, n_voxels, device=device)

    # Generate innovations (white noise)
    epsilon = torch.randn(n_total, n_voxels, device=device, generator=generator)
    y[:p] = epsilon[:p]

    # Sequential generation
    for t in range(p, n_total):
        # Dot product with past p values
        # y[t] = sum(rho_coeffs * y[t-p:t].flip())
        past_values = y[t - p : t].flip(0)  # Reverse to align with coefficients
        y[t] = (rho_coeffs.unsqueeze(1) * past_values).sum(dim=0) + epsilon[t]

    y = y[n_burn:]

    # Normalize if requested
    if normalize:
        y = (y - y.mean(dim=0, keepdim=True)) / (y.std(dim=0, keepdim=True) + 1e-10)

    # Return shape
    if n_voxels == 1:
        return y.squeeze(1)
    else:
        return y


def generate_arma_noise(
    ar_coeffs: torch.Tensor | np.ndarray | list,
    ma_coeffs: torch.Tensor | np.ndarray | list,
    n_timepoints: int,
    n_voxels: int = 1,
    normalize: bool = True,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
    burn_in: int | None = None,
) -> torch.Tensor:
    """
    Generate ARMA(p,q) temporally autocorrelated noise

    ARMA(p,q) model combines autoregressive and moving average:
        y_t = ρ_1*y_{t-1} + ... + ρ_p*y_{t-p} + ε_t + θ_1*ε_{t-1} + ... + θ_q*ε_{t-q}

    ARMA models are more flexible than pure AR models and can capture
    complex autocorrelation patterns with fewer parameters.
    ARMA(1,1) is often sufficient for fMRI.

    Parameters
    ----------
    ar_coeffs : array-like, shape (p,)
        AR coefficients [ρ_1, ρ_2, ..., ρ_p]
    ma_coeffs : array-like, shape (q,)
        MA coefficients [θ_1, θ_2, ..., θ_q]
    n_timepoints : int
        Number of time points
    n_voxels : int, default=1
        Number of independent voxel time series
    normalize : bool, default=True
        Normalize to unit variance
    device : torch.device, optional
        Device for computation

    Returns
    -------
    noise : torch.Tensor
        (n_timepoints, n_voxels) ARMA(p,q) noise
        If n_voxels=1, returns (n_timepoints,)

    Examples
    --------
    >>> # ARMA(1,1) with ρ=0.3 and θ=0.2
    >>> noise = generate_arma_noise([0.3], [0.2], n_timepoints=300, n_voxels=100)

    References
    ----------
    - Bullmore et al. (1996): Statistical methods for fMRI
    - Friston et al. (2000): Nonlinear ARMA models
    """
    if device is None:
        device = get_device()

    # Convert coefficients to tensors
    if not torch.is_tensor(ar_coeffs):
        ar_coeffs = torch.tensor(ar_coeffs, dtype=torch.float32, device=device)
    else:
        ar_coeffs = ar_coeffs.to(device)

    if not torch.is_tensor(ma_coeffs):
        ma_coeffs = torch.tensor(ma_coeffs, dtype=torch.float32, device=device)
    else:
        ma_coeffs = ma_coeffs.to(device)

    # Validate AR coefficients for stationarity
    if len(ar_coeffs) > 0:
        if torch.any(torch.abs(ar_coeffs) >= 1):
            raise ValueError("AR coefficients must be in (-1, 1) for stationarity.")

    p = len(ar_coeffs)
    q = len(ma_coeffs)
    max_order = max(p, q)

    # Burn in and discard, for the same reason as AR(p): seeding the first
    # max_order samples from the innovations alone gives them the innovation
    # variance rather than the process's, so the run opens with a variance step.
    n_burn = max(max_order, burn_in if burn_in is not None else 200)
    n_total = n_timepoints + n_burn

    # Initialize output and innovations
    y = torch.zeros(n_total, n_voxels, device=device)
    epsilon = torch.randn(n_total, n_voxels, device=device, generator=generator)
    y[:max_order] = epsilon[:max_order]

    # Sequential generation
    for t in range(max_order, n_total):
        # AR part: sum of past y values
        if p > 0:
            past_y = y[t - p : t].flip(0)
            ar_part = (ar_coeffs.unsqueeze(1) * past_y).sum(dim=0)
        else:
            ar_part = 0

        # MA part: sum of past innovations
        if q > 0:
            past_epsilon = epsilon[t - q : t].flip(0)
            ma_part = (ma_coeffs.unsqueeze(1) * past_epsilon).sum(dim=0)
        else:
            ma_part = 0

        # Combine: y_t = AR_part + MA_part + innovation_t
        y[t] = ar_part + ma_part + epsilon[t]

    y = y[n_burn:]

    # Normalize if requested
    if normalize:
        y = (y - y.mean(dim=0, keepdim=True)) / (y.std(dim=0, keepdim=True) + 1e-10)

    # Squeeze to (n_timepoints,) for a single voxel, matching both the docstring
    # and the AR(1)/AR(p) generators -- this one alone returned (n_timepoints, 1).
    if n_voxels == 1:
        return y.squeeze(1)
    return y


def estimate_noise_parameters_from_data(
    data: torch.Tensor | np.ndarray,
    design: torch.Tensor | np.ndarray | None = None,
    mask: torch.Tensor | np.ndarray | None = None,
    ar_order: int = 1,
    device: torch.device | None = None,
    time_axis: int | None = None,
) -> dict:
    """
    Estimate noise parameters from real fMRI data

    This extracts the noise characteristics of YOUR specific scanner/acquisition,
    which can then be used to generate realistic simulations matching your data.

    CRITICAL for realistic simulations: Don't guess noise parameters - measure them!

    Parameters
    ----------
    data : array-like, shape (..., n_timepoints) or (n_timepoints, n_voxels)
        Real fMRI data (can be 1D timeseries, 2D, or 4D volume)
    design : array-like, optional
        Design matrix for GLM. If None, only detrends (removes mean/linear trend)
    mask : array-like, optional
        Brain mask (for 3D/4D data). If None, uses all voxels
    ar_order : int, default=1
        Order of AR model to fit (1 for AR(1), 2 for AR(2), etc.)
    device : torch.device, optional
        Device for computation

    Returns
    -------
    params : dict
        'ar_coefficients': list of p floats, mean AR coefficients across voxels
        'ar_coefficients_mean': same list, under the name the examples use
        'ar_coefficients_std': list of p floats, std across voxels
        'ar_coefficients_all': list per sampled voxel (for the distribution)
        'sfnr' / 'sfnr_mean': float, mean SFNR across voxels
        'sfnr_std': float, std SFNR across voxels
        'noise_std': float, residual standard deviation
        'n_voxels': int, number of voxels analyzed
        'n_timepoints': int, number of timepoints

    Examples
    --------
    >>> # Extract noise from pilot data
    >>> import nibabel as nib
    >>> data = nib.load('pilot_run.nii.gz').get_fdata()  # (x, y, z, t)
    >>> params = estimate_noise_parameters_from_data(data)
    >>> print(f"Scanner AR(1) = {params['ar_coefficients_mean']:.3f}")
    >>>
    >>> # Use extracted parameters for simulation
    >>> noise = generate_ar1_noise(
    ...     rho=params['ar_coefficients_mean'],
    ...     n_timepoints=300,
    ...     n_voxels=1000
    ... )

    Notes
    -----
    This implements a simplified version of AFNI's 3dREMLfit parameter estimation:
    1. Fit GLM (or detrend if no design)
    2. Extract residuals
    3. Estimate AR coefficients from residual autocorrelation
    4. Compute SFNR from signal/noise

    For AFNI-style REML estimation, use the GLS functions in metrics_empirical.py
    """
    if device is None:
        device = get_device()

    # Convert to tensor
    if not torch.is_tensor(data):
        data = torch.tensor(data, dtype=torch.float32, device=device)
    else:
        data = data.to(device)

    # Handle different data shapes
    _original_shape = data.shape
    if data.ndim == 1:
        # Single timeseries
        data = data.unsqueeze(1)  # (n_timepoints, 1)
    elif data.ndim == 2:
        # A 2D array is genuinely ambiguous, and the old rule -- "the longer axis
        # is time" -- was backwards for fMRI, where voxels outnumber timepoints by
        # orders of magnitude. Handed the (n_voxels, n_timepoints) matrix every ffs
        # tool stores, it read voxels as time and returned rho ~ 0 for a planted
        # 0.45, silently. Time is now the SHORTER axis unless told otherwise.
        if time_axis is not None:
            if time_axis not in (0, 1):
                raise ValueError(f"time_axis must be 0 or 1 for 2D input, got {time_axis}")
            if time_axis == 1:
                data = data.T
        else:
            design_rows = None
            if design is not None:
                design_rows = (
                    design.shape[0] if torch.is_tensor(design) else np.asarray(design).shape[0]
                )
            mask_elements = None
            if mask is not None:
                mask_elements = (
                    int(mask.numel()) if torch.is_tensor(mask) else int(np.asarray(mask).size)
                )
            if _infer_time_axis(data, design_rows, mask_elements) == 1:
                data = data.T
    elif data.ndim >= 3:
        # 3D or 4D volume - flatten spatial dimensions
        # Assume time is last dimension
        n_timepoints = data.shape[-1]
        data = data.reshape(-1, n_timepoints).T  # (n_timepoints, n_voxels)

    n_timepoints, n_voxels = data.shape

    # Apply mask if provided
    if mask is not None:
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
        else:
            mask = mask.to(device)
        mask_flat = mask.flatten()
        data = data[:, mask_flat]
        n_voxels = data.shape[1]

    # Compute SFNR first (before residuals)
    sfnr_per_voxel = data.mean(dim=0) / (data.std(dim=0) + 1e-10)
    sfnr_mean = sfnr_per_voxel.mean().item()
    sfnr_std = sfnr_per_voxel.std().item()

    # Get residuals
    if design is not None:
        # Fit GLM and extract residuals
        if not torch.is_tensor(design):
            design = torch.tensor(design, dtype=torch.float32, device=device)
        else:
            design = design.to(device)

        # OLS: β = (X'X)^(-1) X'Y
        XtX = design.T @ design
        XtX_reg = XtX + 1e-6 * torch.eye(XtX.shape[0], device=device)
        XtY = design.T @ data
        betas = torch.linalg.solve(XtX_reg, XtY)
        residuals = data - design @ betas
    else:
        # Just detrend (remove mean and linear trend)
        t = torch.arange(n_timepoints, dtype=torch.float32, device=device)
        t_norm = (t - t.mean()) / (t.std() + 1e-10)

        # Fit mean + linear trend per voxel
        ones = torch.ones(n_timepoints, device=device)
        design_detrend = torch.stack([ones, t_norm], dim=1)

        XtX = design_detrend.T @ design_detrend
        XtY = design_detrend.T @ data
        betas = torch.linalg.solve(XtX, XtY)
        residuals = data - design_detrend @ betas

    # Estimate AR coefficients per voxel using Yule-Walker equations
    ar_coeffs_per_voxel = []

    for v in range(min(n_voxels, 1000)):  # Sample up to 1000 voxels for speed
        resid_v = residuals[:, v]

        if ar_order == 1:
            # Simple AR(1): correlation between y[t] and y[t-1]
            rho = torch.corrcoef(torch.stack([resid_v[:-1], resid_v[1:]]))[0, 1]
            rho = torch.clip(rho, -0.99, 0.99)  # Ensure stationarity
            ar_coeffs_per_voxel.append([rho.item()])
        else:
            # General AR(p): Yule-Walker equations
            # R * φ = r, where R is autocorrelation matrix, r is autocorrelation vector

            # Compute autocorrelations
            acf = []
            for lag in range(ar_order + 1):
                if lag == 0:
                    acf.append(1.0)
                else:
                    corr = torch.corrcoef(torch.stack([resid_v[:-lag], resid_v[lag:]]))[0, 1]
                    acf.append(corr.item())

            # Build Toeplitz matrix R from acf[0:p]
            R = torch.tensor(
                [acf[abs(i - j)] for i in range(ar_order) for j in range(ar_order)], device=device
            ).reshape(ar_order, ar_order)
            r = torch.tensor(acf[1 : ar_order + 1], device=device)

            # Solve R * φ = r
            try:
                phi = torch.linalg.solve(R, r)
                ar_coeffs_per_voxel.append(phi.cpu().numpy().tolist())
            except Exception:
                # Singular matrix - use zeros
                ar_coeffs_per_voxel.append([0.0] * ar_order)

    # Aggregate AR coefficients
    ar_coeffs_array = np.array(ar_coeffs_per_voxel)  # (n_voxels_sampled, ar_order)
    ar_coeffs_mean = ar_coeffs_array.mean(axis=0).tolist()
    ar_coeffs_std = ar_coeffs_array.std(axis=0).tolist()

    # Compute noise std from residuals
    noise_std = residuals.std().item()

    return {
        "ar_coefficients": ar_coeffs_mean,  # Mean coefficients across voxels
        # Documented name for the same thing; the docstring's own example used
        # 'ar_coefficients_mean' / 'sfnr_mean', neither of which was ever returned.
        "ar_coefficients_mean": ar_coeffs_mean,
        "ar_coefficients_std": ar_coeffs_std,  # Std of coefficients
        "ar_coefficients_all": ar_coeffs_per_voxel,  # All voxels (for distribution)
        "sfnr": sfnr_mean,
        "sfnr_mean": sfnr_mean,
        "sfnr_std": sfnr_std,
        "sfnr_all": sfnr_per_voxel.cpu().numpy().tolist(),
        "noise_std": noise_std,
        "n_voxels": n_voxels,
        "n_timepoints": n_timepoints,
        "summary": f"AR({ar_order}) = {ar_coeffs_mean}, SFNR = {sfnr_mean:.1f} ± {sfnr_std:.1f}",
    }


def estimate_sfnr(
    data: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray | None = None,
    device: torch.device | None = None,
    detrend: bool = True,
) -> dict:
    """
    Estimate temporal Signal Fluctuation to Noise Ratio (SFNR)

    SFNR = mean(signal) / std(detrended residual) across time, per voxel.
    Set ``detrend=False`` for the raw ratio, which drift will depress.

    This is the standard fMRI quality metric. Typical values:
    - Good 3T: SFNR = 150-200
    - Poor quality: SFNR = 50-100
    - 7T: SFNR = 100-150 (lower due to higher physiological noise)

    Parameters
    ----------
    data : array-like
        fMRI timeseries (any shape, time assumed to be last dimension)
    mask : array-like, optional
        Brain mask
    device : torch.device, optional
        Device for computation

    Returns
    -------
    sfnr_dict : dict
        'sfnr_mean': float, mean SFNR across voxels
        'sfnr_median': float, median SFNR
        'sfnr_std': float, std SFNR across voxels
        'sfnr_map': array, SFNR per voxel (same spatial shape as input)

    Notes
    -----
    SFNR is computed WITHOUT any GLM fitting - it's a pure data quality metric.
    For residual-based metrics (after GLM), use estimate_noise_parameters_from_data().

    References
    ----------
    - Friedman & Glover (2006): Reducing interscanner variability
    """
    if device is None:
        device = get_device()

    if not torch.is_tensor(data):
        data = torch.tensor(data, dtype=torch.float32, device=device)
    else:
        data = data.to(device)

    # Get original spatial shape
    original_shape = data.shape[:-1]  # All except time
    n_timepoints = data.shape[-1]

    # Flatten to (n_voxels, n_timepoints)
    data_flat = data.reshape(-1, n_timepoints)

    # Apply mask
    if mask is not None:
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
        else:
            mask = mask.to(device)
        mask_flat = mask.flatten()
        data_flat = data_flat[mask_flat]

    # Compute SFNR per voxel. Friedman & Glover take the fluctuation as the std
    # of the *detrended* residual: without that, scanner drift lands in the
    # denominator and depresses SFNR (measured: a planted 150 read 135 with drift
    # at amplitude 0.5), which is not what the published reference values mean.
    if detrend:
        n_t = data_flat.shape[1]
        t = torch.arange(n_t, dtype=data_flat.dtype, device=device)
        t = (t - t.mean()) / (t.std() + 1e-10)
        basis = torch.stack([torch.ones_like(t), t, t**2], dim=1)  # (n_t, 3)
        pinv = torch.linalg.pinv(basis)
        fluctuation = data_flat - (data_flat @ pinv.T) @ basis.T
        std_signal = fluctuation.std(dim=1)
    else:
        std_signal = data_flat.std(dim=1)

    mean_signal = data_flat.mean(dim=1)
    sfnr = mean_signal / (std_signal + 1e-10)

    # Summary statistics
    sfnr_mean = sfnr.mean().item()
    sfnr_median = sfnr.median().item()
    sfnr_std = sfnr.std().item()

    # Reconstruct spatial map
    if mask is not None:
        sfnr_map = torch.zeros(original_shape, device=device)
        sfnr_map_flat = sfnr_map.flatten()
        sfnr_map_flat[mask_flat] = sfnr
        sfnr_map = sfnr_map_flat.reshape(original_shape)
    else:
        sfnr_map = sfnr.reshape(original_shape)

    return {
        "sfnr_mean": sfnr_mean,
        "sfnr_median": sfnr_median,
        "sfnr_std": sfnr_std,
        "sfnr_map": sfnr_map,
        "summary": f"SFNR = {sfnr_mean:.1f} ± {sfnr_std:.1f} (median={sfnr_median:.1f})",
    }
