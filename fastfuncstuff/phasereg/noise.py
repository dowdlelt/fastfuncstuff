"""Variance ratio estimation for Deming regression.

The Deming regression slope depends on phi = Var(eps_magnitude) / Var(eps_phase).
This module provides two estimation strategies:

1. **FFT mode** (default, works for task and resting-state data):
   Estimate noise variance from power spectral density outside the typical
   neural signal band.  High-frequency power is dominated by thermal and
   physiological noise, giving clean per-voxel noise estimates.
   Matches the approach in phaseprep (Liem/Stanley) and Curtis 2014:
   high-pass filter the signal and take time-domain std.

2. **Residual mode** (requires task design matrix):
   Fit a GLM (task + nuisance) to both magnitude and phase, then compute
   phi from the residual variances.  More principled when a good task model
   is available.
"""

from __future__ import annotations

import torch


def _psd_noise_variance(
    data: torch.Tensor,
    tr: float,
    freq_lo: float = 0.1,
    freq_hi: float | None = None,
) -> torch.Tensor:
    """Estimate per-voxel noise variance from out-of-band PSD.

    Uses the same method as phaseprep and Curtis 2014: zero out non-noise
    frequencies in the FFT, IFFT back, and take the time-domain variance.
    This avoids PSD normalisation pitfalls.

    Parameters
    ----------
    data : Tensor, shape (n_timepoints, n_voxels)
        Zero-mean time series.
    tr : float
        Repetition time in seconds.
    freq_lo : float
        Lower bound of noise frequency band (Hz).
    freq_hi : float or None
        Upper bound of noise frequency band (Hz).  None = Nyquist.

    Returns
    -------
    variance : Tensor (n_voxels,)
        Estimated noise variance per voxel.
    """
    n_tp = data.shape[0]
    nyquist = 0.5 / tr

    if freq_hi is None or freq_hi > nyquist:
        freq_hi = nyquist

    spectrum = torch.fft.rfft(data, dim=0)
    freqs = torch.fft.rfftfreq(n_tp, d=tr, device=data.device)

    mask = (freqs >= freq_lo) & (freqs <= freq_hi)
    n_bins = mask.sum().item()

    if n_bins < 2:
        return data.var(dim=0)

    filtered = torch.zeros_like(spectrum)
    filtered[mask] = spectrum[mask]
    noise_ts = torch.fft.irfft(filtered, n=n_tp, dim=0)

    return noise_ts.var(dim=0)


def estimate_variance_ratio(
    magnitude: torch.Tensor,
    phase: torch.Tensor,
    tr: float,
    method: str = "fft",
    design: torch.Tensor | None = None,
    freq_range: tuple[float, float | None] = (0.1, None),
) -> torch.Tensor:
    """Estimate phi = Var(eps_magnitude) / Var(eps_phase) per voxel.

    Parameters
    ----------
    magnitude : Tensor, shape (n_timepoints, n_voxels)
        Magnitude time series (after detrending / task removal).
    phase : Tensor, shape (n_timepoints, n_voxels)
        Phase time series (after detrending / task removal).
    tr : float
        Repetition time in seconds.
    method : {"fft", "residual"}
        - "fft": estimate from out-of-band spectral power (Curtis 2014).
        - "residual": estimate from temporal variance of the inputs
          directly (caller should pass GLM residuals).
    design : Tensor or None
        If method="residual" and the inputs are NOT already residualised,
        pass the design matrix here and this function will project it out.
        Shape (n_timepoints, n_regressors).
    freq_range : tuple (lo_hz, hi_hz)
        Frequency range for FFT-based noise estimation.  hi_hz=None means
        Nyquist.

    Returns
    -------
    phi : Tensor (n_voxels,)
        Variance ratio Var(eps_mag)/Var(eps_phase). NOT artificially clamped
        on the upper end: at fMRI scales (raw scanner-unit mag vs radian
        phase) phi naturally lands in 10^4–10^6, and Deming's near-OLS
        approximation only holds when phi >> SYY/SXX. Clamping low (e.g. 100)
        plants Deming in an "orthogonal-ish" regime that explodes the slope
        in low-phase-variance voxels and forces R² negative, so we let phi
        track the actual noise ratio. A tiny floor guards against div-by-zero.
    """
    if method == "fft":
        mag_centered = magnitude - magnitude.mean(dim=0)
        pha_centered = phase - phase.mean(dim=0)
        var_mag = _psd_noise_variance(mag_centered, tr, freq_range[0], freq_range[1])
        var_pha = _psd_noise_variance(pha_centered, tr, freq_range[0], freq_range[1])

    elif method == "residual":
        if design is not None:
            Q, _ = torch.linalg.qr(design)
            mag_resid = magnitude - Q @ (Q.T @ magnitude)
            pha_resid = phase - Q @ (Q.T @ phase)
        else:
            mag_resid = magnitude
            pha_resid = phase
        var_mag = mag_resid.var(dim=0)
        var_pha = pha_resid.var(dim=0)

    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'fft' or 'residual'.")

    phi = torch.where(
        var_pha > 1e-30,
        var_mag / var_pha,
        torch.ones_like(var_mag),
    )
    phi = phi.clamp(min=1e-6)

    return phi
