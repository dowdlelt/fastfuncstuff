"""GPU-accelerated slice-timing correction (matches 3dTshift).

Shifts each slice's time series so all slices appear to be acquired at
the same time within each TR.  Supports Fourier, polynomial (linear
through heptic), and windowed-sinc (wsinc5, wsinc9) interpolation.

Algorithm per-voxel (following AFNI's 3dTshift):
    1. Record original range
    2. Linear detrend (remove mean + slope)
    3. Record detrended range
    4. Temporal shift via chosen kernel
    5. Clip to detrended range
    6. Retrend (add mean + slope back)
    7. Clip to original range

Reference: https://github.com/afni/afni/blob/master/src/3dTshift.c
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import Tensor
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Slice timing loading
# ---------------------------------------------------------------------------

def load_slice_timing(path: str | Path) -> list[float]:
    """Load slice timing from a text file or BIDS sidecar JSON.

    Text file: one number per line (seconds), one per slice.
    JSON: expects a ``SliceTiming`` field (list of floats).

    In both cases, entry *i* gives the acquisition time offset (in seconds)
    for spatial slice *i* (slice 0 = first slice in the file = typically
    the most-inferior slice).
    """
    p = Path(path)
    text = p.read_text().strip()

    if p.suffix == ".json":
        data = json.loads(text)
        if "SliceTiming" not in data:
            raise ValueError(f"JSON file has no 'SliceTiming' field: {p}")
        return [float(v) for v in data["SliceTiming"]]

    # Plain text: one value per line
    vals = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        vals.append(float(line))
    return vals


# ---------------------------------------------------------------------------
# Linear detrend / retrend  (batched, works on any (N, nt) tensor)
# ---------------------------------------------------------------------------

def _linear_detrend(ts: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Remove mean + linear trend from (n_vox, nt) time series."""
    nt = ts.shape[1]
    t = torch.arange(nt, device=ts.device, dtype=ts.dtype)
    t_mean = (nt - 1) / 2.0
    t_centered = t - t_mean
    t_var = (t_centered * t_centered).sum()

    means = ts.mean(dim=1)
    slopes = ((ts - means[:, None]) * t_centered[None, :]).sum(dim=1) / t_var
    intercepts = means

    trend = intercepts[:, None] + slopes[:, None] * t_centered[None, :]
    detrended = ts - trend
    return detrended, intercepts, slopes


def _linear_retrend(ts: Tensor, intercepts: Tensor, slopes: Tensor) -> Tensor:
    """Add mean + linear trend back."""
    nt = ts.shape[1]
    t = torch.arange(nt, device=ts.device, dtype=ts.dtype)
    t_centered = t - (nt - 1) / 2.0
    return ts + intercepts[:, None] + slopes[:, None] * t_centered[None, :]


# ---------------------------------------------------------------------------
# Interpolation kernels — all operate on (n_vox, nt) batches
# ---------------------------------------------------------------------------

def _next_fft_size(n: int) -> int:
    """Next integer >= n that is a product of 2, 3, and 5 only."""
    while True:
        m = n
        while m % 2 == 0:
            m //= 2
        while m % 3 == 0:
            m //= 3
        while m % 5 == 0:
            m //= 5
        if m == 1:
            return n
        n += 1


def _shift_fourier(ts: Tensor, frac_shift: float) -> Tensor:
    """Shift via Fourier phase rotation.  Promoted to float64 internally."""
    orig_dtype = ts.dtype
    ts = ts.to(torch.float64)

    nt = ts.shape[1]
    nup = _next_fft_size(nt + 4)

    # Zero-pad
    padded = torch.zeros(ts.shape[0], nup, device=ts.device, dtype=torch.float64)
    padded[:, :nt] = ts

    # Forward FFT
    spec = torch.fft.rfft(padded, dim=1)

    # Phase shift: exp(-i * 2π * k * frac_shift / nup)
    nfreq = spec.shape[1]
    k = torch.arange(nfreq, device=ts.device, dtype=torch.float64)
    phase = -2.0 * math.pi * frac_shift * k / nup
    shift_kernel = torch.complex(torch.cos(phase), torch.sin(phase))
    spec = spec * shift_kernel[None, :]

    # Zero Nyquist imaginary part
    spec[:, -1] = spec[:, -1].real.to(spec.dtype)

    # Inverse FFT
    shifted = torch.fft.irfft(spec, n=nup, dim=1)
    return shifted[:, :nt].to(orig_dtype)


def _shift_polynomial(ts: Tensor, frac_shift: float, order: int) -> Tensor:
    """Shift using Lagrange polynomial interpolation.  Promoted to float64.

    order: 1=linear, 3=cubic, 5=quintic, 7=heptic
    """
    orig_dtype = ts.dtype
    ts = ts.to(torch.float64)
    nt = ts.shape[1]

    af = -frac_shift
    ia = int(math.floor(af))
    aa = af - ia  # fractional part in [0, 1)

    half = order // 2
    n_pts = order + 1

    # Lagrange weights (float64)
    nodes = [float(-half + j) for j in range(n_pts)]
    weights_list = []
    for j in range(n_pts):
        w = 1.0
        for k_idx in range(n_pts):
            if k_idx != j:
                w *= (aa - nodes[k_idx]) / (nodes[j] - nodes[k_idx])
        weights_list.append(w)
    weights = torch.tensor(weights_list, dtype=torch.float64, device=ts.device)

    # Build output: weighted sum of shifted versions
    out = torch.zeros_like(ts)
    time_idx = torch.arange(nt, device=ts.device)
    for j in range(n_pts):
        offset = ia + int(nodes[j])
        src_idx = time_idx + offset
        valid = (src_idx >= 0) & (src_idx < nt)
        src_clamped = src_idx.clamp(0, nt - 1)
        gathered = ts[:, src_clamped]
        gathered[:, ~valid] = 0.0
        out += weights[j] * gathered

    return out.to(orig_dtype)


def _sinc(x: Tensor) -> Tensor:
    """Normalized sinc: sin(pi*x) / (pi*x), Taylor near zero."""
    result = torch.ones_like(x)
    ax = x.abs()
    big = ax >= 0.01
    px = math.pi * x[big]
    result[big] = torch.sin(px) / px
    small = (~big) & (ax > 0)
    result[small] = 1.0 - 1.6449341 * x[small] * x[small]
    return result


def _m3_window(x: Tensor) -> Tensor:
    """Minimum-sidelobe 3-term window.  Zero for |x| > 1."""
    result = (0.4243801
              + 0.4973406 * torch.cos(math.pi * x)
              + 0.0782793 * torch.cos(2.0 * math.pi * x))
    result[x.abs() > 1.0] = 0.0
    return result


def _shift_wsinc(ts: Tensor, frac_shift: float, half_width: int) -> Tensor:
    """Windowed-sinc shift.  Promoted to float64.  half_width=5 or 9."""
    orig_dtype = ts.dtype
    ts = ts.to(torch.float64)
    nt = ts.shape[1]

    af = -frac_shift
    ia = int(math.floor(af))
    aa = af - ia

    n_pts = 2 * half_width
    offsets = torch.arange(-(half_width - 1), half_width + 1,
                           dtype=torch.float64, device=ts.device)
    dist = aa - offsets
    weights = _sinc(dist) * _m3_window(dist / half_width)
    wsum = weights.sum()
    if wsum.abs() > 1e-10:
        weights = weights / wsum

    out = torch.zeros_like(ts)
    time_idx = torch.arange(nt, device=ts.device)
    for j in range(n_pts):
        offset = ia + int(offsets[j].item())
        src_idx = time_idx + offset
        valid = (src_idx >= 0) & (src_idx < nt)
        src_clamped = src_idx.clamp(0, nt - 1)
        gathered = ts[:, src_clamped]
        gathered[:, ~valid] = 0.0
        out += weights[j] * gathered

    return out.to(orig_dtype)


# ---------------------------------------------------------------------------
# Shift dispatcher
# ---------------------------------------------------------------------------

def shift_timeseries(ts: Tensor, frac_shift: float, method: str = "fourier") -> Tensor:
    """Shift a batch of time series by frac_shift samples.

    Parameters
    ----------
    ts : (n_vox, nt)
    frac_shift : float — shift in units of TR (fractional samples)
    method : str — interpolation method

    Returns
    -------
    (n_vox, nt) shifted time series
    """
    if abs(frac_shift) < 1e-6:
        return ts.clone()

    method = method.lower()
    if method == "fourier":
        return _shift_fourier(ts, frac_shift)
    elif method == "linear":
        return _shift_polynomial(ts, frac_shift, order=1)
    elif method == "cubic":
        return _shift_polynomial(ts, frac_shift, order=3)
    elif method == "quintic":
        return _shift_polynomial(ts, frac_shift, order=5)
    elif method == "heptic":
        return _shift_polynomial(ts, frac_shift, order=7)
    elif method == "wsinc5":
        return _shift_wsinc(ts, frac_shift, half_width=5)
    elif method == "wsinc9":
        return _shift_wsinc(ts, frac_shift, half_width=9)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def slicetime_correct(
    vol4d: Tensor,
    slice_timing: list[float],
    tr: float,
    tzero: float | None = None,
    method: str = "fourier",
    ignore: int = 0,
    device: torch.device | None = None,
    verbose: bool = False,
) -> Tensor:
    """Apply slice-timing correction to a 4D volume.

    Parameters
    ----------
    vol4d : (nt, nz, ny, nx) float tensor
        Input 4D time series in our internal convention.
    slice_timing : list of float
        Per-slice acquisition time offsets in seconds.  Entry *i* gives the
        time offset for spatial slice *i* (slice 0 = most inferior / first
        in file).  Length must equal nz.
    tr : float
        Repetition time in seconds.
    tzero : float, optional
        Target time to align all slices to (seconds within TR).
        Default: mean of slice_timing (same as 3dTshift default).
    method : str
        Interpolation: fourier, linear, cubic, quintic, heptic, wsinc5, wsinc9.
    ignore : int
        Number of initial volumes to skip (pass through unchanged).
    device : torch.device, optional
        Device for computation.
    verbose : bool
        Print progress info.

    Returns
    -------
    (nt, nz, ny, nx) float tensor — corrected 4D volume.
    """
    if device is not None:
        vol4d = vol4d.to(device)

    nt, nz, ny, nx = vol4d.shape
    n_vox_per_slice = ny * nx

    if len(slice_timing) != nz:
        raise ValueError(
            f"slice_timing has {len(slice_timing)} entries but volume has "
            f"{nz} slices"
        )

    if tzero is None:
        tzero = sum(slice_timing) / len(slice_timing)

    if verbose:
        print(f"  slicetime: {nz} slices, TR={tr:.4f}s, tzero={tzero:.4f}s, "
              f"method={method}, device={vol4d.device}")

    nt_shift = nt - ignore
    out = vol4d.clone()

    # Group slices by fractional shift → batch all slices with same shift
    shift_groups: dict[float, list[int]] = {}
    for kk in range(nz):
        fshift = -(tzero - slice_timing[kk]) / tr
        fshift_key = round(fshift, 10)
        shift_groups.setdefault(fshift_key, []).append(kk)

    n_skip = sum(len(ss) for f, ss in shift_groups.items() if abs(f) < 1e-6)
    groups_to_shift = {f: ss for f, ss in shift_groups.items() if abs(f) >= 1e-6}

    pbar = tqdm(
        total=nz - n_skip,
        desc="  Slicetime",
        unit="slices",
        disable=not verbose,
        leave=False,
    )

    for fshift, slices in groups_to_shift.items():
        n_slices = len(slices)

        # Gather all voxels from all slices in this group → (n_slices * ny*nx, nt_shift)
        slabs = vol4d[ignore:, slices, :, :]  # (nt_shift, n_slices, ny, nx)
        ts = slabs.permute(1, 2, 3, 0).reshape(-1, nt_shift).contiguous()

        # 1. Record original range
        orig_min = ts.min(dim=1).values
        orig_max = ts.max(dim=1).values

        # 2. Detrend (float64 for accumulation precision)
        ts_f64 = ts.to(torch.float64)
        ts_dt, intercepts, slopes = _linear_detrend(ts_f64)
        ts_dt = ts_dt.to(ts.dtype)

        # 3. Record detrended range
        dt_min = ts_dt.min(dim=1).values
        dt_max = ts_dt.max(dim=1).values

        # 4. Shift — one call for all voxels in this shift group
        #    (shift functions promote to float64 internally)
        ts_shifted = shift_timeseries(ts_dt, fshift, method=method)

        # 5. Clip to detrended range
        ts_shifted = ts_shifted.clamp(min=dt_min[:, None], max=dt_max[:, None])

        # 6. Retrend (float64 for accumulation precision)
        ts_shifted = _linear_retrend(
            ts_shifted.to(torch.float64),
            intercepts, slopes,
        ).to(ts.dtype)

        # 7. Clip to original range
        ts_shifted = ts_shifted.clamp(min=orig_min[:, None], max=orig_max[:, None])

        # Scatter back: (n_slices*ny*nx, nt_shift) → (nt_shift, n_slices, ny, nx)
        ts_back = ts_shifted.reshape(n_slices, ny, nx, nt_shift).permute(3, 0, 1, 2)
        out[ignore:, slices, :, :] = ts_back

        pbar.update(n_slices)

    pbar.close()

    if verbose:
        print(f"  slicetime: {nz} slices done, {len(shift_groups)} unique shifts "
              f"({n_skip} at tzero, skipped)")

    return out


def temporal_resample(
    vol4d: Tensor,
    tr_old: float,
    tr_new: float,
    method: str = "cubic",
    device: torch.device | None = None,
    verbose: bool = False,
) -> Tensor:
    """Resample a 4D volume to a new TR grid via temporal interpolation.

    After slice-timing correction all slices share the same temporal grid
    (at ``tr_old``).  This function resamples onto a new grid with spacing
    ``tr_new``, which is typically shorter (upsampling) so that event onsets
    align to TR boundaries — a requirement for GLMsingle-style analysis.

    Parameters
    ----------
    vol4d : (nt_old, nz, ny, nx) float tensor
    tr_old : float
        Current TR in seconds.
    tr_new : float
        Desired output TR in seconds.
    method : str
        Interpolation: 'linear', 'cubic' (default).
    device : torch.device, optional
    verbose : bool

    Returns
    -------
    (nt_new, nz, ny, nx) float tensor at the new TR.
    """
    if device is not None:
        vol4d = vol4d.to(device)

    nt_old, nz, ny, nx = vol4d.shape

    # Total duration = (nt_old - 1) * tr_old  (from first to last sample)
    total_duration = (nt_old - 1) * tr_old
    nt_new = int(total_duration / tr_new) + 1

    if verbose:
        print(f"  resample: {nt_old} vols @ {tr_old:.4f}s -> {nt_new} vols @ {tr_new:.4f}s")
        print(f"  duration: {total_duration:.2f}s, method: {method}")

    # Build old and new time grids
    t_old = torch.arange(nt_old, device=vol4d.device, dtype=torch.float64) * tr_old
    t_new = torch.arange(nt_new, device=vol4d.device, dtype=torch.float64) * tr_new

    # Clamp new times to old range (avoid extrapolation)
    t_new = t_new.clamp(max=t_old[-1].item())

    # Process slice-by-slice to keep memory manageable
    out = torch.zeros(nt_new, nz, ny, nx, device=vol4d.device, dtype=vol4d.dtype)

    for kk in range(nz):
        # (ny*nx, nt_old) — each voxel's time series as a row
        ts = vol4d[:, kk, :, :].reshape(nt_old, -1).T.to(torch.float64)
        n_vox = ts.shape[0]

        if method == "linear":
            # Find insertion indices for t_new in t_old
            idx = torch.searchsorted(t_old, t_new).clamp(1, nt_old - 1)
            t0 = t_old[idx - 1]
            t1 = t_old[idx]
            w = ((t_new - t0) / (t1 - t0)).unsqueeze(0)  # (1, nt_new)
            v0 = ts[:, idx - 1]  # (n_vox, nt_new)
            v1 = ts[:, idx]
            interp = v0 * (1 - w) + v1 * w

        elif method == "cubic":
            # Cubic Hermite (Catmull-Rom) spline interpolation
            idx = torch.searchsorted(t_old, t_new).clamp(1, nt_old - 1)
            t0 = t_old[idx - 1]
            t1 = t_old[idx]
            frac = (t_new - t0) / (t1 - t0)  # (nt_new,)
            f = frac.unsqueeze(0)  # (1, nt_new)
            f2 = f * f
            f3 = f2 * f

            # Catmull-Rom basis
            h00 = 2 * f3 - 3 * f2 + 1
            h10 = f3 - 2 * f2 + f
            h01 = -2 * f3 + 3 * f2
            h11 = f3 - f2

            # Values at grid points
            p0 = ts[:, idx - 1]
            p1 = ts[:, idx]

            # Tangents (finite difference, clamped at boundaries)
            im1 = (idx - 2).clamp(0)
            ip1 = idx.clamp(max=nt_old - 1)
            m0 = (ts[:, idx] - ts[:, im1]) / 2.0
            m1 = (ts[:, ip1] - ts[:, idx - 1]) / 2.0

            # Scale tangents by the interval length (for non-uniform grids, though
            # here we have a uniform grid so dt = tr_old for all intervals)
            interp = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

        else:
            raise ValueError(f"Unknown resample method: {method}. Use 'linear' or 'cubic'.")

        # (n_vox, nt_new) -> (nt_new, ny, nx)
        out[:, kk, :, :] = interp.to(vol4d.dtype).T.reshape(nt_new, ny, nx)

    if verbose:
        print(f"  resample: done ({nt_new} output volumes)")

    return out
