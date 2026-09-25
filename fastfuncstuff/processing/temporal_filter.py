"""Temporal filters for (voxels, time) data: FFT low/high/band-pass and a
centred moving average, applied run by run.

The FFT pass-band edges are raised cosines (``transition_width`` of the cutoff)
to avoid a brick wall's ringing. ``symmetric=True`` filters each run's mirror
extension ``[x, x[::-1]]`` rather than ``x`` itself, so the implicit periodic
wrap joins the run's end to itself instead of to its start: no step at the
run edges for the filter to smear. A high- or band-pass removes the mean; a
low-pass and the moving average keep it.

For a FIR/TENT deconvolution only the DATA are filtered, never the design: the
knots are the estimate, and the fitted curve comes out as the true response
blurred by the filter. Assumed-shape models (a canonical HRF, SPMG) are the
case where the design is filtered too.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from tqdm.auto import tqdm

from fastfuncstuff.memory import estimate_chunk_size

TEMPORAL_FILTER_KINDS = ("lowpass", "highpass", "bandpass", "movavg")


@dataclass(frozen=True)
class TemporalFilter:
    kind: str
    low_hz: float | None = None  # high-pass edge
    high_hz: float | None = None  # low-pass edge
    window: int | None = None  # moving-average length, samples (odd)

    def describe(self) -> str:
        if self.kind == "movavg":
            return f"moving average over {self.window} samples"
        if self.kind == "lowpass":
            return f"low-pass {self.high_hz:g} Hz"
        if self.kind == "highpass":
            return f"high-pass {self.low_hz:g} Hz"
        return f"band-pass {self.low_hz:g}-{self.high_hz:g} Hz"


def parse_temporal_filter(spec: str) -> TemporalFilter:
    """``lowpass:0.15`` | ``highpass:0.01`` | ``bandpass:0.01,0.1`` | ``movavg:7``."""
    kind, _, value = spec.strip().lower().partition(":")
    try:
        nums = [float(v) for v in value.split(",")] if value else []
    except ValueError:
        nums = []
    if kind in ("lowpass", "highpass") and len(nums) == 1 and nums[0] > 0:
        hz = nums[0]
        return TemporalFilter(
            kind,
            low_hz=hz if kind == "highpass" else None,
            high_hz=hz if kind == "lowpass" else None,
        )
    if kind == "bandpass" and len(nums) == 2 and 0 < nums[0] < nums[1]:
        return TemporalFilter(kind, low_hz=nums[0], high_hz=nums[1])
    if kind == "movavg" and len(nums) == 1 and nums[0] >= 1 and nums[0] % 2 == 1:
        return TemporalFilter(kind, window=int(nums[0]))
    raise ValueError(
        "temporal filter must be lowpass:HZ, highpass:HZ, bandpass:LO,HI (LO < HI) or "
        f"movavg:N (odd N, samples), got {spec!r}"
    )


def fft_gain(
    freqs: Tensor,
    low_hz: float | None,
    high_hz: float | None,
    transition_width: float = 0.25,
) -> Tensor:
    """Pass-band gain at ``freqs``: 0 below ``low_hz*(1-tw)``, 1 from ``low_hz``
    to ``high_hz``, 0 above ``high_hz*(1+tw)``, raised-cosine in between.
    DC is zeroed whenever there is a high-pass edge."""
    gain = torch.ones_like(freqs)
    if low_hz is not None and low_hz > 0:
        tw = low_hz * transition_width
        if tw < 1e-10:
            edge = (freqs >= low_hz).to(freqs.dtype)
        else:
            edge = torch.clamp((freqs - (low_hz - tw)) / tw, 0.0, 1.0)
            edge = 0.5 * (1.0 - torch.cos(edge * torch.pi))
        gain = gain * edge
        gain[0] = 0.0
    if high_hz is not None and high_hz > 0:
        tw = high_hz * transition_width
        if tw < 1e-10:
            edge = (freqs <= high_hz).to(freqs.dtype)
        else:
            edge = torch.clamp((high_hz + tw - freqs) / tw, 0.0, 1.0)
            edge = 0.5 * (1.0 - torch.cos(edge * torch.pi))
        gain = gain * edge
    return gain


def _fft_segment(x: Tensor, tr: float, filt: TemporalFilter, tw: float, symmetric: bool) -> Tensor:
    n_t = x.shape[1]
    ext = torch.cat([x, x.flip(1)], dim=1) if symmetric else x
    n = ext.shape[1]
    freqs = torch.fft.rfftfreq(n, d=tr).to(device=x.device, dtype=x.dtype)
    gain = fft_gain(freqs, filt.low_hz, filt.high_hz, tw)
    out = torch.fft.irfft(torch.fft.rfft(ext, dim=1) * gain, n=n, dim=1)
    return out[:, :n_t]


def _movavg_segment(x: Tensor, window: int) -> Tensor:
    half = window // 2
    padded = torch.nn.functional.pad(x[:, None, :], (half, half), mode="replicate")
    kernel = torch.full((1, 1, window), 1.0 / window, device=x.device, dtype=x.dtype)
    return torch.nn.functional.conv1d(padded, kernel)[:, 0, :]


def apply_temporal_filter(
    data: Tensor,
    tr: float,
    filt: TemporalFilter,
    *,
    run_starts: list[int] | None = None,
    transition_width: float = 0.25,
    symmetric: bool = True,
    device: torch.device | None = None,
    verbose: bool = False,
) -> Tensor:
    """Filter ``(V, T)`` ``data`` within each run; returns a new tensor on
    ``data``'s device. Voxel chunks stream through ``device`` (default: the
    data's own)."""
    nyquist = 0.5 / tr
    for hz in (filt.low_hz, filt.high_hz):
        if hz is not None and hz >= nyquist:
            raise ValueError(
                f"filter edge {hz:g} Hz is at or above Nyquist ({nyquist:g} Hz, TR {tr:g} s)"
            )
    device = device if device is not None else data.device
    n_vox, n_t = data.shape
    starts = [0] if not run_starts else list(run_starts)
    bounds = starts + [n_t]
    out = torch.empty_like(data)
    chunk = estimate_chunk_size(n_vox, 4 * n_t, 4, device, operation="glm")
    for a in tqdm(
        range(0, n_vox, chunk),
        desc="  Temporal filter",
        unit="chunk",
        leave=True,
        disable=not verbose or n_vox <= chunk,
    ):
        b = min(a + chunk, n_vox)
        y = data[a:b].to(
            device=device, dtype=torch.float64 if data.dtype == torch.float64 else torch.float32
        )
        res = torch.empty_like(y)
        for r in range(len(bounds) - 1):
            seg = y[:, bounds[r] : bounds[r + 1]]
            if filt.kind == "movavg":
                assert filt.window is not None
                res[:, bounds[r] : bounds[r + 1]] = _movavg_segment(seg, filt.window)
            else:
                res[:, bounds[r] : bounds[r + 1]] = _fft_segment(
                    seg, tr, filt, transition_width, symmetric
                )
        out[a:b] = res.to(device=data.device, dtype=data.dtype)
    return out
