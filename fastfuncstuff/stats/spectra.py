"""One-sided power spectral density of many series at once.

Every estimator here is scaled the same way, so their outputs can be compared
and summed into bands: ``psd.sum(-1) * df`` is the series' variance (the DC bin
is dropped, so the mean is not part of it). That makes a band's power a
variance -- its square root a standard deviation in the data's own units -- and
a band's share of the total a fraction that means the same thing under every
method.

Three estimators, trading frequency resolution for variance:

* ``periodogram`` -- one tapered FFT. Full resolution, but each bin is a
  chi-square on two degrees of freedom: its standard deviation equals its mean.
  Hann by default, because an untapered FFT leaks a strong low-frequency drift
  residual into every bin above it.
* ``multitaper`` -- the average of ``2*NW - 1`` periodograms under orthogonal
  DPSS tapers (Thomson). Resolution ``2*NW / duration``, variance down by the
  number of tapers. The usual best compromise for runs of a few hundred volumes.
* ``welch`` -- the average of Hann-tapered half-overlapping segments. Variance
  down roughly by the segment count, resolution down by the same factor.

Rows are series: ``(V, T)``, the layout the viewer and the GLM code stage voxels
in. Computation is on the device the rows are on; the caller chunks.
"""

from __future__ import annotations

import math

import numpy as np
import torch

METHODS = ("periodogram", "multitaper", "welch")


def n_tapers(nw: float) -> int:
    """The DPSS tapers kept for a time-bandwidth product: ``2*NW - 1``, at least one."""
    return max(1, int(math.floor(2.0 * float(nw))) - 1)


def tapers(method: str, n_time: int, *, nw: float = 2.0, segments: int = 4) -> np.ndarray:
    """``(K, L)`` float64 tapers, each scaled to unit energy (``sum w**2 == 1``).

    ``L`` is the segment length: the run's for a periodogram or multitaper, the
    Welch segment's otherwise.
    """
    if method == "periodogram":
        return _unit(np.hanning(n_time + 2)[1:-1])[None, :]
    if method == "multitaper":
        from scipy.signal.windows import dpss

        k = n_tapers(nw)
        w = np.atleast_2d(dpss(n_time, NW=float(nw), Kmax=k, sym=False))
        return np.stack([_unit(row) for row in w])
    if method == "welch":
        return _unit(np.hanning(welch_length(n_time, segments) + 2)[1:-1])[None, :]
    raise ValueError(f"unknown spectral method {method!r}; have {', '.join(METHODS)}")


def _unit(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    return w / math.sqrt(float(np.sum(w * w)))


def welch_length(n_time: int, segments: int) -> int:
    """Segment length giving ``segments`` half-overlapping segments over the run."""
    segments = max(1, int(segments))
    return max(4, int(2 * n_time // (segments + 1)))


def frequencies(method: str, n_time: int, tr: float, *, segments: int = 4) -> np.ndarray:
    """``(F,)`` bin centres in Hz, DC excluded, up to and including Nyquist."""
    length = welch_length(n_time, segments) if method == "welch" else n_time
    return np.fft.rfftfreq(length, d=float(tr))[1:]


def psd(
    rows: torch.Tensor,
    tr: float,
    *,
    method: str = "periodogram",
    nw: float = 2.0,
    segments: int = 4,
    window: torch.Tensor | None = None,
) -> torch.Tensor:
    """``(V, F)`` one-sided PSD of ``(V, T)`` rows, in units**2 / Hz, DC dropped.

    Rows should already be mean-free (and detrended: a drift left in is power
    at the lowest bins, and leaks above them however good the taper). ``window``
    is :func:`tapers`' output on ``rows``' device, for a caller streaming many
    chunks; it is built here when not given.
    """
    if rows.ndim != 2:
        raise ValueError(f"rows must be (V, T); got {tuple(rows.shape)}")
    n_time = int(rows.shape[1])
    fs = 1.0 / float(tr)
    w = window
    if w is None:
        w = torch.as_tensor(
            tapers(method, n_time, nw=nw, segments=segments),
            dtype=rows.dtype,
            device=rows.device,
        )
    x = rows
    if method == "welch":
        length = int(w.shape[-1])
        step = max(1, length // 2)
        seg = x.unfold(1, length, step)  # (V, S, L)
        seg = seg - seg.mean(dim=2, keepdim=True)
        spec = torch.fft.rfft(seg * w[0], dim=2).abs().square().mean(dim=1)
    else:
        length = n_time
        # (V, K, T) -> averaged over the K tapers.
        spec = torch.fft.rfft(x[:, None, :] * w[None, :, :], dim=2).abs().square().mean(dim=1)
    # Unit-energy tapers: |X_k|^2 / fs is the two-sided density. Doubled for
    # one side, except at Nyquist, which has no mirror image.
    spec = spec[:, 1:] / fs
    double = torch.full((spec.shape[1],), 2.0, dtype=spec.dtype, device=spec.device)
    if length % 2 == 0:
        double[-1] = 1.0
    return spec * double


def alias(frequency_hz: float | np.ndarray, tr: float) -> np.ndarray:
    """Where a frequency appears after sampling every ``tr`` seconds.

    Folded into ``[0, 1 / (2 tr)]``: a 1.1 Hz heartbeat at TR 2 s (sampling at
    0.5 Hz, Nyquist 0.25 Hz) shows up at 0.1 Hz.
    """
    fs = 1.0 / float(tr)
    f = np.mod(np.asarray(frequency_hz, dtype=np.float64), fs)
    return np.where(f > fs / 2.0, fs - f, f)


__all__ = ["METHODS", "alias", "frequencies", "n_tapers", "psd", "tapers", "welch_length"]
