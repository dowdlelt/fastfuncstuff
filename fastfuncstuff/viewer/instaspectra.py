"""The InstaSpectra engine: every voxel's power spectrum, and maps read off it.

One run in, ``(V, F)`` one-sided PSDs out (``stats/spectra.py``), each voxel
detrended first with the same Legendre drift InstaPCA removes. The maps are then
views of that one array, cheap enough to remake on every change of a display
control: one map per frequency bin, which the viewer steps through as the
map layer's sub-bricks.

The default map is a voxel's **share** of its own power: the power in a band
around each bin over the voxel's total above a floor (0.04 Hz). A share means
the same thing in every voxel, so a vessel is not bright merely for being loud
-- an absolute power map is a picture of vessels, CSF and the brain's edge at
every frequency. The floor keeps the drift's residue (Legendre removes the
slowest of it, not all) out of the denominator, where it would otherwise set
every other bin's share. Hann-tapered periodogram, quadratic drift and the 0.04
Hz floor are the choices the physio work settled on by looking at cardiac MIPs
(``scripts/specmip.py`` on the physio branch).

Beside it, **peak** asks whether the band stands out of its neighbourhood --
band density over the density of the bins flanking it, 1 where the spectrum is
smooth there -- which is the question for a narrow aliased heartbeat sitting on
a 1/f slope. **amplitude** is the band's standard deviation in percent of the
voxel's mean, what removing it would cost.

Any of them can be shown as a **robust z across the mask**, frame by frame:
``(x - median) / (1.4826 * MAD)``. That is the auto-threshold: one cut (z = 3)
means "far out of the brain's own distribution at this frequency" in every
frame, so a threshold set at one bin still means something ten bins later.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from fastfuncstuff.stats import spectra

ProgressFn = Callable[[float, str], None]

MAP_KINDS = ("share", "peak", "amplitude")
SCALES = ("robust z", "value")
#: What each map's values are called on screen.
MAP_UNITS = {"share": "% of power", "peak": "x neighbours", "amplitude": "% signal sd"}
#: Floor of a z: below this the voxel is ordinary at this frequency, and a
#: negative z is not something anyone looks for in a power map.
Z_FLOOR = 0.0


@dataclass
class Spectra:
    """Masked voxels' PSDs, and what is needed to put a map of them back on the grid."""

    #: ``(F,)`` Hz, DC excluded.
    freqs: np.ndarray
    #: ``(V, F)`` float32, units**2 / Hz: ``psd.sum(1) * df`` is each voxel's
    #: detrended variance.
    psd: np.ndarray
    #: ``(V,)`` each voxel's mean, for amplitude in percent.
    mean: np.ndarray
    mask: np.ndarray
    affine: np.ndarray
    tr: float
    #: Whether ``tr`` came from the header. When it did not, the axis is in
    #: cycles per volume (``tr`` is 1) and says so.
    tr_known: bool
    polort: int
    method: str
    n_time: int
    #: ``(F,)`` the mean PSD over the mask: the brain's spectrum.
    mask_mean: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.mask_mean = self.psd.mean(axis=0) if self.psd.size else np.zeros_like(self.freqs)

    @property
    def n_frames(self) -> int:
        return int(self.freqs.size)

    @property
    def df(self) -> float:
        return float(self.freqs[1] - self.freqs[0]) if self.freqs.size > 1 else 1.0

    @property
    def unit(self) -> str:
        return "Hz" if self.tr_known else "cycles/vol"

    def half_bins(self, width: float) -> int:
        """Bins either side of the centre a band ``width`` Hz wide sums."""
        return max(0, int(round(float(width) / 2.0 / self.df)))

    def band_edges(self, k: int, width: float = 0.0) -> tuple[float, float]:
        """The frequencies bin ``k``'s band covers, outer bin edges included.

        Clipped at the ends of the spectrum exactly as :meth:`maps` clips the
        sum, so what is shown is what was summed.
        """
        half = self.half_bins(width)
        lo = max(0, int(k) - half)
        hi = min(self.n_frames - 1, int(k) + half)
        return (
            max(0.0, float(self.freqs[lo]) - self.df / 2.0),
            float(self.freqs[hi]) + self.df / 2.0,
        )

    def frame_of(self, frequency: float) -> int:
        """The bin nearest a frequency."""
        return int(np.argmin(np.abs(self.freqs - float(frequency))))

    def maps(
        self,
        kind: str = "share",
        *,
        width: float = 0.0,
        floor: float = 0.04,
        scale: str = "robust z",
    ) -> np.ndarray:
        """``(V, F)`` float32, one map per bin. ``width`` is the band, in Hz."""
        if kind not in MAP_KINDS:
            raise ValueError(f"unknown map {kind!r}; have {', '.join(MAP_KINDS)}")
        p = torch.from_numpy(self.psd).double()
        half = self.half_bins(width)
        band_sum, band_n = _band(p, half)
        if kind == "share":
            above = torch.from_numpy(self.freqs >= float(floor))
            if not bool(above.any()):
                above = torch.ones_like(above)
            total = p[:, above].sum(dim=1, keepdim=True).clamp_min(1e-30)
            out = 100.0 * band_sum / total
        elif kind == "amplitude":
            mean = torch.from_numpy(np.abs(self.mean)).double()[:, None]
            sd = torch.sqrt(band_sum * self.df)
            out = torch.where(mean > 0, 100.0 * sd / mean.clamp_min(1e-30), 0.0)
        else:
            out = band_sum / band_n / _flanks(p, half).clamp_min(1e-30)
        out = out.float()
        if scale == "robust z":
            out = robust_z(out).clamp_min(Z_FLOOR)
        return out.numpy()

    def volume(self, column: np.ndarray) -> np.ndarray:
        """``(V,)`` back onto the run's grid, NaN outside the mask."""
        vol = np.full(self.mask.shape, np.nan, dtype=np.float32)
        vol[self.mask] = column
        return vol


def _band(p: torch.Tensor, half: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum and count of the bins within ``half`` of each bin, clipped at the ends."""
    n = p.shape[1]
    c = torch.nn.functional.pad(p.cumsum(dim=1), (1, 0))
    idx = torch.arange(n)
    lo = (idx - half).clamp(min=0)
    hi = (idx + half + 1).clamp(max=n)
    return c[:, hi] - c[:, lo], (hi - lo).to(p.dtype)[None, :]


def _flanks(p: torch.Tensor, half: int) -> torch.Tensor:
    """Mean density of the bins just outside each band, both sides, as many as exist.

    Each flank is as wide as the band and at least two bins, so a one-bin band
    is judged against a neighbourhood rather than against one equally noisy bin.
    """
    n = p.shape[1]
    width = max(2 * half + 1, 2)
    c = torch.nn.functional.pad(p.cumsum(dim=1), (1, 0))
    idx = torch.arange(n)
    l_hi = (idx - half).clamp(min=0)
    l_lo = (l_hi - width).clamp(min=0)
    r_lo = (idx + half + 1).clamp(max=n)
    r_hi = (r_lo + width).clamp(max=n)
    total = (c[:, l_hi] - c[:, l_lo]) + (c[:, r_hi] - c[:, r_lo])
    count = ((l_hi - l_lo) + (r_hi - r_lo)).clamp(min=1).to(p.dtype)[None, :]
    return total / count


def robust_z(maps: torch.Tensor) -> torch.Tensor:
    """Each column as a z against the median and MAD of its own voxels."""
    med = maps.median(dim=0, keepdim=True).values
    mad = (maps - med).abs().median(dim=0, keepdim=True).values * 1.4826
    # A column with no spread (a constant map) is zero everywhere, not NaN.
    return torch.where(mad > 0, (maps - med) / mad.clamp_min(1e-30), torch.zeros_like(maps))


def compute(
    data: np.ndarray,
    *,
    affine: np.ndarray,
    tr: float,
    mask: np.ndarray | None = None,
    polort: int = 2,
    method: str = "periodogram",
    nw: float = 2.0,
    segments: int = 4,
    device: torch.device | None = None,
    progress: ProgressFn | None = None,
) -> Spectra:
    """Detrend every masked voxel of one 4-D run and take its PSD."""
    from fastfuncstuff.memory import spectral_voxel_chunk
    from fastfuncstuff.viewer.derive import orthonormal_basis, read_nuisance

    if data.ndim != 4:
        raise ValueError(f"InstaSpectra needs one 4-D run; got shape {tuple(data.shape)}")
    n_time = int(data.shape[3])
    polort = max(0, int(polort))
    if n_time < max(polort + 3, 8):
        raise ValueError(f"{n_time} volumes is too short for a spectrum")
    if method == "welch" and spectra.welch_length(n_time, segments) >= n_time:
        raise ValueError("welch needs at least two segments")
    device = device or torch.device("cpu")
    tr_known = bool(tr and tr > 0)
    tr = float(tr) if tr_known else 1.0

    if progress is not None:
        progress(0.02, "masking")
    if mask is None:
        from fastfuncstuff.viewer.series import automask_from_series

        mask = automask_from_series(data, device=device)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != data.shape[:3]:
        raise ValueError(f"mask shape {mask.shape} does not match the run {data.shape[:3]}")
    n_vox = int(mask.sum())
    if n_vox < 2:
        raise ValueError("the mask has fewer than two voxels")

    drift = orthonormal_basis(read_nuisance(None, n_time=n_time, polort=polort).columns)
    q = torch.as_tensor(drift, dtype=torch.float32, device=device)
    window = torch.as_tensor(
        spectra.tapers(method, n_time, nw=nw, segments=segments),
        dtype=torch.float32,
        device=device,
    )
    freqs = spectra.frequencies(method, n_time, tr, segments=segments)
    rows = np.asarray(data).reshape(-1, n_time)[mask.reshape(-1)]
    # Each taper is one more spectrum-sized copy of the chunk.
    step = spectral_voxel_chunk(n_vox, n_time * int(window.shape[0]), device)
    psd = np.empty((n_vox, freqs.size), dtype=np.float32)
    mean = np.empty(n_vox, dtype=np.float32)
    for i in range(0, n_vox, step):
        if progress is not None:
            progress(0.05 + 0.9 * i / n_vox, "spectra")
        staged = torch.as_tensor(rows[i : i + step]).to(device).to(torch.float32)
        mean[i : i + step] = staged.mean(dim=1).cpu().numpy()
        c = staged - (staged @ q) @ q.T
        psd[i : i + step] = (
            spectra.psd(c, tr, method=method, nw=nw, segments=segments, window=window).cpu().numpy()
        )
        del staged, c
    if progress is not None:
        progress(1.0, "spectra")
    return Spectra(
        freqs=freqs,
        psd=psd,
        mean=mean,
        mask=mask,
        affine=np.asarray(affine, dtype=float),
        tr=tr,
        tr_known=tr_known,
        polort=polort,
        method=method,
        n_time=n_time,
    )


def voxel_psd(
    series: np.ndarray,
    *,
    tr: float,
    polort: int = 2,
    method: str = "periodogram",
    nw: float = 2.0,
    segments: int = 4,
) -> np.ndarray:
    """One series' PSD, prepared exactly as :func:`compute` prepares a voxel's."""
    from fastfuncstuff.viewer.derive import detrend_trace

    x = detrend_trace(np.asarray(series, dtype=np.float64), max(0, int(polort)))
    rows = torch.as_tensor(x[None, :], dtype=torch.float64)
    return (
        spectra.psd(rows, tr if tr > 0 else 1.0, method=method, nw=nw, segments=segments)
        .numpy()[0]
        .astype(np.float32)
    )


__all__ = ["MAP_KINDS", "MAP_UNITS", "SCALES", "Spectra", "compute", "robust_z", "voxel_psd"]
