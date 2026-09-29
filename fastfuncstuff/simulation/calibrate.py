"""Calibrate simulation noise from a real dataset's ffs_reml outputs.

The question a simulation answers is "at *this* tSNR, is my design hopeless or
fine?", so what matters from real data is the spread of tSNR across the brain
and a reasonable ARMA(1,1) at each level of it -- not every voxel's parameters.
tSNR is binned by quantile inside the mask: the lowest bin is the worst case
(white matter, dropout), the highest the best (cortex near the coil). Each bin
gets the median tSNR and the median physiological noise parameters of its
voxels, ready to hand to :func:`simulate_bold`.

ARMA (a, b) is converted to TR-independent units per voxel before the medians
are taken (see :func:`~fastfuncstuff.simulation.noise.arma11_to_ou`): tau in
seconds and the physiological share of the variance.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .noise import arma11_from_acf, arma11_to_ou


@dataclass
class TsnrBin:
    """One quantile band of tSNR and the noise correlation typical of it.

    The correlation is summarised by the bin's median lag-1 and lag-2
    autocorrelation, and one ARMA(1,1) is fitted to those. A median of the
    per-voxel a would be meaningless: AFNI folds near-white voxels onto
    a = b = 0, so real Rvar maps have a spike at 0 beside the fitted spread.
    """

    tr: float
    quantile_range: tuple[float, float]
    tsnr_range: tuple[float, float]
    tsnr: float  # median
    acf_lag1: float  # median over voxels
    acf_lag2: float
    arma_a: float  # fitted to the median ACF
    arma_b: float
    tau: float  # seconds; nan unless representable
    phys_fraction: float  # nan unless representable
    representable: bool  # the median ACF is white + one exponential
    n_voxels: int
    frac_white: float  # voxels REML put at a = b = 0
    frac_a_at_ceiling: float  # a at the grid maximum: correlation truncated

    def simulation_kwargs(self, tr: float | None = None) -> dict[str, Any]:
        """Noise arguments for :func:`simulate_bold`.

        White + OU (``tau``, ``phys_fraction``) when the correlation fits it,
        which also carries over to another TR. Otherwise the fitted ARMA(1,1)
        directly, which is only valid at the TR it was measured at -- pass
        ``tr`` and a mismatch raises.
        """
        if self.representable:
            return {"tsnr": self.tsnr, "phys_fraction": self.phys_fraction, "tau": self.tau}
        if tr is not None and abs(tr - self.tr) > 1e-6:
            raise ValueError(
                f"this bin's correlation (a={self.arma_a:.2f}, b={self.arma_b:.2f}) is not "
                f"white + OU, so it is only defined at the measured TR {self.tr:g} s, not {tr:g}"
            )
        return {"tsnr": self.tsnr, "arma": (self.arma_a, self.arma_b)}


@dataclass
class NoiseProfile:
    """tSNR bins plus the acquisition they were measured at."""

    tr: float
    bins: list[TsnrBin]
    voxel_size: tuple[float, float, float] | None = None
    n_voxels: int = 0
    tr_source: str = "argument"
    notes: list[str] = field(default_factory=list)

    @property
    def worst(self) -> TsnrBin:
        return self.bins[0]

    @property
    def best(self) -> TsnrBin:
        return self.bins[-1]

    def summary(self) -> str:
        vox = (
            " x ".join(f"{v:g}" for v in self.voxel_size) + " mm"
            if self.voxel_size
            else "unknown voxel size"
        )
        lines = [
            f"Noise profile: TR {self.tr:g} s ({self.tr_source}), {vox}, {self.n_voxels} voxels",
            f"{'quantile':>9} {'tSNR':>6} {'range':>13} {'r1':>5} {'r2':>5} {'a':>5} "
            f"{'b':>6} {'tau(s)':>6} {'phys':>5} {'white':>5} {'a@max':>5}",
        ]
        for b in self.bins:
            q = f"{b.quantile_range[0]:.0%}-{b.quantile_range[1]:.0%}"
            r = f"{b.tsnr_range[0]:.1f}-{b.tsnr_range[1]:.1f}"
            ou = f"{b.tau:6.2f} {b.phys_fraction:5.2f}" if b.representable else f"{'ARMA only':>12}"
            lines.append(
                f"{q:>9} {b.tsnr:6.1f} {r:>13} {b.acf_lag1:5.2f} {b.acf_lag2:5.2f} "
                f"{b.arma_a:5.2f} {b.arma_b:6.2f} {ou} {b.frac_white:5.0%} "
                f"{b.frac_a_at_ceiling:5.0%}"
            )
        lines += [f"note: {n}" for n in self.notes]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tr": self.tr,
            "tr_source": self.tr_source,
            "voxel_size": self.voxel_size,
            "n_voxels": self.n_voxels,
            "notes": self.notes,
            "bins": [b.__dict__ | {"simulation": b.simulation_kwargs()} for b in self.bins],
        }


def noise_profile_from_maps(
    tsnr: np.ndarray | torch.Tensor,
    arma_a: np.ndarray | torch.Tensor,
    arma_b: np.ndarray | torch.Tensor,
    tr: float,
    mask: np.ndarray | torch.Tensor | None = None,
    quantiles: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    a_ceiling: float | None = None,
    voxel_size: tuple[float, float, float] | None = None,
) -> NoiseProfile:
    """Bin tSNR by quantile within the mask and summarise ARMA noise per bin.

    Parameters
    ----------
    tsnr, arma_a, arma_b : same-shape arrays (any spatial shape)
    tr : float, seconds
    mask : boolean array, optional. Default: finite tSNR > 0 -- ffs_reml writes
        its tSNR maps on its automask grid, zero outside it.
    quantiles : bin edges on [0, 1]; the first bin is the worst case
    a_ceiling : the REML grid's maximum a (0.8 by default in ffs_reml / AFNI).
        Voxels sitting on it have a tau that is only a lower bound; the default
        takes the largest a observed.
    """
    t = np.asarray(tsnr, dtype=np.float64).ravel()
    a = np.asarray(arma_a, dtype=np.float64).ravel()
    b = np.asarray(arma_b, dtype=np.float64).ravel()
    if not (t.shape == a.shape == b.shape):
        raise ValueError(f"tsnr, a and b must match in size: {t.size}, {a.size}, {b.size}")
    keep = np.isfinite(t) & (t > 0) & np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool).ravel()
    if keep.sum() < len(quantiles) - 1:
        raise ValueError(f"only {int(keep.sum())} usable voxels in the mask")
    t, a, b = t[keep], a[keep], b[keep]

    lam = (a + b) * (1 + a * b) / (1 + 2 * a * b + b**2)
    r1, r2 = lam, lam * a  # AFNI form: r(k) = lambda * a^(k-1)
    ceiling = float(a.max()) if a_ceiling is None else float(a_ceiling)
    # With every a at 0 there is no ceiling to hit, only an all-zero map.
    at_ceiling = (a >= ceiling - 1e-6) & (ceiling > 0)
    white = (np.abs(a) < 1e-6) & (np.abs(b) < 1e-6)

    edges = np.quantile(t, quantiles)
    bins = []
    notes = []
    for i in range(len(quantiles) - 1):
        lo, hi = edges[i], edges[i + 1]
        inb = (t >= lo) & ((t <= hi) if i == len(quantiles) - 2 else (t < hi))
        m1, m2 = float(np.median(r1[inb])), float(np.median(r2[inb]))
        fa, fb = (float(v) for v in arma11_from_acf(m1, m2))
        if fa > 0:
            ou = arma11_to_ou(tr, fa, fb)
            tau_i, f_i = float(ou["tau"]), float(ou["phys_fraction"])
            # The REML grid steps b by 0.1, so a pure AR(1) (f = 1) can land a
            # hair past the boundary; treat that as AR(1), not as unrepresentable.
            representable = 0.0 <= f_i <= 1.05
            f_i = min(f_i, 1.0)
        else:  # no lag-2 correlation: white if r1 ~ 0, else lag-1 only
            representable = abs(m1) < 1e-3
            tau_i, f_i = (1.0, 0.0) if representable else (np.nan, np.nan)
        if not representable:
            tau_i, f_i = np.nan, np.nan
        bins.append(
            TsnrBin(
                tr=tr,
                quantile_range=(quantiles[i], quantiles[i + 1]),
                tsnr_range=(float(lo), float(hi)),
                tsnr=float(np.median(t[inb])),
                acf_lag1=m1,
                acf_lag2=m2,
                arma_a=fa,
                arma_b=fb,
                tau=tau_i,
                phys_fraction=f_i,
                representable=representable,
                n_voxels=int(inb.sum()),
                frac_white=float(white[inb].mean()),
                frac_a_at_ceiling=float(at_ceiling[inb].mean()),
            )
        )
    if not all(bn.representable for bn in bins):
        notes.append(
            "some bins have more short-lag correlation than white + one exponential makes "
            "(r2 < r1 * a; e.g. interpolation or temporal filtering): they simulate as ARMA "
            "at this TR only"
        )
    frac_ceiling = float(at_ceiling.mean())
    if frac_ceiling > 0.1:
        notes.append(
            f"{frac_ceiling:.0%} of voxels sit at a = {ceiling:g}, the REML grid maximum: "
            f"their correlation is truncated (tau <= {-tr / np.log(ceiling):.1f} s). Refit "
            f"with a wider grid (ffs_reml -a_grid 0:0.95:20) for a real estimate."
        )
    return NoiseProfile(tr=tr, bins=bins, voxel_size=voxel_size, n_voxels=int(t.size), notes=notes)


def _load_volume(path: str | Path) -> tuple[np.ndarray, Any]:
    """NIfTI (.nii/.nii.gz/.nii.zst) or AFNI HEAD/BRIK -- 3dREMLfit's own -Rvar is the latter."""
    from fastfuncstuff.io.afni import load_nifti

    name = str(path)
    if name.endswith((".HEAD", ".BRIK", ".BRIK.gz")):
        import nibabel as nib

        head = name.split(".BRIK")[0] + ".HEAD" if ".BRIK" in name else name
        img: Any = nib.load(head)
    else:
        img = load_nifti(path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    # Drop AFNI's singleton non-spatial axes: (x,y,z,1,6) -> (x,y,z,6).
    while data.ndim > 3 and 1 in data.shape[3:]:
        data = np.squeeze(data, axis=3 + data.shape[3:].index(1))
    return data, img.header


def _header_tr(header) -> float | None:
    """TR from a NIfTI header when its units say time, or an AFNI header's TAXIS."""
    info = getattr(header, "info", None)
    if isinstance(info, dict):  # AFNI: TAXIS_FLOATS = [origin, TR, ...], seconds
        taxis = info.get("TAXIS_FLOATS")
        return float(taxis[1]) if taxis is not None and float(taxis[1]) > 0 else None
    try:
        units = header.get_xyzt_units()[1]
        value = float(header["pixdim"][4])
    except Exception:
        return None
    if value <= 0 or units not in ("sec", "msec", "usec"):
        return None
    return value * {"sec": 1.0, "msec": 1e-3, "usec": 1e-6}[units]


def noise_profile_from_reml(
    rvar: str | Path,
    tsnr: str | Path,
    tr: float | None = None,
    mask: str | Path | None = None,
    quantiles: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    a_ceiling: float | None = None,
) -> NoiseProfile:
    """:func:`noise_profile_from_maps` on ffs_reml outputs.

    Parameters
    ----------
    rvar : the -Rvar file (sub-bricks a, b, lambda, StDev, ...)
    tsnr : a -save_tsnr map; ``PREFIX.resid_tsnr_reml`` is the one to use --
        noise-only tSNR, with task and drift removed. ``raw_tsnr`` counts the
        task response as noise.
    tr : seconds. Derived images often carry no TR, so pass it; the header is
        only read (and only trusted when its units say seconds) as a fallback.
        A header TR that disagrees with the argument is reported.
    mask : optional mask image on the same grid
    """
    rv, rv_header = _load_volume(rvar)
    if rv.ndim != 4 or rv.shape[3] < 2:
        raise ValueError(f"{rvar}: expected a 4D Rvar with a and b sub-bricks, got {rv.shape}")
    a, b = rv[..., 0], rv[..., 1]
    if np.nanmax(np.abs(a)) >= 1 or np.nanmax(np.abs(b)) >= 1:
        raise ValueError(
            f"{rvar}: sub-bricks 0/1 are not ARMA (a, b) -- |values| reach 1. "
            "Is this the stats bucket rather than the Rvar?"
        )
    t, t_header = _load_volume(tsnr)
    if t.ndim != 3 or t.shape != a.shape:
        raise ValueError(f"{tsnr}: expected a 3D map on the Rvar grid {a.shape}, got {t.shape}")
    m = None
    if mask is not None:
        m, _ = _load_volume(mask)
        if m.shape[:3] != a.shape:
            raise ValueError(f"{mask}: grid {m.shape} does not match {a.shape}")
        m = m.reshape(a.shape) > 0

    header_tr = _header_tr(rv_header) or _header_tr(t_header)
    if tr is None:
        if header_tr is None:
            raise ValueError("no TR in the Rvar or tSNR header -- pass tr= (seconds)")
        tr, source = header_tr, "header"
    else:
        source = "argument"
        if header_tr is not None and abs(header_tr - tr) > 1e-3:
            warnings.warn(
                f"header TR {header_tr:g} s differs from tr={tr:g}; using {tr:g}", stacklevel=2
            )

    zooms = tuple(float(z) for z in rv_header.get_zooms()[:3])
    profile = noise_profile_from_maps(
        t, a, b, tr, m, quantiles=quantiles, a_ceiling=a_ceiling, voxel_size=zooms
    )
    profile.tr_source = source
    return profile
