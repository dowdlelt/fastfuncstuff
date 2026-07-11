"""Whole-dataset diagnostics for ffs_reml.

``ffs_reml`` is the one point in a workflow where the ENTIRE timeseries (and,
after the fit, its residuals) is resident in RAM at once. Reloading a multi-GB
4-D stack just to compute a temporal-mean or a residual smoothness estimate is a
waste, so this module collects the cheap per-voxel / per-run maps that fall out
of data we already have.

Organisation — this is meant to be easy to extend. The CLI feeds a
:class:`DatasetDiagnostics` at three hook points, in pipeline order::

    diag.observe_raw(data_2d)          # after load, BEFORE scaling
    diag.observe_scaled(data_2d)       # after per-run scaling to mean 100
    diag.observe_residuals({"ols": r_ols, "reml": r_reml}, voxel_mask)  # after fit

Each hook fills ``diag.maps`` (per-voxel volumes → NIfTI) and/or ``diag.tables``
(text → e.g. the FWHMx ACF report). To add a new diagnostic: compute it inside
the relevant ``observe_*`` (or add a new hook), store it under a name, and have
the CLI save it when its flag is set. The residual hook takes a ``{label:
residuals}`` dict, so any residual-derived map is produced once per label
(``..._ols`` / ``..._reml``) automatically.

All heavy lifting reuses existing primitives (``processing.mask.automask`` for
the mask, ``stats.localstat`` for the ACF model/fit/FWHM); nothing here
re-derives them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Temporal primitives (time is the trailing axis; voxels lead)
# ---------------------------------------------------------------------------


def temporal_mean(data_2d: Tensor) -> Tensor:
    """Per-voxel mean over time. ``data_2d`` is (n_voxels, n_time)."""
    return data_2d.mean(dim=1)


def temporal_std(data_2d: Tensor, ddof: int = 1) -> Tensor:
    """Per-voxel standard deviation over time (sample std, ddof=1 by default)."""
    n = data_2d.shape[1]
    if n <= ddof:
        return torch.zeros(data_2d.shape[0], dtype=data_2d.dtype, device=data_2d.device)
    mean = data_2d.mean(dim=1, keepdim=True)
    ss = ((data_2d - mean) ** 2).sum(dim=1)
    return torch.sqrt(ss / (n - ddof))


def tsnr(mean: Tensor, std: Tensor) -> Tensor:
    """Temporal SNR = mean / std, with std==0 mapped to 0 (not inf)."""
    out = torch.zeros_like(mean)
    nz = std > 0
    out[nz] = mean[nz] / std[nz]
    return out


# ---------------------------------------------------------------------------
# Mask resolution (automask the input mean when none is supplied)
# ---------------------------------------------------------------------------


def resolve_mask(
    mask: Tensor | None,
    mean_vol: Tensor,
    *,
    device: torch.device | None = None,
    verbose: bool = True,
) -> Tensor:
    """Return a boolean (nz,ny,nx) mask, automasking ``mean_vol`` if ``mask`` is None.

    ``mean_vol`` is the per-voxel temporal mean reshaped to the volume grid (the
    grand mean is the natural anatomy-bearing image to threshold). Prints what it
    did so the choice is never silent.
    """
    if mask is not None:
        m = mask.to(torch.bool)
        if verbose:
            print(f"  diagnostics mask: using supplied mask ({int(m.sum())} voxels)")
        return m
    from fastfuncstuff.processing.mask import automask

    m = automask(mean_vol.float(), device=device).to(torch.bool)
    if verbose:
        print(
            f"  diagnostics mask: no -mask given → automasked the grand mean "
            f"({int(m.sum())} of {m.numel()} voxels)"
        )
    return m


# ---------------------------------------------------------------------------
# Per-run whole-volume smoothness (3dFWHMx classic + ACF)
# ---------------------------------------------------------------------------
#
# This measures the OVERALL spatial blurring of the residual field per run, the
# way ``3dFWHMx`` does (spatial ACF within each sub-brick, averaged over
# sub-bricks). It is NOT ``stats.localstat`` / ``3dLocalACF``, which correlates
# neighbours across time to make a spatially-varying map -- the wrong tool for a
# single per-run blur estimate. See :mod:`fastfuncstuff.stats.fwhmx`.


def per_run_fwhmx(
    residuals_2d: Tensor,
    voxel_mask: Tensor,
    run_starts: list[int],
    volume_shape: tuple[int, int, int],
    voxdims: tuple[float, float, float],
    device: torch.device | None = None,
    *,
    unif: bool = True,
    verbose: bool = True,
):
    """3dFWHMx classic + ACF estimate of the residual smoothness, once per run.

    ``residuals_2d`` is (n_masked_voxels, n_time); ``voxel_mask`` places them on
    the ``volume_shape`` grid. ``run_starts`` are the time indices where each run
    begins (the last run runs to the end). Streams the sub-bricks in
    memory-model-sized chunks (never materialises the full 4-D stack).

    ``unif`` (default True) uniformizes per-voxel variance by temporal MAD before
    estimating, matching afni_proc.py's blur estimate (``3dFWHMx -detrend`` sets
    ``-unif``). The residuals are already model-detrended, so only the MAD step is
    applied; it markedly affects the ACF on data with spatially non-uniform
    variance (high-res / anisotropic).

    Returns a list of ``(run_number_1based, FWHMxResult)``.
    """
    from fastfuncstuff.stats.fwhmx import estimate_fwhmx_run

    n_time = residuals_2d.shape[1]
    bounds = list(run_starts) + [n_time]

    rows = []
    n_runs = len(run_starts)
    for r in range(n_runs):
        t0, t1 = bounds[r], bounds[r + 1]
        if t1 <= t0:
            continue
        if verbose:
            print(f"    run {r + 1}/{n_runs}: FWHMx over {t1 - t0} sub-bricks...")
        res = estimate_fwhmx_run(
            residuals_2d[:, t0:t1],
            voxel_mask,
            volume_shape,
            voxdims,
            unif=unif,
            device=device,
            progress=verbose,
            progress_desc=f"    run {r + 1} FWHMx",
        )
        if verbose:
            fx, fy, fz = res.classic_fwhm
            print(
                f"      classic FWHM x/y/z = {fx:.2f}/{fy:.2f}/{fz:.2f} mm; "
                f"ACF a={res.a:.3f} b={res.b:.3f} c={res.c:.3f} FWHM={res.fwhm:.2f} mm "
                f"(radius {res.radius:.1f} mm)"
            )
        rows.append((r + 1, res))
    return rows


def fwhmx_report_text(rows) -> str:
    """Human-readable per-run table: classic FWHM x/y/z + ACF a/b/c/FWHM.

    Trailing ``avg`` row averages each column over runs (matches how AFNI's
    blur-estimate table is consumed downstream).
    """
    lines = [
        "# 3dFWHMx per run: classic Forman FWHM (x,y,z) + ACF model",
        "#   ACF(r) = a*exp(-r*r/(2*b*b)) + (1-a)*exp(-r/c);  all FWHM in mm",
        "# run     FWHMx    FWHMy    FWHMz        a        b        c  ACF_FWHM",
    ]
    acc = []
    for run, res in rows:
        fx, fy, fz = res.classic_fwhm
        acc.append((fx, fy, fz, res.a, res.b, res.c, res.fwhm))
        lines.append(
            f"  {run:>3}  {fx:7.3f}  {fy:7.3f}  {fz:7.3f}  "
            f"{res.a:7.4f}  {res.b:7.4f}  {res.c:7.4f}  {res.fwhm:8.3f}"
        )
    if acc:
        m = np.array(acc).mean(0)
        lines.append(
            f"  avg  {m[0]:7.3f}  {m[1]:7.3f}  {m[2]:7.3f}  "
            f"{m[3]:7.4f}  {m[4]:7.4f}  {m[5]:7.4f}  {m[6]:8.3f}"
        )
    return "\n".join(lines) + "\n"


def blur_est_1D(rows) -> str:
    """Run-averaged ACF params + FWHM as a 1-line .1D (AFNI blur-estimate style).

    One data line ``a b c FWHM`` (the mean over runs), ready to feed the ACF
    triple to ``3dClustSim -acf a b c``. Empty runs are skipped in the mean.
    """
    a = b = c = fwhm = 0.0
    if rows:
        m = np.array([(res.a, res.b, res.c, res.fwhm) for _run, res in rows]).mean(0)
        a, b, c, fwhm = (float(v) for v in m)
    return (
        "# run-averaged spatial ACF of the residuals (3dFWHMx -ACF)\n"
        "# a b c FWHM   (feed 'a b c' to 3dClustSim -acf; FWHM in mm)\n"
        f"{a:.5f} {b:.5f} {c:.5f} {fwhm:.5f}\n"
    )


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


@dataclass
class DatasetDiagnostics:
    """Accumulates the whole-dataset diagnostics as the CLI proceeds.

    ``maps`` holds per-voxel volumes (nz,ny,nx float32) ready to save as NIfTI;
    ``tables`` holds text blobs. Populated by the ``observe_*`` hooks; the CLI
    saves whichever the user requested by flag.
    """

    volume_shape: tuple[int, int, int]
    run_starts: list[int]
    voxdims: tuple[float, float, float]
    device: torch.device | None = None
    verbose: bool = True

    maps: dict[str, np.ndarray] = field(default_factory=dict)
    tables: dict[str, str] = field(default_factory=dict)
    # Cached between hooks.
    _scaled_mean: Tensor | None = None
    _mask: Tensor | None = None

    def _to_vol(self, flat: Tensor) -> np.ndarray:
        return flat.detach().cpu().float().numpy().reshape(self.volume_shape)

    # -- hook 1: raw data, before scaling -----------------------------------
    def observe_raw(self, data_2d: Tensor) -> None:
        """Grand mean (per-voxel temporal mean) of the un-scaled data."""
        if self.verbose:
            print("  diagnostics: grand mean (per-voxel temporal mean)...")
        self.maps["grandmean"] = self._to_vol(temporal_mean(data_2d))

    # -- hook 2: scaled (or unscaled) data -----------------------------------
    def observe_scaled(self, data_2d: Tensor) -> None:
        """Raw tSNR = mean / std of the timeseries.

        tSNR is invariant to per-voxel scaling, so this is valid whether or not
        the data was scaled to mean 100; the cached mean is reused as the signal
        for residual tSNR.
        """
        if self.verbose:
            print("  diagnostics: raw tSNR (mean/std of the timeseries)...")
        mean = temporal_mean(data_2d)
        self._scaled_mean = mean
        self.maps["raw_tsnr"] = self._to_vol(tsnr(mean, temporal_std(data_2d)))

    # -- hook 3: residuals ---------------------------------------------------
    def observe_residuals(
        self,
        residuals: dict[str, Tensor | None],
        voxel_mask: Tensor,
        *,
        want_tsnr: bool = True,
        want_fwhmx: bool = False,
    ) -> None:
        """Residual-derived maps, once per label (``resid_tsnr_ols`` etc.).

        ``residuals`` maps a label ('ols'/'reml') to a (n_masked, n_time) tensor
        on the ``voxel_mask`` grid. resid_tsnr reuses the scaled mean as signal.
        """
        vm = voxel_mask.reshape(self.volume_shape)
        vm_flat = voxel_mask.reshape(-1).cpu().numpy().astype(bool)
        for label, resid in residuals.items():
            if resid is None:
                continue
            if want_tsnr and self._scaled_mean is not None:
                if self.verbose:
                    print(f"  diagnostics: residual tSNR [{label}]...")
                std_resid = temporal_std(resid)  # (n_masked,)
                sig = self._scaled_mean.detach().cpu().float().numpy()[vm_flat]
                sig_t = torch.from_numpy(sig).to(std_resid.device)
                tvals = tsnr(sig_t, std_resid).detach().cpu().float().numpy()
                vol = np.zeros(self.volume_shape, dtype=np.float32)
                vol[np.asarray(vm.cpu())] = tvals
                self.maps[f"resid_tsnr_{label}"] = vol
            if want_fwhmx:
                if self.verbose:
                    print(f"  diagnostics: 3dFWHMx residual smoothness [{label}]...")
                rows = per_run_fwhmx(
                    resid,
                    voxel_mask,
                    self.run_starts,
                    self.volume_shape,
                    self.voxdims,
                    device=self.device,
                    verbose=self.verbose,
                )
                # Per-run detail (classic + ACF) + run-averaged .1D.
                self.tables[f"fwhmx_{label}"] = fwhmx_report_text(rows)
                self.tables[f"blur_est_{label}"] = blur_est_1D(rows)

    # -- saving --------------------------------------------------------------
    def save_map(self, name: str, path, affine: np.ndarray | None = None, header=None) -> bool:
        """Save one per-voxel map as NIfTI. Returns False if the map is absent."""
        if name not in self.maps:
            return False
        from fastfuncstuff.io.afni import save_nifti

        save_nifti(self.maps[name], path, affine=affine, header=header)
        if self.verbose:
            print(f"  • {name}: {path}")
        return True

    def save_table(self, name: str, path) -> bool:
        """Save one text table. Returns False if absent."""
        if name not in self.tables:
            return False
        from pathlib import Path

        Path(path).write_text(self.tables[name])
        if self.verbose:
            print(f"  • {name}: {path}")
        return True
