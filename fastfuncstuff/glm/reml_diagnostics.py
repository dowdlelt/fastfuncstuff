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
# Per-run spatial ACF / FWHMx (3dFWHMx -ACF), reusing the localstat fitters
# ---------------------------------------------------------------------------


def global_acf_fit(
    resid_vol: Tensor,
    mask: Tensor,
    voxdims: tuple[float, float, float],
    nbhd: str = "SPHERE(-9.666)",
    device: torch.device | None = None,
) -> tuple[float, float, float, float]:
    """Fit the AFNI ACF model to the whole-mask spatial autocorrelation.

    Global analogue of ``stats.localstat.local_acf``: instead of a per-voxel
    neighborhood curve, it sums the binned correlations over every masked voxel
    to get one empirical ACF(r) for the volume, then fits
    ``a·exp(-r²/2b²)+(1-a)·exp(-r/c)`` and reports the effective FWHM — i.e.
    3dFWHMx's ``-ACF`` output.

    Args:
        resid_vol: (nt, nz, ny, nx) residual timeseries for one run (time is the
            replicate dimension the correlation is computed over).
        mask: (nz, ny, nx) bool mask.
        voxdims: (dx, dy, dz) mm.
        nbhd: AFNI ``-nbhd`` string setting the ACF radius.

    Returns:
        (a, b, c, fwhm) — model params and effective FWHM (mm).
    """
    from fastfuncstuff.stats.localstat import (
        _accumulate_bins,
        acf_fwhm_batched,
        build_neighborhood,
        fit_acf_batched,
    )

    if device is None:
        device = resid_vol.device
    resid_vol = resid_vol.to(device).float()
    mask = mask.to(device).to(torch.bool)
    nz = resid_vol.shape[1]

    nb = build_neighborhood(nbhd, voxdims)
    n_bins = int(nb.bin_radius.shape[0])
    nrm = torch.sqrt((resid_vol * resid_vol).sum(0))  # (nz,ny,nx)

    # One slab over the whole volume; per-voxel bins, then reduce over the mask.
    bin_sum, bin_cnt = _accumulate_bins(
        resid_vol, nrm, mask, nb, 0, 0, nz, n_bins, None
    )  # (K, nz, ny, nx)
    m = mask[None].float()
    gsum = (bin_sum * m).sum(dim=(1, 2, 3))  # (K,)
    gcnt = (bin_cnt * m).sum(dim=(1, 2, 3))
    y = (gsum / gcnt.clamp_min(1.0)).view(1, -1).double()
    w = (gcnt > 0).view(1, -1).double()

    radii = nb.bin_radius.to(device=device, dtype=torch.float64)
    a, b, c, _ok = fit_acf_batched(radii, y, w)
    fwhm = acf_fwhm_batched(a, b, c, 0.5)
    return float(a.item()), float(b.item()), float(c.item()), float(fwhm.item())


def fwhmx_report(
    residuals_2d: Tensor,
    voxel_mask: Tensor,
    run_starts: list[int],
    volume_shape: tuple[int, int, int],
    voxdims: tuple[float, float, float],
    nbhd: str = "SPHERE(-9.666)",
    device: torch.device | None = None,
) -> str:
    """Per-run ACF fit over the residuals → an AFNI-3dFWHMx-style text report.

    ``residuals_2d`` is (n_masked_voxels, n_time); ``voxel_mask`` places them back
    on the ``volume_shape`` grid. ``run_starts`` are time indices where each run
    begins (the last run runs to the end). One ACF fit per run plus a mean row.
    """
    if device is None:
        device = residuals_2d.device
    nz, ny, nx = volume_shape
    vmask = voxel_mask.reshape(volume_shape).to(device).to(torch.bool)
    n_time = residuals_2d.shape[1]
    bounds = list(run_starts) + [n_time]

    lines = [
        "# 3dFWHMx-style ACF: a*exp(-r*r/(2*b*b)) + (1-a)*exp(-r/c); FWHM in mm",
        "# run        a        b        c     FWHM",
    ]
    rows = []
    for r in range(len(run_starts)):
        t0, t1 = bounds[r], bounds[r + 1]
        if t1 <= t0:
            continue
        # Scatter the run's residual timepoints back into the volume grid.
        run_res = residuals_2d[:, t0:t1].to(device)
        vol = torch.zeros((t1 - t0, nz, ny, nx), dtype=torch.float32, device=device)
        vol[:, vmask] = run_res.T.float()
        a, b, c, fwhm = global_acf_fit(vol, vmask, voxdims, nbhd=nbhd, device=device)
        rows.append((a, b, c, fwhm))
        lines.append(f"  {r + 1:>3}  {a:7.4f}  {b:7.4f}  {c:7.4f}  {fwhm:7.3f}")

    if rows:
        arr = np.array(rows)
        m = arr.mean(0)
        lines.append(f"  avg  {m[0]:7.4f}  {m[1]:7.4f}  {m[2]:7.4f}  {m[3]:7.3f}")
    return "\n".join(lines) + "\n"


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
        self.maps["grandmean"] = self._to_vol(temporal_mean(data_2d))

    # -- hook 2: scaled data -------------------------------------------------
    def observe_scaled(self, data_2d: Tensor) -> None:
        """Raw tSNR = mean / std of the scaled timeseries (mean ≈ 100)."""
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
        nbhd: str = "SPHERE(-9.666)",
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
                std_resid = temporal_std(resid)  # (n_masked,)
                sig = self._scaled_mean.detach().cpu().float().numpy()[vm_flat]
                sig_t = torch.from_numpy(sig).to(std_resid.device)
                tvals = tsnr(sig_t, std_resid).detach().cpu().float().numpy()
                vol = np.zeros(self.volume_shape, dtype=np.float32)
                vol[np.asarray(vm.cpu())] = tvals
                self.maps[f"resid_tsnr_{label}"] = vol
            if want_fwhmx:
                self.tables[f"fwhmx_{label}"] = fwhmx_report(
                    resid,
                    voxel_mask,
                    self.run_starts,
                    self.volume_shape,
                    self.voxdims,
                    nbhd=nbhd,
                    device=self.device,
                )

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
