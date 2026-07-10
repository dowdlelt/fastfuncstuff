"""3dFWHMx-faithful whole-volume smoothness estimation (classic + ACF).

This mirrors AFNI ``3dFWHMx`` (``mri_fwhm.c``), which estimates the *overall*
spatial blurring of an image, NOT the per-voxel local ACF that ``3dLocalACF`` /
:mod:`fastfuncstuff.stats.localstat` computes. The two are easy to confuse but
use different sample dimensions:

* ``3dLocalACF`` correlates neighbouring voxels **across time** (per-voxel-pair
  temporal correlation) and produces a spatially-varying map.
* ``3dFWHMx`` correlates neighbouring voxels **across space** within each
  sub-brick, normalised by that sub-brick's spatial variance, then **averages
  the ACF over sub-bricks**. One (a, b, c, FWHM) per dataset (here: per run).

For residual smoothness / cluster-correction (``3dClustSim -acf``) the FWHMx
estimator is the correct one, so ``ffs_reml -save_acf`` uses this module.

Faithful details replicated from ``mri_fwhm.c``:

* Classic Forman 1-difference FWHM per axis (``mri_estimate_FWHM_1dif``), used
  both as a reported diagnostic and to set the default ACF radius.
* Default ACF radius ``max(2.999 * combined_FWHM, 3.999 * cbrt(dx·dy·dz))``
  (``3dFWHMx.c``); geometric-mean combination over axes and sub-bricks.
* Spatial ACF ``mag[δ] = mean_t[ Σ_x (v-x̄)(v(x+δ)-x̄) / (fvar_t·(n_δ-1)) ]``
  (``mri_estimate_ACF`` + ``THD_estimate_ACF``), sphere cluster subsampled to
  ``NCLU_GOAL`` points like ``get_ACF_cluster``.
* Model ``a·exp(-r²/2b²)+(1-a)·exp(-r/c)`` fit (``ACF_cluster_to_modelE``) —
  reuses the batched Powell/LM fitter in :mod:`localstat`.

Memory: sub-bricks are streamed in timepoint chunks sized by the Memory module,
so peak VRAM is a few 3-D volumes rather than the whole 4-D residual stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from fastfuncstuff.memory import get_available_memory
from fastfuncstuff.stats.localstat import (
    acf_fwhm_batched,
    build_neighborhood,
    fit_acf_batched,
)

# AFNI get_ACF_cluster point budget (mri_fwhm.c).
_NCLU_BASE = 111
_NCLU_GOAL = 666
_S2F = 2.35482  # sqrt(8 ln 2), sigma -> FWHM


@dataclass
class FWHMxResult:
    """Per-run smoothness estimate (see :func:`estimate_fwhmx_run`)."""

    a: float
    b: float
    c: float
    fwhm: float  # ACF effective FWHM (mm)
    radius: float  # ACF radius actually used (mm)
    classic_fwhm: tuple[float, float, float]  # Forman 1-diff (x, y, z) mm
    classic_combined: float  # geometric-mean combined classic FWHM (mm)
    n_subbricks: int  # timepoints that contributed


# ---------------------------------------------------------------------------
# Classic Forman 1-difference FWHM (mri_estimate_FWHM_1dif), batched over time
# ---------------------------------------------------------------------------


def _spatial_mean_demean(vol: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor, int]:
    """Return (mean_t, demeaned_masked_vol, fvar_t, count).

    ``vol`` is (T, Z, Y, X); ``mask`` is (Z, Y, X) bool. The demeaned volume is
    zeroed outside the mask so cross-products with out-of-mask neighbours vanish
    (AFNI only accumulates GOOD-GOOD pairs). ``fvar_t`` is the per-sub-brick
    spatial variance over the mask, exactly ``(Σf² - (Σf)²/n)/(n-1)``.
    """
    count = int(mask.sum())
    m = mask.unsqueeze(0)  # (1, Z, Y, X)
    fsum = (vol * m).sum(dim=(1, 2, 3), dtype=torch.float64)  # (T,)
    mean = fsum / max(count, 1)
    vd = (vol - mean.view(-1, 1, 1, 1).to(vol.dtype)) * m
    # float64 accumulation (not a full float64 copy of vd -- that doubles VRAM).
    fvar = (vd * vd).sum(dim=(1, 2, 3), dtype=torch.float64) / max(count - 1, 1)
    return mean, vd, fvar, count


def _axis_diff_var(vol: Tensor, mask: Tensor, axis: int) -> Tensor:
    """Per-sub-brick variance of the forward first difference along ``axis``.

    ``axis`` is a spatial axis of ``vol`` (1=z, 2=y, 3=x); the corresponding
    mask axis is ``axis-1``. Matches ``mri_estimate_FWHM_1dif``: a difference is
    counted only when both voxels are in the mask; variance uses the same
    ``(Σd² - (Σd)²/n)/(n-1)`` form and returns 0 when fewer than 6 pairs.
    """
    lo = [slice(None)] * 4
    hi = [slice(None)] * 4
    lo[axis] = slice(0, -1)
    hi[axis] = slice(1, None)
    d = vol[tuple(hi)] - vol[tuple(lo)]  # (T, ...)

    mlo = [slice(None)] * 3
    mhi = [slice(None)] * 3
    mlo[axis - 1] = slice(0, -1)
    mhi[axis - 1] = slice(1, None)
    valid = (mask[tuple(mhi)] & mask[tuple(mlo)]).unsqueeze(0)  # (1, ...)
    cnt = int(valid.sum())
    if cnt < 6:
        return torch.zeros(vol.shape[0], dtype=torch.float64, device=vol.device)
    # float64 accumulation over float32 differences (no float64 volume copy).
    # Differences of a smooth field are ~zero-mean, so dsum**2/cnt << dsq and the
    # subtraction is well-conditioned.
    dv = d * valid  # float32
    dsum = dv.sum(dim=(1, 2, 3), dtype=torch.float64)
    dsq = (dv * dv).sum(dim=(1, 2, 3), dtype=torch.float64)
    return (dsq - dsum * dsum / cnt) / (cnt - 1)


def _forman_fwhm(var: Tensor, dvar: Tensor, vox_mm: float) -> Tensor:
    """FWHM per sub-brick from noise/derivative variance (Forman formula).

    ``arg = 1 - 0.5*dvar/var``; ``FWHM = 2.35482*sqrt(-1/(4 ln arg))*vox`` where
    ``0 < arg < 1``, else NaN (invalid, dropped from the geometric mean).
    """
    arg = 1.0 - 0.5 * dvar / var.clamp_min(1e-30)
    ok = (arg > 0.0) & (arg < 1.0)
    fwhm = torch.full_like(arg, float("nan"))
    safe = arg.clamp(1e-12, 1 - 1e-12)
    fwhm[ok] = _S2F * torch.sqrt(-1.0 / (4.0 * torch.log(safe[ok]))) * vox_mm
    return fwhm


# ---------------------------------------------------------------------------
# ACF cluster (get_ACF_cluster) + radius binning (ACF_cluster_to_modelE)
# ---------------------------------------------------------------------------


def _acf_cluster(radius_mm: float, voxdims: tuple[float, float, float]) -> tuple[Tensor, Tensor]:
    """Sphere offsets + per-offset radii for the ACF, subsampled like AFNI.

    ``build_neighborhood`` gives the full ``MCW_spheremask`` (radius-sorted,
    centre first). ``get_ACF_cluster`` caps the count at ``NCLU_GOAL`` by keeping
    the central ``NCLU_BASE`` points and striding the outer shell; replicated
    here so the offset loop stays cheap on large radii.
    """
    nb = build_neighborhood(f"SPHERE({radius_mm})", voxdims)
    offs, radii = nb.offsets, nb.radii  # (P,3), (P,) sorted ascending, centre @0
    p = offs.shape[0]
    if p <= _NCLU_GOAL:
        return offs, radii
    dp = max(1, int(round((p - _NCLU_BASE) / (_NCLU_GOAL - _NCLU_BASE))))
    outer = torch.arange(_NCLU_BASE, p, dp)
    keep = torch.cat([torch.arange(_NCLU_BASE), outer])
    return offs[keep], radii[keep]


def _collapse_radius_bins(radii: Tensor, mag: Tensor) -> tuple[Tensor, Tensor]:
    """Collapse offsets whose radii agree within 1% (ACF_cluster_to_modelE).

    Averages the per-offset ACF magnitudes within each group; the group radius
    is its smallest member. Returns (bin_radii, bin_mag), both 1-D float64.
    """
    order = torch.argsort(radii)
    r = radii[order].tolist()
    y = mag[order].tolist()
    br: list[float] = []
    bm: list[float] = []
    i, n = 0, len(r)
    while i < n:
        thr = r[i] * 1.01
        j = i + 1
        while j < n and r[j] <= thr:
            j += 1
        br.append(r[i])
        bm.append(float(np.mean(y[i:j])))
        i = j
    return (
        torch.tensor(br, dtype=torch.float64),
        torch.tensor(bm, dtype=torch.float64),
    )


# ---------------------------------------------------------------------------
# Main entry: one run's residuals -> FWHMxResult
# ---------------------------------------------------------------------------


def estimate_fwhmx_run(
    resid_2d: Tensor,
    voxel_mask: Tensor,
    volume_shape: tuple[int, int, int],
    voxdims: tuple[float, float, float],
    *,
    radius_mm: float | None = None,
    device: torch.device | None = None,
    progress: bool = True,
    progress_desc: str = "FWHMx",
) -> FWHMxResult:
    """3dFWHMx classic + ACF estimate for a single run.

    Args:
        resid_2d: (n_masked, n_time) residual timeseries; rows correspond to the
            True voxels of ``voxel_mask`` in row-major order.
        voxel_mask: bool (or 0/1) mask on ``volume_shape`` the residual rows
            live on.
        volume_shape: ``(s0, s1, s2)`` grid, in the same axis order the voxel
            index is flattened (e.g. NIfTI ``(nx, ny, nz)``).
        voxdims: per-axis voxel widths (mm), aligned to ``volume_shape`` —
            ``voxdims[i]`` is the width of ``volume_shape[i]``.
        radius_mm: ACF radius; None → AFNI's data-driven default.
        progress: show a tqdm bar over timepoint chunks and the offset loop.

    Returns:
        :class:`FWHMxResult`. ``classic_fwhm`` is per axis, in ``volume_shape``
        order.
    """
    if device is None:
        device = resid_2d.device if resid_2d.is_cuda else torch.device("cpu")
    s0, s1, s2 = (int(s) for s in volume_shape)
    v0, v1, v2 = (abs(float(v)) for v in voxdims)  # widths aligned to axes 0,1,2
    mask = voxel_mask.reshape(volume_shape).to(device).to(torch.bool)
    n_time = int(resid_2d.shape[1])

    # Timepoint chunk sized to fit a handful of float32 working volumes (vol,
    # demeaned copy, transient products). Diagnostics run AFTER the fit, so on
    # CUDA size against the driver's real free memory (mem_get_info), not
    # total-reserved -- other resident tensors would otherwise be ignored and
    # OOM the chunk. ~6 planes/timepoint covers the working copies.
    plane_bytes = s0 * s1 * s2 * 4
    avail = _free_bytes(device)
    tchunk = max(1, min(n_time, int(avail // max(1, plane_bytes * 6))))

    # First pass over chunks: classic FWHM (Forman) to fix the ACF radius.
    # Volume axes 1/2/3 correspond to grid axes 0/1/2 with widths v0/v1/v2;
    # per-axis FWHMs are combined by geometric mean over sub-bricks.
    log_s = [0.0, 0.0, 0.0]
    n_s = [0, 0, 0]
    n_bricks = 0

    chunks = list(range(0, n_time, tchunk))
    bar1 = tqdm(
        chunks,
        desc=f"{progress_desc}: classic",
        leave=False,
        disable=not progress or len(chunks) < 2,
    )
    for t0 in bar1:
        t1 = min(n_time, t0 + tchunk)
        vol = _scatter(resid_2d[:, t0:t1], mask, (t1 - t0, s0, s1, s2), device)
        _, _, fvar, _count = _spatial_mean_demean(vol, mask)
        for ax, width in ((1, v0), (2, v1), (3, v2)):
            arr = _forman_fwhm(fvar, _axis_diff_var(vol, mask, ax), width)
            good = arr[torch.isfinite(arr) & (arr > 0)]
            if good.numel():
                log_s[ax - 1] += float(torch.log(good).sum().item())
                n_s[ax - 1] += good.numel()
        n_bricks += int((fvar > 0).sum().item())
        del vol

    classic = tuple(float(np.exp(log_s[i] / n_s[i])) if n_s[i] else 0.0 for i in range(3))
    combined = _geom_combine(*classic)

    if radius_mm is None:
        radius_mm = max(2.999 * combined, 3.999 * float(np.cbrt(v0 * v1 * v2)))

    # build_neighborhood maps its returned offset columns to (voxdims[2],
    # voxdims[1], voxdims[0]); pass widths reversed so column c aligns to our
    # grid axis c (and the per-offset radii use the matching width).
    offsets, radii_off = _acf_cluster(radius_mm, (v2, v1, v0))
    offsets = offsets.to(device)
    n_off = offsets.shape[0]

    # nacf[δ]: number of GOOD-GOOD pairs at each offset (constant over time).
    nacf = torch.zeros(n_off, dtype=torch.float64, device=device)
    s_acc = torch.zeros(n_off, dtype=torch.float64, device=device)  # Σ_t dot/fvar
    offs_list = offsets.tolist()

    bar2 = tqdm(
        total=len(chunks) * n_off,
        desc=f"{progress_desc}: ACF",
        leave=False,
        disable=not progress or len(chunks) * n_off < 200,
    )
    nout = 0
    first = True
    for t0 in chunks:
        t1 = min(n_time, t0 + tchunk)
        vol = _scatter(resid_2d[:, t0:t1], mask, (t1 - t0, s0, s1, s2), device)
        _, vd, fvar, _count = _spatial_mean_demean(vol, mask)
        del vol
        valid_t = fvar > 0
        inv_fvar = torch.where(valid_t, 1.0 / fvar.clamp_min(1e-30), torch.zeros_like(fvar))
        nout += int(valid_t.sum().item())
        for oi, (o0, o1, o2) in enumerate(offs_list):
            a0, a1 = max(0, -o0), s0 - max(0, o0)
            b0, b1 = max(0, -o1), s1 - max(0, o1)
            c0, c1 = max(0, -o2), s2 - max(0, o2)
            center = vd[:, a0:a1, b0:b1, c0:c1]
            shifted = vd[:, a0 + o0 : a1 + o0, b0 + o1 : b1 + o1, c0 + o2 : c1 + o2]
            # float32 spatial reduction (~2.5x faster than float64 on consumer
            # GPUs, and the hot inner op). vd is already demeaned so there is no
            # catastrophic cancellation; torch's tree reduction keeps the error
            # far below the FWHM tolerance. Only the tiny (T,) accumulate is f64.
            dot = (center * shifted).sum(dim=(1, 2, 3))
            s_acc[oi] += (dot.to(torch.float64) * inv_fvar).sum()
            if first:
                mc = mask[a0:a1, b0:b1, c0:c1]
                ms = mask[a0 + o0 : a1 + o0, b0 + o1 : b1 + o1, c0 + o2 : c1 + o2]
                nacf[oi] = float((mc & ms).sum().item())
            bar2.update(1)
        first = False
        del vd
    bar2.close()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # mag[δ] = mean_t[ dot_t / (fvar_t (n_δ-1)) ]; centre (δ=0) yields 1 exactly.
    denom = (nacf - 1.0).clamp_min(1.0) * max(nout, 1)
    mag = (s_acc / denom).cpu()
    bin_r, bin_y = _collapse_radius_bins(radii_off, mag)

    a, b, c, fwhm = _fit_model(bin_r, bin_y, device)
    return FWHMxResult(
        a=a,
        b=b,
        c=c,
        fwhm=fwhm,
        radius=float(radius_mm),
        classic_fwhm=classic,
        classic_combined=combined,
        n_subbricks=n_bricks,
    )


def _free_bytes(device: torch.device) -> int:
    """Usable bytes for chunking, halved for the caching allocator's overhead.

    On CUDA use the driver's real free figure (``mem_get_info``) so tensors from
    the preceding fit -- or another process -- are accounted for; fall back to
    the Memory module elsewhere.
    """
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
            free, _total = torch.cuda.mem_get_info(device)
            return int(free * 0.5)
        except Exception:
            pass
    return get_available_memory(device)


def _scatter(chunk: Tensor, mask: Tensor, shape: tuple[int, int, int, int], device) -> Tensor:
    """Place (n_masked, T) residual columns onto a (T, Z, Y, X) grid."""
    vol = torch.zeros(shape, dtype=torch.float32, device=device)
    vol[:, mask] = chunk.to(device).float().T
    return vol


def _geom_combine(cx: float, cy: float, cz: float) -> float:
    """Geometric mean over the positive per-axis FWHMs (3dFWHMx default -geom)."""
    vals = [v for v in (cx, cy, cz) if v > 0]
    if not vals:
        return 0.0
    return float(np.exp(np.mean(np.log(vals))))


def _fit_model(bin_r: Tensor, bin_y: Tensor, device) -> tuple[float, float, float, float]:
    """Fit ACF(r)=a e^{-r²/2b²}+(1-a)e^{-r/c} and return (a, b, c, FWHM)."""
    if bin_r.numel() < 5:
        return -1.0, -1.0, -1.0, -1.0
    radii = bin_r.to(device=device, dtype=torch.float64)
    y = bin_y.to(device=device, dtype=torch.float64).view(1, -1)
    w = torch.ones_like(y)
    a, b, c, _ok = fit_acf_batched(radii, y, w)
    fwhm = acf_fwhm_batched(a, b, c, 0.5)
    return float(a.item()), float(b.item()), float(c.item()), float(fwhm.item())
