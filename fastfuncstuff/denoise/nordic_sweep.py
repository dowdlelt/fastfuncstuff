"""NORDIC single-echo factor-sweep residual-correlation diagnostic.

A within-image referee for the global ``-factor-error`` knob, for the common
single-echo case where the multi-echo cross-echo check is unavailable.

NORDIC removes thermal noise, which is spatially independent — so the residual
of a correctly-tuned run should have a voxel-to-voxel correlation matrix sitting
at the timepoint null (signed r centered on 0, |r| at the analytic floor, flat
with distance). Push the factor too high and real, locally-structured signal is
removed: it lands in the residual and the correlation distribution shifts away
from null with a near>far distance interaction.

This module sweeps the factor over a window, reconstructs the residual at each
(SVD computed **once**, then re-thresholded), and summarizes the residual-
magnitude correlation over in-brain / out-of-brain / whole-image masks via
``stats.voxel_correlation``. The two diagnostic figures let the user eyeball the
asymptote-then-liftoff behavior; the liftoff marks the ideal factor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from fastfuncstuff.denoise.nordic import (
    NordicConfig,
    _build_patch_starts,
    _prepare_echo,
)
from fastfuncstuff.io.afni import load_nifti
from fastfuncstuff.stats.voxel_correlation import (
    CorrSummary,
    analytic_r_null,
    corr_histogram_distance,
    voxel_corr_strength,
)
from fastfuncstuff.utils import get_device


@dataclass
class _SVDCache:
    """Per-patch SVD computed once, re-thresholded across the factor sweep."""

    s: torch.Tensor  # (P, K) singular values (descending)
    vh: torch.Tensor  # (P, K, N) right singular vectors (rows)
    corners_flat: torch.Tensor  # (P,) flat corner index into (nx*ny*nz)
    dxyz_flat: torch.Tensor  # (M,) flat per-patch voxel offsets
    xi: torch.Tensor  # (P, M) gather x-index (for re-extracting patches)
    yi: torch.Tensor  # (P, M)
    zi: torch.Tensor  # (P, M)
    shape: tuple[int, int, int, int]  # nx, ny, nz, nt
    m: int
    n: int
    k: int


def _cache_patch_svds(
    data: torch.Tensor,
    kernel_size: tuple[int, int, int],
    patch_overlap: int,
    svd_batch_size: int,
    verbose: bool,
    device: torch.device,
) -> _SVDCache:
    """One pass over patches: SVD each, store ``(s, Vh)`` for the factor sweep.

    Mirrors the patch grid / extraction of ``nordic._llr_denoise`` but keeps the
    decomposition instead of reconstructing, so the sweep only re-thresholds.
    """
    nx, ny, nz, nt = data.shape
    wx, wy, wz = min(kernel_size[0], nx), min(kernel_size[1], ny), min(kernel_size[2], nz)
    m = wx * wy * wz
    n = nt
    k = min(m, n)

    sx = max(1, wx // max(1, patch_overlap))
    sy = max(1, wy // max(1, patch_overlap))
    sz = max(1, wz // max(1, patch_overlap))
    xs = _build_patch_starts(nx, wx, sx)
    ys = _build_patch_starts(ny, wy, sy)
    zs = _build_patch_starts(nz, wz, sz)
    corners = [(x0, y0, z0) for x0 in xs for y0 in ys for z0 in zs]
    total = len(corners)

    _ox, _oy, _oz = torch.arange(wx), torch.arange(wy), torch.arange(wz)
    gx, gy, gz = torch.meshgrid(_ox, _oy, _oz, indexing="ij")
    dx, dy, dz = gx.ravel(), gy.ravel(), gz.ravel()
    dxyz_flat = (dx * (ny * nz) + dy * nz + dz).to(device)
    cxs = torch.tensor([c[0] for c in corners], dtype=torch.long)
    cys = torch.tensor([c[1] for c in corners], dtype=torch.long)
    czs = torch.tensor([c[2] for c in corners], dtype=torch.long)
    corners_flat = (cxs * (ny * nz) + cys * nz + czs).to(device)

    data_dev = data.device
    xi_all = (cxs[:, None] + dx[None, :]).to(data_dev)  # (P, M)
    yi_all = (cys[:, None] + dy[None, :]).to(data_dev)
    zi_all = (czs[:, None] + dz[None, :]).to(data_dev)

    s_out = torch.empty((total, k), dtype=torch.float32, device=device)
    vh_out = torch.empty((total, k, n), dtype=torch.complex64, device=device)

    pbar = tqdm(total=total, desc="SVD cache", unit="patch") if verbose else None
    for b0 in range(0, total, svd_batch_size):
        b1 = min(b0 + svd_batch_size, total)
        mats = data[xi_all[b0:b1], yi_all[b0:b1], zi_all[b0:b1], :]  # (B, M, N)
        if data_dev != device:
            mats = mats.to(device)
        _, s, vh = torch.linalg.svd(mats, full_matrices=False)
        s_out[b0:b1] = s.abs().to(torch.float32)
        vh_out[b0:b1] = vh
        del mats, s, vh
        if pbar is not None:
            pbar.update(b1 - b0)
    if pbar is not None:
        pbar.close()

    return _SVDCache(
        s=s_out,
        vh=vh_out,
        corners_flat=corners_flat,
        dxyz_flat=dxyz_flat,
        xi=xi_all,
        yi=yi_all,
        zi=zi_all,
        shape=(nx, ny, nz, nt),
        m=m,
        n=n,
        k=k,
    )


def _residual_magnitude_at_factor(
    cache: _SVDCache,
    data: torch.Tensor,
    lambda_threshold: float,
    svd_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Reconstruct the residual magnitude ``(nx*ny*nz, nt)`` at one threshold.

    Residual = ``X V_kill V_killᴴ`` per patch (the removed components), patch-
    weight-averaged over overlaps, then ``|.|``. No SVD here — the cached ``(s, Vh)``
    are simply re-thresholded, which is what makes the sweep cheap.
    """
    nx, ny, nz, nt = cache.shape
    nvox = nx * ny * nz
    recon = torch.zeros((nvox, nt), dtype=torch.complex64, device=device)
    weight = torch.zeros(nvox, dtype=torch.float32, device=device)
    data_dev = data.device
    total = cache.s.shape[0]

    for b0 in range(0, total, svd_batch_size):
        b1 = min(b0 + svd_batch_size, total)
        mats = data[cache.xi[b0:b1], cache.yi[b0:b1], cache.zi[b0:b1], :]  # (B, M, N)
        if data_dev != device:
            mats = mats.to(device)
        s = cache.s[b0:b1]  # (B, K)
        vh = cache.vh[b0:b1]  # (B, K, N)
        kill = (s < lambda_threshold).to(torch.complex64)  # below threshold = removed
        vh_kill = vh * kill[:, :, None]  # (B, K, N)
        proj = vh_kill.mH @ vh_kill  # (B, N, N) projector onto killed subspace
        resid = mats @ proj  # (B, M, N)
        del mats, vh_kill, proj

        flat_b = (cache.corners_flat[b0:b1][:, None] + cache.dxyz_flat[None, :]).reshape(-1)
        recon.index_add_(0, flat_b, resid.reshape(-1, nt))
        weight.index_add_(0, flat_b, torch.ones(flat_b.numel(), device=device))
        del resid, flat_b

    weight = weight.clamp(min=1.0)
    recon = recon / weight[:, None]
    return recon.abs()  # (nvox, nt) residual magnitude


def _build_masks(
    magnitude_file: str,
    shape: tuple[int, int, int, int],
    which: tuple[str, ...],
    device: torch.device,
    verbose: bool,
) -> dict[str, torch.Tensor]:
    """Flat boolean voxel masks: in-brain automask, whole image. (top_pairs is
    derived separately from the input correlation in run_nordic_factor_sweep.)"""
    nx, ny, nz, _ = shape
    masks: dict[str, torch.Tensor] = {}
    if "in_brain" in which or "top_pairs" in which:
        try:
            from fastfuncstuff.processing.mask import automask

            mag = np.abs(load_nifti(magnitude_file).get_fdata(dtype=np.float32))
            mag = mag.mean(-1) if mag.ndim == 4 else mag  # (nx, ny, nz)
            bm = automask(torch.from_numpy(mag).float(), dilate_extra=2, verbose=False)
            masks["in_brain"] = bm.to(torch.bool).reshape(-1).to(device)
        except Exception as exc:  # noqa: BLE001 — diagnostic is best-effort
            if verbose:
                print(f"  factor-sweep: automask failed ({exc}); using whole image only.")
    if "whole" in which:
        masks["whole"] = torch.ones(nx * ny * nz, dtype=torch.bool, device=device)
    return masks


def _subsample_mask(idx: torch.Tensor, max_voxels: int | None, seed: int) -> torch.Tensor:
    """Cap a voxel-index set to ``max_voxels`` via a fixed-seed random draw."""
    if max_voxels is None or idx.numel() <= max_voxels:
        return idx
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(idx.numel(), generator=g)[:max_voxels]
    return idx[perm.to(idx.device)]


def run_nordic_factor_sweep(
    magnitude_file: str,
    phase_file: str | None,
    output_prefix: str,
    config: NordicConfig | None = None,
    device: torch.device | None = None,
) -> dict:
    """Sweep ``-factor-error`` and summarize the residual voxel-to-voxel correlation.

    Writes ``{prefix}_factorsweep.json`` (full summary), ``.tsv`` (per-factor/mask
    scalars), and two PNG figures. Returns the summary dict.
    """
    cfg = config or NordicConfig()
    dev = (
        device if device is not None else get_device("cuda" if torch.cuda.is_available() else None)
    )

    factors = list(cfg.factor_sweep_values) if cfg.factor_sweep_values else _default_factors()
    factors = sorted(float(f) for f in factors)

    prepped = _prepare_echo(magnitude_file, phase_file, cfg, dev)
    if prepped.threshold_mode != "nordic":
        raise ValueError(
            "factor sweep requires the NORDIC threshold (mp_mode=0); "
            f"got mode={prepped.threshold_mode!r}."
        )
    # lambda at factor=1, so the sweep scales it directly (lambda is linear in factor).
    lambda_base = prepped.threshold_value / max(cfg.factor_error, 1e-12)

    data = prepped.ksp2
    assert data is not None
    shape = data.shape  # (nx, ny, nz, nt_keep)
    nx, ny, nz, nt = shape

    if cfg.verbose:
        print(f"\nNORDIC factor sweep: {len(factors)} factors {factors}")
        print(f"  lambda(factor=1) = {lambda_base:.6g}")

    cache = _cache_patch_svds(
        data,
        kernel_size=prepped.kernel_pca,
        patch_overlap=max(1, cfg.patch_overlap),
        svd_batch_size=cfg.svd_batch_size,
        verbose=cfg.verbose,
        device=dev,
    )

    masks = _build_masks(magnitude_file, shape, cfg.factor_sweep_masks, dev, cfg.verbose)
    if not masks:
        masks = {"whole": torch.ones(nx * ny * nz, dtype=torch.bool, device=dev)}

    # Flat ijk coordinates for distance binning.
    ii, jj, kk = torch.meshgrid(torch.arange(nx), torch.arange(ny), torch.arange(nz), indexing="ij")
    coords_flat = torch.stack([ii.ravel(), jj.ravel(), kk.ravel()], dim=1).to(dev)

    # Fixed voxel index sets per mask (subsampled once, reused across factors so
    # the curves are comparable point-to-point).
    mask_idx: dict[str, torch.Tensor] = {}
    for name, mflat in masks.items():
        idx = torch.nonzero(mflat, as_tuple=False).squeeze(1)
        idx = _subsample_mask(idx, cfg.factor_sweep_max_voxels, seed=hash(name) & 0xFFFF)
        mask_idx[name] = idx

    # top_pairs: the danger zone. Rank in-brain (else whole) voxels by their
    # correlation strength in the *input* (pre-denoise, noise vols already trimmed),
    # keep the top fraction — these are the shared-structure voxels whose pairs
    # over-removal would corrupt. Inserted between in_brain and whole.
    if "top_pairs" in cfg.factor_sweep_masks:
        base = mask_idx.get("in_brain")
        if base is None:
            base = torch.arange(nx * ny * nz, device=dev)
        strength = voxel_corr_strength(data.reshape(-1, nt)[base].abs())
        k = min(base.numel(), max(2, int(round(cfg.factor_sweep_top_frac * base.numel()))))
        if cfg.factor_sweep_max_voxels:
            k = min(k, cfg.factor_sweep_max_voxels)
        top = base[torch.topk(strength, k).indices]
        mask_idx = {
            "in_brain": mask_idx.get("in_brain"),
            "top_pairs": top,
            "whole": mask_idx.get("whole"),
        }
        mask_idx = {n: i for n, i in mask_idx.items() if i is not None}

    # Input (non-denoised) in-brain magnitude — fixed across factors. The residual
    # correlation structure is compared to this each factor (struct_corr_to_ref);
    # in-brain only (the global structure of interest, and one extra GEMM there).
    data_flat = data.reshape(-1, nt)
    ref_in_brain = data_flat[mask_idx["in_brain"]].abs() if "in_brain" in mask_idx else None

    null = analytic_r_null(nt)
    results: dict[str, dict] = {name: {} for name in mask_idx}

    for f in tqdm(factors, desc="factor sweep", unit="factor", disable=not cfg.verbose):
        resid_flat = _residual_magnitude_at_factor(
            cache, data, lambda_base * f, cfg.svd_batch_size, dev
        )
        for name, idx in mask_idx.items():
            results[name][f] = _corr_or_none(
                resid_flat[idx],
                coords_flat[idx],
                cfg.factor_sweep_r_bins,
                cfg.factor_sweep_dist_bins,
                ref_ts=ref_in_brain if name == "in_brain" else None,
            )
        del resid_flat

    summary = _assemble_summary(results, factors, null, nt, lambda_base, cfg)
    out_paths = _write_outputs(summary, results, factors, null, output_prefix, cfg)
    summary["outputs"] = out_paths
    if cfg.verbose:
        _print_liftoff(summary)
    return summary


def _corr_or_none(
    ts: torch.Tensor,
    coords: torch.Tensor,
    r_bins: int,
    n_dist_bins: int,
    ref_ts: torch.Tensor | None = None,
) -> CorrSummary | None:
    """Correlation summary, or None when the residual is degenerate.

    At a low factor the threshold can drop below every patch singular value, so
    NORDIC removes nothing and the residual is identically zero (the under-removal
    extreme). With no — or <2 — non-constant voxels there is nothing to correlate;
    we record a gap rather than crash, which on the plot reads as "nothing removed
    yet" at that factor.
    """
    if ts.shape[0] < 2:
        return None
    try:
        return corr_histogram_distance(
            ts, coords, r_bins=r_bins, n_dist_bins=n_dist_bins, ref_ts=ref_ts
        )
    except ValueError:
        return None


def _default_factors() -> list[float]:
    """Factor window: 9 evenly-spaced steps over 0.75–1.25, symmetric about — and
    landing exactly on — 1.0 (the calibrated factor)."""
    return [round(float(f), 4) for f in np.linspace(0.75, 1.25, 9)]


def _assemble_summary(
    results: dict[str, dict[float, CorrSummary]],
    factors: list[float],
    null: dict[str, float],
    nt: int,
    lambda_base: float,
    cfg: NordicConfig,
) -> dict:
    nan = float("nan")

    def _g(s: CorrSummary | None, attr: str) -> float:
        v = nan if s is None else getattr(s, attr)
        return nan if v is None else float(v)

    per_mask: dict[str, dict] = {}
    for name, byf in results.items():
        per_mask[name] = {
            "factor": factors,
            "mean_r": [_g(byf[f], "mean_r") for f in factors],
            "mean_abs_r": [_g(byf[f], "mean_abs_r") for f in factors],
            "median_r": [_g(byf[f], "median_r") for f in factors],
            "std_r": [_g(byf[f], "std_r") for f in factors],
            "n_voxels": [0 if byf[f] is None else byf[f].n_voxels for f in factors],
            "near_minus_far_abs_r": [_near_minus_far(byf[f]) for f in factors],
            "struct_corr_to_input": [_g(byf[f], "struct_corr_to_ref") for f in factors],
        }
    summary = {
        "magnitude": None,
        "n_timepoints": int(nt),
        "lambda_base": float(lambda_base),
        "factor_current": float(cfg.factor_error),
        "factors": factors,
        "null": null,
        "masks": per_mask,
        "suggested_factor": _suggest_factor(per_mask, null),
    }
    return summary


def _near_minus_far(summ: CorrSummary | None) -> float:
    """Distance interaction scalar: mean |r| in the nearest populated distance
    bin minus the far half. >~0 means local structure in the residual."""
    if summ is None:
        return float("nan")
    cnt = summ.dist_count
    pop = cnt > 0
    if pop.sum() < 2:
        return 0.0
    mar = summ.dist_mean_abs_r
    near = float(mar[pop][0].item())
    far_half = mar[pop][len(mar[pop]) // 2 :]
    far = float(far_half.mean().item())
    return near - far


def _suggest_factor(per_mask: dict, null: dict) -> dict:
    """Liftoff factor: the largest factor at which in-brain mean |r| is still
    within the null band — i.e. before signal starts entering the residual."""
    name = "in_brain" if "in_brain" in per_mask else next(iter(per_mask))
    d = per_mask[name]
    band = null["mean_abs_r"] + null["ci95_r"]
    liftoff = None
    for f, mabs in zip(d["factor"], d["mean_abs_r"], strict=True):
        if mabs == mabs and mabs <= band:  # skip NaN (nothing removed at f)
            liftoff = f
    return {"mask": name, "null_abs_r": null["mean_abs_r"], "band": band, "liftoff_factor": liftoff}


def _print_liftoff(summary: dict) -> None:
    s = summary["suggested_factor"]
    lf = s["liftoff_factor"]
    msg = (
        f"  Factor sweep ({s['mask']}): null mean|r|={s['null_abs_r']:.3f}; "
        f"residual stays at null up to factor "
    )
    msg += f"~{lf}" if lf is not None else "(< smallest tested)"
    print(msg + " before correlation lifts off (= signal entering the residual).")


def _write_outputs(
    summary: dict,
    results: dict[str, dict[float, CorrSummary]],
    factors: list[float],
    null: dict[str, float],
    output_prefix: str,
    cfg: NordicConfig,
) -> dict:
    out_prefix = Path(output_prefix)
    out_dir = out_prefix.parent if out_prefix.parent != Path("") else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_prefix.name

    json_path = out_dir / f"{stem}_factorsweep.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in summary.items() if k != "outputs"}, fh, indent=2)

    tsv_path = out_dir / f"{stem}_factorsweep.tsv"
    with open(tsv_path, "w", encoding="utf-8") as fh:
        fh.write(
            "mask\tfactor\tmean_r\tmean_abs_r\tmedian_r\tstd_r\t"
            "near_minus_far_abs_r\tstruct_corr_to_input\tn_voxels\n"
        )
        for name, d in summary["masks"].items():
            for i, f in enumerate(d["factor"]):
                fh.write(
                    f"{name}\t{f}\t{d['mean_r'][i]:.6f}\t{d['mean_abs_r'][i]:.6f}\t"
                    f"{d['median_r'][i]:.6f}\t{d['std_r'][i]:.6f}\t"
                    f"{d['near_minus_far_abs_r'][i]:.6f}\t{d['struct_corr_to_input'][i]:.6f}\t"
                    f"{d['n_voxels'][i]}\n"
                )

    fig_paths = _make_plots(summary, results, factors, null, out_dir, stem)
    return {"json": str(json_path), "tsv": str(tsv_path), **fig_paths}


def _make_plots(
    summary: dict,
    results: dict[str, dict[float, CorrSummary]],
    factors: list[float],
    null: dict[str, float],
    out_dir: Path,
    stem: str,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    masks = summary["masks"]
    colors = {"in_brain": "C0", "top_pairs": "C3", "whole": "C2"}

    # Null reported in the titles (text), not as an in-plot line: the residual
    # |r| hugs the null, so a null axhline sits right on the data and flattens the
    # axis, hiding the small factor-to-factor structure that is the whole point.
    nv = null["mean_abs_r"]

    # Fig 1: summary vs factor (the asymptote-then-liftoff curve).
    fig1, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
    for name, d in masks.items():
        c = colors.get(name, None)
        axes[0].plot(d["factor"], d["mean_abs_r"], "-o", color=c, label=name)
        axes[1].plot(d["factor"], d["near_minus_far_abs_r"], "-o", color=c, label=name)
    axes[0].set(
        xlabel="factor_error",
        ylabel="mean |r| (residual)",
        title=f"Residual correlation vs factor  (null E|r|={nv:.3f})",
    )
    axes[0].legend(fontsize=8)
    axes[1].axhline(0.0, ls="--", color="k", lw=1)
    axes[1].set(
        xlabel="factor_error",
        ylabel="near − far mean |r|",
        title="Distance interaction vs factor  (null = 0)",
    )
    axes[1].legend(fontsize=8)
    # Panel 3: how much the in-brain residual's pairwise-correlation structure
    # resembles the input's (one number per factor). Rises as over-removal drags
    # global structure into the residual — a whole-matrix effect the histogram dilutes.
    ib = masks.get("in_brain")
    if ib is not None:
        axes[2].plot(ib["factor"], ib["struct_corr_to_input"], "-o", color=colors["in_brain"])
    axes[2].axhline(0.0, ls="--", color="k", lw=1)
    axes[2].set(
        xlabel="factor_error",
        ylabel="Pearson r (residual △ vs input △)",
        title="Residual-vs-input structure (in_brain)",
    )
    fig1.tight_layout()
    p1 = out_dir / f"{stem}_factorsweep_summary.png"
    fig1.savefig(p1, dpi=120)
    plt.close(fig1)

    # Fig 2: distributions — signed-r histogram + correlation-vs-distance. Prefer
    # top_pairs (the danger zone) so we zoom into where over-removal shows first.
    name = next((m for m in ("top_pairs", "in_brain") if m in results), next(iter(results)))
    byf = results[name]
    cmap = plt.get_cmap("viridis")
    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, f in enumerate(factors):
        col = cmap(i / max(1, len(factors) - 1))
        s = byf[f]
        if s is None:  # nothing removed at this factor — no distribution to draw
            continue
        centers = 0.5 * (s.r_edges[:-1] + s.r_edges[1:])
        hist = s.r_hist / max(1.0, float(s.r_hist.sum()))
        ax2[0].plot(centers.numpy(), hist.numpy(), color=col, lw=1, label=f"f={f}")
        ax2[1].plot(s.dist_centers.numpy(), s.dist_mean_abs_r.numpy(), color=col, lw=1)
    ax2[0].axvline(0.0, ls="--", color="k", lw=1)
    ax2[0].set(
        xlabel="Pearson r",
        ylabel="fraction",
        title=f"Residual r histogram — {name} (null r=0)",
    )
    ax2[0].set_xlim(-0.6, 0.6)
    ax2[0].legend(fontsize=7, ncol=2)
    ax2[1].set(
        xlabel="voxel distance",
        ylabel="mean |r|",
        title=f"Correlation vs distance — {name} (null E|r|={nv:.3f})",
    )
    fig2.tight_layout()
    p2 = out_dir / f"{stem}_factorsweep_distributions.png"
    fig2.savefig(p2, dpi=120)
    plt.close(fig2)

    return {"plot_summary": str(p1), "plot_distributions": str(p2)}
