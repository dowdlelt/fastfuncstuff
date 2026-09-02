"""Reusable workflow helpers for ICA CLIs.

These utilities centralize common processing sections that were previously
implemented inline in CLI scripts.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from . import postprocess as ica_postprocess
from .tools import apply_high_pass_fft, apply_polort_projection
from .varnorm import variance_normalize


def verbose_section(verbose: bool, name: str) -> None:
    """Print a standardized verbose section header."""
    if not verbose:
        return
    print(f"\n  ── {name} {'─' * max(1, 50 - len(name))}")


def verbose_print(verbose: bool, msg: str, t0: float | None = None) -> None:
    """Print a verbose message, optionally annotated with elapsed time."""
    if not verbose:
        return
    if t0 is not None:
        elapsed = time.time() - t0
        print(f"    {msg} [{elapsed:.2f}s]")
    else:
        print(f"    {msg}")


@torch.inference_mode()
def sanitize_finite_tensor(t: torch.Tensor, label: str, verbose: bool = False) -> torch.Tensor:
    """Replace NaN/Inf with 0 and report if needed.

    Uses a chunked scan to avoid allocating a full-sized boolean mask
    (which can OOM on large tensors when .sum() upcasts to int64).
    """
    # Scan rows in chunks to keep peak memory low
    _ROWS_PER_CHUNK = max(1, min(t.shape[0], 4096))
    n_bad = 0
    has_bad = False
    for r0 in range(0, t.shape[0], _ROWS_PER_CHUNK):
        r1 = min(r0 + _ROWS_PER_CHUNK, t.shape[0])
        chunk_bad = ~torch.isfinite(t[r0:r1])
        chunk_n = int(chunk_bad.sum().item())
        if chunk_n > 0:
            if not has_bad:
                t = t.clone()
                has_bad = True
            t[r0:r1][chunk_bad] = 0.0
            n_bad += chunk_n
        del chunk_bad
    if n_bad > 0:
        print(f"  ⚠ {label}: {n_bad:,} NaN/Inf values → zeroed")
    elif verbose:
        print(f"    {label}: finite ✓")
    return t


def estimate_spatial_smoothness_resels(
    data_4d: np.ndarray,
    mask: np.ndarray | None = None,
    device: torch.device | None = None,
    verbose: bool = False,
) -> tuple[float, float]:
    """Estimate spatial smoothness using MELODIC/FSL-compatible resel logic."""
    n_t = data_4d.shape[-1]
    shape3d = data_4d.shape[:3]

    if device is None:
        device = torch.device("cpu")

    mean_t = data_4d.mean(axis=-1)
    std_t = np.std(data_4d, axis=-1, ddof=1)

    valid = std_t > 1e-10
    if mask is not None:
        valid = valid & mask
    std_safe = np.where(valid, std_t, 1.0)
    del std_t

    mask_t = torch.as_tensor(valid, device=device, dtype=torch.bool)

    chunk_size = max(1, min(n_t, 50))
    ssminus = [0.0, 0.0, 0.0]
    s2 = [0.0, 0.0, 0.0]

    n_chunks = (n_t + chunk_size - 1) // chunk_size
    for t_start in tqdm(
        range(0, n_t, chunk_size),
        total=n_chunks,
        desc="  FWHM estimate",
        leave=True,
        disable=n_chunks <= 1,
    ):
        t_end = min(t_start + chunk_size, n_t)
        chunk_np = np.empty((t_end - t_start, *shape3d), dtype=np.float32)
        for i, ti in enumerate(range(t_start, t_end)):
            chunk_np[i] = (data_4d[..., ti] - mean_t) / std_safe
        chunk_np[:, ~valid] = 0.0
        r = torch.as_tensor(chunk_np, device=device, dtype=torch.float32)
        del chunk_np

        for ax in range(3):
            dim = ax + 1
            r_cur = r.narrow(dim, 1, r.shape[dim] - 1)
            r_prev = r.narrow(dim, 0, r.shape[dim] - 1)

            sl_cur = [slice(None)] * 3
            sl_prev = [slice(None)] * 3
            sl_cur[ax] = slice(1, None)
            sl_prev[ax] = slice(None, -1)
            m = mask_t[tuple(sl_cur)] & mask_t[tuple(sl_prev)]
            m = m.unsqueeze(0)

            ssminus[ax] += float((r_cur * r_prev * m).sum().item())
            s2[ax] += float((0.5 * (r_cur**2 + r_prev**2) * m).sum().item())

        del r

    del mask_t, mean_t, std_safe, valid
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fwhm = []
    for ax in range(3):
        if s2[ax] < 1e-15:
            fwhm.append(1.0)
            continue
        rval = ssminus[ax] / s2[ax]
        rval = min(abs(rval), 0.99999)
        if rval < 1e-10:
            fwhm.append(1.0)
            continue
        sigmasq = -1.0 / (4.0 * np.log(rval))
        fwhm_ax = float(np.sqrt(8.0 * np.log(2.0) * sigmasq))
        fwhm.append(fwhm_ax)

    if verbose:
        verbose_print(
            True, f"  FWHM per axis: X={fwhm[0]:.3f}, Y={fwhm[1]:.3f}, Z={fwhm[2]:.3f} voxels"
        )

    resels = fwhm[0] * fwhm[1] * fwhm[2]
    fwhm_geo = float(np.cbrt(resels))
    return resels, fwhm_geo


def apply_voxel_variance_normalization(
    data_vox_t: torch.Tensor,
    num_spec: int | float | str,
    n_t: int,
    n_vox_masked: int,
    trace_dir: Path | None = None,
) -> tuple[torch.Tensor, str]:
    """Apply voxel variance normalisation; residual-noise path for automatic model order.

    Automatic model order reads the eigenspectrum, so the scale each voxel is put on
    directly determines the answer -- hence the residual-noise estimate there rather than
    the cheaper total-stdev divide. See :mod:`fastfuncstuff.decomposition.varnorm`.
    """
    if isinstance(num_spec, str) and num_spec in {"auto", "laplace"}:
        data_vox_t, n_const = variance_normalize(data_vox_t)
        norm_msg = (
            f"Voxel-norm: residual-noise varnorm over {n_vox_masked:,} voxels "
            f"({n_const} constant voxels zeroed)"
        )
        if trace_dir is not None:
            import numpy as _np

            trace_dir.mkdir(parents=True, exist_ok=True)
            _np.save(str(trace_dir / "migp_post_varnorm.npy"), data_vox_t.T.cpu().numpy())
        return data_vox_t, norm_msg

    voxel_std = torch.std(data_vox_t, dim=1, keepdim=True)
    const_mask = voxel_std.squeeze() < 1e-6
    n_const = int(const_mask.sum())
    safe_std = torch.where(const_mask.unsqueeze(1), torch.ones_like(voxel_std), voxel_std)
    data_vox_t = data_vox_t / safe_std
    data_vox_t[const_mask] = 0.0
    if trace_dir is not None:
        import numpy as _np

        trace_dir.mkdir(parents=True, exist_ok=True)
        _np.save(str(trace_dir / "migp_post_varnorm.npy"), data_vox_t.T.cpu().numpy())
        _np.save(str(trace_dir / "ffs_noise_std.npy"), safe_std.squeeze(1).cpu().numpy())
        _np.save(str(trace_dir / "ffs_const_mask.npy"), const_mask.cpu().numpy())
    norm_msg = (
        f"Voxel-norm: divided {n_vox_masked:,} voxels by temporal stdev "
        f"({n_const} constant voxels zeroed, legacy path)"
    )
    return data_vox_t, norm_msg


@torch.inference_mode()
def filter_low_variance_voxels(
    data_vox_t: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Apply FSL-style variability thresholding before MELODIC PPCA dim estimation.

    FSL runs ``update_mask`` before PPCA model-order selection and drops voxels
    with unusually low temporal variability. This helper mirrors the thresholding
    rule without modifying the caller's full ICA data matrix.
    """
    if data_vox_t.ndim != 2 or data_vox_t.shape[0] < 2:
        return data_vox_t, {
            "voxels_in": int(data_vox_t.shape[0]),
            "voxels_kept": int(data_vox_t.shape[0]),
            "voxels_dropped": 0,
            "std_threshold": 0.0,
            "std_mean": 0.0,
            "std_std": 0.0,
        }

    # Drop near-flat voxels before model-order estimation: they contribute no signal but
    # do distort the low end of the eigenspectrum, which is exactly what the Laplace
    # evidence reads. The rule is a plain low-outlier test -- keep a voxel whose temporal
    # stdev is within 3 SDs of the mean stdev -- with an absolute floor at 1% of the mean
    # so a dataset with a tight stdev distribution does not have the test bite everywhere.
    d_std = torch.std(data_vox_t, dim=1, unbiased=True)
    std_mean = float(d_std.mean().item())
    std_std = float(torch.std(d_std, unbiased=True).item()) if d_std.numel() > 1 else 0.0
    thr = float(max(std_mean - 3.0 * std_std, 0.01 * std_mean))
    keep = d_std > thr

    n_in = int(d_std.numel())
    n_keep = int(keep.sum().item())
    if n_keep < 2:
        # Safety fallback: never return an empty/degenerate matrix.
        keep = torch.ones_like(keep, dtype=torch.bool)
        n_keep = n_in

    filtered = data_vox_t[keep]
    return filtered, {
        "voxels_in": n_in,
        "voxels_kept": n_keep,
        "voxels_dropped": int(n_in - n_keep),
        "std_threshold": thr,
        "std_mean": std_mean,
        "std_std": std_std,
    }


def apply_melodic_noise_normalization(
    components: torch.Tensor,
    mixing: torch.Tensor,
    x_t: torch.Tensor,
    trace_dir: Path | None = None,
) -> tuple[torch.Tensor, str]:
    """Apply MELODIC-style IC noise normalization and return status message."""
    try:
        tdim, kdim = mixing.shape
        unmix = torch.linalg.pinv(mixing)
        diagvals = 1.0 / torch.sqrt(torch.clamp(torch.diag(unmix @ unmix.T), min=1e-12))
        # Chunked residual std to avoid materializing full (T, V) tensor
        from fastfuncstuff.memory import estimate_chunk_size

        n_vox = x_t.shape[1]
        chunk_size = estimate_chunk_size(
            n_voxels=n_vox,
            n_timepoints=tdim,
            n_regressors=0,
            device=x_t.device,
            operation="ica_varnorm",
        )
        resid_std = torch.empty(n_vox, device=x_t.device)
        n_chunks = (n_vox + chunk_size - 1) // chunk_size
        for v0 in tqdm(
            range(0, n_vox, chunk_size),
            total=n_chunks,
            desc="  Noise norm",
            leave=True,
            disable=n_chunks <= 1,
        ):
            v1 = min(v0 + chunk_size, n_vox)
            resid_chunk = x_t[:, v0:v1] - mixing @ components[:, v0:v1]
            resid_std[v0:v1] = torch.std(resid_chunk, dim=0)
            del resid_chunk
        resid_std = torch.where(resid_std < 0.05, torch.ones_like(resid_std), resid_std)
        dof_factor = float(np.sqrt((tdim - 1.0) / max(tdim - kdim, 1.0)))
        std_noise = 1.0 / (resid_std * dof_factor)

        if trace_dir is not None:
            trace_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(trace_dir / "resid_std.npy"), resid_std.cpu().numpy())
            np.save(str(trace_dir / "noise_inv.npy"), std_noise.cpu().numpy())
            np.save(str(trace_dir / "diagvals.npy"), diagvals.cpu().numpy())
            np.save(str(trace_dir / "unmix_matrix.npy"), unmix.cpu().numpy())

        components = components * (diagvals.unsqueeze(1) * std_noise.unsqueeze(0))
        del unmix, diagvals, resid_std, std_noise
        return components, "  Noise normalization applied (MELODIC convention)"
    except Exception as exc:
        return components, f"  ⚠ Noise normalization failed: {exc}, using raw IC maps"


def compute_guidance_scores(
    comp_np: np.ndarray,
    z_maps: np.ndarray | None,
    condition_corr: np.ndarray | None,
    ortvec_corr: np.ndarray | None,
    guidance_good_masks: list[dict],
    guidance_bad_masks: list[dict],
    depth_mask_info: dict | None,
    good_z_thresh: float,
    out_prefix: Path,
    run_tag: str,
    run_idx: int,
) -> dict:
    """Compute spatial/temporal guidance scores and optional mask heatmaps."""
    spatial_scores_good = np.zeros(comp_np.shape[0], dtype=np.float32)
    spatial_scores_bad = np.zeros(comp_np.shape[0], dtype=np.float32)
    temporal_good_scores = np.zeros(comp_np.shape[0], dtype=np.float32)
    temporal_bad_scores = np.zeros(comp_np.shape[0], dtype=np.float32)
    good_mask_score_table: dict[str, list[float]] = {}
    bad_mask_score_table: dict[str, list[float]] = {}

    if condition_corr is not None:
        temporal_good_scores = np.max(np.abs(condition_corr), axis=1).astype(np.float32)
    if ortvec_corr is not None:
        temporal_bad_scores = np.max(np.abs(ortvec_corr), axis=1).astype(np.float32)

    for entry in guidance_good_masks:
        selector = entry["selector"]
        if z_maps is not None:
            s = ica_postprocess.mean_z_excess_by_selector(
                z_maps, selector, z_thresh=float(good_z_thresh)
            )
        else:
            s = ica_postprocess.mean_abs_by_selector(comp_np, selector)
        spatial_scores_good += s
        good_mask_score_table[entry["name"]] = s.tolist()

    for entry in guidance_bad_masks:
        selector = entry["selector"]
        s = ica_postprocess.mean_abs_by_selector(comp_np, selector)
        spatial_scores_bad += s
        bad_mask_score_table[entry["name"]] = s.tolist()

    if len(guidance_good_masks) > 0:
        spatial_scores_good /= float(len(guidance_good_masks))
    if len(guidance_bad_masks) > 0:
        spatial_scores_bad /= float(len(guidance_bad_masks))

    spatial_good_norm = ica_postprocess.normalize_0_1(spatial_scores_good)
    spatial_bad_norm = ica_postprocess.normalize_0_1(spatial_scores_bad)
    temporal_good_norm = ica_postprocess.normalize_0_1(temporal_good_scores)
    temporal_bad_norm = ica_postprocess.normalize_0_1(temporal_bad_scores)

    overall_good = 0.65 * spatial_good_norm + 0.35 * temporal_good_norm
    overall_bad = 0.65 * spatial_bad_norm + 0.35 * temporal_bad_norm
    good_minus_bad = overall_good - overall_bad

    comp_labels = np.full(comp_np.shape[0], "uncertain", dtype=object)
    comp_labels[good_minus_bad >= 0.15] = "good"
    comp_labels[good_minus_bad <= -0.15] = "bad"

    depth_profile_abs: dict[str, list[float]] = {}
    depth_profile_zexcess: dict[str, list[float]] = {}
    if depth_mask_info is not None:
        for lbl in depth_mask_info["labels"]:
            sel = depth_mask_info["selectors"][lbl]
            depth_profile_abs[str(lbl)] = ica_postprocess.mean_abs_by_selector(
                comp_np, sel
            ).tolist()
            if z_maps is not None:
                depth_profile_zexcess[str(lbl)] = ica_postprocess.mean_z_excess_by_selector(
                    z_maps,
                    sel,
                    z_thresh=float(good_z_thresh),
                ).tolist()

    guidance_good_plot = None
    guidance_bad_plot = None
    if len(good_mask_score_table) > 0:
        labels = list(good_mask_score_table.keys())
        table = np.column_stack(
            [np.asarray(good_mask_score_table[k], dtype=np.float32) for k in labels]
        )
        guidance_good_plot = Path(f"{out_prefix}_{run_tag}_goodmask_scores.png")
        ica_postprocess.save_score_heatmap(
            scores_kn=table,
            labels=labels,
            out_png=guidance_good_plot,
            title=f"Run {run_idx + 1}: good-mask scores (z>{good_z_thresh:g})",
            cmap="Blues",
        )

    if len(bad_mask_score_table) > 0:
        labels = list(bad_mask_score_table.keys())
        table = np.column_stack(
            [np.asarray(bad_mask_score_table[k], dtype=np.float32) for k in labels]
        )
        guidance_bad_plot = Path(f"{out_prefix}_{run_tag}_badmask_scores.png")
        ica_postprocess.save_score_heatmap(
            scores_kn=table,
            labels=labels,
            out_png=guidance_bad_plot,
            title=f"Run {run_idx + 1}: bad-mask scores (abs IC)",
            cmap="Reds",
        )

    return {
        "spatial_scores_good": spatial_scores_good,
        "spatial_scores_bad": spatial_scores_bad,
        "temporal_good_scores": temporal_good_scores,
        "temporal_bad_scores": temporal_bad_scores,
        "overall_good": overall_good,
        "overall_bad": overall_bad,
        "good_minus_bad": good_minus_bad,
        "comp_labels": comp_labels,
        "good_mask_score_table": good_mask_score_table,
        "bad_mask_score_table": bad_mask_score_table,
        "depth_profile_abs": depth_profile_abs,
        "depth_profile_zexcess": depth_profile_zexcess,
        "guidance_good_plot": guidance_good_plot,
        "guidance_bad_plot": guidance_bad_plot,
    }


def run_depth_lag_analysis(
    enabled: bool,
    depth_mask_info: dict | None,
    z_maps: np.ndarray | None,
    depth_source_vox_t_np: np.ndarray | None,
    comp_np: np.ndarray,
    tr: float,
    polort: int,
    high_pass_hz: float | None,
    device: torch.device,
    depth_lag_match_preproc: bool,
    depth_lag_reference_depth: int,
    depth_lag_z_thresh: float,
    depth_lag_min_voxels: int,
    depth_lag_max_lag_s: float,
    out_prefix: Path,
    run_tag: str,
    run_idx: int,
    verbose: bool = False,
) -> dict:
    """Compute per-component lag-vs-depth summaries and lag heatmap."""
    depth_lag_results: list[dict] = []
    depth_lag_matrix_seconds = None
    depth_lag_matrix_r = None
    depth_lag_plot = None
    depth_lag_method = None

    if not enabled or depth_mask_info is None or z_maps is None or depth_source_vox_t_np is None:
        return {
            "depth_lag_results": depth_lag_results,
            "depth_lag_matrix_seconds": depth_lag_matrix_seconds,
            "depth_lag_matrix_r": depth_lag_matrix_r,
            "depth_lag_plot": depth_lag_plot,
            "depth_lag_method": depth_lag_method,
        }

    verbose_section(verbose, "Depth Lag Analysis")
    t_step = time.time()

    depth_source_proc = depth_source_vox_t_np
    if depth_lag_match_preproc:
        src_tc = torch.as_tensor(depth_source_vox_t_np, device=device, dtype=torch.float32)
        src_tc = apply_polort_projection(src_tc, polort=polort, device=device)
        if high_pass_hz is not None and high_pass_hz > 0:
            src_tc = apply_high_pass_fft(src_tc, tr=tr, high_pass_hz=high_pass_hz)
        depth_source_proc = src_tc.detach().cpu().numpy().astype(np.float32)
        del src_tc

    depth_labels = [int(v) for v in depth_mask_info["labels"]]
    n_comp = int(comp_np.shape[0])
    depth_lag_matrix_seconds = np.full((n_comp, len(depth_labels)), np.nan, dtype=np.float32)
    depth_lag_matrix_r = np.full((n_comp, len(depth_labels)), np.nan, dtype=np.float32)

    for ci in range(n_comp):
        z_w = np.where(z_maps[ci] > float(depth_lag_z_thresh), z_maps[ci], 0.0).astype(np.float32)

        depth_ts: dict[int, np.ndarray] = {}
        depth_nvox: dict[int, int] = {}
        for lbl in depth_labels:
            selector = depth_mask_info["selectors"][lbl]
            ts, n_use = ica_postprocess.weighted_depth_timeseries(
                source_vox_t=depth_source_proc,
                selector_v=selector,
                weight_v=z_w,
                min_voxels=int(depth_lag_min_voxels),
            )
            depth_nvox[int(lbl)] = int(n_use)
            if ts is not None:
                depth_ts[int(lbl)] = ts

        ref_depth = int(depth_lag_reference_depth)
        if ref_depth not in depth_ts:
            depth_lag_results.append(
                {
                    "component_index": int(ci + 1),
                    "status": "missing_reference_depth",
                    "reference_depth": ref_depth,
                    "n_weighted_voxels_by_depth": depth_nvox,
                }
            )
            continue

        lag_by_depth: dict[str, float | None] = {}
        r_by_depth: dict[str, float | None] = {}
        used_depths: list[int] = []
        used_lags_s: list[float] = []

        for dj, lbl in enumerate(depth_labels):
            lbl_i = int(lbl)
            if lbl_i not in depth_ts:
                lag_by_depth[str(lbl_i)] = None
                r_by_depth[str(lbl_i)] = None
                continue

            lag_s, r_val, method = ica_postprocess.best_lag_and_r(
                x_t=depth_ts[lbl_i],
                y_t=depth_ts[ref_depth],
                tr=tr,
                max_lag_s=float(depth_lag_max_lag_s),
            )
            depth_lag_matrix_seconds[ci, dj] = float(lag_s)
            depth_lag_matrix_r[ci, dj] = float(r_val)
            lag_by_depth[str(lbl_i)] = float(lag_s)
            r_by_depth[str(lbl_i)] = float(r_val)
            if lbl_i != ref_depth:
                used_depths.append(lbl_i)
                used_lags_s.append(float(lag_s))
            depth_lag_method = method

        if len(used_depths) >= 3:
            rho, pval = spearmanr(
                np.asarray(used_depths, dtype=np.float64),
                np.asarray(used_lags_s, dtype=np.float64),
            )
            rho_f = None if not np.isfinite(rho) else float(rho)
            pval_f = None if not np.isfinite(pval) else float(pval)
        else:
            rho_f, pval_f = None, None

        depth_lag_results.append(
            {
                "component_index": int(ci + 1),
                "status": "ok",
                "reference_depth": ref_depth,
                "n_weighted_voxels_by_depth": depth_nvox,
                "lag_seconds_by_depth": lag_by_depth,
                "peak_r_by_depth": r_by_depth,
                "spearman_depth_vs_lag": {
                    "rho": rho_f,
                    "pvalue": pval_f,
                    "n_depths": int(len(used_depths)),
                },
            }
        )

    depth_lag_plot = Path(f"{out_prefix}_{run_tag}_depth_lag_seconds.png")
    ica_postprocess.save_depth_lag_plot(
        lag_matrix_kd=depth_lag_matrix_seconds,
        depth_labels=depth_labels,
        out_png=depth_lag_plot,
        title=(
            f"Run {run_idx + 1}: depth lag vs depth {int(depth_lag_reference_depth)} "
            f"(z>{depth_lag_z_thresh:g})"
        ),
    )

    verbose_print(
        verbose,
        (
            f"Depth lag done for {len(depth_lag_results)} components "
            f"(ref depth={int(depth_lag_reference_depth)}, "
            f"z>{depth_lag_z_thresh:g})"
        ),
        t_step,
    )

    return {
        "depth_lag_results": depth_lag_results,
        "depth_lag_matrix_seconds": depth_lag_matrix_seconds,
        "depth_lag_matrix_r": depth_lag_matrix_r,
        "depth_lag_plot": depth_lag_plot,
        "depth_lag_method": depth_lag_method,
    }
