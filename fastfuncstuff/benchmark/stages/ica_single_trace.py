"""ICA single-run trace benchmark: step-by-step parity per individual run.

Self-contained stage that:
1. Uses existing MELODIC single-run debug outputs (``--debug --Oall``).
2. Runs ``ffs_ica -trace`` per single run with matching MELODIC mask.
3. Compares intermediates: eigenvalues, varnorm, whitening, mixing, IC maps.

Removes MIGP from the pipeline — isolates varnorm and ICA solver divergence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..runner import BenchmarkContext, run_timed
from ..validation import _pearson_r, _load_vol, compare_prob_maps

name = "ica_single_trace"
description = "ICA single-run step-by-step parity (MELODIC debug vs ffs_ica -trace)"


def _ica_tasks(ctx: BenchmarkContext) -> list[str]:
    params = ctx.get_stage_params("ica_single_trace")
    return params.get("tasks", ctx.task_names())


def _melodic_dir(ctx: BenchmarkContext, dataset: str, run: int) -> Path:
    return ctx.melodic_ica_dir / f"{dataset}_run{run:02d}_melodic.ica"


def _trace_dir(ctx: BenchmarkContext, dataset: str, run: int) -> Path:
    return ctx.ffs_ica_dir / f"{dataset}_single_trace" / f"run{run:02d}" / "run01"


def _ffs_prefix(ctx: BenchmarkContext, dataset: str, run: int) -> str:
    return str(ctx.ffs_ica_dir / f"{dataset}_single_trace_run{run:02d}")


def _mni_input(ctx: BenchmarkContext, dataset: str, run: int) -> Path:
    return ctx.processing_dir / f"afni_mni_task-{dataset}_run-{run}.nii.gz"


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------

def _compare_eigenvalues(mel_dir: Path, trace_dir: Path) -> dict:
    mel_eig = np.loadtxt(str(mel_dir / "pcaD"))
    ffs_p = trace_dir / "pca_eigenvalues.npy"
    if not ffs_p.exists():
        return {"error": "FFS pca_eigenvalues.npy not found"}
    ffs_eig = np.load(str(ffs_p))

    mel_eig = np.sort(mel_eig)[::-1]
    ffs_eig = np.sort(ffs_eig)[::-1]

    mel_eig_pos = mel_eig[mel_eig > 1e-10]
    ffs_pos = ffs_eig[ffs_eig > 1e-10]
    n = min(len(mel_eig_pos), len(ffs_pos))

    if n == 0:
        return {"error": "no positive eigenvalues"}

    scale = float(mel_eig_pos[0] / ffs_pos[0]) if ffs_pos[0] > 0 else 1.0

    return {
        "melodic_n": len(mel_eig),
        "ffs_n": len(ffs_eig),
        "n_positive_overlap": n,
        "full_spectrum_r": _pearson_r(mel_eig_pos[:n], ffs_pos[:n]),
        "top20_r": _pearson_r(mel_eig_pos[:min(20, n)], ffs_pos[:min(20, n)]),
        "scale_ratio": scale,
        "melodic_first5": mel_eig_pos[:5].tolist(),
        "ffs_first5": ffs_pos[:5].tolist(),
    }


def _compare_varnorm(mel_dir: Path, trace_dir: Path) -> dict:
    post_p = trace_dir / "migp_post_varnorm.npy"
    mel_post_p = mel_dir / "concat_data.nii.gz"
    if not post_p.exists() or not mel_post_p.exists():
        return {"error": "varnorm inputs missing"}

    import nibabel as nib

    ffs_post = np.load(str(post_p)).astype(np.float64)
    mel_4d = nib.load(str(mel_post_p)).get_fdata(dtype=np.float32)
    mask_p = mel_dir / "mask.nii.gz"
    mask = nib.load(str(mask_p)).get_fdata() > 0.5
    mel_post = mel_4d[mask].astype(np.float64).T

    if mel_post.shape != ffs_post.shape:
        return {"error": f"shape mismatch: ffs={ffs_post.shape} mel={mel_post.shape}"}

    T, V = ffs_post.shape

    sample_size = min(V, 10000)
    rng = np.random.default_rng(42)
    sample = rng.choice(V, sample_size, replace=False)
    corrs = np.array([
        _pearson_r(ffs_post[:, v], mel_post[:, v])
        for v in sample
    ])
    corrs = corrs[np.isfinite(corrs)]

    def _topk_evals(X, k=50):
        rm = X.mean(axis=1, keepdims=True)
        C = (X @ X.T - V * (rm @ rm.T)) / float(V)
        return np.sort(np.linalg.eigvalsh(C))[::-1][:k]

    ffs_evals = _topk_evals(ffs_post)
    mel_evals = _topk_evals(mel_post)
    eig_r = _pearson_r(ffs_evals, mel_evals)

    ffs_vox_std = np.std(ffs_post, axis=0)
    mel_vox_std = np.std(mel_post, axis=0)

    valid_mask = (mel_vox_std > 1e-8) & (ffs_vox_std > 1e-8)
    scale_ratio = np.where(valid_mask, ffs_vox_std / mel_vox_std, np.nan)
    scale_ratio = scale_ratio[np.isfinite(scale_ratio)]

    ffs_vox_mean = np.mean(ffs_post, axis=0)
    mel_vox_mean = np.mean(mel_post, axis=0)
    offset_diff = ffs_vox_mean - mel_vox_mean
    offset_diff_valid = offset_diff[valid_mask]

    sample_idx = rng.choice(V, min(V, 2000), replace=False)
    rmses = np.array([
        np.sqrt(np.mean((ffs_post[:, v] - mel_post[:, v]) ** 2))
        for v in sample_idx
    ])
    scales = np.array([
        np.std(mel_post[:, v]) for v in sample_idx
    ])
    nrmse = rmses / np.where(scales > 1e-8, scales, 1.0)

    return {
        "voxels": V,
        "post_vn_mean_r": float(corrs.mean()),
        "post_vn_median_r": float(np.median(corrs)),
        "post_vn_min_r": float(corrs.min()),
        "post_vn_n_above_099": int((corrs > 0.99).sum()),
        "post_vn_n_above_095": int((corrs > 0.95).sum()),
        "post_vn_eig_r": eig_r,
        "ffs_vox_std_mean": float(ffs_vox_std.mean()),
        "mel_vox_std_mean": float(mel_vox_std.mean()),
        "scale_ratio_mean": float(np.mean(scale_ratio)),
        "scale_ratio_median": float(np.median(scale_ratio)),
        "scale_ratio_std": float(np.std(scale_ratio)),
        "offset_mean": float(np.mean(offset_diff_valid)),
        "offset_std": float(np.std(offset_diff_valid)),
        "nrmse_mean": float(np.mean(nrmse)),
        "nrmse_median": float(np.median(nrmse)),
    }


def _compare_whitening(mel_dir: Path, trace_dir: Path) -> dict:
    mel_wm = np.loadtxt(str(mel_dir / "whiteMatrix"))
    ffs_wm_p = trace_dir / "white_matrix.npy"
    if not ffs_wm_p.exists():
        return {"error": "FFS white_matrix.npy not found"}
    ffs_wm = np.load(str(ffs_wm_p))

    mel_dwm = np.loadtxt(str(mel_dir / "dewhiteMatrix"))
    ffs_dwm_p = trace_dir / "dewhite_matrix.npy"
    ffs_dwm = np.load(str(ffs_dwm_p)) if ffs_dwm_p.exists() else None

    k_mel, T_mel = mel_wm.shape
    k_ffs, T_ffs = ffs_wm.shape
    k = min(k_mel, k_ffs)
    T = min(T_mel, T_ffs)

    mel_w = mel_wm[:k, :T]
    ffs_w = ffs_wm[:k, :T]

    cross = mel_w @ ffs_w.T
    cos_angles = np.linalg.svd(cross, compute_uv=False)
    cos_angles = np.clip(cos_angles, 0.0, 1.0)

    result = {
        "melodic_shape": list(mel_wm.shape),
        "ffs_shape": list(ffs_wm.shape),
        "mean_principal_cos": float(cos_angles.mean()),
        "min_principal_cos": float(cos_angles.min()),
        "n_above_099": int((cos_angles > 0.99).sum()),
        "n_above_095": int((cos_angles > 0.95).sum()),
        "k": k,
    }

    if ffs_dwm is not None:
        n_dwm = min(mel_dwm.shape[1], ffs_dwm.shape[1])
        dwm_r = _pearson_r(mel_dwm[:, :n_dwm].ravel(), ffs_dwm[:, :n_dwm].ravel())
        result["dewhite_r"] = dwm_r

    return result


def _compare_mixing(mel_dir: Path, trace_dir: Path) -> dict:
    mel_mix = np.loadtxt(str(mel_dir / "melodic_mix"))
    ffs_p = trace_dir / "mix_matrix"
    if not ffs_p.exists():
        return {"error": "FFS mix_matrix not found"}
    ffs_mix = np.loadtxt(str(ffs_p))
    n_t = min(mel_mix.shape[0], ffs_mix.shape[0])
    n_k = min(mel_mix.shape[1], ffs_mix.shape[1])
    cross = np.abs(np.corrcoef(mel_mix[:n_t, :n_k].T, ffs_mix[:n_t, :n_k].T))
    k = n_k
    cross_block = cross[:k, k:]
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(1.0 - cross_block)
    corrs = cross_block[row_ind, col_ind]
    return {
        "mean_matched_r": float(corrs.mean()),
        "max_matched_r": float(corrs.max()),
        "min_matched_r": float(corrs.min()),
        "n_matched": len(corrs),
    }


def _compare_ic_stats(mel_dir: Path, trace_dir: Path) -> dict:
    mel_stats = np.loadtxt(str(mel_dir / "melodic_ICstats"))
    ffs_p = trace_dir / "ICstats"
    if not ffs_p.exists():
        return {"error": "FFS ICstats not found"}
    ffs_stats = np.loadtxt(str(ffs_p))
    n = min(mel_stats.shape[0], ffs_stats.shape[0])
    return {
        "melodic_n": mel_stats.shape[0],
        "ffs_n": ffs_stats.shape[0],
        "kurtosis_r": _pearson_r(mel_stats[:n, 2], ffs_stats[:n, 2]),
    }


def _compare_ic_maps(mel_dir: Path, trace_dir: Path, mask_path: Path) -> dict:
    mel_ic_p = mel_dir / "melodic_oIC.nii.gz"
    ffs_ic_p = trace_dir / "ic_maps.npy"
    if not mel_ic_p.exists() or not ffs_ic_p.exists():
        return {"error": "IC map files not found"}

    import nibabel as nib

    mel_4d = nib.load(str(mel_ic_p)).get_fdata(dtype=np.float32)
    mask = nib.load(str(mask_path)).get_fdata() > 0.5
    mel_maps = mel_4d[mask].T  # (k, V)

    ffs_maps = np.load(str(ffs_ic_p)).astype(np.float32)

    n_k = min(mel_maps.shape[0], ffs_maps.shape[0])
    mel_k = mel_maps[:n_k]
    ffs_k = ffs_maps[:n_k]

    cross = np.abs(np.corrcoef(mel_k, ffs_k))
    cross_block = cross[:n_k, n_k:]
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(1.0 - cross_block)
    corrs = cross_block[row_ind, col_ind]

    return {
        "mean_matched_r": float(corrs.mean()),
        "max_matched_r": float(corrs.max()),
        "min_matched_r": float(corrs.min()),
        "melodic_n": mel_maps.shape[0],
        "ffs_n": ffs_maps.shape[0],
        "n_matched": len(corrs),
    }


def _compare_noise_norm(mel_dir: Path, trace_dir: Path) -> dict:
    mel_noise_p = mel_dir / "Noise__inv.nii.gz"
    ffs_noise_p = trace_dir / "noise_inv.npy"
    ffs_resid_p = trace_dir / "resid_std.npy"
    ffs_diag_p = trace_dir / "diagvals.npy"
    missing = [n for n, p in [("mel_noise", mel_noise_p), ("ffs_noise", ffs_noise_p)]
               if not p.exists()]
    if missing:
        return {"error": f"missing: {', '.join(missing)}"}

    import nibabel as nib

    mel_noise_vol = nib.load(str(mel_noise_p)).get_fdata(dtype=np.float32)
    mask_p = mel_dir / "mask.nii.gz"
    mask = nib.load(str(mask_p)).get_fdata() > 0.5
    mel_noise = mel_noise_vol[mask].astype(np.float64)

    ffs_noise = np.load(str(ffs_noise_p)).astype(np.float64)

    if mel_noise.shape != ffs_noise.shape:
        return {"error": f"shape mismatch: mel={mel_noise.shape} ffs={ffs_noise.shape}"}

    noise_r = _pearson_r(mel_noise, ffs_noise)

    result = {
        "noise_inv_r": noise_r,
        "noise_inv_mel_mean": float(mel_noise.mean()),
        "noise_inv_ffs_mean": float(ffs_noise.mean()),
        "voxels": len(mel_noise),
    }

    if ffs_resid_p.exists():
        resid = np.load(str(ffs_resid_p)).astype(np.float64)
        result["resid_std_mean"] = float(resid.mean())
        result["resid_std_std"] = float(resid.std())

    if ffs_diag_p.exists():
        diag = np.load(str(ffs_diag_p)).astype(np.float64)
        result["diagvals_mean"] = float(diag.mean())
        result["diagvals_n"] = len(diag)

    return result


def _compare_raw_oic(mel_dir: Path, trace_dir: Path) -> dict:
    mel_oic_p = mel_dir / "melodic_oIC.nii.gz"
    ffs_oic_p = trace_dir / "oic.npy"
    if not mel_oic_p.exists() or not ffs_oic_p.exists():
        return {"error": "raw oIC files not found"}

    import nibabel as nib

    mel_4d = nib.load(str(mel_oic_p)).get_fdata(dtype=np.float32)
    mask_p = mel_dir / "mask.nii.gz"
    mask = nib.load(str(mask_p)).get_fdata() > 0.5
    mel_maps = mel_4d[mask].T.astype(np.float32)

    ffs_maps = np.load(str(ffs_oic_p)).astype(np.float32)

    n_k = min(mel_maps.shape[0], ffs_maps.shape[0])
    cross = np.abs(np.corrcoef(mel_maps[:n_k], ffs_maps[:n_k]))
    cross_block = cross[:n_k, n_k:]
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(1.0 - cross_block)
    corrs = cross_block[row_ind, col_ind]

    return {
        "mean_matched_r": float(corrs.mean()),
        "max_matched_r": float(corrs.max()),
        "min_matched_r": float(corrs.min()),
        "melodic_n": mel_maps.shape[0],
        "ffs_n": ffs_maps.shape[0],
        "n_matched": len(corrs),
    }


def _compare_unmix(mel_dir: Path, trace_dir: Path) -> dict:
    mel_unmix_p = mel_dir / "melodic_unmix"
    ffs_unmix_p = trace_dir / "unmix_matrix.npy"
    if not mel_unmix_p.exists() or not ffs_unmix_p.exists():
        return {"error": "unmix files not found"}

    mel_unmix = np.loadtxt(str(mel_unmix_p))
    ffs_unmix = np.load(str(ffs_unmix_p))

    n_k = min(mel_unmix.shape[0], ffs_unmix.shape[0])
    n_t = min(mel_unmix.shape[1], ffs_unmix.shape[1])

    cross = np.abs(np.corrcoef(mel_unmix[:n_k, :n_t], ffs_unmix[:n_k, :n_t]))
    cross_block = cross[:n_k, n_k:]
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(1.0 - cross_block)
    corrs = cross_block[row_ind, col_ind]

    return {
        "mean_matched_r": float(corrs.mean()),
        "max_matched_r": float(corrs.max()),
        "min_matched_r": float(corrs.min()),
        "n_matched": len(corrs),
    }


def _compare_pca_components(mel_dir: Path, trace_dir: Path) -> dict:
    mel_pca_p = mel_dir / "melodic_pca.nii.gz"
    ffs_pca_p = trace_dir / "pca_components.npy"
    if not mel_pca_p.exists() or not ffs_pca_p.exists():
        return {"error": "PCA component files not found"}

    import nibabel as nib

    mel_4d = nib.load(str(mel_pca_p)).get_fdata(dtype=np.float32)
    mask_p = mel_dir / "mask.nii.gz"
    mask = nib.load(str(mask_p)).get_fdata() > 0.5
    mel_pca = mel_4d[mask].T.astype(np.float32)

    ffs_pca = np.load(str(ffs_pca_p)).astype(np.float32)

    n_k = min(mel_pca.shape[0], ffs_pca.shape[0])
    cross = mel_pca[:n_k] @ ffs_pca[:n_k].T
    cos_angles = np.linalg.svd(cross, compute_uv=False)
    cos_angles = np.clip(cos_angles, 0.0, 1.0)

    return {
        "mean_principal_cos": float(cos_angles.mean()),
        "min_principal_cos": float(cos_angles.min()),
        "n_above_099": int((cos_angles > 0.99).sum()),
        "n_above_095": int((cos_angles > 0.95).sum()),
        "k": n_k,
    }


def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Stage interface
# ---------------------------------------------------------------------------

def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    missing = []
    for dataset in _ica_tasks(ctx):
        for run in ctx.runs_for_task(dataset):
            md = _melodic_dir(ctx, dataset, run)
            for f in ["pcaD", "whiteMatrix", "melodic_mix", "melodic_ICstats",
                       "concat_data.nii.gz", "melodic_oIC.nii.gz", "mask.nii.gz"]:
                if not (md / f).exists():
                    missing.append(str(md / f))
    return missing


def run_ref(ctx: BenchmarkContext) -> float:
    return 0.0


def run_ffs(ctx: BenchmarkContext) -> float:
    ctx.ffs_ica_dir.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for dataset in _ica_tasks(ctx):
        for run in ctx.runs_for_task(dataset):
            td = _trace_dir(ctx, dataset, run)
            if (td / "ICstats").exists() and not ctx.force_ffs:
                continue

            mel_dir = _melodic_dir(ctx, dataset, run)
            mask = mel_dir / "mask.nii.gz"
            inp = _mni_input(ctx, dataset, run)
            pfx = _ffs_prefix(ctx, dataset, run)
            trace_base = str(ctx.ffs_ica_dir / f"{dataset}_single_trace" / f"run{run:02d}")

            mask_arg = f"-mask {mask}" if mask.exists() else ""

            elapsed, _ = run_timed(
                f"ffs_ica -input {inp} "
                f"{mask_arg} "
                f"-ordering stdev "
                f"-trace {trace_base} "
                f"-prefix {pfx} -verbose",
                label=f"ffs_ica trace {dataset} run-{run}",
                cwd=ctx.ffs_ica_dir,
            )
            total += elapsed
    return total


def validate(ctx: BenchmarkContext) -> dict:
    per_run_results = []

    for dataset in _ica_tasks(ctx):
        for run in ctx.runs_for_task(dataset):
            mel_dir = _melodic_dir(ctx, dataset, run)
            td = _trace_dir(ctx, dataset, run)
            mask_path = mel_dir / "mask.nii.gz"

            if not td.exists():
                per_run_results.append({
                    "dataset": dataset, "run": run, "error": "trace dir missing"
                })
                continue

            ffs_zp = (ctx.ffs_ica_dir / f"{dataset}_single_trace_run{run:02d}.ica"
                      / "stats" / "z_prob.nii.gz")
            run_result = {
                "dataset": dataset,
                "run": run,
                "eigenvalues": _compare_eigenvalues(mel_dir, td),
                "varnorm": _compare_varnorm(mel_dir, td),
                "whitening": _compare_whitening(mel_dir, td),
                "ic_stats": _compare_ic_stats(mel_dir, td),
                "mixing": _compare_mixing(mel_dir, td),
                "ic_maps": _compare_ic_maps(mel_dir, td, mask_path),
                "prob_maps": _safe(compare_prob_maps, mel_dir, ffs_zp, mask_path),
                "noise_norm": _safe(_compare_noise_norm, mel_dir, td),
                "raw_oic": _safe(_compare_raw_oic, mel_dir, td),
                "unmix": _safe(_compare_unmix, mel_dir, td),
                "pca_components": _safe(_compare_pca_components, mel_dir, td),
            }
            per_run_results.append(run_result)

    valid = [r for r in per_run_results if "error" not in r]
    if not valid:
        return {
            "passed": False,
            "summary": "No valid single-run trace comparisons",
            "per_run": per_run_results,
        }

    eig_rs = [r["eigenvalues"]["full_spectrum_r"] for r in valid
              if "error" not in r.get("eigenvalues", {})]
    varnorm_eig_rs = [r["varnorm"]["post_vn_eig_r"] for r in valid
                      if "error" not in r.get("varnorm", {})]
    maps_rs = [r["ic_maps"]["mean_matched_r"] for r in valid
               if "error" not in r.get("ic_maps", {})]
    mix_rs = [r["mixing"]["mean_matched_r"] for r in valid
              if "error" not in r.get("mixing", {})]
    white_cos = [r["whitening"]["mean_principal_cos"] for r in valid
                 if "error" not in r.get("whitening", {})]

    prob_rs = [r["prob_maps"]["mean_matched_r"] for r in valid
               if "error" not in r.get("prob_maps", {})]

    mean_eig = float(np.mean(eig_rs)) if eig_rs else 0.0
    mean_vn_eig = float(np.mean(varnorm_eig_rs)) if varnorm_eig_rs else 0.0
    mean_maps = float(np.mean(maps_rs)) if maps_rs else 0.0
    mean_mix = float(np.mean(mix_rs)) if mix_rs else 0.0
    mean_white = float(np.mean(white_cos)) if white_cos else 0.0
    mean_prob = float(np.mean(prob_rs)) if prob_rs else 0.0

    vn_scales = [r["varnorm"]["scale_ratio_mean"] for r in valid
                 if "error" not in r.get("varnorm", {})]
    vn_nrmses = [r["varnorm"]["nrmse_mean"] for r in valid
                 if "error" not in r.get("varnorm", {})]
    mean_vn_scale = float(np.mean(vn_scales)) if vn_scales else 0.0
    mean_vn_nrmse = float(np.mean(vn_nrmses)) if vn_nrmses else 0.0

    passed = mean_eig >= 0.95 and mean_maps >= 0.60

    parts = [
        f"eig_r={mean_eig:.4f}",
        f"vn_eig_r={mean_vn_eig:.4f}",
        f"vn_scale={mean_vn_scale:.4f}",
        f"vn_nrmse={mean_vn_nrmse:.4f}",
        f"white_cos={mean_white:.4f}",
        f"mix_r={mean_mix:.4f}",
        f"maps_r={mean_maps:.4f}",
        f"prob_r={mean_prob:.4f}",
        f"n_runs={len(valid)}",
    ]

    run_parts = []
    for r in valid:
        ds, rn = r["dataset"], r["run"]
        er = r["eigenvalues"].get("full_spectrum_r", 0)
        mr = r["ic_maps"].get("mean_matched_r", 0)
        pr = r["prob_maps"].get("mean_matched_r", None)
        pr_str = f" prob_r={pr:.3f}" if pr is not None else ""
        wc = r["whitening"].get("mean_principal_cos", 0)
        vr = r["varnorm"].get("post_vn_eig_r", 0)
        vs = r["varnorm"].get("scale_ratio_mean", 0)
        vn = r["varnorm"].get("nrmse_mean", 0)
        nn = r.get("noise_norm", {}).get("noise_inv_r", None)
        nn_str = f" noise_r={nn:.3f}" if nn is not None else ""
        run_parts.append(
            f"  {ds}/run{rn:02d}: eig={er:.3f} vn_eig={vr:.3f} "
            f"vn_scale={vs:.4f} vn_nrmse={vn:.4f} "
            f"white_cos={wc:.4f} maps={mr:.3f}{pr_str}{nn_str}"
        )

    summary = ", ".join(parts) + "\n" + "\n".join(run_parts)

    return {
        "passed": passed,
        "summary": summary,
        "overall_eig_r": mean_eig,
        "overall_vn_eig_r": mean_vn_eig,
        "overall_vn_scale": mean_vn_scale,
        "overall_vn_nrmse": mean_vn_nrmse,
        "overall_white_cos": mean_white,
        "overall_mix_r": mean_mix,
        "overall_maps_r": mean_maps,
        "overall_prob_r": mean_prob,
        "n_valid_runs": len(valid),
        "per_run": per_run_results,
    }
