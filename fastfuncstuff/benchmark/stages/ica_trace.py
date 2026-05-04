"""ICA trace benchmark: step-by-step parity validation against MELODIC debug outputs.

Self-contained stage that:
1. Reads MELODIC's ``melodic.log`` to extract the randomized file order.
2. Runs ``ffs_ica -temp_concat -migp -trace -migp_shuffle <order>`` to match.
3. Compares intermediates: eigenvalue spectrum, IC stats, mixing, spatial maps.

Requires MELODIC debug output (``--debug --Oall``) already present.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..runner import BenchmarkContext, run_timed
from ..validation import _load_vol

name = "ica_trace"
description = "ICA step-by-step parity (MELODIC debug vs ffs_ica -trace)"


def _ica_tasks(ctx: BenchmarkContext) -> list[str]:
    params = ctx.get_stage_params("ica_trace")
    return params.get("tasks", ctx.task_names())


def _melodic_dir(ctx: BenchmarkContext, dataset: str) -> Path:
    return ctx.melodic_ica_dir / f"all_{dataset}_melodic.ica"


def _trace_dir(ctx: BenchmarkContext, dataset: str) -> Path:
    return ctx.ffs_ica_dir / f"all_{dataset}_trace"


def _ffs_prefix(ctx: BenchmarkContext, dataset: str) -> Path:
    return ctx.ffs_ica_dir / f"all_{dataset}_concat"


# ---------------------------------------------------------------------------
# MELODIC log parsing
# ---------------------------------------------------------------------------

def _parse_melodic_file_order(log_path: Path) -> list[int]:
    """Extract the randomized file order from MELODIC's setup_migp log.

    MELODIC prints "Randomising input file order" then processes files.
    Each file starts with a "before reading file ... run-N" line.

    Returns 0-based run indices in the order MELODIC processed them.
    """
    text = log_path.read_text(errors="replace")
    lines = text.splitlines()
    order: list[int] = []
    in_migp = False
    for line in lines:
        if "Randomising input file order" in line:
            in_migp = True
            continue
        if in_migp and ("END: setup_migp" in line or "Excluding voxels" in line):
            break
        if in_migp and "before reading file" in line:
            m = re.search(r"run-(\d+)", line)
            if m:
                order.append(int(m.group(1)) - 1)
    if not order:
        raise ValueError(f"Could not parse file order from {log_path}")
    return order


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------

def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = np.sqrt((a_c ** 2).sum() * (b_c ** 2).sum())
    if denom < 1e-15:
        return 0.0
    return float((a_c * b_c).sum() / denom)


def _compare_eigenvalues(mel_dir: Path, trace_dir: Path) -> dict:
    mel_eig = np.loadtxt(mel_dir / "eigenvalues_adjusted")
    for name in ["eigenvalues_adjusted", "pca_eigenvalues.npy"]:
        p = trace_dir / name
        if p.exists():
            ffs_eig = np.load(p) if p.suffix == ".npy" else np.loadtxt(p)
            break
    else:
        return {"error": "FFS eigenvalues not found"}
    n = min(len(mel_eig), len(ffs_eig))
    return {
        "melodic_n": len(mel_eig),
        "ffs_n": len(ffs_eig),
        "full_spectrum_r": _pearson_r(mel_eig[:n], ffs_eig[:n]),
        "top20_r": _pearson_r(mel_eig[:min(20, n)], ffs_eig[:min(20, n)]),
        "melodic_first5": mel_eig[:5].tolist(),
        "ffs_first5": ffs_eig[:5].tolist(),
    }


def _compare_ic_stats(mel_dir: Path, trace_dir: Path) -> dict:
    mel_stats = np.loadtxt(mel_dir / "melodic_ICstats")
    p = trace_dir / "ICstats"
    if not p.exists():
        return {"error": "FFS ICstats not found"}
    ffs_stats = np.loadtxt(p)
    n = min(mel_stats.shape[0], ffs_stats.shape[0])
    return {
        "melodic_n": mel_stats.shape[0],
        "ffs_n": ffs_stats.shape[0],
        "variance_share_r": _pearson_r(mel_stats[:n, 0], ffs_stats[:n, 0]),
        "kurtosis_r": _pearson_r(mel_stats[:n, 2], ffs_stats[:n, 2]),
    }


def _compare_mixing(mel_dir: Path, trace_dir: Path) -> dict:
    mel_mix = np.loadtxt(mel_dir / "melodic_mix")
    p = trace_dir / "mix_matrix"
    if not p.exists():
        return {"error": "FFS mix_matrix not found"}
    ffs_mix = np.loadtxt(p)
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
        "n_matched": len(corrs),
    }


def _compare_subspace(mel_dir: Path, trace_dir: Path) -> dict:
    """Compare column-spaces of FFS pre-varnorm and MELODIC concat_data via
    principal angles.

    SVD has a basis ambiguity within near-equal singular values, so MIGP rows
    are not pointwise comparable between FFS and MELODIC even when both
    pipelines are correct. The right invariant is the column space.

    Procedure:
      X_ffs (T,V), X_mel (T,V) → take SVD, keep top-k right singular vectors
      V_ffs, V_mel  (V, k). Their cross-Gram U = V_ffs.T @ V_mel has SVs
      = cos(principal angles). All ≈ 1 → subspaces match.

    Reports:
      mean_principal_cos, min_principal_cos, n_above_0.99, k
    """
    # IMPORTANT: compare FFS POST-varnorm vs MELODIC POST-varnorm (apples-to-
    # apples). MELODIC's concat_data.nii.gz is post-varnorm; FFS dumps both
    # pre/post — use post here. Pre-vs-post would conflate varnorm scaling
    # (which is a column-rescaling that rotates the temporal SVD basis) with
    # any actual upstream divergence.
    post_p = trace_dir / "migp_post_varnorm.npy"
    mel_post_p = mel_dir / "concat_data.nii.gz"
    mask_p = trace_dir / "mask.nii.gz"
    if not post_p.exists() or not mel_post_p.exists():
        return {"error": "subspace inputs missing (need migp_post_varnorm.npy)"}

    import nibabel as nib

    pre = np.load(post_p).astype(np.float32)  # (T, V) FFS post-varnorm
    mel_4d = nib.load(str(mel_post_p)).get_fdata(dtype=np.float32)  # type: ignore[attr-defined]
    if not mask_p.exists():
        mask_p = mel_dir / "mask.nii.gz"
    mask = nib.load(str(mask_p)).get_fdata() > 0.5  # type: ignore[attr-defined]
    mel = mel_4d[mask].T.astype(np.float32)  # (T, V)

    if mel.shape != pre.shape:
        return {"error": f"shape mismatch ffs={pre.shape} mel={mel.shape}"}

    T = pre.shape[0]
    k = min(T - 1, 200)  # use up to 200 PCs of the column space

    # Right singular vectors via thin SVD on (T, V) — V_r is (k, V) with
    # orthonormal rows. We need orthonormal columns of dim V_x = V → use
    # transpose: rows of V_r ARE the orthonormal basis we want (Vh from
    # np.linalg.svd is already (k, V) with orthonormal rows).
    # Pearson-style: subtract per-row mean? No — column space is invariant to
    # adding constant rows; just use raw matrices.
    _u_f, _s_f, vh_f = np.linalg.svd(pre.astype(np.float64), full_matrices=False)
    _u_m, _s_m, vh_m = np.linalg.svd(mel.astype(np.float64), full_matrices=False)
    Bf = vh_f[:k]  # (k, V) orthonormal rows
    Bm = vh_m[:k]
    # cos(principal angles) = singular values of Bf @ Bm.T  (k, k)
    cos_angles = np.linalg.svd(Bf @ Bm.T, compute_uv=False)
    cos_angles = np.clip(cos_angles, 0.0, 1.0)
    # Principal angles are returned in *descending* cosine order (best-aligned
    # direction first). Index where the cosine drops below 0.99 / 0.95 / 0.50
    # tells us where in the subspace the agreement breaks down. Low index
    # (early drop) → fundamental disagreement; high index (only trailing
    # directions disagree) → numerical noise floor.
    def _first_below(arr: np.ndarray, thr: float) -> int:
        idx = np.where(arr < thr)[0]
        return int(idx[0]) if idx.size else int(arr.size)

    # Save the full curve for inspection.
    np.save(trace_dir / "subspace_principal_cos.npy", cos_angles)

    # Compare cosines against the FFS singular value spectrum to gauge whether
    # mismatched directions correspond to high-variance or noise-floor PCs.
    sv_f = _s_f[:k]
    var_frac = (sv_f ** 2).cumsum() / (sv_f ** 2).sum()
    weak_mask = cos_angles < 0.95
    if weak_mask.any():
        weak_idx = np.where(weak_mask)[0]
        weak_var_share = float(var_frac[weak_idx].max() - (var_frac[weak_idx[0] - 1] if weak_idx[0] > 0 else 0.0))
    else:
        weak_var_share = 0.0

    return {
        "k": int(k),
        "mean_principal_cos": float(cos_angles.mean()),
        "min_principal_cos": float(cos_angles.min()),
        "median_principal_cos": float(np.median(cos_angles)),
        "n_above_0.99": int((cos_angles > 0.99).sum()),
        "n_above_0.95": int((cos_angles > 0.95).sum()),
        "n_above_0.50": int((cos_angles > 0.50).sum()),
        "first_below_0.99": _first_below(cos_angles, 0.99),
        "first_below_0.95": _first_below(cos_angles, 0.95),
        "first_below_0.50": _first_below(cos_angles, 0.50),
        "var_share_of_disagreeing_dims": weak_var_share,
        "first10_cos": cos_angles[:10].tolist(),
        "last10_cos": cos_angles[-10:].tolist(),
    }


def _compare_varnorm(mel_dir: Path, trace_dir: Path) -> dict:
    """Recover MELODIC's noise_std map from concat_data and compare to FFS's.

    Strategy: MELODIC's concat_data.nii.gz is the post-varnorm (T_migp, V) matrix.
    FFS dumps migp_pre_varnorm.npy (T_migp, V) before applying its varnorm.
    If MIGP outputs match, then mel_noise_std[v] = pre[t,v] / mel_post[t,v]
    is constant in t and recovers MELODIC's noise_std per voxel.

    Reports:
      - migp_match_r        : pearson(pre_ffs, mel_post * recovered_std)
                              (sanity: ~1.0 if MIGP outputs truly match)
      - global_scale_ratio  : median(ffs_noise_std / mel_noise_std)
                              (≈1 if same units; far from 1 → scaling bug)
      - per_voxel_r         : pearson(ffs_noise_std, mel_noise_std)
                              (~1 if same algorithm, different scale; <1 if algo differs)
      - max_relative_diff   : max |ffs - mel| / mel  (after global scale correction)
    """
    pre_p = trace_dir / "migp_pre_varnorm.npy"
    mel_post_p = mel_dir / "concat_data.nii.gz"
    mask_p = trace_dir / "mask.nii.gz"
    ffs_std_p = trace_dir / "ffs_noise_std.npy"
    if not pre_p.exists() or not mel_post_p.exists() or not ffs_std_p.exists():
        return {"error": f"missing inputs: pre={pre_p.exists()} mel_post={mel_post_p.exists()} ffs_std={ffs_std_p.exists()}"}

    import nibabel as nib

    pre = np.load(pre_p).astype(np.float64)        # (T, V) FFS pre-varnorm
    ffs_std = np.load(ffs_std_p).astype(np.float64)  # (V,)
    mel_img = nib.load(str(mel_post_p))
    mel_4d = mel_img.get_fdata(dtype=np.float32)   # (X,Y,Z,T)
    if mel_4d.ndim != 4:
        return {"error": f"mel concat_data not 4D: shape={mel_4d.shape}"}
    if not mask_p.exists():
        # Fall back to MELODIC's mask
        mask_p = mel_dir / "mask.nii.gz"
    mask = nib.load(str(mask_p)).get_fdata() > 0.5
    mel_post = mel_4d[mask].astype(np.float64).T   # (T, V_masked)

    if mel_post.shape != pre.shape:
        return {
            "error": f"shape mismatch: ffs_pre={pre.shape}, mel_post={mel_post.shape}",
        }

    # Recover MELODIC's noise_std per voxel: ratio = pre / mel_post (should be
    # constant across t per voxel). Use a robust per-voxel estimate via
    # least-squares fit through origin: std = (pre . mel_post) / (mel_post . mel_post).
    num = (pre * mel_post).sum(axis=0)
    den = (mel_post * mel_post).sum(axis=0)
    valid = (den > 1e-12) & (np.abs(num) > 1e-12) & (ffs_std > 1e-12)
    mel_std = np.full_like(ffs_std, np.nan)
    mel_std[valid] = num[valid] / den[valid]

    # Sanity: how well does mel_post * mel_std reconstruct pre?
    recon = mel_post * mel_std[None, :]
    a = pre[:, valid].ravel()
    b = recon[:, valid].ravel()
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = float(np.sqrt((a_c ** 2).sum() * (b_c ** 2).sum()))
    migp_r = float((a_c * b_c).sum() / denom) if denom > 0 else 0.0

    ratio = ffs_std[valid] / mel_std[valid]
    finite = np.isfinite(ratio) & (ratio > 0)
    median_ratio = float(np.median(ratio[finite]))

    # After applying the median global ratio, look at remaining per-voxel disagreement.
    ffs_std_corr = ffs_std[valid] / max(median_ratio, 1e-12)
    rel = np.abs(ffs_std_corr - mel_std[valid]) / np.maximum(np.abs(mel_std[valid]), 1e-12)
    rel_finite = rel[np.isfinite(rel)]

    pvr = _pearson_r(ffs_std[valid], mel_std[valid])

    return {
        "n_voxels": int(valid.sum()),
        "migp_match_r": migp_r,
        "global_scale_ratio_ffs_over_mel": median_ratio,
        "per_voxel_std_r": pvr,
        "max_relative_diff_after_scale": float(rel_finite.max()) if rel_finite.size else float("nan"),
        "median_relative_diff_after_scale": float(np.median(rel_finite)) if rel_finite.size else float("nan"),
        "ffs_std_median": float(np.median(ffs_std[valid])),
        "mel_std_median": float(np.median(mel_std[valid])),
    }


def _compare_ic_maps(mel_dir: Path, ffs_prefix: Path, mask_path: Path) -> dict:
    mel_ic = mel_dir / "melodic_oIC.nii.gz"
    ffs_ic = Path(str(ffs_prefix) + "_ica_maps.nii.gz")
    if not mel_ic.exists() or not ffs_ic.exists():
        return {"error": "IC map files not found"}
    import torch
    mel_vol, _ = _load_vol(mel_ic)
    ffs_vol, _ = _load_vol(ffs_ic)
    if mel_vol.dim() == 4:
        mel_vol = mel_vol.permute(3, 0, 1, 2)
    if ffs_vol.dim() == 4:
        ffs_vol = ffs_vol.permute(3, 0, 1, 2)
    mask_vol, _ = _load_vol(mask_path) if mask_path.exists() else (None, None)
    mask = mask_vol > 0.5 if mask_vol is not None else mel_vol.abs().sum(0) > 1e-8
    from ...stats.spatial import optimal_matching, spatial_correlation_matrix
    corr_matrix = spatial_correlation_matrix(mel_vol, ffs_vol, mask=mask)
    abs_corr = np.abs(corr_matrix)
    _, _, matched_corrs = optimal_matching(abs_corr)
    return {
        "mean_matched_r": float(matched_corrs.mean()),
        "max_matched_r": float(matched_corrs.max()),
        "melodic_n": mel_vol.shape[0],
        "ffs_n": ffs_vol.shape[0],
    }


# ---------------------------------------------------------------------------
# Stage interface
# ---------------------------------------------------------------------------

def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    missing = []
    for dataset in _ica_tasks(ctx):
        md = _melodic_dir(ctx, dataset)
        if not (md / "melodic.log").exists():
            missing.append(str(md / "melodic.log"))
        if not (md / "melodic_IC.nii.gz").exists():
            missing.append(str(md / "melodic_IC.nii.gz"))
        for f in ["eigenvalues_adjusted", "melodic_ICstats", "melodic_mix"]:
            if not (md / f).exists():
                missing.append(str(md / f))
    return missing


def run_ref(ctx: BenchmarkContext) -> float:
    return 0.0


def run_ffs(ctx: BenchmarkContext) -> float:
    """Run ffs_ica with -trace, matching MELODIC's MIGP file order."""
    ctx.ffs_ica_dir.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for dataset in _ica_tasks(ctx):
        td = _trace_dir(ctx, dataset)
        pfx = _ffs_prefix(ctx, dataset)
        if (td / "ICstats").exists() and not ctx.force_ffs:
            continue

        mel_dir = _melodic_dir(ctx, dataset)
        mel_log = mel_dir / "melodic.log"
        order = _parse_melodic_file_order(mel_log)
        shuffle_arg = ",".join(str(i) for i in order)

        inputs = " ".join(
            str(ctx.processing_dir / f"afni_mni_task-{dataset}_run-{r}.nii.gz")
            for r in ctx.runs_for_task(dataset)
        )
        mask = mel_dir / "mask.nii.gz"
        mask_arg = f"-mask {mask}" if mask.exists() else ""

        elapsed, _ = run_timed(
            f"ffs_ica -input {inputs} "
            f"{mask_arg} "
            f"-temp_concat -ordering stdev "
            f"-migp -migp_shuffle {shuffle_arg} "
            f"-trace {td} "
            f"-prefix {pfx} -verbose",
            label=f"ffs_ica trace {dataset}",
            cwd=ctx.ffs_ica_dir,
        )
        total += elapsed
    return total


def validate(ctx: BenchmarkContext) -> dict:
    results = {}
    for dataset in _ica_tasks(ctx):
        mel_dir = _melodic_dir(ctx, dataset)
        td = _trace_dir(ctx, dataset)
        mask_path = mel_dir / "mask.nii.gz"
        results[dataset] = {
            "eigenvalues": _compare_eigenvalues(mel_dir, td),
            "subspace": _compare_subspace(mel_dir, td),
            "varnorm": _compare_varnorm(mel_dir, td),
            "ic_stats": _compare_ic_stats(mel_dir, td),
            "mixing": _compare_mixing(mel_dir, td),
            "ic_maps": _compare_ic_maps(mel_dir, _ffs_prefix(ctx, dataset), mask_path),
        }

    eig_r = np.mean([r["eigenvalues"]["full_spectrum_r"] for r in results.values()])
    maps_r = np.mean(
        [r["ic_maps"].get("mean_matched_r", 0) for r in results.values()]
    )
    passed = eig_r >= 0.95 and maps_r >= 0.60

    parts = [f"eig_r={eig_r:.4f}", f"maps_r={maps_r:.4f}"]
    for ds, r in results.items():
        er = r["eigenvalues"]["full_spectrum_r"]
        mr = r["ic_maps"].get("mean_matched_r", 0)
        sub = r.get("subspace", {})
        if "mean_principal_cos" in sub:
            parts.append(
                f"{ds}: eig={er:.3f} maps={mr:.3f} "
                f"subsp_cos_mean={sub['mean_principal_cos']:.4f} "
                f"min={sub['min_principal_cos']:.4f} "
                f">.99={sub['n_above_0.99']}/{sub['k']} "
                f"first<.95@{sub['first_below_0.95']} "
                f"first<.50@{sub['first_below_0.50']} "
                f"var_share_off={sub['var_share_of_disagreeing_dims']:.3f}"
            )
        else:
            parts.append(f"{ds}: eig={er:.3f} maps={mr:.3f} subsp=ERR")

    return {
        "passed": passed,
        "summary": ", ".join(parts),
        "per_dataset": results,
    }
