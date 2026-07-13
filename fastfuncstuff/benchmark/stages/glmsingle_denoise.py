"""GLMsingle Type C benchmark: PC denoising (GLMsingle vs ffs_denoise -single_trials).

Compares noise pool selection, PC count, xvaltrend curve, and denoised betas
between GLMsingle's GLMDENOISE step and FFS's ``ffs_denoise -single_trials``.

Both tools:
1. Identify a noise pool (bright voxels with low task R²)
2. Extract PCs from noise pool timeseries (after projecting out polynomials)
3. Cross-validate PC count using beta-space LORO CV
4. Select optimal PC count using pcstop criterion
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..runner import BenchmarkContext, run_timed

name = "glmsingle_denoise"
description = "PC denoising (GLMsingle Type C vs ffs_denoise -single_trials)"

THRESHOLDS = {
    "pcnum_tolerance": 2,  # pcnum must agree within ±2
    "xvaltrend_corr": 0.90,  # xvaltrend curve correlation
    "noisepool_overlap": 0.50,  # Jaccard index of noise pool masks
    "beta_spatial_corr": 0.85,  # spatial correlation of denoised betas (median across trials)
}

_DEFAULT_STIM_LABELS = ["faces", "objects", "scenes", "scrambled", "bodies"]


def _glm_params(ctx: BenchmarkContext) -> dict:
    return ctx.get_stage_params("glm")


def _primary_task(ctx: BenchmarkContext) -> str:
    return _glm_params(ctx).get("primary_task", "localizer")


def _runs(ctx: BenchmarkContext) -> list[int]:
    return ctx.runs_for_task(_primary_task(ctx))


def _stim_labels(ctx: BenchmarkContext) -> list[str]:
    return _glm_params(ctx).get("stim_labels", _DEFAULT_STIM_LABELS)


def _input_files(ctx: BenchmarkContext) -> list[Path]:
    """MNI-space resampled runs: prefer ffs_mni_resampled_*, fall back to afni_mni_resampled_*."""
    task = _primary_task(ctx)
    result = []
    for r in _runs(ctx):
        ffs = ctx.processing_dir / f"ffs_mni_resampled_task-{task}_run-{r}.nii.gz"
        afni = ctx.processing_dir / f"afni_mni_resampled_task-{task}_run-{r}.nii.gz"
        result.append(ffs if ffs.exists() else afni)
    return result


def _onset_files(ctx: BenchmarkContext) -> list[Path]:
    """AFNI-style onset timing files (one per condition)."""
    task = _primary_task(ctx)
    return [ctx.timing_dir / f"onsets.{task}.times.{c}.txt" for c in _stim_labels(ctx)]


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    missing = []
    gs = ctx.glmsingle_dir
    for f in [
        "glmsingle_noisepool.nii.gz",
        "glmsingle_r2_C.nii.gz",
        "glmsingle_betas_C.nii.gz",
        "glmsingle_mask.nii.gz",
        "glmsingle_pcnum.txt",
        "glmsingle_xvaltrend.txt",
    ]:
        p = gs / f
        if not p.exists():
            missing.append(f"GLMsingle: {p}")

    ffs = ctx.ffs_denoise_dir
    if ctx.validate_only:
        for f in ["denoise_denoise_metadata.json", "denoise_single_trial_betas.nii.gz"]:
            p = ffs / f
            if not p.exists():
                missing.append(f"FFS denoise: {p}")
    else:
        # Need HRF index from Type B
        hrfopt = ctx.ffs_hrfopt_dir
        if not (hrfopt / "hrfopt_hrf_index.nii.gz").exists():
            missing.append(f"FFS hrfopt: {hrfopt / 'hrfopt_hrf_index.nii.gz'}")
        task = _primary_task(ctx)
        for r in _runs(ctx):
            ffs_f = ctx.processing_dir / f"ffs_mni_resampled_task-{task}_run-{r}.nii.gz"
            afni_f = ctx.processing_dir / f"afni_mni_resampled_task-{task}_run-{r}.nii.gz"
            if not ffs_f.exists() and not afni_f.exists():
                missing.append(str(ffs_f))
        for f in _onset_files(ctx):
            if not f.exists():
                missing.append(str(f))
    return missing


def run_ref(ctx: BenchmarkContext) -> float:
    """Return the GLMsingle Type C (GLMDENOISE) time parsed from the MATLAB log.

    The MATLAB stage writes per-model timings to
    ``glmsingle/glmsingle_timings.json``. Returns 0.0 when not available.
    """
    from . import glmsingle_matlab

    return float(glmsingle_matlab.load_timings(ctx).get("type_c", 0.0))


def run_ffs(ctx: BenchmarkContext) -> float:
    """Run ffs_denoise -single_trials on localizer data."""
    out_dir = ctx.ffs_denoise_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = str(out_dir / "denoise")

    # Skip if outputs exist and not forcing
    if not ctx.force_ffs and (out_dir / "denoise_single_trial_betas.nii.gz").exists():
        print("  Skipping ffs_denoise (outputs exist, use --force-ffs to re-run)")
        return 0.0

    inputs = " ".join(str(f) for f in _input_files(ctx))
    onsets = " ".join(str(f) for f in _onset_files(ctx))

    # HRF opt prefix: ffs_hrfopt_dir contains hrfopt_hrf_index.nii.gz etc
    hrf_prefix = str(ctx.ffs_hrfopt_dir / "hrfopt")

    cmd = (
        f"ffs_denoise -input {inputs} "
        f"-onsets {onsets} "
        f"-durations 3.0 3.0 3.0 3.0 3.0 "
        f"-prefix {out_prefix} "
        f"-single_trials "
        f"-hrf_opt {hrf_prefix} "
        f"-tr 1.5 "
        f"-cv_metric sse "
        f"-pcstop 1.05 "
        f"-max_comps 10 "
        f"-brainthresh 99 0.1 "
        f"-do_scale"
        f"{ctx.ffs_device_flag()}"
    )

    elapsed, _ = run_timed(cmd, label="ffs_denoise -single_trials", cwd=ctx.processing_dir)
    return elapsed


def validate(ctx: BenchmarkContext) -> dict:
    """Compare PC selection and denoised betas."""
    import json

    import nibabel as nib

    gs = ctx.glmsingle_dir
    ffs = ctx.ffs_denoise_dir

    # Load GLMsingle results
    matlab_noisepool = (
        np.array(nib.load(str(gs / "glmsingle_noisepool.nii.gz")).dataobj).flatten().astype(bool)
    )
    matlab_betas = np.array(nib.load(str(gs / "glmsingle_betas_C.nii.gz")).dataobj)
    matlab_mask = (
        np.array(nib.load(str(gs / "glmsingle_mask.nii.gz")).dataobj).flatten().astype(bool)
    )

    matlab_pcnum = int((gs / "glmsingle_pcnum.txt").read_text().strip())
    matlab_xvaltrend = np.loadtxt(str(gs / "glmsingle_xvaltrend.txt"))

    # Load FFS results
    meta_file = ffs / "denoise_denoise_metadata.json"
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        ffs_pcnum = meta.get("optimal_pcs", -1)
    else:
        ffs_pcnum = -1

    # FFS noise pool mask
    noise_pool_file = ffs / "denoise_noise_pool_mask.nii.gz"
    ffs_noisepool = None
    if noise_pool_file.exists():
        ffs_noisepool = np.array(nib.load(str(noise_pool_file)).dataobj).flatten().astype(bool)

    # FFS betas (Type C = denoised, before ridge)
    betas_file = ffs / "denoise_single_trial_betas.nii.gz"
    ffs_betas = None
    if betas_file.exists():
        ffs_betas = np.array(nib.load(str(betas_file)).dataobj)

    # FFS xvaltrend (pc_selection_curve.npy)
    curve_file = ffs / "denoise_pc_selection_curve.npy"
    ffs_xvaltrend = None
    if curve_file.exists():
        ffs_xvaltrend = np.load(str(curve_file))

    results = {}

    # 1. PC count agreement
    pcnum_diff = abs(ffs_pcnum - matlab_pcnum)
    results["matlab_pcnum"] = matlab_pcnum
    results["ffs_pcnum"] = ffs_pcnum
    results["pcnum_diff"] = pcnum_diff

    # 2. Noise pool overlap (Jaccard index)
    noisepool_jaccard = float("nan")
    if ffs_noisepool is not None:
        intersection = (matlab_noisepool & ffs_noisepool).sum()
        union = (matlab_noisepool | ffs_noisepool).sum()
        noisepool_jaccard = float(intersection / union) if union > 0 else 0.0
        results["matlab_noise_count"] = int(matlab_noisepool.sum())
        results["ffs_noise_count"] = int(ffs_noisepool.sum())
    results["noisepool_jaccard"] = noisepool_jaccard

    # 3. xvaltrend correlation
    xvaltrend_corr = float("nan")
    if ffs_xvaltrend is not None:
        min_len = min(len(matlab_xvaltrend), len(ffs_xvaltrend))
        if min_len >= 2:
            from ..validation import _pearson_r

            xvaltrend_corr = _pearson_r(matlab_xvaltrend[:min_len], ffs_xvaltrend[:min_len])
    results["xvaltrend_corr"] = float(xvaltrend_corr)

    # 4. Beta spatial correlation (median across trials)
    beta_corr = float("nan")
    if ffs_betas is not None:
        # Reshape to (n_voxels, n_trials) if 4D
        if matlab_betas.ndim == 4:
            n_vox = np.prod(matlab_betas.shape[:3])
            n_trials_m = matlab_betas.shape[3]
            matlab_betas_flat = matlab_betas.reshape(n_vox, n_trials_m)
        else:
            matlab_betas_flat = matlab_betas

        if ffs_betas.ndim == 4:
            n_vox_f = np.prod(ffs_betas.shape[:3])
            n_trials_f = ffs_betas.shape[3]
            ffs_betas_flat = ffs_betas.reshape(n_vox_f, n_trials_f)
        else:
            ffs_betas_flat = ffs_betas

        if matlab_betas_flat.shape == ffs_betas_flat.shape:
            from ..validation import _pearson_r

            mask = matlab_mask & (np.abs(ffs_betas_flat).sum(axis=1) > 0)
            trial_corrs = []
            n_sample = min(matlab_betas_flat.shape[1], 50)
            for t in range(n_sample):
                r = _pearson_r(matlab_betas_flat[mask, t], ffs_betas_flat[mask, t])
                trial_corrs.append(r)
            beta_corr = float(np.median(trial_corrs))
        else:
            results["shape_mismatch"] = (
                f"matlab={matlab_betas_flat.shape} vs ffs={ffs_betas_flat.shape}"
            )
    results["beta_spatial_corr"] = beta_corr

    # Pass/fail
    passed = pcnum_diff <= THRESHOLDS["pcnum_tolerance"]
    if not np.isnan(noisepool_jaccard):
        passed = passed and noisepool_jaccard >= THRESHOLDS["noisepool_overlap"]
    if not np.isnan(beta_corr):
        passed = passed and beta_corr >= THRESHOLDS["beta_spatial_corr"]

    summary = (
        f"pcnum={ffs_pcnum} (matlab={matlab_pcnum}), "
        f"noise Jaccard={noisepool_jaccard:.3f}, "
        f"beta r={beta_corr:.4f}"
    )
    results["passed"] = passed
    results["summary"] = summary
    return results
