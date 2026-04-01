"""GLMsingle Type D benchmark: fracridge (GLMsingle vs ffs_ridge -single_trials).

Compares per-voxel FRACvalue selection, regularized betas, and autoscaled
betas between GLMsingle's FITHRF_GLMDENOISE_RR step and FFS's
``ffs_ridge -single_trials``.

Both tools:
1. Fit ridge regression at multiple fractions (0.05:0.05:1.0)
2. Cross-validate using beta-space LORO CV with SSE metric
3. Select optimal fraction per voxel (excluding frac=1.0)
4. Optionally autoscale betas to match OLS amplitude
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..runner import BenchmarkContext, run_timed

name = "glmsingle_ridge"
description = "Fracridge (GLMsingle Type D vs ffs_ridge -single_trials)"

THRESHOLDS = {
    "fracvalue_corr": 0.80,  # per-voxel fraction correlation
    "beta_spatial_corr": 0.85,  # spatial correlation of Type D betas
    "r2_spatial_corr": 0.90,  # spatial correlation of R² maps
}


def _input_files(ctx: BenchmarkContext) -> list[Path]:
    """MNI-space resampled localizer runs."""
    return [
        ctx.processing_dir / f"ffs_mni_resampled_task-localizer_run-{r}.nii.gz" for r in range(1, 6)
    ]


def _onset_files(ctx: BenchmarkContext) -> list[Path]:
    """AFNI-style onset timing files (one per condition)."""
    conds = ["faces", "bodies", "objects", "scenes", "scrambled"]
    return [ctx.timing_dir / f"onsets.localizer.times.{c}.txt" for c in conds]


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    missing = []
    gs = ctx.glmsingle_dir
    for f in [
        "glmsingle_fracvalue.nii.gz",
        "glmsingle_r2_D.nii.gz",
        "glmsingle_betas_D.nii.gz",
        "glmsingle_mask.nii.gz",
    ]:
        p = gs / f
        if not p.exists():
            missing.append(f"GLMsingle: {p}")

    ffs_ridge = ctx.ffs_ridge_dir
    if ctx.validate_only:
        for f in ["ridge_optimal_frac.nii.gz", "ridge_single_trial_betas.nii.gz"]:
            p = ffs_ridge / f
            if not p.exists():
                missing.append(f"FFS ridge: {p}")
    else:
        # Need denoise outputs (metadata with optimal_pcs) and HRF index
        ffs_denoise = ctx.ffs_denoise_dir
        if not (ffs_denoise / "denoise_denoise_metadata.json").exists():
            missing.append(f"FFS denoise: {ffs_denoise / 'denoise_denoise_metadata.json'}")
        hrfopt = ctx.ffs_hrfopt_dir
        if not (hrfopt / "hrfopt_hrf_index.nii.gz").exists():
            missing.append(f"FFS hrfopt: {hrfopt / 'hrfopt_hrf_index.nii.gz'}")
        for f in _input_files(ctx):
            if not f.exists():
                missing.append(str(f))
        for f in _onset_files(ctx):
            if not f.exists():
                missing.append(str(f))
    return missing


def run_ffs(ctx: BenchmarkContext) -> float:
    """Run ffs_ridge -single_trials on localizer data."""
    out_dir = ctx.ffs_ridge_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = str(out_dir / "ridge")

    # Skip if outputs exist and not forcing
    if not ctx.force_ffs and (out_dir / "ridge_single_trial_betas.nii.gz").exists():
        print("  Skipping ffs_ridge (outputs exist, use --force-ffs to re-run)")
        return 0.0

    inputs = " ".join(str(f) for f in _input_files(ctx))
    onsets = " ".join(str(f) for f in _onset_files(ctx))

    hrf_prefix = str(ctx.ffs_hrfopt_dir / "hrfopt")
    denoise_prefix = str(ctx.ffs_denoise_dir / "denoise")

    cmd = (
        f"ffs_ridge -input {inputs} "
        f"-onsets {onsets} "
        f"-durations 3.0 3.0 3.0 3.0 3.0 "
        f"-prefix {out_prefix} "
        f"-single_trials "
        f"-metric sse "
        f"-hrf_opt {hrf_prefix} "
        f"-denoise {denoise_prefix} "
        f"-autoscale "
        f"-device cuda"
    )

    elapsed, _ = run_timed(cmd, label="ffs_ridge -single_trials", cwd=ctx.processing_dir)
    return elapsed


def validate(ctx: BenchmarkContext) -> dict:
    """Compare FRACvalue and Type D betas."""
    import nibabel as nib

    gs = ctx.glmsingle_dir
    ffs = ctx.ffs_ridge_dir

    # Load GLMsingle NIfTI results
    matlab_fracvalue = np.array(nib.load(str(gs / "glmsingle_fracvalue.nii.gz")).dataobj).flatten()
    matlab_r2 = np.array(nib.load(str(gs / "glmsingle_r2_D.nii.gz")).dataobj).flatten()
    matlab_mask = (
        np.array(nib.load(str(gs / "glmsingle_mask.nii.gz")).dataobj).flatten().astype(bool)
    )
    matlab_betas = np.array(nib.load(str(gs / "glmsingle_betas_D.nii.gz")).dataobj)

    results = {}

    # Load FFS FRACvalue
    frac_file = ffs / "ridge_optimal_frac.nii.gz"
    ffs_frac = None
    if frac_file.exists():
        ffs_frac = np.array(nib.load(str(frac_file)).dataobj).flatten()

    # Load FFS betas (Type D = ridge regularized)
    betas_file = ffs / "ridge_single_trial_betas.nii.gz"
    ffs_betas = None
    if betas_file.exists():
        ffs_betas = np.array(nib.load(str(betas_file)).dataobj)

    # Load FFS R²
    r2_file = ffs / "ridge_single_trial_full_r2.nii.gz"
    ffs_r2 = None
    if r2_file.exists():
        ffs_r2 = np.array(nib.load(str(r2_file)).dataobj).flatten()

    from ..validation import _pearson_r

    # 1. FRACvalue correlation
    frac_corr = float("nan")
    if ffs_frac is not None:
        mask = matlab_mask & np.isfinite(ffs_frac) & np.isfinite(matlab_fracvalue)
        mask = mask & (ffs_frac > 0) & (matlab_fracvalue > 0)
        if mask.sum() > 100:
            frac_corr = _pearson_r(matlab_fracvalue[mask], ffs_frac[mask])
        results["frac_n_voxels"] = int(mask.sum())
    results["fracvalue_corr"] = frac_corr

    # 2. Beta spatial correlation (median across trials)
    beta_corr = float("nan")
    if ffs_betas is not None:
        # Reshape to (n_voxels, n_trials) if 4D
        if matlab_betas.ndim == 4:
            n_vox = np.prod(matlab_betas.shape[:3])
            matlab_betas_flat = matlab_betas.reshape(n_vox, matlab_betas.shape[3])
        else:
            matlab_betas_flat = matlab_betas

        if ffs_betas.ndim == 4:
            ffs_betas_flat = ffs_betas.reshape(np.prod(ffs_betas.shape[:3]), ffs_betas.shape[3])
        else:
            ffs_betas_flat = ffs_betas

        if matlab_betas_flat.shape == ffs_betas_flat.shape:
            mask = matlab_mask.copy()
            mask = mask & np.isfinite(matlab_betas_flat).all(axis=1)
            mask = mask & np.isfinite(ffs_betas_flat).all(axis=1)
            mask = mask & (np.abs(ffs_betas_flat).sum(axis=1) > 0)
            trial_corrs = []
            n_sample = min(matlab_betas_flat.shape[1], 50)
            for t in range(n_sample):
                r = _pearson_r(matlab_betas_flat[mask, t], ffs_betas_flat[mask, t])
                trial_corrs.append(r)
            beta_corr = float(np.median(trial_corrs))
            results["beta_n_voxels"] = int(mask.sum())
            results["beta_n_trials_sampled"] = int(n_sample)
        else:
            results["beta_shape_mismatch"] = (
                f"matlab={matlab_betas_flat.shape} vs ffs={ffs_betas_flat.shape}"
            )
    results["beta_spatial_corr"] = beta_corr

    # 3. R² spatial correlation
    r2_corr = float("nan")
    if ffs_r2 is not None:
        mask = matlab_mask & np.isfinite(ffs_r2) & np.isfinite(matlab_r2)
        if mask.sum() > 100:
            r2_corr = _pearson_r(matlab_r2[mask], ffs_r2[mask])
    results["r2_spatial_corr"] = r2_corr

    # Pass/fail
    passed = True
    if not np.isnan(frac_corr):
        passed = passed and frac_corr >= THRESHOLDS["fracvalue_corr"]
    if not np.isnan(beta_corr):
        passed = passed and beta_corr >= THRESHOLDS["beta_spatial_corr"]

    summary = f"frac r={frac_corr:.4f}, beta r={beta_corr:.4f}, R² r={r2_corr:.4f}"
    results["passed"] = passed
    results["summary"] = summary
    return results
