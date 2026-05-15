"""GLMsingle Type B benchmark: HRF selection (GLMsingle vs ffs_hrfopt -single_trials).

Compares per-voxel HRF index selection between GLMsingle's FITHRF step and
FFS's ``ffs_hrfopt -single_trials``. Both use the same 20-HRF library from
getcanonicalhrflibrary.tsv and select the best HRF per voxel based on
in-sample R² of a single-trial OLS fit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..runner import BenchmarkContext, run_timed

name = "glmsingle_hrf"
description = "HRF selection (GLMsingle Type B vs ffs_hrfopt -single_trials)"

# Thresholds: HRF selection can differ at boundaries between similar HRFs,
# so we use agreement percentage rather than correlation
THRESHOLDS = {
    "hrf_index_agreement": 0.70,  # fraction of voxels with same HRF index
    "r2_spatial_corr": 0.90,      # spatial correlation of R² maps
}

_DEFAULT_STIM_LABELS = ["faces", "bodies", "objects", "scenes", "scrambled"]


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
    for f in ["glmsingle_hrf_index.nii.gz", "glmsingle_r2_B.nii.gz",
              "glmsingle_mask.nii.gz"]:
        p = gs / f
        if not p.exists():
            missing.append(f"GLMsingle: {p}")

    if ctx.validate_only:
        ffs = ctx.ffs_hrfopt_dir
        if not (ffs / "hrfopt_hrf_index.nii.gz").exists():
            missing.append(f"FFS: {ffs / 'hrfopt_hrf_index.nii.gz'}")
    else:
        task = _primary_task(ctx)
        for r in _runs(ctx):
            ffs = ctx.processing_dir / f"ffs_mni_resampled_task-{task}_run-{r}.nii.gz"
            afni = ctx.processing_dir / f"afni_mni_resampled_task-{task}_run-{r}.nii.gz"
            if not ffs.exists() and not afni.exists():
                missing.append(str(ffs))
        for f in _onset_files(ctx):
            if not f.exists():
                missing.append(str(f))
    return missing


def run_ref(ctx: BenchmarkContext) -> float:
    """Return the GLMsingle Type B (FITHRF) time parsed from the MATLAB log.

    The MATLAB stage (``glmsingle_matlab``) writes per-model timings to
    ``glmsingle/glmsingle_timings.json``. This stage doesn't re-run MATLAB
    — it just surfaces the Type B time as the reference timing so the
    head-to-head Ref/FFS comparison line lights up against ``ffs_hrfopt``.

    Returns 0.0 when no timing is available (e.g. the MATLAB stage hasn't
    been run on this machine yet); the runner treats 0.0 as "no ref
    timing" and falls back to cached arches.
    """
    from . import glmsingle_matlab
    return float(glmsingle_matlab.load_timings(ctx).get("type_b", 0.0))


def run_ffs(ctx: BenchmarkContext) -> float:
    """Run ffs_hrfopt -single_trials on localizer data."""
    out_dir = ctx.ffs_hrfopt_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = str(out_dir / "hrfopt")

    # Skip if outputs exist and not forcing
    if not ctx.force_ffs and (out_dir / "hrfopt_hrf_index.nii.gz").exists():
        print("  Skipping ffs_hrfopt (outputs exist, use --force-ffs to re-run)")
        return 0.0

    inputs = " ".join(str(f) for f in _input_files(ctx))
    onsets = " ".join(str(f) for f in _onset_files(ctx))

    cmd = (
        f"ffs_hrfopt -input {inputs} "
        f"-onsets {onsets} "
        f"-durations 3.0 3.0 3.0 3.0 3.0 "
        f"-prefix {out_prefix} "
        f"-single_trials "
        f"-save_single_trial_betas "
        f"-hrf_mode library "
        f"-metric cod "
        f"-device cuda"
    )

    elapsed, _ = run_timed(cmd, label="ffs_hrfopt -single_trials", cwd=ctx.processing_dir)
    return elapsed


def validate(ctx: BenchmarkContext) -> dict:
    """Compare HRF indices and R² maps between GLMsingle and FFS."""
    import nibabel as nib

    gs = ctx.glmsingle_dir
    ffs = ctx.ffs_hrfopt_dir

    # Load GLMsingle results
    matlab_hrf_index = np.array(
        nib.load(str(gs / "glmsingle_hrf_index.nii.gz")).dataobj
    ).flatten()
    matlab_r2 = np.array(
        nib.load(str(gs / "glmsingle_r2_B.nii.gz")).dataobj
    ).flatten()
    matlab_mask = np.array(
        nib.load(str(gs / "glmsingle_mask.nii.gz")).dataobj
    ).flatten().astype(bool)

    # Load FFS results. ffs_hrfopt writes hrf_index as 4D with 2 sub-briks:
    # [0] = chosen HRF index (1..N_HRFS), [1] = per-voxel quality/score.
    # Compare against MATLAB's 3D single-brik output using sub-brik 0.
    ffs_idx_raw = np.array(nib.load(str(ffs / "hrfopt_hrf_index.nii.gz")).dataobj)
    if ffs_idx_raw.ndim == 4:
        ffs_idx_raw = ffs_idx_raw[..., 0]
    ffs_hrf_index = ffs_idx_raw.flatten()
    ffs_r2 = np.array(nib.load(str(ffs / "hrfopt_xval_r2.nii.gz")).dataobj).flatten()

    # Compare within mask
    mask = matlab_mask & (ffs_r2 > 0)
    n_mask = mask.sum()

    # HRF index agreement (MATLAB is 1-indexed, FFS is 0-indexed)
    matlab_idx_masked = matlab_hrf_index[mask]
    ffs_idx_masked = ffs_hrf_index[mask]

    # Check if FFS is 0-indexed by looking at range
    if ffs_idx_masked.min() == 0 and matlab_idx_masked.min() == 1:
        ffs_idx_masked = ffs_idx_masked + 1  # Convert to 1-indexed for comparison

    agreement = (matlab_idx_masked == ffs_idx_masked).mean()

    # R² spatial correlation
    from ..validation import _pearson_r
    r2_corr = _pearson_r(matlab_r2[mask], ffs_r2[mask])

    passed = (
        agreement >= THRESHOLDS["hrf_index_agreement"]
        and r2_corr >= THRESHOLDS["r2_spatial_corr"]
    )

    summary = f"HRF agree={agreement:.1%}, R² r={r2_corr:.4f}"
    return {
        "passed": passed,
        "summary": summary,
        "hrf_index_agreement": float(agreement),
        "r2_spatial_corr": float(r2_corr),
        "n_voxels": int(n_mask),
    }
