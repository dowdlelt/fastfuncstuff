"""IM model benchmark: 3dDeconvolve -stim_times_IM vs ffs_reml (OLS)."""

from __future__ import annotations

from pathlib import Path

from ..runner import BenchmarkContext, run_timed
from ..validation import compare_im_bucket

name = "glm_im"
description = "IM model OLS (3dDeconvolve -stim_times_IM vs ffs_reml -Obuck)"

_DEFAULT_STIM_LABELS = ["faces", "bodies", "objects", "scenes", "scrambled"]

THRESHOLDS = {
    # Spatial r per sub-brick (betas, t-stats, F-stat)
    "min_r": 0.95,
    # Voxelwise "temporal" r across the beta / t-stat stack
    "temporal_min_median_r": 0.95,
}


# ── Config helpers ────────────────────────────────────────────────────────────


def _glm_params(ctx: BenchmarkContext) -> dict:
    return ctx.get_stage_params("glm")


def _primary_task(ctx: BenchmarkContext) -> str:
    return _glm_params(ctx).get("primary_task", "localizer")


def _runs(ctx: BenchmarkContext) -> list[int]:
    return ctx.runs_for_task(_primary_task(ctx))


def _stim_labels(ctx: BenchmarkContext) -> list[str]:
    return _glm_params(ctx).get("stim_labels", _DEFAULT_STIM_LABELS)


def _hrf_model(ctx: BenchmarkContext) -> str:
    return _glm_params(ctx).get("hrf_model", "SPMG1(3)")


# ── Path helpers ──────────────────────────────────────────────────────────────


def _afni_dir(ctx: BenchmarkContext) -> Path:
    return ctx.data_dir / "afni_glm_IM"


def _ffs_dir(ctx: BenchmarkContext) -> Path:
    return ctx.data_dir / "ffs_glm_IM"


def _scaled_input(ctx: BenchmarkContext, run: int) -> Path:
    task = _primary_task(ctx)
    return ctx.processing_dir / f"scaled_afni_mni_task-{task}_run-{run}.nii.gz"


def _automask_path(ctx: BenchmarkContext) -> Path:
    return ctx.processing_dir / "MNI_automask.nii.gz"


def _xmat(ctx: BenchmarkContext) -> Path:
    """3dDeconvolve-generated design matrix (shared with glm_im_reml)."""
    return _afni_dir(ctx) / "IM_X.xmat.1D"


def _afni_bucket(ctx: BenchmarkContext) -> Path:
    task = _primary_task(ctx)
    return _afni_dir(ctx) / f"IM_afni_stats_{task}.nii.gz"


def _ffs_bucket(ctx: BenchmarkContext) -> Path:
    task = _primary_task(ctx)
    return _ffs_dir(ctx) / f"IM_ffs_stats_{task}.nii.gz"


# ── Stage interface ───────────────────────────────────────────────────────────


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    missing = []
    if ctx.validate_only:
        for p in [_afni_bucket(ctx), _ffs_bucket(ctx)]:
            if not p.exists():
                missing.append(str(p))
    else:
        for run in _runs(ctx):
            src = _scaled_input(ctx, run)
            if not src.exists():
                missing.append(str(src))
        if not _automask_path(ctx).exists():
            missing.append(str(_automask_path(ctx)))
    return missing


def run_ref(ctx: BenchmarkContext) -> float:
    """Run 3dDeconvolve with -stim_times_IM (one beta per event)."""
    afni = _afni_dir(ctx)
    afni.mkdir(parents=True, exist_ok=True)

    if _afni_bucket(ctx).exists() and not ctx.force_ref:
        return 0.0

    task = _primary_task(ctx)
    stim_labels = _stim_labels(ctx)
    inputs = " ".join(str(_scaled_input(ctx, r)) for r in _runs(ctx))
    mask = _automask_path(ctx)

    stim_args = " ".join(
        f"-stim_times_IM {i} {ctx.timing_dir}/onsets.{task}.times.{label}.txt "
        f"'{_hrf_model(ctx)}' -stim_label {i} {label}"
        for i, label in enumerate(stim_labels, 1)
    )

    elapsed, _ = run_timed(
        f"3dDeconvolve -overwrite "
        f"-input {inputs} "
        f"-polort A -num_stimts {len(stim_labels)} -float "
        f"{stim_args} "
        f"-jobs 10 -noFDR "
        f"-tout "
        f"-mask {mask} "
        f"-x1D {_xmat(ctx)} "
        f"-bucket {_afni_bucket(ctx)}",
        label="3dDeconvolve IM",
        cwd=afni,
    )
    return elapsed


def run_ffs(ctx: BenchmarkContext) -> float:
    """Run ffs_reml OLS using the IM design matrix."""
    ffs = _ffs_dir(ctx)
    ffs.mkdir(parents=True, exist_ok=True)

    if _ffs_bucket(ctx).exists() and not ctx.force_ffs:
        return 0.0

    inputs = " ".join(str(_scaled_input(ctx, r)) for r in _runs(ctx))
    mask = _automask_path(ctx)

    elapsed, _ = run_timed(
        f"ffs_reml "
        f"-input {inputs} "
        f"-matrix {_xmat(ctx)} "
        f"-use_double "
        f"-Obuck {_ffs_bucket(ctx)} "
        f"-mask {mask} "
        f"-tout"
        f"{ctx.ffs_device_flag()}",
        label="ffs_reml IM OLS",
        cwd=ffs,
    )
    return elapsed


def validate(ctx: BenchmarkContext) -> dict:
    """Compare AFNI and FFS IM bucket files."""
    mask_path = _automask_path(ctx) if _automask_path(ctx).exists() else None

    result = compare_im_bucket(
        _afni_bucket(ctx),
        _ffs_bucket(ctx),
        mask_path=mask_path,
    )

    fstat_r      = result["fstat"]["r"]
    beta_min_r   = result["betas"]["min_r"]
    beta_temp_r  = result["betas"]["temporal_median_r"]
    tstat_min_r  = result["tstats"]["min_r"]
    tstat_temp_r = result["tstats"]["temporal_median_r"]

    passed = (
        fstat_r      >= THRESHOLDS["min_r"]
        and beta_min_r   >= THRESHOLDS["min_r"]
        and beta_temp_r  >= THRESHOLDS["temporal_min_median_r"]
        and tstat_min_r  >= THRESHOLDS["min_r"]
        and tstat_temp_r >= THRESHOLDS["temporal_min_median_r"]
    )

    summary = (
        f"F_r={fstat_r:.3f} "
        f"beta min_r={beta_min_r:.3f} temp_r={beta_temp_r:.3f} "
        f"tstat min_r={tstat_min_r:.3f} temp_r={tstat_temp_r:.3f}"
    )

    return {
        "passed": passed,
        "summary": summary,
        **result,
    }
