"""Prepare MNI-resampled localizer data for GLMsingle stages.

This stage creates the ``ffs_mni_resampled_task-localizer_run-{1..5}.nii.gz``
files that all GLMsingle stages (matlab, hrf, denoise, ridge) require.

Pipeline per run:
  1. ffs_slicetime with -resample 1.5 (correct + downsample to TR=1.5s)
  2. ffs_nwarp to MNI space using the same warp chain as the warp stage

These are separate from the main ``ffs_mni_task-*`` warped data because
GLMsingle uses a different TR (1.5s resampled from 1.75s) to match the
MATLAB GLMsingle configuration.
"""

from __future__ import annotations

from pathlib import Path

from ..runner import BenchmarkContext, run_timed

name = "glmsingle_prep"
description = "Prepare slicetime-resampled MNI data for GLMsingle"


def _input_path(ctx: BenchmarkContext, run: int) -> Path:
    return ctx.func_dir / f"sub-01_ses-01_task-localizer_run-{run}_bold.nii"


def _st_path(ctx: BenchmarkContext, run: int) -> Path:
    return ctx.processing_dir / f"ffs_st_resampled_task-localizer_run-{run}.nii.gz"


def _output_path(ctx: BenchmarkContext, run: int) -> Path:
    return ctx.processing_dir / f"ffs_mni_resampled_task-localizer_run-{run}.nii.gz"


def _nwarp_chain(ctx: BenchmarkContext, run: int) -> str:
    """Build warp chain for localizer runs (same as warp stage)."""
    p = ctx.processing_dir
    ssw = p / "sswarper_output"

    parts = [
        str(ssw / "anatQQ.sub-01_WARP.nii"),
        str(ssw / "anatQQ.sub-01.aff12.1D"),
        str(p / "anat_al_keep_e2a_only_mat.aff12.1D"),
    ]

    if run > 1:
        align_mat = p / f"afni_mean_localizer_run-{run}_to_localizer_run-1_mat.aff12.1D"
        parts.append(str(align_mat))

    moco_mat = p / f"afni_moco_sub-01_ses-01_task-localizer_run-{run}_bold_mat.aff12.1D"
    parts.append(str(moco_mat))

    return " ".join(parts)


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    missing = []
    if ctx.validate_only:
        for r in range(1, 6):
            out = _output_path(ctx, r)
            if not out.exists():
                missing.append(str(out))
        return missing

    # Need raw inputs + BIDS JSON (for SliceTiming)
    for r in range(1, 6):
        inp = _input_path(ctx, r)
        if not inp.exists():
            missing.append(str(inp))
    json_path = ctx.func_dir / "sub-01_ses-01_task-localizer_run-1_bold.json"
    if not json_path.exists():
        missing.append(str(json_path))

    # Need warp chain files (from moco + align + warp stages)
    master = ctx.processing_dir / "autobox_anatQQ.sub-01.nii"
    if not master.exists():
        missing.append(str(master))
    e2a = ctx.processing_dir / "anat_al_keep_e2a_only_mat.aff12.1D"
    if not e2a.exists():
        missing.append(str(e2a))

    return missing


def run_ffs(ctx: BenchmarkContext) -> float:
    """Run slicetime+resample then warp to MNI for all localizer runs."""
    total = 0.0
    master = ctx.processing_dir / "autobox_anatQQ.sub-01.nii"

    for run in range(1, 6):
        out = _output_path(ctx, run)
        if out.exists() and not ctx.force_ffs:
            continue

        # Step 1: slicetime correct + resample to TR=1.5
        st = _st_path(ctx, run)
        if not st.exists() or ctx.force_ffs:
            tp = ctx.tpattern_file("localizer", run)
            elapsed, _ = run_timed(
                f"ffs_slicetime "
                f"-input {_input_path(ctx, run)} "
                f"-tzero 0 -wsinc9 "
                f"-tpattern {tp} "
                f"-resample 1.5 "
                f"-prefix {st}",
                label=f"ffs_slicetime -resample 1.5 localizer run-{run}",
                cwd=ctx.processing_dir,
            )
            total += elapsed

        # Step 2: warp to MNI
        nwarp = _nwarp_chain(ctx, run)
        elapsed, _ = run_timed(
            f'ffs_nwarp '
            f'-master {master} -dxyz 3.0 -interp wsinc5 '
            f'-nwarp "{nwarp}" '
            f'-source {st} '
            f'-prefix {out}',
            label=f"ffs_nwarp resampled localizer run-{run}",
            cwd=ctx.processing_dir,
        )
        total += elapsed

    return total


def validate(ctx: BenchmarkContext) -> dict:
    """Check that all resampled MNI files exist and have expected shape."""
    present = []
    missing = []
    for r in range(1, 6):
        out = _output_path(ctx, r)
        if out.exists():
            present.append(str(out.name))
        else:
            missing.append(str(out.name))

    passed = len(missing) == 0
    summary = f"{len(present)}/5 resampled runs present"
    if missing:
        summary += f", missing: {', '.join(missing)}"

    return {
        "passed": passed,
        "summary": summary,
        "present": present,
        "missing": missing,
    }
