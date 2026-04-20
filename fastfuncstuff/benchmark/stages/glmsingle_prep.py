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


def _glm_params(ctx: BenchmarkContext) -> dict:
    return ctx.get_stage_params("glm")


def _primary_task(ctx: BenchmarkContext) -> str:
    return _glm_params(ctx).get("primary_task", "localizer")


def _runs(ctx: BenchmarkContext) -> list[int]:
    return ctx.runs_for_task(_primary_task(ctx))


def _ref_task_run(ctx: BenchmarkContext) -> tuple[str, int]:
    params = ctx.get_stage_params("crossalign")
    return params.get("reference_task", "localizer"), params.get("reference_run", 1)


def _input_path(ctx: BenchmarkContext, run: int) -> Path:
    task = _primary_task(ctx)
    return ctx.func_dir / f"{ctx.bids_prefix(task, run)}_bold.nii"


def _st_path(ctx: BenchmarkContext, run: int) -> Path:
    task = _primary_task(ctx)
    return ctx.processing_dir / f"ffs_st_resampled_task-{task}_run-{run}.nii.gz"


def _output_path(ctx: BenchmarkContext, run: int) -> Path:
    task = _primary_task(ctx)
    return ctx.processing_dir / f"ffs_mni_resampled_task-{task}_run-{run}.nii.gz"


def _nwarp_chain(ctx: BenchmarkContext, run: int) -> str:
    """Build warp chain for runs (same as warp stage)."""
    p = ctx.processing_dir
    ssw = p / "sswarper_output"
    subid = f"sub-{ctx.subject}"
    task = _primary_task(ctx)
    ref_task, ref_run = _ref_task_run(ctx)

    parts = [
        str(ssw / f"anatQQ.{subid}_WARP.nii"),
        str(ssw / f"anatQQ.{subid}.aff12.1D"),
        str(p / "anat_al_keep_e2a_only_mat.aff12.1D"),
    ]

    if not (task == ref_task and run == ref_run):
        align_mat = p / f"afni_mean_{task}_run-{run}_to_{ref_task}_run-{ref_run}_mat.aff12.1D"
        parts.append(str(align_mat))

    moco_mat = p / f"afni_moco_{ctx.bids_prefix(task, run)}_bold_mat.aff12.1D"
    parts.append(str(moco_mat))

    return " ".join(parts)


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    missing = []
    runs = _runs(ctx)
    task = _primary_task(ctx)

    if ctx.validate_only:
        for r in runs:
            out = _output_path(ctx, r)
            if not out.exists():
                missing.append(str(out))
        return missing

    # Need raw inputs + BIDS JSON (for SliceTiming)
    for r in runs:
        inp = _input_path(ctx, r)
        if not inp.exists():
            missing.append(str(inp))
    json_path = ctx.func_dir / f"{ctx.bids_prefix(task, runs[0])}_bold.json"
    if not json_path.exists():
        missing.append(str(json_path))

    # Need warp chain files (from moco + align + warp stages)
    subid = f"sub-{ctx.subject}"
    master = ctx.processing_dir / f"autobox_anatQQ.{subid}.nii"
    if not master.exists():
        missing.append(str(master))
    e2a = ctx.processing_dir / "anat_al_keep_e2a_only_mat.aff12.1D"
    if not e2a.exists():
        missing.append(str(e2a))

    return missing


def run_ffs(ctx: BenchmarkContext) -> float:
    """Run slicetime+resample then warp to MNI for all runs."""
    total = 0.0
    subid = f"sub-{ctx.subject}"
    task = _primary_task(ctx)
    master = ctx.processing_dir / f"autobox_anatQQ.{subid}.nii"

    for run in _runs(ctx):
        out = _output_path(ctx, run)
        if out.exists() and not ctx.force_ffs:
            continue

        # Step 1: slicetime correct + resample to TR=1.5
        st = _st_path(ctx, run)
        if not st.exists() or ctx.force_ffs:
            tp = ctx.tpattern_file(task, run)
            elapsed, _ = run_timed(
                f"ffs_slicetime "
                f"-input {_input_path(ctx, run)} "
                f"-tzero 0 -wsinc9 "
                f"-tpattern {tp} "
                f"-resample 1.5 "
                f"-prefix {st}",
                label=f"ffs_slicetime -resample 1.5 {task} run-{run}",
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
            label=f"ffs_nwarp resampled {task} run-{run}",
            cwd=ctx.processing_dir,
        )
        total += elapsed

    return total


def validate(ctx: BenchmarkContext) -> dict:
    """Check that all resampled MNI files exist and have expected shape."""
    runs = _runs(ctx)
    present = []
    missing = []
    for r in runs:
        out = _output_path(ctx, r)
        if out.exists():
            present.append(str(out.name))
        else:
            missing.append(str(out.name))

    n_runs = len(runs)
    passed = len(missing) == 0
    summary = f"{len(present)}/{n_runs} resampled runs present"
    if missing:
        summary += f", missing: {', '.join(missing)}"

    return {
        "passed": passed,
        "summary": summary,
        "present": present,
        "missing": missing,
    }
