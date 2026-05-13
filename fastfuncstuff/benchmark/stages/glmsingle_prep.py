"""Prepare MNI-resampled localizer data for GLMsingle stages.

Creates TR=1.5s slice-time-corrected MNI-warped data for all GLMsingle stages.

Two variants, both using the same warp chain as the main warp stage:
  run_ref: ffs_slicetime (CPU) + 3dNwarpApply → afni_mni_resampled_task-*_run-*.nii.gz
  run_ffs: ffs_slicetime (CPU, shared intermediate) + ffs_nwarp → ffs_mni_resampled_task-*_run-*.nii.gz

On Linux/CUDA: both run; ffs_nwarp is faster.
On Mac/ref-only: only run_ref runs; downstream stages fall back to afni_mni_resampled_*.

All downstream stages (glmsingle_matlab, glm_tent, glmsingle_hrf, glmsingle_denoise,
glmsingle_ridge) prefer ffs_mni_resampled_* when present, else use afni_mni_resampled_*.
This ensures MATLAB and FFS always run on the same data for a valid comparison.
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


def _afni_output_path(ctx: BenchmarkContext, run: int) -> Path:
    task = _primary_task(ctx)
    return ctx.processing_dir / f"afni_mni_resampled_task-{task}_run-{run}.nii.gz"


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
            ffs = _output_path(ctx, r)
            afni = _afni_output_path(ctx, r)
            if not ffs.exists() and not afni.exists():
                missing.append(str(ffs))
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


def _ensure_st_resampled(ctx: BenchmarkContext, run: int, force: bool) -> float:
    """Run ffs_slicetime -resample 1.5 for one run if not already done.

    Both run_ref and run_ffs share this intermediate so slicetime is never
    duplicated when both paths run on the same machine.

    Always forced to -device cpu: the TR resampling uses float64 FFT-based
    Sinc interpolation that is numerically safer on CPU regardless of platform.
    """
    task = _primary_task(ctx)
    st = _st_path(ctx, run)
    if st.exists() and not force:
        return 0.0
    tp = ctx.tpattern_file(task, run)
    elapsed, _ = run_timed(
        f"ffs_slicetime "
        f"-input {_input_path(ctx, run)} "
        f"-tzero 0 -wsinc9 "
        f"-tpattern {tp} "
        f"-resample 1.5 "
        f"-device cpu "
        f"-prefix {st}",
        label=f"ffs_slicetime -resample 1.5 {task} run-{run}",
        cwd=ctx.processing_dir,
    )
    return elapsed


def run_ref(ctx: BenchmarkContext) -> float:
    """Warp TR=1.5s slice-time-corrected data to MNI using 3dNwarpApply.

    Produces afni_mni_resampled_task-*_run-*.nii.gz. Uses the shared
    ffs_st_resampled_* intermediate (created by ffs_slicetime, CPU-only).
    """
    total = 0.0
    subid = f"sub-{ctx.subject}"
    master = ctx.processing_dir / f"autobox_anatQQ.{subid}.nii"

    for run in _runs(ctx):
        out = _afni_output_path(ctx, run)
        if out.exists() and not ctx.force_ref:
            continue

        total += _ensure_st_resampled(ctx, run, force=ctx.force_ref)

        st = _st_path(ctx, run)
        nwarp = _nwarp_chain(ctx, run)
        task = _primary_task(ctx)
        elapsed, _ = run_timed(
            f'3dNwarpApply -overwrite '
            f'-master {master} -dxyz 3.0 -wsinc5 '
            f'-nwarp "{nwarp}" '
            f'-source {st} '
            f'-prefix {out}',
            label=f"3dNwarpApply resampled {task} run-{run}",
            cwd=ctx.processing_dir,
        )
        total += elapsed

    return total


def run_ffs(ctx: BenchmarkContext) -> float:
    """Warp TR=1.5s slice-time-corrected data to MNI using ffs_nwarp.

    Produces ffs_mni_resampled_task-*_run-*.nii.gz. Shares the
    ffs_st_resampled_* intermediate with run_ref when both run.
    """
    total = 0.0
    subid = f"sub-{ctx.subject}"
    task = _primary_task(ctx)
    master = ctx.processing_dir / f"autobox_anatQQ.{subid}.nii"

    for run in _runs(ctx):
        out = _output_path(ctx, run)
        if out.exists() and not ctx.force_ffs:
            continue

        total += _ensure_st_resampled(ctx, run, force=ctx.force_ffs)

        nwarp = _nwarp_chain(ctx, run)
        elapsed, _ = run_timed(
            f'ffs_nwarp '
            f'-master {master} -dxyz 3.0 -interp wsinc5 '
            f'-nwarp "{nwarp}" '
            f'-source {_st_path(ctx, run)} '
            f'-prefix {out}',
            label=f"ffs_nwarp resampled {task} run-{run}",
            cwd=ctx.processing_dir,
        )
        total += elapsed

    return total


def validate(ctx: BenchmarkContext) -> dict:
    """Check that at least one variant of resampled MNI files exists per run."""
    runs = _runs(ctx)
    ffs_present, afni_present, missing = [], [], []
    for r in runs:
        ffs = _output_path(ctx, r)
        afni = _afni_output_path(ctx, r)
        if ffs.exists():
            ffs_present.append(ffs.name)
        elif afni.exists():
            afni_present.append(afni.name)
        else:
            missing.append(ffs.name)

    n_runs = len(runs)
    n_present = len(ffs_present) + len(afni_present)
    passed = len(missing) == 0
    summary = f"{n_present}/{n_runs} resampled runs present"
    if afni_present:
        summary += f" ({len(afni_present)} afni, {len(ffs_present)} ffs)"
    if missing:
        summary += f", missing: {', '.join(missing)}"

    return {
        "passed": passed,
        "summary": summary,
        "ffs_present": ffs_present,
        "afni_present": afni_present,
        "missing": missing,
    }
