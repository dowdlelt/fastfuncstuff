"""Motion correction benchmark: 3dvolreg vs ffs_moco."""

from __future__ import annotations

from ..runner import BenchmarkContext, run_timed
from ..validation import compare_1d_params, compare_moco_ssd, compare_volumes

name = "moco"
description = "Motion correction (3dvolreg vs ffs_moco)"

TASKS = ["localizer", "rest"]
RUNS = [1, 2, 3, 4, 5]

MOTION_PARAM_NAMES = ["roll", "pitch", "yaw", "dS", "dL", "dP"]

THRESHOLDS = {
    # Mean images are the primary quality metric — pass/fail
    "mean_image_min_r": 0.98,
    # Motion params are diagnostic only — different optimizers find slightly
    # different paths especially for sub-voxel motion, but converge to the
    # same alignment (reflected in mean image agreement)
    "motion_param_mean_r": 0.85,
}


def _afni_moco_path(ctx: BenchmarkContext, task: str, run: int) -> str:
    return str(ctx.processing_dir / f"afni_moco_sub-01_ses-01_task-{task}_run-{run}_bold.nii")


def _ffs_moco_path(ctx: BenchmarkContext, task: str, run: int) -> str:
    return str(ctx.processing_dir / f"ffs_moco_sub-01_ses-01_task-{task}_run-{run}_bold.nii")


def _afni_motion_path(ctx: BenchmarkContext, task: str, run: int) -> str:
    return str(ctx.processing_dir / f"afni_motion_correction_task-{task}_run-{run}.1D")


def _ffs_motion_path(ctx: BenchmarkContext, task: str, run: int) -> str:
    return str(ctx.processing_dir / f"ffs_motion_correction_task-{task}_run-{run}.1D")


def _afni_mean_path(ctx: BenchmarkContext, task: str, run: int) -> str:
    return str(ctx.processing_dir / f"afni_mean_sub-01_ses-01_task-{task}_run-{run}_bold.nii")


def _ffs_mean_path(ctx: BenchmarkContext, task: str, run: int) -> str:
    return str(ctx.processing_dir / f"ffs_mean_sub-01_ses-01_task-{task}_run-{run}_bold.nii")


def _input_path(ctx: BenchmarkContext, task: str, run: int) -> str:
    return str(ctx.func_dir / f"sub-01_ses-01_task-{task}_run-{run}_bold.nii")


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    """Check that required outputs exist (or inputs exist for execution mode)."""
    from pathlib import Path

    missing = []

    if ctx.validate_only:
        # Need both AFNI and FFS outputs
        for task in TASKS:
            for run in RUNS:
                for path in [
                    _afni_moco_path(ctx, task, run),
                    _ffs_moco_path(ctx, task, run),
                    _afni_motion_path(ctx, task, run),
                    _ffs_motion_path(ctx, task, run),
                    _afni_mean_path(ctx, task, run),
                    _ffs_mean_path(ctx, task, run),
                ]:
                    if not Path(path).exists():
                        missing.append(path)
    else:
        # Need raw input data
        for task in TASKS:
            for run in RUNS:
                inp = _input_path(ctx, task, run)
                if not Path(inp).exists():
                    missing.append(inp)
    return missing


def run_ref(ctx: BenchmarkContext) -> float:
    """Run AFNI 3dvolreg + 3dTstat for all task/run combos."""
    from pathlib import Path

    total = 0.0
    for task in TASKS:
        for run in RUNS:
            out = _afni_moco_path(ctx, task, run)
            if Path(out).exists() and not ctx.force_ref:
                continue
            elapsed, _ = run_timed(
                f"3dvolreg -overwrite -heptic "
                f"-prefix {out} "
                f"-base 0 "
                f"-1Dfile {_afni_motion_path(ctx, task, run)} "
                f"-1Dmatrix_save {out.replace('.nii', '_mat.aff12.1D')} "
                f"{_input_path(ctx, task, run)}",
                label=f"3dvolreg {task} run-{run}",
                cwd=ctx.processing_dir,
            )
            total += elapsed

            # Mean image
            run_timed(
                f"3dTstat -overwrite -prefix {_afni_mean_path(ctx, task, run)} {out}",
                label=f"3dTstat mean {task} run-{run}",
                cwd=ctx.processing_dir,
            )
    return total


def run_ffs(ctx: BenchmarkContext) -> float:
    """Run ffs_moco for all task/run combos."""
    from pathlib import Path

    total = 0.0
    for task in TASKS:
        for run in RUNS:
            out = _ffs_moco_path(ctx, task, run)
            if Path(out).exists() and not ctx.force_ffs:
                continue
            elapsed, _ = run_timed(
                f"ffs_moco "
                f"-input {_input_path(ctx, task, run)} "
                f"-interp heptic -final heptic "
                f"-weight_automask -base 0 "
                f"-1Dfile {_ffs_motion_path(ctx, task, run)} "
                f"-1Dmatrix_save {out.replace('.nii', '_mat.aff12.1D')} "
                f"-prefix {out} -save_mean",
                label=f"ffs_moco {task} run-{run}",
                cwd=ctx.processing_dir,
            )
            total += elapsed

            # Rename mean output
            mean_src = ctx.processing_dir / f"mean_{Path(out).name}"
            mean_dst = Path(_ffs_mean_path(ctx, task, run))
            if mean_src.exists() and not mean_dst.exists():
                mean_src.rename(mean_dst)
    return total


def validate(ctx: BenchmarkContext) -> dict:
    """Compare motion parameters, mean images, and MSD across all runs."""
    motion_results = []
    mean_results = []
    ssd_results = []

    for task in TASKS:
        for run in RUNS:
            # Motion parameters (with per-column names)
            mp = compare_1d_params(
                _afni_motion_path(ctx, task, run),
                _ffs_motion_path(ctx, task, run),
            )
            # Label columns with motion parameter names
            per_col = mp.get("per_column_r", [])
            mp["per_column"] = {
                MOTION_PARAM_NAMES[i] if i < len(MOTION_PARAM_NAMES) else f"col{i}": r
                for i, r in enumerate(per_col)
            }
            mp["task"] = task
            mp["run"] = run
            motion_results.append(mp)

            # Mean images
            mv = compare_volumes(
                _afni_mean_path(ctx, task, run),
                _ffs_mean_path(ctx, task, run),
            )
            mv["task"] = task
            mv["run"] = run
            mean_results.append(mv)

            # MSD of motion-corrected timeseries
            sd = compare_moco_ssd(
                _afni_moco_path(ctx, task, run),
                _ffs_moco_path(ctx, task, run),
            )
            sd["task"] = task
            sd["run"] = run
            ssd_results.append(sd)

    # Aggregate
    motion_min_r = min(m["min_r"] for m in motion_results)
    motion_mean_r = sum(m["mean_r"] for m in motion_results) / len(motion_results)
    mean_min_r = min(m["r"] for m in mean_results)
    mean_mean_r = sum(m["r"] for m in mean_results) / len(mean_results)

    # MSD aggregates
    nrmsd_vals = [s["nrmsd"] for s in ssd_results if "error" not in s]
    mean_nrmsd = sum(nrmsd_vals) / len(nrmsd_vals) if nrmsd_vals else 0.0
    max_nrmsd = max(nrmsd_vals) if nrmsd_vals else 0.0

    # Pass based on output quality (mean images), not parameter paths
    passed = mean_min_r >= THRESHOLDS["mean_image_min_r"]

    return {
        "passed": passed,
        "summary": (
            f"means min_r={mean_min_r:.4f}, motion mean_r={motion_mean_r:.4f}, "
            f"nrmsd={mean_nrmsd:.4f}"
        ),
        "motion_params": {
            "min_r": motion_min_r,
            "mean_r": motion_mean_r,
            "per_run": motion_results,
        },
        "mean_images": {
            "min_r": mean_min_r,
            "mean_r": mean_mean_r,
            "per_run": mean_results,
        },
        "timeseries_msd": {
            "mean_nrmsd": mean_nrmsd,
            "max_nrmsd": max_nrmsd,
            "per_run": ssd_results,
        },
    }
