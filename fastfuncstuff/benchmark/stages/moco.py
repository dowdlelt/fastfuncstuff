"""Motion correction benchmark: 3dvolreg vs ffs_moco."""

from __future__ import annotations

from ..runner import BenchmarkContext, run_timed
from ..validation import (
    compare_1d_params,
    compare_aff12_series,
    compare_moco_ssd,
    compare_volumes,
)

name = "moco"
description = "Motion correction (3dvolreg vs ffs_moco)"

MOTION_PARAM_NAMES = ["roll", "pitch", "yaw", "dS", "dL", "dP"]

THRESHOLDS = {
    # Mean images are the primary alignment quality metric
    "mean_image_min_r": 0.98,
    # Motion params: mean correlation across the 6 columns must clear this.
    # Tolerant of the remaining per-volume Euler-decomposition mismatch but
    # will catch any sign-convention regression (would drive a column to ~-1).
    "motion_param_mean_r": 0.90,
    # aff12 matrices must match direction (base→source, per 3dvolreg.c:1507).
    # A spurious inversion historically drove max translation diff to ~0.3 mm and
    # flipped rotation off-diagonals. These thresholds detect any direction
    # regression while tolerating optimizer divergence on sub-voxel motion.
    "aff12_max_trans_diff_mm": 0.5,
    "aff12_max_rot_diff": 0.05,
}


def _afni_moco_path(ctx: BenchmarkContext, task: str, run: int) -> str:
    return str(ctx.processing_dir / f"afni_moco_{ctx.bids_prefix(task, run)}_bold.nii")


def _ffs_moco_path(ctx: BenchmarkContext, task: str, run: int) -> str:
    return str(ctx.processing_dir / f"{ctx.ffs_prefix}_moco_{ctx.bids_prefix(task, run)}_bold.nii")


def _afni_motion_path(ctx: BenchmarkContext, task: str, run: int) -> str:
    return str(ctx.processing_dir / f"afni_motion_correction_task-{task}_run-{run}.1D")


def _ffs_motion_path(ctx: BenchmarkContext, task: str, run: int) -> str:
    return str(ctx.processing_dir / f"ffs_motion_correction_task-{task}_run-{run}.1D")


def _afni_aff12_path(ctx: BenchmarkContext, task: str, run: int) -> str:
    out = _afni_moco_path(ctx, task, run)
    return out.replace(".nii", "_mat.aff12.1D")


def _ffs_aff12_path(ctx: BenchmarkContext, task: str, run: int) -> str:
    out = _ffs_moco_path(ctx, task, run)
    return out.replace(".nii", "_mat.aff12.1D")


def _afni_mean_path(ctx: BenchmarkContext, task: str, run: int) -> str:
    return str(ctx.processing_dir / f"afni_mean_{ctx.bids_prefix(task, run)}_bold.nii")


def _ffs_mean_path(ctx: BenchmarkContext, task: str, run: int) -> str:
    return str(ctx.processing_dir / f"ffs_mean_{ctx.bids_prefix(task, run)}_bold.nii")


def _input_path(ctx: BenchmarkContext, task: str, run: int) -> str:
    return str(ctx.func_dir / f"{ctx.bids_prefix(task, run)}_bold.nii")


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    """Check that required outputs exist (or inputs exist for execution mode)."""
    from pathlib import Path

    missing = []

    if ctx.validate_only:
        # Need both AFNI and FFS outputs
        for task, runs in ctx.all_task_run_pairs():
            for run in runs:
                for path in [
                    _afni_moco_path(ctx, task, run),
                    _ffs_moco_path(ctx, task, run),
                    _afni_motion_path(ctx, task, run),
                    _ffs_motion_path(ctx, task, run),
                    _afni_aff12_path(ctx, task, run),
                    _ffs_aff12_path(ctx, task, run),
                    _afni_mean_path(ctx, task, run),
                    _ffs_mean_path(ctx, task, run),
                ]:
                    if not Path(path).exists():
                        missing.append(path)
    else:
        # Need raw input data
        for task, runs in ctx.all_task_run_pairs():
            for run in runs:
                inp = _input_path(ctx, task, run)
                if not Path(inp).exists():
                    missing.append(inp)
    return missing


def run_ref(ctx: BenchmarkContext) -> float:
    """Run AFNI 3dvolreg + 3dTstat for all task/run combos."""
    from pathlib import Path

    total = 0.0
    for task, runs in ctx.all_task_run_pairs():
        for run in runs:
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
    for task, runs in ctx.all_task_run_pairs():
        for run in runs:
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
                f"-prefix {out} -save_mean"
                f"{ctx.ffs_device_flag()}",
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
    """Compare motion parameters, aff12 matrices, mean images, and MSD."""
    motion_results = []
    mean_results = []
    ssd_results = []
    aff12_results = []

    for task, runs in ctx.all_task_run_pairs():
        for run in runs:
            # Motion parameters (with per-column names)
            mp = compare_1d_params(
                _afni_motion_path(ctx, task, run),
                _ffs_motion_path(ctx, task, run),
            )
            per_col = mp.get("per_column_r", [])
            mp["per_column"] = {
                MOTION_PARAM_NAMES[i] if i < len(MOTION_PARAM_NAMES) else f"col{i}": r
                for i, r in enumerate(per_col)
            }
            mp["task"] = task
            mp["run"] = run
            motion_results.append(mp)

            # aff12 matrix agreement (catches direction/inversion regressions)
            af = compare_aff12_series(
                _afni_aff12_path(ctx, task, run),
                _ffs_aff12_path(ctx, task, run),
            )
            af["task"] = task
            af["run"] = run
            aff12_results.append(af)

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

    motion_min_r = min(m["min_r"] for m in motion_results)
    motion_mean_r = sum(m["mean_r"] for m in motion_results) / len(motion_results)
    mean_min_r = min(m["r"] for m in mean_results)
    mean_mean_r = sum(m["r"] for m in mean_results) / len(mean_results)

    nrmsd_vals = [s["nrmsd"] for s in ssd_results if "error" not in s]
    mean_nrmsd = sum(nrmsd_vals) / len(nrmsd_vals) if nrmsd_vals else 0.0
    max_nrmsd = max(nrmsd_vals) if nrmsd_vals else 0.0

    valid_aff = [a for a in aff12_results if "error" not in a]
    aff12_max_trans = max((a["max_trans_diff"] for a in valid_aff), default=0.0)
    aff12_max_rot = max((a["max_rot_diff"] for a in valid_aff), default=0.0)

    # PASS conditions: alignment (mean image), param agreement (sign-regression
    # catcher), and aff12 direction (inversion-regression catcher).
    passed = (
        mean_min_r >= THRESHOLDS["mean_image_min_r"]
        and motion_mean_r >= THRESHOLDS["motion_param_mean_r"]
        and aff12_max_trans <= THRESHOLDS["aff12_max_trans_diff_mm"]
        and aff12_max_rot <= THRESHOLDS["aff12_max_rot_diff"]
    )

    return {
        "passed": passed,
        "summary": (
            f"means min_r={mean_min_r:.4f}, motion mean_r={motion_mean_r:.4f}, "
            f"aff12 max_trans={aff12_max_trans:.4f}mm, max_rot={aff12_max_rot:.4f}, "
            f"nrmsd={mean_nrmsd:.4f}"
        ),
        "motion_params": {
            "min_r": motion_min_r,
            "mean_r": motion_mean_r,
            "per_run": motion_results,
        },
        "aff12": {
            "max_trans_diff_mm": aff12_max_trans,
            "max_rot_diff": aff12_max_rot,
            "per_run": aff12_results,
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
