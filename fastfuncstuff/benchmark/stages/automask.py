"""Automask benchmark: 3dAutomask vs ffs_util_automask.

Compares brain masks generated from mean EPI images across all runs.
Uses Dice coefficient as the primary agreement metric.
"""

from __future__ import annotations

from pathlib import Path

from ..runner import BenchmarkContext, run_timed
from ..validation import compare_masks

name = "automask"
description = "Brain masking (3dAutomask vs ffs_util_automask)"

TASKS = ["localizer", "rest"]
RUNS = [1, 2, 3, 4, 5]

THRESHOLDS = {
    "min_dice": 0.90,  # masks should agree on at least 90% of voxels
}


def _mean_path(ctx: BenchmarkContext, task: str, run: int) -> Path:
    """AFNI mean image (input to both masking tools)."""
    return ctx.processing_dir / f"afni_mean_sub-01_ses-01_task-{task}_run-{run}_bold.nii"


def _afni_mask(ctx: BenchmarkContext, task: str, run: int) -> Path:
    return ctx.processing_dir / f"automask_afni_mean_sub-01_ses-01_task-{task}_run-{run}_bold.nii"


def _ffs_mask(ctx: BenchmarkContext, task: str, run: int) -> Path:
    return ctx.processing_dir / f"automask_ffs_mean_sub-01_ses-01_task-{task}_run-{run}_bold.nii"


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    missing = []
    if ctx.validate_only:
        for task in TASKS:
            for run in RUNS:
                for p in [_afni_mask(ctx, task, run), _ffs_mask(ctx, task, run)]:
                    if not p.exists():
                        missing.append(str(p))
    else:
        # Need mean images from moco stage
        for task in TASKS:
            for run in RUNS:
                src = _mean_path(ctx, task, run)
                if not src.exists():
                    missing.append(str(src))
    return missing


def run_ref(ctx: BenchmarkContext) -> float:
    """Run AFNI 3dAutomask for all mean images."""
    total = 0.0
    for task in TASKS:
        for run in RUNS:
            out = _afni_mask(ctx, task, run)
            if out.exists() and not ctx.force_ref:
                continue
            src = _mean_path(ctx, task, run)
            elapsed, _ = run_timed(
                f"3dAutomask -overwrite -prefix {out} {src}",
                label=f"3dAutomask {task} run-{run}",
                cwd=ctx.processing_dir,
            )
            total += elapsed
    return total


def run_ffs(ctx: BenchmarkContext) -> float:
    """Run ffs_util_automask for all mean images."""
    total = 0.0
    for task in TASKS:
        for run in RUNS:
            out = _ffs_mask(ctx, task, run)
            if out.exists() and not ctx.force_ffs:
                continue
            src = _mean_path(ctx, task, run)
            elapsed, _ = run_timed(
                f"ffs_util_automask -input {src} -prefix {out}",
                label=f"ffs_util_automask {task} run-{run}",
                cwd=ctx.processing_dir,
            )
            total += elapsed
    return total


def validate(ctx: BenchmarkContext) -> dict:
    """Compare masks using Dice coefficient."""
    results = []

    for task in TASKS:
        for run in RUNS:
            afni_p = _afni_mask(ctx, task, run)
            ffs_p = _ffs_mask(ctx, task, run)

            label = f"{task}_run-{run}"

            if afni_p.exists() and ffs_p.exists():
                m = compare_masks(afni_p, ffs_p)
                m["label"] = label
                results.append(m)

    if not results:
        return {
            "passed": False,
            "summary": "No mask comparisons available",
        }

    min_dice = min(r["dice"] for r in results)
    mean_dice = sum(r["dice"] for r in results) / len(results)

    passed = min_dice >= THRESHOLDS["min_dice"]

    return {
        "passed": passed,
        "summary": f"min Dice={min_dice:.4f}, mean Dice={mean_dice:.4f}",
        "min_dice": min_dice,
        "mean_dice": mean_dice,
        "per_run": results,
    }
