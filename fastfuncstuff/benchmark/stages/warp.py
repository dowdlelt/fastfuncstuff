"""Warping benchmark: 3dNwarpApply vs ffs_nwarp."""

from __future__ import annotations

from pathlib import Path

from ..runner import BenchmarkContext, run_timed
from ..validation import compare_timeseries_4d, compare_volumes

name = "warp"
description = "Warp apply (3dNwarpApply vs ffs_nwarp)"

TASKS_RUNS = [
    ("localizer", [1, 2, 3, 4, 5]),
    ("rest", [1, 2, 3, 4, 5]),
]

THRESHOLDS = {
    "warped_anat_r": 0.80,  # different warping algorithms, expect some disagreement
    "warped_func_min_r": 0.95,
}


def _afni_mni(ctx: BenchmarkContext, task: str, run: int) -> Path:
    return ctx.processing_dir / f"afni_mni_task-{task}_run-{run}.nii.gz"


def _ffs_mni(ctx: BenchmarkContext, task: str, run: int) -> Path:
    return ctx.processing_dir / f"ffs_mni_task-{task}_run-{run}.nii.gz"


def _input_path(ctx: BenchmarkContext, task: str, run: int) -> Path:
    return ctx.func_dir / f"sub-01_ses-01_task-{task}_run-{run}_bold.nii"


def _nwarp_chain(ctx: BenchmarkContext, task: str, run: int) -> str:
    """Build the -nwarp chain string for 3dNwarpApply / ffs_nwarp."""
    p = ctx.processing_dir
    ssw = p / "sswarper_output"

    parts = [
        str(ssw / "anatQQ.sub-01_WARP.nii"),
        str(ssw / "anatQQ.sub-01.aff12.1D"),
        str(p / "anat_al_keep_e2a_only_mat.aff12.1D"),
    ]

    # run-1 localizer uses the moco matrix directly
    # other runs need the inter-run alignment matrix first
    if not (task == "localizer" and run == 1):
        align_mat = p / f"afni_mean_{task}_run-{run}_to_localizer_run-1_mat.aff12.1D"
        parts.append(str(align_mat))

    moco_mat = p / f"afni_moco_sub-01_ses-01_task-{task}_run-{run}_bold_mat.aff12.1D"
    parts.append(str(moco_mat))

    return " ".join(parts)


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    missing = []
    if ctx.validate_only:
        # Warped anatomicals
        afni_anat = ctx.processing_dir / "sswarper_output" / "anatQQ.sub-01.nii"
        ffs_anat = ctx.processing_dir / "ffs_warper" / "anatFFS.sub-01.nii.gz"
        if not afni_anat.exists():
            missing.append(str(afni_anat))
        if not ffs_anat.exists():
            missing.append(str(ffs_anat))

        # Warped functionals
        for task, runs in TASKS_RUNS:
            for run in runs:
                for p in [_afni_mni(ctx, task, run), _ffs_mni(ctx, task, run)]:
                    if not p.exists():
                        missing.append(str(p))
    else:
        # Need moco outputs, sswarper outputs, alignment matrices
        master = ctx.processing_dir / "autobox_anatQQ.sub-01.nii"
        if not master.exists():
            missing.append(str(master))
        e2a = ctx.processing_dir / "anat_al_keep_e2a_only_mat.aff12.1D"
        if not e2a.exists():
            missing.append(str(e2a))
    return missing


def run_afni(ctx: BenchmarkContext) -> float:
    """Run 3dNwarpApply for all runs."""
    master = ctx.processing_dir / "autobox_anatQQ.sub-01.nii"
    total = 0.0

    for task, runs in TASKS_RUNS:
        for run in runs:
            out = _afni_mni(ctx, task, run)
            if out.exists() and not ctx.force_afni:
                continue
            nwarp = _nwarp_chain(ctx, task, run)
            elapsed, _ = run_timed(
                f'3dNwarpApply -overwrite '
                f'-master {master} -dxyz 3.0 -wsinc5 '
                f'-nwarp "{nwarp}" '
                f'-source {_input_path(ctx, task, run)} '
                f'-prefix {out}',
                label=f"3dNwarpApply {task} run-{run}",
                cwd=ctx.processing_dir,
            )
            total += elapsed
    return total


def run_ffs(ctx: BenchmarkContext) -> float:
    """Run ffs_nwarp for all runs."""
    master = ctx.processing_dir / "autobox_anatQQ.sub-01.nii"
    total = 0.0

    for task, runs in TASKS_RUNS:
        for run in runs:
            out = _ffs_mni(ctx, task, run)
            if out.exists() and not ctx.force_ffs:
                continue
            nwarp = _nwarp_chain(ctx, task, run)
            elapsed, _ = run_timed(
                f'ffs_nwarp '
                f'-master {master} -dxyz 3.0 -interp wsinc5 '
                f'-nwarp "{nwarp}" '
                f'-source {_input_path(ctx, task, run)} '
                f'-prefix {out}',
                label=f"ffs_nwarp {task} run-{run}",
                cwd=ctx.processing_dir,
            )
            total += elapsed
    return total


def validate(ctx: BenchmarkContext) -> dict:
    """Compare warped anatomicals and functionals between AFNI and FFS."""
    results = {}

    # 1. Compare warped anatomicals
    afni_anat = ctx.processing_dir / "sswarper_output" / "anatQQ.sub-01.nii"
    ffs_anat = ctx.processing_dir / "ffs_warper" / "anatFFS.sub-01.nii.gz"
    anat_result = compare_volumes(afni_anat, ffs_anat)
    results["anat_r"] = anat_result["r"]

    # 2. Compare warped functionals (mean correlation across runs)
    func_results = []
    for task, runs in TASKS_RUNS:
        for run in runs:
            ts = compare_timeseries_4d(
                _afni_mni(ctx, task, run), _ffs_mni(ctx, task, run),
                sample_frac=0.05,
            )
            ts["task"] = task
            ts["run"] = run
            func_results.append(ts)

    func_median_rs = [r["median_r"] for r in func_results if "median_r" in r]
    func_min_r = min(func_median_rs) if func_median_rs else 0.0
    func_mean_r = sum(func_median_rs) / len(func_median_rs) if func_median_rs else 0.0

    passed = (
        anat_result["r"] >= THRESHOLDS["warped_anat_r"]
        and func_min_r >= THRESHOLDS["warped_func_min_r"]
    )

    return {
        "passed": passed,
        "summary": (
            f"anat r={anat_result['r']:.4f}, "
            f"func min median_r={func_min_r:.4f}"
        ),
        "anat": anat_result,
        "func_min_r": func_min_r,
        "func_mean_r": func_mean_r,
        "func_per_run": func_results,
    }
