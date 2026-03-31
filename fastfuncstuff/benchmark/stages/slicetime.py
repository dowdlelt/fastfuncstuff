"""Slice timing correction benchmark: 3dTshift vs ffs_slicetime."""

from __future__ import annotations

from pathlib import Path

from ..runner import BenchmarkContext
from ..validation import compare_timeseries_4d

name = "slicetime"
description = "Slice timing correction (3dTshift vs ffs_slicetime)"

THRESHOLDS = {
    "median_r": 0.98,  # slight differences from interpolation kernel implementations
    "frac_above_0.95": 0.99,
}


def _input_path(ctx: BenchmarkContext) -> Path:
    return ctx.func_dir / "sub-01_ses-01_task-localizer_run-1_bold.nii"


def _afni_out(ctx: BenchmarkContext) -> Path:
    return ctx.processing_dir / "afni_tshift_sub-01_ses-01_task-localizer_run-1_bold.nii"


def _ffs_out(ctx: BenchmarkContext) -> Path:
    return ctx.processing_dir / "ffs_tshift_sub-01_ses-01_task-localizer_run-1_bold.nii"


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    missing = []
    if ctx.validate_only:
        if not _afni_out(ctx).exists():
            missing.append(str(_afni_out(ctx)))
        if not _ffs_out(ctx).exists():
            missing.append(str(_ffs_out(ctx)))
    else:
        if not _input_path(ctx).exists():
            missing.append(str(_input_path(ctx)))
        # SliceTiming comes from BIDS JSON — just need the JSON to exist
        json_path = ctx.func_dir / "sub-01_ses-01_task-localizer_run-1_bold.json"
        if not json_path.exists():
            missing.append(str(json_path))
    return missing


def run_ref(ctx: BenchmarkContext) -> float:
    """Run AFNI 3dTshift."""
    from ..runner import run_timed

    out = _afni_out(ctx)
    if out.exists() and not ctx.force_ref:
        return 0.0

    tp = ctx.tpattern_file("localizer", 1)
    elapsed, _ = run_timed(
        f"3dTshift -overwrite "
        f"-prefix {out} "
        f"-tzero 0 "
        f"-tpattern @{tp} "
        f"-wsinc9 "
        f"{_input_path(ctx)}",
        label="3dTshift localizer run-1",
        cwd=ctx.processing_dir,
    )
    return elapsed


def run_ffs(ctx: BenchmarkContext) -> float:
    """Run ffs_slicetime."""
    from ..runner import run_timed

    out = _ffs_out(ctx)
    if out.exists() and not ctx.force_ffs:
        return 0.0

    tp = ctx.tpattern_file("localizer", 1)
    elapsed, _ = run_timed(
        f"ffs_slicetime "
        f"-input {_input_path(ctx)} "
        f"-tzero 0 "
        f"-wsinc9 "
        f"-tpattern {tp} "
        f"-prefix {out}",
        label="ffs_slicetime localizer run-1",
        cwd=ctx.processing_dir,
    )
    return elapsed


def validate(ctx: BenchmarkContext) -> dict:
    """Compare slice-timing corrected timeseries voxel-by-voxel."""
    afni = _afni_out(ctx)
    ffs = _ffs_out(ctx)

    result = compare_timeseries_4d(afni, ffs, sample_frac=0.2)

    if "error" in result:
        return {"passed": False, "summary": result["error"], **result}

    passed = (
        result["median_r"] >= THRESHOLDS["median_r"]
        and result["frac_above_0.95"] >= THRESHOLDS["frac_above_0.95"]
    )

    return {
        "passed": passed,
        "summary": (
            f"median_r={result['median_r']:.4f}, "
            f"frac>0.95={result['frac_above_0.95']:.3f}"
        ),
        **result,
    }
