"""Run MATLAB GLMsingle via `matlab -batch`.

This stage runs the MATLAB script test_data/run_glmsingle_comparison.m
which executes GLMsingle (Types B, C, D) and exports NIfTI files for
benchmark comparison. It's a prerequisite for all glmsingle_* stages.

The MATLAB script has two phases:
  1. Run GLMsingle (skip if .mat exists, unless force)
  2. Export NIfTI files (skip if NIfTIs exist, unless force)

Control via ctx.force_ref (treats MATLAB as the "reference" tool):
  - force_ref=False: skip if outputs exist
  - force_ref=True: rerun + reexport
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..runner import BenchmarkContext, run_timed

name = "glmsingle_matlab"
description = "Run MATLAB GLMsingle (reference implementation)"


def _glm_params(ctx: BenchmarkContext) -> dict:
    return ctx.get_stage_params("glm")


def _primary_task(ctx: BenchmarkContext) -> str:
    return _glm_params(ctx).get("primary_task", "localizer")


def _runs(ctx: BenchmarkContext) -> list[int]:
    return ctx.runs_for_task(_primary_task(ctx))


def _matlab_script(ctx: BenchmarkContext) -> Path:
    """Find the MATLAB comparison script."""
    # Project root: fastfuncstuff/benchmark/stages/glmsingle_matlab.py -> ../../..
    project_root = Path(__file__).resolve().parents[3]
    candidates = [
        project_root / "test_data" / "run_glmsingle_comparison.m",
        ctx.data_dir.parent / "run_glmsingle_comparison.m",
        ctx.data_dir / "run_glmsingle_comparison.m",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # Will be caught by check_prerequisites


def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    missing = []

    # Need MATLAB
    if not shutil.which("matlab"):
        missing.append("matlab not found on PATH")

    # Need the script
    script = _matlab_script(ctx)
    if not script.exists():
        missing.append(f"MATLAB script: {script}")

    # Need MNI-resampled inputs
    task = _primary_task(ctx)
    for r in _runs(ctx):
        f = ctx.processing_dir / f"ffs_mni_resampled_task-{task}_run-{r}.nii.gz"
        if not f.exists():
            missing.append(str(f))

    # If validate-only, check that outputs already exist
    if ctx.validate_only:
        gs = ctx.glmsingle_dir
        if not (gs / "glmsingle_hrf_index.nii.gz").exists():
            missing.append(f"GLMsingle: {gs / 'glmsingle_hrf_index.nii.gz'}")

    return missing


def run_ref(ctx: BenchmarkContext) -> float:
    """Run MATLAB GLMsingle (treated as the 'reference' tool)."""
    script = _matlab_script(ctx)
    gs = ctx.glmsingle_dir
    gs.mkdir(parents=True, exist_ok=True)

    # Build MATLAB command
    # Set rerun/reexport flags based on force_ref
    flags = ""
    if ctx.force_ref:
        flags = "rerun = true; reexport = true; "
    else:
        # Still reexport if NIfTIs are missing
        if not (gs / "glmsingle_hrf_index.nii.gz").exists():
            flags = "reexport = true; "

    matlab_cmd = (
        f"matlab -batch \""
        f"cd('{ctx.data_dir}'); "
        f"{flags}"
        f"run('{script}');\""
    )

    elapsed, _ = run_timed(
        matlab_cmd,
        label="MATLAB GLMsingle",
        cwd=ctx.data_dir,
    )
    return elapsed


def validate(ctx: BenchmarkContext) -> dict:
    """Check that GLMsingle outputs exist and are valid."""
    gs = ctx.glmsingle_dir

    expected = [
        "glmsingle_hrf_index.nii.gz",
        "glmsingle_r2_B.nii.gz",
        "glmsingle_betas_B.nii.gz",
        "glmsingle_noisepool.nii.gz",
        "glmsingle_r2_C.nii.gz",
        "glmsingle_betas_C.nii.gz",
        "glmsingle_pcnum.txt",
        "glmsingle_xvaltrend.txt",
        "glmsingle_fracvalue.nii.gz",
        "glmsingle_r2_D.nii.gz",
        "glmsingle_betas_D.nii.gz",
        "glmsingle_mask.nii.gz",
    ]

    present = []
    missing = []
    for f in expected:
        if (gs / f).exists():
            present.append(f)
        else:
            missing.append(f)

    passed = len(missing) == 0
    summary = f"{len(present)}/{len(expected)} files present"
    if missing:
        summary += f", missing: {', '.join(missing[:3])}"

    return {
        "passed": passed,
        "summary": summary,
        "present": present,
        "missing": missing,
    }
