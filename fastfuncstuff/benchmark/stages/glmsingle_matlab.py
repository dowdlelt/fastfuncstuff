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

Per-model timings (Types A/B/C/D) are parsed from the MATLAB stdout log
and saved to ``glmsingle/glmsingle_timings.json`` so that the downstream
glmsingle_hrf / glmsingle_denoise / glmsingle_ridge stages can report
them as reference timings for head-to-head comparison against the FFS
tools.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ..runner import BenchmarkContext

name = "glmsingle_matlab"
description = "Run MATLAB GLMsingle (reference implementation)"


# Patterns matching the per-model timing lines emitted by GLMsingle (Types
# C/D are native; Types A/B come from print statements added to the
# comparison script). Type A is captured for completeness but not consumed
# by any downstream stage yet.
_TIMING_PATTERNS: dict[str, re.Pattern[str]] = {
    "type_a": re.compile(r"TYPE A DONE,\s*total time:\s*([\d.]+)\s*seconds"),
    "type_b": re.compile(r"TYPE B HRF fit DONE,\s*total time:\s*([\d.]+)\s*seconds"),
    "type_c": re.compile(r"Finished processing model 3\.\s*Time taken:\s*([\d.]+)\s*seconds"),
    "type_d": re.compile(r"Finished processing model 4\.\s*Time taken:\s*([\d.]+)\s*seconds"),
}

TIMINGS_FILENAME = "glmsingle_timings.json"


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

    # Need MNI-resampled inputs (either ffs or afni variant)
    task = _primary_task(ctx)
    for r in _runs(ctx):
        ffs_f = ctx.processing_dir / f"ffs_mni_resampled_task-{task}_run-{r}.nii.gz"
        afni_f = ctx.processing_dir / f"afni_mni_resampled_task-{task}_run-{r}.nii.gz"
        if not ffs_f.exists() and not afni_f.exists():
            missing.append(str(ffs_f))

    # If validate-only, check that outputs already exist
    if ctx.validate_only:
        gs = ctx.glmsingle_dir
        if not (gs / "glmsingle_hrf_index.nii.gz").exists():
            missing.append(f"GLMsingle: {gs / 'glmsingle_hrf_index.nii.gz'}")

    return missing


def _run_matlab_with_log(cmd: str, log_file: Path, cwd: Path) -> float:
    """Run MATLAB, capturing the full combined stdout+stderr to ``log_file``.

    Unlike ``runner.run_timed`` which truncates streamed output to a 200-line
    tail, this writes every line to disk so the timing-line regexes can
    match anywhere in the (multi-thousand-line) MATLAB log.
    """
    from ..runner import _show_output

    print("  Running: MATLAB GLMsingle...")
    print(f"    Command: {cmd}")

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    start = time.monotonic()

    with open(log_file, "w") as logf:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            if _show_output:
                sys.stdout.write(line)
                sys.stdout.flush()
        proc.wait()

    elapsed = time.monotonic() - start

    if proc.returncode != 0:
        try:
            tail = log_file.read_text(errors="replace")[-500:]
        except OSError:
            tail = "(no log)"
        raise RuntimeError(
            f"Command failed (MATLAB GLMsingle): exit code {proc.returncode}\n{tail}"
        )

    print(f"  Done: MATLAB GLMsingle ({elapsed:.1f}s)")
    return elapsed


def _parse_matlab_timings(log_file: Path) -> dict[str, float]:
    """Extract per-model timings from a MATLAB run log.

    Returns a dict with whichever of ``type_a/b/c/d`` keys were found.
    Returns ``{}`` if the log is unreadable.
    """
    if not log_file.exists():
        return {}
    try:
        text = log_file.read_text(errors="replace")
    except OSError:
        return {}
    found: dict[str, float] = {}
    for key, pat in _TIMING_PATTERNS.items():
        m = pat.search(text)
        if m:
            try:
                found[key] = float(m.group(1))
            except ValueError:
                pass
    return found


def load_timings(ctx: BenchmarkContext) -> dict[str, float]:
    """Load the per-model timings JSON, if present. Empty dict otherwise."""
    tf = ctx.glmsingle_dir / TIMINGS_FILENAME
    if not tf.exists():
        return {}
    try:
        data = json.loads(tf.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in data.items():
        if isinstance(v, (int, float)):
            out[str(k)] = float(v)
    return out


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

    log_file = gs / "matlab_run.log"
    elapsed = _run_matlab_with_log(matlab_cmd, log_file, ctx.data_dir)

    # Parse per-model timings out of the log and persist for downstream
    # stages. Missing/unmatched keys are silently omitted — the regex may
    # not match on older MATLAB scripts that don't print Type A/B lines.
    timings = _parse_matlab_timings(log_file)
    if timings:
        try:
            (gs / TIMINGS_FILENAME).write_text(json.dumps(timings, indent=2))
        except OSError as e:
            print(f"  WARNING: could not write {TIMINGS_FILENAME}: {e}")
        if ctx.verbose:
            parts = [f"{k}={v:.1f}s" for k, v in sorted(timings.items())]
            print(f"  Parsed GLMsingle per-model timings: {', '.join(parts)}")
    elif ctx.verbose:
        print("  WARNING: no per-model timings parsed from MATLAB log")

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
