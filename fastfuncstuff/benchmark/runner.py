"""Core benchmark orchestration."""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkContext:
    """Shared state for all benchmark stages."""

    data_dir: Path  # ds005165-download root (always resolved to absolute)
    dataset_id: str = ""  # e.g. "ds005165" — auto-detected if empty
    force_afni: bool = False
    force_ffs: bool = False
    validate_only: bool = False
    verbose: bool = True

    def __post_init__(self):
        self.data_dir = self.data_dir.resolve()
        if not self.dataset_id:
            # Auto-detect from directory name (e.g. "ds005165-download" -> "ds005165")
            name = self.data_dir.name
            for suffix in ("-download", "_download", "-data"):
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
                    break
            self.dataset_id = name

    @property
    def processing_dir(self) -> Path:
        return self.data_dir / "processing"

    @property
    def func_dir(self) -> Path:
        return self.data_dir / "sub-01" / "ses-01" / "func"

    @property
    def anat_dir(self) -> Path:
        return self.data_dir / "sub-01" / "ses-01" / "anat"

    def has_afni(self) -> bool:
        return shutil.which("3dvolreg") is not None

    def has_melodic(self) -> bool:
        return shutil.which("melodic") is not None


@dataclass
class StageResult:
    """Result from running one benchmark stage."""

    stage_name: str
    afni_time: float | None = None  # seconds, None if skipped/cached
    ffs_time: float | None = None
    validation: dict[str, Any] = field(default_factory=dict)
    passed: bool = True
    summary: str = ""
    errors: list[str] = field(default_factory=list)


def run_timed(
    cmd: str | list[str],
    label: str,
    cwd: Path,
    verbose: bool = True,
) -> tuple[float, subprocess.CompletedProcess]:
    """Run a shell command with wall-clock timing.

    Args:
        cmd: Command string (run with shell=True) or list of args.
        label: Human-readable label for this command.
        cwd: Working directory.
        verbose: Print progress.

    Returns:
        (elapsed_seconds, completed_process)

    Raises:
        RuntimeError: If command returns non-zero exit code.
    """
    if verbose:
        print(f"  Running: {label}...")

    start = time.monotonic()
    result = subprocess.run(
        cmd,
        shell=isinstance(cmd, str),
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - start

    if result.returncode != 0:
        stderr_tail = result.stderr[-500:] if result.stderr else "(no stderr)"
        raise RuntimeError(
            f"Command failed ({label}): exit code {result.returncode}\n{stderr_tail}"
        )

    if verbose:
        print(f"  Done: {label} ({elapsed:.1f}s)")

    return elapsed, result


def run_stages(
    stages: list,
    ctx: BenchmarkContext,
) -> list[StageResult]:
    """Run a list of benchmark stages.

    Each stage module must have:
        - name: str
        - check_prerequisites(ctx) -> list[str]
        - validate(ctx) -> dict
        - run_afni(ctx) -> float  (optional for validate-only)
        - run_ffs(ctx) -> float   (optional for validate-only)
    """
    results = []
    stage_timings = {}

    for stage in stages:
        name = stage.name
        print(f"\n{'='*60}")
        print(f"Stage: {name}")
        print(f"{'='*60}")

        # Check prerequisites
        missing = stage.check_prerequisites(ctx)
        if missing:
            result = StageResult(
                stage_name=name,
                passed=False,
                summary=f"Missing prerequisites: {len(missing)} files",
                errors=missing,
            )
            results.append(result)
            if ctx.verbose:
                for m in missing[:5]:
                    print(f"  Missing: {m}")
                if len(missing) > 5:
                    print(f"  ... and {len(missing) - 5} more")
            print(f"  SKIP: {result.summary}")
            continue

        result = StageResult(stage_name=name)

        # Run AFNI (if not validate-only)
        if not ctx.validate_only and hasattr(stage, "run_afni"):
            try:
                result.afni_time = stage.run_afni(ctx)
            except Exception as e:
                result.errors.append(f"AFNI: {e}")
                print(f"  AFNI error: {e}")

        # Run FFS (if not validate-only)
        if not ctx.validate_only and hasattr(stage, "run_ffs"):
            try:
                result.ffs_time = stage.run_ffs(ctx)
            except Exception as e:
                result.errors.append(f"FFS: {e}")
                print(f"  FFS error: {e}")

        # Validate
        try:
            result.validation = stage.validate(ctx)
            result.passed = result.validation.get("passed", True)
            result.summary = result.validation.get("summary", "")
        except Exception as e:
            result.passed = False
            result.errors.append(f"Validation: {e}")
            result.summary = f"Validation error: {e}"
            print(f"  Validation error: {e}")

        # Print result
        status = "PASS" if result.passed else "FAIL"
        timing = ""
        if result.afni_time is not None and result.ffs_time is not None:
            speedup = result.afni_time / result.ffs_time if result.ffs_time > 0 else float("inf")
            timing = f" | AFNI={result.afni_time:.1f}s FFS={result.ffs_time:.1f}s ({speedup:.1f}x)"
        print(f"  {status}: {result.summary}{timing}")

        results.append(result)

        # Collect timings for cache (only non-zero real timings)
        timing_entry = {}
        if result.afni_time is not None and result.afni_time > 0:
            timing_entry["afni_seconds"] = result.afni_time
        if result.ffs_time is not None and result.ffs_time > 0:
            timing_entry["ffs_seconds"] = result.ffs_time
        if timing_entry:
            stage_timings[name] = timing_entry

    # Update timing cache if we ran anything
    if stage_timings and not ctx.validate_only:
        from .timing_cache import update_cache

        update_cache(ctx.data_dir, stage_timings, dataset_id=ctx.dataset_id)

    return results
