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
    force_ref: bool = False
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
        """Preprocessing outputs: moco, slicetime, alignment, warps, resampled data."""
        return self.data_dir / "processing"

    @property
    def func_dir(self) -> Path:
        return self.data_dir / "sub-01" / "ses-01" / "func"

    @property
    def anat_dir(self) -> Path:
        return self.data_dir / "sub-01" / "ses-01" / "anat"

    def tpattern_file(self, task: str = "localizer", run: int = 1) -> Path:
        """Get or create a tpattern file from the BIDS JSON sidecar.

        Reads SliceTiming from the paired JSON and writes one value per line.
        Cached in processing_dir so it's only created once.
        """
        cached = self.processing_dir / f"tpattern_{task}_run-{run}.txt"
        if cached.exists():
            return cached

        import json

        json_path = (
            self.func_dir
            / f"sub-01_ses-01_task-{task}_run-{run}_bold.json"
        )
        if not json_path.exists():
            raise FileNotFoundError(f"BIDS JSON not found: {json_path}")

        with open(json_path) as f:
            meta = json.load(f)

        st = meta.get("SliceTiming")
        if not st:
            raise ValueError(f"No SliceTiming in {json_path}")

        self.processing_dir.mkdir(parents=True, exist_ok=True)
        with open(cached, "w") as f:
            for val in st:
                f.write(f"{val}\n")
        return cached

    @property
    def glmsingle_dir(self) -> Path:
        """MATLAB GLMsingle outputs (exported NIfTIs)."""
        return self.data_dir / "glmsingle"

    @property
    def ffs_hrfopt_dir(self) -> Path:
        """FFS HRF optimization outputs (Type B)."""
        return self.data_dir / "ffs_hrfopt"

    @property
    def ffs_denoise_dir(self) -> Path:
        """FFS PC denoising outputs (Type C)."""
        return self.data_dir / "ffs_denoise"

    @property
    def ffs_ridge_dir(self) -> Path:
        """FFS fracridge outputs (Type D)."""
        return self.data_dir / "ffs_ridge"

    @property
    def afni_glm_dir(self) -> Path:
        """AFNI 3dDeconvolve/3dREMLfit outputs."""
        return self.data_dir / "afni_glm"

    @property
    def ffs_glm_dir(self) -> Path:
        """FFS OLS/REML outputs."""
        return self.data_dir / "ffs_glm"

    @property
    def melodic_ica_dir(self) -> Path:
        """MELODIC ICA outputs."""
        return self.data_dir / "melodic_ica"

    @property
    def ffs_ica_dir(self) -> Path:
        """FFS ICA outputs."""
        return self.data_dir / "ffs_ica"

    @property
    def timing_dir(self) -> Path:
        """Onset timing files."""
        return self.processing_dir / "timing_files"

    def has_afni(self) -> bool:
        return shutil.which("3dvolreg") is not None

    def has_melodic(self) -> bool:
        return shutil.which("melodic") is not None


@dataclass
class StageResult:
    """Result from running one benchmark stage."""

    stage_name: str
    ref_time: float | None = None  # seconds, None if skipped/cached
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


def _format_timing(result: StageResult, stage_name: str, data_dir: Path) -> str:
    """Format timing info for per-stage output, falling back to cached refs when needed."""
    ffs_t = result.ffs_time
    ref_t = result.ref_time

    ref_ran = ref_t is not None and ref_t > 0
    ffs_ran = ffs_t is not None and ffs_t > 0

    if ref_ran and ffs_ran:
        # Both measured this run
        speedup = ref_t / ffs_t
        return f" | Ref={ref_t:.1f}s FFS={ffs_t:.1f}s ({speedup:.1f}x)"

    if not ffs_ran:
        return ""

    # FFS ran but ref was skipped — look up cached ref timings across all CPU archs
    from .arch import get_ref_arch_id
    from .timing_cache import get_ref_timings_all_archs

    all_refs = get_ref_timings_all_archs(data_dir, stage_name)
    if not all_refs:
        return f" | FFS={ffs_t:.1f}s (no cached ref)"

    my_ref_id = get_ref_arch_id()
    local = next(((a, r) for a, r in all_refs if a == my_ref_id), None)
    others = [(a, r) for a, r in all_refs if a != my_ref_id]

    parts = [f"FFS={ffs_t:.1f}s"]
    if local:
        _, lr = local
        parts.append(f"cached ref={lr:.1f}s ({lr / ffs_t:.1f}x) [this machine]")
    for other_id, other_ref in others:
        parts.append(f"cached ref={other_ref:.1f}s ({other_ref / ffs_t:.1f}x) [{other_id}]")

    return " | " + " | ".join(parts)


_glmsingle_exported = False  # Module-level flag to avoid re-exporting


def _ensure_glmsingle_niftis(ctx: BenchmarkContext) -> None:
    """Auto-export GLMsingle .mat results to NIfTI if not already done."""
    global _glmsingle_exported
    if _glmsingle_exported:
        return

    nifti_dir = ctx.glmsingle_dir
    # .mat file can be in glmsingle/ or processing/ (legacy)
    mat_file = ctx.glmsingle_dir / "glmsingle_comparison.mat"
    if not mat_file.exists():
        mat_file = ctx.processing_dir / "glmsingle_comparison.mat"
    check_file = nifti_dir / "glmsingle_hrf_index.nii.gz"

    if check_file.exists():
        _glmsingle_exported = True
        return

    if not mat_file.exists():
        return  # Will be caught by check_prerequisites

    # Find a template NIfTI for the affine
    template = ctx.processing_dir / "ffs_mni_resampled_task-localizer_run-1.nii.gz"
    if not template.exists():
        return

    print("  Auto-exporting GLMsingle .mat → NIfTI...")
    from .stages.glmsingle_export import export_glmsingle_niftis

    export_glmsingle_niftis(mat_file, template, nifti_dir)
    _glmsingle_exported = True


def run_stages(
    stages: list,
    ctx: BenchmarkContext,
) -> list[StageResult]:
    """Run a list of benchmark stages.

    Each stage module must have:
        - name: str
        - check_prerequisites(ctx) -> list[str]
        - validate(ctx) -> dict
        - run_ref(ctx) -> float  (reference tool: AFNI, melodic, MATLAB, etc.)
        - run_ffs(ctx) -> float  (FFS tool)
    """
    results = []
    stage_timings = {}

    for stage in stages:
        name = stage.name
        print(f"\n{'='*60}")
        print(f"Stage: {name}")
        print(f"{'='*60}")

        # Auto-export GLMsingle NIfTIs from .mat if needed
        if name.startswith("glmsingle_"):
            _ensure_glmsingle_niftis(ctx)

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

        # Run reference tool (if not validate-only)
        if not ctx.validate_only and hasattr(stage, "run_ref"):
            try:
                result.ref_time = stage.run_ref(ctx)
            except Exception as e:
                result.errors.append(f"Ref: {e}")
                print(f"  Ref error: {e}")

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
        timing = _format_timing(result, name, ctx.data_dir)
        print(f"  {status}: {result.summary}{timing}")

        results.append(result)

        # Collect timings for cache (only non-zero real timings)
        timing_entry = {}
        if result.ref_time is not None and result.ref_time > 0:
            timing_entry["ref_seconds"] = result.ref_time
        if result.ffs_time is not None and result.ffs_time > 0:
            timing_entry["ffs_seconds"] = result.ffs_time
        if timing_entry:
            stage_timings[name] = timing_entry

    # Append this run to the timing cache
    if stage_timings and not ctx.validate_only:
        from .timing_cache import append_run

        append_run(ctx.data_dir, stage_timings, dataset_id=ctx.dataset_id)

    return results
