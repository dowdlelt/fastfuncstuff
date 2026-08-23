"""Core benchmark orchestration."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Module-level flag: print subprocess stdout/stderr when True.
# Set by run_stages() from ctx.show_output so all stage run_timed() calls
# inherit it without needing to thread the flag through every call site.
_show_output: bool = False


@dataclass
class BenchmarkContext:
    """Shared state for all benchmark stages."""

    data_dir: Path  # Dataset root (always resolved to absolute)
    dataset_id: str = ""  # e.g. "ds005165" — auto-detected if empty
    force_ref: bool = False
    force_ffs: bool = False
    validate_only: bool = False
    ref_only: bool = False  # run reference tools only; skip FFS and validation
    verbose: bool = True
    device: str | None = None  # PyTorch device to pass to FFS tools (e.g. "mps", "cpu")
    show_output: bool = False  # Stream subprocess stdout/stderr when True
    config: Any = None  # BenchmarkConfig — typed as Any to avoid circular import

    # Per-stage scratch: how many items each role (ref/ffs) actually ran vs the
    # full set. The runner resets this before each stage; stages fill it via
    # note_items() so a partial rerun's timing can be flagged (and kept out of
    # cached baselines). Not part of the public config surface.
    _timing_meta: dict[str, dict[str, int]] = field(default_factory=dict)

    def note_items(self, role: str, ran: int, total: int) -> None:
        """Record items executed vs total for a role ("ref" or "ffs").

        Lets the runner mark a stage's timing ``partial`` when some runs were
        skipped because their outputs already existed -- a partial time must not
        become the cached full-stage baseline.
        """
        self._timing_meta[role] = {"ran": int(ran), "total": int(total)}

    def ffs_device_flag(self) -> str:
        """Return ' -device <device>' for appending to FFS CLI command strings, or ''."""
        return f" -device {self.device}" if self.device else ""

    def ffs_afni_mode_flag(self) -> str:
        """Return ' -afni_mode' for ffs_reml REML/ARMA stages.

        These stages exist to validate against AFNI 3dREMLfit, so the noise model
        is run in AFNI-faithful mode (banded R, AFNI ltop, corcut grid filter) for
        an apples-to-apples (a,b)/t-stat comparison. FFS's more-accurate defaults
        would otherwise show small, expected divergences from the reference.
        """
        return " -afni_mode"

    @property
    def ffs_tag(self) -> str:
        """File/directory name tag identifying the FFS device variant.

        CUDA keeps the historical empty tag. CPU and MPS outputs are tagged so
        explicit or auto-selected backends cannot overwrite one another.
        """
        arch_id = self.ffs_arch_id
        if arch_id == "cpu":
            return "_cpu"
        if arch_id.startswith("mps-"):
            return "_mps"
        return ""

    @property
    def ffs_arch_id(self) -> str:
        """Architecture ID for the device this benchmark actually requested."""
        from .arch import get_ffs_arch_id

        return get_ffs_arch_id(self.device)

    @property
    def ffs_prefix(self) -> str:
        """Filename prefix for FFS outputs written into shared dirs.

        ``"ffs"`` by default, ``"ffs_cpu"`` for CPU runs. Use this when a
        stage writes ``ffs_*`` files into ``processing_dir`` alongside AFNI
        outputs (e.g. ``ffs_mni_*``, ``ffs_tshift_*``).
        """
        return f"ffs{self.ffs_tag}"

    def __post_init__(self):
        self.data_dir = self.data_dir.resolve()
        if not self.dataset_id:
            if self.config and self.config.dataset_id:
                self.dataset_id = self.config.dataset_id
            else:
                # Auto-detect from directory name (e.g. "ds005165-download" -> "ds005165")
                name = self.data_dir.name
                for suffix in ("-download", "_download", "-data"):
                    if name.endswith(suffix):
                        name = name[: -len(suffix)]
                        break
                self.dataset_id = name

    # ------------------------------------------------------------------
    # Config-driven properties (fall back to ds005165 defaults)
    # ------------------------------------------------------------------

    @property
    def subject(self) -> str:
        return self.config.subject if self.config else "01"

    @property
    def session(self) -> str:
        return self.config.session if self.config else "01"

    @property
    def tasks(self) -> dict[str, list[int]]:
        if self.config and self.config.tasks:
            return self.config.tasks
        return {"localizer": [1, 2, 3, 4, 5], "rest": [1, 2, 3, 4, 5]}

    def task_names(self) -> list[str]:
        """Sorted list of task names."""
        return sorted(self.tasks.keys())

    def runs_for_task(self, task: str) -> list[int]:
        """Run numbers for a specific task."""
        return self.tasks.get(task, [])

    def all_task_run_pairs(self) -> list[tuple[str, list[int]]]:
        """All (task_name, run_list) pairs, sorted by task name."""
        return [(t, self.tasks[t]) for t in self.task_names()]

    def get_stage_params(self, stage_name: str) -> dict[str, Any]:
        """Get stage-specific parameters from the config."""
        if self.config and self.config.stage_params:
            return self.config.stage_params.get(stage_name, {})
        return {}

    def bids_prefix(self, task: str, run: int) -> str:
        """BIDS filename prefix: sub-{subject}_ses-{session}_task-{task}_run-{run}."""
        return f"sub-{self.subject}_ses-{self.session}_task-{task}_run-{run}"

    # ------------------------------------------------------------------
    # Directory properties
    # ------------------------------------------------------------------

    @property
    def processing_dir(self) -> Path:
        """Preprocessing outputs: moco, slicetime, alignment, warps, resampled data."""
        return self.data_dir / "processing"

    @property
    def func_dir(self) -> Path:
        return self.data_dir / f"sub-{self.subject}" / f"ses-{self.session}" / "func"

    @property
    def anat_dir(self) -> Path:
        return self.data_dir / f"sub-{self.subject}" / f"ses-{self.session}" / "anat"

    def tpattern_file(self, task: str = "localizer", run: int = 1) -> Path:
        """Get or create a tpattern file from the BIDS JSON sidecar.

        Reads SliceTiming from the paired JSON and writes one value per line.
        Cached in processing_dir so it's only created once.
        """
        cached = self.processing_dir / f"tpattern_{task}_run-{run}.txt"
        if cached.exists():
            return cached

        import json

        json_path = self.func_dir / f"{self.bids_prefix(task, run)}_bold.json"
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
        return self.data_dir / f"ffs_hrfopt{self.ffs_tag}"

    @property
    def ffs_denoise_dir(self) -> Path:
        """FFS PC denoising outputs (Type C)."""
        return self.data_dir / f"ffs_denoise{self.ffs_tag}"

    @property
    def ffs_ridge_dir(self) -> Path:
        """FFS fracridge outputs (Type D)."""
        return self.data_dir / f"ffs_ridge{self.ffs_tag}"

    @property
    def afni_glm_dir(self) -> Path:
        """AFNI 3dDeconvolve/3dREMLfit outputs."""
        return self.data_dir / "afni_glm"

    @property
    def ffs_glm_dir(self) -> Path:
        """FFS OLS/REML outputs."""
        return self.data_dir / f"ffs_glm{self.ffs_tag}"

    @property
    def melodic_ica_dir(self) -> Path:
        """MELODIC ICA outputs."""
        return self.data_dir / "melodic_ica"

    @property
    def ffs_ica_dir(self) -> Path:
        """FFS ICA outputs."""
        return self.data_dir / f"ffs_ica{self.ffs_tag}"

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
    # True when validation inputs were missing: the stage didn't fail its
    # comparison, it just couldn't run a full comparison (e.g. an upstream stage
    # hasn't produced its output yet). Reported as INCOMPLETE, not FAIL.
    incomplete: bool = False
    # {role: {"ran": n, "total": m}} for roles that ran only some items this
    # invocation (partial timing). Excluded from cached baselines.
    partial: dict[str, dict[str, int]] = field(default_factory=dict)
    # True when the FFS tool (the thing under test) raised while running this
    # invocation. Validation may still "pass" against stale outputs from an
    # earlier run, so a run crash must hard-override to FAIL -- otherwise a
    # command that never produced fresh output is silently reported PASS.
    ffs_crashed: bool = False

    @property
    def status(self) -> str:
        if self.ffs_crashed:
            return "FAIL"
        if self.incomplete:
            return "INCOMPLETE"
        return "PASS" if self.passed else "FAIL"


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
        print(f"    Command: {cmd}")

    start = time.monotonic()

    if _show_output:
        # Stream stdout/stderr line-by-line as the subprocess runs so the
        # user sees progress (tqdm bars, banners) in real time. Capture into
        # tail buffers so we can still report a useful error on non-zero exit.
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        max_tail = 200  # lines kept for error reporting

        # Child Python processes block-buffer stdout when the writer is a
        # pipe (not a tty), which means print() output sits in libc buffers
        # for minutes during slow CPU work. PYTHONUNBUFFERED forces every
        # write to flush immediately so tqdm + banners arrive in real time.
        import os

        env = {**os.environ, "PYTHONUNBUFFERED": "1"}

        proc = subprocess.Popen(
            cmd,
            shell=isinstance(cmd, str),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )

        import selectors

        sel = selectors.DefaultSelector()
        sel.register(proc.stdout, selectors.EVENT_READ, ("stdout", sys.stdout, stdout_lines))
        sel.register(proc.stderr, selectors.EVENT_READ, ("stderr", sys.stderr, stderr_lines))

        open_streams = 2
        while open_streams > 0:
            for key, _ in sel.select(timeout=0.5):
                _, sink, buf = key.data
                # key.fileobj is `int | HasFileno` per selectors' stub (any
                # fileno()-having registerable); we only ever register the
                # actual proc.stdout/proc.stderr text streams above.
                line = key.fileobj.readline()  # ty: ignore[unresolved-attribute]
                if not line:
                    sel.unregister(key.fileobj)
                    open_streams -= 1
                    continue
                sink.write(line)
                sink.flush()
                buf.append(line)
                if len(buf) > max_tail:
                    del buf[: len(buf) - max_tail]

        proc.wait()
        elapsed = time.monotonic() - start
        stdout_tail = "".join(stdout_lines)
        stderr_tail = "".join(stderr_lines)
        returncode = proc.returncode
    else:
        result = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            cwd=str(cwd),
            capture_output=True,
            text=True,
        )
        elapsed = time.monotonic() - start
        stdout_tail = result.stdout or ""
        stderr_tail = result.stderr or ""
        returncode = result.returncode

    if returncode != 0:
        tail = stderr_tail[-500:] if stderr_tail else "(no stderr)"
        raise RuntimeError(f"Command failed ({label}): exit code {returncode}\n{tail}")

    if verbose:
        print(f"  Done: {label} ({elapsed:.1f}s)")

    # Synthesise a CompletedProcess-ish return value for callers that inspect
    # .stdout / .stderr (only really used for error tails today).
    result = subprocess.CompletedProcess(
        args=cmd, returncode=returncode, stdout=stdout_tail, stderr=stderr_tail
    )
    return elapsed, result


def _format_timing(result: StageResult, stage_name: str, data_dir: Path) -> str:
    """Format timing info for per-stage output, falling back to cached refs when needed."""
    ffs_t = result.ffs_time
    ref_t = result.ref_time

    ref_ran = ref_t is not None and ref_t > 0
    ffs_ran = ffs_t is not None and ffs_t > 0

    def _partial(role: str) -> str:
        m = result.partial.get(role)
        return f" (partial {m['ran']}/{m['total']})" if m else ""

    if ref_ran and ffs_ran:
        # Both measured this run
        speedup = ref_t / ffs_t
        return (
            f" | Ref={ref_t:.1f}s{_partial('ref')} "
            f"FFS={ffs_t:.1f}s{_partial('ffs')} ({speedup:.1f}x)"
        )

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

    parts = [f"FFS={ffs_t:.1f}s{_partial('ffs')}"]
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

    # Find a template NIfTI for the affine (prefer ffs, fall back to afni variant)
    template = ctx.processing_dir / "ffs_mni_resampled_task-localizer_run-1.nii.gz"
    if not template.exists():
        template = ctx.processing_dir / "afni_mni_resampled_task-localizer_run-1.nii.gz"
    if not template.exists():
        return

    print("  Auto-exporting GLMsingle .mat → NIfTI...")
    from .stages.glmsingle_export import export_glmsingle_niftis

    export_glmsingle_niftis(mat_file, template, nifti_dir)
    _glmsingle_exported = True


def _missing_validation_inputs(stage: Any, ctx: BenchmarkContext) -> list[str]:
    """Files a stage's validate() will read that don't exist yet.

    A stage opts in by defining ``validation_inputs(ctx) -> list[str|Path]``.
    Stages without it return [] (no preflight; the hardened compare_* helpers
    still keep validate() from crashing on a missing file).
    """
    fn = getattr(stage, "validation_inputs", None)
    if fn is None:
        return []
    try:
        return [str(p) for p in fn(ctx) if not Path(p).exists()]
    except Exception:
        return []


def _missing_input_hint(stage: Any, missing: list[str]) -> str:
    """One-line actionable hint for missing validation inputs.

    Names the upstream stage(s) that produce them when the stage declares
    ``requires`` (see P1), else a generic pointer.
    """
    requires = getattr(stage, "requires", None)
    if requires:
        return (
            f"INCOMPLETE: run upstream stage(s) first: {' '.join(requires)} (then '{stage.name}')"
        )
    return "INCOMPLETE: a needed input is missing (run the upstream stage that produces it)"


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
    global _show_output
    _show_output = ctx.show_output
    results = []
    stage_timings = {}

    # Up-front dependency advisory: if a requested stage depends on stages not in
    # this run, its upstream outputs may be missing -> it'll come back INCOMPLETE.
    # Say so now (0s) rather than after the expensive work.
    from .stages import unsatisfied_deps

    deps = unsatisfied_deps([s.name for s in stages])
    if deps:
        print("\nNote: some stages depend on stages not in this run:")
        for sname, missing in deps.items():
            print(
                f"  {sname} requires: {', '.join(missing)} "
                f"(run them first, or use --with-deps / -stages to include them)"
            )

    for stage in stages:
        name = stage.name
        print(f"\n{'=' * 60}")
        print(f"Stage: {name}")
        print(f"{'=' * 60}")

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
        ctx._timing_meta = {}  # stages fill via note_items(); read after run

        # Run reference tool (if not validate-only)
        if not ctx.validate_only and hasattr(stage, "run_ref"):
            try:
                result.ref_time = stage.run_ref(ctx)
            except Exception as e:
                result.errors.append(f"Ref: {e}")
                print(f"  Ref error: {e}")

        # In ref_only mode, skip FFS only for stages that have a ref tool to time.
        # Prep-only stages (run_ffs but no run_ref) must still run in ref_only mode
        # because they create intermediate files that downstream ref stages depend on.
        has_ref = hasattr(stage, "run_ref")
        skip_ffs = ctx.validate_only or (ctx.ref_only and has_ref)
        if not skip_ffs and hasattr(stage, "run_ffs"):
            try:
                result.ffs_time = stage.run_ffs(ctx)
            except Exception as e:
                result.errors.append(f"FFS: {e}")
                result.ffs_crashed = True
                print(f"  FFS error: {e}")

        # Snapshot which roles ran only part of their work (some outputs already
        # existed) so the timing is flagged partial in display and cache.
        result.partial = {
            role: m
            for role, m in ctx._timing_meta.items()
            if m.get("total", 0) > 0 and m["ran"] < m["total"]
        }

        # Validate: skip in ref_only mode for stages that have a ref tool
        # (FFS outputs won't exist). Prep-only stages still validate so we
        # know the setup succeeded.
        if not (ctx.ref_only and has_ref):
            # Preflight: which files validate() will read are missing? A stage
            # declares these via validation_inputs(ctx). Missing inputs mean an
            # upstream artifact isn't there (e.g. a stage wasn't run) -- that's
            # INCOMPLETE, not a comparison FAIL, and we say so up front instead
            # of letting a raw I/O error surface after the expensive run.
            missing_inputs = _missing_validation_inputs(stage, ctx)

            try:
                result.validation = stage.validate(ctx)
                result.passed = result.validation.get("passed", True)
                result.summary = result.validation.get("summary", "")
            except Exception as e:
                result.passed = False
                result.errors.append(f"Validation: {e}")
                result.summary = f"Validation error: {e}"
                print(f"  Validation error: {e}")

            if missing_inputs:
                result.incomplete = True
                result.passed = False
                result.errors.extend(missing_inputs)
                shown = ", ".join(Path(m).name for m in missing_inputs[:3])
                more = f" (+{len(missing_inputs) - 3} more)" if len(missing_inputs) > 3 else ""
                detail = f" | {result.summary}" if result.summary else ""
                result.summary = f"missing {len(missing_inputs)} input(s): {shown}{more}{detail}"
                hint = _missing_input_hint(stage, missing_inputs)
                if hint:
                    print(f"  {hint}")
        else:
            result.summary = "ref only"

        # A run-phase crash overrides any validation verdict: validation may have
        # "passed" against stale outputs from an earlier run, but this invocation
        # produced no fresh output, so the stage failed. Surface it plainly instead
        # of a misleading PASS.
        if result.ffs_crashed:
            result.passed = False
            result.incomplete = False
            crash_detail = next(
                (e for e in result.errors if e.startswith("FFS:")), "FFS tool crashed"
            )
            prior = f" (validation on stale output: {result.summary})" if result.summary else ""
            result.summary = f"{crash_detail}{prior}"

        # Print result
        timing = _format_timing(result, name, ctx.data_dir)
        print(f"  {result.status}: {result.summary}{timing}")

        results.append(result)

        # Collect timings for cache (only non-zero real timings). Flag a role as
        # partial when the stage ran fewer than all its items this invocation
        # (some outputs already existed) -- a partial time is not a valid
        # full-stage baseline and is excluded from cached-ref lookups.
        timing_entry: dict[str, Any] = {}
        if result.ref_time is not None and result.ref_time > 0:
            timing_entry["ref_seconds"] = result.ref_time
        if result.ffs_time is not None and result.ffs_time > 0:
            timing_entry["ffs_seconds"] = result.ffs_time
        for role, m in result.partial.items():
            timing_entry[f"{role}_partial"] = True
            timing_entry[f"{role}_ran"] = m["ran"]
            timing_entry[f"{role}_total"] = m["total"]
        if timing_entry:
            stage_timings[name] = timing_entry

    # Append this run to the timing cache (including validate-only runs)
    from .timing_cache import append_run

    stage_results = {r.stage_name: {"passed": r.passed, "summary": r.summary} for r in results}
    stage_validations = {r.stage_name: r.validation for r in results if r.validation}

    # Cache if we have any results (timing or validation)
    if stage_timings or stage_results:
        append_run(
            ctx.data_dir,
            stage_timings,
            dataset_id=ctx.dataset_id,
            config=ctx.config,
            stage_results=stage_results,
            stage_validations=stage_validations,
            device_spec=ctx.device,
        )

    return results
