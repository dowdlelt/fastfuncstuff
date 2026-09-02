"""Compact CPU and accelerator profiles for FFS benchmark invocations."""

from __future__ import annotations

import cProfile
import json
import platform
import re
import shlex
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

_T = TypeVar("_T")
_SHELL_TOKENS = {"|", "||", "&&", ";", ">", ">>", "<", "2>", "2>>"}


def command_argv(cmd: str | list[str]) -> list[str] | None:
    """Return argv when *cmd* is a directly runnable FFS CLI command."""
    if isinstance(cmd, str):
        try:
            argv = shlex.split(cmd)
        except ValueError:
            return None
        if any(token in _SHELL_TOKENS for token in argv):
            return None
    else:
        argv = [str(part) for part in cmd]
    if not argv:
        return None
    executable = Path(argv[0]).name
    if not executable.startswith("ffs_") or executable == "ffs_benchmark":
        return None
    return argv


def _slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.").lower()
    return value[:80] or "invocation"


def _requested_device(argv: list[str]) -> str | None:
    for index, token in enumerate(argv[:-1]):
        if token in {"-device", "--device"}:
            return argv[index + 1]
    return None


def _hardware_metadata(requested_device: str | None) -> dict[str, Any]:
    import torch

    metadata: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "requested_device": requested_device,
        "cuda_available": torch.cuda.is_available(),
    }
    if _use_cuda(requested_device):
        index = torch.cuda.current_device()
        metadata.update(
            cuda_device=index,
            cuda_name=torch.cuda.get_device_name(index),
            cuda_version=torch.version.cuda,
        )
    return metadata


def _use_cuda(requested_device: str | None) -> bool:
    import torch

    if requested_device and requested_device.split(",", 1)[0].lower() in {"cpu", "mps"}:
        return False
    return torch.cuda.is_available()


def _python_rows(profiler: cProfile.Profile, limit: int | None = 75) -> list[dict[str, Any]]:
    rows = []
    import pstats

    stats = pstats.Stats(profiler)
    stats_data = getattr(stats, "stats", {})
    for (filename, line, function), values in stats_data.items():
        primitive, calls, self_s, cumulative_s, _ = values
        rows.append(
            {
                "file": filename,
                "line": int(line),
                "function": function,
                "calls": int(calls),
                "primitive_calls": int(primitive),
                "self_cpu_seconds": float(self_s),
                "cumulative_cpu_seconds": float(cumulative_s),
            }
        )
    rows.sort(key=lambda row: row["cumulative_cpu_seconds"], reverse=True)
    return rows[:limit] if limit is not None else rows


def _torch_rows(torch_profiler: Any, limit: int = 75) -> list[dict[str, Any]]:
    rows = []
    for event in torch_profiler.key_averages():
        rows.append(
            {
                "operator": event.key,
                "calls": int(event.count),
                "self_cpu_seconds": float(event.self_cpu_time_total) / 1e6,
                "cpu_seconds": float(event.cpu_time_total) / 1e6,
                "self_device_seconds": float(getattr(event, "self_device_time_total", 0.0)) / 1e6,
                "device_seconds": float(getattr(event, "device_time_total", 0.0)) / 1e6,
            }
        )
    # Rank by device and by CPU separately and keep the head of each. One
    # combined max() ranking let the CPU-side operators fill the whole table:
    # a busy run has thousands of cheap launches whose CPU time outranks every
    # individual kernel, so the GPU rows fell off the end and the report looked
    # as though nothing ran on the card.
    by_device = sorted(rows, key=lambda row: row["self_device_seconds"], reverse=True)
    by_cpu = sorted(rows, key=lambda row: row["self_cpu_seconds"], reverse=True)
    kept: dict[str, dict[str, Any]] = {}
    for row in [r for r in by_device if r["self_device_seconds"] > 0][: limit // 2] + by_cpu:
        kept.setdefault(row["operator"], row)
        if len(kept) >= limit:
            break
    return sorted(
        kept.values(),
        key=lambda row: max(row["self_device_seconds"], row["self_cpu_seconds"]),
        reverse=True,
    )


@dataclass
class _SampleSchedule:
    """Duty-cycled Kineto sampling: skip startup, then sample across the run.

    The window used to be the *first* two seconds of the process, which for an
    FFS CLI is imports, argument parsing and the first read -- so the GPU
    columns came back 0.00 for sixteen of eighteen benchmark stages. Nothing
    was wrong with the collection; it was pointed at the wrong part of the run.

    Kineto costs about 3% while collecting and nothing at all while toggled off
    (measured: 0.109 ms/op against a 0.106 ms baseline), so the budget is set by
    how many events are worth carrying, not by overhead. Sampling ``active``
    seconds out of every ``active + idle`` after a ``warmup`` gives coverage of
    every phase a long tool goes through rather than one arbitrary slice of it.
    """

    warmup: float = 1.5
    active: float = 1.0
    idle: float = 3.0
    budget: float = 4.0


_DEFAULT_SAMPLE_SCHEDULE = _SampleSchedule()


def _arm_collection_schedule(
    torch_profiler: Any,
    activities: list[Any],
    schedule: _SampleSchedule | None,
) -> tuple[dict[str, Any], Callable[[], None]]:
    """Duty-cycle Kineto collection on the main thread via SIGALRM.

    Returns the state dict recorded into the profile and a cleanup callable.
    Collection begins switched off and is turned on once ``warmup`` has passed,
    so process startup never fills the buffer.
    """
    import torch

    state: dict[str, Any] = {
        "schedule": None if schedule is None else vars(schedule).copy(),
        "limited": False,
        "started": schedule is None,
        "collection_seconds": None,
        "windows": 0,
        "error": None,
    }
    no_cleanup = lambda: None
    required = ("SIGALRM", "ITIMER_REAL", "getitimer", "setitimer")
    if (
        schedule is None
        or schedule.active <= 0
        or threading.current_thread() is not threading.main_thread()
        or not all(hasattr(signal, name) for name in required)
    ):
        return state, no_cleanup

    timer_kind = signal.ITIMER_REAL
    if signal.getitimer(timer_kind)[0] > 0:
        state["error"] = "SIGALRM timer already active; collected the full invocation"
        state["started"] = True
        return state, no_cleanup

    sigalrm = signal.SIGALRM
    previous_handler = signal.getsignal(sigalrm)
    collected = 0.0
    window_opened: float | None = None
    recording = False

    # Toggling the CUDA activity loses device attribution: a real ffs_moco run
    # came back with 0.0 device seconds on every operator, while the same run
    # left un-toggled reported 0.466 s across 201 operators. CPU events are
    # where the volume is anyway, and Kineto costs ~3% while collecting, so the
    # GPU side simply stays on for the whole invocation.
    cpu_only = [a for a in activities if a == torch.profiler.ProfilerActivity.CPU]

    def toggle(on: bool) -> bool:
        try:
            torch_profiler.toggle_collection_dynamic(on, cpu_only)
        except Exception as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
            return False
        return True

    def tick(_signum: int, _frame: Any) -> None:
        nonlocal collected, window_opened, recording
        if recording:
            if not toggle(False):
                return
            recording = False
            if window_opened is not None:
                collected += time.monotonic() - window_opened
                window_opened = None
            state["collection_seconds"] = collected
            if collected >= schedule.budget:
                state["limited"] = True
                return  # spent: leave collection off for the rest of the run
            signal.setitimer(timer_kind, schedule.idle)
        else:
            if not toggle(True):
                return
            recording = True
            window_opened = time.monotonic()
            state["started"] = True
            state["windows"] += 1
            signal.setitimer(timer_kind, schedule.active)

    signal.signal(sigalrm, tick)
    if schedule.warmup > 0:
        # Off until the warmup elapses, so imports and startup are never sampled.
        if not toggle(False):
            return state, no_cleanup
        signal.setitimer(timer_kind, schedule.warmup)
    else:
        # setitimer(0) *disarms* the timer rather than firing it, so a zero
        # warmup has to open the first window here instead of via the handler.
        recording = True
        window_opened = time.monotonic()
        state["started"] = True
        state["windows"] = 1
        signal.setitimer(timer_kind, schedule.active)

    def cleanup() -> None:
        nonlocal collected, window_opened
        signal.setitimer(timer_kind, 0)
        signal.signal(sigalrm, previous_handler)
        if window_opened is not None:  # the run ended mid-window
            collected += time.monotonic() - window_opened
            window_opened = None
            state["collection_seconds"] = collected

    return state, cleanup


def capture_profile(
    function: Callable[[], Any],
    output_path: Path,
    *,
    command: list[str],
    label: str,
    stage: str,
    trace: bool = False,
    sample_schedule: _SampleSchedule | None = None,
) -> Any:
    """Profile full Python execution and a bounded window of PyTorch events."""
    capture_started = time.monotonic()
    started = time.time()

    import torch

    requested_device = _requested_device(command)
    cuda = _use_cuda(requested_device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    python_profiler = cProfile.Profile()
    torch_profiler = torch.profiler.profile(activities=activities)
    cuda_allocated_before = 0
    if cuda:
        torch.cuda.synchronize()
        cuda_allocated_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

    error: str | None = None
    schedule = None if trace else (sample_schedule or _DEFAULT_SAMPLE_SCHEDULE)
    torch_profiler.__enter__()
    limit_state, cleanup_limit = _arm_collection_schedule(torch_profiler, activities, schedule)
    tool_started = time.monotonic()
    python_profiler.enable()
    try:
        result = function()
    except BaseException as exc:
        if not isinstance(exc, SystemExit) or exc.code not in (None, 0):
            error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        exc_info = sys.exc_info()
        python_profiler.disable()
        if cuda:
            torch.cuda.synchronize()
        tool_seconds = time.monotonic() - tool_started
        cleanup_limit()

        profiler_stop_started = time.monotonic()
        torch_profiler.__exit__(*exc_info)
        profiler_stop_seconds = time.monotonic() - profiler_stop_started

        report_started = time.monotonic()
        trace_path = output_path.with_suffix(".trace.json")
        if trace:
            torch_profiler.export_chrome_trace(str(trace_path))
        all_python_rows = _python_rows(python_profiler, limit=None)
        python_rows = all_python_rows[:75]
        python_io_rows = [row for row in all_python_rows if _is_io_row(row)][:50]
        torch_rows = _torch_rows(torch_profiler)
        hardware = _hardware_metadata(requested_device)
        cuda_memory = None
        if cuda:
            cuda_memory = {
                "allocated_before_bytes": cuda_allocated_before,
                "allocated_after_bytes": torch.cuda.memory_allocated(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }

        collection_seconds = limit_state["collection_seconds"]
        if collection_seconds is None:
            # Either the schedule never opened a window (a tool shorter than the
            # warmup) or there is no schedule at all, as under -trace.
            collection_seconds = tool_seconds if limit_state["started"] else 0.0

        # cProfile only ever sees the thread that enabled it, so worker threads
        # -- the threaded loader, inductor's compile workers -- and the async
        # CUDA tail are invisible to it. Publishing the gap keeps a reader from
        # reading the Python table as if it accounted for the whole run.
        attributed = max((row["cumulative_cpu_seconds"] for row in all_python_rows), default=0.0)
        unattributed = max(0.0, tool_seconds - attributed)
        payload: dict[str, Any] = {
            "schema_version": 3,
            "stage": stage,
            "label": label,
            "command": command,
            "started_at": datetime.fromtimestamp(started).astimezone().isoformat(),
            "tool_seconds": tool_seconds,
            "main_thread_seconds": attributed,
            "unattributed_seconds": unattributed,
            "profiler_stop_seconds": profiler_stop_seconds,
            "torch_sample_schedule": limit_state["schedule"],
            "torch_collection_seconds": collection_seconds,
            "torch_collection_limited": limit_state["limited"],
            "torch_collection_started": limit_state["started"],
            "torch_collection_windows": limit_state["windows"],
            "torch_collection_error": limit_state["error"],
            "error": error,
            "hardware": hardware,
            "python": python_rows,
            "torch": torch_rows,
            "python_io": python_io_rows,
            "trace": trace_path.name if trace else None,
        }
        if cuda_memory is not None:
            payload["cuda_memory"] = cuda_memory
        payload["report_build_seconds"] = time.monotonic() - report_started
        payload["wall_seconds"] = time.monotonic() - capture_started
        output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return result


def _aggregate_rows(
    invocations: list[dict[str, Any]], section: str, key_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    totals: dict[tuple[Any, ...], dict[str, Any]] = {}
    numeric = (
        ("calls", "primitive_calls", "self_cpu_seconds", "cumulative_cpu_seconds")
        if section in {"python", "python_io"}
        else ("calls", "self_cpu_seconds", "cpu_seconds", "self_device_seconds", "device_seconds")
    )
    for invocation in invocations:
        for row in invocation.get(section, []):
            key = tuple(row[metric] for metric in key_fields)
            merged = totals.setdefault(key, {metric: row[metric] for metric in key_fields})
            for metric in numeric:
                if metric in row:
                    merged[metric] = merged.get(metric, 0) + row[metric]
    sort_field = (
        "cumulative_cpu_seconds" if section in {"python", "python_io"} else "self_device_seconds"
    )
    fallback = "self_cpu_seconds"
    return sorted(
        totals.values(),
        key=lambda row: max(row.get(sort_field, 0), row.get(fallback, 0)),
        reverse=True,
    )[:75]


def _format_seconds(value: float) -> str:
    return f"{value:10.3f}"


def _is_io_row(row: dict[str, Any]) -> bool:
    filename = str(row["file"]).replace("\\", "/").lower()
    function = str(row["function"]).lower()
    if "importlib" in filename or filename.endswith("/threading.py"):
        return False
    if "/nibabel/" in filename or "/fastfuncstuff/io/" in filename:
        return True
    io_prefixes = ("load", "open", "read", "save", "write")
    tokens = re.findall(r"[a-z]+", function)
    return any(token == "tofile" or token.startswith(io_prefixes) for token in tokens)


def aggregate_stage(stage_dir: Path) -> dict[str, Any]:
    """Aggregate invocation JSON files and write a readable stage report."""
    invocation_paths = sorted((stage_dir / "invocations").glob("*.json"))
    if not invocation_paths and (stage_dir / "invocation.json").exists():
        invocation_paths = [stage_dir / "invocation.json"]
    invocations = [json.loads(path.read_text()) for path in invocation_paths]
    python_rows = _aggregate_rows(invocations, "python", ("file", "line", "function"))
    torch_rows = _aggregate_rows(invocations, "torch", ("operator",))
    io_rows = _aggregate_rows(invocations, "python_io", ("file", "line", "function"))
    if not io_rows:
        io_rows = [row for row in python_rows if _is_io_row(row)]
    tool_seconds = sum(
        item.get("tool_seconds", item.get("wall_seconds", 0.0)) for item in invocations
    )
    profiler_stop_seconds = sum(item.get("profiler_stop_seconds", 0.0) for item in invocations)
    report_build_seconds = sum(item.get("report_build_seconds", 0.0) for item in invocations)
    summary = {
        "schema_version": 2,
        "stage": stage_dir.name,
        "invocation_count": len(invocations),
        "tool_seconds": tool_seconds,
        "profiler_stop_seconds": profiler_stop_seconds,
        "report_build_seconds": report_build_seconds,
        "wall_seconds": sum(item.get("wall_seconds", 0.0) for item in invocations),
        "invocations": [
            {
                "file": path.name,
                "label": item.get("label"),
                "command": item.get("command"),
                "tool_seconds": item.get("tool_seconds", item.get("wall_seconds")),
                "profiler_stop_seconds": item.get("profiler_stop_seconds"),
                "report_build_seconds": item.get("report_build_seconds"),
                "wall_seconds": item.get("wall_seconds"),
                "unattributed_seconds": item.get("unattributed_seconds"),
                "torch_collection_seconds": item.get("torch_collection_seconds"),
                "torch_collection_limited": item.get("torch_collection_limited", False),
                "torch_collection_started": item.get("torch_collection_started", True),
                "error": item.get("error"),
            }
            for path, item in zip(invocation_paths, invocations, strict=True)
        ],
        "python": python_rows,
        "io_python": io_rows,
        "torch": torch_rows,
    }
    (stage_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        f"Stage: {stage_dir.name}",
        f"Invocations: {len(invocations)}",
        f"Summed tool time: {summary['tool_seconds']:.3f}s",
        f"Profiler stop time: {summary['profiler_stop_seconds']:.3f}s",
        f"Report build time: {summary['report_build_seconds']:.3f}s",
        f"Summed profiler capture wall time: {summary['wall_seconds']:.3f}s",
    ]
    if not invocations:
        lines.extend(["", "No FFS commands executed; cached outputs may have been reused."])
    else:
        lines.extend(["", "Per-invocation timing:"])
        for item in summary["invocations"]:
            sample_seconds = item["torch_collection_seconds"]
            sample = f"{sample_seconds:.3f}s" if sample_seconds is not None else "full"
            if not item.get("torch_collection_started", True):
                sample = "NONE (tool ended inside the warmup)"
            hidden = item.get("unattributed_seconds")
            # cProfile is main-thread only; say how much of the run it could not see.
            off_thread = f", off-main-thread={hidden:.3f}s" if hidden else ""
            lines.append(
                f"  {item['label']}: tool={item['tool_seconds']:.3f}s, "
                f"torch_sample={sample}, stop={item['profiler_stop_seconds'] or 0:.3f}s, "
                f"report={item['report_build_seconds'] or 0:.3f}s{off_thread}"
            )
        lines.extend(["", "Top Python functions (cumulative CPU seconds):"])
        for row in summary["python"][:25]:
            location = f"{Path(row['file']).name}:{row['line']}:{row['function']}"
            lines.append(
                f"{_format_seconds(row['cumulative_cpu_seconds'])}  "
                f"self={row['self_cpu_seconds']:.3f}  calls={row['calls']}  {location}"
            )
        lines.extend(["", "I/O-related Python functions (cumulative CPU seconds):"])
        for row in summary["io_python"][:15]:
            location = f"{Path(row['file']).name}:{row['line']}:{row['function']}"
            lines.append(
                f"{_format_seconds(row['cumulative_cpu_seconds'])}  "
                f"self={row['self_cpu_seconds']:.3f}  calls={row['calls']}  {location}"
            )
        lines.extend(
            [
                "",
                "PyTorch operators (self device / self CPU seconds).",
                "  device: the WHOLE invocation -- CUDA collection is never toggled,",
                "          because toggling it loses the device timings entirely.",
                "  cpu:    the sampled windows only. The two columns do not share a",
                "          denominator; divide device time by tool time, not by the sample.",
            ]
        )
        for row in summary["torch"][:25]:
            lines.append(
                f"device={row['self_device_seconds']:.3f}  cpu={row['self_cpu_seconds']:.3f}  "
                f"calls={row['calls']}  {row['operator']}"
            )
    (stage_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    return summary


@dataclass
class StageProfiler:
    """Profile collector active while one stage's ``run_ffs`` executes."""

    stage: str
    stage_dir: Path
    trace: bool = False
    count: int = 0

    def wrap_command(self, cmd: str | list[str], label: str) -> list[str] | None:
        argv = command_argv(cmd)
        if argv is None:
            return None
        self.count += 1
        output = self.stage_dir / "invocations" / f"{self.count:03d}-{_slug(label)}.json"
        launcher = [
            sys.executable,
            "-m",
            "fastfuncstuff.benchmark.profile_cli",
            "--output",
            str(output),
            "--stage",
            self.stage,
            "--label",
            label,
        ]
        if self.trace:
            launcher.append("--trace")
        return [*launcher, "--", *argv]

    def run_callable(self, function: Callable[[], _T], label: str, command: list[str]) -> _T:
        self.count += 1
        output = self.stage_dir / "invocations" / f"{self.count:03d}-{_slug(label)}.json"
        return capture_profile(
            function, output, command=command, label=label, stage=self.stage, trace=self.trace
        )

    def finish(self) -> dict[str, Any]:
        return aggregate_stage(self.stage_dir)


@dataclass
class BenchmarkProfiler:
    """Own a timestamped profile run and its per-stage collectors."""

    root: Path
    trace: bool = False
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def create(cls, data_dir: Path, trace: bool = False) -> BenchmarkProfiler:
        base = data_dir / "profiles"
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        root = base / stamp
        suffix = 2
        while root.exists():
            root = base / f"{stamp}-{suffix}"
            suffix += 1
        root.mkdir(parents=True)
        return cls(root=root, trace=trace)

    def start_stage(self, stage: str) -> StageProfiler:
        stage_dir = self.root / _slug(stage)
        (stage_dir / "invocations").mkdir(parents=True, exist_ok=True)
        return StageProfiler(stage=stage, stage_dir=stage_dir, trace=self.trace)

    def finish_stage(self, profiler: StageProfiler) -> None:
        self.stages[profiler.stage] = profiler.finish()

    def finish(self) -> dict[str, Any]:
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "trace_enabled": self.trace,
            "torch_sample_schedule": (
                None if self.trace else vars(_DEFAULT_SAMPLE_SCHEDULE).copy()
            ),
            "profiled_invocations": sum(s["invocation_count"] for s in self.stages.values()),
            "stages": {
                name: {
                    "directory": _slug(name),
                    "invocation_count": summary["invocation_count"],
                    "wall_seconds": summary["wall_seconds"],
                }
                for name, summary in self.stages.items()
            },
        }
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
