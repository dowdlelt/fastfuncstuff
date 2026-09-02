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
_COMPACT_TORCH_WINDOW_SECONDS = 2.0
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
    rows.sort(
        key=lambda row: max(row["self_device_seconds"], row["self_cpu_seconds"]),
        reverse=True,
    )
    return rows[:limit]


def _arm_collection_limit(
    torch_profiler: Any,
    activities: list[Any],
    seconds: float | None,
) -> tuple[dict[str, Any], Callable[[], None]]:
    """Stop Kineto event collection after a short window on the main thread."""
    state: dict[str, Any] = {
        "limit_seconds": seconds,
        "limited": False,
        "collection_seconds": None,
        "error": None,
    }
    no_cleanup = lambda: None
    required = ("SIGALRM", "ITIMER_REAL", "getitimer", "setitimer")
    if (
        seconds is None
        or seconds <= 0
        or threading.current_thread() is not threading.main_thread()
        or not all(hasattr(signal, name) for name in required)
    ):
        return state, no_cleanup

    timer_kind = signal.ITIMER_REAL
    previous_timer = signal.getitimer(timer_kind)
    if previous_timer[0] > 0:
        state["error"] = "SIGALRM timer already active; collected the full invocation"
        return state, no_cleanup

    sigalrm = signal.SIGALRM
    previous_handler = signal.getsignal(sigalrm)
    collection_started = time.monotonic()

    def stop_collection(_signum: int, _frame: Any) -> None:
        try:
            torch_profiler.toggle_collection_dynamic(False, activities)
        except Exception as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
        else:
            state["limited"] = True
            state["collection_seconds"] = time.monotonic() - collection_started

    signal.signal(sigalrm, stop_collection)
    signal.setitimer(timer_kind, seconds)

    def cleanup() -> None:
        signal.setitimer(timer_kind, 0)
        signal.signal(sigalrm, previous_handler)

    return state, cleanup


def capture_profile(
    function: Callable[[], Any],
    output_path: Path,
    *,
    command: list[str],
    label: str,
    stage: str,
    trace: bool = False,
    torch_profile_seconds: float | None = _COMPACT_TORCH_WINDOW_SECONDS,
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
    profile_limit = None if trace else torch_profile_seconds
    torch_profiler.__enter__()
    limit_state, cleanup_limit = _arm_collection_limit(torch_profiler, activities, profile_limit)
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
            collection_seconds = tool_seconds
        payload: dict[str, Any] = {
            "schema_version": 2,
            "stage": stage,
            "label": label,
            "command": command,
            "started_at": datetime.fromtimestamp(started).astimezone().isoformat(),
            "tool_seconds": tool_seconds,
            "profiler_stop_seconds": profiler_stop_seconds,
            "torch_collection_limit_seconds": profile_limit,
            "torch_collection_seconds": collection_seconds,
            "torch_collection_limited": limit_state["limited"],
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
    sort_field = "cumulative_cpu_seconds" if section in {"python", "python_io"} else "self_device_seconds"
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
                "torch_collection_seconds": item.get("torch_collection_seconds"),
                "torch_collection_limited": item.get("torch_collection_limited", False),
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
            lines.append(
                f"  {item['label']}: tool={item['tool_seconds']:.3f}s, "
                f"torch_sample={sample}, stop={item['profiler_stop_seconds'] or 0:.3f}s, "
                f"report={item['report_build_seconds'] or 0:.3f}s"
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
        lines.extend(["", "Top sampled PyTorch operators (self device / self CPU seconds):"])
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
            "torch_collection_limit_seconds": (
                None if self.trace else _COMPACT_TORCH_WINDOW_SECONDS
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
