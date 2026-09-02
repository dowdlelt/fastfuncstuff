"""Compact CPU and accelerator profiles for FFS benchmark invocations."""

from __future__ import annotations

import cProfile
import json
import platform
import re
import shlex
import sys
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


def _python_rows(profiler: cProfile.Profile, limit: int = 75) -> list[dict[str, Any]]:
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
    return rows[:limit]


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


def capture_profile(
    function: Callable[[], Any],
    output_path: Path,
    *,
    command: list[str],
    label: str,
    stage: str,
    trace: bool = False,
) -> Any:
    """Run a callable under cProfile and torch.profiler, then write compact JSON."""
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

    started = time.time()
    monotonic_started = time.monotonic()
    error: str | None = None
    python_profiler.enable()
    torch_profiler.__enter__()
    try:
        result = function()
    except BaseException as exc:
        if not isinstance(exc, SystemExit) or exc.code not in (None, 0):
            error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        exc_info = sys.exc_info()
        torch_profiler.__exit__(*exc_info)
        python_profiler.disable()
        if cuda:
            torch.cuda.synchronize()
        elapsed = time.monotonic() - monotonic_started
        trace_path = output_path.with_suffix(".trace.json")
        if trace:
            torch_profiler.export_chrome_trace(str(trace_path))
        payload: dict[str, Any] = {
            "schema_version": 1,
            "stage": stage,
            "label": label,
            "command": command,
            "started_at": datetime.fromtimestamp(started).astimezone().isoformat(),
            "wall_seconds": elapsed,
            "error": error,
            "hardware": _hardware_metadata(requested_device),
            "python": _python_rows(python_profiler),
            "torch": _torch_rows(torch_profiler),
            "trace": trace_path.name if trace else None,
        }
        if cuda:
            payload["cuda_memory"] = {
                "allocated_before_bytes": cuda_allocated_before,
                "allocated_after_bytes": torch.cuda.memory_allocated(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
        output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return result


def _aggregate_rows(
    invocations: list[dict[str, Any]], section: str, key_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    totals: dict[tuple[Any, ...], dict[str, Any]] = {}
    numeric = (
        ("calls", "primitive_calls", "self_cpu_seconds", "cumulative_cpu_seconds")
        if section == "python"
        else ("calls", "self_cpu_seconds", "cpu_seconds", "self_device_seconds", "device_seconds")
    )
    for invocation in invocations:
        for row in invocation.get(section, []):
            key = tuple(row[metric] for metric in key_fields)
            merged = totals.setdefault(key, {metric: row[metric] for metric in key_fields})
            for metric in numeric:
                if metric in row:
                    merged[metric] = merged.get(metric, 0) + row[metric]
    sort_field = "cumulative_cpu_seconds" if section == "python" else "self_device_seconds"
    fallback = "self_cpu_seconds"
    return sorted(
        totals.values(),
        key=lambda row: max(row.get(sort_field, 0), row.get(fallback, 0)),
        reverse=True,
    )[:75]


def _format_seconds(value: float) -> str:
    return f"{value:10.3f}"


def _is_io_row(row: dict[str, Any]) -> bool:
    description = f"{row['file']} {row['function']}".lower()
    markers = ("nibabel", "fastfuncstuff/io", "load", "save", "read", "write", "open")
    return any(marker in description for marker in markers)


def aggregate_stage(stage_dir: Path) -> dict[str, Any]:
    """Aggregate invocation JSON files and write a readable stage report."""
    invocation_paths = sorted((stage_dir / "invocations").glob("*.json"))
    invocations = [json.loads(path.read_text()) for path in invocation_paths]
    python_rows = _aggregate_rows(invocations, "python", ("file", "line", "function"))
    torch_rows = _aggregate_rows(invocations, "torch", ("operator",))
    summary = {
        "schema_version": 1,
        "stage": stage_dir.name,
        "invocation_count": len(invocations),
        "wall_seconds": sum(item.get("wall_seconds", 0.0) for item in invocations),
        "invocations": [
            {
                "file": path.name,
                "label": item.get("label"),
                "command": item.get("command"),
                "wall_seconds": item.get("wall_seconds"),
                "error": item.get("error"),
            }
            for path, item in zip(invocation_paths, invocations, strict=True)
        ],
        "python": python_rows,
        "io_python": [row for row in python_rows if _is_io_row(row)],
        "torch": torch_rows,
    }
    (stage_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        f"Stage: {stage_dir.name}",
        f"Invocations: {len(invocations)}",
        f"Summed invocation wall time: {summary['wall_seconds']:.3f}s",
    ]
    if not invocations:
        lines.extend(["", "No FFS commands executed; cached outputs may have been reused."])
    else:
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
        lines.extend(["", "Top PyTorch operators (self device / self CPU seconds):"])
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
    def create(cls, processing_dir: Path, trace: bool = False) -> BenchmarkProfiler:
        base = processing_dir / "benchmark_profiles"
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
