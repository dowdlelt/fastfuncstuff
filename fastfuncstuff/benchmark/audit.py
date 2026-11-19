"""Low-overhead speed audits for FFS benchmark invocations.

``ffs_profile`` / ``ffs_benchmark -profile`` answer "where does the time go"
with the torch profiler, whose own overhead hides exactly the launch-bound
behaviour a GPU pipeline suffers from. These audits ask narrower questions
cheaply enough to leave the tool's speed alone:

``idle``  Is the GPU busy while the tool runs, and where is the host while it
          is not? NVML is sampled from a side thread (no CUDA calls
          in-process), and each sample charges its idle fraction to the main
          thread's stack at that moment. Ten samples a second and nothing on
          the tool's own path, so the tool runs at its normal speed.
``syncs`` Which lines of ffs force a host<->device synchronisation:
          ``torch.cuda.set_sync_debug_mode`` reports each implicit sync
          (``.item()``, ``nonzero``, boolean indexing, a device->host copy),
          attributed to the innermost ffs frame. Each report goes through
          Python's warnings machinery, ~100s of us, so this pass *slows* a
          sync-heavy tool (reml: ~10 s on 41k syncs) and reports locations and
          counts only -- never read its timings, and never judge a sync fix by
          this pass: A/B it without the audit. torch calls the mode a
          prototype that misses some syncs, so a low count is a floor.
``ops``   Which (op, shapes, dtype, device) the tool actually runs, counted by
          a ``TorchDispatchMode``. On the CPU each op is timed too, since CPU
          ops are synchronous; on CUDA only counts and output bytes are kept.
          The census is the input to ``scripts/replay_op_census.py``, which
          re-times the hot entries across threads or backends. Under the mode,
          ``torch.compile``'d regions run eagerly, so this pass is for *what*
          runs, never for how long the tool takes.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import threading
import time
import traceback
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .profiling import _requested_device, _slug, command_argv

MODES = ("idle", "syncs", "ops")
# Frames carry the path the module was imported by, which need not be the
# resolved one (the repo sits under a symlinked Dropbox), so keep both forms.
_PKG_DIRS = tuple(
    {
        os.path.dirname(os.path.dirname(p)) + os.sep
        for p in (os.path.abspath(__file__), os.path.realpath(__file__))
    }
)
_THIS_FILES = {os.path.abspath(__file__), os.path.realpath(__file__)}


# ---------------------------------------------------------------------------
# NVML, through ctypes: no extra dependency, and no CUDA context in-process.
# ---------------------------------------------------------------------------


class _Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


def _nvml_index(requested_device: str | None) -> int:
    """NVML index of the card the tool will use (``cuda,N`` / ``cuda:N`` under CUDA_VISIBLE_DEVICES)."""
    ordinal = 0
    if requested_device:
        for sep in (",", ":"):
            head, _, tail = requested_device.partition(sep)
            if head == "cuda" and tail.split(",")[0].isdigit():
                ordinal = int(tail.split(",")[0])
                break
    visible = [v for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    if visible and ordinal < len(visible) and visible[ordinal].strip().isdigit():
        return int(visible[ordinal])
    return ordinal


class BusySampler:
    """Poll NVML's GPU utilisation from a daemon thread.

    NVML reports the fraction of its last sample period (1/6 s to 1 s, by card)
    in which any kernel was resident. Coarse, and blind to *how much* of the
    GPU a kernel fills, but it is exactly the "is the card waiting on the host"
    signal, and it costs nothing in the process being measured.
    """

    def __init__(self, index: int, period: float = 0.1) -> None:
        self.period = period
        # (time, busy %, innermost ffs frame, leaf frame) of the main thread.
        self.samples: list[tuple[float, int, str, str]] = []
        self._main = threading.main_thread().ident
        self.error: str | None = None
        self._stop = threading.Event()
        self._lib: Any = None
        self._handle = ctypes.c_void_p()
        try:
            self._lib = ctypes.CDLL("libnvidia-ml.so.1")
            if self._lib.nvmlInit_v2() != 0:
                raise OSError("nvmlInit_v2 failed")
            if self._lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self._handle)) != 0:
                raise OSError(f"no NVML device {index}")
        except OSError as exc:
            self.error = str(exc)
            self._lib = None
        self._thread = threading.Thread(target=self._run, name="ffs-audit-nvml", daemon=True)

    def read(self) -> int | None:
        if self._lib is None:
            return None
        util = _Utilization()
        if self._lib.nvmlDeviceGetUtilizationRates(self._handle, ctypes.byref(util)) != 0:
            return None
        return int(util.gpu)

    def _run(self) -> None:
        while not self._stop.is_set():
            value = self.read()
            if value is not None:
                frame = sys._current_frames().get(self._main) if self._main else None
                site, leaf = _frame_sites(frame)
                self.samples.append((time.monotonic(), value, site, leaf))
            self._stop.wait(self.period)

    def start(self) -> None:
        if self._lib is not None:
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def summary(self, started: float, ended: float) -> dict[str, Any]:
        samples = [(t, u) for t, u, _, _ in self.samples if started <= t <= ended]
        if not samples:
            return {"available": False, "error": self.error}
        utils = [u for _, u in samples]
        busy = [i for i, u in enumerate(utils) if u > 0]
        out: dict[str, Any] = {
            "available": True,
            "samples": len(samples),
            "mean_busy_pct": sum(utils) / len(utils),
        }
        if busy:
            # Imports, argument parsing and file loading precede the first
            # kernel and follow the last; the window between them is where an
            # idle GPU is a pipeline problem rather than startup.
            window = utils[busy[0] : busy[-1] + 1]
            t0, t1 = samples[busy[0]][0], samples[busy[-1]][0]
            out.update(
                {
                    "first_kernel_s": t0 - started,
                    "active_window_s": t1 - t0,
                    "active_busy_pct": sum(window) / len(window),
                    "active_idle_s": (t1 - t0) * (1 - sum(window) / (100 * len(window))),
                }
            )
            # Where the host was while the card sat idle: each sample charges its
            # idle fraction of one period to the main thread's stack at that moment.
            by_site: Counter[str] = Counter()
            by_leaf: Counter[str] = Counter()
            for t, u, site, leaf in self.samples:
                if t0 <= t <= t1 and u < 100:
                    w = self.period * (100 - u) / 100
                    by_site[site] += w
                    by_leaf[leaf] += w
            out["idle_sites"] = [(k, round(v, 2)) for k, v in by_site.most_common(20)]
            out["idle_leaves"] = [(k, round(v, 2)) for k, v in by_leaf.most_common(20)]
        # A coarse timeline (1 s bins) so a reader can see *where* it idles.
        bins: dict[int, list[int]] = defaultdict(list)
        for t, u in samples:
            bins[int(t - started)].append(u)
        out["timeline_1s"] = [round(sum(v) / len(v)) for _, v in sorted(bins.items())]
        return out


# ---------------------------------------------------------------------------
# Sync attribution
# ---------------------------------------------------------------------------


def _call_site(stack: traceback.StackSummary) -> str:
    """Innermost frame inside fastfuncstuff (not this file); else the innermost non-library frame."""
    stack = [f for f in stack if f.filename not in _THIS_FILES]
    for frame in reversed(stack):
        for root in _PKG_DIRS:
            if frame.filename.startswith(root):
                rel = os.path.relpath(frame.filename, os.path.dirname(root.rstrip(os.sep)))
                return f"{rel}:{frame.lineno} {frame.name}"
    for frame in reversed(stack):
        if "site-packages" not in frame.filename and "/lib/python" not in frame.filename:
            return f"{frame.filename}:{frame.lineno} {frame.name}"
    return "<unattributed>"


def _frame_sites(frame: Any) -> tuple[str, str]:
    """(innermost fastfuncstuff frame, innermost frame of any kind) of a live stack.

    Walks frame objects directly: ``traceback.extract_stack`` reads source lines
    through linecache, and on a sync-heavy tool that showed up in the idle
    attribution as the audit's own cost.
    """
    if frame is None:
        return "<no frame>", "<no frame>"
    leaf = f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_code.co_name}"
    f = frame
    while f is not None:
        name = f.f_code.co_filename
        if name not in _THIS_FILES:
            for root in _PKG_DIRS:
                if name.startswith(root):
                    rel = os.path.relpath(name, os.path.dirname(root.rstrip(os.sep)))
                    return f"{rel}:{f.f_lineno} {f.f_code.co_name}", leaf
        f = f.f_back
    # A script driving ffs from outside the package: name its own frame.
    f = frame
    while f is not None:
        name = f.f_code.co_filename
        if name not in _THIS_FILES and "site-packages" not in name and "/lib/python" not in name:
            return f"{name}:{f.f_lineno} {f.f_code.co_name}", leaf
        f = f.f_back
    return "<outside fastfuncstuff>", leaf


class SyncRecorder:
    """Count implicit CUDA syncs and explicit ``torch.cuda.synchronize`` calls by call site."""

    def __init__(self) -> None:
        self.implicit: Counter[str] = Counter()
        self.explicit: Counter[str] = Counter()
        self._restore: list[Callable[[], None]] = []

    def install(self) -> None:
        import torch

        previous_show = warnings.showwarning

        def show(message, category, filename, lineno, file=None, line=None):
            if "synchronizing CUDA operation" in str(message):
                self.implicit[_frame_sites(sys._getframe(1))[0]] += 1
                return
            previous_show(message, category, filename, lineno, file, line)

        warnings.showwarning = show
        # "always": the default filter shows a warning once per location, which
        # would count each sync site once however hot its loop is.
        warnings.filterwarnings("always", message=".*synchronizing CUDA operation.*")
        with warnings.catch_warnings():
            # "prototype feature, does not detect all syncs" -- true, and said
            # once in the module docstring rather than on every invocation.
            warnings.simplefilter("ignore", UserWarning)
            torch.cuda.set_sync_debug_mode("warn")

        original_sync = torch.cuda.synchronize

        def synchronize(device=None):
            self.explicit[_frame_sites(sys._getframe(1))[0]] += 1
            return original_sync(device)

        torch.cuda.synchronize = synchronize

        def restore() -> None:
            torch.cuda.set_sync_debug_mode("default")
            torch.cuda.synchronize = original_sync
            warnings.showwarning = previous_show

        self._restore.append(restore)

    def uninstall(self) -> None:
        while self._restore:
            self._restore.pop()()

    def summary(self, tool_seconds: float, top: int = 25) -> dict[str, Any]:
        n = sum(self.implicit.values())
        return {
            "implicit_syncs": n,
            "implicit_per_s": n / tool_seconds if tool_seconds > 0 else None,
            "explicit_synchronize": sum(self.explicit.values()),
            "implicit_sites": self.implicit.most_common(top),
            "explicit_sites": self.explicit.most_common(top),
        }


# ---------------------------------------------------------------------------
# Op census
# ---------------------------------------------------------------------------


def _describe(args: Any) -> tuple[str, str, str]:
    """(device, dtype, shapes) of the tensor arguments, nested lists included."""
    import torch

    shapes: list[str] = []
    device = dtype = ""

    def visit(a: Any) -> None:
        nonlocal device, dtype
        if isinstance(a, torch.Tensor):
            # "!" marks a strided view: the same op on a transposed operand can
            # cost several times the contiguous one, so they are separate rows.
            mark = "" if a.is_contiguous() else "!"
            shapes.append(("x".join(map(str, a.shape)) or "scalar") + mark)
            if not device:
                device, dtype = a.device.type, str(a.dtype).removeprefix("torch.")
        elif isinstance(a, (list, tuple)) and len(shapes) < 6:
            for b in a[:6]:
                visit(b)

    for a in args[:6]:
        visit(a)
    return device, dtype, ",".join(shapes[:6])


def _template(value: Any) -> Any:
    """A JSON stand-in for an op argument that ``replay_op_census.py`` can rebuild.

    Tensors become their geometry (values are not kept; replays use random
    data), everything else its literal or its repr.
    """
    import torch

    if isinstance(value, torch.Tensor):
        return {
            "tensor": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype).removeprefix("torch."),
            "device": value.device.type,
        }
    if isinstance(value, (list, tuple)):
        return [_template(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.dtype):
        return {"dtype_literal": str(value).removeprefix("torch.")}
    if isinstance(value, torch.device):
        return {"device_literal": value.type}
    if isinstance(value, (torch.memory_format, torch.layout)):
        return {"torch_attr": repr(value).removeprefix("torch.")}
    return {"repr": repr(value)}


def _out_bytes(out: Any) -> int:
    import torch

    if isinstance(out, torch.Tensor):
        return out.numel() * out.element_size()
    if isinstance(out, (list, tuple)):
        return sum(_out_bytes(o) for o in out)
    return 0


# Ops that only rearrange metadata; counting them buries the real work.
_VIEW_OPS = {
    "aten.view.default",
    "aten._unsafe_view.default",
    "aten.t.default",
    "aten.transpose.int",
    "aten.permute.default",
    "aten.expand.default",
    "aten.unsqueeze.default",
    "aten.squeeze.dim",
    "aten.squeeze.dims",
    "aten.select.int",
    "aten.slice.Tensor",
    "aten.alias.default",
    "aten.detach.default",
    "aten.as_strided.default",
    "aten.lift_fresh.default",
}


def make_census() -> Any:
    from torch.utils._python_dispatch import TorchDispatchMode

    class OpCensus(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.rows: dict[tuple[str, str, str, str], list[float]] = {}
            self.templates: dict[tuple[str, str, str, str], Any] = {}
            self._lock = threading.Lock()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ARG002
            name = str(func)
            if name in _VIEW_OPS:
                return func(*args, **(kwargs or {}))
            device, dtype, shapes = _describe(args)
            t0 = time.perf_counter()
            out = func(*args, **(kwargs or {}))
            # CUDA ops are asynchronous, so their wall time here is launch cost.
            dt = time.perf_counter() - t0 if device == "cpu" else 0.0
            key = (name, device, dtype, shapes)
            with self._lock:
                row = self.rows.get(key)
                if row is None:
                    row = self.rows[key] = [0, 0.0, 0]
                    self.templates[key] = {
                        "args": _template(list(args)),
                        "kwargs": {k: _template(v) for k, v in (kwargs or {}).items()},
                    }
                row[0] += 1
                row[1] += dt
                row[2] += _out_bytes(out)
            return out

        def table(self) -> list[dict[str, Any]]:
            return [
                {
                    "op": k[0],
                    "device": k[1],
                    "dtype": k[2],
                    "shapes": k[3],
                    "calls": int(v[0]),
                    "cpu_seconds": v[1],
                    "out_bytes": int(v[2]),
                    "call": self.templates[k],
                }
                for k, v in self.rows.items()
            ]

    return OpCensus()


# ---------------------------------------------------------------------------
# One invocation
# ---------------------------------------------------------------------------


class AuditSession:
    """One audit, started and stopped explicitly (``_audit_hook`` spans a whole process)."""

    def __init__(self, mode: str, *, command: list[str], label: str, stage: str) -> None:
        if mode not in MODES:
            raise ValueError(f"audit mode must be one of {MODES}, got {mode!r}")
        self.mode, self.command, self.label, self.stage = mode, command, label, stage
        self.requested = _requested_device(command)
        self.sampler: BusySampler | None = None
        self.syncs: SyncRecorder | None = None
        self.census: Any = None
        self.baseline: int | None = None
        self.started_at = ""
        self.t0 = 0.0

    def start(self) -> None:
        self.started_at = datetime.now().astimezone().isoformat()
        wants_cuda = self.requested is None or self.requested.startswith("cuda")
        if self.mode == "idle" and wants_cuda:
            self.sampler = BusySampler(_nvml_index(self.requested))
            self.baseline = self.sampler.read()
            self.sampler.start()
        elif self.mode == "syncs" and wants_cuda:
            self.syncs = SyncRecorder()
            self.syncs.install()
        elif self.mode == "ops":
            self.census = make_census()
            self.census.__enter__()
        self.t0 = time.monotonic()

    def stop(self, output_path: Path, error: str | None) -> None:
        t1 = time.monotonic()
        tool_seconds = t1 - self.t0
        payload: dict[str, Any] = {
            "schema_version": 1,
            "mode": self.mode,
            "stage": self.stage,
            "label": self.label,
            "command": self.command,
            "requested_device": self.requested,
            "started_at": self.started_at,
            "tool_seconds": tool_seconds,
            "error": error,
        }
        if self.census is not None:
            self.census.__exit__(None, None, None)
            import torch

            # The tool set it during its run; replay compares against this width.
            payload["torch_threads"] = torch.get_num_threads()
            payload["ops"] = self.census.table()
        if self.syncs is not None:
            self.syncs.uninstall()
            payload["syncs"] = self.syncs.summary(tool_seconds)
        if self.sampler is not None:
            self.sampler.stop()
            payload["gpu"] = self.sampler.summary(self.t0, t1)
            # Another process on the card inflates busy% and hides idling.
            payload["gpu"]["baseline_busy_pct"] = self.baseline
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=1) + "\n")


def capture_audit[T](
    function: Callable[[], T],
    output_path: Path,
    *,
    mode: str,
    command: list[str],
    label: str,
    stage: str,
) -> T:
    """Run ``function`` under one audit mode and write its JSON report."""
    session = AuditSession(mode, command=command, label=label, stage=stage)
    session.start()
    error: str | None = None
    try:
        return function()
    except BaseException as exc:
        if not isinstance(exc, SystemExit) or exc.code not in (None, 0):
            error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        session.stop(output_path, error)


# ---------------------------------------------------------------------------
# Benchmark integration (same interface as profiling.StageProfiler)
# ---------------------------------------------------------------------------


@dataclass
class StageAuditor:
    stage: str
    stage_dir: Path
    mode: str
    count: int = 0

    def _next_output(self, label: str) -> Path:
        self.count += 1
        return self.stage_dir / "invocations" / f"{self.count:03d}-{_slug(label)}.json"

    def wrap_command(self, cmd: str | list[str], label: str) -> list[str] | None:
        argv = command_argv(cmd)
        if argv is None:
            return None
        output = self._next_output(label)
        launcher = [sys.executable, "-m", "fastfuncstuff.benchmark.audit_cli", "--mode", self.mode]
        launcher += ["--output", str(output), "--stage", self.stage, "--label", label]
        return [*launcher, "--", *argv]

    def run_callable[T](self, function: Callable[[], T], label: str, command: list[str]) -> T:
        return capture_audit(
            function,
            self._next_output(label),
            mode=self.mode,
            command=command,
            label=label,
            stage=self.stage,
        )

    def finish(self) -> dict[str, Any]:
        return aggregate_stage(self.stage_dir, self.stage)


def _load(stage_dir: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted((stage_dir / "invocations").glob("*.json")):
        try:
            out.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def aggregate_stage(stage_dir: Path, stage: str) -> dict[str, Any]:
    runs = _load(stage_dir)
    summary: dict[str, Any] = {
        "stage": stage,
        "invocation_count": len(runs),
        "tool_seconds": sum(r["tool_seconds"] for r in runs),
    }
    implicit: Counter[str] = Counter()
    explicit: Counter[str] = Counter()
    window = busy_weighted = idle = 0.0
    idle_sites: Counter[str] = Counter()
    idle_leaves: Counter[str] = Counter()
    for r in runs:
        g = r.get("gpu") or {}
        for k, v in g.get("idle_sites", []):
            idle_sites[k] += v
        for k, v in g.get("idle_leaves", []):
            idle_leaves[k] += v
    for r in runs:
        s = r.get("syncs")
        if s:
            implicit.update(dict(s["implicit_sites"]))
            explicit.update(dict(s["explicit_sites"]))
        g = r.get("gpu") or {}
        if g.get("active_window_s"):
            window += g["active_window_s"]
            busy_weighted += g["active_window_s"] * g["active_busy_pct"]
            idle += g["active_idle_s"]
    if any(r.get("syncs") for r in runs):
        summary["implicit_syncs"] = sum(implicit.values())
        summary["explicit_synchronize"] = sum(explicit.values())
        summary["implicit_sites"] = implicit.most_common(15)
        summary["explicit_sites"] = explicit.most_common(10)
    if idle_sites:
        summary["idle_sites"] = [(k, round(v, 2)) for k, v in idle_sites.most_common(15)]
        summary["idle_leaves"] = [(k, round(v, 2)) for k, v in idle_leaves.most_common(15)]
    if window:
        summary["active_window_s"] = window
        summary["active_busy_pct"] = busy_weighted / window
        summary["active_idle_s"] = idle
    ops: dict[tuple, list[float]] = {}
    calls: dict[tuple, Any] = {}
    for r in runs:
        for row in r.get("ops", []):
            key = (row["op"], row["device"], row["dtype"], row["shapes"])
            calls.setdefault(key, row.get("call"))
            acc = ops.setdefault(key, [0, 0.0, 0])
            acc[0] += row["calls"]
            acc[1] += row["cpu_seconds"]
            acc[2] += row["out_bytes"]
    if ops:
        summary["ops"] = [
            {
                "op": k[0],
                "device": k[1],
                "dtype": k[2],
                "shapes": k[3],
                "calls": int(v[0]),
                "cpu_seconds": v[1],
                "out_bytes": int(v[2]),
                "call": calls[k],
            }
            for k, v in ops.items()
        ]
    (stage_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def format_report(stages: dict[str, dict[str, Any]], mode: str) -> str:
    """The human-readable run summary written next to the manifest."""
    lines: list[str] = []
    if mode in ("idle", "syncs"):
        lines.append(
            f"{'stage':28s} {'tool s':>8s} {'window s':>9s} {'busy%':>6s} {'idle s':>7s} "
            f"{'syncs':>7s} {'sync()':>7s}"
        )
        for name, s in stages.items():
            if not s["invocation_count"]:
                continue
            busy = f"{s['active_busy_pct']:.0f}" if "active_busy_pct" in s else "-"
            lines.append(
                f"{name:28s} {s['tool_seconds']:8.1f} {s.get('active_window_s', 0):9.1f} "
                f"{busy:>6s} {s.get('active_idle_s', 0):7.1f} {s.get('implicit_syncs', 0):7d} "
                f"{s.get('explicit_synchronize', 0):7d}"
            )
        lines.append("")
        lines.append("busy% = NVML utilisation between the first and last kernel; idle s is the")
        lines.append("time inside that window with no kernel resident. syncs = implicit host syncs")
        lines.append("(set_sync_debug_mode), sync() = explicit torch.cuda.synchronize calls.")
        if mode == "syncs":
            lines.append(
                "syncs pass: the hook slows sync-heavy tools, so 'tool s' is not a timing."
            )
        for name, s in stages.items():
            idle_sites = s.get("idle_sites") or []
            sites = s.get("implicit_sites") or []
            if not (sites or idle_sites):
                continue
            if idle_sites:
                lines.append(f"\n[{name}] GPU-idle seconds by host location (ffs frame | leaf)")
                lines.extend(f"  {v:7.1f}s  {k}" for k, v in idle_sites[:8])
                leaves = s.get("idle_leaves") or []
                lines.append("  leaves: " + ", ".join(f"{k} {v:.1f}s" for k, v in leaves[:6]))
            if sites:
                lines.append(f"[{name}] top implicit sync sites")
                lines.extend(f"  {n:7d}  {site}" for site, n in sites[:8])
    else:
        rows = [dict(r, stage=name) for name, s in stages.items() for r in s.get("ops", [])]
        cpu = sorted((r for r in rows if r["device"] == "cpu"), key=lambda r: -r["cpu_seconds"])
        lines.append("Top CPU ops by time (op | dtype | shapes | calls | seconds | stage)")
        lines.extend(
            f"  {r['cpu_seconds']:8.2f}s {r['calls']:8d}  {r['op']:34s} {r['dtype']:8s} "
            f"{r['shapes'][:60]:60s} {r['stage']}"
            for r in cpu[:40]
        )
        gpu = sorted((r for r in rows if r["device"] == "cuda"), key=lambda r: -r["calls"])
        lines.append("\nTop CUDA ops by calls (launch count is the cost when kernels are small)")
        lines.extend(
            f"  {r['calls']:8d}  {r['op']:34s} {r['dtype']:8s} {r['shapes'][:60]:60s} {r['stage']}"
            for r in gpu[:40]
        )
        f64 = sorted(
            (r for r in rows if r["device"] == "cuda" and r["dtype"] in ("float64", "complex128")),
            key=lambda r: -r["out_bytes"],
        )
        lines.append(
            "\nfloat64 on CUDA (consumer cards run f64 at 1/32-1/64 rate), by output bytes"
        )
        lines.extend(
            f"  {r['out_bytes'] / 2**20:9.1f} MiB {r['calls']:7d}  {r['op']:34s} "
            f"{r['shapes'][:60]:60s} {r['stage']}"
            for r in f64[:40]
        )
    return "\n".join(lines) + "\n"


@dataclass
class BenchmarkAuditor:
    root: Path
    mode: str
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def create(cls, data_dir: Path, mode: str) -> BenchmarkAuditor:
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        root = data_dir / "audits" / f"{stamp}-{mode}"
        suffix = 2
        while root.exists():
            root = data_dir / "audits" / f"{stamp}-{mode}-{suffix}"
            suffix += 1
        root.mkdir(parents=True)
        return cls(root=root, mode=mode)

    def start_stage(self, stage: str) -> StageAuditor:
        stage_dir = self.root / _slug(stage)
        (stage_dir / "invocations").mkdir(parents=True, exist_ok=True)
        return StageAuditor(stage=stage, stage_dir=stage_dir, mode=self.mode)

    def finish_stage(self, auditor: StageAuditor) -> None:
        self.stages[auditor.stage] = auditor.finish()

    def finish(self) -> dict[str, Any]:
        count = sum(s["invocation_count"] for s in self.stages.values())
        manifest = {
            "schema_version": 1,
            "mode": self.mode,
            "created_at": datetime.now().astimezone().isoformat(),
            "profiled_invocations": count,
            "stages": {n: {"directory": _slug(n)} for n in self.stages},
        }
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        (self.root / "summary.txt").write_text(format_report(self.stages, self.mode))
        return manifest


def main(argv: list[str] | None = None) -> None:
    """``python -m fastfuncstuff.benchmark.audit DIR``: summarise an ``FFS_AUDIT`` tree.

    DIR is ``$FFS_AUDIT_DIR/<mode>`` (one folder per tool) or an
    ``ffs_benchmark -audit`` run (one folder per stage).
    """
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("root", type=Path)
    args = ap.parse_args(argv)
    stages: dict[str, dict[str, Any]] = {}
    mode = ""
    for stage_dir in sorted(p for p in args.root.iterdir() if (p / "invocations").is_dir()):
        summary = aggregate_stage(stage_dir, stage_dir.name)
        if summary["invocation_count"]:
            stages[stage_dir.name] = summary
            if not mode:
                mode = (
                    "ops"
                    if "ops" in summary
                    else "syncs"
                    if "implicit_syncs" in summary
                    else "idle"
                )
    if not stages:
        sys.exit(f"no audit invocations under {args.root}")
    print(format_report(stages, mode))


if __name__ == "__main__":
    main()
