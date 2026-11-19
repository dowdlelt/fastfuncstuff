#!/usr/bin/env python3
"""Re-time the ops ffs actually runs, at the shapes it actually runs them.

``bench_cpu_host.py`` measures a hand-picked list of kernels. This script takes
the list from a real workload instead: an op census written by
``ffs_benchmark -audit ops`` (or any ``capture_audit(..., mode="ops")``) records
every (op, shapes, dtype, device) a tool dispatched, with one argument template
per entry. Each hot entry is rebuilt with random data of the same geometry --
strides included -- and timed in isolation:

``--device cpu``   at 1, 2, 4, ... threads up to the cgroup quota. An op that is
                   a real share of a stage's time but gains little from threads
                   is the pattern ``policy_linalg``'s batch split fixed for
                   batched LAPACK; the table ranks entries by recorded seconds
                   and prints the scaling beside them.
``--device cuda``  per call with CUDA events, against this card's measured copy
                   bandwidth and empty-kernel launch time. It flags launch-bound
                   entries (per-call time within a few launches of the floor,
                   where the fix is fewer, larger calls), float64 entries with
                   their float32 time beside them (consumer cards run f64 at
                   1/32-1/64), and for dense linear algebra the cuSOLVER vs MAGMA
                   time -- torch's backend heuristic is a guess per op, not per
                   shape.

Typical use::

    ffs_benchmark -audit ops -force-ffs -device cpu     # census, CPU run
    python scripts/replay_op_census.py <data>/audits/<stamp>-ops --device cpu
    ffs_benchmark -audit ops -force-ffs                 # census, CUDA run
    python scripts/replay_op_census.py <data>/audits/<stamp>-ops --device cuda

Caveats. Values are random, so data-dependent ops (boolean masks, ``nonzero``,
iterative solvers) time a typical case, not the recorded one; integer tensors
are filled with 0/1 so they stay valid as indices. In-place ops are replayed
on the same buffers repeatedly, with denormals flushed so that a drifting value
cannot fall onto the CPU's slow path. Replay times an op alone, so effects
between ops (cache reuse, overlap on the GPU) are not in these numbers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

_HERE = Path(__file__).resolve().parent
_LINALG_HINTS = ("linalg", "cholesky", "triangular", "_lu", "lu_", "qr", "svd", "eigh", "geqrf")


def _bench_cpu_host():
    spec = importlib.util.spec_from_file_location("bench_cpu_host", _HERE / "bench_cpu_host.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Loading the census
# ---------------------------------------------------------------------------


def load_census(root: Path, device: str) -> tuple[list[dict], int | None]:
    """Merge every invocation's rows for ``device`` across stages; also the tools' thread count."""
    merged: dict[tuple, dict] = {}
    threads: set[int] = set()
    for path in sorted(root.glob("*/invocations/*.json")):
        try:
            inv = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if inv.get("torch_threads"):
            threads.add(int(inv["torch_threads"]))
        for row in inv.get("ops", []):
            if row["device"] != device or not row.get("call"):
                continue
            key = (row["op"], row["dtype"], row["shapes"])
            m = merged.setdefault(
                key, {**row, "calls": 0, "cpu_seconds": 0.0, "out_bytes": 0, "stages": set()}
            )
            m["calls"] += row["calls"]
            m["cpu_seconds"] += row["cpu_seconds"]
            m["out_bytes"] += row["out_bytes"]
            m["stages"].add(inv.get("stage", "?"))
    rows = list(merged.values())
    for r in rows:
        r["stages"] = sorted(r["stages"])
    return rows, (max(threads) if threads else None)


# ---------------------------------------------------------------------------
# Rebuilding a call
# ---------------------------------------------------------------------------


def _resolve_op(name: str):
    ns, op, overload = name.split(".")
    return getattr(getattr(getattr(torch.ops, ns), op), overload)


def _make_tensor(t: dict, device: torch.device, spd: bool, f64_to_f32: bool) -> torch.Tensor:
    dtype = getattr(torch, t["dtype"])
    if f64_to_f32 and dtype == torch.float64:
        dtype = torch.float32
    if f64_to_f32 and dtype == torch.complex128:
        dtype = torch.complex64
    shape, stride = t["tensor"], t["stride"]
    if spd and len(shape) >= 2 and shape[-1] == shape[-2] and dtype.is_floating_point:
        n = shape[-1]
        b = torch.randn(*shape, dtype=dtype, device=device)
        return b @ b.transpose(-1, -2) / max(n, 1) + torch.eye(n, dtype=dtype, device=device)
    # Expanded and overlapping views (stride 0) cannot be filled through the view,
    # so fill a flat base of the storage the strides address and view it.
    numel = 1 + sum((s - 1) * st for s, st in zip(shape, stride, strict=True) if s > 0)
    if any(s == 0 for s in shape):
        numel = 0
    if dtype == torch.bool:
        base = torch.rand(max(numel, 1), device=device) > 0.5
    elif dtype.is_floating_point or dtype.is_complex:
        base = torch.randn(max(numel, 1), dtype=dtype, device=device)
    elif spd:
        # Integer tensors around LAPACK calls are status codes ("info"); a
        # nonzero one makes _linalg_check_errors raise.
        base = torch.zeros(max(numel, 1), dtype=dtype, device=device)
    else:
        base = torch.randint(0, 2, (max(numel, 1),), dtype=dtype, device=device)
    return base.as_strided(shape, stride)


def build_call(row: dict, device: torch.device, f64_to_f32: bool = False):
    """``(fn, input_bytes)`` replaying ``row`` with fresh random data, or raise."""
    op = _resolve_op(row["op"])
    spd = any(h in row["op"] for h in _LINALG_HINTS)
    nbytes = 0

    def rebuild(v: Any) -> Any:
        nonlocal nbytes
        if isinstance(v, list):
            return [rebuild(x) for x in v]
        if isinstance(v, dict):
            if "tensor" in v:
                t = _make_tensor(v, device, spd, f64_to_f32)
                nbytes += t.numel() * t.element_size()
                return t
            if "dtype_literal" in v:
                dt = getattr(torch, v["dtype_literal"])
                return torch.float32 if f64_to_f32 and dt == torch.float64 else dt
            if "device_literal" in v:
                return device
            if "torch_attr" in v:
                return getattr(torch, v["torch_attr"])
            raise ValueError(f"cannot rebuild argument {v}")
        return v

    args = rebuild(row["call"]["args"])
    kwargs = {k: rebuild(v) for k, v in row["call"]["kwargs"].items()}
    if "device" in kwargs:
        kwargs["device"] = device
    return (lambda: op(*args, **kwargs)), nbytes


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def time_cuda(fn, min_time: float = 0.2, max_reps: int = 2000) -> float:
    """Mean seconds per call from CUDA events, after a warm-up."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    reps, total = 0, 0.0
    batch = 1
    while total < min_time and reps < max_reps:
        start.record()
        for _ in range(batch):
            fn()
        end.record()
        end.synchronize()
        total += start.elapsed_time(end) / 1e3
        reps += batch
        batch = min(batch * 2, max_reps - reps) or 1
    return total / reps


def cuda_floor() -> dict[str, float]:
    """This card's empty-launch time and device-to-device copy bandwidth."""
    x = torch.zeros(1, device="cuda")
    launch = time_cuda(lambda: x.add_(0), min_time=0.1)
    big = torch.empty(256 * 2**20 // 4, device="cuda")
    dst = torch.empty_like(big)
    copy = time_cuda(lambda: dst.copy_(big), min_time=0.2)
    return {"launch_s": launch, "copy_gbs": 2 * big.numel() * 4 / copy / 1e9}


def replay_cpu(rows: list[dict], threads: list[int], min_time: float, top: int) -> list[dict]:
    bench = _bench_cpu_host()
    rows = sorted(rows, key=lambda r: -r["cpu_seconds"])[:top]
    out = []
    for i, r in enumerate(rows, 1):
        rec: dict[str, Any] = {k: r[k] for k in ("op", "dtype", "shapes", "calls", "cpu_seconds")}
        rec["stages"] = r["stages"]
        try:
            fn, _ = build_call(r, torch.device("cpu"))
            per = {}
            for t in threads:
                torch.set_num_threads(t)
                per[t], _thr = bench.time_call(fn, min_time)
            rec["seconds_per_call"] = per
        except Exception as exc:  # noqa: BLE001 -- an unreplayable op is a row, not a crash
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        out.append(rec)
        print(f"  [{i}/{len(rows)}] {r['op']} {r['shapes'][:50]}", file=sys.stderr)
    return out


def replay_cuda(rows: list[dict], top: int) -> tuple[list[dict], dict]:
    floor = cuda_floor()
    # Counts are all a CUDA census has, so take both the most-called entries
    # (launch overhead) and the biggest movers (bandwidth), then rank by the
    # replayed time.
    by_calls = sorted(rows, key=lambda r: -r["calls"])[:top]
    by_bytes = sorted(rows, key=lambda r: -r["out_bytes"])[:top]
    f64 = [r for r in rows if r["dtype"] in ("float64", "complex128")]
    f64 = sorted(f64, key=lambda r: -r["out_bytes"])[:top]
    chosen = {id(r): r for r in by_calls + by_bytes + f64}.values()
    out = []
    lib0 = torch.backends.cuda.preferred_linalg_library()
    for i, r in enumerate(chosen, 1):
        rec: dict[str, Any] = {k: r[k] for k in ("op", "dtype", "shapes", "calls", "out_bytes")}
        rec["stages"] = r["stages"]
        try:
            fn, in_bytes = build_call(r, torch.device("cuda"))
            t = time_cuda(fn)
            rec["seconds_per_call"] = t
            rec["total_s"] = t * r["calls"]
            moved = in_bytes + r["out_bytes"] / max(r["calls"], 1)
            rec["gbs"] = moved / t / 1e9
            rec["launches_per_call"] = t / floor["launch_s"]
            if r["dtype"] in ("float64", "complex128"):
                fn32, _ = build_call(r, torch.device("cuda"), f64_to_f32=True)
                rec["seconds_per_call_f32"] = time_cuda(fn32)
            if any(h in r["op"] for h in _LINALG_HINTS):
                for lib in ("cusolver", "magma"):
                    torch.backends.cuda.preferred_linalg_library(lib)
                    try:
                        rec[f"seconds_{lib}"] = time_cuda(fn)
                    except Exception as exc:  # noqa: BLE001
                        rec[f"seconds_{lib}"] = f"{type(exc).__name__}"
                torch.backends.cuda.preferred_linalg_library(lib0)
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        out.append(rec)
        print(f"  [{i}] {r['op']} {r['shapes'][:50]}", file=sys.stderr)
        torch.cuda.empty_cache()
    return out, floor


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _fmt_s(x: float) -> str:
    return f"{x * 1e6:7.1f}us" if x < 1e-3 else f"{x * 1e3:7.2f}ms"


def report_cpu(recs: list[dict], tool_threads: int | None) -> str:
    wide = max((max(r.get("seconds_per_call", {1: 0})) for r in recs), default=1)
    lines = [
        f"CPU replay; census ran at {tool_threads or '?'} torch threads. Recorded = seconds the",
        "tools spent in this entry; speedup = 1 thread / widest. 'POOR' = under half of linear.",
        f"{'recorded':>9s} {'calls':>8s} {'1 thr':>9s} {f'{wide} thr':>9s} {'speedup':>7s}  op | dtype | shapes",
    ]
    for r in recs:
        if "error" in r:
            lines.append(
                f"{r['cpu_seconds']:8.2f}s {r['calls']:8d}  (unreplayable: {r['error']})  {r['op']}"
            )
            continue
        per = {int(k): v for k, v in r["seconds_per_call"].items()}
        one, w = per[min(per)], per[max(per)]
        sp = one / w if w > 0 else math.nan
        flag = "POOR" if max(per) > 1 and sp < 0.5 * max(per) else ""
        lines.append(
            f"{r['cpu_seconds']:8.2f}s {r['calls']:8d} {_fmt_s(one)} {_fmt_s(w)} {sp:6.1f}x {flag:4s} "
            f"{r['op']} | {r['dtype']} | {r['shapes'][:70]}  [{','.join(r['stages'][:3])}]"
        )
    return "\n".join(lines)


def report_cuda(recs: list[dict], floor: dict) -> str:
    ok = sorted((r for r in recs if "error" not in r), key=lambda r: -r["total_s"])
    lines = [
        f"CUDA replay; empty launch {floor['launch_s'] * 1e6:.1f} us, copy {floor['copy_gbs']:.0f} GB/s.",
        "total = calls x replayed time. 'LAUNCH' = under 3 launches of work per call;",
        "f64 rows show the float32 time; linalg rows show cuSOLVER / MAGMA.",
        f"{'total':>8s} {'calls':>8s} {'per call':>9s} {'GB/s':>6s}  notes  op | dtype | shapes",
    ]
    for r in ok[:60]:
        notes = []
        if r["launches_per_call"] < 3:
            notes.append("LAUNCH")
        if "seconds_per_call_f32" in r:
            notes.append(f"f32 {r['seconds_per_call'] / r['seconds_per_call_f32']:.0f}x faster")
        libs = [(k[8:], v) for k, v in r.items() if k in ("seconds_cusolver", "seconds_magma")]
        if libs:
            notes.append(
                " / ".join(f"{k} {_fmt_s(v) if isinstance(v, float) else v}" for k, v in libs)
            )
        lines.append(
            f"{r['total_s']:7.2f}s {r['calls']:8d} {_fmt_s(r['seconds_per_call'])} {r['gbs']:6.0f}  "
            f"{'; '.join(notes)}  {r['op']} | {r['dtype']} | {r['shapes'][:60]}  [{','.join(r['stages'][:3])}]"
        )
    bad = [r for r in recs if "error" in r]
    if bad:
        lines.append(
            f"\n{len(bad)} unreplayable entries, e.g. "
            + "; ".join(f"{r['op']}: {r['error']}" for r in bad[:5])
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("audit_dir", type=Path, help="an ffs_benchmark -audit ops run directory")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--top", type=int, default=40, help="entries to replay (per ranking on CUDA)")
    ap.add_argument(
        "--threads", help="comma list for --device cpu (default 1,2,4,.. up to the quota)"
    )
    ap.add_argument("--min-time", type=float, default=0.3, help="seconds per CPU timing")
    ap.add_argument("--json", type=Path, help="write the replay records here")
    args = ap.parse_args()

    torch.set_flush_denormal(True)
    rows, tool_threads = load_census(args.audit_dir, args.device)
    if not rows:
        sys.exit(f"no {args.device} ops with call templates under {args.audit_dir}")
    print(f"{len(rows)} distinct {args.device} entries in the census", file=sys.stderr)

    if args.device == "cpu":
        if args.threads:
            threads = [int(t) for t in args.threads.split(",")]
        else:
            import os

            quota = _bench_cpu_host()._cpu_quota()
            cap = int(quota) if quota else (len(os.sched_getaffinity(0)) // 2 or 1)
            threads = sorted({1, *(2**k for k in range(1, 8) if 2**k <= cap), cap})
        recs = replay_cpu(rows, threads, args.min_time, args.top)
        text = report_cpu(recs, tool_threads)
        payload: dict[str, Any] = {"device": "cpu", "threads": threads, "records": recs}
    else:
        recs, floor = replay_cuda(rows, args.top)
        text = report_cuda(recs, floor)
        payload = {
            "device": "cuda",
            "floor": floor,
            "records": recs,
            "gpu": torch.cuda.get_device_name(),
        }
    print(text)
    if args.json:
        args.json.write_text(json.dumps(payload, indent=1, default=str) + "\n")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"\n({time.perf_counter() - t0:.0f} s)", file=sys.stderr)
