#!/usr/bin/env python3
"""What can *this* GPU do, and which backend switches matter for ffs on it?

The CUDA sibling of ``bench_cpu_host.py``. Where the CPU question is "threads
or jobs", the GPU questions are different, and this script answers them on the
card in front of you:

1. **Host report.** Card, SMs, VRAM, clocks and power limit, PCIe link (a card
   that trained at x4 or gen 1 caps every host<->device copy), driver / CUDA /
   cuDNN / torch versions, and the backend settings an ffs CLI leaves in force
   (``configure_torch_backends``).

2. **Peaks and floors.** GEMM at f32 / TF32 / bf16 / f64, device memory
   bandwidth, the empty-kernel launch time, a ``.item()`` round trip, and
   host<->device bandwidth from pageable vs pinned memory. Peaks turn any ffs
   kernel's time into a fraction of what the card can do; floors say when a
   loop of small calls is launch-bound rather than compute-bound.

3. **Switches, at ffs shapes.** Each row times the same work two or more ways:
   cuSOLVER vs MAGMA for the dense-linalg shapes ffs uses (batched small
   Cholesky/eigh, tall-skinny QR, SVD, pinv, solve) and float32 vs float64;
   ``cudnn.benchmark`` on/off for 3-D smoothing convolutions; eager vs
   ``torch.compile`` for an elementwise chain; eager vs CUDA-graph replay for a
   loop of small kernels. The winner column is what to try in ffs; the ratio
   says whether it is worth the trouble.

For the ops ffs actually dispatches, at the shapes it actually uses, run
``ffs_benchmark -audit ops`` and ``scripts/replay_op_census.py --device cuda``.

Typical use::

    python scripts/bench_gpu_host.py --json gpu_a.json        # ~2-4 min
    python scripts/bench_gpu_host.py --quick                  # ~30 s smoke run
    python scripts/bench_gpu_host.py --compare gpu_a.json gpu_b.json

Run it on an idle card: another process's kernels inflate every time here.
Times are means over a window of CUDA-event-timed calls after warm-up.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def time_cuda(fn: Callable[[], Any], min_time: float = 0.25, max_reps: int = 5000) -> float:
    """Mean seconds per call (CUDA events), after a warm-up that absorbs autotuning."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    reps, total, batch = 0, 0.0, 1
    while total < min_time and reps < max_reps:
        start.record()
        for _ in range(batch):
            fn()
        end.record()
        end.synchronize()
        total += start.elapsed_time(end) / 1e3
        reps += batch
        batch = max(1, min(batch * 2, max_reps - reps))
    return total / reps


def time_host(fn: Callable[[], Any], min_time: float = 0.25) -> float:
    """Wall seconds per call for work that blocks the host anyway (syncs, copies)."""
    fn()
    torch.cuda.synchronize()
    reps, t0 = 0, time.perf_counter()
    while True:
        fn()
        reps += 1
        if time.perf_counter() - t0 >= min_time and reps >= 3:
            break
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


# ---------------------------------------------------------------------------
# 1. Host report
# ---------------------------------------------------------------------------

_SMI_FIELDS = (
    "name,driver_version,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,"
    "pcie.link.width.max,clocks.max.sm,clocks.max.mem,power.limit,power.max_limit,"
    "persistence_mode,compute_mode,utilization.gpu,memory.used,memory.total"
)


def host_report() -> dict[str, Any]:
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    info: dict[str, Any] = {
        "gpu": props.name,
        "sm_count": props.multi_processor_count,
        "capability": f"{props.major}.{props.minor}",
        "vram_gib": props.total_memory / 2**30,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(),
    }
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={_SMI_FIELDS}",
                "--format=csv,noheader,nounits",
                f"-i={dev}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        info["smi"] = dict(
            zip(_SMI_FIELDS.split(","), (v.strip() for v in out.split(",")), strict=False)
        )
    except (OSError, subprocess.SubprocessError):
        info["smi"] = {}
    # What an ffs CLI runs with: configure_torch_backends on a CUDA device.
    try:
        from fastfuncstuff.utils import configure_torch_backends

        configure_torch_backends(torch.device("cuda"))
        info["ffs_backends"] = {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn.benchmark": torch.backends.cudnn.benchmark,
            "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
            "preferred_linalg_library": str(torch.backends.cuda.preferred_linalg_library()),
        }
    except ImportError:
        info["ffs_backends"] = None
    return info


def print_host(info: dict[str, Any]) -> None:
    smi = info.get("smi", {})
    print(
        f"GPU      {info['gpu']}  sm_{info['capability'].replace('.', '')}  {info['sm_count']} SMs  "
        f"{info['vram_gib']:.1f} GiB"
    )
    print(
        f"versions torch {info['torch']}  CUDA {info['cuda']}  cuDNN {info['cudnn']}  "
        f"driver {smi.get('driver_version', '?')}"
    )
    if smi:
        print(
            f"PCIe     gen {smi['pcie.link.gen.current']}/{smi['pcie.link.gen.max']}  "
            f"x{smi['pcie.link.width.current']}/x{smi['pcie.link.width.max']}  "
            "(current link drops at idle; read it from the transfer rows below)"
        )
        print(
            f"clocks   SM {smi['clocks.max.sm']} MHz  mem {smi['clocks.max.mem']} MHz  "
            f"power limit {smi['power.limit']}/{smi['power.max_limit']} W  "
            f"persistence {smi['persistence_mode']}"
        )
        busy = smi.get("utilization.gpu", "0")
        if busy.isdigit() and int(busy) > 5:
            print(f"WARNING  card is {busy}% busy before the run; every number below is inflated")
    if info.get("ffs_backends"):
        print("ffs      " + "  ".join(f"{k}={v}" for k, v in info["ffs_backends"].items()))


# ---------------------------------------------------------------------------
# 2. Peaks and floors
# ---------------------------------------------------------------------------


def _gemm(dtype: torch.dtype, n: int, tf32: bool) -> float:
    a = torch.randn(n, n, device="cuda").to(dtype)
    b = torch.randn(n, n, device="cuda").to(dtype)
    out = torch.empty(n, n, device="cuda", dtype=dtype)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    try:
        t = time_cuda(lambda: torch.matmul(a, b, out=out))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return 2 * n**3 / t / 1e12


def peaks(quick: bool) -> list[dict[str, Any]]:
    n = 2048 if quick else 4096
    rows: list[dict[str, Any]] = []

    def add(name: str, value: float, unit: str, note: str = "") -> None:
        rows.append({"name": name, "value": value, "unit": unit, "note": note})
        print(f"  {name:34s} {value:10.2f} {unit:8s} {note}")

    add(f"GEMM f32 {n} (TF32 off)", _gemm(torch.float32, n, False), "TFLOP/s")
    add(
        f"GEMM f32 {n} (TF32 on)",
        _gemm(torch.float32, n, True),
        "TFLOP/s",
        "ffs default for matmul",
    )
    add(f"GEMM bf16 {n}", _gemm(torch.bfloat16, n, False), "TFLOP/s")
    add(
        f"GEMM f64 {n // 2}",
        _gemm(torch.float64, n // 2, False),
        "TFLOP/s",
        "the f64 rate: anything f64 on the hot path pays this",
    )

    big = torch.empty((64 if quick else 256) * 2**20 // 4, device="cuda")
    dst = torch.empty_like(big)
    t = time_cuda(lambda: dst.copy_(big))
    add("device copy", 2 * big.numel() * 4 / t / 1e9, "GB/s", "read+write")
    x, y = torch.randn_like(big), torch.randn_like(big)
    t = time_cuda(lambda: torch.add(y, x, alpha=1.5, out=dst))
    add("triad out=y+a*x", 3 * big.numel() * 4 / t / 1e9, "GB/s")

    tiny = torch.zeros(1, device="cuda")
    add(
        "empty kernel (queued)",
        time_cuda(lambda: tiny.add_(0), 0.1) * 1e6,
        "us",
        "GPU-side spacing of back-to-back launches",
    )
    add(
        "launch from Python (host)",
        time_host(lambda: tiny.add_(0), 0.1) * 1e6,
        "us",
        "host cost per op: a Python loop of small ops is bound by this",
    )
    add(
        ".item() round trip",
        time_host(lambda: tiny.item(), 0.1) * 1e6,
        "us",
        "every implicit sync costs at least this plus the queue drain",
    )

    nbytes = (64 if quick else 256) * 2**20
    host = torch.empty(nbytes // 4)
    pinned = torch.empty(nbytes // 4, pin_memory=True)
    dev = torch.empty(nbytes // 4, device="cuda")
    for label, src in (("pageable", host), ("pinned", pinned)):
        t = time_host(lambda s=src: dev.copy_(s))
        add(f"H2D {label}", nbytes / t / 1e9, "GB/s")
        t = time_host(lambda s=src: s.copy_(dev))
        add(f"D2H {label}", nbytes / t / 1e9, "GB/s")
    return rows


# ---------------------------------------------------------------------------
# 3. Switches
# ---------------------------------------------------------------------------


def _spd(batch: tuple[int, ...], k: int, dtype: torch.dtype) -> torch.Tensor:
    b = torch.randn(*batch, k, k, device="cuda", dtype=dtype)
    return b @ b.transpose(-1, -2) / k + torch.eye(k, device="cuda", dtype=dtype)


def _linalg_cases(quick: bool) -> list[tuple[str, Callable[[torch.dtype], Callable[[], Any]]]]:
    s = 4 if quick else 1  # shrink batch sizes in quick mode

    def chol_solve(batch: int, k: int, rhs: int):
        def build(dt):
            a = _spd((batch // s,), k, dt)
            b = torch.randn(batch // s, k, rhs, device="cuda", dtype=dt)
            return lambda: torch.cholesky_solve(b, torch.linalg.cholesky(a))

        return build

    def eigh(batch: int, k: int):
        def build(dt):
            a = _spd((batch // s,), k, dt)
            return lambda: torch.linalg.eigh(a)

        return build

    def qr(*shape: int):
        def build(dt):
            a = torch.randn(*shape, device="cuda", dtype=dt)
            return lambda: torch.linalg.qr(a)

        return build

    def svd(*shape: int):
        def build(dt):
            a = torch.randn(*shape, device="cuda", dtype=dt)
            return lambda: torch.linalg.svd(a, full_matrices=False)

        return build

    def pinv(*shape: int):
        def build(dt):
            a = torch.randn(*shape, device="cuda", dtype=dt)
            return lambda: torch.linalg.pinv(a)

        return build

    def solve(n: int, rhs: int):
        def build(dt):
            a = _spd((), n, dt)
            b = torch.randn(n, rhs, device="cuda", dtype=dt)
            return lambda: torch.linalg.solve(a, b)

        return build

    return [
        # Per-voxel normal equations / REML (ARMA) / per-voxel HRF: many tiny systems.
        ("chol+solve 100k x (6,6)", chol_solve(100_000, 6, 1)),
        ("chol+solve 50k x (12,12)", chol_solve(50_000, 12, 1)),
        ("chol+solve 5k x (64,64)", chol_solve(5_000, 64, 1)),
        ("eigh 50k x (8,8)", eigh(50_000, 8)),
        ("eigh 2k x (64,64)", eigh(2_000, 64)),
        ("eigh 1 x (1000,1000)", eigh(1, 1000)),
        # Nuisance projectors and designs: tall-skinny.
        ("QR (20000,60)", qr(20_000, 60)),
        ("QR (2000,300)", qr(2_000, 300)),
        ("QR batched 256 x (300,12)", qr(256 // s, 300, 12)),
        # PCA / denoising / NORDIC patches.
        ("SVD (2000,500)", svd(2_000, 500)),
        ("SVD batched 512 x (100,40)", svd(512 // s, 100, 40)),
        ("pinv (1000,150)", pinv(1_000, 150)),
        ("solve (500,500)x(500,5000)", solve(500, 5_000)),
    ]


def linalg_switches(quick: bool) -> list[dict[str, Any]]:
    print(f"  {'case':32s} {'default':>9s} {'cusolver':>9s} {'magma':>9s} {'f64/f32':>8s}  best")
    lib0 = torch.backends.cuda.preferred_linalg_library()
    rows = []
    for name, build in _linalg_cases(quick):
        row: dict[str, Any] = {"name": name}
        try:
            fn = build(torch.float32)
            torch.backends.cuda.preferred_linalg_library("default")
            row["default"] = time_cuda(fn)
            for lib in ("cusolver", "magma"):
                torch.backends.cuda.preferred_linalg_library(lib)
                try:
                    row[lib] = time_cuda(fn)
                except RuntimeError as exc:
                    row[lib] = None
                    row[f"{lib}_error"] = str(exc)[:80]
            torch.backends.cuda.preferred_linalg_library("default")
            row["f64"] = time_cuda(build(torch.float64), min_time=0.15)
        except RuntimeError as exc:
            row["error"] = str(exc)[:100]
            print(f"  {name:32s} error: {row['error']}")
            rows.append(row)
            continue
        finally:
            torch.backends.cuda.preferred_linalg_library(lib0)
        timed = {k: row[k] for k in ("default", "cusolver", "magma") if row.get(k)}
        best = min(timed, key=lambda k: timed[k])
        gain = row["default"] / timed[best]
        row["best"], row["gain_vs_default"] = best, gain
        verdict = f"{best} {gain:.1f}x" if best != "default" and gain > 1.15 else "default"

        def ms(v: float | None) -> str:
            return f"{v * 1e3:8.2f}ms" if v else "      n/a"

        print(
            f"  {name:32s} {ms(row['default'])} {ms(row.get('cusolver'))} {ms(row.get('magma'))} "
            f"{row['f64'] / row['default']:7.1f}x  {verdict}"
        )
        rows.append(row)
        torch.cuda.empty_cache()
    return rows


def other_switches(quick: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(name: str, a_label: str, a: float, b_label: str, b: float, note: str = "") -> None:
        rows.append({"name": name, a_label: a, b_label: b, "ratio": a / b, "note": note})
        win = b_label if a / b > 1.1 else a_label if b / a > 1.1 else "tie"
        print(
            f"  {name:34s} {a_label} {a * 1e3:8.2f}ms  {b_label} {b * 1e3:8.2f}ms  "
            f"-> {win} ({max(a / b, b / a):.2f}x)  {note}"
        )

    # 3-D smoothing convolution, the shape of every blur/local-correlation in ffs.
    vol = torch.randn(1, 1, 96, 128, 128, device="cuda")
    for k in (3, 7):
        w = torch.randn(1, 1, k, k, k, device="cuda")
        conv = lambda w=w, k=k: torch.nn.functional.conv3d(vol, w, padding=k // 2)  # noqa: E731
        prev = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False
        off = time_cuda(conv)
        torch.backends.cudnn.benchmark = True
        on = time_cuda(conv)
        torch.backends.cudnn.benchmark = prev
        add(f"conv3d k={k} 96x128x128", "bench_off", off, "bench_on", on, "ffs sets bench_on")

    # Elementwise chain: eager launches 6 kernels and round-trips memory 6x.
    n = (16 if quick else 64) * 2**20
    a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")

    def chain(x, y):
        return torch.tanh((x * 1.3 + y).abs().sqrt() - 0.5) * y

    eager = time_cuda(lambda: chain(a, b))
    try:
        compiled = torch.compile(chain)
        comp = time_cuda(lambda: compiled(a, b))
        add(
            f"6-op elementwise {n // 2**20}M",
            "eager",
            eager,
            "compile",
            comp,
            "fusion headroom on bandwidth-bound chains",
        )
    except Exception as exc:  # noqa: BLE001 -- inductor missing is a result, not a crash
        print(f"  torch.compile unavailable: {type(exc).__name__}")

    # A loop of small kernels -- per-iteration work in moco/qwarp/LK-style loops.
    small = torch.randn(64, 64, 64, device="cuda")
    out = torch.empty_like(small)

    def step():
        out.copy_(small)
        for _ in range(20):
            out.mul_(0.999).add_(small, alpha=1e-3)

    eager = time_cuda(step)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        step()
    graphed = time_cuda(g.replay)
    add(
        "41 small kernels (64^3)",
        "eager",
        eager,
        "graph",
        graphed,
        "launch overhead a CUDA graph removes",
    )

    # 3-D resampling: grid_sample is the registration workhorse.
    grid = torch.rand(1, 96, 128, 128, 3, device="cuda") * 2 - 1
    gs = lambda: torch.nn.functional.grid_sample(vol, grid, align_corners=True)  # noqa: E731
    t32 = time_cuda(gs)
    gs16 = lambda: torch.nn.functional.grid_sample(vol.half(), grid.half(), align_corners=True)  # noqa: E731
    add("grid_sample 3-D 96x128x128", "f32", t32, "f16", time_cuda(gs16), "f16 = 11-bit mantissa")
    return rows


# ---------------------------------------------------------------------------
# Main / compare
# ---------------------------------------------------------------------------


def compare(paths: list[str]) -> None:
    runs = [json.load(open(p)) for p in paths]
    print("peaks:")
    names = [r["name"] for r in runs[0]["peaks"]]
    for n in names:
        vals = [next((r["value"] for r in run["peaks"] if r["name"] == n), None) for run in runs]
        print(
            f"  {n:34s} " + "  ".join(f"{v:10.2f}" if v is not None else "       n/a" for v in vals)
        )
    print("linalg (default backend, ms):")
    for row in runs[0]["linalg"]:
        vals = [
            next((r.get("default") for r in run["linalg"] if r["name"] == row["name"]), None)
            for run in runs
        ]
        print(
            f"  {row['name']:34s} "
            + "  ".join(f"{v * 1e3:10.2f}" if v else "       n/a" for v in vals)
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--quick", action="store_true", help="smaller problems, ~30 s")
    ap.add_argument("--skip-linalg", action="store_true")
    ap.add_argument("--skip-other", action="store_true")
    ap.add_argument("--json", help="write host info and results here (feed to --compare)")
    ap.add_argument("--compare", nargs="+", metavar="JSON", help="compare earlier --json runs")
    args = ap.parse_args()
    if args.compare:
        compare(args.compare)
        return
    if not torch.cuda.is_available():
        sys.exit("no CUDA device")

    info = host_report()
    print_host(info)
    print("\npeaks and floors:")
    result: dict[str, Any] = {"host": info, "quick": args.quick, "peaks": peaks(args.quick)}
    torch.cuda.empty_cache()  # peaks' 256 MiB buffers are freed on return
    if not args.skip_linalg:
        print("\ndense linalg at ffs shapes (float32 unless noted):")
        result["linalg"] = linalg_switches(args.quick)
    if not args.skip_other:
        print("\nother switches:")
        result["other"] = other_switches(args.quick)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1, default=str)


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"\n({time.perf_counter() - t0:.0f} s)")
