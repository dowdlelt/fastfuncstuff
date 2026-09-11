#!/usr/bin/env python3
"""Measure which ops belong in ``utils._MPS_CPU_OPS``, and why.

Apple Silicon support is a moving target: torch 2.14 landed native Jacobi SVD,
eigh, QR and Cholesky kernels for Metal, which turned a pile of "MPS cannot do
this" workarounds into stale comments -- and introduced a couple of kernels
that are *correct but pathologically slow*. This script is how we tell those
apart without guessing, so ``_MPS_CPU_OPS`` stays a measured table rather than
folklore.

Run it on the Mac in front of you after a torch upgrade::

    python scripts/bench_mps_policy.py

For each op it reports one of three verdicts:

    MPS         keep it on Metal -- it works and beats the CPU
    CPU-BROKEN  it raises on MPS, so it must be routed off
    CPU-SLOWER  it runs correctly but the CPU wins by more than --margin

then prints the ``_MPS_CPU_OPS`` rows implied by the run, ready to paste.

Timings use the *minimum* of several repeats, not the mean: we want the cost of
the kernel, not of whatever else macOS decided to do on the GPU. Nothing else
should be using the GPU while this runs -- a competing process inflates MPS and
leaves the CPU columns alone, which is exactly the bias that would wrongly
banish an op to the CPU.
"""

from __future__ import annotations

import argparse
import time
import warnings

import torch

# A pathological kernel (torch 2.14's tall-skinny QR runs ~7 s where the CPU
# needs 0.6 ms) must not cost minutes to characterise. One timed call past this
# threshold is already a verdict.
SLOW_ENOUGH_MS = 2000.0


def _build(shape, device, dtype, *, spd=False, seed=0):
    """A deterministic test matrix, optionally symmetric positive definite."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(*shape, generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)
    if spd:
        eye = torch.eye(shape[-1], device=device, dtype=dtype)
        x = x @ x.transpose(-1, -2) + shape[-1] * eye
    return x


# op name -> (label, builder). The op name is the key that goes into
# _MPS_CPU_OPS and is passed to utils.cpu_if_mps at the call site.
CASES: dict[str, tuple[str, object]] = {
    "matmul": (
        "matmul (20000,200)@(200,200)",
        lambda d, t: lambda A=_build((20000, 200), d, t), B=_build((200, 200), d, t, seed=1): A @ B,
    ),
    "ols_solve": (
        "OLS normal equations X(1000,60) Y(1000,50k)",
        lambda d, t: (
            lambda X=_build((1000, 60), d, t), Y=_build((1000, 50000), d, t, seed=1), I=torch.eye(60, device=d, dtype=t): (
                torch.linalg.solve(X.T @ X + I, X.T @ Y)
            )
        ),
    ),
    "qr": (
        "QR (1000,60) -- nuisance projector shape",
        lambda d, t: lambda A=_build((1000, 60), d, t): torch.linalg.qr(A),
    ),
    "qr_wide": (
        "QR (2000,300)",
        lambda d, t: lambda A=_build((2000, 300), d, t): torch.linalg.qr(A),
    ),
    "svd": (
        "SVD (1000,200) -- design/ridge shape",
        lambda d, t: lambda A=_build((1000, 200), d, t): torch.linalg.svd(A, full_matrices=False),
    ),
    "svd_tall": (
        "SVD (5000,100) -- NORDIC patch shape",
        lambda d, t: lambda A=_build((5000, 100), d, t): torch.linalg.svd(A, full_matrices=False),
    ),
    "svd_batched": (
        "SVD batched (64,64,32)",
        lambda d, t: lambda A=_build((64, 64, 32), d, t): torch.linalg.svd(A, full_matrices=False),
    ),
    "eigh": (
        "eigh (500,500)",
        lambda d, t: lambda A=_build((1, 500, 500), d, t, spd=True)[0]: torch.linalg.eigh(A),
    ),
    "eigh_batched": (
        "eigh batched (200,64,64)",
        lambda d, t: lambda A=_build((200, 64, 64), d, t, spd=True): torch.linalg.eigh(A),
    ),
    "cholesky": (
        "cholesky (500,500)",
        lambda d, t: lambda A=_build((1, 500, 500), d, t, spd=True)[0]: torch.linalg.cholesky(A),
    ),
    "cholesky_batched": (
        "cholesky batched (2000,32,32)",
        lambda d, t: lambda A=_build((2000, 32, 32), d, t, spd=True): torch.linalg.cholesky(A),
    ),
    "cholesky_solve_batched": (
        "cholesky_solve batched (2000,32,8)",
        lambda d, t: (
            lambda L=torch.linalg.cholesky(_build((2000, 32, 32), d, t, spd=True)), B=_build((2000, 32, 8), d, t, seed=1): (
                torch.cholesky_solve(B, L)
            )
        ),
    ),
    "lstsq": (
        "lstsq (1000,60)",
        lambda d, t: (
            lambda A=_build((1000, 60), d, t), B=_build((1000, 8), d, t, seed=1): (
                torch.linalg.lstsq(A, B)
            )
        ),
    ),
    "pinv": (
        "pinv (1000,150)",
        lambda d, t: lambda A=_build((1000, 150), d, t): torch.linalg.pinv(A),
    ),
    "pinv_batched": (
        "pinv batched (50,150,150)",
        lambda d, t: lambda A=_build((50, 150, 150), d, t): torch.linalg.pinv(A),
    ),
    "fft": (
        "rfft (100k,256)",
        lambda d, t: lambda A=_build((100000, 256), d, t): torch.fft.rfft(A, dim=-1),
    ),
    "conv3d": (
        "conv3d 64^3 x 7^3",
        lambda d, t: (
            lambda A=_build((1, 1, 64, 64, 64), d, t), K=_build((1, 1, 7, 7, 7), d, t, seed=1): (
                torch.nn.functional.conv3d(A, K, padding=3)
            )
        ),
    ),
    "grid_sample": (
        "grid_sample 3d 128^3 -> 64^3",
        lambda d, t: (
            lambda A=_build((1, 1, 128, 128, 128), d, t), G=_build((1, 64, 64, 64, 3), d, t, seed=1): (
                torch.nn.functional.grid_sample(A, G, align_corners=False)
            )
        ),
    ),
    "reduce_ss": (
        "sum-of-squares (50k,1000)",
        lambda d, t: lambda A=_build((50000, 1000), d, t): (A * A).sum(dim=-1),
    ),
}


def _warm_metal() -> None:
    """Pay Metal's one-time context + shader-cache cost before anything is timed.

    Without this the first MPS case measured absorbs ~100 ms of init and looks
    catastrophically slow, which is how you end up "proving" that matmul is
    20x slower on the GPU.
    """
    w = torch.randn(512, 512, device="mps")
    for _ in range(5):
        w = w @ w.T
        w = w / w.abs().max()
    torch.mps.synchronize()
    del w
    torch.mps.empty_cache()


def _time(
    make_fn, device: str, dtype: torch.dtype, repeats: int
) -> tuple[float | None, str | None]:
    """Best-of-*repeats* milliseconds for a callable, or (None, error)."""
    try:
        fn = make_fn(device, dtype)
    except Exception as exc:  # building the inputs can fail the same way
        return None, f"{type(exc).__name__}: {str(exc).splitlines()[0][:80]}"

    start = time.perf_counter()
    try:
        fn()
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc).splitlines()[0][:80]}"
    if device == "mps":
        torch.mps.synchronize()
    warm_ms = (time.perf_counter() - start) * 1e3
    if warm_ms > SLOW_ENOUGH_MS:
        return warm_ms, None

    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        if device == "mps":
            torch.mps.synchronize()
        best = min(best, time.perf_counter() - start)
    return best * 1e3, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-repeats", type=int, default=5, help="timed calls per op")
    parser.add_argument(
        "-margin",
        type=float,
        default=1.0,
        help="MPS must be at least this many times the CPU's speed to be kept "
        "(1.0 = simply faster)",
    )
    parser.add_argument("-only", nargs="*", help="limit to these op names")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    if not torch.backends.mps.is_available():
        print("MPS is not available on this machine; nothing to measure.")
        return 1

    print(f"torch {torch.__version__}   repeats={args.repeats}   margin={args.margin}x")
    _warm_metal()

    cases = CASES if not args.only else {k: v for k, v in CASES.items() if k in args.only}
    header = (
        f"{'op':24s} {'shape':38s} {'MPS f32':>10s} {'CPU f32':>10s} "
        f"{'CPU f64':>10s} {'speedup':>9s}  verdict"
    )
    print(header)
    print("-" * len(header))

    verdicts: dict[str, tuple[str, str]] = {}
    for op, (label, make_fn) in cases.items():
        mps_ms, mps_err = _time(make_fn, "mps", torch.float32, args.repeats)
        cpu32_ms, _ = _time(make_fn, "cpu", torch.float32, args.repeats)
        cpu64_ms, _ = _time(make_fn, "cpu", torch.float64, args.repeats)

        def fmt(ms):
            return f"{ms:8.2f}ms" if ms is not None else "     ERR"

        if mps_err is not None:
            verdict = "CPU-BROKEN"
            speed = "    -"
            verdicts[op] = (verdict, f"torch {torch.__version__}: {mps_err}")
        else:
            ratio = (cpu32_ms / mps_ms) if (cpu32_ms and mps_ms) else 0.0
            speed = f"{ratio:8.2f}x"
            if ratio >= args.margin:
                verdict = "MPS"
            else:
                verdict = "CPU-SLOWER"
                verdicts[op] = (
                    verdict,
                    f"torch {torch.__version__}: {mps_ms:.1f} ms on MPS vs "
                    f"{cpu32_ms:.2f} ms on CPU ({1 / ratio:.0f}x slower)"
                    if ratio
                    else f"torch {torch.__version__}: slower on MPS",
                )
        print(
            f"{op:24s} {label:38s} {fmt(mps_ms):>10s} {fmt(cpu32_ms):>10s} "
            f"{fmt(cpu64_ms):>10s} {speed:>9s}  {verdict}"
        )

    print("\nspeedup > 1 means MPS wins. Rows implied for utils._MPS_CPU_OPS:\n")
    if not verdicts:
        print("    # (empty -- every measured op belongs on Metal)")
    for op, (_verdict, reason) in sorted(verdicts.items()):
        print(f'    "{op}": "{reason}",')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
