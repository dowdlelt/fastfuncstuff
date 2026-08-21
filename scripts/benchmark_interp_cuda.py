#!/usr/bin/env python3
"""Compare fused CUDA interpolation with the warmed portable backend."""

from __future__ import annotations

import argparse
import os
import statistics

import torch

from fastfuncstuff.processing import interp


def _timed(fn, repeats: int) -> float:
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    timings = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    return statistics.median(timings)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-shape", nargs=3, type=int, default=(96, 112, 96))
    parser.add_argument("-repeats", type=int, default=9)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    nz, ny, nx = args.shape
    source = torch.randn(nz, ny, nx, device="cuda")
    z, y, x = torch.meshgrid(
        torch.arange(nz, device="cuda", dtype=torch.float32),
        torch.arange(ny, device="cuda", dtype=torch.float32),
        torch.arange(nx, device="cuda", dtype=torch.float32),
        indexing="ij",
    )
    x = x + 0.21 + 0.08 * torch.sin(y * 0.07)
    y = y + 0.32 + 0.06 * torch.sin(z * 0.09)
    z = z + 0.43 + 0.05 * torch.sin(x * 0.05)

    # Compare against the best steady-state portable path, not its eager
    # bootstrap calls before the adaptive torch.compile gate trips.
    interp._eager_seconds["cuda"] = 1e6
    os.environ["FFS_INTERP_NO_TRITON"] = "1"
    interp._separable_resample_3d(source, x, y, z, "wsinc5")
    torch.cuda.synchronize()

    print(f"shape={nz}x{ny}x{nx}, GPU={torch.cuda.get_device_name()}")
    print(f"{'kernel':10s} {'portable ms':>12s} {'fused ms':>10s} {'speedup':>9s}")
    for kernel in ("cubic", "quintic", "heptic", "wsinc5"):
        os.environ["FFS_INTERP_NO_TRITON"] = "1"
        old = _timed(
            lambda kernel=kernel: interp._separable_resample_3d(source, x, y, z, kernel),
            args.repeats,
        )
        os.environ.pop("FFS_INTERP_NO_TRITON", None)
        new = _timed(
            lambda kernel=kernel: interp._separable_resample_3d(source, x, y, z, kernel),
            args.repeats,
        )
        print(f"{kernel:10s} {old:12.3f} {new:10.3f} {old / new:8.2f}x")


if __name__ == "__main__":
    main()
