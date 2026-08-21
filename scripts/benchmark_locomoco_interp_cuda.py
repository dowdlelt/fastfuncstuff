#!/usr/bin/env python3
"""Compare locomoco bilinear, bicubic, and Lanczos refinement on CUDA."""

from __future__ import annotations

import time

import numpy as np
import torch

from fastfuncstuff.processing.locomoco import estimate_residual_flow


def _series(nx: int = 48, ny: int = 48, nz: int = 5):
    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    shifts = np.asarray([0.0, 0.5, 1.0, 1.5, -1.0, -0.5, 0.8, -1.3, 0.3, 2.0], np.float32)

    def signal(yy):
        return np.sin(x / 5.0) * np.cos(yy / 4.0) + 0.5 * np.sin((x + yy) / 3.0) + z * 0.02

    base = signal(y).astype(np.float32)
    data = np.stack([signal(y + float(shift)) for shift in shifts], axis=3).astype(np.float32)
    return base, data, shifts


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    base, data, shifts = _series()
    print(f"GPU={torch.cuda.get_device_name()}, shape={data.shape}")
    print(f"{'mode':10s} {'seconds':>9s} {'max shift err':>14s} {'RMSE':>10s} {'sharpness':>10s}")
    for mode in ("bilinear", "bicubic", "lanczos"):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = estimate_residual_flow(
            data,
            pe_axis=1,
            slice_axis=2,
            ref_mode="first",
            n_iters=6,
            refine_rounds=2,
            warp_interp=mode,
            warp_radius=3,
            device=torch.device("cuda"),
            verbose=False,
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        field = result.pe_displacement().numpy()
        estimate = np.median(field.reshape(-1, field.shape[-1]), axis=0)
        corrected = result.corrected_series().numpy()
        interior = corrected[5:-5, 5:-5, :, :]
        target = base[5:-5, 5:-5, :, None]
        rmse = float(np.sqrt(np.mean((interior - target) ** 2)))
        sharpness = float(np.abs(np.diff(np.median(corrected, axis=3), axis=1)).mean())
        shift_error = float(np.max(np.abs(estimate + shifts)))
        print(f"{mode:10s} {seconds:9.3f} {shift_error:14.4f} {rmse:10.5f} {sharpness:10.5f}")


if __name__ == "__main__":
    main()
