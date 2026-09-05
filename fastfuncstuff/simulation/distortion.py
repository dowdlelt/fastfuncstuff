"""Known phase-encode displacement fields, for scoring a distortion estimator.

``ffs_locomoco`` has no reference implementation to validate against: nothing in
AFNI or FSL estimates the per-frame residual shift along the encode axis that it
estimates. The substitute is a field we chose ourselves -- impose a known
displacement on real data, ask the tool to recover it, and score the recovery.

The forward model here deliberately uses a **Catmull-Rom cubic**, not the
Lanczos-3 windowed sinc locomoco corrects with. Distorting and undistorting with
the same kernel would be an inverse crime: the resampler's own error would
cancel, and the score would flatter the estimator by exactly the amount the
interpolator is wrong.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["synthetic_pe_field", "apply_pe_shift"]


def synthetic_pe_field(
    shape: tuple[int, int, int],
    n_frames: int,
    *,
    amplitude: float = 0.8,
    tr: float = 1.5,
    breathing_hz: float = 0.28,
    drift_cycles: float = 1.5,
    drift_frac: float = 0.25,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """A smooth, respiration-like PE displacement field in VOXELS.

    Returns ``(nx, ny, nz, n_frames)``: a fixed smooth spatial pattern modulated
    by a breathing-band sinusoid plus a slower drift. That is the shape the real
    artifact takes -- a B0 modulation whose spatial pattern is set by the body
    and whose time course is set by the chest -- rather than white noise, which
    no estimator with a pooling window could or should recover.

    Deterministic: the same arguments give the same field, so a benchmark
    threshold means the same thing run to run.
    """
    nx, ny, nz = shape
    ax = torch.linspace(-1.0, 1.0, nx, device=device, dtype=dtype).view(nx, 1, 1)
    ay = torch.linspace(-1.0, 1.0, ny, device=device, dtype=dtype).view(1, ny, 1)
    az = torch.linspace(-1.0, 1.0, nz, device=device, dtype=dtype).view(1, 1, nz)

    # Low order on purpose: a real B0 respiration pattern is smooth over the FOV,
    # and a pattern with structure finer than the estimator's pooling window
    # would test the window, not the estimator.
    spatial = (
        0.7 * torch.cos(0.9 * math.pi * az)
        + 0.5 * torch.sin(0.7 * math.pi * ay + 0.3)
        + 0.3 * torch.cos(0.5 * math.pi * ax - 0.2)
        + 0.2 * torch.cos(0.8 * math.pi * ay) * torch.cos(0.6 * math.pi * az)
    )
    spatial = spatial / spatial.abs().max().clamp(min=1e-12)

    t = torch.arange(n_frames, device=device, dtype=dtype) * tr
    duration = max(float(t[-1]), 1e-6) if n_frames > 1 else 1.0
    temporal = torch.sin(2.0 * math.pi * breathing_hz * t)
    temporal = temporal + drift_frac * torch.sin(2.0 * math.pi * drift_cycles * t / duration + 0.7)
    temporal = temporal / temporal.abs().max().clamp(min=1e-12)

    return amplitude * spatial[..., None] * temporal.view(1, 1, 1, n_frames)


def _catmull_rom_weights(frac: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Catmull-Rom cubic weights for taps at floor-1 .. floor+2."""
    f2 = frac * frac
    f3 = f2 * frac
    w_m1 = -0.5 * f3 + f2 - 0.5 * frac
    w_00 = 1.5 * f3 - 2.5 * f2 + 1.0
    w_p1 = -1.5 * f3 + 2.0 * f2 + 0.5 * frac
    w_p2 = 0.5 * f3 - 0.5 * f2
    return w_m1, w_00, w_p1, w_p2


def apply_pe_shift(series: Tensor, field: Tensor, pe_axis: int) -> Tensor:
    """Resample ``series`` along ``pe_axis`` by ``field`` voxels, per frame.

    ``series`` is ``(nx, ny, nz, T)`` and ``field`` the matching displacement in
    voxels. The convention is a PULL, matching locomoco's own: the output at
    index ``i`` samples the input at ``i + field``. Edges clamp.

    Catmull-Rom cubic, deliberately not the Lanczos-3 locomoco resamples with --
    see the module docstring on the inverse crime.
    """
    if series.shape != field.shape:
        raise ValueError(f"series {tuple(series.shape)} != field {tuple(field.shape)}")
    if pe_axis not in (0, 1, 2):
        raise ValueError(f"pe_axis must be 0, 1 or 2; got {pe_axis}")

    n = series.shape[pe_axis]
    view = [1, 1, 1, 1]
    view[pe_axis] = n
    idx = torch.arange(n, device=series.device, dtype=series.dtype).reshape(view)
    coord = idx + field
    base = torch.floor(coord)
    frac = coord - base
    base = base.to(torch.long)

    weights = _catmull_rom_weights(frac)
    out = torch.zeros_like(series)
    for offset, weight in zip((-1, 0, 1, 2), weights, strict=True):
        tap = (base + offset).clamp_(0, n - 1)
        out = out + weight * series.gather(pe_axis, tap)
    return out
