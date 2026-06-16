"""Tests for the SyN engine (processing/formwarp.py).

Synthetic data only — we verify correctness, not speed: that the engine recovers a
known smooth deformation, that the field-inverse and axis-constraint primitives behave,
and that ``-noXdis`` zeros the X displacement end to end.
"""

from __future__ import annotations

import torch

from fastfuncstuff.processing.formwarp import (
    NO_X_DISP,
    NO_Z_DISP,
    SynConfig,
    _apply_axis_flags,
    _convergence_value,
    formwarp,
    invert_displacement_field,
)

DEVICE = torch.device("cpu")


def _blobs(nz: int, ny: int, nx: int, shift=(0.0, 0.0, 0.0)) -> torch.Tensor:
    """A smooth two-blob phantom, optionally shifted by (dx, dy, dz) voxels."""
    dx, dy, dz = shift
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )

    def blob(cx, cy, cz, sx=6.0, sy=6.0, sz=5.0):
        return torch.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2 + ((zz - cz) / sz) ** 2))

    return (
        blob(nx / 2 + dx, ny / 2 + dy, nz / 2 + dz)
        + 0.5 * blob(nx / 2 + 5 + dx, ny / 2 + 4 + dy, nz / 2 + 2 + dz)
    )


def test_config_defaults():
    cfg = SynConfig()
    assert cfg.metric == "cc"
    assert cfg.cc_radius == 4
    assert cfg.update_var == 3.0
    assert cfg.total_var == 0.0
    assert cfg.shrink_factors == (4, 2, 1)
    assert len(cfg.shrink_factors) == len(cfg.smoothing_sigmas) == len(cfg.iterations)


def test_apply_axis_flags_zeros_components():
    xd = torch.randn(4, 4, 4)
    yd = torch.randn(4, 4, 4)
    zd = torch.randn(4, 4, 4)
    nx, ny, nz = _apply_axis_flags(xd, yd, zd, NO_X_DISP | NO_Z_DISP)
    assert torch.count_nonzero(nx) == 0
    assert torch.count_nonzero(nz) == 0
    assert torch.equal(ny, yd)  # untouched axis preserved


def test_invert_displacement_field_roundtrip():
    """Composing a smooth field with its inverse returns ~identity."""
    nz, ny, nx = 16, 18, 20
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    # A gentle, smoothly varying displacement (well within the diffeomorphic regime).
    xd = 1.5 * torch.sin(xx / nx * 3.14159)
    yd = 1.0 * torch.cos(yy / ny * 3.14159)
    zd = 0.5 * torch.sin(zz / nz * 3.14159)

    ex, ey, ez = invert_displacement_field(xd, yd, zd, n_iter=20)

    # Residual of d(x + e(x)) + e(x) should be near zero away from the borders.
    from fastfuncstuff.processing.interp import trilinear_interpolate

    sx = (xx + ex).reshape(-1)
    sy = (yy + ey).reshape(-1)
    sz = (zz + ez).reshape(-1)
    dxe = trilinear_interpolate(xd, sx, sy, sz).reshape(nz, ny, nx)
    resid = (dxe + ex)[2:-2, 2:-2, 2:-2].abs().max().item()
    assert resid < 0.1


def test_formwarp_recovers_known_shift():
    """SyN reduces the fixed/moving mismatch by a large factor."""
    nz, ny, nx = 24, 28, 30
    fixed = _blobs(nz, ny, nx)
    moving = _blobs(nz, ny, nx, shift=(2.5, -1.5, 1.0))

    cfg = SynConfig(
        metric="cc", cc_radius=3, grad_step=0.4,
        shrink_factors=(2, 1), smoothing_sigmas=(1.0, 0.0), iterations=(25, 25),
        verb=0,
    )
    res = formwarp(fixed, moving, config=cfg, device=DEVICE)

    before = ((fixed - moving) ** 2).mean().item()
    after = ((fixed - res.warped) ** 2).mean().item()
    assert after < 0.1 * before  # at least a 10x reduction in residual

    # Shapes preserved; outputs are full-grid voxel-unit fields.
    assert tuple(res.warped.shape) == (nz, ny, nx)
    assert tuple(res.fwd[0].shape) == (nz, ny, nx)
    assert tuple(res.fixed_to_mid[0].shape) == (nz, ny, nx)


def test_convergence_value_trend():
    """Positive while cost falls; ~0/negative once it flattens or rises."""
    falling = [-(0.1 * i) for i in range(10)]  # strictly decreasing
    flat = [-1.0] * 10
    rising = [-1.0 + 0.05 * i for i in range(10)]
    assert _convergence_value(falling, 10) > 1e-3       # still improving
    assert abs(_convergence_value(flat, 10)) < 1e-9     # converged
    assert _convergence_value(rising, 10) < 0.0         # cost going up -> stop


def test_formwarp_exhaustion_returns_best_not_last():
    """An aggressive step that overshoots must still return a good (best) warp.

    With early stopping off and a large step + many iters, the cost overshoots its
    optimum; best-state restoration must hand back the minimum-cost field, so the
    result is still well below baseline rather than the diverged tail.
    """
    nz, ny, nx = 22, 24, 26
    fixed = _blobs(nz, ny, nx)
    moving = _blobs(nz, ny, nx, shift=(2.5, -1.5, 1.0))

    cfg = SynConfig(
        metric="cc", cc_radius=3, grad_step=0.8,  # deliberately large -> overshoot
        shrink_factors=(1,), smoothing_sigmas=(0.0,), iterations=(120,),
        convergence_window=0,  # no early stop: run to exhaustion
        verb=0,
    )
    res = formwarp(fixed, moving, config=cfg, device=DEVICE)
    before = ((fixed - moving) ** 2).mean().item()
    after = ((fixed - res.warped) ** 2).mean().item()
    assert after < 0.2 * before  # best-restore kept it sane despite overshooting


def test_formwarp_noxdis_zeros_x_component():
    """With NO_X_DISP set, every emitted warp has a zero X component."""
    nz, ny, nx = 20, 22, 24
    fixed = _blobs(nz, ny, nx)
    moving = _blobs(nz, ny, nx, shift=(0.0, 2.0, 0.0))

    cfg = SynConfig(
        metric="cc", cc_radius=3, grad_step=0.4,
        shrink_factors=(2, 1), smoothing_sigmas=(1.0, 0.0), iterations=(15, 15),
        warp_flags=NO_X_DISP, verb=0,
    )
    res = formwarp(fixed, moving, config=cfg, device=DEVICE)

    # Half-warps are pure SyN intermediates: their X component must be exactly zero.
    assert res.fixed_to_mid[0].abs().max().item() == 0.0
    assert res.moving_to_mid[0].abs().max().item() == 0.0
    # The Y half-warp should carry real deformation (the mismatch was along Y).
    assert res.moving_to_mid[1].abs().max().item() > 0.1
