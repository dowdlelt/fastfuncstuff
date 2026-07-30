"""Tests for the SyN engine (processing/formwarp.py).

Synthetic data only — we verify correctness, not speed: that the engine recovers a
known smooth deformation, that the field-inverse and axis-constraint primitives behave,
and that ``-noXdis`` zeros the X displacement end to end.
"""

from __future__ import annotations

import pytest
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

    return blob(nx / 2 + dx, ny / 2 + dy, nz / 2 + dz) + 0.5 * blob(
        nx / 2 + 5 + dx, ny / 2 + 4 + dy, nz / 2 + 2 + dz
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
        metric="cc",
        cc_radius=3,
        grad_step=0.4,
        shrink_factors=(2, 1),
        smoothing_sigmas=(1.0, 0.0),
        iterations=(25, 25),
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
    assert _convergence_value(falling, 10) > 1e-3  # still improving
    assert abs(_convergence_value(flat, 10)) < 1e-9  # converged
    assert _convergence_value(rising, 10) < 0.0  # cost going up -> stop


@pytest.mark.slow
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
        metric="cc",
        cc_radius=3,
        grad_step=0.8,  # deliberately large -> overshoot
        shrink_factors=(1,),
        smoothing_sigmas=(0.0,),
        iterations=(120,),
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
        metric="cc",
        cc_radius=3,
        grad_step=0.4,
        shrink_factors=(2, 1),
        smoothing_sigmas=(1.0, 0.0),
        iterations=(15, 15),
        warp_flags=NO_X_DISP,
        verb=0,
    )
    res = formwarp(fixed, moving, config=cfg, device=DEVICE)

    # Half-warps are pure SyN intermediates: their X component must be exactly zero.
    assert res.fixed_to_mid[0].abs().max().item() == 0.0
    assert res.moving_to_mid[0].abs().max().item() == 0.0
    # The Y half-warp should carry real deformation (the mismatch was along Y).
    assert res.moving_to_mid[1].abs().max().item() > 0.1


def test_data_coverage_mask_peels_only_a_clipped_fov():
    """All-true (and un-eroded) on a full volume; the zero wedge plus its resampling
    ramp is excluded when one is present."""
    from fastfuncstuff.processing.mask import data_coverage_mask

    full = _blobs(12, 14, 16) + 1.0  # strictly nonzero everywhere
    assert bool(data_coverage_mask(full, erode=2).all())  # no wedge => no peel

    clipped = full.clone()
    clipped[:, :, :4] = 0.0
    cov = data_coverage_mask(clipped, erode=1)
    assert not bool(cov[:, :, :5].any())  # wedge + one peeled ramp voxel
    assert bool(cov[2:-2, 2:-2, 6:-2].all())  # interior untouched

    raw = data_coverage_mask(clipped, erode=0)
    assert torch.equal(raw, clipped != 0)


def test_coverage_fill_beats_coverage_exclusion_on_a_clipped_wedge():
    """A source whose FoV was clipped must not have tissue dragged into the empty wedge.

    What makes this work is that the CC's *window statistics* are weighted, not just
    its outer sum. Weighting only the sum decides which windows count but not what goes
    into one, so every window within cc_radius of the boundary still averaged in the
    void and chased the tissue/nothing cliff. Accumulating the boxes under the weight
    means a boundary window measures only its covered part, and the cliff stops
    existing as far as the metric is concerned.
    """
    from fastfuncstuff.processing.mask import data_coverage_mask

    nz, ny, nx = 20, 32, 28
    fixed = _blobs(nz, ny, nx) + 0.2
    moving = _blobs(nz, ny, nx, shift=(0.0, 1.5, 0.0)) + 0.2
    cut = 14  # the source lost everything below y < cut
    moving[:, :cut, :] = 0.0

    cfg = SynConfig(
        metric="cc",
        cc_radius=3,
        grad_step=0.4,
        shrink_factors=(2, 1),
        smoothing_sigmas=(1.0, 0.0),
        iterations=(25, 25),
        verb=0,
    )
    cover = data_coverage_mask(moving, erode=1)

    free = formwarp(fixed, moving, config=cfg, device=DEVICE)
    excl = formwarp(fixed, moving, mask=cover.float(), config=cfg, device=DEVICE)
    fill = formwarp(fixed, moving, moving_cover=cover, config=cfg, device=DEVICE)

    # Displacement reaching down the clipped axis, in the band just above the cut.
    def stretch(res) -> float:
        return res.fwd[1][:, : cut + 4, :].abs().max().item()

    assert stretch(excl) < 0.3 * stretch(free)
    assert stretch(fill) < 0.3 * stretch(free)
    # No tissue dragged into the wedge, and the output keeps honest zeros there
    # (the fill is for the metric only -- it must never reach the warped image).
    assert fill.warped[:, : cut - 2, :].abs().max() < 0.1 * free.warped[:, : cut - 2, :].max()
    # The shared-support region is registered at least as well, not worse.
    resid = lambda w: ((fixed - w)[:, cut + 2 :, :] ** 2).mean().item()  # noqa: E731
    assert resid(fill.warped) <= resid(free.warped)


def test_cli_timeseries_5d_and_folder_match(tmp_path):
    """CLI timeseries mode (4D -source): per-volume SyN writes a 4D warped series
    plus a warp series in both -warp_format modes, and the two formats agree."""
    import numpy as np

    nib = pytest.importorskip("nibabel")
    from fastfuncstuff.cli.formwarp import main as fmain
    from fastfuncstuff.processing.io import load_warp_series

    nz, ny, nx = 16, 18, 14
    base = _blobs(nz, ny, nx)
    src = torch.stack(
        [_blobs(nz, ny, nx, shift=(0.0, s, 0.0)) for s in (0.0, 1.5, -1.0)], dim=-1
    )  # (nz,ny,nx,T)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    bpath = tmp_path / "base.nii.gz"
    spath = tmp_path / "src.nii.gz"
    # save_image wants (nz,ny,nx[,T]); NIfTI on-disk is (nx,ny,nz[,T]).
    nib.save(nib.Nifti1Image(base.permute(2, 1, 0).numpy(), affine), bpath)
    nib.save(nib.Nifti1Image(src.permute(2, 1, 0, 3).numpy(), affine), spath)

    common = [
        "-base",
        str(bpath),
        "-source",
        str(spath),
        "-metric",
        "cc",
        "-shrink",
        "2x1",
        "-smooth",
        "1x0",
        "-iters",
        "4x3",
        "-save_warp",
        "-device",
        "cpu",
        "-verb",
        "0",
    ]
    assert fmain([*common, "-prefix", str(tmp_path / "out5.nii.gz"), "-warp_format", "5d"]) == 0
    assert fmain([*common, "-prefix", str(tmp_path / "outF.nii.gz"), "-warp_format", "folder"]) == 0

    f5 = tmp_path / "out5_WARP.nii.gz"
    assert f5.exists()
    # 4D warped series was written.
    assert nib.load(str(tmp_path / "out5.nii.gz")).shape == (nx, ny, nz, 3)

    x5, y5, z5, _, n5 = load_warp_series(f5)
    xf, yf, zf, _, nf = load_warp_series(str(tmp_path / "outF_WARP"))
    assert n5 == nf == 3
    torch.testing.assert_close(y5, yf)
    torch.testing.assert_close(x5, xf)
