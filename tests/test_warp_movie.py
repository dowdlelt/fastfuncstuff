"""Tests for the warp-movie recorder and its slice geometry (fastfuncstuff/viz)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.processing.formwarp import _resize_field
from fastfuncstuff.processing.interp import warp_image_linear
from fastfuncstuff.viz.compose import compose_frame, panel_sizes
from fastfuncstuff.viz.slices import build_slice_planes, center_of_mass_ras, parse_views
from fastfuncstuff.viz.warp_movie import WarpMovieRecorder

CPU = torch.device("cpu")


def _lpi_affine(shape_zyx, zooms=(2.0, 2.0, 3.0)):
    """Storage order LPI: i runs R->L, j runs A->P, k runs I->S."""
    return np.diag([-zooms[0], -zooms[1], zooms[2], 1.0])


def test_display_is_independent_of_storage_orientation():
    """The same brain stored RAS and LPI must draw identically -- only the affine differs."""
    nz, ny, nx = 9, 11, 13
    rng = np.random.default_rng(0)
    ras = rng.random((nz, ny, nx)).astype(np.float32)  # index (k, j, i) with i->R, j->A
    # Storing LPI flips i and j.
    lpi = ras[:, ::-1, ::-1].copy()
    ras_aff = np.diag([2.0, 2.0, 3.0, 1.0])
    lpi_aff = _lpi_affine((nz, ny, nx))

    center = (4, 6, 3)
    p_ras = build_slice_planes((nz, ny, nx), ras_aff, ("ax", "sag", "cor"), center)
    p_lpi = build_slice_planes((nz, ny, nx), lpi_aff, ("ax", "sag", "cor"), center)
    v_ras = ras.reshape(-1)[p_ras.flat_indices()]
    v_lpi = lpi.reshape(-1)[p_lpi.flat_indices()]
    np.testing.assert_array_equal(v_ras, v_lpi)


def test_radiological_layout_and_panel_aspect():
    nz, ny, nx = 10, 20, 30
    vol = np.zeros((nz, ny, nx), np.float32)
    vol[5, 10, nx - 1] = 1.0  # the most subject-RIGHT voxel (RAS storage)
    vol[nz - 1, 10, 15] = 2.0  # the most SUPERIOR voxel
    aff = np.diag([1.0, 1.0, 4.0, 1.0])
    planes = build_slice_planes((nz, ny, nx), aff, "ax,cor", (15, 10, 5))
    ax, cor = planes.split(vol.reshape(-1)[planes.flat_indices()])
    # Radiological: subject right on the panel's LEFT; anterior up on the axial.
    assert ax.shape == (ny, nx)
    assert ax[ny - 1 - 10, 0] == 1.0
    # Superior at the top of the coronal.
    assert cor[0, nx - 1 - 15] == 2.0
    # 10 slices x 4 mm is taller than 20 rows x 1 mm: one shared mm-per-pixel scale.
    (h_ax, w_ax), (h_cor, w_cor) = panel_sizes(planes.views, 80)
    assert h_cor == 80 and h_ax == 40 and w_ax == w_cor == 60


def test_parse_views_aliases_and_errors():
    assert parse_views("axial, sag,cor,ax") == ("ax", "sag", "cor")
    with pytest.raises(ValueError):
        parse_views("ax,top")


def test_center_of_mass_follows_the_brain_not_the_grid():
    vol = np.zeros((20, 20, 20), np.float32)
    vol[2:7, 12:17, 3:8] = 1.0
    assert center_of_mass_ras(vol, np.eye(4)) == (5, 14, 4)


def _texture(shape, seed=0):
    from fastfuncstuff.processing.cost import _separable_smooth_3d

    g = torch.Generator().manual_seed(seed)
    return _separable_smooth_3d(torch.randn(shape, generator=g), 1.5) + 3.0


def test_coarse_level_capture_matches_full_resolution_warp():
    """A pyramid-level field must show exactly what that field does at full resolution."""
    full = (16, 20, 24)
    coarse = (8, 10, 12)
    moving = _texture(full)
    g = torch.Generator().manual_seed(1)
    # Zero on the boundary nodes and under a voxel everywhere, so no sample leaves the
    # volume: out-of-bounds handling is not what this test is about.
    field = []
    for _ in range(3):
        c = (torch.randn(coarse, generator=g) * 0.4).clamp(-0.9, 0.9)
        c[[0, -1]] = 0
        c[:, [0, -1]] = 0
        c[:, :, [0, -1]] = 0
        field.append(c)
    field = tuple(field)

    planes = build_slice_planes(full, np.eye(4), "ax,sag,cor", (12, 10, 8))
    rec = WarpMovieRecorder(moving, planes, every=1, device=CPU)
    rec.capture_displacement(field)  # type: ignore[arg-type]

    up = _resize_field(*field, full)
    expected = warp_image_linear(moving, *up).reshape(-1)[planes.flat_indices()]
    got = rec._frames[0].values.float()
    torch.testing.assert_close(got, expected, atol=2e-3, rtol=1e-3)


def test_frame_budget_keeps_even_spacing_and_pinned_frames():
    shape = (6, 6, 6)
    planes = build_slice_planes(shape, np.eye(4), "ax")
    rec = WarpMovieRecorder(torch.ones(shape), planes, max_frames=10, device=CPU)
    zero = tuple(torch.zeros(shape) for _ in range(3))
    for it in range(1000):
        if rec.tick():
            rec.capture_displacement(zero, label=str(it))  # type: ignore[arg-type]
        if it in (300, 700):
            rec.capture_displacement(zero, label="pin", pinned=True)  # type: ignore[arg-type]
    free = [int(f.label) for f in rec._frames if not f.pinned]
    assert 5 <= len(free) <= 10
    steps = np.diff(free)
    assert (steps == steps[0]).all()
    assert free[0] == 0
    assert sum(f.pinned for f in rec._frames) == 2


def test_render_writes_a_gif_with_edges(tmp_path):
    shape = (12, 14, 16)
    moving = _texture(shape)
    planes = build_slice_planes(shape, np.eye(4), "ax,sag,cor")
    edges = torch.zeros(shape)
    edges[:, 7, :] = 1.0
    rec = WarpMovieRecorder(moving, planes, every=1, edges=edges, tool="test", device=CPU)
    zero = tuple(torch.zeros(shape) for _ in range(3))
    rec.capture_displacement(zero, label="L1 it 0")  # type: ignore[arg-type]
    rec.capture_displacement(zero, label="L1 best", pinned=True)  # type: ignore[arg-type]
    out = rec.render(str(tmp_path / "m"), fps=4, size=64, fmt="gif", overlay="edges")
    assert out is not None and out.endswith(".gif")
    assert (tmp_path / "m.gif").stat().st_size > 0


def test_edge_overlay_is_drawn_in_colour():
    shape = (8, 8, 8)
    planes = build_slice_planes(shape, np.eye(4), "ax")
    img = planes.split(np.full(64, 0.5, np.float32))
    edge = np.zeros((8, 8), np.float32)
    edge[4, :] = 1.0
    frame = compose_frame(img, planes.views, 64, (0.0, 1.0), edges=[edge], edge_vmax=1.0)
    rgb = frame.astype(int)
    coloured = (rgb[..., 0] - rgb[..., 2]) > 100
    assert coloured.any()


def test_optiwarp_pins_one_frame_per_level():
    from fastfuncstuff.processing.optiwarp import OptiwarpConfig, optiwarp

    shape = (16, 20, 20)
    fixed = _texture(shape, seed=2)
    moving = _texture(shape, seed=2).roll(1, dims=2)
    planes = build_slice_planes(shape, np.eye(4), "ax,sag")
    rec = WarpMovieRecorder(moving, planes, max_frames=8, device=CPU)
    cfg = OptiwarpConfig(
        match="none",
        shrink_factors=(2, 1),
        smoothing_sigmas=(1.0, 0.0),
        iterations=(15, 15),
        convergence_window=0,
        metric="pearson",
        verb=0,
    )
    optiwarp(fixed, moving, config=cfg, recorder=rec)
    pinned = [f for f in rec._frames if f.pinned]
    assert len(pinned) == 2
    assert pinned[0].label.startswith("L1/2") and pinned[1].label.startswith("L2/2")
    assert sum(not f.pinned for f in rec._frames) <= 8
