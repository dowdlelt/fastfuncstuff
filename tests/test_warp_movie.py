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
    got = rec._frames[0].values[0].float()
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
    rec = WarpMovieRecorder(moving, planes, every=1, reference=moving, tool="test", device=CPU)
    zero = tuple(torch.zeros(shape) for _ in range(3))
    rec.capture_displacement(zero, label="L1 it 0")  # type: ignore[arg-type]
    rec.capture_displacement(zero, label="L1 best", pinned=True)  # type: ignore[arg-type]
    out = rec.render(str(tmp_path / "m"), fps=4, size=64, fmt="gif", overlay="edges")
    assert out is not None and out.endswith(".gif")
    assert (tmp_path / "m.gif").stat().st_size > 0


def test_display_edges_do_not_fill_a_cut_along_a_surface():
    """A slice lying along a boundary must not draw a filled patch.

    On the MNI template the mid-sagittal cut runs along the medial surface, and a 3-D
    edge map sliced there showed a solid occipital blob. The minimal version: a
    smooth step across z, cut exactly at the step.
    """
    from fastfuncstuff.processing.edges import edge_map
    from fastfuncstuff.viz.compose import display_edges

    n = 24
    z = torch.arange(n).float()[:, None, None].expand(n, n, n)
    step = torch.sigmoid(z - n // 2) * 100.0
    cut = n // 2

    assert (edge_map(step)[cut] > 0).float().mean() > 0.9  # the artifact
    assert (display_edges(step[cut].numpy(), (n * 3, n * 3)) > 0).mean() < 0.01


def test_edge_overlay_is_drawn_in_colour():
    shape = (8, 8, 8)
    planes = build_slice_planes(shape, np.eye(4), "ax")
    img = planes.split(np.full(64, 0.5, np.float32))
    edge = np.zeros((8, 8), np.float32)
    edge[4, :] = 1.0
    frame = compose_frame([img], planes.views, 64, [(0.0, 1.0)], edges=[edge], edge_vmax=1.0)
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


def _padded_centres_field(full, pads, coarse, seed=3):
    """A smooth field on a coarse, align_corners=False grid of a padded full grid."""
    import torch.nn.functional as F

    padded = tuple(n + 2 * p for n, p in zip(full, pads, strict=True))
    g = torch.Generator().manual_seed(seed)
    small = [torch.randn(4, 4, 4, generator=g) * 0.5 for _ in range(3)]
    level = [
        F.interpolate(c[None, None], size=coarse, mode="trilinear", align_corners=True)[0, 0]
        for c in small
    ]
    ratio = [padded[2] / coarse[2], padded[1] / coarse[1], padded[0] / coarse[0]]  # x, y, z
    up = [
        F.interpolate(c[None, None], size=padded, mode="trilinear", align_corners=False)[0, 0] * r
        for c, r in zip(level, ratio, strict=True)
    ]
    return padded, tuple(level), tuple(up)


def test_padded_centres_level_matches_qwarp_upsampling():
    """qwarp: a padded grid, and pyramid octaves made by align_corners=False resizes."""
    from fastfuncstuff.viz.warp_movie import FieldFrame

    full, pads = (14, 16, 18), (2, 3, 4)
    padded, level, up = _padded_centres_field(full, pads, (10, 11, 13))
    moving = _texture(full)
    planes = build_slice_planes(full, np.eye(4), "ax,sag,cor", (9, 8, 7))
    rec = WarpMovieRecorder(moving, planes, every=1, device=CPU)
    rec.set_frame(FieldFrame(offset=pads, full_shape=padded, mapping="centres"))
    rec.capture_displacement(level)  # type: ignore[arg-type]

    # The way qwarp itself applies it: pad the source, warp on the padded grid, crop.
    import torch.nn.functional as F

    pz, py, px = pads
    src_p = F.pad(moving, (px, px, py, py, pz, pz))
    warped = warp_image_linear(src_p, *up)[pz:-pz, py:-py, px:-px]
    expected = warped.reshape(-1)[planes.flat_indices()]
    torch.testing.assert_close(rec._frames[0].values[0].float(), expected, atol=2e-3, rtol=1e-3)


def test_stride_level_scales_displacement_by_the_stride():
    """blipflip: a level made by vol[::s] carries displacement in level voxels."""
    from fastfuncstuff.viz.warp_movie import FieldFrame

    full, s = (13, 16, 16), 2
    level_shape = tuple(-(-n // s) for n in full)
    moving = _texture(full)
    # Linear in the level's y index, so trilinear sampling is exact everywhere.
    yy = torch.arange(level_shape[1]).float()[None, :, None].expand(level_shape)
    level = (torch.zeros(level_shape), 0.05 * yy + 0.2, torch.zeros(level_shape))
    planes = build_slice_planes(full, np.eye(4), "ax,cor", (8, 8, 6))
    rec = WarpMovieRecorder(moving, planes, every=1, device=CPU)
    rec.set_frame(FieldFrame(mapping="stride"))
    rec.capture_displacement(level)  # type: ignore[arg-type]

    yf = torch.arange(full[1]).float()[None, :, None].expand(full)
    full_disp = (torch.zeros(full), s * (0.05 * yf / s + 0.2), torch.zeros(full))
    expected = warp_image_linear(moving, *full_disp).reshape(-1)[planes.flat_indices()]
    # Keep points inside the level grid's span and whose sample stays in the volume:
    # past either edge only the boundary handling differs.
    py = torch.as_tensor(planes.points[:, 1])
    shifted = py + full_disp[1].reshape(-1)[planes.flat_indices()]
    inside = (py <= (level_shape[1] - 1) * s) & (shifted < full[1] - 1)
    got = rec._frames[0].values[0].float()
    torch.testing.assert_close(got[inside], expected[inside], atol=2e-3, rtol=1e-3)


def test_chain_matches_composed_warp():
    """formwarp: the moving->fixed field is two half-warps composed."""
    from fastfuncstuff.processing.nwarpforge import NonlinearWarp, compose_warp_then_warp

    shape = (14, 15, 16)
    moving = _texture(shape)
    g = torch.Generator().manual_seed(5)
    a = tuple((torch.randn(shape, generator=g) * 0.3).clamp(-0.8, 0.8) for _ in range(3))
    b = tuple((torch.randn(shape, generator=g) * 0.3).clamp(-0.8, 0.8) for _ in range(3))
    for f in (a, b):
        for c in f:
            c[[0, 1, -2, -1]] = 0
            c[:, [0, 1, -2, -1]] = 0
            c[:, :, [0, 1, -2, -1]] = 0
    planes = build_slice_planes(shape, np.eye(4), "ax,sag,cor")
    rec = WarpMovieRecorder(moving, planes, every=1, device=CPU)
    rec.capture([[a, b]])  # type: ignore[list-item]

    c = compose_warp_then_warp(NonlinearWarp(*a, {}), NonlinearWarp(*b, {}))
    expected = warp_image_linear(moving, c.xd, c.yd, c.zd).reshape(-1)[planes.flat_indices()]
    torch.testing.assert_close(rec._frames[0].values[0].float(), expected, atol=2e-3, rtol=1e-3)


def test_modulation_scales_by_the_jacobian():
    shape = (12, 12, 12)
    moving = torch.full(shape, 10.0)
    planes = build_slice_planes(shape, np.eye(4), "ax", (6, 6, 6))
    yy = torch.arange(12).float()[None, :, None].expand(shape)
    stretch = (torch.zeros(shape), 0.2 * (yy - 6), torch.zeros(shape))  # d(disp)/dy = 0.2
    rec = WarpMovieRecorder([moving, moving], planes, every=1, device=CPU)
    zero = tuple(torch.zeros(shape) for _ in range(3))
    rec.capture([[stretch], [zero]], modulate=True)  # type: ignore[list-item]
    up, still = rec._frames[0].values.float()
    ax = planes.split(up.numpy())[0]
    np.testing.assert_allclose(ax[3:8, 3:8], 12.0, atol=1e-3)  # 10 * (1 + 0.2)
    np.testing.assert_allclose(still.numpy(), 10.0, atol=1e-4)


def test_two_row_render(tmp_path):
    shape = (10, 12, 12)
    planes = build_slice_planes(shape, np.eye(4), "ax,sag")
    rec = WarpMovieRecorder(
        [_texture(shape, 1), _texture(shape, 2)],
        planes,
        every=1,
        row_labels=["up", "down"],
        device=CPU,
    )
    rec.capture_identity()
    out = rec.render(str(tmp_path / "rows.gif"), size=48, fmt="gif")
    assert out is not None and (tmp_path / "rows.gif").stat().st_size > 0


def test_formwarp_pins_one_frame_per_level():
    from fastfuncstuff.processing.formwarp import SynConfig, formwarp

    shape = (16, 18, 18)
    fixed = _texture(shape, seed=4)
    moving = _texture(shape, seed=4).roll(1, dims=1)
    planes = build_slice_planes(shape, np.eye(4), "ax")
    rec = WarpMovieRecorder(moving, planes, max_frames=6, device=CPU)
    cfg = SynConfig(
        shrink_factors=(2, 1),
        smoothing_sigmas=(1.0, 0.0),
        iterations=(6, 6),
        convergence_window=0,
        verb=0,
    )
    formwarp(fixed, moving, config=cfg, recorder=rec)
    pinned = [f.label for f in rec._frames if f.pinned]
    assert len(pinned) == 2 and pinned[0].startswith("L1/2") and pinned[1].startswith("L2/2")
    assert any(not f.pinned for f in rec._frames)


def test_qwarp_captures_phases_and_levels():
    from fastfuncstuff.processing.warp import QwarpConfig, qwarp

    shape = (20, 20, 20)
    base = _texture(shape, seed=6)
    source = _texture(shape, seed=6).roll(1, dims=0)
    planes = build_slice_planes(shape, np.eye(4), "ax,cor")
    rec = WarpMovieRecorder(source, planes, max_frames=40, device=CPU)
    cfg = QwarpConfig(minpatch=11, max_level=1, cost_method="pearson", verb=0, movie_recorder=rec)
    qwarp(base, source, config=cfg, device=CPU)
    labels = [f.label for f in rec._frames]
    assert labels[0] == "start"
    assert any("phase" in lab for lab in labels)
    assert any(f.pinned and "lev=1" in f.label for f in rec._frames)


def test_blipflip_rows_follow_opposite_blips():
    from test_topup import _make_synthetic

    from fastfuncstuff.processing import topup as T

    _, _, scans = _make_synthetic()
    shape = tuple(scans[0].data.shape)
    planes = build_slice_planes(shape, np.eye(4), "ax")
    rec = WarpMovieRecorder([s.data for s in scans], planes, max_frames=10, device=CPU)
    cfg = T.TopupConfig(
        warpres=[16, 10], fwhm=[5, 2], lam=[1e-3, 1e-4], miter=[4, 4], subsamp=[2, 1]
    )
    res = T.run_topup(scans, (3.0, 2.5, 2.5), cfg, progress=False, recorder=rec)
    pinned = [f for f in rec._frames if f.pinned]
    assert [f.label.split()[0] for f in pinned] == ["start", "L1/2", "L2/2", "final"]
    # The final frame is the tool's own unwarped output, sampled at the planes.
    final = pinned[-1].values.float()
    py = torch.as_tensor(planes.points[:, 1])
    for row, unwarped, scan in zip(final, res.unwarped, scans, strict=True):
        # Where the PE sample leaves the volume topup clamps and the movie pads zero.
        shifted = py + (res.field_hz * scan.readout * scan.sign).reshape(-1)[planes.flat_indices()]
        inner = (py > 1) & (py < shape[1] - 2) & (shifted > 1) & (shifted < shape[1] - 2)
        # run_topup works on copies rescaled to mean 100 and hands back native units.
        ref = unwarped.float().reshape(-1)[planes.flat_indices()] * (
            100.0 / float(scan.data.mean())
        )
        torch.testing.assert_close(row[inner], ref[inner], atol=0.05, rtol=1e-2)


def test_padded_replacement_images_reach_their_padding():
    """A padded working copy shows the same pixels, and the padding is reachable."""
    shape = (8, 10, 12)
    img = _texture(shape)
    planes = build_slice_planes(shape, np.eye(4), "ax,sag")
    rec = WarpMovieRecorder(img, planes, every=1, device=CPU)
    padded = torch.nn.functional.pad(img, (0, 0, 3, 2, 0, 0), value=-7.0)  # y: 3 before, 2 after
    rec.set_images([padded], offset=(0.0, 3.0, 0.0))
    rec.capture_identity()
    torch.testing.assert_close(
        rec._frames[0].values[0].float(), img.reshape(-1)[planes.flat_indices()], atol=1e-3, rtol=0
    )
    yy_shift = (torch.zeros(shape), torch.full(shape, -3.0), torch.zeros(shape))
    rec.capture_displacement(yy_shift)  # type: ignore[arg-type]
    first_rows = torch.as_tensor(planes.points[:, 1] < 0.5)
    assert torch.allclose(rec._frames[1].values[0].float()[first_rows], torch.tensor(-7.0))


def test_affine_capture_matches_the_tools_own_resampling():
    """allineate: a (4, 4) on a source that never leaves its own grid."""
    from fastfuncstuff.processing.affine import apply_affine, params_to_matrix

    base_shape, src_shape = (12, 14, 16), (10, 12, 13)
    moving = _texture(src_shape)
    p = torch.zeros(12)
    p[0:3] = torch.tensor([1.5, -0.5, 0.7])
    p[3], p[6:9] = 7.0, 1.0
    matrix = params_to_matrix(p)

    planes = build_slice_planes(base_shape, np.eye(4), "ax,sag,cor")
    with pytest.raises(ValueError, match="own_grid"):
        WarpMovieRecorder(moving, planes, every=1, device=CPU)  # still guarded without it
    rec = WarpMovieRecorder(moving, planes, every=1, own_grid=True, device=CPU)
    rec.capture_matrix(matrix)

    expected = apply_affine(moving, matrix, base_shape, zero_outside=True).reshape(-1)[
        planes.flat_indices()
    ]
    # Compare where the sample lands a voxel inside the source: at the edge only the
    # out-of-bounds handling differs (a hard zero against grid_sample's ramp).
    pts = torch.as_tensor(planes.points).flip(-1)  # (N, 3) x, y, z
    mapped = pts @ matrix[:3, :3].T + matrix[:3, 3]
    hi = torch.tensor([src_shape[2] - 2, src_shape[1] - 2, src_shape[0] - 2]).float()
    inside = ((mapped > 1.0) & (mapped < hi)).all(dim=1)
    assert inside.any()
    torch.testing.assert_close(
        rec._frames[0].values[0].float()[inside], expected[inside], atol=2e-3, rtol=1e-3
    )


def test_edges_can_skip_the_base_row():
    shape = (8, 8, 8)
    planes = build_slice_planes(shape, np.eye(4), "ax")
    img = planes.split(np.full(64, 0.5, np.float32))
    edge = np.zeros((8, 8), np.float32)
    edge[4, :] = 1.0
    frame = compose_frame(
        [img, img],
        planes.views,
        64,
        [(0.0, 1.0)] * 2,
        edges=[edge],
        edge_vmax=1.0,
        edge_rows=[False, True],
    )
    coloured = (frame.astype(int)[..., 0] - frame.astype(int)[..., 2]) > 100
    assert not coloured[: frame.shape[0] // 2].any()
    assert coloured[frame.shape[0] // 2 :].any()


def test_base_row_adds_a_row_to_the_movie(tmp_path):
    from PIL import Image

    shape = (10, 12, 12)
    planes = build_slice_planes(shape, np.eye(4), "ax")
    rec = WarpMovieRecorder(
        _texture(shape), planes, every=1, reference=_texture(shape, 9), device=CPU
    )
    rec.capture_identity()
    plain = rec.render(str(tmp_path / "plain"), size=48, fmt="gif", overlay="edges")
    stacked = rec.render(str(tmp_path / "stacked"), size=48, fmt="gif", overlay="base+edges")
    assert plain is not None and stacked is not None
    h_plain = Image.open(plain).size[1]
    h_stacked = Image.open(stacked).size[1]
    assert h_stacked - h_plain == 48 + 4  # one more panel row plus the gap


def test_display_window_survives_an_empty_opening_frame(monkeypatch):
    """An affine can start with the source off the grid; that must not set the scale."""
    import fastfuncstuff.viz.warp_movie as wm

    shape = (10, 12, 12)
    planes = build_slice_planes(shape, np.eye(4), "ax")
    rec = WarpMovieRecorder(_texture(shape), planes, every=1, device=CPU)
    away = torch.eye(4)
    away[0, 3] = 1000.0
    rec.capture_matrix(away, label="off the grid")
    rec.capture_matrix(torch.eye(4), label="home")
    assert float(rec._frames[0].values.abs().max()) == 0.0

    written: list[np.ndarray] = []
    monkeypatch.setattr(wm, "write_movie", lambda frames, path, fps, fmt: written.append(frames))
    rec.render("unused", size=48, fmt="gif")
    last = written[0][-1]
    # With a window taken from the empty frame alone the texture clips to solid white.
    assert (last == 255).mean() < 0.5
    assert last.std() > 0
