"""vox3d: the box, the CPU twin of the ray, and the window's reading of the stack."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from fastfuncstuff.viewer import vox3d

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
nib = pytest.importorskip("nibabel")
CPU = torch.device("cpu")
SHAPE = (20, 24, 16)


def _affine(flip: bool = False) -> np.ndarray:
    aff = np.diag([2.0, 2.0, 2.5, 1.0])
    if flip:
        aff[0, 0] = -2.0  # LPI-style storage: the box must still be right
    aff[:3, 3] = [-20.0, -24.0, -20.0]
    return aff


@pytest.mark.parametrize("flip", [False, True])
def test_box_corners_are_the_outer_voxel_faces(flip):
    aff = _affine(flip)
    corners = vox3d.box_corners(SHAPE, aff)
    lo_face = aff[:3, :3] @ np.array([-0.5, -0.5, -0.5]) + aff[:3, 3]
    hi_face = aff[:3, :3] @ (np.array(SHAPE) - 0.5) + aff[:3, 3]
    np.testing.assert_allclose(corners[0], lo_face)
    np.testing.assert_allclose(corners[7], hi_face)
    assert vox3d.box_vertices(SHAPE, aff).shape == (36, 3)


def test_uniform_block_is_the_size_the_shader_reads():
    aff = _affine()
    slot = vox3d.VolumeSlot(
        key=("a",),
        data=np.zeros(SHAPE[::-1], np.float32),
        frame=vox3d.slot_frame(SHAPE, aff),
        shape=SHAPE,
        affine=aff,
    )
    for mode in vox3d.RENDER_MODES:
        raw = vox3d.pack_uniforms(
            np.eye(4), np.zeros(3), slot, overlay=slot, anatomy=None, mode=mode
        )
        assert len(raw) == vox3d.UNIFORM_BYTES


def _blobs() -> np.ndarray:
    vol = np.zeros(SHAPE, np.float32)
    vol[5, 12, 8] = 10.0  # bright, far from the eye on +x
    vol[14, 12, 8] = 4.0  # dimmer, nearer the eye on +x
    return vol


def _ray_along_x(aff, j=12, k=8):
    mm = aff[:3, :3] @ np.array([0.0, j, k]) + aff[:3, 3]
    origin = mm + np.array([200.0, 0.0, 0.0])
    return origin, np.array([-1.0, 0.0, 0.0])


def _ijk(aff, mm):
    return tuple(int(v) for v in np.round(np.linalg.solve(aff[:3, :3], mm - aff[:3, 3])))


@pytest.mark.parametrize("flip", [False, True])
def test_mip_picks_the_brightest_and_first_hit_the_nearest(flip):
    aff = _affine(flip)
    vol = _blobs()
    origin, d = _ray_along_x(aff)
    hit = vox3d.pick(origin, d, volume=vol, affine=aff, mode="mip", threshold=1.0)
    assert _ijk(aff, hit) == (5, 12, 8)
    hit = vox3d.pick(origin, d, volume=vol, affine=aff, mode="first hit", threshold=1.0)
    # Nearest to an eye at +x in scanner space: i=14 normally, i=5 when i runs -x.
    assert _ijk(aff, hit) == ((5, 12, 8) if flip else (14, 12, 8))
    # Over threshold for neither: nothing to pick.
    assert vox3d.pick(origin, d, volume=vol, affine=aff, mode="mip", threshold=20.0) is None


def test_the_threshold_reads_the_stat_volume_when_there_is_one():
    aff = _affine()
    vol = _blobs()
    stat = np.zeros(SHAPE, np.float32)
    stat[14, 12, 8] = 9.0  # the dim blob is the significant one
    origin, d = _ray_along_x(aff)
    hit = vox3d.pick(origin, d, volume=vol, affine=aff, mode="mip", threshold=3.0, stat=stat)
    assert _ijk(aff, hit) == (14, 12, 8)


def test_a_clip_plane_cuts_away_what_is_beyond_it():
    aff = _affine()
    vol = _blobs()
    origin, d = _ray_along_x(aff)
    cut_x = float((aff[:3, :3] @ np.array([10.0, 0, 0]) + aff[:3, 3])[0])
    # Keep x <= cut: the bright blob (i=5) stays, the near one (i=14) goes.
    clip = (1.0, 0.0, 0.0, cut_x)
    hit = vox3d.pick(origin, d, volume=vol, affine=aff, mode="first hit", threshold=1.0, clip=clip)
    assert _ijk(aff, hit) == (5, 12, 8)
    clip = (-1.0, 0.0, 0.0, -cut_x)  # keep x >= cut
    hit = vox3d.pick(origin, d, volume=vol, affine=aff, mode="mip", threshold=1.0, clip=clip)
    assert _ijk(aff, hit) == (14, 12, 8)


def test_minip_skips_zeros_as_outside_the_data():
    aff = _affine()
    vol = np.zeros(SHAPE, np.float32)
    vol[3:17, 12, 8] = 5.0
    vol[9, 12, 8] = 1.0
    origin, d = _ray_along_x(aff)
    hit = vox3d.pick(origin, d, volume=vol, affine=aff, mode="minip", threshold=0.0)
    assert _ijk(aff, hit) == (9, 12, 8)


def test_crosshair_segments_span_the_box_through_the_point():
    aff = _affine()
    mm = aff[:3, :3] @ np.array([4.0, 5.0, 6.0]) + aff[:3, 3]
    segs = vox3d.crosshair_segments(mm, SHAPE, aff)
    a, b = segs[0]
    assert _ijk(aff, 0.5 * (a + b))[1:] == (5, 6)
    assert np.linalg.norm(b - a) == pytest.approx(SHAPE[0] * 2.0)


# -- the window's reading of the stack --------------------------------------


def _write(path, data, aff, tr=0.0):
    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), aff)
    if tr:
        img.header["pixdim"][4] = tr
        img.header.set_xyzt_units("mm", "sec")
    nib.save(img, str(path))


@pytest.fixture
def stack(tmp_path):
    from PySide6 import QtWidgets

    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.vocab import SetOverlay, SetUnderlay

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    aff = _affine()
    rng = np.random.default_rng(0)
    anat = rng.normal(100, 5, SHAPE).astype(np.float32)
    run = rng.normal(100, 1, (*SHAPE, 12)).astype(np.float32)
    run[5, 12, 8, :] += np.arange(12) * 10.0  # frame t is brightest at t=11
    _write(tmp_path / "anat.nii.gz", anat, aff)
    _write(tmp_path / "bold.nii.gz", run, aff, tr=1.0)
    s = ViewerSession(device=CPU)
    s.do(SetUnderlay(str(tmp_path / "anat.nii.gz")))
    s.do(SetOverlay(str(tmp_path / "bold.nii.gz")))
    yield s
    s.close()
    app.processEvents()


def test_the_window_draws_the_top_layer_over_the_base_and_follows_time(stack):
    from fastfuncstuff.viewer.commands import Aspect
    from fastfuncstuff.viewer.ui.vox3dwindow import Vox3DWindow
    from fastfuncstuff.viewer.vocab import OpenView, SetIndex

    s = stack
    s.do(OpenView("X1", "vox3d", "axial"))
    win = Vox3DWindow("X1", s, s.do)
    win.apply(s.state.viewports.get("X1"))
    overlay, anatomy = win.chosen_layers()
    assert overlay is not None and anatomy is not None
    assert overlay.name.startswith("bold") and anatomy.name.startswith("anat")
    ov = win.canvas.overlay
    assert ov is not None and ov.data.shape == SHAPE[::-1]
    first = ov.key
    s.do(SetIndex(11))
    win.refresh(Aspect.TIME)
    assert win.canvas.overlay.key != first
    # (nz, ny, nx) texture layout: voxel (5, 12, 8) is data[8, 12, 5].
    assert win.canvas.overlay.data[8, 12, 5] > 150
    win.close()


def test_a_click_in_a_mip_lands_on_the_voxel_it_shows(stack):
    from PySide6 import QtCore

    from fastfuncstuff.viewer.commands import Aspect
    from fastfuncstuff.viewer.layers import SignMode
    from fastfuncstuff.viewer.ui.vox3dwindow import Vox3DWindow
    from fastfuncstuff.viewer.vocab import OpenView, SetIndex

    s = stack
    s.do(SetIndex(11))
    key = s.state.layers.layers[-1].key
    s.state.layers.update(key, threshold=150.0, sign_mode=SignMode.POS)
    s.do(OpenView("X1", "vox3d", "axial"))
    win = Vox3DWindow("X1", s, s.do)
    win.canvas.resize(300, 300)
    win.apply(s.state.viewports.get("X1"))
    win.refresh(Aspect.ALL)
    win.canvas.reset_camera("axial")
    # Where the bright voxel projects, from above.
    aff = _affine()
    mm = aff[:3, :3] @ np.array([5.0, 12.0, 8.0]) + aff[:3, 3]
    px, front = win.canvas.camera.to_widget(mm[None], 300, 300)
    assert front[0]
    hit = win.canvas.pick(QtCore.QPointF(*px[0]))
    assert hit is not None and _ijk(aff, hit) == (5, 12, 8)
    win.close()


@pytest.mark.gpu
def test_the_rendered_mip_draws_the_blob_in_the_overlay_colour(stack):
    """Through the real QRhiWidget: the bright voxel's pixel is the LUT's top
    colour, and a pixel outside the box is the background."""
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        pytest.skip("QRhi needs a real platform plugin")
    from fastfuncstuff.viewer.commands import Aspect
    from fastfuncstuff.viewer.layers import SignMode
    from fastfuncstuff.viewer.ui import theme
    from fastfuncstuff.viewer.ui.vox3dwindow import Vox3DWindow
    from fastfuncstuff.viewer.vocab import OpenView, SetIndex

    s = stack
    s.do(SetIndex(11))
    key = s.state.layers.layers[-1].key
    s.state.layers.update(
        key, threshold=150.0, sign_mode=SignMode.POS, colormap="hot", range_lo=0.0, range_hi=150.0
    )
    s.do(OpenView("X1", "vox3d", "axial"))
    win = Vox3DWindow("X1", s, s.do)
    win.canvas.resize(300, 300)
    win.apply(s.state.viewports.get("X1"))
    win.refresh(Aspect.ALL)
    win.anat.setValue(0)
    win.canvas.reset_camera("axial")
    grabbed = win.canvas.grabFramebuffer()
    img = grabbed.convertToFormat(grabbed.Format.Format_RGBA8888)
    if img.isNull():
        pytest.skip("no QRhi available to render with")
    px = np.array(img.constBits()).reshape(img.height(), img.width(), 4)[..., :3].astype(int)
    aff = _affine()
    mm = aff[:3, :3] @ np.array([5.0, 12.0, 8.0]) + aff[:3, 3]
    at, _ = win.canvas.camera.to_widget(mm[None], img.width(), img.height())
    x, y = (int(v) for v in at[0])
    lut_top = win.canvas.overlay.lut[255, :3].astype(int)
    assert np.abs(px[y, x] - lut_top).max() <= 12, (px[y, x], lut_top)
    bg = theme.palette().bg
    from PySide6 import QtGui

    c = QtGui.QColor(bg)
    assert np.abs(px[2, 2] - [c.red(), c.green(), c.blue()]).max() <= 3
    win.close()
