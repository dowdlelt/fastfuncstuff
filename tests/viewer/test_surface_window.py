"""Surface window logic that needs Qt but not a GPU: commands, uploads, picking."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import torch

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
nib = pytest.importorskip("nibabel")

from fastfuncstuff.viewer.commands import Aspect  # noqa: E402
from tests.viewer.test_surface_render import _sheet  # noqa: E402


@pytest.fixture
def window(tmp_path):
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.surfacewindow import SurfaceWindow
    from fastfuncstuff.viewer.vocab import OpenView

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    aff[:3, 3] = [-29.0, -29.0, -9.0]
    nib.save(nib.Nifti1Image(np.zeros((30, 30, 10), np.float32), aff), str(tmp_path / "a.nii.gz"))
    session = ViewerSession(device=torch.device("cpu"))
    session.load(str(tmp_path / "a.nii.gz"))
    session.surfaces.hemis = {"lh": _sheet()}
    session.surfaces.version = {"lh": 1}
    session.do(OpenView("S1", "surface", "axial"))
    win = SurfaceWindow("S1", session, session.do)
    win.canvas.resize(320, 320)
    win.apply(session.state.viewports.get("S1"))
    yield session, win
    win.close()
    session.close()
    app.processEvents()


def test_shape_command_morphs_from_what_is_drawn(window):
    from fastfuncstuff.viewer.vocab import SetSurfaceShape

    session, win = window
    c = win.canvas
    before = c.current("lh")
    assert before is not None
    session.do(SetSurfaceShape("S1", "flat"))
    win.apply(session.state.viewports.get("S1"))
    # Shape A is where it was drawn, shape B the new one; the morph runs between.
    np.testing.assert_allclose(c._cpu["lh"]["posA"], before[0], atol=1e-5)
    assert c._anim.state() == c._anim.State.Running
    with pytest.raises(ValueError, match="unknown surface shape"):
        session.do(SetSurfaceShape("S1", "pancake"))


def test_an_edit_under_a_laid_out_shape_uploads_only_where_data_is_sampled(window):
    session, win = window
    c = win.canvas
    c._pending.clear()
    hemi = session.surfaces.hemis["lh"]
    hemi.states["pial"][10] += np.float32([0, 0, 1.0])
    session.surfaces.version["lh"] += 1
    from fastfuncstuff.viewer.commands import Aspect

    win.refresh(Aspect.SLICES)
    # Areas too: equivolume depth depends on them, and an edit changes them.
    assert set(c._pending["lh"]) == {"white", "pial", "areas"}
    np.testing.assert_array_equal(c._cpu["lh"]["pial"], hemi.states["pial"])


def test_click_on_the_surface_gives_the_mid_depth_point_under_it(window):
    from PySide6.QtCore import QPointF

    from fastfuncstuff.viewer.vocab import SetSurfaceDepth, SetSurfaceShape

    session, win = window
    session.do(SetSurfaceShape("S1", "white"))
    session.do(SetSurfaceDepth("S1", 0.5, 0.5, 1))
    win.apply(session.state.viewports.get("S1"))
    c = win.canvas
    c._anim.stop()
    c.morph = 1.0
    win._reset_camera()
    mm = c.pick_mm(QPointF(160.0, 160.0))
    assert mm is not None
    # Seen from straight above the sheet's centre: x, y near the target, and z
    # half-way between white (0) and pial (3).
    target = c.camera.target
    assert abs(mm[0] - target[0]) < 1.0 and abs(mm[1] - target[1]) < 1.0
    assert mm[2] == pytest.approx(1.5, abs=1e-4)


def test_depth_and_hemisphere_commands_round_trip_through_a_script(window):
    from fastfuncstuff.viewer.vocab import SetSurfaceDepth, SetSurfaceHemis

    session, _ = window
    session.do(SetSurfaceDepth("S1", 0.0, 1.0, 5))
    session.do(SetSurfaceHemis("S1", "rh", 20.0))
    vp = session.state.viewports.get("S1")
    assert vp.depth == (0.0, 1.0) and vp.samples == 5
    assert vp.hemis == "rh" and vp.split == 20.0
    script = session.to_script()
    assert "SET_SURFACE_DEPTH S1 0.0 1.0 5" in script
    assert "SET_SURFACE_HEMIS S1 rh 20.0" in script


def test_a_turned_hemisphere_pivots_on_its_centre_and_still_picks_true_mm(window):
    from PySide6.QtCore import QPointF

    from fastfuncstuff.viewer import surface3d as s3
    from fastfuncstuff.viewer.vocab import SetSurfaceShape

    session, win = window
    session.do(SetSurfaceShape("S1", "white"))
    win.apply(session.state.viewports.get("S1"))
    c = win.canvas
    c._anim.stop()
    c.morph = 1.0
    win._reset_camera()
    before = c.drawn_positions("lh")
    assert before is not None
    c.hemi_rotation["lh"] = s3._rotation([0, 0, 1], np.pi / 3)
    after = c.drawn_positions("lh")
    assert after is not None
    np.testing.assert_allclose(after.mean(0), before.mean(0), atol=1e-3)
    assert np.abs(after - before).max() > 5.0
    # Turned on screen, but the point under the cursor is still a point of the
    # sheet in scanner space: picking goes through barycentrics, not display xyz.
    mm = c.pick_mm(QPointF(160.0, 160.0))
    assert mm is not None and mm[2] == pytest.approx(1.5, abs=1e-4)
    win._reset_camera()
    assert c.hemi_rotation == {}


def test_a_recomputed_mode_overlay_reaches_the_surface(window):
    """The bug: instacorr re-adopts its map under the same key on every seed
    click, and the texture cache keyed on the layer alone kept drawing the first
    map while the slices moved on."""
    from fastfuncstuff.viewer.commands import Aspect
    from fastfuncstuff.viewer.modes.base import ComputedOverlay

    session, win = window
    source = session.state.layers.base.key
    aff = session.state.layers.base.affine

    def install(fill: float) -> np.ndarray:
        values = np.full((30, 30, 10), fill, np.float32)
        session.install_computed_overlay(source, ComputedOverlay(values, aff, "icorr"))
        win.refresh(Aspect.LAYERS)
        return win.canvas.overlays[-1].value

    first = install(0.2)
    second = install(0.7)
    assert not np.array_equal(first, second), "the surface still draws the first map"
    assert np.allclose(second, 0.7)


def test_ctrl_click_on_the_surface_seeds_and_a_plain_click_only_locates(window):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    from fastfuncstuff.viewer.vocab import SetSurfaceShape

    session, win = window
    session.do(SetSurfaceShape("S1", "white"))
    win.apply(session.state.viewports.get("S1"))
    c = win.canvas
    c._anim.stop()
    c.morph = 1.0
    win._reset_camera()
    located, seeded = [], []
    win.located.connect(lambda *mm: located.append(mm))
    win.seeded.connect(lambda *mm: seeded.append(mm))

    def click(mods: Qt.KeyboardModifier) -> None:
        at = QPointF(160.0, 160.0)
        for kind in (QMouseEvent.Type.MouseButtonPress, QMouseEvent.Type.MouseButtonRelease):
            buttons = (
                Qt.MouseButton.LeftButton
                if kind == QMouseEvent.Type.MouseButtonPress
                else Qt.MouseButton.NoButton
            )
            ev = QMouseEvent(kind, at, at, Qt.MouseButton.LeftButton, buttons, mods)
            (
                c.mousePressEvent
                if kind == QMouseEvent.Type.MouseButtonPress
                else c.mouseReleaseEvent
            )(ev)

    click(Qt.KeyboardModifier.ControlModifier)
    assert len(seeded) == 1 and not located
    assert seeded[0] == pytest.approx(c.pick_mm(QPointF(160.0, 160.0)))
    click(Qt.KeyboardModifier.NoModifier)
    assert len(seeded) == 1 and len(located) == 1


def test_a_surface_seed_lands_on_the_voxel_the_crosshair_moved_to(window):
    from fastfuncstuff.viewer.ui.manager import WindowManager
    from fastfuncstuff.viewer.vocab import SetSeed, SetXYZ

    session, _ = window
    sent = []

    def dispatch(cmd):
        sent.append(cmd)
        session.do(cmd)

    WindowManager(session, dispatch)._on_seeded(-9.0, -19.0, 1.0)
    assert [type(c) for c in sent] == [SetXYZ, SetSeed]
    # The fixture's grid: origin (-29, -29, -9) mm, 2 mm voxels.
    assert session.state.crosshair == (10, 5, 5)
    assert session.state.seed == (10, 5, 5)


def test_the_crosshair_marks_its_nearest_vertex_and_c_aims_the_camera_there(window):
    """The crosshair is rarely at mid-depth; a mark only within 2 mm of it drew
    nothing. And on an inflated surface its mm is not where it is drawn."""
    from fastfuncstuff.viewer.vocab import SetSurfaceShape, SetXYZ

    session, win = window
    session.do(SetSurfaceShape("S1", "flat"))
    win.apply(session.state.viewports.get("S1"))
    c = win.canvas
    c._anim.stop()
    c.morph = 1.0
    # 9 mm above the sheet (mid-depth is z = 1.5): far off the surface.
    session.do(SetXYZ(5.0, -7.0, 9.0))
    win.refresh(Aspect.ALL)
    cx, cy, cz, r = c.cross
    assert r > 0 and cz == pytest.approx(1.5, abs=1e-4)
    assert abs(cx - 5.0) < 1.5 and abs(cy + 7.0) < 1.5

    hemi, k = c.nearest_vertex((5.0, -7.0, 9.0))
    win._centre_view()
    assert np.allclose(c.camera.target, c.drawn_positions(hemi)[k], atol=1e-4)

    win._toggle_cross()
    assert c.cross[3] == 0.0


def test_a_shape_change_keeps_the_zoom_relative_to_the_brain(window):
    """Pial to inflated filled the window: the camera stayed put while the
    shape grew. It now backs off in step with the morph."""
    from fastfuncstuff.viewer.vocab import SetSurfaceShape

    session, win = window
    session.surfaces.hemis = {"lh": _sheet(pial_scale=2.0)}
    session.surfaces.version = {"lh": 2}
    session.do(SetSurfaceShape("S1", "white"))
    win.apply(session.state.viewports.get("S1"))
    c = win.canvas
    c._anim.stop()
    c._on_morph(1.0)
    win._reset_camera()
    before = c.camera.distance
    session.do(SetSurfaceShape("S1", "pial"))
    win.apply(session.state.viewports.get("S1"))
    c._anim.stop()
    c._on_morph(0.0)
    assert c.camera.distance == pytest.approx(before)
    c._on_morph(1.0)
    # The pial sheet is twice the white's width (its diagonal near twice too).
    assert 1.8 < c.camera.distance / before < 2.1
