"""Surface window logic that needs Qt but not a GPU: commands, uploads, picking."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import torch

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
nib = pytest.importorskip("nibabel")

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
    assert set(c._pending["lh"]) == {"white", "pial"}
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
