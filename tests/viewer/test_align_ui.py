"""Align mode in the image windows: the ring turns, the centre slides.

Real mouse events on the pane, because the arithmetic that can go wrong here is
the screen-to-world one -- a flipped plane, a letterboxed image -- and the only
honest check of that is a drag that starts where the ring is drawn.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

nib = pytest.importorskip("nibabel")
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from fastfuncstuff.viewer import align  # noqa: E402
from fastfuncstuff.viewer.session import ViewerSession  # noqa: E402
from fastfuncstuff.viewer.state import Plane  # noqa: E402
from fastfuncstuff.viewer.ui.window import ViewerWindow  # noqa: E402
from fastfuncstuff.viewer.vocab import Load, SetInput, SetMode  # noqa: E402

CPU = torch.device("cpu")
LEFT = QtCore.Qt.MouseButton.LeftButton
NONE = QtCore.Qt.MouseButton.NoButton


@pytest.fixture
def win(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    rng = np.random.default_rng(2)
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    aff[:3, 3] = -20.0
    for name in ("fixed", "moving"):
        data = rng.random((21, 21, 21)).astype(np.float32)
        nib.save(nib.Nifti1Image(data, aff), str(tmp_path / f"{name}.nii"))
    session = ViewerSession(device=CPU)
    w = ViewerWindow(session)
    for name in ("fixed", "moving"):
        w.refresh(session.do(Load(str(tmp_path / f"{name}.nii"))))
    moving = session.state.layers[1].key
    w._dispatch(SetMode("align"))
    w._dispatch(SetInput(moving))
    w.show()
    app.processEvents()
    yield w, app, moving
    w.close()


def _axial(w):
    for vp in w.session.state.viewports.images:
        if vp.plane is Plane.AXIAL:
            return w.manager.windows[vp.id]
    raise AssertionError("no axial window")


def _send(pane, kind, pos, button, buttons, mods=QtCore.Qt.KeyboardModifier.NoModifier):
    point = QtCore.QPointF(pos)
    event = QtGui.QMouseEvent(kind, point, pane.mapToGlobal(point), button, buttons, mods)
    QtWidgets.QApplication.sendEvent(pane, event)


def _drag(pane, start, end, mods=QtCore.Qt.KeyboardModifier.NoModifier, steps=8):
    _send(pane, QtCore.QEvent.Type.MouseButtonPress, start, LEFT, LEFT, mods)
    for t in np.linspace(0, 1, steps + 1)[1:]:
        _send(pane, QtCore.QEvent.Type.MouseMove, start + (end - start) * t, NONE, LEFT, mods)
    _send(pane, QtCore.QEvent.Type.MouseButtonRelease, end, LEFT, NONE, mods)


def test_the_ring_is_shown_only_in_align_mode(win):
    w, app, _ = win
    pane = _axial(w).pane
    assert pane._handle is not None
    w._dispatch(SetMode("plain"))
    app.processEvents()
    assert pane._handle is None


def test_dragging_the_centre_slides_the_image_under_the_hand(win):
    w, app, moving = win
    pane = _axial(w).pane
    centre, _ = pane._handle_geometry()
    before = w.session.state.crosshair
    scale = pane._image_scale()
    # Right by 5 image pixels: +R in the axial convention, 2 mm per pixel.
    _drag(pane, centre, centre + QtCore.QPointF(5 * scale, 0))
    app.processEvents()
    xform = align.layer_xform(w.session.state.layers.get(moving))
    assert np.allclose(xform[:3, 3], (10.0, 0.0, 0.0), atol=0.3)
    assert np.allclose(xform[:3, :3], np.eye(3))
    # Grabbing the handle is not a click: the crosshair stays where it was.
    assert w.session.state.crosshair == before


def test_dragging_the_ring_a_quarter_turn_turns_the_image_about_its_centre(win):
    w, app, moving = win
    pane = _axial(w).pane
    mode = w.session.mode
    pivot = mode.pivot_mm()
    centre, radius = pane._handle_geometry()
    start = centre + QtCore.QPointF(radius, 0)
    # Along the ring, a quarter of the way round clockwise on screen.
    path = [
        centre + QtCore.QPointF(radius * np.cos(a), radius * np.sin(a))
        for a in np.linspace(0, np.pi / 2, 12)
    ]
    _send(pane, QtCore.QEvent.Type.MouseButtonPress, start, LEFT, LEFT)
    for point in path[1:]:
        _send(pane, QtCore.QEvent.Type.MouseMove, point, NONE, LEFT)
    _send(pane, QtCore.QEvent.Type.MouseButtonRelease, path[-1], LEFT, NONE)
    app.processEvents()
    xform = align.layer_xform(w.session.state.layers.get(moving))
    angle = np.degrees(np.arccos((np.trace(xform[:3, :3]) - 1) / 2))
    assert angle == pytest.approx(90.0, abs=1.0)
    # About the plane's normal, so the pivot has not moved...
    assert np.allclose(mode.pivot_mm(), pivot, atol=1e-6)
    # ...and clockwise on screen: with R to the right and A up, that takes +R to -A.
    assert np.allclose(xform[:3, :3] @ (1, 0, 0), (0, -1, 0), atol=0.02)
    # The sliders read it back.
    assert w.session.mode.params["rz"] == pytest.approx(-90.0, abs=1.0)


def test_a_click_away_from_the_ring_still_moves_the_crosshair(win):
    w, app, _ = win
    pane = _axial(w).pane
    centre, radius = pane._handle_geometry()
    before = w.session.state.crosshair
    spot = centre + QtCore.QPointF(radius * 0.5, radius * 0.5)
    _send(pane, QtCore.QEvent.Type.MouseButtonPress, spot, LEFT, LEFT)
    _send(pane, QtCore.QEvent.Type.MouseButtonRelease, spot, LEFT, NONE)
    app.processEvents()
    assert w.session.state.crosshair != before
