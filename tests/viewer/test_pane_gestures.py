"""Mouse gestures on a slice pane: right-drag zooms, middle or shift+drag pans."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6 import QtCore, QtGui  # noqa: E402

from fastfuncstuff.viewer.slicing import Plane  # noqa: E402

B = QtCore.Qt.MouseButton
M = QtCore.Qt.KeyboardModifier


@pytest.fixture
def pane():
    from fastfuncstuff.viewer.ui.panes import ImagePane

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    p = ImagePane(Plane.AXIAL)
    p.resize(200, 200)
    got: dict[str, list] = {"zoomed": [], "panned": [], "picked": []}
    p.zoomed.connect(lambda f: got["zoomed"].append(f))
    p.panned.connect(lambda r, c: got["panned"].append((r, c)))
    p.picked.connect(lambda r, c: got["picked"].append((r, c)))
    yield p, got
    p.close()
    app.processEvents()


def _drag(widget, button, mods, start, end):
    def send(kind, pos, buttons):
        ev = QtGui.QMouseEvent(
            kind, QtCore.QPointF(*pos), QtCore.QPointF(*pos), button, buttons, mods
        )
        QtWidgets.QApplication.sendEvent(widget, ev)

    none = QtCore.Qt.MouseButton.NoButton
    send(QtCore.QEvent.Type.MouseButtonPress, start, button)
    send(QtCore.QEvent.Type.MouseMove, end, button)
    send(QtCore.QEvent.Type.MouseButtonRelease, end, none)


def test_right_drag_up_zooms_in_and_does_not_pan(pane):
    p, got = pane
    _drag(p, B.RightButton, M.NoModifier, (100, 120), (100, 60))
    assert got["zoomed"] and got["zoomed"][0] > 1.0
    assert not got["panned"]


@pytest.mark.parametrize(
    ("button", "mods"), [(B.MiddleButton, M.NoModifier), (B.LeftButton, M.ShiftModifier)]
)
def test_middle_or_shift_drag_pans_and_does_not_move_the_crosshair(pane, button, mods):
    p, got = pane
    _drag(p, button, mods, (100, 100), (130, 100))
    assert got["panned"]
    assert not got["zoomed"]
    assert not got["picked"]
