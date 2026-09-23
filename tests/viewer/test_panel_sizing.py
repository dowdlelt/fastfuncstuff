"""The controller panel fits in the controller.

Two failures this guards, both of which look like a cosmetic complaint and are
actually a functional one: a section that takes space it does not need pushes
the sections below it out of view, and a section that demands more width than
the window has makes the whole panel scroll sideways and clips the numbers off
the far edge. Either way a control someone needs is not on screen, and there is
nothing about it to suggest where it went.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

nib = pytest.importorskip("nibabel")
pytest.importorskip("PySide6")

# Must be set before any QApplication is constructed.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from fastfuncstuff.viewer.commands import Aspect  # noqa: E402
from fastfuncstuff.viewer.modes import registry  # noqa: E402
from fastfuncstuff.viewer.session import ViewerSession  # noqa: E402
from fastfuncstuff.viewer.ui.window import LAYER_ROWS, ViewerWindow  # noqa: E402
from fastfuncstuff.viewer.vocab import AddOverlay, SetMode, SetUnderlay  # noqa: E402

CPU = torch.device("cpu")


@pytest.fixture(scope="session")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def datadir(tmp_path):
    rng = np.random.default_rng(11)
    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    for i in range(10):
        data = np.asarray(rng.random((8, 9, 6)) * 100, np.float32)
        nib.save(nib.Nifti1Image(data, aff), str(tmp_path / f"v{i}.nii.gz"))
    return tmp_path


def _window(qapp, datadir, mode="plain"):
    session = ViewerSession(device=CPU)
    win = ViewerWindow(session)
    win.read_directory(datadir)
    session.do(SetMode(mode))
    win.refresh(Aspect.THEME | Aspect.LAYERS)
    win.show()
    for _ in range(4):
        qapp.processEvents()
    return win


def _settle(win, qapp, dirty=Aspect.LAYERS):
    win.refresh(dirty)
    for _ in range(4):
        qapp.processEvents()


# -- the layer list ---------------------------------------------------------


def test_the_layer_list_is_as_tall_as_the_stack(qapp, datadir):
    """A stack of two used to claim the same 190 pixels as a stack of nine, and
    the difference came out of whatever was below it."""
    win = _window(qapp, datadir)
    win.session.do(SetUnderlay(str(datadir / "v0.nii.gz")))
    _settle(win, qapp)
    small = win.layer_list.height()

    for i in range(1, 5):
        win.session.do(AddOverlay(str(datadir / f"v{i}.nii.gz")))
    _settle(win, qapp)
    assert win.layer_list.height() > small
    win.close()


def test_a_long_stack_scrolls_rather_than_growing(qapp, datadir):
    """The list gives up, not the panel."""
    win = _window(qapp, datadir)
    win.session.do(SetUnderlay(str(datadir / "v0.nii.gz")))
    for i in range(1, 10):
        win.session.do(AddOverlay(str(datadir / f"v{i}.nii.gz")))
    _settle(win, qapp)
    assert len(win.session.state.layers) > LAYER_ROWS
    assert win.layer_list.verticalScrollBar().isVisible()
    assert win.layer_list.height() < 200
    win.close()


# -- the panel's width ------------------------------------------------------


@pytest.mark.parametrize("mode", sorted(registry.names()))
def test_no_mode_makes_the_panel_scroll_sideways(qapp, datadir, mode):
    """A panel wider than its window clips the range numbers off the far edge,
    and nothing on screen says they are there."""
    win = _window(qapp, datadir, mode=mode)
    win.session.do(SetUnderlay(str(datadir / "v0.nii.gz")))
    win.session.do(AddOverlay(str(datadir / "v1.nii.gz")))
    _settle(win, qapp)
    scroll = win.centralWidget()
    assert not scroll.horizontalScrollBar().isVisible(), (
        f"{mode}: panel wants {scroll.widget().minimumSizeHint().width()}px "
        f"in a {win.width()}px window"
    )
    win.close()


def test_every_window_button_is_reachable_at_the_default_width(qapp, datadir):
    """They used to go into a toolbar overflow menu behind a chevron, which is
    indistinguishable from not being there."""
    win = _window(qapp, datadir)
    buttons = [b for b in win.findChildren(QtWidgets.QPushButton) if b.objectName() == "tool"]
    assert len(buttons) >= 8
    assert all(b.isVisible() for b in buttons)
    # And they wrapped rather than running off the side.
    assert max(b.x() + b.width() for b in buttons) <= win.width()
    win.close()


# -- where a new window lands -----------------------------------------------


def test_a_mode_panel_does_not_open_on_top_of_an_image(qapp, datadir, monkeypatch):
    """The HRF curve and the map it redraws are meant to be read together, so
    the one arriving must not cover the other. Qt's default is to stack."""
    from PySide6 import QtCore

    from fastfuncstuff.viewer.ui import manager as manager_mod

    # Patched before the window exists: its image windows are built on the
    # first refresh. The offscreen screen is 800x800, which cannot hold three
    # image windows and a panel however they are arranged, and the placement
    # is what is under test -- not whether a small screen has room.
    monkeypatch.setattr(
        manager_mod.WindowManager, "_work_area", lambda self, anchor: QtCore.QRect(0, 0, 2400, 1200)
    )
    win = _window(qapp, datadir)
    win.session.do(SetUnderlay(str(datadir / "v0.nii.gz")))
    _settle(win, qapp, Aspect.LAYERS | Aspect.VIEWPORTS)

    win.session.do(SetMode("instaglm"))
    win.refresh(win.session.open_mode_panels() | Aspect.VIEWPORTS)
    for _ in range(4):
        qapp.processEvents()

    rects = {}
    for viewport in win.session.state.viewports:
        rects[viewport.id] = win.manager.windows[viewport.id].frameGeometry()
    assert len(rects) >= 2

    overlaps = [
        (a, b)
        for i, a in enumerate(rects)
        for b in list(rects)[i + 1 :]
        if rects[a].intersected(rects[b]).isValid()
        and rects[a].intersected(rects[b]).width() > 0
        and rects[a].intersected(rects[b]).height() > 0
    ]
    assert not overlaps, f"windows opened on top of each other: {overlaps}"
    win.close()


def test_a_remembered_geometry_still_wins(qapp, datadir):
    """Placement is for windows that have never had one. A replayed session
    must come back to the rectangles it recorded."""
    from fastfuncstuff.viewer.viewports import ViewKind
    from fastfuncstuff.viewer.vocab import SetViewGeometry

    win = _window(qapp, datadir)
    win.session.do(SetUnderlay(str(datadir / "v0.nii.gz")))
    _settle(win, qapp, Aspect.LAYERS | Aspect.VIEWPORTS)
    vid = win.session.open_view(ViewKind.GRAPH, win.session.state.viewports.images[0].plane)
    win.session.do(SetViewGeometry(vid, 111, 222, 333, 444))
    _settle(win, qapp, Aspect.VIEWPORTS)
    assert win.manager.windows[vid].geometry().topLeft().toTuple() == (111, 222)
    win.close()
