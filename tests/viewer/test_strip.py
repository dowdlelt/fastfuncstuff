"""Neighbour strip: cropped slices either side of an image window's own."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

nib = pytest.importorskip("nibabel")

CPU = torch.device("cpu")


def test_offsets_span_the_reach_evenly_and_never_repeat_a_slice():
    from fastfuncstuff.viewer.strip import strip_offsets

    # Auto over an 8-slice brush: half-way and the edge.
    assert strip_offsets(4, 0, 8.0) == [4, 8]
    assert strip_offsets(6, 0, 9.0) == [3, 6, 9]
    # A reach narrower than the half-count: adjacent slices, not duplicates.
    assert strip_offsets(8, 0, 2.0) == [1, 2, 3, 4]
    assert strip_offsets(4, 0, 0.0) == [1, 2]
    # A fixed step ignores the reach.
    assert strip_offsets(6, 3, 8.0) == [3, 6, 9]
    assert strip_offsets(0, 0, 8.0) == []


def test_auto_reach_is_the_brush_in_slices_and_only_while_editing():
    from fastfuncstuff.viewer.state import DisplayGrid, Plane
    from fastfuncstuff.viewer.strip import auto_reach
    from fastfuncstuff.viewer.viewports import ViewKind, Viewport

    grid = DisplayGrid((10, 10, 10), np.diag([1.0, 1.0, 2.0, 1.0]))
    state = SimpleNamespace(
        grid=grid,
        crosshair=(5, 5, 5),
        surface_editing=True,
        surface_brush=(6.0, 1.0, 0.2, 1.5, -1),
    )
    session = SimpleNamespace(state=state, surfaces=SimpleNamespace(hemis={"lh": None}))
    vp = Viewport("A1", ViewKind.IMAGE, Plane.AXIAL)
    # Axial slices are 2 mm thick here: a 6 mm brush reaches 3 slices.
    assert auto_reach(session, vp) == pytest.approx(3.0)
    state.surface_editing = False
    assert auto_reach(session, vp) == 0.0


@pytest.fixture
def session(tmp_path):
    """Value = z in mm, stored with z *decreasing* along k -- so index order and
    anatomical order disagree, which is what the strip's left/right must follow."""
    from fastfuncstuff.viewer.session import ViewerSession

    aff = np.diag([1.0, 1.0, -1.0, 1.0])
    aff[:3, 3] = (-20.0, -20.0, 20.0)
    ijk = np.stack(np.meshgrid(*[np.arange(41)] * 3, indexing="ij"), -1)
    z = (ijk @ aff[:3, :3].T + aff[:3, 3])[..., 2]
    nib.save(nib.Nifti1Image(z.astype(np.float32), aff), str(tmp_path / "z.nii.gz"))
    s = ViewerSession(device=CPU)
    s.load(str(tmp_path / "z.nii.gz"))
    yield s
    s.close()


def test_cells_run_inferior_to_superior_and_crop_around_the_crosshair(session):
    from fastfuncstuff.viewer.strip import strip_cells
    from fastfuncstuff.viewer.vocab import OpenView, SetViewStrip, SetXYZ

    session.do(OpenView("A1", "image", "axial"))
    session.do(SetXYZ(3.0, -2.0, 0.0))
    session.do(SetViewStrip("A1", 4, 2, 4.0))
    vp = session.state.viewports.get("A1")
    cells = strip_cells(session, vp)
    grid = session.state.grid
    here = session.state.crosshair_mm
    z = [grid.ijk_to_mm(_at(session, c.position))[2] - here[2] for c in cells]
    # Left to right is inferior to superior, whatever way k runs.
    assert z == pytest.approx([-4.0, -2.0, 2.0, 4.0])
    assert [c.mm for c in cells] == pytest.approx(z)
    assert [c.side for c in cells] == ["I", "I", "S", "S"]
    assert cells[-1].label == "S 4.0"
    for c in cells:
        assert c.image is not None
        # A quarter of the plane, with the crosshair in its middle.
        assert c.image.size == (10, 10)
        row, col = c.view.to_image(session.state.crosshair)
        assert abs(row - 5) <= 1 and abs(col - 5) <= 1
    assert "SET_VIEW_STRIP" in session.to_script()


def _at(session, position):
    from fastfuncstuff.viewer.slicing import plane_layout
    from fastfuncstuff.viewer.state import Plane

    ijk = list(session.state.crosshair)
    ijk[plane_layout(session.state.grid.affine, Plane.AXIAL).fixed] = position
    return tuple(float(v) for v in ijk)


def test_cells_off_the_grid_are_empty_and_bad_counts_refused(session):
    from fastfuncstuff.viewer.strip import strip_cells
    from fastfuncstuff.viewer.vocab import OpenView, SetViewStrip, SetXYZ

    session.do(OpenView("A1", "image", "axial"))
    session.do(SetXYZ(0.0, 0.0, 19.0))  # one slice below the top
    session.do(SetViewStrip("A1", 4, 1))
    cells = strip_cells(session, session.state.viewports.get("A1"))
    assert [c.position is None for c in cells] == [False, False, False, True]
    assert cells[-1].image is None
    with pytest.raises(ValueError, match="even"):
        session.do(SetViewStrip("A1", 3))
    with pytest.raises(ValueError, match="slices"):
        session.do(SetViewStrip("A1", 4, -1))


def test_window_shows_the_strip_and_a_click_goes_to_that_slice(session):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from fastfuncstuff.viewer.ui.imagewindow import ImageWindow
    from fastfuncstuff.viewer.vocab import OpenView, SetXYZ

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    session.do(OpenView("A1", "image", "axial"))
    session.do(SetXYZ(0.0, 0.0, 0.0))

    def dispatch(cmd):
        session.do(cmd)
        win.apply(session.state.viewports.get("A1"))
        win.redraw()

    win = ImageWindow("A1", session, dispatch)
    try:
        win.show()
        win.redraw()
        assert not win.strip.isVisibleTo(win)
        win._toggle_strip()
        app.processEvents()
        assert win.strip.isVisibleTo(win) and len(win.strip.cells) == 4
        assert win.strip_button.isChecked()
        win._cycle_strip()
        assert len(win.strip.cells) == 6
        # The rightmost cell is superior: clicking its middle moves the
        # crosshair up by three slices and keeps it in place in-plane.
        cell = win._strip_cells[-1]
        before = session.state.crosshair_mm
        row, col = cell.view.to_image(session.state.crosshair)
        win._strip_pick(len(win._strip_cells) - 1, row, col)
        after = session.state.crosshair_mm
        np.testing.assert_allclose(np.subtract(after, before), (0.0, 0.0, 3.0), atol=1e-6)
        win._toggle_strip()
        assert not win.strip.isVisibleTo(win)
        win._toggle_strip()
        assert len(win.strip.cells) == 6  # comes back at the count it had
    finally:
        win.close()


def test_while_editing_the_strip_spans_the_brush_and_draws_outlines(tmp_path):
    """The default that makes the strip an editing aid: the outer cells sit at
    the brush's reach, and each cell carries the surface outlines."""
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.imagewindow import ImageWindow
    from fastfuncstuff.viewer.vocab import (
        LoadSurfaces,
        OpenView,
        SetSurfaceEditing,
        SetViewStrip,
        SetXYZ,
    )
    from tests.viewer.test_surfaces import _shell_anat, _subject

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    session = ViewerSession(device=CPU)
    session.load(str(_shell_anat(tmp_path)))
    session.do(LoadSurfaces(str(_subject(tmp_path))))
    session.do(OpenView("A1", "image", "axial"))
    session.do(SetXYZ(20.0, 0.0, 0.0))
    session.do(SetViewStrip("A1", 4))
    win = ImageWindow("A1", session, session.do)
    try:
        win.redraw()
        assert [c.offset for c in win._strip_cells] == [-2, -1, 1, 2]
        session.do(SetSurfaceEditing(True))
        win.redraw()
        app.processEvents()
        # Default brush 4 mm on 0.5 mm slices: 8 slices, half-way and the edge.
        reach = round(session.state.surface_brush[0] / 0.5)
        assert sorted(abs(c.offset) for c in win._strip_cells) == [
            reach // 2,
            reach // 2,
            reach,
            reach,
        ]
        assert all(cell._outlines for cell in win.strip.cells)

        # An edit grabbed away from the crosshair: the strip goes there...
        def centred_on(ijk):
            view = win._strip_cells[0].view
            row, col = view.to_image(ijk)
            h, w = view.span
            return abs(row - h // 2) <= 1 and abs(col - w // 2) <= 1

        grid = session.state.grid
        away = (-20.0, 0.0, 0.0)
        away_ijk = tuple(int(round(v)) for v in np.linalg.inv(grid.affine)[:3] @ (*away, 1.0))
        win._focus_strip(away)
        assert centred_on(away_ijk)
        assert not centred_on(session.state.crosshair)
        # ...until the crosshair moves, when it follows the crosshair again.
        session.do(SetXYZ(18.0, 0.0, 0.0))
        win.redraw()
        assert win._strip_focus is None
        assert centred_on(session.state.crosshair)
    finally:
        win.close()
        session.close()
