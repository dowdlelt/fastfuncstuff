"""Oblique windows: a slice tilted about the crosshair, and tilted to cut the cortex square-on."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

nib = pytest.importorskip("nibabel")

CPU = torch.device("cpu")


def test_section_tilt_makes_the_slice_contain_the_sheet_normal_turning_least():
    from fastfuncstuff.viewer.compose import section_tilt

    a0 = np.array([0.0, 0.0, 1.0])  # an axial slice
    n = np.array([np.sin(np.radians(40)), 0.0, np.cos(np.radians(40))])  # cortex 40 deg off
    r = section_tilt(np.eye(3), a0, n)
    a = r @ a0
    assert abs(a @ n) < 1e-9
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    # The least turn: the slice normal leaves z by 90 - 40 = 50 degrees,
    # toward -x, and the in-plane y axis is untouched.
    assert np.degrees(np.arccos(a @ a0)) == pytest.approx(50.0)
    np.testing.assert_allclose(r @ [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)
    # A slice lying in the sheet has no nearest square cut: unchanged.
    np.testing.assert_allclose(section_tilt(np.eye(3), a0, a0), np.eye(3))


@pytest.fixture
def session(tmp_path):
    from fastfuncstuff.viewer.session import ViewerSession

    aff = np.diag([1.0, 1.0, 1.0, 1.0])
    aff[:3, 3] = -20.0
    ijk = np.stack(np.meshgrid(*[np.arange(41)] * 3, indexing="ij"), -1)
    # Value = z in mm, so an axial slice is flat and a tilted one ramps.
    z = (ijk @ aff[:3, :3].T + aff[:3, 3])[..., 2]
    nib.save(nib.Nifti1Image(z.astype(np.float32), aff), str(tmp_path / "z.nii.gz"))
    s = ViewerSession(device=CPU)
    s.load(str(tmp_path / "z.nii.gz"))
    yield s
    s.close()


def _tilt_about_x(deg):
    t = np.radians(deg)
    return (1.0, 0.0, 0.0, 0.0, np.cos(t), -np.sin(t), 0.0, np.sin(t), np.cos(t))


def test_a_tilted_window_samples_the_tilted_plane_through_the_crosshair(session):
    from fastfuncstuff.viewer.compose import render_viewport, view_grid
    from fastfuncstuff.viewer.slicing import extract_plane
    from fastfuncstuff.viewer.vocab import OpenView, SetViewTilt, SetXYZ

    session.do(OpenView("A1", "image", "axial"))
    session.do(SetXYZ(0.0, 0.0, 3.0))
    vp = session.state.viewports.get("A1")
    flat = view_grid(session.state, vp)
    assert flat is session.state.grid
    session.do(SetViewTilt("A1", _tilt_about_x(30.0)))
    vp = session.state.viewports.get("A1")
    grid = view_grid(session.state, vp)
    assert grid.layout_affine is not None
    # The crosshair keeps its voxel: same mm through either grid.
    np.testing.assert_allclose(
        grid.ijk_to_mm(session.state.crosshair),
        session.state.grid.ijk_to_mm(session.state.crosshair),
        atol=1e-9,
    )
    vol = session.display_volume(session.state.layers.base.key)
    from fastfuncstuff.viewer.slicing import plane_layout

    layout = plane_layout(session.state.grid.affine, vp.plane)
    pos = session.state.crosshair[layout.fixed]
    values = extract_plane(vol, grid, session.state.layers.base.affine, vp.plane, pos).numpy()
    # z ramps along y by tan... sin(30) per mm of in-plane y, and not along x.
    col_axis_varies = np.ptp(values[15:25, 20])
    row_axis_varies = np.ptp(values[20, 15:25])
    assert max(col_axis_varies, row_axis_varies) == pytest.approx(9 * 0.5, abs=0.3)
    assert min(col_axis_varies, row_axis_varies) < 1e-3
    assert render_viewport(session, vp) is not None
    with pytest.raises(ValueError, match="rotation"):
        session.do(SetViewTilt("A1", (2.0, 0, 0, 0, 1, 0, 0, 0, 1)))
    assert "SET_VIEW_TILT" in session.to_script()


def test_t_tilts_to_the_sphere_and_outlines_and_clicks_follow_the_tilt(tmp_path):
    """On a sphere, a square cut through any point is a great circle: so after
    ``t`` every outline point is at the white radius, which only happens if
    the outlines were cut in the tilted grid -- the untilted axial slice at
    z = 15 cuts a 13 mm circle."""
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.imagewindow import ImageWindow
    from fastfuncstuff.viewer.ui.window import ViewerWindow
    from fastfuncstuff.viewer.vocab import OpenView, SetXYZ
    from tests.viewer.test_surfaces import _shell_anat, _subject

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    session = ViewerSession(device=CPU)
    win = ViewerWindow(session)
    try:
        win.open_path(str(_shell_anat(tmp_path)))
        win.load_surfaces(str(_subject(tmp_path)))
        win._dispatch(OpenView("A1", "image", "axial"))
        rho = np.sqrt(22.0**2 - 15.0**2)  # mid-ribbon, between white 20 and pial 24
        win._dispatch(SetXYZ(rho, 0.0, 15.0))
        app.processEvents()
        image = next(
            w for w in win.manager.windows.values() if isinstance(w, ImageWindow) and w.vid == "A1"
        )
        image._tilt_to_cortex()
        app.processEvents()
        grid = image._grid()
        assert grid is not None and grid.layout_affine is not None
        from fastfuncstuff.viewer.compose import plane_view

        image.redraw()
        view = plane_view(session.state, image._viewport())
        pos = image.pane.position
        whites = session.surfaces.outlines(grid.affine, view, pos, ("white",))
        pts = np.concatenate([o.segments.reshape(-1, 2) for o in whites])
        # Outline points are (row, col) image pixels; back to mm through the tilted grid.
        mm = np.array([grid.ijk_to_mm(tuple(view.image_to_points(r, c, pos))) for r, c in pts])
        np.testing.assert_allclose(np.linalg.norm(mm, axis=1), 20.0, atol=0.3)
        # A click on that outline lands in the shared grid on the sphere too.
        r, c = pts[len(pts) // 3]
        image._pick(int(round(r)), int(round(c)), seed=False)
        assert np.linalg.norm(session.state.crosshair_mm) == pytest.approx(20.0, abs=0.8)
    finally:
        win.close()
        session.close()


def test_a_stroke_drawn_in_a_tilted_slice_replays_from_its_command(tmp_path):
    """EDIT_SURFACE_STROKE names its slice by grid axis and position; in an
    oblique window that slice is in the tilted grid, so the command carries
    the grid or a replay would redraw the stretch on the wrong plane."""
    from fastfuncstuff.viewer.compose import view_grid
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.slicing import plane_layout
    from fastfuncstuff.viewer.vocab import (
        EditSurfaceStroke,
        LoadSurfaces,
        OpenView,
        SetViewTilt,
        SetXYZ,
    )
    from tests.viewer.test_surfaces import _shell_anat, _subject

    session = ViewerSession(device=CPU)
    try:
        session.load(str(_shell_anat(tmp_path)))
        session.do(LoadSurfaces(str(_subject(tmp_path))))
        session.do(OpenView("A1", "image", "axial"))
        session.do(SetXYZ(0.0, 0.0, 0.0))
        session.do(SetViewTilt("A1", _tilt_about_x(35.0)))
        vp = session.state.viewports.get("A1")
        grid = view_grid(session.state, vp)
        layout = plane_layout(session.state.grid.affine, vp.plane)
        pos = float(session.state.crosshair[layout.fixed])
        # The tilted plane through the centre: a great circle. Redraw a short
        # arc of it out at r = 21, ends on the old outline (r = 20).
        u, v = grid.affine[:3, layout.row], grid.affine[:3, layout.col]
        u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
        t = np.linspace(-0.25, 0.25, 25)
        arc = 21.0 * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)
        ends = (
            20.0 * np.stack([np.cos(t[[0, -1]])[:, None] * u + np.sin(t[[0, -1]])[:, None] * v])[0]
        )
        line = np.concatenate([ends[:1], arc[1:-1], ends[1:]])
        r, _, m, q, e = session.state.surface_brush
        cmd = EditSurfaceStroke(
            "lh", "white", int(layout.fixed), pos, EditSurfaceStroke.encode(line),
            r, 0.0, m, q, e, "", True, EditSurfaceStroke.encode_grid(grid.affine),
        )  # fmt: skip
        session.do(cmd)
        white = session.surfaces.hemis["lh"].states["white"]
        mid = np.argmin(np.linalg.norm(white - 20.0 * u, axis=1))
        assert np.linalg.norm(white[mid]) == pytest.approx(21.0, abs=0.25)
        np.testing.assert_allclose(cmd.grid_affine(), grid.affine, atol=1e-6)
    finally:
        session.close()


def test_alt_arrows_tilt_by_hand_and_compose(tmp_path, session):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from fastfuncstuff.viewer.compose import tilt_matrix
    from fastfuncstuff.viewer.ui.imagewindow import ImageWindow
    from fastfuncstuff.viewer.vocab import OpenView

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    session.do(OpenView("A1", "image", "axial"))
    win = ImageWindow("A1", session, session.do)
    try:
        for _ in range(3):
            win._tilt_by(0, 5.0)
        r = tilt_matrix(session.state.viewports.get("A1"))
        assert np.degrees(np.arccos((np.trace(r) - 1) / 2)) == pytest.approx(15.0)
        win._tilt_by(1, 5.0)
        win._untilt()
        np.testing.assert_allclose(tilt_matrix(session.state.viewports.get("A1")), np.eye(3))
    finally:
        win.close()
        app.processEvents()
