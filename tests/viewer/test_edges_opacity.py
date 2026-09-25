"""Edges and opacity: the two quick "are these aligned" looks.

Both are display settings on a layer, so they are tested where a regression
would show: in the rendered pixels and in what the image window's keys do.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.viewer.compose import render_plane
from fastfuncstuff.viewer.session import ViewerSession
from fastfuncstuff.viewer.state import Plane
from fastfuncstuff.viewer.vocab import Load, SetEdges, SetIJK, SetLayerVisible

nib = pytest.importorskip("nibabel")
CPU = torch.device("cpu")


def _ball(tmp_path, name, shape, vox, radius_mm, fov_shift=0.0):
    aff = np.diag([vox, vox, vox, 1.0])
    aff[:3, 3] = [-(n - 1) * vox / 2 + fov_shift for n in shape]
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1)
    mm = ijk * vox + aff[:3, 3]
    data = (np.linalg.norm(mm, axis=-1) < radius_mm).astype(np.float32) * 100 + 5
    p = tmp_path / name
    nib.save(nib.Nifti1Image(data, aff), str(p))
    return p


@pytest.fixture
def session():
    s = ViewerSession(device=CPU)
    yield s
    s.close()


def _lit(img):
    return img.rgba[..., :3].sum(-1) > 0


def test_edges_draw_a_thin_outline_not_the_filled_image(session, tmp_path):
    key = session.load(_ball(tmp_path, "a.nii.gz", (40, 40, 30), 2.0, 24.0))
    filled = _lit(render_plane(session, Plane.AXIAL)).sum()
    session.do(SetEdges(key, True))
    outline = _lit(render_plane(session, Plane.AXIAL)).sum()
    # A 12-voxel-radius disc is ~450 pixels; its boundary is ~75.
    circumference = 2 * np.pi * 12
    assert outline < 2 * circumference
    assert outline > 0.5 * circumference
    assert filled > 4 * outline


def test_edges_skip_the_boundary_of_a_smaller_field_of_view(session, tmp_path):
    """An EPI slab over an anat: the slab's edge is not anatomy.

    Outside the slab samples as zero, so the detector sees a step there. Only
    the ball should be outlined, however far the slab stops short of the grid.
    """
    session.load(_ball(tmp_path, "grid.nii.gz", (60, 60, 40), 1.0, 5.0))
    slab = session.load(_ball(tmp_path, "slab.nii.gz", (20, 20, 12), 2.0, 12.0))
    session.do(SetEdges(slab, True))
    grid = session.state.grid
    centre = grid.clamp(tuple(int(round(v)) for v in grid.mm_to_ijk((0.0, 0.0, 0.0))))
    session.do(SetIJK(*centre))
    session.do(SetLayerVisible(session.state.layers[0].key, False))
    lit = _lit(render_plane(session, Plane.AXIAL)).numpy()
    rows, cols = np.nonzero(lit)
    # The slab spans +-19 mm around the centre; the ball, 12 mm. Nothing lit
    # beyond the ball's radius plus the smoothing.
    mid = (np.array(lit.shape) - 1) / 2.0
    reach = np.hypot(rows - mid[0], cols - mid[1]).max()
    assert reach < 15.0


def test_edges_are_recorded(session, tmp_path):
    key = session.load(_ball(tmp_path, "a.nii.gz", (20, 20, 20), 2.0, 10.0))
    session.do(SetEdges(key, True))
    assert f"SET_EDGES {key} 1" in session.to_script()


# ---------------------------------------------------------------------------
# the image window's keys
# ---------------------------------------------------------------------------


@pytest.fixture
def win(tmp_path):
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from fastfuncstuff.viewer.ui.window import ViewerWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    session = ViewerSession(device=CPU)
    w = ViewerWindow(session)
    w.refresh(session.do(Load(str(_ball(tmp_path, "a.nii.gz", (20, 20, 20), 2.0, 10.0)))))
    w.show()
    app.processEvents()
    yield w, app
    w.close()


def _image_window(w):
    vp = w.session.state.viewports.images[0]
    return w.manager.windows[vp.id]


def test_e_in_an_image_window_toggles_edges_on_the_selected_layer(win):
    w, _ = win
    iw = _image_window(w)
    layer = w.session.state.selected_layer()
    iw._toggle_edges()
    assert w.session.state.layers.get(layer.key).edges
    assert w.edges_check.isChecked()
    iw._toggle_edges()
    assert not w.session.state.layers.get(layer.key).edges


def test_6_opens_the_opacity_slider_and_it_drives_the_layer(win):
    w, app = win
    iw = _image_window(w)
    key = w.session.state.selected_layer().key
    iw._toggle_opacity()
    app.processEvents()
    assert iw.opacity_bar.isVisible()
    # Opened on an opaque layer, the key alone makes the change visible.
    assert w.session.state.layers.get(key).opacity == pytest.approx(0.5)
    iw.opacity_slider.setValue(20)
    assert w.session.state.layers.get(key).opacity == pytest.approx(0.2)
    # And the controller's own slider follows, rather than showing 50%.
    assert w.opacity_slider.value() == 20
    iw._toggle_opacity()
    assert not iw.opacity_bar.isVisible()
    assert w.session.state.layers.get(key).opacity == pytest.approx(0.2)
