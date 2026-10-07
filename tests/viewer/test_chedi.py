"""CHEDI: cortex laid flat around the crosshair, the anatomy sampled at a depth."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

nib = pytest.importorskip("nibabel")
nfs = pytest.importorskip("nibabel.freesurfer")

from tests.viewer.test_surfaces import RADIUS, _shell_anat, _sphere, _subject  # noqa: E402

CPU = torch.device("cpu")
#: The phantom: WM inside 21 mm, GM to 24, CSF beyond; white at 20, pial at 24.
WM, GM, CSF = 110.25, 70.25, 20.25


def _with_shape(subject, name, scale):
    """Add ``lh.<name>`` (sphere or inflated) to a test subject."""
    info = nfs.read_geometry(str(subject / "surf" / "lh.white"), read_metadata=True)[2]
    v, f = _sphere(RADIUS)
    nfs.write_geometry(str(subject / "surf" / f"lh.{name}"), v * scale, f, volume_info=info)
    return subject


@pytest.fixture
def subject(tmp_path):
    return _with_shape(_subject(tmp_path), "sphere", 5.0)


def _hemi(subject):
    from fastfuncstuff.io.freesurfer import load_subject

    return load_subject(subject, ("lh",))["lh"]


def _sampler(tmp_path, mode="linear"):
    from fastfuncstuff.surface.sampling import VolumeSampler

    img = nib.load(str(_shell_anat(tmp_path)))
    return VolumeSampler(img.get_fdata(), img.affine, mode)


@pytest.mark.parametrize("shape", ["sphere", "inflated"])
def test_depth_walks_from_white_matter_through_grey_into_csf(tmp_path, shape):
    from fastfuncstuff.viewer.chedi import PatchSampler, build_patch

    subject = _with_shape(_subject(tmp_path), shape, 5.0 if shape == "sphere" else 1.5)
    h = _hemi(subject)
    patch = build_patch(h, 0, half_mm=6.0, size=48)
    assert patch.source == shape
    assert patch.inside.mean() > 0.95
    sampler = PatchSampler(patch)
    volume = _sampler(tmp_path)
    got = [float(np.nanmedian(sampler.sample(h, d, volume, 1))) for d in (0.0, 0.5, 1.5)]
    assert got == pytest.approx([WM, GM, CSF])


def test_flat_millimetres_are_roughly_cortical_millimetres(tmp_path, subject):
    """Not area-true, but a pixel 5 mm from the centre should be ~5 mm away on the cortex."""
    from fastfuncstuff.viewer.chedi import PatchSampler, build_patch

    h = _hemi(subject)
    patch = build_patch(h, 0, half_mm=10.0, size=64)
    pts = PatchSampler(patch).points(h, 0.5)
    mid = 0.5 * (h.states["white"][0] + h.states["pial"][0])
    flat = patch.to_pixels(np.array([5.0, 0.0]))
    k = int(np.count_nonzero(patch.inside.ravel()[: int(flat[0]) * 64 + int(flat[1])]))
    assert np.linalg.norm(pts[k] - mid) == pytest.approx(5.0, rel=0.25)
    # The centre vertex sits in the middle of the image.
    np.testing.assert_allclose(patch.to_pixels(patch.uv[patch.ids == 0][0]), (32, 32), atol=1.0)


def test_every_sampling_mode_reads_a_voxel_centre_as_that_voxel():
    from fastfuncstuff.surface.sampling import VolumeSampler

    data = np.random.default_rng(0).random((12, 12, 12)).astype(np.float32)
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    s = VolumeSampler(data, aff)
    centre = np.array([[6.0, 8.0, 10.0]])  # voxel (3, 4, 5)
    between = np.array([[7.0, 9.0, 11.0]])
    for mode in ("nearest", "linear", "cubic"):
        assert s.with_mode(mode)(centre)[0] == pytest.approx(data[3, 4, 5])
    # Between voxels, nearest returns a voxel and the others interpolate.
    assert s.with_mode("nearest")(between)[0] in data
    assert s.with_mode("linear")(between)[0] != s.with_mode("cubic")(between)[0]
    assert s.with_mode("cubic").data is s.data  # a mode shares the voxels
    with pytest.raises(ValueError, match="sampling"):
        s.with_mode("sinc")


def test_window_follows_the_crosshair_samples_the_overlay_and_steps_depth(tmp_path, subject):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.chediwindow import ChediWindow
    from fastfuncstuff.viewer.vocab import Load, LoadSurfaces, OpenView, SelectLayer, SetXYZ

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    session = ViewerSession(device=CPU)
    anat = _shell_anat(tmp_path)
    session.load(str(anat))
    session.do(LoadSurfaces(str(subject), "lh"))
    session.do(OpenView("E1", "chedi", "axial"))
    win = ChediWindow("E1", session, session.do)

    def refresh(cmd=None):
        if cmd is not None:
            session.do(cmd)
        win.refresh()

    win._dispatch = refresh
    try:
        refresh(SetXYZ(0.0, 0.0, 22.0))  # mid-ribbon at the top
        assert win.patch is not None
        top = win.patch.centre
        assert session.surfaces.hemis["lh"].states["white"][top][2] > 19.0
        # Off cortex (the middle of the white matter): the patch stays put.
        refresh(SetXYZ(0.0, 0.0, 0.0))
        assert win.patch.centre == top
        # Onto cortex at the side: it follows.
        refresh(SetXYZ(22.0, 0.0, 0.0))
        assert win.patch.centre != top

        # Depth keys go through the bus: 0.5 -> 1.0 is grey -> the pial edge.
        win._depth_by(0.5)
        assert session.state.viewports.get("E1").depth == (1.0, 1.0)
        assert "SET_SURFACE_DEPTH E1" in session.to_script()

        # A second layer on top is what it samples, by name in the caption.
        import shutil

        shutil.copy(anat, tmp_path / "t2ish.nii.gz")
        session.do(Load(str(tmp_path / "t2ish.nii.gz"), "T2ish"))
        refresh()
        assert "t2ish" in win.canvas.caption
        assert win.layer().key == "T2ish"
        session.do(SelectLayer(session.state.layers.base.key))
        assert win.layer().key == "T2ish"  # selecting the base is not choosing it

        win._cycle_sampling()
        assert session.state.viewports.get("E1").sampling == "linear"
        assert "· linear" in win.canvas.caption
        assert "SET_VIEW_SAMPLING E1 linear" in session.to_script()

        # Double-click: the crosshair goes to that cortex, at the depth shown.
        win._set_depth(0.5)
        win._locate(10.0, 32.0)
        r = np.linalg.norm(session.state.crosshair_mm)
        assert 20.5 < r < 23.5
        app.processEvents()
    finally:
        win.close()
        session.close()


def test_a_depth_step_redraws_no_slice_but_an_image_setting_does(tmp_path, subject):
    """VIEWPORTS used to redraw every image window, so a 7 ms depth step cost 65."""
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.imagewindow import ImageWindow
    from fastfuncstuff.viewer.ui.window import ViewerWindow
    from fastfuncstuff.viewer.vocab import OpenView, SetSurfaceDepth, SetZoom

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    session = ViewerSession(device=CPU)
    win = ViewerWindow(session)
    try:
        win.open_path(str(_shell_anat(tmp_path)))
        win.load_surfaces(str(subject))
        win._dispatch(OpenView("A1", "image", "axial"))
        win._dispatch(OpenView("E1", "chedi", "axial"))
        app.processEvents()
        image = next(w for w in win.manager.windows.values() if isinstance(w, ImageWindow))
        calls = []
        image.redraw = lambda: calls.append(1)  # type: ignore[method-assign]
        win._dispatch(SetSurfaceDepth("E1", 0.8, 0.8))
        assert calls == []
        win._dispatch(SetZoom("A1", 2.0))
        assert calls == [1]
    finally:
        win.close()
        session.close()


def test_selection_morphology_on_a_mesh(subject):
    from fastfuncstuff.viewer.chedi import adjacency, dilate, drop_isolated, erode, window_select

    h = _hemi(subject)
    adj = adjacency(h.faces, h.n_vertices)
    everywhere = np.ones(h.n_vertices, bool)
    one = np.zeros(h.n_vertices, bool)
    one[0] = True
    ring = dilate(one, adj, everywhere)
    neighbours = set(np.flatnonzero(ring)) - {0}
    assert 4 <= len(neighbours) <= 8 and ring[0]
    # Eroding the 1-ring gives back its centre; a lone point is a speck.
    np.testing.assert_array_equal(erode(ring, adj, everywhere), one)
    assert not drop_isolated(one, adj).any()
    assert drop_isolated(ring, adj).sum() == ring.sum()
    # Dilation never reaches past what is visible, and the screen's edge is not
    # a selection edge for erosion.
    visible = ring.copy()
    assert dilate(ring, adj, visible).sum() == ring.sum()
    assert erode(ring, adj, visible).sum() == ring.sum()
    np.testing.assert_array_equal(
        window_select(np.array([1.0, 5.0, 9.0]), 5.0, 8.0), [True, True, True]
    )
    np.testing.assert_array_equal(
        window_select(np.array([1.0, 5.0, 9.0]), 5.0, 2.0), [False, True, False]
    )


@pytest.fixture
def chedi(tmp_path, subject):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.chediwindow import ChediWindow
    from fastfuncstuff.viewer.vocab import LoadSurfaces, OpenView, SetPatchSize, SetXYZ

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    session = ViewerSession(device=CPU)
    session.load(str(_shell_anat(tmp_path)))
    session.do(LoadSurfaces(str(subject), "lh"))
    session.do(OpenView("E1", "chedi", "axial"))
    session.do(SetPatchSize("E1", 6.0))
    win = ChediWindow("E1", session, session.do)

    def dispatch(cmd):
        session.do(cmd)
        win.refresh()

    win._dispatch = dispatch
    dispatch(SetXYZ(0.0, 0.0, 22.0))
    yield session, win
    win.close()
    session.close()


def test_push_moves_only_the_selected_vertices_on_screen(chedi):
    """Logan's rule: what is selected but scrolled off the patch stays put."""
    from fastfuncstuff.viewer.vocab import HighlightSurface, encode_ids

    session, win = chedi
    h = session.surfaces.hemis["lh"]
    assert win._vis.size > 10
    # Select everything on screen with a wide value window...
    win._press("window", 30.0, 30.0)
    win._drag(30.0, 30.0, 10 * 300.0, 0.0)
    win._release()
    on_screen = session.surfaces.highlighted("lh")
    assert set(on_screen) == set(win._vis)
    # ...and some cortex on the far side of the sphere too.
    far = np.flatnonzero(h.states["white"][:, 2] < -15.0)[:20]
    session.do(HighlightSurface("lh", encode_ids(far), "add"))
    before = h.states["pial"].copy()
    step = session.state.surface_step
    win._push(-1.0)  # W: pial in
    moved = np.linalg.norm(h.states["pial"] - before, axis=1)
    assert np.all(moved[far] == 0.0)
    r_before = np.linalg.norm(before[win._vis], axis=1)
    r_after = np.linalg.norm(h.states["pial"][win._vis], axis=1)
    assert np.median(r_before - r_after) == pytest.approx(step, rel=0.05)
    # White never moved: one surface at a time.
    assert "MOVE_SURFACE_HIGHLIGHT lh pial" in session.to_script()
    win._toggle_surface()
    assert win.surface == "white"


def test_brush_paint_morph_invert_and_escape(chedi):
    session, win = chedi
    centre = win.patch.size / 2.0
    win._press("add", centre, centre)
    win._drag(centre, centre + 4.0, 0.0, 0.0)
    win._release()
    painted = session.surfaces.highlighted("lh")
    assert 0 < painted.size < win._vis.size
    assert set(painted) <= set(win._vis)
    on = lambda: int(session.surfaces.highlight["lh"][win._vis].sum())  # noqa: E731
    n0 = on()
    win._morph("dilate")
    assert on() > n0
    win._morph("erode")
    assert on() <= n0 + 2
    before = on()
    win._morph("invert")
    assert on() == win._vis.size - before
    win._morph("invert")
    assert on() == before
    win._press("remove", centre, centre)
    win._release()
    win._clear_selection()
    assert session.surfaces.highlighted("lh").size == 0
    assert "HIGHLIGHT_SURFACE" in session.to_script()


def test_shift_click_moves_the_crosshair_and_centres_a_zoomed_slice(tmp_path, subject):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from PySide6 import QtCore
    from PySide6.QtTest import QTest

    from fastfuncstuff.viewer.compose import plane_view
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.chediwindow import ChediWindow
    from fastfuncstuff.viewer.ui.window import ViewerWindow
    from fastfuncstuff.viewer.vocab import OpenView, SetPan, SetXYZ, SetZoom

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    session = ViewerSession(device=CPU)
    win = ViewerWindow(session)
    try:
        win.open_path(str(_shell_anat(tmp_path)))
        win.load_surfaces(str(subject))
        win._dispatch(OpenView("A1", "image", "coronal"))
        win._dispatch(SetZoom("A1", 4.0))
        win._dispatch(SetPan("A1", -40.0, -40.0))  # far off in a corner
        win._dispatch(SetXYZ(0.0, 0.0, 22.0))
        win._dispatch(OpenView("E1", "chedi", "axial"))
        app.processEvents()
        chedi = next(w for w in win.manager.windows.values() if isinstance(w, ChediWindow))
        chedi.resize(300, 320)
        chedi.show()
        app.processEvents()
        before = session.state.crosshair_mm
        selected = session.surfaces.highlighted("lh").size
        canvas = chedi.canvas
        rect = canvas.target()
        at = QtCore.QPoint(int(rect.center().x() + rect.width() / 4), int(rect.center().y()))
        QTest.mouseClick(
            canvas, QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.ShiftModifier, at
        )
        app.processEvents()
        after = session.state.crosshair_mm
        assert np.linalg.norm(np.subtract(after, before)) > 2.0  # it moved...
        assert 20.0 < np.linalg.norm(after) < 24.5  # ...onto the cortex clicked
        assert session.surfaces.highlighted("lh").size == selected  # and selected nothing
        vp = session.state.viewports.get("A1")
        view = plane_view(session.state, vp)
        row, col = view.to_image(session.state.crosshair)
        h, w = view.span
        assert abs(row - h / 2) <= 2 and abs(col - w / 2) <= 2
    finally:
        win.close()
        session.close()


def test_o_shows_gyri_and_sulci_and_f_g_unselect_them(chedi, subject):
    session, win = chedi
    h = session.surfaces.hemis["lh"]
    # No ?h.curv yet: it says so and goes on to the spacing map.
    win._cycle_display()
    assert win.display == "spacing" and "curv" in win.status.text()
    win._cycle_display()
    assert win.display == "data"
    # Half the cap a "sulcus" (+), half a "gyrus" (-), split along x.
    h.morph["curv"] = np.where(h.states["white"][:, 0] > 0, 0.2, -0.2).astype(np.float32)
    win._cycle_display()
    assert win.display == "folding" and "gyri / sulci" in win.canvas.caption
    win._cycle_display()
    assert win.display == "spacing" and "spacing" in win.canvas.caption
    win._cycle_display()
    assert win.display == "data"

    def select_all():
        win._press("window", 30.0, 30.0)
        win._drag(30.0, 30.0, 10 * 300.0, 0.0)
        win._release()

    select_all()
    win._drop_fold(1)  # f: off the sulci
    kept = session.surfaces.highlighted("lh")
    assert kept.size and np.all(h.states["white"][kept, 0] <= 0)
    select_all()
    win._drop_fold(-1)  # g: off the gyri
    kept = session.surfaces.highlighted("lh")
    assert kept.size and np.all(h.states["white"][kept, 0] > 0)
    win._drop_fold(1)
    assert session.surfaces.highlighted("lh").size == 0


def test_l_picks_the_sampled_layer_including_the_base_under_an_overlay(chedi, tmp_path):
    import shutil

    from fastfuncstuff.viewer.vocab import Load

    session, win = chedi
    base = session.state.layers.base
    shutil.copy(session.store.get(base.key).path, tmp_path / "t2.nii.gz")
    session.do(Load(str(tmp_path / "t2.nii.gz"), "T2"))
    win.refresh()
    assert win.layer().key == "T2"  # following: the overlay
    win._cycle_layer()
    assert win.layer().key == base.key  # the base, though an overlay is visible
    assert base.name in win.canvas.caption
    win._cycle_layer()
    assert win.layer().key == "T2"
    win._cycle_layer()
    assert session.state.viewports.get("E1").patch_layer == ""
    assert "SET_PATCH_LAYER E1" in session.to_script()


def test_kmeans_finds_planted_groups_repeatably_and_orders_them_by_brightness():
    from fastfuncstuff.viewer.chedi import cluster_features, kmeans, order_clusters

    rng = np.random.default_rng(3)
    means = (40.0, 10.0, 25.0)
    profile = np.concatenate([rng.normal(m, 1.0, (300, 3)) for m in means])
    truth = np.repeat([2, 0, 1], 300)  # rank of each group's brightness
    fold = np.zeros(900)
    x = cluster_features(profile, fold, np.zeros((900, 2)), 25.0, 0.0)
    assert x.shape == (900, 4)  # 3 depths + folding; no position at weight 0
    labels = order_clusters(kmeans(x, 3), profile[:, 1], 3)
    np.testing.assert_array_equal(labels, truth)
    np.testing.assert_array_equal(kmeans(x, 3), kmeans(x, 3))  # same press, same clusters
    # The profile's shape survives standardising: one scale across depths.
    rising = np.tile([0.0, 1.0, 2.0], (10, 1))
    z = cluster_features(rising, np.zeros(10), np.zeros((10, 2)), 25.0, 0.0)
    assert z[0, 0] < z[0, 1] < z[0, 2]
    assert cluster_features(rising, np.zeros(10), np.ones((10, 2)), 25.0, 1.0).shape == (10, 6)


def test_digits_toggle_a_cluster_and_clusters_follow_the_folding(chedi):
    session, win = chedi
    h = session.surfaces.hemis["lh"]
    # Uniform anatomy at any one depth, so only the folding can split it.
    h.morph["curv"] = np.where(h.states["white"][:, 0] > 0, 0.2, -0.2).astype(np.float32)
    win.spatial = 0.0
    win.k = 2
    win._toggle_clusters()
    assert win.show_clusters and len(win.canvas.legend) == 2
    assert len(win.canvas.cluster_dots) == win._vis.size
    win._pick_cluster(1)
    picked = session.surfaces.highlighted("lh")
    side = np.sign(h.states["white"][picked, 0])
    assert picked.size and np.all(side == side[0])  # one side of the fold
    win._pick_cluster(1)  # again: out
    assert session.surfaces.highlighted("lh").size == 0
    win._pick_cluster(5)
    assert "only 2 clusters" in win.status.text()
    win._k_by(20)
    assert win.k == 10
    keys = {b.keys for b in win.help._bindings}
    assert {"1", "0", "ctrl+0", "ctrl+1", "c", "x"} <= keys


def test_relax_flattens_a_spike_and_snap_mode_reaches_the_command(chedi):
    session, win = chedi
    h = session.surfaces.hemis["lh"]
    win._press("window", 30.0, 30.0)
    win._drag(30.0, 30.0, 10 * 300.0, 0.0)
    win._release()
    # Plant a spike in the middle of the selection.
    spike = int(
        win._vis[np.argmin(np.linalg.norm(win.patch.uv[np.isin(win.patch.ids, win._vis)], axis=1))]
    )
    pial = h.states["pial"]
    out = pial[spike] / np.linalg.norm(pial[spike])
    pial[spike] = pial[spike] + 1.0 * out
    session.surfaces.version["lh"] += 1
    before = pial.copy()
    win._relax()
    r = lambda k: float(np.linalg.norm(h.states["pial"][k]))  # noqa: E731
    assert r(spike) < np.linalg.norm(before[spike]) - 0.3  # the spike came down...
    others = win._vis[win._vis != spike][:20]
    assert np.allclose(
        [r(k) for k in others], np.linalg.norm(before[others], axis=1), atol=0.15
    )  # ...the rest barely moved: no plateau shift
    moves = [ln.split() for ln in session.to_script().splitlines() if ln.startswith("MOVE_SURFACE")]
    # name hemi surface shift radius within flatten snap gate key
    relax = moves[-1]
    assert relax[3] == "0.0" and relax[6] == "0.3" and relax[7] == "0.0"
    win._cycle_snap()
    assert win.snap_mode == "snap"
    win._push(-1.0)
    moves = [ln.split() for ln in session.to_script().splitlines() if ln.startswith("MOVE_SURFACE")]
    push = moves[-1]
    assert float(push[3]) < 0 and push[7] == "1.0" and push[8] == "1"
    assert push[9] == session.state.layers.base.key  # snaps to the layer shown


def test_u_evens_out_crowded_pial_and_the_spacing_map_shows_it(chedi):
    session, win = chedi
    h = session.surfaces.hemis["lh"]
    # As loaded, every vertex is mid grey: the map shows editing, not folding.
    assert np.abs(win._spacing()).max() < 1e-4 and win._spacing_vs == "file"
    # Crowd pial around the patch centre: pull the vertices near it halfway
    # in toward it, along the sphere.
    pial = h.states["pial"]
    c = pial[win.patch.centre].astype(np.float64)
    dist = np.linalg.norm(pial - c, axis=1)
    reach = np.flatnonzero(dist < 6.0)
    near = np.flatnonzero(dist < 3.0)  # the crowded core; the ring past it is stretched
    radius = np.linalg.norm(pial[reach], axis=1, keepdims=True)
    # r -> r^2/6 out to 6 mm: dense at the centre, continuous at the rim.
    squeezed = c + (dist[reach] / 6.0)[:, None] * (pial[reach] - c)
    pial[reach] = (squeezed / np.linalg.norm(squeezed, axis=1, keepdims=True) * radius).astype(
        np.float32
    )
    radius = np.linalg.norm(pial[near], axis=1, keepdims=True)
    session.surfaces.version["lh"] += 1
    crowded = win._spacing()[near]
    assert np.median(crowded) < -1.0  # a quarter of white's area, or less
    # ctrl+drag on the spacing map selects by spacing: the crowded part.
    win._cycle_display()  # no ?h.curv here: straight to the spacing map
    assert win.display == "spacing"
    lo = float(np.median(crowded))
    win._gesture = None
    win._press("window", 30.0, 30.0)
    g = win._gesture
    win._gesture = ("window", g[1], lo, 1.0, g[4], g[5])
    win._update_preview(0.0, 0.0)
    win._release()
    picked = session.surfaces.highlighted("lh")
    assert np.isin(picked, near).mean() > 0.9
    win._even()
    after = win._spacing()[near]
    assert np.median(after) > np.median(crowded) + 0.5
    # Within the surface: the radius of every vertex barely changed.
    r_after = np.linalg.norm(h.states["pial"][near], axis=1)
    assert np.allclose(r_after, radius[:, 0], atol=0.1)
    moves = [ln.split() for ln in session.to_script().splitlines() if ln.startswith("MOVE_SURFACE")]
    assert moves[-1][-1] == str(30)  # even, last on the line


def test_spacing_map_falls_back_to_white_once_the_topology_changes(chedi):
    session, win = chedi
    surfaces = session.surfaces
    assert surfaces.area_change("lh", "pial") is not None
    win._cycle_display()
    assert win.display == "spacing" and "vs file" in win.canvas.caption
    centre = win.patch.centre
    surfaces.delete_vertex("lh", int(surfaces.neighbours("lh", centre)[0]))
    # The file has one vertex more than the mesh now: no per-vertex baseline.
    assert surfaces.area_change("lh", "pial") is None
    win.refresh()
    assert win._spacing_vs == "white" and "vs white" in win.canvas.caption
