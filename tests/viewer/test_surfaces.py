"""Surface outlines in the viewer: right place, right pixels, script-replayable."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.viewer.session import ViewerSession
from fastfuncstuff.viewer.slicing import PlaneView, plane_layout
from fastfuncstuff.viewer.state import Plane
from fastfuncstuff.viewer.vocab import LoadSurfaces, ShowSurfaces

nib = pytest.importorskip("nibabel")
nfs = pytest.importorskip("nibabel.freesurfer")
CPU = torch.device("cpu")
RADIUS = 20.0


def _sphere(radius: float, n: int = 3000):
    from scipy.spatial import ConvexHull

    i = np.arange(n) + 0.5
    phi, theta = np.arccos(1 - 2 * i / n), np.pi * (1 + 5**0.5) * i
    v = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
    f = ConvexHull(v).simplices.astype(np.int32)
    # Wound outward, as FreeSurfer stores every surface: the hull's own order
    # is arbitrary per face, which made vertex normals point every which way
    # and an edit of white "push" pial that was 4 mm clear of it.
    n_face = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    inward = np.einsum("ij,ij->i", n_face, v[f].mean(1)) < 0
    f[inward] = f[inward][:, ::-1]
    return (v * radius).astype(np.float32), f


def _subject(tmp_path, cras=(0.0, 0.0, 0.0)):
    surf = tmp_path / "subj" / "surf"
    surf.mkdir(parents=True)
    info = {
        "head": np.array([2, 0, 20], np.int32),
        "valid": "1  # volume info valid",
        "filename": "orig.mgz",
        "volume": np.array([256, 256, 256]),
        "voxelsize": np.array([1.0, 1.0, 1.0]),
        "xras": np.array([-1.0, 0, 0]),
        "yras": np.array([0, 0, -1.0]),
        "zras": np.array([0, 1.0, 0]),
        "cras": np.asarray(cras, float),
    }
    v, f = _sphere(RADIUS)
    nfs.write_geometry(str(surf / "lh.white"), v, f, volume_info=info)
    nfs.write_geometry(str(surf / "lh.pial"), v * 1.2, f, volume_info=info)
    return tmp_path / "subj"


def _anat(tmp_path, vox=2.0, shape=(40, 40, 30)):
    aff = np.diag([vox, vox, vox, 1.0])
    aff[:3, 3] = [-(n - 1) * vox / 2 for n in shape]
    p = tmp_path / "anat.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(shape, np.float32), aff), str(p))
    return p


@pytest.fixture
def session():
    s = ViewerSession(device=CPU)
    yield s
    s.close()


def test_points_to_image_agrees_with_to_image_under_flips_and_crop():
    # LPS-ish affine: two axes flipped relative to RAS.
    aff = np.diag([-2.0, -2.0, 2.0, 1.0])
    layout = plane_layout(aff, Plane.CORONAL)
    view = PlaneView(layout=layout, shape=(30, 40, 20), zoom=2.0, pan=(3.0, -2.0))
    rng = np.random.default_rng(0)
    for ijk in rng.integers(0, [30, 40, 20], size=(20, 3)):
        frac = view.points_to_image(ijk.astype(float))
        assert tuple(frac.astype(int)) == view.to_image(tuple(int(x) for x in ijk))


def test_outlines_land_on_the_sphere_in_scanner_space(session, tmp_path):
    cras = (4.0, -6.0, 2.0)
    session.load(_anat(tmp_path))
    session.do(LoadSurfaces(str(_subject(tmp_path, cras)), "lh"))
    st = session.state
    assert st.grid is not None
    # Put the axial slice through the sphere's centre, which is c_ras.
    centre_ijk = np.linalg.inv(st.grid.affine) @ [*cras, 1.0]
    layout = plane_layout(st.grid.affine, Plane.AXIAL)
    view = PlaneView(layout=layout, shape=st.grid.shape)
    pos = int(round(centre_ijk[layout.fixed]))
    out = session.surfaces.outlines(st.grid.affine, view, pos, st.surfaces_shown)
    by_name = {o.surface: o for o in out}
    assert set(by_name) == {"white", "pial"}
    centre_img = view.points_to_image(centre_ijk[:3])
    off_plane = (pos - centre_ijk[layout.fixed]) * 2.0
    for name, scale in [("white", 1.0), ("pial", 1.2)]:
        r_vox = np.linalg.norm(by_name[name].segments - centre_img, axis=-1)
        expected = np.sqrt((RADIUS * scale) ** 2 - off_plane**2) / 2.0  # 2 mm voxels
        assert abs(np.median(r_vox) - expected) < 0.05 * expected


def test_show_surfaces_is_a_replayable_command(session, tmp_path):
    session.load(_anat(tmp_path))
    session.do(LoadSurfaces(str(_subject(tmp_path)), "lh"))
    session.do(ShowSurfaces("pial"))
    assert session.state.surfaces_shown == ("pial",)
    st = session.state
    assert st.grid is not None
    view = PlaneView(layout=plane_layout(st.grid.affine, Plane.AXIAL), shape=st.grid.shape)
    out = session.surfaces.outlines(st.grid.affine, view, st.grid.shape[2] // 2, ("pial",))
    assert [o.surface for o in out] == ["pial"]
    session.do(ShowSurfaces(""))
    assert session.state.surfaces_shown == ()
    assert ShowSurfaces("white,pial").to_line() == "SHOW_SURFACES white,pial"


def _shell_anat(tmp_path, cras=(0.0, 0.0, 0.0)):
    """T1-like phantom: WM inside r=21 mm, GM to 24, CSF beyond, centred on c_ras."""
    vox, n = 0.5, 120
    aff = np.diag([vox, vox, vox, 1.0])
    aff[:3, 3] = -(n - 1) * vox / 2 + np.asarray(cras)
    ijk = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1)
    r = np.linalg.norm(ijk * vox + aff[:3, 3] - np.asarray(cras), axis=-1)
    img = np.where(r < 21, 110.0, np.where(r < 24, 70.0, 20.0)).astype(np.float32)
    p = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(img, aff), str(p))
    return p


def _edit(session, drag=(0.0, 0.0, 0.6)):
    from fastfuncstuff.viewer.vocab import EditSurface

    white = session.surfaces.hemis["lh"].states["white"]
    top = int(np.argmax(white[:, 2]))
    r, s, m, q, e = session.state.surface_brush
    return EditSurface("lh", "white", top, tuple(white[top].tolist()), drag, r, s, m, q, e)


def test_edit_undo_save_and_replay(session, tmp_path):
    from fastfuncstuff.io.freesurfer import read_surface
    from fastfuncstuff.viewer.vocab import SaveSurfaces, UndoSurfaceEdit

    session.load(_shell_anat(tmp_path))
    subj = _subject(tmp_path)
    session.do(LoadSurfaces(str(subj), "lh"))
    hemi = session.surfaces.hemis["lh"]
    before = hemi.states["white"].copy()

    session.do(_edit(session))
    after = hemi.states["white"].copy()
    moved = np.flatnonzero(np.any(after != before, axis=1))
    assert moved.size > 0
    # The sphere's white is at 20 mm; the phantom's boundary is at 21.
    assert np.linalg.norm(after[moved], axis=1).max() == pytest.approx(21.0, abs=0.2)

    session.do(UndoSurfaceEdit())
    np.testing.assert_array_equal(hemi.states["white"], before)

    session.do(_edit(session))
    session.do(SaveSurfaces("test"))
    saved = read_surface(subj / "surf" / "lh.white.test")
    assert (subj / "surf" / "surface_edits.test.json").exists()
    m = hemi.tkr_to_scanner
    np.testing.assert_allclose(
        saved.vertices @ m[:3, :3].T + m[:3, 3], hemi.states["white"], atol=1e-4
    )

    # Replay the recorded session from scratch: the gesture, not the result,
    # is recorded, and it must rebuild the same surface.
    script = session.to_script()
    assert "EDIT_SURFACE" in script and "UNDO_SURFACE_EDIT" in script
    fresh = ViewerSession(device=CPU)
    try:
        fresh.run_script(script)
        np.testing.assert_allclose(
            fresh.surfaces.hemis["lh"].states["white"], hemi.states["white"], atol=1e-6
        )
    finally:
        fresh.close()


def test_grab_takes_the_outline_under_the_press(session, tmp_path):
    session.load(_shell_anat(tmp_path))
    session.do(LoadSurfaces(str(_subject(tmp_path)), "lh"))
    st = session.state
    assert st.grid is not None
    layout = plane_layout(st.grid.affine, Plane.AXIAL)
    view = PlaneView(layout=layout, shape=st.grid.shape)
    pos = st.grid.shape[layout.fixed] // 2
    out = {
        o.surface: o
        for o in session.surfaces.outlines(st.grid.affine, view, pos, ("white", "pial"))
    }
    for name in ("white", "pial"):
        row, col = out[name].segments[0, 0]
        grab = session.surfaces.grab(st.grid.affine, view, pos, ("white", "pial"), row, col, 3.0)
        assert grab is not None and grab.surface == name
    # Far from both outlines: nothing to grab.
    assert (
        session.surfaces.grab(st.grid.affine, view, pos, ("white", "pial"), 1.0, 1.0, 3.0) is None
    )


def test_drag_gesture_previews_live_and_records_one_edit(tmp_path):
    """Press, drag, release through the real window: one EDIT_SURFACE, outlines live."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from fastfuncstuff.viewer.compose import plane_view
    from fastfuncstuff.viewer.ui.imagewindow import ImageWindow
    from fastfuncstuff.viewer.ui.window import ViewerWindow
    from fastfuncstuff.viewer.vocab import OpenView, SetSurfaceEditing

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    session = ViewerSession(device=CPU)
    win = ViewerWindow(session)
    try:
        win.open_path(str(_shell_anat(tmp_path)))
        win.load_surfaces(str(_subject(tmp_path)))
        win._dispatch(OpenView("S1", "image", "coronal"))
        win._dispatch(SetSurfaceEditing(True))
        app.processEvents()
        image = next(
            w for w in win.manager.windows.values() if isinstance(w, ImageWindow) and w.vid == "S1"
        )
        st = session.state
        vp = image._viewport()
        assert st.grid is not None and vp is not None
        view = plane_view(st, vp)
        pos = image.pane.position
        assert view is not None and pos is not None
        pial = {
            o.surface: o
            for o in session.surfaces.outlines(st.grid.affine, view, pos, ("white", "pial"))
        }["pial"].segments
        row, col = pial[0, 0]
        n_before = len(session.to_script().splitlines())

        image.pane.edit_pressed.emit(float(row), float(col))
        assert session.surfaces.editing is not None
        before = image.pane._outlines[("lh", "pial")][1].boundingRect()
        for step in (1.0, 2.0, 3.0):
            image.pane.edit_dragged.emit(float(row), float(col) + step)
        app.processEvents()
        assert image.pane._outlines[("lh", "pial")][1].boundingRect() != before
        image.pane.edit_released.emit()
        app.processEvents()

        lines = session.to_script().splitlines()[n_before:]
        assert sum(line.startswith("EDIT_SURFACE") for line in lines) == 1
        assert session.surfaces.editing is None

        # A press nowhere near an outline still means "look here".
        cross = st.crosshair
        image.pane.edit_pressed.emit(2.0, 2.0)
        assert session.surfaces.editing is None
        assert st.crosshair != cross
    finally:
        win.close()


def test_region_lines_name_the_surface_parcel_and_the_atlas_voxel(session, tmp_path):
    from fastfuncstuff.viewer.vocab import SetAtlas, SetXYZ

    subj = _subject(tmp_path)
    (subj / "label").mkdir()
    (subj / "mri").mkdir()
    white = nfs.read_geometry(str(subj / "surf" / "lh.white"))[0]
    labels = np.where(white[:, 2] > 0, 1, 2)  # top half "precentral", bottom "insula"
    ctab = np.array([[25, 5, 25, 0, 0], [60, 20, 220, 0, 0], [255, 192, 32, 0, 0]], np.int32)
    nfs.write_annot(
        str(subj / "label" / "lh.aparc.annot"), labels, ctab, ["unknown", "precentral", "insula"]
    )
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    aff[:3, 3] = -40.0
    atlas = np.zeros((41, 41, 41), np.int32)
    atlas[:, :, 20:] = 17  # upper half: Left-Hippocampus, as far as the LUT is concerned
    nib.save(nib.MGHImage(atlas, aff), str(subj / "mri" / "aparc+aseg.mgz"))
    lut = tmp_path / "lut.txt"
    lut.write_text("0 Unknown 0 0 0 0\n17 Left-Hippocampus 220 216 20 0\n")

    session.load(_anat(tmp_path))
    session.do(LoadSurfaces(str(subj), "lh"))
    session.surfaces._lut = read_color_lut(lut)
    top = 22.0  # mid-thickness of the r=20/24 sphere pair, at the top
    session.do(SetXYZ(0.0, 0.0, top))
    lines = session.surfaces.region_lines(session.state.crosshair_mm, "aparc", "aparc+aseg")
    assert any("precentral" in ln and "(aparc)" in ln for ln in lines)
    assert any("Left-Hippocampus" in ln for ln in lines)
    assert any("precentral" in ln for ln in session.overlay_readout())
    # Far from cortex, no surface name; switching parcellation off drops the line.
    assert not any(
        "(aparc)" in ln for ln in session.surfaces.region_lines((0.0, 0.0, 0.0), "aparc", "")
    )
    session.do(SetAtlas("", "aparc+aseg"))
    assert not any("(aparc)" in ln for ln in session.overlay_readout())


from fastfuncstuff.io.freesurfer import read_color_lut  # noqa: E402


def test_saving_an_edited_white_also_writes_smoothwm_with_its_displacement(session, tmp_path):
    from fastfuncstuff.io.freesurfer import read_surface
    from fastfuncstuff.viewer.vocab import SaveSurfaces

    session.load(_shell_anat(tmp_path))
    subj = _subject(tmp_path)
    v, f = (
        read_surface(subj / "surf" / "lh.white").vertices,
        read_surface(subj / "surf" / "lh.white").faces,
    )
    smooth = v * 0.99  # a stand-in smoothwm, same mesh
    info = nfs.read_geometry(str(subj / "surf" / "lh.white"), read_metadata=True)[2]
    nfs.write_geometry(str(subj / "surf" / "lh.smoothwm"), smooth, f, volume_info=info)
    session.do(LoadSurfaces(str(subj), "lh"))
    session.do(_edit(session))
    session.do(SaveSurfaces("t"))
    white_before = read_surface(subj / "surf" / "lh.white").vertices
    white_after = read_surface(subj / "surf" / "lh.white.t").vertices
    sm_before = read_surface(subj / "surf" / "lh.smoothwm").vertices
    sm_after = read_surface(subj / "surf" / "lh.smoothwm.t").vertices
    moved = np.any(white_after != white_before, axis=1)
    assert moved.any() and not moved.all()
    np.testing.assert_allclose(sm_after - sm_before, white_after - white_before, atol=1e-4)
    # Untouched vertices bit-identical in the smoothwm copy too.
    np.testing.assert_array_equal(sm_after[~moved], sm_before[~moved])


def test_install_replaces_originals_and_keeps_backups(session, tmp_path):
    from fastfuncstuff.io.freesurfer import read_surface

    session.load(_shell_anat(tmp_path))
    subj = _subject(tmp_path)
    surf = subj / "surf"
    v, f = read_surface(surf / "lh.white").vertices, read_surface(surf / "lh.white").faces
    info = nfs.read_geometry(str(surf / "lh.white"), read_metadata=True)[2]
    nfs.write_geometry(str(surf / "lh.smoothwm"), v * 0.99, f, volume_info=info)
    originals = {n: (surf / n).read_bytes() for n in ("lh.white", "lh.pial", "lh.smoothwm")}
    session.do(LoadSurfaces(str(subj), "lh"))
    store = session.surfaces
    with pytest.raises(ValueError, match="no edited"):
        store.install()
    session.do(_edit(session))
    edited = store.hemis["lh"].states["white"].copy()
    plan = store.install_plan(stamp="T")
    # white was edited (and pushed nothing): white and smoothwm, not pial.
    assert [o.name for o, _ in plan.files] == ["lh.white", "lh.smoothwm"]
    store.install(plan)
    for original, backup in plan.files:
        assert backup.read_bytes() == originals[original.name]
        assert original.read_bytes() != originals[original.name]
    assert (surf / "lh.pial").read_bytes() == originals["lh.pial"]
    hemi = store.hemis["lh"]
    np.testing.assert_allclose(hemi.original("white"), edited, atol=1e-4)
    assert not store.edited and not list(surf.glob(".*ffsedit-tmp"))
    assert (surf / "surface_edits.installed-T.json").exists()
    # A second install the same second does not clobber the first backups.
    session.do(_edit(session, drag=(0.0, 0.0, -0.4)))
    again = store.install_plan(stamp="T")
    assert all(b.name.endswith("-2") for _, b in again.files)
    # And installing is not something a replayed script can do.
    assert "INSTALL" not in session.to_script()
