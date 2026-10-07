"""Mesh list: several white/pial meshes per hemisphere, the top-most of each in use."""

from __future__ import annotations

import numpy as np
import pytest
import torch

nib = pytest.importorskip("nibabel")
nfs = pytest.importorskip("nibabel.freesurfer")

from tests.viewer.test_surfaces import (  # noqa: E402
    RADIUS,
    _edit,
    _shell_anat,
    _sphere,
    _subject,
)

CPU = torch.device("cpu")


def test_filenames_say_hemi_and_type_when_they_can():
    from fastfuncstuff.viewer.meshlist import infer_label

    assert infer_label("lh.pial") == ("lh", "pial")
    assert infer_label("/x/surf/rh.pial.ffs") == ("rh", "pial")
    assert infer_label("rh.smoothwm") == ("rh", "white")
    assert infer_label("sub-01_lh_wm.surf") == ("lh", "white")
    assert infer_label("lh.pial.bak-20261007-120000") == ("lh", "pial")
    # "white" inside a word still counts; no hemi in the name says so.
    assert infer_label("whitesurf") == (None, "white")
    assert infer_label("mesh.surf") == (None, None)


@pytest.fixture
def session(tmp_path):
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.vocab import LoadSurfaces

    s = ViewerSession(device=CPU)
    s.load(str(_shell_anat(tmp_path)))
    s.do(LoadSurfaces(str(_subject(tmp_path)), "lh"))
    yield s
    s.close()


def _info(tmp_path):
    return nfs.read_geometry(str(tmp_path / "subj" / "surf" / "lh.white"), read_metadata=True)[2]


def _write(tmp_path, name, scale, n=3000):
    v, f = _sphere(RADIUS, n)
    path = tmp_path / "subj" / "surf" / name
    nfs.write_geometry(str(path), v * scale, f, volume_info=_info(tmp_path))
    return path


def _radius(points):
    return float(np.median(np.linalg.norm(points, axis=1)))


def _outline_radii(session):
    """Median radius (mm) of each outline on the axial slice through the centre."""
    from fastfuncstuff.viewer.compose import plane_view
    from fastfuncstuff.viewer.vocab import OpenView, SetXYZ

    st = session.state
    if st.viewports.find("A1") is None:
        session.do(OpenView("A1", "image", "axial"))
    session.do(SetXYZ(0.0, 0.0, 0.0))
    view = plane_view(st, st.viewports.get("A1"))
    pos = st.crosshair[view.layout.fixed]
    out = {}
    for o in session.surfaces.outlines(st.grid.affine, view, pos, st.surfaces_shown):
        ijk = np.array([view.image_to_points(r, c, pos) for r, c in o.segments.reshape(-1, 2)])
        mm = ijk @ st.grid.affine[:3, :3].T + st.grid.affine[:3, 3]
        out[o.surface] = _radius(mm)
    return out


def test_a_loaded_mesh_is_compared_until_used_then_swaps_with_the_one_in_use(session, tmp_path):
    from fastfuncstuff.viewer.vocab import LoadMesh, UseMesh

    surfaces = session.surfaces
    assert [(m.key, m.hemi, m.kind) for m in surfaces.meshes] == [
        ("m1", "lh", "white"),
        ("m2", "lh", "pial"),
    ]
    session.do(LoadMesh(str(_write(tmp_path, "lh.pial.alt", 1.3)), "lh", "pial"))
    alt = surfaces.mesh("m3")
    assert surfaces.meshes[-1] is alt and not surfaces.is_active(alt)
    # Loading changes nothing in use, and the comparison is drawn under its own key.
    assert _radius(surfaces.hemis["lh"].states["pial"]) == pytest.approx(24.0, rel=1e-3)
    radii = _outline_radii(session)
    assert radii["pial"] == pytest.approx(24.0, abs=0.3)
    assert radii["m3"] == pytest.approx(26.0, abs=0.3)

    session.do(UseMesh("m3"))
    assert [m.key for m in surfaces.meshes] == ["m1", "m3", "m2"]
    assert _radius(surfaces.hemis["lh"].states["pial"]) == pytest.approx(26.0, rel=1e-3)
    assert ("lh", "pial") in surfaces.edited  # differs from lh.pial: save/install write it
    radii = _outline_radii(session)
    assert radii["pial"] == pytest.approx(26.0, abs=0.3)
    assert radii["m2"] == pytest.approx(24.0, abs=0.3)

    # Back to the file's own pial: nothing is edited any more.
    session.do(UseMesh("m2"))
    assert ("lh", "pial") not in surfaces.edited
    assert "USE_MESH m2" in session.to_script()


def test_a_different_mesh_can_be_shown_but_not_used(session, tmp_path):
    from fastfuncstuff.viewer.vocab import LoadMesh, RemoveMesh, UseMesh

    session.do(LoadMesh(str(_write(tmp_path, "lh.pial.other", 1.3, n=2000)), "lh", "pial"))
    assert _outline_radii(session)["m3"] == pytest.approx(26.0, abs=0.3)
    with pytest.raises(ValueError, match="not the same mesh"):
        session.do(UseMesh("m3"))
    with pytest.raises(ValueError, match="in use"):
        session.do(RemoveMesh("m2"))
    session.do(RemoveMesh("m3"))
    assert [m.key for m in session.surfaces.meshes] == ["m1", "m2"]


def test_the_first_edit_keeps_the_surface_as_it_was_below_it(session):
    from fastfuncstuff.viewer.vocab import SetMesh, UseMesh

    surfaces = session.surfaces
    before = surfaces.hemis["lh"].states["white"].copy()
    session.do(_edit(session))
    session.do(_edit(session))  # a second edit adds no second snapshot
    keys = [m.key for m in surfaces.meshes]
    assert keys == ["m1", "m3", "m2"]
    snap = surfaces.mesh("m3")
    assert snap.kind == "white" and not snap.shown and "before edits" in snap.name
    np.testing.assert_array_equal(snap.positions, before)
    assert surfaces.mesh("m1").edited and surfaces.mesh("m1").label.endswith("*")
    # Shown, it is drawn as a comparison; used, it puts the surface back.
    session.do(SetMesh("m3", shown=1))
    assert "m3" in _outline_radii(session)
    session.do(UseMesh("m3"))
    np.testing.assert_array_equal(surfaces.hemis["lh"].states["white"], before)
    assert ("lh", "white") not in surfaces.edited


def test_relabelling_sends_a_row_last_and_never_takes_over(session, tmp_path):
    from fastfuncstuff.viewer.vocab import LoadMesh, SetMesh

    surfaces = session.surfaces
    session.do(LoadMesh(str(_write(tmp_path, "mystery.surf", 1.1)), "lh", "white"))
    session.do(LoadMesh(str(_write(tmp_path, "other.surf", 1.2)), "lh", "white"))
    session.do(SetMesh("m3", kind="pial", rgb="0.1,0.2,0.3", shown=0))
    m3 = surfaces.mesh("m3")
    assert surfaces.meshes[-1] is m3 and m3.kind == "pial" and not surfaces.is_active(m3)
    assert m3.rgb == pytest.approx((0.1, 0.2, 0.3)) and not m3.shown
    with pytest.raises(ValueError, match="in use"):
        session.do(SetMesh("m2", kind="white"))


def test_save_writes_a_copy_and_backup_writes_every_mesh_in_use(session, tmp_path):
    from fastfuncstuff.io.freesurfer import read_scanner_surface
    from fastfuncstuff.viewer.vocab import SaveMesh

    surfaces = session.surfaces
    session.do(_edit(session))
    out = tmp_path / "lh.white.mine"
    session.do(SaveMesh("m1", str(out)))
    saved, _ = read_scanner_surface(out)
    np.testing.assert_allclose(saved, surfaces.hemis["lh"].states["white"], atol=1e-4)
    with pytest.raises(FileExistsError):
        session.do(SaveMesh("m1", str(out)))
    with pytest.raises(ValueError, match="install"):
        session.do(SaveMesh("m1", str(surfaces.hemis["lh"].paths["pial"])))

    written = surfaces.backup("STAMP")
    assert sorted(p.name for p in written) == [
        "lh.pial.bak-STAMP",
        "lh.white.bak-STAMP",
        "surface_edits.bak-STAMP.json",
    ]
    bak, _ = read_scanner_surface(written[0].with_name("lh.white.bak-STAMP"))
    np.testing.assert_allclose(bak, surfaces.hemis["lh"].states["white"], atol=1e-4)


def test_the_window_lists_shows_uses_and_follows_edits(tmp_path):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.meshwindow import NAME, SHOW
    from fastfuncstuff.viewer.ui.window import ViewerWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    session = ViewerSession(device=CPU)
    win = ViewerWindow(session)
    try:
        win.open_path(str(_shell_anat(tmp_path)))
        win.load_surfaces(str(_subject(tmp_path)))
        win._open_meshes()
        meshes = win.mesh_window
        assert meshes is not None
        app.processEvents()

        def names():
            return [meshes.table.item(i, NAME).text() for i in range(meshes.table.rowCount())]

        assert names() == ["● lh.white", "● lh.pial"]
        # The filename says what it is: no questions asked.
        meshes.load_path(str(_write(tmp_path, "lh.pial.alt", 1.3)))
        app.processEvents()
        assert names()[-1] == "   lh.pial.alt"
        # Unticking goes through the bus, so it is in the script.
        box = meshes.table.cellWidget(2, SHOW).findChild(QtWidgets.QCheckBox)
        box.setChecked(False)
        app.processEvents()
        assert not session.surfaces.mesh("m3").shown
        assert "SET_MESH m3" in session.to_script()
        meshes._use(2)
        app.processEvents()
        assert names() == ["● lh.white", "● lh.pial.alt", "   lh.pial"]
        # An edit anywhere adds the before-edits row here, without asking.
        win._dispatch(_edit(session))
        app.processEvents()
        assert names()[:2] == ["● lh.white *", "   lh.white (before edits)"]
        meshes._backup()
        assert "bak-" in meshes.status.text()
    finally:
        win.close()
        session.close()


@pytest.fixture
def loose(tmp_path):
    """A white and a pial file in a plain directory: no subject around them."""
    import shutil

    subj = _subject(tmp_path)
    out = tmp_path / "loose"
    out.mkdir()
    for name in ("lh.white", "lh.pial"):
        shutil.copy(subj / "surf" / name, out / name)
    shutil.rmtree(subj)
    from fastfuncstuff.viewer.session import ViewerSession

    s = ViewerSession(device=CPU)
    s.load(str(_shell_anat(tmp_path)))
    yield s, out
    s.close()


def test_a_lone_mesh_loads_without_a_subject_and_samples_on_itself(loose):
    from fastfuncstuff.viewer.surface3d import shape_positions
    from fastfuncstuff.viewer.vocab import LoadMesh

    session, out = loose
    session.do(LoadMesh(str(out / "lh.pial"), "lh", "pial"))
    surfaces = session.surfaces
    h = surfaces.hemis["lh"]
    assert surfaces.subject is None and surfaces.stand_ins == {"lh": "white"}
    # The missing boundary is the surface itself, so the ribbon has no depth.
    assert h.states["white"] is h.states["pial"]
    assert np.allclose(shape_positions(h, "mid"), h.states["pial"])
    assert [(r.name, surfaces.is_active(r)) for r in surfaces.meshes] == [("lh.pial", True)]
    # Only the real surface is outlined: the stand-in would draw on top of it.
    radii = _outline_radii(session)
    assert set(radii) == {"pial"} and radii["pial"] == pytest.approx(RADIUS * 1.2, abs=1.0)
    # And a script rebuilds it with no subject either.
    from fastfuncstuff.viewer.session import ViewerSession

    again = ViewerSession(device=CPU)
    try:
        again.run_script(session.to_script())
        assert again.surfaces.stand_ins == {"lh": "white"}
        assert np.array_equal(again.surfaces.hemis["lh"].states["pial"], h.states["pial"])
    finally:
        again.close()


def test_the_partner_fills_the_gap_above_a_mesh_that_could_not(loose, tmp_path):
    from fastfuncstuff.viewer.vocab import LoadMesh, UseMesh

    session, out = loose
    session.do(LoadMesh(str(out / "lh.pial"), "lh", "pial"))
    surfaces = session.surfaces
    # Another mesh, labelled white: it cannot fill the gap, so it is a comparison.
    nfs.write_geometry(
        str(out / "lh.white.other"),
        *_sphere(RADIUS, 1000),
        volume_info=nfs.read_geometry(str(out / "lh.pial"), read_metadata=True)[2],
    )
    session.do(LoadMesh(str(out / "lh.white.other"), "lh", "white"))
    assert surfaces.stand_ins == {"lh": "white"}
    other = surfaces.meshes[-1]
    assert not surfaces.is_active(other)
    with pytest.raises(ValueError, match="not the same mesh"):
        session.do(UseMesh(other.key))
    session.do(LoadMesh(str(out / "lh.white"), "lh", "white"))
    h = surfaces.hemis["lh"]
    assert surfaces.stand_ins == {}
    assert h.states["white"] is not h.states["pial"]
    white = surfaces.active_mesh("lh", "white")
    assert white is not None and white.name == "lh.white"
    assert [r.name for r in surfaces.meshes] == ["lh.pial", "lh.white", "lh.white.other"]
    radii = _outline_radii(session)
    assert radii.keys() == {"white", "pial", other.key}
    assert radii["white"] == pytest.approx(RADIUS, abs=1.0)
    assert radii["pial"] == pytest.approx(RADIUS * 1.2, abs=1.0)


def test_a_lone_surface_edits_without_a_partner_but_not_its_topology(loose):
    from fastfuncstuff.viewer.vocab import LoadMesh

    session, out = loose
    session.do(LoadMesh(str(out / "lh.white"), "lh", "white"))
    h = session.surfaces.hemis["lh"]
    before = h.states["white"].copy()
    session.do(_edit(session))
    moved = np.linalg.norm(h.states["white"] - before, axis=1)
    assert moved.max() > 0.1
    assert h.states["pial"] is h.states["white"]
    with pytest.raises(ValueError, match="loose mesh files"):
        session.surfaces.delete_vertex("lh", 0)


def test_the_mesh_window_opens_and_loads_with_no_subject(tmp_path):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.meshwindow import NAME
    from fastfuncstuff.viewer.ui.window import ViewerWindow

    path = _subject(tmp_path) / "surf" / "lh.pial"
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    session = ViewerSession(device=CPU)
    win = ViewerWindow(session)
    try:
        win.open_path(str(_shell_anat(tmp_path)))
        win._open_meshes()
        meshes = win.mesh_window
        assert meshes is not None and meshes.load_button.isEnabled()
        meshes.load_path(str(path))
        app.processEvents()
        assert meshes.table.item(0, NAME).text() == "● lh.pial"
        assert "depth sampling is on this surface" in meshes.status.text()
    finally:
        win.close()
        session.close()


def test_cli_mesh_flag_loads_headless_and_refuses_an_unnamed_file(tmp_path, capsys):
    import shutil

    from fastfuncstuff.cli.viewer import main

    surf = _subject(tmp_path) / "surf"
    anat = _shell_anat(tmp_path)
    assert main(["-no_window", "-device", "cpu", "-mesh", str(surf / "lh.white"), str(anat)]) == 0
    assert "LOAD_MESH" in capsys.readouterr().out
    shutil.copy(surf / "lh.white", tmp_path / "mesh.surf")
    with pytest.raises(SystemExit, match="hemisphere"):
        main(["-no_window", "-device", "cpu", "-mesh", str(tmp_path / "mesh.surf")])
