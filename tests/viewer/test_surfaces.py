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
    return (v * radius).astype(np.float32), ConvexHull(v).simplices.astype(np.int32)


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
