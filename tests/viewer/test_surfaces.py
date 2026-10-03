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
