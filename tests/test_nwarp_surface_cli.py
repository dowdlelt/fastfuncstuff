"""ffs_nwarp -surf end to end: a synthetic FreeSurfer subject and template, real files.

The subject's cortex is a pair of concentric spheres placed with a non-zero c_ras, its
sphere.reg is a rotation of the native directions (a registration), the EPI is a
linear function of scanner position on an oblique grid. A template vertex placed
through the sphere must therefore read the function at a known scanner point.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import nibabel.freesurfer as nfs
import numpy as np
import pytest

from fastfuncstuff.cli.nwarp import main
from fastfuncstuff.io.gifti import load_gifti_data, mesh_fingerprint

CRAS = np.array([3.0, -2.0, 5.0])
CENTRE = np.array([1.0, 2.0, -1.0])  # scanner mm
COEF = np.array([0.6, -0.3, 0.9])


def _icosphere(level: int) -> tuple[np.ndarray, np.ndarray]:
    t = (1 + 5**0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t)]
    v += [(0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4)]
    f += [(11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8)]
    f += [(3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.array(p, float) / np.linalg.norm(p) for p in v]
    faces = f
    for _ in range(level):
        mid: dict[tuple[int, int], int] = {}

        def m(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in mid:
                p = verts[a] + verts[b]
                verts.append(p / np.linalg.norm(p))
                mid[key] = len(verts) - 1
            return mid[key]

        faces = [
            g
            for a, b, c in faces
            for g in (
                (a, m(a, b), m(c, a)),
                (b, m(b, c), m(a, b)),
                (c, m(c, a), m(b, c)),
                (m(a, b), m(b, c), m(c, a)),
            )
        ]
    return np.asarray(verts), np.asarray(faces, np.int32)


def _vinfo() -> dict:
    return {
        "head": np.array([2, 0, 20], np.int32),
        "valid": "1  # volume info valid",
        "filename": "orig.mgz",
        "volume": np.array([256, 256, 256]),
        "voxelsize": np.array([1.0, 1.0, 1.0]),
        "xras": np.array([-1.0, 0, 0]),
        "yras": np.array([0, 0, -1.0]),
        "zras": np.array([0, 1.0, 0]),
        "cras": CRAS,
    }


def _rot(seed: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.diag(r))


@pytest.fixture
def world(tmp_path: Path):
    subj = tmp_path / "subj"
    (subj / "surf").mkdir(parents=True)
    (subj / "label").mkdir()
    d, f = _icosphere(3)
    reg = _rot(1)  # the subject's registration to the common sphere
    for name, pos in (("white", CENTRE + 20 * d), ("pial", CENTRE + 23 * d)):
        nfs.write_geometry(str(subj / "surf" / f"lh.{name}"), pos - CRAS, f, volume_info=_vinfo())
    nfs.write_geometry(str(subj / "surf" / "lh.sphere.reg"), 100 * d @ reg.T, f)

    tpl = tmp_path / "tpl-test"
    (tpl / "surf").mkdir(parents=True)
    (tpl / "label").mkdir()
    td, tf = _icosphere(2)
    td = td @ _rot(2).T  # shares no vertices with the subject
    nfs.write_geometry(str(tpl / "surf" / "lh.sphere.reg"), 100 * td, tf)
    keep = np.flatnonzero(td[:, 2] > -0.5)
    (tpl / "label" / "lh.cortex.label").write_text(
        "#!ascii label\n" + f"{keep.size}\n" + "".join(f"{i} 0 0 0 0\n" for i in keep)
    )

    t = np.deg2rad(10)
    aff = np.eye(4)
    aff[:3, :3] = [[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]]
    aff[:3, 3] = CENTRE - aff[:3, :3] @ np.full(3, 30.0)
    ii, jj, kk = np.meshgrid(*(np.arange(60),) * 3, indexing="ij")
    xyz = np.stack([ii, jj, kk], -1) @ aff[:3, :3].T + aff[:3, 3]
    f0 = xyz @ COEF + 40.0
    epi = np.stack([f0, 2 * f0], axis=-1).astype(np.float32)
    src = tmp_path / "epi.nii.gz"
    img = nib.Nifti1Image(epi, aff)
    img.header.set_xyzt_units("mm", "sec")
    img.header["pixdim"][4] = 1.5
    img.to_filename(str(src))
    ident = tmp_path / "ident.aff12.1D"
    ident.write_text("1 0 0 0 0 1 0 0 0 0 1 0\n")
    # where each template vertex is in the subject: undo the registration
    expect_dir = td @ reg  # (reg.T)^-1 = reg
    return dict(subj=subj, tpl=tpl, src=src, chain=ident, expect_dir=expect_dir, tf=tf, keep=keep)


def _run(world, tmp_path, *extra):
    prefix = tmp_path / "out" / "proj"
    prefix.parent.mkdir(exist_ok=True)
    main(
        [
            "-source", str(world["src"]), "-nwarp", str(world["chain"]),
            "-master", str(world["src"]), "-prefix", str(prefix),
            "-surf", str(world["subj"]), "-surf_hemi", "lh",
            "-interp", "linear", "-device", "cpu", "-verb", "0", *extra,
        ]
    )  # fmt: skip
    return prefix


def test_template_vertices_read_the_epi_where_the_sphere_puts_them(world, tmp_path):
    prefix = _run(world, tmp_path, "-surf_mesh", "tpl-test", "-surf_sample", "point")
    data, meta = load_gifti_data(f"{prefix}.lh.func.gii")
    n = len(world["expect_dir"])
    assert data.shape == (n, 2)
    assert meta["mesh"] == "tpl-test" and meta["TR_seconds"] == "1.5"
    assert meta["mesh_fingerprint"] == mesh_fingerprint(world["tf"], n)
    # Mid-volume between r=20 and r=23 on a sphere (area ~ r^2): solve the quadratic.
    r_mid = np.sqrt(0.5 * 20**2 + 0.5 * 23**2)
    want = (CENTRE + r_mid * world["expect_dir"]) @ COEF + 40.0
    # Flat subject triangles sit inside the sphere by the chord sag (< 0.3 mm here).
    np.testing.assert_allclose(data[:, 0], want, atol=0.5)
    np.testing.assert_allclose(data[:, 1], 2 * data[:, 0], rtol=1e-5)
    cover, _ = load_gifti_data(f"{prefix}.lh.coverage.shape.gii")
    np.testing.assert_array_equal(cover, 1.0)


def test_footprint_and_depths_write_one_file_per_depth(world, tmp_path):
    prefix = _run(world, tmp_path, "-surf_depths", "0.2", "0.8")
    lo, meta = load_gifti_data(f"{prefix}.lh.depth-0.20.func.gii")
    hi, _ = load_gifti_data(f"{prefix}.lh.depth-0.80.func.gii")
    assert meta["mesh"] == "native" and meta["sampling"] == "footprint"
    d, _ = _icosphere(3)
    # The footprint mean of a linear field is the field at the patch centroid, which on
    # this symmetric mesh is the vertex pulled in by the patch's curvature: < 0.5 mm.
    rho = (np.sqrt((1 - 0.2) * 400 + 0.2 * 529) - 20) / 3
    want = (CENTRE + (20 + 3 * rho) * d) @ COEF + 40.0
    np.testing.assert_allclose(lo[:, 0], want, atol=0.6)
    assert not np.allclose(lo, hi)


def test_surf_refuses_volume_only_flags(world, tmp_path):
    with pytest.raises(SystemExit, match="-dxyz"):
        _run(world, tmp_path, "-dxyz", "2")
