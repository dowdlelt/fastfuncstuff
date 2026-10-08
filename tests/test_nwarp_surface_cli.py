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
from fastfuncstuff.io.gifti import load_gifti_data, load_gifti_surface, mesh_fingerprint

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
    data, meta = load_gifti_data(f"{prefix}.tpl-test.lh.func.gii")
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
    cover, _ = load_gifti_data(f"{prefix}.tpl-test.lh.coverage.shape.gii")
    np.testing.assert_array_equal(cover, 1.0)


def test_footprint_and_depths_write_one_file_per_depth(world, tmp_path):
    prefix = _run(world, tmp_path, "-surf_depths", "0.2", "0.8", "-surf_depth_combine", "none")
    lo, meta = load_gifti_data(f"{prefix}.native.lh.depth-0.20.func.gii")
    hi, _ = load_gifti_data(f"{prefix}.native.lh.depth-0.80.func.gii")
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


def test_several_targets_in_one_pass_ship_their_geometry_and_mask(world, tmp_path):
    prefix = _run(
        world, tmp_path, "-surf_mesh", "native", str(world["tpl"]), "-surf_sample", "point"
    )
    nat, _ = load_gifti_data(f"{prefix}.native.lh.func.gii")
    tpl, meta = load_gifti_data(f"{prefix}.tpl-test.lh.func.gii")
    assert nat.shape[0] == 642 and tpl.shape[0] == len(world["expect_dir"])
    # the template's placed midthickness is where its vertices are in the subject
    v, f, smeta = load_gifti_surface(f"{prefix}.tpl-test.lh.midthickness.surf.gii")
    assert np.array_equal(f, world["tf"]) and smeta["mesh_fingerprint"] == meta["mesh_fingerprint"]
    want = CENTRE + 21.5 * world["expect_dir"]
    assert np.abs(v - want).max() < 0.5  # chord sag of the subject mesh
    assert meta["geometry"] == Path(f"{prefix}.tpl-test.lh.midthickness.surf.gii").name
    mask, _ = load_gifti_data(f"{prefix}.tpl-test.lh.mask.shape.gii")
    keep = np.zeros(len(mask), bool)
    keep[world["keep"]] = True
    np.testing.assert_array_equal(mask > 0, keep)  # cortex label AND (full) coverage


def test_subject_and_template_found_by_name_in_subjects_dir(world, tmp_path, monkeypatch):
    """fsaverage, onavg & co. live in $SUBJECTS_DIR (or $FREESURFER_HOME/subjects), not
    necessarily beside the subject; -surf takes a subject name there too."""
    from fastfuncstuff.processing.surface_projection import resolve_mesh, resolve_subject

    fsdir = tmp_path / "fs_subjects"
    fsdir.mkdir()
    world["tpl"].rename(fsdir / "tpl-test")
    world["subj"].rename(fsdir / "subj")
    monkeypatch.setenv("SUBJECTS_DIR", str(fsdir))
    monkeypatch.delenv("FREESURFER_HOME", raising=False)
    assert resolve_subject("subj") == fsdir / "subj"
    elsewhere = tmp_path / "elsewhere" / "subj"
    assert resolve_mesh("tpl-test", elsewhere) == fsdir / "tpl-test"
    with pytest.raises(FileNotFoundError, match="no-such-mesh"):
        resolve_mesh("no-such-mesh", elsewhere)
    world["subj"] = "subj"
    prefix = _run(world, tmp_path, "-surf_mesh", "tpl-test", "-surf_sample", "point")
    assert Path(f"{prefix}.tpl-test.lh.func.gii").is_file()


def _half_fov(world, tmp_path, value=100.0):
    """A constant EPI on the same oblique grid, cut off part way up the sphere."""
    img = nib.load(str(world["src"]))
    data = np.full(img.shape[:3] + (2,), value, np.float32)[:, :, :34]
    half = nib.Nifti1Image(data, img.affine)
    half.header.set_xyzt_units("mm", "sec")
    half.header["pixdim"][4] = 1.5
    path = tmp_path / "half.nii.gz"
    half.to_filename(str(path))
    return path


def test_partly_covered_vertices_keep_their_level_and_leave_the_mask(world, tmp_path):
    """A footprint poking out of the slab is the mean of the reads INSIDE it, not that
    mean diluted by the zeros nwarp returns outside; the mask still drops it."""
    world["src"] = _half_fov(world, tmp_path)
    prefix = _run(world, tmp_path)
    data, _ = load_gifti_data(f"{prefix}.native.lh.func.gii")
    cover, _ = load_gifti_data(f"{prefix}.native.lh.coverage.shape.gii")
    mask, _ = load_gifti_data(f"{prefix}.native.lh.mask.shape.gii")
    part = (cover > 0.05) & (cover < 0.95)
    assert part.sum() > 10 and (cover == 0).sum() > 10 and (cover == 1).sum() > 10
    # A diluted mean would sit at cover * 100, down to ~5 here.
    assert np.abs(data[part] - 100.0).max() < 0.5
    assert np.abs(data[cover == 1] - 100.0).max() < 1e-3
    assert (cover[(data == 0).all(axis=1)] == 0).all()  # nothing read: 0, not a stray value
    assert not mask[part].any() and mask[cover == 1].all()


def test_default_output_is_the_equal_weight_mean_of_five_depths(world, tmp_path):
    each = _run(world, tmp_path, "-surf_depth_combine", "none")
    depths = [load_gifti_data(f"{each}.native.lh.depth-{f:.2f}.func.gii")[0] for f in
              (0.1, 0.3, 0.5, 0.7, 0.9)]  # fmt: skip
    assert not Path(f"{each}.native.lh.func.gii").exists()  # "none": the depths only
    out = tmp_path / "mean"
    out.mkdir()
    prefix = _run(world, out)
    mean, meta = load_gifti_data(f"{prefix}.native.lh.func.gii")
    assert meta["depths"] == "0.1 0.3 0.5 0.7 0.9" and meta["depth_combine"] == "mean"
    np.testing.assert_allclose(mean, np.mean(depths, axis=0), rtol=1e-5)
    wtd = tmp_path / "wtd"
    wtd.mkdir()
    prefix = _run(world, wtd, "-surf_depth_weights", "1", "1", "1", "0", "0")
    low, _ = load_gifti_data(f"{prefix}.native.lh.func.gii")
    np.testing.assert_allclose(low, np.mean(depths[:3], axis=0), rtol=1e-5)
    tmean, _ = load_gifti_data(f"{prefix}.native.lh.mean.shape.gii")
    np.testing.assert_allclose(tmean, low.mean(axis=1), rtol=1e-5)
