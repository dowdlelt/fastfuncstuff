"""FreeSurfer surface I/O: scanner placement, faithful edited copies, patches."""

from __future__ import annotations

from pathlib import Path

import nibabel.freesurfer as nfs
import numpy as np
import pytest

from fastfuncstuff.io.freesurfer import (
    load_hemisphere,
    read_patch,
    read_surface,
    tkr_to_scanner,
    write_surface_like,
)

CRAS = np.array([2.86, 0.74, 7.69])


def _volume_info(cosines=((-1, 0, 0), (0, 0, -1), (0, 1, 0))) -> dict:
    # The default is a conformed LIA volume, the case recon-all produces.
    x, y, z = (np.asarray(c, float) for c in cosines)
    return {
        "head": np.array([2, 0, 20], np.int32),
        "valid": "1  # volume info valid",
        "filename": "orig.mgz",
        "volume": np.array([256, 256, 256]),
        "voxelsize": np.array([1.0, 1.0, 1.0]),
        "xras": x,
        "yras": y,
        "zras": z,
        "cras": CRAS,
    }


def _octahedron() -> tuple[np.ndarray, np.ndarray]:
    v = (
        np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], np.float32)
        * 10.0
    )
    f = np.array(
        [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]],
        np.int32,
    )
    return v, f


def _subject(tmp_path: Path, pial_scale: float = 1.2) -> Path:
    surf = tmp_path / "surf"
    surf.mkdir()
    v, f = _octahedron()
    nfs.write_geometry(str(surf / "lh.white"), v, f, volume_info=_volume_info())
    nfs.write_geometry(str(surf / "lh.pial"), v * pial_scale, f, volume_info=_volume_info())
    return tmp_path


def test_conformed_surface_is_a_pure_translation_by_cras():
    m = tkr_to_scanner(_volume_info())
    np.testing.assert_allclose(m[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(m[:3, 3], CRAS, atol=1e-12)


def test_oblique_volume_rotates_about_cras():
    # Volume centre is the tkregister origin and must land on c_ras whatever
    # the scanner orientation; directions follow the direction cosines.
    c, s = np.cos(0.3), np.sin(0.3)
    m = tkr_to_scanner(_volume_info(cosines=((-c, -s, 0), (0, 0, -1), (-s, c, 0))))
    np.testing.assert_allclose(m @ [0, 0, 0, 1], [*CRAS, 1], atol=1e-9)
    assert not np.allclose(m[:3, :3], np.eye(3))
    np.testing.assert_allclose(m[:3, :3] @ m[:3, :3].T, np.eye(3), atol=1e-9)


def test_unedited_save_is_bit_identical_and_edits_touch_only_moved_vertices(tmp_path):
    hemi = load_hemisphere(_subject(tmp_path), "lh", patches=False)
    out = tmp_path / "lh.pial.edit"
    hemi.save_state("pial", out)
    assert out.read_bytes() == hemi.paths["pial"].read_bytes()

    hemi.states["pial"][2] += np.float32([0.5, -0.25, 0.0])
    hemi.save_state("pial", out)
    before, after = read_surface(hemi.paths["pial"]), read_surface(out)
    changed = np.flatnonzero(np.any(before.vertices != after.vertices, axis=1))
    assert changed.tolist() == [2]
    # Pure translation frame, so the displacement carries over unchanged.
    np.testing.assert_allclose(after.vertices[2] - before.vertices[2], [0.5, -0.25, 0.0])
    # The trailer (volume geometry) is preserved.
    assert after.volume_info["cras"].tolist() == pytest.approx(CRAS.tolist())


def test_writer_refuses_to_overwrite_its_template(tmp_path):
    subj = _subject(tmp_path)
    white = subj / "surf" / "lh.white"
    with pytest.raises(ValueError, match="refusing"):
        write_surface_like(white, white, read_surface(white).vertices)


def test_mismatched_mesh_is_rejected(tmp_path):
    subj = _subject(tmp_path)
    v, f = _octahedron()
    nfs.write_geometry(str(subj / "surf" / "lh.pial"), v, f[:, ::-1], volume_info=_volume_info())
    with pytest.raises(ValueError, match="not the same mesh"):
        load_hemisphere(subj, "lh", patches=False)


def test_patch_ids_are_one_based_and_negated_on_the_border(tmp_path):
    path = tmp_path / "lh.test.patch.3d"
    rec = np.zeros(3, dtype=[("v", ">i4"), ("x", ">f4"), ("y", ">f4"), ("z", ">f4")])
    rec["v"] = [1, -3, 5]  # vertices 0, 2 (border), 4
    rec["x"] = [1.0, 2.0, 3.0]
    path.write_bytes(np.array([-1, 3], ">i4").tobytes() + rec.tobytes())
    coords, in_patch, border = read_patch(path, 6)
    assert np.flatnonzero(in_patch).tolist() == [0, 2, 4]
    assert np.flatnonzero(border).tolist() == [2]
    assert coords[[0, 2, 4], 0].tolist() == [1.0, 2.0, 3.0]


def test_cortex_label_becomes_a_vertex_mask(tmp_path):
    subj = _subject(tmp_path)
    (subj / "label").mkdir()
    ids = np.array([0, 2, 4])
    lines = ["#!ascii label", str(ids.size)] + [f"{i} 0 0 0 0" for i in ids]
    (subj / "label" / "lh.cortex.label").write_text("\n".join(lines) + "\n")
    hemi = load_hemisphere(subj, "lh", patches=False)
    assert hemi.cortex is not None
    assert np.flatnonzero(hemi.cortex).tolist() == [0, 2, 4]


def test_bundle_round_trip_after_a_split_writes_every_file_consistently(tmp_path):
    from fastfuncstuff.io.freesurfer import bundle_from_subject, read_patch, write_bundle
    from fastfuncstuff.surface.topology import split_edge

    subj = _subject(tmp_path)
    surf, label = subj / "surf", subj / "label"
    label.mkdir()
    v, f = _octahedron()
    nfs.write_geometry(str(surf / "lh.sphere"), v * 10.0, f, volume_info=_volume_info())  # r = 100
    nfs.write_morph_data(str(surf / "lh.thickness"), np.arange(6, dtype=np.float32))
    nfs.write_annot(
        str(label / "lh.aparc.annot"),
        np.array([0, 0, 1, 1, 0, 1]),
        np.array([[10, 20, 30, 0, 0], [40, 50, 60, 0, 0]], np.int32),
        ["a", "b"],
    )
    (label / "lh.cortex.label").write_text("#!ascii label\n3\n0 0 0 0 0\n2 0 0 0 0\n4 0 0 0 0\n")
    rec = np.zeros(6, dtype=[("v", ">i4"), ("x", ">f4"), ("y", ">f4"), ("z", ">f4")])
    rec["v"] = np.arange(1, 7)
    rec["x"] = np.arange(6)
    (surf / "lh.flat.patch.3d").write_bytes(np.array([-1, 6], ">i4").tobytes() + rec.tobytes())
    hemi = load_hemisphere(subj, "lh")
    bundle, src = bundle_from_subject(subj, hemi)
    assert set(src.files) >= {
        "surf:white", "surf:pial", "surf:sphere", "morph:thickness",
        "annot:aparc", "label:cortex", "patch:flat",
    }  # fmt: skip
    assert "surf:sphere" in bundle.spherical
    grown, m = split_edge(bundle, 0, 2)  # an octahedron edge
    out = tmp_path / "out"
    out.mkdir()
    write_bundle(grown, src, {k: out / p.name for k, p in src.files.items()})
    white = read_surface(out / "lh.white")
    assert white.vertices.shape == (7, 3) and white.faces.shape == (10, 3)
    assert white.volume_info["cras"].tolist() == pytest.approx(CRAS.tolist())  # trailer kept
    sphere = read_surface(out / "lh.sphere").vertices
    assert np.linalg.norm(sphere[m]) == pytest.approx(100.0, rel=1e-4)
    assert nfs.read_morph_data(str(out / "lh.thickness"))[m] == pytest.approx(1.0)  # mean of 0, 2
    assert nfs.read_annot(str(out / "lh.aparc.annot"))[0][m] == 0
    assert sorted(nfs.read_label(str(out / "lh.cortex.label")).tolist()) == [0, 2, 4, m]
    assert read_patch(out / "lh.flat.patch.3d", 7)[1][m]
