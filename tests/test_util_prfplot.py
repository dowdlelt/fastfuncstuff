"""ffs_util_prfplot: maps from a surface bucket on an equal-area view of the sphere."""

from __future__ import annotations

import numpy as np
import pytest

from fastfuncstuff.io.gifti import save_gifti_data, save_gifti_surface, save_nifti_data_as_gifti


def _sphere(n=1200):
    from scipy.spatial import ConvexHull

    k = np.arange(n) + 0.5
    polar = np.arccos(1 - 2 * k / n)
    azimuth = np.pi * (1 + 5**0.5) * k
    points = np.stack(
        [np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)], 1
    )
    return 100 * points, ConvexHull(points).simplices


@pytest.fixture
def subject(tmp_path):
    """A sphere per hemisphere and a pRF bucket on each, responsive at the back."""
    import nibabel as nib

    vertices, faces = _sphere()
    unit = vertices / 100
    labels = ["x", "y", "angle", "eccentricity", "rfsize", "r2"]
    header = nib.Nifti2Image(np.zeros((len(vertices), 1, 1, 1), np.float32), np.eye(4)).header
    from fastfuncstuff.io.gifti import mesh_fingerprint, set_surface_meta

    set_surface_meta(
        header,
        {
            "mesh_fingerprint": mesh_fingerprint(faces, len(vertices)),
            "prf_units": "deg",
            "prf_stimulus_radius": "12",
        },
    )
    for hemi in ("lh", "rh"):
        save_gifti_surface(tmp_path / f"{hemi}.sphere.surf.gii", vertices, faces)
        x, y = 10 * unit[:, 0], 10 * unit[:, 2]
        r2 = np.clip(-unit[:, 1], 0, 1)  # posterior responds
        bucket = np.stack(
            [x, y, np.degrees(np.arctan2(y, x)) % 360, np.hypot(x, y), 1 + np.hypot(x, y), r2], 1
        )
        save_nifti_data_as_gifti(
            tmp_path / f"prf.{hemi}.func.gii",
            bucket[:, None, None, :].astype(np.float32),
            header=header,
            brick_labels=labels,
        )
        rois = np.where(unit[:, 1] < -0.6, np.where(unit[:, 2] > 0, 1, 2), 0)
        save_gifti_data(tmp_path / f"{hemi}.rois.shape.gii", rois.astype(np.float32))
    return tmp_path


def test_maps_are_drawn_for_both_hemispheres_from_one_named_file(subject):
    from fastfuncstuff.cli.util_prfplot import main

    out = subject / "maps.png"
    argv = [
        "-input", str(subject / "prf.lh.func.gii"),
        "-sphere", str(subject / "lh.sphere.surf.gii"), str(subject / "rh.sphere.surf.gii"),
        "-prefix", str(out),
    ]  # fmt: skip
    assert main(argv) == 0
    assert out.stat().st_size > 10_000

    labelled = subject / "labelled.png"
    assert main([
        *argv[:-2],
        "-rois", str(subject / "lh.rois.shape.gii"), str(subject / "rh.rois.shape.gii"),
        "-center", "rois", "-roi_subset", "1", "-align", "rois", "-rotate", "10",
        "-radius", "70", "-maps", "angle", "rfsize", "-prefix", str(labelled),
    ]) == 0  # fmt: skip
    assert labelled.stat().st_size > 10_000


def test_the_bucket_must_be_on_the_sphere_and_hold_the_maps(subject):
    from fastfuncstuff.cli.util_prfplot import main

    other, other_faces = _sphere(500)
    save_gifti_surface(subject / "small.surf.gii", other, other_faces)
    base = ["-input", str(subject / "prf.lh.func.gii"), "-prefix", str(subject / "x.png")]
    with pytest.raises(SystemExit, match="not the same mesh"):
        main([*base, "-sphere", str(subject / "small.surf.gii"), str(subject / "small.surf.gii")])
    with pytest.raises(SystemExit, match="has no exponent"):
        main([
            *base, "-sphere", str(subject / "lh.sphere.surf.gii"),
            str(subject / "rh.sphere.surf.gii"), "-maps", "exponent",
        ])  # fmt: skip
    with pytest.raises(SystemExit, match="need -rois"):
        main([
            *base, "-sphere", str(subject / "lh.sphere.surf.gii"),
            str(subject / "rh.sphere.surf.gii"), "-align", "rois",
        ])  # fmt: skip


def test_a_template_bucket_finds_its_sphere_and_the_subjects_sulc(tmp_path, capsys):
    """What ffs_pyrf records (mesh, subject) is enough: the template has no ?h.sphere
    and no sulc, so its sphere.reg is used and the subject's sulc carried onto it."""
    import nibabel as nib

    from fastfuncstuff.cli.util_prfplot import _resolve_surfaces, main
    from fastfuncstuff.io.gifti import set_surface_meta

    fine, fine_faces = _sphere(3000)
    coarse, coarse_faces = _sphere(800)
    subject, template = tmp_path / "sub-01", tmp_path / "tmpl-ico"
    for folder in (subject, template):
        (folder / "surf").mkdir(parents=True)
    meta = {"mesh": "tmpl-ico", "subject": str(subject), "prf_units": "deg"}
    labels = ["angle", "eccentricity", "r2"]
    header = nib.Nifti2Image(np.zeros((len(coarse), 1, 1, 1), np.float32), np.eye(4)).header
    set_surface_meta(header, meta)
    unit = coarse / 100
    for hemi in ("lh", "rh"):
        nib.freesurfer.write_geometry(subject / "surf" / f"{hemi}.sphere", fine, fine_faces)
        nib.freesurfer.write_geometry(subject / "surf" / f"{hemi}.sphere.reg", fine, fine_faces)
        nib.freesurfer.write_morph_data(subject / "surf" / f"{hemi}.sulc", fine[:, 2] / 100)
        nib.freesurfer.write_geometry(
            template / "surf" / f"{hemi}.sphere.reg", coarse, coarse_faces
        )
        bucket = np.stack(
            [unit[:, 0] * 180 + 180, 5 + 5 * unit[:, 2], np.clip(-unit[:, 1], 0, 1)], 1
        )
        save_nifti_data_as_gifti(
            tmp_path / f"prf.{hemi}.func.gii",
            bucket[:, None, None, :].astype(np.float32),
            header=header,
            brick_labels=labels,
        )

    sphere, vertices, _, sulc, source = _resolve_surfaces(
        "lh", len(coarse), meta, "prf", None, None, None
    )
    assert sphere == str(template / "surf" / "lh.sphere.reg")
    assert source.endswith("via sphere.reg")
    # interpolated from the subject's mesh: the template's own z, to within a face
    assert np.abs(sulc - vertices[:, 2] / 100).max() < 0.05

    out = tmp_path / "auto.png"
    assert main(["-input", str(tmp_path / "prf.lh.func.gii"), "-prefix", str(out)]) == 0
    assert out.stat().st_size > 10_000

    # a wrong -fs_subj with no template to fall back on says which meshes were tried
    meta_alone = {"subject": str(subject)}
    with pytest.raises(SystemExit, match=r"sphere \(3000\).*not the same mesh"):
        _resolve_surfaces("lh", len(coarse), meta_alone, "prf", None, None, None)


def test_subject_labels_by_name_are_carried_onto_a_template_bucket(tmp_path):
    """-rois BA.annot -roi_labels V1: found in the subject's label/, moved onto the
    template mesh by nearest corner, named by the annot with its shared suffix cut."""
    import nibabel as nib

    from fastfuncstuff.cli.util_prfplot import _load_rois, _roi_specs, main
    from fastfuncstuff.io.gifti import set_surface_meta

    fine, fine_faces = _sphere(3000)
    coarse, coarse_faces = _sphere(800)
    subject, template = tmp_path / "sub-01", tmp_path / "tmpl-ico"
    for d in (subject / "surf", subject / "label", template / "surf"):
        d.mkdir(parents=True)
    meta = {"mesh": "tmpl-ico", "subject": str(subject)}
    header = nib.Nifti2Image(np.zeros((len(coarse), 1, 1, 1), np.float32), np.eye(4)).header
    set_surface_meta(header, meta)
    # annot index 0 unknown, 1 back-top ("V1"), 2 back-bottom ("V2"), 3 front ("BA4")
    y, z = fine[:, 1] / 100, fine[:, 2] / 100
    annot = np.where(y < -0.5, np.where(z > 0, 1, 2), np.where(y > 0.5, 3, 0))
    ctab = np.array([[25, 5, 25, 0, 0], [255, 0, 0, 0, 0], [0, 255, 0, 0, 0], [0, 0, 255, 0, 0]])
    names = [b"unknown", b"V1_exvivo.thresh", b"V2_exvivo.thresh", b"BA4_exvivo.thresh"]
    for hemi in ("lh", "rh"):
        nib.freesurfer.write_geometry(subject / "surf" / f"{hemi}.sphere.reg", fine, fine_faces)
        nib.freesurfer.write_geometry(
            template / "surf" / f"{hemi}.sphere.reg", coarse, coarse_faces
        )
        nib.freesurfer.write_annot(subject / "label" / f"{hemi}.BA.annot", annot, ctab, names)
        ids = np.flatnonzero(annot == 1)  # FreeSurfer's ascii label, which nibabel can't write
        rows = "".join(f"{v} 0 0 0 0\n" for v in ids)
        (subject / "label" / f"{hemi}.V1.label").write_text(f"#!ascii\n{len(ids)}\n{rows}")
        r2 = np.clip(-coarse[:, 1] / 100, 0, 1)
        bucket = np.stack([coarse[:, 0] % 360, 5 + 0 * r2, r2], 1)
        save_nifti_data_as_gifti(
            tmp_path / f"prf.{hemi}.func.gii",
            bucket[:, None, None, :].astype(np.float32),
            header=header,
            brick_labels=["angle", "eccentricity", "r2"],
        )

    specs = _roi_specs(["BA.annot"], ["lh", "rh"], "lh", [subject])
    sphere = str(template / "surf" / "lh.sphere.reg")
    labels, got = _load_rois(specs, "lh", len(coarse), [subject], sphere, coarse, ["V1", "V2"])
    assert got == {2: "V1", 3: "V2"}  # BA4 dropped, suffix cut
    cy, cz = coarse[:, 1] / 100, coarse[:, 2] / 100
    clear = np.abs(cz) > 0.1  # away from the V1/V2 seam, the label is exact
    expect = np.where(cy < -0.55, np.where(cz > 0, 2, 3), 0)
    far = clear & (np.abs(cy + 0.5) > 0.05)
    assert np.array_equal(labels[far], expect[far])

    # a .label in the subject's folder is on the subject's mesh, whatever its length
    v1, got = _load_rois(
        _roi_specs(["V1.label"], ["lh", "rh"], "lh", [subject]),
        "lh", len(coarse), [subject], sphere, coarse, None,
    )  # fmt: skip
    assert got == {1: "V1"} and np.array_equal(v1[far] == 1, expect[far] == 2)

    out = tmp_path / "rois.png"
    assert main([
        "-input", str(tmp_path / "prf.lh.func.gii"), "-rois", "BA.annot",
        "-roi_labels", "V1", "V2", "-center", "rois", "-roi_subset", "V1",
        "-prefix", str(out),
    ]) == 0  # fmt: skip
    with pytest.raises(SystemExit, match="no label 'MT'"):
        main(["-input", str(tmp_path / "prf.lh.func.gii"), "-rois", "BA.annot",
              "-roi_labels", "MT", "-prefix", str(out)])  # fmt: skip
    with pytest.raises(SystemExit, match="not found"):
        main(["-input", str(tmp_path / "prf.lh.func.gii"), "-rois", "nope.annot",
              "-prefix", str(out)])  # fmt: skip
