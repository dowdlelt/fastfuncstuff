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
