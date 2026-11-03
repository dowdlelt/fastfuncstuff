"""ffs_pyrf on surface data: GIfTI runs in, a GIfTI bucket on the same mesh out."""

from __future__ import annotations

import numpy as np
import pytest

from fastfuncstuff.io.gifti import (
    mesh_fingerprint,
    save_gifti_data,
    save_gifti_surface,
)


def _sheet(nx: int, ny: int):
    """A triangulated rectangular sheet: a mesh with real topology, cheap to fit on."""
    vertices = np.array([[x, y, 0.0] for y in range(ny) for x in range(nx)], np.float32)
    faces = []
    for y in range(ny - 1):
        for x in range(nx - 1):
            a = y * nx + x
            faces += [[a, a + 1, a + nx], [a + 1, a + nx + 1, a + nx]]
    return vertices, np.asarray(faces, np.int32)


@pytest.fixture
def surface_runs(tmp_path):
    """Two GIfTI runs of pRF responses to a sweeping bar, on a 10x6 sheet."""
    rng = np.random.default_rng(0)
    n_tps, pixels = 40, 10
    aperture = np.zeros((pixels, pixels, n_tps), np.float32)
    for t in range(n_tps):
        aperture[:, t % pixels, t] = 1.0
    np.save(tmp_path / "aperture.npy", aperture)

    vertices, faces = _sheet(10, 6)
    save_gifti_surface(tmp_path / "mesh.surf.gii", vertices, faces)
    meta = {
        "TR_seconds": "1.0",
        "geometry": "mesh.surf.gii",
        "mesh_fingerprint": mesh_fingerprint(faces, len(vertices)),
    }
    grid_y, grid_x = np.mgrid[0:pixels, 0:pixels]
    runs = []
    for run in range(2):
        series = np.zeros((len(vertices), n_tps), np.float32)
        for v in range(len(vertices)):
            cx, cy = rng.uniform(2, pixels - 2, 2)
            weights = np.exp(-(((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / 8.0))
            series[v] = 100 + 5 * (aperture * weights[:, :, None]).sum((0, 1))
        series += rng.normal(0, 1, series.shape).astype(np.float32)
        path = tmp_path / f"run{run}.func.gii"
        save_gifti_data(path, series, meta=meta, time_series=True)
        runs.append(str(path))
    return runs, str(tmp_path / "aperture.npy"), meta


def _argv(runs, aperture, prefix, *extra):
    return [
        "-input", *runs,
        "-stimulus", *([aperture] * len(runs)),
        "-prefix", prefix,
        "-device", "cpu",
        "-hrf", "canonical",
        "-grid_angles", "4",
        "-grid_ecc_steps", "3",
        "-grid_sigma_steps", "2",
        "-screen_extent", "10",
        *extra,
    ]  # fmt: skip


def test_gifti_runs_fit_and_write_a_bucket_on_the_same_mesh(surface_runs, tmp_path):
    """A surface run goes in and a labeled per-vertex bucket comes out, mesh identity kept."""
    from fastfuncstuff.cli.pyrf import main

    runs, aperture, meta = surface_runs
    prefix = str(tmp_path / "prf.func.gii")

    assert main(_argv(runs, aperture, prefix)) == 0

    # Through load_nifti, the route every ffs tool reads a .gii by: a (V, 1, 1, K)
    # image whose labels and mesh identity ride in the surface extension.
    from fastfuncstuff.io.afni import load_nifti
    from fastfuncstuff.io.gifti import surface_meta

    image = load_nifti(prefix)
    out_meta = surface_meta(image.header) or {}
    assert image.shape[:3] == (60, 1, 1)
    assert out_meta["mesh_fingerprint"] == meta["mesh_fingerprint"]
    labels = out_meta["brick_labels"].split("\t")
    assert {"x", "y", "sigma", "r2"} <= set(labels)
    assert np.isfinite(np.asarray(image.dataobj)[:, 0, 0, labels.index("r2")]).all()


def test_surface_blur_needs_a_mesh_mask(surface_runs, tmp_path):
    """Heat blur reads the mask to know which vertices carry data; say so, don't crash."""
    from fastfuncstuff.cli.pyrf import main

    runs, aperture, _ = surface_runs
    argv = _argv(runs, aperture, str(tmp_path / "prf.func.gii"), "-do_blur", "4")

    with pytest.raises(SystemExit):
        main(argv)


def test_noise_pool_mask_is_refused_on_surface_input(surface_runs, tmp_path):
    """The pool reaches the loader as a flat array, which has no mesh identity."""
    from fastfuncstuff.cli.pyrf import main

    runs, aperture, _ = surface_runs
    argv = _argv(
        runs, aperture, str(tmp_path / "prf.func.gii"), "-denoise", "-noise_pool_mask", "all"
    )

    with pytest.raises(SystemExit):
        main(argv)


# --------------------------------------------------------------------------------
# A surface fit writes surface results, however the surface data arrived
# --------------------------------------------------------------------------------


def test_domain_prefix_is_one_rule_for_every_tool():
    """ffs_pyrf, ffs_denoise and ffs_varpart all need it and all need it to agree."""
    from fastfuncstuff.cli_utils import domain_prefix

    # No extension typed: the right default, silently.
    assert domain_prefix("out", surface=True).nifti_ext == ".func.gii"
    assert domain_prefix("out", surface=True, surface_ext=".shape.gii").nifti_ext == ".shape.gii"
    assert domain_prefix("out", surface=False).nifti_ext == ".nii.gz"
    # An extension the user typed is a choice, so it is kept where it can be.
    assert domain_prefix("out.shape.gii", surface=True).nifti_ext == ".shape.gii"
    assert domain_prefix("out.nii.zst", surface=False).nifti_ext == ".nii.zst"
    assert domain_prefix("out.nii.gz", surface=False).stem == "out"
    # Naming the other domain is refused rather than overridden.
    with pytest.raises(ValueError, match=r"names a NIfTI file.*on a surface"):
        domain_prefix("out.nii.gz", surface=True)
    with pytest.raises(ValueError, match=r"names a GIfTI file.*on a volume"):
        domain_prefix("out.func.gii", surface=False)


def test_gifti_input_writes_gifti_without_fs_subj(capsys):
    """The bug: the promotion was gated on -fs_subj, so GIfTI runs handed straight
    to -input kept a NIfTI prefix and 40962 vertices were written as a (V, 1, 1)
    volume that no surface tool will take."""
    from argparse import Namespace

    from fastfuncstuff.cli.pyrf import _surface_prefix
    from fastfuncstuff.cli_utils import parse_prefix

    def resolve(prefix, inputs, fs_subj=None, verb=1):
        args = Namespace(prefix=prefix, fs_subj=fs_subj, verb=verb)
        return _surface_prefix(args, None, parse_prefix(prefix), inputs)

    gii = ["run1.lh.func.gii", "run2.lh.func.gii"]
    assert resolve("out", gii).nifti_ext == ".func.gii"
    assert "surface -input" in capsys.readouterr().out  # and it says so
    # A sub-brick selector on the input must not hide the extension.
    assert resolve("out", ["run1.lh.func.gii[0..$]"]).nifti_ext == ".func.gii"
    # -fs_subj reaches the same place from volumes.
    assert resolve("out", ["run1.nii.gz"], fs_subj="/fs/sub-X").nifti_ext == ".func.gii"
    # Volumes stay volumes, and say nothing.
    capsys.readouterr()
    assert resolve("out", ["run1.nii.gz"]).nifti_ext == ".nii.gz"
    assert capsys.readouterr().out == ""


def test_a_nifti_prefix_on_a_surface_fit_is_refused_not_overridden():
    """-prefix out.nii.gz with GIfTI input is a contradiction the user typed; it
    used to be ignored for -fs_subj and silently obeyed for GIfTI input."""
    from argparse import Namespace

    from fastfuncstuff.cli.pyrf import _surface_prefix
    from fastfuncstuff.cli_utils import parse_prefix

    errors: list[str] = []

    def fail(message):  # what argparse does: report, then stop
        errors.append(message)
        raise SystemExit(2)

    args = Namespace(prefix="out.nii.gz", fs_subj=None, verb=0)
    with pytest.raises(SystemExit):
        _surface_prefix(args, Namespace(error=fail), parse_prefix(args.prefix), ["r.lh.func.gii"])
    assert len(errors) == 1
    assert "names a NIfTI file" in errors[0] and ".func.gii" in errors[0]


def test_one_hemisphere_is_fit_as_given_not_refused(tmp_path):
    """A dataset with no rh (or no lh) happens. Pairing is a convenience, so when
    there is nothing to pair the fit proceeds on what it was given -- the surface
    prefix still applies, because it is still a surface fit."""
    from argparse import Namespace

    from fastfuncstuff.cli.pyrf import _surface_prefix
    from fastfuncstuff.cli_utils import pair_hemisphere_inputs, parse_prefix

    lone = tmp_path / "run-01.native.lh.func.gii"
    lone.write_bytes(b"")  # never opened: the partner is missing before any read
    args = Namespace(prefix="out", fs_subj=None, verb=0, hemi_pair="auto", hemi_dir=None)
    assert pair_hemisphere_inputs(args, None, [str(lone)], "out") is None
    assert _surface_prefix(args, None, parse_prefix("out"), [str(lone)]).nifti_ext == ".func.gii"


def test_hemi_pair_off_fits_what_was_given(tmp_path):
    """The escape hatch has to work even when a partner is sitting right there."""
    from argparse import Namespace

    from fastfuncstuff.cli_utils import pair_hemisphere_inputs

    for hemi in ("lh", "rh"):
        (tmp_path / f"run-01.native.{hemi}.func.gii").write_bytes(b"")
    runs = [str(tmp_path / "run-01.native.lh.func.gii")]
    off = Namespace(prefix="out", fs_subj=None, verb=0, hemi_pair="off", hemi_dir=None)
    assert pair_hemisphere_inputs(off, None, runs, "out") is None
    # And -fs_subj owns the surface route when it is given, so pairing stands down.
    fs = Namespace(prefix="out", fs_subj="/fs/sub-X", verb=0, hemi_pair="auto", hemi_dir=None)
    assert pair_hemisphere_inputs(fs, None, runs, "out") is None


def test_a_sub_brick_selector_declines_pairing_rather_than_dropping_it(tmp_path):
    """The merge reads whole files, so a selector cannot be honoured; fitting the
    hemispheres separately keeps it, which is the lesser surprise."""
    from argparse import Namespace

    from fastfuncstuff.cli_utils import pair_hemisphere_inputs

    for hemi in ("lh", "rh"):
        (tmp_path / f"run-01.native.{hemi}.func.gii").write_bytes(b"")
    args = Namespace(prefix="out", fs_subj=None, verb=0, hemi_pair="auto", hemi_dir=None)
    runs = [str(tmp_path / "run-01.native.lh.func.gii") + "[0..$]"]
    assert pair_hemisphere_inputs(args, None, runs, "out") is None


# --------------------------------------------------------------------------------
# Volume and surface fit together: a vector of locations is a vector of locations
# --------------------------------------------------------------------------------


def test_paired_volume_and_surface_fit_together_and_write_both_domains(tmp_path):
    """-surface_input: voxels and vertices on one location axis, so there is one
    super-grid, one HRF selection and one noise pool across both -- then every map
    is written twice, each with its own geometry."""
    import nibabel as nib

    from fastfuncstuff.cli.pyrf import main
    from fastfuncstuff.io.gifti import load_gifti_data, save_gifti_data
    from tests.test_prf import _write_prf_dataset

    inputs, stimuli, mask_path, _shape = _write_prf_dataset(tmp_path)
    n_vertices = 6
    surfaces = []
    for run, volume in enumerate(inputs):
        series = np.asarray(nib.load(volume).dataobj).reshape(-1, 120)
        # The responsive voxels' own series, so the vertices have a pRF to find.
        vertex_series = series[:n_vertices].astype(np.float32)
        path = tmp_path / f"surf{run}.func.gii"
        save_gifti_data(
            path,
            vertex_series,
            meta={"mesh_fingerprint": f"V{n_vertices}_F2_test", "TR_seconds": "1"},
            time_series=True,
        )
        surfaces.append(str(path))
    surf_mask = tmp_path / "surfmask.shape.gii"
    keep = np.array([1, 1, 1, 1, 0, 0], np.float32)  # a vertex mask that excludes some
    save_gifti_data(surf_mask, keep, meta={"mesh_fingerprint": f"V{n_vertices}_F2_test"})

    prefix = str(tmp_path / "paired.nii.gz")
    assert (
        main(
            [
                "-input",
                *inputs,
                "-surface_input",
                *surfaces,
                "-surface_mask",
                str(surf_mask),
                "-stimulus",
                *stimuli,
                "-mask",
                mask_path,
                "-prefix",
                prefix,
                "-device",
                "cpu",
                "-verb",
                "0",
                "-tr",
                "1",
            ]
        )  # fmt: skip
        == 0
    )
    volume_out = tmp_path / "paired_volume.nii.gz"
    surface_out = tmp_path / "paired_surface.shape.gii"
    assert volume_out.is_file() and surface_out.is_file()
    # Each domain keeps its own geometry, not the other's or the pseudo-grid's.
    assert nib.load(str(volume_out)).shape[:3] == _shape
    vertices, meta = load_gifti_data(surface_out)
    assert vertices.shape[0] == n_vertices
    assert meta["mesh_fingerprint"] == f"V{n_vertices}_F2_test"
    # -surface_mask is honoured independently of -mask: the vertices it excludes
    # are not fit and stay at the fill value, while the ones it keeps are.
    assert np.all(vertices[4:] == 0)
    assert np.isfinite(vertices[:4]).any()
    assert not np.all(vertices[:4] == 0)
    # Both domains are the same fit, so they carry the same maps in the same order.
    assert nib.load(str(volume_out)).shape[3] == vertices.shape[1]


def test_paired_input_is_refused_when_the_fit_is_already_on_a_surface(tmp_path):
    """-surface_input pairs with VOLUME runs; with GIfTI -input or -fs_subj there is
    nothing to pair it to, and silently ignoring it would drop half the data."""
    from fastfuncstuff.cli.pyrf import main
    from tests.test_prf import _write_prf_dataset

    inputs, stimuli, mask_path, _shape = _write_prf_dataset(tmp_path)
    base = [
        "-stimulus", *stimuli, "-mask", mask_path,
        "-prefix", str(tmp_path / "x"), "-device", "cpu", "-verb", "0",
    ]  # fmt: skip
    with pytest.raises(SystemExit):
        main(["-input", "a.lh.func.gii", "-surface_input", "b.lh.func.gii", *base])
    # One surface run per volume run, or the pairing is not positional at all.
    with pytest.raises(SystemExit):
        main(["-input", *inputs, "-surface_input", "only-one.func.gii", *base])
