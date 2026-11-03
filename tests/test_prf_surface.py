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


def test_screening_is_shared_across_the_volume_and_the_surface(tmp_path):
    """-screen ranks the whole location axis, so a vertex competes with a voxel on
    the same R2 and both scatter back into their own domain.

    This is the point of the pseudo-grid: screening is the step that gathers kept
    locations out of place, and it needs no knowledge of which domain a row is.
    Without it a paired fit pays for every location -- on the run that prompted
    this, screening cut 40,962 vertices to 13,535.
    """
    import nibabel as nib

    from fastfuncstuff.cli.pyrf import main
    from fastfuncstuff.io.gifti import load_gifti_data, save_gifti_data
    from tests.test_prf import _write_prf_dataset

    inputs, stimuli, mask_path, shape = _write_prf_dataset(tmp_path)
    n_vertices = 8
    surfaces = []
    for run, volume in enumerate(inputs):
        series = np.asarray(nib.load(volume).dataobj).reshape(-1, 120)
        # Half responsive (copied from the responsive voxels), half not, so
        # screening has something to reject on the surface side too.
        vertex_series = np.concatenate([series[:4], series[-4:]]).astype(np.float32)
        path = tmp_path / f"s{run}.func.gii"
        save_gifti_data(
            path, vertex_series, meta={"mesh_fingerprint": "V8_F4", "TR_seconds": "1"},
            time_series=True,
        )  # fmt: skip
        surfaces.append(str(path))

    prefix = str(tmp_path / "scr.nii.gz")
    assert (
        main(
            [
                "-input",
                *inputs,
                "-surface_input",
                *surfaces,
                "-stimulus",
                *stimuli,
                "-mask",
                mask_path,
                "-screen",
                "0.0",
                "-prefix",
                prefix,
                "-device",
                "cpu",
                "-verb",
                "1",
                "-tr",
                "1",
            ]
        )  # fmt: skip
        == 0
    )
    volume_out = nib.load(str(tmp_path / "scr_volume.nii.gz"))
    vertices, _meta = load_gifti_data(tmp_path / "scr_surface.shape.gii")
    # Both domains come back on their own geometry, not the pseudo-grid.
    assert volume_out.shape[:3] == shape
    assert vertices.shape[0] == n_vertices
    # The responsive vertices were fit; the unresponsive ones were screened out
    # and come back NaN rather than as a fit of the wrong region -- all but the
    # one map that is a property of the input (meanvol) and so is kept for every
    # location whether or not it was fit.
    finite = np.isfinite(vertices).sum(axis=1)
    n_maps = vertices.shape[1]
    assert list(finite[:4]) == [n_maps] * 4, "the responsive vertices were not fit"
    assert list(finite[4:]) == [1] * 4, "a screened-out vertex kept more than meanvol"
    # And the volume half still has its fit voxels, so screening did not take one
    # domain entirely -- which is what a per-domain threshold could do, and the
    # reason a shared rank over both is the thing being tested.
    vol = np.asarray(volume_out.dataobj)
    assert int((np.isfinite(vol[..., 0]) & (vol[..., 0] != 0)).sum()) == 24


def test_the_noise_pool_of_a_paired_fit_spans_both_domains(tmp_path):
    """-denoise picks its pool from the screening scores, which rank the whole
    location axis -- so a paired fit's nuisance components come from the volume as
    well as cortex. That is the gap a surface-only fit cannot close: its pool can
    only see what was sampled onto the mesh.
    """
    import nibabel as nib

    from fastfuncstuff.cli.pyrf import main
    from fastfuncstuff.io.gifti import save_gifti_data
    from tests.test_prf import _write_prf_dataset

    inputs, stimuli, _mask_path, _shape = _write_prf_dataset(tmp_path)
    surfaces = []
    for run, volume in enumerate(inputs):
        series = np.asarray(nib.load(volume).dataobj).reshape(-1, 120)
        path = tmp_path / f"d{run}.func.gii"
        save_gifti_data(
            path, series[:6].astype(np.float32),
            meta={"mesh_fingerprint": "V6_F2", "TR_seconds": "1"}, time_series=True,
        )  # fmt: skip
        surfaces.append(str(path))

    # No -mask on purpose: the pool needs at least 100 locations, and the point
    # being tested is that it draws them from the volume. The pool comes from the
    # screen, so no region has to be named -- which is why this works paired.
    assert (
        main(
            [
                "-input",
                *inputs,
                "-surface_input",
                *surfaces,
                "-stimulus",
                *stimuli,
                "-denoise",
                "-max_pcs",
                "2",
                "-prefix",
                str(tmp_path / "dn.nii.gz"),
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
    assert (tmp_path / "dn_volume.nii.gz").is_file()
    assert (tmp_path / "dn_surface.shape.gii").is_file()


def test_noise_pool_mask_is_refused_paired_and_says_what_to_use(tmp_path):
    """It names its region as a volume, so it has no spelling for "these
    vertices". The refusal has to point at -denoise, which needs no region."""
    from fastfuncstuff.cli.pyrf import main
    from tests.test_prf import _write_prf_dataset

    inputs, stimuli, mask_path, _shape = _write_prf_dataset(tmp_path)
    with pytest.raises(SystemExit):
        main([
            "-input", *inputs, "-surface_input", "a.func.gii", "b.func.gii",
            "-stimulus", *stimuli, "-mask", mask_path,
            "-denoise", "-noise_pool_mask", "all",
            "-prefix", str(tmp_path / "x"), "-device", "cpu", "-verb", "0",
        ])  # fmt: skip
