"""Header-only dataset introspection (ffs_info) and its AFNI parity.

The numbers here were checked against ``3dinfo`` on the same files; where a value
is AFNI-frame (origin, extent, signed steps) a sign error is silent and easy to
make, so each one is pinned.
"""

import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest

from fastfuncstuff.cli.info import build_parser, main
from fastfuncstuff.io.dsetinfo import (
    afni_orient_code,
    cardinal_affine,
    obliquity_deg,
    read_info,
    same_grid,
)


def _oblique_affine(deg: float = 15.0, zooms=(2.5, 2.5, 3.0), origin=(-40.0, -35.0, -18.0)):
    th = np.deg2rad(deg)
    rot = np.array(
        [[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]], dtype=np.float64
    )
    aff = np.eye(4)
    aff[:3, :3] = rot @ np.diag(zooms)
    aff[:3, 3] = origin
    return aff


@pytest.fixture
def epi(tmp_path):
    """A small oblique 4-D series with a TR, written to .nii / .nii.gz / .nii.zst."""
    data = np.random.default_rng(0).random((8, 7, 6, 5)).astype(np.float32)
    img = nib.Nifti1Image(data, _oblique_affine())
    img.header.set_xyzt_units(xyz="mm", t="sec")
    img.header["pixdim"][4] = 1.2
    path = tmp_path / "epi.nii"
    img.to_filename(str(path))
    return path


class TestGeometry:
    def test_orient_code_is_the_opposite_of_nibabel(self):
        # nibabel names the end each axis points to; AFNI names where it starts.
        assert afni_orient_code(np.diag([1.0, 1.0, 1.0, 1.0])) == "LPI"
        assert afni_orient_code(np.diag([-1.0, -1.0, 1.0, 1.0])) == "RAI"

    def test_obliquity_matches_the_rotation_applied(self):
        assert obliquity_deg(np.eye(4)) == pytest.approx(0.0)
        assert obliquity_deg(_oblique_affine(15.0)) == pytest.approx(15.0, abs=1e-6)

    def test_cardinal_affine_snaps_axes_and_keeps_origin(self):
        aff = _oblique_affine(15.0)
        card = cardinal_affine(aff)
        assert np.allclose(card[:3, 3], aff[:3, 3])
        # One non-zero per column, magnitude = the original voxel size.
        assert np.count_nonzero(card[:3, :3]) == 3
        assert np.allclose(np.abs(card[:3, :3]).sum(axis=0), [2.5, 2.5, 3.0])


class TestReadInfo:
    def test_basic_fields(self, epi):
        info = read_info(epi)
        assert info.exists and info.is_nifti
        assert info.shape == (8, 7, 6, 5)
        assert info.zooms == pytest.approx((2.5, 2.5, 3.0))
        assert info.tr == pytest.approx(1.2)
        assert info.datum == "float"
        assert info.orient == "LPI"
        assert info.is_oblique

    def test_afni_frame_quantities(self, epi):
        """Origin/steps/extent are AFNI-frame (x=Left+, y=Posterior+) — 3dinfo values."""
        info = read_info(epi)
        assert info.origin == pytest.approx((40.0, 35.0, -18.0))
        assert info.signed_steps == pytest.approx((-2.5, -2.5, 3.0))
        # Extent is the *cardinal* bounding box; for oblique data it is not the
        # tilted corner extremes.
        x0, x1, y0, y1, z0, z1 = info.extent
        assert (x0, x1) == pytest.approx((22.5, 40.0))
        assert (y0, y1) == pytest.approx((20.0, 35.0))
        assert (z0, z1) == pytest.approx((-18.0, -3.0))

    def test_selector_changes_the_volume_count(self, epi):
        assert read_info(f"{epi}[0..2]").shape[3] == 3
        assert read_info(f"{epi}[0]").shape[3] == 1
        assert read_info(f"{epi}[1..$]").shape[3] == 4

    def test_missing_file_is_reported_not_raised(self, tmp_path):
        info = read_info(tmp_path / "nope.nii.gz")
        assert not info.exists

    @pytest.mark.parametrize("suffix", [".gz", ".zst"])
    def test_compressed_headers_agree_with_plain(self, epi, suffix):
        """The whole point: a compressed header read must be identical, and partial."""
        out = epi.with_suffix(f".nii{suffix}")
        tool = ["gzip", "-kc"] if suffix == ".gz" else ["zstd", "-qc"]
        with open(out, "wb") as fh:
            if subprocess.run([*tool, str(epi)], stdout=fh).returncode:
                pytest.skip(f"{tool[0]} unavailable")
        plain, comp = read_info(epi), read_info(out)
        assert comp.shape == plain.shape
        assert comp.zooms == pytest.approx(plain.zooms)
        assert comp.extent == pytest.approx(plain.extent)
        assert comp.compression == ("gzip" if suffix == ".gz" else "zstd")

    @pytest.mark.parametrize("suffix", ["", ".gz", ".zst"])
    def test_extensions_past_the_first_read(self, tmp_path, suffix):
        """Labels beyond the first 64 KB peek still come back whole.

        A long bucket's AFNI extension can outrun the first read, and the
        second one that fetches the rest is the path a short header never takes.
        """
        from fastfuncstuff.io.afni import _set_afni_brick_labels

        labels = [f"cond{i:03d}_" + "x" * 300 + "#0_Coef" for i in range(250)]
        img = nib.Nifti1Image(np.zeros((4, 4, 4, len(labels)), np.float32), np.eye(4))
        _set_afni_brick_labels(img.header, labels)
        plain = tmp_path / "bucket.nii"
        img.to_filename(str(plain))
        assert nib.load(str(plain)).dataobj.offset > 1 << 16
        path = plain
        if suffix:
            path = plain.with_suffix(f".nii{suffix}")
            tool = ["gzip", "-kc"] if suffix == ".gz" else ["zstd", "-qc"]
            with open(path, "wb") as fh:
                if subprocess.run([*tool, str(plain)], stdout=fh).returncode:
                    pytest.skip(f"{tool[0]} unavailable")
        assert read_info(path).labels == labels

    def test_same_grid(self, epi, tmp_path):
        other = tmp_path / "shifted.nii"
        aff = _oblique_affine(origin=(0.0, 0.0, 0.0))
        nib.Nifti1Image(np.zeros((8, 7, 6), dtype=np.float32), aff).to_filename(str(other))
        assert same_grid([read_info(epi), read_info(epi)])
        assert not same_grid([read_info(epi), read_info(other)])


class TestReadVolume:
    """-vis draws one volume; it must not decompress the whole series to do it."""

    @pytest.mark.parametrize("suffix", ["", ".gz", ".zst"])
    @pytest.mark.parametrize("index", [0, 2, 4])
    def test_matches_a_full_load(self, epi, suffix, index):
        from fastfuncstuff.io.dsetinfo import read_volume

        path = epi
        if suffix:
            path = epi.with_suffix(f".nii{suffix}")
            tool = ["gzip", "-kc"] if suffix == ".gz" else ["zstd", "-qc"]
            with open(path, "wb") as fh:
                if subprocess.run([*tool, str(epi)], stdout=fh).returncode:
                    pytest.skip(f"{tool[0]} unavailable")
        expected = np.asanyarray(nib.load(str(epi)).dataobj)[..., index]
        got, info = read_volume(path, index)
        assert np.allclose(got, expected)
        assert info.shape[3] == 5

    def test_index_is_clamped_not_wrapped(self, epi):
        from fastfuncstuff.io.dsetinfo import read_volume

        last = np.asanyarray(nib.load(str(epi)).dataobj)[..., -1]
        assert np.allclose(read_volume(epi, 99)[0], last)


class TestIterVolumes:
    """The progressive reader the viewer inflates through: same voxels as a full load."""

    @pytest.mark.parametrize("suffix", ["", ".gz", ".zst"])
    @pytest.mark.parametrize("into", [False, True])
    def test_matches_a_full_load(self, epi, suffix, into):
        from fastfuncstuff.io.dsetinfo import iter_volumes

        path = epi
        if suffix:
            path = epi.with_suffix(f".nii{suffix}")
            tool = ["gzip", "-kc"] if suffix == ".gz" else ["zstd", "-qc"]
            with open(path, "wb") as fh:
                if subprocess.run([*tool, str(epi)], stdout=fh).returncode:
                    pytest.skip(f"{tool[0]} unavailable")
        expected = np.asanyarray(nib.load(str(epi)).dataobj)
        buf = np.empty(expected.shape[::-1], np.float32) if into else None
        got = [np.array(v) for v in iter_volumes(path, into=buf)]
        assert np.allclose(np.stack(got, axis=3), expected)
        if into:
            assert np.allclose(buf.transpose(3, 2, 1, 0), expected)

    def test_scaled_integers_are_scaled(self, tmp_path):
        from fastfuncstuff.io.dsetinfo import iter_volumes

        data = np.arange(4 * 3 * 2 * 3, dtype=np.int16).reshape(4, 3, 2, 3)
        img = nib.Nifti1Image(data, np.eye(4))
        img.header.set_slope_inter(0.5, 10.0)
        path = tmp_path / "scaled.nii"
        img.to_filename(str(path))
        buf = np.empty((3, 2, 3, 4), np.float32)
        got = np.stack([np.array(v) for v in iter_volumes(path, into=buf)], axis=3)
        assert np.allclose(got, data * 0.5 + 10.0)

    def test_a_truncated_file_raises(self, epi, tmp_path):
        from fastfuncstuff.io.dsetinfo import iter_volumes

        cut = tmp_path / "cut.nii"
        cut.write_bytes(epi.read_bytes()[:-100])
        with pytest.raises(ValueError, match="Truncated"):
            list(iter_volumes(cut))


class TestCLI:
    def test_value_flags_print_in_command_line_order(self, epi, capsys):
        assert main(["-nk", "-ni", "-tr", str(epi)]) == 0
        assert capsys.readouterr().out.strip() == "6\t8\t1.200000"

    def test_missing_dataset_prints_no_dset_and_exits_nonzero(self, tmp_path, capsys):
        assert main(["-nv", str(tmp_path / "absent.nii")]) == 1
        assert capsys.readouterr().out.strip() == "NO-DSET"

    def test_report_mentions_the_key_geometry(self, epi, capsys):
        assert main([str(epi)]) == 0
        out = capsys.readouterr().out
        assert "8 × 7 × 6" in out
        assert "LPI" in out
        assert "oblique" in out

    def test_dash_and_underscore_flag_spellings_agree(self, epi, capsys):
        main(["-is_oblique", str(epi)])
        first = capsys.readouterr().out
        main(["-is-oblique", str(epi)])
        assert capsys.readouterr().out == first

    def test_help_builds(self):
        assert build_parser().format_help()


def test_header_read_does_not_import_torch():
    """Regression: ffs_info took 6 s because every import pulled in the torch stack.

    The header path must stay torch-free — this test is the tripwire for anything
    that re-adds a heavyweight import to io.headers / io.dsetinfo / cli.info.
    """
    code = "import sys, fastfuncstuff.cli.info; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stderr


def test_a_gifti_reports_as_a_surface_and_keeps_its_history(tmp_path, capsys):
    """ffs_info on a .func.gii fell into the voxel report and died on its '???'
    orientation; and gii outputs carried no provenance at all."""
    from fastfuncstuff.io.gifti import (
        gifti_as_nifti,
        save_gifti_data,
        save_gifti_surface,
        save_nifti_data_as_gifti,
    )

    rng = np.random.default_rng(0)
    series = tmp_path / "run.lh.func.gii"
    save_gifti_data(series, rng.normal(size=(50, 7)), meta={"TR_seconds": "2", "mesh": "ico"})
    info = read_info(series)
    assert info.surface_kind == "series" and info.shape == (50, 1, 1, 7) and info.tr == 2.0
    assert info.history.count("\n") == 0 and "[" in info.history  # one entry: this save

    # A tool reading it and writing a map continues the trail.
    img = gifti_as_nifti(series)
    out = tmp_path / "mean.lh.shape.gii"
    save_nifti_data_as_gifti(out, np.asarray(img.dataobj)[..., :1], header=img.header)
    assert read_info(out).history.count("\n") == 1

    mesh = tmp_path / "lh.white.surf.gii"
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    save_gifti_surface(mesh, rng.normal(size=(4, 3)), faces)
    m = read_info(mesh)
    assert m.surface_kind == "mesh" and m.shape[0] == 4 and m.n_faces == 2

    assert main([str(series), str(out), str(mesh)]) == 0
    text = capsys.readouterr().out
    assert "50 vertices × 7 time points" in text and "TR 2 s" in text
    assert "50 vertices × 1 map" in text and "4 vertices · 2 triangles" in text
    assert "history" in text and "orient" not in text


def test_info_id_flag_reports_header_idcode(tmp_path, capsys):
    # -id is 3dinfo's IDCODE query; empty (not a made-up code) when the header has none.
    from fastfuncstuff.io.afni import set_afni_atr

    bare = nib.Nifti1Image(np.zeros((3, 3, 3), np.float32), np.eye(4))
    nib.save(bare, tmp_path / "bare.nii.gz")
    tagged = nib.Nifti1Image(np.zeros((3, 3, 3), np.float32), np.eye(4))
    set_afni_atr(tagged.header, "X", "y")
    nib.save(tagged, tmp_path / "tagged.nii.gz")

    assert read_info(tmp_path / "bare.nii.gz").idcode == ""
    idc = read_info(tmp_path / "tagged.nii.gz").idcode
    assert idc.startswith("AFN_")

    main(["-id", str(tmp_path / "tagged.nii.gz")])
    assert capsys.readouterr().out.strip() == idc
