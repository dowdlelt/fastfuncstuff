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
