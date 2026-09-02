"""
Tests for AFNI I/O and NIfTI handling using synthetic data.

Uses the synthetic_data/ folder with known test datasets:
- afni_all_ones+orig: 64x64x16x40 AFNI format, all values = 1
- afni_nii_all_ones.nii.gz: Same data in NIfTI format
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

# Locate synthetic data directory
SYNTHETIC_DATA_DIR = Path(__file__).parent.parent / "synthetic_data"
AFNI_ALL_ONES_HEAD = SYNTHETIC_DATA_DIR / "afni_all_ones+orig.HEAD"
AFNI_ALL_ONES_BRIK = SYNTHETIC_DATA_DIR / "afni_all_ones+orig.BRIK.gz"
NIFTI_ALL_ONES = SYNTHETIC_DATA_DIR / "afni_nii_all_ones.nii.gz"

# Expected properties of synthetic data
EXPECTED_SHAPE = (64, 64, 16, 40)
EXPECTED_VALUE = 1.0


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir()
    return output_dir


class TestNIfTIReading:
    """Test reading NIfTI format files using nibabel."""

    def test_read_nifti_all_ones(self):
        """Test reading the synthetic all-ones NIfTI dataset."""
        if not NIFTI_ALL_ONES.exists():
            pytest.skip("Synthetic NIfTI data not found")

        img = nib.load(str(NIFTI_ALL_ONES))
        data = img.get_fdata()
        affine = img.affine

        # Check shape
        assert data.shape == EXPECTED_SHAPE, f"Expected shape {EXPECTED_SHAPE}, got {data.shape}"

        # Check values (all should be 1.0)
        assert np.allclose(data, EXPECTED_VALUE), "Expected all values to be 1.0"

        # Check affine
        assert affine.shape == (4, 4), f"Affine should be 4x4, got {affine.shape}"

    def test_read_nifti_returns_numpy_array(self):
        """Test that nibabel returns numpy arrays."""
        if not NIFTI_ALL_ONES.exists():
            pytest.skip("Synthetic NIfTI data not found")

        img = nib.load(str(NIFTI_ALL_ONES))
        data = img.get_fdata()
        affine = img.affine

        assert isinstance(data, np.ndarray), "Data should be numpy array"
        assert isinstance(affine, np.ndarray), "Affine should be numpy array"


class TestAFNIReading:
    """Test reading AFNI format files using nibabel."""

    def test_read_afni_all_ones(self):
        """Test reading the synthetic all-ones AFNI dataset."""
        if not (AFNI_ALL_ONES_HEAD.exists() and AFNI_ALL_ONES_BRIK.exists()):
            pytest.skip("Synthetic AFNI data not found")

        # nibabel can read AFNI format too
        img = nib.load(str(AFNI_ALL_ONES_HEAD))
        data = img.get_fdata()

        # Check shape
        assert data.shape == EXPECTED_SHAPE, f"Expected shape {EXPECTED_SHAPE}, got {data.shape}"

        # Check values (all should be 1.0)
        assert np.allclose(data, EXPECTED_VALUE), "Expected all values to be 1.0"

    def test_afni_nifti_equivalence(self):
        """Test that AFNI and NIfTI versions have same data."""
        if not (
            AFNI_ALL_ONES_HEAD.exists() and AFNI_ALL_ONES_BRIK.exists() and NIFTI_ALL_ONES.exists()
        ):
            pytest.skip("Synthetic data not found")

        afni_img = nib.load(str(AFNI_ALL_ONES_HEAD))
        afni_data = afni_img.get_fdata()

        nifti_img = nib.load(str(NIFTI_ALL_ONES))
        nifti_data = nifti_img.get_fdata()

        # Data should be identical
        assert afni_data.shape == nifti_data.shape, "AFNI and NIfTI data shapes should match"
        assert np.allclose(afni_data, nifti_data), "AFNI and NIfTI data values should match"


class TestNIfTIWriting:
    """Test writing NIfTI format files."""

    def test_write_nifti_simple(self, temp_output_dir):
        """Test writing a simple NIfTI dataset."""
        data = np.ones((10, 10, 5, 20), dtype=np.float32)
        affine = np.eye(4)

        output_path = temp_output_dir / "test_output.nii.gz"
        img = nib.Nifti1Image(data, affine)
        nib.save(img, str(output_path))

        # Verify file was created
        assert output_path.exists(), "NIfTI file should be created"

    def test_write_read_roundtrip_nifti(self, temp_output_dir):
        """Test that writing and reading NIfTI data preserves values."""
        np.random.seed(42)
        original_data = np.random.randn(8, 8, 4, 10).astype(np.float32)
        original_affine = np.eye(4)
        original_affine[:3, 3] = [10, 20, 30]

        output_path = temp_output_dir / "roundtrip_test.nii.gz"

        # Write data
        img = nib.Nifti1Image(original_data, original_affine)
        nib.save(img, str(output_path))

        # Read it back
        read_img = nib.load(str(output_path))
        read_data = read_img.get_fdata()
        read_affine = read_img.affine

        # Check shape preserved
        assert read_data.shape == original_data.shape, "Shape should be preserved"

        # Check values preserved
        assert np.allclose(read_data, original_data, rtol=1e-5), "Values should be preserved"

        # Check affine preserved
        assert np.allclose(read_affine, original_affine, rtol=1e-5), "Affine should be preserved"


class TestZstRoundtrip:
    """`.nii.zst` (zstd-compressed) is a third supported format alongside
    `.nii`/`.nii.gz` — big intermediates read many times use it instead of gzip.
    Must round-trip through the shared save_nifti/load_nifti, same as gzip."""

    def test_save_load_roundtrip(self, temp_output_dir):
        import shutil

        if shutil.which("zstd") is None:
            pytest.skip("zstd not on PATH")
        from fastfuncstuff.io.afni import load_nifti, save_nifti

        np.random.seed(0)
        data = np.random.randn(6, 6, 4, 5).astype(np.float32)
        affine = np.eye(4)
        affine[:3, 3] = [1, 2, 3]

        out_path = temp_output_dir / "roundtrip.nii.zst"
        save_nifti(data, str(out_path), affine=affine)
        assert out_path.exists()

        img = load_nifti(str(out_path))
        assert np.allclose(img.get_fdata(), data, rtol=1e-5)
        assert np.allclose(img.affine, affine)

    def test_decode_command_uses_only_its_cpu_share(self, monkeypatch):
        from fastfuncstuff.io import afni

        monkeypatch.setattr(afni.shutil, "which", lambda name: f"/usr/bin/{name}")
        assert afni._zstd_decode_command("big.nii.zst", 3) == [
            "pzstd",
            "-q",
            "-d",
            "-p",
            "3",
            "-c",
            "big.nii.zst",
        ]

    def test_decode_command_falls_back_to_stock_zstd(self, monkeypatch):
        from fastfuncstuff.io import afni

        monkeypatch.setattr(afni.shutil, "which", lambda _name: None)
        assert afni._zstd_decode_command("big.nii.zst", 8) == [
            "zstd",
            "-dc",
            "big.nii.zst",
        ]
        assert afni._zstd_decode_command("big.nii.zst", 1) == [
            "zstd",
            "-dc",
            "big.nii.zst",
        ]

    def test_broken_pzstd_retries_stock_decoder(self, monkeypatch):
        import io
        import subprocess

        from fastfuncstuff.io import afni

        monkeypatch.setattr(afni.shutil, "which", lambda name: f"/usr/bin/{name}")
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if len(calls) == 1:
                kwargs["stdout"].write(b"partial")
                raise subprocess.CalledProcessError(1, cmd)
            assert kwargs["stdout"].tell() == 0
            kwargs["stdout"].write(b"decoded")

        monkeypatch.setattr(afni.subprocess, "run", fake_run)
        output = io.BytesIO()
        afni._run_zstd_decode("big.nii.zst", output, 4)
        assert calls[0][0] == "pzstd"
        assert calls[1] == ["zstd", "-dc", "big.nii.zst"]
        assert output.getvalue() == b"decoded"

    def test_replace_afni_extension_roundtrips_zst(self):
        from fastfuncstuff.io.afni import replace_afni_extension

        assert replace_afni_extension("stats.nii.zst", ".nii.gz") == "stats.nii.gz"
        assert replace_afni_extension("stats", ".nii.zst") == "stats.nii.zst"

    def test_parse_prefix_recognizes_zst(self):
        from fastfuncstuff.cli_utils import parse_prefix

        pfx = parse_prefix("out.nii.zst")
        assert pfx.stem == "out"
        assert pfx.nifti_ext == ".nii.zst"
        assert pfx.as_file() == "out.nii.zst"


class TestCheapHeaderShape:
    """`nifti_shape` / `read_nifti_header` must return the same shape and header
    dims as a full `load_nifti`, across `.nii`/`.nii.gz`/`.nii.zst`, *without*
    decompressing the payload. This is the hot-path fix: run-structure timing and
    the mask/data grid check only need dims, and a `.nii.zst` full-decompress just
    to read one integer was the source of a very slow 'Computing run structure'."""

    def _write_all_formats(self, out_dir, data, affine):
        import shutil

        from fastfuncstuff.io.afni import save_nifti

        # Distinct stems per format: pigz compresses `foo.nii` -> `foo.nii.gz`
        # in place, so a same-stem `.nii` and `.nii.gz` would clobber each other.
        paths = {
            "nii": out_dir / "plain.nii",
            "gz": out_dir / "gzipped.nii.gz",
        }
        save_nifti(data, str(paths["nii"]), affine=affine)
        save_nifti(data, str(paths["gz"]), affine=affine)
        if shutil.which("zstd") is not None:
            paths["zst"] = out_dir / "zstd.nii.zst"
            save_nifti(data, str(paths["zst"]), affine=affine)
        return paths

    def test_shape_matches_full_load(self, temp_output_dir):
        from fastfuncstuff.io.afni import load_nifti, nifti_shape

        data = np.random.default_rng(1).standard_normal((7, 6, 5, 9)).astype(np.float32)
        affine = np.eye(4)
        paths = self._write_all_formats(temp_output_dir, data, affine)

        for fmt, p in paths.items():
            assert nifti_shape(str(p)) == data.shape, fmt
            # Cheap header dims must agree with a full load's header.
            from fastfuncstuff.io.afni import read_nifti_header

            assert np.array_equal(
                read_nifti_header(str(p))["dim"], load_nifti(str(p)).header["dim"]
            ), fmt

    def test_shape_honours_subbrick_selector(self, temp_output_dir):
        from fastfuncstuff.io.afni import nifti_shape

        data = np.random.default_rng(2).standard_normal((7, 6, 5, 12)).astype(np.float32)
        paths = self._write_all_formats(temp_output_dir, data, np.eye(4))

        for p in paths.values():
            assert nifti_shape(f"{p}[0..9]") == (7, 6, 5, 10)
            assert nifti_shape(f"{p}[1..$]") == (7, 6, 5, 11)


class TestLargeWriteChunking:
    """save_nifti chunks large writes so a single >2 GiB write() can't crash the save.

    A 5-D per-frame warp slice (nx·ny·nz·T·3) can exceed 2 GiB, and some Python builds /
    filesystems reject a single write() that big with OSError(EINVAL). We force a tiny
    chunk to exercise the loop on a small array and confirm exact round-trip.
    """

    def test_chunked_write_roundtrips(self, temp_output_dir, monkeypatch):
        from fastfuncstuff.io import afni
        from fastfuncstuff.io.afni import load_nifti, save_nifti

        monkeypatch.setattr(afni._ChunkedFileWriter, "_CHUNK", 4096)  # tiny → many chunks
        data = np.random.default_rng(0).standard_normal((10, 8, 6, 5, 3)).astype(np.float32)
        affine = np.eye(4)
        affine[:3, 3] = [2, 3, 4]

        for name in ("chunk_direct.nii", "chunk_pigz.nii.gz"):
            out = temp_output_dir / name
            save_nifti(data, str(out), affine=affine)
            assert out.exists()
            img = load_nifti(str(out))
            assert np.array_equal(np.asarray(img.get_fdata(dtype=np.float32)), data)
            assert np.array_equal(img.affine, affine)


class TestDataTypeHandling:
    """Test handling of different data types."""

    def test_float32_data(self, temp_output_dir):
        """Test writing float32 data."""
        data = np.random.randn(5, 5, 3, 4).astype(np.float32)
        affine = np.eye(4)

        output_path = temp_output_dir / "float32_test.nii.gz"
        img = nib.Nifti1Image(data, affine)
        nib.save(img, str(output_path))

        read_img = nib.load(str(output_path))
        read_data = read_img.get_fdata()
        assert read_data.dtype in [np.float32, np.float64], "Should preserve float type"

    def test_float64_data(self, temp_output_dir):
        """Test writing float64 data."""
        data = np.random.randn(5, 5, 3, 4).astype(np.float64)
        affine = np.eye(4)

        output_path = temp_output_dir / "float64_test.nii.gz"
        img = nib.Nifti1Image(data, affine)
        nib.save(img, str(output_path))

        read_img = nib.load(str(output_path))
        read_data = read_img.get_fdata()
        assert read_data.dtype in [np.float32, np.float64], "Should preserve float type"

    def test_float32_not_quantized_by_int_header(self, temp_output_dir):
        """save_nifti must write float32 data as float32 even when the supplied
        header was copied from an int16 input -- otherwise the result is silently
        quantized to int16 and the AFNI extension's NIfTI_nums datatype no longer
        matches the file (AFNI's 'dimensions altered' warning)."""
        import re

        from fastfuncstuff.io.afni import _NIFTI_ECODE_AFNI, save_nifti

        hdr = nib.Nifti1Header()
        hdr.set_data_dtype(np.int16)
        xml = (
            '<?xml version="1.0" ?>\n<AFNI_attributes\n  self_idcode="AFN_old"\n'
            '  NIfTI_nums="6,7,8,1,1,4"\n  ni_form="ni_group" >\n</AFNI_attributes>\n'
        )
        hdr.extensions.append(nib.nifti1.Nifti1Extension(_NIFTI_ECODE_AFNI, xml.encode()))

        data = np.random.rand(6, 7, 8).astype(np.float32)
        out = temp_output_dir / "dtype_sync.nii.gz"
        save_nifti(data, str(out), affine=np.eye(4), header=hdr)

        img = nib.load(str(out))
        assert int(img.header["datatype"]) == 16, "on-disk dtype should be float32"
        assert np.allclose(np.asarray(img.dataobj), data, atol=1e-5), "data must not be quantized"
        ext = next(e for e in img.header.extensions if e.get_code() == _NIFTI_ECODE_AFNI)
        nums = re.search(r'NIfTI_nums="([^"]*)"', ext.content.decode()).group(1)
        assert nums.endswith(",16"), f"NIfTI_nums datatype should match float32: {nums}"

    def test_brick_labels_and_stataux(self, temp_output_dir):
        """save_nifti tags sub-bricks: BRICK_LABS names them and BRICK_STATAUX /
        BRICK_STATSYM mark the t sub-brick as a stat AFNI can threshold. Creates a
        minimal AFNI extension for a plain (non-AFNI) input."""
        from fastfuncstuff.io.afni import _NIFTI_ECODE_AFNI, save_nifti

        data = np.random.rand(5, 5, 2, 3).astype(np.float32)
        out = temp_output_dir / "tagged.nii.gz"
        save_nifti(
            data,
            str(out),
            affine=np.eye(4),
            brick_labels=["r", "tstat", "1-q"],
            brick_stataux={1: (3, (35.0,))},  # AFNI code 3 = Ttest, 1 param = dof
        )
        img = nib.load(str(out))
        xml = "".join(
            e.get_content().decode("utf-8", "ignore")
            if isinstance(e.get_content(), bytes)
            else e.get_content()
            for e in (img.header.extensions or [])
            if e.get_code() == _NIFTI_ECODE_AFNI
        )
        assert "BRICK_LABS" in xml and "r~tstat~1-q" in xml
        assert "BRICK_STATAUX" in xml
        assert "none;Ttest(35);none" in xml  # only sub-brick 1 is a stat

    def test_geometry_attributes_dropped_from_extension(self, temp_output_dir):
        """A crop/regrid changes the affine, but the AFNI extension copied from the
        input still describes the input grid. Since NIfTI_nums is updated, AFNI
        trusts the extension and applies its IJK_TO_DICOM_REAL over the sform --
        which moved ffs_util_autobox output ~37mm from its own input on deoblique.
        The geometry attributes must not survive the write."""
        from fastfuncstuff.io.afni import _NIFTI_ECODE_AFNI, save_nifti

        geom = "".join(
            f'<AFNI_atr\n  ni_type="float"\n  ni_dimen="3"\n  atr_name="{name}" >\n'
            " 1.0\n 2.0\n 3.0\n</AFNI_atr>\n"
            for name in ("ORIGIN", "DELTA", "IJK_TO_DICOM", "IJK_TO_DICOM_REAL")
        )
        xml = (
            '<?xml version="1.0" ?>\n<AFNI_attributes\n  self_idcode="AFN_old"\n'
            '  NIfTI_nums="9,9,9,1,1,16"\n  ni_form="ni_group" >\n'
            f"{geom}</AFNI_attributes>\n"
        )
        hdr = nib.Nifti1Header()
        hdr.extensions.append(nib.nifti1.Nifti1Extension(_NIFTI_ECODE_AFNI, xml.encode()))

        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        affine[:3, 3] = [-40.0, 30.0, -20.0]
        out = temp_output_dir / "cropped.nii.gz"
        save_nifti(np.random.rand(5, 5, 5).astype(np.float32), str(out), affine=affine, header=hdr)

        img = nib.load(str(out))
        ext = next(e for e in img.header.extensions if e.get_code() == _NIFTI_ECODE_AFNI)
        content = ext.content.decode()
        for name in ("ORIGIN", "DELTA", "IJK_TO_DICOM", "IJK_TO_DICOM_REAL"):
            assert f'atr_name="{name}"' not in content, f"stale {name} survived the write"
        assert np.allclose(img.affine, affine), "sform must carry the new geometry"

    def test_regrid_keeps_both_qform_and_sform(self, temp_output_dir):
        """A changed affine must land in BOTH forms, with the input's space codes.

        nibabel rewrites the s/qform only when the affine differs from the
        header's -- and then writes sform_code=2 with qform_code=0, i.e. no qform
        at all. Readers that follow the NIfTI precedence rule (qform first) fall
        back to pixdim-only geometry and display the dataset rotated and offset,
        which is what a cropped anat did while 3dinfo still read it correctly."""
        from fastfuncstuff.io.afni import save_nifti

        src_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        hdr = nib.Nifti1Header()
        hdr.set_sform(src_affine, code=1)
        hdr.set_qform(src_affine, code=1)

        # A crop: same voxel lattice, origin walked to the new corner.
        affine = src_affine.copy()
        affine[:3, 3] = [-40.0, 30.0, -20.0]
        out = temp_output_dir / "regridded.nii.gz"
        save_nifti(np.random.rand(5, 5, 5).astype(np.float32), str(out), affine=affine, header=hdr)

        img = nib.load(str(out))
        assert int(img.header["qform_code"]) == 1
        assert int(img.header["sform_code"]) == 1
        assert np.allclose(img.get_sform(), affine)
        assert np.allclose(img.get_qform(), affine)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_read_nonexistent_file(self):
        """Test that reading nonexistent file raises appropriate error."""
        with pytest.raises((FileNotFoundError, Exception)):
            nib.load("/nonexistent/path/to/file.nii.gz")

    def test_single_volume_data(self, temp_output_dir):
        """Test handling of single volume (3D) data."""
        data = np.random.randn(10, 10, 5).astype(np.float32)
        affine = np.eye(4)

        output_path = temp_output_dir / "single_volume.nii.gz"
        img = nib.Nifti1Image(data, affine)
        nib.save(img, str(output_path))

        read_img = nib.load(str(output_path))
        read_data = read_img.get_fdata()
        assert read_data.shape[:3] == data.shape[:3], "3D shape should be preserved"

    def test_large_time_series(self, temp_output_dir):
        """Test handling of large time series."""
        # Smaller spatial dimensions, more timepoints
        data = np.random.randn(8, 8, 4, 200).astype(np.float32)
        affine = np.eye(4)

        output_path = temp_output_dir / "long_timeseries.nii.gz"
        img = nib.Nifti1Image(data, affine)
        nib.save(img, str(output_path))

        read_img = nib.load(str(output_path))
        read_data = read_img.get_fdata()
        assert read_data.shape == data.shape, "Long time series shape should be preserved"
        assert np.allclose(read_data, data, rtol=1e-5), (
            "Long time series values should be preserved"
        )


class TestLoadAndConcatenateRuns:
    """Cover the preallocation and mask streaming paths of load_and_concatenate_runs."""

    @staticmethod
    def _write_runs(tmp_path, shapes):
        """Write a list of 4D runs with distinct known values; return paths + arrays."""
        paths, arrays = [], []
        for i, (nx, ny, nz, nt) in enumerate(shapes):
            arr = (np.arange(nx * ny * nz * nt, dtype=np.float32) + i * 1000.0).reshape(
                nx, ny, nz, nt
            )
            p = tmp_path / f"run{i:02d}.nii.gz"
            nib.save(nib.Nifti1Image(arr, np.eye(4)), str(p))
            paths.append(str(p))
            arrays.append(arr)
        return paths, arrays

    def test_prealloc_matches_cat(self, tmp_path):
        """total_timepoints (single-copy) path is byte-identical to the list+cat path."""
        from fastfuncstuff.io.afni import load_and_concatenate_runs

        shapes = [(6, 5, 4, 7), (6, 5, 4, 9), (6, 5, 4, 5)]
        paths, arrays = self._write_runs(tmp_path, shapes)
        total = sum(s[3] for s in shapes)
        n_vox = 6 * 5 * 4
        expected = np.concatenate(
            [a.reshape(n_vox, s[3]) for a, s in zip(arrays, shapes, strict=True)], axis=1
        )

        data_cat, rs_cat = load_and_concatenate_runs(paths, keep_on_cpu=True)
        data_pre, rs_pre = load_and_concatenate_runs(
            paths, keep_on_cpu=True, total_timepoints=total
        )

        assert rs_cat == rs_pre == [0, 7, 16]
        assert data_pre.shape == (n_vox, total)
        assert np.array_equal(data_pre.numpy(), expected)
        assert np.array_equal(data_pre.numpy(), data_cat.numpy())

    def test_mask_streamed_during_load(self, tmp_path):
        """mask_flat drops out-of-mask voxels before storage; result matches post-mask."""
        from fastfuncstuff.io.afni import load_and_concatenate_runs

        shapes = [(6, 5, 4, 7), (6, 5, 4, 9)]
        paths, arrays = self._write_runs(tmp_path, shapes)
        total = sum(s[3] for s in shapes)
        n_vox = 6 * 5 * 4
        mask = np.zeros(n_vox, dtype=bool)
        mask[::3] = True  # keep every third voxel

        full = np.concatenate(
            [a.reshape(n_vox, s[3]) for a, s in zip(arrays, shapes, strict=True)], axis=1
        )
        data, _ = load_and_concatenate_runs(
            paths, keep_on_cpu=True, mask_flat=mask, total_timepoints=total
        )
        assert data.shape == (int(mask.sum()), total)
        assert np.array_equal(data.numpy(), full[mask])

    def test_wrong_total_timepoints_raises(self, tmp_path):
        """A total_timepoints that disagrees with the run lengths is caught, not silently wrong."""
        from fastfuncstuff.io.afni import load_and_concatenate_runs

        paths, _ = self._write_runs(tmp_path, [(4, 4, 3, 8), (4, 4, 3, 8)])
        with pytest.raises(ValueError, match="total_timepoints"):
            load_and_concatenate_runs(paths, keep_on_cpu=True, total_timepoints=20)


class TestSyntheticDataProperties:
    """Test properties of the provided synthetic data."""

    def test_synthetic_data_statistics(self):
        """Verify synthetic data has expected statistical properties."""
        if not NIFTI_ALL_ONES.exists():
            pytest.skip("Synthetic NIfTI data not found")

        img = nib.load(str(NIFTI_ALL_ONES))
        data = img.get_fdata()

        # All ones dataset should have:
        assert np.isclose(data.mean(), EXPECTED_VALUE), "Mean should be 1.0"
        assert np.isclose(data.std(), 0.0), "Std should be 0.0 (all same value)"
        assert np.isclose(data.min(), EXPECTED_VALUE), "Min should be 1.0"
        assert np.isclose(data.max(), EXPECTED_VALUE), "Max should be 1.0"

    def test_synthetic_data_dimensions(self):
        """Verify synthetic data has correct dimensions for typical fMRI."""
        if not NIFTI_ALL_ONES.exists():
            pytest.skip("Synthetic NIfTI data not found")

        img = nib.load(str(NIFTI_ALL_ONES))
        data = img.get_fdata()
        _affine = img.affine

        # Check it's 4D
        assert len(data.shape) == 4, "Should be 4D (x, y, z, time)"

        # Check spatial dimensions are reasonable
        x, y, z, t = data.shape
        assert x > 0 and y > 0 and z > 0, "Spatial dimensions should be positive"
        assert t > 0, "Time dimension should be positive"

        # Check it's the expected size
        assert (x, y, z, t) == EXPECTED_SHAPE, f"Should be {EXPECTED_SHAPE}"

    def test_synthetic_data_as_glm_input(self):
        """Test that synthetic data can be used as GLM input (correct shape/type)."""
        if not NIFTI_ALL_ONES.exists():
            pytest.skip("Synthetic NIfTI data not found")

        img = nib.load(str(NIFTI_ALL_ONES))
        data = img.get_fdata()

        # Reshape to 2D for GLM: (n_timepoints, n_voxels)
        n_timepoints = data.shape[3]
        n_voxels = np.prod(data.shape[:3])

        data_2d = data.reshape(-1, n_timepoints).T

        assert data_2d.shape == (n_timepoints, n_voxels), "Should reshape correctly for GLM"
        assert data_2d.dtype in [np.float32, np.float64], "Should be float type for GLM"
        assert not np.any(np.isnan(data_2d)), "Should not contain NaN values"
        assert not np.any(np.isinf(data_2d)), "Should not contain inf values"


class TestGzipCompression:
    """The .nii.gz write path: level, thread cap, and temp-file hygiene."""

    def test_default_level_matches_nibabel(self):
        """Level 6 cost 4x the time of level 1 for 9% of the size, and the
        no-pigz fallback in the same function already wrote level 1."""
        from nibabel.openers import Opener

        from fastfuncstuff.io.afni import _gzip_level

        assert _gzip_level() == Opener.default_compresslevel == 1

    def test_env_override(self, monkeypatch):
        from fastfuncstuff.io.afni import _gzip_level

        monkeypatch.setenv("FFS_GZIP_LEVEL", "9")
        assert _gzip_level() == 9
        monkeypatch.setenv("FFS_GZIP_LEVEL", "not a number")
        assert _gzip_level() == 1
        monkeypatch.setenv("FFS_GZIP_LEVEL", "42")
        assert _gzip_level() == 9  # clamped

    def test_output_is_readable_gzip(self, tmp_path):
        from fastfuncstuff.io.afni import save_nifti

        data = np.arange(4 * 5 * 6 * 3, dtype=np.float32).reshape(4, 5, 6, 3)
        path = tmp_path / "out.nii.gz"
        save_nifti(data, str(path), affine=np.eye(4))
        with open(path, "rb") as handle:
            assert handle.read(2) == b"\x1f\x8b"
        assert np.array_equal(np.asarray(nib.load(str(path)).dataobj, dtype=np.float32), data)

    def test_no_temp_file_is_left_behind(self, tmp_path):
        from fastfuncstuff.io.afni import save_nifti

        data = np.zeros((3, 3, 3), dtype=np.float32)
        save_nifti(data, str(tmp_path / "out.nii.gz"), affine=np.eye(4))
        assert sorted(p.name for p in tmp_path.iterdir()) == ["out.nii.gz"]

    def test_remove_original_false_keeps_the_source(self, tmp_path):
        """pigz deletes its input unless told otherwise; the flag used to be
        ignored on that path, so a caller passing False lost the file."""
        from fastfuncstuff.io.afni import _compress_gz

        src = tmp_path / "keep.nii"
        src.write_bytes(b"not really a nifti, but it compresses")
        dst = tmp_path / "keep.nii.gz"
        _compress_gz(src, dst, remove_original=False)
        assert src.exists()
        assert dst.exists()
