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
