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
        assert data.shape == EXPECTED_SHAPE, (
            f"Expected shape {EXPECTED_SHAPE}, got {data.shape}"
        )

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
        assert data.shape == EXPECTED_SHAPE, (
            f"Expected shape {EXPECTED_SHAPE}, got {data.shape}"
        )

        # Check values (all should be 1.0)
        assert np.allclose(data, EXPECTED_VALUE), "Expected all values to be 1.0"

    def test_afni_nifti_equivalence(self):
        """Test that AFNI and NIfTI versions have same data."""
        if not (AFNI_ALL_ONES_HEAD.exists() and AFNI_ALL_ONES_BRIK.exists() and NIFTI_ALL_ONES.exists()):
            pytest.skip("Synthetic data not found")

        afni_img = nib.load(str(AFNI_ALL_ONES_HEAD))
        afni_data = afni_img.get_fdata()

        nifti_img = nib.load(str(NIFTI_ALL_ONES))
        nifti_data = nifti_img.get_fdata()

        # Data should be identical
        assert afni_data.shape == nifti_data.shape, (
            "AFNI and NIfTI data shapes should match"
        )
        assert np.allclose(afni_data, nifti_data), (
            "AFNI and NIfTI data values should match"
        )


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
        assert np.allclose(read_data, original_data, rtol=1e-5), (
            "Values should be preserved"
        )

        # Check affine preserved
        assert np.allclose(read_affine, original_affine, rtol=1e-5), (
            "Affine should be preserved"
        )


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
        assert read_data.shape == data.shape, (
            "Long time series shape should be preserved"
        )
        assert np.allclose(read_data, data, rtol=1e-5), (
            "Long time series values should be preserved"
        )


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

        assert data_2d.shape == (n_timepoints, n_voxels), (
            "Should reshape correctly for GLM"
        )
        assert data_2d.dtype in [np.float32, np.float64], "Should be float type for GLM"
        assert not np.any(np.isnan(data_2d)), "Should not contain NaN values"
        assert not np.any(np.isinf(data_2d)), "Should not contain inf values"
