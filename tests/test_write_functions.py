"""
Comprehensive tests for file writing functions used in analyze_taskforce_ses02_clean.py

Tests coverage for:
- write_glm_bucket_as_nifti() - Writing GLM results to AFNI-style bucket files
- write_ols_arma_comparison() - Side-by-side OLS vs ARMA comparison
- save_arma_rvar() - Saving ARMA parameters for reuse
- slice_glm_results() - Slicing results by regressor indices

These are the critical functions that were previously untested and could cause
the analysis script to fail during file writing (Steps 3-5).
"""

import json
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

import fastfuncsim as ffs
from fastfuncsim.glm_core import fit_glm
from fastfuncsim.arma_glm import fit_glm_arma11
from fastfuncsim.glm_outputs import (
    slice_glm_results,
    write_glm_bucket_as_nifti,
    write_ols_arma_comparison,
)
from fastfuncsim.arma_glm import save_arma_rvar, load_arma_params


# =============================================================================
# Fixtures for test data
# =============================================================================


@pytest.fixture
def device():
    """Get available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture
def simple_glm_results(device):
    """Create simple OLS GLM results for testing."""
    torch.manual_seed(42)

    # Small dataset
    n_timepoints = 100
    n_voxels = 1000
    n_regressors = 4

    # Generate data
    data = torch.randn(n_voxels, n_timepoints, device=device)
    design = torch.randn(n_timepoints, n_regressors, device=device)

    # Fit GLM
    results = fit_glm(data, design, tr=2.0, verbose=False, device=device)

    # Add spatial metadata
    results.full_shape = (10, 10, 10)
    results.voxel_mask = torch.ones(n_voxels, dtype=torch.bool, device=device)
    results.affine = np.eye(4)

    return results


@pytest.fixture
def simple_arma_results(device):
    """Create simple ARMA(1,1) GLM results for testing."""
    torch.manual_seed(42)

    # Small dataset
    n_timepoints = 100
    n_voxels = 500  # Smaller for ARMA (slower)
    n_regressors = 4

    # Generate data with autocorrelation
    data = torch.randn(n_voxels, n_timepoints, device=device)
    design = torch.randn(n_timepoints, n_regressors, device=device)

    # Fit ARMA(1,1)
    results = fit_glm_arma11(
        data,
        design,
        tr=2.0,
        verbose=False,
        device=device,
        want_residuals=True,  # For Ljung-Box
        want_ols=True,  # For comparison tests
    )

    # Add spatial metadata
    results.full_shape = (10, 10, 5)
    results.voxel_mask = torch.ones(n_voxels, dtype=torch.bool, device=device)
    results.affine = np.eye(4)

    # Also add to OLS results
    if results.ols_results is not None:
        results.ols_results.full_shape = (10, 10, 5)
        results.ols_results.voxel_mask = torch.ones(
            n_voxels, dtype=torch.bool, device=device
        )
        results.ols_results.affine = np.eye(4)

    return results


@pytest.fixture
def sample_contrasts():
    """Create sample contrast matrices."""
    # 4 regressors: [Task1, Task2, Baseline, Motion]
    contrasts = [
        [1, 0, 0, 0],  # Task1
        [0, 1, 0, 0],  # Task2
        [1, -1, 0, 0],  # Task1 vs Task2
        [0.5, 0.5, 0, 0],  # Average Task
    ]
    return np.array(contrasts, dtype=np.float32)


# =============================================================================
# Tests for slice_glm_results()
# =============================================================================


class TestSliceGLMResults:
    """Tests for slicing GLM results by regressor indices."""

    def test_slice_basic_indices(self, simple_glm_results):
        """Test basic slicing with list of indices."""
        # Extract first 2 regressors
        sliced = slice_glm_results(simple_glm_results, [0, 1])

        assert sliced.betas.shape[1] == 2, "Should have 2 regressors"
        assert sliced.tstats.shape[1] == 2, "Should have 2 t-stats"

        # Check that values match original
        assert torch.allclose(sliced.betas[:, 0], simple_glm_results.betas[:, 0])
        assert torch.allclose(sliced.betas[:, 1], simple_glm_results.betas[:, 1])

    def test_slice_preserves_scalars(self, simple_glm_results):
        """Test that scalar attributes are preserved."""
        sliced = slice_glm_results(simple_glm_results, [0, 1])

        assert torch.allclose(sliced.r2, simple_glm_results.r2)
        assert torch.allclose(sliced.sigma2, simple_glm_results.sigma2)
        assert sliced.dof == simple_glm_results.dof
        assert sliced.tr == simple_glm_results.tr

    def test_slice_preserves_spatial_metadata(self, simple_glm_results):
        """Test that spatial metadata is preserved."""
        sliced = slice_glm_results(simple_glm_results, [0, 1])

        assert sliced.full_shape == simple_glm_results.full_shape
        assert torch.allclose(sliced.voxel_mask, simple_glm_results.voxel_mask)
        assert np.allclose(sliced.affine, simple_glm_results.affine)

    def test_slice_covariance_matrix(self, simple_glm_results):
        """Test that covariance matrix is sliced in both dimensions."""
        # OLS uses xtx_inv
        sliced = slice_glm_results(simple_glm_results, [0, 2])

        if (
            hasattr(simple_glm_results, "xtx_inv")
            and simple_glm_results.xtx_inv is not None
        ):
            assert sliced.xtx_inv.shape[-2:] == (2, 2), "xtx_inv should be 2x2"

    def test_slice_arma_results(self, simple_arma_results):
        """Test slicing ARMA11Results preserves ARMA-specific attributes."""
        sliced = slice_glm_results(simple_arma_results, [0, 1])

        # ARMA-specific attributes should be preserved (not sliced)
        assert torch.allclose(sliced.arma_params, simple_arma_results.arma_params)
        assert torch.allclose(sliced.arma_lambda, simple_arma_results.arma_lambda)

    def test_slice_with_numpy_indices(self, simple_glm_results):
        """Test slicing with numpy array indices."""
        indices = np.array([0, 2])
        sliced = slice_glm_results(simple_glm_results, indices)

        assert sliced.betas.shape[1] == 2
        assert torch.allclose(sliced.betas[:, 0], simple_glm_results.betas[:, 0])
        assert torch.allclose(sliced.betas[:, 1], simple_glm_results.betas[:, 2])

    def test_slice_with_torch_indices(self, simple_glm_results):
        """Test slicing with torch tensor indices."""
        indices = torch.tensor([1, 3])
        sliced = slice_glm_results(simple_glm_results, indices)

        assert sliced.betas.shape[1] == 2
        assert torch.allclose(sliced.betas[:, 0], simple_glm_results.betas[:, 1])
        assert torch.allclose(sliced.betas[:, 1], simple_glm_results.betas[:, 3])

    def test_slice_creates_independent_copy(self, simple_glm_results):
        """Test that slicing creates independent copy (no aliasing)."""
        sliced = slice_glm_results(simple_glm_results, [0, 1])

        # Modify sliced result
        original_value = sliced.betas[0, 0].item()
        sliced.betas[0, 0] = 999.0

        # Original should be unchanged
        assert simple_glm_results.betas[0, 0].item() != 999.0
        assert simple_glm_results.betas[0, 0].item() == original_value

    def test_slice_empty_indices(self, simple_glm_results):
        """Test slicing with empty indices list."""
        sliced = slice_glm_results(simple_glm_results, [])

        assert sliced.betas.shape[1] == 0, "Should have no regressors"


# =============================================================================
# Tests for write_glm_bucket_as_nifti()
# =============================================================================


class TestWriteAFNIBucket:
    """Tests for writing AFNI-style bucket files."""

    def test_write_basic_nifti(self, simple_glm_results):
        """Test basic NIfTI writing without contrasts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_bucket.nii.gz"

            result_path = write_glm_bucket_as_nifti(
                simple_glm_results,
                output_path,
                condition_names=["Task1", "Task2", "Baseline", "Motion"],
                apply_afni_metadata=False,  # Skip AFNI metadata (no 3drefit)
                compress_output=True,
            )

            assert result_path.exists(), "Output file should exist"

            # Load and check
            img = nib.load(result_path)
            data = img.get_fdata()

            # Should have: F-stat + 4 conditions × 2 (beta, tstat) = 9 volumes
            assert data.shape == (10, 10, 10, 9), (
                f"Expected 9 volumes, got {data.shape}"
            )

    def test_write_with_contrasts(self, simple_glm_results, sample_contrasts):
        """Test writing with contrast results."""
        # Compute contrasts
        contrast_results = ffs.compute_contrasts(simple_glm_results, sample_contrasts)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_contrasts.nii.gz"

            result_path = write_glm_bucket_as_nifti(
                simple_glm_results,
                output_path,
                condition_names=["Task1", "Task2", "Baseline", "Motion"],
                contrast_names=["Task1", "Task2", "Task1vsTask2", "AvgTask"],
                contrast_results=contrast_results,
                apply_afni_metadata=False,
                compress_output=True,
            )

            assert result_path.exists()

            # Load and check
            img = nib.load(result_path)
            data = img.get_fdata()

            # F-stat + 4 conditions × 2 + 4 contrasts × 2 = 17 volumes
            expected_vols = 1 + 4 * 2 + 4 * 2
            assert data.shape[3] == expected_vols, f"Expected {expected_vols} volumes"

    def test_write_uncompressed(self, simple_glm_results):
        """Test writing uncompressed NIfTI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_uncompressed.nii"

            result_path = write_glm_bucket_as_nifti(
                simple_glm_results,
                output_path,
                condition_names=[
                    "Task1",
                    "Task2",
                    "Task3",
                    "Task4",
                ],  # Match 4 regressors
                compress_output=False,
            )

            assert result_path.exists()
            assert result_path.suffix == ".nii", "Should be uncompressed .nii"

    def test_write_with_custom_shape(self, device):
        """Test writing with custom volume shape."""
        torch.manual_seed(42)

        # Create data with known voxel count matching the custom shape
        # Custom shape: 5x5x4 = 100 voxels
        n_voxels = 100
        n_timepoints = 50
        n_regressors = 4

        data = torch.randn(n_voxels, n_timepoints, device=device)
        design = torch.randn(n_timepoints, n_regressors, device=device)

        results = fit_glm(data, design, tr=2.0, verbose=False, device=device)

        # Set consistent spatial metadata
        results.full_shape = (5, 5, 4)  # 5*5*4 = 100 voxels
        results.voxel_mask = torch.ones(n_voxels, dtype=torch.bool, device=device)
        results.affine = np.eye(4)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_shape.nii.gz"

            result_path = write_glm_bucket_as_nifti(
                results,
                output_path,
                condition_names=["Task1", "Task2", "Task3", "Task4"],
                apply_afni_metadata=False,
            )

            img = nib.load(result_path)
            assert img.shape[:3] == (5, 5, 4), (
                f"Should have correct shape, got {img.shape[:3]}"
            )

    def test_write_with_custom_affine(self, simple_glm_results):
        """Test writing with custom affine matrix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_affine.nii.gz"

            # Create custom affine (2mm isotropic)
            affine = np.eye(4)
            affine[:3, :3] *= 2.0

            result_path = write_glm_bucket_as_nifti(
                simple_glm_results,
                output_path,
                condition_names=[
                    "Task1",
                    "Task2",
                    "Task3",
                    "Task4",
                ],  # Match 4 regressors
                affine=affine,
                apply_afni_metadata=False,
            )

            img = nib.load(result_path)
            assert np.allclose(img.affine, affine), "Should use custom affine"

    def test_write_without_fstat_succeeds(self, simple_glm_results):
        """Test that writing succeeds when F-stat is not available.

        F-statistics are optional for AFNI bucket files.
        """
        # Remove F-stat
        simple_glm_results.fstats = None

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_no_fstat.nii.gz"

            # Should succeed - F-stats are optional
            result_path = write_glm_bucket_as_nifti(
                simple_glm_results,
                output_path,
                condition_names=[
                    "Task1",
                    "Task2",
                    "Task3",
                    "Task4",
                ],  # Match 4 regressors
                apply_afni_metadata=False,
            )

            # Verify the file was created
            assert result_path.exists()
            img = nib.load(result_path)
            # Without F-stat, we should only have beta/tstat pairs (8 sub-bricks)
            assert img.shape[-1] == 8  # 4 conditions * 2 (beta + tstat)

    def test_write_creates_parent_directory(self, simple_glm_results):
        """Test that parent directories are created if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dir" / "test.nii.gz"

            result_path = write_glm_bucket_as_nifti(
                simple_glm_results,
                output_path,
                condition_names=[
                    "Task1",
                    "Task2",
                    "Task3",
                    "Task4",
                ],  # Match 4 regressors
                apply_afni_metadata=False,
            )

            assert result_path.exists(), "Should create parent directories"

    def test_write_roundtrip_values(self, simple_glm_results):
        """Test that written values match input values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_roundtrip.nii.gz"

            result_path = write_glm_bucket_as_nifti(
                simple_glm_results,
                output_path,
                condition_names=[
                    "Task1",
                    "Task2",
                    "Task3",
                    "Task4",
                ],  # Match 4 regressors
                apply_afni_metadata=False,
            )

            # Load and check values
            img = nib.load(result_path)
            data = img.get_fdata()

            # Extract beta volume (after F-stat)
            beta_vol = data[:, :, :, 1].flatten()

            # Compare with original betas (first regressor)
            original_betas = simple_glm_results.betas[:, 0].cpu().numpy()

            # Should match (within floating point tolerance)
            assert np.allclose(
                beta_vol[: len(original_betas)], original_betas, rtol=1e-5
            )


# =============================================================================
# Tests for write_ols_arma_comparison()
# =============================================================================


class TestWriteOLSARMAComparison:
    """Tests for writing side-by-side OLS vs ARMA comparison."""

    def test_write_comparison_basic(self, simple_arma_results):
        """Test basic OLS vs ARMA comparison writing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_prefix = Path(tmpdir) / "comparison"

            outputs = write_ols_arma_comparison(
                simple_arma_results,
                output_prefix,
                condition_names=["Task1", "Task2", "Baseline", "Motion"],
                apply_afni_metadata=False,
            )

            # Check all three files exist
            assert outputs["ols"].exists(), "OLS file should exist"
            assert outputs["arma"].exists(), "ARMA file should exist"
            assert outputs["comparison_summary"].exists(), "JSON summary should exist"

    def test_comparison_filenames(self, simple_arma_results):
        """Test that output filenames have correct suffixes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_prefix = Path(tmpdir) / "test_analysis"

            outputs = write_ols_arma_comparison(
                simple_arma_results,
                output_prefix,
                condition_names=[
                    "Task1",
                    "Task2",
                    "Task3",
                    "Task4",
                ],  # Match 4 regressors
                apply_afni_metadata=False,
            )

            assert "OLS" in outputs["ols"].stem, "OLS file should have _OLS suffix"
            assert "ARMA" in outputs["arma"].stem, "ARMA file should have _ARMA suffix"

    def test_comparison_with_contrasts(self, simple_arma_results, sample_contrasts):
        """Test comparison writing with contrasts.

        Note: var_betas is not computed by default for ARMA, so we use
        OLS results for contrast computation on both.
        """
        # Compute contrasts using OLS results (ARMA doesn't have var_betas by default)
        contrast_results_ols = ffs.compute_contrasts(
            simple_arma_results.ols_results, sample_contrasts
        )
        # Use OLS results for ARMA contrasts too since var_betas not available
        contrast_results_arma = contrast_results_ols

        with tempfile.TemporaryDirectory() as tmpdir:
            output_prefix = Path(tmpdir) / "comparison_contrasts"

            outputs = write_ols_arma_comparison(
                simple_arma_results,
                output_prefix,
                condition_names=["Task1", "Task2", "Baseline", "Motion"],
                contrast_names=["Task1", "Task2", "Task1vsTask2", "AvgTask"],
                contrast_results_ols=contrast_results_ols,
                contrast_results_arma=contrast_results_arma,
                apply_afni_metadata=False,
            )

            # Load and verify both files have same number of volumes
            ols_img = nib.load(outputs["ols"])
            arma_img = nib.load(outputs["arma"])

            assert ols_img.shape == arma_img.shape, (
                "OLS and ARMA should have same shape"
            )

    def test_comparison_json_content(self, simple_arma_results):
        """Test that JSON comparison summary contains expected fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_prefix = Path(tmpdir) / "test_json"

            outputs = write_ols_arma_comparison(
                simple_arma_results,
                output_prefix,
                condition_names=[
                    "Task1",
                    "Task2",
                    "Task3",
                    "Task4",
                ],  # Match 4 regressors
                apply_afni_metadata=False,
            )

            # Load and parse JSON
            with open(outputs["comparison_summary"]) as f:
                summary = json.load(f)

            # Check for expected fields (updated structure)
            assert "ols" in summary, "Summary should have 'ols' section"
            assert "arma" in summary, "Summary should have 'arma' section"
            assert "comparison" in summary, "Summary should have 'comparison' section"

            # Check comparison section has key metrics
            assert (
                "r2_improvement" in summary["comparison"]
                or "beta_correlation" in summary["comparison"]
            ), "Comparison should have metrics"

    def test_comparison_requires_ols_results(self):
        """Test that comparison fails if ols_results is missing."""
        # Create ARMA results without OLS
        device = torch.device("cpu")
        torch.manual_seed(42)

        data = torch.randn(100, 50, device=device)
        design = torch.randn(50, 2, device=device)

        arma_results = fit_glm_arma11(
            data,
            design,
            tr=2.0,
            verbose=False,
            want_ols=False,  # No OLS!
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="missing ols_results"):
                write_ols_arma_comparison(
                    arma_results,
                    Path(tmpdir) / "test",
                    condition_names=["Task1", "Task2"],  # Match 2 regressors
                )

    def test_comparison_files_are_independent(self, simple_arma_results):
        """Test that OLS and ARMA files can be loaded independently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_prefix = Path(tmpdir) / "independent"

            outputs = write_ols_arma_comparison(
                simple_arma_results,
                output_prefix,
                condition_names=[
                    "Task1",
                    "Task2",
                    "Task3",
                    "Task4",
                ],  # Match 4 regressors
                apply_afni_metadata=False,
            )

            # Load each file independently
            ols_img = nib.load(outputs["ols"])
            arma_img = nib.load(outputs["arma"])

            # Should both be valid NIfTI images
            assert ols_img.get_fdata().shape[3] > 0
            assert arma_img.get_fdata().shape[3] > 0


# =============================================================================
# Tests for save_arma_rvar()
# =============================================================================


class TestSaveARMARvar:
    """Tests for saving ARMA parameters in AFNI -Rvar format."""

    def test_save_basic(self, simple_arma_results):
        """Test basic ARMA parameter saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "arma_rvar.nii.gz"

            result_path = save_arma_rvar(
                simple_arma_results,
                output_path,
                volume_shape=simple_arma_results.full_shape,
                voxel_mask=simple_arma_results.voxel_mask,
                affine=simple_arma_results.affine,
            )

            assert result_path.exists(), "Output file should exist"

            # Load and check
            img = nib.load(result_path)
            data = img.get_fdata()

            # Should have 6 volumes: a, b, lambda, StDev, -LogLik, LjungBox
            assert data.shape == (*simple_arma_results.full_shape, 6), (
                "Should have 6 volumes"
            )

    def test_save_volumes_content(self, simple_arma_results):
        """Test that saved volumes contain expected values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "arma_rvar.nii.gz"

            save_arma_rvar(
                simple_arma_results,
                output_path,
                volume_shape=simple_arma_results.full_shape,
                voxel_mask=simple_arma_results.voxel_mask,
                affine=simple_arma_results.affine,
            )

            # Load and extract volumes
            img = nib.load(output_path)
            data = img.get_fdata()

            # Volume 0: 'a' parameter
            a_vol = data[:, :, :, 0].flatten()
            a_params = simple_arma_results.arma_params[:, 0].cpu().numpy()

            # Check that values match (within mask)
            mask_flat = simple_arma_results.voxel_mask.cpu().numpy()
            assert np.allclose(a_vol[mask_flat], a_params, rtol=1e-5)

            # Volume 1: 'b' parameter
            b_vol = data[:, :, :, 1].flatten()
            b_params = simple_arma_results.arma_params[:, 1].cpu().numpy()
            assert np.allclose(b_vol[mask_flat], b_params, rtol=1e-5)

    def test_save_roundtrip_with_load(self, simple_arma_results):
        """Test saving and loading ARMA parameters for reuse."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "arma_rvar.nii.gz"

            # Move to CPU for saving
            # GPU tensors handled by write functions

            # Save
            save_arma_rvar(
                simple_arma_results,
                output_path,
                volume_shape=simple_arma_results.full_shape,
                voxel_mask=simple_arma_results.voxel_mask,
                affine=simple_arma_results.affine,
            )

            # Load back
            loaded_params = load_arma_params(
                output_path,
                voxel_mask=simple_arma_results.voxel_mask,
            )

            # Compare
            original_params = simple_arma_results.arma_params
            if torch.is_tensor(original_params):
                original_params = original_params.cpu().numpy()

            assert np.allclose(loaded_params, original_params, rtol=1e-5), (
                "Loaded params should match original"
            )

    def test_save_creates_parent_directory(self, simple_arma_results):
        """Test that save creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dirs" / "arma_rvar.nii.gz"

            # Move to CPU for saving
            # GPU tensors handled by write functions

            result_path = save_arma_rvar(
                simple_arma_results,
                output_path,
                volume_shape=simple_arma_results.full_shape,
                voxel_mask=simple_arma_results.voxel_mask,
                affine=simple_arma_results.affine,
            )

            assert result_path.exists(), "Should create parent directories"

    def test_save_without_residuals(self, device):
        """Test saving when residuals are not available (LjungBox = 0)."""
        torch.manual_seed(42)

        data = torch.randn(100, 50, device=device)
        design = torch.randn(50, 2, device=device)

        # Fit without residuals
        arma_results = fit_glm_arma11(
            data,
            design,
            tr=2.0,
            verbose=False,
            want_residuals=False,  # No residuals!
            device=device,
        )

        arma_results.full_shape = (10, 10, 1)
        arma_results.voxel_mask = torch.ones(100, dtype=torch.bool, device=device)
        arma_results.affine = np.eye(4)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "arma_no_resid.nii.gz"

            result_path = save_arma_rvar(
                arma_results,  # Use the local arma_results, not fixture
                output_path,
                volume_shape=arma_results.full_shape,
                voxel_mask=arma_results.voxel_mask,
                affine=arma_results.affine,
            )

            # Should still create file with 6 volumes
            img = nib.load(result_path)
            assert img.shape[3] == 6

            # LjungBox volume (5) should be all zeros
            ljung_box_vol = img.get_fdata()[:, :, :, 5]
            assert np.allclose(ljung_box_vol, 0.0)

    def test_save_custom_max_lag(self, simple_arma_results):
        """Test saving with custom max_lag for Ljung-Box."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "arma_custom_lag.nii.gz"

            # Move to CPU for saving
            # GPU tensors handled by write functions

            # Use different max_lag
            result_path = save_arma_rvar(
                simple_arma_results,
                output_path,
                volume_shape=simple_arma_results.full_shape,
                voxel_mask=simple_arma_results.voxel_mask,
                affine=simple_arma_results.affine,
                max_lag=20,  # Different from default (30)
            )

            assert result_path.exists()

            # File should still have 6 volumes
            img = nib.load(result_path)
            assert img.shape[3] == 6


# =============================================================================
# Integration tests (full workflow)
# =============================================================================


class TestFullWorkflow:
    """Integration tests for complete analysis workflow."""

    def test_full_analysis_workflow(self, device):
        """Test complete workflow matching analyze_taskforce_ses02_clean.py.

        Note: var_betas is not computed by default for ARMA, so we use
        OLS results for contrast computation.
        """
        torch.manual_seed(42)

        # Step 1: Generate data
        n_timepoints = 100
        n_voxels = 500
        n_regressors = 6  # 4 stimulus + 2 nuisance

        data = torch.randn(n_voxels, n_timepoints, device=device)
        design = torch.randn(n_timepoints, n_regressors, device=device)

        # Step 2: Fit ARMA(1,1) with OLS baseline
        results = fit_glm_arma11(
            data,
            design,
            tr=2.0,
            verbose=False,
            device=device,
            want_ols=True,
            want_residuals=True,
        )

        # Add spatial metadata
        results.full_shape = (10, 10, 5)
        results.voxel_mask = torch.ones(n_voxels, dtype=torch.bool, device=device)
        results.affine = np.eye(4)
        results.ols_results.full_shape = (10, 10, 5)
        results.ols_results.voxel_mask = torch.ones(
            n_voxels, dtype=torch.bool, device=device
        )
        results.ols_results.affine = np.eye(4)

        # Step 3: Compute contrasts using OLS results
        # (ARMA results don't have var_betas by default for memory efficiency)
        contrasts = np.array(
            [
                [1, 0, 0, 0, 0, 0],  # Stim1
                [0, 1, 0, 0, 0, 0],  # Stim2
                [1, -1, 0, 0, 0, 0],  # Stim1 vs Stim2
            ],
            dtype=np.float32,
        )

        contrast_results_ols = ffs.compute_contrasts(results.ols_results, contrasts)
        # Use OLS results for ARMA contrasts too
        contrast_results_arma = contrast_results_ols

        # Step 4: Slice by regressor type
        stim_indices = [0, 1, 2, 3]
        nuisance_indices = [4, 5]

        results_stim = slice_glm_results(results, stim_indices)
        results_stim.ols_results = slice_glm_results(results.ols_results, stim_indices)
        results_nuisance = slice_glm_results(results, nuisance_indices)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Step 5: Write outputs (GPU tensors auto-converted to CPU by write functions)
            # [A] Stimulus + contrasts (OLS vs ARMA comparison)
            outputs = write_ols_arma_comparison(
                results_stim,
                tmpdir / "glm_main",
                condition_names=["Stim1", "Stim2", "Stim3", "Stim4"],
                contrast_names=["Stim1", "Stim2", "Stim1vsStim2"],
                contrast_results_ols=contrast_results_ols,
                contrast_results_arma=contrast_results_arma,
                apply_afni_metadata=False,
            )

            assert outputs["ols"].exists()
            assert outputs["arma"].exists()
            assert outputs["comparison_summary"].exists()

            # [B] Nuisance regressors
            nuisance_file = write_glm_bucket_as_nifti(
                results_nuisance,
                tmpdir / "glm_nuisance.nii.gz",
                condition_names=["Motion", "Baseline"],
                apply_afni_metadata=False,
            )

            assert nuisance_file.exists()

            # [C] Save ARMA parameters
            arma_rvar_file = save_arma_rvar(
                results,
                tmpdir / "arma_rvar.nii.gz",
                volume_shape=results.full_shape,
                voxel_mask=results.voxel_mask.cpu().numpy(),  # Convert to numpy for function
                affine=results.affine,
            )

            assert arma_rvar_file.exists()

            # Step 6: Verify we can reload ARMA params
            loaded_params = load_arma_params(
                arma_rvar_file,
                voxel_mask=results.voxel_mask.cpu().numpy(),  # Convert to numpy
            )

            original_params = results.arma_params
            if torch.is_tensor(original_params):
                original_params = original_params.cpu().numpy()

            assert np.allclose(loaded_params, original_params, rtol=1e-5)

            print("\n✓ Full workflow test passed!")
            print(f"  Created {len(list(tmpdir.glob('*')))} output files")

    def test_workflow_without_contrasts(self, device):
        """Test workflow when no contrasts are defined."""
        torch.manual_seed(42)

        data = torch.randn(200, 50, device=device)
        design = torch.randn(50, 3, device=device)

        results = fit_glm_arma11(
            data,
            design,
            tr=2.0,
            verbose=False,
            device=device,
            want_ols=True,
        )

        results.full_shape = (10, 10, 2)
        results.voxel_mask = torch.ones(200, dtype=torch.bool, device=device)
        results.affine = np.eye(4)
        results.ols_results.full_shape = (10, 10, 2)
        results.ols_results.voxel_mask = torch.ones(
            200, dtype=torch.bool, device=device
        )
        results.ols_results.affine = np.eye(4)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write without contrasts
            outputs = write_ols_arma_comparison(
                results,
                Path(tmpdir) / "no_contrasts",
                condition_names=["Task1", "Task2", "Baseline"],
                apply_afni_metadata=False,
            )

            # Should still work
            assert outputs["ols"].exists()
            assert outputs["arma"].exists()

    def test_analyze_from_design_matrix_sets_metadata(self, device):
        """
        Test that analyze_from_design_matrix() sets spatial metadata on BOTH
        ARMA and OLS results. This catches the bug where OLS results didn't
        get full_shape/voxel_mask/affine, causing write functions to fail.

        Regression test for: write_glm_bucket_as_nifti() failing after slice_glm_results()
        because OLS results had no spatial metadata.
        """
        torch.manual_seed(42)

        # Create synthetic 4D data file
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a small 4D NIfTI file
            data_4d = np.random.randn(10, 10, 5, 100).astype(np.float32)
            affine = np.eye(4)
            affine[:3, :3] *= 2.0  # 2mm voxels
            img = nib.Nifti1Image(data_4d, affine)

            fmri_file = tmpdir / "test_data.nii.gz"
            nib.save(img, fmri_file)

            # Create AFNI-format design matrix with TR in header (100 TRs, 4 regressors)
            design_matrix = np.random.randn(100, 4).astype(np.float32)
            design_file = tmpdir / "design.1D"

            # Write AFNI-style design matrix with TR in header
            with open(design_file, "w") as f:
                f.write("# RowTR = 2.0\n")
                f.write("# NRowFull = 100\n")
                f.write('# ColumnLabels = "cond1;cond2;cond3;cond4"\n')
                np.savetxt(f, design_matrix)

            # Create mask
            mask_3d = np.ones((10, 10, 5), dtype=np.uint8)
            mask_img = nib.Nifti1Image(mask_3d, affine)
            mask_file = tmpdir / "mask.nii.gz"
            nib.save(mask_img, mask_file)

            # Run analyze_from_design_matrix with want_ols=True
            results, design_info = ffs.analyze_from_design_matrix(
                fmri_file,
                design_file,
                method="arma11",
                mask_file=mask_file,
                mask_threshold=0.5,
                device=device,
                want_ols=True,
            )

            # ===== CRITICAL CHECKS =====
            # These are what the bug broke - OLS results had no metadata

            # 1. ARMA results should have spatial metadata
            assert hasattr(results, "full_shape"), "ARMA results missing full_shape"
            assert results.full_shape == (10, 10, 5), (
                f"Wrong ARMA shape: {results.full_shape}"
            )
            assert hasattr(results, "voxel_mask"), "ARMA results missing voxel_mask"
            assert hasattr(results, "affine"), "ARMA results missing affine"

            # 2. OLS results should ALSO have spatial metadata (THIS WAS THE BUG!)
            assert hasattr(results, "ols_results"), "No OLS results"
            assert results.ols_results is not None, "OLS results is None"
            assert hasattr(results.ols_results, "full_shape"), (
                "OLS results missing full_shape"
            )
            assert results.ols_results.full_shape == (10, 10, 5), (
                f"Wrong OLS shape: {results.ols_results.full_shape}"
            )
            assert hasattr(results.ols_results, "voxel_mask"), (
                "OLS results missing voxel_mask"
            )
            assert hasattr(results.ols_results, "affine"), "OLS results missing affine"

            # 3. Test the actual bug scenario: slice then write
            stim_indices = [0, 1]
            results_sliced = slice_glm_results(results, stim_indices)

            # Slice OLS too (this is what the user's script does)
            if hasattr(results, "ols_results") and results.ols_results is not None:
                ols_sliced = slice_glm_results(results.ols_results, stim_indices)
                results_sliced.ols_results = ols_sliced

            # 4. Now try to write - this is where the bug manifested
            # This should NOT raise "GLM results do not contain spatial shape information"
            try:
                written_file = write_ols_arma_comparison(
                    results_sliced,
                    tmpdir / "sliced_comparison",
                    condition_names=["Reg1", "Reg2"],
                    apply_afni_metadata=False,
                )
                assert written_file["ols"].exists(), "OLS file should be written"
                assert written_file["arma"].exists(), "ARMA file should be written"
            except ValueError as e:
                if "spatial shape information" in str(e):
                    pytest.fail(
                        "Bug reproduced! analyze_from_design_matrix() didn't set "
                        "spatial metadata on OLS results, causing write to fail"
                    )
                else:
                    raise

            print(
                "\n✓ Regression test passed: spatial metadata propagated to OLS results"
            )


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
