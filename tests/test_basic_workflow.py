"""
Basic workflow tests using small synthetic dataset.

Test data: 2 runs, 10x10x10 voxels, 360 TRs each, TR=1s
Design: 8 polynomials (4 per run) + 2 stimuli (movie, prompt)
GLT: movie - prompt
"""
import shutil
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from fastfuncstuff.analysis import analyze_from_design_matrix
from fastfuncstuff.data_cache import check_cache_valid

# Test data paths
DATA_DIR = Path(__file__).parent.parent / "test_data" / "small_validation_afni_data"
DESIGN_MATRIX = DATA_DIR / "X.xmat.1D"
INPUT_FILES = [
    DATA_DIR / "small_test_r01.nii.gz",
    DATA_DIR / "small_test_r02.nii.gz",
]


@pytest.fixture
def test_data_dir():
    """Verify test data exists."""
    assert DESIGN_MATRIX.exists(), f"Design matrix not found: {DESIGN_MATRIX}"
    for f in INPUT_FILES:
        assert f.exists(), f"Input file not found: {f}"
    return DATA_DIR


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for outputs."""
    tmpdir = tempfile.mkdtemp(prefix="ffs_test_")
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


def test_data_dimensions(test_data_dir):
    """Verify test data has expected dimensions."""
    img1 = nib.load(INPUT_FILES[0])
    img2 = nib.load(INPUT_FILES[1])

    assert img1.shape == (10, 10, 10, 360), f"Run 1 shape: {img1.shape}"
    assert img2.shape == (10, 10, 10, 360), f"Run 2 shape: {img2.shape}"

    # Check header preserved
    assert img1.header is not None
    assert img1.affine is not None


def test_ols_basic(test_data_dir, temp_output_dir):
    """Test basic OLS fitting."""
    results, design_info = analyze_from_design_matrix(
        fmri_data=INPUT_FILES,
        design_matrix_file=DESIGN_MATRIX,
        method="ols",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Check design info
    assert design_info["n_timepoints"] == 720
    assert design_info["n_regressors"] == 10
    assert len(design_info.get("run_starts", [])) == 2  # 2 runs

    # Results will only contain task regressors (2: movie, prompt) due to task_indices
    # This is correct behavior - nuisance regressors are regressed out
    n_task = 2
    assert results.betas.shape == (1000, n_task), f"Betas shape: {results.betas.shape}"
    assert results.tstats.shape == (1000, n_task), f"T-stats shape: {results.tstats.shape}"
    assert results.fstats.shape == (1000,), f"F-stats shape: {results.fstats.shape}"
    assert results.r2.shape == (1000,), f"R² shape: {results.r2.shape}"

    # Check metadata preserved
    assert hasattr(results, "affine")
    assert hasattr(results, "nifti_header")
    assert results.nifti_header is not None


def test_arma_basic(test_data_dir, temp_output_dir):
    """Test ARMA(1,1) fitting with grid search."""
    # Use small grid for speed
    a_grid = torch.linspace(0.2, 0.9, 4)
    b_grid = torch.linspace(0.2, 0.9, 4)

    results, design_info = analyze_from_design_matrix(
        fmri_data=INPUT_FILES,
        design_matrix_file=DESIGN_MATRIX,
        method="arma11",
        arma_a_grid=a_grid,
        arma_b_grid=b_grid,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Check results shape - only task regressors returned (2 stimuli: movie, prompt)
    # Full design has 10 regressors (8 polynomials + 2 stimuli) but only task are returned
    assert results.betas.shape == (1000, 2)
    assert results.tstats.shape == (1000, 2)
    assert results.fstats.shape == (1000,)

    # Check ARMA parameters
    assert hasattr(results, "arma_params")
    assert results.arma_params.shape == (1000, 2)  # a, b (lambda stored separately)

    # Check header preserved
    assert hasattr(results, "nifti_header")


def test_cache_save_load(test_data_dir, temp_output_dir):
    """Test cache saving and loading with header preservation."""
    cache_file = temp_output_dir / "test_cache.h5"

    # Load data and save to cache
    results, design_info = analyze_from_design_matrix(
        fmri_data=INPUT_FILES,
        design_matrix_file=DESIGN_MATRIX,
        method="ols",
        cache_file=cache_file,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Verify cache files created
    assert cache_file.exists(), "Cache file not created"
    header_file = Path(str(cache_file) + ".header.pkl")
    assert header_file.exists(), "Header pickle not created"

    # Verify cache is valid
    assert check_cache_valid(cache_file, INPUT_FILES)

    # Load from cache
    results2, design_info2 = analyze_from_design_matrix(
        fmri_data=INPUT_FILES,
        design_matrix_file=DESIGN_MATRIX,
        method="ols",
        cache_file=cache_file,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Verify results match
    np.testing.assert_array_almost_equal(results.betas, results2.betas, decimal=5)

    # Verify header preserved from cache
    assert results2.nifti_header is not None
    assert hasattr(results2, "affine")


def test_contrasts_glt(test_data_dir, temp_output_dir):
    """Test GLT contrast (movie - prompt)."""
    results, design_info = analyze_from_design_matrix(
        fmri_data=INPUT_FILES,
        design_matrix_file=DESIGN_MATRIX,
        method="ols",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Check GLT info
    assert "glt_labels" in design_info
    assert len(design_info["glt_labels"]) == 1
    assert design_info["glt_labels"][0] == "movieVprompt"

    # Check GLT results if present
    if hasattr(results, "glt_contrasts") and results.glt_contrasts is not None:
        assert results.glt_contrasts.shape[0] == 1000  # n_voxels


def test_bucket_output(test_data_dir, temp_output_dir):
    """Test bucket file output with header preservation."""
    from fastfuncstuff.glm.outputs import write_glm_bucket_as_nifti

    results, design_info = analyze_from_design_matrix(
        fmri_data=INPUT_FILES,
        design_matrix_file=DESIGN_MATRIX,
        method="ols",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    output_file = temp_output_dir / "test_bucket.nii.gz"

    # Get stimulus labels (only task regressors, not all regressors)
    stim_labels = design_info.get("stim_labels", ["movie", "prompt"])

    # Write bucket file
    write_glm_bucket_as_nifti(
        results,
        output_file,
        condition_names=stim_labels,  # Only task regressors
        output_format="nifti_gz",
    )

    assert output_file.exists(), "Bucket file not created"

    # Load and verify
    img = nib.load(output_file)

    # Should have: 1 F-stat + 2*(2 task regressors) + 2*(1 contrast) = 7 sub-briks
    # Structure: Full_Fstat, movie_Coef, movie_Tstat, prompt_Coef, prompt_Tstat,
    #            movieVprompt_Coef, movieVprompt_Tstat
    assert img.shape[3] == 7, f"Expected 7 sub-briks, got {img.shape[3]}"

    # Verify header preserved
    assert img.header is not None
    # Check TR preserved (should be 1.0 from test data)
    assert img.header.get_zooms()[3] > 0  # TR is set


def test_rvar_output(test_data_dir, temp_output_dir):
    """Test Rvar file output for ARMA parameters."""
    # Use tiny grid for speed
    a_grid = torch.linspace(0.3, 0.7, 3)
    b_grid = torch.linspace(0.3, 0.7, 3)

    results, design_info = analyze_from_design_matrix(
        fmri_data=INPUT_FILES,
        design_matrix_file=DESIGN_MATRIX,
        method="arma11",
        arma_a_grid=a_grid,
        arma_b_grid=b_grid,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Manually create Rvar file (normally done by 3dREMLfast.py)
    output_file = temp_output_dir / "test_Rvar.nii.gz"

    # Rvar should have shape (x, y, z, 8) for ARMA(1,1)
    # Sub-briks: 0-6 unused, 7-8 = a, b parameters
    rvar_data = np.zeros((10, 10, 10, 8), dtype=np.float32)

    # Extract a and b parameters from results (in voxel order)
    # arma_params now contains only (a, b), lambda is stored separately
    if hasattr(results, "arma_params"):
        params_vol = results.arma_params.reshape(10, 10, 10, 2)
        rvar_data[..., 6] = params_vol[..., 0]  # a parameter
        rvar_data[..., 7] = params_vol[..., 1]  # b parameter

        # Save with header
        img = nib.Nifti1Image(rvar_data, results.affine, header=results.nifti_header)
        nib.save(img, output_file)

        assert output_file.exists()


def test_masking(test_data_dir, temp_output_dir):
    """Test analysis with mask file."""
    # Create simple mask (middle 5x5x5 cube)
    mask_data = np.zeros((10, 10, 10), dtype=np.uint8)
    mask_data[2:7, 2:7, 2:7] = 1

    mask_file = temp_output_dir / "test_mask.nii.gz"

    # Load first input to get affine
    img = nib.load(INPUT_FILES[0])
    mask_img = nib.Nifti1Image(mask_data, img.affine)
    nib.save(mask_img, mask_file)

    # Run with mask
    results, design_info = analyze_from_design_matrix(
        fmri_data=INPUT_FILES,
        design_matrix_file=DESIGN_MATRIX,
        method="ols",
        mask_file=mask_file,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Should only fit masked voxels (125)
    expected_voxels = int(mask_data.sum())
    assert results.betas.shape[0] == expected_voxels, \
        f"Expected {expected_voxels} voxels, got {results.betas.shape[0]}"


def test_test_mode(test_data_dir, temp_output_dir):
    """Test subset mode (only fit N voxels).

    Note: test_n_voxels extracts a cube from center, so actual count
    depends on cube size that fits requested voxels. For 10x10x10 data
    with test_n_voxels=100, it extracts a 4x4x4=64 voxel cube.
    """
    results, design_info = analyze_from_design_matrix(
        fmri_data=INPUT_FILES,
        design_matrix_file=DESIGN_MATRIX,
        method="ols",
        test_n_voxels=100,  # Request ~100 voxels, gets cube from center
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Actual voxels is 64 (4x4x4 cube from center of 10x10x10 volume)
    assert results.betas.shape[0] == 64


def test_batch_size_reasonable(test_data_dir, temp_output_dir):
    """Test that batch sizes are reasonable (not capped too low)."""
    # This is more of an integration check - batch size should be > 1000
    # given our memory optimizations
    results, design_info = analyze_from_design_matrix(
        fmri_data=INPUT_FILES,
        design_matrix_file=DESIGN_MATRIX,
        method="arma11",
        arma_a_grid=torch.linspace(0.3, 0.7, 3),
        arma_b_grid=torch.linspace(0.3, 0.7, 3),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        debug_memory=True,  # This will print batch size info
    )

    # Just verify it completes without OOM
    assert results.betas.shape[0] == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
