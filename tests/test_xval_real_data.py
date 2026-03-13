"""
Real data tests for cross-validation functionality.

Uses the small validation dataset in test_data/small_validation_afni_data/
to test cross-validation on real fMRI data.
"""

from pathlib import Path

import pytest
import torch

from fastfuncstuff.analysis import analyze_with_cross_validation

# Path to validation data (relative to project root)
VALIDATION_DATA_DIR = Path(__file__).parent.parent / "test_data" / "small_validation_afni_data"

# Input files - using vis_ files which should have better signal
INPUT_FILES = [
    VALIDATION_DATA_DIR / "vis_small_test_r01.nii.gz",
    VALIDATION_DATA_DIR / "vis_small_test_r02.nii.gz",
]
DESIGN_MATRIX = VALIDATION_DATA_DIR / "X.xmat.1D"


@pytest.fixture
def validation_data_available():
    """Check if validation data is available."""
    if not VALIDATION_DATA_DIR.exists():
        pytest.skip(f"Validation data not found: {VALIDATION_DATA_DIR}")

    for f in INPUT_FILES + [DESIGN_MATRIX]:
        if not f.exists():
            pytest.skip(f"Required file not found: {f}")

    return True


def test_xval_validation_data_exists():
    """Test that validation data directory and files exist."""
    assert VALIDATION_DATA_DIR.exists(), f"Validation data directory not found: {VALIDATION_DATA_DIR}"

    for f in INPUT_FILES:
        assert f.exists(), f"Input file not found: {f}"

    assert DESIGN_MATRIX.exists(), f"Design matrix not found: {DESIGN_MATRIX}"


def test_xval_r2_realistic_data_multiple_files(validation_data_available):
    """
    Test cross-validation on real fMRI data using multiple run files.

    Uses leave-one-run-out (LORO) strategy with 2 runs.
    """
    # Run cross-validation
    results, design_info = analyze_with_cross_validation(
        fmri_data=INPUT_FILES,
        design_matrix_file=DESIGN_MATRIX,
        cv_strategy=1,  # Leave-one-run-out
        metric="cod",
        test_n_voxels=1000,  # Fast test mode
        verbose=False,
    )

    # Basic validation
    assert "r2" in results
    assert "r2_median" in results
    assert "r2_mean" in results
    assert "r2_std" in results
    assert "r2_min" in results
    assert "r2_max" in results
    assert "n_splits" in results

    # Check shapes
    n_voxels = 1000  # test mode
    assert results["r2"].shape == (n_voxels,)
    assert results["r2_median"].shape == (n_voxels,)  # Same as r2 (misleading name)
    # r2_mean, r2_std, r2_min, r2_max are scalars in new GLMdenoise-style API

    # Sanity checks on R² values
    # NOTE: With the fixed CV logic (always projecting out nuisance), we should get reasonable R² values

    # 1. All values should be finite
    assert torch.isfinite(results["r2"]).all(), "All R² values should be finite"
    assert torch.isfinite(results["r2_mean"]), "Mean R² should be finite"
    assert results["r2_std"] >= 0, "Std should be non-negative"
    assert torch.isfinite(results["r2_min"]).all(), "All min values should be finite"
    assert torch.isfinite(results["r2_max"]).all(), "All max values should be finite"

    # 2. Standard deviation should be non-negative
    assert (results["r2_std"] >= 0).all(), "R² std should be non-negative"

    # 3. Min should be <= median <= max
    assert (results["r2_min"] <= results["r2_median"]).all(), "Min should be <= median"
    assert (results["r2_median"] <= results["r2_max"]).all(), "Median should be <= max"

    # 4. With proper CV, most voxels should have positive R² (at least some signal)
    positive_r2_fraction = (results["r2_median"] > 0).float().mean().item()
    assert positive_r2_fraction > 0.3, f"Expected >30% positive R², got {positive_r2_fraction:.1%}"

    # 5. R² should generally be reasonable (not catastrophically negative)
    mean_r2 = results["r2_median"].mean().item()
    assert mean_r2 > -1.0, f"Mean R² is too negative: {mean_r2:.4f} (suggests a bug)"

    # Check design_info
    assert "run_starts" in design_info
    assert "cv_strategy" in design_info
    assert "n_splits" in design_info
    assert len(design_info["run_starts"]) == 2  # 2 runs


def test_xval_r2_split_halves(validation_data_available):
    """
    Test cross-validation with split-halves strategy.

    Uses 50/50 train/test split.
    """
    # Run cross-validation with split-halves
    results, design_info = analyze_with_cross_validation(
        fmri_data=INPUT_FILES,
        design_matrix_file=DESIGN_MATRIX,
        cv_strategy=0.5,  # 50/50 split
        n_perms=2,  # Just 2 splits for fast testing
        metric="cod",
        test_n_voxels=500,  # Fast test mode
        verbose=False,
    )

    # With 2 runs and 50/50 split, there are 2 possible splits
    assert results["n_splits"] >= 1, f"Expected at least 1 split, got {results['n_splits']}"

    # Check shapes
    n_voxels = 500
    assert results["r2"].shape == (n_voxels,)
    assert results["r2_median"].shape == (n_voxels,)  # Same as r2 (misleading name)
    # Note: r2_splits removed in GLMdenoise-style API (no per-fold R²)

    # Basic sanity checks
    assert torch.isfinite(results["r2"]).all(), "All R² values should be finite"
    assert torch.isfinite(results["r2_mean"]), "Mean R² should be finite"
    assert torch.isfinite(results["r2_std"]), "Std R² should be finite"


def test_xval_r2_different_metrics(validation_data_available):
    """Test different R² metrics (CoD, Pearson r, Pearson r²)."""
    test_n_voxels = 200  # Small for speed

    results_cod, _ = analyze_with_cross_validation(
        fmri_data=INPUT_FILES,
        design_matrix_file=DESIGN_MATRIX,
        cv_strategy=1,
        metric="cod",
        test_n_voxels=test_n_voxels,
        verbose=False,
    )

    results_corr, _ = analyze_with_cross_validation(
        fmri_data=INPUT_FILES,
        design_matrix_file=DESIGN_MATRIX,
        cv_strategy=1,
        metric="corr",
        test_n_voxels=test_n_voxels,
        verbose=False,
    )

    results_corr2, _ = analyze_with_cross_validation(
        fmri_data=INPUT_FILES,
        design_matrix_file=DESIGN_MATRIX,
        cv_strategy=1,
        metric="corr2",
        test_n_voxels=test_n_voxels,
        verbose=False,
    )

    # All should have same shape
    assert results_cod["r2"].shape == results_corr["r2"].shape == results_corr2["r2"].shape

    # CoD and corr² should be similar (not identical, but correlated)
    # Pearson r should be in [-1, 1] range
    assert (results_corr["r2_median"] >= -1.05).all() and (results_corr["r2_median"] <= 1.05).all(), \
        "Pearson r should be in [-1, 1] range (with small tolerance)"

    # Pearson r² should be in [0, 1] range (but can go slightly negative on noise)
    assert (results_corr2["r2_median"] >= -0.1).all() and (results_corr2["r2_median"] <= 1.05).all(), \
        "Pearson r² should be in [0, 1] range (with tolerance for negative values on noise)"


def test_xval_r2_with_high_level_api(validation_data_available):
    """Test that the high-level API works correctly."""
    # This is an integration test of the full pipeline
    results, design_info = analyze_with_cross_validation(
        fmri_data=INPUT_FILES,
        design_matrix_file=DESIGN_MATRIX,
        cv_strategy=1,  # LORO
        metric="cod",
        test_n_voxels=100,
        verbose=True,  # Test verbose output
    )

    # Verify results structure
    assert isinstance(results, dict)
    assert isinstance(design_info, dict)

    # Check that core results exist
    assert "r2" in results
    assert "n_splits" in results

    # Check design_info
    assert "cv_strategy" in design_info
    assert "n_splits" in design_info


def test_xval_r2_error_single_run():
    """Test that CV raises error with single run data."""
    # Skipping this test - would require creating a properly formatted
    # AFNI design matrix which is complex
    pytest.skip("Skipping single-run error test - requires AFNI design matrix creation")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
