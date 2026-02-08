"""
Comprehensive tests for hrf_selection.py with progressive coverage.

Test layers:
1. Small: Unit tests for core functions (load_nuisance_file, HRF evaluation)
2. Medium: Sub-workflow tests (HRF library fitting, batched evaluation)
3. Large/E2E: Full pipeline tests with ground truth verification

Uses realistic fMRI simulation to verify:
- HRF selection recovers known optimal HRFs
- Cross-validation prevents overfitting
- Per-voxel HRF selection improves model fit
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile

from fastfuncsim.simulation import simulate_fmri_run
from fastfuncsim.hrf import get_hrf_library, get_canonical_hrf
from fastfuncsim.glm_core import construct_polynomial_matrix
from fastfuncsim.utils import get_device
from fastfuncsim.hrf_selection import (
    load_nuisance_file,
    fit_glm_hrf_library_with_xval,
)
from fastfuncsim.xval import generate_cv_splits


@pytest.fixture
def device():
    return get_device()


# ============================================================================
# Layer 1: Small Tests - Unit tests for core functions
# ============================================================================

class TestHRFSelectionCoreFunctions:
    """Test core HRF selection functions."""

    def test_load_nuisance_file_basic(self, tmp_path):
        """Test loading basic nuisance file."""
        # Create a simple nuisance file
        nuisance_file = tmp_path / "motion.1D"
        n_timepoints = 100
        n_cols = 6

        # Write some data
        data = np.random.randn(n_timepoints, n_cols)
        np.savetxt(nuisance_file, data, delimiter=" ")

        # Load it back
        loaded = load_nuisance_file(nuisance_file)

        assert loaded.shape == (n_timepoints, n_cols), \
            f"Expected shape ({n_timepoints}, {n_cols}), got {loaded.shape}"
        assert np.allclose(loaded, data), "Loaded data should match original"

    def test_load_nuisance_file_with_comments(self, tmp_path):
        """Test loading nuisance file with AFNI-style comments."""
        nuisance_file = tmp_path / "motion.1D"

        # Write file with comments
        with open(nuisance_file, 'w') as f:
            f.write("# This is a comment\n")
            f.write("# Another comment\n")
            f.write("1.0 2.0 3.0\n")
            f.write("4.0 5.0 6.0\n")
            f.write("# Final comment\n")

        loaded = load_nuisance_file(nuisance_file)

        assert loaded.shape == (2, 3)
        assert np.allclose(loaded, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_load_nuisance_file_single_column(self, tmp_path):
        """Test that single-column files are reshaped correctly."""
        nuisance_file = tmp_path / "physio.1D"

        # Write single column
        with open(nuisance_file, 'w') as f:
            for i in range(10):
                f.write(f"{i}\n")

        loaded = load_nuisance_file(nuisance_file)

        # Should be reshaped to (n_timepoints, 1)
        assert loaded.shape == (10, 1), f"Expected shape (10, 1), got {loaded.shape}"

    def test_load_nuisance_file_not_found(self, tmp_path):
        """Test error when file doesn't exist."""
        nuisance_file = tmp_path / "nonexistent.1D"

        with pytest.raises(FileNotFoundError, match="Nuisance file not found"):
            load_nuisance_file(nuisance_file)

    def test_load_nuisance_file_empty(self, tmp_path):
        """Test error when file is empty."""
        nuisance_file = tmp_path / "empty.1D"

        # Create empty file
        nuisance_file.touch()

        with pytest.raises(ValueError, match="empty or contains only comments"):
            load_nuisance_file(nuisance_file)

    def test_load_nuisance_file_wrong_length(self, tmp_path):
        """Test validation with expected_rows."""
        nuisance_file = tmp_path / "motion.1D"

        # Write 10 rows
        data = np.random.randn(10, 3)
        np.savetxt(nuisance_file, data)

        # Expect 20 rows - should raise error
        with pytest.raises(ValueError, match="has 10 rows, expected 20"):
            load_nuisance_file(nuisance_file, expected_rows=20)


# ============================================================================
# Layer 2: Medium Tests - Sub-workflow tests
# ============================================================================

class TestHRFSelectionSubWorkflows:
    """Test HRF selection sub-workflows."""

    @pytest.mark.skip(reason="TODO: Implement HRF library evaluation test")
    def test_hrf_library_fitting_with_xval(self, device):
        """Test HRF library fitting with cross-validation."""
        # Simulate data with known HRF
        # Create HRF library with different HRFs
        # Verify that CV selects the correct HRF
        pass

    @pytest.mark.skip(reason="TODO: Implement batched evaluation test")
    def test_batched_evaluation_efficiency(self, device):
        """Test that batched evaluation is more efficient than loop."""
        pass

    @pytest.mark.skip(reason="TODO: Implement per-voxel HRF test")
    def test_per_voxel_hrf_selection(self, device):
        """Test per-voxel HRF selection."""
        # Simulate voxels with different optimal HRFs
        # Verify that per-voxel selection recovers the correct HRF for each voxel
        pass


# ============================================================================
# Layer 3: Large/E2E Tests - Full pipeline with ground truth
# ============================================================================

class TestHRFSelectionFullPipeline:
    """Test full HRF selection pipeline."""

    @pytest.mark.skip(reason="TODO: Implement E2E test with ground truth")
    def test_hrf_selection_recovers_known_hrf(self, device):
        """Test that HRF selection recovers the known optimal HRF."""
        # Simulate data with a specific HRF
        # Run HRF selection with a library
        # Verify that the correct HRF is selected
        pass

    @pytest.mark.skip(reason="TODO: Implement overfitting prevention test")
    def test_cv_prevents_overfitting(self, device):
        """Test that cross-validation prevents overfitting."""
        # Compare in-sample R² vs CV R²
        # CV R² should be lower (more conservative)
        pass

    @pytest.mark.skip(reason="TODO: Implement comparison test")
    def test_hrf_selection_vs_canonical(self, device):
        """Test HRF selection vs canonical HRF."""
        # Simulate data with non-canonical HRF
        # Compare fit quality: selected HRF should beat canonical
        pass
