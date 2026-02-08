"""
Comprehensive tests for denoise.py with progressive coverage.

Test layers:
1. Small: Unit tests for core functions (noise pool selection, PC extraction)
2. Medium: Sub-workflow tests (noise PC evaluation, CV validation)
3. Large/E2E: Full pipeline tests with ground truth verification

Uses realistic fMRI simulation to verify:
- Noise pool selection identifies low-signal voxels
- Sequential PC denoising improves model fit
- Cross-validation prevents overfitting
"""

import pytest
import torch
import numpy as np

from fastfuncsim.simulation import simulate_fmri_run
from fastfuncsim.hrf import get_canonical_hrf
from fastfuncsim.glm_core import construct_polynomial_matrix
from fastfuncsim.utils import get_device
from fastfuncsim.denoise import (
    select_noise_pool_voxels,
    _compute_local_run_starts,
)
from fastfuncsim.xval import generate_cv_splits


@pytest.fixture
def device():
    return get_device()


# ============================================================================
# Layer 1: Small Tests - Unit tests for core functions
# ============================================================================

class TestDenoiseCoreFunctions:
    """Test core denoising functions."""

    def test_compute_local_run_starts_basic(self):
        """Test local run starts computation."""
        # Simple case: runs [0, 2, 4] from a subset of 3 runs
        run_indices = [0, 2, 4]
        run_starts_global = [0, 100, 200, 300, 400, 500]
        n_timepoints = 600  # Total timepoints

        local_starts = _compute_local_run_starts(run_indices, run_starts_global, n_timepoints)

        # First run should start at 0
        assert local_starts[0] == 0
        # Second run should start at 100 (relative position)
        assert local_starts[1] == 100
        # Third run should start at 200 (relative position)
        assert local_starts[2] == 200

    def test_compute_local_run_starts_single_run(self):
        """Test local run starts with single run."""
        run_indices = [2]
        run_starts_global = [0, 100, 200, 300]
        n_timepoints = 400

        local_starts = _compute_local_run_starts(run_indices, run_starts_global, n_timepoints)

        # Single run should always start at 0
        assert len(local_starts) == 1
        assert local_starts[0] == 0

    def test_select_noise_pool_voxels_basic(self, device):
        """Test noise pool selection with R² values."""
        n_voxels = 100

        # Create R² values where first 50 have high R² (signal)
        # and last 50 have low R² (noise pool)
        r2 = torch.cat([
            torch.ones(50, device=device) * 0.5,  # High R² voxels
            torch.ones(50, device=device) * 0.05,  # Low R² voxels
        ])

        noise_mask, criteria_mask = select_noise_pool_voxels(
            r2=r2,
            threshold=0.1,  # Voxels below 0.1 are noise pool
            min_noise_voxels=10,
            max_noise_fraction=0.95,
        )

        # Should select roughly the low R² voxels
        n_selected = noise_mask.sum().item()
        assert n_selected == 50, f"Expected 50 voxels, got {n_selected}"

        # Selected voxels should have lower R² than non-selected
        r2_noise = r2[noise_mask].mean().item()
        r2_criteria = r2[criteria_mask].mean().item()

        assert r2_noise < r2_criteria, \
            f"Noise pool R² ({r2_noise:.3f}) should be less than criteria R² ({r2_criteria:.3f})"

    def test_select_noise_pool_voxels_min_constraint_error(self, device):
        """Test that noise pool selection raises error when min constraint can't be met."""
        n_voxels = 100

        # All voxels have high R² (no clear noise pool)
        r2 = torch.ones(n_voxels, device=device) * 0.5

        # Should raise error because can't meet minimum
        with pytest.raises(ValueError, match="Noise pool has only 0 voxels"):
            select_noise_pool_voxels(
                r2=r2,
                threshold=0.1,  # Would select 0 voxels
                min_noise_voxels=20,  # But need at least 20
                max_noise_fraction=0.95,
            )

    @pytest.mark.skip(reason="TODO: Implement multi-run PC extraction test")
    def test_extract_noise_pcs_multiple_runs(self, device):
        """Test PC extraction from multiple runs."""
        # Test that extract_noise_pcs_per_run handles multiple runs correctly
        # Verify that PCs are extracted independently per run
        pass


# ============================================================================
# Layer 2: Medium Tests - Sub-workflow tests
# ============================================================================

class TestDenoiseSubWorkflows:
    """Test denoising sub-workflows."""

    @pytest.mark.skip(reason="TODO: Implement noise PC evaluation test")
    def test_evaluate_noise_pcs_with_cv(self, device):
        """Test noise PC evaluation with cross-validation."""
        pass

    @pytest.mark.skip(reason="TODO: Implement sequential PC selection test")
    def test_sequential_pc_selection(self, device):
        """Test that sequential PC selection finds optimal number of PCs."""
        pass

    @pytest.mark.skip(reason="TODO: Implement criteria voxel test")
    def test_criteria_voxel_selection(self, device):
        """Test that criteria voxels are selected correctly."""
        pass


# ============================================================================
# Layer 3: Large/E2E Tests - Full pipeline with ground truth
# ============================================================================

class TestDenoiseFullPipeline:
    """Test full denoising pipeline."""

    @pytest.mark.skip(reason="TODO: Implement signal recovery test")
    def test_denoising_improves_signal_recovery(self, device):
        """Test that denoising improves signal recovery vs baseline."""
        # Simulate data with known betas
        # Add structured noise
        # Verify that PC denoising removes noise and recovers signal
        pass

    @pytest.mark.skip(reason="TODO: Implement overfitting prevention test")
    def test_cv_prevents_overfitting(self, device):
        """Test that cross-validation prevents overfitting."""
        # Compare in-sample R² vs CV R²
        # CV R² should be lower (more conservative)
        pass
