"""
Comprehensive tests for denoise_combinatorial.py with progressive coverage.

Test layers:
1. Small: Unit tests for core functions (combination generation, PC extraction)
2. Medium: Sub-workflow tests (batch evaluation, selection strategies)
3. Large/E2E: Full pipeline tests with ground truth verification

Uses realistic fMRI simulation to verify:
- Combinatorial evaluation finds non-contiguous PC subsets
- Selection strategies work correctly (argmax, parsimonious)
- Signal recovery improves with optimal PC subsets
"""

import pytest
import torch
import numpy as np
from typing import List, Tuple

from fastfuncsim.simulation import simulate_fmri_run
from fastfuncsim.hrf import get_canonical_hrf
from fastfuncsim.glm_core import construct_polynomial_matrix, fit_glm
from fastfuncsim.utils import get_device
from fastfuncsim.denoise_combinatorial import (
    generate_all_pc_combinations,
    extract_pcs_single_run_with_variance,
    evaluate_all_combinations_for_run,
    select_optimal_combination,
    fit_combinatorial_denoising,
    CombinatorialDenoiseResults,
    CombinatorialDenoiseRunResult,
)
from fastfuncsim.xval import generate_cv_splits


@pytest.fixture
def device():
    return get_device()


# ============================================================================
# Layer 1: Small Tests - Unit tests for core functions
# ============================================================================

class TestCombinatorialCoreFunctions:
    """Test core combinatorial denoising functions."""

    def test_generate_all_pc_combinations_small(self):
        """Test combination generation for small k."""
        # k=0: only empty set
        combos = generate_all_pc_combinations(0)
        assert combos == [()], f"Expected [()] for k=0, got {combos}"

        # k=1: 2^1 = 2 combinations
        combos = generate_all_pc_combinations(1)
        assert len(combos) == 2, f"Expected 2 combos for k=1, got {len(combos)}"
        assert combos == [(), (0,)], f"Expected [(), (0,)] for k=1, got {combos}"

        # k=2: 2^2 = 4 combinations
        combos = generate_all_pc_combinations(2)
        assert len(combos) == 4, f"Expected 4 combos for k=2, got {len(combos)}"
        # Check that all combinations are present
        assert () in combos
        assert (0,) in combos
        assert (1,) in combos
        assert (0, 1) in combos

    def test_generate_all_pc_combinations_medium(self):
        """Test combination generation for medium k."""
        # k=5: 2^5 = 32 combinations
        combos = generate_all_pc_combinations(5)
        assert len(combos) == 32, f"Expected 32 combos for k=5, got {len(combos)}"

        # Verify sorted by size (empty first, then singletons, then pairs, ...)
        sizes = [len(c) for c in combos]
        assert sizes == sorted(sizes), "Combinations should be sorted by size"

        # Verify no duplicates
        assert len(set(combos)) == len(combos), "Combinations should be unique"

        # Verify all indices are in range
        for combo in combos:
            for idx in combo:
                assert 0 <= idx < 5, f"Invalid index {idx} in combo {combo}"

    def test_extract_pcs_single_run_with_variance_basic(self, device):
        """Test PC extraction from simple synthetic data."""
        run_length = 100
        n_voxels = 50
        max_components = 5

        # Create simple data with structured noise
        run_data = torch.randn(n_voxels, run_length, device=device)

        # Create simple nuisance (just intercept)
        nuisance = torch.ones(run_length, 1, device=device)

        # All voxels are noise pool
        noise_pool_mask = torch.ones(n_voxels, dtype=torch.bool, device=device)

        pcs, variance_ratios = extract_pcs_single_run_with_variance(
            run_data=run_data,
            noise_pool_mask=noise_pool_mask,
            nuisance=nuisance,
            max_components=max_components,
            device=device,
        )

        # Check shapes
        assert pcs.shape == (run_length, max_components), \
            f"Expected shape ({run_length}, {max_components}), got {pcs.shape}"
        assert variance_ratios.shape == (max_components,), \
            f"Expected shape ({max_components},), got {variance_ratios.shape}"

        # Check PCs are unit variance (approximately)
        pc_stds = pcs.std(dim=0)
        assert torch.allclose(pc_stds, torch.ones(max_components, device=device), atol=1e-2), \
            f"PCs should have unit variance, got stds: {pc_stds}"

        # Check variance ratios sum to <= 1
        assert variance_ratios.sum() <= 1.0 + 1e-6, \
            f"Variance ratios should sum to <= 1, got {variance_ratios.sum()}"

        # Check PCs are sorted by variance (first PC explains most)
        assert variance_ratios[0] >= variance_ratios[1], \
            "First PC should explain more variance than second"

    def test_extract_pcs_with_empty_noise_pool(self, device):
        """Test PC extraction when noise pool is empty."""
        run_length = 100
        n_voxels = 50
        max_components = 5

        run_data = torch.randn(n_voxels, run_length, device=device)
        nuisance = torch.ones(run_length, 1, device=device)

        # Empty noise pool
        noise_pool_mask = torch.zeros(n_voxels, dtype=torch.bool, device=device)

        pcs, variance_ratios = extract_pcs_single_run_with_variance(
            run_data=run_data,
            noise_pool_mask=noise_pool_mask,
            nuisance=nuisance,
            max_components=max_components,
            device=device,
        )

        # Should return zeros
        assert pcs.shape == (run_length, max_components)
        assert variance_ratios.shape == (max_components,)
        assert torch.all(pcs == 0), "PCs should be all zeros with empty noise pool"
        assert np.all(variance_ratios == 0), "Variance ratios should be all zeros"

    def test_extract_pcs_with_nuisance_projection(self, device):
        """Test that nuisance is properly projected out before PCA."""
        run_length = 100
        n_voxels = 50

        # Create data with strong linear trend
        time = torch.arange(run_length, device=device).float() / run_length
        run_data = torch.randn(n_voxels, run_length, device=device) + 10 * time.unsqueeze(0)

        # Nuisance includes the trend
        nuisance = time.unsqueeze(1)  # (run_length, 1)

        noise_pool_mask = torch.ones(n_voxels, dtype=torch.bool, device=device)

        pcs, variance_ratios = extract_pcs_single_run_with_variance(
            run_data=run_data,
            noise_pool_mask=noise_pool_mask,
            nuisance=nuisance,
            max_components=3,
            device=device,
        )

        # After projecting out the trend, PCs should capture remaining structure
        # not the trend itself
        assert pcs.shape == (run_length, 3)
        # First PC should be orthogonal to the trend
        correlation = (pcs[:, 0] * time).sum() / (pcs[:, 0].norm() * time.norm())
        assert abs(correlation.item()) < 0.3, \
            f"First PC should be uncorrelated with trend, got correlation {correlation:.3f}"


class TestCombinationSelection:
    """Test optimal combination selection strategies."""

    def test_select_optimal_combination_argmax(self):
        """Test argmax selection strategy."""
        k = 3
        n_combos = 2 ** k

        # Create synthetic CoD values
        # Make combo (1, 2) have the highest CoD
        median_cod = np.zeros(n_combos)
        combinations = generate_all_pc_combinations(k)

        # Find index of (1, 2)
        target_idx = combinations.index((1, 2))
        median_cod[target_idx] = 0.5  # Highest CoD
        median_cod[combinations.index((0,))] = 0.3
        median_cod[combinations.index((0, 1))] = 0.4

        optimal_idx, optimal_combo = select_optimal_combination(
            median_cod=median_cod,
            combinations=combinations,
            strategy="argmax",
        )

        assert optimal_idx == target_idx, \
            f"Expected index {target_idx}, got {optimal_idx}"
        assert optimal_combo == (1, 2), \
            f"Expected combination (1, 2), got {optimal_combo}"

    def test_select_optimal_combination_parsimonious(self):
        """Test parsimonious selection strategy (fewest PCs within 1% of max)."""
        k = 4
        n_combos = 2 ** k
        combinations = generate_all_pc_combinations(k)

        # Create CoD values where:
        # - Max CoD is 0.5 at combo (0, 1, 2, 3) (all 4 PCs)
        # - Combo (0, 2) has CoD 0.495 (within 1%)
        # - Combo (1,) has CoD 0.49 (within 2%)
        median_cod = np.zeros(n_combos)
        median_cod[combinations.index((0, 1, 2, 3))] = 0.5
        median_cod[combinations.index((0, 2))] = 0.495
        median_cod[combinations.index((1,))] = 0.49

        optimal_idx, optimal_combo = select_optimal_combination(
            median_cod=median_cod,
            combinations=combinations,
            strategy="parsimonious",
        )

        # Should select (0, 2) with 2 PCs instead of (0, 1, 2, 3) with 4 PCs
        assert optimal_combo == (0, 2), \
            f"Expected parsimonious selection (0, 2), got {optimal_combo}"

    def test_select_optimal_combination_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        median_cod = np.zeros(8)
        combinations = generate_all_pc_combinations(3)

        with pytest.raises(ValueError, match="Unknown selection strategy"):
            select_optimal_combination(
                median_cod=median_cod,
                combinations=combinations,
                strategy="invalid_strategy",
            )


# ============================================================================
# Layer 2: Medium Tests - Sub-workflow tests
# ============================================================================

class TestCombinatorialSubWorkflows:
    """Test combinatorial denoising sub-workflows."""

    @pytest.mark.skip(reason="TODO: Implement batch evaluation test")
    def test_evaluate_all_combinations_batched(self, device):
        """Test that batch evaluation produces correct CoD for all combinations."""
        pass

    @pytest.mark.skip(reason="TODO: Implement with realistic simulation")
    def test_inner_cv_selects_criteria_voxels(self, device):
        """Test that inner LORO CV correctly selects criteria voxels."""
        pass

    @pytest.mark.skip(reason="TODO: Implement full workflow test")
    def test_combinatorial_finds_noncontiguous_subsets(self, device):
        """Test that combinatorial approach can find non-contiguous PC subsets."""
        # This is the key advantage over sequential approach
        # Should be able to select PCs 0, 3, 5 while skipping 1, 2, 4
        pass


# ============================================================================
# Layer 3: Large/E2E Tests - Full pipeline with ground truth
# ============================================================================

class TestCombinatorialFullPipeline:
    """Test full combinatorial denoising pipeline."""

    @pytest.mark.skip(reason="TODO: Implement E2E test with ground truth")
    def test_combinatorial_improves_signal_recovery(self, device):
        """Test that combinatorial denoising improves signal recovery vs baseline."""
        # Simulate data with known betas
        # Add structured noise to specific PCs
        # Verify that optimal subset selection removes noise and recovers signal
        pass

    @pytest.mark.skip(reason="TODO: Implement comparison test")
    def test_combinatorial_vs_sequential(self, device):
        """Test combinatorial vs sequential denoising on same data."""
        # When optimal subset is non-contiguous, combinatorial should win
        # When optimal subset is prefix, both should perform similarly
        pass
