"""
Comprehensive tests for denoise_combinatorial.py

Test layers:
1. Small: Unit tests for core functions (combination generation, PC extraction)
2. Medium: Sub-workflow tests (batch evaluation, selection strategies)
3. Large/E2E: Full pipeline tests with ground truth verification
"""

import numpy as np
import pytest
import torch

from fastfuncstuff.denoise.combinatorial import (
    CombinatorialDenoiseResults,
    CombinatorialDenoiseRunResult,
    extract_pcs_single_run_with_variance,
    generate_all_pc_combinations,
    select_optimal_combination,
)
from fastfuncstuff.utils import get_device


@pytest.fixture
def device():
    return get_device()


class TestGenerateAllPcCombinations:
    """Test combination generation."""

    def test_k_zero(self):
        combos = generate_all_pc_combinations(0)
        assert combos == [()]

    def test_k_one(self):
        combos = generate_all_pc_combinations(1)
        assert combos == [(), (0,)]

    def test_k_two(self):
        combos = generate_all_pc_combinations(2)
        assert len(combos) == 4
        assert () in combos
        assert (0,) in combos
        assert (1,) in combos
        assert (0, 1) in combos

    def test_k_five(self):
        combos = generate_all_pc_combinations(5)
        assert len(combos) == 32
        sizes = [len(c) for c in combos]
        assert sizes == sorted(sizes)
        assert len(set(combos)) == len(combos)

    def test_k_ten(self):
        combos = generate_all_pc_combinations(10)
        assert len(combos) == 1024


class TestExtractPcsSingleRun:
    """Test PC extraction from single run."""

    def test_basic_extraction(self, device):
        run_length = 100
        n_voxels = 50
        max_components = 5

        run_data = torch.randn(n_voxels, run_length, device=device)
        nuisance = torch.ones(run_length, 1, device=device)
        noise_pool_mask = torch.ones(n_voxels, dtype=torch.bool, device=device)

        pcs, variance_ratios = extract_pcs_single_run_with_variance(
            run_data=run_data,
            noise_pool_mask=noise_pool_mask,
            nuisance=nuisance,
            max_components=max_components,
            device=device,
        )

        assert pcs.shape == (run_length, max_components)
        assert variance_ratios.shape == (max_components,)
        assert variance_ratios.sum() <= 1.0 + 1e-6
        assert variance_ratios[0] >= variance_ratios[1]

    def test_empty_noise_pool(self, device):
        run_length = 100
        n_voxels = 50

        run_data = torch.randn(n_voxels, run_length, device=device)
        nuisance = torch.ones(run_length, 1, device=device)
        noise_pool_mask = torch.zeros(n_voxels, dtype=torch.bool, device=device)

        pcs, variance_ratios = extract_pcs_single_run_with_variance(
            run_data=run_data,
            noise_pool_mask=noise_pool_mask,
            nuisance=nuisance,
            max_components=5,
            device=device,
        )

        assert torch.all(pcs == 0)
        assert np.all(variance_ratios == 0)

    def test_nuisance_projection(self, device):
        run_length = 100
        n_voxels = 50

        time = torch.arange(run_length, device=device).float() / run_length
        run_data = torch.randn(n_voxels, run_length, device=device) + 10 * time.unsqueeze(0)
        nuisance = time.unsqueeze(1)
        noise_pool_mask = torch.ones(n_voxels, dtype=torch.bool, device=device)

        pcs, variance_ratios = extract_pcs_single_run_with_variance(
            run_data=run_data,
            noise_pool_mask=noise_pool_mask,
            nuisance=nuisance,
            max_components=3,
            device=device,
        )

        correlation = (pcs[:, 0] * time).sum() / (pcs[:, 0].norm() * time.norm())
        assert abs(correlation.item()) < 0.3


class TestSelectOptimalCombination:
    """Test optimal combination selection."""

    def test_argmax_strategy(self):
        k = 3
        combinations = generate_all_pc_combinations(k)
        median_cod = np.zeros(len(combinations))
        median_cod[combinations.index((1, 2))] = 0.5
        median_cod[combinations.index((0,))] = 0.3

        idx, combo = select_optimal_combination(median_cod, combinations, "argmax")
        assert combo == (1, 2)

    def test_parsimonious_strategy(self):
        k = 4
        combinations = generate_all_pc_combinations(k)
        median_cod = np.zeros(len(combinations))
        median_cod[combinations.index((0, 1, 2, 3))] = 0.5
        median_cod[combinations.index((0, 2))] = 0.495
        median_cod[combinations.index((1,))] = 0.49

        idx, combo = select_optimal_combination(median_cod, combinations, "parsimonious")
        assert combo == (0, 2)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown selection strategy"):
            select_optimal_combination(np.zeros(8), generate_all_pc_combinations(3), "invalid")
