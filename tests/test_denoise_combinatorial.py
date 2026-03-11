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

from fastfuncsim.denoise_combinatorial import (
    CombinatorialDenoiseResults,
    CombinatorialDenoiseRunResult,
    evaluate_all_combinations_for_run,
    extract_pcs_single_run_with_variance,
    fit_combinatorial_denoising,
    generate_all_pc_combinations,
    select_optimal_combination,
)
from fastfuncsim.utils import get_device


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


class TestEvaluateAllCombinations:
    """Test batch evaluation of all PC combinations."""

    def test_basic_evaluation(self, device):
        n_voxels = 20
        run_length = 50
        n_conditions = 3
        k = 3

        run_data_criteria = torch.randn(n_voxels, run_length, device=device)
        run_design = torch.randn(run_length, n_conditions, device=device)
        betas_criteria = torch.randn(n_voxels, n_conditions, device=device) * 0.5
        noise_pcs = torch.randn(run_length, k, device=device)
        noise_pcs = noise_pcs / noise_pcs.norm(dim=0, keepdim=True)
        poly_nuisance = torch.ones(run_length, 1, device=device)
        variance_ratios = np.array([0.4, 0.3, 0.2])
        combinations = generate_all_pc_combinations(k)

        median_cod, all_cod, var_explained = evaluate_all_combinations_for_run(
            run_data_criteria=run_data_criteria,
            run_design=run_design,
            betas_criteria=betas_criteria,
            noise_pcs=noise_pcs,
            poly_nuisance=poly_nuisance,
            variance_ratios=variance_ratios,
            combinations=combinations,
            device=device,
            verbose=False,
        )

        assert median_cod.shape == (len(combinations),)
        assert all_cod.shape == (len(combinations), n_voxels)
        assert var_explained.shape == (len(combinations),)

    def test_finds_noncontiguous_subset(self, device):
        n_voxels = 30
        run_length = 100
        n_conditions = 2
        k = 4

        np.random.seed(42)
        torch.manual_seed(42)

        run_design = torch.randn(run_length, n_conditions, device=device)
        true_betas = torch.randn(n_voxels, n_conditions, device=device) * 0.3
        signal = run_design @ true_betas.T

        noise_pcs = torch.randn(run_length, k, device=device)
        noise_pcs = noise_pcs / noise_pcs.norm(dim=0, keepdim=True)
        noise_weights = torch.zeros(n_voxels, k, device=device)
        noise_weights[:, 1] = 0.5
        noise_weights[:, 3] = 0.3
        structured_noise = noise_pcs @ noise_weights.T
        random_noise = torch.randn(n_voxels, run_length, device=device) * 0.2

        run_data = signal + structured_noise + random_noise
        poly_nuisance = torch.ones(run_length, 1, device=device)
        variance_ratios = np.array([0.1, 0.3, 0.1, 0.2])
        combinations = generate_all_pc_combinations(k)

        median_cod, all_cod, var_explained = evaluate_all_combinations_for_run(
            run_data_criteria=run_data,
            run_design=run_design,
            betas_criteria=true_betas,
            noise_pcs=noise_pcs,
            poly_nuisance=poly_nuisance,
            variance_ratios=variance_ratios,
            combinations=combinations,
            device=device,
            verbose=False,
        )

        best_idx = np.argmax(median_cod)
        best_combo = combinations[best_idx]

        assert 1 in best_combo or 3 in best_combo


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


class TestFitCombinatorialDenoising:
    """Test the main fitting function."""

    def test_basic_fit(self, device):
        n_voxels = 50
        run_length = 80
        n_conditions = 2
        n_runs = 3

        np.random.seed(42)
        torch.manual_seed(42)

        data_per_run = []
        design_per_run = []
        for _ in range(n_runs):
            data_per_run.append(torch.randn(n_voxels, run_length, device=device))
            design_per_run.append(torch.randn(run_length, n_conditions, device=device))

        data = torch.cat(data_per_run, dim=1)
        design = torch.cat(design_per_run, dim=0)
        run_starts = [i * run_length for i in range(n_runs)]

        noise_pool_mask = torch.ones(n_voxels, dtype=torch.bool, device=device)

        results = fit_combinatorial_denoising(
            data=data,
            design=design,
            run_starts=run_starts,
            noise_pool_mask=noise_pool_mask,
            polort=2,
            max_pcs=4,
            device=device,
            verbose=False,
        )

        assert isinstance(results, CombinatorialDenoiseResults)
        assert len(results.per_run_results) == n_runs
        assert results.noise_pool_mask.shape == (n_voxels,)

    def test_with_noise_improves_fit(self, device):
        n_voxels = 40
        run_length = 60
        n_conditions = 2
        n_runs = 3
        k = 3

        np.random.seed(123)
        torch.manual_seed(123)

        noise_pcs_template = torch.randn(run_length, k, device=device)
        noise_pcs_template = noise_pcs_template / noise_pcs_template.norm(dim=0, keepdim=True)

        data_per_run = []
        design_per_run = []
        for _run_idx in range(n_runs):
            run_design = torch.randn(run_length, n_conditions, device=device)
            run_signal = run_design @ torch.randn(n_voxels, n_conditions, device=device).T * 0.5
            noise_weights = torch.zeros(n_voxels, k, device=device)
            noise_weights[:, 0] = 0.4
            noise_weights[:, 2] = 0.3
            run_noise = noise_pcs_template @ noise_weights.T
            run_random = torch.randn(n_voxels, run_length, device=device) * 0.1
            data_per_run.append(run_signal + run_noise + run_random)
            design_per_run.append(run_design)

        data = torch.cat(data_per_run, dim=1)
        design = torch.cat(design_per_run, dim=0)
        run_starts = [i * run_length for i in range(n_runs)]

        noise_pool_mask = torch.ones(n_voxels, dtype=torch.bool, device=device)

        results = fit_combinatorial_denoising(
            data=data,
            design=design,
            run_starts=run_starts,
            noise_pool_mask=noise_pool_mask,
            polort=2,
            max_pcs=k,
            device=device,
            verbose=False,
        )

        assert isinstance(results, CombinatorialDenoiseResults)
        for run_result in results.per_run_results:
            assert isinstance(run_result, CombinatorialDenoiseRunResult)
            optimal = run_result.optimal_combination
            assert 0 in optimal or 2 in optimal or len(optimal) == 0
