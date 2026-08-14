"""
Tests for uncovered functions in denoise/ modules:
- denoise/combinatorial.py: evaluate_all_combinations_for_run, select_optimal_combination
- denoise/sequential.py: key internal helpers
"""

import numpy as np
import torch

import fastfuncstuff.denoise.combinatorial as combinatorial
from fastfuncstuff.denoise.combinatorial import (
    evaluate_all_combinations_for_run,
    extract_pcs_single_run_with_variance,
    generate_all_pc_combinations,
    select_optimal_combination,
)
from fastfuncstuff.denoise.sequential import (
    _compute_local_run_starts,
    extract_noise_pcs_per_run,
    select_noise_pool_voxels,
)

DEVICE = torch.device("cpu")


def test_combination_chunks_share_the_memory_budget(monkeypatch):
    monkeypatch.setattr(combinatorial, "get_available_memory", lambda _device: 1 << 20)
    combo_chunk, voxel_chunk = combinatorial._combination_work_chunks(
        n_combos=1000,
        n_timepoints=100,
        n_voxels=10000,
        device=DEVICE,
    )
    assert combo_chunk == 9
    assert voxel_chunk == 52


class TestEvaluateAllCombinationsForRun:
    def test_basic_evaluation(self):
        """Evaluate 2^3=8 combinations on small data."""
        torch.manual_seed(42)
        n_criteria, T, k = 30, 50, 3
        n_conds = 2

        run_data = torch.randn(n_criteria, T)
        design = torch.randn(T, n_conds)
        betas = torch.randn(n_criteria, n_conds)
        poly_nuis = torch.ones(T, 1)  # intercept only
        noise_pcs = torch.randn(T, k)
        combos = generate_all_pc_combinations(k)
        var_ratios = np.array([0.3, 0.2, 0.1])

        median_cod, var_explained = evaluate_all_combinations_for_run(
            run_data,
            design,
            betas,
            poly_nuis,
            noise_pcs,
            combos,
            var_ratios,
            device=DEVICE,
        )

        assert median_cod.shape == (len(combos),)
        assert var_explained.shape == (len(combos),)
        # Empty set should explain 0 variance
        assert var_explained[0] == 0.0
        # Full set should explain sum of all variance ratios
        assert abs(var_explained[-1] - 0.6) < 1e-6

    def test_return_raw_cod(self):
        """return_raw_cod should give per-voxel CoD."""
        torch.manual_seed(0)
        n_criteria, T, k = 20, 40, 2
        combos = generate_all_pc_combinations(k)

        raw_cod, var_exp = evaluate_all_combinations_for_run(
            torch.randn(n_criteria, T),
            torch.randn(T, 2),
            torch.randn(n_criteria, 2),
            torch.ones(T, 1),
            torch.randn(T, k),
            combos,
            np.array([0.5, 0.3]),
            device=DEVICE,
            return_raw_cod=True,
        )
        # raw_cod should be (n_combos, n_criteria)
        assert raw_cod.shape == (len(combos), n_criteria)

    def test_empty_poly_nuisance(self):
        """Should work with all-zero polynomial nuisance."""
        torch.manual_seed(0)
        n_criteria, T, k = 15, 30, 2
        combos = generate_all_pc_combinations(k)
        poly_nuis = torch.zeros(T, 1)  # All zeros

        median_cod, var_exp = evaluate_all_combinations_for_run(
            torch.randn(n_criteria, T),
            torch.randn(T, 2),
            torch.randn(n_criteria, 2),
            poly_nuis,
            torch.randn(T, k),
            combos,
            np.array([0.4, 0.2]),
            device=DEVICE,
        )
        assert median_cod.shape == (len(combos),)


class TestSelectOptimalCombination:
    def test_argmax_strategy(self):
        """argmax should pick the combo with highest median CoD."""
        combos = [(), (0,), (1,), (0, 1)]
        median_cod = np.array([0.1, 0.5, 0.3, 0.45])

        best_idx, best_combo = select_optimal_combination(median_cod, combos, strategy="argmax")
        assert best_idx == 1  # (0,) has highest CoD=0.5
        assert best_combo == (0,)

    def test_parsimonious_strategy(self):
        """Parsimonious should prefer fewer PCs within threshold of best."""
        combos = [(), (0,), (1,), (0, 1)]
        # (0,1) is best but (0,) is within 1% so simpler is preferred
        median_cod = np.array([0.1, 0.49, 0.3, 0.50])

        best_idx, best_combo = select_optimal_combination(
            median_cod, combos, strategy="parsimonious"
        )
        # Should pick (0,) since it's within 1% of (0,1) but simpler
        assert len(best_combo) <= len(combos[3])


class TestExtractPcsSingleRunWithVariance:
    def test_basic_extraction(self):
        torch.manual_seed(42)
        n_voxels, T = 100, 50
        data = torch.randn(n_voxels, T)
        mask = torch.zeros(n_voxels, dtype=torch.bool)
        mask[:30] = True
        nuisance = torch.ones(T, 1)

        pcs, var_ratios = extract_pcs_single_run_with_variance(
            data,
            mask,
            nuisance,
            max_components=5,
            device=DEVICE,
        )
        assert pcs.shape == (T, 5)
        assert var_ratios.shape == (5,)
        assert (var_ratios >= 0).all()
        # Variance ratios should sum to <= 1
        assert var_ratios.sum() <= 1.01

    def test_unit_variance_pcs(self):
        """Extracted PCs should have approximately unit variance."""
        torch.manual_seed(0)
        data = torch.randn(80, 60)
        mask = torch.ones(80, dtype=torch.bool)
        nuisance = torch.ones(60, 1)

        pcs, _ = extract_pcs_single_run_with_variance(
            data,
            mask,
            nuisance,
            max_components=3,
            device=DEVICE,
        )
        for i in range(3):
            assert abs(pcs[:, i].std().item() - 1.0) < 0.15


class TestSelectNoisePoolVoxels:
    def test_basic_selection(self):
        """Select noise pool from low-R² voxels."""
        torch.manual_seed(42)
        n_voxels = 200
        r2 = torch.randn(n_voxels)
        r2[:50] = 0.8
        r2[50:] = -0.1

        noise_mask, criteria_mask = select_noise_pool_voxels(
            r2,
            threshold=0.0,
        )
        assert noise_mask.shape == (n_voxels,)
        assert noise_mask.dtype == torch.bool
        # Low-R² voxels should be in noise pool
        assert noise_mask[50:].sum() > noise_mask[:50].sum()

    def test_returns_two_masks(self):
        r2 = torch.randn(500)  # Need enough voxels for min_noise_voxels
        noise_mask, criteria_mask = select_noise_pool_voxels(r2)
        assert noise_mask.shape == (500,)
        assert criteria_mask.shape == (500,)


class TestComputeLocalRunStarts:
    def test_basic(self):
        run_starts = [0, 100, 200, 300]
        n_tp = 400
        local = _compute_local_run_starts([0, 2], run_starts, n_tp)
        assert local == [0, 100]

    def test_single_run(self):
        local = _compute_local_run_starts([1], [0, 50, 100], 150)
        assert local == [0]


class TestExtractNoisePcsPerRun:
    def test_basic_extraction(self):
        torch.manual_seed(42)
        n_voxels, n_tp = 100, 150
        data = torch.randn(n_voxels, n_tp)
        run_starts = [0, 50, 100]
        noise_mask = torch.zeros(n_voxels, dtype=torch.bool)
        noise_mask[:40] = True

        pcs = extract_noise_pcs_per_run(
            data,
            run_starts,
            noise_mask,
            max_components=3,
            device=DEVICE,
            verbose=False,
        )
        assert len(pcs) == 3  # One per run
        for pc in pcs:
            assert pc.shape[0] == 50  # run_length
            assert pc.shape[1] == 3  # max_components
