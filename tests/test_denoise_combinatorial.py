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


# ============================================================================
# Tests for uncovered functions (lines 253, 910-977, 1031-1117, 1161-1247,
# 1279-1344, 1370-1459, 1484-1569, 1595-1682)
# ============================================================================

from fastfuncstuff.denoise.combinatorial import (
    compute_initial_xval_r2,
    compute_optimized_xval_r2,
    compute_optimized_xval_r2_3dDenoise_style,
    evaluate_all_combinations_for_run,
    plot_combinatorial_results,
    plot_inclusion_heatmap,
    plot_plateau_curves,
    plot_singleton_contributions,
)

CPU = torch.device("cpu")


@pytest.fixture
def multi_run_setup():
    """3 runs of 40 TPs, 50 voxels, 2 conditions."""
    torch.manual_seed(99)
    n_voxels, n_runs, run_len, n_conds = 50, 3, 40, 2
    n_tp = n_runs * run_len
    data = torch.randn(n_voxels, n_tp)
    design = torch.randn(n_tp, n_conds)
    run_starts = [i * run_len for i in range(n_runs)]
    nuisance_per_run = [torch.ones(run_len, 1) for _ in range(n_runs)]
    return dict(
        data=data,
        design=design,
        run_starts=run_starts,
        nuisance_per_run=nuisance_per_run,
        n_voxels=n_voxels,
        n_tp=n_tp,
        n_conds=n_conds,
        n_runs=n_runs,
        run_len=run_len,
    )


def _make_run_results(n_runs, run_len, max_pcs=3, combos=None):
    """Helper: create per-run results and noise PCs."""
    torch.manual_seed(42)
    if combos is None:
        combos = generate_all_pc_combinations(max_pcs)
    rng = np.random.default_rng(0)
    per_run_results = []
    noise_pcs_per_run = []
    for r in range(n_runs):
        pcs = torch.randn(run_len, max_pcs)
        noise_pcs_per_run.append(pcs)
        per_run_results.append(
            CombinatorialDenoiseRunResult(
                run_idx=r,
                optimal_combination=(0,) if r % 2 == 0 else (0, 1),
                optimal_cod=0.1 + r * 0.05,
                all_cod=rng.random(len(combos)),
                all_var_explained=np.array([sum(0.1 / (i + 1) for i in combo) for combo in combos]),
                all_combinations=combos,
                explained_variance_ratios=np.array([0.3, 0.2, 0.1]),
                n_criteria_voxels=20,
            )
        )
    return per_run_results, noise_pcs_per_run


def _make_full_results(n_runs, run_len, n_voxels, max_pcs=3, singleton_only=False):
    """Helper: create CombinatorialDenoiseResults."""
    per_run_results, noise_pcs_per_run = _make_run_results(n_runs, run_len, max_pcs)
    return CombinatorialDenoiseResults(
        per_run_results=per_run_results,
        noise_pool_mask=torch.ones(n_voxels, dtype=torch.bool),
        initial_r2=torch.randn(n_voxels),
        noise_pcs_per_run=noise_pcs_per_run,
        metadata={
            "max_pcs": max_pcs,
            "n_combinations": 2**max_pcs,
            "singleton_only": singleton_only,
        },
    )


class TestEvaluateCombinationsExplicitChunk:
    def test_explicit_criteria_chunk_size(self):
        """Line 253: explicit criteria_chunk_size path."""
        torch.manual_seed(10)
        n_criteria, T, k = 25, 30, 2
        combos = generate_all_pc_combinations(k)
        median_cod, var_exp = evaluate_all_combinations_for_run(
            torch.randn(n_criteria, T),
            torch.randn(T, 2),
            torch.randn(n_criteria, 2),
            torch.ones(T, 1),
            torch.randn(T, k),
            combos,
            np.array([0.4, 0.2]),
            device=CPU,
            criteria_chunk_size=10,
        )
        assert median_cod.shape == (len(combos),)
        assert var_exp.shape == (len(combos),)


class TestComputeOptimizedXvalR2:
    def test_basic(self, multi_run_setup):
        s = multi_run_setup
        per_run_results, noise_pcs = _make_run_results(s["n_runs"], s["run_len"])
        r2 = compute_optimized_xval_r2(
            data=s["data"],
            design=s["design"],
            run_starts=s["run_starts"],
            nuisance_per_run=s["nuisance_per_run"],
            noise_pcs_per_run=noise_pcs,
            per_run_results=per_run_results,
            device=CPU,
            verbose=False,
        )
        assert r2.shape == (s["n_voxels"],)
        assert torch.isfinite(r2).all()

    def test_empty_combination(self, multi_run_setup):
        """All runs select no PCs."""
        s = multi_run_setup
        combos = generate_all_pc_combinations(2)
        per_run_results = []
        noise_pcs = []
        for r in range(s["n_runs"]):
            noise_pcs.append(torch.randn(s["run_len"], 2))
            per_run_results.append(
                CombinatorialDenoiseRunResult(
                    run_idx=r,
                    optimal_combination=(),
                    optimal_cod=0.0,
                    all_cod=np.zeros(len(combos)),
                    all_var_explained=np.zeros(len(combos)),
                    all_combinations=combos,
                    explained_variance_ratios=np.array([0.3, 0.2]),
                    n_criteria_voxels=10,
                )
            )
        r2 = compute_optimized_xval_r2(
            data=s["data"],
            design=s["design"],
            run_starts=s["run_starts"],
            nuisance_per_run=s["nuisance_per_run"],
            noise_pcs_per_run=noise_pcs,
            per_run_results=per_run_results,
            device=CPU,
            verbose=False,
        )
        assert r2.shape == (s["n_voxels"],)


class TestComputeOptimizedXvalR2DenoiseStyle:
    def test_single_design(self, multi_run_setup):
        s = multi_run_setup
        per_run_results, noise_pcs = _make_run_results(s["n_runs"], s["run_len"])
        r2 = compute_optimized_xval_r2_3dDenoise_style(
            data=s["data"],
            design=s["design"],
            run_starts=s["run_starts"],
            nuisance_per_run=s["nuisance_per_run"],
            noise_pcs_per_run=noise_pcs,
            per_run_results=per_run_results,
            device=CPU,
            verbose=False,
        )
        assert r2.shape == (s["n_voxels"],)

    def test_per_hrf_mode(self, multi_run_setup):
        s = multi_run_setup
        torch.manual_seed(77)
        hrf_indices = torch.zeros(s["n_voxels"], dtype=torch.long)
        hrf_indices[s["n_voxels"] // 2 :] = 1
        designs_by_hrf = {
            0: torch.randn(s["n_tp"], s["n_conds"]),
            1: torch.randn(s["n_tp"], s["n_conds"]),
        }
        per_run_results, noise_pcs = _make_run_results(s["n_runs"], s["run_len"], max_pcs=2)
        r2 = compute_optimized_xval_r2_3dDenoise_style(
            data=s["data"],
            design=None,
            run_starts=s["run_starts"],
            nuisance_per_run=s["nuisance_per_run"],
            noise_pcs_per_run=noise_pcs,
            per_run_results=per_run_results,
            designs_by_hrf=designs_by_hrf,
            hrf_indices=hrf_indices,
            device=CPU,
            verbose=False,
        )
        assert r2.shape == (s["n_voxels"],)


class TestComputeInitialXvalR2:
    def test_single_design(self, multi_run_setup):
        s = multi_run_setup
        r2 = compute_initial_xval_r2(
            data=s["data"],
            design=s["design"],
            run_starts=s["run_starts"],
            nuisance_per_run=s["nuisance_per_run"],
            device=CPU,
            verbose=False,
        )
        assert r2.shape == (s["n_voxels"],)
        assert torch.isfinite(r2).all()

    def test_per_hrf_mode(self, multi_run_setup):
        s = multi_run_setup
        torch.manual_seed(55)
        hrf_indices = torch.zeros(s["n_voxels"], dtype=torch.long)
        hrf_indices[s["n_voxels"] // 2 :] = 1
        designs_by_hrf = {
            0: torch.randn(s["n_tp"], s["n_conds"]),
            1: torch.randn(s["n_tp"], s["n_conds"]),
        }
        r2 = compute_initial_xval_r2(
            data=s["data"],
            design=None,
            run_starts=s["run_starts"],
            nuisance_per_run=s["nuisance_per_run"],
            designs_by_hrf=designs_by_hrf,
            hrf_indices=hrf_indices,
            device=CPU,
            verbose=False,
        )
        assert r2.shape == (s["n_voxels"],)


class TestPlotSingletonContributions:
    def test_produces_figures(self, tmp_path):
        results = _make_full_results(3, 40, 50)
        figs = plot_singleton_contributions(results, str(tmp_path / "test"))
        assert len(figs) == 3
        import matplotlib.pyplot as plt

        for f in figs:
            plt.close(f)


class TestPlotPlateauCurves:
    def test_produces_figures(self, tmp_path):
        results = _make_full_results(3, 40, 50)
        figs = plot_plateau_curves(results, str(tmp_path / "test"))
        assert len(figs) == 3
        import matplotlib.pyplot as plt

        for f in figs:
            plt.close(f)

    def test_singleton_only_returns_empty(self, tmp_path):
        results = _make_full_results(3, 40, 50, singleton_only=True)
        figs = plot_plateau_curves(results, str(tmp_path / "test"))
        assert figs == []


class TestPlotInclusionHeatmap:
    def test_produces_single_figure(self, tmp_path):
        results = _make_full_results(3, 40, 50)
        figs = plot_inclusion_heatmap(results, str(tmp_path / "test"))
        assert len(figs) == 1
        import matplotlib.pyplot as plt

        plt.close(figs[0])


class TestPlotCombinatorialResults:
    def test_produces_figures(self, tmp_path):
        results = _make_full_results(3, 40, 50)
        figs = plot_combinatorial_results(results, str(tmp_path / "test"))
        assert len(figs) >= 1
        import matplotlib.pyplot as plt

        for f in figs:
            plt.close(f)

    def test_singleton_only_returns_empty(self, tmp_path):
        results = _make_full_results(3, 40, 50, singleton_only=True)
        figs = plot_combinatorial_results(results, str(tmp_path / "test"))
        assert figs == []
