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
                baseline_cod=0.1,
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
                    baseline_cod=0.0,
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


# ===========================================================================
# Null-calibrated singleton selection (-null_surrogates)
# ===========================================================================

from fastfuncstuff.denoise.combinatorial import (  # noqa: E402
    fit_combinatorial_denoising,
    phase_randomize,
    select_singletons_against_null,
)


class TestPhaseRandomize:
    """Surrogates are only a valid null if they match the PC on everything
    except being real: same variance, same spectrum, same DoF cost."""

    def test_preserves_variance_and_autocorrelation(self):
        torch.manual_seed(0)
        x = torch.cumsum(torch.randn(200, 3), 0)  # strongly autocorrelated
        x = (x - x.mean(0)) * torch.tensor([1.0, 5.0, 0.5])
        gen = torch.Generator().manual_seed(0)
        surr = phase_randomize(x, 25, generator=gen)

        assert surr.shape == (200, 75)
        for i in range(3):
            block = surr[:, i * 25 : (i + 1) * 25]
            # Variance is exactly preserved by construction, not approximately.
            torch.testing.assert_close(
                block.var(dim=0), x[:, i].var().expand(25), rtol=1e-4, atol=1e-4
            )
            lag1 = torch.stack(
                [
                    torch.corrcoef(torch.stack([block[:-1, j], block[1:, j]]))[0, 1]
                    for j in range(25)
                ]
            )
            src_lag1 = torch.corrcoef(torch.stack([x[:-1, i], x[1:, i]]))[0, 1]
            assert abs(lag1.mean() - src_lag1) < 0.1

    def test_surrogates_are_real_valued(self):
        """Rotating the DC or Nyquist phase would make the inverse FFT complex."""
        for n_time in (200, 201):  # even and odd both hit the Nyquist branch
            x = torch.randn(n_time, 2)
            gen = torch.Generator().manual_seed(1)
            surr = phase_randomize(x, 4, generator=gen)
            assert not surr.is_complex()
            assert torch.isfinite(surr).all()

    def test_seed_reproduces_and_device_does_not_matter(self):
        x = torch.randn(128, 3)
        a = phase_randomize(x, 5, generator=torch.Generator().manual_seed(7))
        b = phase_randomize(x, 5, generator=torch.Generator().manual_seed(7))
        torch.testing.assert_close(a, b)
        if torch.cuda.is_available():
            c = phase_randomize(x.cuda(), 5, generator=torch.Generator().manual_seed(7))
            torch.testing.assert_close(a, c.cpu(), rtol=1e-4, atol=1e-4)


class TestSelectSingletonsAgainstNull:
    def test_three_states_are_assigned_correctly(self):
        k, n_sur = 3, 4
        # baseline 0.5; PC deltas: +0.10 (clears), +0.01 (positive, inside null), -0.02
        median_cod = np.array([0.5, 0.60, 0.51, 0.48])
        # every surrogate delta is +0.05, so the p95 threshold is ~0.05
        null_cod = np.concatenate([[0.5], np.full(k * n_sur, 0.55)])

        selected, thresholds, status = select_singletons_against_null(
            median_cod, null_cod, k, n_sur, percentile=95.0
        )
        assert selected == (0,)
        assert status == ("selected", "rejected_null", "not_selected")
        np.testing.assert_allclose(thresholds, 0.05, atol=1e-6)

    def test_each_pc_gets_its_own_threshold(self):
        """A high-variance PC removes more variance, so it must clear a higher bar."""
        k, n_sur = 2, 2
        median_cod = np.array([0.0, 0.05, 0.05])  # identical deltas
        # PC0's surrogates are quiet, PC1's are not
        null_cod = np.array([0.0, 0.01, 0.01, 0.09, 0.09])
        selected, thresholds, status = select_singletons_against_null(
            median_cod, null_cod, k, n_sur, percentile=95.0
        )
        assert selected == (0,)
        assert status == ("selected", "rejected_null")
        assert thresholds[1] > thresholds[0]


class TestNullCalibrationRejectsFalsePositives:
    """The reason the null exists.

    CoD is 1 - SS_res/SS_tot with SS_tot recomputed from the cleaned data, and
    d/dv[(R-v)/(T-v)] < 0 for R < T, so removing residual variance raises CoD
    whether or not the removed direction was real. A bare `delta > 0` therefore
    has no noise floor and admits unrelated regressors about half the time.
    """

    @staticmethod
    def _one_trial(seed, inject_real_structure, device):
        n_time, n_vox, k, n_sur = 160, 300, 5, 20
        torch.manual_seed(seed)
        design = torch.zeros(n_time, 1)
        for onset in range(10, n_time - 10, 25):
            design[onset : onset + 5, 0] = 1.0
        poly = torch.stack([torch.ones(n_time), torch.linspace(-1, 1, n_time)], 1)

        amp = torch.rand(n_vox, 1) * 2 + 1
        data = amp @ design.T + torch.randn(n_vox, n_time) * 1.5
        pcs = torch.randn(n_time, k)
        if inject_real_structure:
            # PC0 becomes genuine shared noise: present in every voxel.
            data = data + (torch.randn(n_vox, 1) * 6.0) @ pcs[:, :1].T
        betas = amp + torch.randn(n_vox, 1) * 0.05

        def score(columns):
            return evaluate_all_combinations_for_run(
                run_data_criteria=data,
                run_design=design,
                betas_criteria=betas,
                poly_nuisance=poly,
                noise_pcs=columns,
                combinations=[()] + [(i,) for i in range(columns.shape[1])],
                variance_ratios=np.zeros(columns.shape[1]),
                device=device,
                verbose=False,
            )[0]

        median_cod = score(pcs)
        gen = torch.Generator().manual_seed(seed)
        null_cod = score(phase_randomize(pcs, n_sur, generator=gen))
        selected, _, _ = select_singletons_against_null(median_cod, null_cod, k, n_sur, 95.0)
        naive = tuple(i for i in range(k) if median_cod[1 + i] - median_cod[0] > 0)
        return naive, selected

    def test_unrelated_pcs_are_mostly_rejected(self):
        """PCs with no relationship to the data: the bare rule keeps ~half."""
        n_trials = 6
        naive_total = null_total = 0
        for seed in range(n_trials):
            naive, selected = self._one_trial(seed, False, CPU)
            naive_total += len(naive)
            null_total += len(selected)

        naive_rate = naive_total / n_trials
        null_rate = null_total / n_trials
        assert naive_rate > 1.5, f"expected the bare rule to over-select, got {naive_rate:.2f}/5"
        assert null_rate < naive_rate / 2, (
            f"null should cut false positives sharply: {naive_rate:.2f} -> {null_rate:.2f}"
        )

    def test_genuine_shared_noise_still_survives(self):
        """Rejecting noise is worthless if it also rejects the real component."""
        for seed in range(4):
            _, selected = self._one_trial(seed, True, CPU)
            assert 0 in selected, f"seed {seed}: real shared-noise PC0 was rejected"


class TestNullCalibrationWiring:
    def test_singleton_only_is_required(self, multi_run_setup):
        s = multi_run_setup
        with pytest.raises(ValueError, match="singleton-mode rule"):
            fit_combinatorial_denoising(
                data=s["data"],
                design=s["design"],
                run_starts=s["run_starts"],
                tr=2.0,
                nuisance_per_run=s["nuisance_per_run"],
                noise_pool_mask=torch.ones(s["n_voxels"], dtype=torch.bool),
                initial_r2=torch.rand(s["n_voxels"]),
                max_pcs=2,
                singleton_only=False,
                n_null_surrogates=5,
                device=CPU,
                verbose=False,
            )

    def test_status_and_thresholds_reach_the_results(self, multi_run_setup):
        s = multi_run_setup
        results = fit_combinatorial_denoising(
            data=s["data"],
            design=s["design"],
            run_starts=s["run_starts"],
            tr=2.0,
            nuisance_per_run=s["nuisance_per_run"],
            noise_pool_mask=torch.ones(s["n_voxels"], dtype=torch.bool),
            initial_r2=torch.rand(s["n_voxels"]),
            max_pcs=2,
            criteria_r2_threshold="50%",
            singleton_only=True,
            n_null_surrogates=4,
            device=CPU,
            verbose=False,
        )
        for run_res in results.per_run_results:
            assert run_res.pc_status is not None
            assert len(run_res.pc_status) == 2
            assert set(run_res.pc_status) <= {"selected", "rejected_null", "not_selected"}
            assert run_res.null_thresholds is not None
            assert run_res.null_thresholds.shape == (2,)
            # The selection and the status must agree, or the plots lie.
            from_status = tuple(i for i, st in enumerate(run_res.pc_status) if st == "selected")
            assert from_status == run_res.optimal_combination
        assert results.metadata["n_null_surrogates"] == 4

    def test_plots_render_with_the_middle_state(self, tmp_path):
        """The discarded-after-initial-selection state has to reach the figures."""
        results = _make_full_results(3, 40, 50, max_pcs=3, singleton_only=True)
        for run_res in results.per_run_results:
            run_res.pc_status = ("selected", "rejected_null", "not_selected")
            run_res.null_thresholds = np.array([0.01, 0.02, 0.03])
            run_res.optimal_combination = (0,)

        import matplotlib.pyplot as plt

        figs = plot_singleton_contributions(results, str(tmp_path / "t"))
        assert len(figs) == 3
        for f in figs:
            plt.close(f)
        heat = plot_inclusion_heatmap(results, str(tmp_path / "t"))
        assert (tmp_path / "t_combinatorial_heatmap.png").exists()
        # "o" for the rejected PC must appear in the heatmap title/marks.
        assert "rejected by null" in heat[0].axes[0].get_title()
        for f in heat:
            plt.close(f)


# ===========================================================================
# Cross-run selection criterion (-criterion cross_run)
# ===========================================================================

from fastfuncstuff.denoise.combinatorial import (  # noqa: E402
    evaluate_combinations_cross_run,
)


def _cross_run_setup(seed, n_runs=6, run_len=120, n_voxels=200, k=4):
    torch.manual_seed(seed)
    n_tp = n_runs * run_len
    run_starts = [r * run_len for r in range(n_runs)]
    design = torch.zeros(n_tp, 1)
    for r in range(n_runs):
        for onset in range(10, run_len - 10, 25):
            design[r * run_len + onset : r * run_len + onset + 5, 0] = 1.0
    nuisance = [
        torch.stack([torch.ones(run_len), torch.linspace(-1, 1, run_len)], 1) for _ in range(n_runs)
    ]
    amp = torch.rand(n_voxels, 1) * 2 + 1
    data = amp @ design.T + torch.randn(n_voxels, n_tp) * 1.5
    pcs = torch.randn(run_len, k)
    combos = [()] + [(i,) for i in range(k)]
    return dict(
        data=data,
        design=design,
        run_starts=run_starts,
        n_tp=n_tp,
        nuisance=nuisance,
        pcs=pcs,
        combos=combos,
        run_len=run_len,
        k=k,
    )


class TestCrossRunCriterion:
    def test_no_positive_bias_for_unrelated_pcs(self):
        """The whole reason the criterion exists.

        within_run recomputes SS_tot from the cleaned run, so removing residual
        variance lifts CoD regardless of merit. cross_run scores runs that are
        never cleaned, so SS_tot cannot move with the candidate and an unrelated
        PC has nothing to gain — if anything it should lose, since it still
        costs a degree of freedom in the beta fit.

        The absolute mean is the robust statistic here. A sign-fraction is far
        noisier per trial, and the size of within_run's bias scales with the
        baseline CoD, so a cross-criterion ratio would be data-dependent — see
        test_within_run_inflates_when_baseline_cod_is_high for that half.
        """
        deltas = []
        for seed in range(5):
            s = _cross_run_setup(seed)
            cod, _ = evaluate_combinations_cross_run(
                data_criteria=s["data"],
                run_starts=s["run_starts"],
                n_timepoints=s["n_tp"],
                nuisance_per_run=s["nuisance"],
                target_run=0,
                pcs=s["pcs"],
                combinations=s["combos"],
                design=s["design"],
                designs_by_hrf=None,
                criteria_hrf_indices=None,
                device=CPU,
            )
            deltas.append(cod[1:] - cod[0])
        mean_delta = float(np.mean(deltas))
        assert abs(mean_delta) < 2e-4, f"expected ~0 mean delta, got {mean_delta:+.6f}"

    def test_within_run_inflates_when_baseline_cod_is_high(self):
        """The other half: the bias cross_run exists to remove.

        CoD = 1 - SS_res/SS_tot with SS_tot re-derived from the cleaned data,
        and d/dv[(R-v)/(T-v)] < 0 for R < T, so the inflation grows as the
        baseline fit improves. Accurate betas and a strong response put it
        clear of the noise; at low baseline CoD it shrinks toward zero, which
        is why the paired ratio is not the thing to assert.
        """
        n_time, n_vox, k = 200, 400, 5
        design = torch.zeros(n_time, 1)
        for onset in range(10, n_time - 10, 25):
            design[onset : onset + 5, 0] = 1.0
        poly = torch.stack([torch.ones(n_time), torch.linspace(-1, 1, n_time)], 1)

        deltas = []
        for seed in range(5):
            torch.manual_seed(seed)
            amp = torch.rand(n_vox, 1) * 2 + 1
            data = amp @ design.T + torch.randn(n_vox, n_time) * 0.5
            betas = amp + torch.randn(n_vox, 1) * 0.05  # near-true => high CoD
            cod, _ = evaluate_all_combinations_for_run(
                run_data_criteria=data,
                run_design=design,
                betas_criteria=betas,
                poly_nuisance=poly,
                noise_pcs=torch.randn(n_time, k),  # unrelated to the data
                combinations=[()] + [(i,) for i in range(k)],
                variance_ratios=np.zeros(k),
                device=CPU,
                verbose=False,
            )
            assert cod[0] > 0.5, "this regime is meant to have a high baseline CoD"
            deltas.append(cod[1:] - cod[0])

        assert float(np.mean(deltas)) > 0, (
            "unrelated regressors should still gain CoD under within_run — if "
            "this stops holding, the mechanical inflation is gone and the "
            "cross_run criterion's main justification needs revisiting"
        )

    def test_ss_tot_is_constant_across_candidates(self):
        """A direct check of the mechanism: identical CoD when betas cannot differ.

        With a single-column design and a PC that is exactly zero, every
        candidate produces the same fit, so any CoD spread would have to come
        from the denominator moving.
        """
        s = _cross_run_setup(11)
        s["pcs"] = torch.zeros_like(s["pcs"])
        cod, _ = evaluate_combinations_cross_run(
            data_criteria=s["data"],
            run_starts=s["run_starts"],
            n_timepoints=s["n_tp"],
            nuisance_per_run=s["nuisance"],
            target_run=0,
            pcs=s["pcs"],
            combinations=s["combos"],
            design=s["design"],
            designs_by_hrf=None,
            criteria_hrf_indices=None,
            device=CPU,
        )
        np.testing.assert_allclose(cod, cod[0], atol=1e-9)

    def test_genuine_shared_noise_is_ranked_first(self):
        """Sensitivity is lower than within_run because the effect it measures is
        genuinely smaller -- one run's artifact only perturbs betas that N runs
        contribute to -- but the real component should still lead the ranking."""
        hits = 0
        trials = 6
        for seed in range(trials):
            s = _cross_run_setup(100 + seed, n_voxels=300)
            rl = s["run_len"]
            s["data"][:, :rl] += (torch.randn(s["data"].shape[0], 1) * 8.0) @ s["pcs"][:, :1].T
            cod, _ = evaluate_combinations_cross_run(
                data_criteria=s["data"],
                run_starts=s["run_starts"],
                n_timepoints=s["n_tp"],
                nuisance_per_run=s["nuisance"],
                target_run=0,
                pcs=s["pcs"],
                combinations=s["combos"],
                design=s["design"],
                designs_by_hrf=None,
                criteria_hrf_indices=None,
                device=CPU,
            )
            delta = cod[1:] - cod[0]
            hits += int(delta.argmax() == 0)
        assert hits >= trials - 2, f"real PC0 led the ranking only {hits}/{trials} times"

    def test_needs_three_runs(self, multi_run_setup):
        s = _cross_run_setup(0, n_runs=1)
        with pytest.raises(ValueError, match="at least 2 runs"):
            evaluate_combinations_cross_run(
                data_criteria=s["data"],
                run_starts=s["run_starts"],
                n_timepoints=s["n_tp"],
                nuisance_per_run=s["nuisance"],
                target_run=0,
                pcs=s["pcs"],
                combinations=s["combos"],
                design=s["design"],
                designs_by_hrf=None,
                criteria_hrf_indices=None,
                device=CPU,
            )

    def test_rejects_bad_criterion_and_too_few_runs(self, multi_run_setup):
        s = multi_run_setup
        common = dict(
            data=s["data"],
            design=s["design"],
            run_starts=s["run_starts"],
            tr=2.0,
            nuisance_per_run=s["nuisance_per_run"],
            noise_pool_mask=torch.ones(s["n_voxels"], dtype=torch.bool),
            initial_r2=torch.rand(s["n_voxels"]),
            max_pcs=2,
            device=CPU,
            verbose=False,
        )
        with pytest.raises(ValueError, match="criterion must be"):
            fit_combinatorial_denoising(criterion="nonsense", **common)
        with pytest.raises(ValueError, match="at least 3 runs"):
            fit_combinatorial_denoising(criterion="cross_run", **{**common, "run_starts": [0, 20]})

    def test_end_to_end_through_fit(self, multi_run_setup):
        s = multi_run_setup
        results = fit_combinatorial_denoising(
            data=s["data"],
            design=s["design"],
            run_starts=s["run_starts"],
            tr=2.0,
            nuisance_per_run=s["nuisance_per_run"],
            noise_pool_mask=torch.ones(s["n_voxels"], dtype=torch.bool),
            initial_r2=torch.rand(s["n_voxels"]),
            max_pcs=2,
            criteria_r2_threshold="50%",
            singleton_only=True,
            criterion="cross_run",
            n_null_surrogates=3,
            device=CPU,
            verbose=False,
        )
        assert results.metadata["criterion"] == "cross_run"
        assert len(results.per_run_results) == 3
        for run_res in results.per_run_results:
            assert run_res.pc_status is not None
            assert set(run_res.optimal_combination) <= {0, 1}


class TestArmSpecificPcs:
    def test_learning_curve_honours_per_arm_pcs(self):
        """A GLMdenoise reference must keep its own noise-pool PCs even when the
        arm under test used -whole_brain_noise_pool, or the comparison is rigged."""
        from fastfuncstuff.denoise.heldout import heldout_learning_curve

        torch.manual_seed(3)
        n_train, run_len, n_vox, k = 4, 60, 20, 3
        design = torch.randn(n_train * run_len, 1)
        test_design = torch.randn(2 * run_len, 1)
        data = torch.randn(n_vox, n_train * run_len)
        test_data = torch.randn(n_vox, 2 * run_len)
        polys = lambda n: [  # noqa: E731
            torch.stack([torch.ones(run_len), torch.linspace(-1, 1, run_len)], 1) for _ in range(n)
        ]
        pcs_a = [torch.randn(run_len, k) for _ in range(n_train)]
        pcs_b = [torch.randn(run_len, k) for _ in range(n_train)]

        curve = heldout_learning_curve(
            train_data=data,
            train_run_starts=[r * run_len for r in range(n_train)],
            train_nuisance_per_run=polys(n_train),
            train_pcs_per_run=pcs_a,
            arms={"shared": [(0,)] * n_train, "own_pcs": [(0,)] * n_train},
            arm_pcs_per_run={"own_pcs": pcs_b},
            test_data=test_data,
            test_run_starts=[0, run_len],
            test_nuisance_per_run=polys(2),
            train_design=design,
            test_design=test_design,
            subset_sizes=[n_train],
            max_subsets=1,
            device=CPU,
            verbose=False,
        )
        # Same selection, different PCs -> different curves. Equality would mean
        # the override was silently ignored.
        assert not torch.allclose(curve["curves"]["shared"], curve["curves"]["own_pcs"], atol=1e-6)


class TestWholeBrainPcSource:
    def test_whole_brain_pc0_is_the_task_response(self):
        """Why -compare must never inherit whole-brain PCs.

        Task variance is large and spatially coherent, so with the noise-pool
        restriction lifted it dominates the top of the variance ordering. A
        GLMdenoise baseline handed these PCs would remove the task as its
        first component — a straw man. It also means -max_pcs has to grow
        alongside -whole_brain_noise_pool to reach real artifacts.
        """
        torch.manual_seed(0)
        n_time, n_vox = 150, 400
        task = torch.zeros(n_time)
        task[::25] = 1.0
        data = torch.randn(n_vox, n_time)
        data[:100] += 8.0 * task  # first quarter of voxels are task-responsive

        noise_pool = torch.zeros(n_vox, dtype=torch.bool)
        noise_pool[100:] = True
        whole_brain = torch.ones(n_vox, dtype=torch.bool)
        nuisance = torch.stack([torch.ones(n_time), torch.linspace(-1, 1, n_time)], 1)

        pool_pcs, _ = extract_pcs_single_run_with_variance(data, noise_pool, nuisance, 3, CPU)
        brain_pcs, _ = extract_pcs_single_run_with_variance(data, whole_brain, nuisance, 3, CPU)

        def abs_corr(x, y):
            return abs(float(torch.corrcoef(torch.stack([x, y]))[0, 1]))

        assert abs_corr(brain_pcs[:, 0], task) > 0.9, "whole-brain PC0 should track the task"
        assert abs_corr(pool_pcs[:, 0], task) < 0.5, "noise-pool PC0 should not"
        assert abs_corr(pool_pcs[:, 0], brain_pcs[:, 0]) < 0.5, (
            "the two PC sources must differ, or guarding the baseline is pointless"
        )


# ---------------------------------------------------------------------------
# Q-based combination evaluation
# ---------------------------------------------------------------------------
# evaluate_all_combinations_for_run used to build an explicit (T, T) projector
# per PC combination. It now factors the polynomial part out once and expresses
# every combination-dependent term as a (k x V) projection.
# FFS_COMBO_LEGACY=1 restores the explicit-projector version.


def _combo_eval_case(T=60, k=4, V=50, n_cond=3, n_poly=4, zero_poly=False):
    import numpy as np

    torch.manual_seed(3)
    design = torch.zeros(T, n_cond)
    for c in range(n_cond):
        design[(c + 1) :: (5 + c), c] = 1.0
    if n_poly > 0:
        poly = torch.stack([torch.linspace(-1, 1, T) ** d for d in range(n_poly)], dim=1)
    else:
        poly = torch.zeros(T, 0)
    if zero_poly:
        poly = torch.zeros_like(poly)  # all-zero columns are stripped as degenerate
    return dict(
        run_data_criteria=torch.randn(V, T),
        run_design=design,
        betas_criteria=torch.randn(V, n_cond),
        poly_nuisance=poly,
        noise_pcs=torch.randn(T, k),
        variance_ratios=np.linspace(0.1, 0.01, k),
    )


@pytest.mark.parametrize(
    "k,n_cond,n_poly,zero_poly,chunk",
    [
        (4, 3, 4, False, None),
        (6, 3, 4, False, None),  # 64 combinations
        (4, 3, 0, False, None),  # no polynomial nuisance at all
        (4, 1, 4, False, None),  # single condition
        (4, 3, 4, True, None),  # polynomial columns all zero
        (4, 3, 4, False, 7),  # multiple voxel chunks
    ],
)
def test_evaluate_all_combinations_q_based_matches_legacy(
    monkeypatch, k, n_cond, n_poly, zero_poly, chunk
):
    """Q-based projection must reproduce the explicit (T, T) projector result."""
    from fastfuncstuff.denoise.combinatorial import (
        evaluate_all_combinations_for_run,
        generate_all_pc_combinations,
    )

    case = _combo_eval_case(k=k, n_cond=n_cond, n_poly=n_poly, zero_poly=zero_poly)
    call = dict(
        **case,
        combinations=generate_all_pc_combinations(k),
        device=torch.device("cpu"),
        criteria_chunk_size=chunk,
        verbose=False,
        return_raw_cod=True,
    )

    monkeypatch.delenv("FFS_COMBO_LEGACY", raising=False)
    fast_cod, fast_var = evaluate_all_combinations_for_run(**call)

    monkeypatch.setenv("FFS_COMBO_LEGACY", "1")
    legacy_cod, legacy_var = evaluate_all_combinations_for_run(**call)

    assert np.abs(fast_cod - legacy_cod).max() < 1e-4
    assert np.allclose(fast_var, legacy_var)


class TestSingletonReportedCoD:
    """The reported CoD must be the selected set's, not some other combination's.

    Bug of record: singleton mode pinned `best_idx = 0` and then reported
    `median_cod[best_idx]`, which is combination 0 -- the EMPTY set. Every run's
    summary line showed the no-PC baseline while appearing to show what the
    selection achieved, so six runs all reported ~0.114 whatever they picked, and
    a run whose global improvement was -0.0002 looked like it had gained 0.11.
    """

    def _fit(self, **kwargs):
        import torch

        from fastfuncstuff.denoise.combinatorial import fit_combinatorial_denoising

        torch.manual_seed(0)
        n_runs, run_len, n_vox = 3, 60, 120
        run_starts = [r * run_len for r in range(n_runs)]
        total = n_runs * run_len

        design = torch.zeros(total, 1)
        for r in range(n_runs):
            design[r * run_len : r * run_len + 30, 0] = 1.0
        design -= design.mean(dim=0, keepdim=True)

        data = torch.randn(n_vox, total) + design[:, 0].unsqueeze(0) * 2.0
        nuisance = [torch.ones(run_len, 1) for _ in range(n_runs)]
        noise_pool = torch.zeros(n_vox, dtype=torch.bool)
        noise_pool[n_vox // 2 :] = True
        initial_r2 = torch.full((n_vox,), 0.1)

        return fit_combinatorial_denoising(
            data=data,
            design=design,
            run_starts=run_starts,
            tr=2.0,
            nuisance_per_run=nuisance,
            noise_pool_mask=noise_pool,
            initial_r2=initial_r2,
            max_pcs=3,
            criteria_r2_threshold=0.0,
            device=torch.device("cpu"),
            verbose=False,
            **kwargs,
        )

    def test_singleton_cod_is_not_the_baseline(self):
        results = self._fit(singleton_only=True)

        for run in results.per_run_results:
            assert run.baseline_cod == run.all_cod[0], "combination 0 is the empty set"
            if run.optimal_combination:
                # The old code made this equality hold for every run by construction.
                assert run.optimal_cod != run.baseline_cod

    def test_empty_selection_reports_the_baseline(self):
        """With nothing selected there is no set to score, so baseline IS the answer."""
        results = self._fit(singleton_only=True)

        for run in results.per_run_results:
            if not run.optimal_combination:
                assert run.optimal_cod == run.baseline_cod

    def test_combination_mode_still_reports_its_winner(self):
        results = self._fit(singleton_only=False)

        for run in results.per_run_results:
            idx = run.all_combinations.index(tuple(run.optimal_combination))
            assert run.optimal_cod == run.all_cod[idx]
            assert run.baseline_cod == run.all_cod[0]


class TestCrossRunBatchedParity:
    """The batched scorer must equal a first-principles fit, not just its own history.

    evaluate_combinations_cross_run builds every candidate's moments as a rank-m
    downdate of the base moments and scores residuals in condition space
    (|y|^2 - 2b'X'y + b'X'Xb) rather than forming predictions. Both are exact
    rearrangements, so a plain lstsq reference is the right thing to check
    against -- and the one that would catch an algebra slip that a
    regression-style golden test would happily enshrine.
    """

    @staticmethod
    def _reference(case, combinations):
        """Project, fit and score every (candidate, held-out run) pair explicitly."""
        import torch

        from fastfuncstuff.glm.xval import project_out_nuisance_per_run

        data, starts = case["data_criteria"], case["run_starts"]
        n_tp, design = case["n_timepoints"], case["design"]
        nuisance, target = case["nuisance_per_run"], case["target_run"]
        n_runs = len(starts)
        other = [h for h in range(n_runs) if h != target]
        ends = [*starts[1:], n_tp]
        ss_res = torch.zeros(len(combinations), data.shape[0], dtype=torch.float64)
        ss_tot = torch.zeros(data.shape[0], dtype=torch.float64)

        def proj(run_idx, extra=None):
            nus = nuisance[run_idx]
            if extra is not None:
                nus = torch.cat([nus, extra], dim=1)
            s, e = starts[run_idx], ends[run_idx]
            d, x = project_out_nuisance_per_run(
                data=data[:, s:e],
                design=design[s:e, :],
                nuisance_per_run=[nus],
                run_starts=[0],
                device=case["device"],
            )
            return d.double(), x.double()

        base = {r: proj(r) for r in range(n_runs)}
        for h in other:
            centred = base[h][0] - base[h][0].mean(dim=1, keepdim=True)
            ss_tot += (centred * centred).sum(dim=1)

        for ci, combo in enumerate(combinations):
            y_t, x_t = proj(target, case["pcs"][:, list(combo)] if combo else None)
            for h in other:
                fit = [r for r in other if r != h]
                xtx = sum(base[r][1].T @ base[r][1] for r in fit) + x_t.T @ x_t
                xty = sum(base[r][1].T @ base[r][0].T for r in fit) + x_t.T @ y_t.T
                xtx = xtx + 1e-6 * torch.eye(xtx.shape[0], dtype=torch.float64)
                resid = base[h][0].T - base[h][1] @ torch.linalg.solve(xtx, xty)
                ss_res[ci] += (resid * resid).sum(dim=0)

        cod = 1.0 - ss_res / ss_tot.clamp(min=1e-10).unsqueeze(0)
        return cod.median(dim=1).values.numpy()

    @staticmethod
    def _case(n_criteria=250, run_len=50, n_runs=4, n_cond=2, k=3, seed=0):
        import torch

        g = torch.Generator().manual_seed(seed)
        n_tp = run_len * n_runs
        return dict(
            data_criteria=torch.randn(n_criteria, n_tp, generator=g),
            run_starts=[r * run_len for r in range(n_runs)],
            n_timepoints=n_tp,
            nuisance_per_run=[torch.randn(run_len, 4, generator=g) for _ in range(n_runs)],
            target_run=0,
            pcs=torch.randn(run_len, k, generator=g),
            design=torch.randn(n_tp, n_cond, generator=g),
            designs_by_hrf=None,
            criteria_hrf_indices=None,
            device=torch.device("cpu"),
        )

    def test_singletons_match_explicit_fits(self):
        import numpy as np

        from fastfuncstuff.denoise.combinatorial import evaluate_combinations_cross_run

        case = self._case()
        combos = [(), (0,), (1,), (2,)]
        got = evaluate_combinations_cross_run(combinations=combos, **case)[0]
        assert np.abs(got - self._reference(case, combos)).max() < 1e-8

    def test_multi_column_candidates_match_explicit_fits(self):
        """The rank-m path: a QR basis, not the batched normalisation."""
        import numpy as np

        from fastfuncstuff.denoise.combinatorial import evaluate_combinations_cross_run

        case = self._case(n_cond=3, seed=1)
        combos = [(), (0,), (1, 2), (0, 1, 2)]
        got = evaluate_combinations_cross_run(combinations=combos, **case)[0]
        assert np.abs(got - self._reference(case, combos)).max() < 1e-8

    def test_candidate_inside_the_nuisance_span_is_a_no_op(self):
        """A PC the nuisance already contains adds nothing and must not divide by ~0."""
        import numpy as np

        from fastfuncstuff.denoise.combinatorial import evaluate_combinations_cross_run

        case = self._case(seed=2)
        # Make PC 0 an exact copy of a nuisance column for the target run.
        case["pcs"][:, 0] = case["nuisance_per_run"][case["target_run"]][:, 1]
        got = evaluate_combinations_cross_run(combinations=[(), (0,)], **case)[0]
        assert np.all(np.isfinite(got))
        assert abs(got[1] - got[0]) < 1e-9

    def test_voxel_chunking_does_not_change_the_answer(self):
        import numpy as np

        from fastfuncstuff.denoise import combinatorial as C

        case = self._case(n_criteria=900, seed=3)
        combos = [(), (0,), (1,), (2,)]
        whole = C.evaluate_combinations_cross_run(combinations=combos, **case)[0]

        # The scorer imports estimate_chunk_size inside the function, so the patch
        # has to land on the source module, not on combinatorial's namespace.
        from fastfuncstuff import memory

        real = memory.estimate_chunk_size
        calls = []
        try:
            memory.estimate_chunk_size = lambda **kw: (calls.append(kw), 100)[1]
            chunked = C.evaluate_combinations_cross_run(combinations=combos, **case)[0]
        finally:
            memory.estimate_chunk_size = real

        assert calls, "the scorer must size its chunks through memory.py, not a constant"
        assert np.abs(whole - chunked).max() < 1e-10
