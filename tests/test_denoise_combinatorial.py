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
