"""Noise-ceiling estimators must recover a planted var(signal)/var(y).

Every test here builds data whose true ceiling is known by construction, so a
regression shows up as a biased ceiling rather than as a plausible-looking map.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.simulation.noise import generate_ar1_noise
from fastfuncstuff.stats.noise_ceiling import (
    CeilingResult,
    df_corrected_ceiling,
    loro_ceiling_by_voxel_group,
    loro_two_half_ceiling,
    mean_train_repeats,
    merge_voxel_group_ceilings,
    ncsnr,
    ncsnr_noise_ceiling,
    zscore_betas_by_run,
)


def _condition_design(n_runs: int, run_length: int, n_conditions: int, seed: int = 0):
    """Block design with every condition present in every run."""
    generator = np.random.default_rng(seed)
    n_timepoints = n_runs * run_length
    design = np.zeros((n_timepoints, n_conditions), dtype=np.float32)
    run_starts = [run * run_length for run in range(n_runs)]
    for run in range(n_runs):
        order = generator.permutation(n_conditions)
        for slot, condition in enumerate(order):
            onset = run * run_length + slot * (run_length // n_conditions)
            design[onset : onset + 6, condition] = 1.0
    return design, run_starts, n_timepoints


class TestLoroCeilingByVoxelGroup:
    """The shared entry point ffs_denoise and ffs_denoisatorial both call."""

    @staticmethod
    def _case(seed=7, n_runs=6, run_length=40, n_voxels=60):
        design, run_starts, n_timepoints = _condition_design(n_runs, run_length, 3, seed=seed)
        generator = np.random.default_rng(seed)
        design_t = torch.from_numpy(design)
        betas = torch.from_numpy(generator.normal(size=(n_voxels, 3)).astype(np.float32))
        data = betas @ design_t.T
        data = data + torch.from_numpy(
            generator.normal(scale=0.5, size=data.shape).astype(np.float32)
        )
        nuisance = [torch.ones(run_length, 1) for _ in range(n_runs)]
        splits = [([r for r in range(n_runs) if r != t], [t]) for t in range(n_runs)]
        return data, design_t, run_starts, nuisance, splits

    def test_single_design_matches_calling_loro_directly(self):
        """The wrapper must not change the answer, only the plumbing."""
        from fastfuncstuff.glm.xval import project_out_nuisance_per_run

        data, design, run_starts, nuisance, splits = self._case()

        wrapped = loro_ceiling_by_voxel_group(
            data=data,
            nuisance_per_run=nuisance,
            run_starts=run_starts,
            cv_splits=splits,
            design_matrix=design,
            device=torch.device("cpu"),
        )
        projected_data, projected_design = project_out_nuisance_per_run(
            data=data,
            design=design,
            nuisance_per_run=nuisance,
            run_starts=run_starts,
            device=torch.device("cpu"),
        )
        direct = loro_two_half_ceiling(
            data=projected_data,
            design_matrix=projected_design,
            run_starts=run_starts,
            stim_indices=list(range(projected_design.shape[1])),
            nuisance_indices=[],
            cv_splits=splits,
            device=torch.device("cpu"),
            verbose=False,
        )
        assert torch.allclose(wrapped.ceiling, direct.ceiling, equal_nan=True, atol=1e-5)

    def test_identical_per_hrf_designs_reproduce_the_single_design_answer(self):
        """Splitting voxels into groups that share a design must change nothing.

        The groups partition the voxels and var(y) is design-free, so a two-group
        run with the same design in both is the control that proves the merge
        does not distort the scale.
        """
        data, design, run_starts, nuisance, splits = self._case()
        hrf_indices = torch.zeros(data.shape[0], dtype=torch.long)
        hrf_indices[data.shape[0] // 2 :] = 1

        single = loro_ceiling_by_voxel_group(
            data=data,
            nuisance_per_run=nuisance,
            run_starts=run_starts,
            cv_splits=splits,
            design_matrix=design,
            device=torch.device("cpu"),
        )
        grouped = loro_ceiling_by_voxel_group(
            data=data,
            nuisance_per_run=nuisance,
            run_starts=run_starts,
            cv_splits=splits,
            designs_by_hrf={0: design, 1: design.clone()},
            hrf_indices=hrf_indices,
            device=torch.device("cpu"),
        )
        assert torch.allclose(single.ceiling, grouped.ceiling, equal_nan=True, atol=1e-5)

    def test_a_pc_block_in_the_nuisance_changes_the_ceiling(self):
        """The PC block must reach the ceiling's denominator, not be ignored.

        The optimized R2 is scored on data with the selected PCs removed. If the
        ceiling were built without them it would bound a different quantity, and
        the ratio would be quietly wrong rather than obviously broken.
        """
        data, design, run_starts, nuisance, splits = self._case()
        generator = np.random.default_rng(99)
        pcs = [
            torch.from_numpy(generator.normal(size=(40, 2)).astype(np.float32)) for _ in run_starts
        ]
        for run, start in enumerate(run_starts):
            loadings = torch.from_numpy(
                generator.normal(size=(data.shape[0], 2)).astype(np.float32)
            )
            data[:, start : start + 40] += loadings @ pcs[run].T

        without = loro_ceiling_by_voxel_group(
            data=data,
            nuisance_per_run=nuisance,
            run_starts=run_starts,
            cv_splits=splits,
            design_matrix=design,
            device=torch.device("cpu"),
        )
        with_pcs = loro_ceiling_by_voxel_group(
            data=data,
            nuisance_per_run=[torch.cat([n, p], dim=1) for n, p in zip(nuisance, pcs, strict=True)],
            run_starts=run_starts,
            cv_splits=splits,
            design_matrix=design,
            device=torch.device("cpu"),
        )
        assert not torch.allclose(without.ceiling, with_pcs.ceiling, equal_nan=True, atol=1e-3)


class TestMergeVoxelGroupCeilings:
    """Per-voxel-HRF modes estimate a ceiling per group; merging must not lose facts."""

    def test_maps_scatter_into_their_own_voxels(self):
        groups = [
            (
                torch.tensor([True, False, True, False]),
                CeilingResult(torch.tensor([0.1, 0.3]), "loro_two_half", 4),
            ),
            (
                torch.tensor([False, True, False, True]),
                CeilingResult(torch.tensor([0.2, 0.4]), "loro_two_half", 4),
            ),
        ]
        merged = merge_voxel_group_ceilings(groups, n_voxels=4)
        assert torch.allclose(merged.ceiling, torch.tensor([0.1, 0.2, 0.3, 0.4]))
        assert merged.n_usable == 4
        assert merged.method == "loro_two_half"

    def test_fold_count_is_the_minimum_not_the_last_group(self):
        """A group that lost folds must not be papered over by one that did not.

        n_usable is a property of the design, so a group whose conditions vanish
        from a training half loses folds the others kept. Reporting any single
        group's count -- the last one, as a naive loop would -- would claim a
        guarantee that does not hold brain-wide.
        """
        groups = [
            (torch.tensor([True, False]), CeilingResult(torch.tensor([0.1]), "loro_two_half", 2)),
            (torch.tensor([False, True]), CeilingResult(torch.tensor([0.2]), "loro_two_half", 5)),
        ]
        merged = merge_voxel_group_ceilings(groups, n_voxels=2)
        assert merged.n_usable == 2
        assert any("varied across voxel groups" in note for note in merged.notes)

    def test_notes_union_without_duplicates(self):
        shared = "3 of 9 folds could not be split"
        groups = [
            (
                torch.tensor([True, False]),
                CeilingResult(torch.tensor([0.1]), "loro_two_half", 4, [shared]),
            ),
            (
                torch.tensor([False, True]),
                CeilingResult(torch.tensor([0.2]), "loro_two_half", 4, [shared, "other"]),
            ),
        ]
        merged = merge_voxel_group_ceilings(groups, n_voxels=2)
        assert merged.notes == [shared, "other"]

    def test_voxels_no_group_claims_stay_nan(self):
        groups = [
            (
                torch.tensor([True, False, False]),
                CeilingResult(torch.tensor([0.1]), "loro_two_half", 4),
            )
        ]
        merged = merge_voxel_group_ceilings(groups, n_voxels=3)
        assert torch.isnan(merged.ceiling[1:]).all()


class TestLoroTwoHalfCeiling:
    def test_recovers_planted_ceiling(self):
        """cov of two independent halves must land on the true signal fraction."""
        n_runs, run_length, n_conditions = 8, 240, 6
        design, run_starts, n_timepoints = _condition_design(n_runs, run_length, n_conditions)
        generator = torch.Generator().manual_seed(11)

        design_t = torch.from_numpy(design)
        betas = torch.randn(400, n_conditions, generator=generator)
        signal = betas @ design_t.T

        # Scale noise so the true var(signal)/var(y) is 0.25 per voxel.
        target = 0.25
        signal_var = signal.var(dim=1, keepdim=True)
        noise_var = signal_var * (1.0 - target) / target
        noise = torch.randn(400, n_timepoints, generator=generator) * noise_var.sqrt()
        data = signal + noise

        cv_splits = [([r for r in range(n_runs) if r != held], [held]) for held in range(n_runs)]
        result = loro_two_half_ceiling(
            data=data,
            design_matrix=design,
            run_starts=run_starts,
            stim_indices=list(range(n_conditions)),
            nuisance_indices=[],
            cv_splits=cv_splits,
            device=torch.device("cpu"),
            verbose=False,
        )

        assert result.n_usable == n_runs
        # The estimator lands within ~1e-3 of truth here; a tolerance loose
        # enough to survive a biased estimator would defeat the test.
        assert result.ceiling.median().item() == pytest.approx(target, abs=0.01)

    def test_pure_noise_gives_zero_ceiling(self):
        """No signal means nothing reproduces; the ceiling must not invent any."""
        n_runs, run_length, n_conditions = 6, 180, 4
        design, run_starts, n_timepoints = _condition_design(n_runs, run_length, n_conditions)
        generator = torch.Generator().manual_seed(3)
        data = torch.randn(300, n_timepoints, generator=generator)

        cv_splits = [([r for r in range(n_runs) if r != held], [held]) for held in range(n_runs)]
        result = loro_two_half_ceiling(
            data=data,
            design_matrix=design,
            run_starts=run_starts,
            stim_indices=list(range(n_conditions)),
            nuisance_indices=[],
            cv_splits=cv_splits,
            device=torch.device("cpu"),
            verbose=False,
        )
        # Clamped at zero, so the median sits at zero and the mean stays tiny.
        assert result.ceiling.median().item() == pytest.approx(0.0, abs=0.02)
        assert result.ceiling.mean().item() < 0.05

    def test_ceiling_bounds_the_xval_r2_it_pairs_with(self):
        """The whole point: the paired xval R2 must not systematically exceed it."""
        from fastfuncstuff.glm.xval import compute_xval_r2

        n_runs, run_length, n_conditions = 8, 240, 5
        design, run_starts, n_timepoints = _condition_design(
            n_runs, run_length, n_conditions, seed=5
        )
        generator = torch.Generator().manual_seed(21)
        design_t = torch.from_numpy(design)
        betas = torch.randn(300, n_conditions, generator=generator)
        signal = betas @ design_t.T
        target = 0.3
        noise_var = signal.var(dim=1, keepdim=True) * (1.0 - target) / target
        data = signal + torch.randn(300, n_timepoints, generator=generator) * noise_var.sqrt()

        cv_splits = [([r for r in range(n_runs) if r != held], [held]) for held in range(n_runs)]
        common = dict(
            data=data,
            design_matrix=design,
            run_starts=run_starts,
            stim_indices=list(range(n_conditions)),
            nuisance_indices=[],
            cv_splits=cv_splits,
            device=torch.device("cpu"),
            verbose=False,
        )
        ceiling = loro_two_half_ceiling(**common).ceiling
        xval = compute_xval_r2(**common, metric="cod")["r2"]

        assert isinstance(xval, torch.Tensor)
        # The model here IS the true model, so xval R2 should approach but not
        # pass the ceiling; a small overshoot is estimator noise, a large one
        # means the units diverged.
        ratio = (xval / ceiling.clamp_min(1e-6)).median().item()
        assert 0.7 < ratio < 1.15

    def test_too_few_runs_returns_nan_not_a_number(self):
        """Two runs cannot be split; NaN says so instead of quietly guessing."""
        design, run_starts, n_timepoints = _condition_design(2, 120, 3)
        data = torch.randn(50, n_timepoints)
        cv_splits = [([0], [1]), ([1], [0])]
        result = loro_two_half_ceiling(
            data=data,
            design_matrix=design,
            run_starts=run_starts,
            stim_indices=[0, 1, 2],
            nuisance_indices=[],
            cv_splits=cv_splits,
            device=torch.device("cpu"),
            verbose=False,
        )
        assert result.n_usable == 0
        assert torch.isnan(result.ceiling).all()
        assert result.notes


class TestNcsnrCeiling:
    def _planted_betas(self, n_voxels, n_conditions, n_repeats, ncsnr_target, seed=0):
        generator = torch.Generator().manual_seed(seed)
        true_betas = torch.randn(n_voxels, n_conditions, generator=generator)
        signal_sd = true_betas.std(dim=1, keepdim=True)
        noise_sd = signal_sd / ncsnr_target
        trials = true_betas.repeat_interleave(n_repeats, dim=1)
        condition_ids = torch.arange(n_conditions).repeat_interleave(n_repeats)
        noise = torch.randn(n_voxels, n_conditions * n_repeats, generator=generator) * noise_sd
        return trials + noise, condition_ids

    def test_recovers_planted_ncsnr(self):
        betas, condition_ids = self._planted_betas(500, 40, 6, ncsnr_target=1.5, seed=7)
        estimated = ncsnr(betas, condition_ids)
        assert estimated.median().item() == pytest.approx(1.5, abs=0.05)

    def test_nsd_form_matches_closed_form(self):
        betas, condition_ids = self._planted_betas(500, 40, 6, ncsnr_target=2.0, seed=9)
        result = ncsnr_noise_ceiling(betas, condition_ids, n_train_repeats=None)
        expected = 2.0**2 / (2.0**2 + 1.0)
        assert result.ceiling.median().item() == pytest.approx(expected, abs=0.02)
        assert (result.ceiling <= 1.0).all()

    def test_fold_matched_form_is_below_nsd_form(self):
        """The predictor's own noise can only lower the achievable R2."""
        betas, condition_ids = self._planted_betas(400, 30, 8, ncsnr_target=1.2, seed=13)
        nsd = ncsnr_noise_ceiling(betas, condition_ids, n_train_repeats=None).ceiling
        matched = ncsnr_noise_ceiling(betas, condition_ids, n_train_repeats=7).ceiling
        assert matched.median().item() < nsd.median().item()

    def test_unrepeated_conditions_are_not_estimable(self):
        betas = torch.randn(20, 10)
        condition_ids = torch.arange(10)
        result = ncsnr_noise_ceiling(betas, condition_ids)
        assert result.n_usable == 0
        assert torch.isnan(result.ceiling).all()

    def test_pure_noise_gives_zero_ceiling(self):
        generator = torch.Generator().manual_seed(4)
        betas = torch.randn(400, 120, generator=generator)
        condition_ids = torch.arange(20).repeat_interleave(6)
        result = ncsnr_noise_ceiling(betas, condition_ids, n_train_repeats=None)
        assert result.ceiling.median().item() == pytest.approx(0.0, abs=0.05)


class TestDfCorrectedCeiling:
    def test_recovers_planted_ceiling(self):
        n_timepoints, n_columns = 400, 10
        target = 0.2
        # SS_model in expectation is n*var_signal + p*sigma2; build it that way.
        sigma2 = torch.full((200,), 2.0)
        var_signal = torch.full((200,), target / (1 - target) * 2.0)
        ss_model = n_timepoints * var_signal + n_columns * sigma2
        ss_total = n_timepoints * (var_signal + sigma2)
        result = df_corrected_ceiling(ss_model, ss_total, sigma2, n_timepoints, n_columns)
        assert result.ceiling.median().item() == pytest.approx(target, abs=0.01)

    def test_pure_noise_gives_zero(self):
        n_timepoints, n_columns = 300, 8
        sigma2 = torch.full((100,), 1.5)
        ss_model = n_columns * sigma2  # nothing but fitted noise
        ss_total = n_timepoints * sigma2
        result = df_corrected_ceiling(ss_model, ss_total, sigma2, n_timepoints, n_columns)
        assert result.ceiling.max().item() == pytest.approx(0.0, abs=1e-6)

    def test_warns_when_correction_is_large(self):
        sigma2 = torch.full((10,), 1.0)
        result = df_corrected_ceiling(
            torch.full((10,), 100.0), torch.full((10,), 200.0), sigma2, 100, 60
        )
        assert result.notes


class TestZscoreBetasByRun:
    def test_each_run_becomes_zero_mean_unit_scale(self):
        betas = torch.randn(20, 40) * 3.0 + 7.0
        run_ids = torch.arange(4).repeat_interleave(10)
        out = zscore_betas_by_run(betas, run_ids)
        for run in range(4):
            block = out[:, run_ids == run]
            assert torch.allclose(block.mean(dim=1), torch.zeros(20), atol=1e-5)
            assert torch.allclose(block.std(dim=1), torch.ones(20), atol=1e-5)

    def test_removes_the_run_offset_that_inflated_explainable_r2(self):
        """Bug of record: ffs_ridge read explainable R2 = 1.26 without this.

        The CV z-scores per run before scoring, so run-level offsets are absent
        from its R2 but present in a raw-beta ceiling's denominator -- pushing
        the ceiling down and the ratio above 1. Adding a large per-run offset
        must not change the ceiling once both sides are normalised.
        """
        generator = torch.Generator().manual_seed(2)
        base = torch.randn(50, 40, generator=generator)
        run_ids = torch.arange(4).repeat_interleave(10)
        condition_ids = torch.arange(10).repeat(4)

        offset = base + (run_ids.float() * 25.0).unsqueeze(0)
        clean = ncsnr_noise_ceiling(
            zscore_betas_by_run(base, run_ids), condition_ids, n_train_repeats=3
        ).ceiling
        shifted = ncsnr_noise_ceiling(
            zscore_betas_by_run(offset, run_ids), condition_ids, n_train_repeats=3
        ).ceiling
        torch.testing.assert_close(clean, shifted, atol=1e-4, rtol=1e-4)

    def test_rejects_mismatched_run_ids(self):
        with pytest.raises(ValueError, match="one entry per trial"):
            zscore_betas_by_run(torch.randn(5, 10), torch.arange(3))


class TestMeanTrainRepeats:
    def test_counts_actual_training_trials_under_loro(self):
        """4 runs x 3 trials per condition: LORO leaves 9 training trials."""
        condition_ids = torch.zeros(12, dtype=torch.long)
        run_ids = torch.arange(4).repeat_interleave(3)
        splits = [([r for r in range(4) if r != held], [held]) for held in range(4)]
        assert mean_train_repeats(condition_ids, run_ids, splits) == pytest.approx(9.0)

    def test_unbalanced_conditions_are_measured_not_assumed(self):
        """The n*(R-1)/R formula is wrong when a condition skips runs.

        Condition 1 appears in runs 0 and 1 only, so folds holding out those
        runs leave it a single training trial -- which the closed form would
        never predict.
        """
        condition_ids = torch.tensor([0, 0, 0, 0, 1, 1])
        run_ids = torch.tensor([0, 1, 2, 3, 0, 1])
        splits = [([r for r in range(4) if r != held], [held]) for held in range(4)]
        # Condition 0 contributes 3 training trials in each of 4 folds; condition
        # 1 is only tested in folds 0 and 1, contributing 1 each time.
        expected = (3 * 4 + 1 * 2) / 6
        assert mean_train_repeats(condition_ids, run_ids, splits) == pytest.approx(expected)

    def test_no_usable_fold_falls_back_to_one(self):
        condition_ids = torch.tensor([0, 1])
        run_ids = torch.tensor([0, 1])
        assert mean_train_repeats(condition_ids, run_ids, [([0], [1])]) == pytest.approx(1.0)


class TestExplainableR2:
    def _result(self, ceiling: torch.Tensor):
        out = ncsnr_noise_ceiling(
            torch.randn(ceiling.numel(), 40),
            torch.arange(20).repeat_interleave(2),
            n_train_repeats=None,
        )
        out.ceiling = ceiling
        return out

    def test_ratio_is_not_clamped_above_one(self):
        """An explainable R2 above 1 is evidence about the ceiling, so keep it."""
        result = self._result(torch.full((10,), 0.5))
        explained = result.explainable_r2(torch.full((10,), 0.6))
        assert explained.median().item() == pytest.approx(1.2)

    def test_undefined_where_nothing_reproduces(self):
        """A near-zero ceiling has no fraction; dividing anyway gives nonsense.

        Regression: dividing by a 1e-6 floor filled two thirds of a test brain
        with values around -2000, which destroys the map's scaling and any
        summary taken over it.
        """
        result = self._result(torch.tensor([0.0, 0.005, 0.01, 0.4]))
        explained = result.explainable_r2(torch.tensor([-0.004, -0.004, 0.005, 0.2]))
        assert torch.isnan(explained[:2]).all()
        assert torch.isfinite(explained[2:]).all()
        assert explained[3].item() == pytest.approx(0.5)

    def test_summary_ignores_voxels_with_no_ceiling(self):
        """Whole-brain medians are dominated by voxels that never had signal."""
        ceiling = torch.cat([torch.zeros(100), torch.full((10,), 0.4)])
        result = self._result(ceiling)
        summary = result.summarize(result.explainable_r2(torch.full((110,), 0.2)))
        assert "10 voxels" in summary
        assert "0.4000" in summary


class TestCeilingUnderColouredNoise:
    """The two-half construction assumes E[e_A e_B] = 0. White noise is where
    that assumption is least likely to fail, and every other test here plants
    its ratio by scaling ``torch.randn`` -- so none of them probe it.

    This one builds the noise with the simulator instead, giving it the temporal
    autocorrelation real fMRI has. The target is *measured* from the generated
    components rather than planted analytically, which keeps ground truth exact
    without requiring the noise to be white.
    """

    @staticmethod
    def _grand_centred_ratio(signal: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """var(signal)/var(y), centred the way the estimator pools its folds.

        Under LORO every timepoint is held out exactly once, so the estimator's
        denominator is the grand-centred variance over the whole timeseries.
        """
        centred_signal = signal - signal.mean(dim=1, keepdim=True)
        centred_data = data - data.mean(dim=1, keepdim=True)
        return centred_signal.square().sum(dim=1) / centred_data.square().sum(dim=1)

    def test_recovers_the_ceiling_under_ar1_noise(self):
        n_runs, run_length, n_conditions, n_voxels = 8, 240, 6, 300
        rho = 0.35
        design, run_starts, n_timepoints = _condition_design(n_runs, run_length, n_conditions)
        generator = torch.Generator().manual_seed(23)

        design_t = torch.from_numpy(design)
        betas = torch.randn(n_voxels, n_conditions, generator=generator)
        signal = betas @ design_t.T

        # Autocorrelated noise, generated per run: real runs are independent of
        # one another, and a single continuous AR(1) draw would correlate across
        # run boundaries in a way no acquisition does.
        runs = [
            generate_ar1_noise(
                rho,
                n_timepoints=run_length,
                n_voxels=n_voxels,
                device=torch.device("cpu"),
                generator=generator,
            ).T
            for _ in range(n_runs)
        ]
        noise = torch.cat(runs, dim=1)

        target = 0.25
        signal_var = signal.var(dim=1, keepdim=True)
        noise = noise * (signal_var * (1.0 - target) / target).sqrt()
        data = signal + noise

        # Guard the premise: if the noise were not actually autocorrelated this
        # test would silently degrade into a duplicate of the white-noise one.
        centred = noise - noise.mean(dim=1, keepdim=True)
        lag1 = ((centred[:, :-1] * centred[:, 1:]).sum(dim=1) / centred.square().sum(dim=1)).mean()
        assert lag1.item() > 0.25

        expected = self._grand_centred_ratio(signal, data).median().item()

        cv_splits = [([r for r in range(n_runs) if r != held], [held]) for held in range(n_runs)]
        result = loro_two_half_ceiling(
            data=data,
            design_matrix=design,
            run_starts=run_starts,
            stim_indices=list(range(n_conditions)),
            nuisance_indices=[],
            cv_splits=cv_splits,
            device=torch.device("cpu"),
            verbose=False,
        )

        assert result.n_usable == n_runs
        assert result.ceiling.median().item() == pytest.approx(expected, abs=0.02)


def _run_confined_design(n_confined, n_shared=4, n_runs=9, tp=120, n_vox=200, seed=3):
    """Many conditions living in exactly one run each -- the case that breaks the ratio.

    A rich event-coded design (one condition per distinct prompt, say) across
    several runs produces exactly this: most conditions appear in one run only,
    so every LORO fold has conditions it cannot fit.
    """
    torch.manual_seed(seed)
    n_cond = n_shared + n_confined
    n_tp = n_runs * tp
    run_starts = [i * tp for i in range(n_runs)]

    design = torch.zeros(n_tp, n_cond)
    for c in range(n_shared):  # present in every run
        for r in range(n_runs):
            design[r * tp + 3 + 4 * c : r * tp + 3 + 4 * c + 3, c] = 1.0
    for c in range(n_confined):  # one run each
        r = c % n_runs
        start = r * tp + 40 + 2 * (c // n_runs) * 3 + 3 * (c % 7)
        design[start : start + 3, n_shared + c] = 1.0

    betas = torch.randn(n_vox, n_cond) * 2.0
    data = betas @ design.T + torch.randn(n_vox, n_tp) * 2.0
    return data, design, run_starts, n_cond


@pytest.mark.parametrize("strategy", ["zero", "nuisance"])
def test_ceiling_bounds_the_r2_it_is_built_against(strategy):
    """explainable_R2 divides two numbers that must share a denominator.

    Bug of record: -zero_event nuisance scores the R2 only on the subspace a
    fold could predict, shrinking its SS_tot, while the ceiling kept dividing by
    the full held-out variance. The ratio then ran well above 1 -- reported from
    a real 9-run run as values of 8 and 25 -- which reads as "the model beat the
    ceiling" when it actually means the two were never on the same scale.

    A ceiling is an upper bound, so the ratio must sit at or below 1 for all but
    a noise-sized minority of voxels, under either policy.
    """
    from fastfuncstuff.glm.xval import compute_xval_r2, generate_cv_splits
    from fastfuncstuff.stats.noise_ceiling import loro_two_half_ceiling

    data, design, run_starts, n_cond = _run_confined_design(n_confined=90)
    device = torch.device("cpu")
    splits = generate_cv_splits(len(run_starts), strategy=1)
    shared = dict(
        data=data,
        design_matrix=design,
        run_starts=run_starts,
        stim_indices=list(range(n_cond)),
        nuisance_indices=[],
        cv_splits=splits,
        device=device,
    )

    r2 = compute_xval_r2(**shared, metric="cod", zero_event_strategy=strategy, verbose=False)["r2"]
    ceiling = loro_two_half_ceiling(**shared, zero_event_strategy=strategy, verbose=False)

    explainable = ceiling.explainable_r2(r2)
    defined = explainable[~explainable.isnan()]
    assert defined.numel() > 50, "need enough defined voxels for this to mean anything"
    assert defined.median() <= 1.0, f"median explainable R2 = {defined.median():.3f}"
    # A bound that is exceeded by a fifth of the brain is not a bound.
    assert (defined > 1.05).float().mean() < 0.1


def test_ceiling_follows_the_strategy_rather_than_ignoring_it():
    """The ceiling must actually move when the policy changes.

    Guards the fix from being satisfied by a ceiling that quietly ignores the
    argument: under 'nuisance' the held-out variance loses the unpredictable
    conditions' span, so the ceiling on what remains is strictly higher.
    """
    from fastfuncstuff.glm.xval import generate_cv_splits
    from fastfuncstuff.stats.noise_ceiling import loro_two_half_ceiling

    data, design, run_starts, n_cond = _run_confined_design(n_confined=90)
    shared = dict(
        data=data,
        design_matrix=design,
        run_starts=run_starts,
        stim_indices=list(range(n_cond)),
        nuisance_indices=[],
        cv_splits=generate_cv_splits(len(run_starts), strategy=1),
        device=torch.device("cpu"),
        verbose=False,
    )
    zero = loro_two_half_ceiling(**shared, zero_event_strategy="zero")
    nuis = loro_two_half_ceiling(**shared, zero_event_strategy="nuisance")

    assert nuis.ceiling.median() > zero.ceiling.median()


def _sparse_repeat_design(reps, n_cond=60, n_runs=9, tp=150, n_vox=200, seed=5):
    """Each condition appears in ``reps`` of the runs, chosen at random.

    This is what a richly event-coded design looks like -- one condition per
    distinct prompt, each recurring a handful of times across the session --
    and it is the case the two-half ceiling cannot measure.
    """
    torch.manual_seed(seed)
    n_tp = n_runs * tp
    run_starts = [i * tp for i in range(n_runs)]
    design = torch.zeros(n_tp, n_cond)
    gen = torch.Generator().manual_seed(7)
    for c in range(n_cond):
        runs = (
            list(range(n_runs))
            if reps >= n_runs
            else torch.randperm(n_runs, generator=gen)[:reps].tolist()
        )
        for j, r in enumerate(runs):
            start = r * tp + 10 + ((c * 7 + j * 13) % (tp - 20))
            design[start : start + 3, c] = 1.0
    betas = torch.randn(n_vox, n_cond) * 2.0
    data = betas @ design.T + torch.randn(n_vox, n_tp) * 2.0
    return data, design, run_starts, n_cond


def test_two_half_ceiling_declares_itself_unreliable_on_sparse_repeats():
    """A ceiling the design cannot support must say so, not just be wrong.

    The two-half split measures reproducibility by correlating two independent
    estimates of the same beta, so a condition has to appear on BOTH sides of
    the split. When most conditions repeat only a few times across runs, most
    are unmeasurable: their signal is absent from cov() while the R2 fits them
    perfectly well, so the ceiling is a lower bound and explainable_R2 exceeds 1
    by construction. Measured here, median explainable R2 tracks the excluded
    fraction monotonically -- 0/60 excluded gives 0.96, 43/60 gives 1.45 -- so
    the number alone can never tell a user which regime they are in.
    """
    from fastfuncstuff.glm.xval import generate_cv_splits
    from fastfuncstuff.stats.noise_ceiling import loro_two_half_ceiling

    def notes_for(reps):
        data, design, run_starts, n_cond = _sparse_repeat_design(reps)
        result = loro_two_half_ceiling(
            data=data,
            design_matrix=design,
            run_starts=run_starts,
            stim_indices=list(range(n_cond)),
            nuisance_indices=[],
            cv_splits=generate_cv_splits(len(run_starts), strategy=1),
            device=torch.device("cpu"),
            verbose=False,
        )
        return result.notes

    sparse = notes_for(3)
    assert any("UNRELIABLE" in n for n in sparse), sparse
    assert any("could not be measured" in n for n in sparse), sparse

    # A design every condition spans is measurable, and must not be flagged.
    assert not any("UNRELIABLE" in n for n in notes_for(9))


def test_explainable_r2_tracks_how_much_of_the_design_is_measurable():
    """Pins the mechanism itself, so a future 'fix' that rescales is caught.

    The bound is exceeded because var(s) is missing conditions, not because of
    any scaling error -- so the ratio has to fall as more of the design becomes
    measurable, and reach <= 1 once all of it is.
    """
    from fastfuncstuff.glm.xval import compute_xval_r2, generate_cv_splits
    from fastfuncstuff.stats.noise_ceiling import loro_two_half_ceiling

    medians = {}
    for reps in (3, 9):
        data, design, run_starts, n_cond = _sparse_repeat_design(reps)
        shared = dict(
            data=data,
            design_matrix=design,
            run_starts=run_starts,
            stim_indices=list(range(n_cond)),
            nuisance_indices=[],
            cv_splits=generate_cv_splits(len(run_starts), strategy=1),
            device=torch.device("cpu"),
        )
        r2 = compute_xval_r2(**shared, metric="cod", verbose=False)["r2"]
        ceiling = loro_two_half_ceiling(**shared, verbose=False)
        explainable = ceiling.explainable_r2(r2)
        medians[reps] = float(explainable[~explainable.isnan()].median())

    assert medians[3] > 1.2, medians
    assert medians[9] <= 1.0, medians


def _wide_vs_narrow_designs(n_runs=9, tp=200, n_prompts=180):
    """One column per prompt, each prompt occurring exactly once, plus a 2-condition
    collapse of the same events. The comparison a rich event design forces."""
    from fastfuncstuff.glm.core import construct_polynomial_matrix

    n_tp = n_runs * tp
    run_starts = [i * tp for i in range(n_runs)]
    wide = torch.zeros(n_tp, n_prompts)
    for c in range(n_prompts):
        start = (c % n_runs) * tp + 10 + (c // n_runs) * 9
        wide[start : start + 4, c] = 1.0
    narrow = torch.zeros(n_tp, 2)
    for c in range(n_prompts):
        narrow[:, c % 2] += wide[:, c]
    nuisance = [
        construct_polynomial_matrix(tp, 3, torch.device("cpu")).float() for _ in range(n_runs)
    ]
    return wide, narrow, nuisance, run_starts


def _df_ceiling(data, design, nuisance, run_starts):
    from fastfuncstuff.stats.noise_ceiling import df_ceiling_by_voxel_group

    return df_ceiling_by_voxel_group(
        data=data,
        nuisance_per_run=nuisance,
        run_starts=run_starts,
        design_matrix=design,
        device=torch.device("cpu"),
    ).ceiling.nanmedian()


def _add_drift(data, nuisance, run_starts, tp):
    for r, block in enumerate(nuisance):
        sl = slice(run_starts[r], run_starts[r] + tp)
        data[:, sl] += torch.randn(data.shape[0], block.shape[1]) @ block.T * 2.0
    return data


def test_df_ceiling_does_not_reward_a_model_for_having_more_columns():
    """Free parameters must not buy ceiling. This is why df is comparable ACROSS models.

    The two-half ceiling cannot rank a 2-condition model against a 180-condition
    one at all, because the wide model's conditions never repeat. The df ceiling
    can, because it subtracts p * sigma2 -- exactly the fit that p free columns
    extract from noise. So when the truth really is two conditions, the wide
    model's 178 extra columns must buy it essentially nothing.
    """
    torch.manual_seed(11)
    n_runs, tp, n_vox = 9, 200, 400
    wide, narrow, nuisance, run_starts = _wide_vs_narrow_designs(n_runs, tp)

    betas = torch.randn(n_vox, 2) * 2.0
    data = betas @ narrow.T + torch.randn(n_vox, n_runs * tp) * 2.0
    data = _add_drift(data, nuisance, run_starts, tp)

    narrow_ceiling = _df_ceiling(data, narrow, nuisance, run_starts)
    wide_ceiling = _df_ceiling(data, wide, nuisance, run_starts)

    assert abs(float(wide_ceiling) - float(narrow_ceiling)) < 0.02, (
        f"178 columns of pure overfit moved the ceiling: "
        f"{float(narrow_ceiling):.4f} -> {float(wide_ceiling):.4f}"
    )


def test_df_ceiling_credits_a_wide_model_when_the_signal_really_is_wide():
    """The correction must not be so blunt that real per-condition signal vanishes.

    Pairs with the overfit test: together they say the ceiling tracks signal
    rather than column count in both directions.
    """
    torch.manual_seed(11)
    n_runs, tp, n_vox = 9, 200, 400
    wide, narrow, nuisance, run_starts = _wide_vs_narrow_designs(n_runs, tp)

    betas = torch.randn(n_vox, wide.shape[1]) * 2.0
    data = betas @ wide.T + torch.randn(n_vox, n_runs * tp) * 2.0
    data = _add_drift(data, nuisance, run_starts, tp)

    assert float(_df_ceiling(data, wide, nuisance, run_starts)) > 0.15
    # The 2-condition model genuinely cannot represent prompt-specific responses.
    assert float(_df_ceiling(data, narrow, nuisance, run_starts)) < 0.02


def test_df_ceiling_is_zero_on_pure_noise_for_both_designs():
    """The df correction has to be exact, not approximate, or wide designs drift up."""
    torch.manual_seed(11)
    n_runs, tp, n_vox = 9, 200, 400
    wide, narrow, nuisance, run_starts = _wide_vs_narrow_designs(n_runs, tp)
    data = _add_drift(torch.randn(n_vox, n_runs * tp) * 2.0, nuisance, run_starts, tp)

    assert float(_df_ceiling(data, narrow, nuisance, run_starts)) < 0.01
    assert float(_df_ceiling(data, wide, nuisance, run_starts)) < 0.01


# ---------------------------------------------------------------------------
# repeat ceiling: runs whose design is identical
# ---------------------------------------------------------------------------


def _repeat_dataset(n_runs=4, run_len=150, n_vox=300, drift=3.0, seed=0):
    """Runs replaying one signal, each with independent noise and its own drift.

    var(signal) = amp**2 and var(noise) = 1 per voxel, so the true ceiling is
    amp**2 / (amp**2 + 1) analytically.
    """
    import torch

    torch.manual_seed(seed)
    run_starts = [i * run_len for i in range(n_runs)]
    signal = torch.randn(run_len)
    signal = (signal - signal.mean()) / signal.std()
    amp = torch.linspace(0.2, 3.0, n_vox).unsqueeze(1)

    ramp = torch.linspace(-1, 1, run_len)
    data = torch.zeros(n_vox, n_runs * run_len)
    for r in range(n_runs):
        block = amp * signal + torch.randn(n_vox, run_len)
        block = block + (torch.randn(n_vox, 1) * drift) * ramp
        data[:, r * run_len : (r + 1) * run_len] = block

    nuisance = [torch.stack([torch.ones(run_len), ramp, ramp**2], dim=1) for _ in range(n_runs)]
    true_ceiling = (amp.squeeze() ** 2) / (amp.squeeze() ** 2 + 1.0)
    return data, nuisance, run_starts, signal, true_ceiling


def test_repeat_ceiling_recovers_the_analytic_value():
    import torch

    from fastfuncstuff.stats.noise_ceiling import repeat_ceiling

    data, nuisance, run_starts, _, true_ceiling = _repeat_dataset()
    result = repeat_ceiling(
        data=data,
        nuisance_per_run=nuisance,
        run_starts=run_starts,
        repeat_groups=[[0, 1, 2, 3]],
    )

    assert result.method == "repeat"
    assert result.n_usable == 6  # pairs, not runs
    assert torch.median((result.ceiling - true_ceiling).abs()) < 0.03


def test_repeat_ceiling_is_in_the_units_of_the_xval_r2():
    """An oracle model that knows the signal must land at explainable R2 ~= 1.

    This is the property that makes the ceiling usable as a divisor, and the one
    a units mistake breaks: a ceiling can be perfectly precise and still be the
    wrong denominator for the R2 it is dividing.
    """
    import torch

    from fastfuncstuff.glm.xval import project_out_nuisance_per_run
    from fastfuncstuff.stats.noise_ceiling import repeat_ceiling

    data, nuisance, run_starts, signal, _ = _repeat_dataset(seed=1)
    n_runs, run_len, n_vox = len(run_starts), signal.shape[0], data.shape[0]

    ceiling = repeat_ceiling(
        data=data,
        nuisance_per_run=nuisance,
        run_starts=run_starts,
        repeat_groups=[list(range(n_runs))],
    ).ceiling

    design = torch.cat([signal] * n_runs).unsqueeze(1)
    projected, projected_design = project_out_nuisance_per_run(
        data=data,
        design=design,
        nuisance_per_run=nuisance,
        run_starts=run_starts,
        device=data.device,
    )

    ss_res = torch.zeros(n_vox, dtype=torch.float64)
    ss_tot = torch.zeros(n_vox, dtype=torch.float64)
    for held in range(n_runs):
        train = torch.cat(
            [torch.arange(r * run_len, (r + 1) * run_len) for r in range(n_runs) if r != held]
        )
        test = torch.arange(held * run_len, (held + 1) * run_len)
        betas = torch.linalg.pinv(projected_design[train]) @ projected[:, train].T
        pred = (projected_design[test] @ betas).T
        actual = projected[:, test]
        ss_res += ((actual - pred) ** 2).sum(dim=1).double()
        ss_tot += ((actual - actual.mean(dim=1, keepdim=True)) ** 2).sum(dim=1).double()

    oracle_r2 = (1 - ss_res / ss_tot).float()
    explainable = torch.median(oracle_r2 / ceiling)
    assert 0.9 < explainable < 1.1, explainable


def test_repeat_ceiling_needs_the_nuisance_projected_out():
    """Shared drift is reproducible, so skipping the projection inflates it.

    Guards the docstring's loudest warning: the drift here is common to every
    run, which is exactly the case where an unprojected ceiling goes to ~1
    everywhere and makes every model look terrible.
    """
    import torch

    from fastfuncstuff.stats.reliability import split_half_noise_ceiling

    data, nuisance, run_starts, _, true_ceiling = _repeat_dataset(drift=0.0, seed=2)
    run_len = nuisance[0].shape[0]
    n_runs = len(run_starts)

    # One drift shared by every run, the way a scanner trend is.
    shared = torch.linspace(-1, 1, run_len) * 6.0
    drifted = data.clone()
    for r in range(n_runs):
        drifted[:, r * run_len : (r + 1) * run_len] += shared

    unprojected = split_half_noise_ceiling(
        data=drifted,
        repeat_groups=[list(range(n_runs))],
        run_starts=run_starts,
        n_timepoints=drifted.shape[1],
    )
    assert torch.median(unprojected) > 0.9
    assert torch.median(unprojected) > torch.median(true_ceiling) + 0.2


def test_repeat_ceiling_without_repeats_is_nan_not_zero():
    import torch

    from fastfuncstuff.stats.noise_ceiling import repeat_ceiling

    data, nuisance, run_starts, _, _ = _repeat_dataset(seed=3)
    result = repeat_ceiling(
        data=data, nuisance_per_run=nuisance, run_starts=run_starts, repeat_groups=[]
    )

    assert result.n_usable == 0
    assert torch.isnan(result.ceiling).all()
    assert result.notes


def test_repeat_ceiling_notes_runs_that_did_not_contribute():
    from fastfuncstuff.stats.noise_ceiling import repeat_ceiling

    data, nuisance, run_starts, _, _ = _repeat_dataset(seed=4)
    result = repeat_ceiling(
        data=data,
        nuisance_per_run=nuisance,
        run_starts=run_starts,
        repeat_groups=[[0, 1]],  # runs 3 and 4 match nothing
    )

    assert result.n_usable == 1
    assert any("did not contribute" in note for note in result.notes)


def test_detect_repeat_groups_separates_unlike_designs():
    """Two runs of design A, one of design B, and near-equal must still match."""
    import torch

    from fastfuncstuff.denoise.sequential import _detect_repeat_groups

    torch.manual_seed(5)
    run_len = 60
    a = torch.randn(run_len, 2)
    b = torch.randn(run_len, 2)
    # Run 3 is run 1 rebuilt: same timing, last-bit convolution differences.
    a_again = a + torch.randn(run_len, 2) * 1e-9

    design = torch.cat([a, b, a_again], dim=0)
    groups = _detect_repeat_groups(design, [0, run_len, 2 * run_len], 3 * run_len)

    assert groups == [[0, 2]]


def test_repeat_ceiling_does_not_move_with_the_design():
    """The property that makes it a design diagnostic.

    The loro ceiling falls when the design misses real signal, so explainable R2
    stays near 1 and the model looks fine. This ceiling is estimated without
    fitting anything, so it holds still and the explainable R2 drops instead --
    which is the answer you want. Measured end to end on synthetic data: with a
    design missing half its blocks, repeat gave 0.50 explainable where loro gave
    0.99.
    """
    import torch

    from fastfuncstuff.stats.noise_ceiling import repeat_ceiling

    data, nuisance, run_starts, signal, _ = _repeat_dataset(seed=7)
    n_runs = len(run_starts)
    groups = [list(range(n_runs))]

    good = torch.cat([signal] * n_runs).unsqueeze(1)
    wrong = torch.randn(good.shape)

    kwargs = dict(data=data, nuisance_per_run=nuisance, run_starts=run_starts, repeat_groups=groups)
    with_good = repeat_ceiling(**kwargs, design_matrix=good).ceiling
    with_wrong = repeat_ceiling(**kwargs, design_matrix=wrong).ceiling
    with_none = repeat_ceiling(**kwargs, design_matrix=None).ceiling

    torch.testing.assert_close(with_good, with_wrong)
    torch.testing.assert_close(with_good, with_none)
