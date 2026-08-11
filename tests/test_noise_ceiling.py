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
    df_corrected_ceiling,
    loro_two_half_ceiling,
    mean_train_repeats,
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
