"""Noise-ceiling estimators must recover a planted var(signal)/var(y).

Every test here builds data whose true ceiling is known by construction, so a
regression shows up as a biased ceiling rather than as a plausible-looking map.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.stats.noise_ceiling import (
    df_corrected_ceiling,
    loro_two_half_ceiling,
    ncsnr,
    ncsnr_noise_ceiling,
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


class TestExplainableR2:
    def test_ratio_is_not_clamped_above_one(self):
        """An explainable R2 above 1 is evidence about the ceiling, so keep it."""
        result = ncsnr_noise_ceiling(
            torch.randn(10, 40), torch.arange(20).repeat_interleave(2), n_train_repeats=None
        )
        result.ceiling = torch.full((10,), 0.5)
        explained = result.explainable_r2(torch.full((10,), 0.6))
        assert explained.median().item() == pytest.approx(1.2)
