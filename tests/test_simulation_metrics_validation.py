"""Design-efficiency metrics must reproduce the theory they cite.

These are the checks that distinguish a metric from a number: the Pareto
trade-off between detection and estimation ([[Liu 2004]]), the optimal event
probability 1/(Q+1), conditional entropy bounded by log2(Q+1), and a GLS
variance that matches the spread actually observed across simulations.

Each corresponds to a defect found on 2026-08-11: detection power that counted
the baseline as signal, a documented FIR mode that returned the onset column
unchanged, an "entropy" that was unbounded and blind to trial-type order, a
Var(beta) missing its noise-variance factor, and -- worst -- a simulator whose
noise reshape destroyed the temporal autocorrelation it had just generated.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.simulation.core import create_parametric_voxels, simulate_fmri_run
from fastfuncstuff.simulation.metrics import (
    compute_conditional_entropy,
    compute_design_matrix_for_condition,
    compute_detection_power,
    compute_estimation_efficiency,
)
from fastfuncstuff.simulation.metrics_empirical import (
    build_ar1_covariance_matrix,
    estimate_ar1_coefficient,
    gls_fit,
)

CPU = torch.device("cpu")
N_TIMEPOINTS = 480
HRF_LAGS = 20


def _canonical_hrf(length: int = HRF_LAGS) -> np.ndarray:
    t = np.arange(length, dtype=np.float64)
    h = (t**5) * np.exp(-t) / 120.0 - 0.35 * (t**7) * np.exp(-t) / 5040.0
    return (h / np.abs(h).max()).astype(np.float32)


def _block_design(n_t: int = N_TIMEPOINTS, block: int = 20) -> np.ndarray:
    onsets = np.zeros((n_t, 1), dtype=np.float32)
    for start in range(0, n_t, block * 2):
        onsets[start : start + block, 0] = 1.0
    return onsets


def _matched_random_design(reference: np.ndarray, seed: int = 0) -> np.ndarray:
    """Same number of event TRs as `reference`, randomly placed."""
    rng = np.random.default_rng(seed)
    onsets = np.zeros_like(reference)
    onsets[rng.choice(reference.shape[0], int(reference.sum()), replace=False), 0] = 1.0
    return onsets


def _lag1(x: torch.Tensor) -> float:
    """Mean lag-1 autocorrelation. x: (n_voxels, n_timepoints)."""
    x = x - x.mean(dim=1, keepdim=True)
    return float(((x[:, :-1] * x[:, 1:]).sum(dim=1) / (x * x).sum(dim=1)).mean().item())


class TestParetoTradeoff:
    """The one qualitative claim the whole module rests on."""

    def test_block_detects_better_and_estimates_worse_than_random(self):
        hrf = _canonical_hrf()
        block = _block_design()
        random = _matched_random_design(block)
        assert block.sum() == random.sum()  # arrangement differs, amount does not

        block_power = compute_detection_power(block, hrf, n_conditions=1, device=CPU)["total"]
        random_power = compute_detection_power(random, hrf, n_conditions=1, device=CPU)["total"]
        block_eff = compute_estimation_efficiency(block, 1, HRF_LAGS, device=CPU)["total"]
        random_eff = compute_estimation_efficiency(random, 1, HRF_LAGS, device=CPU)["total"]

        assert block_power > random_power
        assert block_eff < random_eff

    def test_optimal_event_probability_is_one_over_q_plus_one(self):
        """For Q=1 trial type plus null, theory puts the optimum at p = 0.5."""
        scores = {}
        for p in (0.1, 0.3, 0.5, 0.7, 0.9):
            rng = np.random.default_rng(7)
            onsets = (rng.random((N_TIMEPOINTS, 1)) < p).astype(np.float32)
            scores[p] = compute_estimation_efficiency(onsets, 1, HRF_LAGS, device=CPU)["total"]
        assert max(scores, key=scores.get) == 0.5


class TestBaselineProjection:
    """A design's power is what a contrast can use, and no contrast can use DC."""

    def test_detection_power_excludes_the_baseline(self):
        hrf = _canonical_hrf()
        block = _block_design()
        with_baseline = compute_detection_power(
            block, hrf, n_conditions=1, device=CPU, poly_degree=0
        )["total"]
        without = compute_detection_power(block, hrf, n_conditions=1, device=CPU, poly_degree=-1)[
            "total"
        ]
        # Leaving the baseline in inflates the score; here it was 55% of it.
        assert without > with_baseline * 1.5

    def test_ignoring_the_baseline_understates_the_block_advantage(self):
        """Not merely a rescaling -- it changes the ranking's magnitude.

        The DC term is a larger share of a random design's raw score (84%) than
        of a block design's (55%), so leaving it in compresses precisely the
        contrast this metric exists to report.
        """
        hrf = _canonical_hrf()
        block = _block_design()
        random = _matched_random_design(block)

        def ratio(poly_degree):
            b = compute_detection_power(
                block, hrf, n_conditions=1, device=CPU, poly_degree=poly_degree
            )["total"]
            r = compute_detection_power(
                random, hrf, n_conditions=1, device=CPU, poly_degree=poly_degree
            )["total"]
            return b / r

        assert ratio(0) > 2.0 * ratio(-1)

    def test_higher_polynomial_degrees_cost_a_block_design_more(self):
        """Liu 2004 Fig 2: block designs are the ones drift regressors hurt."""
        hrf = _canonical_hrf()
        block = _block_design()
        random = _matched_random_design(block)

        def loss(design):
            low = compute_detection_power(design, hrf, n_conditions=1, device=CPU, poly_degree=0)[
                "total"
            ]
            high = compute_detection_power(design, hrf, n_conditions=1, device=CPU, poly_degree=4)[
                "total"
            ]
            return (low - high) / low

        assert loss(block) > loss(random)


class TestFirDesignMatrix:
    def test_fir_mode_returns_a_lagged_matrix(self):
        """It used to return the onset column, identical to mode='onoff'."""
        onsets = torch.from_numpy(_matched_random_design(_block_design()))
        onoff = compute_design_matrix_for_condition(onsets, 0, N_TIMEPOINTS, mode="onoff")
        fir = compute_design_matrix_for_condition(
            onsets, 0, N_TIMEPOINTS, mode="fir", hrf_length=HRF_LAGS
        )
        assert onoff.shape == (N_TIMEPOINTS, 1)
        assert fir.shape == (N_TIMEPOINTS, HRF_LAGS)
        for lag in (1, 5, 19):
            assert torch.equal(fir[lag:, lag], onsets[: N_TIMEPOINTS - lag, 0])

    def test_fir_mode_requires_hrf_length(self):
        onsets = torch.zeros(10, 1)
        with pytest.raises(ValueError, match="hrf_length"):
            compute_design_matrix_for_condition(onsets, 0, 10, mode="fir")


class TestConditionalEntropy:
    def test_block_design_is_predictable_and_random_is_not(self):
        block = _block_design()
        rng = np.random.default_rng(1)
        random = (rng.random((N_TIMEPOINTS, 1)) < 0.5).astype(np.float32)

        block_h = compute_conditional_entropy(block, n_conditions=1, device=CPU)
        random_h = compute_conditional_entropy(random, n_conditions=1, device=CPU)

        assert block_h["conditional_entropy"] < 0.35
        assert random_h["conditional_entropy"] == pytest.approx(random_h["max_entropy"], abs=0.05)

    def test_entropy_respects_its_theoretical_bound(self):
        """The ISI entropy this once reported is unbounded -- a random design
        measured 1.95 bits where H_r for Q=1 cannot exceed 1."""
        rng = np.random.default_rng(2)
        random = (rng.random((N_TIMEPOINTS, 1)) < 0.5).astype(np.float32)
        result = compute_conditional_entropy(random, n_conditions=1, device=CPU)
        assert result["conditional_entropy"] <= result["max_entropy"] + 1e-9

    def test_trial_type_predictability_is_visible(self):
        """Identical onsets, different type orders. A metric for design
        randomness that cannot separate ABAB from random is not measuring it."""
        event_times = np.arange(10, N_TIMEPOINTS - 30, 8)
        alternating = np.zeros((N_TIMEPOINTS, 2), dtype=np.float32)
        randomised = np.zeros((N_TIMEPOINTS, 2), dtype=np.float32)
        rng = np.random.default_rng(3)
        for i, t in enumerate(event_times):
            alternating[t, i % 2] = 1.0
            randomised[t, rng.integers(2)] = 1.0

        alt_h = compute_conditional_entropy(alternating, n_conditions=2, device=CPU)
        rnd_h = compute_conditional_entropy(randomised, n_conditions=2, device=CPU)

        assert alt_h["type_entropy"] == pytest.approx(0.0, abs=1e-9)
        assert rnd_h["type_entropy"] > 0.8
        assert rnd_h["type_entropy"] <= rnd_h["max_type_entropy"] + 1e-9


class TestGlsVariance:
    def test_reported_variance_matches_the_observed_spread(self):
        """Var(beta) = sigma^2 (X' S^-1 X)^-1. Dropping sigma^2 understated the
        standard errors by a factor of sigma -- t-statistics twice too large."""
        n_t, true_rho, true_sigma, n_sims = 200, 0.6, 2.0, 300
        hrf = _canonical_hrf()
        rng = np.random.default_rng(4)
        onsets = (rng.random(n_t) < 0.3).astype(np.float32)
        X = np.column_stack([np.convolve(onsets, hrf)[:n_t], np.ones(n_t)]).astype(np.float32)
        sigma = build_ar1_covariance_matrix(n_t, true_rho, device=CPU)
        chol = np.linalg.cholesky(sigma.numpy().astype(np.float64) + 1e-9 * np.eye(n_t))
        true_beta = np.array([3.0, 100.0], dtype=np.float32)

        estimates = []
        for seed in range(n_sims):
            noise = chol @ np.random.default_rng(seed).standard_normal(n_t) * true_sigma
            y = (X @ true_beta + noise).astype(np.float32)
            estimates.append(gls_fit(y, X, sigma, device=CPU)["betas"][:, 0].numpy())
        estimates = np.array(estimates)

        # Unbiased
        assert estimates.mean(axis=0)[0] == pytest.approx(true_beta[0], abs=0.05)

        noise = chol @ np.random.default_rng(999).standard_normal(n_t) * true_sigma
        fit = gls_fit((X @ true_beta + noise).astype(np.float32), X, sigma, device=CPU)
        reported_sd = np.sqrt(np.diag(fit["var_betas"].numpy()))
        assert estimates.std(axis=0)[0] == pytest.approx(reported_sd[0], rel=0.2)
        assert np.sqrt(fit["sigma2"]) == pytest.approx(true_sigma, rel=0.2)

    def test_design_variance_is_free_of_the_noise_level(self):
        """A design-efficiency figure must describe the design, not this dataset."""
        n_t = 200
        hrf = _canonical_hrf()
        rng = np.random.default_rng(5)
        onsets = (rng.random(n_t) < 0.3).astype(np.float32)
        X = np.column_stack([np.convolve(onsets, hrf)[:n_t], np.ones(n_t)]).astype(np.float32)
        sigma = build_ar1_covariance_matrix(n_t, 0.4, device=CPU)
        y = (X @ np.array([2.0, 100.0], dtype=np.float32)).astype(np.float32)

        quiet = gls_fit(y + rng.standard_normal(n_t).astype(np.float32), X, sigma, device=CPU)
        loud = gls_fit(y + 10.0 * rng.standard_normal(n_t).astype(np.float32), X, sigma, device=CPU)
        assert torch.allclose(quiet["var_betas_design"], loud["var_betas_design"], atol=1e-6)
        assert loud["sigma2"] > 10.0 * quiet["sigma2"]

    @pytest.mark.parametrize("true_rho", [0.2, 0.5, 0.8])
    def test_ar1_coefficient_round_trip(self, true_rho):
        rng = np.random.default_rng(6)
        e = np.zeros(4000)
        innovations = rng.standard_normal(4000)
        for t in range(1, 4000):
            e[t] = true_rho * e[t - 1] + innovations[t]
        assert estimate_ar1_coefficient(e) == pytest.approx(true_rho, abs=0.03)


class TestSimulateFmriRun:
    def test_noise_keeps_the_temporal_structure_it_was_given(self):
        """The headline defect: generate_fmri_noise returns (n_trs, nx, ny) --
        time FIRST -- and reshaping it as (-1, n_timepoints) interleaves time
        with space instead of transposing. Each voxel's "timeseries" became a
        stride across space at nearly fixed time, so every simulation built on
        this function silently got white noise from a 1/f generator.
        """
        n_t = 300
        torch.manual_seed(0)
        data = simulate_fmri_run(
            torch.zeros(n_t, 1),
            [0.0],
            torch.ones(1),
            tr=1.0,
            n_timepoints=n_t,
            matrix_size=(4, 4, 2),
            noise_level=1.0,
            baseline=100.0,
            add_scanner_drift=False,
            device=CPU,
        )
        assert _lag1(data.reshape(-1, n_t)) > 0.25

    def test_planted_betas_are_recoverable(self):
        n_t = 300
        onsets = torch.zeros(n_t, 2)
        onsets[np.arange(5, n_t - 40, 17), 0] = 1.0
        onsets[np.arange(12, n_t - 40, 19), 1] = 1.0
        hrf = torch.from_numpy(_canonical_hrf(30))
        data = simulate_fmri_run(
            onsets,
            torch.tensor([2.0, -1.0]),
            hrf,
            tr=1.0,
            n_timepoints=n_t,
            matrix_size=(4, 4, 2),
            noise_level=0.01,
            baseline=100.0,
            add_scanner_drift=False,
            device=CPU,
        )
        design = np.column_stack(
            [np.convolve(onsets[:, k].numpy(), hrf.numpy())[:n_t] for k in range(2)]
            + [np.ones(n_t)]
        ).astype(np.float32)
        fitted = np.linalg.lstsq(design, data.reshape(-1, n_t).T.numpy(), rcond=None)[0]
        assert fitted[0].mean() == pytest.approx(2.0, abs=0.01)
        assert fitted[1].mean() == pytest.approx(-1.0, abs=0.01)
        assert fitted[2].mean() == pytest.approx(100.0, abs=0.05)

    def test_per_voxel_noise_levels_are_honoured(self):
        """create_parametric_voxels returns a noise level per voxel; it was
        unusable because simulate_fmri_run only accepted a scalar."""
        n_t, shape = 300, (8, 4, 3)
        torch.manual_seed(1)
        _, _, noise_levels = create_parametric_voxels(shape, 1, device=CPU)
        data = simulate_fmri_run(
            torch.zeros(n_t, 1),
            [0.0],
            torch.ones(1),
            tr=1.0,
            n_timepoints=n_t,
            matrix_size=shape,
            noise_level=noise_levels,
            baseline=100.0,
            add_scanner_drift=False,
            device=CPU,
        )
        observed = data.reshape(-1, n_t).std(dim=1).reshape(shape)
        for z in range(shape[2]):
            expected = noise_levels.reshape(shape)[:, :, z].mean().item()
            assert observed[:, :, z].mean().item() == pytest.approx(expected, rel=0.15)


class TestParametricVoxels:
    @pytest.mark.parametrize("shape", [(4, 4, 2), (2, 10, 2), (8, 4, 3)])
    def test_small_volumes_do_not_crash(self, shape):
        """ny // 20 is zero for any ny < 20, so every ordinary test volume hit a
        ZeroDivisionError; likewise nx // n_hrfs when there are more HRFs than
        columns."""
        betas, hrf_indices, noise_levels = create_parametric_voxels(
            shape, 1, hrf_library=torch.randn(5, 20), device=CPU
        )
        n_voxels = shape[0] * shape[1] * shape[2]
        assert betas.shape == (n_voxels, 1)
        assert hrf_indices.shape == (n_voxels,)
        assert noise_levels.shape == (n_voxels,)

    def test_layout_matches_the_volume_reshape(self):
        """The docstring promises HRFs along X and noise along Z. The voxels were
        written z-major while simulate_fmri_run reshapes as (nx, ny, nz), so that
        organisation did not survive the round trip into a volume."""
        shape = (8, 4, 3)
        _, hrf_indices, noise_levels = create_parametric_voxels(
            shape, 1, hrf_library=torch.randn(4, 20), device=CPU
        )
        hrf_volume = hrf_indices.reshape(shape)
        noise_volume = noise_levels.reshape(shape)

        for x in range(shape[0]):
            assert len(set(hrf_volume[x].flatten().tolist())) == 1, "HRF must vary along X only"
        for z in range(shape[2]):
            assert len(set(noise_volume[:, :, z].flatten().tolist())) == 1, "noise varies along Z"
        assert len(set(hrf_indices.tolist())) > 1
