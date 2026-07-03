"""Tests for design optimization module - ISI generation, event sequences, onset matrices."""

import numpy as np

from fastfuncstuff.design.optimization import (
    ISIConstraints,
    create_onset_matrix,
    generate_event_sequence,
    generate_isi_sequence,
)
from fastfuncstuff.simulation.metrics_empirical import estimate_ar1_coefficient


class TestGenerateEventSequence:
    def test_alternating_order(self):
        seq = generate_event_sequence(
            n_trials_per_condition=5,
            n_conditions=2,
            ordering="alternating",
            seed=42,
        )
        assert len(seq) == 10
        assert set(seq) == {0, 1}

    def test_random_order_has_all_conditions(self):
        seq = generate_event_sequence(
            n_trials_per_condition=8,
            n_conditions=3,
            ordering="random",
            seed=42,
        )
        assert len(seq) == 24
        assert set(seq) == {0, 1, 2}

    def test_balanced_counts(self):
        seq = generate_event_sequence(
            n_trials_per_condition=10,
            n_conditions=4,
            ordering="random",
            seed=0,
        )
        for c in range(4):
            assert np.sum(seq == c) == 10


class TestISIConstraints:
    def test_basic_creation(self):
        c = ISIConstraints(min_isi=2.0, max_isi=8.0, mean_isi=4.0, tr=1.0)
        assert c.min_isi == 2.0
        assert c.max_isi == 8.0
        assert c.mean_isi == 4.0


class TestGenerateISISequence:
    def test_exponential_distribution(self):
        constraints = ISIConstraints(min_isi=2.0, max_isi=10.0, mean_isi=4.0, tr=1.0)
        isis = generate_isi_sequence(
            n_events=50,
            isi_constraints=constraints,
            distribution="exponential",
            seed=42,
        )
        # ISIs are between events, so n_events-1 intervals
        assert len(isis) >= 49
        assert isis.min() >= 2.0
        assert isis.max() <= 10.0

    def test_uniform_distribution(self):
        constraints = ISIConstraints(min_isi=2.0, max_isi=8.0, mean_isi=5.0, tr=1.0)
        isis = generate_isi_sequence(
            n_events=30,
            isi_constraints=constraints,
            distribution="uniform",
            seed=42,
        )
        assert len(isis) >= 29
        assert isis.min() >= 2.0
        assert isis.max() <= 8.0


class TestCreateOnsetMatrix:
    def test_basic_onset_matrix(self):
        seq = generate_event_sequence(
            n_trials_per_condition=5,
            n_conditions=2,
            ordering="alternating",
            seed=42,
        )
        constraints = ISIConstraints(min_isi=2.0, max_isi=8.0, mean_isi=4.0, tr=1.0)
        isis = generate_isi_sequence(
            n_events=10,
            isi_constraints=constraints,
            distribution="exponential",
            seed=42,
        )
        onsets = create_onset_matrix(
            event_sequence=seq,
            isis=isis,
            duration=100.0,
            tr=1.0,
            n_conditions=2,
        )
        assert onsets.ndim == 2
        # Should have columns for conditions
        assert onsets.shape[1] == 2


class TestEstimateAR1Coefficient:
    def test_known_autocorrelation(self):
        """AR(1) coefficient estimation on synthetic data."""
        rng = np.random.default_rng(42)
        residuals = rng.standard_normal(500)
        rho_true = 0.4
        for i in range(1, len(residuals)):
            residuals[i] = rho_true * residuals[i - 1] + np.sqrt(1 - rho_true**2) * residuals[i]
        rho_est = estimate_ar1_coefficient(residuals)
        assert abs(rho_est - rho_true) < 0.1
