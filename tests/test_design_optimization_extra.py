import numpy as np
import pytest
import torch

from fastfuncstuff.design.matrices import (
    basis_tent,
    convolve_design_hrf,
    convolve_hrf,
    generate_random_onsets,
    is_tr_locked,
    make_fir_design,
    make_penalty_matrix,
    make_singletrialdesign,
)
from fastfuncstuff.design.optimization import (
    ISIConstraints,
    DesignCandidate,
    compare_designs_summary,
    create_onset_matrix,
    find_optimal_designs,
    generate_event_sequence,
    generate_isi_sequence,
)

DEVICE = torch.device("cpu")


class TestGenerateEventSequence:
    def test_generate_event_sequence_random(self):
        n_trials = 10
        n_conditions = 3
        seq = generate_event_sequence(n_trials, n_conditions, ordering="random", seed=42)
        assert len(seq) == n_trials * n_conditions
        assert seq.min() >= 0
        assert seq.max() < n_conditions
        for c in range(n_conditions):
            assert np.sum(seq == c) == n_trials

    def test_generate_event_sequence_blocked(self):
        n_trials = 5
        n_conditions = 2
        seq = generate_event_sequence(n_trials, n_conditions, ordering="blocked", seed=0)
        assert len(seq) == n_trials * n_conditions
        changes = np.where(np.diff(seq) != 0)[0]
        assert len(changes) == n_conditions - 1
        for i in range(len(changes) - 1):
            block_len = changes[i + 1] - changes[i]
            assert block_len == n_trials

    def test_generate_event_sequence_alternating(self):
        n_trials = 4
        n_conditions = 2
        seq = generate_event_sequence(n_trials, n_conditions, ordering="alternating")
        expected = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        np.testing.assert_array_equal(seq, expected)


class TestGenerateISISequence:
    def test_generate_isi_sequence_length(self):
        n_events = 20
        constraints = ISIConstraints(min_isi=2.0, max_isi=6.0, mean_isi=3.5, tr=1.0)
        isis = generate_isi_sequence(n_events, constraints, distribution="exponential", seed=42)
        assert len(isis) == n_events - 1

    def test_generate_isi_sequence_range(self):
        n_events = 15
        constraints = ISIConstraints(min_isi=2.0, max_isi=6.0, mean_isi=4.0, tr=1.0)
        isis = generate_isi_sequence(n_events, constraints, distribution="uniform", seed=0)
        assert isis.min() >= 2.0
        assert isis.max() <= 6.0


class TestCreateOnsetMatrix:
    def test_create_onset_matrix_shape(self):
        event_seq = np.array([0, 1, 0, 1, 0, 1])
        isis = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        onsets = create_onset_matrix(event_seq, isis, duration=20.0, tr=1.0, n_conditions=2)
        assert onsets.shape == (20, 2)

    def test_create_onset_matrix_sum(self):
        event_seq = np.array([0, 1, 0, 1, 0])
        isis = np.array([4.0, 4.0, 4.0, 4.0])
        onsets = create_onset_matrix(event_seq, isis, duration=20.0, tr=1.0, n_conditions=2)
        assert onsets[:, 0].sum() == 3
        assert onsets[:, 1].sum() == 2


class TestFindOptimalDesigns:
    def test_find_optimal_designs(self):
        candidates = []
        for i in range(5):
            onsets = torch.zeros(20, 2)
            isis = np.array([2.0, 3.0])
            dm = torch.randn(20, 2)
            score_power = float(i) / 4.0
            score_eff = float(4 - i) / 4.0
            c = DesignCandidate(
                onsets=onsets,
                isis=isis,
                design_matrix=dm,
                metrics={"detection_power": score_power, "estimation_efficiency": score_eff},
                metadata={"ordering": "random", "distribution": "exponential"},
            )
            candidates.append(c)
        result = find_optimal_designs(candidates, objective="power", top_k=3)
        assert len(result) <= 3
        assert all(isinstance(t, tuple) and len(t) == 3 for t in result)
        scores = [t[2] for t in result]
        assert scores == sorted(scores, reverse=True)


class TestCompareDesignsSummary:
    def test_compare_designs_summary(self):
        candidates = []
        for i in range(3):
            onsets = torch.zeros(20, 2)
            isis = np.array([2.0, 3.0])
            dm = torch.randn(20, 2)
            c = DesignCandidate(
                onsets=onsets,
                isis=isis,
                design_matrix=dm,
                metrics={
                    "detection_power": float(i) * 0.1,
                    "estimation_efficiency": float(i) * 0.2,
                },
                metadata={"ordering": "random", "distribution": "exponential", "n_conditions": 2},
            )
            candidates.append(c)
        summary = compare_designs_summary(candidates, top_k=3)
        assert isinstance(summary, str)
        assert len(summary) > 0


class TestBasisTent:
    def test_basis_tent_peak(self):
        x = torch.tensor([5.0])
        val = basis_tent(x, bot=0.0, mid=5.0, top=10.0)
        assert val.item() == pytest.approx(1.0)

    def test_basis_tent_edges(self):
        x = torch.tensor([0.0, 10.0])
        val = basis_tent(x, bot=0.0, mid=5.0, top=10.0)
        assert val[0].item() == pytest.approx(0.0)
        assert val[1].item() == pytest.approx(0.0)


class TestIsTrLocked:
    def test_is_tr_locked_true(self):
        assert is_tr_locked([0, 2.0, 4.0, 6.0], tr=2.0)

    def test_is_tr_locked_false(self):
        assert not is_tr_locked([0.5, 2.3, 4.7], tr=2.0)


class TestMakeSingleTrialDesign:
    def test_make_singletrialdesign_shape(self):
        onsets = torch.zeros(50, 2)
        for i in range(5):
            onsets[i * 10, 0] = 1
            onsets[i * 10 + 5, 1] = 1
        design, labels = make_singletrialdesign(onsets, device=DEVICE)
        assert design.shape == (50, 10)

    def test_make_singletrialdesign_labels(self):
        onsets = torch.zeros(50, 2)
        onsets[10, 0] = 1
        onsets[20, 1] = 1
        onsets[30, 0] = 1
        onsets[40, 1] = 1
        _, labels = make_singletrialdesign(onsets, device=DEVICE)
        np.testing.assert_array_equal(labels.numpy(), [0, 1, 0, 1])


class TestMakePenaltyMatrix:
    def test_make_penalty_matrix_shape(self):
        n_basis = 10
        D = make_penalty_matrix(n_basis, order=2)
        assert D.shape == (n_basis - 2, n_basis)

    def test_make_penalty_matrix_symmetric(self):
        D = make_penalty_matrix(10, order=2)
        DtD = D.T @ D
        np.testing.assert_array_almost_equal(DtD, DtD.T)


class TestGenerateRandomOnsets:
    def test_generate_random_onsets_shape(self):
        onsets = generate_random_onsets(
            n_timepoints=100, n_conditions=2, isi_mean=4.0, device=DEVICE
        )
        assert onsets.shape[0] == 100
        assert onsets.shape[1] == 2

    def test_generate_random_onsets_binary(self):
        onsets = generate_random_onsets(
            n_timepoints=100, n_conditions=3, isi_mean=5.0, device=DEVICE
        )
        unique_vals = torch.unique(onsets)
        assert all(v in {0.0, 1.0} for v in unique_vals.tolist())


class TestMakeFirDesign:
    def test_make_fir_design_shape(self):
        onsets = torch.zeros(50, 2)
        onsets[10, 0] = 1
        onsets[20, 1] = 1
        design = make_fir_design(onsets, n_lags=10, n_timepoints=50, device=DEVICE)
        assert design.shape == (50, 20)


class TestConvolveHrf:
    def test_convolve_hrf_shape(self):
        onsets = torch.zeros(50, 2)
        onsets[10, 0] = 1
        onsets[20, 1] = 1
        hrf = torch.zeros(20)
        hrf[5] = 1.0
        result = convolve_hrf(onsets, hrf, n_timepoints=50, device=DEVICE)
        assert result.shape == (50, 2)

    def test_convolve_hrf_nonnegative(self):
        onsets = torch.zeros(50, 3)
        onsets[5, 0] = 1
        onsets[15, 1] = 1
        onsets[25, 2] = 1
        hrf = torch.exp(-0.1 * torch.arange(30, dtype=torch.float32))
        result = convolve_hrf(onsets, hrf, n_timepoints=50, device=DEVICE)
        assert (result >= 0).all()


class TestConvolveDesignHrf:
    def test_convolve_design_hrf_shape(self):
        design = torch.zeros(50, 1)
        design[10, 0] = 1
        design[30, 0] = 1
        hrf = torch.exp(-0.1 * torch.arange(20, dtype=torch.float32))
        result = convolve_design_hrf(design, hrf, device=DEVICE)
        assert result.shape == design.shape
