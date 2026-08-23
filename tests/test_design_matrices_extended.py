"""
Tests for uncovered functions in fastfuncstuff/design/matrices.py:
- basis_csplin
- make_tent_design / make_csplin_design
- is_tr_locked
- make_singletrialdesign
- convolve_design_hrf
- build_glm_design (all modes)
- generate_random_onsets
"""

import numpy as np
import pytest
import torch

from fastfuncstuff.design.matrices import (
    basis_csplin,
    build_event_design_microtime,
    build_glm_design,
    convolve_design_hrf,
    generate_random_onsets,
    is_tr_locked,
    make_csplin_design,
    make_singletrialdesign,
    make_tent_design,
)

DEVICE = torch.device("cpu")


class TestEventDesignMicrotime:
    @pytest.mark.parametrize("onsets", [[2.0, 5.0], [2.0, 4.0]])
    def test_touching_and_overlapping_events_are_sums_of_trials(self, onsets):
        """Condition columns must be sums of original events for any HRF basis."""
        bases = torch.tensor(
            [
                [0.0, 0.4, 1.0, 0.5, 0.1],
                [0.0, 1.0, 0.0, -0.5, 0.0],
            ]
        )
        result = build_event_design_microtime(
            all_onsets=[[np.asarray(onsets)]],
            durations=[3.0],
            hrf_bases=bases,
            n_timepoints_per_run=[30],
            tr=1.0,
            microtime_dt=0.5,
            return_single_trials=True,
            device=DEVICE,
        )
        condition, trials, condition_ids, run_ids = result

        assert condition.shape == (30, 2)
        assert trials.shape == (30, 4)
        assert condition_ids.tolist() == [0, 0, 0, 0]
        assert run_ids.tolist() == [0, 0, 0, 0]
        torch.testing.assert_close(condition[:, 0], trials[:, 0] + trials[:, 2])
        torch.testing.assert_close(condition[:, 1], trials[:, 1] + trials[:, 3])

    def test_combined_condition_equals_separate_event_conditions(self):
        """An arbitrary library HRF obeys linear superposition across event labels."""
        hrf = torch.tensor([0.0, 0.25, 1.0, 0.7, 0.2, -0.1])
        design = build_event_design_microtime(
            all_onsets=[
                [np.array([2.0, 5.0])],
                [np.array([2.0])],
                [np.array([5.0])],
            ],
            durations=[3.0, 3.0, 3.0],
            hrf_bases=hrf,
            n_timepoints_per_run=[30],
            tr=1.0,
            microtime_dt=0.5,
            device=DEVICE,
        )
        assert isinstance(design, torch.Tensor)
        torch.testing.assert_close(design[:, 0], design[:, 1] + design[:, 2])


class TestBasisCsplin:
    def test_peak_at_center(self):
        """Cubic spline should peak at 1.0 at its center knot."""
        knots = [0.0, 1.0, 2.0, 3.0, 4.0]
        x = torch.tensor([2.0])
        val = basis_csplin(x, knots, idx=2)
        assert abs(val.item() - 1.0) < 0.01

    def test_zero_far_from_center(self):
        """Should be zero far from the center knot."""
        knots = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        x = torch.tensor([6.0])
        val = basis_csplin(x, knots, idx=1)
        assert abs(val.item()) < 1e-6

    def test_invalid_index(self):
        """Out-of-range index should return zeros."""
        knots = [0.0, 1.0, 2.0]
        x = torch.tensor([1.0])
        val = basis_csplin(x, knots, idx=5)
        assert val.item() == 0.0

    def test_single_knot(self):
        knots = [1.0]
        x = torch.tensor([1.0])
        val = basis_csplin(x, knots, idx=0)
        assert val.item() == 0.0  # Can't compute dx with single knot


class TestMakeTentDesign:
    def test_basic_tent(self):
        """TENT design matrix should have correct shape."""
        onsets = [np.array([2.0, 6.0, 10.0])]
        design = make_tent_design(
            onsets,
            bot=0.0,
            top=10.0,
            tr=2.0,
            n_timepoints=20,
            device=DEVICE,
        )
        # n_basis = round(10/2) + 1 = 6
        assert design.shape == (20, 6)
        assert design.sum() > 0  # Should have non-zero entries

    def test_tent_zero_edges(self):
        """TENTzero should have 2 fewer basis functions."""
        onsets = [np.array([2.0, 6.0])]
        design = make_tent_design(
            onsets,
            bot=0.0,
            top=10.0,
            tr=2.0,
            n_timepoints=20,
            n_basis=6,
            zero_edges=True,
            device=DEVICE,
        )
        assert design.shape == (20, 4)  # 6 - 2 = 4

    def test_bot_ge_top_raises(self):
        with pytest.raises(ValueError, match="bot.*must be < top"):
            make_tent_design(
                [np.array([1.0])], bot=10.0, top=5.0, tr=1.0, n_timepoints=20, device=DEVICE
            )

    def test_empty_onsets(self):
        """Empty onset list should return zero design."""
        design = make_tent_design(
            [np.array([])],
            bot=0.0,
            top=10.0,
            tr=2.0,
            n_timepoints=20,
            device=DEVICE,
        )
        assert design.sum().item() == 0.0

    def test_custom_n_basis(self):
        onsets = [np.array([3.0])]
        design = make_tent_design(
            onsets,
            bot=0.0,
            top=12.0,
            tr=2.0,
            n_timepoints=20,
            n_basis=7,
            device=DEVICE,
        )
        assert design.shape[1] == 7


class TestMakeCsplinDesign:
    def test_basic_csplin(self):
        """CSPLIN design should have correct shape."""
        onsets = [np.array([2.0, 8.0])]
        design = make_csplin_design(
            onsets,
            bot=0.0,
            top=12.0,
            tr=2.0,
            n_timepoints=20,
            n_basis=7,
            device=DEVICE,
        )
        assert design.shape == (20, 7)
        assert design.sum() > 0

    def test_csplin_zero_edges(self):
        onsets = [np.array([2.0])]
        design = make_csplin_design(
            onsets,
            bot=0.0,
            top=12.0,
            tr=2.0,
            n_timepoints=20,
            n_basis=7,
            zero_edges=True,
            device=DEVICE,
        )
        assert design.shape == (20, 5)  # 7 - 2

    def test_too_few_basis_raises(self):
        with pytest.raises(ValueError, match="n_basis must be >= 4"):
            make_csplin_design(
                [np.array([1.0])],
                bot=0.0,
                top=5.0,
                tr=1.0,
                n_timepoints=10,
                n_basis=3,
                device=DEVICE,
            )


class TestIsTrLocked:
    def test_perfect_locking(self):
        assert is_tr_locked([0.0, 2.0, 4.0, 6.0], tr=2.0)

    def test_small_jitter_within_threshold(self):
        assert is_tr_locked([0.0, 2.05, 4.0], tr=2.0, threshold=0.1)

    def test_non_locked(self):
        assert not is_tr_locked([0.5, 2.5, 4.5], tr=2.0)

    def test_numpy_input(self):
        assert is_tr_locked(np.array([0.0, 1.0, 2.0]), tr=1.0)


class TestMakeSingleTrialDesign:
    def test_basic(self):
        # 2 conditions, 3 trials each
        onsets = torch.zeros(20, 2)
        onsets[2, 0] = 1  # Trial 1, cond 0
        onsets[5, 1] = 1  # Trial 2, cond 1
        onsets[8, 0] = 1  # Trial 3, cond 0
        onsets[12, 1] = 1  # Trial 4, cond 1

        design, conditions = make_singletrialdesign(onsets, device=DEVICE)
        assert design.shape == (20, 4)
        assert conditions.shape == (4,)
        # Each column should have exactly one non-zero entry
        for col in range(4):
            assert (design[:, col] > 0).sum() == 1

    def test_1d_input(self):
        """Single condition as 1D vector."""
        onsets = torch.zeros(10)
        onsets[2] = 1
        onsets[7] = 1
        design, conditions = make_singletrialdesign(onsets, device=DEVICE)
        assert design.shape == (10, 2)
        assert (conditions == 0).all()  # All from condition 0

    def test_sorted_by_time(self):
        """Trials should be sorted by onset time."""
        onsets = torch.zeros(20, 2)
        onsets[10, 1] = 1  # Later time, condition 1
        onsets[3, 0] = 1  # Earlier time, condition 0
        design, conditions = make_singletrialdesign(onsets, device=DEVICE)
        # First trial should be at time 3 (condition 0)
        assert conditions[0].item() == 0
        assert conditions[1].item() == 1


class TestConvolveDesignHrf:
    def test_basic_convolution(self):
        """Convolved design should spread impulses over time."""
        design = torch.zeros(50, 2, device=DEVICE)
        design[5, 0] = 1.0
        design[15, 1] = 1.0
        hrf = torch.tensor([0.0, 0.5, 1.0, 0.8, 0.3, 0.1], device=DEVICE)

        result = convolve_design_hrf(design, hrf, device=DEVICE)
        assert result.shape == (50, 2)
        # After convolution, energy should spread beyond the impulse
        assert (result[:, 0] != 0).sum() > 1
        assert (result[:, 1] != 0).sum() > 1

    def test_preserves_shape(self):
        design = torch.randn(100, 5, device=DEVICE)
        hrf = torch.randn(15, device=DEVICE)
        result = convolve_design_hrf(design, hrf, device=DEVICE)
        assert result.shape == (100, 5)


class TestBuildGlmDesign:
    @pytest.fixture
    def simple_onsets(self):
        onsets = torch.zeros(50, 2, device=DEVICE)
        onsets[5, 0] = 1
        onsets[15, 1] = 1
        onsets[25, 0] = 1
        onsets[35, 1] = 1
        return onsets

    @pytest.fixture
    def hrf(self):
        return torch.tensor([0.0, 0.2, 0.8, 1.0, 0.7, 0.3, 0.1, 0.0], device=DEVICE)

    def test_assumed_mode(self, simple_onsets, hrf):
        design = build_glm_design(simple_onsets, hrf=hrf, mode="assumed", device=DEVICE)
        assert design.shape[0] == 50
        assert design.shape[1] == 2

    def test_fir_mode(self, simple_onsets):
        design = build_glm_design(simple_onsets, mode="fir", n_fir_lags=10, device=DEVICE)
        assert design.shape[0] == 50
        # FIR: n_conditions * n_lags
        assert design.shape[1] == 20

    def test_onoff_mode(self, simple_onsets, hrf):
        design = build_glm_design(simple_onsets, hrf=hrf, mode="onoff", device=DEVICE)
        assert design.shape[0] == 50
        assert design.shape[1] == 1  # summed across conditions

    def test_assumed_without_hrf_raises(self, simple_onsets):
        with pytest.raises(ValueError, match="HRF must be provided"):
            build_glm_design(simple_onsets, mode="assumed", device=DEVICE)

    def test_invalid_mode_raises(self, simple_onsets):
        with pytest.raises(ValueError, match="Unknown mode"):
            build_glm_design(simple_onsets, mode="badmode", device=DEVICE)

    def test_multi_run(self, simple_onsets, hrf):
        """Multiple runs should return a list."""
        runs = [simple_onsets, simple_onsets]
        designs = build_glm_design(runs, hrf=hrf, mode="assumed", device=DEVICE)
        assert isinstance(designs, list)
        assert len(designs) == 2

    def test_single_trial_assumed(self, simple_onsets, hrf):
        design = build_glm_design(
            simple_onsets,
            hrf=hrf,
            mode="assumed",
            single_trial=True,
            device=DEVICE,
        )
        # 4 trials total
        assert design.shape == (50, 4)


class TestGenerateRandomOnsets:
    def test_basic_generation(self):
        onsets = generate_random_onsets(
            n_timepoints=100,
            n_conditions=2,
            isi_mean=5.0,
            tr=2.0,
            device=DEVICE,
        )
        assert onsets.shape[0] == 100
        assert onsets.shape[1] == 2
        # Should have some non-zero entries
        assert onsets.sum() > 0

    def test_alternating_conditions(self):
        onsets = generate_random_onsets(
            n_timepoints=200,
            n_conditions=3,
            isi_mean=4.0,
            tr=1.0,
            alternate_conditions=True,
            device=DEVICE,
        )
        assert onsets.shape[1] == 3

    def test_random_conditions(self):
        onsets = generate_random_onsets(
            n_timepoints=200,
            n_conditions=2,
            isi_mean=4.0,
            tr=1.0,
            alternate_conditions=False,
            device=DEVICE,
        )
        assert onsets.shape[1] == 2
