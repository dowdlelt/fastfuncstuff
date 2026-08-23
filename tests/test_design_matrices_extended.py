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
    bins_per_tr_exact,
    build_event_design_microtime,
    build_glm_design,
    commensurate_microtime_dt,
    convolve_design_hrf,
    generate_random_onsets,
    is_tr_locked,
    make_csplin_design,
    make_singletrialdesign,
    make_tent_design,
    onsets_to_tr_matrix,
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


class TestMicrotimeCommensurability:
    """TR=1.75 with a nominal 0.1s step gives round(17.5)=18 bins, an effective
    1.8s TR, and stimulus columns slide progressively earlier through the run."""

    @pytest.mark.parametrize("tr", [1.75, 3.542, 2.13, 0.372, 12.345678, 0.729, 1.0, 2.0, 1 / 3])
    def test_snapped_grid_divides_any_tr_exactly(self, tr):
        dt = commensurate_microtime_dt(tr, 0.1)
        bins = bins_per_tr_exact(tr, dt)
        assert bins >= 1
        # A TR boundary 2000 TRs in must still land on a sampled bin.
        assert abs(2000 * bins * dt - 2000 * tr) < 1e-6
        # Resolution stays near the request; commensurability is the priority.
        assert 0.05 <= dt <= 0.15 or bins == 1

    def test_incommensurate_grid_is_refused(self):
        with pytest.raises(ValueError, match="not commensurate"):
            bins_per_tr_exact(1.75, 0.1)

    def test_event_design_refuses_drifting_grid(self):
        hrf = torch.tensor([0.0, 1.0, 0.5])
        with pytest.raises(ValueError, match="not commensurate"):
            build_event_design_microtime(
                all_onsets=[[np.array([0.0])]],
                durations=[0.0],
                hrf_bases=hrf,
                n_timepoints_per_run=[20],
                tr=1.75,
                microtime_dt=0.1,
                device=DEVICE,
            )

    def test_late_events_do_not_drift_on_a_snapped_grid(self):
        """The regression: an event's response must peak the same distance
        after onset at the end of a run as at the start."""
        tr, n_tp = 1.75, 120
        dt = commensurate_microtime_dt(tr, 0.1)
        # Delta HRF: the column's nonzero TR marks exactly where the onset landed.
        hrf = torch.zeros(int(round(32.0 / dt)))
        hrf[0] = 1.0
        for k in (0, 30, 60, 90):
            design = build_event_design_microtime(
                all_onsets=[[np.array([k * tr])]],
                durations=[0.0],
                hrf_bases=hrf,
                n_timepoints_per_run=[n_tp],
                tr=tr,
                microtime_dt=dt,
                device=DEVICE,
            )
            assert isinstance(design, torch.Tensor)
            assert int(design[:, 0].argmax()) == k, f"event at TR {k} drifted"


class TestOnsetsToTrMatrix:
    def test_off_grid_events_are_rounded_not_dropped(self):
        """Strided decimation of a microtime matrix deleted these outright."""
        design, max_shift = onsets_to_tr_matrix(
            all_onsets=[[np.array([1.4, 5.4, 8.0])]],
            run_starts=[0],
            n_timepoints=20,
            tr=2.0,
            durations=[0.0],
            device=DEVICE,
        )
        assert design[:, 0].nonzero().flatten().tolist() == [1, 3, 4]
        assert max_shift == pytest.approx(0.6)

    def test_blocks_span_every_covered_tr_per_run(self):
        design, _ = onsets_to_tr_matrix(
            all_onsets=[[np.array([1.4]), np.array([3.0])]],
            run_starts=[0, 10],
            n_timepoints=20,
            tr=2.0,
            durations=[5.0],
            device=DEVICE,
        )
        assert design[:, 0].nonzero().flatten().tolist() == [1, 2, 3, 12, 13, 14]

    def test_events_beyond_the_run_are_still_dropped(self):
        design, _ = onsets_to_tr_matrix(
            all_onsets=[[np.array([2.0, 999.0])]],
            run_starts=[0],
            n_timepoints=10,
            tr=2.0,
            device=DEVICE,
        )
        assert design[:, 0].nonzero().flatten().tolist() == [1]


class TestJointBasisAnchorScaling:
    """A jointly fitted basis set shares one scale, set by the anchor (first)
    curve. Peak-normalising each basis separately rescales beta1 by
    peak(h)/peak(h') and every latency read off the ratio is wrong by that
    factor -- 2.56x for SPMG2's time derivative."""

    @staticmethod
    def _spmg_setup(n_basis):
        from fastfuncstuff.design.hrf import get_spm_hrf_with_derivatives

        bases = get_spm_hrf_with_derivatives(
            microtime_dt=0.05, hrf_duration=32.0, n_basis=n_basis, device=DEVICE
        )
        onsets = [[np.arange(5.0, 280.0, 23.0)]]
        return bases, onsets

    def test_anchor_keeps_unit_peak_amplitude(self):
        """beta0 must still carry AFNI's unit-peak amplitude convention."""
        bases, onsets = self._spmg_setup(3)
        design = build_event_design_microtime(
            onsets, [0.0], bases, [300], tr=1.0, microtime_dt=0.05, device=DEVICE
        )
        assert isinstance(design, torch.Tensor)
        # Sampled at TR, so the microtime peak is bracketed, not hit exactly.
        assert design[:, 0].max().item() == pytest.approx(1.0, abs=0.02)

    def test_derivatives_are_not_individually_peak_normalized(self):
        bases, onsets = self._spmg_setup(3)
        design = build_event_design_microtime(
            onsets, [0.0], bases, [300], tr=1.0, microtime_dt=0.05, device=DEVICE
        )
        assert isinstance(design, torch.Tensor)
        # The derivative columns are genuinely smaller than the canonical one.
        # Under the old per-basis scaling all three peaked at 1.0.
        for col in (1, 2):
            assert design[:, col].abs().max().item() < 0.9

    @pytest.mark.parametrize("true_delta", [0.0, 0.3, 0.6, -0.4])
    def test_beta_ratio_recovers_latency_in_seconds(self, true_delta):
        """The property the scaling exists to protect: b1/b0 is the shift, in
        seconds, with no calibration factor."""
        from fastfuncstuff.design.hrf import get_spmg1_hrf

        dt = 0.05
        bases, onsets = self._spmg_setup(2)
        design = build_event_design_microtime(
            onsets, [0.0], bases, [300], tr=1.0, microtime_dt=dt, device=DEVICE
        )
        assert isinstance(design, torch.Tensor)

        hrf = get_spmg1_hrf(
            microtime_dt=dt,
            stim_duration=0.0,
            hrf_duration=32.0,
            normalize_peak=True,
            device=DEVICE,
        )
        shift_bins = int(round(true_delta / dt))
        shifted = torch.roll(hrf, shift_bins)
        if shift_bins > 0:
            shifted[:shift_bins] = 0
        truth = build_event_design_microtime(
            onsets, [0.0], shifted, [300], tr=1.0, microtime_dt=dt, device=DEVICE
        )
        assert isinstance(truth, torch.Tensor)

        betas = torch.linalg.lstsq(design, truth[:, :1]).solution.flatten()
        assert betas[0].item() == pytest.approx(1.0, abs=0.05)
        assert (betas[1] / betas[0]).item() == pytest.approx(true_delta, abs=0.02)

    def test_library_curves_are_still_unit_peak_each(self):
        """A library is n *alternative* HRFs fitted one at a time, not a joint
        set, so each must keep its own unit peak."""
        # tr == microtime_dt samples every bin, so the column peak is the
        # kernel peak exactly -- this tests the scaling, not TR sampling.
        for row in self._spmg_setup(3)[0]:
            design = build_event_design_microtime(
                [[np.array([5.0])]], [0.0], row, [800], tr=0.05, microtime_dt=0.05, device=DEVICE
            )
            assert isinstance(design, torch.Tensor)
            assert design[:, 0].abs().max().item() == pytest.approx(1.0, abs=1e-5)
