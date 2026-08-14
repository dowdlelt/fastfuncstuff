"""Tests for -drop_first/-drop_last: the TR trim and the event shift that follows."""

import numpy as np
import pytest
import torch

from fastfuncstuff.design.builder import create_onset_matrix_microtime
from fastfuncstuff.design.trim import (
    TrimSpec,
    shift_onsets_for_trim,
    trim_run_series,
)


class TestTrimSpec:
    def test_shift_is_drop_first_times_tr(self):
        assert TrimSpec(drop_first=5, tr=2.0).shift_sec == pytest.approx(10.0)

    def test_drop_last_does_not_shift(self):
        # The run's time origin does not move when trailing TRs go.
        assert TrimSpec(drop_last=5, tr=2.0).shift_sec == 0.0

    def test_negative_counts_rejected(self):
        with pytest.raises(ValueError):
            TrimSpec(drop_first=-1)

    def test_trimming_everything_is_an_error(self):
        with pytest.raises(ValueError, match="removes all"):
            TrimSpec(drop_first=50, drop_last=50).trimmed_length(100)

    def test_trimmed_length(self):
        assert TrimSpec(drop_first=4, drop_last=2).trimmed_length(100) == 94

    def test_inactive_spec_needs_no_tr(self):
        assert TrimSpec().shift_sec == 0.0
        assert not TrimSpec().active


class TestShiftOnsets:
    def test_onsets_shift_back_by_dropped_seconds(self):
        onsets = [[np.array([20.0, 40.0, 60.0])]]
        out, rep = shift_onsets_for_trim(
            onsets, [2.0], ["stim"], [190.0], TrimSpec(drop_first=5, tr=2.0)
        )
        np.testing.assert_allclose(out[0][0], [10.0, 30.0, 50.0])
        assert rep.shift_sec == pytest.approx(10.0)
        assert rep.n_shifted == 3

    def test_event_straddling_scan_start_is_kept_negative(self):
        # Onset 8s, 10s dropped, 6s duration: began 2s before the retained
        # window but is still running 4s into it. Real signal -- keep it.
        onsets = [[np.array([8.0])]]
        out, rep = shift_onsets_for_trim(
            onsets, [6.0], ["stim"], [190.0], TrimSpec(drop_first=5, tr=2.0)
        )
        np.testing.assert_allclose(out[0][0], [-2.0])
        assert rep.n_straddling == 1
        assert rep.straddle_conditions == ["stim"]

    def test_event_finishing_before_window_is_dropped(self):
        onsets = [[np.array([2.0])]]
        out, rep = shift_onsets_for_trim(
            onsets, [4.0], ["stim"], [190.0], TrimSpec(drop_first=5, tr=2.0)
        )
        assert out[0][0].size == 0
        assert rep.n_before == 1
        assert rep.n_straddling == 0

    def test_event_ending_exactly_at_window_start_is_dropped(self):
        # Ends at t=0 of the retained window: contributes nothing.
        onsets = [[np.array([6.0])]]
        out, rep = shift_onsets_for_trim(
            onsets, [4.0], ["stim"], [190.0], TrimSpec(drop_first=5, tr=2.0)
        )
        assert out[0][0].size == 0
        assert rep.n_before == 1

    def test_drop_last_strands_late_events(self):
        # 100 TRs @2s = 200s; drop_last 10 leaves 180s. The 190s event is gone.
        onsets = [[np.array([100.0, 190.0])]]
        out, rep = shift_onsets_for_trim(
            onsets, [2.0], ["stim"], [180.0], TrimSpec(drop_last=10, tr=2.0)
        )
        np.testing.assert_allclose(out[0][0], [100.0])
        assert rep.n_after == 1
        assert rep.shift_sec == 0.0

    def test_per_run_lengths_are_respected(self):
        onsets = [[np.array([100.0]), np.array([100.0])]]
        out, _ = shift_onsets_for_trim(
            onsets, [2.0], ["stim"], [90.0, 190.0], TrimSpec(drop_last=5, tr=2.0)
        )
        assert out[0][0].size == 0  # past the short run's end
        assert out[0][1].size == 1

    def test_empty_run_survives(self):
        onsets = [[np.array([]), np.array([10.0])]]
        out, _ = shift_onsets_for_trim(
            onsets, [1.0], ["stim"], [100.0, 100.0], TrimSpec(drop_first=2, tr=1.0)
        )
        assert out[0][0].size == 0
        np.testing.assert_allclose(out[0][1], [8.0])

    def test_report_lines_mention_the_shift(self):
        _, rep = shift_onsets_for_trim(
            [[np.array([20.0])]], [2.0], ["stim"], [190.0], TrimSpec(drop_first=5, tr=2.0)
        )
        assert "shifted back by 10.000s" in rep.lines()[0]


class TestTrimRunSeries:
    def test_untrimmed_file_is_trimmed(self):
        arr = np.arange(100).reshape(100, 1)
        out = trim_run_series(arr, 94, TrimSpec(drop_first=4, drop_last=2))
        assert out.shape[0] == 94
        assert out[0, 0] == 4 and out[-1, 0] == 97

    def test_already_trimmed_file_passes_through(self):
        arr = np.arange(94).reshape(94, 1)
        out = trim_run_series(arr, 94, TrimSpec(drop_first=4, drop_last=2))
        assert out.shape[0] == 94
        assert out[0, 0] == 0

    def test_ambiguous_length_is_an_error(self):
        arr = np.arange(97).reshape(97, 1)
        with pytest.raises(ValueError, match="rows"):
            trim_run_series(arr, 94, TrimSpec(drop_first=4, drop_last=2), path="mot.1D")


class TestOnsetMatrixClamping:
    """create_onset_matrix_microtime must respect run boundaries in both directions."""

    def _matrix(self, onsets, duration, run_starts, n_timepoints, tr=1.0, dt=0.1):
        return create_onset_matrix_microtime(
            all_onsets=onsets,
            run_starts=run_starts,
            tr=tr,
            n_timepoints=n_timepoints,
            microtime_dt=dt,
            stim_durations=[duration],
            device=torch.device("cpu"),
        )

    def test_negative_onset_is_truncated_not_dropped(self):
        # Onset -2s, duration 5s, run starts at t=0: 3s of boxcar should land.
        m = self._matrix([[np.array([-2.0])]], 5.0, [0], 20)
        col = m[:, 0]
        assert col[:30].sum() == 30  # first 3s (30 bins) painted
        assert col[30:].sum() == 0

    def test_negative_onset_in_run_two_stays_in_run_two(self):
        # The bug this guards: a negative onset in run 1 used to paint into run 0.
        m = self._matrix([[np.array([]), np.array([-2.0])]], 5.0, [0, 10], 20)
        col = m[:, 0]
        assert col[:100].sum() == 0  # run 0 untouched
        assert col[100:130].sum() == 30  # painted from run 1's start

    def test_long_event_does_not_bleed_into_next_run(self):
        # Onset 9s into a 10s run with a 5s duration: only 1s belongs to this run.
        m = self._matrix([[np.array([9.0]), np.array([])]], 5.0, [0, 10], 20)
        col = m[:, 0]
        assert col[90:100].sum() == 10
        assert col[100:].sum() == 0

    def test_event_entirely_before_run_paints_nothing(self):
        m = self._matrix([[np.array([-10.0])]], 2.0, [0], 20)
        assert m.sum() == 0

    def test_ordinary_onset_is_unchanged(self):
        m = self._matrix([[np.array([3.0])]], 2.0, [0], 20)
        col = m[:, 0]
        assert col[30:50].sum() == 20
        assert col.sum() == 20


class TestTrimShiftEndToEnd:
    def test_shifted_design_matches_a_natively_short_design(self):
        """Trimming data + shifting timing == having acquired the short run.

        This is the property the whole feature rests on: after -drop_first the
        design must be identical to one built from a run that started later.
        """
        tr, dt = 2.0, 0.1
        drop = 5
        onsets_full = [[np.array([20.0, 40.0, 60.0])]]

        # Reference: build on the full 100-TR run, then cut the leading TRs off
        # the resulting microtime matrix.
        full = create_onset_matrix_microtime(
            all_onsets=onsets_full,
            run_starts=[0],
            tr=tr,
            n_timepoints=100,
            microtime_dt=dt,
            stim_durations=[4.0],
            device=torch.device("cpu"),
        )
        bins_per_tr = int(round(tr / dt))
        reference = full[drop * bins_per_tr :]

        # Feature path: shift the timing, build on the trimmed run.
        spec = TrimSpec(drop_first=drop, tr=tr)
        shifted, _ = shift_onsets_for_trim(onsets_full, [4.0], ["stim"], [(100 - drop) * tr], spec)
        trimmed = create_onset_matrix_microtime(
            all_onsets=shifted,
            run_starts=[0],
            tr=tr,
            n_timepoints=100 - drop,
            microtime_dt=dt,
            stim_durations=[4.0],
            device=torch.device("cpu"),
        )

        assert torch.equal(trimmed, reference)


class TestLoaderTrim:
    """The data trim itself, through the shared loader."""

    def _write_runs(self, tmp_path, n_runs=2, nt=20):
        import nibabel as nib

        paths = []
        for r in range(n_runs):
            # Voxel value encodes (run, timepoint) so trimming is verifiable.
            data = np.zeros((2, 2, 2, nt), dtype=np.float32)
            for t in range(nt):
                data[..., t] = r * 100 + t
            p = tmp_path / f"run{r}.nii.gz"
            nib.save(nib.Nifti1Image(data, np.eye(4)), str(p))
            paths.append(str(p))
        return paths

    def test_runs_are_trimmed_at_both_ends(self, tmp_path):
        from fastfuncstuff.io.afni import load_and_concatenate_runs

        paths = self._write_runs(tmp_path, n_runs=2, nt=20)
        data, run_starts = load_and_concatenate_runs(paths, drop_first=4, drop_last=2)

        assert data.shape[1] == 2 * 14
        assert run_starts == [0, 14]
        # Run 0 now starts at its old t=4 and ends at t=17.
        assert data[0, 0].item() == 4
        assert data[0, 13].item() == 17
        # Run 1 likewise, offset by its 100.
        assert data[0, 14].item() == 104
        assert data[0, 27].item() == 117

    def test_untrimmed_load_is_unchanged(self, tmp_path):
        from fastfuncstuff.io.afni import load_and_concatenate_runs

        paths = self._write_runs(tmp_path, n_runs=2, nt=20)
        data, run_starts = load_and_concatenate_runs(paths)
        assert data.shape[1] == 40
        assert run_starts == [0, 20]
        assert data[0, 0].item() == 0

    def test_over_trimming_a_run_is_an_error(self, tmp_path):
        from fastfuncstuff.io.afni import load_and_concatenate_runs

        paths = self._write_runs(tmp_path, n_runs=1, nt=10)
        with pytest.raises(ValueError, match="removes all"):
            load_and_concatenate_runs(paths, drop_first=8, drop_last=5)


class TestNuisanceTrim:
    """Motion/ortvec files are produced on the untrimmed run -- both must work."""

    def test_untrimmed_per_run_ortvec_is_trimmed(self, tmp_path):
        from fastfuncstuff.cli_utils import make_nuisance_block_from_per_run_file

        f = tmp_path / "mot.1D"
        np.savetxt(f, np.arange(20, dtype=float).reshape(20, 1))
        blk = make_nuisance_block_from_per_run_file(
            f, "mot", 1, run_starts=[0], n_timepoints=14, trim=TrimSpec(drop_first=4, drop_last=2)
        )
        assert blk.per_run[0].shape[0] == 14
        assert blk.per_run[0][0, 0] == 4

    def test_already_trimmed_per_run_ortvec_passes(self, tmp_path):
        from fastfuncstuff.cli_utils import make_nuisance_block_from_per_run_file

        f = tmp_path / "mot.1D"
        np.savetxt(f, np.arange(14, dtype=float).reshape(14, 1))
        blk = make_nuisance_block_from_per_run_file(
            f, "mot", 1, run_starts=[0], n_timepoints=14, trim=TrimSpec(drop_first=4, drop_last=2)
        )
        assert blk.per_run[0].shape[0] == 14
        assert blk.per_run[0][0, 0] == 0

    def test_untrimmed_concatenated_ortvec_is_trimmed_per_run(self, tmp_path):
        from fastfuncstuff.cli_utils import make_nuisance_block_from_full_length

        # Two 20-TR runs on disk; the design has two 14-TR runs.
        f = tmp_path / "all.1D"
        np.savetxt(f, np.concatenate([np.arange(20.0), 100 + np.arange(20.0)]).reshape(40, 1))
        blk = make_nuisance_block_from_full_length(
            f, "mot", run_starts=[0, 14], n_timepoints=28, trim=TrimSpec(drop_first=4, drop_last=2)
        )
        assert [b.shape[0] for b in blk.per_run] == [14, 14]
        assert blk.per_run[0][0, 0] == 4  # run 0's own leading TRs went
        assert blk.per_run[1][0, 0] == 104  # run 1 trimmed on its own grid


class TestCensorTrim:
    def test_untrimmed_censor_file_is_trimmed_per_run(self, tmp_path):
        from fastfuncstuff.io.afni import read_censor_1d

        # 2 runs x 20 TRs; censor TR 0 of each run (which -drop_first removes).
        vals = np.ones(40)
        vals[0] = 0
        vals[20] = 0
        vals[25] = 0  # survives: t=25 -> run 1, index 5 -> concat index 14+1
        f = tmp_path / "cen.1D"
        np.savetxt(f, vals)

        good = read_censor_1d(
            f, n_expected=28, run_lengths_tr=[14, 14], trim=TrimSpec(drop_first=4, drop_last=2)
        )
        assert len(good) == 27  # only the t=25 censor survives the trim
        assert 15 not in good  # run 1 (starts at 14) index 1 == old t=25


class TestSpecCompileTrim:
    """`ffs_reml -spec` compiles its xmat against the trimmed runs."""

    def test_events_shift_and_filter_on_the_retained_window(self):
        from fastfuncstuff.cli.design_spec import _shift_events_for_trim

        # onset 2s / dur 4s ends at 6s -> gone once 10s are dropped.
        # onset 8s / dur 6s straddles the new start -> kept, negative.
        # onset 20s / dur 4s -> plain shift to 10s.
        per_run = [[(2.0, 4.0), (8.0, 6.0), (20.0, 4.0), (195.0, 4.0)]]
        out = _shift_events_for_trim(per_run, TrimSpec(drop_first=5, tr=2.0), [180.0])
        assert out[0] == [(-2.0, 6.0), (10.0, 4.0)]

    def test_per_event_durations_are_respected(self):
        from fastfuncstuff.cli.design_spec import _shift_events_for_trim

        # Same onset, different durations: only the longer one still overlaps.
        per_run = [[(8.0, 1.0), (8.0, 6.0)]]
        out = _shift_events_for_trim(per_run, TrimSpec(drop_first=5, tr=2.0), [180.0])
        assert out[0] == [(-2.0, 6.0)]

    def test_inactive_trim_is_a_no_op(self):
        from fastfuncstuff.cli.design_spec import _shift_events_for_trim

        per_run = [[(2.0, 4.0), (20.0, 4.0)]]
        out = _shift_events_for_trim(per_run, TrimSpec(tr=2.0), [200.0])
        assert out[0] == per_run[0]


class TestAutoprocGlmDropFlags:
    def test_options_carry_the_drop_counts(self):
        from fastfuncstuff.autoproc.plan import Options

        opt = Options(glm_drop_first=8, glm_drop_last=2)
        assert (opt.glm_drop_first, opt.glm_drop_last) == (8, 2)

    def test_default_is_no_trim(self):
        from fastfuncstuff.autoproc.plan import Options

        opt = Options()
        assert (opt.glm_drop_first, opt.glm_drop_last) == (0, 0)

    def test_cli_parses_both_spellings(self):
        from fastfuncstuff.cli.autoproc import build_parser

        args, _ = build_parser().parse_known_args(
            ["-bids_dir", "/tmp", "-subject", "01", "-glm_drop_first", "8", "-glm-drop-last", "2"]
        )
        assert (args.glm_drop_first, args.glm_drop_last) == (8, 2)
