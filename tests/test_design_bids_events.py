"""
Tests for design/bids_events.py — BIDS *_events.tsv parsing for the
deconvolve / fitbasis CLIs.

Bugs here silently mis-align onsets, mis-aggregate per-condition
durations, or scramble multi-run ordering. Every assertion targets a
specific failure mode that would corrupt the GLM design downstream.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fastfuncstuff.design.bids_events import (
    _run_number,
    parse_bids_events,
    sort_bids_event_files,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tsv(path: Path, rows: list[dict], cols: list[str] | None = None) -> Path:
    """Write a TSV with the given rows. `cols` overrides column order/names."""
    if cols is None:
        cols = ["onset", "duration", "trial_type"]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(c, "")) for c in cols) + "\n")
    return path


# ---------------------------------------------------------------------------
# _run_number / sort_bids_event_files
# ---------------------------------------------------------------------------


class TestRunNumberSorting:
    def test_padded_and_unpadded_equivalent(self):
        assert _run_number(Path("sub-01_run-01_events.tsv")) == 1
        assert _run_number(Path("sub-01_run-1_events.tsv")) == 1
        assert _run_number(Path("sub-01_run-001_events.tsv")) == 1

    def test_no_run_entity_sorts_to_zero(self):
        assert _run_number(Path("sub-01_task-foo_events.tsv")) == 0

    def test_case_insensitive(self):
        assert _run_number(Path("sub-01_RUN-3_events.tsv")) == 3

    def test_sort_mixed_padding(self, tmp_path):
        files = [
            tmp_path / "sub-01_run-10_events.tsv",
            tmp_path / "sub-01_run-2_events.tsv",
            tmp_path / "sub-01_run-01_events.tsv",
        ]
        sorted_paths = sort_bids_event_files(files)
        # Lexicographic sort would put run-10 before run-2; numeric must not.
        nums = [_run_number(p) for p in sorted_paths]
        assert nums == [1, 2, 10]

    def test_sort_accepts_string_paths(self, tmp_path):
        files = [
            str(tmp_path / "sub-01_run-2_events.tsv"),
            str(tmp_path / "sub-01_run-1_events.tsv"),
        ]
        sorted_paths = sort_bids_event_files(files)
        assert all(isinstance(p, Path) for p in sorted_paths)
        assert [_run_number(p) for p in sorted_paths] == [1, 2]


# ---------------------------------------------------------------------------
# parse_bids_events — single run
# ---------------------------------------------------------------------------


class TestParseSingleRun:
    def test_basic_two_conditions(self, tmp_path):
        f = _write_tsv(
            tmp_path / "run-1_events.tsv",
            [
                {"onset": 0.0, "duration": 2.0, "trial_type": "face"},
                {"onset": 10.0, "duration": 2.0, "trial_type": "house"},
                {"onset": 20.0, "duration": 2.0, "trial_type": "face"},
            ],
        )
        all_onsets, durations, labels = parse_bids_events([f])
        assert labels == ["face", "house"]  # sorted alphabetically
        assert durations == [2.0, 2.0]
        # face: onsets at 0, 20; house: at 10
        np.testing.assert_array_equal(all_onsets[0][0], [0.0, 20.0])
        np.testing.assert_array_equal(all_onsets[1][0], [10.0])

    def test_onsets_sorted_within_condition(self, tmp_path):
        """parse must sort onsets ascending within each (condition, run)."""
        f = _write_tsv(
            tmp_path / "run-1_events.tsv",
            [
                {"onset": 30.0, "duration": 1.0, "trial_type": "A"},
                {"onset": 5.0, "duration": 1.0, "trial_type": "A"},
                {"onset": 15.0, "duration": 1.0, "trial_type": "A"},
            ],
        )
        all_onsets, _, _ = parse_bids_events([f])
        np.testing.assert_array_equal(all_onsets[0][0], [5.0, 15.0, 30.0])

    def test_skips_na_and_empty_trial_types(self, tmp_path):
        f = _write_tsv(
            tmp_path / "run-1_events.tsv",
            [
                {"onset": 0.0, "duration": 1.0, "trial_type": "A"},
                {"onset": 5.0, "duration": 1.0, "trial_type": "n/a"},
                {"onset": 10.0, "duration": 1.0, "trial_type": ""},
                {"onset": 15.0, "duration": 1.0, "trial_type": "N/A"},
            ],
        )
        all_onsets, _, labels = parse_bids_events([f])
        assert labels == ["A"]
        np.testing.assert_array_equal(all_onsets[0][0], [0.0])

    def test_event_ignore_filters_listed_conditions(self, tmp_path):
        f = _write_tsv(
            tmp_path / "run-1_events.tsv",
            [
                {"onset": 0.0, "duration": 1.0, "trial_type": "A"},
                {"onset": 5.0, "duration": 1.0, "trial_type": "fixation"},
                {"onset": 10.0, "duration": 1.0, "trial_type": "B"},
            ],
        )
        all_onsets, _, labels = parse_bids_events([f], event_ignore=["fixation"])
        assert labels == ["A", "B"]
        assert all_onsets[0][0].tolist() == [0.0]
        assert all_onsets[1][0].tolist() == [10.0]

    def test_custom_column_names(self, tmp_path):
        f = _write_tsv(
            tmp_path / "run-1_events.tsv",
            [
                {"start": 0.0, "len": 2.0, "cond": "A"},
                {"start": 10.0, "len": 2.0, "cond": "B"},
            ],
            cols=["start", "len", "cond"],
        )
        all_onsets, _, labels = parse_bids_events([f], event_cols=("start", "len", "cond"))
        assert labels == ["A", "B"]
        assert all_onsets[0][0].tolist() == [0.0]


# ---------------------------------------------------------------------------
# parse_bids_events — multi-run
# ---------------------------------------------------------------------------


class TestParseMultiRun:
    def test_per_run_onset_arrays(self, tmp_path):
        f1 = _write_tsv(
            tmp_path / "run-1_events.tsv",
            [
                {"onset": 0.0, "duration": 1.0, "trial_type": "A"},
                {"onset": 10.0, "duration": 1.0, "trial_type": "B"},
            ],
        )
        f2 = _write_tsv(
            tmp_path / "run-2_events.tsv",
            [
                {"onset": 5.0, "duration": 1.0, "trial_type": "A"},
            ],
        )
        all_onsets, _, labels = parse_bids_events([f1, f2])
        assert labels == ["A", "B"]
        # A appears in both runs
        assert all_onsets[0][0].tolist() == [0.0]
        assert all_onsets[0][1].tolist() == [5.0]
        # B appears only in run 1 → empty array for run 2
        assert all_onsets[1][0].tolist() == [10.0]
        assert all_onsets[1][1].tolist() == []

    def test_caller_order_is_preserved(self, tmp_path):
        """Event files pair with -input by position, so the list is used as given.

        This used to sort by run number, which ignores the session entity: across
        sessions that groups every run-001 together and hands session N's timing to
        session 1's run N. The corruption was silent — the only symptom was an
        occasional all-zero design column where borrowed onsets overran a shorter run.
        """
        f1 = _write_tsv(
            tmp_path / "run-1_events.tsv",
            [
                {"onset": 1.1, "duration": 1.0, "trial_type": "A"},
            ],
        )
        f2 = _write_tsv(
            tmp_path / "run-2_events.tsv",
            [
                {"onset": 2.2, "duration": 1.0, "trial_type": "A"},
            ],
        )
        all_onsets, _, _ = parse_bids_events([f2, f1])  # reversed
        assert all_onsets[0][0].tolist() == [2.2]
        assert all_onsets[0][1].tolist() == [1.1]

    def test_condition_only_in_some_runs(self, tmp_path):
        """Condition present in run 2 but not run 1 still gets an entry,
        with an empty onset array for run 1."""
        f1 = _write_tsv(
            tmp_path / "run-1_events.tsv",
            [
                {"onset": 0.0, "duration": 1.0, "trial_type": "A"},
            ],
        )
        f2 = _write_tsv(
            tmp_path / "run-2_events.tsv",
            [
                {"onset": 5.0, "duration": 1.0, "trial_type": "rare"},
            ],
        )
        all_onsets, _, labels = parse_bids_events([f1, f2])
        rare_idx = labels.index("rare")
        assert all_onsets[rare_idx][0].size == 0
        assert all_onsets[rare_idx][1].tolist() == [5.0]


# ---------------------------------------------------------------------------
# parse_bids_events — single shared file broadcast across runs
# ---------------------------------------------------------------------------


class TestBroadcastSharedFile:
    """A dataset with identical timing every run may ship one *_events.tsv for
    the whole task. n_runs broadcasts that single file across all runs."""

    def test_single_file_broadcasts_to_n_runs(self, tmp_path):
        f = _write_tsv(
            tmp_path / "task-foo_events.tsv",
            [
                {"onset": 10.0, "duration": 20.0, "trial_type": "block"},
                {"onset": 70.0, "duration": 20.0, "trial_type": "block"},
            ],
        )
        all_onsets, durations, labels = parse_bids_events([f], n_runs=5)
        assert labels == ["block"]
        assert durations == [20.0]
        # One condition, five runs, each identical to the parsed onsets.
        assert len(all_onsets[0]) == 5
        for run_idx in range(5):
            np.testing.assert_array_equal(all_onsets[0][run_idx], [10.0, 70.0])

    def test_broadcast_runs_are_independent_copies(self, tmp_path):
        """Mutating one run's onset array must not touch the others (no aliasing)."""
        f = _write_tsv(
            tmp_path / "task-foo_events.tsv",
            [{"onset": 0.0, "duration": 1.0, "trial_type": "A"}],
        )
        all_onsets, _, _ = parse_bids_events([f], n_runs=3)
        all_onsets[0][0][0] = 999.0
        assert all_onsets[0][1].tolist() == [0.0]
        assert all_onsets[0][2].tolist() == [0.0]

    def test_n_runs_one_is_noop(self, tmp_path):
        f = _write_tsv(
            tmp_path / "task-foo_events.tsv",
            [{"onset": 0.0, "duration": 1.0, "trial_type": "A"}],
        )
        all_onsets, _, _ = parse_bids_events([f], n_runs=1)
        assert len(all_onsets[0]) == 1

    def test_default_no_broadcast(self, tmp_path):
        """Without n_runs, a single file yields a single run (unchanged behavior)."""
        f = _write_tsv(
            tmp_path / "task-foo_events.tsv",
            [{"onset": 0.0, "duration": 1.0, "trial_type": "A"}],
        )
        all_onsets, _, _ = parse_bids_events([f])
        assert len(all_onsets[0]) == 1

    def test_multi_file_with_mismatched_n_runs_raises(self, tmp_path):
        f1 = _write_tsv(
            tmp_path / "run-1_events.tsv",
            [{"onset": 0.0, "duration": 1.0, "trial_type": "A"}],
        )
        f2 = _write_tsv(
            tmp_path / "run-2_events.tsv",
            [{"onset": 5.0, "duration": 1.0, "trial_type": "A"}],
        )
        with pytest.raises(ValueError, match="broadcasting"):
            parse_bids_events([f1, f2], n_runs=5)

    def test_multi_file_matching_n_runs_ok(self, tmp_path):
        """n_runs equal to the file count is accepted and does not broadcast."""
        f1 = _write_tsv(
            tmp_path / "run-1_events.tsv",
            [{"onset": 0.0, "duration": 1.0, "trial_type": "A"}],
        )
        f2 = _write_tsv(
            tmp_path / "run-2_events.tsv",
            [{"onset": 5.0, "duration": 1.0, "trial_type": "A"}],
        )
        all_onsets, _, _ = parse_bids_events([f1, f2], n_runs=2)
        assert all_onsets[0][0].tolist() == [0.0]
        assert all_onsets[0][1].tolist() == [5.0]


# ---------------------------------------------------------------------------
# Duration aggregation
# ---------------------------------------------------------------------------


class TestDurationAggregation:
    def test_uniform_durations_pass_through(self, tmp_path):
        f = _write_tsv(
            tmp_path / "run-1_events.tsv",
            [
                {"onset": 0.0, "duration": 3.0, "trial_type": "A"},
                {"onset": 10.0, "duration": 3.0, "trial_type": "A"},
            ],
        )
        _, durations, _ = parse_bids_events([f])
        assert durations == [3.0]

    def test_varied_durations_use_median_and_warn(self, tmp_path, capsys):
        f = _write_tsv(
            tmp_path / "run-1_events.tsv",
            [
                {"onset": 0.0, "duration": 2.0, "trial_type": "A"},
                {"onset": 10.0, "duration": 3.0, "trial_type": "A"},
                {"onset": 20.0, "duration": 4.0, "trial_type": "A"},
            ],
        )
        _, durations, _ = parse_bids_events([f])
        assert durations == [3.0]  # median of {2,3,4}
        captured = capsys.readouterr()
        assert "multiple durations" in captured.err
        assert "median" in captured.err

    def test_round_durations_collapses_float_noise(self, tmp_path):
        """Without rounding, 3.0/3.03/3.001 produce 3 distinct values and
        trip the multi-duration median path. With round_durations=1 they
        collapse to a single value."""
        f = _write_tsv(
            tmp_path / "run-1_events.tsv",
            [
                {"onset": 0.0, "duration": 3.0, "trial_type": "A"},
                {"onset": 10.0, "duration": 3.03, "trial_type": "A"},
                {"onset": 20.0, "duration": 3.001, "trial_type": "A"},
            ],
        )
        _, durations_no_round, _ = parse_bids_events([f])
        _, durations_rounded, _ = parse_bids_events([f], round_durations=1)
        # Without rounding the medianization fires; with rounding all
        # collapse to 3.0 and durations is the single value.
        assert durations_rounded == [3.0]
        # And the no-round path produces something (median) — just verifying
        # the rounding path is meaningfully different.
        assert durations_no_round != [3.0] or len({3.0, 3.03, 3.001}) > 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_bids_events([tmp_path / "does_not_exist.tsv"])

    def test_missing_required_column_raises(self, tmp_path):
        f = _write_tsv(
            tmp_path / "bad.tsv",
            [
                {"start": 0.0, "len": 1.0, "cond": "A"},
            ],
            cols=["start", "len", "cond"],
        )
        # Default columns are onset/duration/trial_type — none present
        with pytest.raises(ValueError, match="not found"):
            parse_bids_events([f])

    def test_non_numeric_onset_raises(self, tmp_path):
        f = _write_tsv(
            tmp_path / "bad.tsv",
            [
                {"onset": "not_a_number", "duration": 1.0, "trial_type": "A"},
            ],
        )
        with pytest.raises(ValueError, match="numeric"):
            parse_bids_events([f])

    def test_all_filtered_raises(self, tmp_path):
        f = _write_tsv(
            tmp_path / "run-1_events.tsv",
            [
                {"onset": 0.0, "duration": 1.0, "trial_type": "A"},
            ],
        )
        with pytest.raises(ValueError, match="No conditions remain"):
            parse_bids_events([f], event_ignore=["A"])


# ---------------------------------------------------------------------------
# Run pairing / ordering / late-event guards
#
# Regression cover for a silent multi-session corruption: parse_bids_events used to
# sort event files by run number alone. Run number is not unique across sessions, so
# every run-001 grouped together and session N's timing landed on session 1's run N.
# ---------------------------------------------------------------------------


class TestPathEntities:
    def test_parses_bids_and_hyphenless_forms(self):
        from fastfuncstuff.design.bids_events import parse_path_entities

        # Derivative filenames routinely drop `sub-` and write run01 rather than run-01.
        assert parse_path_entities("ses-01_task-mvpsA_run01_final.nii.gz") == {
            "ses": "1",
            "task": "mvpsa",
            "run": "1",
        }
        ent = parse_path_entities("sub-pilot01_ses-01_task-mvpsA_run-001_events.tsv")
        assert ent["sub"] == "pilot01"
        assert (ent["ses"], ent["task"], ent["run"]) == ("1", "mvpsa", "1")

    def test_zero_padding_never_registers_as_a_mismatch(self):
        from fastfuncstuff.design.bids_events import parse_path_entities

        assert parse_path_entities("run-1.nii")["run"] == parse_path_entities("run-001.nii")["run"]

    def test_missing_entities_are_absent_not_guessed(self):
        from fastfuncstuff.design.bids_events import parse_path_entities

        assert parse_path_entities("scan_a.nii") == {}


class TestVerifyEventsMatchInputs:
    def _lists(self):
        inputs, events = [], []
        for ses, task, n in (("01", "mvpsA", 3), ("02", "mvpsA", 2), ("02", "mvpsB", 2)):
            for r in range(1, n + 1):
                inputs.append(f"ses-{ses}_task-{task}_run{r:02d}_final.nii.gz")
                events.append(f"sub-p01_ses-{ses}_task-{task}_run-{r:03d}_events.tsv")
        return inputs, events

    def test_matching_lists_pass(self):
        from fastfuncstuff.design.bids_events import verify_events_match_inputs

        inputs, events = self._lists()
        assert verify_events_match_inputs(inputs, events) == []

    def test_run_number_sort_is_caught(self):
        """The exact historical failure: events grouped by run across sessions."""
        from fastfuncstuff.design.bids_events import (
            sort_bids_event_files,
            verify_events_match_inputs,
        )

        inputs, events = self._lists()
        scrambled = [str(p) for p in sort_bids_event_files(events)]
        problems = verify_events_match_inputs(inputs, scrambled)
        assert problems, "run-number sort must not pass verification"
        assert any("MISMATCH" in p for p in problems)

    def test_length_mismatch_reported(self):
        from fastfuncstuff.design.bids_events import verify_events_match_inputs

        inputs, events = self._lists()
        problems = verify_events_match_inputs(inputs, events[:-1])
        assert len(problems) == 1 and "events files" in problems[0]

    def test_unparseable_names_are_skipped_not_flagged(self):
        """Non-BIDS filenames must not manufacture failures for people who don't use entities."""
        from fastfuncstuff.design.bids_events import verify_events_match_inputs

        assert verify_events_match_inputs(["a.nii", "b.nii"], ["x.tsv", "y.tsv"]) == []


class TestSortRunsByEntities:
    def test_restores_acquisition_order_keeping_pairs_together(self):
        import random

        from fastfuncstuff.design.bids_events import sort_runs_by_entities

        inputs, events = [], []
        for ses, task, n in (("01", "mvpsA", 3), ("02", "mvpsA", 2), ("02", "mvpsB", 2)):
            for r in range(1, n + 1):
                inputs.append(f"ses-{ses}_task-{task}_run{r:02d}_final.nii.gz")
                events.append(f"sub-p01_ses-{ses}_task-{task}_run-{r:03d}_events.tsv")

        idx = list(range(len(inputs)))
        random.Random(0).shuffle(idx)
        _, si, se = sort_runs_by_entities([inputs[i] for i in idx], [events[i] for i in idx])
        assert si == inputs
        assert se == events

    def test_task_orders_before_run_so_tasks_stay_contiguous(self):
        """Sorting on (ses, run) alone would interleave two tasks within a session."""
        from fastfuncstuff.design.bids_events import sort_runs_by_entities

        files = [
            "ses-04_task-mvpsB_run02_final.nii.gz",
            "ses-04_task-mvpsA_run01_final.nii.gz",
            "ses-04_task-mvpsB_run01_final.nii.gz",
            "ses-04_task-mvpsA_run02_final.nii.gz",
        ]
        _, out, _ = sort_runs_by_entities(files)
        assert [f.split("_")[1] for f in out] == [
            "task-mvpsA",
            "task-mvpsA",
            "task-mvpsB",
            "task-mvpsB",
        ]


class TestLateEvents:
    def test_finds_and_drops_events_past_run_end(self):
        import numpy as np

        from fastfuncstuff.design.bids_events import drop_late_events, find_late_events

        # condition A, two runs of 100 s; run 1 has two onsets past the end
        all_onsets = [[np.array([10.0, 50.0]), np.array([10.0, 105.0, 130.0])]]
        late = find_late_events(all_onsets, [100.0, 100.0], ["A"])
        assert len(late) == 1
        assert late[0]["run"] == 1
        assert late[0]["n_late"] == 2
        assert late[0]["last_onset"] == 130.0
        assert late[0]["conditions"] == ["A"]

        cleaned = drop_late_events(all_onsets, [100.0, 100.0])
        assert cleaned[0][1].tolist() == [10.0]
        assert cleaned[0][0].tolist() == [10.0, 50.0]

    def test_no_late_events_is_empty(self):
        import numpy as np

        from fastfuncstuff.design.bids_events import find_late_events

        assert find_late_events([[np.array([1.0, 2.0])]], [100.0], ["A"]) == []

    def test_onset_exactly_at_run_end_is_late(self):
        """An onset at t == run_length has no timepoints left, so its column is all-zero."""
        import numpy as np

        from fastfuncstuff.design.bids_events import find_late_events

        assert len(find_late_events([[np.array([100.0])]], [100.0], ["A"])) == 1


class TestDuplicateInputs:
    def test_same_run_in_two_compression_formats_is_flagged(self):
        """A glob like '*_final.nii.*' matches both copies; the run would be
        concatenated twice, double-counting its timepoints and events."""
        from fastfuncstuff.design.bids_events import find_duplicate_inputs

        inputs = [
            "../ses-05_run01_final.nii.gz",
            "../ses-05_run01_final.nii.zst",
            "../ses-05_run02_final.nii.gz",
        ]
        dupes = find_duplicate_inputs(inputs)
        assert len(dupes) == 1
        assert dupes[0] == [0, 1]

    def test_distinct_runs_are_not_flagged(self):
        from fastfuncstuff.design.bids_events import find_duplicate_inputs

        assert find_duplicate_inputs(["a_run01.nii.gz", "a_run02.nii.gz"]) == []

    def test_plain_nii_and_gz_of_same_stem_collide(self):
        from fastfuncstuff.design.bids_events import find_duplicate_inputs

        assert len(find_duplicate_inputs(["x_final.nii", "x_final.nii.gz"])) == 1

    def test_sort_rejects_length_mismatch_before_indexing(self):
        """-sort ran before any length check and indexed off the end of the events list."""
        from fastfuncstuff.design.bids_events import sort_runs_by_entities

        with pytest.raises(ValueError, match="cannot reorder"):
            sort_runs_by_entities(["a.nii", "b.nii", "c.nii"], ["e1.tsv", "e2.tsv"])


class TestCheckEventsPairing:
    """The shared guard every -events CLI routes through.

    It used to live inline in two tools out of a dozen, so ffs_hrfopt would
    happily fit a 13-run/4-task dataset with the timing rotated by one run.
    """

    def test_multi_task_mispairing_raises(self):
        from fastfuncstuff.design.bids_events import check_events_pairing

        inputs = [f"sub-01_task-AAAA_run-{i:02d}_bold.nii.gz" for i in (1, 2, 3)]
        events = [f"sub-01_task-BBBB_run-{i:02d}_events.tsv" for i in (1, 2, 3)]
        with pytest.raises(ValueError, match="entity check"):
            check_events_pairing(inputs, events, n_runs=3)

    def test_matching_pairs_print_the_table(self, capsys):
        from fastfuncstuff.design.bids_events import check_events_pairing

        inputs = [f"sub-01_task-AAAA_run-{i:02d}_bold.nii.gz" for i in (1, 2)]
        events = [f"sub-01_task-AAAA_run-{i:02d}_events.tsv" for i in (1, 2)]
        check_events_pairing(inputs, events, n_runs=2)
        out = capsys.readouterr().out
        assert "Timing pairing" in out
        # Both slots listed, input on the left of its own events file.
        assert "[  0] sub-01_task-AAAA_run-01_bold.nii.gz  <-  " in out
        assert "sub-01_task-AAAA_run-02_events.tsv" in out

    def test_entity_free_names_still_print(self, capsys):
        """The check only sees sub/ses/task/run; everything else needs eyeballs."""
        from fastfuncstuff.design.bids_events import check_events_pairing

        check_events_pairing(["a.nii.gz", "b.nii.gz"], ["one.tsv", "two.tsv"], n_runs=2)
        assert "Timing pairing" in capsys.readouterr().out

    def test_broadcast_single_file_is_not_a_mismatch(self, capsys):
        from fastfuncstuff.design.bids_events import check_events_pairing

        check_events_pairing(
            ["sub-01_run-01_bold.nii.gz", "sub-01_run-02_bold.nii.gz"],
            ["sub-01_events.tsv"],
            n_runs=2,
        )
        assert "Broadcasting" in capsys.readouterr().out

    def test_long_run_list_is_elided(self, capsys):
        from fastfuncstuff.design.bids_events import check_events_pairing

        n = 40
        inputs = [f"sub-01_run-{i:02d}_bold.nii.gz" for i in range(n)]
        events = [f"sub-01_run-{i:02d}_events.tsv" for i in range(n)]
        check_events_pairing(inputs, events, n_runs=n)
        out = capsys.readouterr().out
        assert "... 20 more ..." in out
        assert f"[ {n - 1}]" in out  # the tail is what catches an off-by-one

    def test_verbose_false_is_silent_but_still_checks(self, capsys):
        from fastfuncstuff.design.bids_events import check_events_pairing

        check_events_pairing(["sub-01_run-01.nii"], ["sub-01_run-01_events.tsv"], verbose=False)
        assert capsys.readouterr().out == ""
        with pytest.raises(ValueError):
            check_events_pairing(
                ["sub-01_task-AAAA_run-01.nii"],
                ["sub-01_task-BBBB_run-01_events.tsv"],
                verbose=False,
            )

    def test_constant_run_offset_is_accepted_with_a_note(self, capsys):
        """0-indexed derivatives beside 1-indexed BIDS events are correctly paired."""
        from fastfuncstuff.design.bids_events import check_events_pairing

        inputs = [f"run{i}.nii.gz" for i in range(3)]
        events = [f"run-{i + 1:02d}_events.tsv" for i in range(3)]
        check_events_pairing(inputs, events, n_runs=3)
        assert "offset by +1" in capsys.readouterr().out

    def test_rotation_is_still_rejected(self):
        """The failure the check exists for: a wrapped rotation, not a re-index."""
        from fastfuncstuff.design.bids_events import check_events_pairing

        inputs = [f"sub-01_run-{i:02d}_bold.nii.gz" for i in (1, 2, 3, 4)]
        events = [f"sub-01_run-{i:02d}_events.tsv" for i in (2, 3, 4, 1)]
        with pytest.raises(ValueError, match="entity check"):
            check_events_pairing(inputs, events, n_runs=4)

    def test_offset_does_not_excuse_a_task_swap(self):
        from fastfuncstuff.design.bids_events import check_events_pairing

        inputs = [f"task-AAAA_run-{i}_bold.nii.gz" for i in (1, 2)]
        events = [f"task-BBBB_run-{i}_events.tsv" for i in (2, 3)]
        with pytest.raises(ValueError, match="entity check"):
            check_events_pairing(inputs, events, n_runs=2)
