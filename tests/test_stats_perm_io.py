"""
Tests for stats/perm_io.py — events loading and group/selection logic
for ffs_perm.

Unlike design.bids_events.parse_bids_events (collapses to per-condition
onsets), this module preserves per-row context so ffs_perm can do
arbitrary-column selection and run-block exchangeability. Bugs here
silently change which trials are tested or which group they belong to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fastfuncstuff.stats.perm_io import (
    EventsTable,
    load_events,
    select_one_sample,
    select_one_vs_all,
    select_two_sample,
)


def _write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    return path


# ---------------------------------------------------------------------------
# load_events
# ---------------------------------------------------------------------------

class TestLoadEvents:
    def test_preserves_row_order_within_run(self, tmp_path):
        f = _write_tsv(
            tmp_path / "run-1_events.tsv",
            ["onset", "duration", "trial_type", "rt"],
            [["0.0", "1.0", "A", "0.41"],
             ["10.0", "1.0", "B", "0.55"],
             ["20.0", "1.0", "A", "0.39"]],
        )
        ev = load_events([f])
        assert len(ev) == 3
        # Order matches file order
        assert [r["trial_type"] for r in ev.rows] == ["A", "B", "A"]
        # rt column preserved verbatim as string
        assert [r["rt"] for r in ev.rows] == ["0.41", "0.55", "0.39"]
        np.testing.assert_array_equal(ev.run_idx, [0, 0, 0])

    def test_concatenates_runs_with_run_idx(self, tmp_path):
        f1 = _write_tsv(
            tmp_path / "run-1_events.tsv",
            ["onset", "duration", "trial_type"],
            [["0.0", "1.0", "A"], ["5.0", "1.0", "B"]],
        )
        f2 = _write_tsv(
            tmp_path / "run-2_events.tsv",
            ["onset", "duration", "trial_type"],
            [["1.0", "1.0", "A"], ["6.0", "1.0", "A"]],
        )
        ev = load_events([f1, f2])
        assert len(ev) == 4
        np.testing.assert_array_equal(ev.run_idx, [0, 0, 1, 1])

    def test_sorts_files_by_run_number_before_loading(self, tmp_path):
        """run-2 passed before run-1; loader must reorder so run_idx=0 maps
        to the run-1 file."""
        f1 = _write_tsv(
            tmp_path / "run-1_events.tsv",
            ["onset", "duration", "trial_type"],
            [["0.0", "1.0", "run1_trial"]],
        )
        f2 = _write_tsv(
            tmp_path / "run-2_events.tsv",
            ["onset", "duration", "trial_type"],
            [["0.0", "1.0", "run2_trial"]],
        )
        ev = load_events([f2, f1])  # reversed
        assert [r["trial_type"] for r in ev.rows] == ["run1_trial", "run2_trial"]
        np.testing.assert_array_equal(ev.run_idx, [0, 1])

    def test_drops_na_trial_types_by_default(self, tmp_path):
        f = _write_tsv(
            tmp_path / "run-1_events.tsv",
            ["onset", "duration", "trial_type"],
            [["0.0", "1.0", "A"],
             ["5.0", "1.0", "n/a"],
             ["10.0", "1.0", ""],
             ["15.0", "1.0", "B"]],
        )
        ev = load_events([f])
        assert [r["trial_type"] for r in ev.rows] == ["A", "B"]

    def test_drop_na_false_keeps_na_rows(self, tmp_path):
        f = _write_tsv(
            tmp_path / "run-1_events.tsv",
            ["onset", "duration", "trial_type"],
            [["0.0", "1.0", "A"],
             ["5.0", "1.0", "n/a"]],
        )
        ev = load_events([f], drop_na=False)
        assert len(ev) == 2

    def test_union_of_columns_across_files_preserves_first_seen_order(self, tmp_path):
        f1 = _write_tsv(
            tmp_path / "run-1_events.tsv",
            ["onset", "duration", "trial_type", "rt"],
            [["0.0", "1.0", "A", "0.4"]],
        )
        f2 = _write_tsv(
            tmp_path / "run-2_events.tsv",
            ["onset", "duration", "trial_type", "accuracy"],
            [["0.0", "1.0", "A", "1"]],
        )
        ev = load_events([f1, f2])
        # rt seen first (in run-1), accuracy seen next (in run-2)
        assert ev.columns == ["onset", "duration", "trial_type", "rt", "accuracy"]


class TestEventsTableColumn:
    def test_returns_column_values(self, tmp_path):
        f = _write_tsv(
            tmp_path / "run-1_events.tsv",
            ["onset", "duration", "trial_type", "rt"],
            [["0.0", "1.0", "A", "0.4"], ["5.0", "1.0", "B", "0.5"]],
        )
        ev = load_events([f])
        np.testing.assert_array_equal(ev.column("trial_type"), ["A", "B"])
        np.testing.assert_array_equal(ev.column("rt"), ["0.4", "0.5"])

    def test_unknown_column_raises_with_available_list(self, tmp_path):
        f = _write_tsv(
            tmp_path / "run-1_events.tsv",
            ["onset", "duration", "trial_type"],
            [["0.0", "1.0", "A"]],
        )
        ev = load_events([f])
        with pytest.raises(KeyError, match="not found.*Available"):
            ev.column("bogus")

    def test_missing_value_in_row_returns_empty_string(self, tmp_path):
        """When a column exists in one file but a row in another file
        doesn't have it, the value comes back as ''."""
        f1 = _write_tsv(
            tmp_path / "run-1_events.tsv",
            ["onset", "duration", "trial_type", "rt"],
            [["0.0", "1.0", "A", "0.5"]],
        )
        f2 = _write_tsv(
            tmp_path / "run-2_events.tsv",
            ["onset", "duration", "trial_type"],  # no rt column
            [["0.0", "1.0", "B"]],
        )
        ev = load_events([f1, f2])
        rts = ev.column("rt")
        assert rts.tolist() == ["0.5", ""]


# ---------------------------------------------------------------------------
# select_one_sample
# ---------------------------------------------------------------------------

def _three_run_events(tmp_path: Path) -> EventsTable:
    """Build a 3-run events table: A appears in runs 0,1,2; B in 0,1; rare in 2."""
    runs = [
        (tmp_path / "run-1_events.tsv",
         [["0.0", "1", "A"], ["5.0", "1", "A"], ["10.0", "1", "B"]]),
        (tmp_path / "run-2_events.tsv",
         [["0.0", "1", "B"], ["5.0", "1", "A"]]),
        (tmp_path / "run-3_events.tsv",
         [["0.0", "1", "A"], ["5.0", "1", "rare"]]),
    ]
    files = []
    for path, rows in runs:
        files.append(_write_tsv(path, ["onset", "duration", "trial_type"], rows))
    return load_events(files)


class TestSelectOneSample:
    def test_picks_matching_trials(self, tmp_path):
        ev = _three_run_events(tmp_path)
        sel = select_one_sample(ev, "trial_type", "A")
        assert sel.label == "A"
        # 4 A trials: rows 0, 1, 4, 5
        np.testing.assert_array_equal(sel.indices, [0, 1, 4, 5])

    def test_blocks_reflect_source_runs(self, tmp_path):
        ev = _three_run_events(tmp_path)
        sel = select_one_sample(ev, "trial_type", "A")
        # A trials come from runs 0 (two), 1 (one), 2 (one)
        np.testing.assert_array_equal(sel.blocks, [0, 0, 1, 2])

    def test_fewer_than_three_matches_raises(self, tmp_path):
        ev = _three_run_events(tmp_path)
        with pytest.raises(ValueError, match="need ≥ 3"):
            select_one_sample(ev, "trial_type", "rare")  # only 1 match


# ---------------------------------------------------------------------------
# select_two_sample
# ---------------------------------------------------------------------------

class TestSelectTwoSample:
    def test_basic_group_assignment(self, tmp_path):
        ev = _three_run_events(tmp_path)
        sel = select_two_sample(ev, "trial_type", "A", "B")
        # A: indices 0,1,4,5 (4 trials, group=1); B: 2,3 (2 trials, group=0)
        np.testing.assert_array_equal(sel.indices, [0, 1, 4, 5, 2, 3])
        np.testing.assert_array_equal(sel.group, [1, 1, 1, 1, 0, 0])
        assert sel.label_a == "A"
        assert sel.label_b == "B"

    def test_blocks_per_selected_trial(self, tmp_path):
        ev = _three_run_events(tmp_path)
        sel = select_two_sample(ev, "trial_type", "A", "B")
        # A from runs [0,0,1,2], B from runs [0,1]
        np.testing.assert_array_equal(sel.blocks, [0, 0, 1, 2, 0, 1])

    def test_same_label_twice_raises(self, tmp_path):
        ev = _three_run_events(tmp_path)
        with pytest.raises(ValueError, match="distinct"):
            select_two_sample(ev, "trial_type", "A", "A")

    def test_insufficient_trials_raises(self, tmp_path):
        ev = _three_run_events(tmp_path)
        with pytest.raises(ValueError, match="≥ 2"):
            select_two_sample(ev, "trial_type", "A", "rare")  # 1 rare trial


# ---------------------------------------------------------------------------
# select_one_vs_all
# ---------------------------------------------------------------------------

class TestSelectOneVsAll:
    def test_group_a_is_label_group_b_is_everything_else(self, tmp_path):
        ev = _three_run_events(tmp_path)
        sel = select_one_vs_all(ev, "trial_type", "A")
        assert sel.label_a == "A"
        assert sel.label_b == "not-A"
        # A: 4 trials, others (B,B,rare): 3 trials
        assert int((sel.group == 1).sum()) == 4
        assert int((sel.group == 0).sum()) == 3

    def test_drop_values_excludes_those_trials_entirely(self, tmp_path):
        ev = _three_run_events(tmp_path)
        # Drop 'rare' → other group becomes just B trials
        sel = select_one_vs_all(ev, "trial_type", "A", drop_values=("rare",))
        # A still 4 trials, B 2 trials
        assert int((sel.group == 1).sum()) == 4
        assert int((sel.group == 0).sum()) == 2
        # The rare trial index (6) must not appear in sel.indices
        assert 6 not in sel.indices.tolist()

    def test_insufficient_other_group_raises(self, tmp_path):
        ev = _three_run_events(tmp_path)
        # Dropping B leaves only rare in the other group → 1 trial → raises
        with pytest.raises(ValueError, match="onevsall"):
            select_one_vs_all(ev, "trial_type", "A", drop_values=("B",))
