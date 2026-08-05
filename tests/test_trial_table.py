"""Companion single-trial events table (fastfuncstuff/design/trial_table.py)."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from fastfuncstuff.cli_utils import parse_timing_spec
from fastfuncstuff.design.trial_table import (
    build_trial_table,
    sanitize_levels,
    write_single_trial_event_table,
)
from fastfuncstuff.glm.ridge import create_single_trial_design

TR = 1.0
RUN_TPS = 60


def _write_events(path, rows, extra_cols=()):
    header = ["onset", "duration", "trial_type", *extra_cols]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(header)
        for row in rows:
            w.writerow(row)
    return str(path)


@pytest.fixture
def two_run_events(tmp_path):
    """Two runs, two conditions, deliberately non-chronological file order."""
    f1 = _write_events(
        tmp_path / "sub-01_ses-02_task-mem_run-01_events.tsv",
        [
            (12.0, 2.0, "face", "a"),
            (4.0, 2.0, "house", "b"),
            (20.0, 2.0, "face", "c"),
            (30.0, 2.0, "house", "d"),
        ],
        extra_cols=("stimulus file",),
    )
    f2 = _write_events(
        tmp_path / "sub-01_ses-02_task-mem_run-02_events.tsv",
        [
            (6.0, 2.0, "house", "e"),
            (18.0, 2.0, "face", "f"),
        ],
        extra_cols=("stimulus file",),
    )
    return [f1, f2]


def test_table_row_order_matches_design_columns(two_run_events):
    """The i-th table row must describe the i-th single-trial beta volume.

    This is the whole contract: the table is generated independently of the design
    builder, so if create_single_trial_design's sort ever changes, this fails.
    """
    run_starts = [0, RUN_TPS]
    spec = parse_timing_spec(
        events=two_run_events,
        onsets=None,
        durations_arg=None,
        n_runs=2,
        verbose=False,
    )
    _design, trial_labels, _cond_ids, run_ids, _cond_design = create_single_trial_design(
        onsets_by_condition=spec.all_onsets,
        durations=spec.durations,
        run_starts=run_starts,
        tr=TR,
        n_timepoints=2 * RUN_TPS,
        condition_labels=spec.condition_labels,
    )

    header, table = build_trial_table(two_run_events, run_starts, TR)

    assert len(table) == len(trial_labels)
    # Condition sequence and run assignment must match the design column for column.
    assert [r["condition"] for r in table] == [lbl.rsplit("_", 1)[0] for lbl in trial_labels]
    assert [int(r["run_index"]) for r in table] == run_ids.tolist()

    # Absolute onsets are monotonically non-decreasing, like the design's sort.
    abs_onsets = [int(r["run_index"]) * RUN_TPS * TR + float(r["onset"]) for r in table]
    assert abs_onsets == sorted(abs_onsets)
    assert "stimulus file" in header


def test_table_carries_every_original_column_and_entities(two_run_events):
    header, table = build_trial_table(two_run_events, [0, RUN_TPS], TR)

    for col in ("onset", "duration", "trial_type", "stimulus file"):
        assert col in header
    for col in ("subject", "session", "task", "run", "trial_index", "events_file"):
        assert col in header

    first = table[0]
    assert first["subject"] == "01"
    assert first["session"] == "02"
    assert first["task"] == "mem"
    # run comes from the filename of the events file that row came from
    assert first["run"] == f"{int(first['run_index']) + 1:02d}"
    # Original values survive verbatim, including a column name with a space.
    assert set(r["stimulus file"] for r in table) == {"a", "b", "c", "d", "e", "f"}


def test_added_columns_yield_to_a_same_named_events_column(tmp_path):
    """A user column called 'condition' must not be shadowed by ours."""
    f = _write_events(
        tmp_path / "sub-01_task-x_run-01_events.tsv",
        [(1.0, 1.0, "a", "user-value"), (5.0, 1.0, "b", "other")],
        extra_cols=("condition",),
    )
    header, table = build_trial_table([f], [0], TR)
    assert header.count("condition") == 1
    assert "ffs_condition" in header
    assert table[0]["condition"] == "user-value"
    assert table[0]["ffs_condition"] == "a"


def test_basis_expansion_gives_one_row_per_volume(two_run_events):
    header, table = build_trial_table(two_run_events, [0, RUN_TPS], TR, n_basis=2)
    assert "basis" in header
    assert len(table) == 12  # 6 trials x 2 basis functions
    assert [r["basis"] for r in table[:2]] == ["canonical", "timederiv"]
    assert [int(r["trial_index"]) for r in table[:4]] == [0, 1, 2, 3]
    assert table[0]["onset"] == table[1]["onset"]


def test_ignored_conditions_are_absent(two_run_events):
    _header, table = build_trial_table(two_run_events, [0, RUN_TPS], TR, event_ignore=["house"])
    assert {r["condition"] for r in table} == {"face"}
    assert len(table) == 3


def test_single_shared_events_file_is_broadcast(tmp_path):
    f = _write_events(
        tmp_path / "task-x_run-01_events.tsv",
        [(2.0, 1.0, "a"), (10.0, 1.0, "b")],
    )
    _header, table = build_trial_table([f], [0, RUN_TPS], TR, n_runs=2)
    assert len(table) == 4
    assert [int(r["run_index"]) for r in table] == [0, 0, 1, 1]


def test_late_events_dropped_when_requested(two_run_events):
    """Mirrors ffs_reml -allow_late_events, which drops them from the design too."""
    _header, table = build_trial_table(
        two_run_events, [0, RUN_TPS], TR, run_lengths_sec=[20.0, 60.0]
    )
    assert len(table) == 4  # run-01's onsets at 20 s and 30 s fall at/after the end
    assert all(float(r["onset"]) < 20.0 for r in table if r["run_index"] == "0")


def test_writer_skips_when_filenames_have_no_entities(tmp_path, capsys):
    f = _write_events(tmp_path / "timing.tsv", [(1.0, 1.0, "a")])
    out = write_single_trial_event_table(str(tmp_path / "pfx"), [f], [0], TR)
    assert out is None
    assert "no BIDS entities" in capsys.readouterr().out


def test_writer_emits_tsv(tmp_path, two_run_events):
    out = write_single_trial_event_table(str(tmp_path / "pfx"), two_run_events, [0, RUN_TPS], TR)
    assert out == str(tmp_path / "pfx_single_trial_events.tsv")
    with open(out, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert len(rows) == 6
    assert [int(r["trial_index"]) for r in rows] == list(range(6))


def test_writer_no_events_is_a_noop(tmp_path):
    assert write_single_trial_event_table(str(tmp_path / "pfx"), None, [0], TR) is None


def test_sanitize_levels_keeps_distinct_labels_distinct():
    values = ["face, inverted", "face-inverted", "face, inverted", "plain"]
    out, mapping = sanitize_levels(values)
    assert out[0] == out[2]
    assert out[0] != out[1]
    assert len(set(mapping.values())) == 3
    assert all(" " not in v and "," not in v for v in out)
    # A label that sanitizes to nothing still gets a usable identifier.
    assert sanitize_levels(["!!!"])[0] == ["unlabeled"]


def test_trial_table_matches_ridge_labels_with_repeats(tmp_path):
    """Repeat numbering in the beta labels lines up with row order per condition."""
    f = _write_events(
        tmp_path / "sub-01_task-x_run-01_events.tsv",
        [(0.0, 1.0, "a"), (3.0, 1.0, "b"), (6.0, 1.0, "a"), (9.0, 1.0, "b")],
    )
    spec = parse_timing_spec(events=[f], onsets=None, durations_arg=None, n_runs=1, verbose=False)
    _d, trial_labels, _c, _r, _cd = create_single_trial_design(
        onsets_by_condition=spec.all_onsets,
        durations=spec.durations,
        run_starts=[0],
        tr=TR,
        n_timepoints=RUN_TPS,
        condition_labels=spec.condition_labels,
    )
    _header, table = build_trial_table([f], [0], TR)
    assert trial_labels == ["a_001", "b_001", "a_002", "b_002"]
    assert [r["condition"] for r in table] == ["a", "b", "a", "b"]
    assert [float(r["onset"]) for r in table] == [0.0, 3.0, 6.0, 9.0]
    assert np.array_equal(np.arange(len(table)), np.array([int(r["trial_index"]) for r in table]))
