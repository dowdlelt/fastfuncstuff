"""
Tests for cli_utils.parse_timing_spec — the shared -events / -onsets parser.

Every ffs_* GLM tool used to hand-roll this block, which is how a single
shared BIDS events TSV came to broadcast across runs in ffs_reml but not in
ffs_hrfopt. These assertions pin the behaviour both paths must share.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fastfuncstuff.cli_utils import parse_timing_spec


def _write_events(path: Path, onsets, duration=2.0, trial_type="stim") -> Path:
    lines = ["onset\tduration\ttrial_type"]
    lines += [f"{o}\t{duration}\t{trial_type}" for o in onsets]
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_timing(path: Path, onsets_per_run) -> Path:
    path.write_text("\n".join(" ".join(str(o) for o in run) for run in onsets_per_run) + "\n")
    return path


def test_single_events_file_broadcasts_across_runs(tmp_path, capsys):
    ev = _write_events(tmp_path / "task_events.tsv", [10.0, 30.0, 50.0])

    spec = parse_timing_spec(
        events=[str(ev)], onsets=None, durations_arg=None, n_runs=5, verbose=True
    )

    assert spec.from_events
    assert spec.condition_labels == ["stim"]
    assert len(spec.all_onsets[0]) == 5
    for run_onsets in spec.all_onsets[0]:
        np.testing.assert_allclose(run_onsets, [10.0, 30.0, 50.0])
    assert "Broadcasting 1 events file across 5 runs" in capsys.readouterr().out


def test_events_count_must_be_one_or_n_runs(tmp_path):
    files = [str(_write_events(tmp_path / f"run-0{i}_events.tsv", [10.0 * i + 5])) for i in (1, 2)]

    with pytest.raises(ValueError, match="one TSV per run or a single shared TSV"):
        parse_timing_spec(events=files, onsets=None, durations_arg=None, n_runs=4, verbose=False)


def test_onsets_path_parses_durations_and_labels(tmp_path):
    f1 = _write_timing(tmp_path / "faces.1D", [[0.0, 20.0], [5.0, 25.0]])
    f2 = _write_timing(tmp_path / "houses.1D", [[10.0], [15.0]])

    spec = parse_timing_spec(
        events=None,
        onsets=[str(f1), str(f2)],
        durations_arg=["3.0"],
        n_runs=2,
        verbose=False,
    )

    assert not spec.from_events
    assert spec.condition_labels == ["faces", "houses"]
    assert spec.durations == [3.0, 3.0]
    assert spec.onset_files == [str(f1), str(f2)]
    np.testing.assert_allclose(spec.all_onsets[0][1], [5.0, 25.0])


def test_onsets_run_count_mismatch_raises(tmp_path):
    f1 = _write_timing(tmp_path / "faces.1D", [[0.0], [5.0]])

    with pytest.raises(ValueError, match="has 2 runs"):
        parse_timing_spec(
            events=None, onsets=[str(f1)], durations_arg=["3.0"], n_runs=3, verbose=False
        )


def test_missing_onset_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_timing_spec(
            events=None,
            onsets=[str(tmp_path / "nope.1D")],
            durations_arg=["2.0"],
            n_runs=1,
            verbose=False,
        )


@pytest.mark.parametrize(
    ("events", "onsets"),
    [(None, None), (["a.tsv"], ["b.1D"])],
)
def test_exactly_one_timing_source_required(events, onsets):
    with pytest.raises(ValueError, match="exactly one"):
        parse_timing_spec(events=events, onsets=onsets, durations_arg=None, n_runs=1, verbose=False)
