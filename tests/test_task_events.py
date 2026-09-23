"""Tests for cli/task_events.task_design_from_events — the per-run task design the
NORDIC and locomoco diagnostics build from -events."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import torch

from fastfuncstuff.cli.task_events import task_design_from_events


def _args(tmp_path: Path, onsets) -> Namespace:
    ev = tmp_path / "events.tsv"
    lines = ["onset\tduration\ttrial_type"] + [f"{o}\t2\tstim" for o in onsets]
    ev.write_text("\n".join(lines) + "\n")
    return Namespace(events=[str(ev)], event_ignore=None, event_cols=None)


def test_events_past_a_shortened_run_end_are_dropped_not_fatal(tmp_path, capsys):
    # A run cut short (ffs_autoproc -cut_task_vols) keeps its full events file: the
    # tail events fall past the new end. That is not the wrong-TR symptom, but the
    # user is told what was left out.
    args = _args(tmp_path, [10.0, 40.0, 190.0])
    design, labels = task_design_from_events(args, 50, 2.0, torch.device("cpu"))
    assert design.shape == (50, 1)
    assert labels == ["stim"]
    assert "1 event(s) start at/after the run end (100s) and are ignored: stim x1" in (
        capsys.readouterr().out
    )


def test_every_event_past_the_end_still_names_the_tr(tmp_path):
    args = _args(tmp_path, [120.0, 150.0])
    with pytest.raises(ValueError, match="TR is almost certainly wrong"):
        task_design_from_events(args, 50, 2.0, torch.device("cpu"))
