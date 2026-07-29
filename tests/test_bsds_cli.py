"""Smoke test for the ffs_bsds CLI: end-to-end on ROI time series."""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "tests")
from test_bsds_model import _simulate  # noqa: E402

from fastfuncstuff.cli.bsds import main  # noqa: E402


def test_cli_roi_timeseries_end_to_end(tmp_path):
    sessions, _, _, _ = _simulate(k=3, d=6, n_sessions=2, seed=1)
    files = []
    for i, s in enumerate(sessions):
        f = tmp_path / f"run{i}.1D"
        np.savetxt(f, s.numpy().T)  # time-major (N, D) -> CLI auto-orients
        files.append(str(f))

    stem = str(tmp_path / "out" / "sub01")
    rc = main(
        [
            "-input",
            *files,
            "-prefix",
            stem,
            "-n_states",
            "3",
            "-max_ldim",
            "3",
            "-n_init",
            "2",
            "-n_iter",
            "50",
            "-tr",
            "0.72",
            "-device",
            "cpu",
        ]
    )
    assert rc == 0

    summary = json.load(open(f"{stem}_summary.json"))
    assert summary["n_states"] == 3
    assert summary["n_sessions"] == 2
    np.testing.assert_allclose(sum(summary["group_occupancy"]), 1.0, atol=1e-6)

    model = np.load(f"{stem}_model.npz")
    assert model["state_covs"].shape == (3, 6, 6)
    assert model["transition"].shape == (3, 3)
    # Per-run MAP paths and posteriors exist and have the right length.
    for r in range(2):
        path = np.loadtxt(f"{stem}_run-{r:02d}_states.1D")
        prob = np.loadtxt(f"{stem}_run-{r:02d}_stateprob.1D")
        assert path.shape[0] == 400
        assert prob.shape == (400, 3)


def test_cli_with_events_task_alignment(tmp_path):
    sessions, _, _, _ = _simulate(k=3, d=6, n_sessions=2, seed=1)
    files, event_files = [], []
    for i, s in enumerate(sessions):
        f = tmp_path / f"run{i}.1D"
        np.savetxt(f, s.numpy().T)
        files.append(str(f))
        # Minimal events.tsv: three conditions blocked across the 400-TR run.
        ev = tmp_path / f"run-{i + 1:02d}_events.tsv"
        with open(ev, "w") as fh:
            fh.write("onset\tduration\ttrial_type\n")
            for blk, onset in enumerate(range(0, 360, 40)):
                fh.write(f"{onset}\t40\tcond{blk % 3}\n")
        event_files.append(str(ev))

    stem = str(tmp_path / "out" / "sub01")
    rc = main(
        [
            "-input",
            *files,
            "-prefix",
            stem,
            "-n_states",
            "3",
            "-max_ldim",
            "3",
            "-n_init",
            "2",
            "-n_iter",
            "50",
            "-tr",
            "1.0",
            "-device",
            "cpu",
            "-events",
            *event_files,
            "-plots",
            "qc",
        ]
    )
    assert rc == 0
    align = json.load(open(f"{stem}_task_alignment.json"))
    assert sorted(align["condition_labels"]) == ["cond0", "cond1", "cond2"]
    assert 0.0 <= align["normalized_mutual_info"] <= 1.0
    assert len(align["state_purity"]) == 3
    corr = np.loadtxt(f"{stem}_task_correlation.txt")
    assert corr.shape == (3, 3)
    import os

    assert os.path.exists(f"{stem}_task_alignment.png")


def test_cli_events_count_mismatch_errors(tmp_path):
    import pytest

    sessions, _, _, _ = _simulate(k=2, d=5, n_sessions=2, seed=1)
    files = []
    for i, s in enumerate(sessions):
        f = tmp_path / f"run{i}.1D"
        np.savetxt(f, s.numpy().T)
        files.append(str(f))
    # 3 events files for 2 runs: neither one-per-run nor the single shared
    # TSV that broadcasts, so it must error.
    evs = []
    for i in (1, 2, 3):
        ev = tmp_path / f"run-0{i}_events.tsv"
        with open(ev, "w") as fh:
            fh.write("onset\tduration\ttrial_type\n0\t10\tA\n")
        evs.append(str(ev))

    stem = str(tmp_path / "out" / "sub01")
    with pytest.raises(SystemExit):
        main(
            [
                "-input",
                *files,
                "-prefix",
                stem,
                "-n_states",
                "2",
                "-max_ldim",
                "2",
                "-n_init",
                "1",
                "-n_iter",
                "20",
                "-device",
                "cpu",
                "-events",
                *evs,
                "-plots",
                "none",
            ]
        )
