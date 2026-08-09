"""End-to-end smoke tests for -single_trials with -cv_design condition.

The point of the split is that per-trial betas can be produced for a design with
NO repeated conditions — the case where beta-space CV has nothing to score
against. These run the real CLI on a tiny synthetic dataset to prove the wiring
holds, not to check numerical quality.
"""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from fastfuncstuff.cli import denoise as denoise_cli

pytestmark = pytest.mark.slow

TR = 1.0
N_TP = 80
N_RUNS = 2
SHAPE = (6, 6, 3)


def _write_run(path: Path, data: np.ndarray) -> None:
    img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
    img.header["pixdim"][4] = TR
    nib.save(img, str(path))


def _make_dataset(tmp_path: Path, onsets_per_run, condition_names):
    """Synthetic 2-run dataset with a boxcar response in a few voxels."""
    rng = np.random.default_rng(0)
    inputs = []
    for run in range(N_RUNS):
        vol = 100.0 + 3.0 * rng.standard_normal((*SHAPE, N_TP))
        for onsets in onsets_per_run[run].values():
            for onset in onsets:
                t0 = int(round(onset / TR)) + 2
                t1 = min(N_TP, t0 + 4)
                vol[:2, :2, :, t0:t1] += 8.0
        path = tmp_path / f"run{run + 1}.nii.gz"
        _write_run(path, vol)
        inputs.append(str(path))

    onset_files = []
    for cond in condition_names:
        path = tmp_path / f"{cond}.txt"
        rows = []
        for run in range(N_RUNS):
            times = onsets_per_run[run].get(cond, [])
            rows.append(" ".join(f"{t:.1f}" for t in times) if times else "*")
        path.write_text("\n".join(rows) + "\n")
        onset_files.append(str(path))

    return inputs, onset_files


def _run_denoise(monkeypatch, tmp_path, inputs, onset_files, extra):
    prefix = str(tmp_path / "out")
    argv = [
        "ffs_denoise",
        "-input",
        *inputs,
        "-onsets",
        *onset_files,
        "-durations",
        "4.0",
        "-tr",
        str(TR),
        "-prefix",
        prefix,
        "-device",
        "cpu",
        "-max_comps",
        "2",
        "-min_noise_voxels",
        "10",
        "-plots",
        "no",
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    denoise_cli.main()
    return prefix


def test_thin_repeats_fall_back_and_still_emit_betas(monkeypatch, tmp_path, capsys):
    """Mostly run-exclusive conditions: auto must fall back to condition scoring.

    This is the case the split actually rescues — beta-space CV would score the
    run-exclusive trials against nothing, while the condition design still has
    one shared column to cross-validate on.
    """
    names = ["shared"] + [f"stim{i}" for i in range(6)]
    onsets_per_run = [
        {"shared": [10.0], **{names[1 + i]: [25.0 + 12 * i] for i in range(3)}},
        {"shared": [10.0], **{names[4 + i]: [25.0 + 12 * i] for i in range(3)}},
    ]
    inputs, onset_files = _make_dataset(tmp_path, onsets_per_run, names)

    prefix = _run_denoise(monkeypatch, tmp_path, inputs, onset_files, ["-single_trials"])

    out = capsys.readouterr().out
    assert "auto fell back" in out
    assert "condition-level design" in out
    # The whole point: per-trial betas exist despite unscoreable beta space.
    assert list(Path(prefix).parent.glob("out*single*.nii.gz"))


def test_no_cross_run_structure_exits_early(monkeypatch, tmp_path, capsys):
    """Every condition confined to one run: neither design can be scored.

    The tool must say so up front instead of dying inside compute_xval_r2.
    """
    names = [f"stim{i}" for i in range(4)]
    onsets_per_run = [
        {names[i]: [10.0 + 8 * i] for i in range(2)},
        {names[i]: [10.0 + 8 * (i - 2)] for i in range(2, 4)},
    ]
    inputs, onset_files = _make_dataset(tmp_path, onsets_per_run, names)

    for extra in (["-single_trials"], ["-single_trials", "-cv_design", "single"]):
        with pytest.raises(SystemExit):
            _run_denoise(monkeypatch, tmp_path, inputs, onset_files, extra)
        out = capsys.readouterr().out
        assert "-pcstop" in out


def test_repeated_conditions_can_still_be_scored_on_conditions(monkeypatch, tmp_path, capsys):
    """Repeats available, but the user asks for condition-level selection anyway."""
    names = ["faces", "houses"]
    onsets_per_run = [
        {"faces": [10.0, 30.0], "houses": [20.0, 40.0]},
        {"faces": [12.0, 32.0], "houses": [22.0, 42.0]},
    ]
    inputs, onset_files = _make_dataset(tmp_path, onsets_per_run, names)

    prefix = _run_denoise(
        monkeypatch,
        tmp_path,
        inputs,
        onset_files,
        ["-single_trials", "-cv_design", "condition"],
    )

    out = capsys.readouterr().out
    assert "-cv_design condition" in out
    assert list(Path(prefix).parent.glob("out*single*.nii.gz"))

    import json

    meta = json.loads(Path(f"{prefix}_denoise_metadata.json").read_text())
    assert meta["cv_design"] == "condition"
    assert meta["trial_repeats"]["n_repeated_conditions"] == 2


def test_hrfopt_condition_selection_still_emits_single_trials(monkeypatch, tmp_path):
    """The combination that was structurally impossible before the split.

    -save_single_trial_betas used to live inside the beta-space branch, so
    condition-level HRF selection could never produce per-trial betas.
    """
    import json

    from fastfuncstuff.cli import hrfopt as hrfopt_cli

    names = ["faces", "houses"]
    onsets_per_run = [
        {"faces": [10.0, 30.0], "houses": [20.0, 40.0]},
        {"faces": [12.0, 32.0], "houses": [22.0, 42.0]},
    ]
    inputs, onset_files = _make_dataset(tmp_path, onsets_per_run, names)

    prefix = str(tmp_path / "hopt")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ffs_hrfopt",
            "-input",
            *inputs,
            "-onsets",
            *onset_files,
            "-durations",
            "4.0",
            "-tr",
            str(TR),
            "-prefix",
            prefix,
            "-device",
            "cpu",
            "-n_hrfs",
            "3",
            "-single_trials",
            "-cv_design",
            "condition",
        ],
    )
    hrfopt_cli.main()

    assert Path(f"{prefix}_stats_single_trial.nii.gz").exists()
    meta = json.loads(Path(f"{prefix}_metadata.json").read_text())
    assert meta["cv_design"] == "condition"
    assert meta["single_trials"] is True
