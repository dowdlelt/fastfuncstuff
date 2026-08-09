"""
Tests for the `-delta_denoise` flag on `ffs_hrfopt`.

This flag re-runs HRF selection with `ortvec_files=None` so the user can
see whether their supplied nuisance regressors actually changed which
HRF won per voxel. The plumbing — argument validation + the second
selection pass + delta-map writing — is what we pin here.

The selection math itself is exercised by test_hrf_selection.py; we
only need to confirm the second pass runs and emits the expected files
and summary stats.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from fastfuncstuff.cli import hrfopt as hrfopt_cli

# ---------------------------------------------------------------------------
# Argument validation (cheap, no GLM fit)
# ---------------------------------------------------------------------------


def _base_argv(prefix: str) -> list[str]:
    """A minimum-required argv that parser.parse_args accepts before any
    semantic validation runs. We're triggering the early validation block
    after parse_args, so we only need the parser to succeed."""
    return [
        "ffs_hrfopt",
        "-input",
        "r1.nii.gz",
        "r2.nii.gz",
        "-onsets",
        "c1.txt",
        "-durations",
        "2.0",
        "-tr",
        "2.0",
        "-prefix",
        prefix,
    ]


class TestArgValidation:
    def test_delta_denoise_without_ortvec_exits(self, monkeypatch, tmp_path, capsys):
        argv = _base_argv(str(tmp_path / "out")) + ["-delta_denoise"]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as excinfo:
            hrfopt_cli.main()
        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "-delta_denoise requires at least one -ortvec" in captured.out

    def test_delta_denoise_with_beta_space_cv_exits(self, monkeypatch, tmp_path, capsys):
        # -delta_denoise is incompatible with the *selection* path, not with
        # emitting single-trial betas, so the guard keys on -cv_design.
        ortvec_path = tmp_path / "motion.1D"
        ortvec_path.write_text(
            "0.0\n0.1\n0.2\n"
        )  # content doesn't matter; arg validation happens first
        argv = _base_argv(str(tmp_path / "out")) + [
            "-delta_denoise",
            "-ortvec",
            str(ortvec_path),
            "motion",
            "-single_trials",
            "-cv_design",
            "single",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as excinfo:
            hrfopt_cli.main()
        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "not yet supported with -cv_design single" in captured.out


# ---------------------------------------------------------------------------
# Functional smoke test: full pipeline with a tiny dataset
# ---------------------------------------------------------------------------


def _write_run(path: Path, data: np.ndarray, tr: float, affine: np.ndarray) -> None:
    """Save a (X, Y, Z, T) numpy array as a NIfTI with the right pixdim[4]."""
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    img.header["pixdim"][4] = tr  # TR
    nib.save(img, str(path))


def _make_synthetic_dataset(tmp_path: Path):
    """Build a 2-run, 4x4x2-voxel synthetic dataset with one stimulus
    condition, plus a motion-like ortvec that's actually correlated with
    something so the nodenoise pass can plausibly differ.

    Returns paths and metadata needed to invoke ffs_hrfopt.
    """
    rng = np.random.default_rng(0)
    tr = 2.0
    n_timepoints_per_run = 60
    nx, ny, nz = 4, 4, 2
    affine = np.eye(4) * 2.0
    affine[3, 3] = 1.0

    # One condition: 4 events per run at fixed TR-aligned times
    onset_times_per_run = [np.array([10.0, 30.0, 50.0, 70.0])]
    onsets_run1_file = tmp_path / "cond1_run1.txt"
    onsets_run1_file.write_text(" ".join(f"{t:.1f}" for t in onset_times_per_run[0]) + "\n")
    onsets_run2_file = tmp_path / "cond1_run2.txt"
    onsets_run2_file.write_text(" ".join(f"{t:.1f}" for t in onset_times_per_run[0]) + "\n")
    # AFNI-style two-row onsets file (one row per run)
    onsets_file = tmp_path / "cond1.txt"
    onsets_file.write_text(
        " ".join(f"{t:.1f}" for t in onset_times_per_run[0])
        + "\n"
        + " ".join(f"{t:.1f}" for t in onset_times_per_run[0])
        + "\n"
    )

    # Make two runs of synthetic data with mild task-like fluctuation +
    # an obvious motion-correlated nuisance component the user will supply.
    motion_per_run = []
    runs = []
    for run_idx in range(2):
        # Motion: smooth low-frequency signal, different per run
        t = np.arange(n_timepoints_per_run)
        motion = 0.4 * np.sin(2 * np.pi * t / 30.0 + run_idx) + 0.1 * rng.standard_normal(
            n_timepoints_per_run
        )
        motion_per_run.append(motion)

        # Build a simple task signal at the onset times (TR-aligned)
        task = np.zeros(n_timepoints_per_run)
        for onset_s in onset_times_per_run[0]:
            tp = int(round(onset_s / tr))
            if 0 <= tp < n_timepoints_per_run - 3:
                # Quick "HRF-ish" bump
                task[tp : tp + 4] += np.array([0.2, 0.6, 0.5, 0.3])

        # Voxel-level data: most voxels get task + motion + noise, scaled to ~100
        data = np.empty((nx, ny, nz, n_timepoints_per_run), dtype=np.float32)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    beta_task = 1.0 + 0.5 * rng.standard_normal()
                    beta_motion = 2.0 + 0.5 * rng.standard_normal()
                    noise = 0.3 * rng.standard_normal(n_timepoints_per_run)
                    data[i, j, k, :] = 100.0 + beta_task * task + beta_motion * motion + noise

        run_path = tmp_path / f"run{run_idx + 1}.nii.gz"
        _write_run(run_path, data, tr, affine)
        runs.append(run_path)

    # Concatenated motion ortvec (one column, two runs stacked)
    motion_concat = np.concatenate(motion_per_run)[:, None]
    ortvec_path = tmp_path / "motion.1D"
    np.savetxt(ortvec_path, motion_concat, fmt="%.6f")

    return {
        "runs": runs,
        "onsets": onsets_file,
        "ortvec": ortvec_path,
        "tr": tr,
        "n_timepoints_total": 2 * n_timepoints_per_run,
    }


def _run_hrfopt(monkeypatch, prefix: str, extra_args: list[str], ds: dict) -> None:
    argv = [
        "ffs_hrfopt",
        "-input",
        *[str(p) for p in ds["runs"]],
        "-onsets",
        str(ds["onsets"]),
        "-durations",
        "2.0",
        "-tr",
        str(ds["tr"]),
        "-prefix",
        prefix,
        "-n_hrfs",
        "5",  # small library for speed
        "-device",
        "cpu",
        "-cv_strategy",
        "loro",
    ] + extra_args
    monkeypatch.setattr(sys, "argv", argv)
    hrfopt_cli.main()


class TestDeltaDenoiseSmoke:
    @pytest.mark.slow
    def test_delta_files_written_and_finite(self, monkeypatch, tmp_path, capsys):
        ds = _make_synthetic_dataset(tmp_path)
        prefix = str(tmp_path / "out")
        _run_hrfopt(
            monkeypatch,
            prefix,
            ["-delta_denoise", "-ortvec", str(ds["ortvec"]), "motion"],
            ds,
        )

        # Primary pass files
        assert Path(f"{prefix}_xval_r2.nii.gz").exists()
        assert Path(f"{prefix}_hrf_index.nii.gz").exists()
        # Nodenoise pass files (same suite)
        assert Path(f"{prefix}_nodenoise_xval_r2.nii.gz").exists()
        assert Path(f"{prefix}_nodenoise_hrf_index.nii.gz").exists()
        # Three delta maps always present; _delta_hrfopt_r2 is only emitted
        # when both passes populated hrfopt_full_r2 (optional field).
        for name in ("delta_xval_r2", "delta_hrf_changed", "delta_hrf_index"):
            p = Path(f"{prefix}_{name}.nii.gz")
            assert p.exists(), f"{p} missing"
            data = nib.load(str(p)).get_fdata()
            assert np.isfinite(data).all(), f"{p} has non-finite values"

        # delta_hrf_changed is a 0/1 indicator
        changed = nib.load(f"{prefix}_delta_hrf_changed.nii.gz").get_fdata()
        unique_vals = set(np.unique(changed).tolist())
        assert unique_vals <= {0.0, 1.0}, f"changed map has unexpected values: {unique_vals}"

        # Summary block landed in stdout
        out = capsys.readouterr().out
        assert "Δ summary" in out
        assert "Voxels with changed HRF" in out
