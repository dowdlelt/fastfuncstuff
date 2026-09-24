"""ffs_deconvolve -tent-smooth end to end on timing plain TENT cannot resolve."""

import sys

import nibabel as nib
import numpy as np
from scipy.stats import gamma


def _dataset(tmp_path):
    rng = np.random.default_rng(3)
    tr, n_tp = 2.0, 120
    t = np.arange(n_tp) * tr
    runs, rows = [], []
    for r in range(3):
        onsets = np.sort(rng.choice(np.arange(3, 105), 14, replace=False)) * tr + 1.0  # mid-TR
        hrf = sum(
            np.where(t - o > 0, gamma.pdf(t - o, 6) - gamma.pdf(t - o, 16) / 6, 0) for o in onsets
        )
        img = nib.Nifti1Image(
            (100 + rng.normal(0, 1, (4, 4, 2, n_tp)) + 10 * hrf).astype(np.float32), np.eye(4)
        )
        img.header.set_zooms((2, 2, 2, tr))
        img.header.set_xyzt_units("mm", "sec")
        path = tmp_path / f"run{r}.nii.gz"
        nib.save(img, path)
        runs.append(str(path))
        rows.append(" ".join(f"{o:.2f}" for o in onsets))
    timing = tmp_path / "stim.1D"
    timing.write_text("\n".join(rows) + "\n")
    return runs, str(timing)


def test_tent_smooth_writes_maps_and_smoothed_cross_validation(monkeypatch, tmp_path, capsys):
    from fastfuncstuff.cli import deconvolve

    runs, timing = _dataset(tmp_path)
    prefix = str(tmp_path / "out")
    argv = [
        "ffs_deconvolve",
        "-input",
        *runs,
        "-onsets",
        timing,
        "-model",
        "TENT",
        "-window",
        "0",
        "16",
        "-prefix",
        prefix,
        "-device",
        "cpu",
        "-verb",
        "1",
        "-save-xval-r2",
        "-fir-smooth",
    ]  # -fir-smooth is the same flag
    monkeypatch.setattr(sys, "argv", argv)
    assert deconvolve.main() == 0
    out = capsys.readouterr().out
    assert "Smoothing (reml, diff2)" in out and "roughness penalty covers it" in out
    for name in ("smooth_edf", "smooth_log10lambda", "xval_r2"):
        vol = nib.load(f"{prefix}_{name}.nii.gz").get_fdata()
        assert np.isfinite(vol).all()
    edf = nib.load(f"{prefix}_smooth_edf.nii.gz").get_fdata()
    assert 2.0 <= edf.min() and edf.max() <= 9.0


def test_plain_tent_warns_that_mid_tr_timing_is_singular(monkeypatch, tmp_path, capsys):
    from fastfuncstuff.cli import deconvolve

    runs, timing = _dataset(tmp_path)
    argv = [
        "ffs_deconvolve",
        "-input",
        *runs,
        "-onsets",
        timing,
        "-model",
        "TENT",
        "-window",
        "0",
        "16",
        "-prefix",
        str(tmp_path / "plain"),
        "-device",
        "cpu",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert deconvolve.main() == 0
    assert "SINGULAR" in capsys.readouterr().err


def test_explicit_window_is_not_snapped_so_aligned_knots_stay_on_the_samples(
    monkeypatch, tmp_path, capsys
):
    """-window 1 17 at TR 2 used to become 1-16 (round(8.5) == 8), knocking the
    knots aligned to mid-TR samples off them again."""
    from fastfuncstuff.cli import deconvolve

    runs, timing = _dataset(tmp_path)
    argv = [
        "ffs_deconvolve",
        "-input",
        *runs,
        "-onsets",
        timing,
        "-model",
        "TENT",
        "-window",
        "1",
        "17",
        "-tent-n-basis",
        "9",
        "-prefix",
        str(tmp_path / "al"),
        "-device",
        "cpu",
        "-verb",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert deconvolve.main() == 0
    captured = capsys.readouterr()
    assert "1.0s–17.0s" in captured.out
    assert "SINGULAR" not in captured.err


def test_loro_rule_with_two_penalties_writes_the_choice_map(monkeypatch, tmp_path, capsys):
    from fastfuncstuff.cli import deconvolve

    runs, timing = _dataset(tmp_path)
    prefix = str(tmp_path / "sel")
    argv = [
        "ffs_deconvolve",
        "-input",
        *runs,
        "-onsets",
        timing,
        "-model",
        "TENT",
        "-window",
        "0",
        "16",
        "-prefix",
        prefix,
        "-device",
        "cpu",
        "-verb",
        "1",
        "-tent-smooth",
        "loro",
        "-smooth-penalty",
        "diff2,gp:3",
        "-save-xval-r2",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert deconvolve.main() == 0
    out = capsys.readouterr().out
    assert "Penalty chosen by held-out runs" in out and "optimistically biased" in out
    choice = nib.load(f"{prefix}_smooth_penalty.nii.gz").get_fdata()
    assert set(np.unique(choice)) <= {0.0, 1.0}


def test_pool_conditions_fits_one_average_response(monkeypatch, tmp_path, capsys):
    from fastfuncstuff.cli import deconvolve

    runs, timing = _dataset(tmp_path)
    rows = (tmp_path / "stim.1D").read_text().splitlines()
    half = [" ".join(r.split()[::2]) for r in rows]
    other = [" ".join(r.split()[1::2]) for r in rows]
    (tmp_path / "a.1D").write_text("\n".join(half) + "\n")
    (tmp_path / "b.1D").write_text("\n".join(other) + "\n")
    prefix = str(tmp_path / "pool")
    argv = [
        "ffs_deconvolve",
        "-input",
        *runs,
        "-onsets",
        str(tmp_path / "a.1D"),
        str(tmp_path / "b.1D"),
        "-model",
        "TENT",
        "-window",
        "1",
        "17",
        "-tent-n-basis",
        "9",
        "-prefix",
        prefix,
        "-device",
        "cpu",
        "-verb",
        "1",
        "-pool-conditions",
        "face",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert deconvolve.main() == 0
    assert "Pooled every condition into 'face' (42 events)" in capsys.readouterr().out
    assert nib.load(f"{prefix}_iresp_face.nii.gz").shape[-1] == 9
    assert not list(tmp_path.glob("pool_iresp_a*"))


def test_pool_timing_merges_runs_and_keeps_the_longest_duration():
    from fastfuncstuff.cli_utils import TimingSpec, pool_timing

    timing = TimingSpec(
        all_onsets=[[np.array([5.0, 1.0]), np.array([3.0])], [np.array([2.0]), np.array([])]],
        durations=[1.0, 2.0],
        condition_labels=["a", "b"],
        from_events=True,
    )
    pooled, note = pool_timing(timing, "all")
    assert pooled.condition_labels == ["all"] and pooled.durations == [2.0]
    assert [o.tolist() for o in pooled.all_onsets[0]] == [[1.0, 2.0, 5.0], [3.0]]
    assert note is not None and "2 s" in note
