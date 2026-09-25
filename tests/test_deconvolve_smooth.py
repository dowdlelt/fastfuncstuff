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


def test_ffs_tps_is_a_smoothed_csplin_preset_with_passthrough(tmp_path, capsys):
    from fastfuncstuff.cli import tps

    runs, timing = _dataset(tmp_path)
    prefix = str(tmp_path / "tps")
    rc = tps.main(
        [
            "-input",
            *runs,
            "-stim-times",
            timing,
            "-stim-labels",
            "face",
            "-tps-window",
            "0",
            "16",
            "-output-prefix",
            prefix,
            "-penalty",
            "gp:3",
            "-verb",
            "1",
            "-device",
            "cpu",
            "-save-xval-r2",
        ]
    )
    assert rc == 0
    assert "Smoothing (reml, gp:3)" in capsys.readouterr().out
    for name in ("iresp_face", "smooth_edf", "xval_r2"):
        assert np.isfinite(nib.load(f"{prefix}_{name}.nii.gz").get_fdata()).all()


def test_smooth_noise_arma_writes_the_ab_map(monkeypatch, tmp_path, capsys):
    from fastfuncstuff.cli import deconvolve

    runs, timing = _dataset(tmp_path)
    prefix = str(tmp_path / "ar")
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
        "-tent-smooth",
        "-smooth-noise",
        "arma",
        "-save-xval-r2",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert deconvolve.main() == 0
    ab = nib.load(f"{prefix}_smooth_arma.nii.gz").get_fdata()
    assert ab.shape[-1] == 2 and np.isfinite(ab).all()
    assert np.isfinite(nib.load(f"{prefix}_xval_r2.nii.gz").get_fdata()).all()


def _per_run_argv(runs, timing, prefix, *extra):
    return [
        "ffs_deconvolve",
        "-input",
        *runs,
        "-onsets",
        timing,
        "-model",
        "TENTzero",
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
        "-per-run",
        *extra,
    ]


def test_per_run_writes_one_curve_per_run_with_se_and_xval(monkeypatch, tmp_path, capsys):
    from fastfuncstuff.cli import deconvolve

    runs, timing = _dataset(tmp_path)
    prefix = str(tmp_path / "pr")
    argv = _per_run_argv(
        runs, timing, prefix, "-per-run-se", "-save-xval-r2", "-smooth-noise", "arma"
    )
    monkeypatch.setattr(sys, "argv", argv)
    assert deconvolve.main() == 0
    out = capsys.readouterr().out
    assert "one run predicting the others (fold lambda)" in out
    pooled = nib.load(f"{prefix}_iresp_stim.nii.gz").get_fdata()
    per_run = np.stack(
        [nib.load(f"{prefix}_perrun_iresp_stim_run0{r}.nii.gz").get_fdata() for r in (1, 2, 3)],
        axis=3,
    )
    se = np.stack(
        [nib.load(f"{prefix}_perrun_se_stim_run0{r}.nii.gz").get_fdata() for r in (1, 2, 3)],
        axis=3,
    )
    assert per_run.shape == (*pooled.shape[:3], 3, pooled.shape[-1]) == se.shape
    # zero edges padded back like the pooled iresp; the runs average near the pooled curve
    assert np.all(per_run[..., 0] == 0) and np.all(per_run[..., -1] == 0)
    assert np.corrcoef(per_run.mean(axis=3).ravel(), pooled.ravel())[0, 1] > 0.9
    assert (se[..., 1:-1] > 0).all()
    smooth_r2 = nib.load(f"{prefix}_perrun_xval_r2.nii.gz").get_fdata()
    ols_r2 = nib.load(f"{prefix}_perrun_xval_r2_ols.nii.gz").get_fdata()
    assert np.median(smooth_r2) > np.median(ols_r2)


def test_per_run_lambda_run_mode_writes_its_lambda_map(monkeypatch, tmp_path):
    from fastfuncstuff.cli import deconvolve

    runs, timing = _dataset(tmp_path)
    prefix = str(tmp_path / "prr")
    monkeypatch.setattr(sys, "argv", _per_run_argv(runs, timing, prefix, "-per-run-lambda", "run"))
    assert deconvolve.main() == 0
    lam = nib.load(f"{prefix}_perrun_log10lambda.nii.gz").get_fdata()
    assert lam.shape[-1] == 3 and np.isfinite(lam).all()


def test_per_run_flags_need_per_run_and_smoothing(monkeypatch, tmp_path, capsys):
    from fastfuncstuff.cli import deconvolve

    runs, timing = _dataset(tmp_path)
    argv = _per_run_argv(runs, timing, str(tmp_path / "x"))
    argv.remove("-tent-smooth")
    monkeypatch.setattr(sys, "argv", argv)
    assert deconvolve.main() == 1
    assert "-per-run needs -tent-smooth" in capsys.readouterr().err
    argv = _per_run_argv(runs, timing, str(tmp_path / "y"), "-per-run-se")
    argv.remove("-per-run")
    monkeypatch.setattr(sys, "argv", argv)
    assert deconvolve.main() == 1
    assert "need -per-run" in capsys.readouterr().err


def test_run_tags_widen_past_99_runs_so_names_sort():
    from fastfuncstuff.cli.deconvolve import _run_tags

    assert _run_tags(3) == ["run01", "run02", "run03"]
    tags = _run_tags(120)
    assert tags[0] == "run001" and tags[-1] == "run120" and tags == sorted(tags)


def test_global_rule_fits_every_voxel_with_one_lambda(monkeypatch, tmp_path, capsys):
    from fastfuncstuff.cli import deconvolve

    runs, timing = _dataset(tmp_path)
    prefix = str(tmp_path / "glob")
    argv = _per_run_argv(runs, timing, prefix, "-per-run-lambda", "global")
    argv.insert(argv.index("-tent-smooth") + 1, "global")
    monkeypatch.setattr(sys, "argv", argv)
    assert deconvolve.main() == 0
    assert "Global lambda: 10^" in capsys.readouterr().out
    lam = nib.load(f"{prefix}_smooth_log10lambda.nii.gz").get_fdata()
    assert np.ptp(lam) < 1e-5
