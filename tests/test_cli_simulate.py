"""ffs_simulate end to end on a tiny problem: described and explicit timing."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from fastfuncstuff.cli.simulate import main


def _rows(prefix):
    with open(f"{prefix}_power.tsv") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_described_experiment(tmp_path, capsys):
    prefix = tmp_path / "er"
    rc = main(
        [
            "-tr",
            "1.25",
            "-nruns",
            "2",
            "-trial",
            "A",
            "2",
            "10",
            "-trial",
            "B",
            "2",
            "10",
            "-null",
            "2",
            "4",
            "-isi",
            "exp:4,2,10",
            "-initial_fix",
            "8",
            "-pattern",
            "A=1",
            "B=0",
            "-tsnr",
            "50",
            "-amplitudes",
            "0.5",
            "2",
            "-effect",
            "1",
            "-ndesigns",
            "2",
            "-nreps",
            "40",
            "-device",
            "cpu",
            "-no_plots",
            "-prefix",
            str(prefix),
        ]
    )
    assert rc == 0
    rows = _rows(prefix)
    assert {r["design"] for r in rows} == {"0", "1"}
    assert {r["contrast"] for r in rows} == {"A", "B", "A-B"}
    assert {float(r["amplitude"]) for r in rows} == {0.0, 0.5, 1.0, 2.0}
    spec = json.loads((tmp_path / "er_spec.json").read_text())
    assert spec["conditions"] == ["A", "B"] and spec["pattern"] == [1.0, 0.0]
    events = (tmp_path / "er_events" / "A.txt").read_text().splitlines()
    assert len(events) == 2 and len(events[0].split()) == 10
    out = capsys.readouterr().out
    assert "80% power" in out and "no true effect" in out


def test_miniblocks_and_explicit_timing_roundtrip(tmp_path):
    """The events a described run writes are valid -events input."""
    mb = tmp_path / "mb"
    assert (
        main(
            [
                "-tr",
                "2",
                "-miniblock",
                "AB",
                "A:2,B:2",
                "6",
                "-within_isi",
                "1",
                "-isi",
                "uniform:6,10",
                "-contrast",
                "A-B",
                "-pattern",
                "A=1",
                "B=0",
                "-tsnr",
                "80",
                "-ndesigns",
                "1",
                "-nreps",
                "30",
                "-device",
                "cpu",
                "-no_plots",
                "-prefix",
                str(mb),
            ]
        )
        == 0
    )
    spec = json.loads((tmp_path / "mb_spec.json").read_text())
    nt = str(spec["run_lengths"][0][0])
    ex = tmp_path / "ex"
    assert (
        main(
            [
                "-tr",
                "2",
                "-events",
                str(tmp_path / "mb_events" / "A.txt"),
                str(tmp_path / "mb_events" / "B.txt"),
                "-durations",
                "2",
                "-nt",
                nt,
                "-contrast",
                "A-B",
                "-pattern",
                "A=1",
                "B=0",
                "-tsnr",
                "80",
                "-nreps",
                "30",
                "-device",
                "cpu",
                "-no_plots",
                "-prefix",
                str(ex),
            ]
        )
        == 0
    )
    a = {(r["amplitude"], r["contrast"]): r["sd_predicted"] for r in _rows(mb)}
    b = {(r["amplitude"], r["contrast"]): r["sd_predicted"] for r in _rows(ex)}
    for key in a:  # same events -> same design -> same analytic precision
        assert float(a[key]) == pytest.approx(float(b[key]), rel=1e-3)


def test_described_and_explicit_are_exclusive(tmp_path, capsys):
    assert main(["-tr", "2", "-prefix", str(tmp_path / "x"), "-device", "cpu"]) == 1
    assert "either -events or" in capsys.readouterr().err


def test_scan_time_sets_the_volumes(tmp_path):
    prefix = tmp_path / "st"
    assert (
        main(
            [
                "-tr",
                "2",
                "-trial",
                "A",
                "2",
                "1",
                "-block",
                "blk",
                "16",
                "1",
                "-num_blocks",
                "3",
                "-scan_time",
                "240",
                "-isi",
                "exp:4,2,10",
                "-tsnr",
                "50",
                "-amplitudes",
                "1",
                "-ndesigns",
                "2",
                "-nreps",
                "20",
                "-device",
                "cpu",
                "-no_plots",
                "-prefix",
                str(prefix),
            ]
        )
        == 0
    )
    spec = json.loads((tmp_path / "st_spec.json").read_text())
    assert spec["run_lengths"] == [[120], [120]]
    events = (tmp_path / "st_events" / "blk.txt").read_text().split()
    assert len(events) == 3


def test_true_hrf_sweep_reports_recovery(tmp_path, capsys):
    prefix = tmp_path / "lib"
    assert (
        main(
            [
                "-tr",
                "2",
                "-trial",
                "A",
                "2",
                "12",
                "-isi",
                "exp:5,3,10",
                "-tsnr",
                "80",
                "-amplitudes",
                "1",
                "2",
                "-true_hrf",
                "lib:0",
                "-ndesigns",
                "1",
                "-nreps",
                "30",
                "-device",
                "cpu",
                "-no_plots",
                "-prefix",
                str(prefix),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "data generated with lib:0" in out and "Recovered fraction" in out
    rows = _rows(prefix)
    assert {r["true_hrf"] for r in rows} == {"lib:0"}
    top = [r for r in rows if r["amplitude"] == "2.0"]
    assert float(top[0]["expected_est"]) < 0.6 * float(top[0]["true_effect"])


def test_bad_hrf_spec_fails_early(tmp_path, capsys):
    assert (
        main(
            [
                "-tr",
                "2",
                "-trial",
                "A",
                "2",
                "5",
                "-hrf",
                "lib:99",
                "-device",
                "cpu",
                "-prefix",
                str(tmp_path / "x"),
            ]
        )
        == 1
    )
    assert "0-19" in capsys.readouterr().err


def _sim(prefix, *extra):
    base = [
        "-tr",
        "2",
        "-trial",
        "A",
        "2",
        "1",
        "-trial",
        "B",
        "2",
        "1",
        "-isi",
        "exp:4,2,10",
        "-pattern",
        "A=1",
        "B=0",
        "-tsnr",
        "40",
        "90",
        "-amplitudes",
        "0.5",
        "1",
        "2",
        "4",
        "-nreps",
        "40",
        "-device",
        "cpu",
        "-prefix",
        str(prefix),
    ]
    assert main(base + list(extra)) == 0


def test_compare_summarises_each_design_over_its_realizations(tmp_path, capsys):
    _sim(tmp_path / "a", "-scan_time", "200", "-ndesigns", "3", "-no_plots")
    _sim(tmp_path / "b", "-scan_time", "200", "-ndesigns", "2", "-no_plots")
    _sim(tmp_path / "c", "-scan_time", "300", "-ndesigns", "1", "-no_plots")
    capsys.readouterr()
    assert (
        main(
            [
                "-compare",
                str(tmp_path / "a_power.tsv"),
                str(tmp_path / "b_power.tsv"),
                "-compare_names",
                "first",
                "second",
                "-prefix",
                str(tmp_path / "cmp"),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "first" in out and "second" in out and "WARNING" not in out
    assert (tmp_path / "cmp_compare.txt").exists()
    assert (tmp_path / "cmp_compare_A.png").exists()
    assert not (tmp_path / "cmp_compare_B.png").exists()  # B has no true effect
    # a longer scan is flagged
    assert (
        main(
            [
                "-compare",
                str(tmp_path / "a_power.tsv"),
                str(tmp_path / "c_power.tsv"),
                "-no_plots",
                "-prefix",
                str(tmp_path / "cmp2"),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "scan times differ" in out
    # ... and the per-minute table puts them on one footing: effect x sqrt(min)
    assert "effect x sqrt(total minutes)" in out
    from fastfuncstuff.simulation.power import compare_designs, load_power_table

    rows = compare_designs(
        {
            "a": load_power_table(tmp_path / "a_power.tsv"),
            "c": load_power_table(tmp_path / "c_power.tsv"),
        }
    )
    r = next(x for x in rows if x["design"] == "c" and x["contrast"] == "A")
    assert r["per_minute"] == pytest.approx(r["median"] * np.sqrt(r["scan_s"] / 60))


def test_compare_needs_matching_names(tmp_path, capsys):
    _sim(tmp_path / "a", "-ndesigns", "1", "-no_plots")
    assert (
        main(
            [
                "-compare",
                str(tmp_path / "a_power.tsv"),
                "-compare_names",
                "x",
                "y",
                "-prefix",
                str(tmp_path / "cmp"),
            ]
        )
        == 1
    )


def test_hrf_figure_written_only_under_mismatch(tmp_path):
    _sim(tmp_path / "same", "-ndesigns", "1")
    assert not (tmp_path / "same_hrf.png").exists()
    _sim(tmp_path / "lib", "-ndesigns", "1", "-true_hrf", "lib:3")
    assert (tmp_path / "lib_hrf.png").stat().st_size > 10_000


def test_tr_required_when_simulating(tmp_path, capsys):
    assert main(["-trial", "A", "2", "5", "-prefix", str(tmp_path / "x")]) == 1
    assert "-tr is required" in capsys.readouterr().err


def test_shared_is_reported_and_recorded(tmp_path, capsys):
    _sim(tmp_path / "sh", "-contrast", "A-B", "-shared", "3", "-ndesigns", "1", "-no_plots")
    out = capsys.readouterr().out
    assert "sweep = the difference itself" in out and "3% shared" in out
    spec = json.loads((tmp_path / "sh_spec.json").read_text())
    assert spec["shared"] == 3.0
    rows = _rows(tmp_path / "sh")
    assert {r["swept"] for r in rows} == {"difference"}


def test_effect_verdict_follows_monte_carlo_under_mismatch():
    from fastfuncstuff.cli.simulate import _effect_cell

    # The analytic curve overstated Monte Carlo 13x under a wrong HRF; a verdict
    # read off it called a hopeless design good.
    sel = [{"true_effect": 1.0, "power_predicted": 0.9, "power": 0.05}]
    assert _effect_cell(sel, "power").endswith("hopeless")
    assert _effect_cell(sel, "power_predicted").endswith("good")
    assert _effect_cell([{**sel[0], "true_effect": 0.0}], "power") == "no true effect"


def test_example_voxels_show_the_effect_each_level_needs():
    from fastfuncstuff.cli.simulate import _voxel_amplitudes

    # A fixed 1% default was undetectable at tSNR 50 and invisible at 20: each
    # row now plants what that level needs, and says when it never gets there.
    rows = []
    for noise, (p1, p2) in {"t20": (0.1, 0.3), "t100": (0.9, 1.0)}.items():
        for amp, pw in ((0.0, 0.0), (1.0, p1), (2.0, p2)):
            rows.append(
                {
                    "design": 0,
                    "true_hrf": "",
                    "noise": noise,
                    "contrast": "A",
                    "amplitude": amp,
                    "true_effect": amp,
                    "expected_est": amp,
                    "power": pw,
                    "power_predicted": pw,
                }
            )
    conds = [{"label": "t20"}, {"label": "t100"}]
    amps, basis, notes = _voxel_amplitudes({"table": rows}, conds, {"A": [1.0]}, [1.0], None)
    assert amps[0] == 2.0 and "not reached" in notes[0]  # top of the sweep
    assert 0 < amps[1] < 1.0 and notes[1] == ""
    assert "80% power" in basis
    amps, basis, notes = _voxel_amplitudes({"table": rows}, conds, {"A": [1.0]}, [1.0], 1.5)
    assert amps == [1.5, 1.5] and "-effect" in basis and notes is None


def test_rank_deficient_design_is_an_error_not_a_traceback(tmp_path, capsys):
    for c in ("A", "B"):
        (tmp_path / f"{c}.txt").write_text("10 40 70 100 130\n")
    argv = [
        "-tr", "2", "-events", str(tmp_path / "A.txt"), str(tmp_path / "B.txt"),
        "-durations", "2", "-nt", "90", "-nreps", "10", "-device", "cpu", "-no_plots",
        "-prefix", str(tmp_path / "rd"),
    ]  # fmt: skip
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert "rank-deficient" in err and "*A" in err and "*B" in err


def test_summary_reports_design_quality(tmp_path, capsys):
    _sim(tmp_path / "q", "-ndesigns", "2", "-no_plots")
    out = capsys.readouterr().out
    assert "Design quality" in out and "VIF" in out
    # one pair: "hardest" and "easiest" would name the same pair
    assert "hardest to tell apart" not in out


def test_scan_times_sweep_writes_its_table_and_figure(tmp_path, capsys):
    blocks = ["-block", "E1", "30", "1", "-block", "E2", "30", "1", "-isi", "10"]
    base = ["-tr", "1", "-nruns", "2", "-scan_time", "330", *blocks, "-tsnr", "60",
            "-ndesigns", "2", "-nreps", "20", "-device", "cpu"]  # fmt: skip
    assert main([*base, "-scan_times", "240", "480", "-prefix", str(tmp_path / "s")]) == 0
    out = capsys.readouterr().out
    assert "How long to scan" in out and "most per minute" in out
    assert (tmp_path / "s_scantime.png").stat().st_size > 10_000
    assert len((tmp_path / "s_scantime.tsv").read_text().splitlines()) > 1
    # fixed counts: longer runs would only add fixation
    argv = [*base, "-num_blocks", "4", "-scan_times", "240", "-prefix", str(tmp_path / "f")]
    assert main(argv) == 1
    assert "only add fixation" in capsys.readouterr().err


def test_null_count_can_be_a_share_of_the_trials():
    from fastfuncstuff.cli.simulate import _build_parser, _null_fraction, _spec_from_args

    assert _null_fraction("20%") == 0.2 and _null_fraction("0.25") == 0.25
    assert _null_fraction("15") is None  # a count
    with pytest.raises(ValueError, match="between 0 and 1"):
        _null_fraction("100%")
    assert _null_fraction("0%") == 0.0  # no blank trials (an explored share can draw it)
    # "identical ISIs, then drop trials": a 3 s train with a fifth of its slots blank
    argv = ["-tr", "1", "-trial", "A", "0.25", "60", "-isi", "3", "-null", "0.25", "20%",
            "-prefix", "x"]  # fmt: skip
    counts = _spec_from_args(_build_parser().parse_args(argv)).resolve_counts()
    assert counts == [60, 15]
