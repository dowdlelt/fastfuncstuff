"""ffs_simulate end to end on a tiny problem: described and explicit timing."""

from __future__ import annotations

import csv
import json

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
    assert "scan times differ" in capsys.readouterr().out


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
