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
