"""benchmark.worklist + check planning: units, claims, records, pruning that spares originals."""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np

from fastfuncstuff.benchmark import census as C
from fastfuncstuff.benchmark.checks import CHECKS
from fastfuncstuff.benchmark.fetch import fetch
from fastfuncstuff.benchmark.worklist import Worklist


def test_add_is_idempotent_and_claims_in_order(tmp_path):
    wl = Worklist(tmp_path / "w.sqlite")
    units = [({"sub": "01", "run": "1"}, ["a.nii.gz"]), ({"sub": "02", "run": "1"}, ["b.nii.gz"])]
    assert wl.add_units("moco", "dsX", units) == 2
    assert wl.add_units("moco", "dsX", units + [({"sub": "03"}, ["c"])]) == 1  # "more subjects"
    u1 = wl.claim_next()
    u2 = wl.claim_next()
    assert json.loads(u1["unit_key"])["sub"] == "01" and u1["status"] == "running"
    assert json.loads(u2["unit_key"])["sub"] == "02"
    wl.record_metrics(u1["id"], "reference", {"r": 0.99})
    wl.record_timing(u1["id"], "ref", "cpu", 10.0, 8)
    wl.record_timing(u1["id"], "ffs", "cuda", 1.0)
    wl.finish(u1["id"])
    wl.finish(u2["id"], "failed", "boom")
    (row,) = wl.results("moco")
    assert row["reference.r"] == 0.99 and row["seconds.ref.cpu"] == 10.0
    assert wl.reset("failed") == 1
    assert {(r["status"], r["n"]) for r in wl.status()} == {("done", 1), ("pending", 2)}


def test_claims_round_robin_over_check_and_dataset(tmp_path):
    """A night cut off early must have touched every (check, dataset), not finished one."""
    wl = Worklist(tmp_path / "w.sqlite")
    wl.add_units("moco", "dsA", [({"sub": f"{i}"}, []) for i in range(3)])
    wl.add_units("moco", "dsB", [({"sub": f"{i}"}, []) for i in range(2)])
    wl.add_units("blipflip", "dsA", [({"sub": "0"}, [])])
    order = [(u["check_name"], u["dataset"]) for u in iter(wl.claim_next, None)]
    assert order[:3] == [("moco", "dsA"), ("blipflip", "dsA"), ("moco", "dsB")]
    assert len(order) == 6
    wl.add_units("newcheck", "dsA", [({"sub": "0"}, [])])
    assert wl.claim_next(known_checks=["moco"]) is None  # unknown check stays pending
    assert wl.claim_next(known_checks=["moco", "newcheck"])["check_name"] == "newcheck"


def test_new_check_first_unit_jumps_the_queue(tmp_path):
    wl = Worklist(tmp_path / "w.sqlite")
    for ds in ("dsA", "dsB", "dsC"):
        wl.add_units("moco", ds, [({"sub": f"{i}"}, []) for i in range(2)])
    wl.add_units("reml", "dsC", [({"sub": f"{i}"}, []) for i in range(2)])
    first, second = wl.claim_next(), wl.claim_next()
    assert first["check_name"] == "moco" and second["check_name"] == "reml"  # smoke unit
    rest = [(u["check_name"], u["dataset"]) for u in iter(wl.claim_next, None)]
    assert rest[:2] == [("moco", "dsB"), ("moco", "dsC")]  # then round-robin resumes


def test_prune_removes_regenerable_only_and_keeps_rows(tmp_path):
    wl = Worklist(tmp_path / "w.sqlite")
    wl.add_units("moco", "dsX", [({"sub": "01"}, [])])
    u = wl.claim_next()
    big, small = tmp_path / "big.nii", tmp_path / "small.1D"
    big.write_bytes(b"x" * 1000)
    small.write_text("1 2 3")
    wl.record_artifact(u["id"], big, "regenerable")
    wl.record_artifact(u["id"], small, "keep")
    assert wl.prune(u["id"]) == 1000
    assert not big.exists() and small.exists()
    rows = wl.db.execute("SELECT path, pruned_at FROM artifacts").fetchall()
    assert len(rows) == 2 and any(r["pruned_at"] for r in rows)


def _bids(root):
    for sub in ("01", "02", "03"):
        f = root / f"sub-{sub}" / "func"
        f.mkdir(parents=True)
        for run in ("1", "2"):
            for echo in ("1", "2"):
                p = f / f"sub-{sub}_task-rest_run-{run}_echo-{echo}_bold.nii.gz"
                nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 3), np.int16), np.eye(4)), p)
    return root


def test_breadth_plan_spreads_subjects_and_is_stable(tmp_path):
    root = _bids(tmp_path / "ds")
    src = C.LocalSource(root)
    m = C.census(src)
    idx = C.Index(src.list_files())
    check = CHECKS["moco"]()
    two = check.plan(m, idx, 2)
    assert len({k["sub"] for k, _ in two}) == 2  # distinct subjects before repeats
    assert all(inp[0].endswith("echo-1_bold.nii.gz") for _, inp in two)  # first echo only
    four = check.plan(m, idx, 4)
    assert [k for k, _ in four][:2] == [k for k, _ in two]  # growing n keeps earlier picks
    assert len({k["sub"] for k, _ in four}) == 3


def test_local_fetch_never_offers_originals_for_pruning(tmp_path):
    root = _bids(tmp_path / "ds")
    m = C.census(C.LocalSource(root))
    rel = "sub-01/func/sub-01_task-rest_run-1_echo-1_bold.nii.gz"
    paths, downloaded = fetch(m, [rel], tmp_path / "cache")
    assert paths[rel] == root / rel and downloaded == []


def test_reml_events_to_stim_times(tmp_path):
    """Fixed durations -> BLOCK(d,1), varying -> married dmBLOCK, n/a and singletons dropped."""
    from fastfuncstuff.benchmark.checks.reml import _conditions, _stim_args

    ev = tmp_path / "x_events.tsv"
    ev.write_text(
        "onset\tduration\ttrial_type\n"
        "0\t2\tface\n10\t2\tface\n"
        "5\t1\thouse\n15\t3\thouse\n"
        "20\t0\tn/a\n25\t0\tn/a\n"
        "30\t1\tlonely\n"
    )
    conds = _conditions(ev)
    assert set(conds) == {"face", "house", "event"}  # n/a pooled as 'event', singleton dropped
    args = _stim_args(conds, tmp_path)
    assert args[:2] == ["-num_stimts", "3"]
    models = {args[i + 3]: args[i + 6] for i, a in enumerate(args) if a == "-stim_times"}
    by_label = dict(zip(models.values(), models.keys(), strict=True))
    assert by_label["face"] == "BLOCK(2,1)"
    assert by_label["house"] == "dmBLOCK(1)"
    assert by_label["event"] == "GAM"
    house_file = next(p for p in tmp_path.glob("c*.1D") if ":" in p.read_text())
    assert house_file.read_text().split() == ["5.000:1.000", "15.000:3.000"]


def test_allineate_affine_distance(tmp_path):
    """A 2 mm shift reads as 2 mm RMS; the aff12 row parses as base->source 3x4."""
    from fastfuncstuff.benchmark.checks.allineate import read_aff12, rms_displacement

    f = tmp_path / "m.aff12.1D"
    f.write_text("# 3dAllineate matrix\n 1 0 0 2  0 1 0 0  0 0 1 0\n")
    m = read_aff12(f)
    assert m[0, 3] == 2 and m[3, 3] == 1
    pts = np.random.default_rng(0).normal(size=(100, 3)) * 50
    assert abs(rms_displacement(m, np.eye(4), pts) - 2.0) < 1e-9
    rot = np.eye(4)
    rot[:2, :2] = [[0, -1], [1, 0]]  # 90 degrees: displacement grows with distance
    assert rms_displacement(rot, np.eye(4), pts) > 50


def test_reml_events_non_bids_columns(tmp_path):
    from fastfuncstuff.benchmark.checks.reml import _conditions

    ev = tmp_path / "y_events.tsv"
    ev.write_text(
        "event_onset\tduration\tstimulus\n1\t20\tcheck\n60\t20\tcheck\n3\t0.4\tletter\n9\t0.4\tletter\n"
    )
    assert {k: len(v) for k, v in _conditions(ev).items()} == {"check": 2, "letter": 2}


def test_timed_cached_reuses_recorded_fit(tmp_path):
    """A rerun with the same command and outputs returns the recorded seconds, no refit."""
    from fastfuncstuff.benchmark.checks.base import timed_cached

    out, stamp, log = tmp_path / "out.txt", tmp_path / "fit.timing.json", tmp_path / "log"
    cmd = ["sh", "-c", f"echo x >> {out}"]
    t1 = timed_cached(cmd, stamp, [out], tmp_path, log)
    t2 = timed_cached(cmd, stamp, [out], tmp_path, log)
    assert t2 == t1 and out.read_text() == "x\n"
    timed_cached([*cmd[:2], cmd[2] + " "], stamp, [out], tmp_path, log)  # new command: refits
    assert out.read_text() == "x\nx\n"
