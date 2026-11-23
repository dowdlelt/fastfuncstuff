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
