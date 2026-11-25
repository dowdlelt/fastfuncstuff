"""benchmark.workers: core grouping and slots for parallel workers; shared-cache pruning."""

from __future__ import annotations

from pathlib import Path

import pytest

from fastfuncstuff.benchmark.workers import core_slots, cpu_cores, default_workers, thread_env
from fastfuncstuff.benchmark.worklist import Worklist


def _sysfs(tmp_path, n_cores, smt):
    """Linux numbering: logical c and c + n_cores are siblings of physical core c."""
    for core in range(n_cores):
        sibs = ",".join(str(core + k * n_cores) for k in range(smt))
        for k in range(smt):
            d = tmp_path / f"cpu{core + k * n_cores}" / "topology"
            d.mkdir(parents=True)
            (d / "thread_siblings_list").write_text(sibs + "\n")
    return str(tmp_path)


def test_cores_group_smt_siblings_and_slots_are_disjoint(tmp_path):
    sysfs = _sysfs(tmp_path, n_cores=8, smt=2)
    cores = cpu_cores(set(range(16)), sysfs)
    assert cores[:2] == [[0, 8], [1, 9]] and len(cores) == 8
    slots = core_slots(cores, workers=2, threads=3)
    assert slots == [[0, 1, 2, 8, 9, 10], [3, 4, 5, 11, 12, 13]]
    with pytest.raises(ValueError, match="physical cores"):
        core_slots(cores, workers=3, threads=3)


def test_restricted_affinity_and_missing_topology(tmp_path):
    sysfs = _sysfs(tmp_path, n_cores=4, smt=2)
    assert cpu_cores({0, 4, 1}, sysfs) == [[0, 4], [1]]  # only allowed CPUs are handed out
    assert cpu_cores({0, 1}, str(tmp_path / "nowhere")) == [[0], [1]]


def test_default_workers_respects_the_cpu_quota():
    cores = [[i] for i in range(128)]  # a k8s node shows every core...
    assert default_workers(cores, 6, quota=32.0) == 5  # ...but the pod gets 32 CPUs of time
    assert default_workers(cores, 6, quota=None) == 21
    assert default_workers(cores[:4], 6, quota=None) == 1
    assert thread_env(6)["OMP_NUM_THREADS"] == thread_env(6)["FFS_NUM_THREADS"] == "6"


def test_prune_spares_cache_files_another_running_unit_reads(tmp_path):
    wl = Worklist(tmp_path / "w.sqlite")
    wl.add_units("moco", "dsX", [({"sub": "01"}, ["sub-01/func/bold.nii.gz"])])
    wl.add_units("slicetime", "dsX", [({"sub": "01"}, ["sub-01/func/bold.nii.gz"])])
    a, b = wl.claim_next(), wl.claim_next()
    f = tmp_path / "cache" / "dsX" / "sub-01" / "func" / "bold.nii.gz"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x" * 100)
    for u in (a, b):
        wl.record_artifact(u["id"], f, "regenerable")
    wl.finish(a["id"], "done")
    assert wl.prune(a["id"]) == 0 and f.exists()  # b still reads it
    wl.finish(b["id"], "done")
    assert wl.prune(b["id"]) == 100 and not f.exists()  # the last user removes it


def _manifest(d, ds, kind):
    import yaml

    uri = f"s3://openneuro.org/{ds}" if kind == "s3" else "/private/tree"
    (d / f"{ds}.yaml").write_text(
        yaml.safe_dump({"dataset_id": ds, "source": {"kind": kind, "uri": uri}, "acquisitions": []})
    )


def test_hostset_export_run_ingest_round_trip(tmp_path, capsys):
    from fastfuncstuff.benchmark.hostset import export_hostset, hostsets, ingest

    root, man = tmp_path / "bench", tmp_path / "bench" / "manifests"
    man.mkdir(parents=True)
    for ds, kind in (("dsA", "s3"), ("dsB", "s3"), ("Private", "local")):
        _manifest(man, ds, kind)
    wl = Worklist(root / "worklist.sqlite")
    for ds in ("dsA", "dsB", "Private"):
        wl.add_units("moco", ds, [({"sub": s}, [f"sub-{s}/bold.nii.gz"]) for s in ("01", "02")])
    while (u := wl.claim_next()) is not None:
        wl.finish(u["id"], "done")
    wl.add_units("moco", "dsA", [({"sub": "03"}, ["x"])])  # pending: not exported

    out = export_hostset(wl, man, tmp_path / "hs", ["moco"], 3, "epyc")
    hs = Worklist(out / "worklist.sqlite")
    rows = hs.db.execute("SELECT id, dataset, status FROM units ORDER BY id").fetchall()
    assert [r["dataset"] for r in rows].count("dsA") == 2 and len(rows) == 3  # spread, no Private
    assert all(r["status"] == "pending" for r in rows)
    assert sorted(p.name for p in (out / "manifests").iterdir()) == ["dsA.yaml", "dsB.yaml"]
    assert "Private" in capsys.readouterr().out

    # "the server": run one unit, keep its output, drop its downloaded input
    u = hs.claim_next()
    unit_out = Path("/server/bench/units/moco") / u["dataset"] / "unit1" / "ffs" / "moco.nii"
    hs.record_timing(u["id"], "ffs", "cpu", 12.5, 6)
    hs.record_artifact(u["id"], unit_out, "regenerable")
    hs.record_provenance(u["id"], "worker", {"hostname": "epyc01"})
    cached = out / "cache" / u["dataset"] / "bold.nii.gz"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"x")
    hs.record_artifact(u["id"], cached, "regenerable")
    hs.finish(u["id"], "done")
    hs.prune(u["id"], under=out / "cache")
    assert not cached.exists()
    hs.close()

    for _ in range(2):  # re-ingest replaces, never duplicates
        dest = ingest(wl, root, out)
    (h,) = hostsets(wl)
    assert (h["label"], h["hostname"], h["n_done"]) == ("epyc", "epyc01", 1)
    got = Worklist(dest / "worklist.sqlite")
    paths = [r["path"] for r in got.db.execute("SELECT path FROM artifacts")]
    assert str(dest / "units/moco" / u["dataset"] / "unit1/ffs/moco.nii") in paths
    (res,) = got.results("moco")
    assert res["unit_id"] == u["id"] and res["seconds.ffs.cpu"] == 12.5
    got.close()

    # the next set leaves out what the server already finished
    out2 = export_hostset(wl, man, tmp_path / "hs2", ["moco"], 3, "epyc")
    ids2 = {r[0] for r in Worklist(out2 / "worklist.sqlite").db.execute("SELECT id FROM units")}
    assert u["id"] not in ids2 and len(ids2) == 3


def test_hostset_bundles_local_inputs_and_pending_units(tmp_path, monkeypatch):
    import yaml

    from fastfuncstuff.benchmark.fetch import fetch
    from fastfuncstuff.benchmark.hostset import export_hostset

    tree = tmp_path / "private"
    (tree / "sub-01").mkdir(parents=True)
    (tree / "sub-01" / "t1.nii.gz").write_bytes(b"img")
    (tree / "sub-01" / "t1.json").write_text("{}")
    (tree / "sub-01" / "other.nii.gz").write_bytes(b"not an input")
    root, man = tmp_path / "bench", tmp_path / "bench" / "manifests"
    man.mkdir(parents=True)
    (man / "Coh.yaml").write_text(yaml.safe_dump(
        {"dataset_id": "Coh", "source": {"kind": "local", "uri": str(tree)}, "acquisitions": []}))  # fmt: skip
    wl = Worklist(root / "worklist.sqlite")
    wl.add_units("warp_mni", "Coh", [({"subject": "01"}, ["sub-01/t1.nii.gz"])])  # pending

    out = export_hostset(wl, man, tmp_path / "hs", ["warp_mni"], 2, "srv", bundle_local=True,
                         include_pending=True, prune_outputs=True)  # fmt: skip
    assert Worklist(out / "worklist.sqlite").meta("prune_outputs") == "yes"
    assert sorted(p.name for p in (out / "data" / "Coh" / "sub-01").iterdir()) == [
        "t1.json",
        "t1.nii.gz",
    ]
    m = yaml.safe_load((out / "manifests" / "Coh.yaml").read_text())
    assert m["source"]["uri"] == "${FFS_BENCH_ROOT}/data/Coh"
    monkeypatch.setenv("FFS_BENCH_ROOT", str(out))  # as -run sets it on the other host
    paths, downloaded = fetch(m, ["sub-01/t1.nii.gz"], out / "cache")
    assert paths["sub-01/t1.nii.gz"].read_bytes() == b"img" and downloaded == []
