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
    # outputs are linked, not duplicated; the worklist is a real copy, so the synced set's
    # own paths are untouched by ingest's rewrite
    linked = [f for f in out.rglob("*") if f.is_file() and "cache" not in f.parts
              and not f.name.startswith("worklist.sqlite")]  # fmt: skip
    assert linked and all(f.samefile(dest / f.relative_to(out)) for f in linked)
    assert not (out / "worklist.sqlite").samefile(dest / "worklist.sqlite")
    src_paths = [
        r["path"]
        for r in Worklist(out / "worklist.sqlite").db.execute("SELECT path FROM artifacts")
    ]
    assert not any(x.startswith(str(dest)) for x in src_paths)

    # the next set leaves out what the server already finished
    out2 = export_hostset(wl, man, tmp_path / "hs2", ["moco"], 3, "epyc")
    ids2 = {r[0] for r in Worklist(out2 / "worklist.sqlite").db.execute("SELECT id FROM units")}
    assert u["id"] not in ids2 and len(ids2) == 3
    # ...unless the next host is a new architecture, where repeating them is the point
    out3 = export_hostset(wl, man, tmp_path / "hs3", ["moco"], 4, "mac", repeat_elsewhere=True)
    ids3 = {r[0] for r in Worklist(out3 / "worklist.sqlite").db.execute("SELECT id FROM units")}
    assert u["id"] in ids3


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


def test_redo_refits_only_the_named_tool(tmp_path):
    from fastfuncstuff.benchmark.checks.base import timed_cached
    from fastfuncstuff.benchmark.work import main, redo

    root = tmp_path / "bench"
    wl = Worklist(root / "worklist.sqlite")
    wl.add_units("warp_mni", "Coh", [({"subject": "01"}, ["t1.nii.gz"])])
    u = wl.claim_next()
    work = root / "units/warp_mni/Coh" / f"unit{u['id']:06d}"
    log = work / "commands.log"
    stamps = {}
    for tool in ("sswarper", "ffs_formwarp"):
        d = work / tool
        d.mkdir(parents=True)
        out = d / "out.txt"
        cmd = ["sh", "-c", f"echo fit >> {out}"]
        timed_cached(cmd, d / "fit.timing.json", [out], work, log)
        stamps[tool] = (cmd, d / "fit.timing.json", out)
    wl.finish(u["id"], "done")

    with pytest.raises(ValueError, match="optiwarp"):
        redo(wl, root, "warp_mni", ["optiwarp"])
    assert redo(wl, root, "warp_mni", ["formwarp"]) == 1
    assert wl.db.execute("SELECT status FROM units").fetchone()[0] == "pending"
    for cmd, stamp, out in stamps.values():
        timed_cached(cmd, stamp, [out], work, log)
    assert stamps["sswarper"][2].read_text() == "fit\n"  # reference reused
    assert stamps["ffs_formwarp"][2].read_text() == "fit\nfit\n"  # changed tool refit
    # no tools: back to pending, every fit reused (only a tool added since would run)
    wl.finish(u["id"], "done")
    assert redo(wl, root, "warp_mni", []) == 1
    assert wl.db.execute("SELECT status FROM units").fetchone()[0] == "pending"
    for cmd, stamp, out in stamps.values():
        timed_cached(cmd, stamp, [out], work, log)
    assert stamps["sswarper"][2].read_text() == "fit\n"
    assert stamps["ffs_formwarp"][2].read_text() == "fit\nfit\n"
    wl.close()

    wl = Worklist(root / "worklist.sqlite")
    wl.claim_next()  # a killed run leaves it 'running'
    wl.close()
    assert main(["-root", str(root), "-reset_running"]) == 0
    wl = Worklist(root / "worklist.sqlite")
    assert wl.db.execute("SELECT status FROM units").fetchone()[0] == "pending"


def test_cohort_units_gathers_hosts_latest_run_wins(tmp_path):
    """-cohort scores a cohort spread over ingested host sets: each unit from wherever it
    finished last, here or on a host, and an excluded host's units left out."""
    from fastfuncstuff.benchmark.checks.warp_mni import cohort_units

    root = tmp_path / "bench"
    wl = Worklist(root / "worklist.sqlite")
    wl.add_units("warp_mni", "MB", [({"subject": s}, ["x"]) for s in ("a", "b", "c", "d")])
    ids = [r["id"] for r in wl.db.execute("SELECT id FROM units ORDER BY id")]
    hosts = {}
    for label, done_at in (("old", {ids[0]: 1.0, ids[1]: 1.0}), ("new", {ids[1]: 2.0, ids[2]: 2.0}),
                           ("skip", {ids[3]: 3.0})):  # fmt: skip
        hosts[label] = tmp_path / label
        hs = Worklist(hosts[label] / "worklist.sqlite")
        for uid, t in done_at.items():
            row = wl.db.execute("SELECT * FROM units WHERE id=?", (uid,)).fetchone()
            cols = row.keys()
            hs.db.execute(f"INSERT INTO units ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                          tuple(row))  # fmt: skip
            hs.db.execute("UPDATE units SET status='done', finished_at=? WHERE id=?", (t, uid))
        hs.db.commit()
        hs.close()
        wl.db.execute(
            "INSERT INTO hostsets (uuid, label, path, ingested_at) VALUES (?, ?, ?, ?)",
            (label, label, str(hosts[label]), len(hosts)),
        )
    wl.db.commit()
    got = cohort_units(wl, root, "MB", exclude_hosts=("skip",))
    unit = lambda r, i: r / "units" / "warp_mni" / "MB" / f"unit{i:06d}"  # noqa: E731
    assert got == [
        unit(hosts["old"], ids[0]),
        unit(hosts["new"], ids[1]),
        unit(hosts["new"], ids[2]),
    ]


def test_until_accepts_hhmm_and_an_iso_datetime():
    import datetime as dt

    from fastfuncstuff.benchmark.work import deadline_from

    assert deadline_from(None) is None
    t = deadline_from("07:00")
    assert t is not None and 0 < t - dt.datetime.now().timestamp() <= 86400
    assert deadline_from("2026-11-30T07:00") == dt.datetime(2026, 11, 30, 7, 0).timestamp()


def test_cpu_helpers_work_without_an_affinity_mask(monkeypatch):
    """macOS has no os.sched_getaffinity / sched_setaffinity: the runner must still count
    CPUs and skip pinning rather than crash (the Mac host sets)."""
    import os

    from fastfuncstuff.benchmark import workers
    from fastfuncstuff.benchmark.checks.base import cpu_threads

    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.delattr(os, "sched_setaffinity", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert workers.allowed_cpus() == set(range(os.cpu_count() or 1))
    workers.pin_to([0])  # no-op, no AttributeError
    assert cpu_threads() == (os.cpu_count() or 1)
    assert len(workers.cpu_cores(sysfs="/nonexistent")) == (os.cpu_count() or 1)
