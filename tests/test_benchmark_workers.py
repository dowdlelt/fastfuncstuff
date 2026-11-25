"""benchmark.workers: core grouping and slots for parallel workers; shared-cache pruning."""

from __future__ import annotations

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
