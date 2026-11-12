"""
Tests for CPU thread budgeting (utils.resolve_cpu_threads).

On a shared machine or under a batch scheduler, "how much of this box may I
use" arrives through the environment. Taking `os.cpu_count()` regardless — the
old behaviour — oversubscribes every other job on the node, so the precedence
order below is the contract.
"""

from __future__ import annotations

import os

import pytest
import torch

from fastfuncstuff import utils
from fastfuncstuff.cli_utils import setup_device
from fastfuncstuff.utils import (
    cgroup_cpu_quota,
    configure_torch_backends,
    resolve_cpu_threads,
)

THREAD_VARS = ("FFS_NUM_THREADS", "OMP_NUM_THREADS", "SLURM_CPUS_PER_TASK")


@pytest.fixture
def clean_env(monkeypatch):
    for var in THREAD_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_explicit_request_wins_over_everything(clean_env):
    clean_env.setenv("FFS_NUM_THREADS", "2")
    clean_env.setenv("OMP_NUM_THREADS", "3")
    n, source = resolve_cpu_threads(requested=7)
    assert n == 7
    assert source == "user-specified"


def test_ffs_num_threads_beats_omp(clean_env):
    clean_env.setenv("FFS_NUM_THREADS", "2")
    clean_env.setenv("OMP_NUM_THREADS", "8")
    assert resolve_cpu_threads() == (2, "$FFS_NUM_THREADS")


def test_omp_num_threads_is_honoured(clean_env):
    clean_env.setenv("OMP_NUM_THREADS", "3")
    assert resolve_cpu_threads() == (3, "$OMP_NUM_THREADS")


def test_slurm_allocation_is_honoured(clean_env):
    clean_env.setenv("SLURM_CPUS_PER_TASK", "4")
    assert resolve_cpu_threads() == (4, "$SLURM_CPUS_PER_TASK")


def test_omp_beats_slurm(clean_env):
    clean_env.setenv("OMP_NUM_THREADS", "3")
    clean_env.setenv("SLURM_CPUS_PER_TASK", "16")
    assert resolve_cpu_threads()[0] == 3


@pytest.mark.parametrize("bad", ["", "not-a-number", "0", "-4"])
def test_unparseable_or_nonsense_values_fall_through(clean_env, bad):
    """A malformed cap must not become 0 threads (which torch reads as 'all')."""
    clean_env.setenv("OMP_NUM_THREADS", bad)
    n, source = resolve_cpu_threads()
    assert n >= 1
    assert source != "$OMP_NUM_THREADS"


def test_auto_never_exceeds_the_machine(clean_env):
    n, _ = resolve_cpu_threads()
    assert 1 <= n <= (os.cpu_count() or 1)


def test_configure_torch_backends_applies_the_budget(clean_env):
    clean_env.setenv("OMP_NUM_THREADS", "2")
    configure_torch_backends(torch.device("cpu"))
    assert torch.get_num_threads() == 2


def test_setup_device_preserves_environment_thread_budget(clean_env):
    clean_env.setenv("FFS_NUM_THREADS", "3")
    setup_device("cpu")
    assert torch.get_num_threads() == 3


def test_setup_device_explicit_thread_budget_wins(clean_env):
    clean_env.setenv("FFS_NUM_THREADS", "3")
    setup_device("cpu,2")
    assert torch.get_num_threads() == 2


def _fake_cgroup_v2(tmp_path, rel, limits):
    """A cgroup v2 tree: ``limits`` maps a path under the root to cpu.max text."""
    root = tmp_path / "cg"
    (root / rel.strip("/")).mkdir(parents=True)
    for sub, text in limits.items():
        (root / sub.strip("/") / "cpu.max").write_text(text)
    proc = tmp_path / "proc_cgroup"
    proc.write_text(f"0::{rel}\n")
    return str(proc), str(root)


def test_cgroup_v2_quota_is_read(tmp_path):
    proc, root = _fake_cgroup_v2(tmp_path, "/kube/pod", {"/kube/pod": "800000 100000\n"})
    assert cgroup_cpu_quota(proc, root) == 8.0


def test_cgroup_v2_tightest_ancestor_binds(tmp_path):
    # The pod is unlimited but its parent slice is not: the parent is the limit.
    proc, root = _fake_cgroup_v2(
        tmp_path, "/kube/pod", {"/kube/pod": "max 100000\n", "/kube": "250000 100000\n"}
    )
    assert cgroup_cpu_quota(proc, root) == 2.5


def test_cgroup_unlimited_or_absent_is_none(tmp_path):
    proc, root = _fake_cgroup_v2(tmp_path, "/a", {"/a": "max 100000\n"})
    assert cgroup_cpu_quota(proc, root) is None
    assert cgroup_cpu_quota(str(tmp_path / "missing"), root) is None


def test_cgroup_v1_quota_is_read(tmp_path):
    d = tmp_path / "cg" / "cpu,cpuacct" / "docker" / "abc"
    d.mkdir(parents=True)
    (d / "cpu.cfs_quota_us").write_text("400000\n")
    (d / "cpu.cfs_period_us").write_text("100000\n")
    proc = tmp_path / "proc_cgroup"
    proc.write_text("5:cpu,cpuacct:/docker/abc\n1:name=systemd:/docker/abc\n")
    assert cgroup_cpu_quota(str(proc), str(tmp_path / "cg")) == 4.0


def test_default_thread_count_is_capped_by_the_quota(clean_env):
    # The bug of record: 64 visible cores under an 8-CPU container quota ran 64
    # threads and was throttled to a crawl. Floors, and never below one thread.
    clean_env.setattr(utils, "cgroup_cpu_quota", lambda: 2.5)
    n, source = resolve_cpu_threads()
    assert n == min(2, os.cpu_count())
    if os.cpu_count() > 2:
        assert "quota" in source
    clean_env.setattr(utils, "cgroup_cpu_quota", lambda: 0.5)
    assert resolve_cpu_threads()[0] == 1


def test_explicit_thread_choice_is_not_capped_by_the_quota(clean_env):
    clean_env.setattr(utils, "cgroup_cpu_quota", lambda: 2.0)
    clean_env.setenv("OMP_NUM_THREADS", "5")
    assert resolve_cpu_threads() == (5, "$OMP_NUM_THREADS")
    assert resolve_cpu_threads(requested=7)[0] == 7
