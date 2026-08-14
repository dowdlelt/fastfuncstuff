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

from fastfuncstuff.cli_utils import setup_device
from fastfuncstuff.utils import configure_torch_backends, resolve_cpu_threads

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
