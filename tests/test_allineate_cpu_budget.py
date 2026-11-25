"""CPU budget rules in ffs_allineate: on the CPU each candidate is paid for, so the
search spends fewer of them than on a GPU, where a batch costs what one does."""

from __future__ import annotations

import importlib

import torch

al = importlib.import_module("fastfuncstuff.processing.allineate")

CPU = torch.device("cpu")


def test_merged_trials_stop_the_worse_of_a_pair_in_one_basin():
    mean = torch.tensor([[0.50, 0.50], [0.505, 0.49], [0.80, 0.20], [0.81, 0.21]])
    best = torch.tensor([0.40, 0.45, 0.30, 0.30])  # 1 beats 0; 2 and 3 tie
    alive = torch.ones(4, dtype=torch.bool)
    assert al._merged_trials(mean, best, alive).tolist() == [True, False, False, True]
    # a stopped trial cannot retire a live one
    alive[1] = False
    assert al._merged_trials(mean, best, alive).tolist() == [False, False, False, True]


def test_cpu_match_points_follow_afni_ntask():
    """sqrt(nvox_source * n_domain), then 47% (3dAllineate.c:4371): a 3 mm EPI is not
    sampled at the density of a 1 mm base grid."""
    base = torch.ones(60, 60, 60)
    n_dom = base.numel()
    s = al._build_sample_set(base, None, (1.0, 1.0, 1.0), 0.47, "tohd", CPU, n_source_vox=20000)
    expected = int(0.47 * int((20000 * n_dom) ** 0.5))
    assert s is not None and s.idx_flat.numel() == expected
    # a source with more voxels than the domain keeps the domain rule
    s = al._build_sample_set(base, None, (1.0, 1.0, 1.0), 0.47, "tohd", CPU, n_source_vox=10**7)
    assert s is not None and s.idx_flat.numel() == int(0.47 * n_dom)


def test_batch_is_free_only_off_the_cpu():
    assert not al._batch_is_free(CPU)
    assert al._batch_is_free(torch.device("meta"))


def test_keep_freed_memory_is_idempotent():
    from fastfuncstuff.utils import keep_freed_memory_in_process

    first = keep_freed_memory_in_process()
    assert keep_freed_memory_in_process() == first
