"""Correctness tests for fastfuncstuff.stats.cluster."""
from __future__ import annotations

import numpy as np

from fastfuncstuff.stats.cluster import (
    DEFAULT_ATHR,
    DEFAULT_PTHR,
    ClusterNull,
    _cluster_extent_mass_one,
    _t_critical,
)


def test_cluster_extent_mass_simple_cube():
    """One 3x3x3 hot cube above threshold, NN1 = 27 voxels."""
    shape = (10, 10, 10)
    stat = np.zeros(shape, dtype=np.float32)
    stat[4:7, 4:7, 4:7] = 5.0
    ext, mass, labels, sizes, masses = _cluster_extent_mass_one(
        stat, tcrit=3.0, sidedness="1-sided", nn=1,
    )
    assert ext == 27
    np.testing.assert_allclose(mass, 27 * 5.0)
    assert labels.max() == 1


def test_cluster_nn_differentiates_diagonal_touch():
    """Two cubes touching at a single corner: NN1 splits, NN3 merges."""
    shape = (10, 10, 10)
    stat = np.zeros(shape, dtype=np.float32)
    stat[2:4, 2:4, 2:4] = 4.0   # cube A
    stat[4:6, 4:6, 4:6] = 4.0   # cube B, shares one corner voxel boundary
    ext1, _, lab1, sizes1, _ = _cluster_extent_mass_one(stat, 3.0, "1-sided", 1)
    ext3, _, lab3, sizes3, _ = _cluster_extent_mass_one(stat, 3.0, "1-sided", 3)
    # NN1: two separate 8-voxel cubes; NN3 merges them through the corner.
    assert sizes1.size == 2
    assert int(sizes1.sum()) == 16
    assert sizes3.size == 1
    assert int(sizes3.sum()) == 16
    assert ext1 == 8 and ext3 == 16


def test_cluster_null_table_shape_and_monotonicity():
    rng = np.random.default_rng(0)
    shape = (12, 12, 12)
    dof = 30
    n_perms = 50
    pthr = (0.05, 0.01, 0.001)
    athr = (0.10, 0.05)
    null = ClusterNull(pthr=pthr, athr=athr, nns=(1, 2, 3), sideds=("1-sided",))
    null.init_storage(n_perms)
    for p in range(n_perms):
        stat = rng.standard_normal(shape).astype(np.float32) * 1.5
        for ip, pth in enumerate(pthr):
            tcrit = _t_critical(pth, dof, "1-sided")
            for nn in (1, 2, 3):
                ext, mass, *_ = _cluster_extent_mass_one(stat, tcrit, "1-sided", nn)
                null.max_extent[("1-sided", nn)][p, ip] = ext
                null.max_mass[("1-sided", nn)][p, ip] = mass
    table = null.extent_table("1-sided", 1)
    assert table.shape == (len(pthr), len(athr))
    # Stricter alpha → larger (or equal) cluster threshold across the row.
    for i in range(table.shape[0]):
        assert table[i, 1] >= table[i, 0]


def test_default_grids_present():
    # AFNI 3dClustSim defaults
    assert len(DEFAULT_PTHR) == 29
    assert len(DEFAULT_ATHR) == 10
    assert DEFAULT_PTHR[0] == 0.10
    assert DEFAULT_ATHR[-1] == 0.01
