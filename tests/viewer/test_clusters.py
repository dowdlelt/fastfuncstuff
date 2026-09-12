"""Clusterize: the table has to describe the picture beside it.

A cluster table is read as a result -- sizes get quoted, peaks get reported --
so the ways it can be wrong are all quiet. These pin the ones that would not
look like bugs: a blob that straddles zero counted as one finding, a negative
peak reported as a positive number, a row numbered differently from the voxels
it stands for, and an alpha invented for a dataset that carries no table.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastfuncstuff.viewer.clusters import clusterize, rois_from_clusters
from fastfuncstuff.viewer.layers import SignMode


def _two_blobs():
    """A big positive blob, a small negative one, and nothing between them."""
    v = np.zeros((12, 12, 6), dtype=np.float32)
    v[1:5, 1:5, 1:3] = 4.0  # 32 voxels
    v[1, 1, 1] = 9.0  # its peak
    v[8:10, 8:10, 1:2] = -6.0  # 4 voxels
    return v


def test_the_biggest_cluster_is_first_and_is_labelled_one():
    """The row number has to be the label value, or 'make ROIs' disagrees."""
    table = clusterize(_two_blobs(), threshold=2.0)
    assert len(table) == 2
    assert [c.n_voxels for c in table] == [32, 4]
    assert [c.index for c in table] == [1, 2]
    assert int((table.labels == 1).sum()) == 32
    assert int((table.labels == 2).sum()) == 4


def test_a_negative_peak_is_reported_negative():
    table = clusterize(_two_blobs(), threshold=2.0)
    assert table.clusters[0].peak == 9.0
    assert table.clusters[1].peak == -6.0


def test_the_peak_is_where_the_peak_is():
    table = clusterize(_two_blobs(), threshold=2.0)
    assert table.clusters[0].peak_ijk == (1, 1, 1)


def test_sign_mode_decides_what_counts_as_a_finding():
    v = _two_blobs()
    assert len(clusterize(v, threshold=2.0, sign_mode=SignMode.BOTH)) == 2
    assert len(clusterize(v, threshold=2.0, sign_mode=SignMode.POS)) == 1
    only_neg = clusterize(v, threshold=2.0, sign_mode=SignMode.NEG)
    assert len(only_neg) == 1
    assert only_neg.clusters[0].peak == -6.0


def test_a_blob_straddling_zero_is_two_findings_not_one():
    """Clustering |v| would merge a positive and a negative lobe that touch,
    and the merged 'cluster' would have a mean near zero and mean nothing."""
    v = np.zeros((8, 8, 4), dtype=np.float32)
    v[1:3, 1:4, 1:3] = 5.0
    v[3:5, 1:4, 1:3] = -5.0  # shares a face with the positive lobe
    table = clusterize(v, threshold=2.0, sign_mode=SignMode.BOTH)
    assert len(table) == 2
    assert {np.sign(c.peak) for c in table} == {1.0, -1.0}


def test_small_clusters_can_be_dropped():
    table = clusterize(_two_blobs(), threshold=2.0, min_voxels=10)
    assert len(table) == 1
    assert int(table.labels.max()) == 1  # and the map is renumbered to match


def test_coordinates_go_through_the_affine():
    affine = np.diag([3.0, 3.0, 3.0, 1.0])
    affine[:3, 3] = [-30.0, -36.0, -12.0]
    table = clusterize(_two_blobs(), threshold=2.0, affine=affine)
    peak = table.clusters[0]
    assert peak.peak_xyz == (-27.0, -33.0, -9.0)


def test_without_a_clustsim_table_no_alpha_is_invented():
    """A cluster-size threshold from somebody else's smoothness looks exactly
    like a real one, which is why the absence is reported instead."""
    table = clusterize(_two_blobs(), threshold=2.0)
    assert all(c.alpha is None for c in table)
    assert "no ClustSim table" in table.note


def test_with_one_every_cluster_gets_the_alpha_it_earned():
    from fastfuncstuff.stats.clustsim import ClustSimTable

    cs = ClustSimTable(
        nn=1,
        sidedness="bi-sided",
        pthr=(0.01,),
        athr=(0.10, 0.05, 0.01),
        sizes=np.array([[10.0, 20.0, 30.0]]),
    )
    table = clusterize(_two_blobs(), threshold=2.0, table=cs, pthr=0.01)
    assert table.note == ""
    assert table.clusters[0].alpha == pytest.approx(0.01)  # 32 voxels, off the strict end
    assert table.clusters[1].alpha == pytest.approx(0.10)  # 4 voxels, off the loose end


def test_clusters_become_an_roi_set():
    """The point of the window: a clusterize result and an atlas are one type."""
    table = clusterize(_two_blobs(), threshold=2.0)
    rois = rois_from_clusters(table)
    assert [r.name for r in rois] == ["C1", "C2"]
    assert rois.find(1).n_voxels == 32
    assert rois.at((1, 1, 1)).index == 1


def test_a_threshold_of_zero_is_refused():
    with pytest.raises(ValueError, match="threshold above zero"):
        clusterize(_two_blobs(), threshold=0.0)
