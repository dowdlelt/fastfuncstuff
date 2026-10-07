"""Per-vertex results in the viewer's store: p thresholds, cluster-area cuts, mesh binding."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
from scipy import stats

from fastfuncstuff.io.gifti import mesh_fingerprint
from fastfuncstuff.surface.mesh import vertex_areas
from fastfuncstuff.surface.statmap import (
    SurfaceData,
    cluster_area_threshold,
    load_surface_data,
    stat_threshold,
    surviving_clusters,
)


def _sheet(n: int, s: float = 1.0):
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    v = np.c_[ii.ravel() * s, jj.ravel() * s, np.zeros(n * n)]
    idx = ii * n + jj
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:], idx[1:, 1:]
    f = np.r_[np.c_[a.ravel(), b.ravel(), c.ravel()], np.c_[b.ravel(), d.ravel(), c.ravel()]]
    return v, f.astype(np.int64)


def test_p_to_threshold_follows_the_stat_code():
    assert stat_threshold(3, (40.0,), 0.001) == pytest.approx(stats.t.isf(0.0005, 40))
    assert stat_threshold(5, (), 0.05) == pytest.approx(1.959964, abs=1e-5)
    assert stat_threshold(4, (2.0, 50.0), 0.01) == pytest.approx(stats.f.isf(0.01, 2, 50))
    assert stat_threshold(None, (), 0.01) is None


def test_cluster_threshold_reads_the_table_and_interpolates_between_rows():
    table = {"pthr": [0.01, 0.001], "athr": [0.10, 0.05],
             "area_mm2": [[100.0, 120.0], [30.0, 40.0]]}  # fmt: skip
    d = SurfaceData("x", np.zeros((4, 1)), ["a"], tables={"bi-sided": table})
    assert cluster_area_threshold(d, 0.001, 0.05) == pytest.approx(40.0)
    mid = cluster_area_threshold(d, np.sqrt(0.01 * 0.001), 0.05)
    assert mid == pytest.approx(np.sqrt(120.0 * 40.0))  # log-log midpoint
    assert cluster_area_threshold(d, 0.1, 0.05) is None  # no extrapolation
    assert cluster_area_threshold(d, 0.001, 0.05, sided="2-sided") is None


def test_only_clusters_as_large_as_the_table_survive_and_signs_stay_apart():
    v, f = _sheet(30)
    area = vertex_areas(v, f)
    x, y = v[:, 0], v[:, 1]
    z = np.zeros(len(v))
    big = (x >= 2) & (x <= 9) & (y >= 2) & (y <= 9)  # 64 vertices, ~49 mm^2
    small = (x >= 20) & (x <= 22) & (y >= 20) & (y <= 22)
    neg = (x >= 10) & (x <= 12) & (y >= 2) & (y <= 9)  # touches big
    z[big], z[small], z[neg] = 6.0, 6.0, -6.0
    keep, labels = surviving_clusters(z, 4.0, f, area, min_area=30.0)
    assert keep[big].all() and not keep[small].any()
    assert labels[big].max() == 1  # the largest survivor is cluster 1
    keep_all, labels_all = surviving_clusters(z, 4.0, f, area, min_area=None)
    assert keep_all[small].all() and keep_all[neg].all()
    # bi-sided: the touching negative patch is its own cluster, never merged with big
    assert set(labels_all[neg]).isdisjoint(set(labels_all[big]))
    _, merged = surviving_clusters(z, 4.0, f, area, min_area=None, sided="2-sided")
    assert set(merged[neg]) == set(merged[big])  # 2-sided lets them join


def test_a_reml_bucket_round_trips(tmp_path):
    # What ffs_reml writes (StatCode/StatParams per array) is what the viewer reads.
    from fastfuncstuff.io.afni import save_nifti
    from fastfuncstuff.io.gifti import set_surface_meta

    v, f = _sheet(40)
    hdr = nib.Nifti2Header()
    set_surface_meta(hdr, {"mesh_fingerprint": mesh_fingerprint(f, len(v))})
    vals = np.random.default_rng(0).normal(size=(len(v), 1, 1, 2)).astype(np.float32)
    save_nifti(vals, tmp_path / "r.func.gii", header=hdr, brick_labels=["c#0_Coef", "c#0_Tstat"],
               brick_stataux={1: (3, (55.0,))})  # fmt: skip
    data = load_surface_data(tmp_path / "r.func.gii")
    assert data.stat == {1: (3, (55.0,))} and data.labels == ["c#0_Coef", "c#0_Tstat"]
    assert data.fingerprint == mesh_fingerprint(f, len(v))
    assert data.default_sub_brick() == 1
