"""Per-vertex results in the viewer's store: p thresholds, cluster-area cuts, mesh binding."""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest
from scipy import stats

from fastfuncstuff.io.gifti import mesh_fingerprint, save_gifti_surface
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


def _store_with_sheet(tmp_path):
    from fastfuncstuff.viewer.surfaces import SurfaceStore

    v, f = _sheet(40)
    save_gifti_surface(tmp_path / "lh.white.surf.gii", v, f)
    save_gifti_surface(tmp_path / "lh.pial.surf.gii", v + [0, 0, 2.5], f)
    store = SurfaceStore()
    store.load_mesh(tmp_path / "lh.white.surf.gii", "lh", "white")
    store.load_mesh(tmp_path / "lh.pial.surf.gii", "lh", "pial")
    return store, v, f


def _bucket(path, v, f, table=None):
    blob = np.exp(-((v[:, 0] - 20) ** 2 + (v[:, 1] - 20) ** 2) / 30) * 8
    arrays = [
        nib.gifti.GiftiDataArray(blob.astype(np.float32), intent="NIFTI_INTENT_NONE",
                                 meta=nib.gifti.GiftiMetaData({"Name": "task#0_Coef"})),
        nib.gifti.GiftiDataArray(blob.astype(np.float32), intent="NIFTI_INTENT_TTEST",
                                 meta=nib.gifti.GiftiMetaData({"Name": "task#0_Tstat",
                                                               "StatCode": "3", "StatParams": "60"})),
    ]  # fmt: skip
    meta = {"mesh_fingerprint": mesh_fingerprint(f, len(v))}
    if table:
        meta["ClustSim_bi-sided"] = json.dumps(table)
    nib.save(nib.gifti.GiftiImage(darrays=arrays, meta=nib.gifti.GiftiMetaData(meta)), str(path))


def test_store_binds_data_to_its_mesh_and_thresholds_it(tmp_path):
    store, v, f = _store_with_sheet(tmp_path)
    table = {"pthr": [0.01, 0.001], "athr": [0.1, 0.05], "area_mm2": [[1e4, 2e4], [500.0, 900.0]]}
    _bucket(tmp_path / "s.func.gii", v, f, table)
    data = store.load_data(tmp_path / "s.func.gii", "lh")
    assert data.default_sub_brick() == 1  # opens on the t, not the coefficient
    store.set_data_view(p=0.001, alpha=0.0)
    colours, scale, caption = store.data_display("lh")
    shown = colours[:, 3] > 0
    assert shown.any() and "task#0_Tstat" in caption
    thr = stats.t.isf(0.0005, 60)
    np.testing.assert_array_equal(shown, data.values[:, 1] > thr)
    store.set_data_view(alpha=0.05)  # the blob is far smaller than 900 mm^2
    colours, _, caption = store.data_display("lh")
    assert not (colours[:, 3] > 0).any() and "900" in caption


def test_store_refuses_data_from_another_mesh(tmp_path):
    store, v, f = _store_with_sheet(tmp_path)
    v2, f2 = _sheet(41)
    _bucket(tmp_path / "other.func.gii", v2, f2)
    with pytest.raises(ValueError, match="another mesh"):
        store.load_data(tmp_path / "other.func.gii", "lh")
    # same vertex count, different faces: the fingerprint catches it
    f3 = f[:, [0, 2, 1]]
    _bucket(tmp_path / "flipped.func.gii", v, f3)
    with pytest.raises(ValueError, match="fingerprint"):
        store.load_data(tmp_path / "flipped.func.gii", "lh")


def test_a_reml_bucket_round_trips_into_the_store(tmp_path):
    # What ffs_reml writes (StatCode/StatParams per array) is what the store reads.
    from fastfuncstuff.io.afni import save_nifti
    from fastfuncstuff.io.gifti import set_surface_meta

    store, v, f = _store_with_sheet(tmp_path)
    hdr = nib.Nifti2Header()
    set_surface_meta(hdr, {"mesh_fingerprint": mesh_fingerprint(f, len(v))})
    vals = np.random.default_rng(0).normal(size=(len(v), 1, 1, 2)).astype(np.float32)
    save_nifti(vals, tmp_path / "r.func.gii", header=hdr, brick_labels=["c#0_Coef", "c#0_Tstat"],
               brick_stataux={1: (3, (55.0,))})  # fmt: skip
    data = store.load_data(tmp_path / "r.func.gii", "lh")
    assert data.stat == {1: (3, (55.0,))} and data.labels == ["c#0_Coef", "c#0_Tstat"]
    assert load_surface_data(tmp_path / "r.func.gii").fingerprint == data.fingerprint


def test_store_finds_the_hemisphere_by_fingerprint(tmp_path):
    store, v, f = _store_with_sheet(tmp_path)
    _bucket(tmp_path / "noname.func.gii", v, f)
    store.load_data(tmp_path / "noname.func.gii")
    assert "lh" in store.data
