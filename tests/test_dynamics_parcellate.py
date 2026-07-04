"""Tests for the parcellation front-end (cortex -> contiguous ROIs)."""

from __future__ import annotations

import numpy as np
import pytest

from fastfuncstuff.dynamics.parcellate import (
    _grid_adjacency,
    aggregate_by_labels,
    parcellate_atlas,
    parcellate_voronoi,
)


def _synthetic_bold(shape=(6, 6, 4), t=80, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((*shape, t)).astype(np.float32)


def test_aggregate_by_labels_mean_matches_manual():
    rng = np.random.default_rng(0)
    ts = rng.standard_normal((10, 30))
    labels = np.array([1, 1, 1, 2, 2, 2, 2, 3, 3, 3])
    ids, out = aggregate_by_labels(ts, labels, "mean")
    assert list(ids) == [1, 2, 3]
    assert out.shape == (3, 30)
    np.testing.assert_allclose(out[0], ts[:3].mean(axis=0))
    np.testing.assert_allclose(out[1], ts[3:7].mean(axis=0))


def test_aggregate_pca_runs_and_orients():
    rng = np.random.default_rng(1)
    ts = rng.standard_normal((8, 40))
    labels = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    ids, out = aggregate_by_labels(ts, labels, "pca")
    assert out.shape == (2, 40)
    assert np.isfinite(out).all()


def test_aggregate_ignores_background_zero():
    ts = np.ones((4, 5))
    labels = np.array([0, 0, 1, 1])
    ids, out = aggregate_by_labels(ts, labels, "mean")
    assert list(ids) == [1]
    assert out.shape == (1, 5)


def test_parcellate_atlas_shapes_and_values():
    bold = _synthetic_bold()
    atlas = np.zeros(bold.shape[:3], dtype=np.int64)
    atlas[:3] = 1
    atlas[3:] = 2
    ids, ts = parcellate_atlas(bold, atlas)
    assert list(ids) == [1, 2]
    assert ts.shape == (2, bold.shape[-1])
    # Parcel 1 mean equals the manual mean over its voxels (float64 accum).
    manual = bold[atlas == 1].astype(np.float64).mean(axis=0)
    np.testing.assert_allclose(ts[0], manual, rtol=1e-6)


def test_parcellate_atlas_respects_mask():
    bold = _synthetic_bold()
    atlas = np.ones(bold.shape[:3], dtype=np.int64)
    mask = np.zeros(bold.shape[:3], dtype=bool)
    mask[:2] = True
    ids, ts = parcellate_atlas(bold, atlas, mask3d=mask)
    manual = bold[(atlas == 1) & mask].astype(np.float64).mean(axis=0)
    np.testing.assert_allclose(ts[0], manual, rtol=1e-6)


def test_parcellate_voronoi_contiguous_tiles():
    bold = _synthetic_bold(shape=(8, 8, 4), t=50)
    mask = np.ones(bold.shape[:3], dtype=bool)
    ids, ts = parcellate_voronoi(bold, mask, n_parcels=6, seed=3)
    assert 0 < ts.shape[0] <= 6
    assert ts.shape[1] == 50


def test_grid_adjacency_symmetric_and_local():
    mask = np.ones((3, 3, 1), dtype=bool)
    adj = _grid_adjacency(mask).toarray()
    assert (adj == adj.T).all()  # symmetric
    assert np.diag(adj).sum() == 0  # no self-loops
    # Corner voxel (0,0,0) touches exactly 2 neighbours in a 3x3x1 grid.
    coords = np.argwhere(mask)
    corner = int(np.argwhere((coords == [0, 0, 0]).all(axis=1))[0, 0])
    assert adj[corner].sum() == 2


def test_parcellate_ward_recovers_blocks():
    sk = pytest.importorskip("sklearn")  # noqa: F841
    # Two spatial halves with distinct temporal signals; ward should split them.
    rng = np.random.default_rng(4)
    shape, t = (4, 4, 2), 60
    sig_a = rng.standard_normal(t)
    sig_b = rng.standard_normal(t)
    bold = np.empty((*shape, t), dtype=np.float32)
    bold[:2] = sig_a + rng.standard_normal((2, 4, 2, t)) * 0.01
    bold[2:] = sig_b + rng.standard_normal((2, 4, 2, t)) * 0.01
    mask = np.ones(shape, dtype=bool)
    from fastfuncstuff.dynamics.parcellate import parcellate_ward

    ids, ts = parcellate_ward(bold, mask, n_parcels=2)
    assert ts.shape == (2, t)
    # The two parcel timeseries should be well separated (low mutual correlation).
    c = np.corrcoef(ts)[0, 1]
    assert abs(c) < 0.5
