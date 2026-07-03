"""
Correctness tests for stats/cluster_fast.py.

This module is the inner kernel for every permutation FWE cluster table
produced by ffs_perm. The DSU walk is delicate (sorted-descending walk
with snapshot-on-threshold-crossing), so we cross-check against
scipy.ndimage.label as an independent oracle: threshold at each pthr,
label with the matching structuring element, take max component size.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import generate_binary_structure, label

from fastfuncstuff.stats.cluster_fast import (
    _offsets_for_nn,
    _walk_dsu_extent,
    cluster_extent_one_perm,
    precompile,
)

# ---------------------------------------------------------------------------
# Oracle: scipy-based reference for max cluster extent above a threshold.
# ---------------------------------------------------------------------------


def _scipy_max_extent(stat3d: np.ndarray, tcrit: float, nn: int) -> int:
    above = stat3d > tcrit
    if not above.any():
        return 0
    structure = generate_binary_structure(3, nn)
    lab, _ = label(above, structure=structure)
    if lab.max() == 0:
        return 0
    # Component sizes; index 0 is background.
    sizes = np.bincount(lab.ravel())[1:]
    return int(sizes.max())


# ---------------------------------------------------------------------------
# _offsets_for_nn
# ---------------------------------------------------------------------------


class TestOffsetsForNN:
    def test_nn1_face_count(self):
        off = _offsets_for_nn(1)
        assert off.shape == (6, 3)
        assert off.dtype == np.int8

    def test_nn2_face_edge_count(self):
        off = _offsets_for_nn(2)
        assert off.shape == (18, 3)
        # No origin
        assert not np.any(np.all(off == 0, axis=1))
        # All offsets within unit cube
        assert (np.abs(off) <= 1).all()
        # Sum of |components| ≤ 2 (faces + edges, no corners)
        assert (np.abs(off).sum(axis=1) <= 2).all()

    def test_nn3_full_moore(self):
        off = _offsets_for_nn(3)
        assert off.shape == (26, 3)
        assert not np.any(np.all(off == 0, axis=1))

    def test_invalid_nn(self):
        with pytest.raises(ValueError, match="NN must be 1, 2, or 3"):
            _offsets_for_nn(4)
        with pytest.raises(ValueError):
            _offsets_for_nn(0)

    def test_no_duplicate_offsets(self):
        for nn in (1, 2, 3):
            off = _offsets_for_nn(nn)
            tuples = {tuple(row) for row in off}
            assert len(tuples) == off.shape[0]


# ---------------------------------------------------------------------------
# _walk_dsu_extent vs scipy oracle on synthetic 3D volumes.
# ---------------------------------------------------------------------------


def _walk_for_test(stat3d: np.ndarray, tcrits_desc: np.ndarray, nn: int) -> np.ndarray:
    """Apply the kernel exactly like cluster_extent_one_perm's inner loop,
    but on a dense 3D array with all voxels in-mask."""
    nx, ny, nz = stat3d.shape
    stat_flat = stat3d.reshape(-1).astype(np.float32)
    lowest = tcrits_desc[-1]
    above = stat_flat > lowest
    idx_above = np.nonzero(above)[0].astype(np.int64)
    stat_above = stat_flat[above]
    order = np.argsort(-stat_above, kind="stable")
    idx_sorted = np.ascontiguousarray(idx_above[order])
    stat_sorted = np.ascontiguousarray(stat_above[order].astype(np.float32))

    parent = np.full(nx * ny * nz, -1, dtype=np.int64)
    size = np.zeros(nx * ny * nz, dtype=np.int64)
    return _walk_dsu_extent(
        idx_sorted,
        stat_sorted,
        parent,
        size,
        _offsets_for_nn(nn),
        nx,
        ny,
        nz,
        tcrits_desc.astype(np.float64),
    )


class TestWalkDSUExtent:
    def test_empty_above_threshold(self):
        stat = np.zeros((4, 4, 4), dtype=np.float32)
        tcrits = np.array([1.0], dtype=np.float64)
        out = _walk_for_test(stat, tcrits, nn=1)
        assert out.tolist() == [0]

    def test_single_isolated_voxel(self):
        stat = np.zeros((4, 4, 4), dtype=np.float32)
        stat[1, 1, 1] = 5.0
        tcrits = np.array([1.0], dtype=np.float64)
        out = _walk_for_test(stat, tcrits, nn=1)
        assert out.tolist() == [1]

    def test_face_connected_pair_nn1(self):
        """Two adjacent voxels along an axis: extent=2 under nn=1."""
        stat = np.zeros((4, 4, 4), dtype=np.float32)
        stat[1, 1, 1] = 5.0
        stat[1, 1, 2] = 5.0
        out = _walk_for_test(stat, np.array([1.0]), nn=1)
        assert out.tolist() == [2]

    def test_edge_connected_pair_nn1_vs_nn2(self):
        """Diagonal pair (edge-adjacent): nn=1 splits them, nn=2 merges."""
        stat = np.zeros((4, 4, 4), dtype=np.float32)
        stat[1, 1, 1] = 5.0
        stat[1, 2, 2] = 5.0  # shares an edge (differs by 1 in two axes)
        out_nn1 = _walk_for_test(stat, np.array([1.0]), nn=1)
        out_nn2 = _walk_for_test(stat, np.array([1.0]), nn=2)
        assert out_nn1.tolist() == [1]
        assert out_nn2.tolist() == [2]

    def test_corner_connected_pair_nn2_vs_nn3(self):
        """Corner pair (differs by 1 on all 3 axes): only nn=3 merges."""
        stat = np.zeros((4, 4, 4), dtype=np.float32)
        stat[1, 1, 1] = 5.0
        stat[2, 2, 2] = 5.0
        out_nn2 = _walk_for_test(stat, np.array([1.0]), nn=2)
        out_nn3 = _walk_for_test(stat, np.array([1.0]), nn=3)
        assert out_nn2.tolist() == [1]
        assert out_nn3.tolist() == [2]

    def test_multiple_thresholds_descending_snapshot(self):
        """Snapshot at descending pthrs: as the threshold drops, more voxels
        join the cluster and extent grows monotonically."""
        stat = np.zeros((5, 5, 5), dtype=np.float32)
        # Build a 1D chain of 4 voxels with descending stat
        stat[1, 1, 1] = 5.0
        stat[1, 1, 2] = 4.0
        stat[1, 1, 3] = 3.0
        stat[2, 1, 3] = 2.0  # branches off, face-adjacent to the last
        tcrits = np.array([4.5, 3.5, 2.5, 1.5], dtype=np.float64)
        out = _walk_for_test(stat, tcrits, nn=1)
        # At 4.5: only [1,1,1] qualifies → 1
        # At 3.5: + [1,1,2], face-adj → 2
        # At 2.5: + [1,1,3], face-adj → 3
        # At 1.5: + [2,1,3], face-adj → 4
        assert out.tolist() == [1, 2, 3, 4]

    @pytest.mark.parametrize("nn", [1, 2, 3])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_random_volume_matches_scipy(self, nn, seed):
        rng = np.random.default_rng(seed)
        stat = rng.standard_normal((8, 9, 7)).astype(np.float32)
        tcrits = np.array([2.0, 1.5, 1.0, 0.5], dtype=np.float64)
        out = _walk_for_test(stat, tcrits, nn=nn)
        expected = np.array([_scipy_max_extent(stat, t, nn) for t in tcrits])
        np.testing.assert_array_equal(out, expected)

    def test_all_voxels_above_single_threshold(self):
        """Whole volume above threshold under nn=1 is one connected blob."""
        stat = np.full((3, 3, 3), 5.0, dtype=np.float32)
        out = _walk_for_test(stat, np.array([1.0]), nn=1)
        assert out.tolist() == [27]


# ---------------------------------------------------------------------------
# cluster_extent_one_perm — end-to-end including sidedness handling.
# ---------------------------------------------------------------------------


def _full_volume_setup(shape):
    nx, ny, nz = shape
    mask_flat_idx = np.arange(nx * ny * nz, dtype=np.int64)
    parent = np.full(nx * ny * nz, -1, dtype=np.int64)
    size = np.zeros(nx * ny * nz, dtype=np.int64)
    return mask_flat_idx, parent, size


class TestClusterExtentOnePerm:
    @pytest.fixture
    def shape(self):
        return (6, 7, 5)

    @pytest.fixture
    def stat3d(self, shape):
        rng = np.random.default_rng(42)
        return rng.standard_normal(shape).astype(np.float32)

    def test_one_sided_matches_scipy(self, stat3d, shape):
        mask_flat_idx, parent, size = _full_volume_setup(shape)
        tcrits_desc = np.array([2.0, 1.5, 1.0, 0.5], dtype=np.float64)
        offsets = {1: _offsets_for_nn(1), 2: _offsets_for_nn(2), 3: _offsets_for_nn(3)}

        out = cluster_extent_one_perm(
            stat3d=stat3d,
            mask_flat_idx=mask_flat_idx,
            shape_xyz=shape,
            nns=(1, 2, 3),
            sideds=("1-sided",),
            tcrits_by_sided={"1-sided": tcrits_desc},
            offsets_by_nn=offsets,
            parent_scratch=parent,
            size_scratch=size,
        )

        for nn in (1, 2, 3):
            expected = np.array([_scipy_max_extent(stat3d, t, nn) for t in tcrits_desc])
            np.testing.assert_array_equal(out[("1-sided", nn)], expected, err_msg=f"nn={nn}")

    def test_two_sided_matches_scipy_on_abs(self, stat3d, shape):
        mask_flat_idx, parent, size = _full_volume_setup(shape)
        tcrits_desc = np.array([2.0, 1.5, 1.0], dtype=np.float64)
        offsets = {1: _offsets_for_nn(1)}

        out = cluster_extent_one_perm(
            stat3d=stat3d,
            mask_flat_idx=mask_flat_idx,
            shape_xyz=shape,
            nns=(1,),
            sideds=("2-sided",),
            tcrits_by_sided={"2-sided": tcrits_desc},
            offsets_by_nn=offsets,
            parent_scratch=parent,
            size_scratch=size,
        )

        abs_stat = np.abs(stat3d)
        expected = np.array([_scipy_max_extent(abs_stat, t, 1) for t in tcrits_desc])
        np.testing.assert_array_equal(out[("2-sided", 1)], expected)

    def test_bi_sided_is_max_of_pos_and_neg_tails(self, shape):
        """bi-sided runs +stat and -stat passes independently and takes the
        elementwise max — verify against an asymmetric volume where the
        positive tail and negative tail have different max extents."""
        nx, ny, nz = shape
        stat = np.zeros(shape, dtype=np.float32)
        # Positive blob: 3 face-connected voxels, all > 1
        stat[1, 1, 1] = 3.0
        stat[1, 1, 2] = 3.0
        stat[1, 1, 3] = 3.0
        # Negative blob: 5 face-connected voxels, all < -1
        for k in range(5):
            stat[3, 2, k] = -3.0

        mask_flat_idx, parent, size = _full_volume_setup(shape)
        tcrits_desc = np.array([1.0], dtype=np.float64)
        offsets = {1: _offsets_for_nn(1)}

        out = cluster_extent_one_perm(
            stat3d=stat,
            mask_flat_idx=mask_flat_idx,
            shape_xyz=shape,
            nns=(1,),
            sideds=("bi-sided",),
            tcrits_by_sided={"bi-sided": tcrits_desc},
            offsets_by_nn=offsets,
            parent_scratch=parent,
            size_scratch=size,
        )

        # bi-sided takes max over the two passes → max(3, 5) = 5
        assert out[("bi-sided", 1)].tolist() == [5]

    def test_partial_mask_only_counts_in_mask_voxels(self, shape):
        """When mask excludes some voxels, out-of-mask voxels don't
        contribute to extent even if their stat would be above threshold."""
        nx, ny, nz = shape
        stat = np.zeros(shape, dtype=np.float32)
        # 3 face-connected voxels along z at i=1,j=1
        stat[1, 1, 1] = 5.0
        stat[1, 1, 2] = 5.0
        stat[1, 1, 3] = 5.0

        # Mask out the middle voxel — should break the chain into two clusters of 1
        all_idx = np.arange(nx * ny * nz, dtype=np.int64)
        middle_flat = 1 * (ny * nz) + 1 * nz + 2
        mask_flat_idx = all_idx[all_idx != middle_flat]
        parent = np.full(nx * ny * nz, -1, dtype=np.int64)
        size = np.zeros(nx * ny * nz, dtype=np.int64)

        out = cluster_extent_one_perm(
            stat3d=stat,
            mask_flat_idx=mask_flat_idx,
            shape_xyz=shape,
            nns=(1,),
            sideds=("1-sided",),
            tcrits_by_sided={"1-sided": np.array([1.0], dtype=np.float64)},
            offsets_by_nn={1: _offsets_for_nn(1)},
            parent_scratch=parent,
            size_scratch=size,
        )

        # Two singletons (the two endpoints) → max extent 1
        assert out[("1-sided", 1)].tolist() == [1]

    def test_unknown_sidedness_raises(self, stat3d, shape):
        mask_flat_idx, parent, size = _full_volume_setup(shape)
        with pytest.raises(ValueError, match="unknown sidedness"):
            cluster_extent_one_perm(
                stat3d=stat3d,
                mask_flat_idx=mask_flat_idx,
                shape_xyz=shape,
                nns=(1,),
                sideds=("totally-bogus",),
                tcrits_by_sided={"totally-bogus": np.array([1.0])},
                offsets_by_nn={1: _offsets_for_nn(1)},
                parent_scratch=parent,
                size_scratch=size,
            )

    def test_extent_monotone_in_threshold(self, stat3d, shape):
        """As tcrit decreases, more voxels qualify so max extent must be
        monotonically non-decreasing."""
        mask_flat_idx, parent, size = _full_volume_setup(shape)
        tcrits_desc = np.array([2.0, 1.5, 1.0, 0.5, 0.0, -0.5], dtype=np.float64)

        out = cluster_extent_one_perm(
            stat3d=stat3d,
            mask_flat_idx=mask_flat_idx,
            shape_xyz=shape,
            nns=(1,),
            sideds=("1-sided",),
            tcrits_by_sided={"1-sided": tcrits_desc},
            offsets_by_nn={1: _offsets_for_nn(1)},
            parent_scratch=parent,
            size_scratch=size,
        )

        ext = out[("1-sided", 1)]
        assert np.all(np.diff(ext) >= 0), f"extents not monotone: {ext}"


# ---------------------------------------------------------------------------
# precompile — exercises the JIT warm-up path.
# ---------------------------------------------------------------------------


def test_precompile_runs():
    """precompile() should JIT-warm the kernel and not raise."""
    precompile()
