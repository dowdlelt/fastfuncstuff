"""Correlation matrices: whether the picture is of the data or of the ordering.

A connectivity matrix is a square of colour with no numbers on it, so the ways
it can be wrong all look like results. These pin the three that matter: that a
node's time course is the ROI it claims, that seriation finds structure the
input order hid, and that the no-ROI fallback reduces voxels honestly rather
than pretending to be a voxelwise matrix.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.viewer.matrix import ORDERINGS, build_matrix, seriate
from fastfuncstuff.viewer.rois import rois_from_labels
from fastfuncstuff.viewer.series import correlation_matrix

CPU = torch.device("cpu")


def _networks(rng, nt=120):
    """Three regions: two that share a signal, one independent of both."""
    shared = rng.normal(size=nt)
    other = rng.normal(size=nt)
    data = np.zeros((9, 3, 3, nt), dtype=np.float32)
    labels = np.zeros((9, 3, 3), dtype=np.int32)
    for block, wave in enumerate((shared, shared, other)):
        lo = block * 3
        labels[lo : lo + 3] = block + 1
        noise = rng.normal(scale=0.25, size=(3, 3, 3, nt))
        data[lo : lo + 3] = (wave[None, None, None, :] + noise).astype(np.float32)
    return data, labels


def test_roi_nodes_recover_the_structure_that_was_put_in():
    rng = np.random.default_rng(0)
    data, labels = _networks(rng)
    rois = rois_from_labels(labels, names=None)
    m = build_matrix(data, rois=rois, order="input", polort=-1, device=CPU)

    assert m.from_rois and m.n_nodes == 3
    assert m.indices == (1, 2, 3)
    assert m.matrix[0, 1] > 0.8  # the two that share a wave
    assert abs(m.matrix[0, 2]) < 0.4  # the one that does not
    assert m.sizes == (27, 27, 27)


def test_a_node_is_the_average_of_its_roi_and_not_of_its_neighbour():
    """The failure this catches is a matrix row attributed to the wrong region."""
    rng = np.random.default_rng(1)
    data, labels = _networks(rng)
    rois = rois_from_labels(labels)
    m = build_matrix(data, rois=rois, order="input", polort=-1, device=CPU)
    for position, value in enumerate((1, 2, 3)):
        raw = data[labels == value].mean(0)
        drawn = m.series[position]
        r = np.corrcoef(raw, drawn)[0, 1]
        assert r > 0.99


def test_seriation_puts_a_hidden_block_back_together():
    """Interleaved input order is the case where 'as listed' shows nothing."""
    rng = np.random.default_rng(2)
    nt = 150
    a, b = rng.normal(size=nt), rng.normal(size=nt)
    # Alternating membership: in input order the matrix is a checkerboard.
    series = np.stack(
        [(a if i % 2 == 0 else b) + rng.normal(scale=0.3, size=nt) for i in range(12)]
    )
    matrix = correlation_matrix(torch.as_tensor(series, dtype=torch.float32))
    order, blocks = seriate(matrix)

    groups = [i % 2 for i in order]
    # Every member of one group before every member of the other, and the
    # boundary reported where it actually is.
    assert groups == sorted(groups) or groups == sorted(groups, reverse=True)
    assert blocks == (6,)


def test_without_rois_the_nodes_are_bins_and_say_so():
    rng = np.random.default_rng(3)
    data, _ = _networks(rng)
    m = build_matrix(data, mask=np.ones(data.shape[:3], bool), max_nodes=8, polort=-1, device=CPU)

    assert not m.from_rois
    assert m.n_nodes == 8
    assert m.indices == (-1,) * 8
    assert m.n_voxels == 81  # every voxel accounted for
    assert sum(m.sizes) == 81  # and none of them counted twice
    assert np.allclose(np.diag(m.matrix), 1.0)


def test_a_grid_mismatch_is_refused_rather_than_resampled():
    rng = np.random.default_rng(4)
    data, labels = _networks(rng)
    rois = rois_from_labels(labels[:6])
    with pytest.raises(ValueError, match="same grid"):
        build_matrix(data, rois=rois, device=CPU)


def test_every_ordering_returns_the_same_matrix_permuted():
    rng = np.random.default_rng(5)
    data, labels = _networks(rng)
    rois = rois_from_labels(labels)
    reference = build_matrix(data, rois=rois, order="input", polort=-1, device=CPU)
    for order in ORDERINGS:
        m = build_matrix(data, rois=rois, order=order, polort=-1, device=CPU)
        assert sorted(m.indices) == sorted(reference.indices)
        # The cell for one pair says the same thing whatever order it is in.
        for i, left in enumerate(m.indices):
            for j, right in enumerate(m.indices):
                a = reference.indices.index(left)
                b = reference.indices.index(right)
                assert m.matrix[i, j] == pytest.approx(reference.matrix[a, b], abs=1e-5)
