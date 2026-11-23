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

from fastfuncstuff.connectivity.matrix import correlation_matrix
from fastfuncstuff.viewer.matrix import ORDERINGS, build_matrix, seriate
from fastfuncstuff.viewer.rois import rois_from_labels

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


def test_the_module_count_is_one_more_than_the_lines_drawn():
    """Boundaries and modules are off by one, and the status line says modules."""
    rng = np.random.default_rng(6)
    data, labels = _networks(rng)
    m = build_matrix(
        data, rois=rois_from_labels(labels), order="hierarchical", polort=-1, device=CPU
    )
    assert f"{len(m.blocks) + 1} modules" in m.status()


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


def test_every_node_knows_where_it_is_after_reordering():
    """A click on a node has to land in that node's region, not the region
    that held its row before the ordering moved it."""
    rng = np.random.default_rng(5)
    data, labels = _networks(rng)
    rois = rois_from_labels(labels)
    m = build_matrix(data, rois=rois, order="hierarchical", polort=-1, device=CPU)
    for node, label in enumerate(m.indices):
        assert labels[m.location_of(node)] == label


def test_a_voxel_bin_points_at_a_voxel_inside_the_mask():
    rng = np.random.default_rng(6)
    data, _ = _networks(rng)
    mask = np.zeros(data.shape[:3], dtype=bool)
    mask[:6] = True
    m = build_matrix(data, mask=mask, order="input", polort=-1, max_nodes=10, device=CPU)
    assert m.locations is not None and m.locations.shape == (m.n_nodes, 3)
    assert all(mask[m.location_of(node)] for node in range(m.n_nodes))


def test_roi_matrix_is_the_ffs_fc_matrix():
    """One computation: the viewer's ROI matrix equals ffs_fc's on the same run and atlas."""
    from fastfuncstuff.connectivity.prep import build_cleaner
    from fastfuncstuff.glm.core import construct_polynomial_matrix
    from fastfuncstuff.processing.roi_collapse import build_roi_weights, collapse_to_rois
    from fastfuncstuff.utils import scale_to_percent_signal

    rng = np.random.default_rng(5)
    data, labels = _networks(rng)
    gain = rng.uniform(200, 2000, size=data.shape[:3])[..., None]  # brightness varies by voxel
    data = (gain + 20 * data + 0.05 * np.arange(data.shape[3])).astype(np.float32)
    m = build_matrix(data, rois=rois_from_labels(labels, names=None), order="input", polort=2,
                     device=CPU)  # fmt: skip

    rows = labels.reshape(-1) > 0
    vox = torch.as_tensor(data.reshape(-1, data.shape[3])[rows])
    psc, _, _ = scale_to_percent_signal(vox, [0], verbose=False, track_violations=False)
    slot, _, sizes = build_roi_weights(labels.reshape(-1)[rows].reshape(-1, 1, 1))
    means = collapse_to_rois(psc, slot, sizes)
    poly = construct_polynomial_matrix(data.shape[3], 2, CPU, torch.float64)
    clean = build_cleaner([poly], [0], data.shape[3]).apply(means)
    np.testing.assert_allclose(m.matrix, correlation_matrix(clean).numpy(), atol=1e-5)


def test_a_layer_without_baseline_is_averaged_in_z():
    """Residual-like data has no baseline to be a percentage of; the matrix still draws."""
    rng = np.random.default_rng(6)
    data, labels = _networks(rng)  # zero-mean by construction
    m = build_matrix(data, rois=rois_from_labels(labels, names=None), order="input", polort=-1,
                     device=CPU)  # fmt: skip
    assert m.matrix[0, 1] > 0.8 and abs(m.matrix[0, 2]) < 0.4
