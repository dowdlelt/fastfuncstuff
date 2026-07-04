"""Tests for per-state graph-theoretic metrics on dynamic FC."""

from __future__ import annotations

import numpy as np
import torch

from fastfuncstuff.dynamics.graph import (
    betweenness_centrality,
    efficiency,
    fc_to_weights,
    node_strength,
    state_graph_metrics,
    weighted_clustering,
)


def _triangle():
    return np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])


def _star(n=4):
    w = np.zeros((n, n))
    w[0, 1:] = 1.0
    w[1:, 0] = 1.0
    return w


def test_fc_to_weights_absolute_and_positive():
    fc = np.array([[1.0, -0.8, 0.5], [-0.8, 1.0, 0.2], [0.5, 0.2, 1.0]])
    wa = fc_to_weights(fc, signed="absolute")
    assert wa[0, 1] == 0.8 and np.all(np.diag(wa) == 0)
    wp = fc_to_weights(fc, signed="positive")
    assert wp[0, 1] == 0.0 and wp[0, 2] == 0.5


def test_density_threshold_keeps_strongest_fraction():
    fc = np.array([[0.0, 0.9, 0.1], [0.9, 0.0, 0.5], [0.1, 0.5, 0.0]])
    w = fc_to_weights(fc, density=1 / 3)  # keep 1 of 3 upper-triangle edges
    assert (w > 0).sum() == 2  # symmetric -> two entries for one edge
    assert w[0, 1] == 0.9  # the strongest edge survives


def test_clustering_of_triangle_is_one():
    c = weighted_clustering(_triangle())
    np.testing.assert_allclose(c, np.ones(3), atol=1e-12)


def test_triangle_has_zero_betweenness():
    # every pair is directly connected, so no node is ever an intermediary
    bc = betweenness_centrality(_triangle())
    np.testing.assert_allclose(bc, np.zeros(3), atol=1e-12)


def test_star_hub_dominates_betweenness_and_strength():
    w = _star(4)
    bc = betweenness_centrality(w, normalized=False)
    assert bc[0] > 0 and np.allclose(bc[1:], 0.0)  # hub bridges every leaf pair
    strength = node_strength(w)
    assert strength[0] == 3 and np.all(strength[1:] == 1)


def test_complete_graph_efficiency_is_one():
    glob, nodal = efficiency(_triangle())
    assert abs(glob - 1.0) < 1e-12
    np.testing.assert_allclose(nodal, np.ones(3), atol=1e-12)


def test_state_graph_metrics_shapes_and_types():
    rng = np.random.default_rng(0)
    fc = torch.tensor(np.stack([_triangle(), _star(3)]))  # (2, 3, 3)
    gm = state_graph_metrics(fc)
    assert gm.strength.shape == (2, 3)
    assert gm.global_efficiency.shape == (2,)
    assert gm.mean_clustering.shape == (2,)
    # the fully-connected triangle is more integrated than the star
    assert gm.global_efficiency[0] >= gm.global_efficiency[1]
