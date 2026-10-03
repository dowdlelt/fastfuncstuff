"""Topology edits keep every surface and per-vertex array of a hemisphere together."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from fastfuncstuff.surface.topology import (
    MeshBundle,
    check_manifold,
    collapse_vertex,
    longest_edge,
    neighbours,
    split_edge,
)


def _bundle(n=400):
    i = np.arange(n) + 0.5
    phi, theta = np.arccos(1 - 2 * i / n), np.pi * (1 + 5**0.5) * i
    u = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
    f = ConvexHull(u).simplices.astype(np.int64)
    fn = np.cross(u[f[:, 1]] - u[f[:, 0]], u[f[:, 2]] - u[f[:, 0]])
    flip = np.einsum("ij,ij->i", fn, u[f].mean(1)) < 0
    f[flip] = f[flip][:, ::-1]
    wobble = 1 + 0.05 * np.sin(5 * u[:, 0])
    b = MeshBundle(
        faces=f,
        positions={
            "white": 20 * u * wobble[:, None],
            "pial": 23 * u * wobble[:, None],
            "sphere": 100 * u + 7.0,  # off-origin sphere: centre is the centroid
        },
        scalars={"thickness": np.linspace(1, 4, n).astype(np.float32)},
        labels={"aparc": (u[:, 2] > 0).astype(np.int64)},
        masks={"cortex": u[:, 0] < 0.9},
        spherical={"sphere"},
    )
    check_manifold(b.faces, b.n_vertices)
    return b


def test_delete_keeps_every_array_in_step_and_the_mesh_a_sphere():
    b = _bundle()
    v = 123
    out, remap, kept = collapse_vertex(b, v)
    assert out.n_vertices == b.n_vertices - 1
    assert out.faces.shape[0] == b.faces.shape[0] - 2
    check_manifold(out.faces, out.n_vertices)
    for store in ("positions", "scalars", "labels", "masks"):
        for k, arr in getattr(out, store).items():
            assert arr.shape[0] == out.n_vertices, (store, k)
    # Every surviving vertex carries its own values to its new index.
    for old in range(b.n_vertices):
        if old == v:
            continue
        new = remap[old]
        for name in b.positions:
            np.testing.assert_array_equal(out.positions[name][new], b.positions[name][old])
        assert out.labels["aparc"][new] == b.labels["aparc"][old]
    assert remap[v] == kept
    # No triangle turned over, on any surface.
    for name, p in out.positions.items():
        c = p.mean(0)
        n = np.cross(
            p[out.faces[:, 1]] - p[out.faces[:, 0]], p[out.faces[:, 2]] - p[out.faces[:, 0]]
        )
        assert np.all(np.einsum("ij,ij->i", n, p[out.faces].mean(1) - c) > 0), name


def test_split_puts_the_new_vertex_mid_edge_on_every_surface_and_on_the_sphere():
    b = _bundle()
    a, c = longest_edge(b, 50)
    out, m = split_edge(b, a, c)
    assert m == b.n_vertices and out.n_vertices == b.n_vertices + 1
    assert out.faces.shape[0] == b.faces.shape[0] + 2
    np.testing.assert_allclose(
        out.positions["white"][m], 0.5 * (b.positions["white"][a] + b.positions["white"][c])
    )
    centre = b.positions["sphere"].mean(0)
    r = np.linalg.norm(out.positions["sphere"][m] - centre)
    assert r == pytest.approx(np.linalg.norm(b.positions["sphere"][a] - centre), rel=1e-3)
    assert out.scalars["thickness"][m] == pytest.approx(
        0.5 * (b.scalars["thickness"][a] + b.scalars["thickness"][c])
    )
    assert set(neighbours(out.faces, m).tolist()) >= {a, c}
    assert c not in neighbours(out.faces, a)  # the edge is gone, replaced by a-m-c


def test_impossible_edits_are_refused():
    b = _bundle()
    far = int(np.setdiff1d(np.arange(b.n_vertices), [0, *neighbours(b.faces, 0)])[0])
    with pytest.raises(ValueError, match="not an edge"):
        split_edge(b, 0, far)
    # Every vertex of a tetrahedron has valence 3: deleting one would leave
    # a triangle's worth of surface, not a closed one.
    tet = MeshBundle(
        faces=np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int64),
        positions={"white": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)},
    )
    with pytest.raises(ValueError):
        collapse_vertex(tet, 3)


def test_delete_after_split_round_trips_the_vertex_count():
    b = _bundle()
    a, c = longest_edge(b, 10)
    grown, m = split_edge(b, a, c)
    back, _, _ = collapse_vertex(grown, m)
    assert back.n_vertices == b.n_vertices
    check_manifold(back.faces, back.n_vertices)
