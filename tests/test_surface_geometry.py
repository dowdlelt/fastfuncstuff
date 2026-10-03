"""Slice/mesh intersection: what the viewer's surface outlines are made of."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from fastfuncstuff.surface.geometry import SliceIndex


def sphere(radius: float = 20.0, n: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """A closed, consistently triangulated sphere (Fibonacci points + hull)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    v = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
    return v * radius, ConvexHull(v).simplices.astype(np.int32)


def test_cut_lies_on_the_plane_and_traces_the_circle():
    v, f = sphere()
    idx = SliceIndex(v, f)
    for axis, pos in [(0, 0.0), (1, 7.5), (2, -12.0)]:
        seg = idx.segments(axis, pos)
        assert len(seg) > 0
        np.testing.assert_allclose(seg[..., axis], pos, atol=1e-9)
        rho = np.sqrt(20.0**2 - pos**2)
        # Endpoints sit on mesh edges, a hair inside the true sphere.
        r = np.linalg.norm(np.delete(seg, axis, axis=-1), axis=-1)
        assert np.all(r <= rho + 1e-9) and np.all(r > 0.98 * rho)
        length = np.linalg.norm(seg[:, 1] - seg[:, 0], axis=-1).sum()
        assert abs(length - 2 * np.pi * rho) < 0.01 * 2 * np.pi * rho


def test_index_finds_exactly_what_brute_force_finds():
    v, f = sphere(n=1500)
    idx = SliceIndex(v, f)
    # Off the Fibonacci lattice (z = 20(1 - 2i/n)); on-plane vertices have their own test.
    for pos in np.linspace(-19.5, 19.5, 13) + 0.0137:
        d = v[f, 2] - pos
        n_above = (d > 0).sum(1)
        expected = int(((n_above == 1) | (n_above == 2)).sum())
        assert len(idx.segments(2, pos)) == expected


def test_vertex_exactly_on_the_plane_gives_no_degenerate_or_doubled_segments():
    v, f = sphere(n=800)
    pos = float(v[123, 0])
    seg = SliceIndex(v, f).segments(0, pos)
    lengths = np.linalg.norm(seg[:, 1] - seg[:, 0], axis=-1)
    # A segment may legitimately start at the on-plane vertex, but no face
    # contributes a zero-length one and the trace still closes.
    assert np.all(lengths > 0)
    ends = np.round(seg.reshape(-1, 3), 9)
    _, counts = np.unique(ends, axis=0, return_counts=True)
    assert np.all(counts == 2)


def test_plane_outside_the_mesh_is_empty():
    v, f = sphere()
    assert SliceIndex(v, f).segments(2, 25.0).shape == (0, 2, 3)


def test_moved_vertices_are_cut_where_they_now_are():
    from fastfuncstuff.surface.mesh import MeshTopology

    v, f = sphere(n=2000)
    topo = MeshTopology.from_faces(f)
    idx = SliceIndex(v, f)
    top = np.flatnonzero(v[:, 2] > 15.0)
    pushed = v[top] * 1.3  # well past z=21, where the original sphere ends
    idx.move(top, pushed, topo.faces_of(top))
    moved = v.copy()
    moved[top] = pushed
    fresh = SliceIndex(moved, f)
    for pos in (18.0, 21.0, 24.0):
        a, b = idx.segments(2, pos), fresh.segments(2, pos)
        assert len(a) == len(b) > 0
        np.testing.assert_allclose(np.sort(a.reshape(-1, 3), 0), np.sort(b.reshape(-1, 3), 0))
