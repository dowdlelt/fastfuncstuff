"""The voxel <-> vertex map of the cortical ribbon."""

from __future__ import annotations

import numpy as np
import pytest

from fastfuncstuff.surface.ribbon import build_ribbon_map


def _icosphere(level: int):
    t = (1 + 5**0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t)]
    v += [(0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4)]
    f += [(11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8)]
    f += [(3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.array(p, float) / np.linalg.norm(p) for p in v]
    faces = f
    for _ in range(level):
        mid: dict[tuple[int, int], int] = {}

        def m(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in mid:
                p = verts[a] + verts[b]
                verts.append(p / np.linalg.norm(p))
                mid[key] = len(verts) - 1
            return mid[key]

        faces = [
            g
            for a, b, c in faces
            for g in ((a, m(a, b), m(c, a)), (b, m(b, c), m(a, b)),
                      (c, m(c, a), m(b, c)), (m(a, b), m(b, c), m(c, a)))
        ]  # fmt: skip
    return np.asarray(verts), np.asarray(faces, np.int64)


@pytest.fixture(scope="module")
def shell():
    d, f = _icosphere(4)
    centre = np.array([1.5, -2.0, 0.5])
    aff = np.eye(4)
    aff[:3, 3] = centre - 32.0  # a 64^3 1 mm grid around the sphere
    rm = build_ribbon_map(centre + 20 * d, centre + 23 * d, f, aff, (64, 64, 64))
    return d, f, centre, aff, rm


def _voxel_mm(rm, aff):
    ijk = np.stack(np.unravel_index(rm.flat, rm.shape), 1)
    return ijk @ aff[:3, :3].T + aff[:3, 3]


def test_ribbon_is_the_shell_and_depth_is_radial(shell):
    d, f, centre, aff, rm = shell
    r = np.linalg.norm(_voxel_mm(rm, aff) - centre, axis=1)
    assert r.min() > 19.8 and r.max() < 23.2  # chord sag of a level-4 mesh only
    assert rm.flat.size > 0.9 * 4 / 3 * np.pi * (23**3 - 20**3)
    np.testing.assert_allclose(rm.depth, (r - 20) / 3, atol=0.08)


def test_each_voxel_belongs_to_the_vertex_above_it(shell):
    d, f, centre, aff, rm = shell
    u = _voxel_mm(rm, aff) - centre
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    cos = np.einsum("ij,ij->i", u, d[rm.vertex])
    edge = np.arccos(np.clip(np.dot(d[f[0, 0]], d[f[0, 1]]), -1, 1))
    assert np.arccos(np.clip(cos, -1, 1)).max() < edge  # within one edge of its vertex


def test_paint_and_voxels_of_are_inverse_views(shell):
    d, f, centre, aff, rm = shell
    north = d[:, 2] > 0.7
    vol = rm.paint(north.astype(np.float32))
    assert set(np.unique(vol)) <= {0.0, 1.0}
    np.testing.assert_array_equal(
        np.sort(np.flatnonzero(vol.ravel())), np.sort(rm.voxels_of(north))
    )
    assert (rm.vertex_counts() > 0).mean() > 0.9  # 1 mm voxels, ~1.3 mm vertex spacing
