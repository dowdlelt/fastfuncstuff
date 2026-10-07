"""remesh_via_sphere: a mesh rebuilt on another vertex set through the sphere."""

from __future__ import annotations

import numpy as np

from fastfuncstuff.surface.remesh import remesh_via_sphere, sphere_lookup
from fastfuncstuff.surface.topology import MeshBundle


def _icosphere(level: int, radius: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
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
            for g in (
                (a, m(a, b), m(c, a)),
                (b, m(b, c), m(a, b)),
                (c, m(c, a), m(b, c)),
                (m(a, b), m(b, c), m(c, a)),
            )
        ]
    return np.asarray(verts) * radius, np.asarray(faces, np.int64)


def test_lookup_of_a_mesh_on_itself_is_the_identity():
    v, f = _icosphere(3)
    look = sphere_lookup(v, f, v)
    hit = look.corners[np.arange(len(v)), look.weights.argmax(1)]
    assert np.array_equal(hit, np.arange(len(v)))
    np.testing.assert_allclose(look.weights.max(1), 1.0, atol=1e-9)
    np.testing.assert_allclose(look.interpolate(v), v, atol=1e-6)


def test_weights_are_radial_barycentrics_of_the_containing_triangle():
    # Different radius and centre on purpose: only directions may matter.
    v, f = _icosphere(2, radius=37.0)
    centre = np.array([5.0, -3.0, 12.0])
    rng = np.random.default_rng(0)
    d = rng.normal(size=(500, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    look = sphere_lookup(v + centre, f, d * 100.0, centre=centre)
    assert (look.weights >= 0).all()
    np.testing.assert_allclose(look.weights.sum(1), 1.0)
    p = look.interpolate(v)  # a point on the flat triangle, on the target's ray
    np.testing.assert_allclose(p / np.linalg.norm(p, axis=1, keepdims=True), d, atol=1e-9)


def test_remesh_carries_every_surface_and_keeps_labels_unblended():
    sphere, f = _icosphere(3)
    target, tf = _icosphere(2)
    rot = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))[0]
    target = target @ rot.T  # a target that shares no vertices with the source
    white = sphere * np.array([0.6, 0.8, 0.7])  # any smooth non-spherical surface
    region = (sphere[:, 2] > 0).astype(np.int64) * 7
    patch = np.zeros_like(sphere)
    b = MeshBundle(
        faces=f,
        positions={"surf:sphere.reg": sphere, "surf:white": white, "patch:occip": patch},
        scalars={"morph:thickness": (2.0 + sphere[:, 0] / 100).astype(np.float32)},
        labels={"annot:region": region},
        masks={"label:cortex": sphere[:, 1] > -20, "patch:occip": np.ones(len(sphere), bool)},
        spherical={"surf:sphere.reg": np.zeros(3)},
        masked={"patch:occip": "patch:occip"},
    )
    out, look = remesh_via_sphere(b, target, tf)
    assert out.n_vertices == len(target) and np.array_equal(out.faces, tf)
    assert "patch:occip" not in out.positions and "patch:occip" not in out.masks
    np.testing.assert_allclose(np.linalg.norm(out.positions["surf:sphere.reg"], axis=1), 100.0)
    # white is a linear image of the sphere, so its remesh is the same image of the
    # interpolated (flat-triangle) point: within the chord sag of a level-3 mesh.
    np.testing.assert_allclose(
        out.positions["surf:white"], look.interpolate(sphere) * [0.6, 0.8, 0.7], atol=1e-9
    )
    assert np.abs(out.positions["surf:white"] - target * [0.6, 0.8, 0.7]).max() < 1.0
    assert set(np.unique(out.labels["annot:region"])) <= {0, 7}
    assert out.scalars["morph:thickness"].dtype == np.float32
    np.testing.assert_allclose(out.scalars["morph:thickness"], 2.0 + target[:, 0] / 100, atol=0.02)
