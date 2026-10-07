"""A subject's mesh rebuilt on another vertex set, through the registered sphere.

``?h.sphere.reg`` puts every subject on one common sphere (fsaverage's). Any mesh
whose vertices live on that sphere -- SUMA's std.N icosahedra, the onavg template,
a mesh made for one acquisition -- is placed in the subject by finding the subject
triangle each target vertex falls in, on the sphere, and interpolating every
surface (white, pial, ...) with the same barycentric weights. This is what AFNI's
MapIcosahedron does with ``-morph sphere.reg``; here it is one primitive for every
target, see [[Surfaces as an analysis space]] section 1b.

Weights are the radial-projection barycentrics: the target direction ``t`` written as
``w0 a + w1 b + w2 c`` over the triangle's corners, rescaled to sum to one. They are
exact on each corner and continuous across edges, which is what keeps the remeshed
surface free of cracks.
"""

from __future__ import annotations

import numpy as np

from .topology import MeshBundle

__all__ = ["SphereLookup", "remesh_via_sphere", "sphere_lookup"]


class SphereLookup:
    """Each target point's containing triangle and barycentric weights on a mesh."""

    def __init__(self, faces: np.ndarray, face: np.ndarray, weights: np.ndarray):
        self.faces = faces  # (F, 3) of the mesh that was searched
        self.face = face  # (N,) containing face per target
        self.weights = weights  # (N, 3), rows sum to 1

    @property
    def corners(self) -> np.ndarray:
        return self.faces[self.face]

    def interpolate(self, values: np.ndarray) -> np.ndarray:
        """Per-vertex ``values`` (V,) or (V, k) at the target points."""
        v = np.asarray(values)
        w = self.weights if v.ndim == 1 else self.weights[..., None]
        return (v[self.corners] * w).sum(axis=1).astype(np.result_type(v.dtype, np.float32))

    def nearest(self, values: np.ndarray) -> np.ndarray:
        """The heaviest corner's value: labels and masks are never blended."""
        pick = self.corners[np.arange(len(self.face)), np.argmax(self.weights, axis=1)]
        return np.asarray(values)[pick]


def _unit(x: np.ndarray, centre: np.ndarray) -> np.ndarray:
    d = np.asarray(x, np.float64) - centre
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _try(units: np.ndarray, faces: np.ndarray, targets: np.ndarray, cand: np.ndarray):
    """Radial barycentrics of every target against its candidate faces (N, k)."""
    corners = units[faces[cand]]  # (N, k, 3 corners, 3 xyz)
    m = np.swapaxes(corners, -1, -2)  # columns are the corners
    rhs = np.broadcast_to(targets[:, None, :, None], (*cand.shape, 3, 1))
    with np.errstate(all="ignore"):
        w = np.linalg.solve(m, rhs)[..., 0]  # (N, k, 3)
    w = np.nan_to_num(w, nan=-np.inf)
    worst = w.min(axis=-1)  # >= 0 inside, the more negative the further out
    best = np.argmax(worst, axis=1)
    rows = np.arange(len(targets))
    return cand[rows, best], w[rows, best], worst[rows, best]


def sphere_lookup(
    sphere: np.ndarray,
    faces: np.ndarray,
    targets: np.ndarray,
    centre: np.ndarray | None = None,
    target_centre: np.ndarray | None = None,
) -> SphereLookup:
    """Find, for each target point on the sphere, its triangle on ``sphere`` and weights.

    ``sphere`` (V, 3) and ``targets`` (N, 3) may have different radii and centres
    (a subject's sphere.reg in scanner mm, a template's about the origin): both are
    reduced to directions. Candidates are the faces around the nearest face
    centroids, widened for the few targets a skinny triangle hides.
    """
    from scipy.spatial import KDTree

    faces = np.asarray(faces, np.int64)
    units = _unit(sphere, np.zeros(3) if centre is None else np.asarray(centre))
    tgt = _unit(targets, np.zeros(3) if target_centre is None else np.asarray(target_centre))
    cent = units[faces].mean(axis=1)
    tree = KDTree(cent)
    face = np.empty(len(tgt), np.int64)
    weights = np.empty((len(tgt), 3))
    todo = np.arange(len(tgt))
    for k in (8, 64, 512):
        k = min(k, len(faces))
        _, cand = tree.query(tgt[todo], k=k)
        cand = np.asarray(cand).reshape(len(todo), k)
        f, w, worst = _try(units, faces, tgt[todo], cand)
        ok = worst >= -1e-9
        last = k == min(512, len(faces))
        keep = np.ones_like(ok) if last else ok
        face[todo[keep]], weights[todo[keep]] = f[keep], w[keep]
        todo = todo[~keep]
        if not todo.size:
            break
    # Outside-by-rounding cases sit on an edge: clip the sliver, then normalise.
    weights = np.clip(weights, 0.0, None)
    weights /= weights.sum(axis=1, keepdims=True)
    return SphereLookup(faces, face, weights)


def remesh_via_sphere(
    bundle: MeshBundle,
    target_sphere: np.ndarray,
    target_faces: np.ndarray,
    sphere: str = "surf:sphere.reg",
    target_centre: np.ndarray | None = None,
) -> tuple[MeshBundle, SphereLookup]:
    """``bundle`` rebuilt on the target mesh's vertices, placed through ``sphere``.

    Every position set and scalar is interpolated barycentrically; spherical sets go
    back onto their own sphere; labels and masks take the heaviest corner. Flat
    patches are dropped: a patch is a cut of the native mesh and does not carry over.
    The returned lookup is what was used, for mapping further per-vertex data.
    """
    if sphere not in bundle.positions:
        raise KeyError(f"{sphere} is not in the bundle (has {sorted(bundle.positions)})")
    centre = bundle.spherical.get(sphere, np.zeros(3))
    look = sphere_lookup(
        bundle.positions[sphere], bundle.faces, target_sphere, centre, target_centre
    )
    out = MeshBundle(faces=np.asarray(target_faces, np.int64))
    for name, pos in bundle.positions.items():
        if name in bundle.masked:
            continue
        p = look.interpolate(pos).astype(np.float64)
        if name in bundle.spherical:
            c = bundle.spherical[name]
            radius = np.linalg.norm(pos - c, axis=1).mean()
            p = c + _unit(p, c) * radius
            out.spherical[name] = c.copy()
        out.positions[name] = p
    out.scalars = {k: look.interpolate(v).astype(v.dtype) for k, v in bundle.scalars.items()}
    out.labels = {k: look.nearest(v) for k, v in bundle.labels.items()}
    out.masks = {
        k: look.nearest(v)
        for k, v in bundle.masks.items()
        if k not in set(bundle.masked.values()) and not k.startswith("patchborder:")
    }
    return out, look
