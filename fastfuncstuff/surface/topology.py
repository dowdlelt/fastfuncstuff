"""Changing a cortical mesh's topology: delete a vertex, split an edge.

Every surface of a FreeSurfer hemisphere -- white, pial, orig, smoothwm,
inflated, sphere, sphere.reg, the flat patch -- and every per-vertex file
(curv, sulc, thickness, the annotations, the labels) shares one vertex
numbering. Geometric edits never touch it. These operations do, so they work
on a :class:`MeshBundle` holding *all* of it, and change it all together: a
surface or overlay left behind would silently belong to another mesh.

* **Delete** is an edge collapse: the vertex merges into a neighbour, the two
  triangles on that edge go. Allowed only when the result is still a closed
  2-manifold (the link condition: the two endpoints share exactly the two
  opposite vertices, neither of which drops below valence 3) and no triangle
  turns over on **any** position set -- a flip on the sphere would corrupt
  registration to the template as surely as one on white corrupts sampling.
  Neighbours are tried shortest edge first.
* **Split** puts a new vertex at an edge's midpoint on every surface (back on
  the sphere for spherical ones) and turns the edge's two triangles into four.

Indices stay contiguous, as FreeSurfer's files require, by moving the last
vertex into a deleted one's slot: one other vertex is renumbered, and the
returned ``remap`` says so for anything held outside the bundle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MeshBundle:
    """One hemisphere's mesh and everything indexed by its vertices."""

    faces: np.ndarray  # (F, 3) int64
    #: Triangle surfaces and patch coordinates, by name: (V, 3).
    positions: dict[str, np.ndarray] = field(default_factory=dict)
    #: Per-vertex reals (curv, thickness...), averaged across a split.
    scalars: dict[str, np.ndarray] = field(default_factory=dict)
    #: Per-vertex integer labels (annotation indices), copied across a split.
    labels: dict[str, np.ndarray] = field(default_factory=dict)
    #: Per-vertex flags (cortex, in-patch, patch border), both ends across a split.
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    #: Position sets that lie on a sphere: a split's midpoint goes back onto it.
    spherical: set[str] = field(default_factory=set)
    #: Position sets that only mean something where a mask says so (a flat
    #: patch: coordinates outside it are zero), name -> mask name.
    masked: dict[str, str] = field(default_factory=dict)

    @property
    def n_vertices(self) -> int:
        return int(next(iter(self.positions.values())).shape[0])

    def copy(self) -> MeshBundle:
        return MeshBundle(
            self.faces.copy(),
            {k: v.copy() for k, v in self.positions.items()},
            {k: v.copy() for k, v in self.scalars.items()},
            {k: v.copy() for k, v in self.labels.items()},
            {k: v.copy() for k, v in self.masks.items()},
            set(self.spherical),
            dict(self.masked),
        )


def is_spherical(positions: np.ndarray, tolerance: float = 0.01) -> bool:
    """Whether points lie on a sphere about their centroid (radius spread < 1%)."""
    r = np.linalg.norm(positions - positions.mean(axis=0), axis=1)
    return bool(r.std() < tolerance * max(r.mean(), 1e-12))


def neighbours(faces: np.ndarray, v: int) -> np.ndarray:
    around = faces[np.any(faces == v, axis=1)]
    return np.setdiff1d(np.unique(around), [v])


def check_manifold(faces: np.ndarray, n_vertices: int) -> None:
    """Raise unless every edge has exactly two triangles and the surface is genus 0."""
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    if not np.all(counts == 2):
        raise ValueError("not a closed 2-manifold: an edge has other than two triangles")
    euler = n_vertices - counts.size + faces.shape[0]
    if euler != 2:
        raise ValueError(f"Euler characteristic {euler}, not 2: no longer a sphere topology")
    if np.unique(faces).size != n_vertices:
        raise ValueError("a vertex belongs to no triangle")


def _normals(p: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return np.cross(p[faces[:, 1]] - p[faces[:, 0]], p[faces[:, 2]] - p[faces[:, 0]])


def _compact(bundle: MeshBundle, gone: int) -> np.ndarray:
    """Drop vertex ``gone`` by moving the last vertex into its slot; return remap."""
    n = bundle.n_vertices
    last = n - 1
    remap = np.arange(n)
    remap[gone] = -1
    if gone != last:
        for store in (bundle.positions, bundle.scalars, bundle.labels, bundle.masks):
            for arr in store.values():
                arr[gone] = arr[last]
        bundle.faces[bundle.faces == last] = gone
        remap[last] = gone
    for store in (bundle.positions, bundle.scalars, bundle.labels, bundle.masks):
        for k in list(store):
            store[k] = store[k][:last]
    return remap


def collapse_vertex(bundle: MeshBundle, v: int) -> tuple[MeshBundle, np.ndarray, int]:
    """Delete vertex ``v`` by collapsing it into a neighbour.

    Returns ``(new bundle, remap, kept)``: ``remap[old] -> new`` (-1 for
    ``v``), and the neighbour it merged into, in new numbering.
    """
    faces = bundle.faces
    v = int(v)
    nbrs = neighbours(faces, v)
    if nbrs.size < 3:
        raise ValueError(f"vertex {v} has {nbrs.size} neighbours; nothing to collapse into")
    ref = next(iter(bundle.positions.values()))
    order = nbrs[np.argsort(np.linalg.norm(ref[nbrs] - ref[v], axis=1))]
    around = np.flatnonzero(np.any(faces == v, axis=1))
    for u in order:
        u = int(u)
        common = np.intersect1d(nbrs, neighbours(faces, u))
        if common.size != 2:
            continue  # link condition: collapsing would pinch the surface
        if any(neighbours(faces, int(c)).size <= 3 for c in common):
            continue  # an opposite vertex would be left with two triangles
        keep = around[~np.any(faces[around] == u, axis=1)]  # faces that survive, renamed
        renamed = faces[keep].copy()
        renamed[renamed == v] = u
        ok = True
        for name, p in bundle.positions.items():
            mask_name = bundle.masked.get(name)
            rows = keep
            new = renamed
            if mask_name is not None:
                inside = bundle.masks[mask_name]
                sel = np.all(inside[faces[keep]], axis=1) & np.all(inside[renamed], axis=1)
                rows, new = keep[sel], renamed[sel]
                if rows.size == 0:
                    continue
            before = _normals(p, faces[rows])
            after = _normals(p, new)
            if np.any(np.einsum("ij,ij->i", before, after) <= 0):
                ok = False
                break
        if not ok:
            continue  # a triangle would turn over on some surface
        out = bundle.copy()
        drop = around[np.any(faces[around] == u, axis=1)]
        out.faces = np.delete(out.faces, drop, axis=0)
        out.faces[out.faces == v] = u
        remap = _compact(out, v)
        check_manifold(out.faces, out.n_vertices)
        new_u = int(remap[u])
        remap[v] = new_u  # what v's attributes now belong to
        return out, remap, new_u
    raise ValueError(
        f"vertex {v} cannot be deleted here without pinching the surface or "
        "turning a triangle over on some surface; split nearby edges first"
    )


def split_edge(bundle: MeshBundle, a: int, b: int) -> tuple[MeshBundle, int]:
    """Insert a vertex at the middle of edge (a, b); return the bundle and its index."""
    a, b = int(a), int(b)
    faces = bundle.faces
    on = np.flatnonzero(np.any(faces == a, axis=1) & np.any(faces == b, axis=1))
    if on.size != 2:
        raise ValueError(f"({a}, {b}) is not an edge of the mesh")
    out = bundle.copy()
    m = bundle.n_vertices
    for name, p in out.positions.items():
        mid = 0.5 * (p[a] + p[b])
        if name in out.spherical:
            c = p.mean(axis=0)
            radius = 0.5 * (np.linalg.norm(p[a] - c) + np.linalg.norm(p[b] - c))
            mid = c + (mid - c) * radius / max(np.linalg.norm(mid - c), 1e-12)
        out.positions[name] = np.vstack([p, mid[None].astype(p.dtype)])
    for k, s in out.scalars.items():
        out.scalars[k] = np.append(s, s.dtype.type(0.5 * (float(s[a]) + float(s[b]))))
    for k, lab in out.labels.items():
        out.labels[k] = np.append(lab, lab[a])
    for k, msk in out.masks.items():
        out.masks[k] = np.append(msk, bool(msk[a]) and bool(msk[b]))
    new_faces = []
    for f in faces[on]:
        # Keep each triangle's winding: with the edge as p -> q in cyclic
        # order and r the third corner, (p, q, r) -> (p, m, r) + (m, q, r).
        i = int(np.flatnonzero(f == a)[0])
        j = (i + 1) % 3
        if f[j] == b:
            p_, q_, r_ = a, b, int(f[(i + 2) % 3])
        else:
            p_, q_, r_ = b, a, int(f[(i + 1) % 3])
        new_faces += [[p_, m, r_], [m, q_, r_]]
    out.faces = np.vstack([np.delete(faces, on, axis=0), np.asarray(new_faces, faces.dtype)])
    check_manifold(out.faces, out.n_vertices)
    return out, m


def longest_edge(bundle: MeshBundle, v: int, reference: str | None = None) -> tuple[int, int]:
    """The longest edge at ``v`` on ``reference`` (default: the first position set)."""
    p = bundle.positions[reference] if reference else next(iter(bundle.positions.values()))
    nbrs = neighbours(bundle.faces, int(v))
    u = int(nbrs[np.argmax(np.linalg.norm(p[nbrs] - p[int(v)], axis=1))])
    return int(v), u


__all__ = [
    "MeshBundle",
    "check_manifold",
    "collapse_vertex",
    "is_spherical",
    "longest_edge",
    "neighbours",
    "split_edge",
]
