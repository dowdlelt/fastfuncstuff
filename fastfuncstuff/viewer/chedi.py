"""CHEDI's view: a small piece of cortex laid flat, the anatomy sampled onto it at a depth.

Looking at the cortex as a wall shows what a slice cannot: a stretch of pial
sitting in dura is a bright patch at depth 0.9, a pial that stops short is
grey matter still showing at 1.1. The view is for *seeing where the mesh is
wrong*, not for measuring, so the flattening only has to be fast and
one-to-one, not area-true: it is a cap of the hemisphere's sphere, unrolled
azimuthally about the centre vertex and scaled so millimetres near the centre
are roughly cortical millimetres.

The work is split so moving through depth is nearly free. Building a patch
(on a new centre) decides, once, which three vertices and weights every pixel
reads; a depth is then ``white + d * (pial - white)`` per pixel and one
trilinear read. See ``../fmri_wiki/concepts/CHEDI.md`` for the design.

Nothing here imports Qt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastfuncstuff.io.freesurfer import Hemisphere

#: Vertices used to estimate millimetres per radian of sphere at the centre.
_SCALE_NEIGHBOURS = 300


@dataclass
class Patch:
    """One flattened piece of a hemisphere and how its pixels read the mesh."""

    hemi: str
    centre: int
    #: Half the patch's width, flat mm; the image spans ``[-half, half]``.
    half_mm: float
    #: Vertex ids in the patch, and their flat (x right, y up) positions in mm.
    ids: np.ndarray
    uv: np.ndarray
    #: Per pixel ``(H, W, 3)``: the three vertices it interpolates, and weights.
    corners: np.ndarray
    weights: np.ndarray
    inside: np.ndarray
    #: Mesh faces with all three corners in the patch, hemisphere vertex ids.
    faces: np.ndarray
    #: What the flattening was taken from: ``sphere``, or ``inflated`` without one.
    source: str

    @property
    def size(self) -> int:
        return int(self.inside.shape[0])

    @property
    def mm_per_pixel(self) -> float:
        return 2.0 * self.half_mm / self.size

    def to_pixels(self, uv: np.ndarray) -> np.ndarray:
        """Flat mm ``(..., 2)`` to fractional (row, col), pixel ``r`` centred at ``r + 0.5``."""
        uv = np.asarray(uv, np.float64)
        px = self.mm_per_pixel
        return np.stack([(self.half_mm - uv[..., 1]) / px, (uv[..., 0] + self.half_mm) / px], -1)

    def uv_of(self, n_vertices: int) -> np.ndarray:
        """Flat positions for every hemisphere vertex, NaN outside the patch."""
        out = np.full((n_vertices, 2), np.nan)
        out[self.ids] = self.uv
        return out


def _tangent_basis(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(right, up) spanning the plane normal to ``c``, seen from outside.

    Up is the scanner's superior direction projected into the plane, so the
    patch is the right way up where that means anything; on the top of the
    brain, where it does not, anterior is up instead.
    """
    for ref in ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)):
        up = np.asarray(ref) - np.dot(ref, c) * c
        if np.linalg.norm(up) > 0.3:
            break
    up /= np.linalg.norm(up)
    # Looking along -c (from outside), right = forward x up = (-c) x up.
    right = np.cross(up, c)
    return right / np.linalg.norm(right), up


def build_patch(
    hemi: Hemisphere,
    centre: int,
    half_mm: float = 25.0,
    size: int = 256,
) -> Patch:
    """Flatten the cortex around vertex ``centre`` into a ``size`` x ``size`` patch.

    Needs ``?h.sphere``; without it, the inflated surface is projected flat
    onto the plane through the centre instead (folds are already smoothed out
    there, so a small piece of it is nearly a plane).
    """
    from scipy.spatial import Delaunay

    white, pial = hemi.states["white"], hemi.states["pial"]
    mid = 0.5 * (white.astype(np.float64) + pial)
    source = "sphere" if "sphere" in hemi.states else "inflated"
    if source not in hemi.states:
        raise ValueError(f"{hemi.name}: no sphere or inflated surface to flatten the cortex with")
    shape = hemi.states[source].astype(np.float64)

    if source == "sphere":
        d = shape - shape.mean(axis=0)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        c = d[centre]
        closeness = d @ c
        right, up = _tangent_basis(c)
        nearest = np.argpartition(-closeness, _SCALE_NEIGHBOURS)[:_SCALE_NEIGHBOURS]
        theta_n = np.arccos(np.clip(closeness[nearest], -1.0, 1.0))
        r_n = np.linalg.norm(mid[nearest] - mid[centre], axis=1)
        ok = theta_n > 1e-6
        scale = float(np.median(r_n[ok] / theta_n[ok]))  # cortical mm per radian
        # Far enough for the square's corners, with a margin of triangles.
        reach = 1.2 * np.sqrt(2.0) * half_mm / scale
        sel = np.flatnonzero(closeness > np.cos(min(reach, np.pi * 0.9)))
        theta = np.arccos(np.clip(closeness[sel], -1.0, 1.0))
        tang = d[sel] - closeness[sel, None] * c
        az = np.arctan2(tang @ up, tang @ right)
        uv = scale * theta[:, None] * np.stack([np.cos(az), np.sin(az)], -1)
    else:
        from fastfuncstuff.surface.mesh import MeshTopology, vertex_normals

        normals = vertex_normals(shape, MeshTopology.from_faces(hemi.faces, hemi.n_vertices))
        c = normals[centre] / np.linalg.norm(normals[centre])
        right, up = _tangent_basis(c)
        offset = shape - shape[centre]
        dist = np.linalg.norm(offset, axis=1)
        nearest = np.argpartition(dist, _SCALE_NEIGHBOURS)[:_SCALE_NEIGHBOURS]
        r_n = np.linalg.norm(mid[nearest] - mid[centre], axis=1)
        ok = dist[nearest] > 1e-6
        scale = float(np.median(r_n[ok] / dist[nearest][ok]))  # cortical mm per inflated mm
        # Facing the viewer: the far side of a thin lobe must not fold in.
        sel = np.flatnonzero((dist * scale < 1.2 * np.sqrt(2.0) * half_mm) & (normals @ c > 0.2))
        uv = scale * np.stack([offset[sel] @ right, offset[sel] @ up], -1)

    px = 2.0 * half_mm / size
    centres = -half_mm + (np.arange(size) + 0.5) * px
    gx, gy = np.meshgrid(centres, centres[::-1])
    pts = np.stack([gx.ravel(), gy.ravel()], -1)
    tri = Delaunay(uv)
    simplex = tri.find_simplex(pts)
    inside = simplex >= 0
    s = np.where(inside, simplex, 0)
    transform = tri.transform[s]
    b = np.einsum("nij,nj->ni", transform[:, :2], pts - transform[:, 2])
    weights = np.concatenate([b, 1.0 - b.sum(axis=1, keepdims=True)], axis=1)
    corners = sel[tri.simplices[s]]
    # Delaunay fills the convex hull; a long sliver across a concave edge of
    # the cap is not cortex.
    corner_uv = uv[tri.simplices]
    edges = np.linalg.norm(corner_uv - np.roll(corner_uv, 1, axis=1), axis=-1).max(axis=1)
    inside &= edges[s] < 4.0 * np.median(edges)

    member = np.zeros(hemi.n_vertices, bool)
    member[sel] = True
    faces = hemi.faces[member[hemi.faces].all(axis=1)]
    return Patch(
        hemi=hemi.name,
        centre=int(centre),
        half_mm=float(half_mm),
        ids=sel,
        uv=uv,
        corners=corners.reshape(size, size, 3),
        weights=weights.reshape(size, size, 3).astype(np.float32),
        inside=inside.reshape(size, size),
        faces=faces,
        source=source,
    )


class PatchSampler:
    """Depth samples of one patch, with the per-pixel mesh points cached.

    The interpolated white point and white-to-pial vector of every pixel are
    kept until the mesh moves (``version``), so stepping through depth costs
    one multiply-add and one trilinear read.
    """

    def __init__(self, patch: Patch) -> None:
        self.patch = patch
        self._points: tuple[object, np.ndarray, np.ndarray] | None = None

    def points(self, hemi: Hemisphere, depth: float, version: object = None) -> np.ndarray:
        """Scanner mm of every inside pixel at ``depth`` (0 white, 1 pial; beyond extrapolates)."""
        cached = self._points
        if cached is None or cached[0] != version or version is None:
            p = self.patch
            idx = p.corners[p.inside]
            w = p.weights[p.inside][..., None].astype(np.float64)
            white = hemi.states["white"].astype(np.float64)
            pial = hemi.states["pial"].astype(np.float64)
            base = (white[idx] * w).sum(axis=1)
            span = ((pial - white)[idx] * w).sum(axis=1)
            cached = self._points = (version, base, span)
        _, base, span = cached
        return base + float(depth) * span

    def sample(self, hemi: Hemisphere, depth: float, sampler, version: object = None) -> np.ndarray:
        """The anatomy at ``depth`` as an ``(H, W)`` image, NaN off the patch."""
        p = self.patch
        out = np.full(p.inside.shape, np.nan, np.float32)
        out[p.inside] = sampler(self.points(hemi, depth, version))
        return out


def visible_vertices(patch: Patch) -> np.ndarray:
    """Patch vertices inside the drawn square: the only ones a CHEDI edit may move.

    The patch reaches past the square's edge (so its corners are covered);
    those extra vertices are not on screen, and moving what cannot be seen is
    the one thing an editor must not do.
    """
    inside = np.all(np.abs(patch.uv) <= patch.half_mm, axis=1)
    return patch.ids[inside]


def adjacency(faces: np.ndarray, n_vertices: int):
    """Vertex-to-vertex adjacency of a mesh, as a boolean CSR matrix."""
    import scipy.sparse as sp

    f = np.asarray(faces, np.int64)
    rows = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    adj = sp.csr_matrix((np.ones(rows.size, bool), (rows, cols)), shape=(n_vertices, n_vertices))
    adj.sum_duplicates()
    return adj


def _neighbour_count(adj, mask: np.ndarray) -> np.ndarray:
    return np.asarray(adj @ mask.astype(np.int32)).ravel()


def window_select(values: np.ndarray, level: float, width: float) -> np.ndarray:
    """Which values lie in the window ``level +- width / 2``: the ctrl+drag selection."""
    half = 0.5 * abs(float(width))
    return np.abs(np.asarray(values, np.float64) - float(level)) <= half


def erode(selected: np.ndarray, adj, within: np.ndarray) -> np.ndarray:
    """Drop selected vertices with any unselected visible neighbour.

    Only neighbours in ``within`` (the visible vertices, as a mask) count: the
    edge of the screen is not the edge of the selection.
    """
    outside = within & ~selected
    return selected & (_neighbour_count(adj, outside) == 0)


def dilate(selected: np.ndarray, adj, within: np.ndarray) -> np.ndarray:
    """Add every visible neighbour of a selected vertex."""
    return selected | (within & (_neighbour_count(adj, selected) > 0))


def drop_isolated(selected: np.ndarray, adj) -> np.ndarray:
    """Drop selected vertices with no selected neighbour: specks a threshold left behind."""
    return selected & (_neighbour_count(adj, selected) > 0)


#: Depths, relative to the one shown, a cluster feature reads: a short
#: profile across the boundary, so "bright here and above" and "bright only
#: here" are different patterns.
PROFILE_OFFSETS = (-0.15, 0.0, 0.15)
#: The most clusters: one key each, 1-9 then 0.
MAX_CLUSTERS = 10


def cluster_features(
    profile: np.ndarray, fold: np.ndarray, uv: np.ndarray, half_mm: float, spatial: float
) -> np.ndarray:
    """``(N, F)`` features: the depth profile, gyrus/sulcus, and (weighted) flat position.

    The profile is standardised with one mean and SD across all its depths,
    so its *shape* survives -- per-depth scaling would make "brighter above"
    look the same as flat. Folding is +-1, worth about one SD of contrast, so
    it splits a cluster only when the contrast alone does not. Position is
    scaled to +-``spatial`` across the patch: 0 ignores where a point is, 1
    makes neighbours strongly prefer the same cluster.
    """
    v = np.asarray(profile, np.float64)
    if v.ndim == 1:
        v = v[:, None]
    finite = v[np.isfinite(v)]
    mu = float(finite.mean()) if finite.size else 0.0
    sd = float(finite.std()) if finite.size else 1.0
    z = np.nan_to_num((v - mu) / (sd or 1.0))
    parts = [z, np.asarray(fold, np.float64)[:, None]]
    if spatial > 0:
        parts.append(np.asarray(uv, np.float64) / max(half_mm, 1e-6) * spatial)
    return np.concatenate(parts, axis=1)


def kmeans(x: np.ndarray, k: int, seed: int = 0, iters: int = 50) -> np.ndarray:
    """Labels ``0..k-1`` from k-means++ seeding and Lloyd steps; deterministic for a seed.

    Small and dependency-free: a patch has a few thousand points and a
    handful of features, and the same press must give the same clusters.
    """
    x = np.asarray(x, np.float64)
    n = x.shape[0]
    k = int(max(1, min(k, n)))
    rng = np.random.default_rng(seed)
    centres = [x[rng.integers(n)]]
    d2 = ((x - centres[0]) ** 2).sum(axis=1)
    for _ in range(1, k):
        total = d2.sum()
        pick = rng.integers(n) if total <= 0 else rng.choice(n, p=d2 / total)
        centres.append(x[pick])
        d2 = np.minimum(d2, ((x - x[pick]) ** 2).sum(axis=1))
    c = np.stack(centres)
    labels = np.zeros(n, np.int64)
    xx = (x * x).sum(axis=1)[:, None]
    for step in range(iters):
        # |x - c|^2 as one matrix product, not an (N, k, F) difference array.
        dist = xx - 2.0 * x @ c.T + (c * c).sum(axis=1)[None, :]
        new = dist.argmin(axis=1)
        if step and np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            members = labels == j
            if members.any():
                c[j] = x[members].mean(axis=0)
            else:
                # An emptied cluster takes the point worst served by the rest.
                far = int(dist[np.arange(n), labels].argmax())
                c[j] = x[far]
                labels[far] = j
    return labels


def order_clusters(labels: np.ndarray, value: np.ndarray, k: int) -> np.ndarray:
    """Relabel so cluster 0 has the lowest mean ``value`` and k-1 the highest.

    Keys then mean the same thing after a re-run: the last one is always the
    brightest -- the dura -- whatever order k-means happened to find them in.
    """
    means = np.array(
        [np.nanmean(value[labels == j]) if np.any(labels == j) else np.inf for j in range(k)]
    )
    rank = np.empty(k, np.int64)
    rank[np.argsort(means, kind="stable")] = np.arange(k)
    return rank[labels]


__all__ = [
    "MAX_CLUSTERS",
    "PROFILE_OFFSETS",
    "cluster_features",
    "kmeans",
    "order_clusters",
    "Patch",
    "PatchSampler",
    "adjacency",
    "build_patch",
    "dilate",
    "drop_isolated",
    "erode",
    "visible_vertices",
    "window_select",
]
