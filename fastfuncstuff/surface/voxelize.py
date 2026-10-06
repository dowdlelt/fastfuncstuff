"""Closed triangle meshes onto voxel grids: solid fills and exact distances.

Sampling a mesh into a volume -- stepping points along normals and marking the
voxels they land in -- leaves holes wherever no sample happens to fall, and the
finer the grid the more mesh densification it takes to close them. A closed,
oriented mesh does not need sampling: every voxel centre is either inside it or
not, which a winding number answers exactly at any resolution. Distances are
likewise exact point-to-triangle distances, not distances to the nearest
sampled point.

Arrays are nibabel-ordered ``(X, Y, Z)`` with a voxel -> world ``affine``, as
everywhere else in :mod:`fastfuncstuff.surface`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree

from fastfuncstuff.memory import (
    bytes_per_point_closest_triangle,
    get_available_memory,
    saturating_point_count,
)


def _to_index(affine: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(np.asarray(affine, np.float64))
    return np.asarray(xyz, np.float64) @ inv[:3, :3].T + inv[:3, 3]


def _ray_crossings(
    ijk: np.ndarray, faces: np.ndarray, shape: tuple[int, int, int], flip: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Where each grid column's +z ray crosses the mesh: ``(x, y, z_cross, sign)``.

    A column at integer ``(x, y)`` crosses a triangle when the point lies in the
    triangle's xy-projection. A column through a shared edge or vertex must be
    counted by exactly one of the triangles there, or the winding number is off
    by one along the whole rest of the column; the rasteriser's top-left rule
    does that, once each triangle is put in counter-clockwise order. ``sign`` is
    +1 where the ray leaves the solid (outward normal along +z).
    """
    nx, ny, _ = shape
    a, b, c = (ijk[faces[:, k]] for k in range(3))
    area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    keep = area2 != 0  # edge-on to the ray: never crossed
    a, b, c, area2 = a[keep], b[keep], c[keep], area2[keep]
    sign = np.sign(area2).astype(np.int8)
    if flip:
        sign = -sign
    # Counter-clockwise in xy for the edge tests; the crossing sign is already taken.
    cw = area2 < 0
    b, c = np.where(cw[:, None], c, b), np.where(cw[:, None], b, c)
    area2 = np.abs(area2)

    lo = np.minimum(np.minimum(a, b), c)
    hi = np.maximum(np.maximum(a, b), c)
    x0 = np.clip(np.ceil(lo[:, 0]), 0, nx).astype(np.int64)
    x1 = np.clip(np.floor(hi[:, 0]), -1, nx - 1).astype(np.int64)
    y0 = np.clip(np.ceil(lo[:, 1]), 0, ny).astype(np.int64)
    y1 = np.clip(np.floor(hi[:, 1]), -1, ny - 1).astype(np.int64)
    wx = np.maximum(x1 - x0 + 1, 0)
    wy = np.maximum(y1 - y0 + 1, 0)
    n = wx * wy
    tri = np.repeat(np.arange(n.size), n)
    local = np.arange(tri.size) - np.repeat(np.cumsum(n) - n, n)
    px = (x0[tri] + local % wx[tri]).astype(np.float64)
    py = (y0[tri] + local // wx[tri]).astype(np.float64)

    def edge(p, q):
        # >0 when the column is left of p->q, i.e. inside for a CCW triangle.
        ex, ey = q[tri, 0] - p[tri, 0], q[tri, 1] - p[tri, 1]
        w = ex * (py - p[tri, 1]) - ey * (px - p[tri, 0])
        # Top-left rule: an edge and its reverse give opposite answers, so a
        # shared edge belongs to exactly one side.
        owns = (ey < 0) | ((ey == 0) & (ex > 0))
        return w, (w > 0) | ((w == 0) & owns)

    w0, in0 = edge(b, c)  # weight of a
    w1, in1 = edge(c, a)  # weight of b
    w2, in2 = edge(a, b)  # weight of c
    hit = in0 & in1 & in2
    tri, px, py = tri[hit], px[hit], py[hit]
    z = (w0[hit] * a[tri, 2] + w1[hit] * b[tri, 2] + w2[hit] * c[tri, 2]) / area2[tri]
    return px.astype(np.int64), py.astype(np.int64), z, sign[tri]


def winding_number(
    vertices: np.ndarray,
    faces: np.ndarray,
    affine: np.ndarray,
    shape: tuple[int, int, int],
    device: torch.device | None = None,
) -> np.ndarray:
    """Winding number of a closed mesh at every voxel centre, ``(X, Y, Z)`` int8.

    1 inside an outward-oriented closed mesh (FreeSurfer's winding), 0 outside.
    Self-intersections (a pial surface folded through itself in a tight sulcus)
    give 2 where the sheets overlap rather than flipping inside to outside the
    way a crossing-parity test would; ``> 0`` is the robust inside test.
    """
    device = device or torch.device("cpu")
    shape = tuple(int(s) for s in shape)
    nx, ny, nz = shape
    ijk = _to_index(affine, vertices)
    # A mirrored affine turns the outward normals inward in index space.
    flip = bool(np.linalg.det(np.asarray(affine, np.float64)[:3, :3]) < 0)
    px, py, zc, sign = _ray_crossings(ijk, np.asarray(faces, np.int64), shape, flip)
    # Voxels k < z_cross see this crossing ahead of them along the ray.
    last = np.ceil(zc).astype(np.int64) - 1
    ahead = last >= 0
    px, py, last, sign = px[ahead], py[ahead], np.minimum(last[ahead], nz - 1), sign[ahead]

    out = np.empty(shape, np.int8)
    # (x-slab, Y, Z) int16 delta + its cumsum, both on device.
    per_x = ny * nz * 2 * 2
    step = max(1, min(nx, get_available_memory(device, empty_cache=False) // max(per_x, 1)))
    order = np.argsort(px, kind="stable")
    px, py, last, sign = px[order], py[order], last[order], sign[order]
    bounds = np.searchsorted(px, np.arange(0, nx + step, step))
    for i, x0 in enumerate(range(0, nx, step)):
        x1 = min(nx, x0 + step)
        s, e = bounds[i], bounds[i + 1]
        delta = torch.zeros((x1 - x0) * ny * nz, dtype=torch.int16, device=device)
        flat = ((px[s:e] - x0) * ny + py[s:e]) * nz + last[s:e]
        delta.index_add_(
            0,
            torch.as_tensor(flat, device=device),
            torch.as_tensor(sign[s:e], dtype=torch.int16, device=device),
        )
        # Reverse cumsum along z: the crossings at or beyond each voxel.
        w = delta.view(x1 - x0, ny, nz).flip(2).cumsum(2, dtype=torch.int16).flip(2)
        out[x0:x1] = w.clamp(-127, 127).to(torch.int8).cpu().numpy()
    return out


def _closest_on_triangles(
    p: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closest point on each triangle to ``p``: ``(squared distance, barycentrics)``.

    Ericson's Voronoi-region test (Real-Time Collision Detection 5.1.5), every
    region evaluated and the first matching one kept, so it vectorises.
    Broadcasting: ``p`` ``(..., 3)`` against ``a``/``b``/``c`` ``(..., 3)``.
    """
    ab, ac, ap = b - a, c - a, p - a
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)
    bp = p - b
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)
    cp = p - c
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    tiny = torch.finfo(p.dtype).tiny

    def safe(num, den):
        return num / torch.where(den.abs() > tiny, den, torch.full_like(den, tiny))

    denom = va + vb + vc
    v = safe(vb, denom)
    w = safe(vc, denom)
    u = 1 - v - w
    # Later assignments win, so go from the least to the most specific region.
    t = safe(d4 - d3, (d4 - d3) + (d5 - d6))
    bc = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
    u, v, w = torch.where(bc, 0.0, u), torch.where(bc, 1 - t, v), torch.where(bc, t, w)
    t = safe(d2, d2 - d6)
    ca = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    u, v, w = torch.where(ca, 1 - t, u), torch.where(ca, 0.0, v), torch.where(ca, t, w)
    t = safe(d1, d1 - d3)
    abr = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    u, v, w = torch.where(abr, 1 - t, u), torch.where(abr, t, v), torch.where(abr, 0.0, w)
    cr = (d6 >= 0) & (d5 <= d6)
    u, v, w = torch.where(cr, 0.0, u), torch.where(cr, 0.0, v), torch.where(cr, 1.0, w)
    br = (d3 >= 0) & (d4 <= d3)
    u, v, w = torch.where(br, 0.0, u), torch.where(br, 1.0, v), torch.where(br, 0.0, w)
    ar = (d1 <= 0) & (d2 <= 0)
    u, v, w = torch.where(ar, 1.0, u), torch.where(ar, 0.0, v), torch.where(ar, 0.0, w)
    q = u[..., None] * a + v[..., None] * b + w[..., None] * c
    return ((p - q) ** 2).sum(-1), torch.stack([u, v, w], -1)


@dataclass
class ClosestPoints:
    """Nearest point on a mesh for each query point."""

    distance: np.ndarray  # (N,) mm
    face: np.ndarray  # (N,) int64
    bary: np.ndarray  # (N, 3) weights of faces[face]

    def interpolate(self, faces: np.ndarray, values: np.ndarray) -> np.ndarray:
        """A per-vertex quantity at the closest points."""
        f = np.asarray(faces)[self.face]
        return (np.asarray(values)[f] * self.bary).sum(-1)


class MeshDistance:
    """Point-to-mesh distance over the faces whose centroids are nearest the query.

    The closest point on a mesh lies on one of the query's nearest faces by
    centroid unless the triangles are badly shaped. On FreeSurfer surfaces,
    with points in the cortical ribbon, 24 candidates left a worst error of
    5 microns on pial (1 in 4000 points, against an all-faces scan) and none on
    white; the nearest *vertices'* incident faces needed 16 vertices -- 160
    padded candidates -- for the same accuracy.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, k: int = 24) -> None:
        self.vertices = np.asarray(vertices, np.float64)
        self.faces = np.asarray(faces, np.int64)
        self.k = int(min(k, self.faces.shape[0]))
        self.tree = cKDTree(self.vertices[self.faces].mean(1))

    def __call__(
        self, points: np.ndarray, device: torch.device | None = None, workers: int = -1
    ) -> ClosestPoints:
        device = device or torch.device("cpu")
        pts = np.asarray(points, np.float64)
        n = pts.shape[0]
        n_cand = self.k
        dist = np.empty(n, np.float32)
        face = np.empty(n, np.int64)
        bary = np.empty((n, 3), np.float32)
        if n == 0:
            return ClosestPoints(dist, face, bary)
        budget = get_available_memory(device, empty_cache=False)
        # Free memory says what fits; past saturation a bigger chunk only costs
        # RAM -- a whole-brain ribbon sized by memory alone peaked at 38 GB on CPU.
        step = min(
            budget // bytes_per_point_closest_triangle(n_cand),
            saturating_point_count(device) // n_cand,
        )
        step = int(max(1024, min(n, step)))
        verts = torch.as_tensor(self.vertices, dtype=torch.float32, device=device)
        faces = torch.as_tensor(self.faces, device=device)
        for s in range(0, n, step):
            e = min(n, s + step)
            _, near = self.tree.query(pts[s:e], k=self.k, workers=workers)
            cand_t = torch.as_tensor(np.asarray(near).reshape(e - s, n_cand), device=device)
            tri = faces[cand_t]  # (B, C, 3)
            # Local frame per query: subtracting it before the float32 maths
            # keeps sub-micron precision 100 mm from the origin.
            origin = torch.as_tensor(pts[s:e], dtype=torch.float32, device=device)
            p = torch.zeros_like(origin)[:, None, :]
            a, b, c = (verts[tri[..., j]] - origin[:, None, :] for j in range(3))
            d2, w = _closest_on_triangles(p, a, b, c)
            best = d2.argmin(1)
            rows = torch.arange(e - s, device=device)
            dist[s:e] = d2[rows, best].sqrt().cpu().numpy()
            face[s:e] = cand_t[rows, best].cpu().numpy()
            bary[s:e] = w[rows, best].cpu().numpy()
        return ClosestPoints(dist, face, bary)


__all__ = ["ClosestPoints", "MeshDistance", "winding_number"]
