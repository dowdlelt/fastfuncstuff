"""Where to read a volume for each vertex, and how to fold the reads back onto it.

The surface route of ``ffs_nwarp`` samples native EPI once, at points, through the
whole transform chain. This module decides the points and the averaging; nwarp does
the reading. Two modes:

* **point**: one read per vertex per depth -- the vertex itself.
* **footprint**: every vertex averages its own patch of cortex. Each triangle (on the
  depth surface being sampled) is cut into ``L x L`` equal sub-triangles, ``L`` chosen
  so a sub-triangle is no bigger than one voxel face, and each sub-triangle's centroid
  is read and goes to the triangle corner it is nearest in barycentric terms. That
  splits every triangle into three equal-area thirds -- the same patch
  ``mesh.vertex_areas`` measures -- so every read counts once, for one vertex. A mesh
  coarser than the voxels (onavg-ico64 is ~4 voxel faces per vertex at 0.8 mm) then
  still reads every voxel instead of one in four; [[Wang 2022]]'s missed voxels.

The averaging is a fixed sparse ``(V, P)`` matrix: motion changes where nwarp reads,
never which reads a vertex averages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .mesh import vertex_areas
from .profiles import equivolume_fraction

__all__ = ["SurfaceSampling", "build_sampling", "depth_surfaces"]


def depth_surfaces(
    white: np.ndarray,
    pial: np.ndarray,
    faces: np.ndarray,
    fractions,
    equivolume: bool = True,
) -> np.ndarray:
    """``(K, V, 3)`` surfaces at depth ``fractions`` (white 0 .. pial 1), equivolume."""
    w = np.asarray(white, np.float64)
    p = np.asarray(pial, np.float64)
    frac = np.atleast_1d(np.asarray(fractions, np.float64))
    if equivolume:
        rho = equivolume_fraction(
            frac[:, None], vertex_areas(w, faces)[None], vertex_areas(p, faces)[None]
        )
        # Outside the ribbon the fraction is a plain extension, as sample_depths does.
        rho = np.where((frac[:, None] >= 0) & (frac[:, None] <= 1), rho, frac[:, None])
    else:
        rho = np.broadcast_to(frac[:, None], (frac.size, w.shape[0]))
    return w[None] + rho[..., None] * (p - w)[None]


def _lattice(level: int) -> tuple[np.ndarray, np.ndarray]:
    """Centroids of the level x level sub-triangles (barycentric) and each corner's
    share of them: 1 for the nearest corner, split evenly on a tie (the triangle's own
    centroid is a sub-centroid when level % 3 == 1), so each corner gets exactly a third.

    Level 1 is the exception: its one sub-triangle is the triangle, and a single read
    at the centroid shared three ways leaves every vertex averaging a ring of
    centroids around it -- never itself -- which on a mesh finer than the voxels is
    an extra blur (the bands benchmark: 0.50 vs 0.57 amplitude at 3 mm). Instead each
    corner's third (corner, mid-edges, centroid) is read once at its own centroid.
    """
    if level == 1:
        a, b = 11.0 / 18.0, 3.5 / 18.0  # centroid of a corner's third
        bary = np.array([[a, b, b], [b, a, b], [b, b, a]])
        return bary, np.eye(3)
    pts = []
    for i in range(level):
        for j in range(level - i):
            # upward sub-triangle (i, j), (i+1, j), (i, j+1)
            pts.append((3 * i + 1, 3 * j + 1))
            if i + j < level - 1:
                # downward one (i+1, j), (i, j+1), (i+1, j+1)
                pts.append((3 * i + 2, 3 * j + 2))
    ij = np.asarray(pts, np.float64) / (3.0 * level)
    bary = np.c_[1.0 - ij.sum(1), ij]
    own = np.isclose(bary, bary.max(axis=1, keepdims=True), atol=1e-12).astype(np.float64)
    return bary, own / own.sum(axis=1, keepdims=True)


@dataclass
class SurfaceSampling:
    """Read ``points``; fold the reads onto ``(K, V)`` with ``operator``."""

    points: np.ndarray  # (P, 3) scanner mm, every depth's reads concatenated
    operator: sparse.csr_matrix  # (K * V, P), rows sum to 1 (or are empty)
    n_vertices: int
    fractions: np.ndarray  # (K,)
    #: Cortical area each row stands for (mm^2), 0 where the surface has none
    #: (white == pial on the medial wall leaves the patch empty at no depth, but a
    #: degenerate triangle can).
    area: np.ndarray  # (K, V)

    @property
    def n_depths(self) -> int:
        return int(self.fractions.size)

    def fold(self, reads: np.ndarray) -> np.ndarray:
        """Reads ``(P,)`` or ``(T, P)`` -> ``(K, V)`` or ``(K, V, T)``."""
        r = np.asarray(reads, np.float32)
        out = self.operator @ (r if r.ndim == 1 else r.T)
        k, v = self.n_depths, self.n_vertices
        return np.asarray(out, np.float32).reshape((k, v) if r.ndim == 1 else (k, v, -1))

    def coverage(self, reads: np.ndarray) -> np.ndarray:
        """Share of each vertex's footprint read inside the EPI in every frame (K, V).

        nwarp returns exactly 0 outside the source; a read that is 0 in any frame is
        counted as outside, the surface twin of the volume path's ``_min`` lane.
        """
        r = np.asarray(reads)
        inside = (np.abs(r) > 0) if r.ndim == 1 else (np.abs(r) > 0).all(axis=0)
        return self.fold(inside.astype(np.float32))


def build_sampling(
    white: np.ndarray,
    pial: np.ndarray,
    faces: np.ndarray,
    fractions=(0.5,),
    voxel_face: float | None = None,
    equivolume: bool = True,
    max_level: int = 16,
) -> SurfaceSampling:
    """Points and folding for a mesh at depth ``fractions``.

    ``voxel_face`` (mm^2) None means point sampling. Otherwise each triangle on each
    depth surface is cut finely enough that no read stands for more than one voxel
    face (capped at ``max_level``^2 reads per triangle).
    """
    faces = np.asarray(faces, np.int64)
    surf = depth_surfaces(white, pial, faces, fractions, equivolume)
    k_n, v_n = surf.shape[:2]
    frac = np.atleast_1d(np.asarray(fractions, np.float64))
    areas = np.stack([vertex_areas(s, faces) for s in surf])
    if voxel_face is None:
        pts = surf.reshape(-1, 3)
        op = sparse.identity(k_n * v_n, format="csr", dtype=np.float32)
        return SurfaceSampling(pts, op, v_n, frac, areas)

    all_pts, rows, cols, vals = [], [], [], []
    p0 = 0
    for k, s in enumerate(surf):
        corner = s[faces]  # (F, 3, 3)
        fa = 0.5 * np.linalg.norm(
            np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0]), axis=1
        )
        level = np.clip(np.ceil(np.sqrt(fa / voxel_face)), 1, max_level).astype(np.int64)
        for lv in np.unique(level):
            fsel = np.flatnonzero(level == lv)
            bary, share = _lattice(int(lv))
            pts = np.einsum("sc,fcx->fsx", bary, corner[fsel]).reshape(-1, 3)
            n_s = bary.shape[0]
            s_idx, c_idx = np.nonzero(share)  # (sample, corner) pairs that own it
            all_pts.append(pts)
            rows.append(k * v_n + faces[fsel][:, c_idx].reshape(-1))
            cols.append(p0 + (np.arange(fsel.size)[:, None] * n_s + s_idx[None]).reshape(-1))
            vals.append((fa[fsel, None] / n_s * share[s_idx, c_idx][None]).reshape(-1))
            p0 += pts.shape[0]
    rows_a, cols_a, vals_a = (np.concatenate(x) for x in (rows, cols, vals))
    op = sparse.csr_matrix((vals_a, (rows_a, cols_a)), shape=(k_n * v_n, p0))
    total = np.asarray(op.sum(axis=1)).ravel()
    scale = np.divide(1.0, total, out=np.zeros_like(total), where=total > 0)
    op = (sparse.diags(scale) @ op).tocsr().astype(np.float32)
    return SurfaceSampling(np.concatenate(all_pts), op, v_n, frac, areas)
