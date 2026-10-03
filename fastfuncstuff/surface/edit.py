"""Snapping surface edits: the eye says roughly where, the image says exactly.

A drag on a slice is a coarse instruction -- "this stretch of pial is ~1 mm
too far out" -- and the brush turns it into a smooth displacement of every
vertex in a patch of cortex, in 3-D, not just on the slice being looked at.
Each vertex then searches along its own normal, near where the drag would put
it, for the strongest intensity edge of the expected polarity, and the patch
is regularised so neighbours move together. Clean-room: the principle (place
the boundary at the maximal gradient of the right sign along the normal,
within a band around the current estimate, with smoothness between
neighbours) is from Dale, Fischl & Sereno 1999 and Fischl & Dale 2000; none
of FreeSurfer's code or constants are used.

Invariants an edit keeps:

* **Vertices move along their normals only.** Tangential motion changes no
  boundary and only distorts the mesh that every derived surface (inflated,
  sphere, flat) is indexed against.
* **Topology never changes**, so vertex correspondence with every other
  surface of the hemisphere -- and between white and pial -- is preserved by
  construction.
* **Pial stays outside white** by at least ``min_thickness``: moving white
  out pushes pial ahead of it; moving pial in stops at white.
* **No triangle flips.** A step that would fold the mesh is scaled back.

Every :meth:`SurfaceEdit.update` recomputes from the positions at the start of
the drag, so a long drag cannot accumulate drift and a drag back to the start
restores the original exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter1d
from scipy.sparse.linalg import spsolve

from fastfuncstuff.surface.mesh import MeshTopology, face_normals, geodesic_ball, vertex_normals
from fastfuncstuff.surface.sampling import VolumeSampler


@dataclass(frozen=True)
class SnapParams:
    #: Brush radius, mm along the surface.
    radius: float = 4.0
    #: How far either side of the dragged position each vertex searches, mm.
    search: float = 1.5
    #: 0 follows the drag exactly; 1 goes all the way to the found edge.
    snap: float = 1.0
    #: Smoothing of the found boundary across neighbours, mm^2 (its sqrt is
    #: roughly the length it is smoothed over). The data term is weighted by
    #: each vertex's area, so the same value means the same thing on a 1 mm
    #: and a 0.4 mm mesh -- per-vertex weighting made a coarse mesh several
    #: times stiffer. Raise it when the anatomy is noisy and the snapped
    #: line comes out jagged.
    smooth: float = 0.2
    #: Sign of the intensity change crossing the boundary *outward*. -1 for T1
    #: (WM > GM > CSF, both boundaries darken outward); +1 for T2.
    edge_sign: int = -1
    #: Profile smoothing, in voxels, before differentiating.
    sigma_vox: float = 0.5
    #: Minimum pial-white separation along the normal, mm.
    min_thickness: float = 0.1
    #: Penalty, at the edge of the search band, for picking an edge far from
    #: the dragged position, relative to the strongest edge in the patch.
    distance_penalty: float = 0.5


@dataclass
class EditResult:
    ids: np.ndarray  # vertices of the edited surface that moved
    positions: np.ndarray  # their new positions
    partner_ids: np.ndarray  # vertices of the partner surface pushed along
    partner_positions: np.ndarray
    #: Per-vertex displacement along the normal, mm, for display/diagnostics.
    displacement: np.ndarray
    #: Per-vertex edge confidence in [0, 1].
    confidence: np.ndarray


def _brush_weight(dist: np.ndarray, radius: float) -> np.ndarray:
    """Smooth compact falloff: 1 at the centre, 0 with zero slope at the rim."""
    x = np.clip(dist / max(radius, 1e-9), 0.0, 1.0)
    return (1.0 - x * x) ** 2


class SurfaceEdit:
    """One drag on one surface, from press to release.

    ``role`` is ``"white"`` or ``"pial"``; ``partner`` is the other surface's
    positions (same vertex order), used to keep pial outside white. Positions
    are scanner-RAS mm, the same frame as ``sampler``.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        topo: MeshTopology,
        centre: int,
        sampler: VolumeSampler,
        params: SnapParams = SnapParams(),
        *,
        role: str = "white",
        partner: np.ndarray | None = None,
    ) -> None:
        if role not in ("white", "pial"):
            raise ValueError(f"role must be 'white' or 'pial', got {role!r}")
        self.params = params
        self.role = role
        self.sampler = sampler
        self.topo = topo
        self.centre = int(centre)
        vertices = np.asarray(vertices, np.float64)
        self.ids, dist = geodesic_ball(vertices, topo, self.centre, params.radius)
        self.weight = _brush_weight(dist, params.radius)
        self.start = vertices[self.ids].copy()
        # Normals of the *whole* mesh evaluated at the start, then frozen for
        # the drag: recomputing them as the patch moves would let the search
        # direction chase its own displacement.
        self.normals = vertex_normals(vertices, topo)[self.ids]
        self.partner_start = (
            None if partner is None else np.asarray(partner, np.float64)[self.ids].copy()
        )
        # Fold checks need only the faces touching the patch, over the
        # vertices those faces use -- reindexed locally so a mouse move never
        # copies the whole hemisphere.
        faces = topo.faces[topo.faces_of(self.ids)]
        used, local = np.unique(faces, return_inverse=True)
        self._fold_faces = local.reshape(faces.shape)
        self._fold_verts = vertices[used]
        self._fold_slot = np.searchsorted(used, self.ids)
        self._system = self._laplacian()
        # Barycentric vertex area (a third of each incident face), mm^2.
        area = np.zeros(used.size)
        face_area = 0.5 * np.linalg.norm(face_normals(self._fold_verts, self._fold_faces), axis=1)
        np.add.at(area, self._fold_faces.ravel(), np.repeat(face_area / 3.0, 3))
        self.area = area[self._fold_slot]
        n = int(round(2 * params.search / (0.1 * sampler.voxel_mm))) + 1
        self._offsets = np.linspace(-params.search, params.search, max(n, 3))

    def _laplacian(self) -> sp.csr_matrix:
        """Graph Laplacian over the patch's own edges (free at its rim).

        Free, not pinned: what is smoothed is the snap *correction*, and
        the rim fade comes from the brush weight. A rim pinned to zero made
        the whole patch a membrane that sagged short of an edge every vertex
        had found -- a constant correction must pass through unchanged.
        """
        edges = self.topo.edges
        n = self.ids.size
        lookup = np.full(self.topo.n_vertices, -1, np.int64)
        lookup[self.ids] = np.arange(n)
        a, b = lookup[edges[:, 0]], lookup[edges[:, 1]]
        both = (a >= 0) & (b >= 0)
        a, b = a[both], b[both]
        degree = np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
        rows = np.concatenate([a, b])
        cols = np.concatenate([b, a])
        off = sp.coo_matrix((-np.ones(rows.size), (rows, cols)), shape=(n, n))
        return (sp.diags(degree.astype(np.float64)) + off).tocsr()

    def _find_edges(self, along: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Best edge offset per vertex near ``along``, and its confidence."""
        p = self.params
        t = along[:, None] + self._offsets[None, :]  # (n, S)
        pts = self.start[:, None, :] + t[..., None] * self.normals[:, None, :]
        prof = self.sampler(pts)
        step = self._offsets[1] - self._offsets[0]
        sigma = p.sigma_vox * self.sampler.voxel_mm / step
        if sigma > 0:
            prof = gaussian_filter1d(prof, sigma, axis=1, mode="nearest")
        grad = np.gradient(prof, step, axis=1)
        strength = np.maximum(p.edge_sign * grad, 0.0)
        scale = float(strength.max())
        if scale <= 0:
            return along.copy(), np.zeros_like(along)
        rel = strength / scale
        score = rel - p.distance_penalty * (self._offsets[None, :] / p.search) ** 2
        best = np.argmax(score, axis=1)
        rows = np.arange(best.size)
        return t[rows, best], rel[rows, best]

    def update(self, drag: np.ndarray) -> EditResult:
        """Positions for a cumulative drag vector ``drag`` (mm) from the press."""
        p = self.params
        drag = np.asarray(drag, np.float64)
        along = self.weight * (self.normals @ drag)
        found, confidence = self._find_edges(along)
        # What is smooth in the anatomy is the boundary, so the found edge
        # offsets are what get regularised -- weighted by how sure each vertex
        # is of its edge -- and the result is faded in toward the rim (sqrt,
        # so most of the brush commits fully). Smoothing the *correction*
        # instead mixed the drag's own falloff into it and overshot.
        a = sp.diags(np.maximum(confidence, 1e-3) * self.area)
        edge = np.asarray(spsolve((a + p.smooth * self._system).tocsc(), a @ found), np.float64)
        d = along + p.snap * np.sqrt(self.weight) * (edge - along)

        gap = None
        if self.partner_start is not None:
            gap = np.einsum("ij,ij->i", self.partner_start - self.start, self.normals)
            if self.role == "pial":
                # Pial may not pass inward through white.
                d = np.maximum(d, -(gap - p.min_thickness))
        d = self._unfold(d)
        positions = self.start + d[:, None] * self.normals

        partner_ids = np.zeros(0, np.int64)
        partner_pos = np.zeros((0, 3))
        if gap is not None and self.partner_start is not None and self.role == "white":
            # White moving out pushes pial ahead of it.
            push = d + p.min_thickness - gap
            pushed = push > 0
            partner_ids = self.ids[pushed]
            partner_pos = self.partner_start[pushed] + push[pushed, None] * self.normals[pushed]
        return EditResult(self.ids, positions, partner_ids, partner_pos, d, confidence)

    def _unfold(self, d: np.ndarray) -> np.ndarray:
        """Scale the displacement back until no face in the patch flips."""
        before = face_normals(self._fold_verts, self._fold_faces)
        moved = self._fold_verts.copy()
        for _ in range(8):
            moved[self._fold_slot] = self.start + d[:, None] * self.normals
            after = face_normals(moved, self._fold_faces)
            if np.all(np.einsum("ij,ij->i", before, after) > 0):
                return d
            d = 0.5 * d
        return np.zeros_like(d)


__all__ = ["EditResult", "SnapParams", "SurfaceEdit"]
