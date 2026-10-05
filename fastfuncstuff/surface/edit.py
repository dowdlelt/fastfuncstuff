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
    #: Keep only edges that cross the intensity this boundary should sit at
    #: (read off the surfaces themselves, see ``_tissue_levels``). Off, the
    #: strongest edge of the right sign wins: for a boundary the surfaces'
    #: own tissue estimate gets wrong -- pial lying in dura reads "CSF" from
    #: dura, and the gate then rejects the GM/dura edge the eye can see.
    gate: bool = True


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
    #: Why the result is not simply the drag -- for telling the person
    #: dragging, while they drag. Vertices the fold guard damped; pial
    #: vertices held at white (``min_thickness``) when dragged through it;
    #: and, with snap on, how far the image's edge moved the brush's core
    #: from where the hand put it (mm along the normal, + outward).
    fold_damped: int = 0
    held: int = 0
    snap_offset: float = 0.0


def explain(res: EditResult) -> str:
    """One line on what held an edit back, or ``""`` when it did what was asked."""
    parts = []
    if res.snap_offset and abs(res.snap_offset) >= 0.25:
        way = "out" if res.snap_offset > 0 else "in"
        parts.append(
            f"snap moved it {abs(res.snap_offset):.1f} mm {way} to the image edge (m: hand)"
        )
    if res.held:
        parts.append(f"pial held at white on {res.held} vertices -- move white first")
    if res.fold_damped:
        parts.append(f"damped on {res.fold_damped} vertices so the mesh does not fold")
    return "; ".join(parts)


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

    #: Neighbour-averaging passes over the vertex normals a drag moves along.
    NORMAL_SMOOTHING = 3

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
        self.centre = int(centre)
        vertices = np.asarray(vertices, np.float64)
        ids, dist = geodesic_ball(vertices, topo, self.centre, params.radius)
        self._setup(vertices, topo, ids, dist, sampler, params, role, partner)

    def _setup(self, vertices, topo, ids, dist, sampler, params, role, partner) -> None:
        """Everything about the patch except how it was chosen."""
        if role not in ("white", "pial"):
            raise ValueError(f"role must be 'white' or 'pial', got {role!r}")
        self.params = params
        self.role = role
        self.sampler = sampler
        self.topo = topo
        self.ids = ids
        self.weight = _brush_weight(dist, params.radius)
        self.start = vertices[self.ids].copy()
        # Normals of the *whole* mesh evaluated at the start, then frozen for
        # the drag: recomputing them as the patch moves would let the search
        # direction chase its own displacement.
        self.normals = vertex_normals(vertices, topo)[self.ids]
        self.partner_start = (
            None if partner is None else np.asarray(partner, np.float64)[self.ids].copy()
        )
        if self.partner_start is not None:
            # "Outward" must mean white -> pial. FreeSurfer winds every surface
            # outward, but a mesh from elsewhere may not, and inward normals
            # silently invert every push and clamp below.
            outward = (
                self.partner_start - self.start
                if role == "white"
                else self.start - self.partner_start
            )
            if np.einsum("ij,ij->i", outward, self.normals).mean() < 0:
                self.normals = -self.normals
        # Fold checks need only the faces touching the patch, over the
        # vertices those faces use -- reindexed locally so a mouse move never
        # copies the whole hemisphere.
        faces = topo.faces[topo.faces_of(self.ids)]
        used, local = np.unique(faces, return_inverse=True)
        self._fold_faces = local.reshape(faces.shape)
        self._fold_verts = vertices[used]
        self._fold_slot = np.searchsorted(used, self.ids)
        self._fold_before = face_normals(self._fold_verts, self._fold_faces)
        self.fold_damped = 0
        self._system = self._laplacian()
        self._degree = np.maximum(self._system.diagonal(), 1.0)
        self._adjacency = (sp.diags(self._system.diagonal()) - self._system).tocsr()
        # The direction each vertex moves: its normal, averaged over a couple
        # of rings. Raw vertex normals jitter from vertex to vertex on a fine
        # mesh, and a drag projected onto them pushed neighbours by different
        # amounts -- on a real subject a third of 2 mm white drags folded the
        # mesh; with these, a quarter (and the fold guard then damps less).
        n = self.normals
        for _ in range(self.NORMAL_SMOOTHING):
            n = n + self._adjacency @ n
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        self.normals = n
        # Barycentric vertex area (a third of each incident face), mm^2.
        area = np.zeros(used.size)
        face_area = 0.5 * np.linalg.norm(face_normals(self._fold_verts, self._fold_faces), axis=1)
        np.add.at(area, self._fold_faces.ravel(), np.repeat(face_area / 3.0, 3))
        self.area = area[self._fold_slot]
        n = int(round(2 * params.search / (0.1 * sampler.voxel_mm))) + 1
        self._offsets = np.linspace(-params.search, params.search, max(n, 3))
        self.levels = self._tissue_levels() if params.gate else None

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

    def _tissue_levels(self) -> tuple[float, float] | None:
        """(crossing level, contrast) for this boundary, from the brush itself.

        Both boundaries darken outward on T1, so edge strength alone cannot
        tell them apart: pial dragged inward found the WM/GM edge and snapped
        onto white. The border sits at an intensity between the two tissues it
        separates (the principle FreeSurfer's placement rests on), and those
        tissues can be read off the surfaces already there: GM at mid-depth,
        WM 1 mm inside white, CSF as the darkest point 0.5-2 mm beyond pial
        (the darkest, because a narrow sulcus puts the next gyrus's GM there
        too). Medians over the brush keep one bad vertex from setting them.
        """
        if self.partner_start is None:
            return None
        white, pial = (
            (self.start, self.partner_start)
            if self.role == "white"
            else (self.partner_start, self.start)
        )
        n = self.normals
        gm = float(np.median(self.sampler(0.5 * (white + pial))))
        wm = float(np.median(self.sampler(white - 1.0 * n)))
        beyond = pial[:, None, :] + np.linspace(0.5, 2.0, 7)[None, :, None] * n[:, None, :]
        csf = float(np.median(self.sampler(beyond).min(axis=1)))
        inner, outer = (wm, gm) if self.role == "white" else (gm, csf)
        contrast = abs(inner - outer)
        if contrast <= 1e-6:
            return None
        return 0.5 * (inner + outer), contrast

    def _find_edges(
        self, along: np.ndarray, limit: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Best edge offset per vertex near ``along``, and its confidence.

        ``limit`` narrows each vertex's search (mm either side) below the
        brush's ``search``.
        """
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
        if self.levels is not None:
            # An edge at the wrong intensity is the other boundary; sigma of a
            # quarter of the contrast puts a GM-level crossing at exp(-2).
            level, contrast = self.levels
            rel = rel * np.exp(-0.5 * ((prof - level) / (0.25 * contrast)) ** 2)
        score = rel - p.distance_penalty * (self._offsets[None, :] / p.search) ** 2
        if limit is not None:
            score = np.where(np.abs(self._offsets)[None, :] > limit[:, None], -np.inf, score)
        best = np.argmax(score, axis=1)
        rows = np.arange(best.size)
        return t[rows, best], rel[rows, best]

    def update(self, drag: np.ndarray) -> EditResult:
        """Positions for a cumulative drag vector ``drag`` (mm) from the press."""
        drag = np.asarray(drag, np.float64)
        return self._finish(self.weight * (self.normals @ drag))

    def _finish(self, along: np.ndarray, limit: np.ndarray | None = None) -> EditResult:
        """From a rough along-normal displacement to the placed, guarded result."""
        p = self.params
        found, confidence = self._find_edges(along, limit)
        # What is smooth in the anatomy is the boundary, so the found edge
        # offsets are what get regularised -- weighted by how sure each vertex
        # is of its edge -- and the result is faded in toward the rim (sqrt,
        # so most of the brush commits fully). Smoothing the *correction*
        # instead mixed the drag's own falloff into it and overshot.
        a = sp.diags(np.maximum(confidence, 1e-3) * self.area)
        edge = np.asarray(spsolve((a + p.smooth * self._system).tocsc(), a @ found), np.float64)
        # Smoothing may carry a confident neighbour's edge further than this
        # vertex searched; it has no evidence there.
        reach = p.search if limit is None else limit
        edge = np.clip(edge, along - reach, along + reach)
        d = along + p.snap * np.sqrt(self.weight) * (edge - along)
        core = self.weight > 0.5
        snap_offset = float(np.median((d - along)[core])) if p.snap > 0 and core.any() else 0.0

        gap = None
        held = 0
        if self.partner_start is not None:
            gap = np.einsum("ij,ij->i", self.partner_start - self.start, self.normals)
            if self.role == "pial":
                # Pial may not pass inward through white. ``gap`` is white
                # relative to pial along the normal, so negative: pial may move
                # in by at most -(gap) - min_thickness.
                floor = gap + p.min_thickness
                held = int(np.count_nonzero((d < floor) & (self.weight > 0.1)))
                d = np.maximum(d, floor)
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
        return EditResult(
            self.ids,
            positions,
            partner_ids,
            partner_pos,
            d,
            confidence,
            fold_damped=self.fold_damped,
            held=held,
            snap_offset=snap_offset,
        )

    def _flipped(self, d: np.ndarray) -> np.ndarray:
        """Faces (of those touching the patch) that ``d`` turns over."""
        moved = self._fold_verts.copy()
        moved[self._fold_slot] = self.start + d[:, None] * self.normals
        after = face_normals(moved, self._fold_faces)
        return np.einsum("ij,ij->i", self._fold_before, after) <= 0

    def _unfold(self, d: np.ndarray) -> np.ndarray:
        """Damp the displacement where it folds the mesh, and only there.

        Halving the whole patch whenever any face flipped cost the median
        2 mm pial drag half its movement on a real subject (the guard fired on
        more than half of them): one tight fundus at the brush's edge held back
        the vertex under the cursor. The damping now starts at the vertices of
        the flipped faces and fades over their neighbours, so the rest of the
        patch keeps what it was asked for. The global halving stays as the
        last resort, and zero after that, so a flip can never be committed.
        """
        self.fold_damped = 0
        bad = self._flipped(d)
        if not bad.any():
            return d
        keep = np.ones_like(d)
        for _ in range(12):
            hit = np.zeros(self._fold_verts.shape[0], bool)
            hit[self._fold_faces[bad].ravel()] = True
            hit = hit[self._fold_slot]
            keep[hit] *= 0.5
            # Spread the damping one ring outward at half strength, so the
            # damped spot does not become a step in the surface.
            loss = 1.0 - keep
            loss = np.maximum(loss, 0.5 * (self._adjacency @ loss) / self._degree)
            keep = 1.0 - loss
            bad = self._flipped(d * keep)
            if not bad.any():
                self.fold_damped = int(np.count_nonzero(keep < 0.999))
                return d * keep
        d = d * keep
        for _ in range(8):
            d = 0.5 * d
            if not self._flipped(d).any():
                self.fold_damped = d.size
                return d
        self.fold_damped = d.size
        return np.zeros_like(d)


class StrokeEdit(SurfaceEdit):
    """A stretch of outline redrawn by hand, with the surface around it following.

    ``seeds`` are the vertices of the faces the slice cut between the two
    ends of the stroke -- the stretch that was redrawn -- and ``stroke`` the
    drawn line, ``(M, 3)`` scanner mm. Each seed moves along its normal to the
    stroke's nearest point; everything within ``params.radius`` mm *along the
    surface* of any seed follows by harmonic interpolation, pinned to the
    seeds and to zero just past the rim, so the change carries into the
    slices above and below without a kink. With ``snap`` on, the result is
    then placed on the image's edge as a grab is -- but the seeds search only
    half as far: the hand drew them, and is trusted over the extrapolation.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        topo: MeshTopology,
        seeds: np.ndarray,
        stroke: np.ndarray,
        sampler: VolumeSampler,
        params: SnapParams = SnapParams(),
        *,
        role: str = "white",
        partner: np.ndarray | None = None,
    ) -> None:
        from scipy.sparse.csgraph import dijkstra

        vertices = np.asarray(vertices, np.float64)
        seeds = np.unique(np.asarray(seeds, np.int64))
        if seeds.size == 0:
            raise ValueError("a stroke needs at least one vertex to move")
        stroke = np.asarray(stroke, np.float64)
        if stroke.ndim != 2 or stroke.shape[0] < 2:
            raise ValueError("a stroke needs at least two points")
        graph = topo.edge_graph(vertices)
        dist = dijkstra(graph, indices=seeds, limit=float(params.radius), min_only=True)
        ids = np.flatnonzero(np.isfinite(dist))
        self.centre = int(seeds[0])
        self._setup(vertices, topo, ids, dist[ids], sampler, params, role, partner)
        self.seed = np.isin(self.ids, seeds)
        targets = closest_on_polyline(self.start[self.seed], stroke)
        self.seed_shift = np.einsum(
            "ij,ij->i", targets - self.start[self.seed], self.normals[self.seed]
        )
        self.along = self._interpolate()

    def _interpolate(self) -> np.ndarray:
        """Harmonic fill: seeds fixed to their shift, zero just past the rim."""
        edges = self.topo.edges
        n = self.ids.size
        lookup = np.full(self.topo.n_vertices, -1, np.int64)
        lookup[self.ids] = np.arange(n)
        a, b = lookup[edges[:, 0]], lookup[edges[:, 1]]
        touching = (a >= 0) | (b >= 0)
        a, b = a[touching], b[touching]
        # Edges leaving the patch count in the degree with nothing on the far
        # side: that is the zero the rim is pinned to.
        degree = np.bincount(a[a >= 0], minlength=n) + np.bincount(b[b >= 0], minlength=n)
        both = (a >= 0) & (b >= 0)
        rows = np.concatenate([a[both], b[both]])
        cols = np.concatenate([b[both], a[both]])
        lap = sp.diags(degree.astype(np.float64)) + sp.coo_matrix(
            (-np.ones(rows.size), (rows, cols)), shape=(n, n)
        )
        pin = np.zeros(n)
        pin[self.seed] = 1e6
        rhs = np.zeros(n)
        rhs[self.seed] = 1e6 * self.seed_shift
        return np.asarray(spsolve((lap + sp.diags(pin)).tocsc(), rhs), np.float64)

    def result(self) -> EditResult:
        """The redrawn surface. Computed from the start, like every update."""
        p = self.params
        if p.snap <= 0:
            limit = np.zeros(self.ids.size)
        else:
            limit = np.where(self.seed, 0.5 * p.search, p.search)
        return self._finish(self.along, limit)


def closest_on_polyline(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    """For each point, the nearest point on the polyline ``line`` (both in mm)."""
    a, b = line[:-1], line[1:]
    ab = b - a
    t = np.einsum("pmk,mk->pm", points[:, None, :] - a[None], ab) / np.maximum(
        np.einsum("mk,mk->m", ab, ab), 1e-12
    )
    t = np.clip(t, 0.0, 1.0)
    near = a[None] + t[..., None] * ab[None]
    k = np.argmin(np.linalg.norm(near - points[:, None, :], axis=2), axis=1)
    return near[np.arange(points.shape[0]), k]


__all__ = [
    "EditResult",
    "SnapParams",
    "StrokeEdit",
    "SurfaceEdit",
    "closest_on_polyline",
    "explain",
]
