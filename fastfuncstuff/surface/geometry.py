"""Mesh geometry: where a surface crosses a slice.

A slice outline is the set of triangle-plane crossings. Testing every triangle
on every crosshair move would cost ~10 ms per surface per pane, so faces are
indexed once per axis by their lowest coordinate: a plane at ``p`` can only cut
a face whose minimum lies in ``[p - longest_face, p]``, which is a contiguous
run of the sorted order found with two binary searches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _AxisIndex:
    order: np.ndarray  # faces sorted by their minimum along the axis
    lo: np.ndarray  # those minima, sorted
    reach: float  # the longest face extent along the axis


class SliceIndex:
    """Fast axis-aligned plane intersection for one mesh.

    ``vertices`` are in whatever frame the planes are given in -- for the
    viewer that is display-grid voxel indices, so a slice is ``axis = k``.
    Call :meth:`set_vertices` when the whole mesh moves or the frame changes,
    and :meth:`move` for a local edit: rebuilding costs ~200 ms per
    hemisphere, which a drag that redraws on every mouse move cannot afford,
    so moved faces are set aside and tested directly until there are enough
    of them that a rebuild is cheaper.
    """

    #: Dirty faces tested by brute force before the sorted index is rebuilt.
    REBUILD_AFTER = 20_000

    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        self.faces = np.ascontiguousarray(faces, dtype=np.int64)
        self.set_vertices(vertices)

    def _build(self, axis: int) -> _AxisIndex:
        coord = self.vertices[:, axis][self.faces]
        lo, hi = coord.min(axis=1), coord.max(axis=1)
        order = np.argsort(lo, kind="stable")
        return _AxisIndex(order=order, lo=lo[order], reach=float((hi - lo).max(initial=0.0)))

    def set_vertices(self, vertices: np.ndarray) -> None:
        self.vertices = np.array(vertices, dtype=np.float64)
        self._axes = [self._build(a) for a in range(3)]
        self._dirty = np.zeros(self.faces.shape[0], bool)
        self._dirty_ids = np.zeros(0, np.int64)

    def move(self, vertex_ids: np.ndarray, positions: np.ndarray, faces: np.ndarray) -> None:
        """Move some vertices; ``faces`` are the face ids touching them."""
        self.vertices[np.asarray(vertex_ids, np.int64)] = positions
        self._dirty[np.asarray(faces, np.int64)] = True
        self._dirty_ids = np.flatnonzero(self._dirty)
        if self._dirty_ids.size > self.REBUILD_AFTER:
            self.set_vertices(self.vertices)

    def _candidates(self, axis: int, position: float) -> np.ndarray:
        idx = self._axes[axis]
        start = np.searchsorted(idx.lo, position - idx.reach, side="left")
        stop = np.searchsorted(idx.lo, position, side="right")
        found = idx.order[start:stop]
        if self._dirty_ids.size:
            # The sorted keys of a dirty face are stale in both directions:
            # drop it from the run and test it from its current corners.
            found = np.concatenate([found[~self._dirty[found]], self._dirty_ids])
        return found

    def segments(self, axis: int, position: float) -> np.ndarray:
        """Line segments where the plane ``x[axis] == position`` cuts the mesh.

        Returns ``(N, 2, 3)`` endpoints in the vertex frame. A vertex lying
        exactly on the plane counts as being on its low side, so each cut
        triangle has exactly two crossing edges and no segment is emitted
        twice or degenerately.
        """
        return self.segments_with_faces(axis, position)[0]

    def segments_with_faces(self, axis: int, position: float) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`segments`, plus the face id each segment came from."""
        empty = (np.zeros((0, 2, 3)), np.zeros(0, np.int64))
        ids = self._candidates(axis, position)
        if ids.size == 0:
            return empty
        faces = self.faces[ids]
        d = self.vertices[faces, axis] - position  # (n, 3)
        above = d > 0
        n_above = above.sum(axis=1)
        cut = (n_above == 1) | (n_above == 2)
        ids, faces, d, above = ids[cut], faces[cut], d[cut], above[cut]
        if faces.size == 0:
            return empty
        # The lone corner is the one on the minority side; the two crossing
        # edges are the ones leaving it.
        lone_is_above = above.sum(axis=1) == 1
        lone = np.where(lone_is_above, np.argmax(above, axis=1), np.argmin(above, axis=1))
        rows = np.arange(faces.shape[0])
        a = faces[rows, lone]
        b = faces[rows, (lone + 1) % 3]
        c = faces[rows, (lone + 2) % 3]
        da, db, dc = d[rows, lone], d[rows, (lone + 1) % 3], d[rows, (lone + 2) % 3]
        va, vb, vc = self.vertices[a], self.vertices[b], self.vertices[c]
        tb = (da / (da - db))[:, None]
        tc = (da / (da - dc))[:, None]
        p0 = va + tb * (vb - va)
        p1 = va + tc * (vc - va)
        # An on-plane vertex that is its face's lone corner yields a point, not
        # a segment; the neighbouring faces already carry the trace through it.
        keep = np.any(p0 != p1, axis=1)
        self._last_edges = np.stack(
            [np.sort(np.stack([a, b], 1), axis=1), np.sort(np.stack([a, c], 1), axis=1)], axis=1
        )[keep]
        return np.stack([p0[keep], p1[keep]], axis=1), ids[keep]

    def segments_with_edges(
        self, axis: int, position: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """:meth:`segments_with_faces`, plus the mesh edge each endpoint lies on.

        ``(N, 2, 2)`` sorted vertex pairs. Two segments that end on the same
        mesh edge are neighbours along the outline -- exactly, with no float
        matching -- which is what lets a contour be walked.
        """
        seg, faces = self.segments_with_faces(axis, position)
        edges = getattr(self, "_last_edges", np.zeros((0, 2, 2), np.int64))
        if seg.shape[0] == 0:
            edges = np.zeros((0, 2, 2), np.int64)
        return seg, faces, edges


def contour_path(edges: np.ndarray, start: int, stop: int) -> np.ndarray | None:
    """Segments along an outline from segment ``start`` to ``stop``, inclusive.

    ``edges`` is :meth:`SliceIndex.segments_with_edges`' third output.
    Segments sharing an endpoint edge are adjacent; the shortest chain is
    returned (on a closed contour, the shorter way round), or ``None`` when
    the two are on different pieces of outline.
    """
    from collections import deque

    n = edges.shape[0]
    if not (0 <= start < n and 0 <= stop < n):
        return None
    by_edge: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        for e in edges[i]:
            by_edge.setdefault((int(e[0]), int(e[1])), []).append(i)
    previous = {start: -1}
    queue = deque([start])
    while queue:
        i = queue.popleft()
        if i == stop:
            break
        for e in edges[i]:
            for j in by_edge[(int(e[0]), int(e[1]))]:
                if j not in previous:
                    previous[j] = i
                    queue.append(j)
    if stop not in previous:
        return None
    path = [stop]
    while path[-1] != start:
        path.append(previous[path[-1]])
    return np.array(path[::-1], np.int64)


def apply_affine(affine: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """``(N, 3)`` points through a 4x4, in float64."""
    m = np.asarray(affine, dtype=np.float64)
    return np.asarray(xyz, dtype=np.float64) @ m[:3, :3].T + m[:3, 3]


__all__ = ["SliceIndex", "apply_affine", "contour_path"]
