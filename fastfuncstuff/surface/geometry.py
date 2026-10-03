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
    Call :meth:`set_vertices` when the vertices move or the frame changes.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        self.faces = np.ascontiguousarray(faces, dtype=np.int64)
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self._axes: list[_AxisIndex] = [self._build(a) for a in range(3)]

    def _build(self, axis: int) -> _AxisIndex:
        coord = self.vertices[:, axis][self.faces]
        lo, hi = coord.min(axis=1), coord.max(axis=1)
        order = np.argsort(lo, kind="stable")
        return _AxisIndex(order=order, lo=lo[order], reach=float((hi - lo).max(initial=0.0)))

    def set_vertices(self, vertices: np.ndarray) -> None:
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self._axes = [self._build(a) for a in range(3)]

    def segments(self, axis: int, position: float) -> np.ndarray:
        """Line segments where the plane ``x[axis] == position`` cuts the mesh.

        Returns ``(N, 2, 3)`` endpoints in the vertex frame. A vertex lying
        exactly on the plane counts as being on its low side, so each cut
        triangle has exactly two crossing edges and no segment is emitted
        twice or degenerately.
        """
        idx = self._axes[axis]
        start = np.searchsorted(idx.lo, position - idx.reach, side="left")
        stop = np.searchsorted(idx.lo, position, side="right")
        faces = self.faces[idx.order[start:stop]]
        if faces.size == 0:
            return np.zeros((0, 2, 3))
        d = self.vertices[faces, axis] - position  # (n, 3)
        above = d > 0
        n_above = above.sum(axis=1)
        cut = (n_above == 1) | (n_above == 2)
        faces, d, above = faces[cut], d[cut], above[cut]
        if faces.size == 0:
            return np.zeros((0, 2, 3))
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
        return np.stack([p0[keep], p1[keep]], axis=1)


def apply_affine(affine: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """``(N, 3)`` points through a 4x4, in float64."""
    m = np.asarray(affine, dtype=np.float64)
    return np.asarray(xyz, dtype=np.float64) @ m[:3, :3].T + m[:3, 3]


__all__ = ["SliceIndex", "apply_affine"]
