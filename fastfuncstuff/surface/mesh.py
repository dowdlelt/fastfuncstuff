"""Mesh topology and differential quantities shared by sampling and editing.

Built once per hemisphere from the faces, which never change: an edit moves
vertices, so the adjacency, the vertex-to-face map and the edge list stay
valid for the life of the mesh and only normals need recomputing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra


@dataclass
class MeshTopology:
    faces: np.ndarray  # (F, 3) int64
    edges: np.ndarray  # (E, 2) unique undirected, i < j
    #: Vertex -> incident faces, CSR: faces of v are
    #: ``vf_index[vf_ptr[v]:vf_ptr[v + 1]]``.
    vf_ptr: np.ndarray
    vf_index: np.ndarray
    n_vertices: int

    @classmethod
    def from_faces(cls, faces: np.ndarray, n_vertices: int | None = None) -> MeshTopology:
        faces = np.ascontiguousarray(faces, dtype=np.int64)
        nv = int(faces.max()) + 1 if n_vertices is None else int(n_vertices)
        e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        e.sort(axis=1)
        # 1-D keys: np.unique(axis=0) on ~850k rows is ~10x slower.
        key = np.unique(e[:, 0] * nv + e[:, 1])
        edges = np.stack([key // nv, key % nv], axis=1)
        owner = np.repeat(np.arange(faces.shape[0]), 3)
        verts = faces.ravel()
        order = np.argsort(verts, kind="stable")
        counts = np.bincount(verts, minlength=nv)
        ptr = np.zeros(nv + 1, np.int64)
        np.cumsum(counts, out=ptr[1:])
        return cls(faces, edges, ptr, owner[order], nv)

    def faces_of(self, vertices: np.ndarray) -> np.ndarray:
        """Unique faces touching any of ``vertices``."""
        vertices = np.asarray(vertices, dtype=np.int64)
        if vertices.size == 0:
            return np.zeros(0, np.int64)
        starts, stops = self.vf_ptr[vertices], self.vf_ptr[vertices + 1]
        idx = np.concatenate([self.vf_index[a:b] for a, b in zip(starts, stops, strict=True)])
        return np.unique(idx)

    def adjacency(self) -> sp.csr_matrix:
        """Vertex-to-vertex adjacency, 0/1 CSR; built once, the connectivity never changes."""
        found = getattr(self, "_adjacency", None)
        if found is None:
            e, n = self.edges, self.n_vertices
            found = sp.csr_matrix(
                (np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                shape=(n, n),
            )
            found.data[:] = 1.0
            self._adjacency = found
        return found

    def edge_graph(self, vertices: np.ndarray) -> sp.csr_matrix:
        """Symmetric edge-length graph for geodesic distances.

        The sparsity pattern is built once and only the lengths are refreshed:
        vertices move under editing, the connectivity never does, and the
        COO-to-CSR conversion is most of the cost.
        """
        i, j = self.edges[:, 0], self.edges[:, 1]
        w = np.linalg.norm(vertices[i] - vertices[j], axis=1)
        pattern = getattr(self, "_graph_pattern", None)
        if pattern is None:
            m = self.edges.shape[0]
            # Carry each entry's edge number (+1, so none is an explicit zero)
            # through the conversion to learn where it lands in CSR order.
            tag = np.concatenate([np.arange(1, m + 1), np.arange(1, m + 1)]).astype(np.float64)
            g = sp.coo_matrix(
                (tag, (np.concatenate([i, j]), np.concatenate([j, i]))),
                shape=(self.n_vertices, self.n_vertices),
            ).tocsr()
            pattern = (g, g.data.astype(np.int64) - 1)
            self._graph_pattern = pattern
        g, slot = pattern
        out = g.copy()
        out.data = w[slot]
        return out


def _rings(topo: MeshTopology, ids: np.ndarray, rings: int):
    """Rows ``ids`` of the ``rings``-ring neighbourhood (self excluded), 0/1 CSR."""
    adj = topo.adjacency()
    ids = np.asarray(ids, np.int64)
    reach = adj[ids]
    for _ in range(rings - 1):
        reach = reach + reach @ adj
    reach = reach.tocsr()
    reach.data[:] = 1.0
    reach[np.arange(ids.size), ids] = 0.0
    reach.eliminate_zeros()
    return reach


def local_height(
    vertices: np.ndarray, topo: MeshTopology, ids: np.ndarray, rings: int = 2
) -> np.ndarray:
    """How far each of ``ids`` sticks out of the surface around it, along its normal, mm.

    The height above the ``rings``-ring mean, *minus* the neighbours' own
    such heights. The second difference matters: a smoothly curved crown
    sits above its ring mean too (by ~r^2/2R), and by the first difference
    alone every press would round crowns off a little more. Under constant
    (or linearly varying) curvature the two cancel; a spike, a pit or a
    ledge one or two vertices wide does not.
    """
    v = np.asarray(vertices, np.float64)
    ids = np.asarray(ids, np.int64)
    reach = _rings(topo, ids, rings)
    # The first difference is needed on the neighbours as well.
    around = np.unique(np.r_[ids, reach.indices])
    ring = _rings(topo, around, rings)
    counts = np.maximum(np.asarray(ring.sum(axis=1)).ravel(), 1)
    mean = np.asarray(ring @ v) / counts[:, None]
    normals = vertex_normals(v, topo, around)
    first = np.einsum("ij,ij->i", v[around] - mean, normals)
    lookup = np.zeros(topo.n_vertices)
    lookup[around] = first
    own = lookup[ids]
    neighbours = np.asarray(reach @ lookup).ravel() / np.maximum(
        np.asarray(reach.sum(axis=1)).ravel(), 1
    )
    return own - neighbours


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unnormalised face normals (length = 2 x area), FreeSurfer winding = outward."""
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return np.cross(v1 - v0, v2 - v0)


def vertex_normals(
    vertices: np.ndarray, topo: MeshTopology, ids: np.ndarray | None = None
) -> np.ndarray:
    """Area-weighted unit vertex normals: of every vertex, or of ``ids`` only, in that order.

    With ``ids``, only the faces touching them are visited -- a vertex normal
    depends on nothing else, so the result is the same, and an edit of a few
    hundred vertices no longer pays for the whole hemisphere's ~230k faces.
    """
    faces = topo.faces if ids is None else topo.faces[topo.faces_of(ids)]
    fn = face_normals(vertices, faces)
    flat = faces.ravel()
    # bincount, not np.add.at: the same sums, several times faster.
    n = np.stack(
        [
            np.bincount(flat, weights=np.repeat(fn[:, c], 3), minlength=topo.n_vertices)
            for c in range(3)
        ],
        axis=1,
    )
    if ids is not None:
        n = n[np.asarray(ids, np.int64)]
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(norm, 1e-12)


def vertex_areas(
    vertices: np.ndarray, faces: np.ndarray, n_vertices: int | None = None
) -> np.ndarray:
    """Barycentric area per vertex (a third of each incident face), mm^2."""
    n = int(faces.max()) + 1 if n_vertices is None else int(n_vertices)
    face_area = 0.5 * np.linalg.norm(face_normals(np.asarray(vertices, np.float64), faces), axis=1)
    out = np.zeros(n)
    np.add.at(out, np.asarray(faces).ravel(), np.repeat(face_area / 3.0, 3))
    return out


def geodesic_ball(
    vertices: np.ndarray, topo: MeshTopology, centre: int, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Vertices within ``radius`` mm of ``centre`` *along the mesh*, and their distances.

    Along the mesh rather than through space: a Euclidean ball centred on one
    bank of a sulcus reaches the opposite bank, and an edit that drags the
    neighbouring gyrus along with it is the one thing a brush must never do.
    Edge-path distance overestimates true geodesic distance by a few percent,
    which only makes the brush marginally smaller.
    """
    graph = topo.edge_graph(np.asarray(vertices, np.float64))
    d = dijkstra(graph, indices=int(centre), limit=float(radius))
    inside = np.flatnonzero(np.isfinite(d))
    return inside, d[inside]


__all__ = [
    "MeshTopology",
    "face_normals",
    "geodesic_ball",
    "vertex_areas",
    "vertex_normals",
]
