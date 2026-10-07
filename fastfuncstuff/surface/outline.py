"""A shape drawn on a surface, filled: the vertices inside a traced outline.

The outline is what the hand traced -- picked vertices along the drag, on whatever
shape is drawn (inflated, sphere, white). Consecutive picks are joined by shortest
paths along mesh edges, measured on that same drawn shape so the boundary follows
what was seen, and the loop is closed back to its start. The boundary splits the
mesh: everything off the **largest** remaining piece is inside. That is the right
answer for a closed cortical surface, where the outside of any hand-drawn shape is
most of the hemisphere, and it fills every pocket of a loop that crossed itself.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra

from .smooth import mesh_edges

__all__ = ["fill_outline", "is_closed", "trace_path"]


def is_closed(faces: np.ndarray) -> bool:
    """Every edge shared by exactly two faces: a closed surface (FreeSurfer's are)."""
    f = np.asarray(faces, np.int64)
    e = np.sort(np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def _graph(positions: np.ndarray, faces: np.ndarray) -> sparse.csr_matrix:
    e = mesh_edges(faces)
    w = np.linalg.norm(positions[e[:, 0]] - positions[e[:, 1]], axis=1)
    n = len(positions)
    return sparse.csr_matrix(
        (np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), (n, n)
    )


def trace_path(positions: np.ndarray, faces: np.ndarray, picks, close: bool = True) -> np.ndarray:
    """The traced picks joined by shortest edge paths (and closed): vertex ids in order."""
    pos = np.asarray(positions, np.float64)
    seq = [int(v) for v in picks]
    seq = [v for i, v in enumerate(seq) if i == 0 or v != seq[i - 1]]
    if close and len(seq) > 2 and seq[0] != seq[-1]:
        seq.append(seq[0])
    if len(seq) < 2:
        return np.asarray(seq, np.int64)
    g = _graph(pos, faces)
    out = [seq[0]]
    for a, b in zip(seq[:-1], seq[1:], strict=True):
        if g[a, b] > 0:
            out.append(b)
            continue
        # Search only as far as the detour can plausibly go.
        reach = 3.0 * float(np.linalg.norm(pos[a] - pos[b])) + 5.0
        _, pred = dijkstra(g, indices=a, limit=reach, return_predecessors=True)
        if pred[b] < 0:  # unreachable within reach (e.g. across hemispheres): skip
            out.append(b)
            continue
        hop, back = b, []
        while hop != a:
            back.append(hop)
            hop = int(pred[hop])
        out.extend(reversed(back))
    return np.asarray(out, np.int64)


def fill_outline(positions: np.ndarray, faces: np.ndarray, picks) -> np.ndarray:
    """Vertex mask of the traced shape: its boundary and everything it encloses."""
    n = len(positions)
    boundary = np.zeros(n, bool)
    path = trace_path(positions, faces, picks)
    boundary[path] = True
    if path.size < 3:
        return boundary
    e = mesh_edges(faces)
    keep = ~boundary[e[:, 0]] & ~boundary[e[:, 1]]
    g = sparse.csr_matrix((np.ones(keep.sum()), (e[keep, 0], e[keep, 1])), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    comp = np.where(boundary, -1, comp)
    sizes = np.bincount(comp[comp >= 0])
    outside = int(np.argmax(sizes)) if sizes.size else -1
    return boundary | ((comp >= 0) & (comp != outside))
