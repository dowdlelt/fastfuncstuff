"""Correlation matrices: every node against every other node, as a picture.

A node is either an ROI or a bin of voxels that already move together, and the
difference between those two is the difference between a matrix you can read
row by row and one you can only read as texture.

* **With ROIs** the rows have names. This is the connectivity matrix everyone
  means -- an atlas, or a set of clusters a threshold just produced, averaged
  to one time course each and correlated. K is tens to hundreds, every cell is
  a pair of regions, and clicking one is a question with an answer.
* **Without ROIs** the honest fallback is not "the voxelwise matrix": 100k
  voxels is a 40 GB matrix and no screen for it. It is the same reduction the
  carpet makes -- order the voxels by how they move, average
  ordering-contiguous blocks -- and then correlate the blocks. The nodes are
  bins of voxels the ordering just declared alike, which is a real statement
  about the data and a different one from "parcels".

**Order is the whole game.** A connectivity matrix in atlas order is a picture
of the atlas's numbering scheme. Seriated by hierarchical clustering, the same
numbers become blocks on the diagonal, and the blocks are the networks. That is
why the default is ``hierarchical`` and not ``input``.

No Qt. The widget paints what this returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fastfuncstuff.viewer.rois import RoiSet, roi_means
from fastfuncstuff.viewer.series import (
    ProgressFn,
    bin_counts,
    bin_representatives,
    bin_rows,
    correlate,
    correlation_matrix,
    first_pc,
    prepare_rows,
)

#: Nodes in the no-ROI case. Chosen so the matrix stays bigger than any
#: sensible window in pixels -- reducing further would throw away structure
#: that would have been on screen -- while an optimal leaf ordering of it is
#: still under a second.
MAX_NODES = 400

#: Above this, seriation drops the optimal-leaf-ordering refinement. It is
#: O(K^3) and the dendrogram's own leaf order is already close; the refinement
#: is worth a second at 300 nodes and is not worth a minute at 1200.
MAX_OPTIMAL_LEAF = 300

ORDERINGS = ("hierarchical", "input", "pc1", "strength", "size")

ORDER_LABELS = {
    "hierarchical": "clustered",
    "input": "as listed",
    "pc1": "corr with PC1",
    "strength": "mean |r|",
    "size": "voxel count",
}


@dataclass(frozen=True)
class CorrMatrix:
    """A rendered correlation matrix and everything needed to read it."""

    matrix: np.ndarray  # (K, K) float32, in display order
    series: np.ndarray  # (K, T) float32, the node time courses, display order
    names: tuple[str, ...]
    colors: tuple[tuple[int, int, int], ...]
    sizes: tuple[int, ...]
    #: Label value in the source ROI set for each displayed row, or ``-1`` when
    #: the nodes are voxel bins. This is what lets a click on a cell name a
    #: region, move the crosshair to it, or seed from it.
    indices: tuple[int, ...]
    order: str
    #: Row positions where a hierarchical module ends, for the separators drawn
    #: on the diagonal. Empty unless the order is ``hierarchical``.
    blocks: tuple[int, ...] = ()
    #: Voxels behind the whole matrix, before any reduction.
    n_voxels: int = 0
    from_rois: bool = False
    #: ``(K, 3)`` voxel indices, one per displayed node: an ROI's centre, or
    #: for a voxel bin the voxel at the middle of the bin -- the same stand-in
    #: a binned carpet row uses. A bin's voxels are scattered by construction,
    #: so no centre of them is inside any of them; the middle one is.
    locations: np.ndarray | None = None

    def location_of(self, node: int) -> tuple[int, int, int] | None:
        if self.locations is None or not (0 <= node < self.locations.shape[0]):
            return None
        i, j, k = (int(v) for v in self.locations[node])
        return (i, j, k)

    @property
    def n_nodes(self) -> int:
        return int(self.matrix.shape[0])

    def name_of(self, row: int) -> str:
        return self.names[row] if 0 <= row < len(self.names) else "--"

    def status(self) -> str:
        what = "ROIs" if self.from_rois else "voxel bins"
        text = f"{self.n_nodes} {what} x {self.series.shape[1]} TR"
        if not self.from_rois and self.n_voxels:
            text += f"   {self.n_voxels:,} voxels"
        text += f"   {ORDER_LABELS.get(self.order, self.order)}"
        if self.blocks:
            # blocks are the boundaries between modules, so there is always one
            # more module than there are lines drawn.
            text += f"   {len(self.blocks) + 1} modules"
        return text


def _node_colors(n: int) -> tuple[tuple[int, int, int], ...]:
    """A position ramp for unnamed nodes.

    Voxel bins have no identity to colour, so the strip beside the matrix shows
    *where in the ordering* a row sits instead -- which is the only thing about
    an unnamed row worth showing.
    """
    out = []
    for i in range(n):
        t = i / max(n - 1, 1)
        out.append((int(40 + 180 * t), int(70 + 120 * (1 - abs(2 * t - 1))), int(220 - 170 * t)))
    return tuple(out)


def seriate(matrix: torch.Tensor) -> tuple[np.ndarray, tuple[int, ...]]:
    """Leaf order and module boundaries from average-linkage clustering.

    The distance is ``1 - r``, which is the one every connectivity paper uses
    and is a metric on the unit sphere of normalised time courses. Average
    linkage rather than Ward: Ward assumes Euclidean coordinates, and these are
    correlations.

    The cut is the classic dendrogram default -- 70% of the tallest merge --
    rather than a fixed module count, because the number of networks in a run
    is not something the viewer knows.
    """
    from scipy.cluster.hierarchy import fcluster, leaves_list, linkage, optimal_leaf_ordering
    from scipy.spatial.distance import squareform

    r = matrix.detach().cpu().numpy().astype(np.float64)
    k = r.shape[0]
    if k < 3:
        return np.arange(k), ()
    # squareform demands exact symmetry and an exact zero diagonal; float
    # arithmetic gives neither, and it raises rather than rounding.
    d = 1.0 - (r + r.T) / 2.0
    np.fill_diagonal(d, 0.0)
    condensed = squareform(np.clip(d, 0.0, 2.0), checks=False)
    tree = linkage(condensed, method="average")
    if k <= MAX_OPTIMAL_LEAF:
        tree = optimal_leaf_ordering(tree, condensed)
    order = np.asarray(leaves_list(tree), dtype=np.int64)

    heights = tree[:, 2]
    modules = fcluster(tree, t=0.7 * float(heights.max()), criterion="distance")[order]
    edges = tuple(int(i + 1) for i in range(len(order) - 1) if modules[i] != modules[i + 1])
    return order, edges


def _order_nodes(
    matrix: torch.Tensor,
    series: torch.Tensor,
    sizes: np.ndarray,
    order: str,
) -> tuple[np.ndarray, tuple[int, ...]]:
    if order == "input":
        return np.arange(matrix.shape[0]), ()
    if order == "hierarchical":
        return seriate(matrix)
    if order == "pc1":
        key = correlate(series, first_pc(series))
    elif order == "strength":
        # The diagonal is 1 for every node and would add a constant, so it is
        # left out rather than dominating a small matrix.
        off = matrix.abs().sum(1) - matrix.diagonal().abs()
        key = off / max(matrix.shape[0] - 1, 1)
    elif order == "size":
        key = torch.as_tensor(sizes.astype(np.float32), device=matrix.device)
    else:
        raise ValueError(f"unknown matrix ordering {order!r}; have {ORDERINGS}")
    return torch.argsort(key, descending=True).cpu().numpy(), ()


def build_matrix(
    data: np.ndarray,
    *,
    rois: RoiSet | None = None,
    mask: np.ndarray | None = None,
    order: str = "hierarchical",
    polort: int = 1,
    normalize: str = "z",
    max_nodes: int = MAX_NODES,
    device: torch.device | None = None,
    progress: ProgressFn | None = None,
) -> CorrMatrix:
    """Correlate a run's nodes against each other.

    With ``rois`` the nodes are its regions, in label order before the display
    ordering is applied. Without, they are bins of voxels grouped by how they
    move, which is the carpet's reduction reused rather than a second opinion
    about what a run's structure is.
    """
    if order not in ORDERINGS:
        raise ValueError(f"unknown matrix ordering {order!r}; have {ORDERINGS}")

    def step(fraction: float, message: str) -> None:
        if progress is not None:
            progress(fraction, message)

    if rois is not None and rois.shape != tuple(data.shape[:3]):
        raise ValueError(
            f"the ROIs are on a {rois.shape} grid and the run is {tuple(data.shape[:3])}; "
            "they have to be on the same grid to be averaged"
        )

    node_mask = mask
    if rois is not None:
        selected = rois.labels > 0
        node_mask = selected if mask is None else (selected & np.asarray(mask, dtype=bool))

    rows, flat_mask, _units, _limit = prepare_rows(
        data, mask=node_mask, polort=polort, normalize=normalize, device=device, progress=progress
    )
    n_voxels = int(rows.shape[0])

    step(0.6, "nodes")
    if rois is not None:
        labels_v = torch.as_tensor(
            rois.labels.reshape(-1)[flat_mask].astype(np.int64), device=rows.device
        )
        present = tuple(i for i in rois.indices if bool((labels_v == i).any()))
        if not present:
            raise ValueError("no ROI has a voxel inside the run's mask")
        series = roi_means(rows, labels_v, present)
        described = [rois.find(i) for i in present]
        names = tuple(r.name if r else f"#{i}" for r, i in zip(described, present, strict=True))
        colors = tuple(r.color if r else (200, 200, 200) for r in described)
        sizes = np.array(
            [int((labels_v == i).sum()) for i in present],
            dtype=np.int64,
        )
        indices = present
        locations = np.array(
            [r.center_ijk if r else (-1, -1, -1) for r in described], dtype=np.int32
        ).reshape(-1, 3)
    else:
        # Order first, then bin: the bins are only meaningful because the rows
        # inside one were already alike, which is the same argument the carpet
        # makes for averaging rather than striding.
        by_pc = torch.argsort(correlate(rows, first_pc(rows)), descending=True)
        rows = rows[by_pc]
        series = bin_rows(rows, max_nodes)
        flat_ids = np.flatnonzero(flat_mask)[by_pc.cpu().numpy()]
        locations = np.stack(
            np.unravel_index(bin_representatives(flat_ids, max_nodes), data.shape[:3]), 1
        ).astype(np.int32)
        counts = bin_counts(n_voxels, max_nodes)
        names = tuple(str(i + 1) for i in range(series.shape[0]))
        colors = _node_colors(int(series.shape[0]))
        sizes = np.array(counts, dtype=np.int64)
        indices = tuple(-1 for _ in names)

    step(0.8, "correlating")
    matrix = correlation_matrix(series)

    step(0.92, "ordering")
    index, blocks = _order_nodes(matrix, series, sizes, order)
    keep = torch.as_tensor(index, device=matrix.device)
    matrix = matrix[keep][:, keep]
    series = series[keep]

    step(1.0, "ready")
    return CorrMatrix(
        matrix=matrix.cpu().numpy().astype(np.float32),
        series=series.cpu().numpy().astype(np.float32),
        names=tuple(names[i] for i in index),
        colors=tuple(colors[i] for i in index),
        sizes=tuple(int(sizes[i]) for i in index),
        indices=tuple(int(indices[i]) for i in index),
        order=order,
        blocks=blocks,
        n_voxels=n_voxels,
        from_rois=rois is not None,
        locations=locations[index],
    )


__all__ = [
    "MAX_NODES",
    "ORDERINGS",
    "ORDER_LABELS",
    "CorrMatrix",
    "build_matrix",
    "seriate",
]
