"""Display order for connectivity matrices: seriation by hierarchical clustering.

A matrix in atlas order is a picture of the atlas's numbering scheme; seriated, the
same numbers become blocks on the diagonal, and the blocks are the networks. Shared
by ``ffs_fc -plot`` and the viewer's matrix window so the two draw the same picture.
"""

from __future__ import annotations

import numpy as np
import torch

#: Above this, seriation drops the optimal-leaf-ordering refinement. It is
#: O(K^3) and the dendrogram's own leaf order is already close; the refinement
#: is worth a second at 300 nodes and is not worth a minute at 1200.
MAX_OPTIMAL_LEAF = 300


def seriate(matrix: torch.Tensor | np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Leaf order and module boundaries from average-linkage clustering.

    The distance is ``1 - r``, which is the one every connectivity paper uses
    and is a metric on the unit sphere of normalised time courses. Average
    linkage rather than Ward: Ward assumes Euclidean coordinates, and these are
    correlations.

    The cut is the classic dendrogram default -- 70% of the tallest merge --
    rather than a fixed module count, because the number of networks in a run
    is not something we know.
    """
    from scipy.cluster.hierarchy import fcluster, leaves_list, linkage, optimal_leaf_ordering
    from scipy.spatial.distance import squareform

    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    r = np.asarray(matrix, dtype=np.float64)
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
