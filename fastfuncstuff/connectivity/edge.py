"""Edge time series: the moment-to-moment co-fluctuation behind every correlation.

Z-score each ROI series and take element-wise products: ``e_ij(t) = z_i(t) z_j(t)``
(Faskowitz et al. 2020; Esfahlani et al. 2020). The time average of ``e_ij`` is the
Pearson r of ``i`` and ``j`` -- exactly, because the z-scores here use the population
standard deviation (MATLAB's ``zscore`` uses ``T - 1``, which scales every edge by
``(T - 1) / T``) -- so an edge series is that correlation unwrapped in time.

The root-sum-square over edges, ``RSS(t)``, is the co-fluctuation amplitude whose
high-amplitude frames carry most of the static FC. It needs no edge matrix:
``sum_{i<j} (z_i z_j)^2 = ((sum_i z_i^2)^2 - sum_i z_i^4) / 2``, an O(N T) reduction
however many ROIs there are. The edge series themselves are N(N-1)/2 rows, so they
are only written on request, streamed straight to disk.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from fastfuncstuff.connectivity.matrix import unit_rows


def zscore_rows(x: Tensor) -> Tensor:
    """Population z-scores per row (constant rows -> 0), float32 on ``x``'s device."""
    xn, _ = unit_rows(x, dtype=torch.float32)
    return xn * math.sqrt(x.shape[1])


def rss(z: Tensor) -> np.ndarray:
    """``(N, T)`` z-scores -> ``(T,)`` root-sum-square of all N(N-1)/2 edge products."""
    zz = z.to(torch.float64) ** 2
    s2 = zz.sum(0)
    s4 = (zz * zz).sum(0)
    return torch.sqrt(((s2 * s2 - s4) / 2.0).clamp_min(0.0)).cpu().numpy()


def edge_index(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Row/column of each edge in the order edge series are written (``triu_indices(n, 1)``)."""
    return np.triu_indices(n, k=1)


def write_edge_timeseries(z: Tensor, path: str, max_bytes: int = 1 << 28) -> tuple[int, int]:
    """Stream the ``(E, T)`` float32 edge series of ``z`` into a ``.npy`` file.

    Edges are in :func:`edge_index` order. Rows are computed on ``z``'s device a
    block of ROIs at a time (about ``max_bytes`` of output per block) and written
    through a memory map, so neither the device nor the host ever holds them all.
    Returns the array shape.
    """
    n, n_t = z.shape
    n_edges = n * (n - 1) // 2
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n_edges, n_t))
    pos, i = 0, 0
    per_row = max(1, n_t * 4)
    while i < n - 1:
        rows, j = 0, i
        while j < n - 1 and (rows == 0 or (rows + n - 1 - j) * per_row <= max_bytes):
            rows += n - 1 - j
            j += 1
        block = [z[k : k + 1] * z[k + 1 :] for k in range(i, j)]
        chunk = torch.cat(block, dim=0).cpu().numpy()
        out[pos : pos + chunk.shape[0]] = chunk
        pos += chunk.shape[0]
        i = j
    out.flush()
    del out
    return n_edges, n_t
