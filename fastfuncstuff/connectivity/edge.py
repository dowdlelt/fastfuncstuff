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

**Edge communities** (Faskowitz et al. 2020) are k-means clusters of edges. The
reference code (``edge-centric_demo``, ``example_code.m``) clusters the edge series
themselves, ``kmeans(ets', k, 'distance', 'sqeuclidean')``: each edge is a point in T
dimensions. That is the default here (``on="ets"``). Its z-scores use T - 1, which
scales every point alike and so leaves the partition unchanged.

**Edge functional connectivity (eFC)** is the similarity of every pair of edge series.
With each edge series scaled to unit length, ``ê_k``,

    eFC = Ê Ê^T                (E x E, E = N(N-1)/2)

which is 4.2e9 entries (17 GB float32) for 360 ROIs and ~1 TB for 1000. It is never
needed whole. ``Ê`` is ``E x T``, so eFC has rank at most T, and what is computed
*from* it lives in T dimensions. Clustering eFC's rows instead of the series
(``on="efc"``, the alternative the reference mentions) needs only distances between
rows:

    ||eFC_k - eFC_l||^2 = (ê_k - ê_l)^T G (ê_k - ê_l),      G = Ê^T Ê   (T x T)

With ``G = V Λ V^T``, the rows of ``Ê V Λ^{1/2}`` (E x rank) have exactly those
distances, so k-means on them *is* k-means on eFC. The correlation distance (each
eFC row centred over its E entries, then unit length; MATLAB's ``'correlation'``) is
the same with ``G_c = G - s s^T / E``, ``s = Σ_k ê_k``, and the embedded rows scaled
to unit length. Memory is ``E x T``, the size of the edge series, not ``E x E``.
"""

from __future__ import annotations

import math
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from fastfuncstuff.connectivity.matrix import unit_rows
from fastfuncstuff.utils import full_fp32_matmul


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


def edge_blocks(z: Tensor, max_bytes: int = 1 << 28) -> Iterator[tuple[int, Tensor]]:
    """Yield ``(first_edge, (b, T) block)`` of edge series in :func:`edge_index` order.

    A block is the products of a run of ROIs with every later ROI, about ``max_bytes``
    of float32, on ``z``'s device. The one place edge series are formed.
    """
    n, n_t = z.shape
    pos, i = 0, 0
    per_row = max(1, n_t * 4)
    while i < n - 1:
        rows, j = 0, i
        while j < n - 1 and (rows == 0 or (rows + n - 1 - j) * per_row <= max_bytes):
            rows += n - 1 - j
            j += 1
        block = torch.cat([z[k : k + 1] * z[k + 1 :] for k in range(i, j)], dim=0)
        yield pos, block
        pos += block.shape[0]
        i = j


def write_edge_timeseries(z: Tensor, path: str, max_bytes: int = 1 << 28) -> tuple[int, int]:
    """Stream the ``(E, T)`` float32 edge series of ``z`` into a ``.npy`` file.

    Written through a memory map, so neither the device nor the host ever holds them
    all. Returns the array shape.
    """
    n, n_t = z.shape
    n_edges = n * (n - 1) // 2
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n_edges, n_t))
    for pos, block in edge_blocks(z, max_bytes):
        out[pos : pos + block.shape[0]] = block.cpu().numpy()
    out.flush()
    del out
    return n_edges, n_t


def _unit(rows: Tensor) -> Tensor:
    norm = rows.norm(dim=1, keepdim=True)
    return torch.where(norm > 0, rows / norm.clamp_min(1e-30), torch.zeros_like(rows))


@full_fp32_matmul()
def edge_gram(z: Tensor, max_bytes: int = 1 << 28) -> tuple[Tensor, Tensor, int]:
    """``G = Ê^T Ê`` (T x T), ``s = Σ_k ê_k`` (T,) in float64 on the CPU, and E."""
    from fastfuncstuff.utils import widest_float_dtype

    dt = widest_float_dtype(z.device)
    n_t = z.shape[1]
    g = torch.zeros(n_t, n_t, dtype=dt, device=z.device)
    s = torch.zeros(n_t, dtype=dt, device=z.device)
    n_edges = 0
    for _, block in edge_blocks(z, max_bytes):
        u = _unit(block).to(dt)
        g += u.T @ u
        s += u.sum(0)
        n_edges += u.shape[0]
    return g.to("cpu").to(torch.float64), s.to("cpu").to(torch.float64), n_edges


#: k-means distances over eFC rows; both exact through the T-dimensional embedding.
EFC_METRICS = ("euclidean", "correlation")
#: What edge communities cluster: the edge series (the reference), or eFC's rows.
COMMUNITY_SPACES = ("ets", "efc", "efc_corr")


@full_fp32_matmul()
def efc_embedding(
    z: Tensor, metric: str = "euclidean", tol: float = 1e-6, max_bytes: int = 1 << 28
) -> Tensor:
    """``(E, rank)`` float32 rows whose pairwise distances are those of eFC's rows.

    ``"euclidean"``: Euclidean distance between eFC rows. ``"correlation"``: each eFC
    row centred over its E entries and scaled to unit length first, so the distance
    is ``sqrt(2 - 2 corr)``. Returned on ``z``'s device.
    """
    from fastfuncstuff import policy_linalg as plinalg

    if metric not in EFC_METRICS:
        raise ValueError(f"metric must be one of {EFC_METRICS}, got {metric!r}")
    g, s, n_edges = edge_gram(z, max_bytes)
    if metric == "correlation":
        g = g - torch.outer(s, s) / n_edges
    lam, vec = plinalg.eigh(g)
    keep = lam > tol * float(lam.max().clamp_min(1e-300))
    w = (vec[:, keep] * lam[keep].sqrt()).to(z.device).to(torch.float32)
    out = torch.empty(n_edges, int(keep.sum()), dtype=torch.float32, device=z.device)
    for pos, block in edge_blocks(z, max_bytes):
        out[pos : pos + block.shape[0]] = _unit(block) @ w
    return _unit(out) if metric == "correlation" else out


@dataclass
class EdgeCommunities:
    labels: np.ndarray  # (E,) community of each edge, 0..k-1, edge_index order
    k: int
    inertia: float

    def matrix(self, n: int) -> np.ndarray:
        """``(n, n)`` community of each ROI pair; -1 on the diagonal."""
        m = np.full((n, n), -1, dtype=np.int64)
        iu, ju = edge_index(n)
        m[iu, ju] = self.labels
        m[ju, iu] = self.labels
        return m

    def node_profile(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Per ROI, the fraction of its edges in each community ``(n, k)`` and the
        normalised entropy of that profile ``(n,)``: 0 when all its edges share one
        community, 1 when they spread evenly over all k (the node's overlap)."""
        m = self.matrix(n)
        frac = np.stack([(m == c).sum(1) for c in range(self.k)], axis=1) / max(n - 1, 1)
        plogp = np.where(frac > 0, frac * np.log(np.where(frac > 0, frac, 1.0)), 0.0)
        return frac, -plogp.sum(1) / math.log(self.k)

    def profile_similarity(self, n: int) -> np.ndarray:
        """``(n, n)``: the fraction of columns where two ROIs' rows of :meth:`matrix`
        agree, i.e. how alike their edge-community profiles are (the reference's
        ``fcn_profilesim``; diagonal entries count as a mismatch, as there)."""
        m = self.matrix(n)
        agree = np.zeros((n, n))
        for c in range(self.k):
            a = (m == c).astype(np.float64)
            agree += a @ a.T
        np.fill_diagonal(agree, n)
        return agree / n


def community_points(z: Tensor, on: str = "ets", max_bytes: int = 1 << 28) -> Tensor:
    """``(E, D)`` float32 points k-means clusters, on ``z``'s device.

    ``"ets"``: the edge series (D = T). ``"efc"`` / ``"efc_corr"``: rows of eFC, by
    the exact embedding with the Euclidean / correlation distance (D = rank <= T).
    """
    if on not in COMMUNITY_SPACES:
        raise ValueError(f"on must be one of {COMMUNITY_SPACES}, got {on!r}")
    if on == "efc":
        return efc_embedding(z, "euclidean", max_bytes=max_bytes)
    if on == "efc_corr":
        return efc_embedding(z, "correlation", max_bytes=max_bytes)
    return torch.cat([b for _, b in edge_blocks(z, max_bytes)], dim=0)


def edge_communities(
    z: Tensor,
    k: int,
    *,
    on: str = "ets",
    replicates: int = 10,
    n_iter: int = 1000,
    seed: int = 0,
) -> EdgeCommunities:
    """k-means edge communities (see the module docstring for ``on``).

    k-means++ seeding, ``replicates`` restarts, the lowest-inertia labelling kept;
    ``n_iter`` caps Lloyd's iterations (MATLAB's ``maxiter`` in the reference: 1000).
    """
    from fastfuncstuff.dynamics.bsds.init import _kmeans_inertia, kmeans

    if k < 2:
        raise ValueError("edge communities need k >= 2")
    x = community_points(z, on)
    gen = torch.Generator(device=x.device).manual_seed(seed)
    best, best_inertia = None, float("inf")
    for _ in range(replicates):
        labels, centers = kmeans(x, k, n_iter=n_iter, generator=gen)
        inertia = _kmeans_inertia(x, labels, centers)
        if inertia < best_inertia:
            best, best_inertia = labels, inertia
    assert best is not None
    return EdgeCommunities(labels=best.cpu().numpy(), k=k, inertia=best_inertia)


def efc_bytes(n_rois: int) -> int:
    """Size of the full float32 eFC for ``n_rois`` ROIs."""
    e = n_rois * (n_rois - 1) // 2
    return e * e * 4


@full_fp32_matmul()
def write_efc(z: Tensor, path: str | Path, max_bytes: int = 1 << 28) -> tuple[int, int]:
    """Stream the full ``(E, E)`` float32 eFC into a ``.npy``, a block of rows at a time.

    Refuses when the disk lacks room. The unit edge series (``E x T``) are held on
    ``z``'s device and each block of rows is one GEMM against them.
    """
    n = z.shape[0]
    n_edges = n * (n - 1) // 2
    need = efc_bytes(n)
    free = shutil.disk_usage(Path(path).resolve().parent).free
    if need > 0.9 * free:
        raise ValueError(f"full eFC is {need / 1e9:.1f} GB; only {free / 1e9:.1f} GB free")
    u = torch.cat([_unit(b) for _, b in edge_blocks(z, max_bytes)], dim=0)
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n_edges, n_edges))
    rows = max(1, max_bytes // max(4 * n_edges, 1))
    for a in range(0, n_edges, rows):
        out[a : a + rows] = (u[a : a + rows] @ u.T).cpu().numpy()
    out.flush()
    del out
    return n_edges, n_edges
