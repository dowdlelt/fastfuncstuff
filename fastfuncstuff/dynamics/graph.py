"""Graph-theoretic metrics on per-state functional connectivity.

Each BSDS state is a weighted graph (its dynamic-FC matrix). The papers
([[Taghia 2018]] Fig. 6) characterise states by graph measures — how integrated
vs segregated each state's network is. These are pure functions of the fit's
per-state FC, so they belong in the CLI's batch outputs (no external data).

Weighted, dependency-light implementations (no networkx): a correlation matrix
becomes a non-negative weight graph (``|r|`` or positive-only, optionally
proportional-thresholded), edge **distance** is ``1/weight`` (a stronger link is
a shorter path — the connectomics convention). From that we derive:

- **node strength** — weighted degree, how strongly a region is coupled;
- **weighted clustering** (Onnela) — local segregation / cliquishness;
- **betweenness centrality** (Brandes on Dijkstra shortest paths) — how much a
  region bridges others;
- **global / nodal efficiency** — network integration (mean inverse path length).

``D`` is tens–low-hundreds and ``K`` is small, so the O(K·D·(E + D log D)) cost is
negligible; everything runs on the CPU in float64.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import torch


def _to_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def fc_to_weights(
    fc: np.ndarray,
    *,
    signed: str = "absolute",
    density: float | None = None,
) -> np.ndarray:
    """Turn a correlation matrix into a non-negative, zero-diagonal weight graph.

    ``signed="absolute"`` uses ``|r|``; ``"positive"`` keeps positive edges only.
    ``density`` (in ``(0, 1]``) proportional-thresholds to the strongest fraction
    of edges, the standard way to compare graphs at matched cost.
    """
    w = _to_numpy(fc).astype(np.float64).copy()
    if signed == "absolute":
        w = np.abs(w)
    elif signed == "positive":
        w = np.clip(w, 0.0, None)
    else:
        raise ValueError("signed must be 'absolute' or 'positive'")
    np.fill_diagonal(w, 0.0)
    w = np.maximum(w, w.T)  # enforce symmetry against tiny numerical asymmetry
    if density is not None:
        if not 0.0 < density <= 1.0:
            raise ValueError("density must be in (0, 1]")
        d = w.shape[0]
        iu, ju = np.triu_indices(d, k=1)
        vals = w[iu, ju]
        keep = int(round(density * vals.size))
        if keep < vals.size:
            thresh = np.sort(vals)[::-1][keep - 1] if keep > 0 else np.inf
            mask = w < thresh
            w[mask] = 0.0
            np.fill_diagonal(w, 0.0)
    return w


def node_strength(w: np.ndarray) -> np.ndarray:
    """Weighted degree per node (``D,``)."""
    return w.sum(axis=1)


def weighted_clustering(w: np.ndarray) -> np.ndarray:
    """Onnela weighted clustering coefficient per node (``D,``).

    ``C_i = (2/(k_i(k_i-1))) Σ_{j<h} (ŵ_ij ŵ_ih ŵ_jh)^{1/3}`` with weights scaled
    by the max edge, ``k_i`` the binary degree. Zero for degree < 2.
    """
    wmax = w.max()
    if wmax <= 0:
        return np.zeros(w.shape[0])
    cw = (w / wmax) ** (1.0 / 3.0)
    tri = np.diagonal(cw @ cw @ cw)  # sum of triangle geometric-mean weights * 2
    deg = (w > 0).sum(axis=1).astype(np.float64)
    denom = deg * (deg - 1.0)
    out = np.zeros(w.shape[0])
    nz = denom > 0
    out[nz] = tri[nz] / denom[nz]
    return out


def _dijkstra(dist: np.ndarray, source: int):
    """Dijkstra from ``source`` on a distance matrix (``inf`` = no edge).

    Returns ``(d, sigma, order, preds)``: shortest-path lengths, path counts,
    nodes in non-decreasing-distance order, and predecessor lists — the inputs
    Brandes betweenness needs.
    """
    n = dist.shape[0]
    d = np.full(n, np.inf)
    sigma = np.zeros(n)
    preds: list[list[int]] = [[] for _ in range(n)]
    d[source] = 0.0
    sigma[source] = 1.0
    seen = {source: 0.0}
    order: list[int] = []
    visited = np.zeros(n, dtype=bool)
    heap: list[tuple[float, int]] = [(0.0, source)]
    while heap:
        dv, v = heapq.heappop(heap)
        if visited[v]:
            continue
        visited[v] = True
        order.append(v)
        for w_ in np.flatnonzero(np.isfinite(dist[v])):
            if w_ == v:
                continue
            alt = dv + dist[v, w_]
            if alt < seen.get(w_, np.inf) - 1e-15:
                seen[w_] = alt
                d[w_] = alt
                sigma[w_] = sigma[v]
                preds[w_] = [v]
                heapq.heappush(heap, (alt, w_))
            elif abs(alt - seen.get(w_, np.inf)) <= 1e-15:
                sigma[w_] += sigma[v]
                preds[w_].append(v)
    return d, sigma, order, preds


def betweenness_centrality(w: np.ndarray, *, normalized: bool = True) -> np.ndarray:
    """Weighted betweenness centrality per node (``D,``), via Brandes + Dijkstra."""
    n = w.shape[0]
    with np.errstate(divide="ignore"):
        dist = np.where(w > 0, 1.0 / w, np.inf)
    np.fill_diagonal(dist, np.inf)
    bc = np.zeros(n)
    for s in range(n):
        d, sigma, order, preds = _dijkstra(dist, s)
        delta = np.zeros(n)
        for v in reversed(order):
            for p in preds[v]:
                delta[p] += (sigma[p] / sigma[v]) * (1.0 + delta[v])
            if v != s:
                bc[v] += delta[v]
    bc /= 2.0  # undirected: each shortest path is counted from both endpoints
    if normalized and n > 2:
        bc /= (n - 1) * (n - 2) / 2.0
    return bc


def efficiency(w: np.ndarray) -> tuple[float, np.ndarray]:
    """Global efficiency (scalar) and nodal efficiency (``D,``).

    Efficiency is the mean inverse shortest-path length; high = integrated.
    """
    n = w.shape[0]
    with np.errstate(divide="ignore"):
        dist = np.where(w > 0, 1.0 / w, np.inf)
    np.fill_diagonal(dist, np.inf)
    inv = np.zeros((n, n))
    for s in range(n):
        d, _, _, _ = _dijkstra(dist, s)
        with np.errstate(divide="ignore"):
            row = np.where(np.isfinite(d) & (d > 0), 1.0 / d, 0.0)
        inv[s] = row
    nodal = inv.sum(axis=1) / max(n - 1, 1)
    glob = float(nodal.mean())
    return glob, nodal


@dataclass
class GraphMetrics:
    """Per-state graph-theoretic metrics on the dynamic-FC networks."""

    n_states: int
    signed: str
    density: float | None
    strength: np.ndarray  # (K, D)
    clustering: np.ndarray  # (K, D)
    betweenness: np.ndarray  # (K, D)
    nodal_efficiency: np.ndarray  # (K, D)
    global_efficiency: np.ndarray  # (K,)
    mean_clustering: np.ndarray  # (K,)


def state_graph_metrics(
    fc,
    *,
    signed: str = "absolute",
    density: float | None = None,
) -> GraphMetrics:
    """Graph metrics for every state's FC matrix.

    ``fc`` is ``(K, D, D)`` (e.g. ``stats.state_fc``). Returns per-node arrays plus
    per-state global-efficiency and mean-clustering summaries.
    """
    fc = _to_numpy(fc)
    k, d, _ = fc.shape
    strength = np.zeros((k, d))
    clustering = np.zeros((k, d))
    betweenness = np.zeros((k, d))
    nodal_eff = np.zeros((k, d))
    glob = np.zeros(k)
    for s in range(k):
        w = fc_to_weights(fc[s], signed=signed, density=density)
        strength[s] = node_strength(w)
        clustering[s] = weighted_clustering(w)
        betweenness[s] = betweenness_centrality(w)
        glob[s], nodal_eff[s] = efficiency(w)
    return GraphMetrics(
        n_states=k,
        signed=signed,
        density=density,
        strength=strength,
        clustering=clustering,
        betweenness=betweenness,
        nodal_efficiency=nodal_eff,
        global_efficiency=glob,
        mean_clustering=clustering.mean(axis=1),
    )
