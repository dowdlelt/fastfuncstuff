"""Spatial autocorrelation on a surface, and blurring data TO a smoothness.

**The ACF** is measured as a function of geodesic distance and fitted to AFNI's mixed
model ``a exp(-r^2 / 2b^2) + (1 - a) exp(-r / c)``, as SUMA's ``SurfFWHM`` does
(``SUMA_SurfACF.h``), with the fit shared with the volume path (``stats.fwhmx``).
A FWHM describes the ACF near the origin; cluster inference depends on the far tail,
and real data have a heavier tail than a FWHM implies (Eklund et al. 2016). On onavg
in a real subject, a 4 mm blur took the 1-difference FWHM from 4.0 to 8.0 mm where
the kernel alone predicts ~5.7. That broad component is what the tail term is for.

Two choices of ours: correlation is taken **over time between vertices** (standardised
residual time series) rather than over space within each frame -- the same quantity
for a stationary field, and here one matrix product per block of centre vertices,
which a GPU does at once -- and distance is Dijkstra along mesh edges, as SUMA's
``SUMA_getoffsets2``.

**Blurring to a smoothness** (SurfSmooth ``-target_fwhm``, 3dBlurToFWHM): find the heat
kernel that brings a master (detrended data, or better the residuals of a fit) to the
target FWHM, then smooth the data with exactly that kernel. The smoother is linear, so
a bisection on the master is the whole search; SUMA instead re-estimates after every
explicit pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .smooth import HeatSmoother, mesh_edges, surface_fwhm

__all__ = ["SurfaceACF", "blur_to_fwhm", "detrend", "fit_surface_acf", "surface_acf"]


@dataclass
class SurfaceACF:
    """A measured ACF curve and its mixed-model fit (AFNI's a, b, c and FWHM)."""

    r: np.ndarray  # (B,) bin centres, mm
    curve: np.ndarray  # (B,) mean correlation, NaN where too few pairs
    a: float
    b: float
    c: float
    fwhm: float


def _edge_graph(
    vertices: np.ndarray, faces: np.ndarray, mask: np.ndarray, rings: int = 2
) -> sparse.csr_matrix:
    """Distance graph for Dijkstra: mesh edges plus chords to ``rings``-ring neighbours.

    Paths along edges zigzag: on a triangulated sheet they run 8% long at 10 mm
    (SUMA's graph distances share this), which stretches a measured ACF outward. Chords
    to second neighbours cut that to 1-2.5% (third: under 1%), and stay on the local
    sheet because they only join vertices a couple of edges apart. More rings on a
    coarse mesh would start to cut across tight folds.
    """
    e = mesh_edges(faces)
    e = e[mask[e[:, 0]] & mask[e[:, 1]]]
    n = len(vertices)
    adj = sparse.csr_matrix(
        (np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), (n, n)
    )
    reach = adj.copy()
    for _ in range(max(rings, 1) - 1):
        reach = reach + reach @ adj
    reach = sparse.triu(reach, 1).tocoo()
    w = np.linalg.norm(vertices[reach.row] - vertices[reach.col], axis=1)
    return sparse.csr_matrix(
        (np.r_[w, w], (np.r_[reach.row, reach.col], np.r_[reach.col, reach.row])), (n, n)
    )


def _standardise(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = x - x.mean(axis=1, keepdims=True)
    sd = z.std(axis=1)
    ok = sd > 0
    z[ok] /= sd[ok, None]
    z[~ok] = 0.0
    return z.astype(np.float32), ok


def surface_acf(
    residuals: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    mask: np.ndarray | None = None,
    radius: float | None = None,
    dr: float | None = None,
    n_centres: int = 1500,
    device=None,
    seed: int = 0,
) -> SurfaceACF:
    """Mean correlation of ``(V, T)`` residuals against geodesic distance, and its fit.

    Defaults follow SUMA: bins one mean edge length wide, out to 20 of them but at
    least 30 mm (a tail several mm long is still at 0.4 after 10 mm, and the fit needs
    to see it fall). ~``n_centres`` centre vertices are enough: every one is correlated
    against every vertex within ``radius``.
    """
    import torch

    from fastfuncstuff.stats.fwhmx import fit_acf_curve

    v = np.asarray(vertices, np.float64)
    m = np.ones(len(v), bool) if mask is None else np.asarray(mask, bool)
    z, ok = _standardise(np.asarray(residuals, np.float64))
    m = m & ok
    graph = _edge_graph(v, faces, m)
    e = mesh_edges(faces)
    lengths = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    dr = float(dr or (lengths.mean() if lengths.size else 1.0))
    radius = float(radius or max(20 * dr, 30.0))
    nb = int(radius / dr) + 1
    rng = np.random.default_rng(seed)
    inside = np.flatnonzero(m)
    centres = np.sort(rng.choice(inside, size=min(n_centres, inside.size), replace=False))

    from scipy.sparse.csgraph import dijkstra

    dev = torch.device("cpu") if device is None else device
    zt = torch.as_tensor(z, device=dev)
    sums = np.zeros(nb)
    counts = np.zeros(nb)
    t = z.shape[1]
    for start in range(0, centres.size, 128):
        block = centres[start : start + 128]
        dist = dijkstra(graph, indices=block, limit=radius)  # (C, V), inf beyond
        corr = (zt[torch.as_tensor(block, device=dev)] @ zt.T / t).cpu().numpy()
        near = np.isfinite(dist) & (dist > 0)
        near[:, ~m] = False
        ib = np.rint(dist[near] / dr).astype(np.int64)
        keep = ib < nb
        np.add.at(sums, ib[keep], corr[near][keep])
        np.add.at(counts, ib[keep], 1.0)
    curve = np.full(nb, np.nan)
    use = counts > 5  # AFNI's guard against thinly populated bins
    curve[use] = sums[use] / counts[use]
    curve[0] = 1.0
    r = np.arange(nb) * dr
    good = np.isfinite(curve)
    a, b, c, fwhm = fit_acf_curve(
        torch.as_tensor(r[good]), torch.as_tensor(curve[good]), torch.device("cpu")
    )
    return SurfaceACF(r, curve, a, b, c, fwhm)


def fit_surface_acf(r: np.ndarray, curve: np.ndarray) -> tuple[float, float, float, float]:
    """``(a, b, c, FWHM)`` of a curve, NaN bins dropped (the shared volume fitter)."""
    import torch

    from fastfuncstuff.stats.fwhmx import fit_acf_curve

    good = np.isfinite(curve)
    return fit_acf_curve(
        torch.as_tensor(np.asarray(r)[good]),
        torch.as_tensor(np.asarray(curve)[good]),
        torch.device("cpu"),
    )


def detrend(data: np.ndarray, run_lengths: list[int] | None = None, degree: int = 3) -> np.ndarray:
    """``(V, T)`` minus a per-run Legendre polynomial fit: a master for :func:`blur_to_fwhm`
    when no residuals are at hand. Task signal stays in; residuals are the better master."""
    x = np.asarray(data, np.float64)
    lengths = run_lengths or [x.shape[1]]
    out = np.empty_like(x)
    start = 0
    for n in lengths:
        tt = np.linspace(-1.0, 1.0, n)
        basis = np.polynomial.legendre.legvander(tt, min(degree, max(n - 1, 0)))
        seg = x[:, start : start + n]
        coef, *_ = np.linalg.lstsq(basis, seg.T, rcond=None)
        out[:, start : start + n] = seg - (basis @ coef).T
        start += n
    return out


def blur_to_fwhm(
    master: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    target: float,
    mask: np.ndarray | None = None,
    n_columns: int = 40,
    tol: float = 0.02,
    device=None,
    seed: int = 0,
) -> tuple[HeatSmoother | None, float, float]:
    """The heat smoother that brings ``master`` to ``target`` FWHM (1-difference estimate).

    Returns ``(smoother, kernel_fwhm, achieved)``; ``smoother`` is None when the master
    is already at least that smooth (nothing to add, as 3dBlurToFWHM reports). A random
    subset of ``n_columns`` time points carries the search: the estimate averages over
    every edge, so a few dozen columns already pin it to a percent.
    """
    x = np.asarray(master)
    cols = np.random.default_rng(seed).choice(
        x.shape[1], size=min(n_columns, x.shape[1]), replace=False
    )
    sub = x[:, np.sort(cols)]

    def measure(kernel: float) -> tuple[float, HeatSmoother | None]:
        if kernel <= 0:
            return surface_fwhm(sub, vertices, faces, mask)[0], None
        hs = HeatSmoother(vertices, faces, kernel, mask=mask, device=device)
        return surface_fwhm(hs(sub), vertices, faces, mask)[0], hs

    start, _ = measure(0.0)
    if not np.isfinite(start) or start >= target:
        return None, 0.0, float(start)
    lo, hi = 0.0, float(target)
    fhi, hs_hi = measure(hi)
    while fhi < target and hi < 8 * target:
        lo, hi = hi, hi * 1.5
        fhi, hs_hi = measure(hi)
    best = (hi, fhi, hs_hi)
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        fmid, hs_mid = measure(mid)
        if fmid >= target:
            hi, best = mid, (mid, fmid, hs_mid)
        else:
            lo = mid
        if abs(best[1] - target) <= tol * target:
            break
    kernel, achieved, hs = best
    return hs, float(kernel), float(achieved)
