"""Streamed voxel-to-voxel temporal correlation summaries (single timeseries).

The full ``V x V`` correlation matrix of an fMRI timeseries is far too large to
materialize (``V`` can be 100k+), but its *summary* — the histogram of pairwise
Pearson r and the mean correlation as a function of voxel separation — is cheap
if we never form the whole matrix. We stream row-blocks of the normalized data,
compute ``Xn[block] @ Xn.T`` (a chunked GEMM, block size from the memory module),
and accumulate histograms + distance-binned sums over the **upper triangle only**
(unordered pairs, no self-pairs).

This is the engine behind the NORDIC factor-sweep diagnostic: applied to the
denoising residual, a correlation distribution centered on the timepoint null
(and flat with distance) means only spatially-independent thermal noise was
removed; a shift away from null / a near>far distance interaction means real,
locally-structured signal landed in the residual.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from fastfuncstuff.memory import get_available_memory


@dataclass
class CorrSummary:
    """Summary of a voxel-to-voxel correlation matrix, computed without forming it."""

    n_voxels: int  # valid (non-constant) voxels actually correlated
    n_pairs: int  # unordered pairs counted
    mean_r: float  # mean signed Pearson r over all pairs
    mean_abs_r: float  # mean |r|
    std_r: float  # std of signed r
    median_r: float  # median signed r (from the histogram)
    iqr_r: tuple[float, float]  # (q25, q75) signed r (from the histogram)
    r_edges: torch.Tensor  # (nbin+1,) signed-r histogram bin edges
    r_hist: torch.Tensor  # (nbin,) signed-r counts
    abs_r_hist: torch.Tensor  # (nbin,) |r| counts over [0, 1]
    dist_edges: torch.Tensor  # (ndist+1,) distance bin edges (voxel units)
    dist_centers: torch.Tensor  # (ndist,) bin centers
    dist_mean_r: torch.Tensor  # (ndist,) mean signed r per distance bin
    dist_mean_abs_r: torch.Tensor  # (ndist,) mean |r| per distance bin
    dist_count: torch.Tensor  # (ndist,) pair count per distance bin


def analytic_r_null(n_timepoints: int) -> dict[str, float]:
    """Timepoint null for Pearson r of two independent series of length ``T``.

    Under H0 (uncorrelated, temporally white) r has mean 0 and variance
    ``1/(T-1)``; ``|r|`` follows a folded normal with ``E|r| = sqrt(2/(pi(T-1)))``.
    These are the reference marks drawn on the diagnostic plots.

    Caveat: assumes the residual is temporally white. Thermal noise is
    approximately white, so this is a sound reference for the NORDIC residual;
    strong temporal autocorrelation would inflate the true null (an
    effective-dof correction is a possible future refinement).
    """
    t = max(2, int(n_timepoints))
    sd = 1.0 / math.sqrt(t - 1)
    return {
        "mean_r": 0.0,
        "sd_r": sd,
        "mean_abs_r": math.sqrt(2.0 / (math.pi * (t - 1))),
        "ci95_r": 1.959963985 * sd,
    }


def voxel_corr_strength(
    ts: torch.Tensor, *, block_size: int | None = None, eps: float = 1e-12
) -> torch.Tensor:
    """Per-voxel mean |r| to all other voxels (streamed, never forms V×V).

    A scalar "how correlated is this voxel with the rest" score, used to pick the
    danger zone — voxels carrying shared structure in the *input* are exactly the
    pairs NORDIC over-removal would corrupt. Constant voxels score 0.
    """
    device = ts.device
    x = ts.to(torch.float32)
    x = x - x.mean(dim=1, keepdim=True)
    norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()
    xn = x / norm.clamp(min=eps)
    v = xn.shape[0]
    acc = torch.zeros(v, dtype=torch.float32, device=device)
    if block_size is None:
        block_size = _row_block_size(v, device)
    for r0 in range(0, v, block_size):
        r1 = min(r0 + block_size, v)
        c = (xn[r0:r1] @ xn.T).abs()  # (b, V)
        acc[r0:r1] = c.sum(dim=1) - 1.0  # drop self (|r|=1)
    return (acc / max(1, v - 1)).where(norm.squeeze(1) > eps, torch.zeros_like(acc))


def _row_block_size(n_voxels: int, device: torch.device, safety: float = 0.4) -> int:
    """Rows of the ``(block, V)`` correlation tile that fit the memory budget.

    A few ``(block, V)`` float32 temporaries are live at once (the GEMM output,
    the distance tile, masks), so we size conservatively against the memory
    module's reported availability rather than hardcoding.
    """
    avail = get_available_memory(device)
    # ~6 simultaneous (block, V) float32 buffers (corr, dist, masked gathers).
    per_row = max(1, n_voxels) * 4 * 6
    block = int(avail * safety / per_row)
    return max(64, min(block, max(64, n_voxels)))


def corr_histogram_distance(
    ts: torch.Tensor,
    coords: torch.Tensor,
    *,
    r_bins: int = 201,
    dist_edges: torch.Tensor | None = None,
    n_dist_bins: int = 24,
    block_size: int | None = None,
    eps: float = 1e-12,
) -> CorrSummary:
    """Histogram + distance profile of the voxel-to-voxel correlation matrix.

    Parameters
    ----------
    ts : (V, T) tensor
        Per-voxel time series (NORDIC residual magnitude). Demeaned and
        unit-normalized internally; constant (zero-variance) voxels are dropped.
    coords : (V, 3) tensor
        Voxel ijk coordinates, same order as ``ts`` rows, for pair distances.
    r_bins : int
        Number of histogram bins spanning [-1, 1] (signed) and [0, 1] (|r|).
    dist_edges : (ndist+1,) tensor, optional
        Distance bin edges (voxel units). If None, ``n_dist_bins`` equal bins
        from 0 to the 99th-percentile pair distance (estimated from the bounding
        box) are used.
    n_dist_bins : int
        Number of distance bins when ``dist_edges`` is None.
    block_size : int, optional
        Row-block size for the streamed GEMM. If None, from the memory module.

    Notes
    -----
    Counts the **upper triangle** only (unordered pairs ``i < j``, no diagonal),
    so each pair is counted exactly once. All work stays on ``ts.device``.
    """
    device = ts.device
    ts = ts.to(torch.float32)
    coords = coords.to(torch.float32).to(device)

    # Normalize rows; drop constant voxels (undefined correlation).
    x = ts - ts.mean(dim=1, keepdim=True)
    norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()
    valid = norm.squeeze(1) > eps
    x = x[valid] / norm[valid]
    coords = coords[valid]
    v = x.shape[0]
    if v < 2:
        raise ValueError(f"need >= 2 non-constant voxels to correlate, got {v}")

    # Distance bin edges from the bounding-box diagonal if not supplied.
    if dist_edges is None:
        extent = coords.amax(dim=0) - coords.amin(dim=0)
        dmax = float(torch.sqrt((extent**2).sum()).item())
        dist_edges = torch.linspace(0.0, max(dmax, 1.0), n_dist_bins + 1, device=device)
    else:
        dist_edges = dist_edges.to(device).to(torch.float32)
    n_dist = dist_edges.numel() - 1

    # Everything stays on-device: float64 accumulators where the device supports
    # them (CPU/CUDA, exact counts for free), float32 on MPS (no float64 there).
    # No per-block host round-trip — the user's speed/accuracy call is "don't ship
    # tensors around; float32 is fine." We sync once, at the end.
    acc_dtype = torch.float32 if device.type == "mps" else torch.float64
    r_edges = torch.linspace(-1.0, 1.0, r_bins + 1, device=device)
    abs_edges = torch.linspace(0.0, 1.0, r_bins + 1, device=device)
    r_hist = torch.zeros(r_bins, dtype=acc_dtype, device=device)
    abs_r_hist = torch.zeros(r_bins, dtype=acc_dtype, device=device)
    dist_sum_r = torch.zeros(n_dist, dtype=acc_dtype, device=device)
    dist_sum_abs = torch.zeros(n_dist, dtype=acc_dtype, device=device)
    dist_count = torch.zeros(n_dist, dtype=acc_dtype, device=device)
    tot = torch.zeros(4, dtype=acc_dtype, device=device)  # n, sum_r, sum_abs, sum_r2

    coord_sq = (coords**2).sum(dim=1)  # (V,)
    arange_v = torch.arange(v, device=device)

    if block_size is None:
        block_size = _row_block_size(v, device)

    def _binsum(idx: torch.Tensor, n: int, vals: torch.Tensor | None) -> torch.Tensor:
        out = torch.zeros(n, dtype=acc_dtype, device=device)
        src = torch.ones_like(idx, dtype=acc_dtype) if vals is None else vals.to(acc_dtype)
        return out.index_add_(0, idx, src)

    for r0 in range(0, v, block_size):
        r1 = min(r0 + block_size, v)
        xb = x[r0:r1]  # (b, T)
        c = (xb @ x.T).clamp(-1.0, 1.0)  # (b, V) correlations

        # Upper-triangle mask: keep column j > global row i only.
        gi = arange_v[r0:r1]  # (b,)
        keep = arange_v[None, :] > gi[:, None]  # (b, V)

        # Pairwise squared distance without a (b, V, 3) temporary.
        cb = coords[r0:r1]  # (b, 3)
        d2 = coord_sq[r0:r1][:, None] + coord_sq[None, :] - 2.0 * cb @ coords.T
        dist = torch.sqrt(d2.clamp(min=0.0))

        cv = c[keep]  # (P,) selected correlations
        av = cv.abs()
        dv = dist[keep]
        del c, dist, keep

        # Histograms via bucketize+index_add (device-safe; avoids histc on MPS).
        rb = (torch.bucketize(cv, r_edges, right=False) - 1).clamp(0, r_bins - 1)
        ab = (torch.bucketize(av, abs_edges, right=False) - 1).clamp(0, r_bins - 1)
        db = (torch.bucketize(dv, dist_edges, right=False) - 1).clamp(0, n_dist - 1)
        r_hist += _binsum(rb, r_bins, None)
        abs_r_hist += _binsum(ab, r_bins, None)
        dist_sum_r += _binsum(db, n_dist, cv)
        dist_sum_abs += _binsum(db, n_dist, av)
        dist_count += _binsum(db, n_dist, None)

        tot[0] += cv.numel()
        tot[1] += cv.sum().to(acc_dtype)
        tot[2] += av.sum().to(acc_dtype)
        tot[3] += (cv * cv).sum().to(acc_dtype)
        del cv, av, dv, rb, ab, db

    tot = tot.cpu()
    total_n = float(tot[0])
    mean_r = float(tot[1]) / total_n if total_n else 0.0
    mean_abs_r = float(tot[2]) / total_n if total_n else 0.0
    var_r = max(0.0, float(tot[3]) / total_n - mean_r**2) if total_n else 0.0
    std_r = math.sqrt(var_r)

    r_hist = r_hist.cpu().double()
    abs_r_hist = abs_r_hist.cpu().double()
    median_r, q25, q75 = _hist_quantiles(r_hist, r_edges.cpu(), (0.5, 0.25, 0.75))
    cnt = dist_count.clamp(min=1.0)
    dist_centers = 0.5 * (dist_edges[:-1] + dist_edges[1:])

    return CorrSummary(
        n_voxels=int(v),
        n_pairs=int(total_n),
        mean_r=mean_r,
        mean_abs_r=mean_abs_r,
        std_r=std_r,
        median_r=median_r,
        iqr_r=(q25, q75),
        r_edges=r_edges.cpu(),
        r_hist=r_hist.cpu(),
        abs_r_hist=abs_r_hist.cpu(),
        dist_edges=dist_edges.cpu(),
        dist_centers=dist_centers.cpu(),
        dist_mean_r=(dist_sum_r / cnt).cpu(),
        dist_mean_abs_r=(dist_sum_abs / cnt).cpu(),
        dist_count=dist_count.cpu(),
    )


def _hist_quantiles(
    hist: torch.Tensor,
    edges: torch.Tensor,
    qs: tuple[float, ...],
) -> tuple[float, ...]:
    """Linear-interpolated quantiles read off a histogram (counts + edges)."""
    total = float(hist.sum().item())
    if total <= 0:
        centers = 0.5 * (edges[:-1] + edges[1:])
        mid = float(centers[len(centers) // 2].item())
        return tuple(mid for _ in qs)
    cdf = torch.cumsum(hist, dim=0) / total
    cdf = cdf.cpu()
    edges_c = edges.cpu()
    out = []
    for q in qs:
        idx = int(torch.searchsorted(cdf, torch.tensor(q)).item())
        idx = min(max(idx, 0), hist.numel() - 1)
        out.append(float(0.5 * (edges_c[idx] + edges_c[idx + 1]).item()))
    return tuple(out)
