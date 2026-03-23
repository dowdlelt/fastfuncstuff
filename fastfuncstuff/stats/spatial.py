"""GPU-accelerated spatial correlation utilities for neuroimaging volumes.

Computes correlations across voxels between 3D spatial maps. Supports:

- **Pearson r** — linear correlation (GPU-accelerated via torch matmul)
- **Spearman rho** — rank-based correlation (GPU-accelerated ranking + matmul)
- **Kendall tau** — concordance-based correlation (CPU, scipy)

Core API
--------
``spatial_correlation(a, b, mask, method)``
    Correlation between two 3D volumes.

``spatial_correlation_matrix(images_a, images_b, mask, method)``
    Full cross-correlation matrix between all volumes of two 4D datasets.

``optimal_matching(corr_matrix)``
    Hungarian algorithm for best 1-to-1 volume matching.

``consistency_report(corr_matrix)``
    Summary statistics: optimal matching quality, coverage at thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_masked_flat(vol: Tensor, mask: Tensor | None) -> Tensor:
    """Flatten a 3D volume to 1D, applying mask if provided.

    Args:
        vol: (nz, ny, nx) volume.
        mask: (nz, ny, nx) boolean mask, or None for all voxels.

    Returns:
        (n_voxels,) flat tensor.
    """
    if mask is not None:
        return vol[mask]
    return vol.reshape(-1)


def _standardize_rows(mat: Tensor) -> Tensor:
    """Zero-mean and unit-variance normalize each row in-place.

    Args:
        mat: (n, p) matrix.

    Returns:
        (n, p) standardized matrix. Rows with zero variance become all zeros.
    """
    mean = mat.mean(dim=1, keepdim=True)
    mat = mat - mean
    std = mat.std(dim=1, keepdim=True)
    std = std.clamp(min=1e-12)
    return mat / std


def _rank_rows(mat: Tensor) -> Tensor:
    """Compute fractional ranks along columns for each row.

    Ties get the average rank (matching scipy's default).

    Args:
        mat: (n, p) matrix.

    Returns:
        (n, p) tensor of fractional ranks (1-based).
    """
    n, p = mat.shape
    # argsort twice gives ranks (0-based ordinal)
    sorted_indices = mat.argsort(dim=1)
    ranks = torch.empty_like(mat)
    rows = torch.arange(n, device=mat.device).unsqueeze(1).expand_as(sorted_indices)
    ranks[rows, sorted_indices] = torch.arange(p, device=mat.device, dtype=mat.dtype).unsqueeze(0).expand(n, -1)

    # Handle ties: average rank for tied values
    # Sort values, find ties, compute average ranks
    sorted_vals, sort_idx = mat.sort(dim=1)
    # Detect ties: where consecutive sorted values are equal
    ties = sorted_vals[:, 1:] == sorted_vals[:, :-1]  # (n, p-1)

    if ties.any():
        # Process ties per row — only needed for rows with ties
        tie_rows = ties.any(dim=1).nonzero(as_tuple=True)[0]
        for r in tie_rows:
            vals = sorted_vals[r]
            row_ranks = ranks[r]
            # Find groups of equal values
            unique_vals, inverse, counts = vals.unique(return_inverse=True, return_counts=True)
            tied_mask = counts > 1
            if tied_mask.any():
                for v_idx in tied_mask.nonzero(as_tuple=True)[0]:
                    val_mask = mat[r] == unique_vals[v_idx]
                    avg_rank = row_ranks[val_mask].float().mean()
                    ranks[r, val_mask] = avg_rank

    return ranks + 1  # 1-based


def _prepare_volumes(
    images: Tensor, mask: Tensor | None, device: torch.device
) -> Tensor:
    """Extract masked voxels from a batch of volumes.

    Args:
        images: (n, nz, ny, nx) or (nz, ny, nx) tensor.
        mask: (nz, ny, nx) boolean mask, or None.
        device: target device.

    Returns:
        (n, n_voxels) matrix on device.
    """
    if images.ndim == 3:
        images = images.unsqueeze(0)
    images = images.to(device, dtype=torch.float32)
    n = images.shape[0]
    if mask is not None:
        mask = mask.to(device)
        vecs = torch.stack([images[i][mask] for i in range(n)])
    else:
        vecs = images.reshape(n, -1)
    return vecs


# ---------------------------------------------------------------------------
# Core correlation functions
# ---------------------------------------------------------------------------

def _pearson_matrix(a_vecs: Tensor, b_vecs: Tensor) -> Tensor:
    """Pearson correlation matrix via standardized matmul.

    Args:
        a_vecs: (n1, p) standardized vectors.
        b_vecs: (n2, p) standardized vectors.

    Returns:
        (n1, n2) correlation matrix.
    """
    a_std = _standardize_rows(a_vecs)
    b_std = _standardize_rows(b_vecs)
    # r = (1/p) * sum(z_a * z_b) but standardize already made std=1
    # so r = dot(a, b) / p
    p = a_std.shape[1]
    return (a_std @ b_std.T) / p


def _spearman_matrix(a_vecs: Tensor, b_vecs: Tensor) -> Tensor:
    """Spearman rho matrix: Pearson correlation on ranks.

    Args:
        a_vecs: (n1, p) vectors.
        b_vecs: (n2, p) vectors.

    Returns:
        (n1, n2) correlation matrix.
    """
    a_ranks = _rank_rows(a_vecs)
    b_ranks = _rank_rows(b_vecs)
    return _pearson_matrix(a_ranks, b_ranks)


def _kendall_matrix(a_vecs: Tensor, b_vecs: Tensor) -> np.ndarray:
    """Kendall tau matrix using scipy (CPU).

    Args:
        a_vecs: (n1, p) vectors.
        b_vecs: (n2, p) vectors.

    Returns:
        (n1, n2) numpy correlation matrix.
    """
    from scipy.stats import kendalltau

    a_np = a_vecs.detach().cpu().numpy()
    b_np = b_vecs.detach().cpu().numpy()
    n1, n2 = a_np.shape[0], b_np.shape[0]
    result = np.empty((n1, n2), dtype=np.float64)
    for i in range(n1):
        for j in range(n2):
            tau, _ = kendalltau(a_np[i], b_np[j])
            result[i, j] = tau
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

METHODS = ("pearson", "spearman", "kendall")


def spatial_correlation(
    a: Tensor,
    b: Tensor,
    mask: Tensor | None = None,
    method: str = "pearson",
    device: torch.device | None = None,
) -> float:
    """Spatial correlation between two 3D volumes.

    Args:
        a: (nz, ny, nx) volume.
        b: (nz, ny, nx) volume (same grid as a).
        mask: (nz, ny, nx) boolean mask. None = all voxels.
        method: "pearson", "spearman", or "kendall".
        device: torch device (default: a's device).

    Returns:
        Scalar correlation value.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if device is None:
        device = a.device

    a_vec = _to_masked_flat(a.to(device, dtype=torch.float32), mask.to(device) if mask is not None else None).unsqueeze(0)
    b_vec = _to_masked_flat(b.to(device, dtype=torch.float32), mask.to(device) if mask is not None else None).unsqueeze(0)

    if method == "pearson":
        return _pearson_matrix(a_vec, b_vec).item()
    elif method == "spearman":
        return _spearman_matrix(a_vec, b_vec).item()
    else:
        return _kendall_matrix(a_vec, b_vec)[0, 0]


def spatial_correlation_matrix(
    images_a: Tensor,
    images_b: Tensor,
    mask: Tensor | None = None,
    method: str = "pearson",
    device: torch.device | None = None,
) -> np.ndarray:
    """Cross-correlation matrix between volumes of two 4D datasets.

    Computes ``corr[i, j] = spatial_correlation(images_a[i], images_b[j])``.

    For Pearson/Spearman, this is GPU-accelerated via a single matmul.
    For Kendall, falls back to CPU scipy (O(n1 * n2 * p log p)).

    Args:
        images_a: (n1, nz, ny, nx) or (nz, ny, nx) first dataset.
        images_b: (n2, nz, ny, nx) or (nz, ny, nx) second dataset.
        mask: (nz, ny, nx) boolean mask. None = all voxels.
        method: "pearson", "spearman", or "kendall".
        device: torch device (default: auto-detect).

    Returns:
        (n1, n2) numpy array of correlations.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if device is None:
        device = images_a.device

    a_vecs = _prepare_volumes(images_a, mask, device)
    b_vecs = _prepare_volumes(images_b, mask, device)

    if method == "pearson":
        return _pearson_matrix(a_vecs, b_vecs).detach().cpu().numpy()
    elif method == "spearman":
        return _spearman_matrix(a_vecs, b_vecs).detach().cpu().numpy()
    else:
        return _kendall_matrix(a_vecs, b_vecs)


def one_to_many_correlation(
    reference: Tensor,
    images: Tensor,
    mask: Tensor | None = None,
    method: str = "pearson",
    device: torch.device | None = None,
) -> np.ndarray:
    """Correlate a single reference volume against many volumes.

    Args:
        reference: (nz, ny, nx) single reference volume.
        images: (n, nz, ny, nx) dataset of volumes.
        mask: (nz, ny, nx) boolean mask.
        method: "pearson", "spearman", or "kendall".
        device: torch device.

    Returns:
        (n,) array of correlations.
    """
    if reference.ndim != 3:
        raise ValueError(f"reference must be 3D, got {reference.ndim}D")
    corr_mat = spatial_correlation_matrix(
        reference.unsqueeze(0), images, mask=mask, method=method, device=device
    )
    return corr_mat[0]


# ---------------------------------------------------------------------------
# Matching & consistency
# ---------------------------------------------------------------------------

def optimal_matching(
    corr_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find optimal 1-to-1 volume matching via Hungarian algorithm.

    Maximizes total correlation of matched pairs. When matrices are
    non-square (different number of volumes), matches min(n1, n2) pairs.

    Args:
        corr_matrix: (n1, n2) correlation matrix.

    Returns:
        (row_indices, col_indices, matched_correlations):
            - row_indices: indices into first dataset
            - col_indices: indices into second dataset
            - matched_correlations: correlation of each matched pair
    """
    from scipy.optimize import linear_sum_assignment

    # Hungarian minimizes cost; we want to maximize correlation
    cost = -corr_matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_corrs = corr_matrix[row_ind, col_ind]
    # Sort by correlation (descending) for readability
    order = np.argsort(-matched_corrs)
    return row_ind[order], col_ind[order], matched_corrs[order]


@dataclass
class ConsistencyReport:
    """Summary of cross-correlation consistency between two datasets."""

    n_volumes_a: int
    n_volumes_b: int
    n_matched: int
    method: str
    mean_matched_r: float
    median_matched_r: float
    min_matched_r: float
    max_matched_r: float
    matched_rows: np.ndarray
    matched_cols: np.ndarray
    matched_correlations: np.ndarray
    coverage_at_thresholds: dict[float, float] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"Consistency Report ({self.method})",
            f"  Dataset A: {self.n_volumes_a} volumes",
            f"  Dataset B: {self.n_volumes_b} volumes",
            f"  Matched pairs: {self.n_matched}",
            f"  Mean matched r:   {self.mean_matched_r:.4f}",
            f"  Median matched r: {self.median_matched_r:.4f}",
            f"  Range: [{self.min_matched_r:.4f}, {self.max_matched_r:.4f}]",
            f"  Coverage:",
        ]
        for thresh, pct in sorted(self.coverage_at_thresholds.items()):
            lines.append(f"    r > {thresh:.2f}: {pct:.1f}% of matched pairs")
        return "\n".join(lines)


def consistency_report(
    corr_matrix: np.ndarray,
    method: str = "pearson",
    thresholds: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9, 0.95),
) -> ConsistencyReport:
    """Compute consistency metrics from a cross-correlation matrix.

    Finds the optimal 1-to-1 matching (Hungarian algorithm) and reports
    how well the two datasets agree under that best-case alignment.

    Args:
        corr_matrix: (n1, n2) cross-correlation matrix.
        method: Correlation method used (for labeling).
        thresholds: Report percentage of matched pairs exceeding each threshold.

    Returns:
        ConsistencyReport with matching details and coverage statistics.
    """
    n1, n2 = corr_matrix.shape
    row_ind, col_ind, matched_corrs = optimal_matching(corr_matrix)
    n_matched = len(matched_corrs)

    coverage = {}
    for t in thresholds:
        pct = 100.0 * (matched_corrs > t).sum() / n_matched
        coverage[t] = float(pct)

    return ConsistencyReport(
        n_volumes_a=n1,
        n_volumes_b=n2,
        n_matched=n_matched,
        method=method,
        mean_matched_r=float(matched_corrs.mean()),
        median_matched_r=float(np.median(matched_corrs)),
        min_matched_r=float(matched_corrs.min()),
        max_matched_r=float(matched_corrs.max()),
        matched_rows=row_ind,
        matched_cols=col_ind,
        matched_correlations=matched_corrs,
        coverage_at_thresholds=coverage,
    )
