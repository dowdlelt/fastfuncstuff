"""Cross-check an ``ffs_bsds`` fit against the reference MATLAB BSDS output.

The reference ``BayesianSwitchingDynamicalSystems`` saves its ``model`` struct as a
``-v7.3`` ``.mat`` (HDF5). This reads it directly (via ``h5py`` — no MATLAB, no
``mat73``), then Hungarian-matches the two fits **by functional-connectivity
pattern** (state labels are arbitrary between fits) so occupancy and per-state FC
can be compared like-for-like.

Two independent VB fits from different k-means inits land in different local
optima, so the point of this is *structural* agreement — matched states with
similar FC and occupancy — not identical numbers. See ``[[BSDS]]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastfuncstuff.dynamics.bsds.fc_match import (
    fc_similarity_matrix as _fc_similarity_matrix,
)
from fastfuncstuff.dynamics.bsds.fc_match import (
    occupancy_from_viterbi as _occupancy_from_viterbi,
)


@dataclass
class MatlabBSDS:
    """The comparison-relevant fields of a reference MATLAB BSDS ``model``."""

    occupancy: np.ndarray  # (K,) fractional_occupancy_group_wise
    lifetime: np.ndarray  # (K,) mean_lifetime_group_wise (TR)
    state_covs: np.ndarray  # (K, D, D) estimated_covariance (dynamic FC)
    state_means: np.ndarray  # (K, D) estimated_mean
    transition: np.ndarray  # (K, K) state_transition_probabilities (from -> to)
    viterbi_states: list[np.ndarray]  # per run (T_i,) 0-indexed MAP path


def load_matlab_bsds(path: str) -> MatlabBSDS:
    """Load a reference MATLAB BSDS ``-v7.3`` ``model.mat`` into arrays.

    Requires ``h5py``. MATLAB cell arrays are stored as HDF5 object references;
    2-D matrices come out transposed (column- vs row-major), which is handled
    here (symmetric covariances are unaffected; the transition matrix is
    transposed back to ``from -> to``).
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "load_matlab_bsds needs h5py to read the reference -v7.3 .mat (pip install h5py)"
        ) from exc

    with h5py.File(path, "r") as f:
        m = f["model"]

        def deref_stack(name: str) -> list[np.ndarray]:
            refs = np.asarray(m[name]).ravel()
            return [np.asarray(f[r]) for r in refs]

        occ = np.asarray(m["fractional_occupancy_group_wise"]).ravel()
        life = np.asarray(m["mean_lifetime_group_wise"]).ravel()
        # Symmetric FC: transpose is a no-op, but symmetrise for safety.
        covs = np.stack([0.5 * (c + c.T) for c in deref_stack("estimated_covariance")])
        means = np.stack([mn.ravel() for mn in deref_stack("estimated_mean")])  # (K, D)
        trans = np.asarray(m["state_transition_probabilities"]).T  # -> from->to
        viterbi = [
            v.ravel().astype(np.int64) - 1 for v in deref_stack("temporal_evolution_of_states")
        ]

    return MatlabBSDS(
        occupancy=occ,
        lifetime=life,
        state_covs=covs,
        state_means=means,
        transition=trans,
        viterbi_states=viterbi,
    )


@dataclass
class ComparisonResult:
    """Hungarian-matched comparison between an ffs fit and a MATLAB fit."""

    ffs_to_matlab: dict[int, int]  # ffs state -> matched MATLAB state
    fc_similarity: np.ndarray  # (n_pairs,) FC correlation of each matched pair
    ffs_occ: np.ndarray  # (n_pairs,) ffs occupancy, matched order
    matlab_occ: np.ndarray  # (n_pairs,) MATLAB occupancy, matched order
    ffs_state: np.ndarray  # (n_pairs,) ffs state id per pair
    matlab_state: np.ndarray  # (n_pairs,) MATLAB state id per pair
    similarity_matrix: np.ndarray  # (Kffs, Kmat) full FC-similarity matrix
    mean_matched_fc: float  # mean FC similarity over pairs occupied in both
    occupancy_correlation: float  # corr of matched occupancy vectors (both-occupied)


def compare_to_matlab(ffs_model, matlab: MatlabBSDS, *, occ_threshold: float = 1e-3):
    """Hungarian-match an ``ffs_bsds`` model to a :class:`MatlabBSDS` by FC pattern.

    Returns a :class:`ComparisonResult`. The match maximises total FC similarity
    over the (arbitrary) state labels; summary metrics (``mean_matched_fc``,
    ``occupancy_correlation``) are computed only over pairs that are occupied in
    **both** fits, since empty states have degenerate FC.
    """
    import torch
    from scipy.optimize import linear_sum_assignment

    ffs_covs = ffs_model.state_covs
    ffs_covs = (
        ffs_covs.detach().cpu().numpy()
        if isinstance(ffs_covs, torch.Tensor)
        else np.asarray(ffs_covs)
    )
    sim = _fc_similarity_matrix(ffs_covs, matlab.state_covs)  # (Kffs, Kmat)
    cost = -np.nan_to_num(sim, nan=-1.0)
    row_ind, col_ind = linear_sum_assignment(cost)

    ffs_occ_all = _occupancy_from_viterbi(ffs_model.viterbi_states, ffs_covs.shape[0])
    mat_occ_all = matlab.occupancy

    ffs_occ = ffs_occ_all[row_ind]
    matlab_occ = mat_occ_all[col_ind]
    fc_sim = sim[row_ind, col_ind]

    both = (ffs_occ > occ_threshold) & (matlab_occ > occ_threshold)
    mean_fc = float(np.nanmean(fc_sim[both])) if both.any() else float("nan")
    occ_corr = (
        float(np.corrcoef(ffs_occ[both], matlab_occ[both])[0, 1])
        if both.sum() > 1
        else float("nan")
    )

    return ComparisonResult(
        ffs_to_matlab={int(i): int(j) for i, j in zip(row_ind, col_ind, strict=True)},
        fc_similarity=fc_sim,
        ffs_occ=ffs_occ,
        matlab_occ=matlab_occ,
        ffs_state=row_ind,
        matlab_state=col_ind,
        similarity_matrix=sim,
        mean_matched_fc=mean_fc,
        occupancy_correlation=occ_corr,
    )


def plot_comparison(
    result: ComparisonResult, path: str | None = None, *, occ_threshold: float = 1e-3
):
    """Side-by-side occupancy + matched-FC-similarity + the assignment heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    # Restrict to pairs occupied in either fit, sorted by ffs occupancy.
    keep = (result.ffs_occ > occ_threshold) | (result.matlab_occ > occ_threshold)
    order = np.argsort(-result.ffs_occ[keep])
    fo = result.ffs_occ[keep][order]
    mo = result.matlab_occ[keep][order]
    fc = result.fc_similarity[keep][order]
    labels = [f"ffs{result.ffs_state[keep][i]}\nmat{result.matlab_state[keep][i]}" for i in order]

    fig = plt.figure(figsize=(14, 4.5), facecolor="#fcfcfb")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.2, 1], wspace=0.3)

    ax0 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(fo))
    ax0.bar(x - 0.2, fo, 0.4, label="ffs", color="#2a78d6")
    ax0.bar(x + 0.2, mo, 0.4, label="MATLAB", color="#e34948")
    ax0.set_xticks(x, labels, fontsize=6, rotation=90)
    ax0.set_ylabel("fractional occupancy")
    ax0.set_title(f"Matched occupancy (r={result.occupancy_correlation:.2f})")
    ax0.legend(frameon=False)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar(x, fc, color="#1baf7a")
    ax1.axhline(result.mean_matched_fc, color="#52514e", ls="--", lw=1)
    ax1.set_xticks(x, labels, fontsize=6, rotation=90)
    ax1.set_ylim(-1, 1)
    ax1.set_ylabel("FC correlation")
    ax1.set_title(f"Matched-state FC similarity (mean={result.mean_matched_fc:.2f})")

    ax2 = fig.add_subplot(gs[0, 2])
    im = ax2.imshow(result.similarity_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax2.scatter(result.matlab_state, result.ffs_state, s=10, c="k", marker="x")
    ax2.set_xlabel("MATLAB state")
    ax2.set_ylabel("ffs state")
    ax2.set_title("FC similarity + assignment")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle("ffs_bsds vs reference MATLAB — state match", fontsize=13)
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
        plt.close(fig)
        return path
    return fig
