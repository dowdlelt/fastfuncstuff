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


def _to_np(x):
    import torch

    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _cohen_kappa(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Chance-corrected agreement between two integer labellings over ``k`` classes."""
    conf = np.zeros((k, k))
    np.add.at(conf, (a, b), 1)
    n = conf.sum()
    if n == 0:
        return float("nan")
    po = np.trace(conf) / n
    pe = (conf.sum(0) * conf.sum(1)).sum() / n**2
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def _temporal_agreement(ffs_viterbi, mat_viterbi, mat2ffs: np.ndarray, k: int):
    """Frame-wise MAP agreement after relabelling MATLAB states into ffs labels.

    Both fits decoded the *same* runs, so once states are matched the two Viterbi
    paths are directly comparable frame by frame — a far stronger check than
    aggregate occupancy. Runs are compared on their common leading length: the
    MATLAB export length-equalises (truncates to the global min), so ffs runs are
    often a frame or two longer; the paths still align from the start, and the
    extra tail is simply dropped. Returns ``(accuracy, kappa, n_frames)``;
    ``nan`` only if the two fits have a different *number* of runs (unalignable).
    """
    if len(ffs_viterbi) != len(mat_viterbi):
        return float("nan"), float("nan"), 0
    ffs_all, mat_all = [], []
    for fv, mv in zip(ffs_viterbi, mat_viterbi, strict=False):
        fv = _to_np(fv).astype(np.int64)
        mv = np.asarray(mv, dtype=np.int64)
        t = min(fv.shape[0], mv.shape[0])  # align on the common leading frames
        remapped = mat2ffs[mv[:t]]  # MATLAB label -> matched ffs label
        ok = remapped >= 0
        ffs_all.append(fv[:t][ok])
        mat_all.append(remapped[ok])
    fa = np.concatenate(ffs_all)
    ma = np.concatenate(mat_all)
    if fa.size == 0:
        return float("nan"), float("nan"), 0
    return float((fa == ma).mean()), _cohen_kappa(fa, ma, k), int(fa.size)


def _matched_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two flattened vectors, nan-safe."""
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


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
    # Extra structural checks over both-occupied matched states:
    n_occupied_ffs: int
    n_occupied_matlab: int
    transition_correlation: float  # matched transition submatrices, flattened
    lifetime_correlation: float  # matched mean-lifetimes
    activation_correlation: float  # mean per-state corr of activation profiles
    temporal_agreement: float  # frame-wise MAP agreement (same runs), or nan
    temporal_kappa: float  # chance-corrected version of the above
    temporal_frames: int  # frames the temporal metrics used (0 if unalignable)


def compare_to_matlab(ffs_model, matlab: MatlabBSDS, *, occ_threshold: float = 1e-3):
    """Hungarian-match an ``ffs_bsds`` model to a :class:`MatlabBSDS` by FC pattern.

    Returns a :class:`ComparisonResult`. States are matched by FC pattern (labels
    are arbitrary), then compared on several axes the reference also produces —
    occupancy, per-state FC, the transition matrix, mean lifetime, activation
    profiles, and (the strongest, since both fits decoded the same runs) the
    frame-by-frame MAP state sequence. Aggregate metrics use only pairs occupied
    in **both** fits, since empty states have degenerate parameters.
    """
    from scipy.optimize import linear_sum_assignment

    ffs_covs = _to_np(ffs_model.state_covs)
    k_ffs = ffs_covs.shape[0]
    sim = _fc_similarity_matrix(ffs_covs, matlab.state_covs)  # (Kffs, Kmat)
    cost = -np.nan_to_num(sim, nan=-1.0)
    row_ind, col_ind = linear_sum_assignment(cost)

    ffs_occ_all = _occupancy_from_viterbi(ffs_model.viterbi_states, k_ffs)
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

    # State ids (in each fit's own labelling) for the both-occupied matched pairs.
    ffs_idx = row_ind[both]
    mat_idx = col_ind[both]

    # Transition matrix: compare the matched submatrices (from->to over shared states).
    ffs_trans = _to_np(ffs_model.transition)
    trans_corr = _matched_corr(
        ffs_trans[np.ix_(ffs_idx, ffs_idx)], matlab.transition[np.ix_(mat_idx, mat_idx)]
    )

    # Mean lifetime (ffs computed from the MAP paths, TR units, matching MATLAB).
    from fastfuncstuff.dynamics.states import mean_lifetime

    pooled = np.concatenate([_to_np(v).astype(np.int64) for v in ffs_model.viterbi_states])
    ffs_life_all = mean_lifetime(pooled, k_ffs, tr=1.0)
    life_corr = _matched_corr(ffs_life_all[ffs_idx], matlab.lifetime[mat_idx])

    # Activation profiles: mean per-state correlation of the D-length mean vectors.
    ffs_means = _to_np(ffs_model.state_means)  # (Kffs, D)
    act = [
        _matched_corr(ffs_means[i], matlab.state_means[j])
        for i, j in zip(ffs_idx, mat_idx, strict=True)
    ]
    act_corr = float(np.nanmean(act)) if act else float("nan")

    # Temporal MAP agreement (same runs): relabel MATLAB Viterbi into ffs labels.
    mat2ffs = np.full(matlab.state_covs.shape[0], -1, dtype=np.int64)
    mat2ffs[col_ind] = row_ind
    temp_acc, temp_kappa, temp_frames = _temporal_agreement(
        ffs_model.viterbi_states, matlab.viterbi_states, mat2ffs, k_ffs
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
        n_occupied_ffs=int((ffs_occ_all > occ_threshold).sum()),
        n_occupied_matlab=int((mat_occ_all > occ_threshold).sum()),
        transition_correlation=trans_corr,
        lifetime_correlation=life_corr,
        activation_correlation=act_corr,
        temporal_agreement=temp_acc,
        temporal_kappa=temp_kappa,
        temporal_frames=temp_frames,
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

    fig = plt.figure(figsize=(17, 4.5), facecolor="#fcfcfb")
    gs = fig.add_gridspec(1, 4, width_ratios=[1.5, 1.2, 1, 0.9], wspace=0.32)

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

    # Structural-agreement scorecard.
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.axis("off")
    temporal = (
        "n/a (run count differs)"
        if not np.isfinite(result.temporal_agreement)
        else f"{result.temporal_agreement:.2f}  (κ={result.temporal_kappa:.2f})"
    )
    rows = [
        ("occupied states", f"{result.n_occupied_ffs} vs {result.n_occupied_matlab}"),
        ("mean matched FC", f"{result.mean_matched_fc:.3f}"),
        ("occupancy r", f"{result.occupancy_correlation:.3f}"),
        ("transition r", f"{result.transition_correlation:.3f}"),
        ("lifetime r", f"{result.lifetime_correlation:.3f}"),
        ("activation r", f"{result.activation_correlation:.3f}"),
        ("MAP agreement", temporal),
    ]
    ax3.set_title("Structural agreement", fontsize=11)
    y = 0.92
    for name, val in rows:
        ax3.text(0.02, y, name, fontsize=9, va="top", color="#52514e")
        ax3.text(0.98, y, val, fontsize=9, va="top", ha="right", color="#0b0b0b")
        y -= 0.13

    fig.suptitle("ffs_bsds vs reference MATLAB — state match", fontsize=13)
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
        plt.close(fig)
        return path
    return fig
