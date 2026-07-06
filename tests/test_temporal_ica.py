"""Temporal ICA (two-stage) correctness tests.

Synthetic ground truth: K temporally-independent sources with sparse spatial
blobs, observed across several runs. The two-stage pipeline (spatial reduction →
dual regression → temporal ICA) must recover the temporal sources up to sign and
permutation. These tests would catch an orientation/transpose bug in the temporal
stage — the single most likely defect in this code.
"""

from __future__ import annotations

import numpy as np
import torch

from fastfuncstuff.decomposition.temporal import (
    group_spatial_ica,
    spatial_regression,
    temporal_ica,
)

CPU = torch.device("cpu")


def _independent_sources(k: int, t: int, rng: np.random.RandomState) -> np.ndarray:
    """K temporally-independent, non-Gaussian, zero-mean unit-var sources (K, T).

    All are iid stochastic draws from distinct non-Gaussian distributions
    (independent by construction, distinct skew/kurtosis), which ICA can
    cleanly separate. Deterministic periodic sources are avoided — they
    autocorrelate and are only weakly separable.
    """
    raw = [
        rng.laplace(size=t),  # super-Gaussian, symmetric
        rng.uniform(-1.0, 1.0, size=t),  # sub-Gaussian, symmetric
        rng.exponential(size=t),  # skewed
        rng.standard_t(df=3, size=t),  # heavy-tailed
        rng.rayleigh(size=t),  # skewed, positive
    ]
    s = np.stack(raw[:k], axis=0).astype(np.float64)
    s -= s.mean(axis=1, keepdims=True)
    s /= s.std(axis=1, keepdims=True)
    return s


def _sparse_maps(v: int, k: int, rng: np.random.RandomState) -> np.ndarray:
    """Sparse, non-overlapping positive spatial blobs (V, K) — super-Gaussian."""
    a = 0.01 * rng.randn(v, k)
    block = v // (k + 2)
    for j in range(k):
        lo = j * block
        a[lo : lo + block, j] += 1.0 + 0.1 * rng.randn(block)
    return a.astype(np.float32)


def _best_match_abs_corr(recovered: np.ndarray, truth: np.ndarray) -> float:
    """Greedy 1-1 match of recovered↔truth rows by |corr|; return mean best |corr|."""
    r = recovered - recovered.mean(axis=1, keepdims=True)
    g = truth - truth.mean(axis=1, keepdims=True)
    r /= np.linalg.norm(r, axis=1, keepdims=True) + 1e-12
    g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-12
    corr = np.abs(r @ g.T)  # (n_rec, n_truth)
    used_cols: set[int] = set()
    scores = []
    for i in range(corr.shape[0]):
        order = np.argsort(-corr[i])
        for c in order:
            if c not in used_cols:
                used_cols.add(int(c))
                scores.append(corr[i, c])
                break
    return float(np.mean(scores))


def _make_dataset(seed: int = 0):
    rng = np.random.RandomState(seed)
    v, k_true, n_runs, t_run = 600, 5, 4, 300
    sources = _independent_sources(k_true, n_runs * t_run, rng)  # (K, T_total)
    maps = _sparse_maps(v, k_true, rng)  # (V, K)
    run_lengths = [t_run] * n_runs
    data_tv = sources.T @ maps.T  # (T_total, V)
    data_tv += 0.02 * rng.randn(*data_tv.shape)  # low noise
    return {
        "sources": sources,
        "maps": maps,
        "run_lengths": run_lengths,
        "data_tv": torch.tensor(data_tv, dtype=torch.float32, device=CPU),
        "k_true": k_true,
        "v": v,
    }


def test_spatial_regression_recovers_timecourses():
    """Dual-regression stage 1 recovers the true loadings from data + maps."""
    rng = np.random.RandomState(1)
    v, k, t = 500, 4, 200
    maps = _sparse_maps(v, k, rng)  # (V, K)
    tc_true = _independent_sources(k, t, rng)  # (K, T)
    data_vt = maps @ tc_true  # (V, T)
    gm = torch.tensor(maps.T, dtype=torch.float32)  # (K, V)
    d = torch.tensor(data_vt, dtype=torch.float32)  # (V, T)
    tc = spatial_regression(gm, d).cpu().numpy().T  # (K, T)
    assert _best_match_abs_corr(tc, tc_true) > 0.99


def test_temporal_ica_end_to_end_pca():
    """Full pipeline with the deterministic PCA reducer recovers the sources."""
    from fastfuncstuff.decomposition.migp import _reduce_to_topk

    ds = _make_dataset(seed=2)
    # Rank-matched reduction: the sICA basis spans exactly the signal
    # subspace (in real data sICA components are meaningful, not pure noise).
    k_sica = ds["k_true"]
    group_maps = _reduce_to_topk(ds["data_tv"], k_sica)  # (K_sica, V)

    # Dual regression per run (data split by run_lengths along time).
    off, blocks = 0, []
    for length in ds["run_lengths"]:
        run_tv = ds["data_tv"][off : off + length]  # (T_i, V)
        blocks.append(spatial_regression(group_maps, run_tv.T))  # (T_i, K_sica)
        off += length
    concat_tcs = torch.cat(blocks, dim=0)  # (T_total, K_sica)

    result = temporal_ica(
        concat_tcs,
        group_maps,
        n_components=ds["k_true"],
        run_lengths=ds["run_lengths"],
        method="fastica",
        seed=0,
        device=CPU,
    )
    assert result.temporal_sources.shape == (ds["k_true"], sum(ds["run_lengths"]))
    assert result.spatial_maps.shape == (ds["k_true"], ds["v"])
    # Per-run split lengths match the run structure.
    assert [b.shape[1] for b in result.per_run_sources] == ds["run_lengths"]
    score = _best_match_abs_corr(result.temporal_sources, ds["sources"])
    assert score > 0.9, f"temporal source recovery too low: {score:.3f}"


def test_temporal_ica_icasso_stability():
    """ICASSO path recovers sources and reports a high, well-formed Iq."""
    from fastfuncstuff.decomposition.migp import _reduce_to_topk

    ds = _make_dataset(seed=5)
    k_sica = ds["k_true"]
    group_maps = _reduce_to_topk(ds["data_tv"], k_sica)
    off, blocks = 0, []
    for length in ds["run_lengths"]:
        blocks.append(spatial_regression(group_maps, ds["data_tv"][off : off + length].T))
        off += length
    concat_tcs = torch.cat(blocks, dim=0)

    result = temporal_ica(
        concat_tcs,
        group_maps,
        n_components=ds["k_true"],
        run_lengths=ds["run_lengths"],
        icasso_runs=15,
        seed=0,
        device=CPU,
    )
    assert result.stability is not None
    assert result.stability.shape == (ds["k_true"],)
    # Rank-matched, clean sources → every component reproducible.
    assert result.stability.min() > 0.8
    assert result.diagnostics["n_stable_iq0.5"] == ds["k_true"]
    assert _best_match_abs_corr(result.temporal_sources, ds["sources"]) > 0.9

    # subset() drops components and keeps arrays consistent.
    keep = result.stability > 0.5
    sub = result.subset(keep)
    assert sub.temporal_sources.shape[0] == int(keep.sum())
    assert sub.spatial_maps.shape[0] == int(keep.sum())
    assert sub.mixing.shape[1] == int(keep.sum())
    assert all(b.shape[0] == int(keep.sum()) for b in sub.per_run_sources)


def test_temporal_ica_sica_reducer():
    """Same recovery via the group spatial ICA reducer (HCP-faithful path)."""
    ds = _make_dataset(seed=3)
    k_sica = ds["k_true"]
    group_maps, _ = group_spatial_ica(
        ds["data_tv"], n_components=k_sica, method="fastica", seed=0, device=CPU
    )
    off, blocks = 0, []
    for length in ds["run_lengths"]:
        run_tv = ds["data_tv"][off : off + length]
        blocks.append(spatial_regression(group_maps, run_tv.T))
        off += length
    concat_tcs = torch.cat(blocks, dim=0)

    result = temporal_ica(
        concat_tcs,
        group_maps,
        n_components=ds["k_true"],
        run_lengths=ds["run_lengths"],
        method="fastica",
        seed=0,
        device=CPU,
    )
    score = _best_match_abs_corr(result.temporal_sources, ds["sources"])
    assert score > 0.9, f"temporal source recovery too low: {score:.3f}"
