"""Init-time k-means seeding: per-session clustering must not degenerate at high D."""

from __future__ import annotations

import numpy as np
import torch

from fastfuncstuff.dynamics.bsds.init import init_state


def _simulate_high_d(k=6, d=80, t=200, n_sessions=6, stay=0.95, seed=0):
    """Well-separated states in high-D ROI space (mimics 'lots of ROIs')."""
    rng = np.random.default_rng(seed)
    means = rng.standard_normal((k, d)) * 5.0
    trans = np.full((k, k), (1 - stay) / (k - 1))
    np.fill_diagonal(trans, stay)
    sessions, truth = [], []
    for _ in range(n_sessions):
        z = np.empty(t, dtype=int)
        z[0] = rng.integers(k)
        for i in range(1, t):
            z[i] = rng.choice(k, p=trans[z[i - 1]])
        y = means[z].T + 0.5 * rng.standard_normal((d, t))
        sessions.append(torch.tensor(y, dtype=torch.float64))
        truth.append(z)
    return sessions, truth


def test_per_session_init_populates_all_states_at_high_d():
    k = 6
    sessions, _ = _simulate_high_d(k=k, d=80, n_sessions=6, seed=1)
    lengths = [s.shape[1] for s in sessions]
    y = torch.cat(sessions, dim=1)

    state = init_state(y, lengths, n_states=k, ldim=4, seed=0)
    occ = state.qns.sum(dim=0).numpy()
    # Every requested state should get a meaningful share of initial responsibility —
    # a joint high-D k-means degenerates to 1-2 populated clusters here.
    assert (occ > 0).all(), f"some states got zero initial responsibility: {occ}"
    min_share = occ.min() / occ.sum()
    assert min_share > 0.02, f"initial responsibilities too skewed: {np.round(occ, 1)}"


def test_kmeans_pca_dim_none_matches_raw_space_at_low_d():
    """At low D, projecting onto D PCs is an isometry, so behaviour should be unaffected."""
    k = 3
    sessions, _ = _simulate_high_d(k=k, d=5, n_sessions=3, t=150, seed=2)
    lengths = [s.shape[1] for s in sessions]
    y = torch.cat(sessions, dim=1)

    state_pca = init_state(y, lengths, n_states=k, ldim=2, seed=0, kmeans_pca_dim=20)
    state_raw = init_state(y, lengths, n_states=k, ldim=2, seed=0, kmeans_pca_dim=None)
    occ_pca = state_pca.qns.sum(dim=0).numpy()
    occ_raw = state_raw.qns.sum(dim=0).numpy()
    assert np.allclose(np.sort(occ_pca), np.sort(occ_raw))


def test_kmeans_batched_recovers_clusters_and_matches_loop_quality():
    # The batched k-means (used by init to collapse n_sessions*n_replicates
    # sequential k-means into one launch-cheap batched call) must recover
    # well-separated clusters, and its inertia must be on par with the
    # per-session loop's (kmeans_best_of) — it is not bit-identical (different
    # shared RNG stream) but must be equivalent quality.
    from fastfuncstuff.dynamics.bsds.init import kmeans_batched, kmeans_best_of

    rng = np.random.default_rng(0)
    k, d, n_per = 4, 6, 80
    centers = rng.standard_normal((k, d)) * 6.0
    # Two independent datasets (batch of 2), each k well-separated blobs.
    datasets = []
    truth = []
    for _ in range(2):
        z = rng.integers(0, k, size=k * n_per)
        datasets.append(centers[z] + 0.3 * rng.standard_normal((k * n_per, d)))
        truth.append(z)
    x = torch.tensor(np.stack(datasets), dtype=torch.float64)  # (2, N, D)

    # Actual usage: tile n_replicates along the batch, keep the lowest-inertia
    # replicate per dataset (this is what _per_session_labels does).
    reps = 10
    gen = torch.Generator().manual_seed(0)
    lab, inertia = kmeans_batched(x.repeat(reps, 1, 1), k, generator=gen)  # (reps*2, N)
    lab = lab.view(reps, 2, k * n_per)
    best = inertia.view(reps, 2).argmin(dim=0)  # (2,)
    labels = torch.stack([lab[best[b], b] for b in range(2)])  # (2, N) best replicate

    for b in range(2):
        assert labels[b].unique().numel() == k  # every cluster used
        purity = 0.0
        for c in range(k):
            lab_c = labels[b][truth[b] == c]
            purity += (lab_c == lab_c.mode().values).float().mean().item()
        assert purity / k > 0.98, f"batched k-means impure: {purity / k:.3f}"

        # Inertia on par with the per-session best-of loop.
        gl = torch.Generator().manual_seed(0)
        loop_labels = kmeans_best_of(x[b], k, n_replicates=reps, generator=gl)
        cen = torch.stack([x[b][loop_labels == c].mean(0) for c in range(k)])
        loop_inertia = (torch.cdist(x[b], cen).amin(1) ** 2).sum().item()
        assert inertia.view(reps, 2)[best[b], b].item() < 1.2 * loop_inertia + 1e-6
