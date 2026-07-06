"""Initialisation for a BSDS fit: k-means seeding of the variational posterior.

The reference (``initPoteriors.m``, default ``method_kmeans='subject'``) seeds
responsibilities with k-means run *independently within each subject/session*
(10 replicates each, kept by inertia), then pools the labels — it never
clusters the pooled multi-session data jointly. Cluster labels are **not**
aligned across sessions at this stage (session A's "state 3" has no relation
to session B's "state 3"); that alignment is not needed at init because every
session shares the same group-level loading matrices, so the shared state
identity emerges from the VB updates, not from the label bookkeeping at
init. For a single densely-sampled subject, each run/session is the natural
analogue of a "subject" cell here.

Clustering jointly across sessions in raw ROI space instead (as an earlier
version of this port did) degenerates badly once ``D`` (ROI count) is more
than a couple dozen: Euclidean distance concentrates in high dimensions, so a
joint k-means with many requested states collapses almost everything into one
or two clusters, starving most states of any initial responsibility mass —
and since every random restart sees the same degenerate joint geometry, more
restarts don't fix it. Two mitigations, both applied here:

1. Per-session k-means (matching the reference default), so each session
   contributes its own locally-diverse partition instead of all sessions
   competing for the same handful of globally-separable clusters.
2. Cluster on each session's leading PCs rather than raw ROI space when ``D``
   is large (not in the reference, which only ever ran at ``D`` ~10-30) — a
   standard curse-of-dimensionality mitigation for k-means.
"""

from __future__ import annotations

import torch

from fastfuncstuff.dynamics.bsds.vb import ALPHA_A, ALPHA_PI, PA, VBState

_DTYPE = torch.float64


def kmeans(
    x: torch.Tensor,
    k: int,
    *,
    n_iter: int = 25,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """k-means++ seeded Lloyd clustering. ``x`` is ``(N, D)``; returns ``(labels, centers)``."""
    n = x.shape[0]
    device = x.device
    # k-means++ seeding.
    first = int(torch.randint(n, (1,), generator=generator, device=device))
    centers = x[first : first + 1].clone()
    for _ in range(1, k):
        d2 = torch.cdist(x, centers).amin(dim=1) ** 2
        probs = d2 / d2.sum().clamp_min(1e-300)
        nxt = int(torch.multinomial(probs, 1, generator=generator))
        centers = torch.cat([centers, x[nxt : nxt + 1]], dim=0)

    labels = torch.zeros(n, dtype=torch.long, device=device)
    for _ in range(n_iter):
        labels = torch.cdist(x, centers).argmin(dim=1)
        # Vectorised centroid update: per-cluster sum via index_add, divide by
        # count; empty clusters keep their old center (matches the masked mean).
        sums = torch.zeros_like(centers).index_add_(0, labels, x)  # (k, D)
        counts = torch.bincount(labels, minlength=k).to(x.dtype).unsqueeze(1)  # (k, 1)
        new = torch.where(counts > 0, sums / counts.clamp_min(1.0), centers)
        if torch.allclose(new, centers):
            centers = new
            break
        centers = new
    return labels, centers


def _kmeans_inertia(x: torch.Tensor, labels: torch.Tensor, centers: torch.Tensor) -> float:
    """Sum of squared distances from each point to its assigned center."""
    return float((torch.cdist(x, centers).gather(1, labels.unsqueeze(1)) ** 2).sum())


def kmeans_best_of(
    x: torch.Tensor,
    k: int,
    *,
    n_replicates: int = 10,
    n_iter: int = 25,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Run k-means ``n_replicates`` times and keep the lowest-inertia labelling.

    Mirrors MATLAB's ``kmeans(..., 'Replicates', 10)`` used by the reference's
    ``initPoteriors``. Returns only ``labels`` — centers computed from a single
    session's k-means are not meaningful once pooled across sessions (see
    module docstring), so callers should recompute state means from the pooled
    labelled data instead.
    """
    n = x.shape[0]
    k_eff = min(k, n)
    best_labels: torch.Tensor | None = None
    best_inertia = float("inf")
    for _ in range(n_replicates):
        labels, centers = kmeans(x, k_eff, n_iter=n_iter, generator=generator)
        inertia = _kmeans_inertia(x, labels, centers)
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels
    assert best_labels is not None
    return best_labels


def _project_top_pcs(x: torch.Tensor, n_components: int) -> torch.Tensor:
    """Project ``(N, D)`` onto its top ``n_components`` principal components."""
    n_components = min(n_components, x.shape[0] - 1, x.shape[1])
    if n_components < 1:
        return x
    centered = x - x.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:n_components].T


def _transition_counts(
    labels: torch.Tensor, session_lengths: list[int], k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Within-session transition counts ``(K, K)`` and state occupancy ``(K,)``."""
    occ = torch.bincount(labels, minlength=k).to(_DTYPE)  # (K,)
    # Within-session transitions only: bincount the flattened (from*K + to) pairs,
    # excluding the pair that would straddle each session boundary.
    pair_counts = torch.zeros(k * k, dtype=_DTYPE, device=labels.device)
    col = 0
    for ti in session_lengths:
        if ti >= 2:
            seg = labels[col : col + ti]
            flat = seg[:-1] * k + seg[1:]
            pair_counts += torch.bincount(flat, minlength=k * k).to(_DTYPE)
        col += ti
    counter = pair_counts.view(k, k)
    return counter, occ


def _per_session_labels(
    y: torch.Tensor,
    session_lengths: list[int],
    n_states: int,
    *,
    n_replicates: int,
    pca_dim: int | None,
    seed: int,
) -> torch.Tensor:
    """K-means labels, clustered independently within each session then pooled.

    See the module docstring: this matches the reference's per-subject k-means
    default and avoids the joint-clustering collapse at high ``D``. Each
    session gets its own generator (derived from ``seed``) so results are
    reproducible but sessions don't share k-means++ seeding draws.
    """
    device = y.device
    n = y.shape[1]
    labels = torch.zeros(n, dtype=torch.long, device=device)
    col = 0
    for i, ti in enumerate(session_lengths):
        block = y[:, col : col + ti].T  # (Ti, D)
        if pca_dim is not None:
            block = _project_top_pcs(block, pca_dim)
        gen = torch.Generator(device=device)
        gen.manual_seed(seed + i)
        labels[col : col + ti] = kmeans_best_of(
            block, n_states, n_replicates=n_replicates, generator=gen
        )
        col += ti
    return labels


def _label_means(y: torch.Tensor, labels: torch.Tensor, n_states: int) -> torch.Tensor:
    """Per-label mean over all pooled points (``(K, D)``); global mean if a label is empty."""
    global_mean = y.mean(dim=1)
    centers = global_mean.unsqueeze(0).repeat(n_states, 1)
    for j in range(n_states):
        mask = labels == j
        if mask.any():
            centers[j] = y[:, mask].mean(dim=1)
    return centers


def init_state(
    y: torch.Tensor,
    session_lengths: list[int],
    n_states: int,
    ldim: int,
    *,
    seed: int = 0,
    device: torch.device | None = None,
    n_kmeans_replicates: int = 10,
    kmeans_pca_dim: int | None = 20,
) -> VBState:
    """Build a k-means-seeded :class:`VBState` for a group-level fit.

    ``y`` is the ``(D, N)`` concatenation of all preprocessed sessions.
    Responsibilities are seeded by clustering *within each session*
    independently (``n_kmeans_replicates`` restarts each, kept by inertia) and
    pooling the labels — matching the reference default (see module
    docstring) rather than one joint k-means over all sessions, which
    degenerates once ``D`` is more than a couple dozen. Clustering runs on
    each session's top ``kmeans_pca_dim`` principal components (``None``
    disables the projection and clusters in raw ROI space).
    """
    device = y.device if device is None else device
    y = y.to(_DTYPE)
    d, n = y.shape
    kt = ldim + 1
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    labels = _per_session_labels(
        y,
        session_lengths,
        n_states,
        n_replicates=n_kmeans_replicates,
        pca_dim=kmeans_pca_dim,
        seed=seed,
    )
    qns = torch.zeros(n, n_states, dtype=_DTYPE, device=device)
    qns[torch.arange(n), labels] = 1.0

    counter, occ = _transition_counts(labels, session_lengths, n_states)
    wa = counter + ALPHA_A / n_states
    wpi = occ + ALPHA_PI / n_states

    var = y.var(dim=1, unbiased=False).clamp_min(1e-6)  # (D,)
    lm = 0.1 * torch.randn(n_states, d, kt, generator=gen, dtype=_DTYPE, device=device)
    lm[:, :, 0] = _label_means(y, labels, n_states)  # column 0 is the state mean
    lcov = torch.eye(kt, dtype=_DTYPE, device=device).expand(n_states, d, kt, kt).clone()
    xm = torch.ones(n_states, kt, n, dtype=_DTYPE, device=device)
    xm[:, 1:, :] = 0.0
    xcov = torch.zeros(n_states, kt, kt, dtype=_DTYPE, device=device)

    return VBState(
        n_states=n_states,
        n_roi=d,
        ldim=ldim,
        n_time=n,
        session_lengths=list(session_lengths),
        lm=lm,
        lcov=lcov,
        xm=xm,
        xcov=xcov,
        psii=1.0 / var,
        a=PA + 0.5 * d,
        b=torch.ones(n_states, ldim, dtype=_DTYPE, device=device),
        mean_mcl=y.mean(dim=1),
        nu_mcl=1.0 / var,
        wa=wa,
        wpi=wpi,
        qns=qns,
    )
