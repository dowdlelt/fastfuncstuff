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


def _project_top_pcs_batched(x: torch.Tensor, n_components: int) -> torch.Tensor:
    """Project each ``(N, D)`` slice of ``(B, N, D)`` onto *its own* top PCs (batched SVD)."""
    _, n, d = x.shape
    n_components = min(n_components, n - 1, d)
    if n_components < 1:
        return x
    centered = x - x.mean(dim=1, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)  # vh: (B, min(N,D), D)
    return centered @ vh[:, :n_components].transpose(1, 2)  # (B, N, n_components)


def kmeans_batched(
    x: torch.Tensor,
    k: int,
    *,
    n_iter: int = 25,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched k-means++ Lloyd over ``B`` independent datasets at once.

    ``x`` is ``(B, N, D)``. Runs one k-means per batch element, but everything is
    vectorised over ``B`` — the k-means++ seeding is still sequential over the
    ``k`` centers and Lloyd over ``n_iter`` steps, yet each of those steps is a
    single batched kernel across all ``B`` datasets. So ``n_sessions ×
    n_replicates`` independent k-means cost ~one k-means worth of kernel launches
    instead of ``B`` of them — the launch-bound win (a single small k-means is
    dominated by per-op dispatch, not arithmetic, on GPU). Returns
    ``(labels (B, N), inertia (B,))``.

    Unlike :func:`kmeans`, Lloyd runs a fixed ``n_iter`` with no early stop:
    batch elements converge at different rates, and an all-converged check would
    force a host sync every step, defeating the purpose.
    """
    b, n, d = x.shape
    device = x.device
    k = min(k, n)
    arange_b = torch.arange(b, device=device)
    # k-means++ seeding, batched over B (sequential over the k centers).
    first = torch.randint(n, (b,), generator=generator, device=device)
    centers = x[arange_b, first].unsqueeze(1)  # (B, 1, D)
    for _ in range(1, k):
        d2 = torch.cdist(x, centers).amin(dim=2) ** 2  # (B, N)
        probs = d2 / d2.sum(dim=1, keepdim=True).clamp_min(1e-300)
        nxt = torch.multinomial(probs, 1, generator=generator)  # (B, 1)
        chosen = x.gather(1, nxt.unsqueeze(2).expand(b, 1, d))  # (B, 1, D)
        centers = torch.cat([centers, chosen], dim=1)

    labels = torch.zeros(b, n, dtype=torch.long, device=device)
    for _ in range(n_iter):
        dist = torch.cdist(x, centers)  # (B, N, k)
        labels = dist.argmin(dim=2)  # (B, N)
        # Vectorised centroid update via one-hot matmul (no per-cluster loop):
        # counts and sums fall out of onehot^T, empty clusters keep their center.
        onehot = torch.zeros(b, n, centers.shape[1], dtype=x.dtype, device=device)
        onehot.scatter_(2, labels.unsqueeze(2), 1.0)
        counts = onehot.sum(dim=1).unsqueeze(2)  # (B, k, 1)
        sums = torch.bmm(onehot.transpose(1, 2), x)  # (B, k, D)
        centers = torch.where(counts > 0, sums / counts.clamp_min(1.0), centers)

    dist = torch.cdist(x, centers)
    inertia = (dist.gather(2, labels.unsqueeze(2)).squeeze(2) ** 2).sum(dim=1)  # (B,)
    return labels, inertia


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
    default and avoids the joint-clustering collapse at high ``D``. Sessions of
    equal length are stacked (with ``n_replicates`` copies) into a single
    :func:`kmeans_batched` call and the lowest-inertia replicate is kept per
    session — so the whole init is a handful of batched k-means instead of
    ``n_sessions * n_replicates`` sequential ones, which otherwise dominates GPU
    time (k-means is launch-bound). One RNG stream is shared across the batch, so
    the seeded init differs from the old per-session loop (equivalent quality,
    not bit-identical to fits made before this change).
    """
    device = y.device
    n = y.shape[1]
    labels = torch.zeros(n, dtype=torch.long, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    offsets = [0]
    for ti in session_lengths:
        offsets.append(offsets[-1] + ti)

    groups: dict[int, list[int]] = {}
    for i, ti in enumerate(session_lengths):
        groups.setdefault(ti, []).append(i)

    for ti, idxs in groups.items():
        # (S, Ti, D): the equal-length sessions in this group, transposed to time-major.
        raw = torch.stack([y[:, offsets[i] : offsets[i] + ti].T for i in idxs], dim=0)
        block = _project_top_pcs_batched(raw, pca_dim) if pca_dim is not None else raw
        s = block.shape[0]
        # Tile replicates along the batch: index r*S + s (so .view(R, S, ...) below).
        rep = block.repeat(n_replicates, 1, 1)  # (R*S, Ti, d')
        lab, inertia = kmeans_batched(rep, n_states, generator=gen)
        lab = lab.view(n_replicates, s, ti)
        best = inertia.view(n_replicates, s).argmin(dim=0)  # (S,) lowest-inertia replicate
        for pos, i in enumerate(idxs):
            labels[offsets[i] : offsets[i] + ti] = lab[best[pos], pos]
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
        # Noise precision starts at ones, matching the reference (vbhafa.m:
        # psii=ones(p,1)). A data-variance seed (1/var) is scale-fragile: on
        # un-standardised data with large between-state spread it seeds a tiny
        # precision (huge assumed noise) and systematically steers restart
        # selection into an over-segmented, higher-free-energy basin (an extra
        # state that splits a true one). psii=ones is scale-robust and is
        # ~identical to 1/var on standardised input (var~1) anyway. Scale is
        # instead carried by the mean-loading prior nu_mcl (=1/std(Y)^2 below),
        # exactly as the reference does. See the over-segmentation note in the wiki.
        psii=torch.ones(d, dtype=_DTYPE, device=device),
        a=PA + 0.5 * d,
        b=torch.ones(n_states, ldim, dtype=_DTYPE, device=device),
        mean_mcl=y.mean(dim=1),
        nu_mcl=1.0 / var,
        wa=wa,
        wpi=wpi,
        qns=qns,
    )
