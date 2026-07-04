"""Initialisation for a BSDS fit: k-means seeding of the variational posterior.

The reference seeds responsibilities with k-means (10 replicates) and random
loadings. We do the same, but with a dependency-free torch k-means (k-means++
seeding + Lloyd iterations) so the core BSDS path needs only torch/numpy.
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
        new = centers.clone()
        for j in range(k):
            mask = labels == j
            if mask.any():
                new[j] = x[mask].mean(dim=0)
        if torch.allclose(new, centers):
            centers = new
            break
        centers = new
    return labels, centers


def _transition_counts(
    labels: torch.Tensor, session_lengths: list[int], k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Within-session transition counts ``(K, K)`` and state occupancy ``(K,)``."""
    counter = torch.zeros(k, k, dtype=_DTYPE, device=labels.device)
    occ = torch.zeros(k, dtype=_DTYPE, device=labels.device)
    col = 0
    for ti in session_lengths:
        seg = labels[col : col + ti]
        for c in seg:
            occ[int(c)] += 1
        for a, b in zip(seg[:-1], seg[1:], strict=True):
            counter[int(a), int(b)] += 1
        col += ti
    return counter, occ


def init_state(
    y: torch.Tensor,
    session_lengths: list[int],
    n_states: int,
    ldim: int,
    *,
    seed: int = 0,
    device: torch.device | None = None,
) -> VBState:
    """Build a k-means-seeded :class:`VBState` for a group-level fit.

    ``y`` is the ``(D, N)`` concatenation of all preprocessed sessions.
    """
    device = y.device if device is None else device
    y = y.to(_DTYPE)
    d, n = y.shape
    kt = ldim + 1
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    labels, centers = kmeans(y.T, n_states, generator=gen)  # centers (K, D)
    qns = torch.zeros(n, n_states, dtype=_DTYPE, device=device)
    qns[torch.arange(n), labels] = 1.0

    counter, occ = _transition_counts(labels, session_lengths, n_states)
    wa = counter + ALPHA_A / n_states
    wpi = occ + ALPHA_PI / n_states

    var = y.var(dim=1, unbiased=False).clamp_min(1e-6)  # (D,)
    lm = 0.1 * torch.randn(n_states, d, kt, generator=gen, dtype=_DTYPE, device=device)
    lm[:, :, 0] = centers  # column 0 is the state mean
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
