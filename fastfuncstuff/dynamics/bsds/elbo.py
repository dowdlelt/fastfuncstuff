"""Variational lower bound (free energy) for BSDS.

This is the ``F`` of the reference ``computeLowerBound`` — the free energy of the
factor-analysis observation model (expected data likelihood, latent and loading
KLs) plus the Dirichlet HMM-parameter KL terms, with the state responsibilities
``q(z)`` held at their E-step values. It is the objective used for convergence
and restart selection (the reference selects restarts on this same ``F``), and it
is non-decreasing across VB iterations — the tests assert that.

Note we deliberately do *not* add the forward-backward ``loglik`` to ``F``: the
per-state emission log-densities already carry the data term, so ``F + loglik``
would double-count it and is not monotone. See :mod:`.model`.
"""

from __future__ import annotations

import torch

from fastfuncstuff.dynamics.bsds.vb import ALPHA_A, ALPHA_PI, PA, PB, VBState

_DTYPE = torch.float64


def kl_gamma(p_a: torch.Tensor, p_b: torch.Tensor, q_a: float, q_b: float) -> torch.Tensor:
    """KL(P||Q) between Gamma(shape p_a, rate p_b) and Gamma(q_a, q_b), summed over p_b."""
    p_a = torch.as_tensor(p_a, dtype=_DTYPE, device=p_b.device)
    return (
        p_a * torch.log(p_b)
        - torch.lgamma(p_a)
        - q_a * torch.log(torch.tensor(q_b, dtype=_DTYPE, device=p_b.device))
        + torch.lgamma(torch.tensor(q_a, dtype=_DTYPE, device=p_b.device))
        + (p_a - q_a) * (torch.digamma(p_a) - torch.log(p_b))
        - (p_b - q_b) * p_a / p_b
    ).sum()


def kl_dirichlet(vec_p: torch.Tensor, vec_q: torch.Tensor) -> torch.Tensor:
    """KL(P||Q) between Dirichlet(vec_p) and Dirichlet(vec_q) (unnormalised params)."""
    alpha_p = vec_p.sum()
    alpha_q = vec_q.sum()
    return (
        torch.lgamma(alpha_p)
        - torch.lgamma(alpha_q)
        - (torch.lgamma(vec_p) - torch.lgamma(vec_q)).sum()
        + ((vec_p - vec_q) * (torch.digamma(vec_p) - torch.digamma(alpha_p))).sum()
    )


def lower_bound(state: VBState, y: torch.Tensor) -> torch.Tensor:
    """Free-energy ``F`` (excludes the HMM data log-likelihood)."""
    s = state.n_states
    p = state.n_roi
    kt = state.kt
    psii = state.psii
    dig_a = torch.digamma(torch.tensor(state.a, dtype=_DTYPE, device=y.device))
    log_det_psi = torch.log(psii).sum()
    two_pi = torch.log(torch.tensor(2 * torch.pi, dtype=_DTYPE, device=y.device))

    qns = state.qns  # (N, K)
    neff = qns.sum(dim=0)  # (K,)
    temp_alt = state.xm * qns.T.unsqueeze(1)  # (K, kt, N)
    xcor = state.xcov * neff.view(-1, 1, 1) + torch.einsum(
        "kin,kjn->kij", state.xm, temp_alt
    )  # (K, kt, kt)

    # Row 3: latent entropy / prior cross term (batched over states).
    xcov_lat = state.xcov[:, 1:, 1:]  # (K, ldim, ldim)
    chol = torch.linalg.cholesky(xcov_lat)  # (K, ldim, ldim)
    logdet_half = torch.log(torch.diagonal(chol, dim1=1, dim2=2)).sum(dim=1)  # (K,)
    f3 = (
        0.5 * neff * (kt - 1)
        - 0.5 * torch.diagonal(xcor[:, 1:, 1:], dim1=1, dim2=2).sum(dim=1)
        + neff * logdet_half
    )  # (K,)

    # Row 4: expected log-likelihood of the observations (batched over states).
    lm_psii_lm = torch.einsum("kqi,q,kqj->kij", state.lm, psii, state.lm)  # (K, kt, kt)
    sum_psii_lcov = torch.einsum("q,kqij->kij", psii, state.lcov)  # (K, kt, kt)
    # data_quad[k] = sum_{d,n} psii_d Qns[n,k] y_dn (y_dn - 2 (Lm_k Xm_k)_dn).
    # Contract D before N so the peak intermediate is (K, kt, N), never (K, D, N).
    g = torch.einsum("q,qn->n", psii, y * y)  # (N,)
    data_first = qns.T @ g  # (K,)
    q_ = torch.einsum("q,qn,kqi->kin", psii, y, state.lm)  # (K, kt, N)
    data_second = 2.0 * torch.einsum("nk,kin,kin->k", qns, state.xm, q_)  # (K,)
    data_quad = data_first - data_second  # (K,)
    xcor_t = xcor.transpose(1, 2)  # (K, kt, kt)
    f4 = (
        -0.5 * neff * (-log_det_psi + p * two_pi)
        - 0.5 * (lm_psii_lm * xcor_t).sum(dim=(1, 2))
        - 0.5 * (sum_psii_lcov * xcor_t).sum(dim=(1, 2))
        - 0.5 * data_quad
    )  # (K,)

    # Row 6: -KL of the loading-precision ARD Gamma posterior (per state).
    f6 = -torch.stack([kl_gamma(state.a, state.b[t], PA, PB) for t in range(s)])  # (K,)

    # Row 1: loading posterior entropy + ARD / mean-hyperprior cross terms.
    priorln = torch.log(state.nu_mcl).sum() + p * (dig_a - torch.log(state.b)).sum(dim=1)  # (K,)
    a_over_b = state.a / state.b  # (K, ldim)
    priornum = torch.empty(s, p, kt, dtype=_DTYPE, device=y.device)
    priornum[:, :, 0] = state.nu_mcl
    priornum[:, :, 1:] = a_over_b.unsqueeze(1)
    _sign, logabsdet = torch.linalg.slogdet(state.lcov)  # (K, D)
    diag_lcov = torch.diagonal(state.lcov, dim1=2, dim2=3)  # (K, D, kt)
    f1 = priorln + (logabsdet - kt).sum(dim=1)  # (K,)
    f1 = f1 - ((diag_lcov + state.lm**2) * priornum).sum(dim=(1, 2))
    f1 = f1 - (state.nu_mcl * (-2 * state.lm[:, :, 0] * state.mean_mcl + state.mean_mcl**2)).sum(
        dim=1
    )
    f1 = 0.5 * f1  # (K,)

    f_total = (f1 + f3 + f4 + f6).sum()

    # Dirichlet HMM-parameter KL penalties.
    ua = torch.full((s,), ALPHA_A / s, dtype=_DTYPE, device=y.device)
    upi = torch.full((s,), ALPHA_PI / s, dtype=_DTYPE, device=y.device)
    f_hmm = -sum(kl_dirichlet(state.wa[i], ua) for i in range(s)) - kl_dirichlet(state.wpi, upi)
    return f_total + f_hmm
