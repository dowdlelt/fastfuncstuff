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

    f_total = torch.zeros((), dtype=_DTYPE, device=y.device)
    for t in range(s):
        qn = state.qns[:, t]
        neff = qn.sum()
        temp_alt = state.xm[t] * qn  # (kt, N)
        xcor = state.xcov[t] * neff + state.xm[t] @ temp_alt.T  # (kt, kt)

        # Row 3: latent entropy / prior cross term.
        chol = torch.linalg.cholesky(state.xcov[t][1:, 1:])
        logdet_half = torch.log(torch.diagonal(chol)).sum()
        f3 = 0.5 * neff * (kt - 1) - 0.5 * torch.diagonal(xcor[1:, 1:]).sum() + neff * logdet_half

        # Row 4: expected log-likelihood of the observations under state t.
        lm_psii_lm = state.lm[t].T @ (psii.unsqueeze(1) * state.lm[t])  # (kt, kt)
        sum_psii_lcov = torch.einsum("q,qij->ij", psii, state.lcov[t])  # (kt, kt)
        lm_xm = state.lm[t] @ state.xm[t]  # (D, N)
        data_quad = (psii * (qn * y * (y - 2 * lm_xm)).sum(dim=1)).sum()
        f4 = (
            -0.5 * neff * (-log_det_psi + p * two_pi)
            - 0.5 * (lm_psii_lm * xcor.T).sum()
            - 0.5 * (sum_psii_lcov * xcor.T).sum()
            - 0.5 * data_quad
        )

        # Row 6: -KL of the loading-precision ARD Gamma posterior.
        f6 = -kl_gamma(state.a, state.b[t], PA, PB)

        # Row 1: loading posterior entropy + ARD / mean-hyperprior cross terms.
        priorln = torch.log(state.nu_mcl).sum() + p * (dig_a - torch.log(state.b[t])).sum()
        a_over_b = state.a / state.b[t]  # (ldim,)
        priornum = torch.empty(p, kt, dtype=_DTYPE, device=y.device)
        priornum[:, 0] = state.nu_mcl
        priornum[:, 1:] = a_over_b.unsqueeze(0)
        sign, logabsdet = torch.linalg.slogdet(state.lcov[t])  # (D,) each
        f1 = priorln + (logabsdet - kt).sum()
        diag_lcov = torch.diagonal(state.lcov[t], dim1=1, dim2=2)  # (D, kt)
        f1 = f1 - ((diag_lcov + state.lm[t] ** 2) * priornum).sum()
        f1 = (
            f1
            - (state.nu_mcl * (-2 * state.lm[t][:, 0] * state.mean_mcl + state.mean_mcl**2)).sum()
        )
        f1 = 0.5 * f1

        f_total = f_total + f1 + f3 + f4 + f6

    # Dirichlet HMM-parameter KL penalties.
    ua = torch.full((s,), ALPHA_A / s, dtype=_DTYPE, device=y.device)
    upi = torch.full((s,), ALPHA_PI / s, dtype=_DTYPE, device=y.device)
    f_hmm = -sum(kl_dirichlet(state.wa[i], ua) for i in range(s)) - kl_dirichlet(state.wpi, upi)
    return f_total + f_hmm
