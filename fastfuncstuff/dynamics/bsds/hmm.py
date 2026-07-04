"""Variational HMM engine for BSDS: forward-backward E-step and Viterbi decode.

The observation model hands us a log-emission matrix ``log_obs`` of shape
``(T, K)`` per session (BSDS computes this in :mod:`.vb`). The HMM couples those
emissions across time with Dirichlet posteriors on the transition matrix
``Wa`` (``K x K``) and the initial distribution ``Wpi`` (``K``).

Following standard variational HMM inference (and the BSDS MATLAB), the E-step
uses *sub-normalised* transition weights ``exp(E[log A])`` where
``E[log A_ij] = psi(Wa_ij) - psi(sum_j Wa_ij)`` — the geometric-mean transitions
that fall out of the mean-field update — rather than the plain posterior mean.
Everything runs in log-space for numerical stability. Viterbi (final MAP state
sequence) instead uses the ordinary normalised posterior-mean transitions.

All routines operate per session; a dataset is a list of ``(T_i, K)`` emission
matrices, so sessions of different lengths are handled directly (unlike the
equal-length assumption baked into the reference MATLAB).
"""

from __future__ import annotations

import torch


def expected_log_transition(wa: torch.Tensor) -> torch.Tensor:
    """``E[log A]`` under a Dirichlet posterior with row concentrations ``wa``.

    ``wa`` is ``(K, K)`` with ``wa[i, j]`` the concentration for ``i -> j``.
    Returns the ``(K, K)`` matrix ``psi(wa) - psi(rowsum(wa))`` (rows are the
    sub-normalised transition log-weights; ``exp`` of them need not sum to 1).
    """
    return torch.digamma(wa) - torch.digamma(wa.sum(dim=1, keepdim=True))


def expected_log_init(wpi: torch.Tensor) -> torch.Tensor:
    """``E[log pi]`` under a Dirichlet posterior with concentrations ``wpi`` (``K``)."""
    return torch.digamma(wpi) - torch.digamma(wpi.sum())


def forward_backward(
    log_obs: torch.Tensor,
    log_a: torch.Tensor,
    log_pi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Log-space forward-backward for one session.

    Parameters
    ----------
    log_obs : ``(T, K)`` log-emission weights.
    log_a : ``(K, K)`` expected-log transition (``i -> j``).
    log_pi : ``(K,)`` expected-log initial.

    Returns
    -------
    gamma : ``(T, K)`` posterior state responsibilities (rows sum to 1).
    xi_sum : ``(K, K)`` summed pairwise responsibilities ``sum_t P(z_t=i, z_{t+1}=j)``.
    loglik : scalar evidence proxy (log-normaliser of the forward pass).
    """
    t_len, k = log_obs.shape
    log_alpha = torch.empty_like(log_obs)
    log_alpha[0] = log_pi + log_obs[0]
    for t in range(1, t_len):
        # alpha[t, j] = obs[t, j] + logsumexp_i(alpha[t-1, i] + A[i, j])
        prev = log_alpha[t - 1].unsqueeze(1) + log_a  # (K_i, K_j)
        log_alpha[t] = log_obs[t] + torch.logsumexp(prev, dim=0)
    loglik = torch.logsumexp(log_alpha[-1], dim=0)

    log_beta = torch.zeros_like(log_obs)
    for t in range(t_len - 2, -1, -1):
        # beta[t, i] = logsumexp_j(A[i, j] + obs[t+1, j] + beta[t+1, j])
        nxt = log_a + (log_obs[t + 1] + log_beta[t + 1]).unsqueeze(0)  # (K_i, K_j)
        log_beta[t] = torch.logsumexp(nxt, dim=1)

    log_gamma = log_alpha + log_beta
    gamma = torch.softmax(log_gamma, dim=1)

    # xi[t, i, j] = alpha[t, i] + A[i, j] + obs[t+1, j] + beta[t+1, j] - loglik
    if t_len > 1:
        log_xi = (
            log_alpha[:-1].unsqueeze(2)
            + log_a.unsqueeze(0)
            + (log_obs[1:] + log_beta[1:]).unsqueeze(1)
        )  # (T-1, K_i, K_j)
        log_xi = log_xi - torch.logsumexp(log_xi.reshape(t_len - 1, -1), dim=1).view(-1, 1, 1)
        xi_sum = log_xi.exp().sum(dim=0)
    else:
        xi_sum = torch.zeros(k, k, dtype=log_obs.dtype, device=log_obs.device)
    return gamma, xi_sum, loglik


def estep(
    session_log_obs: list[torch.Tensor],
    wa: torch.Tensor,
    wpi: torch.Tensor,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the E-step over all sessions.

    Returns per-session ``gamma`` responsibilities, the transition and initial
    sufficient statistics ``(xi_sum (K,K), gamma0_sum (K,))`` needed to update
    the Dirichlet posteriors, and the total evidence ``loglik``.
    """
    log_a = expected_log_transition(wa)
    log_pi = expected_log_init(wpi)
    k = wa.shape[0]
    gammas: list[torch.Tensor] = []
    xi_total = torch.zeros(k, k, dtype=wa.dtype, device=wa.device)
    gamma0_total = torch.zeros(k, dtype=wa.dtype, device=wa.device)
    loglik = torch.zeros((), dtype=wa.dtype, device=wa.device)
    for log_obs in session_log_obs:
        gamma, xi_sum, ll = forward_backward(log_obs.to(wa.dtype), log_a, log_pi)
        gammas.append(gamma)
        xi_total = xi_total + xi_sum
        gamma0_total = gamma0_total + gamma[0]
        loglik = loglik + ll
    return gammas, xi_total, gamma0_total, loglik


def viterbi(
    log_obs: torch.Tensor,
    trans: torch.Tensor,
    init: torch.Tensor,
) -> torch.Tensor:
    """MAP state sequence for one session (log-space Viterbi).

    ``trans`` and ``init`` are ordinary (normalised) posterior-mean transition
    and initial distributions; ``log_obs`` is the ``(T, K)`` log-emission matrix.
    Returns an integer ``(T,)`` path.
    """
    t_len, k = log_obs.shape
    log_trans = torch.log(trans.clamp_min(1e-300))
    log_init = torch.log(init.clamp_min(1e-300))
    delta = log_init + log_obs[0]
    back = torch.empty((t_len, k), dtype=torch.long, device=log_obs.device)
    back[0] = 0
    for t in range(1, t_len):
        scores = delta.unsqueeze(1) + log_trans  # (K_prev, K_cur)
        best, arg = scores.max(dim=0)
        delta = best + log_obs[t]
        back[t] = arg
    path = torch.empty(t_len, dtype=torch.long, device=log_obs.device)
    path[-1] = int(delta.argmax())
    for t in range(t_len - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path
