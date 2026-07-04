"""Per-state autoregressive dynamics for BSDS.

Each latent state carries an AR(1) model of its factor trajectory:
``x_t ~= B_s x_{t-1} + noise``. In the reference MATLAB this is a VB
vector-autoregression (``mstep_VBVAR``) with an ARD prior on ``vec(B)`` and a
Wishart posterior on the noise precision. Two facts about the reference guided
this port:

1. In the main VB loop ``inferAR3`` runs *before* ``inferQX``, which then
   recomputes the latent purely from the data and loadings — so the AR update to
   the latent is overwritten every iteration, and the fitted VAR parameters are
   local and never stored. The AR term is therefore near-inert in the converged
   reference model; its tangible product is the per-state transition matrix
   ``B_s``, which the reference discards. (See ``[[BSDS]]`` for the write-up.)

2. Under mean-field, the VB-VAR posterior mean of ``B`` is exactly the
   ARD-ridge-regularised weighted least-squares solution. We estimate that
   directly from responsibility-weighted lag-covariances — the useful, stable
   part — and surface ``B_s`` and the state noise covariance as diagnostics.
"""

from __future__ import annotations

import torch

_DTYPE = torch.float64
# ARD prior mean on vec(B) precision: reference uses ao=bo=1e-3, i.e. E[alpha]=1.
_AR_RIDGE = 1e-3


def _lag_moments(
    x_lat: torch.Tensor,
    gamma_state: torch.Tensor,
    session_lengths: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Responsibility-weighted lag moments for one state.

    ``x_lat`` is ``(ldim, N)`` (the factor part of the latent, no bias row);
    ``gamma_state`` is ``(N,)`` responsibilities. Pairs never cross a session
    boundary. Returns ``Suu, Svu, Syy`` (each ``ldim x ldim``) and the effective
    sample count ``Neff``, where ``u = x_{t-1}``, ``v = x_t``.
    """
    ldim = x_lat.shape[0]
    suu = torch.zeros(ldim, ldim, dtype=_DTYPE, device=x_lat.device)
    svu = torch.zeros(ldim, ldim, dtype=_DTYPE, device=x_lat.device)
    syy = torch.zeros(ldim, ldim, dtype=_DTYPE, device=x_lat.device)
    neff = torch.zeros((), dtype=_DTYPE, device=x_lat.device)
    col = 0
    for ti in session_lengths:
        if ti >= 2:
            u = x_lat[:, col : col + ti - 1]  # (ldim, Ti-1)
            v = x_lat[:, col + 1 : col + ti]  # (ldim, Ti-1)
            w = gamma_state[col + 1 : col + ti]  # weight each pair by P(state at t)
            uw = u * w
            suu = suu + u @ uw.T
            svu = svu + v @ uw.T
            syy = syy + v @ (v * w).T
            neff = neff + w.sum()
        col += ti
    return suu, svu, syy, neff


def fit_state_var(
    x_lat: torch.Tensor,
    gamma_state: torch.Tensor,
    session_lengths: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit the AR(1) transition ``B`` and noise covariance for one state.

    Returns ``(B (ldim, ldim), noise_cov (ldim, ldim))``. ``B`` is the
    ARD-ridge-regularised weighted least-squares transition; ``noise_cov`` is the
    responsibility-weighted residual covariance of the one-step prediction.
    """
    ldim = x_lat.shape[0]
    suu, svu, syy, neff = _lag_moments(x_lat, gamma_state, session_lengths)
    eye = torch.eye(ldim, dtype=_DTYPE, device=x_lat.device)
    b = svu @ torch.linalg.inv(suu + _AR_RIDGE * eye)  # (ldim, ldim)
    if neff > 1:
        resid_2m = syy - b @ svu.T  # E[(v - Bu)(v - Bu)'] * Neff
        noise_cov = resid_2m / neff
        noise_cov = 0.5 * (noise_cov + noise_cov.T)  # symmetrise
    else:
        noise_cov = eye.clone()
    return b, noise_cov


def fit_all_state_vars(
    xm: torch.Tensor,
    gammas_concat: torch.Tensor,
    session_lengths: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit per-state AR(1) transitions and noise covariances.

    ``xm`` is ``(K, kt, N)`` (row 0 is the bias); ``gammas_concat`` is ``(N, K)``.
    Returns ``B_all (K, ldim, ldim)`` and ``noise_cov_all (K, ldim, ldim)``.
    """
    n_states = xm.shape[0]
    ldim = xm.shape[1] - 1
    b_all = torch.empty(n_states, ldim, ldim, dtype=_DTYPE, device=xm.device)
    nc_all = torch.empty(n_states, ldim, ldim, dtype=_DTYPE, device=xm.device)
    for t in range(n_states):
        b_all[t], nc_all[t] = fit_state_var(xm[t, 1:, :], gammas_concat[:, t], session_lengths)
    return b_all, nc_all
