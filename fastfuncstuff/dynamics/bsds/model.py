"""BSDS fit orchestration: VB coordinate ascent, restarts, group and subject fits.

``fit_bsds`` runs the mean-field VB loop (observation-model updates + HMM E-step
+ ELBO) with several random restarts, keeps the best by free energy, decodes the
MAP state sequence per session, and extracts the reportable quantities: per-state
mean and covariance (the dynamic FC), the transition matrix, per-state AR(1)
dynamics, and the ARD effective dimensionality. ``fit_subject`` re-fits a single
session under a group model's posterior as an informative prior (Fig. 1d of the
reference), giving subject-specific parameters that stay in register with the
group states.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from tqdm.auto import tqdm

from fastfuncstuff.dynamics.bsds import ar, hmm, vb
from fastfuncstuff.dynamics.bsds.elbo import lower_bound
from fastfuncstuff.dynamics.bsds.init import init_state
from fastfuncstuff.dynamics.bsds.vb import VBState

_DTYPE = torch.float64


@dataclass
class BSDSModel:
    """Outputs of a BSDS fit."""

    n_states: int
    ldim: int
    session_lengths: list[int]
    state_means: torch.Tensor  # (K, D) — column-0 loading of each state
    state_covs: torch.Tensor  # (K, D, D) — Lambda Lambda' + diag(1/psii) = dynamic FC
    transition: torch.Tensor  # (K, K) — normalised posterior-mean transition matrix
    init_probs: torch.Tensor  # (K,)
    responsibilities: list[torch.Tensor]  # per session (T_i, K)
    viterbi_states: list[torch.Tensor]  # per session (T_i,) int MAP path
    ar_transitions: torch.Tensor  # (K, ldim, ldim) — per-state AR(1) transition
    ar_noise_cov: torch.Tensor  # (K, ldim, ldim)
    effective_dim: torch.Tensor  # (K,) — ARD-active factor count per state
    loadings: torch.Tensor  # (K, D, ldim) — factor loadings (no mean column)
    psii: torch.Tensor  # (D,) — noise precision
    objective_history: list[float]
    converged: bool
    state: VBState = field(repr=False)


def _one_pass(
    state: VBState,
    y: torch.Tensor,
    sessions: list[torch.Tensor],
    xi_sum: torch.Tensor | None,
    gamma0_sum: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One VB iteration. Returns ``(xi_sum, gamma0_sum, loglik)`` from the E-step."""
    if xi_sum is not None and gamma0_sum is not None:
        vb.update_qtheta(state, xi_sum, gamma0_sum)
    vb.update_qnu(state)
    vb.update_qx(state, y)
    vb.update_ql(state, y)
    vb.update_psi(state, y)
    vb.update_mcl(state)
    log_obs = vb.compute_log_out_probs(state, sessions)
    gammas, xi_sum, gamma0_sum, loglik = hmm.estep(log_obs, state.wa, state.wpi)
    state.qns = torch.cat(gammas, dim=0)
    return xi_sum, gamma0_sum, loglik


def _run_vb(
    state: VBState,
    y: torch.Tensor,
    sessions: list[torch.Tensor],
    n_iter: int,
    tol: float,
    show_progress: bool,
    desc: str,
) -> tuple[VBState, list[float], bool]:
    """Run the VB loop to convergence; returns state, objective history, converged."""
    history: list[float] = []
    xi_sum: torch.Tensor | None = None
    gamma0_sum: torch.Tensor | None = None
    converged = False
    bar = tqdm(range(n_iter), desc=desc, leave=True, disable=not show_progress)
    for it in bar:
        xi_sum, gamma0_sum, _loglik = _one_pass(state, y, sessions, xi_sum, gamma0_sum)
        # Free energy F (excluding the HMM data term, which the emissions already
        # carry): this is the monotone VB objective and the reference's restart
        # selection criterion. Adding loglik would double-count the emission.
        obj = float(lower_bound(state, y))
        history.append(obj)
        if it > 0:
            denom = max(abs(history[-2]), 1.0)
            if abs(history[-1] - history[-2]) / denom < tol:
                converged = True
                if show_progress:
                    bar.set_postfix(obj=f"{obj:.3f}", converged=True)
                break
        if show_progress:
            bar.set_postfix(obj=f"{obj:.3f}")
    return state, history, converged


def _finalize(state: VBState, y: torch.Tensor, sessions: list[torch.Tensor]) -> BSDSModel:
    """Extract reportable outputs from a converged variational posterior."""
    k, d = state.n_states, state.n_roi
    lam = state.lm[:, :, 1:]  # (K, D, ldim)
    noise_var = 1.0 / state.psii  # (D,)
    covs = torch.einsum("kdi,kei->kde", lam, lam) + torch.diag_embed(noise_var.expand(k, d))

    trans = state.wa / state.wa.sum(dim=1, keepdim=True)
    init_probs = state.wpi / state.wpi.sum()

    log_obs = vb.compute_log_out_probs(state, sessions)
    responsibilities: list[torch.Tensor] = []
    viterbi_states: list[torch.Tensor] = []
    xi_sum = torch.zeros(k, k, dtype=_DTYPE, device=y.device)
    gamma0 = torch.zeros(k, dtype=_DTYPE, device=y.device)
    loglik = torch.zeros((), dtype=_DTYPE, device=y.device)
    log_a = hmm.expected_log_transition(state.wa)
    log_pi = hmm.expected_log_init(state.wpi)
    for lo in log_obs:
        gamma, xi, ll = hmm.forward_backward(lo, log_a, log_pi)
        responsibilities.append(gamma)
        xi_sum += xi
        gamma0 += gamma[0]
        loglik += ll
        viterbi_states.append(hmm.viterbi(lo, trans, init_probs))

    b_all, nc_all = ar.fit_all_state_vars(state.xm, state.qns, state.session_lengths)

    # ARD effective dimensionality: a factor is "active" for a state when its
    # posterior precision a/b is not driven far above the prior (=1).
    a_over_b = state.a / state.b  # (K, ldim)
    effective_dim = (a_over_b < 10.0).sum(dim=1).to(_DTYPE)

    return BSDSModel(
        n_states=k,
        ldim=state.ldim,
        session_lengths=list(state.session_lengths),
        state_means=state.lm[:, :, 0].clone(),
        state_covs=covs,
        transition=trans,
        init_probs=init_probs,
        responsibilities=responsibilities,
        viterbi_states=viterbi_states,
        ar_transitions=b_all,
        ar_noise_cov=nc_all,
        effective_dim=effective_dim,
        loadings=lam.clone(),
        psii=state.psii.clone(),
        objective_history=[],
        converged=False,
        state=state,
    )


def fit_bsds(
    sessions: list[torch.Tensor],
    n_states: int,
    *,
    max_ldim: int | None = None,
    n_iter: int = 100,
    n_init: int = 10,
    n_init_iter: int = 10,
    tol: float = 1e-4,
    seed: int = 0,
    device: torch.device | None = None,
    show_progress: bool | None = None,
) -> BSDSModel:
    """Fit a group-level BSDS model to a list of preprocessed ``(D, N)`` sessions.

    ``max_ldim`` bounds the latent factor dimensionality (defaults to ``D - 1``);
    ARD prunes it per state. ``n_init`` short restarts are run and the best by
    free energy is continued to convergence.
    """
    if len(sessions) == 0:
        raise ValueError("sessions is empty")
    device = sessions[0].device if device is None else device
    sessions = [s.to(device=device, dtype=_DTYPE) for s in sessions]
    d = sessions[0].shape[0]
    if max_ldim is None:
        max_ldim = d - 1
    ldim = min(max_ldim, d - 1)
    lengths = [int(s.shape[1]) for s in sessions]
    y = torch.cat(sessions, dim=1)
    if show_progress is None:
        show_progress = y.shape[1] >= 2000

    # Restarts: short VB from each seed, keep the best free energy.
    best_state: VBState | None = None
    best_obj = -float("inf")
    restart_bar = tqdm(
        range(n_init), desc="bsds restarts", leave=True, disable=not show_progress or n_init == 1
    )
    for r in restart_bar:
        state = init_state(y, lengths, n_states, ldim, seed=seed + r, device=device)
        state, hist, _ = _run_vb(
            state, y, sessions, n_init_iter, tol, show_progress=False, desc="init"
        )
        if hist[-1] > best_obj:
            best_obj = hist[-1]
            best_state = state
        if show_progress and n_init > 1:
            restart_bar.set_postfix(best=f"{best_obj:.3f}")
    assert best_state is not None

    # Continue the best restart to convergence.
    state, history, converged = _run_vb(
        best_state, y, sessions, n_iter, tol, show_progress, desc="bsds fit"
    )
    model = _finalize(state, y, sessions)
    model.objective_history = history
    model.converged = converged
    return model


def fit_subject(
    session: torch.Tensor,
    group: BSDSModel,
    *,
    n_iter: int = 50,
    tol: float = 1e-4,
    device: torch.device | None = None,
    show_progress: bool = False,
) -> BSDSModel:
    """Fit one session under a group model's posterior as an informative prior.

    Initialises every variational factor from ``group`` (loadings, noise, ARD,
    HMM Dirichlets) and runs the VB loop on the single session, so the resulting
    subject-specific states stay matched to the group states (Fig. 1d).
    """
    device = group.state.psii.device if device is None else device
    sess = session.to(device=device, dtype=_DTYPE)
    lengths = [int(sess.shape[1])]
    y = sess

    g = group.state
    state = init_state(y, lengths, group.n_states, group.ldim, seed=0, device=device)
    # Warm-start from the group posterior (informative prior).
    state.lm = g.lm.clone()
    state.lcov = g.lcov.clone()
    state.psii = g.psii.clone()
    state.a = g.a
    state.b = g.b.clone()
    state.mean_mcl = g.mean_mcl.clone()
    state.nu_mcl = g.nu_mcl.clone()
    state.wa = g.wa.clone()
    state.wpi = g.wpi.clone()

    state, history, converged = _run_vb(
        state, y, [sess], n_iter, tol, show_progress, desc="bsds subject"
    )
    model = _finalize(state, y, [sess])
    model.objective_history = history
    model.converged = converged
    return model
