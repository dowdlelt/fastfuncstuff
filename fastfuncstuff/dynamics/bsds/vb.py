"""Variational-Bayes updates for the BSDS switching factor-analysis model.

State is carried in :class:`VBState`. Every per-state quantity is a stacked
tensor over the ``K`` states; the augmented latent dimension is ``kt = ldim + 1``
(coordinate 0 is pinned to 1, so loading column 0 is the state mean and columns
``1..ldim`` are the factor loadings). All linear algebra runs in float64 — the
matrices are small (``D`` ROIs, ``ldim`` factors) and the Cholesky/inverse steps
are the numerically sensitive part (see the ``[[Float32 vs float64]]`` principle).

Each ``update_*`` function is one coordinate-ascent step and mirrors one of the
reference MATLAB routines (``inferQnu``, ``inferQX``, ``inferQL``, ``infermcl``,
``inferpsii2``, ``inferQtheta``, ``computeLogOutProbs``), reimplemented from the
model equations and generalised to unequal session lengths.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Fixed hyperparameters (BSDS defaults): Gamma prior on loading precisions, and
# symmetric Dirichlet priors on the HMM transition/initial distributions.
PA = 1.0
PB = 1.0
ALPHA_A = 1.0
ALPHA_PI = 1.0
PSI_MIN = 1e-5  # floor on noise variance (1/psii) to keep emissions finite
_DTYPE = torch.float64


@dataclass
class VBState:
    """Full variational posterior for a BSDS fit (all tensors float64)."""

    # dims
    n_states: int
    n_roi: int
    ldim: int
    n_time: int
    session_lengths: list[int]
    # observation model
    lm: torch.Tensor  # (K, D, kt) loadings; col 0 = state mean
    lcov: torch.Tensor  # (K, D, kt, kt) per-(state, ROI) loading covariance
    xm: torch.Tensor  # (K, kt, N) latent means; row 0 == 1
    xcov: torch.Tensor  # (K, kt, kt) shared latent covariance; row/col 0 == 0
    psii: torch.Tensor  # (D,) noise precision (diagonal, shared across states)
    # ARD
    a: float  # Gamma shape on loading-column precisions (scalar)
    b: torch.Tensor  # (K, ldim) Gamma rates on loading-column precisions
    mean_mcl: torch.Tensor  # (D,) hyperprior mean pooling state means
    nu_mcl: torch.Tensor  # (D,) hyperprior precision pooling state means
    # HMM
    wa: torch.Tensor  # (K, K) Dirichlet transition concentrations
    wpi: torch.Tensor  # (K,) Dirichlet initial concentrations
    # responsibilities
    qns: torch.Tensor  # (N, K) state responsibilities (concat over sessions)

    @property
    def kt(self) -> int:
        return self.ldim + 1


def update_qnu(state: VBState) -> None:
    """ARD Gamma posterior on loading-column precisions (``inferQnu``)."""
    state.a = PA + 0.5 * state.n_roi
    diag_lcov = torch.diagonal(state.lcov[:, :, 1:, 1:], dim1=2, dim2=3)  # (K, D, ldim)
    sum_over_roi = diag_lcov.sum(dim=1)  # (K, ldim)
    sum_lm2 = (state.lm[:, :, 1:] ** 2).sum(dim=1)  # (K, ldim)
    state.b = PB + 0.5 * (sum_over_roi + sum_lm2)


def update_qx(state: VBState, y: torch.Tensor) -> None:
    """Latent posterior means/covariance given loadings and data (``inferQX``)."""
    psii = state.psii
    lam = state.lm[:, :, 1:]  # (K, D, ldim) factor loadings
    mean = state.lm[:, :, 0]  # (K, D) state means
    # T1 = sum_q psii_q Lcov_q[1:,1:] + Lambda' diag(psii) Lambda
    term1 = torch.einsum("q,kqij->kij", psii, state.lcov[:, :, 1:, 1:])
    term2 = torch.einsum("kqi,q,kqj->kij", lam, psii, lam)
    t1 = term1 + term2  # (K, ldim, ldim)
    eye = torch.eye(state.ldim, dtype=_DTYPE, device=y.device)
    inner = torch.linalg.inv(eye + t1)  # (K, ldim, ldim)
    state.xcov = torch.zeros(state.n_states, state.kt, state.kt, dtype=_DTYPE, device=y.device)
    state.xcov[:, 1:, 1:] = inner
    # Xm[1:] = Xcov[1:,1:] Lambda' diag(psii) (Y - mean)
    resid = y.unsqueeze(0) - mean.unsqueeze(2)  # (K, D, N)
    rhs = torch.einsum("kqi,q,kqn->kin", lam, psii, resid)  # (K, ldim, N)
    xm_lat = torch.einsum("kij,kjn->kin", inner, rhs)  # (K, ldim, N)
    state.xm = torch.ones(state.n_states, state.kt, state.n_time, dtype=_DTYPE, device=y.device)
    state.xm[:, 1:, :] = xm_lat


def update_ql(state: VBState, y: torch.Tensor) -> None:
    """Loading posterior given latent, data and ARD priors (``inferQL``)."""
    psii = state.psii
    qns = state.qns  # (N, K)
    a_over_b = state.a / state.b  # (K, ldim)
    # num[t] is (D, kt): column 0 = nu_mcl (per ROI), columns 1: = a/b (per state)
    for t in range(state.n_states):
        n_eff = qns[:, t].sum()
        temp = state.xm[t] * qns[:, t]  # (kt, N)
        t2 = state.xcov[t] * n_eff + state.xm[t] @ temp.T  # (kt, kt)
        t3 = psii.unsqueeze(1) * (y @ temp.T)  # (D, kt)
        num = torch.empty(state.n_roi, state.kt, dtype=_DTYPE, device=y.device)
        num[:, 0] = state.nu_mcl
        num[:, 1:] = a_over_b[t].unsqueeze(0)
        # A_q = diag(num_q) + psii_q * T2  ->  (D, kt, kt)
        a_q = torch.diag_embed(num) + psii.view(-1, 1, 1) * t2.unsqueeze(0)
        lcov_t = torch.linalg.inv(a_q)  # (D, kt, kt)
        rhs = t3.clone()  # (D, kt)
        rhs[:, 0] = rhs[:, 0] + state.mean_mcl * state.nu_mcl
        lm_t = torch.einsum("qk,qkl->ql", rhs, lcov_t)  # (D, kt)
        state.lcov[t] = lcov_t
        state.lm[t] = lm_t


def update_mcl(state: VBState) -> None:
    """Hyperprior pooling state means across states (``infermcl``)."""
    if state.n_states <= 1:
        return
    means = state.lm[:, :, 0].T  # (D, K) each column a state mean
    lcov00 = state.lcov[:, :, 0, 0].T  # (D, K) posterior var of each state mean
    state.mean_mcl = means.mean(dim=1)  # (D,)
    s = state.n_states
    denom = (
        lcov00.sum(dim=1)
        + (means**2).sum(dim=1)
        - 2 * state.mean_mcl * means.sum(dim=1)
        + s * state.mean_mcl**2
    )
    state.nu_mcl = s / denom


def update_psi(state: VBState, y: torch.Tensor) -> None:
    """Diagonal noise precision (``inferpsii2``, per-dimension variance branch)."""
    qns = state.qns
    psi2_diag = torch.zeros(state.n_roi, dtype=_DTYPE, device=y.device)
    for t in range(state.n_states):
        temp_alt = state.xm[t] * qns[:, t]  # (kt, N)
        temp = state.xcov[t] * qns[:, t].sum() + state.xm[t] @ temp_alt.T  # (kt, kt)
        lm_xm = state.lm[t] @ state.xm[t]  # (D, N)
        term1 = (qns[:, t] * y * (y - 2 * lm_xm)).sum(dim=1)  # (D,)
        term2 = torch.einsum("qi,ij,qj->q", state.lm[t], temp, state.lm[t])  # (D,)
        term3 = (state.lcov[t] * temp.T.unsqueeze(0)).sum(dim=(1, 2))  # trace(Lcov_q temp)
        psi2_diag = psi2_diag + term1 + term2 + term3
    psi2_diag = psi2_diag / state.n_time
    # Floor the variance so the emission log-density stays finite.
    psi2_diag = psi2_diag.clamp_min(PSI_MIN)
    state.psii = 1.0 / psi2_diag


def update_qtheta(state: VBState, xi_sum: torch.Tensor, gamma0_sum: torch.Tensor) -> None:
    """Dirichlet posteriors on HMM transition/initial from E-step stats (``inferQtheta``)."""
    ua = ALPHA_A / state.n_states
    upi = ALPHA_PI / state.n_states
    state.wa = xi_sum + ua
    state.wpi = gamma0_sum + upi


def compute_log_out_probs(state: VBState, sessions: list[torch.Tensor]) -> list[torch.Tensor]:
    """Per-session ``(T_i, K)`` log-emission matrices (``computeLogOutProbs``).

    ``sessions`` is the list of ``(D, T_i)`` data blocks in the same order they
    were concatenated into the model's time axis.
    """
    psii = state.psii
    # Per-state constants that don't depend on time.
    temp_all = torch.empty(state.n_states, state.kt, state.kt, dtype=_DTYPE, device=psii.device)
    scalar_bd = torch.empty(state.n_states, dtype=_DTYPE, device=psii.device)
    eye_eps = 1e-12
    for t in range(state.n_states):
        lm_psii_lm = state.lm[t].T @ (psii.unsqueeze(1) * state.lm[t])  # (kt, kt)
        temp = lm_psii_lm + torch.einsum("q,qij->ij", psii, state.lcov[t])
        temp_all[t] = temp
        # <temp, Xcov> + trace(Xcov[1:,1:]) - 2*sum(log diag chol(Xcov[1:,1:]))
        b_term = (temp * state.xcov[t]).sum()
        d_term = torch.diagonal(state.xcov[t][1:, 1:]).sum()
        chol = torch.linalg.cholesky(
            state.xcov[t][1:, 1:]
            + eye_eps * torch.eye(state.ldim, dtype=_DTYPE, device=psii.device)
        )
        f_term = -2.0 * torch.log(torch.diagonal(chol)).sum()
        scalar_bd[t] = b_term + d_term + f_term

    out: list[torch.Tensor] = []
    col = 0
    for sess in sessions:
        ti = sess.shape[1]
        yblk = sess.to(_DTYPE)  # (D, Ti)
        log_obs = torch.empty(ti, state.n_states, dtype=_DTYPE, device=psii.device)
        for t in range(state.n_states):
            xm_blk = state.xm[t][:, col : col + ti]  # (kt, Ti)
            lm_xm = state.lm[t] @ xm_blk  # (D, Ti)
            a_term = (psii.unsqueeze(1) * yblk * (yblk - 2 * lm_xm)).sum(dim=0)  # (Ti,)
            temp = temp_all[t]
            c_term = (xm_blk * (temp @ xm_blk)).sum(dim=0)  # (Ti,)
            e_term = (xm_blk[1:] * xm_blk[1:]).sum(dim=0)  # (Ti,)
            log_obs[:, t] = -0.5 * (a_term + scalar_bd[t] + c_term + e_term)
        out.append(log_obs)
        col += ti
    return out
