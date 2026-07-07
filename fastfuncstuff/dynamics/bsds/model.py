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

import numpy as np
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
    ard_precision: torch.Tensor  # (K, ldim) — ARD precision a/b per factor (>~10 = pruned)
    loadings: torch.Tensor  # (K, D, ldim) — factor loadings (no mean column)
    psii: torch.Tensor  # (D,) — noise precision
    objective_history: list[float]
    converged: bool
    state: VBState = field(repr=False)
    # Per-iteration L1 change in per-state responsibility mass (the reference's
    # weights convergence signal); iteration 0 is nan. Empty on models loaded
    # from disk that predate this field or were saved without it.
    weights_history: list[float] = field(default_factory=list)
    # Free energy of each short restart (the value best-of selection ranks on),
    # in restart order. A tight spread means restarts land in similar-quality
    # basins; the running max updating only ~H_n times over n restarts is
    # expected record statistics, not a stuck fit. Empty on older saved models.
    restart_scores: list[float] = field(default_factory=list)


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


# Convergence criteria, plus the natural singular spellings as aliases. An
# unrecognised value must error, never silently fall back to free energy.
_CRITERION_ALIASES = {
    "free_energy": "free_energy",
    "free-energy": "free_energy",
    "fe": "free_energy",
    "weights": "weights",
    "weight": "weights",
}


def _normalize_criterion(criterion: str) -> str:
    try:
        return _CRITERION_ALIASES[criterion]
    except KeyError:
        raise ValueError(
            f"unknown criterion {criterion!r}; expected 'free_energy' or 'weights'"
        ) from None


def _run_vb(
    state: VBState,
    y: torch.Tensor,
    sessions: list[torch.Tensor],
    n_iter: int,
    tol: float,
    show_progress: bool,
    desc: str,
    criterion: str = "free_energy",
    obj_every: int = 1,
) -> tuple[VBState, list[float], bool, list[float]]:
    """Run the VB loop to convergence; returns state, objective history, converged.

    ``criterion`` picks the stopping test (restart selection always uses the
    free-energy ``history``, matching the reference's ``Fhist`` selection):

    - ``"free_energy"`` (default): relative free-energy change
      ``|ΔF|/|F| < tol``.
    - ``"weights"``: the reference ``vbhafa.m`` criterion —
      ``sum(|Σ_n q(z_n) - previous|) < tol``, the L1 change in per-state
      responsibility mass. This is an **absolute** quantity that scales with the
      number of time points N (unlike the relative free-energy tol), so it wants
      a larger tol; at large N it may never fire and simply runs ``n_iter``. It
      keeps iterating while state occupancy is still migrating, so it does not
      stop early with responsibility mass still draining out of redundant states
      (see the over-segmentation note in the wiki / auto-memory).

    ``obj_every`` computes the free energy only every ``obj_every`` iterations
    (plus always on the final/converged iteration). ``lower_bound`` is a pure
    read of the state — skipping it never changes the fit — and under the
    ``weights`` criterion it is not the stop signal, only the restart-selection
    value (final iteration) and the diagnostic curve. So ``obj_every > 1`` is a
    free speedup there: it drops the per-iteration cholesky/slogdet of the ELBO
    and, on GPU, the host↔device sync it forces. Ignored for ``free_energy``,
    which needs F every iteration. Skipped iterations record ``nan`` in
    ``history`` (the F curve just plots the computed points).
    """
    criterion = _normalize_criterion(criterion)
    obj_every = max(1, obj_every)
    history: list[float] = []
    # Per-iteration L1 change in per-state responsibility mass — the reference's
    # weights convergence signal. Recorded regardless of `criterion` so it is
    # always available as a diagnostic (plot_convergence); iteration 0 has no
    # predecessor and is nan.
    weights_history: list[float] = []
    xi_sum: torch.Tensor | None = None
    gamma0_sum: torch.Tensor | None = None
    converged = False
    prev_weights: torch.Tensor | None = None
    bar = tqdm(range(n_iter), desc=desc, leave=True, disable=not show_progress)
    for it in bar:
        xi_sum, gamma0_sum, _loglik = _one_pass(state, y, sessions, xi_sum, gamma0_sum)
        weights = state.qns.sum(dim=0)  # (K,) per-state responsibility mass
        improvement = (
            float((weights - prev_weights).abs().sum())
            if prev_weights is not None
            else float("nan")
        )
        weights_history.append(improvement)
        # Weights convergence is decided on the responsibility mass, not F.
        weights_converged = (
            criterion == "weights" and prev_weights is not None and improvement < tol
        )
        # Free energy F (excluding the HMM data term, which the emissions already
        # carry): the monotone VB objective and the reference's restart-selection
        # criterion. free_energy needs it every iteration to test |ΔF|; weights
        # needs it only on the final/converged iteration (restart selection) and
        # every `obj_every` for the curve. `lower_bound` never mutates `state`,
        # so skipping it leaves the fit bit-identical.
        need_obj = (
            criterion == "free_energy"
            or weights_converged
            or it == n_iter - 1
            or it % obj_every == 0
        )
        obj = float(lower_bound(state, y)) if need_obj else float("nan")
        history.append(obj)
        if criterion == "weights":
            if weights_converged:
                converged = True
                if show_progress:
                    bar.set_postfix(dW=f"{improvement:.3g}", converged=True)
                break
        elif it > 0:
            denom = max(abs(history[-2]), 1.0)
            if abs(history[-1] - history[-2]) / denom < tol:
                converged = True
                if show_progress:
                    bar.set_postfix(obj=f"{obj:.3f}", converged=True)
                break
        prev_weights = weights
        if show_progress:
            bar.set_postfix(obj=f"{obj:.3f}" if need_obj else f"dW={improvement:.3g}")
    return state, history, converged, weights_history


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
    ard_precision = a_over_b.clone()

    # Reportable tensors go to CPU so downstream save/stats/plots never touch a
    # device tensor (raw .numpy() would crash on GPU). The variational `state`
    # stays on its compute device so decode() can still run on the GPU.
    return BSDSModel(
        n_states=k,
        ldim=state.ldim,
        session_lengths=list(state.session_lengths),
        state_means=state.lm[:, :, 0].cpu(),
        state_covs=covs.cpu(),
        transition=trans.cpu(),
        init_probs=init_probs.cpu(),
        responsibilities=[r.cpu() for r in responsibilities],
        viterbi_states=[v.cpu() for v in viterbi_states],
        ar_transitions=b_all.cpu(),
        ar_noise_cov=nc_all.cpu(),
        effective_dim=effective_dim.cpu(),
        ard_precision=ard_precision.cpu(),
        loadings=lam.cpu(),
        psii=state.psii.cpu(),
        objective_history=[],
        converged=False,
        state=state,
    )


def posterior_arrays(model: BSDSModel) -> dict:
    """The arrays needed to persist a fit and later reconstruct it for :func:`decode`.

    Includes the reportable fields (means/covs/loadings/transition/AR/ARD), the
    variational-posterior pieces the reportable set drops — ``lcov``, the ARD
    ``a``/``b``, the mean hyperprior, and the *unnormalised* Dirichlet ``wa``/``wpi``
    (``transition`` alone loses the concentration magnitude) — and the training
    decode (concatenated responsibilities + Viterbi paths, split back per run on
    load) plus convergence info, so a reloaded model is a full stand-in for the
    original (QC, stats, comparisons) and not just decode-ready. :func:`load_bsds_model`
    reads exactly these keys.
    """
    s = model.state

    def np_(t):
        return t.detach().cpu().numpy()

    resp = (
        torch.cat(model.responsibilities).detach().cpu().numpy()
        if model.responsibilities
        else np.zeros((0, model.n_states))
    )
    vit = (
        torch.cat(model.viterbi_states).detach().cpu().numpy().astype(np.int64)
        if model.viterbi_states
        else np.zeros((0,), dtype=np.int64)
    )
    return {
        "n_states": np.array(model.n_states),
        "ldim": np.array(model.ldim),
        "session_lengths": np.array(model.session_lengths),
        "state_means": np_(model.state_means),
        "state_covs": np_(model.state_covs),
        "loadings": np_(model.loadings),
        "transition": np_(model.transition),
        "init_probs": np_(model.init_probs),
        "ar_transitions": np_(model.ar_transitions),
        "ar_noise_cov": np_(model.ar_noise_cov),
        "effective_dim": np_(model.effective_dim),
        "ard_precision": np_(model.ard_precision),
        "psii": np_(model.psii),
        "post_lcov": np_(s.lcov),
        "post_a": np.array(float(s.a)),
        "post_b": np_(s.b),
        "post_mean_mcl": np_(s.mean_mcl),
        "post_nu_mcl": np_(s.nu_mcl),
        "post_wa": np_(s.wa),
        "post_wpi": np_(s.wpi),
        "responsibilities": resp,
        "viterbi": vit,
        "converged": np.array(bool(model.converged)),
        "objective_history": np.array(model.objective_history, dtype=float),
        "weights_history": np.array(model.weights_history, dtype=float),
        "restart_scores": np.array(model.restart_scores, dtype=float),
    }


def save_bsds_model(model: BSDSModel, path: str) -> str:
    """Write a fit to ``path`` (``.npz``) so it can be reloaded and applied to new data."""
    np.savez(path, **posterior_arrays(model))
    return path


def load_bsds_model(path: str, *, device: torch.device | str | None = None) -> BSDSModel:
    """Reconstruct a :class:`BSDSModel` from :func:`save_bsds_model` (or the run's
    ``*_model.npz``) as a full stand-in for the original fit: usable for
    :func:`decode`, :func:`~fastfuncstuff.dynamics.states.compute_state_stats`, QC
    plots, and the MATLAB cross-check, without refitting. The training
    responsibilities and Viterbi paths are restored (split per run), as are the
    convergence flag and free-energy history.
    """
    z = np.load(path)
    dev = torch.device("cpu") if device is None else torch.device(device)
    k = int(z["n_states"])
    ldim = int(z["ldim"])
    lengths = [int(x) for x in z["session_lengths"]]
    d = int(z["psii"].shape[0])
    kt = ldim + 1

    def t(name):
        return torch.tensor(z[name], dtype=_DTYPE, device=dev)

    # Split the concatenated training decode back into per-run tensors (CPU, like
    # the reportable fields). Older files without these keys load as empty lists.
    responsibilities: list[torch.Tensor] = []
    viterbi_states: list[torch.Tensor] = []
    resp_all = z["responsibilities"] if "responsibilities" in z else np.zeros((0, k))
    vit_all = z["viterbi"] if "viterbi" in z else np.zeros((0,), dtype=np.int64)
    if resp_all.shape[0] == sum(lengths) and resp_all.shape[0] > 0:
        col = 0
        for length in lengths:
            responsibilities.append(torch.tensor(resp_all[col : col + length], dtype=_DTYPE))
            viterbi_states.append(torch.tensor(vit_all[col : col + length], dtype=torch.long))
            col += length

    state_means = t("state_means")  # (K, D)
    loadings = t("loadings")  # (K, D, ldim)
    lm = torch.zeros(k, d, kt, dtype=_DTYPE, device=dev)
    lm[:, :, 0] = state_means
    if ldim > 0:
        lm[:, :, 1:] = loadings
    n_time = max(int(sum(lengths)), 1)
    state = VBState(
        n_states=k,
        n_roi=d,
        ldim=ldim,
        n_time=n_time,
        session_lengths=list(lengths),
        lm=lm,
        lcov=t("post_lcov"),
        xm=torch.ones(k, kt, n_time, dtype=_DTYPE, device=dev),
        xcov=torch.zeros(k, kt, kt, dtype=_DTYPE, device=dev),
        psii=t("psii"),
        a=float(z["post_a"]),
        b=t("post_b"),
        mean_mcl=t("post_mean_mcl"),
        nu_mcl=t("post_nu_mcl"),
        wa=t("post_wa"),
        wpi=t("post_wpi"),
        qns=torch.zeros(n_time, k, dtype=_DTYPE, device=dev),
    )
    return BSDSModel(
        n_states=k,
        ldim=ldim,
        session_lengths=list(lengths),
        state_means=state_means.cpu(),
        state_covs=t("state_covs").cpu(),
        transition=t("transition").cpu(),
        init_probs=t("init_probs").cpu(),
        responsibilities=responsibilities,
        viterbi_states=viterbi_states,
        ar_transitions=t("ar_transitions").cpu(),
        ar_noise_cov=t("ar_noise_cov").cpu(),
        effective_dim=t("effective_dim").cpu(),
        ard_precision=t("ard_precision").cpu(),
        loadings=loadings.cpu(),
        psii=t("psii").cpu(),
        objective_history=(z["objective_history"].tolist() if "objective_history" in z else []),
        converged=bool(z["converged"]) if "converged" in z else False,
        state=state,
        weights_history=(z["weights_history"].tolist() if "weights_history" in z else []),
        restart_scores=(z["restart_scores"].tolist() if "restart_scores" in z else []),
    )


def fit_bsds(
    sessions: list[torch.Tensor],
    n_states: int,
    *,
    max_ldim: int | str | None = None,
    n_iter: int = 100,
    n_init: int = 10,
    n_init_iter: int = 10,
    tol: float = 1e-4,
    seed: int = 0,
    device: torch.device | None = None,
    show_progress: bool | None = None,
    n_kmeans_replicates: int = 10,
    kmeans_pca_dim: int | None = 20,
    criterion: str = "free_energy",
    obj_every: int | None = None,
) -> BSDSModel:
    """Fit a group-level BSDS model to a list of preprocessed ``(D, N)`` sessions.

    ``max_ldim`` bounds the latent factor dimensionality (defaults to ``D - 1``);
    ARD prunes it per state (this bounds the number of *states* returned only in
    that ``n_states`` is itself an upper bound you choose — ARD never drops
    states from the output, only shrinks each state's active factor count; see
    ``[[BSDS]]``). ``n_init`` short restarts are run and the best by free energy
    is continued to convergence.

    ``criterion`` selects the convergence test (see :func:`_run_vb`):
    ``"free_energy"`` (default) stops on relative free-energy change;
    ``"weights"`` ports the reference ``vbhafa.m`` state-mass criterion, which
    keeps iterating while occupancy is still migrating (recommended when you
    care about pruning redundant states rather than stopping as soon as F
    plateaus). Restart *selection* always uses free energy either way, matching
    the reference's ``Fhist`` selection.

    ``n_kmeans_replicates`` and ``kmeans_pca_dim`` control the k-means
    initialisation (per-session clustering, pooled — see
    :mod:`fastfuncstuff.dynamics.bsds.init`); the PCA projection matters once
    ``D`` is more than a couple dozen ROIs.

    ``obj_every`` controls how often the (read-only) free energy is evaluated
    inside the VB loop (see :func:`_run_vb`); ``None`` (default) evaluates every
    iteration for ``free_energy`` and every 10th for ``weights``, where F is only
    a diagnostic — a free speedup that skips the ELBO cholesky/slogdet and its
    GPU sync without changing the fit. Pass ``1`` to force a dense F curve.
    """
    if len(sessions) == 0:
        raise ValueError("sessions is empty")
    criterion = _normalize_criterion(criterion)
    obj_every = obj_every if obj_every is not None else (10 if criterion == "weights" else 1)
    device = sessions[0].device if device is None else device
    sessions = [s.to(device=device, dtype=_DTYPE) for s in sessions]
    d = sessions[0].shape[0]
    if max_ldim == "auto":
        from fastfuncstuff.dynamics.preprocess import estimate_latent_dim

        max_ldim = estimate_latent_dim(sessions)
    elif max_ldim is None:
        max_ldim = d - 1
    ldim = min(int(max_ldim), d - 1)
    lengths = [int(s.shape[1]) for s in sessions]
    y = torch.cat(sessions, dim=1)
    if show_progress is None:
        show_progress = y.shape[1] >= 2000

    # Restarts: short VB from each seed, keep the best free energy.
    best_state: VBState | None = None
    best_obj = -float("inf")
    restart_scores: list[float] = []
    restart_bar = tqdm(
        range(n_init), desc="bsds restarts", leave=True, disable=not show_progress or n_init == 1
    )
    for r in restart_bar:
        state = init_state(
            y,
            lengths,
            n_states,
            ldim,
            seed=seed + r,
            device=device,
            n_kmeans_replicates=n_kmeans_replicates,
            kmeans_pca_dim=kmeans_pca_dim,
        )
        state, hist, _, _ = _run_vb(
            state,
            y,
            sessions,
            n_init_iter,
            tol,
            show_progress=False,
            desc="init",
            criterion=criterion,
            obj_every=obj_every,
        )
        restart_scores.append(hist[-1])
        if hist[-1] > best_obj:
            best_obj = hist[-1]
            best_state = state
        if show_progress and n_init > 1:
            restart_bar.set_postfix(best=f"{best_obj:.3f}")
    assert best_state is not None

    # Continue the best restart to convergence.
    state, history, converged, weights_history = _run_vb(
        best_state,
        y,
        sessions,
        n_iter,
        tol,
        show_progress,
        desc="bsds fit",
        criterion=criterion,
        obj_every=obj_every,
    )
    model = _finalize(state, y, sessions)
    model.objective_history = history
    model.weights_history = weights_history
    model.restart_scores = restart_scores
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

    # Seed responsibilities from the *group* emissions, not k-means: the point of
    # warm-starting is that this session's states are already identified by the
    # group loadings, and the first coordinate-ascent step (update_ql) reads
    # state.qns before the first E-step would otherwise get to run. Using a
    # k-means partition here can pull the warm-started loadings away from the
    # group before the informative prior gets a chance to matter.
    vb.update_qx(state, y)
    log_obs = vb.compute_log_out_probs(state, [sess])
    gammas, _xi, _g0, _ll = hmm.estep(log_obs, state.wa, state.wpi)
    state.qns = torch.cat(gammas, dim=0)

    state, history, converged, weights_history = _run_vb(
        state, y, [sess], n_iter, tol, show_progress, desc="bsds subject"
    )
    model = _finalize(state, y, [sess])
    model.objective_history = history
    model.weights_history = weights_history
    model.converged = converged
    return model


@dataclass
class DecodeResult:
    """State time courses from applying a fitted model to new data (no refit)."""

    n_states: int
    responsibilities: list[torch.Tensor]  # per session (T_i, K)
    viterbi_states: list[torch.Tensor]  # per session (T_i,) MAP path
    loglik: float


def decode(
    model: BSDSModel,
    sessions: list[torch.Tensor],
    *,
    device: torch.device | None = None,
) -> DecodeResult:
    """Apply a fitted model to new sessions with parameters held fixed.

    Runs only the inference E-step — recompute the latent for the new data given
    the fixed loadings, then the emissions and HMM forward-backward — so it yields
    state responsibilities and a MAP path on data the model was never fit on. This
    is how you apply a trained model to held-out runs, and how cross-fit temporal
    matching gets both models onto the same time axis (reference
    ``computeQnsFromGivenNetForNewData``).
    """
    g = model.state
    device = g.lm.device if device is None else device
    sessions = [s.to(device=device, dtype=_DTYPE) for s in sessions]
    if sessions[0].shape[0] != g.n_roi:
        raise ValueError(f"new data has D={sessions[0].shape[0]} but the model expects D={g.n_roi}")
    lengths = [int(s.shape[1]) for s in sessions]
    y = torch.cat(sessions, dim=1)

    # A state carrying the fitted parameters but the new data's time axis.
    state = VBState(
        n_states=g.n_states,
        n_roi=g.n_roi,
        ldim=g.ldim,
        n_time=int(y.shape[1]),
        session_lengths=lengths,
        lm=g.lm,
        lcov=g.lcov,
        xm=torch.ones(g.n_states, g.kt, y.shape[1], dtype=_DTYPE, device=device),
        xcov=torch.zeros(g.n_states, g.kt, g.kt, dtype=_DTYPE, device=device),
        psii=g.psii,
        a=g.a,
        b=g.b,
        mean_mcl=g.mean_mcl,
        nu_mcl=g.nu_mcl,
        wa=g.wa,
        wpi=g.wpi,
        qns=torch.zeros(int(y.shape[1]), g.n_states, dtype=_DTYPE, device=device),
    )
    vb.update_qx(state, y)  # latent posterior for the new data under fixed loadings
    log_obs = vb.compute_log_out_probs(state, sessions)
    gammas, _xi, _g0, loglik = hmm.estep(log_obs, state.wa, state.wpi)
    trans = state.wa / state.wa.sum(dim=1, keepdim=True)
    init = state.wpi / state.wpi.sum()
    viterbi = [hmm.viterbi(lo, trans, init) for lo in log_obs]
    return DecodeResult(
        n_states=g.n_states,
        responsibilities=gammas,
        viterbi_states=viterbi,
        loglik=float(loglik),
    )
