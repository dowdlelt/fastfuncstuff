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

import os

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


def forward_backward_batched(
    log_obs: torch.Tensor,
    log_a: torch.Tensor,
    log_pi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward-backward for a batch of equal-length sessions in one pass.

    ``log_obs`` is ``(B, T, K)`` (``B`` sessions of the same length ``T``). This
    is the identical recursion as :func:`forward_backward`, vectorised over the
    session axis so the sequential time loop runs once for the whole batch
    instead of once per session — the sequential-tiny-op dispatch overhead
    (dominant on CPU) is amortised across all sessions.

    Returns ``(gamma (B, T, K), xi_sum (K, K), loglik_sum scalar,
    gamma0_sum (K,))`` — the pairwise/initial statistics already summed over the
    batch (their consumers only need the totals).
    """
    b, t_len, k = log_obs.shape
    log_alpha = torch.empty_like(log_obs)
    log_alpha[:, 0] = log_pi + log_obs[:, 0]
    for t in range(1, t_len):
        # alpha[b, t, j] = obs[b, t, j] + logsumexp_i(alpha[b, t-1, i] + A[i, j])
        prev = log_alpha[:, t - 1].unsqueeze(2) + log_a  # (B, K_i, K_j)
        log_alpha[:, t] = log_obs[:, t] + torch.logsumexp(prev, dim=1)
    loglik = torch.logsumexp(log_alpha[:, -1], dim=1)  # (B,)

    log_beta = torch.zeros_like(log_obs)
    for t in range(t_len - 2, -1, -1):
        # beta[b, t, i] = logsumexp_j(A[i, j] + obs[b, t+1, j] + beta[b, t+1, j])
        nxt = log_a + (log_obs[:, t + 1] + log_beta[:, t + 1]).unsqueeze(1)  # (B, K_i, K_j)
        log_beta[:, t] = torch.logsumexp(nxt, dim=2)

    gamma = torch.softmax(log_alpha + log_beta, dim=2)  # (B, T, K)

    if t_len > 1:
        log_xi = (
            log_alpha[:, :-1].unsqueeze(3) + log_a + (log_obs[:, 1:] + log_beta[:, 1:]).unsqueeze(2)
        )  # (B, T-1, K_i, K_j)
        log_xi = log_xi - torch.logsumexp(log_xi.reshape(b, t_len - 1, -1), dim=2).view(
            b, t_len - 1, 1, 1
        )
        xi_sum = log_xi.exp().sum(dim=(0, 1))  # (K, K)
    else:
        xi_sum = torch.zeros(k, k, dtype=log_obs.dtype, device=log_obs.device)
    return gamma, xi_sum, loglik.sum(), gamma[:, 0].sum(dim=0)


# --- CUDA-graph acceleration of the batched forward-backward ----------------- #
# At fMRI sequence lengths (T~390) the forward-backward is ~2*(T-1) *sequential*
# tiny kernels, so on GPU the per-kernel launch overhead — not the arithmetic —
# dominates (measured ~7x: 132ms eager -> 18ms graphed). The shapes are fixed
# across VB iterations (the same sessions every pass), so we capture one CUDA
# graph per length-group shape and replay it: copy the fresh log_obs/log_a/log_pi
# into the graph's static input buffers, replay (one launch for the whole
# recursion), and clone the outputs (the graph reuses that memory next replay).
# Replay runs the identical kernels, so results are bit-for-bit the eager ones.

_FB_GRAPH_CACHE: dict = {}
_FB_GRAPH_DISABLED = False


def _cudagraph_enabled(device: torch.device) -> bool:
    return (
        not _FB_GRAPH_DISABLED
        and device.type == "cuda"
        and os.environ.get("FFS_BSDS_CUDAGRAPH", "1") != "0"
    )


def clear_forward_backward_graph_cache() -> None:
    """Drop all captured forward-backward CUDA graphs (frees their static buffers).

    Shapes are keyed per length-group, so a long grid search over different
    ``n_states``/session subsets accumulates one graph per distinct shape; call
    this between unrelated fits to release the VRAM.
    """
    _FB_GRAPH_CACHE.clear()


def _capture_forward_backward(
    stack: torch.Tensor, log_a: torch.Tensor, log_pi: torch.Tensor
) -> dict:
    """Warm up on a side stream (required before capture), then capture the graph."""
    in_obs, in_a, in_pi = stack.clone(), log_a.clone(), log_pi.clone()
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(3):
            forward_backward_batched(in_obs, in_a, in_pi)
    torch.cuda.current_stream().wait_stream(warm)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = forward_backward_batched(in_obs, in_a, in_pi)
    return {"graph": graph, "in_obs": in_obs, "in_a": in_a, "in_pi": in_pi, "out": out}


def _forward_backward_graphed(
    stack: torch.Tensor, log_a: torch.Tensor, log_pi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Capture-once / replay CUDA-graph wrapper around :func:`forward_backward_batched`.

    Falls back to eager permanently (for the whole process) if capture ever
    raises — a bad capture must not wedge the fit.
    """
    global _FB_GRAPH_DISABLED
    key = (tuple(stack.shape), stack.dtype, stack.device)
    entry = _FB_GRAPH_CACHE.get(key)
    if entry is None:
        try:
            entry = _capture_forward_backward(stack, log_a, log_pi)
        except Exception:
            _FB_GRAPH_DISABLED = True
            return forward_backward_batched(stack, log_a, log_pi)
        _FB_GRAPH_CACHE[key] = entry
    entry["in_obs"].copy_(stack)
    entry["in_a"].copy_(log_a)
    entry["in_pi"].copy_(log_pi)
    entry["graph"].replay()
    gamma, xi_sum, loglik, gamma0 = entry["out"]
    # The graph overwrites these on the next replay of this shape; clone what escapes.
    return gamma.clone(), xi_sum.clone(), loglik.clone(), gamma0.clone()


def estep(
    session_log_obs: list[torch.Tensor],
    wa: torch.Tensor,
    wpi: torch.Tensor,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the E-step over all sessions.

    Returns per-session ``gamma`` responsibilities, the transition and initial
    sufficient statistics ``(xi_sum (K,K), gamma0_sum (K,))`` needed to update
    the Dirichlet posteriors, and the total evidence ``loglik``.

    Sessions are grouped by length and each length-group is run as one batched
    forward-backward pass (see :func:`forward_backward_batched`); with fMRI runs
    that are nearly all the same length this collapses ~N sequential time-loops
    into one or two, the single biggest CPU speedup in the VB iteration.
    """
    log_a = expected_log_transition(wa)
    log_pi = expected_log_init(wpi)
    k = wa.shape[0]
    dtype, device = wa.dtype, wa.device

    groups: dict[int, list[int]] = {}
    for i, lo in enumerate(session_log_obs):
        groups.setdefault(int(lo.shape[0]), []).append(i)

    n = len(session_log_obs)
    zero_gamma = torch.zeros(0, k, dtype=dtype, device=device)
    gammas: list[torch.Tensor] = [zero_gamma] * n
    xi_total = torch.zeros(k, k, dtype=dtype, device=device)
    gamma0_total = torch.zeros(k, dtype=dtype, device=device)
    loglik = torch.zeros((), dtype=dtype, device=device)
    fb = _forward_backward_graphed if _cudagraph_enabled(device) else forward_backward_batched
    for idxs in groups.values():
        stack = torch.stack([session_log_obs[i].to(dtype) for i in idxs], dim=0)  # (B, T, K)
        gamma, xi_sum, ll, gamma0 = fb(stack, log_a, log_pi)
        for pos, i in enumerate(idxs):
            gammas[i] = gamma[pos]
        xi_total = xi_total + xi_sum
        gamma0_total = gamma0_total + gamma0
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
