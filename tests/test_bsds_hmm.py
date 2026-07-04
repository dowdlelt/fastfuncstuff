"""HMM engine tests: forward-backward and Viterbi vs brute-force enumeration."""

from __future__ import annotations

import itertools

import numpy as np
import torch

from fastfuncstuff.dynamics.bsds.hmm import (
    expected_log_init,
    expected_log_transition,
    forward_backward,
    viterbi,
)


def _brute_force(pi, a, b):
    """Exact gamma, xi_sum, loglik, and MAP path by enumerating all paths."""
    t_len, k = b.shape
    total = 0.0
    gamma = np.zeros((t_len, k))
    xi = np.zeros((k, k))
    best_w, best_path = -1.0, None
    for path in itertools.product(range(k), repeat=t_len):
        w = pi[path[0]] * b[0, path[0]]
        for t in range(1, t_len):
            w *= a[path[t - 1], path[t]] * b[t, path[t]]
        total += w
        for t in range(t_len):
            gamma[t, path[t]] += w
        for t in range(t_len - 1):
            xi[path[t], path[t + 1]] += w
        if w > best_w:
            best_w, best_path = w, path
    return gamma / total, xi / total, np.log(total), np.array(best_path)


def test_forward_backward_matches_brute_force():
    rng = np.random.default_rng(0)
    k, t_len = 3, 5
    pi = rng.dirichlet(np.ones(k))
    a = np.stack([rng.dirichlet(np.ones(k)) for _ in range(k)])
    b = rng.uniform(0.05, 1.0, size=(t_len, k))  # emission likelihoods

    g_bf, xi_bf, ll_bf, _ = _brute_force(pi, a, b)

    log_obs = torch.tensor(np.log(b))
    gamma, xi_sum, loglik = forward_backward(
        log_obs, torch.tensor(np.log(a)), torch.tensor(np.log(pi))
    )
    np.testing.assert_allclose(gamma.numpy(), g_bf, atol=1e-10)
    np.testing.assert_allclose(xi_sum.numpy(), xi_bf, atol=1e-10)
    np.testing.assert_allclose(loglik.item(), ll_bf, atol=1e-10)


def test_viterbi_matches_brute_force():
    rng = np.random.default_rng(1)
    k, t_len = 4, 7
    pi = rng.dirichlet(np.ones(k))
    a = np.stack([rng.dirichlet(np.ones(k)) for _ in range(k)])
    b = rng.uniform(0.05, 1.0, size=(t_len, k))
    _, _, _, path_bf = _brute_force(pi, a, b)

    path = viterbi(torch.tensor(np.log(b)), torch.tensor(a), torch.tensor(pi))
    assert np.array_equal(path.numpy(), path_bf)


def test_single_timepoint_no_xi():
    log_obs = torch.log(torch.tensor([[0.3, 0.7]], dtype=torch.float64))
    log_a = torch.log(torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float64))
    log_pi = torch.log(torch.tensor([0.4, 0.6], dtype=torch.float64))
    gamma, xi_sum, loglik = forward_backward(log_obs, log_a, log_pi)
    # Posterior at the single step is proportional to pi * b.
    expected = np.array([0.4 * 0.3, 0.6 * 0.7])
    expected /= expected.sum()
    np.testing.assert_allclose(gamma.numpy()[0], expected, atol=1e-12)
    assert xi_sum.abs().sum().item() == 0.0


def test_expected_log_transition_digamma_identity():
    wa = torch.tensor([[5.0, 1.0], [2.0, 3.0]], dtype=torch.float64)
    log_a = expected_log_transition(wa)
    manual = torch.digamma(wa) - torch.digamma(wa.sum(1, keepdim=True))
    torch.testing.assert_close(log_a, manual)
    wpi = torch.tensor([4.0, 6.0], dtype=torch.float64)
    torch.testing.assert_close(
        expected_log_init(wpi), torch.digamma(wpi) - torch.digamma(wpi.sum())
    )
