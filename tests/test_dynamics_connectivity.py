"""Tests for per-state directed (effective) connectivity."""

from __future__ import annotations

import types

import numpy as np
import torch

from fastfuncstuff.dynamics.connectivity import per_state_directed_connectivity


def test_directed_connectivity_recovers_influence():
    # Within one state, ROI 0 at t-1 drives ROI 1 at t (B[1, 0] large), no reverse.
    rng = np.random.default_rng(0)
    d, n = 4, 3000
    x = np.zeros((d, n))
    for t in range(1, n):
        x[:, t] = rng.standard_normal(d) * 0.3
        x[1, t] += 0.8 * x[0, t - 1]
    model = types.SimpleNamespace(
        n_states=1, responsibilities=[torch.ones(n, 1, dtype=torch.float64)]
    )
    b, noise = per_state_directed_connectivity(model, [torch.tensor(x)])
    assert b.shape == (1, d, d)
    assert noise.shape == (1, d, d)
    bm = b[0].numpy()
    assert bm[1, 0] > 0.5, f"missed the driving edge: {bm[1, 0]:.2f}"
    assert abs(bm[0, 1]) < 0.3, f"spurious reverse edge: {bm[0, 1]:.2f}"


def test_directed_connectivity_alignment_check():
    model = types.SimpleNamespace(
        n_states=2, responsibilities=[torch.ones(50, 2, dtype=torch.float64)]
    )
    # Session length (30) disagrees with responsibilities (50) -> clear error.
    import pytest

    with pytest.raises(ValueError, match="do not align"):
        per_state_directed_connectivity(model, [torch.randn(4, 30, dtype=torch.float64)])
