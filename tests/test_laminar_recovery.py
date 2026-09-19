"""
Tests for the parameter-recovery harness (M4).

These are structural: that the simulator produces what it claims, that the noise
level is derived rather than invented, and that the falsification machinery
would actually register a failure. The recovery *results* are an experiment, not
a test -- they live in the notebook and the wiki, because their answer depends
on the data and can legitimately change.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.laminar.integrate import build_input
from fastfuncstuff.laminar.params import ModelSpec, zero_params
from fastfuncstuff.laminar.recovery import (
    RecoveryResult,
    _true_params,
    noise_sd_from_hyperparameters,
    simulate_laminar_data,
)

SPEC = ModelSpec(N=3, K=6, n_inputs=2, n_mod=1)
N_SCANS = 12


@pytest.fixture(scope="module")
def params():
    P = zero_params(SPEC)
    P["C"] = torch.tensor([[0.0, 1.0]] * SPEC.N, dtype=SPEC.dtype)
    return P


@pytest.fixture(scope="module")
def inputs():
    n_micro = int(round(N_SCANS * SPEC.TR / SPEC.dt))
    return build_input(SPEC, [[], [2.0]], [[], [2.0]], n_micro)


def test_noise_sd_inverts_the_precision_not_the_variance():
    """SPM's hyperparameters scale *precision*, so sd is exp(-h/2).

    Bug of record: ``sqrt(exp(Eh))`` on the published ROI gives ~13% signal
    change per depth, against data whose own standard deviation is ~0.62%. A
    simulation built on that would drown every effect and report that nothing is
    recoverable -- a wrong conclusion that looks like a finding.
    """
    Eh = torch.tensor([5.046, 5.247, 4.912], dtype=torch.float64)
    sd = noise_sd_from_hyperparameters(Eh)
    assert torch.allclose(sd, torch.exp(-Eh / 2))
    # The realistic range for this dataset: well under the data's own spread.
    assert float(sd.max()) < 0.12
    assert float(sd.min()) > 0.05
    # Higher hyperparameter means higher precision, so *less* noise.
    assert sd[1] < sd[0] < sd[2]


def test_simulated_data_is_the_prediction_plus_noise(params, inputs):
    y, clean = simulate_laminar_data(SPEC, params, inputs, N_SCANS, noise_sd=0.0, generator=None)
    assert torch.allclose(y, clean), "zero noise must return the prediction exactly"
    assert y.shape == (N_SCANS, SPEC.K)


def test_simulated_noise_has_the_requested_per_depth_scale(params, inputs):
    """Per-depth noise, because the error model is per depth."""
    sd = torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    gen = torch.Generator().manual_seed(0)
    y, clean = simulate_laminar_data(SPEC, params, inputs, N_SCANS, noise_sd=sd, generator=gen)
    resid = (y - clean).numpy()
    assert np.abs(resid[:, 1:]).max() == 0.0, "noise leaked into zero-sd depths"
    assert resid[:, 0].std() > 0.1


def test_simulation_is_reproducible_and_seed_dependent(params, inputs):
    def draw(seed):
        gen = torch.Generator().manual_seed(seed)
        return simulate_laminar_data(SPEC, params, inputs, N_SCANS, noise_sd=0.1, generator=gen)[0]

    assert torch.equal(draw(0), draw(0))
    assert not torch.equal(draw(0), draw(1))


def test_true_params_impose_the_pattern_on_the_right_parameter(params):
    """``on="B"`` is the two-condition hypothesis, ``on="C"`` the one-condition."""
    b = _true_params(SPEC, params, (1, 0, 1), 2.0, on="B")
    assert torch.allclose(
        torch.diagonal(b["B"][0]), torch.tensor([2.0, 0.0, 2.0], dtype=torch.float64)
    )
    assert torch.equal(b["C"], params["C"]), "B-mode must not disturb the drive"

    c = _true_params(SPEC, params, (1, 0, 1), 2.0, on="C")
    assert torch.allclose(c["C"][:, -1], torch.tensor([2.0, 0.0, 2.0], dtype=torch.float64))
    assert torch.all(c["B"] == 0), "C-mode must not introduce modulation"

    with pytest.raises(ValueError, match="on must be"):
        _true_params(SPEC, params, (1, 0, 0), 1.0, on="A")


def test_true_params_does_not_mutate_the_base(params):
    before = {k: v.clone() for k, v in params.items()}
    _true_params(SPEC, params, (1, 1, 1), 9.0, on="B")
    for k, v in before.items():
        assert torch.equal(params[k], v), f"{k} was mutated"


def test_recovery_result_reports_correctness_and_margin():
    r = RecoveryResult(
        true_index=2,
        won_index=2,
        F=[-10.0, -8.0, -5.0, -20.0],
        probs=[0.0, 0.0, 1.0, 0.0],
        names=["a", "b", "c", "d"],
    )
    assert r.correct and r.margin == 0.0

    wrong = RecoveryResult(
        true_index=1,
        won_index=2,
        F=[-10.0, -8.0, -5.0, -20.0],
        probs=[0.0, 0.0, 1.0, 0.0],
        names=["a", "b", "c", "d"],
    )
    assert not wrong.correct
    # Three nats of evidence for the wrong answer: a confident error.
    assert wrong.margin == pytest.approx(3.0)
