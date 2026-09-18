"""
Tests for model-based deveining.

These check the *direction* of the physics, not parity: the ascending vein makes
the neuronal -> BOLD map triangular and delayed, and those two facts are what
decide whether a static per-depth correction is defensible. If they ever stop
holding, a scalar deveining factor silently becomes the right answer for the
wrong reason.
"""

from __future__ import annotations

import pytest
import torch

from fastfuncstuff.laminar.devein import (
    apply_deveining,
    deveined_timecourses,
    deveining_fidelity,
    laminar_impulse_response,
    static_deveining_matrix,
)
from fastfuncstuff.laminar.integrate import build_input
from fastfuncstuff.laminar.params import ModelSpec, zero_params

# Small and short: these test structure, and every one of them integrates.
SPEC = ModelSpec(N=3, K=6, n_inputs=2, n_mod=1)
N_SCANS = 12


@pytest.fixture(scope="module")
def impulse():
    P = zero_params(SPEC)
    P["C"] = torch.tensor([[0.0, 1.0]] * SPEC.N, dtype=SPEC.dtype)
    return laminar_impulse_response(SPEC, P, N_SCANS, amplitude=1.0)


def test_impulse_response_shape_is_output_by_input_by_time(impulse):
    assert impulse.shape == (SPEC.K, SPEC.N, N_SCANS)


def test_drainage_is_superficial_and_the_operator_is_triangular(impulse):
    """Deep neuronal activity must reach superficial BOLD, but not vice versa.

    This is the asymmetry the whole deveining argument rests on: depth 0 is the
    CSF end, blood flows from high index to low, so the *deepest* drive should
    contaminate superficial depths far more than the most superficial drive
    reaches deep ones.
    """
    W, _ = static_deveining_matrix(impulse, mode="auc")
    superficial, deep = 0, SPEC.K - 1
    drive_deep, drive_superficial = SPEC.N - 1, 0
    leak_up = W[superficial, drive_deep].abs()
    leak_down = W[deep, drive_superficial].abs()
    assert leak_up > leak_down, (
        "deep activity must leak into superficial BOLD more than the reverse; "
        f"got up={float(leak_up):.4g} down={float(leak_down):.4g}"
    )


def test_the_deepest_depth_is_clean_and_the_surface_is_not(impulse):
    """The asymmetry deveining exists to undo.

    Every depth below drains through the superficial one, so it collects
    cross-laminar mass from below; nothing drains downward, so the deepest depth
    is essentially its own neurons alone. Measured on the fitted operator, not
    assumed.
    """
    d = deveining_fidelity(impulse, mode="auc")
    asym = d["drainage_asymmetry"]
    assert float(asym[0]) > 0.3, "surface should be heavily contaminated from below"
    assert float(asym[-1]) == pytest.approx(0.0, abs=0.02), "deepest depth should be clean"
    # Superficial half contaminated more than deep half.
    half = SPEC.K // 2
    assert float(asym[:half].mean()) > float(asym[half:].mean())


def test_drainage_is_one_way(impulse):
    """Mass arrives from below, never from above.

    The symmetric neuronal->vascular basis puts a little mass on the shallower
    side at a depth boundary; a vein puts none there at all. So from_shallower
    stays at the basis floor while from_deeper is large.
    """
    d = deveining_fidelity(impulse, mode="auc")
    assert float(d["from_shallower"].max()) < 0.05
    assert float(d["from_deeper"].max()) > 0.3


def test_forward_mixing_is_strictly_triangular(impulse):
    """The deepest vascular depth cannot see the most superficial neurons.

    An exact zero, not a small number: there is no path from a superficial
    neuronal depth down to a deep vein.
    """
    W, _ = static_deveining_matrix(impulse, mode="auc")
    assert float(W[-1, 0].abs()) == pytest.approx(0.0, abs=1e-12)
    assert float(W[0, -1].abs()) > 0.05


def test_transit_imposes_a_lag_spread(impulse):
    """A scalar correction cannot represent a delay. Assert there is one to
    represent: contributions to a depth must not all peak on the same scan."""
    spread = deveining_fidelity(impulse, mode="auc")["lag_spread"]
    assert float(spread.max()) > 0, "no transit delay found; a static operator would be exact"


def test_peak_and_auc_disagree_because_the_vein_smears(impulse):
    """The two summaries coincide only for an undistorted response, so their
    disagreement is a readout of how much shape the vein imposes."""
    w_peak, _ = static_deveining_matrix(impulse, mode="peak")
    w_auc, _ = static_deveining_matrix(impulse, mode="auc")
    rel = (w_peak - w_auc).abs().max() / w_auc.abs().max()
    assert float(rel) > 1e-3


def test_deveining_operator_inverts_the_forward_mixing(impulse):
    """W_inv @ W must be the identity on neuronal depths: K > N, so the tall
    system is over-determined and the pseudo-inverse is exact in that direction.
    """
    W, W_inv = static_deveining_matrix(impulse, mode="auc")
    eye = W_inv @ W
    assert torch.allclose(eye, torch.eye(SPEC.N, dtype=eye.dtype), atol=1e-8)


def test_apply_deveining_recovers_known_neuronal_amplitudes(impulse):
    """End to end on the amplitude path: push known neuronal amplitudes through
    the forward mixing, devein, and get them back."""
    W, W_inv = static_deveining_matrix(impulse, mode="auc")
    truth = torch.tensor([0.3, 1.0, 2.0], dtype=W.dtype)
    measured = truth @ W.transpose(-1, -2)  # (K,)
    assert torch.allclose(apply_deveining(measured, W_inv), truth, atol=1e-8)


def test_apply_deveining_is_batched_over_leading_axes(impulse):
    """Parcels and timepoints are leading axes, so the operator must broadcast."""
    _, W_inv = static_deveining_matrix(impulse, mode="auc")
    values = torch.randn(5, 7, SPEC.K, dtype=W_inv.dtype)
    assert apply_deveining(values, W_inv).shape == (5, 7, SPEC.N)


def test_apply_deveining_rejects_a_depth_mismatch(impulse):
    _, W_inv = static_deveining_matrix(impulse, mode="auc")
    with pytest.raises(ValueError, match="depths"):
        apply_deveining(torch.zeros(SPEC.K + 1, dtype=W_inv.dtype), W_inv)


def test_deveined_timecourses_returns_aligned_axes():
    P = zero_params(SPEC)
    P["C"] = torch.tensor([[0.0, 1.0]] * SPEC.N, dtype=SPEC.dtype)
    n_micro = int(round(N_SCANS * SPEC.TR / SPEC.dt))
    u = build_input(SPEC, [[], [1.0]], [[], [2.0]], n_micro)
    out = deveined_timecourses(SPEC, P, u, N_SCANS)
    assert out.y_predicted.shape == (N_SCANS, SPEC.K)
    assert out.neuronal.shape == (N_SCANS, SPEC.N)
    assert out.cbf.shape == (N_SCANS, SPEC.N)
    assert out.n_scans == N_SCANS
    # Neuronal activity must lead the BOLD it causes.
    assert (
        int(out.neuronal.abs().argmax()) // SPEC.N <= int(out.y_predicted.abs().argmax()) // SPEC.K
    )


def test_static_deveining_rejects_an_unknown_mode(impulse):
    with pytest.raises(ValueError, match="mode must be"):
        static_deveining_matrix(impulse, mode="median")
