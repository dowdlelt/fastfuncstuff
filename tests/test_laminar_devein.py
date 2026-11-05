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
def params():
    P = zero_params(SPEC)
    P["C"] = torch.tensor([[0.0, 1.0]] * SPEC.N, dtype=SPEC.dtype)
    return P


@pytest.fixture(scope="module")
def impulse(params):
    return laminar_impulse_response(SPEC, params, N_SCANS, amplitude=1.0)


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


def test_the_deepest_depth_is_clean_and_the_surface_is_not(impulse, params):
    """The asymmetry deveining exists to undo.

    Every depth below drains through the superficial one, so it sources from
    deeper than the depth mapping alone would give; nothing drains downward, so
    the deepest depth sits exactly on the basis. Measured against the computed
    basis, not against an assumed "own depth".
    """
    d = deveining_fidelity(impulse, SPEC, params, mode="auc")
    shift = d["drainage_shift"]
    assert float(shift[0]) > 0.3, "surface should source well below its own depth"
    assert float(shift[-1]) == pytest.approx(0.0, abs=0.02), "deepest depth should be clean"
    half = SPEC.K // 2
    assert float(shift[:half].mean()) > float(shift[half:].mean())


def test_drainage_is_one_way(impulse, params):
    """No vascular depth sources *shallower* than the basis predicts.

    A vein can only move the centre of mass deeper. A negative shift anywhere
    would mean the mixing draws from above, which has no physical path.
    """
    shift = deveining_fidelity(impulse, SPEC, params, mode="auc")["drainage_shift"]
    assert float(shift.min()) > -1e-6, f"negative drainage shift: {shift.tolist()}"


def test_forward_mixing_is_strictly_triangular(impulse):
    """The deepest vascular depth cannot see the most superficial neurons.

    An exact zero, not a small number: there is no path from a superficial
    neuronal depth down to a deep vein.
    """
    W, _ = static_deveining_matrix(impulse, mode="auc")
    assert float(W[-1, 0].abs()) == pytest.approx(0.0, abs=1e-12)
    assert float(W[0, -1].abs()) > 0.05


def test_transit_imposes_a_lag_spread(impulse, params):
    """A scalar correction cannot represent a delay. Assert there is one to
    represent: contributions to a depth must not all peak on the same scan."""
    spread = deveining_fidelity(impulse, SPEC, params, mode="auc")["lag_spread"]
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


@pytest.mark.slow  # 19s
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


def test_uniform_drive_gives_sloped_bold_that_deveining_flattens():
    """The claim, end to end: uniform neurons, sloped BOLD, flat again after.

    Drive every neuronal depth equally. The draining vein turns that into a BOLD
    profile ramping steeply toward the surface -- the bias this whole module
    exists to undo. Applying the operator must recover the flat profile.

    Not circular in the way it looks: the operator is built from *impulse*
    responses while the measured curve comes from a *boxcar* run, so this also
    checks the static reduction survives a change of stimulus shape.
    """
    P = zero_params(SPEC)
    P["C"] = torch.tensor([[0.0, 1.0]] * SPEC.N, dtype=SPEC.dtype)
    n_micro = int(round(20 * SPEC.TR / SPEC.dt))
    u = build_input(SPEC, [[], [2.0]], [[], [2.0]], n_micro)
    dev = deveined_timecourses(SPEC, P, u, 20)

    measured = dev.y_predicted[1:14].max(dim=0).values  # (K,)
    # The vein imposed a real gradient: surface well above the deepest depth.
    assert float(measured[0] / measured[-1]) > 1.5, "no draining-vein ramp to undo"

    ir = laminar_impulse_response(SPEC, P, 20, amplitude=1.0)
    _, W_inv = static_deveining_matrix(ir, mode="auc")
    deveined = apply_deveining(measured, W_inv)

    spread = float(deveined.max() - deveined.min()) / float(deveined.mean().abs())
    assert spread < 0.15, f"deveined profile should be flat for uniform drive, spread={spread:.3f}"


@pytest.mark.slow  # 28s
def test_drainage_shift_is_basis_referenced_at_every_k():
    """Bug of record: the contamination metric was wrong twice, the same way.

    Summing off-diagonal mass conflates the symmetric neuronal->vascular basis
    with directional drainage. Assigning each vascular depth an "own" neuronal
    depth and measuring asymmetry about it fails too: the symmetric floor is not
    zero and varies with k, so a depth whose centre falls between two neuronal
    centres scored as the most veined in the ROI. No binning fixes it -- the
    basis has to be computed and subtracted.

    Checked at three K because the earlier versions passed at K=6 and failed at
    K=7 and K=9.
    """
    for K in (6, 7, 9):
        spec = ModelSpec(N=3, K=K, n_inputs=2, n_mod=1)
        P = zero_params(spec)
        P["C"] = torch.tensor([[0.0, 1.0]] * spec.N, dtype=spec.dtype)
        ir = laminar_impulse_response(spec, P, 16, amplitude=1.0)
        shift = deveining_fidelity(ir, spec, P, mode="auc")["drainage_shift"]
        assert float(shift[-1]) == pytest.approx(0.0, abs=1e-3), f"K={K} deepest not clean"
        assert float(shift.argmax()) < K / 2, f"K={K} contamination must peak superficially"
        assert float(shift.min()) > -1e-6, f"K={K} drainage must be one-way"


def test_depth_bin_map_groups_vascular_depths_evenly():
    from fastfuncstuff.laminar.devein import depth_bin_map

    assert list(depth_bin_map(9, 3)) == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    # K not a multiple of N still covers every bin and stays monotone
    m = depth_bin_map(7, 3)
    assert list(m) == sorted(m) and set(m) == {0, 1, 2}
    assert depth_bin_map(11, 3)[0] == 0 and depth_bin_map(11, 3)[-1] == 2


def test_local_contribution_keeps_percent_and_drops_the_drained_part():
    """The deveined profile is in PERCENT, not a.u., and removes exactly the signal
    drained in from below -- the claim the profile panel makes on one shared axis."""
    import numpy as np

    from fastfuncstuff.laminar.devein import depth_bin_map, local_contribution

    k, n = 9, 3
    own = depth_bin_map(k, n)
    w = np.zeros((k, n))
    w[np.arange(k), own] = 1.0  # each depth's own neurons, 1% per unit drive
    w[:3, 1] = 0.7  # superficial depths also drain the middle ...
    w[:3, 2] = 0.5  # ... and the deep bin
    drive = np.ones(n)

    measured = w @ drive  # 2.2 at pial, 1.0 at deep
    deveined = local_contribution(w, measured)

    assert np.allclose(deveined, 1.0)  # flat: the planted truth
    assert measured[0] / measured[-1] == pytest.approx(2.2)  # a pial ramp before
    assert deveined[0] / deveined[-1] == pytest.approx(1.0)  # and none after
    assert np.allclose(measured - deveined, [1.2, 1.2, 1.2, 0, 0, 0, 0, 0, 0])


def test_local_contribution_leaves_the_deepest_depth_alone():
    """The built-in sanity check, and the one that caught a units bug: the deepest row
    of W has only its own column, so nothing there can be attributed to draining.

    Scaling W by the fitted NEURONAL STATES instead of the drive broke exactly this --
    W is BOLD per unit drive, the states are the ODE's response to it in arbitrary
    units -- and showed 98% of the deepest depth's signal as drained.
    """
    import numpy as np

    from fastfuncstuff.laminar.devein import local_contribution

    rng = np.random.default_rng(0)
    k, n = 9, 3
    w = np.zeros((k, n))
    for i in range(k):  # near-triangular, as a real fit's W is
        for j in range(n):
            w[i, j] = rng.uniform(0.3, 1.2) if j >= (i * n) // k else 0.0
    assert w[-1, 0] == 0.0 and w[-1, 1] == 0.0 and w[-1, 2] > 0  # deepest: own only

    measured = w @ np.array([1.0, 1.3, 0.8])
    deveined = local_contribution(w, measured)
    assert deveined[-1] == pytest.approx(measured[-1], rel=1e-9)


def test_local_contribution_reconstructs_the_measured_profile():
    """Keeping EVERY column returns the measured profile, which is what makes the
    own-column part a decomposition rather than an arbitrary rescaling."""
    import numpy as np

    from fastfuncstuff.laminar.devein import local_contribution

    rng = np.random.default_rng(3)
    w = np.abs(rng.normal(size=(11, 3))) + 0.1
    measured = w @ np.array([0.9, 1.4, 1.1])
    drive = np.linalg.pinv(w) @ measured
    assert np.allclose(w @ drive, measured, atol=1e-10)
    # and the own-column part is never more than the whole
    dev = local_contribution(w, measured)
    assert np.all(dev <= measured + 1e-9)


def _event_input(spec, n_scans):
    n_micro = int(round(n_scans * spec.TR / spec.dt))
    u = torch.zeros(n_micro, spec.n_inputs, dtype=spec.dtype)
    u[: int(round(1.0 / spec.dt)), -1] = 1.0
    return u


def test_own_depth_responses_split_the_fit_by_driven_depth(params):
    """Each part is the fit with one depth's input left on, and they roughly add back up:
    the attribution is only meaningful if it accounts for what the model predicted."""
    from fastfuncstuff.laminar.devein import own_depth_responses
    from fastfuncstuff.laminar.integrate import integrate

    u = _event_input(SPEC, N_SCANS)
    parts = own_depth_responses(SPEC, params, u, N_SCANS)
    assert parts.shape == (SPEC.N, N_SCANS, SPEC.K)
    full = integrate(u, params, SPEC, N_SCANS)
    assert torch.allclose(parts.sum(0), full, rtol=0.2, atol=0.05)

    only_deep = {k: v.clone() for k, v in params.items()}
    only_deep["C"][:-1] = 0.0
    alone = integrate(u, only_deep, SPEC, N_SCANS)
    assert torch.allclose(parts[-1], alone, rtol=1e-6, atol=1e-6)


def test_uniform_drive_gives_the_middle_depth_its_own_signal(params):
    """The bug of record. V1 at K=7 showed the middle depth with a third of the deep
    depth's own signal, because the static inverse reads a straight profile as having no
    middle. With every depth driven equally, the attribution must not produce that dip."""
    from fastfuncstuff.laminar.devein import depth_bin_map, own_depth_responses

    u = _event_input(SPEC, N_SCANS)
    parts = own_depth_responses(SPEC, params, u, N_SCANS)
    own = depth_bin_map(SPEC.K, SPEC.N)
    peak = parts[torch.as_tensor(own), :, torch.arange(SPEC.K)].abs().max(dim=-1).values
    assert peak[own == 1].mean() > 0.7 * peak[own == 2].mean()
