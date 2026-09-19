"""
Parity tests for the laminar BOLD generative model.

The oracle values in ``tests/laminar_oracle/laminar_oracle.json`` were produced
by ``tests/laminar_oracle/make_oracle.m`` running the predictive_tones MATLAB
tree. Regenerate with::

    cd tests/laminar_oracle && matlab -batch "run('make_oracle.m')"

Matching predicted BOLD is the *weakest* of the gates on this model (see
``../fmri_wiki/concepts/Laminar generative model.md``), but it has to come
first: nothing downstream can be trusted if the forward map is wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.func import jvp

from fastfuncstuff.laminar.forward import f_ode, g_obs, neuronal_to_vascular
from fastfuncstuff.laminar.integrate import _dz_dx, build_input, integrate, sample_indices
from fastfuncstuff.laminar.params import P0, ModelSpec, zero_params

ORACLE = Path(__file__).parent / "laminar_oracle" / "laminar_oracle.json"


def _oracle_cases():
    if not ORACLE.exists():  # pragma: no cover - only when the dump is missing
        pytest.skip(f"no MATLAB oracle at {ORACLE}")
    return json.loads(ORACLE.read_text())["cases"]


def _driver_spec(N: int, K: int) -> ModelSpec:
    """The parameter overrides apply_laminar_BOLD_model.m applies to the priors."""
    return ModelSpec(N=N, K=K, n_inputs=2, n_mod=1, p0=P0(V0t=3.0, nr=3.0, al_v=0.35, w_v=0.5))


def _case_params(spec: ModelSpec, modulated: bool) -> dict[str, torch.Tensor]:
    P = zero_params(spec)
    P["C"] = torch.tensor([[0.0, 1.0]] * spec.N, dtype=torch.float64)
    if modulated:
        P["B"] = torch.diag(torch.tensor([0.7, 0.0, 0.0], dtype=torch.float64))[None]
        P["sigma"] = torch.tensor(0.2, dtype=torch.float64)
        P["s_d"] = torch.tensor(0.5, dtype=torch.float64)
        P["nsig"] = torch.tensor(0.3, dtype=torch.float64)
    return P


@pytest.mark.parametrize("modulated", [False, True])
def test_state_equation_matches_matlab(modulated):
    """f(x, u, P) against the generated symbolic MATLAB functions."""
    for case in _oracle_cases():
        spec = _driver_spec(case["N"], case["K"])
        x = torch.tensor(case["x"], dtype=torch.float64).squeeze()
        u = torch.tensor(case["u2" if modulated else "u"], dtype=torch.float64).squeeze()
        expect = np.asarray(case["f2" if modulated else "f"]).squeeze()
        got = f_ode(x, u, _case_params(spec, modulated), spec).numpy()
        assert np.abs(got - expect).max() < 1e-12, f"K={case['K']} modulated={modulated}"


@pytest.mark.parametrize("modulated", [False, True])
def test_observation_equation_matches_matlab(modulated):
    """The laminar BOLD signal equation plus the depth PSF."""
    for case in _oracle_cases():
        spec = _driver_spec(case["N"], case["K"])
        x = torch.tensor(case["x"], dtype=torch.float64).squeeze()
        kernel = torch.tensor(case["kernel"], dtype=torch.float64).squeeze()
        expect = np.asarray(case["g2" if modulated else "g"]).squeeze()
        got = g_obs(x, _case_params(spec, modulated), spec, kernel=kernel).numpy()
        assert np.abs(got - expect).max() < 1e-12, f"K={case['K']} modulated={modulated}"


@pytest.mark.parametrize("modulated", [False, True])
def test_reference_jacobian_product_matches_matlab(modulated):
    """The Ito-Taylor correction term, reproducing the reference's Jacobian.

    ``LBR_gen_fx_fcn.m`` differentiates against the *exponentiated* states, so
    the reference's ``dfdx`` is missing a chain-rule factor for every log-scaled
    state. Passing ``f / (dz/dx)`` as the JVP tangent reproduces it exactly. If
    this test ever starts failing while the exact-Jacobian test passes, someone
    has "fixed" the bug and broken parity with the published fits.
    """
    for case in _oracle_cases():
        spec = _driver_spec(case["N"], case["K"])
        x = torch.tensor(case["x"], dtype=torch.float64).squeeze()
        u = torch.tensor(case["u2" if modulated else "u"], dtype=torch.float64).squeeze()
        P = _case_params(spec, modulated)
        expect = np.asarray(case["dfdx2_times_f2" if modulated else "dfdx_times_f"]).squeeze()

        def rhs(xx, _u=u, _P=P, _spec=spec):
            return f_ode(xx, _u, _P, _spec)

        fx = rhs(x)
        _, got = jvp(rhs, (x,), (fx / _dz_dx(x, spec.N),))
        assert np.abs(got.numpy() - expect).max() < 1e-11, f"K={case['K']}"


def test_exact_jacobian_differs_from_reference():
    """Guard the bug-of-record: the two Jacobians are genuinely different.

    If a future change to the state layout made every state linear, the two
    would coincide and the parity test above would stop testing anything.
    """
    spec = _driver_spec(3, 7)
    x = 0.05 * torch.sin(torch.arange(1, spec.n_states + 1, dtype=torch.float64))
    u = torch.tensor([0.0, 1.0], dtype=torch.float64)
    P = _case_params(spec, False)

    def rhs(xx):
        return f_ode(xx, u, P, spec)

    fx = rhs(x)
    _, exact = jvp(rhs, (x,), (fx,))
    _, ref = jvp(rhs, (x,), (fx / _dz_dx(x, spec.N),))
    assert np.abs((exact - ref).numpy()).max() > 1e-3


def test_neuronal_to_vascular_shape_and_mass():
    """The N->K depth mapping: interior rows are a partition of unity."""
    spec = _driver_spec(3, 9)
    n2k = neuronal_to_vascular(spec, zero_params(spec))
    assert n2k.shape == (9, 3)
    # The rows are normalised over N+2 depths and then cropped, so every row
    # leaks a little to the two phantom depths -- more the closer it sits to an
    # edge. Interior rows keep essentially all their mass.
    rows = n2k.sum(-1)
    assert torch.all(rows[2:-2] > 1.0 - 1e-3)
    assert torch.all(rows <= 1.0 + 1e-9)
    # The two edge depths lose the most, then get the `nb` top-up from their
    # own neuronal depth -- blood at the boundary has nowhere else to come from.
    assert rows[0] < rows[2] and rows[-1] < rows[-3]


def test_baseline_cbv_increases_toward_surface():
    """s_d > 0 must put more ascending-vein blood volume near the pial surface.

    This is the mechanism behind the monotonic laminar BOLD increase, and the
    parameter Uludag & Havlicek (2021) show is the model's failure mode when
    underestimated.
    """
    from fastfuncstuff.laminar.forward import baseline_hemodynamics

    spec = _driver_spec(3, 9)
    hemo = baseline_hemodynamics(spec, zero_params(spec))
    V0d = hemo["V0d"]
    assert torch.all(V0d[:-1] > V0d[1:]), "CBV0 must decrease with depth"
    # Venules are flat by default (s_v = 0).
    V0v = hemo["V0v"]
    assert torch.allclose(V0v, V0v[0].expand_as(V0v))
    # Flow accumulates upward, so the deepest depth's AV flow is its own venule
    # outflow -- which is why the ODE needs no special case at k = K-1.
    assert torch.allclose(hemo["F0d"][-1], hemo["F0v"][-1])


def test_sample_indices_match_spm_int_it():
    """ceil((0:v-1)*u/v) + D, the reference's output sampling."""
    idx = sample_indices(n_micro=32 * 10, n_scans=10, delay_bins=16)
    # -1 converts the reference's 1-based microtime index to our 0-based one.
    expected = np.ceil(np.arange(10) * 320 / 10).astype(int) + 16 - 1
    assert np.array_equal(idx.numpy(), expected)


def test_laminar_bold_increases_toward_surface():
    """End-to-end: a brief stimulus gives a depth profile peaking superficially.

    The signature result of Havlicek & Uludag (2020) -- the draining vein piles
    signal up at the surface even with uniform neuronal drive across depths.
    """
    spec = _driver_spec(3, 9)
    P = _case_params(spec, False)
    n_micro = int(28 / spec.dt)
    u = build_input(spec, [[]], [[]], n_micro)
    u = torch.cat([u, build_input(spec, [[2.0]], [[2.0]], n_micro)], dim=-1)
    y = integrate(u, P, spec, n_scans=16)

    peak = y.max(dim=0).values
    assert peak.shape == (9,)
    assert peak[0] > peak[-1] * 1.5, "superficial depths must dominate"
    assert 1.0 < peak.max() < 20.0, "response should be a plausible percent change"
    # A post-stimulus undershoot is a property of the physiology, not a fit.
    assert y[:, 0].min() < -0.05


def test_batched_matches_looped():
    """Batch-first is the whole design; it must be numerically identical."""
    spec = _driver_spec(3, 7)
    n_micro = int(14 / spec.dt)
    u = torch.cat(
        [build_input(spec, [[]], [[]], n_micro), build_input(spec, [[2.0]], [[2.0]], n_micro)],
        dim=-1,
    )
    slopes = torch.tensor([-0.3, 0.0, 0.5], dtype=torch.float64)

    batched = zero_params(spec, batch=(3,))
    batched["C"] = torch.tensor([[0.0, 1.0]] * spec.N, dtype=torch.float64).expand(3, 3, 2)
    batched["s_d"] = slopes
    y_batch = integrate(u, batched, spec, n_scans=8)

    for i, s in enumerate(slopes):
        P = _case_params(spec, False)
        P["s_d"] = s.clone()
        y_one = integrate(u, P, spec, n_scans=8)
        assert torch.allclose(y_batch[i], y_one, atol=1e-12)
