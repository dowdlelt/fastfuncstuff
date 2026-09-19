"""
Parity tests for the Variational Laplace inversion.

The oracle is a full inversion of the Faes et al. published test dataset
(Planum Polare, left hemisphere), run through their own MATLAB. Regenerate with::

    cd tests/laminar_oracle && matlab -batch "run('make_inversion_oracle.m')"

The only change from their driver is a seed on the white-noise padding, so the
Python port can be handed byte-identical data.

Free energy is the gate here, not the fit. Two models are inverted -- modulation
targeting superficial layers, and the null -- so the test can check that the
*ranking* survives the port, which is the actual scientific output.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from fastfuncstuff.laminar.integrate import batch_bucket, build_input
from fastfuncstuff.laminar.inversion import (
    Priors,
    masked_vec,
    spm_dx,
    spm_inv,
    spm_logdet,
    unvec,
    variational_laplace,
    vec,
)
from fastfuncstuff.laminar.params import P0, ModelSpec, zero_params

ORACLE = Path(__file__).parent / "laminar_oracle" / "laminar_inversion_oracle.json"
STEP_ORACLE = Path(__file__).parent / "laminar_oracle" / "laminar_vlstep_oracle.json"


def _oracle():
    if not ORACLE.exists():  # pragma: no cover - only when the dump is missing
        pytest.skip(f"no inversion oracle at {ORACLE}")
    return json.loads(ORACLE.read_text())


def _setup(o, model):
    """Rebuild the reference's model, priors and data in FFS terms."""
    N, K = o["N"], o["K"]
    spec = ModelSpec(
        N=N,
        K=K,
        n_inputs=2,
        n_mod=1,
        dt=float(o["dt"]),
        TR=float(o["TR"]),
        p0=P0(V0t=3.0, nr=3.0, al_v=0.35, w_v=0.5),
    )
    t = lambda a: torch.tensor(np.asarray(a, dtype=float), dtype=torch.float64)  # noqa: E731

    pE = zero_params(spec)
    pE["C"] = torch.tensor([[0.0, 1.0]] * N, dtype=torch.float64)
    pE["mu"] = torch.tensor(-0.8, dtype=torch.float64)
    pE["lam"] = torch.tensor(1.8, dtype=torch.float64)

    pC = zero_params(spec)
    pC["C"] = torch.tensor([[0.0, 1.0]] * N, dtype=torch.float64)
    pC["nsig"] = torch.tensor(math.exp(-4), dtype=torch.float64)
    pC["sigma"] = torch.tensor(math.exp(-2), dtype=torch.float64)
    pC["al_d"] = torch.tensor(math.exp(-5), dtype=torch.float64)
    pC["s_d"] = torch.tensor(math.exp(-1), dtype=torch.float64)
    target = {"superficial": [1.0, 0.0, 0.0], "null": [0.0, 0.0, 0.0]}[model["name"]]
    pC["B"] = torch.diag(torch.tensor(target, dtype=torch.float64))[None] * math.exp(0.5)

    y = t(o["y"]).reshape(o["ns"], K)
    u = t(o["u"]).reshape(-1, 2)
    rows = t(o["mask_rows"]).reshape(-1).to(torch.bool)
    kernel = t(o["kernel"]).reshape(-1)
    return spec, Priors(pE=pE, pC=pC), y, u, rows, kernel


def _run(o, model):
    spec, priors, y, u, rows, kernel = _setup(o, model)
    res = variational_laplace(spec, priors, u, y, rows=rows, kernel=kernel)
    return spec, res


# --------------------------------------------------------------------------
# SPM numerical primitives
# --------------------------------------------------------------------------


def test_spm_logdet_handles_rank_deficiency():
    """Zero-diagonal rows are dropped, not turned into -inf."""
    A = torch.diag(torch.tensor([2.0, 0.0, 3.0], dtype=torch.float64))
    assert torch.isclose(spm_logdet(A), torch.tensor(math.log(6.0), dtype=torch.float64))
    # A plain logdet would give -inf on exactly this input.
    assert torch.isinf(torch.logdet(A))


def test_spm_inv_survives_a_singular_matrix():
    A = torch.zeros(3, 3, dtype=torch.float64)
    A[0, 0] = 1.0
    out = spm_inv(A)
    assert torch.isfinite(out).all()
    # The well-conditioned direction still inverts properly.
    assert torch.isclose(out[0, 0], torch.tensor(1.0, dtype=torch.float64), atol=1e-6)


def test_spm_dx_reduces_to_newton_for_large_t():
    """With a huge ascent rate the step is the full Gauss-Newton step."""
    J = -torch.diag(torch.tensor([2.0, 4.0], dtype=torch.float64))
    f = torch.tensor([1.0, 1.0], dtype=torch.float64)
    assert torch.allclose(spm_dx(J, f, 64.0), -torch.linalg.pinv(J) @ f, atol=1e-8)


def test_vec_unvec_roundtrip():
    spec = ModelSpec(N=3, K=9, n_inputs=2, n_mod=1)
    P = zero_params(spec)
    P["s_d"] = torch.tensor(0.4, dtype=torch.float64)
    P["C"] = torch.arange(6, dtype=torch.float64).reshape(3, 2)
    back = unvec(vec(P), spec)
    for name in P:
        assert torch.equal(P[name], back[name]), name


def test_vec_unvec_keeps_batch_dims():
    spec = ModelSpec(N=3, K=7, n_inputs=2, n_mod=1)
    P = zero_params(spec, batch=(5,))
    v = vec(P)
    assert v.shape[0] == 5
    assert unvec(v, spec)["C"].shape == (5, 3, 2)


# --------------------------------------------------------------------------
# End-to-end parity against the published pipeline
# --------------------------------------------------------------------------


def test_input_matches_spm_get_ons():
    """Our boxcar builder against SPM's, on the reference's own design."""
    o = _oracle()
    spec, *_ = _setup(o, o["models"][0])
    u_ref = torch.tensor(np.asarray(o["u"], dtype=float), dtype=torch.float64).reshape(-1, 2)
    u_ours = build_input(spec, [[47.9], [1.6, 46.4]], [[0.1], [1.6, 1.6]], u_ref.shape[0])
    assert torch.equal(u_ours, u_ref)


@pytest.mark.slow
@pytest.mark.parametrize("which", [0, 1])
def test_free_energy_matches_matlab(which):
    """The quantity model comparison ranks, against the reference inversion."""
    o = _oracle()
    model = o["models"][which]
    _, res = _run(o, model)
    assert res.F.isfinite()
    rel = abs(float(res.F) - float(model["F"])) / abs(float(model["F"]))
    assert rel < 1e-6, f"{model['name']}: F={float(res.F):.6f} vs {float(model['F']):.6f}"


@pytest.mark.slow
def test_model_ranking_matches_matlab():
    """The ordering of the two models -- the actual scientific output."""
    o = _oracle()
    ours = [float(_run(o, m)[1].F) for m in o["models"]]
    theirs = [float(m["F"]) for m in o["models"]]
    assert (ours[0] > ours[1]) == (theirs[0] > theirs[1])

    # Posterior model probabilities, the form the ranking is reported in.
    def probs(f):
        d = np.asarray(f) - min(f)
        return np.exp(d) / np.exp(d).sum()

    assert np.allclose(probs(ours), probs(theirs), atol=1e-6)


@pytest.mark.slow
def test_posterior_means_match_matlab():
    """The estimated modulation, draining-vein slope and neuronal parameters."""
    o = _oracle()
    model = o["models"][0]
    _, res = _run(o, model)
    checks = {
        "sigma": float(model["Ep_sigma"]),
        "s_d": float(model["Ep_s_d"]),
        "nsig": float(model["Ep_nsig"]),
        "al_d": float(model["Ep_al_d"]),
    }
    for name, expect in checks.items():
        got = float(res.Ep[name])
        assert abs(got - expect) < 1e-6 + 1e-6 * abs(expect), f"{name}: {got} vs {expect}"

    b_ours = torch.diagonal(res.Ep["B"][0]).numpy()
    b_theirs = np.asarray(model["Ep_B"], dtype=float).reshape(-1)
    assert np.abs(b_ours - b_theirs).max() < 1e-6


@pytest.mark.slow
def test_predicted_response_matches_matlab():
    o = _oracle()
    model = o["models"][0]
    _, res = _run(o, model)
    yp = np.asarray(model["Yp"], dtype=float).reshape(o["ns"], o["K"])
    rows = np.asarray(o["mask_rows"], dtype=float).reshape(-1) > 0
    # Only the fitted points are meaningful; the padding is fitted by nothing.
    err = np.abs(res.y_pred.numpy()[rows] - yp[rows]).max()
    assert err < 1e-8, f"max abs error on fitted points: {err}"


def test_prediction_at_prior_mean_matches_matlab():
    """The forward prediction before any fitting, against a one-iteration dump.

    This is the cheap test that localises a parity failure. An off-by-one in the
    TR sampling index (the reference counts microtime bins from 1) shifted every
    sample by 50 ms: invisible in the shape of the response, worth ~18 nats of
    free energy, and indistinguishable from "the optimiser landed somewhere
    else" if you only ever look at the end of the inversion.
    """
    if not STEP_ORACLE.exists():  # pragma: no cover
        pytest.skip(f"no VL-step oracle at {STEP_ORACLE}")
    step = json.loads(STEP_ORACLE.read_text())
    o = _oracle()
    spec, priors, _y, u, _rows, kernel = _setup(o, o["models"][0])
    from fastfuncstuff.laminar.integrate import integrate

    f0 = integrate(u, priors.pE, spec, o["ns"], kernel=kernel).numpy()
    expect = np.asarray(step["f0"], dtype=float).reshape(o["ns"], o["K"])
    assert np.abs(f0 - expect).max() < 1e-12


def test_noise_components_match_spm_ce():
    """One precision hyperparameter per depth, over that depth's kept samples."""
    if not STEP_ORACLE.exists():  # pragma: no cover
        pytest.skip(f"no VL-step oracle at {STEP_ORACLE}")
    step = json.loads(STEP_ORACLE.read_text())
    assert step["nh"] == 9 and step["nq"] == 1 and step["ny_masked"] == 126
    # Purely diagonal: spm_Ce's AR machinery degenerates to a selection block.
    assert step["Q1_offdiag_nnz"] == 0
    q1 = np.asarray(step["Q1_diag"], dtype=float).reshape(-1)
    assert np.array_equal(np.nonzero(q1)[0], np.arange(14))


def test_default_hyperprior_matches_spm():
    """hE = 4 - log(var(y)), over the *whole* response including the padding."""
    if not STEP_ORACLE.exists():  # pragma: no cover
        pytest.skip(f"no VL-step oracle at {STEP_ORACLE}")
    step = json.loads(STEP_ORACLE.read_text())
    o = _oracle()
    y = torch.tensor(np.asarray(o["y"], dtype=float), dtype=torch.float64).reshape(o["ns"], o["K"])
    ours = 4.0 - float(torch.log(y.var()))
    assert abs(ours - float(np.asarray(step["hE"]).reshape(-1)[0])) < 1e-10


def test_batch_bucket_collapses_the_model_space_to_one_shape():
    """Bug of record: the shape count, not any one shape, was the problem.

    Each distinct batch width is its own ~20 s static compile of the Ito-Taylor
    step, and dynamo stops compiling after `recompile_limit` of them -- silently,
    running eager at 137x the cost from then on. The eight-model space spans
    batch widths 8 to 11, which together with four vascular depths made 16
    shapes against a default limit of 8; it reverted to eager on inversion 17 of
    32. Bucketing must leave the whole model space on a single width.
    """
    from fastfuncstuff.laminar.experiment import faes_priors, layer_model_targets

    spec = ModelSpec(N=3, K=9, n_inputs=2, n_mod=1)
    widths = {
        batch_bucket(int((vec(faes_priors(spec, t).pC) > 0).sum()) + 1)
        for t in layer_model_targets(3)
    }
    assert widths == {16}


def test_batch_bucket_is_monotone_and_powers_of_two():
    assert batch_bucket(1) == 16  # floored: a lone fit shares the model space shape
    assert batch_bucket(16) == 16
    assert batch_bucket(17) == 32
    assert batch_bucket(64) == 64
    assert batch_bucket(65) == 128
    widths = [batch_bucket(n) for n in range(1, 300)]
    assert widths == sorted(widths)


def test_error_precision_is_diagonal_by_construction():
    """The structure the M-step exploits, asserted rather than assumed.

    SPM writes the error precision as sum_i exp(h_i) Q_i, with Q[i] selecting
    depth i's block of the response vector. Because masked_vec is depth-major,
    each Q[i] is the identity on one contiguous block, so the sum is diagonal --
    which is what lets the M-step replace ny^3 matmuls with ns_kept^2 work.

    If masked_vec ever stopped being depth-major, or a correlated noise model
    introduced off-block terms, the M-step would silently compute the wrong
    thing. This is the guard.
    """
    n_scans, K = 10, 4
    rows = torch.zeros(n_scans, dtype=torch.bool)
    rows[2:7] = True
    ns_kept = int(rows.sum())

    # Depth-major means entry (i * ns_kept + t) is depth i, retained time t.
    y = torch.arange(n_scans * K, dtype=torch.float64).reshape(n_scans, K)
    v = masked_vec(y, rows)
    assert v.numel() == ns_kept * K
    for i in range(K):
        block = v[i * ns_kept : (i + 1) * ns_kept]
        assert torch.equal(block, y[rows, i]), f"depth {i} block is not contiguous"

    # Hence the precision built from those blocks is diagonal.
    ny = ns_kept * K
    h = torch.tensor([0.5, -0.2, 1.3, 0.0], dtype=torch.float64)
    Q = torch.zeros(K, ny, ny, dtype=torch.float64)
    for i in range(K):
        sl = slice(i * ns_kept, (i + 1) * ns_kept)
        Q[i, sl, sl] = torch.eye(ns_kept, dtype=torch.float64)
    iS = (Q * torch.exp(h)[:, None, None]).sum(0)
    assert torch.equal(iS, torch.diag(torch.diagonal(iS))), "iS is not diagonal"
    assert torch.allclose(torch.diagonal(iS), torch.exp(h).repeat_interleave(ns_kept))

    # And PS[i] PS[j] vanishes off the diagonal, so dFdhh is diagonal.
    S = torch.diag(1.0 / torch.diagonal(iS))
    PS = (Q * torch.exp(h)[:, None, None]) @ S
    for i in range(K):
        for j in range(K):
            if i != j:
                assert float((PS[i] @ PS[j]).abs().max()) == 0.0, f"blocks {i},{j} overlap"
