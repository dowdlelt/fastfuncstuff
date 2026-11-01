"""The batched Variational Laplace against the serial scheme it vectorises.

Agreement is to rounding, not to the bit: batched linear algebra reorders sums.
The serial path is itself pinned to the MATLAB reference
(``test_laminar_inversion.py``), so matching it here carries that parity over.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from fastfuncstuff.laminar.batched import dx_b, inv_b, logdet_b, variational_laplace_batched
from fastfuncstuff.laminar.inversion import spm_dx, spm_inv, spm_logdet, variational_laplace, vec

CPU = torch.device("cpu")


def _mats(seed=0, n=7):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, n, generator=g, dtype=torch.float64)
    spd = a @ a.mT + n * torch.eye(n, dtype=torch.float64)
    diag = torch.diag(torch.rand(n, generator=g, dtype=torch.float64) * 10 + 0.1)
    return torch.stack([spd, diag @ spd, -spd, a, -torch.eye(n, dtype=torch.float64)])


def test_logdet_b_takes_the_serial_branch_per_matrix():
    # SPD (Cholesky), non-symmetric product (SVD), negative definite (SVD), generic.
    C = _mats()
    got = logdet_b(C)
    want = torch.stack([spm_logdet(c) for c in C])
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)


def test_logdet_b_falls_back_on_a_zero_diagonal():
    C = _mats()[:2].clone()
    C[1, 2, :] = 0
    C[1, :, 2] = 0
    torch.testing.assert_close(logdet_b(C)[1], spm_logdet(C[1]), rtol=1e-12, atol=1e-12)


def test_inv_b_matches_spm_inv_including_singular():
    C = _mats()
    C[3] = torch.zeros_like(C[3])  # singular: survives only through the ridge
    got = inv_b(C)
    for i in range(C.shape[0]):
        torch.testing.assert_close(got[i], spm_inv(C[i]), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("v", [-4.0, 4.0, 40.0])  # 40 takes the pinv branch
def test_dx_b_matches_spm_dx(v):
    C = -_mats()[[0, 1, 4]]
    f = torch.randn(3, C.shape[-1], dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    got = dx_b(C, f, v)
    for i in range(3):
        torch.testing.assert_close(got[i], spm_dx(C[i], f[i], v), rtol=1e-9, atol=1e-12)


def test_dx_b_takes_a_step_per_fit():
    C = -_mats()[[0, 0]]
    f = torch.ones(2, C.shape[-1], dtype=torch.float64)
    got = dx_b(C, f, torch.tensor([-4.0, 40.0], dtype=torch.float64))
    torch.testing.assert_close(got[0], spm_dx(C[0], f[0], -4.0))
    torch.testing.assert_close(got[1], spm_dx(C[1], f[1], 40.0))


def test_per_fit_kernels_must_match_the_batch():
    from fastfuncstuff.laminar.experiment import faes_priors
    from fastfuncstuff.laminar.params import ModelSpec

    spec = ModelSpec(N=3, K=6, n_inputs=2, n_mod=1)
    with pytest.raises(ValueError, match="per-fit kernels"):
        variational_laplace_batched(
            spec, faes_priors(spec, (1, 0, 0)), torch.zeros(100, 2),
            torch.zeros(2, 10, 6), kernel=torch.zeros(3, 6, 6),
        )  # fmt: skip


def _oracle_problem():
    from test_laminar_modelspace import _oracle, _spec

    o = _oracle()
    block = o["per_k"][1]
    K, ns = int(block["K"]), int(block["ns"])
    spec = _spec(o, K)

    def t(a):
        return torch.tensor(np.asarray(a, dtype=float), dtype=torch.float64)

    y = t(block["y"]).reshape(ns, K)
    u = t(block["u"]).reshape(-1, 2)
    rows = t(block["mask_rows"]).reshape(-1).to(torch.bool)
    kernel = t(o["kernel"]).reshape(-1)
    return spec, y, u, rows, kernel


@pytest.mark.slow  # ~60 s: three serial inversions per model and the batch
@pytest.mark.parametrize("target", [(0, 0, 0), (1, 0, 0)])
def test_batched_follows_each_serial_fit(target):
    """Three columns -- the published data and two noisier copies -- batched, each
    against its own serial inversion: same free energy, posterior and iterations."""
    from fastfuncstuff.laminar.experiment import faes_priors

    spec, y, u, rows, kernel = _oracle_problem()
    g = torch.Generator().manual_seed(3)
    Y = torch.stack([y, y + 0.05 * torch.randn(y.shape, generator=g, dtype=torch.float64),
                     0.7 * y + 0.1 * torch.randn(y.shape, generator=g, dtype=torch.float64)])  # fmt: skip
    pr = faes_priors(spec, target)
    batch = variational_laplace_batched(spec, pr, u, Y, rows=rows, kernel=kernel, device=CPU)
    for i in range(Y.shape[0]):
        s = variational_laplace(spec, pr, u, Y[i], rows=rows, kernel=kernel)
        b = batch.result(i)
        assert b.n_iter == s.n_iter, f"column {i}: {b.n_iter} vs {s.n_iter} iterations"
        assert math.isclose(float(b.F), float(s.F), rel_tol=1e-9, abs_tol=1e-7), f"column {i}"
        torch.testing.assert_close(vec(b.Ep), vec(s.Ep), rtol=1e-6, atol=1e-8)
        assert b.converged == s.converged and b.stalled == s.stalled


@pytest.mark.slow  # ~40 s
def test_per_fit_matrix_kernel_matches_serial():
    """A (K, K) PSF per column applied after the integration equals the serial path
    applying it inside."""
    from fastfuncstuff.laminar.experiment import faes_priors

    spec, y, u, rows, _ = _oracle_problem()
    K = spec.K
    mats = []
    for var in (0.01, 0.03):
        # A matrix form of a Gaussian depth blur, rows normalised.
        x = torch.linspace(0, 1, K, dtype=torch.float64)
        m = torch.exp(-((x[:, None] - x[None]) ** 2) / (2 * var))
        mats.append(m / m.sum(1, keepdim=True))
    kern = torch.stack(mats)
    Y = torch.stack([y, 0.8 * y])
    pr = faes_priors(spec, (0, 0, 1))
    batch = variational_laplace_batched(spec, pr, u, Y, rows=rows, kernel=kern, device=CPU)
    for i in range(2):
        s = variational_laplace(spec, pr, u, Y[i], rows=rows, kernel=kern[i])
        assert batch.result(i).n_iter == s.n_iter
        assert math.isclose(float(batch.F[i]), float(s.F), rel_tol=1e-9, abs_tol=1e-7)


@pytest.mark.slow  # 107 s: two models, serial reference per column
def test_fit_columns_matches_fit_roi_per_column():
    """The batched model space reports what the validated per-ROI pipeline does:
    F per model, model and depth probabilities, the best model's B."""
    from fastfuncstuff.laminar.io import LaminarData
    from fastfuncstuff.laminar.pipeline import fit_roi
    from fastfuncstuff.laminar.scope import fit_columns

    spec, y, u, rows, kernel = _oracle_problem()
    g = torch.Generator().manual_seed(5)
    Y = torch.stack([y, 0.8 * y + 0.05 * torch.randn(y.shape, generator=g, dtype=torch.float64)])
    targets = [(0, 0, 0), (1, 0, 0)]
    batched = fit_columns(
        {spec.K: LaminarData(y=Y, rows=rows, u=u, spec=spec)}, {spec.K: kernel},
        targets=targets, device=CPU, progress=False,
    )  # fmt: skip
    for i in range(2):
        one = fit_roi({spec.K: LaminarData(y=Y[i], rows=rows, u=u, spec=spec)}, {spec.K: kernel},
                      targets=targets)  # fmt: skip
        np.testing.assert_allclose(batched.F[i], one.F, rtol=1e-9, atol=1e-6)
        np.testing.assert_allclose(batched.p_model[i], one.p_model, atol=1e-6)
        np.testing.assert_allclose(batched.p_depth[i], one.p_depth, atol=1e-6)
        assert batched.best[i] == one.best
        np.testing.assert_allclose(batched.B_mean[i], one.B_mean, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(batched.B_sd[i], one.B_sd, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(batched.r2[i], one.r2, rtol=1e-6, atol=1e-8)


def _spm_reduce_dense(qE, qC, pE, pC, rE, rC):
    """spm_log_evidence_reduce, dense and literal (rC may be tiny, not zero)."""
    qP, pP, rP = np.linalg.inv(qC), np.linalg.inv(pC), np.linalg.inv(rC)
    sP = qP + rP - pP
    sC = np.linalg.inv(sP)
    sE = sC @ (qP @ qE + rP @ rE - pP @ pE)
    ld = np.linalg.slogdet(rP @ qP @ sC @ pC)[1]
    F = ld / 2 - (qE @ qP @ qE + rE @ rP @ rE - pE @ pP @ pE - sE @ sP @ sE) / 2
    return F, sE, sC


def test_reduction_matches_the_spm_formula():
    """Pinning entries (Savage-Dickey) and shrinking others (general reduction) against
    the dense formula, where pinning is a vanishing prior variance."""
    import dataclasses as dc

    from fastfuncstuff.laminar.batched import BatchedVL, reduce_batched
    from fastfuncstuff.laminar.experiment import faes_priors
    from fastfuncstuff.laminar.params import ModelSpec

    spec = ModelSpec(N=3, K=6, n_inputs=2, n_mod=1)
    full_pr = faes_priors(spec, (1, 1, 1))
    pE_v, pC_v = vec(full_pr.pE), vec(full_pr.pC)
    free = torch.nonzero(pC_v > 0).squeeze(-1)
    n = free.numel()
    g = torch.Generator().manual_seed(2)
    Bn = 3
    a = torch.randn(Bn, n, n, generator=g, dtype=torch.float64) * 0.1
    Cp = a @ a.mT + 0.05 * torch.eye(n, dtype=torch.float64)
    Ep = pE_v.expand(Bn, -1).clone()
    Ep[:, free] += 0.3 * torch.randn(Bn, n, generator=g, dtype=torch.float64)
    full = BatchedVL(Ep_vec=Ep, Cp=Cp, Eh=torch.zeros(Bn, 6, dtype=torch.float64),
                     F=torch.full((Bn,), -100.0, dtype=torch.float64),
                     L=torch.zeros(Bn, 3, dtype=torch.float64),
                     y_pred=torch.zeros(Bn, 4, 6, dtype=torch.float64), free_idx=free,
                     n_iter=torch.ones(Bn, dtype=torch.long),
                     converged=torch.ones(Bn, dtype=torch.bool),
                     stalled=torch.zeros(Bn, dtype=torch.bool),
                     failed=torch.zeros(Bn, dtype=torch.bool), spec=spec)  # fmt: skip
    red_pr = faes_priors(spec, (1, 0, 0))
    red_pr = dc.replace(red_pr, pC={**red_pr.pC, "s_d": red_pr.pC["s_d"] * 0.25},
                        pE={**red_pr.pE, "s_d": red_pr.pE["s_d"] + 0.4})  # fmt: skip
    got = reduce_batched(full, full_pr, red_pr)
    rE_v, rC_v = vec(red_pr.pE)[free].numpy(), vec(red_pr.pC)[free].numpy()
    rC_dense = np.where(rC_v > 0, rC_v, 1e-14)
    keep = rC_v > 0
    for i in range(Bn):
        F, sE, sC = _spm_reduce_dense(
            Ep[i, free].numpy(), Cp[i].numpy(), pE_v[free].numpy(), np.diag(pC_v[free].numpy()),
            rE_v, np.diag(rC_dense),
        )  # fmt: skip
        assert float(got.F[i]) == pytest.approx(-100.0 + F, abs=1e-5)
        np.testing.assert_allclose(got.Ep_vec[i, got.free_idx].numpy(), sE[keep], atol=1e-8)
        np.testing.assert_allclose(got.Cp[i].numpy(), sC[np.ix_(keep, keep)], atol=1e-8)


@pytest.mark.slow  # 89 s: brute force over the model space vs one full inversion
def test_reduced_model_space_tracks_brute_force_on_the_oracle():
    """BMR's evidence for the nested models, from the full model alone, against
    inverting each nested model (the 'internal validation' of Friston & Penny 2011).
    Measured agreement is recorded in the assertion bounds."""
    from fastfuncstuff.laminar.batched import reduce_batched
    from fastfuncstuff.laminar.experiment import faes_priors, layer_model_targets

    spec, y, u, rows, kernel = _oracle_problem()
    kw = dict(rows=rows, kernel=kernel, device=CPU, rescue=2)
    full_pr = faes_priors(spec, (1, 1, 1))
    full = variational_laplace_batched(spec, full_pr, u, y[None], **kw)
    brute, red = [], []
    for target in layer_model_targets(3)[:-1]:
        pr = faes_priors(spec, target)
        brute.append(float(variational_laplace_batched(spec, pr, u, y[None], **kw).F[0]))
        red.append(float(reduce_batched(full, full_pr, pr).F[0]))
    brute, red = np.array(brute), np.array(red)
    # Measured 2026-10-31: the same ranking of all seven; reduction is never above brute
    # force; competitive models (within 10 nats of the best) agree to 1.3-1.6 nats, distant
    # ones read 8-11 nats low (one Laplace point stretched far from where it was fitted).
    assert list(np.argsort(-brute)) == list(np.argsort(-red))
    assert (red <= brute + 0.5).all()
    near = brute >= brute.max() - 10
    assert np.abs(red - brute)[near].max() < 2.0


@pytest.mark.slow  # 26 s: one full inversion + forward passes
def test_fit_columns_reduce_reaches_the_published_answer():
    """The reduced model space on the published data (K = 9 block): MATLAB's brute force
    ranks superficial+middle+deep first (F -120.78) and superficial+deep second
    (-125.72). The full model is inverted, so its F is MATLAB's; the runner-up is a
    reduction, within the measured BMR gap."""
    from fastfuncstuff.laminar.io import LaminarData
    from fastfuncstuff.laminar.scope import fit_columns

    spec, y, u, rows, kernel = _oracle_problem()
    fits = fit_columns(
        {spec.K: LaminarData(y=y[None], rows=rows, u=u, spec=spec)}, {spec.K: kernel},
        device=CPU, progress=False, rescue=2, model_space="reduce",
    )  # fmt: skip
    order = [fits.models[i] for i in np.argsort(-fits.F[0])]
    assert order[:2] == ["superficial+middle+deep", "superficial+deep"]
    assert fits.F[0, fits.models.index("superficial+middle+deep")] == pytest.approx(
        -120.78, abs=0.01
    )
    assert abs(fits.F[0, fits.models.index("superficial+deep")] - (-125.72)) < 2.0
    assert fits.nesting[0] == 0.0  # every nested model reduced from the one posterior
    # Model averaging: the full model holds ~99% here, so B_bma is close to its B; the
    # mixture sd is never below the winner's own.
    assert fits.p_model[0].max() > 0.95
    np.testing.assert_allclose(fits.B_bma[0], fits.B_mean[0], rtol=0.1, atol=0.05)
    assert (fits.B_bma_sd[0] >= 0.9 * fits.B_sd[0]).all()
    # The serial (MATLAB-exact) full model has sigma = 3 exp(-3.171) = 0.126 Hz with B of
    # 2.7 / 2.0 / 4.8 Hz: the modulated self-connection -sigma + B is net excitatory.
    assert fits.sigma_hz[0] == pytest.approx(0.126, abs=0.01)
    assert np.isfinite(fits.r2[0]).all() and (fits.r2[0] <= 1).all()


@pytest.mark.slow  # 42 s: full inversion, one refit
def test_unintegrable_reductions_are_inverted_instead(monkeypatch):
    """Where a reduced posterior cannot be integrated, the model is inverted for those
    columns: its F becomes brute force's (MATLAB: superficial -276.81 on this block)."""
    import fastfuncstuff.laminar.scope as scope
    from fastfuncstuff.laminar.io import LaminarData

    spec, y, u, rows, kernel = _oracle_problem()
    real = scope.predict_batched
    calls = {"n": 0}

    def flaky(*a, **k):
        out = real(*a, **k)
        calls["n"] += 1
        if calls["n"] == 1:  # the first reduced model: column 1 "blows up"
            out = out.clone()
            out[1] = float("nan")
        return out

    monkeypatch.setattr(scope, "predict_batched", flaky)
    fits = scope.fit_columns(
        {spec.K: LaminarData(y=torch.stack([y, y]), rows=rows, u=u, spec=spec)},
        {spec.K: kernel}, targets=[(1, 0, 0), (1, 1, 1)], device=CPU, progress=False,
        model_space="reduce",
    )  # fmt: skip
    assert fits.refit[:, 0].tolist() == [False, True] and not fits.refit[:, 1].any()
    assert fits.F[1, 0] == pytest.approx(-276.81, abs=0.05)  # inverted
    assert fits.F[0, 0] < -276.81  # reduced: the measured BMR gap, below brute force
    assert np.isfinite(fits.r2).all()
