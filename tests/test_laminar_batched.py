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
