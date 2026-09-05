"""design_cholesky / ols_betas: the same betas lstsq gives, without re-factoring.

These exist because a voxel-chunk loop that calls lstsq per chunk re-factors a
design that never changes. The contract is that hoisting the factorization does
not move the answer, and that a design lstsq can still handle (rank-deficient)
falls back rather than raising.
"""

from __future__ import annotations

import torch

from fastfuncstuff.glm.core import design_cholesky, ols_betas

CPU = torch.device("cpu")


def _problem(n_tp=120, p=8, n_vox=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    design = torch.randn(n_tp, p, generator=g, dtype=torch.float64)
    data = torch.randn(n_vox, n_tp, generator=g, dtype=torch.float64)
    return design, data


def test_cholesky_betas_match_lstsq():
    design, data = _problem()
    expected = torch.linalg.lstsq(design, data.T).solution.T
    got = ols_betas(design, data, cholesky_L=design_cholesky(design))
    assert torch.allclose(got, expected, rtol=1e-8, atol=1e-8)


def test_betas_do_not_depend_on_how_the_voxels_are_chunked():
    """The whole point of hoisting: one factor, many chunks, same answer.

    Agreement is to float64 rounding, not bitwise: BLAS blocks a (p, n_vox)
    product differently at different widths, so the last ulp moves with the
    chunk size (measured 2.6e-16 relative). That was equally true of the lstsq
    loop this replaces.
    """
    design, data = _problem(n_vox=64)
    chol = design_cholesky(design)
    whole = ols_betas(design, data, cholesky_L=chol)
    chunked = torch.cat(
        [ols_betas(design, data[i : i + 7], cholesky_L=chol) for i in range(0, 64, 7)]
    )
    assert torch.allclose(whole, chunked, rtol=1e-13, atol=1e-13)


def test_rank_deficient_design_declines_the_factor_and_still_solves():
    design, data = _problem(p=6)
    design[:, 5] = design[:, 2]  # exact collinearity: X'X is singular
    assert design_cholesky(design) is None
    betas = ols_betas(design, data, cholesky_L=None)
    assert betas.shape == (data.shape[0], design.shape[1])
    assert torch.isfinite(betas).all()


def test_a_fit_is_actually_recovered():
    """Guards against an answer that is merely self-consistent but wrong."""
    design, _ = _problem(p=4, n_vox=1)
    true_betas = torch.tensor([[1.5, -2.0, 0.25, 3.0]], dtype=torch.float64)
    data = (design @ true_betas.T).T
    got = ols_betas(design, data, cholesky_L=design_cholesky(design))
    assert torch.allclose(got, true_betas, rtol=1e-8, atol=1e-8)


def test_float32_path_matches_lstsq_to_single_precision():
    design, data = _problem()
    design, data = design.float(), data.float()
    expected = torch.linalg.lstsq(design, data.T).solution.T
    got = ols_betas(design, data, cholesky_L=design_cholesky(design))
    assert torch.allclose(got, expected, rtol=1e-4, atol=1e-5)
