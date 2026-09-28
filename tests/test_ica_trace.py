import numpy as np

from fastfuncstuff.benchmark.stages.ica_single_trace import _row_subspace_principal_cos


def test_row_subspace_principal_cos_ignores_basis_scaling():
    rng = np.random.default_rng(7)
    basis, _ = np.linalg.qr(rng.standard_normal((20, 3)))
    a = basis.T
    b = np.array([[4.0], [0.2], [9.0]]) * a[[2, 0, 1]]

    np.testing.assert_allclose(_row_subspace_principal_cos(a, b), 1.0, atol=1e-12)
