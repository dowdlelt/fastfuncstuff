"""Column-set statistics in the ARMA(1,1) GLM (``column_sets``, ffs_reml -nuisance_summary).

Each set gets a Wald F from the fit's own (X'WX)^-1 and the temporal SD of its fitted
signal. The referee is the nested-model F computed by hand under the same whitener:
drop the set, refit, ((RSS_r - RSS_f) / q) / (RSS_f / dof). One column gives t^2.
"""

import numpy as np
import torch

from fastfuncstuff.glm.arma import fit_glm_arma11, fit_glm_arma11_grouped
from fastfuncstuff.glm.whiten import arma11_whitener

CPU = torch.device("cpu")


def _design(n, rng):
    t = np.arange(n)
    task = (np.sin(2 * np.pi * t / 20) > 0.3).astype(float)
    motion = np.cumsum(rng.standard_normal((n, 3)), 0) * 0.1
    return np.column_stack([np.ones(n), task, motion, (t - n / 2) / n])


def _ar_data(X, beta, rng, a=0.5):
    n_vox, n = beta.shape[0], X.shape[0]
    e = rng.standard_normal((n_vox, n))
    for i in range(1, n):
        e[:, i] += a * e[:, i - 1]
    return beta @ X.T + e


def _nested_f(y, X, cols, a, b):
    w = arma11_whitener(a, b, len(y), device=CPU, dtype=torch.float64)
    Xw = w.cols(torch.from_numpy(X)).numpy()
    yw = w.cols(torch.from_numpy(y[:, None])).numpy()[:, 0]

    def rss(M):
        return float(np.sum((yw - M @ np.linalg.lstsq(M, yw, rcond=None)[0]) ** 2))

    keep = [c for c in range(X.shape[1]) if c not in cols]
    full, red = rss(Xw), rss(Xw[:, keep])
    dof = len(y) - X.shape[1]
    return ((red - full) / len(cols)) / (full / dof)


def test_set_f_is_the_nested_model_f_and_sd_is_the_fitted_signal():
    rng = np.random.default_rng(0)
    n, n_vox = 160, 12
    X = _design(n, rng)
    beta = rng.standard_normal((n_vox, X.shape[1])) * [100, 1, 0.8, 0.8, 0.8, 1]
    beta[:4, 2:5] = 0  # some voxels where motion does nothing
    data = _ar_data(X, beta, rng)
    sets = [[2, 3, 4], [2], [3], [4], [1]]
    res = fit_glm_arma11(
        torch.from_numpy(data), torch.from_numpy(X), tr=2.0, device=CPU, verbose=False,
        use_double=True, column_sets=sets,
    )  # fmt: skip
    F = res.set_fstats.numpy()
    for v in range(n_vox):
        a, b = (float(x) for x in res.arma_params[v])
        for g, cols in enumerate(sets):
            np.testing.assert_allclose(F[v, g], _nested_f(data[v], X, cols, a, b), rtol=1e-5)
    # one column: F = t^2
    np.testing.assert_allclose(F[:, 1:5], res.tstats.numpy()[:, [2, 3, 4, 1]] ** 2, rtol=1e-5)
    # SD of X_S b_S over time, in data units (betas unscaled)
    Xc = X - X.mean(0)
    sig = Xc[:, 2:5] @ res.betas.numpy()[:, 2:5].T
    np.testing.assert_allclose(res.set_sd.numpy()[:, 0], sig.std(0), rtol=1e-5)


def test_dsort_columns_join_a_set_with_their_per_voxel_design():
    """A -dsort column is index n_regressors + k; its set F comes from each voxel's own
    extended (X'WX)^-1."""
    rng = np.random.default_rng(1)
    n, n_vox = 150, 8
    X = _design(n, rng)
    d = rng.standard_normal((n_vox, 1, n))
    beta = rng.standard_normal((n_vox, X.shape[1])) * [100, 1, 0.5, 0.5, 0.5, 1]
    data = _ar_data(X, beta, rng) + 0.7 * d[:, 0]
    p = X.shape[1]
    sets = [[2, 3, 4, p], [p]]
    res = fit_glm_arma11(
        torch.from_numpy(data), torch.from_numpy(X), tr=2.0, device=CPU, verbose=False,
        use_double=True, dsort=torch.from_numpy(d), column_sets=sets,
    )  # fmt: skip
    F = res.set_fstats.numpy()
    for v in range(n_vox):
        a, b = (float(x) for x in res.arma_params[v])
        Xv = np.column_stack([X, d[v, 0]])
        for g, cols in enumerate(sets):
            np.testing.assert_allclose(F[v, g], _nested_f(data[v], Xv, cols, a, b), rtol=1e-4)
    np.testing.assert_allclose(F[:, 1], res.dsort_tstats.numpy()[:, 0] ** 2, rtol=1e-4)


def test_slicewise_designs_each_get_their_own_set_stats():
    rng = np.random.default_rng(2)
    n, n_vox = 140, 10
    X = _design(n, rng)
    slices = np.arange(n_vox) % 2
    blocks = rng.standard_normal((2, n, 2))
    designs = {s: torch.from_numpy(np.column_stack([X, blocks[s]])) for s in (0, 1)}
    beta = rng.standard_normal((n_vox, X.shape[1] + 2)) * [100, 1, 0.5, 0.5, 0.5, 1, 1, 1]
    data = np.stack([designs[slices[v]].numpy() @ beta[v] for v in range(n_vox)])
    data = data + _ar_data(X, np.zeros((n_vox, X.shape[1])), rng)
    p = X.shape[1]
    res = fit_glm_arma11_grouped(
        torch.from_numpy(data), designs, torch.from_numpy(slices), tr=2.0, group_label="slice",
        device=CPU, verbose=False, use_double=True, column_sets=[[p, p + 1], [p]],
    )  # fmt: skip
    F = res.set_fstats.numpy()
    for v in range(n_vox):
        a, b = (float(x) for x in res.arma_params[v])
        Xv = designs[slices[v]].numpy()
        np.testing.assert_allclose(F[v, 0], _nested_f(data[v], Xv, [p, p + 1], a, b), rtol=1e-5)
    np.testing.assert_allclose(F[:, 1], res.tstats.numpy()[:, p] ** 2, rtol=1e-5)
