"""ffs_denoise -nuisance_scope test: user nuisance cleans the held-out run only."""

import numpy as np
import torch

from fastfuncstuff.denoise.sequential import cross_validate_noise_pcs

CPU = torch.device("cpu")


def _problem(seed=0, n_runs=3, T=80, V=40):
    g = torch.Generator().manual_seed(seed)
    run_starts = [r * T for r in range(n_runs)]
    X = torch.zeros(n_runs * T, 2)
    for r in range(n_runs):
        on = torch.randint(0, T - 6, (10,), generator=g)
        for o in on[:5]:
            X[r * T + o : r * T + o + 4, 0] = 1.0
        for o in on[5:]:
            X[r * T + o : r * T + o + 4, 1] = 1.0
    beta = torch.randn(2, V, generator=g)
    poly = [torch.stack([torch.ones(T), torch.linspace(-1, 1, T)], 1) for _ in range(n_runs)]
    motion = [torch.cumsum(torch.randn(T, 1, generator=g), 0) for _ in range(n_runs)]
    gamma = 3 * torch.randn(1, V, generator=g)
    Y = X @ beta + 0.5 * torch.randn(n_runs * T, V, generator=g)
    for r in range(n_runs):
        Y[r * T : (r + 1) * T] += motion[r] @ gamma
    pcs = [torch.randn(T, 3, generator=g) for _ in range(n_runs)]
    return Y.T.contiguous(), X, run_starts, poly, motion, pcs, T


def _resid(M, A):
    Q, _ = torch.linalg.qr(A.double())
    M = M.double()
    return M - Q @ (Q.T @ M)


def test_test_scope_matches_hand_written_loro():
    Y, X, run_starts, poly, motion, pcs, T = _problem()
    full = [torch.cat([p, m], 1) for p, m in zip(poly, motion, strict=True)]
    maps, _ = cross_validate_noise_pcs(
        data=Y,
        design_matrix=X,
        noise_pcs=pcs,
        run_starts=run_starts,
        tr=1.0,
        max_components=0,
        nuisance=poly,
        test_nuisance=full,
        device=CPU,
    )
    # by hand: train on poly-projected runs, score the held-out run poly+motion-projected
    n_runs = len(run_starts)
    ys, preds = [], []
    for h in range(n_runs):
        G = 0
        b = 0
        for r in range(n_runs):
            if r == h:
                continue
            sl = slice(r * T, (r + 1) * T)
            Xr, Yr = _resid(X[sl], poly[r]), _resid(Y[:, sl].T, poly[r])
            G, b = G + Xr.T @ Xr, b + Xr.T @ Yr
        beta = torch.linalg.solve(G, b)
        sl = slice(h * T, (h + 1) * T)
        ys.append(_resid(Y[:, sl].T, full[h]))
        preds.append(_resid(X[sl], full[h]) @ beta)
    y, p = torch.cat(ys), torch.cat(preds)
    ss_res = ((y - p) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    expected = (1 - ss_res / ss_tot).numpy()
    np.testing.assert_allclose(maps[:, 0], expected, atol=2e-4)


def test_test_scope_equal_to_nuisance_is_the_default():
    Y, X, run_starts, poly, _, pcs, _ = _problem(seed=1)
    a, _ = cross_validate_noise_pcs(
        data=Y,
        design_matrix=X,
        noise_pcs=pcs,
        run_starts=run_starts,
        tr=1.0,
        max_components=3,
        nuisance=poly,
        device=CPU,
    )
    b, _ = cross_validate_noise_pcs(
        data=Y,
        design_matrix=X,
        noise_pcs=pcs,
        run_starts=run_starts,
        tr=1.0,
        max_components=3,
        nuisance=poly,
        test_nuisance=poly,
        device=CPU,
    )
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_held_out_cleaning_changes_the_score():
    # Guards against the argument being accepted and ignored. The direction is NOT a
    # gain: motion this strong biases betas trained without it, and a motion-free
    # referee exposes that bias, so the held-out R2 drops.
    Y, X, run_starts, poly, motion, pcs, _ = _problem(seed=2)
    full = [torch.cat([p, m], 1) for p, m in zip(poly, motion, strict=True)]
    kw = dict(
        data=Y,
        design_matrix=X,
        noise_pcs=pcs,
        run_starts=run_starts,
        tr=1.0,
        max_components=0,
        device=CPU,
    )
    base, _ = cross_validate_noise_pcs(nuisance=poly, **kw)
    test, _ = cross_validate_noise_pcs(nuisance=poly, test_nuisance=full, **kw)
    assert abs(float(np.median(test[:, 0]) - np.median(base[:, 0]))) > 0.05
