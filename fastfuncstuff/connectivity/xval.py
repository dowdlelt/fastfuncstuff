"""Cross-validation across runs from per-run sufficient statistics.

Each run is cleaned on its own (its own drift, its own ortvecs, its own censoring),
so a run's contribution to any correlation is fully described by three numbers per
ROI pair: the weighted cross-product ``S_r = X_r W_r X_r^T``, the weighted sums
``s_r = X_r w_r`` and the total weight ``n_r``. The correlation over ANY subset of
runs is then exact from sums of those:

    C(A) = sum_{r in A} S_r  -  (sum s_r)(sum s_r)^T / sum n_r,    R(A) = normalise(C(A))

That is what makes leave-one-run-out (LORO) free: one cleaning pass, one
``(R, N, N)`` stack, and every fold is an ``O(N^2)`` subtraction -- no re-cleaning,
no re-correlating. It is the same move as the GLM's per-run projection
([[LORO cross-validation]]): nuisance is fold-local because it was run-local all
along.

Two uses:

* **Reproducibility** -- each held-out run's matrix against the matrix of the
  others, compared in Fisher-z over the upper triangle. A referee for cleaning
  choices, with a caveat the docs spell out: structure that repeats across runs for
  non-neural reasons (global signal, a cardiac-locked vessel) is reproducible too.
* **Shrinkage by held-out likelihood** -- for a grid of intensities ``a``, the
  training correlation shrunk toward the identity, ``(1-a) R + a I``, scored by the
  Gaussian log-likelihood of the held-out run. One eigendecomposition per fold gives
  the whole grid. An empirical answer to "how much shrinkage", independent of
  Ledoit-Wolf's analytic formula and of what it counts as a sample.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from fastfuncstuff.utils import full_fp32_matmul


@dataclass
class CrossProducts:
    """Per-run weighted sufficient statistics, float64 on CPU."""

    S: Tensor  # (R, N, N)  X_r W_r X_r^T
    s: Tensor  # (R, N)     X_r w_r
    n: Tensor  # (R,)       sum w_r

    @property
    def n_runs(self) -> int:
        return int(self.n.numel())

    def covariance(self, runs: Sequence[int] | Tensor | None = None) -> Tensor:
        """Weighted, centred cross-product over the chosen runs (all by default)."""
        idx = torch.arange(self.n_runs) if runs is None else torch.as_tensor(runs)
        S = self.S[idx].sum(0)
        s = self.s[idx].sum(0)
        n = self.n[idx].sum()
        return S - torch.outer(s, s) / n.clamp_min(1e-300)

    def correlation(self, runs: Sequence[int] | Tensor | None = None) -> Tensor:
        return _normalise(self.covariance(runs))


def _normalise(c: Tensor) -> Tensor:
    d = c.diagonal().clamp_min(0.0).sqrt()
    valid = d > 1e-12 * float(d.max().clamp_min(1e-300))
    d = torch.where(valid, d, torch.ones_like(d))
    r = (c / (d[:, None] * d[None, :])).clamp(-1.0, 1.0)
    r = r * (valid[:, None] & valid[None, :])
    r.diagonal().copy_(valid.to(r.dtype))
    return r


@full_fp32_matmul()
def cross_products(
    x: Tensor,
    run_starts: Sequence[int],
    weights: Tensor | np.ndarray | None = None,
) -> CrossProducts:
    """``(N, T)`` cleaned series cut at ``run_starts`` -> :class:`CrossProducts`.

    ``run_starts`` index the columns of ``x`` (the censored axis: a cleaner's
    ``kept_run_starts``). Accumulated in float64 on ``x``'s device where it can hold
    float64, then moved to the CPU.
    """
    from fastfuncstuff.utils import widest_float_dtype

    dt = widest_float_dtype(x.device)
    xx = x.to(dt)
    n_t = xx.shape[1]
    w_all = None
    if weights is not None:
        w_all = torch.as_tensor(np.asarray(weights) if not isinstance(weights, Tensor) else weights)
        w_all = w_all.to(device=x.device, dtype=dt).reshape(-1)
        if w_all.numel() != n_t:
            raise ValueError(f"{w_all.numel()} weights for {n_t} time points")
    bounds = [*run_starts, n_t]
    S, s, n = [], [], []
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        seg = xx[:, a:b]
        if w_all is None:
            S.append(seg @ seg.T)
            s.append(seg.sum(1))
            n.append(torch.tensor(float(b - a), dtype=dt))
        else:
            w = w_all[a:b]
            S.append((seg * w) @ seg.T)
            s.append(seg @ w)
            n.append(w.sum())
    to = {"device": "cpu"}
    return CrossProducts(
        S=torch.stack(S).to(**to).to(torch.float64),
        s=torch.stack(s).to(**to).to(torch.float64),
        n=torch.stack([v.to(**to) for v in n]).to(torch.float64),
    )


def _upper_z(r: Tensor) -> Tensor:
    from fastfuncstuff.connectivity.matrix import fisher_z

    iu = torch.triu_indices(r.shape[0], r.shape[0], 1)
    return fisher_z(r[iu[0], iu[1]])


@dataclass
class LoroResult:
    reproducibility: np.ndarray  # (R,) r between held-out and rest matrices (Fisher-z, upper tri)
    per_run: Tensor  # (R, N, N) each run's own correlation matrix

    @property
    def mean(self) -> float:
        return float(np.mean(self.reproducibility))


def loro_reproducibility(cp: CrossProducts) -> LoroResult:
    """Each run's matrix against the matrix of all the others."""
    if cp.n_runs < 2:
        raise ValueError("leave-one-run-out needs at least two runs")
    S_all, s_all, n_all = cp.S.sum(0), cp.s.sum(0), cp.n.sum()
    rep, per = [], []
    for k in range(cp.n_runs):
        own = _normalise(cp.S[k] - torch.outer(cp.s[k], cp.s[k]) / cp.n[k])
        s_rest = s_all - cp.s[k]
        rest = _normalise(S_all - cp.S[k] - torch.outer(s_rest, s_rest) / (n_all - cp.n[k]))
        a, b = _upper_z(own), _upper_z(rest)
        a, b = a - a.mean(), b - b.mean()
        rep.append(float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-300)))
        per.append(own)
    return LoroResult(reproducibility=np.asarray(rep), per_run=torch.stack(per))


@dataclass
class ShrinkCV:
    grid: np.ndarray  # (G,) shrinkage intensities
    loglik: np.ndarray  # (R, G) held-out Gaussian log-likelihood per fold, per sample
    best: float  # intensity with the highest mean held-out log-likelihood

    def score(self, a: float) -> float:
        """Mean held-out log-likelihood at intensity ``a`` (linear in the grid)."""
        return float(np.interp(a, self.grid, self.loglik.mean(0)))


def shrinkage_cv(cp: CrossProducts, grid: np.ndarray | None = None) -> ShrinkCV:
    """Pick the shrinkage toward the identity by leave-one-run-out likelihood.

    For each fold the training correlation ``R`` is eigendecomposed once; the
    held-out run's correlation ``Rk`` is scored under ``(1-a) R + a I`` for every
    ``a``: ``-0.5 [log det + tr(inverse @ Rk)]`` per sample, which needs only the
    eigenvalues and the diagonal of ``V^T Rk V``. Correlation scale throughout, so a
    run's overall variance cannot win the comparison.
    """
    from fastfuncstuff import policy_linalg as plinalg

    if cp.n_runs < 2:
        raise ValueError("shrinkage by cross-validation needs at least two runs")
    grid = np.linspace(0.0, 1.0, 101) if grid is None else np.asarray(grid, float)
    a = torch.as_tensor(grid, dtype=torch.float64)
    out = np.zeros((cp.n_runs, grid.size))
    for k in range(cp.n_runs):
        rest = [r for r in range(cp.n_runs) if r != k]
        train = cp.correlation(rest)
        held = cp.correlation([k])
        lam, vec = plinalg.eigh(train)
        lam = lam.clamp_min(0.0)
        q = ((vec.T @ held) * vec.T).sum(1)  # diag(V^T Rk V)
        ev = (1.0 - a)[:, None] * lam[None, :] + a[:, None]  # (G, N)
        with np.errstate(divide="ignore"):
            ll = -0.5 * (torch.log(ev).sum(1) + (q[None, :] / ev).sum(1))
        out[k] = ll.numpy()
    out[~np.isfinite(out)] = -np.inf
    best = float(grid[int(np.argmax(out.mean(0)))])
    return ShrinkCV(grid=grid, loglik=out, best=best)


def shrink(r: Tensor | np.ndarray, a: float) -> np.ndarray:
    """``(1 - a) R + a I`` -- the target every shrinkage here uses."""
    rr = np.asarray(r.cpu() if isinstance(r, Tensor) else r, dtype=np.float64)
    return (1.0 - a) * rr + a * np.eye(rr.shape[0])
