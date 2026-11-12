"""
Mixed-effects meta-analysis of per-subject (beta, t) maps — the 3dMEMA model.

Each subject's beta is taken with a *known* within-subject variance
``v = (beta / t)^2``; one between-subject variance ``tau^2`` per voxel is
estimated, and the group effect is the WLS fit with weights ``1 / (v + tau^2)``
(FSL's FLAME1 is the same model).  Faithful to ``R_scripts/3dMEMA.R``:

* ``v`` is 0 — then replaced by 1e8, i.e. weight ~0 — where ``|beta|`` or
  ``|t|`` is under 1e-16, so a subject with no data at a voxel drops out.
* Method of moments first; Hartung-Knapp scaling of the covariance by
  ``max(y'Py / (n-p), 0)`` (3dMEMA's default ``-HKtest``).
* 3dMEMA runs REML only where some coefficient's MoM |z| exceeds 1.3.  That is
  a speed shortcut, and it leaves a seam in the map at 1.3 where the estimator
  changes; here REML runs everywhere unless ``reml="shortcut"`` asks for the
  3dMEMA behaviour (parity checks).  REML is 3dMEMA's Fisher scoring:
  ``tau^2 += (y'PPy - tr P) / tr(PP)``, step-halved to stay >= 0, stopped at a
  relative change of 1e-3 or 200 iterations (non-converged voxels keep MoM).
* The "t" bricks are ``b / se`` with dof ``N - groups - covariates``.

Everything is batched over voxels: the design is shared, so every per-voxel
quantity is a weighted sum over subjects or a p x p solve, and the traces REML
needs come from p x p products (``tr PP = sum w^2 - 2 tr(A^-1 X'W^3X) +
tr((A^-1 X'W^2X)^2)``) without ever forming an N x N matrix per voxel.
See [[Group-level inference]].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from fastfuncstuff import policy_linalg as plinalg

TOL_L = 1e-16
TOL_U = 1e8
SE_THR = 1e-8  # 3dMEMA con$thr: below this se the z is set to 0
Z_SHORTCUT = 1.3  # 3dMEMA con$thrZ


def variance_from_t(b: Tensor, t: Tensor) -> Tensor:
    """3dMEMA's ``(b/t)^2``, with no-data subjects pushed to weight ~0."""
    ok = (b.abs() >= TOL_L) & (t.abs() >= TOL_L)
    v = torch.where(ok, (b / torch.where(ok, t, torch.ones_like(t))) ** 2, torch.zeros_like(b))
    return torch.where(v < TOL_L, torch.full_like(v, TOL_U), v)


def _xtwx(x: Tensor, w: Tensor) -> Tensor:
    """``[V, p, p]`` X' diag(w_v) X for every voxel; ``w`` is [V, N]."""
    return torch.einsum("np,vn,nq->vpq", x, w, x)


def _trace(a: Tensor) -> Tensor:
    return a.diagonal(dim1=-2, dim2=-1).sum(-1)


@dataclass
class WlsState:
    b: Tensor  # [V, p]
    vb: Tensor  # [V, p, p] covariance of b (after Hartung-Knapp)
    ypy: Tensor  # [V]  y' P y
    tr_p: Tensor  # [V]
    tr_pp: Tensor  # [V]


def _wls(y: Tensor, v: Tensor, x: Tensor, tau2: Tensor, knha: bool) -> WlsState:
    """Weighted fit at a given tau^2, plus the traces REML needs."""
    n, p = x.shape
    w = 1.0 / (v + tau2.unsqueeze(1))  # [V, N]
    a_inv = plinalg.inv(_xtwx(x, w))
    xtwy = torch.einsum("np,vn->vp", x, w * y)
    b = torch.einsum("vpq,vq->vp", a_inv, xtwy)
    r = y - b @ x.T  # [V, N]
    ypy = (w * r * r).sum(1)
    b2 = _xtwx(x, w * w)
    b3 = _xtwx(x, w * w * w)
    ab2 = a_inv @ b2
    tr_p = w.sum(1) - _trace(ab2)
    tr_pp = (w * w).sum(1) - 2 * _trace(a_inv @ b3) + _trace(ab2 @ ab2)
    vb = a_inv
    if knha:
        vb = vb * (ypy / (n - p)).clamp_min(0).view(-1, 1, 1)
    return WlsState(b=b, vb=vb, ypy=ypy, tr_p=tr_p, tr_pp=tr_pp)


def _z(state: WlsState) -> tuple[Tensor, Tensor]:
    se = state.vb.diagonal(dim1=-2, dim2=-1).clamp_min(0).sqrt()
    z = torch.where(se > SE_THR, state.b / torch.where(se > SE_THR, se, 1.0), 0.0)
    return se, z


@dataclass
class MemaFit:
    b: Tensor  # [V, p]
    se: Tensor  # [V, p]
    z: Tensor  # [V, p]
    vb: Tensor  # [V, p, p]
    tau2: Tensor  # [V]
    qe: Tensor  # [V]  heterogeneity Q (at tau^2 = 0)
    i2: Tensor  # [V]
    reml: Tensor  # [V] bool, REML estimate used
    converged: Tensor  # [V] bool (REML voxels)


def fit_mema(
    y: Tensor,
    v: Tensor,
    x: np.ndarray,
    *,
    knha: bool = True,
    reml: str = "always",
    max_iter: int = 200,
) -> MemaFit:
    """Random-effects WLS per voxel.  ``y``, ``v``: [V, N]; ``x``: [N, p].

    ``reml``: ``"always"`` (default), ``"shortcut"`` (3dMEMA: only where a MoM
    |z| > 1.3) or ``"never"`` (method of moments only).
    """
    if reml not in ("always", "shortcut", "never"):
        raise ValueError(f"reml must be always/shortcut/never, got {reml!r}")
    n, p = x.shape
    if n - p < 1:
        raise ValueError(f"no residual degrees of freedom: {n} subjects, {p} parameters")
    xt = torch.as_tensor(x, dtype=y.dtype, device=y.device)
    zero = torch.zeros(y.shape[0], dtype=y.dtype, device=y.device)

    st0 = _wls(y, v, xt, zero, knha=False)
    qe, tr_p0 = st0.ypy, st0.tr_p
    tau2_mom = ((qe - (n - p)) / tr_p0).clamp_min(0)
    st = _wls(y, v, xt, tau2_mom, knha)
    se, z = _z(st)
    tau2 = tau2_mom
    used = torch.zeros_like(tau2, dtype=torch.bool)
    conv = torch.ones_like(used)

    if reml != "never":
        todo = torch.ones_like(used) if reml == "always" else (z.abs() > Z_SHORTCUT).any(1)
        idx = todo.nonzero().squeeze(1)
        if idx.numel():
            t2, ok = _reml_tau2(y[idx], v[idx], xt, max_iter)
            take = idx[ok]
            if take.numel():
                st_r = _wls(y[take], v[take], xt, t2[ok], knha)
                se_r, z_r = _z(st_r)
                st.b[take], st.vb[take] = st_r.b, st_r.vb
                se[take], z[take], tau2 = se_r, z_r, tau2.index_put((take,), t2[ok])
                used[take] = True
            conv[idx[~ok]] = False

    # 3dMEMA: I2 = 1 / (1 + (n-p-1) / (tr(P0) tau^2)); 0 where tau^2 = 0.
    den = tr_p0 * tau2
    i2 = torch.where(den > 0, 1.0 / (1.0 + (n - p - 1) / torch.where(den > 0, den, 1.0)), 0.0)
    return MemaFit(b=st.b, se=se, z=z, vb=st.vb, tau2=tau2, qe=qe, i2=i2, reml=used, converged=conv)


def _reml_tau2(y: Tensor, v: Tensor, x: Tensor, max_iter: int) -> tuple[Tensor, Tensor]:
    """3dMEMA's REML Fisher scoring, vectorised: ``(tau2, converged)`` per voxel."""
    n = y.shape[1]
    var_y = y.var(1, unbiased=True) if n > 1 else torch.zeros_like(y[:, 0])
    tau2 = (var_y - v.mean(1)).clamp_min(0)
    active = torch.ones_like(tau2, dtype=torch.bool)
    for _ in range(max_iter):
        idx = active.nonzero().squeeze(1)
        if idx.numel() == 0:
            break
        st = _wls(y[idx], v[idx], x, tau2[idx], knha=False)
        # y'PPy = ||P y||^2 with P y = w * r.
        w = 1.0 / (v[idx] + tau2[idx].unsqueeze(1))
        r = y[idx] - st.b @ x.T
        py2 = ((w * r) ** 2).sum(1)
        adj = (py2 - st.tr_p) / st.tr_pp
        t_old = tau2[idx]
        # Halve until the step keeps tau^2 >= 0 (3dMEMA's while loop).  R keeps
        # halving until adj underflows to -0, so a step that can never fit --
        # tau^2 already 0, or adj dwarfing it -- ends as no step at all.
        for _ in range(64):
            neg = (t_old + adj) < 0
            if not bool(neg.any()):
                break
            adj = torch.where(neg, adj / 2, adj)
        adj = torch.where((t_old + adj) < 0, torch.zeros_like(adj), adj)
        t_new = t_old + adj
        change = adj.abs() / (t_new.abs() + 1e-8)
        tau2[idx] = t_new
        active[idx] = change > 0.001
    return tau2, ~active


# ---------------------------------------------------------------------------
# Bricks
# ---------------------------------------------------------------------------


def mema_design(n_a: int, n_b: int, cov: np.ndarray | None) -> np.ndarray:
    """3dMEMA's equal-variance design: ``[1, 1_B, cov...]``.

    Coded exactly as 3dMEMA does (intercept = set A, then B - A), because its
    REML shortcut chooses voxels from the z of *these* coefficients; ffs_group
    reports A - B as a contrast on top.
    """
    n = n_a + n_b
    cols = [np.ones((n, 1))]
    if n_b:
        g = np.zeros((n, 1))
        g[n_a:] = 1.0
        cols.append(g)
    if cov is not None:
        cols.append(cov)
    return np.hstack(cols)


def _contrast(fit: MemaFit, c: np.ndarray) -> tuple[Tensor, Tensor]:
    """``(estimate, b/se)`` of a linear combination, from the fitted covariance."""
    ct = torch.as_tensor(c, dtype=fit.b.dtype, device=fit.b.device)
    est = fit.b @ ct
    var = torch.einsum("p,vpq,q->v", ct, fit.vb, ct).clamp_min(0)
    se = var.sqrt()
    return est, torch.where(se > SE_THR, est / torch.where(se > SE_THR, se, 1.0), 0.0)


def mema_covariates(design, how: str) -> np.ndarray | None:
    """Covariates for the MEMA design: one shared slope, centred per ``how``.

    ``SAME`` (3dMEMA's default) centres on the mean over every subject,
    ``DIFF`` on each set's own mean, ``NONE`` leaves them.
    """
    if not design.q:
        return None
    ca = np.asarray(design.cov_a, dtype=np.float64)
    if not design.two_sets or design.paired:
        return ca - ca.mean(0) if how != "NONE" else ca
    cb = np.asarray(design.cov_b, dtype=np.float64)
    if how == "NONE":
        return np.vstack([ca, cb])
    if how == "DIFF":
        return np.vstack([ca - ca.mean(0), cb - cb.mean(0)])
    allc = np.vstack([ca, cb])
    return allc - allc.mean(0)


def run_mema(
    design,
    ba: Tensor,
    ta: Tensor,
    bb: Tensor | None = None,
    tb: Tensor | None = None,
    *,
    knha: bool = True,
    reml: str = "always",
):
    """Output bricks for a MEMA test, in 3dMEMA's order, with A-B as the difference.

    ``ba``/``ta`` (and ``bb``/``tb``) are [N, V] beta and t maps.  Returns
    ``(bricks, fit)``; ``design`` is a :class:`group_ttest.TtestDesign`.
    """
    from fastfuncstuff.stats.group_ttest import FITT, Brick

    if design.unpooled:
        raise ValueError("MEMA with separate per-set tau^2 (-unpooled) is not implemented yet")
    va = variance_from_t(ba, ta)
    la, lb = design.label_a, design.label_b
    cov = mema_covariates(design, design.center)
    q = design.q
    if not design.two_sets:
        y, v, names = ba, va, [la]
        x = mema_design(design.n_a, 0, cov)
    elif design.paired:
        assert bb is not None and tb is not None
        vb_ = variance_from_t(bb, tb)
        v = va + vb_
        # 3dMEMA zeroes the contrast where either condition had no data.
        y = torch.where(v >= TOL_U, torch.zeros_like(ba), ba - bb)
        names = [f"{la}-{lb}"]
        x = mema_design(design.n_a, 0, cov)
    else:
        assert bb is not None and tb is not None
        y = torch.cat([ba, bb])
        v = torch.cat([va, variance_from_t(bb, tb)])
        x = mema_design(design.n_a, design.n_b, cov)
        names = None
    fit = fit_mema(y.T.contiguous(), v.T.contiguous(), x, knha=knha, reml=reml)
    n, p = x.shape
    dof = float(n - (2 if design.two_sets and not design.paired else 1) - q)
    tstat = (FITT, (dof,))
    out: list = []

    def add(label: str, c: np.ndarray) -> None:
        est, z = _contrast(fit, c)
        out.append(Brick(f"{label}:b", est))
        out.append(Brick(f"{label}:t", z, tstat))

    e = np.eye(p)
    if names is not None:
        add(names[0], e[0])
        k0 = 1
    else:
        add(la, e[0])
        add(lb, e[0] + e[1])
        add(f"{la}-{lb}", -e[1])
        k0 = 2
    for j, nm in enumerate(design.cov_names):
        add(nm, e[k0 + j])
    out.append(Brick("I2", fit.i2))
    out.append(Brick("tau", fit.tau2.sqrt()))
    out.append(Brick("QE:Chisq", fit.qe, (6, (dof,))))
    return out, fit
