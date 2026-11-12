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


def _small_inv(a: Tensor) -> Tensor:
    """Batched inverse of ``[V, p, p]``; closed form for p <= 2.

    Every MEMA design without covariates is 1x1 or 2x2, and a batched solver
    call over millions of those is launch- and latency-bound (cuSOLVER) where
    the adjugate is one elementwise pass.
    """
    p = a.shape[-1]
    if p == 1:
        return 1.0 / a
    if p == 2:
        a00, a01, a10, a11 = a[:, 0, 0], a[:, 0, 1], a[:, 1, 0], a[:, 1, 1]
        det = a00 * a11 - a01 * a10
        out = torch.stack([torch.stack([a11, -a01], -1), torch.stack([-a10, a00], -1)], -2)
        return out / det[:, None, None]
    return plinalg.inv(a)


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
    a_inv = _small_inv(_xtwx(x, w))
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


def _reml_tau2(
    y: Tensor, v: Tensor, x: Tensor, max_iter: int, abs_tol: float | None = None
) -> tuple[Tensor, Tensor]:
    """3dMEMA's REML Fisher scoring, vectorised: ``(tau2, converged)`` per voxel.

    Stops at a relative change of 1e-3 (``rmaB``), or with ``abs_tol`` at an
    absolute step below it (``rmaB2``, the unequal-variance fit, uses 1e-8).
    """
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
        # Halve until the step keeps tau^2 >= 0 (3dMEMA's while loop), with the
        # number of halvings in closed form -- the smallest k with
        # t_old + adj / 2^k >= 0 -- so the loop costs no host sync.  R keeps
        # halving until adj underflows to -0, so a step that can never fit
        # (tau^2 already 0) ends as no step at all.
        neg = (t_old + adj) < 0
        ratio = torch.where(neg & (t_old > 0), -adj / torch.where(t_old > 0, t_old, 1.0), 1.0)
        k = torch.ceil(torch.log2(ratio)).clamp(0, 1074)
        halved = adj * torch.pow(2.0, -k)
        halved = torch.where((t_old + halved) < 0, halved / 2, halved)  # rounding at the edge
        adj = torch.where(neg, torch.where(t_old > 0, halved, torch.zeros_like(adj)), adj)
        adj = torch.where((t_old + adj) < 0, torch.zeros_like(adj), adj)
        t_new = t_old + adj
        tau2[idx] = t_new
        if abs_tol is None:
            active[idx] = adj.abs() / (t_new.abs() + 1e-8) > 0.001
        else:
            active[idx] = adj.abs() > abs_tol
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
        if design.q:
            # 3dMEMA's own output indexes b[1], b[2] as the two means, which with
            # covariates are the first set's intercept and slope.
            raise ValueError("MEMA -unpooled (separate tau^2) takes no covariates")
        assert bb is not None and tb is not None
        fit = fit_mema_unequal(
            ba.T.contiguous(), variance_from_t(ba, ta).T.contiguous(),
            bb.T.contiguous(), variance_from_t(bb, tb).T.contiguous(),
            knha=knha, reml=reml,
        )  # fmt: skip
        return unequal_bricks(design, fit), fit
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


# ---------------------------------------------------------------------------
# Separate tau^2 per set (3dMEMA -unequal_variance, rmaB2)
# ---------------------------------------------------------------------------


@dataclass
class UnequalFit:
    b: tuple[Tensor, Tensor]  # [V] each set's mean
    z: tuple[Tensor, Tensor, Tensor]  # A, B, A-B (Hartung-Knapp applied if asked)
    tau2: tuple[Tensor, Tensor]
    qe: tuple[Tensor, Tensor]
    reml: Tensor  # [V] bool
    converged: Tensor


def fit_mema_unequal(
    ya: Tensor, va: Tensor, yb: Tensor, vb: Tensor, *, knha: bool = True, reml: str = "always"
) -> UnequalFit:
    """Each set fitted alone with its own tau^2; inputs [V, n] per set.

    3dMEMA's ``rmaB2``: method of moments per set without Hartung-Knapp; REML
    for both sets (absolute step < 1e-8, at most 50 iterations) where any MoM
    |z| exceeds 1.3, or everywhere with ``reml="always"``; a voxel keeps MoM
    unless both sets converge.  Hartung-Knapp is applied afterwards as one
    divisor per output: sqrt(y'Py / (n-1)) per set, and for A-B the pooled
    projection with each set's own tau^2 on the design [1, 1_B].
    """
    xa = torch.ones((ya.shape[1], 1), dtype=ya.dtype, device=ya.device)
    xb = torch.ones((yb.shape[1], 1), dtype=yb.dtype, device=yb.device)
    zero = torch.zeros(ya.shape[0], dtype=ya.dtype, device=ya.device)

    def mom(y, v, x):
        st0 = _wls(y, v, x, zero, knha=False)
        n = y.shape[1]
        t2 = ((st0.ypy - (n - 1)) / st0.tr_p).clamp_min(0)
        return st0.ypy, t2

    qa, ta2 = mom(ya, va, xa)
    qb, tb2 = mom(yb, vb, xb)

    def state(y, v, x, t2):
        st = _wls(y, v, x, t2, knha=False)
        se, z = _z(st)
        return st, se, z

    sa, sea, za = state(ya, va, xa, ta2)
    sb, seb, zb = state(yb, vb, xb, tb2)
    used = torch.zeros_like(ta2, dtype=torch.bool)
    conv = torch.ones_like(used)
    if reml != "never":
        todo = torch.ones_like(used) if reml == "always" else (
            (za.abs() > Z_SHORTCUT).any(1) | (zb.abs() > Z_SHORTCUT).any(1)
        )  # fmt: skip
        idx = todo.nonzero().squeeze(1)
        if idx.numel():
            ra, oka = _reml_tau2(ya[idx], va[idx], xa, 50, abs_tol=1e-8)
            rb, okb = _reml_tau2(yb[idx], vb[idx], xb, 50, abs_tol=1e-8)
            ok = oka & okb
            take = idx[ok]
            ta2 = ta2.index_put((take,), ra[ok])
            tb2 = tb2.index_put((take,), rb[ok])
            used[take] = True
            conv[idx[~ok]] = False
            sa, sea, za = state(ya, va, xa, ta2)
            sb, seb, zb = state(yb, vb, xb, tb2)

    b_a, b_b = sa.b[:, 0], sb.b[:, 0]
    s_a, s_b = sea[:, 0], seb[:, 0]
    z_a, z_b = za[:, 0], zb[:, 0]
    se_d = torch.sqrt(s_a * s_a + s_b * s_b)
    z_d = torch.where(se_d > TOL_L, (b_a - b_b) / torch.where(se_d > TOL_L, se_d, 1.0), 0.0)
    if knha:
        na, nb = ya.shape[1], yb.shape[1]
        p1 = (sa.ypy / (na - 1)).sqrt()
        p2 = (sb.ypy / (nb - 1)).sqrt()
        # Pooled projection on [1, 1_B] with each set's own weights.
        y = torch.cat([ya, yb], 1)
        w = torch.cat([1.0 / (va + ta2[:, None]), 1.0 / (vb + tb2[:, None])], 1)
        x = torch.zeros((na + nb, 2), dtype=ya.dtype, device=ya.device)
        x[:, 0] = 1.0
        x[na:, 1] = 1.0
        a_inv = _small_inv(_xtwx(x, w))
        bb = torch.einsum("vpq,vq->vp", a_inv, torch.einsum("np,vn->vp", x, w * y))
        r = y - bb @ x.T
        p3 = ((w * r * r).sum(1) / (na + nb - 2)).sqrt()
        ok = (p1 > TOL_L) & (p2 > TOL_L) & (p3 > TOL_L)
        z_a = torch.where(ok, z_a / torch.where(ok, p1, 1.0), z_a)
        z_b = torch.where(ok, z_b / torch.where(ok, p2, 1.0), z_b)
        z_d = torch.where(ok, z_d / torch.where(ok, p3, 1.0), z_d)
    return UnequalFit((b_a, b_b), (z_a, z_b, z_d), (ta2, tb2), (qa, qb), used, conv)


def unequal_bricks(design, fit: UnequalFit):
    """Bricks for the unequal-variance model, correctly labelled.

    3dMEMA writes these one sub-brick past their labels (13 bricks, 12 labels:
    ``G1:tau`` names an empty brick and tau2/tau1 lands in the unlabelled last
    one).  Same values here, each under its own label, no empty brick.
    """
    from fastfuncstuff.stats.group_ttest import FITT, Brick

    la, lb = design.label_a, design.label_b
    na, nb = design.n_a, design.n_b
    (ba, bb), (za, zb, zd), (ta, tb) = fit.b, fit.z, fit.tau2
    ratio_ab = torch.where(tb > TOL_L, (ta / torch.where(tb > TOL_L, tb, 1.0)).sqrt(), 0.0)
    ratio_ba = torch.where(ta > TOL_L, (tb / torch.where(ta > TOL_L, ta, 1.0)).sqrt(), 0.0)
    return [
        Brick(f"{la}:b", ba),
        Brick(f"{la}:t", za, (FITT, (float(na - 1),))),
        Brick(f"{lb}:b", bb),
        Brick(f"{lb}:t", zb, (FITT, (float(nb - 1),))),
        Brick(f"{la}-{lb}:b", ba - bb),
        Brick(f"{la}-{lb}:t", zd, (FITT, (float(na + nb - 2),))),
        Brick(f"{la}:tau", ta.sqrt()),
        Brick(f"{la}:QE", fit.qe[0], (6, (float(na - 1),))),
        Brick(f"{lb}:tau", tb.sqrt()),
        Brick(f"{lb}:QE", fit.qe[1], (6, (float(nb - 1),))),
        Brick("tau1 / tau2", ratio_ab),
        Brick("tau2 / tau1", ratio_ba),
    ]


# ---------------------------------------------------------------------------
# Null maps: tau^2 re-estimated for every relabeling
# ---------------------------------------------------------------------------


@dataclass
class MemaNull:
    """Null z-maps for one MEMA output, refitting tau^2 under every sign flip.

    The residuals of the observed fit (``y - X b``, or each set's deviations
    from its own mean for the unequal-variance model) have their signs flipped
    at random while each subject keeps its own within-subject variance; under
    H0 with symmetric errors that is an exact relabeling, and the flipped data
    are refitted with the same estimator as the observed map (same REML mode,
    same Hartung-Knapp).  Label permutation is not used: subjects with
    different known variances are not exchangeable.

    Every (iteration, voxel) pair is an independent small fit, so a batch of
    ``nb`` maps is one call over ``nb * V`` voxels.

    ``output`` picks the statistic: a contrast vector over the equal-variance
    design, or 0/1/2 (A, B, A-B) for the unequal-variance model.
    """

    y: Tensor  # [N, V] (A then B), float64
    v: Tensor  # [N, V] within-subject variances
    x: np.ndarray | None  # equal-variance design; None for unequal
    output: np.ndarray | int
    dof: float
    n_iter: int
    n_a: int = 0  # unequal-variance: size of set A
    seed: int = 0
    knha: bool = True
    reml: str = "always"

    def __post_init__(self) -> None:
        self.device = self.y.device
        yt, vt = self.y.T.contiguous(), self.v.T.contiguous()
        if self.x is None:
            na = self.n_a
            fit = fit_mema_unequal(yt[:, :na], vt[:, :na], yt[:, na:], vt[:, na:],
                                   knha=self.knha, reml=self.reml)  # fmt: skip
            mean = torch.cat([fit.b[0][:, None].expand(-1, na),
                              fit.b[1][:, None].expand(-1, yt.shape[1] - na)], 1)  # fmt: skip
        else:
            fit = fit_mema(yt, vt, self.x, knha=self.knha, reml=self.reml)
            mean = fit.b @ torch.as_tensor(self.x, dtype=yt.dtype, device=yt.device).T
        self._r = (yt - mean).contiguous()  # [V, N]
        self._v = vt
        rng = np.random.default_rng(self.seed)
        n = yt.shape[1]
        self._signs = rng.choice(np.array([-1.0, 1.0]), size=(self.n_iter, n))
        from fastfuncstuff.stats.group_ttest import TToZ

        self._toz = TToZ(self.dof, self.device)

    @property
    def n_distinct(self) -> float:
        return float(2.0 ** min(self._signs.shape[1], 1000))

    def bytes_per_map(self) -> int:
        n = self._r.shape[1]
        p = 1 if self.x is None else self.x.shape[1]
        return int(self._r.shape[0]) * (8 * n * 16 + 8 * p * p * 12)

    def stat(self, start: int, nb: int) -> Tensor:
        s = torch.as_tensor(
            self._signs[start : start + nb], dtype=self._r.dtype, device=self.device
        )
        v_count, n = self._r.shape
        ys = (s[:, None, :] * self._r[None]).reshape(nb * v_count, n)
        vs = self._v.repeat(nb, 1)
        if self.x is None:
            na = self.n_a
            fit = fit_mema_unequal(ys[:, :na], vs[:, :na], ys[:, na:], vs[:, na:],
                                   knha=self.knha, reml=self.reml)  # fmt: skip
            z = fit.z[int(self.output)]
        else:
            fit = fit_mema(ys, vs, self.x, knha=self.knha, reml=self.reml)
            _, z = _contrast(fit, np.asarray(self.output, dtype=np.float64))
        return z.reshape(nb, v_count)

    def __call__(self, start: int, nb: int) -> Tensor:
        return self._toz(self.stat(start, nb).float())
