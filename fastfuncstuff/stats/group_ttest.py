"""
Group-level t-tests on one map per subject (the 3dttest++ model).

The fits follow 3dttest++'s output conventions exactly (sub-brick order,
labels, degrees of freedom), because that is what an AFNI user will compare
against:

* one-sample: ``A_mean, A_Tstat`` then a (slope, t) pair per covariate;
  dof = n - 1 - q.
* two-sample: the A-B block (pooled model, a separate intercept *and separate
  slopes* per set, dof = nA + nB - 2 - 2q), then each set's own one-sample
  block.  ``-unpooled`` gives Welch with Satterthwaite dof, and every brick is
  reported as z (3dttest++ forces ``-toz`` there).
* paired: a one-sample test of A-B, then each set alone; set B borrows set A's
  covariates (3dttest++ does the same).

The null used for cluster inference (:class:`OlsNull`) never refits: for a
fixed design the flipped/permuted data enter only through ``Q^T y*`` and
``||y*||^2``, and the latter is invariant under sign flips and permutations,
so a batch of null t-maps is a couple of GEMMs against the residual matrix.
See [[Group-level inference]] in the wiki.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

CENTER_MODES = ("DIFF", "SAME", "NONE")


# ---------------------------------------------------------------------------
# Design
# ---------------------------------------------------------------------------


@dataclass
class TtestDesign:
    """What 3dttest++ calls the test: the sets, the pairing and the covariates."""

    n_a: int
    n_b: int = 0
    label_a: str = "SetA"
    label_b: str = "SetB"
    paired: bool = False
    unpooled: bool = False
    cov_a: np.ndarray | None = None  # [nA, q]
    cov_b: np.ndarray | None = None  # [nB, q]
    cov_names: tuple[str, ...] = ()
    center: str = "DIFF"

    def __post_init__(self) -> None:
        self.center = self.center.upper()
        if self.center not in CENTER_MODES:
            raise ValueError(f"center must be one of {CENTER_MODES}, got {self.center!r}")
        if self.paired and self.n_a != self.n_b:
            raise ValueError(f"-paired needs equal set sizes (A has {self.n_a}, B {self.n_b})")
        if self.paired and self.unpooled:
            raise ValueError("-paired and -unpooled are alternatives")
        if self.unpooled and self.n_b == 0:
            raise ValueError("-unpooled needs two sets")
        if self.q and self.unpooled:
            # 3dttest++ silently switches to pooled here; say so instead.
            raise ValueError("covariates need the pooled model: drop -unpooled")
        if self.q:
            if self.cov_a is None or self.cov_a.shape != (self.n_a, self.q):
                raise ValueError("cov_a must be [nA, q]")
            if self.two_sets and self.cov_b is None:
                raise ValueError("cov_b missing for set B")

    @property
    def q(self) -> int:
        return len(self.cov_names)

    @property
    def two_sets(self) -> bool:
        return self.n_b > 0

    def centered_covariates(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Covariates as they enter the model, after ``-center``."""
        if not self.q:
            return None, None
        ca = np.asarray(self.cov_a, dtype=np.float64)
        cb = None
        if self.two_sets:
            cb = ca.copy() if self.paired else np.asarray(self.cov_b, dtype=np.float64)
        if self.center == "NONE":
            return ca, cb
        if self.center == "SAME" and cb is not None and not self.paired:
            mu = np.concatenate([ca, cb]).mean(axis=0)
            return ca - mu, cb - mu
        return ca - ca.mean(axis=0), (None if cb is None else cb - cb.mean(axis=0))


def one_sample_matrix(cov: np.ndarray | None, n: int) -> np.ndarray:
    """``[n, 1+q]``: intercept then covariates."""
    one = np.ones((n, 1))
    return one if cov is None else np.hstack([one, cov])


def two_sample_matrix(ca: np.ndarray | None, cb: np.ndarray | None, na: int, nb: int):
    """``[nA+nB, 2+2q]`` with a separate intercept and slopes per set.

    Columns are ``[1_A, cov_A..., 1_B, cov_B...]`` so the A-B contrasts are
    ``e_k - e_{k+1+q}`` for k = intercept, each covariate.
    """
    xa = one_sample_matrix(ca, na)
    xb = one_sample_matrix(cb, nb)
    p = xa.shape[1]
    x = np.zeros((na + nb, 2 * p))
    x[:na, :p] = xa
    x[na:, p:] = xb
    return x


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


@dataclass
class OlsFit:
    """Contrast estimates and t for one OLS model on [N, V] data."""

    est: Tensor  # [k, V]
    t: Tensor  # [k, V]
    dof: int


def ols_contrasts(y: Tensor, x: np.ndarray, contrasts: np.ndarray) -> OlsFit:
    """OLS of ``y`` [N, V] on ``x`` [N, p]; t for each row of ``contrasts`` [k, p].

    The design is tiny, so it is factored once in float64 on the host; only the
    [N, V] products touch the device.  A voxel whose residual is exactly zero
    (all subjects identical, e.g. outside the data) gets t = 0, as 3dttest++.
    """
    n, p = x.shape
    dof = n - p
    if dof < 1:
        raise ValueError(f"no residual degrees of freedom: {n} samples, {p} parameters")
    xtx_inv = np.linalg.inv(x.T @ x)
    pinv = xtx_inv @ x.T  # [p, N]
    dt = y.dtype
    dev = y.device
    beta = torch.as_tensor(pinv, dtype=dt, device=dev) @ y  # [p, V]
    resid = y - torch.as_tensor(x, dtype=dt, device=dev) @ beta
    s2 = (resid * resid).sum(0) / dof
    c = torch.as_tensor(contrasts, dtype=dt, device=dev)
    est = c @ beta
    var_c = torch.as_tensor(
        np.einsum("kp,pq,kq->k", contrasts, xtx_inv, contrasts), dtype=dt, device=dev
    )
    se = torch.sqrt(s2.unsqueeze(0) * var_c.unsqueeze(1))
    t = torch.where(
        se > 0, est / torch.where(se > 0, se, torch.ones_like(se)), torch.zeros_like(se)
    )
    return OlsFit(est=est, t=t, dof=dof)


def welch(ya: Tensor, yb: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """``(meanA - meanB, t, Satterthwaite dof)`` per voxel."""
    na, nb = ya.shape[0], yb.shape[0]
    va = ya.var(0, unbiased=True) / na
    vb = yb.var(0, unbiased=True) / nb
    diff = ya.mean(0) - yb.mean(0)
    se2 = va + vb
    ok = se2 > 0
    safe = torch.where(ok, se2, torch.ones_like(se2))
    t = torch.where(ok, diff / torch.sqrt(safe), torch.zeros_like(diff))
    den = va * va / (na - 1) + vb * vb / (nb - 1)
    dof = torch.where(den > 0, se2 * se2 / torch.where(den > 0, den, torch.ones_like(den)), 1.0)
    return diff, t, dof


# ---------------------------------------------------------------------------
# t -> z
# ---------------------------------------------------------------------------


def t_to_z_np(t: np.ndarray, dof) -> np.ndarray:
    """Exact z with the same tail probability as ``t`` (sign kept), via log-sf."""
    from scipy.special import ndtri_exp
    from scipy.stats import t as student_t

    t = np.asarray(t, dtype=np.float64)
    logsf = student_t.logsf(np.abs(t), dof)
    z = -ndtri_exp(logsf)
    return np.sign(t) * np.maximum(z, 0.0)


class TToZ:
    """t → z for a fixed dof, on the device, by interpolating an exact table.

    torch has no Student-t CDF.  z(|t|) is smooth and monotone, so a table
    uniform in ``asinh(|t|)`` and linear interpolation reproduce the scipy
    value to ~1e-6 over the range clustering ever looks at; beyond the table it
    extends with the last slope (|t| > 60 does not occur in a null).
    """

    def __init__(self, dof: float, device: torch.device, n: int = 16384, t_max: float = 60.0):
        u = np.linspace(0.0, np.arcsinh(t_max), n)
        self._z = torch.as_tensor(t_to_z_np(np.sinh(u), dof), dtype=torch.float32, device=device)
        self._du = float(u[1] - u[0])
        self._n = n

    def __call__(self, t: Tensor) -> Tensor:
        u = torch.asinh(t.abs()) / self._du
        i = u.floor().clamp(max=self._n - 2)
        w = u - i
        il = i.long()
        z = self._z[il] * (1 - w) + self._z[il + 1] * w
        return torch.copysign(z, t)


# ---------------------------------------------------------------------------
# Bricks in 3dttest++ order
# ---------------------------------------------------------------------------


@dataclass
class Brick:
    label: str
    data: Tensor  # [V]
    stat: tuple[int, tuple[float, ...]] | None = None  # (AFNI code, params)


FITT = 3
FIZT = 5


def _block(prefix: str, fit: OlsFit, cov_names, toz: bool) -> list[Brick]:
    names = ["mean", *cov_names]
    out = []
    for k, nm in enumerate(names):
        out.append(Brick(f"{prefix}_{nm}", fit.est[k]))
        tlab = "Tstat" if nm == "mean" else f"{nm}_Tstat"
        if toz:
            z = torch.as_tensor(t_to_z_np(fit.t[k].cpu().numpy(), fit.dof), dtype=fit.t.dtype)
            out.append(Brick(f"{prefix}_{tlab.replace('Tstat', 'Zscr')}", z, (FIZT, ())))
        else:
            out.append(Brick(f"{prefix}_{tlab}", fit.t[k], (FITT, (float(fit.dof),))))
    return out


def to_z_bricks(bricks: list[Brick]) -> list[Brick]:
    """``-toz``: every t brick rewritten as the z with the same tail probability."""
    out = []
    for b in bricks:
        if b.stat is not None and b.stat[0] == FITT:
            z = t_to_z_np(b.data.detach().cpu().numpy(), b.stat[1][0])
            data = torch.as_tensor(z, dtype=b.data.dtype, device=b.data.device)
            out.append(Brick(b.label.replace("Tstat", "Zscr"), data, (FIZT, ())))
        else:
            out.append(b)
    return out


def one_sample_fit(y: Tensor, cov: np.ndarray | None) -> OlsFit:
    x = one_sample_matrix(cov, y.shape[0])
    return ols_contrasts(y, x, np.eye(x.shape[1]))


def main_test(design: TtestDesign, ya: Tensor, yb: Tensor | None):
    """``(y, X, c)`` of the test that cluster inference thresholds.

    The A-B intercept difference for two sets, the A-B mean for paired, the set
    A mean otherwise — the first t brick 3dttest++ writes.
    """
    ca, cb = design.centered_covariates()
    if not design.two_sets:
        x = one_sample_matrix(ca, design.n_a)
        c = np.zeros(x.shape[1])
        c[0] = 1.0
        return ya, x, c
    assert yb is not None
    if design.paired:
        x = one_sample_matrix(ca, design.n_a)
        c = np.zeros(x.shape[1])
        c[0] = 1.0
        return ya - yb, x, c
    x = two_sample_matrix(ca, cb, design.n_a, design.n_b)
    p = x.shape[1] // 2
    c = np.zeros(x.shape[1])
    c[0], c[p] = 1.0, -1.0
    return torch.cat([ya, yb]), x, c


def run_ttest(
    design: TtestDesign, ya: Tensor, yb: Tensor | None = None, *, toz: bool = False
) -> list[Brick]:
    """All output bricks, in 3dttest++'s order and with its labels."""
    ca, cb = design.centered_covariates()
    la, lb = design.label_a, design.label_b
    names = design.cov_names
    if not design.two_sets:
        return _block(la, one_sample_fit(ya, ca), names, toz)
    assert yb is not None
    if design.unpooled:
        diff, t, dof = welch(ya, yb)
        z = torch.as_tensor(
            t_to_z_np(t.cpu().numpy(), dof.cpu().numpy()), dtype=ya.dtype, device=ya.device
        )
        out = [Brick(f"{la}-{lb}_mean", diff), Brick(f"{la}-{lb}_Zscr", z, (FIZT, ()))]
        out += _block(la, one_sample_fit(ya, None), (), True)
        out += _block(lb, one_sample_fit(yb, None), (), True)
        return out
    if design.paired:
        diff_fit = one_sample_fit(ya - yb, ca)
    else:
        x = two_sample_matrix(ca, cb, design.n_a, design.n_b)
        p = x.shape[1] // 2
        cons = np.zeros((p, 2 * p))
        for k in range(p):
            cons[k, k], cons[k, p + k] = 1.0, -1.0
        diff_fit = ols_contrasts(torch.cat([ya, yb]), x, cons)
    out = _block(f"{la}-{lb}", diff_fit, names, toz)
    out += _block(la, one_sample_fit(ya, ca), names, toz)
    out += _block(lb, one_sample_fit(yb, cb), names, toz)
    return out


# ---------------------------------------------------------------------------
# Null maps for cluster inference
# ---------------------------------------------------------------------------


@dataclass
class OlsNull:
    """Batches of null z-maps for one OLS contrast, from its residuals.

    Each iteration flips the sign of every residual at random and, with
    ``permute``, also shuffles which row of the design each residual sits in
    (3dttest++'s ``-randomsign`` / ``-permute``).  All flips are drawn up front
    from ``seed``, so iteration ``i`` is the same map whatever batch size the
    memory plan chooses.

    For y* = Pi (s . r):  Q^T y* = (Q[Pi^-1] . s)^T r, the contrast numerator is
    the same with Q w in place of Q, and ||y*||^2 = ||r||^2.
    """

    y: Tensor  # [N, V] on device (already blurred, if blurring)
    x: np.ndarray  # [N, p]
    c: np.ndarray  # [p]
    n_iter: int
    seed: int = 0
    permute: bool = False
    dof: int = field(init=False)

    def __post_init__(self) -> None:
        n, p = self.x.shape
        self.dof = n - p
        dev, dt = self.y.device, torch.float32
        q, r = np.linalg.qr(self.x)
        w = np.linalg.solve(r.T, self.c)  # R^-T c
        self._q = torch.as_tensor(q, dtype=dt, device=dev)  # [N, p]
        self._qw = torch.as_tensor(q @ w, dtype=dt, device=dev)  # [N]
        self._scale = float(np.sqrt(self.c @ np.linalg.solve(self.x.T @ self.x, self.c)))
        beta = torch.as_tensor(np.linalg.pinv(self.x), dtype=dt, device=dev) @ self.y.to(dt)
        self._r = self.y.to(dt) - torch.as_tensor(self.x, dtype=dt, device=dev) @ beta
        self._ss = (self._r * self._r).sum(0)
        rng = np.random.default_rng(self.seed)
        self._signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(self.n_iter, n))
        self._perm = (
            np.stack([rng.permutation(n) for _ in range(self.n_iter)]) if self.permute else None
        )
        self._toz = TToZ(self.dof, dev)

    @property
    def n_distinct(self) -> float:
        """How many distinct relabelings exist (2^N signs, times N! with -permute)."""
        from math import lgamma, log

        n = self.x.shape[0]
        logn = n * log(2.0) + (lgamma(n + 1) if self.permute else 0.0)
        return float(np.exp(min(logn, 700.0)))

    def bytes_per_map(self) -> int:
        """Device bytes one null map costs while it is built (t, proj, z)."""
        return 4 * int(self._r.shape[1]) * (self.x.shape[1] + 3)

    def t_maps(self, start: int, nb: int) -> Tensor:
        dev = self._r.device
        s = torch.as_tensor(self._signs[start : start + nb], device=dev)  # [nb, N]
        if self._perm is None:
            a = self._qw.unsqueeze(0) * s  # [nb, N]
            b = self._q.T.unsqueeze(0) * s.unsqueeze(1)  # [nb, p, N]
        else:
            inv = torch.as_tensor(np.argsort(self._perm[start : start + nb], axis=1), device=dev)
            a = self._qw[inv] * s
            b = self._q[inv].transpose(1, 2) * s.unsqueeze(1)
        num = a @ self._r  # [nb, V]
        proj = torch.einsum("bpn,nv->bpv", b, self._r)
        rss = (self._ss.unsqueeze(0) - (proj * proj).sum(1)).clamp_min(0)
        se = torch.sqrt(rss / self.dof) * self._scale
        return torch.where(se > 0, num / torch.where(se > 0, se, torch.ones_like(se)), 0.0)

    def __call__(self, start: int, nb: int) -> Tensor:
        """A :data:`cluster_inference.NullSource`: ``[nb, V]`` null z-maps."""
        return self._toz(self.t_maps(start, nb))
