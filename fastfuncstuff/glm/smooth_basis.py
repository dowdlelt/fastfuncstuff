"""Roughness-penalized FIR / TENT fits with a per-voxel smoothing strength.

The knot values ``beta`` of a FIR/TENT response are estimated by

    min ||y - X beta - N gamma||^2 + lam * ||D beta||^2

where ``D`` is the ``order``-th difference across neighbouring knots of each
condition (no penalty across conditions) and ``N`` is the unpenalized
nuisance (drift polynomials, -ortvec).  This is the smooth FIR of Goutte,
Nielsen & Hansen (2000) / Marrelec et al. (2003).  The penalty is largest on
exactly the knot pattern mid-TR onsets cannot see (+,-,+,-), leaves straight
lines free, and makes grids finer than the TR solvable at all.

``lam`` is chosen per voxel without held-out data, by REML (the default --
Wood 2011 finds it avoids GCV's occasional severe undersmoothing) or GCV.
Both are closed form on a lambda grid after ONE simultaneous diagonalization
of the task Gram matrix and the penalty, shared by every voxel:

    B = A + P = R'R,   R^-T P R^-1 = V diag(s) V',   0 <= s <= 1
    beta(lam) = W diag(d) z,   W = R^-1 V,   z = (X W)' y,   d = 1/((1-s) + lam s)

so each lambda costs O(K) per voxel.  Working from ``A + P`` rather than
``A`` keeps this valid when ``A`` alone is singular (knots finer than the
timing resolves): the penalty supplies what the data cannot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.linalg import block_diag
from tqdm.auto import tqdm

from fastfuncstuff.glm.xval import cod_from_ss_residual
from fastfuncstuff.memory import estimate_chunk_size

#: log10 lambda grid.  lambda is relative (penalty scaled to the Gram trace),
#: so the same grid spans "no smoothing" to "straight lines" for any design.
DEFAULT_LOG10_GRID = np.linspace(-5.0, 7.0, 49)
SELECTION_METHODS = ("reml", "gcv", "fixed")
#: Rules for choosing lambda (and, with several penalties, the penalty).
SMOOTH_RULES = ("reml", "gcv", "loro", "fixed")
#: Relative jitter on a Gaussian-process prior covariance before inverting it:
#: a floor on the prior variance of rough components, which also keeps the
#: precision matrix finite for fine knots and long length scales.
GP_JITTER = 1e-4


def roughness_penalty(
    n_basis_per_condition: list[int], order: int = 2, zero_edges: bool = False
) -> np.ndarray:
    """Block-diagonal ``D'D`` over each condition's knots.

    ``zero_edges`` (TENTzero/CSPLINzero) differences the FULL knot vector,
    whose pinned-zero edge knots are dropped from the design, so the
    response is also kept smooth into its zero ends.
    """
    blocks = []
    for n in n_basis_per_condition:
        full = n + 2 if zero_edges else n
        if full <= order:
            blocks.append(np.zeros((n, n)))
            continue
        d = np.diff(np.eye(full), order, axis=0)
        if zero_edges:
            d = d[:, 1:-1]
        blocks.append(d.T @ d)
    return block_diag(*blocks)


def parse_penalty_spec(spec: str) -> tuple[str, float]:
    """``"diff2"`` -> ("diff", 2); ``"gp:4"`` -> ("gp", 4.0 s length scale)."""
    text = spec.strip().lower()
    if text.startswith("diff") and text[4:].isdigit() and int(text[4:]) >= 1:
        return "diff", float(int(text[4:]))
    if text.startswith("gp:"):
        try:
            scale = float(text[3:])
        except ValueError:
            scale = -1.0
        if scale > 0:
            return "gp", scale
    raise ValueError(f"penalty must be diffN (N >= 1) or gp:SECONDS, got {spec!r}")


def penalty_matrix(
    spec: str,
    n_basis_per_condition: list[int],
    knot_dt_per_condition: list[float],
    zero_edges: bool = False,
) -> np.ndarray:
    """Block-diagonal penalty for one spec over each condition's knots.

    ``diffN`` is the N-th difference roughness penalty (:func:`roughness_penalty`;
    its null space is polynomials of degree < N per condition).  ``gp:L`` is
    the precision of a squared-exponential Gaussian-process prior with length
    scale ``L`` seconds on the knot values (Goutte, Nielsen & Hansen 2000):
    full rank, shrinking toward zero, smooth on the scale ``L`` regardless of
    how finely the knots are spaced.  ``zero_edges`` conditions on the pinned
    edge knots of TENTzero/CSPLINzero in both cases.
    """
    kind, value = parse_penalty_spec(spec)
    if kind == "diff":
        return roughness_penalty(n_basis_per_condition, order=int(value), zero_edges=zero_edges)
    blocks = []
    for n, dt in zip(n_basis_per_condition, knot_dt_per_condition, strict=True):
        full = n + 2 if zero_edges else n
        t = np.arange(full) * dt
        cov = np.exp(-0.5 * ((t[:, None] - t[None, :]) / value) ** 2)
        prec = np.linalg.inv(cov + GP_JITTER * np.eye(full))
        prec = (prec + prec.T) / 2
        blocks.append(prec[1:-1, 1:-1] if zero_edges else prec)
    return block_diag(*blocks)


@dataclass
class SmoothBasisFit:
    betas: torch.Tensor  # (V, K) task knot values
    lam: torch.Tensor  # (V,) chosen relative lambda
    edf: torch.Tensor  # (V,) effective degrees of freedom of the task block
    r2: torch.Tensor  # (V,) in-sample R^2, same definition as fit_glm
    method: str


@dataclass
class _Spectrum:
    g: np.ndarray  # (T, K) X_tilde @ W
    w: np.ndarray  # (K, K)
    s: np.ndarray  # (K,)
    q_nuis: np.ndarray  # (T, p) orthonormal nuisance basis
    n_eff: int  # T - rank(nuisance)
    rank_penalty: int


def _spectrum(design: np.ndarray, n_task: int, penalty: np.ndarray) -> _Spectrum:
    x = design[:, :n_task]
    nuis = design[:, n_task:]
    nuis = nuis[:, np.abs(nuis).sum(axis=0) > 0]
    if nuis.shape[1]:
        u, sv, _ = np.linalg.svd(nuis, full_matrices=False)
        q = u[:, sv > sv.max() * 1e-10]
    else:
        q = np.zeros((design.shape[0], 0))
    x = x - q @ (q.T @ x)
    gram = x.T @ x
    # Relative lambda: the penalty's scale matches the data term's.
    pen = penalty * (np.trace(gram) / max(np.trace(penalty), 1e-300))
    b = gram + pen
    b += np.eye(n_task) * (1e-10 * np.trace(b) / n_task)  # null(A) & null(P) guard
    r = np.linalg.cholesky(b).T  # b = r' r
    r_inv = np.linalg.inv(r)
    m = r_inv.T @ pen @ r_inv
    s, v = np.linalg.eigh((m + m.T) / 2)
    s = np.clip(s, 0.0, 1.0)
    w = r_inv @ v
    p_evals = np.linalg.eigvalsh(pen)
    rank_p = int((p_evals > p_evals.max() * 1e-9).sum()) if p_evals.size else 0
    return _Spectrum(
        g=x @ w,
        w=w,
        s=s,
        q_nuis=q,
        n_eff=design.shape[0] - q.shape[1],
        rank_penalty=rank_p,
    )


def _criterion(
    method: str,
    z2: torch.Tensor,  # (V, K)
    yy: torch.Tensor,  # (V,)
    s: torch.Tensor,  # (K,)
    lams: torch.Tensor,  # (L,)
    n_eff: int,
    rank_p: int,
) -> torch.Tensor:
    one_minus_s = 1.0 - s
    denom = one_minus_s[None, :] + lams[:, None] * s[None, :]  # (L, K)
    d = 1.0 / denom
    if method == "reml":
        # -2 * restricted log-likelihood with sigma^2 profiled out, constants dropped:
        # (n - M_p) log(RSS + lam b'Pb) + log|A + lam P| - rank(P) log lam
        q = (yy[:, None] - z2 @ d.T).clamp_min(1e-30)
        logdet = torch.log(denom).sum(dim=1) - rank_p * torch.log(lams)
        return (n_eff - (s.numel() - rank_p)) * torch.log(q) + logdet[None, :]
    rss = (yy[:, None] - z2 @ (2 * d - d * d * one_minus_s[None, :]).T).clamp_min(1e-30)
    edf = (one_minus_s[None, :] * d).sum(dim=1)
    return n_eff * rss / (n_eff - edf).clamp_min(1.0)[None, :] ** 2


def _refine_log_lambda(crit: torch.Tensor, log_grid: torch.Tensor) -> torch.Tensor:
    """Per-voxel argmin on the grid, refined by a parabola through its neighbours."""
    idx = crit.argmin(dim=1)
    n = log_grid.numel()
    inner = (idx > 0) & (idx < n - 1)
    i0 = idx.clamp(1, n - 2)
    rows = torch.arange(crit.shape[0], device=crit.device)
    f0, f1, f2 = crit[rows, i0 - 1], crit[rows, i0], crit[rows, i0 + 1]
    curv = f0 - 2 * f1 + f2
    step = torch.where(curv > 0, 0.5 * (f0 - f2) / curv, torch.zeros_like(curv)).clamp(-1, 1)
    h = log_grid[1] - log_grid[0]
    refined = log_grid[i0] + step * h
    return torch.where(inner, refined, log_grid[idx])


def fit_smooth_basis(
    data: torch.Tensor,
    design: torch.Tensor,
    n_task: int,
    penalty: np.ndarray,
    *,
    method: str = "reml",
    lam: float | torch.Tensor | None = None,
    log10_grid: np.ndarray | None = None,
    device: torch.device | None = None,
    chunk_size: int | None = None,
    verbose: bool = False,
    time_index: torch.Tensor | None = None,
) -> SmoothBasisFit:
    """Penalized fit of the first ``n_task`` design columns, lambda per voxel.

    ``data`` is ``(V, T)``, ``design`` ``(T, n_task + n_nuisance)`` -- the packed
    concatenated GLM form (task block first, block-diagonal nuisance after).
    ``method="fixed"`` uses ``lam`` -- one value, or one per voxel (a ``(V,)``
    tensor); ``"reml"``/``"gcv"`` pick it per voxel on ``log10_grid``.  The decomposition is float64 on the CPU
    (K x K, tiny); voxel work streams in chunks on ``device``.  ``time_index``
    fits a subset of timepoints (``design`` already sliced to it) without
    copying ``data``: the slice is taken per chunk.
    """
    if method not in SELECTION_METHODS:
        raise ValueError(f"method must be one of {SELECTION_METHODS}, got {method!r}")
    if method == "fixed" and (lam is None or bool((torch.as_tensor(lam) <= 0).any())):
        raise ValueError("method='fixed' needs a positive lam")
    device = device if device is not None else torch.device("cpu")
    spec = _spectrum(design.detach().cpu().double().numpy(), n_task, penalty)

    log_grid = torch.as_tensor(
        DEFAULT_LOG10_GRID if log10_grid is None else log10_grid, dtype=torch.float64
    ).to(device) * np.log(10.0)
    g = torch.as_tensor(spec.g, dtype=torch.float32, device=device)
    q = torch.as_tensor(spec.q_nuis, dtype=torch.float32, device=device)
    s = torch.as_tensor(spec.s, dtype=torch.float64, device=device)
    w = torch.as_tensor(spec.w, dtype=torch.float64, device=device)

    n_vox = data.shape[0]
    n_t = design.shape[0]
    if chunk_size is None:
        chunk_size = estimate_chunk_size(
            n_vox, n_t, n_task + int(log_grid.numel()), device, operation="glm"
        )
    out_b = torch.empty((n_vox, n_task), dtype=torch.float32)
    out_lam = torch.empty(n_vox, dtype=torch.float32)
    out_edf = torch.empty(n_vox, dtype=torch.float32)
    out_r2 = torch.empty(n_vox, dtype=torch.float32)
    starts = range(0, n_vox, chunk_size)
    for a in tqdm(
        starts,
        desc="  Smooth fit",
        unit="chunk",
        leave=True,
        disable=not verbose or len(starts) < 2,
    ):
        b = min(a + chunk_size, n_vox)
        y = data[a:b] if time_index is None else data[a:b][:, time_index]
        y = y.to(device=device, dtype=torch.float32)
        y_t = y - (y @ q) @ q.T if q.shape[1] else y
        z = (y_t @ g).double()
        z2 = z * z
        yy = (y_t.double() ** 2).sum(dim=1)
        if method == "fixed":
            lam_t = torch.as_tensor(lam, dtype=torch.float64)
            lam_t = lam_t[a:b] if lam_t.ndim else lam_t.expand(b - a)
            log_lam = torch.log(lam_t).to(device)
        else:
            crit = _criterion(method, z2, yy, s, torch.exp(log_grid), spec.n_eff, spec.rank_penalty)
            log_lam = _refine_log_lambda(crit, log_grid)
        lam_v = torch.exp(log_lam)
        d = 1.0 / ((1.0 - s)[None, :] + lam_v[:, None] * s[None, :])
        betas = (d * z) @ w.T
        rss = (yy - (z2 * (2 * d - d * d * (1.0 - s)[None, :])).sum(dim=1)).clamp_min(0.0)
        out_b[a:b] = betas.float().cpu()
        # Empty voxels (nothing left after the nuisance) have no lambda to choose:
        # report lambda 1 / edf 0 rather than the grid edge a flat criterion lands on.
        empty = yy <= 1e-12 * n_t
        out_lam[a:b] = torch.where(empty, torch.ones_like(lam_v), lam_v).float().cpu()
        edf = ((1.0 - s)[None, :] * d).sum(dim=1)
        out_edf[a:b] = torch.where(empty, torch.zeros_like(edf), edf).float().cpu()
        out_r2[a:b] = cod_from_ss_residual(y, rss.float()).cpu()
    return SmoothBasisFit(betas=out_b, lam=out_lam, edf=out_edf, r2=out_r2, method=method)


@dataclass
class SmoothSelection:
    """Result of :func:`fit_smooth_selected`."""

    fit: SmoothBasisFit
    #: Index into the candidate penalties, per voxel.
    penalty_index: torch.Tensor
    #: Leave-one-run-out COD of the chosen configuration (None without a LORO pass).
    xval_r2: torch.Tensor | None
    #: True when held-out runs also CHOSE something (lambda or penalty): the
    #: xval R^2 of the winner is then optimistically biased.
    xval_selection_biased: bool


@dataclass
class _FoldPenalty:
    spec: _Spectrum
    m_gram: torch.Tensor  # (K, K) M'M, M = X_test W
    m: torch.Tensor  # (T_test, K)
    q_test: torch.Tensor  # (T_test, p) held-out run's nuisance basis
    train: torch.Tensor
    test: torch.Tensor


def _fold_penalty(design64: np.ndarray, n_task: int, penalty, train, test, device) -> _FoldPenalty:
    spec = _spectrum(design64[train.numpy()], n_task, penalty)
    x_test = design64[test.numpy(), :n_task]
    nuis = design64[test.numpy(), n_task:]
    nuis = nuis[:, np.abs(nuis).sum(axis=0) > 0]
    q = np.zeros((test.numel(), 0))
    if nuis.shape[1]:
        u, sv, _ = np.linalg.svd(nuis, full_matrices=False)
        q = u[:, sv > sv.max() * 1e-10]
        x_test = x_test - q @ (q.T @ x_test)
    m = x_test @ spec.w
    return _FoldPenalty(
        spec=spec,
        m_gram=torch.as_tensor(m.T @ m, dtype=torch.float64, device=device),
        m=torch.as_tensor(m, dtype=torch.float64, device=device),
        q_test=torch.as_tensor(q, dtype=torch.float64, device=device),
        train=train,
        test=test,
    )


def _heldout_ss(v: torch.Tensor, u: torch.Tensor, yy_test: torch.Tensor, m_gram: torch.Tensor):
    """||y - M v||^2 for per-voxel coefficient vectors v, via M'y and M'M."""
    return yy_test - 2.0 * (v * u).sum(dim=-1) + ((v @ m_gram) * v).sum(dim=-1)


def fit_smooth_selected(
    data: torch.Tensor,
    design: torch.Tensor,
    n_task: int,
    penalties: list[np.ndarray],
    run_starts: list[int],
    *,
    rule: str = "reml",
    lam: float | None = None,
    xval: bool = False,
    log10_grid: np.ndarray | None = None,
    device: torch.device | None = None,
    verbose: bool = False,
) -> SmoothSelection:
    """Smooth FIR/TENT fit choosing lambda by ``rule`` and, with several
    ``penalties``, the penalty by held-out runs -- per voxel.

    ``rule``: ``reml``/``gcv`` choose lambda inside each penalty from the data
    being fitted; ``loro`` chooses it (jointly with the penalty) by
    leave-one-run-out prediction error; ``fixed`` uses ``lam``.  A LORO pass
    runs when the rule is ``loro``, when there is more than one penalty, or
    when ``xval`` is asked for; its held-out COD of the winning configuration
    comes back as ``xval_r2`` for free.  Held-out error for every lambda is a
    closed form in the fold's shared spectrum -- ``||y - M d.z||^2`` from
    ``M'y`` and ``M'M`` -- so the grid costs no refits.  The final betas are
    refitted on all runs with the chosen configuration.
    """
    if rule not in SMOOTH_RULES:
        raise ValueError(f"rule must be one of {SMOOTH_RULES}, got {rule!r}")
    if not penalties:
        raise ValueError("need at least one penalty")
    device = device if device is not None else torch.device("cpu")
    n_vox, n_t = data.shape
    n_pen = len(penalties)
    need_loro = rule == "loro" or n_pen > 1 or xval
    if need_loro and len(run_starts) < 2:
        if rule == "loro" or n_pen > 1:
            raise ValueError("choosing by held-out runs needs at least two runs")
        need_loro = False
    fit_method = "fixed" if rule in ("fixed", "loro") else rule
    if not need_loro:
        fit = fit_smooth_basis(
            data,
            design,
            n_task,
            penalties[0],
            method=fit_method,
            lam=lam,
            log10_grid=log10_grid,
            device=device,
            verbose=verbose,
        )
        return SmoothSelection(fit, torch.zeros(n_vox, dtype=torch.long), None, False)

    log_grid = torch.as_tensor(
        DEFAULT_LOG10_GRID if log10_grid is None else log10_grid, dtype=torch.float64
    ).to(device) * np.log(10.0)
    lams = torch.exp(log_grid)
    bounds = list(run_starts) + [n_t]
    design64 = design.detach().cpu().double().numpy()
    folds = []
    for r in range(len(run_starts)):
        test = torch.arange(bounds[r], bounds[r + 1])
        train = torch.cat([torch.arange(0, bounds[r]), torch.arange(bounds[r + 1], n_t)])
        folds.append(
            [_fold_penalty(design64, n_task, pen, train, test, device) for pen in penalties]
        )

    n_lam = int(lams.numel()) if rule == "loro" else 1
    chunk = estimate_chunk_size(n_vox, n_t, n_task * (1 + n_pen * n_lam), device, operation="glm")
    choice_pen = torch.zeros(n_vox, dtype=torch.long)
    choice_lam = torch.ones(n_vox, dtype=torch.float64)
    ss_best = torch.zeros(n_vox, dtype=torch.float64)
    ss_tot = torch.zeros(n_vox, dtype=torch.float64)
    starts = range(0, n_vox, chunk)
    for a in tqdm(starts, desc="  LORO selection", unit="chunk", leave=True, disable=not verbose):
        b = min(a + chunk, n_vox)
        ss = torch.zeros((b - a, n_pen, n_lam), dtype=torch.float64, device=device)
        total = torch.zeros(b - a, dtype=torch.float64, device=device)
        total_sq = torch.zeros(b - a, dtype=torch.float64, device=device)
        n_test = 0
        y_all = data[a:b].to(device=device, dtype=torch.float64)
        for fold in folds:
            y_te = y_all[:, fold[0].test.to(device)]
            if fold[0].q_test.shape[1]:
                y_te = y_te - (y_te @ fold[0].q_test) @ fold[0].q_test.T
            yy_te = (y_te * y_te).sum(dim=1)
            total += y_te.sum(dim=1)
            total_sq += yy_te
            n_test += y_te.shape[1]
            y_tr = y_all[:, fold[0].train.to(device)]
            for p, fp in enumerate(fold):
                sp = fp.spec
                q_tr = torch.as_tensor(sp.q_nuis, dtype=torch.float64, device=device)
                y_trp = y_tr - (y_tr @ q_tr) @ q_tr.T if q_tr.shape[1] else y_tr
                z = y_trp @ torch.as_tensor(sp.g, dtype=torch.float64, device=device)
                s_t = torch.as_tensor(sp.s, dtype=torch.float64, device=device)
                u = y_te @ fp.m
                if rule == "loro":
                    for li in range(n_lam):
                        d = 1.0 / ((1.0 - s_t) + lams[li] * s_t)
                        ss[:, p, li] += _heldout_ss(d * z, u, yy_te, fp.m_gram)
                    continue
                if rule == "fixed":
                    lam_v = torch.full((b - a,), float(lam), dtype=torch.float64, device=device)
                else:
                    yy_tr = (y_trp * y_trp).sum(dim=1)
                    crit = _criterion(rule, z * z, yy_tr, s_t, lams, sp.n_eff, sp.rank_penalty)
                    lam_v = torch.exp(_refine_log_lambda(crit, log_grid))
                d = 1.0 / ((1.0 - s_t)[None, :] + lam_v[:, None] * s_t[None, :])
                ss[:, p, 0] += _heldout_ss(d * z, u, yy_te, fp.m_gram)
        flat = ss.reshape(b - a, -1)
        best = flat.argmin(dim=1)
        rows = torch.arange(b - a, device=device)
        choice_pen[a:b] = (best // n_lam).cpu()
        if rule == "loro":
            curve = ss[rows, best // n_lam, :]
            choice_lam[a:b] = torch.exp(_refine_log_lambda(curve, log_grid)).cpu()
        ss_best[a:b] = flat[rows, best].cpu()
        ss_tot[a:b] = (total_sq - total * total / n_test).cpu()

    live = ss_tot > 1e-12 * n_t
    xval_r2 = torch.zeros(n_vox, dtype=torch.float64)
    xval_r2[live] = 1.0 - ss_best[live] / ss_tot[live]

    # Final fit on all runs, one pass per penalty over the voxels that chose it.
    betas = torch.zeros((n_vox, n_task), dtype=torch.float32)
    out_lam = torch.ones(n_vox, dtype=torch.float32)
    out_edf = torch.zeros(n_vox, dtype=torch.float32)
    out_r2 = torch.zeros(n_vox, dtype=torch.float32)
    for p, pen in enumerate(penalties):
        idx = torch.nonzero(choice_pen == p).flatten()
        if idx.numel() == 0:
            continue
        sub = fit_smooth_basis(
            data[idx],
            design,
            n_task,
            pen,
            method=fit_method,
            lam=choice_lam[idx] if rule == "loro" else lam,
            log10_grid=log10_grid,
            device=device,
            verbose=verbose,
        )
        betas[idx], out_lam[idx], out_edf[idx], out_r2[idx] = sub.betas, sub.lam, sub.edf, sub.r2
    empty = ~live
    out_lam[empty], out_edf[empty] = 1.0, 0.0
    return SmoothSelection(
        fit=SmoothBasisFit(betas=betas, lam=out_lam, edf=out_edf, r2=out_r2, method=rule),
        penalty_index=choice_pen,
        xval_r2=xval_r2.float(),
        xval_selection_biased=rule == "loro" or n_pen > 1,
    )
