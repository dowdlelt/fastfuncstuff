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
    w, s = _gram_spectrum(x.T @ x, penalty)
    return _Spectrum(
        g=x @ w,
        w=w,
        s=s,
        q_nuis=q,
        n_eff=design.shape[0] - q.shape[1],
        rank_penalty=_penalty_rank(penalty),
    )


def _gram_spectrum(gram: np.ndarray, penalty: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(W, s)`` of the simultaneous diagonalization of ``gram`` and the penalty,
    the penalty scaled to ``trace(gram)`` (the relative-lambda convention)."""
    n_task = gram.shape[0]
    pen = penalty * (np.trace(gram) / max(np.trace(penalty), 1e-300))
    b = gram + pen
    b += np.eye(n_task) * (1e-10 * np.trace(b) / n_task)  # null(A) & null(P) guard
    r = np.linalg.cholesky(b).T  # b = r' r
    r_inv = np.linalg.inv(r)
    m = r_inv.T @ pen @ r_inv
    s, v = np.linalg.eigh((m + m.T) / 2)
    return r_inv @ v, np.clip(s, 0.0, 1.0)


def _penalty_rank(penalty: np.ndarray) -> int:
    p_evals = np.linalg.eigvalsh(penalty)
    return int((p_evals > p_evals.max() * 1e-9).sum()) if p_evals.size else 0


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


def _fold_penalty(
    design64: np.ndarray, n_task: int, penalty, train, test, device, score64: np.ndarray
) -> _FoldPenalty:
    spec = _spectrum(design64[train.numpy()], n_task, penalty)
    x_test = score64[test.numpy(), :n_task]
    nuis = score64[test.numpy(), n_task:]
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
    score_data: torch.Tensor | None = None,
    score_design: torch.Tensor | None = None,
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

    ``score_data``/``score_design`` (same shapes as ``data``/``design``) are
    what held-out runs are scored against, when that differs from what is
    fitted -- prewhitened fits are trained on whitened data but scored on the
    raw series, so their R^2 compares with an unwhitened fit's.
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
    score64 = design64 if score_design is None else score_design.detach().cpu().double().numpy()
    score_src = data if score_data is None else score_data
    folds = []
    for r in range(len(run_starts)):
        test = torch.arange(bounds[r], bounds[r + 1])
        train = torch.cat([torch.arange(0, bounds[r]), torch.arange(bounds[r + 1], n_t)])
        folds.append(
            [
                _fold_penalty(design64, n_task, pen, train, test, device, score64)
                for pen in penalties
            ]
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
        y_score = y_all if score_data is None else score_src[a:b].to(device, torch.float64)
        for fold in folds:
            y_te = y_score[:, fold[0].test.to(device)]
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


#: Held-out R^2 a voxel needs, under OLS or honest REML, to count as signal
#: when choosing one global lambda.
GLOBAL_SIGNAL_R2 = 0.05
#: The unpenalized reference fit's lambda (relative): OLS for any usable design.
OLS_LAMBDA = 1e-6


@dataclass
class GlobalLambda:
    """Result of :func:`choose_global_lambda`."""

    lam: float  # relative, like every lambda here
    n_signal: int
    #: Median held-out R^2 of the signal voxels at each grid lambda.
    median_curve: np.ndarray
    log10_grid: np.ndarray
    #: Fell back to every live voxel because none reached ``signal_r2``.
    fallback: bool


def choose_global_lambda(
    data: torch.Tensor,
    design: torch.Tensor,
    n_task: int,
    penalty: np.ndarray,
    run_starts: list[int],
    *,
    signal_r2: float = GLOBAL_SIGNAL_R2,
    rule: str = "reml",
    log10_grid: np.ndarray | None = None,
    device: torch.device | None = None,
    verbose: bool = False,
) -> GlobalLambda:
    """One lambda for every voxel: the grid value that maximizes the median
    leave-one-run-out R^2 over signal voxels.

    Signal voxels are those whose held-out R^2 exceeds ``signal_r2`` under
    OLS or under ``rule`` (REML/GCV, lambda from the training runs only), so
    neither the smoothed nor the unsmoothed fit decides alone.  Taking the
    median over them keeps the choice from being set by the noise voxels,
    which outnumber them and all prefer heavy smoothing.  One scalar chosen
    from many voxels carries negligible selection bias, and applying it
    everywhere makes the fit a linear estimator.  One pass over the data:
    every fold's held-out error at every grid lambda is a closed form.
    """
    if len(run_starts) < 2:
        raise ValueError("choosing a global lambda by held-out runs needs at least two runs")
    device = device if device is not None else torch.device("cpu")
    n_vox, n_t = data.shape
    grid10 = np.asarray(DEFAULT_LOG10_GRID if log10_grid is None else log10_grid, dtype=float)
    log_grid = torch.as_tensor(grid10, dtype=torch.float64, device=device) * np.log(10.0)
    lams = torch.exp(log_grid)
    bounds = list(run_starts) + [n_t]
    design64 = design.detach().cpu().double().numpy()
    folds = []
    for r in range(len(run_starts)):
        test = torch.arange(bounds[r], bounds[r + 1])
        train = torch.cat([torch.arange(0, bounds[r]), torch.arange(bounds[r + 1], n_t)])
        folds.append(_fold_penalty(design64, n_task, penalty, train, test, device, design64))
    n_lam = int(lams.numel())
    curve = torch.zeros((n_vox, n_lam), dtype=torch.float32)
    r2_ols = torch.zeros(n_vox, dtype=torch.float32)
    r2_rule = torch.zeros(n_vox, dtype=torch.float32)
    live_all = torch.zeros(n_vox, dtype=torch.bool)
    chunk = estimate_chunk_size(n_vox, n_t, n_task * (n_lam + 2), device, operation="glm")
    starts = range(0, n_vox, chunk)
    for a in tqdm(
        starts,
        desc="  Global lambda",
        unit="chunk",
        leave=True,
        disable=not verbose or len(starts) < 2,
    ):
        b = min(a + chunk, n_vox)
        y = data[a:b].to(device=device, dtype=torch.float64)
        ss = torch.zeros((b - a, n_lam), dtype=torch.float64, device=device)
        ss_ols = torch.zeros(b - a, dtype=torch.float64, device=device)
        ss_rule = torch.zeros(b - a, dtype=torch.float64, device=device)
        total = torch.zeros(b - a, dtype=torch.float64, device=device)
        total_sq = torch.zeros(b - a, dtype=torch.float64, device=device)
        n_test = 0
        for fp in folds:
            y_te = y[:, fp.test.to(device)]
            if fp.q_test.shape[1]:
                y_te = y_te - (y_te @ fp.q_test) @ fp.q_test.T
            yy_te = (y_te * y_te).sum(dim=1)
            total += y_te.sum(dim=1)
            total_sq += yy_te
            n_test += y_te.shape[1]
            sp = fp.spec
            q_tr = torch.as_tensor(sp.q_nuis, dtype=torch.float64, device=device)
            y_tr = y[:, fp.train.to(device)]
            y_tr = y_tr - (y_tr @ q_tr) @ q_tr.T if q_tr.shape[1] else y_tr
            z = y_tr @ torch.as_tensor(sp.g, dtype=torch.float64, device=device)
            s_t = torch.as_tensor(sp.s, dtype=torch.float64, device=device)
            u = y_te @ fp.m
            for li in range(n_lam):
                d = 1.0 / ((1.0 - s_t) + lams[li] * s_t)
                ss[:, li] += _heldout_ss(d * z, u, yy_te, fp.m_gram)
            d = _shrink(torch.full((b - a,), OLS_LAMBDA, dtype=torch.float64, device=device), s_t)
            ss_ols += _heldout_ss(d * z, u, yy_te, fp.m_gram)
            crit = _criterion(
                rule, z * z, (y_tr * y_tr).sum(dim=1), s_t, lams, sp.n_eff, sp.rank_penalty
            )
            d = _shrink(torch.exp(_refine_log_lambda(crit, log_grid)), s_t)
            ss_rule += _heldout_ss(d * z, u, yy_te, fp.m_gram)
        ss_tot = total_sq - total * total / n_test
        live = ss_tot > 1e-12 * n_t
        safe = torch.where(live, ss_tot, torch.ones_like(ss_tot))
        curve[a:b] = torch.where(live[:, None], 1 - ss / safe[:, None], 0.0).float().cpu()
        r2_ols[a:b] = torch.where(live, 1 - ss_ols / safe, 0.0).float().cpu()
        r2_rule[a:b] = torch.where(live, 1 - ss_rule / safe, 0.0).float().cpu()
        live_all[a:b] = live.cpu()
    signal = (r2_ols > signal_r2) | (r2_rule > signal_r2)
    fallback = not bool(signal.any())
    if fallback:
        signal = live_all if bool(live_all.any()) else torch.ones(n_vox, dtype=torch.bool)
    med = curve[signal].median(dim=0).values.double()
    log_best = _refine_log_lambda(-med[None, :], log_grid.cpu())[0]
    return GlobalLambda(
        lam=float(torch.exp(log_best)),
        n_signal=int(signal.sum()),
        median_curve=med.numpy(),
        log10_grid=grid10,
        fallback=fallback,
    )


def _residual_autocorr(
    data: torch.Tensor,
    design: torch.Tensor,
    n_task: int,
    betas: torch.Tensor,
    run_starts: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Within-run lag-1/lag-2 autocorrelation of the fit's residuals, and its R^2.

    Residual = the data minus the task fit, projected off the nuisance.  Lags
    never pair samples across a run boundary.
    """
    n_vox, n_t = data.shape
    design64 = design.detach().cpu().double().numpy()
    nuis = design64[:, n_task:]
    nuis = nuis[:, np.abs(nuis).sum(axis=0) > 0]
    q = np.zeros((n_t, 0))
    if nuis.shape[1]:
        u, sv, _ = np.linalg.svd(nuis, full_matrices=False)
        q = u[:, sv > sv.max() * 1e-10]
    q_t = torch.as_tensor(q, dtype=torch.float64, device=device)
    x_t = torch.as_tensor(design64[:, :n_task], dtype=torch.float64, device=device)
    same_run = torch.ones(n_t - 1, dtype=torch.bool)
    same_run2 = torch.ones(max(n_t - 2, 0), dtype=torch.bool)
    for start in list(run_starts)[1:]:
        same_run[start - 1] = False
        same_run2[max(start - 2, 0) : start] = False
    same_run, same_run2 = same_run.to(device), same_run2.to(device)
    rho1 = torch.zeros(n_vox, dtype=torch.float64)
    rho2 = torch.zeros(n_vox, dtype=torch.float64)
    r2 = torch.zeros(n_vox, dtype=torch.float32)
    chunk = estimate_chunk_size(n_vox, n_t, n_task, device, operation="glm")
    for a in range(0, n_vox, chunk):
        b = min(a + chunk, n_vox)
        y = data[a:b].to(device=device, dtype=torch.float64)
        res = y - betas[a:b].to(device, torch.float64) @ x_t.T
        if q_t.shape[1]:
            res = res - (res @ q_t) @ q_t.T
        var = (res * res).sum(dim=1).clamp_min(1e-30)
        rho1[a:b] = ((res[:, 1:] * res[:, :-1])[:, same_run].sum(dim=1) / var).cpu()
        rho2[a:b] = ((res[:, 2:] * res[:, :-2])[:, same_run2].sum(dim=1) / var).cpu()
        r2[a:b] = cod_from_ss_residual(y.float(), (res * res).sum(dim=1).float()).cpu()
    return rho1, rho2, r2


#: ARMA(1,1) grid the voxels are binned on for prewhitening.
ARMA_A_GRID = np.round(np.arange(0.0, 0.91, 0.1), 2)
ARMA_B_GRID = np.round(np.arange(-0.8, 0.81, 0.1), 2)


def arma_bins(rho1: torch.Tensor, rho2: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Nearest ARMA(1,1) grid pair per voxel from its lag-1/lag-2 autocorrelation.

    ARMA(1,1) correlations are ``[1, lam, lam*a, lam*a^2, ...]``, so the moments
    give ``lam = rho1`` and ``a = rho2 / rho1``.  Returns ``(grid, index)``:
    the ``(n_pairs, 2)`` grid and each voxel's row in it.
    """
    from fastfuncstuff.glm.arma import compute_arma_lambda

    pairs = [
        (a, b)
        for a in ARMA_A_GRID
        for b in ARMA_B_GRID
        if compute_arma_lambda(float(a), float(b)) >= 0
    ]
    grid = np.array(pairs)
    lam_grid = np.array([compute_arma_lambda(float(a), float(b)) for a, b in pairs])
    lam = rho1.clamp(0.0, 0.95).numpy()
    a_hat = np.where(lam > 1e-3, rho2.numpy() / np.maximum(lam, 1e-3), 0.0).clip(0.0, 0.9)
    cost = (grid[None, :, 0] - a_hat[:, None]) ** 2 + (lam_grid[None, :] - lam[:, None]) ** 2
    return grid, cost.argmin(axis=1)


def fit_smooth_arma(
    data: torch.Tensor,
    design: torch.Tensor,
    n_task: int,
    penalties: list[np.ndarray],
    run_starts: list[int],
    *,
    rule: str = "reml",
    lam: float | None = None,
    xval: bool = False,
    device: torch.device | None = None,
    verbose: bool = False,
) -> tuple[SmoothSelection, torch.Tensor]:
    """:func:`fit_smooth_selected` under ARMA(1,1) noise, prewhitened per voxel group.

    REML and GCV assume white noise; autocorrelated noise looks like signal to
    them.  Pass 1 fits white; its residuals give each voxel's within-run
    lag-1/lag-2 autocorrelation, which sets an ARMA(1,1) bin
    (:func:`arma_bins`).  Each bin's data and design are prewhitened with the
    run-block-diagonal ARMA covariance (``glm.arma.build_arma11_covariance``,
    ffs_reml's noise model) and refitted.  Whitening never mixes runs, so
    held-out runs stay held out; they are scored on the RAW series, so
    ``xval_r2`` compares directly with a white fit's.  Returns the selection
    and each voxel's ``(a, b)`` as ``(V, 2)``.
    """
    from fastfuncstuff.glm.arma import build_arma11_covariance

    device = device if device is not None else torch.device("cpu")
    n_vox, n_t = data.shape
    white = fit_smooth_selected(
        data, design, n_task, penalties, run_starts, rule=rule, lam=lam, device=device
    )
    rho1, rho2, _ = _residual_autocorr(data, design, n_task, white.fit.betas, run_starts, device)
    grid, which = arma_bins(rho1, rho2)
    design64 = design.detach().cpu().double()
    betas = torch.zeros((n_vox, n_task), dtype=torch.float32)
    out_lam = torch.ones(n_vox, dtype=torch.float32)
    out_edf = torch.zeros(n_vox, dtype=torch.float32)
    pen_idx = torch.zeros(n_vox, dtype=torch.long)
    xval_r2 = torch.zeros(n_vox, dtype=torch.float32) if xval else None
    for k in tqdm(
        np.unique(which), desc="  ARMA bins", unit="bin", leave=True, disable=not verbose
    ):
        idx = torch.as_tensor(np.nonzero(which == k)[0])
        a, b = float(grid[k, 0]), float(grid[k, 1])
        cov = build_arma11_covariance(
            a, b, n_t, torch.device("cpu"), dtype=torch.float64, run_starts=list(run_starts)
        )
        chol = (
            torch.eye(n_t, dtype=torch.float64)
            if cov is None or (a == 0.0 and b == 0.0)
            else torch.linalg.cholesky(cov)
        )
        design_w = torch.linalg.solve_triangular(chol, design64, upper=False)
        sub_w = torch.linalg.solve_triangular(chol, data[idx].double().T, upper=False).T
        sel = fit_smooth_selected(
            sub_w.float(),
            design_w,
            n_task,
            penalties,
            run_starts,
            rule=rule,
            lam=lam,
            xval=xval,
            device=device,
            score_data=data[idx],
            score_design=design64,
        )
        betas[idx] = sel.fit.betas
        out_lam[idx] = sel.fit.lam
        out_edf[idx] = sel.fit.edf
        pen_idx[idx] = sel.penalty_index
        if xval_r2 is not None and sel.xval_r2 is not None:
            xval_r2[idx] = sel.xval_r2
    _, _, r2 = _residual_autocorr(data, design, n_task, betas, run_starts, device)
    arma = torch.as_tensor(grid[which], dtype=torch.float32)
    fit = SmoothBasisFit(betas=betas, lam=out_lam, edf=out_edf, r2=r2, method=rule)
    biased = rule == "loro" or len(penalties) > 1
    return SmoothSelection(fit, pen_idx, xval_r2, biased), arma


#: How each run's lambda is set in :func:`fit_smooth_per_run`.
PER_RUN_LAMBDA = ("shared", "global", "run")
#: Where the held-out scoring's lambda comes from.
PER_RUN_XVAL_LAMBDA = ("fold", "all")


@dataclass
class PerRunFit:
    """Result of :func:`fit_smooth_per_run`."""

    betas: torch.Tensor  # (V, R, K) one smoothed curve per run
    log10_lambda: torch.Tensor  # (V, R) relative lambda used in each run's fit
    se: torch.Tensor | None  # (V, R, K) standard error about the SMOOTHED truth
    #: COD of "fit one run, predict every other run", pooled over all pairs.
    xval_r2: torch.Tensor | None
    #: The same score for the unpenalized per-run fit (OLS; GLS when whitened).
    xval_r2_ols: torch.Tensor | None
    #: The xval lambda actually came from all runs (asked for, or forced by a
    #: pooled rule that cannot be re-run without the scored run).
    xval_lambda_used: str


@dataclass
class _RunBlock:
    rows: np.ndarray
    chol: torch.Tensor | None  # whitening factor, None when white
    q_fit: torch.Tensor  # (T_r, p) nuisance basis in the fitted (whitened) space
    x_fit: torch.Tensor  # (T_r, K) task columns, whitened and nuisance-projected
    q_raw: torch.Tensor  # the same in the raw (scoring) space
    x_raw: torch.Tensor
    w: torch.Tensor  # (K, K) spectrum of this run alone
    s: torch.Tensor
    gram_trace: float
    gram_raw: torch.Tensor
    n_eff: int


def _nuisance_basis(nuis: np.ndarray) -> np.ndarray:
    nuis = nuis[:, np.abs(nuis).sum(axis=0) > 0]
    if not nuis.shape[1]:
        return np.zeros((nuis.shape[0], 0))
    u, sv, _ = np.linalg.svd(nuis, full_matrices=False)
    return u[:, sv > sv.max() * 1e-10]


def _run_blocks(design64, n_task, penalty, bounds, a, b, device) -> list[_RunBlock]:
    from fastfuncstuff.glm.arma import build_arma11_covariance

    blocks = []
    for r in range(len(bounds) - 1):
        rows = np.arange(bounds[r], bounds[r + 1])
        d_raw = design64[rows]
        chol = None
        d_fit = d_raw
        if a != 0.0 or b != 0.0:
            cov = build_arma11_covariance(a, b, rows.size, torch.device("cpu"), torch.float64)
            if cov is not None:
                chol = torch.linalg.cholesky(cov)
                d_fit = torch.linalg.solve_triangular(
                    chol, torch.as_tensor(d_raw), upper=False
                ).numpy()
        q_raw = _nuisance_basis(d_raw[:, n_task:])
        x_raw = d_raw[:, :n_task] - q_raw @ (q_raw.T @ d_raw[:, :n_task])
        q_fit = _nuisance_basis(d_fit[:, n_task:])
        x_fit = d_fit[:, :n_task] - q_fit @ (q_fit.T @ d_fit[:, :n_task])
        gram = x_fit.T @ x_fit
        w, s = _gram_spectrum(gram, penalty)

        def dev(v):
            return torch.as_tensor(v, dtype=torch.float64, device=device)

        blocks.append(
            _RunBlock(
                rows=rows,
                chol=None if chol is None else chol.to(device),
                q_fit=dev(q_fit),
                x_fit=dev(x_fit),
                q_raw=dev(q_raw),
                x_raw=dev(x_raw),
                w=dev(w),
                s=dev(s),
                gram_trace=float(np.trace(gram)),
                gram_raw=dev(x_raw.T @ x_raw),
                n_eff=rows.size - q_fit.shape[1],
            )
        )
    return blocks


def _shrink(lam_rel: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """``d = 1/((1-s) + lam s)`` per voxel: (V,) lambdas -> (V, K)."""
    return 1.0 / ((1.0 - s)[None, :] + lam_rel[:, None] * s[None, :])


def _target_ss(beta, yy, c, gram):
    """``||y - X beta||^2`` from ``y'y``, ``X'y`` and ``X'X`` (summed over targets)."""
    return yy - 2 * (beta * c).sum(dim=1) + ((beta @ gram) * beta).sum(dim=1)


def _choose_lambda(rule, z, yy, s, n_eff, rank_p, log_grid, lam_fixed):
    if rule == "fixed":
        return torch.full_like(yy, float(lam_fixed))
    crit = _criterion(rule, z * z, yy, s, torch.exp(log_grid), n_eff, rank_p)
    return torch.exp(_refine_log_lambda(crit, log_grid))


def fit_smooth_per_run(
    data: torch.Tensor,
    design: torch.Tensor,
    n_task: int,
    penalties: list[np.ndarray],
    run_starts: list[int],
    pooled_lam: torch.Tensor,
    *,
    signal_r2: float = GLOBAL_SIGNAL_R2,
    penalty_index: torch.Tensor | None = None,
    arma: torch.Tensor | None = None,
    rule: str = "reml",
    lam: float | None = None,
    lambda_mode: str = "shared",
    xval: bool = False,
    xval_lambda: str = "fold",
    se: bool = False,
    log10_grid: np.ndarray | None = None,
    device: torch.device | None = None,
    verbose: bool = False,
) -> PerRunFit:
    """One smoothed FIR/TENT curve per run, for statistics ACROSS runs.

    ``lambda_mode="shared"`` (default) fits every run with the pooled fit's
    lambda (``pooled_lam``, relative to the pooled design) at the same ABSOLUTE
    strength: ``lam_run = lam_pool * tr(G_pool) / tr(G_run)``, so a noisier
    run is not shrunk harder.  The voxel's pooled lambda was itself chosen from
    its data, which leaves a small bias (a noisy run that looks peaky lowers
    it).  ``"global"`` removes that: one lambda per penalty, chosen by
    :func:`choose_global_lambda` (best median held-out R^2 over the voxels
    whose OLS or ``rule`` R^2 beats ``signal_r2``), makes each run's fit
    exactly linear.  Under ARMA noise it is chosen on the unwhitened series.  Runs with
    different event timing still smooth the same curve slightly differently.
    ``"run"`` picks each
    run's own lambda by REML/GCV -- noisier runs are then shrunk harder, which
    fakes run effects (a sham habituation) when noise differs across runs.

    ``penalty_index``/``arma`` carry the pooled fit's per-voxel penalty choice
    and ARMA(1,1) noise; each run is prewhitened with its voxel's ``(a, b)``.

    ``xval`` scores "fit run r, predict run j" over every pair, on the raw
    series, for the smoothed and the unpenalized per-run fit.  ``xval_lambda
    ="fold"`` re-chooses the pooled lambda by ``rule`` without run j (one pooled
    REML per run, from per-run sufficient statistics -- no extra data pass);
    ``"all"`` keeps the all-run lambda (a small leak, R times cheaper).  A
    ``loro`` pooled rule cannot be re-run inside the fold, so it scores with
    ``"all"``, and so does ``"global"`` (its one lambda comes from every
    voxel and run).  A penalty chosen per voxel by held-out runs stays chosen with
    every run.  ``se`` gives the standard error of each knot about the smoothed
    truth (smoothing bias not included).
    """
    if lambda_mode not in PER_RUN_LAMBDA:
        raise ValueError(f"lambda_mode must be one of {PER_RUN_LAMBDA}, got {lambda_mode!r}")
    if xval_lambda not in PER_RUN_XVAL_LAMBDA:
        raise ValueError(f"xval_lambda must be one of {PER_RUN_XVAL_LAMBDA}, got {xval_lambda!r}")
    if len(run_starts) < 2:
        raise ValueError("per-run curves need at least two runs")
    device = device if device is not None else torch.device("cpu")
    n_vox, n_t = data.shape
    n_runs = len(run_starts)
    bounds = list(run_starts) + [n_t]
    design64 = design.detach().cpu().double().numpy()
    log_grid = torch.as_tensor(
        DEFAULT_LOG10_GRID if log10_grid is None else log10_grid, dtype=torch.float64
    ).to(device) * np.log(10.0)
    run_rule = rule if rule in ("reml", "gcv") else "reml"
    fold_rule = rule if rule in ("reml", "gcv", "fixed") else None
    xval_used = xval_lambda if lambda_mode == "shared" and fold_rule is not None else "all"
    if lambda_mode == "run":
        xval_used = "fold"  # each run's lambda never sees another run

    betas = torch.zeros((n_vox, n_runs, n_task), dtype=torch.float32)
    log10_lam = torch.zeros((n_vox, n_runs), dtype=torch.float32)
    se_out = torch.zeros((n_vox, n_runs, n_task), dtype=torch.float32) if se else None
    xr2 = torch.zeros(n_vox, dtype=torch.float32) if xval else None
    xr2_ols = torch.zeros(n_vox, dtype=torch.float32) if xval else None

    pen_idx = torch.zeros(n_vox, dtype=torch.long) if penalty_index is None else penalty_index
    if arma is None:
        arma_np = np.zeros((n_vox, 2))
    else:
        # Back to the grid values the pooled fit whitened with (float32 on the way out).
        arma_np = np.round(arma.detach().cpu().double().numpy(), 2)
    keys = np.column_stack([pen_idx.numpy(), arma_np])
    uniq, which = np.unique(keys, axis=0, return_inverse=True)
    which = which.reshape(-1)
    pooled_lam64 = pooled_lam.detach().cpu().double()
    if lambda_mode == "global":
        pooled_lam64 = pooled_lam64.clone()
        for p_i, penalty in enumerate(penalties):
            sel = pen_idx == p_i
            if bool(sel.any()):
                pooled_lam64[sel] = choose_global_lambda(
                    data[sel],
                    design,
                    n_task,
                    penalty,
                    run_starts,
                    signal_r2=signal_r2,
                    rule=run_rule,
                    log10_grid=log10_grid,
                    device=device,
                    verbose=verbose,
                ).lam

    chunk = estimate_chunk_size(n_vox, n_t, n_task * (4 * n_runs + 2), device, operation="glm")
    groups = tqdm(
        range(len(uniq)),
        desc="  Per-run fits",
        unit="group",
        leave=True,
        disable=not verbose or len(uniq) < 2,
    )
    for gi in groups:
        p_i, a, b = int(uniq[gi, 0]), float(uniq[gi, 1]), float(uniq[gi, 2])
        penalty = penalties[p_i]
        rank_p = _penalty_rank(penalty)
        blocks = _run_blocks(design64, n_task, penalty, bounds, a, b, device)
        traces = np.array([blk.gram_trace for blk in blocks])
        tr_pool = float(traces.sum())
        n_eff_all = sum(blk.n_eff for blk in blocks)
        fold_spec = []
        if xval and xval_used == "fold" and lambda_mode == "shared":
            g_all = sum(blk.x_fit.T @ blk.x_fit for blk in blocks)
            for blk in blocks:
                g_minus = (g_all - blk.x_fit.T @ blk.x_fit).cpu().numpy()
                w_m, s_m = _gram_spectrum(g_minus, penalty)
                fold_spec.append(
                    (
                        torch.as_tensor(w_m, device=device),
                        torch.as_tensor(s_m, device=device),
                        float(np.trace(g_minus)),
                        n_eff_all - blk.n_eff,
                    )
                )
        idx_all = torch.as_tensor(np.nonzero(which == gi)[0])
        for c0 in range(0, idx_all.numel(), chunk):
            idx = idx_all[c0 : c0 + chunk]
            y = data[idx].to(device=device, dtype=torch.float64)
            nv = y.shape[0]
            c_fit, yy_fit, c_raw, yy_raw, sum_raw = [], [], [], [], []
            for blk in blocks:
                y_r = y[:, blk.rows[0] : blk.rows[-1] + 1]
                y_s = y_r - (y_r @ blk.q_raw) @ blk.q_raw.T if blk.q_raw.shape[1] else y_r
                c_raw.append(y_s @ blk.x_raw)
                yy_raw.append((y_s * y_s).sum(dim=1))
                sum_raw.append(y_s.sum(dim=1))
                if blk.chol is not None:
                    y_r = torch.linalg.solve_triangular(blk.chol, y_r.T, upper=False).T
                y_f = y_r - (y_r @ blk.q_fit) @ blk.q_fit.T if blk.q_fit.shape[1] else y_r
                c_fit.append(y_f @ blk.x_fit)
                yy_fit.append((y_f * y_f).sum(dim=1))
            empty = torch.stack(yy_raw).sum(dim=0) <= 1e-12 * n_t

            lam_pool = pooled_lam64[idx].to(device)
            beta_runs = []
            for r, blk in enumerate(blocks):
                z = c_fit[r] @ blk.w
                if lambda_mode != "run":
                    lam_r = lam_pool * tr_pool / blk.gram_trace
                else:
                    lam_r = _choose_lambda(
                        run_rule, z, yy_fit[r], blk.s, blk.n_eff, rank_p, log_grid, None
                    )
                d = _shrink(lam_r, blk.s)
                beta_r = (d * z) @ blk.w.T
                beta_runs.append(beta_r)
                betas[idx, r] = beta_r.float().cpu()
                log10_lam[idx, r] = torch.log10(lam_r).float().cpu()
                if se_out is not None:
                    one_s = (1.0 - blk.s)[None, :]
                    rss = (yy_fit[r] - (z * z * (2 * d - d * d * one_s)).sum(dim=1)).clamp_min(0)
                    edf = (one_s * d).sum(dim=1)
                    sigma2 = rss / (blk.n_eff - edf).clamp_min(1.0)
                    var = (d * d * one_s) @ (blk.w * blk.w).T
                    se_out[idx, r] = torch.sqrt(var * sigma2[:, None]).float().cpu()

            if xval:
                c_sum = torch.stack(c_raw).sum(dim=0)
                g_sum = sum(blk.gram_raw for blk in blocks)
                yy_sum = torch.stack(yy_raw).sum(dim=0)
                n_all = sum(blk.rows.size for blk in blocks)
                ss_tot = (n_runs - 1) * (yy_sum - torch.stack(sum_raw).sum(dim=0) ** 2 / n_all)

                ss_ols = torch.zeros(nv, dtype=torch.float64, device=device)
                for r, blk in enumerate(blocks):
                    g_fit = blk.x_fit.T @ blk.x_fit
                    beta_ols = c_fit[r] @ torch.linalg.pinv(g_fit, hermitian=True)
                    ss_ols += _target_ss(
                        beta_ols,
                        yy_sum - yy_raw[r],
                        c_sum - c_raw[r],
                        g_sum - blk.gram_raw,
                    )
                ss = torch.zeros(nv, dtype=torch.float64, device=device)
                if xval_used == "all" or lambda_mode == "run":
                    for r in range(n_runs):
                        ss += _target_ss(
                            beta_runs[r],
                            yy_sum - yy_raw[r],
                            c_sum - c_raw[r],
                            g_sum - blocks[r].gram_raw,
                        )
                else:
                    c_fit_sum = torch.stack(c_fit).sum(dim=0)
                    yy_fit_sum = torch.stack(yy_fit).sum(dim=0)
                    for j in range(n_runs):
                        w_m, s_m, tr_m, n_eff_m = fold_spec[j]
                        z_m = (c_fit_sum - c_fit[j]) @ w_m
                        lam_m = _choose_lambda(
                            fold_rule,
                            z_m,
                            yy_fit_sum - yy_fit[j],
                            s_m,
                            n_eff_m,
                            rank_p,
                            log_grid,
                            lam,
                        )
                        for r, blk in enumerate(blocks):
                            if r == j:
                                continue
                            d = _shrink(lam_m * tr_m / blk.gram_trace, blk.s)
                            beta = (d * (c_fit[r] @ blk.w)) @ blk.w.T
                            ss += _target_ss(beta, yy_raw[j], c_raw[j], blocks[j].gram_raw)
                live = ~empty & (ss_tot > 0)
                safe = torch.where(live, ss_tot, torch.ones_like(ss_tot))
                zero = torch.zeros_like(ss)
                assert xr2 is not None and xr2_ols is not None
                xr2[idx] = torch.where(live, 1 - ss / safe, zero).float().cpu()
                xr2_ols[idx] = torch.where(live, 1 - ss_ols / safe, zero).float().cpu()
            betas[idx[empty.cpu()]] = 0.0
            log10_lam[idx[empty.cpu()]] = 0.0
    return PerRunFit(
        betas=betas,
        log10_lambda=log10_lam,
        se=se_out,
        xval_r2=xr2,
        xval_r2_ols=xr2_ols,
        xval_lambda_used=xval_used if xval else xval_lambda,
    )
