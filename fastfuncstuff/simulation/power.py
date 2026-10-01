"""Monte-Carlo power for a single-subject design: "at this tSNR, is it hopeless or fine?"

Every simulated voxel is one replicate. For each noise condition (a tSNR bin
from :mod:`calibrate`, or a hand-set tSNR) and each response amplitude, the
engine plants the response, adds independent noise, fits the design by OLS and
reads out, per contrast: bias, spread, t, and the fraction of replicates past
threshold. Amplitude 0 is always included, so the false-positive rate is
measured, not assumed.

The fit is OLS -- the default the design question needs -- but its standard
errors can be corrected for the noise's known ARMA(1,1) without fitting REML
per voxel. With R the noise correlation matrix and P = (X'X)^-1 X':

    Var(c beta_hat) = sigma^2 c P R P' c'           (sandwich, exact under R)
    E[RSS]          = sigma^2 tr(M R),   M = I - X P
    dof             = tr(M R)^2 / tr((M R)^2)       (Satterthwaite)

so sigma^2 is estimated as RSS / tr(M R), and t is referred to that dof. Naive
OLS t (sigma^2 = RSS / (n - p), Var = sigma^2 c (X'X)^-1 c') is reported
alongside, since how far its false-positive rate drifts from alpha is itself an
answer to "does autocorrelation matter for this design".
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats

from .noise import ou_to_arma11


def _two_tailed_power(crit: float, dof: float, nc: float | np.ndarray) -> Any:
    """P(|T| > crit) for noncentral t, by symmetry on |nc| (scalar or array ``nc``).

    scipy's nct.cdf returns nan far in the tail (nc = 16 at 150 dof), where the
    wrong-sign tail it would contribute is < 1e-50 anyway.
    """
    nc = np.abs(np.asarray(nc, dtype=float))
    # Reflection evaluates both tails in one distribution call, without 1-CDF
    # cancellation in the detection tail.
    near, far = stats.nct.sf(crit, dof, np.stack((nc, -nc)))
    out = near + np.where(np.isnan(far), 0.0, far)
    return float(out) if np.ndim(out) == 0 else out


def _nuisance(run_lengths: list[int], poly_degree: int) -> torch.Tensor:
    from fastfuncstuff.glm.core import construct_polynomial_matrix

    cpu = torch.device("cpu")
    blocks = [construct_polynomial_matrix(n, poly_degree, cpu, torch.float64) for n in run_lengths]
    return torch.block_diag(*blocks)


def _noise_arma(noise: dict[str, Any], tr: float) -> tuple[float, float] | None:
    """AFNI-form ARMA(1,1) (a, b) of a noise condition, or None if white."""
    if noise.get("arma") is not None:
        a, b = (float(v) for v in noise["arma"])
    else:
        f = float(noise.get("phys_fraction", 0.5))
        if f == 0.0:
            return None
        a, b = _ou_arma_parameters(tr, float(noise.get("tau", 6.0)), f)
    return None if a == 0.0 and b == 0.0 else (a, b)


@lru_cache(maxsize=128)
def _ou_arma_parameters(tr: float, tau: float, fraction: float) -> tuple[float, float]:
    a, b = ou_to_arma11(tr, tau, fraction)
    return float(a), float(b)


@lru_cache(maxsize=64)
def _run_correlation(n: int, a: float, b: float) -> tuple[torch.Tensor, torch.Tensor]:
    """One run's ARMA(1,1) correlation and its Cholesky factor (CPU float64).

    Cached: every realization of a fixed-scan design has the same run lengths,
    and every -tsnr level the same (a, b).
    """
    from fastfuncstuff.glm.arma import build_arma11_covariance

    R = build_arma11_covariance(a, b, n, torch.device("cpu"), torch.float64)
    if R is None:
        raise ValueError(f"noise ARMA a={a:.3f}, b={b:.3f} is not a valid correlation")
    return R, torch.linalg.cholesky(R)


def _noise_correlation(
    noise: dict[str, Any], tr: float, run_lengths: list[int], factor: bool = False
) -> torch.Tensor | None:
    """Block-diagonal ARMA(1,1) correlation of a noise condition, or None if white.

    ``factor=True`` returns its lower Cholesky factor instead: ``L @ z`` with
    z standard normal is noise of exactly that correlation, runs independent.
    """
    ab = _noise_arma(noise, tr)
    if ab is None:
        return None
    return _block_correlation(tuple(int(n) for n in run_lengths), *ab, factor)


@lru_cache(maxsize=16)
def _block_correlation(
    run_lengths: tuple[int, ...], a: float, b: float, factor: bool
) -> torch.Tensor:
    return torch.block_diag(*(_run_correlation(n, a, b)[int(factor)] for n in run_lengths))


def _corrected_terms(
    X: torch.Tensor, P: torch.Tensor, R: torch.Tensor
) -> tuple[torch.Tensor, float, float]:
    """P R P', tr(MR) and the Satterthwaite dof tr(MR)^2 / tr((MR)^2), M = I - X P.

    Without forming M: expanding M = I - X P turns every trace into products
    of R with the p design columns, n_t^2 p instead of the n_t^3 of M @ R --
    at TR 0.5 over two 10-minute runs that dense product was most of the run.
    """
    RX, RPt = R @ X, R @ P.T  # (n_t, p)
    PRX = P @ RX  # (p, p)
    tr_MR = float(torch.diagonal(R).sum() - torch.trace(PRX))
    tr_MRMR = float((R * R).sum() - 2.0 * (RPt * RX).sum() + torch.trace(PRX @ PRX))
    return P @ RPt, tr_MR, tr_MR**2 / tr_MRMR


def _noise_terms(
    ab: tuple[float, float] | None,
    X: torch.Tensor,
    P: torch.Tensor,
    C: torch.Tensor,
    run_lengths: list[int],
) -> tuple[torch.Tensor, float, float]:
    """Contrast variances (unit noise), tr(MR) and dof under ARMA ``ab``; white if None."""
    n_t, n_p = X.shape
    if ab is None:
        return torch.einsum("ip,pq,iq->i", C, P @ P.T, C), float(n_t - n_p), float(n_t - n_p)
    R = _block_correlation(tuple(int(n) for n in run_lengths), *ab, False)
    PRPt, tr_MR, dof = _corrected_terms(X, P, R)
    return torch.einsum("ip,pq,iq->i", C, PRPt, C), tr_MR, dof


def _predicted(
    a: torch.Tensor, sigma: float, scale: float, v_true: torch.Tensor, tr_MR: float,
    crit: float, dof: float, glm: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    """Analytic mean estimate, misfit energy and power at amplitudes ``a`` (A, 1).

    Returns (expected estimate (A, K) PSC, misfit sum of squares (A,), power
    (A, K)). The misfit -- what the fitted model cannot absorb of the true
    response -- inflates E[RSS] and shrinks every t; an unbiased-noise formula
    missed it (it promised 80% where Monte Carlo gave ~0 for A-B on a 4%
    shared response).
    """
    exp_est = glm["exp_base"][None] + a * glm["exp_unit"][None]  # (A, K), PSC
    mis2 = scale**2 * ((glm["mis_base"][None] + a * glm["mis_unit"][None]) ** 2).sum(dim=1)
    sigma2_hat = sigma**2 + mis2 / tr_MR  # E[RSS] / tr(MR), (A,)
    se_hat = torch.sqrt(v_true)[None] * torch.sqrt(sigma2_hat)[:, None] / scale
    nc = torch.where(se_hat > 0, exp_est / se_hat, torch.zeros_like(se_hat))
    power = torch.as_tensor(_two_tailed_power(crit, dof, nc.numpy()), dtype=torch.float64)
    return exp_est, mis2, power.reshape(exp_est.shape)


def _glm_setup(
    design: torch.Tensor | np.ndarray,
    run_lengths: list[int],
    tr: float,
    contrasts: dict[str, list[float] | np.ndarray],
    poly_degree: int | None,
    true_design: torch.Tensor | np.ndarray | None,
    beta_pattern: list[float] | np.ndarray | None,
    beta_offset: list[float] | np.ndarray | float,
) -> dict[str, Any]:
    """The fitted GLM and the planted signal's deterministic parts, shared by both engines."""
    from fastfuncstuff.cli_utils import auto_polort

    X_task = torch.as_tensor(design, dtype=torch.float64)
    n_t, n_cond = X_task.shape
    if sum(run_lengths) != n_t:
        raise ValueError(f"run lengths sum to {sum(run_lengths)}, design has {n_t} rows")
    X_true = X_task if true_design is None else torch.as_tensor(true_design, dtype=torch.float64)
    if X_true.shape != X_task.shape:
        raise ValueError(f"true_design {tuple(X_true.shape)} must match design {(n_t, n_cond)}")
    pattern = torch.ones(n_cond, dtype=torch.float64)
    if beta_pattern is not None:
        pattern = torch.as_tensor(beta_pattern, dtype=torch.float64)
    offset = torch.as_tensor(beta_offset, dtype=torch.float64).expand(n_cond).clone()
    if poly_degree is None:
        poly_degree = auto_polort(max(run_lengths) * tr)

    X = torch.cat([X_task, _nuisance(run_lengths, poly_degree)], dim=1)
    n_p = X.shape[1]
    if torch.linalg.matrix_rank(X) < n_p:
        raise ValueError("design + drift polynomials are rank-deficient; check the conditions")
    XtX_inv = torch.linalg.inv(X.T @ X)
    P = XtX_inv @ X.T  # (n_p, n_t)

    names = list(contrasts)
    C = torch.zeros(len(names), n_p, dtype=torch.float64)
    for i, name in enumerate(names):
        w = torch.as_tensor(np.asarray(contrasts[name], dtype=np.float64))
        if w.numel() != n_cond:
            raise ValueError(f"contrast {name!r} has {w.numel()} weights, design {n_cond}")
        C[i, :n_cond] = w
    # Expected contrast estimate c P X_true beta and the misfit (I - X P) X_true
    # beta: both affine in the amplitude, split into base and unit parts.
    s_unit, s_base = X_true @ pattern, X_true @ offset
    return {
        "X": X, "X_true": X_true, "P": P, "XtX_inv": XtX_inv, "C": C, "names": names,
        "pattern": pattern, "offset": offset, "poly_degree": poly_degree, "n_cond": n_cond,
        "exp_unit": C @ P @ s_unit, "exp_base": C @ P @ s_base,
        "mis_unit": s_unit - X @ (P @ s_unit), "mis_base": s_base - X @ (P @ s_base),
    }  # fmt: skip


def analytic_power(
    design: torch.Tensor | np.ndarray,
    run_lengths: list[int],
    tr: float,
    contrasts: dict[str, list[float] | np.ndarray],
    amplitudes: np.ndarray,
    noise: list[dict[str, Any]],
    beta_pattern: list[float] | np.ndarray | None = None,
    beta_offset: list[float] | np.ndarray | float = 0.0,
    poly_degree: int | None = None,
    alpha: float = 0.001,
    true_design: torch.Tensor | np.ndarray | None = None,
    baseline: float = 100.0,
) -> dict[str, np.ndarray]:
    """The analytic power curve of :func:`simulate_design_power`, without the Monte Carlo.

    {noise label: (n_amplitudes, n_contrasts) power}. Cheap enough to evaluate
    on a fine grid -- which is how ``AutoSweep`` places the simulated points.
    """
    glm = _glm_setup(
        design, run_lengths, tr, contrasts, poly_degree, true_design, beta_pattern, beta_offset
    )
    a = torch.as_tensor(np.asarray(amplitudes, dtype=float), dtype=torch.float64)[:, None]
    scale = baseline / 100.0
    out: dict[str, np.ndarray] = {}
    terms: dict[tuple[float, float] | None, tuple] = {}
    for k, cond in enumerate(noise):
        label = str(cond.get("label", f"noise{k}"))
        ab = _noise_arma(cond, tr)
        if ab not in terms:
            terms[ab] = _noise_terms(ab, glm["X"], glm["P"], glm["C"], run_lengths)
        v_true, tr_MR, dof = terms[ab]
        crit = float(stats.t.ppf(1 - alpha / 2, dof))
        sigma = baseline / float(cond["tsnr"])
        out[label] = _predicted(a, sigma, scale, v_true, tr_MR, crit, dof, glm)[2].numpy()
    return out


def simulate_design_power(
    design: torch.Tensor | np.ndarray,
    run_lengths: list[int],
    tr: float,
    contrasts: dict[str, list[float] | np.ndarray],
    amplitudes: list[float] | np.ndarray | dict[str, list[float]],
    noise: list[dict[str, Any]],
    beta_pattern: list[float] | np.ndarray | None = None,
    n_reps: int = 500,
    poly_degree: int | None = None,
    beta_offset: list[float] | np.ndarray | float = 0.0,
    alpha: float = 0.001,
    true_design: torch.Tensor | np.ndarray | None = None,
    baseline: float = 100.0,
    device: torch.device | None = None,
    seed: int = 0,
    keep_t: bool = False,
    estimator: str = "ols",
    null_reps: int | None = None,
    reml_maxa: float = 0.8,
    reml_maxb: float = 0.8,
    reml_cache: dict | None = None,
) -> dict[str, Any]:
    """Monte-Carlo detection power of each contrast, per noise condition and amplitude.

    ``keep_t`` also returns the replicates' t values under 't': {(noise label,
    amplitude, contrast): (corrected t, naive t)}, and the critical values
    under 'crit': {noise label: (corrected, naive)} -- for the t-distribution
    figure.

    Parameters
    ----------
    design : (n_timepoints, n_conditions) task regressors, unit-peak (e.g. from
        ``simulate_bold(...)['design']`` or build_event_design_microtime), all runs
        concatenated. This is the model that is *fitted*.
    run_lengths : timepoints per run; each run gets its own Legendre drift block
    tr : seconds
    contrasts : name -> weights over the conditions
    amplitudes : peak percent signal change swept, or {noise label: list} for a
        grid per noise level; 0 is always added (the null)
    noise : noise conditions, each a dict of generate_thermal_physio_noise
        keywords (``tsnr`` plus ``phys_fraction``/``tau`` or ``arma``), optionally
        with a ``label``. ``TsnrBin.simulation_kwargs()`` returns exactly this.
    beta_offset : (n_conditions,) or scalar, a fixed response (PSC) under every
        amplitude -- the shared level a difference is swept on top of
    beta_pattern : (n_conditions,) response of each condition per unit amplitude;
        default all ones. A contrast whose weights sum to zero (A - B) then has
        no true effect -- pass e.g. [1, 0] to make A respond and B not.
    n_reps : replicates (simulated voxels) per cell
    poly_degree : per-run Legendre degree; default AFNI's 1 + floor(run_s / 150)
        on the longest run
    alpha : two-tailed threshold (default p < 0.001)
    true_design : (n_timepoints, n_conditions) regressors the signal is *generated*
        with, if different from ``design`` -- e.g. a delayed HRF, for mismatch
    baseline : signal baseline; tSNR is baseline / noise std

    Returns
    -------
    dict with
        'table' : list of row dicts, one per noise x amplitude x contrast:
            noise, tsnr, amplitude, contrast, true_effect (PSC), mean_est,
            expected_est (analytic mean, biased under mismatch), sd_est, sd_predicted, mean_t, power, power_predicted, mean_t_naive,
            power_naive
        'alpha', 'poly_degree', 'dof' ({noise label: (naive, corrected)}),
        'n_reps'
    """
    from fastfuncstuff.utils import get_device

    if estimator not in {"ols", "reml"}:
        raise ValueError("estimator must be 'ols' or 'reml'")
    if device is None:
        device = get_device()
    glm = _glm_setup(
        design, run_lengths, tr, contrasts, poly_degree, true_design, beta_pattern, beta_offset
    )
    X, X_true, P, C, names = glm["X"], glm["X_true"], glm["P"], glm["C"], glm["names"]
    XtX_inv, pattern, offset = glm["XtX_inv"], glm["pattern"], glm["offset"]
    poly_degree, n_cond = glm["poly_degree"], glm["n_cond"]
    n_t, n_p = X.shape
    expected_unit, expected_base = glm["exp_unit"], glm["exp_base"]
    misfit_unit, misfit_base = glm["mis_unit"], glm["mis_base"]
    per_label = isinstance(amplitudes, dict)
    P_dev = P.to(device=device, dtype=torch.float32)
    X_dev = X.to(device=device, dtype=torch.float32)
    CP_dev = (C @ P).to(device=device, dtype=torch.float32)  # contrast estimates directly
    scale = baseline / 100.0
    gen = torch.Generator(device=device).manual_seed(seed)
    # Signal parts, in data units: estimate and residual are affine in the amplitude.
    est_base, est_unit = scale * expected_base, scale * expected_unit
    mis_dev = (scale * torch.stack([misfit_base, misfit_unit], dim=1)).to(device, torch.float32)

    v_naive = torch.einsum("ip,pq,iq->i", C, XtX_inv, C)
    dev_mats = (P_dev, X_dev, CP_dev, mis_dev)
    shared: dict[tuple[float, float] | None, tuple] = {}
    t_kept: dict[tuple[str, float, str], tuple[np.ndarray, np.ndarray]] = {}
    crits: dict[str, tuple[float, float]] = {}
    rows: list[dict[str, Any]] = []
    dofs: dict[str, tuple[float, float]] = {}
    for k, cond in enumerate(noise):
        label = str(cond.get("label", f"noise{k}"))
        kw = {key: v for key, v in cond.items() if key != "label"}
        sigma = baseline / float(kw["tsnr"])
        sweep = amplitudes[label] if per_label else amplitudes  # type: ignore[index]
        amps = sorted({0.0, *(float(x) for x in sweep)})

        # Everything below depends on the noise only through its ARMA (a, b)
        # and scales with sigma, so noise levels that share (a, b) -- every
        # -tsnr level does -- share one unit-variance draw and one set of
        # corrected-t terms (common random numbers across levels too).
        ab = _noise_arma(kw, tr)
        if ab not in shared:
            shared[ab] = _unit_noise_fit(ab, X, P, C, run_lengths, n_reps, gen, dev_mats)
        v_true, tr_MR, dof_corr, est_u, rss_u, cross_u = shared[ab]
        est_n, rss_n, cross = sigma * est_u, sigma**2 * rss_u, sigma * cross_u
        dof_naive = float(n_t - n_p)
        dofs[label] = (dof_naive, dof_corr)
        crit_naive = stats.t.ppf(1 - alpha / 2, dof_naive)
        crit_corr = stats.t.ppf(1 - alpha / 2, dof_corr)

        # Every amplitude and contrast at once: (A, K, reps). A per-row loop
        # of scalar torch reductions and scalar scipy calls was ~80% of the
        # run once the draw itself was cheap.
        a = torch.tensor(amps, dtype=torch.float64)[:, None]  # (A, 1)
        est = est_n[None] + (est_base + a * est_unit)[:, :, None]  # (A, K, reps)
        exp_est, mis2, power_pred = _predicted(
            a, sigma, scale, v_true, tr_MR, crit_corr, dof_corr, glm
        )
        rss = rss_n[None] + 2.0 * (cross[0][None] + a * cross[1][None]) + mis2[:, None]
        t_naive = est / torch.sqrt(rss[:, None] / dof_naive * v_naive[None, :, None])
        t_corr = est / torch.sqrt(rss[:, None] / tr_MR * v_true[None, :, None])
        true_eff = (offset[None] + a * pattern[None]) @ C[:, :n_cond].T  # (A, K), PSC
        # exp_est is what the fit returns on average, c P X_true beta: the true
        # effect only when the fitted and generating regressors agree.
        sd_pred = sigma * torch.sqrt(v_true) / scale  # (K,), PSC
        cols = {
            "true_effect": true_eff,
            "mean_est": est.mean(dim=-1) / scale,
            "expected_est": exp_est,
            "sd_est": est.std(dim=-1) / scale,
            "sd_predicted": sd_pred.expand_as(exp_est),
            "mean_t": t_corr.mean(dim=-1),
            "power": (t_corr.abs() > crit_corr).double().mean(dim=-1),
            "power_predicted": power_pred,
            "mean_t_naive": t_naive.mean(dim=-1),
            "power_naive": (t_naive.abs() > crit_naive).double().mean(dim=-1),
        }
        vals = {key: v.tolist() for key, v in cols.items()}
        for j, amp in enumerate(amps):
            for i, name in enumerate(names):
                row = {"noise": label, "tsnr": float(kw["tsnr"]), "amplitude": amp}
                row["contrast"] = name
                row.update({key: v[j][i] for key, v in vals.items()})
                rows.append(row)
        if keep_t:
            for j, amp in enumerate(amps):
                for i, name in enumerate(names):
                    t_kept[(label, amp, name)] = (t_corr[j, i].numpy(), t_naive[j, i].numpy())
            crits[label] = (float(crit_corr), float(crit_naive))

    out = {
        "table": rows,
        "alpha": alpha,
        "poly_degree": poly_degree,
        "dof": dofs,
        "n_reps": n_reps,
    }
    if keep_t:
        out["t"], out["crit"] = t_kept, crits
    if estimator == "reml":
        from .reml_power import validate_reml_power

        validate_reml_power(
            out,
            X,
            X_true,
            C,
            pattern,
            offset,
            run_lengths,
            tr,
            noise,
            device,
            seed,
            null_reps,
            reml_maxa,
            reml_maxb,
            reml_cache,
        )
    return out


def _unit_noise_fit(
    ab: tuple[float, float] | None,
    X: torch.Tensor,
    P: torch.Tensor,
    C: torch.Tensor,
    run_lengths: list[int],
    n_reps: int,
    gen: torch.Generator,
    dev_mats: tuple[torch.Tensor, ...],
) -> tuple:
    """Corrected-t terms and the OLS fit of one unit-variance noise draw of ARMA ``ab``.

    Returns (v_true, tr(MR), dof, est, rss, cross): contrast variances under
    R, the Satterthwaite terms, and per replicate the contrast estimates, the
    RSS and the residual's cross terms with the (base, unit) misfit. Scale
    est and cross by sigma and rss by sigma^2 for a noise SD of sigma.

    The draw is chol(R) @ z -- exactly white + OU, runs independent -- run by
    run, since chol(R) is block-diagonal: one dense product over all runs
    cost n_runs times more. One draw serves every amplitude (common random
    numbers): OLS is linear, so each amplitude adds a fixed signal part. A
    fresh per-amplitude draw through a per-timepoint AR(1) loop on the CPU was
    93% of the run time.
    """
    P_dev, X_dev, CP_dev, mis_dev = dev_mats
    device = X_dev.device
    n_t = X.shape[0]
    Z = torch.randn(n_t, n_reps, device=device, generator=gen, dtype=torch.float32)
    v_true, tr_MR, dof = _noise_terms(ab, X, P, C, run_lengths)
    if ab is None:
        y = Z
    else:
        y = torch.empty_like(Z)
        start = 0
        for n in run_lengths:
            L = _run_correlation(int(n), *ab)[1].to(device, torch.float32)
            y[start : start + n] = L @ Z[start : start + n]
            start += n
    est = (CP_dev @ y).double().cpu()  # (n_contrasts, n_reps)
    resid = y - X_dev @ (P_dev @ y)
    rss = (resid * resid).sum(dim=0).double().cpu()
    cross = (mis_dev.T @ resid).double().cpu()  # (2, n_reps): base, unit
    return v_true, tr_MR, dof, est, rss, cross


def scan_time_sweep(
    spec: Any,
    scan_times: list[float],
    n_designs: int,
    contrasts: dict[str, list[float] | np.ndarray],
    noise: list[dict[str, Any]],
    beta_pattern: list[float] | np.ndarray | None = None,
    hrf: str = "spmg1",
    alpha: float = 0.001,
    target: float = 0.8,
    poly_degree: int | None = None,
    seed: int = 0,
    progress: bool = True,
) -> dict[str, Any]:
    """Effect needed for ``target`` power as the per-run scan time varies. Analytic.

    Each scan time re-resolves the experiment (counts sized to the whole
    units that fit, runs trimmed to them -- see ExperimentSpec.run_seconds),
    draws ``n_designs`` realizations and solves for the effect every contrast
    needs, at every noise level. The effect is in the simulation's units: the
    response amplitude for a condition contrast (``beta_pattern`` applied),
    the difference itself for a difference contrast. Assumes the fitted HRF
    is right; under a mismatch the Monte Carlo at one scan time is the
    referee.

    For a fixed design the effect falls as 1/sqrt(T), so ``per_minute``
    (effect x sqrt(total minutes)) is flat where the design scales ideally
    and compares designs of different lengths per unit of scan time.

    Returns {'rows': one per scan time x realization x noise x contrast,
    'skipped': {scan_time: reason}}.
    """
    from dataclasses import replace

    from tqdm import tqdm

    from .experiment import realize

    scorer = RealizationScorer(
        spec.tr, contrasts, noise, beta_pattern, hrf, alpha, target, poly_degree
    )
    rows: list[dict[str, Any]] = []
    skipped: dict[float, str] = {}
    for st in tqdm(scan_times, desc="scan times", leave=True, disable=not progress):
        sp = replace(spec, scan_time=float(st))
        try:
            reals = [realize(sp, seed + d) for d in range(n_designs)]
        except ValueError as exc:
            skipped[float(st)] = str(exc)
            continue
        for d, real in enumerate(reals):
            sc = scorer.score(real, shape=False)
            if sc is None:
                skipped.setdefault(float(st), "rank-deficient in some realizations")
                continue
            for (label, name), eff in sc["needed"].items():
                rows.append(
                    {
                        "scan_time": float(st),
                        "run_s": sc["run_s"],
                        "minutes": sc["minutes"],
                        "counts": sc["counts"],
                        "design": d,
                        "noise": label,
                        "contrast": name,
                        "needed": eff,
                        "per_minute": eff * float(np.sqrt(sc["minutes"])),
                    }
                )
    return {"rows": rows, "skipped": skipped}


class RealizationScorer:
    """Analytic scores of one realization: what a design search ranks on.

    Detection -- the effect each contrast needs for ``target`` power, in the
    simulation's units (response amplitude for a condition contrast,
    ``beta_pattern`` applied; the difference itself for a difference
    contrast) -- and, unless ``shape=False``, response-shape estimation
    (:func:`estimation_quality`). No Monte Carlo: ~10-20 ms a realization,
    so thousands of candidate designs are affordable. Assumes the fitted HRF.
    """

    def __init__(
        self,
        tr: float,
        contrasts: dict[str, list[float] | np.ndarray],
        noise: list[dict[str, Any]],
        beta_pattern: list[float] | np.ndarray | None = None,
        hrf: str = "spmg1",
        alpha: float = 0.001,
        target: float = 0.8,
        poly_degree: int | None = None,
        mean_response: float = 1.0,
        trial_sd: float = 0.5,
    ):
        from .core import default_microtime_dt, hrfs_from_spec

        self.tr, self.noise, self.alpha, self.target = tr, noise, alpha, target
        self.mean_response, self.trial_sd = mean_response, trial_sd
        self.hrf = hrf
        self.poly_degree = poly_degree
        self.dt = default_microtime_dt(tr)
        self.bases = hrfs_from_spec(hrf, self.dt, torch.device("cpu"))[0][1]
        self.names = list(contrasts)
        self.W = np.array([np.asarray(contrasts[c], dtype=float) for c in self.names])
        pattern = (
            np.ones(self.W.shape[1]) if beta_pattern is None else np.asarray(beta_pattern, float)
        )
        # Which contrasts the sweep gives a true effect (a condition contrast with
        # -pattern B=0 has none). The effect reported is the contrast's own value
        # in % signal -- not the sweep, which -pattern scales.
        self.live = np.array([has_true_effect(w, pattern) for w in self.W])
        self.pattern = pattern

    def score(
        self, real: Any, shape: bool = True, single: bool = False, steps: bool = False
    ) -> dict[str, Any] | None:
        """Scores of one realization, or None if its model is rank-deficient.

        ``single`` adds single-trial estimability (:func:`single_trial_quality`,
        means over conditions): 'lss_sd' and 'lsa_sd' per noise label, 'leakage',
        and per noise label 'unreliability' (1 - the better trial-pattern
        reliability of LSS and ridge) and 'ridge_frac'. ``steps`` adds
        'shape_steps' per noise label: the library steps two shapes must be
        apart to be told apart (:func:`shape_steps`, mean over conditions;
        one past the maximum where never), at mean_response x pattern.
        """
        from fastfuncstuff.cli_utils import auto_polort

        from .core import build_task_design

        cpu = torch.device("cpu")
        lengths = list(real.run_lengths)
        X_task = build_task_design(
            real.onsets, real.durations, self.tr, lengths, self.bases, self.dt, device=cpu
        )
        pdeg = (
            self.poly_degree
            if self.poly_degree is not None
            else auto_polort(max(lengths) * self.tr)
        )
        X = torch.cat([X_task, _nuisance(lengths, pdeg)], dim=1)
        if int(torch.linalg.matrix_rank(X)) < X.shape[1]:
            return None
        need = _contrast_needed(X, lengths, self.tr, self.W, self.noise, self.alpha, self.target)
        out: dict[str, Any] = {
            "minutes": sum(lengths) * self.tr / 60.0,
            "run_s": float(np.mean(lengths)) * self.tr,
            "counts": "/".join(str(c) for c in real.counts),
            "n_dropped": getattr(real, "n_dropped", 0),
            "needed": {
                (label, name): (float(val) if live else float("nan"))
                for label, v in need.items()
                for name, val, live in zip(self.names, v, self.live, strict=True)
            },
        }
        # 'detection': the mean over every contrast with a true effect -- the
        # objective when all contrasts matter, as Liu & Frank average efficiency.
        for label in need:
            vals = [out["needed"][(label, n)] for n in self.names]
            finite = [v for v in vals if np.isfinite(v)]
            out["needed"][(label, "detection")] = float(np.mean(finite)) if finite else np.nan
        if shape:
            est = estimation_quality(real, self.tr, self.noise, poly_degree=pdeg, hrf=self.hrf)
            out["shape_sd"] = {k: float(np.mean(v)) for k, v in est["shape_sd"].items()}
            out["xi"] = est["xi"]
            out["liu_power"] = est["liu_power"]
        if single:
            st = single_trial_quality(
                real, self.tr, self.noise, self.hrf, pdeg, self.mean_response, self.trial_sd
            )
            out["lss_sd"] = {k: float(np.mean(v)) for k, v in st["lss_sd"].items()}
            out["lsa_sd"] = {k: float(np.mean(v)) for k, v in st["lsa_sd"].items()}
            out["leakage"] = float(np.mean(st["leakage"]))
            # What a trial-wise search minimizes: the best of LSS and ridge, as 1 - r.
            out["unreliability"] = {
                k: 1.0 - max(v["lss"], v["ridge"]) for k, v in st["reliability"].items()
            }
            out["ridge_frac"] = {k: v["ridge_frac"] for k, v in st["reliability"].items()}
        if steps:
            ss = shape_steps(
                real, self.tr, self.noise, self.mean_response * self.pattern, self.alpha,
                self.target, pdeg,
            )  # fmt: skip
            cap = ss["max_step"] + 1.0
            out["shape_steps"] = {
                k: float(np.mean(np.where(np.isfinite(v), v, cap))) for k, v in ss["steps"].items()
            }
        return out


def has_mismatch(rows: list[dict[str, Any]], rtol: float = 1e-3) -> bool:
    """Whether the fitted model differs from the generating one (the estimate is biased)."""
    return any(
        abs(r["expected_est"] - r["true_effect"]) > rtol * max(abs(r["true_effect"]), 1e-9)
        for r in rows
        if "expected_est" in r
    )


def power_column(rows: list[dict[str, Any]]) -> str:
    """Which power to trust: analytic when the model is right, Monte Carlo otherwise.

    Under an HRF mismatch the unfit response inflates the residual variance.
    The analytic curve includes that in expectation, but with a large misfit
    the noncentral-t approximation drifts (0.155 off for A-B on a 4% shared
    response), and the Monte Carlo is the referee.
    """
    if any(r.get("estimator") == "reml" for r in rows):
        return "power"  # the fitted-REML Monte Carlo
    return "power" if has_mismatch(rows) else "power_predicted"


def amplitude_for_power(
    result: dict[str, Any], target: float = 0.8, column: str | None = None
) -> dict[tuple[str, str], float]:
    """The effect reaching ``target`` power, per (noise, contrast), by interpolation.

    The effect is the contrast's *true* value in % signal (``true_effect``),
    not the swept amplitude: with ``-pattern E1=3`` the sweep is a third of
    E1's response, and reporting the sweep read E1 as three times easier than
    it is. It is where power last crosses ``target`` -- beyond it power stays
    there. The first crossing was wrong for a non-monotone curve (a wrong-HRF
    shared response gives high power at zero difference, a dip, then a rise).

    nan where the sweep never reaches it, or the contrast has no true effect.
    ``column`` defaults to :func:`power_column` -- the analytic curve, or the
    Monte Carlo under a model mismatch.
    """
    if column is None:
        column = power_column(result["table"])
    out: dict[tuple[str, str], float] = {}
    keys = {(r["noise"], r["contrast"]) for r in result["table"]}
    for key in keys:
        rows = sorted(
            (r for r in result["table"] if (r["noise"], r["contrast"]) == key),
            key=lambda r: r["amplitude"],
        )
        eff = np.array([abs(r["true_effect"]) for r in rows])
        pw = np.array([r[column] for r in rows])
        if not np.isfinite(pw).all():
            out[key] = float("nan")
            continue
        below = np.nonzero(pw < target)[0]
        if not np.any(eff > 0) or below.size == 0 or below[-1] == len(rows) - 1:
            out[key] = float("nan")
            continue
        j = below[-1]
        out[key] = float(np.interp(target, [pw[j], pw[j + 1]], [eff[j], eff[j + 1]]))
    return out


def effect_needed(result: dict[str, Any], target: float = 0.8) -> dict[tuple[str, str], np.ndarray]:
    """:func:`amplitude_for_power` per realization (and true HRF), per (noise, contrast).

    A jittered design's answer is a distribution: each array holds one value
    per (design, true_hrf) cell, nan where that cell never reaches ``target``.
    """
    rows = result["table"]
    cells: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for r in rows:
        cells.setdefault((r["design"], r.get("true_hrf", "")), []).append(r)
    need = [amplitude_for_power({"table": sub}, target) for _, sub in sorted(cells.items())]
    keys = dict.fromkeys((r["noise"], r["contrast"]) for r in rows)
    return {k: np.array([n.get(k, np.nan) for n in need], dtype=float) for k in keys}


def _nc_for_power(target: float, crit: float, dof: float) -> float:
    """Noncentrality at which two-tailed power reaches ``target``."""
    from scipy.optimize import brentq

    upper = min(200.0, max(8.0, crit))
    value = _two_tailed_power(crit, dof, upper) - target
    while value < 0 and upper < 200.0:
        upper = min(200.0, 2 * upper)
        value = _two_tailed_power(crit, dof, upper) - target
    return float(
        brentq(
            lambda nc: value if nc == upper else _two_tailed_power(crit, dof, nc) - target,
            0.0,
            upper,
        )
    )


_NC_U_MAX = 0.5  # 1/dof: the table covers dof >= 2


@lru_cache(maxsize=16)
def _nc_table(target: float, alpha: float) -> Any:
    """Noncentrality for ``target`` power at two-tailed ``alpha``, as a function of 1/dof.

    Smooth in u = 1/dof, so 64 Chebyshev nodes interpolate it to ~1e-10. A
    brentq per realization (~12 noncentral-t evaluations) was 2 ms of every
    design a -explore scored.
    """
    from scipy.interpolate import BarycentricInterpolator

    k = np.arange(64)
    u = 0.5 * _NC_U_MAX * (1 - np.cos(np.pi * k / 63))  # Chebyshev-Lobatto on [0, max]
    dof = 1.0 / np.maximum(u, 1e-9)
    nc = [_nc_for_power(target, float(stats.t.ppf(1 - alpha / 2, d)), float(d)) for d in dof]
    return BarycentricInterpolator(u, nc)


def _nc_needed(target: float, alpha: float, dof: float) -> float:
    """:func:`_nc_for_power` at the two-tailed ``alpha`` critical value, tabulated in 1/dof."""
    if dof < 1.0 / _NC_U_MAX:
        return _nc_for_power(target, float(stats.t.ppf(1 - alpha / 2, dof)), dof)
    return float(_nc_table(float(target), float(alpha))(1.0 / dof))


def design_quality(
    design: torch.Tensor | np.ndarray,
    run_lengths: list[int],
    tr: float,
    noise: list[dict[str, Any]],
    alpha: float = 0.001,
    target: float = 0.8,
    poly_degree: int | None = None,
) -> dict[str, Any]:
    """How estimable each condition and each pairwise difference is, before any simulation.

    The same GLM :func:`simulate_design_power` fits (task + per-run Legendre
    drift), judged three ways:

    - ``rank`` of the full model; if deficient, ``null_weights`` is the
      combination of task regressors the drift and the other conditions
      reproduce exactly (e.g. two conditions with the same timing: +1, -1).
    - ``vif`` per condition, after drift removal: how much the other
      conditions inflate its variance (1 = orthogonal; > 5 hard, > 10 severe).
    - ``corr``: regressor correlation after drift removal -- what the fit sees.
      The raw correlation carries each run's mean and trend.
    - ``needed[label]``: (n_cond, n_cond) effect (PSC) for ``target`` power at
      each noise condition. Diagonal: a condition against baseline; (i, j): the
      difference i - j. Analytic, with the same ARMA-corrected t as the
      engine, so under a correct HRF it is what the simulation will find.
      Positively correlated regressors make their *difference* costly: the
      fit can tell that something happened, not which.
    """
    from fastfuncstuff.cli_utils import auto_polort

    X_task = torch.as_tensor(design, dtype=torch.float64)
    n_t, n_cond = X_task.shape
    if poly_degree is None:
        poly_degree = auto_polort(max(run_lengths) * tr)
    D = _nuisance(run_lengths, poly_degree)
    X = torch.cat([X_task, D], dim=1)
    n_p = X.shape[1]
    rank = int(torch.linalg.matrix_rank(X))
    Q, _ = torch.linalg.qr(D)
    Xt = X_task - Q @ (Q.T @ X_task)  # drift removed
    norms = Xt.norm(dim=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = ((Xt.T @ Xt) / torch.outer(norms, norms)).numpy()
    out: dict[str, Any] = {
        "rank": rank,
        "n_columns": n_p,
        "n_task": n_cond,
        "poly_degree": poly_degree,
        "deficient": rank < n_p,
        "corr": corr,
    }
    if rank < n_p:
        v = torch.linalg.svd(X)[2][-1, :n_cond]
        out["null_weights"] = (v / v.abs().max()).numpy() if v.abs().max() > 0 else v.numpy()
        out["vif"] = np.full(n_cond, np.inf)
        out["needed"] = {}
        return out

    G = Xt.T @ Xt
    out["vif"] = (torch.diag(torch.linalg.inv(G)) * torch.diag(G)).numpy()
    pairs = [(i, j) for i in range(n_cond) for j in range(i + 1)]
    W = np.zeros((len(pairs), n_cond))
    for k, (i, j) in enumerate(pairs):
        W[k, i] += 1.0
        if j != i:
            W[k, j] -= 1.0
    needed: dict[str, np.ndarray] = {}
    for label, v in _contrast_needed(X, run_lengths, tr, W, noise, alpha, target).items():
        m = np.full((n_cond, n_cond), np.nan)
        for (i, j), val in zip(pairs, v, strict=True):
            m[i, j] = m[j, i] = val
        needed[label] = m
    out["needed"] = needed
    return out


def _contrast_needed(
    X: torch.Tensor,
    run_lengths: list[int],
    tr: float,
    W: np.ndarray,
    noise: list[dict[str, Any]],
    alpha: float,
    target: float,
) -> dict[str, np.ndarray]:
    """Contrast value (PSC) each row of weights ``W`` needs for ``target`` power.

    ``X`` is the full model (task columns first, then drift). Analytic, with
    the engine's ARMA-corrected t, so under a correct HRF it is what the
    simulation finds. One value per row of W, per noise label.
    """
    n_p = X.shape[1]
    P = torch.linalg.inv(X.T @ X) @ X.T
    C = torch.zeros(W.shape[0], n_p, dtype=torch.float64)
    C[:, : W.shape[1]] = torch.as_tensor(W, dtype=torch.float64)
    out: dict[str, np.ndarray] = {}
    terms: dict[tuple[float, float] | None, tuple[torch.Tensor, float]] = {}  # var, nc
    for k, cond in enumerate(noise):
        label = str(cond.get("label", f"noise{k}"))
        kw = {key: v for key, v in cond.items() if key != "label"}
        ab = _noise_arma(kw, tr)
        if ab not in terms:  # every -tsnr level shares (a, b)
            var, _, dof = _noise_terms(ab, X, P, C, run_lengths)
            terms[ab] = (var, _nc_needed(target, alpha, dof))
        var, nc = terms[ab]
        sd = 100.0 / float(kw["tsnr"])  # noise SD in PSC
        out[label] = nc * sd * np.sqrt(var.numpy())
    return out


def realizations_design_quality(
    realizations: list[Any],
    tr: float,
    noise: list[dict[str, Any]],
    hrf: str = "spmg1",
    alpha: float = 0.001,
    target: float = 0.8,
    poly_degree: int | None = None,
    single_trial_designs: int = 20,
    mean_response: float = 1.0,
    trial_sd: float = 0.5,
) -> list[dict[str, Any]]:
    """:func:`design_quality` of the fitted model, one dict per realization.

    Also response-shape estimation for each, and single-trial estimability
    (:func:`single_trial_quality`, under 'single') for the first
    ``single_trial_designs`` -- it costs 20-65 ms a realization.
    """
    from .core import build_task_design, default_microtime_dt, hrfs_from_spec

    cpu = torch.device("cpu")
    dt = default_microtime_dt(tr)
    bases = hrfs_from_spec(hrf, dt, cpu)[0][1]
    out = []
    for k, real in enumerate(realizations):
        X = build_task_design(
            real.onsets, real.durations, tr, real.run_lengths, bases, dt, device=cpu
        )
        q = design_quality(X, list(real.run_lengths), tr, noise, alpha, target, poly_degree)
        if not q["deficient"]:
            q.update(estimation_quality(real, tr, noise, poly_degree=q["poly_degree"], hrf=hrf))
            if k < single_trial_designs:  # per-trial numbers barely vary across realizations
                q["single"] = single_trial_quality(
                    real, tr, noise, hrf, q["poly_degree"], mean_response, trial_sd
                )
        out.append(q)
    return out


def fir_onsets(realization: Any, tr: float) -> np.ndarray:
    """(n_timepoints, n_conditions) onset counts on the TR grid, runs concatenated."""
    n = sum(realization.run_lengths)
    on = np.zeros((n, len(realization.conditions)))
    start = 0
    for run, n_run in enumerate(realization.run_lengths):
        for q, per_run in enumerate(realization.onsets):
            for t in per_run[run]:
                k = int(np.floor(t / tr + 1e-9))
                if 0 <= k < n_run:
                    on[start + k, q] += 1
        start += n_run
    return on


def stimulus_pattern(realization: Any, tr: float) -> np.ndarray:
    """(n_timepoints, n_conditions): 1 where a condition's stimulus is on in that TR bin.

    Liu & Frank's designs are stimulus patterns on the TR grid -- an event is
    one bin, a 30 s block thirty -- not onsets; with onsets only, a block was a
    single impulse and scored like a sparse event.
    """
    n = sum(realization.run_lengths)
    pat = np.zeros((n, len(realization.conditions)))
    start = 0
    for run, n_run in enumerate(realization.run_lengths):
        for q, per_run in enumerate(realization.onsets):
            d = float(realization.durations[q])
            for t in per_run[run]:
                a = int(np.floor(t / tr + 1e-9))
                b = max(a + 1, int(np.ceil((t + d) / tr - 1e-9)))
                pat[start + max(a, 0) : start + min(b, n_run), q] = 1.0
        start += n_run
    return pat


def estimation_quality(
    realization: Any,
    tr: float,
    noise: list[dict[str, Any]],
    window: float = 16.0,
    poly_degree: int | None = None,
    hrf: str = "spmg1",
) -> dict[str, Any]:
    """How well the design estimates each condition's response *shape* (FIR), not its size.

    Detection (effect needed for 80% power) and estimation trade off (Liu et
    al. 2001): rapid jittered designs pack events into a plateau that the drift
    absorbs, and detect poorly, but sample the response at many lags and
    recover its shape well; blocks are the reverse. So both are reported.

    The model is a joint FIR -- one column per TR bin over ``window`` seconds
    after onset, every condition together -- plus the same per-run drift.
    Returns
        'shape_sd'  : {noise label: (n_cond,)} SD of one bin's estimate (PSC),
                      root-mean over the bins, under that noise's ARMA -- how
                      precisely the response time course is seen; inf when the
                      FIR is not estimable (e.g. a fixed SOA aliasing the lags)
        'xi'        : Liu & Frank's estimation efficiency as a fraction of its
                      bound (white noise, design only; comparable to the papers)
        'liu_power' : their detection power as a fraction of its bound -- with
                      xi, a point on Liu et al. (2001)'s trade-off plane
        'fir_lags'  : bins per condition
    """
    from fastfuncstuff.cli_utils import auto_polort

    from .metrics import compute_detection_power, compute_estimation_efficiency

    lengths = list(realization.run_lengths)
    if poly_degree is None:
        poly_degree = auto_polort(max(lengths) * tr)
    k = max(1, int(round(window / tr)))
    on = fir_onsets(realization, tr)
    n_t, n_cond = on.shape
    cols = []
    for q in range(n_cond):
        for lag in range(k):
            c = np.zeros(n_t)
            start = 0
            for n_run in lengths:  # lags never cross a run boundary
                seg = on[start : start + n_run, q]
                c[start + lag : start + n_run] = seg[: n_run - lag] if lag < n_run else 0
                start += n_run
            cols.append(c)
    D = _nuisance(lengths, poly_degree)
    X = torch.cat([torch.as_tensor(np.stack(cols, 1)), D], dim=1)
    out: dict[str, Any] = {"fir_lags": k}
    # Liu & Frank's two numbers need no FIR: the stimulus pattern on the TR grid.
    # Estimation efficiency, and detection power beside it -- the two axes of
    # Liu et al. (2001)'s trade-off plane -- with the same k, and the HRF on the
    # TR grid, as their model has it.
    from .core import hrfs_from_spec

    pat = stimulus_pattern(realization, tr)
    eff = compute_estimation_efficiency(
        pat, n_cond, k, poly_degree=-1, nuisance=D.numpy(), normalize=True
    )
    h0 = hrfs_from_spec(hrf, tr, torch.device("cpu"))[0][1][0].numpy()[:k]
    det = compute_detection_power(pat, h0, n_cond, poly_degree=-1, nuisance=D.numpy())
    liu = {"xi": float(eff["total_normalized"]), "liu_power": float(det["total_normalized"])}
    if int(torch.linalg.matrix_rank(X)) < X.shape[1]:
        out["shape_sd"] = {
            str(c.get("label", f"noise{i}")): np.full(n_cond, np.inf) for i, c in enumerate(noise)
        }
        out.update(liu)
        return out
    P = torch.linalg.inv(X.T @ X) @ X.T
    shape: dict[str, np.ndarray] = {}
    cache: dict[tuple[float, float] | None, np.ndarray] = {}
    for i, cond in enumerate(noise):
        label = str(cond.get("label", f"noise{i}"))
        kw = {key: v for key, v in cond.items() if key != "label"}
        ab = _noise_arma(kw, tr)
        if ab not in cache:  # every -tsnr level shares (a, b)
            if ab is None:
                var = torch.diagonal(P @ P.T)
            else:
                R = _block_correlation(tuple(lengths), *ab, False)
                var = torch.diagonal(_corrected_terms(X, P, R)[0])
            cache[ab] = var[: n_cond * k].reshape(n_cond, k).mean(dim=1).sqrt().numpy()
        shape[label] = 100.0 / float(kw["tsnr"]) * cache[ab]
    out["shape_sd"] = shape
    out.update(liu)
    return out


def trial_regressors(realization: Any, tr: float, bases: torch.Tensor, dt: float) -> tuple:
    """One regressor per trial (unit peak, like the condition regressors they sum to).

    The GLM tools' builder in its single-trial mode: one convolution per
    condition, trials in chronological order -- building each trial as its own
    condition took one convolution per trial (72 of 232 ms at 218 trials).
    Returns (X (n_t, n_trials), condition index of each trial).
    """
    from fastfuncstuff.design.matrices import build_event_design_microtime

    _, X, cond, _ = build_event_design_microtime(
        all_onsets=[[np.asarray(r, dtype=np.float64) for r in c] for c in realization.onsets],
        durations=list(realization.durations),
        hrf_bases=bases,
        n_timepoints_per_run=list(realization.run_lengths),
        tr=tr,
        microtime_dt=dt,
        device=torch.device("cpu"),
        return_single_trials=True,
    )
    return X.double(), np.asarray(cond, dtype=int)


def lss_rows(Xt: torch.Tensor, cond: np.ndarray, D: torch.Tensor) -> torch.Tensor:
    """LSS estimator rows (n_trials, n_t): trial i's estimate is ``A[i] @ y``.

    Each trial's own model -- its regressor, the rest of its condition, the
    other conditions, drift ``D`` -- solved at once by FWL: project out the
    other conditions and drift, then a 2-column solve per trial.
    """
    n_t, n_tr = Xt.shape
    n_cond = int(cond.max()) + 1
    S = torch.stack([Xt[:, cond == q].sum(dim=1) for q in range(n_cond)], dim=1)
    # LSS by FWL: for trial i in condition c, project out O = [other conditions, drift];
    # then [u, v] = M_O [x_i, s_c - x_i] and b_i = row 1 of (G^-1 [u v]').
    A = torch.empty(n_tr, n_t, dtype=torch.float64)
    for q in range(n_cond):
        O = torch.cat([S[:, [k for k in range(n_cond) if k != q]], D], dim=1)
        Q, _ = torch.linalg.qr(O)
        idx = np.flatnonzero(cond == q)
        U = Xt[:, idx] - Q @ (Q.T @ Xt[:, idx])  # (n_t, m)
        sc = S[:, q] - Q @ (Q.T @ S[:, q])
        V = sc[:, None] - U
        uu, uv, vv = (U * U).sum(0), (U * V).sum(0), (V * V).sum(0)
        det = uu * vv - uv**2
        ok = det > 1e-10 * uu * vv
        rows = (vv[:, None] * U.T - uv[:, None] * V.T) / torch.where(ok, det, 1.0)[:, None]
        # A condition's only trial has no "rest of the condition" (v = 0): LSS is then
        # just its own regressor. Anything else singular is not estimable.
        alone = vv <= 1e-12 * uu
        rows[alone] = (U.T / uu[:, None])[alone]
        rows[~ok & ~alone] = float("nan")
        A[idx] = rows
    return A


def single_trial_quality(
    realization: Any,
    tr: float,
    noise: list[dict[str, Any]],
    hrf: str = "spmg1",
    poly_degree: int | None = None,
    mean_response: float = 1.0,
    trial_sd: float = 0.5,
) -> dict[str, Any]:
    """How well each *trial* can be estimated on its own -- what MVPA/RSA and trial-wise
    analyses need, and a design can be good for conditions and poor for trials.

    LSS (least squares separate; Mumford et al. 2012): each trial gets its own
    regressor, the rest of its condition one more, the other conditions and
    per-run drift the usual. Two numbers per trial:
      - precision: SD of the trial's estimate from noise (PSC) under the noise ARMA;
      - leakage: sqrt(sum_j w_ij^2) over the other trials j, where w_ij is how
        much of trial j's response the estimate of trial i picks up -- the error
        per unit of trial-to-trial amplitude SD. Zero only if trials don't overlap.
    LSA (least squares all: one regressor per trial) has no leakage by
    construction but its variance explodes as trials overlap; its precision
    is reported too (inf if the trial regressors are not all estimable).

    Ridge (single-trial LSA with fractional ridge, as GLMsingle): biased on
    purpose, so SD and leakage cannot score it. All three are therefore also
    put on one measure, 'reliability': the expected correlation between the
    estimated and the true trial-to-trial deviations (within condition), for
    true amplitudes ``mean_response`` + N(0, ``trial_sd``^2) (PSC). Ridge's
    shrinkage is taken at its best on a grid -- the oracle that choosing it by
    cross-validation across runs estimates (so it needs two or more runs) --
    and reported as GLMsingle's fraction (norm of the ridge solution over the
    unregularized one, in expectation).

    Returns per noise label the median over trials of 'lss_sd', 'lsa_sd'
    (per condition), 'leakage' (per condition; noise-free), and
    'reliability' {label: {'lss', 'lsa', 'ridge', 'ridge_frac'}}.
    """
    from fastfuncstuff.cli_utils import auto_polort

    from .core import default_microtime_dt, hrfs_from_spec

    lengths = list(realization.run_lengths)
    if poly_degree is None:
        poly_degree = auto_polort(max(lengths) * tr)
    dt = default_microtime_dt(tr)
    bases = hrfs_from_spec(hrf, dt, torch.device("cpu"))[0][1]
    Xt, cond = trial_regressors(realization, tr, bases, dt)
    n_t, n_tr = Xt.shape
    n_cond = len(realization.conditions)
    D = _nuisance(lengths, poly_degree)

    A = lss_rows(Xt, cond, D)  # estimator rows: b_i = A[i] @ y
    W = A @ Xt  # (n_tr, n_tr): w_ij, response of trial j in trial i's estimate
    off = W - torch.diag(torch.diagonal(W))
    leak = torch.sqrt((off * off).sum(dim=1)).numpy()

    # LSA and ridge work on the drift-projected trial regressors (ridge must not
    # shrink the drift), through one SVD Xp = U S V'. LSA is estimable iff Xp has
    # full column rank, and its per-trial variance is diag(V S^-1 B S^-1 V') with
    # B = U'RU -- no second SVD for the rank, no separate solve.
    Qd, _ = torch.linalg.qr(D)
    Xp = Xt - Qd @ (Qd.T @ Xt)
    Ux, sx, Vh = torch.linalg.svd(Xp, full_matrices=False)
    V = Vh.T
    lsa_ok = float(sx.min()) > 1e-8 * float(sx.max())
    Vs = V / sx
    b_cache: dict[tuple[float, float] | None, torch.Tensor] = {}

    def noise_basis(ab):
        """B = U'RU for a noise ARMA (shared by LSA and ridge)."""
        if ab not in b_cache:
            R = None if ab is None else _block_correlation(tuple(lengths), *ab, False)
            b_cache[ab] = Ux.T @ Ux if R is None else Ux.T @ R @ Ux
        return b_cache[ab]

    out: dict[str, Any] = {"n_trials": n_tr, "lss_sd": {}, "lsa_sd": {}}
    out["leakage"] = np.array([float(np.nanmedian(leak[cond == q])) for q in range(n_cond)])
    cache: dict[tuple[float, float] | None, tuple[np.ndarray, np.ndarray]] = {}
    for i, c in enumerate(noise):
        label = str(c.get("label", f"noise{i}"))
        kw = {key: v for key, v in c.items() if key != "label"}
        ab = _noise_arma(kw, tr)
        if ab not in cache:  # every -tsnr level shares (a, b)
            R = None if ab is None else _block_correlation(tuple(lengths), *ab, False)
            lss = (A * A).sum(1) if R is None else ((A @ R) * A).sum(1)
            if not lsa_ok:
                lsa = torch.full((n_tr,), float("inf"), dtype=torch.float64)
            else:
                lsa = ((Vs @ noise_basis(ab)) * Vs).sum(1)
            cache[ab] = (lss.sqrt().numpy(), lsa.sqrt().numpy())
        sd = 100.0 / float(kw["tsnr"])
        lss_sd, lsa_sd = cache[ab]
        out["lss_sd"][label] = np.array(
            [sd * float(np.nanmedian(lss_sd[cond == q])) for q in range(n_cond)]
        )
        out["lsa_sd"][label] = np.array(
            [sd * float(np.nanmedian(lsa_sd[cond == q])) for q in range(n_cond)]
        )

    # Trial-pattern reliability for LSS, LSA and ridge.
    same = torch.as_tensor(cond[:, None] == cond[None, :], dtype=torch.float64)
    Cc = torch.eye(n_tr, dtype=torch.float64) - same / same.sum(dim=1, keepdim=True)
    ones = torch.ones(n_tr, dtype=torch.float64)
    mu, tau2 = float(mean_response), float(trial_sd) ** 2
    s2 = sx**2
    lams = [0.0] if lsa_ok else []
    lams += (float(torch.median(s2)) * np.logspace(-3, 2, 26)).tolist()
    # Ridge in the SVD basis Xp = U S V': Z = V g V', N = var V d B d V' with
    # g = s^2/(s^2+lam), d = s/(s^2+lam), B = U'RU. With M = V'CV (C centres
    # within condition) every trace below is a quadratic form in g or d, so the
    # whole lambda grid costs O(n^2) per lambda instead of several n^3 products
    # -- and the noise enters as one scalar per level. 70 of 232 ms before.
    Mv = V.T @ Cc @ V
    MM = Mv * Mv
    a = V.T @ ones
    e_dd = tau2 * float(torch.trace(Cc))
    denominator = s2 + torch.as_tensor(lams, dtype=s2.dtype)[:, None]
    g, d = s2 / denominator, sx / denominator
    ag = a * g
    e_hd = (tau2 * (g * torch.diagonal(Mv)).sum(1)).numpy()
    e_sig = (tau2 * ((g @ MM) * g).sum(1) + mu**2 * ((ag @ Mv) * ag).sum(1)).numpy()
    e_norm = (mu**2 * (ag * ag).sum(1) + tau2 * (g * g).sum(1)).numpy()
    out["reliability"] = {}
    urc: dict[tuple[float, float] | None, tuple] = {}
    for i, c in enumerate(noise):
        label = str(c.get("label", f"noise{i}"))
        kw = {key: v for key, v in c.items() if key != "label"}
        ab = _noise_arma(kw, tr)
        if ab not in urc:  # everything noise-shaped depends on (a, b) only
            R = None if ab is None else _block_correlation(tuple(lengths), *ab, False)
            B = noise_basis(ab)
            BM = B * Mv
            per_lam = torch.stack(
                (((d @ BM) * d).sum(1), (d * d * torch.diagonal(B)).sum(1)), dim=1
            ).numpy()
            AR = A if R is None else A @ R
            urc[ab] = (per_lam, AR @ A.T)
        per_lam, ARA = urc[ab]
        var = (100.0 / float(kw["tsnr"])) ** 2
        rel = {
            "lss": _trial_reliability(
                torch.nan_to_num(W), torch.nan_to_num(var * ARA), Cc, ones, mu, tau2
            )
        }
        # The least shrinkage within a hair of the best: a correlation is blind to
        # scaling, so with barely-overlapping trials it is flat in lambda and the
        # plain argmax reported a meaningless fraction of 0.01.
        e_hh = e_sig + var * per_lam[:, 0]
        scan = np.divide(
            e_hd,
            np.sqrt(np.maximum(e_hh * e_dd, 0.0)),
            out=np.zeros_like(e_hd),
            where=(e_hh > 0) & (e_dd > 0),
        )
        e_b = e_norm + var * per_lam[:, 1]
        best_i = int(np.flatnonzero(scan >= scan.max() - 1e-3)[0])
        e0 = float(e_b[0]) if lsa_ok else 0.0
        rel["ridge"] = float(scan[best_i])
        rel["ridge_frac"] = float(np.sqrt(e_b[best_i] / e0)) if e0 else float("nan")
        rel["lsa"] = float(scan[0]) if lsa_ok else 0.0
        rel["ridge_lambda"] = lams[best_i]
        out["reliability"][label] = rel
    return out


def _trial_reliability(Z, N, C, ones, mu: float, tau2: float) -> float:
    """Expected correlation of estimated with true within-condition trial deviations.

    Estimates b = Z beta + e with beta = mu + delta, delta ~ N(0, tau2 I), and
    Cov(e) = N; deviations are centred within condition by C. A ratio of
    expectations: E[d_hat . d] / sqrt(E[|d_hat|^2] E[|d|^2]).
    """
    CZ = C @ Z
    e_dd = tau2 * float(torch.trace(C))
    e_hd = tau2 * float(torch.trace(CZ @ C))
    e_hh = (
        tau2 * float(torch.trace(CZ @ C @ CZ.T))
        + mu**2 * float((CZ @ ones) @ (CZ @ ones))
        + float(torch.trace(C @ N @ C))
    )
    return e_hd / float(np.sqrt(e_hh * e_dd)) if e_hh > 0 and e_dd > 0 else 0.0


def tent_estimate(
    realization: Any,
    tr: float,
    noise: list[dict[str, Any]],
    amplitudes: list[float] | np.ndarray,
    true_hrf: str = "spmg1",
    window: float | None = None,
    spacing: float = 2.0,
    poly_degree: int | None = None,
    seed: int = 0,
    tail: float = 20.0,
) -> dict[str, Any]:
    """Deconvolve one simulated voxel with TENTs: the response shape a design lets you see.

    AFNI-style TENT (piecewise-linear) knots every ``spacing`` seconds after
    onset, evaluated at the exact time since each onset -- not TR-locked, so
    jittered sub-TR onsets are fine (a TR-grid FIR would round them). Each
    condition gets its own window, its duration plus ``tail`` (the HRF's
    return to baseline), rounded up to a knot: a fixed 20 s showed an 18 s
    block's response cut off at its plateau. ``window`` fixes one for all.
    One voxel per noise condition responds with ``amplitudes`` (PSC, per
    condition) through ``true_hrf``; the fit is TENTs for every condition plus
    the per-run drift.

    Returns per condition (lists, one entry each) 'knots' (s), 'fine_t' and
    'truth' -- the true response to one event, duration included -- and per
    noise label 'est' and 'se' (lists of arrays over that condition's knots):
    the estimate from that voxel and its SE under the noise ARMA (what a fresh
    voxel would scatter by).
    """
    from fastfuncstuff.cli_utils import auto_polort
    from fastfuncstuff.design.matrices import make_tent_design

    from .core import build_task_design, default_microtime_dt, hrfs_from_spec

    cpu = torch.device("cpu")
    lengths = list(realization.run_lengths)
    n_cond = len(realization.conditions)
    if poly_degree is None:
        poly_degree = auto_polort(max(lengths) * tr)
    windows = [
        float(window)
        if window is not None
        else float(np.ceil((d + tail) / spacing - 1e-9) * spacing)
        for d in realization.durations
    ]
    ks = [int(round(w / spacing)) + 1 for w in windows]
    blocks = []
    for r, n_run in enumerate(lengths):
        cols = [
            make_tent_design(
                [np.asarray(realization.onsets[q][r], dtype=float)], 0.0, windows[q], tr, n_run,
                n_basis=ks[q], device=cpu,
            ).double()
            for q in range(n_cond)
        ]  # fmt: skip
        blocks.append(torch.cat(cols, dim=1))
    T = torch.cat(blocks, dim=0)  # (n_t, sum(ks)), condition-major
    X = torch.cat([T, _nuisance(lengths, poly_degree)], dim=1)
    P = torch.linalg.pinv(X)
    edges = np.cumsum([0, *ks])

    dt = default_microtime_dt(tr)
    bases = hrfs_from_spec(true_hrf, dt, cpu)[0][1]
    X_true = build_task_design(
        realization.onsets, realization.durations, tr, lengths, bases, dt, device=cpu
    ).double()
    amps = torch.as_tensor(np.asarray(amplitudes, dtype=float), dtype=torch.float64)
    # The truth on a fine grid: one event at 0, same HRF and duration convention.
    fine_dt = 0.1
    fdt = default_microtime_dt(fine_dt)
    fbases = hrfs_from_spec(true_hrf, fdt, cpu)[0][1]
    n_fine = [int(round(w / fine_dt)) + 1 for w in windows]
    truth = [
        float(amps[q])
        * build_task_design([[np.array([0.0])]], [realization.durations[q]], fine_dt,
                            [n_fine[q]], fbases, fdt, device=cpu)[:, 0].numpy()
        for q in range(n_cond)
    ]  # fmt: skip

    gen = torch.Generator().manual_seed(seed)
    out: dict[str, Any] = {
        "knots": [np.linspace(0.0, w, k) for w, k in zip(windows, ks, strict=True)],
        "fine_t": [np.arange(n) * fine_dt for n in n_fine], "truth": truth,
        "est": {}, "se": {},
    }  # fmt: skip
    for i, c in enumerate(noise):
        label = str(c.get("label", f"noise{i}"))
        kw = {key: v for key, v in c.items() if key != "label"}
        sd = 100.0 / float(kw["tsnr"])
        L = _noise_correlation(kw, tr, lengths, factor=True)
        z = torch.randn(X.shape[0], generator=gen, dtype=torch.float64)
        y = X_true @ amps + sd * (z if L is None else L @ z)
        beta = (P @ y).numpy()
        R = _noise_correlation(kw, tr, lengths)
        var = ((P * P).sum(1) if R is None else ((P @ R) * P).sum(1)).numpy()
        out["est"][label] = [beta[edges[q] : edges[q + 1]] for q in range(n_cond)]
        out["se"][label] = [sd * np.sqrt(var[edges[q] : edges[q + 1]]) for q in range(n_cond)]
    return out


def single_trial_example(
    realization: Any,
    tr: float,
    noise: dict[str, Any],
    mean_response: float = 1.0,
    trial_sd: float = 0.5,
    hrf: str = "spmg1",
    poly_degree: int | None = None,
    seed: int = 0,
    sd_grid: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0),
) -> dict[str, Any]:
    """One voxel's single trials, estimated three ways, beside the truth.

    Trial amplitudes ``mean_response`` + N(0, ``trial_sd``^2) at one noise
    condition; estimates by LSS, LSA and single-trial ridge (at the fraction
    :func:`single_trial_quality` finds best). Also the expected reliability of
    each over ``sd_grid`` -- how variable trials must be for the design to
    resolve them. Returns 'true', 'cond', 'lss', 'lsa', 'ridge' (per trial),
    'expected' {method: r at trial_sd}, 'ridge_frac', 'sd_grid', 'curve'
    {method: r per sd}.
    """
    from fastfuncstuff.cli_utils import auto_polort

    from .core import default_microtime_dt, hrfs_from_spec

    lengths = list(realization.run_lengths)
    if poly_degree is None:
        poly_degree = auto_polort(max(lengths) * tr)
    label = str(noise.get("label", "noise"))
    kw = {k: v for k, v in noise.items() if k != "label"}
    dt = default_microtime_dt(tr)
    bases = hrfs_from_spec(hrf, dt, torch.device("cpu"))[0][1]
    Xt, cond = trial_regressors(realization, tr, bases, dt)
    D = _nuisance(lengths, poly_degree)
    Qd, _ = torch.linalg.qr(D)
    Xp = Xt - Qd @ (Qd.T @ Xt)
    q = single_trial_quality(realization, tr, [noise], hrf, poly_degree, mean_response, trial_sd)
    rel = q["reliability"][label]

    gen = torch.Generator().manual_seed(seed)
    n_tr = Xt.shape[1]
    beta = mean_response + trial_sd * torch.randn(n_tr, generator=gen, dtype=torch.float64)
    sd = 100.0 / float(kw["tsnr"])
    L = _noise_correlation(kw, tr, lengths, factor=True)
    z = torch.randn(Xt.shape[0], generator=gen, dtype=torch.float64)
    y = Xt @ beta + sd * (z if L is None else L @ z)
    yp = y - Qd @ (Qd.T @ y)
    eye = torch.eye(n_tr, dtype=torch.float64)
    lsa = (
        torch.linalg.lstsq(Xp, yp[:, None]).solution[:, 0]
        if rel["lsa"] > 0
        else torch.full((n_tr,), float("nan"), dtype=torch.float64)
    )
    ridge = torch.linalg.solve(Xp.T @ Xp + rel["ridge_lambda"] * eye, Xp.T @ yp)
    curve: dict[str, list[float]] = {"lss": [], "lsa": [], "ridge": []}
    for t_sd in sd_grid:
        r_t = single_trial_quality(realization, tr, [noise], hrf, poly_degree, mean_response, t_sd)[
            "reliability"
        ][label]
        for m in curve:
            curve[m].append(r_t[m])
    return {
        "true": beta.numpy(),
        "cond": cond,
        "lss": (lss_rows(Xt, cond, D) @ y).numpy(),
        "lsa": lsa.numpy(),
        "ridge": ridge.numpy(),
        "expected": {m: rel[m] for m in ("lss", "lsa", "ridge")},
        "ridge_frac": rel["ridge_frac"],
        "sd_grid": list(sd_grid),
        "curve": curve,
        "noise": label,
    }


def t_example(
    realization: Any,
    tr: float,
    contrast: str,
    weights: list[float] | np.ndarray,
    noise: list[dict[str, Any]],
    effects: dict[str, float],
    beta_pattern: list[float] | np.ndarray | None = None,
    hrf: str = "spmg1",
    alpha: float = 0.001,
    poly_degree: int | None = None,
    n_reps: int = 2000,
    seed: int = 0,
    estimator: str = "ols",
    null_reps: int | None = None,
    reml_maxa: float = 0.8,
    reml_maxb: float = 0.8,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """t values of one contrast under the null and at an effect, per noise level.

    ``effects`` is the contrast's true effect (PSC) per noise label, e.g. the
    one each level needs for the target power. Returns 't' {label: {'null':
    (corrected, naive), 'effect': (corrected, naive)}}, 'crit' and 'dof'
    {label: (corrected, naive)} and 'effects' -- what the t-distribution
    figure draws.
    """
    from .core import build_task_design, default_microtime_dt, hrfs_from_spec

    cpu = torch.device("cpu")
    dt = default_microtime_dt(tr)
    bases = hrfs_from_spec(hrf, dt, cpu)[0][1]
    X = build_task_design(
        realization.onsets, realization.durations, tr, realization.run_lengths, bases, dt,
        device=cpu,
    )  # fmt: skip
    w = np.asarray(weights, dtype=float)
    n_cond = len(w)
    pattern = np.ones(n_cond) if beta_pattern is None else np.asarray(beta_pattern, float)
    if is_difference(w):  # the sweep is the difference itself, as in the engine
        pos = np.clip(w, 0, None)
        pattern = pos / float(pos @ pos)
    per_unit = float(w @ pattern)  # contrast value per unit of the sweep
    out: dict[str, Any] = {
        "t": {},
        "crit": {},
        "dof": {},
        "effects": effects,
        "estimator": estimator,
    }
    for k, cond in enumerate(noise):
        label = str(cond.get("label", f"noise{k}"))
        amp = abs(effects[label] / per_unit) if per_unit else 0.0
        res = simulate_design_power(
            X, list(realization.run_lengths), tr, {contrast: w}, [amp], [cond],
            beta_pattern=pattern, n_reps=n_reps, poly_degree=poly_degree, alpha=alpha,
            device=cpu if device is None else device, seed=seed + k, keep_t=True,
            estimator=estimator, null_reps=null_reps, reml_maxa=reml_maxa, reml_maxb=reml_maxb,
        )  # fmt: skip
        out["t"][label] = {
            "null": res["t"][(label, 0.0, contrast)],
            "effect": res["t"][(label, float(amp), contrast)],
        }
        out["crit"][label] = res["crit"][label]
        naive, corr = res["dof"][label]
        out["dof"][label] = (corr, naive)
    return out


def design_spectrum(
    realization: Any,
    tr: float,
    contrasts: dict[str, list[float] | np.ndarray],
    noise: dict[str, Any],
    hrf: str = "spmg1",
    poly_degree: int | None = None,
) -> dict[str, Any]:
    """Where each contrast's information sits in frequency, against the noise and the drift.

    For the first run: each contrast's regressor (weighted sum of the
    condition regressors, mean removed) and its power spectrum; the noise
    power spectrum from the ARMA autocorrelation (relative, peak 1); and the
    fraction of a sinusoid at each frequency the drift polynomials remove --
    by projection, exactly, rather than as a nominal cutoff. Efficiency is
    contrast power where the drift keeps it and the noise is quiet (Josephs &
    Henson 1999; Smith et al. 2007). Returns 'freq' (Hz), 'power' {contrast:
    relative}, 'noise_psd', 'removed', 'drift_share' {contrast: fraction of
    its power the drift removes}.
    """
    from fastfuncstuff.cli_utils import auto_polort

    from .core import build_task_design, default_microtime_dt, hrfs_from_spec

    cpu = torch.device("cpu")
    n = int(realization.run_lengths[0])
    if poly_degree is None:
        poly_degree = auto_polort(max(realization.run_lengths) * tr)
    dt = default_microtime_dt(tr)
    bases = hrfs_from_spec(hrf, dt, cpu)[0][1]
    X = build_task_design(
        realization.onsets, realization.durations, tr, realization.run_lengths, bases, dt,
        device=cpu,
    ).double()[:n].numpy()  # fmt: skip
    freq = np.fft.rfftfreq(n, d=tr)
    t = np.arange(n) * tr
    Q, _ = np.linalg.qr(_nuisance([n], poly_degree).numpy())
    removed = np.empty(len(freq))
    for k, f in enumerate(freq):
        wave = np.stack([np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t)], axis=1)
        e = float((wave * wave).sum())
        proj = Q @ (Q.T @ wave)
        removed[k] = float((proj * proj).sum()) / e if e > 1e-12 else 1.0
    power, share = {}, {}
    for name, w in contrasts.items():
        x = X @ np.asarray(w, dtype=float)
        x = x - x.mean()
        pw = np.abs(np.fft.rfft(x)) ** 2
        power[name] = pw / pw.max() if pw.max() > 0 else pw
        share[name] = float((pw * removed).sum() / pw.sum()) if pw.sum() > 0 else float("nan")
    kw = {key: v for key, v in noise.items() if key != "label"}
    R = _noise_correlation(kw, tr, [n])
    acf = np.zeros(n) if R is None else R[0].numpy()
    if R is None:
        acf[0] = 1.0
    sym = np.concatenate([acf, acf[-2:0:-1]])  # a symmetric autocorrelation's spectrum
    psd = np.interp(freq, np.fft.rfftfreq(len(sym), d=tr), np.abs(np.fft.rfft(sym)))
    return {
        "freq": freq,
        "power": power,
        "noise_psd": psd / psd.max(),
        "removed": removed,
        "drift_share": share,
        "noise_label": str(noise.get("label", "noise")),
    }


def shape_steps(
    realization: Any,
    tr: float,
    noise: list[dict[str, Any]],
    amplitudes: list[float] | np.ndarray,
    alpha: float = 0.001,
    target: float = 0.8,
    poly_degree: int | None = None,
    max_step: int = 12,
) -> dict[str, Any]:
    """How far apart two response shapes must be for this design to tell them apart.

    The 20-HRF library (GLMsingle's) is ordered: peak 2.7 -> 5.7 s, about
    0.16 s of latency per step, widening as it goes. Suppose a condition's
    true shape is library HRF k+s while the model gives every condition HRF k
    (amplitudes free). What that model cannot absorb -- the residual e of the
    true regressor on [X_k, drift] -- is what a shape test detects, with
    noncentrality A (e'e) / (sigma sqrt(e'Re)) under the noise ARMA. Power is
    averaged over library positions k and both directions, per step distance s.

    This is also the two-condition question: 'do A and B differ in shape?' is
    'is B's shape not A's HRF k?' -- the same residual. Model selection among
    library shapes (``ffs_hrfopt``, GLMsingle) needs exactly this separation.

    Returns 'steps' {label: (n_cond,)} -- the step distance at which power
    reaches ``target`` (fractional; nan beyond ``max_step``) -- and
    'power' {label: (n_cond, max_step)} per step distance 1..max_step.
    """
    from scipy import stats as st

    from fastfuncstuff.cli_utils import auto_polort

    from .core import build_task_design, default_microtime_dt, hrfs_from_spec

    cpu = torch.device("cpu")
    lengths = list(realization.run_lengths)
    if poly_degree is None:
        poly_degree = auto_polort(max(lengths) * tr)
    dt = default_microtime_dt(tr)
    lib = hrfs_from_spec("lib:all", dt, cpu)
    Xs = torch.stack(
        [
            build_task_design(realization.onsets, realization.durations, tr, lengths, b, dt,
                              device=cpu).double()
            for _, b in lib
        ]
    )  # (n_lib, n_t, n_cond)  # fmt: skip
    n_lib, n_t, n_cond = Xs.shape
    D = _nuisance(lengths, poly_degree)
    amps = np.abs(np.asarray(amplitudes, dtype=float))
    dof = n_t - n_cond - D.shape[1] - 1
    crit = float(st.t.ppf(1 - alpha / 2, dof))
    # Per noise ARMA: sum over k and both directions of e'e and e'Re per (q, s).
    by_ab: dict[Any, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    out: dict[str, Any] = {"steps": {}, "power": {}, "max_step": max_step}
    for i, c in enumerate(noise):
        label = str(c.get("label", f"noise{i}"))
        kw = {key: v for key, v in c.items() if key != "label"}
        ab = _noise_arma(kw, tr)
        if ab not in by_ab:
            R = None if ab is None else _block_correlation(tuple(lengths), *ab, False)
            ee = [[[] for _ in range(max_step)] for _ in range(n_cond)]
            eRe = [[[] for _ in range(max_step)] for _ in range(n_cond)]
            for k in range(n_lib):
                Q, _ = torch.linalg.qr(torch.cat([Xs[k], D], dim=1))
                for q in range(n_cond):
                    others = [k + s for s in range(-max_step, max_step + 1)
                              if s and 0 <= k + s < n_lib]  # fmt: skip
                    E = Xs[others, :, q].T  # (n_t, len(others))
                    E = E - Q @ (Q.T @ E)
                    RE = E if R is None else R @ E
                    for col, j in enumerate(others):
                        step = abs(j - k) - 1
                        ee[q][step].append(float(E[:, col] @ E[:, col]))
                        eRe[q][step].append(float(E[:, col] @ RE[:, col]))
            by_ab[ab] = (ee, eRe)
        ee, eRe = by_ab[ab]
        sd = 100.0 / float(kw["tsnr"])
        power = np.zeros((n_cond, max_step))
        for q in range(n_cond):
            for s in range(max_step):
                nc = amps[q] * np.array(ee[q][s]) / (sd * np.sqrt(np.array(eRe[q][s])))
                power[q, s] = float(np.mean([_two_tailed_power(crit, dof, v) for v in nc]))
        steps = np.full(n_cond, np.nan)
        for q in range(n_cond):
            hit = np.nonzero(power[q] >= target)[0]
            if hit.size:
                j = hit[0]
                steps[q] = (
                    1.0 if j == 0 else float(np.interp(target, power[q, j - 1 : j + 1], [j, j + 1]))
                )
        out["steps"][label], out["power"][label] = steps, power
    return out


def mean_soa(realization: Any) -> float:
    """Mean onset-to-onset interval (s) over all events, within runs."""
    gaps = []
    for run in range(len(realization.run_lengths)):
        on = np.sort(np.concatenate([np.asarray(c[run], dtype=float) for c in realization.onsets]))
        gaps += np.diff(on).tolist()
    return float(np.mean(gaps)) if gaps else float("nan")


SOA_FAMILIES = ("fixed SOA", "jittered", "jittered, 1/3 blank")


def soa_sweep(
    realization: Any,
    tr: float,
    scorer: Any,
    ref_noise: str,
    initial_fix: float = 0.0,
    post_fix: float = 16.0,
    n_soa: int = 10,
    n_designs: int = 2,
    seed: int = 0,
) -> dict[str, Any]:
    """Efficiency against SOA (Dale 1999; Friston 1999; Josephs & Henson 1999).

    The same conditions, durations, runs and run length as ``realization``,
    rearranged as three classic event designs at each mean SOA (onset to
    onset): a fixed SOA; jittered (exponential gaps, same mean); jittered with
    a third of the slots blank. Each is scored by ``scorer``
    (:class:`RealizationScorer`, shape included). Plus, for the Liu plane,
    blocks of each condition from 4 to 40 s with equal rest.

    Returns 'soa' (grid), 'this_soa', 'families' {name: list of per-SOA lists
    of scores}, 'blocks' {length: [scores]}.
    """
    from .experiment import ExperimentSpec, Interval, Unit, realize

    conds = list(realization.conditions)
    durs = [float(d) for d in realization.durations]
    d = float(np.mean(durs))
    n_runs = len(realization.run_lengths)
    run_s = float(np.mean(realization.run_durations))
    this = mean_soa(realization)
    hi = max(20.0, 1.5 * this if np.isfinite(this) else 20.0)
    soas = np.geomspace(d + max(0.25, 0.25 * tr), hi, n_soa)

    def spec(units, isi):
        return ExperimentSpec(tr=tr, units=units, isi=isi, n_runs=n_runs, initial_fix=initial_fix,
                              post_fix=post_fix, scan_time=run_s)  # fmt: skip

    def score(sp):
        out = []
        for r in range(n_designs):
            try:
                sc = scorer.score(realize(sp, seed + r))
            except ValueError:
                sc = None
            if sc is not None:
                out.append(sc)
        return out

    trials = [Unit.parse(c, f"{c}:{dd:g}", 1) for c, dd in zip(conds, durs, strict=True)]
    fams: dict[str, list[list[dict[str, Any]]]] = {f: [] for f in SOA_FAMILIES}
    for soa in soas:
        g = soa - d
        fixed = Interval.parse(round(g, 4))
        jit = Interval.parse(f"exp:{g:.4g},{0.25 * g:.4g},{4 * g:.4g}")
        blank = Unit.parse("null0", f"null:{d:g}", 0.5 * len(conds))
        fams["fixed SOA"].append(score(spec(trials, fixed)))
        fams["jittered"].append(score(spec(trials, jit)))
        fams["jittered, 1/3 blank"].append(score(spec([*trials, blank], jit)))
    blocks: dict[float, list[dict[str, Any]]] = {}
    for length in (4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 30.0, 40.0):
        units = [Unit.parse(c, f"{c}:{length:g}", 1, "block") for c in conds]
        blocks[length] = score(spec(units, Interval.parse(length)))
    return {"soa": soas, "this_soa": this, "families": fams, "blocks": blocks,
            "ref": ref_noise}  # fmt: skip


def hrf_robustness(
    realization: Any,
    tr: float,
    contrasts: dict[str, list[float] | np.ndarray],
    noise: dict[str, Any],
    beta_pattern: list[float] | np.ndarray | None = None,
    fit_hrf: str = "spmg1",
    shared: float = 0.0,
    alpha: float = 0.001,
    target: float = 0.8,
    poly_degree: int | None = None,
    responses: list[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """What each contrast costs when the true HRF is each of the 20 library shapes.

    The model fits ``fit_hrf``; the data come from library HRF k. Analytic, as
    the engine's power: the expected estimate c P X_true beta (biased), and the
    residual variance inflated by what the model cannot absorb,
    sigma^2 + |M X_true beta|^2 / tr(MR). Both grow with the amplitude, so under
    a mismatch power has a *ceiling* -- some contrasts are never detected, at
    any amplitude. Differences sweep the difference on ``shared`` underneath,
    as the engine does; with ``responses`` each contrast is swept with every
    other condition at its given response (:func:`response_sweep`).

    Returns 'labels', 'peaks' (s), and per contrast: 'needed' (true effect,
    PSC, for ``target`` power; inf past the ceiling), 'recovered' (estimate /
    truth), 'ceiling' (the highest power any amplitude gives), 'fitted'
    (the effect needed when the truth *is* ``fit_hrf``), and 'matched' (the
    effect needed when the truth is library HRF k *and* the model fits HRF k:
    no mismatch, only what that shape does to this design -- a fast HRF
    passes more of a rapid design's high frequencies than a slow one).
    """
    from fastfuncstuff.cli_utils import auto_polort

    from .core import build_task_design, default_microtime_dt, hrfs_from_spec

    cpu = torch.device("cpu")
    lengths = list(realization.run_lengths)
    if poly_degree is None:
        poly_degree = auto_polort(max(lengths) * tr)
    dt = default_microtime_dt(tr)

    def design(spec: str) -> torch.Tensor:
        b = hrfs_from_spec(spec, dt, cpu)[0][1]
        return build_task_design(
            realization.onsets, realization.durations, tr, lengths, b, dt, device=cpu
        ).double()

    Xf = design(fit_hrf)
    n_t, n_cond = Xf.shape
    nuisance = _nuisance(lengths, poly_degree)
    kw = {key: v for key, v in noise.items() if key != "label"}
    R = _noise_correlation(kw, tr, lengths)

    def fit(X_task: torch.Tensor) -> tuple[Any, ...]:
        X = torch.cat([X_task, nuisance], dim=1)
        P = torch.linalg.inv(X.T @ X) @ X.T
        if R is None:
            PRPt, tr_MR, dof = P @ P.T, float(n_t - X.shape[1]), float(n_t - X.shape[1])
        else:
            PRPt, tr_MR, dof = _corrected_terms(X, P, R)
        return X, P, PRPt, tr_MR, dof, float(stats.t.ppf(1 - alpha / 2, dof))

    fitted = fit(Xf)
    sd2 = (100.0 / float(kw["tsnr"])) ** 2
    pattern = np.ones(n_cond) if beta_pattern is None else np.asarray(beta_pattern, float)
    amps = np.geomspace(1e-3, 1e3, 400)

    def cost(
        X_true: torch.Tensor, w: np.ndarray, terms: tuple[Any, ...] = fitted
    ) -> tuple[float, float, float]:
        X, P, PRPt, tr_MR, dof, crit = terms
        if responses is not None:
            p_, off = response_sweep(w, responses)
        elif is_difference(w):
            pos = np.clip(w, 0, None)
            p_, off = pos / float(pos @ pos), np.full(n_cond, shared)
        else:
            p_, off = pattern, np.zeros(n_cond)
        c = torch.zeros(X.shape[1], dtype=torch.float64)
        c[:n_cond] = torch.as_tensor(w)
        v = float(c @ PRPt @ c)
        su, s0 = X_true @ torch.as_tensor(p_), X_true @ torch.as_tensor(off)
        mu, m0 = su - X @ (P @ su), s0 - X @ (P @ s0)
        eu, e0 = float(c @ (P @ su)), float(c @ (P @ s0))
        per = float(w @ p_)  # the contrast's true value per unit amplitude
        if per == 0:
            return float("nan"), float("nan"), float("nan")
        ests = e0 + amps * eu
        mis = float(m0 @ m0) + 2 * amps * float(m0 @ mu) + amps**2 * float(mu @ mu)
        se = np.sqrt(v * (sd2 + mis / tr_MR))
        pw = np.asarray(_two_tailed_power(crit, dof, ests / se), dtype=float)
        below = np.nonzero(pw < target)[0]
        if below.size and below[-1] < len(amps) - 1:
            j = below[-1]
            needed = float(np.interp(target, pw[j : j + 2], amps[j : j + 2]) * abs(per))
        else:
            needed = float("inf")
        return needed, eu / per, float(pw.max())

    lib = hrfs_from_spec("lib:all", 0.1, cpu)
    labels = [lab for lab, _ in lib]
    peaks = [float(np.argmax(b[0].numpy())) * 0.1 for _, b in lib]
    out: dict[str, Any] = {"labels": labels, "peaks": peaks, "contrasts": {}}
    X_libs = [design(lab) for lab in labels]
    own = [fit(Xt) for Xt in X_libs]
    for name, w in contrasts.items():
        w = np.asarray(w, dtype=float)
        rows = [cost(Xt, w) for Xt in X_libs]
        out["contrasts"][name] = {
            "needed": [r[0] for r in rows],
            "recovered": [r[1] for r in rows],
            "ceiling": [r[2] for r in rows],
            "fitted": cost(Xf, w)[0],
            "matched": [cost(Xt, w, t)[0] for Xt, t in zip(X_libs, own, strict=True)],
        }
    return out


@dataclass(frozen=True)
class AutoSweep:
    """Place the swept amplitudes from the analytic power curve, per noise level.

    A fixed grid (0.1-3% in 15 steps) spent most of its points on the flat
    top at high tSNR -- tSNR 100 saturated by 0.6%, leaving 12 of 16 cells at
    power 1.0 -- and could stop short of the target at low tSNR. Instead the
    analytic curve (median over realizations) is evaluated on a fine log grid
    and ``n_points`` amplitudes are spaced evenly in *power travelled*, up to
    where every contrast passes ``ceiling``: dense on the rise, none on the
    plateau, and the range extends itself however hard the contrast is.
    ``extra`` amplitudes (e.g. -effect) join every level's grid.
    """

    n_points: int = 12
    ceiling: float = 0.99
    extra: tuple[float, ...] = ()
    low: float = 1e-3
    high: float = 100.0


def sweep_grid(fine: np.ndarray, power: np.ndarray, auto: AutoSweep) -> list[float]:
    """``auto.n_points`` amplitudes along ``power`` (G, K) evaluated at ``fine`` (G,).

    Spaced evenly in the summed absolute power change (the curve's arc length
    in power), so a non-monotone curve -- a wrong-HRF shared response that dips
    before it rises -- still gets points where it moves.
    """
    reached = np.nonzero(np.all(power >= auto.ceiling, axis=1))[0]
    end = int(reached[0]) if reached.size else len(fine) - 1
    step = np.abs(np.diff(power[: end + 1], axis=0)).max(axis=1) if end else np.zeros(0)
    travelled = np.concatenate([[0.0], np.cumsum(step)])
    if travelled[-1] <= 0:
        return sorted({*auto.extra, float(fine[end])})
    if not reached.size:
        # A curve that plateaus below the ceiling (a mismatched HRF): stop
        # where it has done ~all the moving it will do, not at the grid's edge.
        end = int(np.searchsorted(travelled, auto.ceiling * travelled[-1]))
        travelled = travelled[: end + 1]
    # Half-step offsets put the first point on the toe (~4% of the climb at
    # N=12) rather than a full step up it; the end point closes the curve.
    levels = travelled[-1] * np.append(
        (np.arange(auto.n_points - 1) + 0.5) / (auto.n_points - 1), 1.0
    )
    picked = np.interp(levels, travelled, fine[: end + 1])
    return sorted({*(float(f"{v:.3g}") for v in picked), *auto.extra})


def _auto_grids(
    designs: list[tuple[torch.Tensor, torch.Tensor | None, list[int]]],
    tr: float,
    group: dict[str, Any],
    noise: list[dict[str, Any]],
    pattern: np.ndarray,
    offset: np.ndarray,
    poly_degree: int | None,
    alpha: float,
    auto: AutoSweep,
) -> dict[str, list[float]]:
    """One amplitude grid per noise level for a contrast group, from the median analytic curve."""
    fine = np.geomspace(auto.low, auto.high, 400)
    curves: dict[str, list[np.ndarray]] = {}
    for X, X_true, lengths in designs:
        pw = analytic_power(
            X, lengths, tr, group, fine, noise, pattern, offset, poly_degree, alpha, X_true
        )
        for label, v in pw.items():
            curves.setdefault(label, []).append(v)
    return {
        label: sweep_grid(fine, np.median(np.stack(v), axis=0), auto) for label, v in curves.items()
    }


def neural_durations(durations: list[float], true_duration: str | float | None) -> list[float]:
    """Durations the data are generated with: the stimulus's, a fixed S, or "+S" longer."""
    if true_duration is None:
        return list(durations)
    text = str(true_duration).strip()
    value = float(text)
    if value < 0:
        raise ValueError(f"true duration {text!r} must be >= 0")
    if text.startswith("+"):
        return [d + value for d in durations]
    return [value] * len(durations)


def simulate_realizations_power(
    realizations: list[Any],
    tr: float,
    contrasts: dict[str, list[float] | np.ndarray],
    amplitudes: list[float] | np.ndarray | AutoSweep,
    noise: list[dict[str, Any]],
    beta_pattern: list[float] | np.ndarray | None = None,
    n_reps: int = 500,
    alpha: float = 0.001,
    true_delay: float = 0.0,
    poly_degree: int | None = None,
    device: torch.device | None = None,
    seed: int = 0,
    progress: bool = True,
    hrf: str = "spmg1",
    true_hrf: str = "same",
    true_duration: str | float | None = None,
    shared: float = 0.0,
    estimator: str = "ols",
    null_reps: int | None = None,
    reml_maxa: float = 0.8,
    reml_maxb: float = 0.8,
    reml_cache: dict | None = None,
    responses: list[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """:func:`simulate_design_power` over several realizations of one experiment.

    A jittered, shuffled design is a distribution over event lists, so its
    power is too: each realization (from :func:`~.experiment.realize`, or any
    object with ``onsets``, ``durations`` and ``run_lengths``) is simulated in
    turn and its rows carry a ``design`` index.

    ``hrf`` is the response the GLM fits; ``true_hrf`` generates the data
    (``same``, ``spmg1``, ``lib:K``, or ``lib:all`` to sweep the library as the
    truth), and rows carry a ``true_hrf`` label. ``true_delay`` additionally
    generates the response that many seconds late, and ``true_duration``
    with neural activity that long ("+S": S seconds longer than each
    condition's stimulus) while the model keeps the stimulus durations --
    activity that outlasts the stimulus, as in memory or decision tasks.
    Any mismatch shows up as bias in ``mean_est`` and lost power.

    What is swept depends on the contrast. A **condition contrast** (weights
    not summing to zero, e.g. ``A``) sweeps the response amplitude, every
    condition responding ``amplitude x beta_pattern``. A **difference
    contrast** (weights summing to zero, e.g. ``A-B``) sweeps the difference
    itself: every condition sits at ``shared`` percent and the positive side
    is raised so that the contrast equals the swept value (A = shared + d,
    B = shared). Under a correct HRF the shared level cancels exactly; under a
    mismatch it does not, which is the case ``shared`` exists to measure. Rows
    carry ``swept`` ("amplitude" or "difference") and ``shared``.

    ``responses`` (PSC per condition) replaces both: each contrast is swept on
    its own (:func:`response_sweep`, rows ``swept`` "contrast") with every
    other condition at its given response, so the curve passes through the
    specified experiment at c @ responses. Only a mismatch makes it differ
    from the plain sweep -- under the fitted HRF the other betas cancel.
    """
    from tqdm import tqdm

    from .core import build_task_design, default_microtime_dt, hrfs_from_spec

    cpu = torch.device("cpu")
    dt = default_microtime_dt(tr)
    fit_label, fit_bases = hrfs_from_spec(hrf, dt, cpu)[0]
    truths = [(fit_label, fit_bases)] if true_hrf == "same" else hrfs_from_spec(true_hrf, dt, cpu)

    n_cond = len(realizations[0].conditions)
    pattern = np.ones(n_cond) if beta_pattern is None else np.asarray(beta_pattern, float)
    # One simulation for all condition contrasts, one per difference contrast
    # (each plants its own responses).
    groups: list[tuple[str, dict[str, Any], np.ndarray, np.ndarray]] = []
    condition = {k: w for k, w in contrasts.items() if abs(float(np.sum(w))) > 1e-9}
    if responses is not None:
        for k, w in contrasts.items():
            groups.append(("contrast", {k: w}, *response_sweep(w, responses)))
        condition = dict(contrasts)  # every contrast is grouped already
    elif condition:
        groups.append(("amplitude", condition, pattern, np.zeros(n_cond)))
    for k, w in contrasts.items():
        if k in condition:
            continue
        w = np.asarray(w, dtype=float)
        pos = np.clip(w, 0, None)
        groups.append(("difference", {k: w}, pos / float(pos @ pos), np.full(n_cond, shared)))

    rows: list[dict[str, Any]] = []
    per_design = []
    jobs = [(i, real, t) for i, real in enumerate(realizations) for t in range(len(truths))]
    reml_cache = {} if reml_cache is None else reml_cache

    def matrices(real: Any, ti: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        X = build_task_design(
            real.onsets, real.durations, tr, real.run_lengths, fit_bases, dt, device=cpu
        )
        t_label, t_bases = truths[ti]
        X_true = None
        if true_delay or t_label != fit_label or true_duration is not None:
            X_true = build_task_design(
                real.onsets, neural_durations(real.durations, true_duration), tr,
                real.run_lengths, t_bases, dt, delay=true_delay, device=cpu,
            )  # fmt: skip
        return X, X_true

    built = [matrices(real, ti) for _, real, ti in jobs]
    sweeps: list[Any] = [amplitudes] * len(groups)
    if isinstance(amplitudes, AutoSweep):
        designs = [
            (X, Xt, list(real.run_lengths))
            for (X, Xt), (_, real, _) in zip(built, jobs, strict=True)
        ]
        sweeps = [
            _auto_grids(
                designs, tr, group, noise, g_pattern, g_offset, poly_degree, alpha, amplitudes
            )  # fmt: skip
            for _, group, g_pattern, g_offset in groups
        ]
    for (i, real, ti), (X, X_true) in tqdm(
        list(zip(jobs, built, strict=True)),
        desc="designs",
        leave=True,
        disable=not progress or len(jobs) < 2,
    ):
        t_label = truths[ti][0]
        for g, (swept, group, g_pattern, g_offset) in enumerate(groups):
            res = simulate_design_power(
                X,
                list(real.run_lengths),
                tr,
                group,
                sweeps[g],
                noise,
                beta_pattern=g_pattern,
                beta_offset=g_offset,
                n_reps=n_reps,
                poly_degree=poly_degree,
                alpha=alpha,
                true_design=X_true,
                device=device,
                seed=seed + 7919 * i + 104729 * ti + 15485863 * g,
                estimator=estimator,
                null_reps=null_reps,
                reml_maxa=reml_maxa,
                reml_maxb=reml_maxb,
                reml_cache=reml_cache,
            )
            for r in res["table"]:
                r["design"] = i
                r["true_hrf"] = t_label
                r["swept"] = swept
                r["shared"] = float(g_offset[0]) if swept == "difference" else 0.0
            rows += res["table"]
        if ti == 0:
            per_design.append(
                {
                    "design": i,
                    "seed": getattr(real, "seed", i),
                    "run_lengths": list(real.run_lengths),
                    "dof": res["dof"],
                    "poly_degree": res["poly_degree"],
                    "X": X,
                }
            )
    # Keep the caller's order of noise levels and contrasts (not alphabetical:
    # "tSNR 100" would sort before "tSNR 40").
    c_order = {k: j for j, k in enumerate(contrasts)}
    n_order = {str(n.get("label", f"noise{j}")): j for j, n in enumerate(noise)}
    rows.sort(
        key=lambda r: (
            r["design"],
            r["true_hrf"],
            n_order.get(r["noise"], 0),
            r["amplitude"],
            c_order[r["contrast"]],
        )
    )
    return {
        "table": rows,
        "alpha": alpha,
        "n_reps": n_reps,
        "designs": per_design,
        "hrf": fit_label,
        "true_hrfs": [t for t, _ in truths],
        "shared": shared,
        "estimator": estimator,
    }


_NUMERIC = {
    "power_ols",
    "mean_t_ols",
    "null_rate",
    "null_reps",
    "null_p",
    "null_ci_low",
    "null_ci_high",
    "generating_a",
    "reml_maxa",
    "shared",
    "design",
    "tsnr",
    "amplitude",
    "true_effect",
    "mean_est",
    "expected_est",
    "sd_est",
    "sd_predicted",
    "mean_t",
    "power",
    "power_predicted",
    "mean_t_naive",
    "power_naive",
}


def load_power_table(path: str | Path) -> dict[str, Any]:
    """Read an ffs_simulate ``_power.tsv`` back into the result form the figures take.

    Also picks up the sibling ``_spec.json`` when present (for scan time and
    the contrasts' weights), under ``"spec"``.
    """
    import csv
    import json

    path = Path(path)
    with open(path, newline="") as f:
        rows = []
        for raw in csv.DictReader(f, delimiter="\t"):
            row: dict[str, Any] = dict(raw)
            for key in _NUMERIC & row.keys():
                if row[key] == "":
                    del row[key]
                else:
                    row[key] = int(row[key]) if key == "design" else float(row[key])
            row.setdefault("true_hrf", "")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no rows")
    stem = path.name.removesuffix(".tsv").removesuffix("_power")
    spec_path = path.with_name(f"{stem}_spec.json")
    spec = json.loads(spec_path.read_text()) if spec_path.exists() else None
    return {"table": rows, "spec": spec, "name": stem}


def scan_seconds(result: dict[str, Any]) -> float | None:
    """Total scan time per realization, from the spec, or None if it was not recorded."""
    spec = result.get("spec") or {}
    if spec.get("tr") is None or not spec.get("run_lengths"):
        return None
    return float(np.mean([sum(r) for r in spec["run_lengths"]])) * float(spec["tr"])


def compare_designs(
    results: dict[str, dict[str, Any]], target: float = 0.8
) -> list[dict[str, Any]]:
    """Amplitude each design needs for ``target`` power, summarised over its realizations.

    One row per design x noise level x contrast shared by every design:
    median, min and max over realizations (and true HRFs), and how many of
    them never reach ``target`` within the swept amplitudes.
    """
    names = list(results)
    shared_noise = [
        n
        for n in dict.fromkeys(r["noise"] for r in results[names[0]]["table"])
        if all(any(r["noise"] == n for r in res["table"]) for res in results.values())
    ]
    shared_con = [
        c
        for c in dict.fromkeys(r["contrast"] for r in results[names[0]]["table"])
        if all(any(r["contrast"] == c for r in res["table"]) for res in results.values())
    ]
    out = []
    for name in names:
        rows = results[name]["table"]
        need = effect_needed(results[name], target)
        scan = scan_seconds(results[name])
        for noise in shared_noise:
            for c in shared_con:
                v = need[(noise, c)]
                eff = max(abs(r["true_effect"]) for r in rows if r["contrast"] == c)
                out.append(
                    {
                        "design": name,
                        "noise": noise,
                        "contrast": c,
                        "n_realizations": len(v),
                        "has_effect": eff > 0,
                        "median": float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan,
                        "min": float(np.nanmin(v)) if np.isfinite(v).any() else np.nan,
                        "max": float(np.nanmax(v)) if np.isfinite(v).any() else np.nan,
                        "n_unreached": int(np.isnan(v).sum()),
                        "scan_s": scan,
                        # effect x sqrt(total minutes): per unit of scan time,
                        # since for a fixed design the effect falls as 1/sqrt(T)
                        "per_minute": (
                            float(np.nanmedian(v)) * np.sqrt(scan / 60)
                            if scan is not None and np.isfinite(v).any()
                            else np.nan
                        ),
                    }
                )
    return out


def response_sweep(weights, responses) -> tuple[np.ndarray, np.ndarray]:
    """(pattern, offset) that sweep one contrast with every condition at ``responses``.

    The swept value is the contrast's own: betas = offset + v * pattern with
    c @ pattern = 1, and at v = c @ responses the betas are ``responses``
    exactly. The raised side is the positive weights (A in A-B, so B stays at
    its response), as for a difference without -responses; conditions
    outside the contrast never move.
    """
    w = np.asarray(weights, dtype=float)
    r = np.asarray(responses, dtype=float)
    pos = np.clip(w, 0, None)
    pattern = pos / float(pos @ pos) if pos.any() else w / float(w @ w)
    return pattern, r - float(w @ r) * pattern


def is_difference(weights) -> bool:
    """A zero-sum contrast (A-B, A+B-2C): its sweep is the difference itself."""
    return abs(float(np.sum(np.asarray(weights, dtype=float)))) <= 1e-9


def has_true_effect(weights, beta_pattern) -> bool:
    """Whether the sweep gives this contrast a true effect.

    A difference contrast always has one -- the swept value is the difference.
    A condition contrast has one when the response pattern gives it one
    (``-pattern B=0`` leaves ``B`` with nothing to detect).
    """
    w = np.asarray(weights, dtype=float)
    return is_difference(w) or abs(float(w @ np.asarray(beta_pattern, dtype=float))) > 0
