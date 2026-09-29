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

from typing import Any

import numpy as np
import torch
from scipy import stats

from .noise import generate_thermal_physio_noise, ou_to_arma11


def _two_tailed_power(crit: float, dof: float, nc: float) -> float:
    """P(|T| > crit) for noncentral t, by symmetry on |nc|.

    scipy's nct.cdf returns nan far in the tail (nc = 16 at 150 dof), where the
    wrong-sign tail it would contribute is < 1e-50 anyway.
    """
    nc = abs(nc)
    far = float(stats.nct.cdf(-crit, dof, nc))
    return float(stats.nct.sf(crit, dof, nc)) + (0.0 if np.isnan(far) else far)


def _nuisance(run_lengths: list[int], poly_degree: int) -> torch.Tensor:
    from fastfuncstuff.glm.core import construct_polynomial_matrix

    cpu = torch.device("cpu")
    blocks = [construct_polynomial_matrix(n, poly_degree, cpu, torch.float64) for n in run_lengths]
    return torch.block_diag(*blocks)


def _noise_correlation(
    noise: dict[str, Any], tr: float, run_lengths: list[int]
) -> torch.Tensor | None:
    """Block-diagonal ARMA(1,1) correlation of a noise condition, or None if white."""
    from fastfuncstuff.glm.arma import build_arma11_covariance

    if noise.get("arma") is not None:
        a, b = (float(v) for v in noise["arma"])
    else:
        f = float(noise.get("phys_fraction", 0.5))
        if f == 0.0:
            return None
        a, b = (float(v) for v in ou_to_arma11(tr, float(noise.get("tau", 6.0)), f))
    if a == 0.0 and b == 0.0:
        return None
    n = sum(run_lengths)
    starts = np.concatenate([[0], np.cumsum(run_lengths)[:-1]]).astype(int).tolist()
    R = build_arma11_covariance(a, b, n, torch.device("cpu"), torch.float64, run_starts=starts)
    if R is None:
        raise ValueError(f"noise ARMA a={a:.3f}, b={b:.3f} is not a valid correlation")
    return R


def simulate_design_power(
    design: torch.Tensor | np.ndarray,
    run_lengths: list[int],
    tr: float,
    contrasts: dict[str, list[float] | np.ndarray],
    amplitudes: list[float] | np.ndarray,
    noise: list[dict[str, Any]],
    beta_pattern: list[float] | np.ndarray | None = None,
    n_reps: int = 500,
    poly_degree: int | None = None,
    alpha: float = 0.001,
    true_design: torch.Tensor | np.ndarray | None = None,
    baseline: float = 100.0,
    device: torch.device | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Monte-Carlo detection power of each contrast, per noise condition and amplitude.

    Parameters
    ----------
    design : (n_timepoints, n_conditions) task regressors, unit-peak (e.g. from
        ``simulate_bold(...)['design']`` or build_event_design_microtime), all runs
        concatenated. This is the model that is *fitted*.
    run_lengths : timepoints per run; each run gets its own Legendre drift block
    tr : seconds
    contrasts : name -> weights over the conditions
    amplitudes : peak percent signal change swept; 0 is always added (the null)
    noise : noise conditions, each a dict of generate_thermal_physio_noise
        keywords (``tsnr`` plus ``phys_fraction``/``tau`` or ``arma``), optionally
        with a ``label``. ``TsnrBin.simulation_kwargs()`` returns exactly this.
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
            sd_est, sd_predicted, mean_t, power, power_predicted, mean_t_naive,
            power_naive
        'alpha', 'poly_degree', 'dof' ({noise label: (naive, corrected)}),
        'n_reps'
    """
    from fastfuncstuff.cli_utils import auto_polort
    from fastfuncstuff.utils import get_device

    if device is None:
        device = get_device()
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
    if poly_degree is None:
        poly_degree = auto_polort(max(run_lengths) * tr)

    X = torch.cat([X_task, _nuisance(run_lengths, poly_degree)], dim=1)
    n_p = X.shape[1]
    if torch.linalg.matrix_rank(X) < n_p:
        raise ValueError("design + drift polynomials are rank-deficient; check the conditions")
    XtX_inv = torch.linalg.inv(X.T @ X)
    P = XtX_inv @ X.T  # (n_p, n_t)
    M = torch.eye(n_t, dtype=torch.float64) - X @ P

    names = list(contrasts)
    C = torch.zeros(len(names), n_p, dtype=torch.float64)
    for i, name in enumerate(names):
        w = torch.as_tensor(np.asarray(contrasts[name], dtype=np.float64))
        if w.numel() != n_cond:
            raise ValueError(f"contrast {name!r} has {w.numel()} weights, design {n_cond}")
        C[i, :n_cond] = w

    amps = sorted({0.0, *(float(a) for a in amplitudes)})
    P_dev = P.to(device=device, dtype=torch.float32)
    X_dev = X.to(device=device, dtype=torch.float32)
    CP_dev = (C @ P).to(device=device, dtype=torch.float32)  # contrast estimates directly
    scale = baseline / 100.0
    gen = torch.Generator(device="cpu").manual_seed(seed)

    rows: list[dict[str, Any]] = []
    dofs: dict[str, tuple[float, float]] = {}
    for k, cond in enumerate(noise):
        label = str(cond.get("label", f"noise{k}"))
        kw = {key: v for key, v in cond.items() if key != "label"}
        sigma = baseline / float(kw["tsnr"])

        R = _noise_correlation(kw, tr, run_lengths)
        v_naive = torch.einsum("ip,pq,iq->i", C, XtX_inv, C)
        if R is None:
            v_true, tr_MR, dof_corr = v_naive, float(n_t - n_p), float(n_t - n_p)
        else:
            v_true = torch.einsum("ip,pq,iq->i", C, P @ R @ P.T, C)
            MR = M @ R
            tr_MR = float(torch.trace(MR))
            dof_corr = tr_MR**2 / float((MR * MR.T).sum())
        dof_naive = float(n_t - n_p)
        dofs[label] = (dof_naive, dof_corr)
        crit_naive = stats.t.ppf(1 - alpha / 2, dof_naive)
        crit_corr = stats.t.ppf(1 - alpha / 2, dof_corr)

        for amp in amps:
            signal = (scale * amp) * (X_true @ pattern)  # (n_t,)
            parts = []
            start = 0
            for n_run in run_lengths:
                parts.append(
                    generate_thermal_physio_noise(
                        n_run,
                        tr,
                        baseline=baseline,
                        n_voxels=n_reps,
                        device=torch.device("cpu"),
                        generator=gen,
                        **kw,
                    )
                )
                start += n_run
            Y = torch.cat(parts, dim=0).to(device) + signal.to(device, torch.float32)[:, None]

            est = CP_dev @ Y  # (n_contrasts, n_reps), data units
            resid = Y - X_dev @ (P_dev @ Y)
            rss = (resid * resid).sum(dim=0).double().cpu()
            est = est.double().cpu()

            se_naive = torch.sqrt(rss / dof_naive * v_naive[:, None])
            se_corr = torch.sqrt(rss / tr_MR * v_true[:, None])
            t_naive = est / se_naive
            t_corr = est / se_corr

            for i, name in enumerate(names):
                true_eff = float(C[i, :n_cond] @ pattern) * amp  # PSC
                sd_pred = sigma * float(torch.sqrt(v_true[i])) / scale  # PSC
                nc = true_eff / sd_pred if sd_pred > 0 else 0.0
                p_pred = _two_tailed_power(crit_corr, dof_corr, nc)
                rows.append(
                    {
                        "noise": label,
                        "tsnr": float(kw["tsnr"]),
                        "amplitude": amp,
                        "contrast": name,
                        "true_effect": true_eff,
                        "mean_est": float(est[i].mean()) / scale,
                        "sd_est": float(est[i].std()) / scale,
                        "sd_predicted": sd_pred,
                        "mean_t": float(t_corr[i].mean()),
                        "power": float((t_corr[i].abs() > crit_corr).double().mean()),
                        "power_predicted": p_pred,
                        "mean_t_naive": float(t_naive[i].mean()),
                        "power_naive": float((t_naive[i].abs() > crit_naive).double().mean()),
                    }
                )

    return {
        "table": rows,
        "alpha": alpha,
        "poly_degree": poly_degree,
        "dof": dofs,
        "n_reps": n_reps,
    }


def amplitude_for_power(
    result: dict[str, Any], target: float = 0.8, column: str = "power_predicted"
) -> dict[tuple[str, str], float]:
    """Smallest amplitude reaching ``target`` power, per (noise, contrast), by interpolation.

    nan where the swept amplitudes never reach it. ``column='power'`` uses the
    Monte-Carlo estimate instead of the analytic curve.
    """
    out: dict[tuple[str, str], float] = {}
    keys = {(r["noise"], r["contrast"]) for r in result["table"]}
    for key in keys:
        rows = sorted(
            (r for r in result["table"] if (r["noise"], r["contrast"]) == key),
            key=lambda r: r["amplitude"],
        )
        amps = np.array([r["amplitude"] for r in rows])
        pw = np.array([r[column] for r in rows])
        hit = np.nonzero(pw >= target)[0]
        if hit.size == 0 or rows[hit[0]]["true_effect"] == 0 and hit[0] == 0:
            out[key] = float("nan")
            continue
        j = hit[0]
        if j == 0:
            out[key] = float(amps[0])
        else:
            out[key] = float(np.interp(target, [pw[j - 1], pw[j]], [amps[j - 1], amps[j]]))
    return out


def simulate_realizations_power(
    realizations: list[Any],
    tr: float,
    contrasts: dict[str, list[float] | np.ndarray],
    amplitudes: list[float] | np.ndarray,
    noise: list[dict[str, Any]],
    beta_pattern: list[float] | np.ndarray | None = None,
    n_reps: int = 500,
    alpha: float = 0.001,
    true_delay: float = 0.0,
    poly_degree: int | None = None,
    device: torch.device | None = None,
    seed: int = 0,
    progress: bool = True,
) -> dict[str, Any]:
    """:func:`simulate_design_power` over several realizations of one experiment.

    A jittered, shuffled design is a distribution over event lists, so its
    power is too: each realization (from :func:`~.experiment.realize`, or any
    object with ``onsets``, ``durations`` and ``run_lengths``) is simulated in
    turn and its rows carry a ``design`` index. ``true_delay`` generates the
    response that many seconds late while fitting the nominal onsets.
    """
    from tqdm import tqdm

    from .core import build_task_design

    rows: list[dict[str, Any]] = []
    per_design = []
    cpu = torch.device("cpu")
    for i, real in enumerate(
        tqdm(
            realizations, desc="designs", leave=True, disable=not progress or len(realizations) < 2
        )
    ):
        X = build_task_design(real.onsets, real.durations, tr, real.run_lengths, device=cpu)
        X_true = None
        if true_delay:
            X_true = build_task_design(
                real.onsets, real.durations, tr, real.run_lengths, delay=true_delay, device=cpu
            )
        res = simulate_design_power(
            X,
            list(real.run_lengths),
            tr,
            contrasts,
            amplitudes,
            noise,
            beta_pattern=beta_pattern,
            n_reps=n_reps,
            poly_degree=poly_degree,
            alpha=alpha,
            true_design=X_true,
            device=device,
            seed=seed + 7919 * i,
        )
        for r in res["table"]:
            r["design"] = i
        rows += res["table"]
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
    return {"table": rows, "alpha": alpha, "n_reps": n_reps, "designs": per_design}
