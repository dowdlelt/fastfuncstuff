"""Validate design power through the production REML estimator and independent nulls."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from scipy import stats
from tqdm.auto import tqdm


def validate_reml_power(
    result: dict[str, Any],
    design: torch.Tensor,
    true_design: torch.Tensor,
    contrasts: torch.Tensor,
    pattern: torch.Tensor,
    offset: torch.Tensor,
    run_lengths: list[int],
    tr: float,
    noise: list[dict[str, Any]],
    device: torch.device,
    seed: int,
    null_reps: int | None = None,
    maxa: float = 0.8,
    maxb: float = 0.8,
    shared_cache: dict | None = None,
) -> None:
    """Replace OLS Monte Carlo results with fitted REML results, preserving the OLS reference.

    An independent null sample tests the *nominal* REML threshold. Recommendations
    use ``power_validated``: NaN if nulls detect inflation (one-sided exact binomial
    p < .01), or there are fewer than five expected null exceedances. This does not
    recalibrate the threshold or turn a liberal analysis into apparent sensitivity.
    """
    from fastfuncstuff.glm.arma import (
        calculate_grid_memory_footprint,
        estimate_valid_grid_pairs,
        fit_glm_arma11,
        get_default_arma_grids,
        precompute_autocorr_grid,
    )
    from fastfuncstuff.memory import estimate_chunk_size, get_available_memory, get_memory_config

    from .power import _noise_arma, _run_correlation

    alpha, n_reps = float(result["alpha"]), int(result["n_reps"])
    n_null = max(n_reps, math.ceil(10 / alpha)) if null_reps is None else null_reps
    if n_null < 1:
        raise ValueError("null_reps must be positive")
    if not 0 <= maxa < 1 or not 0 <= maxb < 1:
        raise ValueError("REML search bounds must be >= 0 and strictly below 1")
    n_t, n_p = design.shape
    names = list(dict.fromkeys(r["contrast"] for r in result["table"]))
    a_grid, b_grid = get_default_arma_grids(device)
    if maxa != 0.8:
        a_grid = torch.tensor(
            sorted({*np.arange(0, maxa, 0.1), maxa}), device=device, dtype=a_grid.dtype
        )
    if maxb != 0.8:
        b_grid = torch.tensor(
            sorted({*np.arange(-maxb, maxb, 0.1), 0.0, maxb}), device=device, dtype=b_grid.dtype
        )
    shared_cache = {} if shared_cache is None else shared_cache
    signature = (tuple(run_lengths), maxa, maxb, device.type, device.index)
    if shared_cache.get("signature") != signature:
        grid_bytes = calculate_grid_memory_footprint(
            estimate_valid_grid_pairs(a_grid, b_grid), n_t, n_p
        )
        budget = get_available_memory(torch.device("cpu")) * get_memory_config().cpu_safety_factor
        # R, L and L-inverse coexist during construction, beside the fitted grid.
        shared_cache["autocorr"] = (
            precompute_autocorr_grid(
                n_t,
                a_grid,
                b_grid,
                device,
                dtype=torch.float32,
                run_starts=np.cumsum([0, *run_lengths[:-1]]).tolist(),
            )
            if 4 * grid_bytes < budget
            else None
        )
        shared_cache["signature"] = signature
    cache = shared_cache["autocorr"]
    fit_kw = {
        "device": device,
        "a_grid": a_grid,
        "b_grid": b_grid,
        "verbose": False,
        "use_qr": True,
        "glt_labels": names,
        "glt_matrices": [c.numpy()[None, :] for c in contrasts],
        "run_starts": np.cumsum([0, *run_lengths[:-1]]).tolist(),
        "autocorr_cache": cache,
        "use_grid_batching": True if cache is None else None,
    }
    chunk = estimate_chunk_size(n_null + n_reps, n_t, n_p, device, operation="arma")
    X = design.to(device=device, dtype=torch.float32)
    signal_base = true_design @ offset
    signal_unit = true_design @ pattern
    matched = torch.allclose(true_design, design[:, : true_design.shape[1]], atol=1e-8, rtol=1e-6)
    unit_fits: dict[Any, tuple[torch.Tensor, torch.Tensor, int]] = {}

    def fit_draws(signal, count, draw_seed, sigma, ab):
        gen = torch.Generator(device=device).manual_seed(draw_seed)
        factors = (
            []
            if ab is None
            else [
                _run_correlation(int(n), *ab)[1].to(device=device, dtype=torch.float32)
                for n in run_lengths
            ]
        )
        estimates, ts = [], []
        s = signal.to(device=device, dtype=torch.float32)
        for start in tqdm(
            range(0, count, chunk), desc="REML replicates", leave=True, disable=count <= chunk
        ):
            y = torch.randn(min(chunk, count - start), n_t, device=device, generator=gen)
            if factors:
                at = 0
                for n, L in zip(run_lengths, factors, strict=True):
                    y[:, at : at + n] = y[:, at : at + n] @ L.T
                    at += n
            y = sigma * y + s
            fitted = fit_glm_arma11(y, X, tr, **fit_kw)
            assert fitted.contrast_betas is not None and fitted.contrast_tstats is not None
            estimates.append(fitted.contrast_betas.double().cpu())
            ts.append(fitted.contrast_tstats.double().cpu())
        return torch.cat(estimates), torch.cat(ts), fitted.dof

    for ni, cond in enumerate(noise):
        label = str(cond.get("label", f"noise{ni}"))
        ab = _noise_arma(cond, tr)
        sigma = 100.0 / float(cond["tsnr"])
        if matched:
            if ab not in unit_fits:
                unit_fits[ab] = fit_draws(
                    torch.zeros(n_t), n_null + n_reps, seed + 104729 * ni, 1.0, ab
                )
            beta, t0, dof = unit_fits[ab]
            se = (beta / t0).abs()
            null_t = t0[:n_null]
            sample_beta, sample_se = beta[n_null:] * sigma, se[n_null:] * sigma
        else:
            _, null_t, dof = fit_draws(signal_base, n_null, seed + 104729 * ni, sigma, ab)
        crit = float(stats.t.ppf(1 - alpha / 2, dof))
        null_counts = (null_t.abs() > crit).sum(dim=0).tolist()
        checks = [stats.binomtest(k, n_null, alpha, alternative="greater") for k in null_counts]
        cis = [stats.binomtest(k, n_null).proportion_ci(confidence_level=0.99) for k in null_counts]
        rows = [r for r in result["table"] if r["noise"] == label]
        by_amp = sorted({r["amplitude"] for r in rows})
        for amp in tqdm(
            by_amp, desc="REML amplitudes", leave=True, disable=matched or len(by_amp) < 2
        ):
            if matched:
                b = sample_beta + contrasts[:, : pattern.numel()] @ (offset + amp * pattern)
                ts = b / sample_se
            else:
                b, ts, _ = fit_draws(
                    signal_base + amp * signal_unit, n_reps, seed + 104729 * ni + 7919, sigma, ab
                )
            for ci, name in enumerate(names):
                row = next(r for r in rows if r["contrast"] == name and r["amplitude"] == amp)
                power = float((ts[:, ci].abs() > crit).double().mean())
                status = (
                    "inflated"
                    if checks[ci].pvalue < 0.01
                    else ("limited" if n_null * alpha < 5 else "checked")
                )
                row.update(
                    estimator="reml",
                    power_ols=row["power"],
                    mean_t_ols=row["mean_t"],
                    power=power,
                    power_validated=power if status == "checked" else float("nan"),
                    mean_est=float(b[:, ci].mean()),
                    sd_est=float(b[:, ci].std()),
                    mean_t=float(ts[:, ci].mean()),
                    null_rate=null_counts[ci] / n_null,
                    null_reps=n_null,
                    null_p=float(checks[ci].pvalue),
                    null_ci_low=float(cis[ci].low),
                    null_ci_high=float(cis[ci].high),
                    calibration=status,
                    generating_a=0.0 if ab is None else ab[0],
                    reml_maxa=maxa,
                )
                if "t" in result:
                    key = (label, amp, name)
                    result["t"][key] = (ts[:, ci].numpy(), result["t"][key][1])
                    result["crit"][label] = (crit, result["crit"][label][1])
        result["dof"][label] = (result["dof"][label][0], float(dof))
    result["estimator"], result["null_reps"] = "reml", n_null
