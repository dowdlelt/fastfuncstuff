"""Reusable ICA utilities for component selection, filtering, and interpretation."""
from __future__ import annotations

import numpy as np
import torch

from .denoise import estimate_noise_component_caps_per_run
from .design import convolve_hrf_microtime
from .design_builder import create_onset_matrix_microtime, parse_afni_timing_file, parse_durations
from .glm_core import construct_polynomial_matrix
from .hrf import get_spmg1_hrf
from .pca import PCA


def parse_num_comps_spec(spec: str) -> int | float | str:
    spec_norm = spec.strip().lower()
    if spec_norm in {"auto", "melodic", "hybrid", "current", "erank", "mp"}:
        return spec_norm
    try:
        if any(ch in spec_norm for ch in [".", "e"]):
            return float(spec_norm)
        return int(spec_norm)
    except ValueError as exc:
        raise ValueError(
            "Invalid -num_comps. Use int, float (0-1), or one of: "
            "auto|melodic|hybrid|current|erank|mp"
        ) from exc


def apply_polort_projection(
    data_vox_t: torch.Tensor,
    polort: int,
    device: torch.device,
) -> torch.Tensor:
    if polort is None or polort < 0:
        return data_vox_t
    poly = construct_polynomial_matrix(
        n_timepoints=data_vox_t.shape[1],
        max_degree=polort,
        device=device,
        dtype=data_vox_t.dtype,
    )
    if poly.shape[1] == 0:
        return data_vox_t
    q, _ = torch.linalg.qr(poly)
    return data_vox_t - (data_vox_t @ q) @ q.T


def apply_high_pass_fft(
    data_vox_t: torch.Tensor,
    tr: float,
    high_pass_hz: float,
) -> torch.Tensor:
    if high_pass_hz is None or high_pass_hz <= 0:
        return data_vox_t
    n_t = data_vox_t.shape[1]
    freqs = torch.fft.rfftfreq(n_t, d=tr).to(data_vox_t.device)
    spec = torch.fft.rfft(data_vox_t, dim=1)
    pass_mask = freqs >= float(high_pass_hz)
    spec = spec * pass_mask.unsqueeze(0)
    return torch.fft.irfft(spec, n=n_t, dim=1)


def effective_rank_from_spectrum(evals: np.ndarray) -> int:
    ev = np.clip(evals.astype(np.float64), 1e-12, None)
    p = ev / ev.sum()
    h = -(p * np.log(p)).sum()
    er = int(np.round(np.exp(h)))
    return max(1, min(er, len(evals)))


def mp_spikes_from_spectrum(evals: np.ndarray, n_samples: int, n_features: int) -> int:
    n_ev = len(evals)
    if n_ev < 5:
        return 0
    beta = float(n_features) / float(max(1, n_samples))
    if beta < 1.0:
        return 0
    tail = evals[max(1, int(0.5 * n_ev)) :]
    sigma2 = float(np.median(tail))
    lambda_plus = sigma2 * (1.0 + np.sqrt(beta)) ** 2
    return int((evals > lambda_plus).sum())


def melodic_evidence_proxy_k(
    evals: np.ndarray,
    n_samples: int,
    n_features: int,
    min_k: int,
    max_k: int,
) -> tuple[int, dict]:
    """MELODIC-style Bayesian dimensionality proxy via PPCA-like BIC objective."""
    ev = np.clip(evals.astype(np.float64), 1e-12, None)
    d = int(n_features)
    n = int(n_samples)
    n_ev = len(ev)

    max_k = min(max_k, n_ev - 1) if n_ev > 1 else 1
    min_k = max(1, min(min_k, max_k))

    k_grid = np.arange(min_k, max_k + 1)
    bic_vals = []
    ll_vals = []

    for k in k_grid:
        sigma2 = float(np.mean(ev[k:])) if k < n_ev else 1e-12
        sigma2 = max(sigma2, 1e-12)
        ll = -0.5 * n * (
            d * np.log(2.0 * np.pi)
            + np.sum(np.log(ev[:k]))
            + (d - k) * np.log(sigma2)
            + d
        )
        n_params = k * d - (k * (k - 1)) / 2 + k + 1
        bic = -2.0 * ll + n_params * np.log(max(2, n))
        bic_vals.append(float(bic))
        ll_vals.append(float(ll))

    best_idx = int(np.argmin(np.asarray(bic_vals)))
    best_k = int(k_grid[best_idx])
    return best_k, {
        "k_grid": k_grid.tolist(),
        "bic": bic_vals,
        "ll": ll_vals,
        "selected_k": best_k,
    }


def estimate_ica_component_count(
    data_vox_t: torch.Tensor,
    method: int | float | str,
    max_auto_components: int,
    auto_min_components: int,
    auto_var_threshold: float,
    use_mp_prior: bool,
    device: torch.device,
) -> tuple[int, dict, dict]:
    """Return selected component count and diagnostics for an ICA run."""
    x_t = data_vox_t.T
    n_samples, n_features = int(x_t.shape[0]), int(x_t.shape[1])
    rank_cap = max(1, min(n_samples, n_features, int(max_auto_components)))

    pca = PCA(n_components=rank_cap, device=device)
    pca.fit(x_t)
    ev = pca.explained_variance_.detach().cpu().numpy()
    evr = pca.explained_variance_ratio_.detach().cpu().numpy()
    n_ev = len(ev)

    diagnostics: dict = {
        "rank_cap": int(rank_cap),
        "n_eigs": int(n_ev),
        "scree_ratio": evr.tolist(),
    }

    if isinstance(method, int):
        k = int(max(1, min(method, n_ev)))
        return k, diagnostics, {"mode": "fixed_int", "requested": int(method)}

    if isinstance(method, float):
        if not (0 < method <= 1):
            raise ValueError(f"Float -num_comps must be in (0,1], got {method}")
        cum = np.cumsum(evr)
        k = int(np.searchsorted(cum, method) + 1)
        k = int(max(1, min(k, n_ev)))
        return k, diagnostics, {"mode": "pca_variance_fraction", "requested": float(method)}

    mode = str(method).lower()

    if mode in {"auto", "melodic"}:
        k, melodic_diag = melodic_evidence_proxy_k(
            evals=ev,
            n_samples=n_samples,
            n_features=n_features,
            min_k=auto_min_components,
            max_k=rank_cap,
        )
        return k, diagnostics, {"mode": "melodic_proxy", **melodic_diag}

    if mode in {"hybrid", "current"}:
        all_mask = torch.ones(data_vox_t.shape[0], dtype=torch.bool, device=data_vox_t.device)
        est = estimate_noise_component_caps_per_run(
            data=data_vox_t,
            run_starts=[0],
            noise_pool_mask=all_mask,
            max_components=rank_cap,
            nuisance_per_run=None,
            min_components=auto_min_components,
            variance_threshold=auto_var_threshold,
            use_mp_prior=use_mp_prior,
            device=device,
            verbose=False,
        )
        k = int(est.per_run_caps[0])
        return k, diagnostics, {
            "mode": "hybrid_current",
            "variance_cap": int(est.variance_caps[0]),
            "effective_rank": int(est.entropy_rank_caps[0]),
            "mp_cap": None if est.mp_caps[0] is None else int(est.mp_caps[0]),
            "mp_reason": est.mp_reasons[0],
        }

    if mode == "erank":
        k = effective_rank_from_spectrum(ev)
        return k, diagnostics, {"mode": "effective_rank", "effective_rank": int(k)}

    if mode == "mp":
        spikes = mp_spikes_from_spectrum(ev, n_samples=n_samples, n_features=n_features)
        if spikes <= 0:
            k = effective_rank_from_spectrum(ev)
            info = {
                "mode": "mp_fallback_erank",
                "mp_spikes": int(spikes),
                "effective_rank": int(k),
            }
        else:
            k = int(max(1, min(spikes, n_ev)))
            info = {"mode": "mp", "mp_spikes": int(spikes)}
        return k, diagnostics, info

    raise ValueError(f"Unsupported -num_comps mode: {mode}")


def fit_abs_mixture_em(values_abs: np.ndarray, n_iter: int = 80) -> tuple[float, float, float, float, np.ndarray]:
    """2-Gaussian EM on |x| for ICA map noise/signal mixture diagnostics."""
    x = values_abs.astype(np.float64)
    x = np.clip(x, 0.0, None)

    mu1 = np.percentile(x, 40)
    mu2 = np.percentile(x, 95)
    s1 = np.std(x) * 0.5 + 1e-6
    s2 = np.std(x) + 1e-6
    pi = 0.2

    for _ in range(n_iter):
        p1 = (1.0 - pi) * np.exp(-0.5 * ((x - mu1) / s1) ** 2) / (s1 + 1e-12)
        p2 = pi * np.exp(-0.5 * ((x - mu2) / s2) ** 2) / (s2 + 1e-12)
        den = np.clip(p1 + p2, 1e-18, None)
        r2 = p2 / den
        r1 = 1.0 - r2

        w1 = np.clip(r1.sum(), 1e-8, None)
        w2 = np.clip(r2.sum(), 1e-8, None)

        mu1 = float((r1 * x).sum() / w1)
        mu2 = float((r2 * x).sum() / w2)
        s1 = float(np.sqrt(((r1 * (x - mu1) ** 2).sum() / w1) + 1e-12))
        s2 = float(np.sqrt(((r2 * (x - mu2) ** 2).sum() / w2) + 1e-12))
        pi = float(np.clip(w2 / len(x), 1e-4, 1 - 1e-4))

    return mu1, s1, mu2, s2, r2


def mixture_zscores_signed(comp_map: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    vals = comp_map.astype(np.float64)
    abs_vals = np.abs(vals)
    mu1, s1, mu2, s2, p_signal = fit_abs_mixture_em(abs_vals)
    z_abs = np.maximum(0.0, (abs_vals - mu1) / max(s1, 1e-8))
    z_signed = np.sign(vals) * z_abs
    return z_signed.astype(np.float32), p_signal.astype(np.float32), {
        "mu_noise": float(mu1),
        "sigma_noise": float(s1),
        "mu_signal": float(mu2),
        "sigma_signal": float(s2),
        "mixing_signal": float(np.mean(p_signal)),
    }


def build_task_design_for_run(
    onsets_files: list[str],
    durations_arg: list[str],
    run_idx: int,
    n_timepoints: int,
    tr: float,
    microtime_dt: float,
    device: torch.device,
) -> tuple[torch.Tensor, list[str], list[float]]:
    all_onsets_full = [parse_afni_timing_file(fp) for fp in onsets_files]
    n_conds = len(all_onsets_full)
    labels = [f"cond{i+1}" for i in range(n_conds)]
    durations = parse_durations(durations_arg, n_conds, labels)

    onsets_this_run = []
    for cond_runs in all_onsets_full:
        if run_idx >= len(cond_runs):
            raise ValueError(
                f"Timing file has fewer runs than input data (missing run {run_idx + 1})"
            )
        onsets_this_run.append([cond_runs[run_idx]])

    onset_mt = create_onset_matrix_microtime(
        all_onsets=onsets_this_run,
        run_starts=[0],
        tr=tr,
        n_timepoints=n_timepoints,
        microtime_dt=microtime_dt,
        stim_durations=durations,
        device=device,
    )

    hrf = get_spmg1_hrf(microtime_dt=microtime_dt, device=device)
    design = convolve_hrf_microtime(
        onsets_microtime=onset_mt,
        hrf=hrf,
        n_timepoints=n_timepoints,
        tr=tr,
        microtime_dt=microtime_dt,
        run_starts=[0],
        device=device,
    )
    return design, labels, durations


def component_condition_correlations(
    mixing_tk: torch.Tensor,
    design_tc: torch.Tensor,
) -> np.ndarray:
    """Pearson correlation between ICA timecourses and condition regressors."""
    x = mixing_tk.detach().cpu().numpy().astype(np.float64)
    y = design_tc.detach().cpu().numpy().astype(np.float64)

    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)

    x_std = np.clip(x.std(axis=0, keepdims=True), 1e-8, None)
    y_std = np.clip(y.std(axis=0, keepdims=True), 1e-8, None)

    xz = x / x_std
    yz = y / y_std

    return (xz.T @ yz) / float(x.shape[0])
