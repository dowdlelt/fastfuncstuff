"""Reusable ICA utilities for component selection, filtering, and interpretation.

Key algorithms:
- Minka (2000) Laplace approximation for PPCA dimensionality estimation
- Gaussian-Gamma Mixture (GGM) model for ICA spatial map thresholding
  (following FSL MELODIC: Beckmann & Smith 2004)

References
----------
FSL MELODIC:
    Beckmann CF & Smith SM (2004). Probabilistic Independent Component Analysis
    for Functional Magnetic Resonance Imaging. IEEE Trans Med Imaging 23(2):137-152.
    https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/MELODIC

PPCA dimensionality estimation (Minka's method):
    Minka TP (2000). Automatic Choice of Dimensionality for PCA.
    NIPS 13:598-604. https://proceedings.neurips.cc/paper/2000/hash/7503cfacd12053d309b6bed5c89de212-Abstract.html

Marchenko-Pastur distribution:
    Marchenko VA & Pastur LA (1967). Distribution of eigenvalues for some
    random matrices. Mathematics of the USSR-Sbornik 1(4):457-483.
"""

from __future__ import annotations

from math import log

import numpy as np
import torch
from scipy.special import gammaln as scipy_gammaln
from tqdm.auto import tqdm

from fastfuncstuff._compile import safe_compile
from fastfuncstuff.decomposition.model_order import select_model_order
from fastfuncstuff.denoise.sequential import estimate_noise_component_caps_per_run
from fastfuncstuff.design.builder import (
    parse_afni_timing_file,
    parse_durations,
)
from fastfuncstuff.design.hrf import get_spmg1_hrf
from fastfuncstuff.glm.core import construct_polynomial_matrix
from fastfuncstuff.utils import to_tensor


def parse_num_comps_spec(spec: str) -> int | float | str:
    """Parse CLI component-count specification into normalized value.

    Parameters
    ----------
    spec : str
        User-provided component specification.

    Returns
    -------
    int or float or str
        Parsed numeric value, or normalized mode string.
    """
    spec_norm = spec.strip().lower()
    if spec_norm == "melodic":
        # Back-compat: the mode used to be named after the tool it was matching.
        spec_norm = "laplace"
    if spec_norm in {"auto", "laplace", "hybrid", "current", "erank", "mp"}:
        return spec_norm
    try:
        if any(ch in spec_norm for ch in [".", "e"]):
            return float(spec_norm)
        return int(spec_norm)
    except ValueError as exc:
        raise ValueError(
            "Invalid -num_comps. Use int, float (0-1), or one of: "
            "auto|laplace|hybrid|current|erank|mp"
        ) from exc


@torch.inference_mode()
def apply_polort_projection(
    data_vox_t: torch.Tensor,
    polort: int,
    device: torch.device,
    run_starts: list[int] | None = None,
) -> torch.Tensor:
    """Project out polynomial trends from voxel time series.

    Parameters
    ----------
    data_vox_t : Tensor, shape (n_vox, n_time)
        Voxel-by-time matrix.
    polort : int
        Maximum polynomial degree to remove.
    device : torch.device
        Device used for polynomial basis construction.
    run_starts : list of int, optional
        Start indices of each run in the concatenated timeseries.
        When provided, builds a block-diagonal polynomial basis so each
        run's polynomials only affect that run's timepoints (per CLAUDE.md).
        If None, treats the entire timeseries as a single run.

    Returns
    -------
    torch.Tensor
        Detrended voxel-by-time matrix.
    """
    if polort is None or polort < 0:
        return data_vox_t

    n_t = data_vox_t.shape[1]

    if run_starts is not None and len(run_starts) > 1:
        # Block-diagonal polynomial basis: each run gets its own polynomials,
        # zero-padded so they don't affect other runs' timepoints.
        # One allocation for the whole basis, then write each run's block into
        # its own column slice. The previous torch.cat-per-run rebuilt and
        # recopied the entire growing matrix on every iteration (O(n_runs²)
        # copies), which is the cost that bites on 20+ run concatenations.
        run_ends = list(run_starts[1:]) + [n_t]
        n_cols_per_run = polort + 1
        poly = torch.zeros(
            n_t, n_cols_per_run * len(run_starts), device=device, dtype=data_vox_t.dtype
        )
        for i, (rs, re) in enumerate(zip(run_starts, run_ends, strict=False)):
            run_poly = construct_polynomial_matrix(
                n_timepoints=re - rs,
                max_degree=polort,
                device=device,
                dtype=data_vox_t.dtype,
            )
            c0 = i * n_cols_per_run
            poly[rs:re, c0 : c0 + run_poly.shape[1]] = run_poly
    else:
        poly = construct_polynomial_matrix(
            n_timepoints=n_t,
            max_degree=polort,
            device=device,
            dtype=data_vox_t.dtype,
        )

    if poly.shape[1] == 0:
        return data_vox_t
    q, _ = torch.linalg.qr(poly)
    # Project out polynomials in-place via chunking to avoid allocating
    # a full (V, T) intermediate when V is very large.
    from fastfuncstuff.memory import estimate_chunk_size

    n_vox = data_vox_t.shape[0]
    chunk_size = estimate_chunk_size(
        n_voxels=n_vox,
        n_timepoints=data_vox_t.shape[1],
        n_regressors=q.shape[1],
        device=data_vox_t.device,
        operation="ica_varnorm",  # similar memory profile: (T,) per voxel
    )
    n_chunks = (n_vox + chunk_size - 1) // chunk_size
    for v0 in tqdm(
        range(0, n_vox, chunk_size),
        total=n_chunks,
        desc="  Polort project",
        leave=True,
        disable=n_chunks <= 1,
    ):
        v1 = min(v0 + chunk_size, n_vox)
        data_vox_t[v0:v1] -= (data_vox_t[v0:v1] @ q) @ q.T
    return data_vox_t


def apply_high_pass_fft(
    data_vox_t: torch.Tensor,
    tr: float,
    high_pass_hz: float,
    transition_width: float = 0.25,
    run_starts: list[int] | None = None,
) -> torch.Tensor:
    """Apply Fourier-based high-pass filter with smooth transition.

    Uses a raised-cosine transition band to avoid Gibbs ringing from
    a brick-wall cutoff.

    Parameters
    ----------
    data_vox_t : Tensor of shape (n_vox, n_time)
    tr : float
        Repetition time in seconds.
    high_pass_hz : float or None
        Cutoff frequency in Hz. If None or <= 0, no filtering.
    transition_width : float
        Width of transition band as fraction of cutoff frequency.
        0 = brick wall, 0.25 = smooth over 25% of cutoff.
    run_starts : list of int, optional
        Start indices of each run.  When provided, the filter is applied
        independently to each run segment to avoid spectral leakage
        across run boundaries.
    """
    if high_pass_hz is None or high_pass_hz <= 0:
        return data_vox_t

    # If multi-run, filter each run segment independently
    if run_starts is not None and len(run_starts) > 1:
        n_t = data_vox_t.shape[1]
        run_ends = list(run_starts[1:]) + [n_t]
        for rs, re in zip(run_starts, run_ends, strict=False):
            data_vox_t[:, rs:re] = _apply_high_pass_single(
                data_vox_t[:, rs:re],
                tr,
                high_pass_hz,
                transition_width,
            )
        return data_vox_t

    return _apply_high_pass_single(data_vox_t, tr, high_pass_hz, transition_width)


def _apply_high_pass_single(
    data_vox_t: torch.Tensor,
    tr: float,
    high_pass_hz: float,
    transition_width: float,
) -> torch.Tensor:
    """High-pass filter a single contiguous segment."""
    n_t = data_vox_t.shape[1]
    freqs = torch.fft.rfftfreq(n_t, d=tr).to(data_vox_t.device)
    spec = torch.fft.rfft(data_vox_t, dim=1)

    cutoff = float(high_pass_hz)
    tw = cutoff * transition_width
    if tw < 1e-10:
        # Brick wall
        filt = (freqs >= cutoff).float()
    else:
        # Raised-cosine transition: 0 below (cutoff - tw), 1 above cutoff
        low = cutoff - tw
        filt = torch.clamp((freqs - low) / tw, 0.0, 1.0)
        # Smooth with cosine shape
        filt = 0.5 * (1.0 - torch.cos(filt * torch.pi))

    # Always zero DC
    filt[0] = 0.0

    spec = spec * filt.unsqueeze(0)
    return torch.fft.irfft(spec, n=n_t, dim=1)


def find_constant_voxels(
    data_vox_t: np.ndarray | torch.Tensor,
    tol: float = 1e-6,
) -> np.ndarray:
    """(V,) bool — voxels whose timeseries is constant, non-finite, or all-zero.

    MELODIC drops these from the analysis mask outright (`meldata.cc`). Keeping
    them and merely zeroing them, as we used to, leaves an exact-zero delta
    spike in every IC map; the mixture model's noise Gaussian then collapses
    onto that spike, `pi_noise` falls under the 0.4 fallback threshold, and the
    3-Gaussian fallback labels most of the brain as signal. Measured on real
    unsmoothed data: 5.6% constant voxels turned mean P(signal) into 0.70
    against MELODIC's 0.07 on noise components.
    """
    if isinstance(data_vox_t, torch.Tensor):
        std = data_vox_t.to(torch.float32).std(dim=1, unbiased=True)
        bad = ~torch.isfinite(std) | (std <= tol)
        bad = bad | ~torch.isfinite(data_vox_t).all(dim=1)
        return bad.cpu().numpy()
    arr = np.asarray(data_vox_t, dtype=np.float32)
    std = arr.std(axis=1, ddof=1) if arr.shape[1] > 1 else np.zeros(arr.shape[0], np.float32)
    return ~np.isfinite(std) | (std <= tol) | ~np.isfinite(arr).all(axis=1)


def prune_mask_constant_voxels(
    mask3d: np.ndarray,
    bad_vox: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Remove flagged voxels from a 3D boolean mask.

    ``bad_vox`` is (V,) over the mask's True voxels in C order — i.e. the same
    ordering ``data[mask3d]`` produces. Returns (new_mask3d, n_dropped).
    """
    n_drop = int(np.count_nonzero(bad_vox))
    if n_drop == 0:
        return mask3d, 0
    new_mask = mask3d.copy()
    new_mask[mask3d] = ~np.asarray(bad_vox, dtype=bool)
    return new_mask, n_drop


def effective_rank_from_spectrum(evals: np.ndarray) -> int:
    """Estimate effective rank from eigenvalue spectrum entropy.

    Parameters
    ----------
    evals : np.ndarray
        Nonnegative eigenvalue spectrum.

    Returns
    -------
    int
        Entropy-based effective rank clamped to valid range.
    """
    ev = np.clip(evals.astype(np.float64), 1e-12, None)
    p = ev / ev.sum()
    h = -(p * np.log(p)).sum()
    er = int(np.round(np.exp(h)))
    return max(1, min(er, len(evals)))


def mp_spikes_from_spectrum(evals: np.ndarray, n_samples: int, n_features: int) -> int:
    """Count spectrum spikes above Marchenko–Pastur bulk edge.

    Parameters
    ----------
    evals : np.ndarray
        Eigenvalue spectrum sorted descending.
    n_samples : int
        Effective sample count.
    n_features : int
        Feature dimensionality.

    Returns
    -------
    int
        Number of eigenvalues above the estimated MP upper edge.
    """
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


def estimate_ica_component_count(
    data_vox_t: torch.Tensor,
    method: int | float | str,
    max_auto_components: int,
    auto_min_components: int,
    auto_var_threshold: float,
    use_mp_prior: bool,
    device: torch.device,
    verbose: bool = False,
    n_eff: int | None = None,
    capture_ppca_trace: bool = False,
) -> tuple[int, dict, dict]:
    """Return selected component count and diagnostics for an ICA run.

    For Minka/MELODIC dimensionality estimation, the data is (vox, time)
    and PCA decomposes the temporal covariance. The Minka formula needs:
    - spectrum: ALL eigenvalues up to min(T, V)
    - n_samples: the effective number of spatial samples

    Parameters
    ----------
    n_eff : int or None
        Effective number of spatial samples for the Minka formula,
        accounting for spatial autocorrelation.  If None, uses raw n_vox
        (which typically over-estimates dimensionality for smooth data).
        MELODIC uses n_vox / (2.5 * smoothness_voxels).
    """
    x_t = data_vox_t.T  # (time, vox)
    n_time, n_vox = int(x_t.shape[0]), int(x_t.shape[1])
    n_samples_minka = n_eff if n_eff is not None else n_vox

    if verbose:
        print(f"    Component estimation: data ({n_vox:,} vox × {n_time} time)")
        if n_eff is not None:
            print(f"    Effective spatial DOF for Minka: {n_samples_minka:,} (vs raw {n_vox:,})")

    # rank_cap = hard upper bound on component count from user
    rank_cap = max(1, min(n_time, n_vox, int(max_auto_components)))
    # For Minka: compute ALL possible eigenvalues so evidence curve can find
    # the true peak.  Do NOT cap at max_auto_components here — that only
    # limits the search range, not the required spectrum length.
    n_eigs_for_minka = max(1, min(n_time, n_vox))

    if verbose:
        print(
            f"    PCA for estimation: extracting {n_eigs_for_minka} eigenvalues "
            f"(component cap={rank_cap}) ..."
        )

    # FSL MELODIC ppca_dim() passes remmean(alldat,2), i.e. row-mean removal
    # (spatial mean per timepoint). cov_r then forms row-wise covariance.
    # We match this cov_r semantics directly.
    x_t_dev = to_tensor(x_t, device=device)
    # Row-centred covariance (cov_r semantics): subtract per-timepoint spatial mean.
    # Use identity: (X-m)(X-m)^T = XX^T - V*mm^T to avoid a (T,V) copy.
    row_mean = x_t_dev.mean(dim=1, keepdim=True)  # (T, 1)
    corr_t = (x_t_dev @ x_t_dev.T - n_vox * (row_mean @ row_mean.T)) / float(n_vox)
    ev_all = torch.linalg.eigvalsh(corr_t).flip(0)  # descending
    del corr_t
    ev_all = torch.clamp(ev_all, min=0)
    total_var = ev_all.sum()
    evr_all = ev_all / total_var if total_var > 0 else ev_all

    n_keep = min(n_eigs_for_minka, len(ev_all))
    ev = ev_all[:n_keep].detach().cpu().numpy()
    evr = evr_all[:n_keep].detach().cpu().numpy()
    del ev_all, evr_all, x_t_dev
    n_ev = len(ev)

    if verbose:
        cum90 = int(np.searchsorted(np.cumsum(evr), 0.90) + 1) if len(evr) > 0 else 0
        cum99 = int(np.searchsorted(np.cumsum(evr), 0.99) + 1) if len(evr) > 0 else 0
        print(f"    PCA: {n_ev} eigenvalues, 90%var@{cum90}, 99%var@{cum99}")

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

    if mode in {"auto", "laplace"}:
        # Marchenko-Pastur ceiling + Minka Laplace evidence; see decomposition/model_order.
        # n_samples must be the *effective* spatial sample size -- passing the raw voxel
        # count is what makes this estimator run away into the noise bulk.
        res = select_model_order(
            ev,
            n_samples=n_samples_minka,
            k_min=auto_min_components,
            k_max=rank_cap,
        )
        if res.at_ceiling:
            print(
                f"  ⚠ Laplace evidence was still rising at k={res.k}, the -max_auto_components "
                f"ceiling; the data supports up to k={res.k_mp} (Marchenko-Pastur). "
                f"Consider raising -max_auto_components."
            )
        diagnostics["mp_signal_count"] = int(res.k_mp)
        diagnostics["mp_lambda_plus"] = float(res.lambda_plus)
        diagnostics["noise_sigma2"] = float(res.sigma2)
        if capture_ppca_trace:
            diagnostics["ppca_trace"] = {
                "eigenvalues": ev.tolist(),
                "log_evidence": np.asarray(res.log_evidence, dtype=np.float64).tolist(),
                "k_min": int(res.k_min),
                "n_eigs_for_minka": int(n_eigs_for_minka),
                "rank_cap": int(rank_cap),
                **res.as_dict(),
            }
        if verbose:
            print(
                f"    Model order: k={res.k} "
                f"(MP ceiling {res.k_mp}, sigma^2={res.sigma2:.4g}, "
                f"n_samples={res.n_samples:,})"
            )
        return res.k, diagnostics, {"mode": "laplace_mp", **res.as_dict()}

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
        return (
            k,
            diagnostics,
            {
                "mode": "hybrid_current",
                "variance_cap": int(est.variance_caps[0]),
                "effective_rank": int(est.entropy_rank_caps[0]),
                "mp_cap": None if est.mp_caps[0] is None else int(est.mp_caps[0]),
                "mp_reason": est.mp_reasons[0],
            },
        )

    if mode == "erank":
        k = effective_rank_from_spectrum(ev)
        return k, diagnostics, {"mode": "effective_rank", "effective_rank": int(k)}

    if mode == "mp":
        spikes = mp_spikes_from_spectrum(ev, n_samples=n_time, n_features=n_vox)
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


def _ggm_em_step(
    x: torch.Tensor,
    neg_x: torch.Tensor,
    x_pos_w: torch.Tensor,
    x_neg_w: torch.Tensor,
    x_pos_mask_f: torch.Tensor,
    x_neg_mask_f: torch.Tensor,
    mu_n: torch.Tensor,
    var_n: torch.Tensor,
    mu_p: torch.Tensor,
    var_p: torch.Tensor,
    mu_ng: torch.Tensor,
    var_ng: torch.Tensor,
    pi_n: torch.Tensor,
    pi_p: torch.Tensor,
    pi_ng: torch.Tensor,
    active_mask_2d: torch.Tensor,
    n_scalar: float,
    min_mode_offset: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,  # total (kept for optional convergence check)
]:
    """One E+M iteration of the Gaussian-Gamma mixture EM.

    Pure tensor function — every input is a torch.Tensor and every output
    is a torch.Tensor — so the body can be ahead-of-time compiled via
    torch.compile to fuse the dozen-plus elementwise ops that dominate
    runtime on CPU.

    Returns (mu_n, var_n, mu_p, var_p, mu_ng, var_ng, pi_n, pi_p, pi_ng, total)
    where ``total`` is the per-voxel mixture density (K, V) needed by the
    convergence check.
    """
    # ----- E-step -----
    # Gaussian PDF inlined to keep this body a single graph.
    p_noise = pi_n * torch.exp(-0.5 * (x - mu_n) ** 2 / var_n) / torch.sqrt(2.0 * torch.pi * var_n)
    # Gamma PDF (positive tail), parameterised by mean & variance.
    a_p = mu_p * mu_p / var_p
    b_p = mu_p / var_p
    log_pdf_p = (
        a_p * torch.log(b_p)
        - torch.lgamma(a_p)
        + (a_p - 1.0) * torch.log(x.clamp(min=1e-30))
        - b_p * x
    )
    p_pos = pi_p * torch.where(
        x > 0,
        torch.exp(log_pdf_p.clamp(-700, 700)),
        torch.tensor(1e-32, device=x.device, dtype=x.dtype),
    )
    # Gamma PDF (negative tail) — uses neg_x.
    a_ng = mu_ng * mu_ng / var_ng
    b_ng = mu_ng / var_ng
    log_pdf_ng = (
        a_ng * torch.log(b_ng)
        - torch.lgamma(a_ng)
        + (a_ng - 1.0) * torch.log(neg_x.clamp(min=1e-30))
        - b_ng * neg_x
    )
    p_neg = pi_ng * torch.where(
        neg_x > 0,
        torch.exp(log_pdf_ng.clamp(-700, 700)),
        torch.tensor(1e-32, device=x.device, dtype=x.dtype),
    )

    total = (p_noise + p_pos + p_neg).clamp(min=1e-32)
    r_noise = p_noise / total
    r_pos = p_pos / total
    r_neg = p_neg / total

    # ----- M-step -----
    w_n = r_noise.sum(dim=1, keepdim=True).clamp(min=1e-8)
    w_p = r_pos.sum(dim=1, keepdim=True).clamp(min=1e-8)
    w_ng = r_neg.sum(dim=1, keepdim=True).clamp(min=1e-8)

    new_mu_n = (r_noise * x).sum(dim=1, keepdim=True) / w_n
    # MELODIC floors every component variance at 1e-4 on each iteration
    # (melgmix.cc:631-634), including the noise Gaussian. Our old 1e-8 let the
    # noise collapse onto a delta spike and z = (x - mu)/1e-4 blow up by 1e4;
    # the reference's floor caps that at 100.
    new_var_n = ((r_noise * (x - new_mu_n) ** 2).sum(dim=1, keepdim=True) / w_n).clamp(min=1e-4)

    sqrt_var_n = torch.sqrt(new_var_n)
    new_pi_n_now = (w_n / n_scalar).clamp(1e-4, 1 - 2e-4)

    x_pos_cnt = (r_pos * x_pos_mask_f).sum(dim=1, keepdim=True).clamp(min=1e-8)
    mu_p_cand = (r_pos * x_pos_w).sum(dim=1, keepdim=True) / x_pos_cnt
    const2_p = (2.6 - new_pi_n_now) * sqrt_var_n + new_mu_n
    var_p_cand = ((r_pos * (x - mu_p_cand) ** 2).sum(dim=1, keepdim=True) / w_p).clamp(min=1e-4)
    floor_p = (0.5 * (const2_p + torch.sqrt(const2_p**2 + 4.0 * var_p_cand))).clamp(
        min=min_mode_offset
    )
    new_mu_p = torch.maximum(mu_p_cand, floor_p)
    new_var_p = torch.minimum(var_p_cand, 0.5 * new_mu_p**2).clamp(min=1e-4)

    x_neg_cnt = (r_neg * x_neg_mask_f).sum(dim=1, keepdim=True).clamp(min=1e-8)
    mu_ng_cand = (r_neg * x_neg_w).sum(dim=1, keepdim=True) / x_neg_cnt
    const2_ng = (2.6 - new_pi_n_now) * sqrt_var_n - new_mu_n
    var_ng_cand = ((r_neg * (neg_x - mu_ng_cand) ** 2).sum(dim=1, keepdim=True) / w_ng).clamp(
        min=1e-4
    )
    floor_ng = (0.5 * (const2_ng + torch.sqrt(const2_ng**2 + 4.0 * var_ng_cand))).clamp(
        min=min_mode_offset
    )
    new_mu_ng = torch.maximum(mu_ng_cand, floor_ng)
    new_var_ng = torch.minimum(var_ng_cand, 0.5 * new_mu_ng**2).clamp(min=1e-4)

    new_pi_n = (w_n / n_scalar).clamp(1e-4, 1 - 2e-4)
    new_pi_p = (w_p / n_scalar).clamp(1e-4, 1 - 2e-4)
    new_pi_ng = (w_ng / n_scalar).clamp(1e-4, 1 - 2e-4)
    pi_sum = new_pi_n + new_pi_p + new_pi_ng
    new_pi_n = new_pi_n / pi_sum
    new_pi_p = new_pi_p / pi_sum
    new_pi_ng = new_pi_ng / pi_sum

    # Apply only to active components — converged components keep old state.
    mu_n = torch.where(active_mask_2d, new_mu_n, mu_n)
    var_n = torch.where(active_mask_2d, new_var_n, var_n)
    mu_p = torch.where(active_mask_2d, new_mu_p, mu_p)
    var_p = torch.where(active_mask_2d, new_var_p, var_p)
    mu_ng = torch.where(active_mask_2d, new_mu_ng, mu_ng)
    var_ng = torch.where(active_mask_2d, new_var_ng, var_ng)
    pi_n = torch.where(active_mask_2d, new_pi_n, pi_n)
    pi_p = torch.where(active_mask_2d, new_pi_p, pi_p)
    pi_ng = torch.where(active_mask_2d, new_pi_ng, pi_ng)

    return mu_n, var_n, mu_p, var_p, mu_ng, var_ng, pi_n, pi_p, pi_ng, total


# Compiled variant; the bare function above is the parity reference.
# torch.compile here fuses ~30 elementwise + reduction kernels into 2-4
# fused kernels, which on CPU mainly cuts memory bandwidth (each (K, V)
# tensor allocated/freed inside the loop is reusing the same buffers).
# safe_compile applies the shared inductor policy (PCH disabled) and degrades to
# the eager function if compilation fails at call time, rather than crashing.
_ggm_em_step_compiled = safe_compile(_ggm_em_step, dynamic=True, fullgraph=False, mode="default")


def _gamma_pdf_torch(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Batched Gamma PDF parameterized by mean and variance.

    Parameters
    ----------
    x : (K, V) data
    mean : (K, 1) means  (> 0)
    var : (K, 1) variances (> 0)

    Returns (K, V) pdf values.
    """
    a = mean**2 / var  # shape param
    b = mean / var  # rate param
    log_pdf = a * torch.log(b) - torch.lgamma(a) + (a - 1.0) * torch.log(x.clamp(min=1e-30)) - b * x
    result = torch.where(
        x > 0,
        torch.exp(log_pdf.clamp(-700, 700)),
        # Match input dtype to avoid silent float32→float64 promotion in torch.where
        torch.tensor(1e-32, device=x.device, dtype=x.dtype),
    )
    return result


def _gauss_pdf_torch(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Batched Gaussian PDF.  x: (K,V), mean: (K,1), var: (K,1)."""
    return torch.exp(-0.5 * (x - mean) ** 2 / var) / torch.sqrt(2.0 * torch.pi * var)


@torch.inference_mode()
def batch_fit_ggm(
    components_kv: torch.Tensor,
    n_iter: int = 200,
    min_mode_offset: float = 0.001,
    verbose: bool = False,
    fallback_pi_thresh: float = 0.4,
) -> dict:
    """Fit GGM to ALL ICA spatial maps simultaneously on GPU.

    This is the batched PyTorch equivalent of fit_ggm — processes K
    components × V voxels in parallel.

    Parameters match MELODIC's `ggmix.cc`: input is z-scored per component,
    Gamma mean/variance constraints use MELODIC's adaptive `const2` formula,
    and components with pi_noise<0.4 fall back to a 3-Gaussian mixture.

    Parameters
    ----------
    components_kv : Tensor of shape (K, V)
        Raw ICA spatial map values (will be z-scored internally).
    n_iter : int
        Maximum EM iterations.
    min_mode_offset : float
        Lower floor on Gamma means (MELODIC uses 0.001 in normalized space).
    verbose : bool
        Show tqdm progress bar over EM iterations.
    fallback_pi_thresh : float
        Components whose fitted noise proportion falls below this switch to a
        3-Gaussian mixture. Set to 0.0 to disable the fallback entirely.

    Returns
    -------
    result : dict with Tensor values, each of shape (K,) or (K, V).
        Note: mu/var fields are in z-scored (per-component) space; z_signed
        and p_signal are scale-invariant.
    """
    device = components_kv.device
    K, V = components_kv.shape
    # Keep x in float32: the dominant (K, V) tensor costs 2× VRAM in float64.
    x_raw = components_kv.float()  # (K, V)
    n = float(V)

    # ----- I1: per-component z-score (matches MELODIC ggmix::setup) -----
    data_mean = x_raw.mean(dim=1, keepdim=True)  # (K, 1)
    data_std = x_raw.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-8)
    x = (x_raw - data_mean) / data_std
    del x_raw

    # ----- Initialization (MELODIC melgmix.cc::setup + ggmfit) -----
    # All three components start at the full data variance E[x^2] with equal
    # mixing weights, and the tails at +-2 sigma. An "informed" init that starts
    # the noise Gaussian narrow (we used var/4, pi=0.8) biases the very first
    # E-step: the Gammas immediately claim the shoulders, and because the
    # const2 floor scales with sqrt(var_noise) they then have room to creep
    # further in. Starting wide lets the noise component hold the bulk.
    v0 = (x * x).mean(dim=1, keepdim=True).clamp(min=1e-4)  # = 1 for z-scored x
    s0 = torch.sqrt(v0)
    mu_n = -2.0 * x.mean(dim=1, keepdim=True)  # ggmfit line 1: -2*mean(data)
    var_n = v0.clone()

    neg_x = -x
    # mu_p / mu_ng are both stored positive; mu_ng is the mean of neg_x, i.e.
    # the magnitude of MELODIC's negative means(3) = mu_n - 2*sigma.
    mu_p = (mu_n + 2.0 * s0).clamp(min=min_mode_offset)
    var_p = v0.clone()
    mu_ng = (2.0 * s0 - mu_n).clamp(min=min_mode_offset)
    var_ng = v0.clone()

    # Mixing proportions (K, 1) — float32, negligible size
    pi_n = torch.full((K, 1), 1.0 / 3.0, device=device, dtype=torch.float32)
    pi_p = torch.full((K, 1), 1.0 / 3.0, device=device, dtype=torch.float32)
    pi_ng = torch.full((K, 1), 1.0 / 3.0, device=device, dtype=torch.float32)

    eps_conv = log(V) / 1000.0
    old_ll = torch.full((K, 1), -1e30, device=device, dtype=torch.float32)
    converged = torch.zeros(K, dtype=torch.bool, device=device)

    # Cache the per-iteration invariants. x and neg_x never change inside
    # the loop, but x.clamp(min=0), neg_x.clamp(min=0), and the (>0) masks
    # were being recomputed every iteration — ~50 MB of redundant
    # allocations per iter at typical K, V. Hoisting them is pure
    # bandwidth savings on CPU and a smaller VRAM win on GPU.
    x_pos_w = x.clamp(min=0)
    x_neg_w = neg_x.clamp(min=0)
    x_pos_mask_f = (x > 0).to(x.dtype)
    x_neg_mask_f = (neg_x > 0).to(x.dtype)

    # Convergence check is a global reduction. On CPU especially this is
    # cheap-but-not-free; on GPU it forces a sync on .all(). Check every
    # CONV_CHECK_INTERVAL iters once past the warm-up.
    CONV_CHECK_INTERVAL = 5

    iterator = range(n_iter)
    if verbose:
        iterator = tqdm(iterator, desc="  GGM EM", leave=True, unit="it")

    for it in iterator:
        active = ~converged  # (K,)
        if not active.any():
            break
        active_mask_2d = active.unsqueeze(1)

        # One full E+M step in a single fused (when compiled) graph.
        mu_n, var_n, mu_p, var_p, mu_ng, var_ng, pi_n, pi_p, pi_ng, total = _ggm_em_step_compiled(
            x,
            neg_x,
            x_pos_w,
            x_neg_w,
            x_pos_mask_f,
            x_neg_mask_f,
            mu_n,
            var_n,
            mu_p,
            var_p,
            mu_ng,
            var_ng,
            pi_n,
            pi_p,
            pi_ng,
            active_mask_2d,
            float(n),
            float(min_mode_offset),
        )

        # Log-likelihood convergence check (less frequent for less sync).
        if it > 20 and (it % CONV_CHECK_INTERVAL == 0):
            ll = total.log().sum(dim=1, keepdim=True)  # (K, 1)
            just_converged = ((ll - old_ll) < eps_conv).squeeze(1) & ~converged
            converged = converged | just_converged
            if converged.all():
                break
            old_ll = ll

    # Free EM intermediates before final posterior computation
    del old_ll  # r_noise/r_pos/r_neg now live only inside the compiled step
    if x.device.type == "cuda":
        torch.cuda.empty_cache()

    # I4: GMM fallback for high-signal components (pi_noise < 0.4) ---------
    # MELODIC switches to a 3-Gaussian mixture when the noise proportion is
    # small, since a noise-Gaussian + signal-Gammas model is inappropriate
    # for near-uniform distributions.
    fb_mask = pi_n.squeeze(1) < fallback_pi_thresh  # (K,)
    if fb_mask.any():
        fb_idx = fb_mask.nonzero(as_tuple=True)[0]
        gmm = _batch_gmm_3comp(x[fb_idx], n_iter=n_iter)
        # Overwrite GGM params with GMM means/vars/props for these components.
        # In GMM mode we keep the same field names but mu_pos/mu_neg now hold
        # Gaussian means (positive and negative tails); var_pos/var_neg hold
        # their variances. p_signal is computed as (g_pos + g_neg) / total.
        idx2 = fb_idx.unsqueeze(1)
        mu_n.index_copy_(0, fb_idx, gmm["mu1"])
        var_n.index_copy_(0, fb_idx, gmm["var1"])
        mu_p.index_copy_(0, fb_idx, gmm["mu2"])
        var_p.index_copy_(0, fb_idx, gmm["var2"])
        mu_ng.index_copy_(0, fb_idx, gmm["mu3"])
        var_ng.index_copy_(0, fb_idx, gmm["var3"])
        pi_n.index_copy_(0, fb_idx, gmm["pi1"])
        pi_p.index_copy_(0, fb_idx, gmm["pi2"])
        pi_ng.index_copy_(0, fb_idx, gmm["pi3"])
        del idx2

    # Final posterior — compute sequentially to minimize peak memory.
    # GGM components: gauss(noise) + gamma(pos) + gamma(neg).
    # GMM components: gauss(noise) + gauss(pos) + gauss(neg) — mu_ng holds a
    # *negative* Gaussian mean directly (as set by _batch_gmm_3comp).
    p_noise_f = pi_n * _gauss_pdf_torch(x, mu_n, var_n)
    if fb_mask.any():
        # Per-component selection of Gamma vs Gaussian for the signal components.
        gamma_pos = pi_p * _gamma_pdf_torch(x, mu_p, var_p)
        gauss_pos = pi_p * _gauss_pdf_torch(x, mu_p, var_p)
        fb2 = fb_mask.unsqueeze(1)
        p_pos_f = torch.where(fb2, gauss_pos, gamma_pos)
        del gamma_pos, gauss_pos
        gamma_neg = pi_ng * _gamma_pdf_torch(neg_x, mu_ng, var_ng)
        gauss_neg = pi_ng * _gauss_pdf_torch(x, mu_ng, var_ng)
        p_neg_f = torch.where(fb2, gauss_neg, gamma_neg)
        del gamma_neg, gauss_neg
    else:
        p_pos_f = pi_p * _gamma_pdf_torch(x, mu_p, var_p)
        p_neg_f = pi_ng * _gamma_pdf_torch(neg_x, mu_ng, var_ng)
    del neg_x  # free (K, V) float32

    total_f = (p_noise_f + p_pos_f + p_neg_f).clamp(min=1e-32)
    del p_noise_f
    p_signal = (p_pos_f + p_neg_f) / total_f  # (K, V) float32
    del p_pos_f, p_neg_f, total_f

    sigma_n = torch.sqrt(var_n.clamp(min=1e-4))
    z_signed = (x - mu_n) / sigma_n  # (K, V) float32 — z-scored input space
    del x

    return {
        "z_signed": z_signed,
        "p_signal": p_signal,
        "mu_noise": mu_n.squeeze(1),
        "var_noise": var_n.squeeze(1),
        "mu_pos": mu_p.squeeze(1),
        "var_pos": var_p.squeeze(1),
        "mu_neg": mu_ng.squeeze(1),
        "var_neg": var_ng.squeeze(1),
        "pi_noise": pi_n.squeeze(1),
        "pi_pos": pi_p.squeeze(1),
        "pi_neg": pi_ng.squeeze(1),
        "converged": converged,
        "gmm_fallback": fb_mask,  # (K,) bool
        "data_mean": data_mean.squeeze(1),  # (K,) original-space recovery
        "data_std": data_std.squeeze(1),
    }


@torch.inference_mode()
def _batch_gmm_3comp(x: torch.Tensor, n_iter: int = 200) -> dict:
    """Batched 3-Gaussian mixture EM — used as the GMM fallback in batch_fit_ggm.

    Components are ordered (noise, pos, neg) by initialization, but the EM
    is unconstrained except for variance floor.

    Parameters
    ----------
    x : (K, V) z-scored inputs.

    Returns dict with mu/var/pi tensors of shape (K, 1).
    """
    device = x.device
    K, V = x.shape
    n = float(V)
    # MELODIC GMM init: m1=mean, m2=m1+sqrt(v1), m3=m1-sqrt(v1); v all = E[x²].
    v0 = (x * x).mean(dim=1, keepdim=True).clamp(min=1e-4)
    s0 = torch.sqrt(v0)
    m1 = x.mean(dim=1, keepdim=True)
    m2 = m1 + s0
    m3 = m1 - s0
    v1 = v0.clone()
    v2 = v0.clone()
    v3 = v0.clone()
    pi1 = torch.full((K, 1), 1.0 / 3.0, device=device, dtype=x.dtype)
    pi2 = pi1.clone()
    pi3 = pi1.clone()

    eps_conv = log(V) / 1000.0
    old_ll = torch.full((K, 1), -1e30, device=device, dtype=x.dtype)
    converged = torch.zeros(K, dtype=torch.bool, device=device)
    CONV_CHECK_INTERVAL = 5  # see batch_fit_ggm for rationale

    for it in range(n_iter):
        p1 = pi1 * _gauss_pdf_torch(x, m1, v1)
        p2 = pi2 * _gauss_pdf_torch(x, m2, v2)
        p3 = pi3 * _gauss_pdf_torch(x, m3, v3)
        total = (p1 + p2 + p3).clamp(min=1e-32)
        r1 = p1 / total
        r2 = p2 / total
        r3 = p3 / total

        if it > 20 and (it % CONV_CHECK_INTERVAL == 0):
            ll = total.log().sum(dim=1, keepdim=True)
            converged = converged | (((ll - old_ll) < eps_conv).squeeze(1) & ~converged)
            if converged.all():
                break
            old_ll = ll

        active = ~converged
        if not active.any():
            break

        w1 = r1.sum(dim=1, keepdim=True).clamp(min=1e-8)
        w2 = r2.sum(dim=1, keepdim=True).clamp(min=1e-8)
        w3 = r3.sum(dim=1, keepdim=True).clamp(min=1e-8)
        new_m1 = (r1 * x).sum(dim=1, keepdim=True) / w1
        new_m2 = (r2 * x).sum(dim=1, keepdim=True) / w2
        new_m3 = (r3 * x).sum(dim=1, keepdim=True) / w3
        new_v1 = ((r1 * (x - new_m1) ** 2).sum(dim=1, keepdim=True) / w1).clamp(min=1e-4)
        new_v2 = ((r2 * (x - new_m2) ** 2).sum(dim=1, keepdim=True) / w2).clamp(min=1e-4)
        new_v3 = ((r3 * (x - new_m3) ** 2).sum(dim=1, keepdim=True) / w3).clamp(min=1e-4)
        new_pi1 = (w1 / n).clamp(1e-4, 1 - 2e-4)
        new_pi2 = (w2 / n).clamp(1e-4, 1 - 2e-4)
        new_pi3 = (w3 / n).clamp(1e-4, 1 - 2e-4)
        psum = new_pi1 + new_pi2 + new_pi3
        new_pi1, new_pi2, new_pi3 = new_pi1 / psum, new_pi2 / psum, new_pi3 / psum

        a = active.unsqueeze(1)
        m1 = torch.where(a, new_m1, m1)
        m2 = torch.where(a, new_m2, m2)
        m3 = torch.where(a, new_m3, m3)
        v1 = torch.where(a, new_v1, v1)
        v2 = torch.where(a, new_v2, v2)
        v3 = torch.where(a, new_v3, v3)
        pi1 = torch.where(a, new_pi1, pi1)
        pi2 = torch.where(a, new_pi2, pi2)
        pi3 = torch.where(a, new_pi3, pi3)

    return {
        "mu1": m1,
        "var1": v1,
        "pi1": pi1,
        "mu2": m2,
        "var2": v2,
        "pi2": pi2,
        "mu3": m3,
        "var3": v3,
        "pi3": pi3,
    }


# ---------- Legacy scalar wrappers (kept for API compat) ----------


def _gamma_pdf(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    """Gamma PDF parameterized by mean and variance (mean > 0, var > 0).

    shape a = mean^2 / var, rate b = mean / var.
    """
    if mean <= 0 or var <= 0:
        return np.full_like(x, 1e-32)
    a = mean**2 / var
    b = mean / var
    # Log-space computation for numerical stability
    log_pdf = a * np.log(b) - scipy_gammaln(a) + (a - 1) * np.log(np.clip(x, 1e-30, None)) - b * x
    result = np.where(x > 0, np.exp(np.clip(log_pdf, -700, 700)), 1e-32)
    return result


def _gauss_pdf(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    """Gaussian PDF."""
    if var <= 0:
        return np.full_like(x, 1e-32)
    return np.exp(-0.5 * (x - mean) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


def fit_ggm(
    values: np.ndarray,
    n_iter: int = 200,
    min_mode_offset: float = 0.001,
) -> dict:
    """Fit a Gaussian-Gamma Mixture (GGM) model to ICA spatial map values.

    This follows FSL MELODIC's mixture modeling approach:
    - Component 0: Gaussian (noise, centered near 0)
    - Component 1: Gamma (positive activations, x > 0)
    - Component 2: Flipped Gamma (negative activations, x < 0)

    Parameters
    ----------
    values : 1D array
        Raw ICA spatial map values (NOT absolute values).
    n_iter : int
        Maximum EM iterations.
    min_mode_offset : float
        Minimum separation between noise mean and signal mode.

    Returns
    -------
    result : dict with keys:
        mu_noise, var_noise : Gaussian noise parameters
        mu_pos, var_pos : Gamma parameters for positive signal (mean, var)
        mu_neg, var_neg : Gamma parameters for negative signal (|mean|, var)
        pi_noise, pi_pos, pi_neg : mixing proportions
        p_signal : array of posterior signal probability per voxel
        converged : bool
    """
    x_raw = values.astype(np.float64).ravel()
    # I1: per-component z-score (matches MELODIC ggmix::setup)
    data_mean = float(np.mean(x_raw))
    data_std = max(float(np.std(x_raw)), 1e-8)
    x = (x_raw - data_mean) / data_std
    n = len(x)

    # Initialize noise Gaussian from central mass
    mu_n = float(np.median(x))
    var_n = float(np.var(x) * 0.25) + 1e-8

    # Initialize positive Gamma from positive tail
    pos_vals = x[x > mu_n + np.sqrt(var_n)]
    if len(pos_vals) < 10:
        pos_vals = x[x > np.percentile(x, 80)]
    mu_p = max(float(np.mean(pos_vals)) if len(pos_vals) > 0 else 2.0, min_mode_offset)
    var_p = max(float(np.var(pos_vals)) if len(pos_vals) > 1 else 1.0, 0.1)

    # Initialize negative Gamma from negative tail
    neg_vals = -x[x < mu_n - np.sqrt(var_n)]
    if len(neg_vals) < 10:
        neg_vals = -x[x < np.percentile(x, 20)]
    mu_ng = max(float(np.mean(neg_vals)) if len(neg_vals) > 0 else 2.0, min_mode_offset)
    var_ng = max(float(np.var(neg_vals)) if len(neg_vals) > 1 else 1.0, 0.1)

    # Mixing proportions
    pi_n = 0.8
    pi_p = 0.1
    pi_ng = 0.1

    eps_conv = log(n) / 1000.0
    old_ll = -np.inf
    converged = False

    for it in range(n_iter):
        # E-step: compute responsibilities
        p_noise = pi_n * _gauss_pdf(x, mu_n, var_n)
        p_pos = pi_p * _gamma_pdf(x, mu_p, var_p)
        p_neg = pi_ng * _gamma_pdf(-x, mu_ng, var_ng)

        total = np.clip(p_noise + p_pos + p_neg, 1e-30, None)
        r_noise = p_noise / total
        r_pos = p_pos / total
        r_neg = p_neg / total

        # Log-likelihood for convergence check
        ll = float(np.sum(np.log(total)))
        if it > 20 and ll - old_ll < eps_conv:
            converged = True
            break
        old_ll = ll

        # M-step: update parameters
        w_n = max(float(r_noise.sum()), 1e-8)
        w_p = max(float(r_pos.sum()), 1e-8)
        w_ng = max(float(r_neg.sum()), 1e-8)

        # Noise Gaussian
        mu_n = float((r_noise * x).sum() / w_n)
        var_n = max(float((r_noise * (x - mu_n) ** 2).sum() / w_n), 1e-8)

        # I2: MELODIC adaptive floor — const2 = (2.6 - pi_n)·sqrt(var_n) ± mu_n
        sqrt_var_n = np.sqrt(var_n)
        pi_n_now = float(np.clip(w_n / n, 1e-4, 1 - 2e-4))

        # Positive Gamma (only from x > 0 effectively, but weighted)
        x_pos_weighted = np.where(x > 0, x, 0.0)
        mu_p_new = float((r_pos * x_pos_weighted).sum() / max(float((r_pos * (x > 0)).sum()), 1e-8))
        var_p_new = max(float((r_pos * (x - mu_p_new) ** 2).sum() / w_p), 1e-4)
        const2_p = (2.6 - pi_n_now) * sqrt_var_n + mu_n
        floor_p = max(0.5 * (const2_p + np.sqrt(const2_p**2 + 4.0 * var_p_new)), min_mode_offset)
        mu_p = max(mu_p_new, floor_p)
        # I3: variance upper bound var ≤ 0.5·mu²
        var_p = max(min(var_p_new, 0.5 * mu_p**2), 1e-4)

        # Negative Gamma (use -x; mu_ng kept as positive |mean|)
        neg_x = -x
        x_neg_weighted = np.where(neg_x > 0, neg_x, 0.0)
        mu_ng_new = float(
            (r_neg * x_neg_weighted).sum() / max(float((r_neg * (neg_x > 0)).sum()), 1e-8)
        )
        var_ng_new = max(float((r_neg * (neg_x - mu_ng_new) ** 2).sum() / w_ng), 1e-4)
        const2_ng = (2.6 - pi_n_now) * sqrt_var_n - mu_n
        floor_ng = max(
            0.5 * (const2_ng + np.sqrt(const2_ng**2 + 4.0 * var_ng_new)), min_mode_offset
        )
        mu_ng = max(mu_ng_new, floor_ng)
        var_ng = max(min(var_ng_new, 0.5 * mu_ng**2), 1e-4)

        # Proportions
        pi_n = float(np.clip(w_n / n, 1e-4, 1 - 2e-4))
        pi_p = float(np.clip(w_p / n, 1e-4, 1 - 2e-4))
        pi_ng = float(np.clip(w_ng / n, 1e-4, 1 - 2e-4))
        # Normalize
        pi_sum = pi_n + pi_p + pi_ng
        pi_n /= pi_sum
        pi_p /= pi_sum
        pi_ng /= pi_sum

    # I4: GMM fallback when noise proportion is small (mostly-signal component)
    gmm_fallback = pi_n < 0.4
    if gmm_fallback:
        mu_n, var_n, pi_n, mu_p, var_p, pi_p, mu_ng, var_ng, pi_ng = _gmm_3comp_fit(x, n_iter)
        p_noise_final = pi_n * _gauss_pdf(x, mu_n, var_n)
        p_pos_final = pi_p * _gauss_pdf(x, mu_p, var_p)
        p_neg_final = pi_ng * _gauss_pdf(x, mu_ng, var_ng)
    else:
        p_noise_final = pi_n * _gauss_pdf(x, mu_n, var_n)
        p_pos_final = pi_p * _gamma_pdf(x, mu_p, var_p)
        p_neg_final = pi_ng * _gamma_pdf(-x, mu_ng, var_ng)
    total_final = np.clip(p_noise_final + p_pos_final + p_neg_final, 1e-30, None)
    p_signal = (p_pos_final + p_neg_final) / total_final

    return {
        "mu_noise": mu_n,
        "var_noise": var_n,
        "mu_pos": mu_p,
        "var_pos": var_p,
        "mu_neg": mu_ng,
        "var_neg": var_ng,
        "pi_noise": pi_n,
        "pi_pos": pi_p,
        "pi_neg": pi_ng,
        "p_signal": p_signal,
        "converged": converged,
        "gmm_fallback": gmm_fallback,
        "data_mean": data_mean,
        "data_std": data_std,
    }


def _gmm_3comp_fit(x: np.ndarray, n_iter: int) -> tuple:
    """3-Gaussian mixture EM (scalar fallback for fit_ggm)."""
    n = len(x)
    v0 = max(float(np.mean(x * x)), 1e-4)
    s0 = np.sqrt(v0)
    m1 = float(np.mean(x))
    m2 = m1 + s0
    m3 = m1 - s0
    v1 = v2 = v3 = v0
    pi1 = pi2 = pi3 = 1.0 / 3.0
    eps_conv = log(n) / 1000.0
    old_ll = -np.inf
    for it in range(n_iter):
        p1 = pi1 * _gauss_pdf(x, m1, v1)
        p2 = pi2 * _gauss_pdf(x, m2, v2)
        p3 = pi3 * _gauss_pdf(x, m3, v3)
        total = np.clip(p1 + p2 + p3, 1e-30, None)
        r1, r2, r3 = p1 / total, p2 / total, p3 / total
        ll = float(np.sum(np.log(total)))
        if it > 20 and ll - old_ll < eps_conv:
            break
        old_ll = ll
        w1 = max(float(r1.sum()), 1e-8)
        w2 = max(float(r2.sum()), 1e-8)
        w3 = max(float(r3.sum()), 1e-8)
        m1 = float((r1 * x).sum() / w1)
        m2 = float((r2 * x).sum() / w2)
        m3 = float((r3 * x).sum() / w3)
        v1 = max(float((r1 * (x - m1) ** 2).sum() / w1), 1e-4)
        v2 = max(float((r2 * (x - m2) ** 2).sum() / w2), 1e-4)
        v3 = max(float((r3 * (x - m3) ** 2).sum() / w3), 1e-4)
        pi1 = float(np.clip(w1 / n, 1e-4, 1 - 2e-4))
        pi2 = float(np.clip(w2 / n, 1e-4, 1 - 2e-4))
        pi3 = float(np.clip(w3 / n, 1e-4, 1 - 2e-4))
        psum = pi1 + pi2 + pi3
        pi1, pi2, pi3 = pi1 / psum, pi2 / psum, pi3 / psum
    return m1, v1, pi1, m2, v2, pi2, m3, v3, pi3


def mixture_zscores_signed(comp_map: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute z-scores and signal probability using Gaussian-Gamma Mixture.

    Follows FSL MELODIC's approach: fit a GGM to the spatial map values,
    then compute z-scores relative to the noise distribution and posterior
    signal probabilities.

    Parameters
    ----------
    comp_map : 1D array
        Raw ICA spatial map values for one component.

    Returns
    -------
    z_signed : array
        Signed z-scores (positive for activation, negative for deactivation).
    p_signal : array
        Posterior probability of signal (activation or deactivation) at each voxel.
    meta : dict
        Mixture model parameters and diagnostics.
    """
    vals = comp_map.astype(np.float64)
    result = fit_ggm(vals)

    mu_n = result["mu_noise"]
    sigma_n = max(np.sqrt(result["var_noise"]), 1e-8)
    p_signal = result["p_signal"]

    # Z-score relative to noise distribution. mu_n / sigma_n are in the
    # per-component z-scored frame used internally by fit_ggm, so we apply
    # the same normalization to vals before standardizing by the noise.
    vals_norm = (vals - result["data_mean"]) / max(result["data_std"], 1e-8)
    z_signed = ((vals_norm - mu_n) / sigma_n).astype(np.float32)

    meta = {
        "mu_noise": float(mu_n),
        "sigma_noise": float(sigma_n),
        "mu_signal_pos": float(result["mu_pos"]),
        "sigma_signal_pos": float(np.sqrt(result["var_pos"])),
        "mu_signal_neg": float(result["mu_neg"]),
        "sigma_signal_neg": float(np.sqrt(result["var_neg"])),
        "pi_noise": float(result["pi_noise"]),
        "pi_pos": float(result["pi_pos"]),
        "pi_neg": float(result["pi_neg"]),
        "mixing_signal": float(np.mean(p_signal)),
        "converged": result["converged"],
        "gmm_fallback": bool(result.get("gmm_fallback", False)),
    }

    return z_signed, p_signal.astype(np.float32), meta


def batch_mixture_zscores(
    components_kv: torch.Tensor,
    device: torch.device | None = None,
    verbose: bool = False,
    n_iter: int = 200,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Batched GGM z-scores for all components on GPU.

    Parameters
    ----------
    components_kv : (K, V) Tensor or ndarray of raw ICA spatial maps.
    device : torch device (default: same as input or CPU).
    verbose : show tqdm progress bar for EM iterations.

    Returns
    -------
    z_signed : (K, V) float32 Tensor — signed z-scores.
    p_signal : (K, V) float32 Tensor — posterior P(signal).
    meta_list : list of K dicts with per-component parameters.
    """
    if not isinstance(components_kv, torch.Tensor):
        components_kv = torch.as_tensor(components_kv, dtype=torch.float32)
    if device is not None:
        components_kv = components_kv.to(device)

    K, V = components_kv.shape

    # Determine how many components to process per batch based on available
    # memory.  Peak GGM EM usage is ~9 simultaneous (K_chunk, V) float64
    # tensors plus the float32 input.
    from fastfuncstuff.memory import get_available_memory

    avail = get_available_memory(components_kv.device)
    # Peak GGM EM uses ~12 simultaneous (K_chunk, V) float64 tensors:
    # x, neg_x, p_noise, p_pos, p_neg, total, r_noise, r_pos, r_neg,
    # plus M-step temporaries (torch.where zeros_like, squared diffs, etc.)
    bytes_per_comp = V * 8 * 12 + V * 4  # 12 float64 + 1 float32 per component
    k_chunk = max(1, int(avail * 0.8) // max(bytes_per_comp, 1))  # 80% safety margin
    k_chunk = min(k_chunk, K)

    z_parts: list[torch.Tensor] = []
    p_parts: list[torch.Tensor] = []
    meta_list: list[dict] = []

    n_batches = (K + k_chunk - 1) // k_chunk
    for k0 in tqdm(
        range(0, K, k_chunk),
        total=n_batches,
        desc="  GGM batches",
        leave=True,
        disable=n_batches <= 1,
    ):
        k1 = min(k0 + k_chunk, K)
        result = batch_fit_ggm(components_kv[k0:k1], n_iter=n_iter, verbose=(verbose and k0 == 0))

        z_parts.append(result["z_signed"].cpu())
        p_parts.append(result["p_signal"].cpu())

        for k in range(k1 - k0):
            meta_list.append(
                {
                    "mu_noise": float(result["mu_noise"][k]),
                    "sigma_noise": float(torch.sqrt(result["var_noise"][k].clamp(min=1e-16))),
                    "mu_signal_pos": float(result["mu_pos"][k]),
                    "sigma_signal_pos": float(torch.sqrt(result["var_pos"][k].clamp(min=1e-16))),
                    "mu_signal_neg": float(result["mu_neg"][k]),
                    "sigma_signal_neg": float(torch.sqrt(result["var_neg"][k].clamp(min=1e-16))),
                    "pi_noise": float(result["pi_noise"][k]),
                    "pi_pos": float(result["pi_pos"][k]),
                    "pi_neg": float(result["pi_neg"][k]),
                    "mixing_signal": float(result["p_signal"][k].mean()),
                    "converged": bool(result["converged"][k]),
                    "gmm_fallback": bool(result["gmm_fallback"][k]),
                }
            )
        del result
        if components_kv.device.type == "cuda":
            torch.cuda.empty_cache()

    z_signed = torch.cat(z_parts, dim=0)  # stays on CPU to avoid GPU OOM
    p_signal = torch.cat(p_parts, dim=0)
    del z_parts, p_parts

    return z_signed, p_signal, meta_list


def build_task_design_for_run(
    onsets_files: list[str],
    durations_arg: list[str],
    run_idx: int,
    onset_row: int | None,
    n_timepoints: int,
    tr: float,
    microtime_dt: float,
    device: torch.device,
) -> tuple[torch.Tensor, list[str], list[float]]:
    """Build one-run task design matrix from AFNI timing files.

    Parameters
    ----------
    onsets_files : list[str]
        AFNI timing files, one per condition.
    durations_arg : list[str]
        Duration arguments passed from CLI.
    run_idx : int
        Zero-based run index to select when ``onset_row`` is not provided.
    onset_row : int or None
        Optional 1-based row override in each timing file.
    n_timepoints : int
        Number of TRs in run.
    tr : float
        Repetition time in seconds.
    microtime_dt : float
        Microtime bin size in seconds.
    device : torch.device
        Torch device for matrix construction.

    Returns
    -------
    design : torch.Tensor
        Run design matrix in TR space.
    labels : list[str]
        Condition labels aligned to design columns.
    durations : list[float]
        Parsed per-condition durations in seconds.
    """
    all_onsets_full = [parse_afni_timing_file(fp) for fp in onsets_files]
    n_conds = len(all_onsets_full)
    labels = [f"cond{i + 1}" for i in range(n_conds)]
    durations = parse_durations(durations_arg, n_conds, labels)

    onsets_this_run = []
    for cond_runs in all_onsets_full:
        if onset_row is not None:
            row_idx = int(onset_row) - 1  # 1-based CLI/API to 0-based list index
            if row_idx < 0 or row_idx >= len(cond_runs):
                raise ValueError(
                    f"Requested onset_row={onset_row} is out of range for timing file "
                    f"(has {len(cond_runs)} rows)"
                )
            onsets_this_run.append([cond_runs[row_idx]])
        else:
            if run_idx >= len(cond_runs):
                raise ValueError(
                    f"Timing file has fewer runs than input data (missing run {run_idx + 1})"
                )
            onsets_this_run.append([cond_runs[run_idx]])

    from fastfuncstuff.design.matrices import (
        build_event_design_microtime,
        commensurate_microtime_dt,
    )

    microtime_dt = commensurate_microtime_dt(tr, microtime_dt)
    hrf = get_spmg1_hrf(microtime_dt=microtime_dt, device=device)

    design = build_event_design_microtime(
        all_onsets=onsets_this_run,
        durations=durations,
        hrf_bases=hrf,
        n_timepoints_per_run=[n_timepoints],
        tr=tr,
        microtime_dt=microtime_dt,
        device=device,
    )
    assert isinstance(design, torch.Tensor)
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
