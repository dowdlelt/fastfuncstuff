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

from math import lgamma, log, pi

import numpy as np
import torch
from scipy.special import gammaln as scipy_gammaln
from tqdm.auto import tqdm

from fastfuncstuff.denoise.sequential import estimate_noise_component_caps_per_run
from fastfuncstuff.design.matrices import convolve_hrf_microtime
from fastfuncstuff.design.builder import (
    create_onset_matrix_microtime,
    parse_afni_timing_file,
    parse_durations,
)
from fastfuncstuff.glm.core import construct_polynomial_matrix
from fastfuncstuff.design.hrf import get_spmg1_hrf
from fastfuncstuff.utils import to_tensor
from .pca import PCA


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


@torch.inference_mode()
def apply_polort_projection(
    data_vox_t: torch.Tensor,
    polort: int,
    device: torch.device,
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

    Returns
    -------
    torch.Tensor
        Detrended voxel-by-time matrix.
    """
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
    """
    if high_pass_hz is None or high_pass_hz <= 0:
        return data_vox_t
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


@torch.inference_mode()
def apply_melodic_voxel_varnorm(
    data_vox_t: torch.Tensor,
    pca_dim: int | None = None,
    level: float = 2.3,
) -> tuple[torch.Tensor, int]:
    """MELODIC-style voxel variance normalization via residual noise estimate.

    Mirrors FSL's ``varnorm`` logic used in MELODIC preprocessing:
    1) Compute PCA on demeaned (time, vox) data
    2) Keep top ``dim`` PCs, project data into whitened PC space
    3) Soft-threshold small PC coefficients (|coef| < level)
    4) Reconstruct denoised signal and estimate residual std per voxel
    5) Divide each voxel by residual std; zero constant voxels

    Parameters
    ----------
    data_vox_t : Tensor, shape (n_vox, n_time)
        Voxel-by-time data after detrending/high-pass preprocessing.
    pca_dim : int or None
        Number of PCs for residual model (FSL default: min(30, T-1)).
    level : float
        Threshold level in whitened PC space (FSL default: 2.3).

    Returns
    -------
    normalized : Tensor, shape (n_vox, n_time)
        Residual-variance-normalized data.
    n_constant : int
        Number of voxels treated as constant and zeroed.
    """
    # Work on a contiguous (T, V) copy so the caller's tensor is not modified.
    x_t = data_vox_t.T.clone()  # (T, V) — ORIGINAL data
    n_time, n_vox = int(x_t.shape[0]), int(x_t.shape[1])

    if n_time < 2 or n_vox < 2:
        std = torch.std(x_t, dim=0, keepdim=True)
        const_mask = std.squeeze(0) < 1e-6
        std = torch.where(const_mask.unsqueeze(0), torch.ones_like(std), std)
        out = x_t / std
        out[:, const_mask] = 0.0
        return out.T, int(const_mask.sum().item())

    # FSL MELODIC setup_classic uses min(30, T-1) for varnorm PCA dim.
    dim = min(30, max(1, n_time - 1)) if pca_dim is None else int(pca_dim)
    dim = max(1, min(dim, n_time - 1))

    # MELODIC varnorm flow (melhlprfns.cc):
    #   1. std_pca(remmean(in,2), Corr, ...) where remmean(...,2) is ROW-mean
    #      removal (spatial mean per timepoint), not temporal demeaning.
    #      Inside std_pca, cov_r again row-centres and divides by V.
    #   2. calc_white → white/dewhite from eigenvectors/eigenvalues
    #   3. ws = white * in — whitening applied to ORIGINAL data (not demeaned!)
    #   4. Threshold |ws| < level → 0
    #   5. Residual = in - dewhite * ws (ORIGINAL data)
    #   6. noise_std = stdev(residual) per voxel
    #   7. Normalize in / noise_std, zero constant voxels
    #
    # Step 1: PCA on row-centred data (FSL remmean(...,2) / cov_r semantics)
    row_mean = x_t.mean(dim=1, keepdim=True)  # (T, 1) spatial mean
    corr_t = (x_t @ x_t.T - n_vox * (row_mean @ row_mean.T)) / float(n_vox)

    # Eigendecomposition (ascending -> descending)
    evals, evecs = torch.linalg.eigh(corr_t)
    del corr_t
    order = torch.argsort(evals, descending=True)
    evals = torch.clamp(evals[order][:dim], min=1e-12)
    evecs = evecs[:, order][:, :dim]

    # Step 2: white/dewhite from PCA basis
    sqrt_evals = torch.sqrt(evals)
    white = (evecs / sqrt_evals.unsqueeze(0)).T  # (dim, T)
    dewhite = evecs * sqrt_evals.unsqueeze(0)  # (T, dim)

    # Step 3: Apply whitening to ORIGINAL data (not demeaned), matching MELODIC
    ws = white @ x_t  # (dim, V)
    # Step 4: Threshold small coefficients
    ws = torch.where(torch.abs(ws) < float(level), torch.zeros_like(ws), ws)

    # Chunked residual-std + normalization to avoid materializing full (T, V)
    # residual tensor.  Only the per-voxel std is needed, so we stream chunks.
    from fastfuncstuff.memory import estimate_chunk_size

    chunk_size = estimate_chunk_size(
        n_voxels=n_vox,
        n_timepoints=n_time,
        n_regressors=0,
        device=x_t.device,
        operation="ica_varnorm",
    )

    noise_std = torch.empty(n_vox, device=x_t.device)
    input_std = torch.std(x_t, dim=0, unbiased=True)
    n_chunks = (n_vox + chunk_size - 1) // chunk_size
    for v0 in tqdm(
        range(0, n_vox, chunk_size),
        total=n_chunks,
        desc="  Varnorm residual",
        leave=True,
        disable=n_chunks <= 1,
    ):
        v1 = min(v0 + chunk_size, n_vox)
        resid_chunk = x_t[:, v0:v1] - (dewhite @ ws[:, v0:v1])
        noise_std[v0:v1] = torch.std(resid_chunk, dim=0, unbiased=True)
        del resid_chunk

    del ws  # free (dim, V) before normalization

    const_mask = (noise_std < 0.01) | (input_std < 1e-6)
    safe_std = torch.where(const_mask, torch.ones_like(noise_std), noise_std)

    # In-place normalization (avoids allocating a second (T, V) tensor)
    x_t /= safe_std.unsqueeze(0)
    x_t[:, const_mask] = 0.0

    return x_t.T, int(const_mask.sum().item())


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


def _mp_expected_eigenvalues(n_features: int, n_samples: int) -> np.ndarray:
    """Expected eigenvalue quantiles under the Marchenko-Pastur distribution.

    Port of FSL MELODIC's ``Feta()`` function.  Returns the expected
    eigenvalue at each ordinal rank position, sorted descending
    (index 0 = largest expected eigenvalue).

    Parameters
    ----------
    n_features : int
        Number of eigenvalues (temporal dimensions).
    n_samples : int
        Effective number of spatial samples (after resels correction).

    Returns
    -------
    quantiles : ndarray of shape (n_features,)
        Expected eigenvalues, descending order.
    """
    n1 = n_features
    n2 = max(n1 + 1, n_samples)  # ensure n2 > n1

    nu = float(n1) / float(n2)
    bm = (1.0 - np.sqrt(nu)) ** 2  # MP lower bound
    bp = (1.0 + np.sqrt(nu)) ** 2  # MP upper bound

    # Evaluation grid for the survival function
    lrange = 0.9 * bm
    urange = 1.1 * bp
    n_eta = 30 * n1
    rangestep = (urange - lrange) / n_eta
    eta = lrange + rangestep * np.arange(1, n_eta + 1, dtype=np.float64)

    # Integration grid for MP density
    n_teta = 10 * n1
    stepsize = (bp - bm) / n_teta
    teta = stepsize * np.arange(1, n_teta + 1, dtype=np.float64)

    # MP density: f(x) = sqrt((x-bm)(bp-x)) / (2*pi*nu*x)  for bm <= x <= bp
    x_abs = teta + bm
    inner = np.clip(teta * (bp - bm - teta), 0.0, None)
    feta_vals = np.sqrt(inner) / (2.0 * np.pi * nu * x_abs)

    # Survival function: expected count of eigenvalues > eta[i]
    claw = np.zeros(n_eta, dtype=np.float64)
    cumval = 0.0
    j = 0
    for i in range(n_eta):
        while j < n_teta and x_abs[j] < eta[i]:
            cumval += feta_vals[j]
            j += 1
        claw[i] = max(n1 * (1.0 - stepsize * cumval), 0.0)

    # Invert survival function to get quantiles at each integer rank
    result = np.zeros(n1, dtype=np.float64)
    for i in range(n_eta - 1):
        rank_hi = int(np.floor(claw[i]))
        rank_lo = int(np.floor(claw[i + 1]))
        if rank_hi > rank_lo and 1 <= rank_hi <= n1:
            result[rank_hi - 1] = eta[i]  # 0-indexed

    # Fill gaps via interpolation (some ranks may not be resolved)
    nonzero_idx = np.where(result > 0)[0]
    if len(nonzero_idx) >= 2:
        for i in range(n1):
            if result[i] <= 0:
                result[i] = np.interp(i, nonzero_idx, result[nonzero_idx])
    elif len(nonzero_idx) == 1:
        result[:] = result[nonzero_idx[0]]
    else:
        result[:] = 1.0  # fallback

    # Clamp small values (FSL uses 5e-9)
    result = np.maximum(result, 5e-9)
    return result


def _adjust_eigenspectrum_melodic(
    raw_evals: np.ndarray,
    n_eff: int,
    verbose: bool = False,
    capture_trace: bool = False,
) -> tuple[np.ndarray, int] | tuple[np.ndarray, int, dict]:
    """Adjust eigenvalue spectrum for Minka estimation, MELODIC-style.

    Implements FSL MELODIC's ``adj_eigspec`` preprocessing:

    1. Drop the 2 smallest eigenvalues (absorb mean/trend residuals).
    2. Compute Marchenko-Pastur expected eigenvalues for the ratio
       n_features / n_eff.
    3. Divide raw eigenvalues by MP expected values (normalize noise floor).
    4. Retain eigenvalues up to 98 % cumulative variance of the *original*
       (unadjusted) spectrum.

    Parameters
    ----------
    raw_evals : ndarray
        All eigenvalues, sorted DESCENDING (largest first).
    n_eff : int
        Effective number of spatial samples.
    verbose : bool
        Print diagnostic information.

    Returns
    -------
    adj_evals : ndarray
        Adjusted, truncated eigenvalues for Minka estimation (descending).
    max_ev : int
        Number of eigenvalues retained (98 % variance cutoff).
    """
    # (1) Drop 2 smallest eigenvalues (FSL: AdjEV = in.Columns(3,Ncols).Reverse())
    # Our eigenvalues are already descending, so drop the last 2 (smallest)
    if len(raw_evals) > 4:
        ev = raw_evals[:-2].astype(np.float64).copy()
    else:
        ev = raw_evals.astype(np.float64).copy()

    n_feat = len(ev)

    # (2) Cumulative variance of ORIGINAL (unadjusted) spectrum
    ev_sum = ev.sum()
    if ev_sum > 0:
        cum_pct = np.cumsum(ev) / ev_sum
    else:
        cum_pct = np.ones(n_feat)

    # (3) Compute MP expected eigenvalues
    mp_expected = _mp_expected_eigenvalues(n_feat, n_eff)

    # (4) Divide by MP expected (remove noise floor shape)
    ev_adj = ev / mp_expected

    # Re-sort descending after adjustment
    ev_adj = np.sort(ev_adj)[::-1]

    # (5) Find cutoff at 98% cumulative variance of ORIGINAL spectrum
    threshold = 0.98
    max_ev = n_feat  # default: keep all
    for i in range(n_feat - 1):
        if cum_pct[i] < threshold <= cum_pct[i + 1]:
            # FSL adj_eigspec(): maxEV = ctr_i (1-based lower crossing index).
            # In 0-based indexing, this keeps (i + 1) elements.
            max_ev = max(1, i + 1)
            break
    if max_ev < 3:
        max_ev = n_feat // 2

    # (6) Truncate and take absolute value
    ev_adj = np.abs(ev_adj[:max_ev])

    if verbose:
        print("    MP eigenvalue adjustment (MELODIC-style):")
        print(f"      {len(raw_evals)} raw eigs → drop 2 smallest → {n_feat} eigs")
        print(
            f"      MP range: [{mp_expected[-1]:.4f}, {mp_expected[0]:.4f}] "
            f"(nu={n_feat}/{n_eff}={n_feat / n_eff:.5f})"
        )
        print(f"      98% cumvar cutoff → {max_ev} eigs retained")
        if len(ev_adj) > 2:
            print(
                f"      Adjusted range: [{ev_adj[-1]:.2f}, {ev_adj[0]:.2f}] "
                f"(ratio={ev_adj[0] / max(ev_adj[-1], 1e-15):.1f}×)"
            )

    if not capture_trace:
        return ev_adj, max_ev

    cutoff_idx_1based = int(max_ev + 1) if max_ev < n_feat else int(n_feat)
    trace = {
        "n_raw_eigs": int(len(raw_evals)),
        "n_after_drop2": int(n_feat),
        "n_eff": int(n_eff),
        "raw_after_drop2": ev.tolist(),
        "cum_pct_raw_after_drop2": cum_pct.tolist(),
        "mp_expected": mp_expected.tolist(),
        "adjusted_before_sort": (ev / mp_expected).tolist(),
        "adjusted_sorted": np.sort(ev / mp_expected)[::-1].tolist(),
        "cutoff_threshold": 0.98,
        "cutoff_max_ev": int(max_ev),
        "cutoff_idx_1based": cutoff_idx_1based,
        "adjusted_truncated_abs": ev_adj.tolist(),
    }
    return ev_adj, max_ev, trace


def _minka_assess_dimension(
    spectrum: np.ndarray,
    rank: int,
    n_samples: int,
) -> float:
    """Compute Minka (2000) log-evidence for PPCA with given rank.

    .. deprecated:: Use ``_fsl_ppca_est`` for FSL-matching behaviour.
    Kept for reference / testing.
    """
    n_features = len(spectrum)
    eps = 1e-15

    if rank < 1 or rank >= n_features:
        return -np.inf

    if spectrum[rank - 1] < eps:
        return -np.inf

    pu = -rank * log(2.0)
    for i in range(1, rank + 1):
        pu += lgamma((n_features - i + 1) / 2.0) - log(pi) * (n_features - i + 1) / 2.0

    pl = -float(np.sum(np.log(spectrum[:rank]))) * n_samples / 2.0

    v = max(eps, float(np.sum(spectrum[rank:])) / (n_features - rank))
    pv = -log(v) * n_samples * (n_features - rank) / 2.0

    m = n_features * rank - rank * (rank + 1.0) / 2.0
    pp = log(2.0 * pi) * (m + rank) / 2.0

    spectrum_hat = np.array(spectrum, dtype=np.float64, copy=True)
    spectrum_hat[rank:] = v
    pa = 0.0
    for i in range(rank):
        for j in range(i + 1, n_features):
            diff_eig = spectrum[i] - spectrum[j]
            diff_inv = 1.0 / spectrum_hat[j] - 1.0 / spectrum_hat[i]
            if diff_eig < eps or diff_inv < eps:
                pa += log(eps)
            else:
                pa += log(diff_eig * diff_inv * n_samples)

    ll = pu + pl + pv + pp - pa / 2.0 - rank * log(n_samples) / 2.0
    return ll


def _fsl_ppca_est(eigenvalues: np.ndarray, N: int) -> np.ndarray:
    """Vectorized port of FSL MELODIC's ppca_est() — Laplace evidence for all k.

    This is a line-by-line port of FSL's ``melhlprfns.cc ppca_est()``.
    It computes the Laplace-approximated log-evidence for PPCA at every
    candidate dimension k = 1…d simultaneously, using the exact same
    mathematical formulation as FSL MELODIC.

    Parameters
    ----------
    eigenvalues : ndarray of shape (d,)
        Adjusted eigenvalues (descending), as returned by
        ``_adjust_eigenspectrum_melodic``.
    N : int
        Effective number of spatial samples (n_eff = floor(n_vox/(2.5*resels))).

    Returns
    -------
    l_lap : ndarray of shape (d,)
        Laplace log-evidence at each candidate dimension k = 1…d.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    d = len(eigenvalues)

    logLambda = np.log(np.clip(eigenvalues, 1e-15, None))

    k = np.arange(1, d + 1, dtype=np.float64)  # [1, 2, ..., d]
    m = d * k - 0.5 * k * (k + 1)  # free parameters

    # --- Stiefel manifold volume (l_probU) ---
    k_rev = k[::-1].copy()  # [d, d-1, ..., 1]
    loggam = np.cumsum(np.array([lgamma(0.5 * v) for v in k_rev]))
    l_probU = -np.log(2.0) * k + loggam - np.cumsum(0.5 * np.log(pi) * k_rev)

    # --- Noise variance per dimension ---
    # tmp1(k) = sum of eigenvalues from k+1 to d  (the "noise" sum)
    cum_ev = np.cumsum(eigenvalues)
    total_ev = cum_ev[-1]
    # After FSL's Reverse/cumsum trick:
    # tmp1(k) = total - cumsum(k) = sum from k+1..d
    tmp1 = total_ev - cum_ev  # tmp1[k-1] = sum of eigs from k+1..d (0-indexed)
    # FSL: tmp1(1) = 0.95*tmp1(2)  [1-indexed → 0-indexed: tmp1[d-1] = 0.95*tmp1[d-2]]
    # After Reverse, position d in 1-based was originally position 1.
    # The final tmp1 after Reverse represents: tmp1[0]=noise_sum_for_k=1, ...
    # FSL sets the last entry (k=d) to 0.95× previous
    tmp1[-1] = 0.95 * tmp1[-2] if d >= 2 else 1e-10

    tmp3 = d - k.copy()  # (d-k) for each dimension
    tmp3[-1] = 1.0  # avoid division by zero at k=d

    tmp4 = tmp1 / tmp3  # sigma^2 for each dimension k
    # FSL clamps: if tmp4 < 0.01: tmp4 = 0.01, same for tmp3 and tmp1
    tmp4 = np.maximum(tmp4, 0.01)
    tmp3 = np.maximum(tmp3, 0.01)
    tmp1 = np.maximum(tmp1, 0.01)

    # --- l_nu: noise log-likelihood ---
    l_nu = -(N / 2.0) * (d - k) * np.log(tmp4)
    l_nu[-1] = 0.0

    # --- l_lam: signal eigenvalue log-likelihood ---
    l_lam = -(N / 2.0) * np.cumsum(logLambda)

    # --- l_Az: Hessian log-determinant (Laplace correction) ---
    # FSL builds d×d matrices for eigenvalue and precision differences,
    # then accumulates via cumsum. We replicate this exactly.
    triu = np.triu(np.ones((d, d), dtype=np.float64), k=1)  # upper tri, zero diagonal

    # t1(i,j) = lambda_i - lambda_j for j > i
    eig_row = eigenvalues[np.newaxis, :]  # (1, d)
    t1 = triu * (eig_row.T - eig_row)  # (d, d): lambda_i - lambda_j

    # t2(i,j) = 1/sigma^2_j - 1/lambda_i for j > i
    inv_sigma = (1.0 / tmp4)[:, np.newaxis] * np.ones((1, d))  # (d, d), row i = 1/sigma^2_i
    inv_lambda = np.ones((d, 1)) * (1.0 / np.clip(eigenvalues, 1e-15, None))[np.newaxis, :]
    # FSL: t2 = SP(triu, t2.t() - t3.t())
    #   t2.t()(i,j) = 1/sigma^2_j (transposed: columns become the sigma index)
    #   t3.t()(i,j) = 1/lambda_i  (transposed: rows become the lambda index)
    t2 = triu * (inv_sigma.T - inv_lambda.T)  # (d, d): 1/sigma^2_j - 1/lambda_i for j > i

    # FSL clamps non-positive to 1 before log (→ log(1) = 0)
    t1 = np.where(t1 <= 0, 1.0, t1)
    t2 = np.where(t2 <= 0, 1.0, t2)

    # sum log across columns for each row, then cumsum across rows
    row_sum = np.sum(np.log(t1), axis=1) + np.sum(np.log(t2), axis=1)
    l_Az = np.cumsum(row_sum)

    # --- Combine: Laplace evidence ---
    l_lap = l_probU + l_nu + l_Az + l_lam + 0.5 * np.log(2.0 * pi) * (m + k) - 0.5 * np.log(N) * k

    return l_lap


def _fsl_ppca_est_all(eigenvalues: np.ndarray, N: int) -> dict[str, np.ndarray]:
    """Return all FSL PPCA criteria arrays (lap/bic/mdl/rrn/aic)."""
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    d = len(eigenvalues)

    logLambda = np.log(np.clip(eigenvalues, 1e-15, None))

    k = np.arange(1, d + 1, dtype=np.float64)
    m = d * k - 0.5 * k * (k + 1)

    k_rev = k[::-1].copy()
    loggam = np.cumsum(np.array([lgamma(0.5 * v) for v in k_rev]))
    l_probU = -np.log(2.0) * k + loggam - np.cumsum(0.5 * np.log(pi) * k_rev)

    cum_ev = np.cumsum(eigenvalues)
    total_ev = cum_ev[-1]
    tmp1 = total_ev - cum_ev
    tmp1[-1] = 0.95 * tmp1[-2] if d >= 2 else 1e-10

    cum_log_ev = np.cumsum(logLambda)
    total_log_ev = cum_log_ev[-1]
    tmp2 = total_log_ev - cum_log_ev
    tmp2[-1] = tmp2[-2] if d >= 2 else 0.0

    tmp3 = d - k.copy()
    tmp3[-1] = 1.0

    tmp4 = tmp1 / tmp3
    tmp4 = np.maximum(tmp4, 0.01)
    tmp3 = np.maximum(tmp3, 0.01)
    tmp1 = np.maximum(tmp1, 0.01)

    l_nu = -(N / 2.0) * (d - k) * np.log(tmp4)
    l_nu[-1] = 0.0

    l_lam = -(N / 2.0) * np.cumsum(logLambda)

    triu = np.triu(np.ones((d, d), dtype=np.float64), k=1)
    eig_row = eigenvalues[np.newaxis, :]
    t1 = triu * (eig_row.T - eig_row)

    inv_sigma = (1.0 / tmp4)[:, np.newaxis] * np.ones((1, d))
    inv_lambda = np.ones((d, 1)) * (1.0 / np.clip(eigenvalues, 1e-15, None))[np.newaxis, :]
    t2 = triu * (inv_sigma.T - inv_lambda.T)

    t1 = np.where(t1 <= 0, 1.0, t1)
    t2 = np.where(t2 <= 0, 1.0, t2)
    row_sum = np.sum(np.log(t1), axis=1) + np.sum(np.log(t2), axis=1)
    l_Az = np.cumsum(row_sum)

    l_lap = l_probU + l_nu + l_Az + l_lam + 0.5 * np.log(2.0 * pi) * (m + k) - 0.5 * np.log(N) * k
    l_bic = l_lam + l_nu - 0.5 * np.log(N) * (m + k)

    l_lhood = (tmp2 / tmp3) - np.log(tmp1 / tmp3)
    l_rrn = -0.5 * N * k * np.log(np.cumsum(eigenvalues) / k) + l_nu
    l_aic = -(-2.0 * N * tmp3 * l_lhood + 2.0 * (1.0 + d * k + 0.5 * (k - 1.0)))
    l_mdl = -(-N * tmp3 * l_lhood + 0.5 * (1.0 + d * k + 0.5 * (k - 1.0)) * np.log(N))

    return {
        "lap": l_lap,
        "bic": l_bic,
        "mdl": l_mdl,
        "rrn": l_rrn,
        "aic": l_aic,
    }


def _fsl_first_peak_k(evidence: np.ndarray, max_k: int) -> int:
    """Replicate FSL ppca_select first-peak walk-up on normalized evidence."""
    vals = np.asarray(evidence, dtype=np.float64)
    finite = np.isfinite(vals)
    if not finite.any():
        return 1
    vmin = float(np.nanmin(vals[finite]))
    vmax = float(np.nanmax(vals[finite]))
    if vmax - vmin > 1e-15:
        vals = (vals - vmin) / (vmax - vmin)
    else:
        vals[:] = 0.5

    idx = 0
    ceiling_idx = max(0, min(int(max_k) - 1, len(vals) - 1))
    while idx < (len(vals) - 1) and vals[idx] < vals[idx + 1] and idx < ceiling_idx:
        idx += 1
    return idx + 1


def melodic_evidence_proxy_k(
    evals: np.ndarray,
    n_samples: int,
    n_features: int,
    min_k: int,
    max_k: int,
    criterion: str = "aut",
) -> tuple[int, dict]:
    """MELODIC-style dimensionality estimation via Minka (2000) Laplace approximation.

    Computes the Laplace-approximated log evidence for PPCA at each
    candidate rank k.  Following FSL MELODIC's ``ppca_select``, the
    selected k is the **first local maximum** — i.e. walk up from
    min_k while the evidence is still increasing and stop at the first
    peak.  This avoids false high-k selections caused by numerical
    artifacts in the Hessian determinant term at large rank.

    References
    ----------
    Minka T.P. (2000). Automatic Choice of Dimensionality for PCA. NIPS.
    Beckmann C.F. & Smith S.M. (2004). Probabilistic ICA for fMRI. NeuroImage.
    """
    ev = np.clip(evals.astype(np.float64), 1e-15, None)
    n_ev = len(ev)

    max_k = min(max_k, n_ev - 1) if n_ev > 1 else 1
    min_k = max(1, min(min_k, max_k))

    # Use FSL's vectorized ppca_est criteria for exact match
    all_criteria = _fsl_ppca_est_all(ev, n_samples)
    all_evidence = all_criteria["lap"]

    # Extract the k-range we care about (1-indexed k → 0-indexed array)
    k_grid = np.arange(min_k, max_k + 1)
    ll_arr = all_evidence[k_grid - 1]  # 0-indexed lookup
    ll_vals = ll_arr.tolist()

    finite_mask = np.isfinite(ll_arr)
    if not finite_mask.any():
        best_k = int(min_k)
        global_max_k = int(min_k)
    else:
        estimators = {
            "lap": _fsl_first_peak_k(all_criteria["lap"], max_k=max_k),
            "bic": _fsl_first_peak_k(all_criteria["bic"], max_k=max_k),
            "mdl": _fsl_first_peak_k(all_criteria["mdl"], max_k=max_k),
            "rrn": _fsl_first_peak_k(all_criteria["rrn"], max_k=max_k),
            "aic": _fsl_first_peak_k(all_criteria["aic"], max_k=max_k),
        }

        perc_ev = np.cumsum(ev) / max(float(np.sum(ev)), 1e-15)

        criterion = str(criterion).lower()
        if criterion == "aut":
            lap_k = estimators["lap"]
            bic_k = estimators["bic"]
            if bic_k < lap_k and perc_ev[min(bic_k - 1, len(perc_ev) - 1)] > 0.8:
                best_k = int(bic_k)
                selected_criterion = "bic"
            else:
                best_k = int(lap_k)
                selected_criterion = "lap"
        elif criterion in estimators:
            best_k = int(estimators[criterion])
            selected_criterion = criterion
        else:
            best_k = int(estimators["lap"])
            selected_criterion = "lap"

        global_max_idx = int(np.argmax(ll_arr))
        global_max_k = int(k_grid[global_max_idx])

    return best_k, {
        "k_grid": k_grid.tolist(),
        "log_evidence": ll_vals,
        "selected_k": best_k,
        "global_max_k": global_max_k if finite_mask.any() else best_k,
        "criterion": str(criterion).lower(),
        "selected_criterion": selected_criterion if finite_mask.any() else "lap",
        "estimators": estimators if finite_mask.any() else {"lap": best_k},
    }


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

    if mode in {"auto", "melodic"}:
        # FSL MELODIC preprocessing for Minka/Laplace dimensionality estimation:
        # 1. Adjust eigenvalues by dividing out the Marchenko-Pastur expected
        #    noise distribution (normalises the noise floor so it's uniform).
        # 2. Truncate at 98% cumulative variance of the original spectrum.
        # 3. Run Minka on the adjusted, truncated eigenvalues with N_eff.
        # Without the MP adjustment, noise eigenvalues retain MP-shaped
        # structure that the Laplace estimator mistakes for signal components,
        # causing massive over-estimation of dimensionality.
        if capture_ppca_trace:
            adj_ev, max_ev_retained, ppca_trace = _adjust_eigenspectrum_melodic(
                raw_evals=ev,
                n_eff=n_samples_minka,
                verbose=verbose,
                capture_trace=True,
            )
        else:
            adj_ev, max_ev_retained = _adjust_eigenspectrum_melodic(
                raw_evals=ev,
                n_eff=n_samples_minka,
                verbose=verbose,
                capture_trace=False,
            )
            ppca_trace = None
        n_adj = len(adj_ev)
        max_k_eff = min(rank_cap, n_adj - 1)
        if verbose:
            print(
                f"    Minka/MELODIC evidence scan: k in [{auto_min_components}, {max_k_eff}] "
                f"(n_samples={n_samples_minka:,}, {n_adj} adjusted eigs) ..."
            )
        k, melodic_diag = melodic_evidence_proxy_k(
            evals=adj_ev,
            n_samples=n_samples_minka,
            n_features=n_adj,
            min_k=auto_min_components,
            max_k=max_k_eff,
            criterion="aut",
        )
        # Check if first-peak hit the ceiling — evidence may still be increasing
        if k >= max_k_eff - 1:
            ll_arr = np.asarray(melodic_diag["log_evidence"])
            if len(ll_arr) >= 3 and ll_arr[-1] > ll_arr[-2]:
                print(
                    f"  ⚠ Minka first-peak was STILL INCREASING at k={k} "
                    f"(ceiling={max_k_eff}). True dimensionality may be higher. "
                    f"Consider increasing -max_auto_components."
                )
        diagnostics["mp_adjusted"] = True
        diagnostics["n_eigs_after_adj"] = n_adj
        if capture_ppca_trace and ppca_trace is not None:
            diagnostics["ppca_trace"] = {
                **ppca_trace,
                "n_eigs_for_minka": int(n_eigs_for_minka),
                "rank_cap": int(rank_cap),
                "max_k_eff": int(max_k_eff),
                "criteria": {
                    key: np.asarray(val, dtype=np.float64).tolist()
                    for key, val in _fsl_ppca_est_all(adj_ev, n_samples_minka).items()
                },
                "melodic_diag": melodic_diag,
            }
        if verbose:
            global_k = melodic_diag.get("global_max_k", k)
            if global_k != k:
                print(f"    Minka selected k={k} (first-peak; global max was at k={global_k})")
            else:
                print(f"    Minka selected k={k}")
        return k, diagnostics, {"mode": "melodic_laplace", **melodic_diag}

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
        x > 0, torch.exp(log_pdf.clamp(-700, 700)), torch.tensor(1e-32, device=x.device)
    )
    return result


def _gauss_pdf_torch(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Batched Gaussian PDF.  x: (K,V), mean: (K,1), var: (K,1)."""
    return torch.exp(-0.5 * (x - mean) ** 2 / var) / torch.sqrt(2.0 * torch.pi * var)


@torch.inference_mode()
def batch_fit_ggm(
    components_kv: torch.Tensor,
    n_iter: int = 200,
    min_mode_offset: float = 0.5,
    verbose: bool = False,
) -> dict:
    """Fit GGM to ALL ICA spatial maps simultaneously on GPU.

    This is the batched PyTorch equivalent of fit_ggm — processes K
    components × V voxels in parallel.

    Parameters
    ----------
    components_kv : Tensor of shape (K, V)
        Raw ICA spatial map values.
    n_iter : int
        Maximum EM iterations.
    min_mode_offset : float
        Minimum separation between noise mean and signal mode.
    verbose : bool
        Show tqdm progress bar over EM iterations.

    Returns
    -------
    result : dict with Tensor values, each of shape (K,) or (K, V).
    """
    device = components_kv.device
    K, V = components_kv.shape
    x = components_kv.double()  # (K, V)
    n = float(V)

    # ----- Initialization -----
    # Noise Gaussian: median and 25% of variance per component
    mu_n = x.median(dim=1).values.unsqueeze(1)  # (K, 1)
    var_n = (x.var(dim=1, unbiased=False) * 0.25).unsqueeze(1).clamp(min=1e-8)

    std_n = torch.sqrt(var_n)

    # Positive tail init
    pos_mask = x > (mu_n + std_n)  # (K, V)
    pos_count = pos_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
    pos_sum = (x * pos_mask.float()).sum(dim=1, keepdim=True)
    mu_p = (pos_sum / pos_count).clamp(min=min_mode_offset)
    pos_diff = (x - mu_p) * pos_mask.float()
    var_p = ((pos_diff**2).sum(dim=1, keepdim=True) / pos_count).clamp(min=0.1)

    # Negative tail init
    neg_x = -x
    neg_mask = neg_x > (std_n - mu_n).clamp(min=0)  # (K, V)
    neg_count = neg_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
    neg_sum = (neg_x * neg_mask.float()).sum(dim=1, keepdim=True)
    mu_ng = (neg_sum / neg_count).clamp(min=min_mode_offset)
    neg_diff = (neg_x - mu_ng) * neg_mask.float()
    var_ng = ((neg_diff**2).sum(dim=1, keepdim=True) / neg_count).clamp(min=0.1)

    # Mixing proportions  (K, 1)
    pi_n = torch.full((K, 1), 0.8, device=device, dtype=torch.float64)
    pi_p = torch.full((K, 1), 0.1, device=device, dtype=torch.float64)
    pi_ng = torch.full((K, 1), 0.1, device=device, dtype=torch.float64)

    eps_conv = log(V) / 1000.0
    old_ll = torch.full((K, 1), -1e30, device=device, dtype=torch.float64)
    converged = torch.zeros(K, dtype=torch.bool, device=device)

    iterator = range(n_iter)
    if verbose:
        iterator = tqdm(iterator, desc="  GGM EM", leave=False, unit="it")

    for it in iterator:
        # E-step
        p_noise = pi_n * _gauss_pdf_torch(x, mu_n, var_n)
        p_pos = pi_p * _gamma_pdf_torch(x, mu_p, var_p)
        p_neg = pi_ng * _gamma_pdf_torch(neg_x, mu_ng, var_ng)

        total = (p_noise + p_pos + p_neg).clamp(min=1e-30)
        r_noise = p_noise / total
        r_pos = p_pos / total
        r_neg = p_neg / total

        # Log-likelihood convergence check
        ll = total.log().sum(dim=1, keepdim=True)  # (K, 1)
        if it > 20:
            just_converged = ((ll - old_ll) < eps_conv).squeeze(1) & ~converged
            converged = converged | just_converged
            if converged.all():
                break
        old_ll = ll

        # M-step (only update components that haven't converged)
        active = ~converged  # (K,)
        if not active.any():
            break

        w_n = r_noise.sum(dim=1, keepdim=True).clamp(min=1e-8)
        w_p = r_pos.sum(dim=1, keepdim=True).clamp(min=1e-8)
        w_ng = r_neg.sum(dim=1, keepdim=True).clamp(min=1e-8)

        # Noise Gaussian
        new_mu_n = (r_noise * x).sum(dim=1, keepdim=True) / w_n
        new_var_n = ((r_noise * (x - new_mu_n) ** 2).sum(dim=1, keepdim=True) / w_n).clamp(min=1e-8)

        # Positive Gamma
        x_pos_w = x.clamp(min=0)  # avoid torch.zeros_like allocation
        x_pos_cnt = (r_pos * (x > 0).float()).sum(dim=1, keepdim=True).clamp(min=1e-8)
        mu_p_cand = (r_pos * x_pos_w).sum(dim=1, keepdim=True) / x_pos_cnt
        floor_p = (1.5 * new_mu_n + torch.sqrt(new_var_n)).clamp(min=min_mode_offset)
        new_mu_p = torch.maximum(mu_p_cand, floor_p)
        new_var_p = ((r_pos * (x - new_mu_p) ** 2).sum(dim=1, keepdim=True) / w_p).clamp(min=1e-8)

        # Negative Gamma
        x_neg_w = neg_x.clamp(min=0)  # avoid torch.zeros_like allocation
        x_neg_cnt = (r_neg * (neg_x > 0).float()).sum(dim=1, keepdim=True).clamp(min=1e-8)
        mu_ng_cand = (r_neg * x_neg_w).sum(dim=1, keepdim=True) / x_neg_cnt
        floor_ng = torch.where(
            new_mu_n < 0,
            (-1.5 * new_mu_n + torch.sqrt(new_var_n)).clamp(min=min_mode_offset),
            torch.sqrt(new_var_n).clamp(min=min_mode_offset),
        )
        new_mu_ng = torch.maximum(mu_ng_cand, floor_ng)
        new_var_ng = ((r_neg * (neg_x - new_mu_ng) ** 2).sum(dim=1, keepdim=True) / w_ng).clamp(
            min=1e-8
        )

        # Proportions
        new_pi_n = (w_n / n).clamp(1e-4, 1 - 2e-4)
        new_pi_p = (w_p / n).clamp(1e-4, 1 - 2e-4)
        new_pi_ng = (w_ng / n).clamp(1e-4, 1 - 2e-4)
        pi_sum = new_pi_n + new_pi_p + new_pi_ng
        new_pi_n = new_pi_n / pi_sum
        new_pi_p = new_pi_p / pi_sum
        new_pi_ng = new_pi_ng / pi_sum

        # Apply only to active components
        a = active.unsqueeze(1)  # (K, 1)
        mu_n = torch.where(a, new_mu_n, mu_n)
        var_n = torch.where(a, new_var_n, var_n)
        mu_p = torch.where(a, new_mu_p, mu_p)
        var_p = torch.where(a, new_var_p, var_p)
        mu_ng = torch.where(a, new_mu_ng, mu_ng)
        var_ng = torch.where(a, new_var_ng, var_ng)
        pi_n = torch.where(a, new_pi_n, pi_n)
        pi_p = torch.where(a, new_pi_p, pi_p)
        pi_ng = torch.where(a, new_pi_ng, pi_ng)

    # Free EM intermediates before final posterior computation
    del r_noise, r_pos, r_neg, old_ll
    if x.device.type == "cuda":
        torch.cuda.empty_cache()

    # Final posterior — compute sequentially to minimize peak memory
    p_noise_f = pi_n * _gauss_pdf_torch(x, mu_n, var_n)
    p_pos_f = pi_p * _gamma_pdf_torch(x, mu_p, var_p)
    p_neg_f = pi_ng * _gamma_pdf_torch(neg_x, mu_ng, var_ng)
    del neg_x  # free (K, V) float64

    total_f = (p_noise_f + p_pos_f + p_neg_f).clamp(min=1e-30)
    del p_noise_f  # only need signal components for p_signal
    p_signal = ((p_pos_f + p_neg_f) / total_f).float()  # (K, V)
    del p_pos_f, p_neg_f, total_f

    sigma_n = torch.sqrt(var_n).clamp(min=1e-8)
    z_signed = ((x - mu_n) / sigma_n).float()  # (K, V)
    del x  # free last (K, V) float64

    return {
        "z_signed": z_signed,
        "p_signal": p_signal,
        "mu_noise": mu_n.squeeze(1).float(),
        "var_noise": var_n.squeeze(1).float(),
        "mu_pos": mu_p.squeeze(1).float(),
        "var_pos": var_p.squeeze(1).float(),
        "mu_neg": mu_ng.squeeze(1).float(),
        "var_neg": var_ng.squeeze(1).float(),
        "pi_noise": pi_n.squeeze(1).float(),
        "pi_pos": pi_p.squeeze(1).float(),
        "pi_neg": pi_ng.squeeze(1).float(),
        "converged": converged,
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
    min_mode_offset: float = 0.5,
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
    x = values.astype(np.float64).ravel()
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

        # Positive Gamma (only from x > 0 effectively, but weighted)
        x_pos_weighted = np.where(x > 0, x, 0.0)
        mu_p_new = float((r_pos * x_pos_weighted).sum() / max(float((r_pos * (x > 0)).sum()), 1e-8))
        mu_p = max(mu_p_new, max(1.5 * mu_n + np.sqrt(var_n), min_mode_offset))
        var_p_new = float((r_pos * (x - mu_p) ** 2).sum() / w_p)
        var_p = max(var_p_new, 1e-8)

        # Negative Gamma (use -x)
        neg_x = -x
        x_neg_weighted = np.where(neg_x > 0, neg_x, 0.0)
        mu_ng_new = float(
            (r_neg * x_neg_weighted).sum() / max(float((r_neg * (neg_x > 0)).sum()), 1e-8)
        )
        mu_ng = max(
            mu_ng_new,
            max(-1.5 * mu_n + np.sqrt(var_n) if mu_n < 0 else np.sqrt(var_n), min_mode_offset),
        )
        var_ng_new = float((r_neg * (neg_x - mu_ng) ** 2).sum() / w_ng)
        var_ng = max(var_ng_new, 1e-8)

        # Proportions
        pi_n = float(np.clip(w_n / n, 1e-4, 1 - 2e-4))
        pi_p = float(np.clip(w_p / n, 1e-4, 1 - 2e-4))
        pi_ng = float(np.clip(w_ng / n, 1e-4, 1 - 2e-4))
        # Normalize
        pi_sum = pi_n + pi_p + pi_ng
        pi_n /= pi_sum
        pi_p /= pi_sum
        pi_ng /= pi_sum

    # Final posterior probability of signal (positive OR negative)
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
    }


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

    # Z-score relative to noise distribution
    z_signed = ((vals - mu_n) / sigma_n).astype(np.float32)

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
    }

    return z_signed, p_signal.astype(np.float32), meta


def batch_mixture_zscores(
    components_kv: torch.Tensor,
    device: torch.device | None = None,
    verbose: bool = False,
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
        result = batch_fit_ggm(components_kv[k0:k1], verbose=(verbose and k0 == 0))

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
