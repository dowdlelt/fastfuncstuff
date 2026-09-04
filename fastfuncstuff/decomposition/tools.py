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

import numpy as np
import torch
from tqdm.auto import tqdm

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

    Such voxels must be dropped from the analysis mask outright, not merely zeroed.
    Zeroing leaves an exact-zero delta spike in every IC map; the mixture model's
    background Gaussian then collapses onto that spike and most of the brain gets
    labelled signal. Measured on real unsmoothed data, 5.6% constant voxels drove mean
    P(signal) on noise components to 0.70, against the ~0.07 such a component should
    give. See ``../fmri_wiki/concepts/Constant voxels break the mixture model.md``.
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
        Build it from the resel count -- see
        :func:`~fastfuncstuff.decomposition.model_order.effective_sample_size`.
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

    # Row-mean removal: subtract each timepoint's *spatial* mean before forming the
    # temporal covariance. The global signal at a timepoint is a rank-1 spatial pattern
    # shared by every voxel; leaving it in puts one huge eigenvalue at the top of the
    # spectrum that the model order then has to spend a component on.
    x_t_dev = to_tensor(x_t, device=device)
    # Row-centred covariance: subtract the per-timepoint spatial mean.
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


def component_task_glm(
    mixing_tk: torch.Tensor,
    design_tc: torch.Tensor,
    labels: list[str],
    *,
    explained_share: np.ndarray | None = None,
) -> dict:
    """Fit the whole task design to every ICA timecourse: which components hold task.

    The marginal correlation next door answers "does this component follow condition
    k", which is a contamination question.  This answers the identification one --
    "how much of the model is in this component, and which conditions carry it" --
    and the two disagree whenever a run holds several conditions.  Marginal ``r`` is
    capped at ``corr(x_k, sum_j x_j)`` because the other K-1 responses stay in the
    residual, so a component that is a NOISELESS copy of the full task response can
    still score 0.1--0.3 per condition.  That ceiling belongs to the design, not to
    the component, which is why ranking is on ``r_full`` and the per-condition
    numbers come from the JOINT fit.

    Both inputs must already carry the same temporal preprocessing (polort,
    high-pass) as the ICA data -- ``preprocess_design_for_correlation`` is what puts
    the design on that footing.  Only the mean is removed again here.

    Parameters
    ----------
    mixing_tk : (T, K)
        Component timecourses, time first.
    design_tc : (T, C)
        HRF-convolved condition regressors.
    labels : list of str
        Condition names, one per design column.
    explained_share : (K,), optional
        Each component's share of data variance, carried through to the table so the
        two numbers a reader wants to weigh -- how big and how task-like -- sit in
        the same row.

    Returns
    -------
    dict
        ``r_full`` (K,), ``r2`` (K,), ``f_stat`` (K,), ``p_value`` (K,), the joint
        ``beta`` / ``t`` / ``r_joint`` (K, C), the marginal ``r`` (K, C), the fit's
        ``dof_model`` / ``dof_resid``, and ``r2_chance`` -- the R^2 a component with
        NO task relation returns anyway, which is what keeps a table of small
        numbers readable.
    """
    from fastfuncstuff.stats.task_coupling import task_coupling

    n_t, n_k = int(mixing_tk.shape[0]), int(mixing_tk.shape[1])
    # task_coupling wants (nx,ny,nz,T) with time last; one "voxel" per component gives
    # the whole table in a single call rather than a second implementation of the
    # joint fit ([[Code reuse]]).
    field = mixing_tk.detach().to(torch.float64).T.reshape(n_k, 1, 1, n_t)
    tc = task_coupling(field, design_tc, polort=0, labels=labels)

    r_full = tc.r_full.reshape(n_k).cpu().numpy().astype(np.float64)
    r2 = r_full**2
    dof_model = int(tc.n_fit)
    # polort=0 removes the single constant column.
    dof_resid = max(1, n_t - 1 - dof_model)
    f_stat = (r2 / dof_model) / np.clip((1.0 - r2) / dof_resid, 1e-30, None)

    from scipy import stats as _st

    p_value = np.asarray(_st.f.sf(f_stat, dof_model, dof_resid), dtype=np.float64)

    r_joint = tc.r_joint.reshape(n_k, -1).cpu().numpy().astype(np.float64)
    t_joint = r_joint * np.sqrt(dof_resid / np.clip(1.0 - r_joint**2, 1e-30, None))

    return {
        "labels": list(tc.labels),
        "r_full": r_full,
        "r2": r2,
        "f_stat": f_stat,
        "p_value": p_value,
        "beta": tc.beta_joint.reshape(n_k, -1).cpu().numpy().astype(np.float64),
        "t": t_joint,
        "r_joint": r_joint,
        "r_marginal": tc.r.reshape(n_k, -1).cpu().numpy().astype(np.float64),
        "dof_model": dof_model,
        "dof_resid": dof_resid,
        # chance_share is an rms share, i.e. sqrt(R^2) under the null; square it back.
        "r2_chance": float(tc.chance_share**2),
        "explained_share": None if explained_share is None else np.asarray(explained_share),
    }
