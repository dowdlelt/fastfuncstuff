"""Core phase regression pipeline.

Removes macrovascular BOLD contamination from gradient-echo fMRI by
regressing magnitude on phase.  Oriented vessels (pial/cerebral veins)
produce correlated magnitude and phase changes; randomly oriented
microvasculature produces only magnitude changes.  Subtracting the
phase-predicted magnitude component yields a microvascular-weighted signal.

Pipeline
--------
1. Polynomial detrending (Legendre, per run) of both magnitude and phase.
2. Optional task removal from both (TENT/FIR or canonical HRF).
3. Estimate variance ratio phi from cleaned residuals (FFT or residual).
4. Deming regression on residuals -> per-voxel slope.
5. Apply correction to original (detrended) magnitude.
6. Return corrected magnitude + diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fastfuncstuff.phasereg.deming import deming_regression, ols_regression
from fastfuncstuff.phasereg.noise import estimate_variance_ratio


@dataclass
class PhaseRegResult:
    """Results from phase regression.

    Attributes
    ----------
    magnitude_corrected : Tensor (n_voxels, n_timepoints)
        Macrovascular-suppressed magnitude signal.
    macrovascular_component : Tensor (n_voxels, n_timepoints)
        The removed component: slope * (phase - mean(phase)).
    slope : Tensor (n_voxels,)
        Deming/OLS regression slope per voxel.  Large |slope| indicates
        strong macrovascular contribution.
    intercept : Tensor (n_voxels,)
        Regression intercept per voxel.
    phi : Tensor (n_voxels,)
        Variance ratio used for Deming regression.
    r2_phase : Tensor (n_voxels,)
        Fraction of magnitude variance explained by phase.
    """

    magnitude_corrected: torch.Tensor
    macrovascular_component: torch.Tensor
    slope: torch.Tensor
    intercept: torch.Tensor
    phi: torch.Tensor
    r2_phase: torch.Tensor


def _project_out(data: torch.Tensor, design: torch.Tensor) -> torch.Tensor:
    """Project design out of data via QR decomposition.

    Parameters
    ----------
    data : (n_timepoints, n_voxels)
    design : (n_timepoints, n_regressors)

    Returns
    -------
    residual : (n_timepoints, n_voxels)
    """
    if design.shape[1] == 0:
        return data
    Q, _ = torch.linalg.qr(design)
    return data - Q @ (Q.T @ data)


def _build_task_design_tent(
    onsets_per_condition: list[list],
    tr: float,
    n_timepoints_per_run: list[int],
    window: float,
    device: torch.device,
) -> list[torch.Tensor]:
    """Build per-run TENT design matrices for task removal.

    Returns one design matrix per run (task regressors only, no polynomials).
    """
    from fastfuncstuff.design.matrices import make_tent_design

    n_conditions = len(onsets_per_condition)
    designs = []
    for run_idx, n_tp in enumerate(n_timepoints_per_run):
        cond_parts = []
        for cond_idx in range(n_conditions):
            onset_times = onsets_per_condition[cond_idx][run_idx]
            tent = make_tent_design(
                onset_times_list=[onset_times],
                bot=0.0,
                top=window,
                tr=tr,
                n_timepoints=n_tp,
                zero_edges=False,
                device=device,
            )
            cond_parts.append(tent)
        designs.append(torch.cat(cond_parts, dim=1))
    return designs


def _build_task_design_canonical(
    onsets_per_condition: list[list],
    tr: float,
    n_timepoints_per_run: list[int],
    device: torch.device,
) -> list[torch.Tensor]:
    """Build per-run canonical HRF design matrices for task removal."""
    from fastfuncstuff.design.hrf import get_spmg1_hrf
    from fastfuncstuff.design.matrices import convolve_hrf
    from fastfuncstuff.io.afni import onsets_to_tr_matrix

    n_conditions = len(onsets_per_condition)
    hrf = get_spmg1_hrf(tr)
    designs = []
    for run_idx, n_tp in enumerate(n_timepoints_per_run):
        run_onsets = [onsets_per_condition[c][run_idx] for c in range(n_conditions)]
        onset_matrix = onsets_to_tr_matrix(
            [[o] for o in run_onsets],
            n_timepoints=n_tp,
            tr=tr,
        )
        onset_tensor = torch.tensor(onset_matrix, dtype=torch.float32, device=device)
        convolved = convolve_hrf(onset_tensor, hrf, n_tp)
        designs.append(convolved)
    return designs


def phase_regress(
    magnitude: list[torch.Tensor] | torch.Tensor,
    phase: list[torch.Tensor] | torch.Tensor,
    tr: float,
    task_removal: str = "none",
    onsets_per_condition: list[list] | None = None,
    nuisance_per_run: list[torch.Tensor] | None = None,
    max_poly_degree: int = 3,
    phi: float | torch.Tensor | None = None,
    phi_method: str = "fft",
    phi_freq_range: tuple[float, float | None] = (0.1, None),
    regression: str = "deming",
    tent_window: float = 20.0,
    device: str | torch.device = "cpu",
    chunk_size: int = 50000,
    verbose: bool = False,
) -> PhaseRegResult:
    """Phase regression to suppress macrovascular BOLD contamination.

    Parameters
    ----------
    magnitude : list of Tensor or Tensor
        Magnitude time series per run, each (n_voxels, n_timepoints),
        or a single tensor for single-run data.
    phase : list of Tensor or Tensor
        Phase time series (radians, unwrapped), same shapes as magnitude.
    tr : float
        Repetition time in seconds.
    task_removal : {"none", "tent", "canonical"}
        How to remove task-correlated signal before estimating the
        regression slope.  "tent" uses TENT basis (FIR-like, model-free),
        "canonical" uses SPM canonical HRF.  "none" skips task removal
        (appropriate when NORDIC denoising was applied, or for resting state).
    onsets_per_condition : list of list of ndarray, optional
        Onset times structured as [condition][run] -> ndarray of onset times.
        Required when task_removal != "none".
    nuisance_per_run : list of Tensor, optional
        Additional per-run nuisance regressors (e.g. motion parameters),
        each (n_timepoints, n_regressors).  Projected out alongside
        polynomials.
    max_poly_degree : int
        Polynomial drift order for detrending.  Set to -1 to skip.
    phi : float, Tensor, or None
        Variance ratio for Deming regression.  If None, estimated
        automatically using phi_method.
    phi_method : {"fft", "residual"}
        How to estimate phi when phi=None.  "fft" uses out-of-band
        spectral power; "residual" uses residual variance.
    phi_freq_range : tuple (lo_hz, hi_hz)
        Frequency range for FFT-based noise estimation.
    regression : {"deming", "ols"}
        Regression method.  "deming" corrects for noise in both variables.
    tent_window : float
        TENT window duration in seconds (for task_removal="tent").
    device : str or torch.device
        PyTorch device for computation.
    chunk_size : int
        Number of voxels per processing chunk.
    verbose : bool
        Print progress information.

    Returns
    -------
    PhaseRegResult
        Corrected magnitude and diagnostic maps.
    """
    device = torch.device(device)

    # Normalise to list-of-runs
    if isinstance(magnitude, torch.Tensor):
        magnitude = [magnitude]
        phase = [phase]

    n_runs = len(magnitude)
    n_tp_per_run = [m.shape[1] for m in magnitude]
    n_voxels = magnitude[0].shape[0]

    if verbose:
        print(f"Phase regression: {n_runs} run(s), {n_voxels:,} voxels, "
              f"{sum(n_tp_per_run)} total timepoints")
        print(f"  Task removal: {task_removal}")
        print(f"  Regression: {regression}")
        print(f"  Phi method: {phi_method}")

    # ── 1. Build per-run polynomial + nuisance regressors ────────────────
    from fastfuncstuff.design.builder import legendre_polynomials

    poly_per_run: list[torch.Tensor] = []
    for run_idx, n_tp in enumerate(n_tp_per_run):
        parts = []
        if max_poly_degree >= 0:
            poly = legendre_polynomials(n_tp, max_poly_degree)
            parts.append(torch.tensor(poly, dtype=torch.float32, device=device))
        if nuisance_per_run is not None and run_idx < len(nuisance_per_run):
            nuis = nuisance_per_run[run_idx]
            if nuis.device != device:
                nuis = nuis.to(device)
            parts.append(nuis)
        if parts:
            poly_per_run.append(torch.cat(parts, dim=1))
        else:
            poly_per_run.append(torch.zeros(n_tp, 0, device=device))

    # ── 2. Build per-run task design (if needed) ─────────────────────────
    task_per_run: list[torch.Tensor | None] = [None] * n_runs
    if task_removal == "tent":
        if onsets_per_condition is None:
            raise ValueError("task_removal='tent' requires onsets_per_condition")
        task_per_run = _build_task_design_tent(
            onsets_per_condition, tr, n_tp_per_run, tent_window, device,
        )
    elif task_removal == "canonical":
        if onsets_per_condition is None:
            raise ValueError("task_removal='canonical' requires onsets_per_condition")
        task_per_run = _build_task_design_canonical(
            onsets_per_condition, tr, n_tp_per_run, device,
        )
    elif task_removal != "none":
        raise ValueError(
            f"Unknown task_removal: {task_removal!r}. "
            "Use 'none', 'tent', or 'canonical'."
        )

    # ── 3. Detrend + task-remove per run, concatenate ────────────────────
    # We keep two versions:
    #   detrended_*  : polynomials removed (for applying correction)
    #   residual_*   : polynomials + task removed (for estimating slope)
    mag_detrended_runs: list[torch.Tensor] = []
    pha_detrended_runs: list[torch.Tensor] = []
    mag_residual_runs: list[torch.Tensor] = []
    pha_residual_runs: list[torch.Tensor] = []

    for run_idx in range(n_runs):
        # Work in (n_tp, n_voxels) for regression
        mag_r = magnitude[run_idx].T.to(device)  # (n_tp, n_vox)
        pha_r = phase[run_idx].T.to(device)

        # Detrend: remove polynomials + nuisance
        poly = poly_per_run[run_idx]
        mag_dt = _project_out(mag_r, poly) if poly.shape[1] > 0 else mag_r
        pha_dt = _project_out(pha_r, poly) if poly.shape[1] > 0 else pha_r

        mag_detrended_runs.append(mag_dt.cpu())
        pha_detrended_runs.append(pha_dt.cpu())

        # Task removal for slope estimation
        task = task_per_run[run_idx]
        if task is not None:
            # Combined nuisance: poly + task
            combined = torch.cat([poly, task], dim=1) if poly.shape[1] > 0 else task
            mag_res = _project_out(mag_r, combined)
            pha_res = _project_out(pha_r, combined)
        else:
            mag_res = mag_dt
            pha_res = pha_dt

        mag_residual_runs.append(mag_res.cpu())
        pha_residual_runs.append(pha_res.cpu())

    # Concatenate across runs: (total_tp, n_voxels)
    mag_residual = torch.cat(mag_residual_runs, dim=0)
    pha_residual = torch.cat(pha_residual_runs, dim=0)
    mag_detrended = torch.cat(mag_detrended_runs, dim=0)
    pha_detrended = torch.cat(pha_detrended_runs, dim=0)

    # Free per-run intermediates
    del mag_detrended_runs, pha_detrended_runs
    del mag_residual_runs, pha_residual_runs

    # ── 4. Estimate phi (if not provided) ────────────────────────────────
    if phi is None:
        if verbose:
            print("  Estimating variance ratio phi...")
        phi_tensor = estimate_variance_ratio(
            mag_residual, pha_residual, tr,
            method=phi_method, freq_range=phi_freq_range,
        )
    elif isinstance(phi, (int, float)):
        phi_tensor = torch.full((n_voxels,), float(phi))
    else:
        phi_tensor = phi

    if verbose:
        phi_med = phi_tensor.median().item()
        print(f"  Phi: median={phi_med:.2f}, "
              f"range=[{phi_tensor.min().item():.2f}, {phi_tensor.max().item():.2f}]")

    # ── 5. Regression (chunked for memory) ───────────────────────────────
    slope_all = torch.zeros(n_voxels)
    intercept_all = torch.zeros(n_voxels)

    for start in range(0, n_voxels, chunk_size):
        end = min(start + chunk_size, n_voxels)
        chunk_mag = mag_residual[:, start:end].to(device)
        chunk_pha = pha_residual[:, start:end].to(device)
        chunk_phi = phi_tensor[start:end].to(device)

        if regression == "deming":
            s, b = deming_regression(chunk_pha, chunk_mag, chunk_phi)
        elif regression == "ols":
            s, b = ols_regression(chunk_pha, chunk_mag)
        else:
            raise ValueError(f"Unknown regression: {regression!r}")

        # Zero out slopes where magnitude-phase correlation is negligible.
        # This prevents Deming from producing extreme slopes on pure noise.
        x_c = chunk_pha - chunk_pha.mean(dim=0)
        y_c = chunk_mag - chunk_mag.mean(dim=0)
        sxy = (x_c * y_c).sum(dim=0)
        sxx = (x_c * x_c).sum(dim=0)
        syy = (y_c * y_c).sum(dim=0)
        denom = torch.sqrt(sxx * syy).clamp(min=1e-30)
        corr = (sxy / denom).abs()
        # Approximate p-value threshold: |r| > 2/sqrt(n) for rough significance
        n_tp_total = chunk_mag.shape[0]
        corr_thresh = 2.0 / (n_tp_total ** 0.5)
        sig_mask = corr > corr_thresh
        s = torch.where(sig_mask, s, torch.zeros_like(s))
        b = torch.where(sig_mask, b, chunk_mag.mean(dim=0))

        slope_all[start:end] = s.cpu()
        intercept_all[start:end] = b.cpu()

    if verbose:
        n_sig = (slope_all.abs() > 0.01).sum().item()
        print(f"  Slope: median={slope_all.median().item():.4f}, "
              f"{n_sig:,} voxels with |slope| > 0.01")

    # ── 6. Apply correction to detrended magnitude ───────────────────────
    # S_corrected(t) = S_detrended(t) - slope * (phase_detrended(t) - mean(phase))
    # The correction uses the DETRENDED (but not task-removed) phase,
    # so the full phase time series drives the correction, but the slope
    # was estimated from task-free residuals.
    pha_mean = pha_detrended.mean(dim=0)  # (n_voxels,)

    macro_component = torch.zeros_like(mag_detrended)
    mag_corrected = torch.zeros_like(mag_detrended)

    for start in range(0, n_voxels, chunk_size):
        end = min(start + chunk_size, n_voxels)
        s = slope_all[start:end]  # (chunk,)
        pha_c = pha_detrended[:, start:end] - pha_mean[start:end]  # (n_tp, chunk)
        macro = s.unsqueeze(0) * pha_c  # (n_tp, chunk)
        macro_component[:, start:end] = macro
        mag_corrected[:, start:end] = mag_detrended[:, start:end] - macro

    # ── 7. Compute R2 of phase regression ────────────────────────────────
    # R2 = 1 - SS_residual / SS_total, where residual = corrected signal
    ss_total = (mag_detrended * mag_detrended).sum(dim=0)
    ss_residual = (mag_corrected * mag_corrected).sum(dim=0)
    r2 = torch.where(
        ss_total > 1e-30,
        1.0 - ss_residual / ss_total,
        torch.zeros_like(ss_total),
    )
    r2 = r2.clamp(min=0.0, max=1.0)

    if verbose:
        print(f"  R2 (phase): median={r2.median().item():.4f}, "
              f"max={r2.max().item():.4f}")

    # ── 8. Transpose back to (n_voxels, n_timepoints) ───────────────────
    return PhaseRegResult(
        magnitude_corrected=mag_corrected.T,
        macrovascular_component=macro_component.T,
        slope=slope_all,
        intercept=intercept_all,
        phi=phi_tensor,
        r2_phase=r2,
    )
