"""Core phase regression pipeline.

Removes macrovascular BOLD contamination from gradient-echo fMRI by
regressing magnitude on phase.  Oriented vessels (pial/cerebral veins)
produce correlated magnitude and phase changes; randomly oriented
microvasculature produces only magnitude changes.  Subtracting the
phase-predicted magnitude component yields a microvascular-weighted signal.

Pipeline
--------
1. Polynomial detrending (Legendre, per run) of both magnitude and phase.
2. Optional Savitzky-Golay filtering of phase (Barry & Gore 2014).
3. Optional task removal from both (TENT/FIR or canonical HRF).
4. Signal-based voxel masking (skip air / skull / low-SNR).
5. Estimate variance ratio phi from cleaned residuals (FFT or residual).
6. Deming regression on residuals -> per-voxel slope.
7. Apply correction to detrended magnitude.
8. Return corrected magnitude + diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fastfuncstuff.phasereg.deming import deming_regression, ols_regression
from fastfuncstuff.phasereg.noise import estimate_variance_ratio
from fastfuncstuff.processing.filters import savgol_filter_1d, savgol_filter_explore


@dataclass
class PhaseRegResult:
    """Results from phase regression.

    Attributes
    ----------
    magnitude_corrected : Tensor (n_voxels, n_timepoints)
        Macrovascular-suppressed magnitude signal.  Preserves the original
        mean and slow trends — only the phase-correlated macrovascular
        fluctuations are removed.  Feed directly into a GLM.
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
        Fraction of magnitude variance explained by phase (ODR-shrinkage metric,
        matches phaseprep / Stanley 2021).
    r2_naive : Tensor (n_voxels,) or None
        Naive R²: ``1 - SS_res(observed) / SS_tot`` without ODR shrinkage.
        Highlights voxels where phase regression had the largest raw effect
        (useful for QC/visualisation).  None unless r2_mode='naive' or 'both'.
    voxel_mask : Tensor (n_voxels,), bool
        True for voxels where regression was attempted.
    mag_detrended : Tensor (n_voxels, n_timepoints) or None
        Magnitude after polynomial (+ nuisance) detrending.  None unless
        save_intermediates=True.
    pha_detrended : Tensor (n_voxels, n_timepoints) or None
        Phase after polynomial (+ nuisance) detrending.  None unless
        save_intermediates=True.
    pha_detrended_filt : Tensor (n_voxels, n_timepoints) or None
        Phase after detrending and optional SGF filtering (the series used for
        slope estimation and correction).  None unless save_intermediates=True.
    mag_residual : Tensor (n_voxels, n_timepoints) or None
        Magnitude after polynomial + task removal (used for slope fit).
        Identical to mag_detrended when task_removal='none'.  None unless
        save_intermediates=True.
    pha_residual_filt : Tensor (n_voxels, n_timepoints) or None
        Phase after polynomial + task removal + SGF (used for phi estimation
        and slope fit).  None unless save_intermediates=True.
    """

    magnitude_corrected: torch.Tensor
    macrovascular_component: torch.Tensor
    slope: torch.Tensor
    intercept: torch.Tensor
    phi: torch.Tensor
    r2_phase: torch.Tensor
    r2_naive: torch.Tensor | None
    voxel_mask: torch.Tensor
    mag_detrended: torch.Tensor | None = None
    pha_detrended: torch.Tensor | None = None
    pha_detrended_filt: torch.Tensor | None = None
    mag_residual: torch.Tensor | None = None
    pha_residual_filt: torch.Tensor | None = None


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
    """Build per-run TENT design matrices for task removal."""
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


def _signal_mask(
    mag_detrended: torch.Tensor,
    signal_thresh: float,
) -> torch.Tensor:
    """Compute a boolean mask of voxels with sufficient signal.

    Voxels whose mean |magnitude| is below signal_thresh * max(mean) are
    excluded from regression (air, skull, low-SNR).  Matches phaseprep's
    ``mm > 0.03 * max(mm)`` logic.

    Parameters
    ----------
    mag_detrended : Tensor (n_timepoints, n_voxels)
    signal_thresh : float
        Fraction of max mean signal (e.g. 0.03 = 3%).

    Returns
    -------
    mask : Tensor (n_voxels,), bool
    """
    mean_sig = mag_detrended.abs().mean(dim=0)
    thresh = signal_thresh * mean_sig.max()
    return mean_sig > thresh


def _apply_sgf(
    phase_detrended: torch.Tensor,
    sgf_mode: str,
    sgf_window: int,
    sgf_order: int,
    device: torch.device,
    chunk_size: int = 50000,
    verbose: bool = False,
    mag_detrended: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply Savitzky-Golay filtering to phase on GPU, chunked.

    Parameters
    ----------
    phase_detrended : Tensor (n_timepoints, n_voxels)
    sgf_mode : {"none", "fixed", "explore"}
    sgf_window, sgf_order : int
        Parameters for fixed mode.
    device : torch.device
    chunk_size : int
    verbose : bool

    Returns
    -------
    filtered : Tensor (n_timepoints, n_voxels)
    """
    if sgf_mode == "none":
        return phase_detrended

    n_tp = phase_detrended.shape[0]
    n_vox = phase_detrended.shape[1]
    result = phase_detrended.clone()

    if sgf_mode == "fixed":
        if verbose:
            print(f"  SGF: window={sgf_window}, order={sgf_order}")
        for start in range(0, n_vox, chunk_size):
            end = min(start + chunk_size, n_vox)
            chunk = phase_detrended[:, start:end].to(device)
            filtered = savgol_filter_1d(chunk.T, sgf_window, sgf_order).T
            result[:, start:end] = filtered.cpu()

    elif sgf_mode == "explore":
        if verbose:
            print("  SGF: data-driven per-voxel parameter search (Barry & Gore 2014)...")
        for start in range(0, n_vox, chunk_size):
            end = min(start + chunk_size, n_vox)
            chunk = phase_detrended[:, start:end].T.to(device)

            if mag_detrended is not None:
                mag_chunk = mag_detrended[:, start:end].T.to(device)
                mag_c = mag_chunk - mag_chunk.mean(dim=-1, keepdim=True)
                mag_ss = (mag_c**2).sum(dim=-1).clamp(min=1e-30)

                # mag_c/mag_ss bound as defaults: the closure is consumed within
                # this iteration (passed to savgol_filter_explore below), and the
                # binding makes that capture explicit rather than late.
                def _metric(filt: torch.Tensor, mag_c=mag_c, mag_ss=mag_ss) -> torch.Tensor:
                    f_c = filt - filt.mean(dim=-1, keepdim=True)
                    cross = (f_c * mag_c).sum(dim=-1)
                    f_ss = (f_c**2).sum(dim=-1).clamp(min=1e-30)
                    return (cross / torch.sqrt(f_ss * mag_ss)).abs()
            else:

                def _metric(filt: torch.Tensor) -> torch.Tensor:
                    return -filt.var(dim=-1)

            filtered = savgol_filter_explore(
                chunk,
                n_timepoints=n_tp,
                device=device,
                metric_fn=_metric,
            )
            result[:, start:end] = filtered.T.cpu()

    return result


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
    phase_filter: str = "none",
    sgf_window: int | None = None,
    sgf_order: int = 3,
    signal_thresh: float = 0.03,
    keep_drift: bool = False,
    device: str | torch.device = "cpu",
    chunk_size: int = 50000,
    verbose: bool = False,
    r2_mode: str = "odr",
    save_intermediates: bool = False,
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
        regression slope.
    onsets_per_condition : list of list of ndarray, optional
        Onset times structured as [condition][run] -> ndarray.
    nuisance_per_run : list of Tensor, optional
        Additional per-run nuisance regressors (e.g. motion parameters),
        each (n_timepoints, n_regressors).
    max_poly_degree : int
        Polynomial drift order for detrending.  Set to -1 to skip.
    phi : float, Tensor, or None
        Fixed variance ratio for Deming regression.  If None, estimated
        automatically.
    phi_method : {"fft", "residual"}
        How to estimate phi when phi=None.
    phi_freq_range : tuple (lo_hz, hi_hz)
        Frequency range for FFT-based noise estimation.
    regression : {"deming", "ols"}
        Regression method.
    tent_window : float
        TENT window duration in seconds.
    phase_filter : {"none", "fixed", "explore"}
        Phase filtering before regression:
        - "none": no filtering.
        - "fixed": Savitzky-Golay with sgf_window/sgf_order.
        - "explore": per-voxel data-driven SGF search (Barry & Gore 2014).
    sgf_window : int or None
        SGF window length (odd) for phase_filter="sgf".
        None = auto: ~20s of TRs (HRF-duration based), rounded to odd.
    sgf_order : int
        SGF polynomial order for phase_filter="fixed".
    signal_thresh : float
        Minimum mean-signal fraction to include a voxel (default 0.03,
        matching phaseprep). Voxels below this have their slope and
        intercept zeroed, so R² goes to 0 and corrected = mag_orig (or
        mean(mag_orig) in default mode). Set to 0 to disable; outputs
        stay numerically clean (ODR shrinkage caps every voxel) but the
        R² map will show bright values in air/noise voxels because the
        ODR-style metric reads ill-conditioned slopes as "perfect fits
        within the noise budget" — not a bug, but visually misleading.
    keep_drift : bool
        If False (default, phaseprep parity), the corrected magnitude is
        re-centered on the per-voxel mean and any polynomial drift removed
        by ``max_poly_degree`` is dropped. If True, the original drift is
        preserved in the corrected output (subtract macro from raw mag),
        leaving low-order drift for a downstream GLM to model. The macro
        magnitude removed is unaffected by this flag.
    device : str or torch.device
        PyTorch device for computation.
    chunk_size : int
        Number of voxels per processing chunk.
    verbose : bool
        Print progress information.
    r2_mode : {"odr", "naive", "both"}
        Which R² metric(s) to compute.  "odr" (default) is the ODR-shrinkage
        R² that matches phaseprep/Stanley and is the canonical metric.
        "naive" computes ``1 - SS_res(observed) / SS_tot`` without shrinkage,
        which highlights voxels with the largest raw phase-regression effect
        and is useful for QC/visualisation.  "both" returns both.
    save_intermediates : bool
        If True, populate the intermediate-data fields of PhaseRegResult
        (mag_detrended, pha_detrended, pha_detrended_filt, mag_residual,
        pha_residual_filt).  Off by default to avoid storing extra 4D arrays.

    Returns
    -------
    PhaseRegResult
        Corrected magnitude and diagnostic maps.
    """
    device = torch.device(device)

    if isinstance(magnitude, torch.Tensor):
        magnitude = [magnitude]
        phase = [phase]

    n_runs = len(magnitude)
    n_tp_per_run = [m.shape[1] for m in magnitude]
    n_voxels = magnitude[0].shape[0]

    if verbose:
        print(
            f"Phase regression: {n_runs} run(s), {n_voxels:,} voxels, "
            f"{sum(n_tp_per_run)} total timepoints"
        )
        print(f"  Task removal: {task_removal}")
        print(f"  Regression: {regression}")
        print(f"  Phi method: {phi_method}")
        print(f"  Phase filter: {phase_filter}")

        sample = magnitude[0][0]
        print(f"  Magnitude scale: mean={sample.mean().item():.1f}, std={sample.std().item():.1f}")
        sample_p = phase[0][0]
        print(f"  Phase scale: mean={sample_p.mean().item():.4f}, std={sample_p.std().item():.4f}")

    # ── 1. Keep magnitude in raw scanner units ───────────────────────────
    # phaseprep and Stanley fit ODR on raw-unit magnitude. With raw mag and
    # radian phase, var_mag/var_phase is huge → phi clamps to its max →
    # Deming → OLS, which is the regime phaseprep operates in. Earlier
    # versions normalised mag by its per-voxel mean; that put phi in the
    # tens-of-thousandths range, which sent Deming toward orthogonal
    # regression and produced exploding slopes / negative R² that the
    # downstream `r2 < 0` filter zeroed across the brain.
    mag_original_list = [magnitude[r].clone() for r in range(n_runs)]
    orig_mean = (
        torch.cat([m.mean(dim=1) for m in mag_original_list], dim=0)
        .reshape(n_runs, n_voxels)
        .mean(dim=0)
    )

    # ── 2. Build per-run polynomial + nuisance regressors ────────────────
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
            onsets_per_condition,
            tr,
            n_tp_per_run,
            tent_window,
            device,
        )
    elif task_removal == "canonical":
        if onsets_per_condition is None:
            raise ValueError("task_removal='canonical' requires onsets_per_condition")
        task_per_run = _build_task_design_canonical(
            onsets_per_condition,
            tr,
            n_tp_per_run,
            device,
        )
    elif task_removal != "none":
        raise ValueError(
            f"Unknown task_removal: {task_removal!r}. Use 'none', 'tent', or 'canonical'."
        )

    # ── 3. Detrend per run, concatenate ──────────────────────────────────
    mag_detrended_runs: list[torch.Tensor] = []
    pha_detrended_runs: list[torch.Tensor] = []
    mag_residual_runs: list[torch.Tensor] = []
    pha_residual_runs: list[torch.Tensor] = []

    for run_idx in range(n_runs):
        mag_r = magnitude[run_idx].T.to(device)
        pha_r = phase[run_idx].T.to(device)

        poly = poly_per_run[run_idx]
        mag_dt = _project_out(mag_r, poly) if poly.shape[1] > 0 else mag_r
        pha_dt = _project_out(pha_r, poly) if poly.shape[1] > 0 else pha_r

        mag_detrended_runs.append(mag_dt.cpu())
        pha_detrended_runs.append(pha_dt.cpu())

        task = task_per_run[run_idx]
        if task is not None:
            combined = torch.cat([poly, task], dim=1) if poly.shape[1] > 0 else task
            mag_res = _project_out(mag_r, combined)
            pha_res = _project_out(pha_r, combined)
        else:
            mag_res = mag_dt
            pha_res = pha_dt

        mag_residual_runs.append(mag_res.cpu())
        pha_residual_runs.append(pha_res.cpu())

    mag_residual = torch.cat(mag_residual_runs, dim=0)
    pha_residual = torch.cat(pha_residual_runs, dim=0)
    mag_detrended = torch.cat(mag_detrended_runs, dim=0)
    pha_detrended = torch.cat(pha_detrended_runs, dim=0)

    del mag_detrended_runs, pha_detrended_runs
    del mag_residual_runs, pha_residual_runs

    # ── 4. Signal-based voxel masking ────────────────────────────────────
    orig_thresh = signal_thresh * orig_mean.max()
    vox_mask = orig_mean > orig_thresh
    n_good = int(vox_mask.sum().item())
    if verbose:
        n_bad = n_voxels - n_good
        print(
            f"  Voxel mask: {n_good:,} / {n_voxels:,} voxels above "
            f"{signal_thresh:.0%} signal threshold ({n_bad:,} excluded)"
        )
        if signal_thresh == 0:
            print(
                "  WARNING: signal_thresh=0 → no signal-based gating. "
                "Output is safe (ODR shrinkage caps all voxels), but the "
                "R² map will show bright air/noise voxels."
            )

    # ── 5. Optional Savitzky-Golay filtering on phase ────────────────────
    # Applied to detrended phase (used for correction) AND to the residual
    # phase (used for slope estimation) consistently.
    if sgf_window is None:
        sgf_window = round(20.0 / tr)
        if sgf_window % 2 == 0:
            sgf_window += 1
        sgf_window = max(sgf_window, 5)
    pha_detrended_filt = _apply_sgf(
        pha_detrended,
        phase_filter,
        sgf_window,
        sgf_order,
        device,
        chunk_size,
        verbose,
        mag_detrended=mag_detrended,
    )
    pha_residual_filt = _apply_sgf(
        pha_residual,
        phase_filter,
        sgf_window,
        sgf_order,
        device,
        chunk_size,
        verbose=False,
        mag_detrended=mag_residual,
    )

    # ── 6. Estimate phi (if not provided) ────────────────────────────────
    if phi is None:
        if verbose:
            print("  Estimating variance ratio phi...")
        phi_tensor = estimate_variance_ratio(
            mag_residual,
            pha_residual_filt,
            tr,
            method=phi_method,
            freq_range=phi_freq_range,
        )
    elif isinstance(phi, (int, float)):
        phi_tensor = torch.full((n_voxels,), float(phi))
    else:
        phi_tensor = phi

    if verbose:
        phi_good = phi_tensor[vox_mask]
        phi_med = phi_good.median().item()
        print(
            f"  Phi: median={phi_med:.2f}, "
            f"range=[{phi_good.min().item():.2f}, {phi_good.max().item():.2f}]"
        )

    # ── 7. Regression (chunked for memory) ───────────────────────────────
    slope_all = torch.zeros(n_voxels)
    intercept_all = torch.zeros(n_voxels)

    pha_mean = pha_detrended_filt.mean(dim=0)
    pha_var = pha_detrended_filt.var(dim=0)

    # Phase variance floor: in voxels where the phase barely varies
    # (air, skull, or genuinely flat phase), Var(phase) → 0 makes the
    # slope = Cov/Var explode to ±billions.  Floor at 1e-6 radians²
    # (std ≈ 0.001 rad ≈ 0.06°) — well below any genuine BOLD phase
    # change, catches dead voxels.
    pha_var_floor = 1e-6
    unstable = (pha_var < pha_var_floor) & vox_mask
    if verbose and unstable.sum().item() > 0:
        print(
            f"  Phase variance floor: {unstable.sum().item():,} voxels with "
            f"Var(phase) < {pha_var_floor:.0e} (slopes zeroed)"
        )

    for start in range(0, n_voxels, chunk_size):
        end = min(start + chunk_size, n_voxels)
        chunk_mask = vox_mask[start:end]
        if not chunk_mask.any():
            continue

        chunk_mag = mag_residual[:, start:end].to(device)
        chunk_pha = pha_residual_filt[:, start:end].to(device)
        chunk_phi = phi_tensor[start:end].to(device)

        if regression == "deming":
            s, b = deming_regression(chunk_pha, chunk_mag, chunk_phi)
        elif regression == "ols":
            s, b = ols_regression(chunk_pha, chunk_mag)
        else:
            raise ValueError(f"Unknown regression: {regression!r}")

        # Zero slopes for excluded voxels and unstable (near-zero phase variance)
        bad = ~chunk_mask.to(device) | (pha_var[start:end].to(device) < pha_var_floor)
        s = torch.where(bad, torch.zeros_like(s), s)
        b = torch.where(bad, torch.zeros_like(b), b)

        slope_all[start:end] = s.cpu()
        intercept_all[start:end] = b.cpu()

    if verbose:
        n_sig = (slope_all[vox_mask].abs() > 0.01).sum().item()
        print(
            f"  Slope: median={slope_all[vox_mask].median().item():.4f}, "
            f"{n_sig:,} voxels with |slope| > 0.01"
        )

    # ── 8. Compute detrended macro component (for diagnostics) ────────────
    # R2 and diagnostics use detrended data so the fit quality is measured
    # on the same data the slope was estimated from.
    macro_detrended = torch.zeros_like(mag_detrended)

    for start in range(0, n_voxels, chunk_size):
        end = min(start + chunk_size, n_voxels)
        s = slope_all[start:end]
        pha_c = pha_detrended_filt[:, start:end] - pha_mean[start:end]
        macro_detrended[:, start:end] = s.unsqueeze(0) * pha_c

    # ── 9. ODR-style residual + R² (matches phaseprep / Stanley) ─────────
    # phaseprep's "corrected" output is the ODR residual eps, not the raw
    # observed-x residual. Algebra: with sx=stdp, sy=stdm and slope A,
    # ODR's optimal per-timepoint x-correction delta is
    #     delta_i = A·sx²·r_i / (A²sx² + sy²),
    # giving an effective output residual of
    #     eps_i = r_i / (1 + A²/φ),    where r_i = mag_dt - A·phase_dt.
    # The (1 + A²/φ) factor *shrinks* the correction in ill-conditioned
    # voxels (A² ≫ φ). Without it, a 900-magnitude slope from low-correlation
    # noise data subtracts a 90-amplitude phantom macro from a 280-mean
    # signal and produces speckle. With it, those voxels collapse to ≈ mean.
    # Both R² and the corrected output use this shrinkage so they agree
    # voxelwise with phaseprep.

    inflation = 1.0 + (slope_all * slope_all) / phi_tensor.clamp(min=1e-12)
    shrink = 1.0 / inflation  # (n_voxels,)

    mag_mean_dt = mag_detrended.mean(dim=0)  # ≈ 0 with Legendre
    ss_total = ((mag_detrended - mag_mean_dt) ** 2).sum(dim=0)
    r_obs = mag_detrended - (intercept_all.unsqueeze(0) + macro_detrended)
    eps = r_obs * shrink.unsqueeze(0)  # ODR residual
    ss_residual = (eps**2).sum(dim=0)
    r2 = torch.where(
        ss_total > 1e-30,
        1.0 - ss_residual / ss_total,
        torch.zeros_like(ss_total),
    )

    # Naive R²: observed residual without ODR shrinkage.  Larger where phase
    # regression had the greatest raw effect regardless of ill-conditioning —
    # useful for finding which voxels were most changed, not for reporting.
    r2_naive: torch.Tensor | None = None
    if r2_mode in ("naive", "both"):
        ss_res_naive = (r_obs**2).sum(dim=0)
        r2_naive = torch.where(
            ss_total > 1e-30,
            1.0 - ss_res_naive / ss_total,
            torch.zeros_like(ss_total),
        )

    if verbose:
        r2_good = r2[vox_mask]
        print(
            f"  R2 (phase, ODR-style): median={r2_good.median().item():.4f}, "
            f"mean={r2_good.mean().item():.4f}, "
            f"max={r2_good.max().item():.4f}"
        )
        if r2_naive is not None:
            r2n_good = r2_naive[vox_mask]
            print(
                f"  R2 (naive, no shrinkage): median={r2n_good.median().item():.4f}, "
                f"mean={r2n_good.mean().item():.4f}, "
                f"max={r2n_good.max().item():.4f}"
            )
        infl_good = inflation[vox_mask]
        print(
            f"  ODR inflation (1 + A²/φ): median={infl_good.median().item():.3f}, "
            f"max={infl_good.max().item():.3f}, "
            f">2: {int((infl_good > 2).sum().item()):,} voxels, "
            f">10: {int((infl_good > 10).sum().item()):,} voxels"
        )
        n_total = int(vox_mask.sum().item())
        print(f"  R2 distribution over {n_total:,} masked voxels:")
        for thresh in (0.01, 0.05, 0.10, 0.20, 0.30, 0.50):
            n_above = int((r2_good > thresh).sum().item())
            pct = 100.0 * n_above / max(n_total, 1)
            print(f"    R2 > {thresh:.2f}:  {n_above:>8,} voxels ({pct:5.2f}%)")

    # ── 10. Apply correction ─────────────────────────────────────────────
    # Two output modes:
    #
    # keep_drift=False (default, phaseprep parity):
    #   corrected = mean(mag_orig) + r_obs/(1+A²/φ)
    #   This is phaseprep's `filt = mag_pp − res.y + mm`. The shrunk OLS
    #   residual is re-centered on the per-voxel mean; any polynomial drift
    #   that was projected out is dropped. Voxels where Deming gave an
    #   ill-conditioned huge slope collapse to ≈ mean — graceful degradation.
    #
    # keep_drift=True (GLM-friendly):
    #   corrected = mag_orig − A_eff · (phase_dt − mean(phase_dt))
    #   where A_eff = A / (1 + A²/φ) is the shrunken slope.
    #   Drift preserved (downstream GLM models it), but the same shrinkage
    #   factor that prevents speckle in phaseprep's output also bounds our
    #   subtraction in ill-conditioned voxels — A_eff ≤ √φ/2, so the
    #   correction is well-behaved even when raw A reaches millions.
    #   In well-conditioned voxels (A²/φ ≪ 1) A_eff ≈ A and this reduces to
    #   the textbook M_micro = M − A·φ.
    #
    # The saved macro output is whatever was actually subtracted in the
    # active mode, so `mag_orig − corrected ≡ macro` in both.
    mag_orig_per_voxel = torch.cat([m.T for m in mag_original_list], dim=0)
    mag_orig_mean = mag_orig_per_voxel.mean(dim=0)  # (n_voxels,)
    if keep_drift:
        slope_eff = slope_all * shrink  # A / (1 + A²/φ)
        pha_centered = pha_detrended_filt - pha_mean.unsqueeze(0)
        macro_detrended = slope_eff.unsqueeze(0) * pha_centered
        mag_corrected = mag_orig_per_voxel - macro_detrended
    else:
        mag_corrected = mag_orig_mean.unsqueeze(0) + eps
        macro_detrended = mag_orig_per_voxel - mag_corrected

    # ── 11. Collect optional intermediates ────────────────────────────────
    interm_mag_dt = mag_detrended.T if save_intermediates else None
    interm_pha_dt = pha_detrended.T if save_intermediates else None
    interm_pha_dt_filt = pha_detrended_filt.T if save_intermediates else None
    interm_mag_res = mag_residual.T if save_intermediates else None
    interm_pha_res_filt = pha_residual_filt.T if save_intermediates else None

    # ── 12. Transpose back to (n_voxels, n_timepoints) ───────────────────
    return PhaseRegResult(
        magnitude_corrected=mag_corrected.T,
        macrovascular_component=macro_detrended.T,
        slope=slope_all,
        intercept=intercept_all,
        phi=phi_tensor,
        r2_phase=r2,
        r2_naive=r2_naive,
        voxel_mask=vox_mask,
        mag_detrended=interm_mag_dt,
        pha_detrended=interm_pha_dt,
        pha_detrended_filt=interm_pha_dt_filt,
        mag_residual=interm_mag_res,
        pha_residual_filt=interm_pha_res_filt,
    )
