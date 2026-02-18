#!/usr/bin/env python3

"""
ffs_ica.py - Fast run-wise whole-brain ICA sanity-check / demo CLI.

Core goals
----------
- Simple whole-brain ICA workflow with GPU acceleration when available.
- Automatic component estimation, including MELODIC-style Bayesian evidence proxy.
- Optional ICASSO stability analysis at the selected component count.
- Practical preprocessing knobs for fMRI runs:
  - optional spatial blur
  - optional percent-signal scaling
  - optional polynomial detrending
  - optional Fourier high-pass filtering
- Optional task metadata attachment via condition correlations
  (onsets/durations are used for interpretation only, not model fitting).

Notes
-----
- Multiple input runs are processed independently by default.
- Flags for future modes (`-temp_concat`, `-tensor`) are present as placeholders.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

try:
    import nibabel as nib
except ImportError:
    print("ERROR: nibabel is required. Install with: pip install nibabel")
    sys.exit(1)

# fastfuncsim imports
try:
    from fastfuncsim.afni_io import get_tr_from_file, load_afni_mask, load_nifti
    from fastfuncsim.cli_utils import parse_input_files, print_cli_header
    from fastfuncsim.ica import FastICA
    from fastfuncsim.ica_postprocess import (
        best_lag_and_r as _best_lag_and_r,
    )
    from fastfuncsim.ica_postprocess import (
        component_condition_spectral_correlations as _component_condition_spectral_correlations,
    )
    from fastfuncsim.ica_postprocess import (
        load_run_ortvec_design as _load_run_ortvec_design,
    )
    from fastfuncsim.ica_postprocess import (
        mean_abs_by_selector as _mean_abs_by_selector,
    )
    from fastfuncsim.ica_postprocess import (
        mean_z_excess_by_selector as _mean_z_excess_by_selector,
    )
    from fastfuncsim.ica_postprocess import (
        normalize_0_1 as _normalize_0_1,
    )
    from fastfuncsim.ica_postprocess import (
        prepare_depth_mask as _prepare_depth_mask,
    )
    from fastfuncsim.ica_postprocess import (
        prepare_guidance_masks as _prepare_guidance_masks,
    )
    from fastfuncsim.ica_postprocess import (
        preprocess_design_for_correlation as _preprocess_design_for_correlation,
    )
    from fastfuncsim.ica_postprocess import (
        save_corr_heatmap as _save_corr_heatmap,
    )
    from fastfuncsim.ica_postprocess import (
        save_depth_lag_plot as _save_depth_lag_plot,
    )
    from fastfuncsim.ica_postprocess import (
        save_score_heatmap as _save_score_heatmap,
    )
    from fastfuncsim.ica_postprocess import (
        save_scree_plot as _save_scree_plot,
    )
    from fastfuncsim.ica_postprocess import (
        weighted_depth_timeseries as _weighted_depth_timeseries,
    )
    from fastfuncsim.ica_tools import (
        apply_high_pass_fft,
        apply_melodic_voxel_varnorm,
        apply_polort_projection,
        batch_mixture_zscores,
        build_task_design_for_run,
        component_condition_correlations,
        estimate_ica_component_count,
        parse_num_comps_spec,
    )
    from fastfuncsim.icasso import icasso
    from fastfuncsim.utils import (
        gaussian_blur_3d,
        get_device,
        scale_to_percent_signal,
        to_tensor,
    )
except ImportError as e:
    print(f"ERROR: Could not import fastfuncsim: {e}")
    print("Make sure fastfuncsim is installed: pip install -e .")
    sys.exit(1)


def _save_components_4d(
    components_kv: np.ndarray,
    mask3d: np.ndarray,
    shape3d: tuple[int, int, int],
    affine: np.ndarray,
    out_file: Path,
):
    k, n_vox = components_kv.shape
    out = np.zeros((*shape3d, k), dtype=np.float32)
    if mask3d is None:
        if np.prod(shape3d) != n_vox:
            raise ValueError("Component size does not match full volume size")
        for i in range(k):
            out[..., i] = components_kv[i].reshape(shape3d)
    else:
        flat_mask = mask3d.reshape(-1)
        for i in range(k):
            vol = np.zeros(flat_mask.shape[0], dtype=np.float32)
            vol[flat_mask] = components_kv[i]
            out[..., i] = vol.reshape(shape3d)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out, affine), str(out_file))


def _save_component_3d(
    component_v: np.ndarray,
    mask3d: np.ndarray | None,
    shape3d: tuple[int, int, int],
    affine: np.ndarray,
    out_file: Path,
):
    n_vox = component_v.shape[0]
    if mask3d is None:
        if np.prod(shape3d) != n_vox:
            raise ValueError("Component size does not match full volume size")
        out = component_v.reshape(shape3d).astype(np.float32)
    else:
        flat_mask = mask3d.reshape(-1)
        out = np.zeros(flat_mask.shape[0], dtype=np.float32)
        out[flat_mask] = component_v.astype(np.float32)
        out = out.reshape(shape3d)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out, affine), str(out_file))


def _safe_symlink(target: Path, link_path: Path):
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    rel_target = os.path.relpath(str(target), start=str(link_path.parent))
    link_path.symlink_to(rel_target)


def _write_melodic_compat_outputs(
    compat_dir: Path,
    maps_file: Path,
    zmaps_file: Path | None,
    timecourse_file: Path,
    pca_scree_ratio: np.ndarray,
    component_explained_share_pct: np.ndarray,
    component_total_share_pct: np.ndarray,
    mixing_np: np.ndarray,
    mask3d: np.ndarray | None,
    mean3d: np.ndarray,
    shape3d: tuple[int, int, int],
    affine: np.ndarray,
    z_maps: np.ndarray | None = None,
    p_maps: np.ndarray | None = None,
    thresh_z_maps: np.ndarray | None = None,
):
    compat_dir.mkdir(parents=True, exist_ok=True)

    ic_file = zmaps_file if (zmaps_file is not None and zmaps_file.exists()) else maps_file
    _safe_symlink(ic_file, compat_dir / "melodic_IC.nii.gz")
    _safe_symlink(timecourse_file, compat_dir / "melodic_mix")
    _safe_symlink(timecourse_file, compat_dir / "melodic_Tmodes")

    nib.save(nib.Nifti1Image(mean3d.astype(np.float32), affine), str(compat_dir / "mean.nii.gz"))
    if mask3d is None:
        mask_out = np.ones(shape3d, dtype=np.float32)
    else:
        mask_out = mask3d.astype(np.float32)
    nib.save(nib.Nifti1Image(mask_out, affine), str(compat_dir / "mask.nii.gz"))

    ftmix = np.abs(np.fft.rfft(mixing_np, axis=0)) ** 2
    if ftmix.shape[0] > 1:
        ftmix = ftmix[1:, :]
    np.savetxt(compat_dir / "melodic_FTmix", ftmix, fmt="%.8f")

    icstats = np.column_stack([component_explained_share_pct, component_total_share_pct])
    np.savetxt(compat_dir / "melodic_ICstats", icstats, fmt="%.8f")
    np.savetxt(compat_dir / "eigenvalues_percent", pca_scree_ratio * 100.0, fmt="%.8f")

    if z_maps is not None and p_maps is not None:
        stats_dir = compat_dir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        n_comp = z_maps.shape[0]
        for i in range(n_comp):
            _save_component_3d(
                component_v=p_maps[i],
                mask3d=mask3d,
                shape3d=shape3d,
                affine=affine,
                out_file=stats_dir / f"probmap_{i + 1}.nii.gz",
            )
            if thresh_z_maps is not None:
                _save_component_3d(
                    component_v=thresh_z_maps[i],
                    mask3d=mask3d,
                    shape3d=shape3d,
                    affine=affine,
                    out_file=stats_dir / f"thresh_zstat{i + 1}.nii.gz",
                )


def _estimate_spatial_smoothness_resels(
    data_4d: np.ndarray,
    mask: np.ndarray | None = None,
    device: torch.device | None = None,
    verbose: bool = False,
) -> tuple[float, float]:
    """Estimate spatial smoothness — GPU-accelerated port of FSL's est_resels().

    Follows FSL MELODIC's meldata.cc est_resels() exactly:
    1. Standardize each voxel timeseries to N(0,1) with ddof=1 (FSL convention)
       — voxels with zero/negative variance are removed from the mask
    2. Compute lag-1 spatial cross-products per axis (SSminus/S2)
       using ALL timepoints (no subsampling, matching FSL)
    3. Convert autocorrelation → σ² → FWHM per axis
    4. Return product of FWHMs (= "resels per voxel")

    Uses GPU (if available) for the expensive cross-product accumulation
    by processing timepoints in chunks.

    Returns
    -------
    resels : float
        FWHM_x × FWHM_y × FWHM_z  (product of per-axis FWHMs in voxels).
        Used in FSL's formula: N_eff = n_vox / (2.5 × resels).
    fwhm_geo : float
        Geometric-mean FWHM across the three axes (for display).
    """
    n_t = data_4d.shape[-1]
    shape3d = data_4d.shape[:3]

    if device is None:
        device = torch.device("cpu")

    # --- Standardize per voxel (FSL's standardise()) ---
    # FSL uses (M-1) denominator — match with ddof=1.
    # Also: FSL removes voxels with sdsq<=0 from the mask.
    mean_t = data_4d.mean(axis=-1)  # (X,Y,Z)
    # ddof=1 to match FSL's (SSx - Sx²/M) / (M-1)
    std_t = np.std(data_4d, axis=-1, ddof=1)  # (X,Y,Z)

    # Build effective mask: exclude voxels with zero variance (FSL behavior)
    valid = std_t > 1e-10
    if mask is not None:
        valid = valid & mask
    std_safe = np.where(valid, std_t, 1.0)
    del std_t

    # Move valid mask to GPU
    mask_t = torch.as_tensor(valid, device=device, dtype=torch.bool)

    # --- Accumulate cross-products over ALL timepoints in chunks ---
    # Process in chunks to limit GPU memory (each chunk = (chunk, X, Y, Z))
    chunk_size = max(1, min(n_t, 50))  # 50 timepoints per chunk
    SSminus = [0.0, 0.0, 0.0]
    S2 = [0.0, 0.0, 0.0]

    for t_start in range(0, n_t, chunk_size):
        t_end = min(t_start + chunk_size, n_t)
        # Build standardized chunk: (chunk, X, Y, Z)
        chunk_np = np.empty((t_end - t_start, *shape3d), dtype=np.float32)
        for i, ti in enumerate(range(t_start, t_end)):
            chunk_np[i] = (data_4d[..., ti] - mean_t) / std_safe
        # Zero out invalid voxels
        chunk_np[:, ~valid] = 0.0
        R = torch.as_tensor(chunk_np, device=device, dtype=torch.float32)
        del chunk_np

        for ax in range(3):
            dim = ax + 1
            R_cur = R.narrow(dim, 1, R.shape[dim] - 1)
            R_prev = R.narrow(dim, 0, R.shape[dim] - 1)

            # Per-axis mask: both current and previous voxel must be valid
            sl_cur = [slice(None)] * 3
            sl_prev = [slice(None)] * 3
            sl_cur[ax] = slice(1, None)
            sl_prev[ax] = slice(None, -1)
            m = mask_t[tuple(sl_cur)] & mask_t[tuple(sl_prev)]  # (X', Y', Z')
            m = m.unsqueeze(0)  # (1, X', Y', Z')

            SSminus[ax] += float((R_cur * R_prev * m).sum().item())
            S2[ax] += float((0.5 * (R_cur**2 + R_prev**2) * m).sum().item())

        del R

    # Free GPU memory
    del mask_t, mean_t, std_safe, valid
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- Convert to FWHM per axis (FSL formula) ---
    FWHM = []
    for ax in range(3):
        if S2[ax] < 1e-15:
            FWHM.append(1.0)
            continue
        r = SSminus[ax] / S2[ax]
        # FSL: clamp for extreme smoothness
        r = min(abs(r), 0.99999)
        if r < 1e-10:
            FWHM.append(1.0)
            continue
        # FSL: sigmasq = -1/(4*ln(r)),  FWHM = sqrt(8*ln(2)*sigmasq)
        sigmasq = -1.0 / (4.0 * np.log(r))
        fwhm_ax = float(np.sqrt(8.0 * np.log(2.0) * sigmasq))
        FWHM.append(max(1.0, fwhm_ax))

    if verbose:
        _vprint(True, f"  FWHM per axis: X={FWHM[0]:.3f}, Y={FWHM[1]:.3f}, Z={FWHM[2]:.3f} voxels")

    resels = FWHM[0] * FWHM[1] * FWHM[2]
    fwhm_geo = float(np.cbrt(resels))
    return resels, fwhm_geo


def _auto_mask(data: np.ndarray, verbose: bool = False) -> np.ndarray:
    """Compute a brain mask from mean intensity (Otsu-style threshold).

    Uses the temporal mean image, then threshold at 10% of the 98th
    percentile (robust max) to separate brain from background.
    """
    mean_img = data.mean(axis=-1)
    robust_max = float(np.percentile(mean_img[mean_img > 0], 98)) if (mean_img > 0).any() else 1.0
    thresh = robust_max * 0.10
    mask = mean_img > thresh
    if verbose:
        n_total = int(np.prod(mask.shape))
        n_brain = int(mask.sum())
        print(
            f"    Auto-mask: {n_brain:,} / {n_total:,} voxels "
            f"({100 * n_brain / max(1, n_total):.1f}%) thresh={thresh:.2f}"
        )
    return mask


def _check_finite(t: torch.Tensor, label: str, verbose: bool = False) -> torch.Tensor:
    """Replace NaN/Inf with 0, warn if any found."""
    bad = ~torch.isfinite(t)
    n_bad = int(bad.sum())
    if n_bad > 0:
        print(f"  ⚠ {label}: {n_bad:,} NaN/Inf values → zeroed")
        t = t.clone()
        t[bad] = 0.0
    elif verbose:
        print(f"    {label}: finite ✓")
    return t


def _vsection(verbose: bool, name: str):
    """Print a visible section header in verbose mode."""
    if not verbose:
        return
    print(f"\n  ── {name} {'─' * max(1, 50 - len(name))}")


def _vprint(verbose: bool, msg: str, t0: float | None = None):
    """Conditional verbose print with optional elapsed time."""
    if not verbose:
        return
    if t0 is not None:
        elapsed = time.time() - t0
        print(f"    {msg} [{elapsed:.2f}s]")
    else:
        print(f"    {msg}")


def _run_single_ica(
    run_file: str,
    run_idx: int,
    args,
    device: torch.device,
    shared_mask: np.ndarray | None,
    onsets_files: list[str] | None,
    durations: list[str] | None,
    ortvec_specs: list[list[str]] | None = None,
) -> dict:
    t_total = time.time()
    t_step = time.time()
    run_tag = f"run{run_idx + 1:02d}"

    _vsection(args.verbose, "Load Data")
    _vprint(args.verbose, f"Loading {run_file} ...")
    img = load_nifti(run_file)
    data = img.get_fdata(dtype=np.float32)
    data_unblurred = data.copy() if (args.depth_lag and args.depth_lag_use_unsmoothed) else None
    affine = img.affine
    shape3d = data.shape[:3]
    n_t = data.shape[3]
    voxel_sizes = tuple(float(v) for v in img.header.get_zooms()[:3])
    _vprint(args.verbose, f"Loaded: shape={data.shape}, dtype={data.dtype}", t_step)

    tr = float(args.tr) if args.tr is not None else float(get_tr_from_file(run_file))
    _vprint(args.verbose, f"TR = {tr:.4f}s, duration = {tr * n_t:.1f}s")

    if shared_mask is not None and shared_mask.shape != shape3d:
        raise ValueError(
            f"Mask shape {shared_mask.shape} does not match run shape {shape3d} for {run_file}"
        )

    # --- Spatial blur ---
    if args.do_blur is not None and args.do_blur > 0:
        _vsection(args.verbose, "Spatial Blur")
        t_step = time.time()
        _vprint(args.verbose, f"FWHM={args.do_blur:.1f} mm ...")
        data = gaussian_blur_3d(
            data=data,
            fwhm_mm=float(args.do_blur),
            voxel_sizes=voxel_sizes,
            device=device,
            verbose=args.verbose,
        )
        _vprint(args.verbose, "Spatial blur done", t_step)

    # Save temporal mean in image space for compatibility outputs
    mean3d = data.mean(axis=-1).astype(np.float32)

    # --- Masking ---
    _vsection(args.verbose, "Masking")
    t_step = time.time()
    if shared_mask is not None:
        mask3d = shared_mask
        _vprint(args.verbose, f"Using provided mask: {int(mask3d.sum()):,} voxels")
    elif not args.no_auto_mask:
        # Auto-mask: threshold on mean intensity to exclude background
        mask3d = _auto_mask(data, verbose=args.verbose)
    else:
        mask3d = None  # no masking at all
        _vprint(args.verbose, f"No mask: using all {np.prod(shape3d):,} voxels (no_auto_mask)")

    # Optional mask blurring to match data blur support at edges.
    # Applies only when spatial blur is enabled and a mask exists.
    if args.blur_mask and (args.do_blur is not None and args.do_blur > 0) and mask3d is not None:
        mask_orig = mask3d.copy()
        n_before = int(mask_orig.sum())
        mask_4d = mask3d.astype(np.float32)[..., np.newaxis]
        mask_blurred = gaussian_blur_3d(
            data=mask_4d,
            fwhm_mm=float(args.do_blur),
            voxel_sizes=voxel_sizes,
            device=device,
            verbose=False,
        )[..., 0]
        # Note: blur+threshold alone can erode boundaries. Preserve all original
        # mask voxels and allow only expansion from the blurred mask.
        blur_mask_thresh = 0.33
        mask3d = np.logical_or(mask_orig, mask_blurred > blur_mask_thresh)
        n_after = int(mask3d.sum())
        _vprint(
            args.verbose,
            f"Blur-mask enabled: {n_before:,} -> {n_after:,} voxels "
            f"(orig ∪ (blurred > {blur_mask_thresh}))",
        )

    if mask3d is not None:
        data_vox_t_np = data[mask3d].astype(np.float32)
    else:
        data_vox_t_np = data.reshape(-1, n_t).astype(np.float32)
    n_vox_masked = data_vox_t_np.shape[0]
    _vprint(args.verbose, f"Masked data: ({n_vox_masked:,} vox, {n_t} time)", t_step)

    if n_vox_masked < 100:
        raise ValueError(
            f"Only {n_vox_masked} voxels after masking — data appears empty or mask is wrong"
        )

    # --- Spatial guidance masks (good / bad / depth) ---
    _vsection(args.verbose, "Spatial Guidance Inputs")
    guidance_good_masks = _prepare_guidance_masks(
        mask_paths=args.good_mask,
        kind="good",
        shape3d=shape3d,
        brain_mask3d=mask3d,
        n_vox_masked=n_vox_masked,
        verbose=args.verbose,
    )
    guidance_bad_masks = _prepare_guidance_masks(
        mask_paths=args.bad_mask,
        kind="bad",
        shape3d=shape3d,
        brain_mask3d=mask3d,
        n_vox_masked=n_vox_masked,
        verbose=args.verbose,
    )
    depth_mask_info = _prepare_depth_mask(
        depth_mask_path=args.depth_mask,
        shape3d=shape3d,
        brain_mask3d=mask3d,
        n_vox_masked=n_vox_masked,
    )
    _vprint(
        args.verbose,
        f"good masks={len(guidance_good_masks)}, bad masks={len(guidance_bad_masks)}, "
        f"depth labels={0 if depth_mask_info is None else len(depth_mask_info['labels'])}",
    )

    # Source matrix for depth-lag analysis (post-mask, optional temporal matching).
    depth_source_vox_t_np: np.ndarray | None = None
    if args.depth_lag and depth_mask_info is not None:
        source_4d = data_unblurred if data_unblurred is not None else data
        if mask3d is not None:
            depth_source_vox_t_np = source_4d[mask3d].astype(np.float32)
        else:
            depth_source_vox_t_np = source_4d.reshape(-1, n_t).astype(np.float32)

    # Done with optional unblurred copy once depth source is prepared.
    if data_unblurred is not None:
        del data_unblurred

    # --- Estimate spatial smoothness for effective DOF (MELODIC-style) ---
    # FSL MELODIC computes resels = FWHM_x × FWHM_y × FWHM_z from the data,
    # then: N_eff = n_vox / (2.5 × resels).  This accounts for spatial
    # autocorrelation so the Minka/Laplace estimator doesn't over-count DOF.
    _vsection(args.verbose, "Spatial Smoothness")
    t_step = time.time()
    if args.smoothness_fwhm is not None:
        # User-supplied smoothness in mm → convert to voxels (isotropic)
        voxel_sizes = tuple(float(v) for v in img.header.get_zooms()[:3])
        mean_vox = float(np.mean(voxel_sizes))
        fwhm_vox = max(1.0, args.smoothness_fwhm / mean_vox)
        resels = fwhm_vox**3  # isotropic assumption
        fwhm_geo = fwhm_vox
        _vprint(
            args.verbose,
            f"User smoothness: {args.smoothness_fwhm:.1f} mm = {fwhm_vox:.2f} voxels "
            f"(voxel size={mean_vox:.2f} mm), resels={resels:.1f}",
        )
    else:
        # Estimate from data (must happen before data is freed)
        resels, fwhm_geo = _estimate_spatial_smoothness_resels(
            data,
            mask=mask3d,
            device=device,
            verbose=args.verbose,
        )
        _vprint(
            args.verbose,
            f"Estimated spatial FWHM: {fwhm_geo:.2f} voxels (resels={resels:.2f})",
        )

    # FSL formula: N_eff = n_vox / (2.5 × resels)
    # where resels = FWHM_x × FWHM_y × FWHM_z (product, NOT geometric mean)
    # Floor at n_time to avoid degenerate cases
    n_eff = max(n_t, int(n_vox_masked / (2.5 * resels)))
    _vprint(
        args.verbose,
        f"Effective spatial DOF: {n_eff:,} "
        f"(raw={n_vox_masked:,}, correction=2.5×{resels:.1f}={2.5 * resels:.1f})",
        t_step,
    )

    # --- To GPU ---
    t_step = time.time()
    data_vox_t = to_tensor(data_vox_t_np, device=device)
    _vprint(args.verbose, f"Data on {device}", t_step)

    # Free large numpy array
    del data_vox_t_np, data

    # --- Percent-signal scaling ---
    if args.do_scale:
        _vsection(args.verbose, "Scaling")
        t_step = time.time()
        _vprint(args.verbose, "Percent-signal scaling ...")
        data_vox_t, _, _ = scale_to_percent_signal(data_vox_t, run_starts=[0], verbose=args.verbose)
        data_vox_t = _check_finite(data_vox_t, "post-scale", args.verbose)
        _vprint(args.verbose, "Scaling done", t_step)

    # --- Polort detrending ---
    polort = 0 if args.polort is None else int(args.polort)

    _vsection(args.verbose, "Polort Detrend")
    t_step = time.time()
    _vprint(args.verbose, f"Order={polort} ...")
    data_vox_t = apply_polort_projection(data_vox_t, polort=polort, device=device)
    data_vox_t = _check_finite(data_vox_t, "post-polort", args.verbose)
    _vprint(args.verbose, "Polort done", t_step)

    # --- High-pass filter ---
    if args.high_pass is not None and args.high_pass > 0:
        _vsection(args.verbose, "High-Pass Filter")
        nyquist = 0.5 / tr
        if args.high_pass >= nyquist:
            raise ValueError(
                f"High-pass cutoff ({args.high_pass:.4f} Hz) >= Nyquist ({nyquist:.4f} Hz, TR={tr}s). "
                f"This would remove ALL data. "
                f"If you meant {args.high_pass:.0f} seconds, use: -high_pass_s {args.high_pass:.0f}"
            )
        t_step = time.time()
        period_s = 1.0 / args.high_pass
        _vprint(
            args.verbose,
            f"FFT high-pass: {args.high_pass:.6f} Hz ({period_s:.1f}s period, Nyquist={nyquist:.4f} Hz) ...",
        )
        data_vox_t = apply_high_pass_fft(data_vox_t, tr=tr, high_pass_hz=args.high_pass)
        data_vox_t = _check_finite(data_vox_t, "post-highpass", args.verbose)
        _vprint(args.verbose, "High-pass done", t_step)

    # --- Sanity: check data variance ---
    data_var = float(torch.var(data_vox_t).item())
    if data_var < 1e-10:
        raise ValueError(
            f"Data variance is ~0 after preprocessing ({data_var:.2e}). "
            "Check mask, scaling, and filter settings."
        )
    _vprint(args.verbose, f"Data variance after preprocessing: {data_var:.4f}")

    # Parse component mode before voxel normalization so we can
    # choose the exact preprocessing path for MELODIC-like estimation.
    num_spec = parse_num_comps_spec(args.num_comps)

    # --- Voxel-wise variance normalization ---
    # MELODIC's "variance normalize timecourses": divide each voxel's
    # timeseries by its temporal std so ICA focuses on temporal dynamics
    # rather than being dominated by high-amplitude voxels.
    # Also excludes constant/near-constant voxels (MELODIC step 3).
    if args.voxel_norm:
        _vsection(args.verbose, "Voxel Variance Normalization")
        t_step = time.time()
        if isinstance(num_spec, str) and num_spec in {"auto", "melodic"}:
            data_vox_t, n_const = apply_melodic_voxel_varnorm(
                data_vox_t=data_vox_t,
                pca_dim=min(30, max(1, n_t - 1)),
                level=2.3,
            )
            norm_msg = (
                f"Voxel-norm: MELODIC residual varnorm over {n_vox_masked:,} voxels "
                f"(level=2.3, pca_dim={min(30, max(1, n_t - 1))}, {n_const} constant voxels zeroed)"
            )
        else:
            voxel_std = torch.std(data_vox_t, dim=1, keepdim=True)
            const_mask = voxel_std.squeeze() < 1e-6
            n_const = int(const_mask.sum())
            safe_std = torch.where(const_mask.unsqueeze(1), torch.ones_like(voxel_std), voxel_std)
            data_vox_t = data_vox_t / safe_std
            data_vox_t[const_mask] = 0.0
            norm_msg = (
                f"Voxel-norm: divided {n_vox_masked:,} voxels by temporal stdev "
                f"({n_const} constant voxels zeroed, legacy path)"
            )
        data_vox_t = _check_finite(data_vox_t, "post-voxel-norm", args.verbose)
        _vprint(args.verbose, norm_msg, t_step)

    # --- Component count estimation ---
    _vsection(args.verbose, "Component Estimation")
    t_step = time.time()

    # Resolve max_auto_components: proportion → absolute integer
    if args.max_auto_components <= 1.0:
        max_auto_k = max(5, int(n_t * args.max_auto_components))
        _vprint(
            args.verbose,
            f"max_auto_components: {args.max_auto_components:.0%} of {n_t} timepoints = {max_auto_k}",
        )
    else:
        max_auto_k = int(args.max_auto_components)
        _vprint(args.verbose, f"max_auto_components: {max_auto_k} (absolute)")

    _vprint(args.verbose, f"Method: {args.num_comps} ...")
    n_components, pca_diag, num_diag = estimate_ica_component_count(
        data_vox_t=data_vox_t,
        method=num_spec,
        max_auto_components=max_auto_k,
        auto_min_components=args.auto_min_components,
        auto_var_threshold=args.auto_var_threshold,
        use_mp_prior=not args.auto_no_mp,
        n_eff=n_eff,
        device=device,
        verbose=args.verbose,
        capture_ppca_trace=bool(args.ppca_debug_dump),
    )

    ppca_debug_file = None
    if args.ppca_debug_dump and "ppca_trace" in pca_diag:
        ppca_base = Path(args.ppca_debug_dump)
        if ppca_base.suffix.lower() == ".json":
            ppca_debug_file = ppca_base.with_name(f"{ppca_base.stem}_{run_tag}{ppca_base.suffix}")
        else:
            ppca_debug_file = ppca_base / f"ppca_trace_{run_tag}.json"
        ppca_debug_file.parent.mkdir(parents=True, exist_ok=True)
        ppca_payload = {
            "run_index": int(run_idx + 1),
            "input_file": run_file,
            "n_voxels": int(n_vox_masked),
            "n_timepoints": int(n_t),
            "n_eff": int(n_eff),
            "num_comps_request": args.num_comps,
            "rank_cap": int(pca_diag["rank_cap"]),
            "n_eigs": int(pca_diag["n_eigs"]),
            "num_comps_diagnostics": num_diag,
            "ppca_trace": pca_diag["ppca_trace"],
        }
        with open(ppca_debug_file, "w", encoding="utf-8") as f:
            json.dump(ppca_payload, f, indent=2)
        _vprint(args.verbose, f"PPCA debug trace: {ppca_debug_file}")
    _vprint(
        args.verbose,
        f"Selected {n_components} components (mode={num_diag.get('mode', '?')})",
        t_step,
    )

    x_t = data_vox_t.T  # (time, vox)
    del data_vox_t  # free GPU memory before ICA

    # --- ICA ---
    _vsection(args.verbose, "ICA Decomposition")
    t_step = time.time()
    pca_eigenvalues = None  # PCA explained_variance_ for IC variance computation
    pca_components_for_sort = None  # PCA spatial components (k, V)
    if args.icasso:
        _vprint(args.verbose, f"ICASSO: {args.icasso_runs} runs, k={n_components} ...")
        icasso_res = icasso(
            X=x_t,
            n_components=n_components,
            n_runs=args.icasso_runs,
            pca_components=n_components,
            min_stability=args.icasso_min_stability,
            device=device,
            verbose=args.verbose,
            batch_size=args.icasso_batch_size,
        )
        components = torch.as_tensor(
            icasso_res["all_centroids"], device=device, dtype=torch.float32
        )
        mixing = torch.as_tensor(icasso_res["all_mixing"], device=device, dtype=torch.float32)
        stability = np.asarray(icasso_res["all_stability"], dtype=np.float32)
        icasso_meta = {
            "enabled": True,
            "icasso_runs": int(args.icasso_runs),
            "min_stability": float(args.icasso_min_stability),
            "n_stable": int(icasso_res["n_stable"]),
            "stability": stability.tolist(),
        }
        # Get PCA info for IC variance computation
        if icasso_res.get("pca_eigenvalues") is not None:
            pca_eigenvalues = torch.as_tensor(
                icasso_res["pca_eigenvalues"][:n_components], device=device, dtype=torch.float32
            )
        if icasso_res.get("pca_components") is not None:
            pca_components_for_sort = torch.as_tensor(
                icasso_res["pca_components"][:n_components], device=device, dtype=torch.float32
            )
        _vprint(args.verbose, f"ICASSO done ({icasso_res['n_stable']} stable)", t_step)
    else:
        _vprint(args.verbose, f"FastICA: k={n_components}, max_iter={args.ica_max_iter} ...")
        ica = FastICA(
            n_components=n_components,
            pca_components=n_components,
            max_iter=args.ica_max_iter,
            tol=args.ica_tol,
            random_state=run_idx,
            whiten=False,
            device=device,
        )
        ica.fit(x_t)
        _vprint(args.verbose, f"FastICA converged in {ica.n_iter_} iterations", t_step)

        # Check convergence
        if ica.n_iter_ >= args.ica_max_iter:
            print(f"  ⚠ FastICA did NOT converge in {args.ica_max_iter} iterations")

        components = ica.components_.to(device)
        mixing = ica.mixing_.to(device)
        stability = None
        icasso_meta = {"enabled": False, "fastica_iterations": int(ica.n_iter_)}

        # Get PCA info from FastICA's internal PCA
        if hasattr(ica, "pca_") and ica.pca_ is not None:
            pca_eigenvalues = ica.pca_.explained_variance_[:n_components].to(device)
            pca_components_for_sort = ica.pca_.components_[:n_components].to(device)
        # Free the ICA object (holds full PCA state on GPU)
        del ica

    # NOTE: x_t is kept for MELODIC-style noise normalization (freed after that step)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- Check ICA outputs for NaN ---
    n_nan_comp = int((~torch.isfinite(components)).sum())
    n_nan_mix = int((~torch.isfinite(mixing)).sum())
    if n_nan_comp > 0 or n_nan_mix > 0:
        print(f"  ⚠ ICA produced NaN/Inf: components={n_nan_comp}, mixing={n_nan_mix}")
        print(f"    This often means too many components ({n_components}) for the data.")
        print("    Try reducing -max_auto_components or use a fixed -num_comps.")
        components = torch.nan_to_num(components, nan=0.0, posinf=0.0, neginf=0.0)
        mixing = torch.nan_to_num(mixing, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Sign consistency before sorting (FSL MELODIC convention) ---
    # FSL flips component sign if abs(maxneg) > maxpos, then sorts.
    max_abs = torch.max(torch.abs(components), dim=1).values
    max_pos = torch.max(components, dim=1).values
    flip_mask = max_abs > max_pos
    n_flipped = int(flip_mask.sum().item())
    if n_flipped > 0:
        components = components.clone()
        mixing = mixing.clone()
        components[flip_mask] *= -1.0
        mixing[:, flip_mask] *= -1.0
        _vprint(args.verbose, f"Sign-flipped {n_flipped} ICs to positive-maximum orientation")

    # --- Compute ordering metrics ---
    # FSL meldata.cc sorts by spatial-map stdev. Here we expose both stdev-share
    # and an explained-share proxy, and sort by explained-share per user request.
    ic_stdev = torch.std(components, dim=1)  # (k,)
    total_stdev = float(ic_stdev.sum().item())
    if total_stdev <= 1e-15:
        stdev_share = torch.full_like(ic_stdev, 1.0 / max(1, ic_stdev.numel()))
    else:
        stdev_share = ic_stdev / total_stdev

    # Explained-share from component timecourse variance in retained PCA subspace.
    # With whitened PCA scores this becomes nearly uniform by construction;
    # ffs_ica now uses whiten=False so this remains informative.
    mix_var = torch.var(mixing, dim=0, unbiased=False)
    total_mix_var = float(mix_var.sum().item())
    if total_mix_var <= 1e-15:
        explained_share_t = torch.full_like(mix_var, 1.0 / max(1, mix_var.numel()))
    else:
        explained_share_t = mix_var / total_mix_var

    # Sort by explained-share (descending)
    sort_idx = torch.argsort(explained_share_t, descending=True)
    explained_share = explained_share_t[sort_idx].detach().cpu().numpy().astype(np.float32)
    stdev_share_sorted = stdev_share[sort_idx].detach().cpu().numpy().astype(np.float32)

    # Convert explained-share to total-share using retained PCA variance.
    scree = np.asarray(pca_diag["scree_ratio"], dtype=np.float64)
    retained_frac = float(np.clip(np.sum(scree[:n_components]), 0.0, 1.0))
    total_share = (explained_share * retained_frac).astype(np.float32)

    _vprint(
        args.verbose,
        f"IC explained-share: top3 = {explained_share[0] * 100:.2f}%, {explained_share[1] * 100:.2f}%, {explained_share[2] * 100:.2f}% "
        f"(sorted by explained-share)",
    )
    _vprint(
        args.verbose,
        f"IC total-share: top3 = {total_share[0] * 100:.2f}%, {total_share[1] * 100:.2f}%, {total_share[2] * 100:.2f}% "
        f"(retained PCA variance={retained_frac * 100:.2f}%)",
    )
    components = components[sort_idx, :]
    mixing = mixing[:, sort_idx]

    if stability is not None:
        stability = stability[sort_idx.detach().cpu().numpy()]

    # Free large GPU tensors no longer needed after sorting
    del sort_idx
    if pca_components_for_sort is not None:
        del pca_components_for_sort
    if pca_eigenvalues is not None:
        del pca_eigenvalues
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.var_norm:
        mixing = mixing - mixing.mean(dim=0, keepdim=True)
        mixing_std = torch.clamp(mixing.std(dim=0, keepdim=True), min=1e-8)
        mixing = mixing / mixing_std

    # --- MELODIC-style noise normalization ---
    # FSL meldata.cc save(): IC_norm = IC * diagvals * stdNoisei
    #   diagvals  = pow(diag(unmix * unmix^T), -0.5)       -- per-component
    #   stdNoisei = pow(stdev(Data-mix*IC)*sqrt((T-1)/(T-K)), -1)  -- per-voxel
    # This converts raw IC maps into z-score-like units that reflect
    # the signal-to-noise ratio at each voxel.
    if x_t is not None:
        _vprint(args.verbose, "Applying MELODIC-style noise normalization ...")
        try:
            # x_t: (T, V), mixing: (T, K), components: (K, V)
            T, K = mixing.shape
            unmix = torch.linalg.pinv(mixing)  # (K, T)
            # Per-component scaling from unmixing matrix diagonal
            diagvals = 1.0 / torch.sqrt(torch.clamp(torch.diag(unmix @ unmix.T), min=1e-12))  # (K,)
            # Per-voxel residual noise
            residuals = x_t - mixing @ components  # (T, V)
            resid_std = torch.std(residuals, dim=0)  # (V,)
            del residuals
            # FSL clamps small residuals: if(resids(1,ctr) < 0.05) resids(1,ctr) = 1
            resid_std = torch.where(resid_std < 0.05, torch.ones_like(resid_std), resid_std)
            # DOF-corrected noise precision: 1 / (resid_std * sqrt((T-1)/(T-K)))
            dof_factor = float(np.sqrt((T - 1.0) / max(T - K, 1.0)))
            stdNoisei = 1.0 / (resid_std * dof_factor)  # (V,)
            # Apply to spatial maps: IC_norm = IC * diagvals * stdNoisei
            components = components * (diagvals.unsqueeze(1) * stdNoisei.unsqueeze(0))
            del unmix, diagvals, resid_std, stdNoisei
            _vprint(args.verbose, "  Noise normalization applied (MELODIC convention)")
        except Exception as e:
            _vprint(args.verbose, f"  ⚠ Noise normalization failed: {e}, using raw IC maps")

    # Free ICA input matrix now that noise normalization is done
    del x_t
    if device.type == "cuda":
        torch.cuda.empty_cache()

    condition_corr = None
    condition_spectral_corr = None
    cond_labels = None
    cond_durations = None
    ortvec_corr = None
    ortvec_spectral_corr = None
    ortvec_labels = None
    if onsets_files is not None and durations is not None:
        try:
            design_tc, cond_labels, cond_durations = build_task_design_for_run(
                onsets_files=onsets_files,
                durations_arg=durations,
                run_idx=run_idx,
                onset_row=args.onset_row,
                n_timepoints=n_t,
                tr=tr,
                microtime_dt=args.microtime_dt,
                device=device,
            )
            design_tc = _preprocess_design_for_correlation(
                design_tc=design_tc,
                tr=tr,
                polort=polort,
                high_pass_hz=args.high_pass,
                device=device,
            )
            condition_corr = component_condition_correlations(mixing_tk=mixing, design_tc=design_tc)
            condition_spectral_corr = _component_condition_spectral_correlations(
                mixing_tk=mixing,
                design_tc=design_tc,
            )
        except Exception as e:
            print(f"  Warning: Could not compute condition correlations for run {run_idx + 1}: {e}")

    if ortvec_specs is not None:
        try:
            ort_tc, ortvec_labels = _load_run_ortvec_design(
                ortvec_specs=ortvec_specs,
                run_idx=run_idx,
                n_timepoints=n_t,
                device=device,
            )
            if ort_tc is not None:
                ort_tc = _preprocess_design_for_correlation(
                    design_tc=ort_tc,
                    tr=tr,
                    polort=polort,
                    high_pass_hz=args.high_pass,
                    device=device,
                )
                ortvec_corr = component_condition_correlations(mixing_tk=mixing, design_tc=ort_tc)
                ortvec_spectral_corr = _component_condition_spectral_correlations(
                    mixing_tk=mixing,
                    design_tc=ort_tc,
                )
        except Exception as e:
            print(f"  Warning: Could not compute ortvec correlations for run {run_idx + 1}: {e}")

    out_prefix = Path(args.prefix)

    comp_np = components.detach().cpu().numpy().astype(np.float32)
    mixing_np = mixing.detach().cpu().numpy().astype(np.float32)

    # Free GPU tensors — we have numpy copies now
    del components, mixing
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- Save spatial maps ---
    _vsection(args.verbose, "Save Outputs")
    t_step = time.time()
    _vprint(args.verbose, "Saving ICA spatial maps ...")
    _save_components_4d(
        components_kv=comp_np,
        mask3d=mask3d,
        shape3d=shape3d,
        affine=affine,
        out_file=Path(f"{out_prefix}_{run_tag}_ica_maps.nii.gz"),
    )
    _vprint(args.verbose, f"Maps saved: {out_prefix}_{run_tag}_ica_maps.nii.gz", t_step)

    # --- Save timecourses ---
    np.savetxt(
        f"{out_prefix}_{run_tag}_ica_timecourses.1D",
        mixing_np,
        fmt="%.6f",
        delimiter="\t",
    )
    _vprint(args.verbose, f"Timecourses saved: {out_prefix}_{run_tag}_ica_timecourses.1D")

    # --- Scree plot ---
    _save_scree_plot(
        evr=np.asarray(pca_diag["scree_ratio"], dtype=np.float64),
        out_png=Path(f"{out_prefix}_{run_tag}_pca_scree.png"),
        title=f"Run {run_idx + 1}: PCA scree",
    )

    # --- Correlation heatmap (conditions + ortvec) ---
    corr_blocks = []
    corr_labels = []
    if condition_corr is not None and cond_labels is not None:
        corr_blocks.append(condition_corr)
        corr_labels.extend(cond_labels)
    if ortvec_corr is not None and ortvec_labels is not None:
        corr_blocks.append(ortvec_corr)
        corr_labels.extend(ortvec_labels)
    if len(corr_blocks) > 0 and len(corr_labels) > 0:
        corr_all = np.concatenate(corr_blocks, axis=1).astype(np.float32)
        corr_plot = Path(f"{out_prefix}_{run_tag}_component_correlations.png")
        _save_corr_heatmap(
            corr_kn=corr_all,
            labels=corr_labels,
            out_png=corr_plot,
            title=f"Run {run_idx + 1}: component correlations",
        )
        _vprint(args.verbose, f"Correlation plot saved: {corr_plot}")

    spectral_blocks = []
    spectral_labels = []
    if condition_spectral_corr is not None and cond_labels is not None:
        spectral_blocks.append(condition_spectral_corr)
        spectral_labels.extend(cond_labels)
    if ortvec_spectral_corr is not None and ortvec_labels is not None:
        spectral_blocks.append(ortvec_spectral_corr)
        spectral_labels.extend(ortvec_labels)
    if len(spectral_blocks) > 0 and len(spectral_labels) > 0:
        spectral_all = np.concatenate(spectral_blocks, axis=1).astype(np.float32)
        spectral_plot = Path(f"{out_prefix}_{run_tag}_component_spectral_correlations.png")
        _save_corr_heatmap(
            corr_kn=spectral_all,
            labels=spectral_labels,
            out_png=spectral_plot,
            title=f"Run {run_idx + 1}: component spectral correlations",
        )
        _vprint(args.verbose, f"Spectral correlation plot saved: {spectral_plot}")

    # --- Mixture model z-maps ---
    mixture_meta = []
    z_maps = None
    p_maps = None
    thresh_z_maps = None
    if args.save_mixture_z:
        _vsection(args.verbose, "Mixture Model (GGM)")
        t_step = time.time()
        n_comps_total = comp_np.shape[0]
        _vprint(args.verbose, f"Fitting GGM for {n_comps_total} components on {device} ...")
        comp_tensor = torch.as_tensor(comp_np, device=device)
        z_tensor, p_tensor, mixture_meta = batch_mixture_zscores(
            comp_tensor,
            device=device,
            verbose=args.verbose,
        )
        z_maps = z_tensor.cpu().numpy().astype(np.float32)
        p_maps = p_tensor.cpu().numpy().astype(np.float32)
        thresh_z_maps = z_maps.copy()
        thresh_z_maps[p_maps < float(args.mm_thresh)] = 0.0
        del comp_tensor, z_tensor, p_tensor
        n_conv = sum(1 for m in mixture_meta if m.get("converged", False))
        _vprint(args.verbose, f"GGM done: {n_conv}/{n_comps_total} converged", t_step)

        _save_components_4d(
            components_kv=z_maps,
            mask3d=mask3d,
            shape3d=shape3d,
            affine=affine,
            out_file=Path(f"{out_prefix}_{run_tag}_ica_zmaps.nii.gz"),
        )
        _save_components_4d(
            components_kv=p_maps,
            mask3d=mask3d,
            shape3d=shape3d,
            affine=affine,
            out_file=Path(f"{out_prefix}_{run_tag}_ica_signalprob.nii.gz"),
        )
        _save_components_4d(
            components_kv=thresh_z_maps,
            mask3d=mask3d,
            shape3d=shape3d,
            affine=affine,
            out_file=Path(f"{out_prefix}_{run_tag}_ica_thresh_zmaps.nii.gz"),
        )
        _vprint(args.verbose, "Z-maps and signal-prob maps saved")

    # --- Good/Bad guidance scoring (spatial + temporal) ---
    spatial_scores_good = np.zeros(comp_np.shape[0], dtype=np.float32)
    spatial_scores_bad = np.zeros(comp_np.shape[0], dtype=np.float32)
    temporal_good_scores = np.zeros(comp_np.shape[0], dtype=np.float32)
    temporal_bad_scores = np.zeros(comp_np.shape[0], dtype=np.float32)
    good_mask_score_table: dict[str, list[float]] = {}
    bad_mask_score_table: dict[str, list[float]] = {}

    # Explicitly: onsets/durations are GOOD guide, ortvec is BAD guide.
    if condition_corr is not None:
        temporal_good_scores = np.max(np.abs(condition_corr), axis=1).astype(np.float32)
    if ortvec_corr is not None:
        temporal_bad_scores = np.max(np.abs(ortvec_corr), axis=1).astype(np.float32)

    for entry in guidance_good_masks:
        selector = entry["selector"]
        if z_maps is not None:
            s = _mean_z_excess_by_selector(z_maps, selector, z_thresh=float(args.good_z_thresh))
        else:
            # Fallback if z-maps disabled: use abs(IC) magnitude guide.
            s = _mean_abs_by_selector(comp_np, selector)
        spatial_scores_good += s
        good_mask_score_table[entry["name"]] = s.tolist()

    for entry in guidance_bad_masks:
        selector = entry["selector"]
        s = _mean_abs_by_selector(comp_np, selector)
        spatial_scores_bad += s
        bad_mask_score_table[entry["name"]] = s.tolist()

    if len(guidance_good_masks) > 0:
        spatial_scores_good /= float(len(guidance_good_masks))
    if len(guidance_bad_masks) > 0:
        spatial_scores_bad /= float(len(guidance_bad_masks))

    # Normalize to [0,1] before combining terms from different units/scales.
    spatial_good_norm = _normalize_0_1(spatial_scores_good)
    spatial_bad_norm = _normalize_0_1(spatial_scores_bad)
    temporal_good_norm = _normalize_0_1(temporal_good_scores)
    temporal_bad_norm = _normalize_0_1(temporal_bad_scores)

    overall_good = 0.65 * spatial_good_norm + 0.35 * temporal_good_norm
    overall_bad = 0.65 * spatial_bad_norm + 0.35 * temporal_bad_norm
    good_minus_bad = overall_good - overall_bad

    comp_labels = np.full(comp_np.shape[0], "uncertain", dtype=object)
    comp_labels[good_minus_bad >= 0.15] = "good"
    comp_labels[good_minus_bad <= -0.15] = "bad"

    # Depth profiles (machinery setup for later lag-by-depth work).
    depth_profile_abs: dict[str, list[float]] = {}
    depth_profile_zexcess: dict[str, list[float]] = {}
    if depth_mask_info is not None:
        for lbl in depth_mask_info["labels"]:
            sel = depth_mask_info["selectors"][lbl]
            depth_profile_abs[str(lbl)] = _mean_abs_by_selector(comp_np, sel).tolist()
            if z_maps is not None:
                depth_profile_zexcess[str(lbl)] = _mean_z_excess_by_selector(
                    z_maps, sel, z_thresh=float(args.good_z_thresh)
                ).tolist()

    # Optional per-mask score plots for quick visual QA.
    guidance_good_plot = None
    guidance_bad_plot = None
    if len(good_mask_score_table) > 0:
        labels = list(good_mask_score_table.keys())
        table = np.column_stack([np.asarray(good_mask_score_table[k], dtype=np.float32) for k in labels])
        guidance_good_plot = Path(f"{out_prefix}_{run_tag}_goodmask_scores.png")
        _save_score_heatmap(
            scores_kn=table,
            labels=labels,
            out_png=guidance_good_plot,
            title=f"Run {run_idx + 1}: good-mask scores (z>{args.good_z_thresh:g})",
            cmap="Blues",
        )
    if len(bad_mask_score_table) > 0:
        labels = list(bad_mask_score_table.keys())
        table = np.column_stack([np.asarray(bad_mask_score_table[k], dtype=np.float32) for k in labels])
        guidance_bad_plot = Path(f"{out_prefix}_{run_tag}_badmask_scores.png")
        _save_score_heatmap(
            scores_kn=table,
            labels=labels,
            out_png=guidance_bad_plot,
            title=f"Run {run_idx + 1}: bad-mask scores (abs IC)",
            cmap="Reds",
        )

    if args.melodic_compat:
        is_single_run = int(getattr(args, "_n_runs_total", 1)) == 1
        compat_dir = (
            Path(f"{out_prefix}.ica") if is_single_run else Path(f"{out_prefix}_{run_tag}.ica")
        )
        _write_melodic_compat_outputs(
            compat_dir=compat_dir,
            maps_file=Path(f"{out_prefix}_{run_tag}_ica_maps.nii.gz"),
            zmaps_file=Path(f"{out_prefix}_{run_tag}_ica_zmaps.nii.gz")
            if z_maps is not None
            else None,
            timecourse_file=Path(f"{out_prefix}_{run_tag}_ica_timecourses.1D"),
            pca_scree_ratio=np.asarray(pca_diag["scree_ratio"], dtype=np.float64),
            component_explained_share_pct=np.asarray(explained_share, dtype=np.float64) * 100.0,
            component_total_share_pct=np.asarray(total_share, dtype=np.float64) * 100.0,
            mixing_np=mixing_np,
            mask3d=mask3d,
            mean3d=mean3d,
            shape3d=shape3d,
            affine=affine,
            z_maps=z_maps,
            p_maps=p_maps,
            thresh_z_maps=thresh_z_maps,
        )
        ic_target = (
            Path(f"{out_prefix}_{run_tag}_ica_zmaps.nii.gz")
            if z_maps is not None and Path(f"{out_prefix}_{run_tag}_ica_zmaps.nii.gz").exists()
            else Path(f"{out_prefix}_{run_tag}_ica_maps.nii.gz")
        )
        _vprint(args.verbose, f"MELODIC melodic_IC target: {ic_target}")
        _vprint(args.verbose, f"MELODIC-compatible outputs: {compat_dir}")

    # --- Depth lag analysis (post-step for BOLD-like depth delay signatures) ---
    depth_lag_results: list[dict] = []
    depth_lag_matrix_seconds = None
    depth_lag_matrix_r = None
    depth_lag_plot = None
    depth_lag_method = None
    if args.depth_lag and depth_mask_info is not None and z_maps is not None and depth_source_vox_t_np is not None:
        _vsection(args.verbose, "Depth Lag Analysis")
        t_step = time.time()

        depth_source_proc = depth_source_vox_t_np
        if args.depth_lag_match_preproc:
            src_tc = torch.as_tensor(depth_source_vox_t_np, device=device, dtype=torch.float32)
            src_tc = apply_polort_projection(src_tc, polort=polort, device=device)
            if args.high_pass is not None and args.high_pass > 0:
                src_tc = apply_high_pass_fft(src_tc, tr=tr, high_pass_hz=args.high_pass)
            depth_source_proc = src_tc.detach().cpu().numpy().astype(np.float32)
            del src_tc

        depth_labels = [int(v) for v in depth_mask_info["labels"]]
        n_comp = int(comp_np.shape[0])
        depth_lag_matrix_seconds = np.full((n_comp, len(depth_labels)), np.nan, dtype=np.float32)
        depth_lag_matrix_r = np.full((n_comp, len(depth_labels)), np.nan, dtype=np.float32)

        for ci in range(n_comp):
            z_w = np.where(z_maps[ci] > float(args.depth_lag_z_thresh), z_maps[ci], 0.0).astype(np.float32)

            depth_ts: dict[int, np.ndarray] = {}
            depth_nvox: dict[int, int] = {}
            for lbl in depth_labels:
                selector = depth_mask_info["selectors"][lbl]
                ts, n_use = _weighted_depth_timeseries(
                    source_vox_t=depth_source_proc,
                    selector_v=selector,
                    weight_v=z_w,
                    min_voxels=int(args.depth_lag_min_voxels),
                )
                depth_nvox[int(lbl)] = int(n_use)
                if ts is not None:
                    depth_ts[int(lbl)] = ts

            ref_depth = int(args.depth_lag_reference_depth)
            if ref_depth not in depth_ts:
                depth_lag_results.append(
                    {
                        "component_index": int(ci + 1),
                        "status": "missing_reference_depth",
                        "reference_depth": ref_depth,
                        "n_weighted_voxels_by_depth": depth_nvox,
                    }
                )
                continue

            lag_by_depth: dict[str, float | None] = {}
            r_by_depth: dict[str, float | None] = {}
            used_depths: list[int] = []
            used_lags_s: list[float] = []

            for dj, lbl in enumerate(depth_labels):
                lbl_i = int(lbl)
                if lbl_i not in depth_ts:
                    lag_by_depth[str(lbl_i)] = None
                    r_by_depth[str(lbl_i)] = None
                    continue
                lag_s, r_val, method = _best_lag_and_r(
                    x_t=depth_ts[lbl_i],
                    y_t=depth_ts[ref_depth],
                    tr=tr,
                    max_lag_s=float(args.depth_lag_max_lag_s),
                )
                depth_lag_matrix_seconds[ci, dj] = float(lag_s)
                depth_lag_matrix_r[ci, dj] = float(r_val)
                lag_by_depth[str(lbl_i)] = float(lag_s)
                r_by_depth[str(lbl_i)] = float(r_val)
                if lbl_i != ref_depth:
                    used_depths.append(lbl_i)
                    used_lags_s.append(float(lag_s))
                depth_lag_method = method

            if len(used_depths) >= 3:
                rho, pval = spearmanr(np.asarray(used_depths, dtype=np.float64), np.asarray(used_lags_s, dtype=np.float64))
                rho_f = None if not np.isfinite(rho) else float(rho)
                pval_f = None if not np.isfinite(pval) else float(pval)
            else:
                rho_f, pval_f = None, None

            depth_lag_results.append(
                {
                    "component_index": int(ci + 1),
                    "status": "ok",
                    "reference_depth": ref_depth,
                    "n_weighted_voxels_by_depth": depth_nvox,
                    "lag_seconds_by_depth": lag_by_depth,
                    "peak_r_by_depth": r_by_depth,
                    "spearman_depth_vs_lag": {
                        "rho": rho_f,
                        "pvalue": pval_f,
                        "n_depths": int(len(used_depths)),
                    },
                }
            )

        depth_lag_plot = Path(f"{out_prefix}_{run_tag}_depth_lag_seconds.png")
        _save_depth_lag_plot(
            lag_matrix_kd=depth_lag_matrix_seconds,
            depth_labels=depth_labels,
            out_png=depth_lag_plot,
            title=(
                f"Run {run_idx + 1}: depth lag vs depth {int(args.depth_lag_reference_depth)} "
                f"(z>{args.depth_lag_z_thresh:g})"
            ),
        )
        _vprint(
            args.verbose,
            (
                f"Depth lag done for {len(depth_lag_results)} components "
                f"(ref depth={int(args.depth_lag_reference_depth)}, "
                f"z>{args.depth_lag_z_thresh:g})"
            ),
            t_step,
        )

    elapsed_total = time.time() - t_total
    _vprint(args.verbose, f"Run {run_idx + 1} total elapsed: {elapsed_total:.1f}s")

    mask_type = (
        "provided" if shared_mask is not None else ("auto" if mask3d is not None else "none")
    )
    run_meta = {
        "run_index": int(run_idx + 1),
        "input_file": run_file,
        "tr": float(tr),
        "n_timepoints": int(n_t),
        "n_voxels": int(n_vox_masked),
        "mask_type": mask_type,
        "blur_mask": bool(args.blur_mask),
        "onset_row": None if args.onset_row is None else int(args.onset_row),
        "polort": int(polort),
        "high_pass_hz": None if args.high_pass is None else float(args.high_pass),
        "num_comps_request": args.num_comps,
        "n_components_selected": int(n_components),
        "num_comps_diagnostics": num_diag,
        "pca_diagnostics": {
            "rank_cap": pca_diag["rank_cap"],
            "n_eigs": pca_diag["n_eigs"],
            "first20_scree_ratio": pca_diag["scree_ratio"][:20],
        },
        "icasso": icasso_meta,
        "voxel_norm": bool(args.voxel_norm),
        "tc_var_norm": bool(args.var_norm),
        "smoothness_fwhm_vox": round(float(fwhm_geo), 3),
        "smoothness_resels": round(float(resels), 3),
        "n_eff": int(n_eff),
        "component_variance_share": explained_share.tolist(),
        "component_total_variance_share": total_share.tolist(),
        "component_spatial_stdev_share": stdev_share_sorted.tolist(),
        "sorting": {
            "method": "explained_share",
            "explained_share_source": "mixing_variance_unwhitened_pca",
            "n_sign_flipped": int(n_flipped),
        },
        "condition_labels": cond_labels,
        "condition_durations": cond_durations,
        "condition_corr_preprocessing": {
            "polort": int(polort),
            "high_pass_hz": None if args.high_pass is None else float(args.high_pass),
            "note": "Condition and ortvec vectors are preprocessed to match ICA data temporal filtering",
        },
        "component_condition_corr": None if condition_corr is None else condition_corr.tolist(),
        "component_condition_spectral_corr": None
        if condition_spectral_corr is None
        else condition_spectral_corr.tolist(),
        "ortvec_labels": ortvec_labels,
        "component_ortvec_corr": None if ortvec_corr is None else ortvec_corr.tolist(),
        "component_ortvec_spectral_corr": None
        if ortvec_spectral_corr is None
        else ortvec_spectral_corr.tolist(),
        "spatial_guidance": {
            "good_masks": [
                {
                    "name": m["name"],
                    "source": m["source"],
                    "n_voxels": int(m["n_voxels"]),
                }
                for m in guidance_good_masks
            ],
            "bad_masks": [
                {
                    "name": m["name"],
                    "source": m["source"],
                    "n_voxels": int(m["n_voxels"]),
                }
                for m in guidance_bad_masks
            ],
            "good_z_thresh": float(args.good_z_thresh),
            "temporal_guide_roles": {
                "onsets_durations": "good",
                "ortvec": "bad",
            },
            "component_scores": [
                {
                    "component_index": int(ci + 1),
                    "spatial_good": float(spatial_scores_good[ci]),
                    "spatial_bad": float(spatial_scores_bad[ci]),
                    "temporal_good": float(temporal_good_scores[ci]),
                    "temporal_bad": float(temporal_bad_scores[ci]),
                    "overall_good": float(overall_good[ci]),
                    "overall_bad": float(overall_bad[ci]),
                    "good_minus_bad": float(good_minus_bad[ci]),
                    "label": str(comp_labels[ci]),
                }
                for ci in range(comp_np.shape[0])
            ],
            "good_mask_score_table": good_mask_score_table,
            "bad_mask_score_table": bad_mask_score_table,
            "depth_mask": None
            if depth_mask_info is None
            else {
                "path": depth_mask_info["path"],
                "labels": [int(v) for v in depth_mask_info["labels"]],
                "component_depth_abs_profile": depth_profile_abs,
                "component_depth_zexcess_profile": depth_profile_zexcess,
            },
            "depth_lag": {
                "enabled": bool(args.depth_lag),
                "use_unsmoothed_source": bool(args.depth_lag_use_unsmoothed),
                "match_temporal_preproc": bool(args.depth_lag_match_preproc),
                "reference_depth": int(args.depth_lag_reference_depth),
                "z_thresh": float(args.depth_lag_z_thresh),
                "min_voxels": int(args.depth_lag_min_voxels),
                "max_lag_seconds": float(args.depth_lag_max_lag_s),
                "lag_method": depth_lag_method,
                "component_results": depth_lag_results,
                "lag_seconds_matrix": None
                if depth_lag_matrix_seconds is None
                else depth_lag_matrix_seconds.tolist(),
                "peak_r_matrix": None if depth_lag_matrix_r is None else depth_lag_matrix_r.tolist(),
            },
        },
        "mixture_model": mixture_meta if args.save_mixture_z else None,
        "outputs": {
            "ica_maps": f"{out_prefix}_{run_tag}_ica_maps.nii.gz",
            "ica_timecourses": f"{out_prefix}_{run_tag}_ica_timecourses.1D",
            "pca_scree_plot": f"{out_prefix}_{run_tag}_pca_scree.png",
            "component_correlation_plot": f"{out_prefix}_{run_tag}_component_correlations.png"
            if len(corr_blocks) > 0
            else None,
            "component_spectral_correlation_plot": f"{out_prefix}_{run_tag}_component_spectral_correlations.png"
            if len(spectral_blocks) > 0
            else None,
            "goodmask_score_plot": None if guidance_good_plot is None else str(guidance_good_plot),
            "badmask_score_plot": None if guidance_bad_plot is None else str(guidance_bad_plot),
            "depth_lag_plot": None if depth_lag_plot is None else str(depth_lag_plot),
            "ica_zmaps": f"{out_prefix}_{run_tag}_ica_zmaps.nii.gz"
            if args.save_mixture_z
            else None,
            "ica_signalprob": f"{out_prefix}_{run_tag}_ica_signalprob.nii.gz"
            if args.save_mixture_z
            else None,
            "ica_thresh_zmaps": f"{out_prefix}_{run_tag}_ica_thresh_zmaps.nii.gz"
            if args.save_mixture_z
            else None,
            "melodic_compat_dir": str(
                Path(f"{out_prefix}.ica")
                if int(getattr(args, "_n_runs_total", 1)) == 1
                else Path(f"{out_prefix}_{run_tag}.ica")
            )
            if args.melodic_compat
            else None,
            "melodic_mean": str(
                (
                    Path(f"{out_prefix}.ica")
                    if int(getattr(args, "_n_runs_total", 1)) == 1
                    else Path(f"{out_prefix}_{run_tag}.ica")
                )
                / "mean.nii.gz"
            )
            if args.melodic_compat
            else None,
            "melodic_mask": str(
                (
                    Path(f"{out_prefix}.ica")
                    if int(getattr(args, "_n_runs_total", 1)) == 1
                    else Path(f"{out_prefix}_{run_tag}.ica")
                )
                / "mask.nii.gz"
            )
            if args.melodic_compat
            else None,
            "ppca_debug_trace": None if ppca_debug_file is None else str(ppca_debug_file),
        },
    }

    with open(f"{out_prefix}_{run_tag}_ica_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    return run_meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run-wise whole-brain ICA demo / sanity-check pipeline",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    req = parser.add_argument_group("Required")
    req.add_argument(
        "-input",
        nargs="+",
        required=True,
        help="Input fMRI run files (.nii/.nii.gz/.nii.zst). Multiple files = run-wise ICA.",
    )

    basic = parser.add_argument_group("Core")
    basic.add_argument(
        "-prefix", type=str, default="ffs_ica", help="Output prefix (default: ffs_ica)"
    )
    basic.add_argument(
        "-num_comps",
        type=str,
        default="auto",
        help=(
            "Component selection: INT, FLOAT(0-1), or auto/melodic/hybrid/current/erank/mp. "
            "'auto' and 'melodic' use a MELODIC-style Bayesian evidence proxy."
        ),
    )
    basic.add_argument(
        "-max_auto_components",
        type=float,
        default=0.66,
        help="Upper bound for automatic component selection.  "
        "If <= 1.0, treated as a proportion of the number of timepoints "
        "(e.g. 0.66 = 66%% of T).  If > 1, treated as an absolute integer cap.  "
        "(default: 0.66)",
    )
    basic.add_argument(
        "-auto_min_components",
        type=int,
        default=5,
        help="Lower bound for automatic component selection (default: 5)",
    )
    basic.add_argument(
        "-auto_var_threshold",
        type=float,
        default=0.90,
        help="Variance threshold used by hybrid/current estimator (default: 0.90)",
    )
    basic.add_argument(
        "-auto_no_mp",
        action="store_true",
        help="Disable MP prior in hybrid/current estimator",
    )

    proc = parser.add_argument_group("Preprocessing")
    proc.add_argument("-mask", type=str, default=None, help="Optional brain mask")
    proc.add_argument(
        "-no_auto_mask",
        action="store_true",
        help="Disable automatic masking when no -mask is given (use all voxels — dangerous!)",
    )
    proc.add_argument(
        "-tr", type=float, default=None, help="Override TR (seconds), else from NIfTI header"
    )
    proc.add_argument(
        "-do_scale", action="store_true", help="Scale each voxel timeseries to mean=100"
    )
    proc.add_argument("-do_blur", type=float, default=None, help="Spatial blur FWHM in mm")
    proc.add_argument(
        "-blur_mask",
        action="store_true",
        default=False,
        help="If -do_blur is used, also blur the binary mask and re-threshold at >0.5 (default: off)",
    )
    proc.add_argument(
        "-smoothness_fwhm",
        type=float,
        default=None,
        help="Override estimated spatial smoothness (FWHM in mm) for Minka DOF correction.  "
        "If not set, smoothness is estimated from the data.  Affects dimensionality estimation "
        "via the MELODIC-style effective-sample-size formula: N_eff = n_vox / (2.5 × FWHM_vox).",
    )
    proc.add_argument(
        "-polort",
        type=int,
        default=None,
        help="Polynomial detrend order (default: 0 = demean only)",
    )
    proc.add_argument(
        "-high_pass",
        type=float,
        default=None,
        help="Fourier high-pass cutoff in Hz (e.g., 0.01 Hz)",
    )
    proc.add_argument(
        "-high_pass_s",
        type=float,
        default=None,
        help="Fourier high-pass cutoff in seconds (e.g., 100 = 0.01 Hz). "
        "Mutually exclusive with -high_pass.",
    )
    proc.add_argument(
        "-var_norm",
        dest="var_norm",
        action="store_true",
        default=True,
        help="Normalize ICA mixing timecourses to unit std after decomposition.  "
        "This is cosmetic (output formatting only) and does NOT affect "
        "component estimation or spatial maps.  (default: on)",
    )
    proc.add_argument(
        "-no_var_norm",
        dest="var_norm",
        action="store_false",
        help="Disable post-ICA timecourse normalization (cosmetic only)",
    )
    proc.add_argument(
        "-voxel_norm",
        dest="voxel_norm",
        action="store_true",
        default=True,
        help="MELODIC 'variance normalize timecourses': divide each voxel by "
        "its temporal stdev before PCA/ICA and exclude constant voxels.  "
        "This is MELODIC's standard preprocessing so the estimation is more "
        "influenced by voxel-wise temporal dynamics and less by mean signal "
        "amplitude.  (default: on)",
    )
    proc.add_argument(
        "-no_voxel_norm",
        dest="voxel_norm",
        action="store_false",
        help="Disable MELODIC-style voxel variance normalization before PCA/ICA",
    )

    ica_opts = parser.add_argument_group("ICA / ICASSO")
    ica_opts.add_argument("-ica_max_iter", type=int, default=1000, help="FastICA max iterations")
    ica_opts.add_argument(
        "-ica_tol", type=float, default=1e-6, help="FastICA convergence tolerance"
    )
    ica_opts.add_argument("-icasso", action="store_true", help="Run ICASSO stability analysis")
    ica_opts.add_argument(
        "-icasso_runs", type=int, default=50, help="Number of ICA runs for ICASSO"
    )
    ica_opts.add_argument(
        "-icasso_min_stability", type=float, default=0.7, help="Stability threshold for ICASSO"
    )
    ica_opts.add_argument(
        "-icasso_batch_size",
        type=int,
        default=None,
        help="Optional batch size for ICASSO similarity matrix",
    )

    task = parser.add_argument_group("Task annotation (optional)")
    task.add_argument(
        "-onsets", nargs="+", default=None, help="AFNI timing files (one per condition)"
    )
    task.add_argument(
        "-durations",
        nargs="+",
        default=None,
        help="Durations: one value for all or one per condition",
    )
    task.add_argument(
        "-microtime_dt", type=float, default=0.1, help="Microtime resolution for task regressors"
    )
    task.add_argument(
        "-onset_row",
        type=int,
        default=None,
        help="Optional 1-based row index to use from each AFNI timing file (overrides run-index row selection)",
    )
    task.add_argument(
        "-ortvec",
        nargs="+",
        action="append",
        default=None,
        help=(
            "Optional nuisance regressors: one block per label, format: "
            "-ortvec LABEL file1 [file2 ...]. For multi-run input, provide either one file "
            "(shared) or one file per run."
        ),
    )

    spatial = parser.add_argument_group("Spatial guidance (component labeling)")
    spatial.add_argument(
        "-good_mask",
        action="append",
        default=None,
        help=(
            "Repeatable GOOD spatial mask. Supports 3D binary masks, 3D integer-label masks "
            "(one mask per positive label), or 4D masks (one mask per frame). "
            "Good score uses thresholded |z|-map excess in these regions."
        ),
    )
    spatial.add_argument(
        "-bad_mask",
        action="append",
        default=None,
        help=(
            "Repeatable BAD spatial mask. Supports 3D/4D/label-mask forms like -good_mask. "
            "Bad score uses mean abs(IC) magnitude in these regions."
        ),
    )
    spatial.add_argument(
        "-good_z_thresh",
        type=float,
        default=2.3,
        help="Z threshold used for GOOD-mask spatial scoring on |z|-maps (default: 2.3)",
    )
    spatial.add_argument(
        "-depth_mask",
        type=str,
        default=None,
        help=(
            "Optional 3D integer depth-label map (cortical depth bins). "
            "Current version records per-component depth profiles to metadata "
            "for future lag-by-depth analysis."
        ),
    )
    spatial.add_argument(
        "-depth_lag",
        dest="depth_lag",
        action="store_true",
        default=True,
        help="Enable post-hoc depth lag analysis when -depth_mask is provided (default: on)",
    )
    spatial.add_argument(
        "-no_depth_lag",
        dest="depth_lag",
        action="store_false",
        help="Disable depth lag analysis even if -depth_mask is provided",
    )
    spatial.add_argument(
        "-depth_lag_reference_depth",
        type=int,
        default=3,
        help="Reference depth label for lag estimation (default: 3)",
    )
    spatial.add_argument(
        "-depth_lag_z_thresh",
        type=float,
        default=2.3,
        help="Component z threshold used to build depth-weighted timeseries (default: 2.3)",
    )
    spatial.add_argument(
        "-depth_lag_min_voxels",
        type=int,
        default=20,
        help="Minimum weighted voxels per depth required for lag estimation (default: 20)",
    )
    spatial.add_argument(
        "-depth_lag_max_lag_s",
        type=float,
        default=6.0,
        help="Maximum absolute lag (seconds) searched in xcorr (default: 6.0)",
    )
    spatial.add_argument(
        "-depth_lag_match_preproc",
        dest="depth_lag_match_preproc",
        action="store_true",
        default=True,
        help="Apply same polort/high-pass preprocessing to depth source timeseries before lag analysis (default: on)",
    )
    spatial.add_argument(
        "-no_depth_lag_match_preproc",
        dest="depth_lag_match_preproc",
        action="store_false",
        help="Skip temporal preprocessing on depth source timeseries for lag analysis",
    )
    spatial.add_argument(
        "-depth_lag_use_unsmoothed",
        dest="depth_lag_use_unsmoothed",
        action="store_true",
        default=True,
        help="Use unsmoothed run data (if available) as source for depth lag timeseries (default: on)",
    )
    spatial.add_argument(
        "-no_depth_lag_use_unsmoothed",
        dest="depth_lag_use_unsmoothed",
        action="store_false",
        help="Use current (possibly blurred) data as source for depth lag timeseries",
    )

    out = parser.add_argument_group("Output")
    out.add_argument(
        "-save_mixture_z",
        action="store_true",
        default=True,
        help="Save mixture-model z and signal-prob maps (default: on)",
    )
    out.add_argument(
        "-no_mixture_z",
        dest="save_mixture_z",
        action="store_false",
        help="Disable mixture-model z/signal-prob map outputs",
    )
    out.add_argument(
        "-mm_thresh",
        type=float,
        default=0.5,
        help="Mixture-model posterior threshold for thresholded z-maps (default: 0.5, MELODIC style)",
    )
    out.add_argument(
        "-melodic_compat",
        dest="melodic_compat",
        action="store_true",
        default=True,
        help="Write a MELODIC-style prefix.ica compatibility folder (default: on)",
    )
    out.add_argument(
        "-no_melodic_compat",
        dest="melodic_compat",
        action="store_false",
        help="Disable MELODIC-style prefix.ica compatibility folder",
    )
    out.add_argument(
        "-ppca_debug_dump",
        type=str,
        default=None,
        help=(
            "Optional path to write per-run PPCA debug traces as JSON. "
            "If path ends with .json, files are written as <stem>_runXX.json; "
            "otherwise path is treated as an output directory."
        ),
    )

    future = parser.add_argument_group("Future modes (not yet implemented)")
    future.add_argument(
        "-temp_concat",
        action="store_true",
        help="Placeholder for future temporal concatenation ICA",
    )
    future.add_argument("-tensor", action="store_true", help="Placeholder for future tensorial ICA")

    misc = parser.add_argument_group("Misc")
    misc.add_argument("-cpu", action="store_true", help="Force CPU")
    misc.add_argument("-verbose", action="store_true", help="Verbose logging")
    misc.add_argument("-help", "--help", action="help")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.temp_concat or args.tensor:
        raise NotImplementedError(
            "-temp_concat and -tensor are placeholders for future versions. "
            "Current implementation supports run-wise ICA only."
        )

    if (args.onsets is None) ^ (args.durations is None):
        raise ValueError("Use -onsets and -durations together for task correlation annotation")
    if args.onset_row is not None and args.onset_row < 1:
        raise ValueError("-onset_row must be >= 1 (1-based indexing)")
    if args.good_z_thresh <= 0:
        raise ValueError("-good_z_thresh must be > 0")
    if args.depth_lag_reference_depth < 1:
        raise ValueError("-depth_lag_reference_depth must be >= 1")
    if args.depth_lag_z_thresh <= 0:
        raise ValueError("-depth_lag_z_thresh must be > 0")
    if args.depth_lag_min_voxels < 1:
        raise ValueError("-depth_lag_min_voxels must be >= 1")
    if args.depth_lag_max_lag_s <= 0:
        raise ValueError("-depth_lag_max_lag_s must be > 0")

    # --- Resolve high-pass cutoff: Hz vs seconds ---
    if args.high_pass is not None and args.high_pass_s is not None:
        raise ValueError("Use -high_pass (Hz) or -high_pass_s (seconds), not both.")
    if args.high_pass_s is not None:
        if args.high_pass_s <= 0:
            raise ValueError("-high_pass_s must be a positive period in seconds")
        args.high_pass = 1.0 / args.high_pass_s
        print(f"High-pass: {args.high_pass_s:.1f}s period → {args.high_pass:.6f} Hz")
    elif args.high_pass is not None and args.high_pass > 1.0:
        # Likely user confusion: value > 1 Hz is almost certainly seconds, not Hz
        print(f"⚠ -high_pass {args.high_pass} looks like seconds, not Hz.")
        print(f"  Did you mean: -high_pass_s {args.high_pass} (= {1.0 / args.high_pass:.6f} Hz)?")
        print(f"  Or: -high_pass {1.0 / args.high_pass:.6f}")
        print(
            f"  Interpreting as {args.high_pass} Hz (will remove nearly everything from fMRI data)."
        )
        print("  Press Ctrl-C to abort, or let it continue...")

    input_files = parse_input_files(args.input)
    if args.mm_thresh <= 0.0 or args.mm_thresh >= 1.0:
        raise ValueError("-mm_thresh must be in (0, 1)")

    if args.ortvec is not None:
        for spec in args.ortvec:
            if len(spec) < 2:
                raise ValueError("Each -ortvec requires at least LABEL and one file")
            n_files = len(spec) - 1
            if n_files not in {1, len(input_files)}:
                raise ValueError(
                    f"-ortvec {spec[0]} has {n_files} files; expected 1 or {len(input_files)}"
                )

    for flag_name, paths in [("-good_mask", args.good_mask), ("-bad_mask", args.bad_mask)]:
        if paths is None:
            continue
        for p in paths:
            if not Path(p).exists():
                raise FileNotFoundError(f"{flag_name} file not found: {p}")
    if args.depth_mask is not None and not Path(args.depth_mask).exists():
        raise FileNotFoundError(f"-depth_mask file not found: {args.depth_mask}")

    args._n_runs_total = len(input_files)
    device = torch.device("cpu") if args.cpu else get_device()

    print_cli_header("ffs_ica.py", "Fast run-wise whole-brain ICA")
    print(f"Device: {device}")
    print(f"Runs: {len(input_files)}")
    if args.verbose:
        print(f"Component selection: {args.num_comps}")
        if args.max_auto_components <= 1.0:
            print(f"Max auto components: {args.max_auto_components:.0%} of timepoints")
        else:
            print(f"Max auto components: {int(args.max_auto_components)}")
        print(
            f"Masking: {'provided' if args.mask else 'auto' if not args.no_auto_mask else 'none (dangerous)'}"
        )
        if args.high_pass:
            print(f"High-pass: {args.high_pass:.6f} Hz ({1.0 / args.high_pass:.1f}s period)")
        print(f"Polort: {args.polort if args.polort is not None else '0 (demean)'}")
        print(
            f"Spatial guidance: good_masks={0 if args.good_mask is None else len(args.good_mask)}, "
            f"bad_masks={0 if args.bad_mask is None else len(args.bad_mask)}, "
            f"depth_mask={'yes' if args.depth_mask else 'no'}"
        )
        if args.depth_mask:
            print(
                "Depth lag: "
                f"enabled={args.depth_lag}, ref={args.depth_lag_reference_depth}, "
                f"z>{args.depth_lag_z_thresh:g}, maxlag={args.depth_lag_max_lag_s:g}s, "
                f"minvox={args.depth_lag_min_voxels}, "
                f"match_preproc={args.depth_lag_match_preproc}, "
                f"use_unsmoothed={args.depth_lag_use_unsmoothed}"
            )

    shared_mask = None
    if args.mask is not None:
        shared_mask = load_afni_mask(args.mask)
        print(f"Mask voxels: {int(shared_mask.sum()):,}")
    elif args.no_auto_mask:
        print("⚠ No mask and auto-mask disabled — using ALL voxels including background!")

    t_pipeline = time.time()
    all_meta = []
    for run_idx, run_file in enumerate(input_files):
        print(f"\n[{run_idx + 1}/{len(input_files)}] Processing: {run_file}")
        run_meta = _run_single_ica(
            run_file=run_file,
            run_idx=run_idx,
            args=args,
            device=device,
            shared_mask=shared_mask,
            onsets_files=args.onsets,
            durations=args.durations,
            ortvec_specs=args.ortvec,
        )
        all_meta.append(run_meta)
        print(
            f"  Selected components: {run_meta['n_components_selected']} "
            f"({run_meta['mask_type']} mask, {run_meta['n_voxels']:,} vox) | "
            f"IC1 explained share: {run_meta['component_variance_share'][0] * 100:.2f}%"
        )

    summary_path = f"{args.prefix}_ica_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "n_runs": len(input_files),
                "input_files": input_files,
                "num_comps_request": args.num_comps,
                "device": str(device),
                "runs": all_meta,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    elapsed_pipeline = time.time() - t_pipeline
    print(f"ffs_ica complete ({elapsed_pipeline:.1f}s)")
    print(f"Summary: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
