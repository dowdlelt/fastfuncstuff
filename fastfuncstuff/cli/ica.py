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
- `-temp_concat`: temporal concatenation mode for single-subject multi-run ICA.
  Per-run scaling and varnorm, then concatenation with block-diagonal polort
  and per-run high-pass filtering.
- `-tensor` is a placeholder for future tensorial ICA.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

# fastfuncstuff imports
try:
    from fastfuncstuff.decomposition import (
        io as decomposition_io,
        postprocess as ica_postprocess,
        workflow as ica_workflow,
    )
    from fastfuncstuff.io.afni import get_tr_from_file, load_afni_mask, load_nifti
    from fastfuncstuff.cli_utils import parse_input_files, parse_prefix, print_cli_header
    from fastfuncstuff.decomposition.ica import FastICA, InfoMaxICA, create_ica
    from fastfuncstuff.decomposition.tools import (
        apply_high_pass_fft,
        apply_polort_projection,
        batch_mixture_zscores,
        build_task_design_for_run,
        component_condition_correlations,
        estimate_ica_component_count,
        parse_num_comps_spec,
    )
    from fastfuncstuff.decomposition.icasso import icasso
    from fastfuncstuff.utils import (
        configure_torch_backends,
        gaussian_blur_3d,
        get_device,
        scale_to_percent_signal,
        to_tensor,
    )
except ImportError as e:
    print(f"ERROR: Could not import fastfuncstuff: {e}")
    print("Make sure fastfuncstuff is installed: pip install -e .")
    sys.exit(1)


_estimate_spatial_smoothness_resels = ica_workflow.estimate_spatial_smoothness_resels
_check_finite = ica_workflow.sanitize_finite_tensor
_vsection = ica_workflow.verbose_section
_vprint = ica_workflow.verbose_print


def _run_single_ica(
    run_file: str,
    run_idx: int,
    args,
    device: torch.device,
    shared_mask: np.ndarray | None,
    onsets_files: list[str] | None,
    durations: list[str] | None,
    ortvec_specs: list[list[str]] | None = None,
    bids_task_onsets: "list[list[np.ndarray]] | None" = None,
    bids_task_durations: "list[float] | None" = None,
    bids_task_labels: "list[str] | None" = None,
) -> dict:
    """Run the full ICA workflow for one input run and return run metadata."""
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

    # FSL GUI/FEAT typically runs MELODIC on preprocessed filtered_func_data.
    # Running auto/melodic directly on raw BOLD can yield a lower PPCA model
    # order even with the same final mask.
    num_spec_preview = parse_num_comps_spec(args.num_comps)
    if (
        isinstance(num_spec_preview, str)
        and num_spec_preview in {"auto", "melodic"}
        and Path(run_file).name != "filtered_func_data.nii.gz"
        and Path(run_file).name != "filtered_func_data"
    ):
        print(
            "  Note: auto/melodic on raw input may not match GUI/FEAT model-order "
            "selection. GUI parity is typically against filtered_func_data stage."
        )

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
        from fastfuncstuff.processing.mask import automask

        mask3d = (
            automask(torch.as_tensor(mean3d), dilate_extra=3, device=device, verbose=True)
            .cpu()
            .numpy()
        )
        n_total = int(np.prod(mask3d.shape))
        n_brain = int(mask3d.sum())
        if args.verbose:
            print(
                f"    Auto-mask: {n_brain:,} / {n_total:,} voxels "
                f"({100 * n_brain / max(1, n_total):.1f}%)"
            )
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
    guidance_good_masks = ica_postprocess.prepare_guidance_masks(
        mask_paths=args.good_mask,
        kind="good",
        shape3d=shape3d,
        brain_mask3d=mask3d,
        n_vox_masked=n_vox_masked,
        verbose=args.verbose,
    )
    guidance_bad_masks = ica_postprocess.prepare_guidance_masks(
        mask_paths=args.bad_mask,
        kind="bad",
        shape3d=shape3d,
        brain_mask3d=mask3d,
        n_vox_masked=n_vox_masked,
        verbose=args.verbose,
    )
    depth_mask_info = ica_postprocess.prepare_depth_mask(
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
    data_vox_t = to_tensor(data_vox_t_np, device=device, pin=True)
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
        data_vox_t, norm_msg = ica_workflow.apply_voxel_variance_normalization(
            data_vox_t=data_vox_t,
            num_spec=num_spec,
            n_t=n_t,
            n_vox_masked=n_vox_masked,
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

    # FSL update_mask behavior for model-order estimation: exclude very
    # low-variability voxels before PPCA evidence scan.
    n_eff_for_model_order = n_eff
    data_for_model_order = data_vox_t
    model_order_filter_diag = None
    if isinstance(num_spec, str) and num_spec in {"auto", "melodic"}:
        data_for_model_order, model_order_filter_diag = (
            ica_workflow.filter_voxels_for_melodic_model_order(data_vox_t=data_vox_t)
        )
        n_vox_model_order = int(data_for_model_order.shape[0])
        n_eff_for_model_order = max(n_t, int(n_vox_model_order / (2.5 * resels)))
        _vprint(
            args.verbose,
            "MELODIC dim-est filter: "
            f"kept {n_vox_model_order:,}/{n_vox_masked:,} voxels "
            f"(thr={model_order_filter_diag['std_threshold']:.6g}); "
            f"n_eff={n_eff_for_model_order:,}",
        )

    n_components, pca_diag, num_diag = estimate_ica_component_count(
        data_vox_t=data_for_model_order,
        method=num_spec,
        max_auto_components=max_auto_k,
        auto_min_components=args.auto_min_components,
        auto_var_threshold=args.auto_var_threshold,
        use_mp_prior=not args.auto_no_mp,
        n_eff=n_eff_for_model_order,
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
            "n_eff": int(n_eff_for_model_order),
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
        method_name = getattr(args, "ica_method", "fastica")
        _vprint(
            args.verbose, f"ICASSO ({method_name}): {args.icasso_runs} runs, k={n_components} ..."
        )
        icasso_res = icasso(
            X=x_t,
            n_components=n_components,
            n_runs=args.icasso_runs,
            pca_components=n_components,
            min_stability=args.icasso_min_stability,
            device=device,
            verbose=args.verbose,
            batch_size=args.icasso_batch_size,
            ica_method=method_name,
            base_seed=args.seed,
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

        # Save ICASSO diagnostic plot
        icasso_plot_path = f"{args.prefix}_run{run_idx + 1:02d}_icasso.png"
        try:
            from fastfuncstuff.decomposition.icasso import icasso_plot
            icasso_plot(icasso_res, output_path=icasso_plot_path)
            _vprint(args.verbose, f"ICASSO plot: {icasso_plot_path}")
        except Exception as exc:
            _vprint(args.verbose, f"ICASSO plot failed: {exc}")
    else:
        method_name = getattr(args, "ica_method", "fastica")
        _vprint(
            args.verbose,
            f"{method_name}: k={n_components}, max_iter={args.ica_max_iter}, fun={args.ica_nonlinearity} ...",
        )
        ica = create_ica(
            method=method_name,
            n_components=n_components,
            pca_components=n_components,
            max_iter=args.ica_max_iter,
            tol=args.ica_tol,
            fun=args.ica_nonlinearity,
            random_state=args.seed + run_idx,
            device=device,
        )
        ica.fit(x_t)

        # Check convergence
        if ica.n_iter_ >= args.ica_max_iter:
            _vprint(
                args.verbose,
                f"{method_name} did NOT converge after {args.ica_max_iter} iterations",
                t_step,
            )
            print(f"  ⚠ Consider increasing -ica_max_iter or checking data conditioning")
        else:
            _vprint(args.verbose, f"{method_name} converged in {ica.n_iter_} iterations", t_step)

        # InfoMax diagnostics
        diag = getattr(ica, "diagnostics_", None)
        if diag and args.verbose:
            lr_i = diag["learning_rate_initial"]
            lr_f = diag["learning_rate_final"]
            blk = diag["block_size"]
            n_blk = diag["n_blocks_per_epoch"]
            chg = diag["final_change"]
            _vprint(True, f"  lr: {lr_i:.6f} → {lr_f:.6f}, block={blk}, "
                    f"blocks/epoch={n_blk}, final_change={chg:.2e}")
            if diag.get("extended"):
                _vprint(True, f"  sub-Gaussian: {diag['n_sub_gaussian']}, "
                        f"super-Gaussian: {diag['n_super_gaussian']}")

        components = ica.components_.to(device)
        mixing = ica.mixing_.to(device)
        stability = None
        icasso_meta = {
            "enabled": False,
            "ica_method": method_name,
            "ica_iterations": int(ica.n_iter_),
        }

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
        components, noise_norm_msg = ica_workflow.apply_melodic_noise_normalization(
            components=components,
            mixing=mixing,
            x_t=x_t,
        )
        _vprint(args.verbose, noise_norm_msg)

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
    if bids_task_onsets is not None:
        # BIDS path: use pre-parsed all_onsets and durations
        try:
            from fastfuncstuff.design.builder import create_onset_matrix_microtime
            from fastfuncstuff.design.hrf import get_spmg1_hrf
            from fastfuncstuff.design.matrices import convolve_hrf_microtime
            n_bids_conds = len(bids_task_onsets)
            # Extract this run's onsets (wrapped in list for single-run onset matrix)
            onsets_this_run = [
                [bids_task_onsets[cidx][run_idx]]
                for cidx in range(n_bids_conds)
            ]
            onset_mt = create_onset_matrix_microtime(
                all_onsets=onsets_this_run,
                run_starts=[0],
                tr=tr,
                n_timepoints=n_t,
                microtime_dt=args.microtime_dt,
                stim_durations=bids_task_durations,
                device=device,
            )
            hrf = get_spmg1_hrf(microtime_dt=args.microtime_dt, device=device)
            design_tc = convolve_hrf_microtime(
                onsets_microtime=onset_mt,
                hrf=hrf,
                n_timepoints=n_t,
                tr=tr,
                microtime_dt=args.microtime_dt,
                run_starts=[0],
                device=device,
            )
            cond_labels = bids_task_labels
            cond_durations = bids_task_durations
            design_tc = ica_postprocess.preprocess_design_for_correlation(
                design_tc=design_tc,
                tr=tr,
                polort=polort,
                high_pass_hz=args.high_pass,
                device=device,
            )
            condition_corr = component_condition_correlations(mixing_tk=mixing, design_tc=design_tc)
            condition_spectral_corr = ica_postprocess.component_condition_spectral_correlations(
                mixing_tk=mixing,
                design_tc=design_tc,
            )
        except Exception as e:
            print(f"  Warning: Could not compute condition correlations for run {run_idx + 1}: {e}")
    elif onsets_files is not None and durations is not None:
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
            design_tc = ica_postprocess.preprocess_design_for_correlation(
                design_tc=design_tc,
                tr=tr,
                polort=polort,
                high_pass_hz=args.high_pass,
                device=device,
            )
            condition_corr = component_condition_correlations(mixing_tk=mixing, design_tc=design_tc)
            condition_spectral_corr = ica_postprocess.component_condition_spectral_correlations(
                mixing_tk=mixing,
                design_tc=design_tc,
            )
        except Exception as e:
            print(f"  Warning: Could not compute condition correlations for run {run_idx + 1}: {e}")

    if ortvec_specs is not None:
        try:
            ort_tc, ortvec_labels = ica_postprocess.load_run_ortvec_design(
                ortvec_specs=ortvec_specs,
                run_idx=run_idx,
                n_timepoints=n_t,
                device=device,
            )
            if ort_tc is not None:
                ort_tc = ica_postprocess.preprocess_design_for_correlation(
                    design_tc=ort_tc,
                    tr=tr,
                    polort=polort,
                    high_pass_hz=args.high_pass,
                    device=device,
                )
                ortvec_corr = component_condition_correlations(mixing_tk=mixing, design_tc=ort_tc)
                ortvec_spectral_corr = ica_postprocess.component_condition_spectral_correlations(
                    mixing_tk=mixing,
                    design_tc=ort_tc,
                )
        except Exception as e:
            print(f"  Warning: Could not compute ortvec correlations for run {run_idx + 1}: {e}")

    pfx = parse_prefix(str(args.prefix))
    out_prefix = Path(pfx.stem)
    nii_ext = pfx.nifti_ext

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
    save_items = [
        (comp_np, f"{out_prefix}_{run_tag}_ica_maps{nii_ext}", "maps"),
    ]
    for data_arr, fname, label in tqdm(
        save_items, desc="  Saving NIfTI", leave=False, disable=not args.verbose
    ):
        decomposition_io.save_masked_component_maps_4d(
            components_kv=data_arr,
            mask3d=mask3d,
            shape3d=shape3d,
            affine=affine,
            out_file=Path(fname),
        )
    _vprint(args.verbose, f"Maps saved: {out_prefix}_{run_tag}_ica_maps{nii_ext}", t_step)

    # --- Save timecourses ---
    np.savetxt(
        f"{out_prefix}_{run_tag}_ica_timecourses.1D",
        mixing_np,
        fmt="%.6f",
        delimiter="\t",
    )
    _vprint(args.verbose, f"Timecourses saved: {out_prefix}_{run_tag}_ica_timecourses.1D")

    # --- Scree plot ---
    ica_postprocess.save_scree_plot(
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
        ica_postprocess.save_corr_heatmap(
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
        ica_postprocess.save_corr_heatmap(
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

        ggm_saves = [
            (z_maps, f"{out_prefix}_{run_tag}_ica_zmaps{nii_ext}"),
            (p_maps, f"{out_prefix}_{run_tag}_ica_signalprob{nii_ext}"),
            (thresh_z_maps, f"{out_prefix}_{run_tag}_ica_thresh_zmaps{nii_ext}"),
        ]
        for data_arr, fname in tqdm(
            ggm_saves, desc="  Saving GGM NIfTI", leave=False, disable=not args.verbose
        ):
            decomposition_io.save_masked_component_maps_4d(
                components_kv=data_arr,
                mask3d=mask3d,
                shape3d=shape3d,
                affine=affine,
                out_file=Path(fname),
            )
        _vprint(args.verbose, "Z-maps and signal-prob maps saved")

    # --- Good/Bad guidance scoring (spatial + temporal) ---
    guidance_scores = ica_workflow.compute_guidance_scores(
        comp_np=comp_np,
        z_maps=z_maps,
        condition_corr=condition_corr,
        ortvec_corr=ortvec_corr,
        guidance_good_masks=guidance_good_masks,
        guidance_bad_masks=guidance_bad_masks,
        depth_mask_info=depth_mask_info,
        good_z_thresh=float(args.good_z_thresh),
        out_prefix=out_prefix,
        run_tag=run_tag,
        run_idx=run_idx,
    )
    spatial_scores_good = guidance_scores["spatial_scores_good"]
    spatial_scores_bad = guidance_scores["spatial_scores_bad"]
    temporal_good_scores = guidance_scores["temporal_good_scores"]
    temporal_bad_scores = guidance_scores["temporal_bad_scores"]
    overall_good = guidance_scores["overall_good"]
    overall_bad = guidance_scores["overall_bad"]
    good_minus_bad = guidance_scores["good_minus_bad"]
    comp_labels = guidance_scores["comp_labels"]
    good_mask_score_table = guidance_scores["good_mask_score_table"]
    bad_mask_score_table = guidance_scores["bad_mask_score_table"]
    depth_profile_abs = guidance_scores["depth_profile_abs"]
    depth_profile_zexcess = guidance_scores["depth_profile_zexcess"]
    guidance_good_plot = guidance_scores["guidance_good_plot"]
    guidance_bad_plot = guidance_scores["guidance_bad_plot"]

    if args.melodic_compat:
        is_single_run = int(getattr(args, "_n_runs_total", 1)) == 1
        compat_dir = (
            Path(f"{out_prefix}.ica") if is_single_run else Path(f"{out_prefix}_{run_tag}.ica")
        )
        decomposition_io.write_melodic_compat_outputs(
            compat_dir=compat_dir,
            maps_file=Path(f"{out_prefix}_{run_tag}_ica_maps{nii_ext}"),
            zmaps_file=Path(f"{out_prefix}_{run_tag}_ica_zmaps{nii_ext}")
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
            Path(f"{out_prefix}_{run_tag}_ica_zmaps{nii_ext}")
            if z_maps is not None and Path(f"{out_prefix}_{run_tag}_ica_zmaps{nii_ext}").exists()
            else Path(f"{out_prefix}_{run_tag}_ica_maps{nii_ext}")
        )
        _vprint(args.verbose, f"MELODIC melodic_IC target: {ic_target}")
        _vprint(args.verbose, f"MELODIC-compatible outputs: {compat_dir}")

    # --- Depth lag analysis (post-step for BOLD-like depth delay signatures) ---
    depth_lag_pack = ica_workflow.run_depth_lag_analysis(
        enabled=bool(args.depth_lag),
        depth_mask_info=depth_mask_info,
        z_maps=z_maps,
        depth_source_vox_t_np=depth_source_vox_t_np,
        comp_np=comp_np,
        tr=tr,
        polort=polort,
        high_pass_hz=args.high_pass,
        device=device,
        depth_lag_match_preproc=bool(args.depth_lag_match_preproc),
        depth_lag_reference_depth=int(args.depth_lag_reference_depth),
        depth_lag_z_thresh=float(args.depth_lag_z_thresh),
        depth_lag_min_voxels=int(args.depth_lag_min_voxels),
        depth_lag_max_lag_s=float(args.depth_lag_max_lag_s),
        out_prefix=out_prefix,
        run_tag=run_tag,
        run_idx=run_idx,
        verbose=bool(args.verbose),
    )
    depth_lag_results = depth_lag_pack["depth_lag_results"]
    depth_lag_matrix_seconds = depth_lag_pack["depth_lag_matrix_seconds"]
    depth_lag_matrix_r = depth_lag_pack["depth_lag_matrix_r"]
    depth_lag_plot = depth_lag_pack["depth_lag_plot"]
    depth_lag_method = depth_lag_pack["depth_lag_method"]

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
        "n_eff": int(n_eff_for_model_order),
        "model_order_voxel_filter": model_order_filter_diag,
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
                "peak_r_matrix": None
                if depth_lag_matrix_r is None
                else depth_lag_matrix_r.tolist(),
            },
        },
        "mixture_model": mixture_meta if args.save_mixture_z else None,
        "outputs": {
            "ica_maps": f"{out_prefix}_{run_tag}_ica_maps{nii_ext}",
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
            "ica_zmaps": f"{out_prefix}_{run_tag}_ica_zmaps{nii_ext}"
            if args.save_mixture_z
            else None,
            "ica_signalprob": f"{out_prefix}_{run_tag}_ica_signalprob{nii_ext}"
            if args.save_mixture_z
            else None,
            "ica_thresh_zmaps": f"{out_prefix}_{run_tag}_ica_thresh_zmaps{nii_ext}"
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


# ---------------------------------------------------------------------------
# Temporal concatenation ICA
# ---------------------------------------------------------------------------


def _run_concat_ica(
    input_files: list[str],
    args,
    device: torch.device,
    shared_mask: np.ndarray | None,
) -> dict:
    """Run ICA on temporally concatenated runs (single-subject multi-run).

    Preprocessing order per run:
        load → blur → mask → scale_to_percent_signal → voxel_varnorm
    Then concatenate all runs and apply:
        block-diagonal polort → per-run high-pass → ICA

    Variance normalization is applied per-run before concatenation,
    matching MELODIC's temporal concat behavior (per-file varnorm in
    process_file, then concatenation in setup_classic).
    """
    t_total = time.time()
    n_runs = len(input_files)

    _vsection(args.verbose, "Temporal Concatenation ICA")
    _vprint(args.verbose, f"Concatenating {n_runs} runs for single ICA decomposition")

    # --- Load all runs, apply per-run preprocessing, collect masked data ---
    run_data_list: list[torch.Tensor] = []  # each (n_vox, n_t_run)
    run_lengths: list[int] = []
    mask3d: np.ndarray | None = None
    shape3d: tuple[int, ...] | None = None
    affine: np.ndarray | None = None
    mean3d_accum: np.ndarray | None = None
    tr: float | None = None
    resels_accum: float = 0.0
    fwhm_geo_accum: float = 0.0
    n_vox_masked: int = 0

    num_spec = parse_num_comps_spec(args.num_comps)

    for ri, run_file in enumerate(input_files):
        t_step = time.time()
        _vsection(args.verbose, f"Load Run {ri + 1}/{n_runs}")
        _vprint(args.verbose, f"Loading {run_file} ...")
        img = load_nifti(run_file)
        data = img.get_fdata(dtype=np.float32)
        run_shape3d = data.shape[:3]
        n_t_run = data.shape[3]
        voxel_sizes = tuple(float(v) for v in img.header.get_zooms()[:3])
        _vprint(args.verbose, f"  shape={data.shape}, dtype={data.dtype}", t_step)

        # TR from first run (or CLI override)
        run_tr = float(args.tr) if args.tr is not None else float(get_tr_from_file(run_file))
        if tr is None:
            tr = run_tr
        elif abs(run_tr - tr) > 1e-4:
            print(f"  WARNING: Run {ri + 1} TR={run_tr:.4f}s differs from run 1 TR={tr:.4f}s")

        if shape3d is None:
            shape3d = run_shape3d
            affine = img.affine
        elif run_shape3d != shape3d:
            raise ValueError(f"Run {ri + 1} spatial shape {run_shape3d} != run 1 shape {shape3d}")

        # --- Spatial blur ---
        if args.do_blur is not None and args.do_blur > 0:
            t_step = time.time()
            data = gaussian_blur_3d(
                data=data,
                fwhm_mm=float(args.do_blur),
                voxel_sizes=voxel_sizes,
                device=device,
                verbose=False,
            )
            _vprint(args.verbose, f"  Blurred FWHM={args.do_blur:.1f}mm", t_step)

        # Temporal mean for this run
        run_mean3d = data.mean(axis=-1).astype(np.float32)
        if mean3d_accum is None:
            mean3d_accum = run_mean3d.copy()
        else:
            mean3d_accum += run_mean3d

        # --- Masking (from first run or provided) ---
        if ri == 0:
            if shared_mask is not None:
                mask3d = shared_mask
                _vprint(args.verbose, f"  Using provided mask: {int(mask3d.sum()):,} voxels")
            elif not args.no_auto_mask:
                from fastfuncstuff.processing.mask import automask

                mask3d = (
                    automask(
                        torch.as_tensor(run_mean3d),
                        dilate_extra=2,
                        device=device,
                        verbose=args.verbose,
                    )
                    .cpu()
                    .numpy()
                )
                _vprint(args.verbose, f"  Auto-mask from run 1: {int(mask3d.sum()):,} voxels")

            if mask3d is not None and mask3d.shape != shape3d:
                raise ValueError(f"Mask shape {mask3d.shape} != data shape {shape3d}")

            # Estimate spatial smoothness from first run
            resels_accum, fwhm_geo_accum = _estimate_spatial_smoothness_resels(
                data,
                mask=mask3d,
                device=device,
                verbose=args.verbose,
            )

        # --- Extract masked voxels: (n_vox, n_t_run) ---
        if mask3d is not None:
            run_vox_np = data[mask3d].astype(np.float32)
        else:
            run_vox_np = data.reshape(-1, n_t_run).astype(np.float32)
        del data

        if ri == 0:
            n_vox_masked = run_vox_np.shape[0]
        elif run_vox_np.shape[0] != n_vox_masked:
            raise ValueError(
                f"Run {ri + 1} has {run_vox_np.shape[0]} masked voxels, expected {n_vox_masked}"
            )

        run_vox = to_tensor(run_vox_np, device=device, pin=True)
        del run_vox_np

        # --- Per-run percent-signal scaling ---
        if args.do_scale:
            t_step = time.time()
            run_vox, _, _ = scale_to_percent_signal(
                run_vox,
                run_starts=[0],
                verbose=False,
            )
            run_vox = _check_finite(run_vox, f"run{ri + 1}-post-scale", args.verbose)
            _vprint(args.verbose, f"  Scaled to percent signal", t_step)

        # --- Per-run voxel variance normalization ---
        if args.voxel_norm:
            t_step = time.time()
            run_vox, norm_msg = ica_workflow.apply_voxel_variance_normalization(
                data_vox_t=run_vox,
                num_spec=num_spec,
                n_t=n_t_run,
                n_vox_masked=n_vox_masked,
            )
            run_vox = _check_finite(run_vox, f"run{ri + 1}-post-varnorm", args.verbose)
            _vprint(args.verbose, f"  {norm_msg}", t_step)

        run_data_list.append(run_vox)
        run_lengths.append(n_t_run)
        _vprint(args.verbose, f"  Run {ri + 1}: {n_vox_masked:,} vox x {n_t_run} timepoints")

    # Average the mean images
    mean3d = mean3d_accum / n_runs

    # --- Concatenate all runs ---
    _vsection(args.verbose, "Concatenate")
    t_step = time.time()
    data_vox_t = torch.cat(run_data_list, dim=1)  # (n_vox, total_t)
    del run_data_list
    if device.type == "cuda":
        torch.cuda.empty_cache()

    run_starts = []
    offset = 0
    for rl in run_lengths:
        run_starts.append(offset)
        offset += rl
    total_t = data_vox_t.shape[1]

    _vprint(
        args.verbose,
        f"Concatenated: ({n_vox_masked:,} vox, {total_t} timepoints) from {n_runs} runs",
        t_step,
    )
    _vprint(args.verbose, f"Run starts: {run_starts}, lengths: {run_lengths}")

    # --- Block-diagonal polort detrending ---
    polort = 0 if args.polort is None else int(args.polort)
    _vsection(args.verbose, "Polort Detrend (block-diagonal)")
    t_step = time.time()
    _vprint(args.verbose, f"Order={polort}, {n_runs} runs (block-diagonal)")
    data_vox_t = apply_polort_projection(
        data_vox_t,
        polort=polort,
        device=device,
        run_starts=run_starts,
    )
    data_vox_t = _check_finite(data_vox_t, "post-polort-concat", args.verbose)
    _vprint(args.verbose, "Polort done", t_step)

    # --- Per-run high-pass filter ---
    if args.high_pass is not None and args.high_pass > 0:
        _vsection(args.verbose, "High-Pass Filter (per-run)")
        nyquist = 0.5 / tr
        if args.high_pass >= nyquist:
            raise ValueError(
                f"High-pass cutoff ({args.high_pass:.4f} Hz) >= Nyquist ({nyquist:.4f} Hz)"
            )
        t_step = time.time()
        _vprint(
            args.verbose,
            f"FFT high-pass: {args.high_pass:.6f} Hz, applied independently per run",
        )
        data_vox_t = apply_high_pass_fft(
            data_vox_t,
            tr=tr,
            high_pass_hz=args.high_pass,
            run_starts=run_starts,
        )
        data_vox_t = _check_finite(data_vox_t, "post-highpass-concat", args.verbose)
        _vprint(args.verbose, "High-pass done", t_step)

    # --- Sanity check ---
    data_var = float(torch.var(data_vox_t).item())
    if data_var < 1e-10:
        raise ValueError(f"Data variance is ~0 after preprocessing ({data_var:.2e})")
    _vprint(args.verbose, f"Data variance after preprocessing: {data_var:.4f}")

    # --- Spatial smoothness / effective DOF ---
    resels = resels_accum
    fwhm_geo = fwhm_geo_accum
    n_eff = max(total_t, int(n_vox_masked / (2.5 * resels)))
    _vprint(
        args.verbose,
        f"Effective spatial DOF: {n_eff:,} (resels={resels:.2f}, from run 1)",
    )

    # --- Component count estimation ---
    _vsection(args.verbose, "Component Estimation")
    t_step = time.time()

    if args.max_auto_components <= 1.0:
        max_auto_k = max(5, int(total_t * args.max_auto_components))
    else:
        max_auto_k = int(args.max_auto_components)
    _vprint(args.verbose, f"max_auto_components: {max_auto_k}")
    _vprint(args.verbose, f"Method: {args.num_comps}")

    n_eff_for_model_order = n_eff
    data_for_model_order = data_vox_t
    model_order_filter_diag = None
    if isinstance(num_spec, str) and num_spec in {"auto", "melodic"}:
        data_for_model_order, model_order_filter_diag = (
            ica_workflow.filter_voxels_for_melodic_model_order(data_vox_t=data_vox_t)
        )
        n_vox_model_order = int(data_for_model_order.shape[0])
        n_eff_for_model_order = max(total_t, int(n_vox_model_order / (2.5 * resels)))
        _vprint(
            args.verbose,
            f"MELODIC dim-est filter: kept {n_vox_model_order:,}/{n_vox_masked:,} voxels; "
            f"n_eff={n_eff_for_model_order:,}",
        )

    n_components, pca_diag, num_diag = estimate_ica_component_count(
        data_vox_t=data_for_model_order,
        method=num_spec,
        max_auto_components=max_auto_k,
        auto_min_components=args.auto_min_components,
        auto_var_threshold=args.auto_var_threshold,
        use_mp_prior=not args.auto_no_mp,
        n_eff=n_eff_for_model_order,
        device=device,
        verbose=args.verbose,
        capture_ppca_trace=bool(args.ppca_debug_dump),
    )
    _vprint(
        args.verbose,
        f"Selected {n_components} components (mode={num_diag.get('mode', '?')})",
        t_step,
    )

    x_t = data_vox_t.T  # (time, vox)
    del data_vox_t
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- ICA ---
    _vsection(args.verbose, "ICA Decomposition")
    t_step = time.time()
    if args.icasso:
        method_name = getattr(args, "ica_method", "fastica")
        _vprint(args.verbose, f"ICASSO ({method_name}): {args.icasso_runs} runs, k={n_components}")
        icasso_res = icasso(
            X=x_t,
            n_components=n_components,
            n_runs=args.icasso_runs,
            pca_components=n_components,
            min_stability=args.icasso_min_stability,
            device=device,
            verbose=args.verbose,
            batch_size=args.icasso_batch_size,
            ica_method=method_name,
            base_seed=args.seed,
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
        _vprint(args.verbose, f"ICASSO done ({icasso_res['n_stable']} stable)", t_step)

        # Save ICASSO diagnostic plot
        icasso_plot_path = f"{args.prefix}_tempconcat_icasso.png"
        try:
            from fastfuncstuff.decomposition.icasso import icasso_plot
            icasso_plot(icasso_res, output_path=icasso_plot_path)
            _vprint(args.verbose, f"ICASSO plot: {icasso_plot_path}")
        except Exception as exc:
            _vprint(args.verbose, f"ICASSO plot failed: {exc}")
    else:
        method_name = getattr(args, "ica_method", "fastica")
        _vprint(
            args.verbose,
            f"{method_name}: k={n_components}, max_iter={args.ica_max_iter}, fun={args.ica_nonlinearity}",
        )
        ica = create_ica(
            method=method_name,
            n_components=n_components,
            pca_components=n_components,
            max_iter=args.ica_max_iter,
            tol=args.ica_tol,
            fun=args.ica_nonlinearity,
            random_state=args.seed,
            device=device,
        )
        ica.fit(x_t)
        if ica.n_iter_ >= args.ica_max_iter:
            _vprint(
                args.verbose, f"{method_name} did NOT converge after {args.ica_max_iter} iterations"
            )
        else:
            _vprint(args.verbose, f"{method_name} converged in {ica.n_iter_} iterations", t_step)

        # InfoMax diagnostics
        diag = getattr(ica, "diagnostics_", None)
        if diag and args.verbose:
            lr_i = diag["learning_rate_initial"]
            lr_f = diag["learning_rate_final"]
            blk = diag["block_size"]
            n_blk = diag["n_blocks_per_epoch"]
            chg = diag["final_change"]
            _vprint(True, f"  lr: {lr_i:.6f} → {lr_f:.6f}, block={blk}, "
                    f"blocks/epoch={n_blk}, final_change={chg:.2e}")
            if diag.get("extended"):
                _vprint(True, f"  sub-Gaussian: {diag['n_sub_gaussian']}, "
                        f"super-Gaussian: {diag['n_super_gaussian']}")

        components = ica.components_.to(device)
        mixing = ica.mixing_.to(device)
        stability = None
        icasso_meta = {
            "enabled": False,
            "ica_method": method_name,
            "ica_iterations": int(ica.n_iter_),
        }
        del ica

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- Sign consistency (FSL convention) ---
    max_abs = torch.max(torch.abs(components), dim=1).values
    max_pos = torch.max(components, dim=1).values
    flip_mask = max_abs > max_pos
    n_flipped = int(flip_mask.sum().item())
    if n_flipped > 0:
        components = components.clone()
        mixing = mixing.clone()
        components[flip_mask] *= -1.0
        mixing[:, flip_mask] *= -1.0
        _vprint(args.verbose, f"Sign-flipped {n_flipped} ICs")

    # --- Ordering by explained share ---
    ic_stdev = torch.std(components, dim=1)
    total_stdev = float(ic_stdev.sum().item())
    stdev_share = ic_stdev / max(total_stdev, 1e-15)

    mix_var = torch.var(mixing, dim=0, unbiased=False)
    total_mix_var = float(mix_var.sum().item())
    explained_share_t = mix_var / max(total_mix_var, 1e-15)

    sort_idx = torch.argsort(explained_share_t, descending=True)
    explained_share = explained_share_t[sort_idx].detach().cpu().numpy().astype(np.float32)
    stdev_share_sorted = stdev_share[sort_idx].detach().cpu().numpy().astype(np.float32)

    scree = np.asarray(pca_diag["scree_ratio"], dtype=np.float64)
    retained_frac = float(np.clip(np.sum(scree[:n_components]), 0.0, 1.0))
    total_share = (explained_share * retained_frac).astype(np.float32)

    components = components[sort_idx, :]
    mixing = mixing[:, sort_idx]
    if stability is not None:
        stability = stability[sort_idx.detach().cpu().numpy()]
    del sort_idx
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.var_norm:
        mixing = mixing - mixing.mean(dim=0, keepdim=True)
        mixing_std = torch.clamp(mixing.std(dim=0, keepdim=True), min=1e-8)
        mixing = mixing / mixing_std

    # --- MELODIC-style noise normalization ---
    if x_t is not None:
        _vprint(args.verbose, "Applying MELODIC-style noise normalization ...")
        components, noise_norm_msg = ica_workflow.apply_melodic_noise_normalization(
            components=components,
            mixing=mixing,
            x_t=x_t,
        )
        _vprint(args.verbose, noise_norm_msg)
    del x_t
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- Save outputs (no run tag for concat) ---
    _vsection(args.verbose, "Save Outputs")
    t_step = time.time()
    pfx = parse_prefix(str(args.prefix))
    out_prefix = Path(pfx.stem)
    nii_ext = pfx.nifti_ext

    comp_np = components.detach().cpu().numpy().astype(np.float32)
    mixing_np = mixing.detach().cpu().numpy().astype(np.float32)
    del components, mixing
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Spatial maps
    decomposition_io.save_masked_component_maps_4d(
        components_kv=comp_np,
        mask3d=mask3d,
        shape3d=shape3d,
        affine=affine,
        out_file=Path(f"{out_prefix}_concat_ica_maps{nii_ext}"),
    )
    _vprint(args.verbose, f"Maps: {out_prefix}_concat_ica_maps{nii_ext}", t_step)

    # Timecourses
    np.savetxt(
        f"{out_prefix}_concat_ica_timecourses.1D",
        mixing_np,
        fmt="%.6f",
        delimiter="\t",
    )
    _vprint(args.verbose, f"Timecourses: {out_prefix}_concat_ica_timecourses.1D")

    # Scree plot
    ica_postprocess.save_scree_plot(
        evr=np.asarray(pca_diag["scree_ratio"], dtype=np.float64),
        out_png=Path(f"{out_prefix}_concat_pca_scree.png"),
        title="Temporal concat: PCA scree",
    )

    # Mixture model z-maps
    mixture_meta = []
    z_maps = None
    if args.save_mixture_z:
        _vsection(args.verbose, "Mixture Model (GGM)")
        t_step = time.time()
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

        for data_arr, fname in [
            (z_maps, f"{out_prefix}_concat_ica_zmaps{nii_ext}"),
            (p_maps, f"{out_prefix}_concat_ica_signalprob{nii_ext}"),
            (thresh_z_maps, f"{out_prefix}_concat_ica_thresh_zmaps{nii_ext}"),
        ]:
            decomposition_io.save_masked_component_maps_4d(
                components_kv=data_arr,
                mask3d=mask3d,
                shape3d=shape3d,
                affine=affine,
                out_file=Path(fname),
            )
        _vprint(args.verbose, "Z-maps and signal-prob maps saved", t_step)

    # MELODIC compat
    if args.melodic_compat:
        compat_dir = Path(f"{out_prefix}_concat.ica")
        decomposition_io.write_melodic_compat_outputs(
            compat_dir=compat_dir,
            maps_file=Path(f"{out_prefix}_concat_ica_maps{nii_ext}"),
            zmaps_file=Path(f"{out_prefix}_concat_ica_zmaps{nii_ext}")
            if z_maps is not None
            else None,
            timecourse_file=Path(f"{out_prefix}_concat_ica_timecourses.1D"),
            pca_scree_ratio=np.asarray(pca_diag["scree_ratio"], dtype=np.float64),
            component_explained_share_pct=np.asarray(explained_share, dtype=np.float64) * 100.0,
            component_total_share_pct=np.asarray(total_share, dtype=np.float64) * 100.0,
            mixing_np=mixing_np,
            mask3d=mask3d,
            mean3d=mean3d,
            shape3d=shape3d,
            affine=affine,
            z_maps=z_maps,
            p_maps=p_maps if z_maps is not None else None,
            thresh_z_maps=thresh_z_maps if z_maps is not None else None,
        )
        _vprint(args.verbose, f"MELODIC compat: {compat_dir}")

    elapsed_total = time.time() - t_total
    _vprint(args.verbose, f"Concat ICA total elapsed: {elapsed_total:.1f}s")

    mask_type = (
        "provided" if shared_mask is not None else ("auto" if mask3d is not None else "none")
    )
    concat_meta = {
        "mode": "temp_concat",
        "n_runs": n_runs,
        "input_files": input_files,
        "tr": float(tr),
        "run_lengths": run_lengths,
        "run_starts": run_starts,
        "total_timepoints": total_t,
        "n_voxels": n_vox_masked,
        "mask_type": mask_type,
        "polort": polort,
        "high_pass_hz": None if args.high_pass is None else float(args.high_pass),
        "voxel_norm": bool(args.voxel_norm),
        "voxel_norm_scope": "per_run",
        "num_comps_request": args.num_comps,
        "n_components_selected": int(n_components),
        "num_comps_diagnostics": num_diag,
        "pca_diagnostics": {
            "rank_cap": pca_diag["rank_cap"],
            "n_eigs": pca_diag["n_eigs"],
            "first20_scree_ratio": pca_diag["scree_ratio"][:20],
        },
        "icasso": icasso_meta,
        "smoothness_fwhm_vox": round(float(fwhm_geo), 3),
        "smoothness_resels": round(float(resels), 3),
        "n_eff": int(n_eff_for_model_order),
        "component_variance_share": explained_share.tolist(),
        "component_total_variance_share": total_share.tolist(),
        "component_spatial_stdev_share": stdev_share_sorted.tolist(),
        "elapsed_seconds": round(elapsed_total, 2),
        "outputs": {
            "ica_maps": f"{out_prefix}_concat_ica_maps{nii_ext}",
            "ica_timecourses": f"{out_prefix}_concat_ica_timecourses.1D",
            "pca_scree_plot": f"{out_prefix}_concat_pca_scree.png",
            "ica_zmaps": f"{out_prefix}_concat_ica_zmaps{nii_ext}" if args.save_mixture_z else None,
        },
        "mixture_model": mixture_meta if args.save_mixture_z else None,
    }

    with open(f"{out_prefix}_concat_ica_metadata.json", "w") as f:
        json.dump(concat_meta, f, indent=2)

    return concat_meta


class _HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Show defaults while preserving raw description formatting."""


def build_parser() -> argparse.ArgumentParser:
    """Build and return the ffs_ica command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run-wise whole-brain ICA demo / sanity-check pipeline",
        add_help=False,
        formatter_class=_HelpFormatter,
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
    ica_opts.add_argument(
        "-ica_method",
        type=str,
        default="fastica",
        choices=["fastica", "infomax"],
        help="ICA algorithm: fastica (MELODIC-style deflation) or "
        "infomax (natural gradient, often cleaner fMRI components). Default: fastica",
    )
    ica_opts.add_argument(
        "-ica_max_iter", type=int, default=500, help="ICA max iterations (default: 500)"
    )
    ica_opts.add_argument(
        "-ica_tol",
        type=float,
        default=5e-5,
        help="FastICA convergence tolerance (MELODIC default: 5e-5)",
    )
    ica_opts.add_argument(
        "-ica_nonlinearity",
        type=str,
        default="pow3",
        choices=["pow3", "cube", "logcosh"],
        help="FastICA contrast function (default: pow3, matching MELODIC)",
    )
    ica_opts.add_argument(
        "-seed", type=int, default=0,
        help="Base random seed for ICA. Per-run seed = seed + run_idx. "
        "ICASSO increments from this base. (default: 0)",
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
        "-onsets", nargs="+", default=None,
        help="AFNI timing files (one per condition). Mutually exclusive with -events.",
    )
    task.add_argument(
        "-durations",
        nargs="+",
        default=None,
        help="Durations: one value for all or one per condition. Required with -onsets.",
    )
    task.add_argument(
        "-events",
        nargs="+",
        default=None,
        metavar="TSV",
        help="BIDS *_events.tsv files for task annotation (one per run). "
        "Mutually exclusive with -onsets.",
    )
    task.add_argument(
        "-event_ignore",
        nargs="+",
        default=None,
        metavar="LABEL",
        help="trial_type values to exclude. Only valid with -events.",
    )
    task.add_argument(
        "-event_cols",
        nargs=3,
        default=None,
        metavar=("ONSET_COL", "DURATION_COL", "TRIAL_TYPE_COL"),
        help="Custom column names for onset, duration, trial_type. "
        "Default: onset duration trial_type. Only valid with -events.",
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

    modes = parser.add_argument_group("Multi-run modes")
    modes.add_argument(
        "-temp_concat",
        action="store_true",
        help=(
            "Temporal concatenation: per-run scale+varnorm, then concatenate "
            "and run single ICA with block-diagonal polort and per-run high-pass. "
            "Requires >= 2 input runs."
        ),
    )
    modes.add_argument("-tensor", action="store_true", help="Placeholder for future tensorial ICA")

    misc = parser.add_argument_group("Misc")
    misc.add_argument("-cpu", action="store_true", help="Force CPU")
    misc.add_argument("-verbose", action="store_true", help="Verbose logging")
    misc.add_argument("-help", "--help", action="help")

    return parser


def main() -> None:
    """Parse CLI args and execute run-wise ICA across all input runs."""
    parser = build_parser()
    args = parser.parse_args()

    if args.tensor:
        raise NotImplementedError(
            "-tensor is a placeholder for future tensorial ICA. "
            "Current implementation supports run-wise and temporal concat ICA."
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
    configure_torch_backends(device)

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

    # Validate and parse task annotation sources
    _has_onsets = bool(args.onsets)
    _has_events = bool(getattr(args, "events", None))
    if _has_onsets and _has_events:
        print("ERROR: Specify only one of -onsets or -events for task annotation")
        sys.exit(1)
    if getattr(args, "event_ignore", None) and not _has_events:
        print("ERROR: -event_ignore requires -events")
        sys.exit(1)
    if getattr(args, "event_cols", None) and not _has_events:
        print("ERROR: -event_cols requires -events")
        sys.exit(1)

    # Parse BIDS events (if provided) for task annotation
    bids_task_onsets = None     # list[list[ndarray]] — all_onsets[cond][run]
    bids_task_durations = None  # list[float]
    bids_task_labels = None     # list[str]
    if _has_events:
        from fastfuncstuff.design.bids_events import parse_bids_events
        event_cols = tuple(args.event_cols) if getattr(args, "event_cols", None) else None
        bids_task_onsets, bids_task_durations, bids_task_labels = parse_bids_events(
            event_files=args.events,
            event_ignore=getattr(args, "event_ignore", None),
            event_cols=event_cols,
        )
        print(
            f"Task annotation: {len(bids_task_labels)} conditions from BIDS events: "
            f"{bids_task_labels}"
        )

    t_pipeline = time.time()

    if args.temp_concat:
        # --- Temporal concatenation mode ---
        if len(input_files) < 2:
            raise ValueError("-temp_concat requires at least 2 input runs")
        print(f"\nMode: temporal concatenation ({len(input_files)} runs → single ICA)")
        concat_meta = _run_concat_ica(
            input_files=input_files,
            args=args,
            device=device,
            shared_mask=shared_mask,
        )
        print(
            f"  Selected components: {concat_meta['n_components_selected']} "
            f"({concat_meta['mask_type']} mask, {concat_meta['n_voxels']:,} vox) | "
            f"IC1 explained share: {concat_meta['component_variance_share'][0] * 100:.2f}%"
        )

        pfx = parse_prefix(str(args.prefix))
        summary_path = f"{pfx.stem}_ica_summary.json"
        with open(summary_path, "w") as f:
            json.dump(concat_meta, f, indent=2)

        print("\n" + "=" * 70)
        elapsed_pipeline = time.time() - t_pipeline
        print(f"ffs_ica temp_concat complete ({elapsed_pipeline:.1f}s)")
        print(f"Summary: {summary_path}")
        print("=" * 70)
    else:
        # --- Run-wise mode (default) ---
        all_meta = []
        for run_idx, run_file in enumerate(input_files):
            print(f"\n[{run_idx + 1}/{len(input_files)}] Processing: {run_file}")
            run_meta = _run_single_ica(
                run_file=run_file,
                run_idx=run_idx,
                args=args,
                device=device,
                shared_mask=shared_mask,
                onsets_files=args.onsets if _has_onsets else None,
                durations=args.durations if _has_onsets else None,
                ortvec_specs=args.ortvec,
                bids_task_onsets=bids_task_onsets,
                bids_task_durations=bids_task_durations,
                bids_task_labels=bids_task_labels,
            )
            all_meta.append(run_meta)
            print(
                f"  Selected components: {run_meta['n_components_selected']} "
                f"({run_meta['mask_type']} mask, {run_meta['n_voxels']:,} vox) | "
                f"IC1 explained share: {run_meta['component_variance_share'][0] * 100:.2f}%"
            )

        pfx = parse_prefix(str(args.prefix))
        summary_path = f"{pfx.stem}_ica_summary.json"
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
