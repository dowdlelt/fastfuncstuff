#!/usr/bin/env python3

"""
ffs_ica.py - Fast run-wise whole-brain ICA sanity-check / demo CLI.

Core goals
----------
- Simple whole-brain ICA workflow with GPU acceleration when available.
- Automatic component estimation (Marchenko-Pastur ceiling + Minka Laplace evidence).
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
- `-tensor`: tensorial (spatial-concat) mode. Requires runs with identical T;
  stacks along voxel axis → (T, n_runs*V) and produces a single shared temporal
  mixing with per-run spatial maps.
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

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter

# fastfuncstuff imports
try:
    from fastfuncstuff.cli_utils import (
        add_device_arg,
        add_verbose_arg,
        parse_input_files,
        parse_prefix,
        print_cli_header,
        setup_device,
        spinner,
    )
    from fastfuncstuff.decomposition import (
        io as decomposition_io,
    )
    from fastfuncstuff.decomposition import (
        postprocess as ica_postprocess,
    )
    from fastfuncstuff.decomposition import (
        workflow as ica_workflow,
    )
    from fastfuncstuff.decomposition.ica import FastICA, InfoMaxICA, create_ica  # noqa: F401
    from fastfuncstuff.decomposition.icasso import icasso
    from fastfuncstuff.decomposition.mixture import batch_mixture_zscores
    from fastfuncstuff.decomposition.model_order import effective_sample_size_from_resels
    from fastfuncstuff.decomposition.tools import (
        apply_high_pass_fft,
        apply_polort_projection,
        build_task_design_for_run,
        component_condition_correlations,
        estimate_ica_component_count,
        find_constant_voxels,
        parse_num_comps_spec,
        prune_mask_constant_voxels,
    )
    from fastfuncstuff.io.afni import get_tr_from_file, load_afni_mask, load_nifti
    from fastfuncstuff.utils import (
        gaussian_blur_3d,
        scale_to_percent_signal,
        to_tensor,
    )
except ImportError as e:
    print(f"ERROR: Could not import fastfuncstuff: {e}")
    print("Make sure fastfuncstuff is installed: pip install -e .")
    sys.exit(1)


_estimate_smoothness_resels_acf = ica_workflow.estimate_smoothness_resels_acf
_check_finite = ica_workflow.sanitize_finite_tensor
_vsection = ica_workflow.verbose_section
_vprint = ica_workflow.verbose_print


def _prune_constant_runs(
    mask3d: np.ndarray | None,
    const_bad_vox: np.ndarray | None,
    run_data_list: list[torch.Tensor],
    verbose: bool,
) -> tuple[np.ndarray | None, int]:
    """Drop constant voxels from the mask and every per-run (V, T) matrix.

    ``const_bad_vox`` is the union across runs — a voxel constant in any one
    run is dropped everywhere, so the runs stay row-aligned. Returns the
    updated mask and the number of voxels dropped.
    """
    if mask3d is None or const_bad_vox is None or not const_bad_vox.any():
        return mask3d, 0
    new_mask, n_drop = prune_mask_constant_voxels(mask3d, const_bad_vox)
    keep = torch.from_numpy(~np.asarray(const_bad_vox, dtype=bool))
    for i in range(len(run_data_list)):
        run_data_list[i] = run_data_list[i][keep.to(run_data_list[i].device)]
    _vprint(
        verbose,
        f"Mask updated: dropped {n_drop:,} constant voxels "
        f"({100.0 * n_drop / len(const_bad_vox):.2f}% of mask) — "
        "constant in at least one run",
    )
    return new_mask, n_drop


def _run_single_ica(
    run_file: str,
    run_idx: int,
    args,
    device: torch.device,
    shared_mask: np.ndarray | None,
    onsets_files: list[str] | None,
    durations: list[str] | None,
    ortvec_specs: list[list[str]] | None = None,
    bids_task_onsets: list[list[np.ndarray]] | None = None,
    bids_task_durations: list[float] | None = None,
    bids_task_labels: list[str] | None = None,
) -> dict:
    """Run the full ICA workflow for one input run and return run metadata."""
    t_total = time.time()
    t_step = time.time()
    run_tag = f"run{run_idx + 1:02d}"

    _vsection(args.verb >= 1, "Load Data")
    _vprint(args.verb >= 1, f"Loading {run_file} ...")
    img = load_nifti(run_file)
    data = img.get_fdata(dtype=np.float32)
    data_unblurred = data.copy() if (args.depth_lag and args.depth_lag_use_unsmoothed) else None
    affine = img.affine
    shape3d = data.shape[:3]
    n_t = data.shape[3]
    voxel_sizes = tuple(float(v) for v in img.header.get_zooms()[:3])
    _vprint(args.verb >= 1, f"Loaded: shape={data.shape}, dtype={data.dtype}", t_step)

    tr = float(args.tr) if args.tr is not None else float(get_tr_from_file(run_file))
    _vprint(args.verb >= 1, f"TR = {tr:.4f}s, duration = {tr * n_t:.1f}s")

    # Model order is a property of the data, not just the mask: unpreprocessed BOLD
    # carries drift and motion structure that changes the eigenspectrum, and so the
    # selected order, relative to a filtered/denoised version of the same run.
    num_spec_preview = parse_num_comps_spec(args.num_comps)
    if (
        isinstance(num_spec_preview, str)
        and num_spec_preview in {"auto", "laplace"}
        and Path(run_file).name != "filtered_func_data.nii.gz"
        and Path(run_file).name != "filtered_func_data"
    ):
        print(
            "  Note: automatic model order on raw input reflects drift and motion "
            "structure as well as signal; expect a different order after preprocessing."
        )

    if shared_mask is not None and shared_mask.shape != shape3d:
        raise ValueError(
            f"Mask shape {shared_mask.shape} does not match run shape {shape3d} for {run_file}"
        )

    # --- Spatial blur ---
    if args.do_blur is not None and args.do_blur > 0:
        _vsection(args.verb >= 1, "Spatial Blur")
        t_step = time.time()
        _vprint(args.verb >= 1, f"FWHM={args.do_blur:.1f} mm ...")
        data = gaussian_blur_3d(
            data=data,
            fwhm_mm=float(args.do_blur),
            voxel_sizes=voxel_sizes,
            device=device,
            verbose=args.verb >= 1,
        )
        _vprint(args.verb >= 1, "Spatial blur done", t_step)

    # Save temporal mean in image space for compatibility outputs
    mean3d = data.mean(axis=-1).astype(np.float32)

    # --- Masking ---
    _vsection(args.verb >= 1, "Masking")
    t_step = time.time()
    if shared_mask is not None:
        mask3d = shared_mask
        _vprint(args.verb >= 1, f"Using provided mask: {int(mask3d.sum()):,} voxels")
    elif not args.no_auto_mask:
        from fastfuncstuff.processing.mask import automask

        mask3d = (
            automask(torch.as_tensor(mean3d), dilate_extra=3, device=device, verbose=True)
            .cpu()
            .numpy()
        )
        n_total = int(np.prod(mask3d.shape))
        n_brain = int(mask3d.sum())
        if args.verb >= 1:
            print(
                f"    Auto-mask: {n_brain:,} / {n_total:,} voxels "
                f"({100 * n_brain / max(1, n_total):.1f}%)"
            )
    else:
        mask3d = None  # no masking at all
        _vprint(args.verb >= 1, f"No mask: using all {np.prod(shape3d):,} voxels (no_auto_mask)")

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
            args.verb >= 1,
            f"Blur-mask enabled: {n_before:,} -> {n_after:,} voxels "
            f"(orig ∪ (blurred > {blur_mask_thresh}))",
        )

    if mask3d is not None:
        data_vox_t_np = data[mask3d].astype(np.float32)
    else:
        data_vox_t_np = data.reshape(-1, n_t).astype(np.float32)

    # Constant voxels must leave the mask, not just be zeroed — see
    # tools.find_constant_voxels for why (mixture-model collapse).
    if mask3d is not None and args.drop_constant:
        _bad = find_constant_voxels(data_vox_t_np)
        mask3d, _n_drop = prune_mask_constant_voxels(mask3d, _bad)
        if _n_drop:
            data_vox_t_np = data_vox_t_np[~_bad]
            _vprint(
                args.verb >= 1,
                f"Mask updated: dropped {_n_drop:,} constant voxels "
                f"({100.0 * _n_drop / len(_bad):.2f}% of mask)",
            )

    n_vox_masked = data_vox_t_np.shape[0]
    _vprint(args.verb >= 1, f"Masked data: ({n_vox_masked:,} vox, {n_t} time)", t_step)

    if n_vox_masked < 100:
        raise ValueError(
            f"Only {n_vox_masked} voxels after masking — data appears empty or mask is wrong"
        )

    # --- Spatial guidance masks (good / bad / depth) ---
    _vsection(args.verb >= 1, "Spatial Guidance Inputs")
    guidance_good_masks = ica_postprocess.prepare_guidance_masks(
        mask_paths=args.good_mask,
        kind="good",
        shape3d=shape3d,
        brain_mask3d=mask3d,
        n_vox_masked=n_vox_masked,
        verbose=args.verb >= 1,
    )
    guidance_bad_masks = ica_postprocess.prepare_guidance_masks(
        mask_paths=args.bad_mask,
        kind="bad",
        shape3d=shape3d,
        brain_mask3d=mask3d,
        n_vox_masked=n_vox_masked,
        verbose=args.verb >= 1,
    )
    depth_mask_info = ica_postprocess.prepare_depth_mask(
        depth_mask_path=args.depth_mask,
        shape3d=shape3d,
        brain_mask3d=mask3d,
        n_vox_masked=n_vox_masked,
    )
    _vprint(
        args.verb >= 1,
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

    # --- Estimate spatial smoothness for effective DOF ---
    # A smoothed image has fewer independent observations than voxels, and the Laplace
    # evidence scales with that count, so feeding it the raw voxel count is what makes
    # model order run away. One resel is FWHM_x × FWHM_y × FWHM_z voxels (Worsley et al.
    # 1992), so N_eff = n_vox / resels.
    _vsection(args.verb >= 1, "Spatial Smoothness")
    t_step = time.time()
    if args.smoothness_fwhm is not None:
        # User-supplied smoothness in mm → convert to voxels (isotropic)
        voxel_sizes = tuple(float(v) for v in img.header.get_zooms()[:3])
        mean_vox = float(np.mean(voxel_sizes))
        fwhm_vox = max(1.0, args.smoothness_fwhm / mean_vox)
        resels = fwhm_vox**3  # isotropic assumption
        fwhm_geo = fwhm_vox
        _vprint(
            args.verb >= 1,
            f"User smoothness: {args.smoothness_fwhm:.1f} mm = {fwhm_vox:.2f} voxels "
            f"(voxel size={mean_vox:.2f} mm), resels={resels:.1f}",
        )
    else:
        # Estimate from data (must happen before data is freed)
        resels, fwhm_geo, _smooth_diag = _estimate_smoothness_resels_acf(
            data,
            voxel_sizes,
            mask=mask3d,
            device=device,
            per_axis=bool(getattr(args, "smoothness_per_axis", False)),
            corder=int(getattr(args, "smoothness_corder", -1)),
            verbose=args.verb >= 1,
        )
        _vprint(
            args.verb >= 1,
            f"Estimated spatial FWHM: {fwhm_geo:.2f} voxels (resels={resels:.2f})",
        )

    # resels is the product FWHM_x × FWHM_y × FWHM_z, not a geometric mean.
    # Floor at n_time to avoid degenerate cases.
    n_eff = effective_sample_size_from_resels(n_vox_masked, resels, floor=n_t)
    _vprint(
        args.verb >= 1,
        f"Effective spatial DOF: {n_eff:,} (raw={n_vox_masked:,}, resels={resels:.1f})",
        t_step,
    )

    # --- To GPU ---
    t_step = time.time()
    data_vox_t = to_tensor(data_vox_t_np, device=device, pin=True)
    _vprint(args.verb >= 1, f"Data on {device}", t_step)

    # Free large numpy array
    del data_vox_t_np, data

    # --- Percent-signal scaling ---
    if args.do_scale:
        _vsection(args.verb >= 1, "Scaling")
        t_step = time.time()
        _vprint(args.verb >= 1, "Percent-signal scaling ...")
        data_vox_t, _, _ = scale_to_percent_signal(
            data_vox_t, run_starts=[0], verbose=args.verb >= 1
        )
        data_vox_t = _check_finite(data_vox_t, "post-scale", args.verb >= 1)
        _vprint(args.verb >= 1, "Scaling done", t_step)

    # --- Polort detrending ---
    polort = 0 if args.polort is None else int(args.polort)

    _vsection(args.verb >= 1, "Polort Detrend")
    t_step = time.time()
    _vprint(args.verb >= 1, f"Order={polort} ...")
    data_vox_t = apply_polort_projection(data_vox_t, polort=polort, device=device)
    data_vox_t = _check_finite(data_vox_t, "post-polort", args.verb >= 1)
    _vprint(args.verb >= 1, "Polort done", t_step)

    # --- High-pass filter ---
    if args.high_pass is not None and args.high_pass > 0:
        _vsection(args.verb >= 1, "High-Pass Filter")
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
            args.verb >= 1,
            f"FFT high-pass: {args.high_pass:.6f} Hz ({period_s:.1f}s period, Nyquist={nyquist:.4f} Hz) ...",
        )
        data_vox_t = apply_high_pass_fft(data_vox_t, tr=tr, high_pass_hz=args.high_pass)
        data_vox_t = _check_finite(data_vox_t, "post-highpass", args.verb >= 1)
        _vprint(args.verb >= 1, "High-pass done", t_step)

    # --- Sanity: check data variance ---
    data_var = float(torch.var(data_vox_t).item())
    if data_var < 1e-10:
        raise ValueError(
            f"Data variance is ~0 after preprocessing ({data_var:.2e}). "
            "Check mask, scaling, and filter settings."
        )
    _vprint(args.verb >= 1, f"Data variance after preprocessing: {data_var:.4f}")

    # Parse component mode before voxel normalization so we can
    # choose the exact preprocessing path for MELODIC-like estimation.
    num_spec = parse_num_comps_spec(args.num_comps)

    # --- Voxel-wise variance normalization ---
    # MELODIC's "variance normalize timecourses": divide each voxel's
    # timeseries by its temporal std so ICA focuses on temporal dynamics
    # rather than being dominated by high-amplitude voxels.
    # Also excludes constant/near-constant voxels (MELODIC step 3).
    varnorm_std_np: np.ndarray | None = None
    if args.voxel_norm:
        _vsection(args.verb >= 1, "Voxel Variance Normalization")
        t_step = time.time()
        # Capture std BEFORE normalization — needed for correct PSC amplitudes.
        # PSC formula: 100 * std(mixing) * comp * varnorm_std / mean.
        # Without varnorm_std the values are off by CoV (~50-100x too small).
        varnorm_std_np = data_vox_t.std(dim=1).cpu().numpy()  # (V,) in data units
        _trace_single_dir = getattr(args, "trace", None)
        _trace_vn_dir = None
        if _trace_single_dir:
            from pathlib import Path as _Pvn

            _trace_vn_dir = _Pvn(_trace_single_dir) / run_tag
            _trace_vn_dir.mkdir(parents=True, exist_ok=True)
        data_vox_t, norm_msg = ica_workflow.apply_voxel_variance_normalization(
            signal_rank=getattr(args, "varnorm_rank", None),
            data_vox_t=data_vox_t,
            num_spec=num_spec,
            n_t=n_t,
            n_vox_masked=n_vox_masked,
            trace_dir=_trace_vn_dir,
        )
        data_vox_t = _check_finite(data_vox_t, "post-voxel-norm", args.verb >= 1)
        _vprint(args.verb >= 1, norm_msg, t_step)

    # --- Component count estimation ---
    _vsection(args.verb >= 1, "Component Estimation")
    t_step = time.time()

    # Resolve max_auto_components: proportion → absolute integer
    if args.max_auto_components <= 1.0:
        max_auto_k = max(5, int(n_t * args.max_auto_components))
        _vprint(
            args.verb >= 1,
            f"max_auto_components: {args.max_auto_components:.0%} of {n_t} timepoints = {max_auto_k}",
        )
    else:
        max_auto_k = int(args.max_auto_components)
        _vprint(args.verb >= 1, f"max_auto_components: {max_auto_k} (absolute)")

    _vprint(args.verb >= 1, f"Method: {args.num_comps} ...")

    # Exclude near-flat voxels from the model-order estimate only: they sit at the
    # bottom of the spectrum and drag the estimated noise floor down.
    n_eff_for_model_order = n_eff
    data_for_model_order = data_vox_t
    model_order_filter_diag = None
    if isinstance(num_spec, str) and num_spec in {"auto", "laplace"}:
        data_for_model_order, model_order_filter_diag = ica_workflow.filter_low_variance_voxels(
            data_vox_t=data_vox_t
        )
        n_vox_model_order = int(data_for_model_order.shape[0])
        n_eff_for_model_order = effective_sample_size_from_resels(
            n_vox_model_order, resels, floor=n_t
        )
        _vprint(
            args.verb >= 1,
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
        verbose=args.verb >= 1,
        capture_ppca_trace=bool(args.ppca_debug_dump) or getattr(args, "trace", None) is not None,
    )

    # Opt-in: replace the spectral count with a restart-stability count. Kept off by
    # default because it overcounts -- see -help and decomposition/stability.py.
    if getattr(args, "stability_order", False):
        from fastfuncstuff.decomposition.stability import stability_model_order

        _vsection(args.verb >= 1, "Stability-based model order")
        t_step = time.time()
        k_ceiling = int(pca_diag.get("mp_signal_count") or n_components)
        k_ceiling = max(2, min(k_ceiling, int(max_auto_k)))
        st = stability_model_order(
            data_for_model_order.T,
            k_max=k_ceiling,
            n_runs=int(args.stability_runs),
            min_stability=float(args.icasso_min_stability),
            device=device,
            base_seed=int(getattr(args, "seed", 0) or 0),
            verbose=args.verb >= 2,
        )
        _vprint(
            args.verb >= 1,
            f"Stability order: k={st.k} of k_max={st.k_max} "
            f"(Iq >= {st.min_stability}, {st.n_runs} restarts); "
            f"spectral estimate was {n_components}",
            t_step,
        )
        num_diag = {**num_diag, "stability_order": st.as_dict(), "spectral_k": int(n_components)}
        n_components = int(st.k)

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
        _vprint(args.verb >= 1, f"PPCA debug trace: {ppca_debug_file}")

    _trace_eig_flag = getattr(args, "trace", None)
    if _trace_eig_flag is not None and "ppca_trace" in pca_diag:
        from pathlib import Path as _Peig

        _eig_td = _Peig(_trace_eig_flag) / run_tag
        _eig_td.mkdir(parents=True, exist_ok=True)
        _ppt = pca_diag["ppca_trace"]
        _eigs = np.array(_ppt.get("eigenvalues", []), dtype=np.float64)
        _ll = np.array(_ppt.get("log_evidence", []), dtype=np.float64)
        if len(_eigs) > 0:
            np.savetxt(str(_eig_td / "eigenvalues"), _eigs, fmt="%.10g")
        if len(_ll) > 0:
            np.save(str(_eig_td / "log_evidence.npy"), _ll)
        _vprint(args.verb >= 1, f"Trace: model-order evidence → {_eig_td}")

    _vprint(
        args.verb >= 1,
        f"Selected {n_components} components (mode={num_diag.get('mode', '?')})",
        t_step,
    )

    x_t = data_vox_t.T  # (time, vox)
    del data_vox_t  # free GPU memory before ICA

    # --- ICA ---
    _vsection(args.verb >= 1, "ICA Decomposition")
    t_step = time.time()
    pca_eigenvalues = None  # PCA explained_variance_ for IC variance computation
    pca_components_for_sort = None  # PCA spatial components (k, V)
    if args.icasso:
        method_name = getattr(args, "ica_method", "fastica")
        _vprint(
            args.verb >= 1, f"ICASSO ({method_name}): {args.icasso_runs} runs, k={n_components} ..."
        )
        icasso_res = icasso(
            X=x_t,
            n_components=n_components,
            n_runs=args.icasso_runs,
            pca_components=n_components,
            min_stability=args.icasso_min_stability,
            device=device,
            verbose=args.verb >= 1,
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
        _vprint(args.verb >= 1, f"ICASSO done ({icasso_res['n_stable']} stable)", t_step)

        # Save ICASSO diagnostic plot
        icasso_plot_path = f"{args.prefix}_run{run_idx + 1:02d}_icasso.png"
        try:
            from fastfuncstuff.decomposition.icasso import icasso_plot

            icasso_plot(icasso_res, output_path=icasso_plot_path)
            _vprint(args.verb >= 1, f"ICASSO plot: {icasso_plot_path}")
        except Exception as exc:
            _vprint(args.verb >= 1, f"ICASSO plot failed: {exc}")
    else:
        method_name = getattr(args, "ica_method", "fastica")
        _vprint(
            args.verb >= 1,
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
        assert ica.components_ is not None and ica.mixing_ is not None  # set by fit above

        # Check convergence
        if ica.n_iter_ >= args.ica_max_iter:
            _vprint(
                args.verb >= 1,
                f"{method_name} did NOT converge after {args.ica_max_iter} iterations",
                t_step,
            )
            print("  ⚠ Consider increasing -ica_max_iter or checking data conditioning")
        else:
            _vprint(args.verb >= 1, f"{method_name} converged in {ica.n_iter_} iterations", t_step)

        # InfoMax diagnostics
        diag = getattr(ica, "diagnostics_", None)
        if diag and args.verb >= 1:
            lr_i = diag["learning_rate_initial"]
            lr_f = diag["learning_rate_final"]
            blk = diag["block_size"]
            n_blk = diag["n_blocks_per_epoch"]
            chg = diag["final_change"]
            _vprint(
                True,
                f"  lr: {lr_i:.6f} → {lr_f:.6f}, block={blk}, "
                f"blocks/epoch={n_blk}, final_change={chg:.2e}",
            )
            if diag.get("extended"):
                _vprint(
                    True,
                    f"  sub-Gaussian: {diag['n_sub_gaussian']}, "
                    f"super-Gaussian: {diag['n_super_gaussian']}",
                )

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

        _trace_single_pca = getattr(args, "trace", None)
        if _trace_single_pca is not None and hasattr(ica, "pca_") and ica.pca_ is not None:
            from pathlib import Path as _Ptr

            import numpy as _np_trace

            _td = _Ptr(_trace_single_pca) / run_tag
            _td.mkdir(parents=True, exist_ok=True)
            _ev = ica.pca_.explained_variance_.detach().cpu().numpy()
            _evecs = ica.pca_._eigenvectors.detach().cpu().numpy()
            _np_trace.save(str(_td / "pca_eigenvalues.npy"), _ev)
            _sqrt_ev = np.sqrt(np.maximum(_ev, 1e-12))
            _WM = (_evecs / _sqrt_ev[np.newaxis, :]).T
            _DWM = _evecs * _sqrt_ev[np.newaxis, :]
            _np_trace.save(str(_td / "white_matrix.npy"), _WM)
            _np_trace.save(str(_td / "dewhite_matrix.npy"), _DWM)
            if hasattr(ica.pca_, "components_") and ica.pca_.components_ is not None:
                _np_trace.save(
                    str(_td / "pca_components.npy"), ica.pca_.components_.detach().cpu().numpy()
                )
            _vprint(args.verb >= 1, f"Trace: PCA whitening → {_td}")

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
        _vprint(args.verb >= 1, f"Sign-flipped {n_flipped} ICs to positive-maximum orientation")

    # --- Compute ordering metrics ---
    # Two orderings are exposed: share of summed spatial-map stdev, and an
    # explained-variance-share proxy. Default sort is explained-share, since that is
    # what "component 1 is the biggest" is normally taken to mean.
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
    sort_key = stdev_share if args.ordering == "stdev" else explained_share_t
    sort_idx = torch.argsort(sort_key, descending=True)
    explained_share = explained_share_t[sort_idx].detach().cpu().numpy().astype(np.float32)
    stdev_share_sorted = stdev_share[sort_idx].detach().cpu().numpy().astype(np.float32)

    # Convert explained-share to total-share using retained PCA variance.
    scree = np.asarray(pca_diag["scree_ratio"], dtype=np.float64)
    retained_frac = float(np.clip(np.sum(scree[:n_components]), 0.0, 1.0))
    total_share = (explained_share * retained_frac).astype(np.float32)

    _vprint(
        args.verb >= 1,
        f"IC explained-share: top3 = {explained_share[0] * 100:.2f}%, {explained_share[1] * 100:.2f}%, {explained_share[2] * 100:.2f}% "
        f"(sorted by explained-share)",
    )
    _vprint(
        args.verb >= 1,
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

    # --- Noise normalisation of the IC maps ---
    # Raw IC maps carry the arbitrary scale of the unmixing matrix, so they are put into
    # units of the residual noise at each voxel:
    #   per-component  pow(diag(unmix @ unmix.T), -0.5)  removes the unmixing scale
    #   per-voxel      1 / (stdev(Data - mix @ IC) * sqrt((T-1)/(T-K)))
    # The sqrt((T-1)/(T-K)) is the degrees-of-freedom correction for the K components
    # already fitted, without which the residual understates the noise (the same
    # correction, for the same reason, as in decomposition/varnorm.py).
    # NOTE: must run BEFORE timecourse var_norm — otherwise mixing @ components
    # no longer reconstructs x_t and resid_std is corrupted, compressing IC peaks ~5x.
    raw_oic_np: np.ndarray | None = None
    if x_t is not None:
        # Snapshot raw IC maps for melodic_oIC.nii.gz before noise-norm rescales them.
        raw_oic_np = components.detach().cpu().numpy().astype(np.float32)
        _vprint(args.verb >= 1, "Applying MELODIC-style noise normalization ...")
        _nn_trace_dir = None
        _trace_nn_flag = getattr(args, "trace", None)
        if _trace_nn_flag is not None:
            from pathlib import Path as _Pnn

            _nn_trace_dir = _Pnn(_trace_nn_flag) / run_tag
        components, noise_norm_msg = ica_workflow.apply_melodic_noise_normalization(
            components=components,
            mixing=mixing,
            x_t=x_t,
            trace_dir=_nn_trace_dir,
        )
        _vprint(args.verb >= 1, noise_norm_msg)

    mixing_amplitude_np: np.ndarray | None = None
    if args.var_norm:
        # Capture pre-var_norm std — var_norm sets std(mixing[:,k])=1, so using
        # std(mixing_varnormed) in PSC would inflate amplitudes ~7-10×.
        mixing_amplitude_np = mixing.std(dim=0).cpu().numpy()  # (K,)
        mixing = mixing - mixing.mean(dim=0, keepdim=True)
        mixing_std = torch.clamp(mixing.std(dim=0, keepdim=True), min=1e-8)
        mixing = mixing / mixing_std

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
            from fastfuncstuff.design.hrf import get_spmg1_hrf
            from fastfuncstuff.design.matrices import (
                build_event_design_microtime,
                commensurate_microtime_dt,
            )

            n_bids_conds = len(bids_task_onsets)
            # Extract this run's onsets (wrapped in list for single-run onset matrix)
            onsets_this_run = [[bids_task_onsets[cidx][run_idx]] for cidx in range(n_bids_conds)]
            # TR is resolved per run here, so the grid is snapped per run too.
            microtime_dt = commensurate_microtime_dt(tr, args.microtime_dt)
            hrf = get_spmg1_hrf(microtime_dt=microtime_dt, device=device)
            design_tc = build_event_design_microtime(
                all_onsets=onsets_this_run,
                durations=bids_task_durations,
                hrf_bases=hrf,
                n_timepoints_per_run=[n_t],
                tr=tr,
                microtime_dt=microtime_dt,
                device=device,
            )
            assert isinstance(design_tc, torch.Tensor)
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
    nii_ext = pfx.nifti_ext

    # Output layout: every generated file lives inside <prefix>.ica/ffs_outputs/.
    # melodic_compat symlinks (melodic_IC.nii.gz, melodic_mix, …) sit one
    # level up in <prefix>.ica/ and point INTO ffs_outputs/, so the parent
    # directory stays clean and there are no symlinks-above-folder gymnastics.
    _basename = Path(pfx.stem).name
    _parent_dir = Path(pfx.stem).parent
    _is_single_run = int(getattr(args, "_n_runs_total", 1)) == 1
    compat_dir = _parent_dir / (
        f"{_basename}.ica" if _is_single_run else f"{_basename}_{run_tag}.ica"
    )
    _ffs_dir = compat_dir / "ffs_outputs"
    _ffs_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = _ffs_dir / _basename

    comp_np = components.detach().cpu().numpy().astype(np.float32)
    mixing_np = mixing.detach().cpu().numpy().astype(np.float32)

    _trace_single_ica = getattr(args, "trace", None)
    if _trace_single_ica is not None:
        from pathlib import Path as _Pica

        from scipy.stats import kurtosis as _kurt_t
        from scipy.stats import skew as _skew_t

        _td = _Pica(_trace_single_ica) / run_tag
        _td.mkdir(parents=True, exist_ok=True)
        np.save(str(_td / "ic_maps.npy"), comp_np)
        if raw_oic_np is not None:
            np.save(str(_td / "oic.npy"), raw_oic_np)
        np.savetxt(str(_td / "mix_matrix"), mixing_np, fmt="%.10g")
        if hasattr(explained_share, "tolist"):
            _es = (
                explained_share.tolist()
                if hasattr(explained_share, "tolist")
                else list(explained_share)
            )
        else:
            _es = list(explained_share)
        if hasattr(stdev_share_sorted, "tolist"):
            _ss = (
                stdev_share_sorted.tolist()
                if hasattr(stdev_share_sorted, "tolist")
                else list(stdev_share_sorted)
            )
        else:
            _ss = list(stdev_share_sorted)
        _ic_kurt = np.array([_kurt_t(c) for c in comp_np])
        _ic_skew = np.array([_skew_t(c) for c in comp_np])
        _ic_stats = np.column_stack([_es, _ss, _ic_kurt, _ic_skew])
        np.savetxt(str(_td / "ICstats"), _ic_stats, fmt="%.10g")
        _vprint(args.verb >= 1, f"Trace: ICA outputs → {_td}")

    # Free GPU tensors — we have numpy copies now
    del components, mixing
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- Save spatial maps ---
    _vsection(args.verb >= 1, "Save Outputs")
    t_step = time.time()
    _vprint(args.verb >= 1, "Saving ICA spatial maps ...")
    save_items = [
        (comp_np, f"{out_prefix}_{run_tag}_ica_maps{nii_ext}", "maps"),
    ]
    for data_arr, fname, _label in tqdm(
        save_items, desc="  Saving NIfTI", leave=False, disable=not args.verb >= 1
    ):
        decomposition_io.save_masked_component_maps_4d(
            components_kv=data_arr,
            mask3d=mask3d,
            shape3d=shape3d,
            affine=affine,
            out_file=Path(fname),
        )
    _vprint(args.verb >= 1, f"Maps saved: {out_prefix}_{run_tag}_ica_maps{nii_ext}", t_step)

    # --- Save timecourses ---
    np.savetxt(
        f"{out_prefix}_{run_tag}_ica_timecourses.1D",
        mixing_np,
        fmt="%.6f",
        delimiter="\t",
    )
    _vprint(args.verb >= 1, f"Timecourses saved: {out_prefix}_{run_tag}_ica_timecourses.1D")

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
        _vprint(args.verb >= 1, f"Correlation plot saved: {corr_plot}")

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
        _vprint(args.verb >= 1, f"Spectral correlation plot saved: {spectral_plot}")

    # --- Mixture model z-maps ---
    mixture_meta = []
    z_maps = None
    p_maps = None
    thresh_z_maps = None
    if args.save_mixture_z:
        _vsection(args.verb >= 1, "Mixture Model (GGM)")
        t_step = time.time()
        n_comps_total = comp_np.shape[0]
        _vprint(args.verb >= 1, f"Fitting GGM for {n_comps_total} components on {device} ...")
        comp_tensor = torch.as_tensor(comp_np, device=device)
        z_tensor, p_tensor, mixture_meta = batch_mixture_zscores(
            comp_tensor,
            device=device,
            verbose=args.verb >= 1,
        )
        z_maps = z_tensor.cpu().numpy().astype(np.float32)
        p_maps = p_tensor.cpu().numpy().astype(np.float32)
        thresh_z_maps = z_maps.copy()
        thresh_z_maps[p_maps < float(args.mm_thresh)] = 0.0
        del comp_tensor, z_tensor, p_tensor
        n_conv = sum(1 for m in mixture_meta if m.get("converged", False))
        _vprint(args.verb >= 1, f"GGM done: {n_conv}/{n_comps_total} converged", t_step)

        ggm_saves = [
            (z_maps, f"{out_prefix}_{run_tag}_ica_zmaps{nii_ext}"),
            (p_maps, f"{out_prefix}_{run_tag}_ica_signalprob{nii_ext}"),
            (thresh_z_maps, f"{out_prefix}_{run_tag}_ica_thresh_zmaps{nii_ext}"),
        ]
        for data_arr, fname in tqdm(
            ggm_saves, desc="  Saving GGM NIfTI", leave=False, disable=not args.verb >= 1
        ):
            decomposition_io.save_masked_component_maps_4d(
                components_kv=data_arr,
                mask3d=mask3d,
                shape3d=shape3d,
                affine=affine,
                out_file=Path(fname),
            )
        _vprint(args.verb >= 1, "Z-maps and signal-prob maps saved")

    _trace_mm_flag = getattr(args, "trace", None)
    if _trace_mm_flag is not None and mixture_meta:
        from pathlib import Path as _Pmm

        _mm_td = _Pmm(_trace_mm_flag) / run_tag
        _mm_td.mkdir(parents=True, exist_ok=True)
        _mm_keys = [
            "mu_noise",
            "sigma_noise",
            "mu_signal_pos",
            "sigma_signal_pos",
            "mu_signal_neg",
            "sigma_signal_neg",
            "pi_noise",
            "pi_pos",
            "pi_neg",
            "mixing_signal",
        ]
        _mm_arr = np.array(
            [[m.get(k, 0.0) for k in _mm_keys] for m in mixture_meta],
            dtype=np.float64,
        )
        np.save(str(_mm_td / "mmstats.npy"), _mm_arr)
        _vprint(args.verb >= 1, f"Trace: mmstats → {_mm_td}")

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
            comp_kv_for_stats=comp_np,
            oic_components_kv=raw_oic_np,
            varnorm_std_v=varnorm_std_np,
            mixing_amplitude_k=mixing_amplitude_np,
            write_per_comp_stats=getattr(args, "per_comp_stats", False),
            write_psc_prob=getattr(args, "psc_bucket", True),
            write_zp=getattr(args, "zp_bucket", True),
            psc_clip=float(args.psc_clip),
        )
        ic_target = (
            Path(f"{out_prefix}_{run_tag}_ica_zmaps{nii_ext}")
            if z_maps is not None and Path(f"{out_prefix}_{run_tag}_ica_zmaps{nii_ext}").exists()
            else Path(f"{out_prefix}_{run_tag}_ica_maps{nii_ext}")
        )
        _vprint(args.verb >= 1, f"MELODIC melodic_IC target: {ic_target}")
        _vprint(args.verb >= 1, f"MELODIC-compatible outputs: {compat_dir}")

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
        verbose=bool(args.verb >= 1),
    )
    depth_lag_results = depth_lag_pack["depth_lag_results"]
    depth_lag_matrix_seconds = depth_lag_pack["depth_lag_matrix_seconds"]
    depth_lag_matrix_r = depth_lag_pack["depth_lag_matrix_r"]
    depth_lag_plot = depth_lag_pack["depth_lag_plot"]
    depth_lag_method = depth_lag_pack["depth_lag_method"]

    elapsed_total = time.time() - t_total
    _vprint(args.verb >= 1, f"Run {run_idx + 1} total elapsed: {elapsed_total:.1f}s")

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
        "component_ordering": args.ordering,
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

    _vsection(args.verb >= 1, "Temporal Concatenation ICA")
    _vprint(args.verb >= 1, f"Concatenating {n_runs} runs for single ICA decomposition")

    # --- Load all runs, apply per-run preprocessing, collect masked data ---
    run_data_list: list[torch.Tensor] = []  # each (n_vox, n_t_run)
    run_lengths: list[int] = []
    mask3d: np.ndarray | None = None
    const_bad_vox: np.ndarray | None = None  # union of per-run constant voxels
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
        _vsection(args.verb >= 1, f"Load Run {ri + 1}/{n_runs}")
        _vprint(args.verb >= 1, f"Loading {run_file} ...")
        img = load_nifti(run_file)
        data = img.get_fdata(dtype=np.float32)
        run_shape3d = data.shape[:3]
        n_t_run = data.shape[3]
        voxel_sizes = tuple(float(v) for v in img.header.get_zooms()[:3])
        _vprint(args.verb >= 1, f"  shape={data.shape}, dtype={data.dtype}", t_step)

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
            _vprint(args.verb >= 1, f"  Blurred FWHM={args.do_blur:.1f}mm", t_step)

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
                _vprint(args.verb >= 1, f"  Using provided mask: {int(mask3d.sum()):,} voxels")
            elif not args.no_auto_mask:
                from fastfuncstuff.processing.mask import automask

                mask3d = (
                    automask(
                        torch.as_tensor(run_mean3d),
                        dilate_extra=2,
                        device=device,
                        verbose=args.verb >= 1,
                    )
                    .cpu()
                    .numpy()
                )
                _vprint(args.verb >= 1, f"  Auto-mask from run 1: {int(mask3d.sum()):,} voxels")

            if mask3d is not None and mask3d.shape != shape3d:
                raise ValueError(f"Mask shape {mask3d.shape} != data shape {shape3d}")

            # Estimate spatial smoothness from first run
            resels_accum, fwhm_geo_accum, _ = _estimate_smoothness_resels_acf(
                data,
                voxel_sizes,
                mask=mask3d,
                device=device,
                per_axis=bool(getattr(args, "smoothness_per_axis", False)),
                corder=int(getattr(args, "smoothness_corder", -1)),
                verbose=args.verb >= 1,
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

        # A voxel constant in ANY run is unusable across the whole set, so
        # accumulate here and prune the mask once every run has been read.
        if mask3d is not None and args.drop_constant:
            _run_bad = find_constant_voxels(run_vox_np)
            const_bad_vox = _run_bad if const_bad_vox is None else (const_bad_vox | _run_bad)

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
            run_vox = _check_finite(run_vox, f"run{ri + 1}-post-scale", args.verb >= 1)
            _vprint(args.verb >= 1, "  Scaled to percent signal", t_step)

        # --- Per-run polort (block-diag here = single-run) ---
        # Tcat default: -1 (off) to defer to high-pass; single-run default: 0.
        polort_default = -1 if n_runs > 1 else 0
        polort_r = polort_default if args.polort is None else int(args.polort)
        if polort_r >= 0:
            run_vox = apply_polort_projection(
                run_vox,
                polort=polort_r,
                device=device,
                run_starts=[0],
            )
            run_vox = _check_finite(run_vox, f"run{ri + 1}-post-polort", args.verb >= 1)

        # --- Per-run high-pass filter ---
        if args.high_pass is not None and args.high_pass > 0:
            nyquist = 0.5 / float(tr if tr is not None else run_tr)
            if args.high_pass >= nyquist:
                raise ValueError(
                    f"High-pass cutoff ({args.high_pass:.4f} Hz) >= Nyquist ({nyquist:.4f} Hz)"
                )
            run_vox = apply_high_pass_fft(
                run_vox,
                tr=float(tr if tr is not None else run_tr),
                high_pass_hz=float(args.high_pass),
                run_starts=[0],
            )
            run_vox = _check_finite(run_vox, f"run{ri + 1}-post-hp", args.verb >= 1)

        # --- Per-run temporal mean-center ---
        run_vox = run_vox - run_vox.mean(dim=1, keepdim=True)

        run_data_list.append(run_vox)
        run_lengths.append(n_t_run)
        _vprint(args.verb >= 1, f"  Run {ri + 1}: {n_vox_masked:,} vox x {n_t_run} timepoints")

    mask3d, _n_drop = _prune_constant_runs(mask3d, const_bad_vox, run_data_list, args.verb >= 1)
    if _n_drop:
        n_vox_masked = int(run_data_list[0].shape[0])

    # Average the mean images
    mean3d = mean3d_accum / n_runs

    polort = (
        -1
        if (args.polort is None and n_runs > 1)
        else (0 if args.polort is None else int(args.polort))
    )

    # run_starts / total_t
    run_starts = []
    offset = 0
    for rl in run_lengths:
        run_starts.append(offset)
        offset += rl
    total_t = int(sum(run_lengths))

    # --- Temporal concatenation ---
    # Incremental group PCA (Smith et al. 2014, see decomposition/migp.py) exists to bound
    # memory when the full concatenation will not fit. At typical ffs run counts it does
    # fit, so we stack outright and take one exact SVD on (T_total, V) rather than an
    # approximation. Each run is divided by n_runs first so every run contributes equally
    # to the group subspace instead of in proportion to its length.
    _vsection(args.verb >= 1, "Temporal Concatenation")
    t_step = time.time()
    if getattr(args, "migp", False):
        from fastfuncstuff.decomposition.migp import migp_reduce

        migp_n = args.migp_n
        run_indices = list(range(n_runs))
        if getattr(args, "migp_shuffle", None):
            run_indices = [int(x) for x in args.migp_shuffle.split(",")]
            if len(run_indices) != n_runs or set(run_indices) != set(range(n_runs)):
                raise ValueError(
                    f"-migp_shuffle must be a permutation of 0..{n_runs - 1}, "
                    f"got: {args.migp_shuffle}"
                )
            _vprint(args.verb >= 1, f"MIGP shuffle: {run_indices}")
        runs_tv = [run_data_list[ri].T for ri in run_indices]  # list of (T_r, V)
        del run_data_list
        data_tv = migp_reduce(
            runs_tv,
            migp_n=migp_n,
            migp_factor=args.migp_factor,
            scale_by_n=True,
            device=device,
            verbose=args.verb >= 1,
        ).contiguous()
        del runs_tv
        data_tv = _check_finite(data_tv, "data_tv", args.verb >= 1)
        _vprint(
            args.verb >= 1,
            f"MIGP reduced shape: {tuple(data_tv.shape)} (migp_n, V); "
            f"migp_n={data_tv.shape[0]}{' (auto)' if migp_n is None else ''}",
            t_step,
        )
    else:
        scale = 1.0 / float(n_runs)
        # Build (T_total, V) concat; per-run (V, T_r) → transpose and scale.
        data_tv = torch.cat(
            [run_data_list[ri].T * scale for ri in range(n_runs)], dim=0
        ).contiguous()
        del run_data_list
        data_tv = _check_finite(data_tv, "data_tv", args.verb >= 1)
        _vprint(args.verb >= 1, f"Concat shape: {tuple(data_tv.shape)} (T_total, V)", t_step)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- Trace: dump MIGP/concat output BEFORE varnorm ---
    # Lets us recover MELODIC's noise_std map by dividing FFS pre-varnorm by
    # MELODIC's post-varnorm concat_data.nii.gz, isolating varnorm divergence.
    _trace_dir_pre = getattr(args, "trace", None)
    if _trace_dir_pre:
        _td_pre: Path = Path(_trace_dir_pre)
        _td_pre.mkdir(parents=True, exist_ok=True)
        np.save(_td_pre / "migp_pre_varnorm.npy", data_tv.cpu().numpy())
        if mask3d is not None:
            import nibabel as _nib

            _nib.save(
                _nib.Nifti1Image(mask3d.astype(np.uint8), affine),
                str(_td_pre / "mask.nii.gz"),
            )
        _vprint(args.verb >= 1, f"Trace: pre-varnorm MIGP → {_td_pre}/migp_pre_varnorm.npy")

    # --- Variance normalization on the concatenated matrix ---
    # One voxel-wise varnorm over the fully-concatenated data, not per run: a voxel's
    # noise scale is a property of the voxel, so estimating it per run lets run-to-run
    # scale differences survive into the concatenation as spurious structure. -sep_vn is
    # the escape hatch (per-file varnorm before concat) for debugging that divergence.
    vn_scope = "none"
    varnorm_std_np: np.ndarray | None = None
    if args.voxel_norm:
        from fastfuncstuff.decomposition.varnorm import apply_noise_std_map, noise_std_map

        _vsection(args.verb >= 1, "Variance Normalization (joined on concat)")
        t_step = time.time()
        # noise_std_map expects (V, T); our data_tv is (T, V) → pass transpose.
        noise_std_v, const_mask_vn, n_const_vn = noise_std_map(data_tv.T)
        # Trace: persist FFS varnorm map BEFORE consuming/freeing it.
        _trace_dir_vn = getattr(args, "trace", None)
        if _trace_dir_vn:
            from pathlib import Path as _PathVN

            _td_vn = _PathVN(_trace_dir_vn)
            _td_vn.mkdir(parents=True, exist_ok=True)
            np.save(_td_vn / "ffs_noise_std.npy", noise_std_v.cpu().numpy())
            np.save(_td_vn / "ffs_const_mask.npy", const_mask_vn.cpu().numpy())
        data_tv_vt = apply_noise_std_map(data_tv.T, noise_std_v, const_mask_vn)
        data_tv = data_tv_vt.T.contiguous()
        # Trace: also dump the post-varnorm concat (T,V) for direct comparison
        # to MELODIC's concat_data.nii.gz.
        if _trace_dir_vn:
            np.save(_PathVN(_trace_dir_vn) / "migp_post_varnorm.npy", data_tv.cpu().numpy())
        # Keep the map alive for PSC computation (passed to write_melodic_compat_outputs).
        varnorm_std_np = noise_std_v.cpu().numpy()  # (V,) in data units pre-varnorm
        del data_tv_vt, const_mask_vn
        _vprint(
            args.verb >= 1,
            f"Joined varnorm: residual-noise, {n_const_vn} constant voxels",
            t_step,
        )
        vn_scope = "joined"

    # --- Sanity check ---
    concat_var = float(torch.var(data_tv).item())
    if concat_var < 1e-10:
        raise ValueError(f"concat variance is ~0 after preprocessing ({concat_var:.2e})")
    _vprint(args.verb >= 1, f"Concat variance: {concat_var:.4f}")

    # --- Spatial smoothness / effective DOF ---
    # The effective sample size the Laplace evidence is scaled by; see the note at the
    # single-run site above and decomposition/model_order.
    resels = resels_accum
    fwhm_geo = fwhm_geo_accum
    n_eff = effective_sample_size_from_resels(n_vox_masked, resels, floor=int(total_t))
    _vprint(
        args.verb >= 1,
        f"Effective spatial DOF: n_eff={n_eff:,} (raw V={n_vox_masked:,}, resels={resels:.2f})",
    )

    # --- Component count estimation (on concat, MELODIC-style) ---
    _vsection(args.verb >= 1, "Component Estimation (on concat)")
    t_step = time.time()

    # Dim-est expects (V, T) layout internally — pass data_tv.T.
    T_total = int(data_tv.shape[0])
    if args.max_auto_components <= 1.0:
        max_auto_k = max(5, int((T_total - 2) * args.max_auto_components))
        if args.max_auto_components >= 0.66:
            max_auto_k = max(max_auto_k, max(5, T_total - 2))
    else:
        max_auto_k = int(args.max_auto_components)
    _vprint(args.verb >= 1, f"max_auto_components: {max_auto_k}")
    _vprint(args.verb >= 1, f"Method: {args.num_comps}")

    n_eff_for_model_order = n_eff
    data_for_model_order_vt = data_tv.T  # (V, T_total)
    if isinstance(num_spec, str) and num_spec in {"auto", "laplace"}:
        data_for_model_order_vt, _ = ica_workflow.filter_low_variance_voxels(data_vox_t=data_tv.T)
        n_vox_model_order = int(data_for_model_order_vt.shape[0])
        n_eff_for_model_order = max(
            int(T_total), effective_sample_size_from_resels(n_vox_model_order, resels)
        )
        _vprint(
            args.verb >= 1,
            f"MELODIC dim-est filter: kept {n_vox_model_order:,}/{n_vox_masked:,} voxels; "
            f"n_eff={n_eff_for_model_order:,} (resels={resels:.2f})",
        )

    use_mp = not args.auto_no_mp

    n_components, pca_diag, num_diag = estimate_ica_component_count(
        data_vox_t=data_for_model_order_vt,
        method=num_spec,
        max_auto_components=max_auto_k,
        auto_min_components=args.auto_min_components,
        auto_var_threshold=args.auto_var_threshold,
        use_mp_prior=use_mp,
        n_eff=n_eff_for_model_order,
        device=device,
        verbose=args.verb >= 1,
        capture_ppca_trace=bool(args.ppca_debug_dump),
    )
    order = int(n_components)
    _vprint(
        args.verb >= 1,
        f"Selected {order} components (mode={num_diag.get('mode', '?')})",
        t_step,
    )

    # Opt-in: how many of those components survive an independent decomposition of half
    # the runs? Reported, never used to change `order` -- it is evidence about the count
    # you chose, and overwriting the count with it would remove the check.
    if getattr(args, "split_half", False):
        if len(run_lengths) < 2:
            print("  ⚠ -split_half needs at least 2 runs; skipping.")
        else:
            from fastfuncstuff.decomposition.stability import split_half_reproducibility

            _vsection(args.verb >= 1, "Split-half reproducibility")
            t_sh = time.time()
            _off = 0
            _run_mats = []
            for _rl in run_lengths:
                _run_mats.append(data_tv[_off : _off + _rl].cpu().numpy())
                _off += _rl
            rep = split_half_reproducibility(
                _run_mats,
                n_components=order,
                threshold=float(args.split_half_thresh),
                n_splits=int(args.split_half_splits),
                device=device,
                base_seed=int(getattr(args, "seed", 0) or 0),
                verbose=args.verb >= 2,
            )
            del _run_mats
            _vprint(
                args.verb >= 1,
                f"Split-half: {rep.n_reproducible}/{order} components reproduce at "
                f"|r| >= {rep.threshold} "
                f"(median matched |r| = {float(np.median(rep.matched_r)):.3f})",
                t_sh,
            )
            if rep.n_reproducible < 0.5 * order:
                print(
                    f"  ⚠ under half the components reproduce across runs; "
                    f"{order} is likely too many."
                )
            num_diag = {**num_diag, "split_half": rep.as_dict()}

    del data_for_model_order_vt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- PCA + whitening on the concatenation ---
    # Eigen-decomp of temporal covariance (T_total, T_total), keep top `order`.
    # WM = E^T / sqrt(L) : (order, T_total); DWM = E * sqrt(L) : (T_total, order).
    # whitened = WM @ data_tv  →  (order, V).
    _vsection(args.verb >= 1, "Whitening (joined on concat)")
    t_step = time.time()
    # Centre each timepoint row across voxels: the covariance being eigendecomposed is
    # temporal, over spatially-demeaned data, not a raw cross-product.
    row_mean = data_tv.mean(dim=1, keepdim=True)
    data_centered = data_tv - row_mean
    cov_t = (data_centered @ data_centered.T) / float(n_vox_masked)
    evals_t, evecs_t = torch.linalg.eigh(cov_t)
    _trace_cov = cov_t.cpu().numpy() if getattr(args, "trace", None) else None
    del cov_t
    idx = torch.argsort(evals_t, descending=True)
    evals_t = torch.clamp(evals_t[idx][:order], min=1e-12)
    evecs_t = evecs_t[:, idx][:, :order]
    sqrt_ev = torch.sqrt(evals_t)
    WM = (evecs_t / sqrt_ev.unsqueeze(0)).T  # (order, T_total)
    DWM = evecs_t * sqrt_ev.unsqueeze(0)  # (T_total, order)
    x_t = WM @ data_centered  # (order, V)

    # --- Trace: dump PCA intermediates ---
    trace_dir = getattr(args, "trace", None)
    if trace_dir:
        from pathlib import Path as _P  # noqa: N814

        _td = _P(trace_dir)
        _td.mkdir(parents=True, exist_ok=True)
        _evals_full = evals_t.cpu().numpy()
        _eigenvalues_pct = np.cumsum(_evals_full) / _evals_full.sum()
        np.savetxt(_td / "eigenvalues_adjusted", _evals_full, fmt="%.10g")
        np.savetxt(_td / "eigenvalues_percent", _eigenvalues_pct, fmt="%.10g")
        np.save(_td / "white_matrix.npy", WM.cpu().numpy())
        np.save(_td / "dewhite_matrix.npy", DWM.cpu().numpy())
        np.save(_td / "pca_eigenvalues.npy", _evals_full)
        np.save(_td / "pca_components.npy", x_t.cpu().numpy())  # (k, V) whitened spatial maps
        if _trace_cov is not None:
            np.save(_td / "cov_temporal.npy", _trace_cov)
        if "ppca_trace" in pca_diag:
            _ll = np.array(pca_diag["ppca_trace"].get("log_evidence", []), dtype=np.float64)
            if len(_ll) > 0:
                np.save(str(_td / "log_evidence.npy"), _ll)
        _vprint(args.verb >= 1, f"Trace: PCA intermediates → {_td}")

    del data_centered, evals_t, evecs_t, sqrt_ev
    # Keep unwhitened (T_total, V) for noise-norm + mixing dewhitening.
    orig_concat_tv = data_tv
    del data_tv
    whiten_mode = "joined"
    _vprint(
        args.verb >= 1,
        f"Whiten: order={order}, x_t shape {tuple(x_t.shape)}",
        t_step,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- ICA ---
    _vsection(args.verb >= 1, "ICA Decomposition")
    t_step = time.time()
    if args.icasso:
        method_name = getattr(args, "ica_method", "fastica")
        _vprint(
            args.verb >= 1, f"ICASSO ({method_name}): {args.icasso_runs} runs, k={n_components}"
        )
        icasso_res = icasso(
            X=x_t,
            n_components=n_components,
            n_runs=args.icasso_runs,
            pca_components=n_components,
            min_stability=args.icasso_min_stability,
            device=device,
            verbose=args.verb >= 1,
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
        _vprint(args.verb >= 1, f"ICASSO done ({icasso_res['n_stable']} stable)", t_step)

        # Save ICASSO diagnostic plot
        icasso_plot_path = f"{args.prefix}_tempconcat_icasso.png"
        try:
            from fastfuncstuff.decomposition.icasso import icasso_plot

            icasso_plot(icasso_res, output_path=icasso_plot_path)
            _vprint(args.verb >= 1, f"ICASSO plot: {icasso_plot_path}")
        except Exception as exc:
            _vprint(args.verb >= 1, f"ICASSO plot failed: {exc}")
    else:
        method_name = getattr(args, "ica_method", "fastica")
        _vprint(
            args.verb >= 1,
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
        assert ica.components_ is not None and ica.mixing_ is not None  # set by fit above
        if ica.n_iter_ >= args.ica_max_iter:
            _vprint(
                args.verb >= 1,
                f"{method_name} did NOT converge after {args.ica_max_iter} iterations",
            )
        else:
            _vprint(args.verb >= 1, f"{method_name} converged in {ica.n_iter_} iterations", t_step)

        # InfoMax diagnostics
        diag = getattr(ica, "diagnostics_", None)
        if diag and args.verb >= 1:
            lr_i = diag["learning_rate_initial"]
            lr_f = diag["learning_rate_final"]
            blk = diag["block_size"]
            n_blk = diag["n_blocks_per_epoch"]
            chg = diag["final_change"]
            _vprint(
                True,
                f"  lr: {lr_i:.6f} → {lr_f:.6f}, block={blk}, "
                f"blocks/epoch={n_blk}, final_change={chg:.2e}",
            )
            if diag.get("extended"):
                _vprint(
                    True,
                    f"  sub-Gaussian: {diag['n_sub_gaussian']}, "
                    f"super-Gaussian: {diag['n_super_gaussian']}",
                )

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

    # --- Dewhiten mixing: (order, k) → (T_total, k) via DWM ---
    # ICA on x_t=(order,V) returns mixing=(order, k) in PCA-space samples.
    # Map back to data space with DWM, then split per-run for TCs.
    mixing_pca = mixing  # (order, k)
    mixing = DWM @ mixing_pca  # (T_total, k)
    del mixing_pca
    per_run_tcs: list[np.ndarray] = []
    offs = 0
    for ri in range(n_runs):
        T_r = run_lengths[ri]
        block = mixing[offs : offs + T_r]
        per_run_tcs.append(block.detach().cpu().numpy().astype(np.float32))
        offs += T_r
    # Swap x_t to the unwhitened concat so noise-norm residuals are in data space.
    del x_t
    x_t = orig_concat_tv
    orig_concat_tv = None
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
        _vprint(args.verb >= 1, f"Sign-flipped {n_flipped} ICs")

    # --- Ordering by explained share ---
    ic_stdev = torch.std(components, dim=1)
    total_stdev = float(ic_stdev.sum().item())
    stdev_share = ic_stdev / max(total_stdev, 1e-15)

    mix_var = torch.var(mixing, dim=0, unbiased=False)
    total_mix_var = float(mix_var.sum().item())
    explained_share_t = mix_var / max(total_mix_var, 1e-15)

    sort_key = stdev_share if args.ordering == "stdev" else explained_share_t
    sort_idx = torch.argsort(sort_key, descending=True)
    sort_idx_np = sort_idx.detach().cpu().numpy()
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

    # --- MELODIC-style noise normalization ---
    # Must run BEFORE timecourse var_norm — see note in single-run path.
    raw_oic_np: np.ndarray | None = None
    if x_t is not None:
        raw_oic_np = components.detach().cpu().numpy().astype(np.float32)
        _vprint(args.verb >= 1, "Applying MELODIC-style noise normalization ...")
        _nn_trace_dir = _P(trace_dir) if trace_dir else None
        components, noise_norm_msg = ica_workflow.apply_melodic_noise_normalization(
            components=components,
            mixing=mixing,
            x_t=x_t,
            trace_dir=_nn_trace_dir,
        )
        _vprint(args.verb >= 1, noise_norm_msg)

    mixing_amplitude_np: np.ndarray | None = None
    if args.var_norm:
        # Capture pre-var_norm std — var_norm sets std(mixing[:,k])=1, so using
        # std(mixing_varnormed) in PSC would inflate amplitudes ~7-10×.
        mixing_amplitude_np = mixing.std(dim=0).cpu().numpy()  # (K,)
        mixing = mixing - mixing.mean(dim=0, keepdim=True)
        mixing_std = torch.clamp(mixing.std(dim=0, keepdim=True), min=1e-8)
        mixing = mixing / mixing_std

    # --- Trace: dump ICA intermediates ---
    if trace_dir:
        from pathlib import Path as _P  # noqa: N814

        from scipy.stats import kurtosis as _kurt
        from scipy.stats import skew as _skew

        _td = _P(trace_dir)
        _td.mkdir(parents=True, exist_ok=True)
        comp_np = components.detach().cpu().numpy()
        mix_np = mixing.detach().cpu().numpy()
        np.save(_td / "ic_maps.npy", comp_np)
        if raw_oic_np is not None:
            np.save(_td / "oic.npy", raw_oic_np)
        np.savetxt(_td / "mix_matrix", mix_np, fmt="%.10g")
        if isinstance(WM, torch.Tensor):
            np.savetxt(_td / "white_matrix", WM.cpu().numpy(), fmt="%.10g")
        if isinstance(DWM, torch.Tensor):
            np.savetxt(_td / "dewhite_matrix", DWM.cpu().numpy(), fmt="%.10g")
        ic_kurt = np.array([_kurt(c) for c in comp_np])
        ic_skew = np.array([_skew(c) for c in comp_np])
        ic_stats = np.column_stack([explained_share, stdev_share_sorted, ic_kurt, ic_skew])
        np.savetxt(_td / "ICstats", ic_stats, fmt="%.10g")
        _vprint(args.verb >= 1, f"Trace: ICA intermediates → {_td}")
    del x_t
    if isinstance(WM, torch.Tensor):
        del WM
    if isinstance(DWM, torch.Tensor):
        del DWM
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- Save outputs (no run tag for concat) ---
    _vsection(args.verb >= 1, "Save Outputs")
    t_step = time.time()
    pfx = parse_prefix(str(args.prefix))
    nii_ext = pfx.nifti_ext

    # Output layout: see _run_single_ica for rationale.
    _basename = Path(pfx.stem).name
    _parent_dir = Path(pfx.stem).parent
    compat_dir = _parent_dir / f"{_basename}_concat.ica"
    _ffs_dir = compat_dir / "ffs_outputs"
    _ffs_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = _ffs_dir / _basename

    comp_np = components.detach().cpu().numpy().astype(np.float32)
    mixing_np = mixing.detach().cpu().numpy().astype(np.float32)
    del components, mixing
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Spatial maps
    _maps_out_file = Path(f"{out_prefix}_concat_ica_maps{nii_ext}")
    with spinner(f"Writing {_maps_out_file.name}"):
        decomposition_io.save_masked_component_maps_4d(
            components_kv=comp_np,
            mask3d=mask3d,
            shape3d=shape3d,
            affine=affine,
            out_file=_maps_out_file,
        )
    _vprint(args.verb >= 1, f"Maps: {out_prefix}_concat_ica_maps{nii_ext}", t_step)

    # Timecourses (concatenated, (T_total, k))
    np.savetxt(
        f"{out_prefix}_concat_ica_timecourses.1D",
        mixing_np,
        fmt="%.6f",
        delimiter="\t",
    )
    _vprint(args.verb >= 1, f"Timecourses: {out_prefix}_concat_ica_timecourses.1D")

    # Per-run timecourses (block-diag whiten path only)
    per_run_tc_paths: list[str] = []
    if per_run_tcs is not None:
        for ri, tc in enumerate(per_run_tcs):
            tc = tc[:, sort_idx_np]
            tc_path = f"{out_prefix}_concat_ica_timecourses_run{ri + 1:02d}.1D"
            np.savetxt(tc_path, tc, fmt="%.6f", delimiter="\t")
            per_run_tc_paths.append(tc_path)
        _vprint(
            args.verb >= 1,
            f"Per-run timecourses: {len(per_run_tc_paths)} files saved",
        )

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
        _vsection(args.verb >= 1, "Mixture Model (GGM)")
        t_step = time.time()
        comp_tensor = torch.as_tensor(comp_np, device=device)
        z_tensor, p_tensor, mixture_meta = batch_mixture_zscores(
            comp_tensor,
            device=device,
            verbose=args.verb >= 1,
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
        _vprint(args.verb >= 1, "Z-maps and signal-prob maps saved", t_step)

    if trace_dir and mixture_meta:
        _mm_keys = [
            "mu_noise",
            "sigma_noise",
            "mu_signal_pos",
            "sigma_signal_pos",
            "mu_signal_neg",
            "sigma_signal_neg",
            "pi_noise",
            "pi_pos",
            "pi_neg",
            "mixing_signal",
        ]
        _mm_arr = np.array(
            [[m.get(k, 0.0) for k in _mm_keys] for m in mixture_meta],
            dtype=np.float64,
        )
        np.save(_P(trace_dir) / "mmstats.npy", _mm_arr)

    # MELODIC compat (includes psc_prob and z_prob buckets in stats/ by default)
    if args.melodic_compat:
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
            comp_kv_for_stats=comp_np,
            oic_components_kv=raw_oic_np,
            varnorm_std_v=varnorm_std_np if args.voxel_norm else None,
            mixing_amplitude_k=mixing_amplitude_np,
            write_per_comp_stats=getattr(args, "per_comp_stats", False),
            write_psc_prob=getattr(args, "psc_bucket", True),
            write_zp=getattr(args, "zp_bucket", True),
            psc_clip=float(args.psc_clip),
        )
        _vprint(args.verb >= 1, f"MELODIC compat: {compat_dir}")

    elapsed_total = time.time() - t_total
    _vprint(args.verb >= 1, f"Concat ICA total elapsed: {elapsed_total:.1f}s")

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
        "voxel_norm_scope": vn_scope,
        "whiten_mode": whiten_mode,
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
        "component_ordering": args.ordering,
        "elapsed_seconds": round(elapsed_total, 2),
        "outputs": {
            "ica_maps": f"{out_prefix}_concat_ica_maps{nii_ext}",
            "ica_timecourses": f"{out_prefix}_concat_ica_timecourses.1D",
            "ica_timecourses_per_run": per_run_tc_paths if per_run_tc_paths else None,
            "pca_scree_plot": f"{out_prefix}_concat_pca_scree.png",
            "ica_zmaps": f"{out_prefix}_concat_ica_zmaps{nii_ext}" if args.save_mixture_z else None,
        },
        "mixture_model": mixture_meta if args.save_mixture_z else None,
    }

    with open(f"{out_prefix}_concat_ica_metadata.json", "w") as f:
        json.dump(concat_meta, f, indent=2)

    return concat_meta


def _temporal_ica_preprocess_runs(
    input_files: list[str],
    args,
    device: torch.device,
    shared_mask: np.ndarray | None,
) -> dict:
    """Load + preprocess each run for temporal ICA (mirrors the concat per-run loop).

    Returns processed per-run (V, T_i) tensors kept on CPU (dual regression moves
    them to `device` one at a time to bound VRAM), plus the shared mask/grid and
    smoothness diagnostics. Preprocessing order per run:
    load → blur → mask → scale → polort → high-pass → temporal mean-center.
    """
    n_runs = len(input_files)
    run_data_list: list[torch.Tensor] = []  # each (V, T_i) on CPU
    run_lengths: list[int] = []
    mask3d: np.ndarray | None = None
    const_bad_vox: np.ndarray | None = None  # union of per-run constant voxels
    shape3d: tuple[int, ...] | None = None
    affine: np.ndarray | None = None
    mean3d_accum: np.ndarray | None = None
    tr: float | None = None
    resels: float = 0.0
    fwhm_geo: float = 0.0
    n_vox_masked: int = 0

    polort_default = -1 if n_runs > 1 else 0
    polort = polort_default if args.polort is None else int(args.polort)

    for ri, run_file in enumerate(
        tqdm(input_files, desc="Preprocessing runs", disable=n_runs < 3, leave=True)
    ):
        img = load_nifti(run_file)
        data = img.get_fdata(dtype=np.float32)
        run_shape3d = data.shape[:3]
        n_t_run = data.shape[3]
        voxel_sizes = tuple(float(v) for v in img.header.get_zooms()[:3])

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

        if args.do_blur is not None and args.do_blur > 0:
            data = gaussian_blur_3d(
                data=data,
                fwhm_mm=float(args.do_blur),
                voxel_sizes=voxel_sizes,
                device=device,
                verbose=False,
            )

        run_mean3d = data.mean(axis=-1).astype(np.float32)
        mean3d_accum = run_mean3d.copy() if mean3d_accum is None else mean3d_accum + run_mean3d

        if ri == 0:
            if shared_mask is not None:
                mask3d = shared_mask
            elif not args.no_auto_mask:
                from fastfuncstuff.processing.mask import automask

                mask3d = (
                    automask(
                        torch.as_tensor(run_mean3d),
                        dilate_extra=2,
                        device=device,
                        verbose=args.verb >= 1,
                    )
                    .cpu()
                    .numpy()
                )
            if mask3d is not None and mask3d.shape != shape3d:
                raise ValueError(f"Mask shape {mask3d.shape} != data shape {shape3d}")
            resels, fwhm_geo, _ = _estimate_smoothness_resels_acf(
                data,
                voxel_sizes,
                mask=mask3d,
                device=device,
                per_axis=bool(getattr(args, "smoothness_per_axis", False)),
                corder=int(getattr(args, "smoothness_corder", -1)),
                verbose=args.verb >= 1,
            )

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

        # A voxel constant in ANY run is unusable across the whole set, so
        # accumulate here and prune the mask once every run has been read.
        if mask3d is not None and args.drop_constant:
            _run_bad = find_constant_voxels(run_vox_np)
            const_bad_vox = _run_bad if const_bad_vox is None else (const_bad_vox | _run_bad)

        run_vox = to_tensor(run_vox_np, device=device)
        del run_vox_np

        if args.do_scale:
            run_vox, _, _ = scale_to_percent_signal(run_vox, run_starts=[0], verbose=False)
            run_vox = _check_finite(run_vox, f"run{ri + 1}-post-scale", args.verb >= 1)

        if polort >= 0:
            run_vox = apply_polort_projection(run_vox, polort=polort, device=device, run_starts=[0])
            run_vox = _check_finite(run_vox, f"run{ri + 1}-post-polort", args.verb >= 1)

        if args.high_pass is not None and args.high_pass > 0:
            nyquist = 0.5 / float(tr)
            if args.high_pass >= nyquist:
                raise ValueError(
                    f"High-pass cutoff ({args.high_pass:.4f} Hz) >= Nyquist ({nyquist:.4f} Hz)"
                )
            run_vox = apply_high_pass_fft(
                run_vox, tr=float(tr), high_pass_hz=float(args.high_pass), run_starts=[0]
            )
            run_vox = _check_finite(run_vox, f"run{ri + 1}-post-hp", args.verb >= 1)

        # Per-run temporal mean-center; dual regression demeans
        # again defensively but this keeps the stage-1 concat consistent.
        run_vox = run_vox - run_vox.mean(dim=1, keepdim=True)

        run_data_list.append(run_vox.cpu())
        run_lengths.append(n_t_run)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mask3d, _n_drop = _prune_constant_runs(mask3d, const_bad_vox, run_data_list, args.verb >= 1)
    if _n_drop:
        n_vox_masked = int(run_data_list[0].shape[0])

    assert mean3d_accum is not None and tr is not None
    return {
        "run_data_list": run_data_list,
        "run_lengths": run_lengths,
        "mask3d": mask3d,
        "shape3d": shape3d,
        "affine": affine,
        "mean3d": mean3d_accum / n_runs,
        "tr": float(tr),
        "resels": float(resels),
        "fwhm_geo": float(fwhm_geo),
        "n_vox_masked": int(n_vox_masked),
        "polort": polort,
    }


def _run_temporal_ica(
    input_files: list[str],
    args,
    device: torch.device,
    shared_mask: np.ndarray | None,
) -> dict:
    """Two-stage temporal ICA (Glasser 2018): spatial reduction → dual regression
    → temporal ICA. Outputs mirror the single-run/concat layout so results are
    viewable with the existing maps+timecourses machinery: a group `.ica` folder
    plus per-run `.ica` folders (shared tICA maps + that run's timecourses)."""
    import shutil

    from fastfuncstuff.decomposition.migp import _reduce_to_topk, migp_reduce
    from fastfuncstuff.decomposition.temporal import (
        group_spatial_ica,
        spatial_regression,
        temporal_ica,
    )

    t_total = time.time()
    n_runs = len(input_files)
    _vsection(args.verb >= 1, "Temporal ICA (two-stage)")

    # --- Stage 0: per-run preprocessing ------------------------------------
    pre = _temporal_ica_preprocess_runs(input_files, args, device, shared_mask)
    run_data_list: list[torch.Tensor] = pre["run_data_list"]  # (V, T_i) on CPU
    run_lengths: list[int] = pre["run_lengths"]
    mask3d = pre["mask3d"]
    shape3d = pre["shape3d"]
    affine = pre["affine"]
    tr = pre["tr"]
    resels = pre["resels"]
    n_vox_masked = pre["n_vox_masked"]
    total_t = int(sum(run_lengths))

    # --- Stage 1: group spatial reduction basis ----------------------------
    _vsection(args.verb >= 1, f"Stage 1: spatial reduction ({args.tica_reducer})")
    t_step = time.time()
    scale = 1.0 / float(n_runs)
    runs_tv = [run_data_list[ri].to(device).T for ri in range(n_runs)]  # (T_i, V)
    if args.migp:
        concat_tv = migp_reduce(
            runs_tv,
            migp_n=args.migp_n,
            migp_factor=args.migp_factor,
            scale_by_n=True,
            device=device,
            verbose=args.verb >= 1,
        ).contiguous()
    else:
        concat_tv = torch.cat([t * scale for t in runs_tv], dim=0).contiguous()
    del runs_tv
    concat_tv = _check_finite(concat_tv, "concat_tv", args.verb >= 1)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Number of spatial components K_sica (reuse -num_comps).
    num_spec = parse_num_comps_spec(args.num_comps)
    stack_t = int(concat_tv.shape[0])
    if isinstance(num_spec, int):
        k_sica = min(num_spec, stack_t, int(concat_tv.shape[1]))
    else:
        if args.max_auto_components <= 1.0:
            max_auto_k = max(5, int((stack_t - 2) * args.max_auto_components))
        else:
            max_auto_k = int(args.max_auto_components)
        n_eff = effective_sample_size_from_resels(n_vox_masked, resels, floor=stack_t)
        k_sica, _, _ = estimate_ica_component_count(
            data_vox_t=concat_tv.T,
            method=num_spec,
            max_auto_components=max_auto_k,
            auto_min_components=args.auto_min_components,
            auto_var_threshold=args.auto_var_threshold,
            use_mp_prior=not args.auto_no_mp,
            n_eff=n_eff,
            device=device,
            verbose=args.verb >= 1,
        )
        k_sica = int(k_sica)
    # tICA needs T_total large relative to K_sica or the temporal decomposition
    # is unstable (few reproducible components). Warn — but don't override the
    # user; the temporal-stage ICASSO Iq is the empirical referee.
    tica_ratio = total_t / max(k_sica, 1)
    _vprint(args.verb >= 1, f"K_sica = {k_sica}  (T_total/K_sica = {tica_ratio:.0f})")
    if tica_ratio < 50:
        print(
            f"⚠ K_sica={k_sica} is large for T_total={total_t} "
            f"(ratio {tica_ratio:.0f} < 50). Temporal ICA may be unstable — "
            f"consider a smaller -num_comps. Check the ICASSO Iq values."
        )

    if args.tica_reducer == "pca":
        group_maps = _reduce_to_topk(concat_tv, k_sica)  # (K_sica, V)
        sica_iters = 0
    else:
        method_sica = args.tica_method or getattr(args, "ica_method", "fastica")
        group_maps, sica_iters = group_spatial_ica(
            concat_tv,
            n_components=k_sica,
            method=method_sica,
            max_iter=args.ica_max_iter,
            tol=args.ica_tol,
            fun=args.ica_nonlinearity,
            seed=args.seed,
            device=device,
        )
    del concat_tv
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _vprint(args.verb >= 1, f"Group maps: {tuple(group_maps.shape)} (K_sica, V)", t_step)

    # --- Stage 2: dual-regression back-projection per run ------------------
    _vsection(args.verb >= 1, "Stage 2: dual regression (spatial back-projection)")
    t_step = time.time()
    tc_blocks: list[torch.Tensor] = []
    for ri in tqdm(range(n_runs), desc="Dual regression", disable=n_runs < 3, leave=True):
        run_dev = run_data_list[ri].to(device)  # (V, T_i)
        tc = spatial_regression(group_maps, run_dev)  # (T_i, K_sica)
        tc_blocks.append(tc.cpu())
        del run_dev
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del run_data_list
    concat_tcs = torch.cat(tc_blocks, dim=0)  # (T_total, K_sica) on CPU
    del tc_blocks
    _vprint(args.verb >= 1, f"Concatenated timecourses: {tuple(concat_tcs.shape)}", t_step)

    # Number of temporal components K_tica.
    # 'auto' uses ICASSO reproducibility (HCP's method): decompose at the full
    # K_sica and keep the components whose Iq clears -tica_iq_thresh. An int/float
    # requests exactly that many; ICASSO (if enabled) still reports each Iq.
    icasso_runs = int(args.tica_icasso_runs)
    iq_thresh = float(args.tica_iq_thresh)
    tica_spec = parse_num_comps_spec(args.n_temporal_comps)
    auto_tica = isinstance(tica_spec, str)
    if isinstance(tica_spec, int):
        k_tica_req = min(tica_spec, k_sica)
    elif isinstance(tica_spec, float):
        k_tica_req = max(1, min(int(round(tica_spec * k_sica)), k_sica))
    else:
        k_tica_req = k_sica  # decompose at full rank, then keep reproducible
        if icasso_runs <= 1:
            icasso_runs = 25
            _vprint(
                args.verb >= 1,
                "auto K_tica needs ICASSO to judge reproducibility → using 25 runs",
            )

    # --- Stage 3: temporal ICA ---------------------------------------------
    _vsection(args.verb >= 1, "Stage 3: temporal ICA")
    t_step = time.time()
    method_tica = args.tica_method or getattr(args, "ica_method", "fastica")
    _vprint(
        args.verb >= 1,
        f"{method_tica}: K_tica={'auto' if auto_tica else k_tica_req} "
        f"over T_total={total_t}, icasso_runs={icasso_runs}",
    )
    # Temporal stage uses logcosh (general contrast): temporal sources are often
    # symmetric, which the spatial default pow3 (skewness) cannot separate.
    result = temporal_ica(
        concat_tcs.to(device),
        group_maps,
        n_components=k_tica_req,
        run_lengths=run_lengths,
        method=method_tica,
        max_iter=args.ica_max_iter,
        tol=args.ica_tol,
        fun="logcosh",
        seed=args.seed,
        variance_normalize=args.tica_varnorm,
        icasso_runs=icasso_runs,
        device=device,
        verbose=args.verb >= 1,
    )
    result.reducer = args.tica_reducer
    del concat_tcs, group_maps
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Auto K_tica: keep only reproducible components (Iq > threshold).
    if auto_tica and result.stability is not None:
        keep = result.stability > iq_thresh
        if int(keep.sum()) < 1:
            keep = np.zeros_like(result.stability, dtype=bool)
            keep[int(np.argmax(result.stability))] = True
        _vprint(
            args.verb >= 1,
            f"auto K_tica: {int(keep.sum())}/{k_tica_req} components with Iq>{iq_thresh}",
        )
        result = result.subset(keep)
    k_tica = int(result.diagnostics["k_tica"])
    if result.stability is not None:
        _vprint(
            args.verb >= 1,
            f"ICASSO Iq range: {result.stability.min():.2f}–{result.stability.max():.2f}",
        )
    _vprint(args.verb >= 1, f"Temporal ICA done ({k_tica} components)", t_step)

    # --- Save outputs (group folder + per-run folders) ---------------------
    _vsection(args.verb >= 1, "Save Outputs")
    t_step = time.time()
    pfx = parse_prefix(str(args.prefix))
    nii_ext = pfx.nifti_ext
    _basename = Path(pfx.stem).name
    _parent_dir = Path(pfx.stem).parent

    group_dir = _parent_dir / f"{_basename}_temporalica.ica"
    _gffs = group_dir / "ffs_outputs"
    _gffs.mkdir(parents=True, exist_ok=True)
    out_prefix = _gffs / _basename
    labels = [f"tIC_{i + 1}" for i in range(k_tica)]

    # Group tICA spatial maps (shared across runs) + concatenated timecourses.
    maps_file = Path(f"{out_prefix}_temporalica_maps{nii_ext}")
    with spinner(f"Writing {maps_file.name}"):
        decomposition_io.save_masked_component_maps_4d(
            components_kv=result.spatial_maps,
            mask3d=mask3d,
            shape3d=shape3d,
            affine=affine,
            out_file=maps_file,
        )
    decomposition_io.save_timeseries(
        result.temporal_sources.T,  # (T_total, K_tica)
        f"{out_prefix}_temporalica_timecourses.1D",
        tr=tr,
        labels=labels,
    )
    # Stage-1 spatial basis, saved for inspection.
    _sica_maps_file = Path(f"{out_prefix}_temporalica_sica_maps{nii_ext}")
    with spinner(f"Writing {_sica_maps_file.name}"):
        decomposition_io.save_masked_component_maps_4d(
            components_kv=result.group_spatial_maps,
            mask3d=mask3d,
            shape3d=shape3d,
            affine=affine,
            out_file=_sica_maps_file,
        )

    # Optional GGM z-maps so thresholded viewing matches the single-run tool.
    z_maps = None
    if args.save_mixture_z:
        comp_tensor = torch.as_tensor(result.spatial_maps, device=device)
        z_tensor, p_tensor, _ = batch_mixture_zscores(
            comp_tensor, device=device, verbose=args.verb >= 1
        )
        z_maps = z_tensor.cpu().numpy().astype(np.float32)
        p_maps = p_tensor.cpu().numpy().astype(np.float32)
        del comp_tensor, z_tensor, p_tensor
        for arr, fname in [
            (z_maps, f"{out_prefix}_temporalica_zmaps{nii_ext}"),
            (p_maps, f"{out_prefix}_temporalica_signalprob{nii_ext}"),
        ]:
            decomposition_io.save_masked_component_maps_4d(
                components_kv=arr,
                mask3d=mask3d,
                shape3d=shape3d,
                affine=affine,
                out_file=Path(fname),
            )

    # Per-run folders: symlink the (shared) group maps in + that run's timecourses.
    per_run_dirs: list[str] = []
    for ri in range(n_runs):
        rdir = _parent_dir / f"{_basename}_temporalica_run{ri + 1:02d}.ica"
        rffs = rdir / "ffs_outputs"
        rffs.mkdir(parents=True, exist_ok=True)
        rprefix = rffs / _basename
        run_maps = Path(f"{rprefix}_temporalica_maps{nii_ext}")
        try:
            if run_maps.exists() or run_maps.is_symlink():
                run_maps.unlink()
            run_maps.symlink_to(maps_file.resolve())
        except OSError:
            shutil.copy2(maps_file, run_maps)
        decomposition_io.save_timeseries(
            result.per_run_sources[ri].T,  # (T_i, K_tica)
            f"{rprefix}_temporalica_timecourses.1D",
            tr=tr,
            labels=labels,
        )
        per_run_dirs.append(str(rdir))
    _vprint(
        args.verb >= 1,
        f"Saved group + {n_runs} per-run folders under {_parent_dir}",
        t_step,
    )

    mask_type = (
        "provided" if shared_mask is not None else ("auto" if mask3d is not None else "none")
    )
    elapsed_total = time.time() - t_total
    meta = {
        "mode": "temporal_ica",
        "reducer": args.tica_reducer,
        "n_runs": n_runs,
        "input_files": input_files,
        "tr": float(tr),
        "run_lengths": run_lengths,
        "total_timepoints": total_t,
        "n_voxels": n_vox_masked,
        "mask_type": mask_type,
        "n_spatial_components": int(k_sica),
        "tica_timepoints_per_component": round(tica_ratio, 1),
        "n_temporal_components": int(k_tica),
        "n_temporal_comps_request": args.n_temporal_comps,
        "variance_normalized": bool(args.tica_varnorm),
        "tica_method": method_tica,
        "icasso_runs": int(icasso_runs),
        "iq_threshold": float(iq_thresh),
        "component_stability_iq": (
            None if result.stability is None else [round(float(v), 3) for v in result.stability]
        ),
        "sica_iterations": int(sica_iters),
        "tica_iterations": int(result.n_iter),
        "component_variance_share": result.explained_share.tolist(),
        "group_dir": str(group_dir),
        "per_run_dirs": per_run_dirs,
        "outputs": {
            "ica_maps": str(maps_file),
            "ica_timecourses": f"{out_prefix}_temporalica_timecourses.1D",
            "sica_maps": f"{out_prefix}_temporalica_sica_maps{nii_ext}",
            "ica_zmaps": (
                f"{out_prefix}_temporalica_zmaps{nii_ext}" if z_maps is not None else None
            ),
        },
        "elapsed_seconds": round(elapsed_total, 2),
    }
    with open(f"{out_prefix}_temporalica_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def _run_tensorial_ica(
    input_files: list[str],
    args,
    device: torch.device,
    shared_mask: np.ndarray | None,
) -> dict:
    """Run ICA on spatially concatenated runs (same T, stacked along V).

    Complement of `_run_concat_ica`: assumes all runs share a common temporal
    grid (same T, same TR) and stacks them voxelwise → (T, n_runs * V). A
    single set of temporal mixing is shared across runs; spatial maps are
    per-run slices of the (k, n_runs * V) component matrix.

    Unlike MELODIC's proper tensor-ICA (Kronecker factorisation of the full
    mixing), this is the simpler "spatial concat" form the user requested.
    Dim-estimation, whitening, and ICA follow the same MELODIC-style
    machinery as the temporal-concat path.
    """
    t_total = time.time()
    n_runs = len(input_files)

    _vsection(args.verb >= 1, "Tensorial (spatial-concat) ICA")
    _vprint(
        args.verb >= 1, f"Spatially concatenating {n_runs} runs → single shared temporal mixing"
    )

    # --- Per-run preprocessing (identical to concat path) -------------------
    run_data_list: list[torch.Tensor] = []  # each (V, T)
    run_lengths: list[int] = []
    mask3d: np.ndarray | None = None
    const_bad_vox: np.ndarray | None = None  # union of per-run constant voxels
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
        _vsection(args.verb >= 1, f"Load Run {ri + 1}/{n_runs}")
        _vprint(args.verb >= 1, f"Loading {run_file} ...")
        img = load_nifti(run_file)
        data = img.get_fdata(dtype=np.float32)
        run_shape3d = data.shape[:3]
        n_t_run = data.shape[3]
        voxel_sizes = tuple(float(v) for v in img.header.get_zooms()[:3])
        _vprint(args.verb >= 1, f"  shape={data.shape}", t_step)

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

        if args.do_blur is not None and args.do_blur > 0:
            t_step = time.time()
            data = gaussian_blur_3d(
                data=data,
                fwhm_mm=float(args.do_blur),
                voxel_sizes=voxel_sizes,
                device=device,
                verbose=False,
            )
            _vprint(args.verb >= 1, f"  Blurred FWHM={args.do_blur:.1f}mm", t_step)

        run_mean3d = data.mean(axis=-1).astype(np.float32)
        if mean3d_accum is None:
            mean3d_accum = run_mean3d.copy()
        else:
            mean3d_accum += run_mean3d

        if ri == 0:
            if shared_mask is not None:
                mask3d = shared_mask
                _vprint(args.verb >= 1, f"  Using provided mask: {int(mask3d.sum()):,} voxels")
            elif not args.no_auto_mask:
                from fastfuncstuff.processing.mask import automask

                mask3d = (
                    automask(
                        torch.as_tensor(run_mean3d),
                        dilate_extra=2,
                        device=device,
                        verbose=args.verb >= 1,
                    )
                    .cpu()
                    .numpy()
                )
                _vprint(args.verb >= 1, f"  Auto-mask from run 1: {int(mask3d.sum()):,} voxels")

            if mask3d is not None and mask3d.shape != shape3d:
                raise ValueError(f"Mask shape {mask3d.shape} != data shape {shape3d}")

            resels_accum, fwhm_geo_accum, _ = _estimate_smoothness_resels_acf(
                data,
                voxel_sizes,
                mask=mask3d,
                device=device,
                per_axis=bool(getattr(args, "smoothness_per_axis", False)),
                corder=int(getattr(args, "smoothness_corder", -1)),
                verbose=args.verb >= 1,
            )

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

        # A voxel constant in ANY run is unusable across the whole set, so
        # accumulate here and prune the mask once every run has been read.
        if mask3d is not None and args.drop_constant:
            _run_bad = find_constant_voxels(run_vox_np)
            const_bad_vox = _run_bad if const_bad_vox is None else (const_bad_vox | _run_bad)

        run_vox = to_tensor(run_vox_np, device=device, pin=True)
        del run_vox_np

        if args.do_scale:
            run_vox, _, _ = scale_to_percent_signal(run_vox, run_starts=[0], verbose=False)
            run_vox = _check_finite(run_vox, f"run{ri + 1}-post-scale", args.verb >= 1)

        # Tensorial: same T across runs is assumed; polort defaults to 0 (demean)
        # since there's no block-diagonal trend to separate (each run sits in
        # its own column-block).
        polort_default = 0
        polort_r = polort_default if args.polort is None else int(args.polort)
        if polort_r >= 0:
            run_vox = apply_polort_projection(
                run_vox, polort=polort_r, device=device, run_starts=[0]
            )
            run_vox = _check_finite(run_vox, f"run{ri + 1}-post-polort", args.verb >= 1)

        if args.high_pass is not None and args.high_pass > 0:
            nyquist = 0.5 / float(tr if tr is not None else run_tr)
            if args.high_pass >= nyquist:
                raise ValueError(
                    f"High-pass cutoff ({args.high_pass:.4f} Hz) >= Nyquist ({nyquist:.4f} Hz)"
                )
            run_vox = apply_high_pass_fft(
                run_vox,
                tr=float(tr if tr is not None else run_tr),
                high_pass_hz=float(args.high_pass),
                run_starts=[0],
            )
            run_vox = _check_finite(run_vox, f"run{ri + 1}-post-hp", args.verb >= 1)

        # Per-voxel temporal mean-center
        run_vox = run_vox - run_vox.mean(dim=1, keepdim=True)

        run_data_list.append(run_vox)
        run_lengths.append(n_t_run)
        _vprint(args.verb >= 1, f"  Run {ri + 1}: {n_vox_masked:,} vox x {n_t_run} TPs")

    mask3d, _n_drop = _prune_constant_runs(mask3d, const_bad_vox, run_data_list, args.verb >= 1)
    if _n_drop:
        n_vox_masked = int(run_data_list[0].shape[0])

    mean3d = mean3d_accum / n_runs

    # --- Enforce common T across runs (tensorial requires it) --------------
    T = run_lengths[0]
    if any(rl != T for rl in run_lengths):
        raise ValueError(
            "Tensorial ICA requires identical T across runs; "
            f"got {run_lengths}. Use -temp_concat for mismatched run lengths."
        )

    # --- Spatial concatenation: stack (V, T) → (T, n_runs * V) --------------
    _vsection(args.verb >= 1, "Spatial Concatenation")
    t_step = time.time()
    # Transpose each (V, T) to (T, V) and cat along dim=1 → (T, n_runs * V).
    data_tv = torch.cat([run_data_list[ri].T for ri in range(n_runs)], dim=1).contiguous()
    del run_data_list
    data_tv = _check_finite(data_tv, "data_tv", args.verb >= 1)
    V_total = int(data_tv.shape[1])
    _vprint(
        args.verb >= 1,
        f"Spatial concat: {tuple(data_tv.shape)} (T, n_runs*V) = ({T}, {n_runs} * {n_vox_masked})",
        t_step,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- Variance normalization (joined on the full stack) ------------------
    vn_scope = "none"
    if args.voxel_norm:
        from fastfuncstuff.decomposition.varnorm import apply_noise_std_map, noise_std_map

        _vsection(args.verb >= 1, "Variance Normalization (joined on spatial concat)")
        t_step = time.time()
        noise_std_v, const_mask_vn, n_const_vn = noise_std_map(data_tv.T)
        data_tv_vt = apply_noise_std_map(data_tv.T, noise_std_v, const_mask_vn)
        data_tv = data_tv_vt.T.contiguous()
        del data_tv_vt, noise_std_v, const_mask_vn
        _vprint(
            args.verb >= 1,
            f"Joined varnorm: {n_const_vn} constant columns",
            t_step,
        )
        vn_scope = "joined"

    concat_var = float(torch.var(data_tv).item())
    if concat_var < 1e-10:
        raise ValueError(f"spatial-concat variance is ~0 after preprocessing ({concat_var:.2e})")
    _vprint(args.verb >= 1, f"Spatial-concat variance: {concat_var:.4f}")

    # --- Spatial smoothness / effective DOF ---------------------------------
    # For spatial concat, each run contributes V voxels with the same resel
    # structure, so n_eff scales by n_runs.
    resels = resels_accum
    fwhm_geo = fwhm_geo_accum
    n_eff = effective_sample_size_from_resels(V_total, resels, floor=int(T))
    _vprint(
        args.verb >= 1,
        f"Effective spatial DOF: n_eff={n_eff:,} (V_total={V_total:,}, resels={resels:.2f})",
    )

    # --- Component estimation ----------------------------------------------
    _vsection(args.verb >= 1, "Component Estimation (on spatial concat)")
    t_step = time.time()
    if args.max_auto_components <= 1.0:
        max_auto_k = max(5, int((T - 2) * args.max_auto_components))
        if args.max_auto_components >= 0.66:
            max_auto_k = max(max_auto_k, max(5, T - 2))
    else:
        max_auto_k = int(args.max_auto_components)
    _vprint(args.verb >= 1, f"max_auto_components: {max_auto_k}")
    _vprint(args.verb >= 1, f"Method: {args.num_comps}")

    n_eff_for_model_order = n_eff
    data_for_model_order_vt = data_tv.T  # (V_total, T)
    if isinstance(num_spec, str) and num_spec in {"auto", "laplace"}:
        data_for_model_order_vt, _ = ica_workflow.filter_low_variance_voxels(data_vox_t=data_tv.T)
        n_vox_model_order = int(data_for_model_order_vt.shape[0])
        n_eff_for_model_order = effective_sample_size_from_resels(
            n_vox_model_order, resels, floor=int(T)
        )
        _vprint(
            args.verb >= 1,
            f"MELODIC dim-est filter: kept {n_vox_model_order:,}/{V_total:,} columns; "
            f"n_eff={n_eff_for_model_order:,}",
        )

    use_mp = not args.auto_no_mp
    n_components, pca_diag, num_diag = estimate_ica_component_count(
        data_vox_t=data_for_model_order_vt,
        method=num_spec,
        max_auto_components=max_auto_k,
        auto_min_components=args.auto_min_components,
        auto_var_threshold=args.auto_var_threshold,
        use_mp_prior=use_mp,
        n_eff=n_eff_for_model_order,
        device=device,
        verbose=args.verb >= 1,
        capture_ppca_trace=bool(args.ppca_debug_dump),
    )
    order = int(n_components)
    _vprint(
        args.verb >= 1,
        f"Selected {order} components (mode={num_diag.get('mode', '?')})",
        t_step,
    )
    del data_for_model_order_vt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- Whitening on spatial concat (same structure as tcat) --------------
    _vsection(args.verb >= 1, "Whitening (joined on spatial concat)")
    t_step = time.time()
    row_mean = data_tv.mean(dim=1, keepdim=True)
    data_centered = data_tv - row_mean
    cov_t = (data_centered @ data_centered.T) / float(V_total)
    evals_t, evecs_t = torch.linalg.eigh(cov_t)
    del cov_t
    idx = torch.argsort(evals_t, descending=True)
    evals_t = torch.clamp(evals_t[idx][:order], min=1e-12)
    evecs_t = evecs_t[:, idx][:, :order]
    sqrt_ev = torch.sqrt(evals_t)
    WM = (evecs_t / sqrt_ev.unsqueeze(0)).T  # (order, T)
    DWM = evecs_t * sqrt_ev.unsqueeze(0)  # (T, order)
    x_t = WM @ data_centered  # (order, V_total)
    del data_centered, WM, evals_t, evecs_t, sqrt_ev
    orig_concat_tv = data_tv
    del data_tv
    _vprint(
        args.verb >= 1,
        f"Whiten: order={order}, x_t shape {tuple(x_t.shape)}",
        t_step,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- ICA ----------------------------------------------------------------
    _vsection(args.verb >= 1, "ICA Decomposition")
    t_step = time.time()
    if args.icasso:
        method_name = getattr(args, "ica_method", "fastica")
        _vprint(
            args.verb >= 1, f"ICASSO ({method_name}): {args.icasso_runs} runs, k={n_components}"
        )
        icasso_res = icasso(
            X=x_t,
            n_components=n_components,
            n_runs=args.icasso_runs,
            pca_components=n_components,
            min_stability=args.icasso_min_stability,
            device=device,
            verbose=args.verb >= 1,
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
        _vprint(args.verb >= 1, f"ICASSO done ({icasso_res['n_stable']} stable)", t_step)
    else:
        method_name = getattr(args, "ica_method", "fastica")
        _vprint(args.verb >= 1, f"{method_name}: k={n_components}")
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
        assert ica.components_ is not None and ica.mixing_ is not None  # set by fit above
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

    # --- Dewhiten mixing to data space -------------------------------------
    mixing_pca = mixing  # (order, k)
    mixing = DWM @ mixing_pca  # (T, k) — shared across runs
    del mixing_pca, DWM
    x_t = orig_concat_tv  # un-whitened for noise-norm
    orig_concat_tv = None
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- Sign consistency (FSL convention) ---------------------------------
    max_abs = torch.max(torch.abs(components), dim=1).values
    max_pos = torch.max(components, dim=1).values
    flip_mask = max_abs > max_pos
    n_flipped = int(flip_mask.sum().item())
    if n_flipped > 0:
        components = components.clone()
        mixing = mixing.clone()
        components[flip_mask] *= -1.0
        mixing[:, flip_mask] *= -1.0
        _vprint(args.verb >= 1, f"Sign-flipped {n_flipped} ICs")

    # --- Ordering by explained share (of shared mixing variance) ----------
    ic_stdev = torch.std(components, dim=1)
    total_stdev = float(ic_stdev.sum().item())
    stdev_share = ic_stdev / max(total_stdev, 1e-15)

    mix_var = torch.var(mixing, dim=0, unbiased=False)
    total_mix_var = float(mix_var.sum().item())
    explained_share_t = mix_var / max(total_mix_var, 1e-15)

    sort_key = stdev_share if args.ordering == "stdev" else explained_share_t
    sort_idx = torch.argsort(sort_key, descending=True)
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

    # --- MELODIC-style noise normalization (on the wide component matrix) -
    # Must run BEFORE timecourse var_norm — see note in single-run path.
    if x_t is not None:
        _vprint(args.verb >= 1, "Applying MELODIC-style noise normalization ...")
        components, noise_norm_msg = ica_workflow.apply_melodic_noise_normalization(
            components=components,
            mixing=mixing,
            x_t=x_t,
        )
        _vprint(args.verb >= 1, noise_norm_msg)
    del x_t

    if args.var_norm:
        mixing = mixing - mixing.mean(dim=0, keepdim=True)
        mixing_std = torch.clamp(mixing.std(dim=0, keepdim=True), min=1e-8)
        mixing = mixing / mixing_std
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- Split components into per-run spatial maps ------------------------
    _vsection(args.verb >= 1, "Save Outputs")
    t_step = time.time()
    pfx = parse_prefix(str(args.prefix))
    nii_ext = pfx.nifti_ext

    # Output layout: see _run_single_ica for rationale.
    _basename = Path(pfx.stem).name
    _parent_dir = Path(pfx.stem).parent
    compat_dir = _parent_dir / f"{_basename}_tensor.ica"
    _ffs_dir = compat_dir / "ffs_outputs"
    _ffs_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = _ffs_dir / _basename

    comp_np = components.detach().cpu().numpy().astype(np.float32)  # (k, n_runs*V)
    mixing_np = mixing.detach().cpu().numpy().astype(np.float32)  # (T, k) — shared
    del components, mixing
    if device.type == "cuda":
        torch.cuda.empty_cache()

    per_run_map_paths: list[str] = []
    for ri in range(n_runs):
        v0 = ri * n_vox_masked
        v1 = v0 + n_vox_masked
        run_maps = comp_np[:, v0:v1]
        map_path = f"{out_prefix}_tensor_ica_maps_run{ri + 1:02d}{nii_ext}"
        decomposition_io.save_masked_component_maps_4d(
            components_kv=run_maps,
            mask3d=mask3d,
            shape3d=shape3d,
            affine=affine,
            out_file=Path(map_path),
        )
        per_run_map_paths.append(map_path)
    _vprint(args.verb >= 1, f"Per-run maps: {len(per_run_map_paths)} files", t_step)

    # Also save the averaged-across-runs map for convenience
    avg_maps = np.stack(
        [comp_np[:, ri * n_vox_masked : (ri + 1) * n_vox_masked] for ri in range(n_runs)],
        axis=0,
    ).mean(axis=0)
    _avg_maps_out_file = Path(f"{out_prefix}_tensor_ica_maps_mean{nii_ext}")
    with spinner(f"Writing {_avg_maps_out_file.name}"):
        decomposition_io.save_masked_component_maps_4d(
            components_kv=avg_maps,
            mask3d=mask3d,
            shape3d=shape3d,
            affine=affine,
            out_file=_avg_maps_out_file,
        )

    # Temporal mean image (cross-run mean underlay), the same full-volume mean
    # the temp_concat path emits via its MELODIC-compat output. Distinct from
    # ica_maps_mean above, which is the run-averaged component maps.
    from fastfuncstuff.io.afni import save_nifti

    mean_image_path = f"{out_prefix}_tensor_mean{nii_ext}"
    with spinner(f"Writing {Path(mean_image_path).name}"):
        save_nifti(mean3d.astype(np.float32), output_path=Path(mean_image_path), affine=affine)
    _vprint(args.verb >= 1, f"Mean image: {mean_image_path}")

    # Shared timecourses
    np.savetxt(f"{out_prefix}_tensor_ica_timecourses.1D", mixing_np, fmt="%.6f", delimiter="\t")
    _vprint(args.verb >= 1, f"Shared timecourses: {out_prefix}_tensor_ica_timecourses.1D")

    # Scree plot
    ica_postprocess.save_scree_plot(
        evr=np.asarray(pca_diag["scree_ratio"], dtype=np.float64),
        out_png=Path(f"{out_prefix}_tensor_pca_scree.png"),
        title="Spatial concat: PCA scree",
    )

    # Mixture model on per-run-averaged maps (optional)
    z_maps = None
    mixture_meta = []
    if args.save_mixture_z:
        _vsection(args.verb >= 1, "Mixture Model (GGM) on run-averaged maps")
        t_step = time.time()
        comp_tensor = torch.as_tensor(avg_maps, device=device)
        z_tensor, p_tensor, mixture_meta = batch_mixture_zscores(
            comp_tensor, device=device, verbose=args.verb >= 1
        )
        z_maps = z_tensor.cpu().numpy().astype(np.float32)
        p_maps = p_tensor.cpu().numpy().astype(np.float32)
        thresh_z_maps = z_maps.copy()
        thresh_z_maps[p_maps < float(args.mm_thresh)] = 0.0
        del comp_tensor, z_tensor, p_tensor

        for data_arr, fname in [
            (z_maps, f"{out_prefix}_tensor_ica_zmaps{nii_ext}"),
            (p_maps, f"{out_prefix}_tensor_ica_signalprob{nii_ext}"),
            (thresh_z_maps, f"{out_prefix}_tensor_ica_thresh_zmaps{nii_ext}"),
        ]:
            decomposition_io.save_masked_component_maps_4d(
                components_kv=data_arr,
                mask3d=mask3d,
                shape3d=shape3d,
                affine=affine,
                out_file=Path(fname),
            )
        _vprint(args.verb >= 1, "Z-maps and signal-prob maps saved", t_step)

    elapsed_total = time.time() - t_total
    _vprint(args.verb >= 1, f"Tensorial ICA total elapsed: {elapsed_total:.1f}s")

    mask_type = (
        "provided" if shared_mask is not None else ("auto" if mask3d is not None else "none")
    )
    tensor_meta = {
        "mode": "tensorial_spatial_concat",
        "n_runs": n_runs,
        "input_files": input_files,
        "tr": float(tr),
        "T": int(T),
        "n_voxels_per_run": int(n_vox_masked),
        "n_voxels_total": int(V_total),
        "mask_type": mask_type,
        "polort": 0 if args.polort is None else int(args.polort),
        "high_pass_hz": None if args.high_pass is None else float(args.high_pass),
        "voxel_norm": bool(args.voxel_norm),
        "voxel_norm_scope": vn_scope,
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
        "component_ordering": args.ordering,
        "elapsed_seconds": round(elapsed_total, 2),
        "outputs": {
            "ica_maps_per_run": per_run_map_paths,
            "ica_maps_mean": f"{out_prefix}_tensor_ica_maps_mean{nii_ext}",
            "mean_image": mean_image_path,
            "ica_timecourses_shared": f"{out_prefix}_tensor_ica_timecourses.1D",
            "pca_scree_plot": f"{out_prefix}_tensor_pca_scree.png",
            "ica_zmaps": f"{out_prefix}_tensor_ica_zmaps{nii_ext}" if args.save_mixture_z else None,
        },
        "mixture_model": mixture_meta if args.save_mixture_z else None,
    }

    with open(f"{out_prefix}_tensor_ica_metadata.json", "w") as f:
        json.dump(tensor_meta, f, indent=2)

    return tensor_meta


def build_parser() -> argparse.ArgumentParser:
    """Build and return the ffs_ica command-line argument parser."""
    parser = FfsArgumentParser(
        description="Run-wise whole-brain ICA demo / sanity-check pipeline",
        formatter_class=FfsHelpFormatter,
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
            "Component selection: INT, FLOAT(0-1), or auto/laplace/hybrid/current/erank/mp. "
            "'auto' and 'laplace' take the Marchenko-Pastur count as a ceiling and pick "
            "within it by Minka's PPCA Laplace evidence.  ('melodic' is accepted as an "
            "old name for 'laplace'.)"
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
    basic.add_argument(
        "-ordering",
        choices=["var", "stdev"],
        default="var",
        help=(
            "Component sort order: 'var' (default — by mixing variance share) "
            "or 'stdev' (share of summed spatial-map stdev). "
            "Hungarian matching is order-invariant; use 'stdev' for "
            "MELODIC parity diagnostics where component indices must align."
        ),
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
        "via the resel-based effective-sample-size N_eff = n_vox / (FWHM_x·FWHM_y·FWHM_z).",
    )
    proc.add_argument(
        "-smoothness_per_axis",
        action="store_true",
        help="Estimate the ACF smoothness separately along each axis instead of using the "
        "radial (isotropic) fit that 3dFWHMx reports.  Worth it on anisotropic voxels "
        "(thick slices), where one FWHM for three axes is wrong for all of them; on "
        "near-isotropic data the two agree closely and the radial fit, having all "
        "directions to constrain it, is the better-determined number.",
    )
    proc.add_argument(
        "-smoothness_corder",
        type=int,
        default=-1,
        help="Detrend order for the smoothness estimate (3dFWHMx -detrend).  -1 uses AFNI's "
        "own default of n_time/30; 0 disables detrending.  Smoothness is measured before "
        "the pipeline's own detrend, and low-frequency drift is spatially structured, so "
        "leaving it in biases the ACF low (3.22 vs 3.57 mm measured on ds005165).",
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
    proc.add_argument(
        "-varnorm_rank",
        type=int,
        default=None,
        help="Rank of the signal subspace removed when estimating each voxel's NOISE "
        "standard deviation for variance normalisation (default: 30).\n"
        "A real quality/count trade-off, not a nuisance -- but HOW MUCH IT MATTERS DEPENDS "
        "ON YOUR SMOOTHING.  On UNSMOOTHED 3 mm data (ds005165 rest, 5 runs, k fixed at "
        "62) raising it markedly improves cross-run component reproducibility: components "
        "reproducing at |r|>=0.25 go 11.1 / 18.0 / 21.2 / 21.6 at rank 2 / 30 / 80 / 120, "
        "against an unmatched-pair null that barely moves.  With -do_blur 5 the same sweep "
        "gives 23.9 / 25.0 / 26.5 / 26.9 -- a 13%% span instead of 95%%, because blurring "
        "raises per-voxel SNR and the noise estimate stops being the limiting factor.\n"
        "It also raises the automatic model order, again more without blur (62->92 across "
        "the sweep) than with (30->51).  Leave it alone on smoothed data.",
    )
    proc.add_argument(
        "-drop_constant",
        "-drop-constant",
        dest="drop_constant",
        action="store_true",
        default=True,
        help="Remove constant (zero-variance) voxels from the analysis mask, as "
        "MELODIC does.  With multiple runs a voxel constant in ANY run is "
        "dropped from all of them.  Leaving them in puts an exact-zero spike in "
        "every IC map, which collapses the mixture model's noise Gaussian and "
        "grossly inflates P(signal).  The updated mask is what gets written "
        "out.  (default: on)",
    )
    proc.add_argument(
        "-no_drop_constant",
        "-no-drop-constant",
        dest="drop_constant",
        action="store_false",
        help="Keep constant voxels in the mask (not recommended; breaks "
        "mixture-model parity with MELODIC)",
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
        "-seed",
        type=int,
        default=0,
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
    ica_opts.add_argument(
        "-stability_order",
        action="store_true",
        help="Choose the component count by restart-cluster stability rather than the "
        "eigenspectrum: run -stability_runs restarts at the Marchenko-Pastur ceiling and "
        "keep the clusters reaching -icasso_min_stability.\n"
        "A reasonable second opinion on the overall level, and cheap (~20s/run at 30 "
        "restarts on unsmoothed data, less on smoothed).  On UNSMOOTHED ds005165 rest it "
        "returns 58-67 where the spectral default returns 76-84, and cross-run matching "
        "shows the spectral extras reproduce no better -- so the lower count is not losing "
        "anything.  With -do_blur 5 both fall together (35 of a spectral 40) and the Iq "
        "floor lifts from .36 to .50, i.e. there is less junk to reject.\n"
        "Two limits.  It varies the initialisation but never the data, so on small "
        "synthetic problems it overcounts badly (11 clusters on rank-6 data).  And it "
        "does not discriminate between runs: its per-run counts show no detectable "
        "relationship to the run-to-run variation the spectral estimate tracks.  Use it "
        "for the level and for the Iq convergence curve, not for per-run differences.",
    )
    ica_opts.add_argument(
        "-stability_runs",
        type=int,
        default=30,
        help="Restarts for -stability_order.",
    )
    ica_opts.add_argument(
        "-split_half",
        action="store_true",
        help="Requires -temp_concat.  Report how many components reproduce across two "
        "disjoint halves of the "
        "input runs: decompose each half independently, match components one-to-one by "
        "|correlation|, and count the matches clearing -split_half_thresh.  This measures "
        "whether a component is a property of the data rather than of the fit, which "
        "restart stability cannot.  Reported only, never used to change the count.  "
        "Needs >= 2 runs and costs two extra decompositions per split.",
    )
    ica_opts.add_argument(
        "-split_half_thresh",
        type=float,
        default=0.5,
        help="Minimum |r| for a matched component pair to count as reproducible.",
    )
    ica_opts.add_argument(
        "-split_half_splits",
        type=int,
        default=1,
        help="Number of random half-partitions to average over; >1 helps when the run "
        "count is small and one partition can be unlucky.",
    )

    task = parser.add_argument_group("Task annotation (optional)")
    task.add_argument(
        "-onsets",
        nargs="+",
        default=None,
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
        "-psc_bucket",
        action="store_true",
        default=True,
        help=(
            "Write stats/psc_prob.nii.gz — interleaved [PSC_k, Prob_k] 4D bucket "
            "inside the .ica compat folder (default: on). PSC uses pre-noise-norm "
            "maps so mixing@oic ≈ data. Requires -save_mixture_z."
        ),
    )
    out.add_argument(
        "-no_psc_bucket",
        dest="psc_bucket",
        action="store_false",
        help="Disable the PSC+Prob bucket output.",
    )
    out.add_argument(
        "-zp_bucket",
        action="store_true",
        default=True,
        help=(
            "Write stats/z_prob.nii.gz — interleaved [Z_k, Prob_k] 4D bucket "
            "inside the .ica compat folder (default: on). Z bricks typed FIZT "
            "for AFNI thresholding. Requires -save_mixture_z."
        ),
    )
    out.add_argument(
        "-no_zp_bucket",
        dest="zp_bucket",
        action="store_false",
        help="Disable the Z+Prob bucket output.",
    )
    out.add_argument(
        "-per_comp_stats",
        action="store_true",
        default=False,
        help=(
            "Write per-component 3D probmap_NNN.nii.gz and thresh_zstatNNN.nii.gz "
            "files inside stats/ (zero-padded to component count). Off by default — "
            "the 4D psc_prob and z_prob buckets are more compact."
        ),
    )
    out.add_argument(
        "-psc_clip",
        type=float,
        default=50.0,
        help=(
            "Clip absolute PSC values to this magnitude (default: 50%%). "
            "Prevents low-mean voxels from blowing up the color scale. "
            "Set <=0 to disable."
        ),
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
    out.add_argument(
        "-trace",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Dump intermediate matrices for step-by-step parity validation. "
            "Writes eigenvalues, whitening/dewhitening, unmixing, mixing, "
            "and IC stats as .npy files to DIR (matches MELODIC --debug outputs)."
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
    modes.add_argument(
        "-tensor",
        action="store_true",
        help=(
            "Tensorial (spatial-concat) ICA: requires runs to share the same T. "
            "Stacks runs along the voxel axis → (T, n_runs*V) and decomposes "
            "to a single shared temporal mixing with per-run spatial maps."
        ),
    )
    modes.add_argument(
        "-temporal_ica",
        "-tica",
        dest="temporal_ica",
        action="store_true",
        help=(
            "Two-stage temporal ICA (Glasser 2018 / HCP): reduce the spatial "
            "dimension with a group spatial ICA (or PCA), dual-regress the group "
            "maps onto each run to recover per-run component timecourses, then run "
            "temporal ICA on the concatenation. -num_comps sets K_sica; "
            "-n_temporal_comps sets K_tica. Requires >= 2 input runs."
        ),
    )
    modes.add_argument(
        "-tica_reducer",
        "-tica-reducer",
        dest="tica_reducer",
        choices=["sica", "pca"],
        default="sica",
        help=(
            "Stage-1 spatial reducer for -temporal_ica: 'sica' (group spatial "
            "ICA, HCP-faithful, default) or 'pca' (MIGP top-K principal maps, "
            "lighter/faster)."
        ),
    )
    modes.add_argument(
        "-n_temporal_comps",
        "-n-temporal-comps",
        dest="n_temporal_comps",
        type=str,
        default="30",
        help=(
            "Number of temporal ICA components K_tica for -temporal_ica. Int, "
            "float in (0,1) as a fraction of K_sica, or 'auto' to estimate. "
            "Default: 30."
        ),
    )
    modes.add_argument(
        "-tica_method",
        "-tica-method",
        dest="tica_method",
        choices=["fastica", "infomax"],
        default=None,
        help=(
            "ICA solver for the temporal stage. Defaults to -ica_method. "
            "FastICA (default) is fine for the small (K_sica x T_total) matrix."
        ),
    )
    modes.add_argument(
        "-no_tica_varnorm",
        "-no-tica-varnorm",
        dest="tica_varnorm",
        action="store_false",
        default=True,
        help=(
            "Disable per-component variance normalization of the sICA "
            "timecourses before the temporal stage (HCP normalizes by default)."
        ),
    )
    modes.add_argument(
        "-tica_icasso_runs",
        "-tica-icasso-runs",
        dest="tica_icasso_runs",
        type=int,
        default=25,
        help=(
            "ICASSO repetitions to stabilize the temporal ICA and estimate "
            "per-component reproducibility (Iq). 0/1 = single FastICA (no "
            "stability). HCP uses 100; 25 is a good default. Required (auto-set "
            "to 25) when -n_temporal_comps auto."
        ),
    )
    modes.add_argument(
        "-tica_iq_thresh",
        "-tica-iq-thresh",
        dest="tica_iq_thresh",
        type=float,
        default=0.5,
        help=(
            "ICASSO Iq threshold. With -n_temporal_comps auto, keep only "
            "temporal components with Iq above this (HCP uses 0.5). The Iq of "
            "every component is written to the metadata regardless."
        ),
    )
    modes.add_argument(
        "-migp",
        action="store_true",
        help=(
            "tcat only: use MELODIC's incremental group PCA (Smith 2014) to "
            "reduce the running stack to migp_n PC time-courses on the fly. "
            "Bounds peak memory at migp_factor*migp_n*V regardless of run count."
        ),
    )
    modes.add_argument(
        "-migp_n",
        type=int,
        default=None,
        help=(
            "MIGP target dimensionality. Default (unset) → 2*T_first_run - 1 "
            "matching MELODIC's auto-pick."
        ),
    )
    modes.add_argument(
        "-migp_factor",
        type=float,
        default=2.0,
        help=(
            "MIGP reduction trigger: reduce when stack exceeds factor * migp_n rows. "
            "Default 2.0 matches MELODIC (--migp_factor). Larger values batch more "
            "files between SVD reductions (fewer calls, more peak memory)."
        ),
    )
    modes.add_argument(
        "-migp_shuffle",
        type=str,
        default=None,
        metavar="ORDER",
        help=(
            "Reorder runs before MIGP accumulation. Comma-separated 0-based "
            "indices, e.g. '1,0,3,2,4'. MELODIC randomizes file order; use this "
            "to reproduce MELODIC's order for parity validation. Only used with -migp."
        ),
    )
    vn_group = modes.add_mutually_exclusive_group()
    vn_group.add_argument(
        "-joined_vn",
        dest="joined_vn",
        action="store_true",
        default=True,
        help=(
            "tcat: compute variance-normalization stdev map jointly from the "
            "across-run average (MELODIC default). Applies same stdev map to every run."
        ),
    )
    vn_group.add_argument(
        "-sep_vn",
        dest="joined_vn",
        action="store_false",
        help="tcat: compute variance-normalization per run (legacy behavior).",
    )
    modes.add_argument(
        "-joined_whiten",
        action="store_true",
        default=False,
        help=(
            "tcat: whiten the concatenated data with a single global PCA. "
            "Default is per-run PCA+whitening (MELODIC-style block-diagonal) "
            "which preserves per-run subspaces before ICA."
        ),
    )

    misc = parser.add_argument_group("Misc")
    add_device_arg(
        misc, extra="MPS keeps bulk ICA in float32 and runs only tiny linalg islands on CPU."
    )
    misc.add_argument("-cpu", action="store_true", help="Alias for -device cpu.")
    add_verbose_arg(misc, default=0)

    return parser


def main() -> None:
    """Parse CLI args and execute run-wise ICA across all input runs."""
    parser = build_parser()
    args = parser.parse_args()

    _n_modes = int(bool(args.tensor)) + int(bool(args.temp_concat)) + int(bool(args.temporal_ica))
    if _n_modes > 1:
        raise ValueError("-tensor, -temp_concat, and -temporal_ica are mutually exclusive")

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
    device = setup_device("cpu" if args.cpu else args.device)

    print_cli_header("ffs_ica.py", "Fast run-wise whole-brain ICA")
    print(f"Device: {device}")
    print(f"Runs: {len(input_files)}")
    if args.verb >= 1:
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
    bids_task_onsets = None  # list[list[ndarray]] — all_onsets[cond][run]
    bids_task_durations = None  # list[float]
    bids_task_labels = None  # list[str]
    if _has_events:
        from fastfuncstuff.design.bids_events import check_events_pairing, parse_bids_events

        n_event_runs = len(input_files)
        if len(args.events) not in (1, n_event_runs):
            print(
                f"ERROR: -events requires one TSV per run or a single shared TSV: "
                f"got {len(args.events)} events files but {n_event_runs} input datasets."
            )
            sys.exit(1)
        try:
            check_events_pairing(input_files, args.events, n_runs=n_event_runs)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        event_cols = tuple(args.event_cols) if getattr(args, "event_cols", None) else None
        bids_task_onsets, bids_task_durations, bids_task_labels = parse_bids_events(
            event_files=args.events,
            event_ignore=getattr(args, "event_ignore", None),
            event_cols=event_cols,
            n_runs=n_event_runs,
        )
        print(
            f"Task annotation: {len(bids_task_labels)} conditions from BIDS events: "
            f"{bids_task_labels}"
        )

    # -split_half is only implemented on the concatenation path (it needs the runs in one
    # matrix to split them). Accepting it silently elsewhere made it look like the check
    # had run and found nothing to report.
    if getattr(args, "split_half", False) and not args.temp_concat:
        raise ValueError(
            "-split_half requires -temp_concat: it splits the input *runs* into two "
            "disjoint halves, which the per-run path never forms. Re-run with "
            "-temp_concat, or drop -split_half."
        )

    t_pipeline = time.time()

    if args.tensor:
        # --- Tensorial (spatial-concat) mode ---
        if len(input_files) < 2:
            raise ValueError("-tensor requires at least 2 input runs")
        print(f"\nMode: tensorial spatial-concat ({len(input_files)} runs → shared temporal ICA)")
        tensor_meta = _run_tensorial_ica(
            input_files=input_files,
            args=args,
            device=device,
            shared_mask=shared_mask,
        )
        print(
            f"  Selected components: {tensor_meta['n_components_selected']} "
            f"({tensor_meta['mask_type']} mask, {tensor_meta['n_voxels_per_run']:,} vox/run, "
            f"{tensor_meta['n_voxels_total']:,} total) | "
            f"IC1 explained share: {tensor_meta['component_variance_share'][0] * 100:.2f}%"
        )
        pfx = parse_prefix(str(args.prefix))
        summary_path = f"{pfx.stem}_ica_summary.json"
        with open(summary_path, "w") as f:
            json.dump(tensor_meta, f, indent=2)
        print("\n" + "=" * 70)
        elapsed_pipeline = time.time() - t_pipeline
        print(f"ffs_ica tensor complete ({elapsed_pipeline:.1f}s)")
        print(f"Summary: {summary_path}")
        print("=" * 70)
    elif args.temporal_ica:
        # --- Two-stage temporal ICA mode ---
        if len(input_files) < 2:
            raise ValueError("-temporal_ica requires at least 2 input runs")
        print(
            f"\nMode: temporal ICA ({len(input_files)} runs → "
            f"{args.tica_reducer} reduction → temporal ICA)"
        )
        tica_meta = _run_temporal_ica(
            input_files=input_files,
            args=args,
            device=device,
            shared_mask=shared_mask,
        )
        print(
            f"  Temporal components: {tica_meta['n_temporal_components']} "
            f"(from {tica_meta['n_spatial_components']} spatial comps, "
            f"{tica_meta['n_voxels']:,} vox, {tica_meta['total_timepoints']:,} TRs) | "
            f"tIC1 share: {tica_meta['component_variance_share'][0] * 100:.2f}%"
        )
        pfx = parse_prefix(str(args.prefix))
        summary_path = f"{pfx.stem}_ica_summary.json"
        with open(summary_path, "w") as f:
            json.dump(tica_meta, f, indent=2)
        print("\n" + "=" * 70)
        elapsed_pipeline = time.time() - t_pipeline
        print(f"ffs_ica temporal_ica complete ({elapsed_pipeline:.1f}s)")
        print(f"Summary: {summary_path}")
        print("=" * 70)
    elif args.temp_concat:
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
