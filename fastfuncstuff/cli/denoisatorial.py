#!/usr/bin/env python3
"""
ffs_denoisatorial - Combinatorial PC denoising via exhaustive subset evaluation

Instead of testing prefix subsets {0}, {0,1}, {0,1,2}, ... like 3dDenoisefast,
this tool tests ALL 2^k subsets of noise PCs to find the optimal per-run
combination. This discovers non-contiguous optimal subsets (e.g., PCs 0, 3, 5).

Algorithm:
  For each held-out run (outer LORO):
    1. Fit OLS betas on N-1 training runs
    2. Inner LORO CV on training runs -> criteria voxel pool
    3. Extract k PCs from held-out run's noise pool
    4. Evaluate all 2^k combinations on held-out run
    5. Select optimal combination (argmax median CoD)

Basic usage:
    ffs_denoisatorial -input run1.nii.gz run2.nii.gz run3.nii.gz \\
                      -onsets cond1.txt cond2.txt \\
                      -durations 2.0 5.0 \\
                      -tr 2.0 \\
                      -prefix subject01_combinatorial

For help:
    ffs_denoisatorial -help
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

try:
    import nibabel as nib
except ImportError:
    print("ERROR: nibabel is required. Install with: pip install nibabel")
    sys.exit(1)

try:
    from fastfuncstuff.cli_utils import (
        LoadResult,
        add_verbose_arg,
        auto_polort,
        load_and_preprocess_runs,
        parse_input_files,
        parse_prefix,
        print_cli_header,
    )
    from fastfuncstuff.denoise.sequential import select_noise_pool_voxels
    from fastfuncstuff.denoise.combinatorial import (
        CombinatorialDenoiseResults,
        compute_initial_xval_r2,
        compute_optimized_xval_r2_3dDenoise_style,
        fit_combinatorial_denoising,
        plot_combinatorial_results,
        plot_inclusion_heatmap,
        plot_plateau_curves,
        plot_singleton_contributions,
    )
    from fastfuncstuff.design.builder import parse_afni_timing_file, parse_durations
    from fastfuncstuff.glm.core import construct_polynomial_matrix
    from fastfuncstuff.design.hrf import get_hrf_library
    from fastfuncstuff.design.hrf_selection import load_nuisance_file
    from fastfuncstuff.glm.ridge import load_hrf_indices
    from fastfuncstuff.io.afni import save_nifti
    from fastfuncstuff.utils import configure_torch_backends, get_device, scale_to_percent_signal, to_tensor
except ImportError as e:
    print(f"ERROR: Could not import fastfuncstuff: {e}")
    print("Make sure fastfuncstuff is installed: pip install -e .")
    sys.exit(1)


# ============================================================================
# Argument parser
# ============================================================================


class _HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Show defaults while preserving raw description formatting."""


def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="ffs_denoisatorial - Combinatorial PC Denoising",
        formatter_class=_HelpFormatter,
        epilog="""
Examples:
  # Basic combinatorial denoising (7 PCs, 128 combinations)
  ffs_denoisatorial -input run1.nii.gz run2.nii.gz run3.nii.gz \\
                    -onsets cond1.txt cond2.txt \\
                    -durations 2.0 \\
                    -tr 2.0 \\
                    -prefix subject01_combinatorial

  # Fewer PCs for faster evaluation
  ffs_denoisatorial -input run*.nii.gz \\
                    -onsets face.txt house.txt \\
                    -durations 2.0 \\
                    -tr 2.0 \\
                    -max_pcs 5 \\
                    -prefix sub01_combo5

  # With diagnostic plots
  ffs_denoisatorial -input run*.nii.gz \\
                    -onsets stim.txt \\
                    -durations 1.0 \\
                    -tr 2.0 \\
                    -plots full \\
                    -prefix sub01_full_diagnostics

Outputs:
    Core outputs:
        {prefix}_initial_r2.nii.gz               - Initial xval R2 (task-only)
        {prefix}_optimized_xval_r2.nii.gz         - Xval R2 with optimal per-run PCs
        {prefix}_noise_pool_mask.nii.gz           - Noise pool voxels
        {prefix}_run{NN}_optimal_pcs.json         - Optimal PC indices per run
        {prefix}_run{NN}_selected_PCs.txt         - Selected PC timecourses per run
        {prefix}_combinatorial_results.pt         - Full results (PyTorch)
        {prefix}_metadata.json                    - Reproducibility metadata

    With -plots yes/full:
        {prefix}_figures/combinatorial_scatter.png  - Per-run CoD vs variance plots
        {prefix}_figures/combinatorial_heatmap.png  - PC selection heatmap

Workflow:
  1. Compute initial task-only cross-validated R2
  2. Select noise pool (low R2) and criteria voxels (high R2)
  3. For each held-out run:
     a. Fit betas on training runs
     b. Inner CV to refine criteria pool
     c. Extract PCs from held-out run
     d. Evaluate all 2^k PC combinations
     e. Select optimal combination
  4. Compute final cross-validated R2 with optimal PCs
  5. Save results and plots

Notes:
  - At least 3 runs required (outer LORO + inner LORO needs >=2 training runs)
  - Default max_pcs=7 gives 128 combinations (GPU-friendly)
  - max_pcs=10 gives 1024 combinations (still fast on GPU)
  - max_pcs>12 not recommended (4096+ combinations, memory-heavy)
        """,
    )

    # Required arguments
    required = parser.add_argument_group("Required Arguments")
    required.add_argument(
        "-input",
        nargs="+",
        required=True,
        help="Input fMRI dataset(s). Multiple files = multiple runs.",
    )
    required.add_argument(
        "-onsets",
        nargs="+",
        required=True,
        help="Onset timing files (AFNI format). One file per condition.",
    )
    required.add_argument(
        "-durations",
        nargs="+",
        required=True,
        help="Stimulus durations in seconds. Either single value or one per condition.",
    )
    required.add_argument(
        "-tr",
        type=float,
        required=False,
        default=None,
        help="Repetition time (TR) in seconds. If not specified, read from NIfTI header.",
    )
    required.add_argument(
        "-prefix",
        required=True,
        help="Output file prefix (e.g., 'output/subject01')",
    )

    # Combinatorial options
    combo_opts = parser.add_argument_group("Combinatorial Options")
    combo_opts.add_argument(
        "-max_pcs",
        type=int,
        default=7,
        help="Number of PCs to extract per run. 2^max_pcs combinations evaluated. "
        "(default: 7, giving 128 combinations)",
    )
    combo_opts.add_argument(
        "-r2_threshold",
        type=float,
        default=0.05,
        help="R2 threshold for noise pool selection (default: 0.05). "
        "Voxels with R2 < threshold are noise pool.",
    )
    combo_opts.add_argument(
        "-criteria_r2_threshold",
        type=float,
        default=0.0,
        help="Min inner-CV R2 for criteria voxels (default: 0.0).",
    )
    combo_opts.add_argument(
        "-selection_strategy",
        type=str,
        choices=["argmax", "parsimonious"],
        default="argmax",
        help="Strategy for selecting optimal combination: "
        "'argmax' (highest CoD, default), "
        "'parsimonious' (fewest PCs within 1%% of max). "
        "Ignored if -singleton_only is set.",
    )
    combo_opts.add_argument(
        "-singleton_only",
        action="store_true",
        help="Singleton-only mode: evaluate only individual PCs, not all combinations. "
        "Selects all PCs with positive delta vs baseline. "
        "Much faster (k+1 combos instead of 2^k).",
    )
    combo_opts.add_argument(
        "-compare",
        action="store_true",
        help="Also run a standard GLMdenoise-style baseline (incremental PCs in "
        "variance order, stop when adding the next PC gains less than 5%% in "
        "median R² — i.e. pcstop=1.05). Reports R² boost difference between the "
        "combinatorial/singleton selection and the GLMdenoise selection, and "
        "writes {prefix}_r2_glmdenoise and {prefix}_r2_delta NIfTIs alongside "
        "the usual outputs. Useful for quantifying whether the combinatorial "
        "approach is actually buying you anything on your data.",
    )
    combo_opts.add_argument(
        "-brainthresh",
        nargs=2,
        type=float,
        metavar=("PERCENTILE", "FRACTION"),
        default=None,
        help="Signal intensity threshold for noise pool selection. "
        "Example: -brainthresh 99 0.5",
    )
    combo_opts.add_argument(
        "-min_noise_voxels",
        type=int,
        default=100,
        help="Minimum voxels required in noise pool (default: 100)",
    )
    combo_opts.add_argument(
        "-max_noise_fraction",
        type=float,
        default=0.95,
        help="Maximum fraction of voxels in noise pool (default: 0.95)",
    )

    # Processing options
    proc_opts = parser.add_argument_group("Processing Options")
    proc_opts.add_argument(
        "-hrf_opt",
        type=str,
        default=None,
        help="3dHRFoptfast output prefix. Loads {prefix}_hrf_index.nii.gz for "
        "per-voxel HRF optimization. Each voxel uses its assigned HRF for "
        "design construction. Mutually exclusive with -canonical.",
    )
    proc_opts.add_argument(
        "-mask",
        help="Mask file to restrict analysis to brain voxels",
    )
    proc_opts.add_argument(
        "-polort",
        type=int,
        default=None,
        help="Polynomial order for drift modeling (default: auto based on run length)",
    )
    proc_opts.add_argument(
        "-ortvec",
        action="append",
        nargs=2,
        metavar=("FILE", "LABEL"),
        help="Additional nuisance regressors (can be repeated). "
        "FILE: text file with nuisance columns. "
        "LABEL: prefix for column names. "
        "Must span all runs concatenated.",
    )
    proc_opts.add_argument(
        "-microtime_dt",
        type=float,
        default=0.1,
        help="Microtime resolution in seconds (default: 0.1)",
    )
    proc_opts.add_argument(
        "-hrf_model",
        type=str,
        default="spmg1",
        help="HRF model: 'spmg1' (default), 'spmg2', 'spmg3', 'glmsingle', 'FIR', 'TENT', or 'TENT(bot,top,n)'. "
        "SPMG2 = canonical + temporal derivative. SPMG3 = canonical + time + dispersion derivatives. "
        "FIR/TENT use durations to set window. Mutually exclusive with -hrf_opt.",
    )
    proc_opts.add_argument(
        "-canonical",
        type=str,
        default=None,
        help="DEPRECATED: Use -hrf_model instead.",
    )
    proc_opts.add_argument(
        "-do_scale",
        action="store_true",
        help="Scale each voxel per run to mean=100 (percent signal change)",
    )
    proc_opts.add_argument(
        "-do_blur",
        type=float,
        metavar="FWHM",
        default=None,
        help="Apply 3D Gaussian spatial smoothing with FWHM in mm",
    )
    proc_opts.add_argument(
        "-device",
        type=str,
        help="Force device: 'cpu' or 'cuda' (default: auto-detect GPU)",
    )
    proc_opts.add_argument(
        "-keep_on_cpu",
        action="store_true",
        help="Load data to CPU and process in GPU chunks (for large datasets)",
    )
    add_verbose_arg(proc_opts, default=0)
    proc_opts.add_argument(
        "-dry_run",
        action="store_true",
        help="Fast testing mode: load only first run, generate synthetic data for rest. "
        "Results are nonsensical but pipeline runs quickly for testing.",
    )

    # Output options
    out_opts = parser.add_argument_group("Output Options")
    out_opts.add_argument(
        "-plots",
        type=str,
        choices=["no", "yes", "full"],
        default="no",
        help="Save diagnostic plots: 'no' (none), 'yes' (scatter only), "
        "'full' (scatter + heatmap)",
    )
    out_opts.add_argument(
        "-save_pcs",
        type=str,
        choices=["no", "timecourse", "both"],
        default="timecourse",
        help="Save noise PCs: 'no', 'timecourse' (default: selected PCs as txt), 'both' (.pt + txt)",
    )

    return parser


# ============================================================================
# Output saving
# ============================================================================


def save_combinatorial_results(
    results: CombinatorialDenoiseResults,
    initial_r2_full: torch.Tensor,
    optimized_r2_full: torch.Tensor,
    output_prefix: str,
    volume_shape: tuple,
    affine: np.ndarray,
    run_starts: list[int],
    tr: float,
    condition_labels: list[str],
    mask_flat: np.ndarray | None = None,
    plots_mode: str = "no",
    save_pcs_mode: str = "timecourse",
    nii_ext: str = ".nii.gz",
) -> dict:
    """Save combinatorial denoising results to disk."""
    output_files = {}

    # Helper to reshape flat data to volume
    def to_volume(flat_data):
        if torch.is_tensor(flat_data):
            flat_np = flat_data.cpu().numpy()
        else:
            flat_np = flat_data
        flat_np = flat_np.astype(np.float32)

        if mask_flat is not None:
            vol = np.zeros(mask_flat.size, dtype=np.float32)
            vol[mask_flat] = flat_np
        else:
            vol = flat_np
        return vol.reshape(volume_shape)

    # Ensure output directory exists
    prefix_dir = Path(output_prefix).parent
    if prefix_dir != Path("."):
        prefix_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initial R2 volume
    initial_r2_vol = to_volume(initial_r2_full)
    initial_r2_path = f"{output_prefix}_initial_r2{nii_ext}"
    save_nifti(initial_r2_vol, output_path=initial_r2_path, affine=affine)
    output_files["initial_r2"] = initial_r2_path
    print(f"  Saved: {initial_r2_path}")

    # 2. Optimized R2 volume
    opt_r2_vol = to_volume(optimized_r2_full)
    opt_r2_path = f"{output_prefix}_optimized_xval_r2{nii_ext}"
    save_nifti(opt_r2_vol, output_path=opt_r2_path, affine=affine)
    output_files["optimized_xval_r2"] = opt_r2_path
    print(f"  Saved: {opt_r2_path}")

    # 3. Noise pool mask
    noise_pool_vol = to_volume(results.noise_pool_mask)
    noise_pool_path = f"{output_prefix}_noise_pool_mask{nii_ext}"
    save_nifti(noise_pool_vol, output_path=noise_pool_path, affine=affine)
    output_files["noise_pool_mask"] = noise_pool_path
    print(f"  Saved: {noise_pool_path}")

    # 4. Per-run optimal PC indices (JSON)
    for run_res in results.per_run_results:
        run_idx = run_res.run_idx
        pc_info = {
            "run_idx": run_idx,
            "optimal_combination": list(run_res.optimal_combination),
            "optimal_cod": float(run_res.optimal_cod),
            "n_criteria_voxels": run_res.n_criteria_voxels,
            "explained_variance_ratios": run_res.explained_variance_ratios.tolist(),
        }
        json_path = f"{output_prefix}_run{run_idx:02d}_optimal_pcs.json"
        with open(json_path, "w") as f:
            json.dump(pc_info, f, indent=2)
        output_files[f"run{run_idx:02d}_optimal_pcs"] = json_path

    print(f"  Saved optimal PC indices for {len(results.per_run_results)} runs")

    # 5. Selected PC timecourses as text files
    if save_pcs_mode in ["timecourse", "both"]:
        for run_res in results.per_run_results:
            run_idx = run_res.run_idx
            pcs = results.noise_pcs_per_run[run_idx]
            if torch.is_tensor(pcs):
                pcs_np = pcs.cpu().numpy()
            else:
                pcs_np = pcs

            selected_idx = list(run_res.optimal_combination)
            if len(selected_idx) > 0:
                selected_pcs = pcs_np[:, selected_idx]
            else:
                selected_pcs = np.zeros((pcs_np.shape[0], 0))

            pc_txt_path = f"{output_prefix}_run{run_idx:02d}_selected_PCs.txt"
            with open(pc_txt_path, "w") as f:
                f.write(f"# Selected noise PCs for run {run_idx}\n")
                f.write(f"# Selected PC indices: {selected_idx}\n")
                f.write(f"# Shape: {selected_pcs.shape[0]} timepoints x {selected_pcs.shape[1]} PCs\n")
                if selected_pcs.shape[1] > 0:
                    np.savetxt(f, selected_pcs, fmt="%.6f", delimiter="\t")
            output_files[f"run{run_idx:02d}_selected_pcs_txt"] = pc_txt_path

        print(f"  Saved selected PC timecourses for {len(results.per_run_results)} runs")

    # 6. Full results as PyTorch file
    if save_pcs_mode == "both":
        results_path = f"{output_prefix}_combinatorial_results.pt"
        torch.save(
            {
                "noise_pcs_per_run": results.noise_pcs_per_run,
                "per_run_optimal_combinations": [
                    r.optimal_combination for r in results.per_run_results
                ],
                "per_run_all_cod": [r.all_cod for r in results.per_run_results],
                "per_run_all_var_explained": [
                    r.all_var_explained for r in results.per_run_results
                ],
                "per_run_variance_ratios": [
                    r.explained_variance_ratios for r in results.per_run_results
                ],
                "metadata": results.metadata,
            },
            results_path,
        )
        output_files["combinatorial_results"] = results_path
        print(f"  Saved: {results_path}")

    # 7. Metadata JSON
    metadata = {
        **results.metadata,
        "per_run_optimal_combinations": {
            f"run{r.run_idx:02d}": list(r.optimal_combination)
            for r in results.per_run_results
        },
        "per_run_optimal_cod": {
            f"run{r.run_idx:02d}": float(r.optimal_cod)
            for r in results.per_run_results
        },
        "condition_labels": condition_labels,
        "volume_shape": list(volume_shape),
        "tr": tr,
        "run_starts": run_starts,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    metadata_path = f"{output_prefix}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    output_files["metadata"] = metadata_path
    print(f"  Saved: {metadata_path}")

    # 8. Plots
    if plots_mode in ["yes", "full"]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig_prefix = f"{output_prefix}_figures"
            Path(fig_prefix).mkdir(parents=True, exist_ok=True)

            figs = plot_combinatorial_results(results, f"{fig_prefix}/")

            # Singleton PC contributions: individual effect of each PC
            singleton_figs = plot_singleton_contributions(results, f"{fig_prefix}/")
            figs.extend(singleton_figs)

            # Plateau curves: best achievable CoD with N PCs
            plateau_figs = plot_plateau_curves(results, f"{fig_prefix}/")
            figs.extend(plateau_figs)

            # Inclusion heatmap: delta R² coloring with X marks
            heatmap_figs = plot_inclusion_heatmap(results, f"{fig_prefix}/")
            figs.extend(heatmap_figs)

            for fig in figs:
                plt.close(fig)

            print(f"  Saved plots to {fig_prefix}/")
            output_files["plots_dir"] = fig_prefix
        except Exception as e:
            print(f"  Warning: Could not create plots: {e}")

    return output_files


# ============================================================================
# Main
# ============================================================================


def main():
    parser = create_parser()

    # Show help and exit when called with no args (argparse's -h/--help is fine)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    pfx = parse_prefix(args.prefix)
    args.prefix = pfx.stem  # overwrite with clean stem
    _nii_ext = pfx.nifti_ext

    print_cli_header("ffs_denoisatorial", "Combinatorial PC Denoising")

    # ======================================================================
    # Parse and validate inputs
    # ======================================================================
    input_files = parse_input_files(args.input)
    n_runs = len(input_files)

    if n_runs < 3:
        print("ERROR: At least 3 runs required for combinatorial denoising")
        print("  (outer LORO needs >=2 training runs for inner LORO)")
        sys.exit(1)

    # Parse onset files
    onset_files = args.onsets
    n_conditions = len(onset_files)
    from fastfuncstuff.cli_utils import clean_condition_labels
    condition_labels = clean_condition_labels([Path(f).stem for f in onset_files])

    for f in onset_files:
        if not Path(f).exists():
            print(f"ERROR: Onset file not found: {f}")
            sys.exit(1)

    # Parse durations
    durations = parse_durations(args.durations, n_conditions, condition_labels)
    print(f"  Conditions: {n_conditions} ({', '.join(condition_labels)})")
    print(f"  Durations: {durations}s")

    # Parse HRF model arguments
    from fastfuncstuff.cli_utils import parse_hrf_model_args, validate_hrf_compatibility

    hrf_info = parse_hrf_model_args(
        hrf_model_arg=args.hrf_model,
        canonical_arg=args.canonical,
        durations=durations,
        condition_labels=condition_labels,
        tr=args.tr,
    )

    hrf_model_name = hrf_info["hrf_model_name"]
    _hrf_params = hrf_info["hrf_params"]
    is_fir_model = hrf_info["is_fir_model"]
    fir_bot = hrf_info["fir_bot"]
    fir_top = hrf_info["fir_top"]
    n_basis = hrf_info["n_basis"]
    _condition_labels_full = hrf_info["condition_labels_full"]

    # Check for incompatible options with FIR models
    validate_hrf_compatibility(
        is_fir_model=is_fir_model,
        single_trial=False,  # ffs_denoisatorial doesn't have -single_trial
        hrf_opt=args.hrf_opt,
    )

    # Warn about large max_pcs
    if args.max_pcs > 12:
        print(f"  WARNING: max_pcs={args.max_pcs} gives {2**args.max_pcs} combinations.")
        print("  This may be slow and memory-intensive. Consider max_pcs <= 10.")

    n_combos = 2 ** args.max_pcs
    print(f"  Max PCs: {args.max_pcs} -> {n_combos} combinations per run")

    # Setup device
    if args.device:
        device = torch.device(args.device if args.device.lower() != "cpu" else "cpu")
    else:
        device = get_device()
    configure_torch_backends(device)
    print(f"  Device: {device}")

    # ======================================================================
    # Load data
    # ======================================================================
    print()
    load_result: LoadResult = load_and_preprocess_runs(
        input_files=input_files,
        tr=args.tr,
        mask_file=args.mask,
        blur_fwhm=args.do_blur,
        do_scale=False,
        device=device,
        force_cpu=args.keep_on_cpu,
        dry_run=args.dry_run,
        verbose=True,
    )

    # Modify prefix for dry run mode
    if args.dry_run:
        args.prefix = f"dry_run_{args.prefix}"

    data = load_result.data
    run_starts = load_result.run_starts
    affine = load_result.affine
    volume_shape = load_result.volume_shape
    mask = load_result.mask
    mask_flat = load_result.mask_flat
    n_voxels = load_result.n_voxels
    n_timepoints = load_result.n_timepoints
    n_runs = load_result.n_runs

    if args.tr is None:
        args.tr = load_result.tr

    # Compute brainthresh intensity mask BEFORE scaling
    brainthresh_mask = None
    if args.brainthresh is not None:
        percentile, fraction = args.brainthresh
        print()
        print(f"Computing intensity threshold (brainthresh={percentile}, {fraction})...")
        mean_intensity = data.mean(dim=1)
        percentile_value = torch.quantile(mean_intensity, percentile / 100.0)
        threshold = percentile_value * fraction
        brainthresh_mask = mean_intensity > threshold
        n_above = brainthresh_mask.sum().item()
        print(f"  {percentile:.0f}th percentile intensity: {percentile_value:.2f}")
        print(f"  Threshold: {threshold:.2f}")
        print(f"  Voxels above: {n_above:,} of {n_voxels:,} ({n_above / n_voxels * 100:.1f}%)")

    # Filter zero-variance voxels
    print()
    print("Filtering voxels with invalid data in any run...")
    valid_per_run_mask = torch.ones(n_voxels, dtype=torch.bool, device=data.device)
    for run_idx in range(n_runs):
        start_tp = run_starts[run_idx]
        end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
        run_data = data[:, start_tp:end_tp]
        run_std = run_data.std(dim=1)
        run_valid = run_std > 1e-6
        valid_per_run_mask &= run_valid

    n_valid = valid_per_run_mask.sum().item()
    n_invalid = n_voxels - n_valid
    if n_invalid > 0:
        print(f"  Removed {n_invalid:,} voxels with zero/constant values in any run")
        if brainthresh_mask is not None:
            brainthresh_mask = brainthresh_mask & valid_per_run_mask
        else:
            brainthresh_mask = valid_per_run_mask

    # Optional scaling
    if args.do_scale:
        print()
        data, _, _ = scale_to_percent_signal(
            data=data, run_starts=run_starts, max_scale=200.0, verbose=True,
        )

    print(f"\n  Data shape: {data.shape} ({n_voxels:,} voxels x {n_timepoints} timepoints)")
    print(f"  Runs: {n_runs} starting at {run_starts}")

    # ======================================================================
    # Build design matrix
    # ======================================================================
    print()
    print("Building design matrix...")

    # Parse onset files
    all_onsets = []
    for onset_file in onset_files:
        onsets_by_run = parse_afni_timing_file(onset_file)
        if len(onsets_by_run) != n_runs:
            print(f"ERROR: Onset file {onset_file} has {len(onsets_by_run)} runs, expected {n_runs}")
            sys.exit(1)
        all_onsets.append(onsets_by_run)

    # Build onset matrix at microtime resolution
    bins_per_tr = int(np.round(args.tr / args.microtime_dt))
    n_microtime = n_timepoints * bins_per_tr
    onset_matrix_micro = torch.zeros((n_microtime, n_conditions), device=device)

    for cond_idx in range(n_conditions):
        duration_bins = max(1, int(np.round(durations[cond_idx] / args.microtime_dt)))
        for run_idx in range(n_runs):
            onsets = all_onsets[cond_idx][run_idx]
            run_start_tr = run_starts[run_idx]
            run_start_micro = run_start_tr * bins_per_tr
            for onset_time in onsets:
                onset_bin = run_start_micro + int(np.round(onset_time / args.microtime_dt))
                if onset_bin < n_microtime:
                    onset_matrix_micro[
                        onset_bin: min(onset_bin + duration_bins, n_microtime),
                        cond_idx,
                    ] = 1.0

    # Convolve with HRF(s)
    task_design = None
    designs_by_hrf = None
    hrf_indices = None

    if args.hrf_opt:
        # Per-voxel HRF mode: load HRF indices and library from 3dHRFoptfast output
        print(f"  Loading HRF optimization results from {args.hrf_opt}...")
        hrf_index_file = f"{args.hrf_opt}_hrf_index.nii.gz"
        if not Path(hrf_index_file).exists():
            print(f"ERROR: HRF index file not found: {hrf_index_file}")
            print("  Expected output from 3dHRFoptfast with prefix:", args.hrf_opt)
            sys.exit(1)

        mask_for_hrf = mask if mask is not None else None
        hrf_indices = load_hrf_indices(hrf_index_file, mask=mask_for_hrf)
        hrf_indices = hrf_indices.to(data.device)
        print(f"  Loaded HRF indices: {hrf_indices.shape}")

        # Load or reconstruct HRF library
        hrf_lib_file = f"{args.hrf_opt}_hrf_library.pt"
        if Path(hrf_lib_file).exists():
            hrf_lib_data = torch.load(hrf_lib_file, weights_only=False)
            hrf_library = hrf_lib_data["hrf_library"]
            print(f"  Loaded HRF library from {hrf_lib_file}: {hrf_library.shape}")
        else:
            # Determine n_hrfs from the unique indices
            n_hrfs = int(hrf_indices.max().item()) + 1
            hrf_library = get_hrf_library(
                mode="library", tr=args.tr, n_hrfs=n_hrfs,
                microtime_dt=args.microtime_dt, device=device,
            )
            print(f"  Using default HRF library with {hrf_library.shape[0]} HRFs")

        # Show HRF distribution
        unique_hrfs, counts = torch.unique(hrf_indices, return_counts=True)
        print(f"  HRF distribution across {len(unique_hrfs)} unique HRFs:")
        for hrf_idx_show, count in zip(unique_hrfs[:5].tolist(), counts[:5].tolist(), strict=False):
            print(f"    HRF {hrf_idx_show}: {count:,} voxels ({count / n_voxels * 100:.1f}%)")
        if len(unique_hrfs) > 5:
            print(f"    ... and {len(unique_hrfs) - 5} more HRFs")

        # Build per-HRF design matrices using refactored function
        from fastfuncstuff.cli_utils import build_task_design_from_args
        task_design, designs_by_hrf = build_task_design_from_args(
            hrf_model_name=hrf_model_name,
            is_fir_model=is_fir_model,
            fir_bot=fir_bot,
            fir_top=fir_top,
            n_basis=n_basis,
            all_onsets=all_onsets,
            onset_matrix_micro=onset_matrix_micro,
            n_conditions=n_conditions,
            n_timepoints=n_timepoints,
            run_starts=run_starts,
            tr=args.tr,
            microtime_dt=args.microtime_dt,
            device=device,
            hrf_opt=args.hrf_opt,
            hrf_library=hrf_library,
            hrf_indices=hrf_indices,
            n_voxels=n_voxels,
        )
    else:
        # Single HRF model for all voxels - use refactored function
        from fastfuncstuff.cli_utils import build_task_design_from_args
        task_design, designs_by_hrf = build_task_design_from_args(
            hrf_model_name=hrf_model_name,
            is_fir_model=is_fir_model,
            fir_bot=fir_bot,
            fir_top=fir_top,
            n_basis=n_basis,
            all_onsets=all_onsets,
            onset_matrix_micro=onset_matrix_micro,
            n_conditions=n_conditions,
            n_timepoints=n_timepoints,
            run_starts=run_starts,
            tr=args.tr,
            microtime_dt=args.microtime_dt,
            device=device,
            hrf_opt=None,
            hrf_library=None,
            hrf_indices=None,
            n_voxels=None,
        )

    # Build nuisance per run (polynomials + ortvec)
    nuisance_per_run = []
    max_nuisance_cols = 0

    for run_idx in range(n_runs):
        start_tp = run_starts[run_idx]
        end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
        run_length = end_tp - start_tp

        if args.polort is None:
            run_duration = run_length * args.tr
            polort = auto_polort(run_duration, formula="afni")
        else:
            polort = args.polort

        if polort >= 0:
            poly = construct_polynomial_matrix(run_length, polort, device=device)
        else:
            poly = torch.zeros((run_length, 0), device=device)

        nuisance_per_run.append(poly)
        max_nuisance_cols = max(max_nuisance_cols, poly.shape[1])

    # Add ortvec if provided
    if args.ortvec:
        ortvec_all = []
        for ortvec_file, _label in args.ortvec:
            ortvec_data = load_nuisance_file(ortvec_file)
            ortvec_data = to_tensor(ortvec_data, device=device)
            if ortvec_data.shape[0] != n_timepoints:
                print(f"ERROR: ortvec file {ortvec_file} has {ortvec_data.shape[0]} rows, "
                      f"expected {n_timepoints}")
                sys.exit(1)
            ortvec_all.append(ortvec_data)

        ortvec_concat = torch.cat(ortvec_all, dim=1) if ortvec_all else None

        if ortvec_concat is not None:
            for run_idx in range(n_runs):
                start_tp = run_starts[run_idx]
                end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
                ortvec_run = ortvec_concat[start_tp:end_tp, :]
                nuisance_per_run[run_idx] = torch.cat(
                    [nuisance_per_run[run_idx], ortvec_run], dim=1,
                )
            max_nuisance_cols = nuisance_per_run[0].shape[1]

    # Pad nuisance to same columns
    for run_idx in range(n_runs):
        n_cols = nuisance_per_run[run_idx].shape[1]
        if n_cols < max_nuisance_cols:
            padding = torch.zeros(
                (nuisance_per_run[run_idx].shape[0], max_nuisance_cols - n_cols),
                device=device,
            )
            nuisance_per_run[run_idx] = torch.cat(
                [nuisance_per_run[run_idx], padding], dim=1,
            )

    print(f"  Nuisance per run: {nuisance_per_run[0].shape[1]} cols "
          f"(polort{'+ortvec' if args.ortvec else ''})")

    # ======================================================================
    # Step 1: Compute initial cross-validated R2 (task-only, for noise pool)
    # ======================================================================
    print()
    print("=" * 70)
    print("Step 1: Computing initial cross-validated R2 (task-only)...")
    print("=" * 70)

    initial_r2 = compute_initial_xval_r2(
        data=data,
        design=task_design,
        run_starts=run_starts,
        nuisance_per_run=nuisance_per_run,
        designs_by_hrf=designs_by_hrf,
        hrf_indices=hrf_indices,
        device=device,
        verbose=args.verb >= 1,
    )

    print(f"  Initial R2: median={initial_r2.median().item():.4f}, "
          f"mean={initial_r2.mean().item():.4f}")

    # ======================================================================
    # Step 2: Select noise pool
    # ======================================================================
    print()
    print("=" * 70)
    print("Step 2: Selecting noise pool...")
    print("=" * 70)

    noise_pool_mask, criteria_mask = select_noise_pool_voxels(
        r2=initial_r2,
        threshold=args.r2_threshold,
        min_noise_voxels=args.min_noise_voxels,
        max_noise_fraction=args.max_noise_fraction,
    )

    # Apply brainthresh mask to noise pool
    if brainthresh_mask is not None:
        noise_pool_mask = noise_pool_mask & brainthresh_mask.to(noise_pool_mask.device)

    n_noise = noise_pool_mask.sum().item()
    n_criteria = criteria_mask.sum().item()
    print(f"  Noise pool: {n_noise:,} voxels")
    print(f"  Criteria: {n_criteria:,} voxels")

    # ======================================================================
    # Step 3: Run combinatorial denoising
    # ======================================================================
    print()
    print("=" * 70)
    print("Step 3: Combinatorial PC denoising...")
    print("=" * 70)

    results = fit_combinatorial_denoising(
        data=data,
        design=task_design,
        run_starts=run_starts,
        tr=args.tr,
        nuisance_per_run=nuisance_per_run,
        noise_pool_mask=noise_pool_mask,
        initial_r2=initial_r2,
        max_pcs=args.max_pcs,
        criteria_r2_threshold=args.criteria_r2_threshold,
        selection_strategy=args.selection_strategy,
        singleton_only=args.singleton_only,
        designs_by_hrf=designs_by_hrf,
        hrf_indices=hrf_indices,
        device=device,
        verbose=True,
    )

    # ======================================================================
    # Step 4: Compute final cross-validated R2 with optimal PCs
    # ======================================================================
    print()
    print("=" * 70)
    print("Step 4: Computing optimized cross-validated R2...")
    print("=" * 70)

    optimized_r2 = compute_optimized_xval_r2_3dDenoise_style(
        data=data,
        design=task_design,
        run_starts=run_starts,
        nuisance_per_run=nuisance_per_run,
        noise_pcs_per_run=results.noise_pcs_per_run,
        per_run_results=results.per_run_results,
        designs_by_hrf=designs_by_hrf,
        hrf_indices=hrf_indices,
        device=device,
        verbose=True,
    )

    print(f"  Optimized R2: median={optimized_r2.median().item():.4f}, "
          f"mean={optimized_r2.mean().item():.4f}")

    improvement = optimized_r2.median().item() - initial_r2.median().item()
    print(f"  Improvement: {improvement:+.4f} (median)")

    # ======================================================================
    # Step 4b (optional): GLMdenoise-style baseline comparison
    # ======================================================================
    # When -compare is set, run the standard incremental noise-PC sweep
    # (variance-ordered PCs, take 0..k, pick k via the GLMdenoise pcstop=1.05
    # rule on median R²) on the *same* noise pool and noise PCs. Reports how
    # much R² the combinatorial / singleton choice buys you over the
    # GLMdenoise default. No-op when -compare isn't set.
    baseline_r2_t: torch.Tensor | None = None
    delta_r2_t: torch.Tensor | None = None
    baseline_k: int | None = None
    if args.compare:
        print()
        print("=" * 70)
        print("Step 4b: GLMdenoise-style incremental baseline (pcstop=1.05)...")
        print("=" * 70)

        from fastfuncstuff.denoise.sequential import cross_validate_noise_pcs

        r2_maps_inc, r2_summary_inc = cross_validate_noise_pcs(
            data=data,
            design_matrix=task_design,
            noise_pcs=results.noise_pcs_per_run,
            run_starts=run_starts,
            tr=args.tr,
            max_components=args.max_pcs,
            nuisance=nuisance_per_run,
            cv_strategy=1,  # LORO — matches the combinatorial path
            device=device,
            verbose=args.verb >= 1,
            designs_by_hrf=designs_by_hrf,
            hrf_indices=hrf_indices,
        )

        # pcstop=1.05 rule: walk up while next/current >= 1.05 on median R².
        # If median is non-positive, fall back to argmax to stay defensive.
        PCSTOP = 1.05
        baseline_k = 0
        for k in range(len(r2_summary_inc) - 1):
            cur, nxt = r2_summary_inc[k], r2_summary_inc[k + 1]
            if cur > 1e-6 and nxt / cur >= PCSTOP:
                baseline_k = k + 1
            else:
                break
        if r2_summary_inc[baseline_k] <= 0:
            baseline_k = int(np.argmax(r2_summary_inc))

        baseline_r2_np = r2_maps_inc[:, baseline_k]
        baseline_r2_t = torch.from_numpy(baseline_r2_np).to(optimized_r2.device)
        delta_r2_t = optimized_r2 - baseline_r2_t

        print(f"  Baseline picked k={baseline_k} PCs (pcstop=1.05)")
        print(
            f"  Baseline R²: median={float(baseline_r2_t.median()):.4f}, "
            f"mean={float(baseline_r2_t.mean()):.4f}"
        )
        print(f"  Δ R² (combinatorial − baseline):")
        print(
            f"    Mean:    {float(delta_r2_t.mean()):+.4f}    "
            f"Median: {float(delta_r2_t.median()):+.4f}"
        )
        q25, q75 = (
            float(torch.quantile(delta_r2_t, 0.25)),
            float(torch.quantile(delta_r2_t, 0.75)),
        )
        print(f"    IQR:     [{q25:+.4f}, {q75:+.4f}]")
        combo_wins = int((delta_r2_t > 0.01).sum())
        base_wins = int((delta_r2_t < -0.01).sum())
        n_total = delta_r2_t.numel()
        print(
            f"    Voxels combinatorial wins (Δ > 0.01): "
            f"{combo_wins:,}/{n_total:,} ({100 * combo_wins / max(n_total, 1):.1f}%)"
        )
        print(
            f"    Voxels baseline wins (Δ < -0.01):     "
            f"{base_wins:,}/{n_total:,} ({100 * base_wins / max(n_total, 1):.1f}%)"
        )

    # ======================================================================
    # Step 5: Save results
    # ======================================================================
    print()
    print("=" * 70)
    print("Saving results...")
    print("=" * 70)

    output_files = save_combinatorial_results(
        results=results,
        initial_r2_full=initial_r2,
        optimized_r2_full=optimized_r2,
        output_prefix=args.prefix,
        volume_shape=volume_shape,
        affine=affine,
        run_starts=run_starts,
        tr=args.tr,
        condition_labels=condition_labels,
        mask_flat=mask_flat,
        plots_mode=args.plots,
        save_pcs_mode=args.save_pcs,
        nii_ext=_nii_ext,
    )

    # Save -compare outputs alongside the standard ones.
    if args.compare and baseline_r2_t is not None and delta_r2_t is not None:
        def _flat_to_vol(flat_t: torch.Tensor) -> np.ndarray:
            flat_np = flat_t.detach().cpu().numpy().astype(np.float32)
            if mask_flat is not None:
                vol = np.zeros(mask_flat.size, dtype=np.float32)
                vol[mask_flat] = flat_np
            else:
                vol = flat_np
            return vol.reshape(volume_shape)

        baseline_path = f"{args.prefix}_r2_glmdenoise{_nii_ext}"
        delta_path = f"{args.prefix}_r2_delta{_nii_ext}"
        save_nifti(_flat_to_vol(baseline_r2_t), output_path=baseline_path, affine=affine)
        save_nifti(_flat_to_vol(delta_r2_t), output_path=delta_path, affine=affine)
        output_files["r2_glmdenoise"] = baseline_path
        output_files["r2_delta"] = delta_path
        print(f"  Saved: {baseline_path}")
        print(f"  Saved: {delta_path}")

    # ======================================================================
    # Summary
    # ======================================================================
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Runs: {n_runs}")
    print(f"  Max PCs: {args.max_pcs} ({n_combos} combinations per run)")
    print(f"  Noise pool: {n_noise:,} voxels")
    print(f"  Initial median R2: {initial_r2.median().item():.4f}")
    print(f"  Optimized median R2: {optimized_r2.median().item():.4f}")
    print(f"  Improvement: {improvement:+.4f}")
    print()
    print("  Per-run selections:")
    for run_res in results.per_run_results:
        print(f"    Run {run_res.run_idx}: PCs {run_res.optimal_combination} "
              f"(CoD={run_res.optimal_cod:.4f})")
    print()
    print(f"  Outputs: {len(output_files)} files saved with prefix '{args.prefix}'")
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
