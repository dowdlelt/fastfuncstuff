#!/usr/bin/env python

"""

Command-line tool for PCA and ICA decomposition of fMRI data

Performs GPU-accelerated PCA and/or ICA on fMRI data and saves results
as 4D spatial maps and timeseries files.

Examples
--------
PCA only (keep 85% variance):
    decompose_fmri.py \
        -input func.nii.gz \
        -mask mask.nii.gz \
        -pca 0.85 \
        -output results/pca

ICA with automatic PCA reduction:
    decompose_fmri.py \
        -input func.nii.gz \
        -mask mask.nii.gz \
        -ica 25 \
        -pca 0.85 \
        -output results/ica

Both PCA and ICA:
    decompose_fmri.py \
        -input func.nii.gz \
        -mask mask.nii.gz \
        -pca 50 \
        -ica 25 \
        -output results/decomp

ICA with stability analysis:
    decompose_fmri.py \
        -input func.nii.gz \
        -mask mask.nii.gz \
        -ica 25 \
        -pca 0.85 \
        -stability 100 \
        -output results/ica_stable

Automatic component selection (slow but powerful):
    decompose_fmri.py \
        -input func.nii.gz \
        -mask mask.nii.gz \
        -ica auto \
        -ica_range 15 35 5 \
        -stability 50 \
        -output results/ica_auto

ICASSO mode (recommended for reliable components):
    decompose_fmri.py \
        -input func.nii.gz \
        -mask mask.nii.gz \
        -icasso 25 \
        -pca 0.85 \
        -n_runs 100 \
        -output results/icasso

ICASSO with automatic component selection:
    decompose_fmri.py \
        -input func.nii.gz \
        -mask mask.nii.gz \
        -icasso auto \
        -ica_range 15 35 5 \
        -n_runs 50 \
        -output results/icasso_auto
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.cli_utils import add_verbose_arg, parse_prefix, spinner
from fastfuncstuff.decomposition.ica import (
    FastICA,
    ica_stability_analysis,
    select_n_components_by_stability,
)
from fastfuncstuff.decomposition.icasso import icasso, icasso_auto_select
from fastfuncstuff.decomposition.io import save_decomposition_results
from fastfuncstuff.decomposition.pca import PCA
from fastfuncstuff.decomposition.tools import parse_num_comps_spec
from fastfuncstuff.io.afni import get_tr_from_file, load_fmri_data
from fastfuncstuff.utils import configure_torch_backends, get_device


class _HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Show defaults while preserving raw description formatting."""


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="PCA and ICA decomposition for fMRI data",
        formatter_class=_HelpFormatter,
        epilog=__doc__,
    )

    # Input data
    parser.add_argument(
        "-input",
        required=True,
        metavar="FILE",
        help="Input fMRI data (3D or 4D NIfTI)",
    )

    parser.add_argument(
        "-mask",
        required=True,
        metavar="FILE",
        help="Brain mask (3D NIfTI, values > 0 define brain voxels)",
    )

    # PCA options
    parser.add_argument(
        "-pca",
        metavar="N",
        help="Perform PCA. N can be: "
        "float (0-1) for variance percentage (e.g., 0.85 = 85 percent), "
        "int for number of components, "
        'or "all" for all components',
    )

    # ICA options
    parser.add_argument(
        "-ica",
        metavar="N",
        help="Perform ICA. N can be: "
        "int for number of components, "
        'or "auto" for automatic selection (requires -ica_range)',
    )

    parser.add_argument(
        "-icasso",
        metavar="N",
        help="Perform ICASSO (ICA with clustering for stability). N can be: "
        "int for number of components, "
        'or "auto" for automatic selection (requires -ica_range). '
        "Recommended for reliable component selection.",
    )

    parser.add_argument(
        "-ica_range",
        nargs=3,
        type=int,
        metavar=("START", "STOP", "STEP"),
        help="Range for automatic component selection (requires -ica auto or -icasso auto)",
    )

    parser.add_argument(
        "-n_runs",
        type=int,
        default=100,
        metavar="N",
        help="Number of ICA runs for ICASSO clustering (default: 100)",
    )

    parser.add_argument(
        "-stability",
        type=int,
        metavar="N",
        help="Run ICA N times with different seeds for stability analysis (non-ICASSO mode)",
    )

    parser.add_argument(
        "-min_stability",
        type=float,
        default=0.7,
        metavar="THRESHOLD",
        help="Minimum stability threshold for component selection (default: 0.7)",
    )

    # Output (canonical -prefix; -output is a silent alias)
    out_group = parser.add_mutually_exclusive_group(required=True)
    out_group.add_argument(
        "-prefix",
        dest="output",
        metavar="PREFIX",
        help="Output prefix for files (e.g., results/decomp)",
    )
    out_group.add_argument(
        "-output",
        dest="output",
        metavar="PREFIX",
        help="Alias for -prefix.",
    )

    # Advanced options
    parser.add_argument(
        "-fun",
        default="pow3",
        choices=["pow3", "cube", "logcosh", "exp"],
        help="ICA nonlinearity function (default: pow3, matching MELODIC)",
    )

    parser.add_argument(
        "-max_iter",
        type=int,
        default=200,
        metavar="N",
        help="Maximum ICA iterations (default: 200)",
    )

    parser.add_argument(
        "-seed",
        type=int,
        metavar="N",
        help="Random seed for reproducibility",
    )

    dev_group = parser.add_argument_group("Device")
    dev_group.add_argument(
        "-device",
        default=None,
        help="PyTorch device: cuda, mps, cpu (auto-detected by default)",
    )
    dev_group.add_argument(
        "-cpu",
        action="store_true",
        help="Alias for -device cpu.",
    )

    add_verbose_arg(parser, default=0)

    return parser.parse_args()


def parse_component_arg(arg_str):
    """
    Parse component specification argument

    Returns
    -------
    value : int, float, str, or None
        Parsed component specification
    """
    if arg_str is None:
        return None

    spec = arg_str.strip().lower()

    if spec == "all":
        return None
    elif spec == "auto":
        return "auto"

    try:
        value = parse_num_comps_spec(spec)
    except ValueError:
        raise ValueError(
            f"Invalid component specification: '{arg_str}'. "
            f"Use int (number), float (variance fraction), 'all', or 'auto'"
        ) from None

    if isinstance(value, float) and not 0.0 < value < 1.0:
        raise ValueError(f"Float component specification must be between 0 and 1, got {value}")

    if isinstance(value, str):
        raise ValueError(
            f"Invalid component specification: '{arg_str}'. "
            f"Use int (number), float (variance fraction), 'all', or 'auto'"
        )

    return value


def main():
    """Main CLI entry point"""
    args = parse_args()

    pfx = parse_prefix(args.output)
    args.output = pfx.stem
    _nii_ext = pfx.nifti_ext

    # Setup device (-cpu is an alias for -device cpu)
    if args.cpu:
        device = torch.device("cpu")
    elif args.device is not None:
        device = torch.device(args.device)
    else:
        device = get_device()
    configure_torch_backends(device)

    if args.verb >= 1:
        print(f"Using device: {device}")

    # Parse component arguments
    pca_n_components = parse_component_arg(args.pca) if args.pca else None
    ica_n_components = parse_component_arg(args.ica) if args.ica else None
    icasso_n_components = parse_component_arg(args.icasso) if args.icasso else None

    # Validate arguments
    if pca_n_components is None and ica_n_components is None and icasso_n_components is None:
        print("Error: Must specify at least one of -pca, -ica, or -icasso", file=sys.stderr)
        return 1

    if args.ica and args.icasso:
        print(
            "Error: Cannot specify both -ica and -icasso. Use -icasso for stability analysis.",
            file=sys.stderr,
        )
        return 1

    if ica_n_components == "auto" and args.ica_range is None:
        print("Error: -ica auto requires -ica_range START STOP STEP", file=sys.stderr)
        return 1

    if icasso_n_components == "auto" and args.ica_range is None:
        print("Error: -icasso auto requires -ica_range START STOP STEP", file=sys.stderr)
        return 1

    if args.stability and ica_n_components is None:
        print(
            "Warning: -stability specified but -ica not requested. Ignoring stability analysis.",
            file=sys.stderr,
        )
        args.stability = None

    # Validate PCA/ICA compatibility
    if pca_n_components is not None and isinstance(pca_n_components, int):
        # Check against ICA
        if ica_n_components is not None and isinstance(ica_n_components, int):
            if pca_n_components < ica_n_components:
                print(
                    f"Error: PCA components ({pca_n_components}) < ICA components ({ica_n_components})",
                    file=sys.stderr,
                )
                print(
                    "       ICA cannot extract more components than available from PCA!",
                    file=sys.stderr,
                )
                print("       Either increase -pca or decrease -ica", file=sys.stderr)
                return 1

        # Check against ICASSO
        if icasso_n_components is not None and isinstance(icasso_n_components, int):
            if pca_n_components < icasso_n_components:
                print(
                    f"Error: PCA components ({pca_n_components}) < ICASSO components ({icasso_n_components})",
                    file=sys.stderr,
                )
                print(
                    "       ICASSO cannot extract more components than available from PCA!",
                    file=sys.stderr,
                )
                print("       Either increase -pca or decrease -icasso", file=sys.stderr)
                return 1

    # Load data
    if args.verb >= 1:
        print(f"\nLoading data from: {args.input}")
        print(f"Loading mask from: {args.mask}")

    try:
        with spinner(f"Loading {Path(args.input).name}"):
            data = load_fmri_data(args.input, args.mask)
        tr = get_tr_from_file(args.input)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        return 1

    if args.verb >= 1:
        print(f"  Data shape: {data.shape}")
        print(f"  TR: {tr}s")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ========== PCA ==========
    if pca_n_components is not None:
        if args.verb >= 1:
            print(f"\n{'=' * 60}")
            print("Running PCA...")
            print(f"  n_components: {pca_n_components}")

        pca = PCA(n_components=pca_n_components, device=device)
        pca_scores = pca.fit_transform(data)

        if args.verb >= 1:
            print(f"  Kept {pca.n_components_} components")
            print(f"  Explained variance: {pca.explained_variance_ratio_.sum().item():.3f}")
            print(f"  Output shape: {pca_scores.shape}")

        # Save PCA results
        if args.verb >= 1:
            print("  Saving PCA results...")

        pca_labels = [f"PC_{i:03d}" for i in range(pca.n_components_)]

        try:
            with spinner("Writing PCA results"):
                pca_files = save_decomposition_results(
                    components=pca.components_,
                    timeseries=pca_scores,
                    mask_file=args.mask,
                    output_prefix=f"{args.output}_pca",
                    reference_file=args.input,
                    labels=pca_labels,
                    method="PCA",
                    nii_ext=_nii_ext,
                )

            if args.verb >= 1:
                print(f"    Maps: {pca_files['maps']}")
                print(f"    Timeseries: {pca_files['timeseries_1D']}")

        except Exception as e:
            print(f"Error saving PCA results: {e}", file=sys.stderr)
            return 1

    # ========== ICA ==========
    if ica_n_components is not None:
        if args.verb >= 1:
            print(f"\n{'=' * 60}")
            print("Running ICA...")

        # Determine PCA preprocessing
        if pca_n_components is not None:
            # Use PCA components from above
            pca_for_ica = pca_n_components
            if args.verb >= 1:
                print(f"  Using PCA reduction: {pca_for_ica}")
        else:
            # Default: keep 85% variance
            pca_for_ica = 0.85
            if args.verb >= 1:
                print("  Using default PCA reduction: 85% variance")

        # Auto component selection
        if ica_n_components == "auto":
            if args.verb >= 1:
                print("  Automatic component selection enabled")
                print(
                    f"    Range: {args.ica_range[0]} to {args.ica_range[1]} (step {args.ica_range[2]})"
                )
                print(f"    Stability runs: {args.stability}")

            n_range = range(args.ica_range[0], args.ica_range[1], args.ica_range[2])

            try:
                auto_results = select_n_components_by_stability(
                    data,
                    n_components_range=n_range,
                    pca_components=pca_for_ica,
                    n_runs=args.stability if args.stability else 50,
                    min_stability=args.min_stability,
                    device=device,
                    verbose=args.verb >= 1,
                )

                ica_n_components = auto_results["optimal_n_components"]

                if args.verb >= 1:
                    print(f"\n  Optimal n_components: {ica_n_components}")

                # Save stability results
                stability_file = output_path.parent / f"{output_path.name}_stability.txt"
                with open(stability_file, "w") as f:
                    f.write("# ICA automatic component selection\n")
                    f.write(f"# Optimal n_components: {ica_n_components}\n")
                    f.write("# n_components mean_stability\n")
                    for n_comp, stability in sorted(
                        auto_results["stability_by_n_components"].items()
                    ):
                        f.write(f"{n_comp} {stability:.4f}\n")

                if args.verb >= 1:
                    print(f"  Saved stability analysis: {stability_file}")

            except Exception as e:
                print(f"Error in automatic component selection: {e}", file=sys.stderr)
                return 1

        # Run ICA
        if args.verb >= 1:
            print(f"  n_components: {ica_n_components}")
            print(f"  PCA reduction: {pca_for_ica}")

        try:
            ica = FastICA(
                n_components=ica_n_components,
                pca_components=pca_for_ica,
                max_iter=args.max_iter,
                fun=args.fun,
                random_state=args.seed,
                device=device,
            )

            ica_timeseries = ica.fit_transform(data)

            if args.verb >= 1:
                print(f"  Converged in {ica.n_iter_} iterations")
                print(f"  Output shape: {ica_timeseries.shape}")

        except Exception as e:
            print(f"Error running ICA: {e}", file=sys.stderr)
            return 1

        # Stability analysis
        if args.stability and ica_n_components != "auto":
            if args.verb >= 1:
                print(f"\n  Running stability analysis ({args.stability} runs)...")

            try:
                stability_results = ica_stability_analysis(
                    data,
                    n_components=ica_n_components,
                    pca_components=pca_for_ica,
                    n_runs=args.stability,
                    device=device,
                    verbose=args.verb >= 1,
                )

                mean_stability = stability_results["stability_scores"].mean()
                n_stable = (stability_results["stability_scores"] > args.min_stability).sum()

                if args.verb >= 1:
                    print(f"    Mean stability: {mean_stability:.3f}")
                    print(
                        f"    Components with stability > {args.min_stability}: {n_stable}/{ica_n_components}"
                    )

                # Save stability scores
                stability_file = output_path.parent / f"{output_path.name}_ica_stability.1D"
                np.savetxt(
                    stability_file,
                    stability_results["stability_scores"],
                    fmt="%.4f",
                    header="Component stability scores (0-1, higher = more stable)",
                )

                if args.verb >= 1:
                    print(f"    Saved stability scores: {stability_file}")

            except Exception as e:
                print(f"Warning: Stability analysis failed: {e}", file=sys.stderr)

        # Save ICA results
        if args.verb >= 1:
            print("  Saving ICA results...")

        ica_labels = [f"IC_{i:03d}" for i in range(ica.components_.shape[0])]

        try:
            with spinner("Writing ICA results"):
                ica_files = save_decomposition_results(
                    components=ica.components_,
                    timeseries=ica_timeseries,
                    mask_file=args.mask,
                    output_prefix=f"{args.output}_ica",
                    reference_file=args.input,
                    labels=ica_labels,
                    method="ICA",
                    nii_ext=_nii_ext,
                )

            if args.verb >= 1:
                print(f"    Maps: {ica_files['maps']}")
                print(f"    Timeseries: {ica_files['timeseries_1D']}")

        except Exception as e:
            print(f"Error saving ICA results: {e}", file=sys.stderr)
            return 1

    # ========== ICASSO ==========
    if icasso_n_components is not None:
        if args.verb >= 1:
            print(f"\n{'=' * 60}")
            print("Running ICASSO...")

        # Determine PCA preprocessing
        # CRITICAL: PCA components must be >= ICA components!
        if pca_n_components is not None:
            pca_for_icasso = pca_n_components
            if args.verb >= 1:
                print(f"  User-specified PCA: {pca_for_icasso}")
        else:
            # Auto-set PCA to extract all possible components
            # PCA should always decompose the data entirely up to min(n_timepoints, n_voxels)
            # Then we select the TOP N for ICA
            if args.verb >= 1:
                n_timepoints, n_voxels = data.shape
                max_pca = min(n_timepoints, n_voxels)
                print(f"  Auto-set PCA components: {max_pca} (full decomposition)")
                print(f"    Data shape: ({n_timepoints} timepoints, {n_voxels} voxels)")
                print("    Will select top N components for each ICA run")
            pca_for_icasso = None  # Let PCA extract all components

        # Validate ica_range is only used with auto mode
        if icasso_n_components != "auto" and args.ica_range is not None:
            print("\nWARNING: -ica_range is ignored when -icasso specifies a fixed number")
            print("         -ica_range only works with '-icasso auto'")
            print(
                f"         Current: Running {args.n_runs} iterations with {icasso_n_components} components"
            )
            print()

        # Auto component selection
        if icasso_n_components == "auto":
            if args.verb >= 1:
                print("  Automatic component selection via ICASSO")
                print(
                    f"    Range: {args.ica_range[0]} to {args.ica_range[1]} (step {args.ica_range[2]})"
                )
                print(f"    Runs per n_components: {args.n_runs}")

            n_range = range(args.ica_range[0], args.ica_range[1], args.ica_range[2])

            try:
                icasso_results = icasso_auto_select(
                    data,
                    n_components_range=n_range,
                    n_runs=args.n_runs,
                    pca_components=pca_for_icasso,
                    min_stability=args.min_stability,
                    device=device,
                    verbose=args.verb >= 1,
                    batch_size=None,  # Auto-select based on size
                )

                # Extract results from optimal run
                optimal_results = icasso_results["optimal_results"]
                stable_components = optimal_results["components"]
                stable_mixing = optimal_results["mixing"]
                stability_scores = optimal_results["stability"]
                n_stable = optimal_results["n_stable"]
                all_stability = optimal_results["all_stability"]
                n_components_total = optimal_results["n_components"]
                cluster_quality = optimal_results["cluster_quality"]

                if args.verb >= 1:
                    print("\nOptimal configuration:")
                    print(f"  Requested: {icasso_results['optimal_n_components']} components")
                    print(f"  Stable: {n_stable} components")
                    print(
                        f"  Stability range: {all_stability.min():.3f} - {all_stability.max():.3f}"
                    )
                    print("\nCluster quality:")
                    print(f"  Mean compactness: {cluster_quality['compactness'].mean():.3f}")
                    print(
                        f"  Mean cluster size: {cluster_quality['size'].mean():.1f} (expected: {args.n_runs})"
                    )
                    print(
                        f"  Size range: {cluster_quality['size'].min()}-{cluster_quality['size'].max()}"
                    )

                    # Report PCA variance
                    pca_var = optimal_results.get("pca_variance_cumsum")
                    if pca_var is not None:
                        print("\nPCA preprocessing:")
                        print(
                            f"  {n_components_total} components explain {pca_var[n_components_total - 1]:.1%} variance"
                        )

                # Always save ALL components from the optimal n_components choice
                # User wants to see all components from the most stable configuration
                print(
                    f"\nSaving ALL {n_components_total} components from optimal n_components = {icasso_results['optimal_n_components']}"
                )
                if n_stable == 0:
                    print(f"  NOTE: None met stability threshold {args.min_stability}")
                    print(f"        Best stability: {all_stability.max():.3f}")
                print("  Full stability report saved for component selection")

                # Get ALL centroids from optimal run
                all_centroids = optimal_results["all_centroids"]
                stable_components = all_centroids  # ALL components

                # Get mixing for ALL components (matched to centroids)
                stable_mixing = optimal_results["all_mixing"]

                stability_scores = all_stability  # All stability scores
                n_stable = n_components_total  # Save all

                # Save component selection summary with QC metrics
                summary_file = output_path.parent / f"{output_path.name}_icasso_selection.txt"
                with open(summary_file, "w") as f:
                    f.write("# ICASSO Automatic Component Selection Results\n")
                    f.write(f"# Optimal n_components: {icasso_results['optimal_n_components']}\n")
                    f.write(f"# Components meeting threshold: {n_stable}\n")
                    f.write(f"# Runs per n_components: {args.n_runs}\n")
                    f.write(f"# Min stability threshold: {args.min_stability}\n")
                    f.write("#\n")
                    f.write("# n_comp | n_stable | ratio | mean_stab | mean_size | pca_var_exp\n")
                    f.write("# -------|----------|-------|-----------|-----------|------------\n")
                    for n in sorted(icasso_results["n_stable_by_n_components"].keys()):
                        result_n = icasso_results["all_results"][n]
                        n_s = icasso_results["n_stable_by_n_components"][n]
                        ratio = icasso_results["stability_ratios"][n]
                        mean_stab = result_n["all_stability"].mean()
                        mean_size = result_n["cluster_quality"]["size"].mean()

                        # Get PCA variance for top n components
                        pca_var_cumsum = result_n.get("pca_variance_cumsum")
                        if pca_var_cumsum is not None and len(pca_var_cumsum) >= n:
                            pca_var = pca_var_cumsum[n - 1]  # Variance for first n components
                            f.write(
                                f"{n:7d} | {n_s:8d} | {ratio:5.2f} | {mean_stab:9.3f} | {mean_size:9.1f} | {pca_var:11.1%}\n"
                            )
                        else:
                            f.write(
                                f"{n:7d} | {n_s:8d} | {ratio:5.2f} | {mean_stab:9.3f} | {mean_size:9.1f} | N/A\n"
                            )

                    # Add cluster size distribution for optimal configuration
                    f.write("\n# Cluster Size Distribution (optimal n_components):\n")
                    f.write(f"# Expected cluster size: {args.n_runs} (one component per ICA run)\n")
                    cluster_sizes = cluster_quality["size"]
                    unique_sizes = sorted(np.unique(cluster_sizes))
                    f.write("# Size | Count | Percent\n")
                    f.write("# -----|-------|--------\n")
                    for size in unique_sizes:
                        count = (cluster_sizes == size).sum()
                        percent = 100.0 * count / len(cluster_sizes)
                        f.write(f"# {size:4.0f} | {count:5d} | {percent:6.1f}%\n")

                if args.verb >= 1:
                    print(f"  Saved selection summary: {summary_file}")

                # Save similarity matrix for visualization
                similarity_file = output_path.parent / f"{output_path.name}_icasso_similarity.npy"
                np.save(similarity_file, optimal_results["similarity"])
                if args.verb >= 1:
                    print(f"  Saved similarity matrix: {similarity_file}")
                    print(f"    Shape: {optimal_results['similarity'].shape}")
                    print(
                        "    This matrix shows pairwise similarity between all component instances"
                    )
                    print("    Can be used to create GIFT-style component matching visualizations")

            except Exception as e:
                print(f"Error in ICASSO automatic selection: {e}", file=sys.stderr)
                import traceback

                traceback.print_exc()
                return 1

        else:
            # Fixed number of components
            if args.verb >= 1:
                print(f"  n_components: {icasso_n_components}")
                print(f"  n_runs: {args.n_runs}")
                print(f"  min_stability: {args.min_stability}")

            try:
                icasso_results = icasso(
                    data,
                    n_components=icasso_n_components,
                    n_runs=args.n_runs,
                    pca_components=pca_for_icasso,
                    min_stability=args.min_stability,
                    device=device,
                    verbose=args.verb >= 1,
                    batch_size=None,  # Auto-select based on size
                )

                stable_components = icasso_results["components"]
                stable_mixing = icasso_results["mixing"]
                stability_scores = icasso_results["stability"]
                n_stable = icasso_results["n_stable"]
                all_stability = icasso_results["all_stability"]
                n_components_total = icasso_results["n_components"]
                cluster_quality = icasso_results["cluster_quality"]

                if args.verb >= 1:
                    print(f"\nStable components: {n_stable}/{n_components_total}")
                    print(
                        f"  Stability range: {all_stability.min():.3f} - {all_stability.max():.3f}"
                    )
                    print(f"  Threshold: {args.min_stability}")
                    print("\nCluster quality:")
                    print(f"  Mean compactness: {cluster_quality['compactness'].mean():.3f}")
                    print(
                        f"  Mean cluster size: {cluster_quality['size'].mean():.1f} (expected: {args.n_runs})"
                    )
                    print(
                        f"  Size range: {cluster_quality['size'].min()}-{cluster_quality['size'].max()}"
                    )

                    # Report PCA variance
                    pca_var = icasso_results.get("pca_variance_cumsum")
                    if pca_var is not None:
                        print("\nPCA preprocessing:")
                        print(
                            f"  {n_components_total} components explain {pca_var[n_components_total - 1]:.1%} variance"
                        )

                # Always save ALL components - user wants to see all from this run
                print(f"\nSaving ALL {n_components_total} components")
                if n_stable == 0:
                    print(f"  NOTE: None met stability threshold {args.min_stability}")
                    print(f"         Mean stability: {all_stability.mean():.3f}")
                    print(f"         Best stability: {all_stability.max():.3f}")
                    print("         Consider:")
                    print("           - Lowering -min_stability threshold")
                    print(f"           - Increasing -n_runs (currently {args.n_runs})")
                    print("           - Using fewer ICA components")
                else:
                    print(f"  {n_stable}/{n_components_total} met stability threshold")
                print("  Full stability report saved for component selection")

                # Get ALL components
                all_centroids = icasso_results["all_centroids"]
                stable_components = all_centroids  # ALL components

                # Get mixing for ALL components (matched to centroids)
                stable_mixing = icasso_results["all_mixing"]

                stability_scores = all_stability  # All stability scores
                n_stable = n_components_total  # Save all

            except Exception as e:
                print(f"Error running ICASSO: {e}", file=sys.stderr)
                import traceback

                traceback.print_exc()
                return 1

        # Note: Variance explained computation not meaningful for ICASSO centroids
        # ICASSO components are cluster centroids (averaged spatial patterns), not
        # the original ICA components that can reconstruct data. Use PCA variance instead.
        if args.verb >= 1:
            print("\n  Skipping variance explained computation for ICASSO")
            print("    (ICASSO centroids don't reconstruct original data)")
            print(f"    Use PCA variance above: {n_stable} components explain X% of variance")

        # Create empty variance array for saving
        var_ratio_np = np.zeros(n_stable)

        # Save ICASSO results
        if args.verb >= 1:
            print("\n  Saving ICASSO results...")

        icasso_labels = [f"ICASSO_{i:03d}" for i in range(n_stable)]

        try:
            with spinner("Writing ICASSO results"):
                icasso_files = save_decomposition_results(
                    components=stable_components,
                    timeseries=stable_mixing,
                    mask_file=args.mask,
                    output_prefix=f"{args.output}_icasso",
                    reference_file=args.input,
                    labels=icasso_labels,
                    method="ICASSO",
                    nii_ext=_nii_ext,
                )

            # Determine how many actually met threshold
            n_met_threshold = (all_stability >= args.min_stability).sum()

            # Save stability scores (for saved components only)
            stability_file = output_path.parent / f"{output_path.name}_icasso_stability.1D"
            np.savetxt(
                stability_file,
                stability_scores,
                fmt="%.4f",
                header=f"ICASSO stability scores (0-1, higher = more stable)\n{n_stable} components saved (top by stability)",
            )

            # Save comprehensive stability report (ALL components, sorted)
            stability_report = output_path.parent / f"{output_path.name}_icasso_stability_full.txt"
            with open(stability_report, "w") as f:
                f.write("# ICASSO Stability Report - ALL Components\n")
                f.write(f"# Total components tested: {n_components_total}\n")
                f.write(f"# Components saved: {n_stable}\n")
                f.write(
                    f"# Components meeting threshold ({args.min_stability}): {n_met_threshold}\n"
                )
                f.write(f"# Runs: {args.n_runs}\n")
                f.write(f"# PCA: {pca_for_icasso}\n")
                f.write("#\n")
                if n_met_threshold == 0:
                    f.write(
                        f"# Saved components are top {n_stable} by stability (none met threshold)\n"
                    )
                else:
                    f.write("# Saved components met stability threshold\n")
                f.write("#\n")
                f.write("# Component  Stability  Status\n")
                f.write("# ---------  ---------  ------\n")

                # Sort by stability (descending)
                sorted_idx = np.argsort(all_stability)[::-1]
                for idx in sorted_idx:
                    status = "STABLE" if all_stability[idx] >= args.min_stability else "unstable"
                    f.write(f"{idx:10d}  {all_stability[idx]:9.4f}  {status}\n")

                # Add cluster size distribution
                f.write("\n# Cluster Size Distribution:\n")
                f.write(f"# Expected cluster size: {args.n_runs} (one component per ICA run)\n")
                cluster_sizes = cluster_quality["size"]
                unique_sizes = sorted(np.unique(cluster_sizes))
                f.write("# Size | Count | Percent\n")
                f.write("# -----|-------|--------\n")
                for size in unique_sizes:
                    count = (cluster_sizes == size).sum()
                    percent = 100.0 * count / len(cluster_sizes)
                    f.write(f"# {size:4.0f} | {count:5d} | {percent:6.1f}%\n")

                # Add PCA variance information
                pca_var = icasso_results.get("pca_variance_cumsum")
                if pca_var is not None:
                    f.write("\n# PCA Preprocessing:\n")
                    f.write(
                        f"# {n_components_total} components explain {pca_var[n_components_total - 1]:.1%} variance\n"
                    )

            # Save variance explained
            variance_file = output_path.parent / f"{output_path.name}_icasso_variance.1D"
            if len(var_ratio_np) > 0:
                np.savetxt(
                    variance_file,
                    var_ratio_np,
                    fmt="%.6f",
                    header=f"Variance explained by each ICASSO component\nTotal: {var_ratio_np.sum():.4f}",
                )
            else:
                # Empty file with explanation
                with open(variance_file, "w") as f:
                    f.write("# No components saved - all had stability < threshold\n")

            # Save similarity matrix for visualization (fixed mode only - auto mode already saved it)
            if "similarity" in icasso_results:
                similarity_file = output_path.parent / f"{output_path.name}_icasso_similarity.npy"
                np.save(similarity_file, icasso_results["similarity"])

                if args.verb >= 1:
                    print(f"    Similarity matrix: {similarity_file}")
                    print(f"      Shape: {icasso_results['similarity'].shape}")
                    print(
                        "      This matrix shows pairwise similarity between all component instances"
                    )
                    print(
                        "      Can be used to create GIFT-style component matching visualizations"
                    )

            if args.verb >= 1:
                print(f"    Maps: {icasso_files['maps']}")
                print(f"    Timeseries: {icasso_files['timeseries_1D']}")
                print(f"    Stability (saved components): {stability_file}")
                print(f"    Stability (all components): {stability_report}")
                print(f"    Variance: {variance_file}")

        except Exception as e:
            print(f"Error saving ICASSO results: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            return 1

    if args.verb >= 1:
        print(f"\n{'=' * 60}")
        print("✓ Decomposition complete!")
    else:
        print(f"Decomposition complete. Output: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
