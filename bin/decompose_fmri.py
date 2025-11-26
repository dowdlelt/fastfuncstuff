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

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fastfuncsim.afni_io import get_tr_from_file, load_fmri_data
from fastfuncsim.decomposition_io import save_decomposition_results
from fastfuncsim.ica import (
    FastICA,
    ica_stability_analysis,
    select_n_components_by_stability,
)
from fastfuncsim.icasso import icasso, icasso_auto_select
from fastfuncsim.pca import PCA, explained_variance_analysis
from fastfuncsim.utils import get_device


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='PCA and ICA decomposition for fMRI data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input data
    parser.add_argument(
        '-input',
        required=True,
        metavar='FILE',
        help='Input fMRI data (3D or 4D NIfTI)',
    )

    parser.add_argument(
        '-mask',
        required=True,
        metavar='FILE',
        help='Brain mask (3D NIfTI, values > 0 define brain voxels)',
    )

    # PCA options
    parser.add_argument(
        '-pca',
        metavar='N',
        help='Perform PCA. N can be: '
             'float (0-1) for variance % (e.g., 0.85 = 85%%), '
             'int for number of components, '
             'or "all" for all components',
    )

    # ICA options
    parser.add_argument(
        '-ica',
        metavar='N',
        help='Perform ICA. N can be: '
             'int for number of components, '
             'or "auto" for automatic selection (requires -ica_range)',
    )

    parser.add_argument(
        '-icasso',
        metavar='N',
        help='Perform ICASSO (ICA with clustering for stability). N can be: '
             'int for number of components, '
             'or "auto" for automatic selection (requires -ica_range). '
             'Recommended for reliable component selection.',
    )

    parser.add_argument(
        '-ica_range',
        nargs=3,
        type=int,
        metavar=('START', 'STOP', 'STEP'),
        help='Range for automatic component selection (requires -ica auto or -icasso auto)',
    )

    parser.add_argument(
        '-n_runs',
        type=int,
        default=100,
        metavar='N',
        help='Number of ICA runs for ICASSO clustering (default: 100)',
    )

    parser.add_argument(
        '-stability',
        type=int,
        metavar='N',
        help='Run ICA N times with different seeds for stability analysis (non-ICASSO mode)',
    )

    parser.add_argument(
        '-min_stability',
        type=float,
        default=0.7,
        metavar='THRESHOLD',
        help='Minimum stability threshold for component selection (default: 0.7)',
    )

    # Output
    parser.add_argument(
        '-output',
        required=True,
        metavar='PREFIX',
        help='Output prefix for files (e.g., results/decomp)',
    )

    # Advanced options
    parser.add_argument(
        '-fun',
        default='logcosh',
        choices=['logcosh', 'exp', 'cube'],
        help='ICA nonlinearity function (default: logcosh)',
    )

    parser.add_argument(
        '-max_iter',
        type=int,
        default=200,
        metavar='N',
        help='Maximum ICA iterations (default: 200)',
    )

    parser.add_argument(
        '-seed',
        type=int,
        metavar='N',
        help='Random seed for reproducibility',
    )

    parser.add_argument(
        '-cpu',
        action='store_true',
        help='Force CPU computation (default: use GPU if available)',
    )

    parser.add_argument(
        '-verbose',
        action='store_true',
        help='Print detailed progress information',
    )

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

    if arg_str == 'all':
        return None
    elif arg_str == 'auto':
        return 'auto'

    try:
        # Try parsing as int
        value = int(arg_str)
        return value
    except ValueError:
        pass

    try:
        # Try parsing as float
        value = float(arg_str)
        if not 0.0 < value < 1.0:
            raise ValueError(
                f"Float component specification must be between 0 and 1, got {value}"
            )
        return value
    except ValueError:
        raise ValueError(
            f"Invalid component specification: '{arg_str}'. "
            f"Use int (number), float (variance fraction), 'all', or 'auto'"
        )


def main():
    """Main CLI entry point"""
    args = parse_args()

    # Setup device
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = get_device()

    if args.verbose:
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
        print("Error: Cannot specify both -ica and -icasso. Use -icasso for stability analysis.", file=sys.stderr)
        return 1

    if ica_n_components == 'auto' and args.ica_range is None:
        print("Error: -ica auto requires -ica_range START STOP STEP", file=sys.stderr)
        return 1

    if icasso_n_components == 'auto' and args.ica_range is None:
        print("Error: -icasso auto requires -ica_range START STOP STEP", file=sys.stderr)
        return 1

    if args.stability and ica_n_components is None:
        print("Warning: -stability specified but -ica not requested. Ignoring stability analysis.",
              file=sys.stderr)
        args.stability = None

    # Load data
    if args.verbose:
        print(f"\nLoading data from: {args.input}")
        print(f"Loading mask from: {args.mask}")

    try:
        data = load_fmri_data(args.input, args.mask)
        tr = get_tr_from_file(args.input)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"  Data shape: {data.shape}")
        print(f"  TR: {tr}s")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ========== PCA ==========
    if pca_n_components is not None:
        if args.verbose:
            print(f"\n{'='*60}")
            print("Running PCA...")
            print(f"  n_components: {pca_n_components}")

        pca = PCA(n_components=pca_n_components, device=device)
        pca_scores = pca.fit_transform(data)

        if args.verbose:
            print(f"  Kept {pca.n_components_} components")
            print(f"  Explained variance: {pca.explained_variance_ratio_.sum().item():.3f}")
            print(f"  Output shape: {pca_scores.shape}")

        # Save PCA results
        if args.verbose:
            print("  Saving PCA results...")

        pca_labels = [f'PC_{i:03d}' for i in range(pca.n_components_)]

        try:
            pca_files = save_decomposition_results(
                components=pca.components_,
                timeseries=pca_scores,
                mask_file=args.mask,
                output_prefix=f"{args.output}_pca",
                reference_file=args.input,
                labels=pca_labels,
                method='PCA',
            )

            if args.verbose:
                print(f"    Maps: {pca_files['maps']}")
                print(f"    Timeseries: {pca_files['timeseries_1D']}")

        except Exception as e:
            print(f"Error saving PCA results: {e}", file=sys.stderr)
            return 1

    # ========== ICA ==========
    if ica_n_components is not None:
        if args.verbose:
            print(f"\n{'='*60}")
            print("Running ICA...")

        # Determine PCA preprocessing
        if pca_n_components is not None:
            # Use PCA components from above
            pca_for_ica = pca_n_components
            if args.verbose:
                print(f"  Using PCA reduction: {pca_for_ica}")
        else:
            # Default: keep 85% variance
            pca_for_ica = 0.85
            if args.verbose:
                print(f"  Using default PCA reduction: 85% variance")

        # Auto component selection
        if ica_n_components == 'auto':
            if args.verbose:
                print("  Automatic component selection enabled")
                print(f"    Range: {args.ica_range[0]} to {args.ica_range[1]} (step {args.ica_range[2]})")
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
                    verbose=args.verbose,
                )

                ica_n_components = auto_results['optimal_n_components']

                if args.verbose:
                    print(f"\n  Optimal n_components: {ica_n_components}")

                # Save stability results
                stability_file = output_path.parent / f"{output_path.name}_stability.txt"
                with open(stability_file, 'w') as f:
                    f.write("# ICA automatic component selection\n")
                    f.write(f"# Optimal n_components: {ica_n_components}\n")
                    f.write("# n_components mean_stability\n")
                    for n_comp, stability in sorted(auto_results['stability_by_n_components'].items()):
                        f.write(f"{n_comp} {stability:.4f}\n")

                if args.verbose:
                    print(f"  Saved stability analysis: {stability_file}")

            except Exception as e:
                print(f"Error in automatic component selection: {e}", file=sys.stderr)
                return 1

        # Run ICA
        if args.verbose:
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

            if args.verbose:
                print(f"  Converged in {ica.n_iter_} iterations")
                print(f"  Output shape: {ica_timeseries.shape}")

        except Exception as e:
            print(f"Error running ICA: {e}", file=sys.stderr)
            return 1

        # Stability analysis
        if args.stability and ica_n_components != 'auto':
            if args.verbose:
                print(f"\n  Running stability analysis ({args.stability} runs)...")

            try:
                stability_results = ica_stability_analysis(
                    data,
                    n_components=ica_n_components,
                    pca_components=pca_for_ica,
                    n_runs=args.stability,
                    device=device,
                    verbose=args.verbose,
                )

                mean_stability = stability_results['stability_scores'].mean()
                n_stable = (stability_results['stability_scores'] > args.min_stability).sum()

                if args.verbose:
                    print(f"    Mean stability: {mean_stability:.3f}")
                    print(f"    Components with stability > {args.min_stability}: {n_stable}/{ica_n_components}")

                # Save stability scores
                stability_file = output_path.parent / f"{output_path.name}_ica_stability.1D"
                np.savetxt(
                    stability_file,
                    stability_results['stability_scores'],
                    fmt='%.4f',
                    header='Component stability scores (0-1, higher = more stable)',
                )

                if args.verbose:
                    print(f"    Saved stability scores: {stability_file}")

            except Exception as e:
                print(f"Warning: Stability analysis failed: {e}", file=sys.stderr)

        # Save ICA results
        if args.verbose:
            print("  Saving ICA results...")

        ica_labels = [f'IC_{i:03d}' for i in range(ica.components_.shape[0])]

        try:
            ica_files = save_decomposition_results(
                components=ica.components_,
                timeseries=ica_timeseries,
                mask_file=args.mask,
                output_prefix=f"{args.output}_ica",
                reference_file=args.input,
                labels=ica_labels,
                method='ICA',
            )

            if args.verbose:
                print(f"    Maps: {ica_files['maps']}")
                print(f"    Timeseries: {ica_files['timeseries_1D']}")

        except Exception as e:
            print(f"Error saving ICA results: {e}", file=sys.stderr)
            return 1

    # ========== ICASSO ==========
    if icasso_n_components is not None:
        if args.verbose:
            print(f"\n{'='*60}")
            print("Running ICASSO...")

        # Determine PCA preprocessing
        if pca_n_components is not None:
            pca_for_icasso = pca_n_components
            if args.verbose:
                print(f"  Using PCA reduction: {pca_for_icasso}")
        else:
            pca_for_icasso = 0.85
            if args.verbose:
                print(f"  Using default PCA reduction: 85% variance")

        # Auto component selection
        if icasso_n_components == 'auto':
            if args.verbose:
                print("  Automatic component selection via ICASSO")
                print(f"    Range: {args.ica_range[0]} to {args.ica_range[1]} (step {args.ica_range[2]})")
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
                    verbose=args.verbose,
                )

                # Extract stable components
                stable_components = icasso_results['optimal_results']['components']
                stable_mixing = icasso_results['optimal_results']['mixing']
                stability_scores = icasso_results['optimal_results']['stability']
                n_stable = icasso_results['optimal_results']['n_stable']

                if args.verbose:
                    print(f"\nOptimal configuration:")
                    print(f"  Requested: {icasso_results['optimal_n_components']} components")
                    print(f"  Stable: {n_stable} components")

                # Save component selection summary
                summary_file = output_path.parent / f"{output_path.name}_icasso_selection.txt"
                with open(summary_file, 'w') as f:
                    f.write("# ICASSO Automatic Component Selection\n")
                    f.write(f"# Optimal n_components: {icasso_results['optimal_n_components']}\n")
                    f.write(f"# Stable components found: {n_stable}\n")
                    f.write("# n_components n_stable ratio\n")
                    for n in sorted(icasso_results['n_stable_by_n_components'].keys()):
                        n_s = icasso_results['n_stable_by_n_components'][n]
                        ratio = icasso_results['stability_ratios'][n]
                        f.write(f"{n:12d} {n_s:8d} {ratio:5.2f}\n")

                if args.verbose:
                    print(f"  Saved selection summary: {summary_file}")

            except Exception as e:
                print(f"Error in ICASSO automatic selection: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                return 1

        else:
            # Fixed number of components
            if args.verbose:
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
                    verbose=args.verbose,
                )

                stable_components = icasso_results['components']
                stable_mixing = icasso_results['mixing']
                stability_scores = icasso_results['stability']
                n_stable = icasso_results['n_stable']

                if args.verbose:
                    print(f"\nStable components: {n_stable}/{icasso_n_components}")

            except Exception as e:
                print(f"Error running ICASSO: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                return 1

        # Compute variance explained for stable components
        if args.verbose:
            print("\n  Computing variance explained...")

        # Create temporary ICA object to use variance computation
        temp_ica = FastICA(device=device)
        temp_ica.components_ = torch.tensor(stable_components, device=device)
        temp_ica.mixing_ = torch.tensor(stable_mixing, device=device)
        _, var_ratio = temp_ica.compute_variance_explained(data)

        var_ratio_np = var_ratio.cpu().numpy()

        if args.verbose:
            total_var = var_ratio_np.sum()
            print(f"    Total variance explained: {total_var:.1%}")
            print(f"    Per component: {var_ratio_np.mean():.2%} ± {var_ratio_np.std():.2%}")

        # Save ICASSO results
        if args.verbose:
            print("\n  Saving ICASSO results...")

        icasso_labels = [f'ICASSO_{i:03d}' for i in range(n_stable)]

        try:
            icasso_files = save_decomposition_results(
                components=stable_components,
                timeseries=stable_mixing,
                mask_file=args.mask,
                output_prefix=f"{args.output}_icasso",
                reference_file=args.input,
                labels=icasso_labels,
                method='ICASSO',
            )

            # Save stability scores
            stability_file = output_path.parent / f"{output_path.name}_icasso_stability.1D"
            np.savetxt(
                stability_file,
                stability_scores,
                fmt='%.4f',
                header=f'ICASSO stability scores (0-1, higher = more stable)\n{n_stable} stable components selected',
            )

            # Save variance explained
            variance_file = output_path.parent / f"{output_path.name}_icasso_variance.1D"
            np.savetxt(
                variance_file,
                var_ratio_np,
                fmt='%.6f',
                header=f'Variance explained by each ICASSO component\nTotal: {var_ratio_np.sum():.4f}',
            )

            if args.verbose:
                print(f"    Maps: {icasso_files['maps']}")
                print(f"    Timeseries: {icasso_files['timeseries_1D']}")
                print(f"    Stability: {stability_file}")
                print(f"    Variance: {variance_file}")

        except Exception as e:
            print(f"Error saving ICASSO results: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1

    if args.verbose:
        print(f"\n{'='*60}")
        print("✓ Decomposition complete!")
    else:
        print(f"Decomposition complete. Output: {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
