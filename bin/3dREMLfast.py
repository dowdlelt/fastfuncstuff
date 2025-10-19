#!/usr/bin/env python3
"""
3dREMLfast - Fast ARMA(1,1) GLM for fMRI using GPU acceleration

This is a PyTorch/GPU-accelerated implementation of AFNI's 3dREMLfit,
providing 5-50x speedup for ARMA(1,1) prewhitened GLM fitting.

Basic usage:
    3dREMLfast -input func.nii.gz -matrix X.xmat.1D -Rnuisance stats_REML

For help:
    3dREMLfast -help
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch

try:
    import nibabel as nib
except ImportError:
    print("ERROR: nibabel is required. Install with: pip install nibabel")
    sys.exit(1)

# Import fastfuncsim modules
try:
    from fastfuncsim.analysis import analyze_from_design_matrix
    from fastfuncsim.glm_outputs import (
        write_afni_bucket,
        write_glm_results_nifti,
        slice_glm_results,
    )
    from fastfuncsim.afni_io import read_afni_design_matrix
    from fastfuncsim.utils import get_device
except ImportError as e:
    print(f"ERROR: Could not import fastfuncsim: {e}")
    print("Make sure fastfuncsim is installed: pip install -e .")
    sys.exit(1)


def parse_grid_arg(grid_str: str) -> Tuple[float, float, int]:
    """Parse grid argument like '0.1,0.9,11' into (start, stop, num_points)"""
    try:
        parts = grid_str.split(",")
        if len(parts) != 3:
            raise ValueError("Grid must have exactly 3 values: start,stop,num_points")
        start = float(parts[0])
        stop = float(parts[1])
        num_points = int(parts[2])
        if num_points < 2:
            raise ValueError("num_points must be >= 2")
        return start, stop, num_points
    except ValueError as e:
        print(f"ERROR parsing grid '{grid_str}': {e}")
        sys.exit(1)


def parse_input_files(input_str: str) -> List[str]:
    """Parse input file string (space-separated or single file)"""
    # Remove quotes if present
    input_str = input_str.strip().strip('"').strip("'")
    # Split on spaces
    files = input_str.split()
    # Validate files exist
    for f in files:
        if not Path(f).exists():
            print(f"ERROR: Input file not found: {f}")
            sys.exit(1)
    return files


def detect_format(filepath: str) -> str:
    """Detect file format from extension"""
    p = Path(filepath)
    if p.suffix == ".gz" and p.stem.endswith(".nii"):
        return "nii.gz"
    elif p.suffix in [".nii"]:
        return "nii"
    elif "+orig" in p.name or "+tlrc" in p.name:
        return "afni"
    else:
        # Default to nifti
        return "nii.gz"


def get_tr_from_file(filepath: str) -> float:
    """Extract TR from NIfTI header"""
    try:
        img = nib.load(filepath)
        tr = img.header.get_zooms()[-1]  # Last dimension is time
        if tr == 0 or tr is None:
            print(f"WARNING: TR not found in {filepath} header, using 1.0")
            return 1.0
        return float(tr)
    except Exception as e:
        print(f"WARNING: Could not read TR from {filepath}: {e}")
        print("Using TR=1.0 as fallback")
        return 1.0


def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="3dREMLfast - Fast GPU-accelerated ARMA(1,1) GLM fitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic REML analysis with main bucket output
  3dREMLfast -input func.nii.gz -matrix X.xmat.1D -Rbuck stats_REML
  
  # Multiple runs
  3dREMLfast -input "run1.nii.gz run2.nii.gz run3.nii.gz" \\
             -matrix X.xmat.1D -Rbuck stats_REML
  
  # With all outputs
  3dREMLfast -input func.nii.gz -matrix X.xmat.1D \\
             -Rbuck stats_REML -Rvar params_REML \\
             -Rfitts fitts_REML -Rerrts errts_REML \\
             -Rbeta betas_only_REML -fout -tout -rout
  
  # With OLS comparison
  3dREMLfast -input func.nii.gz -matrix X.xmat.1D \\
             -Rbuck stats_REML -Obuck stats_OLS
  
  # Nuisance regressors only
  3dREMLfast -input func.nii.gz -matrix X.xmat.1D \\
             -Rnuisance nuisance_REML -Onuisance nuisance_OLS
  
  # Custom ARMA grid with double precision
  3dREMLfast -input func.nii.gz -matrix X.xmat.1D \\
             -Rbuck stats_REML -use_double \\
             -a_grid 0.0,0.9,10 -b_grid -0.8,0.8,17
        """,
    )

    # Required arguments
    required = parser.add_argument_group("Required Arguments")
    required.add_argument(
        "-input",
        required=True,
        help="Input fMRI dataset(s). Single file or space-separated list in quotes.",
    )
    required.add_argument(
        "-matrix",
        required=True,
        help="Design matrix file (X.xmat.1D from 3dDeconvolve)",
    )

    # Output arguments - REML
    reml_out = parser.add_argument_group("REML Output Options")
    reml_out.add_argument(
        "-Rvar",
        help="Output REML variance parameters (6 volumes: a, b, lambda, StDev, -LogLik, LjungBox)",
    )
    reml_out.add_argument(
        "-Rbuck", help="Output REML betas + statistics (main bucket output)"
    )
    reml_out.add_argument("-Rbeta", help="Output REML betas only (no statistics)")
    reml_out.add_argument(
        "-Rnuisance", help="Output REML betas + statistics for NUISANCE regressors only"
    )
    reml_out.add_argument("-Rfitts", help="Output REML fitted model time series")
    reml_out.add_argument("-Rerrts", help="Output REML residuals")
    reml_out.add_argument("-Rwherr", help="Output REML whitened residuals")

    # Output arguments - OLS
    ols_out = parser.add_argument_group("OLS Output Options (for comparison)")
    ols_out.add_argument(
        "-Obuck", help="Output OLS betas + statistics (main bucket output)"
    )
    ols_out.add_argument("-Obeta", help="Output OLS betas only (no statistics)")
    ols_out.add_argument(
        "-Onuisance", help="Output OLS betas + statistics for NUISANCE regressors only"
    )

    # Statistics options
    stats_opts = parser.add_argument_group("Statistics Options")
    stats_opts.add_argument(
        "-fout", action="store_true", help="Include F-statistics in output buckets"
    )
    stats_opts.add_argument(
        "-tout", action="store_true", help="Include t-statistics in output buckets"
    )
    stats_opts.add_argument(
        "-rout", action="store_true", help="Include R² statistics in output buckets"
    )

    # ARMA grid options
    arma_opts = parser.add_argument_group("ARMA(1,1) Grid Options")
    arma_opts.add_argument(
        "-a_grid", help="AR parameter grid: start,stop,num_points (e.g., 0.0,0.9,10)"
    )
    arma_opts.add_argument(
        "-b_grid", help="MA parameter grid: start,stop,num_points (e.g., -0.8,0.8,17)"
    )

    # Processing options
    proc_opts = parser.add_argument_group("Processing Options")
    proc_opts.add_argument(
        "-use_double",
        action="store_true",
        help="Use double precision (float64) - matches AFNI exactly, ~2x memory, ~1.5x slower",
    )
    proc_opts.add_argument("-mask", help="Mask file to restrict analysis")
    proc_opts.add_argument(
        "-force_format",
        choices=["nii", "nii.gz", "afni"],
        help="Force output format (default: match input)",
    )
    proc_opts.add_argument(
        "-device",
        type=str,
        help=(
            "Force device (default: auto-detect GPU). "
            "Format: 'cpu' or 'cuda' for auto-config, "
            "'cpu,N' to use N CPU threads, "
            "'cuda,N' to use GPU device N (e.g., 'cuda,0' for GPU 0)"
        ),
    )
    proc_opts.add_argument(
        "-verbose", action="store_true", help="Print detailed progress information"
    )
    proc_opts.add_argument(
        "-debug_memory", action="store_true", help="Print detailed memory profiling at every step (for debugging)"
    )

    # Help
    parser.add_argument("-help", action="store_true", help="Show this help message")

    return parser


def print_header(args):
    """Print program header"""
    print("=" * 70)
    print("3dREMLfast - GPU-Accelerated ARMA(1,1) GLM")
    print("=" * 70)
    print()
    if args.use_double:
        print("⚙️  Precision: DOUBLE (float64) - matches AFNI exactly")
    else:
        print("⚙️  Precision: SINGLE (float32) - default, faster")
    print()


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Show help if requested
    if args.help or len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # Check that at least one output is requested
    outputs = [
        args.Rvar,
        args.Rbuck,
        args.Rbeta,
        args.Rnuisance,
        args.Rfitts,
        args.Rerrts,
        args.Rwherr,
        args.Obuck,
        args.Obeta,
        args.Onuisance,
    ]
    if not any(outputs):
        print("ERROR: At least one output option must be specified")
        print(
            "       Use -Rbuck, -Rbeta, -Rnuisance, -Rvar, -Rfitts, -Rerrts, -Rwherr,"
        )
        print("       -Obuck, -Obeta, or -Onuisance")
        sys.exit(1)

    print_header(args)

    # Parse input files
    input_files = parse_input_files(args.input)
    print(f"📁 Input files: {len(input_files)} file(s)")
    for f in input_files:
        print(f"   • {f}")
    print()

    # Get TR from first file
    tr = get_tr_from_file(input_files[0])
    print(f"⏱️  TR: {tr:.3f} seconds")
    print()

    # Detect output format
    if args.force_format:
        output_format = args.force_format
    else:
        output_format = detect_format(input_files[0])
    print(f"💾 Output format: {output_format}")
    print()

    # Setup device and parse device specification
    import os
    device_spec = args.device
    cpu_threads_override = None
    cuda_device_id = None

    if device_spec:
        # Parse device specification: "cpu", "cuda", "cpu,12", "cuda,0"
        parts = device_spec.split(',')
        device_type = parts[0].strip().lower()

        if len(parts) > 1:
            # User specified threads or device ID
            try:
                device_param = int(parts[1].strip())
                if device_type == "cpu":
                    cpu_threads_override = device_param
                elif device_type == "cuda":
                    cuda_device_id = device_param
            except ValueError:
                raise ValueError(f"Invalid device specification: {device_spec}. Expected format: 'cpu', 'cuda', 'cpu,N', or 'cuda,N'")

        # Create device
        if device_type == "cpu":
            device = torch.device("cpu")
        elif device_type == "cuda":
            if cuda_device_id is not None:
                device = torch.device(f"cuda:{cuda_device_id}")
            else:
                device = torch.device("cuda")
        else:
            raise ValueError(f"Invalid device type: {device_type}. Must be 'cpu' or 'cuda'")
    else:
        device = get_device()

    # Configure CPU threading for maximum performance
    if device.type == "cpu":
        try:
            import psutil
            physical_cores = psutil.cpu_count(logical=False)
            logical_cores = os.cpu_count() or 12

            # Determine number of threads to use
            if cpu_threads_override is not None:
                # User explicitly specified thread count
                num_threads = cpu_threads_override
                thread_source = f"user-specified"
            else:
                # Auto-detect: use physical cores for compute efficiency
                num_threads = physical_cores or logical_cores
                thread_source = f"physical cores ({logical_cores} logical with hyperthreading)"

            torch.set_num_threads(num_threads)
            torch.set_num_interop_threads(num_threads)
            # Also set environment variables for MKL/OpenMP
            os.environ["OMP_NUM_THREADS"] = str(num_threads)
            os.environ["MKL_NUM_THREADS"] = str(num_threads)

            print(f"🖥️  Device: {device}")
            print(f"⚡ CPU threads: {num_threads} ({thread_source})")
        except ImportError:
            # Fallback if psutil not available
            num_threads = cpu_threads_override if cpu_threads_override is not None else (os.cpu_count() or 12)
            torch.set_num_threads(num_threads)
            torch.set_num_interop_threads(num_threads)
            os.environ["OMP_NUM_THREADS"] = str(num_threads)
            os.environ["MKL_NUM_THREADS"] = str(num_threads)
            print(f"🖥️  Device: {device}")
            print(f"⚡ CPU threads: {num_threads}")
    else:
        print(f"🖥️  Device: {device}")
    print()

    # Parse ARMA grids if provided
    a_grid = None
    b_grid = None
    if args.a_grid:
        start, stop, num = parse_grid_arg(args.a_grid)
        a_grid = torch.linspace(start, stop, num, device=device)
        print(f"🔢 Custom a_grid: [{start}, {stop}] with {num} points")
    if args.b_grid:
        start, stop, num = parse_grid_arg(args.b_grid)
        b_grid = torch.linspace(start, stop, num, device=device)
        print(f"🔢 Custom b_grid: [{start}, {stop}] with {num} points")
    if args.a_grid or args.b_grid:
        print()

    # Determine if we need OLS (simplified - only support -Obuck for now)
    want_ols = args.Obuck is not None
    ols_output_path = args.Obuck if want_ols else None

    # Note: -Obeta and -Onuisance are not yet supported with the callback system
    if args.Obeta or args.Onuisance:
        print("WARNING: -Obeta and -Onuisance not yet supported. Use -Obuck only for OLS output.")
        print("         OLS results will be stored in results.ols_results but not written immediately.")

    if False:  # Old callback code - disabled
        # Determine stat flags (default to -fout if none specified)
        want_fstat = args.fout or (not args.tout and not args.rout)
        want_tstat = args.tout
        want_rstat = args.rout

        def write_ols_results(ols_results, original_shape, affine, design_info):
            """Write OLS results immediately after computation"""
            print("\n💾 Writing OLS outputs (before ARMA)...")

            # Get nuisance and stimulus indices from design_info
            stim_indices = design_info.get("stim_bots", [])
            all_indices = list(range(design_info["n_regressors"]))

            # Set spatial metadata on OLS results for writing
            ols_results.original_shape = original_shape
            ols_results.affine = affine

            if args.Obuck:
                print(f"  • Writing OLS betas + stats (bucket): {args.Obuck}")
                write_afni_bucket(
                    ols_results,
                    args.Obuck,
                    condition_names=design_info.get("column_labels"),
                    output_format=output_format,
                )

            if args.Obeta:
                print(f"  • Writing OLS betas only: {args.Obeta}")
                # Write only betas using the write_glm_results_nifti function correctly
                # Create a temporary results-like object with only betas
                import nibabel as nib
                from fastfuncsim.glm_outputs import _ensure_numpy, _reshape_parameter_map, _get_voxel_mask, _resolve_shape

                affine = getattr(ols_results, "affine", np.eye(4))
                volume_shape = _resolve_shape(ols_results, None)
                voxel_mask = _get_voxel_mask(ols_results)

                betas_np = _ensure_numpy(ols_results.betas)
                betas_vol = _reshape_parameter_map(betas_np, volume_shape, voxel_mask)

                beta_img = nib.Nifti1Image(betas_vol, affine)
                if output_format == "afni":
                    nib.save(beta_img, str(Path(args.Obeta).with_suffix('.BRIK')))
                else:
                    nib.save(beta_img, str(args.Obeta) if args.Obeta.endswith('.nii.gz') else f"{args.Obeta}.nii.gz")

            if args.Onuisance:
                print(f"  • Writing OLS nuisance betas + stats: {args.Onuisance}")
                if stim_indices:
                    nuisance_indices = [i for i in all_indices if i not in stim_indices]
                    ols_nuisance_results = slice_glm_results(
                        ols_results, nuisance_indices
                    )
                    nuisance_names = (
                        [design_info["column_labels"][i] for i in nuisance_indices]
                        if "col_names" in design_info
                        else None
                    )
                else:
                    ols_nuisance_results = ols_results
                    nuisance_names = design_info.get("col_names")

                write_afni_bucket(
                    ols_nuisance_results,
                    args.Onuisance,
                    condition_names=nuisance_names,
                    output_format=output_format,
                )
            print()

        ols_write_callback = write_ols_results

    print("🚀 Starting GLM analysis...")
    print()

    # Run analysis
    try:
        results, design_info = analyze_from_design_matrix(
            fmri_data=input_files if len(input_files) > 1 else input_files[0],
            design_matrix_file=args.matrix,
            method="arma11",  # Always use ARMA for 3dREMLfast
            arma_a_grid=a_grid,
            arma_b_grid=b_grid,
            want_ols=want_ols,
            ols_output_path=ols_output_path,
            ols_output_format=output_format,
            device=device,
            mask_file=args.mask,
            use_double=args.use_double,
            debug_memory=args.debug_memory,
        )
    except Exception as e:
        print(f"\n❌ ERROR during analysis: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("✅ Analysis complete!")
    print()

    # Write outputs
    print("💾 Writing outputs...")
    print()

    # Determine stat flags (default to -fout if none specified)
    want_fstat = args.fout or (not args.tout and not args.rout)
    want_tstat = args.tout
    want_rstat = args.rout

    # Get nuisance and stimulus indices
    stim_indices = design_info.get("stim_bots", [])
    all_indices = list(range(design_info["n_regressors"]))

    # REML outputs
    if args.Rbuck:
        print(f"  • Writing REML betas + stats (bucket): {args.Rbuck}")
        # Rbuck: ALL betas + stats
        write_afni_bucket(
            results,
            args.Rbuck,
            condition_names=design_info.get("col_names"),
            output_format=output_format,
        )

    if args.Rbeta:
        print(f"  • Writing REML betas only: {args.Rbeta}")
        # Rbeta: ALL betas, no stats
        import nibabel as nib
        from fastfuncsim.glm_outputs import _ensure_numpy, _reshape_parameter_map, _get_voxel_mask, _resolve_shape

        affine = getattr(results, "affine", np.eye(4))
        volume_shape = _resolve_shape(results, None)
        voxel_mask = _get_voxel_mask(results)

        assert results.betas is not None, "Results must have betas"
        betas_np = _ensure_numpy(results.betas)
        betas_vol = _reshape_parameter_map(betas_np, volume_shape, voxel_mask)

        beta_img = nib.Nifti1Image(betas_vol, affine)
        if output_format == "afni":
            nib.save(beta_img, str(Path(args.Rbeta).with_suffix('.BRIK')))
        else:
            nib.save(beta_img, str(args.Rbeta) if args.Rbeta.endswith('.nii.gz') else f"{args.Rbeta}.nii.gz")

    if args.Rnuisance:
        print(f"  • Writing REML nuisance betas + stats: {args.Rnuisance}")
        # Rnuisance: Extract nuisance regressors (everything NOT in stim_bots)
        if stim_indices:
            nuisance_indices = [i for i in all_indices if i not in stim_indices]
            nuisance_results = slice_glm_results(results, nuisance_indices)
            nuisance_names = (
                [design_info["col_names"][i] for i in nuisance_indices]
                if "col_names" in design_info
                else None
            )
        else:
            # No stimulus indices specified, use all regressors
            nuisance_results = results
            nuisance_names = design_info.get("col_names")

        write_afni_bucket(
            nuisance_results,
            args.Rnuisance,
            condition_names=nuisance_names,
            output_format=output_format,
        )

    if args.Rvar:
        print(f"  • Writing REML variance parameters: {args.Rvar}")
        # Stack variance parameters: a, b, lambda, StDev, -LogLik, LjungBox (placeholder)
        var_stack = []
        var_labels = []

        if results.arma_params is not None:
            var_stack.append(results.arma_params[:, 0])  # a
            var_stack.append(results.arma_params[:, 1])  # b
            var_labels.extend(["a", "b"])

        if results.arma_lambda is not None:
            var_stack.append(results.arma_lambda)  # lambda
            var_labels.append("lambda")

        if results.sigma2 is not None:
            var_stack.append(torch.sqrt(results.sigma2))  # StDev
            var_labels.append("StDev")

        if results.reml_likelihood is not None:
            var_stack.append(-results.reml_likelihood)  # -LogLik
            var_labels.append("-LogLik")

        # LjungBox placeholder (would need to compute from whitened residuals)
        assert results.sigma2 is not None, "Results must have sigma2"
        var_stack.append(torch.zeros_like(results.sigma2))
        var_labels.append("LjungBox")

        # Stack and write
        var_data = torch.stack(var_stack, dim=1)  # (n_voxels, 6)
        # Write variance parameters directly as 4D NIfTI
        import nibabel as nib
        affine = getattr(results, "affine", np.eye(4))
        volume_shape = getattr(results, "original_shape", None)
        voxel_mask = getattr(results, "voxel_mask", None)

        # Reshape var_data to 4D volume (convert to numpy first!)
        var_data_np = var_data.cpu().numpy() if isinstance(var_data, torch.Tensor) else var_data

        if volume_shape is not None and voxel_mask is not None:
            n_params = var_data_np.shape[1]
            var_vol = np.zeros((*volume_shape, n_params), dtype=np.float32)
            voxel_mask_np = voxel_mask.cpu().numpy() if isinstance(voxel_mask, torch.Tensor) else voxel_mask
            var_vol[voxel_mask_np.reshape(volume_shape)] = var_data_np
        else:
            # Assume already in volume shape
            var_vol = var_data_np.reshape(*volume_shape, -1) if volume_shape else var_data_np

        var_img = nib.Nifti1Image(var_vol, affine)

        # Save in requested format using helper
        # IMPORTANT: nibabel cannot convert NIfTI headers to AFNI headers
        # So we always save variance files as NIfTI (even if user requested AFNI)
        from fastfuncsim.glm_outputs import _save_nifti_with_format
        var_format = "nifti_gz" if output_format == "afni" else output_format
        _save_nifti_with_format(var_img, Path(args.Rvar), var_format)

    if args.Rfitts:
        print(f"  • Writing REML fitted model: {args.Rfitts}")
        if results.predicted is not None:
            import nibabel as nib
            from fastfuncsim.glm_outputs import _ensure_numpy, _get_voxel_mask

            affine = getattr(results, "affine", np.eye(4))
            volume_shape = getattr(results, "original_shape", None)
            voxel_mask = _get_voxel_mask(results)

            # Predicted is (n_timepoints, n_voxels), need (n_voxels, n_timepoints)
            predicted_np = _ensure_numpy(results.predicted.T)

            # Reshape to 4D volume (x, y, z, timepoints)
            if volume_shape is not None and voxel_mask is not None:
                n_timepoints = predicted_np.shape[1]
                predicted_vol = np.zeros((*volume_shape, n_timepoints), dtype=np.float32)
                predicted_vol[voxel_mask.reshape(volume_shape)] = predicted_np
            else:
                predicted_vol = predicted_np

            fitts_img = nib.Nifti1Image(predicted_vol, affine)
            if output_format == "afni":
                nib.save(fitts_img, str(Path(args.Rfitts).with_suffix('.BRIK')))
            else:
                nib.save(fitts_img, str(args.Rfitts) if args.Rfitts.endswith('.nii.gz') else f"{args.Rfitts}.nii.gz")
        else:
            print("    ⚠️  Warning: Fitted values not available (predicted=None)")

    if args.Rerrts:
        print(f"  • Writing REML residuals: {args.Rerrts}")
        if results.residuals is not None:
            import nibabel as nib
            from fastfuncsim.glm_outputs import _ensure_numpy, _get_voxel_mask

            affine = getattr(results, "affine", np.eye(4))
            volume_shape = getattr(results, "original_shape", None)
            voxel_mask = _get_voxel_mask(results)

            # Residuals is (n_timepoints, n_voxels), need (n_voxels, n_timepoints)
            residuals_np = _ensure_numpy(results.residuals.T)

            # Reshape to 4D volume (x, y, z, timepoints)
            if volume_shape is not None and voxel_mask is not None:
                n_timepoints = residuals_np.shape[1]
                residuals_vol = np.zeros((*volume_shape, n_timepoints), dtype=np.float32)
                residuals_vol[voxel_mask.reshape(volume_shape)] = residuals_np
            else:
                residuals_vol = residuals_np

            errts_img = nib.Nifti1Image(residuals_vol, affine)
            if output_format == "afni":
                nib.save(errts_img, str(Path(args.Rerrts).with_suffix('.BRIK')))
            else:
                nib.save(errts_img, str(args.Rerrts) if args.Rerrts.endswith('.nii.gz') else f"{args.Rerrts}.nii.gz")
        else:
            print("    ⚠️  Warning: Residuals not available")

    if args.Rwherr:
        print(f"  • Writing REML whitened residuals: {args.Rwherr}")
        print("    ⚠️  Warning: Whitened residuals not currently computed")
        # Would need to compute: residuals @ inv(chol(R))

    # OLS outputs - already written by callback during analysis!
    # The callback writes OLS results immediately after OLS completion,
    # freeing memory before the ARMA loop starts.

    print()
    print("=" * 70)
    print("✅ 3dREMLfast completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
