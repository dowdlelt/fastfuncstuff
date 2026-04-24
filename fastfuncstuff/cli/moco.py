"""CLI for GPU-accelerated motion correction (ffs_moco).

Command: ffs_moco (registered as entry point in pyproject.toml)

Usage:
    ffs_moco -input epi.nii.gz -prefix epi_mc.nii.gz -1Dfile motion.1D
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from fastfuncstuff.processing.affine import save_matrix_1D
from fastfuncstuff.processing.ffs_moco import (
    MocoConfig,
    moco,
    save_maxdisp_1D,
    save_moco_1D,
    save_moco_dfile,
)
from fastfuncstuff.processing.io import derive_mean_output_path, load_image, save_image
from fastfuncstuff.cli_utils import add_verbose_arg, parse_prefix


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="ffs_moco",
        description="GPU-accelerated motion correction for fMRI/fNIRS timeseries "
        "(inspired by 3dvolreg)",
        epilog="""Examples:
  # Standard motion correction:
  ffs_moco -input epi.nii.gz -prefix epi_mc.nii.gz -1Dfile motion.1D

  # With affine matrix output:
  ffs_moco -input epi.nii.gz -prefix epi_mc.nii.gz -1Dmatrix_save mat.aff12.1D

  # Use LPA cost function:
  ffs_moco -input epi.nii.gz -prefix epi_mc.nii.gz -cost lpa

  # Two-pass with base volume 10:
  ffs_moco -input epi.nii.gz -prefix epi_mc.nii.gz -base 10 -twopass
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Input/Output ---
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "-input",
        required=True,
        dest="input_file",
        help="4D timeseries (.nii/.nii.gz) [required]",
    )
    io_group.add_argument("-prefix", required=True, help="Output prefix [required]")
    io_group.add_argument(
        "-base",
        default="0",
        help="Base volume index (default: 0), or path to external 3D file",
    )

    # --- Method ---
    method_group = parser.add_argument_group("Method")
    method_group.add_argument(
        "-cost",
        choices=["wls", "lpa", "quad"],
        default="wls",
        help="Cost function: wls=weighted least squares "
        "(default), lpa=local Pearson absolute, quad=quadrature phase",
    )
    method_group.add_argument(
        "-maxiter",
        type=int,
        default=5,
        help="GN iterations per volume (default: 5, was 23)",
    )
    method_group.add_argument(
        "-dxy_thresh",
        type=float,
        default=0.07,
        help="Translation convergence threshold in voxels (default: 0.07)",
    )
    method_group.add_argument(
        "-dph_thresh",
        type=float,
        default=0.21,
        help="Rotation convergence threshold in degrees (default: 0.21)",
    )
    method_group.add_argument(
        "-twopass", action="store_true", help="Coarse blur + fine pass"
    )
    method_group.add_argument(
        "-nochain",
        action="store_true",
        help="Don't chain-initialize from previous volume",
    )
    method_group.add_argument(
        "-automask", action="store_true", help="Use automask for weighting"
    )
    method_group.add_argument(
        "-weight_automask",
        action="store_true",
        help="Use automask × continuous weight (tight mask + quality weighting)",
    )
    method_group.add_argument(
        "-blur",
        type=float,
        default=0.0,
        help="Pre-blur FWHM in mm for estimation (default: 0)",
    )
    method_group.add_argument(
        "-fast",
        action="store_true",
        help="Fast mode: fixed iterations, no convergence check (runs exactly -maxiter)",
    )
    method_group.add_argument(
        "-no-compile",
        action="store_true",
        help="Disable torch.compile for hot path (default: compile on CUDA)",
    )

    # --- Interpolation ---
    interp_group = parser.add_argument_group("Interpolation")
    interp_group.add_argument(
        "-interp",
        choices=["linear", "cubic", "quintic", "heptic", "wsinc5"],
        default="heptic",
        help="During estimation (default: heptic)",
    )
    interp_group.add_argument(
        "-final",
        choices=["linear", "cubic", "quintic", "heptic", "wsinc5"],
        default="wsinc5",
        dest="final_interp",
        help="For output (default: wsinc5)",
    )

    # --- Output files ---
    out_group = parser.add_argument_group("Output files")
    out_group.add_argument(
        "-1Dfile", default=None, help="Save 6-column motion parameters (.1D)"
    )
    out_group.add_argument(
        "-1Dmatrix_save", default=None, help="Save affine matrices (.aff12.1D)"
    )
    out_group.add_argument("-dfile", default=None, help="Save 9-column diagnostic file")
    out_group.add_argument(
        "-maxdisp1D", default=None, help="Save max displacement per volume"
    )
    out_group.add_argument(
        "-iterfile", default=None, help="Save iterations per volume (.1D)"
    )
    out_group.add_argument(
        "-save_mean",
        action="store_true",
        help="Save mean of output timeseries as mean_{prefix_basename}{ext}",
    )

    # --- Hardware ---
    hw_group = parser.add_argument_group("Hardware")
    hw_group.add_argument(
        "-device", default=None, help="PyTorch device: cuda, mps, cpu (auto-detected)"
    )
    add_verbose_arg(hw_group, default=1)
    hw_group.add_argument(
        "-debug_memory",
        action="store_true",
        help="Print VRAM usage vs. prediction after registration and resampling loops",
    )

    args = parser.parse_args(argv)
    return args


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point for ffs_moco."""
    args = parse_args(argv)

    # --- Device selection ---
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    verb = args.verb
    if verb >= 1:
        print(f"ffs_moco: device={device}")

    # --- Load input ---
    t0 = time.time()
    data, header_info = load_image(args.input_file)

    if data.ndim != 4:
        print(f"Error: input must be 4D, got {data.ndim}D", file=sys.stderr)
        sys.exit(1)

    if verb >= 1:
        print(f"Input: {args.input_file} {data.shape} ({data.shape[0]} volumes)")
        print(f"Load time: {time.time() - t0:.2f}s")

    # --- Parse base ---
    base_vol = None
    try:
        base_index = int(args.base)
    except ValueError:
        # External base volume
        if verb >= 1:
            print(f"Loading external base: {args.base}")
        base_vol, _ = load_image(args.base)
        if base_vol.ndim == 4:
            base_vol = base_vol[0]
        base_index = 0

    # --- Build config ---
    config = MocoConfig(
        base_index=base_index,
        cost=args.cost,
        interp=args.interp,
        final_interp=args.final_interp,
        max_iter=args.maxiter,
        twopass=args.twopass,
        blur_fwhm=args.blur,
        chain_init=not args.nochain,
        automask=args.automask,
        weight_automask=args.weight_automask,
        dxy_thresh=args.dxy_thresh,
        dph_thresh=args.dph_thresh,
        fixed_iter=args.fast,
        compile=not args.no_compile,
        device=str(device),
        verb=verb,
        debug_memory=args.debug_memory,
    )

    # --- Run motion correction ---
    t1 = time.time()
    result = moco(data, config, header_info=header_info, base_vol=base_vol)

    if verb >= 1:
        print(f"Total registration: {time.time() - t1:.2f}s")

    # --- Save outputs ---
    pfx = parse_prefix(args.prefix)
    out_path = pfx.as_file()
    save_image(result.aligned, out_path, header_info=header_info)
    if verb >= 1:
        print(f"Saved: {out_path}")

    if args.save_mean:
        mean_path = derive_mean_output_path(out_path)
        mean_image = result.aligned.mean(dim=0)
        save_image(mean_image, mean_path, header_info=header_info)
        if verb >= 1:
            print(f"Saved mean: {mean_path}")

    # Motion parameters
    onedfile = getattr(args, "1Dfile", None)
    if onedfile is not None:
        save_moco_1D(result.params, onedfile)
        if verb >= 1:
            print(f"Saved 1Dfile: {onedfile}")

    # Affine matrices
    matrix_save = getattr(args, "1Dmatrix_save", None)
    if matrix_save is not None:
        save_matrix_1D(
            result.matrices_dicom,
            matrix_save,
            header="ffs_moco matrices (DICOM-to-DICOM, row-by-row):",
        )
        if verb >= 1:
            print(f"Saved matrices: {matrix_save}")

    # Diagnostic file
    if args.dfile is not None:
        save_moco_dfile(result.params, result.rms_before, result.rms_after, args.dfile)
        if verb >= 1:
            print(f"Saved dfile: {args.dfile}")

    # Max displacement
    if args.maxdisp1D is not None:
        save_maxdisp_1D(result.max_displacement, args.maxdisp1D)
        if verb >= 1:
            print(f"Saved maxdisp: {args.maxdisp1D}")

    # Iterations per volume
    if args.iterfile is not None:
        with open(args.iterfile, "w") as f:
            for it in result.n_iters:
                f.write(f"{it}\n")
        if verb >= 1:
            print(f"Saved iterfile: {args.iterfile}")

    if verb >= 1:
        print(f"Total time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
