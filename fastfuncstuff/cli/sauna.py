#!/usr/bin/env python3
"""CLI for SAUNA denoising (ffs_sauna)."""

from __future__ import annotations

import argparse
import sys
import time

from fastfuncstuff.cli_utils import parse_prefix, print_cli_header
from fastfuncstuff.denoise.sauna import SaunaConfig, run_sauna
from fastfuncstuff.utils import configure_torch_backends, get_device


class _HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Preserve examples while showing defaults."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ffs_sauna",
        description=(
            "SAUNA: Signal-Adaptive Unbiased Noise Attenuation.\n"
            "Next-gen fMRI denoiser using noise-volume g-factor estimation\n"
            "and Gavish-Donoho optimal singular value shrinkage."
        ),
        formatter_class=_HelpFormatter,
        epilog="""
Examples:
  # Complex data with 3 trailing noise volumes
  ffs_sauna -input-magn sub-01_bold.nii.gz \\
            -input-phase sub-01_phase.nii.gz \\
            -prefix SAUNA_sub-01_bold \\
            -noise-volume-last 3

  # Magnitude-only with custom smoothing
  ffs_sauna -input-magn sub-01_bold.nii.gz \\
            -prefix SAUNA_sub-01_bold \\
            -magnitude-only \\
            -noise-volume-last 3 \\
            -gfactor-smooth-fwhm 8

  # Fall back to MP-PCA hard threshold (instead of optimal shrinkage)
  ffs_sauna -input-magn sub-01_bold.nii.gz \\
            -input-phase sub-01_phase.nii.gz \\
            -prefix SAUNA_sub-01_bold \\
            -noise-volume-last 3 \\
            -shrinkage hard
""",
    )

    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument("-input-magn", required=True, help="Input magnitude NIfTI file")
    io_group.add_argument(
        "-input-phase",
        default=None,
        help="Input phase NIfTI file (required unless -magnitude-only)",
    )
    io_group.add_argument("-prefix", required=True, help="Output prefix")
    io_group.add_argument(
        "-make-complex-nii",
        action="store_true",
        help="Write separate magnitude and phase outputs",
    )
    io_group.add_argument(
        "-save-gfactor-map",
        action="store_true",
        help="Save estimated g-factor map from noise volumes",
    )
    io_group.add_argument(
        "-save-residual-map",
        action="store_true",
        help="Save denoising residual map (magnitude of complex difference)",
    )

    algo_group = parser.add_argument_group("Algorithm")
    algo_group.add_argument(
        "-noise-volume-last",
        type=int,
        required=True,
        help="Number of trailing noise-only volumes (REQUIRED, >= 2)",
    )
    algo_group.add_argument(
        "-temporal-phase",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="Temporal phase correction mode",
    )
    algo_group.add_argument(
        "-phase-filter-width",
        type=float,
        default=10.0,
        help="Phase low-pass strength",
    )
    algo_group.add_argument(
        "-magnitude-only",
        action="store_true",
        help="Ignore phase input and denoise magnitude as complex-with-zero-phase",
    )
    algo_group.add_argument(
        "-kernel-size-pca",
        nargs=3,
        type=int,
        default=None,
        metavar=("KX", "KY", "KZ"),
        help="Patch size for LLR denoising",
    )
    algo_group.add_argument(
        "-patch-overlap",
        type=int,
        default=2,
        help="Patch overlap divisor",
    )
    algo_group.add_argument(
        "-phase-slice-average",
        action="store_true",
        help="Enable mean-phase removal per slice",
    )
    algo_group.add_argument(
        "-gfactor-smooth-fwhm",
        default="auto",
        help="FWHM (voxels) for Gaussian smoothing of noise-volume g-factor. "
        "'auto' uses LOO cross-validation on noise volumes to find optimal FWHM.",
    )
    algo_group.add_argument(
        "-shrinkage",
        choices=["optimal", "hard"],
        default="optimal",
        help="Shrinkage mode: optimal (Gavish-Donoho) or hard (MP-PCA threshold)",
    )

    perf_group = parser.add_argument_group("Performance")
    perf_group.add_argument(
        "-device",
        default=None,
        help="Device override (cuda, mps, cpu). Default: auto",
    )
    perf_group.add_argument(
        "-svd-batch-size",
        type=int,
        default=512,
        help="Number of patches per batched SVD call",
    )
    perf_group.add_argument(
        "-decomp-method",
        choices=["auto", "svd", "eigh"],
        default="auto",
        help="Decomposition method: auto (eigh when M/N>=2), svd, or eigh",
    )
    perf_group.add_argument(
        "-quiet",
        action="store_true",
        help="Disable tqdm/progress prints",
    )

    return parser.parse_args(argv)


def _parse_fwhm(val: str) -> float | str:
    """Parse -gfactor-smooth-fwhm: 'auto' or a float."""
    if val.lower() == "auto":
        return "auto"
    try:
        return float(val)
    except ValueError:
        print(f"ERROR: -gfactor-smooth-fwhm must be 'auto' or a number, got '{val}'")
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.magnitude_only and args.input_phase is None:
        print("ERROR: -input-phase is required unless -magnitude-only is set")
        sys.exit(1)

    if args.noise_volume_last < 2:
        print("ERROR: -noise-volume-last must be >= 2 for SAUNA")
        sys.exit(1)

    prefix_info = parse_prefix(args.prefix)
    prefix = prefix_info.stem

    device = get_device(args.device)
    configure_torch_backends(device)

    print_cli_header("ffs_sauna", "SAUNA: Signal-Adaptive Unbiased Noise Attenuation")
    print(f"Input magnitude: {args.input_magn}")
    print(f"Input phase: {args.input_phase}")
    print(f"Output prefix: {prefix}")
    print(f"Device: {device}")

    cfg = SaunaConfig(
        temporal_phase=args.temporal_phase,
        phase_filter_width=args.phase_filter_width,
        noise_volume_last=args.noise_volume_last,
        magnitude_only=args.magnitude_only,
        kernel_size_pca=tuple(args.kernel_size_pca) if args.kernel_size_pca is not None else None,
        patch_overlap=max(1, args.patch_overlap),
        phase_slice_average=args.phase_slice_average,
        save_gfactor_map=args.save_gfactor_map,
        save_residual_map=args.save_residual_map,
        make_complex_nii=args.make_complex_nii,
        write_gzipped_niftis=True,
        svd_batch_size=args.svd_batch_size,
        decomp_method=args.decomp_method,
        verbose=not args.quiet,
        gfactor_smooth_fwhm=_parse_fwhm(args.gfactor_smooth_fwhm),
        shrinkage=args.shrinkage,
    )

    t0 = time.time()
    outputs = run_sauna(
        magnitude_file=args.input_magn,
        phase_file=args.input_phase,
        output_prefix=prefix,
        config=cfg,
        device=device,
    )
    elapsed = time.time() - t0

    print("\nDone")
    print(f"  Magnitude output: {outputs.magnitude_file}")
    if outputs.phase_file is not None:
        print(f"  Phase output: {outputs.phase_file}")
    if outputs.gfactor_file is not None:
        print(f"  G-factor output: {outputs.gfactor_file}")
    if outputs.residual_file is not None:
        print(f"  Residual output: {outputs.residual_file}")
    print(f"  Metadata: {outputs.metadata_file}")
    print(f"  Elapsed: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
