#!/usr/bin/env python3
"""CLI for NORDIC-style denoising (ffs_nordic)."""

from __future__ import annotations

import argparse
import sys
import time

from fastfuncstuff.cli_utils import add_verbose_arg, parse_prefix, print_cli_header
from fastfuncstuff.denoise.nordic import NordicConfig, run_nordic, run_nordic_multiecho
from fastfuncstuff.utils import configure_torch_backends, get_device


class _HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Preserve examples while showing defaults."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ffs_nordic",
        description="NORDIC-style denoising for magnitude-only or complex (magnitude+phase) fMRI data.",
        formatter_class=_HelpFormatter,
        epilog="""
Examples:
  # Closest to MATLAB call
  ffs_nordic -input-magn sub-08_bold.nii.gz \
             -input-phase sub-08_phase.nii.gz \
             -prefix NORDIC_sub-08_bold \
             -temporal-phase 1 \
             -phase-filter-width 10 \
             -noise-volume-last 3 \
             -nordic

  # Magnitude-only mode
  ffs_nordic -input-magn sub-08_bold.nii.gz \
             -prefix NORDIC_sub-08_bold_magonly \
             -magnitude-only
""",
    )

    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "-input-magn",
        "-input_magn",
        nargs="+",
        required=True,
        help="Input magnitude NIfTI file(s). Pass one per echo (>=2) to enable "
        "the multi-echo cross-echo signal-rescue path.",
    )
    io_group.add_argument(
        "-input-phase",
        "-input_phase",
        nargs="+",
        default=None,
        help="Input phase NIfTI file(s); one per magnitude file. Required unless -magnitude-only.",
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
        help="Save estimated g-factor proxy map",
    )
    io_group.add_argument(
        "-save-residual-map",
        action="store_true",
        help="Save denoising residual map (magnitude of complex difference)",
    )
    io_group.add_argument(
        "-save-num-comps",
        "-save_num_comps",
        action="store_true",
        help="Save per-voxel count of components removed (patch-averaged, fractional)",
    )
    io_group.add_argument(
        "-no-resid-qc",
        "-no_resid_qc",
        dest="resid_qc",
        action="store_false",
        help="Disable the multi-echo cross-echo residual correlation QC maps (on by default)",
    )

    algo_group = parser.add_argument_group("Algorithm")
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
        "-noise-volume-last",
        type=int,
        default=0,
        help="Number of trailing volumes used as noise-only",
    )
    algo_group.add_argument(
        "-factor-error",
        type=float,
        default=1.0,
        help="NORDIC threshold scaling (>1 higher floor, <1 lower floor)",
    )
    algo_group.add_argument(
        "-nordic",
        action="store_true",
        help="Use NORDIC thresholding (default if MP not selected)",
    )
    algo_group.add_argument(
        "-mp",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="MP mode (1/2 enables MP-PCA style thresholding)",
    )
    algo_group.add_argument(
        "-magnitude-only",
        action="store_true",
        help="Ignore phase input and denoise magnitude as complex-with-zero-phase",
    )
    algo_group.add_argument(
        "-per-echo-gfactor",
        "-per_echo_gfactor",
        action="store_true",
        help=(
            "Multi-echo: estimate each echo's own g-factor (and thermal sigma) "
            "instead of sharing echo 1's. Use when thermal sigma is not "
            "TE-invariant; costs one g-factor pass per echo (default: share echo 1)"
        ),
    )
    algo_group.add_argument(
        "-kernel-size-pca",
        nargs=3,
        type=int,
        default=None,
        metavar=("KX", "KY", "KZ"),
        help="Patch size for main LLR denoising",
    )
    algo_group.add_argument(
        "-kernel-size-gfactor",
        nargs=3,
        type=int,
        default=[14, 14, 1],
        metavar=("KX", "KY", "KZ"),
        help="Patch size for g-factor proxy estimation",
    )
    algo_group.add_argument(
        "-gfactor-nvols",
        type=int,
        default=90,
        help="Number of volumes for g-factor proxy estimation",
    )
    algo_group.add_argument(
        "-patch-overlap",
        type=int,
        default=2,
        help="Patch overlap divisor for main pass (MATLAB default: 2)",
    )
    algo_group.add_argument(
        "-gfactor-patch-overlap",
        type=int,
        default=2,
        help="Patch overlap divisor for g-factor pass",
    )
    algo_group.add_argument(
        "-use-magn-for-gfactor",
        action="store_true",
        help="Estimate g-factor from magnitude-only data (MATLAB use_magn_for_gfactor)",
    )
    algo_group.add_argument(
        "-phase-slice-average",
        action="store_true",
        help="Enable mean-phase removal per slice (MATLAB phase_slice_average_for_kspace_centering=1)",
    )

    me_group = parser.add_argument_group("Multi-echo rescue (>=2 echoes)")
    me_group.add_argument(
        "-no-rescue",
        "-no_rescue",
        dest="rescue",
        action="store_false",
        help="Disable the cross-echo signal-rescue guard (denoise each echo "
        "independently). Default: rescue on for multi-echo input.",
    )
    me_group.add_argument(
        "-rescue-band",
        "-rescue_band",
        type=float,
        default=0.25,
        help="Fraction of each echo's kill set (top singular values, nearest the "
        "threshold) tested for rescue. Larger = more components considered.",
    )
    me_group.add_argument(
        "-rescue-alpha",
        "-rescue_alpha",
        type=float,
        default=0.05,
        help="Per-patch false-rescue rate. The rescue threshold is the "
        "(1 - alpha) quantile of the all-thermal-noise null.",
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
        help="Number of patches per batched SVD call (tune for GPU memory vs speed)",
    )
    perf_group.add_argument(
        "-decomp-method",
        choices=["auto", "svd", "eigh"],
        default="auto",
        help="Decomposition method: auto (eigh when M/N>=2), svd, or eigh",
    )
    add_verbose_arg(perf_group, default=1)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    magn_files: list[str] = list(args.input_magn)
    phase_files: list[str] | None = list(args.input_phase) if args.input_phase is not None else None
    n_echoes = len(magn_files)

    if not args.magnitude_only and phase_files is None:
        print("ERROR: -input-phase is required unless -magnitude-only is set")
        sys.exit(1)
    if phase_files is not None and len(phase_files) != n_echoes:
        print(
            f"ERROR: got {n_echoes} magnitude file(s) but {len(phase_files)} phase file(s) "
            "(need one phase per echo)"
        )
        sys.exit(1)

    prefix_info = parse_prefix(args.prefix)
    prefix = prefix_info.stem

    device = get_device(args.device)
    configure_torch_backends(device)

    print_cli_header("ffs_nordic", "NORDIC-style denoising")
    print(f"Input magnitude: {magn_files}")
    print(f"Input phase: {phase_files}")
    print(f"Output prefix: {prefix}")
    print(f"Device: {device}")
    if n_echoes > 1:
        print(f"Multi-echo: {n_echoes} echoes, rescue={'on' if args.rescue else 'off'}")

    cfg = NordicConfig(
        temporal_phase=args.temporal_phase,
        phase_filter_width=args.phase_filter_width,
        noise_volume_last=args.noise_volume_last,
        factor_error=args.factor_error,
        nordic=(True if (args.nordic or args.mp == 0) else False),
        mp_mode=args.mp,
        magnitude_only=args.magnitude_only,
        kernel_size_pca=tuple(args.kernel_size_pca) if args.kernel_size_pca is not None else None,
        kernel_size_gfactor=tuple(args.kernel_size_gfactor),
        gfactor_nvols=args.gfactor_nvols,
        patch_overlap=max(1, args.patch_overlap),
        gfactor_patch_overlap=max(1, args.gfactor_patch_overlap),
        use_magn_for_gfactor=args.use_magn_for_gfactor,
        phase_slice_average=args.phase_slice_average,
        save_gfactor_map=args.save_gfactor_map,
        save_residual_map=args.save_residual_map,
        save_num_comps=args.save_num_comps,
        make_complex_nii=args.make_complex_nii,
        svd_batch_size=args.svd_batch_size,
        decomp_method=args.decomp_method,
        rescue=args.rescue,
        rescue_band=args.rescue_band,
        rescue_alpha=args.rescue_alpha,
        per_echo_gfactor=args.per_echo_gfactor,
        resid_qc=args.resid_qc,
        verbose=args.verb >= 1,
    )

    t0 = time.time()
    if n_echoes > 1:
        all_outputs = run_nordic_multiecho(
            magnitude_files=magn_files,
            phase_files=phase_files,  # type: ignore[arg-type]
            output_prefix=prefix,
            config=cfg,
            device=device,
        )
    else:
        all_outputs = [
            run_nordic(
                magnitude_file=magn_files[0],
                phase_file=phase_files[0] if phase_files is not None else None,
                output_prefix=prefix,
                config=cfg,
                device=device,
            )
        ]
    elapsed = time.time() - t0

    print("\nDone")
    for i, outputs in enumerate(all_outputs):
        if n_echoes > 1:
            print(f"  [echo {i + 1}]")
        print(f"  Magnitude output: {outputs.magnitude_file}")
        if outputs.phase_file is not None:
            print(f"  Phase output: {outputs.phase_file}")
        if outputs.gfactor_file is not None:
            print(f"  G-factor output: {outputs.gfactor_file}")
        if outputs.residual_file is not None:
            print(f"  Residual output: {outputs.residual_file}")
        if outputs.num_comps_file is not None:
            print(f"  Num-comps output: {outputs.num_comps_file}")
        print(f"  Metadata: {outputs.metadata_file}")
    print(f"  Elapsed: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
