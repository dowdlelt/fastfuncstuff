"""CLI for GPU-accelerated multi-warp composition and application.

Command: nwarpforge (registered as entry point in pyproject.toml)

Usage:
    nwarpforge -source input.nii -nwarp 'warp1.nii mat.1D warp2.nii' -master template.nii -prefix output.nii

Equivalent to AFNI's 3dNwarpApply but with GPU acceleration and wsinc5 interpolation.
"""

from __future__ import annotations

import argparse
import time

import torch

from .nwarpforge import nwarpforge, parse_nwarp_string


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="nwarpforge",
        description="GPU-accelerated multi-warp composition and application",
        epilog="""Composition order:
  For -nwarp 'A B C', the result is C(B(A(x)))
  where x is a coordinate in output space.

  This is a 'pull' or 'backward' mapping convention:
  the composed warp tells where to sample FROM in source space
  for each output location.

Examples:
  # Apply single warp:
  nwarpforge -source subj.nii -nwarp warp.nii -prefix out.nii

  # Compose warp + affine matrix:
  nwarpforge -source epi.nii -nwarp 'warp.nii matrix.aff12.1D' -master template.nii -prefix out.nii

  # Full preprocessing chain (motion + warp to template):
  nwarpforge -source epi.nii \\
      -nwarp 'template_warp.nii motion_correct.aff12.1D fieldmap_warp.nii' \\
      -master template.nii \\
      -interp wsinc5 \\
      -prefix epi_in_template.nii

  # 4D time series with per-volume motion matrices:
  nwarpforge -source epi_4d.nii \\
      -nwarp 'template_warp.nii motion_matrices.aff12.1D' \\
      -master template.nii \\
      -prefix epi_4d_warped.nii
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "-source", required=True, help="Source dataset to warp (3D or 4D NIfTI)"
    )
    io_group.add_argument(
        "-nwarp",
        required=True,
        help="Warp chain (quoted, space-separated). "
        "E.g., 'warp1.nii matrix.1D warp2.nii'",
    )
    io_group.add_argument("-prefix", required=True, help="Output dataset path")
    io_group.add_argument(
        "-master",
        default=None,
        help="Master dataset defining output grid. "
        "If not specified, uses source grid or first warp grid.",
    )

    interp_group = parser.add_argument_group("Interpolation")
    interp_group.add_argument(
        "-interp",
        choices=["linear", "wsinc5"],
        default="wsinc5",
        help="Final interpolation for source data (default: wsinc5)",
    )
    interp_group.add_argument(
        "-ainterp",
        choices=["linear"],
        default="linear",
        help="Warp interpolation during composition (default: linear)",
    )

    hw_group = parser.add_argument_group("Hardware")
    hw_group.add_argument(
        "-device", default=None, help="PyTorch device: cuda, mps, cpu (auto-detected)"
    )
    hw_group.add_argument(
        "-verb",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity: 0=quiet, 1=normal, 2=debug",
    )

    debug_group = parser.add_argument_group("Debug")
    debug_group.add_argument(
        "-time_range",
        default=None,
        help="Process only specified time points (e.g., '0,5' for first 6 volumes). "
        "Useful for debugging.",
    )
    debug_group.add_argument(
        "-debug",
        action="store_true",
        help="Print detailed debug info (matrices, warp stats)",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

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
        print(f"nwarpforge: device={device}")
        print(f"nwarpforge: interp={args.interp}")

    # Parse time_range if provided
    time_range = None
    if args.time_range:
        parts = args.time_range.split(",")
        if len(parts) == 2:
            time_range = (int(parts[0]), int(parts[1]))
        else:
            time_range = (0, int(parts[0]))
        if verb >= 1:
            print(f"nwarpforge: time_range={time_range}")

    nwarp_specs = parse_nwarp_string(args.nwarp)
    if verb >= 1:
        print(f"nwarpforge: chain has {len(nwarp_specs)} transform(s)")
        for i, spec in enumerate(nwarp_specs):
            print(f"  [{i}] {spec}")

    t0 = time.time()

    nwarpforge(
        source_path=args.source,
        nwarp_specs=nwarp_specs,
        prefix=args.prefix,
        master_path=args.master,
        interp=args.interp,
        device=device,
        verb=verb,
        time_range=time_range,
        debug=args.debug,
    )

    if verb >= 1:
        print(f"nwarpforge: total time {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
