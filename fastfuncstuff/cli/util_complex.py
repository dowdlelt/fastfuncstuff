#!/usr/bin/env python3
"""Convert MRI magnitude/phase data to real/imaginary data, and vice versa."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from fastfuncstuff.cli_utils import add_verbose_arg, parse_device_arg, parse_prefix, spinner
from fastfuncstuff.memory import estimate_chunk_size
from fastfuncstuff.processing.complex import (
    magnitude_phase_to_real_imag,
    real_imag_to_magnitude_phase,
    scale_phase_to_radians,
)
from fastfuncstuff.processing.io import load_image, save_image


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ffs_util_complex",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  ffs_util_complex -mag magnitude.nii.gz -phase phase.nii.gz -prefix complex
  ffs_util_complex -mag magnitude.nii.gz -phase phase_raw.nii.gz \\
      -phase_units scale -prefix complex
  ffs_util_complex -real complex_real.nii.gz -imag complex_imag.nii.gz -prefix polar
  ffs_util_complex -real complex_real.nii.gz -imag complex_imag.nii.gz \\
      -nomag -prefix phase_only
""",
    )
    inputs = parser.add_argument_group("Input (give exactly one pair)")
    inputs.add_argument("-mag", metavar="FILE", help="Magnitude NIfTI input.")
    inputs.add_argument("-phase", metavar="FILE", help="Phase NIfTI input.")
    inputs.add_argument("-real", metavar="FILE", help="Real-component NIfTI input.")
    inputs.add_argument("-imag", metavar="FILE", help="Imaginary-component NIfTI input.")

    output = parser.add_argument_group("Output")
    output.add_argument(
        "-prefix",
        required=True,
        metavar="PREFIX",
        help="Output prefix. Writes {prefix}_real and {prefix}_imag for -mag/-phase, "
        "or {prefix}_mag and {prefix}_phase for -real/-imag.",
    )
    output.add_argument("-nomag", "-no-mag", dest="no_mag", action="store_true")
    output.add_argument("-no_phase", "-no-phase", dest="no_phase", action="store_true")
    output.add_argument("-no_real", "-no-real", dest="no_real", action="store_true")
    output.add_argument("-no_imag", "-no-imag", dest="no_imag", action="store_true")

    options = parser.add_argument_group("Options")
    options.add_argument(
        "-phase_units",
        "-phase-units",
        dest="phase_units",
        choices=("rad", "scale"),
        default="rad",
        help="Input phase units for -mag/-phase: 'rad' (default) preserves radians; "
        "'scale' maps the loaded data range to [-pi, pi].",
    )
    options.add_argument(
        "-device",
        default=None,
        metavar="DEVICE",
        help="Compute device: cpu, cuda, cuda,N, or mps (default: auto).",
    )
    add_verbose_arg(options, default=1)
    return parser


def _convert_in_chunks(
    first: torch.Tensor,
    second: torch.Tensor,
    direction: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stream elementwise conversion through the requested device when needed."""
    first_flat = first.reshape(-1)
    second_flat = second.reshape(-1)
    result_one = torch.empty_like(first_flat, memory_format=torch.contiguous_format)
    result_two = torch.empty_like(second_flat, memory_format=torch.contiguous_format)
    n_voxels = first_flat.numel()

    # Keep complete NIfTI inputs and outputs in CPU RAM.  CUDA receives only a
    # bounded chunk, avoiding an all-volume device allocation for long 4-D runs.
    if device.type == "cuda":
        chunk_size = estimate_chunk_size(
            n_voxels=n_voxels,
            n_timepoints=1,
            n_regressors=0,
            device=device,
            operation="glm",
        )
    else:
        chunk_size = n_voxels

    starts = range(0, n_voxels, chunk_size)
    n_chunks = (n_voxels + chunk_size - 1) // chunk_size
    for start in tqdm(
        starts,
        total=n_chunks,
        desc="Converting complex data",
        unit="chunk",
        leave=True,
        disable=n_chunks <= 1,
    ):
        stop = min(start + chunk_size, n_voxels)
        x = first_flat[start:stop].to(device)
        y = second_flat[start:stop].to(device)
        if direction == "mag_phase_to_real_imag":
            out_one, out_two = magnitude_phase_to_real_imag(x, y)
        else:
            out_one, out_two = real_imag_to_magnitude_phase(x, y)
        result_one[start:stop] = out_one.cpu()
        result_two[start:stop] = out_two.cpu()

    return result_one.reshape(first.shape), result_two.reshape(second.shape)


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    mag_phase = bool(args.mag) or bool(args.phase)
    real_imag = bool(args.real) or bool(args.imag)
    if (
        mag_phase == real_imag
        or (mag_phase and not (args.mag and args.phase))
        or (real_imag and not (args.real and args.imag))
    ):
        print(
            "ERROR: give exactly one complete input pair: -mag -phase or -real -imag.",
            file=sys.stderr,
        )
        return 1

    input_paths = (args.mag, args.phase) if mag_phase else (args.real, args.imag)
    missing = [path for path in input_paths if not Path(path).is_file()]
    if missing:
        print(f"ERROR: input file not found: {missing[0]}", file=sys.stderr)
        return 1

    if mag_phase:
        if args.no_mag or args.no_phase:
            print("ERROR: -nomag and -no_phase apply only to -real -imag input.", file=sys.stderr)
            return 1
        if args.no_real and args.no_imag:
            print("ERROR: -no_real and -no_imag would suppress every output.", file=sys.stderr)
            return 1
    else:
        if args.no_real or args.no_imag:
            print("ERROR: -no_real and -no_imag apply only to -mag -phase input.", file=sys.stderr)
            return 1
        if args.no_mag and args.no_phase:
            print("ERROR: -nomag and -no_phase would suppress every output.", file=sys.stderr)
            return 1

    try:
        device, _, _ = parse_device_arg(args.device)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    with spinner(f"Loading {Path(input_paths[0]).name} and {Path(input_paths[1]).name}"):
        first, header_info = load_image(input_paths[0])
        second, _ = load_image(input_paths[1])
    if first.shape != second.shape:
        print(
            f"ERROR: input shape mismatch: {tuple(first.shape)} vs {tuple(second.shape)}.",
            file=sys.stderr,
        )
        return 1

    if mag_phase:
        if args.phase_units == "scale":
            try:
                second = scale_phase_to_radians(second)
            except ValueError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1
        output_one, output_two = _convert_in_chunks(first, second, "mag_phase_to_real_imag", device)
        output_specs = (("real", output_one, args.no_real), ("imag", output_two, args.no_imag))
    else:
        output_one, output_two = _convert_in_chunks(first, second, "real_imag_to_mag_phase", device)
        output_specs = (("mag", output_one, args.no_mag), ("phase", output_two, args.no_phase))

    prefix = parse_prefix(args.prefix)
    paths = [prefix.with_suffix(name) for name, _, disabled in output_specs if not disabled]
    with spinner(f"Saving {len(paths)} complex-data output{'s' if len(paths) != 1 else ''}"):
        for (_, data, _), path in zip(
            (spec for spec in output_specs if not spec[2]), paths, strict=True
        ):
            save_image(data, path, header_info=header_info)

    if args.verb >= 1:
        direction = (
            "magnitude/phase -> real/imaginary"
            if mag_phase
            else "real/imaginary -> magnitude/phase"
        )
        print(f"ffs_util_complex: {direction} on {device}")
        for path in paths:
            print(f"  Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
