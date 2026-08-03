#!/usr/bin/env python3
"""Convert MRI magnitude/phase data to real/imaginary data, and vice versa."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from fastfuncstuff.cli_utils import add_verbose_arg, parse_device_arg, parse_prefix, spinner
from fastfuncstuff.memory import get_available_memory
from fastfuncstuff.processing.complex import (
    magnitude_phase_to_real_imag,
    real_imag_to_magnitude_phase,
    scale_phase_to_radians,
)
from fastfuncstuff.processing.io import load_image, save_image
from fastfuncstuff.utils import REGISTRATION_TF32, configure_torch_backends


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


def _convert_streaming(
    first: torch.Tensor,
    second: torch.Tensor,
    direction: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Elementwise polar<->Cartesian conversion, streamed in place on the host.

    The math (``mag·cos φ`` / ``hypot`` / ``atan2``) is a single memory-bandwidth-bound
    elementwise pass, so it stays on the CPU on purpose: a GPU would only add PCIe
    round-trips (H2D + D2H) for a cos/sin that costs almost nothing — PCIe is slower
    than host RAM and torch's CPU kernels are already vectorised. Each output is written
    back over its input buffer — real over ``first``, imag over ``second`` (or mag/phase)
    — because every element's outputs depend only on that same element's two inputs, so
    the peak footprint is the two in/out buffers, not four. Chunking bounds only the
    per-chunk cos/sin temporaries, sized from free RAM via the [[Memory module]].
    """
    fn = (
        magnitude_phase_to_real_imag
        if direction == "mag_phase_to_real_imag"
        else real_imag_to_magnitude_phase
    )
    # reshape(-1) is a view for contiguous inputs (load_image returns contiguous), so the
    # in-place writes below land in `first`/`second`'s own storage — no extra full copy.
    first_flat = first.reshape(-1)
    second_flat = second.reshape(-1)
    n = first_flat.numel()

    # Per-chunk transient inside `fn`: 2 outputs + ~2 cos/sin temporaries. Bound it to
    # free host RAM (>=1M elems so tiny-RAM runs still progress; <= the whole array).
    live_buffers = 4
    budget = get_available_memory(torch.device("cpu"))
    chunk = max(1 << 20, min(int(budget // (live_buffers * first_flat.element_size())), n))

    n_chunks = (n + chunk - 1) // chunk
    for start in tqdm(
        range(0, n, chunk),
        total=n_chunks,
        desc="Converting complex data",
        unit="chunk",
        leave=True,
        disable=n_chunks <= 1,
    ):
        stop = min(start + chunk, n)
        # fn materialises both outputs before we overwrite either input slice, so the
        # in-place write is safe (no read-after-write aliasing).
        out_one, out_two = fn(first_flat[start:stop], second_flat[start:stop])
        first_flat[start:stop] = out_one
        second_flat[start:stop] = out_two

    return first_flat.reshape(first.shape), second_flat.reshape(second.shape)


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
    configure_torch_backends(device, tf32=REGISTRATION_TF32)

    # This is a memory-bandwidth-bound elementwise pass; a GPU only adds PCIe
    # transfers for a trivial cos/sin, so it runs on the host regardless of -device.
    # Say so rather than silently overriding the flag.
    if device.type != "cpu" and args.verb >= 1:
        print(f"Note: complex conversion is memory-bound; running on host, not {device.type}.")

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
        output_one, output_two = _convert_streaming(first, second, "mag_phase_to_real_imag")
        output_specs = (("real", output_one, args.no_real), ("imag", output_two, args.no_imag))
    else:
        output_one, output_two = _convert_streaming(first, second, "real_imag_to_mag_phase")
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
        print(f"ffs_util_complex: {direction}")
        for path in paths:
            print(f"  Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
