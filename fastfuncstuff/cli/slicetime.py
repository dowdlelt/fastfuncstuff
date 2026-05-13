"""CLI for GPU slice-timing correction (matches 3dTshift).

Command: ffs_slicetime (registered as entry point in pyproject.toml)

Usage:
    ffs_slicetime -input func.nii.gz -prefix func_st.nii.gz -tpattern timing.1D -TR 1.5
    ffs_slicetime -input func.nii.gz -prefix func_st.nii.gz -tpattern func.json -Fourier
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from fastfuncstuff.cli_utils import add_verbose_arg
from fastfuncstuff.io.afni import get_tr_from_file
from fastfuncstuff.processing.io import load_image, save_image
from fastfuncstuff.processing.slicetime import load_slice_timing, slicetime_correct, temporal_resample


def _resolve_tr(
    cli_tr: float | None,
    tpattern_path: str,
    input_path: str,
    verb: int,
) -> float:
    """Resolve TR from CLI flag, JSON sidecar, or NIfTI header.

    Priority: CLI flag > JSON RepetitionTime > NIfTI header.
    Warns if sources disagree.
    """
    json_tr: float | None = None
    header_tr: float | None = None

    # Try JSON sidecar
    p = Path(tpattern_path)
    if p.suffix == ".json":
        try:
            data = json.loads(p.read_text())
            if "RepetitionTime" in data:
                json_tr = float(data["RepetitionTime"])
        except Exception:
            pass

    # Try NIfTI header
    try:
        hdr_val = get_tr_from_file(input_path)
        if hdr_val > 0:
            header_tr = hdr_val
    except Exception:
        pass

    # Resolve with priority
    if cli_tr is not None:
        tr = cli_tr
        # Warn if disagrees with other sources
        if json_tr is not None and abs(json_tr - tr) > 1e-4:
            print(f"  WARNING: -TR {tr:.4f}s overrides JSON RepetitionTime {json_tr:.4f}s")
        if header_tr is not None and abs(header_tr - tr) > 1e-4:
            print(f"  WARNING: -TR {tr:.4f}s overrides NIfTI header TR {header_tr:.4f}s")
        return tr

    if json_tr is not None:
        tr = json_tr
        if verb >= 1:
            print(f"  TR from JSON: {tr:.4f}s")
        if header_tr is not None and abs(header_tr - tr) > 1e-4:
            print(f"  WARNING: JSON TR {tr:.4f}s differs from header TR {header_tr:.4f}s "
                  f"(using JSON)")
        return tr

    if header_tr is not None:
        if verb >= 1:
            print(f"  TR from header: {header_tr:.4f}s")
        return header_tr

    raise ValueError(
        "Could not determine TR from JSON or NIfTI header. Use -TR to specify."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ffs_slicetime",
        description="Slice-timing correction for 4D fMRI (GPU, 3dTshift-compatible)",
    )
    parser.add_argument("-input", required=True, help="Input 4D volume (.nii/.nii.gz)")
    parser.add_argument("-prefix", required=True, help="Output file")
    parser.add_argument(
        "-tpattern", required=True,
        help="Slice timing: text file (one time per line, seconds) or BIDS JSON",
    )
    parser.add_argument(
        "-TR", type=float, default=None,
        help="Force TR in seconds (overrides JSON/header; warns on mismatch)",
    )
    parser.add_argument(
        "-tzero", type=float, default=None,
        help="Target time within TR to align to (seconds). "
             "Default: mean of slice times (3dTshift default)",
    )
    parser.add_argument(
        "-ignore", type=int, default=0,
        help="Number of initial volumes to skip (pass through unchanged)",
    )
    parser.add_argument(
        "-resample", type=float, default=None, metavar="TR_NEW",
        help="After slice-timing correction, resample to a new TR grid "
             "(seconds). E.g., -resample 1.5 resamples 1.75s data to 1.5s. "
             "Output NIfTI header TR is updated. Useful for TR-locking onsets "
             "for GLMsingle-style analysis.",
    )

    # Interpolation method — mutually exclusive flags
    interp = parser.add_mutually_exclusive_group()
    interp.add_argument("-Fourier", action="store_const", const="fourier",
                        dest="method", help="Fourier interpolation (default)")
    interp.add_argument("-fourier", action="store_const", const="fourier",
                        dest="method", help=argparse.SUPPRESS)
    interp.add_argument("-linear", action="store_const", const="linear",
                        dest="method", help="Linear interpolation")
    interp.add_argument("-cubic", action="store_const", const="cubic",
                        dest="method", help="Cubic Lagrange interpolation")
    interp.add_argument("-quintic", action="store_const", const="quintic",
                        dest="method", help="Quintic Lagrange interpolation")
    interp.add_argument("-heptic", action="store_const", const="heptic",
                        dest="method", help="Heptic Lagrange interpolation")
    interp.add_argument("-wsinc5", action="store_const", const="wsinc5",
                        dest="method", help="Windowed sinc (10-point)")
    interp.add_argument("-wsinc9", action="store_const", const="wsinc9",
                        dest="method", help="Windowed sinc (18-point)")
    parser.set_defaults(method="fourier")

    parser.add_argument("-device", default=None, help="PyTorch device (cuda, mps, cpu)")
    add_verbose_arg(parser, default=1)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Device selection
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
        print(f"slicetime: device={device}, method={args.method}")

    t0 = time.time()
    vol, header = load_image(args.input, device=device)

    if vol.ndim != 4:
        raise ValueError(f"Expected 4D input, got {vol.ndim}D with shape {vol.shape}")

    nt, nz, ny, nx = vol.shape
    if verb >= 1:
        print(f"Input: {args.input} ({nt} volumes, {nz} slices, {ny}x{nx})")

    # Resolve TR: CLI flag > JSON > header
    tr = _resolve_tr(args.TR, args.tpattern, args.input, verb)
    if verb >= 1:
        print(f"TR: {tr:.4f}s")

    # Load slice timing
    slice_timing = load_slice_timing(args.tpattern)
    if verb >= 1:
        print(f"Slice timing: {len(slice_timing)} entries from {args.tpattern}")
        print(f"  range: [{min(slice_timing):.4f}, {max(slice_timing):.4f}]s")

    if len(slice_timing) != nz:
        raise ValueError(
            f"Slice timing has {len(slice_timing)} entries but volume has "
            f"{nz} slices. These must match."
        )

    # Run correction
    corrected = slicetime_correct(
        vol,
        slice_timing=slice_timing,
        tr=tr,
        tzero=args.tzero,
        method=args.method,
        ignore=args.ignore,
        device=device,
        verbose=verb >= 1,
    )

    # Optional temporal resampling to a new TR grid
    output_tr = tr
    if args.resample is not None:
        tr_new = args.resample
        if tr_new <= 0:
            raise ValueError(f"-resample must be positive, got {tr_new}")
        if verb >= 1:
            print(f"Resampling: {tr:.4f}s -> {tr_new:.4f}s")
        corrected = temporal_resample(
            corrected,
            tr_old=tr,
            tr_new=tr_new,
            method="cubic",
            device=device,
            verbose=verb >= 1,
        )
        output_tr = tr_new

    # Update TR in header before saving
    if header is not None and header.get("header") is not None:
        header["header"].set_xyzt_units(xyz="mm", t="sec")
        header["header"]["pixdim"][4] = output_tr

    # Save
    save_image(corrected, args.prefix, header_info=header)

    if verb >= 1:
        print(f"Saved: {args.prefix}")
        if args.resample is not None:
            print(f"  Output TR: {output_tr:.4f}s ({corrected.shape[0]} volumes)")
        print(f"Time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
