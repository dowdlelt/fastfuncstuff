"""CLI for GPU automask creation (AFNI-compatible).

Command: ffs_util_automask (registered as entry point in pyproject.toml)

Usage:
    ffs_util_automask -input vol.nii -prefix mask.nii [-clfrac 0.5] [-dilate 0] [-device cuda]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from fastfuncstuff.cli_utils import add_verbose_arg, spinner
from fastfuncstuff.processing.io import load_image, save_image
from fastfuncstuff.processing.mask import automask
from fastfuncstuff.utils import REGISTRATION_TF32, configure_torch_backends


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ffs_util_automask",
        description="Create a binary brain mask from a 3D volume (GPU, AFNI-compatible)",
    )
    parser.add_argument("-input", required=True, help="Input volume (.nii/.nii.gz)")
    parser.add_argument("-prefix", required=True, help="Output mask file")
    parser.add_argument(
        "-clfrac",
        type=float,
        default=0.5,
        help="Clip-level fraction for THD_cliplevel (default: 0.5, matching AFNI)",
    )
    parser.add_argument(
        "-dilate", type=int, default=2, help="Extra dilation iterations after mask (default: 2)"
    )
    parser.add_argument(
        "-peelcount", type=int, default=1, help="Peel erosion iterations (AFNI default: 1)"
    )
    parser.add_argument(
        "-peelthr",
        type=int,
        default=17,
        help="Min 18-neighbors to survive peeling (AFNI default: 17)",
    )
    # Keep -clip_frac as hidden alias for backwards compat
    parser.add_argument("-clip_frac", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("-device", default=None, help="PyTorch device (cuda, mps, cpu)")
    add_verbose_arg(parser, default=1)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Handle backwards-compat alias
    clip_frac = args.clip_frac if args.clip_frac is not None else args.clfrac

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    configure_torch_backends(device, tf32=REGISTRATION_TF32)

    verb = args.verb
    if verb >= 1:
        print(f"automask: device={device}")

    t0 = time.time()
    with spinner(f"Loading {Path(args.input).name}"):
        vol, header = load_image(args.input, device=device)

    # Handle 4D: use first volume
    if vol.ndim == 4:
        if verb >= 1:
            print(f"4D input ({vol.shape[0]} volumes), using first volume")
        vol = vol[0]

    if verb >= 1:
        print(f"Input: {args.input} {vol.shape}")

    mask = automask(
        vol,
        clip_frac=clip_frac,
        dilate_extra=args.dilate,
        peelcount=args.peelcount,
        peelthr=args.peelthr,
        device=device,
        verbose=verb >= 1,
    )

    # Save as short integer (0/1)
    mask_out = mask.short()
    with spinner(f"Writing {Path(args.prefix).name}"):
        save_image(mask_out, args.prefix, header_info=header)

    if verb >= 1:
        n_vox = int(mask.sum().item())
        total = mask.numel()
        print(f"Mask: {n_vox}/{total} voxels ({100.0 * n_vox / total:.1f}%)")
        print(f"Saved: {args.prefix}")
        print(f"Time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
