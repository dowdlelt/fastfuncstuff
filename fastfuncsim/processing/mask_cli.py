"""CLI for GPU automask creation.

Command: automask (registered as entry point in pyproject.toml)

Usage:
    automask -input vol.nii -prefix mask.nii [-clip_frac 0.3] [-dilate 2] [-device cuda]
"""

from __future__ import annotations

import argparse
import time

import torch

from .io import load_image, save_image
from .mask import automask


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="automask",
        description="Create a binary brain mask from a 3D volume (GPU-accelerated)",
    )
    parser.add_argument("-input", required=True, help="Input volume (.nii/.nii.gz)")
    parser.add_argument("-prefix", required=True, help="Output mask file")
    parser.add_argument("-clip_frac", type=float, default=0.3,
                        help="Fraction of clip level for threshold (default: 0.3)")
    parser.add_argument("-dilate", type=int, default=2,
                        help="Extra dilation iterations (default: 2)")
    parser.add_argument("-device", default=None, help="PyTorch device (cuda, mps, cpu)")
    parser.add_argument("-verb", type=int, default=1, choices=[0, 1],
                        help="Verbosity (0/1)")

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
        print(f"automask: device={device}")

    t0 = time.time()
    vol, header = load_image(args.input, device=device)

    # Handle 4D: use first volume
    if vol.ndim == 4:
        if verb >= 1:
            print(f"4D input ({vol.shape[0]} volumes), using first volume")
        vol = vol[0]

    if verb >= 1:
        print(f"Input: {args.input} {vol.shape}")

    mask = automask(vol, clip_frac=args.clip_frac, dilate_extra=args.dilate, device=device)

    # Save as short integer (0/1)
    mask_out = mask.short()
    save_image(mask_out, args.prefix, header_info=header)

    if verb >= 1:
        n_vox = int(mask.sum().item())
        total = mask.numel()
        print(f"Mask: {n_vox}/{total} voxels ({100.0 * n_vox / total:.1f}%)")
        print(f"Saved: {args.prefix}")
        print(f"Time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
