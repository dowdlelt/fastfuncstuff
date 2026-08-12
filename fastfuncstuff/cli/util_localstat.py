"""CLI for GPU local spatial statistics (AFNI 3dLocalstat / 3dLocalACF).

Command: ffs_util_localstat (registered as entry point in pyproject.toml)

Currently implements the local spatial ACF (the job of AFNI's 3dLocalACF):
estimates ACF(r) = a*exp(-r^2/2b^2) + (1-a)*exp(-r/c) in a neighborhood around
every voxel and writes a 5-volume dataset (a, b, c, FWHM, FWQM).

Usage:
    ffs_util_localstat -input errts.nii.gz -prefix LocalACF.nii.gz \\
        -stat ACF -nbhd 'SPHERE(25)' -mask mask.nii.gz [-device cuda]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from fastfuncstuff.cli_utils import add_verbose_arg, setup_device, spinner
from fastfuncstuff.processing.io import load_image, save_image
from fastfuncstuff.stats.localstat import (
    ACF_LABELS,
    FWHM_LABELS,
    local_acf,
    local_fwhm,
)
from fastfuncstuff.utils import REGISTRATION_TF32

# Per-stat default neighborhoods (used when -nbhd is omitted).
_DEFAULT_NBHD = {"ACF": "SPHERE(-9.666)", "FWHM": "SPHERE(-2.0)"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ffs_util_localstat",
        description="Per-voxel local spatial statistics on a neighborhood (GPU, "
        "AFNI 3dLocalstat/3dLocalACF compatible).",
    )
    parser.add_argument("-input", required=True, help="Input time-series dataset")
    parser.add_argument("-prefix", required=True, help="Output dataset")
    parser.add_argument(
        "-stat",
        default="ACF",
        help="Statistic to compute: ACF (local autocorrelation model, 5 bricks) "
        "or FWHM (local smoothness, 4 bricks).",
    )
    parser.add_argument(
        "-nbhd",
        default=None,
        help="Neighborhood: SPHERE(r), RECT(a,b,c), RHDD(r), TOHD(r). Negative "
        "size = voxel-index units. Default depends on -stat: SPHERE(-9.666) for "
        "ACF (matches 3dLocalACF), SPHERE(-2.0) for FWHM.",
    )
    parser.add_argument("-mask", default=None, help="Mask dataset (strongly recommended)")
    parser.add_argument(
        "-automask",
        action="store_true",
        help="Build a brain mask from the input instead of -mask",
    )
    parser.add_argument(
        "-nomedian",
        action="store_true",
        help="ACF only: skip AFNI's 19-voxel median post-filter (FWHM is never "
        "median-filtered, matching 3dLocalstat)",
    )
    parser.add_argument(
        "-lm_iters",
        "-lm-iters",
        type=int,
        default=50,
        help="Levenberg-Marquardt iterations for the ACF fit (default: 50)",
    )
    parser.add_argument("-device", default=None, help="PyTorch device (cuda, mps, cpu)")
    add_verbose_arg(parser, default=1)
    return parser.parse_args(argv)


def _select_device(name: str | None) -> torch.device:
    return setup_device(name, tf32=REGISTRATION_TF32)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    stat = args.stat.upper()
    if stat not in _DEFAULT_NBHD:
        raise SystemExit(
            f"-stat '{args.stat}' not implemented. Supported: {', '.join(_DEFAULT_NBHD)}."
        )
    nbhd = args.nbhd if args.nbhd is not None else _DEFAULT_NBHD[stat]

    device = _select_device(args.device)
    verb = args.verb
    t0 = time.time()

    with spinner(f"Loading {Path(args.input).name}"):
        data, header = load_image(args.input, device=device)
    if stat == "ACF" and data.ndim != 4:
        raise SystemExit(f"ACF needs a 4D time series; got shape {tuple(data.shape)}.")
    if data.ndim not in (3, 4):
        raise SystemExit(f"Input must be 3D or 4D; got shape {tuple(data.shape)}.")
    nz, ny, nx = data.shape[-3:]

    # Voxel sizes in (x, y, z) mm from the NIfTI zooms.
    zooms = header["header"].get_zooms()
    voxdims = (abs(float(zooms[0])), abs(float(zooms[1])), abs(float(zooms[2])))

    mask = None
    if args.mask is not None:
        with spinner(f"Loading {Path(args.mask).name}"):
            mvol, _ = load_image(args.mask, device=device)
        if mvol.ndim == 4:
            mvol = mvol[0]
        if tuple(mvol.shape) != (nz, ny, nx):
            raise SystemExit("-mask grid does not match input grid")
        mask = mvol > 0.5
    elif args.automask:
        from fastfuncstuff.processing.mask import automask

        ref = data[0] if data.ndim == 4 else data
        mask = automask(ref, device=device, verbose=verb >= 1) > 0

    if verb >= 1:
        print(f"ffs_util_localstat: stat={stat} input={args.input} {tuple(data.shape)}")
        print(f"  voxel sizes (mm) = {voxdims}")

    if stat == "ACF":
        out = local_acf(
            data,
            voxdims=voxdims,
            nbhd=nbhd,
            mask=mask,
            device=device,
            do_median=not args.nomedian,
            lm_iters=args.lm_iters,
            verbose=verb,
        )
        labels = ACF_LABELS
    else:  # FWHM
        # 3dLocalstat does not median-filter (that is a 3dLocalACF step); with it
        # off the maps match AFNI to machine precision.
        out = local_fwhm(
            data,
            voxdims=voxdims,
            nbhd=nbhd,
            mask=mask,
            device=device,
            do_median=False,
            verbose=verb,
        )
        labels = FWHM_LABELS

    with spinner(f"Writing {Path(args.prefix).name}"):
        save_image(out, args.prefix, header_info=header, brick_labels=list(labels))

    if verb >= 1:
        print(f"  brick labels: {', '.join(labels)}")
        print(f"Saved: {args.prefix}")
        print(f"Time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
