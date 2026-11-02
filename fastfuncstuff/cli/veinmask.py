"""ffs_veinmask — vein candidates from an EPI mean and anat tissue fractions.

For ffs_phasereg -spr -spr_mask when there is no GRE/SWI: see
:mod:`fastfuncstuff.processing.veinmask` for the model.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from fastfuncstuff.cli_help import FfsArgumentParser
from fastfuncstuff.cli_utils import (
    add_device_arg,
    add_verbose_arg,
    input_nifti_ext,
    parse_prefix,
    setup_device,
    spinner,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = FfsArgumentParser(
        prog="ffs_veinmask",
        description=(
            "Vein map from the GE-EPI mean and the anat's GM/WM/CSF fractions, for "
            "ffs_phasereg -spr_mask and as a QC map, when there is no GRE/SWI. The tissue "
            "mix predicts each EPI voxel; voxels in cortex (or in CSF touching it) much "
            "darker than predicted are veins: a vein darkens at the EPI's echo time, "
            "while white matter and tissue boundaries are predicted. With -model local "
            "(default) the tissue levels are fitted per neighbourhood, and the mask "
            "follows continuous dark ridges (plates: sulci and the brain edge; tubes: "
            "vessels through slices; blobs: punctate) without dilating. Outputs "
            "(PREFIX_): mask, ratio (EPI / expected; 1 = as predicted), expected, region "
            "(where veins are looked for); local also thin (dark-ridge strength) and "
            "shape (1 plate, 2 tube, 3 blob, inside the mask). PROTOTYPE, judged by eye "
            "on ds003427 (2026-10-14); the GRE label it was checked against is noisy."
        ),
    )
    p.add_argument(
        "-epi",
        nargs="+",
        required=True,
        help="EPI mean(s) or runs (4-D: the temporal mean is taken), averaged. Use the "
        "runs phase regression will correct: a spin-echo run has no vein darkening",
    )
    p.add_argument(
        "-tissue",
        nargs="+",
        required=True,
        help="GM, WM, CSF fractions: one 4-D file whose first three volumes are GM, WM, "
        "CSF (SPM c1/c2/c3 order, a subject TPM), or three 3-D files in that order. Any "
        "grid: resampled onto the EPI's through the headers (smoothed to the EPI voxel "
        "size first when finer, so the fractions stay partial-volume fractions)",
    )
    p.add_argument(
        "-prefix",
        required=True,
        help="output prefix; an extension (.nii, .nii.gz, .nii.zst) sets the format, "
        "else the first -epi's format is kept",
    )
    p.add_argument(
        "-mask",
        help="brain mask on the EPI grid (default: EPI above 20%% of its 98th percentile)",
    )
    p.add_argument(
        "-model",
        choices=["local", "global"],
        default="local",
        help="'local': GM/WM/CSF levels fitted per voxel over -local_fwhm, so shading "
        "and drifting tissue contrast do not read as darkness, plus the shape-aware "
        "ridge-following mask. 'global': one level per tissue times a -bias_fwhm "
        "shading field and a plain cut (the 2026-10-12 model; its baseline drifts "
        "across the slab)",
    )
    p.add_argument(
        "-local_fwhm",
        type=float,
        default=4.0,
        help="-model local: FWHM (mm) of the neighbourhood the tissue levels are fitted "
        "over. Smaller absorbs wide dark plates into the reference; larger lets "
        "shading back in",
    )
    p.add_argument(
        "-grow",
        type=float,
        default=0.15,
        help="-model local: how much darker than predicted a continuation of a vein "
        "must be (0.15 = 15%%) to join it: only along the ridge centre line and on a "
        "real ridge, so it follows a dark structure rather than dilating it. 0 = no "
        "growth (plain -darker cut)",
    )
    p.add_argument(
        "-darker",
        type=float,
        default=0.3,
        help="seed voxels this much darker than predicted (0.3 = 30%% darker). Judged "
        "by eye on ds003427: with -model local it seeds 8-12%% of cortex+pial CSF",
    )
    p.add_argument(
        "-dilate",
        type=int,
        default=0,
        help="grow the mask this many voxels (6-connected). Default 0: once the anat "
        "is registered to the EPI, growing only smears the mask (a 3x3x3 growth "
        "took ds003427's GRE vein mask from 7-12%% of the brain to 40-54%%)",
    )
    p.add_argument(
        "-bias_fwhm",
        type=float,
        default=24.0,
        help="-model global: FWHM (mm) of the bias field. Smaller follows coil shading more closely "
        "but starts to absorb clusters of veins into the bias",
    )
    p.add_argument(
        "-gm_min",
        type=float,
        default=0.3,
        help="GM fraction from which a voxel counts as cortex (the search region also "
        "takes CSF-dominant voxels touching GM >= 0.5: the pial veins)",
    )
    add_device_arg(p)
    add_verbose_arg(p)
    return p.parse_args(argv)


def _tissue_on_grid(paths, epi_shape, epi_affine, device) -> torch.Tensor:
    from fastfuncstuff.processing.grid import resample_to_grid
    from fastfuncstuff.processing.io import load_image

    if len(paths) == 1:
        t, h = load_image(paths[0], device=device)
        if t.ndim != 4 or t.shape[0] < 3:
            raise SystemExit("ffs_veinmask: a single -tissue file needs >= 3 volumes (GM, WM, CSF)")
        t = t[:3]
    elif len(paths) == 3:
        vols = [load_image(q, device=device) for q in paths]
        t, h = torch.stack([v[0] for v in vols]), vols[0][1]
    else:
        raise SystemExit("ffs_veinmask: -tissue takes one 4-D file or three 3-D files")
    t = t.float()
    src_aff = np.asarray(h["affine"], np.float64)
    if t.shape[1:] == epi_shape and np.allclose(src_aff, epi_affine, atol=1e-4):
        return t
    # Finer tissue maps are blurred to the EPI voxel first: point sampling a 0.3 mm
    # map at 0.8 mm spacing would turn fractions back into near-binary labels.
    from fastfuncstuff.processing.veinmask import masked_blur

    d_src = np.linalg.norm(src_aff[:3, :3], axis=0)
    d_epi = float(np.cbrt(abs(np.linalg.det(epi_affine[:3, :3]))))
    sig = [math.sqrt(max(d_epi**2 - d**2, 0.0)) / 2.3548 / d for d in d_src]
    one = torch.ones(t.shape[1:], device=t.device)
    t = torch.stack([masked_blur(v, one, (sig[2], sig[1], sig[0])) for v in t])
    return resample_to_grid(t, src_aff, epi_shape, epi_affine, interp="linear")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = setup_device(args.device, float64_only="ffs_veinmask")
    t0 = time.time()
    from fastfuncstuff.processing.io import load_image, save_image
    from fastfuncstuff.processing.veinmask import (
        _dilate6,
        dark_shapes,
        local_tissue_darkness,
        ridge_follow_mask,
        tissue_darkness,
        vein_mask,
        vein_region,
    )

    with spinner("Reading the EPI", enabled=args.verb >= 1):
        means, header = [], None
        for path in args.epi:
            v, h = load_image(path, device=device)
            means.append(v.float().mean(0) if v.ndim == 4 else v.float())
            header = header or h
        assert header is not None
        mean = torch.stack(means).mean(0)
    epi_aff = np.asarray(header["affine"], np.float64)
    shape = tuple(mean.shape)
    with spinner("Tissue fractions onto the EPI grid", enabled=args.verb >= 1):
        tissue = _tissue_on_grid(args.tissue, shape, epi_aff, device)
    if args.mask:
        brain = load_image(args.mask, device=device)[0] > 0.5
    else:
        brain = mean > 0.2 * torch.quantile(mean.flatten()[::7], 0.98)
    brain &= tissue.sum(0) > 0.5

    zooms = header["header"].get_zooms()[:3]  # x, y, z; the volume is (z, y, x)
    zyx = tuple(float(z) for z in reversed(zooms))
    region = vein_region(tissue, brain, gm_min=args.gm_min)
    extra: dict[str, torch.Tensor] = {}
    if args.model == "global":
        sig = tuple(args.bias_fwhm / 2.3548 / z for z in zyx)
        ratio, expected, levels = tissue_darkness(mean, tissue, brain, sig)
        mask = vein_mask(ratio, region, args.darker, args.dilate)
    else:
        sig = tuple(args.local_fwhm / 2.3548 / z for z in zyx)
        with spinner("Local tissue fit", enabled=args.verb >= 1):
            ratio, expected = local_tissue_darkness(mean, tissue, brain, sig)
        with spinner("Ridge shapes", enabled=args.verb >= 1):
            thin, shape, ridge = dark_shapes(ratio, brain, zyx)
        mask = ridge_follow_mask(
            ratio, region, thin, ridge, seed=1.0 - args.darker, grow=1.0 - args.grow
        )
        for _ in range(args.dilate):
            mask = _dilate6(mask)
        extra = {"thin": thin, "shape": torch.where(mask, shape, torch.zeros_like(shape))}
    core = region & (ratio < 1.0 - args.darker)

    if args.verb >= 1:
        if args.model == "global":
            lv = levels.tolist()
            print(f"  tissue levels before bias: GM {lv[0]:.0f}, WM {lv[1]:.0f}, CSF {lv[2]:.0f}")
        nreg = max(int(region.sum()), 1)
        print(
            f"  region {nreg} voxels; {int(core.sum())} ({float(core.sum()) / nreg:.1%}) > "
            f"{args.darker:g} darker than predicted; mask {int(mask.sum())} "
            f"({float(mask.sum()) / nreg:.1%})"
        )
        if extra:
            sh = extra["shape"][mask]
            print(
                "  mask shapes: "
                + ", ".join(
                    f"{n} {float((sh == v).float().mean()):.0%}"
                    for v, n in ((1, "plate"), (2, "tube"), (3, "blob"))
                )
            )
    outs = {
        "mask": mask.to(torch.uint8),
        "ratio": ratio,
        "expected": expected,
        "region": region.to(torch.uint8),
        **extra,
    }
    pfx = parse_prefix(args.prefix, default_ext=input_nifti_ext(args.epi[0]))
    for name, vol in outs.items():
        save_image(vol, f"{pfx.stem}_{name}{pfx.nifti_ext}", header_info=header)
    if args.verb >= 1:
        print(f"  wrote {pfx.stem}_{{{','.join(outs)}}}{pfx.nifti_ext} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
