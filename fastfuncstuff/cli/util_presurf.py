"""CLI: MP2RAGE → FreeSurfer-ready image (the presurfer workflow, no MATLAB).

Command: ffs_util_presurf

Usage:
    ffs_util_presurf -uni UNI.nii.gz -inv2 INV2.nii.gz -tpm TPM.nii -prefix sub01_presurf
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.cli_help import FfsArgumentParser
from fastfuncstuff.cli_utils import add_device_arg, parse_prefix, setup_device

_DESCRIPTION = """\
Prepare an MP2RAGE UNI/INV2 pair for recon-all: the presurfer workflow (Kashyap,
github.com/srikash/presurfer) on ffs_segment, with no MATLAB/SPM.

  1. Segment INV2 -> bias-corrected INV2 + tissue classes.
  2. MPRAGEise:  UNI * scale01(INV2_biascorrected)  -- suppresses UNI's amplified
     background noise.
  3. stripmask from INV2:  1 - ((c3+c4+c5+c6) > 0.5)
  4. Segment the MPRAGEised UNI -> brainmask (c1+c2+c3) > 0.3, WMmask c2 > 0.5.
  5. MPRAGEised * stripmask -> the image to give recon-all.

The _raw masks are presurfer's arithmetic exactly. The cleaned masks also drop voxels
the warped template says cannot be brain (-prior_gate: removes the eyes, which INV2
can label as tissue), keep the largest connected component, and fill enclosed holes.
_removed shows what the cleaning took away -- what would otherwise be hand-edited.

-denoise (needs -inv1) first denoises INV1/INV2 jointly (ffs_util_mp2rage_denoise) and
uses the denoised INV2 for its segmentation. The recon-all image is then either
  reg     the rebuilt UNI with O'Brien's robust combination (dark, clean background)
  mprage  the rebuilt UNI, MPRAGEised by the bias-corrected denoised INV2

Segmentation settings default to presurfer's (biasreg 0.001, biasfwhm 30, stiffer
warp reg, samp 2, ngaus 2 2 2 3 4 2), not ffs_segment's.

Outputs (prefix P):
  P_MPRAGEised, P_MPRAGEised_stripped (the recon-all input; P_UNIreg* with -denoise reg)
  P_stripmask, P_stripmask_raw, P_stripmask_removed
  P_brainmask, P_brainmask_raw, P_brainmask_removed, P_WMmask
  P_inv2_biascorrected, P_uni_biascorrected
  P_inv2_cN, P_uni_cN (tissue classes; skip with -no_classes)
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = FfsArgumentParser(prog="ffs_util_presurf", description=_DESCRIPTION)
    req = p.add_argument_group("required")
    req.add_argument("-uni", required=True, help="MP2RAGE UNI (uniform / T1-weighted) image")
    req.add_argument("-inv2", required=True, help="MP2RAGE second-inversion magnitude image")
    req.add_argument("-tpm", required=True, help="4-D tissue-probability template (SPM TPM.nii)")
    req.add_argument("-prefix", required=True, help="Output prefix")
    req.add_argument("-inv1", default=None, help="MP2RAGE first-inversion magnitude (for -denoise)")

    d = p.add_argument_group("denoising (needs -inv1)")
    d.add_argument(
        "-denoise",
        default="off",
        choices=("off", "reg", "mprage"),
        help="Jointly denoise INV1/INV2 first. 'reg': recon-all image is the rebuilt UNI with "
        "O'Brien regularisation (clean dark background). 'mprage': the rebuilt UNI "
        "MPRAGEised. 'off' (default): presurfer as published.",
    )
    d.add_argument("-dn_beta", type=float, default=0.35, help="Denoising strength (default 0.35)")
    d.add_argument("-dn_search_radius", type=int, default=2, help="Search radius (default 2)")
    d.add_argument(
        "-dn_keep", type=float, default=0.25, help="Original blended back in (default 0.25)"
    )
    d.add_argument(
        "-dn_reg_mult", type=float, default=2.0, help="O'Brien strength, INV2-noise units (2)"
    )

    m = p.add_argument_group("masks")
    m.add_argument(
        "-strip_thresh",
        type=float,
        default=0.5,
        help="stripmask = NOT (sum of the non-GM/WM classes > this) (default 0.5)",
    )
    m.add_argument(
        "-brain_thresh",
        type=float,
        default=0.3,
        help="brainmask = (c1+c2+c3) > this; presurfer's deliberately liberal 0.3",
    )
    m.add_argument("-wm_thresh", type=float, default=0.5, help="WMmask = c2 > this")
    m.add_argument(
        "-prior_gate",
        type=float,
        default=0.05,
        metavar="P",
        help="Drop mask voxels whose warped-template brain probability (GM+WM+CSF) is <= P. "
        "Removes the eyes regardless of how they connect to the brain. 0 disables.",
    )
    m.add_argument(
        "-open_radius",
        type=int,
        default=0,
        metavar="VOX",
        help="Also cut bridges thinner than ~2*VOX voxels with a morphological opening "
        "before keeping the largest component (default 0 = off).",
    )
    m.add_argument("-no_cluster", action="store_true", help="Don't keep only the largest component")
    m.add_argument("-no_fill", action="store_true", help="Don't fill enclosed holes")
    m.add_argument(
        "-norm",
        default="minmax",
        choices=("minmax", "robust"),
        help="INV2 scaling for MPRAGEise: 'minmax' (presurfer's mat2gray) or 'robust' "
        "(99.9th percentile as the top, so one hot voxel can't darken the image).",
    )
    m.add_argument(
        "-no_mprageise",
        action="store_true",
        help="Segment and strip UNI as-is (presurfer's step 0 is optional)",
    )

    s = p.add_argument_group("segmentation (presurfer's settings by default)")
    s.add_argument("-ngaus", type=int, nargs="+", default=[2, 2, 2, 3, 4, 2])
    s.add_argument("-biasreg", type=float, default=0.001)
    s.add_argument("-biasfwhm", type=float, default=30.0)
    s.add_argument("-reg", type=float, nargs="+", default=[0.0, 0.001, 0.5, 0.05, 0.2])
    s.add_argument("-samp", type=float, default=2.0)
    s.add_argument(
        "-affreg",
        default="mni",
        choices=("mni", "imni", "eastern", "subj", "rigid", "none", "off"),
        help="Zoom/shear prior for the automatic affine to the TPM (as ffs_segment)",
    )
    s.add_argument("-mrf", type=float, default=1.0)
    s.add_argument("-cleanup", type=int, default=1, choices=(0, 1, 2))

    o = p.add_argument_group("execution")
    o.add_argument("-no_classes", action="store_true", help="Don't write the cN maps")
    add_device_arg(o)
    o.add_argument("-quiet", action="store_true")
    return p.parse_args(sys.argv[1:] if argv is None else argv)


def main(argv: list[str] | None = None) -> int:
    from fastfuncstuff.cli.segment import _dither_step
    from fastfuncstuff.processing.io import load_image, save_image
    from fastfuncstuff.processing.mp2rage import denoise_mp2rage
    from fastfuncstuff.processing.presurf import (
        class_sum_mask,
        clean_mask,
        mprageise,
        segment_image,
        strip_mask,
        warped_brain_prior,
    )
    from fastfuncstuff.processing.segment import load_tpm

    args = parse_args(argv)
    device = setup_device(args.device)
    if device.type == "mps":
        raise SystemExit("ffs_util_presurf: segmentation needs float64 geometry; use -device cpu")
    verbose = not args.quiet
    pinfo = parse_prefix(args.prefix)
    stem, ext = pinfo.stem, pinfo.nifti_ext
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    uni, uni_hdr = load_image(args.uni, device=device)
    inv2, inv2_hdr = load_image(args.inv2, device=device)
    if uni.shape != inv2.shape or not np.allclose(uni_hdr["affine"], inv2_hdr["affine"], atol=1e-3):
        raise SystemExit("-uni and -inv2 must share a grid (they come from one MP2RAGE)")
    if args.denoise != "off" and args.inv1 is None:
        raise SystemExit("-denoise needs -inv1")
    if args.denoise != "off" and args.no_mprageise:
        raise SystemExit("-denoise already picks the recon-all image; drop -no_mprageise")
    affine = torch.as_tensor(uni_hdr["affine"], dtype=torch.float64, device=device)
    log_prior, tpm_affine, bg_low, bg_high, _ = load_tpm(
        args.tpm, add_background="no", device=device, verbose=verbose
    )
    if len(args.ngaus) != log_prior.shape[0]:
        raise SystemExit(f"-ngaus has {len(args.ngaus)} entries; the TPM has {log_prior.shape[0]}")
    shape = tuple(uni.shape)
    seg_kw = dict(
        ngaus=args.ngaus,
        biasreg=args.biasreg,
        biasfwhm=args.biasfwhm,
        reg=tuple(args.reg),
        samp=args.samp,
        affreg=args.affreg,
        mrf=args.mrf,
        cleanup=args.cleanup,
        device=device,
        verbose=verbose,
    )

    def save(arr: torch.Tensor, name: str, hdr=uni_hdr) -> None:
        if arr.dtype == torch.bool:
            arr = arr.to(torch.uint8)
        save_image(arr, f"{stem}_{name}{ext}", header_info=hdr, affine=hdr["affine"])

    def gate(fit: dict) -> torch.Tensor | None:
        if args.prior_gate <= 0:
            return None
        return warped_brain_prior(log_prior, tpm_affine, fit, shape, device=device)

    clean_kw = dict(
        prior_thresh=args.prior_gate,
        open_radius=args.open_radius,
        largest_cluster=not args.no_cluster,
        fill_holes=not args.no_fill,
    )

    rebuilt = None
    inv2_dither = _dither_step(inv2_hdr, "auto")
    if args.denoise != "off":
        if verbose:
            print("== Jointly denoising INV1/INV2")
        inv1, inv1_hdr = load_image(args.inv1, device=device)
        if inv1.shape != inv2.shape:
            raise SystemExit("-inv1 must share -inv2's grid")
        dn = denoise_mp2rage(
            inv1, inv2, uni,
            beta=args.dn_beta, search_radius=args.dn_search_radius,
            keep=args.dn_keep, reg_mult=args.dn_reg_mult,
        )  # fmt: skip
        del inv1
        inv2 = dn["inv2"]
        inv2_dither = 0.0  # denoised values are continuous, no integer lattice to dither
        rebuilt = (dn["uni_reg"] if args.denoise == "reg" else dn["uni"]).clamp(0, 4095)
        save(inv2, "inv2_denoised", inv2_hdr)
        if verbose:
            s1, s2 = dn["sigma"].tolist()
            print(f"  sigma INV1 {s1:.2f}, INV2 {s2:.2f}; k {dn['k']:.4f}")
        del dn

    # ── INV2: bias field + stripmask ──
    if verbose:
        print("== Segmenting INV2")
    fit2, out2 = segment_image(
        inv2, affine, log_prior, tpm_affine, bg_low, bg_high,
        dither=inv2_dither, **seg_kw,
    )  # fmt: skip
    save(out2["corrected"], "inv2_biascorrected", inv2_hdr)
    if not args.no_classes:
        for k in range(out2["posteriors"].shape[0]):
            save(out2["posteriors"][k], f"inv2_c{k + 1}", inv2_hdr)
    strip_raw = strip_mask(out2["posteriors"], thresh=args.strip_thresh)
    strip, strip_removed = clean_mask(strip_raw, brain_prior=gate(fit2), **clean_kw)
    save(strip_raw, "stripmask_raw")
    save(strip, "stripmask")
    save(strip_removed, "stripmask_removed")

    # ── the recon-all image, then segment it: brainmask + WMmask ──
    if args.denoise == "reg":
        assert rebuilt is not None
        image, image_name = rebuilt, "UNIreg"
    elif args.no_mprageise:
        image, image_name = uni.to(torch.float32), "UNI"
    else:
        base = uni if rebuilt is None else rebuilt
        image, image_name = mprageise(base, out2["corrected"], args.norm), "MPRAGEised"
    if image_name != "UNI":
        save(image, image_name)
    del out2, fit2
    if verbose:
        print(f"== Segmenting {image_name}")
    uni_dither = _dither_step(uni_hdr, "auto") if image_name == "UNI" else 0.0
    fit1, out1 = segment_image(
        image, affine, log_prior, tpm_affine, bg_low, bg_high,
        dither=uni_dither, **seg_kw,
    )  # fmt: skip
    post = out1["posteriors"]
    save(out1["corrected"], "uni_biascorrected")
    if not args.no_classes:
        for k in range(post.shape[0]):
            save(post[k], f"uni_c{k + 1}")
    brain_raw = class_sum_mask(post, (0, 1, 2), args.brain_thresh)
    brain, brain_removed = clean_mask(brain_raw, brain_prior=gate(fit1), **clean_kw)
    save(brain_raw, "brainmask_raw")
    save(brain, "brainmask")
    save(brain_removed, "brainmask_removed")
    save(class_sum_mask(post, (1,), args.wm_thresh), "WMmask")

    stripped = image * strip.to(device=image.device, dtype=image.dtype)
    save(stripped, f"{image_name}_stripped")

    if verbose:
        vox_ml = float(abs(np.linalg.det(uni_hdr["affine"][:3, :3]))) / 1000.0
        for name, raw, cln, rem in (
            ("stripmask", strip_raw, strip, strip_removed),
            ("brainmask", brain_raw, brain, brain_removed),
        ):
            print(
                f"  {name}: raw {raw.sum().item() * vox_ml:.0f} mL -> cleaned "
                f"{cln.sum().item() * vox_ml:.0f} mL (removed {rem.sum().item() * vox_ml:.1f} mL)"
            )
        print(f"ffs_util_presurf done in {time.time() - t0:.0f}s -> {stem}_*{ext}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
