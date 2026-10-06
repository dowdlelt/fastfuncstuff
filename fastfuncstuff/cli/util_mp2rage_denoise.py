"""CLI: joint INV1/INV2 denoising for MP2RAGE, then a rebuilt UNI.

Command: ffs_util_mp2rage_denoise

Usage:
    ffs_util_mp2rage_denoise -inv1 INV1.nii.gz -inv2 INV2.nii.gz -uni UNI.nii.gz -prefix sub01
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from fastfuncstuff.cli_help import FfsArgumentParser
from fastfuncstuff.cli_utils import add_device_arg, parse_prefix, setup_device

_DESCRIPTION = """\
Denoise an MP2RAGE INV1/INV2 pair jointly and rebuild UNI from the denoised pair.

UNI = INV1*INV2 / (INV1^2 + INV2^2) divides out the receive field and scales the noise
back up wherever the coil sees least, so the denoising happens before the division.
INV1 and INV2 are one anatomy with independent noise: non-local means computes ONE set
of weights from both (a patch must match in both contrasts) and applies it to both.
INV1 is denoised signed (polarity from -uni): grey matter sits at its null, where
averaging magnitudes would keep the noise floor at the cortex.

Outputs (prefix P):
  P_uni_reg        UNI with O'Brien's robust combination: dark, clean background
                   (beta = (reg_mult * sigma_INV2)^2) -- usually the one to use
  P_uni            UNI without regularisation
  P_inv1, P_inv2   denoised inversions (INV1 as magnitude; -save_signed for signed)
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = FfsArgumentParser(prog="ffs_util_mp2rage_denoise", description=_DESCRIPTION)
    req = p.add_argument_group("required")
    req.add_argument("-inv1", required=True, help="First-inversion magnitude image")
    req.add_argument("-inv2", required=True, help="Second-inversion magnitude image")
    req.add_argument("-uni", required=True, help="Scanner UNI (supplies the INV1 polarity)")
    req.add_argument("-prefix", required=True, help="Output prefix")
    k = p.add_argument_group("denoising")
    k.add_argument(
        "-beta",
        type=float,
        default=0.35,
        help="Weight decay: higher averages less-similar patches (smoother). Default 0.35",
    )
    k.add_argument(
        "-search_radius",
        type=int,
        default=2,
        help="Search window radius in voxels (default 2). 3 smears cerebellar folia at 0.7 mm.",
    )
    k.add_argument(
        "-keep",
        type=float,
        default=0.25,
        help="Fraction of the original blended back in, so tissue keeps natural grain "
        "instead of a plastic look (default 0.25; 0 = pure non-local means)",
    )
    k.add_argument(
        "-reg_mult",
        type=float,
        default=2.0,
        help="O'Brien regularisation strength for P_uni_reg, in units of the INV2 noise. "
        "Higher darkens the background more but also darkens grey matter (2: ~14 counts).",
    )
    o = p.add_argument_group("execution")
    o.add_argument("-save_signed", action="store_true", help="Also write the signed INV1")
    o.add_argument("-save_neff", action="store_true", help="Write the effective N averaged")
    add_device_arg(o)
    o.add_argument("-quiet", action="store_true")
    return p.parse_args(sys.argv[1:] if argv is None else argv)


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    from fastfuncstuff.processing.io import load_image, save_image
    from fastfuncstuff.processing.mp2rage import denoise_mp2rage

    args = parse_args(argv)
    device = setup_device(args.device)
    pinfo = parse_prefix(args.prefix)
    stem, ext = pinfo.stem, pinfo.nifti_ext
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    inv1, hdr = load_image(args.inv1, device=device)
    inv2, hdr2 = load_image(args.inv2, device=device)
    uni, hdr3 = load_image(args.uni, device=device)
    for name, img, h in (("-inv2", inv2, hdr2), ("-uni", uni, hdr3)):
        if img.shape != inv1.shape or not np.allclose(h["affine"], hdr["affine"], atol=1e-3):
            raise SystemExit(f"{name} must share -inv1's grid (they come from one MP2RAGE)")

    out = denoise_mp2rage(
        inv1,
        inv2,
        uni,
        beta=args.beta,
        search_radius=args.search_radius,
        keep=args.keep,
        reg_mult=args.reg_mult,
    )

    def save(arr, name: str) -> None:
        save_image(arr, f"{stem}_{name}{ext}", header_info=hdr, affine=hdr["affine"])

    save(out["uni_reg"].clamp(0, 4095), "uni_reg")
    save(out["uni"].clamp(0, 4095), "uni")
    save(out["inv1_signed"].abs(), "inv1")
    save(out["inv2"], "inv2")
    if args.save_signed:
        save(out["inv1_signed"], "inv1_signed")
    if args.save_neff:
        save(out["neff"], "neff")
    if not args.quiet:
        s1, s2 = out["sigma"].tolist()
        print(
            f"ffs_util_mp2rage_denoise: sigma INV1 {s1:.2f}, INV2 {s2:.2f}; k {out['k']:.4f}; "
            f"reg beta {out['reg']:.0f}; {time.time() - t0:.1f}s -> {stem}_*{ext}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
