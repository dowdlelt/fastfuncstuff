"""CLI for GPU-accelerated affine/rigid alignment.

Command: allineate (registered as entry point in pyproject.toml)

Usage:
    allineate -base ref.nii -source mov.nii -prefix out.nii -1Dmatrix_save mat.aff12.1D

Speed presets:
    -fast       : Skip Powell polish, fewer iterations (~2x faster)
    -superfast  : Skip coarse search + Powell, minimal Adam (~5x faster, small motion only)
    Default is balanced quality/speed. For maximum accuracy, use -slow.
"""

from __future__ import annotations

import argparse
import time

import torch

from fastfuncstuff.cli_utils import add_verbose_arg
from fastfuncstuff.processing.affine import (
    apply_affine,
    apply_affine_wsinc5,
    load_matrix_1D,
    save_matrix_1D,
)
from fastfuncstuff.processing.allineate import AffineAlignConfig, allineate
from fastfuncstuff.processing.io import derive_mean_output_path, load_image, save_image


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="allineate",
        description="GPU-accelerated affine/rigid alignment (inspired by 3dAllineate)",
        epilog="""Speed/quality presets:
  -superfast  Onepass, <=150 Adam iters, no Powell (small motion only)
  -fast       <=300 Adam iters, no Powell (~2x faster)
  (default)   <=400 Adam iters + 500 Powell evals (early-stops on plateau)
  -slow       300 iters at 2x, 400 at full-res, 2000 Powell evals

Examples:
  # Standard brain alignment:
  allineate -base mni.nii -source subj.nii -prefix out.nii -cost lpa

  # Fast rigid alignment (e.g., motion correction):
  allineate -base vol0.nii -source vol1.nii -prefix out.nii -rigid -fast

  # High-quality with wsinc5 output:
  allineate -base mni.nii -source subj.nii -prefix out.nii -slow -final wsinc5

  # Apply existing matrix:
  allineate -base mni.nii -source subj.nii -prefix out.nii -1Dmatrix_apply mat.aff12.1D
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Input/Output ---
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument("-base", required=True, help="Base/reference image (.nii/.nii.gz)")
    io_group.add_argument("-source", required=True, help="Source/moving image to align")
    io_group.add_argument("-prefix", required=True, help="Output aligned image")
    io_group.add_argument(
        "-1Dmatrix_save", default=None, help="Save affine matrix as .aff12.1D (AFNI format)"
    )
    io_group.add_argument(
        "-1Dmatrix_apply", default=None, help="Apply existing matrix (skip alignment)"
    )
    io_group.add_argument("-base_index", type=int, default=None, help="Use volume N from 4D base")
    io_group.add_argument(
        "-save_mean",
        action="store_true",
        help="If source is 4D, save mean output as mean_{prefix_basename}{ext}",
    )

    # --- Alignment mode ---
    mode_group = parser.add_argument_group("Alignment mode")
    mode_ex = mode_group.add_mutually_exclusive_group()
    mode_ex.add_argument("-rigid", action="store_true", help="6 DoF: translation + rotation only")
    mode_ex.add_argument(
        "-affine", action="store_true", default=True, help="12 DoF: full affine (default)"
    )
    mode_ex.add_argument(
        "-EPI", action="store_true", help="9 DoF: EPI-specific (freeze x/z scale, z shear)"
    )

    # --- Cost function ---
    cost_group = parser.add_argument_group("Cost function")
    cost_group.add_argument(
        "-cost",
        choices=["ls", "lpa", "lpc", "lps", "lpsc", "mi", "nmi", "je", "hel", "cru", "cra", "crm"],
        default="lpa",
        help="Cost function (AFNI-faithful unless noted): "
        "ls=clipped Pearson; "
        "lpa=local Pearson absolute (default, similar contrast); "
        "lpc=local Pearson signed (cross-modal, e.g. EPI-to-anat); "
        "mi/nmi=(normalized) mutual information; je=joint entropy; "
        "hel=Hellinger; cru/cra/crm=correlation ratio "
        "(unsym/additive/multiplicative). "
        "lps/lpsc=ffs-special per-voxel Gaussian local Pearson "
        "(absolute/signed).",
    )
    cost_group.add_argument(
        "-blok",
        "-bloktype",
        dest="bloktype",
        choices=["tohd", "rhdd", "cube"],
        default="tohd",
        help="Blok shape for lpa/lpc local neighborhoods (default: tohd, like 3dAllineate)",
    )
    cost_group.add_argument(
        "-blokrad",
        type=float,
        default=None,
        help="Blok radius in mm for lpa/lpc (default: auto, ~555 voxels per blok)",
    )
    cost_group.add_argument(
        "-lpa_sigma",
        type=float,
        default=4.0,
        help="Kernel parameter for lps/lpsc neighborhoods "
        "in voxels. For gauss: sigma. "
        "For box: half-width radius. "
        "Use 0 with -lpa_kernel box to auto-size "
        "to ~500 voxels (default: 4.0)",
    )
    cost_group.add_argument(
        "-lpa_kernel",
        choices=["gauss", "box"],
        default="gauss",
        help="lps/lpsc neighborhood kernel: "
        "gauss=Gaussian weighting (default), "
        "box=uniform weighting",
    )
    cost_group.add_argument(
        "-ov",
        "-overlap",
        dest="ov",
        type=float,
        default=0.0,
        help="Overlap penalty weight (AFNI lpc+/lpa+ 'ov'; default 0=off). "
        "Adds a differentiable (max(0,9.95-10*overlap))^2 term that pushes the "
        "refiner back toward full base/source overlap. Try ~0.05-0.5 if a "
        "whole-brain alignment drifts partly out of overlap.",
    )

    # --- Interpolation ---
    interp_group = parser.add_argument_group("Interpolation")
    interp_group.add_argument(
        "-interp",
        choices=["linear", "cubic"],
        default="linear",
        help="During optimization (default: linear)",
    )
    interp_group.add_argument(
        "-final",
        choices=["linear", "cubic", "wsinc5"],
        default="linear",
        dest="final_interp",
        help="For output image (default: linear). wsinc5 gives sharpest results",
    )

    # --- Masking ---
    mask_group = parser.add_argument_group("Masking")
    mask_group.add_argument(
        "-automask",
        dest="source_automask",
        action="store_true",
        help="Automask source to exclude background",
    )
    mask_group.add_argument(
        "-source_automask", dest="source_automask", action="store_true", help="Alias for -automask."
    )
    mask_group.add_argument(
        "-autoweight",
        action="store_true",
        default=True,
        help="Weight by base intensity (default: on)",
    )
    mask_group.add_argument(
        "-noautoweight", action="store_true", help="Disable intensity-based weighting"
    )
    mask_group.add_argument(
        "-save_automask", default=None, metavar="PREFIX", help="Save computed automask to PREFIX"
    )
    mask_group.add_argument(
        "-save_weight",
        "-save-weight",
        default=None,
        metavar="PREFIX",
        help="Save the exact optimisation weight (autoweight × validity × "
        "source automask) — compare to AFNI 3dAllineate -wtprefix",
    )

    # --- Search control ---
    search_group = parser.add_argument_group("Search control")
    search_group.add_argument(
        "-cmass",
        action="store_true",
        default=True,
        help="Center-of-mass pre-alignment (default: on)",
    )
    search_group.add_argument(
        "-nocmass", action="store_true", help="Disable center-of-mass pre-alignment"
    )
    search_group.add_argument(
        "-cmass_direct",
        "-cmass-direct",
        type=float,
        nargs=3,
        default=None,
        metavar=("DX", "DY", "DZ"),
        help="Manual cmass shift in base-grid voxels "
        "(same space/sign the auto cmass prints); "
        "skips automatic center-of-mass",
    )
    search_group.add_argument(
        "-save_cmass",
        "-save-cmass",
        default=None,
        metavar="PREFIX",
        help="Save the source positioned by the cmass shift "
        "alone (for checking/hand-tuning placement)",
    )
    search_group.add_argument(
        "-twopass", action="store_true", default=True, help="Coarse search + refinement (default)"
    )
    search_group.add_argument(
        "-onepass", action="store_true", help="Skip coarse search (small motion only)"
    )
    search_group.add_argument(
        "-coarse_range",
        type=float,
        default=30.0,
        help="Coarse rotation half-range in degrees (default: 30)",
    )
    search_group.add_argument(
        "-coarse_step", type=float, default=5.0, help="Coarse angular step in degrees (default: 5)"
    )
    search_group.add_argument(
        "-coarse_shift_frac",
        type=float,
        default=0.32,
        help="Coarse translation half-range as a fraction of grid size (default: 0.32, like AFNI)",
    )
    range_ex = search_group.add_mutually_exclusive_group()
    range_ex.add_argument(
        "-smallrange",
        action="store_true",
        help="Halve all coarse search ranges (angle/shift/scale)",
    )
    range_ex.add_argument(
        "-verysmallrange", action="store_true", help="Quarter all coarse search ranges"
    )
    search_group.add_argument(
        "-tbest", type=int, default=3, help="Coarse candidates to refine (default: 3)"
    )
    search_group.add_argument(
        "-nmatch",
        "-n_match",
        dest="n_match",
        type=float,
        default=0.47,
        help="Match-point budget for lpa/lpc refinement (AFNI npt_match), "
        "unit-free: <=1.0 is a FRACTION of the in-mask voxels (0.47 = AFNI "
        "default, 1.0 = all); >1.0 is an absolute count (e.g. 150000). The cost "
        "is evaluated on that many random weight-domain points per iteration "
        "instead of the full grid.",
    )
    search_group.add_argument(
        "-noautocrop", action="store_true", help="Disable auto-cropping of zero margins"
    )

    # --- Speed/quality presets ---
    speed_group = parser.add_argument_group("Speed/quality")
    speed_ex = speed_group.add_mutually_exclusive_group()
    speed_ex.add_argument(
        "-superfast", action="store_true", help="Minimal: onepass, <=150 Adam iters, no Powell"
    )
    speed_ex.add_argument(
        "-fast", action="store_true", help="Quick: <=300 Adam iters, no Powell polish"
    )
    speed_ex.add_argument(
        "-slow", action="store_true", help="Thorough: <=600/800 Adam iters, 2000 Powell evals"
    )
    speed_group.add_argument(
        "-lr",
        "-adam_lr",
        dest="adam_lr",
        type=float,
        default=0.005,
        help="Adam learning rate for full-res refinement "
        "(default: 0.005; higher converges faster but "
        "can overshoot to a worse optimum)",
    )

    # --- Hardware ---
    hw_group = parser.add_argument_group("Hardware")
    hw_group.add_argument(
        "-device", default=None, help="PyTorch device: cuda, mps, cpu (auto-detected)"
    )
    add_verbose_arg(hw_group, default=1)

    args = parser.parse_args(argv)
    return args


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point for allineate."""
    args = parse_args(argv)

    # --- Device selection ---
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
        print(f"allineate: device={device}")

    # --- Load images ---
    t0 = time.time()
    base, base_header = load_image(args.base, device=device)
    source, source_header = load_image(args.source, device=device)

    # Handle 4D base
    if base.ndim == 4:
        idx = args.base_index if args.base_index is not None else 0
        if verb >= 1:
            print(f"Using volume {idx} from 4D base ({base.shape[0]} volumes)")
        base = base[idx]

    # Handle 4D source
    source_4d = None
    if source.ndim == 4:
        source_4d = source
        source = source[0]
        if verb >= 1:
            print(f"Source is 4D ({source_4d.shape[0]} volumes), aligning first volume")

    if verb >= 1:
        print(f"Base: {args.base} {base.shape}")
        print(f"Source: {args.source} {source.shape}")
        print(f"Load time: {time.time() - t0:.2f}s")

    # --- Apply existing matrix ---
    if getattr(args, "1Dmatrix_apply", None) is not None:
        matrix_path = getattr(args, "1Dmatrix_apply")
        if verb >= 1:
            print(f"Applying matrix from {matrix_path}")

        matrix = load_matrix_1D(
            matrix_path,
            base_affine=base_header["affine"],
            source_affine=source_header["affine"],
        )
        matrix = matrix.to(device)
        if args.final_interp == "wsinc5":
            warped = apply_affine_wsinc5(source, matrix, base.shape)
        else:
            warped = apply_affine(source, matrix, base.shape, zero_outside=True)
        save_image(warped, args.prefix, header_info=base_header)
        if verb >= 1:
            print(f"Saved: {args.prefix}")
        return

    # --- Build config with speed/quality presets ---
    dof = "affine"
    if args.rigid:
        dof = "rigid"
    elif args.EPI:
        dof = "epi"

    # Iteration counts are *ceilings*: Adam stops early once the cost plateaus
    # (relative tolerance), so a generous cap costs nothing when converged but
    # lets hard cases keep improving instead of cutting off mid-descent.
    adam_iters_2x = 300
    adam_iters_1x = 400
    powell_maxfev = 500
    twopass = not args.onepass

    # Apply presets
    if args.superfast:
        adam_iters_2x = 150
        adam_iters_1x = 150
        powell_maxfev = 0
        twopass = False
        if verb >= 1:
            print("Mode: superfast (onepass, <=150 iters, no Powell)")
    elif args.fast:
        adam_iters_2x = 300
        adam_iters_1x = 300
        powell_maxfev = 0
        if verb >= 1:
            print("Mode: fast (<=300 iters, no Powell)")
    elif args.slow:
        adam_iters_2x = 600
        adam_iters_1x = 800
        powell_maxfev = 2000
        if verb >= 1:
            print("Mode: slow (<=600/800 iters, 2000 Powell evals)")

    # Auto-size box radius if requested
    lpa_sigma = args.lpa_sigma
    if args.lpa_kernel == "box" and lpa_sigma <= 0:
        from fastfuncstuff.processing.cost import auto_box_radius

        lpa_sigma = float(auto_box_radius(500))
        if verb >= 1:
            side = 2 * int(lpa_sigma) + 1
            print(f"Auto box radius: {int(lpa_sigma)} ({side}³ = {side**3} voxels)")

    config = AffineAlignConfig(
        dof=dof,
        cost=args.cost,
        lpa_sigma=lpa_sigma,
        lpa_kernel=args.lpa_kernel,
        bloktype=args.bloktype,
        blokrad=args.blokrad,
        ov=args.ov,
        n_match=args.n_match,
        twopass=twopass,
        coarse_range=args.coarse_range,
        coarse_step=args.coarse_step,
        coarse_shift_frac=args.coarse_shift_frac,
        range_scale=(0.25 if args.verysmallrange else 0.5 if args.smallrange else 1.0),
        adam_lr=args.adam_lr,
        tbest=args.tbest,
        adam_iters_2x=adam_iters_2x,
        adam_iters_1x=adam_iters_1x,
        powell_maxfev=powell_maxfev,
        cmass=not args.nocmass,
        cmass_direct=(tuple(args.cmass_direct) if args.cmass_direct is not None else None),
        interp=args.interp,
        final_interp=args.final_interp,
        source_automask=args.source_automask,
        autoweight=not args.noautoweight,
        autocrop=not args.noautocrop,
        device=str(device),
        verb=verb,
    )

    # --- Run alignment ---
    t1 = time.time()
    matrix, warped = allineate(
        base,
        source,
        config,
        base_header=base_header,
        source_header=source_header,
        save_automask_path=args.save_automask,
        save_cmass_path=args.save_cmass,
        save_weight_path=args.save_weight,
    )

    if verb >= 1:
        print(f"Alignment time: {time.time() - t1:.2f}s")

    # --- Save outputs ---
    save_image(warped, args.prefix, header_info=base_header)
    if verb >= 1:
        print(f"Saved: {args.prefix}")

    matrix_save_path = getattr(args, "1Dmatrix_save", None)
    if matrix_save_path is not None:
        save_matrix_1D(
            matrix,
            matrix_save_path,
            base_affine=base_header["affine"],
            source_affine=source_header["affine"],
        )
        if verb >= 1:
            print(f"Saved matrix: {matrix_save_path}")

    # --- Apply to all 4D volumes ---
    if source_4d is not None:
        if verb >= 1:
            print(f"Applying alignment to all {source_4d.shape[0]} volumes...")
        aligned_vols = []
        for t in range(source_4d.shape[0]):
            if args.final_interp == "wsinc5":
                vol = apply_affine_wsinc5(source_4d[t], matrix, base.shape)
            else:
                vol = apply_affine(source_4d[t], matrix, base.shape, zero_outside=True)
            aligned_vols.append(vol)
        result_4d = torch.stack(aligned_vols)
        save_image(result_4d, args.prefix, header_info=base_header)
        if verb >= 1:
            print(f"Saved 4D result: {args.prefix}")

        if args.save_mean:
            mean_path = derive_mean_output_path(args.prefix)
            mean_image = result_4d.mean(dim=0)
            save_image(mean_image, mean_path, header_info=base_header)
            if verb >= 1:
                print(f"Saved mean: {mean_path}")
    elif args.save_mean and verb >= 1:
        print("-save_mean requested, but source is not 4D; skipping mean output")

    if verb >= 1:
        print(f"Total time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
