"""CLI for MotSim: Motion-simulation regressors (Patriat, Reynolds & Birn 2017).

See fastfuncstuff.processing.motsim for the library implementation.

ffs_moco runs the same code in-line via its own ``-motsim`` flag, which is the
cheaper route: it already holds the base volume and the per-volume matrices, and
its backward pass inherits the settings of the correction that actually ran. This
tool is for the after-the-fact case — regressors from motion someone else
estimated, or a second model over motion you already have on disk.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter, suggest
from fastfuncstuff.cli_utils import add_device_arg, add_verbose_arg, setup_device, spinner
from fastfuncstuff.processing.io import load_image, save_image
from fastfuncstuff.processing.motsim import (
    load_dfile,
    load_motion_1d,
    motsim_regressors,
    params_to_voxel_matrices,
    parse_motsim_spec,
    save_1d,
)
from fastfuncstuff.processing.nwarpforge import load_affine_1D
from fastfuncstuff.utils import REGISTRATION_TF32

EPILOG = """\
model spec (-model MODE[,N]):
  MODE   forward   the simulated series itself (MotSim). No second registration
                   pass, so roughly half the cost of 'both' and not measurably
                   worse in the paper.
         backward  that series re-registered (MotSimReg): what a real correction
                   leaves behind — interpolation error and motion-estimation error.
         both      forward and backward spatially concatenated, then one PCA.
  N      an integer    exactly N components.
         0 < N < 1     however many reach that fraction of the simulated series'
                       variance. The paper's medians for 'both': 5 PCs -> 90%,
                       7 -> 95%, 16 -> 99%.
         omitted       12.

  The paper's four models spell out as: both,12 (12Both) · both,24 (24Both) ·
  forward,12 (12Forw) · backward,12 (12Back). 12 was chosen only to match the
  regressor count of the 6-params-plus-derivatives model it competed with;
  explained variance asymptotes to the slope of random regressors at ~12-15.

examples:
  # the paper's headline model, from an ffs_moco run you already have
  ffs_motsim -base epi_mc_mean.nii.gz -aff12 epi_mc.aff12.1D \\
             -model both,12 -prefix epi

  # cheap variant, components chosen by variance rather than by count
  ffs_motsim -base epi_mc_mean.nii.gz -1Dfile motion.1D \\
             -model forward,0.95 -prefix epi
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = FfsArgumentParser(
        prog="ffs_motsim",
        description=(
            "Generate motion-simulation nuisance regressors (Patriat et al. 2017). "
            "Moves a reference EPI by the inverse of the estimated motion to simulate "
            "the signal changes that motion caused, then extracts temporal PCs as "
            "regressors of no interest."
        ),
        epilog=EPILOG,
        formatter_class=FfsHelpFormatter,
    )

    # ── I/O ──
    g_io = p.add_argument_group("Input / Output")
    g_io.add_argument(
        "-base",
        required=True,
        metavar="BASE.nii.gz",
        help="Reference EPI volume (3D). The registration base is the faithful "
        "choice — that is the volume the motion was estimated against. A 4D input "
        "is averaged, which trades simulation sharpness for less noise",
    )
    g_mot = p.add_mutually_exclusive_group(required=True)
    g_mot.add_argument(
        "-aff12",
        metavar="MOCO.aff12.1D",
        help="AFNI-format .aff12.1D matrix file from ffs_moco "
        "(-1Dmatrix_save output, one 3x4 matrix per volume)",
    )
    g_mot.add_argument(
        "-1Dfile",
        dest="onedfile",
        metavar="MOTION.1D",
        help="6-column motion parameter file from ffs_moco "
        "(-1Dfile output: roll pitch yaw dS dL dP)",
    )
    g_mot.add_argument(
        "-dfile",
        metavar="DFILE.1D",
        help="9-column diagnostic file from ffs_moco "
        "(-dfile output: vol# roll pitch yaw dS dL dP rms_bef rms_aft)",
    )
    g_io.add_argument(
        "-prefix",
        required=True,
        metavar="PREFIX",
        help="Output prefix. Produces PREFIX_motsim.1D (regressors)",
    )
    add_verbose_arg(g_io, default=1)

    # ── Model ──
    g_model = p.add_argument_group("Model")
    suggest(
        g_model.add_argument(
            "-model",
            default="both,12",
            metavar="MODE[,N]",
            help="Which simulation and how many components; see the spec table "
            "below [default: %(default)s]",
        ),
        ("both,12", "both,24", "both,0.95", "forward,12", "forward,0.95", "backward,12"),
    )

    # ── Mask ──
    g_mask = p.add_argument_group("Masking")
    g_mask.add_argument(
        "-mask",
        default=None,
        metavar="MASK.nii.gz",
        help="Brain mask. Auto-generated from the reference (ffs automask) otherwise. "
        "Either way it is dilated by -dilate",
    )
    g_mask.add_argument(
        "-dilate",
        type=int,
        default=2,
        metavar="N",
        help="Dilate the mask N voxels outward. The improvement over the standard "
        "motion model lives at the brain edge, so a tight mask discards it "
        "[default: %(default)s]",
    )

    # ── Processing ──
    g_proc = p.add_argument_group("Processing")
    g_proc.add_argument(
        "-interp",
        default="cubic",
        choices=["linear", "cubic", "quintic", "heptic", "wsinc5"],
        help="Interpolation for the simulated resampling [default: %(default)s]",
    )
    g_proc.add_argument(
        "-save_sim",
        action="store_true",
        help="Also save the simulated 4D volumes as NIfTI "
        "(PREFIX_forward.nii.gz, PREFIX_backward.nii.gz)",
    )
    g_proc.add_argument(
        "-save_mask",
        action="store_true",
        help="Also save the dilated PCA mask (PREFIX_motsim_mask.nii.gz)",
    )
    add_device_arg(g_proc)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.time()

    try:
        spec = parse_motsim_spec(args.model)
    except ValueError as exc:
        print(f"ffs_motsim: {exc}", file=sys.stderr)
        return 1

    device = setup_device(args.device, tf32=REGISTRATION_TF32)

    if args.verb >= 1:
        print(f"ffs_motsim: device={device}, model={spec}")

    # Load reference
    with spinner(f"Loading {Path(args.base).name}"):
        ref_data, header_info = load_image(args.base, device=torch.device("cpu"))
    if ref_data.ndim == 4:
        if args.verb >= 1:
            print(f"Reference is 4D ({ref_data.shape[0]} vols), using mean")
        ref_data = ref_data.float().mean(dim=0)
    reference = ref_data.float()
    nz, ny, nx = reference.shape
    if args.verb >= 1:
        print(f"Reference: {nx}x{ny}x{nz}")

    # Load motion matrices (from whichever format was provided)
    nifti_affine = header_info["affine"]
    if args.aff12:
        aff_xform = load_affine_1D(
            args.aff12,
            output_affine=nifti_affine,
            device=torch.device("cpu"),
            debug=(args.verb >= 2),
        )
        matrices_vox = aff_xform.matrices  # (nt, 4, 4) in voxel space
        src_label = args.aff12
    elif args.onedfile:
        params_dicom = load_motion_1d(args.onedfile)
        matrices_vox = params_to_voxel_matrices(params_dicom, nifti_affine)
        src_label = args.onedfile
    else:
        params_dicom = load_dfile(args.dfile)
        matrices_vox = params_to_voxel_matrices(params_dicom, nifti_affine)
        src_label = args.dfile
    nt = matrices_vox.shape[0]
    if args.verb >= 1:
        print(f"Motion matrices: {nt} timepoints (from {src_label})")

    # Mask
    mask = None
    if args.mask:
        with spinner(f"Loading {Path(args.mask).name}"):
            mask_data, _ = load_image(args.mask, device=torch.device("cpu"))
        mask = mask_data > 0.5

    # Prefix
    prefix = args.prefix
    for ext in (".nii.gz", ".nii"):
        if prefix.endswith(ext):
            prefix = prefix[: -len(ext)]

    result = motsim_regressors(
        reference,
        matrices_vox,
        spec,
        device,
        interp=args.interp,
        mask=mask,
        dilate=args.dilate,
        header_info=header_info,
        keep_sims=args.save_sim,
        verb=args.verb,
    )

    n_vox = int(result.mask.sum())
    if args.verb >= 1:
        print(f"Mask: {n_vox} voxels ({n_vox / result.mask.numel() * 100:.1f}%)")

    if args.save_sim:
        for name, sim in (("forward", result.forward), ("backward", result.backward)):
            if sim is None:
                continue
            with spinner(f"Writing {Path(prefix).name}_{name}.nii.gz"):
                save_image(sim, f"{prefix}_{name}.nii.gz", header_info=header_info)
            if args.verb >= 1:
                print(f"Saved: {prefix}_{name}.nii.gz")

    if args.save_mask:
        with spinner(f"Writing {Path(prefix).name}_motsim_mask.nii.gz"):
            save_image(result.mask.float(), f"{prefix}_motsim_mask.nii.gz", header_info=header_info)
        if args.verb >= 1:
            print(f"Saved: {prefix}_motsim_mask.nii.gz")

    out_path = f"{prefix}_motsim.1D"
    save_1d(result.pcs, result.var_explained, out_path, spec.variant, nt)

    elapsed = time.time() - t0
    if args.verb >= 1:
        var_pct = [f"{v * 100:.1f}%" for v in result.var_explained.tolist()]
        print(
            f"Extracted {result.pcs.shape[1]} MotSim PCs ({spec.variant}), "
            f"var explained: {', '.join(var_pct)}"
        )
        print(f"Saved: {out_path}")
        print(f"Done. ({elapsed:.1f}s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
