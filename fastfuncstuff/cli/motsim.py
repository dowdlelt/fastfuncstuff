"""CLI for MotSim: Motion-simulation regressors (Patriat, Reynolds & Birn 2017).

See fastfuncstuff.processing.motsim for the library implementation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from fastfuncstuff.cli_utils import add_verbose_arg, setup_device, spinner
from fastfuncstuff.processing.io import load_image, save_image
from fastfuncstuff.processing.motsim import (
    automask_dilate,
    expand_mask_both,
    extract_pcs,
    load_dfile,
    load_motion_1d,
    params_to_voxel_matrices,
    run_backward_sim,
    run_forward_sim,
    save_1d,
)
from fastfuncstuff.processing.nwarpforge import load_affine_1D
from fastfuncstuff.utils import REGISTRATION_TF32


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ffs_motsim",
        description=(
            "Generate motion-simulation nuisance regressors (Patriat et al. 2017). "
            "Applies motion parameters to a reference EPI to simulate motion-induced "
            "signal changes, then extracts PCs as regressors of no interest."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── I/O ──
    g_io = p.add_argument_group("Input / Output")
    g_io.add_argument(
        "-base",
        required=True,
        metavar="MEAN.nii.gz",
        help="Reference EPI volume (3D). Typically the mean or base volume from motion correction",
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
        "(-dfile output: vol# roll pitch yaw dI dS dL rms_bef rms_aft)",
    )
    g_io.add_argument(
        "-prefix",
        required=True,
        metavar="PREFIX",
        help="Output prefix. Produces PREFIX_motsim.1D (regressors)",
    )
    add_verbose_arg(g_io, default=1)

    # ── PCA options ──
    g_pca = p.add_argument_group("PCA Options")
    g_pca.add_argument(
        "-n_pcs",
        type=int,
        default=12,
        metavar="N",
        help="Number of PCs to extract [default: %(default)s]",
    )
    g_pca.add_argument(
        "-variant",
        choices=["forward", "backward", "both"],
        default="both",
        help="Which simulation(s) to use: 'forward' = inverse-motion "
        "simulation only, 'backward' = re-registered simulation only, "
        "'both' = spatial concatenation (recommended, Patriat et al.) "
        "[default: %(default)s]",
    )

    # ── Mask ──
    g_mask = p.add_argument_group("Masking")
    g_mask.add_argument(
        "-mask",
        default=None,
        metavar="MASK.nii.gz",
        help="Brain mask. If not provided, auto-generated from "
        "the reference via intensity thresholding",
    )
    g_mask.add_argument(
        "-dilate",
        type=int,
        default=2,
        metavar="N",
        help="Dilate mask by N voxels to capture edge effects [default: %(default)s]",
    )

    # ── Processing ──
    g_proc = p.add_argument_group("Processing")
    g_proc.add_argument(
        "-interp",
        default="cubic",
        choices=["linear", "cubic", "quintic", "heptic", "wsinc5"],
        help="Interpolation method for resampling [default: %(default)s]",
    )
    g_proc.add_argument(
        "-save_sim",
        action="store_true",
        help="Also save the simulated 4D volumes as NIfTI "
        "(PREFIX_forward.nii.gz, PREFIX_backward.nii.gz)",
    )
    g_proc.add_argument("-device", type=str, default=None, help="Force device: 'cuda', 'cpu', etc.")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.time()

    # Device
    device = setup_device(args.device, tf32=REGISTRATION_TF32)

    if args.verb >= 1:
        print(f"ffs_motsim: device={device}")

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
    if args.mask:
        with spinner(f"Loading {Path(args.mask).name}"):
            mask_data, _ = load_image(args.mask, device=torch.device("cpu"))
        mask = mask_data > 0.5
    else:
        mask = automask_dilate(reference, dilate_voxels=args.dilate)
    n_vox = mask.sum().item()
    if args.verb >= 1:
        print(f"Mask: {n_vox} voxels ({n_vox / mask.numel() * 100:.1f}%)")

    # Prefix
    prefix = args.prefix
    for ext in (".nii.gz", ".nii"):
        if prefix.endswith(ext):
            prefix = prefix[: -len(ext)]

    # --- Forward simulation ---
    forward_sim = run_forward_sim(
        reference, matrices_vox, device, interp=args.interp, verb=args.verb
    )

    if args.save_sim:
        with spinner(f"Writing {Path(prefix).name}_forward.nii.gz"):
            save_image(forward_sim, f"{prefix}_forward.nii.gz", header_info=header_info)
        if args.verb >= 1:
            print(f"Saved: {prefix}_forward.nii.gz")

    # --- Backward simulation (if needed) ---
    backward_sim = None
    if args.variant in ("backward", "both"):
        backward_sim = run_backward_sim(
            forward_sim,
            reference,
            device,
            interp=args.interp,
            verb=args.verb,
        )
        if args.save_sim:
            with spinner(f"Writing {Path(prefix).name}_backward.nii.gz"):
                save_image(backward_sim, f"{prefix}_backward.nii.gz", header_info=header_info)
            if args.verb >= 1:
                print(f"Saved: {prefix}_backward.nii.gz")

    # --- Extract PCs ---
    if args.variant == "forward":
        pca_input = forward_sim
    elif args.variant == "backward":
        pca_input = backward_sim
    else:  # "both"
        pca_input = torch.cat([forward_sim, backward_sim], dim=1)

    pcs, var_explained = extract_pcs(
        pca_input,
        mask if args.variant != "both" else expand_mask_both(mask),
        args.n_pcs,
        args.verb,
    )

    # Save
    out_path = f"{prefix}_motsim.1D"
    save_1d(pcs, var_explained, out_path, args.variant, nt)

    elapsed = time.time() - t0
    if args.verb >= 1:
        var_pct = [f"{v * 100:.1f}%" for v in var_explained.tolist()]
        print(
            f"Extracted {args.n_pcs} MotSim PCs ({args.variant}), "
            f"var explained: {', '.join(var_pct)}"
        )
        print(f"Saved: {out_path}")
        print(f"Done. ({elapsed:.1f}s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
