#!/usr/bin/env python3

"""
ffs_util_convert_medic - convert warpkit MEDIC output into ffs_nwarp warps.

Takes a warpkit ``wk-medic`` output and writes a per-frame distortion warp in the
SAME mm convention as ffs_qwarp / ffs_medic, so ffs_nwarp composes it (with
-master) into a one-step resample to atlas. Output is either per-frame 3D files
(a ``warp_*`` wildcard) or a single 5D ``(nx,ny,nz,T,3)`` file.

Two input options (give exactly one):

  -fieldmap FILE         warpkit NATIVE field map (Hz, distorted space; the
                         "fieldmap native" output). RECOMMENDED — converted via
                         our verified field->displacement->invert->warp chain
                         (no warpkit sign/orientation conventions involved).

  -displacement_map FILE warpkit displacement map (mm, undistorted space, 4D
                         scalar along the PE axis). More direct (reuses
                         warpkit's ITK inversion) but carries warpkit's
                         DICOM/ITK sign conventions — verify on your data and
                         add -flip if the warp doubles distortion instead of
                         removing it.

EXAMPLE
  ffs_util_convert_medic -fieldmap sub_medic_fieldmap_native.nii.gz \\
      -pe_dir j- -total_readout_time 0.0166951 -prefix sub_dc
  # -> sub_dc_warp/warp_*.nii.gz, ready for:
  ffs_nwarp -source e1.nii.gz -master atlas.nii.gz \\
      -nwarp 'mc.aff12.1D coreg.aff12.1D atlas_warp.nii.gz sub_dc_warp/warp_*.nii.gz' \\
      -prefix e1_atlas.nii.gz

CREDIT
  warpkit / MEDIC: Van A.N. et al. (2026) Imaging Neuroscience; doi:10.1162/IMAG.a.1262
  github.com/vanandrew/warpkit (MIT).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from fastfuncstuff.cli_help import FfsHelpFormatter
from fastfuncstuff.utils import REGISTRATION_TF32, configure_torch_backends


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert warpkit MEDIC output into ffs_nwarp-compatible warps",
        formatter_class=FfsHelpFormatter,
    )
    src = parser.add_argument_group("Input (give exactly one)")
    src.add_argument(
        "-fieldmap",
        metavar="FILE",
        help="warpkit NATIVE field map (Hz, distorted space). Recommended.",
    )
    src.add_argument(
        "-displacement_map",
        "-displacement-map",
        dest="displacement_map",
        metavar="FILE",
        help="warpkit displacement map (mm, undistorted, 4D scalar along PE).",
    )

    req = parser.add_argument_group("Required")
    req.add_argument(
        "-pe_dir",
        "-pe-dir",
        dest="pe_dir",
        required=True,
        metavar="DIR",
        help="Phase-encoding direction: i, j, k, i-, j-, k- (or x/y/z).",
    )
    req.add_argument(
        "-prefix",
        required=True,
        metavar="OUTPUT",
        help="Output prefix (writes {prefix}_warp/warp_*.nii.gz or {prefix}_warp.nii.gz).",
    )

    opt = parser.add_argument_group("Options")
    opt.add_argument(
        "-total_readout_time",
        "-total-readout-time",
        dest="total_readout_time",
        type=float,
        default=None,
        metavar="SEC",
        help="Total EPI readout time (s). Required with -fieldmap.",
    )
    opt.add_argument(
        "-warp_5d",
        "-warp-5d",
        dest="warp_5d",
        action="store_true",
        help="Write one 5D (nx,ny,nz,T,3) file instead of per-frame 3D files.",
    )
    opt.add_argument(
        "-flip",
        action="store_true",
        help="Negate the displacement (escape hatch if the warp doubles "
        "distortion instead of removing it). Mainly for -displacement_map.",
    )
    opt.add_argument(
        "-device",
        default=None,
        help="PyTorch device (default: auto).",
    )
    opt.add_argument(
        "-verb",
        "-verbose",
        dest="verb",
        type=int,
        default=1,
        help="Verbosity.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)

    try:
        import torch

        from fastfuncstuff.cli_utils import parse_device_arg, parse_prefix, spinner
        from fastfuncstuff.io.afni import load_nifti
        from fastfuncstuff.processing.medic import (
            PE_AXIS_MAP,
            field_to_displacement_pe,
            invert_displacement_pe,
            save_medic_warp,
        )
    except ImportError as e:
        print(f"ERROR: could not import fastfuncstuff: {e}", file=sys.stderr)
        return 1

    import nibabel as nib

    if bool(args.fieldmap) == bool(args.displacement_map):
        print("ERROR: give exactly one of -fieldmap or -displacement_map.", file=sys.stderr)
        return 1
    if args.pe_dir not in PE_AXIS_MAP:
        print(f"ERROR: -pe_dir must be one of {sorted(PE_AXIS_MAP)}", file=sys.stderr)
        return 1

    in_path = args.fieldmap or args.displacement_map
    if not Path(in_path).exists():
        print(f"ERROR: file not found: {in_path}", file=sys.stderr)
        return 1

    device, _, _ = parse_device_arg(args.device)
    configure_torch_backends(device, tf32=REGISTRATION_TF32)
    pfx = parse_prefix(args.prefix)
    prefix_stem, nii_ext = pfx.stem, pfx.nifti_ext
    pe_axis = PE_AXIS_MAP[args.pe_dir]

    with spinner(f"Loading {Path(in_path).name}"):
        img = load_nifti(in_path)
        affine = img.affine
        data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 3:
        data = data[..., None]
    if data.ndim != 4:
        print(f"ERROR: expected a 4D (x,y,z,T) input, got shape {data.shape}", file=sys.stderr)
        return 1
    nx, ny, nz, nt = data.shape
    voxel_size_pe = float(nib.Nifti1Header.from_header(img.header).get_zooms()[pe_axis])
    series = torch.from_numpy(np.ascontiguousarray(data)).to(device)

    if args.verb >= 1:
        kind = "field map (Hz, native)" if args.fieldmap else "displacement map (mm)"
        print(
            f"Input: {kind}  {nx}x{ny}x{nz}, {nt} frames, "
            f"PE={args.pe_dir} (voxel {voxel_size_pe} mm)"
        )

    if args.fieldmap:
        if args.total_readout_time is None:
            print("ERROR: -total_readout_time is required with -fieldmap.", file=sys.stderr)
            return 1
        # field (Hz, distorted) -> PE displacement (voxels) -> invert -> pull warp
        disp_native = field_to_displacement_pe(series, args.total_readout_time, args.pe_dir)
        disp_pull = torch.empty_like(disp_native)
        for t in range(nt):
            disp_pull[..., t] = invert_displacement_pe(disp_native[..., t], pe_axis)
    else:
        # warpkit displacement map is already the undistorted-space pull (mm).
        disp_pull = series / voxel_size_pe  # mm -> voxels along PE

    if args.flip:
        disp_pull = -disp_pull

    spec = save_medic_warp(disp_pull, pe_axis, affine, prefix_stem, nii_ext, as_5d=args.warp_5d)
    if args.verb >= 1:
        print(f"Wrote warp: {spec}")
        print(
            "Apply with: ffs_nwarp -source <echo> -nwarp '... " + spec + "' "
            "-master <grid> -prefix <out>"
        )
        if args.displacement_map:
            print(
                "NOTE: verify the sign on your data (apply + check it REMOVES "
                "distortion). If it doubles distortion, re-run with -flip."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
