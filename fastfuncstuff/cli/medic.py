#!/usr/bin/env python3

"""
ffs_medic - Multi-Echo DIstortion Correction (frame-wise B0 field maps)

Frame-wise B0 distortion correction from complex multi-echo EPI. The field-map
ESTIMATION (multi-echo unwrap + field map) is done by warpkit (Van et al. 2026;
github.com/vanandrew/warpkit) via its in-process C++ ROMEO — fast and correct,
no reason to reimplement it. ffs_medic adds the GPU warping/apply side and the
ffs pipeline integration:

  * applies the per-frame undistortion to each echo on the GPU (wsinc5), and/or
  * writes an ffs_qwarp-compatible mm warp (-save_warp) that ffs_nwarp composes
    with motion/coreg/atlas affines into ONE final resample.

PIPELINE
    # 1. estimate + apply distortion -> undistorted echoes (native grid)
    ffs_medic -magnitude e1m.nii.gz e2m.nii.gz e3m.nii.gz \\
              -phase    e1p.nii.gz e2p.nii.gz e3p.nii.gz \\
              -metadata e1.json e2.json e3.json -prefix sub_dc
    # (or pass -tes / -total_readout_time / -pe_dir explicitly instead of -metadata)
    # -> sub_dc_e1_undist.nii.gz ...

    # 2. motion on the undistorted echoes (rigid is now valid)
    ffs_moco -input sub_dc_e1_undist.nii.gz sub_dc_e2_undist.nii.gz \\
             sub_dc_e3_undist.nii.gz -reg_echo mean -1Dmatrix_save mc.aff12.1D \\
             -prefix sub_mc.nii.gz

    # 3. compose motion + coreg + atlas (all affines) in one resample
    ffs_nwarp -source sub_dc_e1_undist.nii.gz -master atlas.nii.gz \\
              -nwarp 'mc.aff12.1D coreg.aff12.1D atlas_warp.nii.gz' \\
              -prefix sub_e1_atlas.nii.gz

CREDIT
  Estimation: warpkit / MEDIC, Van A.N. et al. (2026) Imaging Neuroscience,
  doi:10.1162/IMAG.a.1262, github.com/vanandrew/warpkit (MIT). Unwrapping: ROMEO
  (Dymerska et al. 2021, MRM).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


class _HelpFormatter(
    argparse.RawDescriptionHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter,
):
    pass


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-Echo DIstortion Correction (MEDIC) — warpkit estimate + GPU apply",
        formatter_class=_HelpFormatter,
    )

    req = parser.add_argument_group("Required Arguments")
    req.add_argument(
        "-magnitude",
        nargs="+",
        metavar="FILE",
        required=True,
        help="Magnitude files, one 4D file per echo (x,y,z,t), in echo order.",
    )
    req.add_argument(
        "-phase",
        nargs="+",
        metavar="FILE",
        required=True,
        help="Phase files, one 4D file per echo, matching -magnitude.",
    )
    req.add_argument(
        "-prefix",
        required=True,
        metavar="OUTPUT",
        help="Output prefix (writes _e{N}_undist undistorted echoes + _fieldmap).",
    )

    acq = parser.add_argument_group("Acquisition (give -metadata OR the three explicit values)")
    acq.add_argument(
        "-metadata",
        nargs="+",
        metavar="JSON",
        help="BIDS sidecar JSON per echo: EchoTime (per file) + TotalReadoutTime "
        "and PhaseEncodingDirection (from the first). Replaces the three below.",
    )
    acq.add_argument(
        "-tes",
        nargs="+",
        type=float,
        metavar="MS",
        help="Echo times in milliseconds, one per echo.",
    )
    acq.add_argument(
        "-total_readout_time",
        "-total-readout-time",
        dest="total_readout_time",
        type=float,
        default=None,
        metavar="SEC",
        help="Total EPI readout time in seconds.",
    )
    acq.add_argument(
        "-pe_dir",
        "-pe-dir",
        dest="pe_dir",
        default=None,
        metavar="DIR",
        help="Phase-encoding direction: i, j, k, i-, j-, k- (or x/y/z).",
    )

    opt = parser.add_argument_group("MEDIC Options")
    opt.add_argument(
        "-svd_filt",
        "-svd-filt",
        dest="svd_filt",
        type=int,
        default=10,
        metavar="N",
        help="warpkit SVD components kept for field-map denoising.",
    )
    opt.add_argument(
        "-interp",
        choices=["wsinc5", "linear"],
        default="wsinc5",
        help="Interpolation for applying the undistortion (default: wsinc5).",
    )
    opt.add_argument(
        "-apply_phase",
        "-apply-phase",
        dest="apply_phase",
        action="store_true",
        help="Also undistort the phase (circular cos/sin interpolation, radians). "
        "Off by default; magnitude echoes are always undistorted.",
    )
    opt.add_argument(
        "-save_warp",
        "-save-warp",
        dest="save_warp",
        action="store_true",
        help="Also write the per-frame mm warp (ffs_qwarp convention) for "
        "ffs_nwarp. By default ffs_medic applies the warp itself. See -warp_5d.",
    )
    opt.add_argument(
        "-warp_5d",
        "-warp-5d",
        dest="warp_5d",
        action="store_true",
        help="With -save_warp, write ONE 5D (nx,ny,nz,T,3) file instead of "
        "per-frame 3D files (applied via a 'warp_*' wildcard).",
    )
    opt.add_argument(
        "-save_undistorted_fieldmap",
        "-save-undistorted-fieldmap",
        dest="save_undist",
        action="store_true",
        help="Also write the undistorted-space field map (Hz).",
    )

    sel = parser.add_argument_group("Frame Selection (trimming / testing)")
    sel.add_argument(
        "-skip_first",
        "-skip-first",
        dest="skip_first",
        type=int,
        default=0,
        help="Drop this many volumes from the start of every echo (match ffs_moco).",
    )
    sel.add_argument(
        "-skip_last",
        "-skip-last",
        dest="skip_last",
        type=int,
        default=0,
        help="Drop this many volumes from the end of every echo.",
    )
    sel.add_argument(
        "-nframes",
        dest="nframes",
        type=int,
        default=None,
        metavar="N",
        help="Use only the first N frames (after -skip_first/-skip_last). For quick tests.",
    )

    hw = parser.add_argument_group("Hardware Options")
    hw.add_argument(
        "-n_cpus",
        "-n-cpus",
        dest="n_cpus",
        type=int,
        default=4,
        metavar="N",
        help="CPU workers for warpkit's per-frame unwrap.",
    )
    hw.add_argument(
        "-device",
        default=None,
        help="PyTorch device for the GPU warp/apply: cpu, cuda (default: auto).",
    )

    out = parser.add_argument_group("Output Options")
    out.add_argument(
        "-debug",
        action="store_true",
        help="Write sanity-check intermediates to {prefix}_debug/: warpkit native "
        "field map, brain mask, and the PE displacement.",
    )
    out.add_argument(
        "-verb",
        "-verbose",
        dest="verb",
        type=int,
        default=1,
        help="Verbosity (0 = quiet).",
    )
    return parser


def _read_metadata(paths: list[str]) -> tuple[list[float], float, str]:
    """Read BIDS sidecars: per-echo EchoTime (s -> ms), TRT + PE from the first."""
    tes_ms: list[float] = []
    trt: float | None = None
    pe: str | None = None
    for i, p in enumerate(paths):
        with open(p) as f:
            m = json.load(f)
        tes_ms.append(float(m["EchoTime"]) * 1000.0)
        if i == 0:
            trt = float(m["TotalReadoutTime"])
            pe = str(m["PhaseEncodingDirection"])
    assert trt is not None and pe is not None
    return tes_ms, trt, pe


def _load_echoes(paths: list[str]) -> tuple[np.ndarray, np.ndarray, object]:
    """Load per-echo 4D files into (nx, ny, nz, ne, t); return data + affine + header."""
    from typing import cast

    import nibabel as nib

    vols = []
    affine = np.eye(4)
    header: object = None
    for i, p in enumerate(paths):
        if not Path(p).exists():
            raise FileNotFoundError(f"File not found: {p}")
        img = cast(nib.Nifti1Image, nib.load(p))
        data = np.asarray(img.dataobj, dtype=np.float32)
        if data.ndim == 3:
            data = data[..., None]
        if i == 0:
            affine = img.affine
            header = img.header
        vols.append(data)
    shapes = {v.shape for v in vols}
    if len(shapes) != 1:
        raise ValueError(f"All echo files must share a shape; got {shapes}")
    stacked = np.stack(vols, axis=3)  # (nx, ny, nz, ne, t)
    return stacked, affine, header


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    try:
        import torch

        from fastfuncstuff.cli_utils import parse_device_arg, parse_prefix
        from fastfuncstuff.io.afni import save_nifti
        from fastfuncstuff.processing.medic import (
            PE_AXIS_MAP,
            medic_fieldmaps,
            rescale_phase_to_radians,
            save_medic_warp,
            undistort_series,
            warpkit_available,
        )
        from fastfuncstuff.utils import configure_torch_backends
    except ImportError as e:
        print(f"ERROR: Could not import fastfuncstuff: {e}", file=sys.stderr)
        return 1

    if not warpkit_available():
        print(
            "ERROR: ffs_medic needs warpkit for the field-map estimation.\n"
            "       pip install warpkit  (github.com/vanandrew/warpkit)",
            file=sys.stderr,
        )
        return 1

    # ── Acquisition: metadata sidecars OR explicit -tes/-total_readout_time/-pe_dir
    if args.metadata is not None:
        if len(args.metadata) != len(args.magnitude):
            print(
                f"ERROR: {len(args.metadata)} -metadata files but {len(args.magnitude)} echoes.",
                file=sys.stderr,
            )
            return 1
        try:
            tes, total_readout_time, pe_dir = _read_metadata(args.metadata)
        except (OSError, KeyError, ValueError) as e:
            print(f"ERROR: reading -metadata sidecars: {e}", file=sys.stderr)
            return 1
    else:
        if args.tes is None or args.total_readout_time is None or args.pe_dir is None:
            print(
                "ERROR: give -metadata, or all of -tes / -total_readout_time / -pe_dir.",
                file=sys.stderr,
            )
            return 1
        tes, total_readout_time, pe_dir = args.tes, args.total_readout_time, args.pe_dir

    if len(args.magnitude) != len(args.phase):
        print(
            f"ERROR: {len(args.magnitude)} magnitude files but {len(args.phase)} phase files.",
            file=sys.stderr,
        )
        return 1
    if len(tes) != len(args.magnitude):
        print(f"ERROR: {len(tes)} echo times but {len(args.magnitude)} echoes.", file=sys.stderr)
        return 1
    if pe_dir not in PE_AXIS_MAP:
        print(f"ERROR: phase-encoding dir must be one of {sorted(PE_AXIS_MAP)}", file=sys.stderr)
        return 1

    pfx = parse_prefix(args.prefix)
    prefix_stem, nii_ext = pfx.stem, pfx.nifti_ext
    device, _, _ = parse_device_arg(args.device)
    configure_torch_backends(device)

    if args.verb >= 1:
        print("=" * 70)
        print("MEDIC distortion correction (ffs_medic; warpkit estimate + GPU apply)")
        print("=" * 70)
        print(f"Start: {datetime.now():%Y-%m-%d %H:%M:%S}   Device: {device}")
        print("\nLoading echoes...")

    mag, affine, _ = _load_echoes(args.magnitude)
    phase, _, _ = _load_echoes(args.phase)
    if mag.shape != phase.shape:
        print(f"ERROR: magnitude shape {mag.shape} != phase shape {phase.shape}", file=sys.stderr)
        return 1

    # Frame selection: trim ends, then optionally cap to first N (testing).
    nt_full = mag.shape[4]
    if args.skip_first < 0 or args.skip_last < 0:
        print("ERROR: -skip_first and -skip_last must be non-negative.", file=sys.stderr)
        return 1
    if args.skip_first + args.skip_last >= nt_full:
        print(
            f"ERROR: -skip_first={args.skip_first} + -skip_last={args.skip_last} "
            f"removes all {nt_full} frames.",
            file=sys.stderr,
        )
        return 1
    t0, t1 = args.skip_first, nt_full - args.skip_last
    if t0 or args.skip_last:
        mag, phase = mag[:, :, :, :, t0:t1], phase[:, :, :, :, t0:t1]
    if args.nframes is not None:
        if args.nframes < 1:
            print("ERROR: -nframes must be >= 1.", file=sys.stderr)
            return 1
        mag, phase = mag[:, :, :, :, : args.nframes], phase[:, :, :, :, : args.nframes]
    mag = np.ascontiguousarray(mag)
    phase = np.ascontiguousarray(phase)

    nx, ny, nz, ne, nt = mag.shape
    if (t0 or args.skip_last or args.nframes is not None) and args.verb >= 1:
        print(
            f"  frame selection: {nt_full} -> {nt} frames "
            f"(skip_first={args.skip_first}, skip_last={args.skip_last}, nframes={args.nframes})"
        )
    if args.verb >= 1:
        print(f"  {ne} echoes, {nt} frames, grid {nx}x{ny}x{nz}")
        print(f"  TEs: {tes} ms   readout: {total_readout_time}s   PE: {pe_dir}\n")

    result = medic_fieldmaps(
        phase=phase,
        mag=mag,
        tes=tes,
        affine=affine,
        total_readout_time=total_readout_time,
        pe_dir=pe_dir,
        svd_filt=args.svd_filt,
        n_cpus=args.n_cpus,
        device=device,
        debug_dir=f"{prefix_stem}_debug" if args.debug else None,
        verbose=args.verb >= 1,
    )

    pe_axis = result.pe_tensor_axis
    disp_pull = result.displacement_pe  # (nx,ny,nz,T) torch, on device

    if args.verb >= 1:
        print("\nApplying undistortion (native space, GPU)...")

    # Undistort each magnitude echo with the per-frame warp (feeds ffs_moco).
    undist_paths = []
    for e in range(ne):
        series = torch.from_numpy(np.ascontiguousarray(mag[:, :, :, e, :])).to(device)
        undist = undistort_series(
            series,
            disp_pull,
            pe_axis,
            interp=args.interp,
            verbose=args.verb >= 1,
            desc=f"undistort mag e{e + 1}",
        )
        out_path = f"{prefix_stem}_e{e + 1}_undist{nii_ext}"
        save_nifti(undist.cpu().numpy(), out_path, affine=affine)
        undist_paths.append(out_path)
        if args.verb >= 1:
            print(f"  undistorted magnitude echo {e + 1}: {out_path}")

    if args.apply_phase:
        ph_lo, ph_hi = float(phase.min()), float(phase.max())
        for e in range(ne):
            ph_rad = rescale_phase_to_radians(phase[:, :, :, e, :].astype(np.float32), ph_lo, ph_hi)
            series = torch.from_numpy(np.ascontiguousarray(ph_rad)).to(device)
            undist = undistort_series(
                series,
                disp_pull,
                pe_axis,
                interp=args.interp,
                circular=True,
                verbose=args.verb >= 1,
                desc=f"undistort phase e{e + 1}",
            )
            out_path = f"{prefix_stem}_e{e + 1}_phase_undist{nii_ext}"
            save_nifti(undist.cpu().numpy(), out_path, affine=affine)
            if args.verb >= 1:
                print(f"  undistorted phase echo {e + 1} (rad): {out_path}")

    # Field map (Hz, native space) QC.
    fmap_path = f"{prefix_stem}_fieldmap{nii_ext}"
    save_nifti(result.field_native.cpu().numpy(), fmap_path, affine=affine)
    if args.verb >= 1:
        print(f"\n  field map (Hz, native): {fmap_path}")

    if args.save_undist:
        undist_path = f"{prefix_stem}_fieldmap_undistorted{nii_ext}"
        save_nifti(result.field_undistorted.cpu().numpy(), undist_path, affine=affine)
        if args.verb >= 1:
            print(f"  field map (Hz, undistorted): {undist_path}")

    if args.save_warp:
        warp_spec = save_medic_warp(
            disp_pull, pe_axis, affine, prefix_stem, nii_ext, as_5d=args.warp_5d
        )
        if args.verb >= 1:
            kind = "5D mm file" if args.warp_5d else f"per-frame mm files ({nt})"
            print(f"  distortion warp ({kind}, PE={pe_dir}): {warp_spec}")

    if args.verb >= 1:
        print(f"\nDone: {datetime.now():%Y-%m-%d %H:%M:%S}")
        print("Next: motion-correct the undistorted echoes, then compose to atlas:")
        print(
            f"  ffs_moco -input {' '.join(undist_paths)} -reg_echo mean "
            "-1Dmatrix_save mc.aff12.1D -prefix mc.nii.gz"
        )
        print(
            f"  ffs_nwarp -source {undist_paths[0]} -master atlas.nii.gz "
            "-nwarp 'mc.aff12.1D coreg.aff12.1D atlas_warp.nii.gz' -prefix out.nii.gz"
        )
    else:
        print(f"MEDIC complete. Undistorted echoes: {', '.join(undist_paths)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
