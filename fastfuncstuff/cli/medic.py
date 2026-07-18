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
        "-recompute",
        dest="recompute",
        action="store_true",
        help="Force a fresh warpkit estimate even if {prefix}_fieldmap already exists. "
        "By default a re-run reuses the cached field map (the slow part) and only "
        "re-derives the cheap pull warp, so adding an output later is near-instant.",
    )
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

    d3 = parser.add_argument_group(
        "3D-EPI slice-direction debug (experimental; for finding residual k-shift)"
    )
    d3.add_argument(
        "-debug_3d",
        "-debug-3d",
        dest="debug_3d",
        action="store_true",
        help="Add an experimental slice-direction (k) displacement to the later "
        "echoes, proportional to the demeaned+detrended per-frame field map and "
        "growing linearly with echo index (0 at echo 1, max at the last echo). "
        "Writes {prefix}_e{N}_3ddebug_{pos,neg} for both signs so you can see which "
        "direction cancels the residual up/down motion, plus {prefix}_fieldmap_detrend.",
    )
    d3.add_argument(
        "-3d_debug_shift",
        "-3d-debug-shift",
        dest="debug3d_shift",
        type=float,
        default=1.0,
        help="Multiplier on the k-shift (default 1.0). With -3d_debug_dfield the shift is "
        "already the physical TE_e[s] * d(field)[Hz] in voxels (TE plays the role the "
        "readout time plays for the primary axis), so this is a DIMENSIONLESS fudge on that "
        "prediction: 1.0 = the raw physics (a 3 Hz change at TE=36 ms -> 3*0.036 = 0.11 vox, "
        "NOT 3 vox). In the legacy (non-dfield) model it is voxels per Hz of the detrended "
        "field at the LAST echo (3 Hz -> 3 vox), with earlier echoes ramping to 0 at echo 1.",
    )
    d3.add_argument(
        "-3d_debug_detrend",
        "-3d-debug-detrend",
        dest="debug3d_detrend",
        type=int,
        default=1,
        help="Polynomial order for detrending the field-map time series before "
        "deriving the shift (0 = demean only, 1 = demean + linear; default 1). "
        "Use -1 to skip detrending and drive the shift off the RAW field map "
        "(mean + trend included).",
    )
    d3.add_argument(
        "-debug_3d_echo",
        "-debug-3d-echo",
        dest="debug3d_echo",
        type=int,
        default=None,
        metavar="N",
        help="Only warp echo N (1-based) — usually the last echo, where the residual "
        "is largest. Skips the other echoes for an ~n_echoes-fold speedup while tuning.",
    )
    d3.add_argument(
        "-3d_debug_subtract_first",
        "-3d-debug-subtract-first",
        dest="debug3d_subtract_first",
        action="store_true",
        help="Subtract the first field-map volume (after any -3d_debug_detrend) before "
        "deriving the shift, so frame 1 gets no k-shift and later frames shift relative "
        "to it. Pair with -3d_debug_detrend -1 to reference the raw field map to frame 1.",
    )
    d3.add_argument(
        "-3d_debug_multiply",
        "-3d-debug-multiply",
        dest="debug3d_multiply",
        action="store_true",
        help="Multiply the (detrended) per-frame field by the per-voxel temporal MEAN of "
        "the field (computed before detrending), so the shift is largest where a big "
        "static field and a big oscillation coincide. Note: this is Hz x Hz, so the "
        "effective magnitude jumps — expect to need a much smaller -3d_debug_shift.",
    )
    d3.add_argument(
        "-3d_debug_proportion",
        "-3d-debug-proportion",
        dest="debug3d_proportion",
        action="store_true",
        help="Divide the (detrended) per-frame field by the per-voxel temporal MEAN of "
        "the field (proportion / fractional change). The opposite of -3d_debug_multiply "
        "(can't combine the two); voxels with ~0 mean field are set to 0.",
    )
    d3.add_argument(
        "-3d_debug_invert",
        "-3d-debug-invert",
        dest="debug3d_invert",
        action="store_true",
        help="Invert the k displacement with the same per-frame fixed-point used by the "
        "real PE undistortion (invert_displacement_pe), instead of applying scale*field "
        "directly as the pull. Correct for large/spatially-varying shifts (the direct "
        "mode pulls from the wrong location there); note it ~negates the effective sign.",
    )
    d3.add_argument(
        "-3d_debug_sign",
        "-3d-debug-sign",
        dest="debug3d_sign",
        choices=["neg", "pos", "both"],
        default="neg",
        help="Which sign(s) of the slice shift to write. The residual was found to be "
        "negative, so the default 'neg' writes only that (2x faster); 'both' writes "
        "pos+neg, 'pos' only positive.",
    )
    d3.add_argument(
        "-3d_debug_recompute",
        "-3d-debug-recompute",
        dest="recompute",
        action="store_true",
        help="Deprecated alias for -recompute (forces a fresh warpkit estimate).",
    )
    d3.add_argument(
        "-3d_debug_dfield",
        "-3d-debug-dfield",
        dest="debug3d_dfield",
        action="store_true",
        help="Field-CHANGE model: drive the k-shift off the per-frame field derivative "
        "(field[t]-field[t-1]) scaled by each echo's TIME (TE_e in seconds), not the "
        "detrended field value scaled by echo index. Physical premise: partition-axis "
        "distortion is set by how fast the field drifts, and grows with TE. Overrides "
        "-3d_debug_detrend/-multiply/-proportion/-subtract_first. Per-echo scale is TE_e "
        "so echo 1 is corrected too; frame 1 gets no shift (no derivative). "
        "-3d_debug_shift becomes voxels per Hz·s.",
    )
    d3.add_argument(
        "-diff_use_interp",
        "-diff-use-interp",
        dest="diff_use_interp",
        action="store_true",
        help="With -3d_debug_dfield: interpolate the field onto the acquisition midpoints "
        "(linear slicetime 'tween' — the field as it was mid-acquisition) BEFORE "
        "differencing, instead of the raw backward difference. A smoother/centered "
        "estimate of the same field change.",
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


def _load_cached_volume(path: str, expected_shape: tuple[int, ...], device):
    """Load a cached 4D NIfTI as a torch tensor if it exists and matches shape.

    Used only by -debug_3d to skip re-running the (slow) warpkit field-map estimate
    and the displacement inversion when iterating on shift/sign/detrend. Returns None
    if the file is missing or its shape disagrees with the current frame selection.
    """
    import os

    import torch

    from fastfuncstuff.io.afni import load_nifti

    if not os.path.exists(path):
        return None
    img = load_nifti(path)
    arr = np.asarray(img.dataobj, dtype=np.float32)
    if arr.shape != tuple(expected_shape):
        return None
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device)


def _load_echoes(paths: list[str]) -> tuple[np.ndarray, np.ndarray, object]:
    """Load per-echo 4D files into (nx, ny, nz, ne, t); return data + affine + header."""
    from fastfuncstuff.io.afni import load_nifti

    vols = []
    affine = np.eye(4)
    header: object = None
    for i, p in enumerate(paths):
        if not Path(p).exists():
            raise FileNotFoundError(f"File not found: {p}")
        img = load_nifti(p)
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

        from fastfuncstuff.cli_utils import parse_device_arg, parse_prefix, spinner
        from fastfuncstuff.io.afni import save_nifti
        from fastfuncstuff.processing.medic import (
            PE_AXIS_MAP,
            detrend_time,
            displacement_pe_to_field,
            field_temporal_change,
            field_to_pull_warp,
            invert_displacement_pe,
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

    with spinner(f"Loading {len(args.magnitude)} magnitude echo(es)"):
        mag, affine, _ = _load_echoes(args.magnitude)
    with spinner(f"Loading {len(args.phase)} phase echo(es)"):
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

    pe_axis = PE_AXIS_MAP[pe_dir]
    Z_AXIS = 2  # slice / partition-encode (k) — the residual axis for -debug_3d
    if args.debug_3d and pe_axis == Z_AXIS:
        print(
            "ERROR: -debug_3d targets the slice (k) direction, but the phase-encode "
            f"direction is also k ({pe_dir}); the residual would collide with the "
            "primary correction. -debug_3d is for in-plane PE (i/j) acquisitions.",
            file=sys.stderr,
        )
        return 1
    if args.debug3d_echo is not None and not (1 <= args.debug3d_echo <= ne):
        print(f"ERROR: -debug_3d_echo {args.debug3d_echo} out of range 1..{ne}.", file=sys.stderr)
        return 1
    if args.debug3d_multiply and args.debug3d_proportion:
        print(
            "ERROR: -3d_debug_multiply and -3d_debug_proportion are opposites; pick one.",
            file=sys.stderr,
        )
        return 1

    # Field map (Hz, native) + per-frame pull warp. warpkit estimation is the slow
    # part and depends only on the inputs, so any re-run (e.g. to add an output, or to
    # iterate on -debug_3d knobs) reuses the cached {prefix}_fieldmap and re-derives the
    # cheap pull warp. -recompute forces a fresh estimate. Fields stay host-resident.
    field_cache = f"{prefix_stem}_fieldmap{nii_ext}"
    disp_cache = f"{prefix_stem}_disp_pull_vox{nii_ext}"
    expected = (nx, ny, nz, nt)
    host = torch.device("cpu")
    result = None
    field_native = None
    disp_pull = None
    field_undist = None
    if not args.recompute:
        field_native = _load_cached_volume(field_cache, expected, host)
        if field_native is not None:
            disp_pull = _load_cached_volume(disp_cache, expected, host)
            if disp_pull is None:
                # Field map cached but not the pull warp — redo just the cheap tail.
                disp_pull, field_undist = field_to_pull_warp(
                    field_native, total_readout_time, pe_dir, pe_axis, device, args.verb >= 1
                )
            if args.verb >= 1:
                print(f"\nReusing cached field map: {field_cache} (skipping warpkit).")
                print("  (pass -recompute to force a fresh estimate.)")

    if field_native is None or disp_pull is None:
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
        field_native = result.field_native
        disp_pull = result.displacement_pe  # (nx,ny,nz,T) torch, host
        field_undist = result.field_undistorted
        # Cache the pull warp so any later re-run is instant; the field map cache is
        # written below (the "field map (Hz, native) QC" output).
        with spinner(f"Writing {Path(disp_cache).name}"):
            save_nifti(disp_pull.cpu().numpy(), disp_cache, affine=affine)

    # In -debug_3d mode the standard per-echo undistortion is redundant with the
    # 3ddebug outputs, and the whole point is to iterate fast, so skip it.
    undist_paths: list[str] = []
    if not args.debug_3d:
        if args.verb >= 1:
            print("\nApplying undistortion (native space, GPU)...")
        # Undistort each magnitude echo with the per-frame warp (feeds ffs_moco).
        for e in range(ne):
            # Keep the echo on the host; undistort_series streams frames to the GPU.
            series = torch.from_numpy(np.ascontiguousarray(mag[:, :, :, e, :]))
            undist = undistort_series(
                series,
                disp_pull,
                pe_axis,
                interp=args.interp,
                verbose=args.verb >= 1,
                desc=f"undistort mag e{e + 1}",
                device=device,
            )
            out_path = f"{prefix_stem}_e{e + 1}_undist{nii_ext}"
            save_nifti(undist.cpu().numpy(), out_path, affine=affine)
            undist_paths.append(out_path)
            if args.verb >= 1:
                print(f"  undistorted magnitude echo {e + 1}: {out_path}")

    # 3D-EPI slice-direction debug: experimental residual k-shift on later echoes.
    if args.debug_3d:
        # Two drivers for the k-shift field. The field-CHANGE model (-3d_debug_dfield) is
        # the physical one: partition-axis distortion tracks how fast the field DRIFTS and
        # grows with TE, so shift = TE_e[s] * (field[t]-field[t-1]). It supersedes the older
        # detrended-value knobs (detrend/multiply/proportion/subtract_first).
        if args.debug3d_dfield:
            field_dt = field_temporal_change(field_native, use_interp=args.diff_use_interp)
            processed = True
        else:
            raw_field = args.debug3d_detrend < 0
            # Per-voxel temporal mean of the field BEFORE detrending (static component).
            mean_field = field_native.mean(dim=-1, keepdim=True)
            field_dt = detrend_time(field_native, args.debug3d_detrend)
            if args.debug3d_subtract_first:
                # Reference to frame 1: its field becomes 0 (no shift), others relative to it.
                field_dt = field_dt - field_dt[..., :1]
            if args.debug3d_multiply:
                # Scale by the static field so the shift is largest where a big static field
                # and a big oscillation coincide (Hz x Hz — expect a smaller -3d_debug_shift).
                field_dt = field_dt * mean_field
            elif args.debug3d_proportion:
                # Fractional change: detrended field / static field. Near-zero mean (e.g.
                # background) would blow up, so those voxels are zeroed.
                near0 = mean_field.abs() < 1e-6
                denom = torch.where(near0, torch.ones_like(mean_field), mean_field)
                field_dt = torch.where(
                    near0.expand_as(field_dt), torch.zeros_like(field_dt), field_dt / denom
                )
            # The field that is multiplied through. Write it unless it is the untouched raw
            # field map (already on disk as the {prefix}_fieldmap cache).
            processed = (
                (not raw_field)
                or args.debug3d_subtract_first
                or args.debug3d_multiply
                or args.debug3d_proportion
            )
        dt_path = f"{prefix_stem}_fieldmap_detrend{nii_ext}"
        if processed:
            with spinner(f"Writing {Path(dt_path).name}"):
                save_nifti(field_dt.cpu().numpy(), dt_path, affine=affine)
        echo_list = range(ne) if args.debug3d_echo is None else [args.debug3d_echo - 1]
        signs = {
            "neg": [(-1.0, "neg")],
            "pos": [(1.0, "pos")],
            "both": [(1.0, "pos"), (-1.0, "neg")],
        }
        sign_list = signs[args.debug3d_sign]
        # Per-echo weight of the k-shift. Field-change model: echo TIME in seconds (larger
        # TE -> more accumulated partition-phase -> more shift; echo 1 IS corrected). Legacy
        # model: linear echo-index ramp, 0 at echo 1 (reference) -> 1 at the last echo.
        te_s = [t / 1000.0 for t in tes]

        def _perecho_weight(e: int) -> float:
            if args.debug3d_dfield:
                return te_s[e]
            return e / (ne - 1) if ne > 1 else 0.0

        if args.verb >= 1:
            if args.debug3d_dfield:
                interp = " (midpoint-interpolated)" if args.diff_use_interp else ""
                print(f"\n3D debug: field CHANGE d(field)/frame{interp}: {dt_path}")
                mode = "inverted pull (fixed-point)" if args.debug3d_invert else "direct pull"
                print(
                    f"  slice(k) shift = sign * {args.debug3d_shift} vox/(Hz·s) * TE_e[s] * "
                    f"d(field); sign(s): {args.debug3d_sign}; {mode}."
                )
            else:
                src = (
                    "RAW field map"
                    if args.debug3d_detrend < 0
                    else f"detrended field map (polort {args.debug3d_detrend})"
                )
                if args.debug3d_subtract_first:
                    src += " minus first volume"
                if args.debug3d_multiply:
                    src += " x mean field"
                if args.debug3d_proportion:
                    src += " / mean field"
                src += f": {dt_path}" if processed else " (no detrend)"
                print(f"\n3D debug: {src}")
                mode = "inverted pull (fixed-point)" if args.debug3d_invert else "direct pull"
                print(
                    f"  slice(k) shift = sign * {args.debug3d_shift} vox/Hz * "
                    f"(echo_idx / (ne-1)) * field; sign(s): {args.debug3d_sign}; {mode}."
                )
            if args.debug3d_echo is not None:
                print(f"  restricting to echo {args.debug3d_echo} (-debug_3d_echo).")

        for e in echo_list:
            series = torch.from_numpy(np.ascontiguousarray(mag[:, :, :, e, :]))
            frac = _perecho_weight(e)
            if frac == 0.0:
                # Zero-weight echo (echo 1 in the legacy ramp): no slice shift -> reference.
                undist = undistort_series(
                    series,
                    disp_pull,
                    pe_axis,
                    interp=args.interp,
                    verbose=args.verb >= 1,
                    desc=f"3ddebug e{e + 1}",
                    device=device,
                )
                out_path = f"{prefix_stem}_e{e + 1}_3ddebug{nii_ext}"
                save_nifti(undist.cpu().numpy(), out_path, affine=affine)
                if args.verb >= 1:
                    print(f"  3ddebug echo {e + 1} (no shift): {out_path}")
                continue
            for sign, tag in sign_list:
                # Forward k displacement (voxels) we believe the data underwent.
                z_disp = (sign * args.debug3d_shift * frac) * field_dt
                if args.debug3d_invert:
                    # Invert to the undistorted-space pull, per frame — the same
                    # fixed-point the real PE undistortion uses. Required when the
                    # shift is large/varying, else the pull samples the wrong voxel.
                    z_pull = torch.empty_like(z_disp)
                    for t in range(z_disp.shape[-1]):
                        z_pull[..., t] = invert_displacement_pe(z_disp[..., t], Z_AXIS)
                    z_disp = z_pull
                undist = undistort_series(
                    series,
                    disp_pull,
                    pe_axis,
                    interp=args.interp,
                    verbose=args.verb >= 1,
                    desc=f"3ddebug e{e + 1} {tag}",
                    extra_disp=z_disp,
                    extra_nifti_axis=Z_AXIS,
                    device=device,
                )
                out_path = f"{prefix_stem}_e{e + 1}_3ddebug_{tag}{nii_ext}"
                save_nifti(undist.cpu().numpy(), out_path, affine=affine)
                if args.verb >= 1:
                    wlabel = "TE" if args.debug3d_dfield else "frac"
                    print(f"  3ddebug echo {e + 1} {tag} ({wlabel} {frac:.3f}): {out_path}")

    if args.apply_phase:
        ph_lo, ph_hi = float(phase.min()), float(phase.max())
        for e in range(ne):
            ph_rad = rescale_phase_to_radians(phase[:, :, :, e, :].astype(np.float32), ph_lo, ph_hi)
            series = torch.from_numpy(np.ascontiguousarray(ph_rad))
            undist = undistort_series(
                series,
                disp_pull,
                pe_axis,
                interp=args.interp,
                circular=True,
                verbose=args.verb >= 1,
                desc=f"undistort phase e{e + 1}",
                device=device,
            )
            out_path = f"{prefix_stem}_e{e + 1}_phase_undist{nii_ext}"
            save_nifti(undist.cpu().numpy(), out_path, affine=affine)
            if args.verb >= 1:
                print(f"  undistorted phase echo {e + 1} (rad): {out_path}")

    # Field map (Hz, native space) QC + field-map cache. When reusing the cache
    # (result is None) it already exists on disk — don't rewrite it.
    if result is not None:
        fmap_path = f"{prefix_stem}_fieldmap{nii_ext}"
        with spinner(f"Writing {Path(fmap_path).name}"):
            save_nifti(field_native.cpu().numpy(), fmap_path, affine=affine)
        if args.verb >= 1:
            print(f"\n  field map (Hz, native): {fmap_path}")

    if args.save_undist:
        # field_undist is set whenever we ran the warp tail (fresh or pull re-derive);
        # only the disp-cache fast path leaves it None, so derive it from the pull warp.
        if field_undist is None:
            field_undist = displacement_pe_to_field(disp_pull, total_readout_time, pe_dir)
        undist_path = f"{prefix_stem}_fieldmap_undistorted{nii_ext}"
        with spinner(f"Writing {Path(undist_path).name}"):
            save_nifti(field_undist.cpu().numpy(), undist_path, affine=affine)
        if args.verb >= 1:
            print(f"  field map (Hz, undistorted): {undist_path}")

    if args.save_warp:
        warp_spec = save_medic_warp(
            disp_pull, pe_axis, affine, prefix_stem, nii_ext, as_5d=args.warp_5d
        )
        if args.verb >= 1:
            kind = "5D mm file" if args.warp_5d else f"per-frame mm files ({nt})"
            print(f"  distortion warp ({kind}, PE={pe_dir}): {warp_spec}")

    if args.debug_3d:
        if args.verb >= 1:
            print(f"\nDone: {datetime.now():%Y-%m-%d %H:%M:%S}")
            print(
                "3d-debug: overlay {prefix}_e{N}_3ddebug_{pos,neg} on the brain's "
                "up/down (k) motion;"
            )
            print(
                "  pick the sign that cancels it, then bisect -3d_debug_shift. Reruns "
                "reuse the cached field map (skip warpkit); -recompute forces fresh."
            )
        else:
            print("MEDIC 3d-debug complete.")
        return 0

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
