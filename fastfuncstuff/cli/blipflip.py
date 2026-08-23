#!/usr/bin/env python3

"""
ffs_blipflip - fast blip-up/blip-down susceptibility distortion correction.

A GPU (CPU-capable) port of the *logic* of FSL's ``topup``: estimate one
off-resonance field from two EPI images acquired with opposing phase-encode
polarity ("blip up" / "blip down"), and write a single 4-D undistortion warp that
``ffs_nwarp`` composes with motion/coreg/atlas transforms into one final resample.

Unlike FSL topup, there is no acquisition-parameter text file and no need to
concatenate the inputs beforehand: pass the two images (or an N-image list) with
their phase-encode direction and readout time on the command line. If an input is a
4-D timeseries it is collapsed (median by default) to one volume first. Motion
estimation is deferred; the only motion modelled is an optional single global
translation along the PE axis (``-pe_shift``), the tiny scanner-induced shift
possible between back-to-back scans.

    ffs_blipflip -blip_up AP.nii.gz -blip_down PA.nii.gz \\
                 -pe_dir j -readout 0.045 -prefix sub_dc

    # -> sub_dc_warp.nii.gz     (distorted[blip_up] -> undistorted "middle" space)
    #    sub_dc_invwarp.nii.gz  (undistorted "middle" -> distorted geometry; the inverse)
    #    sub_dc_field.nii.gz    (off-resonance field, Hz; needs -readout)
    #    sub_dc_unwarped.nii.gz + sub_dc_mean.nii.gz  (corrected inputs + mean)
    #    sub_dc_jac.nii.gz      (Jacobian modulation map)
    #
    # -readout is optional (the warp is insensitive to a common factor); omit it and the
    # Hz field map is simply not written. Precision tiers: default 9-level b02b0, or
    # -workhard (FSL 7T, 12 levels), or -superhard (14 levels, knots to 2 mm).

    # apply to the matching timeseries in one resample:
    ffs_nwarp -source epi.nii.gz -master epi.nii.gz \\
              -nwarp sub_dc_warp.nii.gz -prefix epi_dc.nii.gz

METHOD / CREDIT
  The groupwise blip-up/down field model (one B-spline field, per-scan PE
  displacement + Jacobian intensity modulation, bending-energy regularisation, and
  the coarse-to-fine warpres/fwhm/lambda schedule) is that of FSL ``topup``:
  Andersson, Skare & Ashburner (2003), NeuroImage 20:870-888; Smith et al. (2004),
  NeuroImage 23:S208-219. Original topup by Jesper Andersson, FMRIB Analysis Group,
  University of Oxford. This is an independent reimplementation, not FSL code.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from datetime import datetime

import numpy as np
import torch

from fastfuncstuff.cli_utils import (
    add_batch_args,
    add_device_arg,
    collect_batch_jobs,
    parse_prefix,
    run_batch_jobs,
    setup_device,
)
from fastfuncstuff.utils import REGISTRATION_TF32


class _HelpFormatter(
    argparse.RawDescriptionHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter,
):
    pass


def _float_list(s: str) -> list[float]:
    return [float(x) for x in s.replace(",", " ").split()]


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.replace(",", " ").split()]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ffs_blipflip",
        description="Blip-up/blip-down susceptibility distortion correction "
        "(FSL topup-style field model; independent GPU reimplementation).",
        formatter_class=_HelpFormatter,
        epilog=__doc__.split("METHOD / CREDIT")[1].strip(),
    )

    inp = parser.add_argument_group("Inputs (pick the pair form OR -imain)")
    inp.add_argument("-blip_up", metavar="FILE", help="Blip-up EPI (3-D, or 4-D to collapse).")
    inp.add_argument("-blip_down", metavar="FILE", help="Blip-down EPI (opposite PE polarity).")
    inp.add_argument(
        "-imain",
        nargs="+",
        metavar="FILE",
        help="General N-image form: two or more EPIs. Needs matching -pe_dir/-readout "
        "lists (one entry each, or one shared value).",
    )
    inp.add_argument(
        "-pe_dir",
        "-pe-dir",
        nargs="+",
        metavar="DIR",
        help="Phase-encode direction of the blip_up (or of each -imain image): one of "
        "i/j/k (or x/y/z) with optional trailing '-'. The blip_down is the opposite.",
    )
    inp.add_argument(
        "-readout",
        "-total_readout_time",
        nargs="+",
        type=float,
        default=None,
        metavar="SEC",
        help="Total readout time (s), one shared value or one per image. Optional: it only "
        "scales the Hz<->voxel conversion, and the warp is insensitive to a common factor. "
        "If omitted the warp is still correct, but the Hz field map cannot be written and is "
        "forced off.",
    )
    inp.add_argument(
        "-collapse",
        choices=("median", "mean"),
        default="median",
        help="How to collapse a 4-D input to one volume.",
    )

    out = parser.add_argument_group("Outputs")
    out.add_argument("-prefix", help="Output prefix (stem[.nii.gz]) [required unless -batch].")
    out.add_argument(
        "-warp_for",
        choices=("up", "down"),
        default="up",
        help="Which acquisition geometry the undistortion warp corrects.",
    )
    out.add_argument(
        "-no_invwarp",
        action="store_true",
        help="Do not write the inverse warp (undistorted 'middle' -> distorted geometry).",
    )
    out.add_argument("-no_fmap", action="store_true", help="Do not write the Hz field map.")
    out.add_argument(
        "-no_unwarped", action="store_true", help="Do not write corrected inputs + mean."
    )
    out.add_argument("-no_jac", action="store_true", help="Do not write the Jacobian map.")
    out.add_argument(
        "-no_mask_field",
        action="store_true",
        help="Do not taper the field to zero outside the object. By default the field "
        "(and warp) is rolled off in air so ffs_nwarp does not replicate tissue into an "
        "auto-pad margin; disable to keep the raw extrapolated field everywhere.",
    )

    sched = parser.add_argument_group("Schedule (defaults reproduce FSL b02b0)")
    sched.add_argument(
        "-config",
        choices=("b02b0", "quick", "workhard", "superhard"),
        default="b02b0",
        help="Preset schedule: 'quick' (3 levels), 'b02b0' (9 levels, FSL default), "
        "'workhard' (12 levels, FSL 7T config — coarser start + finer, more precise), "
        "'superhard' (14 levels, pushed further: knots down to 2 mm, extra low-lambda "
        "refinement). Higher tiers are slower.",
    )
    sched.add_argument(
        "-workhard",
        action="store_const",
        const="workhard",
        dest="config",
        help="Shortcut for -config workhard (FSL 7T schedule).",
    )
    sched.add_argument(
        "-superhard",
        action="store_const",
        const="superhard",
        dest="config",
        help="Shortcut for -config superhard (most precise, slowest).",
    )
    sched.add_argument("-warpres", type=_float_list, help="Override knot spacing (mm) per level.")
    sched.add_argument("-fwhm", type=_float_list, help="Override smoothing FWHM (mm) per level.")
    sched.add_argument(
        "-lambda", dest="lam", type=_float_list, help="Override penalty weight per level."
    )
    sched.add_argument("-miter", type=_int_list, help="Override max GN iters per level.")
    sched.add_argument("-subsamp", type=_int_list, help="Override subsampling factor per level.")
    sched.add_argument(
        "-reg_mode",
        choices=("bending", "membrane"),
        default="bending",
        help="Field regularisation model.",
    )
    sched.add_argument(
        "-no_ssqlambda",
        action="store_true",
        help="Do not scale the penalty weight by the current SSD each iteration.",
    )
    sched.add_argument(
        "-pe_shift",
        action="store_true",
        help="EXPERIMENTAL: first estimate + remove a single global PE-axis translation "
        "between the two scans (tiny scanner shift), then estimate the field. A rigid "
        "shift is confounded by the field itself, so it is capped small; off by default.",
    )

    mot = parser.add_argument_group("Movement (topup --estmov analogue)")
    mot.add_argument(
        "-estmov",
        "-motion",
        action="store_true",
        help="Jointly estimate rigid head movement between the scans, like FSL topup. "
        "Movement is estimated block-coordinate with the field at the COARSE levels only "
        "(knot spacing >= 10 mm, reproducing topup's --estmov schedule) and frozen for the "
        "fine levels; the first scan is the fixed reference and the PE-axis translation "
        "(degenerate with the field) is excluded. Reuses ffs_moco's GPU rigid registration "
        "on the field-corrected images. Off by default (assumes the pair is aligned).",
    )
    mot.add_argument(
        "-motion_ref",
        type=int,
        default=0,
        help="Index of the fixed reference scan for movement (default 0, the first).",
    )
    mot.add_argument(
        "-motion_interp",
        default="cubic",
        choices=("linear", "cubic", "quintic", "heptic", "wsinc5"),
        help="Interpolation used when baking the estimated rigid transform into the data.",
    )

    fold = parser.add_argument_group("Anti-fold barrier (keeps the field diffeomorphic)")
    fold.add_argument(
        "-jac_penalty",
        "-jac-penalty",
        type=float,
        default=1e-3,
        help="Weight of the anti-fold barrier that forbids a negative Jacobian (a fold is "
        "physically impossible for a real blip pair). Relative to the data SSD, like -lambda. "
        "Set 0 to disable. Default reaches a positive min-Jacobian while changing the field "
        "only ~3 Hz (below the float32/64 noise floor).",
    )
    fold.add_argument(
        "-jac_floor",
        "-jac-floor",
        type=float,
        default=0.1,
        help="Jacobian margin: the barrier keeps the Jacobian in [floor, 2-floor].",
    )
    fold.add_argument(
        "-no_antifold",
        action="store_true",
        help="Disable the anti-fold barrier entirely (equivalent to -jac_penalty 0).",
    )
    fold.add_argument(
        "-reject_folds",
        action="store_true",
        help="Additionally veto any line-search step that introduces a fold. Off by default: "
        "it perturbs the field far more than the barrier for no extra fold removal.",
    )

    misc = parser.add_argument_group("Misc")
    misc.add_argument(
        "-precision",
        choices=("float32", "float64"),
        default="float32",
        help="Working precision of the solve. float32 is ~2x faster on consumer GPUs with "
        "no accuracy cost here (the sensitive reductions accumulate in float64 anyway); "
        "float64 reproduces the original all-double behaviour.",
    )
    add_device_arg(
        misc,
        extra="Use CPU on Mac: the stable CG reductions require float64, which MPS does not support.",
    )
    misc.add_argument("-verb", type=int, default=1, help="Verbosity (0/1/2).")
    add_batch_args(
        parser,
        tool="ffs_blipflip",
        what="fieldmap solves",
        example="-blip_up up.nii.gz -blip_down down.nii.gz -pe_dir j -prefix dc",
        skip_note="-prefix (its _warp and _unwarped images)",
    )
    return parser


def _load_volume(path: str, collapse: str) -> tuple[torch.Tensor, np.ndarray, object]:
    """Load a NIfTI, collapse 4-D -> 3-D, return (nz,ny,nx) tensor + affine + header."""
    from fastfuncstuff.io.afni import load_nifti

    img = load_nifti(path)
    arr = np.asarray(img.dataobj, dtype=np.float32)  # (nx, ny, nz[, t])
    if arr.ndim == 4:
        arr = np.median(arr, axis=3) if collapse == "median" else arr.mean(axis=3)
    elif arr.ndim != 3:
        raise ValueError(f"{path}: expected 3-D or 4-D, got shape {arr.shape}")
    # (nx,ny,nz) -> (nz,ny,nx) tensor order used throughout processing.topup.
    t = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 1, 0)))
    return t, np.asarray(img.affine), img.header


def _voxel_sizes_zyx(affine: np.ndarray) -> tuple[float, float, float]:
    vx, vy, vz = (float(np.linalg.norm(affine[:3, i])) for i in range(3))
    return (vz, vy, vx)  # (vz, vy, vx) to match (nz, ny, nx) data


def _build_config(name: str):
    """Return the TopupConfig for a preset name.

    The tiers track the FSL progression b02b0 -> b02b0_7T -> (extrapolated further):
    coarser/smoother starts, more levels, finer final knot spacing, and lower final
    regularisation for a more precise field, at increasing cost.
    """
    from fastfuncstuff.processing.topup import TopupConfig

    if name == "quick":
        return TopupConfig(
            warpres=[20, 12, 8],
            fwhm=[8, 4, 1],
            lam=[1e-3, 1e-4, 1e-5],
            miter=[6, 8, 10],
            subsamp=[1, 1, 1],
        )
    if name == "workhard":  # FSL b02b0_7T.cnf (12 levels)
        return TopupConfig(
            warpres=[30, 25, 20, 18, 16, 14, 12, 10, 6, 4, 4, 4],
            subsamp=[2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1],
            fwhm=[14, 12, 10, 8, 6, 4, 3, 3, 2, 1, 0, 0],
            miter=[5, 5, 5, 5, 5, 5, 5, 5, 10, 10, 20, 20],
            lam=[3e-4, 2e-4, 1e-4, 5e-5, 2.5e-5, 1e-5, 2.5e-6, 5e-7, 5e-8, 5e-9, 5e-11, 1e-12],
        )
    if name == "superhard":  # pushed past 7T: finer final knots (2 mm) + extra refinement
        return TopupConfig(
            warpres=[30, 25, 20, 18, 16, 14, 12, 10, 8, 6, 4, 4, 3, 2],
            subsamp=[2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1],
            fwhm=[14, 12, 10, 8, 6, 4, 3, 3, 2, 1, 0, 0, 0, 0],
            miter=[5, 5, 5, 5, 5, 5, 5, 5, 10, 10, 20, 20, 30, 30],
            lam=[
                3e-4,
                2e-4,
                1e-4,
                5e-5,
                2.5e-5,
                1e-5,
                2.5e-6,
                5e-7,
                8e-8,
                8e-9,
                8e-10,
                8e-11,
                8e-12,
                1e-13,
            ],
        )
    return TopupConfig()  # b02b0 defaults (9 levels)


def _resolve_device(device_spec) -> torch.device:
    """Device for a blipflip run: auto falls back to CPU without CUDA, and MPS is
    refused outright (the CG reductions are float64)."""
    if (
        device_spec is None or str(device_spec).lower() == "auto"
    ) and not torch.cuda.is_available():
        device_spec = "cpu"
    device = setup_device(device_spec, tf32=REGISTRATION_TF32)
    if device.type == "mps":
        raise SystemExit(
            "ffs_blipflip: MPS cannot run the required float64 CG reductions; "
            "use -device cpu on Mac."
        )
    return device


def _expected_outputs(args: argparse.Namespace) -> list[str]:
    """Concrete output paths a solo run of ``args`` would write, for -batch_skip.

    The warp is the estimate; the unwarped stack is checked with it because the
    cross-fmap stage reads that, not the warp."""
    if not args.prefix:
        return []
    pinfo = parse_prefix(args.prefix)
    outs = [f"{pinfo.stem}_warp{pinfo.nifti_ext}"]
    if not args.no_unwarped:
        outs.append(f"{pinfo.stem}_unwarped{pinfo.nifti_ext}")
    return outs


def _validate_batch_run(run_args: argparse.Namespace) -> None:
    """Per-run validation for a batch job: the flags argparse would have enforced."""
    missing = [f for f in ("pe_dir", "prefix") if not getattr(run_args, f, None)]
    if missing:
        raise ValueError("run is missing " + ", ".join("-" + m for m in missing))


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)

    if args.batch is not None or args.batch_run:
        # A fieldmap solve is seconds of GN on a B-spline field wrapped in ~2s of
        # interpreter/CUDA startup, and a session has one per blip-up/down pair.
        run_batch_jobs(
            tool="ffs_blipflip",
            jobs=collect_batch_jobs(args.batch, args.batch_run),
            device=_resolve_device(args.device),
            parse_line=lambda line, base: create_parser().parse_args(shlex.split(line), base),
            defaults=args,
            dispatch=_dispatch_raising,
            validate=_validate_batch_run,
            is_nested=lambda ra: ra.batch is not None or ra.batch_run is not None,
            expected_outputs=_expected_outputs,
            skip_existing=args.batch_skip,
        )
        return 0

    try:
        _validate_batch_run(args)
    except ValueError as exc:
        print(f"ERROR: {exc} (or use -batch FILE / -batch_run ARGS).", file=sys.stderr)
        return 2

    return _dispatch_run(args, _resolve_device(args.device))


def _dispatch_raising(args: argparse.Namespace, device: torch.device) -> None:
    rc = _dispatch_run(args, device)
    if rc:
        raise ValueError(f"run exited {rc}")


def _dispatch_run(args: argparse.Namespace, device: torch.device) -> int:
    """Solve one blip-up/down pair (the whole per-fieldmap body).

    Both the standalone path and every batch job go through here, so a manifest
    line reproduces a solo invocation bit-for-bit."""
    from fastfuncstuff.processing import topup as T
    from fastfuncstuff.processing.io import save_warp_field
    from fastfuncstuff.processing.medic import PE_AXIS_MAP, invert_displacement_pe

    pinfo = parse_prefix(args.prefix)
    stem, ext = pinfo.stem, pinfo.nifti_ext

    # ---- assemble the image list + geometry ----
    if args.imain:
        paths = list(args.imain)
        if len(paths) < 2:
            print("ffs_blipflip: -imain needs at least two images.", file=sys.stderr)
            return 2
    else:
        if not (args.blip_up and args.blip_down):
            print("ffs_blipflip: provide -blip_up and -blip_down, or -imain.", file=sys.stderr)
            return 2
        paths = [args.blip_up, args.blip_down]

    def _broadcast(vals: list, n: int, what: str) -> list:
        if len(vals) == 1:
            return vals * n
        if len(vals) != n:
            raise SystemExit(f"ffs_blipflip: {what} needs 1 or {n} values, got {len(vals)}")
        return vals

    pe_dirs = _broadcast(args.pe_dir, len(paths), "-pe_dir")
    # Readout is optional: absent -> use 1.0 (warp unaffected by a common factor) and force
    # the Hz field map off, since without a real readout the field is in voxels, not Hz.
    no_readout = args.readout is None
    readouts = (
        [1.0] * len(paths) if no_readout else _broadcast(args.readout, len(paths), "-readout")
    )

    # For the plain pair form, -pe_dir describes blip_up; blip_down is opposite.
    if not args.imain and len(args.pe_dir) == 1:
        up = args.pe_dir[0]
        down = up[:-1] if up.endswith("-") else up + "-"
        pe_dirs = [up, down]

    scans: list[T.ScanSpec] = []
    affine = None
    header = None
    for p, pe, ro in zip(paths, pe_dirs, readouts, strict=True):
        if pe not in PE_AXIS_MAP:
            raise SystemExit(f"ffs_blipflip: bad -pe_dir '{pe}'")
        data, aff, hdr = _load_volume(p, args.collapse)
        if affine is None:
            affine, header = aff, hdr
        sign = -1.0 if pe.endswith("-") else 1.0
        scans.append(
            T.ScanSpec(data=data.to(device), pe_axis=PE_AXIS_MAP[pe], sign=sign, readout=float(ro))
        )
    assert affine is not None
    vox = _voxel_sizes_zyx(affine)

    # ---- schedule ----
    cfg = _build_config(args.config)
    if args.warpres:
        cfg.warpres = args.warpres
    if args.fwhm:
        cfg.fwhm = args.fwhm
    if args.lam:
        cfg.lam = args.lam
    if args.miter:
        cfg.miter = args.miter
    if args.subsamp:
        cfg.subsamp = args.subsamp
    cfg.reg_mode = args.reg_mode
    cfg.ssqlambda = not args.no_ssqlambda
    cfg.jac_penalty = 0.0 if args.no_antifold else args.jac_penalty
    cfg.jac_floor = args.jac_floor
    cfg.reject_folds = args.reject_folds

    if args.verb >= 1:
        print(
            f"ffs_blipflip: {len(scans)} scans, grid={tuple(scans[0].data.shape)}, "
            f"voxel(zyx)={tuple(round(v, 3) for v in vox)} mm, device={device}, "
            f"precision={args.precision}"
        )
        print(f"  schedule '{args.config}': {cfg.n_levels()} levels")
        if no_readout:
            print("  no -readout given: field map (Hz) disabled; warp is unaffected.")

    if args.estmov and not (0 <= args.motion_ref < len(scans)):
        print(f"ffs_blipflip: -motion_ref {args.motion_ref} out of range.", file=sys.stderr)
        return 2

    solve_dtype = torch.float64 if args.precision == "float64" else torch.float32
    result = T.run_topup(
        scans,
        vox,
        cfg,
        pe_shift=args.pe_shift,
        progress=args.verb >= 1,
        solve_dtype=solve_dtype,
        mask_field=not args.no_mask_field,
        estimate_motion=args.estmov,
        motion_ref=args.motion_ref,
        motion_interp=args.motion_interp,
    )

    # ---- outputs ----
    # Choose the scan whose geometry the warp corrects.
    ref_idx = 0 if args.warp_for == "up" else (1 if len(scans) > 1 else 0)
    ref = scans[ref_idx]
    pe_tdim = T._NIFTI_AXIS_TO_TDIM[ref.pe_axis]

    # Geometric undistortion warp. Our forward model (like topup's update()) recovers
    # the undistorted image by resampling the scan at index + disp, so the pull warp
    # used by ffs_nwarp (output(i) = source(i + warp)) IS +disp — no inversion. This maps
    # the reference (distorted) geometry INTO the undistorted "middle" space.
    disp_forward = result.field_hz * (ref.readout * ref.sign)
    pull = disp_forward
    if args.pe_shift and result.pe_shift_vox != 0.0:
        # The reference scan was pre-shifted by +/- shift/2 (scan0 +, scan1 -); fold that
        # constant PE translation into the warp so it corrects the raw (unshifted) input.
        half = result.pe_shift_vox * (0.5 if ref_idx == 0 else -0.5)
        pull = pull + half

    def _save_pe_warp(disp_pe: torch.Tensor, path: str) -> None:
        """Save a single 4-D mm warp whose only nonzero component is on the PE axis."""
        z = torch.zeros_like(disp_pe)
        comps = [z, z, z]  # x, y, z slots, each (nz,ny,nx)
        comps[{0: 0, 1: 1, 2: 2}[ref.pe_axis]] = disp_pe
        save_warp_field(
            comps[0],
            comps[1],
            comps[2],
            path,
            header_info={"affine": affine, "header": header},
            units="mm",
        )

    warp_path = f"{stem}_warp{ext}"
    _save_pe_warp(pull, warp_path)
    if args.verb >= 1:
        print(f"  wrote {warp_path}  (distorted[{pe_dirs[ref_idx]}] -> undistorted 'middle')")

    if not args.no_invwarp:
        # Inverse: undistorted 'middle' -> distorted (forward) geometry, the 1-D PE inverse.
        invwarp = invert_displacement_pe(pull, pe_tdim)
        invwarp_path = f"{stem}_invwarp{ext}"
        _save_pe_warp(invwarp, invwarp_path)
        if args.verb >= 1:
            print(
                f"  wrote {invwarp_path}  (undistorted 'middle' -> distorted[{pe_dirs[ref_idx]}])"
            )

    if not args.no_fmap and not no_readout:
        _save_zyx(f"{stem}_field{ext}", result.field_hz, affine)
        if args.verb >= 1:
            print(f"  wrote {stem}_field{ext}  (off-resonance field, Hz)")

    if not args.no_unwarped:
        undist = torch.stack(result.unwarped, dim=0)  # (S, nz, ny, nx)
        _save_zyx(f"{stem}_unwarped{ext}", undist.movedim(0, -1), affine)  # (nz,ny,nx,S)
        _save_zyx(f"{stem}_mean{ext}", result.mean_unwarped, affine)
        if args.verb >= 1:
            print(f"  wrote {stem}_unwarped{ext}, {stem}_mean{ext}")
    # Jacobian intensity-modulation map. A negative Jacobian means the displacement
    # gradient dropped below -1, i.e. the warp folded over (non-diffeomorphic) --
    # those voxels invert intensity and show up as streaks. Warn regardless of -no_jac;
    # a nonzero fold fraction is a quality signal that the field over-fit (too fine a
    # knot spacing / too little regularisation, e.g. -superhard on modest data).
    jac = T._jacobian_pe(disp_forward, pe_tdim)
    neg_frac = float((jac < 0).float().mean())
    if neg_frac > 0 and args.verb >= 0:
        hint = (
            "raise -jac_penalty (anti-fold barrier)"
            if not args.no_antifold
            else "drop -no_antifold to enable the anti-fold barrier"
        )
        print(
            f"  WARNING: Jacobian is negative in {100 * neg_frac:.3f}% of voxels "
            f"(min {float(jac.min()):.2f}) -- the warp folds there (non-diffeomorphic); "
            f"expect intensity streaks. A real blip pair cannot fold, so {hint}."
        )
    if not args.no_jac:
        _save_zyx(f"{stem}_jac{ext}", jac, affine)
        if args.verb >= 1:
            print(f"  wrote {stem}_jac{ext}")

    # Estimated rigid movement parameters (topup's movpar analogue): one row per scan,
    # decomposed to dx dy dz (voxels) rz rx ry (degrees) in the ffs_moco convention.
    if args.estmov:
        from fastfuncstuff.processing.affine import matrix_to_params

        movpar_path = f"{stem}_movpar.txt"
        rows = []
        for mat in result.motion_matrices:
            p = matrix_to_params(mat.float())[:6].tolist()
            rows.append("  ".join(f"{v:+.6f}" for v in p))
        with open(movpar_path, "w") as fh:
            fh.write("# dx dy dz (vox)  rz rx ry (deg); one row per scan; scan 0 = reference\n")
            fh.write("\n".join(rows) + "\n")
        if args.verb >= 1:
            print(f"  wrote {movpar_path}  (per-scan rigid movement)")

    pe_letter = {0: "i", 1: "j", 2: "k"}[ref.pe_axis]
    if args.verb >= 1:
        print(f"Done: {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(
            f"  apply:  ffs_nwarp -source epi{ext} -master epi{ext} "
            f"-nwarp {warp_path} -jac {pe_letter} -prefix epi_dc{ext}"
        )
        print("          (-jac applies the phase-encode Jacobian intensity correction)")
    return 0


def _save_zyx(path: str, vol_zyx: torch.Tensor, affine: np.ndarray) -> None:
    """Save a (nz,ny,nx[,...]) tensor as NIfTI (nx,ny,nz[,...])."""
    from fastfuncstuff.io.afni import save_nifti

    arr = vol_zyx.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr.transpose(2, 1, 0)
    elif arr.ndim == 4:
        arr = arr.transpose(2, 1, 0, 3)
    save_nifti(np.ascontiguousarray(arr.astype(np.float32)), path, affine=affine)


if __name__ == "__main__":
    sys.exit(main())
