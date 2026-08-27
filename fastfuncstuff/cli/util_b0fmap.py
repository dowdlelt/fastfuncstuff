#!/usr/bin/env python3

"""
ffs_util_b0fmap - B0 (dual-echo GRE) fieldmap -> EPI undistortion warp.

The companion to ``ffs_blipflip``. Where blipflip *estimates* the off-resonance
field from two EPIs with opposing phase-encode polarity, this reads a field that was
*measured* directly, from the phase evolution between two gradient-echo acquisitions:

    f[Hz] = (phi(TE2) - phi(TE1)) / (2*pi * (TE2 - TE1))

Phase unwrapping is delegated to ROMEO (the ``romeo`` binary from MRItools), which
also does the SNR-weighted multi-echo B0 combination, the robust magnitude mask, and
the global n*2pi offset removal. The output is the same 4-D mm warp that ffs_blipflip
writes, so ``ffs_nwarp`` composes it with motion/coreg/atlas transforms unchanged.

    # BIDS phase1/phase2 + magnitude1/2, matched to the EPI it corrects
    ffs_util_b0fmap -phase sub-1_phase1.nii.gz sub-1_phase2.nii.gz \\
                    -magnitude sub-1_magnitude1.nii.gz sub-1_magnitude2.nii.gz \\
                    -epi sub-1_task-rest_bold.nii.gz -prefix sub-1_b0

    # Siemens phasediff (one volume that is already the inter-echo difference)
    ffs_util_b0fmap -phasediff sub-1_phasediff.nii.gz \\
                    -magnitude sub-1_magnitude1.nii.gz \\
                    -epi sub-1_bold.nii.gz -prefix sub-1_b0

    # -> sub-1_b0_warp.nii.gz     (distorted EPI -> undistorted; on the EPI grid)
    #    sub-1_b0_invwarp.nii.gz  (the inverse)
    #    sub-1_b0_field.nii.gz    (conditioned off-resonance field, Hz)
    #    sub-1_b0_wfmag.nii.gz    (magnitude forward-warped to look like the EPI)
    #    sub-1_b0_epi2fmap.aff12.1D  (the matching affine)

    # apply, exactly as with ffs_blipflip:
    ffs_nwarp -source epi.nii.gz -master epi.nii.gz \\
              -nwarp sub-1_b0_warp.nii.gz -jac j -prefix epi_dc.nii.gz

Echo times, phase-encode direction and readout time are read from the BIDS JSON
sidecars when they sit next to the images; -te / -pe_dir / -readout override.

METHOD / CREDIT
  The processing chain (unwrap -> Hz -> voxel displacement map -> forward-warp the
  magnitude -> coregister that to the EPI -> resample) is that of the SPM FieldMap
  toolbox: Jezzard & Balaban (1995), MRM 34:65-73; Hutton et al. (2002), NeuroImage
  16:217-240; Jenkinson (2003), MRM 49:193-197. Original toolbox by Chloe Hutton and
  Jesper Andersson, FIL/Wellcome Centre for Human Neuroimaging. Phase unwrapping and
  the B0 combination are performed by ROMEO, not reimplemented here: Dymerska et al.
  (2021), MRM 85:2294-2308. This is an independent implementation, not SPM code.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

import numpy as np
import torch

from fastfuncstuff.cli_help import FfsHelpFormatter
from fastfuncstuff.utils import REGISTRATION_TF32, configure_torch_backends


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ffs_util_b0fmap",
        description="B0 dual-echo GRE fieldmap -> off-resonance field -> EPI "
        "undistortion warp (SPM FieldMap-style chain; ROMEO does the unwrapping).",
        formatter_class=FfsHelpFormatter,
        epilog=__doc__.split("METHOD / CREDIT")[1].strip(),
    )

    inp = parser.add_argument_group("Fieldmap input (pick one form)")
    inp.add_argument(
        "-phase",
        nargs="+",
        metavar="FILE",
        help="Phase image(s), one per echo (BIDS phase1 phase2). Raw scanner units are "
        "fine — they are converted to radians from the nominal quantisation range "
        "(see -phase_units / -phase_range).",
    )
    inp.add_argument(
        "-phasediff",
        metavar="FILE",
        help="Siemens phasediff: a single volume that is already the inter-echo phase "
        "difference. Needs -te T1 T2 (or a sidecar with EchoTime1/EchoTime2).",
    )
    inp.add_argument(
        "-fieldmap",
        metavar="FILE",
        help="A ready-made fieldmap in Hz (BIDS 'fieldmap' form). Skips ROMEO entirely; "
        "still conditioned, matched and converted to a warp.",
    )
    inp.add_argument(
        "-magnitude",
        nargs="+",
        metavar="FILE",
        help="Magnitude image(s). The first is the reference used for masking and for "
        "the EPI match. Required except with -fieldmap + -no_match.",
    )
    inp.add_argument(
        "-te",
        "-echo_times",
        nargs="+",
        type=float,
        metavar="MS",
        help="Echo times in ms. Two values for -phase/-phasediff. Read from the BIDS "
        "sidecars when omitted.",
    )

    tgt = parser.add_argument_group("Target EPI")
    tgt.add_argument(
        "-epi",
        metavar="FILE",
        help="The EPI to correct. Its geometry defines the output warp's grid, and the "
        "forward-warped magnitude is affine-registered to it. Omit with -no_match to "
        "write the warp on the fieldmap's own grid.",
    )
    tgt.add_argument(
        "-pe_dir",
        "-pe-dir",
        metavar="DIR",
        help="EPI phase-encode direction: i/j/k (or x/y/z) with optional trailing '-'. "
        "Read from the EPI sidecar when omitted.",
    )
    tgt.add_argument(
        "-readout",
        "-total_readout_time",
        type=float,
        metavar="SEC",
        help="EPI total readout time (s). Read from the EPI sidecar when omitted. This "
        "sets the Hz->voxel scale, so unlike ffs_blipflip it is NOT optional: a measured "
        "field has an absolute scale and the warp inherits it.",
    )
    tgt.add_argument(
        "-collapse",
        choices=("median", "mean"),
        default="median",
        help="How to collapse a 4-D EPI to one volume for the match.",
    )

    cond = parser.add_argument_group("Field conditioning")
    cond.add_argument(
        "-fwhm",
        type=float,
        default=4.0,
        help="FWHM (mm) of the SNR-weighted smoothing inside the mask. Much lower than "
        "SPM's 10 mm on purpose: ROMEO's B0 is already an SNR-weighted combination "
        "across echoes, so heavy smoothing here only blurs the sharp sinus gradients.",
    )
    cond.add_argument(
        "-extend",
        type=float,
        default=16.0,
        help="How far (mm) to extrapolate the field past the mask. A measured field stops "
        "at tissue; without this the mask edge is a step in the field and therefore a tear "
        "in the warp, right where orbitofrontal/temporal signal needs correcting.",
    )
    cond.add_argument(
        "-rolloff",
        type=float,
        default=8.0,
        help="Length scale (mm) of the exponential decay to zero beyond -extend, ALONG PE "
        "only, so far air is identity instead of dragging tissue into the pad margin. "
        "Widened per column where needed to respect -jac_margin.",
    )
    cond.add_argument(
        "-no_taper",
        action="store_true",
        help="Never decay: hold the boundary value out to the FOV edge along PE.",
    )
    cond.add_argument(
        "-jac_margin",
        type=float,
        default=0.5,
        help="Cap on |d(displacement)/d(PE)| in the extended region, which bounds the "
        "Jacobian below by 1-margin and so keeps the warp from folding out there. The "
        "decay length is widened per column as needed to respect it. 0 disables.",
    )
    cond.add_argument(
        "-mask",
        default="robustmask",
        help="ROMEO -k mask option: robustmask, nomask, 'qualitymask <thr>', or a mask file.",
    )
    cond.add_argument(
        "-no_global_correct",
        action="store_true",
        help="Do not remove the global n*2pi phase offset (ROMEO -g). Leaving it in shows "
        "up as a constant Hz bias, i.e. a constant shift of the whole brain along PE.",
    )
    cond.add_argument(
        "-phase_units",
        choices=("auto", "radians", "scanner"),
        default="auto",
        help="Units of the input phase. 'scanner' converts to radians from the nominal "
        "quantisation range; 'radians' passes it through; 'auto' decides by data range. "
        "We never let ROMEO do the rescale: it maps the OBSERVED [min,max] onto [-pi,pi], "
        "which silently over-stretches phase that does not span its full range.",
    )
    cond.add_argument(
        "-phase_range",
        nargs=2,
        type=float,
        metavar=("LO", "HI"),
        help="Nominal quantisation range of raw scanner phase, e.g. '0 4095' (Siemens "
        "12-bit) or '-4096 4094'. Inferred from the data by default; give it explicitly "
        "when the phase does not span its full range (a short-dTE phasediff, a pre-masked "
        "image), where inference from the observed extremes would under-estimate it.",
    )
    cond.add_argument(
        "-romeo_opts",
        default="",
        help="Extra flags passed verbatim to romeo (e.g. '--merge-regions -s 3').",
    )
    cond.add_argument("-romeo_bin", default="romeo", help="ROMEO executable.")

    match = parser.add_argument_group("EPI matching")
    match.add_argument(
        "-no_match",
        action="store_true",
        help="Skip the affine match; write everything on the fieldmap's own grid. Use when "
        "the fieldmap and EPI are already on the same grid, or to inspect the raw field.",
    )
    match.add_argument(
        "-match_cost",
        default="lpa",
        help="Cost for the magnitude<->EPI registration. lpa (absolute local Pearson), "
        "NOT lpc: the forward-warped magnitude and the EPI are both brain-bright with the "
        "same polarity, and lpc's signed form actively seeks an anti-correlated placement "
        "— measured on data where the truth was exactly identity, lpc landed 43 mm and 15 "
        "deg out while lpa was within 1 mm and 1 deg.",
    )
    match.add_argument(
        "-match_dof",
        default="rigid",
        choices=("rigid", "affine", "epi"),
        help="Degrees of freedom for the match. Rigid is right for same-session data.",
    )
    match.add_argument(
        "-1Dmatrix_apply",
        dest="matrix_apply",
        metavar="FILE",
        help="Use this EPI->fieldmap .aff12.1D instead of registering (e.g. one computed "
        "once and reused across the runs a fieldmap serves).",
    )

    out = parser.add_argument_group("Outputs")
    out.add_argument("-prefix", required=True, help="Output prefix (stem[.nii.gz]).")
    out.add_argument("-no_invwarp", action="store_true", help="Do not write the inverse warp.")
    out.add_argument("-no_jac", action="store_true", help="Do not write the Jacobian map.")
    out.add_argument(
        "-save_raw", action="store_true", help="Also write the unconditioned ROMEO field."
    )
    out.add_argument(
        "-romeo_dir",
        metavar="DIR",
        help="Keep ROMEO's own outputs (unwrapped phase, quality, SNR) in this directory "
        "instead of a temporary one.",
    )

    misc = parser.add_argument_group("Misc")
    misc.add_argument("-device", default="cuda", help="Compute device (cuda/cpu/mps).")
    misc.add_argument("-verb", type=int, default=1, help="Verbosity (0/1/2).")
    return parser


def _load(path: str, collapse: str | None = None) -> tuple[np.ndarray, np.ndarray, object]:
    """Load a NIfTI as ``(nx, ny, nz[, t])`` float32 + affine + header."""
    from fastfuncstuff.io.afni import load_nifti

    img = load_nifti(path)
    arr = np.asarray(img.dataobj, dtype=np.float32)
    if collapse is not None and arr.ndim == 4:
        arr = np.median(arr, axis=3) if collapse == "median" else arr.mean(axis=3)
    return arr, np.asarray(img.affine), img.header


def _to_tensor(arr: np.ndarray, device) -> torch.Tensor:
    """(nx, ny, nz) numpy -> (nz, ny, nx) tensor, the layout processing/ uses."""
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 1, 0))).float().to(device)


def _save_zyx(path: str, vol_zyx: torch.Tensor, affine: np.ndarray) -> None:
    from fastfuncstuff.io.afni import save_nifti

    arr = vol_zyx.detach().cpu().numpy()
    arr = arr.transpose(2, 1, 0) if arr.ndim == 3 else arr.transpose(2, 1, 0, 3)
    save_nifti(np.ascontiguousarray(arr.astype(np.float32)), path, affine=affine)


def _voxel_sizes_zyx(affine: np.ndarray) -> tuple[float, float, float]:
    vx, vy, vz = (float(np.linalg.norm(affine[:3, i])) for i in range(3))
    return (vz, vy, vx)


def main(argv: list[str] | None = None) -> int:
    from fastfuncstuff.cli_utils import parse_device_arg, parse_prefix
    from fastfuncstuff.processing import b0fmap as B
    from fastfuncstuff.processing.io import save_warp_field
    from fastfuncstuff.processing.medic import PE_AXIS_MAP

    args = create_parser().parse_args(argv)
    device, _, _ = parse_device_arg(args.device)
    configure_torch_backends(device, tf32=REGISTRATION_TF32)
    pinfo = parse_prefix(args.prefix)
    stem, ext = pinfo.stem, pinfo.nifti_ext
    verb = args.verb

    forms = [bool(args.phase), bool(args.phasediff), bool(args.fieldmap)]
    if sum(forms) != 1:
        print(
            "ffs_util_b0fmap: give exactly one of -phase, -phasediff, -fieldmap.",
            file=sys.stderr,
        )
        return 2
    if not args.magnitude and not (args.fieldmap and args.no_match):
        print("ffs_util_b0fmap: -magnitude is required.", file=sys.stderr)
        return 2
    if not args.epi and not args.no_match:
        print("ffs_util_b0fmap: -epi is required (or pass -no_match).", file=sys.stderr)
        return 2

    # ---- EPI geometry: PE direction + readout ----
    pe_dir, readout = args.pe_dir, args.readout
    if args.epi:
        js_pe, js_ro = B.read_epi_geometry(args.epi)
        pe_dir = pe_dir or js_pe
        readout = readout if readout is not None else js_ro
    if not pe_dir:
        print(
            "ffs_util_b0fmap: no phase-encode direction — pass -pe_dir (no "
            "PhaseEncodingDirection in the EPI sidecar).",
            file=sys.stderr,
        )
        return 2
    if readout is None:
        print(
            "ffs_util_b0fmap: no readout time — pass -readout (no TotalReadoutTime in "
            "the EPI sidecar). A measured field has an absolute Hz scale, so unlike "
            "ffs_blipflip the readout cannot be left out.",
            file=sys.stderr,
        )
        return 2
    if pe_dir not in PE_AXIS_MAP:
        print(f"ffs_util_b0fmap: bad phase-encode direction {pe_dir!r}", file=sys.stderr)
        return 2

    # ---- the field, in Hz, on the fieldmap grid ----
    mag_arr = affine = header = None
    if args.magnitude:
        mag_arr, affine, header = _load(args.magnitude[0])

    if args.fieldmap:
        field_np, f_affine, f_header = _load(args.fieldmap)
        if affine is None:
            affine, header = f_affine, f_header
        mask_np = np.isfinite(field_np) & (field_np != 0)
        weight_np = None
        if verb >= 1:
            print(f"ffs_util_b0fmap: using precomputed Hz fieldmap {args.fieldmap}")
    else:
        is_pd = bool(args.phasediff)
        ph_paths = [args.phasediff] if is_pd else list(args.phase)
        tes = args.te
        if tes is None:
            tes = B.read_echo_times(ph_paths, phasediff=is_pd)
            if tes is None:
                print(
                    "ffs_util_b0fmap: no echo times — pass -te (the sidecars have no "
                    "EchoTime/EchoTime1+EchoTime2).",
                    file=sys.stderr,
                )
                return 2
        elif is_pd:
            if len(tes) == 2:
                tes = [abs(tes[1] - tes[0])]
            elif len(tes) != 1:
                print("ffs_util_b0fmap: -phasediff wants -te T1 T2 (or one dTE).", file=sys.stderr)
                return 2
        phase_np = np.stack([_load(p)[0] for p in ph_paths], axis=-1)
        if affine is None:
            _, affine, header = _load(ph_paths[0])
        mags = [_load(p)[0] for p in (args.magnitude or ph_paths)]
        mag4d = np.stack(mags, axis=-1)
        if verb >= 1:
            kind = "phasediff (dTE)" if is_pd else "multi-echo phase"
            print(
                f"ffs_util_b0fmap: {kind}, TE = "
                + ", ".join(f"{t:g}" for t in tes)
                + f" ms, grid {phase_np.shape[:3]}"
            )
        romeo = B.run_romeo(
            phase_np,
            mag4d,
            list(tes),
            affine,
            outdir=args.romeo_dir,
            romeo_bin=args.romeo_bin,
            mask=args.mask,
            correct_global=not args.no_global_correct,
            phase_units=args.phase_units,
            phase_range=tuple(args.phase_range) if args.phase_range else None,
            extra_args=args.romeo_opts.split() if args.romeo_opts else None,
            verbose=verb >= 2,
        )
        field_np, mask_np = romeo.b0_hz, romeo.mask
        weight_np = romeo.snr if romeo.snr is not None else romeo.quality

    assert affine is not None
    vox = _voxel_sizes_zyx(affine)
    field_raw = _to_tensor(field_np, device)
    mask = _to_tensor(mask_np.astype(np.float32), device) > 0.5
    weight = _to_tensor(weight_np, device) if weight_np is not None else None
    if verb >= 1:
        inm = field_raw[mask]
        print(
            f"  field: {int(mask.sum()):,} voxels in mask, "
            f"Hz 1-99 pct [{torch.quantile(inm, 0.01):.0f}, {torch.quantile(inm, 0.99):.0f}]"
        )

    pe_axis = PE_AXIS_MAP[pe_dir]
    pe_tdim = {0: 2, 1: 1, 2: 0}[pe_axis]
    field, support = B.condition_field(
        field_raw,
        mask,
        vox,
        pe_tdim,
        weight=weight,
        fwhm_mm=args.fwhm,
        extend_mm=args.extend,
        rolloff_mm=1e9 if args.no_taper else args.rolloff,
        disp_per_unit=readout,
        jac_margin=args.jac_margin,
    )
    if verb >= 1:
        print(
            f"  conditioned: fwhm {args.fwhm:g} mm in-mask; extended along {pe_dir} "
            f"by {args.extend:g} mm then decaying over "
            + ("no decay" if args.no_taper else f"{args.rolloff:g} mm")
            + f" ({int(support.sum()):,} voxels of support)"
        )

    # ---- warp on the fieldmap grid, and the synthetic distorted magnitude ----
    # The displacement is in EPI PE voxels; expressed on the fieldmap grid it has to be
    # rescaled by the PE voxel-size ratio, since the same physical mm shift is a
    # different number of voxels on a different grid.
    fmap_pe_mm = _voxel_sizes_zyx(affine)[{0: 2, 1: 1, 2: 0}[pe_axis]]
    epi_pe_mm = fmap_pe_mm
    if args.epi:
        _, epi_affine_probe, _ = _load(args.epi, collapse=args.collapse)
        epi_pe_mm = _voxel_sizes_zyx(epi_affine_probe)[{0: 2, 1: 1, 2: 0}[pe_axis]]

    _, inv_f, pe_tdim = B.field_to_pe_warp(field, readout * (epi_pe_mm / fmap_pe_mm), pe_dir)

    wfmag = None
    if mag_arr is not None:
        wfmag = B.synthesize_distorted(_to_tensor(mag_arr, device), inv_f, pe_tdim)
        _save_zyx(f"{stem}_wfmag{ext}", wfmag, affine)
        if verb >= 1:
            print(f"  wrote {stem}_wfmag{ext}  (magnitude forward-warped to look like the EPI)")

    # ---- match to the EPI and move the field onto its grid ----
    out_affine, out_field = affine, field
    if args.no_match:
        if verb >= 1:
            print("  -no_match: warp stays on the fieldmap grid")
    else:
        from fastfuncstuff.processing.affine import load_matrix_1D, save_matrix_1D
        from fastfuncstuff.processing.allineate import AffineAlignConfig, allineate
        from fastfuncstuff.processing.grid import resample_to_grid

        epi_arr, epi_affine, epi_header = _load(args.epi, collapse=args.collapse)
        epi = _to_tensor(epi_arr, device)
        if args.matrix_apply:
            matrix = load_matrix_1D(args.matrix_apply, base_affine=epi_affine, source_affine=affine)
            matrix = torch.as_tensor(matrix, dtype=torch.float32, device=device)
            if verb >= 1:
                print(f"  using supplied matrix {args.matrix_apply}")
        else:
            if wfmag is None:
                print("ffs_util_b0fmap: -magnitude needed to match to the EPI.", file=sys.stderr)
                return 2
            if verb >= 1:
                print(f"  matching forward-warped magnitude -> {args.epi} ({args.match_cost})")
            cfg = AffineAlignConfig(
                dof=args.match_dof,
                cost=args.match_cost,
                # Same-session fieldmap and EPI: the scanner headers already place them
                # within a few mm, so the coarse ranges are halved to keep the search
                # near that start rather than letting it wander.
                range_scale=0.5,
                device=str(device),
                verb=max(0, verb - 1),
            )
            matrix, _ = allineate(
                epi,
                wfmag,
                cfg,
                base_header={"affine": epi_affine, "header": epi_header},
                source_header={"affine": affine, "header": header},
            )
            save_matrix_1D(
                matrix,
                f"{stem}_epi2fmap.aff12.1D",
                base_affine=epi_affine,
                source_affine=affine,
            )
            if verb >= 1:
                print(f"  wrote {stem}_epi2fmap.aff12.1D  (EPI -> fieldmap)")

        # resample_to_grid maps out-voxel -> src-voxel via inv(src_affine) @ out_affine.
        # We want that composite to BE the match matrix (EPI voxel -> fieldmap voxel),
        # so hand it a source affine of epi_affine @ inv(matrix).
        m = matrix.detach().cpu().numpy().astype(np.float64)
        src_affine_eff = epi_affine @ np.linalg.inv(m)
        out_field = resample_to_grid(
            field, src_affine_eff, tuple(epi.shape), epi_affine, interp="linear"
        )
        out_affine = epi_affine
        if verb >= 1:
            print(f"  field resampled onto the EPI grid {tuple(epi.shape)}")

    # ---- outputs ----
    pull, inv, pe_tdim = B.field_to_pe_warp(out_field, readout, pe_dir)

    def _save_pe_warp(disp_pe: torch.Tensor, path: str) -> None:
        z = torch.zeros_like(disp_pe)
        comps = [z, z, z]
        comps[pe_axis] = disp_pe
        save_warp_field(
            comps[0],
            comps[1],
            comps[2],
            path,
            header_info={"affine": out_affine, "header": header},
            units="mm",
        )

    _save_pe_warp(pull, f"{stem}_warp{ext}")
    _save_zyx(f"{stem}_field{ext}", out_field, out_affine)
    if verb >= 1:
        print(f"  wrote {stem}_warp{ext}  (distorted[{pe_dir}] -> undistorted)")
        print(f"  wrote {stem}_field{ext}  (off-resonance field, Hz)")
    if not args.no_invwarp:
        _save_pe_warp(inv, f"{stem}_invwarp{ext}")
        if verb >= 1:
            print(f"  wrote {stem}_invwarp{ext}  (undistorted -> distorted[{pe_dir}])")
    if args.save_raw:
        _save_zyx(f"{stem}_field_raw{ext}", field_raw, affine)
        _save_zyx(f"{stem}_mask{ext}", mask.float(), affine)
    if not args.no_jac:
        from fastfuncstuff.processing.topup import _jacobian_pe

        jac = _jacobian_pe(pull, pe_tdim)
        _save_zyx(f"{stem}_jac{ext}", jac, out_affine)
        neg = jac < 0
        # A fold only matters where there is tissue to tear. ROMEO's robustmask is
        # generous -- it keeps scalp and neck, whose low-SNR phase unwraps poorly --
        # so the great majority of folds sit out there and are harmless once the warp
        # is applied to brain. Reporting the raw whole-volume fraction cries wolf.
        brain = None
        if mag_arr is not None:
            from fastfuncstuff.processing.mask import automask

            brain = automask(_to_tensor(mag_arr, device), device=device).bool()
            if brain.shape != neg.shape:  # field was moved onto the EPI grid
                brain = None
        n_brain = int((neg & brain).sum()) if brain is not None else None
        if bool(neg.any()):
            where = (
                f"{n_brain:,} of them inside the brain"
                if n_brain is not None
                else "brain fraction not assessed (no magnitude on this grid)"
            )
            print(
                f"  Jacobian negative in {int(neg.sum()):,} voxels "
                f"({100 * float(neg.float().mean()):.2f}%), {where}; "
                f"min {float(jac.min()):.2f}."
            )
            if n_brain is None or n_brain > 0.002 * jac.numel():
                print(
                    "  WARNING: that is a lot of in-brain folding. A measured field that "
                    "folds in tissue usually means the readout time is wrong for this "
                    f"field, or the unwrap failed; check {stem}_field{ext} near the sinuses."
                )

    if verb >= 1:
        print(f"Done: {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(
            f"  apply:  ffs_nwarp -source epi{ext} -master epi{ext} "
            f"-nwarp {stem}_warp{ext} -jac {pe_dir[0]} -prefix epi_dc{ext}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
