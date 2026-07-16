"""``ffs_segment`` — GPU Unified Segmentation (SPM ``spm_preproc8`` / New Segment).

One EM fit that **jointly** estimates, from a single structural (or EPI) volume:
  * a **bias field** (smooth multiplicative intensity inhomogeneity / INU),
  * **tissue classes** (a Gaussian mixture: GM, WM, CSF, … as many as the TPM has),
  * a **deformation** aligning a tissue-probability template (TPM) to the subject.

Because the three are estimated together, each helps the others: the bias field
lets the mixture see true tissue intensities, the warped TPM tells the mixture where
each tissue is likely to be, and the tissue labels drive the warp. This is the
principled, bias-aware successor to the ``ffs_bbr -target tissue`` synthesis cost.

Method + validation: ``../fmri_wiki/concepts/Unified Segmentation.md``. Library:
``processing/segment.py``.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import threading
import time
from contextlib import contextmanager

import numpy as np
import torch

from fastfuncstuff.processing.affine import load_matrix_chain
from fastfuncstuff.processing.io import load_image, save_image, save_warp_field
from fastfuncstuff.processing.segment import (
    cast_template_to_input,
    fit_segment,
    full_resolution_warp,
    input_in_template,
    load_tpm,
    plot_intensity_fit,
    segment_apply,
    undistort_input,
)

_IMG_EXTS = (".nii.gz", ".nii.zst", ".nii", ".HEAD", ".BRIK.gz", ".BRIK")


@contextmanager
def _step(msg: str, *, enabled: bool = True):
    """Announce a stage; on a TTY, animate a spinner while it runs (for slow steps
    without their own tqdm bar). Prints the elapsed time when the block finishes."""
    if not enabled:
        yield
        return
    t0 = time.time()
    tty = sys.stderr.isatty()
    stop = threading.Event()

    def _spin() -> None:
        for ch in itertools.cycle("|/-\\"):
            if stop.is_set():
                break
            print(f"\r{msg} {ch}", end="", file=sys.stderr, flush=True)
            time.sleep(0.1)

    worker = threading.Thread(target=_spin, daemon=True) if tty else None
    if worker is not None:
        worker.start()
    else:
        print(f"{msg} ...", flush=True)
    try:
        yield
    finally:
        stop.set()
        if worker is not None:
            worker.join()
            print(f"\r{msg} done ({time.time() - t0:.1f}s)", file=sys.stderr, flush=True)
        else:
            print(f"{msg} done ({time.time() - t0:.1f}s)", flush=True)


def _strip_ext(prefix: str) -> str:
    """Drop a trailing image extension from a prefix so ``t1_seg.nii.gz`` →
    ``t1_seg`` and outputs are named ``t1_seg_c1.nii.gz`` (not
    ``t1_seg.nii.gz_c1``)."""
    for ext in _IMG_EXTS:
        if prefix.endswith(ext):
            return prefix[: -len(ext)]
    return prefix


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ffs_segment",
        description=(
            "Unified Segmentation: joint bias-field correction + tissue classification\n"
            "+ TPM-driven spatial normalisation, in one GPU EM fit. Reproduces SPM's\n"
            "New Segment (spm_preproc8) — GM/WM tissue maps validated to Dice >0.92 —\n"
            "in seconds rather than minutes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "WHAT THE KNOBS DO (all have sensible SPM defaults; touch these only to\n"
            "steer a specific behaviour):\n"
            "\n"
            "  -ngaus g1 g2 ...   How many Gaussians model each tissue's intensity\n"
            "                     histogram. One per class is enough for a clean\n"
            "                     unimodal tissue (GM, WM); give more where a class\n"
            "                     spans several intensities (bone+marrow, air+scanner\n"
            "                     background). Length MUST equal the number of TPM\n"
            "                     classes. Default matches SPM: 1 1 2 3 4 2 for the\n"
            "                     6-class MNI TPM (GM WM CSF bone soft air).\n"
            "\n"
            "  -biasfwhm MM       Spatial scale (FWHM, mm) of the bias field. LARGER =\n"
            "                     smoother, gentler shading correction (safe, the\n"
            "                     default 60 mm); SMALLER = the field can follow finer\n"
            "                     intensity variation but risks eating real anatomy.\n"
            "                     Drop toward 30-40 for strong surface-coil shading.\n"
            "  -biasreg R         How hard to penalise a wiggly bias field. HIGHER =\n"
            "                     stiffer/flatter field (default 1e-4). Raise it if the\n"
            "                     bias correction looks like it's chasing tissue.\n"
            "\n"
            "  -reg A M B         Deformation stiffness = (absolute, membrane, bending)\n"
            "                     penalties. Membrane resists stretch, bending resists\n"
            "                     folding; HIGHER = stiffer, more conservative warp.\n"
            "                     Default 0 0 0.1. Increase to keep the warp tame on\n"
            "                     low-contrast or noisy data.\n"
            "\n"
            "  -samp MM           Voxel sampling step (mm) for the fit. This is a\n"
            "                     SPEED/accuracy dial, not a quality knob: 3 mm (default)\n"
            "                     matches SPM and is plenty; 2 mm is slower and slightly\n"
            "                     finer. Outputs are always written at full resolution.\n"
            "  -niter N           Max EM iterations (default 20); it early-stops on\n"
            "                     convergence, so this is just a ceiling.\n"
            "  -no_warp           Skip the deformation entirely (bias + tissue only).\n"
            "                     Use when the subject is already in template space or\n"
            "                     you only want bias correction + a quick classification.\n"
            "  -mrf M             Markov-random-field cleanup strength on the final\n"
            "                     tissue maps. Pulls each voxel toward the class its\n"
            "                     neighbours agree on — removes speckle and fills small\n"
            "                     holes. Default 1 (SPM); 0 turns it off (rawer, noisier\n"
            "                     posteriors); raise for smoother, more contiguous labels.\n"
            "                     On very high-res data (<=0.8 mm) try 0.5 to avoid\n"
            "                     eroding thin cortex.\n"
            "  -cleanup L         Morphological brain extraction of GM/WM/CSF: grows a\n"
            "                     brain mask from WM and strips dura/skull/eyeball voxels\n"
            "                     misclassified as brain tissue (so c1/c2/c3 are usable\n"
            "                     as masks). 0=off, 1=default, 2=stricter. Needs a TPM\n"
            "                     with >3 classes (GM WM CSF + others).\n"
            "\n"
            "AFFINE INIT (aligns the input into the TEMPLATE's space):\n"
            "  -1Dmatrix M ...    The .aff12.1D that aligns your INPUT to the TEMPLATE —\n"
            "                     i.e. exactly what ffs_allineate/3dAllineate writes with\n"
            "                     -base <template> -source <input>. (In AFNI's base->source\n"
            "                     convention that matrix maps template->input; you do NOT\n"
            "                     invert it — pass it as-is.) Stack several base-side->\n"
            "                     source-side (AFNI -nwarp order). WITHOUT it the input is\n"
            "                     assumed already in template space (identity) — fine for\n"
            "                     AC-PC'd anatomicals, risky otherwise.\n"
            "\n"
            "EPI / PE-MODE (distortion correction, the novel use):\n"
            "  Segment the EPI itself with -pe_axis set to the phase-encode axis. The\n"
            "  deformation is then constrained to that ONE axis, so it IS the EPI's PE\n"
            "  distortion field (composable, like ffs_rbr/ffs_medic). Feed the subject's\n"
            "  own native tissue maps as a 4-D -tpm (stack c1..cN) and the anat->epi\n"
            "  affine as -1Dmatrix, and one fit gives: EPI bias correction + EPI-space\n"
            "  tissue maps + the PE warp. See the PE example below.\n"
            "\n"
            "OUTPUTS (prefix_*); note which SPACE each lives in:\n"
            "  INPUT space (the -input grid): c1..cN (tissue posteriors); biascorrected;\n"
            "    biasfield; undistorted (deformation removed, PE-mode = the corrected EPI\n"
            "    in its own grid — analysis-ready).\n"
            "  TEMPLATE space (the -tpm grid; = the ANAT in EPI mode, so these overlay the\n"
            "    anat): in_template_initial (affine only) and in_template (affine + warp) —\n"
            "    the input cast into template space, a before/after pair like ffs_rbr's\n"
            "    initial_epi_in_anat / epi_in_anat.\n"
            "  warp: the input-space deformation, composable mm for ffs_nwarp. It is a\n"
            "    SOURCE-SIDE (EPI-space) warp like MEDIC/RBR — it corrects the EPI in its\n"
            "    own space, THEN the affine(s) normalise. In an ffs_nwarp chain it goes\n"
            "    with the other EPI-space corrections (fieldmap, moco), to the RIGHT of\n"
            "    the epi->anat affines (applied before them), NOT as the final warp:\n"
            "      ffs_nwarp -source raw_ts -master anat -prefix ts_in_anat \\\n"
            "        -nwarp 'ref2anat.aff12.1D epi2ref.aff12.1D epi_seg_warp.nii.gz \\\n"
            "                fieldmap_warp.nii moco.aff12.1D'\n"
            "    (segment the fieldmap-corrected epi_mean so the PE warp is a residual and\n"
            "    doesn't double-correct with the fieldmap). Skipped with -no_warp.\n"
            "\n"
            "Anatomical -> MNI:\n"
            "  ffs_allineate -base MNI_T1.nii -source T1.nii -1Dmatrix_save t1_to_mni.aff12.1D\n"
            "  ffs_segment -input T1.nii -tpm TPM.nii -1Dmatrix t1_to_mni.aff12.1D -prefix t1_seg\n"
            "\n"
            "EPI distortion correction (PE = y/AP), native TPMs as template:\n"
            "  ffs_segment -input epi_mean.nii -tpm subj_native_c1to6.nii \\\n"
            "              -1Dmatrix anat2epi.aff12.1D -pe_axis y -prefix epi_seg\n"
        ),
    )
    req = p.add_argument_group("required inputs")
    req.add_argument(
        "-input",
        required=True,
        nargs="+",
        help="Structural (or EPI) volume(s) to segment. Give ONE image, or SEVERAL "
        "co-registered channels (e.g. T1 T2 PD) for multi-spectral segmentation — they "
        "must already be on the same grid. Each channel gets its own bias field; the "
        "tissue mixture is joint across channels (as SPM's New Segment).",
    )
    req.add_argument(
        "-tpm",
        required=True,
        help="4-D tissue-probability template (e.g. SPM's tpm/TPM.nii). The number of "
        "classes is read from its 4th dimension — any count works.",
    )
    req.add_argument("-prefix", required=True, help="Output prefix for all written files")

    aff = p.add_argument_group("affine initialisation")
    aff.add_argument(
        "-1Dmatrix",
        dest="matrix",
        nargs="+",
        default=None,
        metavar="AFF",
        help="The .aff12.1D aligning input->template (as written by ffs_allineate "
        "-base template -source input); pass as-is, base-side->source-side. Omit to "
        "assume the input is already roughly in template space.",
    )
    aff.add_argument(
        "-pe_axis",
        default=None,
        choices=("x", "y", "z"),
        help="Phase-encode axis for EPI PE-mode: constrain the deformation to this voxel "
        "axis so the warp is the 1-D EPI distortion field (like ffs_rbr). Omit for a "
        "full 3-D deformation (anatomicals).",
    )
    aff.add_argument(
        "-tpm_source",
        default=None,
        help="A template-space image (3-D or 4-D; e.g. the anat or native tissue maps the "
        "TPM came from) to resample into the INPUT's space via the fitted warp+affine, "
        "written as prefix_tpm_source_in_epi, plus prefix_tpm_source_in_epi_initial "
        "(affine only, before the warp) as a before/after baseline. QC / anat-in-EPI.",
    )

    knobs = p.add_argument_group("model knobs (SPM defaults)")
    knobs.add_argument(
        "-ngaus",
        type=int,
        nargs="+",
        default=None,
        help="Gaussians per tissue class (length = #TPM classes). Default 1 1 2 3 4 2 "
        "for a 6-class TPM.",
    )
    knobs.add_argument(
        "-biasfwhm", type=float, default=60.0, help="Bias field FWHM in mm (default 60)"
    )
    knobs.add_argument(
        "-biasreg", type=float, default=1e-4, help="Bias field regularisation (default 1e-4)"
    )
    knobs.add_argument(
        "-reg",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.1),
        metavar=("ABS", "MEM", "BEND"),
        help="Deformation (absolute, membrane, bending) penalties (default 0 0 0.1)",
    )
    knobs.add_argument(
        "-samp",
        type=float,
        default=3.0,
        help="Voxel sampling step in mm for the fit (default 3, matching SPM). This is a "
        "speed/accuracy dial: smaller sees finer detail (helps a thin distortion feature) "
        "but is slower. Memory is bounded by chunking, so a fine samp no longer OOMs.",
    )
    knobs.add_argument(
        "-fit_chunk",
        type=int,
        default=None,
        metavar="N",
        help="Samples processed per chunk during the fit (bounds VRAM). Default: sized "
        "automatically from free memory. Lower it if you still hit OOM at a very fine "
        "-samp; it does not change the result, only peak memory.",
    )
    knobs.add_argument("-niter", type=int, default=20, help="Max EM iterations (default 20)")
    knobs.add_argument(
        "-tol",
        type=float,
        default=1e-4,
        help="Convergence tolerance: stop when the relative log-likelihood change drops "
        "below this (default 1e-4). Larger = stop sooner/rougher; smaller = run longer.",
    )
    knobs.add_argument(
        "-warp_smooth",
        type=float,
        default=0.8,
        help="Deformation smoothing (grid-node sigma) applied each iteration to keep the "
        "warp smooth, not speckly (default 0.8; 0 = off). Raise for a smoother warp; "
        "LOWER to let more data-driven displacement survive (warps more, but risks speckle).",
    )
    knobs.add_argument(
        "-warp_solver",
        default="adam",
        choices=("adam", "gn"),
        help="Deformation optimiser. 'adam' (default) = autograd + Sobolev smoothing "
        "(fast, robust; -warp_lr/-warp_smooth/-warp_focus apply). 'gn' = SPM's "
        "Gauss-Newton (per-node GN Hessian + conjugate-gradient solve + Armijo "
        "backtracking): no learning rate, converges in ~2-3 -warp_iters, and stays "
        "monotone so it won't inflate at high -niter. For 'gn' give -reg a non-zero "
        "membrane term (e.g. -reg 0 0.001 0.1) for a well-posed solve.",
    )
    knobs.add_argument(
        "-warp_lr",
        type=float,
        default=1.0,
        help="How AGGRESSIVELY the deformation moves overall (Adam step size, default 1.0). "
        "Raise (2-4) if the warp isn't stretching enough in general; too high oscillates. "
        "This is the main global-aggressiveness dial (vs -warp_focus, which is local).",
    )
    knobs.add_argument(
        "-warp_iters",
        type=int,
        default=8,
        help="Deformation gradient steps per EM iteration (default 8). More = the warp "
        "makes more progress each iteration (another way to warp harder without more -niter).",
    )
    knobs.add_argument(
        "-blur_tpms",
        type=float,
        default=0.0,
        metavar="SIGMA",
        help="Coarse-to-fine warp: blur the tissue priors by SIGMA (TPM voxels) for a "
        "first pass, then refine with the sharp TPM. Widens the warp's capture range so "
        "it doesn't miss boundaries when the input still carries large residual "
        "distortion (an under-corrected EPI). Default 0 (off); try 2-4 to start.",
    )
    knobs.add_argument(
        "-blur_frac",
        type=float,
        default=0.4,
        help="Fraction of the iterations spent in the -blur_tpms coarse pass before "
        "switching to the sharp TPM (default 0.4). Ignored unless -blur_tpms is set.",
    )
    knobs.add_argument(
        "-warp_focus",
        type=float,
        default=0.0,
        metavar="S",
        help="Make the warp work harder on localised misfit (0..1). Each iteration the "
        "worst-fitting region — where the TPM prior disagrees most with the intensity, "
        "i.e. a stretched peninsula of leftover distortion — has its smoothing relaxed by "
        "up to S (1 = deform freely there, 0 = off/default), while the rest of the field "
        "stays stiff. Use when the warp under-corrects one spot but the whole is fine.",
    )
    knobs.add_argument(
        "-focus_quantile",
        type=float,
        default=0.9,
        help="Only nodes above this quantile of the misfit distribution get the "
        "-warp_focus relaxation (default 0.9 = the worst 10%%). Lower to widen the "
        "focused region. Ignored unless -warp_focus is set.",
    )
    knobs.add_argument(
        "-no_warp", action="store_true", help="Skip the deformation (bias + tissue only)"
    )
    knobs.add_argument(
        "-wp_reg",
        type=float,
        default=100.0,
        help="Tissue-weight regularisation toward uniform (SPM wp_reg, default 100). The "
        "mixing weights use SPM's self-correcting observed/expected update; a LOW value "
        "lets one tissue's weight run away over many iterations and inflate past its "
        "boundary ('growing brains'). Keep it high; lower only to let weights adapt more.",
    )
    knobs.add_argument(
        "-mrf",
        type=float,
        default=1.0,
        help="MRF cleanup strength on the tissue maps (default 1; 0 off)",
    )
    knobs.add_argument(
        "-cleanup",
        type=int,
        default=1,
        choices=(0, 1, 2),
        help="Morphological GM/WM/CSF brain extraction (0 off, 1 default, 2 strict)",
    )
    knobs.add_argument(
        "-debridge",
        type=int,
        default=0,
        metavar="R",
        help="Strip thin GM 'bridges' / peripheral sheets that are really mislabelled "
        "dura, via a morphological opening (radius R voxels) of the GM map: structures "
        "thinner than ~2R vanish, thick cortex stays, and the removed probability falls "
        "back to the other classes. 0 off (default); try 1 for ~1-2 voxel dura. Unlike "
        "-cleanup this also removes bridges that TOUCH cortex. Watch tight sulci at high R.",
    )
    knobs.add_argument(
        "-save_precleanup",
        action="store_true",
        help="Also write the tissue maps as they were BEFORE the -mrf/-debridge/-cleanup "
        "post-passes (as prefix_cN_precleanup), so you can see/undo what cleanup changed.",
    )
    knobs.add_argument(
        "-save_histogram",
        nargs="?",
        const="__auto__",
        default=None,
        metavar="PATH.png",
        help="Write an intensity-histogram diagnostic: the bias-corrected data with the "
        "fitted Gaussian-mixture overlaid, one panel per input channel, tissues coloured "
        "and labelled with their %% mass. Where the grey data rises above the black total "
        "model, the fit missed that intensity. Bare flag → prefix_histogram.png; or give a "
        "path. Needs matplotlib.",
    )
    knobs.add_argument(
        "-tissue_names",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Labels for the histogram legend, one per TPM class in order (default: SPM "
        "GM WM CSF bone 'soft tissue' air). Quote names with spaces.",
    )
    knobs.add_argument(
        "-prior_interp",
        default="cubic",
        choices=("linear", "cubic", "quintic", "heptic", "wsinc5"),
        help="Interpolation used to sample the low-res TPM for the full-res outputs "
        "(SPM uses a degree-2 B-spline ≈ cubic, the default; linear is faster/blockier)",
    )

    other = p.add_argument_group("execution")
    other.add_argument("-device", default=None, help="cuda | cpu (default: cuda if available)")
    other.add_argument(
        "-fit_dtype",
        default="float32",
        choices=("float32", "float64"),
        help="Precision of the fit hot path (interp, bias/warp autograd, likelihood). "
        "float32 (default) is much faster, especially on consumer GPUs where float64 "
        "runs at a fraction of the rate; the covariance factorisation and moment "
        "reduction stay float64 internally regardless. Use float64 for parity checks.",
    )
    other.add_argument(
        "-quiet",
        action="store_true",
        help="Suppress the fit report, per-iteration log-likelihood, and progress bar.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    verbose = not args.quiet
    if verbose:
        print(f"ffs_segment on {device}: {args.input}")

    with _step("Loading input + TPM", enabled=verbose):
        channels = []
        hdr = None
        for path in args.input:  # one or several co-registered channels
            ch, ch_hdr = load_image(path, device=device)
            if hdr is None:
                hdr = ch_hdr
            elif ch.shape != channels[0].shape or not np.allclose(
                ch_hdr["affine"], hdr["affine"], atol=1e-3
            ):
                raise SystemExit(
                    f"-input channels must be aligned (same grid + affine); {path} differs. "
                    "Reslice them onto a common grid first."
                )
            channels.append(ch)
        volume = channels[0] if len(channels) == 1 else channels
        subj_affine = torch.as_tensor(hdr["affine"], dtype=torch.float64, device=device)
        log_prior, tpm_affine, bg_low, bg_high = load_tpm(args.tpm, device=device)
    n_tissue = log_prior.shape[0]

    if args.ngaus is not None and len(args.ngaus) != n_tissue:
        raise SystemExit(f"-ngaus has {len(args.ngaus)} entries but the TPM has {n_tissue} classes")

    # Affine init: build the subject-voxel -> TPM-voxel map. A template->input
    # .aff12.1D chain gives template_vox -> input_vox (base=template); we want its
    # inverse. Without a matrix, assume the input already sits in template space.
    vox2vox: torch.Tensor | None = None
    world_affine: torch.Tensor | None = None
    if args.matrix:
        tpm_to_input = load_matrix_chain(
            args.matrix, base_affine=np.asarray(tpm_affine.cpu()), source_affine=hdr["affine"]
        ).to(dtype=torch.float64, device=device)
        vox2vox = torch.linalg.inv(tpm_to_input)  # input(subject) vox -> tpm vox
    else:
        world_affine = torch.eye(4, dtype=torch.float64, device=device)

    fit = fit_segment(
        volume,
        subj_affine,
        log_prior,
        tpm_affine,
        bg_low,
        bg_high,
        world_affine,
        vox2vox=vox2vox,
        ngaus=args.ngaus,
        biasreg=args.biasreg,
        biasfwhm=args.biasfwhm,
        reg=tuple(args.reg),
        samp=args.samp,
        n_iter=args.niter,
        tol=args.tol,
        fit_warp=not args.no_warp,
        pe_axis={"x": 0, "y": 1, "z": 2}[args.pe_axis] if args.pe_axis else None,
        blur_tpms=args.blur_tpms,
        blur_frac=args.blur_frac,
        warp_focus=args.warp_focus,
        focus_quantile=args.focus_quantile,
        wp_reg=args.wp_reg,
        warp_solver=args.warp_solver,
        warp_lr=args.warp_lr,
        warp_iters=args.warp_iters,
        warp_smooth=args.warp_smooth,
        fit_chunk=args.fit_chunk,
        dtype=torch.float32 if args.fit_dtype == "float32" else torch.float64,
        device=device,
        verbose=not args.quiet,
    )

    if verbose:
        print("Applying model at full resolution + cleanup:")
    out = segment_apply(
        volume,
        log_prior,
        bg_low,
        bg_high,
        fit,
        mrf=args.mrf,
        cleanup=args.cleanup,
        debridge=args.debridge,
        prior_kernel=args.prior_interp,
        save_precleanup=args.save_precleanup,
        device=device,
        verbose=verbose,
    )
    prefix = _strip_ext(args.prefix)
    n_chan = len(channels)
    ref = channels[0]  # reference channel for the shared-geometry outputs
    with _step("Writing tissue maps + bias", enabled=verbose):
        for t in range(n_tissue):
            save_image(
                out["posteriors"][t],
                f"{prefix}_c{t + 1}.nii.gz",
                header_info=hdr,
                affine=hdr["affine"],
            )
            if "posteriors_precleanup" in out:
                save_image(
                    out["posteriors_precleanup"][t],
                    f"{prefix}_c{t + 1}_precleanup.nii.gz",
                    header_info=hdr,
                    affine=hdr["affine"],
                )
        # bias-corrected + bias field, per channel (suffix _chN only when multi-channel)
        for c in range(n_chan):
            suffix = "" if n_chan == 1 else f"_ch{c + 1}"
            corr_c = out["corrected"] if n_chan == 1 else out["corrected"][c]
            bias_c = out["bias"] if n_chan == 1 else out["bias"][c]
            save_image(
                corr_c,
                f"{prefix}_biascorrected{suffix}.nii.gz",
                header_info=hdr,
                affine=hdr["affine"],
            )
            save_image(
                bias_c, f"{prefix}_biasfield{suffix}.nii.gz", header_info=hdr, affine=hdr["affine"]
            )

    if args.save_histogram is not None:
        hist_path = (
            f"{prefix}_histogram.png" if args.save_histogram == "__auto__" else args.save_histogram
        )
        if args.tissue_names is not None and len(args.tissue_names) != n_tissue:
            raise SystemExit(
                f"-tissue_names has {len(args.tissue_names)} entries but the TPM has "
                f"{n_tissue} classes"
            )
        with _step(f"Writing histogram diagnostic ({hist_path})", enabled=verbose):
            # prefer the pre-cleanup posteriors (the raw GMM fit) so morphological cleanup
            # doesn't bias the tissue masses; fall back to the final maps if not retained
            hist_post = out.get("posteriors_precleanup", out["posteriors"])
            plot_intensity_fit(
                fit,
                out["corrected"],
                hist_post,
                tissue_names=args.tissue_names,
                path=hist_path,
            )

    extra = []
    if "posteriors_precleanup" in out:
        extra.append(", _cN_precleanup")
    if args.save_histogram is not None:
        extra.append(", _histogram.png")
    norm_step = _step("Writing normalisation outputs (warp + template space)", enabled=verbose)
    norm_step.__enter__()
    # ── input resampled into TEMPLATE space (overlays the anat/template) ──
    # tpl_shape/tpl_aff = the -tpm grid; in EPI mode that IS the anat space.
    tpl_shape = tuple(log_prior.shape[1:])
    tpl_aff = np.asarray(tpm_affine.cpu())
    initial_in_tpl = input_in_template(
        ref, fit, tpl_shape, use_warp=False, kernel=args.prior_interp, device=device
    )
    save_image(initial_in_tpl, f"{prefix}_in_template_initial.nii.gz", affine=tpl_aff)
    extra.append(", _in_template_initial")

    if not args.no_warp:
        warp = full_resolution_warp(fit, tuple(ref.shape), device=device)  # (nz,ny,nx,3) vox disp
        save_warp_field(
            warp[..., 0],
            warp[..., 1],
            warp[..., 2],
            f"{prefix}_warp.nii.gz",
            header_info=hdr,
            affine=hdr["affine"],
            units="mm",
        )
        # geometry-corrected input in its OWN space (PE-mode: the undistorted EPI)
        undist = undistort_input(ref, fit, device=device)
        save_image(undist, f"{prefix}_undistorted.nii.gz", header_info=hdr, affine=hdr["affine"])
        # corrected input in TEMPLATE space (affine + warp) — aligns with the anat
        corrected_in_tpl = input_in_template(
            ref, fit, tpl_shape, kernel=args.prior_interp, device=device
        )
        save_image(corrected_in_tpl, f"{prefix}_in_template.nii.gz", affine=tpl_aff)
        extra += [", _warp", ", _undistorted", ", _in_template"]

    if args.tpm_source is not None:
        src, src_hdr = load_image(args.tpm_source, device=device)
        src_affine = torch.as_tensor(src_hdr["affine"], dtype=torch.float64, device=device)
        shape = tuple(ref.shape)
        # affine-only "initial" cast (before the nonlinear warp) — a before/after baseline
        initial = cast_template_to_input(
            src,
            src_affine,
            tpm_affine,
            fit,
            shape,
            use_warp=False,
            kernel=args.prior_interp,
            device=device,
        )
        save_image(
            initial,
            f"{prefix}_tpm_source_in_epi_initial.nii.gz",
            header_info=hdr,
            affine=hdr["affine"],
        )
        extra.append(", _tpm_source_in_epi_initial")
        if not args.no_warp:
            final = cast_template_to_input(
                src, src_affine, tpm_affine, fit, shape, kernel=args.prior_interp, device=device
            )
            save_image(
                final, f"{prefix}_tpm_source_in_epi.nii.gz", header_info=hdr, affine=hdr["affine"]
            )
            extra.append(", _tpm_source_in_epi")

    norm_step.__exit__(None, None, None)
    if verbose:
        print(f"wrote {prefix}_c1..c{n_tissue}, _biascorrected, _biasfield" + "".join(extra))
    return 0


if __name__ == "__main__":
    sys.exit(main())
