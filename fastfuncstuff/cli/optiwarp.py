"""Command-line interface for ffs_optiwarp — GPU optical-flow nonlinear registration.

The third nonlinear-registration backend, alongside ``ffs_qwarp`` (AFNI 3dQwarp patch
optimization) and ``ffs_formwarp`` (ANTs SyN). This one solves the optical-flow
brightness-constancy equation for a dense 3-D displacement field — the same idea
``ffs_locomoco`` applies to residual EPI motion, here as a general image-to-image warper.
The output warp is in the same on-disk format ``ffs_nwarp`` consumes.

It assumes the two images already agree affinely (run ``ffs_allineate`` first) and solves
for the residual deformation: anat -> MNI, ses-01 -> ses-12, EPI -> fieldmap.

Usage:
    # Same-modality subtle warp, save the moving->fixed warp
    ffs_optiwarp -base fixed.nii.gz -source moving.nii.gz -prefix warped.nii.gz -save_warp

    # Anat -> template (same contrast, different scanner/bias): localnorm is the default
    ffs_optiwarp -base MNI152_T1.nii.gz -source anat.nii.gz -prefix anat2mni.nii.gz \\
        -force lk -shrink 8x4x2x1 -smooth 3x2x1x0 -iters 150x100x70x40 -save_warp

    # Genuinely cross-modal (EPI -> T1): register edges, which are sign-free
    ffs_optiwarp -base T1.nii.gz -source epi.nii.gz -prefix epi2anat.nii.gz \\
        -match gradmag -automask -save_warp

    # Flow for the global fit, then hand off to the qwarp engine for fine detail
    ffs_optiwarp -base fixed.nii.gz -source moving.nii.gz -prefix out.nii.gz \\
        -final_qwarp -minpatch 13 -save_warp

    # Distortion-style: restrict the deformation to the Y (phase-encode) axis
    ffs_optiwarp -base b0.nii.gz -source epi.nii.gz -prefix corr.nii.gz -noXdis -noZdis
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import torch

from fastfuncstuff.cli_utils import (
    add_coverage_args,
    add_verbose_arg,
    combine_brain_masks,
    image_support,
    parse_prefix,
    sanitize_volume,
    spinner,
)
from fastfuncstuff.processing.interp import WARP_INTERP_MODES
from fastfuncstuff.processing.io import load_image, save_image, save_warp_field, save_warp_series
from fastfuncstuff.processing.mask import data_coverage_mask
from fastfuncstuff.processing.optiwarp import (
    FORCES,
    MATCH_MODES,
    NO_X_DISP,
    NO_Y_DISP,
    NO_Z_DISP,
    STEP_MODES,
    OptiwarpConfig,
    optiwarp,
)
from fastfuncstuff.processing.warp import QwarpConfig


def _int_list(spec: str) -> tuple[int, ...]:
    """Parse an ANTs-style ``4x2x1`` (or ``4,2,1``) spec into a tuple of ints."""
    return tuple(int(p) for p in spec.replace(",", "x").split("x") if p != "")


def _float_list(spec: str) -> tuple[float, ...]:
    """Parse an ANTs-style ``2x1x0`` (or ``2,1,0``) spec into a tuple of floats."""
    return tuple(float(p) for p in spec.replace(",", "x").split("x") if p != "")


class _HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Show defaults on each option, but leave the epilog's layout alone."""


_EPILOG = """\
THE FLOW MODELS (-force)
------------------------
All three start from the same brightness-constancy statement: a voxel keeps its
intensity as it moves, so for the currently warped moving image W and the fixed
image F, the displacement increment d must satisfy

    d . grad(W)  +  (W - F)  =  0

That is ONE equation per voxel in THREE unknowns -- the aperture problem. You can
see motion across an edge but not along it. The three models differ only in how
they supply the missing information.

  demons  Take the minimum-norm solution, which lies along the gradient:
              d = -(W-F) * grad(W) / ( |grad(W)|^2 + (W-F)^2 / K^2 )
          The first denominator term is the plain least-squares one and blows up
          where the image is flat. The second (K = -demons_noise) is the Thirion /
          Cachier regularizer: where the two images disagree strongly -- noise, or
          no true correspondence at all -- it dominates and shrinks the step
          instead of trusting a bad linearization. Bounds any single step near
          K/2 voxels. One pass over the volume, no solve: the cheapest model, and
          the default. Weakness: motion tangential to an edge is invisible to it,
          so it leans on the regularizer to fill in.

  lk      Lucas-Kanade. Assume d is CONSTANT over a (2*-lk_radius+1)^3 window and
          least-squares the resulting 3x3 system per voxel:
              [ sum g g^T ] d  =  -[ sum g * (W-F) ]
          Spatial support, not a prior, is what closes the aperture problem, so
          this is the model that recovers displacement ALONG an edge -- as long as
          the window contains some structure with a second orientation.
          -lk_reg ridges the 3x3 (relative to its mean trace) for the directions
          that are still degenerate. Costs a 3x3 solve per voxel; the sharpest
          choice when the anatomy has corners and junctions rather than long
          smooth boundaries.

  hs      Horn-Schunck. Close the system GLOBALLY: minimize the brightness error
          plus -hs_alpha times the gradient of the flow field, over the whole
          volume at once, by -hs_iters Jacobi sweeps. Each voxel relaxes toward
          its neighbours' average and is then corrected along the gradient.
          Information propagates from textured regions into flat ones, which
          neither of the others do. The smoothest and slowest-moving; raise
          -hs_alpha for more rigidity.

Rule of thumb: start with demons. Switch to lk if the warp looks like it is
sliding along boundaries instead of across them. Use hs when large homogeneous
regions need to be carried by their edges.

MATCHING ACROSS CONTRAST (-match)
---------------------------------
Brightness constancy is a statement about MODALITY, and -match is how far it is
relaxed. -match localnorm (the default) locally z-scores both images, removing
bias fields, shading and any spatially varying gain -- the right choice across
sessions or scanners at the same contrast. It does NOT survive a contrast
inversion: the local z-score of an inverted image is the negated map, so the
force points backwards everywhere. For genuinely cross-modal pairs (T1 vs T2,
EPI vs anat) use -match gradmag, which registers locally normalized gradient
magnitude: edges sit in the same place with the same sign in every modality.

KEEPING THE FIELD HONEST
------------------------
Nothing in the flow equation knows about topology, so by default each update is
exponentiated (scaling-and-squaring) and composed rather than added, making every
increment a diffeomorphism -- the field cannot fold. -save_jacobian writes the
proof; values <= 0 are folded voxels. -step_mode additive is the classic, faster,
unguarded update. -final_qwarp then hands the converged field to the ffs_qwarp
engine for a fine-scale, pure image-match polish: flow is good at FINDING the
deformation, patch optimization is good at nailing the last half-voxel.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ffs_optiwarp",
        description="GPU optical-flow (demons / Lucas-Kanade / Horn-Schunck) "
        "nonlinear registration.",
        epilog=_EPILOG,
        formatter_class=_HelpFormatter,
    )

    # I/O
    p.add_argument("-base", required=True, help="Fixed/target image (3D).")
    p.add_argument("-source", required=True, help="Moving image to deform (3D or 4D series).")
    p.add_argument("-prefix", required=True, help="Output path for the warped image.")
    p.add_argument(
        "-save_warp",
        "-save-warp",
        action="store_true",
        help="Save the moving->fixed warp ({prefix}_WARP.nii.gz).",
    )
    p.add_argument(
        "-save_inverse",
        "-save-inverse",
        action="store_true",
        help="Save the fixed->moving inverse warp ({prefix}_WARPINV.nii.gz).",
    )
    p.add_argument(
        "-save_jacobian",
        "-save-jacobian",
        action="store_true",
        help="Save the Jacobian determinant map ({prefix}_JAC.nii.gz). <=0 marks folding.",
    )
    p.add_argument(
        "-warp_format",
        "-warp-format",
        choices=["5d", "folder"],
        default="5d",
        help="Timeseries mode only (4D -source): warp on-disk format. '5d' = one "
        "{prefix}_WARP file (nx,ny,nz,T,3, default); 'folder' = {prefix}_WARP/ of "
        "numbered 4D frames. Both are consumed by ffs_nwarp -nwarp / ffs_util_pcwarp.",
    )

    # Flow model
    flow = p.add_argument_group("optical-flow model")
    flow.add_argument(
        "-force",
        choices=FORCES,
        default="demons",
        help="How to close the aperture problem (see the FLOW MODELS section below). "
        "demons: step along the gradient, damped where the images disagree - cheapest, "
        "blind to motion along an edge. lk: least-squares a 3x3 system over a "
        "neighbourhood - sees motion along edges, needs local structure. hs: global "
        "smoothness prior by Jacobi relaxation - propagates flow from textured regions "
        "into flat ones, smoothest and slowest.",
    )
    flow.add_argument(
        "-asymmetric_force",
        "-asymmetric-force",
        action="store_true",
        help="Use only the moving image's gradient in the flow equation instead of the "
        "symmetric (grad_moving + grad_fixed)/2 default.",
    )
    flow.add_argument(
        "-match",
        choices=MATCH_MODES,
        default="localnorm",
        help="Intensity prep before the flow solve: localnorm (local z-score; removes "
        "bias fields and gain, the cross-session default), gradmag (locally normalized "
        "gradient magnitude; the only mode that survives a contrast inversion, so use "
        "it for cross-modal T1/T2/EPI pairs), meanstd (global z-score), none (raw). "
        "Estimation only; the saved image is warped from the raw source.",
    )
    flow.add_argument(
        "-match_sigma",
        "-match-sigma",
        type=float,
        default=6.0,
        help="Neighborhood sigma (voxels) for -match localnorm.",
    )
    flow.add_argument(
        "-demons_noise",
        "-demons-noise",
        type=float,
        default=1.0,
        help="Demons normalization K: intensity difference (in prepped units) at which "
        "the force is damped. Smaller = more conservative.",
    )
    flow.add_argument(
        "-lk_radius",
        "-lk-radius",
        type=int,
        default=2,
        help="Lucas-Kanade neighborhood half-width in voxels (window (2r+1)^3).",
    )
    flow.add_argument(
        "-lk_reg",
        "-lk-reg",
        type=float,
        default=1e-2,
        help="Ridge on the LK structure tensor, relative to its mean trace.",
    )
    flow.add_argument(
        "-hs_alpha", "-hs-alpha", type=float, default=1.0, help="Horn-Schunck smoothness weight."
    )
    flow.add_argument(
        "-hs_iters",
        "-hs-iters",
        type=int,
        default=20,
        help="Horn-Schunck Jacobi iterations per flow solve.",
    )

    # Stepping and regularization
    reg = p.add_argument_group("stepping and regularization")
    reg.add_argument(
        "-step_mode",
        "-step-mode",
        choices=STEP_MODES,
        default="diffeo",
        help="diffeo: exponentiate each update (scaling-and-squaring) and compose, so "
        "the field cannot fold. additive: classic demons, faster and unguarded.",
    )
    reg.add_argument(
        "-max_step",
        "-max-step",
        type=float,
        default=1.0,
        help="Cap on the largest per-iteration displacement, in voxels.",
    )
    reg.add_argument(
        "-update_sigma",
        "-update-sigma",
        type=float,
        default=1.0,
        help="Fluid regularization: update-field Gaussian sigma (voxels; 0=off).",
    )
    reg.add_argument(
        "-total_sigma",
        "-total-sigma",
        type=float,
        default=1.0,
        help="Elastic/diffusion regularization: total-field Gaussian sigma (voxels; "
        "0=off, which lets the field get loose at small scales).",
    )
    reg.add_argument(
        "-invert_iters",
        "-invert-iters",
        type=int,
        default=8,
        help="Fixed-point iterations for displacement-field inversion.",
    )

    # Multiresolution
    mr = p.add_argument_group("multiresolution")
    mr.add_argument(
        "-shrink", type=str, default="4x2x1", help="Per-level isotropic shrink factors, e.g. 4x2x1."
    )
    mr.add_argument(
        "-smooth",
        type=str,
        default="2x1x0",
        help="Per-level Gaussian pre-smoothing sigmas in voxels, e.g. 2x1x0.",
    )
    mr.add_argument(
        "-iters",
        type=str,
        default="100x70x40",
        help="Per-level max iteration counts, e.g. 100x70x40 (an upper bound; a level "
        "usually stops earlier via convergence).",
    )
    mr.add_argument(
        "-conv_window",
        "-conv-window",
        type=int,
        default=10,
        help="Trailing-window size for convergence/early-stopping. <=0 disables it "
        "(run the full -iters). The best-metric warp is always returned, so running "
        "to exhaustion never over-warps.",
    )
    mr.add_argument(
        "-conv_thresh",
        "-conv-thresh",
        type=float,
        default=1e-6,
        help="Convergence slope threshold; larger stops sooner.",
    )

    # Monitoring metric (referee, not the driver)
    met = p.add_argument_group("monitoring metric")
    met.add_argument(
        "-metric",
        choices=("cc", "lpa", "lpc", "pearson", "mse"),
        default="cc",
        help="Metric used to pick the best iterate and detect convergence. The flow "
        "equation drives the update; this decides which iterate to keep.",
    )
    met.add_argument(
        "-cc_radius",
        "-cc-radius",
        type=int,
        default=4,
        help="CC neighborhood half-width in voxels (window (2r+1)^3).",
    )
    met.add_argument(
        "-lpa_sigma",
        "-lpa-sigma",
        type=float,
        default=4.0,
        help="Neighborhood size (voxels) for the lpa/lpc metrics.",
    )
    met.add_argument(
        "-lpa_kernel",
        "-lpa-kernel",
        choices=("gauss", "box"),
        default="gauss",
        help="Neighborhood kernel for lpa/lpc.",
    )

    # Axis constraints (match qwarp/formwarp)
    ax = p.add_argument_group("axis constraints")
    ax.add_argument("-noXdis", "-noxdis", action="store_true", help="No x-displacement.")
    ax.add_argument("-noYdis", "-noydis", action="store_true", help="No y-displacement.")
    ax.add_argument("-noZdis", "-nozdis", action="store_true", help="No z-displacement.")

    # qwarp hand-off
    qw = p.add_argument_group("qwarp hand-off")
    qw.add_argument(
        "-final_qwarp",
        "-final-qwarp",
        action="store_true",
        help="After the flow levels converge, refine with the 3dQwarp engine "
        "initialized from the flow field (fine-scale pure image match).",
    )
    qw.add_argument(
        "-minpatch",
        type=int,
        default=25,
        help="-final_qwarp: smallest qwarp patch size in voxels (3dQwarp -minpatch).",
    )
    qw.add_argument(
        "-qwarp_cost",
        "-qwarp-cost",
        type=str,
        default="pearclp",
        help="-final_qwarp: qwarp cost function (e.g. pearclp, pcl, lpa, lpc).",
    )
    qw.add_argument(
        "-qwarp_penfac",
        "-qwarp-penfac",
        type=float,
        default=0.033,
        help="-final_qwarp: qwarp deformation penalty factor (3dQwarp -penfac).",
    )
    qw.add_argument(
        "-qwarp_keep_worse",
        "-qwarp-keep-worse",
        action="store_true",
        help="-final_qwarp: run every qwarp level even if one makes the global cost "
        "worse (AFNI's behaviour). By default the polish stops at the last level that "
        "improved, so the hand-off can only help.",
    )

    # Weight / mask
    p.add_argument("-weight", help="Weight image emphasizing the metric (overrides automask).")
    add_coverage_args(
        p,
        automask_help="Restrict the metric and the flow force to the automask of BOTH "
        "images (unioned; see -automask_intersect). Independent of the data-coverage "
        "restriction below.",
    )

    # Output interpolation
    p.add_argument(
        "-final_interp",
        "-final-interp",
        choices=WARP_INTERP_MODES,
        default="wsinc5",
        help="Interpolation for the final warped image.",
    )

    p.add_argument("-device", default=None, help="torch device (cuda/cpu/mps). Auto if unset.")
    add_verbose_arg(p)
    return p.parse_args(argv)


def _build_config(args: argparse.Namespace) -> OptiwarpConfig:
    warp_flags = 0
    if args.noXdis:
        warp_flags |= NO_X_DISP
    if args.noYdis:
        warp_flags |= NO_Y_DISP
    if args.noZdis:
        warp_flags |= NO_Z_DISP

    qcfg = None
    if args.final_qwarp:
        qcfg = QwarpConfig(
            minpatch=args.minpatch,
            cost_method=args.qwarp_cost,
            penalty_factor=args.qwarp_penfac,
            warp_flags=warp_flags,
            final_interp=args.final_interp,
            reject_worse_levels=not args.qwarp_keep_worse,
            verb=args.verb,
        )

    return OptiwarpConfig(
        force=args.force,
        symmetric_force=not args.asymmetric_force,
        match=args.match,
        match_sigma=args.match_sigma,
        demons_noise=args.demons_noise,
        lk_radius=args.lk_radius,
        lk_reg=args.lk_reg,
        hs_alpha=args.hs_alpha,
        hs_iters=args.hs_iters,
        step_mode=args.step_mode,
        max_step=args.max_step,
        update_sigma=args.update_sigma,
        total_sigma=args.total_sigma,
        shrink_factors=_int_list(args.shrink),
        smoothing_sigmas=_float_list(args.smooth),
        iterations=_int_list(args.iters),
        metric=args.metric,
        cc_radius=args.cc_radius,
        lpa_sigma=args.lpa_sigma,
        lpa_kernel=args.lpa_kernel,
        convergence_window=args.conv_window,
        convergence_threshold=args.conv_thresh,
        invert_iters=args.invert_iters,
        void_guard=args.void_guard,
        warp_flags=warp_flags,
        final_qwarp=args.final_qwarp,
        qwarp_config=qcfg,
        final_interp=args.final_interp,
        verb=args.verb,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Select device (prefer CUDA > MPS > CPU), honouring -device end to end.
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        if args.verb >= 1:
            print("WARNING: no GPU available, running on CPU")

    if args.verb >= 1:
        print(f"ffs_optiwarp: device={device}")

    t0 = time.time()
    with spinner(f"Loading {Path(args.base).name}"):
        base, base_info = load_image(args.base, device=torch.device("cpu"))
    with spinner(f"Loading {Path(args.source).name}"):
        source, _ = load_image(args.source, device=torch.device("cpu"))

    base = sanitize_volume(base, "-base", args.verb)
    source = sanitize_volume(source, "-source", args.verb)

    if base.ndim == 4:
        if args.verb >= 1:
            print("WARNING: 4D -base; using vol[0] as the fixed target")
        base = base[0]

    timeseries = source.ndim == 4
    src_shape = tuple(source.shape[1:]) if timeseries else tuple(source.shape)
    if src_shape != tuple(base.shape):
        print(
            f"ERROR: base {tuple(base.shape)} and source {src_shape} must be on the "
            "same grid (resample first, e.g. ffs_allineate)."
        )
        return 1

    nz, ny, nx = base.shape
    if args.verb >= 1:
        print(f"Base/source: {nx}x{ny}x{nz}, loaded in {time.time() - t0:.1f}s")

    weight = None
    if args.weight is not None:
        with spinner(f"Loading {Path(args.weight).name}"):
            weight, _ = load_image(args.weight, device=torch.device("cpu"))
        weight = weight.float().to(device)

    base_brain, base_cover = image_support(
        base, args, device, args.automask or args.automask_base, "base", args.verb
    )
    # Source automask on the temporal mean, not per frame: the brain does not move
    # nearly enough between frames to matter and automask is the expensive part.
    # Data coverage IS per frame -- see _run_timeseries.
    src_ref = source.mean(dim=0) if source.ndim == 4 else source
    src_brain, src_cover = image_support(
        src_ref, args, device, args.automask or args.automask_source, "source", args.verb
    )
    brain = combine_brain_masks(base_brain, src_brain, args.automask_intersect)
    mask = None
    if brain is not None:
        if weight is not None:
            weight = weight * brain.float()
        else:
            mask = brain.float()

    config = _build_config(args)
    pfx = parse_prefix(args.prefix)
    prefix, nii_ext = pfx.stem, pfx.nifti_ext

    if timeseries:
        return _run_timeseries(
            args,
            base,
            source,
            weight,
            mask,
            base_cover,
            config,
            base_info,
            prefix,
            nii_ext,
            device,
            t0,
        )

    res = optiwarp(
        base,
        source,
        weight=weight,
        mask=mask,
        fixed_cover=base_cover,
        moving_cover=src_cover,
        config=config,
        device=device,
    )

    warped_path = pfx.as_file()
    with spinner(f"Writing {Path(warped_path).name}"):
        save_image(res.warped, warped_path, header_info=base_info)
    if args.verb >= 1:
        print(f"Saved warped image: {warped_path}")

    def _save_warp(triple, path: str, label: str) -> None:
        with spinner(f"Writing {Path(path).name}"):
            save_warp_field(
                triple[0].cpu(),
                triple[1].cpu(),
                triple[2].cpu(),
                path,
                header_info=base_info,
                units="mm",
            )
        if args.verb >= 1:
            print(f"Saved {label}: {path}")

    if args.save_warp:
        _save_warp(res.fwd, f"{prefix}_WARP{nii_ext}", "moving->fixed warp")
    if args.save_inverse:
        _save_warp(res.inv, f"{prefix}_WARPINV{nii_ext}", "fixed->moving inverse warp")
    if args.save_jacobian:
        jac_path = f"{prefix}_JAC{nii_ext}"
        with spinner(f"Writing {Path(jac_path).name}"):
            save_image(res.jacobian.cpu(), jac_path, header_info=base_info)
        if args.verb >= 1:
            print(f"Saved Jacobian map: {jac_path}")

    if args.verb >= 1:
        print(f"Done in {time.time() - t0:.1f}s")
    return 0


def _run_timeseries(
    args, base, source, weight, mask, base_cover, config, base_info, prefix, nii_ext, device, t0
) -> int:
    """Register every volume of a 4D source to the 3D base (per-volume optical flow).

    Writes the 4D warped series and, with -save_warp/-save_inverse, a per-volume warp
    series in the chosen -warp_format (one 5D file or a folder of 4D frames).
    """
    from tqdm import tqdm

    n_t = source.shape[0]
    if args.verb >= 1:
        print(f"Timeseries mode: {n_t} volumes -> base (per-volume optical flow)")

    per_vol_cfg = replace(config, verb=0) if config.verb else config
    warped_frames: list[torch.Tensor] = []
    fwd_frames: list[tuple] = []
    inv_frames: list[tuple] = []
    min_jac = float("inf")
    for t in tqdm(range(n_t), desc="optiwarp", disable=args.verb < 1, leave=True):
        cover_t = (
            None
            if args.nocoverage
            else data_coverage_mask(
                source[t].float().to(device), erode=args.coverage_erode, device=device
            )
        )
        res = optiwarp(
            base,
            source[t],
            weight=weight,
            mask=mask,
            fixed_cover=base_cover,
            moving_cover=cover_t,
            config=per_vol_cfg,
            device=device,
        )
        warped_frames.append(res.warped.cpu())
        min_jac = min(min_jac, res.min_jacobian)
        if args.save_warp:
            fwd_frames.append(tuple(c.cpu() for c in res.fwd))
        if args.save_inverse:
            inv_frames.append(tuple(c.cpu() for c in res.inv))

    if args.verb >= 1:
        print(f"Worst per-volume min Jacobian: {min_jac:.4f}")

    warped_path = f"{prefix}{nii_ext}"
    with spinner(f"Writing {Path(warped_path).name}"):
        save_image(torch.stack(warped_frames), warped_path, header_info=base_info)
    if args.verb >= 1:
        print(f"Saved warped series: {warped_path} ({n_t} volumes)")

    def _save_series(frames, tag: str, label: str) -> None:
        as_5d = args.warp_format == "5d"
        xs = torch.stack([f[0] for f in frames])
        ys = torch.stack([f[1] for f in frames])
        zs = torch.stack([f[2] for f in frames])
        dest = f"{prefix}_{tag}{nii_ext}" if as_5d else f"{prefix}_{tag}"
        with spinner(f"Writing {Path(dest).name}"):
            out = save_warp_series(xs, ys, zs, dest, as_5d=as_5d, header_info=base_info, units="mm")
        if args.verb >= 1:
            fmt = "5D" if as_5d else "folder"
            print(f"Saved {label} ({fmt}): {out}")

    if args.save_warp:
        _save_series(fwd_frames, "WARP", "moving->fixed warp series")
    if args.save_inverse:
        _save_series(inv_frames, "WARPINV", "fixed->moving inverse warp series")

    if args.verb >= 1:
        print(f"Done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
