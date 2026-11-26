"""Command-line interface for ffs_formwarp — GPU SyN nonlinear registration.

A second nonlinear-registration backend alongside ``ffs_qwarp`` (AFNI 3dQwarp). This one
implements ANTs-style symmetric normalization (SyN): dense displacement fields with
fluid + elastic Gaussian regularization and a symmetric midpoint formulation. The output
warp is in the same on-disk format ``ffs_nwarp`` consumes.

Usage:
    # Same-modality SyN, save the moving->fixed warp
    ffs_formwarp -base fixed.nii.gz -source moving.nii.gz -prefix warped.nii.gz -save_warp

    # ANTs-style multiresolution, EPI->anat (cross contrast), with inverse + halfway warps
    ffs_formwarp -base anat.nii.gz -source epi.nii.gz -prefix out.nii.gz \\
        -metric lpc -shrink 8x4x2x1 -smooth 3x2x1x0 -iters 100x70x50x20 \\
        -save_warp -save_inverse -save_halfway

    # Distortion-style: restrict deformation to the Y (phase-encode) axis
    ffs_formwarp -base b0.nii.gz -source epi.nii.gz -prefix corr.nii.gz -noXdis -noZdis
"""

from __future__ import annotations

import argparse
import shlex
import time
from dataclasses import replace
from pathlib import Path

import torch

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter
from fastfuncstuff.cli_utils import (
    add_batch_args,
    add_coverage_args,
    add_deterministic_arg,
    add_device_arg,
    add_preallineate_args,
    add_recipe_arg,
    add_verbose_arg,
    add_warp_movie_args,
    apply_recipe_preset,
    build_warp_movie_recorder,
    collect_batch_jobs,
    combine_brain_masks,
    enable_determinism,
    frame_coverage,
    image_support,
    parse_prefix,
    preallineate_outputs,
    render_warp_movie,
    resolve_movie_path,
    run_batch_jobs,
    run_with_preallineate,
    sanitize_volume,
    setup_device,
    spinner,
)
from fastfuncstuff.processing.affine import base_into_source_frame
from fastfuncstuff.processing.formwarp import (
    LEGACY_SYN,
    METRICS,
    NO_X_DISP,
    NO_Y_DISP,
    NO_Z_DISP,
    SynConfig,
    formwarp,
)
from fastfuncstuff.processing.interp import WARP_INTERP_MODES
from fastfuncstuff.processing.io import load_image, save_image, save_warp_field, save_warp_series
from fastfuncstuff.utils import REGISTRATION_TF32


def _fmt_schedule(values) -> str:
    """A schedule tuple as the ANTs-style ``4x2x1`` string argparse shows as default."""
    return "x".join(f"{v:g}" for v in values)


# Argparse defaults are READ FROM the engine's config dataclass, never re-typed
# here. A default that lives in two places drifts, and it did: the iteration
# ceiling was raised in the engine after measuring that the old one starved the
# finest level, and this CLI kept passing the old value, so every command-line
# run silently kept the schedule the change existed to fix. The dataclass is the
# single source of truth; this is a view of it.
# Which PRESETS family `-type` should look up for this tool. The optiwarp
# force models share a parameter set, so one entry covers all three.
_PRESET_BACKEND = "formwarp"

_D = SynConfig()


def _int_list(spec: str) -> tuple[int, ...]:
    """Parse an ANTs-style ``4x2x1`` (or ``4,2,1``) spec into a tuple of ints."""
    return tuple(int(p) for p in spec.replace(",", "x").split("x") if p != "")


def _float_list(spec: str) -> tuple[float, ...]:
    """Parse an ANTs-style ``2x1x0`` (or ``2,1,0``) spec into a tuple of floats."""
    return tuple(float(p) for p in spec.replace(",", "x").split("x") if p != "")


def _maybe_list(spec: str | float) -> float | tuple[float, ...]:
    """A single number stays one, applying to every level; ``4x3x2`` becomes per-level.

    The default comes through as a float (argparse does not run ``type=`` on defaults),
    so both forms have to be accepted here rather than at the parser.
    """
    if not isinstance(spec, str):
        return spec
    values = _float_list(spec)
    return values[0] if len(values) == 1 else values


def _style_knobs(args) -> dict:
    """The style's values for the style-dependent knobs, overridden by explicit flags."""
    knobs = dict(LEGACY_SYN) if args.style == "legacy" else {}
    given = {
        "grad_step": args.grad_step,
        "update_var": None if args.update_var is None else _maybe_list(args.update_var),
        "total_var": None if args.total_var is None else _maybe_list(args.total_var),
        "shrink_factors": None if args.shrink is None else _int_list(args.shrink),
        "smoothing_sigmas": None if args.smooth is None else _float_list(args.smooth),
        "iterations": None if args.iters is None else _int_list(args.iters),
        "convergence_window": args.conv_window,
        "convergence_threshold": args.conv_thresh,
        "metric_gradient": args.metric_gradient,
        "step_norm": args.step_norm,
        "update_rule": args.update_rule,
        "inverter": args.inverter,
        "pin_boundary": None if args.pin_boundary is None else args.pin_boundary == "yes",
        "convergence_rule": args.conv_rule,
        "pyramid": args.pyramid,
        "field_kernel": args.field_kernel,
        "restore_best": None if args.keep is None else args.keep == "best",
        "fold_penalty": args.fold_penalty,
    }
    knobs.update({k: v for k, v in given.items() if v is not None})
    return knobs


_EPILOG = """\
WHICH NONLINEAR BACKEND
-----------------------
  ffs_formwarp (this)  ANTs SyN: one dense symmetric field. Two half-warps meet at a
                       midpoint, so the inverse and the halfway warps are free and
                       the fit is inverse-consistent. Best for large, smooth
                       deformations -- subject to template, session to session.
  ffs_qwarp            AFNI 3dQwarp overlapping polynomial patches. Sharper at fine
                       local detail once the pair is already close.
  ffs_optiwarp         Optical flow: solves for the field instead of optimizing a
                       cost. The fastest way to FIND a deformation.
All three assume the pair is already affinely aligned -- run ffs_allineate first
(or pass its matrix with -matrix, which registers in the source's own frame).

HOW A LEVEL RUNS (read before tuning anything)
----------------------------------------------
The default (-style ants) is antsRegistration's SyN, in ANTs' units: the flags map
one to one, so `-t SyN[0.1,3,0] -m CC[f,m,1,4] -c [100x70x50x20,1e-6,10] -f 8x4x2x1
-s 3x2x1x0vox` is `-grad_step 0.1 -update_var 3 -total_var 0 -cc_radius 4 -iters
100x70x50x20 -conv_thresh 1e-6 -conv_window 10 -shrink 8x4x2x1 -smooth 3x2x1x0` --
which is also the default. -update_var/-total_var are VARIANCES in voxels^2, as in
ANTs (they were sigmas before 2026-11-26).

-shrink/-smooth/-iters are ANTs' -f/-s/-c, one entry per level, coarse to fine.
Each iteration: evaluate the metric at the midpoint, smooth each half's update field
by -update_var (fluid), scale it so its largest displacement is -grad_step voxels,
compose it onto the field, smooth by -total_var (elastic), and re-derive each field
and its inverse. A level stops when ITK's windowed convergence measure drops below
-conv_thresh, and returns its last iterate, as ANTs does (-keep best restores the
lowest-cost one instead).

NOT ENOUGH WARP (structures still visibly misaligned)
-----------------------------------------------------
  * Ran out of iterations, or stopped too eagerly: raise -iters (100x70x50x20 ->
    200x150x100x40) and/or lower -conv_thresh (1e-6 -> 1e-8); -conv_window 0
    disables early stopping entirely and runs the full budget.
  * Fine detail never appears: -update_var is the fluid prior and the main thing
    smoothing detail away. Drop it (3 -> 1.5, a variance) and make sure the finest
    level has -smooth 0 and -shrink 1.
  * Everything moves a little, nothing moves enough: -grad_step is the per-iteration
    cap in voxels, so total travel is roughly grad_step x iterations. Raise it
    (0.1 -> 0.25) or add iterations; raising it too far makes the level oscillate.
  * Large deformation never bridged: add a coarser level (-shrink 16x8x4x2x1 -smooth
    4x3x2x1x0 -iters 100x100x70x50x20). Coarse levels are cheap and are where big,
    smooth displacement is actually won.
  * Cost dominated by background or by a clipped FoV edge: -automask, or supply
    -weight. The tissue/nothing cliff at a cut FoV is the strongest gradient in the
    volume; the coverage handling (-void_guard, -coverage_erode) exists for it.

TOO MUCH WARP (anatomy distorted, ripples, folding)
---------------------------------------------------
  * Raise -update_var (more fluid smoothing of each update) -- the first knob, and
    the only fold control ANTs has.
  * Turn on elastic regularization: -total_var 0.5-1 (variance) smooths the
    ACCUMULATED field, off by default (ANTs' antsRegistrationSyN.sh -t s uses 0).
  * -fold_penalty 0.03 charges folding (off by default, as in ANTs), or -fold_guard
    0.5 forbids it outright (the guard can refuse a step, so it costs fit).
  * Lower -grad_step (0.1 -> 0.05) so each iteration commits less.
  * Cut the finest level (-shrink 8x4x2 -smooth 3x2x1 -iters 100x70x50): most
    implausible deformation is bought at full resolution.
  * Blur more at the coarse levels (-smooth 3x2x1) so the fit is not chasing noise.
  * Distortion-only: -noXdis / -noYdis / -noZdis zero a component of every field
    on every iteration, which is a hard constraint, not a penalty.

WRONG-LOOKING WARP / CROSS-MODAL PAIRS
--------------------------------------
-metric cc (neighborhood cross-correlation, radius -cc_radius) is the SyN default
and assumes the two images covary locally. For contrast-inverted pairs (EPI to
anat) use -metric lpc; -metric lpa is the unsigned version for same-contrast pairs
with varying local gain (tune -lpa_sigma / -lpa_kernel). -metric mse only for
identical-contrast sanity checks. A big -cc_radius smooths the metric and behaves
like extra regularization; a small one is sharper and noisier.

TOO SLOW
--------
  * The finest level dominates -- shorten its -iters before touching anything else.
  * Coarser -shrink at the top (16x8x4x2x1) costs almost nothing and often removes
    iterations from the expensive levels.
  * -metric cc with a large -cc_radius is the expensive metric; radius 2-3 is much
    cheaper than 4.
  * Many pairs: -batch FILE runs them in one process, so CUDA context and warmup
    are paid once; -batch_skip skips jobs whose outputs already exist.

DIAGNOSTICS
-----------
-save_inverse and -save_halfway write the other three fields for free (they already
exist inside the algorithm). The default verbosity already prints, per level, the
iterations actually run and the best cost -- a level that used far fewer iterations
than its budget converged (or stalled); one that used all of them was still moving
when it ran out. If the warp looks fine but lands in the wrong space, the issue is
usually grids/headers, not tuning -- see the -matrix help text.
"""


def parse_args(
    argv: list[str] | None = None, namespace: argparse.Namespace | None = None
) -> argparse.Namespace:
    p = FfsArgumentParser(
        prog="ffs_formwarp",
        description="GPU SyN (symmetric normalization) nonlinear registration.",
        epilog=_EPILOG,
        formatter_class=FfsHelpFormatter,
    )

    # I/O
    p.add_argument("-base", default=None, help="Fixed/target image (3D) [required unless -batch].")
    p.add_argument(
        "-source", default=None, help="Moving image to deform (3D) [required unless -batch]."
    )
    p.add_argument(
        "-prefix", default=None, help="Output path for the warped image [required unless -batch]."
    )
    add_batch_args(
        p,
        tool="ffs_formwarp",
        what="SyN registrations",
        example="-base fixed.nii -source moving.nii -prefix out.nii -save_warp",
        skip_note="-prefix / -save_warp / -save_inverse / -save_halfway",
    )
    p.add_argument(
        "-save_warp",
        action="store_true",
        help="Save the moving->fixed warp ({prefix}_WARP.nii.gz).",
    )
    p.add_argument(
        "-warp_prefix",
        "-warp-prefix",
        default=None,
        help="Name the warp outputs off this stem instead of -prefix. For when the "
        "warped image and the warp want different names: the image belongs to the "
        "one input that produced it, the warp is shared by everything it is applied "
        "to. Does not change what is computed.",
    )
    p.add_argument(
        "-save_inverse",
        action="store_true",
        help="Save the fixed->moving inverse warp ({prefix}_WARPINV.nii.gz).",
    )
    p.add_argument(
        "-save_halfway",
        action="store_true",
        help="Save the four SyN half-warps (mid<->fixed, mid<->moving). Single-pair only.",
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

    # Metric
    p.add_argument(
        "-metric",
        choices=METRICS,
        default="cc",
        help="Image metric. cc=neighborhood cross-correlation (the SyN default); "
        "lncc is the same thing under its registry name; mse=squared difference "
        "(same-modality only); ngf=normalized gradient fields and mind/mindssc="
        "modality-independent neighbourhood descriptors, all three of which are "
        "invariant to a contrast inversion and so are the cross-modal choices.",
    )
    p.add_argument(
        "-cc_radius",
        "-cc-radius",
        type=int,
        default=_D.cc_radius,
        help="CC neighborhood half-width in voxels (window (2r+1)^3).",
    )
    p.add_argument(
        "-lpa_sigma",
        "-lpa-sigma",
        type=float,
        default=_D.lpa_sigma,
        help="Neighborhood size (voxels) for the lpa/lpc metrics.",
    )
    p.add_argument(
        "-lpa_kernel",
        "-lpa-kernel",
        choices=("gauss", "box"),
        default="gauss",
        help="Neighborhood kernel for lpa/lpc.",
    )

    # Engine style: which defaults the knobs below start from
    p.add_argument(
        "-style",
        choices=("ants", "legacy"),
        default="ants",
        help="'ants' (default): antsRegistration's SyN, step for step -- the defaults "
        "below are antsRegistrationSyN.sh's SyN stage (SyN[0.1,3,0], CC radius 4, "
        "100x70x50x20, 8x4x2x1, 3x2x1x0vox, [1e-6,10]) in ANTs' units, and the engine "
        "matches antsRegistration's warps (Dice 0.6237 vs 0.6240 on 48 NIREP pairs). "
        "'legacy': formwarp before 2026-11-26 (exact metric gradient, joint step, "
        "fixed-point inverter, 4x2x1 / 300x210x120, grad_step 0.25, update_var 9, "
        "fold_penalty 0.0333, best-iterate restore). Any knob given explicitly overrides "
        "the style's value.",
    )

    # SyN regularization (ANTs units: SyN[gradientStep, updateVar, totalVar])
    p.add_argument(
        "-grad_step",
        "-grad-step",
        type=float,
        default=None,
        help="Max per-iteration displacement in voxels of each level (ANTs SyN "
        f"gradientStep; default {_D.grad_step:g}).",
    )
    p.add_argument(
        "-update_var",
        "-update-var",
        default=None,
        help="Fluid regularization: update-field Gaussian VARIANCE in voxels^2 of each "
        f"level's grid, exactly ANTs' SyN[,updateFieldVariance] (default {_D.update_var:g}, "
        "i.e. sigma 1.73). Before 2026-11-26 this flag was a sigma: square old values. "
        "One value for all levels, or one per level, e.g. 16x9x4x4.",
    )
    p.add_argument(
        "-total_var",
        "-total-var",
        default=None,
        help="Elastic regularization: total-field Gaussian VARIANCE (voxels^2; 0=off; "
        "ANTs' SyN[,,totalFieldVariance]). One value for all levels, or one per level. "
        "Applied to the ACCUMULATED field every iteration, so a nonzero value at every "
        "level decays the warp as it is built.",
    )
    p.add_argument(
        "-invert_iters",
        "-invert-iters",
        type=int,
        default=_D.invert_iters,
        help="Fixed-point sweeps for -inverter fixed_point.",
    )

    # Multiresolution (ANTs -f / -s / -c)
    p.add_argument(
        "-shrink",
        type=str,
        default=None,
        help=f"Per-level isotropic shrink factors (ANTs -f; default {_fmt_schedule(_D.shrink_factors)}).",
    )
    p.add_argument(
        "-smooth",
        type=str,
        default=None,
        help="Per-level Gaussian pre-smoothing sigmas in voxels (ANTs -s ...vox; "
        f"default {_fmt_schedule(_D.smoothing_sigmas)}).",
    )
    p.add_argument(
        "-iters",
        type=str,
        default=None,
        help="Per-level max iteration counts (ANTs -c; default "
        f"{_fmt_schedule(_D.iterations)}); a level usually stops earlier via convergence.",
    )
    p.add_argument(
        "-conv_window",
        "-conv-window",
        type=int,
        default=None,
        help="Trailing-window size for convergence (ANTs -c [,,window]; default "
        f"{_D.convergence_window}). <=0 disables early stopping.",
    )
    p.add_argument(
        "-conv_thresh",
        "-conv-thresh",
        type=float,
        default=None,
        help="Convergence threshold (ANTs -c [,thresh,]; default "
        f"{_D.convergence_threshold:g}); larger stops sooner.",
    )
    p.add_argument(
        "-metric_gradient",
        choices=("auto", "exact", "local"),
        default=None,
        help="Update field: 'exact' (autograd gradient of the metric), 'local' (ANTs' "
        "centre-window CC approximation times the warped image's gradient; cc only), or "
        "'auto' (local for cc, exact otherwise; the ants default).",
    )
    p.add_argument(
        "-step_norm",
        choices=("joint", "per_half"),
        default=None,
        help="Scale the step so each half's largest displacement is -grad_step "
        "('per_half', ANTs) or the two halves' combined largest is ('joint', legacy).",
    )
    p.add_argument(
        "-update_rule",
        choices=("additive", "compose"),
        default=None,
        help="'compose' (ITK SyN: compose the update onto each half's inverse and "
        "re-derive; needs -inverter itk) or 'additive' (phi += step, legacy).",
    )
    p.add_argument(
        "-inverter",
        choices=("fixed_point", "itk"),
        default=None,
        help="Field inversion: 'itk' (warm-started, damped, to 0.1 voxel max error, "
        "round trip always kept) or 'fixed_point' (-invert_iters undamped sweeps, legacy).",
    )
    p.add_argument(
        "-pin_boundary",
        choices=("yes", "no"),
        default=None,
        help="Zero the smoothed update on the volume faces (ITK SyN does; ants default yes).",
    )
    p.add_argument(
        "-conv_rule",
        choices=("window", "itk"),
        default=None,
        help="Early-stop measure: 'itk' (ANTs: trailing slope over the level's "
        "cumulative |cost|) or 'window' (range-normalised trailing slope, legacy).",
    )
    p.add_argument(
        "-pyramid",
        choices=("corners", "itk"),
        default=None,
        help="Level grids: 'itk' (floor(n/f) voxels at spacing f, centred, sampling the "
        "full-resolution smoothed images, as ANTs) or 'corners' (legacy).",
    )
    p.add_argument(
        "-field_kernel",
        choices=("sampled", "discrete"),
        default=None,
        help="Field smoothing kernel: ITK's 'discrete' Gaussian (ANTs) or a 'sampled' "
        "Gaussian (legacy).",
    )
    p.add_argument(
        "-keep",
        choices=("last", "best"),
        default=None,
        help="Return each level's 'last' iterate (ANTs) or its lowest-cost legal one "
        "('best', legacy).",
    )
    p.add_argument(
        "-best_patience",
        type=int,
        default=_D.best_patience,
        help="With -keep best: stop a level once the kept (best fold-free) cost has not "
        "improved for this many iterations; 0 disables (default). Not result-preserving "
        "when legality flickers around -jac_floor.",
    )

    # Folding: discouraged by default, forbidden on request
    p.add_argument(
        "-fold_penalty",
        "-fold-penalty",
        type=float,
        default=None,
        help="Strength of the soft anti-folding penalty (AFNI's warp-distortion "
        "energy, same primitive as qwarp's -penfac); 0 disables it (ants default; "
        "legacy 0.0333). Its deadband charges nothing until a voxel inverts or more "
        "than doubles in volume, so ordinary compression is free.",
    )
    p.add_argument(
        "-guard_floor",
        "-guard-floor",
        type=float,
        default=_D.guard_floor,
        help="det(J) the hard guard holds the warp to (needs -fold_guard). At the "
        "default it only stops actual folding. Raised, it becomes a COMPRESSION LIMIT: "
        "-fold_guard 0.5 -guard_floor 0.3 forbids compressing any voxel past 3.3x, "
        "which on same-modality brain-to-brain scored LNCC 0.615 against 0.549 for the "
        "penalty alone -- near-conformality is a good prior there. It is the wrong "
        "prior wherever anatomy genuinely compresses hard (a large ventricle matched to "
        "a small one), so it is off by default.",
    )
    p.add_argument(
        "-guard_refuses",
        "-guard-refuses",
        action="store_true",
        help="Let -fold_guard drop a step entirely when local damping cannot make it "
        "safe, instead of keeping the best-damped one. Stricter, and measurably worse "
        "where it is not needed (LNCC 0.520 against 0.587 on a NIREP pair, with damping "
        "alone already leaving zero folded voxels): refusing discards the whole step "
        "including the parts that were fine.",
    )
    p.add_argument(
        "-fold_guard",
        "-fold-guard",
        type=float,
        default=_D.fold_guard,
        help="Strength (0..1) of the HARD anti-folding guard, which damps or refuses "
        "any step that would increase folding; 0 (default) leaves it off in favour of "
        "-fold_penalty. Turn it on when a fold is genuinely inadmissible rather than "
        "merely unwanted. As a default it costs more than it buys: on a NIREP pair it "
        "fought 47,788 voxels of legitimate compression to prevent 13 folds, and "
        "throttled the finest level to a standstill doing it.",
    )

    # Axis constraints (match qwarp)
    p.add_argument("-noXdis", "-noxdis", action="store_true", help="No x-displacement.")
    p.add_argument("-noYdis", "-noydis", action="store_true", help="No y-displacement.")
    p.add_argument("-noZdis", "-nozdis", action="store_true", help="No z-displacement.")

    p.add_argument(
        "-matrix",
        "-1Dmatrix",
        "-1Dmatrix_apply",
        default=None,
        help="An .aff12.1D affine (as ffs_allineate -1Dmatrix_save writes) taking the "
        "SOURCE to the base. Given this, the source is NOT expected to be pre-aligned: "
        "the matrix is inverted and the BASE is resampled (wsinc5) onto the source's "
        "own grid, and the warp is estimated there. The source is never resampled, so "
        "it keeps every voxel it acquired. Everything written out -- warped image and "
        "warp fields -- is then on the SOURCE grid; carry it to base space with the "
        "warp applied to the source first: "
        "ffs_nwarp -source src.nii -nwarp 'matrix.aff12.1D out_WARP.nii' -master base.nii. "
        "NOTE: this does not recover the clipped FoV, it relocates it -- the base is "
        "what falls out of frame now, and the fixed image is the one interpolated. "
        "Measured worse than the default arrangement at a clipped edge; see "
        "../fmri_wiki/concepts/SyN.md.",
    )
    add_preallineate_args(p, "ffs_formwarp")

    # Weight / mask
    p.add_argument("-weight", help="Weight image emphasizing the metric (overrides automask).")
    add_coverage_args(p)

    # Output interpolation
    p.add_argument(
        "-final_interp",
        "-final-interp",
        choices=WARP_INTERP_MODES,
        default="wsinc5",
        help="Interpolation for the final warped image.",
    )

    add_warp_movie_args(p)

    # Device
    add_recipe_arg(p, _PRESET_BACKEND)
    add_deterministic_arg(p)
    add_device_arg(
        p,
        extra="On Apple Silicon use CPU: MPS falls back to CPU for SyN's 3-D grid-sample backward pass.",
    )
    add_verbose_arg(p)
    args = p.parse_args(argv, namespace or argparse.Namespace())
    # After parsing, so that "did the user type this flag" is answerable from argv
    # rather than guessed by comparing values against defaults.
    apply_recipe_preset(
        args,
        _PRESET_BACKEND,
        argv,
        verb=getattr(args, "verb", 1),
        image=getattr(args, "base", None),
    )
    return args


def _select_device(args: argparse.Namespace) -> torch.device:
    """Honour explicit devices; keep auto on CPU when only MPS is available."""
    spec = args.device
    if (spec is None or str(spec).lower() == "auto") and not torch.cuda.is_available():
        spec = "cpu"
    device = setup_device(spec, tf32=REGISTRATION_TF32)
    if device.type == "cpu" and args.device is None and args.verb >= 1:
        note = (
            " (recommended on Mac: MPS SyN backward falls back to CPU)"
            if torch.backends.mps.is_available()
            else ""
        )
        print(f"Using CPU{note}")
    return device


def _warp_stem(args: argparse.Namespace, pfx) -> str:
    """Stem the warp outputs hang off: ``-warp_prefix`` when given, else ``-prefix``."""
    wp = getattr(args, "warp_prefix", None)
    return parse_prefix(wp).stem if wp else pfx.stem


def _expected_outputs(args: argparse.Namespace) -> list[str]:
    """Concrete output paths a solo run of ``args`` would write, for -batch_skip.

    The warp files are named off ``-warp_prefix`` (else the parsed prefix),
    matching ``_dispatch_run``. A timeseries run with ``-warp_format folder``
    writes directories under these stems rather than files, so such a job is
    simply never skipped — safe."""
    pfx = parse_prefix(args.prefix)
    prefix, ext = _warp_stem(args, pfx), pfx.nifti_ext
    outs: list[str] = [pfx.as_file()]
    if args.save_warp or args.allineate:
        outs.append(f"{prefix}_WARP{ext}")
    if args.allineate:
        outs += preallineate_outputs(pfx.stem, ext, prefix)
    if args.save_inverse:
        outs.append(f"{prefix}_WARPINV{ext}")
    if args.save_halfway:
        outs += [
            f"{prefix}_HALF_mid2fixed{ext}",
            f"{prefix}_HALF_mid2moving{ext}",
            f"{prefix}_HALF_fixed2mid{ext}",
            f"{prefix}_HALF_moving2mid{ext}",
        ]
    movie = resolve_movie_path(args)
    if movie is not None:
        outs.append(movie)
    return outs


def _validate_batch_run(run_args: argparse.Namespace) -> None:
    """Per-run validation for a batch job: needs -base/-source/-prefix."""
    missing = [f for f in ("base", "source", "prefix") if getattr(run_args, f, None) is None]
    if missing:
        raise ValueError("run is missing " + ", ".join("-" + m for m in missing))


def _batch_dispatch(run_args: argparse.Namespace, device: torch.device) -> None:
    """Batch adapter: turn a nonzero return (a grid mismatch, say) into an
    exception so the shared runner records the job as failed."""
    rc = _dispatch_run(run_args, device)
    if rc != 0:
        raise ValueError(f"run failed (exit {rc})")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if getattr(args, "deterministic", False):
        enable_determinism(getattr(args, "verb", 1))

    if args.batch is not None or args.batch_run:
        # One process, many registrations: SyN's fixed costs (CUDA context,
        # torch.compile warmup) are paid once instead of per pair.
        run_batch_jobs(
            tool="ffs_formwarp",
            jobs=collect_batch_jobs(args.batch, args.batch_run),
            device=_select_device(args),
            parse_line=lambda line, base: parse_args(shlex.split(line), base),
            defaults=args,
            dispatch=_batch_dispatch,
            validate=_validate_batch_run,
            is_nested=lambda ra: ra.batch is not None or ra.batch_run is not None,
            expected_outputs=_expected_outputs,
            skip_existing=args.batch_skip,
            verb=args.verb,
        )
        return 0

    missing = [f for f in ("base", "source", "prefix") if getattr(args, f, None) is None]
    if missing:
        print(
            "ERROR: " + ", ".join("-" + m for m in missing) + " required "
            "(or use -batch FILE / -batch_run ARGS)."
        )
        return 1

    return _dispatch_run(args, _select_device(args))


def _dispatch_run(args: argparse.Namespace, device: torch.device) -> int:
    """Register one self-contained base/source pair (the entire per-pair body).

    Both the standalone path and every batch job go through here, so a manifest
    line reproduces a solo invocation bit-for-bit."""
    if args.allineate:
        pfx = parse_prefix(args.prefix)
        ws = _warp_stem(args, pfx)
        return run_with_preallineate(
            args,
            device,
            _dispatch_run,
            image_stem=pfx.stem,
            ext=pfx.nifti_ext,
            warp_stem=ws,
            warp_path=f"{ws}_WARP{pfx.nifti_ext}",
            final_interp=args.final_interp,
        )
    if args.verb >= 1:
        print(
            f"ffs_formwarp\n  device: {device}\n"
            f"  final data resampling kernel: {args.final_interp}\n"
        )

    t0 = time.time()
    with spinner(f"Loading {Path(args.base).name}"):
        base, base_info = load_image(args.base, device=torch.device("cpu"))
    with spinner(f"Loading {Path(args.source).name}"):
        source, source_info = load_image(args.source, device=torch.device("cpu"))

    base = sanitize_volume(base, "-base", args.verb)
    source = sanitize_volume(source, "-source", args.verb)

    if base.ndim == 4:
        if args.verb >= 1:
            print("WARNING: 4D -base; using vol[0] as the fixed target")
        base = base[0]

    # -matrix: invert the source->base affine and pull the base onto the source's grid,
    # so the registration runs in the source's own frame with the source untouched.
    # Everything after this point is identical to the pre-aligned path -- the only
    # difference is which grid "the grid" means, hence which header the outputs carry.
    out_info = base_info
    if args.matrix is not None:
        src_shape = tuple(source.shape[1:]) if source.ndim == 4 else tuple(source.shape)
        with spinner(f"Resampling base onto the {Path(args.source).name} grid"):
            base, out_info = base_into_source_frame(
                base, base_info, src_shape, source_info, args.matrix, device
            )
        if args.verb >= 1:
            print(f"Base resampled into source space: grid {src_shape[::-1]}")

    # Timeseries mode: a 4D -source registers every volume to the (3D) base and
    # writes a 4D warped series + a per-volume warp series (5D file or folder).
    timeseries = source.ndim == 4
    if timeseries and tuple(source.shape[1:]) != tuple(base.shape):
        print(
            f"ERROR: source volumes {tuple(source.shape[1:])} and base {tuple(base.shape)} "
            "must be on the same grid (resample first with ffs_allineate, or pass "
            "that alignment as -matrix)."
        )
        return 1
    if not timeseries and tuple(base.shape) != tuple(source.shape):
        print(
            f"ERROR: base {tuple(base.shape)} and source {tuple(source.shape)} "
            "must be on the same grid (resample first with ffs_allineate, or pass "
            "that alignment as -matrix)."
        )
        return 1

    nz, ny, nx = base.shape
    if args.verb >= 1:
        print(f"Base/source: {nx}x{ny}x{nz}, loaded in {time.time() - t0:.1f}s")

    # Weight / mask. The base side is pair-invariant, so build it once here; the
    # source side depends on the volume being registered and is rebuilt per volume in
    # timeseries mode (motion correction clips a different wedge out of every frame).
    weight = None
    if args.weight is not None:
        with spinner(f"Loading {Path(args.weight).name}"):
            weight, _ = load_image(args.weight, device=torch.device("cpu"))
        weight = weight.float().to(device)

    base_brain, base_cover = image_support(
        base, args, device, args.automask or args.automask_base, "base", args.verb
    )
    # A source automask on the temporal mean, not per frame: the brain doesn't move
    # between frames anywhere near enough to matter, and automask is far and away the
    # most expensive thing here. Data coverage IS per frame -- see _run_timeseries.
    src_ref = source.mean(dim=0) if source.ndim == 4 else source
    src_brain, src_cover = image_support(
        src_ref, args, device, args.automask or args.automask_source, "source", args.verb
    )

    # The brain masks damp the metric; the coverage masks are passed through so the
    # engine can cross-fill as well as exclude.
    brain = combine_brain_masks(base_brain, src_brain, args.automask_intersect)
    mask = None
    if brain is not None:
        if weight is not None:
            weight = weight * brain.float()
        else:
            mask = brain.float()

    warp_flags = 0
    if args.noXdis:
        warp_flags |= NO_X_DISP
    if args.noYdis:
        warp_flags |= NO_Y_DISP
    if args.noZdis:
        warp_flags |= NO_Z_DISP

    config = SynConfig(
        metric=args.metric,
        cc_radius=args.cc_radius,
        lpa_sigma=args.lpa_sigma,
        lpa_kernel=args.lpa_kernel,
        invert_iters=args.invert_iters,
        best_patience=args.best_patience,
        void_guard=args.void_guard,
        fold_guard=args.fold_guard,
        guard_floor=args.guard_floor,
        guard_refuses=args.guard_refuses,
        warp_flags=warp_flags,
        final_interp=args.final_interp,
        verb=args.verb,
        **_style_knobs(args),
    )

    pfx = parse_prefix(args.prefix)
    prefix, nii_ext = pfx.stem, pfx.nifti_ext
    warp_stem = _warp_stem(args, pfx)

    if timeseries:
        if args.movie is not None and args.verb >= 1:
            print("WARNING: -movie records a single 3D registration; ignored for a 4D -source")
        return _run_timeseries(
            args,
            base,
            source,
            weight,
            mask,
            base_cover,
            config,
            out_info,
            prefix,
            warp_stem,
            nii_ext,
            device,
            t0,
        )

    recorder = build_warp_movie_recorder(
        args, base, source, out_info["affine"], device, "formwarp", mask=mask
    )
    res = formwarp(
        base,
        source,
        weight=weight,
        mask=mask,
        fixed_cover=base_cover,
        moving_cover=src_cover,
        config=config,
        device=device,
        recorder=recorder,
    )

    warped_path = pfx.as_file()
    with spinner(f"Writing {Path(warped_path).name}"):
        save_image(res.warped, warped_path, header_info=out_info)
    if args.verb >= 1:
        print(f"Saved warped image: {warped_path}")

    def _save_warp(
        triple: tuple[torch.Tensor, torch.Tensor, torch.Tensor], path: str, label: str
    ) -> None:
        with spinner(f"Writing {Path(path).name}"):
            save_warp_field(
                triple[0].cpu(),
                triple[1].cpu(),
                triple[2].cpu(),
                path,
                header_info=out_info,
                units="mm",
            )
        if args.verb >= 1:
            print(f"Saved {label}: {path}")

    p = warp_stem
    if args.save_warp:
        _save_warp(res.fwd, f"{p}_WARP{nii_ext}", "moving->fixed warp")
    if args.save_inverse:
        _save_warp(res.inv, f"{p}_WARPINV{nii_ext}", "fixed->moving inverse warp")
    if args.save_halfway:
        _save_warp(res.fixed_to_mid, f"{p}_HALF_mid2fixed{nii_ext}", "mid->fixed half-warp")
        _save_warp(res.moving_to_mid, f"{p}_HALF_mid2moving{nii_ext}", "mid->moving half-warp")
        _save_warp(res.mid_to_fixed, f"{p}_HALF_fixed2mid{nii_ext}", "fixed->mid half-warp")
        _save_warp(res.mid_to_moving, f"{p}_HALF_moving2mid{nii_ext}", "moving->mid half-warp")

    render_warp_movie(recorder, args, args.verb)

    if args.verb >= 1:
        print(f"Done in {time.time() - t0:.1f}s")
    return 0


def _run_timeseries(
    args,
    base,
    source,
    weight,
    mask,
    base_cover,
    config,
    out_info,
    prefix,
    warp_stem,
    nii_ext,
    device,
    t0,
) -> int:
    """Register every volume of a 4D source to the 3D base (per-volume SyN).

    Writes the 4D warped series and, with -save_warp/-save_inverse, a per-volume
    warp series in the chosen -warp_format (one 5D file or a folder of 4D frames).
    -save_halfway is single-pair only and is ignored here.

    Data coverage is recomputed per volume: motion correction clips a different wedge
    out of every frame. The brain masks are not -- see _dispatch_run.
    """
    from tqdm import tqdm

    n_t = source.shape[0]
    if args.save_halfway and args.verb >= 1:
        print("NOTE: -save_halfway is ignored in timeseries mode.")
    if args.verb >= 1:
        print(f"Timeseries mode: {n_t} volumes -> base (per-volume SyN)")

    warped_frames: list[torch.Tensor] = []
    fwd_frames: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    inv_frames: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    quiet = config.verb
    for t in tqdm(range(n_t), desc="formwarp", disable=args.verb < 1, leave=True):
        cover_t = frame_coverage(source[t], args, device)
        # Silence per-volume SyN chatter; the bar is the progress signal.
        res = formwarp(
            base,
            source[t],
            weight=weight,
            mask=mask,
            fixed_cover=base_cover,
            moving_cover=cover_t,
            config=replace(config, verb=0) if quiet else config,
            device=device,
        )
        warped_frames.append(res.warped.cpu())
        if args.save_warp:
            fwd_frames.append(tuple(c.cpu() for c in res.fwd))
        if args.save_inverse:
            inv_frames.append(tuple(c.cpu() for c in res.inv))

    warped_path = f"{prefix}{nii_ext}"
    with spinner(f"Writing {Path(warped_path).name}"):
        save_image(torch.stack(warped_frames), warped_path, header_info=out_info)
    if args.verb >= 1:
        print(f"Saved warped series: {warped_path} ({n_t} volumes)")

    def _save_series(frames, tag: str, label: str) -> None:
        as_5d = args.warp_format == "5d"
        xs = torch.stack([f[0] for f in frames])
        ys = torch.stack([f[1] for f in frames])
        zs = torch.stack([f[2] for f in frames])
        dest = f"{warp_stem}_{tag}{nii_ext}" if as_5d else f"{warp_stem}_{tag}"
        with spinner(f"Writing {Path(dest).name}"):
            out = save_warp_series(xs, ys, zs, dest, as_5d=as_5d, header_info=out_info, units="mm")
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
