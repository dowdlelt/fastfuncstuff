"""CLI for GPU-accelerated affine/rigid alignment.

Command: allineate (registered as entry point in pyproject.toml)

Usage:
    allineate -base ref.nii -source mov.nii -prefix out.nii -1Dmatrix_save mat.aff12.1D

Speed presets:
    -fast       : Fewer iterations (~2x faster)
    -superfast  : Skip coarse search, minimal Adam (~5x faster, small motion only)
    Default is balanced quality/speed. For maximum accuracy, use -slow.
    The Powell polish is opt-in (-polish); +ZZ costs always get it.
"""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.cli_utils import (
    add_batch_args,
    add_deterministic_arg,
    add_device_arg,
    add_verbose_arg,
    collect_batch_jobs,
    enable_determinism,
    run_batch_jobs,
    setup_device,
    spinner,
)
from fastfuncstuff.processing.affine import (
    apply_affine,
    apply_affine_wsinc5,
    grid_from_dxyz,
    load_matrix_1D,
    save_matrix_1D,
)
from fastfuncstuff.processing.allineate import AffineAlignConfig, allineate
from fastfuncstuff.processing.io import derive_mean_output_path, load_image, save_image
from fastfuncstuff.utils import REGISTRATION_TF32


def _output_grid(
    base_affine: np.ndarray, base_shape: tuple[int, int, int], dxyz: list[float] | None
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Output affine + shape: the base grid, or a ``-dxyz`` master grid at a new voxel size."""
    if dxyz is None:
        return np.asarray(base_affine), tuple(base_shape)
    return grid_from_dxyz(base_affine, base_shape, dxyz)


def _out_matrix(
    matrix: torch.Tensor, base_affine: np.ndarray, out_affine: np.ndarray, device: torch.device
) -> torch.Tensor:
    """Compose the base-voxel→source matrix onto the output grid (out-voxel→source).

    For the base grid this is a no-op; for a ``-dxyz`` grid it prepends the out-voxel→
    base-voxel map ``inv(base_affine)·out_affine`` (both are voxel→world in the same space).
    """
    if np.array_equal(np.asarray(out_affine), np.asarray(base_affine)):
        return matrix
    # Header transforms are tiny and precision-sensitive.  Compose them on CPU
    # in float64, then return the 4x4 result to the requested compute device;
    # this avoids both slow consumer-CUDA float64 and MPS's lack of float64.
    ab = torch.as_tensor(np.asarray(base_affine), dtype=torch.float64)
    ao = torch.as_tensor(np.asarray(out_affine), dtype=torch.float64)
    out = matrix.detach().cpu().double() @ torch.linalg.inv(ab) @ ao
    return out.to(device=device, dtype=matrix.dtype)


def _resample(
    source_vol: torch.Tensor,
    matrix: torch.Tensor,
    out_shape: tuple[int, int, int],
    final_interp: str,
    no_neg: bool = False,
) -> torch.Tensor:
    """Pull-resample a source volume through ``matrix`` onto ``out_shape`` (final interp)."""
    if final_interp == "wsinc5":
        warped = apply_affine_wsinc5(source_vol, matrix, out_shape)
    else:
        warped = apply_affine(source_vol, matrix, out_shape, zero_outside=True)
    return warped.clamp_min(0.0) if no_neg else warped


def _follower_pairs(args: argparse.Namespace) -> list[tuple[str, str]]:
    """(dataset, prefix) pairs from -source_follower / -follower_prefix, validated."""
    followers = args.source_follower or []
    prefixes = args.follower_prefix or []
    if not followers and not prefixes:
        return []
    if not followers:
        raise SystemExit("-follower_prefix given without -source_follower")
    if len(prefixes) != len(followers):
        raise SystemExit(
            f"-follower_prefix takes one prefix per -source_follower "
            f"({len(followers)} follower(s), {len(prefixes)} prefix(es))"
        )
    return list(zip(followers, prefixes, strict=True))


def _apply_followers(
    args: argparse.Namespace,
    matrix: torch.Tensor,
    base_header: dict,
    source_affine: np.ndarray,
    out_affine: np.ndarray,
    out_shape: tuple[int, int, int],
    device: torch.device,
) -> None:
    """Warp each follower with the source's transform and save it on the output grid.

    ``matrix`` is out-voxel -> source-voxel. A follower on its own grid needs
    out-voxel -> follower-voxel, i.e. ``inv(A_follower) @ A_source @ matrix``
    (world coordinates are the common frame), so followers do not have to share
    the source's grid.
    """
    verb = args.verb
    a_src = torch.as_tensor(np.asarray(source_affine), dtype=torch.float64)
    for path, prefix in _follower_pairs(args):
        with spinner(f"Loading {Path(path).name}"):
            follower, follower_header = load_image(path, device=device)
        a_fol = torch.as_tensor(np.asarray(follower_header["affine"]), dtype=torch.float64)
        fm = torch.linalg.inv(a_fol) @ a_src @ matrix.detach().cpu().double()
        fm = fm.to(device=device, dtype=matrix.dtype)
        if follower.ndim == 4:
            warped = torch.stack(
                [
                    _resample(follower[t], fm, out_shape, args.final_interp, args.no_neg)
                    for t in range(follower.shape[0])
                ]
            )
        else:
            warped = _resample(follower, fm, out_shape, args.final_interp, args.no_neg)
        with spinner(f"Writing {Path(prefix).name}"):
            save_image(warped, prefix, header_info=base_header, affine=out_affine)
        if verb >= 1:
            print(f"Saved follower: {prefix}")


def _resolve_ov(args: argparse.Namespace) -> float:
    """Overlap-penalty weight, defaulting the way AFNI does per cost.

    ``-ov`` is off unless asked for, except under the lpa+/lpc+ family, where
    the overlap term is part of the published combination (DEFAULT_MICHO_*_OV
    = 0.4 in 3dAllineate.c) and leaving it off would not be that cost.
    """
    if args.ov is not None:
        return args.ov
    return 0.4 if args.cost.lower().startswith(("lpa+", "lpc+")) else 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="allineate",
        description="GPU-accelerated affine/rigid alignment (inspired by 3dAllineate)",
        epilog="""Speed/quality presets:
  -superfast  Onepass, <=150 Adam iters (small motion only)
  -fast       <=300 Adam iters (~2x faster)
  (default)   <=400 Adam iters, early-stops on plateau
  -slow       600 iters at 2x, 800 at full-res, 2000 Powell evals
  -polish     add the Powell polish (automatic for +ZZ costs)

Examples:
  # Standard brain alignment:
  allineate -base mni.nii -source subj.nii -prefix out.nii -cost lpa

  # Fast rigid alignment (e.g., motion correction):
  allineate -base vol0.nii -source vol1.nii -prefix out.nii -rigid -fast

  # High-quality with wsinc5 output:
  allineate -base mni.nii -source subj.nii -prefix out.nii -slow -final wsinc5

  # Apply existing matrix:
  allineate -base mni.nii -source subj.nii -prefix out.nii -1Dmatrix_apply mat.aff12.1D

  # Solve on the skull-stripped volume, carry the original along:
  allineate -base mni.nii -source subj_ss.nii -prefix out_ss.nii \\
            -source_follower subj.nii -follower_prefix out_orig.nii
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Input/Output ---
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "-base", default=None, help="Base/reference image (.nii/.nii.gz) [required unless -batch]"
    )
    io_group.add_argument(
        "-source", default=None, help="Source/moving image to align [required unless -batch]"
    )
    io_group.add_argument(
        "-prefix", default=None, help="Output aligned image [required unless -batch]"
    )
    add_batch_args(
        io_group,
        tool="ffs_allineate",
        what="affine alignments",
        example="-base ref.nii -source mov.nii -prefix out.nii -1Dmatrix_save m.aff12.1D",
        skip_note="-prefix / -follower_prefix / -1Dmatrix_save / -save_mean / -save_weight "
        "/ -save_automask / -save_cmass",
    )
    io_group.add_argument(
        "-1Dmatrix_save", default=None, help="Save affine matrix as .aff12.1D (AFNI format)"
    )
    io_group.add_argument(
        "-1Dmatrix_apply", default=None, help="Apply existing matrix (skip alignment)"
    )
    io_group.add_argument("-base_index", type=int, default=None, help="Use volume N from 4D base")
    io_group.add_argument(
        "-dxyz",
        type=float,
        nargs="+",
        default=None,
        metavar="MM",
        help="Write the output at this voxel size (mm) instead of the base's grid — AFNI "
        "-mast_dxyz. Keeps the base's SPACE (orientation, centre, field of view) but resamples "
        "to the given resolution: one value for isotropic (e.g. -dxyz 0.8 to preserve hi-res), "
        "or three for x y z. The transform is unchanged; only the output grid differs. Works "
        "with both the normal alignment and -1Dmatrix_apply.",
    )
    io_group.add_argument(
        "-source_follower",
        "-source-follower",
        dest="source_follower",
        nargs="+",
        default=None,
        metavar="DSET",
        help="Extra dataset(s) that ride along on the transform solved for -source, written "
        "with -follower_prefix. The usual case is aligning a skull-stripped volume while the "
        "original (or any derived map) needs the identical transform, without a second run to "
        "replay the saved matrix. A follower on a different grid than -source is handled "
        "(the matrix is re-expressed through its affine); 4D followers are warped per volume.",
    )
    io_group.add_argument(
        "-follower_prefix",
        "-follower-prefix",
        dest="follower_prefix",
        nargs="+",
        default=None,
        metavar="PREFIX",
        help="Output path for each -source_follower, in the same order (one per follower).",
    )
    io_group.add_argument(
        "-save_mean",
        action="store_true",
        help="If source is 4D, save mean output as mean_{prefix_basename}{ext}",
    )

    # --- Alignment mode ---
    mode_group = parser.add_argument_group("Alignment mode")
    mode_ex = mode_group.add_mutually_exclusive_group()
    mode_ex.add_argument("-rigid", action="store_true", help="6 DoF: translation + rotation only")
    mode_ex.add_argument(
        "-affine", action="store_true", default=True, help="12 DoF: full affine (default)"
    )
    mode_ex.add_argument(
        "-EPI", action="store_true", help="9 DoF: EPI-specific (freeze x/z scale, z shear)"
    )

    # --- Cost function ---
    cost_group = parser.add_argument_group("Cost function")
    cost_group.add_argument(
        "-cost",
        choices=[
            "ls",
            "lpa",
            "lpc",
            "lpa+",
            "lpc+",
            "lpa+ZZ",
            "lpc+ZZ",
            "lps",
            "lpsc",
            "mi",
            "nmi",
            "je",
            "hel",
            "cru",
            "cra",
            "crm",
        ],
        default="lpa",
        help="Cost function (AFNI-faithful unless noted): "
        "ls=clipped Pearson; "
        "lpa=local Pearson absolute (default, similar contrast); "
        "lpc=local Pearson signed (cross-modal, e.g. EPI-to-anat); "
        "lpa+/lpc+=the same plus weighted hel/mi/nmi/crA/overlap terms, which "
        "widen the basin and make the search more robust; "
        "lpa+ZZ/lpc+ZZ=search with the combination, then finish on the pure "
        "cost (AFNI's most robust pair: lpc+ZZ for EPI-to-T1, lpa+ZZ for "
        "T1-to-T1); "
        "mi/nmi=(normalized) mutual information; je=joint entropy; "
        "hel=Hellinger; cru/cra/crm=correlation ratio "
        "(unsym/additive/multiplicative). "
        "lps/lpsc=ffs-special per-voxel Gaussian local Pearson "
        "(absolute/signed).",
    )
    cost_group.add_argument(
        "-blok",
        "-bloktype",
        dest="bloktype",
        choices=["tohd", "rhdd", "cube"],
        default="tohd",
        help="Blok shape for lpa/lpc local neighborhoods (default: tohd, like 3dAllineate)",
    )
    cost_group.add_argument(
        "-blokrad",
        type=float,
        default=None,
        help="Blok radius in mm for lpa/lpc (default: auto, ~555 voxels per blok)",
    )
    cost_group.add_argument(
        "-lpa_sigma",
        type=float,
        default=4.0,
        help="Kernel parameter for lps/lpsc neighborhoods "
        "in voxels. For gauss: sigma. "
        "For box: half-width radius. "
        "Use 0 with -lpa_kernel box to auto-size "
        "to ~500 voxels (default: 4.0)",
    )
    cost_group.add_argument(
        "-lpa_kernel",
        choices=["gauss", "box"],
        default="gauss",
        help="lps/lpsc neighborhood kernel: "
        "gauss=Gaussian weighting (default), "
        "box=uniform weighting",
    )
    cost_group.add_argument(
        "-ov",
        "-overlap",
        dest="ov",
        type=float,
        default=None,
        help="Overlap penalty weight (AFNI lpc+/lpa+ 'ov'; default 0=off, but "
        "0.4 for the lpa+/lpc+ costs, matching AFNI). "
        "Adds a differentiable (max(0,9.95-10*overlap))^2 term that pushes the "
        "refiner back toward full base/source overlap. Try ~0.05-0.5 if a "
        "whole-brain alignment drifts partly out of overlap.",
    )

    # --- Interpolation ---
    interp_group = parser.add_argument_group("Interpolation")
    interp_group.add_argument(
        "-interp",
        choices=["linear", "cubic"],
        default="linear",
        help="During optimization (default: linear)",
    )
    interp_group.add_argument(
        "-final",
        choices=["linear", "cubic", "wsinc5"],
        default="wsinc5",
        dest="final_interp",
        help="For output image (default: wsinc5). Use linear for compatibility/speed",
    )
    interp_group.add_argument(
        "-no_neg",
        "-no-neg",
        dest="no_neg",
        action="store_true",
        help="Clamp the output image at 0 to suppress wsinc5/cubic negative ringing "
        "on non-negative data (magnitude, masks, probability maps).",
    )

    # --- Masking ---
    mask_group = parser.add_argument_group("Masking")
    mask_group.add_argument(
        "-automask",
        dest="source_automask",
        action="store_true",
        help="Automask source to exclude background",
    )
    mask_group.add_argument(
        "-source_automask", dest="source_automask", action="store_true", help="Alias for -automask."
    )
    mask_group.add_argument(
        "-autoweight",
        action="store_true",
        default=True,
        help="Weight by base intensity (default: on)",
    )
    mask_group.add_argument(
        "-noautoweight", action="store_true", help="Disable intensity-based weighting"
    )
    mask_group.add_argument(
        "-save_automask", default=None, metavar="PREFIX", help="Save computed automask to PREFIX"
    )
    mask_group.add_argument(
        "-save_weight",
        "-save-weight",
        default=None,
        metavar="PREFIX",
        help="Save the exact optimisation weight (autoweight × validity × "
        "source automask) — compare to AFNI 3dAllineate -wtprefix",
    )

    # --- Search control ---
    search_group = parser.add_argument_group("Search control")
    search_group.add_argument(
        "-cmass",
        action="store_true",
        default=True,
        help="Center-of-mass pre-alignment (default: on)",
    )
    search_group.add_argument(
        "-nocmass", action="store_true", help="Disable center-of-mass pre-alignment"
    )
    search_group.add_argument(
        "-cmass_direct",
        "-cmass-direct",
        type=float,
        nargs=3,
        default=None,
        metavar=("DX", "DY", "DZ"),
        help="Manual cmass shift in base-grid voxels "
        "(same space/sign the auto cmass prints); "
        "skips automatic center-of-mass",
    )
    search_group.add_argument(
        "-save_cmass",
        "-save-cmass",
        default=None,
        metavar="PREFIX",
        help="Save the source positioned by the cmass shift "
        "alone (for checking/hand-tuning placement)",
    )
    search_group.add_argument(
        "-twopass", action="store_true", default=True, help="Coarse search + refinement (default)"
    )
    search_group.add_argument(
        "-onepass", action="store_true", help="Skip coarse search (small motion only)"
    )
    search_group.add_argument(
        "-coarse_range",
        type=float,
        default=30.0,
        help="Coarse rotation half-range in degrees (default: 30)",
    )
    search_group.add_argument(
        "-coarse_step", type=float, default=5.0, help="Coarse angular step in degrees (default: 5)"
    )
    search_group.add_argument(
        "-coarse_shift_frac",
        type=float,
        default=0.321,
        help="Coarse translation half-range as a fraction of grid size (default: 0.321, like AFNI)",
    )
    range_ex = search_group.add_mutually_exclusive_group()
    range_ex.add_argument(
        "-smallrange",
        action="store_true",
        help="Halve all coarse search ranges (angle/shift/scale)",
    )
    range_ex.add_argument(
        "-verysmallrange", action="store_true", help="Quarter all coarse search ranges"
    )
    range_ex.add_argument(
        "-hugerange",
        action="store_true",
        help="Widen all coarse search ranges by 1.5x (rotation ±45° instead of "
        "±30°) and keep the same angular step, so the seed grid gets "
        "correspondingly denser. For a source that starts badly off — grossly "
        "oblique, wrong nominal orientation, an eyeballed misplacement. Costs a "
        "few seconds in the coarse pass; the default range is right for data "
        "that is merely imperfectly positioned.",
    )
    search_group.add_argument(
        "-tbest",
        type=int,
        default=None,
        help="Coarse candidates to refine (default: 10 for lpa/lpc, 3 otherwise). "
        "The blok costs refine their trials in one batch, so up to ~10 costs no "
        "measurable time and buys extra chances that the right basin survives the "
        "coarse search; the other costs pay for each trial in full.",
    )
    search_group.add_argument(
        "-nmatch",
        "-n_match",
        dest="n_match",
        type=float,
        default=0.47,
        help="Match-point budget for lpa/lpc refinement (AFNI npt_match), "
        "unit-free: <=1.0 is a FRACTION of the in-mask voxels (0.47 = AFNI "
        "default, 1.0 = all); >1.0 is an absolute count (e.g. 150000). The cost "
        "is evaluated on that many random weight-domain points per iteration "
        "instead of the full grid.",
    )
    search_group.add_argument(
        "-noautocrop", action="store_true", help="Disable auto-cropping of zero margins"
    )
    search_group.add_argument(
        "-work_dxyz",
        "-work-dxyz",
        default="auto",
        metavar="auto|off|MM",
        help="Voxel size (mm) the SEARCH runs at. 'auto' (default) uses the "
        "coarser of the base and source spacings -- aligning a 3 mm EPI on a "
        "1 mm anat grid costs 27x the voxels and resolves nothing the EPI does "
        "not; 'off' searches on the base's own grid; a number forces that "
        "spacing. The fit is mapped back exactly, so the saved matrix and the "
        "output volume are on the base grid either way.",
    )

    # --- Speed/quality presets ---
    speed_group = parser.add_argument_group("Speed/quality")
    speed_ex = speed_group.add_mutually_exclusive_group()
    speed_ex.add_argument(
        "-superfast", action="store_true", help="Minimal: onepass, <=150 Adam iters, no Powell"
    )
    speed_ex.add_argument(
        "-fast", action="store_true", help="Quick: <=300 Adam iters, no Powell polish"
    )
    speed_ex.add_argument(
        "-slow", action="store_true", help="Thorough: <=600/800 Adam iters, 2000 Powell evals"
    )
    speed_group.add_argument(
        "-polish",
        action="store_true",
        help="Run the final Powell polish. Off by default: on same-modality data it "
        "moves the cost by ~1e-5 relative (below this tool's ~2e-2 run-to-run spread) "
        "for ~15%% of the runtime, and on cross-modal lpc it more often drifts out of "
        "the basin and gets rejected. It is still applied automatically for +ZZ costs, "
        "where it is not a polish but the step that re-optimizes on the pure cost.",
    )
    speed_group.add_argument(
        "-no_polish",
        "-no-polish",
        dest="no_polish",
        action="store_true",
        help="Never polish, even for a +ZZ cost. The +ZZ combination terms are still "
        "dropped, so the reported cost stays the pure one -- it just is not re-optimized.",
    )
    speed_group.add_argument(
        "-lr",
        "-adam_lr",
        dest="adam_lr",
        type=float,
        default=0.005,
        help="Adam learning rate for full-res refinement "
        "(default: 0.005; higher converges faster but "
        "can overshoot to a worse optimum)",
    )

    # --- Hardware ---
    hw_group = parser.add_argument_group("Hardware")
    add_device_arg(
        hw_group,
        extra="On Apple Silicon, MPS is recommended for typical full-size brain volumes; CPU may win on small jobs.",
    )
    add_deterministic_arg(hw_group)
    hw_group.add_argument(
        "-optimizer",
        choices=("auto", "adam", "pattern", "cmaes"),
        default="auto",
        help="Refinement optimizer. 'auto' (default) picks per stage: CMA-ES "
        "while the cost evaluation is launch-bound (small/subsampled problems, "
        "where its population is free and it is ~4.7x faster), Adam once a "
        "generation's work -- points x trials x population -- would be real "
        "compute (big volumes with many trials, where CMA-ES is ~2x slower). "
        "'adam' is autograd + Adam. 'pattern' is a "
        "batched derivative-free coordinate search. 'cmaes' is batched CMA-ES, "
        "which additionally adapts a covariance and so follows the correlated "
        "rotation/translation valley that defeats a coordinate stencil. Both "
        "spend the (free) batch dimension on search instead of on a backward "
        "pass, which suits this cost's small-scale roughness.",
    )
    hw_group.add_argument(
        "-compile",
        action="store_true",
        help="torch.compile the batched refinement forward to cut per-iteration "
        "launch overhead (also enabled by FFS_ALLINEATE_COMPILE=1). First stage "
        "pays a one-time compile warmup.",
    )
    add_verbose_arg(hw_group, default=1)

    args = parser.parse_args(argv)
    return args


def _parse_work_dxyz(value: str | float | None) -> str | float:
    """``-work_dxyz`` -> what AffineAlignConfig wants: "auto", "off", or mm."""
    if value is None:
        return "auto"
    if isinstance(value, (int, float)):
        return float(value)
    v = value.strip().lower()
    if v in ("auto", "off", "none"):
        return v
    try:
        mm = float(v)
    except ValueError:
        raise SystemExit(
            f"-work_dxyz: expected auto, off, or a size in mm, got {value!r}"
        ) from None
    if mm <= 0:
        raise SystemExit(f"-work_dxyz: size must be positive, got {mm}")
    return mm


def _select_device(device_arg: str | None) -> torch.device:
    """Honour ``-device`` if given, else CUDA > MPS > CPU."""
    return setup_device(device_arg, tf32=REGISTRATION_TF32)


def _expected_outputs(args: argparse.Namespace) -> list[str]:
    """Concrete output paths a solo run of ``args`` would write, for -batch_skip.

    Paths are used verbatim (allineate does not run them through parse_prefix).
    -save_mean and the diagnostic maps are listed on intent: if a runtime guard
    means one is never written (a 3-D source has no mean; -save_automask needs
    -source_automask), the job simply isn't skipped next time — safe, since
    re-running costs less than a wrong skip."""
    outs: list[str] = [args.prefix]
    matrix_save = getattr(args, "1Dmatrix_save", None)
    if matrix_save is not None:
        outs.append(matrix_save)
    if args.save_mean:
        outs.append(derive_mean_output_path(args.prefix))
    outs.extend(prefix for _, prefix in _follower_pairs(args))
    for name in ("save_weight", "save_automask", "save_cmass"):
        val = getattr(args, name, None)
        if val is not None:
            outs.append(val)
    return outs


def _validate_batch_run(run_args: argparse.Namespace) -> None:
    """Per-run validation for a batch job: needs -base/-source/-prefix."""
    missing = [f for f in ("base", "source", "prefix") if getattr(run_args, f, None) is None]
    if missing:
        raise ValueError("run is missing " + ", ".join("-" + m for m in missing))


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point for allineate."""
    args = parse_args(argv)

    # Before any work: enable_determinism may re-exec to get CUBLAS_WORKSPACE_CONFIG
    # in place, and for a batch that has to happen once, up front, not per run.
    if getattr(args, "deterministic", False):
        enable_determinism(getattr(args, "verb", 1))

    if args.batch is not None or args.batch_run:
        # One process, many alignments: the Python/CUDA/torch.compile startup is
        # paid once instead of per pair. The per-pair work is unchanged.
        run_batch_jobs(
            tool="ffs_allineate",
            jobs=collect_batch_jobs(args.batch, args.batch_run),
            device=_select_device(args.device),
            parse_line=lambda line: parse_args(shlex.split(line)),
            dispatch=_dispatch_run,
            validate=_validate_batch_run,
            is_nested=lambda ra: ra.batch is not None or ra.batch_run is not None,
            expected_outputs=_expected_outputs,
            skip_existing=args.batch_skip,
            verb=args.verb,
        )
        return

    missing = [f for f in ("base", "source", "prefix") if getattr(args, f, None) is None]
    if missing:
        print(
            "Error: " + ", ".join("-" + m for m in missing) + " required "
            "(or use -batch FILE / -batch_run ARGS).",
            file=sys.stderr,
        )
        sys.exit(1)

    _dispatch_run(args, _select_device(args.device))


def _dispatch_run(args: argparse.Namespace, device: torch.device) -> None:
    """Align one self-contained base/source pair (the entire per-pair body).

    Both the standalone path and every batch job go through here, so a manifest
    line reproduces a solo invocation bit-for-bit."""
    verb = args.verb
    if verb >= 1:
        print(f"allineate: device={device}")

    # Validate follower pairing before the (long) alignment, not after it.
    _follower_pairs(args)

    # --- Load images ---
    t0 = time.time()
    with spinner(f"Loading {Path(args.base).name}"):
        base, base_header = load_image(args.base, device=device)
    with spinner(f"Loading {Path(args.source).name}"):
        source, source_header = load_image(args.source, device=device)

    # Handle 4D base
    if base.ndim == 4:
        idx = args.base_index if args.base_index is not None else 0
        if verb >= 1:
            print(f"Using volume {idx} from 4D base ({base.shape[0]} volumes)")
        base = base[idx]

    # Handle 4D source
    source_4d = None
    if source.ndim == 4:
        source_4d = source
        source = source[0]
        if verb >= 1:
            print(f"Source is 4D ({source_4d.shape[0]} volumes), aligning first volume")

    if verb >= 1:
        print(f"Base: {args.base} {base.shape}")
        print(f"Source: {args.source} {source.shape}")
        print(f"Load time: {time.time() - t0:.2f}s")

    # Output grid: the base's grid by default, or a -dxyz master grid (base space/FOV/centre
    # at a new voxel size). The transform is applied into `out_shape` and saved with `out_affine`.
    if args.dxyz is not None and len(args.dxyz) not in (1, 3):
        raise SystemExit(f"-dxyz takes 1 (isotropic) or 3 (x y z) values, got {len(args.dxyz)}")
    out_affine, out_shape = _output_grid(base_header["affine"], tuple(base.shape), args.dxyz)
    if args.dxyz is not None and verb >= 1:
        print(f"Output grid (-dxyz {args.dxyz}): {base.shape} -> {out_shape}")

    # --- Apply existing matrix ---
    if getattr(args, "1Dmatrix_apply", None) is not None:
        matrix_path = getattr(args, "1Dmatrix_apply")
        if verb >= 1:
            print(f"Applying matrix from {matrix_path}")

        matrix = load_matrix_1D(
            matrix_path,
            base_affine=base_header["affine"],
            source_affine=source_header["affine"],
        )
        matrix = matrix.to(device)
        om = _out_matrix(matrix, base_header["affine"], out_affine, device)
        warped = _resample(source, om, out_shape, args.final_interp, args.no_neg)
        with spinner(f"Writing {Path(args.prefix).name}"):
            save_image(warped, args.prefix, header_info=base_header, affine=out_affine)
        if verb >= 1:
            print(f"Saved: {args.prefix}")
        _apply_followers(
            args, om, base_header, source_header["affine"], out_affine, out_shape, device
        )
        return

    # --- Build config with speed/quality presets ---
    dof = "affine"
    if args.rigid:
        dof = "rigid"
    elif args.EPI:
        dof = "epi"

    # Iteration counts are *ceilings*: Adam stops early once the cost plateaus
    # (relative tolerance), so a generous cap costs nothing when converged but
    # lets hard cases keep improving instead of cutting off mid-descent.
    adam_iters_2x = 300
    adam_iters_1x = 400
    # The polish is opt-in. Measured on same-modality EPI->EPI lpa it was rejected
    # by the never-worse guard in 4 of 9 pairs and gained at most 9e-5 on a cost of
    # ~3.6 in the rest -- below allineate's own ~2e-2 run-to-run spread -- for ~15%
    # of the runtime. A +ZZ cost is the exception: there the "polish" is the pass
    # that re-optimizes on the pure cost after the basin-widening terms are dropped,
    # so it stays on unless -no_polish.
    _is_zz = args.cost.lower().endswith("+zz")
    powell_maxfev = 500 if (args.polish or _is_zz) and not args.no_polish else 0
    twopass = not args.onepass

    # Apply presets
    if args.superfast:
        adam_iters_2x = 150
        adam_iters_1x = 150
        powell_maxfev = 0
        twopass = False
        if verb >= 1:
            print("Mode: superfast (onepass, <=150 iters, no Powell)")
    elif args.fast:
        adam_iters_2x = 300
        adam_iters_1x = 300
        powell_maxfev = 0
        if verb >= 1:
            print("Mode: fast (<=300 iters, no Powell)")
    elif args.slow:
        adam_iters_2x = 600
        adam_iters_1x = 800
        # -slow is a request for maximum accuracy, so it opts into the polish.
        powell_maxfev = 0 if args.no_polish else 2000
        if verb >= 1:
            print("Mode: slow (<=600/800 iters, 2000 Powell evals)")

    # Auto-size box radius if requested
    lpa_sigma = args.lpa_sigma
    if args.lpa_kernel == "box" and lpa_sigma <= 0:
        from fastfuncstuff.processing.cost import auto_box_radius

        lpa_sigma = float(auto_box_radius(500))
        if verb >= 1:
            side = 2 * int(lpa_sigma) + 1
            print(f"Auto box radius: {int(lpa_sigma)} ({side}³ = {side**3} voxels)")

    config = AffineAlignConfig(
        dof=dof,
        cost=args.cost,
        lpa_sigma=lpa_sigma,
        lpa_kernel=args.lpa_kernel,
        bloktype=args.bloktype,
        blokrad=args.blokrad,
        ov=_resolve_ov(args),
        n_match=args.n_match,
        compile=args.compile,
        optimizer=args.optimizer,
        twopass=twopass,
        coarse_range=args.coarse_range,
        coarse_step=args.coarse_step,
        coarse_shift_frac=args.coarse_shift_frac,
        range_scale=(
            0.25
            if args.verysmallrange
            else 0.5
            if args.smallrange
            else 1.5
            if args.hugerange
            else 1.0
        ),
        adam_lr=args.adam_lr,
        tbest=args.tbest,
        adam_iters_2x=adam_iters_2x,
        adam_iters_1x=adam_iters_1x,
        powell_maxfev=powell_maxfev,
        cmass=not args.nocmass,
        cmass_direct=(tuple(args.cmass_direct) if args.cmass_direct is not None else None),
        interp=args.interp,
        final_interp=args.final_interp,
        source_automask=args.source_automask,
        autoweight=not args.noautoweight,
        autocrop=not args.noautocrop,
        work_dxyz=_parse_work_dxyz(args.work_dxyz),
        device=str(device),
        verb=verb,
    )

    # --- Run alignment ---
    t1 = time.time()
    matrix, warped = allineate(
        base,
        source,
        config,
        base_header=base_header,
        source_header=source_header,
        save_automask_path=args.save_automask,
        save_cmass_path=args.save_cmass,
        save_weight_path=args.save_weight,
    )

    if verb >= 1:
        print(f"Alignment time: {time.time() - t1:.2f}s")

    # --- Save outputs ---
    # -dxyz: re-resample onto the master grid at the requested voxel size (allineate returned
    # `warped` on the base grid). Same transform, finer/coarser output.
    if args.dxyz is not None:
        om = _out_matrix(matrix, base_header["affine"], out_affine, device)
        warped = _resample(source, om, out_shape, args.final_interp, args.no_neg)
    elif args.no_neg:
        warped = warped.clamp_min(0.0)
    with spinner(f"Writing {Path(args.prefix).name}"):
        save_image(warped, args.prefix, header_info=base_header, affine=out_affine)
    if verb >= 1:
        print(f"Saved: {args.prefix}")

    _apply_followers(
        args,
        _out_matrix(matrix, base_header["affine"], out_affine, device),
        base_header,
        source_header["affine"],
        out_affine,
        out_shape,
        device,
    )

    matrix_save_path = getattr(args, "1Dmatrix_save", None)
    if matrix_save_path is not None:
        save_matrix_1D(
            matrix,
            matrix_save_path,
            base_affine=base_header["affine"],
            source_affine=source_header["affine"],
        )
        if verb >= 1:
            print(f"Saved matrix: {matrix_save_path}")

    # --- Apply to all 4D volumes ---
    if source_4d is not None:
        if verb >= 1:
            print(f"Applying alignment to all {source_4d.shape[0]} volumes...")
        om = _out_matrix(matrix, base_header["affine"], out_affine, device)
        aligned_vols = [
            _resample(source_4d[t], om, out_shape, args.final_interp, args.no_neg)
            for t in range(source_4d.shape[0])
        ]
        result_4d = torch.stack(aligned_vols)
        with spinner(f"Writing {Path(args.prefix).name}"):
            save_image(result_4d, args.prefix, header_info=base_header, affine=out_affine)
        if verb >= 1:
            print(f"Saved 4D result: {args.prefix}")

        if args.save_mean:
            mean_path = derive_mean_output_path(args.prefix)
            mean_image = result_4d.mean(dim=0)
            with spinner(f"Writing {Path(mean_path).name}"):
                save_image(mean_image, mean_path, header_info=base_header, affine=out_affine)
            if verb >= 1:
                print(f"Saved mean: {mean_path}")
    elif args.save_mean and verb >= 1:
        print("-save_mean requested, but source is not 4D; skipping mean output")

    if verb >= 1:
        print(f"Total time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
