"""Command-line interface for qwarp_torch.

Provides a CLI similar to AFNI's 3dQwarp with PyTorch/GPU acceleration.

Usage:
    # Standard: separate base and source
    qwarp -base base.nii.gz -source source.nii.gz -prefix warped.nii.gz

    # Timeseries: single 4D file, first volume = base, warp all others
    qwarp -base epi_4d.nii.gz -prefix corrected.nii.gz

    # Timeseries with temporal smoothing and mean base
    qwarp -base epi_4d.nii.gz -prefix corrected.nii.gz -chainwarp \\
          -base_method mean -base_navg 10 -tsmooth 2.0
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import deque
from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import Tensor

from .interp import warp_image_linear
from .io import load_image, load_warp_field, save_image, save_warp_field
from .memory import estimate_gpu_memory_gb, print_memory_report
from .warp import QwarpConfig, _compute_padding, _pad_volume, qwarp


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="qwarp",
        description=(
            "GPU-accelerated nonlinear image registration (PyTorch port of AFNI 3dQwarp).\n"
            "\n"
            "Two modes of operation:\n"
            "  Standard:    qwarp -base T1.nii.gz -source T2.nii.gz -prefix out\n"
            "  Timeseries:  qwarp -base epi_4d.nii.gz -prefix out  (omit -source)\n"
            "\n"
            "In timeseries mode, the 4D input is split into volumes. By default vol[0]\n"
            "is the base and vol[1:] are warped to it. Use -base_method to build a\n"
            "mean/median base instead. Per-volume warps are saved alongside the 4D output."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Input / Output ──────────────────────────────────────────────────
    g_io = p.add_argument_group(
        "Input / Output",
        "Required arguments for specifying images and output location.",
    )
    g_io.add_argument("-base", required=True,
                      help="Base (target/reference) image (.nii/.nii.gz). "
                           "If 4D and -source is omitted, enters timeseries mode")
    g_io.add_argument("-source", default=None,
                      help="Source (moving) image to warp to the base. "
                           "Omit for timeseries mode with a single 4D -base file")
    g_io.add_argument("-base_index", type=int, default=None, metavar="N",
                      help="When -base is 4D and -source is provided, use volume N "
                           "(0-indexed) from -base as the reference. E.g. -base_index 0 "
                           "uses the first volume. Without this, a 4D -base with -source "
                           "is an error")
    g_io.add_argument("-prefix", required=True,
                      help="Output prefix. Produces {prefix}.nii.gz (warped image) "
                           "and {prefix}_WARP*.nii.gz (displacement fields)")
    g_io.add_argument("-save_warp", action="store_true", default=True,
                      help="Save displacement warp field(s) [default: yes]")
    g_io.add_argument("-no_save_warp", action="store_true",
                      help="Do not save displacement warp field(s)")
    g_io.add_argument("-verb", type=int, default=1, metavar="LEVEL",
                      help="Verbosity: 0=quiet, 1=normal, 2=debug [default: %(default)s]")

    # ── Timeseries Options ──────────────────────────────────────────────
    g_ts = p.add_argument_group(
        "Timeseries Options (only used when -source is omitted)",
        "Control how volumes are selected, how the base is constructed,\n"
        "and post-hoc temporal smoothing of warp fields.",
    )
    g_ts.add_argument("-tprange", type=str, default=None, metavar="START,STOP[,STEP]",
                      help="Select a subset of timepoints before processing. "
                           "Uses Python slice syntax: '0,10' = first 10 vols, "
                           "'0,-1,5' = every 5th vol, '100,200' = vols 100-199")
    g_ts.add_argument("-base_method", choices=["first", "mean", "median"], default="first",
                      help="How to construct the reference base volume. "
                           "'first' uses vol[0] as-is (only vol[1:] are warped). "
                           "'mean'/'median' computes the base from the first N vols "
                           "(see -base_navg), and ALL volumes are then warped to it "
                           "[default: %(default)s]")
    g_ts.add_argument("-base_navg", type=int, default=0, metavar="N",
                      help="Number of volumes to average for mean/median base. "
                           "0 = use all volumes. E.g. -base_navg 10 uses vol[0..9] "
                           "[default: %(default)s]")
    g_ts.add_argument("-chainwarp", action="store_true",
                      help="Initialize each volume's warp from the previous volume's "
                           "result. Useful when consecutive volumes are similar (e.g. "
                           "fMRI timeseries) -- can speed convergence and improve "
                           "temporal consistency")
    g_ts.add_argument("-chain_inilev", type=int, default=0, metavar="N",
                      help="When using -chainwarp, skip to level N for chained volumes "
                           "(2nd onward). The first volume gets the full multi-level "
                           "treatment; subsequent volumes start at level N since the "
                           "chained warp already provides coarse structure. E.g. "
                           "-chain_inilev 5 skips levels 0-4 for chained volumes. "
                           "0 = no skipping (default) [default: %(default)s]")
    g_ts.add_argument("-lookback", type=int, default=1, metavar="N",
                      help="When using -chainwarp, choose the best initialization warp "
                           "from the previous N volumes (based on which minimizes initial "
                           "cost). Useful when distortion is rhythmic and the immediately "
                           "preceding warp isn't the best starting point. Requires keeping "
                           "N warps in CPU RAM. 1 = use only the previous volume's warp "
                           "(classic chainwarp behavior) [default: %(default)s]")
    g_ts.add_argument("-tsmooth", type=float, default=0.0, metavar="SIGMA",
                      help="Temporal Gaussian smoothing of warp fields AFTER registration. "
                           "SIGMA is the width of the Gaussian kernel in units of volumes "
                           "(TRs). E.g. -tsmooth 2.0 blends each warp with its neighbors "
                           "using a Gaussian with sigma=2 TRs (~6 TR effective window). "
                           "Saves BOTH original and smoothed outputs: "
                           "{prefix}.nii.gz (original), {prefix}_tsmooth.nii.gz (smoothed), "
                           "plus per-volume _WARP_tXXX and _WARPsmooth_tXXX warp files. "
                           "0 = off [default: %(default)s]")
    g_ts.add_argument("-n_pcs", type=int, default=0, metavar="N",
                      help="Extract N principal components from the 4D warp fields "
                           "for use as regressors of no interest. PCs are extracted "
                           "per active displacement axis (i.e. axes not disabled by "
                           "-noXdis/-noYdis/-noZdis). When multiple axes are active, "
                           "their warps are concatenated before PCA. Output is saved "
                           "as {prefix}_warps/{basename}_warpPCs.1D with N columns. "
                           "0 = off [default: %(default)s]")

    # ── Warp Initialization ─────────────────────────────────────────────
    g_init = p.add_argument_group(
        "Warp Initialization",
        "Start registration from an existing warp field.",
    )
    g_init.add_argument("-iniwarp", default=None, metavar="WARP.nii.gz",
                        help="Load an initial warp displacement field and refine from there. "
                             "The warp should be on the unpadded source grid in mm units")

    # ── Registration Control ────────────────────────────────────────────
    g_reg = p.add_argument_group(
        "Registration Control",
        "Multi-level patch refinement parameters. The image is iteratively\n"
        "subdivided into smaller patches until minpatch is reached. Larger\n"
        "minpatch = smoother/faster, smaller = more detail/slower.",
    )
    g_reg.add_argument("-minpatch", type=int, default=25, metavar="SIZE",
                       help="Minimum patch size in voxels (must be odd, >= 5). "
                            "Smaller values allow finer warp detail but take longer. "
                            "For fMRI timeseries, 35-75 is often sufficient "
                            "[default: %(default)s]")
    g_reg.add_argument("-maxlev", type=int, default=666, metavar="N",
                       help="Maximum number of refinement levels. Caps how many times "
                            "patches are subdivided, even if minpatch has not been reached. "
                            "E.g. -maxlev 2 does at most the global warp + 2 subdivision "
                            "levels. 666 = no cap (auto from minpatch) [default: %(default)s]")
    g_reg.add_argument("-inilev", type=int, default=0, metavar="N",
                       help="Skip the first N refinement levels (start at coarser patches). "
                            "Useful to skip the expensive global warp if -iniwarp provides "
                            "a good starting point [default: %(default)s]")
    g_reg.add_argument("-workhard", nargs=2, type=int, default=None, metavar=("START", "END"),
                       help="Run extra optimization passes for levels START..END. "
                            "E.g. -workhard 0 2 works harder on the first 3 levels")
    g_reg.add_argument("-nopad", action="store_true",
                       help="Disable internal zero-padding of images. Padding adds ~12%% "
                            "border to allow warps near edges. Disabling saves memory but "
                            "may cause edge artifacts")

    # ── Basis Functions ─────────────────────────────────────────────────
    g_basis = p.add_argument_group(
        "Basis Functions",
        "Control the polynomial basis used for warp parameterization.",
    )
    g_basis.add_argument("-quintic", action="store_true",
                         help="Use quintic (5th order) Hermite basis at the final "
                              "refinement level. More flexible but slower [default: cubic]")
    g_basis.add_argument("-nolite", action="store_true",
                         help="Use full (non-lite) basis function sets. Lite uses fewer "
                              "parameters per patch for speed; nolite uses all cross-terms")

    # ── Displacement Constraints ────────────────────────────────────────
    g_disp = p.add_argument_group(
        "Displacement Constraints",
        "Restrict which axes can warp and set displacement limits.\n"
        "For EPI distortion correction, typically only one axis has\n"
        "distortion (e.g. -noYdis -noZdis for AP phase-encode).",
    )
    g_disp.add_argument("-noXdis", action="store_true",
                        help="Disable displacement along X (left-right)")
    g_disp.add_argument("-noYdis", action="store_true",
                        help="Disable displacement along Y (anterior-posterior)")
    g_disp.add_argument("-noZdis", action="store_true",
                        help="Disable displacement along Z (inferior-superior)")
    g_disp.add_argument("-axweight", nargs=3, type=float, default=None,
                        metavar=("WX", "WY", "WZ"),
                        help="Per-axis displacement weights (0.0-1.0). Soft version of "
                             "-noXdis etc. E.g. -axweight 1.0 0.2 0.0 allows full X, "
                             "reduced Y, no Z")
    g_disp.add_argument("-maxdisp", type=float, default=0.0, metavar="VOXELS",
                        help="Hard limit on maximum displacement in voxels. "
                             "0 = no limit [default: %(default)s]")
    g_disp.add_argument("-motparams", type=str, default=None, metavar="FILE.1D",
                        help="Motion parameters from 3dvolreg (-1Dfile output). "
                             "6 columns: roll pitch yaw dS dL dP (degrees, mm). "
                             "Used with -noXdis/-noYdis/-noZdis to dynamically adjust "
                             "per-axis displacement weights based on head rotation. "
                             "The disabled axes define the phase-encode direction; as "
                             "the head rotates, distortion projects onto other axes. "
                             "E.g. with -noXdis -noZdis (PE along Y/AP), a pitch rotation "
                             "projects some distortion onto Z. Requires timeseries mode")

    # ── Cost Function ───────────────────────────────────────────────────
    g_cost = p.add_argument_group(
        "Cost Function",
        "Similarity metric for comparing base and warped source.\n"
        "Pearson correlation variants work well for same-modality images.",
    )
    g_cost.add_argument("-pcl", action="store_true", default=True,
                        help="Clipped Pearson correlation: standard Pearson with negative "
                             "values clipped to zero. Good default for most cases")
    g_cost.add_argument("-pear", action="store_true",
                        help="Plain (unclipped) Pearson correlation")
    g_cost.add_argument("-lpa", action="store_true",
                        help="Local Pearson Absolute: computes Pearson in local Gaussian "
                             "neighborhoods, applies Fisher Z-transform (atanh), and uses "
                             "z*|z| weighting. Produces larger warps than pearclp but can "
                             "capture finer local structure. Slower (~3x)")
    g_cost.add_argument("-lpa_sigma", type=float, default=4.0, metavar="VOXELS",
                        help="Gaussian sigma for LPA local neighborhoods. Controls the "
                             "spatial scale of local correlation. Effective window radius "
                             "is ~3x sigma. Larger = smoother cost, smaller = more local "
                             "[default: %(default)s voxels]")
    g_cost.add_argument("-penfac", type=float, default=0.001, metavar="FACTOR",
                        help="Warp distortion penalty factor (Jacobian-based). Prevents "
                             "excessive warp folding. Our Adam optimizer amplifies penalty "
                             "gradients vs AFNI's NEWUOA, so this is much lower than AFNI's "
                             "default of 0.033 [default: %(default)s]")

    # ── Smoothing ───────────────────────────────────────────────────────
    g_blur = p.add_argument_group(
        "Image Smoothing",
        "Pre-blur base and/or source before registration.",
    )
    g_blur.add_argument("-blur", nargs=2, type=float, default=None,
                        metavar=("BASE_FWHM", "SRC_FWHM"),
                        help="Gaussian blur FWHM in mm applied to base and source. "
                             "E.g. -blur 2.0 2.0")
    g_blur.add_argument("-pblur", nargs=2, type=float, default=None,
                        metavar=("BASE_FRAC", "SRC_FRAC"),
                        help="Progressive blur: fraction of patch size used as blur FWHM "
                             "at each level. Blur decreases as patches get smaller")

    # ── Weight Image ────────────────────────────────────────────────────
    g_wt = p.add_argument_group(
        "Weight Image",
        "Control which voxels contribute to the cost function.",
    )
    g_wt.add_argument("-useweight", type=str, default=None, metavar="WEIGHT.nii.gz",
                      help="3D weight image (same grid as base). Voxels with weight=0 "
                           "are ignored. If omitted, a weight mask is auto-generated from "
                           "the base image (nonzero voxels)")

    # ── Optimizer Tuning ────────────────────────────────────────────────
    g_opt = p.add_argument_group(
        "Optimizer Tuning",
        "Advanced: control the batched Adam optimizer and warp parameter\n"
        "scaling. Defaults are tuned for typical neuroimaging data.",
    )
    g_opt.add_argument("-batch_lr", type=float, default=0.008, metavar="LR",
                       help="Learning rate for the batched Adam optimizer. Higher = faster "
                            "convergence but may overshoot [default: %(default)s]")
    g_opt.add_argument("-batch_iters", type=int, default=60, metavar="N",
                       help="Maximum optimizer iterations per checkerboard phase. "
                            "More iters = better convergence per phase but slower "
                            "[default: %(default)s]")
    g_opt.add_argument("-level_stop", type=float, default=0.0, metavar="TOL",
                       help="Early stopping: if a refinement level improves cost by less "
                            "than TOL fraction, skip all finer levels. E.g. -level_stop 0.0001 "
                            "stops when improvement drops below 0.01%% of current cost. "
                            "0 = disabled [default: %(default)s]")
    g_opt.add_argument("-hfactor_q", type=float, default=1.0, metavar="Q",
                       help="Hfactor parameter scaling for large patches. Controls how "
                            "much to reduce max displacement for patches larger than "
                            "minpatch. 1.0 = no scaling (AFNI default), 0.5 = moderate "
                            "reduction. Range: 0.1-1.0 [default: %(default)s]")

    # ── GPU / Hardware ──────────────────────────────────────────────────
    g_hw = p.add_argument_group(
        "GPU / Hardware",
        "Device selection and memory management.",
    )
    g_hw.add_argument("-device", type=str, default=None, metavar="DEV",
                      help="PyTorch device string. E.g. 'cuda', 'cuda:1', 'cpu'. "
                           "Auto-detected if omitted (prefers CUDA)")
    g_hw.add_argument("-gpu_mem", type=float, default=15.0, metavar="GB",
                      help="Available GPU memory in GB. Used for batch size estimation "
                           "[default: %(default)s]")
    g_hw.add_argument("-compile", action="store_true",
                      help="Enable torch.compile for building-block functions (CUDA). "
                           "Adds warmup overhead on first volume but may speed up "
                           "large datasets. Off by default")
    g_hw.add_argument("-memcheck", action="store_true",
                      help="Print GPU memory estimate for the input data and exit "
                           "without running registration")

    return p.parse_args(argv)


def _zeropad_width(n: int) -> int:
    """Minimum zero-pad width to represent indices up to n-1."""
    if n <= 0:
        return 3
    return max(3, int(math.log10(n)) + 1)


def _read_motion_params(filepath: str) -> list[tuple[float, float, float, float, float, float]]:
    """Read a 3dvolreg -1Dfile motion parameter file.

    Format: one row per volume, 6 columns:
        roll  pitch  yaw  dS  dL  dP
    Rotations in degrees, translations in mm.
    Lines starting with '#' are comments. Blank lines are skipped.

    Returns:
        List of (roll, pitch, yaw, dS, dL, dP) tuples.
    """
    params = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            if len(vals) < 6:
                continue
            params.append(tuple(float(v) for v in vals[:6]))
    return params


def _pe_direction_from_flags(noXdis: bool, noYdis: bool, noZdis: bool) -> tuple[float, float, float]:
    """Determine phase-encode direction unit vector from displacement flags.

    The PE direction is the axis that IS enabled (not disabled).
    E.g. -noXdis -noZdis means PE is along Y: (0, 1, 0).
    """
    pe = [1.0, 1.0, 1.0]
    if noXdis:
        pe[0] = 0.0
    if noYdis:
        pe[1] = 0.0
    if noZdis:
        pe[2] = 0.0
    # Normalize (should already be a unit vector if exactly 1 axis is enabled)
    mag = math.sqrt(sum(v * v for v in pe))
    if mag < 1e-10:
        return (0.0, 0.0, 0.0)
    return (pe[0] / mag, pe[1] / mag, pe[2] / mag)


def _rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> list[list[float]]:
    """Build 3x3 rotation matrix from 3dvolreg parameters.

    Convention (from 3dvolreg.c):
        roll  = rotation about I-S axis (Z) — shaking head 'no'
        pitch = rotation about R-L axis (X) — nodding 'yes'
        yaw   = rotation about A-P axis (Y) — ear to shoulder

    Returns R = Rz(roll) @ Rx(pitch) @ Ry(yaw), matching AFNI's DICOM order.
    """
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    # Rz(roll)
    rz = [[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]
    # Rx(pitch)
    rx = [[1, 0, 0], [0, cp, -sp], [0, sp, cp]]
    # Ry(yaw)
    ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]

    # R = Rz @ Rx @ Ry
    def matmul(a, b):
        return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    return matmul(matmul(rz, rx), ry)


def _compute_axis_weights_from_motion(
    roll_deg: float, pitch_deg: float, yaw_deg: float,
    pe_dir: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Compute per-axis displacement weights from rotation + PE direction.

    After rigid-body motion correction, the distortion that was along the PE
    direction gets projected by the rotation matrix. The component magnitudes
    of the rotated PE vector give the appropriate axis weights.

    Args:
        roll_deg, pitch_deg, yaw_deg: Rotation from 3dvolreg (degrees).
        pe_dir: Unit vector of the phase-encode direction at rest.

    Returns:
        (wx, wy, wz) axis weights in [0, 1].
    """
    R = _rotation_matrix(roll_deg, pitch_deg, yaw_deg)

    # Rotated PE direction = R @ pe_dir
    rx = sum(R[0][j] * pe_dir[j] for j in range(3))
    ry = sum(R[1][j] * pe_dir[j] for j in range(3))
    rz = sum(R[2][j] * pe_dir[j] for j in range(3))

    return (abs(rx), abs(ry), abs(rz))


def _build_timeseries_base(
    data_4d: Tensor, method: str, navg: int, verb: int,
) -> Tensor:
    """Construct a 3D base volume from a 4D timeseries.

    Args:
        data_4d: (nt, nz, ny, nx) full timeseries.
        method: 'first', 'mean', or 'median'.
        navg: Number of volumes to use (0=all).
        verb: Verbosity level.

    Returns:
        (nz, ny, nx) base volume.
    """
    nt = data_4d.shape[0]

    if method == "first":
        if verb >= 1:
            print("  Base: vol[0]")
        return data_4d[0].clone()

    n = min(navg, nt) if navg > 0 else nt
    subset = data_4d[:n]

    if method == "mean":
        if verb >= 1:
            print(f"  Base: mean of vol[0]..vol[{n-1}] ({n} volumes)")
        return subset.float().mean(dim=0)

    if method == "median":
        if verb >= 1:
            print(f"  Base: median of vol[0]..vol[{n-1}] ({n} volumes)")
        return subset.float().median(dim=0).values

    raise ValueError(f"Unknown base method: {method}")


def _temporal_smooth_warps(
    warp_list: list[tuple[Tensor, Tensor, Tensor]],
    sigma: float,
    verb: int,
) -> list[tuple[Tensor, Tensor, Tensor]]:
    """Apply Gaussian temporal smoothing to a list of warp fields.

    Args:
        warp_list: List of (xd, yd, zd) tuples, each (nz, ny, nx) on padded grid.
        sigma: Gaussian sigma in volumes (temporal units).
        verb: Verbosity level.

    Returns:
        List of temporally smoothed (xd, yd, zd) tuples.
    """
    T = len(warp_list)
    if T < 3 or sigma <= 0:
        return warp_list

    if verb >= 1:
        print(f"Temporal smoothing: sigma={sigma:.1f} volumes, {T} warp fields")

    # Build 1D Gaussian kernel
    radius = int(3.0 * sigma + 0.5)
    radius = min(radius, T - 1)  # can't be wider than the data
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    klen = kernel.shape[0]

    # Stack into (T, nz, ny, nx) per component
    xd_stack = torch.stack([w[0] for w in warp_list])  # (T, nz, ny, nx)
    yd_stack = torch.stack([w[1] for w in warp_list])
    zd_stack = torch.stack([w[2] for w in warp_list])

    smoothed = []
    for stack in [xd_stack, yd_stack, zd_stack]:
        # Reshape to (N, 1, T) where N = nz*ny*nx for 1D conv along T
        shape = stack.shape  # (T, nz, ny, nx)
        flat = stack.permute(1, 2, 3, 0).reshape(-1, 1, T)  # (N, 1, T)

        # Pad temporally with replicate (edge values)
        padded = F.pad(flat, (radius, radius), mode='replicate')

        # Convolve
        k = kernel.reshape(1, 1, klen)
        out = F.conv1d(padded, k)  # (N, 1, T)

        # Reshape back
        out = out.reshape(shape[1], shape[2], shape[3], T).permute(3, 0, 1, 2)
        smoothed.append(out)

    result = []
    for t in range(T):
        result.append((smoothed[0][t], smoothed[1][t], smoothed[2][t]))

    if verb >= 1:
        # Report how much smoothing changed things
        delta = 0.0
        for t in range(T):
            for c in range(3):
                orig = warp_list[t][c]
                smth = result[t][c]
                delta += (orig - smth).abs().mean().item()
        delta /= (T * 3)
        print(f"  Mean warp change from smoothing: {delta:.4f} voxels")

    return result


def _apply_warp_to_volume(
    source: Tensor, xd: Tensor, yd: Tensor, zd: Tensor,
    pad_x: int, pad_y: int, pad_z: int,
    device: torch.device,
) -> Tensor:
    """Apply a padded-grid warp to a source volume, returning cropped result.

    Args:
        source: (nz, ny, nx) original unpadded source volume (CPU).
        xd, yd, zd: (nz_pad, ny_pad, nx_pad) displacement fields in voxels (CPU).
        pad_x, pad_y, pad_z: Padding amounts.
        device: GPU device for computation.

    Returns:
        (nz, ny, nx) warped and cropped result (CPU).
    """
    nz, ny, nx = source.shape
    source_p = _pad_volume(source.float().to(device), pad_x, pad_y, pad_z)
    xd_gpu = xd.float().to(device)
    yd_gpu = yd.float().to(device)
    zd_gpu = zd.float().to(device)
    warped_full = warp_image_linear(source_p, xd_gpu, yd_gpu, zd_gpu)
    warped = warped_full[pad_z:pad_z+nz, pad_y:pad_y+ny, pad_x:pad_x+nx]
    result = warped.cpu()
    del source_p, xd_gpu, yd_gpu, zd_gpu, warped_full, warped
    return result


def _extract_warp_pcs(
    all_warps_raw: list[tuple[Tensor, Tensor, Tensor]],
    n_pcs: int,
    do_x: bool, do_y: bool, do_z: bool,
    pad_x: int, pad_y: int, pad_z: int,
    nx: int, ny: int, nz: int,
    out_path: str,
    verb: int = 1,
) -> None:
    """Extract temporal PCs from 4D warp fields and save as .1D regressor file.

    Concatenates active axes' 3D warps (unpadded) into a (n_vols, n_voxels)
    matrix, runs PCA to get temporal PCs, and writes them as columns.
    Uses the project's PCA class (covariance-trick SVD for efficiency).
    """
    from ..pca import PCA

    n_vols = len(all_warps_raw)
    if n_vols < 3:
        if verb >= 1:
            print(f"WARNING: Only {n_vols} volumes, need >= 3 for warp PCA — skipping")
        return

    n_pcs = min(n_pcs, n_vols - 1)

    # Collect active axis warps, cropping padding
    axis_labels = []
    vol_vecs = []
    for t in range(n_vols):
        xd, yd, zd = all_warps_raw[t]
        parts = []
        if do_x:
            crop = xd[pad_z:pad_z+nz, pad_y:pad_y+ny, pad_x:pad_x+nx] if (pad_x or pad_y or pad_z) else xd
            parts.append(crop.reshape(-1))
            if t == 0:
                axis_labels.append("X")
        if do_y:
            crop = yd[pad_z:pad_z+nz, pad_y:pad_y+ny, pad_x:pad_x+nx] if (pad_x or pad_y or pad_z) else yd
            parts.append(crop.reshape(-1))
            if t == 0:
                axis_labels.append("Y")
        if do_z:
            crop = zd[pad_z:pad_z+nz, pad_y:pad_y+ny, pad_x:pad_x+nx] if (pad_x or pad_y or pad_z) else zd
            parts.append(crop.reshape(-1))
            if t == 0:
                axis_labels.append("Z")
        vol_vecs.append(torch.cat(parts))

    if not axis_labels:
        if verb >= 1:
            print("WARNING: No active displacement axes for warp PCA — skipping")
        return

    # (n_vols, n_voxels_concat) matrix — PCA class handles centering + covariance trick
    mat = torch.stack(vol_vecs).float()
    del vol_vecs

    pca = PCA(n_components=n_pcs)
    scores = pca.fit_transform(mat)  # (n_vols, n_pcs)

    # Normalize scores to unit variance for use as regressors
    sc_std = scores.std(dim=0, keepdim=True).clamp(min=1e-10)
    pcs = scores / sc_std

    var_explained = pca.explained_variance_ratio_[:n_pcs]

    if verb >= 1:
        axes_str = "+".join(axis_labels)
        var_pct = [f"{v*100:.1f}%" for v in var_explained.tolist()]
        print(f"Warp PCs ({axes_str}, {n_vols} vols): extracted {n_pcs} PCs, "
              f"var explained: {', '.join(var_pct)}")

    # Write .1D file (AFNI-style: space-separated columns, one row per volume)
    with open(out_path, "w") as f:
        f.write("# Warp displacement PCs from ffs_qwarp\n")
        f.write(f"# Active axes: {'+'.join(axis_labels)}, {n_vols} volumes, {n_pcs} PCs\n")
        f.write(f"# Variance explained: {' '.join(f'{v*100:.2f}%' for v in var_explained.tolist())}\n")
        for row in pcs.cpu().numpy():
            f.write("  ".join(f"{v: .6f}" for v in row) + "\n")

    if verb >= 1:
        print(f"Saved: {out_path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Select device (prefer CUDA > MPS > CPU)
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        if args.verb >= 1:
            print("WARNING: No GPU available, running on CPU (will be slow)")

    if args.verb >= 1:
        print(f"qwarp_torch: device={device}")

    # Load images
    t0 = time.time()
    base_data, base_info = load_image(args.base, device=torch.device("cpu"))

    # Determine mode: timeseries (single 4D) vs standard (separate base+source)
    timeseries_mode = args.source is None

    if timeseries_mode:
        if base_data.ndim != 4:
            print("ERROR: -base must be 4D when -source is omitted")
            return 1
        nt_orig = base_data.shape[0]

        # Apply timepoint range selection
        if args.tprange is not None:
            parts = [int(x) for x in args.tprange.split(",")]
            if len(parts) == 1:
                sl = slice(0, parts[0])
            elif len(parts) == 2:
                sl = slice(parts[0], parts[1] if parts[1] != -1 else None)
            elif len(parts) == 3:
                stop = parts[1] if parts[1] != -1 else None
                sl = slice(parts[0], stop, parts[2])
            else:
                print("ERROR: -tprange expects 1-3 comma-separated ints (start,stop[,step])")
                return 1
            base_data = base_data[sl]
            if args.verb >= 1:
                print(f"Timepoint selection: {nt_orig} -> {base_data.shape[0]} volumes (slice {sl})")

        nt_full, nz, ny, nx = base_data.shape
        if nt_full < 2:
            print("ERROR: 4D -base must have at least 2 volumes after -tprange")
            return 1

        if args.verb >= 1:
            print(f"Timeseries mode: {nt_full} volumes, {nx}x{ny}x{nz}")

        # Build base from timeseries
        base_3d = _build_timeseries_base(
            base_data, args.base_method, args.base_navg, args.verb,
        )

        # Determine which volumes to warp
        if args.base_method == "first":
            # vol[0] is the base, warp vol[1:]
            source_indices = list(range(1, nt_full))
        else:
            # mean/median base: warp ALL volumes (including vol[0])
            source_indices = list(range(0, nt_full))

        nt = len(source_indices)
        if args.verb >= 1:
            if args.base_method == "first":
                print(f"  Warping vol[1]..vol[{nt_full-1}] -> base ({nt} volumes)")
            else:
                print(f"  Warping all {nt} volumes -> {args.base_method} base")
            if args.tsmooth > 0:
                print(f"  Temporal warp smoothing: sigma={args.tsmooth:.1f} volumes")
    else:
        source_data, source_info = load_image(args.source, device=torch.device("cpu"))

        # Extract base volume from 4D if needed
        if base_data.ndim == 4:
            if args.base_index is not None:
                if args.base_index < 0 or args.base_index >= base_data.shape[0]:
                    print(f"ERROR: -base_index {args.base_index} out of range "
                          f"(0..{base_data.shape[0]-1})")
                    return 1
                base_3d = base_data[args.base_index]
                if args.verb >= 1:
                    print(f"Using base vol[{args.base_index}] from 4D ({base_data.shape[0]} volumes)")
            else:
                base_3d = base_data[0]
                if args.verb >= 1:
                    print("WARNING: 4D base with -source; using vol[0]. "
                          "Use -base_index to select a specific volume")
        else:
            base_3d = base_data

        if source_data.ndim == 4:
            nt, nz, ny, nx = source_data.shape
        else:
            nz, ny, nx = base_3d.shape
            nt = 1

    if args.verb >= 1:
        if not timeseries_mode:
            print(f"Base: {nx}x{ny}x{nz}, Source: {nx}x{ny}x{nz} x {nt}t")
        print(f"Loaded in {time.time()-t0:.1f}s")

    # Memory check
    if args.memcheck:
        print_memory_report(nx, ny, nz, nt)
        return 0

    mem_gb = estimate_gpu_memory_gb(nx, ny, nz)
    if args.verb >= 1:
        print(f"Estimated GPU memory per volume: {mem_gb:.2f} GB")

    if mem_gb > args.gpu_mem * 0.95:
        print(f"WARNING: Estimated {mem_gb:.1f} GB may exceed "
              f"available {args.gpu_mem:.1f} GB")

    # Build config
    warp_flags = 0
    if args.noXdis:
        warp_flags |= 1
    if args.noYdis:
        warp_flags |= 2
    if args.noZdis:
        warp_flags |= 4

    axis_weights = (1.0, 1.0, 1.0)
    if args.axweight is not None:
        axis_weights = tuple(max(0.0, min(1.0, w)) for w in args.axweight)

    config = QwarpConfig(
        minpatch=args.minpatch,
        max_level=args.maxlev,
        start_level=args.inilev,
        use_quintic=args.quintic,
        use_lite=not args.nolite,
        workhard=tuple(args.workhard) if args.workhard else (0, -1),
        cost_method="lpa" if args.lpa else ("pearson" if args.pear else "pearclp"),
        penalty_factor=args.penfac,
        warp_flags=warp_flags,
        axis_weights=axis_weights,
        verb=args.verb,
        batch_optimizer_lr=args.batch_lr,
        batch_optimizer_iters=args.batch_iters,
        hfactor_q=args.hfactor_q,
        maxdisp=args.maxdisp,
        lpa_sigma=args.lpa_sigma,
        level_stop_tol=args.level_stop,
        compile=args.compile,
    )

    if args.blur is not None:
        config.blur_base = args.blur[0]
        config.blur_source = args.blur[1]
    if args.pblur is not None:
        config.pblur_base = args.pblur[0]
        config.pblur_source = args.pblur[1]

    # Load motion parameters for dynamic axis weighting
    motion_params = None
    pe_direction = None
    if args.motparams is not None:
        if not timeseries_mode:
            print("ERROR: -motparams requires timeseries mode (omit -source)")
            return 1
        motion_params = _read_motion_params(args.motparams)
        pe_direction = _pe_direction_from_flags(args.noXdis, args.noYdis, args.noZdis)
        if sum(pe_direction) < 0.5:
            print("ERROR: -motparams requires -noXdis/-noYdis/-noZdis to define PE direction")
            return 1
        if args.verb >= 1:
            pe_labels = []
            if pe_direction[0] > 0.5:
                pe_labels.append("X(RL)")
            if pe_direction[1] > 0.5:
                pe_labels.append("Y(AP)")
            if pe_direction[2] > 0.5:
                pe_labels.append("Z(IS)")
            print(f"Motion-projected distortion: PE along {'+'.join(pe_labels)}, "
                  f"{len(motion_params)} motion parameter rows")
        # When using motparams, clear warp_flags -- axis_weights will do the work
        config = replace(config, warp_flags=0)

    # Load or compute weight
    weight = None
    if args.useweight is not None:
        weight, _ = load_image(args.useweight, device=torch.device("cpu"))

    # Load initial warp if provided
    initial_warp = None
    if args.iniwarp is not None:
        ini_xd, ini_yd, ini_zd, _ = load_warp_field(args.iniwarp, device=torch.device("cpu"))
        initial_warp = (ini_xd, ini_yd, ini_zd)
        if args.verb >= 1:
            print(f"Loaded initial warp: {args.iniwarp}")

    # Output prefix (strip extension)
    prefix = args.prefix
    if prefix.endswith(".nii.gz"):
        prefix = prefix[:-7]
    elif prefix.endswith(".nii"):
        prefix = prefix[:-4]

    # Compute padding for warp field header
    use_pad = not args.nopad
    if use_pad:
        warp_padding = _compute_padding(nx, ny, nz)
        pad_x, pad_y, pad_z = warp_padding
    else:
        warp_padding = None
        pad_x, pad_y, pad_z = 0, 0, 0

    # --- Process volumes ---
    # Enable TF32 matmul precision on Ampere+ GPUs (free perf, ~1e-5 precision)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    all_warped = []

    if timeseries_mode:
        # For 'first' base method: base = vol[0], prepend unwarped base to output
        if args.base_method == "first":
            all_warped.append(base_3d.clone())

        zpad = _zeropad_width(nt_full)
        do_tsmooth = args.tsmooth > 0
        do_warp_pcs = args.n_pcs > 0

        # Create warp output directory
        warp_dir = f"{prefix}_warps"
        if not args.no_save_warp or do_warp_pcs:
            os.makedirs(warp_dir, exist_ok=True)
            if args.verb >= 1:
                print(f"Warp output directory: {warp_dir}/")
        # Base name for warp files inside the directory
        warp_basename = os.path.basename(prefix)

        # Keep raw voxel-unit warps in CPU RAM when temporal smoothing is needed
        all_warps_raw: list[tuple[Tensor, Tensor, Tensor]] = []

        # Track previous warps for chaining (-chainwarp / -lookback)
        chain_warps = args.chainwarp
        lookback_n = max(1, args.lookback) if chain_warps else 1
        # Ring buffer of (unpadded_warp, source_index) for lookback
        warp_buffer: deque[tuple[tuple[Tensor, Tensor, Tensor], int]] = deque(
            maxlen=lookback_n,
        )
        # Seed with initial_warp if provided
        if initial_warp is not None and chain_warps:
            warp_buffer.append((initial_warp, -1))

        if chain_warps and args.verb >= 1:
            chain_msg = "Chaining warps: each volume initialized from previous result"
            if lookback_n > 1:
                chain_msg += f", lookback={lookback_n}"
            if args.chain_inilev > 0:
                chain_msg += f", skipping to level {args.chain_inilev} for chained vols"
            print(chain_msg)

        for i, src_idx in enumerate(source_indices):
            src_vol = base_data[src_idx]

            # Per-volume config: chain_inilev and/or motion-projected axis weights
            vol_config = config
            vol_overrides = {}
            if chain_warps and i > 0 and args.chain_inilev > 0:
                vol_overrides["start_level"] = args.chain_inilev
            if motion_params is not None and pe_direction is not None:
                if src_idx < len(motion_params):
                    roll, pitch, yaw = motion_params[src_idx][:3]
                    aw = _compute_axis_weights_from_motion(roll, pitch, yaw, pe_direction)
                    vol_overrides["axis_weights"] = aw
                    if args.verb >= 2:
                        print(f"  Motion-projected weights: "
                              f"X={aw[0]:.4f} Y={aw[1]:.4f} Z={aw[2]:.4f} "
                              f"(roll={roll:.2f} pitch={pitch:.2f} yaw={yaw:.2f})")
            if vol_overrides:
                vol_config = replace(config, **vol_overrides)

            # Choose best init warp from lookback buffer
            chosen_warp = None
            chosen_label = ""
            if chain_warps and len(warp_buffer) > 0:
                if len(warp_buffer) == 1:
                    chosen_warp, chosen_src = warp_buffer[-1]
                    chosen_label = f"vol[{chosen_src}]" if chosen_src >= 0 else "iniwarp"
                else:
                    # Evaluate each candidate: compute pearson corr with base
                    # after applying the candidate warp to this source volume
                    best_cost = float("inf")
                    base_gpu = base_3d.float().to(device)
                    src_gpu = src_vol.float().to(device)
                    for cand_warp, cand_src in warp_buffer:
                        # Pad source and warp, apply, crop, compute cost
                        src_p = _pad_volume(src_gpu, pad_x, pad_y, pad_z)
                        cxd = cand_warp[0].float().to(device)
                        cyd = cand_warp[1].float().to(device)
                        czd = cand_warp[2].float().to(device)
                        # Pad the unpadded warp to padded grid for application
                        cxd_p = _pad_volume(cxd, pad_x, pad_y, pad_z)
                        cyd_p = _pad_volume(cyd, pad_x, pad_y, pad_z)
                        czd_p = _pad_volume(czd, pad_x, pad_y, pad_z)
                        warped_test = warp_image_linear(src_p, cxd_p, cyd_p, czd_p)
                        # Crop to unpadded and compute correlation with base
                        wt = warped_test[pad_z:pad_z+nz, pad_y:pad_y+ny, pad_x:pad_x+nx]
                        # Simple global Pearson correlation
                        bf = base_gpu.reshape(-1)
                        wf = wt.reshape(-1)
                        bm = bf - bf.mean()
                        wm = wf - wf.mean()
                        cost = -(bm * wm).sum() / ((bm * bm).sum() * (wm * wm).sum()).sqrt()
                        cost_val = cost.item()
                        del src_p, cxd, cyd, czd, cxd_p, cyd_p, czd_p, warped_test, wt
                        if cost_val < best_cost:
                            best_cost = cost_val
                            chosen_warp = cand_warp
                            chosen_src = cand_src
                    chosen_label = f"vol[{chosen_src}]" if chosen_src >= 0 else "iniwarp"
                    if args.verb >= 1:
                        print(f"  Lookback: best init from {chosen_label} "
                              f"(cost={best_cost:.5f}, {len(warp_buffer)} candidates)")
                    del base_gpu, src_gpu
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            if args.verb >= 1:
                init_str = ""
                if chain_warps and chosen_warp is not None:
                    init_str = f", init from {chosen_label} warp"
                    if args.chain_inilev > 0:
                        init_str += f", inilev={args.chain_inilev}"
                elif initial_warp is not None and i == 0:
                    init_str = ", init from -iniwarp"
                if motion_params is not None and "axis_weights" in vol_overrides:
                    aw = vol_overrides["axis_weights"]
                    init_str += f", axwt=[{aw[0]:.3f},{aw[1]:.3f},{aw[2]:.3f}]"
                print(f"\n=== Warping vol[{src_idx}] -> base ({i+1}/{nt}){init_str} ===")

            base_gpu = base_3d.float().to(device)
            src_gpu = src_vol.float().to(device)
            w_gpu = weight.float().to(device) if weight is not None else None

            warped, xd, yd, zd = qwarp(
                base_gpu, src_gpu, weight=w_gpu,
                initial_warp=chosen_warp if chain_warps else initial_warp,
                config=vol_config, device=device, pad=use_pad,
            )

            all_warped.append(warped.cpu())

            # Move warps to CPU
            xd_cpu, yd_cpu, zd_cpu = xd.cpu(), yd.cpu(), zd.cpu()

            # Save per-volume warp (original, unsmoothed)
            if not args.no_save_warp:
                save_warp_field(
                    xd_cpu, yd_cpu, zd_cpu,
                    os.path.join(warp_dir, f"{warp_basename}_WARP_t{src_idx:0{zpad}d}.nii.gz"),
                    header_info=base_info,
                    padding=warp_padding,
                    units="mm",
                )

            # Keep raw warps for temporal smoothing or PC extraction
            if do_tsmooth or do_warp_pcs:
                all_warps_raw.append((xd_cpu, yd_cpu, zd_cpu))

            # Chain: crop padded warp back to unpadded size, add to buffer
            if chain_warps:
                if pad_x > 0 or pad_y > 0 or pad_z > 0:
                    unpadded_warp = (
                        xd_cpu[pad_z:pad_z+nz, pad_y:pad_y+ny, pad_x:pad_x+nx],
                        yd_cpu[pad_z:pad_z+nz, pad_y:pad_y+ny, pad_x:pad_x+nx],
                        zd_cpu[pad_z:pad_z+nz, pad_y:pad_y+ny, pad_x:pad_x+nx],
                    )
                else:
                    unpadded_warp = (xd_cpu, yd_cpu, zd_cpu)
                warp_buffer.append((unpadded_warp, src_idx))
            elif not (do_tsmooth or do_warp_pcs):
                del xd_cpu, yd_cpu, zd_cpu

            # Free GPU tensors (caching allocator reuses memory for next volume)
            del warped, xd, yd, zd, base_gpu, src_gpu, w_gpu

        # Save original (unsmoothed) 4D warped timeseries
        warped_4d = torch.stack(all_warped, dim=0)
        save_image(warped_4d, f"{prefix}.nii.gz", header_info=base_info)
        if args.verb >= 1:
            print(f"\nSaved: {prefix}.nii.gz ({warped_4d.shape[0]} volumes)")

        # --- Temporal warp smoothing ---
        if do_tsmooth and all_warps_raw:
            smoothed_warps = _temporal_smooth_warps(
                all_warps_raw, args.tsmooth, args.verb,
            )

            # Save smoothed warps and recompute warped 4D
            all_warped_smooth = []
            if args.base_method == "first":
                all_warped_smooth.append(base_3d.clone())

            for i, src_idx in enumerate(source_indices):
                sxd, syd, szd = smoothed_warps[i]

                # Save smoothed warp
                if not args.no_save_warp:
                    save_warp_field(
                        sxd, syd, szd,
                        os.path.join(warp_dir, f"{warp_basename}_WARPsmooth_t{src_idx:0{zpad}d}.nii.gz"),
                        header_info=base_info,
                        padding=warp_padding,
                        units="mm",
                    )

                # Recompute warped volume from smoothed warp
                src_vol = base_data[src_idx]
                warped_smooth = _apply_warp_to_volume(
                    src_vol, sxd, syd, szd,
                    pad_x, pad_y, pad_z, device,
                )
                all_warped_smooth.append(warped_smooth)

                if device.type == "cuda":
                    torch.cuda.empty_cache()

            # Save temporally smoothed 4D
            warped_4d_smooth = torch.stack(all_warped_smooth, dim=0)
            save_image(warped_4d_smooth, f"{prefix}_tsmooth.nii.gz", header_info=base_info)
            if args.verb >= 1:
                print(f"Saved: {prefix}_tsmooth.nii.gz ({warped_4d_smooth.shape[0]} volumes, smoothed)")

        # --- Extract warp PCs as regressors of no interest ---
        if do_warp_pcs and all_warps_raw:
            _extract_warp_pcs(
                all_warps_raw, args.n_pcs,
                do_x=not args.noXdis, do_y=not args.noYdis, do_z=not args.noZdis,
                pad_x=pad_x, pad_y=pad_y, pad_z=pad_z,
                nx=nx, ny=ny, nz=nz,
                out_path=os.path.join(warp_dir, f"{warp_basename}_warpPCs.1D"),
                verb=args.verb,
            )

    else:
        # Standard mode: separate base and source
        all_xd, all_yd, all_zd = [], [], []

        for t in range(nt):
            if nt > 1:
                src_vol = source_data[t]
                if args.verb >= 1:
                    print(f"\n=== Timepoint {t+1}/{nt} ===")
            else:
                src_vol = source_data if source_data.ndim == 3 else source_data[0]

            base_gpu = base_3d.float().to(device)
            src_gpu = src_vol.float().to(device)
            w_gpu = weight.float().to(device) if weight is not None else None

            warped, xd, yd, zd = qwarp(
                base_gpu, src_gpu, weight=w_gpu,
                initial_warp=initial_warp,
                config=config, device=device, pad=use_pad,
            )

            all_warped.append(warped.cpu())
            all_xd.append(xd.cpu())
            all_yd.append(yd.cpu())
            all_zd.append(zd.cpu())

            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Save results
        if nt > 1:
            warped_4d = torch.stack(all_warped, dim=0)
            save_image(warped_4d, f"{prefix}.nii.gz", header_info=base_info)
            if not args.no_save_warp:
                zpad = _zeropad_width(nt)
                for t in range(nt):
                    save_warp_field(
                        all_xd[t], all_yd[t], all_zd[t],
                        f"{prefix}_WARP_t{t:0{zpad}d}.nii.gz",
                        header_info=base_info,
                        padding=warp_padding,
                        units="mm",
                    )
        else:
            save_image(all_warped[0], f"{prefix}.nii.gz", header_info=base_info)
            if not args.no_save_warp:
                save_warp_field(
                    all_xd[0], all_yd[0], all_zd[0],
                    f"{prefix}_WARP.nii.gz",
                    header_info=base_info,
                    padding=warp_padding,
                    units="mm",
                )

    elapsed = time.time() - t0
    if args.verb >= 1:
        print(f"\nDone. Total time: {elapsed:.1f}s")
        print(f"Output: {prefix}.nii.gz")

    return 0


if __name__ == "__main__":
    sys.exit(main())
