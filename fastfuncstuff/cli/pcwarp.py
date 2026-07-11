#!/usr/bin/env python3
"""Extract temporal PCs from a folder of ffs_qwarp warp displacement files.

Given a directory of per-volume warp NIfTI files (e.g. *_WARP_t0001.nii.gz),
this utility loads them, auto-detects which displacement axes are active
(non-zero), runs PCA, and writes an AFNI-style .1D regressor file.

This avoids re-running the full qwarp pipeline just to get warp PCs.

Examples
--------
  # Auto-detect axes, extract 5 PCs
  ffs_util_pcwarp -warp_dir sub01_warps -n_pcs 5

  # Specify glob pattern and output path
  ffs_util_pcwarp -warp_dir sub01_warps -pattern '*_WARP_t*.nii.gz' \\
                  -n_pcs 10 -output sub01_warpPCs.1D

  # Force specific axes (skip auto-detection)
  ffs_util_pcwarp -warp_dir sub01_warps -n_pcs 5 -axes Y

  # Use GPU for PCA
  ffs_util_pcwarp -warp_dir sub01_warps -n_pcs 5 -device cuda
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from fastfuncstuff.cli_utils import add_verbose_arg


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract temporal PCs from ffs_qwarp warp displacement files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  ffs_util_pcwarp -warp_dir sub01_warps -n_pcs 5\n"
            "  ffs_util_pcwarp -warp_dir sub01_warps -n_pcs 10 -output warpPCs.1D\n"
            "  ffs_util_pcwarp -warp_dir sub01_warps -n_pcs 5 -axes Y\n"
            "  ffs_util_pcwarp -warp_dir sub01_warps -n_pcs 5 -device cuda\n"
        ),
    )

    req = parser.add_argument_group("Required Arguments (give -warp_dir OR -warp)")
    req.add_argument(
        "-warp_dir",
        default=None,
        metavar="DIR",
        help="Directory of per-volume 4D warp files (glob -pattern).",
    )
    req.add_argument(
        "-warp",
        default=None,
        metavar="FILE",
        help="A single 5D (nx,ny,nz,T,3) warp file (e.g. ffs_locomoco/qwarp/medic "
        "output). Alternative to -warp_dir.",
    )
    req.add_argument(
        "-n_pcs",
        type=int,
        required=True,
        metavar="N",
        help="Number of principal components to extract.",
    )

    opt = parser.add_argument_group("Options")
    opt.add_argument(
        "-pattern",
        default="*_WARP_t*.nii.gz",
        metavar="GLOB",
        help="Glob pattern for warp files [default: %(default)s].",
    )
    opt.add_argument(
        "-prefix",
        dest="output",
        default=None,
        metavar="PATH",
        help="Output .1D file path. Default: {warp_dir}/warpPCs.1D",
    )
    opt.add_argument(
        "-output",
        dest="output",
        default=None,
        metavar="PATH",
        help="Alias for -prefix.",
    )
    opt.add_argument(
        "-axes",
        default=None,
        metavar="XYZ",
        help="Force active axes (e.g. 'Y', 'XY', 'XYZ'). "
        "By default, axes are auto-detected from the warp files by "
        "checking which displacement components are non-zero.",
    )
    opt.add_argument(
        "-device",
        default="cpu",
        metavar="DEV",
        help="Torch device for PCA computation [default: %(default)s].",
    )
    add_verbose_arg(opt, default=1)

    return parser


def _detect_active_axes(
    xd: torch.Tensor,
    yd: torch.Tensor,
    zd: torch.Tensor,
    verb: int,
) -> tuple[bool, bool, bool]:
    """Auto-detect which displacement axes are non-zero.

    ``xd/yd/zd`` are ``(T, nz, ny, nx)``. Skips frame 0 (volume 0 may be identity
    / base-to-base) and samples a few frames to see which axes carry displacement.
    """
    # Sample up to 3 non-first frames.
    hi = min(4, xd.shape[0])
    sl = slice(1, hi) if xd.shape[0] > 1 else slice(0, 1)
    has_x = xd[sl].abs().max().item() > 1e-8
    has_y = yd[sl].abs().max().item() > 1e-8
    has_z = zd[sl].abs().max().item() > 1e-8

    if verb >= 1:
        labels = []
        if has_x:
            labels.append("X")
        if has_y:
            labels.append("Y")
        if has_z:
            labels.append("Z")
        if labels:
            print(f"  Auto-detected active axes: {'+'.join(labels)}")
        else:
            print("  WARNING: All displacement axes appear to be zero!")

    return has_x, has_y, has_z


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()

    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)

    # --- Validate inputs: exactly one of -warp_dir / -warp ---
    if bool(args.warp_dir) == bool(args.warp):
        print("ERROR: give exactly one of -warp_dir (folder) or -warp (5D file).", file=sys.stderr)
        return 1
    source = args.warp_dir if args.warp_dir else args.warp
    is_dir = bool(args.warp_dir)
    if is_dir and not os.path.isdir(source):
        print(f"ERROR: warp directory does not exist: {source}", file=sys.stderr)
        return 1
    if not is_dir and not os.path.isfile(source):
        print(f"ERROR: warp file does not exist: {source}", file=sys.stderr)
        return 1

    n_pcs = args.n_pcs
    if n_pcs < 1:
        print("ERROR: -n_pcs must be >= 1", file=sys.stderr)
        return 1

    # --- Load the warp series (folder of 4D frames OR a single 5D file) ---
    from fastfuncstuff.decomposition.pca import PCA
    from fastfuncstuff.processing.io import load_warp_series

    try:
        xd, yd, zd, _hdr, n_vols = load_warp_series(source, pattern=args.pattern)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if n_vols < 3:
        print(f"ERROR: Only {n_vols} warp frames found — need at least 3 for PCA.", file=sys.stderr)
        return 1

    n_pcs = min(n_pcs, n_vols - 1)

    # --- Output path ---
    out_path = args.output
    if out_path is None:
        out_dir = source if is_dir else os.path.dirname(os.path.abspath(source))
        out_path = os.path.join(out_dir, "warpPCs.1D")

    # --- Banner ---
    if args.verb >= 1:
        print("=" * 70)
        print("ffs_util_pcwarp — Warp displacement PC extraction")
        print("=" * 70)
        print(f"  Warp source    : {os.path.abspath(source)} ({'folder' if is_dir else '5D file'})")
        if is_dir:
            print(f"  Pattern        : {args.pattern}")
        print(f"  Volumes found  : {n_vols}")
        print(f"  PCs requested  : {n_pcs}")
        print(f"  Output         : {out_path}")
        print(f"  Device         : {args.device}")

    # --- Determine active axes ---
    if args.axes is not None:
        ax = args.axes.upper()
        do_x, do_y, do_z = "X" in ax, "Y" in ax, "Z" in ax
        if args.verb >= 1:
            labels = [a for a, active in zip("XYZ", [do_x, do_y, do_z], strict=False) if active]
            print(f"  Forced axes    : {'+'.join(labels)}")
    else:
        if args.verb >= 1:
            print("  Detecting active axes...")
        do_x, do_y, do_z = _detect_active_axes(xd, yd, zd, args.verb)

    if not (do_x or do_y or do_z):
        print("ERROR: No active displacement axes — nothing to compute.", file=sys.stderr)
        return 1

    # --- Build the (n_vols, n_active_vox) matrix from the loaded series ---
    axis_labels = [a for a, on in zip("XYZ", (do_x, do_y, do_z), strict=False) if on]
    comps = [c for c, on in zip((xd, yd, zd), (do_x, do_y, do_z), strict=False) if on]
    # Each component is (T, nz, ny, nx) -> flatten spatial, concat active axes.
    mat = torch.cat([c.reshape(n_vols, -1) for c in comps], dim=1).float()
    del xd, yd, zd, comps

    # --- PCA ---
    if args.verb >= 1:
        print(f"  Matrix shape: ({n_vols}, {mat.shape[1]})  [{'+'.join(axis_labels)}]")
        print(f"  Running PCA for {n_pcs} components on {args.device}...")

    if args.device != "cpu":
        mat = mat.to(args.device)

    pca = PCA(n_components=n_pcs, device=args.device if args.device != "cpu" else None)
    scores = pca.fit_transform(mat)  # (n_vols, n_pcs)
    del mat

    # Normalize to unit variance for use as regressors
    sc_std = scores.std(dim=0, keepdim=True).clamp(min=1e-10)
    pcs = (scores / sc_std).cpu()

    var_explained = pca.explained_variance_ratio_[:n_pcs]

    if args.verb >= 1:
        var_pct = [f"{v * 100:.1f}%" for v in var_explained.tolist()]
        print(f"  Variance explained: {', '.join(var_pct)}")
        total = sum(var_explained.tolist()) * 100
        print(f"  Total variance captured: {total:.1f}%")

    # --- Write output ---
    pcs_np = pcs.numpy()
    with open(out_path, "w") as f:
        f.write("# Warp displacement PCs from ffs_util_pcwarp\n")
        f.write(f"# Source: {os.path.abspath(source)}\n")
        f.write(f"# Active axes: {'+'.join(axis_labels)}, {n_vols} volumes, {n_pcs} PCs\n")
        f.write(
            f"# Variance explained: {' '.join(f'{v * 100:.2f}%' for v in var_explained.tolist())}\n"
        )
        for row in pcs_np:
            f.write("  ".join(f"{v: .6f}" for v in row) + "\n")

    if args.verb >= 1:
        print(f"\n  Saved: {out_path}")
        print("=" * 70)

    return 0
