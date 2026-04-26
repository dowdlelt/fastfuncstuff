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
import glob
import os
import re
import sys

import torch

from fastfuncstuff.cli_utils import add_verbose_arg


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract temporal PCs from ffs_qwarp warp displacement files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
        epilog=(
            "Examples:\n"
            "  ffs_util_pcwarp -warp_dir sub01_warps -n_pcs 5\n"
            "  ffs_util_pcwarp -warp_dir sub01_warps -n_pcs 10 -output warpPCs.1D\n"
            "  ffs_util_pcwarp -warp_dir sub01_warps -n_pcs 5 -axes Y\n"
            "  ffs_util_pcwarp -warp_dir sub01_warps -n_pcs 5 -device cuda\n"
        ),
    )

    req = parser.add_argument_group("Required Arguments")
    req.add_argument(
        "-warp_dir", required=True, metavar="DIR",
        help="Directory containing per-volume warp files (*_WARP_t*.nii.gz).",
    )
    req.add_argument(
        "-n_pcs", type=int, required=True, metavar="N",
        help="Number of principal components to extract.",
    )

    opt = parser.add_argument_group("Options")
    opt.add_argument(
        "-pattern", default="*_WARP_t*.nii.gz", metavar="GLOB",
        help="Glob pattern for warp files [default: %(default)s].",
    )
    opt.add_argument(
        "-prefix", dest="output", default=None, metavar="PATH",
        help="Output .1D file path. Default: {warp_dir}/warpPCs.1D",
    )
    opt.add_argument(
        "-output", dest="output", default=None, metavar="PATH",
        help="Alias for -prefix.",
    )
    opt.add_argument(
        "-axes", default=None, metavar="XYZ",
        help="Force active axes (e.g. 'Y', 'XY', 'XYZ'). "
             "By default, axes are auto-detected from the warp files by "
             "checking which displacement components are non-zero.",
    )
    opt.add_argument(
        "-device", default="cpu", metavar="DEV",
        help="Torch device for PCA computation [default: %(default)s].",
    )
    add_verbose_arg(opt, default=1)

    hlp = parser.add_argument_group("Help")
    hlp.add_argument("-help", action="store_true", help="Show this help and exit.")

    return parser


def _natural_sort_key(s: str):
    """Sort key that handles embedded numbers naturally (t2 < t10)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def _detect_active_axes(
    warp_dir: str,
    warp_files: list[str],
    verb: int,
) -> tuple[bool, bool, bool]:
    """Auto-detect which displacement axes are non-zero.

    Skips the first file (volume 0 may be identity / base-to-base).
    Samples a few files to determine which axes carry displacement.
    """
    from fastfuncstuff.processing.io import load_warp_field

    # Sample up to 3 non-first files
    candidates = warp_files[1:4] if len(warp_files) > 1 else warp_files[:1]

    has_x, has_y, has_z = False, False, False
    for fname in candidates:
        path = os.path.join(warp_dir, fname)
        xd, yd, zd, _ = load_warp_field(path)
        if xd.abs().max().item() > 1e-8:
            has_x = True
        if yd.abs().max().item() > 1e-8:
            has_y = True
        if zd.abs().max().item() > 1e-8:
            has_z = True
        # Early exit if all active
        if has_x and has_y and has_z:
            break

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
    if not argv or "-help" in argv:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)

    # --- Validate inputs ---
    warp_dir = args.warp_dir
    if not os.path.isdir(warp_dir):
        print(f"ERROR: warp directory does not exist: {warp_dir}", file=sys.stderr)
        return 1

    n_pcs = args.n_pcs
    if n_pcs < 1:
        print("ERROR: -n_pcs must be >= 1", file=sys.stderr)
        return 1

    # --- Discover warp files ---
    matched = glob.glob(os.path.join(warp_dir, args.pattern))
    warp_files = sorted(
        [os.path.basename(f) for f in matched],
        key=_natural_sort_key,
    )
    n_vols = len(warp_files)

    if n_vols == 0:
        print(
            f"ERROR: No files matching '{args.pattern}' in {warp_dir}",
            file=sys.stderr,
        )
        return 1

    if n_vols < 3:
        print(
            f"ERROR: Only {n_vols} warp files found — need at least 3 for PCA.",
            file=sys.stderr,
        )
        return 1

    n_pcs = min(n_pcs, n_vols - 1)

    # --- Output path ---
    out_path = args.output
    if out_path is None:
        out_path = os.path.join(warp_dir, "warpPCs.1D")

    # --- Banner ---
    if args.verb >= 1:
        print("=" * 70)
        print("ffs_util_pcwarp — Warp displacement PC extraction")
        print("=" * 70)
        print(f"  Warp directory : {os.path.abspath(warp_dir)}")
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
            labels = [a for a, active in zip("XYZ", [do_x, do_y, do_z]) if active]
            print(f"  Forced axes    : {'+'.join(labels)}")
    else:
        if args.verb >= 1:
            print("  Detecting active axes...")
        do_x, do_y, do_z = _detect_active_axes(warp_dir, warp_files, args.verb)

    if not (do_x or do_y or do_z):
        print("ERROR: No active displacement axes — nothing to compute.", file=sys.stderr)
        return 1

    # --- Load warps and build matrix ---
    from fastfuncstuff.processing.io import load_warp_field
    from fastfuncstuff.decomposition.pca import PCA

    if args.verb >= 1:
        print(f"\n  Loading {n_vols} warp files...")

    axis_labels = []
    if do_x:
        axis_labels.append("X")
    if do_y:
        axis_labels.append("Y")
    if do_z:
        axis_labels.append("Z")

    vol_vecs: list[torch.Tensor] = []
    for i, fname in enumerate(warp_files):
        path = os.path.join(warp_dir, fname)
        xd, yd, zd, _ = load_warp_field(path)

        parts = []
        if do_x:
            parts.append(xd.reshape(-1))
        if do_y:
            parts.append(yd.reshape(-1))
        if do_z:
            parts.append(zd.reshape(-1))
        vol_vecs.append(torch.cat(parts))

        if args.verb >= 2 and (i + 1) % 50 == 0:
            print(f"    loaded {i + 1}/{n_vols}")

    if args.verb >= 1:
        print(f"  Loaded all {n_vols} volumes.")

    # --- PCA ---
    if args.verb >= 1:
        n_vox = vol_vecs[0].numel()
        print(f"  Matrix shape: ({n_vols}, {n_vox})  [{'+'.join(axis_labels)}]")
        print(f"  Running PCA for {n_pcs} components on {args.device}...")

    mat = torch.stack(vol_vecs).float()
    del vol_vecs

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
        f.write(f"# Source: {os.path.abspath(warp_dir)}\n")
        f.write(f"# Active axes: {'+'.join(axis_labels)}, {n_vols} volumes, {n_pcs} PCs\n")
        f.write(
            f"# Variance explained: "
            f"{' '.join(f'{v * 100:.2f}%' for v in var_explained.tolist())}\n"
        )
        for row in pcs_np:
            f.write("  ".join(f"{v: .6f}" for v in row) + "\n")

    if args.verb >= 1:
        print(f"\n  Saved: {out_path}")
        print("=" * 70)

    return 0
