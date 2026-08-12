"""CLI for spatial cross-correlation between 4D neuroimaging volumes.

Command: ffs_spatial_xcorr

Computes the full spatial cross-correlation matrix between all volumes of
two 4D NIfTI datasets within a mask, finds the optimal 1-to-1 matching,
and reports consistency metrics. Optionally saves a heatmap plot.

Examples:
    # Basic: correlate ICA components from two runs (auto-masks from nonzero)
    ffs_spatial_xcorr -a ica_run1.nii.gz -b ica_run2.nii.gz

    # Separate masks per dataset (intersected automatically)
    ffs_spatial_xcorr -a ica_run1.nii.gz -b ica_run2.nii.gz \\
        -mask_a mask_run1.nii.gz -mask_b mask_run2.nii.gz

    # Shared mask for both
    ffs_spatial_xcorr -a comps_a.nii.gz -b comps_b.nii.gz -mask brain.nii.gz

    # Spearman with plot, save matrix to text
    ffs_spatial_xcorr -a comps_a.nii.gz -b comps_b.nii.gz -mask brain.nii.gz \\
        -method spearman -plot xcorr.png -save_matrix xcorr.txt

    # One-to-many: correlate single reference against all volumes
    ffs_spatial_xcorr -a reference.nii.gz -b candidates.nii.gz -mask brain.nii.gz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.utils import REGISTRATION_TF32


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ffs_spatial_xcorr",
        description="Spatial cross-correlation between 4D NIfTI volumes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Masking:
  The final mask is always the INTERSECTION of all mask sources:
    -mask        Single mask applied to both datasets.
    -mask_a/b    Per-dataset masks (intersected).
    (no mask)    Auto-mask: intersection of nonzero voxels across all volumes
                 in both datasets. Good for ICA maps, z-maps, etc.

  Per-dataset masks are useful when data come from different analysis runs
  with slightly different brain coverage.

Method guidance:
  pearson    Best for data on comparable scales (z-maps, beta maps from same
             model). Sensitive to outliers and scale differences.
  spearman   Rank-based, robust to scale differences and outliers. Recommended
             when comparing data from different pipelines or with different
             scaling (e.g., raw ICA maps vs z-scored maps, or t-maps vs
             beta-maps). Also handles nonlinear monotonic relationships.
  kendall    Concordance-based, most robust to ties and outliers but slower
             (CPU-only, O(n log n) per pair). Best for ordinal comparisons
             or when many tied values exist.

  Use -abs for sign-ambiguous data (ICA components, eigenvectors) where a
  perfect anti-correlation is as good as a perfect correlation.

Output:
  Prints the consistency report to stdout.
  -save_matrix writes the full correlation matrix as whitespace-delimited text.
  -plot saves a heatmap PNG with the optimal matching highlighted.
""",
    )

    parser.add_argument("-a", required=True, help="First 4D NIfTI dataset (or 3D for one-to-many)")
    parser.add_argument("-b", required=True, help="Second 4D NIfTI dataset (or 3D for one-to-many)")

    mask_group = parser.add_argument_group("Masking")
    mask_group.add_argument(
        "-mask",
        default=None,
        help="Single 3D mask applied to both datasets (must match voxel grid)",
    )
    mask_group.add_argument(
        "-mask_a",
        default=None,
        help="Mask for dataset A (intersected with -mask_b and/or -mask)",
    )
    mask_group.add_argument(
        "-mask_b",
        default=None,
        help="Mask for dataset B (intersected with -mask_a and/or -mask)",
    )

    parser.add_argument(
        "-method",
        choices=["pearson", "spearman", "kendall"],
        default="pearson",
        help="Correlation method (default: pearson). See below for guidance.",
    )
    parser.add_argument(
        "-abs",
        action="store_true",
        dest="use_abs",
        help="Use absolute correlations (for sign-ambiguous data like ICA)",
    )
    parser.add_argument(
        "-save_matrix",
        default=None,
        help="Save full correlation matrix to text file",
    )
    parser.add_argument("-plot", default=None, help="Save heatmap plot to file (PNG/PDF/SVG)")
    parser.add_argument("-plot_dpi", type=int, default=150, help="Plot DPI (default: 150)")
    parser.add_argument(
        "-device",
        default=None,
        help="PyTorch device: cuda, cpu, mps (auto-detected)",
    )
    parser.add_argument(
        "-thresholds",
        default="0.5,0.7,0.8,0.9,0.95",
        help="Comma-separated correlation thresholds for coverage report "
        "(default: 0.5,0.7,0.8,0.9,0.95)",
    )

    return parser.parse_args(argv)


def _load_mask(path: str, expected_shape: tuple[int, ...]) -> torch.Tensor:
    """Load a mask file and validate its shape."""
    from fastfuncstuff.processing.io import load_image

    mask_vol, _ = load_image(path, device=torch.device("cpu"))
    if mask_vol.ndim == 4:
        mask_vol = mask_vol[0]
    if mask_vol.shape != expected_shape:
        print(
            f"ERROR: Mask {path} shape {tuple(mask_vol.shape)} doesn't match "
            f"data grid {expected_shape}",
            file=sys.stderr,
        )
        sys.exit(1)
    return mask_vol > 0


def _build_mask(
    args: argparse.Namespace,
    images_a: torch.Tensor,
    images_b: torch.Tensor,
) -> torch.Tensor:
    """Build the final intersection mask from all mask sources.

    Priority: intersect all of -mask, -mask_a, -mask_b.
    If none provided, auto-mask from intersection of nonzero voxels.
    """
    grid_shape = images_a.shape[1:]  # (nz, ny, nx)
    mask_parts: list[torch.Tensor] = []

    if args.mask:
        m = _load_mask(args.mask, grid_shape)
        print(f"  -mask:   {args.mask} — {m.sum().item()} voxels")
        mask_parts.append(m)

    if args.mask_a:
        m = _load_mask(args.mask_a, grid_shape)
        print(f"  -mask_a: {args.mask_a} — {m.sum().item()} voxels")
        mask_parts.append(m)

    if args.mask_b:
        m = _load_mask(args.mask_b, grid_shape)
        print(f"  -mask_b: {args.mask_b} — {m.sum().item()} voxels")
        mask_parts.append(m)

    if not mask_parts:
        # Auto-mask: any voxel nonzero in at least one volume per dataset
        nonzero_a = (images_a != 0).any(dim=0)
        nonzero_b = (images_b != 0).any(dim=0)
        auto_mask = nonzero_a & nonzero_b
        n_a = nonzero_a.sum().item()
        n_b = nonzero_b.sum().item()
        n_both = auto_mask.sum().item()
        print(f"  Auto-mask: A nonzero={n_a}, B nonzero={n_b}, intersection={n_both} voxels")
        return auto_mask

    # Intersect all mask sources
    final = mask_parts[0]
    for m in mask_parts[1:]:
        final = final & m
    n_final = final.sum().item()
    if len(mask_parts) > 1:
        print(f"  Intersection: {n_final} voxels")
    return final


def _make_plot(
    corr_matrix: np.ndarray,
    matched_rows: np.ndarray,
    matched_cols: np.ndarray,
    matched_corrs: np.ndarray,
    method: str,
    use_abs: bool,
    path: str,
    dpi: int,
) -> None:
    """Create and save the cross-correlation heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    n1, n2 = corr_matrix.shape

    fig_w = max(6, min(20, n2 * 0.4 + 2))
    fig_h = max(5, min(18, n1 * 0.4 + 2))
    fig, axes = plt.subplots(1, 2, figsize=(fig_w + 4, fig_h), width_ratios=[3, 1])

    ax_heat = axes[0]
    ax_bar = axes[1]

    # Heatmap
    if use_abs:
        vmin, vmax = 0, 1
        cmap = "hot"
        norm = None
    else:
        vmax = max(abs(corr_matrix.min()), abs(corr_matrix.max()), 0.1)
        vmin = -vmax
        cmap = "RdBu_r"
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    im = ax_heat.imshow(
        corr_matrix,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        vmin=vmin if norm is None else None,
        vmax=vmax if norm is None else None,
        interpolation="nearest",
    )

    # Mark optimal matching with green squares
    for r, c in zip(matched_rows, matched_cols, strict=False):
        ax_heat.plot(
            c,
            r,
            "s",
            color="lime",
            markersize=max(3, min(8, 100 / max(n1, n2))),
            markeredgecolor="black",
            markeredgewidth=0.5,
            alpha=0.8,
        )

    ax_heat.set_xlabel("Dataset B volume")
    ax_heat.set_ylabel("Dataset A volume")
    method_label = f"|{method}|" if use_abs else method
    ax_heat.set_title(f"Spatial cross-correlation ({method_label})")
    fig.colorbar(im, ax=ax_heat, shrink=0.8, label="r")

    # Matched correlations bar chart (sorted descending)
    n_matched = len(matched_corrs)
    colors = []
    for r in matched_corrs:
        if r >= 0.9:
            colors.append("#2ecc71")
        elif r >= 0.7:
            colors.append("#f39c12")
        elif r >= 0.5:
            colors.append("#e67e22")
        else:
            colors.append("#e74c3c")

    ax_bar.barh(range(n_matched), matched_corrs, color=colors, edgecolor="black", linewidth=0.3)
    ax_bar.set_xlabel("r")
    ax_bar.set_ylabel("Matched pair rank")
    ax_bar.set_title("Optimal matching")
    ax_bar.set_ylim(-0.5, n_matched - 0.5)
    ax_bar.invert_yaxis()
    ax_bar.axvline(0.9, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax_bar.axvline(0.7, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
    if use_abs:
        ax_bar.set_xlim(0, 1.05)
    else:
        ax_bar.set_xlim(min(0, matched_corrs.min() - 0.05), 1.05)

    mean_r = matched_corrs.mean()
    ax_bar.axvline(
        mean_r, color="blue", linestyle="-", alpha=0.6, linewidth=1.2, label=f"mean={mean_r:.3f}"
    )
    ax_bar.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    from fastfuncstuff.cli_utils import setup_device, spinner
    from fastfuncstuff.processing.io import load_image
    from fastfuncstuff.stats.spatial import (
        consistency_report,
        spatial_correlation_matrix,
    )

    # Device selection
    device = setup_device(args.device, tf32=REGISTRATION_TF32)

    t0 = time.time()
    thresholds = tuple(float(x) for x in args.thresholds.split(","))

    # Load inputs
    with spinner(f"Loading {Path(args.a).name}"):
        images_a, _ = load_image(args.a, device=torch.device("cpu"))
    with spinner(f"Loading {Path(args.b).name}"):
        images_b, _ = load_image(args.b, device=torch.device("cpu"))

    if images_a.ndim == 3:
        images_a = images_a.unsqueeze(0)
    if images_b.ndim == 3:
        images_b = images_b.unsqueeze(0)

    print(f"Dataset A: {args.a} — {images_a.shape[0]} volumes, shape {tuple(images_a.shape[1:])}")
    print(f"Dataset B: {args.b} — {images_b.shape[0]} volumes, shape {tuple(images_b.shape[1:])}")

    # Validate grid match
    if images_a.shape[1:] != images_b.shape[1:]:
        print(
            f"ERROR: Grid mismatch — A is {tuple(images_a.shape[1:])} "
            f"vs B is {tuple(images_b.shape[1:])}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build mask from all sources
    print("Mask:")
    mask = _build_mask(args, images_a, images_b)
    n_vox = mask.sum().item()
    if n_vox == 0:
        print("ERROR: Final mask has zero voxels", file=sys.stderr)
        sys.exit(1)
    print(f"  Final mask: {n_vox} voxels")

    print(f"Method: {args.method}, device: {device}")

    # Compute cross-correlation matrix
    corr_matrix = spatial_correlation_matrix(
        images_a, images_b, mask=mask, method=args.method, device=device
    )
    if args.use_abs:
        corr_matrix = np.abs(corr_matrix)

    t_corr = time.time() - t0
    print(f"Correlation matrix: {corr_matrix.shape[0]} x {corr_matrix.shape[1]} ({t_corr:.2f}s)")

    # Consistency report
    report = consistency_report(corr_matrix, method=args.method, thresholds=thresholds)
    print()
    print(report)

    # Save matrix
    if args.save_matrix:
        with spinner(f"Writing {Path(args.save_matrix).name}"):
            np.savetxt(args.save_matrix, corr_matrix, fmt="%.6f", delimiter="\t")
        print(f"\nMatrix saved: {args.save_matrix}")

    # Plot
    if args.plot:
        _make_plot(
            corr_matrix,
            report.matched_rows,
            report.matched_cols,
            report.matched_correlations,
            args.method,
            args.use_abs,
            args.plot,
            args.plot_dpi,
        )
        print(f"Plot saved: {args.plot}")

    print(f"\nTotal time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
