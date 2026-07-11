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
        -metric lpc -shrink 4x2x1 -smooth 2x1x0 -iters 100x70x40 \\
        -save_warp -save_inverse -save_halfway

    # Distortion-style: restrict deformation to the Y (phase-encode) axis
    ffs_formwarp -base b0.nii.gz -source epi.nii.gz -prefix corr.nii.gz -noXdis -noZdis
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import torch

from fastfuncstuff.cli_utils import add_verbose_arg, parse_prefix, spinner
from fastfuncstuff.processing.formwarp import (
    METRICS,
    NO_X_DISP,
    NO_Y_DISP,
    NO_Z_DISP,
    SynConfig,
    formwarp,
)
from fastfuncstuff.processing.interp import WARP_INTERP_MODES
from fastfuncstuff.processing.io import load_image, save_image, save_warp_field, save_warp_series
from fastfuncstuff.processing.mask import automask


def _int_list(spec: str) -> tuple[int, ...]:
    """Parse an ANTs-style ``4x2x1`` (or ``4,2,1``) spec into a tuple of ints."""
    return tuple(int(p) for p in spec.replace(",", "x").split("x") if p != "")


def _float_list(spec: str) -> tuple[float, ...]:
    """Parse an ANTs-style ``2x1x0`` (or ``2,1,0``) spec into a tuple of floats."""
    return tuple(float(p) for p in spec.replace(",", "x").split("x") if p != "")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ffs_formwarp",
        description="GPU SyN (symmetric normalization) nonlinear registration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    p.add_argument("-base", required=True, help="Fixed/target image (3D).")
    p.add_argument("-source", required=True, help="Moving image to deform (3D).")
    p.add_argument("-prefix", required=True, help="Output path for the warped image.")
    p.add_argument(
        "-save_warp",
        action="store_true",
        help="Save the moving->fixed warp ({prefix}_WARP.nii.gz).",
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
        help="Image metric (cc=neighborhood cross-correlation, the SyN default).",
    )
    p.add_argument(
        "-cc_radius",
        "-cc-radius",
        type=int,
        default=4,
        help="CC neighborhood half-width in voxels (window (2r+1)^3).",
    )
    p.add_argument(
        "-lpa_sigma",
        "-lpa-sigma",
        type=float,
        default=4.0,
        help="Neighborhood size (voxels) for the lpa/lpc metrics.",
    )
    p.add_argument(
        "-lpa_kernel",
        "-lpa-kernel",
        choices=("gauss", "box"),
        default="gauss",
        help="Neighborhood kernel for lpa/lpc.",
    )

    # SyN regularization
    p.add_argument(
        "-grad_step",
        "-grad-step",
        type=float,
        default=0.25,
        help="Max per-iteration displacement in voxels (SyN gradientStep).",
    )
    p.add_argument(
        "-update_var",
        "-update-var",
        type=float,
        default=3.0,
        help="Fluid regularization: update-field Gaussian sigma (voxels).",
    )
    p.add_argument(
        "-total_var",
        "-total-var",
        type=float,
        default=0.0,
        help="Elastic regularization: total-field Gaussian sigma (voxels; 0=off).",
    )
    p.add_argument(
        "-invert_iters",
        "-invert-iters",
        type=int,
        default=8,
        help="Fixed-point iterations for displacement-field inversion.",
    )

    # Multiresolution (ANTs -f / -s / -c)
    p.add_argument(
        "-shrink", type=str, default="4x2x1", help="Per-level isotropic shrink factors, e.g. 4x2x1."
    )
    p.add_argument(
        "-smooth",
        type=str,
        default="2x1x0",
        help="Per-level Gaussian pre-smoothing sigmas in voxels, e.g. 2x1x0.",
    )
    p.add_argument(
        "-iters",
        type=str,
        default="100x70x40",
        help="Per-level max iteration counts, e.g. 100x70x40 (an upper bound; "
        "a level usually stops earlier via convergence).",
    )
    p.add_argument(
        "-conv_window",
        "-conv-window",
        type=int,
        default=10,
        help="Trailing-window size for convergence/early-stopping. <=0 disables "
        "early stopping (run the full -iters). The best-cost warp is always "
        "returned, so running to exhaustion never over-warps.",
    )
    p.add_argument(
        "-conv_thresh",
        "-conv-thresh",
        type=float,
        default=1e-6,
        help="Convergence slope threshold; larger stops sooner.",
    )

    # Axis constraints (match qwarp)
    p.add_argument("-noXdis", "-noxdis", action="store_true", help="No x-displacement.")
    p.add_argument("-noYdis", "-noydis", action="store_true", help="No y-displacement.")
    p.add_argument("-noZdis", "-nozdis", action="store_true", help="No z-displacement.")

    # Weight / mask
    p.add_argument("-weight", help="Weight image emphasizing the metric (overrides automask).")
    p.add_argument(
        "-automask", action="store_true", help="Restrict the metric to an automask of the base."
    )

    # Output interpolation
    p.add_argument(
        "-final_interp",
        "-final-interp",
        choices=WARP_INTERP_MODES,
        default="wsinc5",
        help="Interpolation for the final warped image.",
    )

    # Device
    p.add_argument("-device", default=None, help="torch device (cuda/cpu/mps). Auto if unset.")
    add_verbose_arg(p)
    return p.parse_args(argv)


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
        print(f"ffs_formwarp: device={device}")

    t0 = time.time()
    with spinner(f"Loading {Path(args.base).name}"):
        base, base_info = load_image(args.base, device=torch.device("cpu"))
    with spinner(f"Loading {Path(args.source).name}"):
        source, _ = load_image(args.source, device=torch.device("cpu"))

    if base.ndim == 4:
        if args.verb >= 1:
            print("WARNING: 4D -base; using vol[0] as the fixed target")
        base = base[0]

    # Timeseries mode: a 4D -source registers every volume to the (3D) base and
    # writes a 4D warped series + a per-volume warp series (5D file or folder).
    timeseries = source.ndim == 4
    if timeseries and tuple(source.shape[1:]) != tuple(base.shape):
        print(
            f"ERROR: source volumes {tuple(source.shape[1:])} and base {tuple(base.shape)} "
            "must be on the same grid (resample first, e.g. ffs_allineate)."
        )
        return 1
    if not timeseries and tuple(base.shape) != tuple(source.shape):
        print(
            f"ERROR: base {tuple(base.shape)} and source {tuple(source.shape)} "
            "must be on the same grid (resample first, e.g. ffs_allineate)."
        )
        return 1

    nz, ny, nx = base.shape
    if args.verb >= 1:
        print(f"Base/source: {nx}x{ny}x{nz}, loaded in {time.time() - t0:.1f}s")

    # Weight / mask
    weight = None
    mask = None
    if args.weight is not None:
        with spinner(f"Loading {Path(args.weight).name}"):
            weight, _ = load_image(args.weight, device=torch.device("cpu"))
    elif args.automask:
        mask = automask(base.float().to(device), device=device).float()
        if args.verb >= 1:
            frac = 100.0 * (mask > 0).float().mean().item()
            print(f"Automask: {frac:.1f}% of voxels inside brain")

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
        grad_step=args.grad_step,
        update_var=args.update_var,
        total_var=args.total_var,
        invert_iters=args.invert_iters,
        shrink_factors=_int_list(args.shrink),
        smoothing_sigmas=_float_list(args.smooth),
        iterations=_int_list(args.iters),
        convergence_window=args.conv_window,
        convergence_threshold=args.conv_thresh,
        warp_flags=warp_flags,
        final_interp=args.final_interp,
        verb=args.verb,
    )

    pfx = parse_prefix(args.prefix)
    prefix, nii_ext = pfx.stem, pfx.nifti_ext

    if timeseries:
        return _run_timeseries(
            args, base, source, weight, mask, config, base_info, prefix, nii_ext, device, t0
        )

    res = formwarp(base, source, weight=weight, mask=mask, config=config, device=device)

    warped_path = pfx.as_file()
    with spinner(f"Writing {Path(warped_path).name}"):
        save_image(res.warped, warped_path, header_info=base_info)
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
                header_info=base_info,
                units="mm",
            )
        if args.verb >= 1:
            print(f"Saved {label}: {path}")

    if args.save_warp:
        _save_warp(res.fwd, f"{prefix}_WARP{nii_ext}", "moving->fixed warp")
    if args.save_inverse:
        _save_warp(res.inv, f"{prefix}_WARPINV{nii_ext}", "fixed->moving inverse warp")
    if args.save_halfway:
        _save_warp(res.fixed_to_mid, f"{prefix}_HALF_mid2fixed{nii_ext}", "mid->fixed half-warp")
        _save_warp(res.moving_to_mid, f"{prefix}_HALF_mid2moving{nii_ext}", "mid->moving half-warp")
        _save_warp(res.mid_to_fixed, f"{prefix}_HALF_fixed2mid{nii_ext}", "fixed->mid half-warp")
        _save_warp(res.mid_to_moving, f"{prefix}_HALF_moving2mid{nii_ext}", "moving->mid half-warp")

    if args.verb >= 1:
        print(f"Done in {time.time() - t0:.1f}s")
    return 0


def _run_timeseries(
    args, base, source, weight, mask, config, base_info, prefix, nii_ext, device, t0
) -> int:
    """Register every volume of a 4D source to the 3D base (per-volume SyN).

    Writes the 4D warped series and, with -save_warp/-save_inverse, a per-volume
    warp series in the chosen -warp_format (one 5D file or a folder of 4D frames).
    -save_halfway is single-pair only and is ignored here.
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
        # Silence per-volume SyN chatter; the bar is the progress signal.
        res = formwarp(
            base,
            source[t],
            weight=weight,
            mask=mask,
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
