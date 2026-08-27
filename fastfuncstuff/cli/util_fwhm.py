"""CLI for whole-volume spatial smoothness estimation (AFNI 3dFWHMx).

Command: ffs_util_fwhm (registered as entry point in pyproject.toml)

Estimates the OVERALL spatial blur of a 4-D dataset the way ``3dFWHMx`` does:
the spatial autocorrelation within each sub-brick, averaged over sub-bricks, fit
to the mixed ACF model ``a*exp(-r^2/2b^2) + (1-a)*exp(-r/c)``. One
``(a, b, c, FWHM)`` for the dataset, plus the classic Forman per-axis FWHM.

This is the right tool for a residual (errts) dataset — feed it a model's
residuals and a brain mask. It is NOT ``ffs_util_localstat -stat ACF`` /
``3dLocalACF`` (a per-voxel spatially-varying map).

Usage:
    ffs_util_fwhm -input errts.nii.gz -mask mask.nii.gz
    ffs_util_fwhm -input errts+tlrc.HEAD -mask mask+tlrc.HEAD -acf1D out.1D
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter
from fastfuncstuff.cli_utils import add_verbose_arg, setup_device, spinner
from fastfuncstuff.processing.io import load_image
from fastfuncstuff.stats.fwhmx import estimate_fwhmx_run
from fastfuncstuff.utils import REGISTRATION_TF32


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = FfsArgumentParser(
        formatter_class=FfsHelpFormatter,
        prog="ffs_util_fwhm",
        description="Whole-volume spatial smoothness (classic Forman FWHM + mixed "
        "ACF model), GPU, matching AFNI 3dFWHMx. Intended for model residuals.",
    )
    parser.add_argument("-input", required=True, help="4D dataset (.nii/.nii.gz or AFNI .HEAD)")
    parser.add_argument(
        "-mask",
        default=None,
        help="Mask dataset (nonzero = in). Strongly recommended; without it the "
        "whole volume is used (or -automask).",
    )
    parser.add_argument(
        "-automask",
        action="store_true",
        help="Compute a brain mask from the data (mean volume, 3dAutomask) when "
        "-mask is not given.",
    )
    parser.add_argument(
        "-detrend",
        nargs="?",
        const=-1,
        type=int,
        default=None,
        metavar="q",
        help="Polynomial-detrend each voxel time series to order q before "
        "estimating (order 0 = remove the mean). Given without q, an order is "
        "picked from the run length. Off by default — model residuals (errts) "
        "are already detrended.",
    )
    parser.add_argument(
        "-acf_radius",
        "-acf-radius",
        type=float,
        default=None,
        metavar="MM",
        help="ACF radius in mm (default: AFNI's data-driven "
        "max(2.999*FWHM, 3.999*cbrt(voxel volume))).",
    )
    parser.add_argument(
        "-unif",
        action="store_true",
        help="Uniformize spatial variance: divide each voxel by its temporal MAD "
        "(subtracting the temporal median first) before estimating. Matches "
        "3dFWHMx, which enables this automatically with -detrend; it changes the "
        "ACF on data with non-uniform variance (high-res / anisotropic).",
    )
    parser.add_argument(
        "-demed",
        action="store_true",
        help="Subtract each voxel's temporal median before estimating "
        "(3dFWHMx -demed). Implied by -unif.",
    )
    parser.add_argument(
        "-nounif",
        "-no_unif",
        "-no-unif",
        action="store_true",
        help="Do not uniformize even when -detrend is given (overrides the "
        "AFNI-matching default that -detrend implies -unif).",
    )
    parser.add_argument(
        "-keep_const",
        "-keep-const",
        action="store_true",
        help="Keep voxels that are constant in time. By default they are dropped "
        "from the mask (matches 3dFWHMx, which removes constant-in-time voxels).",
    )
    parser.add_argument(
        "-acf1D",
        default=None,
        metavar="FILE",
        help="Write the fitted ACF params to a text file (a b c FWHM), one row.",
    )
    parser.add_argument("-device", default=None, help="cuda | cpu | mps (default: auto).")
    add_verbose_arg(parser, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    device = setup_device(args.device, tf32=REGISTRATION_TF32)
    verb = args.verb
    t0 = time.time()

    with spinner(f"Loading {Path(args.input).name}"):
        data, header = load_image(args.input)  # (nt, nz, ny, nx)
    if data.ndim != 4:
        raise SystemExit(f"ffs_util_fwhm: expected a 4D dataset, got shape {tuple(data.shape)}")
    nt, nz, ny, nx = data.shape
    volume_shape = (nz, ny, nx)

    # Voxel widths aligned to (nz, ny, nx): NIfTI affine columns are (x, y, z).
    affine = np.asarray(header["affine"], dtype=np.float64)
    sx, sy, sz = (float(np.sqrt((affine[:3, i] ** 2).sum())) for i in range(3))
    voxdims = (sz, sy, sx)

    # Mask: -mask file > -automask > whole volume.
    if args.mask is not None:
        with spinner(f"Loading {Path(args.mask).name}"):
            mask_img, _ = load_image(args.mask)
        if mask_img.ndim == 4:
            mask_img = mask_img[0]
        if tuple(mask_img.shape) != volume_shape:
            raise SystemExit(
                f"ffs_util_fwhm: mask shape {tuple(mask_img.shape)} != data grid {volume_shape}"
            )
        mask = mask_img != 0
    elif args.automask:
        from fastfuncstuff.processing.mask import automask

        with spinner("Automask (mean volume)"):
            mask = automask(data.mean(dim=0).to(device), device=device).cpu().to(torch.bool)
        if verb >= 1:
            print(f"  automask: {int(mask.sum())} voxels")
    else:
        mask = torch.ones(volume_shape, dtype=torch.bool)

    # Residual rows for the True voxels, row-major over (nz, ny, nx).
    flat_mask = mask.reshape(-1)
    resid = data.permute(1, 2, 3, 0).reshape(-1, nt)[flat_mask]  # (n_masked, nt)
    n_in = int(flat_mask.sum())

    # Optional polynomial detrend of each voxel time series (order 0 = demean).
    if args.detrend is not None:
        from fastfuncstuff.design.builder import legendre_polynomials

        q = args.detrend
        if q < 0:  # auto order from run length
            q = min(20, max(2, round(nt / 30)))
        p_np = legendre_polynomials(nt, q, normalize=False)
        P = torch.from_numpy(p_np).to(device=device, dtype=torch.float32)
        Q, _ = torch.linalg.qr(P)  # orthonormal basis for the polynomial span
        rd = resid.to(device)
        resid = (rd - (rd @ Q) @ Q.T).cpu()
        if verb >= 1:
            print(f"  detrending: {q + 1} baseline funcs (polort {q}), {nt} time points")

    # Drop constant-in-time voxels (3dFWHMx does this; they have no ACF signal
    # and would divide by ~0 in the temporal demeaning).
    if not args.keep_const:
        keep = resid.std(dim=1) > 1e-6
        n_dropped = int((~keep).sum())
        if n_dropped:
            idx = flat_mask.nonzero(as_tuple=False).flatten()
            flat_mask = flat_mask.clone()
            flat_mask[idx[~keep]] = False
            mask = flat_mask.reshape(volume_shape)
            resid = resid[keep]
            if verb >= 1:
                print(f"  removed {n_dropped} constant-in-time voxels from mask")

    if verb >= 1:
        print(
            f"ffs_util_fwhm: {nt} sub-bricks, grid {volume_shape}, "
            f"{int(mask.sum())}/{n_in} mask voxels, voxdims {voxdims}, device={device}"
        )

    # AFNI enables -unif automatically when -detrend is used; mirror that unless
    # -nounif is given. -unif implies -demed.
    unif = args.unif or (args.detrend is not None and not args.nounif)
    demed = args.demed or unif
    if verb >= 1 and (unif or demed):
        print(f"  {'uniformizing (median + MAD)' if unif else 'de-medianing'} per voxel")

    res = estimate_fwhmx_run(
        resid,
        mask,
        volume_shape,
        voxdims,
        radius_mm=args.acf_radius,
        demed=demed,
        unif=unif,
        device=device,
        progress=verb >= 1,
    )

    fz, fy, fx = res.classic_fwhm  # aligned to (nz, ny, nx)
    # AFNI 3dFWHMx-style two-line summary (x y z combined ; a b c FWHM), so it
    # reads like the reference tool.
    print(f"  {fx:.4f}  {fy:.4f}  {fz:.4f}    {res.classic_combined:.4f}")
    print(f"  {res.a:.6g}  {res.b:.6g}  {res.c:.6g}    {res.fwhm:.6g}")
    if verb >= 1:
        print(
            f"ACF: a={res.a:.4f} b={res.b:.4f} c={res.c:.4f} -> FWHM {res.fwhm:.4f} mm "
            f"(radius {res.radius:.2f} mm); classic FWHM x/y/z = "
            f"{fx:.2f}/{fy:.2f}/{fz:.2f} mm"
        )

    if args.acf1D is not None:
        Path(args.acf1D).write_text(
            f"# a b c FWHM (mm); ACF(r)=a*exp(-r^2/2b^2)+(1-a)*exp(-r/c)\n"
            f"{res.a:.6f} {res.b:.6f} {res.c:.6f} {res.fwhm:.6f}\n"
        )
        if verb >= 1:
            print(f"Wrote ACF params: {args.acf1D}")

    if verb >= 1:
        print(f"Time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
