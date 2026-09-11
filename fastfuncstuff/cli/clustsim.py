#!/usr/bin/env python3
"""
ffs_clustsim — Monte-Carlo cluster-size thresholds (3dClustSim).

Simulates noise-only volumes with a prescribed spatial autocorrelation,
finds the largest null cluster at each per-voxel threshold, and turns the
distribution of that maximum into the cluster-size table AFNI's viewer
reads out of a stats dataset's header.

Unlike ``ffs_perm`` this needs no permutable design, so it applies to an
ordinary first-level GLM::

    ffs_clustsim -mask mask.nii.gz -acf 0.6 3.0 5.0 -niter 10000 \\
                 -prefix CStemp -refit stats.nii.gz

``-acf_from`` closes the loop in one process: the ACF is estimated off a
residual dataset with the same 3dFWHMx port ffs_reml uses, so there is no
3dFWHMx → parse-a-.1D → 3dClustSim → 3drefit round trip::

    ffs_clustsim -acf_from errts.nii.gz -mask mask.nii.gz -refit stats.nii.gz \\
                 -prefix CStemp

The threshold is a z, not a t: the fields are renormalised to unit
standard deviation, so one table serves every sub-brick regardless of its
degrees of freedom.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter

# The hyphenated spelling is what the NIML `thresholding` attribute carries; the
# bare form (stats.clustsim.SIDED_ATTR) names the file and the 3drefit attribute.
_SIDED_FROM_CLI = {"1sided": "1-sided", "2sided": "2-sided", "bisided": "bi-sided"}


def build_parser() -> FfsArgumentParser:
    p = FfsArgumentParser(
        prog="ffs_clustsim",
        description=__doc__,
        formatter_class=FfsHelpFormatter,
    )

    inp = p.add_argument_group("Inputs")
    inp.add_argument(
        "-mask",
        required=True,
        help="Mask dataset.  Simulated clusters are confined to it, and its "
        "grid and voxel size set the simulation geometry.",
    )
    smooth = inp.add_mutually_exclusive_group(required=True)
    smooth.add_argument(
        "-acf",
        nargs=3,
        type=float,
        metavar=("A", "B", "C"),
        help="Mixed-model ACF parameters from 3dFWHMx -acf (or ffs_reml "
        "-save_acf):  ACF(r) = a*exp(-r^2/2b^2) + (1-a)*exp(-r/c).",
    )
    smooth.add_argument(
        "-acf_from",
        metavar="RESID",
        help="Estimate the ACF from this residual dataset (errts) instead of "
        "taking it as a number.  Uses the same 3dFWHMx port as ffs_reml.",
    )
    smooth.add_argument(
        "-fwhm",
        type=float,
        help="Pure-Gaussian smoothness in mm, as an equivalent ACF.  Real fMRI "
        "residuals have a heavier tail than a Gaussian, so -acf is the "
        "defensible choice; this exists for comparison against old results.",
    )

    sim = p.add_argument_group("Simulation")
    sim.add_argument(
        "-niter",
        type=int,
        default=10000,
        help="Monte-Carlo iterations.  Below ~2000 the tail of the table is too noisy to trust.",
    )
    sim.add_argument(
        "-seed",
        type=int,
        default=None,
        help="Random seed (default: nondeterministic).  Reproduces a run "
        "exactly only together with -batch, since the automatic batch size "
        "depends on how much memory is free at the time.",
    )
    sim.add_argument(
        "-batch",
        type=int,
        default=None,
        help="Volumes simulated per batch.  Default: from free memory on the "
        "target device.  Lower it if the GPU is shared.",
    )
    sim.add_argument(
        "-cpu_cluster",
        action="store_true",
        help="Cluster on CPU worker processes even when simulating on a GPU.  "
        "The two agree exactly; this is the fallback if the fused kernels "
        "are unavailable.",
    )
    sim.add_argument(
        "-pthr",
        nargs="+",
        type=float,
        default=None,
        help="Per-voxel uncorrected p thresholds (rows of the table).",
    )
    sim.add_argument(
        "-athr",
        nargs="+",
        type=float,
        default=None,
        help="Family-wise corrected alphas (columns of the table).",
    )
    sim.add_argument(
        "-LOTS",
        action="store_true",
        help="Use AFNI's larger 29-pthr x 10-athr grid.",
    )
    sim.add_argument(
        "-NN",
        nargs="+",
        type=int,
        choices=(1, 2, 3),
        default=[1, 2, 3],
        help="Connectivities to tabulate: 1=faces, 2=+edges, 3=+corners.",
    )
    sim.add_argument(
        "-sided",
        nargs="+",
        choices=("1sided", "2sided", "bisided"),
        default=["1sided", "2sided", "bisided"],
        help="Thresholding schemes.  bisided clusters each sign separately; "
        "2sided lets opposite-sign voxels join one cluster.",
    )

    out = p.add_argument_group("Output")
    out.add_argument(
        "-prefix",
        required=True,
        help="Output prefix for the .1D tables, the NIML files and the mask blob.",
    )
    out.add_argument(
        "-refit",
        metavar="DSET",
        default=None,
        help="Inject the tables into this stats dataset's AFNI header so the "
        "viewer reports cluster significance.  Done in-script; 3drefit is "
        "not required, though the equivalent script is still written.",
    )
    out.add_argument(
        "-nodec",
        action="store_true",
        help="Round cluster sizes up to whole voxels in the .1D tables.",
    )

    misc = p.add_argument_group("Misc")
    try:
        from fastfuncstuff.cli_utils import add_device_arg, add_verbose_arg

        add_device_arg(misc)
        add_verbose_arg(misc)
    except ImportError:  # pragma: no cover
        misc.add_argument("-device", default=None)
        misc.add_argument("-verb", type=int, default=1)
    misc.add_argument(
        "-jobs",
        "-j",
        type=int,
        default=None,
        help="Worker processes for the clustering pass.  Default: cpu_count-1.",
    )
    return p


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from fastfuncstuff.cli_utils import parse_prefix, setup_device
    from fastfuncstuff.io.afni import load_afni_mask, load_nifti
    from fastfuncstuff.stats.clustsim import (
        ACF,
        DEFAULT_CS_ATHR,
        DEFAULT_CS_PTHR,
        LOTS_ATHR,
        LOTS_PTHR,
        acf_fwhm,
        attach_clustsim_tables,
        random_field_grid,
    )
    from fastfuncstuff.stats.niml import resolve_mask_idcode

    t_start = time.time()
    verb = getattr(args, "verb", 1)
    device = setup_device(args.device)

    # ── Geometry ───────────────────────────────────────────────────────────
    mask = load_afni_mask(args.mask)
    ref = load_nifti(args.mask)
    voxmm = tuple(abs(float(z)) for z in ref.header.get_zooms()[:3])
    shape = tuple(int(s) for s in mask.shape)
    n_mask = int(mask.sum())
    if n_mask == 0:
        print("[ffs_clustsim] mask is empty", file=sys.stderr)
        return 1

    # ── Smoothness ─────────────────────────────────────────────────────────
    if args.acf is not None:
        acf = ACF(*args.acf)
    elif args.fwhm is not None:
        acf = ACF.from_fwhm(args.fwhm)
    else:
        acf = _estimate_acf(args.acf_from, mask, shape, voxmm, device, verb)

    pthr = tuple(args.pthr) if args.pthr else (LOTS_PTHR if args.LOTS else DEFAULT_CS_PTHR)
    athr = tuple(args.athr) if args.athr else (LOTS_ATHR if args.LOTS else DEFAULT_CS_ATHR)
    nns = tuple(sorted(set(args.NN)))
    sideds = tuple(_SIDED_FROM_CLI[s] for s in dict.fromkeys(args.sided))

    grid = random_field_grid(shape, voxmm, acf)
    if verb >= 1:
        print(
            f"[ffs_clustsim] {n_mask} voxels in mask "
            f"({100.0 * n_mask / np.prod(shape):.2f}% of {shape[0]}x{shape[1]}x{shape[2]})\n"
            f"[ffs_clustsim] ACF({acf.a:.2f},{acf.b:.2f},{acf.c:.2f}) => "
            f"FWHM={acf_fwhm(acf):.2f}mm => pads to {grid[0]}x{grid[1]}x{grid[2]}\n"
            f"[ffs_clustsim] {args.niter} iterations on {device}",
            file=sys.stderr,
        )

    # ── Tables + refit ─────────────────────────────────────────────────────
    prefix = parse_prefix(args.prefix)
    attach_clustsim_tables(
        mask,
        voxmm,
        acf,
        prefix=Path(prefix.stem),
        refit=args.refit,
        n_iter=args.niter,
        pthr=pthr,
        athr=athr,
        nns=nns,
        sideds=sideds,
        device=device,
        n_jobs=args.jobs,
        batch=args.batch,
        seed=args.seed,
        on_device=False if args.cpu_cluster else None,
        nodec=args.nodec,
        commandline=" ".join(["ffs_clustsim", *sys.argv[1:]]),
        mask_name=str(Path(args.mask).resolve()),
        mask_idcode=resolve_mask_idcode(args.mask),
        verbose=verb >= 1,
    )
    if args.refit is not None and verb >= 1:
        print(f"[ffs_clustsim] cluster tables inserted into {args.refit}", file=sys.stderr)

    if verb >= 1:
        print(f"[ffs_clustsim] done in {time.time() - t_start:.1f}s", file=sys.stderr)
    return 0


def _estimate_acf(resid_path, mask, shape, voxmm, device, verb):
    """Estimate the ACF off a residual dataset, the way 3dFWHMx -acf does."""
    import torch

    from fastfuncstuff.io.afni import load_nifti
    from fastfuncstuff.stats.clustsim import ACF
    from fastfuncstuff.stats.fwhmx import estimate_fwhmx_run

    img = load_nifti(resid_path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim != 4:
        raise SystemExit(f"-acf_from needs a 4-D residual dataset, got shape {data.shape}")
    if data.shape[:3] != shape:
        raise SystemExit(f"-acf_from grid {data.shape[:3]} does not match the mask grid {shape}")
    resid = torch.from_numpy(data[mask])  # (V, T)
    del data
    est = estimate_fwhmx_run(
        resid,
        torch.from_numpy(mask),
        shape,
        voxmm,
        device=device,
        progress=verb >= 1,
    )
    if verb >= 1:
        print(
            f"[ffs_clustsim] estimated ACF from {Path(resid_path).name}: "
            f"a={est.a:.4f} b={est.b:.4f} c={est.c:.4f}  FWHM={est.fwhm:.2f}mm",
            file=sys.stderr,
        )
    return ACF(est.a, est.b, est.c)


if __name__ == "__main__":
    sys.exit(main())
