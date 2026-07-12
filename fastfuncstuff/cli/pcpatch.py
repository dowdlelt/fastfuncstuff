#!/usr/bin/env python3
"""CLI for patch-based residual-PC projection (ffs_pcpatch).

Projects the temporal noise subspace estimated from a NORDIC residual (carried
through the same preprocessing as the data) out of the non-denoised data —
NORDIC's thermal-noise benefit at a fraction of the degrees-of-freedom cost.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from fastfuncstuff.cli_utils import add_verbose_arg, parse_prefix, print_cli_header
from fastfuncstuff.denoise.pcpatch import PCPatchConfig, run_pcpatch
from fastfuncstuff.utils import configure_torch_backends, get_device


class _HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


def _parse_ncomps(token: str) -> tuple[str, float]:
    """`-ncomps`: an int (fixed component count) or a float in (0,1] (variance
    fraction). Returns ('count', n) or ('frac', f)."""
    is_float = ("." in token) or ("e" in token.lower())
    if is_float:
        f = float(token)
        if not (0.0 < f <= 1.0):
            raise argparse.ArgumentTypeError(
                f"variance fraction must be in (0, 1], got {f} "
                "(use an integer for a fixed component count)"
            )
        return ("frac", f)
    n = int(token)
    if n < 0:
        raise argparse.ArgumentTypeError(f"component count must be >= 0, got {n}")
    return ("count", float(n))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ffs_pcpatch",
        description="[FAILED EXPERIMENT — DO NOT USE] Patch-based residual-PC projection. "
        "The premise fails: NORDIC's residual is the near-full-rank thermal-noise floor, so "
        "this over-removes and guts signal. Use 'ffs_nordic -retain_dof' to cap DoF loss, or "
        "'ffs_reml -adjust_dof' to account for it. Kept only for the record.",
        formatter_class=_HelpFormatter,
        epilog="""
[FAILED EXPERIMENT] This tool does not work as intended and is retained only for
reproducibility of the write-up (fmri_wiki/concepts/Residual PC projection.md).
A variance fraction on a noise residual removes ~f*T components almost uniformly
everywhere and destroys the signal. Do not use it in a pipeline.

Original (non-working) workflow:
  1. ffs_nordic ... -save-residual-map -add-mean -save-num-comps
  2. Apply your preprocessing (motion/blur/warp) to BOTH the raw data and the
     NORDIC residual (and, optionally, the numcomps map) identically.
  3. ffs_pcpatch -data <preproc_raw> -residual <preproc_residual> -prefix clean \\
                 -ncomps 0.95 -polort 3 -ort motion.1D
  4. Feed clean_numcomps.nii.gz to ffs_util_updatedof / ffs_reml -adjust_dof.

Examples:
  # 95% of the post-skip residual variance per patch
  ffs_pcpatch -data raw_preproc.nii.gz -residual resid_preproc.nii.gz \\
              -prefix clean -ncomps 0.95

  # Fixed 20 components, skip the top injected-signal PC, de-nuisance the residual
  ffs_pcpatch -data raw_preproc.nii.gz -residual resid_preproc.nii.gz \\
              -prefix clean -ncomps 20 -skip-first 1 -polort 3 -ort motion.1D
""",
    )

    io = p.add_argument_group("Input/Output")
    io.add_argument("-data", "-input", required=True, help="4D non-denoised (preprocessed) series")
    io.add_argument(
        "-residual",
        required=True,
        help="4D NORDIC residual carried through the SAME preprocessing as -data",
    )
    io.add_argument("-prefix", required=True, help="Output prefix")
    io.add_argument(
        "-mask", default=None, help="Restrict to this mask (intersected with nonzero residual)"
    )
    io.add_argument(
        "-orig-numcomps",
        "-orig_numcomps",
        default=None,
        help="NORDIC numcomps map (same preprocessing) — reported as an original-rank guideline",
    )
    io.add_argument(
        "-no-gzip",
        "-no_gzip",
        dest="write_gzipped",
        action="store_false",
        help="Write .nii instead of .nii.gz",
    )

    algo = p.add_argument_group("Algorithm")
    algo.add_argument(
        "-ncomps",
        type=_parse_ncomps,
        default=("frac", 0.95),
        metavar="N|FRAC",
        help="Components to remove per patch: integer = fixed count, float in (0,1] = "
        "cumulative post-skip variance fraction (default: 0.95)",
    )
    algo.add_argument(
        "-skip-first",
        "-skip_first",
        type=int,
        default=0,
        help="Ignore this many leading PCs (transform-injected signal) before selecting "
        "and normalizing variance. Without -ort you likely want >=1.",
    )
    algo.add_argument(
        "-polort",
        type=int,
        default=-1,
        help="Legendre drift degree projected out of the residual before PCA (<0 = off). "
        "Use the same value as your GLM to avoid double-removing drift.",
    )
    algo.add_argument(
        "-ort",
        default=None,
        help="Nuisance regressors (.1D, one column per regressor) projected out of the "
        "residual before PCA — pass the SAME motion/nuisance you use in the GLM.",
    )
    algo.add_argument(
        "-kernel-size",
        "-kernel_size",
        type=int,
        nargs=3,
        default=None,
        metavar=("KX", "KY", "KZ"),
        help="Patch size (default: NORDIC's round((T*11)^(1/3)) cube)",
    )
    algo.add_argument(
        "-patch-overlap",
        "-patch_overlap",
        type=int,
        default=2,
        help="Patch step = kernel // overlap (default: 2)",
    )

    perf = p.add_argument_group("Performance")
    perf.add_argument("-device", default=None, help="cuda | cpu | mps (default: auto)")
    perf.add_argument("-svd-batch-size", "-svd_batch_size", type=int, default=512)
    add_verbose_arg(p)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    kind, value = args.ncomps
    n_comps = int(value) if kind == "count" else None
    var_frac = value if kind == "frac" else None

    ort = None
    if args.ort is not None:
        ort = np.loadtxt(args.ort, ndmin=2).astype(np.float32)

    prefix = parse_prefix(args.prefix).stem
    device = get_device(args.device)
    configure_torch_backends(device)

    print_cli_header("ffs_pcpatch", "Patch-based residual-PC projection")
    print(
        "  [FAILED EXPERIMENT] This tool over-removes on real data (see its --help). "
        "Prefer 'ffs_nordic -retain_dof' or 'ffs_reml -adjust_dof'."
    )
    print(f"Data:     {args.data}")
    print(f"Residual: {args.residual}")
    print(f"Output prefix: {prefix}")
    print(f"Device: {device}")

    cfg = PCPatchConfig(
        n_comps=n_comps,
        var_frac=var_frac,
        skip_first=args.skip_first,
        kernel_size=tuple(args.kernel_size) if args.kernel_size is not None else None,
        patch_overlap=max(1, args.patch_overlap),
        polort=args.polort,
        ort=ort,
        write_gzipped=args.write_gzipped,
        svd_batch_size=args.svd_batch_size,
        verbose=args.verb >= 1,
    )

    t0 = time.time()
    out = run_pcpatch(
        data_file=args.data,
        residual_file=args.residual,
        output_prefix=prefix,
        config=cfg,
        device=device,
        mask_file=args.mask,
        orig_numcomps_file=args.orig_numcomps,
    )
    print(f"\nWrote {out.data_file}")
    print(f"Wrote {out.num_comps_file}  (feed to ffs_util_updatedof / ffs_reml -adjust_dof)")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
