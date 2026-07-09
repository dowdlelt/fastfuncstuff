#!/usr/bin/env python
"""
ffs_util_3dmath — voxelwise math over one or more datasets (≈ 3dMean / 3dcalc).

Two modes:

  Reduction across inputs (like 3dMean):
      ffs_util_3dmath -input a.nii b.nii c.nii -mean   -prefix mean.nii.gz
      ffs_util_3dmath -input run*.nii.gz          -max    -prefix max.nii.gz
    Supported: -mean -sum -max -min -std -median

  Expression (like 3dcalc): inputs bind to a, b, c, ... in order.
      ffs_util_3dmath -input a.nii b.nii -expr 'a-b'         -prefix diff.nii.gz
      ffs_util_3dmath -input epi.nii     -expr 'step(a-100)' -prefix mask.nii.gz

All inputs must share a shape. The output header (affine, TR, units) is copied
from the first input. Small, deliberately: extend with more ops as needed.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from fastfuncstuff.processing.io import load_image, save_image

# Expression helpers exposed to -expr (3dcalc-style), kept intentionally small.
_EXPR_FUNCS = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "min": np.minimum,
    "max": np.maximum,
    "mean": lambda *a: np.mean(np.stack(a), axis=0),
    "step": lambda x: (np.asarray(x) > 0).astype(np.float32),
    "ispositive": lambda x: (np.asarray(x) > 0).astype(np.float32),
    "isnegative": lambda x: (np.asarray(x) < 0).astype(np.float32),
    "not": lambda x: (np.asarray(x) == 0).astype(np.float32),
    "np": np,
}

_REDUCTIONS = {
    "mean": lambda s: s.mean(0),
    "sum": lambda s: s.sum(0),
    "max": lambda s: s.amax(0),
    "min": lambda s: s.amin(0),
    "std": lambda s: s.std(0),
    "median": lambda s: s.median(0).values,
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ffs_util_3dmath",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "-input",
        nargs="+",
        required=True,
        metavar="FILE",
        help="One or more datasets (.nii/.nii.gz/.nii.zst). Bound to "
        "a, b, c, ... for -expr, in the order given.",
    )
    p.add_argument("-prefix", required=True, metavar="FILE", help="Output path.")
    grp = p.add_mutually_exclusive_group(required=True)
    for op in _REDUCTIONS:
        grp.add_argument(
            f"-{op}",
            dest="op",
            action="store_const",
            const=op,
            help=f"Voxelwise {op} across all inputs.",
        )
    grp.add_argument(
        "-expr", metavar="EXPR", help="3dcalc-style expression over a, b, c, ... (input order)."
    )
    p.add_argument("-mask", metavar="FILE", help="Only compute inside mask>0; zero elsewhere.")
    p.add_argument("-device", default="cpu", help="torch device (cpu/cuda).")
    p.add_argument("-overwrite", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    from pathlib import Path

    if Path(args.prefix).exists() and not args.overwrite:
        print(f"ERROR: {args.prefix} exists (use -overwrite).", file=sys.stderr)
        return 1

    dev = torch.device(args.device)
    vols, hdr0 = [], None
    for f in args.input:
        d, h = load_image(f)
        if hdr0 is None:
            hdr0, shape0 = h, tuple(d.shape)
        elif tuple(d.shape) != shape0:
            print(
                f"ERROR: shape mismatch: {f} is {tuple(d.shape)}, expected {shape0}.",
                file=sys.stderr,
            )
            return 1
        vols.append(d.to(dev).float())

    if args.op is not None:
        stack = torch.stack(vols, dim=0)
        out = _REDUCTIONS[args.op](stack)
    else:
        # -expr: bind inputs to a, b, c, ... as numpy arrays.
        names = [chr(ord("a") + i) for i in range(len(vols))]
        if len(vols) > 26:
            print("ERROR: -expr supports at most 26 inputs (a..z).", file=sys.stderr)
            return 1
        env = {n: v.cpu().numpy() for n, v in zip(names, vols, strict=True)}
        env.update(_EXPR_FUNCS)
        try:
            res = eval(args.expr, {"__builtins__": {}}, env)  # noqa: S307 (3dcalc-style)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: could not evaluate -expr {args.expr!r}: {exc}", file=sys.stderr)
            return 1
        out = torch.as_tensor(np.asarray(res, dtype=np.float32), device=dev)
        if out.shape != vols[0].shape:
            out = out.expand(vols[0].shape).clone()

    if args.mask:
        m, _ = load_image(args.mask)
        out = out * (m.to(dev).float() > 0)

    save_image(out.cpu(), args.prefix, header_info=hdr0)
    print(f"ffs_util_3dmath: wrote {args.prefix}  ({tuple(out.shape)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
