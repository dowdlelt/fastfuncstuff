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

  Concatenation along time (like 3dTcat):
      ffs_util_3dmath -input a.nii b.nii c.nii -tcat -labels ses-01 ses-02 ses-03 \
          -prefix stack.nii.gz

All inputs must share a shape (spatial shape only, for -tcat). The output header
(affine, TR, units) is copied from the first input. Small, deliberately: extend
with more ops as needed.
"""

from __future__ import annotations

import argparse
import shlex
import sys

import numpy as np
import torch

from fastfuncstuff.cli_utils import (
    add_batch_args,
    collect_batch_jobs,
    run_batch_jobs,
    setup_device,
    spinner,
)
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
        metavar="FILE",
        help="One or more datasets (.nii/.nii.gz/.nii.zst). Bound to "
        "a, b, c, ... for -expr, in the order given.",
    )
    p.add_argument("-prefix", metavar="FILE", help="Output path. [required unless -batch]")
    grp = p.add_mutually_exclusive_group()
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
    grp.add_argument(
        "-tcat",
        dest="op",
        action="store_const",
        const="tcat",
        help="Concatenate the inputs along TIME into one 4-D stack (≈ 3dTcat). "
        "Only the spatial shape has to match; a 4-D input contributes all of its "
        "volumes. Pair with -labels so a viewer names the sub-bricks.",
    )
    p.add_argument(
        "-labels",
        nargs="+",
        metavar="LAB",
        help="Sub-brick labels for the output (AFNI BRICK_LABS). Give one per "
        "output volume, or one per -input (a 4-D input's volumes are then "
        "suffixed #0, #1, ...).",
    )
    p.add_argument("-mask", metavar="FILE", help="Only compute inside mask>0; zero elsewhere.")
    p.add_argument("-device", default="cpu", help="torch device (cpu/cuda).")
    p.add_argument("-overwrite", action="store_true")
    add_batch_args(
        p,
        tool="ffs_util_3dmath",
        what="voxelwise math jobs",
        example="-input a.nii b.nii -mean -prefix mean.nii.gz",
        skip_note="-prefix",
    )
    return p


def _expected_outputs(args: argparse.Namespace) -> list[str]:
    """Concrete output paths a solo run of ``args`` would write, for -batch_skip."""
    return [args.prefix] if args.prefix else []


def _validate_batch_run(run_args: argparse.Namespace) -> None:
    """Per-run validation for a batch job: the flags argparse would have enforced."""
    missing = [f for f in ("input", "prefix") if not getattr(run_args, f, None)]
    if missing:
        raise ValueError("run is missing " + ", ".join("-" + m for m in missing))
    if run_args.op is None and run_args.expr is None:
        raise ValueError("run has no operation (-mean/-max/.../-tcat/-expr)")


def _resolve_labels(
    labels: list[str] | None, per_input: list[int], out: torch.Tensor
) -> list[str] | None:
    """Expand ``-labels`` to one label per OUTPUT sub-brick, or None.

    One label per input is the form a caller naturally writes ("these files, in
    this order"), so a 4-D input's volumes are suffixed ``#j`` rather than making
    the caller count volumes it does not control. A count that matches neither is
    a caller bug worth saying out loud, but not worth failing a whole job over —
    the data is still correct, it just loses its names.
    """
    if not labels:
        return None
    nvol = out.shape[0] if out.ndim == 4 else 1
    if len(labels) == nvol:
        return list(labels)
    if len(labels) == len(per_input):
        expanded: list[str] = []
        for lab, n in zip(labels, per_input, strict=True):
            expanded.extend([lab] if n == 1 else [f"{lab}#{j}" for j in range(n)])
        if len(expanded) == nvol:
            return expanded
    print(
        f"WARNING: -labels has {len(labels)} entries but the output has {nvol} "
        f"sub-brick(s) ({len(per_input)} inputs); labels not written.",
        file=sys.stderr,
    )
    return None


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.batch is not None or args.batch_run:
        # Reductions and tcats are seconds of I/O around ~2s of interpreter and
        # torch startup, and autoproc emits a dozen of them (session means,
        # grandmeans, every QC stack). One process pays that once.
        run_batch_jobs(
            tool="ffs_util_3dmath",
            jobs=collect_batch_jobs(args.batch, args.batch_run),
            device=setup_device(args.device),
            parse_line=lambda line, base: _build_parser().parse_args(shlex.split(line), base),
            defaults=args,
            dispatch=_dispatch_run,
            validate=_validate_batch_run,
            is_nested=lambda ra: ra.batch is not None or ra.batch_run is not None,
            expected_outputs=_expected_outputs,
            skip_existing=args.batch_skip,
        )
        return 0

    try:
        _validate_batch_run(args)
        _dispatch_run(args, setup_device(args.device))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def _dispatch_run(args: argparse.Namespace, dev: torch.device) -> None:
    """Run one self-contained math job (the whole per-output body).

    Both the standalone path and every batch job go through here, so a manifest
    line reproduces a solo invocation bit-for-bit. Bad requests raise so that a
    batch records the failure and carries on with the remaining jobs."""
    from pathlib import Path

    if Path(args.prefix).exists() and not args.overwrite:
        raise ValueError(f"{args.prefix} exists (use -overwrite)")

    # -tcat stacks along time, so only the spatial lattice has to agree; every
    # other op is voxelwise and needs the full shape to match.
    key = (lambda s: s[-3:]) if args.op == "tcat" else (lambda s: s)
    vols, hdr0 = [], None
    for f in args.input:
        d, h = load_image(f)
        if hdr0 is None:
            hdr0, shape0 = h, key(tuple(d.shape))
        elif key(tuple(d.shape)) != shape0:
            raise ValueError(f"shape mismatch: {f} is {tuple(d.shape)}, expected {shape0}")
        vols.append(d.to(dev).float())

    if args.op == "tcat":
        out = torch.cat([v if v.ndim == 4 else v[None] for v in vols], dim=0)
    elif args.op is not None:
        stack = torch.stack(vols, dim=0)
        out = _REDUCTIONS[args.op](stack)
    else:
        # -expr: bind inputs to a, b, c, ... as numpy arrays.
        names = [chr(ord("a") + i) for i in range(len(vols))]
        if len(vols) > 26:
            raise ValueError("-expr supports at most 26 inputs (a..z)")
        env = {n: v.cpu().numpy() for n, v in zip(names, vols, strict=True)}
        env.update(_EXPR_FUNCS)
        try:
            res = eval(args.expr, {"__builtins__": {}}, env)  # noqa: S307 (3dcalc-style)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"could not evaluate -expr {args.expr!r}: {exc}") from exc
        out = torch.as_tensor(np.asarray(res, dtype=np.float32), device=dev)
        if out.shape != vols[0].shape:
            out = out.expand(vols[0].shape).clone()

    labels = _resolve_labels(args.labels, [v.shape[0] if v.ndim == 4 else 1 for v in vols], out)

    if args.mask:
        with spinner(f"Loading {Path(args.mask).name}"):
            m, _ = load_image(args.mask)
        out = out * (m.to(dev).float() > 0)

    with spinner(f"Writing {Path(args.prefix).name}"):
        save_image(out.cpu(), args.prefix, header_info=hdr0, brick_labels=labels)
    print(f"ffs_util_3dmath: wrote {args.prefix}  ({tuple(out.shape)})")


if __name__ == "__main__":
    raise SystemExit(main())
