"""CLI: one surface statistics mask from every run's projection.

Command: ffs_util_surfmask

Usage:
    ffs_util_surfmask -mask run*.onavg-ico64.lh.mask.shape.gii \
        -mean run*.onavg-ico64.lh.mean.shape.gii -prefix all.onavg-ico64.lh
"""

from __future__ import annotations

import argparse

import numpy as np

from fastfuncstuff.cli_help import FfsArgumentParser

_DESCRIPTION = """\
The vertex twin of autoproc's volume masks, for ffs_reml -mask on surface data.

Each run's ffs_nwarp -surf already writes PREFIX.SPACE.?h.mask.shape.gii: cortex label
AND that run's footprints inside the EPI in every frame. This intersects them (a
vertex any run lost would have a hole in its series), then drops vertices whose mean
over runs is below AFNI's clip level (THD_cliplevel, 3dAutomask's threshold) of the
vertices still in: signal dropout, not ordinary intensity bias.

Outputs (prefix P): P.mask.shape.gii, and with -mean P.meanall.shape.gii. The mesh
metadata (fingerprint, geometry) is carried from the first -mask.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = FfsArgumentParser(prog="ffs_util_surfmask", description=_DESCRIPTION)
    p.add_argument("-mask", nargs="+", required=True, metavar="MASK.gii",
                   help="Every run's ?h.mask.shape.gii, one hemisphere and mesh.")  # fmt: skip
    p.add_argument("-mean", nargs="+", default=None, metavar="MEAN.gii",
                   help="Every run's ?h.mean.shape.gii, same order. Without it, no clip.")  # fmt: skip
    p.add_argument("-clfrac", type=float, default=0.5,
                   help="Clip fraction (3dAutomask -clfrac); 0 = intersection only.")  # fmt: skip
    p.add_argument("-prefix", required=True, help="Output stem: PREFIX.mask.shape.gii.")
    return p.parse_args(argv)


def _stem(prefix: str) -> str:
    for ext in (".shape.gii", ".func.gii", ".gii"):
        if prefix.endswith(ext):
            return prefix[: -len(ext)].removesuffix(".mask")
    return prefix


def main(argv: list[str] | None = None) -> int:
    from fastfuncstuff.io.gifti import load_gifti_data, save_gifti_data
    from fastfuncstuff.surface.mask import combine_run_masks

    args = parse_args(argv)
    loaded = [load_gifti_data(p) for p in args.mask]
    fps = {m.get("mesh_fingerprint") for _, m in loaded}
    if len(fps) > 1:
        raise SystemExit(
            f"ffs_util_surfmask: the masks are on different meshes: {sorted(map(str, fps))}"
        )
    means = [load_gifti_data(p)[0] for p in args.mean] if args.mean else None
    if means is not None and len(means) != len(loaded):
        raise SystemExit(f"ffs_util_surfmask: {len(means)} -mean for {len(loaded)} -mask")
    mask, meanall, clip = combine_run_masks([d for d, _ in loaded], means, args.clfrac)
    meta = {k: v for k, v in loaded[0][1].items() if k not in ("source",)}
    meta.update(runs=str(len(loaded)), clip_level=f"{clip:g}")
    stem = _stem(args.prefix)
    save_gifti_data(f"{stem}.mask.shape.gii", mask.astype(np.float32), meta, time_series=False)
    if meanall is not None:
        save_gifti_data(f"{stem}.meanall.shape.gii", meanall, meta, time_series=False)
    inter = np.stack([np.asarray(d).reshape(-1) > 0 for d, _ in loaded]).all(axis=0)
    print(
        f"ffs_util_surfmask: {len(loaded)} runs, {int(inter.sum())} vertices in every run's "
        f"mask, {int(mask.sum())} after the clip at {clip:g} -> {stem}.mask.shape.gii"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
