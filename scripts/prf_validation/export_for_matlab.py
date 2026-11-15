"""Write the vertices ffs_pyrf's screen passed, for analyzePRF to fit.

analyzePRF costs seconds per vertex, so it gets only what ffs_pyrf's linear
screen kept (cross-validated screening R2 above --screen). Both hemispheres go
in one array, lh first, with the vertex indices saved alongside so compare.py
can put the results back on the mesh.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio

from fastfuncstuff.io.gifti import load_gifti_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument(
        "screen_prefix", help="ffs_pyrf prefix whose _screen.{lh,rh}.func.gii to read"
    )
    parser.add_argument("--screen", type=float, default=0.1, help="screening R2 threshold")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="first N vertices only (timing probe)")
    args = parser.parse_args()

    meta = json.loads((args.work / "meta.json").read_text())
    index = {}
    for hemi in ("lh", "rh"):
        screen, _ = load_gifti_data(f"{args.screen_prefix}_screen.{hemi}.func.gii")
        screen = np.asarray(screen).reshape(-1)
        index[hemi] = np.flatnonzero(np.nan_to_num(screen, nan=-np.inf) > args.screen)
        print(
            f"{hemi}: {index[hemi].size} of {screen.size} vertices pass screen R2 > {args.screen}"
        )

    data = []
    for run in meta["runs"]:
        parts = []
        for hemi in ("lh", "rh"):
            series, _ = load_gifti_data(args.work / "runs" / f"run-{run:02d}.{hemi}.func.gii")
            parts.append(np.asarray(series, np.float32)[index[hemi]])
        data.append(np.concatenate(parts))
    if args.limit:
        data = [d[: args.limit] for d in data]

    # Same rule as analyzePRF's own default, written out so ffs uses it too.
    maxpolydeg = round(meta["n_frames"] * meta["tr"] / 60 / 2)
    cell = np.empty((len(data), 1), dtype=object)
    for i, d in enumerate(data):
        cell[i, 0] = d
    out = args.out or args.work / "apr_input.mat"
    sio.savemat(
        out,
        {
            "stimulus": np.load(args.work / "aperture.npy"),
            "data": cell,
            "tr": float(meta["tr"]),
            "maxpolydeg": float(maxpolydeg),
            "lh_index": index["lh"] + 1,  # one-based, for MATLAB readers
            "rh_index": index["rh"] + 1,
            "screen_threshold": args.screen,
        },
        do_compression=False,
    )
    print(f"wrote {out}: {data[0].shape[0]} vertices x {len(data)} runs, maxpolydeg {maxpolydeg}")


if __name__ == "__main__":
    main()
