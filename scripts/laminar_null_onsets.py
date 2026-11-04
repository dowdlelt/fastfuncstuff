#!/usr/bin/env python
"""Write N permuted onset sets for an ffs_laminar permutation null, and the deconvolve loop.

Why upstream: ffs_laminar's input is already-deconvolved ROI x depth curves, and two
curves have nothing left to permute.  The exchangeable unit is the TRIAL, so the
permutation has to happen at the onsets and the deconvolution has to be redone -- which
is cheap in ROI x depth mode and is why a rigorous null is affordable for a few ROIs and
not for a few thousand columns.

    python scripts/laminar_null_onsets.py -onsets std.1D dev.1D -n 100 -outdir perms
    # then, per replicate (the script prints this loop):
    ffs_deconvolve -input bold.nii.gz -onsets perms/perm042_std.1D perms/perm042_dev.1D \
        -model TENT -window 0 14 -atlas V1.lh.label -fs_subj $SUBJ -depths 7 9 11 \
        -equivolume -prefix perms/perm042
    ffs_laminar -fit_prefix fit/sub01 -conds std dev -duration 1.6 \
        -null_fit_prefix perms/perm*/... -prefix lam/sub01 -plots

The onsets are POOLED and redealt, keeping each condition's trial count exactly (evidence
grows with N, so an unmatched null is anti-conservative).  Timing structure is otherwise
untouched: every onset that existed still exists, so the design's efficiency, its
collinearity with drift and its autocorrelation all survive -- only which condition a
trial belongs to is randomised.

AFNI 1D onset files (one row per run, times in seconds) and BIDS events TSVs are both
read; a TSV keeps every other column and rewrites `trial_type`.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="laminar_null_onsets", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("-onsets", nargs="+", required=True, metavar="FILE",
                   help="The real onsets: one .1D per condition (the order -conds uses), or "
                   "one events TSV per run.")  # fmt: skip
    p.add_argument("-conds", nargs="*", default=None, metavar="NAME",
                   help="For TSV input, the two trial_type values to permute.  Other trial "
                   "types are left alone, so the design keeps its other conditions.")  # fmt: skip
    p.add_argument("-n", type=int, default=100, metavar="N",
                   help="Replicates.  19 is the fewest that can express p = 0.05; 100 makes "
                   "the distribution readable and 1000 resolves a tail.")  # fmt: skip
    p.add_argument("-seed", type=int, default=0)
    p.add_argument("-outdir", required=True)
    return p.parse_args(argv)


def _permute_1d(paths, rng, out, tag):
    """Pool the onset times of each run across conditions and redeal them."""
    per_cond = [[line.split() for line in open(q).read().splitlines()] for q in paths]
    n_runs = max(len(c) for c in per_cond)
    dealt: list[list[list[str]]] = [[] for _ in paths]
    for r in range(n_runs):
        rows = [c[r] if r < len(c) else [] for c in per_cond]
        pool = [t for row in rows for t in row]
        counts = [len(row) for row in rows]
        order = rng.permutation(len(pool))
        at = 0
        for i, k in enumerate(counts):
            take = sorted(float(pool[j]) for j in order[at:at + k])  # fmt: skip
            dealt[i].append([f"{t:g}" for t in take])
            at += k
    written = []
    for i, q in enumerate(paths):
        dest = out / f"{tag}_{Path(q).name}"
        dest.write_text("\n".join(" ".join(row) if row else "*" for row in dealt[i]) + "\n")
        written.append(str(dest))
    return written


def _permute_tsv(paths, conds, rng, out, tag):
    """Redeal `trial_type` among the named conditions, within each run."""
    written = []
    for q in paths:
        rows = list(csv.DictReader(open(q), delimiter="\t"))
        fields = list(rows[0]) if rows else []
        if "trial_type" not in fields:
            print(f"{q}: no trial_type column", file=sys.stderr)
            raise SystemExit(1)
        idx = [i for i, r in enumerate(rows) if r["trial_type"] in conds]
        labels = [rows[i]["trial_type"] for i in idx]
        perm = rng.permutation(len(idx))
        for j, i in enumerate(idx):
            rows[i] = {**rows[i], "trial_type": labels[int(perm[j])]}
        dest = out / f"{tag}_{Path(q).name}"
        with open(dest, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        written.append(str(dest))
    return written


def main(argv=None) -> int:
    a = parse_args(argv)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    is_tsv = all(q.endswith((".tsv", ".txt")) and "events" in Path(q).name for q in a.onsets)
    if is_tsv and not a.conds:
        print("-conds is needed for events TSVs: which two trial_types to permute",
              file=sys.stderr)  # fmt: skip
        return 1
    if not is_tsv and len(a.onsets) != 2:
        print(f"-onsets takes two .1D files (one per condition), got {len(a.onsets)}",
              file=sys.stderr)  # fmt: skip
        return 1
    tags = []
    for i in range(a.n):
        tag = f"perm{i:04d}"
        if is_tsv:
            _permute_tsv(a.onsets, set(a.conds or []), rng, out, tag)
        else:
            _permute_1d(a.onsets, rng, out, tag)
        tags.append(tag)
    print(f"wrote {a.n} replicate(s) of {len(a.onsets)} file(s) to {out}/")
    print(f"\n# one deconvolve per replicate, then ONE ffs_laminar:\nfor P in {' '.join(tags)}; do")
    print(f"  ffs_deconvolve ... -onsets {' '.join(f'{out}/$P_{Path(q).name}' for q in a.onsets)}"
          f" ... -prefix {out}/$P\ndone")  # fmt: skip
    print(f"ffs_laminar ... -null_fit_prefix {out}/perm[0-9][0-9][0-9][0-9] ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
