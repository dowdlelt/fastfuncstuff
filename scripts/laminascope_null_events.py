#!/usr/bin/env python
"""Write permuted events that turn a real comparison into a matched NULL comparison.

Why this exists: ffs_laminascope's model space asks WHICH DEPTHS are modulated, and its
``null`` model is "no modulation at any depth" -- a model, fitted to the same data, not a
null DISTRIBUTION.  Its evidence therefore says "something differs", which on real columns
is almost always true (median 24.5 nats over 903 columns), and gives no false-positive
rate.  The only honest calibration is to run the identical pipeline on data where the two
sides are known not to differ, and the cheapest such data is the same data with the
contrast labels permuted WITHIN each stimulus stratum.

Each side keeps its exact trial count and stimulus composition -- evidence grows with N,
so an unmatched null is anti-conservative -- and every other condition is left alone, so
the design the columns are fitted in is the same one.

    python scripts/laminascope_null_events.py -events stimuli/*_events.tsv \\
        -pool gender fixate -strata stim_contrast -outdir null_events -seed 0

Prints the -event_cols and the comparison to run with.  Permuting labels does NOT
permute the data: autocorrelation, drift and motion stay exactly where they were, which
is the point -- the null inherits every nuisance the real fit had.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="laminascope_null_events", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("-events", nargs="+", required=True, metavar="TSV",
                   help="The run's events files, in the order given to the tool.")  # fmt: skip
    p.add_argument("-pool", nargs="+", required=True, metavar="TASK",
                   help="Task labels (task-<X> in the filename) whose trials are pooled and "
                   "split in half.  Give the two sides of the comparison you are calibrating.")  # fmt: skip
    p.add_argument("-strata", nargs="*", default=["stim_contrast"], metavar="COL",
                   help="Event columns to balance the split within, so each null side holds "
                   "the same stimuli as the other.  Empty balances only within task and run.")  # fmt: skip
    p.add_argument("-col", default="nullcond", metavar="NAME",
                   help="Name of the condition column written into the output events.")  # fmt: skip
    p.add_argument("-labels", nargs=2, default=("nullA", "nullB"), metavar="NAME",
                   help="The two null side labels.")  # fmt: skip
    p.add_argument("-seed", type=int, default=0, help="Permutation seed.")
    p.add_argument("-outdir", required=True, help="Where to write the rewritten events.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    pool = set(a.pool)
    counts: dict[str, int] = {}
    written = []
    for path in a.events:
        m = re.search(r"task-([A-Za-z0-9]+)", Path(path).name)
        if m is None:
            print(f"{path}: no task-<label> in the filename", file=sys.stderr)
            return 1
        task = m.group(1)
        with open(path) as fh:
            rd = csv.DictReader(fh, delimiter="\t")
            rows, fields = list(rd), list(rd.fieldnames or [])
        missing = [c for c in a.strata if c not in fields]
        if missing:
            print(f"{path}: no column(s) {missing}", file=sys.stderr)
            return 1
        labels = [""] * len(rows)
        if task in pool:
            # Balance within every stratum of this run, so each side gets half of each
            # stimulus class here rather than half on average over the session.
            keys = [tuple(r[c] for c in a.strata) for r in rows]
            for key in sorted(set(keys)):
                idx = [i for i, k in enumerate(keys) if k == key]
                perm = rng.permutation(len(idx))
                for j, i in enumerate(idx):
                    labels[i] = a.labels[int(perm[j]) % 2]
        else:
            for i, r in enumerate(rows):
                labels[i] = "_".join([task] + [r[c] for c in a.strata])
        for lab in labels:
            counts[lab] = counts.get(lab, 0) + 1
        dest = out / Path(path).name
        with open(dest, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=[*fields, a.col], delimiter="\t",
                               lineterminator="\n")  # fmt: skip
            w.writeheader()
            for r, lab in zip(rows, labels, strict=True):
                w.writerow({**r, a.col: lab})
        written.append(str(dest))
    print(f"wrote {len(written)} events file(s) to {out}/")
    for lab in sorted(counts):
        print(f"  {lab:24s} {counts[lab]:5d} events")
    n1, n2 = counts.get(a.labels[0], 0), counts.get(a.labels[1], 0)
    if n1 != n2:
        print(f"  NOTE: sides differ ({n1} vs {n2}); evidence scales with N, so this null "
              "is not matched", file=sys.stderr)  # fmt: skip
    print(f"\nrun the tool with:  -events {out}/... -event_cols onset duration {a.col}"
          " (drop -event_prefix)")  # fmt: skip
    print(f"and the comparison:  -compare null {a.labels[0]} {a.labels[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
