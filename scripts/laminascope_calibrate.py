#!/usr/bin/env python
"""Calibrate an ffs_laminascope comparison against a matched null run: one brick to threshold.

Why this exists: the model space's ``null`` is a MODEL ("no modulation at any depth"),
fitted to the same data, so its Bayes factor says "something differs" and gives no
false-positive rate -- on real columns its median was 24.5 nats against a 3-nat
convention.  A false-positive rate needs data where nothing differs.  Run the identical
pipeline on permuted labels (scripts/laminascope_null_events.py), then this turns the two
runs into the quantity that was missing:

    p_emp = (1 + #{null columns with statistic >= this column's}) / (1 + n_null)

the share of a NULL search over the same cortex that reached this column's evidence.
Threshold it like an uncorrected p-value, because that is what it is -- and read
fwer_logbf, printed here, for the whole-map claim: the largest evidence the null search
produced anywhere.  A column above that is one no permutation of these labels reached.

    python scripts/laminascope_calibrate.py \\
        -real laminascope/ss_gender_vs_fixate_columns.tsv \\
        -null laminascope/nullfit_null_columns.tsv -out laminascope/ss_gender_calibrated.tsv

Writes a TSV keyed by `column`, paintable with scripts/paint_laminascope_table.py -table.
One permutation gives one draw of the null search, so fwer_logbf is an estimate with one
sample; several seeds give the 95th percentile of the max, which is the real FWER
threshold.  The columns overlap by construction, so no vertex-wise correction applies to
either number.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="laminascope_calibrate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("-real", required=True, help="The comparison's _columns.tsv.")
    p.add_argument("-null", nargs="+", required=True, metavar="TSV",
                   help="One or more null runs' _columns.tsv (more seeds, better tail).")  # fmt: skip
    p.add_argument("-gate_baseline", type=float, default=0.5, metavar="PCT",
                   help="Columns below this peak response are excluded from BOTH "
                   "distributions: a column with no response is not a test that could "
                   "have gone either way.")  # fmt: skip
    p.add_argument("-out", required=True, help="Output TSV.")
    return p.parse_args(argv)


def _num(row, key):
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return math.nan


def _read(path, gate_baseline):
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    keep = [r for r in rows
            if _num(r, "baseline_peak") >= gate_baseline
            and _num(r, "nesting_nats") <= 3.0 and _num(r, "in_fov") >= 0.9
            and r.get("stalled") == "0" and r.get("failed") == "0"]  # fmt: skip
    return rows, keep


def main(argv=None) -> int:
    a = parse_args(argv)
    real_all, real = _read(a.real, a.gate_baseline)
    if not real:
        print(f"{a.real}: no column passes the response/soundness gate", file=sys.stderr)
        return 1
    depths = [k[len("p_mod_"):] for k in real[0] if k.startswith("p_mod_")]  # fmt: skip
    stats = ["logBF_vs_null"] + [f"p_mod_{d}" for d in depths]

    null: list[dict] = []
    for path in a.null:
        _, keep = _read(path, a.gate_baseline)
        null.extend(keep)
    if not null:
        print(f"{a.null}: no null column passes the gate", file=sys.stderr)
        return 1
    dists = {s: sorted(_num(r, s) for r in null if not math.isnan(_num(r, s))) for s in stats}
    print(f"{len(real)} real columns vs {len(null)} null columns "
          f"(from {len(a.null)} null run(s)), gate baseline_peak >= {a.gate_baseline:g}%")  # fmt: skip
    for s in stats:
        d = dists[s]
        q = [d[min(len(d) - 1, int(f * len(d)))] for f in (0.5, 0.95, 0.99)]
        rq = sorted(_num(r, s) for r in real if not math.isnan(_num(r, s)))
        print(f"  {s:22s} null median {q[0]:8.3f}  95th {q[1]:8.3f}  99th {q[2]:8.3f}  "
              f"max {d[-1]:8.3f}   | real median {rq[len(rq) // 2]:8.3f}")  # fmt: skip

    def p_emp(stat, x):
        d = dists[stat]
        if math.isnan(x):
            return math.nan
        # (1 + #{null >= x}) / (1 + n): never 0, because one permutation cannot license
        # a rate below its own resolution.
        lo, hi = 0, len(d)
        while lo < hi:  # first index with d[mid] >= x
            mid = (lo + hi) // 2
            if d[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        return (1 + (len(d) - lo)) / (1 + len(d))

    fwer = {s: dists[s][-1] for s in stats}
    with open(a.out, "w") as fh:
        fh.write("column\tp_emp\tbeats_null_max\t"
                 + "\t".join(f"p_emp_{d}" for d in depths) + "\n")  # fmt: skip
        for r in real_all:
            if r not in real:  # gated out: no calibration, not a passing 0
                fh.write(f"{r['column']}\tnan\tnan\t" + "\t".join("nan" for _ in depths) + "\n")
                continue
            bf = _num(r, "logBF_vs_null")
            fh.write(f"{r['column']}\t{p_emp('logBF_vs_null', bf):.4g}\t"
                     f"{int(bf > fwer['logBF_vs_null'])}\t"
                     + "\t".join(f"{p_emp(f'p_mod_{d}', _num(r, f'p_mod_{d}')):.4g}"
                                 for d in depths) + "\n")  # fmt: skip
    n_sig = sum(1 for r in real if p_emp("logBF_vs_null", _num(r, "logBF_vs_null")) <= 0.05)
    n_fwer = sum(1 for r in real if _num(r, "logBF_vs_null") > fwer["logBF_vs_null"])
    print(f"\n  p_emp <= 0.05: {n_sig} of {len(real)} gated columns")
    print(f"  above the null search's maximum ({fwer['logBF_vs_null']:.1f} nats): {n_fwer}")
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
