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

THE NULL IS NOT SPATIALLY CONSTANT, and a pooled threshold would therefore be a map of
where the null is largest.  Three answers, all written out, in increasing honesty and
cost:

  p_emp            against the null columns in the same quantile bin of -strata_by
                   (default baseline_peak, 4 bins), which removes the dependence that
                   is predictable from the column's own response size.  One null run.
  p_fwer           against the distribution of the MAXIMUM PER PERMUTATION -- the
                   family-wise claim, and the only one that needs no assumption about how
                   the null varies.  One null run gives ONE draw of that maximum, so
                   p_fwer floors at 1/2: exceeding a single observed maximum is a ~50%
                   family-wise error rate, not 5%, and 19 permutations are the minimum
                   that can reach 0.05.  The null must also search the SAME columns, since
                   a maximum over a smaller search space is smaller.
  paired_excess    this column's evidence minus the SAME column's evidence under the
                   null run, which assumes nothing at all about spatial structure but is
                   one noisy draw per column.  n permutations turn it into a per-column
                   test, p = (1 + #{own null >= real}) / (1 + n), at n times the cost.

The bin-by-bin null medians are printed, so whether pooling would have been safe is a
measurement you can read rather than a hope.  The columns overlap by construction, so no
vertex-wise correction applies to any of these numbers.
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
    p.add_argument("-strata_by", default="baseline_peak", metavar="FIELD",
                   help="Compare each real column against the null columns in its own "
                   "quantile bin of this field, not against all of them.  The null is not "
                   "spatially constant -- evidence under permuted labels grows with response "
                   "size and with how many trials actually drove the column -- so a pooled "
                   "threshold preferentially flags wherever the null is largest.  'none' "
                   "pools.")  # fmt: skip
    p.add_argument("-n_strata", type=int, default=4, metavar="N",
                   help="Quantile bins for -strata_by.  More bins track the spatial "
                   "dependence better and coarsen the tail: with n null columns a bin "
                   "resolves p_emp no finer than N/n.")  # fmt: skip
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
    # One maximum PER PERMUTATION is the max-statistic distribution; the maximum over a
    # pooled bag of null columns is a single draw from it, and exceeding a single draw is
    # a ~50% false-positive rate, not 5%.
    per_run_max: list[float] = []
    per_run_n: list[int] = []
    for path in a.null:
        _, keep = _read(path, a.gate_baseline)
        null.extend(keep)
        vals = [v for r in keep if not math.isnan(v := _num(r, "logBF_vs_null"))]
        if vals:
            per_run_max.append(max(vals))
            per_run_n.append(len(vals))
    if not null:
        print(f"{a.null}: no null column passes the gate", file=sys.stderr)
        return 1
    print(f"{len(real)} real columns vs {len(null)} null columns "
          f"(from {len(a.null)} null run(s)), gate baseline_peak >= {a.gate_baseline:g}%")  # fmt: skip

    # Bin edges from the NULL run's own quantiles of -strata_by, so a real column is
    # compared with null columns that look like it rather than with all of cortex.
    use_strata = a.strata_by.lower() != "none" and a.n_strata > 1
    edges: list[float] = []
    if use_strata:
        key = sorted(v for r in null if not math.isnan(v := _num(r, a.strata_by)))
        if not key:
            print(f"-strata_by {a.strata_by}: not in the null table", file=sys.stderr)
            return 1
        edges = [key[int(i * len(key) / a.n_strata)] for i in range(1, a.n_strata)]

    def stratum(row):
        x = _num(row, a.strata_by) if use_strata else math.nan
        return 0 if math.isnan(x) else sum(1 for e in edges if x >= e)

    n_bins = len(edges) + 1
    dists = {(s, b): sorted(v for r in null if stratum(r) == b
                            and not math.isnan(v := _num(r, s)))
             for s in stats for b in range(n_bins)}  # fmt: skip

    # Whether pooling was safe is a measurement, not an assumption: if the null's own
    # median moves across the bins, a pooled threshold is a map of the null rather than
    # of the effect.  Printed so -strata_by can be checked against the data.
    print(f"\n  null distribution by {a.strata_by if use_strata else 'nothing'} "
          f"({n_bins} bin(s)):")  # fmt: skip
    for b in range(n_bins):
        d = dists[("logBF_vs_null", b)]
        if not d:
            print(f"    bin {b}: empty")
            continue
        lo = "-inf" if b == 0 else f"{edges[b - 1]:.2f}"
        hi = "inf" if b == n_bins - 1 else f"{edges[b]:.2f}"
        print(f"    bin {b} ({a.strata_by} {lo}..{hi}): n {len(d):4d}  logBF median "
              f"{d[len(d) // 2]:7.2f}  95th {d[min(len(d) - 1, int(0.95 * len(d)))]:7.2f}  "
              f"max {d[-1]:7.2f}")  # fmt: skip
    for s in stats:
        rq = sorted(v for r in real if not math.isnan(v := _num(r, s)))
        pooled = sorted(v for r in null if not math.isnan(v := _num(r, s)))
        print(f"  {s:22s} null median {pooled[len(pooled) // 2]:8.3f}  "
              f"max {pooled[-1]:8.3f}   | real median {rq[len(rq) // 2]:8.3f}")  # fmt: skip

    def p_emp(stat, x, b):
        d = dists[(stat, b)] or dists[(stat, 0)]
        if math.isnan(x) or not d:
            return math.nan
        # (1 + #{null >= x}) / (1 + n): never 0, because one permutation cannot license a
        # rate finer than its own resolution.
        lo, hi = 0, len(d)
        while lo < hi:  # first index with d[mid] >= x
            mid = (lo + hi) // 2
            if d[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        return (1 + (len(d) - lo)) / (1 + len(d))

    # The same column's OWN null fit, where the null run fitted it: one draw of its own
    # null, free of any assumption that the null is spatially constant.  One permutation
    # makes this noisy per column but unbiased; several make it a per-column test.
    paired = {r["column"]: _num(r, "logBF_vs_null") for r in null}
    maxima = sorted(per_run_max)

    def p_fwer(x):
        # (1 + #{permutation maxima >= x}) / (1 + n_permutations).  With one null run the
        # floor is 1/2: one draw of the max cannot license any family-wise rate.  n = 19
        # is the smallest n that can reach 0.05.
        if math.isnan(x) or not maxima:
            return math.nan
        return (1 + sum(1 for m in maxima if m >= x)) / (1 + len(maxima))

    floor = 1 / (1 + len(maxima))
    print(f"\n  max-statistic distribution: {len(maxima)} permutation(s), "
          f"maxima {', '.join(f'{m:.1f}' for m in maxima)}")  # fmt: skip
    print(f"    p_fwer cannot go below {floor:.3g} with {len(maxima)} permutation(s)"
          + ("; 19 are needed to reach 0.05" if floor > 0.05 else ""))  # fmt: skip
    if per_run_n and abs(sum(per_run_n) / len(per_run_n) - len(real)) > 0.25 * len(real):
        print(f"    WARNING: the null searched {sum(per_run_n) // len(per_run_n)} columns "
              f"and the real run {len(real)}; a maximum over a smaller search is smaller, "
              "so p_fwer here is anti-conservative.  Fit the null on the same columns.",
              file=sys.stderr)  # fmt: skip
    with open(a.out, "w") as fh:
        fh.write("column\tp_emp\tp_fwer\tnull_logBF_this_column\tpaired_excess\t"
                 + "\t".join(f"p_emp_{d}" for d in depths) + "\n")  # fmt: skip
        for r in real_all:
            if r not in real:  # gated out: no calibration, not a passing 0
                fh.write(f"{r['column']}\tnan\tnan\tnan\tnan\t"
                         + "\t".join("nan" for _ in depths) + "\n")  # fmt: skip
                continue
            b, bf = stratum(r), _num(r, "logBF_vs_null")
            own = paired.get(r["column"], math.nan)
            fh.write(f"{r['column']}\t{p_emp('logBF_vs_null', bf, b):.4g}\t"
                     f"{p_fwer(bf):.4g}\t{own:.4g}\t{bf - own:.4g}\t"
                     + "\t".join(f"{p_emp(f'p_mod_{d}', _num(r, f'p_mod_{d}'), b):.4g}"
                                 for d in depths) + "\n")  # fmt: skip
    n_sig = sum(1 for r in real
                if p_emp("logBF_vs_null", _num(r, "logBF_vs_null"), stratum(r)) <= 0.05)  # fmt: skip
    n_fwer = sum(1 for r in real if p_fwer(_num(r, "logBF_vs_null")) <= max(0.05, floor))
    both = [r for r in real if r["column"] in paired]
    n_paired = sum(1 for r in both if _num(r, "logBF_vs_null") > paired[r["column"]])
    print(f"\n  p_emp <= 0.05: {n_sig} of {len(real)} gated columns")
    print(f"  p_fwer <= {max(0.05, floor):.3g} (the best this many permutations can "
          f"claim): {n_fwer}")  # fmt: skip
    if both:
        print(f"  beating their OWN null fit: {n_paired} of {len(both)} columns the null "
              f"run also fitted ({100 * n_paired / len(both):.0f}%; 50% is chance)")  # fmt: skip
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
