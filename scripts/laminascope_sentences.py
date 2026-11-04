#!/usr/bin/env python
"""Turn an ffs_laminascope per-column TSV into one plain sentence per believable column.

Why this exists: the maps and the TSV hold a whole model space per column, and a reader
who wants "these conditions differed HERE, at THAT depth, by THIS much, and here is how
sure" has to assemble it from eight numbers.  This writes that sentence, for the columns
that pass the same gate the maps' `gate` brick applies, sorted by evidence.

    python scripts/laminascope_sentences.py laminascope/ss_gender_vs_fixate_columns.tsv \
        -comparison "gender vs fixate" -gate_baseline 0.5 -p_depth 0.75

Writes <input stem>_readable.tsv (and prints the first few).  Nothing is refit and nothing
is inferred that is not in the table: it is a rendering, so a sentence it writes is exactly
as trustworthy as the fit behind it.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="laminascope_sentences", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("table", help="{prefix}_{comparison}_columns.tsv from the run.")
    p.add_argument("-comparison", default=None, help="How to name the contrast in the sentence.")
    p.add_argument("-gate_baseline", type=float, default=0.5, metavar="PCT",
                   help="Peak baseline response below which a column is not reported.")  # fmt: skip
    p.add_argument("-p_depth", type=float, default=0.75, metavar="P",
                   help="P(depth modulated) at or above which a depth is called modulated.")  # fmt: skip
    p.add_argument("-min_logbf", type=float, default=3.0, metavar="NATS",
                   help="Evidence over the null below which a column is not reported.")  # fmt: skip
    p.add_argument("-out", default=None, help="Output TSV (default: input with _readable).")
    return p.parse_args(argv)


def _num(row, key):
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def main(argv=None) -> int:
    a = parse_args(argv)
    rows = list(csv.DictReader(open(a.table), delimiter="\t"))
    if not rows:
        print(f"{a.table}: empty", file=sys.stderr)
        return 1
    depths = [k[len("p_mod_"):] for k in rows[0] if k.startswith("p_mod_")]  # fmt: skip
    if not depths:
        print(f"{a.table}: no p_mod_<depth> columns; is this a _columns.tsv?", file=sys.stderr)
        return 1
    cmp_name = a.comparison or "the comparison"

    def passes(r):
        return (_num(r, "baseline_peak") >= a.gate_baseline
                and _num(r, "nesting_nats") <= 3.0
                and _num(r, "in_fov") >= 0.9
                and r.get("stalled") == "0" and r.get("failed") == "0"
                and not (_num(r, "logBF_vs_null") < a.min_logbf))  # fmt: skip

    keep = sorted((r for r in rows if passes(r)), key=lambda r: -_num(r, "logBF_vs_null"))
    out = a.out or a.table.replace("_columns.tsv", "") + "_readable.tsv"
    with open(out, "w") as fh:
        fh.write("column\themi\tcentre_vertex\tpeak_pct\tevidence_nats\tdepths_modulated\t"
                 "B_Hz\tdirection\tbest_model\tp_best\tsentence\n")  # fmt: skip
        for r in keep:
            mod = [d for d in depths if _num(r, f"p_mod_{d}") >= a.p_depth]
            bs = [_num(r, f"B_{d}") for d in mod]
            signs = {b > 0 for b in bs if b != 0}
            # B modulates self-inhibition (A_kk = -sigma + u*B), so positive B lowers it.
            direction = ("both ways across depths" if len(signs) > 1 else
                         "less self-inhibition, i.e. a larger neuronal response" if signs == {True}
                         else "more self-inhibition, i.e. a smaller neuronal response"
                         if signs == {False} else "no modulation at the reported depths")  # fmt: skip
            where = "+".join(mod) if mod else f"no depth above P {a.p_depth:g}"
            blist = ", ".join(f"{b:+.2f}" for b in bs) or "n/a"
            sentence = (
                f"{cmp_name} differed here (peak {_num(r, 'baseline_peak'):.2f}%, "
                f"{_num(r, 'logBF_vs_null'):.0f} nats over no difference) at {where}; "
                f"B {blist} Hz, {direction}; best single model {r.get('best', '?')} "
                f"(p_best {_num(r, 'p_best'):.2f})"
            )
            fh.write(f"{r['column']}\t{r.get('hemi', '')}\t{r.get('centre_vertex', '')}\t"
                     f"{_num(r, 'baseline_peak'):.2f}\t{_num(r, 'logBF_vs_null'):.1f}\t{where}\t"
                     f"{blist}\t{direction}\t{r.get('best', '')}\t"
                     f"{_num(r, 'p_best'):.2f}\t{sentence}\n")  # fmt: skip
    print(f"{len(keep)} of {len(rows)} columns pass; wrote {out}")
    for r in keep[:3]:
        print("  " + open(out).read().splitlines()[keep.index(r) + 1].split("\t")[-1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
