"""Apply candidate acceptance rules to cm_detect JSONL (no image access)."""

import json
import sys

import numpy as np

RULES = {
    # alias statistic used for selection AND (maximised over aliases) for the null
    "a1": lambda a: a["a1"],
    "a1+a2": lambda a: a["a1"] + a["a2"],
}


def decide(rec, stat):
    best = []
    for s in rec["starts"]:
        a = max(s["aliases"], key=stat)
        null = np.array([max(stat(r) for r in t) for t in s["null"]]) if s["null"] else np.zeros(1)
        best.append((s, a, null))
    pulsed = [b for b in best if b[1]["a2"] > b[1]["controls"]] or best
    s, a, null = max(pulsed, key=lambda b: stat(b[1]) - b[2].max())
    floor = max(b[2].max() for b in best)  # Bonferroni: every start's shuffles
    found = bool(stat(a) > floor and a["a2"] > a["controls"])
    return dict(
        found=found,
        start=s["label"],
        rate=a["rate"],
        lock=a["lock"],
        wlock=a.get("wlock", np.nan),
        stat=stat(a),
        floor=float(floor),
        a2=a["a2"],
        controls=a["controls"],
    )


def main():
    recs = [json.loads(ln) for path in sys.argv[1:] for ln in open(path) if ln.strip()]
    for name, stat in RULES.items():
        print(f"== rule {name}")
        tally = dict(right=0, wrong=0, miss_good=0, miss=0)
        for rec in recs:
            d = decide(rec, stat)
            good = d["lock"] > 0.5
            key = (
                ("right" if good else "wrong") if d["found"] else ("miss_good" if good else "miss")
            )
            tally[key] += 1
            print(
                f"  {rec['run'][-34:]:34s} TR {rec['tr']:<5} ref {rec['ref_hr']:5.1f} "
                f"{'FOUND' if d['found'] else 'no   '} {d['rate']:5.1f} lock {d['lock']:.2f} "
                f"stat {d['stat']:.3f}/{d['floor']:.3f} h {d['a2']:.3f}/{d['controls']:.3f}"
            )
        print("  ", tally)


if __name__ == "__main__":
    main()
