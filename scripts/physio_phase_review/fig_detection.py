"""docs/physio/fig6_detection.png: cardiac detection against recordings, per dataset.

Reads the production-detector results (verify_detect JSONL: found, lock, rate, ref)
and, where the same runs were scored by the previous path, its results too.
Right = found and phase-locked to the recording (lock > 0.5); wrong = found, not locked.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RIGHT, WRONG, MISS, INK, MUTED = "#2f7fd6", "#d64541", "#9aa1aa", "#222222", "#666666"

GROUPS = [  # (label, predicate on run name), drawn top to bottom
    ("TASKFORCE, TR 1 s, MB5 (dev)", lambda r: r.startswith("tf.")),
    ("ds002547, TR 2 s, sequential (dev)", lambda r: r.startswith("on.ds002547")),
    (
        "ds004645/ds006947, TR 0.23/0.56 s (dev)",
        lambda r: r.startswith(("on.ds004645", "on.ds006947")),
    ),  # fmt: skip
    ("ds002547, TR 2 s, sequential (held-out)", lambda r: r.startswith("ho.ds002547")),
    ("ds002603, TR 0.55-2 s, 4 sequences (held-out)", lambda r: r.startswith("ho.ds002603")),
    ("Rockland, TR 0.65/1.4/2.5 s (held-out)", lambda r: "ockland" in r),
    ("ds001168 full brain, TR 3 s (fresh)", lambda r: "ds001168" in r and "fullbrain" in r),
    ("ds001168 prefrontal slab, TR 4 s (fresh)", lambda r: "ds001168" in r and "prefrontal" in r),
]


def outcome(rec, key="found"):
    if not rec.get(key):
        return "miss"
    return "right" if rec["lock"] > 0.5 else "wrong"


def load(paths):
    out = {}
    for path in paths:
        for ln in Path(path).read_text().splitlines():
            if ln.strip():
                rec = json.loads(ln)
                out[rec["run"]] = rec
    return out


def bar(ax, y, counts, height):
    """Stacked percent-of-runs bar, each segment labelled with its run count."""
    total = sum(counts.values())
    left = 0.0
    for key, colour in (("right", RIGHT), ("wrong", WRONG), ("miss", MISS)):
        n = counts[key]
        width = 100 * n / total
        if n:
            ax.barh(y, width - 0.4, left=left, height=height, color=colour, edgecolor="none")
            if width >= 5 or key == "wrong":
                ax.text(left + width / 2, y, str(n), ha="center", va="center", fontsize=8.5,
                        color="white" if key != "miss" else INK)  # fmt: skip
        left += width


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new", nargs="+", required=True)
    ap.add_argument("--old", nargs="*", default=[])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    new, old = load(args.new), load(args.old)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 6.2), width_ratios=[1.7, 1],
                                 sharey=True, constrained_layout=True)  # fmt: skip
    ticks, labels = [], []
    y = 0.0
    for label, keep in GROUPS:
        runs = sorted(r for r in new if keep(r))
        if not runs:
            continue
        counts = {k: sum(outcome(new[r]) == k for r in runs) for k in ("right", "wrong", "miss")}
        paired = [r for r in runs if r in old]
        if paired:
            prev = {
                k: sum(outcome(old[r]) == k for r in paired) for k in ("right", "wrong", "miss")
            }
            bar(ax, y + 0.22, counts, 0.4)
            bar(ax, y - 0.22, prev, 0.4)
            scored = "" if len(paired) == len(runs) else f", {len(paired)} of {len(runs)} runs"
            ax.text(101.5, y + 0.22, "now", va="center", fontsize=8, color=MUTED)
            ax.text(101.5, y - 0.22, f"previous detector{scored}", va="center",
                    fontsize=8, color=MUTED)  # fmt: skip
        else:
            bar(ax, y, counts, 0.5)
        locks = [new[r]["lock"] for r in runs if new[r]["found"]]
        jitter = np.random.default_rng(0).uniform(-0.18, 0.18, len(locks))
        colours = [RIGHT if v > 0.5 else WRONG for v in locks]
        bx.scatter(locks, y + jitter, s=16, c=colours, alpha=0.8, edgecolors="none")
        if locks:
            bx.plot([np.median(locks)] * 2, [y - 0.3, y + 0.3], color=INK, lw=2)
        ticks.append(y)
        labels.append(f"{label}\n{len(runs)} runs")
        y -= 1.25
    ax.set_yticks(ticks, labels, fontsize=8.5)
    ax.set_xlabel("% of runs (segments labelled with run counts)")
    ax.set_title("A  Heartbeat found from the images", loc="left", fontweight="bold")
    for colour, name in ((RIGHT, "right"), (WRONG, "wrong"), (MISS, "not found")):
        ax.barh(np.nan, 0, color=colour, label=name)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.09), ncol=3, frameon=False, fontsize=9)
    ax.set_xlim(-0.5, 135)
    ax.set_xticks([0, 25, 50, 75, 100])
    bx.set_xlim(0, 1.02)
    bx.axvline(0.5, color=MUTED, lw=0.8, ls=":")
    bx.set_xlabel("lock to the recorded pulse (1 = exact)")
    bx.set_title("B  How well each find locks", loc="left", fontweight="bold")
    for a in (ax, bx):
        a.grid(axis="x", color="#e6e6e6", lw=0.8)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.savefig(args.output, dpi=110)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
