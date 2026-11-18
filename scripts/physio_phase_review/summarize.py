"""Collect small TASKFORCE result records and draw the phase-recovery comparison."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(path.read_text()) for path in sorted(args.input.glob("s*.json"))]
    (args.output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for record in records:
        variants = record["variants"]
        base = variants["current"]
        selected = variants[record["selected_by_prediction"]]
        label = record["run"]
        axes[0].plot(base["lock"], selected["lock"], "o", color="tab:blue")
        axes[0].annotate(
            label,
            (base["lock"], selected["lock"]),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
        names = ["current", "smooth_2s", "spline_2s"]
        axes[1].plot(
            range(3), [variants[name]["lock"] for name in names], "o-", alpha=0.65, label=label
        )
        print(
            label,
            "current",
            round(base["lock"], 4),
            "smooth2",
            round(variants["smooth_2s"]["lock"], 4),
            "fit2",
            round(variants["spline_2s"]["lock"], 4),
            "selected",
            record["selected_by_prediction"],
            round(selected["lock"], 4),
            "median condition",
            round(float(np.median(record["baseline"]["conditioning"])), 2),
        )
    axes[0].plot([0, 1], [0, 1], "--", color="0.6", linewidth=1)
    axes[0].set(
        xlabel="Current phase locking to reviewed pulse peaks",
        ylabel="Phase locking after selection by image prediction",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[1].set(
        xticks=range(3),
        xticklabels=["Current", "Smooth only\n2 s knots", "Direct fit\n2 s knots"],
        ylabel="Phase locking to reviewed pulse peaks",
        ylim=(0, 1),
    )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("TASKFORCE cardiac phase: image-only estimation; recordings used for evaluation")
    fig.savefig(args.output / "comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
