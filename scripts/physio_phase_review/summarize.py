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
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), width_ratios=(1.7, 1), layout="constrained")
    valid = [record for record in records if record["run"] != "s02_mvpsA_04"]
    names = ["current", "smooth_2s", "spline_2s"]
    labels = ["Current detector phase", "Smoothing only (2 s)", "Direct fit (2 s)"]
    for row, record in enumerate(valid):
        values = [record["variants"][name]["lock"] for name in names]
        axes[0].plot([min(values), max(values)], [row, row], color="0.75", linewidth=2)
    for name, label, colour, marker in zip(
        names, labels, ("0.4", "tab:orange", "tab:blue"), ("o", "^", "o"), strict=True
    ):
        axes[0].scatter(
            [record["variants"][name]["lock"] for record in valid],
            np.arange(len(valid)),
            label=label,
            color=colour,
            marker=marker,
            s=45,
            zorder=3,
        )
    wrong = next((record for record in records if record["run"] == "s02_mvpsA_04"), None)
    if wrong is not None:
        for offset, name, colour in ((-0.18, "current", "0.4"), (0.18, "spline_2s", "tab:blue")):
            axes[1].bar(
                np.arange(2) + offset,
                [wrong["variants"][name][key] for key in ("prediction", "lock")],
                width=0.36,
                color=colour,
            )
    for record in records:
        variants = record["variants"]
        base = variants["current"]
        selected = variants[record["selected_by_prediction"]]
        label = record["run"]
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
    axes[0].set(
        xlabel="Phase locking to reviewed pulse peaks (expanded scale)",
        yticks=np.arange(len(valid)),
        yticklabels=[record["run"] for record in valid],
        xlim=(0.90, 1.00),
        title="Ten valid cardiac starts",
    )
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower left", fontsize=8)
    axes[0].grid(axis="x", alpha=0.2)
    axes[1].set(
        xticks=range(2),
        xticklabels=["Image prediction\n(excluded voxels)", "Pulse phase\nlocking"],
        ylim=(0, 0.5),
        title="Wrong start: s02_mvpsA_04\nBetter fit still follows the impostor",
    )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("TASKFORCE: phase fitted from images; pulse recordings used only for evaluation")
    fig.savefig(args.output / "comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
