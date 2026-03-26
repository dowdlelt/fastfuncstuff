"""Benchmark plots — multi-architecture timing and speedup charts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


# Ordered stage names for consistent plot layout
STAGE_ORDER = ["moco", "slicetime", "align", "warp", "glm", "ica"]

STAGE_LABELS = {
    "moco": "Motion\nCorrection",
    "slicetime": "Slice\nTiming",
    "align": "Alignment\n+ Qwarp",
    "warp": "Warp\nApply",
    "glm": "GLM\n(OLS+REML)",
    "ica": "ICA",
}

# Short arch labels for plot legends
def _short_arch(arch_id: str) -> str:
    """Convert arch_id to a short label for plots."""
    # linux-x86_64-cuda-NVIDIA_GeForce_RTX_5070_Ti -> RTX 5070 Ti
    # darwin-arm64-mps-Apple_M4_Pro -> M4 Pro
    parts = arch_id.split("-", 3)
    if len(parts) >= 4:
        accel = parts[3]
        # Strip common prefixes
        for prefix in ("NVIDIA_GeForce_", "NVIDIA_", "Apple_"):
            if accel.startswith(prefix):
                accel = accel[len(prefix):]
        return accel.replace("_", " ")
    return arch_id


def _extract_plot_data(cache: dict[str, Any]) -> list[dict]:
    """Extract per-architecture stage timing data from cache.

    Returns list of dicts with keys: arch_id, short_label, dataset_id, stages.
    Each stage entry has afni_seconds and ffs_seconds.
    """
    entries = []
    for run in cache.get("runs", []):
        arch_id = run.get("arch_id", "unknown")
        stages = run.get("stages", {})
        if not stages:
            continue
        dataset_id = run.get("dataset_id", "")
        short = _short_arch(arch_id)
        if dataset_id:
            short = f"{short} ({dataset_id})"
        entries.append({
            "arch_id": arch_id,
            "dataset_id": dataset_id,
            "short_label": short,
            "stages": stages,
        })
    return entries


def plot_timing_bars(
    cache: dict[str, Any],
    output_path: str | Path | None = None,
    title: str | None = None,
) -> None:
    """Grouped bar chart: AFNI vs FFS per stage, grouped by architecture.

    Each architecture gets a pair of bars (AFNI / FFS) for each stage.
    """
    import matplotlib.pyplot as plt

    entries = _extract_plot_data(cache)
    if not entries:
        print("No timing data to plot.")
        return

    # Collect all stages that have data
    all_stages = []
    for s in STAGE_ORDER:
        if any(s in e["stages"] for e in entries):
            all_stages.append(s)

    n_stages = len(all_stages)
    n_archs = len(entries)

    fig, ax = plt.subplots(figsize=(max(10, n_stages * 2.5), 6))

    # Bar layout: for each stage group, n_archs pairs of bars
    bar_width = 0.35 / max(n_archs, 1)
    group_width = (2 * bar_width + 0.05) * n_archs + 0.3

    colors_afni = plt.cm.Greys(np.linspace(0.4, 0.7, n_archs))
    colors_ffs = plt.cm.Blues(np.linspace(0.4, 0.8, n_archs))

    for arch_idx, entry in enumerate(entries):
        label_afni = f"AFNI ({entry['short_label']})"
        label_ffs = f"FFS ({entry['short_label']})"

        afni_times = []
        ffs_times = []
        for s in all_stages:
            stage_data = entry["stages"].get(s, {})
            afni_times.append(stage_data.get("afni_seconds", 0) or 0)
            ffs_times.append(stage_data.get("ffs_seconds", 0) or 0)

        x = np.arange(n_stages) * group_width
        offset = arch_idx * (2 * bar_width + 0.05)

        ax.bar(x + offset, afni_times, bar_width,
               label=label_afni, color=colors_afni[arch_idx],
               edgecolor="black", linewidth=0.5)
        ax.bar(x + offset + bar_width, ffs_times, bar_width,
               label=label_ffs, color=colors_ffs[arch_idx],
               edgecolor="black", linewidth=0.5)

    # Labels
    center_offset = (n_archs - 1) * (2 * bar_width + 0.05) / 2 + bar_width / 2
    ax.set_xticks(np.arange(n_stages) * group_width + center_offset)
    ax.set_xticklabels([STAGE_LABELS.get(s, s) for s in all_stages])
    ax.set_ylabel("Time (seconds)")
    ax.set_title(title or "AFNI vs FFS Benchmark Timing")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        print(f"Timing plot saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_speedup_bars(
    cache: dict[str, Any],
    output_path: str | Path | None = None,
    title: str | None = None,
) -> None:
    """Horizontal bar chart: FFS speedup over AFNI per stage, by architecture."""
    import matplotlib.pyplot as plt

    entries = _extract_plot_data(cache)
    if not entries:
        print("No timing data to plot.")
        return

    all_stages = []
    for s in STAGE_ORDER:
        if any(s in e["stages"] for e in entries):
            all_stages.append(s)

    n_stages = len(all_stages)
    n_archs = len(entries)

    fig, ax = plt.subplots(figsize=(10, max(4, n_stages * 0.8 * n_archs)))

    bar_height = 0.7 / max(n_archs, 1)
    colors = plt.cm.Set2(np.linspace(0, 0.8, n_archs))

    for arch_idx, entry in enumerate(entries):
        speedups = []
        for s in all_stages:
            stage_data = entry["stages"].get(s, {})
            afni = stage_data.get("afni_seconds", 0) or 0
            ffs = stage_data.get("ffs_seconds", 0) or 0
            if ffs > 0 and afni > 0:
                speedups.append(afni / ffs)
            else:
                speedups.append(0.0)

        y = np.arange(n_stages) + arch_idx * bar_height
        bars = ax.barh(y, speedups, bar_height,
                       label=entry["short_label"],
                       color=colors[arch_idx],
                       edgecolor="black", linewidth=0.5)

        # Value labels on bars
        for bar, sp in zip(bars, speedups):
            if sp > 0:
                ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                        f"{sp:.1f}x", va="center", fontsize=8)

    # Reference line at 1.0x
    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(1.02, -0.5, "parity", color="red", fontsize=8, alpha=0.7)

    center_offset = (n_archs - 1) * bar_height / 2
    ax.set_yticks(np.arange(n_stages) + center_offset)
    ax.set_yticklabels([STAGE_LABELS.get(s, s) for s in all_stages])
    ax.set_xlabel("Speedup (AFNI time / FFS time)")
    ax.set_title(title or "FFS Speedup over AFNI")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    fig.tight_layout()
    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        print(f"Speedup plot saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_all(
    cache: dict[str, Any],
    output_dir: str | Path,
    prefix: str = "benchmark",
) -> list[Path]:
    """Generate all benchmark plots, return list of saved paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    timing_path = output_dir / f"{prefix}_timing.png"
    plot_timing_bars(cache, output_path=timing_path)
    saved.append(timing_path)

    speedup_path = output_dir / f"{prefix}_speedup.png"
    plot_speedup_bars(cache, output_path=speedup_path)
    saved.append(speedup_path)

    return saved
