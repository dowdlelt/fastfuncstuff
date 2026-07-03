"""Benchmark plots — multi-architecture timing and speedup charts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Ordered stage names for consistent plot layout
STAGE_ORDER = [
    "moco",
    "slicetime",
    "crossalign",
    "align",
    "warp",
    "glm",
    "ica",
    "ica_single",
    "glmsingle_prep",
    "glmsingle_hrf",
    "glmsingle_denoise",
    "glmsingle_ridge",
]

STAGE_LABELS = {
    "moco": "Motion\nCorrection",
    "slicetime": "Slice\nTiming",
    "crossalign": "Cross-run\nAlignment",
    "align": "Alignment\n+ Qwarp",
    "warp": "Warp\nApply",
    "glm": "GLM\n(OLS+REML)",
    "ica": "ICA\n(concat)",
    "ica_single": "ICA\n(single-run)",
    "glmsingle_prep": "GLMsingle\nPrep",
    "glmsingle_hrf": "GLMsingle\nHRF (B)",
    "glmsingle_denoise": "GLMsingle\nDenoise (C)",
    "glmsingle_ridge": "GLMsingle\nRidge (D)",
}


# Short arch labels for plot legends
def _short_arch(arch_id: str) -> str:
    """Convert an arch_id to a short label for plots.

    Handles both v2 IDs (e.g. "cuda-NVIDIA_GeForce_RTX_5070_Ti", "linux-x86_64")
    and legacy v1 combined IDs (e.g. "linux-x86_64-cuda-NVIDIA_GeForce_RTX_5070_Ti").
    """
    for prefix in ("cuda-NVIDIA_GeForce_", "cuda-NVIDIA_", "cuda-", "mps-Apple_", "mps-"):
        if arch_id.startswith(prefix):
            return arch_id[len(prefix) :].replace("_", " ")
    # Legacy: linux-x86_64-cuda-NVIDIA_GeForce_RTX_5070_Ti
    parts = arch_id.split("-", 3)
    if len(parts) >= 4:
        accel = parts[3]
        for prefix in ("NVIDIA_GeForce_", "NVIDIA_", "Apple_"):
            if accel.startswith(prefix):
                accel = accel[len(prefix) :]
        return accel.replace("_", " ")
    # ref_arch_id style: linux-x86_64
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}"
    return arch_id


def _extract_plot_data(cache: dict[str, Any]) -> list[dict]:
    """Extract per-machine stage timing data from cache.

    Returns one entry per unique (ref_arch_id, ffs_arch_id) pair, using the
    latest timing for each stage. Compatible with both v1 and v2 schemas.
    """
    from .timing_cache import get_latest_per_arch

    collapsed = get_latest_per_arch(cache)
    entries = []
    for run in collapsed:
        ffs_id = run.get("ffs_arch_id", run.get("arch_id", "unknown"))
        stages = run.get("stages", {})
        if not stages:
            continue
        dataset_id = run.get("dataset_id", "")
        short = _short_arch(ffs_id)
        if dataset_id:
            short = f"{short} ({dataset_id})"
        entries.append(
            {
                "arch_id": run.get("arch_id", ffs_id),
                "ref_arch_id": run.get("ref_arch_id", ""),
                "ffs_arch_id": ffs_id,
                "dataset_id": dataset_id,
                "short_label": short,
                "stages": stages,
            }
        )
    return entries


def _get_ref_seconds(stage_data: dict) -> float:
    """Get reference tool timing, supporting both old and new key names."""
    return stage_data.get("ref_seconds") or stage_data.get("afni_seconds", 0) or 0


def plot_timing_bars(
    cache: dict[str, Any],
    output_path: str | Path | None = None,
    title: str | None = None,
) -> None:
    """Grouped bar chart: Reference vs FFS per stage, grouped by architecture."""
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

    colors_ref = plt.cm.Greys(np.linspace(0.4, 0.7, n_archs))
    colors_ffs = plt.cm.Blues(np.linspace(0.4, 0.8, n_archs))

    for arch_idx, entry in enumerate(entries):
        label_ref = f"Ref ({entry['short_label']})"
        label_ffs = f"FFS ({entry['short_label']})"

        ref_times = []
        ffs_times = []
        for s in all_stages:
            stage_data = entry["stages"].get(s, {})
            ref_times.append(_get_ref_seconds(stage_data))
            ffs_times.append(stage_data.get("ffs_seconds", 0) or 0)

        x = np.arange(n_stages) * group_width
        offset = arch_idx * (2 * bar_width + 0.05)

        ax.bar(
            x + offset,
            ref_times,
            bar_width,
            label=label_ref,
            color=colors_ref[arch_idx],
            edgecolor="black",
            linewidth=0.5,
        )
        ax.bar(
            x + offset + bar_width,
            ffs_times,
            bar_width,
            label=label_ffs,
            color=colors_ffs[arch_idx],
            edgecolor="black",
            linewidth=0.5,
        )

    # Labels
    center_offset = (n_archs - 1) * (2 * bar_width + 0.05) / 2 + bar_width / 2
    ax.set_xticks(np.arange(n_stages) * group_width + center_offset)
    ax.set_xticklabels([STAGE_LABELS.get(s, s) for s in all_stages])
    ax.set_ylabel("Time (seconds)")
    ax.set_title(title or "Reference vs FFS Benchmark Timing")
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
    """Horizontal bar chart: FFS speedup over reference per stage, by architecture."""
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
            ref = _get_ref_seconds(stage_data)
            ffs = stage_data.get("ffs_seconds", 0) or 0
            if ffs > 0 and ref > 0:
                speedups.append(ref / ffs)
            else:
                speedups.append(0.0)

        y = np.arange(n_stages) + arch_idx * bar_height
        bars = ax.barh(
            y,
            speedups,
            bar_height,
            label=entry["short_label"],
            color=colors[arch_idx],
            edgecolor="black",
            linewidth=0.5,
        )

        # Value labels on bars
        for bar, sp in zip(bars, speedups, strict=False):
            if sp > 0:
                ax.text(
                    bar.get_width() + 0.05,
                    bar.get_y() + bar.get_height() / 2,
                    f"{sp:.1f}x",
                    va="center",
                    fontsize=8,
                )

    # Reference line at 1.0x
    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(1.02, -0.5, "parity", color="red", fontsize=8, alpha=0.7)

    center_offset = (n_archs - 1) * bar_height / 2
    ax.set_yticks(np.arange(n_stages) + center_offset)
    ax.set_yticklabels([STAGE_LABELS.get(s, s) for s in all_stages])
    ax.set_xlabel("Speedup (Ref time / FFS time)")
    ax.set_title(title or "FFS Speedup over Reference Tools")
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
