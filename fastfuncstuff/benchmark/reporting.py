"""Benchmark reporting — terminal tables and summary output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .arch import get_arch_id, get_arch_info
from .runner import StageResult


def print_validation_report(results: list[StageResult]) -> None:
    """Print a validation-only report table to terminal."""
    arch_id = get_arch_id()

    print(f"\nBENCHMARK VALIDATION: {arch_id}")
    print("=" * 64)
    print(f"{'Stage':<16} {'Status':<8} {'Summary'}")
    print("-" * 64)

    n_pass = 0
    n_fail = 0
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        if r.passed:
            n_pass += 1
        else:
            n_fail += 1
        print(f"{r.stage_name:<16} {status:<8} {r.summary}")
        for err in r.errors:
            print(f"  ERROR: {err}")

        # Detailed moco output: per-column motion correlations and MSD
        if r.stage_name == "moco" and r.validation:
            _print_moco_detail(r.validation)

    print("-" * 64)
    print(f"Total: {n_pass} passed, {n_fail} failed out of {len(results)} stages")


def _print_moco_detail(validation: dict) -> None:
    """Print per-column motion correlations and MSD for moco stage."""
    # Per-column motion parameter correlations
    motion = validation.get("motion_params", {})
    per_run = motion.get("per_run", [])
    if per_run:
        print("  Motion params (per-column r):")
        for entry in per_run:
            task, run = entry.get("task", "?"), entry.get("run", "?")
            per_col = entry.get("per_column", {})
            cols = " ".join(f"{name}={r:.3f}" for name, r in per_col.items())
            print(f"    {task} run-{run}: {cols}")

    # MSD of difference images
    msd_data = validation.get("timeseries_msd", {})
    msd_runs = msd_data.get("per_run", [])
    if msd_runs:
        print(f"  Timeseries MSD (nrmsd: mean={msd_data.get('mean_nrmsd', 0):.4f}, "
              f"max={msd_data.get('max_nrmsd', 0):.4f}):")
        for entry in msd_runs:
            if "error" in entry:
                print(f"    {entry.get('task', '?')} run-{entry.get('run', '?')}: ERROR {entry['error']}")
            else:
                task, run = entry.get("task", "?"), entry.get("run", "?")
                print(f"    {task} run-{run}: msd={entry['mean_msd']:.2f}, "
                      f"nrmsd={entry['nrmsd']:.4f}")


def print_timing_report(results: list[StageResult]) -> None:
    """Print a full timing + validation report table."""
    arch_id = get_arch_id()

    print(f"\nBENCHMARK: {arch_id}")
    print("=" * 72)
    print(f"{'Stage':<16} {'AFNI (s)':>10} {'FFS (s)':>10} {'Speedup':>10} {'Status'}")
    print("-" * 72)

    total_afni = 0.0
    total_ffs = 0.0

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        afni_str = f"{r.afni_time:.1f}" if r.afni_time is not None else "-"
        ffs_str = f"{r.ffs_time:.1f}" if r.ffs_time is not None else "-"

        if r.afni_time is not None and r.ffs_time is not None and r.ffs_time > 0:
            speedup = r.afni_time / r.ffs_time
            speedup_str = f"{speedup:.1f}x"
            total_afni += r.afni_time
            total_ffs += r.ffs_time
        else:
            speedup_str = "-"

        summary_short = r.summary[:30] if r.summary else ""
        print(f"{r.stage_name:<16} {afni_str:>10} {ffs_str:>10} {speedup_str:>10} {status}  {summary_short}")

    print("-" * 72)
    if total_ffs > 0:
        total_speedup = f"{total_afni / total_ffs:.1f}x"
        print(f"{'Total':<16} {total_afni:>10.1f} {total_ffs:>10.1f} {total_speedup:>10}")


def results_to_json(results: list[StageResult]) -> dict[str, Any]:
    """Convert results to a JSON-serializable dict."""
    arch_info = get_arch_info()
    stages = {}
    for r in results:
        stages[r.stage_name] = {
            "passed": r.passed,
            "summary": r.summary,
            "afni_seconds": r.afni_time,
            "ffs_seconds": r.ffs_time,
            "validation": _sanitize_for_json(r.validation),
            "errors": r.errors,
        }

    return {
        "architecture": arch_info,
        "stages": stages,
    }


def save_json_report(results: list[StageResult], path: str | Path) -> None:
    """Save results as JSON."""
    data = results_to_json(results)
    with open(str(path), "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nJSON report saved: {path}")


def _sanitize_for_json(obj: Any) -> Any:
    """Make an object JSON-serializable."""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
