"""Benchmark reporting — terminal tables and summary output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .arch import get_arch_info, get_ffs_arch_id, get_ref_arch_id
from .runner import StageResult
from .timing_cache import get_ref_timings_all_archs


def print_validation_report(results: list[StageResult]) -> None:
    """Print a validation-only report table to terminal."""
    ref_id = get_ref_arch_id()
    ffs_id = get_ffs_arch_id()

    print(f"\nBENCHMARK VALIDATION  ref={ref_id}  ffs={ffs_id}")
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

        if r.stage_name == "moco" and r.validation:
            _print_moco_detail(r.validation)

    print("-" * 64)
    print(f"Total: {n_pass} passed, {n_fail} failed out of {len(results)} stages")


def _print_moco_detail(validation: dict) -> None:
    """Print per-column motion correlations and MSD for moco stage."""
    motion = validation.get("motion_params", {})
    per_run = motion.get("per_run", [])
    if per_run:
        print("  Motion params (per-column r):")
        for entry in per_run:
            task, run = entry.get("task", "?"), entry.get("run", "?")
            per_col = entry.get("per_column", {})
            cols = " ".join(f"{name}={r:.3f}" for name, r in per_col.items())
            print(f"    {task} run-{run}: {cols}")

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


def print_timing_report(
    results: list[StageResult], data_dir: Path | None = None,
) -> None:
    """Print a full timing + validation report table.

    When data_dir is provided, missing ref timings are pulled from the cache
    (queried by CPU arch for ref, GPU for FFS) and annotated:
      †  = cached from this CPU architecture
      ‡  = cached from a different CPU architecture
    """
    my_ref_id = get_ref_arch_id()
    my_ffs_id = get_ffs_arch_id()

    # Pre-fetch cached ref timings for stages where ref didn't run this time
    cached_refs: dict[str, list[tuple[str, float]]] = {}
    if data_dir is not None:
        for r in results:
            if not (r.ref_time and r.ref_time > 0) and r.ffs_time:
                cached_refs[r.stage_name] = get_ref_timings_all_archs(data_dir, r.stage_name)

    print(f"\nBENCHMARK  ref={my_ref_id}  ffs={my_ffs_id}")
    print("=" * 80)
    print(f"{'Stage':<16} {'Ref (s)':>12} {'FFS (s)':>10} {'Speedup':>10} {'Status'}")
    print("-" * 80)

    total_ref = 0.0
    total_ffs = 0.0
    any_cached_ref = False
    any_other_arch_ref = False

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        ffs_t = r.ffs_time or 0.0
        ref_t = r.ref_time or 0.0
        ffs_str = f"{ffs_t:.1f}" if ffs_t > 0 else "-"

        # Determine best ref timing for display
        if ref_t > 0:
            ref_str = f"{ref_t:.1f}"
        elif r.stage_name in cached_refs:
            all_refs = cached_refs[r.stage_name]
            local = next((rv for av, rv in all_refs if av == my_ref_id), None)
            if local is not None:
                ref_t = local
                ref_str = f"{ref_t:.1f}†"
                any_cached_ref = True
            elif all_refs:
                _, best_ref = all_refs[0]  # sorted by ref_seconds asc
                ref_t = best_ref
                ref_str = f"{ref_t:.1f}‡"
                any_other_arch_ref = True
            else:
                ref_str = "-"
        else:
            ref_str = "-"

        if ref_t > 0 and ffs_t > 0:
            speedup_str = f"{ref_t / ffs_t:.1f}x"
            total_ref += ref_t
            total_ffs += ffs_t
        else:
            speedup_str = "-"

        summary_short = r.summary[:28] if r.summary else ""
        print(f"{r.stage_name:<16} {ref_str:>12} {ffs_str:>10} {speedup_str:>10} {status}  {summary_short}")

    print("-" * 80)
    if total_ffs > 0:
        print(f"{'Total':<16} {total_ref:>12.1f} {total_ffs:>10.1f} {total_ref / total_ffs:>9.1f}x")

    if any_cached_ref:
        print(f"  † cached ref timing (CPU arch: {my_ref_id})")
    if any_other_arch_ref:
        print(f"  ‡ cached ref timing from a different CPU architecture")


def _short_arch(arch_id: str) -> str:
    """Shorten an arch_id for compact display.

    Handles both v2 IDs (ref_arch_id like "linux-x86_64", ffs_arch_id like
    "cuda-NVIDIA_GeForce_RTX_5070_Ti") and legacy combined IDs.
    """
    for prefix in ("cuda-NVIDIA_GeForce_", "cuda-NVIDIA_", "cuda-", "mps-Apple_", "mps-"):
        if arch_id.startswith(prefix):
            return arch_id[len(prefix):].replace("_", " ")
    # linux-x86_64 or darwin-arm64 style
    parts = arch_id.split("-", 1)
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}"
    return arch_id


def results_to_json(results: list[StageResult]) -> dict[str, Any]:
    """Convert results to a JSON-serializable dict."""
    arch_info = get_arch_info()
    stages = {}
    for r in results:
        stages[r.stage_name] = {
            "passed": r.passed,
            "summary": r.summary,
            "ref_seconds": r.ref_time,
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
