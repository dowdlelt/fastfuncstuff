"""Timing cache — persist benchmark timing results per architecture."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .arch import get_arch_id, get_arch_info


SCHEMA_VERSION = 1


def _cache_path(data_dir: Path) -> Path:
    return data_dir / "benchmark_cache.json"


def load_cache(data_dir: Path) -> dict[str, Any]:
    """Load the timing cache, or return an empty cache."""
    path = _cache_path(data_dir)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        if data.get("schema_version") == SCHEMA_VERSION:
            return data
    return {"schema_version": SCHEMA_VERSION, "runs": []}


def save_cache(data_dir: Path, cache: dict[str, Any]) -> None:
    """Write the timing cache to disk."""
    path = _cache_path(data_dir)
    with open(str(path), "w") as f:
        json.dump(cache, f, indent=2, default=str)


def get_cached_timing(
    data_dir: Path, stage_name: str, dataset_id: str = "",
) -> tuple[float | None, float | None]:
    """Get cached (afni_seconds, ffs_seconds) for a stage on current arch.

    Returns (None, None) if no cached data found.
    """
    cache = load_cache(data_dir)
    arch_id = get_arch_id()

    # Find most recent run for this architecture + dataset
    for run in reversed(cache.get("runs", [])):
        if run.get("arch_id") == arch_id:
            if dataset_id and run.get("dataset_id", "") != dataset_id:
                continue
            stage = run.get("stages", {}).get(stage_name, {})
            return stage.get("afni_seconds"), stage.get("ffs_seconds")
    return None, None


def update_cache(
    data_dir: Path,
    stage_timings: dict[str, dict[str, float | None]],
    dataset_id: str = "",
) -> None:
    """Update or append timing data for current architecture.

    stage_timings: {stage_name: {"afni_seconds": float, "ffs_seconds": float}}
    """
    cache = load_cache(data_dir)
    arch_id = get_arch_id()
    arch_info = get_arch_info()

    # Find existing run for this arch + dataset, or create new
    existing = None
    for run in cache["runs"]:
        if run.get("arch_id") == arch_id:
            if dataset_id and run.get("dataset_id", "") != dataset_id:
                continue
            existing = run
            break

    if existing is None:
        existing = {
            "arch_id": arch_id,
            "dataset_id": dataset_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "architecture": arch_info,
            "stages": {},
        }
        cache["runs"].append(existing)

    # Update stages
    existing["timestamp"] = datetime.now(timezone.utc).isoformat()
    for stage_name, timings in stage_timings.items():
        if stage_name not in existing["stages"]:
            existing["stages"][stage_name] = {}
        for key, val in timings.items():
            if val is not None:
                existing["stages"][stage_name][key] = val

    save_cache(data_dir, cache)
