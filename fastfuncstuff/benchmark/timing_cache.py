"""Timing cache — append-only run log, queryable by CPU arch (ref) or GPU (FFS).

Schema v2
---------
Every invocation of ffs_benchmark appends one entry to ``runs``. Nothing is
ever overwritten, so the file is a full history that can be committed to git
and shared across machines.

Each entry stores:
- ``ref_arch_id``: CPU-based ID used to group *reference* tool timings
  (format: "{os}-{cpu_arch}", e.g. "linux-x86_64")
- ``ffs_arch_id``: GPU-based ID used to group *FFS* tool timings
  (format: "cuda-{gpu}" | "mps-{proc}" | "cpu")
- Full ``hardware`` dict for rich querying / plotting
- Per-stage timings for whatever was measured this run (None if skipped)

Query helpers
-------------
- ``get_ref_timings_all_archs(data_dir, stage)``  → latest ref time per CPU arch
- ``get_ffs_timings_all_gpus(data_dir, stage)``   → latest FFS time per GPU
- ``query_runs(cache, stage, role, arch_id)``      → all matching entries (distribution)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .arch import get_ffs_arch_id, get_hardware_info, get_ref_arch_id


SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _cache_path(data_dir: Path) -> Path:
    return data_dir / "benchmark_cache.json"


def load_cache(data_dir: Path) -> dict[str, Any]:
    """Load the timing cache (migrating v1 → v2 if needed)."""
    path = _cache_path(data_dir)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        if data.get("schema_version") == SCHEMA_VERSION:
            return data
        if data.get("schema_version") == 1:
            return _migrate_v1(data)
    return {"schema_version": SCHEMA_VERSION, "runs": []}


def save_cache(data_dir: Path, cache: dict[str, Any]) -> None:
    """Write the timing cache to disk."""
    path = _cache_path(data_dir)
    with open(str(path), "w") as f:
        json.dump(cache, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------

def _migrate_v1(v1: dict[str, Any]) -> dict[str, Any]:
    """Convert a v1 cache (one-entry-per-arch, mutable) to v2 (append-only)."""
    runs = []
    for old in v1.get("runs", []):
        arch_id = old.get("arch_id", "unknown")
        # v1 arch_id: {os}-{machine}-{accel...}  →  split off first two parts
        parts = arch_id.split("-", 2)
        os_part = parts[0] if len(parts) > 0 else "unknown"
        machine = parts[1] if len(parts) > 1 else "unknown"
        accel = parts[2] if len(parts) > 2 else "cpu"

        old_hw = old.get("architecture", {})
        hardware = {
            "ref_arch_id": f"{os_part}-{machine}",
            "ffs_arch_id": accel,
            "os": os_part,
            "cpu_arch": machine,
            "cpu_model": "unknown",
            "n_logical_cores": None,
            "n_physical_cores": None,
            "gpu": old_hw.get("gpu"),
            "gpu_memory_gb": old_hw.get("gpu_memory_gb"),
            "python": old_hw.get("python"),
            "torch": old_hw.get("torch"),
            "omp_num_threads": old_hw.get("omp_num_threads"),
        }
        # Migrate stage timing keys
        stages: dict[str, Any] = {}
        for sname, sdata in old.get("stages", {}).items():
            ref = sdata.get("ref_seconds") or sdata.get("afni_seconds")
            ffs = sdata.get("ffs_seconds")
            stages[sname] = {}
            if ref is not None:
                stages[sname]["ref_seconds"] = ref
            if ffs is not None:
                stages[sname]["ffs_seconds"] = ffs

        runs.append({
            "id": str(uuid.uuid4()),
            "timestamp": old.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "ref_arch_id": f"{os_part}-{machine}",
            "ffs_arch_id": accel,
            "hardware": hardware,
            "dataset_id": old.get("dataset_id", ""),
            "stages": stages,
        })

    return {"schema_version": SCHEMA_VERSION, "runs": runs}


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def append_run(
    data_dir: Path,
    stage_timings: dict[str, dict[str, float | None]],
    dataset_id: str = "",
) -> None:
    """Append one benchmark run to the cache.

    stage_timings: {stage_name: {"ref_seconds": float|None, "ffs_seconds": float|None}}
    Only non-None values are stored.

    Every call creates a new entry — nothing is overwritten.
    """
    cache = load_cache(data_dir)
    hw = get_hardware_info()

    stages: dict[str, Any] = {}
    for sname, timings in stage_timings.items():
        entry: dict[str, float] = {}
        for key, val in timings.items():
            if val is not None:
                entry[key] = val
        if entry:
            stages[sname] = entry

    if not stages:
        return  # nothing to store

    run = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ref_arch_id": get_ref_arch_id(),
        "ffs_arch_id": get_ffs_arch_id(),
        "hardware": hw,
        "dataset_id": dataset_id,
        "stages": stages,
    }
    cache["runs"].append(run)
    save_cache(data_dir, cache)


# Backward-compatible alias
def update_cache(
    data_dir: Path,
    stage_timings: dict[str, dict[str, float | None]],
    dataset_id: str = "",
) -> None:
    """Backward-compatible alias for append_run()."""
    append_run(data_dir, stage_timings, dataset_id)


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------

def query_runs(
    cache: dict[str, Any],
    stage_name: str,
    role: str,  # "ref" or "ffs"
    arch_id: str | None = None,
    dataset_id: str = "",
) -> list[dict[str, Any]]:
    """Return all run entries that have timing data for stage_name/role.

    Args:
        cache: Loaded cache dict.
        stage_name: e.g. "glm"
        role: "ref" or "ffs"  →  keys "ref_seconds" / "ffs_seconds"
        arch_id: If given, filter to this ref_arch_id (for role="ref") or
                 ffs_arch_id (for role="ffs"). None = all archs.
        dataset_id: If non-empty, filter to this dataset.

    Returns:
        List of matching run dicts (each has "timestamp", "hardware",
        "ref_arch_id", "ffs_arch_id", and the stage timing), sorted
        oldest-first.
    """
    key = f"{role}_seconds"
    arch_field = "ref_arch_id" if role == "ref" else "ffs_arch_id"
    results = []
    for run in cache.get("runs", []):
        if dataset_id and run.get("dataset_id", "") != dataset_id:
            continue
        if arch_id and run.get(arch_field) != arch_id:
            continue
        seconds = run.get("stages", {}).get(stage_name, {}).get(key)
        if seconds is None:
            continue
        results.append({**run, "_seconds": seconds})
    results.sort(key=lambda r: r["timestamp"])
    return results


def get_ref_timings_all_archs(
    data_dir: Path,
    stage_name: str,
    dataset_id: str = "",
) -> list[tuple[str, float]]:
    """Latest reference timing per CPU architecture (ref_arch_id).

    Returns list of (ref_arch_id, ref_seconds) sorted by ref_seconds
    ascending (fastest reference first).
    """
    cache = load_cache(data_dir)
    # Collect most recent timing per ref_arch_id
    latest: dict[str, float] = {}
    for run in cache.get("runs", []):
        if dataset_id and run.get("dataset_id", "") != dataset_id:
            continue
        ref_id = run.get("ref_arch_id", "unknown")
        seconds = run.get("stages", {}).get(stage_name, {}).get("ref_seconds")
        if seconds is None:
            # Support legacy key
            seconds = run.get("stages", {}).get(stage_name, {}).get("afni_seconds")
        if seconds is not None:
            # Later entries overwrite earlier — "latest" wins
            latest[ref_id] = float(seconds)
    return sorted(latest.items(), key=lambda x: x[1])


def get_ffs_timings_all_gpus(
    data_dir: Path,
    stage_name: str,
    dataset_id: str = "",
) -> list[tuple[str, float]]:
    """Latest FFS timing per GPU/accelerator (ffs_arch_id).

    Returns list of (ffs_arch_id, ffs_seconds) sorted by ffs_seconds
    ascending (fastest FFS first).
    """
    cache = load_cache(data_dir)
    latest: dict[str, float] = {}
    for run in cache.get("runs", []):
        if dataset_id and run.get("dataset_id", "") != dataset_id:
            continue
        ffs_id = run.get("ffs_arch_id", "unknown")
        seconds = run.get("stages", {}).get(stage_name, {}).get("ffs_seconds")
        if seconds is not None:
            latest[ffs_id] = float(seconds)
    return sorted(latest.items(), key=lambda x: x[1])


def get_cached_timing(
    data_dir: Path, stage_name: str, dataset_id: str = "",
) -> tuple[float | None, float | None]:
    """Get latest (ref_seconds, ffs_seconds) for current machine's archs.

    ref_seconds: latest for current ref_arch_id (CPU arch)
    ffs_seconds: latest for current ffs_arch_id (GPU)

    Returns (None, None) if no cached data found.
    """
    cache = load_cache(data_dir)
    my_ref_id = get_ref_arch_id()
    my_ffs_id = get_ffs_arch_id()

    ref_val: float | None = None
    ffs_val: float | None = None

    for run in cache.get("runs", []):
        if dataset_id and run.get("dataset_id", "") != dataset_id:
            continue
        stage = run.get("stages", {}).get(stage_name, {})
        if run.get("ref_arch_id") == my_ref_id:
            v = stage.get("ref_seconds") or stage.get("afni_seconds")
            if v is not None:
                ref_val = float(v)  # later wins → latest
        if run.get("ffs_arch_id") == my_ffs_id:
            v = stage.get("ffs_seconds")
            if v is not None:
                ffs_val = float(v)

    return ref_val, ffs_val


def list_cache_entries(data_dir: Path) -> list[dict[str, Any]]:
    """Return all cache entries as a list, oldest first, with a 1-based index.

    Each dict has the run fields plus ``"index"`` (1-based display index).
    """
    cache = load_cache(data_dir)
    runs = cache.get("runs", [])
    return [{**r, "index": i + 1} for i, r in enumerate(runs)]


def print_cache(data_dir: Path, stage_filter: str | None = None) -> None:
    """Pretty-print all cache entries to stdout.

    Args:
        data_dir: Directory containing benchmark_cache.json.
        stage_filter: If given, only show timings for this stage name.
    """
    entries = list_cache_entries(data_dir)
    n = len(entries)
    print(f"Cache: {_cache_path(data_dir)}  ({n} {'entry' if n == 1 else 'entries'})")
    if not entries:
        print("  (empty)")
        return

    print()
    print(f"  {'#':>3}  {'ID':8}  {'Timestamp':19}  {'ref_arch':14}  {'ffs_arch':36}  Stages")
    print(f"  {'-'*3}  {'-'*8}  {'-'*19}  {'-'*14}  {'-'*36}  {'-'*32}")

    for e in entries:
        idx = e["index"]
        eid = e.get("id", "?")[:8]
        ts = e.get("timestamp", "")[:19].replace("T", " ")
        ref_id = e.get("ref_arch_id", "?")[:14]
        ffs_id = e.get("ffs_arch_id", "?")[:36]
        stages = e.get("stages", {})

        stage_parts = []
        for sname, sdata in sorted(stages.items()):
            if stage_filter and sname != stage_filter:
                continue
            ref = sdata.get("ref_seconds")
            ffs = sdata.get("ffs_seconds")
            ref_str = f"{ref:.0f}" if ref is not None else "-"
            ffs_str = f"{ffs:.0f}" if ffs is not None else "-"
            stage_parts.append(f"{sname}:{ref_str}/{ffs_str}")

        stages_str = "  ".join(stage_parts) if stage_parts else "(no matching stages)"
        print(f"  {idx:>3}  {eid:8}  {ts:19}  {ref_id:14}  {ffs_id:36}  {stages_str}")


def remove_cache_entries(
    data_dir: Path,
    indices: list[int] | None = None,
    ids: list[str] | None = None,
    dry_run: bool = False,
) -> int:
    """Remove cache entries by 1-based index or UUID prefix.

    Args:
        data_dir: Directory containing benchmark_cache.json.
        indices: 1-based entry indices to remove (as shown by print_cache).
        ids: UUID strings or prefixes to remove.
        dry_run: If True, print what would be removed but don't write.

    Returns:
        Number of entries removed.
    """
    cache = load_cache(data_dir)
    runs = cache.get("runs", [])

    to_remove: set[int] = set()  # 0-based positions in runs list

    if indices:
        for idx in indices:
            pos = idx - 1  # convert 1-based → 0-based
            if 0 <= pos < len(runs):
                to_remove.add(pos)
            else:
                print(f"  WARNING: index {idx} out of range (1–{len(runs)}), skipped")

    if ids:
        for uid in ids:
            matched = False
            for pos, run in enumerate(runs):
                if run.get("id", "").startswith(uid):
                    to_remove.add(pos)
                    matched = True
            if not matched:
                print(f"  WARNING: ID prefix '{uid}' not found, skipped")

    if not to_remove:
        print("  Nothing to remove.")
        return 0

    print(f"  {'Would remove' if dry_run else 'Removing'} {len(to_remove)} entr{'y' if len(to_remove) == 1 else 'ies'}:")
    for pos in sorted(to_remove):
        run = runs[pos]
        ts = run.get("timestamp", "")[:19].replace("T", " ")
        ref_id = run.get("ref_arch_id", "?")
        ffs_id = run.get("ffs_arch_id", "?")
        stages = ", ".join(
            f"{s}:{d.get('ref_seconds', '-')}/{d.get('ffs_seconds', '-')}"
            for s, d in run.get("stages", {}).items()
        )
        print(f"    [{pos + 1}] {run.get('id', '?')[:8]}  {ts}  {ref_id} / {ffs_id}  {stages}")

    if dry_run:
        print("  (dry run — nothing written)")
        return len(to_remove)

    cache["runs"] = [r for i, r in enumerate(runs) if i not in to_remove]
    save_cache(data_dir, cache)
    print(f"  Saved. Cache now has {len(cache['runs'])} entr{'y' if len(cache['runs']) == 1 else 'ies'}.")
    return len(to_remove)


def merge_cache_from_file(
    data_dir: Path,
    source_path: Path,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Merge runs from another benchmark_cache.json into the local cache.

    Deduplicates by UUID — runs already present (by id) are skipped.
    Migrates v1 source files automatically.

    Args:
        data_dir: Directory containing the local benchmark_cache.json.
        source_path: Path to the external benchmark_cache.json to import from.
        dry_run: If True, report what would be added without writing.

    Returns:
        (added, skipped) counts.
    """
    import json

    if not source_path.exists():
        raise FileNotFoundError(f"Source cache not found: {source_path}")

    with open(source_path) as f:
        source = json.load(f)

    if source.get("schema_version") != SCHEMA_VERSION:
        source = _migrate_v1(source)

    local = load_cache(data_dir)
    existing_ids: set[str] = {r.get("id", "") for r in local.get("runs", [])}

    to_add = []
    skipped = 0
    for run in source.get("runs", []):
        uid = run.get("id", "")
        if uid and uid in existing_ids:
            skipped += 1
        else:
            to_add.append(run)

    print(f"  Source: {source_path}  ({len(source.get('runs', []))} entries)")
    print(f"  Local:  {_cache_path(data_dir)}  ({len(local.get('runs', []))} entries)")
    print(f"  New: {len(to_add)}  |  Already present: {skipped}")

    if to_add:
        for run in to_add:
            ref_id = run.get("ref_arch_id", "?")
            ffs_id = run.get("ffs_arch_id", "?")
            ts = run.get("timestamp", "")[:19].replace("T", " ")
            stages = ", ".join(
                f"{s}:{d.get('ref_seconds', '-')}/{d.get('ffs_seconds', '-')}"
                for s, d in run.get("stages", {}).items()
            )
            verb = "Would add" if dry_run else "Adding"
            print(f"    {verb}: {run.get('id', '?')[:8]}  {ts}  {ref_id} / {ffs_id}  {stages}")

    if dry_run:
        print("  (dry run — nothing written)")
        return len(to_add), skipped

    if to_add:
        local["runs"].extend(to_add)
        save_cache(data_dir, local)
        total = len(local["runs"])
        print(f"  Saved. Cache now has {total} entr{'y' if total == 1 else 'ies'}.")

    return len(to_add), skipped


def get_latest_per_arch(cache: dict[str, Any]) -> list[dict[str, Any]]:
    """Collapse runs to one entry per (ref_arch_id, ffs_arch_id) pair.

    Takes the most recent run for each unique hardware combination.
    Used by plots to show one bar per machine (like the old v1 schema).

    Each returned dict mirrors the v1 run structure:
      {"arch_id", "ref_arch_id", "ffs_arch_id", "hardware", "dataset_id", "stages"}
    where stages contains the *latest* timing for each stage on that machine.
    """
    # Key: (ref_arch_id, ffs_arch_id, dataset_id)
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}

    for run in cache.get("runs", []):
        ref_id = run.get("ref_arch_id", "unknown")
        ffs_id = run.get("ffs_arch_id", "unknown")
        ds = run.get("dataset_id", "")
        key = (ref_id, ffs_id, ds)

        if key not in buckets:
            buckets[key] = {
                "arch_id": f"{ref_id}-{ffs_id}",
                "ref_arch_id": ref_id,
                "ffs_arch_id": ffs_id,
                "dataset_id": ds,
                "hardware": run.get("hardware", {}),
                "timestamp": run.get("timestamp", ""),
                "stages": {},
            }

        existing = buckets[key]
        # Later entries win (most recent timing per stage)
        if run.get("timestamp", "") >= existing["timestamp"]:
            existing["timestamp"] = run.get("timestamp", "")
            existing["hardware"] = run.get("hardware", existing["hardware"])
        # Merge stage timings — later entry overwrites per key
        for sname, sdata in run.get("stages", {}).items():
            if sname not in existing["stages"]:
                existing["stages"][sname] = {}
            existing["stages"][sname].update(sdata)

    return list(buckets.values())
