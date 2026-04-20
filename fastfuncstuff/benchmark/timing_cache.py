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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .arch import get_ffs_arch_id, get_hardware_info, get_ref_arch_id


SCHEMA_VERSION = 3


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _cache_path(data_dir: Path) -> Path:
    return data_dir / "benchmark_cache.json"


def load_cache(data_dir: Path) -> dict[str, Any]:
    """Load the timing cache (migrating v1/v2 → v3 if needed)."""
    path = _cache_path(data_dir)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        v = data.get("schema_version", 1)
        if v == SCHEMA_VERSION:
            return data
        if v == 1:
            data = _migrate_v1(data)
        # v2 → v3: just bump version, new fields are additive
        data["schema_version"] = SCHEMA_VERSION
        return data
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
# Git info
# ---------------------------------------------------------------------------

def _get_git_info() -> dict[str, Any]:
    """Capture current git state: commit, branch, message, dirty flag."""
    import subprocess

    def _run(args: list[str]) -> str:
        try:
            return subprocess.run(
                args, capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        except Exception:
            return ""

    commit = _run(["git", "rev-parse", "HEAD"])
    short = _run(["git", "rev-parse", "--short", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    message = _run(["git", "log", "-1", "--format=%s"])
    dirty = _run(["git", "status", "--porcelain"]) != ""

    return {
        "commit": commit,
        "commit_short": short,
        "branch": branch,
        "message": message,
        "dirty": dirty,
    }


# ---------------------------------------------------------------------------
# Metric sanitization
# ---------------------------------------------------------------------------

def _sanitize_for_cache(obj: Any) -> Any:
    """Make a validation dict JSON-serializable and compact.

    Converts numpy types to native Python. Strips None values.
    Keeps all scalar metrics and small lists/dicts (per-run, per-pair results).
    """
    try:
        import numpy as np
        _has_numpy = True
    except ImportError:
        _has_numpy = False

    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_cache(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_cache(v) for v in obj]
    if _has_numpy:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            # Only keep small arrays (e.g. per-run lists)
            if obj.size <= 100:
                return obj.tolist()
            # Large arrays → summary stats only
            return {
                "_type": "array_summary",
                "shape": list(obj.shape),
                "mean": float(np.nanmean(obj)),
                "std": float(np.nanstd(obj)),
                "min": float(np.nanmin(obj)),
                "max": float(np.nanmax(obj)),
            }
        if isinstance(obj, np.bool_):
            return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def extract_scalar_metrics(
    validation: dict[str, Any], prefix: str = "",
) -> dict[str, float]:
    """Recursively extract all scalar numeric values from a validation dict.

    Returns a flat dict with dotted keys, e.g.:
        {"ols.min_r": 0.99, "reml.min_r": 0.95, "anat.r": 0.92}

    Skips non-numeric values, large arrays, and metadata keys.
    """
    _SKIP_KEYS = {"passed", "summary", "error", "errors", "label", "dataset",
                  "task", "run", "shape", "_type", "n_components_a", "n_components_b"}
    metrics: dict[str, float] = {}

    for key, val in validation.items():
        if key in _SKIP_KEYS:
            continue
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(val, (int, float)) and not isinstance(val, bool):
            metrics[full_key] = float(val)
        elif isinstance(val, dict):
            # Recurse into sub-dicts (e.g. "ols": {"min_r": 0.99})
            metrics.update(extract_scalar_metrics(val, full_key))
        elif isinstance(val, list) and val and isinstance(val[0], (int, float)):
            # Numeric list → store summary
            if len(val) <= 20:
                for i, v in enumerate(val):
                    metrics[f"{full_key}[{i}]"] = float(v)
            metrics[f"{full_key}.mean"] = sum(val) / len(val)
            metrics[f"{full_key}.min"] = min(val)
            metrics[f"{full_key}.max"] = max(val)

    return metrics


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def append_run(
    data_dir: Path,
    stage_timings: dict[str, dict[str, float | None]],
    dataset_id: str = "",
    config: Any = None,
    stage_results: dict[str, dict[str, Any]] | None = None,
    stage_validations: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Append one benchmark run to the cache.

    Args:
        data_dir: Dataset root directory.
        stage_timings: {stage_name: {"ref_seconds": float|None, "ffs_seconds": float|None}}.
            Only non-None values are stored.  May be empty for validate-only runs.
        dataset_id: Dataset identifier (e.g. "ds005165").
        config: Optional BenchmarkConfig — serialized via to_dict() if present.
        stage_results: {stage_name: {"passed": bool, "summary": str}}.
        stage_validations: {stage_name: <full validation dict from stage.validate()>}.
            Sanitized and stored alongside extracted scalar metrics for trend tracking.

    Every call creates a new entry — nothing is overwritten.
    """
    cache = load_cache(data_dir)
    hw = get_hardware_info()

    stages: dict[str, Any] = {}

    # Merge timings, results, and validation data per stage
    all_stage_names = set()
    if stage_timings:
        all_stage_names.update(stage_timings.keys())
    if stage_results:
        all_stage_names.update(stage_results.keys())
    if stage_validations:
        all_stage_names.update(stage_validations.keys())

    for sname in sorted(all_stage_names):
        entry: dict[str, Any] = {}

        # Timing
        if stage_timings and sname in stage_timings:
            for key, val in stage_timings[sname].items():
                if val is not None:
                    entry[key] = val

        # Pass/fail
        if stage_results and sname in stage_results:
            entry["passed"] = stage_results[sname].get("passed")
            entry["summary"] = stage_results[sname].get("summary", "")

        # Full validation + extracted scalar metrics
        if stage_validations and sname in stage_validations:
            raw = stage_validations[sname]
            entry["validation"] = _sanitize_for_cache(raw)
            entry["metrics"] = extract_scalar_metrics(
                entry["validation"],
            )

        if entry:
            stages[sname] = entry

    if not stages:
        return  # nothing to store

    run: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ref_arch_id": get_ref_arch_id(),
        "ffs_arch_id": get_ffs_arch_id(),
        "hardware": hw,
        "dataset_id": dataset_id,
        "git": _get_git_info(),
        "stages": stages,
    }
    if config is not None and hasattr(config, "to_dict"):
        run["config"] = config.to_dict()

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

    v = source.get("schema_version", 1)
    if v == 1:
        source = _migrate_v1(source)
    source["schema_version"] = SCHEMA_VERSION

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


# ---------------------------------------------------------------------------
# Historical comparison
# ---------------------------------------------------------------------------

def get_recent_runs(
    data_dir: Path,
    ffs_arch_id: str | None = None,
    dataset_id: str = "",
    max_runs: int = 20,
) -> list[dict[str, Any]]:
    """Get the most recent runs for a given FFS arch + dataset.

    If ffs_arch_id is None, uses the current machine's FFS arch.
    Returns newest-first.
    """
    if ffs_arch_id is None:
        ffs_arch_id = get_ffs_arch_id()

    cache = load_cache(data_dir)
    matching = []
    for run in cache.get("runs", []):
        if dataset_id and run.get("dataset_id", "") != dataset_id:
            continue
        if run.get("ffs_arch_id") != ffs_arch_id:
            continue
        matching.append(run)

    # Sort by timestamp descending (newest first)
    matching.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return matching[:max_runs]


def get_previous_run(
    data_dir: Path,
    current_commit: str = "",
    ffs_arch_id: str | None = None,
    dataset_id: str = "",
) -> dict[str, Any] | None:
    """Get the most recent previous run (different commit or earlier timestamp).

    Used for regression detection: compare current results against the last run.
    """
    recent = get_recent_runs(data_dir, ffs_arch_id, dataset_id, max_runs=50)
    for run in recent:
        run_commit = run.get("git", {}).get("commit", "")
        # Skip if same commit as current (we want the *previous* run)
        if current_commit and run_commit == current_commit:
            continue
        # Return the first (most recent) run that has metrics
        if any(
            run.get("stages", {}).get(s, {}).get("metrics")
            for s in run.get("stages", {})
        ):
            return run
    return None


@dataclass
class MetricDelta:
    """Change in a single metric between two runs."""
    name: str
    current: float
    previous: float
    delta: float
    pct_change: float  # percent change
    is_regression: bool  # True if metric got worse
    # For metrics like correlation, higher is better.
    # For metrics like angle_diff/trans_diff, lower is better.
    higher_is_better: bool = True


def compare_stage_metrics(
    current_metrics: dict[str, float],
    previous_metrics: dict[str, float],
    threshold_pct: float = 1.0,
) -> list[MetricDelta]:
    """Compare scalar metrics between two runs for the same stage.

    Returns a list of MetricDelta for metrics that exist in both runs.
    Only includes metrics where the change exceeds threshold_pct.

    Convention: metrics containing 'diff', 'msd', 'error' are "lower is better".
    Everything else (correlations, r², agreement) is "higher is better".
    """
    _LOWER_IS_BETTER = {"diff", "msd", "error", "rmsd", "nrmsd", "tolerance"}

    deltas = []
    common_keys = sorted(set(current_metrics) & set(previous_metrics))

    for key in common_keys:
        cur = current_metrics[key]
        prev = previous_metrics[key]
        delta = cur - prev

        # Determine direction
        lower_better = any(tok in key.lower() for tok in _LOWER_IS_BETTER)

        # Percent change (relative to previous, clamped to avoid div-by-zero)
        if abs(prev) > 1e-10:
            pct = (delta / abs(prev)) * 100
        else:
            pct = 0.0 if abs(delta) < 1e-10 else (100.0 if delta > 0 else -100.0)

        if abs(pct) < threshold_pct:
            continue

        # Regression = metric moved in the bad direction
        if lower_better:
            is_regression = delta > 0  # got bigger = worse
        else:
            is_regression = delta < 0  # got smaller = worse

        deltas.append(MetricDelta(
            name=key,
            current=cur,
            previous=prev,
            delta=delta,
            pct_change=pct,
            is_regression=is_regression,
            higher_is_better=not lower_better,
        ))

    return deltas


def get_metric_history(
    data_dir: Path,
    stage_name: str,
    metric_name: str,
    ffs_arch_id: str | None = None,
    dataset_id: str = "",
    max_points: int = 20,
) -> list[dict[str, Any]]:
    """Get the history of a single metric across runs.

    Returns a list of {commit_short, timestamp, value, passed} dicts,
    newest first.
    """
    recent = get_recent_runs(data_dir, ffs_arch_id, dataset_id, max_runs=max_points)
    history = []
    for run in recent:
        stage = run.get("stages", {}).get(stage_name, {})
        metrics = stage.get("metrics", {})
        if metric_name in metrics:
            git = run.get("git", {})
            history.append({
                "commit_short": git.get("commit_short", "?"),
                "commit": git.get("commit", ""),
                "branch": git.get("branch", ""),
                "timestamp": run.get("timestamp", "")[:19],
                "value": metrics[metric_name],
                "passed": stage.get("passed"),
            })
    return history
