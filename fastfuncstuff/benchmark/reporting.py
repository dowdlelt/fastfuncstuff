"""Benchmark reporting — terminal tables and summary output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .arch import get_arch_info, get_ffs_arch_id, get_ref_arch_id
from .runner import StageResult
from .timing_cache import get_ref_timings_all_archs


def _dataset_header(config: Any = None) -> str:
    """Build a dataset info string from config, if available."""
    if config is None:
        return ""
    parts = []
    if hasattr(config, "dataset_id") and config.dataset_id:
        parts.append(f"dataset={config.dataset_id}")
    if hasattr(config, "subject"):
        parts.append(f"sub-{config.subject}")
    if hasattr(config, "session"):
        parts.append(f"ses-{config.session}")
    if hasattr(config, "tasks") and config.tasks:
        task_strs = [f"{t}({len(r)} runs)" for t, r in config.tasks.items()]
        parts.append("tasks: " + ", ".join(task_strs))
    return "  ".join(parts)


def print_validation_report(
    results: list[StageResult],
    config: Any = None,
    data_dir: Path | None = None,
) -> None:
    """Print a validation-only report table to terminal.

    Includes detailed per-stage metrics and historical regression analysis
    when data_dir is provided.
    """
    from .timing_cache import _get_git_info

    ref_id = get_ref_arch_id()
    ffs_id = get_ffs_arch_id()
    git = _get_git_info()

    header = f"\nBENCHMARK VALIDATION  ref={ref_id}  ffs={ffs_id}"
    if git["commit_short"]:
        dirty = "*" if git["dirty"] else ""
        header += f"  commit={git['commit_short']}{dirty} ({git['branch']})"
    ds = _dataset_header(config)
    if ds:
        header += f"\n  {ds}"
    print(header)
    print("=" * 80)
    print(f"{'Stage':<16} {'Status':<8} {'Summary'}")
    print("-" * 80)

    n_pass = 0
    n_fail = 0
    n_incomplete = 0
    for r in results:
        status = r.status
        if r.incomplete:
            n_incomplete += 1
        elif r.passed:
            n_pass += 1
        else:
            n_fail += 1
        print(f"{r.stage_name:<16} {status:<11} {r.summary}")
        for err in r.errors:
            print(f"  ERROR: {err}")

        if r.stage_name == "moco" and r.validation:
            _print_moco_detail(r.validation)

    print("-" * 80)
    tail = f", {n_incomplete} incomplete" if n_incomplete else ""
    print(f"Total: {n_pass} passed, {n_fail} failed{tail} out of {len(results)} stages")

    # Timing comparison (Ref/FFS/Speedup) — only if any stage has timing data.
    # Mirrors the table in print_timing_report so users don't need to pass
    # -report to see how FFS stacks up against the reference tool.
    if data_dir is not None:
        _print_timing_summary(results, data_dir)

    # Detailed metrics + historical comparison
    _print_stage_details(results)
    if data_dir:
        _print_regression_analysis(results, data_dir, git)


def _print_timing_summary(
    results: list[StageResult],
    data_dir: Path,
) -> None:
    """Compact Ref/FFS/Speedup table; cached refs pulled in for missing live values.

    Skipped entirely when no stage has any timing data — keeps validate-only
    runs noise-free when they're truly validation-only (no commands timed).
    """
    my_ref_id = get_ref_arch_id()

    cached_refs: dict[str, list[tuple[str, float]]] = {}
    for r in results:
        if not (r.ref_time and r.ref_time > 0):
            cached_refs[r.stage_name] = get_ref_timings_all_archs(data_dir, r.stage_name)

    # Also pull cached glmsingle_matlab ref for the pipeline-aggregate row, in
    # case MATLAB wasn't re-run this invocation (common: run MATLAB once, then
    # iterate on the FFS stages).
    if _GLMSINGLE_REF_STAGE not in cached_refs:
        cached_refs[_GLMSINGLE_REF_STAGE] = get_ref_timings_all_archs(
            data_dir, _GLMSINGLE_REF_STAGE
        )

    # Resolve a display ref for each stage (live, then cached-local, then cached-other)
    rows: list[tuple[str, float, float, str]] = []  # (name, ref_t, ffs_t, ref_marker)
    any_timing = False
    any_local_cached = False
    any_other_cached = False
    for r in results:
        ffs_t = r.ffs_time or 0.0
        ref_t = r.ref_time or 0.0
        marker = ""
        if ref_t <= 0 and r.stage_name in cached_refs:
            all_refs = cached_refs[r.stage_name]
            local = next((rv for av, rv in all_refs if av == my_ref_id), None)
            if local is not None:
                ref_t, marker = local, "†"
                any_local_cached = True
            elif all_refs:
                _, ref_t = all_refs[0]
                marker = "‡"
                any_other_cached = True
        if ref_t > 0 or ffs_t > 0:
            any_timing = True
        rows.append((r.stage_name, ref_t, ffs_t, marker))

    if not any_timing:
        return

    print(f"\n{' TIMING (Ref vs FFS) ':=^80}")
    print(f"  {'Stage':<16} {'Ref (s)':>12} {'FFS (s)':>10} {'Speedup':>10}")
    print(f"  {'-' * 50}")

    for sname, ref_t, ffs_t, marker in rows:
        ref_str = f"{ref_t:.1f}{marker}" if ref_t > 0 else "-"
        ffs_str = f"{ffs_t:.1f}" if ffs_t > 0 else "-"
        speedup = f"{ref_t / ffs_t:.1f}x" if (ref_t > 0 and ffs_t > 0) else "-"
        print(f"  {sname:<16} {ref_str:>12} {ffs_str:>10} {speedup:>10}")

        # GLMsingle pipeline aggregate after the last FFS stage in that pipeline
        if sname == _GLMSINGLE_FFS_STAGES[-1]:
            agg = _glmsingle_aggregate(results, cached_refs, my_ref_id)
            if agg is not None:
                a_ref, a_ffs, a_local, a_other = agg
                a_marker = "†" if a_local else ("‡" if a_other else "")
                print(
                    f"  {'glmsingle(tot)':<16} {f'{a_ref:.1f}{a_marker}':>12} "
                    f"{a_ffs:>10.1f} {a_ref / a_ffs:>9.1f}x  (hrf+denoise+ridge vs matlab)"
                )
                if a_local:
                    any_local_cached = True
                elif a_other:
                    any_other_cached = True

    if any_local_cached:
        print(f"  † cached ref timing (CPU arch: {my_ref_id})")
    if any_other_cached:
        print("  ‡ cached ref timing from a different CPU architecture")


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
        print(
            f"  Timeseries MSD (nrmsd: mean={msd_data.get('mean_nrmsd', 0):.4f}, "
            f"max={msd_data.get('max_nrmsd', 0):.4f}):"
        )
        for entry in msd_runs:
            if "error" in entry:
                print(
                    f"    {entry.get('task', '?')} run-{entry.get('run', '?')}: ERROR {entry['error']}"
                )
            else:
                task, run = entry.get("task", "?"), entry.get("run", "?")
                print(
                    f"    {task} run-{run}: msd={entry['mean_msd']:.2f}, nrmsd={entry['nrmsd']:.4f}"
                )


_GLMSINGLE_FFS_STAGES = ("glmsingle_hrf", "glmsingle_denoise", "glmsingle_ridge")
_GLMSINGLE_REF_STAGE = "glmsingle_matlab"


def _glmsingle_aggregate(
    results: list[StageResult],
    cached_refs: dict[str, list[tuple[str, float]]],
    my_ref_id: str,
) -> tuple[float, float, bool, bool] | None:
    """Return (ref_t, ffs_t, from_cached_local, from_cached_other) for GLMsingle pipeline.

    Returns None if any FFS stage time is missing or the ref stage is absent.
    ref_t  = glmsingle_matlab ref_time (live or cached)
    ffs_t  = sum of hrf + denoise + ridge ffs_times
    """
    by_name = {r.stage_name: r for r in results}

    # Sum FFS pipeline times
    ffs_total = 0.0
    for stage in _GLMSINGLE_FFS_STAGES:
        r = by_name.get(stage)
        t = (r.ffs_time or 0.0) if r else 0.0
        if t <= 0:
            return None  # incomplete — don't show a misleading aggregate
        ffs_total += t

    # Get ref time (live or cached)
    ref_result = by_name.get(_GLMSINGLE_REF_STAGE)
    ref_t = (ref_result.ref_time or 0.0) if ref_result else 0.0
    from_cached_local = from_cached_other = False

    if ref_t <= 0:
        # Try cache — caller pre-populates cached_refs for all stages including
        # glmsingle_matlab (even though it has no ffs_time, so it was previously
        # excluded from the cache prefetch loop).
        all_refs = cached_refs.get(_GLMSINGLE_REF_STAGE) or []
        local = next((rv for av, rv in all_refs if av == my_ref_id), None)
        if local is not None:
            ref_t = local
            from_cached_local = True
        elif all_refs:
            _, ref_t = all_refs[0]
            from_cached_other = True
        else:
            return None  # no ref timing available at all

    return ref_t, ffs_total, from_cached_local, from_cached_other


def print_timing_report(
    results: list[StageResult],
    data_dir: Path | None = None,
    config: Any = None,
) -> None:
    """Print a full timing + validation report table.

    When data_dir is provided, missing ref timings are pulled from the cache
    (queried by CPU arch for ref, GPU for FFS) and annotated:
      †  = cached from this CPU architecture
      ‡  = cached from a different CPU architecture

    Also prints detailed per-stage metrics and regression analysis.
    """
    from .timing_cache import _get_git_info

    my_ref_id = get_ref_arch_id()
    my_ffs_id = get_ffs_arch_id()
    git = _get_git_info()

    # Pre-fetch cached ref timings for stages where ref didn't run this time.
    # Also always fetch for glmsingle_matlab (it has no ffs_time so it's normally
    # skipped by the loop below, but we need it for the aggregate row).
    cached_refs: dict[str, list[tuple[str, float]]] = {}
    if data_dir is not None:
        for r in results:
            if not (r.ref_time and r.ref_time > 0):
                cached_refs[r.stage_name] = get_ref_timings_all_archs(data_dir, r.stage_name)

    header = f"\nBENCHMARK  ref={my_ref_id}  ffs={my_ffs_id}"
    if git["commit_short"]:
        dirty = "*" if git["dirty"] else ""
        header += f"  commit={git['commit_short']}{dirty} ({git['branch']})"
    ds = _dataset_header(config)
    if ds:
        header += f"\n  {ds}"
    print(header)
    print("=" * 80)
    print(f"{'Stage':<16} {'Ref (s)':>12} {'FFS (s)':>10} {'Speedup':>10} {'Status'}")
    print("-" * 80)

    total_ref = 0.0
    total_ffs = 0.0
    any_cached_ref = False
    any_other_arch_ref = False

    for r in results:
        status = r.status
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
        print(
            f"{r.stage_name:<16} {ref_str:>12} {ffs_str:>10} {speedup_str:>10} {status}  {summary_short}"
        )

        # After the last GLMsingle FFS stage, print the pipeline aggregate row
        if r.stage_name == _GLMSINGLE_FFS_STAGES[-1]:
            agg = _glmsingle_aggregate(results, cached_refs, my_ref_id)
            if agg is not None:
                agg_ref, agg_ffs, agg_local, agg_other = agg
                agg_speedup = f"{agg_ref / agg_ffs:.1f}x"
                if agg_local:
                    agg_ref_str = f"{agg_ref:.1f}†"
                    any_cached_ref = True
                elif agg_other:
                    agg_ref_str = f"{agg_ref:.1f}‡"
                    any_other_arch_ref = True
                else:
                    agg_ref_str = f"{agg_ref:.1f}"
                agg_ffs_str = f"{agg_ffs:.1f}"
                print(
                    f"{'glmsingle(tot)':<16} {agg_ref_str:>12} {agg_ffs_str:>10} {agg_speedup:>10}        (hrf+denoise+ridge vs matlab)"
                )

    print("-" * 80)
    if total_ffs > 0:
        print(f"{'Total':<16} {total_ref:>12.1f} {total_ffs:>10.1f} {total_ref / total_ffs:>9.1f}x")

    if any_cached_ref:
        print(f"  † cached ref timing (CPU arch: {my_ref_id})")
    if any_other_arch_ref:
        print("  ‡ cached ref timing from a different CPU architecture")

    # Detailed metrics + historical comparison
    _print_stage_details(results)
    if data_dir:
        _print_regression_analysis(results, data_dir, git)


# ---------------------------------------------------------------------------
# Detailed per-stage metrics
# ---------------------------------------------------------------------------

# Key metrics to display per stage (in display order).
# Each entry: (metric_key_substring, display_name, format_spec)
# If a stage has metrics matching these patterns, they're shown.
_METRIC_DISPLAY = [
    # Correlations / R²
    ("min_r", "min r", ".4f"),
    ("mean_r", "mean r", ".4f"),
    ("median_r", "median r", ".4f"),
    ("overall_mean_r", "overall mean r", ".4f"),
    ("r2_spatial_corr", "R² spatial r", ".4f"),
    ("hrf_index_agreement", "HRF agree", ".1%"),
    ("fracvalue_corr", "frac r", ".4f"),
    ("beta_spatial_corr", "beta r", ".4f"),
    ("beta_timeseries_corr", "beta ts r", ".4f"),
    ("noisepool_jaccard", "noise Jaccard", ".3f"),
    ("xvaltrend_corr", "xval r", ".3f"),
    # Differences
    ("max_angle_diff", "max angle diff", ".3f"),
    ("max_trans_diff", "max trans diff", ".3f"),
    ("mean_angle_diff", "mean angle diff", ".3f"),
    ("mean_trans_diff", "mean trans diff", ".3f"),
    # Counts
    ("pcnum_diff", "PC num diff", ".0f"),
    ("n_valid_runs", "valid runs", ".0f"),
    ("n_voxels", "voxels", "d"),
    # Coverage
    ("coverage_0.5", "cov@0.5", ".2f"),
    ("overall_coverage_0.5", "overall cov@0.5", ".2f"),
    # MSD
    ("mean_nrmsd", "mean nrmsd", ".4f"),
    ("max_nrmsd", "max nrmsd", ".4f"),
    # ICA varnorm scale/offset
    ("vn_scale", "vn scale", ".4f"),
    ("vn_nrmse", "vn nrmse", ".4f"),
]


def _print_stage_details(results: list[StageResult]) -> None:
    """Print detailed metrics for each stage."""
    from .timing_cache import extract_scalar_metrics

    has_details = False
    for r in results:
        if not r.validation:
            continue
        metrics = extract_scalar_metrics(r.validation)
        if not metrics:
            continue

        if not has_details:
            print(f"\n{'DETAILED METRICS':=^80}")
            has_details = True

        print(f"\n  {r.stage_name}")
        print(f"  {'-' * 40}")

        # Show key metrics first (matched by display table)
        shown = set()
        for pattern, display_name, fmt in _METRIC_DISPLAY:
            for key, val in sorted(metrics.items()):
                if key in shown:
                    continue
                if pattern in key:
                    # Include parent context for disambiguation
                    parts = key.split(".")
                    if len(parts) > 1 and parts[-1] == pattern:
                        label = f"{parts[-2]}.{display_name}"
                    elif key != pattern:
                        label = key
                    else:
                        label = display_name
                    try:
                        val_str = f"{val:{fmt}}"
                    except (ValueError, TypeError):
                        val_str = str(val)
                    print(f"    {label:<30} {val_str}")
                    shown.add(key)

        # Show remaining metrics not in the display table
        remaining = {k: v for k, v in sorted(metrics.items()) if k not in shown}
        # Group by top-level key
        groups: dict[str, list[tuple[str, float]]] = {}
        for key, val in remaining.items():
            parts = key.split(".", 1)
            group = parts[0] if len(parts) > 1 else ""
            groups.setdefault(group, []).append((key, val))

        for group, items in groups.items():
            if len(items) > 8:
                # Summarize large groups (e.g. per-run arrays)
                vals = [v for _, v in items]
                print(
                    f"    {group:<30} [{len(items)} values] "
                    f"mean={sum(vals) / len(vals):.4f} "
                    f"range=[{min(vals):.4f}, {max(vals):.4f}]"
                )
            else:
                for key, val in items:
                    # Use full dotted key for disambiguation
                    print(f"    {key:<30} {val:.4f}")


# ---------------------------------------------------------------------------
# Regression analysis
# ---------------------------------------------------------------------------


def _print_regression_analysis(
    results: list[StageResult],
    data_dir: Path,
    current_git: dict[str, Any],
) -> None:
    """Compare current metrics against the previous run and flag regressions."""
    from .timing_cache import (
        compare_stage_metrics,
        extract_scalar_metrics,
        get_previous_run,
    )

    current_commit = current_git.get("commit", "")
    prev_run = get_previous_run(data_dir, current_commit=current_commit)
    if prev_run is None:
        return  # no history to compare against

    prev_git = prev_run.get("git", {})
    prev_commit = prev_git.get("commit_short", "?")
    prev_ts = prev_run.get("timestamp", "")[:19].replace("T", " ")
    prev_msg = prev_git.get("message", "")[:50]

    print(f"\n{'REGRESSION ANALYSIS':=^80}")
    print(f"  Comparing against: {prev_commit} ({prev_ts})")
    if prev_msg:
        print(f"  Previous commit: {prev_msg}")

    any_regression = False
    any_improvement = False
    n_stages_compared = 0

    for r in results:
        if not r.validation:
            continue
        current_metrics = extract_scalar_metrics(r.validation)
        if not current_metrics:
            continue

        prev_stage = prev_run.get("stages", {}).get(r.stage_name, {})
        prev_metrics = prev_stage.get("metrics", {})
        if not prev_metrics:
            continue

        deltas = compare_stage_metrics(current_metrics, prev_metrics, threshold_pct=0.1)
        if not deltas:
            continue

        n_stages_compared += 1
        regressions = [d for d in deltas if d.is_regression]
        improvements = [d for d in deltas if not d.is_regression]

        if regressions:
            any_regression = True
        if improvements:
            any_improvement = True

        # Only print stages that have changes
        if not regressions and not improvements:
            continue

        prev_passed = prev_stage.get("passed")
        status_change = ""
        if prev_passed is not None and prev_passed != r.passed:
            if r.passed and not prev_passed:
                status_change = "  [FIXED]"
            elif not r.passed and prev_passed:
                status_change = "  [BROKEN]"

        print(f"\n  {r.stage_name}{status_change}")

        if regressions:
            for d in sorted(regressions, key=lambda x: abs(x.pct_change), reverse=True)[:5]:
                arrow = "v" if d.higher_is_better else "^"
                print(
                    f"    {arrow} {d.name:<30} {d.previous:.4f} -> {d.current:.4f}  "
                    f"({d.pct_change:+.1f}%)"
                )

        if improvements:
            for d in sorted(improvements, key=lambda x: abs(x.pct_change), reverse=True)[:5]:
                arrow = "^" if d.higher_is_better else "v"
                print(
                    f"    {arrow} {d.name:<30} {d.previous:.4f} -> {d.current:.4f}  "
                    f"({d.pct_change:+.1f}%)"
                )

    if n_stages_compared == 0:
        print("  No previous metrics to compare against.")
    elif not any_regression and not any_improvement:
        print("  All metrics stable (< 0.1% change).")
    else:
        print(f"\n  {'=' * 60}")
        parts = []
        if any_regression:
            parts.append("REGRESSIONS DETECTED")
        if any_improvement:
            parts.append("improvements found")
        print(f"  {' | '.join(parts)}")

    # FFS timing trend (if we have historical data)
    _print_ffs_timing_trend(results, data_dir)


def _print_ffs_timing_trend(
    results: list[StageResult],
    data_dir: Path,
) -> None:
    """Show FFS timing trend across recent runs."""
    from .timing_cache import get_recent_runs

    # Build a table of FFS timings per stage across commits
    stage_names = [r.stage_name for r in results if r.ffs_time and r.ffs_time > 0]
    if not stage_names:
        return

    # Pull a larger window so we can drop runs that didn't touch any of the
    # stages we care about (e.g. previous invocations that benchmarked only a
    # different sub-pipeline) and still show ~9 useful columns of history.
    max_cols = 9
    raw_recent = get_recent_runs(data_dir, max_runs=50)
    recent = [
        run
        for run in raw_recent
        if any(run.get("stages", {}).get(s, {}).get("ffs_seconds") is not None for s in stage_names)
    ][:max_cols]
    if len(recent) < 2:
        return

    print(f"\n{f' FFS TIMING TREND (last {len(recent)} runs with these stages) ':=^80}")
    # Header: commit hashes
    commits = []
    for run in recent:
        git = run.get("git", {})
        short = git.get("commit_short", "?")
        dirty = "*" if git.get("dirty") else ""
        commits.append(f"{short}{dirty}")

    header = f"  {'Stage':<16}" + "".join(f"{c:>12}" for c in commits)
    print(header)
    print(f"  {'-' * (16 + 12 * len(commits))}")

    for sname in stage_names:
        row = f"  {sname:<16}"
        for run in recent:
            t = run.get("stages", {}).get(sname, {}).get("ffs_seconds")
            if t is not None:
                row += f"{t:>11.1f}s"
            else:
                row += f"{'-':>12}"
        print(row)


def _short_arch(arch_id: str) -> str:
    """Shorten an arch_id for compact display.

    Handles both v2 IDs (ref_arch_id like "linux-x86_64", ffs_arch_id like
    "cuda-NVIDIA_GeForce_RTX_5070_Ti") and legacy combined IDs.
    """
    for prefix in ("cuda-NVIDIA_GeForce_", "cuda-NVIDIA_", "cuda-", "mps-Apple_", "mps-"):
        if arch_id.startswith(prefix):
            return arch_id[len(prefix) :].replace("_", " ")
    # linux-x86_64 or darwin-arm64 style
    parts = arch_id.split("-", 1)
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}"
    return arch_id


def results_to_json(results: list[StageResult], config: Any = None) -> dict[str, Any]:
    """Convert results to a JSON-serializable dict."""
    from .timing_cache import _get_git_info, extract_scalar_metrics

    arch_info = get_arch_info()
    stages = {}
    for r in results:
        entry: dict[str, Any] = {
            "passed": r.passed,
            "summary": r.summary,
            "ref_seconds": r.ref_time,
            "ffs_seconds": r.ffs_time,
            "validation": _sanitize_for_json(r.validation),
            "errors": r.errors,
        }
        if r.validation:
            entry["metrics"] = extract_scalar_metrics(
                _sanitize_for_json(r.validation),
            )
        stages[r.stage_name] = entry

    result: dict[str, Any] = {
        "architecture": arch_info,
        "git": _get_git_info(),
        "stages": stages,
    }
    if config is not None and hasattr(config, "to_dict"):
        result["config"] = config.to_dict()

    return result


def save_json_report(results: list[StageResult], path: str | Path, config: Any = None) -> None:
    """Save results as JSON."""
    data = results_to_json(results, config=config)
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
