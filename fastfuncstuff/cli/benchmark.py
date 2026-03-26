"""CLI entry point for ffs_benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ffs_benchmark",
        description=(
            "Benchmark AFNI vs FFS tools: accuracy validation and timing comparison.\n"
            "\n"
            "Requires OpenNeuro ds005165 data and existing AFNI+FFS outputs.\n"
            "Use --validate-only to compare existing outputs without re-running tools."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-help", action="store_true",
        help="Show this help message",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to ds005165-download directory. Default: auto-detect.",
    )
    parser.add_argument(
        "--stages", type=str, default=None,
        help="Comma-separated stage names (moco,slicetime,align,warp,glm,ica). "
             "Default: all.",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only validate existing outputs, don't run any tools.",
    )
    parser.add_argument(
        "--force-ffs", action="store_true",
        help="Re-run FFS tools even if outputs exist.",
    )
    parser.add_argument(
        "--force-afni", action="store_true",
        help="Re-run AFNI tools even if outputs exist.",
    )
    parser.add_argument(
        "--force-all", action="store_true",
        help="Re-run everything.",
    )
    parser.add_argument(
        "--json", type=str, default=None, metavar="PATH",
        help="Save results as JSON to this path.",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Print detailed timing report (requires timing data).",
    )
    parser.add_argument(
        "--plot", type=str, default=None, metavar="DIR",
        help="Save benchmark plots (timing bars, speedup chart) to this directory.",
    )
    parser.add_argument(
        "--plot-from-cache", type=str, nargs="*", default=None,
        metavar="CACHE_JSON",
        help="Generate plots from one or more benchmark_cache.json files. "
             "No stages are run. Multiple files are merged for cross-arch comparison.",
    )

    args = parser.parse_args(argv)

    if args.help or len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    return args


def _find_data_dir() -> Path | None:
    """Try to auto-detect the ds005165 data directory."""
    candidates = [
        Path("test_data/ds005165-download"),
        Path(__file__).resolve().parents[2] / "test_data" / "ds005165-download",
    ]
    for c in candidates:
        if c.exists() and (c / "processing").exists():
            return c
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # --- Plot-from-cache mode: just generate plots, no stages ---
    if args.plot_from_cache is not None:
        return _plot_from_cache(args)

    from ..benchmark.reporting import (
        print_timing_report,
        print_validation_report,
        save_json_report,
    )
    from ..benchmark.runner import BenchmarkContext, run_stages
    from ..benchmark.stages import get_stages

    # Resolve data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = _find_data_dir()
        if data_dir is None:
            print("ERROR: Cannot find ds005165-download directory.")
            print("Specify with --data-dir or run from the project root.")
            return 1

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return 1

    processing = data_dir / "processing"
    if not processing.exists():
        print(f"ERROR: Processing directory not found: {processing}")
        return 1

    # Resolve stages
    stage_names = None
    if args.stages:
        stage_names = [s.strip() for s in args.stages.split(",")]

    try:
        stages = get_stages(stage_names)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    # Build context
    force_all = args.force_all
    ctx = BenchmarkContext(
        data_dir=data_dir,
        force_afni=args.force_afni or force_all,
        force_ffs=args.force_ffs or force_all,
        validate_only=args.validate_only,
    )

    print(f"Data directory: {data_dir}")
    print(f"Stages: {', '.join(s.name for s in stages)}")
    print(f"Mode: {'validate-only' if ctx.validate_only else 'full'}")

    # Run
    results = run_stages(stages, ctx)

    # Enrich results with cached timings if available
    if args.report:
        from ..benchmark.timing_cache import get_cached_timing

        for r in results:
            if r.afni_time is None or r.ffs_time is None:
                cached_afni, cached_ffs = get_cached_timing(data_dir, r.stage_name)
                if r.afni_time is None and cached_afni is not None:
                    r.afni_time = cached_afni
                if r.ffs_time is None and cached_ffs is not None:
                    r.ffs_time = cached_ffs

    # Report
    if args.report:
        print_timing_report(results)
    else:
        print_validation_report(results)

    if args.json:
        save_json_report(results, args.json)

    # Plots
    if args.plot:
        from ..benchmark.plots import plot_all
        from ..benchmark.timing_cache import load_cache

        cache = load_cache(data_dir)
        plot_all(cache, output_dir=args.plot)

    # Exit code: 0 if all passed, 1 if any failed
    all_passed = all(r.passed for r in results)
    return 0 if all_passed else 1


def _plot_from_cache(args) -> int:
    """Generate plots from committed benchmark_cache.json files."""
    import json

    from ..benchmark.plots import plot_all

    cache_files = args.plot_from_cache
    if not cache_files:
        # Default: look for cache in data dir
        data_dir = Path(args.data_dir) if args.data_dir else _find_data_dir()
        if data_dir is None:
            print("ERROR: No cache files specified and cannot find data directory.")
            return 1
        cache_files = [str(data_dir / "benchmark_cache.json")]

    # Merge caches from multiple files
    merged = {"schema_version": 1, "runs": []}
    seen_archs = set()
    for cf in cache_files:
        path = Path(cf)
        if not path.exists():
            print(f"WARNING: Cache file not found: {cf}")
            continue
        with open(path) as f:
            data = json.load(f)
        for run in data.get("runs", []):
            arch_id = run.get("arch_id", "unknown")
            if arch_id not in seen_archs:
                merged["runs"].append(run)
                seen_archs.add(arch_id)
            else:
                # Merge stages into existing
                for existing in merged["runs"]:
                    if existing.get("arch_id") == arch_id:
                        existing["stages"].update(run.get("stages", {}))
                        break

    if not merged["runs"]:
        print("ERROR: No timing data found in cache files.")
        return 1

    output_dir = args.plot or "benchmark_plots"
    plot_all(merged, output_dir=output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
