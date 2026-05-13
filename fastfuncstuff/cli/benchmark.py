"""CLI entry point for ffs_benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fastfuncstuff.cli_utils import add_verbose_arg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ffs_benchmark",
        description=(
            "Benchmark AFNI vs FFS tools: accuracy validation and timing comparison.\n"
            "\n"
            "Supports arbitrary BIDS datasets via -config YAML files.\n"
            "Default: OpenNeuro ds005165 (sub-01, ses-01).\n"
            "Use -validate-only to compare existing outputs without re-running tools."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-config", type=str, default=None, metavar="YAML",
        help="Path to benchmark config YAML. Defines dataset, subject, session, "
             "tasks, runs, stages, and stage-specific params. "
             "Default: auto-detect from data directory or use built-in ds005165 config.",
    )
    parser.add_argument(
        "-data-dir", type=str, default=None,
        help="Path to BIDS dataset directory. Default: auto-detect.",
    )
    from ..benchmark.stages import STAGE_MAP as _STAGE_MAP
    _stage_names = ",".join(_STAGE_MAP)
    parser.add_argument(
        "-stages", type=str, default=None,
        help=f"Comma-separated stage names ({_stage_names}). Default: all.",
    )
    parser.add_argument(
        "-validate-only", action="store_true",
        help="Only validate existing outputs, don't run any tools.",
    )
    parser.add_argument(
        "-force-ffs", action="store_true",
        help="Re-run FFS tools even if outputs exist.",
    )
    parser.add_argument(
        "-force-ref", "-force-afni", action="store_true", dest="force_ref",
        help="Re-run reference tools (AFNI/melodic/MATLAB) even if outputs exist.",
    )
    parser.add_argument(
        "-force-all", action="store_true",
        help="Re-run everything.",
    )
    parser.add_argument(
        "-ref-only", action="store_true",
        help="Run reference tools only (AFNI/melodic/MATLAB). "
             "Skip FFS and validation. Useful for collecting ref timings "
             "on machines without GPU support (e.g. Mac).",
    )
    parser.add_argument(
        "-json", type=str, default=None, metavar="PATH",
        help="Save results as JSON to this path.",
    )
    parser.add_argument(
        "-report", action="store_true",
        help="Print detailed timing report (requires timing data).",
    )
    parser.add_argument(
        "-plot", type=str, default=None, metavar="DIR",
        help="Save benchmark plots (timing bars, speedup chart) to this directory.",
    )
    parser.add_argument(
        "-download", action="store_true",
        help="Download raw data for all benchmark datasets defined in built-in configs "
             "(ds005165, ds003427, etc.). Safe to re-run — skips datasets already present. "
             "Requires awscli (pip install awscli  or  brew install awscli).",
    )
    parser.add_argument(
        "-plot-from-cache", type=str, nargs="*", default=None,
        metavar="CACHE_JSON",
        help="Generate plots from one or more benchmark_cache.json files. "
             "No stages are run. Multiple files are merged for cross-arch comparison.",
    )
    parser.add_argument(
        "-list-cache", action="store_true",
        help="List all entries in benchmark_cache.json and exit.",
    )
    parser.add_argument(
        "-remove-cache", type=str, nargs="+", default=None, metavar="IDX_OR_ID",
        help="Remove cache entries by 1-based index (e.g. 3) or UUID prefix "
             "(e.g. a1b2c3d4). Use -list-cache first to see indices.",
    )
    parser.add_argument(
        "-import-cache", type=str, default=None, metavar="CACHE_JSON",
        help="Import runs from another benchmark_cache.json into the local cache. "
             "Deduplicates by UUID.",
    )
    parser.add_argument(
        "-dry_run", action="store_true",
        help="Preview -import-cache or -remove-cache without writing changes.",
    )
    parser.add_argument(
        "-device", type=str, default=None,
        help="PyTorch device passed to FFS tools: cpu, cuda, mps (default: auto-detect).",
    )
    add_verbose_arg(parser, default=0)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args(argv)
    return args


def _default_data_dir() -> Path:
    from .._paths import get_benchmark_data_dir
    return get_benchmark_data_dir() / "ds005165-download"


def _find_data_dir() -> Path | None:
    candidates = [
        _default_data_dir(),
        Path("test_data/ds005165-download"),
    ]
    for c in candidates:
        if c.exists() and (c / "sub-01").exists():
            return c
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # --- Plot-from-cache mode: just generate plots, no stages ---
    if args.plot_from_cache is not None:
        return _plot_from_cache(args)

    # --- Cache management modes (need data_dir, but no stages) ---
    if args.list_cache or args.remove_cache is not None or args.import_cache is not None:
        return _manage_cache(args)

    from ..benchmark.config import default_config, find_config, load_config
    from ..benchmark.reporting import (
        print_timing_report,
        print_validation_report,
        save_json_report,
    )
    from ..benchmark.runner import BenchmarkContext, run_stages
    from ..benchmark.stages import get_stages

    # Download all datasets first (global setup step)
    if args.download:
        _ensure_data()

    # Resolve data directory for this run
    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
    else:
        data_dir = _find_data_dir()
        if data_dir is None:
            print("ERROR: Cannot find data directory.")
            print("Specify with -data-dir, or use -download to fetch from OpenNeuro.")
            return 1

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Use -download to fetch from OpenNeuro.")
        return 1

    # Load config: explicit -config > auto-detect from data_dir > default
    if args.config:
        config_path = Path(args.config)
        try:
            config = load_config(config_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR: {e}")
            return 1
        print(f"Config: {config_path}")
    else:
        config_path = find_config(data_dir)
        if config_path is not None:
            try:
                config = load_config(config_path)
            except (FileNotFoundError, ValueError) as e:
                print(f"WARNING: Found config {config_path} but failed to load: {e}")
                config = default_config()
            else:
                print(f"Config: {config_path}")
        else:
            config = default_config()

    # Create processing dir if needed (from-zero runs won't have it yet)
    processing = data_dir / "processing"
    processing.mkdir(parents=True, exist_ok=True)

    # Resolve stages: CLI -stages overrides config, config overrides all-stages
    stage_names = None
    if args.stages:
        stage_names = [s.strip() for s in args.stages.split(",")]
    elif config.stages:
        stage_names = config.stages

    try:
        stages = get_stages(stage_names)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    # Build context
    force_all = args.force_all
    ctx = BenchmarkContext(
        data_dir=data_dir,
        force_ref=args.force_ref or force_all,
        force_ffs=args.force_ffs or force_all,
        validate_only=args.validate_only,
        ref_only=args.ref_only,
        device=args.device,
        show_output=args.verb >= 1,
        config=config,
    )

    # Print dataset header
    tasks_summary = ", ".join(
        f"{t}({len(r)} runs)" for t, r in ctx.all_task_run_pairs()
    )
    print(f"Dataset: {ctx.dataset_id}  sub-{ctx.subject}/ses-{ctx.session}  {tasks_summary}")
    print(f"Data directory: {data_dir}")
    print(f"Stages: {', '.join(s.name for s in stages)}")
    mode = "validate-only" if ctx.validate_only else "ref-only" if ctx.ref_only else "full"
    print(f"Mode: {mode}")
    if ctx.device:
        print(f"Device: {ctx.device}")
    if ctx.show_output:
        print("Verbose: streaming subprocess output")

    # Run
    results = run_stages(stages, ctx)

    # Enrich results with cached timings if available
    if args.report:
        from ..benchmark.timing_cache import get_cached_timing

        for r in results:
            if r.ref_time is None or r.ffs_time is None:
                cached_ref, cached_ffs = get_cached_timing(data_dir, r.stage_name)
                if r.ref_time is None and cached_ref is not None:
                    r.ref_time = cached_ref
                if r.ffs_time is None and cached_ffs is not None:
                    r.ffs_time = cached_ffs

    # Report
    if args.report:
        print_timing_report(results, data_dir=data_dir, config=ctx.config)
    else:
        print_validation_report(results, config=ctx.config, data_dir=data_dir)

    if args.json:
        save_json_report(results, args.json, config=ctx.config)

    # Plots
    if args.plot:
        from ..benchmark.plots import plot_all
        from ..benchmark.timing_cache import load_cache

        cache = load_cache(data_dir)
        plot_all(cache, output_dir=args.plot)

    # Exit code: 0 if all passed, 1 if any failed
    all_passed = all(r.passed for r in results)
    return 0 if all_passed else 1


def _manage_cache(args) -> int:
    """Handle -list-cache, -remove-cache, and -import-cache operations."""
    from ..benchmark.timing_cache import (
        merge_cache_from_file,
        print_cache,
        remove_cache_entries,
    )

    # Resolve data directory
    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
    else:
        data_dir = _find_data_dir()
        if data_dir is None:
            print("ERROR: Cannot find data directory. Specify with -data-dir.")
            return 1

    if args.list_cache:
        stage = args.stages.split(",")[0].strip() if args.stages else None
        print_cache(data_dir, stage_filter=stage)
        return 0

    if args.remove_cache is not None:
        indices = []
        ids = []
        for t in args.remove_cache:
            if t.isdigit():
                indices.append(int(t))
            else:
                ids.append(t)

        if not indices and not ids:
            print("ERROR: Provide at least one index or ID prefix to remove.")
            print("       Use -list-cache to see entries.")
            return 1

        remove_cache_entries(
            data_dir,
            indices=indices or None,
            ids=ids or None,
            dry_run=args.dry_run,
        )
        return 0

    if args.import_cache is not None:
        source = Path(args.import_cache)
        try:
            merge_cache_from_file(data_dir, source, dry_run=args.dry_run)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return 1
        return 0

    return 0


def _plot_from_cache(args) -> int:
    """Generate plots from one or more benchmark_cache.json files."""
    import json

    from ..benchmark.plots import plot_all
    from ..benchmark.timing_cache import _migrate_v1

    cache_files = args.plot_from_cache
    if not cache_files:
        data_dir = Path(args.data_dir) if args.data_dir else _find_data_dir()
        if data_dir is None:
            print("ERROR: No cache files specified and cannot find data directory.")
            return 1
        cache_files = [str(data_dir / "benchmark_cache.json")]

    # Merge all files: concatenate runs (v2 entries have UUIDs, so safe to concat)
    seen_ids: set[str] = set()
    merged: dict = {"schema_version": 2, "runs": []}
    for cf in cache_files:
        path = Path(cf)
        if not path.exists():
            print(f"WARNING: Cache file not found: {cf}")
            continue
        with open(path) as f:
            data = json.load(f)
        # Migrate v1 if needed
        if data.get("schema_version") != 2:
            data = _migrate_v1(data)
        for run in data.get("runs", []):
            uid = run.get("id", "")
            if uid and uid in seen_ids:
                continue  # skip exact duplicates across files
            seen_ids.add(uid)
            merged["runs"].append(run)

    if not merged["runs"]:
        print("ERROR: No timing data found in cache files.")
        return 1

    output_dir = args.plot or "benchmark_plots"
    plot_all(merged, output_dir=output_dir)
    return 0


def _ensure_data() -> None:
    """Download raw data for all datasets defined in built-in configs.

    Iterates every configs/*.yaml that has a ``download.s3_url`` entry and
    runs ``aws s3 sync`` for each one that is not already present.
    Manual-only datasets (no s3_url) print their instructions instead.
    """
    import shutil
    import subprocess

    from .._paths import get_benchmark_data_dir
    from ..benchmark.config import list_builtin_configs, load_config

    base = get_benchmark_data_dir()

    configs = list_builtin_configs()
    if not configs:
        print("No built-in configs found — nothing to download.")
        return

    for cfg_path in configs:
        try:
            cfg = load_config(cfg_path)
        except Exception as e:
            print(f"  WARNING: skipping {cfg_path.name}: {e}")
            continue

        dl = cfg.download
        if dl is None:
            continue

        data_dir = base / (dl.data_dir_name or f"{cfg.dataset_id}-download")

        if not dl.s3_url:
            if dl.instructions:
                print(f"\n[{cfg.dataset_id}] Manual dataset — no auto-download.")
                print(f"  {dl.instructions}")
            continue

        # Quick presence check — if the subject directory already exists, skip
        subj_dir = data_dir / f"sub-{cfg.subject}"
        if subj_dir.exists() and any(subj_dir.rglob("*.nii*")):
            print(f"[{cfg.dataset_id}] Data already present: {data_dir}")
            continue

        if not shutil.which("aws"):
            print("ERROR: awscli not found. Install with:  pip install awscli  or  brew install awscli")
            sys.exit(1)

        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{cfg.dataset_id}] Downloading to {data_dir} ...")

        cmd = ["aws", "s3", "sync", "--no-sign-request", dl.s3_url, str(data_dir),
               "--exclude", "*"]
        for pattern in dl.include:
            cmd += ["--include", pattern]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Download failed:\n{result.stderr[-500:]}")
            sys.exit(1)
        print(f"  Done.")


if __name__ == "__main__":
    sys.exit(main())
