"""Benchmark runner: plan units into the worklist, then work through them.

    python -m fastfuncstuff.benchmark.work -plan moco -datasets ds002603 -n 2
    python -m fastfuncstuff.benchmark.work -run [-check moco] [-max_units 5]
    python -m fastfuncstuff.benchmark.work -run -device cpu -workers 0 -threads 6
    python -m fastfuncstuff.benchmark.work -status
    python -m fastfuncstuff.benchmark.work -results moco

Each unit: fetch inputs (S3 into R/cache, local trees in place) -> reference -> ffs ->
score -> record -> prune regenerable outputs and downloaded inputs. ``-keep`` skips pruning.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from fastfuncstuff._paths import get_bench_root
from fastfuncstuff.cli_help import FfsArgumentParser


def worker_info() -> dict:
    """Which worker ran the unit, on which CPUs, with how many threads."""
    import os
    import socket

    return {
        "hostname": socket.gethostname(),
        "worker": os.environ.get("FFS_BENCH_WORKER", "0/1"),
        "cpus": os.environ.get("FFS_BENCH_CPUS")
        or ",".join(map(str, sorted(os.sched_getaffinity(0)))),
        "threads": int(os.environ.get("OMP_NUM_THREADS") or 0) or None,
    }


def _manifest(manifests: Path, dataset: str) -> dict:
    from .census import apply_curation, load_manifests

    for m in load_manifests(manifests, curated=False):
        if m["dataset_id"] == dataset:
            return apply_curation(m)
    raise KeyError(f"no manifest for {dataset} in {manifests}")


def plan(wl, manifests: Path, check_name: str, datasets: list[str], n: int) -> None:
    from .census import Index
    from .checks import CHECKS
    from .fetch import source_from_manifest

    check = CHECKS[check_name]()
    for ds in datasets:
        m = _manifest(manifests, ds)
        if m.get("excluded"):
            print(f"{ds}: excluded ({m['excluded']}), skipped")
            continue
        index = Index(source_from_manifest(m).list_files())
        from .census import unit_exclusion

        units = [u for u in check.plan(m, index, n) if not unit_exclusion(m, check.name, u[0])]
        added = wl.add_units(check.name, ds, units)
        print(f"{ds}: {len(units)} units planned for {check.name}, {added} new")


def deadline_from(hhmm: str | None) -> float | None:
    """Next occurrence of HH:MM local time, as a timestamp (tomorrow if already past)."""
    if not hhmm:
        return None
    import datetime as dt

    h, m = (int(x) for x in hhmm.split(":"))
    now = dt.datetime.now()
    t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if t <= now:
        t += dt.timedelta(days=1)
    return t.timestamp()


def gpu_busy(device: str, min_free_gb: float = 6.0, util_pct: float = 20.0) -> str | None:
    """Why the card should be left alone right now, or None.

    Called between units, when this runner has nothing on the card, so any sustained
    utilization belongs to someone else's fit. Idle contexts (a viewer, a notebook kernel)
    hold a few hundred MB at 0 % and must not block a night's run."""
    import subprocess
    import time

    if not device.startswith("cuda"):
        return None
    idx = device.split(",")[1] if "," in device else "0"
    q = ["nvidia-smi", f"--id={idx}", "--query-gpu=utilization.gpu,memory.used,memory.total",
         "--format=csv,noheader,nounits"]  # fmt: skip
    samples = []
    for _ in range(5):
        try:
            out = subprocess.run(q, capture_output=True, text=True, timeout=30).stdout
            u, used, total = (float(x) for x in out.strip().split(","))
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return None
        samples.append((u, used, total))
        time.sleep(2)
    util = min(s[0] for s in samples)
    free_gb = (samples[-1][2] - samples[-1][1]) / 1024
    if util >= util_pct:
        return f"utilization {util:.0f}% sustained over 10 s"
    if free_gb < min_free_gb:
        return f"only {free_gb:.1f} GB free"
    return None


def _load_checks(previous: dict) -> dict:
    """The checks registry, re-imported from disk between units.

    A night's runner then picks up checks written (and committed) while it runs, and
    their planned units join the queue. A broken edit mid-night keeps the last good set
    rather than stopping the loop."""
    import importlib
    import sys

    names = sorted(n for n in sys.modules if n.startswith(f"{__package__}.checks."))
    try:
        if names:
            base = f"{__package__}.checks.base"
            for n in [base] + [n for n in names if n != base]:
                importlib.reload(sys.modules[n])
            pkg = importlib.reload(sys.modules[f"{__package__}.checks"])
        else:
            pkg = importlib.import_module(f"{__package__}.checks")
        return dict(pkg.CHECKS)
    except Exception as e:
        if not previous:
            raise
        print(f"    checks reload failed ({type(e).__name__}: {e}); keeping the last set")
        return previous


def run(
    wl,
    root: Path,
    manifests: Path,
    check: str | None,
    max_units: int,
    device: str,
    keep: bool,
    until: float | None = None,
    gpu_wait_s: float = 300.0,
) -> int:
    import time

    from .fetch import fetch, sha256

    checks: dict = {}
    n_done = 0
    while n_done < max_units:
        checks = _load_checks(checks)
        if until is not None and time.time() >= until:
            print("stop time reached; leaving the rest pending", flush=True)
            break
        busy = gpu_busy(device)
        if busy:
            print(f"GPU in use by someone else ({busy}); waiting {gpu_wait_s:.0f} s", flush=True)
            time.sleep(gpu_wait_s)
            continue
        u = wl.claim_next(check=check, known_checks=checks)
        if u is None:
            break
        key = json.loads(u["unit_key"])
        label = f"[{u['id']}] {u['check_name']} {u['dataset']} {' '.join(f'{k}-{v}' for k, v in key.items())}"
        print(label, flush=True)
        work = root / "units" / u["check_name"] / u["dataset"] / f"unit{u['id']:06d}"
        from .census import unit_exclusion

        why = unit_exclusion(_manifest(manifests, u["dataset"]), u["check_name"], key)
        if why:  # curated after the unit was planned
            wl.finish(u["id"], "excluded", why)
            print(f"    excluded: {why}", flush=True)
            continue
        try:
            m = _manifest(manifests, u["dataset"])
            paths, _ = fetch(m, json.loads(u["inputs"]), root / "cache")
            # Every cached input, not only the ones this unit downloaded: with parallel
            # workers another unit may have fetched it, and the last user prunes it.
            cached = [p for p in paths.values() if (root / "cache") in p.parents]
            wl.record_provenance(
                u["id"],
                "inputs",
                {rel: sha256(p) for rel, p in paths.items()} | {"source": m["source"]["uri"]},
            )
            from .census import unit_curation

            chk = checks[u["check_name"]]()
            chk.curation = unit_curation(m, key.get("acq"))
            if chk.curation["sidecar"] or chk.curation["noise_volumes"] is not None or chk.curation["events_columns"]:  # fmt: skip
                wl.record_provenance(u["id"], "curation", chk.curation)
            res = chk.run(key, paths, work, device)
            for judge, vals in res.metrics.items():
                wl.record_metrics(u["id"], judge, vals)
            for side, dev, sec, thr in res.timings:
                wl.record_timing(u["id"], side, dev, sec, thr)
            for path, retention in res.artifacts:
                wl.record_artifact(u["id"], path, retention)
            for path in cached:
                wl.record_artifact(u["id"], path, "regenerable")
            for side, vals in res.provenance.items():
                wl.record_provenance(u["id"], side, vals)
            from ..benchmark.arch import get_hardware_info

            wl.record_provenance(u["id"], "host", get_hardware_info(device))
            wl.record_provenance(u["id"], "worker", worker_info())
            wl.finish(u["id"], "done")
            t = {s: sec for s, _, sec, _ in res.timings}
            ref_m = res.metrics.get("reference", {})
            brief = ", ".join(f"{k}={v:.4g}" for k, v in list(ref_m.items())[:4] if v is not None)
            print(
                f"    done: ref {t.get('ref', 0):.1f} s, ffs {t.get('ffs', 0):.1f} s | {brief}",
                flush=True,
            )
        except Exception as e:  # a bad unit is recorded, the loop goes on
            wl.finish(
                u["id"], "failed", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"
            )
            print(f"    FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
        if not keep:
            # A host set keeps its outputs for inspection unless exported with -prune_outputs;
            # outputs marked keep (warps, label maps) always stay.
            keep_out = wl.meta("hostset") and wl.meta("prune_outputs") != "yes"
            freed = wl.prune(u["id"], under=root / "cache" if keep_out else None)
            if freed:
                print(f"    pruned {freed / 1e6:.0f} MB", flush=True)
        n_done += 1
    return n_done


def redo(wl, root: Path, check: str, tools: list[str]) -> int:
    """Send finished units of ``check`` back to pending with ``tools`` marked stale.

    For a tool whose code changed while its command line did not: ``timed_cached`` keys on
    the command, so a rerun would reuse the old fit. Removing just that tool's stamp makes
    it refit, while the reference fits (hours of @SSwarper or 3dQwarp) are reused, as long
    as their outputs are still on disk. A tool is its unit subfolder, ``ffs_<tool>`` or
    ``<tool>``. Returns the number of units sent back."""
    rows = wl.db.execute(
        "SELECT id, dataset FROM units WHERE check_name=? AND status IN ('done', 'failed')",
        (check,),
    ).fetchall()
    seen: set[str] = set()
    for r in rows:
        work = root / "units" / check / r["dataset"] / f"unit{r['id']:06d}"
        for t in tools:
            for d in (work / f"ffs_{t}", work / t):
                if d.is_dir():
                    seen.add(t)
                    for stamp in ("fit.timing.json", "stamp.json"):  # timed_cached stamps
                        (d / stamp).unlink(missing_ok=True)
                    break
    if rows and (missing := [t for t in tools if t not in seen]):
        raise ValueError(f"no unit of {check} has a folder for {missing} (ffs_<tool> or <tool>)")
    with wl._tx():
        wl.db.executemany(
            "UPDATE units SET status='pending', claimed_by=NULL, error=NULL WHERE id=?",
            [(r["id"],) for r in rows],
        )
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    p = FfsArgumentParser(prog="python -m fastfuncstuff.benchmark.work", description=__doc__)
    p.add_argument(
        "-root",
        type=Path,
        default=get_bench_root(),
        help="Benchmark root (worklist.sqlite, cache/, units/). Default: FFS_BENCH_ROOT, "
        "else bench/ beside FFS_BENCHMARK_DATA_DIR.",
    )
    p.add_argument(
        "-manifests", type=Path, help="Directory of census manifests (default ROOT/manifests)."
    )
    act = p.add_mutually_exclusive_group(required=True)
    act.add_argument("-plan", metavar="CHECK", help="Add units of CHECK for -datasets.")
    act.add_argument("-run", action="store_true", help="Work through pending units.")
    act.add_argument("-status", action="store_true", help="Unit counts by check, dataset, status.")
    act.add_argument("-results", metavar="CHECK", help="Print finished units of CHECK as TSV.")
    act.add_argument("-retry_failed", action="store_true", help="Return failed units to pending.")
    act.add_argument(
        "-reset_running",
        action="store_true",
        help="Return units left 'running' by a killed run to pending. Only when no -run is "
        "active on this root.",
    )
    act.add_argument(
        "-redo",
        metavar="CHECK",
        help="Return finished units of CHECK to pending and refit -tools on the next -run "
        "(their code changed, their command line did not); other fits are reused.",
    )
    act.add_argument(
        "-export_hostset",
        metavar="DIR",
        type=Path,
        help="Write a host set to DIR: -n finished units of each of -checks (default 2), from "
        "public datasets, plus their manifests. Run it elsewhere with FFS_BENCH_ROOT=DIR.",
    )
    act.add_argument(
        "-ingest", metavar="DIR", type=Path, help="Register a synced host set (re-ingest replaces)."
    )
    act.add_argument("-hosts", action="store_true", help="List ingested host sets.")
    act.add_argument(
        "-cohort",
        metavar="CHECK",
        help="Score a cohort check across its finished units (warp_mni: pairwise label Dice "
        "in template space) for each of -datasets; writes cohort_*.json beside the units.",
    )
    p.add_argument("-datasets", nargs="+", default=[], help="Dataset ids (with -plan).")
    p.add_argument(
        "-n",
        type=int,
        help="Units per acquisition with -plan (default 1); per check with "
        "-export_hostset (default 2).",
    )
    p.add_argument("-checks", nargs="+", default=[], help="Checks for -export_hostset.")
    p.add_argument(
        "-tools",
        nargs="+",
        default=[],
        help="Tools to refit with -redo, by unit subfolder: formwarp for ffs_formwarp, ants.",
    )
    p.add_argument("-label", default="host", help="Name of the host set (-export_hostset).")
    p.add_argument(
        "-include_pending",
        action="store_true",
        help="-export_hostset: also take units not yet run here (default: finished only).",
    )
    p.add_argument(
        "-bundle_local",
        action="store_true",
        help="-export_hostset: copy inputs of local-tree datasets (NIREP16, Mindboggle101, "
        "private sets) into the host set instead of skipping them.",
    )
    p.add_argument(
        "-prune_outputs",
        action="store_true",
        help="-export_hostset: the other host prunes regenerable outputs too, as here "
        "(default keeps them all for inspection). Halves what a large warp set syncs back.",
    )
    p.add_argument("-check", help="Only run units of this check (with -run).")
    p.add_argument("-max_units", type=int, default=10**9, help="Stop after this many units.")
    p.add_argument("-device", default="cuda", help="ffs device for -run (cpu, cuda, cuda,N, mps).")
    p.add_argument("-keep", action="store_true", help="Do not prune regenerable outputs.")
    p.add_argument(
        "-workers",
        type=int,
        default=1,
        help="Parallel workers for -run, each pinned to its own -threads cores; 0 = as many as "
        "fit the cores and the CPU quota. More than 1 needs -device cpu.",
    )
    p.add_argument(
        "-threads",
        type=int,
        help="Threads (and physical cores) per worker, given to every tool it runs: AFNI, "
        "ffs, MATLAB, FSL. Default: whatever the environment says.",
    )
    p.add_argument(
        "-until",
        metavar="HH:MM",
        help="Claim no new unit after this local time (next occurrence), e.g. 07:00.",
    )
    args = p.parse_args(argv)
    if args.root is None:
        p.error("-root is required (or set FFS_BENCH_ROOT / FFS_BENCHMARK_DATA_DIR)")
    if args.manifests is None:
        args.manifests = args.root / "manifests"

    from .worklist import Worklist

    # Only -plan starts a worklist; anything else on a missing one is a mistyped root, and
    # opening it would leave an empty database there that reports nothing.
    if not args.plan and not (args.root / "worklist.sqlite").exists():
        p.error(f"no worklist at {args.root} (wrong -root / FFS_BENCH_ROOT?)")
    wl = Worklist(args.root / "worklist.sqlite")
    if args.plan:
        if not args.datasets:
            p.error("-plan needs -datasets")
        plan(wl, args.manifests, args.plan, args.datasets, args.n or 1)
    elif args.export_hostset:
        from .hostset import export_hostset

        if not args.checks:
            p.error("-export_hostset needs -checks")
        export_hostset(wl, args.manifests, args.export_hostset, args.checks, args.n or 2,
                       args.label, args.datasets, args.include_pending, args.bundle_local,
                       args.prune_outputs)  # fmt: skip
    elif args.ingest:
        from .hostset import ingest

        ingest(wl, args.root, args.ingest)
    elif args.cohort:
        if args.cohort != "warp_mni":
            p.error("-cohort knows warp_mni only")
        from .checks.warp_mni import write_cohort

        for ds in args.datasets or ["Mindboggle101"]:
            done = [r["id"] for r in wl.db.execute(
                "SELECT id FROM units WHERE check_name='warp_mni' AND dataset=? AND status='done'", (ds,))]  # fmt: skip
            out = write_cohort(args.root, ds, done)
            print(out.read_text())
    elif args.hosts:
        from .hostset import hostsets

        for h in hostsets(wl):
            print(f"{h['label']:<12} {h['uuid'][:8]}  {h['hostname'] or '?':<16} "
                  f"{h['cpu_model'] or '?':<40} done {h['n_done']} failed {h['n_failed']}  {h['path']}")  # fmt: skip
    elif args.run and args.workers != 1:
        from .workers import launch

        if not args.threads:
            p.error("-workers needs -threads")
        child = ["-run", "-root", str(args.root), "-manifests", str(args.manifests),
                 "-device", args.device]  # fmt: skip
        child += ["-check", args.check] if args.check else []
        child += ["-until", args.until] if args.until else []
        child += ["-keep"] if args.keep else []
        wl.close()
        return launch(child, args.workers or None, args.threads, args.device, args.max_units)
    elif args.run:
        import os

        # bundled host-set manifests name their data as ${FFS_BENCH_ROOT}/data/<ds>
        os.environ.setdefault("FFS_BENCH_ROOT", str(args.root))
        if os.environ.get("FFS_BENCH_CPUS"):  # a pool worker: its own cores, inherited by tools
            os.sched_setaffinity(0, [int(c) for c in os.environ["FFS_BENCH_CPUS"].split(",")])
        if args.threads:
            from .workers import thread_env

            os.environ.update(thread_env(args.threads))
        run(
            wl,
            args.root,
            args.manifests,
            args.check,
            args.max_units,
            args.device,
            args.keep,
            until=deadline_from(args.until),
        )
    elif args.status:
        for r in wl.status():
            print(f"{r['check_name']:<12} {r['dataset']:<24} {r['status']:<10} {r['n']}")
    elif args.retry_failed:
        print(f"{wl.reset('failed', args.check)} units back to pending")
    elif args.reset_running:
        print(f"{wl.reset('running', args.check)} units back to pending")
    elif args.redo:
        if not args.tools:
            p.error("-redo needs -tools")
        n = redo(wl, args.root, args.redo, args.tools)
        print(f"{n} {args.redo} units back to pending; {' '.join(args.tools)} will refit")
    elif args.results:
        from .hostset import hostsets

        rows = [{"host": "local", **r} for r in wl.results(args.results)]
        for h in hostsets(wl):
            hs = Worklist(Path(h["path"]) / "worklist.sqlite")
            rows += [{"host": h["label"], **r} for r in hs.results(args.results)]
            hs.close()
        cols = sorted({k for r in rows for k in r}, key=lambda c: (c.count("."), c))
        print("\t".join(cols))
        for r in rows:
            print(
                "\t".join(
                    ""
                    if r.get(c) is None
                    else (f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c]))
                    for c in cols
                )
            )
    wl.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
