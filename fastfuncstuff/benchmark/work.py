"""Benchmark runner: plan units into the worklist, then work through them one at a time.

    python -m fastfuncstuff.benchmark.work -root R -manifests M -plan moco -datasets ds002603 -n 2
    python -m fastfuncstuff.benchmark.work -root R -manifests M -run [-check moco] [-max_units 5]
    python -m fastfuncstuff.benchmark.work -root R -status
    python -m fastfuncstuff.benchmark.work -root R -results moco

Each unit: fetch inputs (S3 into R/cache, local trees in place) -> reference -> ffs ->
score -> record -> prune regenerable outputs and downloaded inputs. ``-keep`` skips pruning.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

from fastfuncstuff.cli_help import FfsArgumentParser


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
        units = check.plan(m, index, n)
        added = wl.add_units(check.name, ds, units)
        print(f"{ds}: {len(units)} units planned for {check.name}, {added} new")


def run(
    wl, root: Path, manifests: Path, check: str | None, max_units: int, device: str, keep: bool
) -> int:
    from .checks import CHECKS
    from .fetch import fetch, sha256

    n_done = 0
    while n_done < max_units:
        u = wl.claim_next(check=check)
        if u is None:
            break
        key = json.loads(u["unit_key"])
        label = f"[{u['id']}] {u['check_name']} {u['dataset']} {' '.join(f'{k}-{v}' for k, v in key.items())}"
        print(label, flush=True)
        work = root / "units" / u["check_name"] / u["dataset"] / f"unit{u['id']:06d}"
        try:
            m = _manifest(manifests, u["dataset"])
            paths, downloaded = fetch(m, json.loads(u["inputs"]), root / "cache")
            wl.record_provenance(
                u["id"],
                "inputs",
                {rel: sha256(p) for rel, p in paths.items()} | {"source": m["source"]["uri"]},
            )
            res = CHECKS[u["check_name"]]().run(key, paths, work, device)
            for judge, vals in res.metrics.items():
                wl.record_metrics(u["id"], judge, vals)
            for side, dev, sec, thr in res.timings:
                wl.record_timing(u["id"], side, dev, sec, thr)
            for path, retention in res.artifacts:
                wl.record_artifact(u["id"], path, retention)
            for path in downloaded:
                wl.record_artifact(u["id"], path, "regenerable")
            for side, vals in res.provenance.items():
                wl.record_provenance(u["id"], side, vals)
            from ..benchmark.arch import get_hardware_info

            wl.record_provenance(u["id"], "host", get_hardware_info(device))
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
            freed = wl.prune(u["id"])
            if freed:
                print(f"    pruned {freed / 1e6:.0f} MB", flush=True)
        n_done += 1
    return n_done


def main(argv: list[str] | None = None) -> int:
    p = FfsArgumentParser(prog="python -m fastfuncstuff.benchmark.work", description=__doc__)
    p.add_argument(
        "-root",
        type=Path,
        default=os.environ.get("FFS_BENCH_ROOT"),
        help="Benchmark root (worklist.sqlite, cache/, units/).",
    )
    p.add_argument("-manifests", type=Path, help="Directory of census manifests.")
    act = p.add_mutually_exclusive_group(required=True)
    act.add_argument("-plan", metavar="CHECK", help="Add units of CHECK for -datasets.")
    act.add_argument("-run", action="store_true", help="Work through pending units.")
    act.add_argument("-status", action="store_true", help="Unit counts by check, dataset, status.")
    act.add_argument("-results", metavar="CHECK", help="Print finished units of CHECK as TSV.")
    act.add_argument("-retry_failed", action="store_true", help="Return failed units to pending.")
    p.add_argument("-datasets", nargs="+", default=[], help="Dataset ids (with -plan).")
    p.add_argument("-n", type=int, default=1, help="Units per acquisition (breadth checks).")
    p.add_argument("-check", help="Only run units of this check (with -run).")
    p.add_argument("-max_units", type=int, default=10**9, help="Stop after this many units.")
    p.add_argument("-device", default="cuda", help="ffs device for -run (cpu, cuda, cuda,N, mps).")
    p.add_argument("-keep", action="store_true", help="Do not prune regenerable outputs.")
    args = p.parse_args(argv)
    if args.root is None:
        p.error("-root (or FFS_BENCH_ROOT) is required")

    from .worklist import Worklist

    wl = Worklist(args.root / "worklist.sqlite")
    if args.plan:
        if not args.manifests or not args.datasets:
            p.error("-plan needs -manifests and -datasets")
        plan(wl, args.manifests, args.plan, args.datasets, args.n)
    elif args.run:
        if not args.manifests:
            p.error("-run needs -manifests")
        run(wl, args.root, args.manifests, args.check, args.max_units, args.device, args.keep)
    elif args.status:
        for r in wl.status():
            print(f"{r['check_name']:<12} {r['dataset']:<24} {r['status']:<10} {r['n']}")
    elif args.retry_failed:
        print(f"{wl.reset('failed', args.check)} units back to pending")
    elif args.results:
        rows = wl.results(args.results)
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
