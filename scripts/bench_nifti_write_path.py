#!/usr/bin/env python
"""How long does fastfuncstuff take to write and read a NIfTI, and what does each fix buy?

``bench_nifti_compression.py`` times the codecs. This times the toolbox's own
routes, ``save_nifti`` from an array to a flushed file and ``load_nifti`` from a
file into a float32 array, with each speed fix switched on and off:

* **pipe**: grow the pipe to and from ``pzstd`` from Linux's default 64 KB to
  1 MiB (``afni._grow_pipe``). Linux only; on macOS the call does nothing, so
  the pipe routes there match the base ones, which is a result in itself.
* **nocopy**: write an array that is already in file order without nibabel's
  per-slice copies (``afni._write_nifti_without_copying``).

    python bench_nifti_write_path.py run -out ab.csv -diskdir /fast/disk/scratch \\
        -file bold=bold.nii.gz -file warp=warp.nii.zst
    python bench_nifti_write_path.py summary ab.csv

Every trial is a fresh process, routes are reshuffled each rep, and writes are
timed with their flush to the drive, since a write that only reached the page
cache is not finished. Reads start evicted from the page cache where the OS
allows it (Linux); on macOS they are warm and the run says so. Every write route
is checked once, voxel for voxel, against its source.

Peak RAM is recorded per trial: ``base`` is the high-water mark after loading the
source and before writing, so ``peak`` above ``base`` is what the write added.
Compare a slice at a time: casting two whole arrays to float64 for a comparison
took 48 GB on a 7.9 GB warp and got an earlier version of this run OOM-killed.

Needs fastfuncstuff importable, plus ``pzstd`` (or ``zstd``) on PATH. The
``isal`` package makes ``.nii.gz`` writes use ISA-L; without it they use pigz.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_nifti_compression import (  # noqa: E402
    CAN_EVICT,
    default_threads,
    flush_and_evict,
    peak_rss_gb,
    scratch_dir,
    warn_if_not_ram,
)

#: route -> (suffix, zstd level or "", pipe fix on, nocopy fix on)
WRITE_ROUTES = {
    "zst-1 base": (".nii.zst", "1", False, False),
    "zst-1 pipe": (".nii.zst", "1", True, False),
    "zst-1 nocopy": (".nii.zst", "1", False, True),
    "zst-1 both": (".nii.zst", "1", True, True),
    "zst-3 both": (".nii.zst", "3", True, True),
    "gz copy": (".nii.gz", "", False, False),
    "gz nocopy": (".nii.gz", "", False, True),
    "nii copy": (".nii", "", False, False),
    "nii nocopy": (".nii", "", False, True),
}
#: route -> (zstd level, pipe fix on). Reads never copy on the way in.
READ_ROUTES = {
    "zst-1 64KB": ("1", False),
    "zst-1 1MB": ("1", True),
    "zst-3 64KB": ("3", False),
    "zst-3 1MB": ("3", True),
}
FIELDS = "file,op,route,rep,wall,flush,peak,base,bytes,load1".split(",")


# --- one trial, in its own process -------------------------------------------------


def worker(op: str, src: str, dst: str) -> dict:
    import nibabel as nib
    import numpy as np

    from fastfuncstuff.io import afni

    if os.environ["BENCH_PIPE"] == "0":
        afni._grow_pipe = lambda *_: None
    if os.environ["BENCH_NOCOPY"] == "0":
        afni._write_nifti_without_copying = lambda *_: False

    if op == "write":
        img = nib.load(src)
        data = np.array(img.dataobj)
        base = peak_rss_gb()
        t0 = time.perf_counter()
        afni.save_nifti(data, dst, affine=img.affine)
        return {"wall": time.perf_counter() - t0, "peak": peak_rss_gb(), "base": base}
    if op == "read":
        t0 = time.perf_counter()
        arr = np.asarray(afni.load_nifti(dst).dataobj, dtype=np.float32)
        wall = time.perf_counter() - t0
        assert arr.size
        return {"wall": wall, "peak": peak_rss_gb()}
    if op == "check":
        a = np.asarray(afni.load_nifti(dst).dataobj)
        b = np.asarray(nib.load(src).dataobj)
        same = a.shape == b.shape
        for i in range(a.shape[-1] if same else 0):
            # A slice at a time, so the comparison never holds two float64 copies.
            ai = np.asarray(a[..., i], dtype=np.float64)
            bi = np.asarray(b[..., i], dtype=np.float64)
            if not np.array_equal(ai, bi, equal_nan=True):
                same = False
                break
        return {"same": bool(same), "peak": peak_rss_gb()}
    raise ValueError(op)


def trial(op, src, dst, *, pipe=True, nocopy=True, level="", threads=1) -> dict:
    env = dict(
        os.environ,
        FFS_NUM_THREADS=str(threads),
        OMP_NUM_THREADS=str(threads),
        BENCH_PIPE=str(int(pipe)),
        BENCH_NOCOPY=str(int(nocopy)),
    )
    if level:
        env["FFS_ZSTD_LEVEL"] = level
    res = subprocess.run(
        [sys.executable, __file__, "_worker", op, str(src), str(dst)],
        capture_output=True,
        text=True,
        env=env,
    )
    lines = [ln for ln in res.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        raise RuntimeError(f"{op} {dst} failed:\n{res.stderr[-2000:]}")
    return json.loads(lines[-1])


# --- the run -----------------------------------------------------------------------


def decompress_to(src: Path, dst: Path) -> None:
    """A byte-exact uncompressed copy, standing in for an array already in memory."""
    name = src.name.lower()
    with open(dst, "wb") as out:
        if name.endswith(".nii.zst"):
            subprocess.run(["zstd", "-dcq", str(src)], stdout=out, check=True)
        elif name.endswith(".nii.gz"):
            with gzip.open(src, "rb") as fh:
                shutil.copyfileobj(fh, out, 1 << 24)
        elif name.endswith(".nii"):
            with open(src, "rb") as fh:
                shutil.copyfileobj(fh, out, 1 << 24)
        else:
            raise ValueError(f"{src}: expected .nii, .nii.gz or .nii.zst")


def run(args) -> None:
    work, disk = Path(args.workdir), Path(args.diskdir)
    work.mkdir(parents=True, exist_ok=True)
    disk.mkdir(parents=True, exist_ok=True)
    warn_if_not_ram(work)
    if not CAN_EVICT:
        print("note: this OS cannot evict a file without root, so reads are warm", flush=True)
    import fcntl

    if not hasattr(fcntl, "F_SETPIPE_SZ"):
        print("note: no F_SETPIPE_SZ here, so the pipe fix does nothing on this OS", flush=True)
    inputs = dict(spec.split("=", 1) for spec in args.file)
    # Check every input up front, not an hour in when its turn comes.
    for label, path in inputs.items():
        if not Path(path).is_file() or not path.lower().endswith((".nii", ".nii.gz", ".nii.zst")):
            sys.exit(f"-file {label}={path}: not a .nii, .nii.gz or .nii.zst file")
    rng = random.Random(args.seed)
    t_start, checked = time.time(), 0
    with open(args.out, "w", newline="", buffering=1) as fh:
        log = csv.writer(fh)
        log.writerow(FIELDS)
        for label, path in inputs.items():
            src = work / f"{label}.nii"
            decompress_to(Path(path), src)
            for rep in range(args.reps):
                routes = list(WRITE_ROUTES)
                rng.shuffle(routes)
                for route in routes:
                    suffix, level, pipe, nocopy = WRITE_ROUTES[route]
                    dst = disk / f"out{suffix}"
                    dst.unlink(missing_ok=True)
                    os.sync()
                    res = trial(
                        "write",
                        src,
                        dst,
                        pipe=pipe,
                        nocopy=nocopy,
                        level=level,
                        threads=args.threads,
                    )
                    flush = flush_and_evict(dst)
                    log.writerow(
                        [
                            label,
                            "write",
                            route,
                            rep,
                            res["wall"],
                            flush,
                            res["peak"],
                            res["base"],
                            dst.stat().st_size,
                            os.getloadavg()[0],
                        ]
                    )
                    if rep == 0:
                        if not trial("check", src, dst, threads=args.threads)["same"]:
                            raise RuntimeError(f"voxels differ: {label} via {route}")
                        checked += 1
                    dst.unlink()
            files = {}
            for level in ("1", "3"):
                files[level] = disk / f"read-{level}.nii.zst"
                trial("write", src, files[level], level=level, threads=args.threads)
            for rep in range(args.reps):
                routes = list(READ_ROUTES)
                rng.shuffle(routes)
                for route in routes:
                    level, pipe = READ_ROUTES[route]
                    os.sync()
                    flush_and_evict(files[level])
                    res = trial("read", src, files[level], pipe=pipe, threads=args.threads)
                    log.writerow(
                        [
                            label,
                            "read",
                            route,
                            rep,
                            res["wall"],
                            "",
                            res["peak"],
                            "",
                            files[level].stat().st_size,
                            os.getloadavg()[0],
                        ]
                    )
            for f in files.values():
                f.unlink()
            src.unlink()
            print(f"[{time.time() - t_start:6.0f}s] {label}", flush=True)
    print(f"voxels identical on {checked} file/route pairs")


# --- reading the results -----------------------------------------------------------


def summary(path: Path) -> None:
    """Medians with min-max per file and route; writes are wall plus flush."""
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    order: list[str] = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if row["file"] not in order:
                order.append(row["file"])
            groups[(row["op"], row["file"], row["route"])].append(row)
    loads = [float(r["load1"]) for rows in groups.values() for r in rows]
    if not loads:
        sys.exit(f"{path}: no trials recorded")
    print(f"load average: median {statistics.median(loads):.1f}, max {max(loads):.1f}")
    for op, title in (("write", "seconds to write, wall + flush"), ("read", "seconds to read")):
        print(f"\n== {title}")
        for label in order:
            rows = {k[2]: v for k, v in groups.items() if k[0] == op and k[1] == label}
            if not rows:
                continue
            print(f"\n{label}")
            stats = []
            for route, rs in rows.items():
                waits = [float(r["wall"]) + float(r["flush"] or 0) for r in rs]
                extra = max(
                    (float(r["peak"]) - float(r["base"]) for r in rs if r["base"]), default=None
                )
                stats.append((statistics.median(waits), min(waits), max(waits), route, rs, extra))
            for med, lo, hi, route, rs, extra in sorted(stats):
                size = statistics.median(int(r["bytes"]) for r in rs) / 1e6
                added = f"  write added {extra:5.2f} GB" if extra is not None else ""
                print(
                    f"  {route:14s} {med:7.3f} s  [{lo:.3f}-{hi:.3f}]  n={len(rs)}"
                    f"  {size:8.0f} MB  peak {max(float(r['peak']) for r in rs):5.1f} GB{added}"
                )


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        print(json.dumps(worker(*sys.argv[2:5])), flush=True)
        # Skip interpreter teardown: freeing a multi-GB array is not part of a read.
        os._exit(0)
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("-out", required=True, help="CSV to write, one row per trial")
    r.add_argument("-diskdir", required=True, help="scratch space on the disk under test")
    r.add_argument(
        "-workdir",
        default=scratch_dir("bench_nifti_write_path"),
        help="RAM-backed space for the uncompressed sources",
    )
    r.add_argument("-file", action="append", required=True, metavar="LABEL=PATH")
    r.add_argument("-reps", type=int, default=3)
    r.add_argument("-threads", type=int, default=default_threads())
    r.add_argument("-seed", type=int, default=0)
    s = sub.add_parser("summary")
    s.add_argument("csv")
    args = p.parse_args()
    # Work in fresh subfolders, so the cleanup below can never delete a folder
    # that was passed in and already held something.
    import tempfile

    for name in ("workdir", "diskdir"):
        if getattr(args, name, None):
            Path(getattr(args, name)).mkdir(parents=True, exist_ok=True)
            setattr(args, name, tempfile.mkdtemp(prefix="bench-", dir=getattr(args, name)))
    if args.cmd == "run":
        try:
            run(args)
        finally:
            shutil.rmtree(args.workdir, ignore_errors=True)
            shutil.rmtree(args.diskdir, ignore_errors=True)
    else:
        summary(Path(args.csv))


if __name__ == "__main__":
    main()
