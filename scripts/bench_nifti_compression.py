#!/usr/bin/env python
"""How fast are the ways of compressing and reading a NIfTI? A fair comparison.

Compares ISA-L, pigz, zstd and pzstd as encoders, and those plus rapidgzip as
decoders, on your own files. Single-core zlib is left out on purpose: it is the
slowest thing here by far and nobody needs a benchmark to know it. Standalone: it needs numpy,
nibabel, matplotlib, the ``isal`` and ``rapidgzip`` Python packages and the
``pigz``, ``zstd`` and ``pzstd`` commands. If ``fastfuncstuff`` is importable it
also times that toolbox's own save and load routes.

    python bench_nifti_compression.py run -out results -workdir /dev/shm/bench \
        -file mask=mask.nii.gz -file t1=t1.nii.gz -file bold=bold.nii.gz \
        -file warp=warp.nii.zst -sweep bold
    python bench_nifti_compression.py plot results

What makes it fair:

* Everything runs from and to tmpfs, so it measures the codec and not the disk.
  Decoders write to /dev/null, which costs nothing, so a fast decoder is not
  capped by the sink. The ``e2e`` tier then decodes into a real array.
* The order of trials is reshuffled on every repetition, so drift (temperature,
  other load, cache state) is spread across the methods rather than landing on
  whichever always runs last. The median is reported and the load is logged.
* Thread counts are explicit and compared like for like: 1 against 1, N against N.
* Every encoder/decoder pair is checked once against the source: exact length and
  CRC32, outside the timed section. A wrong answer fails the run.
* gzip decoders are crossed with gzip encoders, because who wrote a file changes
  how fast it reads, and any tool can read any other tool's file.
* ISA-L is called in-process since that is the only way to use it from Python, so
  it pays no process start. The command-line tools do, about 1 ms, which only
  matters on files far smaller than a volume series.

Runs on Linux and macOS. What differs on a Mac:

* There is no /dev/shm. Without a RAM disk the scratch space falls back to the
  temp directory, which is a real disk, and the run says so. Make a RAM disk
  first (8 GiB here; the count is 512-byte sectors) and pass it as -workdir:
  ``diskutil erasevolume HFS+ ramdisk $(hdiutil attach -nomount ram://16777216)``
* macOS cannot drop one file from the page cache without root, so ``disk``
  reads there are warm, and its figure says so. Writes are flushed with
  F_FULLFSYNC, since a plain fsync on macOS does not reach the drive.
* Peak RAM comes from ru_maxrss rather than Linux's VmHWM. If the OS carries a
  parent's high-water mark across exec it reads high, so treat it as a bound.
* Apple Silicon has no SMT, so the default (half the logical CPUs) may be fewer
  threads than you want. Pass -threads.
* ISA-L's hand-tuned code is mostly written for x86, so its numbers are the
  most likely to move on ARM.

Traps that produce a convincing wrong answer, whatever the machine:

* A float32 ``.nii`` opened with nibabel is memory-mapped, so a "read" that
  never copies the voxels has read nothing.
* Decoding to /dev/null is not a load. A reader that wins there can lose once
  the bytes go through a pipe into an array, as ``pzstd -1`` did against
  ``-3`` through Linux's 64 KB default pipe. ``bench_nifti_write_path.py``
  times the toolbox's real routes.
* Check that a cut of a file compresses before trusting a size sweep. One
  real warp's middle was all zeros.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import resource
import shutil
import statistics
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

CHUNK = 1 << 20
LARGE_BYTES = 5 * 10**8


@dataclass(frozen=True)
class Encoder:
    name: str  # stable id, e.g. "isal-2.p6"; also the file stem
    tool: str  # isal | pigz | zstd | pzstd
    level: int
    threads: int

    @property
    def ext(self) -> str:
        return ".gz" if self.tool in ("isal", "pigz") else ".zst"


@dataclass(frozen=True)
class Decoder:
    name: str
    tool: str  # isal | pigz | rapidgzip | zstd | pzstd
    threads: int

    @property
    def ext(self) -> str:
        return ".gz" if self.tool in ("isal", "pigz", "rapidgzip") else ".zst"


def enc(tool: str, level: int, threads: int) -> Encoder:
    """A negative level is zstd's --fast=N."""
    name = f"f{-level}" if level < 0 else str(level)
    return Encoder(f"{tool}-{name}.p{threads}", tool, level, threads)


def core_encoders(n: int) -> list[Encoder]:
    """The settings people actually run, on every file.

    ISA-L 2 is its default and the tedana PR's choice, pigz -1 and pzstd -3 are what
    fastfuncstuff writes, zstd -1 is its fast setting.
    """
    return [
        enc("isal", 2, 1),
        enc("isal", 2, n),
        enc("pigz", 1, n),
        enc("zstd", 1, 1),
        enc("zstd", 3, 1),
        enc("zstd", 3, n),
        enc("pzstd", 1, n),
        enc("pzstd", 3, n),
    ]


def sweep_encoders(n: int) -> list[Encoder]:
    """Level sweeps, for the size-against-speed curve."""
    out = [enc("isal", lv, t) for t in (1, n) for lv in (0, 1, 2, 3)]
    out += [enc("pigz", lv, n) for lv in (1, 3, 6, 9)]
    out += [enc("zstd", lv, 1) for lv in (1, 3)]
    out += [enc("pzstd", lv, n) for lv in (1, 3, 6, 9)]
    return out


def decoders(n: int) -> list[Decoder]:
    return [
        Decoder("isal.p1", "isal", 1),
        Decoder(f"pigz.p{n}", "pigz", n),  # pigz never decodes in parallel; N buys helper threads
        Decoder(f"rapidgzip.p{n}", "rapidgzip", n),
        Decoder("zstd.p1", "zstd", 1),
        Decoder(f"pzstd.p{n}", "pzstd", n),
    ]


def decode_encoders(n: int) -> list[Encoder]:
    """Whose files get decoded. Same format and level give the same read speed, so
    the other core encoders would only repeat these."""
    return [
        enc("isal", 2, 1),
        enc("isal", 2, n),
        enc("pigz", 1, n),
        enc("zstd", 3, 1),
        enc("pzstd", 1, n),
        enc("pzstd", 3, n),
    ]


def encode_cmd(e: Encoder, src: Path) -> list[str]:
    if e.tool == "pigz":
        return ["pigz", "-c", f"-{e.level}", "-p", str(e.threads), str(src)]
    if e.tool == "zstd":
        level = f"--fast={-e.level}" if e.level < 0 else f"-{e.level}"
        return ["zstd", "-q", "-c", level, f"-T{e.threads}", str(src)]
    return ["pzstd", "-q", "-c", f"-{e.level}", "-p", str(e.threads), str(src)]


def decode_cmd(d: Decoder, src: Path) -> list[str]:
    if d.tool == "pigz":
        return ["pigz", "-d", "-c", "-p", str(d.threads), str(src)]
    if d.tool == "rapidgzip":
        return ["rapidgzip", "-d", "-c", "-q", "-P", str(d.threads), str(src)]
    if d.tool == "zstd":
        return ["zstd", "-q", "-d", "-c", str(src)]
    return ["pzstd", "-q", "-d", "-c", "-p", str(d.threads), str(src)]


def _timed_cmd(cmd: list[str], stdout_path: Path | str) -> tuple[float, float]:
    """Wall and CPU seconds for one child, including all its threads."""
    with open(stdout_path, "wb") as out:
        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.PIPE)
        _, status, ru = os.wait4(proc.pid, 0)
        wall = time.perf_counter() - t0
    proc.returncode = os.waitstatus_to_exitcode(status)
    if proc.returncode != 0:
        assert proc.stderr is not None
        raise RuntimeError(f"{cmd}: {proc.stderr.read().decode(errors='replace')[:300]}")
    return wall, ru.ru_utime + ru.ru_stime


def _self_cpu() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


def run_encoder(e: Encoder, src: Path, dst: Path) -> tuple[float, float]:
    if e.tool != "isal":
        return _timed_cmd(encode_cmd(e, src), dst)
    t0, c0 = time.perf_counter(), _self_cpu()
    with open(src, "rb") as fin:
        if e.threads > 1:
            from isal import igzip_threaded

            with igzip_threaded.open(dst, "wb", compresslevel=e.level, threads=e.threads) as gz:
                shutil.copyfileobj(fin, gz, CHUNK)
        else:
            from isal import igzip

            # Empty name and mtime=0, as nibabel and the tedana PR write them.
            with open(dst, "wb") as fout, igzip.IGzipFile("", "wb", e.level, fout, mtime=0) as gz:
                shutil.copyfileobj(fin, gz, CHUNK)
    return time.perf_counter() - t0, _self_cpu() - c0


def _open_in_process(src: Path):
    from isal import igzip

    return igzip.open(src, "rb")


def run_decoder(d: Decoder, src: Path) -> tuple[float, float]:
    if d.tool != "isal":
        return _timed_cmd(decode_cmd(d, src), os.devnull)
    buf = bytearray(CHUNK)
    t0, c0 = time.perf_counter(), _self_cpu()
    with _open_in_process(src) as fh:
        while fh.readinto(buf):
            pass
    return time.perf_counter() - t0, _self_cpu() - c0


def stream_crc(fobj) -> tuple[int, int]:
    crc, total, buf = 0, 0, bytearray(CHUNK)
    view = memoryview(buf)
    while n := fobj.readinto(buf):
        crc = zlib.crc32(view[:n], crc)
        total += n
    return crc, total


def source_crc(path: Path) -> tuple[int, int]:
    with open(path, "rb", buffering=0) as fh:
        return stream_crc(fh)


def verify_decode(d: Decoder, src: Path, expected: tuple[int, int]) -> None:
    if d.tool == "isal":
        with _open_in_process(src) as fh:
            got = stream_crc(fh)
    else:
        proc = subprocess.Popen(decode_cmd(d, src), stdout=subprocess.PIPE, bufsize=0)
        assert proc.stdout is not None
        got = stream_crc(proc.stdout)
        if proc.wait() != 0:
            raise RuntimeError(f"{d.name} failed on {src.name}")
    if got != expected:
        raise RuntimeError(f"{d.name} on {src.name}: got {got}, source is {expected}")


# --- input preparation ------------------------------------------------------------


def prepare_inputs(args, work: Path) -> dict[str, dict]:
    """Byte-exact uncompressed copies in tmpfs. The result carries no source paths."""
    import nibabel as nib

    src_dir = work / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    inputs: dict[str, dict] = {}
    for spec in args.file:
        label, _, path = spec.partition("=")
        dst = src_dir / f"{label}.nii"
        if path.endswith(".gz"):
            cmd = ["pigz", "-dc", path]
        elif path.endswith(".zst"):
            cmd = ["zstd", "-qdc", path]
        else:
            cmd = ["cat", path]
        with open(dst, "wb") as out:
            subprocess.run(cmd, stdout=out, check=True)
        hdr = nib.load(dst).header
        shape = "x".join(str(v) for v in hdr.get_data_shape())
        inputs[label] = {
            "path": dst,
            "bytes": dst.stat().st_size,
            "dtype": str(hdr.get_data_dtype()),
            "kind": shape,
        }
    return dict(sorted(inputs.items(), key=lambda kv: kv[1]["bytes"]))


# --- end-to-end routes, each timed in a fresh process -----------------------------

WRITE_ROUTES = ["nii", "nibabel-isal2", "ffs-gz", "ffs-zst"]
READ_ROUTES = ["nii", "nibabel-isal", "nibabel-rapidgzip", "ffs-gz", "ffs-zst"]


def _peak_rss_gb() -> float:
    return peak_rss_gb()


def peak_rss_gb() -> float:
    """This process's peak resident memory, in GiB."""
    # VmHWM, not ru_maxrss, where there is one: on Linux ru_maxrss survives exec,
    # so every worker would report the harness's own peak.
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024**2
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024**3 if sys.platform == "darwin" else peak / 1024**2


def default_threads() -> int:
    """Half the CPUs this process may use: the physical cores on SMT x86."""
    if hasattr(os, "sched_getaffinity"):
        n = len(os.sched_getaffinity(0))
    else:
        n = os.cpu_count() or 2
    return max(1, n // 2)


def scratch_dir(name: str) -> str:
    """tmpfs where there is one, else the temp directory, which is a real disk."""
    if Path("/dev/shm").is_dir():
        return f"/dev/shm/{name}"
    import tempfile

    return str(Path(tempfile.gettempdir()) / name)


def warn_if_not_ram(path: Path) -> None:
    """Say so when the scratch space is a disk, since then codec times include it."""
    probe = path if path.exists() else path.parent
    try:
        out = subprocess.run(["df", "-P", str(probe)], capture_output=True, text=True).stdout
        device = out.splitlines()[-1].split()[0]
    except (OSError, IndexError):
        return
    ram = device in ("tmpfs", "shm", "none") or device.startswith("/dev/ram")
    if sys.platform == "darwin":
        # An hdiutil RAM disk is a disk image whose source is ram://.
        info = subprocess.run(["hdiutil", "info"], capture_output=True, text=True).stdout
        import re

        whole = re.match(r"/dev/disk\d+", device)
        ram = whole is not None and any(
            "ram://" in block and re.search(re.escape(whole.group()) + r"\b", block)
            for block in info.split("=" * 48)
        )
    if not ram:
        print(
            f"note: {probe} is on {device}, not a RAM disk, so these times include that disk."
            " See the docstring for making one on macOS.",
            flush=True,
        )


def worker_write(route: str, src: Path, dst: Path) -> dict:
    import nibabel as nib
    import numpy as np

    src_img = nib.load(src)
    data = np.asarray(src_img.dataobj)
    img = nib.Nifti1Image(data, src_img.affine, src_img.header)
    if route.startswith("ffs"):
        from fastfuncstuff.io.afni import save_nifti

        write = lambda: save_nifti(data, str(dst), affine=src_img.affine, header=src_img.header)  # noqa: E731
    elif route == "nibabel-isal2":
        from isal.igzip import IGzipFile

        def write():  # tedana PR 1524's write_nifti, verbatim in effect
            with open(dst, "wb") as fobj, IGzipFile("", "wb", fileobj=fobj, mtime=0) as gz:
                img.to_file_map({"image": nib.FileHolder(filename=str(dst), fileobj=gz)})
    else:
        write = lambda: nib.save(img, str(dst))  # noqa: E731
    t0 = time.perf_counter()
    write()
    return {"wall": time.perf_counter() - t0, "peak_gb": _peak_rss_gb()}


def _route_nibabel_gz_through_isal() -> None:
    from isal import igzip
    from nibabel import _compression, openers, volumeutils

    _, arg_names = openers.ImageOpener.compress_ext_map[".gz"]

    def gz_open(filename, mode="rb", *a, **k):
        return igzip.open(filename, mode)

    openers.ImageOpener.compress_ext_map[".gz"] = (gz_open, arg_names)
    for module in (_compression, volumeutils):
        module.COMPRESSED_FILE_LIKES += (igzip.IGzipFile,)


def _route_nibabel_gz_through_rapidgzip(threads: int) -> None:
    import rapidgzip
    from nibabel import _compression, openers, volumeutils

    _, arg_names = openers.ImageOpener.compress_ext_map[".gz"]

    def gz_open(filename, mode="rb", *a, **k):
        return rapidgzip.open(filename, parallelization=threads)

    openers.ImageOpener.compress_ext_map[".gz"] = (gz_open, arg_names)
    for module in (_compression, volumeutils):
        module.COMPRESSED_FILE_LIKES += (rapidgzip.RapidgzipFile,)


def worker_read(route: str, src: Path, threads: int) -> dict:
    import nibabel as nib
    import numpy as np

    if route.startswith("ffs"):
        from fastfuncstuff.io.afni import load_nifti

        load = load_nifti
    else:
        load = nib.load
    if route == "nibabel-isal":
        _route_nibabel_gz_through_isal()
    elif route == "nibabel-rapidgzip":
        _route_nibabel_gz_through_rapidgzip(threads)
    t0 = time.perf_counter()
    # copy=True for a plain .nii: a float32 one is memory-mapped, so asarray reads nothing.
    arr = np.array(load(str(src)).dataobj, dtype=np.float32, copy=route == "nii" or None)
    wall = time.perf_counter() - t0
    if arr.size == 0:
        raise RuntimeError("empty read")
    out = {
        "wall": wall,
        "peak_gb": _peak_rss_gb(),
        "sum": float(np.nansum(arr[..., 0], dtype=np.float64)),
    }
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()
    # A rapidgzip file still open at shutdown can abort the interpreter; the
    # result is already out, so leave without running teardown.
    os._exit(0)


def worker_main(argv: list[str]) -> None:
    op, route, src, dst, threads = argv
    if op == "write":
        print(json.dumps(worker_write(route, Path(src), Path(dst))))
    else:
        worker_read(route, Path(src), int(threads))


def run_worker(op: str, route: str, src: Path, dst: Path, threads: int) -> dict:
    env = dict(os.environ, FFS_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads))
    cmd = [sys.executable, __file__, "_worker", op, route, str(src), str(dst), str(threads)]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    lines = [ln for ln in res.stdout.splitlines() if ln.startswith("{")]
    if res.returncode != 0 or not lines:
        raise RuntimeError(f"{op} {route} on {src.name}: {res.stderr[-600:]}")
    return json.loads(lines[-1])


# --- the run ----------------------------------------------------------------------


def versions() -> dict:
    out: dict = {"python": platform.python_version()}
    for tool, flag in [
        ("pigz", "--version"),
        ("zstd", "--version"),
        ("pzstd", "--version"),
        ("rapidgzip", "--version"),
    ]:
        res = subprocess.run([tool, flag], capture_output=True, text=True)
        out[tool] = (res.stdout + res.stderr).strip().splitlines()[0]
    import isal
    import nibabel

    out["isal"] = isal.__version__
    out["nibabel"] = nibabel.__version__
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                out["cpu"] = line.split(":", 1)[1].strip()
                break
    elif sys.platform == "darwin":
        res = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        )
        out["cpu"] = res.stdout.strip()
    out["platform"] = platform.platform()
    out["logical_cpus"] = os.cpu_count()
    return out


class Log:
    FIELDS = "tier,file,bytes,dtype,kind,method,encoder,decoder,tool,level,threads,rep,wall,cpu,out_bytes,peak_gb,load1".split(
        ","
    )

    def __init__(self, path: Path):
        self.fh = open(path, "w", newline="")
        self.w = csv.DictWriter(self.fh, self.FIELDS)
        self.w.writeheader()

    def row(self, **kw) -> None:
        kw["load1"] = round(os.getloadavg()[0], 2)
        self.w.writerow(kw)
        self.fh.flush()


def reps_for(info: dict, args) -> int:
    return args.reps if info["bytes"] < LARGE_BYTES else args.reps_large


def run(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(args.workdir)
    warn_if_not_ram(work)
    n = args.threads
    rng = random.Random(args.seed)
    print("preparing inputs", flush=True)
    inputs = prepare_inputs(args, work)
    log = Log(out / "results.csv")
    for label, info in inputs.items():
        print(f"  {label}: {info['bytes'] / 1e6:.0f} MB {info['dtype']}", flush=True)

    core = core_encoders(n)
    decs = decoders(n)
    meta = versions() | {
        "threads": n,
        "reps": args.reps,
        "reps_large": args.reps_large,
        "seed": args.seed,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    t_start = time.time()
    for label, info in inputs.items():
        fdir = work / "c" / label
        fdir.mkdir(parents=True, exist_ok=True)
        common = {k: info[k] for k in ("bytes", "dtype", "kind")} | {"file": label}
        expected = source_crc(info["path"])

        # Encoders. Each rep starts from an unlinked target, so no trial inherits
        # another's tmpfs pages.
        encs = list(core)
        if label in args.sweep:
            encs += [e for e in sweep_encoders(n) if e not in core]
        for rep in range(reps_for(info, args)):
            rng.shuffle(encs)
            for e in encs:
                dst = fdir / f"{e.name}{e.ext}"
                dst.unlink(missing_ok=True)
                wall, cpu = run_encoder(e, info["path"], dst)
                log.row(
                    tier="encode",
                    method=e.name,
                    encoder=e.name,
                    decoder="",
                    tool=e.tool,
                    level=e.level,
                    threads=e.threads,
                    rep=rep,
                    wall=wall,
                    cpu=cpu,
                    out_bytes=dst.stat().st_size,
                    **common,
                )
            print(f"[{time.time() - t_start:6.0f}s] {label}: encode rep {rep + 1}", flush=True)

        # Sweep-only outputs are not needed for decoding.
        for e in encs:
            if e not in core:
                (fdir / f"{e.name}{e.ext}").unlink(missing_ok=True)

        # Decoders: every core encoder's file through every decoder of its format.
        pairs = [(e, d) for e in decode_encoders(n) for d in decs if d.ext == e.ext]
        # Large files check each encoder once rather than every pair; the full cross
        # has by then passed on the smaller files.
        checks = [
            (e, d)
            for e, d in pairs
            if info["bytes"] < LARGE_BYTES or d.name in ("isal.p1", "zstd.p1")
        ]
        for e, d in checks:
            verify_decode(d, fdir / f"{e.name}{e.ext}", expected)
        print(
            f"[{time.time() - t_start:6.0f}s] {label}: {len(checks)} pairs decode to the source bytes",
            flush=True,
        )
        for rep in range(reps_for(info, args)):
            rng.shuffle(pairs)
            for e, d in pairs:
                src = fdir / f"{e.name}{e.ext}"
                wall, cpu = run_decoder(d, src)
                log.row(
                    tier="decode",
                    method=f"{e.name}>{d.name}",
                    encoder=e.name,
                    decoder=d.name,
                    tool=d.tool,
                    level=e.level,
                    threads=d.threads,
                    rep=rep,
                    wall=wall,
                    cpu=cpu,
                    out_bytes=src.stat().st_size,
                    **common,
                )
            print(f"[{time.time() - t_start:6.0f}s] {label}: decode rep {rep + 1}", flush=True)

        if not args.no_e2e:
            run_e2e(args, label, info, fdir, common, log, rng, t_start)
        shutil.rmtree(fdir)
    print(f"done in {time.time() - t_start:.0f}s", flush=True)


def run_e2e(args, label, info, fdir, common, log, rng, t_start) -> None:
    n = args.threads
    have_ffs = shutil.which("pigz") is not None and _can_import("fastfuncstuff.io.afni")
    writes = [r for r in WRITE_ROUTES if have_ffs or not r.startswith("ffs")]
    reads = [r for r in READ_ROUTES if have_ffs or not r.startswith("ffs")]
    # Every gzip read route gets the same file, so the route is the only difference.
    read_src = {"zst": fdir / f"pzstd-3.p{n}.zst", "gz": fdir / f"pigz-1.p{n}.gz"}
    e2e_dir = fdir / "e2e"
    e2e_dir.mkdir(exist_ok=True)
    reference = None
    for rep in range(reps_for(info, args)):
        rng.shuffle(writes)
        for route in writes:
            ext = ".nii" if route == "nii" else (".nii.zst" if route.endswith("zst") else ".nii.gz")
            dst = e2e_dir / f"w{ext}"
            dst.unlink(missing_ok=True)
            res = run_worker("write", route, info["path"], dst, n)
            log.row(
                tier="e2e-write",
                method=route,
                tool=route,
                threads=n,
                rep=rep,
                wall=res["wall"],
                peak_gb=res["peak_gb"],
                out_bytes=dst.stat().st_size,
                **common,
            )
            dst.unlink()
        rng.shuffle(reads)
        for route in reads:
            if route == "nii":
                src = info["path"]
            else:
                # The readers go by file name; a .nii.gz they get regardless of who wrote it.
                kind = "zst" if route.endswith("zst") else "gz"
                src = e2e_dir / f"r.nii.{kind}"
                if not src.exists():
                    os.link(read_src[kind], src)
            res = run_worker("read", route, src, Path(os.devnull), n)
            # A method that returned the wrong voxels has no business on the chart.
            reference = reference if reference is not None else res["sum"]
            if res["sum"] != reference:
                raise RuntimeError(f"{route} read different data from {label}")
            log.row(
                tier="e2e-read",
                method=route,
                tool=route,
                threads=n,
                rep=rep,
                wall=res["wall"],
                peak_gb=res["peak_gb"],
                out_bytes=src.stat().st_size,
                **common,
            )
        print(f"[{time.time() - t_start:6.0f}s] {label}: e2e rep {rep + 1}", flush=True)


def _can_import(name: str) -> bool:
    res = subprocess.run([sys.executable, "-c", f"import {name}"], capture_output=True)
    return res.returncode == 0


# --- at what file size does pzstd start to pay? -----------------------------------


def crossover_writers(n: int) -> list[Encoder]:
    return [
        *(enc("zstd", lv, t) for t in (1, n) for lv in (-3, -1, 1, 3)),
        enc("pzstd", 1, n),
        enc("pzstd", 3, n),
    ]


def count_frames(path: Path) -> int:
    """Data frames in a zstd file. pzstd's parallel read needs more than one."""
    res = subprocess.run(["zstd", "-l", str(path)], capture_output=True, text=True, check=True)
    # The Frames column counts pzstd's skippable index frames too.
    frames, skips = res.stdout.splitlines()[1].split()[:2]
    return int(frames) - int(skips)


def run_crossover(args) -> None:
    """Middle-of-the-file cuts of each input, doubling in size.

    The middle, because the start of a NIfTI is mostly air and would flatter every
    codec. A cut is not a valid image, which does not matter: a codec only sees bytes.
    """
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(args.workdir)
    warn_if_not_ram(work)
    n = args.threads
    rng = random.Random(args.seed)
    inputs = prepare_inputs(args, work)
    writers = crossover_writers(n)
    readers = [Decoder("zstd.p1", "zstd", 1), Decoder(f"pzstd.p{n}", "pzstd", n)]
    t_start = time.time()
    with open(out / "crossover.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "file",
                "mb",
                "op",
                "method",
                "tool",
                "level",
                "threads",
                "reader",
                "rep",
                "wall",
                "cpu",
                "out_bytes",
                "frames",
                "load1",
            ]
        )
        for label, info in inputs.items():
            for mb in args.sizes_mb:
                size = mb * 2**20
                if size > info["bytes"]:
                    continue
                cut = work / "cut.bin"
                with open(info["path"], "rb") as fin, open(cut, "wb") as fout:
                    fin.seek((info["bytes"] - size) // 2)
                    fout.write(fin.read(size))
                expected = source_crc(cut)
                # Small files are the noisy ones, and the cheap ones.
                reps = 9 if mb <= 32 else 5 if mb <= 128 else 3
                paths = {e: work / f"{e.name}.zst" for e in writers}
                for rep in range(reps):
                    rng.shuffle(writers)
                    for e in writers:
                        paths[e].unlink(missing_ok=True)
                        wall, cpu = run_encoder(e, cut, paths[e])
                        w.writerow(
                            [
                                label,
                                mb,
                                "write",
                                e.name,
                                e.tool,
                                e.level,
                                e.threads,
                                "",
                                rep,
                                wall,
                                cpu,
                                paths[e].stat().st_size,
                                "",
                                os.getloadavg()[0],
                            ]
                        )
                frames = {e: count_frames(p) for e, p in paths.items()}
                # Same level and writer family read alike, so the 1-thread zstd files are not read.
                pairs = [(e, d) for e in writers if e.threads > 1 for d in readers]
                for e, d in pairs:
                    verify_decode(d, paths[e], expected)
                for rep in range(reps):
                    rng.shuffle(pairs)
                    for e, d in pairs:
                        wall, cpu = run_decoder(d, paths[e])
                        w.writerow(
                            [
                                label,
                                mb,
                                "read",
                                e.name,
                                e.tool,
                                e.level,
                                e.threads,
                                d.name,
                                rep,
                                wall,
                                cpu,
                                paths[e].stat().st_size,
                                frames[e],
                                os.getloadavg()[0],
                            ]
                        )
                fh.flush()
                print(
                    f"[{time.time() - t_start:5.0f}s] {label} {mb} MB: frames "
                    + " ".join(
                        f"{e.name}={frames[e]}"
                        for e in sorted(writers, key=lambda x: x.name)
                        if e.tool == "pzstd"
                    ),
                    flush=True,
                )
    print(f"done in {time.time() - t_start:.0f}s", flush=True)


# --- on a real disk ---------------------------------------------------------------


#: Whether a file can be dropped from the page cache without root. Linux can;
#: macOS cannot, so ``disk`` reads there are warm.
CAN_EVICT = hasattr(os, "posix_fadvise")


def flush_and_evict(path: Path) -> float:
    """Flush to the drive, then drop the file from the page cache where the OS allows.

    posix_fadvise needs no root and takes effect once the pages are clean. On macOS
    a plain fsync stops at the drive's cache, so F_FULLFSYNC is used instead. Returns
    the seconds the flush took: the part of a write the disk is responsible for.
    """
    import fcntl

    fd = os.open(path, os.O_RDONLY)
    try:
        t0 = time.perf_counter()
        if hasattr(fcntl, "F_FULLFSYNC"):
            fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
        else:
            os.fsync(fd)
        seconds = time.perf_counter() - t0
        if CAN_EVICT:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return seconds


def disk_encoders(n: int) -> list[Encoder]:
    return [
        enc("isal", 2, 1),
        enc("isal", 2, n),
        enc("pigz", 1, n),
        enc("zstd", 1, 1),
        enc("pzstd", 1, n),
        enc("pzstd", 3, n),
    ]


def disk_pairs(n: int) -> list[tuple[str, str]]:
    """(encoder name, decoder name) for the cold reads."""
    return [
        ("isal-2.p1", "isal.p1"),
        ("isal-2.p1", f"rapidgzip.p{n}"),
        (f"pigz-1.p{n}", f"rapidgzip.p{n}"),
        (f"pzstd-1.p{n}", "zstd.p1"),
        (f"pzstd-1.p{n}", f"pzstd.p{n}"),
        (f"pzstd-3.p{n}", f"pzstd.p{n}"),
    ]


#: Real write route -> the route that reads its file back.
DISK_ROUTES = {
    "nii": "nii",
    "nibabel-isal2": "nibabel-isal",
    "ffs-gz": "ffs-gz",
    "ffs-zst": "ffs-zst",
}


def run_disk(args) -> None:
    """The codec and real routes against an actual disk.

    Sources stay in tmpfs, standing in for an array already in memory. Every write is
    timed with its fsync, since a write that only reached the page cache is not finished,
    and every read starts from a file evicted from that cache.
    """
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work, disk = Path(args.workdir), Path(args.diskdir)
    disk.mkdir(parents=True, exist_ok=True)
    warn_if_not_ram(work)
    if not CAN_EVICT:
        print("note: this OS cannot evict a file without root, so reads will be warm", flush=True)
    (out / "disk_meta.json").write_text(
        json.dumps({"cold_reads": CAN_EVICT, "platform": platform.platform()}, indent=2)
    )
    n = args.threads
    rng = random.Random(args.seed)
    inputs = prepare_inputs(args, work)
    log = Log(out / "results_disk.csv")
    encoders = disk_encoders(n)
    decs = {d.name: d for d in decoders(n)}
    t_start = time.time()
    for label, info in inputs.items():
        common = {k: info[k] for k in ("bytes", "dtype", "kind")} | {"file": label}
        expected = source_crc(info["path"])
        paths = {e.name: disk / f"{label}.{e.name}{e.ext}" for e in encoders}
        plain = disk / f"{label}.plain.nii"
        for rep in range(args.reps):
            os.sync()
            jobs = [*encoders, None]
            rng.shuffle(jobs)
            for e in jobs:
                dst = plain if e is None else paths[e.name]
                dst.unlink(missing_ok=True)
                os.sync()
                if e is None:
                    t0, c0 = time.perf_counter(), _self_cpu()
                    with open(info["path"], "rb") as fin, open(dst, "wb") as fout:
                        shutil.copyfileobj(fin, fout, CHUNK)
                    wall, cpu = time.perf_counter() - t0, _self_cpu() - c0
                else:
                    wall, cpu = run_encoder(e, info["path"], dst)
                flush = flush_and_evict(dst)
                log.row(
                    tier="encode",
                    method="plain" if e is None else e.name,
                    encoder="",
                    decoder="",
                    tool="plain" if e is None else e.tool,
                    level="" if e is None else e.level,
                    threads=1 if e is None else e.threads,
                    rep=rep,
                    wall=wall + flush,
                    cpu=flush,
                    out_bytes=dst.stat().st_size,
                    **common,
                )
            pairs = disk_pairs(n)
            rng.shuffle(pairs)
            reads = [*pairs, None]
            rng.shuffle(reads)
            for pair in reads:
                if pair is None:
                    flush_and_evict(plain)
                    wall, cpu = _timed_cmd(["cat", str(plain)], os.devnull)
                    log.row(
                        tier="decode",
                        method="plain",
                        encoder="",
                        decoder="",
                        tool="plain",
                        level="",
                        threads=1,
                        rep=rep,
                        wall=wall,
                        cpu=cpu,
                        out_bytes=plain.stat().st_size,
                        **common,
                    )
                    continue
                ename, dname = pair
                flush_and_evict(paths[ename])
                wall, cpu = run_decoder(decs[dname], paths[ename])
                log.row(
                    tier="decode",
                    method=f"{ename}>{dname}",
                    encoder=ename,
                    decoder=dname,
                    tool=decs[dname].tool,
                    level="",
                    threads=decs[dname].threads,
                    rep=rep,
                    wall=wall,
                    cpu=cpu,
                    out_bytes=paths[ename].stat().st_size,
                    **common,
                )
            if rep == 0:
                for ename, dname in pairs:
                    verify_decode(decs[dname], paths[ename], expected)
            routes = list(DISK_ROUTES)
            rng.shuffle(routes)
            reference = None
            for route in routes:
                ext = (
                    ".nii" if route == "nii" else ".nii.zst" if route.endswith("zst") else ".nii.gz"
                )
                dst = disk / f"route{ext}"
                dst.unlink(missing_ok=True)
                os.sync()
                res = run_worker("write", route, info["path"], dst, n)
                flush = flush_and_evict(dst)
                log.row(
                    tier="e2e-write",
                    method=route,
                    tool=route,
                    threads=n,
                    rep=rep,
                    wall=res["wall"] + flush,
                    cpu=flush,
                    peak_gb=res["peak_gb"],
                    out_bytes=dst.stat().st_size,
                    **common,
                )
                read_route = DISK_ROUTES[route]
                res = run_worker("read", read_route, dst, Path(os.devnull), n)
                reference = reference if reference is not None else res["sum"]
                if res["sum"] != reference:
                    raise RuntimeError(f"{read_route} read different data from {label}")
                log.row(
                    tier="e2e-read",
                    method=read_route,
                    tool=read_route,
                    threads=n,
                    rep=rep,
                    wall=res["wall"],
                    peak_gb=res["peak_gb"],
                    out_bytes=dst.stat().st_size,
                    **common,
                )
                dst.unlink()
            print(f"[{time.time() - t_start:5.0f}s] {label}: rep {rep + 1}", flush=True)
        for p in [*paths.values(), plain]:
            p.unlink(missing_ok=True)
    print(f"done in {time.time() - t_start:.0f}s", flush=True)


# --- figures and tables -----------------------------------------------------------

BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#7d7c78"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
# Colour is the tool in every figure. The marker tells settings of one tool apart.
TOOL_COLOR = {
    "isal": BLUE,
    "pigz": ORANGE,
    "zstd": AQUA,
    "pzstd": AQUA,
    "rapidgzip": YELLOW,
    "plain": GRAY,
    "nii": GRAY,
    "nibabel-isal2": BLUE,
    "nibabel-isal": BLUE,
    "ffs-gz": ORANGE,
    "ffs-zst": AQUA,
}
ROUTE_NAME = {
    "nii": "uncompressed .nii (nibabel)",
    "nibabel-isal2": "nibabel through ISA-L 2 (the tedana PR)",
    "nibabel-isal": "nibabel through ISA-L",
    "nibabel-rapidgzip": "nibabel through rapidgzip",
    "ffs-gz": "fastfuncstuff .nii.gz (threaded ISA-L / rapidgzip)",
    "ffs-zst": "fastfuncstuff .nii.zst (pzstd)",
}
ROUTE_COLOR = {
    "nii": GRAY,
    "nibabel-isal2": BLUE,
    "nibabel-isal": BLUE,
    "nibabel-rapidgzip": YELLOW,
    "ffs-gz": ORANGE,
    "ffs-zst": AQUA,
}


def load_rows(results: Path, name: str = "results.csv") -> list[dict]:
    rows = list(csv.DictReader(open(results / name)))
    for r in rows:
        r["bytes"] = int(r["bytes"])
        r["wall"] = float(r["wall"])
        r["cpu"] = float(r["cpu"]) if r["cpu"] else float("nan")
        r["out_bytes"] = int(r["out_bytes"])
        r["peak_gb"] = float(r["peak_gb"]) if r["peak_gb"] else float("nan")
        r["threads"] = int(r["threads"])
    return rows


def medians(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    """(tier, file, method) -> medians, with the spread across repetitions."""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["tier"], r["file"], r["method"]), []).append(r)
    out = {}
    for key, g in groups.items():
        wall = statistics.median(x["wall"] for x in g)
        out[key] = dict(g[0]) | {
            "wall": wall,
            "wall_min": min(x["wall"] for x in g),
            "wall_max": max(x["wall"] for x in g),
            "n": len(g),
            "cpu": statistics.median(x["cpu"] for x in g),
            "peak_gb": statistics.median(x["peak_gb"] for x in g),
            "mbps": g[0]["bytes"] / 1e6 / wall,
        }
    return out


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _plain_log_ticks(axis) -> None:
    """1, 2, 5 steps as ordinary numbers; 3x10^2 is no way to write a speed."""
    from matplotlib.ticker import LogLocator, NullFormatter, NullLocator

    axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    axis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())


def _seconds_axis(axis) -> None:
    """Log seconds, written as plain numbers down to the millisecond range."""
    from matplotlib.ticker import LogLocator, NullFormatter, NullLocator

    axis.axes.set_xscale("log")
    axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    axis.set_major_formatter(lambda v, _: f"{v:g}")
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())
    axis.axes.tick_params(axis="x", labelsize=8)


def _mb_label(mb: float) -> str:
    return f"{mb / 1000:.1f} GB" if mb >= 1000 else f"{mb:.0f} MB"


def _title(fig, title: str, sub: str, height: float) -> None:
    fig.text(0.012, 1 - 0.12 / height, title, fontsize=13.5, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 1 - 0.42 / height, sub, fontsize=9.5, color=MUTED, va="top")


def _dot_panel(ax, med, tier, files, methods, value="mbps") -> list:
    """One row per file, one dot per method: (id, label, tool, marker[, alpha])."""
    handles = []
    ys = list(range(len(files)))[::-1]
    for method, label, tool, marker, *fade in methods:
        xs, yy = [], []
        for f, y in zip(files, ys, strict=True):
            v = med.get((tier, f, method))
            if v:
                xs.append(
                    100 * v["out_bytes"] / v["bytes"]
                    if value == "pct"
                    else v["wall"]
                    if value == "seconds"
                    else v["mbps"]
                )
                yy.append(y)
        handles.append(
            ax.scatter(
                xs,
                yy,
                s=62,
                marker=marker,
                alpha=fade[0] if fade else 1.0,
                color=TOOL_COLOR[tool],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
                label=label,
            )
        )
    _style(ax)
    ax.grid(False, axis="x") if value == "pct" else None
    if value == "seconds":
        _seconds_axis(ax.xaxis)
    elif value != "pct":
        ax.set_xscale("log")
        _plain_log_ticks(ax.xaxis)
    return handles


def _save_strip(results, name, title, sub, files, sizes, panels, legend_cols=4) -> None:
    import matplotlib.pyplot as plt

    h = 0.5 * len(files) + 2.6
    fig, axes = plt.subplots(
        1, len(panels), figsize=(max(12.5, 3.6 * len(panels) + 2.6), h), sharey=True
    )
    axes = axes if len(panels) > 1 else [axes]
    seen: dict[str, object] = {}
    for ax, (panel_title, xlabel, handles_fn) in zip(axes, panels, strict=True):
        for handle in handles_fn(ax):
            seen.setdefault(handle.get_label(), handle)
        ax.set_title(panel_title, loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ys = list(range(len(files)))[::-1]
    axes[0].set_yticks(
        ys, [f"{f}  {_mb_label(sizes[f] / 1e6)}" for f in files], color=INK, fontsize=9
    )
    fig.legend(
        list(seen.values()),
        list(seen),
        frameon=False,
        fontsize=9,
        ncols=legend_cols,
        loc="upper left",
        bbox_to_anchor=(0.005, 1 - 0.62 / h),
    )
    fig.subplots_adjust(
        top=1 - 1.55 / h,
        bottom=0.55 / h,
        left=2.65 / max(12.5, 3.6 * len(panels) + 2.6),
        right=0.985,
        wspace=0.06,
    )
    _title(fig, title, sub, h)
    fig.savefig(results / name, dpi=170)
    plt.close(fig)


def _machine(results: Path) -> str:
    meta_path = results / "meta.json"
    return json.loads(meta_path.read_text()).get("cpu", "unknown CPU") if meta_path.exists() else "unknown CPU"


def plot_disk(results: Path, n: int) -> None:
    rows = load_rows(results, "results_disk.csv")
    meta_path = results / "disk_meta.json"
    cold_reads = json.loads(meta_path.read_text())["cold_reads"] if meta_path.exists() else True
    cold = "reads start cold" if cold_reads else "reads are WARM (this OS cannot evict)"
    evicted = (
        "evicted from the page cache before each read"
        if cold_reads
        else "files stayed in the page cache"
    )
    med = medians(rows)
    sizes = {r["file"]: r["bytes"] for r in rows}
    files = sorted(sizes, key=lambda f: sizes[f])
    plain = ("plain", "uncompressed .nii", "plain", "s")
    write = [
        plain,
        (f"isal-2.p{n}", "ISA-L 2", "isal", "o"),
        (f"pzstd-1.p{n}", "pzstd -1", "pzstd", "D"),
        (f"pzstd-3.p{n}", "pzstd -3", "pzstd", "^"),
        (f"pigz-1.p{n}", "pigz -1", "pigz", "o"),
    ]
    read = [
        plain,
        (
            "isal-2.p1>isal.p1",
            "ISA-L, still 1 thread (faded: it cannot use more)",
            "isal",
            "o",
            0.32,
        ),
        (f"isal-2.p1>rapidgzip.p{n}", "rapidgzip", "rapidgzip", "o"),
        (f"pzstd-1.p{n}>pzstd.p{n}", "pzstd -1", "pzstd", "D"),
        (f"pzstd-3.p{n}>pzstd.p{n}", "pzstd -3", "pzstd", "^"),
    ]
    routes_write = [
        ("nii", "uncompressed .nii", "nii", "s"),
        ("nibabel-isal2", "nibabel + ISA-L 2, 1 thread (tedana PR)", "nibabel-isal2", "P"),
        ("ffs-gz", "save_nifti / load_nifti .nii.gz", "ffs-gz", "X"),
        ("ffs-zst", "save_nifti / load_nifti .nii.zst", "ffs-zst", "v"),
    ]
    routes_read = [
        ("nii", "uncompressed .nii", "nii", "s"),
        ("nibabel-isal", "nibabel + ISA-L 2, 1 thread (tedana PR)", "nibabel-isal", "P"),
        ("ffs-gz", "save_nifti / load_nifti .nii.gz", "ffs-gz", "X"),
        ("ffs-zst", "save_nifti / load_nifti .nii.zst", "ffs-zst", "v"),
    ]
    xlabel = "seconds, left is less waiting"
    _save_strip(
        results,
        "fig6_nvme.png",
        f"On a real disk: writes include fsync, {cold}",
        f"{_machine(results)}, {n} threads. Median of 3 shuffled repetitions, {evicted}. Codec panels read to /dev/null; real routes end in a float32 array.",
        files,
        sizes,
        [
            (
                f"Codec write, {n} threads",
                xlabel,
                lambda ax: _dot_panel(ax, med, "encode", files, write, "seconds"),
            ),
            (
                f"Codec read, {n} threads",
                xlabel,
                lambda ax: _dot_panel(ax, med, "decode", files, read, "seconds"),
            ),
            (
                "Real write",
                xlabel,
                lambda ax: _dot_panel(ax, med, "e2e-write", files, routes_write, "seconds"),
            ),
            (
                "Real read",
                xlabel,
                lambda ax: _dot_panel(ax, med, "e2e-read", files, routes_read, "seconds"),
            ),
        ],
        legend_cols=4,
    )


def plot_crossover(results: Path, n: int) -> None:
    """Speed against size. The colours here are settings, not the tools of the other figures."""
    import matplotlib.pyplot as plt

    groups: dict[tuple, list] = {}
    for r in csv.DictReader(open(results / "crossover.csv")):
        key = (r["file"], int(r["mb"]), r["op"], r["method"], r["reader"])
        groups.setdefault(key, []).append(r)
    files = sorted({k[0] for k in groups})

    def curve(ax, f, op, method, reader, label, color, ls, wait=False):
        pts = sorted(k[1] for k in groups if k[:1] == (f,) and k[2:] == (op, method, reader))
        walls = [
            statistics.median(float(x["wall"]) for x in groups[(f, mb, op, method, reader)])
            for mb in pts
        ]
        ys = (
            [1000 * w for w in walls]
            if wait
            else [mb * 2**20 / 1e6 / w for mb, w in zip(pts, walls, strict=True)]
        )
        ax.plot(
            pts, ys, color=color, ls=ls, lw=2, marker="o", ms=4, mec="white", mew=0.7, label=label
        )

    write = [
        (f"pzstd-1.p{n}", f"pzstd -1, {n} threads", AQUA, "-"),
        (f"pzstd-3.p{n}", f"pzstd -3, {n} threads", AQUA, "--"),
        ("zstd-1.p1", "zstd -1, 1 thread", ORANGE, "-"),
        ("zstd-3.p1", "zstd -3, 1 thread", ORANGE, "--"),
        (f"zstd-f1.p{n}", f"zstd --fast=1, {n} threads", BLUE, "-"),
    ]
    read = [
        (f"pzstd-1.p{n}", f"pzstd.p{n}", f"pzstd -1 file, pzstd {n} threads", AQUA, "-"),
        (f"pzstd-3.p{n}", f"pzstd.p{n}", f"pzstd -3 file, pzstd {n} threads", AQUA, "--"),
        (f"pzstd-1.p{n}", "zstd.p1", "pzstd -1 file, zstd 1 thread", ORANGE, "-"),
        (f"pzstd-3.p{n}", "zstd.p1", "pzstd -3 file, zstd 1 thread", ORANGE, "--"),
        (f"zstd-f1.p{n}", "zstd.p1", "--fast=1 file, zstd 1 thread", BLUE, "-"),
    ]
    h = 3.5 * len(files) + 1.9
    fig, axes = plt.subplots(len(files), 4, figsize=(17, h), sharex=True, squeeze=False)
    sizes = sorted({k[1] for k in groups})
    for i, f in enumerate(files):
        for method, label, color, ls in write:
            curve(axes[i][0], f, "write", method, "", label, color, ls, wait=True)
            curve(axes[i][2], f, "write", method, "", label, color, ls)
        for method, reader, label, color, ls in read:
            curve(axes[i][1], f, "read", method, reader, label, color, ls, wait=True)
            curve(axes[i][3], f, "read", method, reader, label, color, ls)
        titles = ("write, wait", "read, wait", "write, speed", "read, speed")
        for j, (ax, what) in enumerate(zip(axes[i], titles, strict=True)):
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            _style(ax)
            _plain_log_ticks(ax.yaxis)
            ax.set_xticks(sizes, [str(v) for v in sizes], fontsize=8)
            ax.minorticks_off()
            ax.set_title(f"{f}\n{what}", loc="left", fontsize=9.5, color=INK)
            ax.set_ylabel(
                "milliseconds, lower is better" if j < 2 else "MB/s", color=MUTED, fontsize=9
            )
    for ax in axes[-1]:
        ax.set_xlabel("size of the cut, MB", color=MUTED, fontsize=9)
    for j in (0, 1):
        handles, labels = axes[0][j].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            frameon=False,
            fontsize=8.5,
            ncols=2,
            loc="upper left",
            bbox_to_anchor=(0.04 + 0.42 * j, 1 - 0.62 / h),
        )
    fig.subplots_adjust(
        top=1 - 1.75 / h, bottom=0.55 / h, left=0.05, right=0.99, wspace=0.26, hspace=0.32
    )
    _title(
        fig,
        "At what file size does pzstd start to pay?",
        f"{_machine(results)}, {n} threads. Cuts from the middle of real files. Process start is included: it is the floor of about 10 ms on the left of each wait panel.",
        h,
    )
    fig.savefig(results / "fig5_crossover.png", dpi=170)
    plt.close(fig)


def make_all(results: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_rows(results)
    meta = json.loads((results / "meta.json").read_text())
    n = meta["threads"]
    med = medians(rows)
    sizes = {r["file"]: r["bytes"] for r in rows}
    files = sorted(sizes, key=lambda f: sizes[f])
    spread = (
        f"{meta.get('cpu', 'unknown CPU')}, {n} threads. "
        f"Median of {meta['reps_large']}-{meta['reps']} shuffled repetitions, tmpfs to tmpfs."
    )

    level_1 = "zstd -1 (pzstd at N threads)"
    level_3 = "zstd -3 (pzstd at N threads)"
    enc_1 = [
        ("isal-2.p1", "ISA-L 2", "isal", "o"),
        ("zstd-1.p1", level_1, "zstd", "D"),
        ("zstd-3.p1", level_3, "zstd", "^"),
    ]
    enc_n = [
        (f"isal-2.p{n}", "ISA-L 2", "isal", "o"),
        (f"pigz-1.p{n}", "pigz -1", "pigz", "o"),
        (f"pzstd-1.p{n}", level_1, "pzstd", "D"),
        (f"pzstd-3.p{n}", level_3, "pzstd", "^"),
    ]
    _save_strip(
        results,
        "fig1_compress.png",
        "Compression: how long you wait, and how fast that is",
        spread + " Size is a check, not the goal.",
        files,
        sizes,
        [
            (
                "Wait, 1 thread",
                "seconds, left is less waiting",
                lambda ax: _dot_panel(ax, med, "encode", files, enc_1, "seconds"),
            ),
            (
                f"Wait, {n} threads",
                "seconds, left is less waiting",
                lambda ax: _dot_panel(ax, med, "encode", files, enc_n, "seconds"),
            ),
            (
                "Speed, 1 thread",
                "MB/s of uncompressed data",
                lambda ax: _dot_panel(ax, med, "encode", files, enc_1),
            ),
            (
                f"Speed, {n} threads",
                "MB/s of uncompressed data",
                lambda ax: _dot_panel(ax, med, "encode", files, enc_n),
            ),
            (
                "Compressed size",
                "% of original",
                lambda ax: _dot_panel(ax, med, "encode", files, enc_n, "pct"),
            ),
        ],
        legend_cols=4,
    )

    dec_1 = [
        ("isal-2.p1>isal.p1", "ISA-L reading an ISA-L file", "isal", "o"),
        (f"pigz-1.p{n}>isal.p1", "ISA-L reading a pigz file", "isal", "s"),
        (f"pzstd-1.p{n}>zstd.p1", "zstd reading a pzstd -1 file", "zstd", "D"),
        (f"pzstd-3.p{n}>zstd.p1", "zstd reading a pzstd -3 file", "zstd", "^"),
    ]
    # ISA-L cannot read in parallel. Its 1-thread time stays on the N-thread panels,
    # faded, because a reader choosing a format still needs to compare against it.
    one_thread_only = "ISA-L, still 1 thread (faded: it cannot use more)"
    dec_n = [
        ("isal-2.p1>isal.p1", one_thread_only, "isal", "o", 0.32),
        (f"pigz-1.p{n}>isal.p1", one_thread_only, "isal", "s", 0.32),
        (f"isal-2.p1>rapidgzip.p{n}", "rapidgzip reading an ISA-L file", "rapidgzip", "o"),
        (f"pigz-1.p{n}>rapidgzip.p{n}", "rapidgzip reading a pigz file", "rapidgzip", "s"),
        (f"zstd-3.p1>pzstd.p{n}", "pzstd reading a single-frame file", "zstd", "v"),
        (f"pzstd-1.p{n}>pzstd.p{n}", "pzstd reading a pzstd -1 file", "pzstd", "D"),
        (f"pzstd-3.p{n}>pzstd.p{n}", "pzstd reading a pzstd -3 file", "pzstd", "^"),
    ]
    _save_strip(
        results,
        "fig2_decompress.png",
        "Decompression: how long you wait, and how fast that is",
        "Output to /dev/null, so nothing caps a fast decoder. " + spread,
        files,
        sizes,
        [
            (
                "Wait, 1 thread",
                "seconds, left is less waiting",
                lambda ax: _dot_panel(ax, med, "decode", files, dec_1, "seconds"),
            ),
            (
                f"Wait, {n} threads",
                "seconds, left is less waiting",
                lambda ax: _dot_panel(ax, med, "decode", files, dec_n, "seconds"),
            ),
            (
                "Speed, 1 thread",
                "MB/s of uncompressed data",
                lambda ax: _dot_panel(ax, med, "decode", files, dec_1),
            ),
            (
                f"Speed, {n} threads",
                "MB/s of uncompressed data",
                lambda ax: _dot_panel(ax, med, "decode", files, dec_n),
            ),
        ],
        legend_cols=4,
    )

    # Size against speed, one curve per tool.
    swept = sorted(
        {
            r["file"]
            for r in rows
            if r["tier"] == "encode" and r["tool"] == "pigz" and r["level"] == "9"
        },
        key=lambda f: sizes[f],
    )
    if swept:
        h = 3.9 * len(swept) + 0.9
        fig, axes = plt.subplots(len(swept), 2, figsize=(12.5, h), sharey="row", squeeze=False)
        for i, f in enumerate(swept):
            for j, threads in enumerate((1, n)):
                ax = axes[i][j]
                for tool, name in (
                    ("isal", "ISA-L"),
                    ("pigz", "pigz"),
                    ("zstd", "zstd"),
                    ("pzstd", "pzstd"),
                ):
                    pts = sorted(
                        (
                            v
                            for k, v in med.items()
                            if k[0] == "encode"
                            and k[1] == f
                            and v["tool"] == tool
                            and v["threads"] == threads
                        ),
                        key=lambda v: int(v["level"]),
                    )
                    if tool == "zstd" and threads > 1:
                        continue  # one point; pzstd is the curve at this thread count
                    # ISA-L 0 barely compresses (about 97%) and would squash every other point.
                    pts = [v for v in pts if not (tool == "isal" and v["level"] == "0")]
                    if not pts:
                        continue
                    xs = [100 * v["out_bytes"] / v["bytes"] for v in pts]
                    ys = [v["mbps"] for v in pts]
                    ax.plot(
                        xs,
                        ys,
                        color=TOOL_COLOR[tool],
                        lw=2,
                        marker="o",
                        ms=6,
                        mec="white",
                        mew=1,
                        label=name,
                    )
                    for x, y, v in zip(xs, ys, pts, strict=True):
                        ax.annotate(
                            v["level"],
                            (x, y),
                            xytext=(0, 7),
                            textcoords="offset points",
                            fontsize=8.5,
                            color=MUTED,
                            ha="center",
                        )
                ax.set_yscale("log")
                _style(ax)
                _plain_log_ticks(ax.yaxis)
                ax.set_title(
                    f"{f}, {threads} thread{'s' if threads > 1 else ''}",
                    loc="left",
                    fontsize=10.5,
                    color=INK,
                )
                ax.set_xlabel("compressed size, % of original", color=MUTED, fontsize=9)
                ax.legend(frameon=False, fontsize=9)
            axes[i][0].set_ylabel("compression MB/s", color=MUTED, fontsize=9)
        fig.subplots_adjust(
            top=1 - 0.9 / h, bottom=0.55 / h, left=0.06, right=0.99, wspace=0.05, hspace=0.32
        )
        _title(
            fig,
            "Size against speed, by compression level",
            "Each point is a level, written beside it. Higher is faster. Note the narrow size axes. ISA-L level 0 is left off: about 97% of original, and at best slightly faster.",
            h,
        )
        fig.savefig(results / "fig3_tradeoff.png", dpi=170)
        plt.close(fig)

    # What a caller waits for, on the two largest files.
    integer = [f for f in files if "int" in next(r["dtype"] for r in rows if r["file"] == f)]
    picks = [f for f in [*integer[-1:], *files[-2:]] if ("e2e-read", f, "nii") in med]
    picks = list(dict.fromkeys(picks))
    if picks:
        h = 3.1 * len(picks) + 0.9
        fig, axes = plt.subplots(len(picks), 2, figsize=(12.5, h), squeeze=False)
        for i, f in enumerate(picks):
            for ax, (tier, title, routes) in zip(
                axes[i],
                [
                    ("e2e-write", "write", WRITE_ROUTES),
                    ("e2e-read", "read", READ_ROUTES),
                ],
                strict=True,
            ):
                items = sorted(
                    (
                        med[(tier, f, r)]
                        for r in routes
                        # A float32 .nii is memory-mapped: "reading" it reads nothing yet.
                        if (tier, f, r) in med and not (tier == "e2e-read" and r == "nii")
                    ),
                    key=lambda v: -v["wall"],
                )
                ys = list(range(len(items)))
                ax.barh(
                    ys,
                    [v["wall"] for v in items],
                    color=[ROUTE_COLOR[v["method"]] for v in items],
                    height=0.62,
                )
                ax.set_yticks(ys, [ROUTE_NAME[v["method"]] for v in items], fontsize=9, color=INK)
                for y, v in zip(ys, items, strict=True):
                    ax.text(
                        v["wall"],
                        y,
                        f"  {v['wall']:.2f} s" if v["wall"] < 10 else f"  {v['wall']:.1f} s",
                        va="center",
                        fontsize=9,
                        color=INK,
                    )
                _style(ax)
                ax.grid(False, axis="y")
                ax.set_xlim(right=max(v["wall"] for v in items) * 1.18)
                ax.set_xlabel("seconds", color=MUTED, fontsize=9)
                ax.set_title(
                    f"{f}  {_mb_label(sizes[f] / 1e6)}, {title}",
                    loc="left",
                    fontsize=10.5,
                    color=INK,
                )
        fig.subplots_adjust(
            top=1 - 0.9 / h, bottom=0.5 / h, left=0.245, right=0.985, wspace=1.05, hspace=0.45
        )
        _title(
            fig,
            "What a Python caller actually waits for",
            f"Array to file and file to float32 array. Fresh process per trial, tmpfs, {n} threads where the route can use them.",
            h,
        )
        fig.savefig(results / "fig4_end_to_end.png", dpi=170)
        plt.close(fig)

    if (results / "crossover.csv").exists():
        plot_crossover(results, n)
    if (results / "results_disk.csv").exists():
        plot_disk(results, n)

    lines = [f"# Results ({meta.get('cpu', '')}, {n} threads)\n"]
    for f in files:
        info = next(r for r in rows if r["file"] == f)
        lines += [
            f"\n## {f}: {info['bytes'] / 1e6:.0f} MB {info['dtype']} {info['kind']}\n",
            "\n| encoder | seconds | MB/s | spread s | CPU s | size % |\n|---|--:|--:|--:|--:|--:|",
        ]
        for k, v in sorted(med.items(), key=lambda kv: -kv[1]["mbps"]):
            if k[0] == "encode" and k[1] == f:
                lines.append(
                    f"| {k[2]} | {v['wall']:.2f} | {v['mbps']:.0f} | {v['wall_min']:.2f}-{v['wall_max']:.2f} | {v['cpu']:.1f} | {100 * v['out_bytes'] / v['bytes']:.1f} |"
                )
        lines.append(
            "\n| file > decoder | seconds | MB/s | spread s | CPU s |\n|---|--:|--:|--:|--:|"
        )
        for k, v in sorted(med.items(), key=lambda kv: -kv[1]["mbps"]):
            if k[0] == "decode" and k[1] == f:
                lines.append(
                    f"| {k[2]} | {v['wall']:.2f} | {v['mbps']:.0f} | {v['wall_min']:.2f}-{v['wall_max']:.2f} | {v['cpu']:.1f} |"
                )
        for tier in ("e2e-write", "e2e-read"):
            lines.append(
                f"\n| {tier} | seconds | spread s | peak GB | file MB |\n|---|--:|--:|--:|--:|"
            )
            for k, v in sorted(med.items(), key=lambda kv: kv[1]["wall"]):
                if k[0] == tier and k[1] == f:
                    lines.append(
                        f"| {k[2]} | {v['wall']:.2f} | {v['wall_min']:.2f}-{v['wall_max']:.2f} | {v['peak_gb']:.2f} | {v['out_bytes'] / 1e6:.0f} |"
                    )
    (results / "summary.md").write_text("\n".join(lines) + "\n")
    loads = [float(r["load1"]) for r in rows]
    print(
        f"{len(rows)} trials; load average median {statistics.median(loads):.1f}, max {max(loads):.1f}"
    )


# --- entry ------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        worker_main(sys.argv[2:])
        return
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("-out", required=True)
    r.add_argument(
        "-workdir", default=scratch_dir("bench_nifti_compression"), help="RAM-backed scratch space"
    )
    r.add_argument(
        "-file", action="append", default=[], metavar="LABEL=PATH", help="another file; repeatable"
    )
    r.add_argument(
        "-sweep", nargs="*", default=[], help="labels that also get the compression level sweep"
    )
    r.add_argument("-threads", type=int, default=default_threads())
    r.add_argument("-reps", type=int, default=5)
    r.add_argument(
        "-reps_large", type=int, default=3, help="repetitions for files of 500 MB and up"
    )
    r.add_argument("-seed", type=int, default=0)
    r.add_argument("-no_e2e", action="store_true", help="skip the end-to-end array routes")
    x = sub.add_parser("crossover", help="at what file size does pzstd start to pay?")
    x.add_argument("-out", required=True)
    x.add_argument(
        "-workdir", default=scratch_dir("bench_nifti_crossover"), help="RAM-backed scratch space"
    )
    x.add_argument("-file", action="append", default=[], metavar="LABEL=PATH")
    x.add_argument(
        "-sizes_mb", nargs="*", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    )
    x.add_argument("-threads", type=int, default=default_threads())
    x.add_argument("-seed", type=int, default=0)
    d = sub.add_parser("disk", help="cold reads and fsynced writes on a real disk")
    d.add_argument("-out", required=True)
    d.add_argument("-diskdir", required=True, help="scratch space on the disk under test")
    d.add_argument(
        "-workdir", default=scratch_dir("bench_nifti_disk"), help="RAM-backed space for the sources"
    )
    d.add_argument("-file", action="append", default=[], metavar="LABEL=PATH")
    d.add_argument("-threads", type=int, default=default_threads())
    d.add_argument("-reps", type=int, default=3)
    d.add_argument("-seed", type=int, default=0)
    q = sub.add_parser("plot")
    q.add_argument("results")
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
    elif args.cmd == "disk":
        try:
            run_disk(args)
        finally:
            shutil.rmtree(args.workdir, ignore_errors=True)
            shutil.rmtree(args.diskdir, ignore_errors=True)
    elif args.cmd == "crossover":
        try:
            run_crossover(args)
        finally:
            shutil.rmtree(args.workdir, ignore_errors=True)
    else:
        make_all(Path(args.results))


if __name__ == "__main__":
    main()
