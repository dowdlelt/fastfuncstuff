#!/usr/bin/env python
"""How fast are the ways of compressing and reading a NIfTI? A fair comparison.

Compares zlib (what nibabel uses), ISA-L, pigz, zstd and pzstd as encoders, and
those plus rapidgzip as decoders, on your own files. Standalone: it needs numpy,
nibabel, matplotlib, the ``isal`` and ``rapidgzip`` Python packages and the
``pigz``, ``zstd`` and ``pzstd`` commands. If ``fastfuncstuff`` is importable it
also times that toolbox's own save and load routes.

    python bench_nifti_compression.py run -out results -workdir /dev/shm/bench \
        -ladder echo1.nii.gz echo2.nii.gz -ladder_mb 10 100 1000 \
        -file raw_phase=phase.nii.gz -file t1=t1.nii.gz -sweep echo-ladder-1000
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
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import platform
import random
import resource
import shutil
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
    tool: str  # zlib | isal | pigz | zstd | pzstd
    level: int
    threads: int

    @property
    def ext(self) -> str:
        return ".gz" if self.tool in ("zlib", "isal", "pigz") else ".zst"


@dataclass(frozen=True)
class Decoder:
    name: str
    tool: str  # zlib | isal | pigz | rapidgzip | zstd | pzstd
    threads: int

    @property
    def ext(self) -> str:
        return ".gz" if self.tool in ("zlib", "isal", "pigz", "rapidgzip") else ".zst"


def enc(tool: str, level: int, threads: int) -> Encoder:
    return Encoder(f"{tool}-{level}.p{threads}", tool, level, threads)


def core_encoders(n: int) -> list[Encoder]:
    """The settings people actually run, on every file.

    zlib-1 is nibabel's default, ISA-L 2 is its default and the tedana PR's choice,
    pigz -1 and pzstd -3 are what fastfuncstuff writes, zstd -1 is its fast setting.
    """
    return [
        enc("zlib", 1, 1),
        enc("isal", 2, 1),
        enc("isal", 2, n),
        enc("pigz", 1, 1),
        enc("pigz", 1, n),
        enc("zstd", 1, 1),
        enc("zstd", 3, 1),
        enc("zstd", 3, n),
        enc("pzstd", 1, n),
        enc("pzstd", 3, n),
    ]


def sweep_encoders(n: int) -> list[Encoder]:
    """Level sweeps, for the size-against-speed curve."""
    out = [enc("zlib", lv, 1) for lv in (1, 6, 9)]
    for t in (1, n):
        out += [enc("isal", lv, t) for lv in (0, 1, 2, 3)]
        out += [enc("pigz", lv, t) for lv in (1, 3, 6, 9)]
    out += [enc("zstd", lv, 1) for lv in (1, 3, 6, 9, 12)]
    out += [enc("pzstd", lv, n) for lv in (1, 3, 6, 9, 12)]
    return out


def decoders(n: int) -> list[Decoder]:
    return [
        Decoder("zlib.p1", "zlib", 1),
        Decoder("isal.p1", "isal", 1),
        Decoder(f"pigz.p{n}", "pigz", n),  # pigz never decodes in parallel; N buys helper threads
        Decoder("rapidgzip.p1", "rapidgzip", 1),
        Decoder(f"rapidgzip.p{n}", "rapidgzip", n),
        Decoder("zstd.p1", "zstd", 1),
        Decoder(f"pzstd.p{n}", "pzstd", n),
    ]


def decode_encoders(n: int) -> list[Encoder]:
    """Whose files get decoded. Same format and level give the same read speed, so
    the other core encoders would only repeat these."""
    return [
        enc("zlib", 1, 1),
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
        return ["zstd", "-q", "-c", f"-{e.level}", f"-T{e.threads}", str(src)]
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


def _gz_writer(e: Encoder, fobj):
    # Empty name and mtime=0 in both, as nibabel and the tedana PR write them.
    if e.tool == "zlib":
        return gzip.GzipFile("", "wb", e.level, fobj, mtime=0)
    from isal import igzip

    return igzip.IGzipFile("", "wb", e.level, fobj, mtime=0)


def run_encoder(e: Encoder, src: Path, dst: Path) -> tuple[float, float]:
    if e.tool in ("pigz", "zstd", "pzstd"):
        return _timed_cmd(encode_cmd(e, src), dst)
    t0, c0 = time.perf_counter(), _self_cpu()
    with open(src, "rb") as fin:
        if e.tool == "isal" and e.threads > 1:
            from isal import igzip_threaded

            with igzip_threaded.open(dst, "wb", compresslevel=e.level, threads=e.threads) as gz:
                shutil.copyfileobj(fin, gz, CHUNK)
        else:
            with open(dst, "wb") as fout, _gz_writer(e, fout) as gz:
                shutil.copyfileobj(fin, gz, CHUNK)
    return time.perf_counter() - t0, _self_cpu() - c0


def _open_in_process(d: Decoder, src: Path):
    if d.tool == "zlib":
        return gzip.open(src, "rb")
    from isal import igzip

    return igzip.open(src, "rb")


def run_decoder(d: Decoder, src: Path) -> tuple[float, float]:
    if d.tool not in ("zlib", "isal"):
        return _timed_cmd(decode_cmd(d, src), os.devnull)
    buf = bytearray(CHUNK)
    t0, c0 = time.perf_counter(), _self_cpu()
    with _open_in_process(d, src) as fh:
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
    if d.tool in ("zlib", "isal"):
        with _open_in_process(d, src) as fh:
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


def _write_series(paths: list[Path], dst: Path, target_bytes: int) -> tuple[int, str]:
    """Join ``paths`` along time and keep whole volumes, up to about ``target_bytes``."""
    import nibabel as nib
    import numpy as np

    first = nib.load(paths[0])
    shape, dtype = first.shape, np.dtype(first.get_data_dtype())
    vol_bytes = int(np.prod(shape[:3])) * dtype.itemsize
    keep = max(1, round(target_bytes / vol_bytes))
    arrays, have = [], 0
    for p in paths:
        img = nib.load(p)
        if img.shape[:3] != shape[:3] or np.dtype(img.get_data_dtype()) != dtype:
            raise ValueError(f"{p}: cannot join with {paths[0]}: different grid or datatype")
        arr = np.asanyarray(img.dataobj)
        if arr.ndim == 3:
            arr = arr[..., None]
        arrays.append(arr[..., : keep - have])
        have += arrays[-1].shape[3]
        if have >= keep:
            break
    data = np.concatenate(arrays, axis=3) if len(arrays) > 1 else arrays[0]
    if data.dtype != dtype:
        data = data.astype(dtype)  # a scaled input decodes wider than its stored type
    nib.save(nib.Nifti1Image(data, first.affine, first.header), dst)
    return dst.stat().st_size, str(dtype)


def prepare_inputs(args, work: Path) -> dict[str, dict]:
    """Uncompressed .nii copies in tmpfs. The result carries no source paths."""
    src_dir = work / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    inputs: dict[str, dict] = {}
    if args.ladder:
        paths = [Path(p) for p in args.ladder]
        for mb in sorted(args.ladder_mb):
            label = f"ladder-{mb}"
            dst = src_dir / f"{label}.nii"
            size, dtype = _write_series(paths, dst, mb * 10**6)
            inputs[label] = {"path": dst, "bytes": size, "dtype": dtype, "kind": "ladder"}
    for spec in args.file:
        label, _, path = spec.partition("=")
        dst = src_dir / f"{label}.nii"
        size, dtype = _write_series([Path(path)], dst, 10**15)
        inputs[label] = {"path": dst, "bytes": size, "dtype": dtype, "kind": "type"}
    return inputs


# --- end-to-end routes, each timed in a fresh process -----------------------------

WRITE_ROUTES = ["nii", "nibabel-zlib1", "nibabel-isal2", "ffs-gz", "ffs-zst"]
READ_ROUTES = ["nii", "nibabel-zlib", "nibabel-isal", "nibabel-rapidgzip", "ffs-gz", "ffs-zst"]


def _peak_rss_gb() -> float:
    # VmHWM, not ru_maxrss: ru_maxrss survives exec, so every worker would report
    # the harness's own peak.
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024**2
    return float("nan")


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
    arr = np.asarray(load(str(src)).dataobj, dtype=np.float32)
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
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            out["cpu"] = line.split(":", 1)[1].strip()
            break
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
        "-workdir", default="/dev/shm/bench_nifti_compression", help="tmpfs scratch space"
    )
    r.add_argument(
        "-ladder", nargs="*", default=[], help="series to join along time for the size ladder"
    )
    r.add_argument("-ladder_mb", nargs="*", type=int, default=[10, 100, 1000])
    r.add_argument(
        "-file", action="append", default=[], metavar="LABEL=PATH", help="another file; repeatable"
    )
    r.add_argument(
        "-sweep", nargs="*", default=[], help="labels that also get the compression level sweep"
    )
    r.add_argument("-threads", type=int, default=len(os.sched_getaffinity(0)) // 2)
    r.add_argument("-reps", type=int, default=5)
    r.add_argument(
        "-reps_large", type=int, default=3, help="repetitions for files of 500 MB and up"
    )
    r.add_argument("-seed", type=int, default=0)
    r.add_argument("-no_e2e", action="store_true", help="skip the end-to-end array routes")
    q = sub.add_parser("plot")
    q.add_argument("results")
    args = p.parse_args()
    if args.cmd == "run":
        try:
            run(args)
        finally:
            shutil.rmtree(args.workdir, ignore_errors=True)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
