"""Run the ``ffs_*`` tools with torch's MKL on its AVX-512 kernels on AMD CPUs.

MKL picks its code path from ``mkl_serv_cpu_detect()``, which answers "AVX2" on
any non-Intel CPU. On an AVX-512 AMD CPU (Zen 4, Zen 5) that halves torch's
GEMM: measured on an EPYC 9555, preloading ``_native/mkl_force_avx512.c`` took
float32 GEMM from 119 to 271 GFLOP/s per core and the GLM shape ``pinv(X) @ Y``
6.6x faster at 32 threads. See that file for how the override works.

A preload only takes effect when a process starts, so a CLI that wants it
re-executes itself once, before torch is imported, with ``LD_PRELOAD`` set. The
library is compiled on first use into the user cache (well under a second) so
the package stays pure Python; without a C compiler nothing happens.

Only the command-line tools do this -- ``import fastfuncstuff`` from a notebook
cannot restart its own interpreter. ``FFS_NO_SHIM=1`` turns it off;
``FFS_SHIM_FORCE=1`` applies it on any AVX-512 CPU, Intel included (for tests:
on Intel it can only hold MKL to an older path).

Must not import torch, numpy or anything heavy: it runs on every CLI launch.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SOURCE = Path(__file__).with_name("_native") / "mkl_force_avx512.c"
_NAME = "mkl_force_avx512"
_FLAGS = ("avx512f", "avx512cd", "avx512bw", "avx512dq", "avx512vl")


def _first_cpu(cpuinfo_path: str = "/proc/cpuinfo") -> str:
    """The first processor's block: vendor and flags, without reading 256 of them."""
    lines = []
    try:
        with open(cpuinfo_path) as fh:
            for line in fh:
                if not line.strip() and lines:
                    break
                lines.append(line)
    except OSError:
        return ""
    return "".join(lines)


def cpu_wants_shim(cpuinfo: str, *, force: bool = False) -> bool:
    """AMD with the AVX-512 subsets MKL's Skylake-SP path needs."""
    fields = dict(
        (k.strip(), v.strip()) for k, _, v in (ln.partition(":") for ln in cpuinfo.splitlines())
    )
    flags = set(fields.get("flags", "").split())
    if not all(f in flags for f in _FLAGS):
        return False
    return force or fields.get("vendor_id") == "AuthenticAMD"


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() not in ("", "0", "false", "no", "off")


def should_reexec(env, argv0: str, cpuinfo: str, modules) -> bool:
    if _truthy(env.get("FFS_NO_SHIM")) or env.get("_FFS_SHIM_EXEC"):
        return False
    if _NAME in env.get("LD_PRELOAD", ""):
        return False
    # Too late once torch is loaded: MKL is linked into libtorch_cpu.
    if "torch" in modules:
        return False
    if not os.path.basename(argv0).startswith("ffs_"):
        return False
    return cpu_wants_shim(cpuinfo, force=_truthy(env.get("FFS_SHIM_FORCE")))


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "fastfuncstuff"


def built_library() -> Path | None:
    """The compiled shim, building it if needed; ``None`` if that is not possible."""
    import hashlib
    import shutil
    import subprocess
    import tempfile

    try:
        src = _SOURCE.read_bytes()
    except OSError:
        return None
    out = _cache_dir() / f"{_NAME}-{hashlib.sha1(src).hexdigest()[:12]}.so"
    if out.exists():
        return out
    cc = next((c for c in ("cc", "gcc", "clang") if shutil.which(c)), None)
    if cc is None:
        return None
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        # Build beside the target and rename: two jobs launched together must not
        # see each other's half-written file.
        fd, tmp = tempfile.mkstemp(suffix=".so", dir=out.parent)
        os.close(fd)
        try:
            subprocess.run(
                [cc, "-O2", "-shared", "-fPIC", "-o", tmp, str(_SOURCE), "-ldl"],
                check=True,
                capture_output=True,
                timeout=60,
            )
            os.replace(tmp, out)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    except (OSError, subprocess.SubprocessError):
        return None
    return out


def maybe_reexec() -> None:
    """Restart this CLI with the shim preloaded, when it would help. Never raises."""
    if sys.platform != "linux" or os.uname().machine != "x86_64":
        return
    try:
        if not should_reexec(os.environ, sys.argv[0], _first_cpu(), sys.modules):
            return
        lib = built_library()
        if lib is None:
            return
        env = dict(os.environ)
        env["LD_PRELOAD"] = ":".join(p for p in (str(lib), env.get("LD_PRELOAD", "")) if p)
        env["_FFS_SHIM_EXEC"] = "1"  # one attempt, even if the loader drops the preload
        sys.stdout.flush()
        sys.stderr.flush()
        os.execve(sys.executable, sys.orig_argv, env)
    except Exception:
        return
