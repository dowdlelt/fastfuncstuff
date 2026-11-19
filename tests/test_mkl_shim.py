"""The CLI-launch MKL shim: when it relaunches, and that the relaunch really preloads it.

The decision runs on every ``ffs_*`` start, so a wrong "yes" would restart tools
on machines it cannot help, and a wrong "no" silently halves torch's BLAS on
AVX-512 AMD servers.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

from fastfuncstuff import _mkl_shim as S

AVX512 = "fpu sse2 avx2 fma avx512f avx512dq avx512cd avx512bw avx512vl"
AMD = f"processor\t: 0\nvendor_id\t: AuthenticAMD\nflags\t\t: {AVX512}\n"
INTEL = f"processor\t: 0\nvendor_id\t: GenuineIntel\nflags\t\t: {AVX512}\n"
ZEN3 = "processor\t: 0\nvendor_id\t: AuthenticAMD\nflags\t\t: fpu sse2 avx2 fma\n"


def test_only_avx512_amd_wants_it():
    assert S.cpu_wants_shim(AMD)
    assert not S.cpu_wants_shim(INTEL)  # MKL already picks its best path on Intel
    assert not S.cpu_wants_shim(ZEN3)  # AVX2 is already the best it has
    assert S.cpu_wants_shim(INTEL, force=True)
    assert not S.cpu_wants_shim(ZEN3, force=True)


@pytest.mark.parametrize(
    ("env", "argv0", "modules", "want"),
    [
        ({}, "/env/bin/ffs_nordic", {}, True),
        ({"FFS_NO_SHIM": "1"}, "/env/bin/ffs_nordic", {}, False),
        ({"FFS_NO_SHIM": "0"}, "/env/bin/ffs_nordic", {}, True),
        ({"LD_PRELOAD": "/x/mkl_force_avx512-abc.so"}, "/env/bin/ffs_nordic", {}, False),
        ({"_FFS_SHIM_EXEC": "1"}, "/env/bin/ffs_nordic", {}, False),
        ({}, "/env/bin/ffs_nordic", {"torch": None}, False),  # too late to preload
        ({}, "/env/bin/ipython", {}, False),  # notebooks and scripts are left alone
    ],
)
def test_relaunch_decision(env, argv0, modules, want):
    assert S.should_reexec(env, argv0, AMD, modules) is want


def test_reads_only_the_first_cpu(tmp_path):
    p = tmp_path / "cpuinfo"
    p.write_text(AMD + "\n" + INTEL.replace("processor\t: 0", "processor\t: 1"))
    assert "AuthenticAMD" in S._first_cpu(str(p))
    assert "GenuineIntel" not in S._first_cpu(str(p))


needs_cc = pytest.mark.skipif(
    sys.platform != "linux" or not any(shutil.which(c) for c in ("cc", "gcc", "clang")),
    reason="needs Linux and a C compiler",
)


@needs_cc
def test_builds_once_into_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    lib = S.built_library()
    assert lib is not None and lib.exists() and lib.parent == tmp_path / "fastfuncstuff"
    mtime = lib.stat().st_mtime_ns
    assert S.built_library() == lib and lib.stat().st_mtime_ns == mtime
    assert [p.name for p in lib.parent.iterdir()] == [lib.name]  # no temp files left


@needs_cc
@pytest.mark.skipif(not S.cpu_wants_shim(S._first_cpu(), force=True), reason="needs AVX-512")
def test_an_ffs_cli_relaunches_with_the_shim_loaded(tmp_path):
    # A stand-in console script: the name is what marks it as a CLI.
    script = tmp_path / "ffs_probe"
    script.write_text(
        "import fastfuncstuff, os\n"
        "maps = open('/proc/self/maps').read()\n"
        "print('loaded' if 'mkl_force_avx512' in maps else 'absent')\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("LD_PRELOAD", "_FFS_SHIM_EXEC")}
    env.update(XDG_CACHE_HOME=str(tmp_path / "cache"), FFS_SHIM_FORCE="1")

    def run(**extra):
        out = subprocess.run(
            [sys.executable, str(script)], env={**env, **extra}, capture_output=True, text=True
        )
        assert out.returncode == 0, out.stderr
        return out.stdout.strip()

    assert run() == "loaded"
    assert run(FFS_NO_SHIM="1") == "absent"
    assert run(FFS_SHIM_FORCE="0") == ("loaded" if S.cpu_wants_shim(S._first_cpu()) else "absent")
