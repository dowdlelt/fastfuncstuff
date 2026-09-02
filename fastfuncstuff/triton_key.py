"""Cache Triton's compilation-cache key across processes.

Triton derives the key that namespaces its on-disk kernel cache by sha256-ing
its own source tree *and* ``libtriton.so`` -- 461 MB in Triton 3.7 -- on the
first kernel launch of every process.  It memoises the result in-process, so a
long-running fit pays it once and never notices; a toolbox of one-shot CLIs
pays it on every invocation.  Measured here: **1.003 s**, which is 26% of an
``ffs_moco`` run and 27% of an ``ffs_nwarp`` run.  Ten subjects processed as
ten separate calls pay it ten times.

The key only changes when the Triton installation changes, so we keep it in a
small JSON file next to the rest of the user cache, guarded by a fingerprint of
the installation.  A stale key would silently reuse kernels compiled by a
different Triton, so the fingerprint is deliberately conservative: version,
interpreter tag, the size and mtime of ``libtriton.so``, and the count and
newest mtime of every ``.py`` under the package.  Any edit, reinstall, or
upgrade moves it, and a fingerprint miss falls back to the real computation.

Set ``FFS_NO_TRITON_KEY_CACHE=1`` to disable.
"""

from __future__ import annotations

import functools
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

_SCHEMA = 1
_installed = False
_patched: Any = None


def _cache_path() -> Path:
    root = os.environ.get("XDG_CACHE_HOME") or "~/.cache"
    return Path(root).expanduser() / "fastfuncstuff" / "triton_key.json"


def _fingerprint(triton: Any) -> str:
    package = Path(triton.__file__).parent
    newest = 0
    count = 0
    for path in package.rglob("*.py"):
        try:
            newest = max(newest, path.stat().st_mtime_ns)
        except OSError:
            continue
        count += 1
    library = 0, 0
    for candidate in (package / "_C").glob("libtriton*"):
        try:
            stat = candidate.stat()
        except OSError:
            continue
        library = max(library, (stat.st_size, stat.st_mtime_ns))
    return "|".join(
        str(part)
        for part in (
            _SCHEMA,
            getattr(triton, "__version__", "?"),
            sys.implementation.cache_tag,
            package,
            count,
            newest,
            *library,
        )
    )


def _read(fingerprint: str) -> str | None:
    try:
        stored = json.loads(_cache_path().read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(stored, dict) or stored.get("fingerprint") != fingerprint:
        return None
    key = stored.get("key")
    return key if isinstance(key, str) else None


def _write(fingerprint: str, key: str) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Another CLI may be writing the same file; rename is atomic.
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=path.name, suffix=".tmp", delete=False
        ) as handle:
            json.dump({"fingerprint": fingerprint, "key": key}, handle)
            temporary = Path(handle.name)
        temporary.replace(path)
    except OSError:
        pass  # A read-only or full cache directory costs a second, not a run.


def install_triton_key_cache() -> bool:
    """Back ``triton.runtime.cache.triton_key`` with a cross-process cache.

    Returns True when the patch is in place.  Safe to call repeatedly and from
    any import order; every failure path leaves Triton exactly as it was.
    """
    global _installed, _patched
    if _installed:
        return True
    if os.environ.get("FFS_NO_TRITON_KEY_CACHE", "").strip() not in ("", "0"):
        return False
    try:
        import triton
        import triton.runtime.cache as triton_cache
    except Exception:
        return False

    original = getattr(triton_cache, "triton_key", None)
    if original is None or original is _patched:
        return False

    try:
        fingerprint = _fingerprint(triton)
    except Exception:
        return False

    @functools.lru_cache(maxsize=1)
    def triton_key() -> str:
        key = _read(fingerprint)
        if key is None:
            key = original()
            _write(fingerprint, key)
        return key

    _patched = triton_key
    triton_cache.triton_key = triton_key
    # autotuner does ``from .cache import triton_key`` at import time, so it
    # holds its own reference that a module-attribute swap would not reach.
    autotuner = sys.modules.get("triton.runtime.autotuner")
    if autotuner is not None and hasattr(autotuner, "triton_key"):
        # Writing through __dict__ keeps both ruff and ty quiet about
        # assigning an attribute onto an arbitrary module object.
        autotuner.__dict__["triton_key"] = triton_key
    _installed = True
    return True
