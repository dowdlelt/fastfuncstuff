"""Central, robust torch.compile policy for fastfuncstuff.

Why this exists
---------------
Inductor's C++ (CPU) backend caches a *precompiled header* (PCH) to speed up
compiling many kernels. Its clang validation rejects the cached PCH whenever the
header file's mtime drifts from when the PCH was built — emitting

    fatal error: file '...h' has been modified since the precompiled header
    '...h.pch' was built: mtime changed

and aborting the *entire* compile. That drift happens routinely: editing the code
base, concurrent build workers, bleeding-edge toolchains (e.g. Python 3.14), or a
synced filesystem touching mtimes. We compile only a handful of distinct kernels,
so the PCH buys us almost nothing — but it can crash a whole analysis.

Policy
------
1. **Disable the PCH** process-wide (`cpp_cache_precompile_headers = False`). Kernels
   still compile and the compiled-``.so`` cache is still used, so steady-state speed
   is unchanged — we only drop a fragile *build-time* optimization. This removes the
   staleness failure mode entirely.
2. Compile through :func:`safe_compile` so any *other* compile failure (no C++
   compiler, an inductor bug on a new torch) **degrades to eager** for the rest of
   the process instead of crashing, warning at most once.

Use ``safe_compile`` everywhere instead of calling ``torch.compile`` directly.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any

import torch

_configured = False
_warned_fallback = False


def configure_inductor() -> None:
    """Apply the inductor policy once. Safe to call repeatedly and before any compile.

    The inductor config is read at *compile* time (first call of a compiled fn), so
    setting it at import — before any kernel is invoked — is sufficient and global.
    """
    global _configured
    if _configured:
        return
    _configured = True
    try:
        import torch._inductor.config as ind

        # The fragile bit: disable the precompiled-header optimization. The .so
        # kernel cache is untouched, so we keep fused-kernel speed and caching.
        if hasattr(ind, "cpp_cache_precompile_headers"):
            ind.cpp_cache_precompile_headers = False
        # AOT path (used by some export/inductor flows) has its own flag.
        if hasattr(ind, "aot_inductor") and hasattr(ind.aot_inductor, "precompile_headers"):
            ind.aot_inductor.precompile_headers = False
    except Exception:
        # Never let compile configuration break import — worst case we keep defaults
        # and safe_compile's eager fallback still prevents crashes.
        pass


# Apply at import so it is in effect before any compiled kernel is first called.
configure_inductor()


def safe_compile(fn: Callable | None = None, **compile_kwargs: Any) -> Callable:
    """``torch.compile`` with the shared inductor policy and a permanent eager fallback.

    Usable as a direct call (``safe_compile(fn, dynamic=True)``) or a decorator
    (``@safe_compile(dynamic=True)``). If inductor compilation fails at call time,
    the wrapper switches to the eager ``fn`` for the rest of the process and warns
    once — a fused-kernel speedup must never crash or silently spam a run.
    """
    if fn is None:
        return functools.partial(safe_compile, **compile_kwargs)

    configure_inductor()
    try:
        compiled = torch.compile(fn, **compile_kwargs)
    except Exception:
        return fn  # compile unavailable entirely → just run eager

    state = {"disabled": False}

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        global _warned_fallback
        if state["disabled"]:
            return fn(*args, **kwargs)
        try:
            return compiled(*args, **kwargs)
        except Exception as e:  # inductor/clang failures are Exception subclasses
            state["disabled"] = True
            if not _warned_fallback:
                _warned_fallback = True
                warnings.warn(
                    f"torch.compile fell back to eager ({type(e).__name__}); results "
                    "are unaffected. Delete the torchinductor cache "
                    "(TORCHINDUCTOR_CACHE_DIR, else the temp 'torchinductor_*' dir) to "
                    "retry the compiled path.",
                    stacklevel=2,
                )
            return fn(*args, **kwargs)

    wrapper._ffs_eager = fn  # type: ignore[attr-defined]  # escape hatch for tests
    return wrapper
