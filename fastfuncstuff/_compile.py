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
import time
import warnings
from collections.abc import Callable
from typing import Any

import torch

_configured = False
_warned_fallback = False


def configure_inductor() -> None:
    """Apply the inductor policy once. Safe to call repeatedly and before any compile.

    Call this before a compiled function is first *invoked*, not at import — the
    inductor config is read at compile time, and importing it is expensive.
    :func:`safe_compile` already does so; callers rarely need this directly.
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


def safe_compile(fn: Callable | None = None, **compile_kwargs: Any) -> Callable:
    """``torch.compile`` with the shared inductor policy and a permanent eager fallback.

    Usable as a direct call (``safe_compile(fn, dynamic=True)``) or a decorator
    (``@safe_compile(dynamic=True)``). If inductor compilation fails at call time,
    the wrapper switches to the eager ``fn`` for the rest of the process and warns
    once — a fused-kernel speedup must never crash or silently spam a run.
    """
    if fn is None:
        return functools.partial(safe_compile, **compile_kwargs)

    # Everything inductor-related is deferred to the first *call*. Both
    # `torch.compile()` and `import torch._inductor.config` drag in dynamo and
    # sympy (~1 s), and most modules wrap their kernels at import time — so doing
    # this eagerly taxed every CLI startup, including ones that never run a kernel.
    state: dict[str, Any] = {"disabled": False, "compiled": None}

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        global _warned_fallback
        if state["disabled"]:
            return fn(*args, **kwargs)
        compiled = state["compiled"]
        if compiled is None:
            configure_inductor()
            try:
                compiled = torch.compile(fn, **compile_kwargs)
            except Exception:
                state["disabled"] = True  # compile unavailable entirely → run eager
                return fn(*args, **kwargs)
            state["compiled"] = compiled
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

    wrapper._ffs_eager = fn  # ty: ignore[unresolved-attribute]  # escape hatch for tests
    return wrapper


def compile_after_eager_time(
    fn: Callable | None = None,
    *,
    min_eager_seconds: float = 3.0,
    **compile_kwargs: Any,
) -> Callable:
    """Compile only after this process has spent enough time in eager execution.

    This is for kernels shared by both one-shot and highly repetitive workloads.
    A fresh process avoids Dynamo's multi-second startup cost when eager execution
    is already cheaper, while repeated calls eventually compile once and reuse the
    compiled function. CUDA timing uses asynchronous events and never synchronizes.
    """
    if fn is None:
        return functools.partial(
            compile_after_eager_time,
            min_eager_seconds=min_eager_seconds,
            **compile_kwargs,
        )

    state: dict[str, Any] = {
        "eager_seconds": 0.0,
        "cuda_unmeasured_calls": 0,
        "pending_cuda": None,
        "compiled": None,
    }

    def _collect_cuda_time() -> None:
        pending = state["pending_cuda"]
        if pending is None:
            return
        start, end, represented_calls = pending
        if end.query():
            state["eager_seconds"] += start.elapsed_time(end) * represented_calls / 1000.0
            state["pending_cuda"] = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        tensor = next((arg for arg in args if isinstance(arg, torch.Tensor)), None)
        device_type = tensor.device.type if tensor is not None else "cpu"

        if device_type == "cuda":
            _collect_cuda_time()
        if state["eager_seconds"] >= min_eager_seconds:
            compiled = state["compiled"]
            if compiled is None:
                compiled = safe_compile(fn, **compile_kwargs)
                state["compiled"] = compiled
            return compiled(*args, **kwargs)

        if device_type == "cuda":
            state["cuda_unmeasured_calls"] += 1
            if state["pending_cuda"] is not None or state["cuda_unmeasured_calls"] < 64:
                return fn(*args, **kwargs)

            # Sparse sampling avoids doubling the launch bookkeeping of the tiny
            # reductions this gate protects. One timed call represents all eager
            # calls accumulated since the previous sample.
            represented_calls = state["cuda_unmeasured_calls"]
            state["cuda_unmeasured_calls"] = 0
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            result = fn(*args, **kwargs)
            end_event.record()
            state["pending_cuda"] = (start_event, end_event, represented_calls)
            return result

        if device_type == "mps":
            return fn(*args, **kwargs)

        start = time.perf_counter()
        result = fn(*args, **kwargs)
        state["eager_seconds"] += time.perf_counter() - start
        return result

    wrapper._ffs_eager = fn  # ty: ignore[unresolved-attribute]
    wrapper._ffs_compile_state = state  # ty: ignore[unresolved-attribute]
    return wrapper
