"""
Utility functions for fastfuncstuff
Device management and helper functions
"""

from __future__ import annotations

import contextlib
import functools
import os
import threading
import warnings
from collections.abc import Generator, Iterator
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    pass

# Low-level I/O routines announce slow writes themselves so a bare library call
# doesn't look wedged. When a CLI already wraps the call in a spinner, that
# announcement duplicates the spinner's own line; the spinner claims the terminal
# for the duration and the writer stays quiet.
_io_progress_lock = threading.Lock()
_io_progress_depth = 0


def io_progress_suppressed() -> bool:
    """True while a caller-level progress display owns the terminal."""
    return _io_progress_depth > 0


@contextlib.contextmanager
def suppress_io_progress() -> Iterator[None]:
    """Silence low-level write/read progress messages inside the block."""
    global _io_progress_depth
    with _io_progress_lock:
        _io_progress_depth += 1
    try:
        yield
    finally:
        with _io_progress_lock:
            _io_progress_depth -= 1


# Indexed, because a tensor made on bare ``torch.device("mps")`` reports ``mps:0``
# and the two compare unequal. Every ``x.device == device`` in the tree (is this
# data resident? does this plan match?) would otherwise be false on a Mac: wrong
# branch at best, a refusal at worst. There is only ever one Metal device.
MPS_DEVICE = torch.device("mps", 0)


def get_device(prefer_device: str | None = None) -> torch.device:
    """
    Select the execution device.

    This is a CUDA-first codebase with a first-class CPU fallback; Apple Silicon
    (MPS) is the third supported backend. An explicitly requested backend is
    always honoured end-to-end — we never silently override the caller's choice.
    With no preference we auto-detect: CUDA, then MPS on Apple Silicon, then CPU.

    MPS used to be an explicit opt-in because picking it meant accepting that
    some ops would simply raise. That is no longer the trade: unimplemented ops
    fall back to the CPU automatically (PYTORCH_ENABLE_MPS_FALLBACK, set in
    ``fastfuncstuff/__init__.py``), and the handful of ops where Metal is broken
    or slower than the Mac CPU are routed off it by :func:`cpu_if_mps`. So a Mac
    user no longer has to know which of our tools happen to be Metal-friendly —
    the default is "use the GPU for what it is good at". We still decline MPS if
    the fallback did not arm in time (see :func:`mps_fallback_active`), since
    without it an unimplemented op raises in the middle of a long fit.

    Two things stay true on MPS. float64 is unsupported by Metal outright, so
    numerically sensitive steps run CPU-float64 via :func:`linalg_device` and
    reduction accumulators stay on-device in float32 (:func:`accum_dtype`).
    Users who want guaranteed full float64 precision on a Mac can pass
    ``-device cpu``, which remains fully supported.

    Parameters
    ----------
    prefer_device : str, optional
        Preferred device ('mps', 'cuda', 'cpu'). The specified backend must be
        available; otherwise a RuntimeError is raised.

    Returns
    -------
    device : torch.device
        The selected device.

    Raises
    ------
    RuntimeError
        If an explicitly requested backend is unavailable.
    """
    if prefer_device is not None:
        prefer_device = prefer_device.lower()
        if prefer_device == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError(
                    "MPS device requested but not available. Enable Apple Metal Performance "
                    "Shaders (macOS 13+ with Apple Silicon) before running."
                )
            return MPS_DEVICE
        if prefer_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA device requested but not available. Ensure NVIDIA drivers and CUDA are installed."
                )
            return torch.device("cuda")
        if prefer_device == "cpu":
            return torch.device("cpu")
        raise ValueError(
            f"Unknown prefer_device='{prefer_device}'. Expected 'mps', 'cuda', or 'cpu'."
        )

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        # Without the CPU fallback armed, an op Metal lacks raises rather than
        # falling back, and auto-detect must not hand the user a device that
        # dies partway through a fit. -device mps is still available to them.
        if not _torch_at_least(_MPS_MIN_TORCH):
            warnings.warn(
                f"MPS is available but torch {torch.__version__} predates "
                f"{_MPS_MIN_TORCH[0]}.{_MPS_MIN_TORCH[1]}, which this codebase's "
                "Metal policy is measured against. Using CPU; upgrade torch, or "
                "pass -device mps to override.",
                stacklevel=2,
            )
            return torch.device("cpu")
        if mps_fallback_active():
            return MPS_DEVICE
        warnings.warn(
            "MPS is available but the PyTorch CPU fallback did not arm (torch was "
            "imported before fastfuncstuff, or PYTORCH_ENABLE_MPS_FALLBACK=0). "
            "Using CPU; import fastfuncstuff before torch, or pass -device mps to "
            "override.",
            stacklevel=2,
        )
        return torch.device("cpu")

    warnings.warn(
        "No GPU backend detected; falling back to CPU execution. Performance will be limited.",
        stacklevel=2,
    )
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# MPS policy -- one table, two questions
# ---------------------------------------------------------------------------
# Apple Silicon is the default device on a Mac now (see get_device), so every
# op runs on Metal unless this table says otherwise. An op earns a row for one
# of exactly two measured reasons:
#
#   BROKEN  it raises on MPS. PYTORCH_ENABLE_MPS_FALLBACK -- set in
#           fastfuncstuff/__init__.py before torch is imported -- already
#           rescues the "not implemented for the MPS device" kind silently and
#           automatically. It does NOT rescue a kernel that *is* registered and
#           then fails inside Metal, which is how torch 2.14's batched SVD and
#           lstsq break. Those need a row.
#   SLOWER  it runs and is correct, but the CPU wins by enough to matter.
#
# Both are measured, never assumed: scripts/bench_mps_policy.py regenerates the
# numbers on the machine in front of you. Keeping them in one dict is the whole
# point -- when a torch release lands better kernels, re-run the script and
# delete a row, rather than grepping the tree for `device.type == "mps"`.
#
# float64 is deliberately not in this table. That is a dtype limitation, not an
# op one, and it has its own routing: linalg_device / accum_dtype / factor_device.
# The torch release the table below was measured against. Auto-detect declines
# MPS under this floor: on an older torch the native Metal linalg and 3-D
# grid_sample this policy assumes are absent, so both halves of the table --
# what we keep on Metal and what we route off it -- would be wrong. -device mps
# still works for anyone who wants to override that.
_MPS_MIN_TORCH = (2, 14)


def _torch_at_least(minimum: tuple[int, int]) -> bool:
    """True when the running torch is at least *minimum*, ignoring any suffix."""
    try:
        parts = torch.__version__.split("+", 1)[0].split(".")
        return (int(parts[0]), int(parts[1])) >= minimum
    except (ValueError, IndexError):
        # An unparseable version string (a nightly, a source build) is assumed
        # new enough -- refusing the GPU over a cosmetic string would be worse.
        return True


# Measured on an M4 Max (32 GPU cores), torch 2.14.0, quiet GPU. Ratios are
# CPU-float32 time over MPS time, so "12000x slower" means MPS took that much
# longer. Re-run scripts/bench_mps_policy.py after a torch upgrade.
_MPS_CPU_OPS: dict[str, str] = {
    # Unmeasured, routed off Metal conservatively: torch's CSR @ dense coverage on
    # MPS is patchy across releases, and the surface fold is one small product a
    # frame. Re-measure with scripts/bench_mps_policy.py before moving it.
    "sparse_csr_mm": "unmeasured: CSR @ dense support on Metal not verified (surface fold)",
    # 2.14's Jacobi QR is a catastrophe at the tall-skinny shapes every nuisance
    # projector in this codebase uses. Not a typo: a (2000,300) QR takes over
    # nine minutes on Metal against 5.8 ms on the CPU.
    "qr": "torch 2.14: (1000,60) 7166 ms vs 0.60 ms CPU (12000x slower)",
    # Registered but broken -- these abort inside the Metal shader compiler
    # ("Failed to created pipeline state object"), which is a runtime error
    # rather than a missing op, so PYTORCH_ENABLE_MPS_FALLBACK does not catch
    # it. They must be routed off MPS explicitly.
    "svd_batched": "torch 2.14: Metal pipeline-state compile failure (not rescued by the env fallback)",
    "lstsq": "torch 2.14: Metal pipeline-state compile failure (not rescued by the env fallback)",
    # These all work correctly on Metal now; they are simply slower than Apple's
    # CPU LAPACK at the matrix sizes we hand them (designs, folds, projectors).
    "svd": "torch 2.14: (1000,200) 6.2 ms vs 4.3 ms CPU (1.5x slower)",
    "cholesky": "torch 2.14: (500,500) 1.97 ms vs 0.20 ms CPU (10x slower)",
    "solve": "torch 2.14: (500,500) 2.45 ms vs 0.56 ms CPU (4.3x slower)",
    "pinv": "torch 2.14: (1000,150) 4.4 ms vs 3.0 ms CPU (1.5x slower)",
    "eigh": "torch 2.14: (500,500) 9.0 ms vs 7.9 ms CPU (1.15x slower); the "
    "*batched* case is 3.2x faster on MPS -- use op 'eigh_batched' for that",
    # The hand-rolled separable gather (wsinc5/cubic/quintic/heptic), not
    # grid_sample -- which stays on Metal and wins by 4-6x. This one is
    # launch-bound on Metal: ~105-120 ms whatever the output size, while the CPU
    # scales with the work and wins everywhere.
    "separable_resample": "torch 2.14: wsinc5 affine resample to (60,80,80) 580 ms on "
    "MPS vs 30 ms CPU (19x slower; Metal's best run of several was 105 ms, still 3.3x). "
    "Metal is launch-bound on it -- at its best it is flat ~110 ms from 0.1M to 12.6M "
    "output voxels -- while the CPU scales with the work and wins at every size",
    # The viewer's compositing path, and the only entry here that is about
    # *safety* before speed. Metal has no cross-thread story: two Python threads
    # in MPS at once abort the process outright with "failed assertion _status <
    # MTLCommandBufferStatusCommitted", not an exception anything can catch. The
    # viewer composites on the GUI thread while a worker runs a mode or a preproc
    # tool, so leaving the repaint on MPS makes every long-running tool a race
    # against the user's next click. Routing the repaint to the CPU leaves the
    # worker as the only thread on the GPU -- and it is faster anyway, because a
    # slice-sized array is all copy and no arithmetic.
    "viewer_compose": "torch 2.14: slice-sized host->device copies, 1.77 ms vs 1.29 ms "
    "CPU per 3-plane repaint (104x94x57) -- and Metal aborts outright if the GUI "
    "thread enters it while a worker is already inside it",
}

# Ops deliberately kept on Metal, recorded so a future reader does not "fix"
# them back onto the CPU. Measured in the same run as _MPS_CPU_OPS.
#
#   grid_sample 3-D   17.5x forward, 2.7x forward+backward -- new in 2.14, and
#                     the single biggest win for registration/warping. Before
#                     2.14 this fell back to the CPU and *lost* (0.68x).
#   eigh (batched)    3.2x on (200,64,64)
#   rfft              13x        elementwise   3.8x
#   matmul (large)    3.6x       conv3d        1.7x (was 1.03x in 2.8)
_MPS_KEEP_OPS: frozenset[str] = frozenset(
    {"eigh_batched", "grid_sample", "fft", "conv3d", "matmul", "reduce"}
)


def prefer_cpu_row_convolution() -> bool:
    """Whether per-row FIR convolution has a suitable CPU execution backend.

    M4 Max / torch 2.14: without oneDNN, eight-thread grouped convolution
    took 12.51s versus 0.86s for gather in a real-data motion estimate.
    Row convolution still won at one thread; preserve that route and enabled
    oneDNN rather than changing global thread counts around an operation.
    """
    return torch.get_num_threads() == 1 or (
        torch.backends.mkldnn.is_available() and bool(torch.backends.mkldnn.enabled)
    )


def cpu_if_mps(device: torch.device, op: str) -> torch.device:
    """Where to run *op*: the CPU when MPS would break or lose, else *device*.

    The one place the "does MPS do this well?" question gets answered. Call it
    instead of writing ``torch.device("cpu") if device.type == "mps" else device``
    inline -- that idiom is correct but scatters the policy across dozens of
    files, each with its own stale comment about what Metal could do in 2024.

    ``op`` must be a key of :data:`_MPS_CPU_OPS` or the literal ``"*"`` group
    names below; an unknown name raises rather than quietly leaving the work on
    MPS, because a typo that silently keeps a 7-second QR on Metal is exactly
    the bug this function exists to prevent.
    """
    if device.type != "mps":
        return device
    if op in _MPS_KEEP_OPS:
        return device
    if op not in _MPS_CPU_OPS:
        raise KeyError(
            f"cpu_if_mps: unknown op {op!r}. Add it to _MPS_CPU_OPS (with a "
            "measured reason) or _MPS_KEEP_OPS in fastfuncstuff/utils.py. "
            "Known: " + ", ".join(sorted(set(_MPS_CPU_OPS) | _MPS_KEEP_OPS))
        )
    return torch.device("cpu")


def shape_bucket(n: int, device: torch.device, limit: int | None = None) -> int:
    """Round a working-set size up so a shrinking batch reuses compiled kernels.

    Metal compiles a graph per distinct tensor shape and caches it, so a loop
    whose batch shrinks by a few rows each step pays a compile on every op of
    every step. torch 2.14, M4 Max, a matmul + bmm + pow + sum step:

        same shape every step     0.73 ms
        a new shape every step    3.04 ms   (4.2x)
        those shapes, repeated    0.83 ms

    On MPS this returns the next of four geometric steps per octave (at most
    25% padding, ~64 shapes across the whole size range, each compiled once
    per process). CUDA and the CPU have no per-shape cost, so they get ``n``
    back unchanged and callers keep their exact unpadded behaviour.
    ``limit`` caps the result, e.g. at the size of the set being drawn from.
    """
    if device.type != "mps" or n <= 0:
        return n
    bucket = 64
    while bucket < n:
        octave = 1 << (bucket.bit_length() - 1)
        bucket += octave // 4
    return bucket if limit is None else min(bucket, limit)


def mps_is_good_for(op: str) -> bool:
    """True when *op* should stay on Metal. The inverse view of :data:`_MPS_CPU_OPS`.

    Use it for the "should I even take this branch?" question -- picking a
    batched algorithm over a serial one, say -- where :func:`cpu_if_mps` would
    give you a device you do not need.
    """
    return op not in _MPS_CPU_OPS


@functools.lru_cache(maxsize=1)
def mps_fallback_active() -> bool:
    """True when an op Metal lacks silently falls back to the CPU.

    Asks the question by *doing* it rather than by inferring it from import
    order. PyTorch registers the fallback when the ATen MPS library loads, so
    ``import torch`` before ``import fastfuncstuff`` misses the window -- but so
    does nothing else, and an import-order proxy gets the common case backwards:
    a user whose shell already exports PYTORCH_ENABLE_MPS_FALLBACK=1 is armed no
    matter when torch was imported.

    ``linalg_eig`` is the canary: it has no Metal kernel, so with the fallback
    registered it quietly computes on the CPU and without it raises
    NotImplementedError. The environment variable is checked too, because a
    future torch that implements eig natively would otherwise make the probe
    succeed for the wrong reason.

    Cached -- the answer cannot change within a process.
    """
    if not torch.backends.mps.is_available():
        return False
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "") in ("", "0"):
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.linalg.eig(torch.eye(2, device="mps"))
    except NotImplementedError:
        return False
    except Exception:
        # Some other failure (a driver hiccup, an unrelated build problem) is
        # not evidence about the fallback; don't refuse the GPU over it.
        return True
    return True


def linalg_device(device: torch.device) -> torch.device:
    """Return the device to use for float64 / LAPACK-backed linear algebra.

    MPS has no float64 support, so numerically sensitive steps that require
    float64 (REML likelihood, Cholesky/SVD/pinv on near-singular or small
    matrices) run on CPU and have their result moved back to the original
    device. For CUDA and CPU this is a no-op. Centralising the pattern here
    keeps the float64-fallback policy in one place.
    """
    return torch.device("cpu") if device.type == "mps" else device


def factor_device(device: torch.device) -> torch.device:
    """Where to run a *small* float64 factorization: pinv, SVD, eigh, inv, Cholesky.

    A different question from :func:`linalg_device`, which asks where float64
    arithmetic is *possible*. This asks where a fold- or design-sized float64
    factorization is *fastest*, and on a consumer CUDA card the answer is the
    CPU: float64 runs at 1/64 rate there, and at these sizes cuSOLVER's cost is
    dominated by per-call latency rather than arithmetic. Measured on an
    RTX 5070 Ti, CUDA against CPU:

        pinv (21, 141, 141)   f64    794 ms  vs    44 ms   (18x)
        pinv (1, 153, 153)    f64   38.5 ms  vs   3.3 ms   (12x)
        pinv (1890, 462)      f64    216 ms  vs    50 ms   (4.3x)
        pinv (1, 2048, 2048)  f64   3041 ms  vs  1622 ms   (1.9x)

    The CPU won at every float64 size measured. Two exceptions worth knowing:
    a *large batch of tiny* matrices -- (100, 32, 32) went 5.6 ms on CUDA
    against 13.4 ms on CPU -- where the batch saturates the card and the
    transfer does not pay for itself; and float32, which is not affected at all
    (CUDA pinv stayed within 1.5x of CPU everywhere and won outright from
    N=512 up). Leave float32 factorizations where they are.

    Only use this when the result is small and reused -- a fold plan, a design
    solver, a projector -- never for anything carrying a voxel axis, where the
    transfer would dwarf what the factorization saves.
    """
    return torch.device("cpu") if device.type in ("cuda", "mps") else device


def linalg_lstsq(a: torch.Tensor, b: torch.Tensor, **kwargs):
    """``torch.linalg.lstsq`` that survives Metal, returning on the input device.

    torch 2.14's MPS lstsq aborts inside the Metal shader compiler for some
    shapes and not others -- (256,32) fails every time, (1000,60) succeeds every
    time -- and it surfaces as a RuntimeError, which is the one kind of failure
    PYTORCH_ENABLE_MPS_FALLBACK does *not* intercept. A call site cannot know
    which side of that line its matrix falls on, so on MPS every lstsq runs on
    the CPU and comes back. A no-op on CUDA and CPU.

    Same story for the batched SVD that lstsq drives internally; see
    :data:`_MPS_CPU_OPS`.
    """
    work = cpu_if_mps(a.device, "lstsq")
    if work == a.device:
        return torch.linalg.lstsq(a, b, **kwargs)
    result = torch.linalg.lstsq(a.to(work), b.to(work), **kwargs)
    # Rebuild the named tuple on the caller's device; .solution is what almost
    # every call site reads, but residuals/rank/singular_values must survive too.
    # torch's return_types are structseqs: they take one sequence, not *args.
    return type(result)([x.to(a.device) if isinstance(x, torch.Tensor) else x for x in result])


def pinv_f64(matrix: torch.Tensor) -> torch.Tensor:
    """Pseudo-inverse computed in float64 on :func:`factor_device`.

    Returned in the input's dtype and on the input's device, so this is a drop-in
    for ``torch.linalg.pinv(x.double()).to(x.dtype)`` that does not pay the
    consumer-GPU float64 penalty. For a symmetric positive semi-definite matrix
    prefer an ``eigh``-based inverse instead, which is the same answer for a
    fraction of the cost.
    """
    work = factor_device(matrix.device)
    # Device before dtype -- see to_linalg_f64.
    inverse = torch.linalg.pinv(matrix.to(device=work).to(torch.float64))
    return inverse.to(device=matrix.device, dtype=matrix.dtype)


def to_factor_f64(t: torch.Tensor) -> torch.Tensor:
    """Move a tensor to float64 on its :func:`factor_device`.

    The counterpart to :func:`to_linalg_f64` for the *small factorization* case:
    use this when the float64 tensor is about to be handed to ``svd``/``eigh``/
    ``inv``/``cholesky`` and the result is design- or component-sized. On a
    consumer CUDA card that lands the factorization on the CPU, where float64
    is 12-18x faster at these sizes (see :func:`factor_device`). The caller
    casts the result back, e.g. ``result.to(device=orig.device, dtype=orig.dtype)``.

    Never use it on anything carrying a voxel axis -- the transfer would dwarf
    what the factorization saves.
    """
    return t.to(device=factor_device(t.device)).to(torch.float64)


def widest_float_dtype(device: torch.device) -> torch.dtype:
    """The widest float *device* supports: float64, or float32 on Metal.

    For work that wants float64 and can live with float32 when Metal is the
    only option -- normal equations, closed-form solves, optimizer state,
    covariance eigendecompositions. Call it instead of writing
    ``torch.float32 if device.type == "mps" else torch.float64`` inline; that
    idiom is correct but scatters the float64 policy across the tree, which is
    the same mistake :func:`cpu_if_mps` exists to prevent for op routing.

    This answers "what dtype can I use?", not "should this run on MPS at all?"
    (:func:`cpu_if_mps`) nor "where does float64 linalg run?"
    (:func:`linalg_device` / :func:`factor_device`). Keep the comment at the
    call site explaining why float32 is tolerable for *that* operation -- this
    function cannot know that.
    """
    return torch.float32 if device.type == "mps" else torch.float64


def accum_dtype(device: torch.device) -> torch.dtype:
    """Return the dtype for reduction accumulators (sum-of-squares, R², RSS).

    On CUDA/CPU we accumulate in float64 to eliminate rounding in long
    sum-of-squares reductions. MPS cannot hold float64, so we accumulate in
    float32 there; callers should pair this with a numerically stable
    *two-pass* (mean-centred) reduction to avoid catastrophic cancellation.

    The value is :func:`widest_float_dtype`; this name carries the extra
    two-pass obligation that a plain solve or optimizer state does not have.
    """
    return widest_float_dtype(device)


def to_linalg_f64(t: torch.Tensor) -> torch.Tensor:
    """Move a tensor to float64 on its :func:`linalg_device`.

    Convenience for the common "promote to float64 for a sensitive linalg step"
    pattern. On MPS this lands on CPU (float64); elsewhere it stays put. The
    caller is responsible for casting the result back, e.g.
    ``result.to(orig.dtype).to(orig.device)``.

    Device is changed *before* dtype: a combined ``.to(cpu, float64)`` on an MPS
    tensor attempts the float64 cast on MPS first. Older torch raised; torch 2.14
    silently returns **zeros**, so the two steps must be ordered.
    """
    return t.to(device=linalg_device(t.device)).to(torch.float64)


_MPS_FLOAT32_WARNED = False


def warn_mps_float32_precision(context: str = "") -> None:
    """Emit a one-time warning that MPS uses float32 where CPU/CUDA use float64.

    MPS has no float64, so sum-of-squares accumulation runs in float32 there.
    With a numerically stable (mean-centred) reduction the precision loss is
    small, but we still tell the user once how to get guaranteed full precision.
    Deduplicated process-wide to avoid spamming long voxel-chunk loops.
    """
    global _MPS_FLOAT32_WARNED
    if _MPS_FLOAT32_WARNED:
        return
    _MPS_FLOAT32_WARNED = True
    suffix = f" ({context})" if context else ""
    warnings.warn(
        "MPS has no float64; accumulation uses float32" + suffix + ". This may slightly "
        "reduce sigma²/t-stat precision versus CPU/CUDA. Use -device cpu for full float64.",
        stacklevel=2,
    )


_MPS_CPU_FALLBACK_WARNED: set[str] = set()


def warn_mps_cpu_fallback(op: str = "") -> None:
    """One-time warning that a float64-only op falls back to CPU on MPS.

    Some computations (e.g. the Bayesian constrained-ridge FLOBS fit) are
    float64 end-to-end with no acceptable float32 variant. MPS has no float64,
    so we run the whole op on CPU there — correct and full-precision, just not
    GPU-accelerated. Deduplicated per ``op`` to avoid repeat spam.
    """
    if op in _MPS_CPU_FALLBACK_WARNED:
        return
    _MPS_CPU_FALLBACK_WARNED.add(op)
    label = op or "this float64 computation"
    warnings.warn(
        f"{label} is float64-only and has no MPS support; running on CPU. "
        "Use a CUDA GPU or -device cpu for the intended path.",
        stacklevel=2,
    )


def print_device_info(device: torch.device):
    """Print information about the device being used"""
    if device.type == "cuda":
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(device)}")
        print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
    elif device.type == "mps":
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
        print("Using CPU")


def to_tensor(
    x: torch.Tensor | np.ndarray | list | tuple,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
    pin: bool = False,
) -> torch.Tensor:
    """
    Convert input to torch tensor with specified dtype and device.

    Parameters
    ----------
    x : array-like or torch.Tensor
        Input data (numpy array, list, tuple, or torch.Tensor)
    dtype : torch.dtype
        Target dtype
    device : torch.device, optional
        Target device. If None, keep on current device
    pin : bool, default=False
        If True and transferring to a CUDA device, use pinned (page-locked)
        memory for the intermediate CPU tensor to speed up the transfer.

    Returns
    -------
    tensor : torch.Tensor
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=dtype)
    else:
        x = x.to(dtype=dtype)

    if device is not None:
        if pin and device.type == "cuda" and x.device.type == "cpu":
            x = x.pin_memory().to(device=device, non_blocking=True)
        else:
            x = x.to(device=device)

    return x


def resolve_cpu_threads(requested: int | None = None) -> tuple[int, str]:
    """How many CPU threads this process may use, and why.

    A shared server or batch scheduler expresses "use this much of the
    machine" through the environment, and a tool that calls
    ``os.cpu_count()`` ignores every one of those signals — it takes the whole
    box regardless of ``OMP_NUM_THREADS``, a cpuset, or a SLURM allocation.
    Precedence here, first hit wins:

    1. *requested* — an explicit flag (``-device cpu,N``).
    2. ``FFS_NUM_THREADS`` — ours, for when a site wants to cap only us.
    3. ``OMP_NUM_THREADS`` — the standard knob, which torch itself honours.
    4. ``SLURM_CPUS_PER_TASK`` — what the scheduler actually granted.
    5. CPU affinity (``sched_getaffinity``), which respects cpusets, taskset
       and containers, falling back to ``os.cpu_count()``.

    In case 5 only, and only when nothing has narrowed the affinity mask, the
    count drops to *physical* cores: hyperthread siblings share an FPU and
    rarely help dense linear algebra, which is most of what we run.

    Returns ``(n_threads, source)``; *source* is a short human-readable
    explanation for the startup banner.
    """
    if requested is not None and requested > 0:
        return max(1, int(requested)), "user-specified"

    for var in ("FFS_NUM_THREADS", "OMP_NUM_THREADS", "SLURM_CPUS_PER_TASK"):
        raw = os.environ.get(var)
        if raw:
            try:
                val = int(raw)
            except ValueError:
                continue
            if val > 0:
                return val, f"${var}"

    try:
        n_avail = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        n_avail = os.cpu_count() or 1

    n_logical = os.cpu_count() or n_avail
    if n_avail < n_logical:
        # Something restricted us (cpuset/taskset/container) — take it at
        # face value rather than second-guessing it with core topology.
        return max(1, n_avail), "CPU affinity"

    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        if physical and physical < n_logical:
            return max(1, physical), f"physical cores ({n_logical} logical)"
    except ImportError:
        pass
    return max(1, n_logical), "all CPUs"


REGISTRATION_TF32 = False
"""TF32 policy for the registration / resampling CLIs.

These tools are gather- and grid_sample-bound rather than GEMM-bound, so TF32
is expected to buy them little, while their cost functions (lpc/lpa and
friends) key on small float32 differences that a 10-bit mantissa could blur.
They are wired to this flag rather than a hardcoded ``False`` so the question
can be settled with numbers instead of argument: run ``ffs_benchmark`` on main,
flip this single constant to ``True`` on a branch, and re-run. If the
registration stages hold their thresholds and the timings improve, delete this
constant and let those CLIs take the default ``tf32=True``.

Does not affect the GLM-side CLIs, which already call
:func:`configure_torch_backends` with the default and want TF32.
"""


@contextlib.contextmanager
def full_fp32_matmul() -> Generator[None]:
    """Run the block with TF32 off, restoring the caller's matmul precision after.

    For geometry: a voxel coordinate in the hundreds keeps ~0.1 voxel through a
    TF32 (10-bit mantissa) matmul. Several solvers still switch TF32 on process-
    wide, so a transform applied later in the same process -- the -allineate
    rebuild, a batch -- inherited it: measured 0.7 median / 70 max intensity
    difference against the CPU on a T1->MNI chain. Also usable as a decorator.
    """
    try:
        prev = torch.get_float32_matmul_precision()
    except RuntimeError:
        # Once any code toggles the legacy allow_tf32 flags after
        # set_float32_matmul_precision() (the ICA and warp paths do), the getter
        # refuses to answer.  The legacy flags below still carry the state, so
        # leave the global setting alone rather than fail a REML search for it.
        prev = None
    prev_mm = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    if prev is not None:
        torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        if prev is not None:
            torch.set_float32_matmul_precision(prev)
        torch.backends.cuda.matmul.allow_tf32 = prev_mm
        torch.backends.cudnn.allow_tf32 = prev_cudnn


def configure_torch_backends(
    device: torch.device, n_threads: int | None = None, *, tf32: bool = True
) -> None:
    """Configure PyTorch backends for optimal performance.

    Call once at the start of a CLI entry-point after selecting the device.

    Sets:
      - float32 matmul precision to 'high' (use TF32 on Ampere+), when ``tf32``
      - cudnn.benchmark = True (autotuner for convolutions)
      - CPU thread count from :func:`resolve_cpu_threads` (for CPU and MPS
        fallback paths), which honours ``FFS_NUM_THREADS`` / ``OMP_NUM_THREADS``
        / the scheduler's allocation rather than seizing every core.
      - numpy's huge-page madvise off when the host's free 2 MB blocks are
        short and THP would compact synchronously
        (:func:`memory.configure_numpy_hugepages`).

    Pass ``tf32=False`` from the registration tools. They are gather- and
    grid_sample-bound rather than GEMM-bound, so TF32 buys them almost nothing,
    while their cost functions (lpc/lpa and friends) key on small float32
    differences that a 10-bit mantissa would blur. ``warp.py`` still opts into
    ``allow_tf32`` deliberately and narrowly where it does pay. The thread count
    is the part every CLI needs either way.
    """
    if tf32:
        torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        # Our convolutions are filters (box/Gaussian smoothing, local correlation), not
        # network layers, and PyTorch leaves cudnn.allow_tf32 ON by default -- independent
        # of the matmul precision above.  Whether cuDNN actually picks a TF32 algorithm is
        # its choice: torch 2.12.1 did not for these shapes, torch 2.14.1 / cuDNN 9.24 does,
        # and the 10-bit mantissa put ~1e-3 of error into every smoothed cost term.
        # warp.py re-enables it narrowly where it pays.
        torch.backends.cudnn.allow_tf32 = False
    # CPU is the linalg fallback under MPS and the primary device for
    # device=cpu, so the thread count matters even on a GPU run.
    from .memory import configure_numpy_hugepages

    configure_numpy_hugepages()
    n_cpu, _source = resolve_cpu_threads(n_threads)
    torch.set_num_threads(n_cpu)
    try:
        torch.set_num_interop_threads(min(4, n_cpu))
    except RuntimeError:
        # set_num_interop_threads can only be called once before any parallel
        # work starts. Tests/notebooks may call configure_torch_backends
        # multiple times; the second call is a no-op for interop threads.
        pass


def calc_memory_usage(shape: tuple, dtype: torch.dtype = torch.float32) -> float:
    """
    Calculate memory usage in GB for a tensor of given shape

    Parameters
    ----------
    shape : tuple
        Tensor shape
    dtype : torch.dtype
        Data type

    Returns
    -------
    memory_gb : float
        Memory usage in gigabytes
    """
    num_elements = 1
    for dim in shape:
        num_elements *= dim

    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    return (num_elements * bytes_per_element) / 1e9


def scale_to_percent_signal(
    data: torch.Tensor,
    run_starts: list[int],
    max_scale: float = 200.0,
    verbose: bool = True,
    track_violations: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
    """
    Scale voxel timeseries to mean=100 per run (percent signal change units).

    This is equivalent to AFNI's scaling: min(max_scale, a/b*100) where
    a is the timeseries and b is the mean of that timeseries.

    The max_scale (default 200) prevents extreme values - a voxel should
    never more than double its mean signal in physiologically plausible data.

    Parameters
    ----------
    data : torch.Tensor
        fMRI data (n_voxels, n_timepoints) - will be modified in-place
    run_starts : list of int
        Starting timepoint for each run
    max_scale : float, default=200.0
        Maximum allowed scaled value (clips to this)
    verbose : bool, default=True
        Print scaling statistics
    track_violations : bool, default=True
        If True, return the full ``(n_voxels, n_timepoints)`` boolean violations
        mask. If False, skip that allocation and return ``None`` for the mask —
        ``scale_info`` (counts, per-voxel indices) is still fully populated from
        cheap per-voxel accumulators. At whole-dataset scale the full mask is
        ~1 byte/sample (tens of GB), so callers that only need the counts should
        pass ``track_violations=False``.

    Returns
    -------
    data_scaled : torch.Tensor
        Scaled data (n_voxels, n_timepoints) with mean~100 per run
    violations_mask : torch.Tensor or None
        Boolean mask (n_voxels, n_timepoints) where values hit max_scale ceiling,
        or ``None`` when ``track_violations=False``
    scale_info : dict
        Statistics about the scaling:
        - 'n_violations': total number of timepoints that hit ceiling
        - 'n_voxels_with_violations': number of voxels with any violations
        - 'violation_voxel_indices': 1D tensor of voxel indices with violations
        - 'mean_per_run': (n_voxels, n_runs) mean before scaling
        - 'scale_factors': (n_voxels, n_runs) scale factors used (100/mean)

    Notes
    -----
    The scaling is: scaled = min(max_scale, raw / mean * 100)

    This converts raw signal to percent signal change units where:
    - Mean = 100 (by construction)
    - A value of 101 = 1% signal increase
    - A value of 99 = 1% signal decrease

    Violations (hitting max_scale) indicate potentially problematic voxels
    that may have:
    - Very low mean signal (near noise floor)
    - Motion spikes or other artifacts
    - Edge voxels with partial volume effects
    """
    n_voxels, n_timepoints_total = data.shape
    n_runs = len(run_starts)
    device = data.device

    # Compute run boundaries
    run_ends = run_starts[1:] + [n_timepoints_total]
    _run_lengths = [end - start for start, end in zip(run_starts, run_ends, strict=False)]

    # Storage for per-run statistics
    mean_per_run = torch.zeros(n_voxels, n_runs, device=device)
    scale_factors = torch.zeros(n_voxels, n_runs, device=device)

    # Track violations (keep on CPU to avoid GPU OOM). The full time-resolved
    # mask is 1 byte/sample -- tens of GB at whole-dataset scale -- so only
    # allocate it when the caller asks. Either way we accumulate the cheap
    # per-voxel count that scale_info actually reports.
    violations_mask = (
        torch.zeros(n_voxels, n_timepoints_total, dtype=torch.bool, device="cpu")
        if track_violations
        else None
    )
    viol_count_per_voxel = torch.zeros(n_voxels, dtype=torch.int64, device="cpu")

    if verbose:
        print("Scaling to percent signal change (mean=100 per run)...")

    for run_idx in range(n_runs):
        start = run_starts[run_idx]
        end = run_ends[run_idx]

        # Get this run's data
        run_data = data[:, start:end]  # (n_voxels, run_length)

        # Compute mean per voxel for this run
        run_mean = run_data.mean(dim=1, keepdim=True)  # (n_voxels, 1)
        mean_per_run[:, run_idx] = run_mean.squeeze()

        # Avoid division by zero (set scale factor to 0 for zero-mean voxels)
        # These voxels will become all zeros after scaling
        safe_mean = run_mean.clone()
        zero_mask = run_mean.abs() < 1e-10
        safe_mean[zero_mask] = 1.0  # Prevent div by zero

        # Compute scale factor: 100 / mean
        scale_factor = 100.0 / safe_mean  # (n_voxels, 1)
        scale_factors[:, run_idx] = scale_factor.squeeze()

        # Scale: a / b * 100 = a * scale_factor
        scaled_run = run_data * scale_factor  # (n_voxels, run_length)

        # Apply ceiling and track violations
        # Values above max_scale (e.g., 200) indicate >100% signal increase
        run_violations = (scaled_run > max_scale).cpu()
        if violations_mask is not None:
            violations_mask[:, start:end] = run_violations
        viol_count_per_voxel += run_violations.sum(dim=1).to(torch.int64)

        # Clip to max_scale (only upper bound - lower values are fine)
        # AFNI uses min(max_scale, scaled_value) - we preserve negative values
        # since fMRI can have signal decreases
        scaled_run = torch.clamp(scaled_run, max=max_scale)

        # Handle zero-mean voxels (set to 100 to avoid weird values)
        # Actually, set them to 0 since they're essentially dead voxels
        zero_voxels = zero_mask.squeeze()
        if zero_voxels.any():
            scaled_run[zero_voxels, :] = 0.0

        # Store back
        data[:, start:end] = scaled_run

    # Compute violation statistics from the per-voxel counts (works whether or
    # not the full mask was materialized).
    n_violations = int(viol_count_per_voxel.sum().item())
    voxels_with_violations = viol_count_per_voxel > 0
    n_voxels_with_violations = int(voxels_with_violations.sum().item())
    violation_voxel_indices = torch.where(voxels_with_violations)[0]

    scale_info = {
        "n_violations": int(n_violations),
        "n_voxels_with_violations": int(n_voxels_with_violations),
        "violation_voxel_indices": violation_voxel_indices,
        "mean_per_run": mean_per_run,
        "scale_factors": scale_factors,
    }

    if verbose:
        print(f"  Scaled {n_voxels:,} voxels × {n_runs} runs")
        if n_violations > 0:
            pct_violations = 100 * n_violations / (n_voxels * n_timepoints_total)
            print(
                f"  ⚠️  Ceiling violations (>{max_scale}): {n_violations:,} timepoints ({pct_violations:.4f}%)"
            )
            print(
                f"      Affecting {n_voxels_with_violations:,} voxels ({100 * n_voxels_with_violations / n_voxels:.2f}%)"
            )
        else:
            print(f"  ✓ No ceiling violations (all values ≤ {max_scale})")

    return data, violations_mask, scale_info


def gaussian_blur_3d(
    data: np.ndarray,
    fwhm_mm: float,
    voxel_sizes: tuple[float, float, float],
    device: torch.device | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Apply 3D Gaussian spatial smoothing to 4D fMRI data.

    Uses separable 1D convolutions along each spatial axis for efficiency.
    Can process on GPU for speed, chunking by timepoint if needed.

    Parameters
    ----------
    data : np.ndarray
        4D fMRI data (x, y, z, t) - will NOT be modified in place
    fwhm_mm : float
        Full-width at half-maximum of Gaussian kernel in millimeters
    voxel_sizes : tuple of float
        Voxel dimensions in mm (voxel_x, voxel_y, voxel_z)
    device : torch.device, optional
        Device for computation. If None, auto-detect GPU/CPU.
    verbose : bool, default=True
        Print progress information

    Returns
    -------
    data_blurred : np.ndarray
        Blurred 4D data (x, y, z, t), same shape as input

    Notes
    -----
    FWHM to sigma conversion: sigma = FWHM / (2 * sqrt(2 * ln(2))) ≈ FWHM / 2.355

    The kernel is computed in voxel units using the voxel sizes.
    For anisotropic voxels, the sigma differs in each dimension.

    Memory: streams blocks of timepoints, sized by ``memory.get_available_memory``
    for the input block plus two working copies, so the device never holds
    more than a budgeted slice of the run.

    Each block is blurred in the array's own layout (``stats.smooth3d.blur_axis``):
    no per-volume transpose, so a C-ordered ``(x, y, z, t)`` array and a
    nibabel F-ordered one cost the same.
    """
    from tqdm import tqdm

    from fastfuncstuff.memory import get_available_memory
    from fastfuncstuff.stats.smooth3d import blur_axis

    if device is None:
        device = get_device()

    if data.ndim != 4:
        raise ValueError(f"Expected 4D data (x, y, z, t), got shape {data.shape}")

    nx, ny, nz, nt = data.shape

    # Convert FWHM to sigma (FWHM = 2.355 * sigma)
    fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # ≈ 0.4247
    sigma_mm = fwhm_mm * fwhm_to_sigma

    # Convert sigma from mm to voxels for each dimension
    sigma_vox = [sigma_mm / vs for vs in voxel_sizes]

    if verbose:
        print(f"Gaussian blur: FWHM = {fwhm_mm:.1f} mm")
        print(
            f"  Voxel sizes: {voxel_sizes[0]:.2f} × {voxel_sizes[1]:.2f} × {voxel_sizes[2]:.2f} mm"
        )
        print(f"  Sigma (voxels): {sigma_vox[0]:.2f} × {sigma_vox[1]:.2f} × {sigma_vox[2]:.2f}")

    # 1D Gaussian kernels, 4 sigma each side; None skips an axis whose sigma
    # is too small to matter.
    kernels: list[torch.Tensor | None] = []
    for sigma in sigma_vox:
        if sigma < 0.1:
            kernels.append(None)
            continue
        radius = max(1, int(np.ceil(4 * sigma)))
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernels.append(kernel / kernel.sum())

    if verbose:
        kernel_sizes = [len(k) if k is not None else 1 for k in kernels]
        print(f"  Kernel sizes: {kernel_sizes[0]} × {kernel_sizes[1]} × {kernel_sizes[2]} voxels")

    # Stream slabs along whichever axis is slowest in memory, so each slab is
    # one contiguous span of both the input and the output: it uploads with no
    # host gather and lands with no strided write-back.  Cutting a C-ordered
    # (x, y, z, t) run along t -- the old way -- spent ~4 of 6 s of a 4.2M-voxel
    # x 246 blur in those two host copies and a download into fresh memory.
    # Cutting along x needs a halo of the x kernel's radius, whose outputs are
    # discarded; the result is the same bit for bit.
    if data.flags.c_contiguous:
        axis_cut = 0
    elif data.flags.f_contiguous:
        axis_cut = 3
    else:
        data = np.ascontiguousarray(data)
        axis_cut = 0
    data_blurred = np.empty_like(data)
    k_cut = kernels[axis_cut] if axis_cut < 3 else None
    halo = (len(k_cut) - 1) // 2 if k_cut is not None else 0
    n_cut = data.shape[axis_cut]
    slab_bytes = data.nbytes // max(n_cut, 1)
    # The slab on the device plus the two arrays each axis pass ping-pongs.
    chunk = max(1, min(n_cut, get_available_memory(device) // max(3 * slab_bytes, 1) - 2 * halo))
    starts = range(0, n_cut, chunk)
    for a0 in tqdm(
        starts, desc="  Blurring", unit="block", leave=True, disable=not verbose or len(starts) < 2
    ):
        a1 = min(a0 + chunk, n_cut)
        h0, h1 = max(0, a0 - halo), min(n_cut, a1 + halo)
        block = data[h0:h1] if axis_cut == 0 else data[..., h0:h1]
        if not block.flags.writeable:
            # torch refuses to wrap read-only memory (a memmap, a nibabel
            # proxy); a copy in the same layout is cheaper than a warning.
            block = block.copy(order="K")
        vol = torch.from_numpy(block).to(device=device, dtype=torch.float32)
        for axis, k in enumerate(kernels):
            if k is not None:
                vol = blur_axis(vol, k, axis)
        vol = vol.narrow(axis_cut, a0 - h0, a1 - a0)
        dest = data_blurred[a0:a1] if axis_cut == 0 else data_blurred[..., a0:a1]
        torch.from_numpy(dest).copy_(vol)
        del vol

    if verbose:
        print(f"  ✓ Blurred {nt} volumes")

    return data_blurred


# ============================================================================
# Dry run / synthetic data generation
# ============================================================================


def generate_synthetic_runs(
    first_run_data: torch.Tensor | None,
    n_runs_total: int,
    run_length: int,
    n_voxels: int | None = None,
    generator: torch.Generator | None = None,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Generate synthetic fMRI data for dry-run testing.

    For fast testing, generates random positive data without loading real BOLD data.
    Only the header info (shape, dimensions) is needed from the first run.

    Parameters
    ----------
    first_run_data : torch.Tensor, optional
        Real data from the first run. If None, all data is synthetic.
    n_runs_total : int
        Total number of runs to simulate
    run_length : int
        Number of timepoints per run
    n_voxels : int, optional
        Number of voxels. Required if first_run_data is None.
    generator : torch.Generator, optional
        Random number generator for reproducibility
    verbose : bool, default=True
        Print progress information

    Returns
    -------
    synthetic_data : torch.Tensor
        Combined data: first_run (if provided) + synthetic runs, shape (n_voxels, n_runs_total * run_length)

    Notes
    -----
    - Synthetic data is generated with random positive values (10-100 range)
    - Data is generated on CPU for speed
    - When first_run_data is None, ALL runs are synthetic (fastest mode)
    """
    if first_run_data is not None:
        n_voxels = first_run_data.shape[0]
        n_runs_to_generate = n_runs_total - 1
        use_first_run = True
    else:
        if n_voxels is None:
            raise ValueError("n_voxels must be provided when first_run_data is None")
        n_runs_to_generate = n_runs_total
        use_first_run = False

    if verbose:
        print("\n" + "=" * 70)
        print("🎭 DRY RUN MODE - Generating Synthetic Data")
        print("=" * 70)
        if first_run_data is not None:
            print(f"  Using real data from run 1: {first_run_data.shape}")
        else:
            print(f"  All-synthetic mode: {n_voxels:,} voxels, {n_runs_total} runs")
        print(f"  Generating {n_runs_to_generate} synthetic runs...")
        print(f"  Total shape will be: ({n_voxels:,}, {n_runs_total * run_length:,})")

    # Pre-allocate full data tensor on CPU
    total_tps = n_runs_total * run_length
    synthetic_data = torch.zeros((n_voxels, total_tps), dtype=torch.float32, device="cpu")

    # Copy first run data if provided
    if use_first_run:
        synthetic_data[:, :run_length] = first_run_data
        start_idx = 1
    else:
        start_idx = 0

    # ======================================================================
    # FAST PATH: Generate all random data at once, then distribute to runs
    # ======================================================================
    # Much faster than looping: single torch.randn call instead of N calls
    synthetic_tps = n_runs_to_generate * run_length
    if synthetic_tps > 0:
        # Generate all random data at once: (n_voxels, synthetic_tps)
        all_random = 50.0 + torch.randn((n_voxels, synthetic_tps), generator=generator) * 15.0

        # Clip to positive range
        all_random = torch.clamp(all_random, min=10.0, max=100.0)

        # Distribute to runs with progress bar
        try:
            from tqdm import tqdm

            if verbose:
                print()
            for run_idx in tqdm(
                range(n_runs_to_generate), desc="  Simulating runs", disable=not verbose
            ):
                run_number = run_idx + start_idx
                start_tp = run_number * run_length
                end_tp = start_tp + run_length
                # Slice from the pre-generated random data
                src_start = run_idx * run_length
                src_end = src_start + run_length
                synthetic_data[:, start_tp:end_tp] = all_random[:, src_start:src_end]
        except ImportError:
            # Fallback without tqdm
            for run_idx in range(n_runs_to_generate):
                run_number = run_idx + start_idx
                start_tp = run_number * run_length
                end_tp = start_tp + run_length
                src_start = run_idx * run_length
                src_end = src_start + run_length
                synthetic_data[:, start_tp:end_tp] = all_random[:, src_start:src_end]

                if verbose and (run_idx + 1) % 10 == 0:
                    print(f"  Generated {run_idx + 1}/{n_runs_to_generate} synthetic runs...")

    if verbose:
        print(f"  ✓ Synthetic data ready: {synthetic_data.shape}")
        print(f"  Memory: {synthetic_data.numel() * 4 / 1e9:.2f} GB (CPU)")

    return synthetic_data


def load_per_run_nuisance_files(
    prefix: str,
    n_runs: int,
    suffix: str = "_selected_PCs.txt",
    verbose: bool = False,
) -> list[np.ndarray | None]:
    """
    Load per-run nuisance/regressor files.

    This is a general function for loading nuisance files that are stored
    separately per run, with a naming pattern like:
        {prefix}_run01{suffix}
        {prefix}_run02{suffix}
        ...

    Files can be empty (no regressors for that run), and different runs
    can have different numbers of regressors.

    Parameters
    ----------
    prefix : str
        Prefix for the per-run files (e.g., "dncond_w_condhrf")
    n_runs : int
        Number of runs (will load run01 through run{n_runs:02d})
    suffix : str, default="_selected_PCs.txt"
        Suffix for each file (includes extension)
    verbose : bool, default=False
        Print loading information

    Returns
    -------
    nuisance_per_run : list of ndarray or None
        List of per-run nuisance matrices, each (n_timepoints_run, n_regressors_run)
        None for runs with empty or missing files (no regressors)

    Examples
    --------
    Load per-run noise PCs:
        >>> pcs = load_per_run_nuisance_files(
        ...     "dncond_w_condhrf",
        ...     n_runs=17,
        ...     suffix="_selected_PCs.txt"
        ... )
    """
    from pathlib import Path

    nuisance_per_run = []

    for run_idx in range(1, n_runs + 1):
        run_file = f"{prefix}_run{run_idx:02d}{suffix}"

        try:
            # Check if file exists first
            if not Path(run_file).exists():
                if verbose:
                    print(f"  Run {run_idx:02d}: No file ({run_file})")
                nuisance_per_run.append(None)
                continue

            # Load data for this run
            data = np.loadtxt(run_file)

            # Handle empty files (0 rows or 0 cols)
            if data.size == 0:
                if verbose:
                    print(f"  Run {run_idx:02d}: Empty file")
                nuisance_per_run.append(None)
                continue

            # Handle 1D arrays (single regressor) - convert to 2D column vector
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            n_tps, n_regs = data.shape
            if verbose:
                print(f"  Run {run_idx:02d}: Loaded {n_regs} regressor(s), {n_tps} timepoints")

            nuisance_per_run.append(data)

        except (OSError, ValueError) as e:
            raise RuntimeError(f"Error loading nuisance file '{run_file}': {e}") from e

    return nuisance_per_run


def compute_power_spectrum(
    signal: np.ndarray | torch.Tensor,
    sampling_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the one-sided amplitude spectrum of a 1-D signal.

    Uses a real FFT and returns the positive-frequency half only (DC through
    Nyquist).  Amplitude is normalised so that a pure sine at frequency f
    with amplitude A gives a spectral peak of A at f.

    Parameters
    ----------
    signal : array-like, shape (n_samples,)
        The input time series.  Must be 1-D.
    sampling_rate : float
        Samples per second (Hz).  Determines the frequency axis.

    Returns
    -------
    freqs : np.ndarray, shape (n_freqs,)
        Frequency axis in Hz.  n_freqs = n_samples // 2 + 1.
    amplitudes : np.ndarray, shape (n_freqs,)
        One-sided amplitude spectrum.  Units match the input signal.

    Notes
    -----
    The two-sided FFT amplitude is halved for all bins except DC (index 0)
    and the Nyquist bin (index n//2 when n is even) so that the one-sided
    spectrum has the same total energy as the two-sided spectrum.
    """
    if isinstance(signal, torch.Tensor):
        signal = signal.detach().cpu().numpy()
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError(
            f"signal must be 1-D, got shape {signal.shape}. "
            "For multiple signals use compute_power_spectra()."
        )

    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    amplitudes = np.abs(fft_vals) / n

    # Double all non-DC, non-Nyquist bins to account for the missing
    # negative-frequency mirror (one-sided → two-sided energy equivalence).
    amplitudes[1:] *= 2
    if n % 2 == 0:
        amplitudes[-1] /= 2  # Nyquist bin is its own mirror — don't double it

    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    return freqs, amplitudes


def compute_power_spectra(
    signals: np.ndarray | torch.Tensor,
    sampling_rate: float,
    axis: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute one-sided amplitude spectra for a batch of signals.

    Applies :func:`compute_power_spectrum` logic across the specified axis.
    All signals must have the same length along that axis.

    Parameters
    ----------
    signals : array-like, shape (..., n_samples, ...)
        One or more time series.  The time axis is given by ``axis``.
    sampling_rate : float
        Samples per second (Hz).
    axis : int, default=-1
        Which axis contains the time samples.

    Returns
    -------
    freqs : np.ndarray, shape (n_freqs,)
        Frequency axis in Hz, shared across all signals.
    amplitudes : np.ndarray, same shape as ``signals`` except along ``axis``
        One-sided amplitude spectra.  Shape along ``axis`` is
        ``n_samples // 2 + 1``.

    Examples
    --------
    Spectra for all voxels in an (n_voxels, n_timepoints) matrix:

    >>> freqs, amps = compute_power_spectra(data, sampling_rate=1/tr)
    >>> peak_freq_per_voxel = freqs[amps.argmax(axis=-1)]
    """
    if isinstance(signals, torch.Tensor):
        signals = signals.detach().cpu().numpy()
    signals = np.asarray(signals, dtype=float)

    n = signals.shape[axis]
    fft_vals = np.fft.rfft(signals, axis=axis)
    amplitudes = np.abs(fft_vals) / n

    # Build a broadcastable multiplier: 2× everywhere except DC and Nyquist
    n_freqs = n // 2 + 1
    multiplier = np.full(n_freqs, 2.0)
    multiplier[0] = 1.0  # DC — no mirror
    if n % 2 == 0:
        multiplier[-1] = 1.0  # Nyquist — its own mirror

    # Reshape multiplier to broadcast along `axis`
    shape = [1] * signals.ndim
    shape[axis] = n_freqs
    amplitudes *= multiplier.reshape(shape)

    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    return freqs, amplitudes


def parabolic_peak_offset(
    curve: torch.Tensor,
    peak_idx: torch.Tensor,
    *,
    dim: int = 0,
) -> torch.Tensor:
    """Sub-grid peak offset by a 5-point least-squares parabola.

    A grid search reports the peak quantised to the grid.  The underlying
    objective is smooth, so fitting a parabola through the samples around
    the argmax and taking its vertex recovers the peak to a fraction of a
    step — the standard cross-correlation sub-sample trick.

    Five points via least squares rather than the 3-point analytic vertex:
    the 3-point vertex uses only the immediate neighbours and so inherits
    their noise directly, while the 5-point fit averages over a wider
    window.  Falls back to 3 points one sample from an edge, and to no
    refinement at the boundary (where a vertex would be an extrapolation
    outside the searched range).

    Formulas are shared with the ``locomoco`` xcorr searchlights, which
    compute the same vertex from a streaming running-peak because they do
    not hold the full curve.

    Parameters
    ----------
    curve : Tensor
        Objective sampled on a regular grid, peak along ``dim``.
    peak_idx : Tensor
        ``argmax(curve, dim=dim)``; shape is ``curve``'s shape minus ``dim``.

    Returns
    -------
    Tensor
        Offset in units of ONE GRID STEP, clamped to [-1, 1], same shape as
        ``peak_idx``.  Add ``offset * step`` to the grid-valued peak.
    """
    n = curve.shape[dim]
    c = curve.movedim(dim, 0)
    idx = torch.stack(
        [(peak_idx + k).clamp(0, n - 1) for k in (-2, -1, 0, 1, 2)], dim=0
    )  # (5, ...)
    y = c.gather(0, idx)
    ym2, ym1, y0, yp1, yp2 = y[0], y[1], y[2], y[3], y[4]

    b5 = (-2.0 * ym2 - ym1 + yp1 + 2.0 * yp2) / 10.0
    a5 = (5.0 * (4.0 * ym2 + ym1 + yp1 + 4.0 * yp2) - 10.0 * (ym2 + ym1 + y0 + yp1 + yp2)) / 70.0
    vtx5 = torch.where(a5.abs() > 1e-9, -b5 / (2.0 * a5), torch.zeros_like(a5)).clamp(-1.0, 1.0)
    den3 = ym1 - 2.0 * y0 + yp1
    vtx3 = torch.where(den3.abs() > 1e-6, 0.5 * (ym1 - yp1) / den3, torch.zeros_like(den3)).clamp(
        -1.0, 1.0
    )
    can5 = (peak_idx >= 2) & (peak_idx <= n - 3)
    can3 = (peak_idx >= 1) & (peak_idx <= n - 2)
    return torch.where(can5, vtx5, torch.where(can3, vtx3, torch.zeros_like(vtx5)))


def _prefers_cuda_batching(device: torch.device) -> bool:
    """Whether measured CUDA-only batching specializations should be selected."""
    return device.type == "cuda"


def quantiles(v: torch.Tensor, qs: tuple[float, ...] | list[float]) -> list[float]:
    """Quantiles of a tensor (flattened, nearest rank), with no size ceiling.

    ``torch.quantile`` refuses inputs beyond 2**24 elements ("input tensor is
    too large"), which an unmasked warp field on a padded grid or a
    whole-head anatomical exceeds. Sorting has no such limit and is a few
    milliseconds on the GPU at that size.
    """
    s, _ = v.float().flatten().sort()
    n = s.numel()
    if n == 0:
        raise ValueError("quantiles of an empty tensor")
    idx = torch.tensor([min(n - 1, max(0, int(round(q * (n - 1))))) for q in qs], device=s.device)
    return s[idx].tolist()
