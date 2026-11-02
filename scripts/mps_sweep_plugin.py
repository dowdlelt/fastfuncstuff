"""pytest plugin: run the suite on Metal and name every site that breaks there.

A static scan cannot judge a dtype held in a variable, ``x.to(other)``, or a helper that
returns a CPU tensor to a caller holding Metal ones; those only show at run time. This
turns each test's own ``torch.device("cpu")`` and ``-device cpu`` into Metal (only from
frames in ``tests/``, so internal routing still works), makes the silent-zero casts
raise, and logs each failure as one JSON line keyed by its innermost package line::

    FFS_SWEEP_LOG=/tmp/sweep.jsonl PYTHONPATH=scripts \\
        pytest -p mps_sweep_plugin -m "gpu or not gpu" -n 4 -q

3-5 min on an M4 Max, and it occupies the GPU, so treat it like ``pytest -m gpu``. Plenty
of what it logs is a test artefact (a test passing ``dtype=float64`` to a helper, half its
tensors built on the CPU), so read the chain before fixing. ``{log}.ops.{pid}`` counts the
solver calls that reached Metal. See "MPS traps in torch 2.14" in the wiki.
"""

from __future__ import annotations

import atexit
import collections
import functools
import json
import os
import signal
import sys
import traceback

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

_REAL_DEVICE = torch.device
_MPS = _REAL_DEVICE("mps", 0)
_TESTS = os.sep + "tests" + os.sep
_PACKAGE = "/fastfuncstuff/fastfuncstuff/"
_WIDE = (torch.float64, torch.complex128)
_WIDE_TENSOR_TYPES = ("torch.DoubleTensor", torch.DoubleTensor)
_LOG = os.environ["FFS_SWEEP_LOG"]
_TIMEOUT_S = int(os.environ.get("FFS_SWEEP_TIMEOUT", "90"))


class _DeviceMeta(type):
    def __instancecheck__(cls, obj):
        return isinstance(obj, _REAL_DEVICE)


class _FlippedDevice(metaclass=_DeviceMeta):
    def __new__(cls, *args, **kwargs):
        if args and args[0] == "cpu" and _TESTS in sys._getframe(1).f_code.co_filename:
            return _MPS
        return _REAL_DEVICE(*args, **kwargs)


torch.device = _FlippedDevice


def _silent_zero(what: str) -> RuntimeError:
    return RuntimeError(f"sweep: silent-zero cast via {what}")


def _trap_to(original):
    @functools.wraps(original)
    def to(self, *args, **kwargs):
        if self.device.type == "mps":
            dtypes = [a for a in args if isinstance(a, torch.dtype)] + [kwargs.get("dtype")]
            devices = [a for a in args if isinstance(a, (str, _REAL_DEVICE))]
            devices.append(kwargs.get("device"))
            others = [a for a in args if isinstance(a, torch.Tensor)]
            named = any(d in _WIDE for d in dtypes) and any(d is not None for d in devices)
            if named or any(o.dtype in _WIDE for o in others):
                raise _silent_zero(".to()")
        return original(self, *args, **kwargs)

    return to


def _trap_type(original):
    @functools.wraps(original)
    def type_(self, *args, **kwargs):
        if self.device.type == "mps" and args and args[0] in _WIDE_TENSOR_TYPES:
            raise _silent_zero(".type()")
        return original(self, *args, **kwargs)

    return type_


def _trap_type_as(original):
    @functools.wraps(original)
    def type_as(self, other):
        if self.device.type == "mps" and other.dtype in _WIDE:
            raise _silent_zero(".type_as()")
        return original(self, other)

    return type_as


def _trap_constructor(original, name):
    @functools.wraps(original)
    def construct(data, *args, **kwargs):
        on_metal = isinstance(data, torch.Tensor) and data.device.type == "mps"
        if on_metal and kwargs.get("dtype") in _WIDE:
            raise _silent_zero(name)
        return original(data, *args, **kwargs)

    return construct


torch.Tensor.to = _trap_to(torch.Tensor.to)
torch.Tensor.type = _trap_type(torch.Tensor.type)
torch.Tensor.type_as = _trap_type_as(torch.Tensor.type_as)
torch.as_tensor = _trap_constructor(torch.as_tensor, "as_tensor")
torch.tensor = _trap_constructor(torch.tensor, "tensor")

import fastfuncstuff.cli_utils as _cli_utils  # noqa: E402
import fastfuncstuff.utils as _utils  # noqa: E402

_get_device = _utils.get_device
_parse_device_arg = _cli_utils.parse_device_arg


def _flipped_get_device(preferred=None):
    return _MPS if preferred in (None, "cpu") else _get_device(preferred)


def _flipped_parse_device_arg(spec):
    if spec is None or str(spec).split(",")[0].lower() in ("cpu", "auto"):
        return _MPS, None, None
    return _parse_device_arg(spec)


_utils.get_device = _flipped_get_device
_cli_utils.parse_device_arg = _flipped_parse_device_arg

_metal_ops: collections.Counter = collections.Counter()


def _package_site() -> str | None:
    frame = sys._getframe(2)
    while frame:
        name = frame.f_code.co_filename
        if _PACKAGE in name and not name.endswith(("/utils.py", "/policy_linalg.py")):
            return f"{name.split(_PACKAGE)[-1]}:{frame.f_lineno}"
        frame = frame.f_back
    return None


def _count_metal_calls(module, name):
    original = getattr(module, name)

    @functools.wraps(original)
    def counted(*args, **kwargs):
        if args and isinstance(args[0], torch.Tensor) and args[0].device.type == "mps":
            site = _package_site()
            if site:
                _metal_ops[(name, args[0].ndim, site)] += 1
        return original(*args, **kwargs)

    setattr(module, name, counted)


for _name in (
    "qr", "svd", "svdvals", "lstsq", "cholesky", "cholesky_ex", "solve", "solve_triangular",
    "pinv", "eigh", "eigvalsh", "inv", "inv_ex", "eig", "det", "slogdet", "matrix_rank",
    "lu_factor",
):  # fmt: skip
    _count_metal_calls(torch.linalg, _name)
for _name in ("cholesky_solve", "cholesky_inverse", "unique"):
    _count_metal_calls(torch, _name)


@atexit.register
def _dump_metal_ops() -> None:
    with open(f"{_LOG}.ops.{os.getpid()}", "w") as f:
        json.dump([[*key, count] for key, count in _metal_ops.items()], f)


def _timed_out(*_args):
    raise TimeoutError("sweep: test timed out")


def pytest_runtest_call(item):
    signal.signal(signal.SIGALRM, _timed_out)
    signal.alarm(_TIMEOUT_S)


def pytest_runtest_teardown(item):
    signal.alarm(0)
    torch.mps.empty_cache()


def pytest_runtest_makereport(item, call):
    if call.when != "call" or call.excinfo is None:
        return
    error = call.excinfo.value
    frames = [
        f
        for f in traceback.extract_tb(error.__traceback__)
        if _PACKAGE in f.filename and "__init__" not in f.filename
    ]
    chain = [f"{f.filename.split(_PACKAGE)[-1]}:{f.lineno}" for f in frames[-5:]]
    record = {
        "site": chain[-1] if chain else "(test)",
        "chain": chain,
        "test": item.nodeid,
        "type": type(error).__name__,
        "msg": str(error)[:150].replace("\n", " "),
        "line": frames[-1].line[:100] if frames else "",
    }
    with open(_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")
