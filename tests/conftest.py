"""
Pytest configuration for fastfuncstuff tests.
"""

import os

# ONE CPU thread per test process, and the parallelism comes from xdist workers
# instead (see addopts). Measured 2026-10-20: these tests barely thread-scale --
# test_blipflip.py is 40 s on 4 threads and 50 s on 1 -- while `-n 4` with the old
# 4-thread cap put 16 threads on 6 cores and made the whole suite SLOWER than
# serial (47 min vs 39 min). Four processes at one thread each beat one process
# at four threads by far more than that 20%.
#
# One thread is also the only setting that makes the suite's floats reproducible:
# parallel reductions sum in whatever order the pool finishes, which moves results
# at ~1e-7 and is enough to flip a convergence branch. test_blipflip.py's
# test_level_stops_when_the_field_stops_moving failed exactly that way under the
# old cap (reported 'flat cost' one level early) and passed when run alone.
#
# The env vars also cover numpy/BLAS and are inherited by spawned worker processes.
# FFS_TEST_NUM_THREADS is the one explicit escape hatch -- raising it trades that
# reproducibility back, so a test that only passes above 1 thread is a bug in the
# test, not a reason to raise the cap.
try:
    _TEST_THREADS = max(1, int(os.environ.get("FFS_TEST_NUM_THREADS", "1")))
except ValueError:
    _TEST_THREADS = 1
_TEST_THREADS = min(_TEST_THREADS, os.cpu_count() or 1)
for _var in (
    # FFS_NUM_THREADS FIRST in spirit: utils.thread_budget() checks it BEFORE
    # OMP_NUM_THREADS, so capping only the standard knob left every ffs thread
    # pool (I/O loaders, chunked compute) running at whatever the developer's
    # shell exported. Measured 2026-10-21 with FFS_NUM_THREADS=6 in the
    # environment: each xdist worker burned 278% CPU with 8-9 threads, four
    # workers saturated all 12 CPUs, and --durations reported a 20 s test as
    # 567 s. A cap that any ambient variable can beat is not a cap.
    "FFS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_var] = str(_TEST_THREADS)

import pytest  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(_TEST_THREADS)
torch.set_num_interop_threads(min(2, _TEST_THREADS))

# Bump dynamo recompile cache size: many tests trigger compiled kernels
# (e.g. glm.xval._cod_kernel_compiled) with varying shapes, dtypes, devices,
# and grad-mode states. The default limit (8) is fine for production but
# fills up across a full test run and turns into FailOnRecompileLimitHit
# pollution in unrelated downstream tests.
try:
    import torch._dynamo

    torch._dynamo.config.cache_size_limit = 256
    torch._dynamo.config.accumulated_cache_size_limit = 1024
except Exception:
    pass


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line(
        "markers", "benchmark_validation: benchmark validation tests (need existing outputs)"
    )
    config.addinivalue_line(
        "markers", "benchmark_full: full benchmark execution tests (need AFNI + data)"
    )


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """A gpu test that names CUDA has nothing to check on a machine without it.

    ``gpu`` means "lets a device default happen", and on a Mac that is Metal. Most of
    that set is meaningful there, so it is worth running; the tests that spell out
    ``.cuda()`` skip here rather than each carrying its own availability check.
    """
    try:
        return (yield)
    except AssertionError as exc:
        if item.get_closest_marker("gpu") and "not compiled with CUDA" in str(exc):
            pytest.skip("needs CUDA")
        raise


@pytest.fixture(scope="session")
def device():
    """Ordinary tests are CPU-only; GPU tests must opt in with @pytest.mark.gpu."""
    return torch.device("cpu")


@pytest.fixture(autouse=True)
def cpu_backend_by_default(monkeypatch, request):
    """Prevent unmarked tests and library defaults from discovering accelerators."""
    if request.node.get_closest_marker("gpu") is None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


@pytest.fixture(scope="session")
def _surface_cache_root(tmp_path_factory):
    return tmp_path_factory.mktemp("surface_cache")


@pytest.fixture(autouse=True)
def surface_cache_in_tmp(monkeypatch, _surface_cache_root):
    """Keep surface/diskcache entries out of the real ~/.cache.

    Patched rather than via XDG_CACHE_HOME, which also roots the Triton and
    compile-cost caches that GPU tests want warm.
    """
    from fastfuncstuff.surface import diskcache

    monkeypatch.setattr(diskcache, "_dir", lambda kind: _surface_cache_root / kind)


@pytest.fixture(autouse=True)
def reset_random_seed(request):
    """Reset only the RNGs that the selected test is allowed to use."""
    torch.manual_seed(42)
    if request.node.get_closest_marker("gpu") is not None and torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture(autouse=True)
def isolate_float32_matmul_precision():
    """Contain the global float32 matmul precision within each test.

    CLI entry points call ``utils.configure_torch_backends``, which runs
    ``torch.set_float32_matmul_precision("high")`` — a good default that
    enables TF32 on CUDA for GLM/registration-style workloads. But it is a
    *global, persistent* switch: once a CLI-exercising test flips it, every
    later test inherits TF32's ~1e-3 matmul error. Precision-sensitive tests
    (e.g. permutation t-stats validated against scipy at 1e-4) then pass or
    fail purely on collection order. Snapshot before / restore after so no
    test leaks the setting into the next; the chain keeps every test at the
    import-time default unless it opts in explicitly.
    """
    prev = torch.get_float32_matmul_precision()
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev)


@pytest.fixture(autouse=True)
def isolate_cudnn_tf32():
    """Run every test with cuDNN TF32 off, as the CLIs do (``configure_torch_backends``).

    PyTorch leaves ``cudnn.allow_tf32`` True, and from torch 2.14.1 / cuDNN 9.24 the
    float32 convolutions behind our smoothing and cost terms really use it, so a library
    test that never calls ``configure_torch_backends`` would measure TF32 noise (~1e-3)
    instead of the algorithm.  Restored afterwards so no test leaks the flag.
    """
    prev = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = prev
