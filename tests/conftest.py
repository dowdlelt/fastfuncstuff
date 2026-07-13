"""
Pytest configuration for fastfuncstuff tests.
"""

import os

import pytest
import torch

# Cap CPU threads for the whole test session. Test problems are tiny, so torch's
# default intra-op pool (one thread per core) mostly spins up threads to do
# near-nothing: it momentarily pegs every core without speeding the suite up, and
# it oversubscribes badly when a test spawns its own worker pool. A small fixed cap
# keeps the suite well-behaved (and reduction order deterministic across machines)
# without slowing the little real work there is. The env vars also cover numpy/BLAS
# and are inherited by spawned worker processes; setdefault respects an explicit
# override from the caller.
_TEST_THREADS = str(min(4, os.cpu_count() or 1))
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, _TEST_THREADS)
torch.set_num_threads(int(_TEST_THREADS))

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


@pytest.fixture(scope="session")
def device():
    """Provide device for all tests."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
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
