"""The CPU batch split in ``policy_linalg`` must change nothing but the speed.

ATen runs batched LAPACK work largely one matrix at a time, so ``_routed`` deals a
big CPU batch across Python threads. Every matrix still gets the same LAPACK call,
so these tests pin *bit-identity* against the unsplit call, for every routed op and
for each operand shape the split has to mirror (batched, vector-batched,
broadcast, several batch dims) -- and check that it really did split, so a test
cannot pass because the split quietly declined.
"""

from __future__ import annotations

import pytest
import torch

import fastfuncstuff.policy_linalg as P


@pytest.fixture(autouse=True)
def _split_everything(monkeypatch):
    # The suite pins workers to one thread, which would make every call decline.
    # The lane count is torch.get_num_threads(); report four. Thresholds to zero
    # so test-sized batches split.
    monkeypatch.setattr(torch, "get_num_threads", lambda: 4)
    monkeypatch.setattr(P, "_SPLIT_MIN_WORK", {})
    monkeypatch.setattr(P, "_SPLIT_MIN_WORK_DEFAULT", 0)
    monkeypatch.delenv("FFS_LINALG_SPLIT", raising=False)


@pytest.fixture
def splits(monkeypatch):
    """Count calls that actually went through the lane pool."""
    calls = []
    real = P._lane_pool

    def counting(lanes):
        calls.append(lanes)
        return real(lanes)

    monkeypatch.setattr(P, "_lane_pool", counting)
    return calls


def _unsplit(fn, monkeypatch):
    monkeypatch.setenv("FFS_LINALG_SPLIT", "0")
    try:
        return fn()
    finally:
        monkeypatch.delenv("FFS_LINALG_SPLIT")


def _identical(a, b):
    if isinstance(a, torch.Tensor):
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    return all(_identical(x, y) for x, y in zip(a, b, strict=True))


def _spd(batch, k, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(*batch, k, k, generator=g, dtype=dtype)
    return x @ x.mT + k * torch.eye(k, dtype=dtype)


def _rhs(shape, dtype=torch.float64, seed=1):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=dtype)


A = _spd((50,), 6)
L = torch.linalg.cholesky(A)
CASES = {
    "solve": lambda: P.solve(A, _rhs((50, 6, 2))),
    "solve_vector_batch": lambda: P.solve(A, _rhs((50, 6))),
    "solve_broadcast_rhs": lambda: P.solve(A, _rhs((6, 2))),
    "solve_left_false": lambda: P.solve(A, _rhs((50, 2, 6)), left=False),
    "solve_triangular": lambda: P.solve_triangular(L, _rhs((50, 6, 3)), upper=False),
    "cholesky": lambda: P.cholesky(A),
    "cholesky_ex": lambda: P.cholesky_ex(A),
    "cholesky_solve": lambda: P.cholesky_solve(_rhs((50, 6, 2)), L),
    "cholesky_inverse": lambda: P.cholesky_inverse(L),
    "inv": lambda: P.inv(A),
    "inv_ex": lambda: P.inv_ex(A),
    "det": lambda: P.det(A),
    "slogdet": lambda: P.slogdet(A),
    "logdet": lambda: P.logdet(A),
    "eigh": lambda: P.eigh(A),
    "eigh_upper": lambda: P.eigh(A, UPLO="U"),
    "eigvalsh": lambda: P.eigvalsh(A),
    "svd": lambda: P.svd(A, full_matrices=False),
    "svdvals": lambda: P.svdvals(A),
    "matrix_rank": lambda: P.matrix_rank(A),
    "cond": lambda: P.cond(A),
    "pinv": lambda: P.pinv(_rhs((50, 8, 5))),
    "qr": lambda: P.qr(_rhs((50, 9, 4))),
    "qr_complete": lambda: P.qr(_rhs((50, 9, 4)), mode="complete"),
    "lu_factor": lambda: P.lu_factor(A),
    "lu_solve": lambda: P.lu_solve(*P.lu_factor(A), _rhs((50, 6, 2))),
    "lstsq_gelsy": lambda: P.lstsq(_rhs((50, 12, 4)), _rhs((50, 12, 2), seed=3)),
    "lstsq_gelsd": lambda: P.lstsq(_rhs((50, 12, 4)), _rhs((50, 12, 2), seed=3), driver="gelsd"),
    "several_batch_dims": lambda: P.eigh(_spd((5, 10), 6)),
    "float32": lambda: P.solve(_spd((50,), 6, torch.float32), _rhs((50, 6, 1), torch.float32)),
}


@pytest.mark.parametrize("name", CASES)
def test_split_is_bit_identical(name, splits, monkeypatch):
    ref = _unsplit(CASES[name], monkeypatch)
    assert splits == []
    got = CASES[name]()
    assert splits, f"{name}: expected the batch to be split"
    assert _identical(ref, got), name


def test_one_rhs_is_shared_not_sliced(splits):
    # b of shape (k, 1) against a batch whose size equals k: slicing b by batch
    # would be shape-valid and silently wrong.
    a = _spd((6,), 6)
    b = _rhs((6, 1))
    want = torch.stack([torch.linalg.solve(a[i], b) for i in range(6)])
    # 6 matrices / 4 lanes is under the per-lane floor; lower it for this case.
    P_min = P._MIN_PER_LANE
    try:
        P._MIN_PER_LANE = 1
        got = P.solve(a, b)
    finally:
        P._MIN_PER_LANE = P_min
    assert splits
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)


def test_declines_under_autograd(splits):
    a = A.clone().requires_grad_(True)
    P.inv(a).sum().backward()
    assert splits == []
    assert a.grad is not None


def test_runs_with_grad_inputs_under_no_grad(splits):
    a = A.clone().requires_grad_(True)
    with torch.no_grad():
        P.inv(a)
    assert splits


def test_declines_for_small_batches_and_one_thread(splits, monkeypatch):
    P.inv(_spd((3,), 6))  # fewer matrices than lanes x _MIN_PER_LANE
    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    P.inv(A)
    assert splits == []


def test_declines_below_the_work_threshold(splits, monkeypatch):
    monkeypatch.setattr(P, "_SPLIT_MIN_WORK_DEFAULT", 1e12)
    P.inv(A)
    assert splits == []


def test_unique_rows_is_never_split(splits):
    x = torch.tensor([[1, 2], [1, 2], [3, 4]] * 40).reshape(40, 3, 2)
    P.unique_rows(x)
    assert splits == []


def test_singular_batch_raises_like_the_unsplit_call(monkeypatch):
    a = A.clone()
    a[37] = 0.0
    with pytest.raises(torch.linalg.LinAlgError) as unsplit:
        _unsplit(lambda: P.solve(a, _rhs((50, 6, 1))), monkeypatch)
    with pytest.raises(torch.linalg.LinAlgError) as split:
        P.solve(a, _rhs((50, 6, 1)))
    # The retry runs the whole batch unsplit, so the index is the caller's.
    assert str(split.value) == str(unsplit.value)


def test_inference_mode_inputs(splits, monkeypatch):
    with torch.inference_mode():
        a = _spd((50,), 6)
        ref = _unsplit(lambda: P.eigh(a), monkeypatch)
        got = P.eigh(a)
    assert splits
    assert _identical(ref, got)


def test_a_forked_child_builds_its_own_pool(monkeypatch):
    first = P._lane_pool(4)
    monkeypatch.setattr(P.os, "getpid", lambda: -12345)
    assert P._lane_pool(4) is not first


def test_lanes_do_not_open_their_own_thread_teams(monkeypatch):
    # Each lane's ATen/MKL call used to open a full OpenMP team: six lanes on six
    # cores peaked at 36 running threads, which a container CPU quota punishes by
    # freezing the process. A lane must report one OpenMP thread of its own.
    import ctypes

    setters = P._thread_local_setters()
    if not setters:
        pytest.skip("no OpenMP runtime found in /proc/self/maps")
    import re

    with open("/proc/self/maps") as fh:
        path = sorted(set(re.findall(r"(/\S*lib(?:gomp|iomp5|omp)[\w.-]*\.so[\w.]*)", fh.read())))[
            0
        ]
    get_max = ctypes.CDLL(path).omp_get_max_threads

    def after_real_work():
        # ATen resets a thread's count on its first parallel op, so a fresh
        # lane reading 1 proves nothing: check after the lane has done work.
        torch.empty(1 << 20).copy_(torch.ones(1 << 20))
        return get_max()

    # The suite pins the real count to 1, which would make ATen's reset harmless;
    # set a real 4 and build a fresh pool so every lane initialises under it.
    real = torch._C.get_num_threads()
    torch.set_num_threads(4)
    monkeypatch.setattr(P.os, "getpid", lambda: -54321)
    try:
        pool = P._lane_pool(4)
        got = list(pool.map(lambda _: after_real_work(), range(8)))
    finally:
        torch.set_num_threads(real)
    assert got == [1] * 8


@pytest.mark.parametrize("op", ["solve", "inv", "det", "slogdet", "lu_factor"])
def test_batched_lu_past_the_hang_size_always_takes_the_lanes(op, splits, monkeypatch):
    # torch 2.14.1's multi-threaded batched getrf hangs once k > 128, even for a
    # batch of two. The guard must engage below every speed threshold and with
    # the speed switch off.
    monkeypatch.setattr(P, "_SPLIT_MIN_WORK_DEFAULT", 1e30)
    monkeypatch.setenv("FFS_LINALG_SPLIT", "0")
    a = _spd((2,), P._LU_HANG_K + 8)
    b = _rhs((2, P._LU_HANG_K + 8, 1))
    call = {
        "solve": lambda: P.solve(a, b),
        "inv": lambda: P.inv(a),
        "det": lambda: P.det(a),
        "slogdet": lambda: P.slogdet(a),
        "lu_factor": lambda: P.lu_factor(a),
    }[op]
    got = call()
    assert splits
    torch.set_num_threads(1)  # the real thread count; one thread cannot hang
    want = {
        "solve": lambda: torch.linalg.solve(a, b),
        "inv": lambda: torch.linalg.inv(a),
        "det": lambda: torch.linalg.det(a),
        "slogdet": lambda: torch.linalg.slogdet(a),
        "lu_factor": lambda: torch.linalg.lu_factor(a),
    }[op]()
    assert _identical(want, got)


def test_lu_guard_runs_an_unmirrorable_broadcast_in_one_lane(splits):
    a = _spd((2, 3), P._LU_HANG_K + 8)
    b = _rhs((1, 3, P._LU_HANG_K + 8, 1))  # broadcasts over the first batch dim
    got = P.solve(a, b)
    assert splits
    torch.testing.assert_close(got, torch.linalg.solve(a, b.expand(2, -1, -1, -1)))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_batched_svd_routing_rule():
    from fastfuncstuff.policy_linalg import cuda_svd_belongs_on_cpu as rule

    def t(*shape, dtype=torch.float32):
        return torch.empty(*shape, device="cuda", dtype=dtype)

    assert not rule("svd_batched", t(512, 32, 32))  # batched Jacobi: stays
    assert rule("svd_batched", t(512, 33, 33))  # looped gesvd: leaves
    assert rule("svd_batched", t(64, 300, 50))
    assert not rule("svd_batched", t(4, 1000, 300))  # big enough that CUDA wins
    assert rule("svd_batched", t(4, 1000, 300, dtype=torch.float64))  # f64 always leaves
    assert not rule("svd", t(100, 40))  # unbatched is not routed
    assert not rule("svd_batched", t(1, 100, 40))  # a batch of one is unbatched
    assert not rule("pinv", t(64, 100, 40), {"hermitian": True})  # that is an eigh
    assert rule("pinv", t(64, 100, 40))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_batched_svd_routed_to_cpu_returns_on_cuda_and_matches():
    from fastfuncstuff import policy_linalg as pl

    a = torch.randn(64, 100, 40, device="cuda", dtype=torch.float64)
    u, s, vh = pl.svd(a, full_matrices=False)
    assert u.device.type == s.device.type == vh.device.type == "cuda"
    torch.testing.assert_close(u @ torch.diag_embed(s) @ vh, a, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(s, torch.linalg.svdvals(a.cpu()).cuda(), rtol=1e-12, atol=1e-12)
    p = pl.pinv(a)
    assert p.device.type == "cuda"
    torch.testing.assert_close(p, torch.linalg.pinv(a.cpu()).cuda(), rtol=1e-8, atol=1e-10)
