"""The Lanczos tap loop is compiled on CPU once the work pays for the warmup.

Eager, it is ~48 unfused passes over the array (clamp, two sincs, a gather, a
multiply and two accumulations, six taps over). It was 27% of ffs_locomoco on a
CPU profile once the conv2d blur stopped dominating. CUDA already had a fused
Triton path; this is its CPU counterpart, and these tests pin that it is only a
speed change.
"""

from __future__ import annotations

import pytest
import torch

from fastfuncstuff.processing import locomoco as L

CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def reset_compile_state():
    """Each test decides the gate for itself."""
    L._compiled_shift1d.pop("cpu", None)
    L._shift1d_compile_pending.discard("cpu")
    L._shift1d_eager_seconds["cpu"] = 0.0
    yield
    L._compiled_shift1d.pop("cpu", None)
    L._shift1d_compile_pending.discard("cpu")
    L._shift1d_eager_seconds["cpu"] = 0.0


def _eager(vol, shift, axis, radius=3):
    dim = axis + 1
    return L._shift1d_lanczos_body(vol, shift, dim, vol.shape[dim], radius)


def test_gate_is_shut_until_the_work_pays_for_a_warmup():
    torch.manual_seed(0)
    vol, shift = torch.randn(8, 20, 20), torch.randn(8, 20, 20) * 0.4
    assert L._get_shift1d_body(CPU) is L._shift1d_lanczos_body
    assert torch.equal(L._shift1d_windowed_sinc(vol, shift, 1, 3), _eager(vol, shift, 1))
    # Eager calls feed the budget so the decision can eventually flip.
    assert L._shift1d_eager_seconds["cpu"] > 0.0


def test_compiled_result_matches_eager_to_float32_rounding():
    torch.manual_seed(0)
    vol, shift = torch.randn(32, 24, 24), torch.randn(32, 24, 24) * 0.5
    expected = _eager(vol, shift, 1)
    L._shift1d_eager_seconds["cpu"] = 1e9  # the budget a real run has spent
    got = L._shift1d_windowed_sinc(vol, shift, 1, 3)
    assert L._compiled_shift1d["cpu"] is not L._shift1d_lanczos_body
    assert float((got - expected).abs().max()) < 1e-5


def test_float64_shows_it_is_the_same_computation():
    """float32 differs by ~1e-07 only from accumulation order; float64 pins that
    the fused kernel computes the same thing, not merely something close."""
    torch.manual_seed(0)
    vol = torch.randn(16, 20, 20, dtype=torch.float64)
    shift = torch.randn(16, 20, 20, dtype=torch.float64) * 0.5
    expected = _eager(vol, shift, 1)
    L._shift1d_eager_seconds["cpu"] = 1e9
    assert torch.allclose(L._shift1d_windowed_sinc(vol, shift, 1, 3), expected, atol=1e-12)


@pytest.mark.parametrize(
    "shape,axis,scalar_shift",
    [
        ((8, 20, 20), 0, False),
        ((8, 20, 20), 1, False),
        ((8, 20, 20), 1, True),  # a scalar shift traces to a different graph
        ((4, 10, 12, 14), 2, False),  # 4-D volume, not a slice stack
    ],
)
def test_axes_ranks_and_scalar_shifts_all_agree(shape, axis, scalar_shift):
    torch.manual_seed(1)
    vol = torch.randn(*shape)
    shift = 0.35 if scalar_shift else torch.randn(*shape) * 0.4
    expected = _eager(vol, shift, axis)
    L._shift1d_eager_seconds["cpu"] = 1e9
    got = L._shift1d_windowed_sinc(vol, shift, axis, 3)
    assert got.shape == expected.shape
    assert float((got - expected).abs().max()) < 1e-5


def test_differentiable_calls_stay_eager():
    """The optimizers backward through this; the compiled body is not on that path."""
    L._shift1d_eager_seconds["cpu"] = 1e9
    vol = torch.randn(4, 12, 12, requires_grad=True)
    out = L._shift1d_windowed_sinc(vol, torch.zeros(4, 12, 12), 1, 3)
    out.sum().backward()
    assert vol.grad is not None and torch.isfinite(vol.grad).all()
    assert "cpu" not in L._compiled_shift1d  # never even consulted the gate


def test_env_kill_switch_keeps_it_eager(monkeypatch):
    monkeypatch.setenv("FFS_NWARP_NO_COMPILE", "1")
    L._shift1d_eager_seconds["cpu"] = 1e9
    assert L._get_shift1d_body(CPU) is L._shift1d_lanczos_body


def test_a_flat_region_keeps_its_intensity():
    """DC normalisation: the weights are renormalised so a constant survives."""
    L._shift1d_eager_seconds["cpu"] = 1e9
    flat = torch.full((4, 16, 16), 2.75)
    got = L._shift1d_windowed_sinc(flat, torch.full((4, 16, 16), 0.37), 1, 3)
    assert torch.allclose(got, flat, atol=1e-5)
