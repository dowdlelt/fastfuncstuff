"""Compiling the tap-weight kernels must not change what they return.

Only the gather was ever compiled; the weights feeding it cost more than it does
(_wsinc5_kernel was 32.5s of a 135.6s ffs_nwarp CPU run against ~13.7s in the
gather). Compiling them is worth ~5x, but only if the weights are the same
weights -- and for wsinc5 that is not automatic, because its shape comes from
env vars read through an lru_cache that dynamo traces straight through.
"""

from __future__ import annotations

import os

import pytest
import torch

from fastfuncstuff.processing import interp
from fastfuncstuff.processing.interp import (
    _axis_weights,
    _cubic_kernel,
    _get_kernel_fn,
    _heptic_kernel,
    _quintic_kernel,
    _wsinc5_kernel,
    _wsinc5_params,
)

CPU = torch.device("cpu")


@pytest.fixture
def compiled_gather():
    """Open the gate: the weight compile is gated on the gather having compiled."""
    previous = interp._compiled_gather_contract.get("cpu")
    interp._compiled_gather_contract["cpu"] = torch.compile(interp._gather_contract)
    yield
    if previous is None:
        interp._compiled_gather_contract.pop("cpu", None)
    else:
        interp._compiled_gather_contract["cpu"] = previous


@pytest.fixture
def clean_wsinc5_env():
    previous = os.environ.get("AFNI_WSINC5_RADIUS")
    yield
    if previous is None:
        os.environ.pop("AFNI_WSINC5_RADIUS", None)
    else:
        os.environ["AFNI_WSINC5_RADIUS"] = previous
    _wsinc5_params.cache_clear()


def test_gate_is_shut_until_the_gather_compiles():
    interp._compiled_gather_contract.pop("cpu", None)
    assert _get_kernel_fn(_cubic_kernel, CPU) is _cubic_kernel


def test_an_eager_sentinel_is_not_mistaken_for_a_compile(compiled_gather):
    """A failed compile stores the eager function; presence alone must not open
    the gate, or the weights would compile on a build where compile is broken."""
    interp._compiled_gather_contract["cpu"] = interp._gather_contract
    assert _get_kernel_fn(_cubic_kernel, CPU) is _cubic_kernel


@pytest.mark.parametrize("kernel", [_cubic_kernel, _quintic_kernel, _heptic_kernel])
def test_polynomial_kernels_are_bit_identical(kernel, compiled_gather):
    frac = torch.rand(50_000)
    assert torch.equal(_axis_weights(kernel, frac), kernel(frac))


def test_wsinc5_matches_eager_to_float32_rounding(compiled_gather):
    """Not bit-identical: inductor reassociates the sinc/window chain. ~2 ulp."""
    frac = torch.rand(50_000)
    got, expected = _axis_weights(_wsinc5_kernel, frac), _wsinc5_kernel(frac)
    assert got.shape == expected.shape
    assert float((got - expected).abs().max()) < 1e-6


@pytest.mark.parametrize("radius,taps", [("3", 6), ("9", 18), ("5", 10)])
def test_wsinc5_radius_env_is_honoured_after_compiling(
    radius, taps, compiled_gather, clean_wsinc5_env
):
    """The regression this split exists for.

    dynamo traces through the lru_cache on _wsinc5_params and bakes the values in
    without a guard, and its cache is keyed on the CODE OBJECT -- so a fresh
    torch.compile() call reuses the stale trace. Before the params became
    arguments, this returned 10 taps where eager returned 18: silently wrong
    weights for any site that sets AFNI_WSINC5_RADIUS globally.
    """
    os.environ["AFNI_WSINC5_RADIUS"] = radius
    _wsinc5_params.cache_clear()
    frac = torch.rand(4096)
    got, expected = _axis_weights(_wsinc5_kernel, frac), _wsinc5_kernel(frac)
    assert got.shape[1] == taps == expected.shape[1]
    assert float((got - expected).abs().max()) < 1e-6


def test_differentiable_weights_stay_eager(compiled_gather):
    frac = torch.rand(4096, requires_grad=True)
    out = _axis_weights(_cubic_kernel, frac)
    out.sum().backward()
    assert frac.grad is not None and torch.isfinite(frac.grad).all()
