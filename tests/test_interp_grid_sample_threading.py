"""The N=1 CPU grid_sample re-slice must be invisible in its results.

ATen threads grid_sampler_3d over the batch dimension only, and every FFS call
site passes N=1, so the points are dealt across N to get the cores working. The
whole change is worthless if it perturbs a single sampled value, so these tests
pin bit-identity against the stock call rather than a tolerance.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fastfuncstuff.processing.interp import (
    _CPU_GRID_SPLIT_MIN_POINTS,
    _grid_sample_3d,
    _grid_sample_3d_cpu_threaded,
)


def _stock(inp, grid, mode="bilinear"):
    return F.grid_sample(inp, grid, mode=mode, padding_mode="border", align_corners=True)


def _source(channels=1):
    torch.manual_seed(0)
    return torch.randn(1, channels, 9, 11, 13)


def _grid(*shape):
    torch.manual_seed(1)
    return torch.rand(1, *shape, 3) * 2.4 - 1.2  # deliberately over-runs the border


@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize(
    "shape",
    [
        (_CPU_GRID_SPLIT_MIN_POINTS, 1, 1),  # exactly at the threshold
        (9973, 1, 1),  # a prime point count: every lane split is uneven
        (40, 50, 7),  # a real D/H/W grid, not a packed column
        (100, 1, 1),  # below the threshold: falls through to the stock call
    ],
)
def test_threaded_sample_is_bit_identical(shape, mode):
    src, grid = _source(), _grid(*shape)
    expected = _stock(src, grid, mode)
    got = _grid_sample_3d(src, grid, mode=mode)
    assert got.shape == expected.shape
    assert torch.equal(got, expected)


def test_multichannel_sources_keep_their_channel_order():
    """The lane merge permutes (lanes, C, points); a wrong axis order would still
    return the right shape while silently transposing the channels."""
    src, grid = _source(channels=3), _grid(4096, 1, 1)
    assert torch.equal(_grid_sample_3d(src, grid), _stock(src, grid))


def test_autograd_stays_on_the_single_threaded_path():
    """Expanding the source would make grad_input one full volume per lane."""
    src = _source()
    grid = _grid(_CPU_GRID_SPLIT_MIN_POINTS, 1, 1).requires_grad_(True)
    assert _grid_sample_3d_cpu_threaded(src, grid, "bilinear", True) is None
    _grid_sample_3d(src, grid).sum().backward()
    assert grid.grad is not None and torch.isfinite(grid.grad).all()


def test_split_declines_when_it_cannot_help():
    src = _source()
    # Too few points to pay for the re-slice.
    assert _grid_sample_3d_cpu_threaded(src, _grid(8, 1, 1), "bilinear", True) is None
    # A real batch already gives ATen something to thread over.
    batched = torch.randn(4, 1, 9, 11, 13)
    grid = torch.rand(4, 4096, 1, 1, 3) * 2 - 1
    assert _grid_sample_3d_cpu_threaded(batched, grid, "bilinear", True) is None


def test_single_thread_runtime_declines_the_split(monkeypatch):
    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    src, grid = _source(), _grid(_CPU_GRID_SPLIT_MIN_POINTS, 1, 1)
    assert _grid_sample_3d_cpu_threaded(src, grid, "bilinear", True) is None
    assert torch.equal(_grid_sample_3d(src, grid), _stock(src, grid))
