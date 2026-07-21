"""Clip-level estimation must tolerate full-resolution anatomicals.

``torch.quantile`` refuses inputs above 2**24 elements ("input tensor is too
large"), which crashed ``automask`` on large T1s (ffs_segment -autobox). The
``_quantile`` helper falls back to ``kthvalue`` for those.
"""

import torch

from fastfuncstuff.processing.mask import _cliplevel, _quantile


def test_quantile_matches_torch_on_small_input():
    x = torch.rand(10_000)
    assert abs(_quantile(x, 0.35) - float(x.quantile(0.35))) < 1e-5


def test_quantile_handles_tensor_above_torch_limit():
    # > 2**24 elements: torch.quantile raises, _quantile must not.
    big = torch.rand(17_000_000)
    q = _quantile(big, 0.35)
    assert 0.30 < q < 0.40  # uniform → ~q


def test_cliplevel_runs_on_large_volume():
    # A volume large enough that the positive-voxel set exceeds the quantile limit.
    vol = torch.rand(300, 300, 300)
    clip = _cliplevel(vol)
    assert clip > 0
