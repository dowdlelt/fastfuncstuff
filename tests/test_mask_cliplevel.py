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


def test_automask_survives_nan_and_coverage_excludes_it():
    """NaN is not brain, and it is not data either.

    Both were silent failures: the clip level came out NaN so every ``vol >= clip``
    was False and ``automask`` returned an *empty* mask, while ``vol != 0`` is True
    for NaN so a coverage test passed the whole void through as valid. A NaN rim is
    common in the wild -- anything that divides by the data turns the exact-zero
    background into NaN, leaving a volume with an empty slab and not one zero in it.
    """
    from fastfuncstuff.processing.mask import automask, data_coverage_mask

    zz, yy, xx = torch.meshgrid(
        *(torch.arange(n, dtype=torch.float32) for n in (20, 24, 24)), indexing="ij"
    )
    # A blob on a faint but nonzero background, so "brain" and "has data" differ.
    vol = torch.exp(-(((xx - 12) / 6) ** 2 + ((yy - 12) / 6) ** 2 + ((zz - 10) / 5) ** 2)) + 0.02
    clean = automask(vol)
    assert 0.05 < clean.float().mean() < 0.95  # a sane mask to compare against

    poisoned = vol.clone()
    poisoned[:4] = float("nan")  # a clipped slab, spelled NaN rather than 0
    assert automask(poisoned).float().mean() > 0.5 * clean.float().mean()

    cover = data_coverage_mask(poisoned, erode=0)
    assert not bool(cover[:4].any())
    assert bool(cover[6:].all())
