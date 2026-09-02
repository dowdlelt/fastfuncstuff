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


class TestSixConnPrimitives:
    """The pooled dilation and the two flood-fill paths must agree exactly.

    A 6-connectivity dilation used to be a cross-shaped conv3d and the flood
    fills iterated it ~60 times; both are now cheaper forms of the same thing,
    and the CPU takes a labelling shortcut. Any drift here silently changes
    every automask in the toolbox.
    """

    @staticmethod
    def _conv_dilate(x):
        """The original cross-kernel convolution, kept as the reference."""
        import torch.nn.functional as F

        kernel = torch.zeros(1, 1, 3, 3, 3, dtype=torch.float32)
        for index in ((1, 1, 1), (0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)):
            kernel[0, 0][index] = 1
        return (F.conv3d(x, kernel, padding=1) > 0.5).float()

    @staticmethod
    def _blobs(seed=0):
        torch.manual_seed(seed)
        vol = torch.zeros(20, 18, 16)
        vol[4:12, 4:12, 3:10] = 1.0  # main body
        vol[6:9, 6:9, 5:7] = 0.0  # interior hole
        vol[16:19, 14:17, 12:15] = 1.0  # detached blob
        return vol

    def test_pooled_dilation_matches_the_convolution(self):
        from fastfuncstuff.processing.mask import _dilate_6conn_once

        x = self._blobs()[None, None]
        assert torch.equal(_dilate_6conn_once(x), self._conv_dilate(x))

    def test_dilation_is_six_connected_not_twenty_six(self):
        """A 3x3x3 max-pool would also light the diagonals. This must not."""
        from fastfuncstuff.processing.mask import _dilate_6conn_once

        x = torch.zeros(1, 1, 3, 3, 3)
        x[0, 0, 1, 1, 1] = 1.0
        assert int(_dilate_6conn_once(x).sum()) == 7

    def test_flood_fill_keeps_only_the_seeded_component(self):
        from fastfuncstuff.processing.mask import _flood_fill_6conn

        allowed = self._blobs()
        seed = torch.zeros_like(allowed)
        seed[5, 5, 4] = 1.0
        out = _flood_fill_6conn(seed, allowed)
        assert bool(out[5, 5, 4])
        assert not bool(out[17, 15, 13])  # the detached blob is not reached

    def test_cpu_labelling_matches_the_iterative_fill(self):
        from fastfuncstuff.processing import mask as mask_module

        allowed = self._blobs()
        seed = torch.zeros_like(allowed)
        seed[5, 5, 4] = 1.0
        labelled = mask_module._flood_fill_6conn_labelled(seed, allowed)

        # The GPU form, run here on CPU tensors: dilate to saturation.
        current = seed[None, None]
        for _ in range(sum(allowed.shape)):
            current = mask_module._dilate_6conn_once(current) * allowed[None, None]

        assert torch.equal(labelled, current[0, 0] > 0.5)

    def test_holes_are_filled_and_the_detached_blob_survives(self):
        from fastfuncstuff.processing.mask import _fill_holes_3d

        filled = _fill_holes_3d(self._blobs() > 0.5)
        assert bool(filled[7, 7, 5])  # interior hole closed
        assert bool(filled[17, 15, 13])  # hole filling does not delete components
