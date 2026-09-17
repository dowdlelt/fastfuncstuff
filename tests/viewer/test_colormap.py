"""Colour mapping, thresholding and compositing.

The properties that matter are the ones a stats map is read through: a voxel at
threshold must be fully opaque, a voxel at zero must be fully transparent under
a ramp, and a one-sided sign mode must not leak the other sign.
"""

from __future__ import annotations

import pytest
import torch

from fastfuncstuff.viewer.colormap import (
    apply_colormap,
    available_colormaps,
    build_lut,
    composite,
    normalize,
    quantize,
    suprathreshold_edges,
    threshold_alpha,
    to_rgba8,
)
from fastfuncstuff.viewer.layers import AlphaMode, SignMode

CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# LUTs
# ---------------------------------------------------------------------------


def test_every_advertised_colormap_builds():
    for name in available_colormaps():
        lut = build_lut(name, device=CPU)
        assert lut.shape == (256, 3)
        assert torch.isfinite(lut).all()
        assert (lut >= 0).all() and (lut <= 1).all()


def test_lut_endpoints_match_the_scale_definition():
    lut = build_lut("gray", 256, device=CPU)
    assert torch.allclose(lut[0], torch.zeros(3), atol=1e-6)
    assert torch.allclose(lut[-1], torch.ones(3), atol=1e-6)


def test_lut_is_monotonic_for_gray():
    lut = build_lut("gray", 64, device=CPU)
    assert torch.all(lut[1:, 0] >= lut[:-1, 0] - 1e-6)


def test_unknown_colormap_names_the_alternatives():
    with pytest.raises(KeyError, match="gray"):
        build_lut("not-a-map", device=CPU)


def test_tiny_lut_is_rejected():
    with pytest.raises(ValueError):
        build_lut("gray", 1, device=CPU)


# ---------------------------------------------------------------------------
# normalization and sign modes
# ---------------------------------------------------------------------------


def test_both_sign_puts_zero_at_the_midpoint_of_a_symmetric_range():
    """A diverging scale is only honest if zero lands on its neutral colour."""
    v = torch.tensor([-4.0, 0.0, 4.0])
    unit = normalize(v, -4.0, 4.0, sign_mode=SignMode.BOTH)
    assert torch.allclose(unit, torch.tensor([0.0, 0.5, 1.0]), atol=1e-6)


def test_pos_mode_clamps_negatives_to_zero():
    v = torch.tensor([-5.0, 0.0, 2.5, 5.0])
    unit = normalize(v, -5.0, 5.0, sign_mode=SignMode.POS)
    assert torch.allclose(unit, torch.tensor([0.0, 0.0, 0.5, 1.0]), atol=1e-6)


def test_neg_mode_mirrors_pos_mode():
    v = torch.tensor([-5.0, -2.5, 0.0, 5.0])
    unit = normalize(v, -5.0, 5.0, sign_mode=SignMode.NEG)
    assert torch.allclose(unit, torch.tensor([1.0, 0.5, 0.0, 0.0]), atol=1e-6)


def test_degenerate_range_does_not_divide_by_zero():
    unit = normalize(torch.tensor([1.0, 2.0]), 3.0, 3.0)
    assert torch.isfinite(unit).all()


# ---------------------------------------------------------------------------
# discrete panes
# ---------------------------------------------------------------------------


def test_quantize_produces_exactly_n_distinct_bands():
    unit = torch.linspace(0.0, 1.0, 1000)
    assert torch.unique(quantize(unit, 8)).numel() == 8


def test_quantize_is_a_noop_for_a_continuous_scale():
    unit = torch.linspace(0.0, 1.0, 17)
    assert torch.equal(quantize(unit, 0), unit)


def test_panelled_map_reads_back_a_recoverable_band():
    """Banding exists so a value can be read off by eye; bands must not drift."""
    lut = build_lut("gray", 256, device=CPU)
    v = torch.tensor([0.1, 0.6])
    rgb = apply_colormap(v, lut=lut, lo=0.0, hi=1.0, n_panes=4)
    # 0.1 lands in band 0 (centre 0.125), 0.6 in band 2 (centre 0.625)
    assert rgb[0, 0] < rgb[1, 0]
    assert abs(float(rgb[0, 0]) - 0.125) < 0.02
    assert abs(float(rgb[1, 0]) - 0.625) < 0.02


# ---------------------------------------------------------------------------
# threshold and alpha
# ---------------------------------------------------------------------------


def test_hard_threshold_is_a_binary_mask():
    stat = torch.tensor([0.0, 1.9, 2.0, 5.0])
    a = threshold_alpha(stat, 2.0, mode=AlphaMode.OFF)
    assert torch.equal(a, torch.tensor([0.0, 0.0, 1.0, 1.0]))


def test_a_voxel_exactly_at_threshold_passes():
    a = threshold_alpha(torch.tensor([3.0]), 3.0, mode=AlphaMode.OFF)
    assert float(a[0]) == 1.0


def test_linear_alpha_fades_to_zero_at_zero_and_one_at_threshold():
    stat = torch.tensor([0.0, 1.0, 2.0, 4.0])
    a = threshold_alpha(stat, 2.0, mode=AlphaMode.LINEAR)
    assert torch.allclose(a, torch.tensor([0.0, 0.5, 1.0, 1.0]), atol=1e-6)


def test_quadratic_alpha_is_the_square_of_linear_below_threshold():
    stat = torch.tensor([1.0])
    lin = threshold_alpha(stat, 2.0, mode=AlphaMode.LINEAR)
    quad = threshold_alpha(stat, 2.0, mode=AlphaMode.QUADRATIC)
    assert torch.allclose(quad, lin**2, atol=1e-6)


def test_alpha_is_symmetric_in_sign_under_both_mode():
    a = threshold_alpha(torch.tensor([-3.0, 3.0]), 2.0, mode=AlphaMode.LINEAR)
    assert float(a[0]) == float(a[1]) == 1.0


def test_pos_sign_mode_hides_negative_voxels_entirely():
    """A one-sided map must never leak the other sign, at any alpha mode."""
    stat = torch.tensor([-9.0, 9.0])
    for mode in (AlphaMode.OFF, AlphaMode.LINEAR, AlphaMode.QUADRATIC):
        a = threshold_alpha(stat, 2.0, mode=mode, sign_mode=SignMode.POS)
        assert float(a[0]) == 0.0, mode
        assert float(a[1]) == 1.0, mode


def test_zero_threshold_shows_everything_under_both_mode():
    a = threshold_alpha(torch.tensor([-1.0, 0.0, 1.0]), 0.0)
    assert torch.equal(a, torch.ones(3))


def test_zero_threshold_still_respects_a_one_sided_mode():
    a = threshold_alpha(torch.tensor([-1.0, 1.0]), 0.0, sign_mode=SignMode.POS)
    assert torch.equal(a, torch.tensor([0.0, 1.0]))


# ---------------------------------------------------------------------------
# boxed outlines
# ---------------------------------------------------------------------------


def test_edges_outline_a_block_without_filling_it():
    stat = torch.zeros(7, 7)
    stat[2:5, 2:5] = 9.0
    edges = suprathreshold_edges(stat, 2.0)
    assert not bool(edges[3, 3]), "interior must not be an edge"
    assert bool(edges[2, 2]) and bool(edges[4, 4])
    assert int(edges.sum()) == 8  # 3x3 block minus its centre


def test_edges_are_empty_when_nothing_passes():
    stat = torch.zeros(5, 5)
    assert int(suprathreshold_edges(stat, 1.0).sum()) == 0


def test_edges_reject_non_2d_input():
    with pytest.raises(ValueError):
        suprathreshold_edges(torch.zeros(3, 3, 3), 1.0)


# ---------------------------------------------------------------------------
# compositing
# ---------------------------------------------------------------------------


def test_opaque_top_layer_hides_the_one_below():
    red = torch.tensor([[1.0, 0.0, 0.0]])
    blue = torch.tensor([[0.0, 0.0, 1.0]])
    out = composite([(red, torch.ones(1)), (blue, torch.ones(1))])
    assert torch.allclose(out, blue, atol=1e-6)


def test_transparent_top_layer_leaves_the_one_below():
    red = torch.tensor([[1.0, 0.0, 0.0]])
    blue = torch.tensor([[0.0, 0.0, 1.0]])
    out = composite([(red, torch.ones(1)), (blue, torch.zeros(1))])
    assert torch.allclose(out, red, atol=1e-6)


def test_half_alpha_blends_evenly():
    red = torch.tensor([[1.0, 0.0, 0.0]])
    blue = torch.tensor([[0.0, 0.0, 1.0]])
    out = composite([(red, torch.ones(1)), (blue, torch.full((1,), 0.5))])
    assert torch.allclose(out, torch.tensor([[0.5, 0.0, 0.5]]), atol=1e-6)


def test_composite_needs_at_least_one_layer():
    with pytest.raises(ValueError):
        composite([])


def test_rgba8_packs_to_bytes():
    rgb = torch.tensor([[[1.0, 0.0, 0.5]]])
    out = to_rgba8(rgb, torch.tensor([[0.5]]))
    assert out.dtype is torch.uint8
    assert out.shape == (1, 1, 4)
    assert out[0, 0, 0] == 255 and out[0, 0, 1] == 0
    assert out[0, 0, 3] == 128


def test_rgba8_defaults_to_opaque():
    out = to_rgba8(torch.zeros(2, 2, 3))
    assert (out[..., 3] == 255).all()


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


def test_a_stats_overlay_over_anatomy_reads_as_expected():
    """The path a real overlay takes: anatomy, then a thresholded stat map."""
    anat = torch.rand(16, 16)
    stat = torch.zeros(16, 16)
    stat[4:8, 4:8] = 6.0

    gray = build_lut("gray", device=CPU)
    hot = build_lut("hot", device=CPU)

    anat_rgb = apply_colormap(anat, lut=gray, lo=0.0, hi=1.0)
    stat_rgb = apply_colormap(stat, lut=hot, lo=0.0, hi=8.0, sign_mode=SignMode.POS)
    stat_a = threshold_alpha(stat, 3.0, mode=AlphaMode.LINEAR, sign_mode=SignMode.POS)

    out = composite([(anat_rgb, torch.ones_like(anat)), (stat_rgb, stat_a)])
    assert out.shape == (16, 16, 3)
    # Inside the blob the stat colour wins; outside, anatomy is untouched.
    assert not torch.allclose(out[5, 5], anat_rgb[5, 5], atol=1e-3)
    assert torch.allclose(out[0, 0], anat_rgb[0, 0], atol=1e-6)
