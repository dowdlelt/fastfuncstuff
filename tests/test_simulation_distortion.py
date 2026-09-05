"""The synthetic PE field and its forward warp.

These back the only correctness gate ffs_locomoco can have: there is no
reference implementation, so the benchmark imposes a field it knows and scores
the recovery. That is worth nothing if the imposed field is not what we think
it is, so the forward model's invariants are pinned here.
"""

from __future__ import annotations

import pytest
import torch

from fastfuncstuff.simulation.distortion import apply_pe_shift, synthetic_pe_field


def test_field_honours_its_amplitude_and_shape():
    field = synthetic_pe_field((12, 14, 8), 40, amplitude=0.8)
    assert tuple(field.shape) == (12, 14, 8, 40)
    assert float(field.abs().max()) == pytest.approx(0.8, rel=1e-5)


def test_field_is_deterministic():
    """A benchmark threshold is meaningless if the field moves run to run."""
    a = synthetic_pe_field((10, 10, 6), 20, amplitude=0.5, tr=1.5)
    b = synthetic_pe_field((10, 10, 6), 20, amplitude=0.5, tr=1.5)
    assert torch.equal(a, b)


def test_field_actually_varies_in_space_and_time():
    """A field constant in either axis would be recovered by a broken estimator."""
    field = synthetic_pe_field((12, 12, 8), 30, amplitude=0.7)
    assert float(field.std(dim=-1).mean()) > 0.05  # varies over time
    assert float(field.std(dim=(0, 1, 2)).mean()) > 0.05  # varies over space


def test_zero_field_is_exactly_the_identity():
    torch.manual_seed(0)
    series = torch.randn(9, 11, 5, 6)
    assert torch.equal(apply_pe_shift(series, torch.zeros_like(series), 1), series)


@pytest.mark.parametrize("pe_axis", [0, 1, 2])
def test_unit_shift_equals_a_roll(pe_axis):
    """Pins the sign convention: a +1 field pulls from index i+1."""
    torch.manual_seed(0)
    series = torch.randn(10, 11, 9, 3)
    got = apply_pe_shift(series, torch.ones_like(series), pe_axis)
    expected = torch.roll(series, shifts=-1, dims=pe_axis)
    interior = [slice(None)] * 4
    interior[pe_axis] = slice(2, -2)  # away from the clamped edges
    assert torch.equal(got[tuple(interior)], expected[tuple(interior)])


def test_a_constant_image_survives_the_warp():
    """Catmull-Rom weights sum to 1, so a flat region keeps its intensity."""
    flat = torch.full((10, 10, 6, 4), 3.25)
    field = synthetic_pe_field((10, 10, 6), 4, amplitude=0.6)
    assert torch.allclose(apply_pe_shift(flat, field, 1), flat, atol=1e-5)


def test_mismatched_shapes_are_rejected():
    series = torch.randn(8, 8, 4, 5)
    with pytest.raises(ValueError, match="!="):
        apply_pe_shift(series, torch.zeros(8, 8, 4, 6), 1)


def test_a_bad_axis_is_rejected():
    series = torch.randn(8, 8, 4, 5)
    with pytest.raises(ValueError, match="pe_axis"):
        apply_pe_shift(series, torch.zeros_like(series), 3)


def test_the_warp_is_not_lanczos():
    """The forward model must differ from locomoco's resampler, or the score
    flatters the estimator by exactly the interpolator's own error.

    A half-voxel shift is where the two kernels disagree most.
    """
    from fastfuncstuff.processing.locomoco import _shift1d_lanczos_body

    torch.manual_seed(0)
    series = torch.randn(1, 24, 1, 1)
    half = torch.full_like(series, 0.5)
    cubic = apply_pe_shift(series, half, 1)
    # locomoco's body wants (batch, ...) with the batch in dim 0.
    lanczos = _shift1d_lanczos_body(series[0], half[0], 0, 24, 3)
    assert not torch.allclose(cubic[0], lanczos, atol=1e-4)
