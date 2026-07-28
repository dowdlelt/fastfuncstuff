"""Criteria-pool specification for combinatorial / singleton PC selection."""

import pytest
import torch

from fastfuncstuff.denoise.combinatorial import parse_criteria_spec, select_criteria_voxels


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (0.05, ("abs", 0.05)),
        (0, ("abs", 0.0)),
        ("0.05", ("abs", 0.05)),
        ("5%", ("pct", 5.0)),
        (" 5% ", ("pct", 5.0)),
        ("(1000)", ("topn", 1000.0)),
    ],
)
def test_parse_criteria_spec(spec, expected):
    assert parse_criteria_spec(spec) == expected


@pytest.mark.parametrize("bad", ["banana", "200%", "-1%", "(0)", "()"])
def test_parse_criteria_spec_rejects_garbage(bad):
    with pytest.raises(ValueError):
        parse_criteria_spec(bad)


def test_absolute_threshold_selects_by_value():
    r2 = torch.linspace(-0.1, 0.4, 10_000)
    mask = select_criteria_voxels(r2, 0.05, verbose=False)
    assert mask.sum() == int((r2 > 0.05).sum())


def test_percentile_and_topn_are_self_sizing():
    r2 = torch.linspace(-0.1, 0.4, 10_000)
    assert select_criteria_voxels(r2, "5%", verbose=False).sum() == 500
    assert select_criteria_voxels(r2, "(1000)", verbose=False).sum() == 1000
    # Both must take the HIGH end of the distribution, not the low.
    top = select_criteria_voxels(r2, "(1000)", verbose=False)
    assert r2[top].min() > r2[~top].max()


def test_absolute_threshold_falls_back_to_percentile():
    # Nothing clears 0.9, so the pool comes from the top percentile instead.
    r2 = torch.linspace(-0.1, 0.4, 10_000)
    mask = select_criteria_voxels(r2, 0.9, fallback_percentile=5.0, verbose=False)
    assert mask.sum() == 500
    assert r2[mask].min() > r2[~mask].max()


def test_percentile_spec_never_falls_back():
    # A tiny percentile is a deliberate choice, not a failure to be rescued.
    r2 = torch.linspace(-0.1, 0.4, 10_000)
    assert select_criteria_voxels(r2, "1%", verbose=False).sum() == 100


def test_topn_clamps_to_available_voxels():
    r2 = torch.linspace(0.0, 1.0, 50)
    assert select_criteria_voxels(r2, "(1000)", verbose=False).sum() == 50
