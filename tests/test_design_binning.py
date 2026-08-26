"""Task-state binning for the condition-paired reference."""

import numpy as np
import pytest
import torch

from fastfuncstuff.design.binning import design_state_bins, format_bin_report


def _block_design(n_t: int, block: int = 20) -> torch.Tensor:
    box = ((np.arange(n_t) // block) % 2).astype(np.float64)
    k = np.exp(-np.arange(0, 20) / 3.0)
    conv = np.convolve(box, k / k.sum())[:n_t]
    return torch.tensor(conv)[:, None]


def test_bin_width_sets_the_level_count():
    x = _block_design(120)
    for width, levels in ((0.5, 2), (0.2, 5), (0.1, 10)):
        _, info = design_state_bins(x, bin_width=width, min_frames=1)
        assert info["levels_requested"] == levels


def test_plateaus_get_the_frames_and_slopes_are_thin():
    """The shape a block design must produce: two fat bins and thin slope bins.

    This is the whole reason for binning on the CONVOLVED design — the slope frames
    exist and are their own state, rather than being filed under whichever label the
    boxcar carried.
    """
    x = _block_design(120, block=20)
    bin_of, info = design_state_bins(x, bin_width=0.2, min_frames=1)
    counts = sorted(info["counts"], reverse=True)
    assert info["n_bins"] == 5
    assert counts[0] > 30 and counts[1] > 30  # baseline and peak plateaus
    assert sum(counts[2:]) < 30  # the slopes
    assert int(bin_of.max()) == info["n_bins"] - 1


def test_every_frame_is_assigned_exactly_once():
    x = _block_design(120)
    bin_of, info = design_state_bins(x, bin_width=0.2, min_frames=1)
    assert bin_of.shape == (120,)
    assert sum(info["counts"]) == 120
    assert set(int(v) for v in bin_of) == set(range(info["n_bins"]))


def test_sparse_bins_are_merged_not_shipped():
    """A template averaged over two frames hands its own noise to every frame using it."""
    x = _block_design(120, block=20)
    _, fine = design_state_bins(x, n_bins=20, min_frames=1)
    _, merged = design_state_bins(x, n_bins=20, min_frames=8)
    assert merged["n_merged"] > 0
    assert merged["n_bins"] < fine["n_bins"]
    assert min(merged["counts"]) >= 8


def test_merging_stops_rather_than_collapsing_to_one_bin():
    x = _block_design(60, block=10)
    _, info = design_state_bins(x, n_bins=8, min_frames=10_000)
    assert info["n_bins"] == 1  # nothing can satisfy it; it must terminate, not hang


def test_columns_are_scaled_independently():
    """A weak condition must not be binned into one level by a strong one's range."""
    n_t = 120
    strong = _block_design(n_t, block=20) * 100.0
    weak = _block_design(n_t, block=15) * 0.01
    x = torch.cat([strong, weak], dim=1)
    _, info = design_state_bins(x, bin_width=0.34, min_frames=1)
    states = np.array(info["states"])
    assert np.ptp(states[:, 1]) > 0.4  # the weak column still spans its own range


def test_top_of_range_lands_in_the_last_bin():
    """floor(z*levels) puts z==1.0 one past the end without the clamp."""
    x = torch.linspace(0, 1, 50, dtype=torch.float64)[:, None]
    bin_of, info = design_state_bins(x, n_bins=5, min_frames=1)
    assert info["n_bins"] == 5
    assert int(bin_of[-1]) == int(bin_of.max())


def test_too_few_levels_is_refused():
    with pytest.raises(ValueError, match="at least 2 levels"):
        design_state_bins(_block_design(60), n_bins=1)


def test_report_names_the_conditions():
    x = _block_design(120)
    _, info = design_state_bins(x, bin_width=0.2, min_frames=1)
    txt = format_bin_report(info, ["checkerboard"])
    assert "checkerboard=" in txt
    assert f"{info['n_bins']} bins" in txt


@pytest.mark.parametrize(
    ("ref", "paired", "stat"),
    [
        ("mean", False, "mean"),
        ("median", False, "median"),
        ("max", False, "max"),
        ("first_mean", False, "first_mean"),
        ("paired", True, "mean"),
        ("paired_mean", True, "mean"),
        ("paired_median", True, "median"),
    ],
)
def test_ref_paired_splits_into_bins_and_statistic(ref, paired, stat):
    """`-ref paired[_stat]` is one user-facing name over two independent axes.

    Which frames form the reference, and what statistic reduces them, are separate
    questions; the library keeps them separate and only the CLI name is compound —
    the same shape as the existing first_mean / first_median.
    """
    from fastfuncstuff.cli.locomoco import parse_ref_mode

    is_paired, statistic, label = parse_ref_mode(ref)
    assert is_paired is paired
    assert statistic == stat
    assert label == ref  # the banner keeps the name the user typed
