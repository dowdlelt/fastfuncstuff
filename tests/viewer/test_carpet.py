"""Carpet plots: what makes one readable, and what would quietly lie.

A carpet is a picture with no axis labels and no numbers on it, so a mistake in
it does not look like a mistake -- it looks like data. These pin the properties
that decide whether the picture means what it appears to: that normalisation
puts voxels on a comparable scale, that an ordering actually groups what it
claims to, and that reducing 900k rows to a screenful does not depend on where
a stride happened to land.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.viewer.carpet import (
    MAX_ROWS,
    ORDERINGS,
    build_carpet,
    normalize_rows,
)

CPU = torch.device("cpu")


def _run(nt: int = 120, shape=(16, 18, 12), seed: int = 4):
    """A synthetic run: a bright brain, a global wave, and one coherent band."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = shape
    data = (rng.normal(size=(nx, ny, nz, nt)) * 2).astype(np.float32)
    brain = np.zeros(shape, bool)
    pad = (max(1, nx // 6), max(1, ny // 6), max(1, nz // 6))
    brain[pad[0] : nx - pad[0], pad[1] : ny - pad[1], pad[2] : nz - pad[2]] = True
    data[brain] += 1000.0

    t = np.arange(nt)
    data[brain] += (8 * np.sin(2 * np.pi * t / 40)).astype(np.float32)

    band = np.zeros(shape, bool)
    band[pad[0] : pad[0] + 4, pad[1] : pad[1] + 5, pad[2] : pad[2] + 3] = True
    special = np.cos(2 * np.pi * t / 13).astype(np.float32)
    data[band] += 25 * special
    return data, brain, band, special


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------


def test_z_puts_every_voxel_on_one_scale():
    """Without it the carpet is a map of tissue type, not of time."""
    rng = np.random.default_rng(1)
    flat = torch.as_tensor(rng.normal(size=(50, 80)).astype(np.float32))
    flat[:25] *= 100.0  # a tenfold-brighter tissue class
    rows, units, _ = normalize_rows(flat, "z")
    assert units == "z"
    assert rows.std(-1).max() == pytest.approx(1.0, abs=1e-3)
    assert rows.std(-1).min() == pytest.approx(1.0, abs=1e-3)


def test_percent_change_is_relative_to_each_voxels_own_baseline():
    flat = torch.ones((3, 40), dtype=torch.float32) * 500.0
    flat[:, ::2] = 505.0  # +1% on half the points, about a 502.5 mean
    rows, units, _ = normalize_rows(flat, "psc")
    assert units == "% change"
    assert rows.abs().max() == pytest.approx(0.5, abs=0.02)


def test_a_zero_baseline_does_not_blow_up_the_scale():
    """A derived layer with the mean projected out has no baseline at all."""
    flat = torch.zeros((4, 30), dtype=torch.float32)
    rows, _, _ = normalize_rows(flat, "psc")
    assert torch.isfinite(rows).all()


# ---------------------------------------------------------------------------
# orderings
# ---------------------------------------------------------------------------


def test_every_declared_ordering_runs():
    data, brain, _, special = _run()
    stat = np.zeros(brain.shape, np.float32)
    stat[brain] = 1.0
    for order in ORDERINGS:
        carpet = build_carpet(
            data,
            mask=brain,
            order=order,
            seed_series=special,
            order_volume=stat,
            device=CPU,
        )
        assert carpet.shape[1] == data.shape[3]
        assert carpet.order == order


def _row_corr(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Correlation of every carpet row with one time course.

    Amplitude is the wrong probe here: z-scoring deliberately puts every row on
    one scale, so a band that is ten times stronger looks identical to its
    neighbours. What an ordering changes is *which* rows sit together, and that
    only shows up in correlation.
    """
    rows = image - image.mean(axis=1, keepdims=True)
    rows = rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-9)
    ref = reference - reference.mean()
    ref = ref / max(float(np.linalg.norm(ref)), 1e-9)
    return rows @ ref


def test_seed_ordering_gathers_the_band_it_was_given():
    """The property that says an ordering orders rather than shuffles."""
    data, brain, _, special = _run()
    carpet = build_carpet(
        data, mask=brain, order="seed", seed_series=special, max_rows=MAX_ROWS, device=CPU
    )
    corr = _row_corr(carpet.image, special)
    n = corr.size
    assert corr[: n // 10].mean() > corr[-n // 10 :].mean() + 0.3
    # And it is a sort, so it descends the whole way down, not just at the ends.
    thirds = [corr[: n // 3].mean(), corr[n // 3 : 2 * n // 3].mean(), corr[2 * n // 3 :].mean()]
    assert thirds[0] > thirds[1] > thirds[2]


def test_acquisition_order_does_not_gather_it():
    """The contrast that makes the previous test mean something."""
    data, brain, _, special = _run()
    carpet = build_carpet(data, mask=brain, order="voxel", device=CPU)
    corr = _row_corr(carpet.image, special)
    n = corr.size
    assert abs(corr[: n // 10].mean() - corr[-n // 10 :].mean()) < 0.2


def test_pc1_ordering_puts_the_dominant_component_at_the_top():
    """The global wave is what a carpet is usually read for."""
    nt = 120
    data, brain, _, _ = _run(nt=nt)
    wave = np.sin(2 * np.pi * np.arange(nt) / 40).astype(np.float32)
    carpet = build_carpet(data, mask=brain, order="pc1", device=CPU)
    corr = _row_corr(carpet.image, wave)
    n = corr.size
    assert corr[: n // 10].mean() > corr[-n // 10 :].mean()


def test_overlay_ordering_sorts_by_the_value_it_is_handed():
    data, brain, band, _ = _run()
    stat = np.zeros(brain.shape, np.float32)
    stat[band] = 9.0
    carpet = build_carpet(
        data, mask=brain, order="overlay", order_volume=stat, sidebar_volume=stat, device=CPU
    )
    assert carpet.sidebar is not None
    # Descending, so the strongest rows are on top.
    assert carpet.sidebar[0] >= carpet.sidebar[-1]
    assert carpet.sidebar[0] > 0 and carpet.sidebar[-1] == 0


def test_the_roi_ordering_uses_a_proportion_not_the_display_threshold():
    """Otherwise the picture would change every time the colour slider moved."""
    data, brain, band, _ = _run()
    stat = np.zeros(brain.shape, np.float32)
    stat[band] = 9.0
    a = build_carpet(data, mask=brain, order="roi", order_volume=stat, device=CPU)
    b = build_carpet(data, mask=brain, order="roi", order_volume=stat * 1000.0, device=CPU)
    assert np.allclose(a.image, b.image)


def test_the_sidebar_follows_the_row_order(_=None):
    data, brain, band, special = _run()
    stat = np.zeros(brain.shape, np.float32)
    stat[band] = 5.0
    carpet = build_carpet(
        data,
        mask=brain,
        order="seed",
        seed_series=special,
        order_volume=stat,
        sidebar_volume=stat,
        device=CPU,
    )
    assert carpet.sidebar is not None
    n = carpet.sidebar.size
    assert carpet.sidebar[: n // 10].mean() > carpet.sidebar[-n // 10 :].mean()


def test_a_seed_of_the_wrong_length_is_refused():
    data, brain, _, _ = _run()
    with pytest.raises(ValueError, match="same length"):
        build_carpet(data, mask=brain, order="seed", seed_series=np.zeros(7), device=CPU)


def test_ordering_by_an_overlay_that_was_not_given_is_refused():
    data, brain, _, _ = _run()
    with pytest.raises(ValueError, match="needs an overlay"):
        build_carpet(data, mask=brain, order="overlay", device=CPU)


def test_an_unknown_ordering_is_refused():
    data, brain, _, _ = _run()
    with pytest.raises(ValueError, match="unknown ordering"):
        build_carpet(data, mask=brain, order="spiral", device=CPU)


# ---------------------------------------------------------------------------
# reduction
# ---------------------------------------------------------------------------


def test_binning_averages_within_the_ordering_rather_than_striding():
    """A stride drops whole voxels and makes the picture depend on the offset."""
    nt = 30
    data = np.zeros((4, 4, 1, nt), np.float32)
    mask = np.ones((4, 4, 1), bool)
    # Sixteen voxels, each a constant offset plus one shared shape.
    shape = np.sin(np.arange(nt) / 3.0).astype(np.float32)
    for i in range(16):
        data.reshape(-1, nt)[i] = shape * (i + 1) + 100.0
    carpet = build_carpet(data, mask=mask, order="voxel", max_rows=4, device=CPU)
    assert carpet.shape == (4, nt)
    assert carpet.binned and carpet.n_voxels == 16
    # Every voxel contributed: a mean of z-scored copies of one shape is that
    # shape, so no row is flat and none is missing.
    assert (np.abs(carpet.image).max(axis=1) > 0.5).all()


def test_a_small_run_is_not_binned_at_all():
    data, brain, _, _ = _run(shape=(8, 8, 4))
    carpet = build_carpet(data, mask=brain, order="voxel", device=CPU)
    assert not carpet.binned
    assert carpet.shape[0] == carpet.n_voxels


def test_the_status_line_says_what_was_done():
    data, brain, _, _ = _run()
    carpet = build_carpet(data, mask=brain, order="pc1", max_rows=100, device=CPU)
    status = carpet.status()
    assert "voxels" in status and "100 rows" in status and "PC1" in status


# ---------------------------------------------------------------------------
# the awkward inputs
# ---------------------------------------------------------------------------


def test_a_three_d_dataset_is_refused():
    with pytest.raises(ValueError, match="4-D"):
        build_carpet(np.zeros((4, 4, 4), np.float32), device=CPU)


def test_a_single_volume_is_refused():
    with pytest.raises(ValueError, match="more than one volume"):
        build_carpet(np.zeros((4, 4, 4, 1), np.float32), device=CPU)


def test_an_empty_mask_is_refused():
    data, brain, _, _ = _run()
    with pytest.raises(ValueError, match="mask is empty"):
        build_carpet(data, mask=np.zeros_like(brain), device=CPU)


def test_detrending_removes_the_drift_that_would_swamp_it():
    """A carpet is unreadable through a linear ramp; polort is why."""
    data, brain, _, _ = _run()
    data = data + (0.5 * np.arange(data.shape[3])).astype(np.float32)
    raw = build_carpet(data, mask=brain, order="voxel", polort=-1, device=CPU)
    clean = build_carpet(data, mask=brain, order="voxel", polort=1, device=CPU)

    # The drift dominates every row before detrending: first half against
    # second half separates hard, and should not after.
    def split(carpet):
        half = carpet.shape[1] // 2
        return abs(carpet.image[:, :half].mean() - carpet.image[:, half:].mean())

    assert split(raw) > split(clean) * 3


def test_automask_finds_the_brain_and_not_the_air():
    from fastfuncstuff.viewer.carpet import automask_from_series

    data, brain, _, _ = _run()
    mask = automask_from_series(data, device=CPU)
    overlap = (mask & brain).sum() / brain.sum()
    assert overlap > 0.9
    assert mask.sum() < brain.size * 0.9
