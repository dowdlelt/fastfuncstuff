"""Ribbon profiles and their flags on a phantom with known boundaries."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from fastfuncstuff.surface.profiles import (
    ProfileSpec,
    profile_scores,
    sample_profiles,
    slab_contour_order,
)

WHITE_R, PIAL_R = 21.0, 24.0
WM, GM, CSF = 110.0, 70.0, 20.0
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def phantom():
    vox, n = 0.5, 120
    aff = np.diag([vox, vox, vox, 1.0])
    aff[:3, 3] = -(n - 1) * vox / 2
    ijk = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1)
    r = np.linalg.norm(ijk * vox + aff[:3, 3], axis=-1)
    img = np.where(r < WHITE_R, WM, np.where(r < PIAL_R, GM, CSF)).astype(np.float32)
    # A scanner's blur and noise. Without them every profile is nearly
    # identical, the robust spread collapses, and grid-dependent partial
    # volume at a perfectly sharp edge reads as an anomaly.
    img = gaussian_filter(img, 0.5) + np.random.default_rng(1).normal(0, 5, img.shape)
    img = img.astype(np.float32)
    k = 4000
    i = np.arange(k) + 0.5
    phi, theta = np.arccos(1 - 2 * i / k), np.pi * (1 + 5**0.5) * i
    u = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
    return img, aff, u


def test_profiles_read_wm_gm_csf_where_they_are(phantom):
    img, aff, u = phantom
    prof = sample_profiles(
        WHITE_R * u, PIAL_R * u, img, aff, ProfileSpec(tube_radius=0.0), device=CPU
    )
    mid = (prof.kind == 0) & (prof.columns > 0.3) & (prof.columns < 0.7)
    deep = (prof.kind < 0) & (prof.columns <= -1.0)
    far = (prof.kind > 0) & (prof.columns >= 1.0)
    assert np.median(prof.values[:, mid]) == pytest.approx(GM, abs=1)
    assert np.median(prof.values[:, deep]) == pytest.approx(WM, abs=1)
    assert np.median(prof.values[:, far]) == pytest.approx(CSF, abs=1)
    np.testing.assert_allclose(prof.thickness, PIAL_R - WHITE_R, atol=1e-4)


def test_tube_averaging_quiets_noise_without_moving_the_mean(phantom):
    img, aff, u = phantom
    noisy = img + np.random.default_rng(0).normal(0, 15, img.shape).astype(np.float32)
    line = sample_profiles(
        WHITE_R * u, PIAL_R * u, noisy, aff, ProfileSpec(tube_radius=0.0), device=CPU
    )
    tube = sample_profiles(
        WHITE_R * u, PIAL_R * u, noisy, aff, ProfileSpec(tube_radius=0.75), device=CPU
    )
    mid = (line.kind == 0) & (line.columns > 0.3) & (line.columns < 0.7)
    assert tube.values[:, mid].std() < 0.7 * line.values[:, mid].std()
    assert tube.values[:, mid].mean() == pytest.approx(line.values[:, mid].mean(), abs=1.5)


def test_a_pial_patch_run_out_through_csf_is_flagged_and_nothing_else_is(phantom):
    img, aff, u = phantom
    pial = PIAL_R * u
    patch = u[:, 2] > 0.95  # a cap at the top
    pial[patch] = (PIAL_R + 1.5) * u[patch]
    prof = sample_profiles(WHITE_R * u, pial, img, aff, device=CPU)
    sc = profile_scores(prof)
    # Scores are robust z / 4 against the brain: an ordinary vertex's 99th
    # percentile is ~2.3 / 4. What matters is that the patch is cleanly apart.
    assert np.percentile(sc["pial_out"][patch], 10) > 0.8
    assert np.percentile(sc["pial_out"][~patch], 99) < 0.7
    assert np.median(sc["worst"][patch]) > 0.8


def test_a_pial_patch_that_stopped_short_is_flagged(phantom):
    img, aff, u = phantom
    pial = PIAL_R * u
    patch = u[:, 2] > 0.95
    pial[patch] = (PIAL_R - 1.0) * u[patch]
    sc = profile_scores(sample_profiles(WHITE_R * u, pial, img, aff, device=CPU))
    assert np.percentile(sc["pial_short"][patch], 10) > 0.8
    assert np.percentile(sc["pial_short"][~patch], 99) < 0.7


def test_mm_mode_puts_pial_where_the_cortex_is_thick(phantom):
    img, aff, u = phantom
    prof = sample_profiles(
        WHITE_R * u, PIAL_R * u, img, aff, ProfileSpec(mode="mm", tube_radius=0.0), device=CPU
    )
    # GM until 3 mm from white, CSF after.
    at = {c: prof.values[:, i].mean() for i, c in enumerate(prof.columns)}
    assert at[1.5] == pytest.approx(GM, abs=2)
    assert at[4.5] == pytest.approx(CSF, abs=2)
    with pytest.raises(ValueError, match="fraction-mode"):
        profile_scores(prof)


def test_rows_walk_the_ring_rather_than_jump_across_it():
    # Folded "anatomy" (a wavy ring) whose smooth twin is a circle: ordering
    # by angle on the twin follows the ring, on the anatomy itself it jumps.
    t = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    wobble = 1 + 0.45 * np.sin(9 * t)
    folded = np.stack([30 * wobble * np.sin(t), np.zeros_like(t), -30 * wobble * np.cos(t)], 1)
    folded[:, 0] += 25 * np.sin(9 * t) * np.cos(t)  # fold over itself
    smooth = np.stack([30 * np.sin(t), np.zeros_like(t), -30 * np.cos(t)], 1)
    good = slab_contour_order(folded, smooth)
    step = np.linalg.norm(np.diff(folded[good], axis=0), axis=1)
    ring_step = np.median(np.linalg.norm(np.diff(folded, axis=0), axis=1))
    assert np.median(step) == pytest.approx(ring_step, rel=0.05)
    assert sorted(good.tolist()) == list(range(len(t)))


def test_sample_depths_matches_the_cpu_sampler_and_extends_past_the_ribbon(phantom):
    from fastfuncstuff.surface.profiles import sample_depths
    from fastfuncstuff.surface.sampling import VolumeSampler

    img, aff, u = phantom
    white, pial = WHITE_R * u[:200], PIAL_R * u[:200]
    fr = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])
    got = sample_depths(white, pial, img, aff, fr, device=CPU)
    pts = white[:, None, :] + fr[None, :, None] * (pial - white)[:, None, :]
    np.testing.assert_allclose(got, VolumeSampler(img, aff)(pts), atol=1e-2)
    # Past white is WM, past pial CSF -- the margins see beyond the ribbon.
    assert got[:, 0].mean() > got[:, 2].mean() > got[:, 4].mean()


def test_sample_depths_equivolume_and_time_as_channels(phantom):
    from fastfuncstuff.surface.profiles import equivolume_fraction, sample_depths

    img, aff, u = phantom
    white, pial = WHITE_R * u[:100], PIAL_R * u[:100]
    aw, ap = np.full(100, 1.0), np.full(100, 2.5)
    fr = np.linspace(-0.2, 1.2, 8)
    eq = sample_depths(white, pial, img, aff, fr, white_area=aw, pial_area=ap, device=CPU)
    rho = fr.copy()
    inside = (fr >= 0) & (fr <= 1)
    rho[inside] = equivolume_fraction(fr[inside], 1.0, 2.5)
    lin = sample_depths(white, pial, img, aff, rho, device=CPU)
    np.testing.assert_allclose(eq, lin, atol=1e-4)
    # A 4-D volume: every time point in one pass, identical to one at a time.
    series = np.stack([img, 2 * img, img - 5], axis=-1)
    four = sample_depths(white, pial, series, aff, fr, device=CPU)
    assert four.shape == (100, 8, 3)
    one = sample_depths(white, pial, img, aff, fr, device=CPU)
    np.testing.assert_allclose(four[..., 0], one, atol=1e-4)
    np.testing.assert_allclose(four[..., 1], 2 * one, atol=1e-3)
    np.testing.assert_allclose(four[..., 2], one - 5, atol=1e-3)
