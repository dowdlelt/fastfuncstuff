"""Gaussian-Gamma mixture: does it separate activation from background we planted?

Every test builds a map whose background and activation are known by construction, so the
bar is recovery of the truth rather than agreement with another implementation.
"""

from __future__ import annotations

import numpy as np
import torch

from fastfuncstuff.decomposition.mixture import (
    MIN_GAMMA_SHAPE,
    batch_mixture_zscores,
    fit_gaussian_gamma_mixture,
    mixture_zscores_signed,
)

CPU = torch.device("cpu")


def _map_with_activation(n_vox, frac_pos, frac_neg, noise_sd=1.0, offset=6.0, seed=0):
    """A background Gaussian plus a known fraction of positive/negative activated voxels."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, noise_sd, n_vox)
    n_pos, n_neg = int(frac_pos * n_vox), int(frac_neg * n_vox)
    truth = np.zeros(n_vox, dtype=int)
    if n_pos:
        x[:n_pos] = offset + rng.gamma(3.0, 1.0, n_pos)
        truth[:n_pos] = 1
    if n_neg:
        x[n_pos : n_pos + n_neg] = -(offset + rng.gamma(3.0, 1.0, n_neg))
        truth[n_pos : n_pos + n_neg] = -1
    return torch.tensor(x[None, :], dtype=torch.float64, device=CPU), truth


def test_recovers_the_activated_fraction():
    x, truth = _map_with_activation(20000, frac_pos=0.05, frac_neg=0.03, seed=1)
    fit = fit_gaussian_gamma_mixture(x)

    assert bool(fit["converged"][0])
    assert abs(float(fit["pi_pos"][0]) - 0.05) < 0.02
    assert abs(float(fit["pi_neg"][0]) - 0.03) < 0.02
    assert abs(float(fit["pi_noise"][0]) - 0.92) < 0.03


def test_posterior_separates_activation_from_background():
    x, truth = _map_with_activation(20000, frac_pos=0.05, frac_neg=0.05, seed=2)
    _, p_signal, _ = batch_mixture_zscores(x)
    p = p_signal[0].numpy()

    active = truth != 0
    # Activated voxels get high posterior, background voxels low. This is the whole job.
    assert p[active].mean() > 0.9
    assert p[~active].mean() < 0.05
    # And thresholding the posterior recovers the planted set well.
    detected = p > 0.5
    tp = int((detected & active).sum())
    fp = int((detected & ~active).sum())
    assert tp / int(active.sum()) > 0.9, "missed too much planted activation"
    assert fp / int((~active).sum()) < 0.02, "too many background voxels called active"


def test_null_map_finds_almost_no_activation():
    """A pure-noise component must not be handed an activation class."""
    rng = np.random.default_rng(3)
    x = torch.tensor(rng.normal(0, 1, (1, 20000)), dtype=torch.float64)
    _, p_signal, meta = batch_mixture_zscores(x)
    assert meta[0]["pi_noise"] > 0.95
    assert float((p_signal[0] > 0.5).float().mean()) < 0.01


def test_background_z_is_unit_normal_whatever_the_activated_fraction():
    """The scale of the background z-distribution must not depend on how much signal is
    present. A component that is 15% activated and one that is not activated at all should
    report their background voxels on the same scale, or thresholds mean different things
    on different components.

    Note this is an invariance of the *z-scores*, not of ``sigma_noise``: the latter lives
    in standardised units and legitimately moves with the standardising scale, which
    divides back out of the z computation.
    """
    stats = []
    for frac in (0.0, 0.05, 0.15):
        x, truth = _map_with_activation(20000, frac_pos=frac, frac_neg=0.0, offset=10.0, seed=4)
        z, _, _ = batch_mixture_zscores(x)
        bg = z[0].numpy()[truth == 0]
        stats.append((bg.mean(), bg.std()))

    for mean, sd in stats:
        assert abs(mean) < 0.1, f"background z not centred: {mean}"
        assert abs(sd - 1.0) < 0.1, f"background z not unit-scale: {sd}"
    # And it is stable across the three, not merely near 1 in each.
    assert max(s for _, s in stats) - min(s for _, s in stats) < 0.05


def test_separation_floor_keeps_activation_off_the_background():
    """Without a separation constraint the Gammas park on the Gaussian's own tails.

    At a 2-SD floor a pure-noise map gets ~12% of its voxels assigned to activation classes,
    because a Gamma sitting 2 SDs out is a good local fit to a Gaussian tail. The default
    floor has to be clear of that.
    """
    rng = np.random.default_rng(3)
    x = torch.tensor(rng.normal(0, 1, (1, 20000)), dtype=torch.float64)

    fit = fit_gaussian_gamma_mixture(x)
    assert float(fit["pi_pos"][0] + fit["pi_neg"][0]) < 0.01
    # The background keeps its own variance rather than surrendering the tails.
    assert abs(float(fit["var_noise"][0]) - 1.0) < 0.1


def test_z_scores_are_in_units_of_fitted_background():
    x, truth = _map_with_activation(20000, frac_pos=0.04, frac_neg=0.04, seed=5)
    z, _, meta = batch_mixture_zscores(x)
    zz = z[0].numpy()

    background = zz[truth == 0]
    assert abs(background.mean()) < 0.1
    assert abs(background.std() - 1.0) < 0.15
    # Activation sits far out in those units, with the sign preserved.
    assert zz[truth == 1].mean() > 4
    assert zz[truth == -1].mean() < -4


def test_gamma_shape_constraint_holds():
    """Fitted tails must stay unimodal with an interior mode (shape >= MIN_GAMMA_SHAPE)."""
    x, _ = _map_with_activation(20000, frac_pos=0.08, frac_neg=0.08, seed=6)
    fit = fit_gaussian_gamma_mixture(x)
    for m, v in (("mu_pos", "var_pos"), ("mu_neg", "var_neg")):
        shape = float(fit[m][0]) ** 2 / float(fit[v][0])
        assert shape >= MIN_GAMMA_SHAPE - 1e-6, (m, shape)


def test_degenerate_map_does_not_explode():
    """A collapsing background must not divide the z-scores by ~0.

    Flooring only the Gamma variances and not the background's was a real bug: it produced
    z-maps inflated by orders of magnitude on degenerate input.
    """
    for vals in (
        np.zeros(5000),
        np.concatenate([np.zeros(4900), np.full(100, 50.0)]),
        np.full(5000, 3.14),
    ):
        z, p, meta = mixture_zscores_signed(vals)
        assert np.isfinite(z).all(), "non-finite z on degenerate input"
        assert np.isfinite(p).all()
        assert np.abs(z).max() < 1e4, f"z blew up to {np.abs(z).max():.3g}"
        assert 0.0 <= p.min() <= p.max() <= 1.0


def test_batched_matches_single_component():
    maps = torch.cat([_map_with_activation(8000, 0.05, 0.03, seed=s)[0] for s in (7, 8, 9)], dim=0)
    z, p, meta = batch_mixture_zscores(maps)
    assert z.shape == maps.shape and p.shape == maps.shape
    assert len(meta) == 3

    for i in range(3):
        z1, p1, m1 = mixture_zscores_signed(maps[i].numpy())
        assert np.allclose(z[i].numpy(), z1, atol=1e-4)
        assert np.allclose(p[i].numpy(), p1, atol=1e-4)
        assert abs(m1["pi_noise"] - meta[i]["pi_noise"]) < 1e-6


def test_posteriors_sum_to_one():
    x, _ = _map_with_activation(5000, 0.05, 0.05, seed=10)
    fit = fit_gaussian_gamma_mixture(x)
    total = fit["p_noise"] + fit["p_pos"] + fit["p_neg"]
    assert torch.allclose(total, torch.ones_like(total), atol=1e-6)
    weights = fit["pi_noise"] + fit["pi_pos"] + fit["pi_neg"]
    assert torch.allclose(weights, torch.ones_like(weights), atol=1e-6)
