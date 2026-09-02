"""Variance normalisation: does it recover a noise level we planted?

The claim under test is that dividing by *noise* std beats dividing by total std, and
that the degrees-of-freedom correction removes a real bias. Both are checkable against
data whose noise level we set ourselves.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.decomposition.varnorm import (
    apply_noise_std_map,
    noise_std_map,
    variance_normalize,
)

CPU = torch.device("cpu")


def _data(n_vox, n_time, noise_sd, rank=5, signal_amp=0.0, seed=0):
    """(n_vox, n_time) = shared low-rank signal + per-voxel iid noise of known sd."""
    rng = np.random.default_rng(seed)
    noise_sd = np.broadcast_to(np.asarray(noise_sd, dtype=float), (n_vox,))
    x = rng.standard_normal((n_vox, n_time)) * noise_sd[:, None]
    if signal_amp:
        ts = rng.standard_normal((rank, n_time))
        load = rng.standard_normal((n_vox, rank))
        x = x + signal_amp * (load @ ts)
    return torch.tensor(x, dtype=torch.float64, device=CPU)


def test_recovers_a_planted_noise_level():
    """With no signal, the estimated noise std must match the sd we generated with."""
    true_sd = 2.5
    x = _data(n_vox=20000, n_time=120, noise_sd=true_sd, seed=1)
    est, const, n_const = noise_std_map(x)
    assert n_const == 0 and not const.any()
    assert abs(float(est.mean()) - true_sd) < 0.03 * true_sd


def test_bias_shrinks_as_voxels_outnumber_timepoints():
    """The documented assumption, pinned as a test rather than left as a footnote.

    The top r principal directions of a finite sample capture more than their share of
    noise variance (Marchenko-Pastur spread), so the residual under-estimates the noise
    when V is close to T. fMRI runs far from that regime; this records where the estimator
    stops being trustworthy.
    """
    true_sd, n_time, r = 2.5, 200, 30
    ratios = []
    for n_vox in (400, 2000, 20000):
        x = _data(n_vox=n_vox, n_time=n_time, noise_sd=true_sd, seed=1)
        est, _, _ = noise_std_map(x, signal_rank=r)
        ratios.append(float(est.mean()) / true_sd)

    assert ratios[0] < ratios[1] < ratios[2], ratios
    assert ratios[0] < 0.92, "V~2T should show visible downward bias"
    assert ratios[-1] > 0.97, "at fMRI scale the bias should be under a few percent"


def test_dof_correction_removes_a_real_bias():
    """Without the correction the residual std is biased low by sqrt((T-r-1)/(T-1)).

    This test would fail if the correction were dropped, which is the point of having it.
    """
    n_time, r, true_sd = 60, 30, 1.0
    x = _data(n_vox=30000, n_time=n_time, noise_sd=true_sd, seed=2)
    est, _, _ = noise_std_map(x, signal_rank=r)

    corrected = float(est.mean())
    uncorrected = corrected * np.sqrt((n_time - r - 1) / (n_time - 1))

    assert abs(corrected - true_sd) < 0.04 * true_sd
    # The uncorrected value is materially off at this rank -- ~30% here.
    assert uncorrected < 0.85 * true_sd


def test_noise_std_ignores_signal_that_total_std_does_not():
    """The whole reason for this module: total std tracks signal, noise std should not."""
    n_vox, n_time = 20000, 120
    quiet = _data(n_vox, n_time, noise_sd=1.0, signal_amp=0.0, seed=3)
    loud = _data(n_vox, n_time, noise_sd=1.0, signal_amp=3.0, rank=5, seed=3)

    total_quiet = float(torch.std(quiet, dim=1, unbiased=True).mean())
    total_loud = float(torch.std(loud, dim=1, unbiased=True).mean())
    noise_quiet = float(noise_std_map(quiet)[0].mean())
    noise_loud = float(noise_std_map(loud)[0].mean())

    # Adding strong signal inflates total std a lot...
    assert total_loud > 1.5 * total_quiet
    # ...but must leave the noise estimate essentially where it was.
    assert abs(noise_loud - noise_quiet) < 0.10 * noise_quiet


def test_normalisation_equalises_heteroscedastic_noise():
    """Voxels generated with wildly different noise levels end up on a common scale."""
    n_vox, n_time = 20000, 120
    sds = np.linspace(0.5, 20.0, n_vox)
    x = _data(n_vox, n_time, noise_sd=sds, signal_amp=0.0, seed=4)

    before = torch.std(x, dim=1, unbiased=True)
    out, n_const = variance_normalize(x)
    after = torch.std(out, dim=1, unbiased=True)

    assert n_const == 0
    # A 40x spread in noise scale collapses to near-uniform.
    assert float(before.max() / before.min()) > 20
    assert float(after.max() / after.min()) < 1.6


def test_normalisation_preserves_snr_differences():
    """Noise normalisation must NOT flatten total variance -- that is total-std's job.

    Equal signal on top of unequal noise means unequal SNR, and a high-SNR voxel should
    come out of normalisation with more variance than a low-SNR one. Dividing by total
    std would erase exactly that, which is why this module does not.
    """
    n_vox, n_time = 20000, 120
    sds = np.linspace(0.5, 20.0, n_vox)
    x = _data(n_vox, n_time, noise_sd=sds, signal_amp=2.0, rank=5, seed=4)

    out, _ = variance_normalize(x)
    after = torch.std(out, dim=1, unbiased=True)

    # Voxels are ordered by increasing noise, so SNR falls across the array.
    high_snr = float(after[: n_vox // 10].mean())
    low_snr = float(after[-n_vox // 10 :].mean())
    assert high_snr > 1.5 * low_snr, (high_snr, low_snr)

    # Total-std normalisation, by contrast, would drive both to exactly 1.
    total_norm = x / torch.std(x, dim=1, keepdim=True, unbiased=True)
    tn = torch.std(total_norm, dim=1, unbiased=True)
    assert abs(float(tn.mean()) - 1.0) < 1e-6


def test_constant_voxels_are_zeroed_not_amplified():
    """A flat voxel must not be divided by its own near-zero spread into unit variance.

    This is the failure recorded in the wiki: constant voxels inside the mask become
    perfect unit-variance "signal" and the mixture model fits them.
    """
    x = _data(n_vox=20000, n_time=80, noise_sd=1.0, seed=5)
    x[7, :] = 3.14  # exactly constant
    x[11, :] = 0.0

    est, const, n_const = noise_std_map(x)
    assert n_const == 2
    assert bool(const[7]) and bool(const[11])

    out, _ = variance_normalize(x)
    assert torch.all(out[7, :] == 0)
    assert torch.all(out[11, :] == 0)
    assert torch.isfinite(out).all()
    # The ordinary voxels are untouched by the presence of the constant ones.
    assert float(torch.std(out[0, :], unbiased=True)) == pytest.approx(1.0, abs=0.25)


def test_map_can_be_estimated_once_and_reused():
    """Splitting estimate from apply is what keeps multiple runs on a common scale."""
    n_vox, n_time = 20000, 100
    sds = np.linspace(1.0, 5.0, n_vox)
    a = _data(n_vox, n_time, noise_sd=sds, seed=6)
    b = _data(n_vox, n_time, noise_sd=sds, seed=7)

    est, const, _ = noise_std_map((a + b) / 2.0)
    out_a = apply_noise_std_map(a, est, const)
    out_b = apply_noise_std_map(b, est, const)

    # Both runs land on the same scale, rather than each being normalised to its own.
    sa = float(torch.std(out_a, dim=1, unbiased=True).mean())
    sb = float(torch.std(out_b, dim=1, unbiased=True).mean())
    assert abs(sa - sb) < 0.1 * sa
    assert out_a.shape == a.shape


def test_rank_is_clamped_to_leave_residual_dof():
    """An absurd rank request must not consume every degree of freedom."""
    x = _data(n_vox=50, n_time=10, noise_sd=1.0, seed=8)
    est, _, _ = noise_std_map(x, signal_rank=999)
    assert torch.isfinite(est).all()
    assert float(est.min()) > 0

    tiny = _data(n_vox=5, n_time=3, noise_sd=1.0, seed=9)
    est2, _, _ = noise_std_map(tiny)
    assert est2.shape == (5,)
    assert torch.isfinite(est2).all()
