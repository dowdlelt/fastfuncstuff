"""Model-order selection: does it recover a rank we planted?

The bar for this module is not "matches some other tool" but "finds the right answer on
data whose answer we know, and says so with a defensible ceiling". See
../fmri_wiki/notes/FSL clean-room policy.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastfuncstuff.decomposition.model_order import (
    effective_sample_size,
    effective_sample_size_from_resels,
    laplace_evidence_curve,
    mp_noise_level,
    mp_signal_count,
    select_model_order,
)


def _planted(n_time, n_vox, rank, snr, rng, smooth=None):
    """(time, vox) data = rank-`rank` signal + iid noise, returns its temporal spectrum."""
    x = rng.standard_normal((n_time, n_vox))
    if rank > 0:
        sources = rng.standard_normal((rank, n_vox))
        mixing = rng.standard_normal((n_time, rank))
        signal = mixing @ sources
        signal *= snr / np.sqrt(np.mean(signal**2))
        x = x + signal
    if smooth:
        # Crude spatial smoothing: a moving average along the voxel axis, which makes
        # neighbouring "voxels" dependent exactly the way real smoothing does.
        kern = np.ones(smooth) / smooth
        x = np.apply_along_axis(lambda r: np.convolve(r, kern, mode="same"), 1, x)
    x -= x.mean(axis=1, keepdims=True)
    cov = (x @ x.T) / n_vox
    return np.sort(np.linalg.eigvalsh(cov))[::-1]


@pytest.mark.parametrize("rank", [3, 8, 20])
def test_recovers_planted_rank(rank):
    rng = np.random.default_rng(0xC1EA + rank)
    ev = _planted(n_time=120, n_vox=4000, rank=rank, snr=3.0, rng=rng)
    res = select_model_order(ev, n_samples=4000, k_min=1)
    # Exact recovery is not the claim; being within a component or two of the planted
    # rank, and never wildly above it, is.
    assert abs(res.k - rank) <= 2, f"k={res.k} for planted rank {rank}"
    assert res.k_mp >= rank - 1, f"MP ceiling {res.k_mp} cut below the planted rank"


def test_pure_noise_selects_almost_nothing():
    # The failure this module exists to prevent: an evidence curve that keeps rising
    # through the noise bulk and returns a large k for data with no structure at all.
    rng = np.random.default_rng(7)
    ev = _planted(n_time=100, n_vox=5000, rank=0, snr=0.0, rng=rng)
    res = select_model_order(ev, n_samples=5000, k_min=1)
    assert res.k_mp <= 3, f"MP found {res.k_mp} signal directions in pure noise"
    assert res.k <= 5, f"selected k={res.k} on pure noise"


def test_mp_ceiling_binds():
    rng = np.random.default_rng(11)
    ev = _planted(n_time=100, n_vox=4000, rank=5, snr=3.0, rng=rng)
    capped = select_model_order(ev, n_samples=4000, use_mp_ceiling=True)
    uncapped = select_model_order(ev, n_samples=4000, use_mp_ceiling=False)
    assert capped.k <= capped.k_mp
    assert uncapped.k >= capped.k


def test_overstated_sample_size_is_what_inflates_k():
    """The documented failure mode: raw voxel count instead of effective sample size.

    This is the reason the module insists on effective_sample_size, so it is worth a test
    that the inflation is real and in the direction claimed.
    """
    rng = np.random.default_rng(3)
    n_vox = 6000
    ev = _planted(n_time=120, n_vox=n_vox, rank=6, snr=2.5, rng=rng, smooth=8)
    honest = select_model_order(ev, n_samples=effective_sample_size(n_vox, (8, 1, 1)))
    inflated = select_model_order(ev, n_samples=n_vox)
    assert inflated.k >= honest.k


def test_effective_sample_size_divides_by_the_resel():
    assert effective_sample_size(1000, (1.0, 1.0, 1.0)) == 1000
    assert effective_sample_size(1000, (2.0, 2.0, 1.0)) == 250
    # Sub-voxel smoothness is no smoothing, not a sample-size *increase*.
    assert effective_sample_size(1000, (0.5, 0.5, 0.5)) == 1000


def test_effective_sample_size_from_resels_cannot_exceed_the_voxel_count():
    """The product form must clamp too -- it is the one every caller uses.

    Bug of record: the per-axis form clamped at 1.0 per axis but the product form did
    not, so a sub-voxel resel (barely-smoothed data, which reads FWHM < 1 voxel on every
    axis) reported *more* independent samples than there were voxels, and inflated the
    selected ICA model order by ~15%.
    """
    assert effective_sample_size_from_resels(1000, 1.0) == 1000
    assert effective_sample_size_from_resels(1000, 4.0) == 250
    assert effective_sample_size_from_resels(1000, 0.83) == 1000
    assert effective_sample_size_from_resels(1000, 1e-9) == 1000
    with pytest.raises(ValueError):
        effective_sample_size(0, (1.0, 1.0, 1.0))


def test_mp_noise_level_is_robust_to_spikes():
    """The iterative bulk fit must not be dragged upward by strong signal eigenvalues."""
    rng = np.random.default_rng(5)
    ev_clean = _planted(n_time=100, n_vox=4000, rank=0, snr=0.0, rng=rng)
    sigma2_clean, edge_clean = mp_noise_level(ev_clean, 4000)

    spiked = ev_clean.copy()
    spiked[:6] += np.array([500.0, 400.0, 300.0, 200.0, 150.0, 100.0])
    spiked = np.sort(spiked)[::-1]
    sigma2_spiked, edge_spiked = mp_noise_level(spiked, 4000)

    # Six huge spikes must barely move the estimated noise floor.
    assert abs(sigma2_spiked - sigma2_clean) < 0.15 * sigma2_clean
    assert abs(edge_spiked - edge_clean) < 0.15 * edge_clean
    assert mp_signal_count(spiked, 4000)[0] == 6


def test_laplace_curve_shape_and_guards():
    rng = np.random.default_rng(13)
    ev = _planted(n_time=60, n_vox=2000, rank=4, snr=3.0, rng=rng)
    curve = laplace_evidence_curve(ev, n_samples=2000, k_min=1, k_max=20)
    assert curve.shape == (20,)
    assert np.isfinite(curve).all()
    # The peak should sit near the planted rank rather than at an endpoint.
    assert 1 <= int(np.argmax(curve)) + 1 <= 8

    # k_max beyond the spectrum is clamped, not an error.
    short = laplace_evidence_curve(ev, n_samples=2000, k_min=1, k_max=10_000)
    assert short.size == ev.size - 1
    assert laplace_evidence_curve(ev, 2000, k_min=50, k_max=10).size == 0


def test_unsorted_input_is_sorted_not_trusted():
    rng = np.random.default_rng(17)
    ev = _planted(n_time=80, n_vox=3000, rank=5, snr=3.0, rng=rng)
    shuffled = ev.copy()
    rng.shuffle(shuffled)
    assert select_model_order(shuffled, 3000).k == select_model_order(ev, 3000).k


def test_result_is_serialisable():
    rng = np.random.default_rng(19)
    ev = _planted(n_time=60, n_vox=2000, rank=4, snr=3.0, rng=rng)
    d = select_model_order(ev, 2000).as_dict()
    assert set(d) >= {"k", "k_mp", "sigma2", "lambda_plus", "n_samples", "log_evidence"}
    assert isinstance(d["log_evidence"], list)


def test_at_ceiling_flags_a_user_cap_not_the_mp_cap():
    """The warning must fire for a k_max that truncates, and stay quiet for the MP cap.

    The MP cap binding is the estimator working as designed -- if that raised a warning,
    it would raise one on every correct answer.
    """
    rng = np.random.default_rng(0xC1EA + 8)
    ev = _planted(n_time=120, n_vox=4000, rank=8, snr=3.0, rng=rng)

    natural = select_model_order(ev, n_samples=4000)
    assert natural.k == natural.k_mp
    assert natural.ceiling_source == "mp"
    assert not natural.at_ceiling, "MP cap must not raise the still-rising warning"

    truncated = select_model_order(ev, n_samples=4000, k_max=4)
    assert truncated.k == 4
    assert truncated.ceiling_source == "k_max"
    assert truncated.at_ceiling, "a k_max that cuts a rising curve must warn"
