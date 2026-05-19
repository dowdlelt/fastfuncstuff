"""Correctness tests for fastfuncstuff.stats.permutation."""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import ttest_1samp, ttest_ind

from fastfuncstuff.stats.permutation import (
    count_unique_label_perms,
    generate_label_swaps,
    generate_sign_flips,
    one_sample_t_perm,
    two_sample_t_perm,
)


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Observed t matches scipy
# ---------------------------------------------------------------------------

def test_one_sample_observed_t_matches_scipy():
    rng = np.random.default_rng(0)
    n, v = 50, 100
    y = rng.normal(size=(n, v)).astype(np.float32) + 0.4
    signs = generate_sign_flips(n, n_perms=4, rng=rng)
    out = one_sample_t_perm(y, signs, device=_device(), show_progress=False)
    # Row 0 is the identity (all +1), so out.t[0] should match ttest_1samp.
    obs = out.t[0].numpy()
    ref = ttest_1samp(y, 0.0, axis=0).statistic
    np.testing.assert_allclose(obs, ref, rtol=1e-4, atol=1e-4)


def test_two_sample_observed_t_matches_scipy():
    rng = np.random.default_rng(1)
    nA, nB, v = 25, 35, 80
    yA = rng.normal(size=(nA, v)).astype(np.float32)
    yB = rng.normal(size=(nB, v)).astype(np.float32) + 0.5
    y = np.concatenate([yA, yB], axis=0)
    group = np.concatenate([np.ones(nA, dtype=np.int8), np.zeros(nB, dtype=np.int8)])
    swaps = generate_label_swaps(group, n_perms=4, rng=rng)
    out = two_sample_t_perm(y, swaps, device=_device(), show_progress=False)
    obs = out.t[0].numpy()
    # scipy.ttest_ind: A first, B second; equal_var=True is the pooled t-test.
    ref = ttest_ind(yA, yB, axis=0, equal_var=True).statistic
    np.testing.assert_allclose(obs, ref, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Exhaustive sign-flip enumeration
# ---------------------------------------------------------------------------

def test_generate_sign_flips_exhaustive():
    """For small N, the generator returns the full 2**N enumeration."""
    rng = np.random.default_rng(0)
    signs = generate_sign_flips(n_trials=6, n_perms=10000, rng=rng)
    assert signs.shape == (64, 6)
    assert np.array_equal(signs[0], np.ones(6, dtype=np.int8))
    # All rows distinct
    rows = {tuple(r) for r in signs}
    assert len(rows) == 64


# ---------------------------------------------------------------------------
# Run-block label swaps stay within blocks
# ---------------------------------------------------------------------------

def test_label_swaps_respect_blocks():
    rng = np.random.default_rng(42)
    blocks = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    group = np.array([1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0], dtype=np.int8)
    swaps = generate_label_swaps(group, n_perms=200, rng=rng, blocks=blocks)
    # Within each block, the number of A's is preserved.
    for b in np.unique(blocks):
        idx = np.where(blocks == b)[0]
        a_count = int(group[idx].sum())
        for row in swaps:
            assert int(row[idx].sum()) == a_count


def test_count_unique_label_perms_blocked():
    # Two blocks of 4 trials, 2 A's per block → C(4,2)² = 36 unique perms.
    blocks = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    group = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int8)
    assert count_unique_label_perms(group, blocks) == 36


# ---------------------------------------------------------------------------
# Under H0, the rank of the observed t is approximately uniform
# ---------------------------------------------------------------------------

def test_null_distribution_uniform_one_sample():
    """When the data are null (mean 0), observed-vs-permuted ranks are uniform."""
    rng = np.random.default_rng(7)
    n, v = 30, 500
    y = rng.normal(size=(n, v)).astype(np.float32)  # zero-mean, no effect
    signs = generate_sign_flips(n, n_perms=200, rng=rng)
    out = one_sample_t_perm(y, signs, device=_device(), show_progress=False)
    # For each voxel: rank of |t_obs| in {|t_perm|}.  Should be ~uniform in [1, P].
    t = out.t.abs().numpy()
    obs = t[0]
    rank = (t >= obs[None, :]).sum(axis=0)  # 1..P
    p_unc = rank / t.shape[0]
    # 5% tail rate should be near 0.05.  Tolerate Monte-Carlo wiggle.
    rate = float((p_unc <= 0.05).mean())
    assert 0.025 < rate < 0.085, f"FPR was {rate:.3f}, expected ~0.05"


def test_two_sample_welch_matches_scipy():
    """Welch path returns scipy.ttest_ind(equal_var=False) for the observed row."""
    rng = np.random.default_rng(11)
    nA, nB, v = 30, 20, 60
    yA = rng.normal(size=(nA, v)).astype(np.float32) * 1.5     # higher variance
    yB = rng.normal(size=(nB, v)).astype(np.float32) * 0.5 + 0.3
    y = np.concatenate([yA, yB], axis=0)
    group = np.concatenate([np.ones(nA, dtype=np.int8), np.zeros(nB, dtype=np.int8)])
    swaps = generate_label_swaps(group, n_perms=4, rng=rng)
    out_w = two_sample_t_perm(y, swaps, device=_device(), show_progress=False, welch=True)
    out_p = two_sample_t_perm(y, swaps, device=_device(), show_progress=False, welch=False)
    ref_welch = ttest_ind(yA, yB, axis=0, equal_var=False).statistic
    ref_pool = ttest_ind(yA, yB, axis=0, equal_var=True).statistic
    np.testing.assert_allclose(out_w.t[0].numpy(), ref_welch, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out_p.t[0].numpy(), ref_pool, rtol=1e-4, atol=1e-4)
    # Sanity: Welch differs from pooled when variances are unequal.
    assert not np.allclose(out_w.t[0].numpy(), out_p.t[0].numpy(), atol=1e-3)


def test_keep_perm_data_one_sample():
    """When keep_perm_data=True, extras let us recover variance and recompute t."""
    rng = np.random.default_rng(13)
    n, v = 25, 40
    y = rng.normal(size=(n, v)).astype(np.float32) + 0.2
    signs = generate_sign_flips(n, n_perms=8, rng=rng)
    out = one_sample_t_perm(y, signs, device=_device(),
                            show_progress=False, keep_perm_data=True)
    m = out.extras["perm_means"].numpy()
    sum_y2 = out.extras["sum_y2"].numpy()
    var = (sum_y2[None, :] - n * m * m) / (n - 1)
    t_reconstructed = m * np.sqrt(n) / np.sqrt(np.clip(var, 1e-30, None))
    np.testing.assert_allclose(t_reconstructed, out.t.numpy(), rtol=1e-4, atol=1e-4)


def test_alternative_detected_one_sample():
    """When the data have a real effect, low p-values are common."""
    rng = np.random.default_rng(9)
    n, v = 60, 200
    y = rng.normal(size=(n, v)).astype(np.float32) + 0.5  # solid effect
    signs = generate_sign_flips(n, n_perms=200, rng=rng)
    out = one_sample_t_perm(y, signs, device=_device(), show_progress=False)
    t = out.t.numpy()
    obs = t[0]
    rank = (t >= obs[None, :]).sum(axis=0)
    p_unc = rank / t.shape[0]
    # Almost every voxel should reject at p < 0.05 with effect 0.5 in N=60.
    assert (p_unc < 0.05).mean() > 0.95
