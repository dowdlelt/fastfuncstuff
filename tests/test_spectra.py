"""stats/spectra: one scaling for every estimator, and aliasing."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.stats import spectra

TR = 0.8
N = 400


def _rows(seed: int = 0, n: int = 64) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 2.0, (n, N))
    return torch.as_tensor(x - x.mean(axis=1, keepdims=True))


@pytest.mark.parametrize("method", spectra.METHODS)
def test_density_integrates_to_the_variance(method):
    """The contract every map leans on: sum(psd) * df is the variance. White
    noise, so every estimator is unbiased for it up to sampling error."""
    rows = _rows()
    p = spectra.psd(rows, TR, method=method, nw=3.0, segments=4)
    f = spectra.frequencies(method, N, TR, segments=4)
    assert p.shape == (rows.shape[0], f.size)
    df = f[1] - f[0]
    total = p.sum(dim=1).numpy() * df
    var = rows.var(dim=1, unbiased=False).numpy()
    assert np.median(total / var) == pytest.approx(1.0, abs=0.08)


def test_untapered_periodogram_is_parseval_exact():
    """With a flat taper the scaling is exactly Parseval, which pins the
    factor of two and the unpaired Nyquist bin."""
    rows = _rows(1, 4)
    flat = torch.full((1, N), 1.0 / np.sqrt(N), dtype=rows.dtype)
    p = spectra.psd(rows, TR, window=flat)
    df = 1.0 / (N * TR)
    np.testing.assert_allclose(
        p.sum(dim=1).numpy() * df, rows.var(dim=1, unbiased=False).numpy(), rtol=1e-10
    )


@pytest.mark.parametrize("method", spectra.METHODS)
def test_a_sinusoid_peaks_in_its_own_bin(method):
    t = np.arange(N) * TR
    f0 = 0.27
    rows = torch.as_tensor(np.sin(2 * np.pi * f0 * t)[None, :])
    p = spectra.psd(rows, TR, method=method, segments=4)[0].numpy()
    f = spectra.frequencies(method, N, TR, segments=4)
    assert abs(f[int(np.argmax(p))] - f0) <= (f[1] - f[0])


def test_multitaper_and_welch_lower_the_variance_of_white_noise():
    rows = _rows(2, 256)
    spread = {
        m: float((spectra.psd(rows, TR, method=m, nw=3.0) / 1.0).std(dim=0).mean())
        for m in spectra.METHODS
    }
    assert spread["multitaper"] < 0.7 * spread["periodogram"]
    assert spread["welch"] < 0.7 * spread["periodogram"]


def test_alias_folds_a_heartbeat_below_nyquist():
    # TR 2 s samples at 0.5 Hz: 1.1 Hz lands at 0.1 Hz.
    assert float(spectra.alias(1.1, 2.0)) == pytest.approx(0.1)
    # Below Nyquist nothing moves.
    assert float(spectra.alias(0.3, 0.8)) == pytest.approx(0.3)
    # Folding from above Nyquist: 1 Hz at TR 0.8 (fs 1.25) -> 0.25 Hz.
    assert float(spectra.alias(1.0, 0.8)) == pytest.approx(0.25)


def test_dpss_count_follows_nw():
    assert spectra.tapers("multitaper", 100, nw=3.0).shape == (5, 100)
    w = spectra.tapers("multitaper", 100, nw=2.0)
    np.testing.assert_allclose((w * w).sum(axis=1), 1.0)
