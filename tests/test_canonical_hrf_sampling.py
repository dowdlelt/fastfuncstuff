"""get_canonical_hrf samples at exact multiples of the TR it is given.

It used a fixed 0.1 s grid strided by int(tr / 0.1): TR 0.72 came out every
0.7 s, and the sub-0.1 s microtime step single-trial designs pass (TR 1.75 ->
0.0972 s) divided by zero, so ffs_reml -single_trials crashed on such data.
"""

import numpy as np
import pytest
import torch
from scipy import stats

from fastfuncstuff.design.hrf import get_canonical_hrf, get_canonical_hrf_library

CPU = torch.device("cpu")


def _double_gamma(t):
    return stats.gamma.pdf(t, 6.0) - stats.gamma.pdf(t, 16.0) / 6.0


@pytest.mark.parametrize("tr", [2.0, 1.5, 0.72, 1.75 / 18, 1 / 12])
def test_samples_fall_on_the_tr_grid(tr):
    h = get_canonical_hrf(0.0, tr, device=CPU).numpy()
    t = np.arange(len(h)) * tr
    assert t[-1] < 32.0 <= t[-1] + tr + 1e-9
    peak = _double_gamma(np.arange(0, 32, tr / max(1, int(np.ceil(tr / 0.1 - 1e-9))))).max()
    np.testing.assert_allclose(h, _double_gamma(t) / peak, atol=1e-6)


def test_library_uses_the_same_grid():
    lib = get_canonical_hrf_library(0.0, 0.72, n_hrfs=3, device=CPU).numpy()
    assert lib.shape == (3, len(get_canonical_hrf(0.0, 0.72, device=CPU)))


def test_stimulus_duration_is_not_truncated():
    # int(0.3 / 0.1) == 2: a 0.3 s boxcar used to be 0.2 s. The response to it
    # is the impulse response summed over 3 consecutive 0.1 s steps.
    h3 = get_canonical_hrf(0.3, 0.1, device=CPU).numpy()
    h0 = get_canonical_hrf(0.0, 0.1, device=CPU).numpy()
    box = np.convolve(h0, np.ones(3))[: len(h0)]
    np.testing.assert_allclose(h3, box / box.max(), atol=1e-6)
