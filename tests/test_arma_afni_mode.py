"""Tests for -afni_mode: AFNI-faithful REML divergences.

Default FFS behaviour is the exact dense covariance + a slightly more thorough
search. set_afni_mode(True) switches the small (a,b) divergences to AFNI's
choices (banded R, corcut grid filter). Each test resets the global afterwards.
"""

import math

import pytest
import torch

from fastfuncstuff.glm.arma import (
    _afni_bmax,
    build_arma11_covariance,
    build_arma11_covariance_batch,
    set_afni_mode,
)

DEVICE = torch.device("cpu")
CORCUT = 1e-4


def _lam(a: float, b: float) -> float:
    return ((b + a) * (1 + a * b)) / (1 + 2 * a * b + b**2)


@pytest.fixture(autouse=True)
def _reset_mode():
    """Guarantee the module global is restored after each test."""
    yield
    set_afni_mode(False)


class TestBandedR:
    def test_default_is_dense(self):
        a, b, n = 0.6, 0.2, 60
        R = build_arma11_covariance(a, b, n, DEVICE)
        # Far off-diagonal is tiny but nonzero in the exact dense Toeplitz.
        assert R[0, n - 1].item() != 0.0

    def test_afni_mode_truncates_at_bmax(self):
        a, b, n = 0.6, 0.2, 60
        lam = _lam(a, b)
        bmax = _afni_bmax(a, lam, CORCUT)
        assert bmax >= 2  # sanity: this (a,b) is well above white

        set_afni_mode(True)
        R = build_arma11_covariance(a, b, n, DEVICE)
        # Last kept off-diagonal is bmax; beyond it is exactly zero.
        assert R[0, bmax].item() != 0.0
        assert R[0, bmax + 1].item() == 0.0
        # Within the band the values still match the exact correlation.
        assert abs(R[0, 1].item() - lam) < 1e-6

    def test_bmax_formula_matches_afni(self):
        # remla.c:692: bmax = 1 + ceil(log(corcut/alam)/log|rho|)
        a, lam = 0.5, 0.4
        expected = 1 + int(math.ceil(math.log(CORCUT / lam) / math.log(abs(a))))
        assert _afni_bmax(a, lam, CORCUT) == expected
        # Near-white → identity (bmax 0).
        assert _afni_bmax(0.5, 1e-6, CORCUT) == 0
        # Pure MA(1) (a=0) → bmax 1.
        assert _afni_bmax(0.0, 0.3, CORCUT) == 1

    def test_batch_banding_matches_scalar(self):
        a_grid = torch.tensor([0.3, 0.7])
        b_grid = torch.tensor([0.0, 0.2])
        set_afni_mode(True)
        R_batch, params, _ = build_arma11_covariance_batch(
            a_grid, b_grid, n=50, device=DEVICE
        )
        for k in range(params.shape[0]):
            a, b = params[k, 0].item(), params[k, 1].item()
            R_scalar = build_arma11_covariance(a, b, n=50, device=DEVICE)
            assert torch.allclose(R_batch[k], R_scalar, atol=1e-6)


class TestGridFilter:
    # a=0.5, b≈-0.5 gives 0 < lam < corcut (a near-white point).
    A, B = 0.5, -0.49995

    def test_near_white_kept_by_default(self):
        assert 0 < _lam(self.A, self.B) < CORCUT
        R = build_arma11_covariance(self.A, self.B, n=20, device=DEVICE)
        assert R is not None  # FFS evaluates it

    def test_near_white_dropped_in_afni_mode(self):
        set_afni_mode(True)
        R = build_arma11_covariance(self.A, self.B, n=20, device=DEVICE)
        assert R is None  # AFNI folds it into the a=b=0 case

    def test_exact_white_always_kept(self):
        # lam == 0 (b == -a) stays valid even in afni_mode.
        set_afni_mode(True)
        R = build_arma11_covariance(0.5, -0.5, n=20, device=DEVICE)
        assert R is not None
        assert torch.allclose(R, torch.eye(20), atol=1e-6)
