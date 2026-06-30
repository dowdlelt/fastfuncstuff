"""Tests for tau-based censoring and run-boundary unification in the ARMA(1,1)
covariance builder.

Two distinct concepts (see build_censor_run_info):
- Run boundary: a hard cut. Correlation must be *exactly* zero across runs.
- Censoring: a dropped "bad" timepoint. Correlation reaches *across* the hole,
  weakened by the true time gap (lag-2 instead of lag-1 for the flanking pair).
"""

import torch

from fastfuncstuff.glm.arma import (
    build_arma11_covariance,
    build_arma11_covariance_batch,
    build_censor_run_info,
)

DEVICE = torch.device("cpu")


def _lam(a: float, b: float) -> float:
    return ((b + a) * (1 + a * b)) / (1 + 2 * a * b + b**2)


class TestTauCensoring:
    def test_tau_none_matches_plain_toeplitz(self):
        """tau=None must reproduce the original |i-j| Toeplitz exactly."""
        a, b, n = 0.5, 0.2, 30
        R_default = build_arma11_covariance(a, b, n, DEVICE)
        R_tau = build_arma11_covariance(
            a, b, n, DEVICE, tau=torch.arange(n)
        )
        assert torch.allclose(R_default, R_tau, atol=1e-7)

    def test_censored_point_creates_lag2_gap(self):
        """A censored TR makes its two survivors sit at lag-2, not lag-1."""
        a, b = 0.6, 0.2
        lam = _lam(a, b)
        # Original TRs [0,1,2,3,4]; TR 2 censored → survivors [0,1,3,4].
        tau = torch.tensor([0, 1, 3, 4])
        R = build_arma11_covariance(a, b, n=4, device=DEVICE, tau=tau)

        # Survivors at tau=1 (index 1) and tau=3 (index 2): real gap is 2.
        # Correlation reaches across the hole, decayed to lag-2 = lam * a.
        assert abs(R[1, 2].item() - lam * a) < 1e-5
        # Adjacent survivors with no gap stay lag-1 = lam.
        assert abs(R[0, 1].item() - lam) < 1e-5
        assert abs(R[2, 3].item() - lam) < 1e-5

    def test_batch_matches_scalar_with_tau(self):
        a_grid = torch.tensor([0.3, 0.6])
        b_grid = torch.tensor([0.0, 0.2])
        tau = torch.tensor([0, 1, 3, 4, 7])
        R_batch, params, _ = build_arma11_covariance_batch(
            a_grid, b_grid, n=5, device=DEVICE, tau=tau
        )
        for k in range(params.shape[0]):
            a, b = params[k, 0].item(), params[k, 1].item()
            R_scalar = build_arma11_covariance(a, b, n=5, device=DEVICE, tau=tau)
            assert torch.allclose(R_batch[k], R_scalar, atol=1e-6)


class TestRunBoundaryExactZero:
    def test_cross_run_is_exactly_zero(self):
        """Run boundary is a hard cut: cross-run entries must be exactly 0."""
        a, b, n = 0.9, 0.2, 20  # high a → would leak badly without a hard mask
        run_starts = [0, 10]
        R = build_arma11_covariance(a, b, n, DEVICE, run_starts=run_starts)
        # Block (run0=rows 0:10, run1=rows 10:20) off-diagonal must be 0.
        assert torch.count_nonzero(R[:10, 10:]).item() == 0
        # Within-run structure is preserved.
        assert abs(R[0, 1].item() - _lam(a, b)) < 1e-5


class TestBuildCensorRunInfo:
    def test_no_censoring_passthrough(self):
        starts, tau = build_censor_run_info([0, 100], n_total=200, good_list=None)
        assert starts == [0, 100]
        assert tau is None

    def test_censoring_remaps_runs_and_tau(self):
        # Two 5-TR runs: original starts [0, 5], n_total=10.
        # Censor TR 2 (run0) and TR 7 (run1) → keep [0,1,3,4, 5,6,8,9].
        good = [0, 1, 3, 4, 5, 6, 8, 9]
        starts, tau = build_censor_run_info([0, 5], n_total=10, good_list=good)
        # Retained-space run starts: run0 begins at retained idx 0,
        # run1 begins at retained idx 4 (the 5th survivor).
        assert starts == [0, 4]
        # tau = within-run original index. Run0: [0,1,3,4]; run1: [0,1,3,4].
        assert tau.tolist() == [0, 1, 3, 4, 0, 1, 3, 4]

    def test_helper_drives_block_diagonal_with_censoring(self):
        """End-to-end: helper output must give exact block-diagonal R with
        correct within-run lag-2 gaps."""
        a, b = 0.7, 0.1
        lam = _lam(a, b)
        good = [0, 1, 3, 4, 5, 6, 8, 9]
        starts, tau = build_censor_run_info([0, 5], n_total=10, good_list=good)
        n = len(good)
        R = build_arma11_covariance(
            a, b, n, DEVICE, run_starts=starts, tau=tau
        )
        # Cross-run exactly zero (rows 0:4 vs 4:8).
        assert torch.count_nonzero(R[:4, 4:]).item() == 0
        # Within run0: survivors at tau=1,3 (indices 1,2) → lag-2.
        assert abs(R[1, 2].item() - lam * a) < 1e-5
