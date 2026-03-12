"""
Tests for GLMsingle-style fracridge cross-validation.

Covers:
- single_trial_cv_helper with test_variant_idx (compare all fracs against OLS test betas)
- Correct frac selection direction: high-SNR → high frac, low-SNR → low frac
- _fit_ridge_multiple_fracs chunk_size path (CPU accumulation)
- Integration: nuisance projection + fracridge + CV selects sensible fracs
"""

import numpy as np
import pytest
import torch

from fastfuncsim.glm.ridge import _fit_ridge_multiple_fracs
from fastfuncsim.utils import get_device
from fastfuncsim.glm.xval import generate_cv_splits, single_trial_cv_helper


@pytest.fixture
def device():
    return get_device()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fracridge_variants(
    n_voxels: int = 80,
    n_trials: int = 48,
    n_conditions: int = 8,
    n_runs: int = 4,
    high_snr_n: int = 40,
    high_snr_signal: float = 5.0,
    low_snr_signal: float = 0.0,
    noise_scale: float = 1.0,
    fracs: list[float] | None = None,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list, np.ndarray]:
    """
    Build a synthetic beta tensor that mimics fracridge outputs.

    Returns
    -------
    beta_variants : (n_fracs, n_voxels, n_trials)
    trial_cond_ids : (n_trials,)
    trial_run_ids  : (n_trials,)
    cv_splits      : list of (train_runs, test_runs)
    fracs_np       : np.ndarray of frac values used
    """
    torch.manual_seed(seed)
    if fracs is None:
        fracs = [0.05, 0.25, 0.50, 0.75, 0.95, 1.0]
    fracs_np = np.array(fracs, dtype=np.float32)

    trials_per_run = n_trials // n_runs
    trial_cond_ids = torch.tile(torch.arange(n_conditions), (n_trials // n_conditions,))
    trial_run_ids = torch.repeat_interleave(torch.arange(n_runs), trials_per_run)

    # True condition effects: strong for high-SNR voxels, none for low-SNR
    cond_effects = torch.zeros(n_voxels, n_conditions)
    cond_effects[:high_snr_n] = torch.randn(high_snr_n, n_conditions) * high_snr_signal
    cond_effects[high_snr_n:] = torch.randn(n_voxels - high_snr_n, n_conditions) * low_snr_signal

    # OLS betas = true signal + full noise (frac=1.0)
    ols_betas = cond_effects[:, trial_cond_ids] + torch.randn(n_voxels, n_trials) * noise_scale

    # Simulate fracridge: frac × signal + frac × noise  (simplified linear model)
    beta_variants = torch.stack([
        cond_effects[:, trial_cond_ids] * f + torch.randn(n_voxels, n_trials) * noise_scale * f
        for f in fracs_np
    ])  # (n_fracs, n_voxels, n_trials)

    # Override last variant with the proper OLS betas
    beta_variants[-1] = ols_betas

    cv_splits = generate_cv_splits(n_runs, strategy=1)
    return beta_variants, trial_cond_ids, trial_run_ids, cv_splits, fracs_np


# ---------------------------------------------------------------------------
# test_variant_idx correctness
# ---------------------------------------------------------------------------

class TestTestVariantIdx:
    """Verify that test_variant_idx correctly routes test targets."""

    def test_high_snr_gets_higher_r2_for_high_fracs(self):
        """High-SNR voxels: R² should increase with frac when scored vs OLS targets."""
        bvars, cids, rids, splits, fracs = _make_fracridge_variants()
        n_fracs = len(fracs)

        result = single_trial_cv_helper(
            bvars, cids, rids, splits,
            test_variant_idx=n_fracs - 1,  # OLS as test target
            device=torch.device("cpu"),
            verbose=False,
        )
        r2 = result["r2"]  # (n_fracs, n_voxels)

        # High-SNR voxels (first 40): R² should be highest at frac=1.0
        # and increase roughly monotonically with frac
        high_snr_r2 = r2[:, :40].mean(dim=1)  # (n_fracs,) mean over high-SNR voxels
        low_snr_r2 = r2[:, 40:].mean(dim=1)

        print(f"  High-SNR mean R² by frac: {high_snr_r2.tolist()}")
        print(f"  Low-SNR mean R² by frac: {low_snr_r2.tolist()}")

        # High-SNR: frac=0.95 should beat frac=0.05 by a clear margin
        assert high_snr_r2[-2].item() > high_snr_r2[0].item() + 0.3, (
            f"High-SNR voxels: frac=0.95 R² ({high_snr_r2[-2]:.3f}) should well exceed "
            f"frac=0.05 R² ({high_snr_r2[0]:.3f})"
        )

    def test_frac_selection_direction(self):
        """CV-selected best frac is higher for high-SNR than low-SNR voxels."""
        bvars, cids, rids, splits, fracs = _make_fracridge_variants()
        n_fracs = len(fracs)

        result = single_trial_cv_helper(
            bvars, cids, rids, splits,
            test_variant_idx=n_fracs - 1,
            device=torch.device("cpu"),
            verbose=False,
        )
        r2 = result["r2"]  # (n_fracs, n_voxels)

        # Select best frac excluding OLS (last column) — GLMsingle pattern
        best_idx = r2[:-1].max(dim=0).indices  # (n_voxels,)
        best_fracs_all = fracs[best_idx.numpy()]

        high_snr_median = np.median(best_fracs_all[:40])
        low_snr_median = np.median(best_fracs_all[40:])

        print(f"  High-SNR median best frac: {high_snr_median:.3f}")
        print(f"  Low-SNR median best frac: {low_snr_median:.3f}")

        assert high_snr_median > low_snr_median, (
            f"High-SNR voxels should prefer higher fracs than low-SNR voxels: "
            f"{high_snr_median:.3f} vs {low_snr_median:.3f}"
        )

    def test_test_variant_idx_vs_same_variant(self):
        """
        Same-variant comparison (no test_variant_idx) fails to differentiate fracs.

        This documents the bug we fixed: after z-scoring, COD is scale-invariant,
        so all fracs get the same score when compared to themselves.  With
        test_variant_idx pointing to OLS, the amplitude of the prediction vs the
        fixed-scale OLS target provides the discriminating signal.
        """
        bvars, cids, rids, splits, fracs = _make_fracridge_variants()
        n_fracs = len(fracs)

        # GLMsingle pattern: compare to OLS test betas
        res_glmsingle = single_trial_cv_helper(
            bvars, cids, rids, splits,
            test_variant_idx=n_fracs - 1,
            device=torch.device("cpu"),
            verbose=False,
        )

        # Broken pattern: each variant vs itself (no z-score)
        res_self = single_trial_cv_helper(
            bvars, cids, rids, splits,
            test_variant_idx=None,
            zscore_by_run=False,
            device=torch.device("cpu"),
            verbose=False,
        )

        r2_glmsingle = res_glmsingle["r2"]  # (n_fracs, n_voxels)
        r2_self = res_self["r2"]

        # GLMsingle: variance in R² across fracs should be substantial for high-SNR voxels
        glmsingle_frac_spread = r2_glmsingle[:, :40].mean(dim=1).max() - r2_glmsingle[:, :40].mean(dim=1).min()

        # Self-comparison: COD is scale-invariant → all fracs get nearly the same score
        self_frac_spread = r2_self[:, :40].mean(dim=1).max() - r2_self[:, :40].mean(dim=1).min()

        print(f"  GLMsingle frac spread (high-SNR): {glmsingle_frac_spread:.3f}")
        print(f"  Self-comparison frac spread (high-SNR): {self_frac_spread:.3f}")

        assert glmsingle_frac_spread > self_frac_spread + 0.1, (
            "GLMsingle pattern should have larger R² spread across fracs than "
            "same-variant comparison"
        )

    def test_test_variant_idx_chunked_matches_full(self):
        """Chunked processing with test_variant_idx gives identical results."""
        bvars, cids, rids, splits, fracs = _make_fracridge_variants(n_voxels=100)
        n_fracs = len(fracs)

        r_full = single_trial_cv_helper(
            bvars, cids, rids, splits,
            test_variant_idx=n_fracs - 1,
            chunk_size=None,
            device=torch.device("cpu"),
            verbose=False,
        )
        r_chunk = single_trial_cv_helper(
            bvars, cids, rids, splits,
            test_variant_idx=n_fracs - 1,
            chunk_size=13,
            device=torch.device("cpu"),
            verbose=False,
        )

        torch.testing.assert_close(r_full["r2"], r_chunk["r2"], atol=1e-5, rtol=1e-5)

    def test_ols_variant_gets_best_r2_for_high_snr(self):
        """
        OLS (frac=1) itself should get the highest R² when scored against OLS test
        betas, because training-OLS condition means best match test-OLS betas.
        After excluding frac=1 from selection, frac=0.95 should be the runner-up.
        """
        bvars, cids, rids, splits, fracs = _make_fracridge_variants(
            high_snr_signal=8.0, noise_scale=0.5)  # very high SNR
        n_fracs = len(fracs)

        result = single_trial_cv_helper(
            bvars, cids, rids, splits,
            test_variant_idx=n_fracs - 1,
            device=torch.device("cpu"),
            verbose=False,
        )
        r2 = result["r2"]  # (n_fracs, n_voxels)

        high_snr_r2_mean = r2[:, :40].mean(dim=1)

        # OLS should be at or near the top for high-SNR voxels
        ols_r2 = high_snr_r2_mean[-1].item()
        max_r2 = high_snr_r2_mean.max().item()
        assert ols_r2 >= max_r2 - 0.02, (
            f"OLS (frac=1) should have best/near-best R² for high-SNR voxels "
            f"(OLS={ols_r2:.3f}, max={max_r2:.3f})"
        )


# ---------------------------------------------------------------------------
# _fit_ridge_multiple_fracs chunk_size path
# ---------------------------------------------------------------------------

class TestFitRidgeMultipleFracsChunking:
    """Verify chunk_size path of _fit_ridge_multiple_fracs."""

    def test_chunked_matches_unchunked(self, device):
        """chunk_size path produces identical coefs as single-pass path."""
        torch.manual_seed(7)
        n_tp, n_features, n_voxels = 80, 10, 50
        fracs = np.array([0.05, 0.25, 0.5, 0.75, 0.95, 1.0], dtype=np.float32)

        X = torch.randn(n_tp, n_features, device=device)
        y = torch.randn(n_tp, n_voxels, device=device)

        # Single pass (no chunking)
        coefs_full = _fit_ridge_multiple_fracs(X, y, fracs, device, chunk_size=None)

        # Chunked pass (chunk_size=7, so 8 chunks)
        coefs_chunked = _fit_ridge_multiple_fracs(X, y.cpu(), fracs, device, chunk_size=7)

        torch.testing.assert_close(
            coefs_full.cpu(), coefs_chunked.cpu(), atol=1e-4, rtol=1e-4)

    def test_chunked_output_on_cpu(self, device):
        """chunk_size path returns a CPU tensor regardless of compute device."""
        torch.manual_seed(11)
        X = torch.randn(60, 8, device=device)
        y = torch.randn(60, 30)  # CPU input
        fracs = np.array([0.5, 1.0], dtype=np.float32)

        coefs = _fit_ridge_multiple_fracs(X, y, fracs, device, chunk_size=10)
        assert coefs.device.type == "cpu", (
            f"chunk_size path should return CPU tensor, got {coefs.device}")

    def test_single_pass_output_on_device(self, device):
        """No-chunk path returns tensor on compute device."""
        X = torch.randn(60, 8, device=device)
        y = torch.randn(60, 20, device=device)
        fracs = np.array([0.5, 1.0], dtype=np.float32)

        coefs = _fit_ridge_multiple_fracs(X, y, fracs, device, chunk_size=None)
        assert coefs.device.type == device.type, (
            f"Single-pass path should return tensor on {device}, got {coefs.device}")

    def test_frac_monotonicity_shrinkage(self, device):
        """Higher frac → larger coefficient norms (less shrinkage)."""
        torch.manual_seed(3)
        n_tp, n_features, n_voxels = 100, 5, 20
        fracs = np.array([0.05, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)

        X = torch.randn(n_tp, n_features, device=device)
        y = torch.randn(n_tp, n_voxels, device=device)

        coefs = _fit_ridge_multiple_fracs(X, y, fracs, device)
        # coefs: (n_features, n_fracs, n_voxels)
        norms = coefs.norm(dim=0).mean(dim=1)  # (n_fracs,) mean L2-norm

        # Monotonically non-decreasing (fracridge definition)
        for i in range(len(fracs) - 1):
            assert norms[i] <= norms[i + 1] + 1e-4, (
                f"frac={fracs[i]:.2f} norm ({norms[i]:.4f}) should be ≤ "
                f"frac={fracs[i+1]:.2f} norm ({norms[i+1]:.4f})"
            )


# ---------------------------------------------------------------------------
# Integration: projection + fracridge + frac selection
# ---------------------------------------------------------------------------

class TestFracridgeIntegration:
    """End-to-end test: nuisance projection → fracridge → CV frac selection."""

    def test_projection_then_fracridge_selects_sensible_fracs(self, device):
        """
        After projecting out polynomials, fracridge with GLMsingle CV should
        select higher fracs for high-SNR voxels than for noise-only voxels.

        Uses a well-spaced block-style design (>5 TRs between events) to ensure
        fracridge estimates are well-conditioned and frac discrimination works.
        """
        from fastfuncsim.glm.core import construct_polynomial_matrix
        from fastfuncsim.glm.xval import project_out_nuisance_per_run

        torch.manual_seed(99)
        n_runs, n_tp_run = 4, 120       # 120 TPs per run gives ample spacing
        n_tp_total = n_runs * n_tp_run
        n_conditions, trials_per_run = 4, 5
        n_voxels_high, n_voxels_low = 30, 30
        n_voxels = n_voxels_high + n_voxels_low
        spacing = n_tp_run // (n_conditions * trials_per_run + 1)  # ~5 TPs apart

        run_starts = [i * n_tp_run for i in range(n_runs)]

        # Spread trials evenly within each run with ≥5 TP spacing
        n_trials = n_conditions * trials_per_run * n_runs
        design = torch.zeros(n_tp_total, n_trials)
        trial_cond_ids = torch.zeros(n_trials, dtype=torch.long)
        trial_run_ids = torch.zeros(n_trials, dtype=torch.long)
        trial_idx = 0
        for run_i in range(n_runs):
            for cond_i in range(n_conditions):
                for t in range(trials_per_run):
                    tp = run_i * n_tp_run + (cond_i * trials_per_run + t) * spacing
                    design[tp, trial_idx] = 1.0
                    trial_cond_ids[trial_idx] = cond_i
                    trial_run_ids[trial_idx] = run_i
                    trial_idx += 1

        # True betas: condition-level signal for high-SNR voxels (same beta
        # per condition across all trials → condition average is non-zero →
        # cross-validation can predict test betas from training averages).
        true_cond_betas = torch.zeros(n_voxels, n_conditions)
        true_cond_betas[:n_voxels_high] = torch.randn(n_voxels_high, n_conditions) * 5.0
        # Expand to per-trial: (n_voxels, n_trials)
        true_betas = true_cond_betas[:, trial_cond_ids]

        # Data = task signal + linear drift + small noise
        noise = torch.randn(n_voxels, n_tp_total) * 0.3
        drift = torch.zeros(n_tp_total)
        for r in range(n_runs):
            t_run = torch.linspace(-1, 1, n_tp_run)
            drift[r * n_tp_run:(r + 1) * n_tp_run] = t_run * 3.0
        data = true_betas @ design.T + drift.unsqueeze(0) + noise

        nuisance_per_run = [
            construct_polynomial_matrix(n_tp_run, max_degree=1, device=torch.device("cpu"))
            for _ in range(n_runs)
        ]
        data_clean, design_clean = project_out_nuisance_per_run(
            data, design, nuisance_per_run, run_starts, device=device)

        fracs = np.arange(0.05, 1.01, 0.05, dtype=np.float32)
        coefs = _fit_ridge_multiple_fracs(
            design_clean.to(device), data_clean.T, fracs, device, chunk_size=None)
        beta_variants = coefs.permute(1, 2, 0)  # (n_fracs, n_voxels, n_trials)

        cv_splits = generate_cv_splits(n_runs, strategy=1)
        xval = single_trial_cv_helper(
            beta_variants, trial_cond_ids, trial_run_ids, cv_splits,
            test_variant_idx=len(fracs) - 1,
            device=device,
            verbose=False,
        )
        r2 = xval["r2"]  # (n_fracs, n_voxels)

        # Select best frac excluding OLS (last column)
        best_idx = r2[:-1].max(dim=0).indices.cpu().numpy()
        best_fracs_arr = fracs[best_idx]

        high_snr_median = float(np.median(best_fracs_arr[:n_voxels_high]))
        low_snr_median = float(np.median(best_fracs_arr[n_voxels_high:]))

        print(f"  High-SNR median best frac: {high_snr_median:.3f}")
        print(f"  Low-SNR median best frac: {low_snr_median:.3f}")

        # High-SNR voxels should prefer substantially less regularization
        assert high_snr_median > low_snr_median + 0.05, (
            "After projection+fracridge, high-SNR voxels should prefer less "
            f"regularization than low-SNR voxels ({high_snr_median:.3f} vs "
            f"{low_snr_median:.3f})"
        )

    def test_four_runs_frac_selection_works(self, device):
        """
        Regression test: 4-run fracridge (the 'n_runs=2 workaround' case).

        Previously a comment in test_ridge_comprehensive.py noted using only 2
        runs to 'avoid ridge.py bug'.  This test verifies that the standard
        n_runs=4 case at least runs without errors, produces finite outputs, and
        gives non-degenerate frac selection (not all voxels pick the same frac).
        """
        from fastfuncsim.glm.core import construct_polynomial_matrix
        from fastfuncsim.glm.xval import project_out_nuisance_per_run

        torch.manual_seed(17)
        n_runs, n_tp_run = 4, 100
        n_tp = n_runs * n_tp_run
        n_conditions, trials_per_run = 4, 4
        n_voxels = 40
        spacing = n_tp_run // (n_conditions * trials_per_run + 1)  # ≥5 TPs between events
        run_starts = [i * n_tp_run for i in range(n_runs)]

        n_trials = n_conditions * trials_per_run * n_runs
        design = torch.zeros(n_tp, n_trials)
        trial_cond_ids = torch.zeros(n_trials, dtype=torch.long)
        trial_run_ids = torch.zeros(n_trials, dtype=torch.long)
        trial_idx = 0
        for run_i in range(n_runs):
            for cond_i in range(n_conditions):
                for t in range(trials_per_run):
                    tp = run_i * n_tp_run + (cond_i * trials_per_run + t) * spacing
                    design[tp, trial_idx] = 1.0
                    trial_cond_ids[trial_idx] = cond_i
                    trial_run_ids[trial_idx] = run_i
                    trial_idx += 1

        # Mix of high- and low-SNR voxels.
        # Use condition-level betas (same per condition across trials) so the
        # condition-average training betas can predict test betas.
        true_cond_betas = torch.zeros(n_voxels, n_conditions)
        true_cond_betas[:n_voxels // 2] = torch.randn(n_voxels // 2, n_conditions) * 3.0
        true_betas = true_cond_betas[:, trial_cond_ids]  # (n_voxels, n_trials)
        data = true_betas @ design.T + torch.randn(n_voxels, n_tp) * 0.5

        nuisance_per_run = [
            construct_polynomial_matrix(n_tp_run, max_degree=1, device=torch.device("cpu"))
            for _ in range(n_runs)
        ]
        data_clean, design_clean = project_out_nuisance_per_run(
            data, design, nuisance_per_run, run_starts, device=device)

        fracs = np.array([0.05, 0.25, 0.50, 0.75, 0.95, 1.0], dtype=np.float32)
        coefs = _fit_ridge_multiple_fracs(
            design_clean.to(device), data_clean.T, fracs, device)
        beta_variants = coefs.permute(1, 2, 0)  # (n_fracs, n_voxels, n_trials)

        cv_splits = generate_cv_splits(n_runs, strategy=1)
        xval = single_trial_cv_helper(
            beta_variants, trial_cond_ids, trial_run_ids, cv_splits,
            test_variant_idx=len(fracs) - 1,
            device=device,
            verbose=False,
        )
        r2 = xval["r2"]  # (n_fracs, n_voxels)

        # Outputs must be finite
        assert torch.all(torch.isfinite(r2)), "R² values must be finite for 4-run case"

        # Frac selection must not be degenerate — at least 2 distinct fracs chosen
        best_idx = r2[:-1].max(dim=0).indices.cpu().numpy()
        n_unique_fracs = len(np.unique(best_idx))
        assert n_unique_fracs > 1, (
            f"Frac selection should not be degenerate (all same frac) for 4 runs; "
            f"got {n_unique_fracs} unique frac indices"
        )

        # High-SNR voxels should prefer higher fracs than low-SNR voxels
        high_snr_median = float(np.median(fracs[best_idx[:n_voxels // 2]]))
        low_snr_median = float(np.median(fracs[best_idx[n_voxels // 2:]]))
        print(f"  High-SNR median frac: {high_snr_median:.3f}, Low-SNR: {low_snr_median:.3f}")
        assert high_snr_median >= low_snr_median, (
            f"High-SNR voxels should prefer ≥ frac than low-SNR: "
            f"{high_snr_median:.3f} vs {low_snr_median:.3f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
