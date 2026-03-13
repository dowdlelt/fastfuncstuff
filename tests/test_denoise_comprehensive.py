"""
Comprehensive tests for denoise.py with progressive coverage.

Test layers:
1. Small: Unit tests for core functions (noise pool selection, PC extraction)
2. Medium: Sub-workflow tests (noise PC evaluation, CV validation)
3. Large/E2E: Full pipeline tests with ground truth verification

Uses realistic fMRI simulation to verify:
- Noise pool selection identifies low-signal voxels
- Sequential PC denoising improves model fit
- Cross-validation prevents overfitting
"""

import numpy as np
import pytest
import torch

from fastfuncstuff.denoise.sequential import (
    _compute_local_run_starts,
    select_noise_pool_voxels,
)
from fastfuncstuff.glm.core import construct_polynomial_matrix
from fastfuncstuff.utils import get_device


@pytest.fixture
def device():
    return get_device()


# ============================================================================
# Layer 1: Small Tests - Unit tests for core functions
# ============================================================================

class TestDenoiseCoreFunctions:
    """Test core denoising functions."""

    def test_compute_local_run_starts_basic(self):
        """Test local run starts computation."""
        # Simple case: runs [0, 2, 4] from a subset of 3 runs
        run_indices = [0, 2, 4]
        run_starts_global = [0, 100, 200, 300, 400, 500]
        n_timepoints = 600  # Total timepoints

        local_starts = _compute_local_run_starts(run_indices, run_starts_global, n_timepoints)

        # First run should start at 0
        assert local_starts[0] == 0
        # Second run should start at 100 (relative position)
        assert local_starts[1] == 100
        # Third run should start at 200 (relative position)
        assert local_starts[2] == 200

    def test_compute_local_run_starts_single_run(self):
        """Test local run starts with single run."""
        run_indices = [2]
        run_starts_global = [0, 100, 200, 300]
        n_timepoints = 400

        local_starts = _compute_local_run_starts(run_indices, run_starts_global, n_timepoints)

        # Single run should always start at 0
        assert len(local_starts) == 1
        assert local_starts[0] == 0

    def test_select_noise_pool_voxels_basic(self, device):
        """Test noise pool selection with R² values."""
        _n_voxels = 100

        # Create R² values where first 50 have high R² (signal)
        # and last 50 have low R² (noise pool)
        r2 = torch.cat([
            torch.ones(50, device=device) * 0.5,  # High R² voxels
            torch.ones(50, device=device) * 0.05,  # Low R² voxels
        ])

        noise_mask, criteria_mask = select_noise_pool_voxels(
            r2=r2,
            threshold=0.1,  # Voxels below 0.1 are noise pool
            min_noise_voxels=10,
            max_noise_fraction=0.95,
        )

        # Should select roughly the low R² voxels
        n_selected = noise_mask.sum().item()
        assert n_selected == 50, f"Expected 50 voxels, got {n_selected}"

        # Selected voxels should have lower R² than non-selected
        r2_noise = r2[noise_mask].mean().item()
        r2_criteria = r2[criteria_mask].mean().item()

        assert r2_noise < r2_criteria, \
            f"Noise pool R² ({r2_noise:.3f}) should be less than criteria R² ({r2_criteria:.3f})"

    def test_select_noise_pool_voxels_min_constraint_error(self, device):
        """Test that noise pool selection raises error when min constraint can't be met."""
        n_voxels = 100

        # All voxels have high R² (no clear noise pool)
        r2 = torch.ones(n_voxels, device=device) * 0.5

        # Should raise error because can't meet minimum
        with pytest.raises(ValueError, match="Noise pool has only 0 voxels"):
            select_noise_pool_voxels(
                r2=r2,
                threshold=0.1,  # Would select 0 voxels
                min_noise_voxels=20,  # But need at least 20
                max_noise_fraction=0.95,
            )

    def test_extract_noise_pcs_multiple_runs(self, device):
        """
        extract_noise_pcs_per_run returns one tensor per run, each with shape
        (n_tp_run, n_components).  Components should be independent per run.
        """
        from fastfuncstuff.denoise.sequential import extract_noise_pcs_per_run

        n_runs, n_tp_run = 3, 80
        n_voxels, n_noise = 100, 40
        data = torch.randn(n_voxels, n_runs * n_tp_run, device=device)
        noise_pool_mask = torch.zeros(n_voxels, dtype=torch.bool, device=device)
        noise_pool_mask[:n_noise] = True
        run_starts = [i * n_tp_run for i in range(n_runs)]

        pcs_per_run = extract_noise_pcs_per_run(
            data, run_starts, noise_pool_mask,
            max_components=5, device=device, verbose=False)

        assert len(pcs_per_run) == n_runs, "One PC tensor per run expected"
        for r, pcs in enumerate(pcs_per_run):
            assert pcs.shape[0] == n_tp_run, (
                f"Run {r}: expected {n_tp_run} timepoints, got {pcs.shape[0]}")
            assert pcs.shape[1] <= 5, "At most max_components PCs"
            assert torch.all(torch.isfinite(pcs)), f"Run {r} PCs must be finite"


# ============================================================================
# Layer 2: Medium Tests - Sub-workflow tests
# ============================================================================

class TestDenoiseSubWorkflows:
    """Test denoising sub-workflows."""

    def test_evaluate_noise_pcs_with_cv(self, device):
        """
        cross_validate_noise_pcs returns an (n_voxels, n_pc_counts) R² map
        and an aggregated (n_pc_counts,) array.  All values should be finite.
        """
        from fastfuncstuff.denoise.sequential import cross_validate_noise_pcs, extract_noise_pcs_per_run

        torch.manual_seed(0)
        n_runs, n_tp_run, tr = 3, 60, 2.0
        n_voxels, n_noise = 50, 20
        n_tp_total = n_runs * n_tp_run
        run_starts = [i * n_tp_run for i in range(n_runs)]
        data = torch.randn(n_voxels, n_tp_total, device=device)
        design = torch.randn(n_tp_total, 4, device=device)

        noise_mask = torch.zeros(n_voxels, dtype=torch.bool, device=device)
        noise_mask[:n_noise] = True
        noise_pcs_out = extract_noise_pcs_per_run(
            data, run_starts, noise_mask, max_components=5, device=device, verbose=False)
        assert isinstance(noise_pcs_out, list)
        noise_pcs = noise_pcs_out

        nuisance = [construct_polynomial_matrix(n_tp_run, max_degree=1, device=device)
                    for _ in range(n_runs)]

        r2_maps, r2_agg = cross_validate_noise_pcs(
            data=data,
            design_matrix=design,
            noise_pcs=noise_pcs,
            run_starts=run_starts,
            tr=tr,
            max_components=3,
            nuisance=nuisance,
            device=device,
            verbose=False,
        )

        assert r2_maps.shape == (n_voxels, 4), (
            f"Expected (n_voxels, max_components+1) = ({n_voxels}, 4), got {r2_maps.shape}")
        assert r2_agg.shape == (4,)
        assert np.all(np.isfinite(r2_maps)), "R² maps must be finite"
        assert np.all(np.isfinite(r2_agg)), "Aggregated R² must be finite"

    def test_sequential_pc_selection(self, device):
        """
        select_optimal_pcs returns an index into 0..n_pc_counts-1 and a
        criteria mask.  When PC-0 is clearly the best, index 0 is selected.
        """
        from fastfuncstuff.denoise.sequential import select_optimal_pcs

        n_voxels, n_pc_counts = 200, 6

        # Deterministic: first 100 voxels exceed threshold at PC index 0 only;
        # last 100 voxels are always below threshold.
        r2_maps = np.zeros((n_voxels, n_pc_counts))
        r2_maps[:100, 0] = 0.5  # criteria voxels: best at PC 0

        optimal, criteria_mask = select_optimal_pcs(r2_maps, threshold=0.1)

        assert optimal == 0, f"Expected 0 PCs to be optimal, got {optimal}"
        assert criteria_mask.sum() == 100, (
            f"Expected 100 criteria voxels, got {criteria_mask.sum()}")

    def test_criteria_voxel_selection(self, device):
        """
        select_optimal_pcs uses voxels that exceed the threshold in ANY PC
        count as criteria voxels.  Threshold=0 means all voxels are criteria.
        """
        from fastfuncstuff.denoise.sequential import select_optimal_pcs

        n_voxels, n_pc_counts = 100, 4
        # Only voxels 0..19 exceed 0.1 in at least one PC count
        r2_maps = np.zeros((n_voxels, n_pc_counts))
        r2_maps[:20, 2] = 0.3   # these 20 have R² > 0.1 at PC count 2
        r2_maps[20:, :] = 0.0   # remaining 80 voxels never exceed threshold

        optimal, criteria_mask = select_optimal_pcs(r2_maps, threshold=0.1)

        assert criteria_mask.sum() == 20, (
            f"Expected 20 criteria voxels (R²>0.1), got {criteria_mask.sum()}")
        assert optimal == 2, (
            f"Expected PC count 2 to be selected (highest median for criteria), "
            f"got {optimal}")


class TestExtractNoisePCs:
    """Test noise PC extraction from data"""

    def test_extract_noise_pcs_with_mask(self, device):
        """Test PC extraction with noise pool mask"""
        from fastfuncstuff.denoise.sequential import extract_noise_pcs_per_run

        n_voxels = 100
        n_timepoints = 200
        n_noise_voxels = 50

        # Create data with structured noise in noise pool
        data = torch.randn(n_voxels, n_timepoints, device=device) * 10 + 100

        # Add common noise component to noise pool voxels
        noise_component = torch.randn(1, n_timepoints, device=device) * 5
        data[:n_noise_voxels, :] += noise_component

        # Noise pool mask (first 50 voxels)
        noise_mask = torch.zeros(n_voxels, dtype=torch.bool, device=device)
        noise_mask[:n_noise_voxels] = True

        run_starts = [0]

        # Extract PCs
        pcs_per_run = extract_noise_pcs_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=5,
            variance_threshold=0.95,
            device=device
        )

        # Should return list of PCs (one per run)
        assert len(pcs_per_run) == 1, "Should have 1 run of PCs"
        pcs = pcs_per_run[0]

        # PCs should capture the common noise structure
        assert pcs.shape[0] == n_timepoints, f"PCs should have {n_timepoints} timepoints"
        assert pcs.shape[1] <= 5, f"Should have at most 5 PCs, got {pcs.shape[1]}"

    def test_extract_noise_pcs_returns_empty_for_no_noise(self, device):
        """Test that empty noise pool returns empty PCs"""
        from fastfuncstuff.denoise.sequential import extract_noise_pcs_per_run

        n_voxels = 100
        n_timepoints = 200

        data = torch.randn(n_voxels, n_timepoints, device=device)

        # Empty noise pool
        noise_mask = torch.zeros(n_voxels, dtype=torch.bool, device=device)
        _run_starts = [0]

        # Should handle empty noise pool - need to include last run start
        run_starts_full = [0, n_timepoints]  # Need end point for last run

        # Since we have no noise voxels, function may raise error or return empty
        try:
            pcs_per_run = extract_noise_pcs_per_run(
                data=data,
                run_starts=run_starts_full,
                noise_pool_mask=noise_mask,
                max_components=5,
                device=device
            )
            # If it succeeds, should have empty PCs for the run
            assert len(pcs_per_run) == 1
            assert pcs_per_run[0].shape[1] == 0, "Empty noise pool should return 0 PCs"
        except (ValueError, RuntimeError):
            # Also acceptable to raise error for empty noise pool
            pass


class TestCrossValidateNoisePCs:
    """Test cross-validation for noise PC selection"""

    def test_cross_validate_noise_pcs_returns_arrays(self, device):
        """Test that CV returns R² maps and summary"""
        from fastfuncstuff.denoise.sequential import cross_validate_noise_pcs

        n_voxels = 50
        n_timepoints = 200
        n_runs = 2
        max_components = 3
        n_trials = 5

        # Create design matrix
        design_matrix = torch.randn(n_timepoints * n_runs, n_trials, device=device)

        data = torch.randn(n_voxels, n_timepoints * n_runs, device=device)
        run_starts = [0, n_timepoints]

        # Create simple noise PCs
        noise_pcs = []
        for _run_idx in range(n_runs):
            pc = torch.randn(n_timepoints, max_components, device=device)
            noise_pcs.append(pc)

        # Cross-validate (testing just 0, 1, 2, 3 PCs)
        r2_maps, r2_summary = cross_validate_noise_pcs(
            data=data,
            design_matrix=design_matrix,
            noise_pcs=noise_pcs,
            run_starts=run_starts,
            tr=2.0,
            max_components=max_components,
            device=device,
            verbose=False
        )

        # Check return types
        assert isinstance(r2_maps, np.ndarray), "r2_maps should be numpy array"
        assert isinstance(r2_summary, np.ndarray), "r2_summary should be numpy array"

        # Check shapes
        assert r2_maps.shape == (n_voxels, max_components + 1), \
            f"r2_maps should be ({n_voxels}, {max_components + 1})"
        assert r2_summary.shape == (max_components + 1,), \
            f"r2_summary should have {max_components + 1} elements"

    def test_cross_validate_noise_pcs_with_design_matrix(self, device):
        """Test CV with explicit design matrix"""
        from fastfuncstuff.denoise.sequential import cross_validate_noise_pcs

        n_voxels = 30
        n_timepoints = 100
        n_runs = 2
        n_trials = 5
        max_components = 2

        # Create design matrix
        design_matrix = torch.randn(n_timepoints * n_runs, n_trials, device=device)

        data = torch.randn(n_voxels, n_timepoints * n_runs, device=device)
        run_starts = [0, n_timepoints]

        # Create noise PCs
        noise_pcs = [
            torch.randn(n_timepoints, max_components, device=device),
            torch.randn(n_timepoints, max_components, device=device),
        ]

        # Cross-validate with design matrix
        r2_maps, r2_summary = cross_validate_noise_pcs(
            data=data,
            design_matrix=design_matrix,
            noise_pcs=noise_pcs,
            run_starts=run_starts,
            tr=2.0,
            max_components=max_components,
            device=device,
            verbose=False
        )

        # Should return valid R² values
        assert np.all(np.isfinite(r2_maps)), "R² maps should be finite"
        assert np.all(np.isfinite(r2_summary)), "R² summary should be finite"


class TestSelectOptimalPCs:
    """Test optimal PC selection from CV results"""

    def test_select_optimal_pcs_returns_integer(self, device):
        """Test that optimal PC selection returns integer"""
        from fastfuncstuff.denoise.sequential import select_optimal_pcs

        # Create mock R² maps (n_voxels, n_pc_counts)
        n_voxels = 50
        max_components = 10

        # Each column is R² for that PC count across all voxels
        r2_maps = np.random.rand(n_voxels, max_components + 1) * 0.3
        # Make higher PC counts better
        for i in range(max_components + 1):
            r2_maps[:, i] += i * 0.02

        # Select optimal
        optimal_n_pcs, criteria_mask = select_optimal_pcs(
            r2_maps=r2_maps,
            threshold=0.0,
            metric="median"
        )

        assert isinstance(optimal_n_pcs, (int, np.integer)), "Should return integer"
        assert 0 <= optimal_n_pcs <= max_components, f"Should be in range [0, {max_components}]"
        assert criteria_mask.shape == (n_voxels,), "Criteria mask should match n_voxels"

    def test_select_optimal_pcs_selects_best(self, device):
        """Test that function selects PC count with highest median R²"""
        from fastfuncstuff.denoise.sequential import select_optimal_pcs

        n_voxels = 50

        # Create R² maps where PC=5 is clearly best
        r2_maps = np.zeros((n_voxels, 11))
        r2_maps[:, 0] = 0.10  # 0 PCs
        r2_maps[:, 1] = 0.15  # 1 PC
        r2_maps[:, 2] = 0.18
        r2_maps[:, 3] = 0.20
        r2_maps[:, 4] = 0.22
        r2_maps[:, 5] = 0.30  # 5 PCs - best
        r2_maps[:, 6:] = 0.25  # 6+ PCs - worse

        optimal_n_pcs, criteria_mask = select_optimal_pcs(
            r2_maps=r2_maps,
            threshold=0.0,
            metric="median"
        )

        assert optimal_n_pcs == 5, f"Should select 5 PCs (best R²), got {optimal_n_pcs}"

    def test_select_optimal_pcs_respects_threshold(self, device):
        """Test that threshold filters criteria voxels"""
        from fastfuncstuff.denoise.sequential import select_optimal_pcs

        n_voxels = 50

        # Create R² maps where only some voxels have high R²
        r2_maps = np.random.rand(n_voxels, 6) * 0.5
        # First 20 voxels have high R²
        r2_maps[:20, :] = 0.3 + np.random.rand(20, 6) * 0.2

        # High threshold should select fewer criteria voxels
        _, criteria_high = select_optimal_pcs(
            r2_maps=r2_maps,
            threshold=0.4,  # High threshold
            metric="median"
        )

        _, criteria_low = select_optimal_pcs(
            r2_maps=r2_maps,
            threshold=0.0,  # Low threshold
            metric="median"
        )

        # High threshold should have fewer criteria voxels
        assert criteria_high.sum() <= criteria_low.sum(), \
            "Higher threshold should select fewer criteria voxels"


# ============================================================================
# Layer 3: Large/E2E Tests - Full pipeline with ground truth
# ============================================================================

class TestDenoiseFullPipeline:
    """Test full denoising pipeline."""

    def test_denoising_improves_signal_recovery(self, device):
        """
        Adding structured noise PCs to the GLM model should improve CV R²
        compared to no PCs.  We synthesize data with a strong shared noise
        component and verify that denoising with 1 PC is better than 0 PCs.
        """
        from fastfuncstuff.denoise.sequential import cross_validate_noise_pcs, extract_noise_pcs_per_run

        torch.manual_seed(42)
        n_runs, n_tp_run, tr = 4, 80, 2.0
        n_signal_voxels, n_noise_voxels = 70, 30
        n_voxels = n_signal_voxels + n_noise_voxels
        n_tp_total = n_runs * n_tp_run
        run_starts = [i * n_tp_run for i in range(n_runs)]

        # Shared structured noise timecourse (strong, present in all voxels)
        shared_noise_tc = torch.randn(n_tp_total, device=device) * 3.0

        # Task signal: simple cosine regressor, one per run
        task_tc = torch.zeros(n_tp_total, device=device)
        for r in range(n_runs):
            t = torch.linspace(0, 2 * 3.14159, n_tp_run, device=device)
            task_tc[r * n_tp_run:(r + 1) * n_tp_run] = torch.cos(t)

        # Signal voxels: task signal + shared noise
        signal_betas = torch.randn(n_signal_voxels, 1, device=device) * 2.0
        task_signal = signal_betas @ task_tc.unsqueeze(0)  # (n_signal, n_tp)
        signal_data = task_signal + shared_noise_tc.unsqueeze(0)

        # Noise pool voxels: ONLY shared noise (no task signal, for clean PC extraction)
        noise_data = shared_noise_tc.unsqueeze(0).expand(n_noise_voxels, -1).clone()
        noise_data = noise_data + torch.randn_like(noise_data) * 0.3

        data = torch.cat([signal_data, noise_data], dim=0)

        design = task_tc.unsqueeze(1)  # (n_tp_total, 1)
        # Noise pool = last n_noise_voxels (pure structured noise)
        noise_mask = torch.zeros(n_voxels, dtype=torch.bool, device=device)
        noise_mask[n_signal_voxels:] = True

        noise_pcs_raw = extract_noise_pcs_per_run(
            data, run_starts, noise_mask, max_components=3, device=device, verbose=False)
        assert isinstance(noise_pcs_raw, list)  # return_loadings=False -> list, not tuple
        nuisance = [construct_polynomial_matrix(n_tp_run, max_degree=1, device=device)
                    for _ in range(n_runs)]

        _, r2_agg = cross_validate_noise_pcs(
            data=data,
            design_matrix=design,
            noise_pcs=noise_pcs_raw,
            run_starts=run_starts,
            tr=tr,
            max_components=3,
            nuisance=nuisance,
            device=device,
            verbose=False,
        )

        # With structured noise, using 1 PC should be better than 0 PCs
        assert r2_agg[1] > r2_agg[0], (
            f"Adding 1 noise PC should improve median R² "
            f"(0 PCs: {r2_agg[0]:.4f}, 1 PC: {r2_agg[1]:.4f})")

    def test_cv_prevents_overfitting(self, device):
        """
        cross_validate_noise_pcs R² should not monotonically increase with
        PC count (if it did, we'd always pick the maximum).  With random noise
        PCs (no structured noise), adding more PCs should not substantially
        improve cross-validated R² vs baseline.
        """
        from fastfuncstuff.denoise.sequential import cross_validate_noise_pcs, extract_noise_pcs_per_run

        torch.manual_seed(99)
        n_runs, n_tp_run, tr = 3, 60, 2.0
        n_voxels, n_noise = 60, 20
        n_tp_total = n_runs * n_tp_run
        run_starts = [i * n_tp_run for i in range(n_runs)]

        # Pure noise data — no task signal, no structured noise PCs
        data = torch.randn(n_voxels, n_tp_total, device=device)
        design = torch.randn(n_tp_total, 2, device=device)

        noise_mask = torch.zeros(n_voxels, dtype=torch.bool, device=device)
        noise_mask[:n_noise] = True
        noise_pcs_raw = extract_noise_pcs_per_run(
            data, run_starts, noise_mask, max_components=5, device=device, verbose=False)
        assert isinstance(noise_pcs_raw, list)  # return_loadings=False -> list
        nuisance = [construct_polynomial_matrix(n_tp_run, max_degree=1, device=device)
                    for _ in range(n_runs)]

        _, r2_agg = cross_validate_noise_pcs(
            data=data,
            design_matrix=design,
            noise_pcs=noise_pcs_raw,
            run_starts=run_starts,
            tr=tr,
            max_components=4,
            nuisance=nuisance,
            device=device,
            verbose=False,
        )

        # All aggregated R² values should be finite
        assert np.all(np.isfinite(r2_agg)), "Aggregated R² must be finite"

        # With pure noise, CV should not monotonically increase (otherwise we'd
        # always add more PCs).  The maximum should not be at the last PC count.
        max_idx = int(np.argmax(r2_agg))
        assert max_idx < len(r2_agg) - 1 or r2_agg[-1] - r2_agg[0] < 0.02, (
            "For pure noise data, CV R² should not substantially increase with "
            f"PC count (r2_agg={r2_agg})")
