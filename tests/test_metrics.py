"""
Tests for design efficiency and power metrics (Liu & Frank 2004 theory)

These metrics are for DESIGN EVALUATION/SIMULATION - evaluating hypothetical
designs before running experiments. Separate from design_builder which is
for analyzing existing data.

IMPORTANT: The theoretical metrics in metrics.py expect ONSET matrices (binary),
not convolved design matrices. They build FIR matrices internally.

Realistic fMRI design parameters used:
- TR = 1-2s (typical)
- Event duration = 3s (event-related) or 15-20s (block)
- Minimum ISI = 1s after event ends
- HRF duration = ~30s (canonical HRF)

Tests validate CORRECTNESS of metric calculations, not just coverage.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncsim.hrf import get_canonical_hrf
from fastfuncsim.metrics import (
    compare_designs,
    compute_conditional_entropy,
    compute_detection_power,
    compute_efficiency_power_tradeoff,
    compute_estimation_efficiency,
    evaluate_design,
)

# =============================================================================
# REALISTIC fMRI DESIGN PARAMETERS
# =============================================================================

TR = 1.0  # Repetition time in seconds
EVENT_DURATION = 3.0  # Event-related design duration in seconds
BLOCK_DURATION = 16.0  # Block design duration in seconds
MIN_ISI = 1.0  # Minimum inter-stimulus interval after event ends
HRF_DURATION = 32.0  # HRF kernel duration in seconds
HRF_LENGTH_TRS = int(HRF_DURATION / TR)  # ~32 TRs


def create_realistic_event_onsets(
    n_timepoints: int,
    n_conditions: int,
    event_duration: float = EVENT_DURATION,
    min_isi: float = MIN_ISI,
    tr: float = TR,
    device: torch.device = torch.device("cpu"),
    seed: int | None = None,
) -> torch.Tensor:
    """
    Create realistic event-related onset matrix with proper spacing.

    Events are placed with minimum ISI constraint: next event starts at least
    min_isi seconds after the previous event ENDS (regardless of condition).

    Returns: onset matrix (n_timepoints, n_conditions), binary, in TR units.
    """
    if seed is not None:
        np.random.seed(seed)

    onsets = torch.zeros(n_timepoints, n_conditions, device=device)

    event_duration_trs = int(np.ceil(event_duration / tr))
    min_isi_trs = int(np.ceil(min_isi / tr))
    min_gap = event_duration_trs + min_isi_trs  # Total minimum spacing between ANY events

    available_time = n_timepoints - HRF_LENGTH_TRS  # Leave room for HRF at end

    # Create interleaved event sequence across all conditions
    current_time = HRF_LENGTH_TRS
    cond_idx = 0

    while current_time < available_time:
        onsets[current_time, cond_idx] = 1.0
        # Add jitter: 0-4 TRs extra
        jitter = np.random.randint(0, 5) if seed is not None else 2
        current_time += min_gap + jitter
        cond_idx = (cond_idx + 1) % n_conditions

    return onsets


def create_blocked_onsets(
    n_timepoints: int,
    n_conditions: int,
    block_duration: float = BLOCK_DURATION,
    rest_duration: float = MIN_ISI,
    tr: float = TR,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Create blocked design onset matrix.

    Blocks of one condition alternate with blocks of another, separated by rest.

    Returns: onset matrix (n_timepoints, n_conditions), binary.
    """
    onsets = torch.zeros(n_timepoints, n_conditions, device=device)

    block_trs = int(np.ceil(block_duration / tr))
    rest_trs = int(np.ceil(rest_duration / tr))
    cycle_trs = (block_trs + rest_trs) * n_conditions

    current_time = HRF_LENGTH_TRS
    cond_idx = 0

    while current_time + block_trs < n_timepoints - HRF_LENGTH_TRS:
        # Mark block onset
        onsets[current_time, cond_idx] = 1.0
        current_time += block_trs + rest_trs
        cond_idx = (cond_idx + 1) % n_conditions

    return onsets


# =============================================================================
# TESTS: ESTIMATION EFFICIENCY
# =============================================================================


class TestEstimationEfficiency:
    """
    Test estimation efficiency computation (HRF shape estimation).

    Efficiency measures how well we can estimate the HRF shape using FIR.
    Higher efficiency = lower variance in HRF estimates.
    """

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_basic_efficiency_returns_valid_structure(self, device):
        """Test that efficiency computation returns correct structure."""
        onsets = create_realistic_event_onsets(
            n_timepoints=200, n_conditions=2, device=device, seed=42
        )

        result = compute_estimation_efficiency(
            design=onsets,  # Pass onset matrix, not convolved design
            n_conditions=2,
            hrf_length=HRF_LENGTH_TRS,
            device=device,
        )

        assert "per_condition" in result
        assert "total" in result
        assert "mean" in result
        assert len(result["per_condition"]) == 2
        assert result["total"] > 0

    def test_efficiency_well_spaced_vs_overlapping_events(self, device):
        """
        Well-spaced events (non-overlapping HRF responses) should have higher
        efficiency PER EVENT than densely overlapping events.

        When events overlap, the FIR design matrix becomes rank-deficient,
        reducing our ability to estimate each time point of the HRF.
        """
        # Well-spaced: events spaced > HRF_LENGTH apart (no overlap)
        onsets_spaced = torch.zeros(300, 1, device=device)
        for t in range(50, 280, 50):  # 50 TRs apart > 32 TR HRF
            onsets_spaced[t, 0] = 1.0
        n_events_spaced = 5

        # Densely packed: events very close (massive overlap)
        onsets_dense = torch.zeros(300, 1, device=device)
        for t in range(50, 280, 5):  # 5 TRs apart (massive overlap)
            onsets_dense[t, 0] = 1.0
        n_events_dense = 46

        result_spaced = compute_estimation_efficiency(
            onsets_spaced, 1, HRF_LENGTH_TRS, device=device
        )
        result_dense = compute_estimation_efficiency(onsets_dense, 1, HRF_LENGTH_TRS, device=device)

        # Efficiency PER EVENT should be higher for well-spaced events
        # (total efficiency might be higher for dense due to more events,
        # but per-event tells the real story)
        eff_per_event_spaced = result_spaced["total"] / n_events_spaced
        eff_per_event_dense = result_dense["total"] / n_events_dense

        assert eff_per_event_spaced > eff_per_event_dense, (
            f"Well-spaced events should have higher efficiency per event: "
            f"{eff_per_event_spaced:.6f} vs {eff_per_event_dense:.6f}"
        )

    def test_efficiency_empty_design(self, device):
        """Efficiency should be near zero with no events."""
        onsets_empty = torch.zeros(100, 2, device=device)

        result = compute_estimation_efficiency(
            design=onsets_empty, n_conditions=2, hrf_length=HRF_LENGTH_TRS, device=device
        )

        # Due to regularization, may not be exactly zero
        assert result["total"] < 1e-4

    def test_efficiency_scales_with_events(self, device):
        """
        More events (with good spacing) should increase total efficiency.
        We use events spaced > HRF_LENGTH apart to avoid overlap.
        """
        # Few events, well-spaced (> 32 TR HRF)
        onsets_few = torch.zeros(200, 1, device=device)
        for t in [40, 90, 140]:  # 50 TRs apart
            onsets_few[t, 0] = 1.0

        # More events, same good spacing
        onsets_more = torch.zeros(300, 1, device=device)
        for t in [40, 90, 140, 190, 240]:  # 50 TRs apart, more events
            onsets_more[t, 0] = 1.0

        result_few = compute_estimation_efficiency(onsets_few, 1, HRF_LENGTH_TRS, device=device)
        result_more = compute_estimation_efficiency(onsets_more, 1, HRF_LENGTH_TRS, device=device)

        # With 5 events vs 3, all well-spaced, efficiency should be higher
        assert result_more["total"] > result_few["total"], (
            f"More well-spaced events should increase efficiency: {result_more['total']:.6f} vs {result_few['total']:.6f}"
        )


# =============================================================================
# TESTS: DETECTION POWER
# =============================================================================


class TestDetectionPower:
    """
    Test detection power computation (activation amplitude detection).

    Power measures how well we can detect the presence of activation.
    Higher power = lower variance in amplitude estimates.
    """

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    @pytest.fixture
    def canonical_hrf(self, device):
        return get_canonical_hrf(stim_duration=0.0, tr=TR, duration=HRF_DURATION, device=device)

    def test_basic_power_returns_valid_structure(self, canonical_hrf, device):
        """Test that power computation returns correct structure."""
        onsets = create_realistic_event_onsets(
            n_timepoints=200, n_conditions=2, device=device, seed=42
        )

        result = compute_detection_power(
            design=onsets,  # Pass onset matrix
            hrf_assumed=canonical_hrf,
            n_conditions=2,
            device=device,
        )

        assert "per_condition" in result
        assert "total" in result
        assert "mean" in result
        assert "snr" in result
        assert len(result["per_condition"]) == 2

    def test_more_events_higher_power(self, canonical_hrf, device):
        """More events should give higher detection power."""
        # Few events
        onsets_few = torch.zeros(300, 1, device=device)
        for t in [50, 150, 250]:
            onsets_few[t, 0] = 1.0

        # More events (well-spaced)
        onsets_many = torch.zeros(300, 1, device=device)
        for t in range(50, 280, 40):
            onsets_many[t, 0] = 1.0

        result_few = compute_detection_power(onsets_few, canonical_hrf, 1, device=device)
        result_many = compute_detection_power(onsets_many, canonical_hrf, 1, device=device)

        assert result_many["total"] > result_few["total"], (
            f"More events should give higher power: {result_many['total']:.4f} vs {result_few['total']:.4f}"
        )

    def test_power_empty_design(self, canonical_hrf, device):
        """Power should be near zero with no events."""
        onsets_empty = torch.zeros(100, 2, device=device)

        result = compute_detection_power(onsets_empty, canonical_hrf, 2, device=device)
        assert result["total"] < 1e-4

    def test_snr_scales_with_effect_size(self, canonical_hrf, device):
        """SNR should scale linearly with effect size."""
        onsets = create_realistic_event_onsets(
            n_timepoints=200, n_conditions=1, device=device, seed=42
        )

        result_small = compute_detection_power(
            onsets, canonical_hrf, 1, effect_size=0.5, noise_std=1.0, device=device
        )
        result_large = compute_detection_power(
            onsets, canonical_hrf, 1, effect_size=2.0, noise_std=1.0, device=device
        )

        # SNR scales linearly: 2.0/0.5 = 4x
        snr_ratio = result_large["mean_snr"] / (result_small["mean_snr"] + 1e-10)
        expected_ratio = 2.0 / 0.5

        assert abs(snr_ratio - expected_ratio) < 0.2, (
            f"SNR should scale with effect size: ratio={snr_ratio:.2f}, expected={expected_ratio}"
        )

    def test_snr_decreases_with_noise(self, canonical_hrf, device):
        """SNR should decrease with higher noise."""
        onsets = create_realistic_event_onsets(
            n_timepoints=200, n_conditions=1, device=device, seed=42
        )

        result_low = compute_detection_power(
            onsets, canonical_hrf, 1, effect_size=1.0, noise_std=0.5, device=device
        )
        result_high = compute_detection_power(
            onsets, canonical_hrf, 1, effect_size=1.0, noise_std=2.0, device=device
        )

        assert result_low["mean_snr"] > result_high["mean_snr"], (
            "Lower noise should give higher SNR"
        )


# =============================================================================
# TESTS: CONDITIONAL ENTROPY
# =============================================================================


class TestConditionalEntropy:
    """Test conditional entropy computation (design randomness)."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_fixed_isi_zero_entropy(self, device):
        """Fixed ISI should have zero entropy."""
        onsets = torch.zeros(200, 1, device=device)
        for t in range(40, 180, 20):  # Exactly every 20 TRs
            onsets[t, 0] = 1.0

        result = compute_conditional_entropy(onsets, 1, tr=TR, device=device)
        assert result["total"] < 0.1

    def test_variable_isi_higher_entropy(self, device):
        """Variable ISI should have higher entropy than fixed ISI."""
        np.random.seed(42)

        # Fixed ISI
        onsets_fixed = torch.zeros(200, 1, device=device)
        for t in range(40, 180, 20):
            onsets_fixed[t, 0] = 1.0

        # Variable ISI
        onsets_variable = torch.zeros(200, 1, device=device)
        current_t = 40
        while current_t < 180:
            onsets_variable[current_t, 0] = 1.0
            current_t += np.random.randint(15, 30)

        result_fixed = compute_conditional_entropy(onsets_fixed, 1, tr=TR, device=device)
        result_variable = compute_conditional_entropy(onsets_variable, 1, tr=TR, device=device)

        assert result_variable["total"] > result_fixed["total"]

    def test_empty_design_zero_entropy(self, device):
        """Empty design should have zero entropy."""
        onsets = torch.zeros(100, 2, device=device)
        result = compute_conditional_entropy(onsets, 2, device=device)
        assert result["total"] == 0.0

    def test_single_event_zero_entropy(self, device):
        """Single event should have zero entropy (no ISI to measure)."""
        onsets = torch.zeros(100, 1, device=device)
        onsets[50, 0] = 1.0
        result = compute_conditional_entropy(onsets, 1, device=device)
        assert result["total"] == 0.0


# =============================================================================
# TESTS: TRADE-OFF CURVE
# =============================================================================


class TestEfficiencyPowerTradeoff:
    """Test efficiency-power trade-off curve."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_inverse_relationship(self, device):
        """Efficiency and power should trade off inversely."""
        result = compute_efficiency_power_tradeoff(hrf_length=20, device=device)

        correlation = np.corrcoef(result["efficiency"], result["power"])[0, 1]
        assert correlation < 0, "Efficiency and power should be negatively correlated"

    def test_efficiency_increases_with_alpha(self, device):
        """Efficiency should generally increase with α."""
        result = compute_efficiency_power_tradeoff(hrf_length=20, device=device)
        assert result["efficiency"][-1] > result["efficiency"][0]

    def test_power_decreases_with_alpha(self, device):
        """Power should generally decrease with α."""
        result = compute_efficiency_power_tradeoff(hrf_length=20, device=device)
        assert result["power"][0] > result["power"][-1]


# =============================================================================
# TESTS: COMPLETE EVALUATION
# =============================================================================


class TestEvaluateDesign:
    """Test complete design evaluation."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    @pytest.fixture
    def hrf(self, device):
        return get_canonical_hrf(stim_duration=0.0, tr=TR, duration=HRF_DURATION, device=device)

    def test_returns_all_metrics(self, hrf, device):
        """Complete evaluation should return all metric types."""
        onsets = create_realistic_event_onsets(200, 2, device=device, seed=42)

        result = evaluate_design(onsets, hrf, 2, TR, device=device)

        assert "efficiency" in result
        assert "power" in result
        assert "entropy" in result
        assert "summary" in result

    def test_summary_has_key_metrics(self, hrf, device):
        """Summary should contain key metrics."""
        onsets = create_realistic_event_onsets(200, 2, device=device, seed=42)
        result = evaluate_design(onsets, hrf, 2, TR, device=device)

        assert "efficiency_mean" in result["summary"]
        assert "power_mean" in result["summary"]
        assert "entropy_total" in result["summary"]


# =============================================================================
# TESTS: DESIGN COMPARISON
# =============================================================================


class TestCompareDesigns:
    """Test design comparison."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    @pytest.fixture
    def hrf(self, device):
        return get_canonical_hrf(stim_duration=0.0, tr=TR, duration=HRF_DURATION, device=device)

    def test_comparison_structure(self, hrf, device):
        """Comparison should return results for each design."""
        designs = {
            "design_a": create_realistic_event_onsets(200, 2, device=device, seed=42),
            "design_b": create_realistic_event_onsets(200, 2, device=device, seed=123),
        }

        result = compare_designs(designs, hrf, 2, TR, device=device)

        assert "design_a" in result
        assert "design_b" in result
        assert "summary_table" in result


# =============================================================================
# TESTS: NUMERICAL STABILITY
# =============================================================================


class TestNumericalStability:
    """Test numerical stability."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    @pytest.fixture
    def hrf(self, device):
        return get_canonical_hrf(stim_duration=0.0, tr=TR, duration=HRF_DURATION, device=device)

    def test_many_conditions(self, hrf, device):
        """Should handle many conditions (e.g., 8 stimulus types)."""
        n_conditions = 8
        onsets = torch.zeros(400, n_conditions, device=device)

        for cond in range(n_conditions):
            for t in range(40 + cond * 5, 380, 40):
                onsets[t, cond] = 1.0

        result = evaluate_design(onsets, hrf, n_conditions, TR, device=device)

        assert len(result["efficiency"]["per_condition"]) == n_conditions
        assert result["summary"]["power_mean"] > 0

    def test_long_design(self, hrf, device):
        """Should handle long scans (e.g., 20 minutes)."""
        onsets = create_realistic_event_onsets(1200, 2, device=device, seed=42)
        result = evaluate_design(onsets, hrf, 2, TR, device=device)

        assert result["efficiency"]["total"] > 0


# =============================================================================
# TESTS: THEORETICAL VALIDATION
# =============================================================================


class TestTheoreticalValidation:
    """Tests validating theoretical relationships from Liu & Frank (2004)."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    @pytest.fixture
    def hrf(self, device):
        return get_canonical_hrf(stim_duration=0.0, tr=TR, duration=HRF_DURATION, device=device)

    def test_longer_scan_more_events_higher_power(self, hrf, device):
        """
        Longer scan with more well-spaced events should have higher power.
        Events must be spaced > HRF_LENGTH apart to avoid overlap in FIR matrix.
        """
        # Short scan: 3 events, spaced > 32 TR apart
        onsets_short = torch.zeros(150, 1, device=device)
        for t in [40, 80, 120]:  # 40 TR spacing
            onsets_short[t, 0] = 1.0

        # Long scan: 6 events, same spacing
        onsets_long = torch.zeros(300, 1, device=device)
        for t in [40, 80, 120, 160, 200, 240]:  # 40 TR spacing
            onsets_long[t, 0] = 1.0

        result_short = compute_detection_power(onsets_short, hrf, 1, device=device)
        result_long = compute_detection_power(onsets_long, hrf, 1, device=device)

        # With non-overlapping events, 6 events vs 3 should give higher power
        assert result_long["total"] > result_short["total"], (
            f"Longer scan with more events should have higher power: "
            f"{result_long['total']:.4f} vs {result_short['total']:.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
