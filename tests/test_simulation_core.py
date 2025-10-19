"""
Comprehensive tests for fMRI simulation functionality.

Tests cover:
- simulate_fmri_run with various parameters
- Multi-condition simulations
- Noise and drift
- Multi-run experiments
- Parameter validation
"""

import pytest
import torch
from fastfuncsim.simulation import simulate_fmri_run, simulate_fmri_experiment
from fastfuncsim.hrf import get_canonical_hrf
from fastfuncsim.utils import get_device


class TestSingleRunSimulation:
    """Test simulate_fmri_run function."""
    
    @pytest.fixture
    def device(self):
        return get_device()
    
    def test_basic_simulation(self, device):
        """Test basic single-run simulation."""
        n_timepoints = 100
        tr = 2.0
        matrix_size = (10, 10, 5)  # Small for speed
        
        # Simple design: one event at t=20
        onsets = torch.zeros(n_timepoints, 1, device=device)
        onsets[20, 0] = 1.0
        
        # Single beta for all voxels
        betas = [5.0]
        
        # Get canonical HRF
        hrf = get_canonical_hrf(stim_duration=0.0, tr=tr, duration=30.0, device=device)
        
        # Simulate
        data = simulate_fmri_run(
            onsets=onsets,
            betas=betas,
            hrf=hrf,
            tr=tr,
            n_timepoints=n_timepoints,
            matrix_size=matrix_size,
            noise_level=1.0,
            baseline=100.0,
            device=device
        )
        
        # Check output
        assert data.shape == (*matrix_size, n_timepoints)
        assert torch.all(torch.isfinite(data))
        
        # Mean should be around baseline
        assert abs(data.mean().item() - 100.0) < 10.0
    
    def test_multi_condition_simulation(self, device):
        """Test simulation with multiple conditions."""
        n_timepoints = 150
        tr = 1.0
        matrix_size = (8, 8, 4)
        
        # Two conditions
        onsets = torch.zeros(n_timepoints, 2, device=device)
        onsets[30, 0] = 1.0  # Cond 1 at t=30
        onsets[60, 1] = 1.0  # Cond 2 at t=60
        onsets[90, 0] = 1.0  # Cond 1 at t=90
        
        # Different betas for each condition
        betas = [3.0, 5.0]
        
        hrf = get_canonical_hrf(stim_duration=0.0, tr=tr, duration=20.0, device=device)
        
        data = simulate_fmri_run(
            onsets=onsets,
            betas=betas,
            hrf=hrf,
            tr=tr,
            n_timepoints=n_timepoints,
            matrix_size=matrix_size,
            noise_level=0.5,
            baseline=100.0,
            device=device
        )
        
        assert data.shape == (*matrix_size, n_timepoints)
        assert torch.all(torch.isfinite(data))
    
    def test_heterogeneous_voxel_betas(self, device):
        """Test simulation with different betas per voxel."""
        n_timepoints = 100
        tr = 2.0
        matrix_size = (5, 5, 2)
        n_voxels = 5 * 5 * 2
        
        onsets = torch.zeros(n_timepoints, 2, device=device)
        onsets[30, 0] = 1.0
        onsets[60, 1] = 1.0
        
        # Different betas for each voxel
        betas = torch.randn(n_voxels, 2, device=device)
        
        hrf = get_canonical_hrf(stim_duration=0.0, tr=tr, duration=30.0, device=device)
        
        data = simulate_fmri_run(
            onsets=onsets,
            betas=betas,
            hrf=hrf,
            tr=tr,
            n_timepoints=n_timepoints,
            matrix_size=matrix_size,
            noise_level=1.0,
            baseline=100.0,
            device=device
        )
        
        assert data.shape == (*matrix_size, n_timepoints)
        
        # Different voxels should have different responses
        data_flat = data.reshape(-1, n_timepoints)
        variances = data_flat.var(dim=1)
        # Not all variances should be identical
        assert variances.std() > 0.1
    
    def test_no_noise_simulation(self, device):
        """Test simulation without noise."""
        n_timepoints = 80
        tr = 1.0
        matrix_size = (4, 4, 2)
        
        onsets = torch.zeros(n_timepoints, 1, device=device)
        onsets[20, 0] = 1.0
        
        betas = [10.0]
        hrf = get_canonical_hrf(stim_duration=0.0, tr=tr, duration=20.0, device=device)
        
        data = simulate_fmri_run(
            onsets=onsets,
            betas=betas,
            hrf=hrf,
            tr=tr,
            n_timepoints=n_timepoints,
            matrix_size=matrix_size,
            noise_level=0.0,  # No noise
            baseline=100.0,
            add_scanner_drift=False,
            device=device
        )
        
        # All voxels should be identical (no noise, same beta)
        data_flat = data.reshape(-1, n_timepoints)
        for v in range(1, data_flat.shape[0]):
            assert torch.allclose(data_flat[v], data_flat[0], atol=1e-5)
    
    def test_high_noise_simulation(self, device):
        """Test simulation with high noise."""
        n_timepoints = 100
        tr = 2.0
        matrix_size = (5, 5, 3)
        
        onsets = torch.zeros(n_timepoints, 1, device=device)
        onsets[30, 0] = 1.0
        
        betas = [2.0]
        hrf = get_canonical_hrf(stim_duration=0.0, tr=tr, duration=30.0, device=device)
        
        data = simulate_fmri_run(
            onsets=onsets,
            betas=betas,
            hrf=hrf,
            tr=tr,
            n_timepoints=n_timepoints,
            matrix_size=matrix_size,
            noise_level=10.0,  # High noise
            baseline=100.0,
            device=device
        )
        
        assert torch.all(torch.isfinite(data))
        # High noise should increase variance
        assert data.std() > 5.0
    
    def test_different_hrf_shapes(self, device):
        """Test simulation with different HRF functions."""
        n_timepoints = 100
        tr = 2.0
        matrix_size = (4, 4, 2)
        
        onsets = torch.zeros(n_timepoints, 1, device=device)
        onsets[30, 0] = 1.0
        
        betas = [5.0]
        
        # Test different HRFs with different durations/parameters
        hrf_short = get_canonical_hrf(stim_duration=0.0, tr=tr, duration=20.0, device=device)
        hrf_long = get_canonical_hrf(stim_duration=2.0, tr=tr, duration=30.0, device=device)
        
        data_short = simulate_fmri_run(
            onsets=onsets, betas=betas, hrf=hrf_short,
            tr=tr, n_timepoints=n_timepoints, matrix_size=matrix_size,
            noise_level=0.0, baseline=100.0, add_scanner_drift=False, device=device
        )
        
        data_long = simulate_fmri_run(
            onsets=onsets, betas=betas, hrf=hrf_long,
            tr=tr, n_timepoints=n_timepoints, matrix_size=matrix_size,
            noise_level=0.0, baseline=100.0, add_scanner_drift=False, device=device
        )
        
        # Different HRFs should produce different signals
        assert not torch.allclose(data_short, data_long, atol=0.1)


class TestMultiRunExperiment:
    """Test simulate_fmri_experiment for multi-run simulations."""
    
    @pytest.fixture
    def device(self):
        return get_device()
    
    def test_basic_multi_run(self, device):
        """Test basic multi-run experiment."""
        n_runs = 3
        n_timepoints = 100
        tr = 2.0
        matrix_size = (5, 5, 3)
        n_conditions = 2
        
        # Create onsets for all runs
        onsets = torch.zeros(n_timepoints, n_conditions, device=device)
        onsets[20, 0] = 1.0
        onsets[50, 1] = 1.0
        onsets[80, 0] = 1.0
        
        # Betas for each condition
        betas = [3.0, 5.0]
        
        # Get HRF
        hrf = get_canonical_hrf(stim_duration=0.0, tr=tr, duration=30.0, device=device)
        
        # Run simulation
        data_list = simulate_fmri_experiment(
            n_runs=n_runs,
            onsets=onsets,
            betas=betas,
            hrf=hrf,
            tr=tr,
            n_timepoints=n_timepoints,
            matrix_size=matrix_size,
            noise_level=1.0,
            baseline=100.0,
            device=device,
            verbose=False
        )
        
        # Check outputs
        assert len(data_list) == n_runs
        
        for run_data in data_list:
            assert run_data.shape == (*matrix_size, n_timepoints)
            assert torch.all(torch.isfinite(run_data))


class TestSimulationEdgeCases:
    """Test edge cases and parameter validation."""
    
    @pytest.fixture
    def device(self):
        return get_device()
    
    def test_single_timepoint(self, device):
        """Test with minimal timepoints."""
        n_timepoints = 10
        tr = 2.0
        matrix_size = (2, 2, 1)
        
        onsets = torch.zeros(n_timepoints, 1, device=device)
        onsets[5, 0] = 1.0
        
        betas = [3.0]
        hrf = get_canonical_hrf(stim_duration=0.0, tr=tr, duration=10.0, device=device)
        
        data = simulate_fmri_run(
            onsets=onsets, betas=betas, hrf=hrf,
            tr=tr, n_timepoints=n_timepoints,
            matrix_size=matrix_size, noise_level=0.5,
            baseline=100.0, device=device
        )
        
        assert data.shape == (*matrix_size, n_timepoints)
        assert torch.all(torch.isfinite(data))
    
    def test_zero_beta(self, device):
        """Test with zero activation (no signal)."""
        n_timepoints = 80
        tr = 2.0
        matrix_size = (4, 4, 2)
        
        onsets = torch.zeros(n_timepoints, 1, device=device)
        onsets[30, 0] = 1.0
        
        betas = [0.0]  # No activation
        hrf = get_canonical_hrf(stim_duration=0.0, tr=tr, duration=20.0, device=device)
        
        data = simulate_fmri_run(
            onsets=onsets, betas=betas, hrf=hrf,
            tr=tr, n_timepoints=n_timepoints,
            matrix_size=matrix_size, noise_level=1.0,
            baseline=100.0, device=device
        )
        
        # Should be just noise + baseline
        assert abs(data.mean().item() - 100.0) < 5.0
    
    def test_negative_beta(self, device):
        """Test with negative activation (deactivation)."""
        n_timepoints = 80
        tr = 2.0
        matrix_size = (4, 4, 2)
        
        onsets = torch.zeros(n_timepoints, 1, device=device)
        onsets[30, 0] = 1.0
        
        betas = [-5.0]  # Deactivation
        hrf = get_canonical_hrf(stim_duration=0.0, tr=tr, duration=20.0, device=device)
        
        data = simulate_fmri_run(
            onsets=onsets, betas=betas, hrf=hrf,
            tr=tr, n_timepoints=n_timepoints,
            matrix_size=matrix_size, noise_level=0.0,
            baseline=100.0, add_scanner_drift=False, device=device
        )
        
        # Should have some negative deflections
        # (signal goes below baseline)
        min_val = data.min().item()
        assert min_val < 100.0
    
    def test_reproducibility(self, device):
        """Test that simulation is reproducible with same seed."""
        torch.manual_seed(42)
        
        n_timepoints = 80
        tr = 2.0
        matrix_size = (5, 5, 2)
        
        onsets = torch.zeros(n_timepoints, 1, device=device)
        onsets[30, 0] = 1.0
        
        betas = [5.0]
        hrf = get_canonical_hrf(stim_duration=0.0, tr=tr, duration=20.0, device=device)
        
        # First run
        torch.manual_seed(123)
        data1 = simulate_fmri_run(
            onsets=onsets, betas=betas, hrf=hrf,
            tr=tr, n_timepoints=n_timepoints,
            matrix_size=matrix_size, noise_level=1.0,
            baseline=100.0, device=device
        )
        
        # Second run with same seed
        torch.manual_seed(123)
        data2 = simulate_fmri_run(
            onsets=onsets, betas=betas, hrf=hrf,
            tr=tr, n_timepoints=n_timepoints,
            matrix_size=matrix_size, noise_level=1.0,
            baseline=100.0, device=device
        )
        
        # Should match exactly
        assert torch.allclose(data1, data2, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
