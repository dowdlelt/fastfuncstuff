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

from fastfuncstuff.design.hrf import get_canonical_hrf
from fastfuncstuff.simulation.core import simulate_fmri_experiment, simulate_fmri_run
from fastfuncstuff.utils import get_device


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


class TestCreateParametricVoxels:
    """Test create_parametric_voxels for structured voxel generation."""
    
    @pytest.fixture
    def device(self):
        return get_device()
    
    def test_basic_parametric_voxels(self, device):
        """Test basic parametric voxel creation."""
        from fastfuncstuff.simulation.core import create_parametric_voxels
        
        matrix_size = (10, 20, 5)
        n_conditions = 3
        
        betas, hrf_indices, noise_levels = create_parametric_voxels(
            matrix_size=matrix_size,
            n_conditions=n_conditions,
            device=device
        )
        
        n_voxels = 10 * 20 * 5
        
        assert betas.shape == (n_voxels, n_conditions)
        assert hrf_indices.shape == (n_voxels,)
        assert noise_levels.shape == (n_voxels,)
        assert torch.all(torch.isfinite(betas))
    
    def test_parametric_voxels_with_hrf_library(self, device):
        """Test parametric voxels with HRF library."""
        from fastfuncstuff.simulation.core import create_parametric_voxels
        
        matrix_size = (20, 20, 5)
        n_conditions = 2
        n_hrfs = 5
        hrf_length = 15
        
        hrf_library = torch.randn(n_hrfs, hrf_length, device=device)
        
        betas, hrf_indices, noise_levels = create_parametric_voxels(
            matrix_size=matrix_size,
            n_conditions=n_conditions,
            hrf_library=hrf_library,
            device=device
        )
        
        n_voxels = 20 * 20 * 5
        
        assert betas.shape == (n_voxels, n_conditions)
        assert hrf_indices.max() < n_hrfs
        assert hrf_indices.min() >= 0
    
    def test_parametric_voxels_with_custom_beta_ranges(self, device):
        """Test parametric voxels with custom beta ranges."""
        from fastfuncstuff.simulation.core import create_parametric_voxels
        
        matrix_size = (10, 20, 3)
        n_conditions = 2
        beta_ranges = [(0, 10), (-5, 5)]
        
        betas, hrf_indices, noise_levels = create_parametric_voxels(
            matrix_size=matrix_size,
            n_conditions=n_conditions,
            beta_ranges=beta_ranges,
            device=device
        )
        
        # Check betas are within expected ranges
        assert betas[:, 0].min() >= 0
        assert betas[:, 0].max() <= 10
        assert betas[:, 1].min() >= -5
        assert betas[:, 1].max() <= 5
    
    def test_noise_levels_vary_by_z(self, device):
        """Test that noise levels vary across z slices."""
        from fastfuncstuff.simulation.core import create_parametric_voxels
        
        matrix_size = (5, 20, 10)  # 10 slices
        n_conditions = 1
        
        betas, hrf_indices, noise_levels = create_parametric_voxels(
            matrix_size=matrix_size,
            n_conditions=n_conditions,
            device=device
        )
        
        # Noise levels should range from 0.5 to 2.0 across z
        assert noise_levels.min() >= 0.5 - 0.01
        assert noise_levels.max() <= 2.0 + 0.01
        # Should have variation
        assert noise_levels.std() > 0.1


class TestSimulateBatchExperiments:
    """Test simulate_batch_experiments for batch simulation."""
    
    @pytest.fixture
    def device(self):
        return get_device()
    
    def test_basic_batch_experiments(self, device):
        """Test basic batch experiment simulation."""
        from fastfuncstuff.simulation.core import simulate_batch_experiments
        
        n_experiments = 5
        sim_config = {
            'n_runs': 2,
            'tr': 2.0,
            'n_timepoints': 100,
        }
        
        experiments = simulate_batch_experiments(
            n_experiments=n_experiments,
            sim_config=sim_config,
            device=device,
            verbose=False
        )
        
        assert len(experiments) == n_experiments
        for exp in experiments:
            assert 'id' in exp
    
    def test_batch_experiments_verbose(self, device, capsys):
        """Test batch experiments with verbose output."""
        from fastfuncstuff.simulation.core import simulate_batch_experiments
        
        n_experiments = 10
        sim_config = {'n_runs': 1}
        
        _experiments = simulate_batch_experiments(
            n_experiments=n_experiments,
            sim_config=sim_config,
            device=device,
            verbose=True
        )
        
        captured = capsys.readouterr()
        assert "Simulating" in captured.out
        assert "Batch simulation complete" in captured.out


class TestSimulationFileWriting:
    """Test file writing functions in simulation module."""
    
    @pytest.fixture
    def device(self):
        return get_device()
    
    def test_write_afni_onset_files(self, device, tmp_path):
        """Test AFNI onset file writing."""
        from fastfuncstuff.simulation.core import write_afni_onset_files
        
        # Create onset matrices for 2 runs, 2 conditions
        n_timepoints = 100
        onsets1 = torch.zeros(n_timepoints, 2)
        onsets1[10, 0] = 1.0
        onsets1[30, 1] = 1.0
        
        onsets2 = torch.zeros(n_timepoints, 2)
        onsets2[20, 0] = 1.0
        onsets2[50, 1] = 1.0
        
        onset_files = write_afni_onset_files(
            onsets_list=[onsets1, onsets2],
            tr=2.0,
            output_dir=tmp_path,
            prefix="test_onsets"
        )
        
        assert len(onset_files) == 2
        for f in onset_files:
            assert f.exists()
            content = f.read_text()
            assert len(content) > 0
    
    def test_write_afni_onset_single_tensor(self, device, tmp_path):
        """Test AFNI onset file writing with single tensor."""
        from fastfuncstuff.simulation.core import write_afni_onset_files
        
        onsets = torch.zeros(100, 3)
        onsets[10, 0] = 1.0
        onsets[20, 1] = 1.0
        
        onset_files = write_afni_onset_files(
            onsets_list=onsets,
            tr=1.0,
            output_dir=tmp_path,
            prefix="single_onset"
        )
        
        assert len(onset_files) == 3
    
    def test_write_afni_onset_empty_run(self, device, tmp_path):
        """Test AFNI onset file with empty run (no events)."""
        from fastfuncstuff.simulation.core import write_afni_onset_files
        
        # All zeros - no events
        onsets = torch.zeros(50, 1)
        
        onset_files = write_afni_onset_files(
            onsets_list=[onsets],
            tr=2.0,
            output_dir=tmp_path,
            prefix="empty"
        )
        
        assert len(onset_files) == 1
        content = onset_files[0].read_text()
        assert '*' in content  # AFNI convention for no events
    
    def test_write_nifti_files(self, device, tmp_path):
        """Test NIfTI file writing."""
        import nibabel as nib

        from fastfuncstuff.simulation.core import write_nifti_files
        
        # Create mock data
        data1 = torch.randn(10, 10, 5, 50)
        data2 = torch.randn(10, 10, 5, 50)
        
        nifti_files = write_nifti_files(
            data_list=[data1, data2],
            tr=2.0,
            output_dir=tmp_path,
            prefix="run",
            voxel_size=(2.0, 2.0, 2.0)
        )
        
        assert len(nifti_files) == 2
        for f in nifti_files:
            assert f.exists()
            img = nib.load(f)
            assert img.shape == (10, 10, 5, 50)
    
    def test_write_nifti_custom_affine(self, device, tmp_path):
        """Test NIfTI file writing with custom affine."""
        import nibabel as nib
        import numpy as np

        from fastfuncstuff.simulation.core import write_nifti_files
        
        data = torch.randn(5, 5, 3, 20)
        
        custom_affine = np.eye(4)
        custom_affine[0, 0] = 3.0
        custom_affine[1, 1] = 3.0
        custom_affine[2, 2] = 4.0
        
        nifti_files = write_nifti_files(
            data_list=[data],
            tr=1.5,
            output_dir=tmp_path,
            affine=custom_affine
        )
        
        img = nib.load(nifti_files[0])
        np.testing.assert_array_equal(img.affine, custom_affine)


class TestSaveSimulationOutputs:
    """Test save_simulation_outputs for complete output saving."""
    
    @pytest.fixture
    def device(self):
        return get_device()
    
    def test_save_simulation_outputs_basic(self, device, tmp_path):
        """Test basic simulation output saving."""
        from fastfuncstuff.simulation.core import save_simulation_outputs
        
        data1 = torch.randn(5, 5, 3, 30)
        data2 = torch.randn(5, 5, 3, 30)
        
        onsets = torch.zeros(30, 2)
        onsets[5, 0] = 1.0
        onsets[15, 1] = 1.0
        
        result = save_simulation_outputs(
            data_list=[data1, data2],
            onsets_list=[onsets, onsets],
            tr=2.0,
            output_dir=tmp_path,
            label="test_sim",
            verbose=False
        )
        
        assert result['output_dir'].exists()
        assert len(result['onset_files']) == 2
        assert len(result['nifti_files']) == 2
        assert result['metadata_file'].exists()
    
    def test_save_simulation_outputs_with_metadata(self, device, tmp_path):
        """Test simulation output saving with custom metadata."""
        from fastfuncstuff.simulation.core import save_simulation_outputs
        
        data = torch.randn(4, 4, 2, 20)
        onsets = torch.zeros(20, 1)
        onsets[5, 0] = 1.0
        
        metadata = {
            'betas': [3.0, 5.0],
            'noise_level': 1.5,
            'hrf_type': 'canonical'
        }
        
        result = save_simulation_outputs(
            data_list=[data],
            onsets_list=onsets,
            tr=1.0,
            output_dir=tmp_path,
            label="meta_test",
            metadata=metadata,
            verbose=False
        )
        
        # Check metadata was written
        content = result['metadata_file'].read_text()
        assert 'betas' in content
        assert 'noise_level' in content
    
    def test_save_simulation_outputs_with_tensor_metadata(self, device, tmp_path):
        """Test simulation output saving with tensor in metadata."""
        from fastfuncstuff.simulation.core import save_simulation_outputs
        
        data = torch.randn(4, 4, 2, 20)
        onsets = torch.zeros(20, 1)
        
        # Small tensor should be serialized as list
        small_tensor = torch.tensor([1.0, 2.0, 3.0])
        # Large tensor should show shape only
        large_tensor = torch.randn(100, 100)
        
        metadata = {
            'small': small_tensor,
            'large': large_tensor
        }
        
        result = save_simulation_outputs(
            data_list=[data],
            onsets_list=onsets,
            tr=1.0,
            output_dir=tmp_path,
            label="tensor_meta",
            metadata=metadata,
            verbose=False
        )
        
        content = result['metadata_file'].read_text()
        assert 'small' in content
        assert 'Tensor' in content  # Large tensor shows Tensor(shape)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

