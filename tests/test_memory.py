"""Tests for the memory module."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import torch

from fastfuncsim.memory import (
    MemoryConfig,
    bytes_per_voxel_arma,
    bytes_per_voxel_denoise,
    bytes_per_voxel_glm,
    bytes_per_voxel_ridge,
    bytes_per_voxel_xval,
    dyn_chunk_estimator,
    estimate_chunk_size,
    estimate_keep_on_cpu,
    get_available_memory,
    get_memory_config,
    reset_memory_config,
    set_memory_config,
)


class TestMemoryConfig:
    """Tests for MemoryConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MemoryConfig()
        assert config.gpu_safety_factor == 0.5
        assert config.cpu_safety_factor == 0.75
        assert config.min_chunk_size == 1000
        assert config.max_chunk_size_gpu == 90000
        assert config.max_chunk_size_cpu == 1000000
        assert config.data_threshold_gb == 4.0
        assert config.arma_safety_factor == 0.6
        assert config.double_precision_multiplier == 2.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = MemoryConfig(
            gpu_safety_factor=0.7,
            cpu_safety_factor=0.5,
            min_chunk_size=500,
            max_chunk_size_gpu=150000,
            max_chunk_size_cpu=2000000,
        )
        assert config.gpu_safety_factor == 0.7
        assert config.cpu_safety_factor == 0.5
        assert config.min_chunk_size == 500
        assert config.max_chunk_size_gpu == 150000
        assert config.max_chunk_size_cpu == 2000000

    def test_invalid_gpu_safety_factor_high(self):
        """Test that gpu_safety_factor > 1 raises ValueError."""
        with pytest.raises(ValueError, match="gpu_safety_factor"):
            MemoryConfig(gpu_safety_factor=1.5)

    def test_invalid_gpu_safety_factor_zero(self):
        """Test that gpu_safety_factor <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="gpu_safety_factor"):
            MemoryConfig(gpu_safety_factor=0.0)

    def test_invalid_gpu_safety_factor_negative(self):
        """Test that negative gpu_safety_factor raises ValueError."""
        with pytest.raises(ValueError, match="gpu_safety_factor"):
            MemoryConfig(gpu_safety_factor=-0.5)

    def test_invalid_cpu_safety_factor(self):
        """Test that invalid cpu_safety_factor raises ValueError."""
        with pytest.raises(ValueError, match="cpu_safety_factor"):
            MemoryConfig(cpu_safety_factor=0.0)

    def test_invalid_min_chunk_size(self):
        """Test that min_chunk_size < 100 raises ValueError."""
        with pytest.raises(ValueError, match="min_chunk_size"):
            MemoryConfig(min_chunk_size=50)


class TestGlobalConfig:
    """Tests for global configuration management."""

    def teardown_method(self):
        """Reset global config after each test."""
        reset_memory_config()

    def test_get_memory_config_default(self):
        """Test getting default config."""
        reset_memory_config()
        config = get_memory_config()
        assert isinstance(config, MemoryConfig)
        assert config.gpu_safety_factor == 0.5

    def test_set_memory_config(self):
        """Test setting custom config."""
        custom = MemoryConfig(gpu_safety_factor=0.8)
        set_memory_config(custom)
        config = get_memory_config()
        assert config.gpu_safety_factor == 0.8

    def test_reset_memory_config(self):
        """Test resetting config to defaults."""
        custom = MemoryConfig(gpu_safety_factor=0.8)
        set_memory_config(custom)
        reset_memory_config()
        config = get_memory_config()
        assert config.gpu_safety_factor == 0.5

    def test_env_override_gpu_safety_factor(self):
        """Test environment variable override for gpu_safety_factor."""
        reset_memory_config()
        with patch.dict(os.environ, {"FFS_GPU_SAFETY_FACTOR": "0.7"}):
            config = get_memory_config()
            assert config.gpu_safety_factor == 0.7

    def test_env_override_min_chunk_size(self):
        """Test environment variable override for min_chunk_size."""
        reset_memory_config()
        with patch.dict(os.environ, {"FFS_MIN_CHUNK_SIZE": "2000"}):
            config = get_memory_config()
            assert config.min_chunk_size == 2000

    def test_env_override_max_chunk_size_gpu(self):
        """Test environment variable override for max_chunk_size_gpu."""
        reset_memory_config()
        with patch.dict(os.environ, {"FFS_MAX_CHUNK_SIZE_GPU": "100000"}):
            config = get_memory_config()
            assert config.max_chunk_size_gpu == 100000

    def test_env_override_cpu_safety_factor(self):
        """Test environment variable override for cpu_safety_factor."""
        reset_memory_config()
        with patch.dict(os.environ, {"FFS_CPU_SAFETY_FACTOR": "0.6"}):
            config = get_memory_config()
            assert config.cpu_safety_factor == 0.6


class TestBytesPerVoxel:
    """Tests for per-voxel memory estimation functions."""

    def test_bytes_per_voxel_glm_basic(self):
        """Test GLM memory estimation."""
        result = bytes_per_voxel_glm(n_timepoints=100, n_regressors=10)
        expected = (5 * 100 + 10) * 4
        assert result == expected

    def test_bytes_per_voxel_glm_large(self):
        """Test GLM memory estimation with large values."""
        result = bytes_per_voxel_glm(n_timepoints=1000, n_regressors=100)
        expected = (5 * 1000 + 100) * 4
        assert result == expected

    def test_bytes_per_voxel_xval_basic(self):
        """Test cross-validation memory estimation."""
        result = bytes_per_voxel_xval(n_timepoints=100, n_regressors=10)
        expected = (8 * 100 + max(10, 10)) * 4
        assert result == expected

    def test_bytes_per_voxel_xval_small_regressors(self):
        """Test xval with fewer than 10 regressors uses min of 10."""
        result = bytes_per_voxel_xval(n_timepoints=100, n_regressors=5)
        expected = (8 * 100 + 10) * 4
        assert result == expected

    def test_bytes_per_voxel_ridge_basic(self):
        """Test ridge regression memory estimation."""
        result = bytes_per_voxel_ridge(n_timepoints=100, n_regressors=10, n_fractions=50)
        base = (5 * 100 + 10) * 4
        expected = base + (50 * 10 * 4)
        assert result == expected

    def test_bytes_per_voxel_denoise_basic(self):
        """Test denoising memory estimation."""
        result = bytes_per_voxel_denoise(n_timepoints=100, n_noise_pcs=20)
        expected = (6 * 100 + 20) * 4
        assert result == expected

    def test_bytes_per_voxel_arma_basic(self):
        """Test ARMA memory estimation."""
        result = bytes_per_voxel_arma(n_timepoints=100, n_regressors=10, max_lag=10)
        expected = (8 * 100 + 2 * 10 + 10) * 4
        assert result == expected


class TestGetAvailableMemory:
    """Tests for get_available_memory function."""

    def teardown_method(self):
        """Reset global config after each test."""
        reset_memory_config()

    def test_cpu_memory(self):
        """Test CPU memory detection."""
        device = torch.device("cpu")
        result = get_available_memory(device)
        assert result > 0
        assert isinstance(result, int)

    def test_cpu_memory_with_safety_factor(self):
        """Test CPU memory with custom safety factor."""
        device = torch.device("cpu")
        result_full = get_available_memory(device, safety_factor=1.0)
        result_half = get_available_memory(device, safety_factor=0.5)
        assert result_half < result_full

    def test_cpu_memory_uses_config(self):
        """Test that CPU memory uses config default."""
        reset_memory_config()
        config = MemoryConfig(cpu_safety_factor=0.5)
        set_memory_config(config)
        device = torch.device("cpu")
        result = get_available_memory(device)
        assert result > 0


class TestEstimateChunkSize:
    """Tests for estimate_chunk_size function."""

    def teardown_method(self):
        """Reset global config after each test."""
        reset_memory_config()

    def test_basic_estimation(self):
        """Test basic chunk size estimation."""
        result = estimate_chunk_size(
            n_voxels=10000,
            n_timepoints=100,
            n_regressors=10,
            device=torch.device("cpu"),
            operation="glm",
        )
        assert result > 0
        assert result <= 10000

    def test_chunk_size_respects_min(self):
        """Test that chunk size respects minimum."""
        config = MemoryConfig(min_chunk_size=5000)
        set_memory_config(config)
        result = estimate_chunk_size(
            n_voxels=100000,
            n_timepoints=100,
            n_regressors=10,
            device=torch.device("cpu"),
            operation="glm",
        )
        assert result >= 5000

    def test_chunk_size_respects_max_cpu(self):
        """Test that chunk size respects CPU maximum."""
        config = MemoryConfig(max_chunk_size_cpu=50000)
        set_memory_config(config)
        result = estimate_chunk_size(
            n_voxels=1000000,
            n_timepoints=100,
            n_regressors=10,
            device=torch.device("cpu"),
            operation="glm",
        )
        assert result <= 50000

    def test_chunk_size_different_operations(self):
        """Test chunk size for different operations."""
        base_params = {
            "n_voxels": 10000,
            "n_timepoints": 100,
            "n_regressors": 10,
            "device": torch.device("cpu"),
        }

        for op in ["glm", "xval", "ridge", "denoise", "arma"]:
            result = estimate_chunk_size(**base_params, operation=op)
            assert result > 0, f"Failed for operation: {op}"

    def test_chunk_size_double_precision(self):
        """Test that double precision doubles memory estimate."""
        result_single = estimate_chunk_size(
            n_voxels=100000,
            n_timepoints=100,
            n_regressors=10,
            device=torch.device("cpu"),
            operation="glm",
            use_double=False,
        )
        result_double = estimate_chunk_size(
            n_voxels=100000,
            n_timepoints=100,
            n_regressors=10,
            device=torch.device("cpu"),
            operation="glm",
            use_double=True,
        )
        assert result_double <= result_single

    def test_chunk_size_verbose(self, capsys):
        """Test verbose output."""
        estimate_chunk_size(
            n_voxels=10000,
            n_timepoints=100,
            n_regressors=10,
            device=torch.device("cpu"),
            operation="glm",
            verbose=True,
        )
        captured = capsys.readouterr()
        assert "Chunk size estimation" in captured.out

    def test_chunk_size_does_not_exceed_n_voxels(self):
        """Test that chunk size doesn't exceed total voxels."""
        result = estimate_chunk_size(
            n_voxels=1000,
            n_timepoints=100,
            n_regressors=10,
            device=torch.device("cpu"),
            operation="glm",
        )
        assert result <= 1000


class TestDynChunkEstimator:
    """Tests for dyn_chunk_estimator function."""

    def teardown_method(self):
        """Reset global config after each test."""
        reset_memory_config()

    def test_basic_estimation(self):
        """Test basic dynamic chunk estimation."""
        result = dyn_chunk_estimator(
            n_voxels=10000,
            n_timepoints=100,
            n_task_regressors=10,
            n_nuisance_regressors=5,
            device=torch.device("cpu"),
            operation="glm",
        )
        assert result > 0
        assert result <= 10000

    def test_loro_cv_strategy(self):
        """Test LORO CV strategy."""
        result = dyn_chunk_estimator(
            n_voxels=10000,
            n_timepoints=100,
            n_task_regressors=10,
            device=torch.device("cpu"),
            operation="denoise",
            cv_strategy=1,
            n_runs=5,
            streaming_stats=True,
        )
        assert result > 0

    def test_split_half_cv_strategy(self):
        """Test split-half CV strategy."""
        result = dyn_chunk_estimator(
            n_voxels=10000,
            n_timepoints=100,
            n_task_regressors=10,
            device=torch.device("cpu"),
            operation="xval",
            cv_strategy=0.5,
        )
        assert result > 0

    def test_denoise_streaming_vs_full(self):
        """Test denoise with streaming vs full accumulators."""
        base_params = {
            "n_voxels": 10000,
            "n_timepoints": 100,
            "n_task_regressors": 10,
            "n_nuisance_regressors": 20,
            "device": torch.device("cpu"),
            "operation": "denoise",
            "n_runs": 5,
            "max_components": 10,
        }

        result_streaming = dyn_chunk_estimator(**base_params, cv_strategy=1, streaming_stats=True)
        result_full = dyn_chunk_estimator(**base_params, cv_strategy=0.5, streaming_stats=False)

        assert result_streaming > 0
        assert result_full > 0

    def test_verbose_output(self, capsys):
        """Test verbose output."""
        dyn_chunk_estimator(
            n_voxels=10000,
            n_timepoints=100,
            n_task_regressors=10,
            device=torch.device("cpu"),
            operation="glm",
            verbose=True,
        )
        captured = capsys.readouterr()
        assert "Dynamic Chunk Size Estimation" in captured.out


class TestEstimateKeepOnCpu:
    """Tests for estimate_keep_on_cpu function."""

    def teardown_method(self):
        """Reset global config after each test."""
        reset_memory_config()

    def test_force_cpu_true(self):
        """Test that force_cpu=True returns True."""
        result = estimate_keep_on_cpu(
            n_voxels=10000,
            n_timepoints_total=100,
            device=torch.device("cpu"),
            force_cpu=True,
        )
        assert result is True

    def test_small_data_on_gpu(self):
        """Test that small data doesn't force CPU."""
        result = estimate_keep_on_cpu(
            n_voxels=1000,
            n_timepoints_total=100,
            device=torch.device("cpu"),
            force_cpu=False,
            data_threshold_gb=100.0,
        )
        assert result is False

    def test_large_data_forces_cpu(self):
        """Test that large data forces CPU storage."""
        result = estimate_keep_on_cpu(
            n_voxels=1000000,
            n_timepoints_total=1000,
            device=torch.device("cpu"),
            force_cpu=False,
            data_threshold_gb=0.1,
        )
        assert result is True


class TestIntegration:
    """Integration tests for memory module."""

    def teardown_method(self):
        """Reset global config after each test."""
        reset_memory_config()

    def test_config_affects_chunk_size(self):
        """Test that config changes affect chunk size estimation."""
        reset_memory_config()

        result_default = estimate_chunk_size(
            n_voxels=100000,
            n_timepoints=100,
            n_regressors=10,
            device=torch.device("cpu"),
            operation="glm",
        )

        aggressive = MemoryConfig(cpu_safety_factor=0.9, max_chunk_size_cpu=2000000)
        set_memory_config(aggressive)

        result_aggressive = estimate_chunk_size(
            n_voxels=100000,
            n_timepoints=100,
            n_regressors=10,
            device=torch.device("cpu"),
            operation="glm",
        )

        assert result_aggressive >= result_default

    def test_memory_estimation_consistency(self):
        """Test that memory estimation is consistent across calls."""
        params = {
            "n_voxels": 50000,
            "n_timepoints": 200,
            "n_regressors": 20,
            "device": torch.device("cpu"),
            "operation": "glm",
        }

        results = [estimate_chunk_size(**params) for _ in range(5)]
        assert len(set(results)) == 1
