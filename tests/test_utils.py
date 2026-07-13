"""
Tests for utility functions in fastfuncstuff/utils.py

These tests cover device management, tensor conversion,
memory calculations, and chunk size optimization.
"""

import warnings
from unittest import mock

import numpy as np
import pytest
import torch

from fastfuncstuff.memory import estimate_chunk_size
from fastfuncstuff.utils import (
    calc_memory_usage,
    get_device,
    print_device_info,
    to_tensor,
)


class TestGetDevice:
    """Test device selection logic"""

    def test_prefer_cuda_when_available(self):
        """Test requesting CUDA when it's available"""
        with mock.patch("torch.cuda.is_available", return_value=True):
            device = get_device(prefer_device="cuda")
            assert device.type == "cuda"

    def test_prefer_cuda_when_unavailable_raises(self):
        """Test requesting CUDA when unavailable raises RuntimeError"""
        with mock.patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA device requested but not available"):
                get_device(prefer_device="cuda")

    def test_prefer_mps_when_available(self):
        """Test requesting MPS when it's available"""
        with mock.patch("torch.backends.mps.is_available", return_value=True):
            device = get_device(prefer_device="mps")
            assert device.type == "mps"

    def test_prefer_mps_when_unavailable_raises(self):
        """Test requesting MPS when unavailable raises RuntimeError"""
        with mock.patch("torch.backends.mps.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="MPS device requested but not available"):
                get_device(prefer_device="mps")

    def test_prefer_cpu_on_non_mac(self):
        """Requesting CPU is honoured on any platform, with no warning."""
        with mock.patch("platform.system", return_value="Linux"):
            with mock.patch("torch.backends.mps.is_available", return_value=False):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    device = get_device(prefer_device="cpu")
                    assert device.type == "cpu"
                    # CPU is a first-class device choice — no warning on request.
                    assert not any("CPU" in str(x.message) for x in w)

    def test_prefer_cpu_on_mac_returns_cpu(self):
        """Requesting CPU on macOS is honoured (no longer raises)."""
        with mock.patch("platform.system", return_value="Darwin"):
            with mock.patch("torch.backends.mps.is_available", return_value=True):
                device = get_device(prefer_device="cpu")
                assert device.type == "cpu"

    def test_invalid_device_preference(self):
        """Test invalid device preference raises ValueError"""
        with pytest.raises(ValueError, match="Unknown prefer_device"):
            get_device(prefer_device="tpu")

    def test_auto_select_mps_when_available(self):
        """Test auto-selecting MPS when available (and CUDA is not)."""
        with mock.patch("torch.cuda.is_available", return_value=False):
            with mock.patch("torch.backends.mps.is_available", return_value=True):
                device = get_device()
                assert device.type == "mps"

    def test_auto_select_cuda_when_mps_unavailable(self):
        """Test auto-selecting CUDA when MPS unavailable on non-Mac"""
        with mock.patch("torch.backends.mps.is_available", return_value=False):
            with mock.patch("platform.system", return_value="Linux"):
                with mock.patch("torch.cuda.is_available", return_value=True):
                    device = get_device()
                    assert device.type == "cuda"

    def test_fallback_to_cpu_on_non_mac(self):
        """Test falling back to CPU when no GPU on non-Mac"""
        with mock.patch("torch.backends.mps.is_available", return_value=False):
            with mock.patch("platform.system", return_value="Linux"):
                with mock.patch("torch.cuda.is_available", return_value=False):
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        device = get_device()
                        assert device.type == "cpu"
                        assert len(w) == 1
                        assert "No GPU backend detected" in str(w[0].message)

    def test_mac_without_mps_falls_back_to_cpu(self):
        """macOS without MPS (and no CUDA) falls back to CPU with a warning."""
        with mock.patch("torch.backends.mps.is_available", return_value=False):
            with mock.patch("platform.system", return_value="Darwin"):
                with mock.patch("torch.cuda.is_available", return_value=False):
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        device = get_device()
                        assert device.type == "cpu"
                        assert any("No GPU backend detected" in str(x.message) for x in w)


class TestPrintDeviceInfo:
    """Test device info printing"""

    def test_print_cuda_info(self, capsys):
        """Test printing CUDA device info"""
        mock_props = mock.MagicMock()
        mock_props.total_memory = 16e9  # 16 GB

        with mock.patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"):
            with mock.patch("torch.cuda.get_device_properties", return_value=mock_props):
                print_device_info(torch.device("cuda"))
                captured = capsys.readouterr()
                assert "NVIDIA RTX 4090" in captured.out
                assert "16.00 GB" in captured.out

    def test_print_mps_info(self, capsys):
        """Test printing MPS device info"""
        print_device_info(torch.device("mps"))
        captured = capsys.readouterr()
        assert "Apple Metal Performance Shaders" in captured.out

    def test_print_cpu_info(self, capsys):
        """Test printing CPU device info"""
        print_device_info(torch.device("cpu"))
        captured = capsys.readouterr()
        assert "Using CPU" in captured.out


class TestToTensor:
    """Test tensor conversion utility"""

    def test_numpy_to_tensor(self):
        """Test converting numpy array to tensor"""
        arr = np.array([1.0, 2.0, 3.0])
        tensor = to_tensor(arr)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        np.testing.assert_array_almost_equal(tensor.numpy(), arr)

    def test_list_to_tensor(self):
        """Test converting list to tensor"""
        data = [1.0, 2.0, 3.0]
        tensor = to_tensor(data)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert list(tensor.numpy()) == data

    def test_tuple_to_tensor(self):
        """Test converting tuple to tensor"""
        data = (1.0, 2.0, 3.0)
        tensor = to_tensor(data)
        assert isinstance(tensor, torch.Tensor)
        assert len(tensor) == 3

    def test_tensor_passthrough(self):
        """Test that tensor is converted to correct dtype"""
        original = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        tensor = to_tensor(original, dtype=torch.float32)
        assert tensor.dtype == torch.float32

    def test_with_device(self):
        """Test tensor conversion with device specification"""
        arr = np.array([1.0, 2.0, 3.0])
        tensor = to_tensor(arr, device=torch.device("cpu"))
        assert tensor.device.type == "cpu"


class TestCalcMemoryUsage:
    """Test memory calculation utility"""

    def test_simple_shape(self):
        """Test memory calculation for simple shape"""
        # 1000 float32 elements = 4000 bytes = 4e-6 GB
        memory_gb = calc_memory_usage((1000,), dtype=torch.float32)
        assert abs(memory_gb - 4e-6) < 1e-10

    def test_multidimensional_shape(self):
        """Test memory calculation for multi-dimensional shape"""
        # 100 x 100 x 100 float32 = 1e6 * 4 bytes = 4e-3 GB
        memory_gb = calc_memory_usage((100, 100, 100), dtype=torch.float32)
        assert abs(memory_gb - 4e-3) < 1e-7

    def test_different_dtype(self):
        """Test memory calculation with different dtype"""
        # 1000 float64 elements = 8000 bytes = 8e-6 GB
        memory_gb = calc_memory_usage((1000,), dtype=torch.float64)
        assert abs(memory_gb - 8e-6) < 1e-10


class TestOptimalChunkSize:
    """Test optimal chunk size calculation"""

    def test_cuda_chunk_size(self):
        """Test chunk size calculation for CUDA device"""
        mock_props = mock.MagicMock()
        mock_props.total_memory = 16e9  # 16 GB

        with mock.patch("torch.cuda.get_device_properties", return_value=mock_props):
            with mock.patch("torch.cuda.memory_reserved", return_value=0):
                chunk_size = estimate_chunk_size(
                    n_voxels=100000,
                    n_timepoints=1000,
                    n_regressors=50,
                    device=torch.device("cuda"),
                    operation="glm",
                )
                assert chunk_size >= 1000
                assert chunk_size <= 100000

    def test_mps_chunk_size(self):
        """Test chunk size calculation for MPS device"""
        chunk_size = estimate_chunk_size(
            n_voxels=100000,
            n_timepoints=1000,
            n_regressors=50,
            device=torch.device("mps"),
            operation="glm",
        )
        # MPS uses conservative 4GB estimate
        assert chunk_size >= 1000
        assert chunk_size <= 100000

    def test_cpu_chunk_size(self):
        """Test chunk size calculation for CPU device"""
        chunk_size = estimate_chunk_size(
            n_voxels=100000,
            n_timepoints=1000,
            n_regressors=50,
            device=torch.device("cpu"),
            operation="glm",
        )
        # CPU uses 8GB estimate
        assert chunk_size >= 1000
        assert chunk_size <= 100000

    def test_small_dataset_returns_all(self):
        """Test that small datasets return all voxels"""
        chunk_size = estimate_chunk_size(
            n_voxels=500,
            n_timepoints=100,
            n_regressors=20,
            device=torch.device("cpu"),
            operation="glm",
        )
        # Should return all 500 voxels for small dataset
        assert chunk_size == 500

    def test_minimum_chunk_size(self):
        """Test that chunk size respects minimum bounds"""
        # Very large memory requirement per voxel
        chunk_size = estimate_chunk_size(
            n_voxels=100000,
            n_timepoints=10000,
            n_regressors=1000,
            device=torch.device("cpu"),
            operation="glm",
            safety_factor=0.01,  # Very conservative
        )
        # Should be at least min_chunk
        assert chunk_size >= 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
