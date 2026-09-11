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
    _MPS_CPU_OPS,
    _MPS_KEEP_OPS,
    _torch_at_least,
    calc_memory_usage,
    cpu_if_mps,
    get_device,
    linalg_lstsq,
    mps_is_good_for,
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

    def test_auto_prefers_mps_over_cpu(self):
        """A Mac user should get the GPU without having to ask for it."""
        with mock.patch("torch.cuda.is_available", return_value=False):
            with mock.patch("torch.backends.mps.is_available", return_value=True):
                with mock.patch("fastfuncstuff.utils.mps_fallback_active", return_value=True):
                    with mock.patch("fastfuncstuff.utils._torch_at_least", return_value=True):
                        assert get_device().type == "mps"

    def test_auto_declines_mps_without_the_cpu_fallback(self):
        """Without the fallback armed, an op Metal lacks raises mid-fit.

        Auto-detect must not hand the user a device that dies partway through a
        long run just because torch was imported before fastfuncstuff.
        """
        with mock.patch("torch.cuda.is_available", return_value=False):
            with mock.patch("torch.backends.mps.is_available", return_value=True):
                with mock.patch("fastfuncstuff.utils.mps_fallback_active", return_value=False):
                    with mock.patch("fastfuncstuff.utils._torch_at_least", return_value=True):
                        with pytest.warns(UserWarning, match="fallback did not arm"):
                            assert get_device().type == "cpu"

    def test_auto_declines_mps_on_old_torch(self):
        """The Metal policy table is measured against a specific torch."""
        with mock.patch("torch.cuda.is_available", return_value=False):
            with mock.patch("torch.backends.mps.is_available", return_value=True):
                with mock.patch("fastfuncstuff.utils._torch_at_least", return_value=False):
                    with pytest.warns(UserWarning, match="predates"):
                        assert get_device().type == "cpu"

    def test_explicit_mps_is_honoured(self):
        with mock.patch("torch.backends.mps.is_available", return_value=True):
            assert get_device("mps").type == "mps"

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


class TestFactorDevice:
    """Small float64 factorizations belong on the CPU when the data is on CUDA."""

    def test_cuda_and_mps_factor_on_cpu(self):
        from fastfuncstuff.utils import factor_device

        assert factor_device(torch.device("cuda")).type == "cpu"
        assert factor_device(torch.device("mps")).type == "cpu"

    def test_cpu_stays_put(self):
        from fastfuncstuff.utils import factor_device

        assert factor_device(torch.device("cpu")).type == "cpu"

    def test_distinct_from_linalg_device(self):
        """The two answer different questions and must not be conflated.

        linalg_device asks where float64 arithmetic is possible (CUDA: yes, so
        no-op); factor_device asks where a small float64 factorization is fast
        (CUDA: no, 1/64 rate plus cuSOLVER latency). Voxel-sized accumulators
        follow the first; fold-sized factorizations follow the second.
        """
        from fastfuncstuff.utils import factor_device, linalg_device

        assert linalg_device(torch.device("cuda")).type == "cuda"
        assert factor_device(torch.device("cuda")).type == "cpu"


class TestPinvF64:
    def test_matches_float64_pinv_and_round_trips_dtype(self):
        from fastfuncstuff.utils import pinv_f64

        torch.manual_seed(0)
        x = torch.randn(60, 12)
        reference = torch.linalg.pinv(x.double()).float()
        result = pinv_f64(x)

        assert result.dtype == x.dtype
        assert result.device == x.device
        assert torch.allclose(result, reference, atol=1e-6)

    def test_beats_float32_pinv_on_an_ill_conditioned_design(self):
        """The float64 promotion is the point; a float32 pinv loses accuracy."""
        from fastfuncstuff.utils import pinv_f64

        torch.manual_seed(1)
        x = torch.randn(200, 20)
        x[:, 1] = x[:, 0] + 1e-4 * torch.randn(200)  # nearly collinear pair
        exact = torch.linalg.pinv(x.double())

        promoted = (pinv_f64(x).double() - exact).abs().max()
        native = (torch.linalg.pinv(x).double() - exact).abs().max()
        assert promoted < native


class TestSaveR2CeilingStack:
    """The family's shared R2-plus-ceiling writer."""

    @staticmethod
    def _read(path):
        import nibabel as nib

        return nib.load(path)

    def test_stacks_in_order_with_labels(self, tmp_path):
        from fastfuncstuff.cli_utils import save_r2_ceiling_stack

        r2 = np.arange(8, dtype=np.float32)
        ceiling = r2 + 10.0
        explainable = r2 + 20.0
        path = save_r2_ceiling_stack(
            [(r2, "xval_R2"), (ceiling, "noise_ceiling"), (explainable, "explainable_R2")],
            str(tmp_path / "s.nii.gz"),
            (2, 2, 2),
            np.eye(4),
        )
        image = self._read(path)
        assert image.shape == (2, 2, 2, 3)
        for index, expected in enumerate((r2, ceiling, explainable)):
            assert np.allclose(image.get_fdata()[..., index].ravel(), expected)

    def test_none_layers_drop_out_leaving_a_plain_3d_map(self, tmp_path):
        """No ceiling must give a 3-D map, not a one-volume stack."""
        from fastfuncstuff.cli_utils import save_r2_ceiling_stack

        path = save_r2_ceiling_stack(
            [(np.arange(8, dtype=np.float32), "xval_R2"), (None, "noise_ceiling")],
            str(tmp_path / "s.nii.gz"),
            (2, 2, 2),
            np.eye(4),
        )
        assert self._read(path).shape == (2, 2, 2)

    def test_masked_input_is_unmasked_onto_the_grid(self, tmp_path):
        from fastfuncstuff.cli_utils import save_r2_ceiling_stack

        mask = np.zeros(8, dtype=bool)
        mask[[1, 4]] = True
        path = save_r2_ceiling_stack(
            [(np.array([5.0, 7.0], dtype=np.float32), "xval_R2")],
            str(tmp_path / "s.nii.gz"),
            (2, 2, 2),
            np.eye(4),
            mask_flat=mask,
        )
        assert np.allclose(self._read(path).get_fdata().ravel(), [0, 5, 0, 0, 7, 0, 0, 0])

    def test_all_none_is_an_error_not_an_empty_file(self, tmp_path):
        from fastfuncstuff.cli_utils import save_r2_ceiling_stack

        with pytest.raises(ValueError):
            save_r2_ceiling_stack(
                [(None, "xval_R2")], str(tmp_path / "s.nii.gz"), (2, 2, 2), np.eye(4)
            )


class TestToFactorF64:
    """to_factor_f64 sends small float64 factorizations to the CPU."""

    def test_cpu_tensor_is_promoted_in_place(self):
        from fastfuncstuff.utils import to_factor_f64

        x = torch.eye(4, dtype=torch.float32)
        out = to_factor_f64(x)
        assert out.dtype is torch.float64
        assert out.device.type == "cpu"

    def test_cuda_tensor_lands_on_the_cpu(self):
        """The whole point: a consumer card runs float64 at 1/64 rate."""
        from fastfuncstuff.utils import factor_device

        assert factor_device(torch.device("cuda")).type == "cpu"
        assert factor_device(torch.device("cpu")).type == "cpu"


class TestSymmetricDecorrelation:
    """The FastICA whitening step must not change device or dtype."""

    def test_returns_orthonormal_rows_on_the_input_device(self):
        from fastfuncstuff.decomposition.ica import FastICA

        torch.manual_seed(0)
        w = torch.randn(6, 6, dtype=torch.float32)
        out = FastICA._symmetric_decorrelation(w)
        assert out.dtype is w.dtype
        assert out.device == w.device
        assert torch.allclose(out @ out.T, torch.eye(6), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestMPSPolicy:
    """The single table that decides what runs on Metal (utils._MPS_CPU_OPS).

    These are policy tests, not performance tests: they pin the *shape* of the
    decision so a future edit cannot quietly send a 7-second QR back to the GPU.
    The numbers behind the table come from scripts/bench_mps_policy.py.
    """

    def test_cpu_if_mps_is_identity_off_metal(self):
        for dev in (torch.device("cpu"), torch.device("cuda:0")):
            assert cpu_if_mps(dev, "qr") == dev
            assert cpu_if_mps(dev, "grid_sample") == dev

    def test_slow_and_broken_ops_leave_metal(self):
        mps = torch.device("mps")
        # QR is the headline: torch 2.14 runs it natively and ~12000x slower
        # than Apple's CPU LAPACK at the tall-skinny shapes we use.
        assert cpu_if_mps(mps, "qr").type == "cpu"
        # Registered but aborts inside the Metal shader compiler, which the
        # PYTORCH_ENABLE_MPS_FALLBACK env var does not intercept.
        assert cpu_if_mps(mps, "lstsq").type == "cpu"
        assert cpu_if_mps(mps, "svd_batched").type == "cpu"

    def test_ops_metal_wins_stay_on_metal(self):
        mps = torch.device("mps")
        for op in ("grid_sample", "eigh_batched", "fft", "conv3d", "matmul"):
            assert cpu_if_mps(mps, op).type == "mps", op

    def test_unknown_op_raises_rather_than_defaulting_to_metal(self):
        """A typo must not silently leave work on a backend we never measured."""
        with pytest.raises(KeyError, match="unknown op"):
            cpu_if_mps(torch.device("mps"), "qrr")

    def test_batched_and_unbatched_eigh_disagree(self):
        """The split that makes the table worth having.

        Unbatched eigh loses to the CPU; the batched case wins by 3.2x. One
        'is eigh ok on MPS?' answer would be wrong for one of them.
        """
        mps = torch.device("mps")
        assert cpu_if_mps(mps, "eigh").type == "cpu"
        assert cpu_if_mps(mps, "eigh_batched").type == "mps"

    def test_every_reason_names_the_torch_it_was_measured_on(self):
        """A reason without a version is folklore; it must be re-checkable."""
        for op, reason in _MPS_CPU_OPS.items():
            assert "torch" in reason.lower(), f"{op}: {reason!r}"

    def test_keep_and_cpu_lists_are_disjoint(self):
        assert not (set(_MPS_CPU_OPS) & set(_MPS_KEEP_OPS))

    def test_mps_is_good_for_matches_the_table(self):
        assert not mps_is_good_for("qr")
        assert mps_is_good_for("grid_sample")

    def test_torch_version_floor_parses(self):
        assert _torch_at_least((0, 1))
        assert not _torch_at_least((99, 0))


class TestLinalgLstsq:
    """lstsq must survive Metal's shape-dependent shader-compiler failure."""

    def test_matches_torch_on_cpu(self):
        torch.manual_seed(0)
        a, b = torch.randn(64, 4), torch.randn(64, 3)
        assert torch.allclose(linalg_lstsq(a, b).solution, torch.linalg.lstsq(a, b).solution)

    def test_returns_a_full_result_tuple(self):
        """Call sites read .solution, but the other fields must survive the trip."""
        a, b = torch.randn(64, 4), torch.randn(64, 3)
        result = linalg_lstsq(a, b)
        assert hasattr(result, "solution")
        assert hasattr(result, "residuals")
        assert hasattr(result, "rank")

    @pytest.mark.gpu
    def test_survives_the_shape_that_kills_metal(self):
        """(256,32) reliably aborts in the Metal shader compiler on torch 2.14.

        Not a hypothetical: this exact shape raises RuntimeError through bare
        torch.linalg.lstsq, and it is a RuntimeError rather than a
        NotImplementedError, so PYTORCH_ENABLE_MPS_FALLBACK does not catch it.
        """
        if not torch.backends.mps.is_available():
            pytest.skip("no MPS")
        torch.manual_seed(0)
        a = torch.randn(256, 32, device="mps")
        b = torch.randn(256, 8, device="mps")
        got = linalg_lstsq(a, b).solution
        assert got.device.type == "mps"
        expected = torch.linalg.lstsq(a.cpu(), b.cpu()).solution
        assert torch.allclose(got.cpu(), expected, atol=1e-5)
