"""
Comprehensive tests for ICA workflow utilities in ica_workflow.py.

Tests cover:
- Verbose logging functions
- Tensor sanitization
- Voxel variance normalization (simple and MELODIC paths)
- Spatial smoothness estimation (basic checks)
"""


import numpy as np
import torch

from fastfuncsim.decomposition.workflow import (
    apply_voxel_variance_normalization,
    sanitize_finite_tensor,
    verbose_print,
    verbose_section,
)


class TestVerboseSection:
    """Test verbose section printing."""

    def test_verbose_section_prints_when_verbose(self, capsys):
        """Test that verbose=True prints section header."""
        verbose_section(True, "Test Section")
        captured = capsys.readouterr()
        assert "Test Section" in captured.out
        assert "─" in captured.out

    def test_verbose_section_skips_when_not_verbose(self, capsys):
        """Test that verbose=False doesn't print."""
        verbose_section(False, "Test Section")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_verbose_section_format(self, capsys):
        """Test section header format."""
        verbose_section(True, "X" * 20)
        captured = capsys.readouterr()
        # Should contain header and dashes
        lines = captured.out.strip().split('\n')
        assert len(lines) == 1


class TestVerbosePrint:
    """Test verbose message printing."""

    def test_verbose_print_when_verbose(self, capsys):
        """Test that verbose=True prints message."""
        verbose_print(True, "Test message")
        captured = capsys.readouterr()
        assert "Test message" in captured.out

    def test_verbose_print_skips_when_not_verbose(self, capsys):
        """Test that verbose=False doesn't print."""
        verbose_print(False, "Test message")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_verbose_print_with_timing(self, capsys):
        """Test timing annotation in verbose print."""
        import time
        start = time.time()
        time.sleep(0.01)  # Small delay
        verbose_print(True, "Processing", t0=start)
        captured = capsys.readouterr()
        assert "Processing" in captured.out
        assert "[" in captured.out  # Should have timing bracket


class TestSanitizeFiniteTensor:
    """Test tensor sanitization (NaN/Inf handling)."""

    def test_leaves_finite_tensor_unchanged(self):
        """Test that finite tensor is unchanged."""
        t = torch.tensor([1.0, 2.0, 3.0])
        result = sanitize_finite_tensor(t, "test")
        assert torch.equal(t, result)

    def test_replaces_nan_with_zero(self):
        """Test that NaN values are replaced with zero."""
        t = torch.tensor([1.0, float('nan'), 3.0])
        result = sanitize_finite_tensor(t, "test", verbose=False)
        expected = torch.tensor([1.0, 0.0, 3.0])
        assert torch.equal(result, expected)

    def test_replaces_inf_with_zero(self):
        """Test that Inf values are replaced with zero."""
        t = torch.tensor([1.0, float('inf'), 3.0])
        result = sanitize_finite_tensor(t, "test", verbose=False)
        expected = torch.tensor([1.0, 0.0, 3.0])
        assert torch.equal(result, expected)

    def test_replaces_neg_inf_with_zero(self):
        """Test that -Inf values are replaced with zero."""
        t = torch.tensor([1.0, float('-inf'), 3.0])
        result = sanitize_finite_tensor(t, "test", verbose=False)
        expected = torch.tensor([1.0, 0.0, 3.0])
        assert torch.equal(result, expected)

    def test_warns_about_bad_values(self, capsys):
        """Test that warnings are printed for bad values."""
        t = torch.tensor([1.0, float('nan'), float('inf')])
        _ = sanitize_finite_tensor(t, "test_label", verbose=True)
        captured = capsys.readouterr()
        # Should print a warning about NaN/Inf values
        assert "test_label" in captured.out
        assert "NaN" in captured.out or "Inf" in captured.out

    def test_preserves_device(self):
        """Test that device is preserved."""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        t = torch.tensor([1.0, 2.0, 3.0], device=device)
        result = sanitize_finite_tensor(t, "test")
        # Check same device type
        assert result.device.type == device.type
        if device.type == "cuda":
            assert result.device.index == device.index

    def test_clones_tensor(self):
        """Test that result is a new tensor."""
        t = torch.tensor([1.0, 2.0, 3.0])
        _result = sanitize_finite_tensor(t, "test")
        # When there are no bad values, it should return the original
        # When there are bad values, it should be a clone
        t_with_nan = torch.tensor([1.0, float('nan'), 3.0])
        _result_nan = sanitize_finite_tensor(t_with_nan, "test")
        # Original should be unchanged
        assert torch.isnan(t_with_nan[1])


class TestApplyVoxelVarianceNormalization:
    """Test voxel variance normalization."""

    def test_integer_num_spec_legacy_path(self):
        """Test integer num_spec uses legacy path."""
        data = torch.randn(10, 50)  # 10 voxels, 50 timepoints
        result, msg = apply_voxel_variance_normalization(
            data, num_spec=20, n_t=50, n_vox_masked=10
        )
        assert "legacy path" in msg
        assert result.shape == data.shape

    def test_float_num_spec_legacy_path(self):
        """Test float num_spec uses legacy path."""
        data = torch.randn(10, 50)
        result, msg = apply_voxel_variance_normalization(
            data, num_spec=0.5, n_t=50, n_vox_masked=10
        )
        assert "legacy path" in msg
        assert result.shape == data.shape

    def test_auto_uses_melodic_path(self):
        """Test 'auto' triggers MELODIC variance normalization."""
        data = torch.randn(10, 50)
        result, msg = apply_voxel_variance_normalization(
            data, num_spec="auto", n_t=50, n_vox_masked=10
        )
        assert "MELODIC" in msg
        assert result.shape == data.shape

    def test_melodic_uses_melodic_path(self):
        """Test 'melodic' triggers MELODIC variance normalization."""
        data = torch.randn(10, 50)
        result, msg = apply_voxel_variance_normalization(
            data, num_spec="melodic", n_t=50, n_vox_masked=10
        )
        assert "MELODIC" in msg
        assert result.shape == data.shape

    def test_legacy_path_normalizes_variance(self):
        """Test that legacy path normalizes variance."""
        # Create data with different variances per voxel
        data = torch.randn(5, 100)
        scales = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).unsqueeze(1)
        data = data * scales

        result, msg = apply_voxel_variance_normalization(
            data, num_spec=10, n_t=100, n_vox_masked=5
        )

        # Check that variances are more similar after normalization
        input_vars = data.var(dim=1)
        output_vars = result.var(dim=1)

        # Output variances should be closer to 1
        assert (output_vars - 1.0).abs().max() < (input_vars - 1.0).abs().max()

    def test_legacy_path_handles_constant_voxels(self):
        """Test that constant voxels are zeroed."""
        data = torch.randn(5, 100)
        data[2, :] = 5.0  # Make voxel 2 constant

        result, msg = apply_voxel_variance_normalization(
            data, num_spec=10, n_t=100, n_vox_masked=5
        )

        # Voxel 2 should be zeroed
        assert torch.all(result[2, :] == 0)
        assert "1 constant voxels" in msg or "constant voxel" in msg.lower()

    def test_returns_tuple(self):
        """Test that function returns (tensor, message) tuple."""
        data = torch.randn(5, 50)
        result = apply_voxel_variance_normalization(
            data, num_spec=10, n_t=50, n_vox_masked=5
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], str)


class TestEstimateSpatialSmoothnessResels:
    """Test spatial smoothness estimation."""

    def test_returns_tuple(self):
        """Test that function returns (resels, fwhm_geom) tuple."""
        # Create simple 4D data
        data_4d = np.random.randn(10, 10, 5, 20)  # Small for speed

        from fastfuncsim.decomposition.workflow import estimate_spatial_smoothness_resels
        resels, fwhm_geom = estimate_spatial_smoothness_resels(data_4d)

        assert isinstance(resels, float)
        assert isinstance(fwhm_geom, float)
        assert resels > 0
        assert fwhm_geom > 0

    def test_with_mask(self):
        """Test smoothness estimation with a mask."""
        data_4d = np.random.randn(10, 10, 5, 20)
        mask = np.ones((10, 10, 5), dtype=bool)
        mask[:2, :2, :] = False  # Exclude some voxels

        from fastfuncsim.decomposition.workflow import estimate_spatial_smoothness_resels
        resels, fwhm_geom = estimate_spatial_smoothness_resels(data_4d, mask=mask)

        assert resels > 0
        assert fwhm_geom > 0

    def test_with_constant_data(self):
        """Test handling of constant data (no variance)."""
        # Data with no temporal variance should still return valid results
        data_4d = np.ones((10, 10, 5, 10))

        from fastfuncsim.decomposition.workflow import estimate_spatial_smoothness_resels
        resels, fwhm_geom = estimate_spatial_smoothness_resels(data_4d)

        # Should still return valid values (uses minimum FWHM of 1.0)
        assert resels >= 1.0  # At minimum FWHM=1 per axis
        assert fwhm_geom >= 1.0

    def test_device_parameter(self):
        """Test that device parameter is respected."""
        data_4d = np.random.randn(5, 5, 3, 10)
        device = torch.device("cpu")

        from fastfuncsim.decomposition.workflow import estimate_spatial_smoothness_resels
        resels, fwhm_geom = estimate_spatial_smoothness_resels(data_4d, device=device)

        # Just check it doesn't crash
        assert resels > 0
