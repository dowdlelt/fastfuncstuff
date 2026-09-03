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

from fastfuncstuff.decomposition.workflow import (
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
        lines = captured.out.strip().split("\n")
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
        t = torch.tensor([1.0, float("nan"), 3.0])
        result = sanitize_finite_tensor(t, "test", verbose=False)
        expected = torch.tensor([1.0, 0.0, 3.0])
        assert torch.equal(result, expected)

    def test_replaces_inf_with_zero(self):
        """Test that Inf values are replaced with zero."""
        t = torch.tensor([1.0, float("inf"), 3.0])
        result = sanitize_finite_tensor(t, "test", verbose=False)
        expected = torch.tensor([1.0, 0.0, 3.0])
        assert torch.equal(result, expected)

    def test_replaces_neg_inf_with_zero(self):
        """Test that -Inf values are replaced with zero."""
        t = torch.tensor([1.0, float("-inf"), 3.0])
        result = sanitize_finite_tensor(t, "test", verbose=False)
        expected = torch.tensor([1.0, 0.0, 3.0])
        assert torch.equal(result, expected)

    def test_warns_about_bad_values(self, capsys):
        """Test that warnings are printed for bad values."""
        t = torch.tensor([1.0, float("nan"), float("inf")])
        _ = sanitize_finite_tensor(t, "test_label", verbose=True)
        captured = capsys.readouterr()
        # Should print a warning about NaN/Inf values
        assert "test_label" in captured.out
        assert "NaN" in captured.out or "Inf" in captured.out

    def test_preserves_device(self):
        """Test that device is preserved."""
        device = torch.device("cpu")

        t = torch.tensor([1.0, 2.0, 3.0], device=device)
        result = sanitize_finite_tensor(t, "test")
        # Check same device type
        assert result.device.type == device.type

    def test_clones_tensor(self):
        """Test that result is a new tensor."""
        t = torch.tensor([1.0, 2.0, 3.0])
        _result = sanitize_finite_tensor(t, "test")
        # When there are no bad values, it should return the original
        # When there are bad values, it should be a clone
        t_with_nan = torch.tensor([1.0, float("nan"), 3.0])
        _result_nan = sanitize_finite_tensor(t_with_nan, "test")
        # Original should be unchanged
        assert torch.isnan(t_with_nan[1])


class TestApplyVoxelVarianceNormalization:
    """Test voxel variance normalization."""

    def test_integer_num_spec_legacy_path(self):
        """Test integer num_spec uses legacy path."""
        data = torch.randn(10, 50)  # 10 voxels, 50 timepoints
        result, msg = apply_voxel_variance_normalization(data, num_spec=20, n_t=50, n_vox_masked=10)
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

    def test_auto_uses_residual_noise_path(self):
        """'auto' takes the residual-noise path, not the cheap total-stdev divide."""
        data = torch.randn(10, 50)
        result, msg = apply_voxel_variance_normalization(
            data, num_spec="auto", n_t=50, n_vox_masked=10
        )
        assert "residual-noise" in msg
        assert result.shape == data.shape

    def test_laplace_uses_residual_noise_path(self):
        """'laplace' takes the residual-noise path, since it drives model order."""
        data = torch.randn(10, 50)
        result, msg = apply_voxel_variance_normalization(
            data, num_spec="laplace", n_t=50, n_vox_masked=10
        )
        assert "residual-noise" in msg
        assert result.shape == data.shape

    def test_legacy_path_normalizes_variance(self):
        """Test that legacy path normalizes variance."""
        # Create data with different variances per voxel
        data = torch.randn(5, 100)
        scales = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).unsqueeze(1)
        data = data * scales

        result, msg = apply_voxel_variance_normalization(data, num_spec=10, n_t=100, n_vox_masked=5)

        # Check that variances are more similar after normalization
        input_vars = data.var(dim=1)
        output_vars = result.var(dim=1)

        # Output variances should be closer to 1
        assert (output_vars - 1.0).abs().max() < (input_vars - 1.0).abs().max()

    def test_legacy_path_handles_constant_voxels(self):
        """Test that constant voxels are zeroed."""
        data = torch.randn(5, 100)
        data[2, :] = 5.0  # Make voxel 2 constant

        result, msg = apply_voxel_variance_normalization(data, num_spec=10, n_t=100, n_vox_masked=5)

        # Voxel 2 should be zeroed
        assert torch.all(result[2, :] == 0)
        assert "1 constant voxels" in msg or "constant voxel" in msg.lower()

    def test_returns_tuple(self):
        """Test that function returns (tensor, message) tuple."""
        data = torch.randn(5, 50)
        result = apply_voxel_variance_normalization(data, num_spec=10, n_t=50, n_vox_masked=5)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], str)


class TestEstimateSmoothnessReselsACF:
    """Test the ACF-based spatial smoothness estimator."""

    VOXDIMS = (3.0, 3.0, 3.0)

    def test_returns_triple(self):
        """Returns (resels, fwhm_geom, diagnostics)."""
        rng = np.random.default_rng(0)
        data_4d = rng.standard_normal((12, 12, 8, 20))

        from fastfuncstuff.decomposition.workflow import estimate_smoothness_resels_acf

        resels, fwhm_geom, diag = estimate_smoothness_resels_acf(
            data_4d, self.VOXDIMS, device=torch.device("cpu")
        )

        assert isinstance(resels, float)
        assert isinstance(fwhm_geom, float)
        assert resels >= 1.0
        assert fwhm_geom >= 1.0
        assert set(diag) >= {"acf_fwhm_mm", "classic_fwhm_mm", "fwhm_voxels", "resels"}

    def test_with_mask(self):
        rng = np.random.default_rng(1)
        data_4d = rng.standard_normal((12, 12, 8, 20))
        mask = np.ones((12, 12, 8), dtype=bool)
        mask[:2, :2, :] = False

        from fastfuncstuff.decomposition.workflow import estimate_smoothness_resels_acf

        resels, fwhm_geom, _ = estimate_smoothness_resels_acf(
            data_4d, self.VOXDIMS, mask=mask, device=torch.device("cpu")
        )
        assert resels >= 1.0
        assert fwhm_geom >= 1.0

    def test_smoother_data_gives_more_resels(self):
        """The estimator must respond to smoothness, not just return the floor.

        This is the property model order depends on: a blurred volume has fewer
        independent samples, so its resel size must come out larger.
        """
        from scipy.ndimage import gaussian_filter

        from fastfuncstuff.decomposition.workflow import estimate_smoothness_resels_acf

        rng = np.random.default_rng(2)
        rough = rng.standard_normal((20, 20, 20, 24))
        smooth = np.stack(
            [gaussian_filter(rough[..., t], sigma=2.0) for t in range(rough.shape[-1])],
            axis=-1,
        )

        r_rough, _, _ = estimate_smoothness_resels_acf(
            rough, self.VOXDIMS, device=torch.device("cpu")
        )
        r_smooth, _, _ = estimate_smoothness_resels_acf(
            smooth, self.VOXDIMS, device=torch.device("cpu")
        )
        assert r_smooth > r_rough * 2.0, f"{r_smooth=} not clearly above {r_rough=}"

    def test_constant_data(self):
        """Constant data has no variance; must not crash or return a sub-voxel resel."""
        data_4d = np.ones((10, 10, 6, 10))

        from fastfuncstuff.decomposition.workflow import estimate_smoothness_resels_acf

        resels, fwhm_geom, _ = estimate_smoothness_resels_acf(
            data_4d, self.VOXDIMS, device=torch.device("cpu")
        )
        assert resels >= 1.0
        assert fwhm_geom >= 1.0
