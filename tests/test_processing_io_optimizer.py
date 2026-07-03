"""Tests for processing/io.py and processing/optimizer.py."""

import numpy as np
import torch

from fastfuncstuff.processing.io import derive_mean_output_path, load_image, save_image
from fastfuncstuff.processing.optimizer import (
    _coordinate_descent,
    optimize_warp_params_batched,
    optimize_warp_params_torch,
)

DEVICE = torch.device("cpu")


# ── derive_mean_output_path ──


class TestDeriveMeanOutputPath:
    def test_nii_gz(self):
        assert derive_mean_output_path("epi_mc.nii.gz") == "mean_epi_mc.nii.gz"

    def test_nii(self):
        assert derive_mean_output_path("out.nii") == "mean_out.nii"

    def test_no_extension(self):
        assert derive_mean_output_path("out") == "mean_out"

    def test_with_directory(self):
        result = derive_mean_output_path("/tmp/run01.nii.gz")
        assert result == "/tmp/mean_run01.nii.gz"


# ── load_image / save_image roundtrip ──


class TestLoadSaveImage:
    def test_roundtrip_3d(self, tmp_path):
        """Save then load a 3D volume."""
        vol = torch.randn(8, 10, 12)
        path = str(tmp_path / "test3d.nii.gz")
        save_image(vol, path)
        loaded, info = load_image(path)
        assert loaded.shape == vol.shape
        # Values should be close (float32 roundtrip)
        torch.testing.assert_close(loaded, vol, atol=1e-5, rtol=1e-5)

    def test_roundtrip_4d(self, tmp_path):
        """Save then load a 4D volume."""
        vol = torch.randn(5, 8, 10, 12)
        path = str(tmp_path / "test4d.nii.gz")
        save_image(vol, path)
        loaded, info = load_image(path)
        assert loaded.shape == vol.shape
        torch.testing.assert_close(loaded, vol, atol=1e-5, rtol=1e-5)

    def test_header_info_preserved(self, tmp_path):
        """Header info should roundtrip."""
        vol = torch.randn(6, 8, 10)
        path = str(tmp_path / "test.nii.gz")
        save_image(vol, path)
        _, info = load_image(path)
        assert "affine" in info
        assert "header" in info
        assert info["affine"].shape == (4, 4)

    def test_custom_affine(self, tmp_path):
        """Custom affine should be preserved."""
        vol = torch.randn(4, 6, 8)
        affine = np.diag([2.0, 2.0, 3.0, 1.0])
        path = str(tmp_path / "custom_aff.nii.gz")
        save_image(vol, path, affine=affine)
        _, info = load_image(path)
        np.testing.assert_allclose(info["affine"], affine, atol=1e-6)

    def test_device_placement(self, tmp_path):
        vol = torch.randn(4, 4, 4)
        path = str(tmp_path / "dev.nii.gz")
        save_image(vol, path)
        loaded, _ = load_image(path, device=DEVICE)
        assert loaded.device == DEVICE


# ── optimize_warp_params_torch ──


class TestOptimizeWarpParamsTorch:
    def test_quadratic_minimum(self):
        """Should find minimum of sum((x-1)^2)."""
        target = torch.tensor([1.0, 0.5, -0.3], device=DEVICE)

        def cost_fn(p):
            return ((p - target) ** 2).sum().item()

        best_p, best_c = optimize_warp_params_torch(
            cost_fn,
            n_params=3,
            param_max=2.0,
            device=DEVICE,
            max_iter=100,
            tolerance=1e-6,
        )
        assert best_c < 0.1
        torch.testing.assert_close(best_p, target, atol=0.2, rtol=0.2)

    def test_returns_zero_for_zero_min(self):
        """Minimum at origin should return near-zero params."""

        def cost_fn(p):
            return (p**2).sum().item()

        best_p, best_c = optimize_warp_params_torch(
            cost_fn,
            n_params=2,
            param_max=1.0,
            device=DEVICE,
        )
        assert best_c < 0.01


# ── _coordinate_descent ──


class TestCoordinateDescent:
    def test_finds_minimum(self):
        target = torch.tensor([0.5, -0.3], device=DEVICE)

        def cost_fn(p):
            return ((p - target) ** 2).sum().item()

        best_p, best_c = _coordinate_descent(
            cost_fn,
            n_params=2,
            param_max=1.0,
            device=DEVICE,
            max_iter=50,
            prad=0.3,
        )
        assert best_c < 0.1

    def test_respects_param_max(self):
        """Should never exceed param_max."""

        def cost_fn(p):
            # Minimum is at p=5 but param_max=1
            return ((p - 5) ** 2).sum().item()

        best_p, _ = _coordinate_descent(
            cost_fn,
            n_params=2,
            param_max=1.0,
            device=DEVICE,
            max_iter=20,
            prad=0.3,
        )
        assert (best_p.abs() <= 1.0 + 1e-6).all()


# ── optimize_warp_params_batched ──


class TestOptimizeWarpParamsBatched:
    def test_batched_quadratic(self):
        """Should optimize B quadratics in parallel."""
        B, n_params = 4, 3
        targets = torch.randn(B, n_params, device=DEVICE) * 0.3

        def batched_cost(p):
            return ((p - targets) ** 2).sum(dim=1)

        best_p, best_c, stats = optimize_warp_params_batched(
            batched_cost,
            B=B,
            n_params=n_params,
            param_max=1.0,
            device=DEVICE,
            max_iter=100,
            lr=0.05,
        )
        assert best_p.shape == (B, n_params)
        assert best_c.shape == (B,)
        assert (best_c < 0.1).all()
        assert stats.n_patches == B
        assert 0 < stats.steps_run <= 100

    def test_early_stopping(self):
        """Should stop early when converged."""
        B, n_params = 2, 2

        def cost_fn(p):
            return (p**2).sum(dim=1)

        best_p, best_c, stats = optimize_warp_params_batched(
            cost_fn,
            B=B,
            n_params=n_params,
            param_max=1.0,
            device=DEVICE,
            max_iter=500,
            tolerance=1e-6,
        )
        # Should have converged near zero, and stopped before the full budget
        assert (best_c < 0.01).all()
        assert stats.steps_run < 500
