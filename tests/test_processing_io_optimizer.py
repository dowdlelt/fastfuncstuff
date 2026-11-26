"""Tests for processing/io.py and processing/optimizer.py."""

import nibabel as nib
import numpy as np
import torch

import fastfuncstuff.processing.io as processing_io
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

    def test_load_uses_shared_axis_reversal(self, monkeypatch):
        source = np.asfortranarray(np.arange(4 * 5 * 6 * 7, dtype=np.float32).reshape(4, 5, 6, 7))
        image = nib.Nifti1Image(source, np.eye(4))
        captured = {}
        real_from_numpy = torch.from_numpy

        def capture(array):
            captured["array"] = array
            return real_from_numpy(array)

        monkeypatch.setattr(processing_io, "load_nifti", lambda _path: image)
        monkeypatch.setattr(processing_io.torch, "from_numpy", capture)

        loaded, _ = processing_io.load_image("ignored.nii")

        assert np.shares_memory(captured["array"], source)
        assert loaded.is_contiguous()
        torch.testing.assert_close(loaded, real_from_numpy(source.transpose(3, 2, 1, 0)))

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


class TestGaussNewtonStopRule:
    """A first step that overshoots must not end the fit under the freeze rule."""

    @staticmethod
    def _problem():
        # Cost (p - 1)^2 per patch; the model's curvature is 10x too small, so the
        # undamped step lands at p = 10 (cost 81) and only a damped retry gains.
        def normal_eqs(p):
            b = p.shape[0]
            return torch.full((b, 1, 1), 0.2), 2.0 * (1.0 - p)

        def cost(p):
            return ((p - 1.0) ** 2).sum(-1)

        return normal_eqs, cost

    def test_first_reject_stops_on_the_overshoot(self):
        from fastfuncstuff.processing.optimizer import optimize_warp_params_gauss_newton

        ne, cost = self._problem()
        params, costs, stats = optimize_warp_params_gauss_newton(
            ne, cost, 3, 1, 20.0, torch.device("cpu"), max_iter=12
        )
        assert stats.steps_run == 1
        assert torch.equal(params, torch.zeros(3, 1))

    def test_freeze_retries_with_more_damping(self):
        from fastfuncstuff.processing.optimizer import optimize_warp_params_gauss_newton

        ne, cost = self._problem()
        calls = []

        def counted(p):
            calls.append(p.clone())
            return ne(p)

        params, costs, stats = optimize_warp_params_gauss_newton(
            counted, cost, 3, 1, 20.0, torch.device("cpu"), max_iter=12, stop_rule="freeze"
        )
        assert bool((costs < 0.5).all())
        # Rejected rounds leave the parameters where they were, so the normal
        # equations are not rebuilt for them.
        assert len(calls) < stats.steps_run


def test_gn_ball_constraint_bounds_the_norm_not_each_coefficient():
    """AFNI's BALLOPT: one coefficient may exceed the box half-width, the norm may not."""
    from fastfuncstuff.processing.optimizer import optimize_warp_params_gauss_newton

    target = torch.tensor([[0.5, 0.0, 0.0]])  # far outside both bounds, along one axis

    def normal_eqs(p):
        return torch.eye(3).expand(1, 3, 3).clone(), (target - p)

    def cost(p):
        return ((p - target) ** 2).sum(-1)

    box, _, _ = optimize_warp_params_gauss_newton(
        normal_eqs, cost, 1, 3, 0.0421, torch.device("cpu"), max_iter=4
    )
    ball, _, _ = optimize_warp_params_gauss_newton(
        normal_eqs, cost, 1, 3, 0.0421, torch.device("cpu"), max_iter=4, ball_radius=0.1141
    )
    assert box.abs().max() <= 0.0421 + 1e-7
    assert ball[0, 0] > 0.0421 * 2
    assert ball.norm() <= 0.1141 * (1 + 1e-6)
