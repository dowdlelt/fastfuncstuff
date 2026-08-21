"""Geometry, parity, and fidelity tests for locomoco interpolation."""

from __future__ import annotations

import pytest
import torch

from fastfuncstuff.processing import locomoco as lm


def test_cli_auto_prefers_fidelity_by_geometry():
    from fastfuncstuff.cli.locomoco import _resolve_warp_interp

    assert _resolve_warp_interp("auto", rotaware=False) == "lanczos"
    assert _resolve_warp_interp("auto", rotaware=True) == "bicubic"
    assert _resolve_warp_interp("bilinear", rotaware=False) == "bilinear"


def test_portable_2d_cubic_matches_grid_sample():
    torch.manual_seed(2)
    image = torch.randn(3, 17, 19)
    u = torch.randn_like(image) * 0.3
    v = torch.randn_like(image) * 0.3
    expected = lm._warp2d(image, u, v, "bicubic")
    got = lm._shift2d_high_order(image, v, u, 1, 2, mode="bicubic")
    assert torch.allclose(got, expected, atol=3e-6, rtol=2e-5)


def test_lanczos_preserves_flat_field_in_two_dimensions():
    image = torch.full((2, 15, 18), 7.25)
    u = torch.randn_like(image) * 0.45
    v = torch.randn_like(image) * 0.45
    got = lm._shift2d_high_order(image, v, u, 1, 2, mode="lanczos", radius=3)
    assert torch.allclose(got, image, atol=2e-5)


def test_lanczos_beats_linear_on_subvoxel_high_frequency_signal():
    h, w = 48, 52
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    image = (torch.sin(0.72 * xx) + 0.7 * torch.cos(0.61 * yy))[None]
    u = torch.full_like(image, 0.37)
    v = torch.full_like(image, -0.29)
    truth = (torch.sin(0.72 * (xx + 0.37)) + 0.7 * torch.cos(0.61 * (yy - 0.29)))[None]
    linear = lm._warp2d(image, u, v, "bilinear")
    lanczos = lm._shift2d_high_order(image, v, u, 1, 2, mode="lanczos", radius=3)
    interior = (..., slice(4, -4), slice(4, -4))
    linear_rmse = (linear[interior] - truth[interior]).square().mean().sqrt()
    lanczos_rmse = (lanczos[interior] - truth[interior]).square().mean().sqrt()
    assert lanczos_rmse < 0.2 * linear_rmse


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_fused_1d_lanczos_matches_portable(monkeypatch, axis):
    torch.manual_seed(3)
    vol = torch.randn(2, 13, 14, 15, device="cuda")
    shift = torch.randn_like(vol) * 0.35
    fused = lm._shift1d_windowed_sinc(vol, shift, axis, radius=3)
    monkeypatch.setattr(lm, "shift1d_lanczos_triton", None)
    portable = lm._shift1d_windowed_sinc(vol, shift, axis, radius=3)
    assert torch.allclose(fused, portable, atol=2e-6, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["bicubic", "lanczos"])
def test_fused_2d_embedded_in_3d_matches_portable(monkeypatch, mode):
    torch.manual_seed(5)
    vol = torch.randn(2, 12, 13, 14, device="cuda")
    shifts = [torch.randn_like(vol) * 0.3, torch.randn_like(vol) * 0.3]
    fused = lm._shift3d_axes(vol, shifts, [0, 2], mode=mode, radius=3)
    monkeypatch.setattr(lm, "shift2d_triton", None)
    portable = lm._shift3d_axes(vol, shifts, [0, 2], mode=mode, radius=3)
    assert torch.allclose(fused, portable, atol=7e-6, rtol=3e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_autograd_keeps_portable_2d_path(monkeypatch):
    vol = torch.randn(1, 9, 10, device="cuda")
    u = torch.zeros_like(vol, requires_grad=True)
    v = torch.zeros_like(vol, requires_grad=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("forward-only kernel received gradient coordinates")

    monkeypatch.setattr(lm, "shift2d_triton", forbidden)
    lm._warp2d(vol, u, v, "lanczos").square().mean().backward()
    assert u.grad is not None and v.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_launch_chunking_is_exact(monkeypatch):
    from fastfuncstuff import memory
    from fastfuncstuff.processing.locomoco_interp_triton import shift2d_triton

    torch.manual_seed(8)
    vol = torch.randn(2, 11, 13, device="cuda")
    v = torch.randn_like(vol) * 0.3
    u = torch.randn_like(vol) * 0.3
    reference = shift2d_triton(vol, v, u, 1, 2, "lanczos", 3)
    # 12 bytes/point gives ~41 points/launch: several non-aligned chunks.
    monkeypatch.setattr(memory, "get_available_memory", lambda *args, **kwargs: 500)
    chunked = shift2d_triton(vol, v, u, 1, 2, "lanczos", 3)
    assert torch.equal(chunked, reference)
