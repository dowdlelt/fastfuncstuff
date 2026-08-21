"""Parity and dispatch guards for fused forward CUDA interpolation."""

from __future__ import annotations

import importlib

import pytest
import torch

from fastfuncstuff.processing import interp

nwarpforge_module = importlib.import_module("fastfuncstuff.processing.nwarpforge")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _case(n: int = 41):
    torch.manual_seed(4)
    source = torch.randn(n, n + 1, n + 2, device="cuda")
    z, y, x = torch.meshgrid(
        torch.arange(n, device="cuda", dtype=torch.float32),
        torch.arange(n + 1, device="cuda", dtype=torch.float32),
        torch.arange(n + 2, device="cuda", dtype=torch.float32),
        indexing="ij",
    )
    # Exercise arbitrary fractions, clamped kernel taps, and center-OOB zeros.
    x = x + 0.37 + 0.11 * torch.sin(y * 0.2)
    y = y - 0.28 + 0.09 * torch.sin(z * 0.3)
    z = z + 0.19 + 0.07 * torch.sin(x * 0.1)
    x[0, 0, 0] = -0.6
    return source, x, y, z


@pytest.mark.parametrize("kernel", ["cubic", "quintic", "heptic", "wsinc5"])
def test_fused_matches_portable(monkeypatch, kernel):
    source, x, y, z = _case()
    monkeypatch.setenv("FFS_INTERP_NO_TRITON", "1")
    expected = interp._separable_resample_3d(source, x, y, z, kernel)
    monkeypatch.delenv("FFS_INTERP_NO_TRITON")
    got = interp._separable_resample_3d(source, x, y, z, kernel)
    assert torch.allclose(got, expected, atol=6e-6, rtol=2e-5)
    assert got[0, 0, 0] == 0


def test_coordinate_gradients_keep_portable_path(monkeypatch):
    source, x, y, z = _case()
    x.requires_grad_()

    def forbidden(*args, **kwargs):
        raise AssertionError("forward-only Triton path received an autograd input")

    monkeypatch.setattr(interp, "separable_resample_3d_triton", forbidden)
    out = interp._separable_resample_3d(source, x, y, z, "cubic")
    out.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_custom_wsinc_settings_keep_portable_path(monkeypatch):
    source, x, y, z = _case()
    monkeypatch.setenv("AFNI_WSINC5_RADIUS", "7")
    interp._wsinc5_params.cache_clear()

    def forbidden(*args, **kwargs):
        raise AssertionError("fused kernel does not implement custom AFNI wsinc settings")

    monkeypatch.setattr(interp, "separable_resample_3d_triton", forbidden)
    interp._separable_resample_3d(source, x, y, z, "wsinc5")
    interp._wsinc5_params.cache_clear()


def test_single_source_multi_path_reaches_fused_dispatch(monkeypatch):
    source, x, y, z = _case()
    calls = []

    def record(source_arg, x_arg, y_arg, z_arg, kernel):
        calls.append((source_arg.shape, kernel))
        return torch.zeros_like(x_arg)

    monkeypatch.setattr(interp, "separable_resample_3d_triton", record)
    out = interp.warp_image_multi([source], x, y, z, mode="wsinc5")

    assert calls == [(source.shape, "wsinc5")]
    assert out[0].shape == x.shape


def test_warp_components_reach_scalar_fused_dispatch(monkeypatch):
    source, x, y, z = _case()
    fields = (source, source + 1.0, source + 2.0)
    calls = []

    def record(source_arg, x_arg, y_arg, z_arg, kernel):
        calls.append((source_arg.ndim, kernel))
        return torch.zeros_like(x_arg)

    monkeypatch.setattr(nwarpforge_module, "_separable_resample_3d", record)
    out = nwarpforge_module._sample_fields(fields, x, y, z, interp="cubic")

    assert calls == [(3, "cubic")] * 3
    assert len(out) == 3 and all(component.shape == x.shape for component in out)
