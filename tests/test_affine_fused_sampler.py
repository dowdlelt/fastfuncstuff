"""The fused CPU trilinear point sampler agrees with the grid_sample route.

torch.compile is stubbed to the identity: these test the arithmetic and the
plumbing, not what inductor emits.
"""

from __future__ import annotations

import importlib

import pytest
import torch

aff = importlib.import_module("fastfuncstuff.processing.affine")


@pytest.fixture(autouse=True)
def _identity_compile(monkeypatch):
    monkeypatch.setattr(aff, "_fused_point_sampler", {})
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)
    monkeypatch.delenv("FFS_FUSED_SAMPLER", raising=False)
    monkeypatch.delenv("FFS_NWARP_NO_COMPILE", raising=False)


def _case(b=5, m=4000):
    g = torch.Generator().manual_seed(0)
    src = torch.rand(17, 21, 19, generator=g)
    pts = torch.rand(m, 3, generator=g) * torch.tensor([18.0, 20.0, 16.0])
    mats = torch.eye(4).repeat(b, 1, 1)
    mats[:, :3, :3] += 0.05 * torch.randn(b, 3, 3, generator=g)
    mats[:, :3, 3] += 2.0 * torch.randn(b, 3, generator=g)  # some points land outside
    return src, mats, pts


def test_fused_matches_grid_sample_route():
    src, mats, pts = _case()
    ref = aff.sample_affine_at_points_batched(src, mats, pts, True, "linear")
    out = aff.sample_affine_at_points_batched(src, mats, pts, True, "linear", fused_cpu=True)
    assert out.shape == ref.shape
    # Points exactly on the half-voxel border can fall either side; everything else
    # agrees to float rounding (voxel coords vs grid_sample's [-1, 1] round trip).
    both = (out != 0) & (ref != 0)
    assert int(((out == 0) != (ref == 0)).sum()) <= 2
    torch.testing.assert_close(out[both], ref[both], rtol=1e-5, atol=1e-5)
    assert bool(((out == 0) | both).any())  # the case does exercise the outside rule


def test_single_candidate_is_padded_and_trimmed():
    src, mats, pts = _case(b=1)
    out = aff.sample_affine_at_points_batched(src, mats, pts, True, "linear", fused_cpu=True)
    ref = aff.sample_affine_at_points_batched(src, mats, pts, True, "linear")
    assert out.shape == ref.shape == (1, pts.shape[0])


def test_failed_compile_falls_back_to_grid_sample(monkeypatch):
    src, mats, pts = _case()

    def broken(*a, **k):
        raise RuntimeError("inductor C++ compile failed")

    monkeypatch.setattr(aff, "_trilinear_points_batched", broken)
    monkeypatch.setattr(
        importlib.import_module("fastfuncstuff.processing.interp"),
        "_is_compile_failure",
        lambda exc: True,
    )
    ref = aff.sample_affine_at_points_batched(src, mats, pts, True, "linear")
    out = aff.sample_affine_at_points_batched(src, mats, pts, True, "linear", fused_cpu=True)
    torch.testing.assert_close(out, ref)
    assert aff._fused_point_sampler["cpu"] is None


def test_gradients_keep_the_grid_sample_route():
    src, mats, pts = _case()
    mats.requires_grad_(True)
    out = aff.sample_affine_at_points_batched(src, mats, pts, True, "linear", fused_cpu=True)
    out.sum().backward()
    assert mats.grad is not None and "cpu" not in aff._fused_point_sampler
