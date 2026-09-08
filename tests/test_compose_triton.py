"""The fused compose-and-sample must match the portable grid_sample path.

It replaces ~40 elementwise passes and two interleaved coordinate grids with one
kernel that gathers in voxel space, so the risk is not arithmetic but convention:
align_corners, border padding, and the degenerate-axis cases have to land exactly
where grid_sample puts them.
"""

import pytest
import torch

import fastfuncstuff.processing.interp as interp_mod
from fastfuncstuff.processing.interp import batched_compose_and_interpolate

triton_mod = pytest.importorskip("fastfuncstuff.processing.compose_triton")
compose_and_interpolate_triton = triton_mod.compose_and_interpolate_triton


def _inputs(b, v, nz, ny, nx, amp, dev):
    torch.manual_seed(9)
    return dict(
        source=torch.randn(nz, ny, nx, device=dev),
        warp_3ch=torch.randn(3, nz, ny, nx, device=dev) * amp,
        patch_xd=torch.randn(b, v, device=dev) * amp,
        patch_yd=torch.randn(b, v, device=dev) * amp,
        patch_zd=torch.randn(b, v, device=dev) * amp,
        base_i=torch.rand(b, v, device=dev) * (nx - 1),
        base_j=torch.rand(b, v, device=dev) * (ny - 1),
        base_k=torch.rand(b, v, device=dev) * (nz - 1),
    )


def _portable(a, nx, ny, nz):
    return batched_compose_and_interpolate(
        a["source"],
        None,
        None,
        None,
        a["patch_xd"],
        a["patch_yd"],
        a["patch_zd"],
        None,
        None,
        None,
        None,
        None,
        None,
        nx,
        ny,
        nz,
        global_warp_3ch=a["warp_3ch"],
        base_i=a["base_i"],
        base_j=a["base_j"],
        base_k=a["base_k"],
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "b,v,nz,ny,nx,amp",
    [
        (3, 200, 9, 9, 9, 1.0),  # displacements inside the volume
        (3, 200, 9, 9, 9, 8.0),  # and far outside it, so border clamping binds
        (5, 512, 12, 15, 11, 2.0),  # anisotropic
        (2, 64, 1, 8, 8, 2.0),  # degenerate z: grid_sample pins the coordinate
        (4, 64, 6, 1, 7, 2.0),  # degenerate y
        (2, 64, 5, 6, 1, 2.0),  # degenerate x
        (64, 729, 20, 20, 20, 1.5),  # a fine level
        (1, 65536, 40, 44, 40, 1.0),  # one patch over a whole grid, as at level 0
    ],
)
def test_fused_compose_matches_portable(monkeypatch, b, v, nz, ny, nx, amp):
    if not torch.cuda.is_available():
        pytest.skip("the fused compose is CUDA-only")
    dev = torch.device("cuda")
    a = _inputs(b, v, nz, ny, nx, amp, dev)

    # batched_compose_and_interpolate now takes the fused path itself, so the
    # reference has to be the grid_sample path it replaced.
    monkeypatch.setattr(interp_mod, "compose_and_interpolate_triton", None)
    expected = _portable(a, nx, ny, nz)
    monkeypatch.undo()
    actual = compose_and_interpolate_triton(
        a["source"],
        a["warp_3ch"],
        a["patch_xd"],
        a["patch_yd"],
        a["patch_zd"],
        a["base_i"],
        a["base_j"],
        a["base_k"],
    )

    assert len(actual) == 4
    for got, want in zip(actual, expected, strict=True):
        assert got.shape == (b, v)
        scale = want.abs().max().clamp_min(1e-6)
        assert float((got - want).abs().max() / scale) < 2e-5


@pytest.mark.gpu
def test_autograd_path_keeps_the_portable_compose():
    """Under grad, the non-differentiable kernel must not be taken."""
    if not torch.cuda.is_available():
        pytest.skip("the fused compose is CUDA-only")
    dev = torch.device("cuda")
    a = _inputs(2, 128, 8, 9, 10, 1.0, dev)
    a["patch_xd"].requires_grad_(True)

    with torch.enable_grad():
        warped, ah_x, _, _ = _portable(a, 10, 9, 8)
        warped.sum().backward()

    assert a["patch_xd"].grad is not None
    assert torch.isfinite(a["patch_xd"].grad).all()
    assert float(a["patch_xd"].grad.abs().sum()) > 0.0
