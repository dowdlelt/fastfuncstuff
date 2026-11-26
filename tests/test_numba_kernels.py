"""CPU Numba kernels (processing/numba_kernels.py) against the torch paths they replace.

Each kernel claims either bit-identity with its torch route (inversion, Jacobian, warp)
or agreement to float32 rounding (the smoothing, the fused local correlation). The torch
routes are reached by pretending to be under torch.compile, which is exactly the switch
the code uses to stay off Numba.
"""

from __future__ import annotations

import pytest
import torch

import fastfuncstuff.processing.formwarp as fw
from fastfuncstuff.processing.cost import LocalCorrelationCPU, _separable_smooth_3d
from fastfuncstuff.processing.interp import trilinear_interpolate_multi

SHAPES = [(9, 11, 10), (5, 6, 7), (2, 9, 4), (3, 1, 8)]


@pytest.fixture
def torch_route(monkeypatch):
    def force():
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    return force


def _field(shape, scale=2.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(shape, generator=g) * scale for _ in range(3))


@pytest.mark.parametrize("shape", [(30, 7, 40), (5, 6, 7), (1, 4, 9)])
@pytest.mark.parametrize("kernel", ["gauss", "box"])
def test_smoothing_and_its_adjoint_match_the_torch_path(shape, kernel, torch_route):
    """Axes shorter than the kernel (radius 12 on a 5-voxel axis) included."""
    vol = torch.randn(1, 2, *shape, generator=torch.Generator().manual_seed(1))
    vol.requires_grad_(True)
    new = _separable_smooth_3d(vol, 4.0 if kernel == "gauss" else 3, kernel)
    seed = torch.randn_like(new)
    (g_new,) = torch.autograd.grad(new, vol, seed)
    torch_route()
    old = _separable_smooth_3d(vol, 4.0 if kernel == "gauss" else 3, kernel)
    (g_old,) = torch.autograd.grad(old, vol, seed)
    torch.testing.assert_close(new, old, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(g_new, g_old, rtol=1e-5, atol=1e-6)


def test_smoothing_backward_is_the_exact_adjoint():
    vol = torch.randn(1, 2, 9, 4, 11, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: _separable_smooth_3d(x, 4.0, "gauss"), (vol,))


def _torch_inversion(xd, yd, zd, n_iter):
    nz, ny, nx = xd.shape
    kk, jj, ii = torch.meshgrid(
        *(torch.arange(n, dtype=torch.float32) for n in (nz, ny, nx)), indexing="ij"
    )
    ex, ey, ez = -xd, -yd, -zd
    field = torch.stack([xd, yd, zd])
    for _ in range(n_iter):
        s = trilinear_interpolate_multi(
            field, (ii + ex).reshape(-1), (jj + ey).reshape(-1), (kk + ez).reshape(-1)
        )
        dx, dy, dz = s.T.reshape(3, nz, ny, nx).unbind(0)
        ex, ey, ez = -dx, -dy, -dz
    return ex, ey, ez


@pytest.mark.parametrize("shape", SHAPES)
def test_fused_inversion_is_bit_identical_to_the_grid_sample_sweeps(shape):
    xd, yd, zd = _field(shape)
    new = fw.invert_displacement_field(xd, yd, zd, n_iter=6)
    for a, b in zip(new, _torch_inversion(xd, yd, zd, 6), strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("shape", SHAPES)
def test_fused_jacobian_and_penalty_offenders_are_bit_identical(shape, torch_route):
    from fastfuncstuff.processing.numba_kernels import jacobian_cpu
    from fastfuncstuff.processing.penalty import compute_jacobian_energy, penalty_energy

    field = _field(shape, scale=1.5)
    det_new = fw.jacobian_determinant(*field)
    owing_new = jacobian_cpu(*field, cut=1.0)[1]
    torch_route()
    assert torch.equal(det_new, fw.jacobian_determinant(*field))
    je, se = compute_jacobian_energy(*field)
    assert torch.equal(owing_new, penalty_energy(je, se) > 0)


@pytest.mark.parametrize("shape", SHAPES)
def test_fused_warp_and_its_displacement_gradient_are_bit_identical(shape, torch_route):
    src = torch.rand(shape, generator=torch.Generator().manual_seed(3))
    field = _field(shape, scale=3.0)

    def run():
        leaves = [t.clone().requires_grad_(True) for t in field]
        out = fw._warp_diff(src, *leaves)
        return out, torch.autograd.grad((out * out).sum(), leaves)

    out_new, g_new = run()
    torch_route()
    out_old, g_old = run()
    assert torch.equal(out_new, out_old)
    assert all(torch.equal(a, b) for a, b in zip(g_new, g_old, strict=True))


@pytest.mark.parametrize("mode,sigma,kernel", [("lpa", 4.0, "gauss"), ("cc", 3, "box")])
def test_fused_local_correlation_matches_autograd(mode, sigma, kernel, torch_route):
    g = torch.Generator().manual_seed(4)
    shape = (12, 9, 14)
    a = torch.rand(shape, generator=g) * 100
    b = 0.5 * a + torch.rand(shape, generator=g) * 40
    w = (torch.rand(shape, generator=g) > 0.3).float()

    la, lb = a.clone().requires_grad_(True), b.clone().requires_grad_(True)
    new = LocalCorrelationCPU.apply(la, lb, w, None, float(sigma), kernel, mode)
    g_new = torch.autograd.grad(new, (la, lb))

    torch_route()
    la, lb = a.clone().requires_grad_(True), b.clone().requires_grad_(True)
    old = fw.image_metric(la, lb, w, metric=mode, cc_radius=int(sigma), lpa_sigma=sigma)
    g_old = torch.autograd.grad(old, (la, lb))

    # Measured: cost within 2e-7, gradients within 2e-6 relative (float32 op order and a
    # double sum here against autograd's float32 chain).
    torch.testing.assert_close(new.detach(), old.detach(), rtol=1e-6, atol=1e-6)
    for x, y in zip(g_new, g_old, strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=1e-5 * float(y.abs().max()))
