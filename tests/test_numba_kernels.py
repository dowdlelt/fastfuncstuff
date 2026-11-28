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


@pytest.mark.parametrize(
    "vol,patch,n_patch",
    [((20, 23, 21), (5, 6, 4), 7), ((9, 1, 8), (4, 1, 5), 3), ((12, 14, 13), (12, 14, 13), 1)],
)
def test_fused_compose_and_sample_is_bit_identical(vol, patch, n_patch, torch_route):
    """qwarp's compose-and-sample, and the gradient sample Gauss-Newton takes after it."""
    from fastfuncstuff.processing.interp import (
        batched_compose_and_interpolate,
        batched_interp_3ch_packed,
    )
    from fastfuncstuff.processing.numba_kernels import compose_and_sample_cpu

    g = torch.Generator().manual_seed(0)
    nz, ny, nx = vol
    src = torch.randn(vol, generator=g)
    warp = torch.randn(3, *vol, generator=g) * 2
    grad = torch.randn(3, *vol, generator=g)
    kk, jj, ii = (
        t.reshape(-1).float()
        for t in torch.meshgrid(*(torch.arange(n) for n in patch), indexing="ij")
    )
    origin = [
        torch.randint(0, n - p + 1, (n_patch,), generator=g).float()
        for n, p in zip(vol, patch, strict=True)
    ]
    kb, jb, ib = origin
    bi, bj, bk = ib[:, None] + ii, jb[:, None] + jj, kb[:, None] + kk
    # Displacements of a few voxels, so both clamps (the warp's and the source's) bind.
    h = [torch.randn(n_patch, ii.numel(), generator=g) * 3 for _ in range(3)]
    args = (src, *warp, *h, ii, jj, kk, ib, jb, kb, nx, ny, nz)
    kw = dict(global_warp_3ch=warp, base_i=bi, base_j=bj, base_k=bk)
    with torch.no_grad():
        new = batched_compose_and_interpolate(*args, **kw)
        fused = compose_and_sample_cpu(src, warp, *h, bi, bj, bk, grad)
        torch_route()
        old = batched_compose_and_interpolate(*args, **kw)
        pos = [
            (a + b).clamp(-0.499, n - 0.501)
            for a, b, n in zip(old[1:], (bi, bj, bk), (nx, ny, nz), strict=True)
        ]
        g_old = batched_interp_3ch_packed(grad, *pos).permute(1, 0, 2)
    for a, b, c in zip(new, fused[:4], old, strict=True):
        assert torch.equal(a, c)
        assert torch.equal(b, c)
    assert torch.equal(fused[4], g_old)


@pytest.mark.parametrize(
    "n_dir,n_patch,n_vox,n_basis,centred",
    [(3, 7, 300, 4, True), (2, 3, 500, 8, False), (3, 1, 9000, 12, True), (1, 40, 64, 4, True)],
)
def test_fused_gn_normal_equations_match_the_chunked_einsum(
    n_dir, n_patch, n_vox, n_basis, centred, torch_route
):
    """Float32 rounding against the torch route, and closer than it to float64.

    The single-patch case is split across threads (level 0's shape); zero-weight
    voxels are skipped rather than multiplied by zero.
    """
    import fastfuncstuff.processing.warp as W

    g0 = torch.Generator().manual_seed(0)
    # A strided (D, B, V) view of a (B, 3, V) buffer, as the compose kernel hands it over.
    g = torch.randn(n_patch, 3, n_vox, generator=g0).permute(1, 0, 2)[:n_dir]
    hw = torch.rand(n_dir, 1, 1, generator=g0) + 0.5
    bt = torch.randn(n_vox, n_basis, generator=g0)
    omega = torch.rand(n_patch, n_vox, generator=g0)
    omega[omega < 0.2] = 0
    scale = (torch.rand(n_patch, 1, generator=g0) + 0.5).expand(-1, n_vox)
    res = torch.randn(n_patch, n_vox, generator=g0)
    mean = torch.randn(n_patch, 1, n_dir * n_basis, generator=g0) if centred else None
    args = (g, hw, bt, scale, omega, res, mean)
    h_new, g_new = W._gn_accumulate(*args, 4096)
    torch_route()
    h_old, g_old = W._gn_accumulate(*args, 4096)
    h_ref, g_ref = W._gn_accumulate(*(a.double() if a is not None else None for a in args), 4096)

    def rel(a, ref):
        return ((a.double() - ref).abs().max() / ref.abs().max()).item()

    # float32 rounding: under 2e-7 on arm64, 2.2e-7 on x86 (different summation order).
    for new, old, ref in ((h_new, h_old, h_ref), (g_new, g_old, g_ref)):
        assert rel(new, ref) < 5e-7
        assert rel(new, ref) <= rel(old, ref)
    assert torch.equal(h_new, h_new.transpose(-1, -2))


@pytest.mark.parametrize(
    "shape,scale", [((9, 11, 10), 0.8), ((4, 5, 6, 7), 1.0), ((2, 1, 4), 0.7), ((40, 3, 3, 3), 0.3)]
)
def test_fused_hexahedron_penalty_matches_the_torch_route(shape, scale, torch_route):
    """Float32 rounding, not bits: ATen's pow is Sleef's and Numba's is libm's, so a
    voxel near the deadband or with a huge energy can differ in its last bits."""
    from fastfuncstuff.processing.penalty import (
        compute_hexahedron_energy,
        hexahedron_penalty_energy,
        penalty_energy,
    )

    g = torch.Generator().manual_seed(0)
    f = [torch.randn(shape, generator=g) * scale for _ in range(3)]
    new = hexahedron_penalty_energy(*f)
    torch_route()
    old = hexahedron_penalty_energy(*f)
    ref = penalty_energy(*compute_hexahedron_energy(*(t.double() for t in f)))
    assert new.shape == old.shape == shape
    torch.testing.assert_close(new, old, rtol=2e-5, atol=1e-6)
    # What the optimiser consumes is the sum; both routes hold it to float32 rounding.
    total = ref.sum()
    assert abs(new.double().sum() - total) / total < 2e-6
    assert abs(old.double().sum() - total) / total < 2e-6


@pytest.mark.parametrize(
    "shape,scale", [((1, 8, 9, 7), 0.7), ((3, 6, 7, 5), 1.1), ((8, 5, 5, 6), 0.9)]
)
def test_fused_hexahedron_penalty_value_and_grad_matches_autograd(shape, scale, torch_route):
    """Covers both the slice-parallel (B < 4) and batch-parallel (B >= 4) kernels."""
    from fastfuncstuff.processing.penalty import penalty_value_and_grad

    g = torch.Generator().manual_seed(7)
    f = [torch.randn(shape, generator=g) * scale for _ in range(3)]
    ext = torch.rand(shape[0], generator=g) * 2.0

    val_new, grad_new = penalty_value_and_grad(*f, pen_fac=0.05, external_sums=ext)
    torch_route()
    val_old, grad_old = penalty_value_and_grad(*f, pen_fac=0.05, external_sums=ext)

    torch.testing.assert_close(val_new, val_old, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(grad_new, grad_old, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "base_clip,source_clip",
    [
        (None, None),
        ((50.0, 350.0), (20.0, 300.0)),
        ((50.0, 350.0, 100.0, 300.0), (20.0, 300.0, 60.0, 260.0)),
    ],
)
@pytest.mark.parametrize("fisher_z", [False, True])
def test_fused_incor_eval_and_grad_match_torch(base_clip, source_clip, fisher_z, torch_route):
    from fastfuncstuff.processing.cost import BatchedIncrementalCorrelation

    g = torch.Generator().manual_seed(11)
    nz, ny, nx = 16, 16, 16
    base = torch.randn(nz, ny, nx, generator=g) * 100 + 200
    src = base * 0.8 + torch.randn(nz, ny, nx, generator=g) * 40
    wt = (torch.rand(nz, ny, nx, generator=g) > 0.25).float() * torch.rand(nz, ny, nx, generator=g)
    slices = [(1, 8, 1, 8, 1, 8), (6, 13, 1, 8, 1, 8), (1, 8, 6, 13, 6, 13)]
    bp = torch.stack(
        [
            base[k0 : k1 + 1, j0 : j1 + 1, i0 : i1 + 1].reshape(-1)
            for i0, i1, j0, j1, k0, k1 in slices
        ]
    )
    sp = torch.stack(
        [
            src[k0 : k1 + 1, j0 : j1 + 1, i0 : i1 + 1].reshape(-1)
            for i0, i1, j0, j1, k0, k1 in slices
        ]
    )
    wp = torch.stack(
        [wt[k0 : k1 + 1, j0 : j1 + 1, i0 : i1 + 1].reshape(-1) for i0, i1, j0, j1, k0, k1 in slices]
    )

    ic = BatchedIncrementalCorrelation("pearclp", base_clip, source_clip, fisher_z=fisher_z)
    ic.precompute_fixed_parts(base, src, wt, slices, bp, wp, source_patches=sp)
    with torch.no_grad():
        r_eval = ic.evaluate(bp, sp, wp)
    r_vg, g_vg = ic.evaluate_with_grad(bp, sp, wp)

    torch_route()
    with torch.no_grad():
        r_old = ic.evaluate(bp, sp, wp)
    r_old_vg, g_old_vg = ic.evaluate_with_grad(bp, sp, wp)

    torch.testing.assert_close(r_eval, r_old, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(r_vg, r_old_vg, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(g_vg, g_old_vg, rtol=1e-5, atol=1e-6)


def test_scatter_last_wins_cpu_is_bit_identical_to_dedup():
    from fastfuncstuff.processing.numba_kernels import scatter_last_wins_cpu
    from fastfuncstuff.processing.warp import _dedup_last_wins

    g = torch.Generator().manual_seed(13)
    shape = (6, 7, 8)
    n_vox = 6 * 7 * 8
    idx = torch.randint(0, n_vox, (5, 90), generator=g)
    vals = tuple(torch.randn(5, 90, generator=g) for _ in range(4))

    ref_fields = tuple(torch.zeros(shape) for _ in range(4))
    dst, src = _dedup_last_wins(idx.reshape(-1))
    for f, v in zip(ref_fields, vals, strict=True):
        f.reshape(-1)[dst] = v.reshape(-1)[src]

    new_fields = tuple(torch.zeros(shape) for _ in range(4))
    scatter_last_wins_cpu(new_fields, idx, vals)
    for a, b in zip(new_fields, ref_fields, strict=True):
        assert torch.equal(a, b)
