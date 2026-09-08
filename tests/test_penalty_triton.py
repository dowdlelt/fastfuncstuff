"""The fused deformation penalty must match the tensor implementation exactly.

It replaces nine central-difference fields and some forty elementwise
expressions with one register-resident pass, so only a comparison against the
code it replaces can show it kept AFNI's semantics -- the one-sided differences
at the patch faces and the HPEN_CUT deadband especially.
"""

import pytest
import torch

from fastfuncstuff.processing.penalty import (
    HPEN_CUT,
    compute_jacobian_energy,
    compute_penalty_batched,
    penalty_energy,
)

triton_mod = pytest.importorskip("fastfuncstuff.processing.penalty_triton")
penalty_sums_triton = triton_mod.penalty_sums_triton


def _reference_sums(xd, yd, zd):
    je, se = compute_jacobian_energy(xd, yd, zd)
    return penalty_energy(je, se).sum(dim=(-3, -2, -1))


@pytest.mark.gpu
@pytest.mark.parametrize(
    "b,nz,ny,nx,amp",
    [
        (3, 9, 9, 9, 0.1),  # everything under the deadband: must be exactly zero
        (3, 9, 9, 9, 3.0),  # well over it
        (5, 7, 11, 13, 1.5),  # anisotropic patch
        (2, 1, 8, 8, 2.0),  # a degenerate axis: the diff along it is zero
        (4, 5, 1, 6, 2.0),  # and along another
        (7, 2, 2, 2, 4.0),  # every voxel is a face voxel (one-sided everywhere)
        (64, 25, 25, 25, 2.0),  # a fine level
        (1, 64, 80, 64, 1.2),  # one big patch, as at level 0
    ],
)
def test_fused_penalty_matches_reference(b, nz, ny, nx, amp):
    if not torch.cuda.is_available():
        pytest.skip("the fused penalty is CUDA-only")
    dev = torch.device("cuda")
    torch.manual_seed(5)
    xd = torch.randn(b, nz, ny, nx, device=dev) * amp
    yd = torch.randn(b, nz, ny, nx, device=dev) * amp
    zd = torch.randn(b, nz, ny, nx, device=dev) * amp

    expected = _reference_sums(xd, yd, zd)
    actual = penalty_sums_triton(xd, yd, zd, HPEN_CUT)

    assert actual.shape == (b,)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.gpu
def test_autograd_path_keeps_the_tensor_penalty():
    """A field that needs gradients must not take the non-differentiable kernel."""
    if not torch.cuda.is_available():
        pytest.skip("the fused penalty is CUDA-only")
    dev = torch.device("cuda")
    torch.manual_seed(11)
    xd = (torch.randn(2, 6, 6, 6, device=dev) * 2.0).requires_grad_(True)
    yd = (torch.randn(2, 6, 6, 6, device=dev) * 2.0).requires_grad_(True)
    zd = (torch.randn(2, 6, 6, 6, device=dev) * 2.0).requires_grad_(True)
    ext = torch.zeros(2, device=dev)

    out = compute_penalty_batched(xd, yd, zd, 0.033333, ext)
    out.sum().backward()

    assert xd.grad is not None and torch.isfinite(xd.grad).all()
    assert float(xd.grad.abs().sum()) > 0.0
