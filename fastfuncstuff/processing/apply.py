"""Apply precomputed warp fields to images.

Equivalent to 3dNwarpApply - takes a displacement warp field and applies it
to an image using trilinear or higher-quality interpolation.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .interp import trilinear_interpolate, warp_image_linear, warp_image_wsinc5


def apply_warp(
    source: Tensor,
    warp_xd: Tensor,
    warp_yd: Tensor,
    warp_zd: Tensor,
    interp: str = "linear",
    mask: Tensor | None = None,
) -> Tensor:
    """Apply a displacement warp to a source image.

    Args:
        source: (nz, ny, nx) source image.
        warp_xd, warp_yd, warp_zd: (nz, ny, nx) displacement fields.
            The voxel at (i,j,k) in the output maps to (i+xd, j+yd, k+zd) in source.
        interp: Interpolation method - "linear" or "wsinc5".
        mask: Optional (nz, ny, nx) mask. Output is zeroed where mask == 0.

    Returns:
        (nz, ny, nx) warped image.
    """
    if interp == "wsinc5":
        result = warp_image_wsinc5(source, warp_xd, warp_yd, warp_zd)
    else:
        result = warp_image_linear(source, warp_xd, warp_yd, warp_zd)

    if mask is not None:
        result = result * mask.float()

    return result


def compose_warps(
    warp_a: tuple[Tensor, Tensor, Tensor],
    warp_b: tuple[Tensor, Tensor, Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    """Compose two displacement warps: C = B(A(x)).

    Given warp A mapping x -> x + a(x) and warp B mapping x -> x + b(x),
    the composed warp C maps x -> x + a(x) + b(x + a(x)).

    Args:
        warp_a: (xd_a, yd_a, zd_a) first warp to apply.
        warp_b: (xd_b, yd_b, zd_b) second warp to apply.

    Returns:
        (xd_c, yd_c, zd_c) composed displacement fields.
    """
    xd_a, yd_a, zd_a = warp_a
    xd_b, yd_b, zd_b = warp_b
    nz, ny, nx = xd_a.shape
    device = xd_a.device

    # Coordinates after first warp
    kk, jj, ii = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32, device=device),
        torch.arange(ny, dtype=torch.float32, device=device),
        torch.arange(nx, dtype=torch.float32, device=device),
        indexing="ij",
    )

    x_after_a = (ii + xd_a).clamp(0, nx - 1)
    y_after_a = (jj + yd_a).clamp(0, ny - 1)
    z_after_a = (kk + zd_a).clamp(0, nz - 1)

    # Interpolate warp B at positions after warp A
    bx_at_a = trilinear_interpolate(
        xd_b, x_after_a.reshape(-1), y_after_a.reshape(-1), z_after_a.reshape(-1)
    ).reshape(nz, ny, nx)
    by_at_a = trilinear_interpolate(
        yd_b, x_after_a.reshape(-1), y_after_a.reshape(-1), z_after_a.reshape(-1)
    ).reshape(nz, ny, nx)
    bz_at_a = trilinear_interpolate(
        zd_b, x_after_a.reshape(-1), y_after_a.reshape(-1), z_after_a.reshape(-1)
    ).reshape(nz, ny, nx)

    # C(x) = a(x) + b(x + a(x))
    xd_c = xd_a + bx_at_a
    yd_c = yd_a + by_at_a
    zd_c = zd_a + bz_at_a

    return xd_c, yd_c, zd_c


def invert_warp(
    warp_xd: Tensor,
    warp_yd: Tensor,
    warp_zd: Tensor,
    n_iter: int = 10,
) -> tuple[Tensor, Tensor, Tensor]:
    """Approximate warp inversion using iterative method.

    Uses the fixed-point iteration: inv_{n+1}(x) = -warp(x + inv_n(x)).
    This converges for smooth, small-displacement warps.

    Args:
        warp_xd, warp_yd, warp_zd: (nz, ny, nx) forward displacement fields.
        n_iter: Number of iterations.

    Returns:
        (inv_xd, inv_yd, inv_zd) approximate inverse displacement fields.
    """
    nz, ny, nx = warp_xd.shape
    device = warp_xd.device

    inv_xd = -warp_xd.clone()
    inv_yd = -warp_yd.clone()
    inv_zd = -warp_zd.clone()

    kk, jj, ii = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32, device=device),
        torch.arange(ny, dtype=torch.float32, device=device),
        torch.arange(nx, dtype=torch.float32, device=device),
        indexing="ij",
    )

    for _ in range(n_iter):
        # Evaluate forward warp at x + inv(x)
        xq = (ii + inv_xd).clamp(0, nx - 1)
        yq = (jj + inv_yd).clamp(0, ny - 1)
        zq = (kk + inv_zd).clamp(0, nz - 1)

        fx = trilinear_interpolate(warp_xd, xq.reshape(-1), yq.reshape(-1), zq.reshape(-1)).reshape(
            nz, ny, nx
        )
        fy = trilinear_interpolate(warp_yd, xq.reshape(-1), yq.reshape(-1), zq.reshape(-1)).reshape(
            nz, ny, nx
        )
        fz = trilinear_interpolate(warp_zd, xq.reshape(-1), yq.reshape(-1), zq.reshape(-1)).reshape(
            nz, ny, nx
        )

        inv_xd = -fx
        inv_yd = -fy
        inv_zd = -fz

    return inv_xd, inv_yd, inv_zd
