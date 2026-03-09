"""Warp distortion penalty functions.

Implements the energy-based penalty from AFNI's IW3D_load_energy(), which
penalizes excessive bulk distortion (Jacobian deviation) and shear/vorticity.
The penalty encourages the warp to be a smooth diffeomorphism.

The penalty has two components:
  - je: bulk volume distortion energy (Jacobian determinant deviation from 1)
  - se: shear and vorticity energy

Total penalty = pen_fac * (sum_of_energies)^0.25

Both single-volume and batched versions are provided. The batched version
processes B patches in parallel on GPU without Python loops.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _central_diff_batched(vol: Tensor, dim: int) -> Tensor:
    """Central difference derivative along a dimension.

    Works for both single volumes (..., nz, ny, nx) and batched (B, nz, ny, nx).
    Uses forward/backward differences at boundaries.
    The `dim` argument refers to spatial dims: 0=z(-3), 1=y(-2), 2=x(-1).
    """
    # Map spatial dim to tensor dim (last 3 dims are z, y, x)
    tdim = dim - 3  # -3, -2, or -1

    n = vol.shape[tdim]
    if n < 2:
        return torch.zeros_like(vol)

    result = torch.zeros_like(vol)

    # Interior: central difference
    result.narrow(tdim, 1, n - 2).copy_(
        0.5 * (vol.narrow(tdim, 2, n - 2) - vol.narrow(tdim, 0, n - 2))
    )

    # Boundary: forward difference at start
    result.narrow(tdim, 0, 1).copy_(
        vol.narrow(tdim, 1, 1) - vol.narrow(tdim, 0, 1)
    )

    # Boundary: backward difference at end
    result.narrow(tdim, n - 1, 1).copy_(
        vol.narrow(tdim, n - 1, 1) - vol.narrow(tdim, n - 2, 1)
    )

    return result


def compute_jacobian_energy(
    xd: Tensor, yd: Tensor, zd: Tensor
) -> tuple[Tensor, Tensor]:
    """Compute Jacobian-based energy fields for a displacement warp.

    Works for both single volumes (nz, ny, nx) and batched (B, nz, ny, nx).

    Args:
        xd, yd, zd: (..., nz, ny, nx) displacement fields.

    Returns:
        (je, se): Bulk distortion and shear/vorticity energy, same shape.
    """
    dxd_di = _central_diff_batched(xd, dim=2)
    dxd_dj = _central_diff_batched(xd, dim=1)
    dxd_dk = _central_diff_batched(xd, dim=0)

    dyd_di = _central_diff_batched(yd, dim=2)
    dyd_dj = _central_diff_batched(yd, dim=1)
    dyd_dk = _central_diff_batched(yd, dim=0)

    dzd_di = _central_diff_batched(zd, dim=2)
    dzd_dj = _central_diff_batched(zd, dim=1)
    dzd_dk = _central_diff_batched(zd, dim=0)

    a11 = 1.0 + dxd_di
    a12 = dxd_dj
    a13 = dxd_dk
    a21 = dyd_di
    a22 = 1.0 + dyd_dj
    a23 = dyd_dk
    a31 = dzd_di
    a32 = dzd_dj
    a33 = 1.0 + dzd_dk

    det = (
        a11 * (a22 * a33 - a23 * a32)
        - a12 * (a21 * a33 - a23 * a31)
        + a13 * (a21 * a32 - a22 * a31)
    )

    je = (det - 1.0) ** 2

    e12 = 0.5 * (a12 + a21)
    e13 = 0.5 * (a13 + a31)
    e23 = 0.5 * (a23 + a32)
    w12 = 0.5 * (a12 - a21)
    w13 = 0.5 * (a13 - a31)
    w23 = 0.5 * (a23 - a32)
    e11 = a11 - 1.0
    e22 = a22 - 1.0
    e33 = a33 - 1.0

    se = (
        e12 * e12 + e13 * e13 + e23 * e23
        + w12 * w12 + w13 * w13 + w23 * w23
        + 0.5 * (e11 * e11 + e22 * e22 + e33 * e33)
    )

    return je, se


def compute_penalty(
    xd: Tensor, yd: Tensor, zd: Tensor,
    pen_fac: float = 0.033333,
    external_sum: float = 0.0,
) -> float:
    """Compute total warp distortion penalty (single volume, serial path)."""
    je, se = compute_jacobian_energy(xd, yd, zd)
    hsum = external_sum + float((je + se).sum().item())
    if hsum > 0:
        return pen_fac * (hsum ** 0.25)
    return 0.0


def compute_penalty_batched(
    xd: Tensor, yd: Tensor, zd: Tensor,
    pen_fac: float,
    external_sums: Tensor,
) -> Tensor:
    """Compute penalty for B patches in parallel. Returns (B,) tensor, differentiable.

    No Python loops - fully vectorized on GPU.

    Args:
        xd, yd, zd: (B, nz, ny, nx) composed displacement fields per patch.
        pen_fac: Penalty scaling factor.
        external_sums: (B,) pre-computed external penalty per patch.

    Returns:
        (B,) penalty values, differentiable.
    """
    # Batched Jacobian energy: (B, nz, ny, nx)
    je, se = compute_jacobian_energy(xd, yd, zd)

    # Sum over spatial dims, keep batch: (B,)
    patch_sums = (je + se).sum(dim=(-3, -2, -1))
    hsum = (external_sums + patch_sums).clamp(min=0)

    return pen_fac * hsum.pow(0.25)
