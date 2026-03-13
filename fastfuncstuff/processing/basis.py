"""Hermite polynomial basis functions for patch warping.

Implements C1 cubic and C2 quintic Hermite basis functions over [-1, 1],
matching AFNI's 3dQwarp basis function definitions. These are used to
parameterize local displacement fields within patches.

The key idea: each patch warp is a weighted sum of tensor-product basis
functions. The weights (parameters) are optimized to minimize a cost function.

Basis function counts per displacement direction:
  - cubic:        2 funcs/dim → 2x2x2 = 8 products (full), 4 (lite)
  - quintic:      3 funcs/dim → 3x3x3 = 27 products (full), 10 (lite)
  - cubic_lite:   uses only products where sum of indices ≤ 1 → 4 per dim
  - quintic_lite: uses only products where sum of indices ≤ 2 → 10 per dim

Total parameters = num_products * 3 (for x, y, z displacements).
"""

from __future__ import annotations

import torch
from torch import Tensor


class HermiteCubic:
    """C1 Hermite cubic basis functions on [-1, 1].

    Two functions:
      b0(x) = (1-|x|)^2 * (1 + 2|x|)    -- peaked at 0, value=1
      b1(x) = (1-|x|)^2 * x * 6.75       -- zero at 0, derivative=6.75
    Both vanish at x = ±1 along with first derivatives.
    """

    @staticmethod
    def eval_1d(x: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate the two cubic basis functions at points x in [-1, 1]."""
        aa = x.abs()
        mask = aa < 1.0
        bb = (1.0 - aa).clamp(min=0.0)
        bb2 = bb * bb
        b0 = bb2 * (1.0 + 2.0 * aa) * mask
        b1 = bb2 * x * 6.75 * mask
        return b0, b1


class HermiteQuintic:
    """C2 Hermite quintic basis functions on [-1, 1].

    Three functions:
      b0(x) = (1-|x|)^3 * (6|x|^2 + 3|x| + 1)   -- f(0)=1
      b1(x) = (1-|x|)^3 * x * (3|x|+1) * 5.0625  -- f'(0) ≠ 0
      b2(x) = |x|^2 * (1-|x|)^3 * 28.935          -- f''(0) ≠ 0
    All vanish at x = ±1 along with first two derivatives.
    """

    @staticmethod
    def eval_1d(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate the three quintic basis functions at points x in [-1, 1]."""
        aa = x.abs()
        mask = aa < 1.0
        bb = (1.0 - aa).clamp(min=0.0)
        bb3 = bb * bb * bb
        aq = aa * aa
        b0 = bb3 * ((6.0 * aq + 3.0) * aa + 1.0) * mask
        b1 = bb3 * x * (3.0 * aa + 1.0) * 5.0625 * mask
        b2 = aq * bb3 * 28.935 * mask
        return b0, b1, b2


def compute_basis_coords(n: int, device: torch.device) -> Tensor:
    """Compute the [-1, 1] mapped coordinates for a patch of size n.

    Maps grid indices 0..n-1 to the range [-1, 1] using the AFNI convention:
      ILEFT = -0.5, IRGHT = n - 0.5
      x = ca + cb * i  where cb = 2/(IRGHT - ILEFT), ca = -1 - cb*ILEFT

    Returns:
        Tensor of shape (n,) with values in approximately [-1, 1].
    """
    ileft = -0.5
    irght = n - 0.5
    cb = 2.0 / (irght - ileft)
    ca = -1.0 - cb * ileft
    indices = torch.arange(n, dtype=torch.float32, device=device)
    return ca + cb * indices


def build_3d_basis_cubic(
    nx: int, ny: int, nz: int, device: torch.device, lite: bool = True
) -> Tensor:
    """Build 3D cubic basis function arrays for a patch.

    Args:
        nx, ny, nz: Patch dimensions.
        device: Torch device.
        lite: If True, use only 4 products (lite mode = cubic12).
              If False, use all 8 products (full cubic24).

    Returns:
        Tensor of shape (n_basis, nz*ny*nx) where n_basis is 4 (lite) or 8 (full).
        Each row is one 3D basis function evaluated at every voxel in the patch.
    """
    cx = compute_basis_coords(nx, device)
    cy = compute_basis_coords(ny, device)
    cz = compute_basis_coords(nz, device)

    b0x, b1x = HermiteCubic.eval_1d(cx)
    b0y, b1y = HermiteCubic.eval_1d(cy)
    b0z, b1z = HermiteCubic.eval_1d(cz)

    # Build outer products: index order is [z, y, x] flattened
    # Shape manipulations for broadcasting: z(nz,1,1) * y(1,ny,1) * x(1,1,nx)
    b0z_3d = b0z[:, None, None]
    b1z_3d = b1z[:, None, None]
    b0y_3d = b0y[None, :, None]
    b1y_3d = b1y[None, :, None]
    b0x_3d = b0x[None, None, :]
    b1x_3d = b1x[None, None, :]

    if lite:
        # Lite mode: only products where (iz + iy + ix) <= 1
        # That gives us: (0,0,0), (1,0,0), (0,1,0), (0,0,1) = 4 basis funcs
        basis = torch.stack([
            (b0z_3d * b0y_3d * b0x_3d).reshape(-1),  # 000
            (b1z_3d * b0y_3d * b0x_3d).reshape(-1),  # 100 (z)
            (b0z_3d * b1y_3d * b0x_3d).reshape(-1),  # 010 (y)
            (b0z_3d * b0y_3d * b1x_3d).reshape(-1),  # 001 (x)
        ])
    else:
        # Full mode: all 2x2x2 = 8 tensor products
        basis = torch.stack([
            (b0z_3d * b0y_3d * b0x_3d).reshape(-1),  # 000
            (b1z_3d * b0y_3d * b0x_3d).reshape(-1),  # 100
            (b0z_3d * b1y_3d * b0x_3d).reshape(-1),  # 010
            (b1z_3d * b1y_3d * b0x_3d).reshape(-1),  # 110
            (b0z_3d * b0y_3d * b1x_3d).reshape(-1),  # 001
            (b1z_3d * b0y_3d * b1x_3d).reshape(-1),  # 101
            (b0z_3d * b1y_3d * b1x_3d).reshape(-1),  # 011
            (b1z_3d * b1y_3d * b1x_3d).reshape(-1),  # 111
        ])

    return basis


def build_3d_basis_quintic(
    nx: int, ny: int, nz: int, device: torch.device, lite: bool = True
) -> Tensor:
    """Build 3D quintic basis function arrays for a patch.

    Args:
        nx, ny, nz: Patch dimensions.
        device: Torch device.
        lite: If True, use only 10 products where sum of indices ≤ 2 (quintic30).
              If False, use all 27 products (quintic81).

    Returns:
        Tensor of shape (n_basis, nz*ny*nx) where n_basis is 10 (lite) or 27 (full).
    """
    cx = compute_basis_coords(nx, device)
    cy = compute_basis_coords(ny, device)
    cz = compute_basis_coords(nz, device)

    b0x, b1x, b2x = HermiteQuintic.eval_1d(cx)
    b0y, b1y, b2y = HermiteQuintic.eval_1d(cy)
    b0z, b1z, b2z = HermiteQuintic.eval_1d(cz)

    bx = [b0x, b1x, b2x]
    by = [b0y, b1y, b2y]
    bz = [b0z, b1z, b2z]

    bases = []
    for ix in range(3):
        for iy in range(3):
            for iz in range(3):
                if lite and (ix + iy + iz) > 2:
                    continue
                prod = (
                    bz[iz][:, None, None]
                    * by[iy][None, :, None]
                    * bx[ix][None, None, :]
                )
                bases.append(prod.reshape(-1))

    return torch.stack(bases)


def compute_half_widths_cubic(
    nx: int, ny: int, nz: int
) -> tuple[float, float, float]:
    """Compute the half-width scaling factors (dxci, dyci, dzci) for cubic basis.

    These scale the displacement parameters to physical displacement magnitudes.
    dxci = 1/cb where cb = 2/(IRGHT - ILEFT).
    """
    def _half_width(n: int) -> float:
        ileft = -0.5
        irght = n - 0.5
        cb = 2.0 / (irght - ileft)
        return 1.0 / cb if cb != 0 else 0.0

    return _half_width(nx), _half_width(ny), _half_width(nz)


def compute_half_widths_quintic(
    nx: int, ny: int, nz: int
) -> tuple[float, float, float]:
    """Compute half-width scaling factors for quintic basis (same formula)."""
    return compute_half_widths_cubic(nx, ny, nz)


def evaluate_patch_warp(
    basis: Tensor,
    params: Tensor,
    half_widths: tuple[float, float, float],
    do_xyz: tuple[bool, bool, bool] = (True, True, True),
) -> tuple[Tensor, Tensor, Tensor]:
    """Evaluate patch displacement field from basis functions and parameters.

    The displacement at each voxel is:
      disp_d(v) = half_width_d * sum_p { params_d[p] * basis[p, v] }
    for each direction d in {x, y, z}.

    Args:
        basis: (n_basis, n_voxels) - 3D basis functions.
        params: (3 * n_basis,) - parameters [x_params | y_params | z_params].
        half_widths: (dx, dy, dz) scaling factors.
        do_xyz: Which displacement directions are active.

    Returns:
        (xd, yd, zd) displacement tensors, each of shape (n_voxels,).
    """
    n_basis = basis.shape[0]
    n_voxels = basis.shape[1]

    x_params = params[:n_basis]
    y_params = params[n_basis : 2 * n_basis]
    z_params = params[2 * n_basis : 3 * n_basis]

    dxci, dyci, dzci = half_widths

    # Matrix-vector product: basis.T @ params gives displacement at each voxel
    # basis is (n_basis, n_voxels), params is (n_basis,)
    # Result is (n_voxels,)
    if do_xyz[0]:
        xd = dxci * (basis.T @ x_params)
    else:
        xd = torch.zeros(n_voxels, device=basis.device)

    if do_xyz[1]:
        yd = dyci * (basis.T @ y_params)
    else:
        yd = torch.zeros(n_voxels, device=basis.device)

    if do_xyz[2]:
        zd = dzci * (basis.T @ z_params)
    else:
        zd = torch.zeros(n_voxels, device=basis.device)

    return xd, yd, zd


def evaluate_patch_warp_batched(
    basis: Tensor,
    params: Tensor,
    half_widths: tuple[float, float, float],
    do_xyz: tuple[bool, bool, bool] = (True, True, True),
) -> tuple[Tensor, Tensor, Tensor]:
    """Evaluate patch displacement for a BATCH of patches simultaneously.

    This is the batched version of evaluate_patch_warp - the key GPU speedup.
    All B patches share the same basis functions (same patch size at a level).

    Args:
        basis: (n_basis, V) where V = nxh*nyh*nzh. Shared across all patches.
        params: (B, 3*n_basis) parameter vectors for B patches.
        half_widths: Shared scaling factors.
        do_xyz: Which axes are active.

    Returns:
        (xd, yd, zd) each of shape (B, V).
    """
    n_basis = basis.shape[0]
    V = basis.shape[1]
    B = params.shape[0]

    dxci, dyci, dzci = half_widths

    x_params = params[:, :n_basis]              # (B, n_basis)
    y_params = params[:, n_basis:2*n_basis]     # (B, n_basis)
    z_params = params[:, 2*n_basis:]            # (B, n_basis)

    # Batched matmul: (B, n_basis) @ (n_basis, V) -> (B, V)
    if do_xyz[0]:
        xd = dxci * (x_params @ basis)
    else:
        xd = torch.zeros(B, V, device=basis.device)

    if do_xyz[1]:
        yd = dyci * (y_params @ basis)
    else:
        yd = torch.zeros(B, V, device=basis.device)

    if do_xyz[2]:
        zd = dzci * (z_params @ basis)
    else:
        zd = torch.zeros(B, V, device=basis.device)

    return xd, yd, zd
