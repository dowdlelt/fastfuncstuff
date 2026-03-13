"""Affine transform math for rigid/affine alignment.

12-parameter representation (matching AFNI order):
  [0-2]  dx, dy, dz     — translations (mm)
  [3-5]  rz, rx, ry     — rotations (degrees) — Z, X, Y order per AFNI
  [6-8]  sx, sy, sz     — scales (1.0 = identity)
  [9-11] shyx, shzx, shzy — shear (0.0 = identity)

Matrix decomposition: M = T · S · D · U
  T = translation, S = shear, D = scale, U = rotation

Coordinate conventions:
  - Internal optimization works in voxel index space (z, y, x order)
  - For save/load, convert to/from DICOM mm using base and source affines:
    M_dicom = source_ijk2xyz @ M_ijk @ inv(base_ijk2xyz)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .interp import _grid_sample_3d, _separable_resample_3d, wsinc5_resample_3d

# ---------------------------------------------------------------------------
# Parameter ↔ Matrix conversion
# ---------------------------------------------------------------------------


def params_to_matrix(params: Tensor) -> Tensor:
    """Convert 12 affine parameters to a 4x4 transformation matrix.

    Args:
        params: (12,) tensor with [dx, dy, dz, rz, rx, ry, sx, sy, sz,
                shyx, shzx, shzy].

    Returns:
        (4, 4) homogeneous transformation matrix.
    """
    dx, dy, dz = params[0], params[1], params[2]
    rz, rx, ry = params[3], params[4], params[5]
    sx, sy, sz = params[6], params[7], params[8]
    shyx, shzx, shzy = params[9], params[10], params[11]

    # Rotation matrices (angles in degrees → radians)
    rz_rad = rz * (math.pi / 180.0)
    rx_rad = rx * (math.pi / 180.0)
    ry_rad = ry * (math.pi / 180.0)

    cz, sz_ = torch.cos(rz_rad), torch.sin(rz_rad)
    cx, sx_ = torch.cos(rx_rad), torch.sin(rx_rad)
    cy, sy_ = torch.cos(ry_rad), torch.sin(ry_rad)

    zero = torch.zeros(1, device=params.device, dtype=params.dtype).squeeze()
    one = torch.ones(1, device=params.device, dtype=params.dtype).squeeze()

    # Rz @ Rx @ Ry  (AFNI convention: Z, X, Y order)
    # Rz
    Rz = torch.stack(
        [
            torch.stack([cz, -sz_, zero]),
            torch.stack([sz_, cz, zero]),
            torch.stack([zero, zero, one]),
        ]
    )
    # Rx
    Rx = torch.stack(
        [
            torch.stack([one, zero, zero]),
            torch.stack([zero, cx, -sx_]),
            torch.stack([zero, sx_, cx]),
        ]
    )
    # Ry
    Ry = torch.stack(
        [
            torch.stack([cy, zero, sy_]),
            torch.stack([zero, one, zero]),
            torch.stack([-sy_, zero, cy]),
        ]
    )

    U = Rz @ Rx @ Ry  # (3, 3) rotation

    # Scale matrix D
    D = torch.diag(torch.stack([sx, sy, sz]))

    # Shear matrix S (upper triangular)
    S = torch.eye(3, device=params.device, dtype=params.dtype)
    S = S.clone()
    S[0, 1] = shyx
    S[0, 2] = shzx
    S[1, 2] = shzy

    # Combined 3x3: S @ D @ U
    M3 = S @ D @ U

    # Build 4x4
    M = torch.eye(4, device=params.device, dtype=params.dtype)
    M = M.clone()
    M[:3, :3] = M3
    M[0, 3] = dx
    M[1, 3] = dy
    M[2, 3] = dz

    return M


def params_to_matrix_batched(params: Tensor) -> Tensor:
    """Convert B sets of 12 parameters to B 4x4 matrices.

    Args:
        params: (B, 12) tensor.

    Returns:
        (B, 4, 4) transformation matrices.
    """
    B = params.shape[0]
    device = params.device
    dtype = params.dtype

    dx, dy, dz = params[:, 0], params[:, 1], params[:, 2]
    rz, rx, ry = params[:, 3], params[:, 4], params[:, 5]
    sx, sy, sz = params[:, 6], params[:, 7], params[:, 8]
    shyx, shzx, shzy = params[:, 9], params[:, 10], params[:, 11]

    deg2rad = math.pi / 180.0
    rz_r, rx_r, ry_r = rz * deg2rad, rx * deg2rad, ry * deg2rad

    cz, sz_ = torch.cos(rz_r), torch.sin(rz_r)
    cx, sx_ = torch.cos(rx_r), torch.sin(rx_r)
    cy, sy_ = torch.cos(ry_r), torch.sin(ry_r)

    _zero = torch.zeros(B, device=device, dtype=dtype)
    one = torch.ones(B, device=device, dtype=dtype)

    # Rz @ Rx @ Ry — computed directly as the product
    # R = Rz @ Rx @ Ry, each element computed analytically
    r00 = cz * cy + sz_ * sx_ * sy_
    r01 = -sz_ * cx
    r02 = cz * (-sy_) + sz_ * sx_ * cy
    r10 = sz_ * cy + (-cz) * sx_ * (-sy_)  # sz*cy + cz*sx*sy
    r11 = cz * cx
    r12 = (
        sz_ * (-sy_) + (-cz) * sx_ * cy
    )  # -sz*sy - cz*sx*cy ... wait let me redo this properly

    # Rz @ Rx @ Ry product:
    # Row 0: [cz*cy + sz*sx*sy,  -sz*cx,  -cz*sy + sz*sx*cy]
    # Row 1: [sz*cy - cz*sx*sy,   cz*cx,  -sz*sy - cz*sx*cy]  ... no
    # Let me compute it correctly:
    # Rz = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]
    # Rx = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]]
    # Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
    #
    # Rz @ Rx:
    # [[cz, -sz*cx, -sz*(-sx)], [sz, cz*cx, cz*(-sx)], [0, sx, cx]]
    # = [[cz, -sz*cx, sz*sx], [sz, cz*cx, -cz*sx], [0, sx, cx]]
    #
    # (Rz@Rx) @ Ry:
    # Row 0: [cz*cy + sz*sx*(-sy), -sz*cx, cz*sy + sz*sx*cy]
    # Wait: Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
    #
    # (Rz@Rx) @ Ry:
    # [0,0]: cz*cy + (-sz*cx)*0 + sz*sx*(-sy) = cz*cy - sz*sx*sy
    # Hmm, let me just be careful:
    # A = Rz@Rx = [[cz, -sz*cx, sz*sx],
    #              [sz,  cz*cx, -cz*sx],
    #              [0,   sx,     cx]]
    # A @ Ry:
    # [i,j] = sum_k A[i,k] * Ry[k,j]
    # Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
    #
    # [0,0] = cz*cy + 0 + sz*sx*(-sy) = cz*cy - sz*sx*sy
    # [0,1] = 0 + (-sz*cx)*1 + 0 = -sz*cx
    # [0,2] = cz*sy + 0 + sz*sx*cy
    # [1,0] = sz*cy + 0 + (-cz*sx)*(-sy) = sz*cy + cz*sx*sy
    # [1,1] = 0 + cz*cx + 0 = cz*cx
    # [1,2] = sz*sy + 0 + (-cz*sx)*cy = sz*sy - cz*sx*cy
    # [2,0] = 0 + sx*0 + cx*(-sy) = -cx*sy
    # [2,1] = 0 + sx*1 + 0 = sx
    # [2,2] = 0 + 0 + cx*cy

    r00 = cz * cy - sz_ * sx_ * sy_
    r01 = -sz_ * cx
    r02 = cz * sy_ + sz_ * sx_ * cy
    r10 = sz_ * cy + cz * sx_ * sy_
    r11 = cz * cx
    r12 = sz_ * sy_ - cz * sx_ * cy
    r20 = -cx * sy_
    r21 = sx_
    r22 = cx * cy

    # Apply scale: column j of R gets multiplied by scale[j]
    # Then shear: S @ (D @ U)
    # D @ U: column 0 *= sx, column 1 *= sy, column 2 *= sz
    du00, du01, du02 = r00 * sx, r01 * sy, r02 * sz
    du10, du11, du12 = r10 * sx, r11 * sy, r12 * sz
    du20, du21, du22 = r20 * sx, r21 * sy, r22 * sz

    # S @ (D@U): S = [[1, shyx, shzx], [0, 1, shzy], [0, 0, 1]]
    # row 0 += shyx * row1 + shzx * row2
    # row 1 += shzy * row2
    m00 = du00 + shyx * du10 + shzx * du20
    m01 = du01 + shyx * du11 + shzx * du21
    m02 = du02 + shyx * du12 + shzx * du22
    m10 = du10 + shzy * du20
    m11 = du11 + shzy * du21
    m12 = du12 + shzy * du22
    m20 = du20
    m21 = du21
    m22 = du22

    # Build (B, 4, 4)
    M = torch.zeros(B, 4, 4, device=device, dtype=dtype)
    M[:, 0, 0] = m00
    M[:, 0, 1] = m01
    M[:, 0, 2] = m02
    M[:, 0, 3] = dx
    M[:, 1, 0] = m10
    M[:, 1, 1] = m11
    M[:, 1, 2] = m12
    M[:, 1, 3] = dy
    M[:, 2, 0] = m20
    M[:, 2, 1] = m21
    M[:, 2, 2] = m22
    M[:, 2, 3] = dz
    M[:, 3, 3] = one

    return M


def matrix_to_params(mat: Tensor) -> Tensor:
    """Decompose a 4x4 affine matrix into 12 parameters.

    Uses polar decomposition to separate rotation from scale+shear.

    Args:
        mat: (4, 4) homogeneous transformation matrix.

    Returns:
        (12,) parameter tensor.
    """
    device = mat.device
    dtype = mat.dtype

    # Translation
    dx, dy, dz = mat[0, 3], mat[1, 3], mat[2, 3]

    # 3x3 part = S @ D @ R
    M3 = mat[:3, :3]

    # Polar decomposition: M3 = U @ P where U is rotation, P is symmetric
    # Use SVD: M3 = U @ S @ Vh, rotation = U @ Vh, scale+shear = Vh.T @ S @ Vh
    U_svd, S_svd, Vh = torch.linalg.svd(M3)

    # Handle reflections (det < 0)
    det = torch.det(U_svd @ Vh)
    if det < 0:
        U_svd[:, -1] *= -1
        S_svd[-1] *= -1

    R = U_svd @ Vh  # Rotation matrix
    P = Vh.T @ torch.diag(S_svd) @ Vh  # Symmetric positive (scale + shear)

    # Extract rotation angles from R (Rz @ Rx @ Ry convention)
    # R[2,1] = sin(rx)
    sx_val = R[2, 1].clamp(-1.0, 1.0)
    rx = torch.asin(sx_val) * (180.0 / math.pi)

    cx_val = torch.cos(rx * (math.pi / 180.0))
    if cx_val.abs() > 1e-6:
        # rz from R[1,1] = cz*cx, R[0,1] = -sz*cx
        rz = torch.atan2(-R[0, 1], R[1, 1]) * (180.0 / math.pi)
        # ry from R[2,2] = cx*cy, R[2,0] = -cx*sy
        ry = torch.atan2(-R[2, 0], R[2, 2]) * (180.0 / math.pi)
    else:
        # Gimbal lock
        rz = torch.atan2(R[1, 0], R[0, 0]) * (180.0 / math.pi)
        ry = torch.tensor(0.0, device=device, dtype=dtype)

    # Extract scale and shear from P (symmetric positive definite)
    # P = S @ D where S is upper-triangular shear, D is diagonal scale
    # Use Cholesky-like decomposition: P is symmetric, so do QR or direct extraction
    # For simplicity, P = [[p00,p01,p02],[p01,p11,p12],[p02,p12,p22]]
    # Scale: diagonal of P gives approximate scales
    sx_scale = P[0, 0]
    sy_scale = P[1, 1]
    sz_scale = P[2, 2]

    # Shear: off-diagonal / diagonal
    shyx = (
        P[0, 1] / sy_scale
        if sy_scale.abs() > 1e-10
        else torch.tensor(0.0, device=device, dtype=dtype)
    )
    shzx = (
        P[0, 2] / sz_scale
        if sz_scale.abs() > 1e-10
        else torch.tensor(0.0, device=device, dtype=dtype)
    )
    shzy = (
        P[1, 2] / sz_scale
        if sz_scale.abs() > 1e-10
        else torch.tensor(0.0, device=device, dtype=dtype)
    )

    return torch.stack(
        [dx, dy, dz, rz, rx, ry, sx_scale, sy_scale, sz_scale, shyx, shzx, shzy]
    )


def identity_params(
    device: torch.device = None, dtype: torch.dtype = torch.float32
) -> Tensor:
    """Return identity affine parameters (12,)."""
    p = torch.zeros(12, device=device, dtype=dtype)
    p[6] = 1.0  # sx
    p[7] = 1.0  # sy
    p[8] = 1.0  # sz
    return p


# ---------------------------------------------------------------------------
# Apply affine transforms via grid_sample
# ---------------------------------------------------------------------------


def apply_affine(
    source: Tensor,
    matrix: Tensor,
    output_shape: tuple[int, int, int] | None = None,
    zero_outside: bool = False,
) -> Tensor:
    """Resample source image using an affine transformation matrix.

    The matrix maps output (base) voxel indices to source voxel indices.

    Args:
        source: (nz, ny, nx) source image.
        matrix: (4, 4) affine matrix in voxel index space.
        output_shape: (nz, ny, nx) of output grid. Defaults to source shape.
        zero_outside: If True, zero voxels that map outside the source volume
            instead of using border padding. Use True for final output,
            False during optimization (border padding gives smooth gradients).

    Returns:
        Resampled image with output_shape.
    """
    if output_shape is None:
        output_shape = source.shape
    onz, ony, onx = output_shape
    device = source.device
    dtype = source.dtype

    # Build output grid in voxel indices
    kk, jj, ii = torch.meshgrid(
        torch.arange(onz, dtype=dtype, device=device),
        torch.arange(ony, dtype=dtype, device=device),
        torch.arange(onx, dtype=dtype, device=device),
        indexing="ij",
    )

    # Apply affine: source_coords = M @ output_coords
    # Stack to (N, 4) homogeneous coordinates
    coords = torch.stack(
        [
            ii.reshape(-1),
            jj.reshape(-1),
            kk.reshape(-1),
            torch.ones(onz * ony * onx, device=device, dtype=dtype),
        ],
        dim=0,
    )  # (4, N)
    src_coords = matrix @ coords  # (4, N)

    src_x = src_coords[0].reshape(onz, ony, onx)
    src_y = src_coords[1].reshape(onz, ony, onx)
    src_z = src_coords[2].reshape(onz, ony, onx)

    # Convert to normalized [-1, 1] for grid_sample
    snz, sny, snx = source.shape
    gx = 2.0 * src_x / (snx - 1) - 1.0 if snx > 1 else src_x * 0.0
    gy = 2.0 * src_y / (sny - 1) - 1.0 if sny > 1 else src_y * 0.0
    gz = 2.0 * src_z / (snz - 1) - 1.0 if snz > 1 else src_z * 0.0

    grid = torch.stack([gx, gy, gz], dim=-1)[None]  # (1, D, H, W, 3)
    vol = source[None, None]  # (1, 1, D, H, W)

    result = _grid_sample_3d(vol, grid)
    result = result[0, 0]

    if zero_outside:
        oob = (
            (src_x < -0.5)
            | (src_x > snx - 0.5)
            | (src_y < -0.5)
            | (src_y > sny - 0.5)
            | (src_z < -0.5)
            | (src_z > snz - 0.5)
        )
        result[oob] = 0.0

    return result


def apply_affine_wsinc5(
    source: Tensor,
    matrix: Tensor,
    output_shape: tuple[int, int, int] | None = None,
) -> Tensor:
    """Resample source image using an affine transform with wsinc5 interpolation.

    Same interface as apply_affine but uses Hanning-windowed sinc (11-tap)
    instead of trilinear, giving much sharper results for final output.
    Out-of-bounds voxels are zeroed (wsinc5 is only used for final output).

    Args:
        source: (nz, ny, nx) source image.
        matrix: (4, 4) affine matrix in voxel index space.
        output_shape: (nz, ny, nx) of output grid. Defaults to source shape.

    Returns:
        Resampled image with output_shape.
    """
    if output_shape is None:
        output_shape = source.shape
    onz, ony, onx = output_shape
    device = source.device
    dtype = source.dtype
    snz, sny, snx = source.shape

    # Build output grid in voxel indices
    kk, jj, ii = torch.meshgrid(
        torch.arange(onz, dtype=dtype, device=device),
        torch.arange(ony, dtype=dtype, device=device),
        torch.arange(onx, dtype=dtype, device=device),
        indexing="ij",
    )

    # Apply affine: source_coords = M @ output_coords
    coords = torch.stack(
        [
            ii.reshape(-1),
            jj.reshape(-1),
            kk.reshape(-1),
            torch.ones(onz * ony * onx, device=device, dtype=dtype),
        ],
        dim=0,
    )
    src_coords = matrix @ coords  # (4, N)

    src_x = src_coords[0].reshape(onz, ony, onx)
    src_y = src_coords[1].reshape(onz, ony, onx)
    src_z = src_coords[2].reshape(onz, ony, onx)

    result = wsinc5_resample_3d(source, src_x, src_y, src_z)

    # Zero out-of-bounds voxels (wsinc5 clamps internally, so undo that)
    oob = (
        (src_x < -0.5)
        | (src_x > snx - 0.5)
        | (src_y < -0.5)
        | (src_y > sny - 0.5)
        | (src_z < -0.5)
        | (src_z > snz - 0.5)
    )
    result[oob] = 0.0

    return result


def apply_affine_interp(
    source: Tensor,
    matrix: Tensor,
    interp: str = "heptic",
    output_shape: tuple[int, int, int] | None = None,
    zero_outside: bool = False,
) -> Tensor:
    """Resample source image using an affine transform with selectable interpolation.

    Supports: "linear", "cubic", "quintic", "heptic", "wsinc5".
    "linear" delegates to grid_sample (fast); others use separable kernels.

    Args:
        source: (nz, ny, nx) source image.
        matrix: (4, 4) affine matrix in voxel index space.
        interp: interpolation method name.
        output_shape: (nz, ny, nx) of output grid. Defaults to source shape.
        zero_outside: If True, zero voxels that map outside the source volume.

    Returns:
        Resampled image with output_shape.
    """
    if interp == "linear":
        return apply_affine(source, matrix, output_shape, zero_outside=zero_outside)

    if output_shape is None:
        output_shape = source.shape
    onz, ony, onx = output_shape
    device = source.device
    dtype = source.dtype
    snz, sny, snx = source.shape

    # Build output grid
    kk, jj, ii = torch.meshgrid(
        torch.arange(onz, dtype=dtype, device=device),
        torch.arange(ony, dtype=dtype, device=device),
        torch.arange(onx, dtype=dtype, device=device),
        indexing="ij",
    )
    coords = torch.stack(
        [
            ii.reshape(-1),
            jj.reshape(-1),
            kk.reshape(-1),
            torch.ones(onz * ony * onx, device=device, dtype=dtype),
        ],
        dim=0,
    )
    src_coords = matrix @ coords

    src_x = src_coords[0].reshape(onz, ony, onx)
    src_y = src_coords[1].reshape(onz, ony, onx)
    src_z = src_coords[2].reshape(onz, ony, onx)

    result = _separable_resample_3d(source, src_x, src_y, src_z, interp)

    if zero_outside:
        oob = (
            (src_x < -0.5)
            | (src_x > snx - 0.5)
            | (src_y < -0.5)
            | (src_y > sny - 0.5)
            | (src_z < -0.5)
            | (src_z > snz - 0.5)
        )
        result[oob] = 0.0

    return result


def _build_homo_coords(
    shape: tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Build a (4, N) homogeneous coordinate grid for a given output shape.

    This is the grid that gets reused across GN iterations — build once,
    then just do matrix @ coords each iteration.

    Args:
        shape: (nz, ny, nx) output volume shape.
        device: torch device.
        dtype: torch dtype.

    Returns:
        (4, N) tensor where N = nz*ny*nx.
    """
    onz, ony, onx = shape
    kk, jj, ii = torch.meshgrid(
        torch.arange(onz, dtype=dtype, device=device),
        torch.arange(ony, dtype=dtype, device=device),
        torch.arange(onx, dtype=dtype, device=device),
        indexing="ij",
    )
    N = onz * ony * onx
    return torch.stack(
        [
            ii.reshape(-1),
            jj.reshape(-1),
            kk.reshape(-1),
            torch.ones(N, device=device, dtype=dtype),
        ],
        dim=0,
    )  # (4, N)


def resample_affine_fast(
    source: Tensor,
    matrix: Tensor,
    coords: Tensor,
    interp: str,
    output_shape: tuple[int, int, int],
    zero_outside: bool = False,
) -> Tensor:
    """Resample source using a pre-built coordinate grid (skips grid construction).

    Same result as apply_affine_interp but avoids rebuilding the meshgrid
    and homogeneous coordinate tensor every call.

    Args:
        source: (nz, ny, nx) source image.
        matrix: (4, 4) affine matrix in voxel index space.
        coords: (4, N) pre-built homogeneous coordinate grid from _build_homo_coords.
        interp: interpolation method name.
        output_shape: (nz, ny, nx) of output grid.
        zero_outside: If True, zero voxels that map outside the source volume.

    Returns:
        Resampled image with output_shape.
    """
    if interp == "linear":
        # For linear, delegate to grid_sample path (needs normalized coords)
        onz, ony, onx = output_shape
        snz, sny, snx = source.shape
        src_coords = matrix @ coords  # (4, N)

        src_x = src_coords[0].reshape(onz, ony, onx)
        src_y = src_coords[1].reshape(onz, ony, onx)
        src_z = src_coords[2].reshape(onz, ony, onx)

        gx = 2.0 * src_x / (snx - 1) - 1.0 if snx > 1 else src_x * 0.0
        gy = 2.0 * src_y / (sny - 1) - 1.0 if sny > 1 else src_y * 0.0
        gz = 2.0 * src_z / (snz - 1) - 1.0 if snz > 1 else src_z * 0.0

        grid = torch.stack([gx, gy, gz], dim=-1)[None]
        vol = source[None, None]
        result = _grid_sample_3d(vol, grid)[0, 0]

        if zero_outside:
            oob = (
                (src_x < -0.5) | (src_x > snx - 0.5)
                | (src_y < -0.5) | (src_y > sny - 0.5)
                | (src_z < -0.5) | (src_z > snz - 0.5)
            )
            result[oob] = 0.0
        return result

    onz, ony, onx = output_shape
    snz, sny, snx = source.shape
    src_coords = matrix @ coords  # (4, N)

    src_x = src_coords[0].reshape(onz, ony, onx)
    src_y = src_coords[1].reshape(onz, ony, onx)
    src_z = src_coords[2].reshape(onz, ony, onx)

    result = _separable_resample_3d(source, src_x, src_y, src_z, interp)

    if zero_outside:
        oob = (
            (src_x < -0.5) | (src_x > snx - 0.5)
            | (src_y < -0.5) | (src_y > sny - 0.5)
            | (src_z < -0.5) | (src_z > snz - 0.5)
        )
        result[oob] = 0.0

    return result


def apply_affine_batched(
    source: Tensor,
    matrices: Tensor,
    output_shape: tuple[int, int, int] | None = None,
) -> Tensor:
    """Apply B affine transforms to the same source in one call.

    This is the hot path for coarse search: evaluate thousands of candidate
    transforms simultaneously.

    Args:
        source: (nz, ny, nx) source image.
        matrices: (B, 4, 4) affine matrices in voxel index space.
        output_shape: (nz, ny, nx) of output grid. Defaults to source shape.

    Returns:
        (B, nz, ny, nx) resampled images.
    """
    B = matrices.shape[0]
    if output_shape is None:
        output_shape = source.shape
    onz, ony, onx = output_shape
    device = source.device
    dtype = source.dtype

    # Build output grid once: (4, N) homogeneous coords
    kk, jj, ii = torch.meshgrid(
        torch.arange(onz, dtype=dtype, device=device),
        torch.arange(ony, dtype=dtype, device=device),
        torch.arange(onx, dtype=dtype, device=device),
        indexing="ij",
    )
    N = onz * ony * onx
    coords = torch.stack(
        [
            ii.reshape(-1),
            jj.reshape(-1),
            kk.reshape(-1),
            torch.ones(N, device=device, dtype=dtype),
        ],
        dim=0,
    )  # (4, N)

    # Batched matmul: (B, 4, 4) @ (4, N) → (B, 4, N)
    src_coords = matrices @ coords[None].expand(
        B, -1, -1
    )  # broadcast: (B, 4, 4) @ (B, 4, N)

    src_x = src_coords[:, 0].reshape(B, onz, ony, onx)
    src_y = src_coords[:, 1].reshape(B, onz, ony, onx)
    src_z = src_coords[:, 2].reshape(B, onz, ony, onx)

    # Normalize to [-1, 1]
    snz, sny, snx = source.shape
    gx = 2.0 * src_x / (snx - 1) - 1.0 if snx > 1 else src_x * 0.0
    gy = 2.0 * src_y / (sny - 1) - 1.0 if sny > 1 else src_y * 0.0
    gz = 2.0 * src_z / (snz - 1) - 1.0 if snz > 1 else src_z * 0.0

    grid = torch.stack([gx, gy, gz], dim=-1)  # (B, D, H, W, 3)
    vol = source[None, None].expand(B, 1, snz, sny, snx)  # (B, 1, D, H, W)

    result = _grid_sample_3d(vol, grid)
    return result[:, 0]  # (B, D, H, W)


# ---------------------------------------------------------------------------
# Coordinate system conversion (voxel ↔ DICOM mm)
# ---------------------------------------------------------------------------


def _ijk2xyz_matrix(nifti_affine: np.ndarray) -> Tensor:
    """Get the voxel-index-to-xyz (DICOM mm) matrix from a NIfTI affine.

    NIfTI affine maps (i_nifti, j_nifti, k_nifti) to (x, y, z) mm.
    Our internal convention is (x_idx, y_idx, z_idx) = (i_nifti, j_nifti, k_nifti)
    since load_image transposes data from (nx, ny, nz) to (nz, ny, nx) but
    the affine stays the same — index (ix, iy, iz) in our convention maps to
    column ix, row iy, slice iz in NIfTI, which is (ix, iy, iz) in the affine.

    Args:
        nifti_affine: (4, 4) numpy array from the NIfTI header.

    Returns:
        (4, 4) torch tensor mapping our voxel indices to xyz mm.
    """
    return torch.from_numpy(nifti_affine.astype(np.float64)).float()


def voxel_matrix_to_dicom(
    M_ijk: Tensor,
    base_affine: np.ndarray,
    source_affine: np.ndarray,
) -> Tensor:
    """Convert a voxel-space affine to DICOM mm space.

    M_dicom = source_ijk2xyz @ M_ijk @ inv(base_ijk2xyz)

    Args:
        M_ijk: (4, 4) matrix mapping base voxels to source voxels.
        base_affine: NIfTI affine for the base image.
        source_affine: NIfTI affine for the source image.

    Returns:
        (4, 4) matrix in DICOM mm coordinates.
    """
    base_ijk2xyz = _ijk2xyz_matrix(base_affine)
    source_ijk2xyz = _ijk2xyz_matrix(source_affine)
    base_xyz2ijk = torch.linalg.inv(base_ijk2xyz)
    return source_ijk2xyz @ M_ijk @ base_xyz2ijk


def dicom_matrix_to_voxel(
    M_dicom: Tensor,
    base_affine: np.ndarray,
    source_affine: np.ndarray,
) -> Tensor:
    """Convert DICOM mm space affine(s) to voxel-space.

    M_ijk = inv(source_ijk2xyz) @ M_dicom @ base_ijk2xyz

    Args:
        M_dicom: (4, 4) or (T, 4, 4) matrix/matrices in DICOM mm coordinates.
        base_affine: NIfTI affine for the base image.
        source_affine: NIfTI affine for the source image.

    Returns:
        (4, 4) or (T, 4, 4) matrix/matrices mapping base voxels to source voxels.
    """
    base_ijk2xyz = _ijk2xyz_matrix(base_affine)
    source_ijk2xyz = _ijk2xyz_matrix(source_affine)
    source_xyz2ijk = torch.linalg.inv(source_ijk2xyz)

    if M_dicom.ndim == 2:
        return source_xyz2ijk @ M_dicom @ base_ijk2xyz
    else:
        T = M_dicom.shape[0]
        result = torch.zeros_like(M_dicom)
        for t in range(T):
            result[t] = source_xyz2ijk @ M_dicom[t] @ base_ijk2xyz
        return result


# ---------------------------------------------------------------------------
# AFNI .aff12.1D I/O
# ---------------------------------------------------------------------------


def save_matrix_1D(
    matrix_ijk: Tensor,
    path: str | Path,
    base_affine: np.ndarray,
    source_affine: np.ndarray,
) -> None:
    """Save affine matrix as AFNI-compatible .aff12.1D file.

    The .aff12.1D format stores 12 numbers (row-major 3x4) in DICOM mm
    coordinates. The matrix maps base xyz to source xyz.

    Args:
        matrix_ijk: (4, 4) affine in voxel index space.
        path: Output file path.
        base_affine: NIfTI affine for base image.
        source_affine: NIfTI affine for source image.
    """
    M_dicom = voxel_matrix_to_dicom(matrix_ijk, base_affine, source_affine)
    M = M_dicom.detach().cpu().numpy()

    # Write 12 numbers: rows 0-2, columns 0-3 (row-major)
    vals = []
    for i in range(3):
        for j in range(4):
            vals.append(f"{M[i, j]:.10f}")

    with open(str(path), "w") as f:
        f.write("  ".join(vals) + "\n")


def load_matrix_1D(
    path: str | Path,
    base_affine: np.ndarray | None = None,
    source_affine: np.ndarray | None = None,
) -> Tensor:
    """Load an AFNI .aff12.1D matrix file.

    If base_affine and source_affine are provided, converts from DICOM mm
    to voxel index space. Otherwise returns the DICOM mm matrix.

    Args:
        path: Path to .aff12.1D file.
        base_affine: NIfTI affine for base image (for voxel conversion).
        source_affine: NIfTI affine for source image (for voxel conversion).

    Returns:
        (4, 4) affine matrix.
    """
    with open(str(path)) as f:
        text = f.read().strip()

    vals = [float(x) for x in text.split()]
    if len(vals) != 12:
        raise ValueError(f"Expected 12 values in .aff12.1D, got {len(vals)}")

    M = torch.eye(4, dtype=torch.float32)
    for i in range(3):
        for j in range(4):
            M[i, j] = vals[i * 4 + j]

    if base_affine is not None and source_affine is not None:
        M = dicom_matrix_to_voxel(M, base_affine, source_affine)

    return M


# ---------------------------------------------------------------------------
# Utility: resample source to base grid
# ---------------------------------------------------------------------------


def resample_to_base_grid(
    source: Tensor,
    base_shape: tuple[int, int, int],
    source_affine: np.ndarray,
    base_affine: np.ndarray,
) -> Tensor:
    """Resample source image onto the base image grid.

    This is needed before optimization when source and base have different
    voxel grids (different resolution, FOV, or orientation).

    Args:
        source: (nz, ny, nx) source image in its native grid.
        base_shape: (nz, ny, nx) shape of the base image grid.
        source_affine: NIfTI affine for the source image.
        base_affine: NIfTI affine for the base image.

    Returns:
        Source image resampled to base grid shape.
    """
    device = source.device
    dtype = source.dtype

    # Matrix that maps base voxel indices to source voxel indices:
    # base_ijk → xyz → source_ijk
    base_ijk2xyz = torch.from_numpy(base_affine.astype(np.float64)).to(
        dtype=dtype, device=device
    )
    source_xyz2ijk = torch.linalg.inv(
        torch.from_numpy(source_affine.astype(np.float64)).to(
            dtype=dtype, device=device
        )
    )
    M = source_xyz2ijk @ base_ijk2xyz  # (4, 4): base voxel → source voxel

    return apply_affine(source, M, output_shape=base_shape)
