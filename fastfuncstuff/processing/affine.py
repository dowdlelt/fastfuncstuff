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
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from fastfuncstuff.memory import get_available_memory

from .interp import _grid_sample_3d, _separable_resample_3d, wsinc5_resample_3d

# Cache of constant homogeneous output-grid coordinates, keyed by
# (shape, device, dtype).  The grid is rebuilt on every apply_affine call
# otherwise, which dominates the per-iteration cost during optimization.
_GRID_CACHE: dict[tuple, Tensor] = {}


def _homogeneous_grid(
    shape: tuple[int, int, int], device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Return cached (4, N) homogeneous voxel coords for an output grid.

    Rows are (x, y, z, 1) in reshape(-1) (z, y, x) order.  Cached because the
    grid is constant across optimizer iterations; only the matrix changes.
    """
    key = (tuple(shape), device, dtype)
    coords = _GRID_CACHE.get(key)
    if coords is None:
        onz, ony, onx = shape
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
        )  # (4, N)
        _GRID_CACHE[key] = coords
    return coords


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
    r12 = sz_ * (-sy_) + (-cz) * sx_ * cy  # -sz*sy - cz*sx*cy ... wait let me redo this properly

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

    # D @ U with D = diag(sx, sy, sz): scale each ROW of U (row i by s_i), which
    # matches params_to_matrix's M3 = S @ D @ U. (Scaling by COLUMN here would be
    # U @ D — a different, wrong matrix whenever the scales are anisotropic, and
    # it silently diverged from the single-matrix path until batched affine
    # refinement used the gradient.)
    du00, du01, du02 = r00 * sx, r01 * sx, r02 * sx
    du10, du11, du12 = r10 * sy, r11 * sy, r12 * sy
    du20, du21, du22 = r20 * sz, r21 * sz, r22 * sz

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

    # Assemble (B, 4, 4) with a few stacks rather than ~13 scalar index-writes:
    # fewer kernel launches per call, which matters in the launch-bound batched
    # Adam loop. Still differentiable (stack carries grad to the m../d. terms).
    row0 = torch.stack([m00, m01, m02, dx], dim=1)
    row1 = torch.stack([m10, m11, m12, dy], dim=1)
    row2 = torch.stack([m20, m21, m22, dz], dim=1)
    row3 = torch.stack([_zero, _zero, _zero, one], dim=1)
    return torch.stack([row0, row1, row2, row3], dim=1)


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

    return torch.stack([dx, dy, dz, rz, rx, ry, sx_scale, sy_scale, sz_scale, shyx, shzx, shzy])


def decompose_affine_sdu(mat: Tensor) -> Tensor:
    """Decompose a 4x4 affine into the 12 params matching :func:`params_to_matrix`.

    Exact inverse of the T·S·D·U build (``params_to_matrix``): the 3x3 part is
    factored as ``M3 = S @ D @ U`` with ``U`` a proper rotation (Rz·Rx·Ry), ``D``
    a positive diagonal scale, and ``S`` a unit upper-triangular shear, via an RQ
    factorisation (``M3 = R @ Q``; ``R = S@D`` upper-triangular, ``Q = U``
    orthogonal) with signs fixed so the scales are positive. Unlike the polar
    decomposition in :func:`matrix_to_params`, this recovers AFNI's
    shift/angle/scale/shear parameters (used for the final-fit report).
    """
    from scipy.linalg import rq

    M = mat.detach().cpu().numpy().astype(np.float64)
    dx, dy, dz = M[0, 3], M[1, 3], M[2, 3]
    R, Q = rq(M[:3, :3])

    # Make diag(R) positive: R@diag(s) scales columns, diag(s)@Q scales rows, so
    # R@Q is preserved while D=diag(R) becomes the positive scales.
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    R = R * s[None, :]
    Q = s[:, None] * Q
    if np.linalg.det(Q) < 0:  # reflection (det M3 < 0): take a negative x-scale
        R[:, 0] *= -1.0
        Q[0, :] *= -1.0

    D = np.diag(R)
    shyx, shzx, shzy = R[0, 1] / D[1], R[0, 2] / D[2], R[1, 2] / D[2]

    rx = math.degrees(math.asin(float(np.clip(Q[2, 1], -1.0, 1.0))))
    if abs(math.cos(math.radians(rx))) > 1e-6:
        rz = math.degrees(math.atan2(-Q[0, 1], Q[1, 1]))
        ry = math.degrees(math.atan2(-Q[2, 0], Q[2, 2]))
    else:  # gimbal lock (rx ≈ ±90°): fold ry into rz
        rz = math.degrees(math.atan2(Q[1, 0], Q[0, 0]))
        ry = 0.0

    return torch.tensor(
        [dx, dy, dz, rz, rx, ry, D[0], D[1], D[2], shyx, shzx, shzy],
        dtype=torch.float32,
    )


def format_final_fit_params(params: Tensor, space: str = "DICOM mm") -> str:
    """Format 12 affine params as AFNI's '+ Final fine fit Parameters:' block.

    ``params`` is the :func:`params_to_matrix` ordering
    ``[dx,dy,dz, rz,rx,ry, sx,sy,sz, shyx,shzx,shzy]`` (translations in the units
    of ``space``). Mirrors 3dAllineate's report: translation enorm, the net
    rotation magnitude, the scale volume factor, and the three shears.
    """
    p = [float(v) for v in params]
    dx, dy, dz, rz, rx, ry, sx, sy, sz, shyx, shzx, shzy = p
    enorm = math.sqrt(dx * dx + dy * dy + dz * dz)
    total = math.sqrt(rz * rz + rx * rx + ry * ry)
    vol = sx * sy * sz
    cube = abs(vol) ** (1.0 / 3.0)
    if vol > 1.0:
        volnote = " [base smaller than source]"
    elif vol < 1.0:
        volnote = " [base larger than source]"
    else:
        volnote = ""
    return (
        f"+ Final fine fit Parameters ({space}):\n"
        f"     x-shift={dx:9.4f}   y-shift={dy:9.4f}   z-shift={dz:9.4f}  "
        f"...  enorm={enorm:9.4f}\n"
        f"     z-angle={rz:9.4f}   x-angle={rx:9.4f}   y-angle={ry:9.4f}  "
        f"...  total={total:9.4f} deg\n"
        f"     x-scale={sx:9.4f}   y-scale={sy:9.4f}   z-scale={sz:9.4f}  "
        f"...  vol3D={vol:8.4f}=({cube:.4f})^3{volnote}\n"
        f"   y/x-shear={shyx:9.4f} z/x-shear={shzx:9.4f} z/y-shear={shzy:9.4f}"
    )


def identity_params(device: torch.device = None, dtype: torch.dtype = torch.float32) -> Tensor:
    """Return identity affine parameters (12,)."""
    p = torch.zeros(12, device=device, dtype=dtype)
    p[6] = 1.0  # sx
    p[7] = 1.0  # sy
    p[8] = 1.0  # sz
    return p


# ---------------------------------------------------------------------------
# Apply affine transforms via grid_sample
# ---------------------------------------------------------------------------


def grid_from_dxyz(
    base_affine: np.ndarray,
    base_shape: tuple[int, int, int],
    dxyz: float | tuple[float, ...] | np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Master grid at a new voxel size — AFNI ``-mast_dxyz`` / ``-newgrid``.

    Keeps the base grid's **orientation** (direction cosines), **centre**, and **field of
    view**, but resamples to voxel size ``dxyz`` (one value → isotropic, or three in x,y,z
    voxel/world order). Use it to apply a transform into the base's *space* while preserving
    (or coarsening) resolution rather than snapping to the base's voxel grid.

    Args:
        base_affine: the base image's ``(x,y,z)`` voxel→world affine (4×4).
        base_shape: the base array shape ``(nz, ny, nx)``.
        dxyz: target voxel size(s) in mm.

    Returns:
        ``(new_affine, new_shape)`` — the new 4×4 voxel→world affine and array shape
        ``(nz, ny, nx)`` covering the same FOV, centred identically, at ``dxyz`` spacing.
    """
    a = np.asarray(base_affine, dtype=np.float64)
    nz, ny, nx = base_shape
    dim_xyz = np.array([nx, ny, nz], dtype=np.float64)  # dims in affine (x,y,z) axis order
    r = a[:3, :3]
    vox = np.linalg.norm(r, axis=0)
    vox = np.where(vox > 0.0, vox, 1.0)
    directions = r / vox  # unit direction cosines per axis
    d = np.asarray(dxyz, dtype=np.float64).ravel()
    if d.size == 1:
        d = np.repeat(d, 3)
    elif d.size != 3:
        raise ValueError(f"dxyz must be 1 or 3 values, got {d.size}")
    if np.any(d <= 0):
        raise ValueError(f"dxyz must be positive, got {d.tolist()}")
    new_dim = np.maximum(1, np.rint(dim_xyz * vox / d)).astype(int)  # preserve FOV = dim·vox
    new_r = directions * d  # scale each unit column by its target spacing
    centre_world = r @ ((dim_xyz - 1.0) / 2.0) + a[:3, 3]  # keep the same centre
    new_origin = centre_world - new_r @ ((new_dim - 1.0) / 2.0)
    new_affine = np.eye(4, dtype=np.float64)
    new_affine[:3, :3] = new_r
    new_affine[:3, 3] = new_origin
    return new_affine, (int(new_dim[2]), int(new_dim[1]), int(new_dim[0]))


def _slab_src_coords(
    matrix: Tensor,
    z0: int,
    z1: int,
    ony: int,
    onx: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    """Source (x, y, z) voxel coords for output slices ``[z0:z1]``.

    Same math as the whole-volume path, restricted to a slab of the output
    grid, and deliberately *not* routed through ``_homogeneous_grid``: that
    cache is keyed on the full output shape and would pin the very allocation
    slabbing exists to avoid (a 640^3 grid is 4.2 GB of (4, N) coordinates).
    """
    nz = z1 - z0
    kk, jj, ii = torch.meshgrid(
        torch.arange(z0, z1, dtype=dtype, device=device),
        torch.arange(ony, dtype=dtype, device=device),
        torch.arange(onx, dtype=dtype, device=device),
        indexing="ij",
    )
    coords = torch.stack(
        [
            ii.reshape(-1),
            jj.reshape(-1),
            kk.reshape(-1),
            torch.ones(nz * ony * onx, device=device, dtype=dtype),
        ],
        dim=0,
    )
    src = matrix @ coords  # (4, M)
    return (
        src[0].reshape(nz, ony, onx),
        src[1].reshape(nz, ony, onx),
        src[2].reshape(nz, ony, onx),
    )


def _z_slab_size(
    onz: int, ony: int, onx: int, device: torch.device, dtype: torch.dtype, live_per_voxel: int
) -> int:
    """How many output z-slices fit in the memory budget at once.

    ``live_per_voxel`` counts the simultaneously-live per-output-voxel tensors
    (coords, transformed coords, normalized grid, result, bounds masks). Returns
    ``onz`` when the whole volume fits, so ordinary-sized images keep the
    single-shot path and its cached grid.
    """
    itemsize = torch.empty((), dtype=dtype).element_size()
    per_z = max(ony * onx * live_per_voxel * itemsize, 1)
    try:
        budget = get_available_memory(device)
    except Exception:
        # Never let a memory probe be the thing that fails the resample.
        budget = 1 << 30
    return max(1, min(onz, int(budget // per_z)))


def apply_affine(
    source: Tensor,
    matrix: Tensor,
    output_shape: tuple[int, int, int] | None = None,
    zero_outside: bool = False,
) -> Tensor:
    """Resample source image using an affine transformation matrix.

    The matrix maps output (base) voxel indices to source voxel indices.

    Large output grids are resampled in z-slabs sized from the memory module.
    A single-shot 640^3 output needs ~16 bytes/voxel live across the coordinate
    grid, the normalized grid and the bounds masks -- about 16 GB, which OOMs
    even though the resample itself is trivially separable along z.

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
        onz, ony, onx = source.shape
    else:
        onz, ony, onx = output_shape
    device = source.device
    dtype = source.dtype

    snz, sny, snx = source.shape
    vol = source[None, None]  # (1, 1, D, H, W)

    def _slab(src_x: Tensor, src_y: Tensor, src_z: Tensor) -> Tensor:
        # Convert to normalized [-1, 1] for grid_sample
        gx = 2.0 * src_x / (snx - 1) - 1.0 if snx > 1 else src_x * 0.0
        gy = 2.0 * src_y / (sny - 1) - 1.0 if sny > 1 else src_y * 0.0
        gz = 2.0 * src_z / (snz - 1) - 1.0 if snz > 1 else src_z * 0.0

        grid = torch.stack([gx, gy, gz], dim=-1)[None]  # (1, D, H, W, 3)
        out = _grid_sample_3d(vol, grid)[0, 0]

        if zero_outside:
            oob = (
                (src_x < -0.5)
                | (src_x > snx - 0.5)
                | (src_y < -0.5)
                | (src_y > sny - 0.5)
                | (src_z < -0.5)
                | (src_z > snz - 0.5)
            )
            out[oob] = 0.0
        return out

    slab = _z_slab_size(onz, ony, onx, device, dtype, live_per_voxel=16)
    if slab >= onz:
        # Whole-volume path: keeps the cached grid, which is what makes the
        # thousands of calls in the optimizer loop cheap.
        coords = _homogeneous_grid((onz, ony, onx), device, dtype)
        src_coords = matrix @ coords  # (4, N)
        return _slab(
            src_coords[0].reshape(onz, ony, onx),
            src_coords[1].reshape(onz, ony, onx),
            src_coords[2].reshape(onz, ony, onx),
        )

    result = torch.empty((onz, ony, onx), device=device, dtype=dtype)
    for z0 in range(0, onz, slab):
        z1 = min(z0 + slab, onz)
        sx, sy, sz = _slab_src_coords(matrix, z0, z1, ony, onx, device, dtype)
        result[z0:z1] = _slab(sx, sy, sz)
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
        onz, ony, onx = source.shape
    else:
        onz, ony, onx = output_shape
    device = source.device
    dtype = source.dtype
    snz, sny, snx = source.shape

    def _slab(src_x: Tensor, src_y: Tensor, src_z: Tensor) -> Tensor:
        out = wsinc5_resample_3d(source, src_x, src_y, src_z)
        # Zero out-of-bounds voxels (wsinc5 clamps internally, so undo that)
        oob = (
            (src_x < -0.5)
            | (src_x > snx - 0.5)
            | (src_y < -0.5)
            | (src_y > sny - 0.5)
            | (src_z < -0.5)
            | (src_z > snz - 0.5)
        )
        out[oob] = 0.0
        return out

    # Higher per-voxel budget than the trilinear path: the 11-tap separable
    # resampler holds its own intermediates on top of the coordinate grids.
    slab = _z_slab_size(onz, ony, onx, device, dtype, live_per_voxel=28)

    if slab >= onz:
        sx, sy, sz = _slab_src_coords(matrix, 0, onz, ony, onx, device, dtype)
        return _slab(sx, sy, sz)

    result = torch.empty((onz, ony, onx), device=device, dtype=dtype)
    for z0 in range(0, onz, slab):
        z1 = min(z0 + slab, onz)
        sx, sy, sz = _slab_src_coords(matrix, z0, z1, ony, onx, device, dtype)
        result[z0:z1] = _slab(sx, sy, sz)
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
        onz, ony, onx = source.shape
    else:
        onz, ony, onx = output_shape
    device = source.device
    dtype = source.dtype
    snz, sny, snx = source.shape

    def _slab(src_x: Tensor, src_y: Tensor, src_z: Tensor) -> Tensor:
        out = _separable_resample_3d(source, src_x, src_y, src_z, interp)
        if zero_outside:
            oob = (
                (src_x < -0.5)
                | (src_x > snx - 0.5)
                | (src_y < -0.5)
                | (src_y > sny - 0.5)
                | (src_z < -0.5)
                | (src_z > snz - 0.5)
            )
            out[oob] = 0.0
        return out

    # As in apply_affine_wsinc5: the separable multi-tap resamplers carry their
    # own intermediates on top of the coordinate grids.
    slab = _z_slab_size(onz, ony, onx, device, dtype, live_per_voxel=28)

    if slab >= onz:
        sx, sy, sz = _slab_src_coords(matrix, 0, onz, ony, onx, device, dtype)
        return _slab(sx, sy, sz)

    result = torch.empty((onz, ony, onx), device=device, dtype=dtype)
    for z0 in range(0, onz, slab):
        z1 = min(z0 + slab, onz)
        sx, sy, sz = _slab_src_coords(matrix, z0, z1, ony, onx, device, dtype)
        result[z0:z1] = _slab(sx, sy, sz)
    return result


def apply_affine_interp_batched(
    sources: Tensor,
    matrices: Tensor,
    interp: str = "heptic",
    output_shape: tuple[int, int, int] | None = None,
    zero_outside: bool = False,
) -> Tensor:
    """Resample a batch of volumes with a batch of affine transforms.

    Builds the coordinate grid once and computes all B transforms via a
    single batched matmul, then loops calling the per-volume separable
    kernel.  For Pass 2 moco resampling the cost is dominated by the
    kernel, but this eliminates B repeated grid-builds and transfers.

    Args:
        sources: (B, nz, ny, nx) source volumes, already on device.
        matrices: (B, 4, 4) affine matrices in voxel index space.
        interp: interpolation method ("linear", "cubic", "quintic",
            "heptic", "wsinc5").
        output_shape: (nz, ny, nx) of output grid. Defaults to source shape.
        zero_outside: If True, zero voxels that map outside the source.

    Returns:
        (B, nz, ny, nx) resampled volumes.
    """
    B = sources.shape[0]
    device = sources.device
    dtype = sources.dtype
    snz, sny, snx = sources.shape[1:]
    onz, ony, onx = output_shape if output_shape is not None else (snz, sny, snx)

    # Build output coord grid once: (4, N)
    coords = _build_homo_coords((onz, ony, onx), device, dtype)

    # Batched coord transform: (B, 4, 4) @ (4, N) → (B, 4, N)
    # Expand coords to (1, 4, N) then broadcast via bmm
    src_coords = torch.bmm(matrices, coords.unsqueeze(0).expand(B, -1, -1))  # (B, 4, N)

    results = torch.zeros(B, onz, ony, onx, device=device, dtype=dtype)

    if interp == "linear":
        for b in range(B):
            M = matrices[b]
            results[b] = apply_affine(sources[b], M, output_shape, zero_outside=zero_outside)
        return results

    for b in range(B):
        src_x = src_coords[b, 0].reshape(onz, ony, onx)
        src_y = src_coords[b, 1].reshape(onz, ony, onx)
        src_z = src_coords[b, 2].reshape(onz, ony, onx)

        results[b] = _separable_resample_3d(sources[b], src_x, src_y, src_z, interp)

        if zero_outside:
            oob = (
                (src_x < -0.5)
                | (src_x > snx - 0.5)
                | (src_y < -0.5)
                | (src_y > sny - 0.5)
                | (src_z < -0.5)
                | (src_z > snz - 0.5)
            )
            results[b][oob] = 0.0

    return results


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
                (src_x < -0.5)
                | (src_x > snx - 0.5)
                | (src_y < -0.5)
                | (src_y > sny - 0.5)
                | (src_z < -0.5)
                | (src_z > snz - 0.5)
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
            (src_x < -0.5)
            | (src_x > snx - 0.5)
            | (src_y < -0.5)
            | (src_y > sny - 0.5)
            | (src_z < -0.5)
            | (src_z > snz - 0.5)
        )
        result[oob] = 0.0

    return result


def sample_affine_at_points(
    source: Tensor,
    matrix: Tensor,
    points_xyz: Tensor,
    zero_outside: bool = True,
    interp: str = "linear",
) -> Tensor:
    """Sample ``source`` at M base points transformed by ``matrix`` -> (M,).

    The point-wise analogue of :func:`apply_affine`: instead of warping the full
    output grid, only the given base points are transformed and interpolated, so
    the cost is O(M) rather than O(N_voxels). Used by the subsampled refinement
    path (AFNI npt_match). Differentiable in ``matrix``.

    Args:
        source: (nz, ny, nx) source image.
        matrix: (4, 4) affine mapping base voxels -> source voxels.
        points_xyz: (M, 3) base voxel coordinates as (x, y, z) (x fastest).
        zero_outside: zero points mapping outside the source (AFNI outval=0).

    Returns:
        (M,) interpolated source values.
    """
    device = source.device
    dtype = source.dtype
    M = points_xyz.shape[0]
    pts = points_xyz.to(device=device, dtype=dtype)
    homog = torch.cat([pts, torch.ones(M, 1, device=device, dtype=dtype)], dim=1)  # (M,4)
    src = homog @ matrix.T  # (M, 4)
    sx, sy, sz = src[:, 0], src[:, 1], src[:, 2]

    snz, sny, snx = source.shape
    gx = 2.0 * sx / (snx - 1) - 1.0 if snx > 1 else sx * 0.0
    gy = 2.0 * sy / (sny - 1) - 1.0 if sny > 1 else sy * 0.0
    gz = 2.0 * sz / (snz - 1) - 1.0 if snz > 1 else sz * 0.0

    if interp == "linear":
        # grid_sample wants (N, D, H, W, 3); pack the M points along one axis.
        grid = torch.stack([gx, gy, gz], dim=-1).view(1, M, 1, 1, 3)
        vol = source[None, None]  # (1, 1, nz, ny, nx)
        vals = _grid_sample_3d(vol, grid).reshape(M)
    else:
        vals = _separable_resample_3d(source, sx, sy, sz, interp).reshape(M)

    if zero_outside:
        oob = (
            (sx < -0.5)
            | (sx > snx - 0.5)
            | (sy < -0.5)
            | (sy > sny - 0.5)
            | (sz < -0.5)
            | (sz > snz - 0.5)
        )
        vals = vals.masked_fill(oob, 0.0)
    return vals


def batched_sample_bytes_per_point(interp: str, *, grad: bool = False) -> int:
    """Peak bytes per (candidate, point) pair inside :func:`sample_affine_at_points_batched`.

    Callers size their candidate chunk from this. The branches are not remotely
    alike, and modelling them all with the grid_sample number is what OOM'd the
    coarse search on a cubic run:

    linear     : the grid (12) plus values and the out-of-bounds mask (~9).
                 Folding the normalization into the candidate matrix removed the
                 (B, M, 4) voxel-coordinate tensor and the three gx/gy/gz planes
                 that used to dominate this; measured 21.2 B/point at B=170.
    separable  : homogeneous coords (16) plus the flat-coordinate working set of
                 :func:`_separable_resample_3d` -- contiguous x/y/z copies (12),
                 floor/round bases (12), in-bounds/tiny/heavy masks (3), the
                 int64 survivor index (8), gathered survivor coords (12), the
                 result (4) -- and values plus the mask (~9). The gather slab
                 itself is chunked separately by ``_resample_chunk_size``.

    ``grad`` is a different regime, not a correction factor. Nothing transient
    can be freed: every gather chunk's saved tensors stay live until backward,
    so the ``ntaps**2`` slab and the three int64 tap-index grids the forward
    pass frees per chunk are all retained at once. Measured at 588 B/point for
    cubic (17 trials x 1.22M points peaked at 12.2 GiB); 600 with a little
    headroom. Linear-with-grad measures 41.8 B/point at the 17 trials that
    regime actually runs, where the fixed cost is worst.
    """
    if interp == "linear":
        return 48 if grad else 24
    return 600 if grad else 80


@lru_cache(maxsize=32)
def _normalized_grid_constants(
    shape: tuple[int, int, int], device: torch.device, dtype: torch.dtype
) -> tuple[Tensor, Tensor, Tensor]:
    """Grid-coordinate map and in-bounds box for one source grid.

    Memoised because these are three tiny host-built tensors, and the ~110 us of
    transfer and launch to rebuild them was a third of the sampler's wall time at
    the small candidate counts ``-onepass`` produces.

    Returns ``(matrix, lo, hi)``: a (3, 4) map from homogeneous voxel coords to
    ``align_corners`` grid coords, and AFNI's [-0.5, n-0.5] in-bounds box carried
    through that same map. A degenerate axis keeps grid_sample's convention of
    sampling its single plane, so its scale is zero rather than 2/(n-1).
    """
    snz, sny, snx = shape
    scale = torch.tensor(
        [
            2.0 / (snx - 1) if snx > 1 else 0.0,
            2.0 / (sny - 1) if sny > 1 else 0.0,
            2.0 / (snz - 1) if snz > 1 else 0.0,
        ],
        device=device,
        dtype=dtype,
    )
    matrix = torch.zeros(3, 4, device=device, dtype=dtype)
    matrix[0, 0], matrix[1, 1], matrix[2, 2] = scale[0], scale[1], scale[2]
    matrix[:, 3] = -1.0
    upper = torch.tensor([snx - 0.5, sny - 0.5, snz - 0.5], device=device, dtype=dtype)
    return matrix, scale * (-0.5) - 1.0, scale * upper - 1.0


def sample_affine_at_points_batched(
    source: Tensor,
    matrices: Tensor,
    points_xyz: Tensor,
    zero_outside: bool = True,
    interp: str = "linear",
) -> Tensor:
    """Sample ``source`` at M points under B transforms -> (B, M).

    Batched :func:`sample_affine_at_points` for the subsampled coarse search:
    one source volume, B candidate matrices, the same M base points. "linear"
    uses a single grid_sample with the points packed into the (D, H) grid dims,
    so only one copy of the source is needed regardless of B; other kernels go
    through the separable resampler, which takes scattered coordinates directly.
    Caller should chunk B to bound the B*M grid memory.
    """
    device = source.device
    dtype = source.dtype
    B = matrices.shape[0]
    M = points_xyz.shape[0]
    pts = points_xyz.to(device=device, dtype=dtype)
    homog = torch.cat([pts, torch.ones(M, 1, device=device, dtype=dtype)], dim=1)  # (M,4)
    snz, sny, snx = source.shape

    if interp == "linear":
        # Normalizing *after* the transform was half the sampler's wall time: it
        # reads three (B, M) planes, writes three more, then stacks a fourth copy
        # -- 124 MB of pure traffic at a 170-candidate generation, against a
        # gather that costs a quarter of that. The map to grid_sample's [-1, 1]
        # is itself affine, so it composes into the candidate matrix instead and
        # the einsum emits the grid directly. That also drops the fourth
        # homogeneous component, which nothing downstream ever read.
        norm, lo, hi = _normalized_grid_constants((snz, sny, snx), device, dtype)
        grid = torch.einsum("bij,mj->bmi", norm @ matrices.to(dtype), homog)  # (B, M, 3)
        vals = _grid_sample_3d(source[None, None], grid.view(1, B, M, 1, 3)).reshape(B, M)
        if zero_outside:
            # AFNI's outval=0 half-voxel border, carried through the same affine
            # map. Testing the grid rather than the voxel coordinate tests the
            # position grid_sample actually reads.
            vals = vals.masked_fill((grid < lo).any(-1) | (grid > hi).any(-1), 0.0)
        return vals

    src = torch.einsum("bij,mj->bmi", matrices.to(dtype), homog)  # (B, M, 4)
    sx, sy, sz = src[..., 0], src[..., 1], src[..., 2]  # (B, M)
    vals = _separable_resample_3d(source, sx, sy, sz, interp).reshape(B, M)
    if zero_outside:
        oob = (
            (sx < -0.5)
            | (sx > snx - 0.5)
            | (sy < -0.5)
            | (sy > sny - 0.5)
            | (sz < -0.5)
            | (sz > snz - 0.5)
        )
        vals = vals.masked_fill(oob, 0.0)
    return vals


def apply_affine_batched(
    source: Tensor,
    matrices: Tensor,
    output_shape: tuple[int, int, int] | None = None,
    zero_outside: bool = False,
) -> Tensor:
    """Apply B affine transforms to the same source in one call.

    This is the hot path for coarse search: evaluate thousands of candidate
    transforms simultaneously.

    Args:
        source: (nz, ny, nx) source image.
        matrices: (B, 4, 4) affine matrices in voxel index space.
        output_shape: (nz, ny, nx) of output grid. Defaults to source shape.
        zero_outside: If True, zero voxels that map outside the source volume
            instead of using border padding (AFNI outval=0).

    Returns:
        (B, nz, ny, nx) resampled images.
    """
    B = matrices.shape[0]
    if output_shape is None:
        onz, ony, onx = source.shape
    else:
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
    src_coords = matrices @ coords[None].expand(B, -1, -1)  # broadcast: (B, 4, 4) @ (B, 4, N)

    # Normalize coords to [-1, 1] grid directly, freeing intermediates
    snz, sny, snx = source.shape
    sx = src_coords[:, 0].reshape(B, onz, ony, onx)
    sy = src_coords[:, 1].reshape(B, onz, ony, onx)
    sz = src_coords[:, 2].reshape(B, onz, ony, onx)
    del src_coords
    gx = 2.0 * sx / (snx - 1) - 1.0 if snx > 1 else sx * 0.0
    gy = 2.0 * sy / (sny - 1) - 1.0 if sny > 1 else sy * 0.0
    gz = 2.0 * sz / (snz - 1) - 1.0 if snz > 1 else sz * 0.0

    if zero_outside:
        oob = (
            (sx < -0.5)
            | (sx > snx - 0.5)
            | (sy < -0.5)
            | (sy > sny - 0.5)
            | (sz < -0.5)
            | (sz > snz - 0.5)
        )
    del sx, sy, sz

    grid = torch.stack([gx, gy, gz], dim=-1)  # (B, D, H, W, 3)
    del gx, gy, gz
    vol = source[None, None].expand(B, 1, snz, sny, snx)  # (B, 1, D, H, W)

    result = _grid_sample_3d(vol, grid)[:, 0]  # (B, D, H, W)
    if zero_outside:
        result = result.masked_fill(oob, 0.0)
    return result


# ---------------------------------------------------------------------------
# Coordinate system conversion (voxel ↔ DICOM mm)
# ---------------------------------------------------------------------------


def _ijk2ras_matrix(nifti_affine: np.ndarray) -> Tensor:
    """Get the voxel-index-to-RAS mm matrix from a NIfTI affine.

    NIfTI affine maps (i, j, k) to RAS (x_right, y_anterior, z_superior) mm.

    Args:
        nifti_affine: (4, 4) numpy array from the NIfTI header.

    Returns:
        (4, 4) torch tensor mapping voxel indices to RAS mm.
    """
    return torch.from_numpy(nifti_affine.astype(np.float64)).float()


# RAS-to-DICOM sign flip: AFNI's DICOM convention is (x=-R+L, y=-A+P, z=-I+S),
# which negates the first two axes relative to NIfTI's RAS.
_RAS_TO_DICOM = torch.tensor(
    [
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
    dtype=torch.float32,
)


def voxel_matrix_to_dicom(
    M_ijk: Tensor,
    base_affine: np.ndarray,
    source_affine: np.ndarray,
) -> Tensor:
    """Convert a voxel-space affine to AFNI DICOM mm space.

    AFNI's .aff12.1D format uses DICOM coordinates (x=-R+L, y=-A+P, z=-I+S),
    which differs from NIfTI's RAS by negating x and y.

    Steps:
      1. M_ras = source_ijk2ras @ M_ijk @ inv(base_ijk2ras)
      2. M_dicom = D @ M_ras @ D  where D = diag(-1,-1,1,1)

    The ijk→ras matrices use the CARDINAL (deobliqued) affine, matching AFNI's
    ``ijk_to_dicom`` (axis-snapped), which every AFNI matrix conversion uses —
    NOT the raw oblique sform. For cardinal data the two coincide; for oblique
    data, using the raw sform here silently rotated cross-modal matrices by the
    obliquity angle, so ffs's own ``.aff12.1D`` matrices disagreed with
    ``3dNwarpApply`` / ``ffs_nwarp`` (which are cardinal). See
    [[project_nwarp_oblique_matrix_interop]].

    Args:
        M_ijk: (4, 4) matrix mapping base voxels to source voxels.
        base_affine: NIfTI affine for the base image.
        source_affine: NIfTI affine for the source image.

    Returns:
        (4, 4) matrix in DICOM mm coordinates (AFNI convention).
    """
    from .nwarpforge import compute_cardinal_affine

    device = M_ijk.device
    base_affine = compute_cardinal_affine(np.asarray(base_affine, dtype=np.float64))
    source_affine = compute_cardinal_affine(np.asarray(source_affine, dtype=np.float64))
    base_ijk2ras = _ijk2ras_matrix(base_affine).to(device)
    source_ijk2ras = _ijk2ras_matrix(source_affine).to(device)
    base_ras2ijk = torch.linalg.inv(base_ijk2ras)

    # Step 1: voxel → RAS
    M_ras = source_ijk2ras @ M_ijk @ base_ras2ijk

    # Step 2: RAS → DICOM (negate x,y rows and columns)
    D = _RAS_TO_DICOM.to(device)
    return D @ M_ras @ D


def dicom_matrix_to_voxel(
    M_dicom: Tensor,
    base_affine: np.ndarray,
    source_affine: np.ndarray,
) -> Tensor:
    """Convert AFNI DICOM mm space affine(s) to voxel-space.

    Steps:
      1. M_ras = D @ M_dicom @ D  where D = diag(-1,-1,1,1)
      2. M_ijk = inv(source_ijk2ras) @ M_ras @ base_ijk2ras

    Uses the CARDINAL (deobliqued) affine to match AFNI's ``ijk_to_dicom``; the
    exact inverse of :func:`voxel_matrix_to_dicom`. See
    [[project_nwarp_oblique_matrix_interop]].

    Args:
        M_dicom: (4, 4) or (T, 4, 4) matrix/matrices in DICOM mm coordinates.
        base_affine: NIfTI affine for the base image.
        source_affine: NIfTI affine for the source image.

    Returns:
        (4, 4) or (T, 4, 4) matrix/matrices mapping base voxels to source voxels.
    """
    from .nwarpforge import compute_cardinal_affine

    device = M_dicom.device
    D = _RAS_TO_DICOM.to(device)
    base_affine = compute_cardinal_affine(np.asarray(base_affine, dtype=np.float64))
    source_affine = compute_cardinal_affine(np.asarray(source_affine, dtype=np.float64))
    base_ijk2ras = _ijk2ras_matrix(base_affine).to(device)
    source_ijk2ras = _ijk2ras_matrix(source_affine).to(device)
    source_ras2ijk = torch.linalg.inv(source_ijk2ras)

    if M_dicom.ndim == 2:
        # DICOM → RAS → voxel
        M_ras = D @ M_dicom @ D
        return source_ras2ijk @ M_ras @ base_ijk2ras
    else:
        T = M_dicom.shape[0]
        result = torch.zeros_like(M_dicom)
        for t in range(T):
            M_ras = D @ M_dicom[t] @ D
            result[t] = source_ras2ijk @ M_ras @ base_ijk2ras
        return result


# ---------------------------------------------------------------------------
# AFNI .aff12.1D I/O
# ---------------------------------------------------------------------------


def save_matrix_1D(
    matrix: Tensor | np.ndarray,
    path: str | Path,
    base_affine: np.ndarray | None = None,
    source_affine: np.ndarray | None = None,
    header: str | None = None,
) -> None:
    """Save affine matrix/matrices as an AFNI-compatible ``.aff12.1D`` file.

    Writes 12 row-major numbers (3×4) per line in DICOM mm coordinates,
    base→source, matching ``3dvolreg -1Dmatrix_save`` and
    ``3dAllineate -1Dmatrix_save`` (see AFNI ``3dvolreg.c:1507``).

    Accepts either:
      - voxel-space matrix ``(4, 4)`` plus ``base_affine`` and ``source_affine``
        (the canonical allineate case), which are used to convert to DICOM; or
      - one or more pre-converted DICOM-space matrices ``(4, 4)`` or
        ``(nt, 4, 4)`` with ``base_affine`` / ``source_affine`` left ``None``
        (the moco case where per-volume DICOM matrices are already computed).

    Args:
        matrix: affine matrix/matrices, shape (4,4) or (nt,4,4).
        path: output file path.
        base_affine: NIfTI affine for base image. Pass together with
            ``source_affine`` to indicate ``matrix`` is in voxel-index space.
        source_affine: NIfTI affine for source image.
        header: optional comment line written as ``# {header}`` at the top.
    """
    if (base_affine is None) != (source_affine is None):
        raise ValueError("save_matrix_1D: pass both base_affine and source_affine, or neither")

    if base_affine is not None:
        m_t = matrix if isinstance(matrix, Tensor) else torch.as_tensor(matrix)
        M = voxel_matrix_to_dicom(m_t, base_affine, source_affine)
    else:
        M = matrix

    if isinstance(M, Tensor):
        M = M.detach().cpu().numpy()
    M = np.asarray(M)
    if M.ndim == 2:
        M = M[None]
    if M.ndim != 3 or M.shape[-2:] != (4, 4):
        raise ValueError(f"save_matrix_1D: expected (4,4) or (nt,4,4), got {M.shape}")

    with open(str(path), "w") as f:
        if header:
            f.write(f"# {header}\n")
        for t in range(M.shape[0]):
            vals = [f"{M[t, i, j]:.10f}" for i in range(3) for j in range(4)]
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


def load_matrix_chain(
    paths: list[str] | str,
    base_affine: np.ndarray,
    source_affine: np.ndarray,
) -> Tensor:
    """Load and compose a stack of ``.aff12.1D`` affines into one voxel matrix.

    The stack order matches AFNI's ``-nwarp`` / ``3dNwarpCat`` catenation: the
    list runs **base-side → source-side** (leftmost closest to the output/base
    space, rightmost closest to the source). For example an ``epi → ref → anat``
    alignment is passed as ``["ref2anat.aff12.1D", "epi2ref.aff12.1D"]`` and
    yields the single ``anat → epi`` voxel map (base = anat, source = epi).

    Because the files are in DICOM mm (a physical space), the catenation is a
    plain matrix product there — the *intermediate* grids (e.g. the reference
    image) are never needed, so no reference dataset has to be supplied. Only the
    final ``base_affine`` (output grid) and ``source_affine`` (the volume the
    composite maps *into*) are used, to convert the composite to voxel indices.

    A single-element list reproduces ``load_matrix_1D(path, base, source)``.

    Args:
        paths: one ``.aff12.1D`` path or a base-side→source-side list of them.
        base_affine: NIfTI affine of the final base (output) grid.
        source_affine: NIfTI affine of the final source grid.

    Returns:
        (4, 4) voxel-index matrix mapping base voxels → source voxels.
    """
    if isinstance(paths, str):
        paths = [paths]
    if not paths:
        raise ValueError("load_matrix_chain: empty matrix list")
    # Compose in DICOM mm: C = M_last @ … @ M_first (each new file left-multiplies).
    c = torch.eye(4, dtype=torch.float64)
    for p in paths:
        c = load_matrix_1D(p).to(torch.float64) @ c
    return dicom_matrix_to_voxel(c.to(torch.float32), base_affine, source_affine)


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
    base_ijk2xyz = torch.from_numpy(base_affine.astype(np.float64)).to(dtype=dtype, device=device)
    source_xyz2ijk = torch.linalg.inv(
        torch.from_numpy(source_affine.astype(np.float64)).to(dtype=dtype, device=device)
    )
    M = source_xyz2ijk @ base_ijk2xyz  # (4, 4): base voxel → source voxel

    return apply_affine(source, M, output_shape=base_shape)


def base_into_source_frame(
    base: Tensor,
    base_info: dict,
    source_shape: tuple[int, ...],
    source_info: dict,
    matrix_path: str | Path,
    device: torch.device,
) -> tuple[Tensor, dict]:
    """Pull the base onto the source's own grid, given source->base as a matrix.

    The ``-matrix`` half of every nonlinear registration CLI: instead of the caller
    resampling the source onto the base grid and handing over the result, they hand
    over the *un*-resampled source and the affine that would have moved it. The
    matrix is inverted, the base is pulled onto the source's grid, and the warp is
    solved there, so the source keeps every voxel it acquired and is interpolated
    exactly once -- when the whole chain is finally applied.

    Shared by the backends rather than reimplemented per tool: the inversion, the
    interpolation kernel and which header the outputs inherit all have to agree
    across them, or a chain built by one and consumed by another composes in the
    wrong frame. Returns the resampled base and the header info the outputs carry
    (the source's, since that is now the grid everything lives on).
    """
    m_b2s = load_matrix_1D(matrix_path, base_info["affine"], source_info["affine"])
    m_s2b = torch.linalg.inv(m_b2s.double()).float().to(device)
    resampled = apply_affine_interp(
        base.float().to(device),
        m_s2b,
        interp="wsinc5",
        output_shape=tuple(source_shape),
        zero_outside=True,
    ).cpu()
    return resampled, source_info
