"""GPU-accelerated multi-warp composition and application.

Equivalent to 3dNwarpApply - composes chains of affine matrices and nonlinear
warps, then applies to a source dataset with high-quality interpolation.

Composition semantics (pull/backward mapping):
    For -nwarp 'A B C', the result is C(B(A(x)))
    where x is a coordinate in output space, and the composed warp
    tells where to sample from in source space.

Key functions:
    - nwarpforge: Main pipeline
    - compose_chain: Compose list of transforms
    - apply_composed_warp: Apply composed warp to source
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from .interp import trilinear_interpolate, warp_image_wsinc5
from .io import derive_mean_output_path, load_image, load_warp_field, save_image


@dataclass
class AffineTransform:
    """Time-dependent affine matrix (analogous to AFNI's mat44_vec)."""

    matrices: Tensor  # (T, 4, 4) where T>=1
    base_affine: np.ndarray | None = None
    source_affine: np.ndarray | None = None

    @property
    def is_time_dependent(self) -> bool:
        return self.matrices.shape[0] > 1

    def at_time(self, t: int) -> Tensor:
        idx = min(t, self.matrices.shape[0] - 1)
        return self.matrices[idx]


@dataclass
class NonlinearWarp:
    """Displacement field in voxel units."""

    xd: Tensor  # (nz, ny, nx)
    yd: Tensor
    zd: Tensor
    header_info: dict

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.xd.shape


Transform = AffineTransform | NonlinearWarp


def load_affine_1D(
    path: str | Path,
    output_affine: np.ndarray,
    device: torch.device | None = None,
    debug: bool = False,
) -> AffineTransform:
    """Load AFNI .aff12.1D matrix file (single or multi-row).

    Matrices are converted from DICOM mm to voxel index space using
    the OUTPUT grid's affine (matching AFNI's behavior where all
    matrices use cmat/imat from the output grid).

    Args:
        path: Path to .1D file
        output_affine: NIfTI affine for output/master grid (all matrices
                       are converted to this grid's index space)
        device: Torch device
        debug: Print debug info

    Returns:
        AffineTransform with matrices in voxel space on output grid
    """
    with open(str(path)) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    if len(lines) == 1:
        vals = [float(x) for x in lines[0].split()]
        if len(vals) == 12:
            mats_12 = np.array([vals], dtype=np.float32)
        elif len(vals) == 3:
            raise ValueError(f"3x4 matrix format not yet supported: {path}")
        else:
            raise ValueError(f"Expected 12 values per row in {path}, got {len(vals)}")
    else:
        mats_12 = np.array(
            [[float(x) for x in line.split()[:12]] for line in lines], dtype=np.float32
        )
        if mats_12.shape[1] != 12:
            raise ValueError(
                f"Expected 12 values per row in {path}, got {mats_12.shape[1]}"
            )

    T = mats_12.shape[0]
    matrices = torch.zeros(T, 4, 4, dtype=torch.float32)
    for t in range(T):
        for i in range(3):
            for j in range(4):
                matrices[t, i, j] = float(mats_12[t, i * 4 + j])
        matrices[t, 3, 3] = 1.0

    if debug:
        m = matrices[0].cpu().numpy()
        print("[DEBUG] load_affine_1D: Raw matrix from file (t=0):")
        print(f"  [{m[0, 0]:9.5f} {m[0, 1]:9.5f} {m[0, 2]:9.5f} {m[0, 3]:9.5f}]")
        print(f"  [{m[1, 0]:9.5f} {m[1, 1]:9.5f} {m[1, 2]:9.5f} {m[1, 3]:9.5f}]")
        print(f"  [{m[2, 0]:9.5f} {m[2, 1]:9.5f} {m[2, 2]:9.5f} {m[2, 3]:9.5f}]")
        print(f"  [{m[3, 0]:9.5f} {m[3, 1]:9.5f} {m[3, 2]:9.5f} {m[3, 3]:9.5f}]")

    # Convert DICOM mm -> output grid voxel indices
    # AFNI uses a coordinate system that differs from NIfTI by negating x and y.
    # When AFNI reads NIfTI, it creates ijk_to_dicom by negating rows 0,1 of sform.
    # So AFNI_DICOM = flip @ NIfTI_RAS where flip = diag(-1,-1,1,1)
    # To use DICOM matrices with NIfTI affines, we need:
    #   M_nifti = flip @ M_dicom @ flip
    # Then: M_index = inv(nifti_affine) @ M_nifti @ nifti_affine
    flip = torch.tensor(
        [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
    )
    output_ijk2xyz = torch.from_numpy(output_affine.astype(np.float32)).float()
    output_xyz2ijk = torch.linalg.inv(output_ijk2xyz)

    if debug:
        print("[DEBUG] load_affine_1D: output_affine (NIfTI sform):")
        a = output_ijk2xyz.cpu().numpy()
        print(f"  [{a[0, 0]:9.5f} {a[0, 1]:9.5f} {a[0, 2]:9.5f} {a[0, 3]:9.5f}]")
        print(f"  [{a[1, 0]:9.5f} {a[1, 1]:9.5f} {a[1, 2]:9.5f} {a[1, 3]:9.5f}]")
        print(f"  [{a[2, 0]:9.5f} {a[2, 1]:9.5f} {a[2, 2]:9.5f} {a[2, 3]:9.5f}]")
        print(f"  [{a[3, 0]:9.5f} {a[3, 1]:9.5f} {a[3, 2]:9.5f} {a[3, 3]:9.5f}]")

    for t in range(T):
        M_nifti = flip @ matrices[t] @ flip
        matrices[t] = output_xyz2ijk @ M_nifti @ output_ijk2xyz

    if device is not None:
        matrices = matrices.to(device)

    return AffineTransform(
        matrices=matrices,
        base_affine=output_affine.copy(),
        source_affine=output_affine.copy(),
    )


def load_warp(
    path: str | Path,
    device: torch.device | None = None,
    units: str = "mm",
    debug: bool = False,
) -> NonlinearWarp:
    """Load nonlinear warp from 4D NIfTI.

    Args:
        path: Path to warp file
        device: Torch device
        units: "mm" (convert DICOM mm to voxels, default), "voxels" (as-is)
        debug: Print debug info

    Returns:
        NonlinearWarp with displacement fields in voxel units
    """
    xd, yd, zd, header_info = load_warp_field(path, device=device)

    if debug:
        print("[DEBUG] load_warp: Raw displacement ranges (before conversion):")
        print(f"  xd: [{xd.min().item():.3f}, {xd.max().item():.3f}]")
        print(f"  yd: [{yd.min().item():.3f}, {yd.max().item():.3f}]")
        print(f"  zd: [{zd.min().item():.3f}, {zd.max().item():.3f}]")
        a = header_info["affine"]
        print("[DEBUG] load_warp: Warp file affine:")
        print(f"  [{a[0, 0]:9.5f} {a[0, 1]:9.5f} {a[0, 2]:9.5f} {a[0, 3]:9.5f}]")
        print(f"  [{a[1, 0]:9.5f} {a[1, 1]:9.5f} {a[1, 2]:9.5f} {a[1, 3]:9.5f}]")
        print(f"  [{a[2, 0]:9.5f} {a[2, 1]:9.5f} {a[2, 2]:9.5f} {a[2, 3]:9.5f}]")

    if units == "mm":
        xd, yd, zd = _convert_warp_mm_to_voxels(xd, yd, zd, header_info["affine"])
        if debug:
            print("[DEBUG] load_warp: After DICOM mm->voxel conversion:")
            print(f"  xd: [{xd.min().item():.3f}, {xd.max().item():.3f}]")
            print(f"  yd: [{yd.min().item():.3f}, {yd.max().item():.3f}]")
            print(f"  zd: [{zd.min().item():.3f}, {zd.max().item():.3f}]")
    elif debug:
        print("[DEBUG] load_warp: Warps assumed to be in voxel units (no conversion)")

    return NonlinearWarp(xd=xd, yd=yd, zd=zd, header_info=header_info)


def _warp_likely_in_mm(xd: Tensor, yd: Tensor, zd: Tensor, header_info: dict) -> bool:
    """Heuristic: if max displacement > typical voxel size, likely in mm.

    AFNI warps are ALWAYS in DICOM mm, but we check anyway.
    Uses max absolute displacement across all three components.
    """
    max_disp = max(xd.abs().max().item(), yd.abs().max().item(), zd.abs().max().item())
    # If max displacement > 2 voxels worth, assume mm
    # Typical voxel is 1-3mm, so displacements > 5 are definitely mm
    return max_disp > 5.0


def _convert_warp_mm_to_voxels(
    xd: Tensor, yd: Tensor, zd: Tensor, affine: np.ndarray
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert warp displacements from DICOM mm to voxel units.

    AFNI warp files store displacements in DICOM mm (LPS convention).
    NIfTI uses RAS convention. We need to negate x and y displacements
    to convert from DICOM to NIfTI coordinates, then apply the inverse
    of the NIfTI affine's rotation/scaling to get voxel displacements.
    """
    xd_np = xd.cpu().numpy() if xd.device.type != "cpu" else xd.numpy()
    yd_np = yd.cpu().numpy() if yd.device.type != "cpu" else yd.numpy()
    zd_np = zd.cpu().numpy() if zd.device.type != "cpu" else zd.numpy()

    # Convert from DICOM mm to NIfTI mm by negating x and y
    xd_nifti = -xd_np
    yd_nifti = -yd_np
    zd_nifti = zd_np

    # Convert NIfTI mm to voxel indices using inverse of affine 3x3
    rs = affine[:3, :3]
    rs_inv = np.linalg.inv(rs)

    disp_mm = np.stack([xd_nifti, yd_nifti, zd_nifti], axis=-1)
    disp_vox = np.einsum("ij,...j->...i", rs_inv, disp_mm)

    device = xd.device
    xd = torch.from_numpy(disp_vox[..., 0].copy()).to(device)
    yd = torch.from_numpy(disp_vox[..., 1].copy()).to(device)
    zd = torch.from_numpy(disp_vox[..., 2].copy()).to(device)

    return xd, yd, zd


def parse_nwarp_string(nwarp_str: str) -> list[str]:
    """Parse -nwarp argument into list of file paths.

    Args:
        nwarp_str: Space-separated list of warp/matrix files

    Returns:
        List of file paths
    """
    return nwarp_str.strip().split()


def identify_transform_type(path: str) -> str:
    """Identify transform type from file extension.

    Args:
        path: File path

    Returns:
        'affine' or 'nonlinear'
    """
    path_lower = path.lower()
    if path_lower.endswith(".1d") or path_lower.endswith(".txt"):
        return "affine"
    elif path_lower.endswith(".nii") or path_lower.endswith(".nii.gz"):
        return "nonlinear"
    else:
        raise ValueError(f"Cannot identify transform type for: {path}")


def resample_warp_to_grid(
    warp: NonlinearWarp,
    target_shape: tuple[int, int, int],
    target_affine: np.ndarray,
    device: torch.device,
    verb: int = 1,
) -> NonlinearWarp:
    """Resample a warp field to a different grid using trilinear interpolation.

    Handles the critical case where warp and target grids have different axis
    orderings (e.g., permuted j↔k). The displacement vectors are transformed
    from warp-voxel space to target-voxel space during resampling.

    Steps:
      1. Map target voxel coords → warp voxel coords (for sampling locations)
      2. Interpolate displacement values at those locations (in warp-voxel units)
      3. Rotate displacement vectors: warp-voxel → target-voxel space

    Args:
        warp: Input warp field
        target_shape: (nz, ny, nx) target dimensions
        target_affine: Target affine matrix
        device: Torch device
        verb: Verbosity level

    Returns:
        Warp resampled to target grid, with displacements in target-voxel units
    """
    src_shape = tuple(warp.xd.shape)
    src_affine = warp.header_info["affine"].astype(np.float64)
    tgt_affine = target_affine.astype(np.float64)

    # Check if shapes and affines match — skip if identical grid
    if src_shape == target_shape and np.allclose(src_affine, tgt_affine, atol=1e-4):
        if verb >= 2:
            print(f"  [resample] grids match {src_shape}, skipping resample")
        return warp

    if verb >= 2:
        print(f"  [resample] {src_shape} -> {target_shape}")

    src_nz, src_ny, src_nx = src_shape
    tgt_nz, tgt_ny, tgt_nx = target_shape

    # Step 1: mapping target voxels → warp voxels for sampling
    src_xyz2ijk = np.linalg.inv(src_affine)
    M = (src_xyz2ijk @ tgt_affine).astype(np.float32)

    kk, jj, ii = torch.meshgrid(
        torch.arange(tgt_nz, dtype=torch.float32, device=device),
        torch.arange(tgt_ny, dtype=torch.float32, device=device),
        torch.arange(tgt_nx, dtype=torch.float32, device=device),
        indexing="ij",
    )

    coords = torch.stack(
        [
            ii.reshape(-1),
            jj.reshape(-1),
            kk.reshape(-1),
            torch.ones(tgt_nz * tgt_ny * tgt_nx, dtype=torch.float32, device=device),
        ],
        dim=0,
    )

    M_t = torch.from_numpy(M).float().to(device)
    src_coords = M_t @ coords

    src_x = src_coords[0].reshape(tgt_nz, tgt_ny, tgt_nx)
    src_y = src_coords[1].reshape(tgt_nz, tgt_ny, tgt_nx)
    src_z = src_coords[2].reshape(tgt_nz, tgt_ny, tgt_nx)

    # Step 2: interpolate displacement components at source locations
    flat_x = src_x.reshape(-1)
    flat_y = src_y.reshape(-1)
    flat_z = src_z.reshape(-1)
    new_xd = trilinear_interpolate(warp.xd, flat_x, flat_y, flat_z).float().reshape(tgt_nz, tgt_ny, tgt_nx)
    new_yd = trilinear_interpolate(warp.yd, flat_x, flat_y, flat_z).float().reshape(tgt_nz, tgt_ny, tgt_nx)
    new_zd = trilinear_interpolate(warp.zd, flat_x, flat_y, flat_z).float().reshape(tgt_nz, tgt_ny, tgt_nx)

    # Step 3: rotate displacement vectors from warp-voxel space to target-voxel space
    # Displacement in warp-voxel → mm: R_warp @ d_warp_vox
    # mm → target-voxel: R_tgt_inv @ mm_disp
    # Combined: R_tgt_inv @ R_warp @ d_warp_vox
    src_R = src_affine[:3, :3]
    tgt_R_inv = np.linalg.inv(tgt_affine[:3, :3])
    disp_xform = (tgt_R_inv @ src_R).astype(np.float32)

    if not np.allclose(disp_xform, np.eye(3), atol=1e-4):
        if verb >= 2:
            print("  [resample] rotating displacement vectors (grids have different axes)")
        D = torch.from_numpy(disp_xform).float().to(device)
        # Stack displacements and rotate
        disp = torch.stack([new_xd.reshape(-1), new_yd.reshape(-1), new_zd.reshape(-1)], dim=0).float()
        disp_rot = D @ disp
        new_xd = disp_rot[0].reshape(tgt_nz, tgt_ny, tgt_nx)
        new_yd = disp_rot[1].reshape(tgt_nz, tgt_ny, tgt_nx)
        new_zd = disp_rot[2].reshape(tgt_nz, tgt_ny, tgt_nx)

    new_header = {
        "affine": target_affine.copy(),
        "header": warp.header_info.get("header"),
    }

    result = NonlinearWarp(xd=new_xd, yd=new_yd, zd=new_zd, header_info=new_header)
    if verb >= 2:
        print(f"  [resample] result shape = {result.shape}")
    return result


def compose_warp_then_matrix(warp: NonlinearWarp, matrix: Tensor) -> NonlinearWarp:
    """Compose: output = matrix @ (x + warp(x))

    The matrix is applied to the warped position directly.
    Result: new_displacement = matrix @ (x + old_displacement) - x

    Args:
        warp: Displacement field
        matrix: (4, 4) affine matrix in voxel space

    Returns:
        Composed warp
    """
    nz, ny, nx = warp.shape
    device = warp.xd.device

    kk, jj, ii = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32, device=device),
        torch.arange(ny, dtype=torch.float32, device=device),
        torch.arange(nx, dtype=torch.float32, device=device),
        indexing="ij",
    )

    src_x = ii + warp.xd
    src_y = jj + warp.yd
    src_z = kk + warp.zd

    coords = torch.stack(
        [
            src_x.reshape(-1),
            src_y.reshape(-1),
            src_z.reshape(-1),
            torch.ones(nz * ny * nx, device=device),
        ],
        dim=0,
    )

    transformed = matrix @ coords

    new_xd = transformed[0].reshape(nz, ny, nx) - ii
    new_yd = transformed[1].reshape(nz, ny, nx) - jj
    new_zd = transformed[2].reshape(nz, ny, nx) - kk

    return NonlinearWarp(
        xd=new_xd,
        yd=new_yd,
        zd=new_zd,
        header_info=warp.header_info,
    )


def compose_matrix_then_warp(
    matrix: Tensor,
    warp: NonlinearWarp,
    device: torch.device,
) -> NonlinearWarp:
    """Compose: output = warp(matrix @ x)

    First apply matrix, then sample warp at transformed position.
    Result: new_displacement = warp(matrix @ x) + matrix @ x - x

    Args:
        matrix: (4, 4) affine matrix in voxel space
        warp: Displacement field
        device: Torch device

    Returns:
        Composed warp
    """
    nz, ny, nx = warp.shape

    kk, jj, ii = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32, device=device),
        torch.arange(ny, dtype=torch.float32, device=device),
        torch.arange(nx, dtype=torch.float32, device=device),
        indexing="ij",
    )

    coords = torch.stack(
        [
            ii.reshape(-1),
            jj.reshape(-1),
            kk.reshape(-1),
            torch.ones(nz * ny * nx, device=device),
        ],
        dim=0,
    )

    transformed = matrix @ coords

    tx = transformed[0].reshape(nz, ny, nx)
    ty = transformed[1].reshape(nz, ny, nx)
    tz = transformed[2].reshape(nz, ny, nx)

    _N = nz * ny * nx
    warp_x_at_t = trilinear_interpolate(
        warp.xd, tx.reshape(-1), ty.reshape(-1), tz.reshape(-1)
    ).reshape(nz, ny, nx)
    warp_y_at_t = trilinear_interpolate(
        warp.yd, tx.reshape(-1), ty.reshape(-1), tz.reshape(-1)
    ).reshape(nz, ny, nx)
    warp_z_at_t = trilinear_interpolate(
        warp.zd, tx.reshape(-1), ty.reshape(-1), tz.reshape(-1)
    ).reshape(nz, ny, nx)

    new_xd = (tx + warp_x_at_t) - ii
    new_yd = (ty + warp_y_at_t) - jj
    new_zd = (tz + warp_z_at_t) - kk

    return NonlinearWarp(
        xd=new_xd,
        yd=new_yd,
        zd=new_zd,
        header_info=warp.header_info,
    )


def compose_warp_then_warp(
    warp_a: NonlinearWarp,
    warp_b: NonlinearWarp,
) -> NonlinearWarp:
    """Compose: output = B(A(x))

    where A(x) = x + a(x), B(x) = x + b(x)
    Result: C(x) = x + a(x) + b(x + a(x))

    Args:
        warp_a: First warp to apply
        warp_b: Second warp to apply

    Returns:
        Composed warp
    """
    nz, ny, nx = warp_a.shape
    device = warp_a.xd.device

    kk, jj, ii = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32, device=device),
        torch.arange(ny, dtype=torch.float32, device=device),
        torch.arange(nx, dtype=torch.float32, device=device),
        indexing="ij",
    )

    x_after_a = ii + warp_a.xd
    y_after_a = jj + warp_a.yd
    z_after_a = kk + warp_a.zd

    _N = nz * ny * nx
    bx_at_a = trilinear_interpolate(
        warp_b.xd, x_after_a.reshape(-1), y_after_a.reshape(-1), z_after_a.reshape(-1)
    ).reshape(nz, ny, nx)
    by_at_a = trilinear_interpolate(
        warp_b.yd, x_after_a.reshape(-1), y_after_a.reshape(-1), z_after_a.reshape(-1)
    ).reshape(nz, ny, nx)
    bz_at_a = trilinear_interpolate(
        warp_b.zd, x_after_a.reshape(-1), y_after_a.reshape(-1), z_after_a.reshape(-1)
    ).reshape(nz, ny, nx)

    new_xd = warp_a.xd + bx_at_a
    new_yd = warp_a.yd + by_at_a
    new_zd = warp_a.zd + bz_at_a

    return NonlinearWarp(
        xd=new_xd,
        yd=new_yd,
        zd=new_zd,
        header_info=warp_a.header_info,
    )


def compose_chain(
    transforms: list[Transform],
    output_shape: tuple[int, int, int],
    output_affine: np.ndarray,
    device: torch.device,
    time_idx: int = 0,
    verb: int = 1,
) -> NonlinearWarp:
    """Compose a chain of transforms: C(B(A(x))).

    Args:
        transforms: List of AffineTransform or NonlinearWarp (in order A, B, C)
        output_shape: (nz, ny, nx) of output grid
        output_affine: Affine matrix for output grid
        device: Torch device
        time_idx: Which time point for time-dependent affines
        verb: Verbosity level

    Returns:
        Single composed warp field on output grid
    """
    if not transforms:
        raise ValueError("Empty transform chain")

    result_warp: NonlinearWarp | None = None

    for i, xform in enumerate(transforms):
        if isinstance(xform, AffineTransform):
            mat = xform.at_time(time_idx)

            if result_warp is None:
                result_warp = NonlinearWarp(
                    xd=torch.zeros(output_shape, dtype=torch.float32, device=device),
                    yd=torch.zeros(output_shape, dtype=torch.float32, device=device),
                    zd=torch.zeros(output_shape, dtype=torch.float32, device=device),
                    header_info={"affine": output_affine.copy()},
                )
                if verb >= 2:
                    print(f"  [compose] created identity warp {output_shape}")

            result_warp = compose_warp_then_matrix(result_warp, mat)
            if verb >= 2:
                print(
                    f"  [compose] after affine [{i}]: warp shape = {result_warp.shape}"
                )

        elif isinstance(xform, NonlinearWarp):
            if result_warp is None:
                resampled = resample_warp_to_grid(
                    xform, output_shape, output_affine, device, verb=verb
                )
                result_warp = resampled
                if verb >= 2:
                    print(
                        f"  [compose] first warp [{i}]: {xform.shape} -> resampled to {result_warp.shape}"
                    )
            else:
                if verb >= 2:
                    print(
                        f"  [compose] resampling warp [{i}] from {xform.shape} to {output_shape}"
                    )
                resampled = resample_warp_to_grid(
                    xform, output_shape, output_affine, device, verb=verb
                )
                if verb >= 2:
                    print(f"  [compose] resampled shape = {resampled.shape}")
                result_warp = compose_warp_then_warp(result_warp, resampled)
                if verb >= 2:
                    print(
                        f"  [compose] after compose: warp shape = {result_warp.shape}"
                    )

    if result_warp is None:
        result_warp = NonlinearWarp(
            xd=torch.zeros(output_shape, dtype=torch.float32, device=device),
            yd=torch.zeros(output_shape, dtype=torch.float32, device=device),
            zd=torch.zeros(output_shape, dtype=torch.float32, device=device),
            header_info={"affine": output_affine.copy()},
        )

    return result_warp


def apply_composed_warp(
    source: Tensor,
    warp: NonlinearWarp,
    source_affine: np.ndarray,
    output_affine: np.ndarray,
    interp: str = "wsinc5",
) -> Tensor:
    """Apply composed warp to source volume.

    The composed warp has displacements in output-voxel space. To sample the
    source image, we need to convert the sampling coordinates from output-voxel
    to source-voxel space.

    For each output voxel (i,j,k):
      1. Compute output-space coordinate: (i + xd, j + yd, k + zd)
      2. Convert to mm: output_affine @ coord
      3. Convert to source voxels: inv(source_affine) @ mm_coord
      4. Sample source at that location

    Args:
        source: (nz, ny, nx) source image
        warp: Composed displacement field in output-voxel units
        source_affine: NIfTI affine of source image
        output_affine: NIfTI affine of output grid
        interp: Interpolation method ("linear" or "wsinc5")

    Returns:
        Warped image on output grid
    """
    nz, ny, nx = warp.shape
    device = source.device

    # Build coordinate grids
    kk, jj, ii = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32, device=device),
        torch.arange(ny, dtype=torch.float32, device=device),
        torch.arange(nx, dtype=torch.float32, device=device),
        indexing="ij",
    )

    # Output-space coordinates after warp
    out_x = ii + warp.xd
    out_y = jj + warp.yd
    out_z = kk + warp.zd

    # Convert output-voxel → source-voxel
    # M = inv(source_affine) @ output_affine
    M = np.linalg.inv(source_affine.astype(np.float64)) @ output_affine.astype(np.float64)
    M = M.astype(np.float32)
    M_t = torch.from_numpy(M).float().to(device)

    N = nz * ny * nx
    coords = torch.stack([
        out_x.reshape(-1),
        out_y.reshape(-1),
        out_z.reshape(-1),
        torch.ones(N, dtype=torch.float32, device=device),
    ], dim=0)

    src_coords = M_t @ coords
    src_xd = src_coords[0].reshape(nz, ny, nx) - ii
    src_yd = src_coords[1].reshape(nz, ny, nx) - jj
    src_zd = src_coords[2].reshape(nz, ny, nx) - kk

    if interp == "wsinc5":
        return warp_image_wsinc5(source, src_xd, src_yd, src_zd)
    else:
        from .interp import warp_image_linear

        return warp_image_linear(source, src_xd, src_yd, src_zd)


def nwarpforge(
    source_path: str,
    nwarp_specs: list[str],
    prefix: str,
    master_path: str | None = None,
    interp: str = "wsinc5",
    device: torch.device | None = None,
    verb: int = 1,
    time_range: tuple[int, int] | None = None,
    debug: bool = False,
    save_mean: bool = False,
) -> None:
    """Main pipeline: compose warps and apply to source.

    Args:
        source_path: Path to source dataset (3D or 4D)
        nwarp_specs: List of warp/matrix file paths
        prefix: Output path
        master_path: Path to master dataset for output grid (optional)
        interp: Final interpolation method
        device: Torch device
        verb: Verbosity level
        time_range: If set, only process volumes in range [start, end)
        debug: Print detailed matrix/warp debug info
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0_load = __import__("time").time()
    source, source_header = load_image(source_path, device=device)
    if verb >= 1:
        print(
            f"Loaded source: {source_path} {source.shape} ({__import__('time').time() - t0_load:.2f}s)"
        )

    is_4d = source.ndim == 4
    nt = source.shape[0] if is_4d else 1

    if master_path is not None:
        master, master_header = load_image(master_path, device=device)
        if master.ndim == 4:
            master = master[0]
        output_shape = tuple(master.shape)
        output_affine = master_header["affine"]
        del master
    else:
        output_shape = tuple(source.shape[-3:]) if is_4d else tuple(source.shape)
        output_affine = source_header["affine"]

    # Matrix conversion: AFNI uses `actual_cmat` from the first nonlinear warp's
    # dataset for converting DICOM matrices to voxel space. However, since we
    # resample ALL warps to the output grid before composition, the composed
    # warp is in the output grid's voxel space. Therefore matrices must also
    # be converted to the output grid's voxel space for consistency.
    #
    # (AFNI keeps everything in the warp's voxel space and only resamples
    # the final result. We resample warps first, so we must use output_affine.)
    affine_for_matrices = output_affine
    if debug:
        a = affine_for_matrices
        print("[DEBUG] Affine for matrix conversion (output grid):")
        print(f"  [{a[0, 0]:9.5f} {a[0, 1]:9.5f} {a[0, 2]:9.5f} {a[0, 3]:9.5f}]")
        print(f"  [{a[1, 0]:9.5f} {a[1, 1]:9.5f} {a[1, 2]:9.5f} {a[1, 3]:9.5f}]")
        print(f"  [{a[2, 0]:9.5f} {a[2, 1]:9.5f} {a[2, 2]:9.5f} {a[2, 3]:9.5f}]")

    transforms: list[Transform] = []
    max_time_points = 1

    for _i_spec, spec in enumerate(nwarp_specs):
        kind = identify_transform_type(spec)
        if verb >= 1:
            print(f"Loading {kind}: {spec}")

        if kind == "affine":
            xform = load_affine_1D(
                spec, affine_for_matrices, device=device, debug=debug
            )
            transforms.append(xform)
            max_time_points = max(max_time_points, xform.matrices.shape[0])
            if verb >= 1:
                print(f"  Affine: {xform.matrices.shape[0]} time point(s)")
            if debug:
                m = xform.matrices[0].cpu().numpy()
                print("  [DEBUG] Matrix after conversion (t=0):")
                print(
                    f"    [{m[0, 0]:9.5f} {m[0, 1]:9.5f} {m[0, 2]:9.5f} {m[0, 3]:9.5f}]"
                )
                print(
                    f"    [{m[1, 0]:9.5f} {m[1, 1]:9.5f} {m[1, 2]:9.5f} {m[1, 3]:9.5f}]"
                )
                print(
                    f"    [{m[2, 0]:9.5f} {m[2, 1]:9.5f} {m[2, 2]:9.5f} {m[2, 3]:9.5f}]"
                )
                print(
                    f"    [{m[3, 0]:9.5f} {m[3, 1]:9.5f} {m[3, 2]:9.5f} {m[3, 3]:9.5f}]"
                )
        else:
            xform = load_warp(spec, device=device, debug=debug)
            transforms.append(xform)
            if verb >= 1:
                print(f"  Warp: {xform.shape}")
            if debug:
                print("  [DEBUG] Warp displacement ranges:")
                print(
                    f"    xd: [{xform.xd.min().item():.3f}, {xform.xd.max().item():.3f}]"
                )
                print(
                    f"    yd: [{xform.yd.min().item():.3f}, {xform.yd.max().item():.3f}]"
                )
                print(
                    f"    zd: [{xform.zd.min().item():.3f}, {xform.zd.max().item():.3f}]"
                )

    if is_4d and max_time_points > 1:
        nt = max(nt, max_time_points)

    # Apply time_range filter if specified
    if time_range is not None:
        t_start, t_end = time_range
        t_end = min(t_end, nt)  # Don't exceed available volumes
        if verb >= 1:
            print(f"Processing time points {t_start} to {t_end - 1} of {nt}")
    else:
        t_start, t_end = 0, nt

    output_volumes = []

    time_iter = tqdm(
        range(t_start, t_end),
        desc="Warping volumes",
        disable=verb == 0 or (t_end - t_start) == 1,
    )
    for t in time_iter:
        composed = compose_chain(
            transforms,
            output_shape,
            output_affine,
            device,
            time_idx=t,
            verb=0 if (t_end - t_start) > 1 else verb,
        )

        src_vol = source[t] if is_4d else source

        warped = apply_composed_warp(
            src_vol, composed,
            source_affine=source_header["affine"],
            output_affine=output_affine,
            interp=interp,
        )
        output_volumes.append(warped)

    if is_4d or len(output_volumes) > 1:
        output = torch.stack(output_volumes)
    else:
        output = output_volumes[0]

    output_header = {"affine": output_affine, "header": source_header.get("header")}
    save_image(output, prefix, header_info=output_header)

    if verb >= 1:
        print(f"Saved: {prefix}")

    if save_mean:
        if output.ndim == 4:
            mean_path = derive_mean_output_path(prefix)
            mean_image = output.mean(dim=0)
            save_image(mean_image, mean_path, header_info=output_header)
            if verb >= 1:
                print(f"Saved mean: {mean_path}")
        elif verb >= 1:
            print("-save_mean requested, but output is not 4D; skipping mean output")
