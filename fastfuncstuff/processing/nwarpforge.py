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

from ..io.afni import get_afni_space_info, set_afni_space_info


def compute_cardinal_affine(oblique_aff: np.ndarray) -> np.ndarray:
    """Compute cardinal (deobliqued) affine matching AFNI's ijk_to_dicom.

    When AFNI reads a NIfTI file, it stores the raw sform/qform as
    ``ijk_to_dicom_real`` (oblique) and computes a cardinal version as
    ``ijk_to_dicom`` where each voxel axis is aligned to its closest
    cardinal (x/y/z) direction.  All matrix conversions in AFNI use
    ``ijk_to_dicom`` (the cardinal version), so we must do the same.

    For each column of the 3×3 rotation/scale block:
      1. Find the dominant (largest-magnitude) row
      2. Set that entry to ±voxel_size, zero all others

    The origin (translation) is preserved.
    """
    R = oblique_aff[:3, :3]
    cardinal = np.zeros((4, 4), dtype=np.float64)
    cardinal[3, 3] = 1.0
    cardinal[:3, 3] = oblique_aff[:3, 3]

    for col in range(3):
        vec = R[:, col]
        voxel_size = np.sqrt(np.sum(vec ** 2))
        dominant = int(np.argmax(np.abs(vec)))
        sign = np.sign(vec[dominant])
        cardinal[dominant, col] = sign * voxel_size

    return cardinal


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
    """Displacement field (voxel units after prepare_warp, NIfTI mm after load)."""

    xd: Tensor  # (nz, ny, nx)
    yd: Tensor
    zd: Tensor
    header_info: dict
    units: str = "voxels"  # "voxels" or "nifti_mm"

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
    #
    # AFNI uses CARDINAL (deobliqued) coordinate matrices for this conversion.
    # When AFNI reads a NIfTI file, it stores the oblique sform as
    # ijk_to_dicom_real and computes a cardinal (axis-aligned) version as
    # ijk_to_dicom.  All matrix conversions use ijk_to_dicom (cardinal).
    #
    # AFNI_DICOM = flip @ NIfTI_RAS where flip = diag(-1,-1,1,1)
    # M_nifti = flip @ M_dicom @ flip
    # M_index = inv(cardinal_affine) @ M_nifti @ cardinal_affine
    cardinal = compute_cardinal_affine(output_affine)
    flip = torch.tensor(
        [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
    )
    output_ijk2xyz = torch.from_numpy(cardinal.astype(np.float32)).float()
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

    AFNI warps store displacements in DICOM mm.  We convert to NIfTI mm
    (negate x, y) but do NOT convert to voxels here.  The voxel conversion
    happens later in ``prepare_warp_for_grid`` using the *output* grid's
    affine, matching AFNI's ``THD_setup_nwarp`` approach.

    Args:
        path: Path to warp file
        device: Torch device
        units: "mm" (convert DICOM→NIfTI mm, default), "voxels" (as-is)
        debug: Print debug info

    Returns:
        NonlinearWarp with displacements in NIfTI mm (units="mm") or
        voxel units (units="voxels")
    """
    xd, yd, zd, header_info = load_warp_field(path, device=device)

    if debug:
        print("[DEBUG] load_warp: Raw displacement ranges (DICOM mm):")
        print(f"  xd: [{xd.min().item():.3f}, {xd.max().item():.3f}]")
        print(f"  yd: [{yd.min().item():.3f}, {yd.max().item():.3f}]")
        print(f"  zd: [{zd.min().item():.3f}, {zd.max().item():.3f}]")
        a = header_info["affine"]
        print("[DEBUG] load_warp: Warp file affine:")
        print(f"  [{a[0, 0]:9.5f} {a[0, 1]:9.5f} {a[0, 2]:9.5f} {a[0, 3]:9.5f}]")
        print(f"  [{a[1, 0]:9.5f} {a[1, 1]:9.5f} {a[1, 2]:9.5f} {a[1, 3]:9.5f}]")
        print(f"  [{a[2, 0]:9.5f} {a[2, 1]:9.5f} {a[2, 2]:9.5f} {a[2, 3]:9.5f}]")

    if units == "mm":
        # DICOM mm → NIfTI mm: just negate x and y
        xd = -xd
        yd = -yd
        warp_units = "nifti_mm"
        if debug:
            print("[DEBUG] load_warp: After DICOM→NIfTI mm conversion:")
            print(f"  xd: [{xd.min().item():.3f}, {xd.max().item():.3f}]")
            print(f"  yd: [{yd.min().item():.3f}, {yd.max().item():.3f}]")
            print(f"  zd: [{zd.min().item():.3f}, {zd.max().item():.3f}]")
    else:
        warp_units = "voxels"
        if debug:
            print("[DEBUG] load_warp: Warps assumed to be in voxel units")

    return NonlinearWarp(
        xd=xd, yd=yd, zd=zd, header_info=header_info, units=warp_units
    )


def _nifti_mm_to_voxels(
    xd: Tensor, yd: Tensor, zd: Tensor, affine: np.ndarray
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert NIfTI mm displacements to voxel-index displacements.

    Uses the inverse of the affine's 3x3 rotation/scale to convert
    displacement vectors from NIfTI mm to the affine's voxel space.
    """
    rs_inv = np.linalg.inv(affine[:3, :3].astype(np.float64)).astype(np.float32)
    R = torch.from_numpy(rs_inv).to(xd.device)

    # Apply R to displacement vectors: disp_vox = R @ disp_mm
    vx = R[0, 0] * xd + R[0, 1] * yd + R[0, 2] * zd
    vy = R[1, 0] * xd + R[1, 1] * yd + R[1, 2] * zd
    vz = R[2, 0] * xd + R[2, 1] * yd + R[2, 2] * zd

    return vx, vy, vz


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


def prepare_warp_for_grid(
    warp: NonlinearWarp,
    target_shape: tuple[int, int, int],
    target_affine: np.ndarray,
    device: torch.device,
    verb: int = 1,
) -> NonlinearWarp:
    """Resample a NIfTI-mm warp to the target grid and convert to voxel units.

    Follows AFNI's ``THD_setup_nwarp`` approach:
      1. Map target voxel coords → warp voxel coords (for interpolation)
      2. Interpolate NIfTI mm displacement values at those locations
      3. Convert NIfTI mm → target-voxel units using inv(target_R)

    This avoids the error-prone displacement-vector rotation by converting
    directly from mm to the target grid's voxel space.

    Args:
        warp: Input warp with displacements in NIfTI mm (units="nifti_mm")
        target_shape: (nz, ny, nx) target dimensions
        target_affine: Target affine matrix
        device: Torch device
        verb: Verbosity level

    Returns:
        Warp on target grid with displacements in target-voxel units
    """
    assert warp.units == "nifti_mm", (
        f"prepare_warp_for_grid expects NIfTI mm warp, got units={warp.units}"
    )

    src_shape = tuple(warp.xd.shape)
    src_affine = warp.header_info["affine"].astype(np.float64)
    tgt_affine = target_affine.astype(np.float64)

    # Use cardinal affines for grid mapping (matching AFNI)
    src_cardinal = compute_cardinal_affine(src_affine)
    tgt_cardinal = compute_cardinal_affine(tgt_affine)

    tgt_nz, tgt_ny, tgt_nx = target_shape

    # Check if grids match — skip resampling if so
    needs_resample = not (
        src_shape == target_shape
        and np.allclose(src_cardinal, tgt_cardinal, atol=1e-4)
    )

    if needs_resample:
        if verb >= 2:
            print(f"  [prepare_warp] resampling {src_shape} -> {target_shape}")

        # Map target voxels → warp voxels for interpolation
        src_xyz2ijk = np.linalg.inv(src_cardinal)
        M = (src_xyz2ijk @ tgt_cardinal).astype(np.float32)

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
                torch.ones(
                    tgt_nz * tgt_ny * tgt_nx,
                    dtype=torch.float32,
                    device=device,
                ),
            ],
            dim=0,
        )

        M_t = torch.from_numpy(M).float().to(device)
        src_coords = M_t @ coords

        flat_x = src_coords[0]
        flat_y = src_coords[1]
        flat_z = src_coords[2]

        # Interpolate NIfTI mm displacements at warp grid locations
        mm_xd = trilinear_interpolate(warp.xd, flat_x, flat_y, flat_z).float().reshape(tgt_nz, tgt_ny, tgt_nx)
        mm_yd = trilinear_interpolate(warp.yd, flat_x, flat_y, flat_z).float().reshape(tgt_nz, tgt_ny, tgt_nx)
        mm_zd = trilinear_interpolate(warp.zd, flat_x, flat_y, flat_z).float().reshape(tgt_nz, tgt_ny, tgt_nx)
    else:
        if verb >= 2:
            print(f"  [prepare_warp] grids match {src_shape}, no resample needed")
        mm_xd, mm_yd, mm_zd = warp.xd, warp.yd, warp.zd

    # Convert NIfTI mm → target-voxel units using CARDINAL affine
    # (AFNI uses cardinal coordinate matrices for all conversions)
    tgt_cardinal = compute_cardinal_affine(target_affine)
    vx, vy, vz = _nifti_mm_to_voxels(mm_xd, mm_yd, mm_zd, tgt_cardinal)

    new_header = {
        "affine": target_affine.copy(),
        "header": warp.header_info.get("header"),
    }

    result = NonlinearWarp(
        xd=vx, yd=vy, zd=vz, header_info=new_header, units="voxels"
    )
    if verb >= 2:
        print(f"  [prepare_warp] result shape = {result.shape}")
        print(f"  [prepare_warp] voxel disp ranges: "
              f"x=[{vx.min().item():.2f}, {vx.max().item():.2f}] "
              f"y=[{vy.min().item():.2f}, {vy.max().item():.2f}] "
              f"z=[{vz.min().item():.2f}, {vz.max().item():.2f}]")
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
            # Convert NIfTI mm warp to output-grid voxel units
            prepared = prepare_warp_for_grid(
                xform, output_shape, output_affine, device, verb=verb
            )
            if result_warp is None:
                result_warp = prepared
                if verb >= 2:
                    print(
                        f"  [compose] first warp [{i}]: {xform.shape} -> {result_warp.shape}"
                    )
            else:
                if verb >= 2:
                    print(
                        f"  [compose] composing warp [{i}] ({xform.shape} -> {output_shape})"
                    )
                result_warp = compose_warp_then_warp(result_warp, prepared)
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

    # Convert output-voxel → source-voxel using CARDINAL affines.
    # AFNI uses cardinal (deobliqued) coordinate matrices for all
    # index-to-coordinate conversions, so we must do the same.
    src_card = compute_cardinal_affine(source_affine)
    out_card = compute_cardinal_affine(output_affine)
    M = np.linalg.inv(src_card) @ out_card
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


def _regrid_to_dxyz(
    output_shape: tuple[int, ...],
    output_affine: np.ndarray,
    dxyz: float,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Recompute output grid for isotropic voxel size while preserving FOV.

    Takes the existing output grid (shape + affine) and returns a new grid
    whose voxels are ``dxyz`` mm isotropic, covering the same bounding box.

    Origin is adjusted to preserve the FOV center, matching AFNI's
    ``r_dxyz_mod_dataxes()`` behavior.
    """
    # Current voxel sizes from the affine columns
    old_voxel_sizes = np.sqrt((output_affine[:3, :3] ** 2).sum(axis=0))

    # output_shape is (nz, ny, nx) = (nk, nj, ni) but affine columns are
    # (i, j, k) order.  Convert shape to affine axis order.
    shape_ijk = np.array([output_shape[2], output_shape[1], output_shape[0]])

    # FOV in mm along each voxel axis
    fov = old_voxel_sizes * shape_ijk

    # New grid dimensions in (i, j, k) order
    new_ijk = tuple(int(f / dxyz + 0.499) for f in fov)
    # Convert back to internal (nz, ny, nx) = (nk, nj, ni)
    new_shape = (new_ijk[2], new_ijk[1], new_ijk[0])

    # New affine: same orientation (unit vectors), scaled by new voxel size
    direction = output_affine[:3, :3] / old_voxel_sizes[np.newaxis, :]
    old_R = output_affine[:3, :3]
    new_R = direction * dxyz

    new_affine = np.eye(4)
    new_affine[:3, :3] = new_R

    # Adjust origin to preserve FOV center (matches AFNI r_dxyz_mod_dataxes)
    # Uses (i,j,k) ordered shapes since affine columns are (i,j,k)
    old_center_offset = old_R @ (shape_ijk - 1.0) * 0.5
    new_center_offset = new_R @ (np.array(new_ijk) - 1.0) * 0.5
    new_affine[:3, 3] = output_affine[:3, 3] + old_center_offset - new_center_offset

    return new_shape, new_affine


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
    dxyz: float | None = None,
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
        dxyz: If set, force isotropic output voxel size (mm)
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

    master_hdr_obj = None  # nibabel header for AFNI extension propagation
    if master_path is not None:
        master, master_header = load_image(master_path, device=device)
        if master.ndim == 4:
            master = master[0]
        output_shape = tuple(master.shape)
        output_affine = compute_cardinal_affine(master_header["affine"])
        master_space_info = get_afni_space_info(master_header.get("header"))
        master_hdr_obj = master_header.get("header")
        del master
    else:
        output_shape = tuple(source.shape[-3:]) if is_4d else tuple(source.shape)
        output_affine = compute_cardinal_affine(source_header["affine"])
        master_space_info = get_afni_space_info(source_header.get("header"))

    # Apply -dxyz: recompute grid for isotropic voxel size
    if dxyz is not None:
        old_shape = output_shape
        output_shape, output_affine = _regrid_to_dxyz(output_shape, output_affine, dxyz)
        if verb >= 1:
            print(f"nwarpforge: -dxyz {dxyz} mm: {old_shape} -> {output_shape}")

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

    # If master didn't provide space info, try the first NIfTI warp in chain
    if master_hdr_obj is None:
        for xform in transforms:
            if isinstance(xform, NonlinearWarp):
                warp_hdr = xform.header_info.get("header")
                if warp_hdr is not None:
                    master_space_info = get_afni_space_info(warp_hdr)
                    master_hdr_obj = warp_hdr
                    break

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

    # Build output header: use output_affine (cardinal), don't inherit
    # source's qform/sform which would conflict with the output grid.
    # Preserve temporal metadata (TR, units) from source.
    # Propagate AFNI view/space from master (e.g. tlrc + MNI_2009c_asym).
    src_hdr = source_header.get("header")
    try:
        import nibabel as nib
        out_hdr = nib.Nifti1Header()
        if src_hdr is not None:
            # Copy temporal metadata (units always safe; zooms need valid ndim)
            out_hdr.set_xyzt_units(*src_hdr.get_xyzt_units())
            src_zooms = src_hdr.get_zooms()
            if len(src_zooms) > 3 and output.ndim == 4:
                # 4D output: set shape first so set_zooms accepts 4 values
                out_hdr.set_data_shape(output.shape)
                out_hdr.set_zooms((1.0, 1.0, 1.0, src_zooms[3]))

        # Copy AFNI extension: prefer master (has correct space),
        # fall back to source (has history).
        afni_ext_copied = False
        for hdr_candidate in (master_hdr_obj, src_hdr):
            if hdr_candidate is None:
                continue
            try:
                for ext in hdr_candidate.extensions:
                    if ext.get_code() == 4:
                        out_hdr.extensions.append(ext)
                        afni_ext_copied = True
                        break
            except AttributeError:
                pass
            if afni_ext_copied:
                break

        # Set AFNI view/space from master
        set_afni_space_info(
            out_hdr,
            view=master_space_info["view"],
            space=master_space_info["space"],
        )
    except Exception as exc:
        import warnings
        warnings.warn(f"nwarpforge: failed to build output header: {exc}", stacklevel=2)
        out_hdr = None

    output_header = {"affine": output_affine, "header": out_hdr}
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
