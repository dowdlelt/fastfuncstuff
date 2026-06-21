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

from .interp import normalize_interp_mode, trilinear_interpolate, warp_image
from .io import derive_mean_output_path, load_image, load_warp_field, save_image


def derive_phase_output_path(prefix: str) -> str:
    """Derive phase output path from magnitude prefix.

    Examples:
        out.nii.gz   -> out_phase.nii.gz
        out.nii      -> out_phase.nii
    """
    from pathlib import Path

    p = Path(prefix)
    if p.name.endswith(".nii.gz"):
        stem = p.name[: -len(".nii.gz")]
        return str(p.parent / f"{stem}_phase.nii.gz")
    elif p.name.endswith(".nii"):
        stem = p.name[: -len(".nii")]
        return str(p.parent / f"{stem}_phase.nii")
    else:
        return str(p.parent / f"{p.stem}_phase{p.suffix}")


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
        voxel_size = np.sqrt(np.sum(vec**2))
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


@dataclass
class TimeVaryingWarp:
    """Per-frame displacement field (e.g. a MEDIC frame-wise distortion warp).

    Holds one 3-vector displacement field per time point in NIfTI mm (same
    convention as a loaded ffs_qwarp warp), so each frame goes through
    :func:`prepare_warp_for_grid` and composes to any ``-master`` output grid.
    Analogous to ``AffineTransform``'s time dependence: ``at_time(t)`` selects
    the frame's :class:`NonlinearWarp`.  On disk it is either per-frame 3D files
    (a ``warp_*`` wildcard) or one 5D ``(nx, ny, nz, T, 3)`` file.
    """

    xd: Tensor  # (T, nz, ny, nx)
    yd: Tensor
    zd: Tensor
    header_info: dict
    units: str = "nifti_mm"

    @property
    def n_time(self) -> int:
        return self.xd.shape[0]

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        s = self.xd.shape
        return (int(s[1]), int(s[2]), int(s[3]))

    def at_time(self, t: int) -> NonlinearWarp:
        idx = min(t, self.xd.shape[0] - 1)
        return NonlinearWarp(
            xd=self.xd[idx],
            yd=self.yd[idx],
            zd=self.zd[idx],
            header_info=self.header_info,
            units=self.units,
        )


Transform = AffineTransform | NonlinearWarp | TimeVaryingWarp


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
            raise ValueError(f"Expected 12 values per row in {path}, got {mats_12.shape[1]}")

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

    return NonlinearWarp(xd=xd, yd=yd, zd=zd, header_info=header_info, units=warp_units)


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


def _is_time_varying_warp(path: str | Path) -> bool:
    """True if ``path`` is a 5D ``(nx, ny, nz, T, 3)`` per-frame warp (T > 1)."""
    from typing import cast

    import nibabel as nib

    shape = cast(nib.Nifti1Image, nib.load(str(path))).shape
    return len(shape) == 5 and shape[-1] == 3 and shape[3] > 1


def load_time_varying_warp(
    path: str | Path,
    device: torch.device | None = None,
    debug: bool = False,
) -> TimeVaryingWarp:
    """Load a 5D ``(nx, ny, nz, T, 3)`` per-frame warp (ffs_medic / qwarp mm).

    Displacements use the same DICOM-mm convention as ffs_qwarp warps; converted
    to NIfTI mm here (negate x/y, the per-frame analogue of :func:`load_warp`
    with ``units="mm"``) so the warp composes to any ``-master`` grid via
    :func:`prepare_warp_for_grid`.
    """
    from typing import cast

    import nibabel as nib

    img = cast(nib.Nifti1Image, nib.load(str(path)))
    data = np.asarray(img.dataobj, dtype=np.float32)  # (nx, ny, nz, T, 3)
    if data.ndim != 5 or data.shape[-1] != 3:
        raise ValueError(f"Expected 5D (nx,ny,nz,T,3) warp, got shape {data.shape}: {path}")

    # (nx, ny, nz, T, c) -> per-component (T, nz, ny, nx); negate x/y to go
    # DICOM mm -> NIfTI mm (matches load_warp units="mm").
    def _comp(c: int, sign: float) -> Tensor:
        arr = np.ascontiguousarray(data[..., c].transpose(3, 2, 1, 0)) * sign
        t = torch.from_numpy(arr)
        return t.to(device) if device is not None else t

    header_info = {"affine": img.affine.copy(), "header": img.header.copy()}
    warp = TimeVaryingWarp(
        xd=_comp(0, -1.0),
        yd=_comp(1, -1.0),
        zd=_comp(2, 1.0),
        header_info=header_info,
        units="nifti_mm",
    )
    if debug:
        print(
            f"[DEBUG] load_time_varying_warp: {warp.n_time} frames, "
            f"spatial {warp.spatial_shape}, units=nifti_mm"
        )
    return warp


def load_time_varying_warp_from_files(
    paths: list[str],
    device: torch.device | None = None,
    debug: bool = False,
) -> TimeVaryingWarp:
    """Build a per-frame warp from a sorted list of 3D ``(nx,ny,nz,3)`` files.

    Each file is one frame's displacement field in the same mm warp convention
    as ffs_qwarp (DICOM mm on disk); loaded with :func:`load_warp` exactly as a
    static warp, so per-frame distortion warps compose to any ``-master`` grid.
    ``paths`` should already be sorted into frame order.
    """
    frames = [load_warp(p, device=device, units="mm") for p in paths]
    warp = TimeVaryingWarp(
        xd=torch.stack([f.xd for f in frames]),
        yd=torch.stack([f.yd for f in frames]),
        zd=torch.stack([f.zd for f in frames]),
        header_info=frames[0].header_info,
        units="nifti_mm",
    )
    if debug:
        print(
            f"[DEBUG] load_time_varying_warp_from_files: {warp.n_time} frames "
            f"from {len(paths)} files, spatial {warp.spatial_shape}"
        )
    return warp


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
        src_shape == target_shape and np.allclose(src_cardinal, tgt_cardinal, atol=1e-4)
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
        mm_xd = (
            trilinear_interpolate(warp.xd, flat_x, flat_y, flat_z)
            .float()
            .reshape(tgt_nz, tgt_ny, tgt_nx)
        )
        mm_yd = (
            trilinear_interpolate(warp.yd, flat_x, flat_y, flat_z)
            .float()
            .reshape(tgt_nz, tgt_ny, tgt_nx)
        )
        mm_zd = (
            trilinear_interpolate(warp.zd, flat_x, flat_y, flat_z)
            .float()
            .reshape(tgt_nz, tgt_ny, tgt_nx)
        )
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

    result = NonlinearWarp(xd=vx, yd=vy, zd=vz, header_info=new_header, units="voxels")
    if verb >= 2:
        print(f"  [prepare_warp] result shape = {result.shape}")
        print(
            f"  [prepare_warp] voxel disp ranges: "
            f"x=[{vx.min().item():.2f}, {vx.max().item():.2f}] "
            f"y=[{vy.min().item():.2f}, {vy.max().item():.2f}] "
            f"z=[{vz.min().item():.2f}, {vz.max().item():.2f}]"
        )
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
                print(f"  [compose] after affine [{i}]: warp shape = {result_warp.shape}")

        elif isinstance(xform, NonlinearWarp | TimeVaryingWarp):
            # Resolve a time-varying warp to this frame's field, then convert the
            # NIfTI mm warp to output-grid voxel units (composes to any -master).
            frame_warp = xform.at_time(time_idx) if isinstance(xform, TimeVaryingWarp) else xform
            prepared = prepare_warp_for_grid(
                frame_warp, output_shape, output_affine, device, verb=verb
            )
            if result_warp is None:
                result_warp = prepared
                if verb >= 2:
                    print(f"  [compose] first warp [{i}]: {prepared.shape} -> {result_warp.shape}")
            else:
                if verb >= 2:
                    print(f"  [compose] composing warp [{i}] -> {output_shape}")
                result_warp = compose_warp_then_warp(result_warp, prepared)
                if verb >= 2:
                    print(f"  [compose] after compose: warp shape = {result_warp.shape}")

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
    no_neg: bool = False,
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
        interp: Interpolation method ("nearest"/"NN", "linear", "cubic",
            "quintic", "heptic", or "wsinc5")
        no_neg: If True, clamp the warped output at 0 to suppress negative
            ringing from wsinc5/cubic/quintic on non-negative data.

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
    coords = torch.stack(
        [
            out_x.reshape(-1),
            out_y.reshape(-1),
            out_z.reshape(-1),
            torch.ones(N, dtype=torch.float32, device=device),
        ],
        dim=0,
    )

    src_coords = M_t @ coords
    src_xd = src_coords[0].reshape(nz, ny, nx) - ii
    src_yd = src_coords[1].reshape(nz, ny, nx) - jj
    src_zd = src_coords[2].reshape(nz, ny, nx) - kk

    warped = warp_image(source, src_xd, src_yd, src_zd, mode=interp)
    if no_neg:
        warped = warped.clamp_min(0.0)
    return warped


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


def _first_nonlinear_warp_grid(
    nwarp_specs: list[str],
) -> tuple[tuple[int, int, int], np.ndarray, object]:
    """Grid (shape, cardinal affine, header) of the first nonlinear warp in the chain.

    Backs ``-master WARP/NWARP``: the output is produced on the grid on which the
    nonlinear warp is defined (which is usually the grid the warp pulls source
    data onto). Only the header is read, not the displacement data.
    """
    import glob as _glob

    import nibabel as nib

    for spec in nwarp_specs:
        s = spec
        if any(c in s for c in "*?[") and not Path(s).exists():
            matches = sorted(_glob.glob(s))
            if not matches:
                continue
            s = matches[0]
        try:
            if identify_transform_type(s) != "nonlinear":
                continue
        except ValueError:
            continue
        img = nib.load(str(s))
        shp = img.shape  # (nx, ny, nz, 3) or (nx, ny, nz, T, 3)
        nx, ny, nz = int(shp[0]), int(shp[1]), int(shp[2])
        return (nz, ny, nx), compute_cardinal_affine(np.asarray(img.affine)), img.header.copy()
    raise ValueError(
        "-master WARP/NWARP requires at least one nonlinear warp dataset in -nwarp"
    )


def _estimate_warp_padding(
    transforms: list[Transform],
    output_shape: tuple[int, ...],
    output_affine: np.ndarray,
) -> tuple[int, int, int]:
    """Upper-bound the composed warp's displacement (output voxels) per (x,y,z) axis.

    Returns ``(pad_x, pad_y, pad_z)``: how far, in output-grid voxels, the warp
    can pull data. Padding the output grid by this guarantees the warped source
    is fully captured (no data loss), at the cost of a larger grid. The bound is
    deliberately conservative -- it sums each nonlinear warp's worst-case
    displacement and adds the largest affine corner displacement -- so it may
    over-pad (memory only), never under-pad.
    """
    nz, ny, nx = output_shape[-3:]
    R = output_affine[:3, :3].astype(np.float64)
    inv_r_abs = np.abs(np.linalg.inv(R))

    # Output-grid corners in (i,j,k)=(x,y,z) index space, homogeneous.
    corners = np.array(
        [[i, j, k, 1.0] for i in (0, nx - 1) for j in (0, ny - 1) for k in (0, nz - 1)],
        dtype=np.float64,
    ).T  # (4, 8)

    bound = np.zeros(3)  # (x, y, z) voxels
    for xf in transforms:
        if isinstance(xf, AffineTransform):
            mats = xf.matrices.detach().cpu().numpy().astype(np.float64)  # (T,4,4)
            # Worst case over time rows: only one row applies per frame, so take
            # the max corner displacement rather than summing across time.
            aff_bound = np.zeros(3)
            for m in mats:
                disp = (m @ corners)[:3] - corners[:3]  # (3, 8) in voxels
                aff_bound = np.maximum(aff_bound, np.abs(disp).max(axis=1))
            bound += aff_bound
        else:  # NonlinearWarp / TimeVaryingWarp, displacements in NIfTI mm
            mm_max = np.array(
                [
                    float(xf.xd.abs().max()),
                    float(xf.yd.abs().max()),
                    float(xf.zd.abs().max()),
                ]
            )
            # mm -> |voxels| upper bound via |inv(R)| @ |disp_mm|.
            bound += inv_r_abs @ mm_max

    import math

    return tuple(int(math.ceil(b)) for b in bound)  # type: ignore[return-value]


def _pad_output_grid(
    output_shape: tuple[int, ...],
    output_affine: np.ndarray,
    pad: tuple[int, int, int],
) -> tuple[tuple[int, int, int], np.ndarray]:
    """Symmetrically grow the output grid by ``pad`` (pad_x, pad_y, pad_z) voxels.

    Existing voxels keep their world coordinates -- only the origin shifts so the
    added border surrounds the original FOV.
    """
    nz, ny, nx = output_shape[-3:]
    pad_x, pad_y, pad_z = pad
    new_shape = (nz + 2 * pad_z, ny + 2 * pad_y, nx + 2 * pad_x)
    new_affine = output_affine.copy()
    shift = output_affine[:3, :3] @ np.array([pad_x, pad_y, pad_z], dtype=np.float64)
    new_affine[:3, 3] = output_affine[:3, 3] - shift
    return new_shape, new_affine


def nwarpforge(
    source_path: str,
    nwarp_specs: list[str],
    prefix: str,
    phase_path: str | None = None,
    phase_prefix: str | None = None,
    master_path: str | None = None,
    interp: str = "wsinc5",
    phase_warp: str = "complex",
    phase_units: str = "raw",
    device: torch.device | None = None,
    verb: int = 1,
    time_range: tuple[int, int] | None = None,
    debug: bool = False,
    save_mean: bool = False,
    dxyz: float | None = None,
    no_neg: bool = False,
    auto_pad: bool = True,
    expad: int = 0,
) -> None:
    """Main pipeline: compose warps and apply to source.

    Args:
        source_path: Path to source (magnitude) dataset (3D or 4D)
        nwarp_specs: List of warp/matrix file paths
        prefix: Output path for magnitude (or the only output if no phase)
        phase_path: Optional phase dataset (any range — automatically scaled
                    to radians).  When provided, magnitude+phase are converted
                    to real+imag, each component is warped independently, then
                    converted back to magnitude+phase for output.
        phase_prefix: Output path for warped phase.  Auto-derived from prefix
                      (e.g. out.nii.gz -> out_phase.nii.gz) when not given.
        master_path: Path to master dataset for output grid (optional). The
            literal string "WARP"/"NWARP" uses the first nonlinear warp's grid
            as the output master (matching 3dNwarpApply -master WARP).
        interp: Final interpolation method ("nearest"/"NN", "linear", "cubic",
            "quintic", "heptic", or "wsinc5")
        phase_warp: How to warp phase data when -phase is given:
            "complex" (default): convert mag+phase to real/imag, warp each,
                convert back.  Magnitude is derived from warped real/imag
                (can be corrupted near phase wraps).
            "split": warp magnitude directly (clean interpolation on smooth
                signal), then warp real/imag and extract phase only.
                Magnitude is never touched by phase data.
            "direct": warp magnitude and phase independently.  Assumes phase
                is already unwrapped and has no wraps.  Fastest option.
            "circular": warp cos(phase) and sin(phase) separately (unit
                circle interpolation), then atan2 back.  Handles wraps
                without magnitude corruption.  Best for wrapped phase.
        phase_units: Units of the input phase data:
            "raw" (default): scanner units (e.g. -4096..4095 integer range).
                Automatically scaled to radians [-pi, pi].
            "rad": already in radians (e.g. unwrapped phase).  No scaling.
        device: Torch device
        verb: Verbosity level
        time_range: If set, only process volumes in range [start, end)
        debug: Print detailed matrix/warp debug info
        dxyz: If set, force isotropic output voxel size (mm)
        no_neg: Clamp warped output at 0 (suppress wsinc5/cubic ringing on
            non-negative data).
        auto_pad: Grow the output grid to encompass the warped source so a large
            translation / warp cannot push data off the edge (we never lose data
            on a warp). Costs memory proportional to the padding. Disable for
            exact master-grid output.
        expad: Extra padding voxels added on every side on top of the auto
            estimate (AFNI -expad analogue). Also forces padding when
            auto_pad is False.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    interp = normalize_interp_mode(interp)

    t0_load = __import__("time").time()
    source, source_header = load_image(source_path, device=device)
    if verb >= 1:
        print(
            f"Loaded source: {source_path} {source.shape} ({__import__('time').time() - t0_load:.2f}s)"
        )

    # --- Phase dataset handling ---
    phase_data: Tensor | None = None
    if phase_path is not None:
        phase_raw, _ = load_image(phase_path, device=device)
        if phase_raw.shape != source.shape:
            raise ValueError(
                f"-phase shape {tuple(phase_raw.shape)} does not match "
                f"-source shape {tuple(source.shape)}"
            )
        if phase_units == "raw":
            # Scale arbitrary scanner phase to radians [-pi, pi].
            # Works for any input range (raw integer, 0-4096, etc.)
            ph_min = phase_raw.min().item()
            ph_max = phase_raw.max().item()
            range_norm = ph_max - ph_min
            if range_norm == 0.0:
                raise ValueError(f"-phase data is constant ({ph_min}); cannot compute phase")
            range_center = (ph_max + ph_min) / range_norm * 0.5
            phase_data = (phase_raw / range_norm - range_center) * (2.0 * torch.pi)
            if verb >= 1:
                print(
                    f"Loaded phase: {phase_path} {phase_raw.shape}  "
                    f"raw range=[{ph_min:.3f}, {ph_max:.3f}] -> "
                    f"scaled to [{phase_data.min().item():.3f}, {phase_data.max().item():.3f}] rad"
                )
        elif phase_units == "rad":
            phase_data = phase_raw
            if verb >= 1:
                print(
                    f"Loaded phase: {phase_path} {phase_raw.shape}  "
                    f"range=[{phase_data.min().item():.3f}, {phase_data.max().item():.3f}] rad (no scaling)"
                )
        else:
            raise ValueError(f"Unknown phase_units: {phase_units!r}. Use 'raw' or 'rad'.")
        if phase_prefix is None:
            phase_prefix = derive_phase_output_path(prefix)
        if verb >= 1:
            print(f"Phase output: {phase_prefix}")

    is_4d = source.ndim == 4
    nt = source.shape[0] if is_4d else 1

    master_hdr_obj = None  # nibabel header for AFNI extension propagation
    # "-master WARP/NWARP": use the first nonlinear warp's grid. The warps
    # aren't loaded yet, so defer the grid decision until after the chain loads.
    use_warp_master = master_path is not None and master_path.upper() in ("WARP", "NWARP")
    output_shape: tuple[int, ...] | None
    output_affine: np.ndarray | None
    if use_warp_master:
        output_shape = None
        output_affine = None
        master_space_info = {"view": None, "space": None}
        if verb >= 1:
            print("nwarpforge: -master WARP -> output grid = first nonlinear warp")
    elif master_path is not None:
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

    # Resolve "-master WARP": peek the first nonlinear warp's header for the grid
    # (its affine is needed now to convert affine matrices into voxel space).
    if use_warp_master:
        output_shape, output_affine, warp_hdr = _first_nonlinear_warp_grid(nwarp_specs)
        master_hdr_obj = warp_hdr
        master_space_info = get_afni_space_info(warp_hdr)
        if verb >= 1:
            print(f"nwarpforge: -master WARP grid = {output_shape}")

    assert output_shape is not None and output_affine is not None

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
        # A glob token (e.g. 'warp_*.nii.gz') expands, sorted, to one warp per
        # frame -> a single time-varying slot in the chain. A pattern matching
        # exactly one file falls through to normal single-warp handling.
        if any(c in spec for c in "*?[") and not Path(spec).exists():
            import glob as _glob

            matches = sorted(_glob.glob(spec))
            if not matches:
                raise FileNotFoundError(f"-nwarp pattern matched no files: {spec}")
            if len(matches) > 1:
                tv = load_time_varying_warp_from_files(matches, device=device, debug=debug)
                transforms.append(tv)
                max_time_points = max(max_time_points, tv.n_time)
                if verb >= 1:
                    print(
                        f"Loading time-varying warp: {tv.n_time} frames from "
                        f"{len(matches)} files matching {spec}"
                    )
                continue
            spec = matches[0]

        kind = identify_transform_type(spec)
        if verb >= 1:
            print(f"Loading {kind}: {spec}")

        if kind == "affine":
            xform = load_affine_1D(spec, affine_for_matrices, device=device, debug=debug)
            transforms.append(xform)
            max_time_points = max(max_time_points, xform.matrices.shape[0])
            if verb >= 1:
                print(f"  Affine: {xform.matrices.shape[0]} time point(s)")
            if debug:
                m = xform.matrices[0].cpu().numpy()
                print("  [DEBUG] Matrix after conversion (t=0):")
                print(f"    [{m[0, 0]:9.5f} {m[0, 1]:9.5f} {m[0, 2]:9.5f} {m[0, 3]:9.5f}]")
                print(f"    [{m[1, 0]:9.5f} {m[1, 1]:9.5f} {m[1, 2]:9.5f} {m[1, 3]:9.5f}]")
                print(f"    [{m[2, 0]:9.5f} {m[2, 1]:9.5f} {m[2, 2]:9.5f} {m[2, 3]:9.5f}]")
                print(f"    [{m[3, 0]:9.5f} {m[3, 1]:9.5f} {m[3, 2]:9.5f} {m[3, 3]:9.5f}]")
        elif _is_time_varying_warp(spec):
            tv = load_time_varying_warp(spec, device=device, debug=debug)
            transforms.append(tv)
            max_time_points = max(max_time_points, tv.n_time)
            if verb >= 1:
                print(f"  Time-varying warp: {tv.n_time} frames, spatial {tv.spatial_shape}")
        else:
            xform = load_warp(spec, device=device, debug=debug)
            transforms.append(xform)
            if verb >= 1:
                print(f"  Warp: {xform.shape}")
            if debug:
                print("  [DEBUG] Warp displacement ranges:")
                print(f"    xd: [{xform.xd.min().item():.3f}, {xform.xd.max().item():.3f}]")
                print(f"    yd: [{xform.yd.min().item():.3f}, {xform.yd.max().item():.3f}]")
                print(f"    zd: [{xform.zd.min().item():.3f}, {xform.zd.max().item():.3f}]")

    # If master didn't provide space info, try the first NIfTI warp in chain
    if master_hdr_obj is None:
        for xform in transforms:
            if isinstance(xform, NonlinearWarp):
                warp_hdr = xform.header_info.get("header")
                if warp_hdr is not None:
                    master_space_info = get_afni_space_info(warp_hdr)
                    master_hdr_obj = warp_hdr
                    break

    # Auto-pad the output grid so a large translation / warp can't push data off
    # the edge. We estimate the worst-case displacement from the loaded
    # transforms and grow the grid by that much (plus -expad). Affine matrices,
    # already in output-voxel space, are conjugated by the index shift so the
    # composed warp is correct on the padded grid; nonlinear warps re-resample to
    # the padded grid automatically in compose_chain.
    pad = (0, 0, 0)
    if auto_pad:
        pad = _estimate_warp_padding(transforms, output_shape, output_affine)
    if expad > 0:
        pad = tuple(p + expad for p in pad)  # type: ignore[assignment]
    # Cap runaway padding (likely a units/affine bug, not real data) but honor
    # generous growth -- up to one full FOV per side.
    cap = (output_shape[-1], output_shape[-2], output_shape[-3])  # (x, y, z) caps
    capped = tuple(min(p, c) for p, c in zip(pad, cap, strict=True))
    if capped != pad and verb >= 1:
        print(
            f"nwarpforge: estimated pad {pad} exceeds one-FOV cap {cap}; "
            f"clamping to {capped} (check warp units if data is clipped)"
        )
    pad = capped

    if any(p > 0 for p in pad):
        shift = torch.tensor(
            [pad[0], pad[1], pad[2]], dtype=torch.float32, device=device
        )
        t_mat = torch.eye(4, dtype=torch.float32, device=device)
        t_mat[:3, 3] = shift
        t_inv = torch.eye(4, dtype=torch.float32, device=device)
        t_inv[:3, 3] = -shift
        for xf in transforms:
            if isinstance(xf, AffineTransform):
                # M_pad = T(pad) @ M @ T(-pad), batched over time rows.
                xf.matrices = t_mat @ xf.matrices @ t_inv
        old_shape = output_shape
        output_shape, output_affine = _pad_output_grid(output_shape, output_affine, pad)
        if verb >= 1:
            print(
                f"nwarpforge: auto-pad +{pad} (x,y,z) voxels: "
                f"{tuple(old_shape[-3:])} -> {tuple(output_shape[-3:])}"
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
    phase_volumes: list[Tensor] = []

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

        if phase_data is not None:
            ph_vol = phase_data[t] if is_4d else phase_data

            if phase_warp == "complex":
                # Current approach: convert mag+phase -> real/imag, warp
                # each, convert back.  Magnitude is derived from warped
                # real/imag and can be corrupted near phase wraps.
                real_vol = src_vol * torch.cos(ph_vol)
                imag_vol = src_vol * torch.sin(ph_vol)
                warped_real = apply_composed_warp(
                    real_vol,
                    composed,
                    source_affine=source_header["affine"],
                    output_affine=output_affine,
                    interp=interp,
                    no_neg=no_neg,
                )
                warped_imag = apply_composed_warp(
                    imag_vol,
                    composed,
                    source_affine=source_header["affine"],
                    output_affine=output_affine,
                    interp=interp,
                    no_neg=no_neg,
                )
                warped = torch.sqrt(warped_real**2 + warped_imag**2)
                warped_phase = torch.atan2(warped_imag, warped_real)

            elif phase_warp == "split":
                # Warp magnitude directly (smooth, interpolates cleanly),
                # then warp real/imag and extract phase only.  Magnitude
                # is never touched by phase data.
                warped = apply_composed_warp(
                    src_vol,
                    composed,
                    source_affine=source_header["affine"],
                    output_affine=output_affine,
                    interp=interp,
                    no_neg=no_neg,
                )
                real_vol = src_vol * torch.cos(ph_vol)
                imag_vol = src_vol * torch.sin(ph_vol)
                warped_real = apply_composed_warp(
                    real_vol,
                    composed,
                    source_affine=source_header["affine"],
                    output_affine=output_affine,
                    interp=interp,
                    no_neg=no_neg,
                )
                warped_imag = apply_composed_warp(
                    imag_vol,
                    composed,
                    source_affine=source_header["affine"],
                    output_affine=output_affine,
                    interp=interp,
                    no_neg=no_neg,
                )
                warped_phase = torch.atan2(warped_imag, warped_real)

            elif phase_warp == "direct":
                # Warp magnitude and phase independently.  Assumes phase
                # is already unwrapped (no wraps).  Fastest — one warp
                # per volume instead of two.
                warped = apply_composed_warp(
                    src_vol,
                    composed,
                    source_affine=source_header["affine"],
                    output_affine=output_affine,
                    interp=interp,
                    no_neg=no_neg,
                )
                warped_phase = apply_composed_warp(
                    ph_vol,
                    composed,
                    source_affine=source_header["affine"],
                    output_affine=output_affine,
                    interp=interp,
                    no_neg=no_neg,
                )

            elif phase_warp == "circular":
                # Warp cos(phase) and sin(phase) separately (unit circle
                # interpolation), then atan2 back.  Handles wraps without
                # magnitude corruption.  Best for wrapped phase data.
                warped = apply_composed_warp(
                    src_vol,
                    composed,
                    source_affine=source_header["affine"],
                    output_affine=output_affine,
                    interp=interp,
                    no_neg=no_neg,
                )
                cos_ph = torch.cos(ph_vol)
                sin_ph = torch.sin(ph_vol)
                warped_cos = apply_composed_warp(
                    cos_ph,
                    composed,
                    source_affine=source_header["affine"],
                    output_affine=output_affine,
                    interp=interp,
                    no_neg=no_neg,
                )
                warped_sin = apply_composed_warp(
                    sin_ph,
                    composed,
                    source_affine=source_header["affine"],
                    output_affine=output_affine,
                    interp=interp,
                    no_neg=no_neg,
                )
                warped_phase = torch.atan2(warped_sin, warped_cos)

            else:
                raise ValueError(
                    f"Unknown phase_warp: {phase_warp!r}. "
                    "Use 'complex', 'split', 'direct', or 'circular'."
                )
            phase_volumes.append(warped_phase)
        else:
            warped = apply_composed_warp(
                src_vol,
                composed,
                source_affine=source_header["affine"],
                output_affine=output_affine,
                interp=interp,
                no_neg=no_neg,
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

    # Save warped phase if requested
    if phase_volumes:
        assert phase_prefix is not None  # set earlier when phase_path was set
        if len(phase_volumes) > 1:
            phase_output = torch.stack(phase_volumes)
        else:
            phase_output = phase_volumes[0]
        save_image(phase_output, phase_prefix, header_info=output_header)
        if verb >= 1:
            print(f"Saved phase: {phase_prefix}")
