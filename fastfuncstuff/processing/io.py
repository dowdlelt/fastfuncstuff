"""NIfTI I/O utilities for loading and saving images and warp fields.

Handles 3D and 4D NIfTI images via nibabel, converting to/from PyTorch tensors.
Warp fields are saved as 4D NIfTI (nx, ny, nz, 3) with the three displacement
components along the 4th dimension.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from fastfuncstuff.io.afni import load_nifti, save_nifti

try:
    import nibabel as nib
except ImportError:
    nib = None


def derive_mean_output_path(prefix: str | Path) -> str:
    """Build mean-image output path as mean_{basename}{ext} in same directory.

    Examples:
        epi_mc.nii.gz -> mean_epi_mc.nii.gz
        /tmp/out.nii  -> /tmp/mean_out.nii
        out           -> mean_out
    """
    p = Path(prefix)
    name = p.name

    if name.endswith(".nii.gz"):
        stem = name[:-7]
        ext = ".nii.gz"
    else:
        ext = p.suffix
        stem = p.stem if ext else name

    return str(p.with_name(f"mean_{stem}{ext}"))


def _require_nibabel() -> None:
    if nib is None:
        raise ImportError("nibabel is required for NIfTI I/O. Install with: pip install nibabel")


def load_image(path: str | Path, device: torch.device | None = None) -> tuple[Tensor, object]:
    """Load a NIfTI image as a torch tensor.

    Args:
        path: Path to .nii or .nii.gz file.
        device: Torch device for the tensor.

    Returns:
        (data, header_info): data is (nz, ny, nx) for 3D or (nt, nz, ny, nx) for 4D.
        header_info is a dict with 'affine' and 'header' for saving back.
    """
    img = load_nifti(path)
    data = np.asarray(img.dataobj, dtype=np.float32)

    header_info = {
        "affine": img.affine.copy(),
        "header": img.header.copy(),
    }

    # AFNI/NIfTI files are frequently "fake" 5D (or 4D) with singleton higher
    # dimensions (e.g. a single 3D volume stored as (x,y,z,1) or (x,y,z,1,1),
    # or sub-bricks in dim[5] with dim[4]==1). Drop those singleton non-spatial
    # axes so a lone volume reads as 3D and a real time series as 4D. The first
    # three (spatial) axes are always kept.
    if data.ndim > 3:
        squeeze_axes = tuple(ax for ax in range(3, data.ndim) if data.shape[ax] == 1)
        if squeeze_axes:
            data = np.squeeze(data, axis=squeeze_axes)
        if data.ndim > 4:
            raise ValueError(
                f"Expected 3D or 4D image after squeezing singleton dims, got shape {data.shape}"
            )

    # NIfTI convention: (x, y, z [, t]) -> we want (z, y, x [, t transposed])
    # nibabel loads as (i, j, k [, t]) in file order
    if data.ndim == 3:
        # (nx, ny, nz) in NIfTI -> transpose to (nz, ny, nx) for our convention
        data = data.transpose(2, 1, 0)
        tensor = torch.from_numpy(data.copy())
    elif data.ndim == 4:
        # (nx, ny, nz, nt) -> (nt, nz, ny, nx)
        data = data.transpose(3, 2, 1, 0)
        tensor = torch.from_numpy(data.copy())
    else:
        raise ValueError(f"Expected 3D or 4D NIfTI, got {data.ndim}D")

    if device is not None:
        tensor = tensor.to(device)

    return tensor, header_info


def save_image(
    data: Tensor,
    path: str | Path,
    header_info: dict | None = None,
    affine: np.ndarray | None = None,
    use_pigz: bool = True,
    brick_labels: list[str] | None = None,
) -> None:
    """Save a torch tensor as a NIfTI image.

    Args:
        data: (nz, ny, nx) for 3D or (nt, nz, ny, nx) for 4D.
        path: Output path (.nii or .nii.gz).
        header_info: Dict from load_image with 'affine' and 'header'.
        affine: 4x4 affine matrix. Uses header_info's affine or identity if None.
        use_pigz: Use pigz for parallel gzip compression (default: True if available).
        brick_labels: Optional per-sub-brick labels written into the AFNI NIfTI
            extension (BRICK_LABS) so AFNI viewers show them.
    """
    arr = data.detach().cpu().numpy()

    if affine is None:
        if header_info is not None:
            affine = header_info["affine"]
        else:
            affine = np.eye(4)

    if arr.ndim == 3:
        arr = arr.transpose(2, 1, 0)
    elif arr.ndim == 4:
        arr = arr.transpose(3, 2, 1, 0)

    # Extract header from header_info to preserve TR, xyzt_units, etc.
    header = header_info.get("header") if header_info is not None else None
    save_nifti(
        arr.astype(np.float32),
        output_path=path,
        affine=affine,
        header=header,
        brick_labels=brick_labels,
    )


def save_warp_field(
    xd: Tensor,
    yd: Tensor,
    zd: Tensor,
    path: str | Path,
    header_info: dict | None = None,
    affine: np.ndarray | None = None,
    units: str = "voxels",
    padding: tuple[int, int, int] | None = None,
    use_pigz: bool = True,
) -> None:
    """Save displacement warp field as a 4D NIfTI.

    The warp is saved as (nx, ny, nz, 3) with the 4th dimension being
    [x_displacement, y_displacement, z_displacement].

    When units="voxels", displacements are in voxel index units (our internal format).
    When units="mm", displacements are converted to mm using the CARDINAL
    (deobliqued) affine, matching AFNI: all of AFNI's warp conversions go through
    ijk_to_dicom (cardinal), so a warp constrained to one grid axis stays in one
    component even for oblique data. The original (possibly oblique) affine is still
    written to the file header.

    Args:
        xd, yd, zd: (nz, ny, nx) displacement fields in voxel units.
        path: Output path (.nii or .nii.gz).
        header_info: Dict from load_image with 'affine' and 'header'.
        affine: 4x4 affine matrix.
        units: "voxels" (default) or "mm" (AFNI-compatible).
        padding: (pad_x, pad_y, pad_z) if warp is on a padded grid.
            The affine origin will be shifted to account for padding.
    """
    _require_nibabel()

    if affine is None:
        if header_info is not None:
            affine = header_info["affine"].copy()
        else:
            affine = np.eye(4)
    else:
        affine = affine.copy()

    # Shift affine origin to account for padding
    if padding is not None:
        pad_x, pad_y, pad_z = padding
        if pad_x > 0 or pad_y > 0 or pad_z > 0:
            # The padded grid starts pad voxels earlier in each direction
            # New origin = old_origin + affine[:3,:3] @ (-pad_x, -pad_y, -pad_z)
            pad_vox = np.array([-pad_x, -pad_y, -pad_z], dtype=np.float64)
            affine[:3, 3] += affine[:3, :3] @ pad_vox

    xd_np = xd.detach().cpu().numpy()
    yd_np = yd.detach().cpu().numpy()
    zd_np = zd.detach().cpu().numpy()

    if units == "mm":
        # Convert voxel displacements to mm with the CARDINAL (deobliqued) affine,
        # not the raw oblique one. AFNI does every warp conversion through
        # ijk_to_dicom (the cardinal version), carrying obliquity separately rather
        # than folding it into the displacement values (3dQwarp -help: "(xd,yd,zd)
        # are stored in DICOM order"; ijk_to_dicom aligns grid axes to x/y/z). Using
        # the oblique rs here would rotate a single-grid-axis warp across components
        # and would NOT be undone on apply, which goes through compute_cardinal_affine
        # too. For axis-aligned data cardinal == oblique, so this is a no-op there.
        from .nwarpforge import compute_cardinal_affine

        rs = compute_cardinal_affine(affine)[:3, :3]
        # Apply scale (cardinal rs is diagonal up to axis permutation/sign) to the
        # displacement vectors. Shape: each is (nz, ny, nx), stack to (..., 3).
        disp_vox = np.stack([xd_np, yd_np, zd_np], axis=-1)  # (nz, ny, nx, 3)
        disp_mm = np.einsum("ij,...j->...i", rs, disp_vox)  # (nz, ny, nx, 3)
        xd_np = disp_mm[..., 0]
        yd_np = disp_mm[..., 1]
        zd_np = disp_mm[..., 2]

    # Stack: (nz, ny, nx, 3), transpose to NIfTI (nx, ny, nz, 3)
    arr = np.stack([xd_np, yd_np, zd_np], axis=-1)
    arr = arr.transpose(2, 1, 0, 3).astype(np.float32)

    save_nifti(arr, output_path=path, affine=affine)


def load_warp_field(
    path: str | Path, device: torch.device | None = None
) -> tuple[Tensor, Tensor, Tensor, dict]:
    """Load a displacement warp field from a 4D NIfTI.

    Args:
        path: Path to warp .nii or .nii.gz file.
        device: Torch device.

    Returns:
        (xd, yd, zd, header_info): Each displacement field is (nz, ny, nx).
    """
    img = load_nifti(path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    header_info = {
        "affine": img.affine.copy(),
        "header": img.header.copy(),
    }

    if data.ndim == 5 and data.shape[3] == 1 and data.shape[4] == 3:
        data = data.squeeze(3)
    elif data.ndim == 5 and data.shape[4] == 1 and data.shape[3] == 3:
        data = data.squeeze(4)

    if data.ndim != 4 or data.shape[3] != 3:
        raise ValueError(f"Expected 4D NIfTI with shape (nx,ny,nz,3), got shape {data.shape}")

    # (nx, ny, nz, 3) -> transpose to (nz, ny, nx, 3)
    data = data.transpose(2, 1, 0, 3)

    xd = torch.from_numpy(data[:, :, :, 0].copy())
    yd = torch.from_numpy(data[:, :, :, 1].copy())
    zd = torch.from_numpy(data[:, :, :, 2].copy())

    if device is not None:
        xd = xd.to(device)
        yd = yd.to(device)
        zd = zd.to(device)

    return xd, yd, zd, header_info
