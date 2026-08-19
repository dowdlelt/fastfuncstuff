"""NIfTI I/O utilities for loading and saving images and warp fields.

Handles 3D and 4D NIfTI images via nibabel, converting to/from PyTorch tensors.
Warp fields are saved as 4D NIfTI (nx, ny, nz, 3) with the three displacement
components along the 4th dimension.
"""

from __future__ import annotations

import glob as _glob
import os
import re
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from fastfuncstuff.io.afni import load_nifti, save_nifti

try:
    import nibabel as nib
except ImportError:
    nib = None


def derive_prefixed_output_path(prefix: str | Path, token: str) -> str:
    """Build a sibling output path as ``{token}_{basename}{ext}``.

    The NIfTI extension is preserved so the derived file matches the compression
    the user asked for on the original prefix.

    Examples:
        derive_prefixed_output_path("epi_mc.nii.gz", "mean") -> "mean_epi_mc.nii.gz"
        derive_prefixed_output_path("/tmp/out.nii", "firstlast") -> "/tmp/firstlast_out.nii"
    """
    p = Path(prefix)
    name = p.name

    if name.endswith(".nii.gz"):
        stem = name[:-7]
        ext = ".nii.gz"
    else:
        ext = p.suffix
        stem = p.stem if ext else name

    return str(p.with_name(f"{token}_{stem}{ext}"))


def derive_mean_output_path(prefix: str | Path) -> str:
    """Build mean-image output path as mean_{basename}{ext} in same directory.

    Examples:
        epi_mc.nii.gz -> mean_epi_mc.nii.gz
        /tmp/out.nii  -> /tmp/mean_out.nii
        out           -> mean_out
    """
    return derive_prefixed_output_path(prefix, "mean")


def save_first_last(
    data: Tensor,
    base_path: str | Path,
    header_info: dict | None = None,
    *,
    include_diff: bool = False,
    initial: bool = False,
    verb: int = 1,
) -> str | None:
    """Save the first & last volumes (optionally + their difference) as one 4-D file.

    The output is a small switchable stack — flip between the volumes in a viewer
    to see how much a correction moved the data. ``data`` is a 4-D time-first
    tensor ``(nt, nz, ny, nx)``; ``base_path`` is the series output path the file
    is named after (``firstlast_{basename}`` / ``firstlastdiff_{basename}``, with
    ``_initial`` appended when ``initial`` marks the pre-correction data).

    Returns the written path, or ``None`` when ``data`` is not a >=2-volume 4-D
    series.
    """
    if data.ndim != 4 or data.shape[0] < 2:
        if verb >= 1:
            print("  -save_first_last skipped: needs a 4-D series with >=2 volumes")
        return None

    first, last = data[0], data[-1]
    vols = [first, last]
    labels = ["first", "last"]
    if include_diff:
        # The difference is signed (last - first); leave it un-clamped so a viewer
        # with a diverging map shows where signal moved in either direction.
        vols.append(last - first)
        labels.append("diff(last-first)")

    stack = torch.stack(vols, dim=0)
    token = "firstlastdiff" if include_diff else "firstlast"
    if initial:
        token += "_initial"
    out_path = derive_prefixed_output_path(base_path, token)
    save_image(stack, out_path, header_info=header_info, brick_labels=labels)
    if verb >= 1:
        tag = " (pre-correction)" if initial else ""
        print(f"Saved {'first/last/diff' if include_diff else 'first/last'}{tag}: {out_path}")
    return out_path


def save_tsnr(
    data: Tensor,
    base_path: str | Path,
    header_info: dict | None = None,
    *,
    initial: bool = False,
    verb: int = 1,
) -> str | None:
    """Save a temporal-SNR map (mean / temporal std over time) for QC.

    ``data`` is a 4-D time-first tensor ``(nt, nz, ny, nx)``. Voxels with zero
    temporal std map to 0 (rather than inf/nan). The file is named
    ``tsnr_{basename}`` (``tsnr_initial_{basename}`` when ``initial`` marks the
    pre-correction data).

    Returns the written path, or ``None`` when ``data`` is not a >=2-volume 4-D
    series.
    """
    if data.ndim != 4 or data.shape[0] < 2:
        if verb >= 1:
            print("  -save_tsnr skipped: needs a 4-D series with >=2 volumes")
        return None

    mean = data.mean(dim=0)
    std = data.std(dim=0)
    tsnr = torch.where(std > 0, mean / std, torch.zeros_like(mean))
    token = "tsnr_initial" if initial else "tsnr"
    out_path = derive_prefixed_output_path(base_path, token)
    save_image(tsnr, out_path, header_info=header_info)
    if verb >= 1:
        tag = " (pre-correction)" if initial else ""
        print(f"Saved tSNR{tag}: {out_path}")
    return out_path


def _require_nibabel() -> None:
    if nib is None:
        raise ImportError("nibabel is required for NIfTI I/O. Install with: pip install nibabel")


def load_image(path: str | Path, device: torch.device | None = None) -> tuple[Tensor, dict]:
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
    padding: tuple[int, int, int] | tuple[int, int, int, int, int, int] | None = None,
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
        padding: Symmetric ``(x, y, z)`` or per-face
            ``(x-, x+, y-, y+, z-, z+)`` padding. The affine origin is shifted
            by the three lower-face amounts.
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
        if len(padding) == 3:
            pad_x, pad_y, pad_z = padding
        else:
            pad_x, _, pad_y, _, pad_z, _ = padding
        if pad_x > 0 or pad_y > 0 or pad_z > 0:
            # The padded grid starts pad voxels earlier in each direction
            # New origin = old_origin + affine[:3,:3] @ (-pad_x, -pad_y, -pad_z)
            pad_vox = np.array([-pad_x, -pad_y, -pad_z], dtype=np.float64)
            affine[:3, 3] += affine[:3, :3] @ pad_vox

    xd_np = xd.detach().cpu().numpy()
    yd_np = yd.detach().cpu().numpy()
    zd_np = zd.detach().cpu().numpy()

    if units == "mm":
        disp_vox = np.stack([xd_np, yd_np, zd_np], axis=-1)  # (nz, ny, nx, 3)
        disp = _disp_vox_to_dicom_mm(disp_vox, affine)
        xd_np, yd_np, zd_np = disp[..., 0], disp[..., 1], disp[..., 2]

    # Stack: (nz, ny, nx, 3), transpose to NIfTI (nx, ny, nz, 3)
    arr = np.stack([xd_np, yd_np, zd_np], axis=-1)
    arr = arr.transpose(2, 1, 0, 3).astype(np.float32)

    save_nifti(arr, output_path=path, affine=affine)


def _disp_vox_to_dicom_mm(disp_vox: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Convert voxel-index displacement vectors to AFNI DICOM-mm.

    ``disp_vox`` is ``(..., 3)`` in voxel-index units; returns the same shape in
    DICOM-mm. Used by :func:`save_warp_field` (units="mm") and
    :func:`save_warp_series` so single and time-varying warps share one convention.

    Converts with the CARDINAL (deobliqued) affine, not the raw oblique one. AFNI
    does every warp conversion through ijk_to_dicom (cardinal), carrying obliquity
    separately rather than folding it into the displacement values (3dQwarp -help:
    "(xd,yd,zd) are stored in DICOM order"; ijk_to_dicom aligns grid axes to x/y/z).
    Using the oblique rs here would rotate a single-grid-axis warp across components
    and would NOT be undone on apply (which goes through compute_cardinal_affine too).
    For axis-aligned data cardinal == oblique, so this is a no-op there.

    On-disk convention is AFNI DICOM-mm (matches 3dQwarp), so the file is directly
    consumable by AFNI 3dNwarpApply AND round-trips cleanly through
    nwarpforge.load_warp (which negates x,y DICOM->NIfTI on load). Two steps:
      1. voxel -> RAS/NIfTI-mm via the CARDINAL (deobliqued) rs.
      2. RAS-mm -> DICOM-mm: negate x,y (DICOM = diag(-1,-1,1) @ RAS).
    History: this used to stop at step 1 (RAS-mm on disk), which double-flipped x,y
    on reload (load_warp assumes DICOM) -- 3dQwarp on the same data showed inverted
    x,y. medic worked around it by pre-negating x,y at its save site; that
    compensation was removed from the per-frame path when this negation was added
    (medic's own 5d path still writes DICOM inline).
    """
    from .nwarpforge import compute_cardinal_affine

    rs = compute_cardinal_affine(affine)[:3, :3]
    disp_mm = np.einsum("ij,...j->...i", rs, disp_vox)  # RAS-mm
    out = np.empty_like(disp_mm)
    out[..., 0] = -disp_mm[..., 0]  # RAS -> DICOM: negate x
    out[..., 1] = -disp_mm[..., 1]  # RAS -> DICOM: negate y
    out[..., 2] = disp_mm[..., 2]
    return out


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


# ---------------------------------------------------------------------------
# Time-varying (per-frame) warp series — one 5D file OR a folder of 4D frames
# ---------------------------------------------------------------------------
#
# A time-series of warps (one displacement field per fMRI volume) is stored one
# of two interchangeable ways, and every ffs tool that reads/writes warp series
# goes through this pair so the two formats stay in lock-step:
#   * 5D  : a single NIfTI ``(nx, ny, nz, T, 3)`` — compact, "much better".
#   * folder: a directory of numbered 4D ``(nx, ny, nz, 3)`` files — every tool
#             (AFNI included) plays nice with these.
# On disk both are AFNI DICOM-mm (units="mm"), matching save_warp_field, so
# ffs_nwarp -nwarp consumes either (a 5D file, or a ``warp_*`` glob).


def _natural_key(s: str) -> list:
    """Sort key that orders embedded integers numerically (t2 < t10)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def save_warp_series(
    xd: Tensor,
    yd: Tensor,
    zd: Tensor,
    dest: str | Path,
    *,
    as_5d: bool,
    header_info: dict | None = None,
    affine: np.ndarray | None = None,
    units: str = "mm",
    padding: tuple[int, int, int] | tuple[int, int, int, int, int, int] | None = None,
    frame_prefix: str = "warp",
    frame_ext: str = ".nii.gz",
) -> str:
    """Write a time-varying warp as one 5D file or a folder of numbered 4D frames.

    ``xd, yd, zd`` are ``(T, nz, ny, nx)`` voxel-index displacement fields (the
    same internal convention as :func:`save_warp_field`, with a leading time axis).

    Args:
        dest: 5D → the output file path (``.nii``/``.nii.gz``); folder → the output
            directory (created; frames written as ``{frame_prefix}_{t:05d}{ext}``).
        as_5d: True → single 5D ``(nx, ny, nz, T, 3)`` file; False → per-frame folder.
        units: "mm" (AFNI DICOM-mm on disk, default) or "voxels".
        padding: Symmetric ``(x, y, z)`` or per-face
            ``(x-, x+, y-, y+, z-, z+)`` padding; the affine origin is shifted
            once by the lower-face amounts, exactly like :func:`save_warp_field`.

    Returns:
        The path to hand ``ffs_nwarp -nwarp``: the 5D file, or a ``{prefix}_*{ext}``
        glob for the folder.
    """
    _require_nibabel()

    if affine is None:
        affine = header_info["affine"].copy() if header_info is not None else np.eye(4)
    else:
        affine = affine.copy()
    if padding is not None:
        if len(padding) == 3:
            pad_x, pad_y, pad_z = padding
        else:
            pad_x, _, pad_y, _, pad_z, _ = padding
        if pad_x > 0 or pad_y > 0 or pad_z > 0:
            affine[:3, 3] += affine[:3, :3] @ np.array([-pad_x, -pad_y, -pad_z], dtype=np.float64)

    disp = np.stack(
        [t.detach().cpu().numpy() for t in (xd, yd, zd)], axis=-1
    )  # (T, nz, ny, nx, 3), voxels
    if disp.ndim != 5:
        raise ValueError(f"save_warp_series expects (T,nz,ny,nx) components, got {tuple(xd.shape)}")
    if units == "mm":
        disp = _disp_vox_to_dicom_mm(disp, affine)  # einsum broadcasts over (T,nz,ny,nx)
    elif units != "voxels":
        raise ValueError(f"units must be 'mm' or 'voxels', got {units!r}")

    if as_5d:
        # (T, nz, ny, nx, 3) -> NIfTI (nx, ny, nz, T, 3)
        arr = np.ascontiguousarray(disp.transpose(3, 2, 1, 0, 4), dtype=np.float32)
        save_nifti(arr, output_path=str(dest), affine=affine)
        return str(dest)

    os.makedirs(dest, exist_ok=True)
    n_t = disp.shape[0]
    for t in range(n_t):
        frame = np.ascontiguousarray(
            disp[t].transpose(2, 1, 0, 3), dtype=np.float32
        )  # (nx,ny,nz,3)
        save_nifti(
            frame,
            output_path=os.path.join(str(dest), f"{frame_prefix}_{t:05d}{frame_ext}"),
            affine=affine,
        )
    return os.path.join(str(dest), f"{frame_prefix}_*{frame_ext}")


def load_warp_series(
    source: str | Path,
    *,
    pattern: str = "warp_*.nii.gz",
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor, Tensor, dict, int]:
    """Load a time-varying warp from a 5D file OR a folder of numbered 4D frames.

    Returns displacement components ``(xd, yd, zd)`` each ``(T, nz, ny, nx)`` in
    the file's on-disk values — NO DICOM↔NIfTI sign conversion (callers that need
    it, e.g. warp composition, apply it themselves; PCA/analysis callers don't care
    about global sign). ``source`` may be:

      * a directory — globbed with ``pattern``, natural-sorted into frame order;
      * a 5D ``(nx, ny, nz, T, 3)`` file — split into T frames;
      * a 4D ``(nx, ny, nz, 3)`` file — a single frame (T=1).

    Returns ``(xd, yd, zd, header_info, n_frames)``.
    """
    if os.path.isdir(source):
        files = sorted(_glob.glob(os.path.join(str(source), pattern)), key=_natural_key)
        if not files:
            raise FileNotFoundError(f"No files matching {pattern!r} in {source}")
        xs, ys, zs, header_info = [], [], [], {}
        for f in files:
            x, y, z, hdr = load_warp_field(f)
            xs.append(x)
            ys.append(y)
            zs.append(z)
            header_info = hdr
        xd, yd, zd = torch.stack(xs), torch.stack(ys), torch.stack(zs)
    else:
        img = load_nifti(str(source))
        data = np.asarray(img.dataobj, dtype=np.float32)
        header_info = {"affine": img.affine.copy(), "header": img.header.copy()}
        if data.ndim == 4 and data.shape[3] == 3:
            data = data[:, :, :, None, :]  # (nx,ny,nz,1,3)
        if data.ndim != 5 or data.shape[-1] != 3:
            raise ValueError(
                f"Expected 5D (nx,ny,nz,T,3) or 4D (nx,ny,nz,3) warp, got {data.shape}: {source}"
            )
        # (nx, ny, nz, T, c) -> per component (T, nz, ny, nx)
        xd = torch.from_numpy(np.ascontiguousarray(data[..., 0].transpose(3, 2, 1, 0)))
        yd = torch.from_numpy(np.ascontiguousarray(data[..., 1].transpose(3, 2, 1, 0)))
        zd = torch.from_numpy(np.ascontiguousarray(data[..., 2].transpose(3, 2, 1, 0)))

    if device is not None:
        xd, yd, zd = xd.to(device), yd.to(device), zd.to(device)
    return xd, yd, zd, header_info, int(xd.shape[0])
