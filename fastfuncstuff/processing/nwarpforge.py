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

import contextlib
import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from .interp import (
    _separable_resample_3d,
    normalize_interp_mode,
    trilinear_interpolate,
    trilinear_interpolate_multi,
    warp_image_multi,
)
from .io import (
    derive_prefixed_output_path,
    load_image,
    load_warp_field,
    save_first_last,
    save_image,
)
from .spacetime import TissueFollowingSampler, apply_spacetime_sample

# Interpolation kernels usable for warp-field (displacement) interpolation during
# composition. NN is excluded -- a nearest-sampled displacement field would be
# piecewise-constant and tear the warp.
WARP_COMPOSE_INTERP = ("linear", "cubic", "quintic", "heptic", "wsinc5")


def _sample_fields(
    fields: tuple[Tensor, ...], x: Tensor, y: Tensor, z: Tensor, interp: str = "linear"
) -> tuple[Tensor, ...]:
    """Sample co-registered displacement components at arbitrary coordinates.

    ``linear`` keeps the fast grid_sample path (border-clamped). Higher-order
    kernels reduce the smoothing each composition step adds to the warp -- the
    whole point of composing the chain in one shot is to interpolate as little as
    possible, so a sharper kernel here preserves warp detail (matches AFNI using
    the interp kernel for the warp itself, not just the data).
    """
    if interp == "linear":
        stack = torch.stack(fields, dim=0)
        sampled = trilinear_interpolate_multi(stack, x, y, z).T
        return tuple(sampled.unbind(0))
    # Edge-extend out-of-bounds (clamp coords to the field) instead of the
    # separable kernel's zero-fill: a displacement field must extrapolate its
    # border value, matching grid_sample's border mode used by the linear path.
    # Zero-filling would tear the warp where a composed coord leaves the grid.
    nz, ny, nx = fields[0].shape
    xc = x.clamp(0, nx - 1)
    yc = y.clamp(0, ny - 1)
    zc = z.clamp(0, nz - 1)
    if (
        fields[0].device.type == "cuda"
        and all(field.dtype == torch.float32 and field.ndim == 3 for field in fields)
        and os.environ.get("FFS_NWARP_NO_FUSED_FIELDS") != "1"
    ):
        # The fused dispatcher is scalar-volume only.  Three small fused launches
        # are substantially cheaper on CUDA than hiding displacement components
        # behind the portable (C,z,y,x) gather, even though the latter shares its
        # coordinate setup.  CPU/MPS keep that channel-sharing path below.
        return tuple(_separable_resample_3d(field, xc, yc, zc, interp) for field in fields)
    stack = torch.stack(fields, dim=0)
    sampled = _separable_resample_3d(stack, xc, yc, zc, interp)
    return tuple(sampled.unbind(0))


def _sample_field(field: Tensor, x: Tensor, y: Tensor, z: Tensor, interp: str = "linear") -> Tensor:
    """Single-component compatibility wrapper around :func:`_sample_fields`."""
    return _sample_fields((field,), x, y, z, interp)[0]


def derive_phase_output_path(prefix: str) -> str:
    """Derive phase output path from magnitude prefix.

    Examples:
        out.nii.gz   -> out_phase.nii.gz
        out.nii.zst  -> out_phase.nii.zst
        out.nii      -> out_phase.nii
    """
    from pathlib import Path

    p = Path(prefix)
    if p.name.endswith(".nii.gz"):
        stem = p.name[: -len(".nii.gz")]
        return str(p.parent / f"{stem}_phase.nii.gz")
    elif p.name.endswith(".nii.zst"):
        stem = p.name[: -len(".nii.zst")]
        return str(p.parent / f"{stem}_phase.nii.zst")
    elif p.name.endswith(".nii"):
        stem = p.name[: -len(".nii")]
        return str(p.parent / f"{stem}_phase.nii")
    else:
        return str(p.parent / f"{p.stem}_phase{p.suffix}")


from ..io.afni import get_afni_space_info, set_afni_space_info  # noqa: E402


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
    from fastfuncstuff.io.dsetinfo import cardinal_affine

    return cardinal_affine(oblique_aff)


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
        return (self.xd.shape[0], self.xd.shape[1], self.xd.shape[2])


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

    xd: Tensor  # (T, nz, ny, nx) — kept on CPU; only the active frame goes to GPU
    yd: Tensor
    zd: Tensor
    header_info: dict
    units: str = "nifti_mm"
    # Compute device that at_time() moves the selected frame onto. The full 5-D
    # field stays on CPU (it is T× a single-frame warp — resident on the GPU it
    # dwarfs the data and OOMs); only one frame is on the GPU at a time.
    device: torch.device | None = None

    @property
    def n_time(self) -> int:
        return self.xd.shape[0]

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        s = self.xd.shape
        return (int(s[1]), int(s[2]), int(s[3]))

    def at_time(self, t: int) -> NonlinearWarp:
        idx = min(t, self.xd.shape[0] - 1)
        xd, yd, zd = self.xd[idx], self.yd[idx], self.zd[idx]
        if self.device is not None:
            xd = xd.to(self.device)
            yd = yd.to(self.device)
            zd = zd.to(self.device)
        return NonlinearWarp(
            xd=xd,
            yd=yd,
            zd=zd,
            header_info=self.header_info,
            units=self.units,
        )


Transform = AffineTransform | NonlinearWarp | TimeVaryingWarp


@dataclass(frozen=True)
class VoxelGridPlan:
    """Reusable homogeneous coordinates for one output grid."""

    shape: tuple[int, int, int]
    coords: Tensor  # (4, N), x/y/z/1

    @property
    def ii(self) -> Tensor:
        return self.coords[0].reshape(self.shape)

    @property
    def jj(self) -> Tensor:
        return self.coords[1].reshape(self.shape)

    @property
    def kk(self) -> Tensor:
        return self.coords[2].reshape(self.shape)


def make_voxel_grid_plan(shape: tuple[int, int, int], device: torch.device) -> VoxelGridPlan:
    """Build one compact coordinate grid for reuse across frames and transforms."""
    nz, ny, nx = shape
    kk, jj, ii = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32, device=device),
        torch.arange(ny, dtype=torch.float32, device=device),
        torch.arange(nx, dtype=torch.float32, device=device),
        indexing="ij",
    )
    coords = torch.stack(
        (ii.reshape(-1), jj.reshape(-1), kk.reshape(-1), torch.ones_like(ii).reshape(-1))
    )
    return VoxelGridPlan(shape=shape, coords=coords)


@dataclass(frozen=True)
class WarpApplyPlan:
    """Grid geometry and affine conversion shared by every frame in a series."""

    grid: VoxelGridPlan
    source_from_output: Tensor


def make_warp_apply_plan(
    shape: tuple[int, int, int],
    source_affine: np.ndarray,
    output_affine: np.ndarray,
    device: torch.device,
    *,
    grid_plan: VoxelGridPlan | None = None,
) -> WarpApplyPlan:
    """Prepare invariant output-to-source geometry once."""
    grid = grid_plan or make_voxel_grid_plan(shape, device)
    if grid.shape != shape or grid.coords.device != device:
        raise ValueError("Grid plan does not match warp shape and device")
    src_card = compute_cardinal_affine(source_affine)
    out_card = compute_cardinal_affine(output_affine)
    matrix = (np.linalg.inv(src_card) @ out_card).astype(np.float32)
    source_from_output = torch.from_numpy(matrix).to(device)
    return WarpApplyPlan(grid=grid, source_from_output=source_from_output)


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

    This negation assumes the file is in AFNI's DICOM-mm convention, which is true
    for 3dQwarp output AND for FFS's own ``io.save_warp_field(units="mm")`` output
    (which now writes DICOM-mm too -- it negates x,y at save). So save->reload is a
    clean round-trip, and an FFS-saved mm warp is directly consumable by AFNI
    3dNwarpApply.

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
    from fastfuncstuff.io.afni import load_nifti

    shape = load_nifti(str(path)).shape
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
    from fastfuncstuff.io.afni import load_nifti

    img = load_nifti(str(path))
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
    if path_lower.endswith((".1d", ".txt")):
        return "affine"
    elif path_lower.endswith((".nii", ".nii.gz", ".nii.zst")):
        return "nonlinear"
    else:
        raise ValueError(f"Cannot identify transform type for: {path}")


# Phase-encode axis spelling -> spatial axis index (0=x/i/LR, 1=y/j/AP, 2=z/k/IS).
# The orientation names are treated as fixed shorthands for the voxel axes (i≡x≡LR,
# j≡y≡AP, k≡z≡IS), i.e. we do NOT read the image's actual anatomical orientation --
# "-jac AP" is just another way to write "-jac j", matching how the flag is used.
_PE_AXIS_ALIASES = {
    "i": 0,
    "x": 0,
    "lr": 0,
    "rl": 0,
    "j": 1,
    "y": 1,
    "ap": 1,
    "pa": 1,
    "k": 2,
    "z": 2,
    "is": 2,
    "si": 2,
}


def parse_pe_axis(spec: str) -> int:
    """Map a phase-encode axis spelling to a spatial axis index (0/1/2).

    Accepts ``i/j/k``, ``x/y/z`` and the orientation pairs ``LR/RL``, ``AP/PA``,
    ``IS/SI`` (case-insensitive) -- ``-jac j``, ``-jac y`` and ``-jac AP`` are
    equivalent. Returns 0 for x/i, 1 for y/j, 2 for z/k.
    """
    key = spec.strip().lower()
    if key not in _PE_AXIS_ALIASES:
        raise ValueError(f"unrecognised phase-encode axis {spec!r}; use i/j/k, x/y/z or LR/AP/IS")
    return _PE_AXIS_ALIASES[key]


def compute_pe_jacobian(warp: NonlinearWarp, pe_axis: int) -> tuple[Tensor, bool, float]:
    """Jacobian intensity-modulation map for a phase-encode distortion warp.

    For a warp whose displacement is confined to the phase-encode (PE) axis, the
    determinant of its Jacobian is simply ``1 + d(disp_pe)/d(pe)`` -- the same
    factor topup/blipflip apply to make a geometrically-undistorted image
    quantitatively correct (compression regions are dimmed back down, stretched
    regions brightened). Applying a bare displacement warp omits it, leaving the
    pile-up "streaks" at high-distortion edges.

    Returns ``(jac, single_axis, off_on_ratio)``. ``single_axis`` is False when the
    composed warp has non-PE displacement (an affine mixed axes, or it is a genuine
    3-D warp) -- then the 1-D Jacobian is not meaningful and the caller should skip
    modulation. ``jac`` is always returned so the caller can decide.
    """
    from .penalty import _central_diff_batched

    comps = (warp.xd, warp.yd, warp.zd)
    disp = comps[pe_axis]
    on = float(disp.abs().max())
    off = max(float(comps[a].abs().max()) for a in range(3) if a != pe_axis)
    ratio = off / (on + 1e-6)
    single_axis = ratio < 0.02
    # Spatial axis a (0=x,1=y,2=z) -> _central_diff_batched dim (0=z,1=y,2=x) is 2-a.
    jac = 1.0 + _central_diff_batched(disp, dim=2 - pe_axis)
    return jac, single_axis, ratio


def match_nwarp_spec(nwarp_specs: list[str], pattern: str) -> int:
    """Index of the single ``-nwarp`` entry matching ``pattern`` (for ``-jac axis:pat``).

    Matches an exact path, an exact basename, or a unique substring. Errors if the
    pattern matches zero or more than one entry, so the user must name the fieldmap
    unambiguously (a bare ``fmap`` is fine as long as nothing else in the chain matches).
    """
    exact = [i for i, s in enumerate(nwarp_specs) if s == pattern or Path(s).name == pattern]
    if len(exact) == 1:
        return exact[0]
    subs = [i for i, s in enumerate(nwarp_specs) if pattern in s]
    if len(subs) == 1:
        return subs[0]
    if not exact and not subs:
        raise ValueError(f"-jac: no -nwarp entry matches {pattern!r}")
    hits = [nwarp_specs[i] for i in (exact or subs)]
    raise ValueError(
        f"-jac: {pattern!r} matches multiple -nwarp entries {hits}; name it more specifically"
    )


def fieldmap_jacobian_transported(
    nwarp_specs: list[str],
    transforms: list[Transform],
    pattern: str,
    pe_axis: int,
    output_shape: tuple[int, int, int],
    output_affine: np.ndarray,
    device: torch.device,
    ainterp: str = "cubic",
) -> Tensor:
    """Fieldmap Jacobian on its native grid, transported to the output grid.

    The intensity modulation is a scalar attached to tissue: ``J = 1 + d(disp)/d(pe)``
    computed in the *fieldmap's own space* (clean, single-axis), then carried onto the
    output grid through **only the transforms downstream of the fieldmap** (those between
    it and the output). Upstream transforms (fieldmap -> source: motion, run-to-run) do
    not affect the modulation map -- they only decide which source voxel supplies the raw
    value. Downstream transforms must be static (no per-frame op), which holds when the
    per-frame motion/locomoco sit upstream (source side); the transported map is then a
    single constant field applied to every frame. This is exact for the full multi-run
    chain -- not a small-motion approximation -- and it deliberately transports only the
    fieldmap's Jacobian, never the downstream normalisation Jacobians.
    """
    k = match_nwarp_spec(nwarp_specs, pattern)
    fmap = transforms[k]
    if not isinstance(fmap, NonlinearWarp):
        raise ValueError(f"-jac: matched -nwarp entry {nwarp_specs[k]!r} is not a nonlinear warp")
    fmap_aff = fmap.header_info["affine"]
    # Voxel-unit displacement on the fieldmap's own grid (same mm->voxel convention the
    # chain applies), so J's sign matches the geometry that gets applied.
    fmap_vox = (
        fmap
        if fmap.units == "voxels"
        else prepare_warp_for_grid(fmap, fmap.shape, fmap_aff, device, verb=0)
    )
    jac_native, single, ratio = compute_pe_jacobian(fmap_vox, pe_axis)
    if not single:
        raise ValueError(
            f"-jac: {nwarp_specs[k]!r} is not confined to the phase-encode axis "
            f"(off/on={ratio:.3f}); is it the fieldmap warp, and is the axis right?"
        )
    # Downstream (output-side) sub-chain: transforms applied to the output coordinate
    # before it reaches the fieldmap. compose_chain applies transforms[0] first, so these
    # are transforms[:k]. They must be static for a constant transported map.
    down = transforms[:k]
    if any(_is_time_dependent(x) for x in down):
        raise ValueError(
            "-jac: a per-frame transform sits downstream of the fieldmap (between it and "
            "the output); the transported Jacobian would vary per frame -- unsupported"
        )
    if down:
        w_sub = compose_chain(
            down, output_shape, output_affine, device, time_idx=0, interp=ainterp, verb=0
        )
    else:
        z = torch.zeros(output_shape, dtype=torch.float32, device=device)
        w_sub = NonlinearWarp(
            xd=z, yd=z.clone(), zd=z.clone(), header_info={"affine": output_affine.copy()}
        )
    fx, fy, fz = _output_to_source_voxel_coords(w_sub, fmap_aff, output_affine)
    jac = trilinear_interpolate(jac_native, fx.reshape(-1), fy.reshape(-1), fz.reshape(-1)).reshape(
        output_shape
    )
    return jac


def prepare_warp_for_grid(
    warp: NonlinearWarp,
    target_shape: tuple[int, int, int],
    target_affine: np.ndarray,
    device: torch.device,
    verb: int = 1,
    interp: str = "linear",
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
        sampled = _sample_fields((warp.xd, warp.yd, warp.zd), flat_x, flat_y, flat_z, interp)
        mm_xd, mm_yd, mm_zd = (
            component.float().reshape(tgt_nz, tgt_ny, tgt_nx) for component in sampled
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


def compose_warp_then_matrix(
    warp: NonlinearWarp,
    matrix: Tensor,
    grid_plan: VoxelGridPlan | None = None,
) -> NonlinearWarp:
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

    grid_plan = grid_plan or make_voxel_grid_plan(warp.shape, device)
    if grid_plan.shape != warp.shape:
        raise ValueError("Grid plan shape does not match warp")
    ii, jj, kk = grid_plan.ii, grid_plan.jj, grid_plan.kk
    coords = grid_plan.coords.clone()
    coords[0].add_(warp.xd.reshape(-1))
    coords[1].add_(warp.yd.reshape(-1))
    coords[2].add_(warp.zd.reshape(-1))

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
    interp: str = "linear",
    grid_plan: VoxelGridPlan | None = None,
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

    grid_plan = grid_plan or make_voxel_grid_plan(warp.shape, device)
    if grid_plan.shape != warp.shape:
        raise ValueError("Grid plan shape does not match warp")
    ii, jj, kk = grid_plan.ii, grid_plan.jj, grid_plan.kk
    coords = grid_plan.coords

    transformed = matrix @ coords

    tx = transformed[0].reshape(nz, ny, nx)
    ty = transformed[1].reshape(nz, ny, nx)
    tz = transformed[2].reshape(nz, ny, nx)

    sampled = _sample_fields(
        (warp.xd, warp.yd, warp.zd), tx.reshape(-1), ty.reshape(-1), tz.reshape(-1), interp
    )
    warp_x_at_t, warp_y_at_t, warp_z_at_t = (component.reshape(nz, ny, nx) for component in sampled)

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
    interp: str = "linear",
    grid_plan: VoxelGridPlan | None = None,
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

    grid_plan = grid_plan or make_voxel_grid_plan(warp_a.shape, device)
    if grid_plan.shape != warp_a.shape:
        raise ValueError("Grid plan shape does not match warp")
    ii, jj, kk = grid_plan.ii, grid_plan.jj, grid_plan.kk
    x_after_a = ii + warp_a.xd
    y_after_a = jj + warp_a.yd
    z_after_a = kk + warp_a.zd

    sampled = _sample_fields(
        (warp_b.xd, warp_b.yd, warp_b.zd),
        x_after_a.reshape(-1),
        y_after_a.reshape(-1),
        z_after_a.reshape(-1),
        interp,
    )
    bx_at_a, by_at_a, bz_at_a = (component.reshape(nz, ny, nx) for component in sampled)

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
    interp: str = "linear",
    grid_plan: VoxelGridPlan | None = None,
) -> NonlinearWarp:
    """Compose a chain of transforms: C(B(A(x))).

    Args:
        transforms: List of AffineTransform or NonlinearWarp (in order A, B, C)
        output_shape: (nz, ny, nx) of output grid
        output_affine: Affine matrix for output grid
        device: Torch device
        time_idx: Which time point for time-dependent affines
        verb: Verbosity level
        interp: Kernel for warp-field interpolation during composition.

    Returns:
        Single composed warp field on output grid

    A ``NonlinearWarp`` already on the output grid (``units="voxels"``) is taken
    as pre-prepared and used as-is -- this is what lets :func:`reduce_chain`
    collapse static runs once and skip per-frame resampling.
    """
    if not transforms:
        raise ValueError("Empty transform chain")
    grid_plan = grid_plan or make_voxel_grid_plan(output_shape, device)

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

            result_warp = compose_warp_then_matrix(result_warp, mat, grid_plan)
            if verb >= 2:
                print(f"  [compose] after affine [{i}]: warp shape = {result_warp.shape}")

        elif isinstance(xform, NonlinearWarp | TimeVaryingWarp):
            # Resolve a time-varying warp to this frame's field. A static warp
            # already prepared to the output grid (units="voxels") is reused
            # as-is; an mm warp is resampled to output-grid voxel units now.
            frame_warp = xform.at_time(time_idx) if isinstance(xform, TimeVaryingWarp) else xform
            if isinstance(frame_warp, NonlinearWarp) and frame_warp.units == "voxels":
                prepared = frame_warp
            else:
                prepared = prepare_warp_for_grid(
                    frame_warp, output_shape, output_affine, device, verb=verb, interp=interp
                )
            if result_warp is None:
                result_warp = prepared
                if verb >= 2:
                    print(f"  [compose] first warp [{i}]: {prepared.shape} -> {result_warp.shape}")
            else:
                if verb >= 2:
                    print(f"  [compose] composing warp [{i}] -> {output_shape}")
                result_warp = compose_warp_then_warp(
                    result_warp, prepared, interp=interp, grid_plan=grid_plan
                )
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


def _is_time_dependent(xform: Transform) -> bool:
    """True if a transform's mapping changes per frame (multi-row affine / per-frame warp)."""
    if isinstance(xform, AffineTransform):
        return xform.matrices.shape[0] > 1
    return isinstance(xform, TimeVaryingWarp)


def reduce_chain(
    transforms: list[Transform],
    output_shape: tuple[int, int, int],
    output_affine: np.ndarray,
    device: torch.device,
    interp: str = "linear",
    verb: int = 1,
) -> list[Transform]:
    """Collapse maximal runs of *static* transforms into single pre-prepared warps.

    The per-frame warp loop otherwise re-resamples and re-composes every static
    transform (distortion, cross-run, affine-to-anat, nonlinear-to-MNI) on each
    volume, even though only the time-dependent pieces (per-volume motion, a
    per-frame distortion field) actually change. This reduces the chain to an
    alternating list of [pre-composed static warp | time-dependent transform],
    where every static warp is already on the output grid (``units="voxels"``),
    so :func:`compose_chain` skips its resampling on every subsequent frame.

    Collapsing the statics into one warp also composes them in a single shot,
    which minimises the interpolation smoothing those steps add.
    """
    reduced: list[Transform] = []
    run: list[Transform] = []

    def flush() -> None:
        if not run:
            return
        reduced.append(
            compose_chain(
                run, output_shape, output_affine, device, time_idx=0, interp=interp, verb=0
            )
        )
        run.clear()

    for xform in transforms:
        if _is_time_dependent(xform):
            flush()
            reduced.append(xform)
        else:
            run.append(xform)
    flush()

    if verb >= 2:
        print(f"  [reduce] {len(transforms)} transforms -> {len(reduced)} reduced slots")
    return reduced


def apply_composed_warp(
    source: Tensor,
    warp: NonlinearWarp,
    source_affine: np.ndarray,
    output_affine: np.ndarray,
    interp: str = "wsinc5",
    no_neg: bool = False,
    plan: WarpApplyPlan | None = None,
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
    return apply_composed_warp_multi(
        [source],
        warp,
        source_affine,
        output_affine,
        interp=interp,
        no_neg=no_neg,
        plan=plan,
    )[0]


def apply_composed_warp_multi(
    sources: list[Tensor],
    warp: NonlinearWarp,
    source_affine: np.ndarray,
    output_affine: np.ndarray,
    interp: str = "wsinc5",
    no_neg: bool = False,
    plan: WarpApplyPlan | None = None,
) -> list[Tensor]:
    """Apply one composed warp to co-registered channels or time points.

    Coordinate conversion and interpolation setup are shared across every input.
    Callers remain responsible for memory-planned channel batching.
    """
    if not sources:
        return []
    nz, ny, nx = warp.shape
    device = sources[0].device

    # Convert output-voxel → source-voxel using CARDINAL affines.
    # AFNI uses cardinal (deobliqued) coordinate matrices for all
    # index-to-coordinate conversions, so we must do the same.
    plan = plan or make_warp_apply_plan(warp.shape, source_affine, output_affine, device)
    if plan.grid.shape != warp.shape or plan.grid.coords.device != device:
        raise ValueError("Warp application plan does not match warp shape and device")
    coords = plan.grid.coords.clone()
    coords[0].add_(warp.xd.reshape(-1))
    coords[1].add_(warp.yd.reshape(-1))
    coords[2].add_(warp.zd.reshape(-1))
    src_coords = plan.source_from_output @ coords
    src_x = src_coords[0].reshape(nz, ny, nx)
    src_y = src_coords[1].reshape(nz, ny, nx)
    src_z = src_coords[2].reshape(nz, ny, nx)

    warped = warp_image_multi(
        sources,
        warp.xd,
        warp.yd,
        warp.zd,
        mode=interp,
        sample_coords=(src_x, src_y, src_z),
    )
    if no_neg:
        warped = [vol.clamp_min(0.0) for vol in warped]
    return warped


def _apply_affine_chain_batched(
    transforms: list[AffineTransform],
    source: Tensor,
    is_4d: bool,
    t_start: int,
    t_end: int,
    source_affine: np.ndarray,
    output_affine: np.ndarray,
    output_shape: tuple[int, int, int],
    interp: str,
    no_neg: bool,
    device: torch.device,
) -> list[Tensor]:
    """Fast path for an all-affine chain: compose to one pull matrix per frame
    and resample batched, skipping dense displacement-field materialization.

    For a chain ``[M0, M1, ..., Mk]`` (pull order ``C(B(A(x)))``), the composed
    output-voxel map is ``A_c = Mk @ ... @ M0`` and the source-sample location is
    ``inv(src_card) @ out_card @ A_c @ x`` — exactly what :func:`compose_chain` +
    :func:`apply_composed_warp` produce, but as a single 4x4 matmul per frame
    instead of a per-frame ``(nz,ny,nx,3)`` field build + meshgrid + coord matmul.
    Batched over time via :func:`apply_affine_interp_batched` (shared coord grid,
    one batched matmul), time-chunked per the [[Memory module]].
    """
    from fastfuncstuff.memory import compute_moco_resample_batch_size

    from .affine import apply_affine_interp_batched

    # Fixed output-voxel -> source-voxel grid conversion (AFNI cardinal frames,
    # same as apply_composed_warp).
    src_card = compute_cardinal_affine(source_affine)
    out_card = compute_cardinal_affine(output_affine)
    M_grid = torch.from_numpy((np.linalg.inv(src_card) @ out_card).astype(np.float32)).to(device)

    onz, ony, onx = output_shape
    # apply_affine_interp_batched supports linear/cubic/quintic/heptic/wsinc5.
    interp_k = interp if interp in ("cubic", "quintic", "heptic", "wsinc5") else "linear"

    def compose_at(t: int) -> Tensor:
        A_c = torch.eye(4, device=device)
        for xf in transforms:  # A_c = Mk @ ... @ M0 (matches compose_chain order)
            A_c = xf.at_time(t) @ A_c
        return M_grid @ A_c

    times = list(range(t_start, t_end))
    bs = (
        compute_moco_resample_batch_size(onz, ony, onx, len(times), device, interp=interp_k)
        if is_4d
        else 1
    )
    bs = max(1, min(bs, len(times)))

    out: list[Tensor] = []
    for i in range(0, len(times), bs):
        chunk = times[i : i + bs]
        mats = torch.stack([compose_at(t) for t in chunk])  # (b, 4, 4)
        srcs = (
            torch.stack([(source[t] if is_4d else source) for t in chunk]).to(device).float()
        )  # (b, snz, sny, snx)
        warped = apply_affine_interp_batched(
            srcs, mats, interp=interp_k, output_shape=(onz, ony, onx), zero_outside=True
        )
        if no_neg:
            warped = warped.clamp_min(0.0)
        out.extend(warped[j] for j in range(warped.shape[0]))
    return out


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

    from fastfuncstuff.io.afni import load_nifti

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
        img = load_nifti(str(s))
        shp = img.shape  # (nx, ny, nz, 3) or (nx, ny, nz, T, 3)
        nx, ny, nz = int(shp[0]), int(shp[1]), int(shp[2])
        return (nz, ny, nx), compute_cardinal_affine(np.asarray(img.affine)), img.header.copy()
    raise ValueError("-master WARP/NWARP requires at least one nonlinear warp dataset in -nwarp")


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

    px, py, pz = (int(math.ceil(b)) for b in bound)
    return px, py, pz


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


def _output_to_source_voxel_coords(
    warp: NonlinearWarp, source_affine: np.ndarray, output_affine: np.ndarray
) -> tuple[Tensor, Tensor, Tensor]:
    """Source-voxel coordinates each output voxel samples (the same map apply uses)."""
    nz, ny, nx = warp.shape
    device = warp.xd.device
    kk, jj, ii = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32, device=device),
        torch.arange(ny, dtype=torch.float32, device=device),
        torch.arange(nx, dtype=torch.float32, device=device),
        indexing="ij",
    )
    M = np.linalg.inv(compute_cardinal_affine(source_affine)) @ compute_cardinal_affine(
        output_affine
    )
    M_t = torch.from_numpy(M.astype(np.float32)).to(device)
    coords = torch.stack(
        [
            (ii + warp.xd).reshape(-1),
            (jj + warp.yd).reshape(-1),
            (kk + warp.zd).reshape(-1),
            torch.ones(nz * ny * nx, dtype=torch.float32, device=device),
        ],
        dim=0,
    )
    s = M_t @ coords
    return (
        s[0].reshape(nz, ny, nx),
        s[1].reshape(nz, ny, nx),
        s[2].reshape(nz, ny, nx),
    )


def _footprint_extent(
    transforms: list[Transform],
    grid_shape: tuple[int, ...],
    grid_affine: np.ndarray,
    source_shape: tuple[int, int, int],
    source_affine: np.ndarray,
    device: torch.device,
    time_indices: list[int],
) -> tuple[list[int], list[int]] | None:
    """Index-space bounding box of the *in-source* footprint on a given grid.

    Composes the warp on ``grid_shape`` for each representative frame and finds
    which grid voxels actually pull real source data (sample inside the source
    bounds). Returns ``(lo, hi)`` index bounds per (x, y, z) over all frames, or
    ``None`` if no voxel samples source. This is the basis for *overlap-based*
    padding: only the region that pulls real data matters, not the raw
    displacement magnitude (a warp to a far-away template can have huge
    displacements yet lose no data).
    """
    snz, sny, snx = source_shape
    nz, ny, nx = grid_shape[-3:]
    lo = [nx, ny, nz]
    hi = [-1, -1, -1]
    found = False
    for t in time_indices:
        comp = compose_chain(transforms, grid_shape, grid_affine, device, time_idx=t, verb=0)
        sx, sy, sz = _output_to_source_voxel_coords(comp, source_affine, grid_affine)
        in_src = (
            (sx >= -0.5)
            & (sx <= snx - 0.5)
            & (sy >= -0.5)
            & (sy <= sny - 0.5)
            & (sz >= -0.5)
            & (sz <= snz - 0.5)
        )
        if not bool(in_src.any()):
            continue
        found = True
        axes = [
            in_src.any(dim=0).any(dim=0),  # x
            in_src.any(dim=0).any(dim=1),  # y
            in_src.any(dim=1).any(dim=1),  # z
        ]
        for a in range(3):
            idx = axes[a].nonzero().flatten()
            lo[a] = min(lo[a], int(idx[0].item()))
            hi[a] = max(hi[a], int(idx[-1].item()))
    return (lo, hi) if found else None


def _needed_padding(
    transforms: list[Transform],
    output_shape: tuple[int, ...],
    output_affine: np.ndarray,
    source: Tensor,
    source_affine: np.ndarray,
    device: torch.device,
    time_indices: list[int],
    verb: int = 1,
    mass_tol: float = 1e-3,
) -> tuple[int, int, int]:
    """Per-axis padding (x, y, z) needed so no meaningful source data is clipped.

    Overlap- and mass-based, so a warp grows the grid only when it actually
    drops real signal (not just a zero background edge, and not merely because
    the displacement is large):

      1. Compose on the nominal grid and find the in-source footprint. If it does
         not touch any grid face, nothing is lost -> pad 0 (the common
         warp-to-template-with-margin case; no slowdown).
      2. If a face is touched, probe a grid grown on the touched axes (capped at
         one FOV and the raw displacement bound). Sample the source over the
         probe footprint and measure the fraction of source signal that falls
         *outside* the nominal region. If that lost fraction is below
         ``mass_tol`` the clipping is background-level -> pad 0; otherwise pad to
         the foreground overshoot and warn.
    """
    snz, sny, snx = (source.shape[-3], source.shape[-2], source.shape[-1])
    source_shape = (snz, sny, snx)
    # The full 4-D source normally remains host-backed; only this representative
    # volume is needed for the foreground-mass probe.
    src3d = (source[0] if source.ndim == 4 else source).to(device)
    nx, ny, nz = output_shape[-1], output_shape[-2], output_shape[-3]
    dims = (nx, ny, nz)

    ext = _footprint_extent(
        transforms,
        output_shape,
        output_affine,
        source_shape,
        source_affine,
        device,
        time_indices,
    )
    if ext is None:
        return (0, 0, 0)
    lo, hi = ext
    touched = [lo[a] == 0 or hi[a] == dims[a] - 1 for a in range(3)]
    if not any(touched):
        return (0, 0, 0)

    # Stage 2: grow a probe on the touched axes and weigh the clipped signal.
    raw = _estimate_warp_padding(transforms, output_shape, output_affine)
    probe = tuple(min(dims[a], raw[a] + 2) if touched[a] else 0 for a in range(3))
    pshape, paffine = _pad_output_grid(output_shape, output_affine, probe)
    pnz, pny, pnx = pshape

    need = [0, 0, 0]
    lost_frac = 0.0
    probe_edge_hit = [False, False, False]
    for t in time_indices:
        comp = compose_chain(transforms, pshape, paffine, device, time_idx=t, verb=0)
        sx, sy, sz = _output_to_source_voxel_coords(comp, source_affine, paffine)
        in_src = (
            (sx >= -0.5)
            & (sx <= snx - 0.5)
            & (sy >= -0.5)
            & (sy <= sny - 0.5)
            & (sz >= -0.5)
            & (sz <= snz - 0.5)
        )
        if not bool(in_src.any()):
            continue
        # |source| sampled over the probe footprint (the captured signal).
        vals = (
            trilinear_interpolate(src3d, sx.reshape(-1), sy.reshape(-1), sz.reshape(-1))
            .reshape(pnz, pny, pnx)
            .abs()
            * in_src.float()
        )
        total = float(vals.sum().item())
        if total <= 0:
            continue
        # Mass outside the nominal region (the probe border) is what padding-less
        # output would drop.
        keep = torch.zeros_like(vals)
        keep[probe[2] : probe[2] + nz, probe[1] : probe[1] + ny, probe[0] : probe[0] + nx] = 1.0
        lost_frac = max(lost_frac, float((vals * (1.0 - keep)).sum().item()) / total)

        # Foreground bbox (signal above a small fraction of its own max) so a
        # noise edge doesn't drive the pad amount.
        fg = vals > (0.02 * float(vals.max().item()))
        if not bool(fg.any()):
            continue
        ax = [fg.any(dim=0).any(dim=0), fg.any(dim=0).any(dim=1), fg.any(dim=1).any(dim=1)]
        pdims = (pnx, pny, pnz)
        for a in range(3):
            idx = ax[a].nonzero().flatten()
            flo, fhi = int(idx[0].item()), int(idx[-1].item())
            need[a] = max(0, need[a], probe[a] - flo, fhi - (probe[a] + dims[a] - 1))
            if flo == 0 or fhi == pdims[a] - 1:
                probe_edge_hit[a] = True

    if lost_frac < mass_tol:
        return (0, 0, 0)

    if verb >= 1:
        for a in range(3):
            if probe_edge_hit[a]:
                print(
                    f"nwarpforge: source signal still reaches the grid edge on axis "
                    f"{'xyz'[a]} at the padding cap; increase -expad if data is clipped"
                )
    return (need[0], need[1], need[2])


def _phase_spacetime_channels(
    mag_series: Tensor,
    phase_series: Tensor,
    phase_warp: str,
    base_no_neg: bool,
):
    """Complex channels + recombine for joint slice-timing of phase data.

    The space-time sampler is linear in source values, so warping the complex
    channels and recombining reproduces the per-frame ``phase_warp`` modes (see
    the non-slice-timing loop). Returns ``(channels, no_neg_per_channel,
    recombine)`` where ``recombine`` maps the sampled channel volumes to
    ``(magnitude, phase)``. ``no_neg`` is applied only to a true magnitude channel;
    real/imag/cos/sin legitimately go negative and must never be clamped.
    """
    cos_p = torch.cos(phase_series)
    sin_p = torch.sin(phase_series)
    if phase_warp == "complex":
        # Magnitude derived from the warped complex parts (can blur across wraps).
        channels = [mag_series * cos_p, mag_series * sin_p]
        no_neg = [False, False]

        def recombine(w: list[Tensor]) -> tuple[Tensor, Tensor]:
            wr, wi = w
            return torch.sqrt(wr**2 + wi**2), torch.atan2(wi, wr)

    elif phase_warp == "split":
        channels = [mag_series, mag_series * cos_p, mag_series * sin_p]
        no_neg = [bool(base_no_neg), False, False]

        def recombine(w: list[Tensor]) -> tuple[Tensor, Tensor]:
            wm, wr, wi = w
            return wm, torch.atan2(wi, wr)

    elif phase_warp == "direct":
        # Phase interpolated directly; assumes it is already unwrapped.
        channels = [mag_series, phase_series]
        no_neg = [bool(base_no_neg), False]

        def recombine(w: list[Tensor]) -> tuple[Tensor, Tensor]:
            wm, wp = w
            return wm, wp

    elif phase_warp == "circular":
        channels = [mag_series, cos_p, sin_p]
        no_neg = [bool(base_no_neg), False, False]

        def recombine(w: list[Tensor]) -> tuple[Tensor, Tensor]:
            wm, wc, ws = w
            return wm, torch.atan2(ws, wc)

    else:
        raise ValueError(
            f"Unknown phase_warp: {phase_warp!r}. Use 'complex', 'split', 'direct', or 'circular'."
        )
    return channels, no_neg, recombine


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
    save_max: bool = False,
    save_min: bool = False,
    save_first_last_flag: bool = False,
    dxyz: float | None = None,
    no_neg: bool = False,
    auto_pad: bool = True,
    expad: int = 0,
    ainterp: str = "cubic",
    slice_times: list[float] | None = None,
    tr: float | None = None,
    tzero: float | None = None,
    tinterp: str = "heptic",
    follow_tissue: bool = True,
    jac_axis: int | None = None,
    jac_match: str | None = None,
    progress: Callable[[str], AbstractContextManager[None]] | None = None,
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
        auto_pad: Grow the output grid only when a warp would clip real source
            signal off the edge (decided from in-source footprint overlap and a
            clipped-mass test, not raw displacement -- so a warp to a far-off
            template with margin does not grow the grid). Ignored when an explicit
            -master is given: the master grid is always honoured exactly. Only
            affects the master-less case (grid derived from the source).
        expad: Extra padding voxels added on every side on top of the auto
            estimate (AFNI -expad analogue). Also forces padding when
            auto_pad is False.
        ainterp: Kernel for warp-field interpolation during composition
            ("linear", "cubic", "quintic", "heptic", "wsinc5"). Higher order
            reduces the smoothing each composition step adds to the warp;
            "cubic" is a good accuracy/cost default for smooth displacement
            fields.
        slice_times: Per-slice acquisition offsets (seconds), length == source
            slices. When given, slice-timing correction is folded into the same
            resample as the warp chain (Roche 2011 joint space-time). Requires a
            4-D source and a real TR. Phase rides along (the complex channels go
            through the same space-time map; see _phase_spacetime_channels). See
            processing/spacetime.py.
        tr: Repetition time (seconds), required with slice_times.
        tzero: Reference time within the TR all slices align to (seconds).
            Defaults to the mean of slice_times (matches 3dTshift).
        tinterp: Temporal interpolation kernel for the joint path ("linear",
            "cubic", "quintic", "heptic", "wsinc5", "wsinc9"; default
            "heptic"). Fourier is not
            available here -- the per-voxel continuous shift is not a single
            per-slice phase rotation.
        jac_axis: If set (0=x/i, 1=y/j, 2=z/k), multiply the warped magnitude by
            the phase-encode Jacobian ``1 + d(disp)/d(axis)`` so a geometry-only
            distortion warp (fieldmap / MEDIC / locomoco) is intensity-corrected,
            like ``applytopup --method=jac``. With ``jac_match`` (``-jac axis:pattern``) the
            named fieldmap's Jacobian is computed on its native grid and transported through
            only the downstream (output-side) transforms -- exact for a full multi-run chain
            (motion/locomoco sit upstream and don't affect the modulation). Without a match,
            it auto-uses the static single-axis warp in the chain (fine when the fieldmap is
            the only nonlinear warp). Applied as a constant field per frame, so it works with
            per-frame motion and -tpattern; an affine-mixed / 3-D chain is left unmodulated.
        jac_match: Substring/filename selecting which -nwarp entry is the fieldmap for
            ``jac_axis`` (see :func:`match_nwarp_spec`). Required to modulate correctly
            through a multi-transform chain.
        follow_tissue: On the joint slice-timing path, sample each temporal
            neighbour at *its own* pose (tissue-following, the default) instead of
            freezing the output frame's pose (the slow-motion assumption). Recovers
            the right signal when motion moves tissue between scanner locations
            frame to frame (e.g. a brain edge sweeping in and out of a voxel), at
            ~1% GPU cost over the frozen path. Set False for the frozen-pose
            behaviour. See processing/spacetime.py:TissueFollowingSampler.
        progress: Optional context-manager factory for opaque loads and writes. The CLI
            supplies its shared spinner; library callers remain silent by default.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    interp = normalize_interp_mode(interp)
    if ainterp not in WARP_COMPOSE_INTERP:
        raise ValueError(f"ainterp must be one of {WARP_COMPOSE_INTERP}, got {ainterp!r}")

    t0_load = __import__("time").time()
    # A full fMRI series can exceed VRAM even though one interpolation batch fits.
    # Keep 4-D data on the host and stream memory-planned frame batches below;
    # 3-D inputs are small enough to place directly on the compute device.
    source, source_header = load_image(source_path, device=None)
    if source.ndim == 3:
        source = source.to(device)
    # load_image's header_info is declared `object` (loose I/O boundary type)
    # but is always the {"affine": ..., "header": ...} dict it documents.
    assert isinstance(source_header, dict)
    if verb >= 1:
        print(
            f"Loaded source: {source_path} {source.shape} ({__import__('time').time() - t0_load:.2f}s)"
        )

    # --- Phase dataset handling ---
    phase_data: Tensor | None = None
    if phase_path is not None:
        phase_raw, _ = load_image(phase_path, device=None)
        if phase_raw.ndim == 3:
            phase_raw = phase_raw.to(device)
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

    # Joint space-time (slice-timing) resampling: fold slice-timing correction
    # into the same interpolation as the warp chain (Roche 2011). Requires a 4-D
    # series and a real TR. Phase data rides along: the space-time sampler is
    # linear in source values, so we run it over the complex channels (real/imag
    # or cos/sin) and recombine, exactly as the per-frame phase path does. See
    # processing/spacetime.py and _phase_spacetime_channels below.
    slice_times_t: Tensor | None = None
    if slice_times is not None:
        if not is_4d:
            raise ValueError("slice-timing (-tpattern) requires a 4-D source")
        if tr is None or tr <= 0:
            raise ValueError("slice-timing (-tpattern) requires a positive TR (-TR or header)")
        snz = source.shape[1]
        if len(slice_times) != snz:
            raise ValueError(
                f"slice timing has {len(slice_times)} entries but source has {snz} slices"
            )
        if tzero is None:
            tzero = sum(slice_times) / len(slice_times)
        slice_times_t = torch.tensor(slice_times, dtype=torch.float32, device=device)
        if verb >= 1:
            print(
                f"nwarpforge: joint slice-timing on ({snz} slices, TR={tr:.4f}s, "
                f"tzero={tzero:.4f}s, tinterp={tinterp})"
            )

    # Phase + slice-timing: sample the complex channels through the same space-time
    # map and recombine (see _phase_spacetime_channels). Both samplers below drive
    # off ``st_channels``/``st_recombine`` when set.
    st_channels: list[Tensor] | None = None
    st_channel_no_neg: bool | list[bool] = no_neg
    st_recombine = None
    if slice_times_t is not None and phase_data is not None:
        # Build the complex channels on the HOST. They are several full 4-D series
        # and the space-time samplers stream frames to the device one tap at a time,
        # so the channels never need to sit on the GPU in full -- doing so OOMs on
        # real EPI (2+ extra copies of a multi-GiB series beside source/phase). The
        # phase+slice-timing loop reads only these channels, so free the GPU copies
        # of source/phase_data once they exist.
        st_channels, st_channel_no_neg, st_recombine = _phase_spacetime_channels(
            source.detach().to("cpu"), phase_data.detach().to("cpu"), phase_warp, no_neg
        )
        # The loop samples only the host channels; drop the GPU phase copy. source
        # stays on-device (still needed by padding/coord math below).
        phase_data = None
        # ``phase_raw`` otherwise remains a function-local reference to the
        # original GPU-sized series after its host channels have been built.
        del phase_raw
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if verb >= 1:
            print(f"nwarpforge: joint slice-timing carries phase (phase_warp={phase_warp})")

    master_hdr_obj: Any = None  # nibabel header for AFNI extension propagation
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
        master, master_header = load_image(master_path, device=None)
        assert isinstance(master_header, dict)
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

    progress_context = progress or (lambda _message: contextlib.nullcontext())
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
                # Keep the 5-D field on CPU; at_time() streams each frame to GPU.
                with progress_context(f"Loading {len(matches)} warp files matching {spec}"):
                    tv = load_time_varying_warp_from_files(
                        matches, device=torch.device("cpu"), debug=debug
                    )
                tv.device = device
                transforms.append(tv)
                max_time_points = max(max_time_points, tv.n_time)
                continue
            spec = matches[0]

        kind = identify_transform_type(spec)

        if kind == "affine":
            with progress_context(f"Loading affine {spec}"):
                xform = load_affine_1D(spec, affine_for_matrices, device=device, debug=debug)
            transforms.append(xform)
            max_time_points = max(max_time_points, xform.matrices.shape[0])
            if debug:
                m = xform.matrices[0].cpu().numpy()
                print("  [DEBUG] Matrix after conversion (t=0):")
                print(f"    [{m[0, 0]:9.5f} {m[0, 1]:9.5f} {m[0, 2]:9.5f} {m[0, 3]:9.5f}]")
                print(f"    [{m[1, 0]:9.5f} {m[1, 1]:9.5f} {m[1, 2]:9.5f} {m[1, 3]:9.5f}]")
                print(f"    [{m[2, 0]:9.5f} {m[2, 1]:9.5f} {m[2, 2]:9.5f} {m[2, 3]:9.5f}]")
                print(f"    [{m[3, 0]:9.5f} {m[3, 1]:9.5f} {m[3, 2]:9.5f} {m[3, 3]:9.5f}]")
        elif _is_time_varying_warp(spec):
            # Keep the 5-D field on CPU; at_time() streams each frame to GPU.
            with progress_context(f"Loading time-varying warp {spec}"):
                tv = load_time_varying_warp(spec, device=torch.device("cpu"), debug=debug)
            tv.device = device
            transforms.append(tv)
            max_time_points = max(max_time_points, tv.n_time)
        else:
            with progress_context(f"Loading nonlinear warp {spec}"):
                xform = load_warp(spec, device=device, debug=debug)
            transforms.append(xform)
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

    # Auto-pad the output grid so a warp can't push data off the edge -- but only
    # where data is *actually* lost. Padding is decided from the in-source
    # footprint overlap (see _needed_padding), not the raw displacement: a warp
    # to a far-off template has huge displacements yet loses no data, so the
    # common case returns pad 0 and the grid (and runtime) is unchanged. Affine
    # matrices, already in output-voxel space, are conjugated by the index shift
    # so composition stays correct on the padded grid; nonlinear warps re-resample
    # to it automatically in compose_chain.
    # Representative frames for the footprint probe: 0 and the last time-dependent
    # index (covers motion drift; static template affines dominate either way).
    # An explicit -master (a file, or WARP/NWARP) is a hard contract: the output
    # grid IS the master grid, full stop. Auto-pad must never grow it -- otherwise
    # each run's grid depends on its own warped-source footprint and independently
    # warped runs land on mismatched grids (the exact failure that breaks
    # concatenating multi-run data onto a shared MNI template). Padding then
    # cropping back to master is a no-op for the master-region values anyway, so we
    # simply skip it. Matches AFNI 3dNwarpApply -master (no autopad). expad stays an
    # explicit opt-in below.
    explicit_master = master_path is not None
    reps = sorted({0, max(0, max_time_points - 1)})
    pad = (0, 0, 0)
    if auto_pad and explicit_master and verb >= 1:
        print("nwarpforge: -master given -> output grid fixed to master (auto-pad skipped)")
    if auto_pad and not explicit_master:
        pad = _needed_padding(
            transforms,
            output_shape,
            output_affine,
            source,
            source_header["affine"],
            device,
            reps,
            verb=verb,
        )
    if expad > 0:
        pad = tuple(p + expad for p in pad)  # type: ignore[assignment]

    if any(p > 0 for p in pad):
        shift = torch.tensor([pad[0], pad[1], pad[2]], dtype=torch.float32, device=device)
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

    # The phase slice-timing path now samples only ``st_channels``, which live
    # on the host and are streamed by TissueFollowingSampler. At this point the
    # source was only needed for padding, so retaining its full 4-D GPU copy
    # would needlessly compete with per-frame composition.
    if st_channels is not None and source.device.type == "cuda":
        del source
        torch.cuda.empty_cache()

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

    # Accumulate finished volumes on the HOST. The caller stacks the whole series at
    # the end, so holding every warped frame on the GPU grows without bound (doubly
    # so when phase adds a second series) and starves the per-frame compose/resample
    # -- the failure mode on large master grids. stack/save_image/mean all accept CPU
    # tensors; on a CPU device this is a no-op. The affine-only fast path below keeps
    # its own (already bounded) batched output and is exempt.
    def _stash(vol: Tensor) -> Tensor:
        return vol.to("cpu")

    # Affine-only fast path: a chain with no nonlinear warp is a pure per-frame
    # 4x4 map. Composing it as matrices + a single batched resample is far
    # cheaper than materializing a displacement field per frame (the general
    # path below). NN falls through (batched sampler is interp-only); phase
    # warping keeps the field path.
    if jac_axis is not None and not any(
        isinstance(x, NonlinearWarp | TimeVaryingWarp) for x in transforms
    ):
        print(
            "nwarpforge: -jac given but the chain has no nonlinear warp; "
            "Jacobian modulation has no effect (nothing distorts the phase-encode axis)"
        )
        jac_axis = None

    affine_only = (
        phase_data is None
        and slice_times_t is None
        and interp not in ("NN", "nearest")
        and all(isinstance(x, AffineTransform) for x in transforms)
    )
    if affine_only:
        if verb >= 1:
            print("nwarpforge: affine-only chain -> batched affine fast path (no field build)")
        output_volumes = _apply_affine_chain_batched(
            transforms,
            source,
            is_4d,
            t_start,
            t_end,
            source_header["affine"],
            output_affine,
            output_shape,
            interp,
            no_neg,
            device,
        )

    # Geometry is invariant across a 4-D series. Keep one compact homogeneous
    # grid instead of rebuilding arange/meshgrid/ones for every frame and slot.
    grid_plan = None if affine_only else make_voxel_grid_plan(output_shape, device)
    apply_plan = (
        make_warp_apply_plan(
            output_shape, source_header["affine"], output_affine, device, grid_plan=grid_plan
        )
        if not affine_only and slice_times_t is None
        else None
    )

    # Collapse static runs once: the per-frame loop then only recomposes the
    # time-dependent pieces against pre-prepared static warps (no per-frame
    # resampling of distortion/anat/MNI warps). If nothing is time-dependent,
    # the whole chain composes a single time.
    reduced = (
        None
        if affine_only
        else reduce_chain(
            transforms, output_shape, output_affine, device, interp=ainterp, verb=verb
        )
    )
    static_composed: NonlinearWarp | None = None
    if reduced is not None and not any(_is_time_dependent(x) for x in reduced):
        static_composed = compose_chain(
            reduced,
            output_shape,
            output_affine,
            device,
            time_idx=0,
            interp=ainterp,
            verb=verb,
            grid_plan=grid_plan,
        )

    # Tissue-following joint path: a sliding-window sampler that keeps only the
    # ~2*half+2 frames it needs resident on-device, composing each frame's pose once
    # as the window advances (O(nt) work, O(window) memory). See
    # spacetime.py:TissueFollowingSampler.
    follow_sampler: TissueFollowingSampler | None = None
    if slice_times_t is not None and follow_tissue and reduced is not None:
        assert tr is not None and tzero is not None

        def _coords_for_frame(f: int) -> tuple[Tensor, Tensor, Tensor]:
            comp_f = (
                static_composed
                if static_composed is not None
                else compose_chain(
                    reduced,
                    output_shape,
                    output_affine,
                    device,
                    time_idx=f,
                    interp=ainterp,
                    verb=0,
                    grid_plan=grid_plan,
                )
            )
            return _output_to_source_voxel_coords(comp_f, source_header["affine"], output_affine)

        follow_sampler = TissueFollowingSampler(
            st_channels if st_channels is not None else source,
            _coords_for_frame,
            output_shape,
            tr,
            tzero,
            slice_times_t,
            device,
            tinterp=tinterp,
            interp=interp,
            no_neg=st_channel_no_neg,
            n_out=t_end - t_start,
            verb=verb,
        )

    # Jacobian intensity modulation (-jac): multiply the warped magnitude by
    # 1 + d(disp_pe)/d(pe) so a geometry-only distortion warp becomes quantitatively
    # correct. Gated to warps whose displacement is confined to the PE axis; a mixed
    # (affine-rotated or 3-D) composed warp is skipped with one warning.
    #
    # The distortion Jacobian is dominated by the *static* fieldmap; per-frame motion is
    # rigid (intensity-neutral) and any time-varying field component is far smaller. So we
    # derive one constant Jacobian from the static single-phase-encode-axis warp in the
    # chain and apply it to every frame -- even when time-varying transforms (motion, a
    # per-frame field) are also present, and including on the joint slice-timing path where
    # a constant scalar field commutes with the temporal resample. This is exact when the
    # chain is just the distortion warp; with downstream motion/normalisation it treats the
    # modulation as fixed in the output grid (a good approximation for small motion). Only
    # when *no* static PE-axis warp exists (a purely time-varying field) do we fall back to
    # the per-frame composed Jacobian -- which slice timing cannot use, hence the error.
    _jac_warned = [False]
    _static_jac: Tensor | None = None
    if jac_axis is not None and jac_match is not None:
        # Explicit fieldmap: compute its Jacobian on its native grid and transport it
        # through the downstream (output-side) sub-chain -- exact for a full multi-run
        # chain regardless of upstream motion. (See fieldmap_jacobian_transported.)
        _static_jac = fieldmap_jacobian_transported(
            nwarp_specs,
            transforms,
            jac_match,
            jac_axis,
            output_shape,
            output_affine,
            device,
            ainterp=ainterp,
        )
        if verb >= 1:
            print(f"nwarpforge: -jac using fieldmap '{jac_match}', transported to the output grid")
    elif jac_axis is not None:
        # Auto: use the static single-phase-encode-axis warp in the chain (works when the
        # fieldmap is the only nonlinear warp / the chain is otherwise identity-like).
        static_cands = (
            [static_composed]
            if static_composed is not None
            else [x for x in (reduced or []) if isinstance(x, NonlinearWarp)]
        )
        for cand in static_cands:
            jm, single, _ = compute_pe_jacobian(cand, jac_axis)
            if single:
                _static_jac = jm
                break
    if jac_axis is not None and slice_times_t is not None and _static_jac is None:
        raise ValueError(
            "-jac with -tpattern needs a static distortion warp confined to the "
            "phase-encode axis; none found (a purely time-varying field, or an "
            "axis-mixed chain) -- apply distortion correction as its own step"
        )

    def _apply_jac(vol: Tensor, composed: NonlinearWarp | None) -> Tensor:
        if jac_axis is None:
            return vol
        if _static_jac is not None:
            return vol * _static_jac
        # Dynamic (time-varying warp, no slice timing): per-frame Jacobian from composed.
        assert composed is not None
        jac, single_axis, ratio = compute_pe_jacobian(composed, jac_axis)
        if not single_axis:
            if not _jac_warned[0]:
                print(
                    f"nwarpforge: -jac skipped -- composed warp is not confined to the "
                    f"phase-encode axis (off/on={ratio:.3f}); only pure phase-encode "
                    f"distortion warps are Jacobian-modulated"
                )
                _jac_warned[0] = True
            return vol
        return vol * jac

    # Static nonlinear chain + 4-D source: every frame uses identical sample
    # coordinates. Share coordinate conversion, index tables, kernel weights and
    # (for linear) the grid_sample launch across memory-planned frame batches.
    # This retains the single-resample semantics while avoiding T repetitions of
    # interpolation setup. Time-varying motion and joint slice timing stay on the
    # per-frame/sliding-window paths below because their coordinates differ.
    static_series_batched = (
        is_4d
        and static_composed is not None
        and phase_data is None
        and slice_times_t is None
        and not affine_only
    )
    if static_series_batched:
        from fastfuncstuff.memory import compute_moco_resample_batch_size

        onz, ony, onx = output_shape
        n_frames = t_end - t_start
        frame_batch = compute_moco_resample_batch_size(
            onz, ony, onx, n_frames, device, interp=interp
        )
        frame_batch = max(1, min(frame_batch, n_frames))
        if verb >= 1:
            print(
                "nwarpforge: static nonlinear chain -> shared-coordinate "
                f"frame batches ({frame_batch} at a time)"
            )
        starts = range(t_start, t_end, frame_batch)
        if verb > 0 and n_frames > frame_batch:
            starts = tqdm(starts, desc="Warping frame batches", leave=True)
        for start in starts:
            stop = min(start + frame_batch, t_end)
            source_batch = source[start:stop].to(device)
            warped_batch = apply_composed_warp_multi(
                list(source_batch.unbind(0)),
                static_composed,
                source_affine=source_header["affine"],
                output_affine=output_affine,
                interp=interp,
                no_neg=no_neg,
                plan=apply_plan,
            )
            output_volumes.extend(_stash(_apply_jac(vol, static_composed)) for vol in warped_batch)

    time_iter = tqdm(
        range(t_start, t_end) if not (affine_only or static_series_batched) else range(0),
        desc="Warping volumes",
        disable=verb == 0 or (t_end - t_start) == 1,
    )
    for t in time_iter:
        if follow_sampler is not None:
            # Tissue-following joint: each temporal tap uses its own frame's pose,
            # served from the sampler's sliding window (skip the compose here).
            sampled = follow_sampler.sample(t)
            if st_recombine is not None:
                mag_vol, phase_vol = st_recombine(sampled)
                output_volumes.append(_stash(_apply_jac(mag_vol, None)))
                phase_volumes.append(_stash(phase_vol))
            else:
                output_volumes.append(_stash(_apply_jac(sampled, None)))
            continue

        composed = (
            static_composed
            if static_composed is not None
            else compose_chain(
                reduced,
                output_shape,
                output_affine,
                device,
                time_idx=t,
                interp=ainterp,
                verb=0 if (t_end - t_start) > 1 else verb,
                grid_plan=grid_plan,
            )
        )

        if slice_times_t is not None:
            # Joint slice-timing: sample the raw 4-D series at this frame's pose,
            # letting the temporal coordinate vary per voxel by the scanner slice
            # each output voxel lands in. sz is that scanner slice index.
            sx, sy, sz = _output_to_source_voxel_coords(
                composed, source_header["affine"], output_affine
            )
            assert tr is not None and tzero is not None
            warped = apply_spacetime_sample(
                st_channels if st_channels is not None else source,
                sx,
                sy,
                sz,
                t,
                tr,
                tzero,
                slice_times_t,
                tinterp=tinterp,
                interp=interp,
                no_neg=st_channel_no_neg,
            )
            if st_recombine is not None:
                mag_vol, phase_vol = st_recombine(warped)
                output_volumes.append(_stash(_apply_jac(mag_vol, composed)))
                phase_volumes.append(_stash(phase_vol))
            else:
                output_volumes.append(_stash(_apply_jac(warped, composed)))
            continue

        src_vol = (source[t] if is_4d else source).to(device)

        if phase_data is not None:
            ph_vol = (phase_data[t] if is_4d else phase_data).to(device)

            if phase_warp == "complex":
                # Current approach: convert mag+phase -> real/imag, warp
                # each, convert back.  Magnitude is derived from warped
                # real/imag and can be corrupted near phase wraps.
                real_vol = src_vol * torch.cos(ph_vol)
                imag_vol = src_vol * torch.sin(ph_vol)
                warped_real, warped_imag = apply_composed_warp_multi(
                    [real_vol, imag_vol],
                    composed,
                    source_affine=source_header["affine"],
                    output_affine=output_affine,
                    interp=interp,
                    no_neg=no_neg,
                    plan=apply_plan,
                )
                warped = torch.sqrt(warped_real**2 + warped_imag**2)
                warped_phase = torch.atan2(warped_imag, warped_real)

            elif phase_warp == "split":
                # Warp magnitude directly (smooth, interpolates cleanly),
                # then warp real/imag and extract phase only.  Magnitude
                # is never touched by phase data.
                real_vol = src_vol * torch.cos(ph_vol)
                imag_vol = src_vol * torch.sin(ph_vol)
                warped, warped_real, warped_imag = apply_composed_warp_multi(
                    [src_vol, real_vol, imag_vol],
                    composed,
                    source_affine=source_header["affine"],
                    output_affine=output_affine,
                    interp=interp,
                    no_neg=no_neg,
                    plan=apply_plan,
                )
                warped_phase = torch.atan2(warped_imag, warped_real)

            elif phase_warp == "direct":
                # Warp magnitude and phase independently.  Assumes phase
                # is already unwrapped (no wraps).  Fastest — one warp
                # per volume instead of two.
                warped, warped_phase = apply_composed_warp_multi(
                    [src_vol, ph_vol],
                    composed,
                    source_affine=source_header["affine"],
                    output_affine=output_affine,
                    interp=interp,
                    no_neg=no_neg,
                    plan=apply_plan,
                )

            elif phase_warp == "circular":
                # Warp cos(phase) and sin(phase) separately (unit circle
                # interpolation), then atan2 back.  Handles wraps without
                # magnitude corruption.  Best for wrapped phase data.
                cos_ph = torch.cos(ph_vol)
                sin_ph = torch.sin(ph_vol)
                warped, warped_cos, warped_sin = apply_composed_warp_multi(
                    [src_vol, cos_ph, sin_ph],
                    composed,
                    source_affine=source_header["affine"],
                    output_affine=output_affine,
                    interp=interp,
                    no_neg=no_neg,
                    plan=apply_plan,
                )
                warped_phase = torch.atan2(warped_sin, warped_cos)

            else:
                raise ValueError(
                    f"Unknown phase_warp: {phase_warp!r}. "
                    "Use 'complex', 'split', 'direct', or 'circular'."
                )
            phase_volumes.append(_stash(warped_phase))
        else:
            warped = apply_composed_warp(
                src_vol,
                composed,
                source_affine=source_header["affine"],
                output_affine=output_affine,
                interp=interp,
                no_neg=no_neg,
                plan=apply_plan,
            )
        output_volumes.append(_stash(_apply_jac(warped, composed)))

    if is_4d or len(output_volumes) > 1:
        output = torch.stack(output_volumes)
    else:
        output = output_volumes[0]

    # Build output header: use output_affine (cardinal), don't inherit
    # source's qform/sform which would conflict with the output grid.
    # Preserve temporal metadata (TR, units) from source.
    # Propagate AFNI view/space from master (e.g. tlrc + MNI_2009c_asym).
    # header_info["header"] is a nibabel header object; the dict itself is
    # typed loosely (see load_image), so this duck-typed access is expected.
    src_hdr: Any = source_header.get("header")
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
                for ext in hdr_candidate.extensions:  # ty: ignore[unresolved-attribute]
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
    with progress_context(f"Writing {prefix}"):
        save_image(output, prefix, header_info=output_header)

    # Temporal reductions of the warped series. max/min are coverage images: a
    # voxel that left the FoV (motion, or the warp itself) is 0 in the volumes
    # that lost it, so max = everything ever imaged, min = 0 wherever any volume
    # lost it. Taken here, after the single final resample, they describe the
    # OUTPUT grid exactly — no zero-boundary smearing from warping a min/max map.
    for want, which, reduce in (
        (save_mean, "mean", lambda t: t.mean(dim=0)),
        (save_max, "max", lambda t: t.amax(dim=0)),
        (save_min, "min", lambda t: t.amin(dim=0)),
    ):
        if not want:
            continue
        if output.ndim != 4:
            if verb >= 1:
                print(f"-save_{which} requested, but output is not 4D; skipping")
            continue
        out_path = derive_prefixed_output_path(prefix, which)
        with progress_context(f"Writing {which} {out_path}"):
            save_image(reduce(output), out_path, header_info=output_header)

    if save_first_last_flag:
        save_first_last(output, prefix, header_info=output_header, verb=verb)

    # Save warped phase if requested
    if phase_volumes:
        assert phase_prefix is not None  # set earlier when phase_path was set
        if len(phase_volumes) > 1:
            phase_output = torch.stack(phase_volumes)
        else:
            phase_output = phase_volumes[0]
        with progress_context(f"Writing phase {phase_prefix}"):
            save_image(phase_output, phase_prefix, header_info=output_header)
