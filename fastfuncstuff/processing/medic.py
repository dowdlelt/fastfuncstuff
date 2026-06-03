"""MEDIC — Multi-Echo DIstortion Correction (frame-wise B0 field maps).

The field-map **estimation** is delegated entirely to **warpkit** (a hard
dependency) — its compiled C++ ROMEO + MCPC-3D-S + temporal/SVD pipeline is
fast and correct, and there's no reason to reimplement it:

    Van, A.N., Montez, D.F., Laumann, T.O., et al. (2026). Frame-wise multi-echo
    distortion correction for superior functional MRI. Imaging Neuroscience.
    https://doi.org/10.1162/IMAG.a.1262   ·   github.com/vanandrew/warpkit (MIT)

What this module owns is the **GPU warping/apply** side and the ffs pipeline
integration:
    - warpkit_field_native     call warpkit's in-process unwrap -> native field map
    - field_to_displacement_pe  field map (Hz) -> PE displacement (voxels)
    - invert_displacement_pe    1-D fixed-point inversion along PE
    - undistort_series          GPU-accelerated per-frame undistortion (wsinc5)
    - save_medic_warp           write ffs_qwarp-compatible mm warps for ffs_nwarp
    - medic_fieldmaps           orchestrator: warpkit estimate -> our warp + apply

Design notes live in the wiki: ``concepts/MEDIC.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

# Map a phase-encoding direction string to a NIfTI spatial axis (0=i/x, 1=j/y,
# 2=k/z).  The sign ("j-") only affects displacement polarity, handled later.
PE_AXIS_MAP = {
    "i": 0,
    "j": 1,
    "k": 2,
    "i-": 0,
    "j-": 1,
    "k-": 2,
    "x": 0,
    "y": 1,
    "z": 2,
    "x-": 0,
    "y-": 1,
    "z-": 2,
}


def rescale_phase_to_radians(data: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linearly map raw scanner phase in ``[lo, hi]`` to ``[-pi, pi]``.

    Only used to feed wrapped phase into the circular undistortion path
    (``-apply_phase``); warpkit rescales its own inputs internally.
    """
    return (data - lo) / (hi - lo) * (2.0 * np.pi) - np.pi


# ----------------------------------------------------------------------------
# Estimation: delegated to warpkit (their C++ ROMEO unwrap + field map)
# ----------------------------------------------------------------------------
def warpkit_available() -> bool:
    """True if the warpkit package (with its compiled C++ ROMEO) is importable."""
    import importlib.util

    return importlib.util.find_spec("warpkit") is not None


def warpkit_field_native(
    phase: np.ndarray,
    mag: np.ndarray,
    affine: np.ndarray,
    tes: Sequence[float],
    n_cpus: int = 4,
    svd_filt: int = 10,
    border_filt: tuple[int, int] = (1, 5),
    verbose: bool = True,
) -> np.ndarray:
    """Native-space field map (Hz) via warpkit's in-process C++ ROMEO unwrap.

    Calls ``warpkit.unwrap.unwrap_and_compute_field_maps`` — the *exact* warpkit
    pipeline (MCPC-3D-S offset, combined automask, multi-echo ROMEO, per-echo
    global-mode correction, temporal consistency, SVD) on its compiled C++
    ROMEO. ``phase`` is RAW scanner phase (warpkit rescales internally).

    Returns the field map in Hz, distorted space, shape ``(nx, ny, nz, T)``.
    """
    import nibabel as nib
    from warpkit.unwrap import unwrap_and_compute_field_maps

    _nx, _ny, _nz, ne, nt = phase.shape
    phase_imgs = [
        nib.Nifti1Image(np.ascontiguousarray(phase[:, :, :, e, :]), affine) for e in range(ne)
    ]
    mag_imgs = [
        nib.Nifti1Image(np.ascontiguousarray(mag[:, :, :, e, :]), affine) for e in range(ne)
    ]
    if verbose:
        print(f"  warpkit: in-process C++ ROMEO unwrap + field map ({ne} echoes, {nt} frames)")
    field_img = unwrap_and_compute_field_maps(
        phase_imgs,
        mag_imgs,
        np.asarray(tes, dtype=np.float32),
        svd_filt=svd_filt,
        border_filt=border_filt,
        n_cpus=n_cpus,
    )
    return np.asarray(field_img.dataobj, dtype=np.float32)  # (nx, ny, nz, T)


def compute_frame_masks(
    mag: np.ndarray,
    tes: Sequence[float],
    automask_dilation: int = 3,
    device: torch.device | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Brain mask valued 0/1/2 (outside / core / dilated border), broadcast to T.

    A single AFNI-style automask on the temporal-mean shortest-echo magnitude
    (on ``device``) + GPU max-pool dilation. Used for the debug mask image and
    ``MedicResult.masks``.
    """
    import torch.nn.functional as F

    from .mask import automask

    nx, ny, nz, _ne, nt = mag.shape
    echo_idx = int(np.argmin(tes))
    mean_mag = mag[..., echo_idx, :].mean(axis=-1)  # (nx, ny, nz)
    vol = torch.from_numpy(np.ascontiguousarray(mean_mag.transpose(2, 1, 0))).float()
    if device is not None:
        vol = vol.to(device)

    core = automask(vol, dilate_extra=0, device=device).bool()  # (nz, ny, nx)
    x = core.float()[None, None]
    for _ in range(max(0, automask_dilation)):
        x = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
    dil = x[0, 0] > 0.5
    mask3d = (core.to(torch.int8) + dil.to(torch.int8)).cpu().numpy().transpose(2, 1, 0)
    if verbose:
        n_in = int((mask3d > 0).sum())
        print(f"  brain mask: {n_in:,} voxels (core+border), broadcast to {nt} frames")
    return np.broadcast_to(mask3d[..., None], (nx, ny, nz, nt)).copy()


# ----------------------------------------------------------------------------
# Our GPU warping side: field <-> PE displacement, 1-D inversion, apply.
# ----------------------------------------------------------------------------
def field_to_displacement_pe(field_hz: Tensor, total_readout_time: float, pe_dir: str) -> Tensor:
    """Field map (Hz) -> displacement along PE, in voxels.

    voxel shift = field[Hz] * total_readout_time[s].  Sign flips for a negative
    PE polarity ("j-").  Result stays in voxel units (our warp convention).
    """
    disp = field_hz * float(total_readout_time)
    if pe_dir.endswith("-"):
        disp = -disp
    return disp


def displacement_pe_to_field(disp: Tensor, total_readout_time: float, pe_dir: str) -> Tensor:
    """Inverse of :func:`field_to_displacement_pe` (voxels -> Hz)."""
    field = disp / float(total_readout_time)
    if pe_dir.endswith("-"):
        field = -field
    return field


def _interp_along_last_axis(field: Tensor, coord: Tensor) -> Tensor:
    """Linear sample of ``field`` along its last axis at fractional ``coord``.

    ``field`` and ``coord`` share shape; ``coord`` holds fractional indices into
    the last axis. Out-of-range samples clamp to the edge.
    """
    n = field.shape[-1]
    c = coord.clamp(0, n - 1)
    lo = torch.floor(c).long()
    hi = (lo + 1).clamp(max=n - 1)
    frac = c - lo.to(c.dtype)
    f_lo = torch.gather(field, -1, lo)
    f_hi = torch.gather(field, -1, hi)
    return f_lo * (1 - frac) + f_hi * frac


def invert_displacement_pe(disp: Tensor, pe_tensor_axis: int, iters: int = 50) -> Tensor:
    """Invert a 1-D displacement field along one axis (voxel units).

    Fixed-point iteration h(x) = -g(x + h(x)), the 1-D analogue of ITK's
    iterative displacement-field inverse used by warpkit.  ``disp`` (= g) is the
    distorted-space PE displacement; the result (= h) is the undistorted-space
    pull displacement for resampling distorted data onto the undistorted grid.
    """
    g = disp.movedim(pe_tensor_axis, -1).contiguous()
    n = g.shape[-1]
    idx = torch.arange(n, device=g.device, dtype=g.dtype)
    idx = idx.expand_as(g)
    h = torch.zeros_like(g)
    for _ in range(iters):
        h = -_interp_along_last_axis(g, idx + h)
    return h.movedim(-1, pe_tensor_axis).contiguous()


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------
@dataclass
class MedicResult:
    """End-to-end MEDIC output (all torch tensors, (nx, ny, nz, T) NIfTI layout)."""

    field_native: Tensor  # Hz, distorted space (from warpkit)
    displacement_pe: Tensor  # voxels along PE, undistorted-space pull warp
    field_undistorted: Tensor  # Hz, undistorted space
    masks: Tensor  # int8 0/1/2
    pe_tensor_axis: int


def medic_fieldmaps(
    phase: np.ndarray,
    mag: np.ndarray,
    tes: Sequence[float],
    affine: np.ndarray,
    total_readout_time: float,
    pe_dir: str,
    svd_filt: int = 10,
    border_filt: tuple[int, int] = (1, 5),
    automask_dilation: int = 3,
    n_cpus: int = 4,
    device: torch.device | None = None,
    debug_dir: str | None = None,
    verbose: bool = True,
) -> MedicResult:
    """Estimate frame-wise field maps (warpkit) and build our GPU pull warp.

    Estimation (multi-echo unwrap + field map) is warpkit's. We convert its
    native field map to a per-frame phase-encode displacement, invert it to the
    undistorted-space pull, and return everything the warp/apply tail needs.

    ``debug_dir``: if set, write sanity-check intermediates (warpkit native field
    map, brain mask, PE displacement) there.

    Parameters
    ----------
    phase, mag : np.ndarray
        (nx, ny, nz, ne, T) RAW arrays (warpkit rescales phase internally).
    affine : np.ndarray
        4x4 NIfTI affine.
    total_readout_time : float
        Total EPI readout time in seconds.
    pe_dir : str
        Phase-encoding direction, e.g. "j" / "j-" / "y".
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tes_np = np.asarray(tes, dtype=np.float32)
    _nx, _ny, _nz, _ne, nt = phase.shape
    pe_axis = PE_AXIS_MAP[pe_dir]

    _dbg = None
    if debug_dir is not None:
        import os as _os

        from ..io.afni import save_nifti as _save_nifti

        _os.makedirs(debug_dir, exist_ok=True)

        def _dbg(name: str, arr: np.ndarray) -> None:
            _save_nifti(
                np.ascontiguousarray(arr.astype(np.float32)),
                _os.path.join(debug_dir, f"{name}.nii.gz"),
                affine=affine,
            )

    # 1. estimation: warpkit native field map (Hz, distorted space).
    field_native_np = warpkit_field_native(
        phase,
        mag,
        affine,
        tes_np,
        n_cpus=n_cpus,
        svd_filt=svd_filt,
        border_filt=border_filt,
        verbose=verbose,
    )
    field_native = torch.nan_to_num(
        torch.from_numpy(np.ascontiguousarray(field_native_np)).to(device)
    )
    masks_t = torch.from_numpy(
        compute_frame_masks(mag, tes_np, automask_dilation, device=device, verbose=verbose)
    ).to(device)
    if _dbg is not None:
        _dbg("fieldmap_native", field_native_np)
        _dbg("mask", masks_t.cpu().numpy())

    # 2. our warp: field -> PE displacement (voxels, distorted) -> invert (pull).
    disp_native = field_to_displacement_pe(field_native, total_readout_time, pe_dir)
    disp_pull = torch.empty_like(disp_native)
    inv_it = tqdm(range(nt), desc="invert displacement", disable=not verbose or nt == 1)
    for t in inv_it:
        disp_pull[..., t] = invert_displacement_pe(disp_native[..., t], pe_axis)
    if _dbg is not None:
        _dbg("disp_pull_vox", disp_pull.cpu().numpy())
        _dbg("disp_native_vox", disp_native.cpu().numpy())

    # 3. undistorted-space field map (sign check + deliverable).
    field_undist = displacement_pe_to_field(disp_pull, total_readout_time, pe_dir)
    a = field_undist[..., 0].reshape(-1)
    b = field_native[..., 0].reshape(-1)
    if torch.dot(a - a.mean(), b - b.mean()) < 0:
        field_undist = -field_undist

    return MedicResult(
        field_native=field_native,
        displacement_pe=disp_pull,
        field_undistorted=field_undist,
        masks=masks_t,
        pe_tensor_axis=pe_axis,
    )


def save_medic_warp(
    disp_pull: Tensor,
    pe_nifti_axis: int,
    affine: np.ndarray,
    prefix_stem: str,
    nii_ext: str = ".nii.gz",
    as_5d: bool = False,
) -> str:
    """Write the per-frame distortion warp as an ffs_qwarp-compatible mm warp.

    ffs_nwarp then composes it (with ``-master``) into a final single-resample
    chain to atlas, exactly like a static warp.

    ``save_warp_field(units="mm")`` + ``load_warp(units="mm")`` apply a net x/y
    sign flip (the DICOM<->NIfTI convention), so we feed the negated displacement
    for i/j phase encoding; the cardinal voxel-size factor cancels, so this is
    affine-independent. Verified against :func:`undistort_series` in tests.

    Returns the path/glob to pass to ``ffs_nwarp -nwarp`` (a ``warp_*`` wildcard
    for the default per-frame files, or the single 5D file when ``as_5d``).
    """
    import os

    from ..io.afni import save_nifti
    from .io import save_warp_field
    from .nwarpforge import compute_cardinal_affine

    cardinal = compute_cardinal_affine(affine)
    feed_sign = -1.0 if pe_nifti_axis in (0, 1) else 1.0
    disp_feed = (feed_sign * disp_pull).detach().cpu().numpy()  # (nx,ny,nz,T)
    nx, ny, nz, nt = disp_feed.shape

    if as_5d:
        rpp = float(cardinal[pe_nifti_axis, pe_nifti_axis])  # signed PE voxel mm
        warp = np.zeros((nx, ny, nz, nt, 3), dtype=np.float32)
        warp[..., pe_nifti_axis] = rpp * disp_feed
        path = f"{prefix_stem}_warp{nii_ext}"
        save_nifti(warp, path, affine=cardinal)
        return path

    warp_dir = f"{prefix_stem}_warp"
    os.makedirs(warp_dir, exist_ok=True)
    for t in range(nt):
        comp = torch.from_numpy(disp_feed[:, :, :, t]).permute(2, 1, 0).contiguous()
        zeros = torch.zeros_like(comp)
        xyz = [zeros, zeros, zeros]
        xyz[pe_nifti_axis] = comp  # (nz,ny,nx) voxel disp on the PE axis
        save_warp_field(
            xyz[0],
            xyz[1],
            xyz[2],
            os.path.join(warp_dir, f"warp_{t:05d}{nii_ext}"),
            header_info={"affine": cardinal, "header": None},
            units="mm",
        )
    return os.path.join(warp_dir, f"warp_*{nii_ext}")


def undistort_series(
    series: Tensor,
    disp_pull: Tensor,
    pe_nifti_axis: int,
    interp: str = "wsinc5",
    circular: bool = False,
    verbose: bool = True,
    desc: str = "undistort",
) -> Tensor:
    """Apply the per-frame undistortion pull warp to a 4D series, in native space.

    The displacement is along the phase-encode axis only and lives on the data's
    own grid, so this is a single high-quality interpolation per frame — no grid
    change. Distortion correction is intrinsically a native-space operation;
    motion / coregistration / atlas come afterward as affines (compose those with
    ffs_nwarp -master).

    Parameters
    ----------
    series : Tensor
        (nx, ny, nz, T) one echo's data (NIfTI axis order).
    disp_pull : Tensor
        (nx, ny, nz, T) PE-axis voxel displacement — the undistorted-space pull
        warp from :func:`medic_fieldmaps` (``MedicResult.displacement_pe``).
    pe_nifti_axis : int
        Phase-encode axis (0=i, 1=j, 2=k).
    interp : str
        "wsinc5" (default, high quality) or "linear".
    circular : bool
        Treat ``series`` as wrapped phase in radians: warp cos/sin and atan2 back
        so wraps don't smear. Use for phase; leave False for magnitude.

    Returns
    -------
    Tensor
        (nx, ny, nz, T) undistorted series.
    """
    from .interp import warp_image_linear, warp_image_wsinc5

    wfun = warp_image_wsinc5 if interp == "wsinc5" else warp_image_linear
    nt = series.shape[-1]
    out = torch.empty_like(series)
    it = tqdm(range(nt), desc=desc, disable=not verbose or nt == 1, leave=False)
    for t in it:
        # (nx,ny,nz) NIfTI order -> (nz,ny,nx) for the warp kernels.
        d = disp_pull[..., t].permute(2, 1, 0).contiguous()
        zero = torch.zeros_like(d)
        comps = [zero, zero, zero]
        comps[pe_nifti_axis] = d  # xd/yd/zd indexed by i/j/k
        if circular:
            vol = series[..., t].permute(2, 1, 0).contiguous()
            cos_w = wfun(torch.cos(vol), comps[0], comps[1], comps[2])
            sin_w = wfun(torch.sin(vol), comps[0], comps[1], comps[2])
            out[..., t] = torch.atan2(sin_w, cos_w).permute(2, 1, 0)
        else:
            vol = series[..., t].permute(2, 1, 0).contiguous()
            warped = wfun(vol, comps[0], comps[1], comps[2])
            out[..., t] = warped.permute(2, 1, 0)
    return out
