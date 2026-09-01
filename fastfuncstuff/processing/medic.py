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

import time
from collections.abc import Sequence
from contextlib import contextmanager
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
    return_unwrapped: bool = False,
    verbose: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Native-space field map (Hz) via warpkit's in-process C++ ROMEO unwrap.

    Runs the *exact* warpkit pipeline (MCPC-3D-S offset, combined automask,
    multi-echo ROMEO, per-echo global-mode correction, temporal consistency,
    SVD) on its compiled C++ ROMEO. ``phase`` is RAW scanner phase (warpkit
    rescales internally).

    We call warpkit's two public stages (``unwrap_phases`` then
    ``compute_field_maps``) separately rather than the combined
    ``unwrap_and_compute_field_maps`` wrapper, so the per-echo unwrapped phase
    — otherwise discarded by the wrapper — can be handed back. The field map is
    bit-identical either way (the wrapper is just these two calls back-to-back).

    Returns the field map in Hz, distorted space, shape ``(nx, ny, nz, T)``.
    If ``return_unwrapped``, also returns the per-echo unwrapped phase (radians),
    shape ``(nx, ny, nz, ne, T)`` — warpkit's ROMEO output before the echoes are
    collapsed into the field map.
    """
    import nibabel as nib
    from warpkit.unwrap import compute_field_maps, unwrap_phases

    _nx, _ny, _nz, ne, nt = phase.shape
    phase_imgs = [
        nib.Nifti1Image(np.ascontiguousarray(phase[:, :, :, e, :]), affine) for e in range(ne)
    ]
    mag_imgs = [
        nib.Nifti1Image(np.ascontiguousarray(mag[:, :, :, e, :]), affine) for e in range(ne)
    ]
    tes_np = np.asarray(tes, dtype=np.float32)
    if verbose:
        print(f"  warpkit: in-process C++ ROMEO unwrap + field map ({ne} echoes, {nt} frames)")
    t0 = time.perf_counter()
    with _warpkit_progress(nt, enabled=verbose and nt > 1):
        unwrapped_imgs, masks_img = unwrap_phases(phase_imgs, mag_imgs, tes_np, n_cpus=n_cpus)
        field_img = compute_field_maps(
            unwrapped_imgs,
            masks_img,
            mag_imgs,
            tes_np,
            border_filt=border_filt,
            svd_filt=svd_filt,
            n_cpus=n_cpus,
        )
    if verbose:
        dt = time.perf_counter() - t0
        print(f"  warpkit: done in {dt:.1f}s ({dt / nt:.2f}s/frame)")
    field = np.asarray(field_img.dataobj, dtype=np.float32)  # (nx, ny, nz, T)
    if not return_unwrapped:
        return field
    # Stack the per-echo 4D unwrapped phases (radians) -> (nx, ny, nz, ne, T).
    unwrapped = np.stack([np.asarray(u.dataobj, dtype=np.float32) for u in unwrapped_imgs], axis=-2)
    return field, unwrapped


def unwrapped_phase_to_field_hz(unwrapped_rad: Tensor, tes_ms: Sequence[float]) -> Tensor:
    """Per-echo unwrapped phase (radians) -> per-echo field estimate (Hz).

    Each echo's phase is a single-echo, through-origin B0 estimate: ``phi = 2*pi*f*TE``,
    so ``f[Hz] = phi / (2*pi*TE[s])``. Uses warpkit's exact convention
    (``compute_field_map`` regresses phase-vs-TE through the origin, then scales by
    ``1000/(2*pi)``), so these maps are directly comparable to — and their
    magnitude-weighted combination is — the collapsed field map.

    ``unwrapped_rad`` is ``(nx, ny, nz, ne, T)``; ``tes_ms`` are echo times in ms.
    """
    te = torch.as_tensor(list(tes_ms), dtype=unwrapped_rad.dtype, device=unwrapped_rad.device)
    scale = 1000.0 / (2.0 * np.pi * te)  # (ne,) Hz per radian, per echo
    return unwrapped_rad * scale.reshape(1, 1, 1, -1, 1)


# warpkit's per-frame stages (unwrap, temporal-consistency, field map) are each
# dispatched through ``warpkit.unwrap.run_executor``, which invokes ``post_fn``
# once per completed frame.  We have no other window into the compiled C++
# pipeline, so we temporarily wrap that symbol to drive a tqdm bar per stage.
_WARPKIT_STAGE_DESC = {
    "unwrap_phase": "ROMEO unwrap",
    "check_temporal_consistency_corr": "temporal consistency",
    "compute_field_map": "field map",
}


@contextmanager
def _warpkit_progress(n_frames: int, *, enabled: bool):
    """Patch ``warpkit.unwrap.run_executor`` to show a per-stage frame bar."""
    import warpkit.unwrap as _wku

    if not enabled:
        yield
        return

    orig = _wku.run_executor

    def run_executor_with_bar(ncpus, type, fn, iterator, initializer=None, post_fn=None):
        desc = _WARPKIT_STAGE_DESC.get(getattr(fn, "__name__", ""), "warpkit")
        bar = tqdm(total=n_frames, desc=f"    {desc}", unit="frame", leave=True)

        def wrapped_post(idx, result):
            if post_fn is not None:
                post_fn(idx, result)
            bar.update(1)

        try:
            orig(ncpus, type, fn, iterator, initializer=initializer, post_fn=wrapped_post)
        finally:
            bar.close()

    _wku.run_executor = run_executor_with_bar
    try:
        yield
    finally:
        _wku.run_executor = orig


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
    """End-to-end MEDIC output (all torch tensors, (nx, ny, nz, T) NIfTI layout).

    The per-frame fields are full-resolution 4D and live on the **host**: a single
    ``(nx, ny, nz, T)`` float32 volume is multiple GiB at typical grids, so several
    of them cannot co-reside on a consumer GPU. The GPU only ever holds a time-chunk
    (see :func:`medic_fieldmaps`); apply (:func:`undistort_series`) streams frames.
    """

    field_native: Tensor  # Hz, distorted space (from warpkit)
    displacement_pe: Tensor  # voxels along PE, undistorted-space pull warp
    field_undistorted: Tensor  # Hz, undistorted space
    masks: Tensor  # int8 0/1/2
    pe_tensor_axis: int
    unwrapped_field_hz: Tensor | None = None  # per-echo Hz (nx,ny,nz,ne,T); None unless requested


def _time_chunk_frames(n_voxels: int, device: torch.device, n_fields: int = 6) -> int:
    """Frames per GPU chunk so ~``n_fields`` full-volume float32 fields fit the budget.

    The field-map -> displacement -> 1-D inversion math is per-frame independent, so we
    stream time-chunks: only ``chunk`` frames of each ``(nx, ny, nz, T)`` field are
    GPU-resident at once. ``n_fields`` covers the simultaneously-live volumes
    (native / disp / pull / undistorted + the inversion's working tensors). The byte
    budget comes from the [[Memory module]] (``get_available_memory`` applies the GPU
    safety factor); CPU returns a large budget so the whole series is one chunk there.
    """
    from ..memory import get_available_memory

    budget = get_available_memory(device)
    per_frame = max(1, n_fields * n_voxels * 4)  # float32
    return max(1, int(budget // per_frame))


def field_to_pull_warp(
    field_native: Tensor,
    total_readout_time: float,
    pe_dir: str,
    pe_axis: int,
    device: torch.device,
    verbose: bool = True,
) -> tuple[Tensor, Tensor]:
    """Native field map (Hz) -> (PE pull warp, undistorted field map), host tensors.

    The cheap deterministic tail of MEDIC: field -> PE displacement -> 1-D inversion
    (the undistorted-space pull) -> undistorted-space field map. It is fully derived
    from ``field_native``, so a cached field map can be turned back into the warp
    without re-running warpkit. Streamed in time-chunks (:func:`_time_chunk_frames`)
    so only a chunk of frames is ever GPU-resident; inputs and outputs live on the host.

    Returns ``(disp_pull, field_undist)``, both ``(nx, ny, nz, T)`` on the host.
    """
    nx, ny, nz, nt = field_native.shape
    disp_pull = torch.empty_like(field_native)  # host
    field_undist = torch.empty_like(field_native)  # host
    chunk = min(nt, _time_chunk_frames(nx * ny * nz, device))
    inv_it = tqdm(
        range(0, nt, chunk),
        desc="invert displacement",
        disable=not verbose or nt == 1,
        leave=True,
    )
    for t0 in inv_it:
        t1 = min(t0 + chunk, nt)
        disp_c = field_to_displacement_pe(
            field_native[..., t0:t1].to(device), total_readout_time, pe_dir
        )
        pull_c = torch.empty_like(disp_c)
        for k in range(t1 - t0):
            pull_c[..., k] = invert_displacement_pe(disp_c[..., k], pe_axis)
        field_undist[..., t0:t1] = displacement_pe_to_field(
            pull_c, total_readout_time, pe_dir
        ).cpu()
        disp_pull[..., t0:t1] = pull_c.cpu()
        del disp_c, pull_c
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Sign check (frame 0): undistorted field should correlate with the native one.
    a = field_undist[..., 0].reshape(-1)
    b = field_native[..., 0].reshape(-1)
    if torch.dot(a - a.mean(), b - b.mean()) < 0:
        field_undist = -field_undist
    return disp_pull, field_undist


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
    return_unwrapped: bool = False,
    verbose: bool = True,
) -> MedicResult:
    """Estimate frame-wise field maps (warpkit) and build our GPU pull warp.

    Estimation (multi-echo unwrap + field map) is warpkit's. We convert its
    native field map to a per-frame phase-encode displacement, invert it to the
    undistorted-space pull, and return everything the warp/apply tail needs.

    ``debug_dir``: if set, write sanity-check intermediates (warpkit native field
    map, brain mask, PE displacement) there.

    ``return_unwrapped``: also keep warpkit's per-echo unwrapped phase, converted
    to a per-echo field estimate (Hz), in ``MedicResult.unwrapped_field_hz``.

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

    # 1. estimation: warpkit native field map (Hz, distorted space). The per-frame
    #    fields stay on the host; the GPU only ever sees a time-chunk (step 2).
    unwrapped_np = None
    est = warpkit_field_native(
        phase,
        mag,
        affine,
        tes_np,
        n_cpus=n_cpus,
        svd_filt=svd_filt,
        border_filt=border_filt,
        return_unwrapped=return_unwrapped,
        verbose=verbose,
    )
    if return_unwrapped:
        field_native_np, unwrapped_np = est
    else:
        field_native_np = est
    field_native = torch.nan_to_num(torch.from_numpy(np.ascontiguousarray(field_native_np)))
    unwrapped_field_hz = None
    if unwrapped_np is not None:
        # radians -> per-echo Hz, host-resident (nx,ny,nz,ne,T); several GiB at scale.
        unwrapped_field_hz = unwrapped_phase_to_field_hz(
            torch.nan_to_num(torch.from_numpy(np.ascontiguousarray(unwrapped_np))), tes_np
        )
    masks_t = torch.from_numpy(
        compute_frame_masks(mag, tes_np, automask_dilation, device=device, verbose=verbose)
    )
    if _dbg is not None:
        _dbg("fieldmap_native", field_native_np)
        _dbg("mask", masks_t.numpy())

    # 2. our warp: field -> PE displacement -> invert (pull) -> undistorted field,
    #    chunked over time on ``device`` with host-resident results (see helper).
    disp_pull, field_undist = field_to_pull_warp(
        field_native, total_readout_time, pe_dir, pe_axis, device, verbose=verbose
    )
    if _dbg is not None:
        _dbg("disp_pull_vox", disp_pull.numpy())
        # disp_native is just field x readout (elementwise); recompute on host for QC.
        _dbg(
            "disp_native_vox",
            field_to_displacement_pe(field_native, total_readout_time, pe_dir).numpy(),
        )

    return MedicResult(
        field_native=field_native,
        displacement_pe=disp_pull,
        field_undistorted=field_undist,
        masks=masks_t,
        pe_tensor_axis=pe_axis,
        unwrapped_field_hz=unwrapped_field_hz,
    )


def save_medic_warp(
    disp_pull: Tensor,
    pe_nifti_axis: int,
    affine: np.ndarray,
    prefix_stem: str,
    nii_ext: str = ".nii.gz",
    as_5d: bool = False,
    extra_components: list[tuple[Tensor, int]] | None = None,
) -> str:
    """Write the per-frame distortion warp as an ffs_qwarp-compatible mm warp.

    ffs_nwarp then composes it (with ``-master``) into a final single-resample
    chain to atlas, exactly like a static warp.

    On-disk convention is AFNI DICOM-mm. The per-frame path routes through
    ``save_warp_field(units="mm")``, which now does the RAS->DICOM x,y negation
    itself, so we feed the raw pull displacement. The 5D path bypasses
    save_warp_field (it writes one 5D file directly), so it applies the RAS->DICOM
    negation inline via ``feed_sign``. Verified against :func:`undistort_series`.

    ``extra_components`` adds further ``(disp_pull, nifti_axis)`` displacements on
    other axes — e.g. a dual-phase-encode acquisition warped on two in-plane axes at
    once. Each is converted with its own axis sign/scale. Honoured by BOTH on-disk
    formats; the per-frame path used to ignore it silently.

    Returns the path/glob to pass to ``ffs_nwarp -nwarp`` (a ``warp_*`` wildcard
    for the default per-frame files, or the single 5D file when ``as_5d``).
    """
    import os

    from ..io.afni import save_nifti
    from .io import save_warp_field
    from .nwarpforge import compute_cardinal_affine

    cardinal = compute_cardinal_affine(affine)
    disp_np = disp_pull.detach().cpu().numpy()  # (nx,ny,nz,T) raw pull, voxels
    nx, ny, nz, nt = disp_np.shape

    if as_5d:
        # Direct 5D write (no save_warp_field): convert RAS-mm -> DICOM-mm inline.
        # RAS-mm = cardinal[pe,pe]*disp; DICOM negates x,y (feed_sign).
        warp = np.zeros((nx, ny, nz, nt, 3), dtype=np.float32)
        for comp, axis in [
            (disp_np, pe_nifti_axis),
            *[(c.detach().cpu().numpy(), a) for c, a in (extra_components or [])],
        ]:
            feed_sign = -1.0 if axis in (0, 1) else 1.0
            rpp = float(cardinal[axis, axis])  # signed voxel mm on this axis
            warp[..., axis] = rpp * feed_sign * comp
        path = f"{prefix_stem}_warp{nii_ext}"
        save_nifti(warp, path, affine=cardinal)
        return path

    warp_dir = f"{prefix_stem}_warp"
    os.makedirs(warp_dir, exist_ok=True)
    # Every component, exactly as the 5-D branch does. This used to write the PE axis
    # alone and drop `extra_components` silently, so a dual-encode or rotation-aware
    # run wrote a warp that was missing an axis -- with no error, and a corrected
    # series that was nonetheless right, which is what made it invisible.
    extras_np = [(c.detach().cpu().numpy(), a) for c, a in (extra_components or [])]
    for t in range(nt):
        comp = torch.from_numpy(disp_np[:, :, :, t]).permute(2, 1, 0).contiguous()
        xyz = [torch.zeros_like(comp) for _ in range(3)]
        xyz[pe_nifti_axis] = comp  # (nz,ny,nx) voxel disp on the PE axis
        for extra, axis in extras_np:
            # Accumulate rather than assign: nothing in this codebase currently emits
            # two components on one axis, but summing is the correct composition if it
            # ever does, and assignment would silently keep only the last.
            xyz[axis] = xyz[axis] + torch.from_numpy(extra[:, :, :, t]).permute(2, 1, 0)
        save_warp_field(
            xyz[0],
            xyz[1],
            xyz[2],
            os.path.join(warp_dir, f"warp_{t:05d}{nii_ext}"),
            header_info={"affine": cardinal, "header": None},
            units="mm",
        )
    return os.path.join(warp_dir, f"warp_*{nii_ext}")


def detrend_time(field: Tensor, order: int = 1) -> Tensor:
    """Remove the per-voxel mean + polynomial trend along time (Legendre basis).

    For a per-frame field map ``(nx, ny, nz, T)``, regress each voxel's time series
    on Legendre polynomials up to ``order`` (0 = demean, 1 = demean + linear detrend)
    and return the residual — the oscillation about the slow trend. Reuses
    :func:`glm.core.construct_polynomial_matrix` (orthogonal Legendre, AFNI-style)
    so the drift model matches the rest of the codebase.

    ``order < 0`` is a no-op: the raw field map is returned unchanged (use this to
    drive the slice shift off the original field map, mean and trend included).
    """
    if order < 0:
        return field.clone()

    from ..glm.core import construct_polynomial_matrix

    nx, ny, nz, nt = field.shape
    p = construct_polynomial_matrix(nt, order, field.device, field.dtype)  # (T, order+1)
    y = field.reshape(-1, nt).t()  # (T, V)
    betas = torch.linalg.lstsq(p, y).solution
    resid = y - p @ betas
    return resid.t().reshape(nx, ny, nz, nt)


def field_temporal_change(field: Tensor, use_interp: bool = False) -> Tensor:
    """Per-frame field CHANGE (Hz) that drives 2nd-PE (slice/partition) distortion.

    In a 3-D / slice-partition-encoded EPI the slow phase-encode axis winds phase
    across the whole volume TR, so a field that DRIFTS between frames displaces the
    reconstructed data along that axis. The quantity that sets the local distortion
    is therefore how much the field changed by the moment each frame was acquired —
    not the field value itself (that is the primary in-plane correction's job).

    ``use_interp=False`` (default): backward difference ``d[t] = field[t] - field[t-1]``
    — the raw change that "led to" frame ``t``. Frame 0 has no predecessor, so
    ``d[0] = 0`` (the first frame is uncorrectable, by construction).

    ``use_interp=True``: first interpolate the field onto the acquisition MIDPOINTS
    (``m[t] = ½(field[t-1]+field[t])`` — the linear slicetime 'tween', the field as it
    was mid-acquisition) and difference those. A smoother, centered estimate of the
    same change: ``d[t] = ½(field[t]-field[t-2])`` for ``t≥2``, ``d[1]=½(field[1]-field[0])``.

    Returns a ``(nx, ny, nz, T)`` change field aligned to the acquired frames (Hz).
    Multiply by echo time (s) and a voxels-per-Hz·s scale to get the k-axis voxel shift.
    """
    src = field
    if use_interp:
        # Backward midpoints m[t]=½(field[t-1]+field[t]); m[0]:=field[0] (no predecessor).
        m = field.clone()
        m[..., 1:] = 0.5 * (field[..., :-1] + field[..., 1:])
        src = m
    d = torch.zeros_like(field)
    d[..., 1:] = src[..., 1:] - src[..., :-1]
    return d


def undistort_series(
    series: Tensor,
    disp_pull: Tensor,
    pe_nifti_axis: int,
    interp: str = "wsinc5",
    circular: bool = False,
    verbose: bool = True,
    desc: str = "undistort",
    extra_disp: Tensor | None = None,
    extra_nifti_axis: int | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Apply the per-frame undistortion pull warp to a 4D series, in native space.

    The displacement is along the phase-encode axis only and lives on the data's
    own grid, so this is a single high-quality interpolation per frame — no grid
    change. Distortion correction is intrinsically a native-space operation;
    motion / coregistration / atlas come afterward as affines (compose those with
    ffs_nwarp -master).

    ``series``, ``disp_pull`` and the output stay on whatever device they arrive on
    (the host, for a multi-GiB series): each frame is the natural work unit, so the
    warp is done one frame at a time on ``device`` (default: ``series``'s own device)
    and copied back. This streaming is what keeps a giant series off the GPU — only a
    single frame's tensors are device-resident at once.

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
    extra_disp : Tensor or None
        Optional second per-frame displacement (nx, ny, nz, T) on ``extra_nifti_axis``,
        applied in the *same* resample as ``disp_pull`` so there is no double
        interpolation. Used by the 3D-EPI slice-direction debug (``-debug_3d``).
    extra_nifti_axis : int or None
        Axis for ``extra_disp`` (0=i, 1=j, 2=k).
    device : torch.device or None
        Compute device for the per-frame warp. Default: ``series``'s own device.
        Pass a GPU here with a host-resident ``series`` to stream frame-by-frame.

    Returns
    -------
    Tensor
        (nx, ny, nz, T) undistorted series.
    """
    from .interp import warp_image_linear, warp_image_wsinc5

    if (extra_disp is None) != (extra_nifti_axis is None):
        raise ValueError("extra_disp and extra_nifti_axis must be given together")

    wfun = warp_image_wsinc5 if interp == "wsinc5" else warp_image_linear
    cdev = device if device is not None else series.device
    nt = series.shape[-1]
    out = torch.empty_like(series)  # follows series' device (host for a big series)
    it = tqdm(range(nt), desc=desc, disable=not verbose or nt == 1, leave=True)
    for t in it:
        # (nx,ny,nz) NIfTI order -> (nz,ny,nx) for the warp kernels; stream to cdev.
        d = disp_pull[..., t].permute(2, 1, 0).contiguous().to(cdev)
        zero = torch.zeros_like(d)
        comps = [zero, zero, zero]
        comps[pe_nifti_axis] = d  # xd/yd/zd indexed by i/j/k
        if extra_disp is not None and extra_nifti_axis is not None:
            ed = extra_disp[..., t].permute(2, 1, 0).contiguous().to(cdev)
            comps[extra_nifti_axis] = comps[extra_nifti_axis] + ed
        vol = series[..., t].permute(2, 1, 0).contiguous().to(cdev)
        if circular:
            cos_w = wfun(torch.cos(vol), comps[0], comps[1], comps[2])
            sin_w = wfun(torch.sin(vol), comps[0], comps[1], comps[2])
            out[..., t] = torch.atan2(sin_w, cos_w).permute(2, 1, 0).to(out.device)
        else:
            warped = wfun(vol, comps[0], comps[1], comps[2])
            out[..., t] = warped.permute(2, 1, 0).to(out.device)
    return out
