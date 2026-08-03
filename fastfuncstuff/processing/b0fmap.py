"""B0 fieldmap (dual-echo GRE) -> off-resonance field -> PE undistortion warp.

The other half of susceptibility distortion correction from :mod:`.topup`
(``ffs_blipflip``). There the field is *estimated* from two EPIs with opposing
phase-encode polarity; here it is *measured* directly from the phase evolution
between two gradient-echo acquisitions at different echo times::

    f[Hz] = (phi(TE2) - phi(TE1)) / (2*pi * (TE2 - TE1))

which is only that simple once the phase is unwrapped. Unwrapping is the whole
difficulty (see the SPM FieldMap toolbox's ``FieldMap_principles.md``), and we do
not reimplement it: ROMEO (Dymerska et al. 2021, MRM 85:2294-2308; the ``romeo``
binary from MRItools) already does the unwrap, the multi-echo SNR-weighted B0
combination, the robust magnitude mask, and the global n*2pi offset correction.
ROMEO is an established dependency of this repo -- ``ffs_autoproc`` already shells
out to it for temporal phase unwrapping.

What is left for us, and what this module is:

* marshalling the BIDS ``fmap/`` forms (``phase1``/``phase2``, ``phasediff``, or a
  ready-made Hz map) into the one 4-D call ROMEO wants,
* **conditioning** the field where ROMEO's mask ends -- a measured field is only
  defined over tissue, and a hard mask edge is a cliff in the field and therefore
  a tear in the warp. We extrapolate smoothly outward past the mask and then taper
  to zero in far air (the same reasoning as ``topup.taper_field_to_object``),
* Hz -> PE voxel displacement -> the 4-D mm pull warp that ``ffs_nwarp`` composes,
  in exactly the convention ``ffs_blipflip`` writes,
* the geometry hand-off to the EPI. A GRE fieldmap is measured in *undistorted*
  space while the EPI is distorted, so -- following SPM's ``FieldMap('MatchVDM')``
  -- the magnitude is first forward-warped by the field into a synthetic
  "distorted magnitude" that actually resembles an EPI, and *that* is what gets
  affine-registered to the EPI.

Note there is no field inversion anywhere here. SPM inverts the voxel displacement
map only for ``epifm==1``, i.e. a fieldmap that was itself acquired with EPI and so
lives in distorted space. A GRE fieldmap is already in undistorted space, which is
precisely what a pull-resample ``undistorted(i) = distorted(i + disp)`` wants.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .cost import _separable_smooth_3d
from .medic import PE_AXIS_MAP, field_to_displacement_pe, invert_displacement_pe
from .topup import _NIFTI_AXIS_TO_TDIM

__all__ = [
    "B0FieldResult",
    "RomeoOutputs",
    "condition_field",
    "extrapolate_outward",
    "field_to_pe_warp",
    "read_echo_times",
    "romeo_available",
    "run_romeo",
    "synthesize_distorted",
]


# ---------------------------------------------------------------------------
# ROMEO
# ---------------------------------------------------------------------------
def romeo_available(romeo_bin: str = "romeo") -> bool:
    """True if the ROMEO binary is on ``PATH`` (or is a path to an executable)."""
    return shutil.which(romeo_bin) is not None


@dataclass
class RomeoOutputs:
    """What ROMEO wrote, loaded back as (nx, ny, nz) numpy arrays."""

    b0_hz: np.ndarray
    mask: np.ndarray  # bool
    snr: np.ndarray | None = None  # B0_snr.nii, the per-voxel B0 SNR
    quality: np.ndarray | None = None  # quality.nii, ROMEO's unwrap quality [0,1]
    outdir: Path | None = None


def run_romeo(
    phase: np.ndarray,
    magnitude: np.ndarray,
    tes_ms: list[float],
    affine: np.ndarray,
    *,
    outdir: str | Path | None = None,
    romeo_bin: str = "romeo",
    mask: str = "robustmask",
    correct_global: bool = True,
    phase_offset_correction: bool = True,
    phase_units: str = "auto",
    phase_range: tuple[float, float] | None = None,
    extra_args: list[str] | None = None,
    verbose: bool = True,
) -> RomeoOutputs:
    """Unwrap phase and compute a B0 map in Hz with ROMEO.

    ``phase`` and ``magnitude`` are ``(nx, ny, nz)`` or ``(nx, ny, nz, ne)``; ``tes_ms``
    has one echo time (ms) per echo.

    Phase is converted to radians **here**, not by ROMEO, and ``--no-phase-rescale`` is
    always passed. ROMEO's own rescale pools the whole 4-D: measured directly, holding
    one echo fixed and merely widening another echo's range halved the first echo's
    scaling (6.2832 -> 3.1424 rad). Scaling has to be per echo, and radians must not be
    rescaled at all.

    ``phase_units``: ``"radians"`` passes the data through untouched; ``"scanner"``
    maps each echo's own ``[min, max]`` onto the circle, **independently per echo**,
    because the reconstruction writes every phase image filling its full quantisation
    range -- the range belongs to the image, not the series. ``"auto"`` (default) picks
    by data range, treating anything within +/-pi as radians. ``phase_range`` pins an
    explicit ``(lo, hi)`` for every echo, for the rare input that does not fill its
    range (a pre-masked phase image).

    The single-echo form is how a Siemens ``phasediff`` image is handled: the volume
    already *is* the inter-echo phase difference, so passing ``tes_ms=[dTE]`` makes
    ROMEO's ``phi / (2*pi*TE)`` the correct ``dphi / (2*pi*dTE)``.

    ``correct_global`` (ROMEO ``-g``) removes the global n*2pi offset, which otherwise
    shows up as a constant Hz bias and hence a constant PE shift of the whole brain.
    ``phase_offset_correction`` (MCPC3D-S) is meaningful only for >= 2 echoes and is
    silently skipped for one.
    """
    if not romeo_available(romeo_bin):
        raise RuntimeError(
            f"ROMEO binary {romeo_bin!r} not found on PATH. It ships with MRItools "
            "(https://github.com/korbinian90/MRItools/releases); ffs_autoproc's phase "
            "unwrapping stage needs it too."
        )
    phase = np.asarray(phase, dtype=np.float32)
    magnitude = np.asarray(magnitude, dtype=np.float32)
    if phase.ndim == 3:
        phase = phase[..., None]
    if magnitude.ndim == 3:
        magnitude = magnitude[..., None]
    ne = phase.shape[3]
    if len(tes_ms) != ne:
        raise ValueError(f"{ne} phase echoes but {len(tes_ms)} echo times")
    if magnitude.shape[3] not in (1, ne):
        raise ValueError(f"magnitude has {magnitude.shape[3]} echoes, expected 1 or {ne}")
    if magnitude.shape[3] == 1 and ne > 1:
        magnitude = np.repeat(magnitude, ne, axis=3)

    obs_lo, obs_hi = float(np.nanmin(phase)), float(np.nanmax(phase))
    if phase_units == "auto":
        phase_units = (
            "radians" if (obs_lo >= -np.pi * 1.01 and obs_hi <= np.pi * 1.01) else "scanner"
        )
    elif phase_units not in ("radians", "scanner"):
        raise ValueError(f"phase_units must be auto/radians/scanner, got {phase_units!r}")

    if phase_units == "scanner":
        # Per echo, each from its OWN min/max -- the reconstruction writes every phase
        # image filling its full quantisation range, so the range is a property of the
        # image, not of the series. This is SPM's conversion verbatim
        # (``pm_scale_phase.m``: ``-pi + (vol-mn)*2*pi/(mx-mn)``).
        for i in range(ne):
            e = phase[..., i]
            lo, hi = phase_range if phase_range is not None else (float(e.min()), float(e.max()))
            if hi <= lo:
                raise ValueError(f"echo {i}: phase is constant ({lo:g}); cannot scale it")
            phase[..., i] = (e - lo) / (hi - lo) * (2.0 * np.pi) - np.pi
            if verbose:
                print(f"  phase echo {i}: [{lo:g}, {hi:g}] -> [-pi, pi]")

    import nibabel as nib

    tmp = None
    if outdir is None:
        tmp = tempfile.TemporaryDirectory(prefix="ffs_b0fmap_")
        work = Path(tmp.name)
    else:
        work = Path(outdir)
        work.mkdir(parents=True, exist_ok=True)

    try:
        # ROMEO reads files, and cannot read .zst; plain .nii keeps it fastest.
        p_path, m_path = work / "phase.nii", work / "mag.nii"
        nib.save(nib.Nifti1Image(phase.squeeze(-1) if ne == 1 else phase, affine), p_path)
        nib.save(nib.Nifti1Image(magnitude.squeeze(-1) if ne == 1 else magnitude, affine), m_path)
        out = work / "romeo"
        cmd = [
            romeo_bin,
            "-p",
            str(p_path),
            "-m",
            str(m_path),
            "-t",
            "[" + ",".join(f"{t:g}" for t in tes_ms) + "]",
            "-o",
            str(out),
            "-B",  # combined B0 map in Hz -- the whole point
            "-k",
            mask,
            "--no-phase-rescale",  # we converted to radians ourselves, above
            "-q",  # quality map, reused below as the smoothing weight
        ]
        # -u masks the unwrapped result, but ROMEO documents that -u with "nomask"
        # *substitutes* robustmask -- so passing it unconditionally silently masks the
        # output of a run that explicitly asked for no masking.
        if mask != "nomask":
            cmd.append("-u")
        if correct_global:
            cmd.append("-g")
        if phase_offset_correction and ne > 1:
            cmd.append("--phase-offset-correction")
        if extra_args:
            cmd.extend(extra_args)
        if verbose:
            print("  " + " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"romeo failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
            )

        from ..io.afni import load_nifti

        def _load(name: str) -> np.ndarray | None:
            """Load a ROMEO output as 3-D. Some (B0_snr, quality) come back 4-D with a
            per-echo axis even for a single echo; average it away."""
            p = out / name
            if not p.exists():
                return None
            arr = np.asarray(load_nifti(str(p)).dataobj, dtype=np.float32)
            return arr.mean(axis=-1) if arr.ndim == 4 else arr

        b0 = _load("B0.nii")
        if b0 is None:
            raise RuntimeError(f"romeo wrote no B0.nii in {out} (stdout:\n{proc.stdout})")
        msk = _load("mask.nii")
        return RomeoOutputs(
            b0_hz=np.nan_to_num(b0, nan=0.0, posinf=0.0, neginf=0.0),
            mask=(msk > 0) if msk is not None else np.isfinite(b0),
            snr=_load("B0_snr.nii"),
            quality=_load("quality.nii"),
            outdir=None if tmp is not None else out,
        )
    finally:
        if tmp is not None:
            tmp.cleanup()


# ---------------------------------------------------------------------------
# Field conditioning: extrapolate past the mask, then taper to zero
# ---------------------------------------------------------------------------
def _weighted_smooth(
    field: Tensor, weight: Tensor, sigma_vox: tuple[float, float, float]
) -> Tensor:
    """Normalised-convolution smooth: ``G*(f*w) / G*w``, so zeros outside the
    weight do not bleed in and drag the edge of the field toward zero."""
    num = _separable_smooth_3d(field * weight, sigma_vox)
    den = _separable_smooth_3d(weight, sigma_vox)
    return torch.where(den > 1e-6, num / den.clamp_min(1e-6), field)


def extrapolate_outward(
    field: Tensor, mask: Tensor, n_iter: int, kernel: int = 3
) -> tuple[Tensor, Tensor]:
    """Grow the field ``n_iter`` voxels past ``mask`` by neighbourhood averaging.

    Each pass fills the one-voxel shell around the current support with the mean of
    its already-valid neighbours, then adopts that shell as valid. This is the fast
    GPU stand-in for solving a Laplace equation in the exterior: it is smooth, it is
    exact at the boundary, and it decays to the local field rather than to zero.

    The alternative -- leaving the field at zero outside the mask -- puts a step of
    tens of Hz right at the brain edge. Through :func:`field_to_pe_warp` that step is
    a discontinuous PE displacement, so the resample tears exactly where orbitofrontal
    and temporal signal (the voxels most in need of correction) lives.

    Returns the extended field and the extended (dilated) mask.
    """
    f = field.clone()
    m = mask.to(field.dtype).clone()
    pad = kernel // 2
    for _ in range(max(0, n_iter)):
        num = F.avg_pool3d(
            (f * m)[None, None], kernel_size=kernel, stride=1, padding=pad, count_include_pad=True
        )[0, 0]
        den = F.avg_pool3d(
            m[None, None], kernel_size=kernel, stride=1, padding=pad, count_include_pad=True
        )[0, 0]
        grown = (den > 0).to(field.dtype)
        shell = grown * (1.0 - m)
        f = f * m + shell * torch.where(den > 0, num / den.clamp_min(1e-8), torch.zeros_like(num))
        m = grown
    return f, m


def condition_field(
    field_hz: Tensor,
    mask: Tensor,
    voxel_sizes: tuple[float, float, float],
    *,
    weight: Tensor | None = None,
    fwhm_mm: float = 4.0,
    extend_mm: float = 16.0,
    rolloff_mm: float = 8.0,
) -> tuple[Tensor, Tensor]:
    """Smooth inside the mask, extrapolate outward, taper to zero in far air.

    ``voxel_sizes`` is ``(vz, vy, vx)`` to match the ``(nz, ny, nx)`` tensor layout.
    ``weight`` (ROMEO's B0 SNR or quality map) makes the smoothing inverse-noise
    weighted the way SPM's ``pm_smooth_phasemap`` is; without it the mask alone is
    the weight.

    ``fwhm_mm`` defaults far below SPM's 10 mm on purpose: SPM smooths a raw
    single-pair phase difference, whereas ROMEO's ``-B`` output is already an
    SNR-weighted combination across echoes, so heavy smoothing here only blurs away
    the sharp sinus gradients that matter.

    Returns ``(conditioned_field, support_mask)`` where the support mask is the
    extended (pre-taper) one -- useful as the "field is trustworthy here" map.
    """
    m = mask.to(field_hz.dtype)
    w = m if weight is None else (m * weight.clamp_min(0.0).to(field_hz.dtype))

    sigma_vox = tuple((fwhm_mm / 2.354820045) / v for v in voxel_sizes)  # FWHM -> sigma, per axis
    if fwhm_mm > 0:
        field_hz = _weighted_smooth(field_hz * m, w, sigma_vox)

    mean_vox = sum(voxel_sizes) / 3.0
    n_ext = int(round(extend_mm / mean_vox))
    field_hz, support = extrapolate_outward(field_hz * m, m > 0.5, n_ext)

    if rolloff_mm > 0:
        env = _separable_smooth_3d(support.to(field_hz.dtype), rolloff_mm / mean_vox)
        field_hz = field_hz * env.clamp(0.0, 1.0)
    return field_hz, support > 0.5


# ---------------------------------------------------------------------------
# Field -> warp, and the synthetic distorted magnitude
# ---------------------------------------------------------------------------
@dataclass
class B0FieldResult:
    """A conditioned off-resonance field and everything derived from it.

    All volumes are ``(nz, ny, nx)`` tensors in the fieldmap's own grid unless
    they have been moved onto the EPI grid by the caller.
    """

    field_hz: Tensor
    field_raw_hz: Tensor
    mask: Tensor  # ROMEO's mask
    support: Tensor  # mask after extrapolation
    magnitude: Tensor  # reference (first-echo) magnitude
    affine: np.ndarray


def field_to_pe_warp(field_hz: Tensor, readout_s: float, pe_dir: str) -> tuple[Tensor, Tensor, int]:
    """Hz field -> (pull warp, inverse warp, PE tensor dim), all in voxel units.

    The pull warp is the ``ffs_blipflip`` / ``ffs_nwarp`` convention
    ``undistorted(i) = distorted(i + warp)``. A GRE field is measured in undistorted
    space, so the displacement *is* the pull warp -- no inversion (see module
    docstring). The returned inverse maps undistorted -> distorted, which is what
    :func:`synthesize_distorted` needs.
    """
    if pe_dir not in PE_AXIS_MAP:
        raise ValueError(f"bad phase-encode direction {pe_dir!r}")
    pe_tdim = _NIFTI_AXIS_TO_TDIM[PE_AXIS_MAP[pe_dir]]
    pull = field_to_displacement_pe(field_hz, readout_s, pe_dir)
    return pull, invert_displacement_pe(pull, pe_tdim), pe_tdim


def synthesize_distorted(vol: Tensor, inv_disp: Tensor, pe_tdim: int) -> Tensor:
    """Push an undistorted volume through the field to look like a distorted EPI.

    This is SPM's forward-warped magnitude (``wfmag_``), and it exists for one
    reason: the fieldmap magnitude is geometrically faithful while the EPI is not,
    so registering them directly asks an affine to absorb a nonlinear susceptibility
    warp. Distorting the magnitude first leaves the affine only the rigid-body
    difference it can actually represent.

    SPM approximates the inverse displacement by negating it; we use the true 1-D
    inverse from ``invert_displacement_pe``, which is exact where the warp is
    monotone and costs one fixed-point loop.
    """
    from .topup import _resample_pe

    return _resample_pe(vol, inv_disp, pe_tdim)


# ---------------------------------------------------------------------------
# BIDS sidecar helpers
# ---------------------------------------------------------------------------
def _sidecar(path: str | Path) -> dict:
    """The JSON sidecar next to a NIfTI, or ``{}``."""
    p = Path(str(path).split("[")[0])
    name = p.name
    for ext in (".nii.gz", ".nii.zst", ".nii"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    j = p.with_name(name + ".json")
    if j.exists():
        with open(j) as fh:
            return json.load(fh)
    return {}


def read_echo_times(paths: list[str], *, phasediff: bool = False) -> list[float] | None:
    """Echo times in **ms** from BIDS sidecars, or ``None`` if unavailable.

    For a ``phasediff`` image BIDS stores ``EchoTime1``/``EchoTime2`` on the one
    sidecar and the meaningful quantity is their difference, so a single-element
    list ``[dTE]`` comes back -- exactly what :func:`run_romeo` wants for that form.
    """
    if phasediff:
        js = _sidecar(paths[0])
        t1, t2 = js.get("EchoTime1"), js.get("EchoTime2")
        if t1 is None or t2 is None:
            return None
        return [abs(float(t2) - float(t1)) * 1000.0]
    tes = []
    for p in paths:
        te = _sidecar(p).get("EchoTime")
        if te is None:
            return None
        tes.append(float(te) * 1000.0)
    return tes


def read_epi_geometry(path: str) -> tuple[str | None, float | None]:
    """``(PhaseEncodingDirection, TotalReadoutTime)`` from an EPI's BIDS sidecar.

    ``TotalReadoutTime`` is preferred; ``EffectiveEchoSpacing`` times the number of
    PE lines minus one is the BIDS-sanctioned fallback for datasets that only
    recorded the former.
    """
    js = _sidecar(path)
    pe = js.get("PhaseEncodingDirection")
    trt = js.get("TotalReadoutTime")
    if trt is None and "EffectiveEchoSpacing" in js and "ReconMatrixPE" in js:
        trt = float(js["EffectiveEchoSpacing"]) * (int(js["ReconMatrixPE"]) - 1)
    return (str(pe) if pe else None), (float(trt) if trt is not None else None)
