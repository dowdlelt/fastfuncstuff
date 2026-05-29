"""Main warping engine - the heart of qwarp_torch.

This module implements IW3D_warpomatic() and IW3D_improve_warp() from
AFNI's mri_nwarp.c in PyTorch, with GPU-parallel batch processing.

Algorithm:
1. Level 0: global warp (full image patch, progressive basis complexity)
2. Levels 1..N: shrinking patches with 50% overlap
3. Within each level: 3D checkerboard decomposition into 8 phases
4. Within each phase: ALL non-overlapping patches optimized in parallel
   on GPU using batched Adam optimizer with autograd
5. After each phase: update global warp, proceed to next phase

Key speedups vs serial version:
  - Checkerboard batching: 8-50x (process B patches per GPU call)
  - Adam optimizer: 3-5x fewer evaluations than Powell (autograd gradients)
  - Fused 3-channel grid_sample: 2-3x less kernel overhead
  - No CPU-GPU sync in hot loop (all tensor ops, no .item())
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import Tensor

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

from .basis import (
    build_3d_basis_cubic,
    build_3d_basis_quintic,
    compute_half_widths_cubic,
    evaluate_patch_warp,
    evaluate_patch_warp_batched,
)
from .cost import (
    BatchedIncrementalCorrelation,
    IncrementalCorrelation,
    _auto_clip,
    batched_lpa_cost,
)
from .interp import (
    batched_compose_and_interpolate,
    trilinear_interpolate,
    warp_image_linear,
)
from .optimizer import optimize_warp_params_batched, optimize_warp_params_torch
from .penalty import compute_jacobian_energy, compute_penalty_batched
from .weight import compute_weight_image

# Cache for torch.compile'd building-block functions (stable identity, compiled once)
_compile_cache: dict[str, Callable[..., Any]] = {}


def _maybe_compile(fn: Callable[..., Any], name: str, device: torch.device, do_compile: bool) -> Callable[..., Any]:
    """Return a compiled version of fn for CUDA, caching by name."""
    if device.type != "cuda" or not do_compile:
        return fn
    if name not in _compile_cache:
        _compile_cache[name] = torch.compile(fn, dynamic=True)
    return _compile_cache[name]


@dataclass
class QwarpConfig:
    """Configuration for the qwarp algorithm."""

    minpatch: int = 25
    """Minimum patch size (odd number, >= 5). Controls detail level."""

    blur_base: float = 0.0
    """Gaussian FWHM blur for base image (voxels)."""

    blur_source: float = 0.0
    """Gaussian FWHM blur for source image (voxels)."""

    pblur_base: float = 0.0
    """Progressive blur fraction for base (0 = off)."""

    pblur_source: float = 0.0
    """Progressive blur fraction for source (0 = off)."""

    use_quintic: bool = False
    """Use quintic (5th order) basis at final level."""

    use_lite: bool = True
    """Use 'lite' (reduced parameter) basis functions."""

    workhard: tuple[int, int] = (0, -1)
    """(start_lev, end_lev) range for double-pass optimization."""

    cost_method: str = "pearclp"
    """Cost function: 'pearclp' (clipped Pearson), 'pearson', or 'lpa'."""

    penalty_factor: float = 0.033
    """Base Jacobian-energy penalty factor. Matches AFNI's Hpen_fbase=0.033."""

    penalty_first_level: int = 3
    """First level to apply penalty (no penalty before this). Matches AFNI's
    Hpen_first_lev=3; on first activation the factor is halved."""

    shrink: float = 0.749999
    """Patch shrinkage factor between levels."""

    max_level: int = 666
    """Maximum refinement level."""

    start_level: int = 0
    """Starting level (0 = global)."""

    warp_flags: int = 0
    """Bit flags: 1=no-x-disp, 2=no-y-disp, 4=no-z-disp."""

    axis_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Per-axis displacement scaling (0-1). E.g., (0.2, 0.8, 0.3) = mostly Y."""

    verb: int = 1
    """Verbosity level (0=quiet, 1=normal, 2=detailed)."""

    batch_optimizer_lr: float = 0.008
    """Learning rate for the batched Adam optimizer."""

    batch_optimizer_iters: int = 60
    """Max iterations for batched Adam optimizer per phase."""

    batch_optimizer_iters_lev0: int = 120
    """Max Adam iterations for the single global level-0 patch (CUDA only).
    Level 0 is one B=1 patch and the foundational alignment, so it gets a
    larger budget than the per-phase fine patches. CPU still uses the serial
    derivative-free optimizer (no launch overhead to amortize there)."""

    hfactor_q: float = 0.5
    """AFNI-style Hfactor shrinkage for *fine* patches: at the lev=1 patch size
    Hfactor=1.0, and as patches shrink Hfactor decreases toward hfactor_q.
    This tightens the per-patch displacement bound at deep levels, which is
    AFNI's primary defense against high-frequency over-warping. 1.0 disables
    the mechanism. Range [0.1, 1.0]."""

    maxdisp: float = 0.0
    """Maximum allowed displacement in voxels. 0 = no limit (default).
    If set, displacement fields are clamped after each level."""

    lpa_sigma: float = 4.0
    """Kernel parameter (voxels) for LPA local neighborhoods.
    For ``lpa_kernel="gauss"``: Gaussian sigma (effective radius ~3*sigma).
    For ``lpa_kernel="box"``: half-width radius (cube side = 2*r+1)."""

    lpa_kernel: str = "gauss"
    """Kernel type for LPA neighborhoods: ``"gauss"`` or ``"box"``."""

    level_stop_tol: float = 0.0
    """Early stopping: if a level improves cost by less than this fraction, stop
    refining further. 0 = disabled (default). E.g. 0.0001 stops when improvement
    drops below 0.01% of current cost."""

    compile: bool = False
    """Use torch.compile for building-block functions (CUDA only).
    Requires warmup on first volume; may not help for small patches."""

    pyramid_factor: int = 1
    """Coarse-to-fine resolution pyramid factor (1 = off, the default).
    When >1, the coarse levels are solved on a volume downsampled by this
    factor per axis, then the warp is upsampled to seed full-resolution
    refinement of the fine levels. The coarse levels are the GPU-compute
    bottleneck on large (e.g. 1mm anatomical) volumes -- an N× downsample is
    ~N³ less work there. Opt-in because aggressive downsampling can change the
    warp; validate against the non-pyramid result. Only applies when
    start_level==0 and no initial warp is supplied."""

    level_callback: Callable[..., None] | None = None
    """Optional callback fired after each level completes. Signature:
    ``cb(level: int, xd: Tensor, yd: Tensor, zd: Tensor, warped_source: Tensor)``.
    Tensors are on the padded grid in whatever device the warp ran on; the
    callee is responsible for cropping and moving to CPU if needed. Used to
    save per-level intermediate warps and warped images."""


@dataclass
class WarpState:
    """Internal state during warp optimization."""

    xd: Tensor = field(default_factory=lambda: torch.empty(0))
    yd: Tensor = field(default_factory=lambda: torch.empty(0))
    zd: Tensor = field(default_factory=lambda: torch.empty(0))
    warped_source: Tensor = field(default_factory=lambda: torch.empty(0))

    nx: int = 0
    ny: int = 0
    nz: int = 0
    cost: float = 666.666
    patches_done: int = 0
    patches_skipped: int = 0
    last_level: int = 0  # highest refinement level executed (for pyramid hand-off)


@dataclass
class PatchSpec:
    """Specification for one patch in the grid."""
    ibot: int
    itop: int
    jbot: int
    jtop: int
    kbot: int
    ktop: int
    gi: int  # grid index for checkerboard coloring
    gj: int
    gk: int


def _compute_padding(nx: int, ny: int, nz: int) -> tuple[int, int, int]:
    """Compute AFNI-style internal padding per axis.

    Uses ~12.34% of each dimension + 1, with minimum of 3 voxels per side.
    """
    import math
    pad_x = max(3, int(math.ceil(0.1234 * nx)) + 1)
    pad_y = max(3, int(math.ceil(0.1234 * ny)) + 1)
    pad_z = max(3, int(math.ceil(0.1234 * nz)) + 1)
    return pad_x, pad_y, pad_z


def _pad_volume(vol: Tensor, pad_x: int, pad_y: int, pad_z: int) -> Tensor:
    """Zero-pad a 3D volume symmetrically."""
    # F.pad order: (x_left, x_right, y_left, y_right, z_left, z_right)
    import torch.nn.functional as F
    return F.pad(vol, (pad_x, pad_x, pad_y, pad_y, pad_z, pad_z), mode='constant', value=0)


def qwarp(
    base: Tensor,
    source: Tensor,
    weight: Tensor | None = None,
    mask: Tensor | None = None,
    initial_warp: tuple[Tensor, Tensor, Tensor] | None = None,
    config: QwarpConfig | None = None,
    device: torch.device | None = None,
    pad: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute nonlinear warp from source to base image.

    This is the main entry point, equivalent to IW3D_warp_s2bim().

    AFNI-style internal padding is applied by default. The returned warp field
    covers the padded grid (larger than input), while the warped image is
    cropped back to the original size.

    Args:
        base: (nz, ny, nx) base/target image.
        source: (nz, ny, nx) source image to warp.
        weight: Optional (nz, ny, nx) weight image. Auto-generated if None.
        mask: Optional (nz, ny, nx) byte mask. Derived from weight if None.
        initial_warp: Optional (xd, yd, zd) tuple of initial displacement fields.
        config: QwarpConfig settings. Uses defaults if None.
        device: Torch device. Inferred from base if None.
        pad: Apply AFNI-style internal zero-padding (default True).

    Returns:
        (warped_image, warp_xd, warp_yd, warp_zd):
          - warped_image: (nz, ny, nx) source warped to match base (original size).
          - warp_xd/yd/zd: (nz_pad, ny_pad, nx_pad) displacement fields on
            padded grid, in voxel units. Larger than input to allow edge warping.
    """
    if config is None:
        config = QwarpConfig()
    if device is None:
        device = base.device

    # Enable TF32 for matmuls on Ampere+ GPUs (free perf, ~1e-5 precision)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    base = base.float().to(device)
    source = source.float().to(device)
    nz_orig, ny_orig, nx_orig = base.shape

    # Internal padding
    if pad:
        pad_x, pad_y, pad_z = _compute_padding(nx_orig, ny_orig, nz_orig)
    else:
        pad_x, pad_y, pad_z = 0, 0, 0

    if pad_x > 0 or pad_y > 0 or pad_z > 0:
        base_p = _pad_volume(base, pad_x, pad_y, pad_z)
        source_p = _pad_volume(source, pad_x, pad_y, pad_z)
        if config.verb >= 1:
            print(f"qwarp_torch: padding +{pad_x},{pad_y},{pad_z} => "
                  f"{base_p.shape[2]}x{base_p.shape[1]}x{base_p.shape[0]}")
    else:
        base_p = base
        source_p = source

    nz, ny, nx = base_p.shape

    if weight is None:
        weight_p = compute_weight_image(base_p)
    else:
        weight_p = _pad_volume(weight.float().to(device), pad_x, pad_y, pad_z) if (pad_x > 0 or pad_y > 0 or pad_z > 0) else weight.float().to(device)

    if mask is None:
        mask_p = (weight_p > 0).byte()
    else:
        mask_p = _pad_volume(mask.byte().to(device).float(), pad_x, pad_y, pad_z).byte() if (pad_x > 0 or pad_y > 0 or pad_z > 0) else mask.byte().to(device)

    state = WarpState(nx=nx, ny=ny, nz=nz)

    if initial_warp is not None:
        # Pad the initial warp if needed
        if pad_x > 0 or pad_y > 0 or pad_z > 0:
            state.xd = _pad_volume(initial_warp[0].float().to(device), pad_x, pad_y, pad_z)
            state.yd = _pad_volume(initial_warp[1].float().to(device), pad_x, pad_y, pad_z)
            state.zd = _pad_volume(initial_warp[2].float().to(device), pad_x, pad_y, pad_z)
        else:
            state.xd = initial_warp[0].float().to(device)
            state.yd = initial_warp[1].float().to(device)
            state.zd = initial_warp[2].float().to(device)
        state.warped_source = warp_image_linear(source_p, state.xd, state.yd, state.zd)
    else:
        state.xd = torch.zeros(nz, ny, nx, device=device)
        state.yd = torch.zeros(nz, ny, nx, device=device)
        state.zd = torch.zeros(nz, ny, nx, device=device)
        state.warped_source = source_p.clone()

    if config.pyramid_factor > 1 and initial_warp is None and config.start_level == 0:
        _warpomatic_pyramid(base_p, source_p, weight_p, mask_p, state, config, device)
    else:
        _warpomatic(base_p, source_p, weight_p, mask_p, state, config, device)

    # Final warped image: apply warp to unpadded source, crop to original size
    warped_full = warp_image_linear(source_p, state.xd, state.yd, state.zd)
    warped = warped_full[pad_z:pad_z+nz_orig, pad_y:pad_y+ny_orig, pad_x:pad_x+nx_orig]

    # Return full padded warp field (caller/io.py handles grid info)
    return warped, state.xd, state.yd, state.zd


# ---------------------------------------------------------------------------
# Patch grid and checkerboard decomposition
# ---------------------------------------------------------------------------

def _generate_patch_grid(
    ibbb: int, ittt: int, jbbb: int, jttt: int, kbbb: int, kttt: int,
    xwid: int, ywid: int, zwid: int,
    xdel: int, ydel: int, zdel: int,
) -> list[PatchSpec]:
    """Enumerate all patch positions in the sweep grid."""
    patches = []
    gi = 0
    ibot = ibbb
    while ibot <= ittt:
        itop = min(ibot + xwid - 1, ittt)
        if itop >= ittt:
            ibot = itop + 1 - xwid

        gj = 0
        jbot = jbbb
        while jbot <= jttt:
            jtop = min(jbot + ywid - 1, jttt)
            if jtop >= jttt:
                jbot = jtop + 1 - ywid

            gk = 0
            kbot = kbbb
            while kbot <= kttt:
                ktop = min(kbot + zwid - 1, kttt)
                if ktop >= kttt:
                    kbot = ktop + 1 - zwid

                patches.append(PatchSpec(
                    ibot=ibot, itop=itop,
                    jbot=jbot, jtop=jtop,
                    kbot=kbot, ktop=ktop,
                    gi=gi, gj=gj, gk=gk,
                ))

                if ktop >= kttt:
                    break
                kbot += zdel
                gk += 1

            if jtop >= jttt:
                break
            jbot += ydel
            gj += 1

        if itop >= ittt:
            break
        ibot += xdel
        gi += 1

    return patches


def _checkerboard_phases(
    patches: list[PatchSpec],
) -> list[list[PatchSpec]]:
    """Group patches into 8 checkerboard phases (non-overlapping within each)."""
    phases: list[list[PatchSpec]] = [[] for _ in range(8)]
    for p in patches:
        idx = (p.gi % 2) * 4 + (p.gj % 2) * 2 + (p.gk % 2)
        phases[idx].append(p)
    return phases


def _filter_patches(
    patches: list[PatchSpec],
    weight: Tensor, mask: Tensor,
    nx: int, ny: int, nz: int,
) -> list[PatchSpec]:
    """Filter out patches with insufficient weight/mask coverage."""
    w_avg = float(weight[mask > 0].mean().item()) if (mask > 0).any() else 1.0
    valid = []
    for p in patches:
        nxh = p.itop - p.ibot + 1
        nyh = p.jtop - p.jbot + 1
        nzh = p.ktop - p.kbot + 1
        n_voxels = nxh * nyh * nzh

        if nxh < 5 and nyh < 5 and nzh < 5:
            continue

        m_patch = mask[p.kbot:p.ktop+1, p.jbot:p.jtop+1, p.ibot:p.itop+1]
        w_patch = weight[p.kbot:p.ktop+1, p.jbot:p.jtop+1, p.ibot:p.itop+1]
        n_masked = int(m_patch.sum().item())
        w_sum = float(w_patch.sum().item())

        if n_masked < 0.333 * n_voxels or w_sum < 0.166 * n_voxels * w_avg:
            continue

        # Check base isn't constant in this patch
        valid.append(p)

    return valid


# ---------------------------------------------------------------------------
# Basis type configuration
# ---------------------------------------------------------------------------

def _get_basis_config(
    basis_type: str, nxh: int, nyh: int, nzh: int, device: torch.device,
    hfactor: float = 1.0,
):
    """Get basis functions and parameters for a given type.

    Uses AFNI's BOXOPT param_max values, scaled by hfactor (which decreases
    for larger patches relative to minpatch, following AFNI's Hfactor logic).
    """
    if basis_type == "cubic_lite":
        basis = build_3d_basis_cubic(nxh, nyh, nzh, device, lite=True)
        half_widths = compute_half_widths_cubic(nxh, nyh, nzh)
        param_max = 0.0421
    elif basis_type == "cubic":
        basis = build_3d_basis_cubic(nxh, nyh, nzh, device, lite=False)
        half_widths = compute_half_widths_cubic(nxh, nyh, nzh)
        param_max = 0.0280
    elif basis_type == "quintic_lite":
        basis = build_3d_basis_quintic(nxh, nyh, nzh, device, lite=True)
        half_widths = compute_half_widths_cubic(nxh, nyh, nzh)
        param_max = 0.0267
    elif basis_type == "quintic":
        basis = build_3d_basis_quintic(nxh, nyh, nzh, device, lite=False)
        half_widths = compute_half_widths_cubic(nxh, nyh, nzh)
        param_max = 0.0099
    else:
        raise ValueError(f"Unknown basis type: {basis_type}")

    param_max *= hfactor
    return basis, half_widths, param_max


def _compute_hfactor(patch_size: int, patch_size_lev1: int, hfactor_q: float = 0.5) -> float:
    """AFNI-style Hfactor scaling on param_max.

    AFNI's Hfactor_from_patchsize_ratio uses prat = psize / psize0 where
    psize0 is the lev=1 (coarsest non-global) patch size. At lev=1 prat=1
    so hfactor=1; at finer levels prat<1 so hfactor<1, tightening the
    per-patch displacement bound. hfactor = prat^alpha with
    alpha = log(hfactor_q) / log(0.1).
    """
    import math
    if hfactor_q >= 1.0 or hfactor_q < 0.1 or patch_size_lev1 <= 0:
        return 1.0
    if patch_size >= patch_size_lev1:
        return 1.0
    prat = patch_size / patch_size_lev1
    alpha = math.log(hfactor_q) / math.log(0.1)
    return prat ** alpha


def _global_correlation(
    base: Tensor,
    warped_source: Tensor,
    weight: Tensor,
    base_clip: tuple[float, float] | None,
    source_clip: tuple[float, float] | None,
) -> float:
    """Weighted (optionally clipped) Pearson cost over the whole volume.

    Returns the negated correlation (lower = better), matching state.cost
    convention. Single source of truth for the level-boundary cost readout.
    """
    b_flat = base.reshape(-1)
    s_flat = warped_source.reshape(-1)
    w_flat = weight.reshape(-1)
    if base_clip:
        b_flat = b_flat.clamp(base_clip[0], base_clip[1])
    if source_clip:
        s_flat = s_flat.clamp(source_clip[0], source_clip[1])
    wm = w_flat * (w_flat > 0).float()
    sw = wm.sum()
    if sw <= 0:
        return 0.0
    xbar = (wm * b_flat).sum() / sw
    ybar = (wm * s_flat).sum() / sw
    vxx = ((wm * b_flat * b_flat).sum() / sw - xbar * xbar).clamp(min=1e-20)
    vyy = ((wm * s_flat * s_flat).sum() / sw - ybar * ybar).clamp(min=1e-20)
    vxy = (wm * b_flat * s_flat).sum() / sw - xbar * ybar
    return float((-vxy / (vxx * vyy).sqrt()).item())


def _warpomatic_pyramid(
    base: Tensor, source: Tensor, weight: Tensor, mask: Tensor,
    state: WarpState, config: QwarpConfig, device: torch.device,
) -> None:
    """Coarse-to-fine resolution pyramid (opt-in via config.pyramid_factor).

    Solves the coarse levels on a volume downsampled by ``pyramid_factor`` per
    axis -- where the warp is smooth and low-frequency, so full resolution is
    wasted compute -- then upsamples the warp to seed full-resolution
    refinement of the fine levels. ``state`` is the full-resolution state
    (zero warp). All volumes are the padded full-resolution grid.
    """
    import torch.nn.functional as F

    f = config.pyramid_factor
    nz, ny, nx = base.shape
    lz = max(16, round(nz / f))
    ly = max(16, round(ny / f))
    lx = max(16, round(nx / f))

    def _down(v: Tensor) -> Tensor:
        return F.interpolate(
            v[None, None].float(), size=(lz, ly, lx), mode="trilinear", align_corners=False
        )[0, 0]

    base_lo = _down(base)
    source_lo = _down(source)
    weight_lo = _down(weight)
    mask_lo = (_down(mask) > 0.5).byte()

    lo_state = WarpState(nx=lx, ny=ly, nz=lz)
    lo_state.xd = torch.zeros(lz, ly, lx, device=device)
    lo_state.yd = torch.zeros(lz, ly, lx, device=device)
    lo_state.zd = torch.zeros(lz, ly, lx, device=device)
    lo_state.warped_source = source_lo.clone()

    if config.verb >= 1:
        print(f"qwarp_torch: pyramid coarse pass at {lx}x{ly}x{lz} (factor {f})")
    # pad=False semantics: these are already on the padded grid. Pyramid off in
    # the recursive call so it doesn't recurse further.
    lo_cfg = replace(config, pyramid_factor=1)
    _warpomatic(base_lo, source_lo, weight_lo, mask_lo, lo_state, lo_cfg, device)

    # Upsample the coarse warp to full resolution. Displacements are in voxel
    # units, so they scale by the per-axis resolution ratio (~f).
    sx, sy, sz = nx / lx, ny / ly, nz / lz

    def _up(v: Tensor) -> Tensor:
        return F.interpolate(
            v[None, None], size=(nz, ny, nx), mode="trilinear", align_corners=False
        )[0, 0]

    state.xd = _up(lo_state.xd) * sx
    state.yd = _up(lo_state.yd) * sy
    state.zd = _up(lo_state.zd) * sz
    state.warped_source = warp_image_linear(source, state.xd, state.yd, state.zd)

    # Resume full-resolution refinement where the coarse pass left off, with a
    # one-level overlap so no spatial-frequency band is skipped at the seam.
    start = max(1, lo_state.last_level - 1)
    if config.verb >= 1:
        print(f"qwarp_torch: pyramid full-res refine from lev={start}")
    refine_cfg = replace(config, pyramid_factor=1, start_level=start)
    _warpomatic(base, source, weight, mask, state, refine_cfg, device)


# ---------------------------------------------------------------------------
# Main warpomatic loop
# ---------------------------------------------------------------------------

def _warpomatic(
    base: Tensor, source: Tensor, weight: Tensor, mask: Tensor,
    state: WarpState, config: QwarpConfig, device: torch.device,
) -> None:
    """Main multi-level patch optimization loop (batched GPU version).

    Implements IW3D_warpomatic() with checkerboard parallel processing.
    """
    nx, ny, nz = state.nx, state.ny, state.nz
    t0 = time.time()

    do_x = not (config.warp_flags & 1)
    do_y = not (config.warp_flags & 2)
    do_z = not (config.warp_flags & 4)
    do_xyz = (do_x, do_y, do_z)
    axis_w = config.axis_weights

    imin, imax, jmin, jmax, kmin, kmax = _autobox(weight)

    if config.verb >= 1:
        print(
            f"qwarp_torch: {nx}x{ny}x{nz} volume; "
            f"autobox={imin}..{imax} {jmin}..{jmax} {kmin}..{kmax}"
        )

    base_clip = _auto_clip(base.reshape(-1), weight.reshape(-1))
    source_clip = _auto_clip(source.reshape(-1), weight.reshape(-1))

    # Compute initial cost so it's never the 666.666 sentinel
    with torch.no_grad():
        state.cost = _global_correlation(
            base, state.warped_source, weight, base_clip, source_clip
        )

    # --- Level 0 bounds (always computed for level 1+ patch sizing) ---
    xwid = (imax - imin) // 8
    ywid = (jmax - jmin) // 8
    zwid = (kmax - kmin) // 8

    ibbb = max(1, imin - xwid)
    jbbb = max(1, jmin - ywid)
    kbbb = max(1, kmin - zwid)
    ittt = min(nx - 2, imax + xwid)
    jttt = min(ny - 2, jmax + ywid)
    kttt = min(nz - 2, kmax + zwid)

    if nz == 1:
        kbbb = kttt = 0

    # --- Level 0: global warp (single patch) ---
    if config.start_level == 0:
        first_cost = state.cost

        # Level 0: progressive basis complexity. On CUDA the single global
        # patch runs through the GPU-resident batched optimizer (B=1) -- this
        # is the big win, since the serial Powell path syncs to the host on
        # every cost evaluation and used to dominate the whole runtime.
        lev0_bases = ["cubic_lite", "cubic", "quintic_lite"]
        use_gpu_lev0 = device.type == "cuda"

        if use_gpu_lev0:
            lev0_patch = PatchSpec(
                ibot=ibbb, itop=ittt, jbot=jbbb, jtop=jttt, kbot=kbbb, ktop=kttt,
                gi=0, gj=0, gk=0,
            )
            nxh0 = ittt - ibbb + 1
            nyh0 = jttt - jbbb + 1
            nzh0 = kttt - kbbb + 1
            kk0, jj0, ii0 = torch.meshgrid(
                torch.arange(nzh0, dtype=torch.float32, device=device),
                torch.arange(nyh0, dtype=torch.float32, device=device),
                torch.arange(nxh0, dtype=torch.float32, device=device),
                indexing="ij",
            )
            ii0_flat, jj0_flat, kk0_flat = ii0.reshape(-1), jj0.reshape(-1), kk0.reshape(-1)

        if config.verb >= 1 and _tqdm is not None:
            pbar = _tqdm(
                lev0_bases,
                desc=f"lev=0 {ibbb}..{ittt} {jbbb}..{jttt} {kbbb}..{kttt}",
                bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}",
                leave=True,
            )
        else:
            pbar = lev0_bases
            if config.verb >= 1:
                print(f"lev=0 {ibbb}..{ittt} {jbbb}..{jttt} {kbbb}..{kttt}: ", end="", flush=True)

        for basis_type in pbar:
            if use_gpu_lev0:
                basis0, half_widths0, param_max0 = _get_basis_config(
                    basis_type, nxh0, nyh0, nzh0, device, hfactor=1.0,
                )
                _improve_warp_batched(
                    base, source, weight, mask, state, config, device,
                    [lev0_patch], basis0, half_widths0, param_max0,
                    ii0_flat, jj0_flat, kk0_flat,
                    nxh0, nyh0, nzh0,
                    do_xyz=do_xyz,
                    axis_weights=axis_w,
                    use_penalty=False,
                    pen_fac=0.0,
                    base_clip=base_clip,
                    source_clip=source_clip,
                    max_iter=config.batch_optimizer_iters_lev0,
                )
            else:
                _improve_warp_serial(
                    base, source, weight, mask, state, config, device,
                    ibbb, ittt, jbbb, jttt, kbbb, kttt,
                    basis_type=basis_type,
                    do_xyz=do_xyz,
                    axis_weights=axis_w,
                    use_penalty=False,
                    pen_fac=0.0,
                    base_clip=base_clip,
                    source_clip=source_clip,
                )

        # The batched path doesn't set state.cost (only the serial path does),
        # so recompute the global cost for an accurate level-0 readout.
        if use_gpu_lev0:
            with torch.no_grad():
                state.cost = _global_correlation(
                    base, state.warped_source, weight, base_clip, source_clip
                )

        if config.verb >= 1:
            elapsed = time.time() - t0
            if _tqdm is not None and isinstance(pbar, _tqdm):
                pbar.set_postfix_str(
                    f"cost={first_cost:.5f}=>{state.cost:.5f} {elapsed:.1f}s"
                )
                pbar.close()
            else:
                print(f" done [cost:{first_cost:.5f}==>{state.cost:.5f}] ({elapsed:.1f}s)")

        if config.level_callback is not None:
            config.level_callback(0, state.xd, state.yd, state.zd, state.warped_source)

    # --- Levels 1..N: progressively smaller patches (batched GPU) ---
    xwid0 = ittt - ibbb + 1
    ywid0 = jttt - jbbb + 1
    zwid0 = kttt - kbbb + 1

    # Lev=1 patch size, the reference for Hfactor scaling
    max_patch_lev1 = max(1, int(max(xwid0, ywid0, zwid0) * config.shrink))

    ngmin = max(config.minpatch, 5)
    if ngmin % 2 == 0:
        ngmin -= 1

    levdone = False
    lev_start = max(1, config.start_level)

    for lev in range(lev_start, config.max_level + 1):
        if levdone:
            break

        state.last_level = lev

        flev = config.shrink ** lev
        xwid = int(xwid0 * flev)
        ywid = int(ywid0 * flev)
        zwid = int(zwid0 * flev)

        if xwid % 2 == 0:
            xwid += 1
        if ywid % 2 == 0:
            ywid += 1
        if zwid % 2 == 0:
            zwid += 1

        dox = xwid >= ngmin and do_x
        doy = ywid >= ngmin and do_y
        doz = zwid >= ngmin and do_z

        if not (dox or doy or doz):
            break

        if xwid < ngmin:
            xwid = min(nx, ngmin)
        if ywid < ngmin:
            ywid = min(ny, ngmin)
        if zwid < ngmin:
            zwid = min(nz, ngmin)

        ftest = max(xwid, ywid, zwid) / ngmin
        if ftest <= 1.0 / config.shrink + 0.0001:
            xwid = min(xwid, ngmin)
            ywid = min(ywid, ngmin)
            zwid = min(zwid, ngmin)
            levdone = True

        xdel = max(1, (xwid - 1) // 2)
        ydel = max(1, (ywid - 1) // 2)
        zdel = max(1, (zwid - 1) // 2)

        ibbb = max(1, imin - xdel // 4 - 1)
        jbbb = max(1, jmin - ydel // 4 - 1)
        kbbb = max(1, kmin - zdel // 4 - 1)
        ittt = min(nx - 2, imax + xdel // 4 + 1)
        jttt = min(ny - 2, jmax + ydel // 4 + 1)
        kttt = min(nz - 2, kmax + zdel // 4 + 1)

        if nz == 1:
            kbbb = kttt = 0

        # Penalty settings
        pen_lev = (lev - lev_start + 1) ** 0.333
        pen_fff = config.penalty_factor * min(3.21, pen_lev)
        use_pen = pen_fff > 0 and lev >= config.penalty_first_level
        if lev == config.penalty_first_level:
            pen_fff *= 0.5

        # Basis type
        basis_type = "cubic_lite" if config.use_lite else "cubic"
        if levdone and config.use_quintic:
            basis_type = "quintic_lite" if config.use_lite else "quintic"

        # Workhard passes
        wh1, wh2 = config.workhard
        nlevr = 2 if (wh1 <= lev <= wh2) else 1

        # Generate patch grid and filter
        all_patches = _generate_patch_grid(
            ibbb, ittt, jbbb, jttt, kbbb, kttt,
            xwid, ywid, zwid, xdel, ydel, zdel,
        )
        valid_patches = _filter_patches(all_patches, weight, mask, nx, ny, nz)
        phases = _checkerboard_phases(valid_patches)

        # Pre-compute shared basis (all patches at this level have same size)
        nxh = xwid
        nyh = ywid
        nzh = zwid
        max_patch = max(nxh, nyh, nzh)
        hfactor = _compute_hfactor(max_patch, max_patch_lev1, config.hfactor_q)
        basis, half_widths, param_max = _get_basis_config(
            basis_type, nxh, nyh, nzh, device, hfactor=hfactor,
        )

        # Pre-compute shared coordinate grids
        kk_p, jj_p, ii_p = torch.meshgrid(
            torch.arange(nzh, dtype=torch.float32, device=device),
            torch.arange(nyh, dtype=torch.float32, device=device),
            torch.arange(nxh, dtype=torch.float32, device=device),
            indexing="ij",
        )
        ii_flat = ii_p.reshape(-1)
        jj_flat = jj_p.reshape(-1)
        kk_flat = kk_p.reshape(-1)

        n_valid = sum(len(ph) for ph in phases)
        state.patches_done = 0
        state.patches_skipped = len(all_patches) - len(valid_patches)
        cost_at_start = state.cost

        # Progress bar for this level: total = n_valid patches * nlevr passes
        total_patches = n_valid * nlevr
        lev_desc = f"lev={lev} {xwid}x{ywid}x{zwid}"
        lev_pbar = None
        if config.verb >= 1 and _tqdm is not None:
            lev_pbar = _tqdm(
                total=total_patches,
                desc=lev_desc,
                bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}",
                leave=True,
            )
            lev_pbar.set_postfix_str(f"cost={state.cost:.5f}")
        elif config.verb >= 1:
            print(
                f"lev={lev} patch={xwid}x{ywid}x{zwid} "
                f"patches={n_valid} [{time.time()-t0:.0f}s]",
                end="", flush=True,
            )

        for _pass in range(nlevr):
            phase_order = list(range(8))
            if (lev + _pass) % 2 == 1:
                phase_order = phase_order[::-1]

            for phase_idx in phase_order:
                phase_patches = phases[phase_idx]
                if not phase_patches:
                    continue

                # Split into same-size (batchable) and odd-size patches
                good = [p for p in phase_patches
                        if (p.itop - p.ibot + 1 == nxh and
                            p.jtop - p.jbot + 1 == nyh and
                            p.ktop - p.kbot + 1 == nzh)]
                odd = [p for p in phase_patches
                       if (p.itop - p.ibot + 1 != nxh or
                           p.jtop - p.jbot + 1 != nyh or
                           p.ktop - p.kbot + 1 != nzh)]

                if good:
                    # BATCHED GPU path (works for B>=1)
                    _improve_warp_batched(
                        base, source, weight, mask, state, config, device,
                        good, basis, half_widths, param_max,
                        ii_flat, jj_flat, kk_flat,
                        nxh, nyh, nzh,
                        do_xyz=do_xyz,
                        axis_weights=axis_w,
                        use_penalty=use_pen,
                        pen_fac=pen_fff,
                        base_clip=base_clip,
                        source_clip=source_clip,
                    )

                # Serial fallback for odd-sized boundary patches
                for p in odd:
                    _improve_warp_serial(
                        base, source, weight, mask, state, config, device,
                        p.ibot, p.itop, p.jbot, p.jtop, p.kbot, p.ktop,
                        basis_type=basis_type,
                        do_xyz=do_xyz,
                        axis_weights=axis_w,
                        use_penalty=use_pen,
                        pen_fac=pen_fff,
                        base_clip=base_clip,
                        source_clip=source_clip,
                    )

                # warped_source already updated in _improve_warp_batched and
                # _improve_warp_serial write-back steps -- no redundant refresh needed

                # Update progress bar
                if lev_pbar is not None:
                    lev_pbar.update(len(phase_patches))

        # Light warp smoothing to reduce patch boundary artifacts
        # Sigma scales with patch overlap: half the overlap width
        # xdel is the step between patches, (nxh - xdel) is the overlap
        smooth_sigma = 1.5
        with torch.no_grad():
            from .weight import _gaussian_smooth_3d
            state.xd = _gaussian_smooth_3d(state.xd, smooth_sigma)
            state.yd = _gaussian_smooth_3d(state.yd, smooth_sigma)
            state.zd = _gaussian_smooth_3d(state.zd, smooth_sigma)
            # Optional max displacement clamp
            if config.maxdisp > 0:
                state.xd.clamp_(-config.maxdisp, config.maxdisp)
                state.yd.clamp_(-config.maxdisp, config.maxdisp)
                state.zd.clamp_(-config.maxdisp, config.maxdisp)
            # Refresh warped_source with smoothed warp
            state.warped_source = warp_image_linear(source, state.xd, state.yd, state.zd)

        # Compute actual global correlation after this level
        with torch.no_grad():
            state.cost = _global_correlation(
                base, state.warped_source, weight, base_clip, source_clip
            )

        if config.verb >= 1:
            elapsed = time.time() - t0
            if lev_pbar is not None:
                lev_pbar.set_postfix_str(
                    f"cost={cost_at_start:.5f}=>{state.cost:.5f} "
                    f"({state.patches_done}done {state.patches_skipped}skip) {elapsed:.1f}s"
                )
                lev_pbar.close()
            else:
                print(
                    f" done [cost:{cost_at_start:.5f}==>{state.cost:.5f};"
                    f" {state.patches_done} done, {state.patches_skipped} skip]"
                    f" ({elapsed:.1f}s)"
                )

        if config.level_callback is not None:
            config.level_callback(lev, state.xd, state.yd, state.zd, state.warped_source)

        # Early stopping: if this level barely improved cost, skip finer levels
        if config.level_stop_tol > 0 and cost_at_start < 0:
            improvement = abs(state.cost - cost_at_start)
            threshold = config.level_stop_tol * abs(cost_at_start)
            if improvement < threshold:
                if config.verb >= 1:
                    print(f"  Early stop: improvement {improvement:.6f} < "
                          f"threshold {threshold:.6f} ({config.level_stop_tol:.1e})")
                break


# ---------------------------------------------------------------------------
# Batched GPU patch optimization (the fast path)
# ---------------------------------------------------------------------------

def _improve_warp_batched(
    base: Tensor, source: Tensor, weight: Tensor, mask: Tensor,
    state: WarpState, config: QwarpConfig, device: torch.device,
    patches: list[PatchSpec],
    basis: Tensor,
    half_widths: tuple[float, float, float],
    param_max: float,
    ii_flat: Tensor, jj_flat: Tensor, kk_flat: Tensor,
    nxh: int, nyh: int, nzh: int,
    do_xyz: tuple[bool, bool, bool] = (True, True, True),
    axis_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    use_penalty: bool = False,
    pen_fac: float = 0.033333,
    base_clip: tuple[float, float] | None = None,
    source_clip: tuple[float, float] | None = None,
    max_iter: int | None = None,
) -> None:
    """Optimize ALL patches in one checkerboard phase simultaneously on GPU.

    Also used for the single global level-0 patch (B=1): keeping that
    optimization GPU-resident avoids the per-evaluation CPU<->GPU sync that
    made the serial Powell path dominate the runtime.
    """
    B = len(patches)
    if B == 0:
        return

    nx, ny, nz = state.nx, state.ny, state.nz
    n_basis = basis.shape[0]
    _V = basis.shape[1]

    active_dims = [d for d in range(3) if do_xyz[d]]
    n_active = len(active_dims) * n_basis
    n_total = 3 * n_basis

    if n_active == 0:
        return

    # Gather patch offsets
    ibots = torch.tensor([p.ibot for p in patches], device=device, dtype=torch.float32)
    jbots = torch.tensor([p.jbot for p in patches], device=device, dtype=torch.float32)
    kbots = torch.tensor([p.kbot for p in patches], device=device, dtype=torch.float32)

    # Gather fixed data: base, weight, mask as (B, V) tensors
    base_patches = torch.stack([
        base[p.kbot:p.ktop+1, p.jbot:p.jtop+1, p.ibot:p.itop+1].reshape(-1)
        for p in patches
    ])
    weight_patches = torch.stack([
        weight[p.kbot:p.ktop+1, p.jbot:p.jtop+1, p.ibot:p.itop+1].reshape(-1)
        for p in patches
    ])
    mask_patches = torch.stack([
        mask[p.kbot:p.ktop+1, p.jbot:p.jtop+1, p.ibot:p.itop+1].reshape(-1).float()
        for p in patches
    ])

    # Pre-compute cost function fixed parts
    use_lpa = config.cost_method == "lpa"
    batch_incor = None
    if not use_lpa:
        batch_incor = BatchedIncrementalCorrelation(
            method=config.cost_method,
            base_clip=base_clip,
            source_clip=source_clip,
        )
        patch_slices = [
            (p.ibot, p.itop, p.jbot, p.jtop, p.kbot, p.ktop) for p in patches
        ]
        batch_incor.precompute_fixed_parts(
            base, state.warped_source, weight, patch_slices,
            base_patches=base_patches, weight_patches=weight_patches,
        )

    # Pre-compute external penalty if needed (vectorized: global sum minus each patch)
    external_pen = torch.zeros(B, device=device)
    if use_penalty:
        with torch.no_grad():
            je_global, se_global = compute_jacobian_energy(state.xd, state.yd, state.zd)
            energy_global = je_global + se_global
            global_energy_sum = energy_global.sum()
            # Gather patch energies as (B,) via stacking
            patch_energies = torch.stack([
                energy_global[p.kbot:p.ktop+1, p.jbot:p.jtop+1, p.ibot:p.itop+1].sum()
                for p in patches
            ])
            external_pen = global_energy_sum - patch_energies

    # Axis weight scales
    _ax_scales = torch.tensor([axis_weights[d] for d in active_dims],
                              device=device, dtype=torch.float32)

    # Pre-stack global warp as 3-channel volume ONCE (avoids re-stacking
    # every optimizer iteration -- saves ~45 MB of memory copies per iter)
    global_warp_3ch = torch.stack([state.xd, state.yd, state.zd], dim=0)

    # Pre-build expansion matrix: (n_active, n_total) maps active params to
    # full param vector with axis weight scaling. Replaces a Python for-loop
    # + per-call torch.zeros allocation from the hot path.
    expand_mat = torch.zeros(n_active, n_total, device=device)
    idx = 0
    for dim_i in active_dims:
        offset = dim_i * n_basis
        scale = axis_weights[dim_i]
        expand_mat[idx:idx+n_basis, offset:offset+n_basis] = scale * torch.eye(n_basis, device=device)
        idx += n_basis

    # Pre-compute coordinate base offsets: (B, V), constant per phase
    base_i = ibots[:, None] + ii_flat[None, :]
    base_j = jbots[:, None] + jj_flat[None, :]
    base_k = kbots[:, None] + kk_flat[None, :]

    # Optionally compile stable building-block functions (cached by name, compiled once)
    _eval_warp = _maybe_compile(evaluate_patch_warp_batched, "eval_warp", device, config.compile)
    _compose = _maybe_compile(batched_compose_and_interpolate, "compose", device, config.compile)

    # Define the batched cost function (differentiable through autograd)
    def batched_cost(active_params: Tensor) -> Tensor:
        """(B, n_active) -> (B,) costs. Differentiable."""
        # Expand active params to full params with axis weights
        full_params = active_params @ expand_mat  # (B, n_active) @ (n_active, n_total) → (B, n_total)

        # Batched basis evaluation: (B, V) displacements
        hxd, hyd, hzd = _eval_warp(basis, full_params, half_widths, do_xyz)

        # Batched compose + interpolate (fused 4-ch: source + warp in one grid_sample)
        warped_vals, ah_xd, ah_yd, ah_zd = _compose(
            source, state.xd, state.yd, state.zd,
            hxd, hyd, hzd,
            ii_flat, jj_flat, kk_flat,
            ibots, jbots, kbots,
            nx, ny, nz,
            global_warp_3ch=global_warp_3ch,
            base_i=base_i, base_j=base_j, base_k=base_k,
        )

        warped_vals = warped_vals * mask_patches

        # Batched cost: (B,)
        if use_lpa:
            corr = batched_lpa_cost(
                base_patches, warped_vals, weight_patches,
                nzh, nyh, nxh, sigma=config.lpa_sigma,
                kernel_type=config.lpa_kernel,
            )
        else:
            corr = batch_incor.evaluate(base_patches, warped_vals, weight_patches)
        cost = -corr  # negate for minimization

        # Batched penalty (fully vectorized, no Python loop)
        if use_penalty and pen_fac > 0:
            ah_xd_3d = ah_xd.reshape(B, nzh, nyh, nxh)
            ah_yd_3d = ah_yd.reshape(B, nzh, nyh, nxh)
            ah_zd_3d = ah_zd.reshape(B, nzh, nyh, nxh)
            cost = cost + compute_penalty_batched(
                ah_xd_3d, ah_yd_3d, ah_zd_3d, pen_fac, external_pen
            )

        return cost

    # Run batched Adam optimizer
    best_params, best_costs = optimize_warp_params_batched(
        batched_cost, B, n_active, param_max, device,
        max_iter=config.batch_optimizer_iters if max_iter is None else max_iter,
        lr=config.batch_optimizer_lr,
    )

    # Apply optimized parameters - update global warp AND warped_source in one pass
    with torch.no_grad():
        full_params = best_params @ expand_mat  # reuse pre-built expansion matrix

        hxd, hyd, hzd = _eval_warp(basis, full_params, half_widths, do_xyz)
        warped_vals, ah_xd, ah_yd, ah_zd = _compose(
            source, state.xd, state.yd, state.zd,
            hxd, hyd, hzd,
            ii_flat, jj_flat, kk_flat,
            ibots, jbots, kbots,
            nx, ny, nz,
            global_warp_3ch=global_warp_3ch,
            base_i=base_i, base_j=base_j, base_k=base_k,
        )

        # Write back warp AND warped_source (non-overlapping patches, safe)
        for idx_p, p in enumerate(patches):
            state.xd[p.kbot:p.ktop+1, p.jbot:p.jtop+1, p.ibot:p.itop+1] = \
                ah_xd[idx_p].reshape(nzh, nyh, nxh)
            state.yd[p.kbot:p.ktop+1, p.jbot:p.jtop+1, p.ibot:p.itop+1] = \
                ah_yd[idx_p].reshape(nzh, nyh, nxh)
            state.zd[p.kbot:p.ktop+1, p.jbot:p.jtop+1, p.ibot:p.itop+1] = \
                ah_zd[idx_p].reshape(nzh, nyh, nxh)
            state.warped_source[p.kbot:p.ktop+1, p.jbot:p.jtop+1, p.ibot:p.itop+1] = \
                warped_vals[idx_p].reshape(nzh, nyh, nxh)

    state.patches_done += B

    if config.verb >= 2:
        # Route through tqdm.write so it doesn't stomp on the level progress bar.
        msg = f"  phase: B={B} cost={state.cost:.5f}"
        if _tqdm is not None:
            _tqdm.write(msg)
        else:
            print(msg)


# ---------------------------------------------------------------------------
# Serial patch optimization (for level 0 and edge cases)
# ---------------------------------------------------------------------------

def _improve_warp_serial(
    base: Tensor, source: Tensor, weight: Tensor, mask: Tensor,
    state: WarpState, config: QwarpConfig, device: torch.device,
    ibot: int, itop: int, jbot: int, jtop: int, kbot: int, ktop: int,
    basis_type: str = "cubic_lite",
    do_xyz: tuple[bool, bool, bool] = (True, True, True),
    axis_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    use_penalty: bool = False,
    pen_fac: float = 0.033333,
    base_clip: tuple[float, float] | None = None,
    source_clip: tuple[float, float] | None = None,
) -> bool:
    """Optimize the warp over one patch (serial, for lev=0 and edge cases)."""
    nx, ny, nz = state.nx, state.ny, state.nz

    ibot = max(0, min(ibot, nx - 1))
    itop = max(0, min(itop, nx - 1))
    jbot = max(0, min(jbot, ny - 1))
    jtop = max(0, min(jtop, ny - 1))
    kbot = max(0, min(kbot, nz - 1))
    ktop = max(0, min(ktop, nz - 1))

    nxh = itop - ibot + 1
    nyh = jtop - jbot + 1
    nzh = ktop - kbot + 1
    n_voxels = nxh * nyh * nzh

    if nxh < 5 and nyh < 5 and nzh < 5:
        state.patches_skipped += 1
        return False

    w_patch = weight[kbot:ktop+1, jbot:jtop+1, ibot:itop+1]
    m_patch = mask[kbot:ktop+1, jbot:jtop+1, ibot:itop+1]
    n_masked = int(m_patch.sum().item())
    w_sum = float(w_patch.sum().item())
    w_avg = float(weight[mask > 0].mean().item()) if (mask > 0).any() else 1.0

    if n_masked < 0.333 * n_voxels or w_sum < 0.166 * n_voxels * w_avg:
        state.patches_skipped += 1
        return False

    basis, half_widths, param_max = _get_basis_config(basis_type, nxh, nyh, nzh, device)
    n_basis_per_dim = basis.shape[0]

    active_dims = [d for d in range(3) if do_xyz[d]]
    n_active_params = len(active_dims) * n_basis_per_dim
    n_total_params = 3 * n_basis_per_dim

    if n_masked < 5 * n_active_params:
        state.patches_skipped += 1
        return False

    b_local = base[kbot:ktop+1, jbot:jtop+1, ibot:itop+1].reshape(-1)
    w_local = weight[kbot:ktop+1, jbot:jtop+1, ibot:itop+1].reshape(-1)

    if b_local.max() == b_local.min():
        state.patches_skipped += 1
        return False

    incor = IncrementalCorrelation(method=config.cost_method)
    if base_clip is not None and source_clip is not None:
        incor.set_clips(base_clip, source_clip)

    weight_for_fixed = weight.clone()
    weight_for_fixed[kbot:ktop+1, jbot:jtop+1, ibot:itop+1] = 0.0
    incor.add_fixed(
        base.reshape(-1),
        state.warped_source.reshape(-1),
        weight_for_fixed.reshape(-1),
    )

    pen_external = 0.0
    if use_penalty:
        je_global, se_global = compute_jacobian_energy(state.xd, state.yd, state.zd)
        je_ext = je_global.clone()
        se_ext = se_global.clone()
        je_ext[kbot:ktop+1, jbot:jtop+1, ibot:itop+1] = 0.0
        se_ext[kbot:ktop+1, jbot:jtop+1, ibot:itop+1] = 0.0
        pen_external = float((je_ext + se_ext).sum().item())

    global_xd = state.xd
    global_yd = state.yd
    global_zd = state.zd

    # Pre-compute coordinate grids (reuse across evaluations)
    kk_p, jj_p, ii_p = torch.meshgrid(
        torch.arange(nzh, dtype=torch.float32, device=device),
        torch.arange(nyh, dtype=torch.float32, device=device),
        torch.arange(nxh, dtype=torch.float32, device=device),
        indexing="ij",
    )

    def cost_function(params: Tensor) -> float:
        full_params = torch.zeros(n_total_params, device=device)
        idx = 0
        for dim_i in active_dims:
            offset = dim_i * n_basis_per_dim
            scale = axis_weights[dim_i]
            full_params[offset:offset + n_basis_per_dim] = params[idx:idx + n_basis_per_dim] * scale
            idx += n_basis_per_dim

        hxd, hyd, hzd = evaluate_patch_warp(basis, full_params, half_widths, do_xyz)
        hxd_3d = hxd.reshape(nzh, nyh, nxh)
        hyd_3d = hyd.reshape(nzh, nyh, nxh)
        hzd_3d = hzd.reshape(nzh, nyh, nxh)

        xq = (ibot + ii_p + hxd_3d).clamp(0, nx - 1)
        yq = (jbot + jj_p + hyd_3d).clamp(0, ny - 1)
        zq = (kbot + kk_p + hzd_3d).clamp(0, nz - 1)

        axd_interp = trilinear_interpolate(global_xd, xq.reshape(-1), yq.reshape(-1), zq.reshape(-1)).reshape(nzh, nyh, nxh)
        ayd_interp = trilinear_interpolate(global_yd, xq.reshape(-1), yq.reshape(-1), zq.reshape(-1)).reshape(nzh, nyh, nxh)
        azd_interp = trilinear_interpolate(global_zd, xq.reshape(-1), yq.reshape(-1), zq.reshape(-1)).reshape(nzh, nyh, nxh)

        ah_xd = hxd_3d + axd_interp
        ah_yd = hyd_3d + ayd_interp
        ah_zd = hzd_3d + azd_interp

        src_x = (ah_xd + ii_p + ibot).clamp(-0.499, nx - 0.501)
        src_y = (ah_yd + jj_p + jbot).clamp(-0.499, ny - 0.501)
        src_z = (ah_zd + kk_p + kbot).clamp(-0.499, nz - 0.501)

        warped_vals = trilinear_interpolate(
            source, src_x.reshape(-1), src_y.reshape(-1), src_z.reshape(-1)
        )

        m_flat = m_patch.reshape(-1).float()
        warped_vals = warped_vals * m_flat

        corr = incor.evaluate(b_local, warped_vals, w_local)
        cost = -corr

        if use_penalty:
            from .penalty import compute_penalty
            pen = compute_penalty(ah_xd, ah_yd, ah_zd, pen_fac, pen_external)
            cost += pen

        return cost

    best_params, best_cost = optimize_warp_params_torch(
        cost_function,
        n_active_params,
        param_max,
        device,
        max_iter=8 * n_active_params + 31,
    )

    # Apply optimized parameters
    full_params = torch.zeros(n_total_params, device=device)
    idx = 0
    for dim_i in active_dims:
        offset = dim_i * n_basis_per_dim
        scale = axis_weights[dim_i]
        full_params[offset:offset + n_basis_per_dim] = best_params[idx:idx + n_basis_per_dim] * scale
        idx += n_basis_per_dim

    hxd, hyd, hzd = evaluate_patch_warp(basis, full_params, half_widths, do_xyz)
    hxd_3d = hxd.reshape(nzh, nyh, nxh)
    hyd_3d = hyd.reshape(nzh, nyh, nxh)
    hzd_3d = hzd.reshape(nzh, nyh, nxh)

    xq = (ibot + ii_p + hxd_3d).clamp(0, nx - 1)
    yq = (jbot + jj_p + hyd_3d).clamp(0, ny - 1)
    zq = (kbot + kk_p + hzd_3d).clamp(0, nz - 1)

    axd_interp = trilinear_interpolate(global_xd, xq.reshape(-1), yq.reshape(-1), zq.reshape(-1)).reshape(nzh, nyh, nxh)
    ayd_interp = trilinear_interpolate(global_yd, xq.reshape(-1), yq.reshape(-1), zq.reshape(-1)).reshape(nzh, nyh, nxh)
    azd_interp = trilinear_interpolate(global_zd, xq.reshape(-1), yq.reshape(-1), zq.reshape(-1)).reshape(nzh, nyh, nxh)

    ah_xd = hxd_3d + axd_interp
    ah_yd = hyd_3d + ayd_interp
    ah_zd = hzd_3d + azd_interp

    state.xd[kbot:ktop+1, jbot:jtop+1, ibot:itop+1] = ah_xd
    state.yd[kbot:ktop+1, jbot:jtop+1, ibot:itop+1] = ah_yd
    state.zd[kbot:ktop+1, jbot:jtop+1, ibot:itop+1] = ah_zd

    src_x = (ah_xd + ii_p + ibot).clamp(-0.499, nx - 0.501)
    src_y = (ah_yd + jj_p + jbot).clamp(-0.499, ny - 0.501)
    src_z = (ah_zd + kk_p + kbot).clamp(-0.499, nz - 0.501)

    warped_patch = trilinear_interpolate(
        source, src_x.reshape(-1), src_y.reshape(-1), src_z.reshape(-1)
    ).reshape(nzh, nyh, nxh)

    state.warped_source[kbot:ktop+1, jbot:jtop+1, ibot:itop+1] = warped_patch
    state.cost = best_cost
    state.patches_done += 1

    if config.verb >= 2:
        print(
            f"     {basis_type} patch {ibot}..{itop} {jbot}..{jtop} {kbot}..{ktop}"
            f" : cost={best_cost:.5f}"
        )

    return True


def _autobox(weight: Tensor) -> tuple[int, int, int, int, int, int]:
    """Find bounding box of nonzero weights."""
    nz, ny, nx = weight.shape
    nonzero = (weight > 0).nonzero(as_tuple=True)

    if len(nonzero[0]) == 0:
        return 0, nx - 1, 0, ny - 1, 0, nz - 1

    imin = int(nonzero[2].min().item())
    imax = int(nonzero[2].max().item())
    jmin = int(nonzero[1].min().item())
    jmax = int(nonzero[1].max().item())
    kmin = int(nonzero[0].min().item())
    kmax = int(nonzero[0].max().item())

    return imin, imax, jmin, jmax, kmin, kmax
