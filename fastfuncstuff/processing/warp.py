"""Main warping engine - the heart of qwarp_torch.

This module implements IW3D_warpomatic() and IW3D_improve_warp() from
AFNI's mri_nwarp.c in PyTorch, with GPU-parallel batch processing.

Algorithm:
1. Level 0: global warp (full image patch, progressive basis complexity)
2. Levels 1..N: shrinking patches with 50% overlap
3. Within each level: 3D checkerboard decomposition into 8 phases
4. Within each phase: ALL patches optimized in parallel. They are *nearly* disjoint
   -- same-parity patches still share the voxel where they abut -- so write-backs go
   through `_dedup_last_wins` (or a serial loop) to stay deterministic.
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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import Tensor

from .._compile import safe_compile
from ..memory import plan_nonlinear_memory

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
    _make_kernel_1d,
    batched_lpa_cost,
)
from .cost_blok import (
    assign_bloks,
    lpa_value_pairs,
    lpc_value_pairs,
    prepare_blok_pairs,
)
from .interp import (
    batched_compose_and_interpolate,
    batched_compose_and_interpolate_multi,
    batched_interp_3ch,
    batched_trilinear_interpolate,
    batched_trilinear_interpolate_multi,
    trilinear_interpolate,
    warp_image,
)
from .metrics import PATCH_METRICS, batched_patch_cost
from .optimizer import (
    BatchOptStats,
    optimize_warp_params_batched,
    optimize_warp_params_gauss_newton,
    optimize_warp_params_torch,
)
from .penalty import compute_jacobian_energy, compute_penalty_batched, penalty_energy
from .weight import compute_weight_image

# Cache for torch.compile'd building-block functions (stable identity, compiled once)
_compile_cache: dict[str, Callable[..., Any]] = {}


def _maybe_compile(
    fn: Callable[..., Any], name: str, device: torch.device, do_compile: bool
) -> Callable[..., Any]:
    """Return a compiled version of fn for CUDA, caching by name.

    Uses :func:`safe_compile` so an inductor/clang failure degrades to eager for the
    rest of the process (never crashes a run) and the shared PCH policy is applied.
    The name-keyed cache is process-global, so a building block compiled once is
    reused across every volume/frame -- warmup is paid a single time and amortized
    over an entire 4-D series (the timeseries win; see principles/torch.compile.md)."""
    if device.type != "cuda" or not do_compile:
        return fn
    if name not in _compile_cache:
        _compile_cache[name] = safe_compile(fn, dynamic=True)
    return _compile_cache[name]


@contextmanager
def _tf32_matmul(enable: bool) -> Iterator[None]:
    """Temporarily allow TF32 tensor-core float32 matmul, restoring on exit.

    Scoped -- never a process-global default -- so the qwarp fast path can use tensor
    cores without shifting float32 matmul precision in other tools (allineate/GLM),
    which would break their reference-parity benchmarks. Safe for this solver: the
    TF32-affected matmuls only assemble the Gauss-Newton step, which is accepted only
    if the exact NCC cost (computed in full float32, no matmul) improves -- so reduced
    step precision can at worst waste an iteration, never corrupt the result."""
    if not enable:
        yield
        return
    prev_mm = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_mm
        torch.backends.cudnn.allow_tf32 = prev_cudnn


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

    interp: str = "linear"
    """Image interpolation used when sampling the source during cost evaluation
    (warped_source). AFNI 3dQwarp optimizes with linear; higher-order kernels
    (cubic/quintic/heptic/wsinc5) cost more for ~no output-quality gain because
    output sharpness comes from `final_interp`, not estimation. The fused per-patch
    inner loop is always linear regardless of this setting."""

    final_interp: str = "wsinc5"
    """Image interpolation for the final warped output (single pass over the
    original source through the total warp). Defaults to wsinc5 like 3dQwarp."""

    use_lite: bool = True
    """Use 'lite' (reduced parameter) basis functions."""

    workhard: tuple[int, int] | None = None
    """(start_lev, end_lev) inclusive range for double-pass optimization, or None
    to disable. end_lev < 0 means "through the last level"."""

    cost_method: str = "pearclp"
    """Cost function: 'pearclp' (clipped Pearson), 'pearson', 'lpc' / 'lpa'
    (AFNI-faithful local Pearson over a single global truncated-octahedron blok
    lattice, matching 3dQwarp -- lpc negates, lpa takes |.|), or 'lpa_alt' (the
    older Gaussian/box separable-convolution local Pearson)."""

    blok_rad: float = 6.54321
    """Blok radius in voxels for the lpc/lpa cost. Matches 3dQwarp's fixed tohd
    radius (create_INCORR_BLOK_set(..., GA_BLOK_TOHD, 6.54321))."""

    lpc_ppow: float = 1.0
    """AFNI ppow exponent on |Fisher-z| in the local-Pearson aggregation."""

    penalty_factor: float = 0.033
    """Base Jacobian-energy penalty factor. Matches AFNI's Hpen_fbase=0.033."""

    hybrid_polish_iters: int = 10
    """Adam steps after the Gauss-Newton pass when ``optimizer="hybrid"``. Short by
    design: GN has already done the travelling, this only recovers what the
    least-squares surrogate could not see."""

    gn_iters: int = 8
    """Gauss-Newton iterations per patch group (one solve + one cost evaluation)."""

    cc_radius: int = 4
    """Neighbourhood half-width for ``cost_method="lncc"``. Clamped per level so the
    window fits inside the patch -- an oversized window makes every voxel see the same
    whole-patch statistics, silently turning the local metric into a global one."""

    mind_radius: float = 1.0
    """Patch radius for the MIND / MIND-SSC descriptors."""

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

    optimizer: str = "adam"
    """Per-patch optimizer, for **both** the ME-scaled polish and the ordinary warp:
    'adam' (autodiff, any cost) or 'gn' (Levenberg-damped Gauss-Newton with an
    analytic image-gradient Jacobian).

    'gn' needs a least-squares surrogate, which the correlation costs have -- the
    plain ones through a patch-wide zero-normalised residual, lpa and lncc through a
    locally normalised one. It falls back to adam for anything else, notably lpc
    (which rewards anti-correlation, so a sum-of-squares residual would point
    backwards) and the descriptor costs.

    Measured on a 193^3 T1->MNI fit against AFNI 3dQwarp's 543.2s: pearclp 35.0s ->
    13.1s, lpa 96.7s -> 15.9s, lncc 131.5s -> 15.0s. On lpa it is also *better* than
    adam on every independent metric. Every step is accepted or rejected on the true
    cost, so the surrogate steers but the reported functional still decides."""

    batch_optimizer_lr: float = 0.008
    """Learning rate for the batched Adam optimizer."""

    batch_optimizer_iters: int = 60
    """Max iterations for batched Adam optimizer per phase."""

    batch_optimizer_tol: float = 1e-4
    """Per-patch relative-improvement threshold below which a patch counts as
    'stalled'. Lower => patches keep optimizing longer (more warp, slower)."""

    batch_optimizer_patience: int = 5
    """Consecutive stalled steps before a patch is considered converged. The
    batch loop stops only once every patch has converged (or hits the iter cap)."""

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

    reject_worse_levels: bool = False
    """If a refinement level makes the global cost worse, restore the previous
    level's warp and stop refining. The finest levels over-warp first, so once a
    level degrades the fit, going finer rarely recovers it. Off by default
    (AFNI-style 'always run every level'); the global-cost readout it keys on is
    a coarse proxy that can trip on volumes near the threshold, and in the
    source-batched path that made a couple of volumes bifurcate to a different
    (whole-level) warp than their solo qwarp run. Set True to re-enable."""

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

    # Gradient of the (unchanging) source volume, stacked (3, nz, ny, nx). Built on
    # first use by the Gauss-Newton path and kept, because it is the same for every
    # patch of every level and rebuilding it per phase would cost more than the
    # solve it feeds.
    source_grad_3ch: Tensor = field(default_factory=lambda: torch.empty(0))

    # Per-level optimizer-budget telemetry (batched path only); reset each level.
    opt_steps_weighted: int = 0  # sum over batches of steps_run * B
    opt_patches_counted: int = 0  # patches that went through the batched optimizer
    opt_hit_budget: int = 0  # patches still improving at the iteration cap


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

    return F.pad(vol, (pad_x, pad_x, pad_y, pad_y, pad_z, pad_z), mode="constant", value=0)


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
            print(
                f"qwarp_torch: padding +{pad_x},{pad_y},{pad_z} => "
                f"{base_p.shape[2]}x{base_p.shape[1]}x{base_p.shape[0]}"
            )
    else:
        base_p = base
        source_p = source

    nz, ny, nx = base_p.shape
    predicted, available = plan_nonlinear_memory((nz, ny, nx), device, "qwarp")
    if predicted > available and config.verb >= 1:
        print(
            f"WARNING: estimated qwarp peak {predicted / 2**30:.1f} GiB exceeds "
            f"the {available / 2**30:.1f} GiB safe memory budget; enable the pyramid or use CPU."
        )

    if weight is None:
        weight_p = compute_weight_image(base_p)
    else:
        weight_p = (
            _pad_volume(weight.float().to(device), pad_x, pad_y, pad_z)
            if (pad_x > 0 or pad_y > 0 or pad_z > 0)
            else weight.float().to(device)
        )

    if mask is None:
        mask_p = (weight_p > 0).byte()
    else:
        mask_p = (
            _pad_volume(mask.byte().to(device).float(), pad_x, pad_y, pad_z).byte()
            if (pad_x > 0 or pad_y > 0 or pad_z > 0)
            else mask.byte().to(device)
        )

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
        state.warped_source = warp_image(source_p, state.xd, state.yd, state.zd, mode=config.interp)
    else:
        state.xd = torch.zeros(nz, ny, nx, device=device)
        state.yd = torch.zeros(nz, ny, nx, device=device)
        state.zd = torch.zeros(nz, ny, nx, device=device)
        state.warped_source = source_p.clone()

    if config.pyramid_factor > 1 and initial_warp is None and config.start_level == 0:
        _warpomatic_pyramid(base_p, source_p, weight_p, mask_p, state, config, device)
    else:
        _warpomatic(base_p, source_p, weight_p, mask_p, state, config, device)

    # Final warped image: apply warp to unpadded source, crop to original size.
    # Single pass through the total warp; final_interp (wsinc5 by default, like
    # 3dQwarp) is where output sharpness comes from.
    warped_full = warp_image(source_p, state.xd, state.yd, state.zd, mode=config.final_interp)
    warped = warped_full[pad_z : pad_z + nz_orig, pad_y : pad_y + ny_orig, pad_x : pad_x + nx_orig]

    # Return full padded warp field (caller/io.py handles grid info)
    return warped, state.xd, state.yd, state.zd


# ---------------------------------------------------------------------------
# Patch grid and checkerboard decomposition
# ---------------------------------------------------------------------------


def _generate_patch_grid(
    ibbb: int,
    ittt: int,
    jbbb: int,
    jttt: int,
    kbbb: int,
    kttt: int,
    xwid: int,
    ywid: int,
    zwid: int,
    xdel: int,
    ydel: int,
    zdel: int,
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

                patches.append(
                    PatchSpec(
                        ibot=ibot,
                        itop=itop,
                        jbot=jbot,
                        jtop=jtop,
                        kbot=kbot,
                        ktop=ktop,
                        gi=gi,
                        gj=gj,
                        gk=gk,
                    )
                )

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


def _dedup_last_wins(flat_dst: Tensor) -> tuple[Tensor, Tensor]:
    """Collapse a patch write-back index to one entry per target voxel.

    Patches in a checkerboard phase are *nearly* disjoint, not disjoint: the sweep
    steps ``(width-1)//2``, so same-parity patches share the voxel where they abut
    (and more where the sweep snaps the last patch back inside the box). Assigning
    through the raw index is then an ``index_put_`` with duplicate targets, whose
    winner PyTorch leaves unspecified -- two identical runs drift apart by tenths of
    a voxel. AFNI writes its patches serially, so the last patch in lattice order
    wins; a stable sort reproduces that exactly.

    Args:
        flat_dst: ``(N,)`` flat target indices, in lattice (patch) order.

    Returns:
        ``(dst, src)`` -- unique targets and the position in ``flat_dst`` that wins
        each one, so ``vol.view(-1)[dst] = values.reshape(-1)[src]`` is well-defined.
    """
    order = torch.argsort(flat_dst, stable=True)
    sorted_dst = flat_dst[order]
    last = torch.ones_like(sorted_dst, dtype=torch.bool)
    last[:-1] = sorted_dst[1:] != sorted_dst[:-1]
    return sorted_dst[last], order[last]


def _checkerboard_phases(
    patches: list[PatchSpec],
    flat_axis: int | None = None,
) -> list[list[PatchSpec]]:
    """Group patches into 8 checkerboard phases (non-overlapping within each).

    ``flat_axis`` (grid axis 0=x/1=y/2=z) is one voxel thick in slicewise mode, so
    patches can never overlap along it and splitting on its parity is pure overhead:
    collapsing it halves the phase count and doubles the batch each solve sees.
    """
    phases: list[list[PatchSpec]] = [[] for _ in range(8)]
    for p in patches:
        gi = 0 if flat_axis == 0 else p.gi % 2
        gj = 0 if flat_axis == 1 else p.gj % 2
        gk = 0 if flat_axis == 2 else p.gk % 2
        phases[gi * 4 + gj * 2 + gk].append(p)
    return phases


def _filter_patches(
    patches: list[PatchSpec],
    weight: Tensor,
    mask: Tensor,
    nx: int,
    ny: int,
    nz: int,
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

        m_patch = mask[p.kbot : p.ktop + 1, p.jbot : p.jtop + 1, p.ibot : p.itop + 1]
        w_patch = weight[p.kbot : p.ktop + 1, p.jbot : p.jtop + 1, p.ibot : p.itop + 1]
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
    basis_type: str,
    nxh: int,
    nyh: int,
    nzh: int,
    device: torch.device,
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
    return prat**alpha


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
    base: Tensor,
    source: Tensor,
    weight: Tensor,
    mask: Tensor,
    state: WarpState,
    config: QwarpConfig,
    device: torch.device,
) -> None:
    """Multi-octave coarse-to-fine resolution pyramid (opt-in, pyramid_factor>1).

    Builds a halving scale ladder from ``pyramid_factor`` down to full res
    (e.g. 4 -> [4, 2, 1], 2 -> [2, 1]). The global warp is solved at the
    coarsest scale, where the volume is factor**3 smaller -- microtime -- and
    every finer octave is *seeded* by upsampling the previous octave's warp and
    only refines a few levels from the hand-off point. Good seeds mean early
    stopping fires fast at each scale, so the expensive "full schedule at N x"
    of a single-hop pyramid is replaced by "a few seeded levels per octave".

    ``state`` is the full-resolution state (zero warp). All volumes are the
    padded full-resolution grid.
    """
    import torch.nn.functional as F

    nz, ny, nx = base.shape

    # Descending halving ladder down to full resolution.
    scales: list[int] = []
    s = max(1, config.pyramid_factor)
    while s >= 2:
        scales.append(s)
        s //= 2
    scales.append(1)

    def _resize(v: Tensor, size: tuple[int, int, int], is_mask: bool = False) -> Tensor:
        out = F.interpolate(
            v[None, None].float(), size=size, mode="trilinear", align_corners=False
        )[0, 0]
        return (out > 0.5).byte() if is_mask else out

    # prev carries the previous octave's warp + grid + last level reached.
    prev: tuple[Tensor, Tensor, Tensor, int, int, int, int] | None = None
    work: WarpState | None = None

    for scale in scales:
        if scale == 1:
            gz, gy, gx = nz, ny, nx
            b_s, src_s, w_s, m_s = base, source, weight, mask
        else:
            gz = max(16, round(nz / scale))
            gy = max(16, round(ny / scale))
            gx = max(16, round(nx / scale))
            size = (gz, gy, gx)
            b_s = _resize(base, size)
            src_s = _resize(source, size)
            w_s = _resize(weight, size)
            m_s = _resize(mask, size, is_mask=True)

        st = WarpState(nx=gx, ny=gy, nz=gz)
        if prev is None:
            # Coarsest octave: solve the global warp from scratch (microtime).
            st.xd = torch.zeros(gz, gy, gx, device=device)
            st.yd = torch.zeros(gz, gy, gx, device=device)
            st.zd = torch.zeros(gz, gy, gx, device=device)
            st.warped_source = src_s.clone()
            start_level = 0
            if config.verb >= 1:
                print(f"qwarp_torch: pyramid 1/{scale} at {gx}x{gy}x{gz} (global warp)")
        else:
            pxd, pyd, pzd, pgz, pgy, pgx, plast = prev
            # Upsample the seed warp; displacements are voxel units, so scale by
            # the per-axis resolution ratio (~2 between octaves).
            st.xd = _resize(pxd, (gz, gy, gx)) * (gx / pgx)
            st.yd = _resize(pyd, (gz, gy, gx)) * (gy / pgy)
            st.zd = _resize(pzd, (gz, gy, gx)) * (gz / pgz)
            st.warped_source = warp_image(src_s, st.xd, st.yd, st.zd, mode=config.interp)
            # Resume one level coarser than where the previous octave stopped
            # (one-level overlap so no spatial-frequency band is skipped).
            start_level = max(1, plast - 1)
            if config.verb >= 1:
                tag = "full res" if scale == 1 else f"1/{scale}"
                print(
                    f"qwarp_torch: pyramid {tag} at {gx}x{gy}x{gz}, refine from lev={start_level}"
                )

        # Recurse with pyramid off. Per-level dumps (-partials/-partial_warps)
        # only fire at full resolution, not on the downsampled octaves.
        cfg_s = replace(
            config,
            pyramid_factor=1,
            start_level=start_level,
            level_callback=(config.level_callback if scale == 1 else None),
        )
        _warpomatic(b_s, src_s, w_s, m_s, st, cfg_s, device)
        prev = (st.xd, st.yd, st.zd, gz, gy, gx, st.last_level)
        work = st

    # Hand the full-resolution result back to the caller's state.
    assert work is not None
    state.xd, state.yd, state.zd = work.xd, work.yd, work.zd
    state.warped_source = work.warped_source
    state.cost = work.cost
    state.last_level = work.last_level


# ---------------------------------------------------------------------------
# Main warpomatic loop
# ---------------------------------------------------------------------------


def _warpomatic(
    base: Tensor,
    source: Tensor,
    weight: Tensor,
    mask: Tensor,
    state: WarpState,
    config: QwarpConfig,
    device: torch.device,
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

    # AFNI-faithful local Pearson (lpc/lpa): one global truncated-octahedron
    # blok lattice over the whole grid; each patch is later scored over the
    # bloks that fall inside it. Built once per grid (so once per pyramid octave).
    blok_index_vol = None
    nblok = 0
    if config.cost_method in ("lpc", "lpa"):
        bs = assign_bloks((nz, ny, nx), (1.0, 1.0, 1.0), "tohd", config.blok_rad, mask=(mask > 0))
        blok_index_vol = bs.index.reshape(nz, ny, nx)
        nblok = bs.nblok

    # Compute initial cost so it's never the 666.666 sentinel
    with torch.no_grad():
        state.cost = _global_correlation(base, state.warped_source, weight, base_clip, source_clip)

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

        # Level 0: progressive basis complexity. CUDA and CPU use the resident
        # batched optimizer (B=1), avoiding SciPy/Powell's repeated tensor↔NumPy
        # boundary. MPS stays serial because 3-D grid-sample backward currently
        # falls to CPU and makes the autograd path substantially slower.
        lev0_bases = ["cubic_lite", "cubic", "quintic_lite"]
        use_batched_lev0 = device.type != "mps"

        if use_batched_lev0:
            lev0_patch = PatchSpec(
                ibot=ibbb,
                itop=ittt,
                jbot=jbbb,
                jtop=jttt,
                kbot=kbbb,
                ktop=kttt,
                gi=0,
                gj=0,
                gk=0,
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
            if use_batched_lev0:
                basis0, half_widths0, param_max0 = _get_basis_config(
                    basis_type,
                    nxh0,
                    nyh0,
                    nzh0,
                    device,
                    hfactor=1.0,
                )
                _improve_warp_batched(
                    base,
                    source,
                    weight,
                    mask,
                    state,
                    config,
                    device,
                    [lev0_patch],
                    basis0,
                    half_widths0,
                    param_max0,
                    ii0_flat,
                    jj0_flat,
                    kk0_flat,
                    nxh0,
                    nyh0,
                    nzh0,
                    do_xyz=do_xyz,
                    axis_weights=axis_w,
                    use_penalty=False,
                    pen_fac=0.0,
                    base_clip=base_clip,
                    source_clip=source_clip,
                    max_iter=config.batch_optimizer_iters_lev0,
                    blok_index_vol=blok_index_vol,
                    nblok=nblok,
                )
            else:
                _improve_warp_serial(
                    base,
                    source,
                    weight,
                    mask,
                    state,
                    config,
                    device,
                    ibbb,
                    ittt,
                    jbbb,
                    jttt,
                    kbbb,
                    kttt,
                    basis_type=basis_type,
                    do_xyz=do_xyz,
                    axis_weights=axis_w,
                    use_penalty=False,
                    pen_fac=0.0,
                    base_clip=base_clip,
                    source_clip=source_clip,
                    blok_index_vol=blok_index_vol,
                    nblok=nblok,
                )

        # The batched path doesn't set state.cost (only the serial path does),
        # so recompute the global cost for an accurate level-0 readout.
        if use_batched_lev0:
            with torch.no_grad():
                state.cost = _global_correlation(
                    base, state.warped_source, weight, base_clip, source_clip
                )

        if config.verb >= 1:
            elapsed = time.time() - t0
            if _tqdm is not None and isinstance(pbar, _tqdm):
                pbar.set_postfix_str(f"cost={first_cost:.5f}=>{state.cost:.5f} {elapsed:.1f}s")
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

        flev = config.shrink**lev
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

        # Penalty settings. Ramp with ABSOLUTE level (finer patches => stronger
        # anti-over-warp pressure), not levels-into-this-pass: a pyramid or
        # -inilev run resumes at a high lev_start, and the old (lev-lev_start+1)
        # reset the ramp there, under-penalizing the fine levels and letting
        # them over-warp. For a full run (lev_start=1) this is identical.
        pen_lev = lev**0.333
        pen_fff = config.penalty_factor * min(3.21, pen_lev)
        use_pen = pen_fff > 0 and lev >= config.penalty_first_level
        if lev == config.penalty_first_level:
            pen_fff *= 0.5

        # Basis type
        basis_type = "cubic_lite" if config.use_lite else "cubic"
        if levdone and config.use_quintic:
            basis_type = "quintic_lite" if config.use_lite else "quintic"

        # Workhard passes. workhard is None when disabled; END < 0 means "through
        # the last level" (the loop self-terminates at minpatch), so -workhard 0 -1
        # doubles every level as it reads.
        nlevr = 1
        if config.workhard is not None:
            wh1, wh2 = config.workhard
            if wh2 < 0:
                wh2 = config.max_level
            if wh1 <= lev <= wh2:
                nlevr = 2

        # Generate patch grid and filter
        all_patches = _generate_patch_grid(
            ibbb,
            ittt,
            jbbb,
            jttt,
            kbbb,
            kttt,
            xwid,
            ywid,
            zwid,
            xdel,
            ydel,
            zdel,
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
            basis_type,
            nxh,
            nyh,
            nzh,
            device,
            hfactor=hfactor,
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
        state.opt_steps_weighted = 0
        state.opt_patches_counted = 0
        state.opt_hit_budget = 0
        cost_at_start = state.cost
        # Snapshot the warp so a level that worsens the global cost can be
        # rolled back (see the reject-worse-levels guard at the end of the loop).
        if config.reject_worse_levels:
            saved_xd, saved_yd, saved_zd = state.xd.clone(), state.yd.clone(), state.zd.clone()

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
                f"lev={lev} patch={xwid}x{ywid}x{zwid} patches={n_valid} [{time.time() - t0:.0f}s]",
                end="",
                flush=True,
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
                good = [
                    p
                    for p in phase_patches
                    if (
                        p.itop - p.ibot + 1 == nxh
                        and p.jtop - p.jbot + 1 == nyh
                        and p.ktop - p.kbot + 1 == nzh
                    )
                ]
                odd = [
                    p
                    for p in phase_patches
                    if (
                        p.itop - p.ibot + 1 != nxh
                        or p.jtop - p.jbot + 1 != nyh
                        or p.ktop - p.kbot + 1 != nzh
                    )
                ]

                if good:
                    # BATCHED GPU path (works for B>=1)
                    _improve_warp_batched(
                        base,
                        source,
                        weight,
                        mask,
                        state,
                        config,
                        device,
                        good,
                        basis,
                        half_widths,
                        param_max,
                        ii_flat,
                        jj_flat,
                        kk_flat,
                        nxh,
                        nyh,
                        nzh,
                        do_xyz=do_xyz,
                        axis_weights=axis_w,
                        use_penalty=use_pen,
                        pen_fac=pen_fff,
                        base_clip=base_clip,
                        source_clip=source_clip,
                        blok_index_vol=blok_index_vol,
                        nblok=nblok,
                    )

                # Serial fallback for odd-sized boundary patches
                for p in odd:
                    _improve_warp_serial(
                        base,
                        source,
                        weight,
                        mask,
                        state,
                        config,
                        device,
                        p.ibot,
                        p.itop,
                        p.jbot,
                        p.jtop,
                        p.kbot,
                        p.ktop,
                        basis_type=basis_type,
                        do_xyz=do_xyz,
                        axis_weights=axis_w,
                        use_penalty=use_pen,
                        pen_fac=pen_fff,
                        base_clip=base_clip,
                        source_clip=source_clip,
                        blok_index_vol=blok_index_vol,
                        nblok=nblok,
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
            state.warped_source = warp_image(
                source, state.xd, state.yd, state.zd, mode=config.interp
            )

        # Compute actual global correlation after this level
        with torch.no_grad():
            state.cost = _global_correlation(
                base, state.warped_source, weight, base_clip, source_clip
            )

        # Reject a level that made the global cost worse: restore the previous
        # level's warp and stop. cost is negated correlation (lower = better),
        # so worsening means it went up. A small epsilon avoids stopping on noise.
        if config.reject_worse_levels and state.cost > cost_at_start + 1e-4:
            worsened = state.cost
            with torch.no_grad():
                state.xd, state.yd, state.zd = saved_xd, saved_yd, saved_zd
                state.warped_source = warp_image(
                    source, state.xd, state.yd, state.zd, mode=config.interp
                )
            state.cost = cost_at_start
            if config.verb >= 1:
                msg = (
                    f"lev={lev} worsened cost {cost_at_start:.5f}=>{worsened:.5f}; "
                    f"rolled back and stopping"
                )
                if lev_pbar is not None:
                    lev_pbar.set_postfix_str(msg)
                    lev_pbar.close()
                else:
                    print(f"  {msg}")
            break

        if config.verb >= 1:
            elapsed = time.time() - t0
            # Optimizer-budget readout: mean Adam steps/patch and the share of
            # patches that were still improving when they hit the iter cap (a high
            # % means raising -batch_iters would buy more warp).
            budget_str = ""
            if state.opt_patches_counted > 0:
                mean_iters = state.opt_steps_weighted / state.opt_patches_counted
                hit_pct = 100.0 * state.opt_hit_budget / state.opt_patches_counted
                budget_str = f" iters~{mean_iters:.0f} cap {hit_pct:.0f}%"
            if lev_pbar is not None:
                lev_pbar.set_postfix_str(
                    f"cost={cost_at_start:.5f}=>{state.cost:.5f} "
                    f"({state.patches_done}done {state.patches_skipped}skip){budget_str} {elapsed:.1f}s"
                )
                lev_pbar.close()
            else:
                print(
                    f" done [cost:{cost_at_start:.5f}==>{state.cost:.5f};"
                    f" {state.patches_done} done, {state.patches_skipped} skip]"
                    f"{budget_str} ({elapsed:.1f}s)"
                )

        if config.level_callback is not None:
            config.level_callback(lev, state.xd, state.yd, state.zd, state.warped_source)

        # Early stopping: if this level barely improved cost, skip finer levels
        if config.level_stop_tol > 0 and cost_at_start < 0:
            improvement = abs(state.cost - cost_at_start)
            threshold = config.level_stop_tol * abs(cost_at_start)
            if improvement < threshold:
                if config.verb >= 1:
                    print(
                        f"  Early stop: improvement {improvement:.6f} < "
                        f"threshold {threshold:.6f} ({config.level_stop_tol:.1e})"
                    )
                break


# ---------------------------------------------------------------------------
# Batched GPU patch optimization (the fast path)
# ---------------------------------------------------------------------------


def _gn_normal_eqs_3d(
    w: Tensor,
    g: Tensor,
    hw: Tensor,
    bt: Tensor,
    omega: Tensor,
    wsum: Tensor,
    base_hat: Tensor,
) -> tuple[Tensor, Tensor]:
    """Zero-normalised-NCC Gauss-Newton normal equations for a 3-D patch warp.

    The three-direction generalisation of :func:`_mescaled_gn_normal_eqs`, which
    solves the same problem for a phase-encode-only field. A parameter here is
    (direction, basis function), so the Jacobian gains a column block per active
    direction and the solve is over ``D * nb`` unknowns instead of ``nb``.

    The normalisation is the point. The cost is a *correlation*, so the residual has
    to be taken between zero-mean unit-variance patches; differentiating that
    normalisation is what the ``mdW`` and ``sW`` terms are. Skipping it and
    regressing on raw intensities would solve a least-squares problem the engine is
    not scoring.

    Args:
        w: (B, V) warped patch values at the current parameters.
        g: (D, B, V) source-image gradient sampled at the same locations, one per
            active direction.
        hw: (D, 1, 1) half-width scale per active direction.
        bt: (V, nb) basis, transposed.
        omega: (B, V) per-voxel weight * mask.
        wsum: (B, 1) weight sum per patch.
        base_hat: (B, V) zero-normalised base patches.

    Returns:
        ``hmat`` (B, D*nb, D*nb) and ``grad`` (B, D*nb).
    """
    mw = (omega * w).sum(-1, keepdim=True) / wsum
    sw = ((omega * w * w).sum(-1, keepdim=True) / wsum - mw * mw).clamp_min(1e-12).sqrt()
    res = base_hat - (w - mw) / sw  # (B, V)

    # Steepest-descent images, one block of nb columns per active direction.
    dw = torch.cat([(hw[d] * g[d]).unsqueeze(-1) * bt for d in range(g.shape[0])], dim=-1)
    mdw = (omega.unsqueeze(-1) * dw).sum(1, keepdim=True) / wsum.unsqueeze(-1)
    jn = (dw - mdw) / sw.unsqueeze(-1)
    jnw = jn * omega.unsqueeze(-1)
    hmat = torch.einsum("bvn,bvm->bnm", jnw, jn)
    grad = torch.einsum("bvn,bv->bn", jnw, res)
    return hmat, grad


def _gn_normal_eqs_local(
    w: Tensor,
    g: Tensor,
    hw: Tensor,
    bt: Tensor,
    omega: Tensor,
    base_hat: Tensor,
    kernel: Tensor,
    dims: tuple[int, int, int],
) -> tuple[Tensor, Tensor]:
    """Gauss-Newton normal equations for the **local** Pearson costs.

    Same construction as :func:`_gn_normal_eqs_3d` with one substitution: the
    zero-normalisation uses locally smoothed statistics instead of one mean and one
    standard deviation per patch. Minimising ``sum (a_hat - b_hat)^2`` over locally
    normalised images maximises local correlation, which is what ``lpa`` scores --
    so this is a least-squares surrogate for a cost that has no residual form of its
    own. It is not the reported functional (AFNI aggregates ``z*|z|`` over Fisher-z
    transformed correlations); it only has to point the same way, and the caller
    accepts or rejects every step on the real cost.

    The local mean and standard deviation of the *moving* patch are frozen at the
    current parameters rather than differentiated. That is the standard cheap
    approximation for local-correlation gradients, and it is what keeps the
    Jacobian from needing its own smoothing pass per column -- with ``D*nb`` columns
    over a patch-sized grid, that pass would cost more than the solve it feeds.
    Freezing makes the quadratic model slightly wrong, which under an accept/reject
    loop buys iterations rather than errors.

    Args:
        w: (B, V) warped patch values.
        g: (D, B, V) source gradient at the same locations.
        hw: (D, 1, 1) half-width scale per active direction.
        bt: (V, nb) basis, transposed.
        omega: (B, V) weight * mask.
        base_hat: (B, V) locally normalised base patches.
        kernel: 1-D smoothing kernel defining the neighbourhood.
        dims: (nzh, nyh, nxh) patch shape, so (B, V) can be seen as a grid again.
    """
    from .cost import _batched_separable_smooth_3d

    b, v = w.shape
    nzh, nyh, nxh = dims

    def sm(flat: Tensor) -> Tensor:
        return _batched_separable_smooth_3d(flat.reshape(b, 1, nzh, nyh, nxh), kernel).reshape(b, v)

    sw = sm(omega).clamp_min(1e-10)
    mw = sm(omega * w) / sw
    sd = (sm(omega * w * w) / sw - mw * mw).clamp_min(1e-10).sqrt()

    res = base_hat - (w - mw) / sd
    dw = torch.cat([(hw[d] * g[d]).unsqueeze(-1) * bt for d in range(g.shape[0])], dim=-1)
    jn = dw / sd.unsqueeze(-1)
    jnw = jn * omega.unsqueeze(-1)
    hmat = torch.einsum("bvn,bvm->bnm", jnw, jn)
    grad = torch.einsum("bvn,bv->bn", jnw, res)
    return hmat, grad


def _improve_warp_batched(
    base: Tensor,
    source: Tensor,
    weight: Tensor,
    mask: Tensor,
    state: WarpState,
    config: QwarpConfig,
    device: torch.device,
    patches: list[PatchSpec],
    basis: Tensor,
    half_widths: tuple[float, float, float],
    param_max: float,
    ii_flat: Tensor,
    jj_flat: Tensor,
    kk_flat: Tensor,
    nxh: int,
    nyh: int,
    nzh: int,
    do_xyz: tuple[bool, bool, bool] = (True, True, True),
    axis_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    use_penalty: bool = False,
    pen_fac: float = 0.033333,
    base_clip: tuple[float, float] | None = None,
    source_clip: tuple[float, float] | None = None,
    max_iter: int | None = None,
    blok_index_vol: Tensor | None = None,
    nblok: int = 0,
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
    base_patches = torch.stack(
        [
            base[p.kbot : p.ktop + 1, p.jbot : p.jtop + 1, p.ibot : p.itop + 1].reshape(-1)
            for p in patches
        ]
    )
    weight_patches = torch.stack(
        [
            weight[p.kbot : p.ktop + 1, p.jbot : p.jtop + 1, p.ibot : p.itop + 1].reshape(-1)
            for p in patches
        ]
    )
    mask_patches = torch.stack(
        [
            mask[p.kbot : p.ktop + 1, p.jbot : p.jtop + 1, p.ibot : p.itop + 1].reshape(-1).float()
            for p in patches
        ]
    )

    # Cost selection: blok-based local Pearson (lpc/lpa, AFNI-faithful),
    # the older convolution LPA (lpa_alt), or INCOR (pearson/pearclp).
    use_blok = config.cost_method in ("lpc", "lpa")
    use_conv_lpa = config.cost_method == "lpa_alt"
    # The registry's neighbourhood metrics, evaluated per patch. They need the 3-D
    # block back, which a flat (B, V) patch trivially reshapes to.
    use_patch_metric = config.cost_method in PATCH_METRICS
    blok_prep = None
    if use_blok:
        blok_idx_patches = torch.stack(
            [
                blok_index_vol[
                    p.kbot : p.ktop + 1, p.jbot : p.jtop + 1, p.ibot : p.itop + 1
                ].reshape(-1)
                for p in patches
            ]
        )
        blok_prep = prepare_blok_pairs(blok_idx_patches, nblok)
        _blok_value = lpc_value_pairs if config.cost_method == "lpc" else lpa_value_pairs
    batch_incor = None
    if not (use_blok or use_conv_lpa or use_patch_metric):
        batch_incor = BatchedIncrementalCorrelation(
            method=config.cost_method,
            base_clip=base_clip,
            source_clip=source_clip,
        )
        patch_slices = [(p.ibot, p.itop, p.jbot, p.jtop, p.kbot, p.ktop) for p in patches]
        batch_incor.precompute_fixed_parts(
            base,
            state.warped_source,
            weight,
            patch_slices,
            base_patches=base_patches,
            weight_patches=weight_patches,
        )

    # Pre-compute external penalty if needed (vectorized: global sum minus each patch)
    external_pen = torch.zeros(B, device=device)
    if use_penalty:
        with torch.no_grad():
            je_global, se_global = compute_jacobian_energy(state.xd, state.yd, state.zd)
            energy_global = penalty_energy(je_global, se_global)
            global_energy_sum = energy_global.sum()
            # Gather patch energies as (B,) via stacking
            patch_energies = torch.stack(
                [
                    energy_global[
                        p.kbot : p.ktop + 1, p.jbot : p.jtop + 1, p.ibot : p.itop + 1
                    ].sum()
                    for p in patches
                ]
            )
            external_pen = global_energy_sum - patch_energies

    # Axis weight scales
    _ax_scales = torch.tensor(
        [axis_weights[d] for d in active_dims], device=device, dtype=torch.float32
    )

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
        expand_mat[idx : idx + n_basis, offset : offset + n_basis] = scale * torch.eye(
            n_basis, device=device
        )
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
        full_params = (
            active_params @ expand_mat
        )  # (B, n_active) @ (n_active, n_total) → (B, n_total)

        # Batched basis evaluation: (B, V) displacements
        hxd, hyd, hzd = _eval_warp(basis, full_params, half_widths, do_xyz)

        # Batched compose + interpolate (fused 4-ch: source + warp in one grid_sample)
        warped_vals, ah_xd, ah_yd, ah_zd = _compose(
            source,
            state.xd,
            state.yd,
            state.zd,
            hxd,
            hyd,
            hzd,
            ii_flat,
            jj_flat,
            kk_flat,
            ibots,
            jbots,
            kbots,
            nx,
            ny,
            nz,
            global_warp_3ch=global_warp_3ch,
            base_i=base_i,
            base_j=base_j,
            base_k=base_k,
        )

        warped_vals = warped_vals * mask_patches

        # Batched cost: (B,) (all conventions are higher == better here)
        if use_patch_metric:
            corr = batched_patch_cost(
                config.cost_method,
                base_patches,
                warped_vals,
                weight_patches,
                nzh,
                nyh,
                nxh,
                cc_radius=config.cc_radius,
                mind_radius=config.mind_radius,
            )
        elif use_blok:
            corr = _blok_value(
                base_patches, warped_vals, weight_patches, blok_prep, config.lpc_ppow
            )
        elif use_conv_lpa:
            corr = batched_lpa_cost(
                base_patches,
                warped_vals,
                weight_patches,
                nzh,
                nyh,
                nxh,
                sigma=config.lpa_sigma,
                kernel_type=config.lpa_kernel,
            )
        else:
            # batch_incor is built exactly when `not (use_blok or use_conv_lpa)`,
            # which is this branch's condition.
            assert batch_incor is not None
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

    # Gauss-Newton needs a least-squares surrogate, which the correlation costs
    # have (the zero-normalised residual) and the local-Pearson / descriptor costs
    # do not. Where it does not apply the Adam path is used unchanged.
    # lpa (and lncc) get the *local* surrogate; the plain correlations get the
    # global one. lpc is excluded on purpose: it rewards anti-correlation, and a
    # sum-of-squares residual between normalised patches can only ever pull them
    # together, so the surrogate would point the wrong way for that cost alone.
    gn_local = config.cost_method in ("lpa", "lpa_alt", "lncc")
    gn_global = config.cost_method in ("pearson", "pearclp", "ncc")
    gn_capable = gn_local or gn_global
    use_gn = config.optimizer in ("gn", "hybrid") and gn_capable
    # Hybrid: Gauss-Newton to get close cheaply, then a short Adam pass on the
    # *reported* cost to close the gap the least-squares surrogate leaves. On
    # pearclp, GN lands on AFNI's answer and Adam lands past it; the surrogate is
    # the difference, and it is worth a few autograd steps to recover.
    use_hybrid = config.optimizer == "hybrid" and gn_capable

    if use_gn:
        if state.source_grad_3ch.numel() == 0:
            gz, gy, gx = torch.gradient(source)
            state.source_grad_3ch = torch.stack([gx, gy, gz], dim=0).contiguous()
        source_grad_3ch = state.source_grad_3ch
        bt = basis.reshape(basis.shape[0], -1).t().contiguous()  # (V, n_basis)
        hw_active = torch.tensor(
            [half_widths[d] * axis_weights[d] for d in active_dims],
            device=device,
            dtype=torch.float32,
        ).view(-1, 1, 1)
        omega = (weight_patches * mask_patches).to(torch.float32)
        wsum = omega.sum(-1, keepdim=True).clamp_min(1e-12)
        gn_kernel = _make_kernel_1d(
            config.lpa_kernel, min(config.lpa_sigma, (min(nzh, nyh, nxh) - 1) / 2.0), device
        )
        if gn_local:
            # The base never moves, so its local statistics are computed once.
            from .cost import _batched_separable_smooth_3d

            def _sm(flat: Tensor) -> Tensor:
                return _batched_separable_smooth_3d(
                    flat.reshape(B, 1, nzh, nyh, nxh), gn_kernel
                ).reshape(B, -1)

            sw_b = _sm(omega).clamp_min(1e-10)
            bm = _sm(omega * base_patches) / sw_b
            bs = (_sm(omega * base_patches * base_patches) / sw_b - bm * bm).clamp_min(1e-10).sqrt()
        else:
            bm = (omega * base_patches).sum(-1, keepdim=True) / wsum
            bs = (
                ((omega * base_patches * base_patches).sum(-1, keepdim=True) / wsum - bm * bm)
                .clamp_min(1e-12)
                .sqrt()
            )
        base_hat = (base_patches - bm) / bs

        def gn_normal_eqs(active_params: Tensor) -> tuple[Tensor, Tensor]:
            with torch.no_grad():
                full = active_params @ expand_mat
                hxd_, hyd_, hzd_ = _eval_warp(basis, full, half_widths, do_xyz)
                w_, ax_, ay_, az_ = _compose(
                    source,
                    state.xd,
                    state.yd,
                    state.zd,
                    hxd_,
                    hyd_,
                    hzd_,
                    ii_flat,
                    jj_flat,
                    kk_flat,
                    ibots,
                    jbots,
                    kbots,
                    nx,
                    ny,
                    nz,
                    global_warp_3ch=global_warp_3ch,
                    base_i=base_i,
                    base_j=base_j,
                    base_k=base_k,
                )
                # Sample the source gradient at the very positions the source was
                # sampled at -- recomputed rather than returned, because it is two
                # adds and a clamp against another grid_sample.
                sx = (ax_ + base_i).clamp(-0.499, nx - 0.501)
                sy = (ay_ + base_j).clamp(-0.499, ny - 0.501)
                sz = (az_ + base_k).clamp(-0.499, nz - 0.501)
                gx_, gy_, gz_ = batched_interp_3ch(source_grad_3ch, sx, sy, sz)
                gall = torch.stack([gx_, gy_, gz_], dim=0)[list(active_dims)]
                if gn_local:
                    return _gn_normal_eqs_local(
                        w_ * mask_patches,
                        gall * mask_patches,
                        hw_active,
                        bt,
                        omega,
                        base_hat,
                        gn_kernel,
                        (nzh, nyh, nxh),
                    )
                return _gn_normal_eqs_3d(
                    w_ * mask_patches, gall * mask_patches, hw_active, bt, omega, wsum, base_hat
                )

        def gn_cost(active_params: Tensor) -> Tensor:
            with torch.no_grad():
                return batched_cost(active_params)

        best_params, _best_costs, opt_stats = optimize_warp_params_gauss_newton(
            gn_normal_eqs,
            gn_cost,
            B,
            n_active,
            param_max,
            device,
            max_iter=config.gn_iters,
        )
        if use_hybrid:
            best_params, _best_costs, polish_stats = optimize_warp_params_batched(
                batched_cost,
                B,
                n_active,
                param_max,
                device,
                max_iter=config.hybrid_polish_iters,
                lr=config.batch_optimizer_lr,
                tolerance=config.batch_optimizer_tol,
                patience=config.batch_optimizer_patience,
                init=best_params,
            )
            opt_stats = BatchOptStats(
                steps_run=opt_stats.steps_run + polish_stats.steps_run,
                n_patches=B,
                hit_budget=polish_stats.hit_budget,
            )
    else:
        best_params, _best_costs, opt_stats = optimize_warp_params_batched(
            batched_cost,
            B,
            n_active,
            param_max,
            device,
            max_iter=config.batch_optimizer_iters if max_iter is None else max_iter,
            lr=config.batch_optimizer_lr,
            tolerance=config.batch_optimizer_tol,
            patience=config.batch_optimizer_patience,
        )
    # Accumulate per-level optimizer-budget telemetry (batched path only).
    state.opt_steps_weighted += opt_stats.steps_run * opt_stats.n_patches
    state.opt_patches_counted += opt_stats.n_patches
    state.opt_hit_budget += opt_stats.hit_budget

    # Apply optimized parameters - update global warp AND warped_source in one pass
    with torch.no_grad():
        full_params = best_params @ expand_mat  # reuse pre-built expansion matrix

        hxd, hyd, hzd = _eval_warp(basis, full_params, half_widths, do_xyz)
        warped_vals, ah_xd, ah_yd, ah_zd = _compose(
            source,
            state.xd,
            state.yd,
            state.zd,
            hxd,
            hyd,
            hzd,
            ii_flat,
            jj_flat,
            kk_flat,
            ibots,
            jbots,
            kbots,
            nx,
            ny,
            nz,
            global_warp_3ch=global_warp_3ch,
            base_i=base_i,
            base_j=base_j,
            base_k=base_k,
        )

        # Write back warp AND warped_source. Patches in a phase overlap by a voxel at
        # their seams, so this serial loop is load-bearing: the last patch in lattice
        # order wins, matching AFNI. Any vectorised replacement must go through
        # _dedup_last_wins -- a raw scatter would be nondeterministic there.
        for idx_p, p in enumerate(patches):
            state.xd[p.kbot : p.ktop + 1, p.jbot : p.jtop + 1, p.ibot : p.itop + 1] = ah_xd[
                idx_p
            ].reshape(nzh, nyh, nxh)
            state.yd[p.kbot : p.ktop + 1, p.jbot : p.jtop + 1, p.ibot : p.itop + 1] = ah_yd[
                idx_p
            ].reshape(nzh, nyh, nxh)
            state.zd[p.kbot : p.ktop + 1, p.jbot : p.jtop + 1, p.ibot : p.itop + 1] = ah_zd[
                idx_p
            ].reshape(nzh, nyh, nxh)
            state.warped_source[p.kbot : p.ktop + 1, p.jbot : p.jtop + 1, p.ibot : p.itop + 1] = (
                warped_vals[idx_p].reshape(nzh, nyh, nxh)
            )

    state.patches_done += B

    if config.verb >= 2:
        # Route through tqdm.write so it doesn't stomp on the level progress bar.
        msg = f"  phase: B={B} cost={state.cost:.5f}"
        if _tqdm is not None:
            _tqdm.write(msg)
        else:
            print(msg)


# ---------------------------------------------------------------------------
# Source-batched (many-to-one) qwarp: N sources sharing one reference
# ---------------------------------------------------------------------------
#
# The single-volume path above batches over the PATCHES of one volume. When the
# volumes are small, one volume's patch count doesn't fill the GPU (measured:
# ~21% util, 2 GB on 100 small EPI volumes). Since all N volumes register to the
# SAME base, they share the patch lattice, weight, mask and blok assignment; only
# the source and the per-volume warp differ. So we can stack N volumes into one
# batch of size N*P and optimize them together, N× the arithmetic per launch.
#
# Correctness contract: qwarp_batch reproduces N independent qwarp() calls to
# sub-voxel. The only thing that couples the batch is gradient clipping, which is
# made per-volume-group in the optimizer (clip_group_size=P); everything else
# (Adam, per-patch convergence, penalty, smoothing, reject-worse-levels) is
# per-volume, with a per-volume active mask so a volume that would stop early in
# the single path stops here too.


@dataclass
class MultiWarpState:
    """Per-volume warp state for the source-batched path.

    Holds N stacked displacement fields and warped sources on the padded grid.
    Mirrors WarpState but with a leading volume axis; per-volume scalars are
    tensors/lists of length N.
    """

    xd_all: Tensor  # (N, nz, ny, nx)
    yd_all: Tensor
    zd_all: Tensor
    warped_all: Tensor  # (N, nz, ny, nx)

    nx: int = 0
    ny: int = 0
    nz: int = 0

    # Per-level optimizer-budget telemetry (summed across volumes).
    opt_steps_weighted: int = 0
    opt_patches_counted: int = 0
    opt_hit_budget: int = 0
    patches_done: int = 0
    patches_skipped: int = 0


def _improve_warp_batched_multi(
    base: Tensor,
    sources_all: Tensor,
    weight: Tensor,
    mask: Tensor,
    mstate: MultiWarpState,
    config: QwarpConfig,
    device: torch.device,
    active_idx: Tensor,
    patches: list[PatchSpec],
    basis: Tensor,
    half_widths: tuple[float, float, float],
    param_max: float,
    ii_flat: Tensor,
    jj_flat: Tensor,
    kk_flat: Tensor,
    nxh: int,
    nyh: int,
    nzh: int,
    do_xyz: tuple[bool, bool, bool] = (True, True, True),
    axis_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    use_penalty: bool = False,
    pen_fac: float = 0.033333,
    base_clip: tuple[float, float] | None = None,
    source_clip: tuple[float, float] | None = None,
    max_iter: int | None = None,
    blok_index_vol: Tensor | None = None,
    nblok: int = 0,
) -> None:
    """Optimize one checkerboard phase across ALL active volumes at once.

    Batch layout is row-major (volume, patch): rows [v0p0..v0pP, v1p0..v1pP, ...].
    Base/weight/mask/blok are shared (one reference), so they are built once for
    the P patches and tiled over the N active volumes. The source and warp state
    are per-volume via the ``_multi`` compose primitive.
    """
    P = len(patches)
    if P == 0:
        return
    Na = int(active_idx.numel())
    if Na == 0:
        return

    nx, ny, nz = mstate.nx, mstate.ny, mstate.nz
    n_basis = basis.shape[0]

    active_dims = [d for d in range(3) if do_xyz[d]]
    n_active = len(active_dims) * n_basis
    n_total = 3 * n_basis
    if n_active == 0:
        return

    B = Na * P  # flat optimizer batch

    # Shared patch geometry (one reference grid for all volumes). All patches are
    # the same size on a regular lattice, so build a single (P, V) flat gather
    # index and pull base/weight/mask/blok in ONE op each -- the per-patch Python
    # stack() loops were launching hundreds of tiny kernels per phase (P up to a
    # few hundred at fine levels) and starving the GPU (launch-bound, not
    # compute-bound). Same trick vectorizes the penalty and write-back below.
    ibots = torch.tensor([p.ibot for p in patches], device=device, dtype=torch.float32)
    jbots = torch.tensor([p.jbot for p in patches], device=device, dtype=torch.float32)
    kbots = torch.tensor([p.kbot for p in patches], device=device, dtype=torch.float32)
    base_i = ibots[:, None] + ii_flat[None, :]  # (P, V)
    base_j = jbots[:, None] + jj_flat[None, :]
    base_k = kbots[:, None] + kk_flat[None, :]

    # Flat row-major index of every patch voxel into a (nz,ny,nx) volume. Matches
    # the reshape(-1) order of base[kslice, jslice, islice] (k slowest, i fastest).
    flat_idx = (base_k.long() * ny + base_j.long()) * nx + base_i.long()  # (P, V)
    flat_all = flat_idx.reshape(-1)  # (P*V,)

    base_patches = base.reshape(-1)[flat_idx]  # (P, V)
    weight_patches = weight.reshape(-1)[flat_idx]
    mask_patches = mask.reshape(-1)[flat_idx].float()
    base_rep = base_patches.repeat(Na, 1)  # (B, V)
    weight_rep = weight_patches.repeat(Na, 1)  # (B, V)

    # Only the *local* costs (within-patch) are shareable across volumes here:
    # lpc/lpa are scored over the shared blok lattice, lpa_alt is a separable
    # convolution over the patch. INCOR (pearson/pearclp) carries per-volume
    # "outside-patch" fixed statistics, so it can't be tiled -- qwarp_batch
    # guards against those methods before reaching this point.
    use_blok = config.cost_method in ("lpc", "lpa")
    blok_prep = None
    _blok_value = None
    if use_blok:
        assert blok_index_vol is not None
        # (B, V); blok assignment identical per volume, so gather once and tile.
        blok_idx_patches = blok_index_vol.reshape(-1)[flat_idx].repeat(Na, 1)
        blok_prep = prepare_blok_pairs(blok_idx_patches, nblok)
        _blok_value = lpc_value_pairs if config.cost_method == "lpc" else lpa_value_pairs

    # Per-volume source + warp for the active subset.
    src = sources_all[active_idx]  # (Na, nz, ny, nx)
    xd_a = mstate.xd_all[active_idx]
    yd_a = mstate.yd_all[active_idx]
    zd_a = mstate.zd_all[active_idx]
    global_warp_3ch = torch.stack([xd_a, yd_a, zd_a], dim=1)  # (Na, 3, nz, ny, nx)

    # External penalty (energy of the rest of each volume's warp), per (vol, patch).
    # Per-volume total energy needs the full-volume jacobian (loop over the few
    # active volumes); the per-patch energy is one gather+sum, not a P loop.
    external_pen = torch.zeros(B, device=device)
    if use_penalty:
        with torch.no_grad():
            for a in range(Na):
                v = int(active_idx[a])
                je, se = compute_jacobian_energy(
                    mstate.xd_all[v], mstate.yd_all[v], mstate.zd_all[v]
                )
                energy = penalty_energy(je, se).reshape(-1)
                patch_e = energy[flat_idx].sum(dim=1)  # (P,)
                external_pen[a * P : (a + 1) * P] = energy.sum() - patch_e

    # Active-param -> full-param expansion with axis weights (shared).
    expand_mat = torch.zeros(n_active, n_total, device=device)
    idx = 0
    for dim_i in active_dims:
        offset = dim_i * n_basis
        scale = axis_weights[dim_i]
        expand_mat[idx : idx + n_basis, offset : offset + n_basis] = scale * torch.eye(
            n_basis, device=device
        )
        idx += n_basis

    def batched_cost(active_params: Tensor) -> Tensor:
        """(B, n_active) -> (B,) costs, row-major (volume, patch). Differentiable."""
        full_params = active_params @ expand_mat  # (B, n_total)
        hxd, hyd, hzd = evaluate_patch_warp_batched(basis, full_params, half_widths, do_xyz)
        # (B, V) -> (Na, P, V) for the source-batched compose
        hxd = hxd.reshape(Na, P, -1)
        hyd = hyd.reshape(Na, P, -1)
        hzd = hzd.reshape(Na, P, -1)
        warped_vals, ah_xd, ah_yd, ah_zd = batched_compose_and_interpolate_multi(
            src,
            global_warp_3ch,
            hxd,
            hyd,
            hzd,
            base_i,
            base_j,
            base_k,
            nx,
            ny,
            nz,
        )
        warped_vals = warped_vals * mask_patches[None]  # (Na, P, V)
        warped_flat = warped_vals.reshape(B, -1)

        if use_blok:
            assert _blok_value is not None
            corr = _blok_value(base_rep, warped_flat, weight_rep, blok_prep, config.lpc_ppow)
        else:  # use_conv_lpa
            corr = batched_lpa_cost(
                base_rep,
                warped_flat,
                weight_rep,
                nzh,
                nyh,
                nxh,
                sigma=config.lpa_sigma,
                kernel_type=config.lpa_kernel,
            )
        cost = -corr

        if use_penalty and pen_fac > 0:
            ah_x3d = ah_xd.reshape(B, nzh, nyh, nxh)
            ah_y3d = ah_yd.reshape(B, nzh, nyh, nxh)
            ah_z3d = ah_zd.reshape(B, nzh, nyh, nxh)
            cost = cost + compute_penalty_batched(ah_x3d, ah_y3d, ah_z3d, pen_fac, external_pen)
        return cost

    best_params, _best_costs, opt_stats = optimize_warp_params_batched(
        batched_cost,
        B,
        n_active,
        param_max,
        device,
        max_iter=config.batch_optimizer_iters if max_iter is None else max_iter,
        lr=config.batch_optimizer_lr,
        tolerance=config.batch_optimizer_tol,
        patience=config.batch_optimizer_patience,
        clip_group_size=P,  # each volume clips exactly as its own single-volume run
    )
    mstate.opt_steps_weighted += opt_stats.steps_run * opt_stats.n_patches
    mstate.opt_patches_counted += opt_stats.n_patches
    mstate.opt_hit_budget += opt_stats.hit_budget

    with torch.no_grad():
        full_params = best_params @ expand_mat
        hxd, hyd, hzd = evaluate_patch_warp_batched(basis, full_params, half_widths, do_xyz)
        hxd = hxd.reshape(Na, P, -1)
        hyd = hyd.reshape(Na, P, -1)
        hzd = hzd.reshape(Na, P, -1)
        warped_vals, ah_xd, ah_yd, ah_zd = batched_compose_and_interpolate_multi(
            src,
            global_warp_3ch,
            hxd,
            hyd,
            hzd,
            base_i,
            base_j,
            base_k,
            nx,
            ny,
            nz,
        )
        # Scatter all patches of all active volumes in one indexed assignment per
        # field. (Replaces an Na*P Python loop that fired a tiny kernel per patch.)
        # Patches within a phase are NOT disjoint -- the lattice steps (width-1)//2, so
        # same-parity patches share a voxel at their seams -- and `index_put_` with
        # duplicate targets leaves the winner unspecified. Deduplicate to "last patch in
        # lattice order wins", which is what the solo path's serial patch loop does, so
        # this is both deterministic and the behaviour qwarp_batch is matched against.
        w_dst, w_src = _dedup_last_wins(flat_all)
        rows = active_idx.unsqueeze(1)  # (Na, 1)
        cols = w_dst.unsqueeze(0)  # (1, U)
        mstate.xd_all.view(mstate.xd_all.shape[0], -1)[rows, cols] = ah_xd.reshape(Na, -1)[:, w_src]
        mstate.yd_all.view(mstate.yd_all.shape[0], -1)[rows, cols] = ah_yd.reshape(Na, -1)[:, w_src]
        mstate.zd_all.view(mstate.zd_all.shape[0], -1)[rows, cols] = ah_zd.reshape(Na, -1)[:, w_src]
        mstate.warped_all.view(mstate.warped_all.shape[0], -1)[rows, cols] = warped_vals.reshape(
            Na, -1
        )[:, w_src]

    mstate.patches_done += B


def _warpomatic_multi(
    base: Tensor,
    sources_all: Tensor,
    weight: Tensor,
    mask: Tensor,
    mstate: MultiWarpState,
    config: QwarpConfig,
    device: torch.device,
) -> None:
    """Source-batched multi-level loop; N volumes share one reference.

    Structurally identical to :func:`_warpomatic` but every per-volume decision
    (cost readout, penalty, level smoothing, reject-worse-levels, early stop) is
    made per volume via an ``active`` mask, so each volume follows the same level
    schedule it would under a solo :func:`qwarp` call.
    """
    N = mstate.xd_all.shape[0]
    nx, ny, nz = mstate.nx, mstate.ny, mstate.nz
    t0 = time.time()
    from .weight import _gaussian_smooth_3d

    do_x = not (config.warp_flags & 1)
    do_y = not (config.warp_flags & 2)
    do_z = not (config.warp_flags & 4)
    do_xyz = (do_x, do_y, do_z)
    axis_w = config.axis_weights

    imin, imax, jmin, jmax, kmin, kmax = _autobox(weight)
    base_clip = _auto_clip(base.reshape(-1), weight.reshape(-1))
    # Source clip is per-volume in the single path; use each volume's own but a
    # shared base clip. Compute per-volume source clips up front.
    source_clips = [_auto_clip(sources_all[v].reshape(-1), weight.reshape(-1)) for v in range(N)]

    blok_index_vol = None
    nblok = 0
    if config.cost_method in ("lpc", "lpa"):
        bs = assign_bloks((nz, ny, nx), (1.0, 1.0, 1.0), "tohd", config.blok_rad, mask=(mask > 0))
        blok_index_vol = bs.index.reshape(nz, ny, nx)
        nblok = bs.nblok

    # Per-volume global cost (negated correlation).
    def _vol_cost(v: int) -> float:
        with torch.no_grad():
            return _global_correlation(
                base, mstate.warped_all[v], weight, base_clip, source_clips[v]
            )

    costs = [_vol_cost(v) for v in range(N)]
    active = torch.ones(N, dtype=torch.bool, device=device)

    # --- Level 0 bounds (also used to size level 1+) ---
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

    # --- Level 0: global warp (single patch), all volumes ---
    if config.start_level == 0:
        lev0_patch = PatchSpec(
            ibot=ibbb,
            itop=ittt,
            jbot=jbbb,
            jtop=jttt,
            kbot=kbbb,
            ktop=kttt,
            gi=0,
            gj=0,
            gk=0,
        )
        nxh0, nyh0, nzh0 = ittt - ibbb + 1, jttt - jbbb + 1, kttt - kbbb + 1
        kk0, jj0, ii0 = torch.meshgrid(
            torch.arange(nzh0, dtype=torch.float32, device=device),
            torch.arange(nyh0, dtype=torch.float32, device=device),
            torch.arange(nxh0, dtype=torch.float32, device=device),
            indexing="ij",
        )
        ii0f, jj0f, kk0f = ii0.reshape(-1), jj0.reshape(-1), kk0.reshape(-1)
        all_idx = torch.arange(N, device=device)

        lev0_iter = ["cubic_lite", "cubic", "quintic_lite"]
        if config.verb >= 1 and _tqdm is not None:
            lev0_iter = _tqdm(
                lev0_iter,
                desc=f"batch[{N}] lev=0",
                leave=True,
                bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}",
            )
        for basis_type in lev0_iter:
            basis0, hw0, pmax0 = _get_basis_config(
                basis_type, nxh0, nyh0, nzh0, device, hfactor=1.0
            )
            _improve_warp_batched_multi(
                base,
                sources_all,
                weight,
                mask,
                mstate,
                config,
                device,
                all_idx,
                [lev0_patch],
                basis0,
                hw0,
                pmax0,
                ii0f,
                jj0f,
                kk0f,
                nxh0,
                nyh0,
                nzh0,
                do_xyz=do_xyz,
                axis_weights=axis_w,
                use_penalty=False,
                pen_fac=0.0,
                base_clip=base_clip,
                source_clip=None,
                max_iter=config.batch_optimizer_iters_lev0,
                blok_index_vol=blok_index_vol,
                nblok=nblok,
            )
        costs = [_vol_cost(v) for v in range(N)]
        if config.verb >= 1 and _tqdm is not None and isinstance(lev0_iter, _tqdm):
            lev0_iter.set_postfix_str(f"cost~{sum(costs) / N:.5f} {time.time() - t0:.1f}s")
            lev0_iter.close()

    # --- Levels 1..N ---
    xwid0, ywid0, zwid0 = ittt - ibbb + 1, jttt - jbbb + 1, kttt - kbbb + 1
    max_patch_lev1 = max(1, int(max(xwid0, ywid0, zwid0) * config.shrink))
    ngmin = max(config.minpatch, 5)
    if ngmin % 2 == 0:
        ngmin -= 1

    levdone = False
    lev_start = max(1, config.start_level)
    for lev in range(lev_start, config.max_level + 1):
        if levdone or not bool(active.any()):
            break

        flev = config.shrink**lev
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

        pen_lev = lev**0.333
        pen_fff = config.penalty_factor * min(3.21, pen_lev)
        use_pen = pen_fff > 0 and lev >= config.penalty_first_level
        if lev == config.penalty_first_level:
            pen_fff *= 0.5

        basis_type = "cubic_lite" if config.use_lite else "cubic"
        if levdone and config.use_quintic:
            basis_type = "quintic_lite" if config.use_lite else "quintic"

        nlevr = 1
        if config.workhard is not None:
            wh1, wh2 = config.workhard
            if wh2 < 0:
                wh2 = config.max_level
            if wh1 <= lev <= wh2:
                nlevr = 2

        all_patches = _generate_patch_grid(
            ibbb,
            ittt,
            jbbb,
            jttt,
            kbbb,
            kttt,
            xwid,
            ywid,
            zwid,
            xdel,
            ydel,
            zdel,
        )
        valid_patches = _filter_patches(all_patches, weight, mask, nx, ny, nz)
        phases = _checkerboard_phases(valid_patches)

        nxh, nyh, nzh = xwid, ywid, zwid
        max_patch = max(nxh, nyh, nzh)
        hfactor = _compute_hfactor(max_patch, max_patch_lev1, config.hfactor_q)
        basis, half_widths, param_max = _get_basis_config(
            basis_type,
            nxh,
            nyh,
            nzh,
            device,
            hfactor=hfactor,
        )
        kk_p, jj_p, ii_p = torch.meshgrid(
            torch.arange(nzh, dtype=torch.float32, device=device),
            torch.arange(nyh, dtype=torch.float32, device=device),
            torch.arange(nxh, dtype=torch.float32, device=device),
            indexing="ij",
        )
        ii_flat, jj_flat, kk_flat = ii_p.reshape(-1), jj_p.reshape(-1), kk_p.reshape(-1)

        active_idx = torch.nonzero(active, as_tuple=False).flatten()
        # Snapshot for per-volume reject-worse-levels rollback.
        cost_at_start = list(costs)
        saved = None
        if config.reject_worse_levels:
            saved = (
                mstate.xd_all[active_idx].clone(),
                mstate.yd_all[active_idx].clone(),
                mstate.zd_all[active_idx].clone(),
            )

        n_valid = sum(len(ph) for ph in phases)
        lev_pbar = None
        if config.verb >= 1 and _tqdm is not None:
            lev_pbar = _tqdm(
                total=n_valid * nlevr * int(active_idx.numel()),
                desc=f"batch[{int(active_idx.numel())}] lev={lev} {xwid}x{ywid}x{zwid}",
                bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}",
                leave=True,
            )

        for _pass in range(nlevr):
            phase_order = list(range(8))
            if (lev + _pass) % 2 == 1:
                phase_order = phase_order[::-1]
            for phase_idx in phase_order:
                phase_patches = phases[phase_idx]
                if not phase_patches:
                    continue
                good = [
                    p
                    for p in phase_patches
                    if (
                        p.itop - p.ibot + 1 == nxh
                        and p.jtop - p.jbot + 1 == nyh
                        and p.ktop - p.kbot + 1 == nzh
                    )
                ]
                if good:
                    _improve_warp_batched_multi(
                        base,
                        sources_all,
                        weight,
                        mask,
                        mstate,
                        config,
                        device,
                        active_idx,
                        good,
                        basis,
                        half_widths,
                        param_max,
                        ii_flat,
                        jj_flat,
                        kk_flat,
                        nxh,
                        nyh,
                        nzh,
                        do_xyz=do_xyz,
                        axis_weights=axis_w,
                        use_penalty=use_pen,
                        pen_fac=pen_fff,
                        base_clip=base_clip,
                        source_clip=None,
                        blok_index_vol=blok_index_vol,
                        nblok=nblok,
                    )
                # Odd boundary patches (different size) don't batch cleanly across
                # volumes; fall back to per-volume serial for just those.
                odd = [p for p in phase_patches if p not in good]
                for v in active_idx.tolist():
                    svd = _single_state_view(mstate, v)
                    for p in odd:
                        _improve_warp_serial(
                            base,
                            sources_all[v],
                            weight,
                            mask,
                            svd,
                            config,
                            device,
                            p.ibot,
                            p.itop,
                            p.jbot,
                            p.jtop,
                            p.kbot,
                            p.ktop,
                            basis_type=basis_type,
                            do_xyz=do_xyz,
                            axis_weights=axis_w,
                            use_penalty=use_pen,
                            pen_fac=pen_fff,
                            base_clip=base_clip,
                            source_clip=source_clips[v],
                            blok_index_vol=blok_index_vol,
                            nblok=nblok,
                        )
                if lev_pbar is not None:
                    lev_pbar.update(len(phase_patches) * int(active_idx.numel()))

        # Per-volume level smoothing + cost + reject/early-stop.
        newly_inactive = []
        with torch.no_grad():
            for ai, v in enumerate(active_idx.tolist()):
                mstate.xd_all[v] = _gaussian_smooth_3d(mstate.xd_all[v], 1.5)
                mstate.yd_all[v] = _gaussian_smooth_3d(mstate.yd_all[v], 1.5)
                mstate.zd_all[v] = _gaussian_smooth_3d(mstate.zd_all[v], 1.5)
                if config.maxdisp > 0:
                    mstate.xd_all[v].clamp_(-config.maxdisp, config.maxdisp)
                    mstate.yd_all[v].clamp_(-config.maxdisp, config.maxdisp)
                    mstate.zd_all[v].clamp_(-config.maxdisp, config.maxdisp)
                mstate.warped_all[v] = warp_image(
                    sources_all[v],
                    mstate.xd_all[v],
                    mstate.yd_all[v],
                    mstate.zd_all[v],
                    mode=config.interp,
                )
                new_cost = _vol_cost(v)
                if config.reject_worse_levels and new_cost > cost_at_start[v] + 1e-4:
                    mstate.xd_all[v] = saved[0][ai]
                    mstate.yd_all[v] = saved[1][ai]
                    mstate.zd_all[v] = saved[2][ai]
                    mstate.warped_all[v] = warp_image(
                        sources_all[v],
                        mstate.xd_all[v],
                        mstate.yd_all[v],
                        mstate.zd_all[v],
                        mode=config.interp,
                    )
                    costs[v] = cost_at_start[v]
                    newly_inactive.append(v)
                    continue
                costs[v] = new_cost
                if config.level_stop_tol > 0 and cost_at_start[v] < 0:
                    improvement = abs(new_cost - cost_at_start[v])
                    if improvement < config.level_stop_tol * abs(cost_at_start[v]):
                        newly_inactive.append(v)

        for v in newly_inactive:
            active[v] = False

        if lev_pbar is not None:
            avg = sum(costs) / N
            lev_pbar.set_postfix_str(
                f"cost~{avg:.5f} active={int(active.sum())}/{N} {time.time() - t0:.1f}s"
            )
            lev_pbar.close()


def _single_state_view(mstate: MultiWarpState, v: int) -> WarpState:
    """Wrap one volume's tensors as a WarpState for the serial odd-patch path."""
    s = WarpState(nx=mstate.nx, ny=mstate.ny, nz=mstate.nz)
    s.xd = mstate.xd_all[v]
    s.yd = mstate.yd_all[v]
    s.zd = mstate.zd_all[v]
    s.warped_source = mstate.warped_all[v]
    return s


def qwarp_batch(
    base: Tensor,
    sources: Tensor,
    weight: Tensor | None = None,
    mask: Tensor | None = None,
    config: QwarpConfig | None = None,
    device: torch.device | None = None,
    pad: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Nonlinear-warp a batch of sources onto ONE shared base (many-to-one).

    Equivalent to calling :func:`qwarp` once per source volume against the same
    ``base`` (verified to sub-voxel), but all volumes are optimized together so a
    batch of small volumes fills the GPU instead of running one-at-a-time at low
    utilization. Base-derived quantities (padding, weight, mask, blok lattice)
    are computed once and shared.

    Args:
        base: (nz, ny, nx) shared reference image.
        sources: (N, nz, ny, nx) source volumes to warp to ``base``.
        weight, mask: optional shared (nz, ny, nx) weight/mask (auto if None).
        config: QwarpConfig (defaults if None). ``initial_warp`` / pyramid are
            not supported here; use per-volume :func:`qwarp` for those.
        device: torch device (inferred from base if None).
        pad: AFNI-style internal zero-padding (default True).

    Returns:
        (warped, xd, yd, zd):
          - warped: (N, nz, ny, nx) each source warped to base (original size).
          - xd/yd/zd: (N, nz_pad, ny_pad, nx_pad) displacement fields (voxels).
    """
    if config is None:
        config = QwarpConfig()
    if device is None:
        device = base.device
    if sources.dim() != 4:
        raise ValueError(f"sources must be (N, nz, ny, nx); got {tuple(sources.shape)}")
    if config.cost_method not in ("lpc", "lpa", "lpa_alt"):
        # INCOR (pearson/pearclp) uses per-volume outside-patch statistics that
        # can't be shared across the batch. Those are cheap per-volume anyway.
        raise NotImplementedError(
            f"qwarp_batch supports local costs (lpc/lpa/lpa_alt); "
            f"cost_method={config.cost_method!r} needs per-volume qwarp()."
        )
    if config.pyramid_factor > 1:
        raise NotImplementedError(
            "qwarp_batch does not support the resolution pyramid; use qwarp() per volume."
        )

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    base = base.float().to(device)
    sources = sources.float().to(device)
    N = sources.shape[0]
    nz_orig, ny_orig, nx_orig = base.shape

    if pad:
        pad_x, pad_y, pad_z = _compute_padding(nx_orig, ny_orig, nz_orig)
    else:
        pad_x, pad_y, pad_z = 0, 0, 0

    do_pad = pad_x > 0 or pad_y > 0 or pad_z > 0
    if do_pad:
        base_p = _pad_volume(base, pad_x, pad_y, pad_z)
        sources_p = torch.stack([_pad_volume(sources[v], pad_x, pad_y, pad_z) for v in range(N)])
    else:
        base_p = base
        sources_p = sources
    nz, ny, nx = base_p.shape
    predicted, available = plan_nonlinear_memory((nz, ny, nx), device, "qwarp", n_sources=N)
    if predicted > available and config.verb >= 1:
        print(
            f"WARNING: estimated batched qwarp peak {predicted / 2**30:.1f} GiB exceeds "
            f"the {available / 2**30:.1f} GiB safe memory budget; reduce the source batch."
        )

    if weight is None:
        weight_p = compute_weight_image(base_p)
    else:
        w = weight.float().to(device)
        weight_p = _pad_volume(w, pad_x, pad_y, pad_z) if do_pad else w
    if mask is None:
        mask_p = (weight_p > 0).byte()
    else:
        m = mask.byte().to(device)
        mask_p = _pad_volume(m.float(), pad_x, pad_y, pad_z).byte() if do_pad else m

    mstate = MultiWarpState(
        xd_all=torch.zeros(N, nz, ny, nx, device=device),
        yd_all=torch.zeros(N, nz, ny, nx, device=device),
        zd_all=torch.zeros(N, nz, ny, nx, device=device),
        warped_all=sources_p.clone(),
        nx=nx,
        ny=ny,
        nz=nz,
    )

    _warpomatic_multi(base_p, sources_p, weight_p, mask_p, mstate, config, device)

    # Final warped images: single pass through each total warp with final_interp.
    warped = torch.empty(N, nz_orig, ny_orig, nx_orig, device=device)
    for v in range(N):
        wf = warp_image(
            sources_p[v],
            mstate.xd_all[v],
            mstate.yd_all[v],
            mstate.zd_all[v],
            mode=config.final_interp,
        )
        warped[v] = wf[pad_z : pad_z + nz_orig, pad_y : pad_y + ny_orig, pad_x : pad_x + nx_orig]

    return warped, mstate.xd_all, mstate.yd_all, mstate.zd_all


# ---------------------------------------------------------------------------
# Joint multi-echo TE-scaled PE-only polish (locomoco hand-off)
# ---------------------------------------------------------------------------
#
# The multi-echo 3-D-EPI partition/PE wiggle scales linearly with echo time: one
# shared displacement field ``w`` (echo-1 scale) and echo ``e`` sees ``alpha_e * w``
# with ``alpha_e = TE_e / TE_1`` FIXED (not fitted). locomoco estimates ``w`` well
# but leaves a residual its cross-correlation/flow search can't resolve. This polish
# seeds that ``w`` and refines it with a few fine nonlinear levels under the JOINT
# objective ``sum_e lpa(warp(source_e, alpha_e*w), base_e)`` -- one PE-only field,
# every echo constraining it at its own TE scale (late echoes see the big shift,
# early echoes carry the SNR). Single-echo is the E=1, alpha=[1] special case.
#
# This is NOT qwarp_batch (which solves N *independent* warps). Here all echoes
# share ONE parameter set; only the sampled displacement is scaled per echo.


def _weighted_pearson_patches(base: Tensor, warped: Tensor, weight: Tensor) -> Tensor:
    """Weighted Pearson correlation over the last axis, per leading index.

    ``base``/``warped`` are ``(..., V)`` and ``weight`` broadcasts to them; returns
    ``(...)`` correlations. Same-modality patch NCC for the qwarp polish -- cheap
    (a few weighted reductions), differentiable, and linear enough to admit a
    Gauss-Newton update later. Variances are floored (never zeroed) so a flat patch
    gives r=0 with a finite gradient rather than a NaN."""
    sw = weight.sum(-1).clamp_min(1e-6)
    mx = (weight * base).sum(-1) / sw
    my = (weight * warped).sum(-1) / sw
    vx = (weight * base * base).sum(-1) / sw - mx * mx
    vy = (weight * warped * warped).sum(-1) / sw - my * my
    cxy = (weight * base * warped).sum(-1) / sw - mx * my
    return cxy / (vx.clamp_min(1e-12) * vy.clamp_min(1e-12)).sqrt()


def _mescaled_gn_normal_eqs(
    w: Tensor,
    g: Tensor,
    a4hw: Tensor,
    bt: Tensor,
    omega: Tensor,
    wsum: Tensor,
    base_hat: Tensor,
) -> tuple[Tensor, Tensor]:
    """Assemble the batched zero-normalised-NCC Gauss-Newton normal equations.

    Pure tensor-in/tensor-out extraction of the per-iteration hot block in
    :func:`_improve_warp_batched_mescaled`'s ``_gn_solve``: it produces the
    steepest-descent images and forms ``(hmat, grad) = (JᵀJ, Jᵀres)`` for the
    ``(E,B,V,nb)`` Jacobian. This is the one part of the default ncc/gn path that is a
    genuine multi-op, bandwidth-bound elementwise chain -- ``dW``, ``mdW``, ``jn``,
    ``jnw`` each materialise a full ``(E,B,V,nb)`` tensor in eager -- so it is the
    piece :func:`_maybe_compile` can actually fuse (the surrounding interpolation is
    ``grid_sample``, a vendor kernel with nothing to fuse). Pulled to module scope,
    taking only tensors, so ``torch.compile`` sees a clean graph with stable shapes
    across frames. ``a4hw = alpha_e * half_width_pe`` folds the two python scalars in
    so the compiled signature carries no recompiling constant.

    Args:
        w: ``(E, B, V)`` warped patch values at the current params.
        g: ``(E, B, V)`` PE image gradient sampled at the same locations.
        a4hw: ``(E, 1, 1, 1)`` per-echo ``alpha_e * hw_pe``.
        bt: ``(V, nb)`` basis transpose.
        omega: ``(1, B, V)`` per-voxel weight*mask.
        wsum: ``(1, B, 1)`` weight sum per patch.
        base_hat: ``(E, B, V)`` zero-normalised reference patches.

    Returns:
        ``hmat`` ``(B, nb, nb)`` and ``grad`` ``(B, nb)``.
    """
    mW = (omega * w).sum(-1, keepdim=True) / wsum
    sW = ((omega * w * w).sum(-1, keepdim=True) / wsum - mW * mW).clamp_min(1e-12).sqrt()
    res = (w - mW) / sW - base_hat  # (E, B, V)
    dW = a4hw * g.unsqueeze(-1) * bt  # (E, B, V, nb) steepest-descent images
    mdW = (omega.unsqueeze(-1) * dW).sum(2, keepdim=True) / wsum.unsqueeze(-1)
    jn = (dW - mdW) / sW.unsqueeze(-1)  # zero-normalised Jacobian
    jnw = jn * omega.unsqueeze(-1)
    hmat = torch.einsum("ebvn,ebvm->bnm", jnw, jn)  # (B, nb, nb)
    grad = torch.einsum("ebvn,ebv->bn", jnw, res)  # (B, nb)
    return hmat, grad


@dataclass
class _MEPhasePlan:
    """Frame-invariant per-phase gather for the ME-scaled polish.

    Everything here depends only on the fixed geometry (patch lattice + base
    references), so it is built once and reused across all frames of a series.
    ``gather_idx`` is the flat ``(B, V)`` index into ``vol.reshape(-1)`` -- one
    advanced-index op replaces the per-patch ``torch.stack`` loop.

    Patches in one checkerboard phase are NOT quite disjoint: the lattice steps
    ``(width-1)//2``, so same-parity patches abut with a one-voxel overlap (more where
    the sweep snaps the last patch back inside the box). Writing the solved warp
    straight through ``gather_idx`` therefore hits duplicate targets, and
    ``index_put_`` leaves the winner unspecified -- run-to-run drift of a few tenths of
    a voxel. ``write_dst``/``write_src`` are the deduplicated write: one entry per
    target voxel, taking the last patch in lattice order, which is what AFNI's serial
    patch-by-patch overwrite does."""

    gather_idx: Tensor  # (B, V) long
    write_dst: Tensor  # (U,) unique flat targets
    write_src: Tensor  # (U,) positions into the flattened (B*V) patch values
    ibots: Tensor  # (B,) float patch origins
    jbots: Tensor
    kbots: Tensor
    base_i: Tensor  # (B, V) voxel coords
    base_j: Tensor
    base_k: Tensor
    weight_patches: Tensor  # (B, V)
    mask_patches: Tensor  # (B, V)
    base_echo_patches: Tensor  # (E, B, V)
    blok_prep: object | None  # BlokPairs for lpc/lpa; None for ncc
    B: int


@dataclass
class _MELevelPlan:
    nxh: int
    nyh: int
    nzh: int
    basis: Tensor
    half_widths: tuple[float, float, float]
    param_max: float
    ii_flat: Tensor
    jj_flat: Tensor
    kk_flat: Tensor
    expand_mat: Tensor
    phases: list[_MEPhasePlan]


@dataclass
class _MEScaledPlan:
    pad: tuple[int, int, int]  # pad_x, pad_y, pad_z
    do_pad: bool
    orig: tuple[int, int, int]  # nz_orig, ny_orig, nx_orig
    padded: tuple[int, int, int]  # nz, ny, nx
    pe_grid_axis: int
    do_xyz: tuple[bool, bool, bool]
    n_active: int
    levels: list[_MELevelPlan]
    use_ncc: bool
    slicewise_axis: int | None = None


def _build_mescaled_plan(
    base_echoes: Tensor,
    pe_grid_axis: int,
    weight: Tensor | None,
    mask: Tensor | None,
    config: QwarpConfig,
    device: torch.device,
    n_levels: int,
    pad: bool,
    slicewise_axis: int | None = None,
) -> _MEScaledPlan:
    """Precompute all frame-invariant structure for the ME-scaled polish.

    Padded base references, weight/mask, blok lattice, patch lattice, per-level
    basis, and the vectorised per-phase gathers of the base/weight/mask patches.
    A series calls this ONCE and reuses it for every frame; the per-frame solve
    (:func:`_solve_mescaled_frame`) then only touches the moving data.

    ``slicewise_axis`` (grid axis 0=x/1=y/2=z) makes the patches 2-D: one voxel
    thick along that axis, with the basis functions that modulate along it dropped.
    That is the right geometry for 2-D multi-slice acquisitions, where each slice is
    sampled at its own instant and the field is genuinely discontinuous through
    plane -- a cubic patch spanning ``minpatch`` slices would smooth across
    acquisition times. Must differ from ``pe_grid_axis`` (PE has to lie in-plane).
    """
    if slicewise_axis is not None:
        if slicewise_axis not in (0, 1, 2):
            raise ValueError(f"slicewise_axis must be 0, 1, 2 or None; got {slicewise_axis}.")
        if slicewise_axis == pe_grid_axis:
            raise ValueError(
                f"slicewise_axis ({slicewise_axis}) must differ from pe_grid_axis "
                f"({pe_grid_axis}): PE must lie inside the patch plane to be solvable."
            )
    base_echoes = base_echoes.float().to(device)
    E = base_echoes.shape[0]
    nz_orig, ny_orig, nx_orig = base_echoes.shape[1:]

    pad_x, pad_y, pad_z = _compute_padding(nx_orig, ny_orig, nz_orig) if pad else (0, 0, 0)
    do_pad = pad_x > 0 or pad_y > 0 or pad_z > 0

    base_p = (
        torch.stack([_pad_volume(base_echoes[e], pad_x, pad_y, pad_z) for e in range(E)])
        if do_pad
        else base_echoes
    )
    nz, ny, nx = base_p.shape[1:]

    if weight is None:
        weight_p = compute_weight_image(base_p.mean(0))
    else:
        w = weight.float().to(device)
        weight_p = _pad_volume(w, pad_x, pad_y, pad_z) if do_pad else w
    if mask is None:
        mask_p = (weight_p > 0).byte()
    else:
        m = mask.byte().to(device)
        mask_p = _pad_volume(m.float(), pad_x, pad_y, pad_z).byte() if do_pad else m

    do_xyz = (pe_grid_axis == 0, pe_grid_axis == 1, pe_grid_axis == 2)
    use_ncc = config.cost_method == "ncc"
    if use_ncc:
        blok_index_vol = None
        nblok = 0
    else:
        bs = assign_bloks((nz, ny, nx), (1.0, 1.0, 1.0), "tohd", config.blok_rad, mask=(mask_p > 0))
        blok_index_vol = bs.index.reshape(nz, ny, nx)
        nblok = bs.nblok

    imin, imax, jmin, jmax, kmin, kmax = _autobox(weight_p)

    ngmin = max(config.minpatch, 5)
    if ngmin % 2 == 0:
        ngmin -= 1
    widths: list[int] = []
    w = ngmin
    for _ in range(max(1, n_levels)):
        widths.append(w)
        w = int(round(w / config.shrink))
        if w % 2 == 0:
            w += 1
    widths = widths[::-1]  # coarsest first
    max_patch_lev1 = widths[0]

    basis_type = "cubic_lite" if config.use_lite else "cubic"
    ny_nx = ny * nx
    base_p_flat = base_p.reshape(E, -1)
    weight_flat = weight_p.reshape(-1)
    mask_flat = mask_p.float().reshape(-1)
    blok_flat = blok_index_vol.reshape(-1).long() if blok_index_vol is not None else None

    active_dims = [d for d in range(3) if do_xyz[d]]
    levels: list[_MELevelPlan] = []
    n_active = 0

    # Only the axes the patch actually spans constrain its size: in slicewise mode a
    # 2-slice run is legal, since the through-plane extent is 1 by construction.
    axis_limit = (nx - 2, ny - 2, nz - 2)
    spanned = [d for d in range(3) if d != slicewise_axis]

    for pw in widths:
        pw = min(pw, *(axis_limit[d] for d in spanned))
        if pw % 2 == 0:
            pw -= 1
        if pw < 5:
            continue
        wid = [pw, pw, pw]
        if slicewise_axis is not None:
            wid[slicewise_axis] = 1
        xwid, ywid, zwid = wid
        # A width-1 axis has del 1, so the lattice steps one slice at a time.
        xdel, ydel, zdel = (max(1, (w - 1) // 2) for w in wid)

        ibbb = max(1, imin - xdel // 4 - 1)
        jbbb = max(1, jmin - ydel // 4 - 1)
        kbbb = max(1, kmin - zdel // 4 - 1)
        ittt = min(nx - 2, imax + xdel // 4 + 1)
        jttt = min(ny - 2, jmax + ydel // 4 + 1)
        kttt = min(nz - 2, kmax + zdel // 4 + 1)
        if nz == 1:
            kbbb = kttt = 0

        hfactor = _compute_hfactor(pw, max_patch_lev1, config.hfactor_q)
        basis, half_widths, param_max = _get_basis_config(
            basis_type, xwid, ywid, zwid, device, hfactor=hfactor
        )
        if slicewise_axis is not None:
            # Hermite b1 is zero at the patch centre, so on a width-1 axis every basis
            # function that modulates along it collapses to an identically-zero row --
            # a null column in the Jacobian. Drop them; what remains IS the 2-D
            # in-plane tensor basis (cubic_lite 4->3, cubic 8->4).
            basis = basis[basis.abs().amax(dim=1) > 0].contiguous()
        n_basis = basis.shape[0]
        n_active = len(active_dims) * n_basis
        n_total = 3 * n_basis
        expand_mat = torch.zeros(n_active, n_total, device=device)
        col = 0
        for dim_i in active_dims:
            off = dim_i * n_basis
            expand_mat[col : col + n_basis, off : off + n_basis] = torch.eye(n_basis, device=device)
            col += n_basis

        kk_p, jj_p, ii_p = torch.meshgrid(
            torch.arange(zwid, dtype=torch.float32, device=device),
            torch.arange(ywid, dtype=torch.float32, device=device),
            torch.arange(xwid, dtype=torch.float32, device=device),
            indexing="ij",
        )
        ii_flat, jj_flat, kk_flat = ii_p.reshape(-1), jj_p.reshape(-1), kk_p.reshape(-1)
        # Flat local offset within a patch (row-major (z,y,x)), shared by all patches.
        local_off = (kk_flat.long() * ny_nx + jj_flat.long() * nx + ii_flat.long())[None, :]

        all_patches = _generate_patch_grid(
            ibbb, ittt, jbbb, jttt, kbbb, kttt, xwid, ywid, zwid, xdel, ydel, zdel
        )
        valid = _filter_patches(all_patches, weight_p, mask_p, nx, ny, nz)
        checker = _checkerboard_phases(valid, slicewise_axis)

        phase_plans: list[_MEPhasePlan] = []
        for phase_idx in range(8):
            pp = [
                p
                for p in checker[phase_idx]
                if (p.itop - p.ibot + 1 == xwid)
                and (p.jtop - p.jbot + 1 == ywid)
                and (p.ktop - p.kbot + 1 == zwid)
            ]
            if not pp:
                continue
            B = len(pp)
            ib = torch.tensor([p.ibot for p in pp], device=device)
            jb = torch.tensor([p.jbot for p in pp], device=device)
            kb = torch.tensor([p.kbot for p in pp], device=device)
            patch_base = (kb * ny_nx + jb * nx + ib)[:, None]  # (B,1)
            gather_idx = patch_base + local_off  # (B, V) long
            write_dst, write_src = _dedup_last_wins(gather_idx.reshape(-1))
            ibf, jbf, kbf = ib.float(), jb.float(), kb.float()
            phase_plans.append(
                _MEPhasePlan(
                    gather_idx=gather_idx,
                    write_dst=write_dst,
                    write_src=write_src,
                    ibots=ibf,
                    jbots=jbf,
                    kbots=kbf,
                    base_i=ibf[:, None] + ii_flat[None, :],
                    base_j=jbf[:, None] + jj_flat[None, :],
                    base_k=kbf[:, None] + kk_flat[None, :],
                    weight_patches=weight_flat[gather_idx],
                    mask_patches=mask_flat[gather_idx],
                    base_echo_patches=base_p_flat[:, gather_idx],
                    blok_prep=(
                        None if use_ncc else prepare_blok_pairs(blok_flat[gather_idx], nblok)
                    ),
                    B=B,
                )
            )

        levels.append(
            _MELevelPlan(
                nxh=xwid,
                nyh=ywid,
                nzh=zwid,
                basis=basis,
                half_widths=half_widths,
                param_max=param_max,
                ii_flat=ii_flat,
                jj_flat=jj_flat,
                kk_flat=kk_flat,
                expand_mat=expand_mat,
                phases=phase_plans,
            )
        )

    return _MEScaledPlan(
        pad=(pad_x, pad_y, pad_z),
        do_pad=do_pad,
        orig=(nz_orig, ny_orig, nx_orig),
        padded=(nz, ny, nx),
        pe_grid_axis=pe_grid_axis,
        do_xyz=do_xyz,
        n_active=n_active,
        levels=levels,
        use_ncc=use_ncc,
        slicewise_axis=slicewise_axis,
    )


def _improve_warp_batched_mescaled(
    phase: _MEPhasePlan,
    level: _MELevelPlan,
    source_echoes: Tensor,
    alpha: Tensor,
    state: WarpState,
    config: QwarpConfig,
    device: torch.device,
    do_xyz: tuple[bool, bool, bool],
    use_penalty: bool,
    pen_fac: float,
    max_iter: int,
    use_ncc: bool,
) -> None:
    """Optimize one checkerboard phase against the JOINT multi-echo scaled cost.

    The free parameters are ONE shared PE-only field; the cost samples every echo
    at ``alpha_e * (shared displacement)`` and sums the correlation. Gathers are
    precomputed in ``phase`` (frame-invariant); only the moving ``source_echoes``
    and the current ``state`` warp change per frame."""
    B = phase.B
    if B == 0 or level.expand_mat.shape[0] == 0:
        return
    E = source_echoes.shape[0]
    nx, ny, nz = state.nx, state.ny, state.nz
    nxh, nyh, nzh = level.nxh, level.nyh, level.nzh
    basis, half_widths = level.basis, level.half_widths
    ii_flat, jj_flat, kk_flat = level.ii_flat, level.jj_flat, level.kk_flat
    expand_mat = level.expand_mat
    ibots, jbots, kbots = phase.ibots, phase.jbots, phase.kbots
    base_i, base_j, base_k = phase.base_i, phase.base_j, phase.base_k
    weight_patches, mask_patches = phase.weight_patches, phase.mask_patches
    base_echo_patches = phase.base_echo_patches

    _blok_value = lpc_value_pairs if config.cost_method == "lpc" else lpa_value_pairs

    external_pen = torch.zeros(B, device=device)
    if use_penalty:
        with torch.no_grad():
            je_g, se_g = compute_jacobian_energy(state.xd, state.yd, state.zd)
            energy_g = penalty_energy(je_g, se_g).reshape(-1)
            external_pen = energy_g.sum() - energy_g[phase.gather_idx].sum(-1)

    global_warp_3ch = torch.stack([state.xd, state.yd, state.zd], dim=0)
    a_e = alpha.view(E, 1, 1)  # (E,1,1) per-echo TE scaling for broadcast

    # Optional torch.compile of the stable building blocks. The plan geometry is
    # frame-invariant, so shapes are identical across every frame of the series and
    # the (process-global, name-keyed) cache pays warmup exactly once -- the whole
    # point of compiling the timeseries path. See principles/torch.compile.md.
    _eval_warp = _maybe_compile(evaluate_patch_warp_batched, "eval_warp", device, config.compile)
    _compose_fn = _maybe_compile(batched_compose_and_interpolate, "compose", device, config.compile)
    _gn_normal_eqs = _maybe_compile(
        _mescaled_gn_normal_eqs, "mescaled_gn_normal_eqs", device, config.compile
    )

    def _sample_echoes(ah_xd: Tensor, ah_yd: Tensor, ah_zd: Tensor) -> Tensor:
        """(B,) mean -corr over echoes for the shared displacement ``ah``."""
        if use_ncc:
            # All echoes sampled at alpha_e*ah in ONE grid_sample (E as the batch dim).
            sx = (base_i[None] + a_e * ah_xd[None]).clamp(-0.499, nx - 0.501)
            sy = (base_j[None] + a_e * ah_yd[None]).clamp(-0.499, ny - 0.501)
            sz = (base_k[None] + a_e * ah_zd[None]).clamp(-0.499, nz - 0.501)
            warped = batched_trilinear_interpolate_multi(source_echoes, sx, sy, sz)
            warped = warped * mask_patches[None]
            r = _weighted_pearson_patches(base_echo_patches, warped, weight_patches[None])
            return (-r).mean(0)
        cost = torch.zeros(B, device=device)
        for e in range(E):
            ae = alpha[e]
            sx = (base_i + ae * ah_xd).clamp(-0.499, nx - 0.501)
            sy = (base_j + ae * ah_yd).clamp(-0.499, ny - 0.501)
            sz = (base_k + ae * ah_zd).clamp(-0.499, nz - 0.501)
            w_e = batched_trilinear_interpolate(source_echoes[e], sx, sy, sz) * mask_patches
            corr_e = _blok_value(
                base_echo_patches[e], w_e, weight_patches, phase.blok_prep, config.lpc_ppow
            )
            cost = cost - corr_e
        return cost / E

    def _compose(hxd: Tensor, hyd: Tensor, hzd: Tensor):
        return _compose_fn(
            source_echoes[0],
            state.xd,
            state.yd,
            state.zd,
            hxd,
            hyd,
            hzd,
            ii_flat,
            jj_flat,
            kk_flat,
            ibots,
            jbots,
            kbots,
            nx,
            ny,
            nz,
            global_warp_3ch=global_warp_3ch,
            base_i=base_i,
            base_j=base_j,
            base_k=base_k,
        )

    def batched_cost(active_params: Tensor) -> Tensor:
        full_params = active_params @ expand_mat
        hxd, hyd, hzd = _eval_warp(basis, full_params, half_widths, do_xyz)
        # ah = shared TOTAL displacement w (patch increment composed with global w).
        _wv, ah_xd, ah_yd, ah_zd = _compose(hxd, hyd, hzd)
        cost = _sample_echoes(ah_xd, ah_yd, ah_zd)
        if use_penalty and pen_fac > 0:
            cost = cost + compute_penalty_batched(
                ah_xd.reshape(B, nzh, nyh, nxh),
                ah_yd.reshape(B, nzh, nyh, nxh),
                ah_zd.reshape(B, nzh, nyh, nxh),
                pen_fac,
                external_pen,
            )
        return cost

    def _gn_solve(gn_iters: int) -> Tensor:
        """Batched Levenberg-Marquardt on the normalised-NCC cost (PE-only).

        The warp is linear in the params (``d = hw*(p@B)``), so the residual
        Jacobian is ``(alpha_e/sW) * gradS · (hw·Bᵀ)`` -- an analytic image-gradient
        term, no autograd. Per patch we assemble the tiny ``n_basis×n_basis`` normal
        equations (summed over echoes, weighted, mean-centred = zero-normalised LK),
        take a damped step, and accept it only if the exact NCC cost improves
        (per-patch damping ``lam`` up on reject, down on accept)."""
        pe = active_dims[0]
        hw_pe = float(half_widths[pe])
        bmat = basis  # (nb, V)
        bt = bmat.t()  # (V, nb)
        nb = bmat.shape[0]
        pe_dim = (2, 1, 0)[pe]  # PE grid axis -> tensor dim in (nz, ny, nx)
        gpe = (
            torch.roll(source_echoes, -1, pe_dim + 1) - torch.roll(source_echoes, 1, pe_dim + 1)
        ) * 0.5  # ∂S/∂(PE voxel), one per echo
        omega = (weight_patches * mask_patches)[None]  # (1, B, V)
        wsum = omega.sum(-1, keepdim=True).clamp_min(1e-6)  # (1, B, 1)
        a4hw = alpha.view(E, 1, 1, 1) * hw_pe  # per-echo alpha_e * hw_pe (folds scalars)

        base = base_echo_patches  # (E, B, V)
        mB = (omega * base).sum(-1, keepdim=True) / wsum
        sB = ((omega * base * base).sum(-1, keepdim=True) / wsum - mB * mB).clamp_min(1e-12).sqrt()
        base_hat = (base - mB) / sB

        def _forward(p: Tensor):
            d = hw_pe * (p @ bmat)  # (B, V)
            zero = torch.zeros_like(d)
            hxd, hyd, hzd = (
                (d, zero, zero) if pe == 0 else (zero, d, zero) if pe == 1 else (zero, zero, d)
            )
            _wv, ax, ay, az = _compose(hxd, hyd, hzd)
            sx = (base_i[None] + a_e * ax[None]).clamp(-0.499, nx - 0.501)
            sy = (base_j[None] + a_e * ay[None]).clamp(-0.499, ny - 0.501)
            sz = (base_k[None] + a_e * az[None]).clamp(-0.499, nz - 0.501)
            w = batched_trilinear_interpolate_multi(source_echoes, sx, sy, sz)  # (E,B,V)
            return w, sx, sy, sz

        def _cost(w: Tensor) -> Tensor:
            mW = (omega * w).sum(-1, keepdim=True) / wsum
            sW = ((omega * w * w).sum(-1, keepdim=True) / wsum - mW * mW).clamp_min(1e-12).sqrt()
            res = (w - mW) / sW - base_hat
            return ((omega * res * res).sum(-1) / wsum.squeeze(-1)).sum(0)  # (E,B)->(B,)

        p = torch.zeros(B, nb, device=device)
        w0, *_ = _forward(p)
        best_cost = _cost(w0)
        best_p = p.clone()
        lam = torch.full((B,), 1e-2, device=device)
        for _it in range(gn_iters):
            w, sx, sy, sz = _forward(best_p)
            g = batched_trilinear_interpolate_multi(gpe, sx, sy, sz)  # (E,B,V)
            hmat, grad = _gn_normal_eqs(w, g, a4hw, bt, omega, wsum, base_hat)
            diag = torch.diagonal(hmat, dim1=-2, dim2=-1).clamp_min(1e-9)
            amat = hmat + lam[:, None, None] * torch.diag_embed(diag)
            delta = torch.linalg.solve(amat, -grad.unsqueeze(-1)).squeeze(-1)
            p_new = (best_p + delta).clamp(-param_max, param_max)
            wn, *_ = _forward(p_new)
            cost_new = _cost(wn)
            improved = cost_new < best_cost
            best_p = torch.where(improved[:, None], p_new, best_p)
            best_cost = torch.where(improved, cost_new, best_cost)
            lam = torch.where(improved, lam * 0.5, lam * 4.0).clamp(1e-8, 1e8)
        return best_p

    active_dims = [d for d in range(3) if do_xyz[d]]
    param_max = level.param_max
    if use_ncc and config.optimizer == "gn":
        with torch.no_grad():
            best_params = _gn_solve(min(max_iter, 6))
    else:
        best_params, _bc, _stats = optimize_warp_params_batched(
            batched_cost,
            B,
            level.expand_mat.shape[0],
            level.param_max,
            device,
            max_iter=max_iter,
            lr=config.batch_optimizer_lr,
            tolerance=config.batch_optimizer_tol,
            patience=config.batch_optimizer_patience,
        )

    with torch.no_grad():
        full_params = best_params @ expand_mat
        hxd, hyd, hzd = _eval_warp(basis, full_params, half_widths, do_xyz)
        _wv, ah_xd, ah_yd, ah_zd = _compose(hxd, hyd, hzd)
        # Patches in a phase overlap by a voxel at their seams, so write through the
        # deduplicated index (last patch wins) -- a raw scatter would be nondeterministic.
        dst, src = phase.write_dst, phase.write_src
        state.xd.view(-1)[dst] = ah_xd.reshape(-1)[src]
        state.yd.view(-1)[dst] = ah_yd.reshape(-1)[src]
        state.zd.view(-1)[dst] = ah_zd.reshape(-1)[src]
    state.patches_done += B


def _solve_mescaled_frame(
    plan: _MEScaledPlan,
    source_echoes: Tensor,
    seed_field: Tensor | None,
    alpha: Tensor,
    config: QwarpConfig,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Solve one frame's ME-scaled PE-only polish against a prebuilt ``plan``."""
    pad_x, pad_y, pad_z = plan.pad
    nz, ny, nx = plan.padded
    nz_orig, ny_orig, nx_orig = plan.orig
    E = source_echoes.shape[0]

    source_echoes = source_echoes.float().to(device)
    source_p = (
        torch.stack([_pad_volume(source_echoes[e], pad_x, pad_y, pad_z) for e in range(E)])
        if plan.do_pad
        else source_echoes
    )

    state = WarpState(nx=nx, ny=ny, nz=nz)
    state.xd = torch.zeros(nz, ny, nx, device=device)
    state.yd = torch.zeros(nz, ny, nx, device=device)
    state.zd = torch.zeros(nz, ny, nx, device=device)
    if seed_field is not None:
        seed = seed_field.float().to(device)
        seed_p = _pad_volume(seed, pad_x, pad_y, pad_z) if plan.do_pad else seed
        (state.xd, state.yd, state.zd)[plan.pe_grid_axis][...] = seed_p

    for level in plan.levels:
        # The penalty's h**0.25 term has infinite gradient at an all-zero field, which
        # would poison the batch gradient; enable it only once the field has content
        # (deferred one level only in the unseeded case -- a real seed is nonzero).
        use_pen = float((state.xd.pow(2) + state.yd.pow(2) + state.zd.pow(2)).sum()) > 0.0
        for phase in level.phases:
            _improve_warp_batched_mescaled(
                phase,
                level,
                source_p,
                alpha,
                state,
                config,
                device,
                plan.do_xyz,
                use_penalty=use_pen,
                pen_fac=config.penalty_factor,
                max_iter=config.batch_optimizer_iters,
                use_ncc=plan.use_ncc,
            )

    warped = torch.empty(E, nz_orig, ny_orig, nx_orig, device=device)
    for e in range(E):
        ae = float(alpha[e])
        wf = warp_image(
            source_p[e], ae * state.xd, ae * state.yd, ae * state.zd, mode=config.final_interp
        )
        warped[e] = wf[pad_z : pad_z + nz_orig, pad_y : pad_y + ny_orig, pad_x : pad_x + nx_orig]

    field_pad = (state.xd, state.yd, state.zd)[plan.pe_grid_axis]
    field = field_pad[pad_z : pad_z + nz_orig, pad_y : pad_y + ny_orig, pad_x : pad_x + nx_orig]
    return warped, field.contiguous()


def qwarp_pe_scaled_polish(
    base_echoes: Tensor,
    source_echoes: Tensor,
    pe_grid_axis: int,
    alpha: Tensor | None = None,
    seed_field: Tensor | None = None,
    weight: Tensor | None = None,
    mask: Tensor | None = None,
    config: QwarpConfig | None = None,
    device: torch.device | None = None,
    n_levels: int = 2,
    pad: bool = True,
    slicewise_axis: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Polish a seed PE displacement field with joint multi-echo TE-scaled qwarp.

    One shared PE-only field ``w`` is refined over ``n_levels`` fine patch levels
    (ending at ``config.minpatch``) under the joint objective
    ``sum_e lpa(warp(source_e, alpha_e*w), base_e)``. This is the qwarp "polish"
    hand-off for locomoco: seed ``w`` with locomoco's estimate and let a couple of
    fine nonlinear levels remove the residual it could not resolve. Single-echo is
    ``E=1, alpha=[1]``.

    Args:
        base_echoes: ``(E, nz, ny, nx)`` per-echo reference (one TR, all echoes).
        source_echoes: ``(E, nz, ny, nx)`` per-echo moving volume to align.
        pe_grid_axis: which displacement channel carries PE — 0=x (fastest, ``nx``),
            1=y (``ny``), 2=z (``nz``); the other two channels are held at zero.
        alpha: ``(E,)`` fixed per-echo TE scaling (echo-1 relative). ``None`` => ones
            (flat scaling / single echo).
        seed_field: ``(nz, ny, nx)`` seed PE displacement (voxels). ``None`` => zeros.
        weight, mask: optional shared ``(nz, ny, nx)`` weight/mask (auto if None).
        config: :class:`QwarpConfig`; ``minpatch`` sets the finest level and
            ``cost_method`` must be ``lpc``/``lpa``.
        n_levels: number of fine levels (coarsest is ~``minpatch/shrink**(n-1)``).
        pad: AFNI-style internal zero-padding (default True).
        slicewise_axis: grid axis (0=x, 1=y, 2=z) to make patches 2-D across -- one
            voxel thick with an in-plane-only basis, so nothing is smoothed through
            plane. Use it for 2-D multi-slice acquisitions (each slice has its own
            acquisition instant); ``None`` keeps the isotropic 3-D patches, right for
            3-D-acquired EPI. Must differ from ``pe_grid_axis``.

    Returns:
        ``(warped, field)`` where ``warped`` is ``(E, nz, ny, nx)`` each source warped
        by ``alpha_e * w``, and ``field`` is the ``(nz, ny, nx)`` polished PE
        displacement (voxels), both cropped back to the original grid.
    """
    if config is None:
        config = QwarpConfig()
    if config.cost_method not in ("ncc", "lpc", "lpa"):
        raise ValueError(
            f"qwarp_pe_scaled_polish uses ncc (patch Pearson) or lpc/lpa (blok-local); "
            f"got cost_method={config.cost_method!r}."
        )
    if pe_grid_axis not in (0, 1, 2):
        raise ValueError(f"pe_grid_axis must be 0, 1 or 2; got {pe_grid_axis}.")
    if source_echoes.dim() != 4 or base_echoes.dim() != 4:
        raise ValueError("base_echoes/source_echoes must be (E, nz, ny, nx).")

    if device is None:
        device = base_echoes.device if base_echoes.is_cuda else torch.device("cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    E = source_echoes.shape[0]
    alpha = torch.ones(E, device=device) if alpha is None else alpha.float().to(device)

    plan = _build_mescaled_plan(
        base_echoes, pe_grid_axis, weight, mask, config, device, n_levels, pad, slicewise_axis
    )
    return _solve_mescaled_frame(plan, source_echoes, seed_field, alpha, config, device)


def qwarp_pe_scaled_polish_series(
    base_echoes: Tensor,
    source_series: Tensor,
    seed_series: Tensor,
    pe_grid_axis: int,
    alpha: Tensor | None = None,
    weight: Tensor | None = None,
    mask: Tensor | None = None,
    config: QwarpConfig | None = None,
    device: torch.device | None = None,
    n_levels: int = 2,
    show_progress: bool = True,
    slicewise_axis: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Per-frame joint multi-echo TE-scaled PE-only polish over a 4-D series.

    Applies the joint scaled polish to every frame ``t`` independently: each frame's
    echoes are aligned to the shared per-echo reference under the joint scaled
    objective, seeded by ``seed_series[..., t]``. This is the real-data hand-off from
    locomoco -- one shared PE field per frame, refined against every echo at its own TE.

    The geometry (patch lattice, basis, per-phase base/weight/mask gathers) is fixed
    across frames, so it is built ONCE via :func:`_build_mescaled_plan` and reused;
    each frame only pays the moving-data padding + the optimizer. This is the bulk of
    the per-frame speed -- the reused setup was ~2/3 of a from-scratch call.

    Args:
        base_echoes: ``(E, nz, ny, nx)`` per-echo reference template (motion-free).
        source_series: ``(E, nz, ny, nx, T)`` per-echo moving series.
        seed_series: ``(nz, ny, nx, T)`` per-frame seed PE displacement (voxels,
            echo-1 scale) -- e.g. locomoco's shared field ``w``.
        pe_grid_axis: PE displacement channel (0=x, 1=y, 2=z); see
            :func:`qwarp_pe_scaled_polish`.
        alpha: ``(E,)`` fixed per-echo TE scaling (echo-1 relative). ``None`` => ones.
        weight, mask: optional shared ``(nz, ny, nx)`` weight/mask (auto if None).
        config: :class:`QwarpConfig`; ``cost_method`` must be ``ncc``/``lpc``/``lpa``.
        n_levels: fine levels per frame.
        show_progress: draw a persistent tqdm bar over frames.
        slicewise_axis: 2-D slicewise patches across this grid axis; see
            :func:`qwarp_pe_scaled_polish`.

    Returns:
        ``(warped, field)`` with ``warped`` ``(E, nz, ny, nx, T)`` and ``field``
        ``(nz, ny, nx, T)`` the polished per-frame PE displacement.
    """
    if source_series.dim() != 5:
        raise ValueError(
            f"source_series must be (E, nz, ny, nx, T); got {tuple(source_series.shape)}"
        )
    E, nz, ny, nx, T = source_series.shape
    if seed_series.shape != (nz, ny, nx, T):
        raise ValueError(
            f"seed_series must be (nz, ny, nx, T)={(nz, ny, nx, T)}; got {tuple(seed_series.shape)}"
        )
    if config is None:
        config = QwarpConfig()
    if device is None:
        device = base_echoes.device if base_echoes.is_cuda else torch.device("cpu")
    alpha = torch.ones(E, device=device) if alpha is None else alpha.float().to(device)

    # Build the frame-invariant geometry + base/weight/mask gathers ONCE.
    plan = _build_mescaled_plan(
        base_echoes, pe_grid_axis, weight, mask, config, device, n_levels, True, slicewise_axis
    )

    warped = torch.empty(E, nz, ny, nx, T)
    field = torch.empty(nz, ny, nx, T)

    frames = range(T)
    if show_progress and _tqdm is not None and T > 1:
        frames = _tqdm(frames, desc="qwarp polish", leave=True)

    # The opt-in fast path (config.compile) also enables TF32 tensor cores for the
    # solve's float32 matmuls -- scoped here, never leaking to other tools.
    use_tf32 = bool(config.compile) and device.type == "cuda"
    with _tf32_matmul(use_tf32):
        for t in frames:
            w_t, f_t = _solve_mescaled_frame(
                plan, source_series[..., t], seed_series[..., t], alpha, config, device
            )
            warped[..., t] = w_t.cpu()
            field[..., t] = f_t.cpu()

    return warped, field


# ---------------------------------------------------------------------------
# Serial patch optimization (for level 0 and edge cases)
# ---------------------------------------------------------------------------


def _improve_warp_serial(
    base: Tensor,
    source: Tensor,
    weight: Tensor,
    mask: Tensor,
    state: WarpState,
    config: QwarpConfig,
    device: torch.device,
    ibot: int,
    itop: int,
    jbot: int,
    jtop: int,
    kbot: int,
    ktop: int,
    basis_type: str = "cubic_lite",
    do_xyz: tuple[bool, bool, bool] = (True, True, True),
    axis_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    use_penalty: bool = False,
    pen_fac: float = 0.033333,
    base_clip: tuple[float, float] | None = None,
    source_clip: tuple[float, float] | None = None,
    blok_index_vol: Tensor | None = None,
    nblok: int = 0,
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

    w_patch = weight[kbot : ktop + 1, jbot : jtop + 1, ibot : itop + 1]
    m_patch = mask[kbot : ktop + 1, jbot : jtop + 1, ibot : itop + 1]
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

    b_local = base[kbot : ktop + 1, jbot : jtop + 1, ibot : itop + 1].reshape(-1)
    w_local = weight[kbot : ktop + 1, jbot : jtop + 1, ibot : itop + 1].reshape(-1)

    if b_local.max() == b_local.min():
        state.patches_skipped += 1
        return False

    # Blok local-Pearson (lpc/lpa) is scored on this single patch (B=1) so odd
    # boundary patches match the batched path; other methods use INCOR.
    use_blok_serial = config.cost_method in ("lpc", "lpa")
    incor = None
    blok_prep_s = None
    if use_blok_serial:
        blok_idx_s = blok_index_vol[kbot : ktop + 1, jbot : jtop + 1, ibot : itop + 1].reshape(-1)[
            None, :
        ]
        blok_prep_s = prepare_blok_pairs(blok_idx_s, nblok)
        _blok_value_s = lpc_value_pairs if config.cost_method == "lpc" else lpa_value_pairs
    else:
        incor = IncrementalCorrelation(method=config.cost_method)
        if base_clip is not None and source_clip is not None:
            incor.set_clips(base_clip, source_clip)
        weight_for_fixed = weight.clone()
        weight_for_fixed[kbot : ktop + 1, jbot : jtop + 1, ibot : itop + 1] = 0.0
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
        je_ext[kbot : ktop + 1, jbot : jtop + 1, ibot : itop + 1] = 0.0
        se_ext[kbot : ktop + 1, jbot : jtop + 1, ibot : itop + 1] = 0.0
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
            full_params[offset : offset + n_basis_per_dim] = (
                params[idx : idx + n_basis_per_dim] * scale
            )
            idx += n_basis_per_dim

        hxd, hyd, hzd = evaluate_patch_warp(basis, full_params, half_widths, do_xyz)
        hxd_3d = hxd.reshape(nzh, nyh, nxh)
        hyd_3d = hyd.reshape(nzh, nyh, nxh)
        hzd_3d = hzd.reshape(nzh, nyh, nxh)

        xq = (ibot + ii_p + hxd_3d).clamp(0, nx - 1)
        yq = (jbot + jj_p + hyd_3d).clamp(0, ny - 1)
        zq = (kbot + kk_p + hzd_3d).clamp(0, nz - 1)

        axd_interp = trilinear_interpolate(
            global_xd, xq.reshape(-1), yq.reshape(-1), zq.reshape(-1)
        ).reshape(nzh, nyh, nxh)
        ayd_interp = trilinear_interpolate(
            global_yd, xq.reshape(-1), yq.reshape(-1), zq.reshape(-1)
        ).reshape(nzh, nyh, nxh)
        azd_interp = trilinear_interpolate(
            global_zd, xq.reshape(-1), yq.reshape(-1), zq.reshape(-1)
        ).reshape(nzh, nyh, nxh)

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

        if use_blok_serial:
            corr = float(
                _blok_value_s(
                    b_local[None], warped_vals[None], w_local[None], blok_prep_s, config.lpc_ppow
                )[0]
            )
        else:
            # incor is built exactly when `not use_blok_serial`, this branch's
            # condition; it's a captured closure variable so ty can't narrow
            # it across the outer `if use_blok_serial` the way it would inline.
            assert incor is not None
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
        full_params[offset : offset + n_basis_per_dim] = (
            best_params[idx : idx + n_basis_per_dim] * scale
        )
        idx += n_basis_per_dim

    hxd, hyd, hzd = evaluate_patch_warp(basis, full_params, half_widths, do_xyz)
    hxd_3d = hxd.reshape(nzh, nyh, nxh)
    hyd_3d = hyd.reshape(nzh, nyh, nxh)
    hzd_3d = hzd.reshape(nzh, nyh, nxh)

    xq = (ibot + ii_p + hxd_3d).clamp(0, nx - 1)
    yq = (jbot + jj_p + hyd_3d).clamp(0, ny - 1)
    zq = (kbot + kk_p + hzd_3d).clamp(0, nz - 1)

    axd_interp = trilinear_interpolate(
        global_xd, xq.reshape(-1), yq.reshape(-1), zq.reshape(-1)
    ).reshape(nzh, nyh, nxh)
    ayd_interp = trilinear_interpolate(
        global_yd, xq.reshape(-1), yq.reshape(-1), zq.reshape(-1)
    ).reshape(nzh, nyh, nxh)
    azd_interp = trilinear_interpolate(
        global_zd, xq.reshape(-1), yq.reshape(-1), zq.reshape(-1)
    ).reshape(nzh, nyh, nxh)

    ah_xd = hxd_3d + axd_interp
    ah_yd = hyd_3d + ayd_interp
    ah_zd = hzd_3d + azd_interp

    state.xd[kbot : ktop + 1, jbot : jtop + 1, ibot : itop + 1] = ah_xd
    state.yd[kbot : ktop + 1, jbot : jtop + 1, ibot : itop + 1] = ah_yd
    state.zd[kbot : ktop + 1, jbot : jtop + 1, ibot : itop + 1] = ah_zd

    src_x = (ah_xd + ii_p + ibot).clamp(-0.499, nx - 0.501)
    src_y = (ah_yd + jj_p + jbot).clamp(-0.499, ny - 0.501)
    src_z = (ah_zd + kk_p + kbot).clamp(-0.499, nz - 0.501)

    warped_patch = trilinear_interpolate(
        source, src_x.reshape(-1), src_y.reshape(-1), src_z.reshape(-1)
    ).reshape(nzh, nyh, nxh)

    state.warped_source[kbot : ktop + 1, jbot : jtop + 1, ibot : itop + 1] = warped_patch
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
