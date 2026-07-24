"""Symmetric Normalization (SyN) nonlinear registration on the GPU.

A second nonlinear-registration backend alongside :mod:`fastfuncstuff.processing.warp`
(the AFNI-3dQwarp engine). This one implements ANTs-style **SyN**: dense displacement
fields, fluid + elastic Gaussian regularization, and the defining *symmetric midpoint*
formulation. Where 3dQwarp optimizes shrinking overlapping patches, SyN drives two
half-transforms that meet the fixed and moving images at a common midpoint, which makes
the registration inverse-consistent (registering F->M yields the inverse of M->F) and
yields a free inverse warp plus the two halfway warps.

Algorithm (greedy SyN, as shipped by ANTs ``antsRegistration -t SyN[...]``):

  Maintain two displacement fields on the *middle* grid, each with a maintained inverse:
    phi_f : middle -> fixed   (inverse inv_f : fixed -> middle)
    phi_m : middle -> moving  (inverse inv_m : moving -> middle)

  Per iteration, at the current pyramid level:
    1. Pull fixed/moving to the middle:  Fmid = F(phi_f),  Mmid = M(phi_m).
    2. Evaluate the image metric between Fmid and Mmid and take its gradient w.r.t.
       *both* half-fields (one backward pass over a symmetric cost gives the two
       symmetric updates automatically).
    3. Fluid regularization: Gaussian-smooth the update fields (``update_var``).
    4. Step: phi_f += s*u_f, phi_m += s*u_m, normalized so the max step is ``grad_step``.
    5. Elastic regularization: Gaussian-smooth the total fields (``total_var``, default 0).
    6. Invertibility: recompute each inverse and re-derive the forward from it (the
       symmetrizing step that keeps both fields diffeomorphic and inverse-consistent).
    7. Axis constraints: zero the X/Y/Z component of every field for -noX/Y/Zdis.

  Final moving->fixed warp (to resample moving into fixed space) = inv_f then phi_m.
  Inverse (fixed->moving)                                         = inv_m then phi_f.

Displacement convention matches the rest of the project (see ``processing/interp.py``):
output voxel ``(i,j,k)`` samples its source at ``(i+xd, j+yd, k+zd)`` in **voxel units**.

The image metric is evaluated through PyTorch autograd, so the project's existing cost
primitives (``cost.lpa_correlation`` / ``cost.lpc_correlation`` / ``cost.pearson_correlation``)
plug in as SyN metrics with no new gradient math; CC and MSE are expressed in the same
differentiable form. See ``../fmri_wiki/concepts/SyN.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover - tqdm is a hard dep in practice
    _tqdm = None

from .cost import (
    _separable_smooth_3d,
    lpa_correlation,
    lpc_correlation,
    pearson_correlation,
)
from .interp import _grid_sample_3d, trilinear_interpolate, warp_image
from .nwarpforge import NonlinearWarp, compose_warp_then_warp

_EPS = 1e-6

# A displacement field as a (xd, yd, zd) triple, each (nz, ny, nx) in voxel units.
Field = tuple[Tensor, Tensor, Tensor]

# Bit flags shared with the qwarp engine's warp_flags (no-displacement per axis).
NO_X_DISP = 1
NO_Y_DISP = 2
NO_Z_DISP = 4

METRICS = ("cc", "lpa", "lpc", "pearson", "mse")


@dataclass
class SynConfig:
    """Configuration for the SyN engine."""

    metric: str = "cc"
    """Image metric: 'cc' (neighborhood cross-correlation, the ANTs SyN default),
    'lpa'/'lpc' (the project's local Pearson costs, for similar / cross contrast),
    'pearson' (global), or 'mse' (squared difference, same-modality sanity)."""

    cc_radius: int = 4
    """Neighborhood half-width (voxels) for the CC metric; window is (2r+1)^3."""

    lpa_sigma: float = 4.0
    """Neighborhood size (voxels) for the lpa/lpc metrics (Gaussian sigma or box radius)."""

    lpa_kernel: str = "gauss"
    """Neighborhood kernel for lpa/lpc: 'gauss' or 'box'."""

    grad_step: float = 0.25
    """Max per-iteration displacement (voxels) after update-field normalization.
    The ANTs SyN[gradientStep]; smaller is safer, larger converges faster."""

    update_var: float = 3.0
    """Fluid regularization: Gaussian sigma (voxels) smoothing the per-iteration
    update field. ANTs SyN[,updateFieldVarianceInVoxelSpace=3]."""

    total_var: float = 0.0
    """Elastic regularization: Gaussian sigma (voxels) smoothing the accumulated
    field. ANTs SyN[,,totalFieldVarianceInVoxelSpace=0] (0 = off)."""

    shrink_factors: tuple[int, ...] = (4, 2, 1)
    """Per-level isotropic downsample factor (ANTs -f). Coarse-to-fine."""

    smoothing_sigmas: tuple[float, ...] = (2.0, 1.0, 0.0)
    """Per-level Gaussian pre-smoothing sigma in voxels (ANTs -s ...vox)."""

    iterations: tuple[int, ...] = (100, 70, 40)
    """Per-level SyN iteration count (ANTs -c)."""

    invert_iters: int = 8
    """Fixed-point iterations for displacement-field inversion."""

    convergence_window: int = 10
    """Iterations in the trailing window used to detect convergence (ANTs -c
    convergenceWindowSize). The window's cost trend (slope) decides when a level
    has stopped improving. Set <=0 to disable early stopping (run all iterations)."""

    convergence_threshold: float = 1e-6
    """Convergence slope threshold (ANTs -c convergenceThreshold). A level stops
    once the (range-normalized) downward slope of cost over the window falls below
    this — i.e. the cost has flattened. Larger = stop sooner."""

    warp_flags: int = 0
    """Bit flags: 1=no-x-disp, 2=no-y-disp, 4=no-z-disp (matches qwarp)."""

    final_interp: str = "wsinc5"
    """Interpolation kernel for the final warped output (reuses warp_image's kernels,
    like 3dQwarp's final pass). Estimation iterations always use linear interpolation."""

    verb: int = 1
    """Verbosity (0=quiet, 1=normal, 2=detailed)."""


# ---------------------------------------------------------------------------
# Small primitives
# ---------------------------------------------------------------------------


def _warp_diff(source: Tensor, xd: Tensor, yd: Tensor, zd: Tensor) -> Tensor:
    """Differentiable linear warp: output[p] = source[p + d(p)].

    A grad-friendly companion to :func:`interp.warp_image_linear` (which masks
    out-of-bounds samples in place and is meant for the non-autograd path). Here we
    rely on grid_sample's border padding so the whole thing stays in the autograd
    graph w.r.t. the displacement fields.
    """
    nz, ny, nx = source.shape
    device = source.device
    dtype = source.dtype
    kk, jj, ii = torch.meshgrid(
        torch.arange(nz, dtype=dtype, device=device),
        torch.arange(ny, dtype=dtype, device=device),
        torch.arange(nx, dtype=dtype, device=device),
        indexing="ij",
    )
    x = ii + xd
    y = jj + yd
    z = kk + zd
    gx = 2.0 * x / (nx - 1) - 1.0 if nx > 1 else x * 0.0
    gy = 2.0 * y / (ny - 1) - 1.0 if ny > 1 else y * 0.0
    gz = 2.0 * z / (nz - 1) - 1.0 if nz > 1 else z * 0.0
    grid = torch.stack([gx, gy, gz], dim=-1)[None]
    vol5d = source[None, None]
    return _grid_sample_3d(vol5d, grid)[0, 0]


def _local_cc_cost(a: Tensor, b: Tensor, radius: int, weight: Tensor) -> Tensor:
    """ANTs neighborhood cross-correlation as a differentiable scalar (lower=better).

    Local windowed CC over a (2r+1)^3 box, summed (the N-per-window factor cancels
    out of ``cov^2/(varA*varB)``), so we compute it from normalized box means. The
    autograd gradient of this is the exact local-CC update field that the fluid
    smoothing then regularizes.
    """
    r = float(radius)

    def box(v: Tensor) -> Tensor:
        return _separable_smooth_3d(v, r, kernel_type="box")

    am = box(a)
    bm = box(b)
    cov = box(a * b) - am * bm
    va = (box(a * a) - am * am).clamp(min=_EPS)
    vb = (box(b * b) - bm * bm).clamp(min=_EPS)
    cc = cov * cov / (va * vb)  # in [0, 1], higher = better
    return -(weight * cc).sum() / weight.sum().clamp(min=_EPS)


def image_metric(
    a: Tensor,
    b: Tensor,
    weight: Tensor,
    metric: str = "cc",
    cc_radius: int = 4,
    lpa_sigma: float = 4.0,
    lpa_kernel: str = "gauss",
) -> Tensor:
    """Scalar image metric (lower = better) between two same-grid images.

    All metrics route through autograd-friendly ops, so the project's existing cost
    primitives serve directly as registration metrics (SyN update fields here, and
    convergence monitoring for the optical-flow engine).
    """
    if metric == "cc":
        return _local_cc_cost(a, b, cc_radius, weight)
    if metric == "mse":
        return (weight * (a - b) ** 2).sum() / weight.sum().clamp(min=_EPS)
    if metric == "lpa":
        return -lpa_correlation(a, b, weight, sigma=lpa_sigma, kernel_type=lpa_kernel)
    if metric == "lpc":
        return -lpc_correlation(a, b, weight, sigma=lpa_sigma, kernel_type=lpa_kernel)
    if metric == "pearson":
        return -pearson_correlation(a.reshape(-1), b.reshape(-1), weight.reshape(-1))
    raise ValueError(f"unknown metric {metric!r}; choose from {METRICS}")


def _metric_cost(a: Tensor, b: Tensor, weight: Tensor, config: SynConfig) -> Tensor:
    """SyN-config wrapper around :func:`image_metric`."""
    return image_metric(
        a,
        b,
        weight,
        metric=config.metric,
        cc_radius=config.cc_radius,
        lpa_sigma=config.lpa_sigma,
        lpa_kernel=config.lpa_kernel,
    )


def _smooth_field(xd: Tensor, yd: Tensor, zd: Tensor, sigma: float) -> Field:
    """Gaussian-smooth each component of a displacement field (no-op if sigma<=0)."""
    if sigma <= 0:
        return xd, yd, zd
    return (
        _separable_smooth_3d(xd, sigma, kernel_type="gauss"),
        _separable_smooth_3d(yd, sigma, kernel_type="gauss"),
        _separable_smooth_3d(zd, sigma, kernel_type="gauss"),
    )


def _apply_axis_flags(xd: Tensor, yd: Tensor, zd: Tensor, flags: int) -> Field:
    """Zero the X/Y/Z displacement component per the no-disp bit flags."""
    if flags & NO_X_DISP:
        xd = torch.zeros_like(xd)
    if flags & NO_Y_DISP:
        yd = torch.zeros_like(yd)
    if flags & NO_Z_DISP:
        zd = torch.zeros_like(zd)
    return xd, yd, zd


def invert_displacement_field(
    xd: Tensor, yd: Tensor, zd: Tensor, n_iter: int = 8
) -> tuple[Tensor, Tensor, Tensor]:
    """Approximate inverse of a displacement field by fixed-point iteration.

    For a field ``d`` (output->source), the inverse ``e`` satisfies
    ``e(x) = -d(x + e(x))``. Iterating from ``e0 = -d`` converges for the smooth,
    small-to-moderate fields SyN produces. Sampling uses trilinear interpolation.
    """
    nz, ny, nx = xd.shape
    device = xd.device
    kk, jj, ii = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32, device=device),
        torch.arange(ny, dtype=torch.float32, device=device),
        torch.arange(nx, dtype=torch.float32, device=device),
        indexing="ij",
    )
    ex = -xd
    ey = -yd
    ez = -zd
    for _ in range(n_iter):
        sx = (ii + ex).reshape(-1)
        sy = (jj + ey).reshape(-1)
        sz = (kk + ez).reshape(-1)
        dx = trilinear_interpolate(xd, sx, sy, sz).reshape(nz, ny, nx)
        dy = trilinear_interpolate(yd, sx, sy, sz).reshape(nz, ny, nx)
        dz = trilinear_interpolate(zd, sx, sy, sz).reshape(nz, ny, nx)
        ex = -dx
        ey = -dy
        ez = -dz
    return ex, ey, ez


# ---------------------------------------------------------------------------
# Multiresolution helpers
# ---------------------------------------------------------------------------


def _shrunk_shape(shape: tuple[int, int, int], factor: int) -> tuple[int, int, int]:
    nz, ny, nx = shape
    return (
        max(1, (nz + factor - 1) // factor),
        max(1, (ny + factor - 1) // factor),
        max(1, (nx + factor - 1) // factor),
    )


def _resize_volume(vol: Tensor, target: tuple[int, int, int]) -> Tensor:
    """Trilinear-resample a (nz,ny,nx) volume to ``target`` shape."""
    if tuple(vol.shape) == target:
        return vol
    out = F.interpolate(vol[None, None], size=target, mode="trilinear", align_corners=True)
    return out[0, 0]


def _resize_field(xd: Tensor, yd: Tensor, zd: Tensor, target: tuple[int, int, int]) -> Field:
    """Resample a displacement field to ``target`` and rescale displacements.

    Displacements are in voxel units, so a change of grid spacing scales each
    component by the per-axis size ratio.
    """
    nz, ny, nx = xd.shape
    tz, ty, tx = target
    if (nz, ny, nx) == target:
        return xd, yd, zd
    sx = (tx - 1) / (nx - 1) if nx > 1 else 1.0
    sy = (ty - 1) / (ny - 1) if ny > 1 else 1.0
    sz = (tz - 1) / (nz - 1) if nz > 1 else 1.0
    rx = _resize_volume(xd, target) * sx
    ry = _resize_volume(yd, target) * sy
    rz = _resize_volume(zd, target) * sz
    return rx, ry, rz


# ---------------------------------------------------------------------------
# One pyramid level
# ---------------------------------------------------------------------------


def _convergence_value(costs: list[float], window: int) -> float:
    """Trailing-window convergence measure (ANTs WindowConvergenceMonitoringFunction).

    Fits a line to the last ``window`` cost values (range-normalized to [0,1] so the
    measure is metric-scale-free) and returns the downward slope: positive while the
    cost is still falling, ~0 or negative once it flattens, rises, or oscillates.
    A level is "converged" when this drops below the convergence threshold.
    """
    w = costs[-window:]
    n = len(w)
    y = torch.tensor(w, dtype=torch.float64)
    rng = (y.max() - y.min()).item()
    if rng < 1e-12:
        return 0.0  # perfectly flat -> converged
    y = (y - y.min()) / rng
    x = torch.arange(n, dtype=torch.float64)
    x = x - x.mean()
    slope = (x * (y - y.mean())).sum() / (x * x).sum()
    return float(-slope)


def _syn_level(
    fixed: Tensor,
    moving: Tensor,
    weight: Tensor,
    fields: tuple[Field, Field, Field, Field],
    n_iter: int,
    config: SynConfig,
    level_tag: str = "",
) -> tuple[Field, Field, Field, Field]:
    """Run up to ``n_iter`` symmetric SyN updates at one resolution.

    ``fields`` is ``(phi_f, inv_f, phi_m, inv_m)`` as (xd, yd, zd) triples on this
    grid. Returns the **best-cost** fields seen, not the last: greedy SyN overshoots
    its optimum (cost falls, then rises/oscillates), so we snapshot the lowest-cost
    state and restore it. This makes "run to exhaustion" safe — extra iterations can
    never return a worse warp than the minimum. Early stopping (a trailing-window cost
    trend) ends the level once it has flattened; disable it (convergence_window<=0) to
    always run the full ``n_iter``.
    """
    (fxd, fyd, fzd), (ifxd, ifyd, ifzd), (mxd, myd, mzd), (imxd, imyd, imzd) = fields
    flags = config.warp_flags
    window = config.convergence_window

    def _snapshot() -> tuple[Field, Field, Field, Field]:
        return (
            (fxd.clone(), fyd.clone(), fzd.clone()),
            (ifxd.clone(), ifyd.clone(), ifzd.clone()),
            (mxd.clone(), myd.clone(), mzd.clone()),
            (imxd.clone(), imyd.clone(), imzd.clone()),
        )

    best_cost = float("inf")
    best_fields = _snapshot()
    costs: list[float] = []

    bar = None
    if _tqdm is not None and config.verb >= 1 and n_iter >= 5:
        bar = _tqdm(total=n_iter, desc=f"SyN {level_tag}", leave=True)

    for _ in range(n_iter):
        # Leaf copies for this step's gradient.
        leaves = [t.detach().requires_grad_(True) for t in (fxd, fyd, fzd, mxd, myd, mzd)]
        lf, lm = leaves[:3], leaves[3:]

        fmid = _warp_diff(fixed, lf[0], lf[1], lf[2])
        mmid = _warp_diff(moving, lm[0], lm[1], lm[2])
        cost = _metric_cost(fmid, mmid, weight, config)
        cost_val = cost.item()
        costs.append(cost_val)

        # The cost reflects the current (pre-update) fields -- snapshot them as best.
        if cost_val < best_cost:
            best_cost = cost_val
            best_fields = _snapshot()

        if bar is not None:
            bar.update(1)
            bar.set_postfix(cost=f"{cost_val:.5f}", best=f"{best_cost:.5f}")

        # Early stop once the windowed cost trend has flattened (or turned upward).
        if window > 0 and len(costs) >= window:
            if _convergence_value(costs, window) < config.convergence_threshold:
                break

        grads = torch.autograd.grad(cost, leaves)
        # Descent direction (cost is lower=better) for each half-field.
        ufx, ufy, ufz = -grads[0], -grads[1], -grads[2]
        umx, umy, umz = -grads[3], -grads[4], -grads[5]

        # Fluid regularization of the update fields.
        ufx, ufy, ufz = _smooth_field(ufx, ufy, ufz, config.update_var)
        umx, umy, umz = _smooth_field(umx, umy, umz, config.update_var)

        # Normalize the joint update so the largest displacement step is grad_step.
        norm = torch.sqrt(ufx**2 + ufy**2 + ufz**2 + umx**2 + umy**2 + umz**2)
        scale = config.grad_step / norm.max().clamp(min=_EPS)

        fxd = fxd + scale * ufx
        fyd = fyd + scale * ufy
        fzd = fzd + scale * ufz
        mxd = mxd + scale * umx
        myd = myd + scale * umy
        mzd = mzd + scale * umz

        # Elastic regularization of the total fields.
        fxd, fyd, fzd = _smooth_field(fxd, fyd, fzd, config.total_var)
        mxd, myd, mzd = _smooth_field(mxd, myd, mzd, config.total_var)

        if flags:
            fxd, fyd, fzd = _apply_axis_flags(fxd, fyd, fzd, flags)
            mxd, myd, mzd = _apply_axis_flags(mxd, myd, mzd, flags)

        # Re-derive inverses, then re-derive forwards from them (symmetrize).
        ifxd, ifyd, ifzd = invert_displacement_field(fxd, fyd, fzd, config.invert_iters)
        fxd, fyd, fzd = invert_displacement_field(ifxd, ifyd, ifzd, config.invert_iters)
        imxd, imyd, imzd = invert_displacement_field(mxd, myd, mzd, config.invert_iters)
        mxd, myd, mzd = invert_displacement_field(imxd, imyd, imzd, config.invert_iters)

    if bar is not None:
        bar.close()
    if config.verb >= 1:
        print(f"  {level_tag}: {len(costs)} iters, best cost {best_cost:.5f}")

    return best_fields


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


@dataclass
class SynResult:
    """Outputs of :func:`formwarp`. All fields are (nz, ny, nx) in voxel units."""

    warped: Tensor
    """Moving image resampled into fixed space (original size)."""

    fwd: tuple[Tensor, Tensor, Tensor]
    """moving->fixed total warp (use this to resample moving into fixed space)."""

    inv: tuple[Tensor, Tensor, Tensor]
    """fixed->moving total warp (the inverse)."""

    fixed_to_mid: tuple[Tensor, Tensor, Tensor] = field(
        default_factory=lambda: (torch.empty(0),) * 3
    )
    """phi_f: middle->fixed half-warp."""

    moving_to_mid: tuple[Tensor, Tensor, Tensor] = field(
        default_factory=lambda: (torch.empty(0),) * 3
    )
    """phi_m: middle->moving half-warp."""

    mid_to_fixed: tuple[Tensor, Tensor, Tensor] = field(
        default_factory=lambda: (torch.empty(0),) * 3
    )
    """inv_f: fixed->middle half-warp (inverse of phi_f)."""

    mid_to_moving: tuple[Tensor, Tensor, Tensor] = field(
        default_factory=lambda: (torch.empty(0),) * 3
    )
    """inv_m: moving->middle half-warp (inverse of phi_m)."""


def formwarp(
    fixed: Tensor,
    moving: Tensor,
    weight: Tensor | None = None,
    mask: Tensor | None = None,
    config: SynConfig | None = None,
    device: torch.device | None = None,
) -> SynResult:
    """Register ``moving`` to ``fixed`` with symmetric SyN.

    Args:
        fixed: (nz, ny, nx) fixed/target image (the ``-base``).
        moving: (nz, ny, nx) moving image to deform (the ``-source``).
        weight: Optional (nz, ny, nx) weight image (metric emphasis). Defaults to ones.
        mask: Optional (nz, ny, nx) mask; restricts the metric to mask>0.
        config: :class:`SynConfig`. Uses defaults if None.
        device: Torch device. Inferred from ``fixed`` if None.

    Returns:
        :class:`SynResult` with the moving->fixed warp, its inverse, the warped image,
        and the four SyN half-warps.
    """
    if config is None:
        config = SynConfig()
    if device is None:
        device = fixed.device
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    fixed = fixed.float().to(device)
    moving = moving.float().to(device)
    full_shape = tuple(fixed.shape)  # type: ignore[assignment]

    if weight is not None:
        weight = weight.float().to(device)
    if mask is not None:
        mask = mask.float().to(device)

    n_levels = len(config.shrink_factors)
    if not (len(config.smoothing_sigmas) == len(config.iterations) == n_levels):
        raise ValueError("shrink_factors, smoothing_sigmas and iterations must have equal length")

    # Half-fields on the full grid, refined coarse-to-fine. zero == identity.
    zeros = lambda: torch.zeros(full_shape, device=device)  # noqa: E731
    phi_f = (zeros(), zeros(), zeros())
    inv_f = (zeros(), zeros(), zeros())
    phi_m = (zeros(), zeros(), zeros())
    inv_m = (zeros(), zeros(), zeros())

    for lev in range(n_levels):
        factor = config.shrink_factors[lev]
        sigma = config.smoothing_sigmas[lev]
        n_iter = config.iterations[lev]
        target = _shrunk_shape(full_shape, factor)  # type: ignore[arg-type]

        # Pre-smooth (anti-alias) then shrink the images for this level.
        f_lvl = _resize_volume(_separable_smooth_3d(fixed, sigma) if sigma > 0 else fixed, target)
        m_lvl = _resize_volume(_separable_smooth_3d(moving, sigma) if sigma > 0 else moving, target)

        if weight is not None:
            w_lvl = _resize_volume(weight, target).clamp(min=0.0)
        elif mask is not None:
            w_lvl = (_resize_volume(mask, target) > 0.5).float()
        else:
            w_lvl = torch.ones(target, device=device)

        # Bring the running fields onto this level's grid.
        phi_f = _resize_field(*phi_f, target)
        inv_f = _resize_field(*inv_f, target)
        phi_m = _resize_field(*phi_m, target)
        inv_m = _resize_field(*inv_m, target)

        if config.verb >= 1:
            print(
                f"formwarp: level {lev + 1}/{n_levels} shrink={factor} "
                f"smooth={sigma:g}vox grid={target[2]}x{target[1]}x{target[0]} "
                f"iters={n_iter} metric={config.metric}"
            )

        phi_f, inv_f, phi_m, inv_m = _syn_level(
            f_lvl,
            m_lvl,
            w_lvl,
            (phi_f, inv_f, phi_m, inv_m),
            n_iter,
            config,
            level_tag=f"L{lev + 1}",
        )

    # Restore to full resolution if the finest level was still shrunk.
    phi_f = _resize_field(*phi_f, full_shape)  # type: ignore[arg-type]
    inv_f = _resize_field(*inv_f, full_shape)  # type: ignore[arg-type]
    phi_m = _resize_field(*phi_m, full_shape)  # type: ignore[arg-type]
    inv_m = _resize_field(*inv_m, full_shape)  # type: ignore[arg-type]

    # Compose the half-warps into total transforms.
    #   moving->fixed (resample moving into fixed grid): apply inv_f (fixed->middle)
    #     then phi_m (middle->moving).
    #   fixed->moving (inverse): apply inv_m (moving->middle) then phi_f (middle->fixed).
    dummy_hdr: dict = {}
    fwd = compose_warp_then_warp(NonlinearWarp(*inv_f, dummy_hdr), NonlinearWarp(*phi_m, dummy_hdr))
    inv = compose_warp_then_warp(NonlinearWarp(*inv_m, dummy_hdr), NonlinearWarp(*phi_f, dummy_hdr))
    fwd_t = (fwd.xd, fwd.yd, fwd.zd)
    inv_t = (inv.xd, inv.yd, inv.zd)

    # Final warped image: one pass of the original moving through the total warp,
    # with the high-order final kernel (sharpness), like qwarp's final pass.
    warped = warp_image(moving, fwd_t[0], fwd_t[1], fwd_t[2], mode=config.final_interp)

    return SynResult(
        warped=warped,
        fwd=fwd_t,
        inv=inv_t,
        fixed_to_mid=phi_f,
        moving_to_mid=phi_m,
        mid_to_fixed=inv_f,
        mid_to_moving=inv_m,
    )
