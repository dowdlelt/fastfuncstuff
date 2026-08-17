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

from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from fastfuncstuff.memory import plan_nonlinear_memory

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
from .interp import _grid_sample_3d, trilinear_interpolate_multi, warp_image
from .mask import cross_fill_no_data
from .nwarpforge import NonlinearWarp, compose_warp_then_warp

_EPS = 1e-6

# A displacement field as a (xd, yd, zd) triple, each (nz, ny, nx) in voxel units.
Field = tuple[Tensor, Tensor, Tensor]

# Bit flags shared with the qwarp engine's warp_flags (no-displacement per axis).
NO_X_DISP = 1
NO_Y_DISP = 2
NO_Z_DISP = 4

# The engine's own short names, plus everything the shared registry declares
# differentiable. One list, so a metric added to the registry is immediately
# available to optimise with and not only to score.
_ENGINE_METRICS = ("cc", "lpa", "lpc", "pearson", "mse")


def _all_metrics() -> tuple[str, ...]:
    from .metrics import differentiable_metrics

    return tuple(dict.fromkeys((*_ENGINE_METRICS, *differentiable_metrics())))


METRICS = _all_metrics()


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

    iterations: tuple[int, ...] = (300, 210, 120)
    """Per-level iteration **ceiling** — a bound, not a target.

    Deliberately set about 3x higher than the level actually needs, because this is
    not a quality knob to be tuned: too few iterations is always wrong, and too many
    is free. Early stopping ends a level once its cost has flattened, and best-iterate
    restore means even exhaustion cannot return a worse warp than the minimum seen. So
    the only thing a low ceiling can do is cut a level off while it is still improving.

    That was not hypothetical. At the previous default of (100, 70, 40) the finest
    level ran to its cap of 40 and stopped; given room it ran 47 and kept improving —
    the shipped default was starving it, invisibly, because a level that hits its cap
    looks exactly like one that finished. Raising the ceiling cost 15% of the runtime
    on that pair and nothing at all on the flow engine, which stopped at [12, 14, 37]
    either way.

    If :attr:`~formwarp.LevelStats.starved` ever comes back true, this is too low.
    """

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

    fold_guard: float = 0.5
    """Per-round shrink (0..1) applied by the **local** anti-folding damping; 0 disables it.

    SyN has no fold control at all other than ``update_var``/``total_var``, which are
    global: the only way to stop a handful of voxels inverting was to blur the entire
    field, paying fit everywhere for a defect in a few places. This tests each half
    field's Jacobian before accepting a step and halves the step only in the
    neighbourhood that would fold. When nothing would fold it costs two determinants
    per iteration and changes nothing."""

    jac_floor: float = 0.05
    """Prospective ``det(J)`` below which :attr:`fold_guard` damps a step. Guarded on
    each half field rather than on the composed warp: the halves are what
    ``invert_displacement_field`` has to invert, and an inverted half is where a folded
    composite comes from."""

    fold_damp_rounds: int = 6
    """Maximum local-damping retries per iteration before taking the least-folded step."""

    fold_aware_best: bool = True
    """Require a state to be fold-free before it can become the best-so-far. Without
    it the snapshot is decided by the image metric alone, and folding *improves* the
    image metric — so the mechanism that makes over-running safe selects the folded
    state."""

    void_guard: float = 1.0
    """Strength (0..1) of the no-data-boundary guard: how much of the void-normal
    component of each update is removed near a coverage boundary. 0 disables it.
    Only has an effect when the caller supplies coverage masks."""

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


def _warp_diff(
    source: Tensor,
    xd: Tensor,
    yd: Tensor,
    zd: Tensor,
    voxel_grid: tuple[Tensor, Tensor, Tensor] | None = None,
) -> Tensor:
    """Differentiable linear warp: output[p] = source[p + d(p)].

    A grad-friendly companion to :func:`interp.warp_image_linear` (which masks
    out-of-bounds samples in place and is meant for the non-autograd path). Here we
    rely on grid_sample's border padding so the whole thing stays in the autograd
    graph w.r.t. the displacement fields.
    """
    nz, ny, nx = source.shape
    device = source.device
    dtype = source.dtype
    if voxel_grid is None:
        kk, jj, ii = torch.meshgrid(
            torch.arange(nz, dtype=dtype, device=device),
            torch.arange(ny, dtype=dtype, device=device),
            torch.arange(nx, dtype=dtype, device=device),
            indexing="ij",
        )
    else:
        kk, jj, ii = voxel_grid
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

    The window statistics are **weighted**. Weighting only the outer sum decides which
    windows count but not what goes into one, so a window near the edge of the mask
    still averages in everything the mask was meant to exclude: the no-data void, and
    the air beside the brain. Both wreck the gradient exactly where the deformation is
    largest -- a void cross-filled from the other image correlates perfectly and
    inflates the local CC, so real misalignment in the rest of the window produces a
    weak update, and the tissue at the boundary refuses to move. Accumulating every
    box under ``w`` instead means a boundary window measures only its valid part: full
    drive from the real tissue, nothing from the void or the air. With ``w`` constant
    this is algebraically identical to the unweighted form.
    """
    r = float(radius)

    def box(v: Tensor) -> Tensor:
        return _separable_smooth_3d(v, r, kernel_type="box")

    wn = box(weight).clamp(min=_EPS)
    am = box(weight * a) / wn
    bm = box(weight * b) / wn
    cov = box(weight * a * b) / wn - am * bm
    va = (box(weight * a * a) / wn - am * am).clamp(min=_EPS)
    vb = (box(weight * b * b) / wn - bm * bm).clamp(min=_EPS)
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
    # Anything else is looked up in the shared registry, so a metric declared
    # differentiable there is optimisable here without being re-listed. Imported
    # lazily: metrics.py reaches back into this module for the LNCC kernel, and a
    # module-level import would close the loop.
    from .metrics import METRICS as _REG
    from .metrics import differentiable_cost

    if metric in _REG:
        return differentiable_cost(metric, a, b, weight, cc_radius=cc_radius)
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
    packed = torch.stack((xd, yd, zd), dim=0)[None]
    smoothed = _separable_smooth_3d(packed, sigma, kernel_type="gauss")[0]
    return smoothed[0], smoothed[1], smoothed[2]


def _void_guard_field(cover: Tensor, sigma: float = 2.0) -> tuple[Tensor, ...] | None:
    """Local outward normal of a no-data region, plus a band weight around it.

    Returns ``(nx, ny, nz, band)`` on ``cover``'s grid, or None if there is no void.

    The point is that the constraint at a coverage boundary is **directional**, and
    treating it isotropically throws away good deformation. Where a slab of missing
    data sits below the brain, motion *down into it* is the artifact -- but an in-plane
    shift along that same boundary is perfectly legitimate anatomy, and often exactly
    what the registration needs. ``-noZdis`` gets this right by accident when the void
    happens to be axis-aligned and nothing else in the volume needs z. Deriving the
    normal from the coverage mask itself gets it right in general: it constrains the
    one direction that reaches into the void, wherever the boundary is and however it
    is oriented, and leaves the two tangential directions completely free.

    The band weight is the normalized gradient magnitude, so the constraint is
    strongest exactly at the boundary and fades smoothly to nothing in the interior --
    no hard cutoff to tune, and identically zero for a volume with full coverage.
    """
    c = _separable_smooth_3d(cover.float(), sigma, kernel_type="gauss")
    gz, gy, gx = torch.gradient(c)  # array dims are (z, y, x)
    mag = torch.sqrt(gx * gx + gy * gy + gz * gz)
    peak = mag.max()
    if float(peak) < _EPS:
        return None  # fully covered: nothing to guard against
    inv = 1.0 / mag.clamp(min=_EPS)
    return gx * inv, gy * inv, gz * inv, (mag / peak).clamp(0.0, 1.0)


def _apply_void_guard(xd: Tensor, yd: Tensor, zd: Tensor, guard, strength: float) -> Field:
    """Remove the void-normal component of an update field, scaled by the band weight."""
    nx, ny, nz, band = guard
    k = strength * band * (xd * nx + yd * ny + zd * nz)
    return xd - k * nx, yd - k * ny, zd - k * nz


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
    xd: Tensor,
    yd: Tensor,
    zd: Tensor,
    n_iter: int = 8,
    voxel_grid: tuple[Tensor, Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Approximate inverse of a displacement field by fixed-point iteration.

    For a field ``d`` (output->source), the inverse ``e`` satisfies
    ``e(x) = -d(x + e(x))``. Iterating from ``e0 = -d`` converges for the smooth,
    small-to-moderate fields SyN produces. Sampling uses trilinear interpolation.
    """
    nz, ny, nx = xd.shape
    device = xd.device
    if voxel_grid is None:
        kk, jj, ii = torch.meshgrid(
            torch.arange(nz, dtype=torch.float32, device=device),
            torch.arange(ny, dtype=torch.float32, device=device),
            torch.arange(nx, dtype=torch.float32, device=device),
            indexing="ij",
        )
    else:
        kk, jj, ii = voxel_grid
    ex = -xd
    ey = -yd
    ez = -zd
    field = torch.stack([xd, yd, zd], dim=0)
    for _ in range(n_iter):
        sx = (ii + ex).reshape(-1)
        sy = (jj + ey).reshape(-1)
        sz = (kk + ez).reshape(-1)
        sampled = trilinear_interpolate_multi(field, sx, sy, sz).T.reshape(3, nz, ny, nx)
        dx, dy, dz = sampled.unbind(0)
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


@dataclass
class LevelStats:
    """What one pyramid level actually did — the telemetry behind ``-search``.

    Both engines run to an iteration cap, monitor a metric, and keep the best
    iterate rather than the last. That makes over-running *safe*, which is good, but
    it also makes it invisible: a level that found its answer at iteration 4 and then
    burned 96 more looks exactly like a level that needed all 100. Recording where
    the best iterate actually landed is what separates the two, and it turns "how
    many iterations should we use" from a guess into a reading:

    * ``best_iter`` far below ``iters_run`` — the level over-ran and fell back. The
      extra iterations bought nothing; the convergence test is too lax.
    * ``best_iter`` at the end with ``hit_cap`` — the level was starved. It was still
      improving when it ran out, so the cap is the binding constraint.
    """

    iters_run: int
    best_iter: int
    n_iter_cap: int
    early_stopped: bool
    best_cost: float = float("nan")
    damped_iters: int = 0  # iterations where the anti-fold guard had to intervene
    fold_fallback: bool = False
    """No iterate at this level was fold-free, so the best *illegal* one was kept.

    Usually means the level above handed down an already-folded field, which the
    guard cannot undo — it damps updates, it does not repair history. A config that
    reports this is one whose whole schedule is too aggressive, not one that merely
    took a bad step."""

    @property
    def hit_cap(self) -> bool:
        return self.iters_run >= self.n_iter_cap

    @property
    def starved(self) -> bool:
        """Still improving when the iteration cap stopped it."""
        return self.hit_cap and self.best_iter >= self.iters_run - 1

    @property
    def wasted_iters(self) -> int:
        """Iterations run after the one that was ultimately kept."""
        return max(0, self.iters_run - 1 - self.best_iter)

    def describe(self) -> str:
        bits = [f"{self.iters_run} iters", f"best @{self.best_iter}"]
        if self.starved:
            bits.append("STARVED (still improving at the cap)")
        elif self.wasted_iters:
            bits.append(f"{self.wasted_iters} wasted")
        if self.damped_iters:
            bits.append(f"{self.damped_iters} fold-damped")
        if self.fold_fallback:
            bits.append("NO LEGAL ITERATE (kept a folded one)")
        bits.append(f"best cost {self.best_cost:.5f}")
        return ", ".join(bits)

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(hit_cap=self.hit_cap, starved=self.starved, wasted_iters=self.wasted_iters)
        return d


def _grad3(vol: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Central-difference spatial gradient of a (nz,ny,nx) volume, per voxel index.

    Returns ``(gx, gy, gz)`` — the derivative along the last, middle and first axis
    respectively, matching the (x, y, z) displacement component order used everywhere
    else in the project. Edges use a one-sided difference via replicate padding.
    """
    gz = torch.zeros_like(vol)
    gy = torch.zeros_like(vol)
    gx = torch.zeros_like(vol)
    if vol.shape[0] > 2:
        gz[1:-1] = 0.5 * (vol[2:] - vol[:-2])
        gz[0] = vol[1] - vol[0]
        gz[-1] = vol[-1] - vol[-2]
    if vol.shape[1] > 2:
        gy[:, 1:-1] = 0.5 * (vol[:, 2:] - vol[:, :-2])
        gy[:, 0] = vol[:, 1] - vol[:, 0]
        gy[:, -1] = vol[:, -1] - vol[:, -2]
    if vol.shape[2] > 2:
        gx[..., 1:-1] = 0.5 * (vol[..., 2:] - vol[..., :-2])
        gx[..., 0] = vol[..., 1] - vol[..., 0]
        gx[..., -1] = vol[..., -1] - vol[..., -2]
    return gx, gy, gz


def jacobian_determinant(xd: Tensor, yd: Tensor, zd: Tensor) -> Tensor:
    """Determinant of the transform Jacobian ``I + grad(d)`` per voxel.

    Values <= 0 mark folded (non-injective) voxels; this is the diagnostic that says
    whether a flow field stayed anatomically plausible.
    """
    dxx, dxy, dxz = _grad3(xd)
    dyx, dyy, dyz = _grad3(yd)
    dzx, dzy, dzz = _grad3(zd)
    j11, j12, j13 = 1.0 + dxx, dxy, dxz
    j21, j22, j23 = dyx, 1.0 + dyy, dyz
    j31, j32, j33 = dzx, dzy, 1.0 + dzz
    return (
        j11 * (j22 * j33 - j23 * j32)
        - j12 * (j21 * j33 - j23 * j31)
        + j13 * (j21 * j32 - j22 * j31)
    )


def _fold_damping_mask(jac: Tensor, floor: float, strength: float, radius: float = 1.0) -> Tensor:
    """A smooth 0..1 field: how much of the step to give back, per voxel.

    Dilate-then-smooth rather than smooth-alone. A Gaussian applied straight to an
    isolated bad voxel peaks at a fraction of 1, so the damping would be weakest
    exactly where a fold *starts* — and folds start as one voxel. Dilating first puts
    a solid core over the offending neighbourhood, and the smoothing only softens its
    edge. Measured at these settings: a lone folded voxel reaches 0.97 of full
    strength and a solid cluster 1.00, against 0.33 and 0.76 for smoothing alone.

    The soft edge is not cosmetic: scaling an update discontinuously writes a step
    into the displacement field, which is a fresh source of the defect being repaired.
    """
    bad = (jac < floor).to(jac.dtype)[None, None]
    core = torch.nn.functional.max_pool3d(bad, kernel_size=5, stride=1, padding=2)[0, 0]
    return (strength * _separable_smooth_3d(core, radius, kernel_type="gauss")).clamp(0.0, 1.0)


def _additive_step_with_fold_guard(
    prev: Field, update: Field, total_var: float, config: SynConfig
) -> tuple[Field, Tensor, int]:
    """Add ``update`` to ``prev``, backing off locally wherever that would fold.

    The elastic smoothing happens inside, because the guard has to judge the field the
    level will actually keep, and building it twice would be the only alternative.
    """

    def _apply(u: Field) -> tuple[Field, Tensor]:
        cand = (prev[0] + u[0], prev[1] + u[1], prev[2] + u[2])
        cand = _smooth_field(cand[0], cand[1], cand[2], total_var)
        return cand, jacobian_determinant(*cand)

    ux, uy, uz = update
    cand, jac = _apply((ux, uy, uz))
    if config.fold_guard <= 0:
        return cand, jac, 0

    best_cand, best_jac, damped = cand, jac, 0
    for _ in range(config.fold_damp_rounds):
        if float(best_jac.min()) >= config.jac_floor:
            break
        keep = 1.0 - _fold_damping_mask(best_jac, config.jac_floor, config.fold_guard)
        ux, uy, uz = ux * keep, uy * keep, uz * keep
        cand, jac = _apply((ux, uy, uz))
        damped += 1
        if float(jac.min()) > float(best_jac.min()):
            best_cand, best_jac = cand, jac
    return best_cand, best_jac, damped


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
    guard: tuple[Tensor, ...] | None = None,
) -> tuple[tuple[Field, Field, Field, Field], LevelStats]:
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
    nz, ny, nx = fixed.shape
    voxel_grid = torch.meshgrid(
        torch.arange(nz, dtype=fixed.dtype, device=fixed.device),
        torch.arange(ny, dtype=fixed.dtype, device=fixed.device),
        torch.arange(nx, dtype=fixed.dtype, device=fixed.device),
        indexing="ij",
    )

    def _snapshot() -> tuple[Field, Field, Field, Field]:
        return (
            (fxd.clone(), fyd.clone(), fzd.clone()),
            (ifxd.clone(), ifyd.clone(), ifzd.clone()),
            (mxd.clone(), myd.clone(), mzd.clone()),
            (imxd.clone(), imyd.clone(), imzd.clone()),
        )

    # Two running bests: the lowest-cost *legal* state, and the lowest-cost state of
    # any kind. The second is a fallback, not a preference -- a level can inherit an
    # already-folded field from the level above, in which case nothing it produces is
    # legal and refusing to return anything would fail a run that is merely imperfect.
    best_cost, best_fields, best_iter = float("inf"), _snapshot(), 0
    any_cost, any_fields, any_iter = float("inf"), _snapshot(), 0
    costs: list[float] = []
    jac_f = jacobian_determinant(fxd, fyd, fzd)
    jac_m = jacobian_determinant(mxd, myd, mzd)
    n_damped = 0

    bar = None
    if _tqdm is not None and config.verb >= 1 and n_iter >= 5:
        bar = _tqdm(total=n_iter, desc=f"SyN {level_tag}", leave=True)

    for _ in range(n_iter):
        # Leaf copies for this step's gradient.
        leaves = [t.detach().requires_grad_(True) for t in (fxd, fyd, fzd, mxd, myd, mzd)]
        lf, lm = leaves[:3], leaves[3:]

        fmid = _warp_diff(fixed, lf[0], lf[1], lf[2], voxel_grid)
        mmid = _warp_diff(moving, lm[0], lm[1], lm[2], voxel_grid)
        cost = _metric_cost(fmid, mmid, weight, config)
        cost_val = cost.item()
        costs.append(cost_val)

        # The cost reflects the current (pre-update) fields -- snapshot them as best.
        # Legality of those fields was established when they were built, at the end of
        # the previous iteration, so no determinant is recomputed here.
        legal = not config.fold_aware_best or (
            min(float(jac_f.min()), float(jac_m.min())) >= config.jac_floor
        )
        if cost_val < any_cost:
            any_cost, any_iter = cost_val, len(costs) - 1
            any_fields = _snapshot()
        if cost_val < best_cost and legal:
            best_cost, best_iter = cost_val, len(costs) - 1
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

        # Then drop whatever of that update points into a no-data region, leaving the
        # two tangential directions untouched. After the smoothing, so the fluid step
        # cannot reintroduce a normal component from a neighbour.
        if guard is not None:
            ufx, ufy, ufz = _apply_void_guard(ufx, ufy, ufz, guard, config.void_guard)
            umx, umy, umz = _apply_void_guard(umx, umy, umz, guard, config.void_guard)

        # Normalize the joint update so the largest displacement step is grad_step.
        norm = torch.sqrt(ufx**2 + ufy**2 + ufz**2 + umx**2 + umy**2 + umz**2)
        scale = config.grad_step / norm.max().clamp(min=_EPS)

        # Step + elastic regularization, with the local anti-fold backoff applied to
        # each half field. Both happen inside the guard: it has to judge the field the
        # level will keep, so it builds it once and hands it back.
        (fxd, fyd, fzd), jac_f, df = _additive_step_with_fold_guard(
            (fxd, fyd, fzd), (scale * ufx, scale * ufy, scale * ufz), config.total_var, config
        )
        (mxd, myd, mzd), jac_m, dm = _additive_step_with_fold_guard(
            (mxd, myd, mzd), (scale * umx, scale * umy, scale * umz), config.total_var, config
        )
        n_damped += 1 if (df or dm) else 0

        if flags:
            fxd, fyd, fzd = _apply_axis_flags(fxd, fyd, fzd, flags)
            mxd, myd, mzd = _apply_axis_flags(mxd, myd, mzd, flags)

        # Re-derive inverses, then re-derive forwards from them (symmetrize).
        ifxd, ifyd, ifzd = invert_displacement_field(fxd, fyd, fzd, config.invert_iters, voxel_grid)
        fxd, fyd, fzd = invert_displacement_field(ifxd, ifyd, ifzd, config.invert_iters, voxel_grid)
        imxd, imyd, imzd = invert_displacement_field(mxd, myd, mzd, config.invert_iters, voxel_grid)
        mxd, myd, mzd = invert_displacement_field(imxd, imyd, imzd, config.invert_iters, voxel_grid)

    if bar is not None:
        bar.close()

    fold_fallback = best_cost == float("inf") and any_cost < float("inf")
    if fold_fallback:
        best_cost, best_fields, best_iter = any_cost, any_fields, any_iter

    stats = LevelStats(
        iters_run=len(costs),
        best_iter=best_iter,
        n_iter_cap=n_iter,
        early_stopped=len(costs) < n_iter,
        damped_iters=n_damped,
        best_cost=best_cost,
        fold_fallback=fold_fallback,
    )
    if config.verb >= 1:
        print(f"  {level_tag}: {stats.describe()}")

    # Every iteration's cost was non-finite, so best-restore handed back the fields we
    # started with: this level did nothing, and the run would go on to write an
    # identity warp as though it had succeeded. Fail loudly instead -- the cause is in
    # the inputs (non-finite voxels, an all-constant image, an empty mask), not here.
    if any_cost == float("inf"):
        raise ValueError(
            f"{level_tag}: metric was non-finite on every iteration, so no warp could be "
            "estimated. Check the base/source for NaN/Inf or constant-valued data, and "
            "that the metric mask/weight is not empty."
        )

    return best_fields, stats


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

    levels: list[LevelStats] = field(default_factory=list)
    """Per-level convergence telemetry: iterations run, which one was kept, whether the
    level was starved or over-ran, and how often the fold guard intervened."""


def formwarp(
    fixed: Tensor,
    moving: Tensor,
    weight: Tensor | None = None,
    mask: Tensor | None = None,
    fixed_cover: Tensor | None = None,
    moving_cover: Tensor | None = None,
    config: SynConfig | None = None,
    device: torch.device | None = None,
) -> SynResult:
    """Register ``moving`` to ``fixed`` with symmetric SyN.

    Args:
        fixed: (nz, ny, nx) fixed/target image (the ``-base``).
        moving: (nz, ny, nx) moving image to deform (the ``-source``).
        weight: Optional (nz, ny, nx) weight image (metric emphasis). Defaults to ones.
        mask: Optional (nz, ny, nx) mask; restricts the metric to mask>0.
        fixed_cover: Optional (nz, ny, nx) mask of where ``fixed`` holds real data.
        moving_cover: Optional (nz, ny, nx) mask of where ``moving`` holds real data.
            Both are excluded from the metric *and* cross-filled (see below).
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
    predicted, available = plan_nonlinear_memory(full_shape, device, "formwarp")
    if predicted > available and config.verb >= 1:
        print(
            f"WARNING: estimated SyN peak {predicted / 2**30:.1f} GiB exceeds "
            f"the {available / 2**30:.1f} GiB safe memory budget; use stronger shrink factors."
        )

    # Non-finite voxels have to go before the metric ever sees them. NaN survives the
    # CC box filter and every other smoothing here, spreading to the whole window, so
    # a single NaN slab makes the cost NaN on iteration 1 -- and then no iteration
    # beats the initial best, so every level returns the identity warp and the tool
    # writes a plausible-looking file having done nothing. Zeroing them is not a data
    # change we can avoid either way: grid_sample would smear the NaN across each
    # interpolation neighbourhood on the final warp pass regardless. Callers should
    # pair this with a coverage mask (mask/weight) so the zeroed rim doesn't become a
    # phantom edge in its own right; see mask.data_coverage_mask.
    n_bad = int((~torch.isfinite(fixed)).sum().item() + (~torch.isfinite(moving)).sum().item())
    if n_bad:
        if config.verb >= 1:
            pct = 100.0 * n_bad / (2 * fixed.numel())
            print(
                f"WARNING: {n_bad:,} non-finite voxels ({pct:.1f}% of base+source) "
                "treated as no-data (set to 0)"
            )
        fixed = torch.nan_to_num(fixed, nan=0.0, posinf=0.0, neginf=0.0)
        moving = torch.nan_to_num(moving, nan=0.0, posinf=0.0, neginf=0.0)

    if weight is not None:
        weight = weight.float().to(device)
    if mask is not None:
        mask = mask.float().to(device)

    # Exclude each image's no-data region from the metric and cross-fill it from the
    # other image, so the void presents no edge for the warp to chase. The fill is for
    # the METRIC ONLY -- ``moving`` is untouched, so the warped output below keeps
    # honest zeros where the source had no data. See mask.cross_fill_no_data.
    f_metric, m_metric, cover = cross_fill_no_data(fixed, moving, fixed_cover, moving_cover)
    if cover is not None:
        # Fold the shared support into whichever restriction the caller supplied.
        if weight is not None:
            weight = weight * cover.float()
        else:
            mask = cover.float() if mask is None else mask * cover.float()

    n_levels = len(config.shrink_factors)
    if not (len(config.smoothing_sigmas) == len(config.iterations) == n_levels):
        raise ValueError("shrink_factors, smoothing_sigmas and iterations must have equal length")

    # Half-fields on the full grid, refined coarse-to-fine. zero == identity.
    zeros = lambda: torch.zeros(full_shape, device=device)  # noqa: E731
    phi_f = (zeros(), zeros(), zeros())
    inv_f = (zeros(), zeros(), zeros())
    phi_m = (zeros(), zeros(), zeros())
    inv_m = (zeros(), zeros(), zeros())
    level_stats: list[LevelStats] = []

    for lev in range(n_levels):
        factor = config.shrink_factors[lev]
        sigma = config.smoothing_sigmas[lev]
        n_iter = config.iterations[lev]
        target = _shrunk_shape(full_shape, factor)  # type: ignore[arg-type]

        # Pre-smooth (anti-alias) then shrink the images for this level.
        f_lvl = _resize_volume(
            _separable_smooth_3d(f_metric, sigma) if sigma > 0 else f_metric, target
        )
        m_lvl = _resize_volume(
            _separable_smooth_3d(m_metric, sigma) if sigma > 0 else m_metric, target
        )

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

        # Rebuilt per level: the normal is a grid-relative quantity, so it has to be
        # measured on the grid the updates actually live on.
        guard = None
        if cover is not None and config.void_guard > 0.0:
            guard = _void_guard_field(_resize_volume(cover.float(), target))

        (phi_f, inv_f, phi_m, inv_m), stats = _syn_level(
            f_lvl,
            m_lvl,
            w_lvl,
            (phi_f, inv_f, phi_m, inv_m),
            n_iter,
            config,
            level_tag=f"L{lev + 1}",
            guard=guard,
        )
        level_stats.append(stats)

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
        levels=level_stats,
    )
