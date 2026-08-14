"""Optical-flow nonlinear registration (3D->3D) on the GPU.

A third nonlinear-registration backend alongside :mod:`fastfuncstuff.processing.warp`
(AFNI 3dQwarp patches) and :mod:`fastfuncstuff.processing.formwarp` (ANTs SyN). Where
those two *optimize* a similarity cost — qwarp over polynomial patch bases, SyN by
autograd on a dense field — this one *solves* for the displacement directly from the
optical-flow brightness-constancy equation, the same machinery ``ffs_locomoco`` uses for
residual EPI motion, promoted from a 1-D/2-D per-slice estimator to a full 3-D field.

The brightness-constancy assumption is that a voxel keeps its intensity as it moves:

    W(p + delta) = F(p)   =>   delta . grad(W) + (W - F) = 0                        (*)

which is one equation per voxel in three unknowns (the aperture problem). The three
force models differ only in how they close that system:

  ``demons``   Thirion's demons: take the minimum-norm solution of (*) along the
               gradient, with the Cachier/Vercauteren regularized denominator
               ``|grad|^2 + (W-F)^2 / K^2`` so that flat regions (tiny gradient) and
               large intensity mismatches produce a *small* step rather than an
               explosion. Cheap, and the workhorse default.
  ``lk``       Lucas-Kanade: assume delta is constant over a small neighbourhood and
               least-squares solve the resulting 3x3 structure-tensor system per voxel.
               Closes the aperture problem with spatial support instead of a prior, so
               it recovers displacement components tangential to an edge that demons
               cannot see. The 3-D analogue of ``locomoco.optical_flow_lk_3d``.
  ``hs``       Horn-Schunck: close (*) globally with a smoothness prior, solved by
               Jacobi iterations. The smoothest, slowest-moving option.

Because the raw flow field is only as rigid as its regularizer ("loosey goosey" at
small scales), two things keep it honest:

  * ``step_mode="diffeo"`` (default) treats each update as a stationary velocity field
    and exponentiates it by scaling-and-squaring before composing, so every increment
    is a diffeomorphism and the running field cannot fold. ``"additive"`` is the
    classic (faster, unguarded) demons update.
  * ``final_qwarp=True`` hands the converged flow field to the 3dQwarp engine as an
    initial warp for a fine-scale, pure image-match polish. Optical flow is good at
    finding the deformation; patch optimization is good at nailing the last half-voxel.

Brightness constancy is a modality assumption, and ``match`` is how far it is relaxed.
``localnorm`` (the default) locally z-scores both images, which removes bias fields,
shading and any spatially varying gain — enough for same-contrast pairs across sessions
or scanners. It does *not* survive a contrast inversion: the local z-score of an
inverted image is the negated map, so the force points backwards. For genuinely
cross-contrast pairs (T1 vs T2, EPI vs anat) use ``gradmag``, which registers locally
normalized gradient magnitude: edges land in the same place with the same sign in every
modality.

Displacement convention matches the rest of the project (see ``processing/interp.py``):
output voxel ``(i,j,k)`` samples its source at ``(i+xd, j+yd, k+zd)`` in **voxel units**.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from fastfuncstuff.memory import plan_nonlinear_memory

if TYPE_CHECKING:
    from .warp import QwarpConfig

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover - tqdm is a hard dep in practice
    _tqdm = None

from .cost import _separable_smooth_3d
from .formwarp import (
    NO_X_DISP,
    NO_Y_DISP,
    NO_Z_DISP,
    Field,
    _apply_axis_flags,
    _apply_void_guard,
    _convergence_value,
    _resize_field,
    _resize_volume,
    _shrunk_shape,
    _smooth_field,
    _void_guard_field,
    image_metric,
    invert_displacement_field,
)
from .interp import warp_image, warp_image_linear
from .mask import cross_fill_no_data
from .nwarpforge import NonlinearWarp, compose_warp_then_warp

_EPS = 1e-6

FORCES = ("demons", "lk", "hs")
MATCH_MODES = ("none", "meanstd", "localnorm", "gradmag")
STEP_MODES = ("diffeo", "additive")

# Re-exported so CLIs can build warp_flags without importing formwarp too.
__all__ = [
    "FORCES",
    "MATCH_MODES",
    "NO_X_DISP",
    "NO_Y_DISP",
    "NO_Z_DISP",
    "STEP_MODES",
    "OptiwarpConfig",
    "OptiwarpResult",
    "jacobian_determinant",
    "optiwarp",
]


@dataclass
class OptiwarpConfig:
    """Configuration for the optical-flow engine."""

    force: str = "demons"
    """Flow model: 'demons' (normalized gradient force), 'lk' (windowed Lucas-Kanade
    structure-tensor solve), or 'hs' (Horn-Schunck global smoothness)."""

    symmetric_force: bool = True
    """Use the symmetric demons force ``(grad W + grad F)/2`` instead of ``grad W``
    alone. Halves the direction bias between the two images; costs one extra gradient.
    Applies to 'demons' and 'hs'."""

    match: str = "localnorm"
    """Intensity preprocessing before the flow solve (see :func:`prep_intensity`):
    'none' (raw), 'meanstd' (global z-score), 'localnorm' (local z-score over
    ``match_sigma`` — removes bias fields and gain, the cross-session default), or
    'gradmag' (locally normalized gradient magnitude — the only mode that survives a
    contrast inversion, so the cross-modal choice).
    Estimation-only: the final warped image comes from the raw source."""

    match_sigma: float = 6.0
    """Gaussian sigma (voxels) of the neighbourhood for match='localnorm'/'gradmag'."""

    demons_noise: float = 1.0
    """Thirion/Cachier normalization K: the intensity difference (in units of the
    prepped images, so ~1 sigma after localnorm) at which the force is damped. Smaller
    = more conservative steps where the images disagree strongly."""

    lk_radius: int = 2
    """Half-width (voxels) of the Lucas-Kanade neighbourhood; window is (2r+1)^3."""

    lk_reg: float = 1e-2
    """Tikhonov ridge added to the LK structure tensor's diagonal, relative to its
    mean trace. Guards the aperture-problem-degenerate directions."""

    hs_alpha: float = 1.0
    """Horn-Schunck smoothness weight. Larger = smoother, slower-moving flow."""

    hs_iters: int = 20
    """Jacobi iterations per Horn-Schunck flow solve."""

    step_mode: str = "diffeo"
    """'diffeo' exponentiates each update (scaling-and-squaring) and composes, keeping
    the running field foldless; 'additive' adds the update directly (classic demons)."""

    max_step: float = 1.0
    """Cap (voxels) on the largest per-iteration displacement, applied after fluid
    smoothing. The step-size control; smaller is safer."""

    update_sigma: float = 1.0
    """Fluid regularization: Gaussian sigma (voxels) smoothing each update field.
    Demons' ``sigma_fluid``. 0 = off."""

    total_sigma: float = 1.0
    """Elastic/diffusion regularization: Gaussian sigma (voxels) smoothing the
    accumulated field. Demons' ``sigma_diffusion``. 0 = off (relies on the fluid term
    alone, which is what makes flow fields loose)."""

    shrink_factors: tuple[int, ...] = (4, 2, 1)
    """Per-level isotropic downsample factor, coarse-to-fine."""

    smoothing_sigmas: tuple[float, ...] = (2.0, 1.0, 0.0)
    """Per-level Gaussian pre-smoothing sigma in voxels."""

    iterations: tuple[int, ...] = (100, 70, 40)
    """Per-level maximum iteration count (usually cut short by convergence)."""

    metric: str = "cc"
    """Monitoring metric used for convergence and best-state selection: 'cc', 'lpa',
    'lpc', 'pearson', 'mse'. It does *not* drive the update (the flow equation does) —
    it is the referee that decides which iterate to keep."""

    cc_radius: int = 4
    """Neighborhood half-width (voxels) for metric='cc'."""

    lpa_sigma: float = 4.0
    """Neighborhood size (voxels) for metric='lpa'/'lpc'."""

    lpa_kernel: str = "gauss"
    """Neighborhood kernel for metric='lpa'/'lpc': 'gauss' or 'box'."""

    convergence_window: int = 10
    """Trailing-window size for early stopping. <=0 disables it (run all iterations);
    the best-metric field is returned either way, so exhaustion never over-warps."""

    convergence_threshold: float = 1e-6
    """Convergence slope threshold; larger stops sooner."""

    invert_iters: int = 8
    """Fixed-point iterations for displacement-field inversion (the -save_inverse warp
    and, in diffeo mode, nothing else — the forward field is built by composition)."""

    void_guard: float = 1.0
    """Strength (0..1) of the no-data-boundary guard: how much of the void-normal
    component of each update is removed near a coverage boundary. 0 disables it.
    Only has an effect when the caller supplies coverage masks."""

    warp_flags: int = 0
    """Bit flags: 1=no-x-disp, 2=no-y-disp, 4=no-z-disp (matches qwarp/formwarp)."""

    final_qwarp: bool = False
    """After the flow levels converge, refine with the 3dQwarp engine, initialized
    from the flow field. Fine-scale pure image match on top of the flow's global fit."""

    qwarp_config: QwarpConfig | None = None
    """A :class:`~fastfuncstuff.processing.warp.QwarpConfig` for the ``final_qwarp``
    pass. Built with matching warp_flags/verb defaults if None."""

    final_interp: str = "wsinc5"
    """Interpolation kernel for the final warped output. Estimation always uses linear."""

    verb: int = 1
    """Verbosity (0=quiet, 1=normal, 2=detailed)."""


# ---------------------------------------------------------------------------
# Small primitives
# ---------------------------------------------------------------------------


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


def prep_intensity(vol: Tensor, mode: str, sigma: float, mask: Tensor | None = None) -> Tensor:
    """Intensity preprocessing that makes brightness constancy approximately true.

    - ``none``: pass through. Correct only when the two images are the same modality
      on the same intensity scale.
    - ``meanstd``: remove a global mean and scale.
    - ``localnorm``: remove a *local* mean and scale. Kills bias fields, shading and
      any spatially varying gain, which is what makes same-contrast pairs from
      different sessions or scanners registrable. It does **not** fix a contrast
      *inversion*: local z-scoring an inverted image gives the negated map, and the
      flow force then points the wrong way everywhere.
    - ``gradmag``: locally normalized gradient magnitude. Edges sit in the same place
      with the same sign in every modality, so this is the mode that survives genuine
      cross-contrast (T1 vs T2, EPI vs anat) — at the cost of throwing away everything
      except boundaries, which makes it blinder inside homogeneous tissue.
    """
    if mode == "none":
        return vol
    if mode == "gradmag":
        gx, gy, gz = _grad3(_separable_smooth_3d(vol, 1.0, kernel_type="gauss"))
        mag = (gx * gx + gy * gy + gz * gz).clamp(min=_EPS).sqrt()
        return prep_intensity(mag, "localnorm", sigma, mask)
    if mode == "meanstd":
        if mask is not None:
            m = mask > 0
            mu = vol[m].mean() if m.any() else vol.mean()
            sd = vol[m].std() if m.any() else vol.std()
        else:
            mu, sd = vol.mean(), vol.std()
        return (vol - mu) / sd.clamp(min=_EPS)
    if mode == "localnorm":
        # Weighted local statistics. An unweighted local mean/variance at the edge of
        # the mask is computed partly from whatever the mask exists to exclude -- a
        # no-data void, or air beside the brain -- so the normalization is wrong
        # exactly where the deformation is largest, and the flow force it feeds is
        # wrong with it. With no mask this is the plain local z-score as before.
        if mask is not None:
            w = mask.float()
            wn = _separable_smooth_3d(w, sigma, kernel_type="gauss").clamp(min=_EPS)
            mu = _separable_smooth_3d(w * vol, sigma, kernel_type="gauss") / wn
            var = _separable_smooth_3d(w * vol * vol, sigma, kernel_type="gauss") / wn - mu * mu
        else:
            mu = _separable_smooth_3d(vol, sigma, kernel_type="gauss")
            var = _separable_smooth_3d(vol * vol, sigma, kernel_type="gauss") - mu * mu
        return (vol - mu) / var.clamp(min=_EPS).sqrt()
    raise ValueError(f"unknown match mode {mode!r}; choose from {MATCH_MODES}")


def _compose(a: Field, b: Field) -> Field:
    """Compose displacement fields: apply ``a`` first, then ``b``."""
    hdr: dict = {}
    c = compose_warp_then_warp(NonlinearWarp(*a, hdr), NonlinearWarp(*b, hdr))
    return c.xd, c.yd, c.zd


def _exp_field(v: Field, max_norm: float = 0.5, max_squarings: int = 8) -> Field:
    """Exponentiate a stationary velocity field by scaling-and-squaring.

    ``exp(v)`` is the displacement field of the flow that ``v`` generates in unit time.
    Scale ``v`` down until its largest magnitude is below ``max_norm`` (where the
    field is trivially a diffeomorphism), then square by self-composition. This is what
    makes the diffeomorphic-demons update foldless by construction.
    """
    # max-then-sqrt, not sqrt-then-max: sqrt is monotone, so this is the same
    # scalar without a full 3-D sqrt over the velocity field. Kept as a tensor
    # op rather than math.sqrt so the sqrt still happens in the field's dtype —
    # widening to float64 first would shift `mag` by ~1e-7 relative, which is
    # enough to flip the squaring-count comparison below right at its boundary.
    mag = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2).max().sqrt().item()
    if mag < _EPS:
        return v
    n = 0
    while mag > max_norm and n < max_squarings:
        mag *= 0.5
        n += 1
    scale = 0.5**n
    f: Field = (v[0] * scale, v[1] * scale, v[2] * scale)
    for _ in range(n):
        f = _compose(f, f)
    return f


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


# ---------------------------------------------------------------------------
# Flow models: three ways to close the brightness-constancy equation
# ---------------------------------------------------------------------------


def _flow_demons(
    warped: Tensor, fixed: Tensor, grad: tuple[Tensor, Tensor, Tensor], noise: float
) -> Field:
    """Thirion demons force with the Cachier/Vercauteren regularized denominator.

    The minimum-norm solution of ``delta . g + diff = 0`` is ``-diff * g / |g|^2``, which
    blows up in flat regions. Adding ``diff^2 / K^2`` to the denominator bounds the step
    at ``K/2`` voxels and smoothly kills the force where gradient information is absent.
    """
    gx, gy, gz = grad
    diff = warped - fixed
    denom = gx * gx + gy * gy + gz * gz + (diff * diff) / (noise * noise)
    scale = -diff / denom.clamp(min=_EPS)
    return scale * gx, scale * gy, scale * gz


def _flow_lk(
    warped: Tensor,
    fixed: Tensor,
    grad: tuple[Tensor, Tensor, Tensor],
    radius: int,
    reg: float,
    weight: Tensor | None = None,
) -> Field:
    """Lucas-Kanade: per-voxel 3x3 least squares over a box neighbourhood.

    Assumes the displacement is constant across a ``(2r+1)^3`` window, giving the normal
    equations ``A delta = b`` with ``A = sum(g g^T)`` and ``b = -sum(g * diff)`` over the
    window (both computed as box sums, i.e. separable smoothing). Solved in closed form
    by cofactor inversion — a batched ``linalg.solve`` over every voxel would allocate a
    (N,3,3) system and is pure overhead for a fixed 3x3.

    ``weight`` makes it *weighted* least squares. Masking the resulting update is not
    the same thing and is not enough: a window straddling the mask edge still builds
    its structure tensor from voxels the mask excludes, so the solve is driven by a
    no-data boundary or by air. Folding the weight into the box sums means each window
    is fit only to the data it is allowed to see.
    """
    gx, gy, gz = grad
    diff = warped - fixed

    def box(v: Tensor) -> Tensor:
        return _separable_smooth_3d(v, float(radius), kernel_type="box")

    if weight is not None:
        gx, gy, gz = gx * weight, gy * weight, gz * weight

    a11, a22, a33 = box(gx * gx), box(gy * gy), box(gz * gz)
    a12, a13, a23 = box(gx * gy), box(gx * gz), box(gy * gz)
    b1, b2, b3 = -box(gx * diff), -box(gy * diff), -box(gz * diff)

    # Ridge scaled to the typical structure-tensor magnitude, so `reg` is unitless.
    lam = reg * ((a11 + a22 + a33).mean() / 3.0).clamp(min=_EPS)
    a11 = a11 + lam
    a22 = a22 + lam
    a33 = a33 + lam

    c11 = a22 * a33 - a23 * a23
    c12 = a13 * a23 - a12 * a33
    c13 = a12 * a23 - a13 * a22
    c22 = a11 * a33 - a13 * a13
    c23 = a12 * a13 - a11 * a23
    c33 = a11 * a22 - a12 * a12
    det = a11 * c11 + a12 * c12 + a13 * c13
    inv_det = 1.0 / det.clamp(min=_EPS)

    dx = (c11 * b1 + c12 * b2 + c13 * b3) * inv_det
    dy = (c12 * b1 + c22 * b2 + c23 * b3) * inv_det
    dz = (c13 * b1 + c23 * b2 + c33 * b3) * inv_det
    return dx, dy, dz


def _flow_hs(
    warped: Tensor,
    fixed: Tensor,
    grad: tuple[Tensor, Tensor, Tensor],
    alpha: float,
    n_iter: int,
) -> Field:
    """Horn-Schunck: close the flow equation with a global smoothness prior.

    Jacobi iteration of the Euler-Lagrange equations: each voxel's flow relaxes toward
    its neighbourhood average, corrected along the gradient by the residual brightness
    error. The neighbourhood average uses a small Gaussian rather than the classic
    weighted 6/12-neighbour stencil (separable, and the difference is a constant factor
    absorbed into ``alpha``).
    """
    gx, gy, gz = grad
    diff = warped - fixed
    denom = (alpha * alpha + gx * gx + gy * gy + gz * gz).clamp(min=_EPS)
    dx = torch.zeros_like(warped)
    dy = torch.zeros_like(warped)
    dz = torch.zeros_like(warped)
    for _ in range(n_iter):
        ax = _separable_smooth_3d(dx, 1.0, kernel_type="gauss")
        ay = _separable_smooth_3d(dy, 1.0, kernel_type="gauss")
        az = _separable_smooth_3d(dz, 1.0, kernel_type="gauss")
        resid = (gx * ax + gy * ay + gz * az + diff) / denom
        dx = ax - gx * resid
        dy = ay - gy * resid
        dz = az - gz * resid
    return dx, dy, dz


def _flow_update(
    warped: Tensor,
    fixed: Tensor,
    fixed_grad: tuple[Tensor, Tensor, Tensor],
    cfg: OptiwarpConfig,
    weight: Tensor | None = None,
) -> Field:
    """One optical-flow displacement increment for the current warped image."""
    grad = _grad3(warped)
    if cfg.symmetric_force and cfg.force in ("demons", "hs"):
        grad = tuple(0.5 * (g + gf) for g, gf in zip(grad, fixed_grad, strict=True))  # type: ignore[assignment]
    if cfg.force == "demons":
        return _flow_demons(warped, fixed, grad, cfg.demons_noise)
    if cfg.force == "lk":
        return _flow_lk(warped, fixed, grad, cfg.lk_radius, cfg.lk_reg, weight)
    if cfg.force == "hs":
        return _flow_hs(warped, fixed, grad, cfg.hs_alpha, cfg.hs_iters)
    raise ValueError(f"unknown force {cfg.force!r}; choose from {FORCES}")


# ---------------------------------------------------------------------------
# One pyramid level
# ---------------------------------------------------------------------------


def _optiflow_level(
    fixed: Tensor,
    moving: Tensor,
    weight: Tensor,
    fwd: Field,
    n_iter: int,
    cfg: OptiwarpConfig,
    level_tag: str = "",
    guard: tuple[Tensor, ...] | None = None,
) -> tuple[Field, float]:
    """Run up to ``n_iter`` optical-flow updates at one resolution.

    Returns the **best-metric** field seen, not the last. Like greedy SyN, demons-style
    flow overshoots: the metric falls, then wanders once the update is chasing noise.
    Snapshotting the best iterate makes "run to exhaustion" safe.
    """
    flags = cfg.warp_flags
    window = cfg.convergence_window
    fixed_grad = _grad3(fixed)
    nz, ny, nx = fixed.shape
    voxel_grid = torch.meshgrid(
        torch.arange(nz, dtype=fixed.dtype, device=fixed.device),
        torch.arange(ny, dtype=fixed.dtype, device=fixed.device),
        torch.arange(nx, dtype=fixed.dtype, device=fixed.device),
        indexing="ij",
    )

    best_cost = float("inf")
    best: Field = tuple(c.clone() for c in fwd)  # type: ignore[assignment]
    costs: list[float] = []

    bar = None
    if _tqdm is not None and cfg.verb >= 1 and n_iter >= 5:
        bar = _tqdm(total=n_iter, desc=f"optiflow {level_tag}", leave=True)

    with torch.no_grad():
        for _ in range(n_iter):
            warped = warp_image_linear(moving, fwd[0], fwd[1], fwd[2], voxel_grid=voxel_grid)

            cost_val = float(
                image_metric(
                    warped,
                    fixed,
                    weight,
                    metric=cfg.metric,
                    cc_radius=cfg.cc_radius,
                    lpa_sigma=cfg.lpa_sigma,
                    lpa_kernel=cfg.lpa_kernel,
                )
            )
            costs.append(cost_val)
            if cost_val < best_cost:
                best_cost = cost_val
                best = tuple(c.clone() for c in fwd)  # type: ignore[assignment]

            if bar is not None:
                bar.update(1)
                bar.set_postfix(cost=f"{cost_val:.5f}", best=f"{best_cost:.5f}")

            if window > 0 and len(costs) >= window:
                if _convergence_value(costs, window) < cfg.convergence_threshold:
                    break

            ux, uy, uz = _flow_update(warped, fixed, fixed_grad, cfg, weight)

            # Drive the flow only from voxels the metric cares about; an unmasked
            # background gradient otherwise pulls the field outward at the edges.
            ux, uy, uz = ux * weight, uy * weight, uz * weight

            # Fluid regularization, then step-size control.
            ux, uy, uz = _smooth_field(ux, uy, uz, cfg.update_sigma)
            # Drop whatever of the update points into a no-data region, leaving both
            # tangential directions free. After the smoothing, so the fluid step cannot
            # reintroduce a normal component from a neighbour.
            if guard is not None:
                ux, uy, uz = _apply_void_guard(ux, uy, uz, guard, cfg.void_guard)
            if flags:
                ux, uy, uz = _apply_axis_flags(ux, uy, uz, flags)
            mag = float(torch.sqrt(ux**2 + uy**2 + uz**2).max())
            if mag > _EPS and mag > cfg.max_step:
                s = cfg.max_step / mag
                ux, uy, uz = ux * s, uy * s, uz * s

            if cfg.step_mode == "diffeo":
                # Compose "update first, then the running field": the new warped image
                # is W(p + u(p)), which is exactly the compositive demons update.
                fwd = _compose(_exp_field((ux, uy, uz)), fwd)
            elif cfg.step_mode == "additive":
                fwd = (fwd[0] + ux, fwd[1] + uy, fwd[2] + uz)
            else:
                raise ValueError(f"unknown step_mode {cfg.step_mode!r}; choose from {STEP_MODES}")

            # Elastic regularization of the accumulated field.
            fwd = _smooth_field(fwd[0], fwd[1], fwd[2], cfg.total_sigma)
            if flags:
                fwd = _apply_axis_flags(fwd[0], fwd[1], fwd[2], flags)

    if bar is not None:
        bar.close()
    if cfg.verb >= 1:
        print(f"  {level_tag}: {len(costs)} iters, best cost {best_cost:.5f}")
    return best, best_cost


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


@dataclass
class OptiwarpResult:
    """Outputs of :func:`optiwarp`. All fields are (nz, ny, nx) in voxel units."""

    warped: Tensor
    """Moving image resampled into fixed space."""

    fwd: Field
    """moving->fixed total warp (use this to resample moving into fixed space)."""

    inv: Field
    """fixed->moving warp, by numerical inversion of ``fwd``."""

    jacobian: Tensor
    """Per-voxel determinant of ``I + grad(fwd)``; <=0 means a folded voxel."""

    cost: float = float("nan")
    """Best monitoring-metric value reached (lower = better)."""

    min_jacobian: float = float("nan")
    """Minimum of ``jacobian`` — the headline foldedness diagnostic."""


def optiwarp(
    fixed: Tensor,
    moving: Tensor,
    weight: Tensor | None = None,
    mask: Tensor | None = None,
    fixed_cover: Tensor | None = None,
    moving_cover: Tensor | None = None,
    config: OptiwarpConfig | None = None,
    device: torch.device | None = None,
) -> OptiwarpResult:
    """Register ``moving`` to ``fixed`` by multiresolution 3-D optical flow.

    Assumes the two images already agree in the affine sense (run ``ffs_allineate``
    first); this solves for the residual nonlinear deformation only.

    Args:
        fixed: (nz, ny, nx) fixed/target image (the ``-base``).
        moving: (nz, ny, nx) moving image to deform (the ``-source``).
        weight: Optional (nz, ny, nx) weight image. Weights both the monitoring metric
            and the flow force. Defaults to ones (or ``mask`` if given).
        mask: Optional (nz, ny, nx) mask; used as a binary weight when ``weight`` is None.
        fixed_cover: Optional (nz, ny, nx) mask of where ``fixed`` holds real data.
        moving_cover: Optional (nz, ny, nx) mask of where ``moving`` holds real data.
            Both are excluded from the metric and the flow force, *and* cross-filled
            so the void presents no edge -- which matters more here than for SyN,
            since ``-match gradmag`` is built out of edges.
        config: :class:`OptiwarpConfig`. Uses defaults if None.
        device: Torch device. Inferred from ``fixed`` if None.

    Returns:
        :class:`OptiwarpResult` with the moving->fixed warp, its inverse, the warped
        image and the Jacobian map.
    """
    cfg = config if config is not None else OptiwarpConfig()
    if device is None:
        device = fixed.device
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    fixed = fixed.float().to(device)
    moving = moving.float().to(device)
    full_shape: tuple[int, int, int] = tuple(fixed.shape)  # type: ignore[assignment]
    predicted, available = plan_nonlinear_memory(full_shape, device, "optiwarp")
    if predicted > available and cfg.verb >= 1:
        print(
            f"WARNING: estimated optical-flow peak {predicted / 2**30:.1f} GiB exceeds "
            f"the {available / 2**30:.1f} GiB safe memory budget; use stronger shrink factors."
        )

    if weight is not None:
        weight = weight.float().to(device)
    if mask is not None:
        mask = mask.float().to(device)

    # Exclude each image's no-data region and cross-fill it from the other, so the
    # void presents no edge for the flow to chase. Metric/force only -- ``moving`` is
    # untouched, so the warped output keeps honest zeros. See mask.cross_fill_no_data.
    f_metric, m_metric, cover = cross_fill_no_data(fixed, moving, fixed_cover, moving_cover)
    if cover is not None:
        if weight is not None:
            weight = weight * cover.float()
        else:
            mask = cover.float() if mask is None else mask * cover.float()

    n_levels = len(cfg.shrink_factors)
    if not (len(cfg.smoothing_sigmas) == len(cfg.iterations) == n_levels):
        raise ValueError("shrink_factors, smoothing_sigmas and iterations must have equal length")

    fwd: Field = (
        torch.zeros(full_shape, device=device),
        torch.zeros(full_shape, device=device),
        torch.zeros(full_shape, device=device),
    )
    best_cost = float("nan")

    for lev in range(n_levels):
        factor = cfg.shrink_factors[lev]
        sigma = cfg.smoothing_sigmas[lev]
        n_iter = cfg.iterations[lev]
        target = _shrunk_shape(full_shape, factor)

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

        # Intensity prep is per-level: the local-norm neighbourhood must scale with the
        # grid, and prepping the full-res image once would blur differently per level.
        m_bin = (w_lvl > 0.5) if (weight is not None or mask is not None) else None
        f_prep = prep_intensity(f_lvl, cfg.match, cfg.match_sigma, m_bin)
        m_prep = prep_intensity(m_lvl, cfg.match, cfg.match_sigma, m_bin)

        fwd = _resize_field(*fwd, target)

        if cfg.verb >= 1:
            print(
                f"optiwarp: level {lev + 1}/{n_levels} shrink={factor} "
                f"smooth={sigma:g}vox grid={target[2]}x{target[1]}x{target[0]} "
                f"iters={n_iter} force={cfg.force} match={cfg.match}"
            )

        # Rebuilt per level: the normal is grid-relative, so it must be measured on
        # the grid the updates live on.
        guard = None
        if cover is not None and cfg.void_guard > 0.0:
            guard = _void_guard_field(_resize_volume(cover.float(), target))

        fwd, best_cost = _optiflow_level(
            f_prep, m_prep, w_lvl, fwd, n_iter, cfg, level_tag=f"L{lev + 1}", guard=guard
        )

    fwd = _resize_field(*fwd, full_shape)

    if cfg.final_qwarp:
        fwd = _run_final_qwarp(fixed, moving, weight, mask, fwd, cfg, device)

    inv = invert_displacement_field(*fwd, cfg.invert_iters)
    jac = jacobian_determinant(*fwd)
    min_jac = float(jac.min())
    if cfg.verb >= 1:
        n_fold = int((jac <= 0).sum())
        print(
            f"optiwarp: min Jacobian {min_jac:.4f}"
            + (f", {n_fold} folded voxels" if n_fold else ", no folding")
        )

    warped = warp_image(moving, fwd[0], fwd[1], fwd[2], mode=cfg.final_interp)
    return OptiwarpResult(
        warped=warped, fwd=fwd, inv=inv, jacobian=jac, cost=best_cost, min_jacobian=min_jac
    )


def _run_final_qwarp(
    fixed: Tensor,
    moving: Tensor,
    weight: Tensor | None,
    mask: Tensor | None,
    fwd: Field,
    cfg: OptiwarpConfig,
    device: torch.device,
) -> Field:
    """Polish the flow field with the 3dQwarp engine, initialized from it.

    Optical flow finds the deformation; patch optimization against the actual image
    cost nails the residual fraction of a voxel that the smoothness prior smeared out.
    Run with ``pad=False`` so the returned warp stays on the base grid and composes
    with everything else this module produces.

    ``reject_worse_levels`` defaults **on** here, unlike a standalone qwarp run. Coming
    in from a converged flow field the fit is already good, so the finest patch levels
    have little left to win and can lose: measured on a phantom, the polish walked the
    global cost from -0.999 back to -0.969 across the last two levels. Rejecting a level
    that degrades the fit is what makes the hand-off a strict improvement.
    """
    from .warp import QwarpConfig, qwarp

    qcfg = cfg.qwarp_config
    if qcfg is None:
        qcfg = QwarpConfig(
            warp_flags=cfg.warp_flags,
            verb=cfg.verb,
            final_interp=cfg.final_interp,
            reject_worse_levels=True,
        )
    if cfg.verb >= 1:
        print(f"optiwarp: handing off to the qwarp engine (minpatch={qcfg.minpatch})")

    _, qxd, qyd, qzd = qwarp(
        fixed,
        moving,
        weight=weight,
        mask=(mask.byte() if mask is not None else None),
        initial_warp=fwd,
        config=qcfg,
        device=device,
        pad=False,
    )
    return qxd, qyd, qzd
