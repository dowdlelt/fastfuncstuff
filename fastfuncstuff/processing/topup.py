"""FSL-topup-style susceptibility distortion correction on the GPU.

A third distortion-correction engine alongside the AFNI-3dQwarp
(:mod:`fastfuncstuff.processing.warp`) and ANTs-SyN
(:mod:`fastfuncstuff.processing.formwarp`) backends. This one is a faithful port of
the *logic* of FSL's ``topup`` (Andersson et al., 2003): the groupwise "blip-up /
blip-down" model that estimates a single off-resonance field from images acquired
with opposing phase-encode polarity.

Model (see ``../fmri_wiki/concepts/topup.md`` and FSL ``topup_costfunctions.cpp``):

  One shared scalar field ``b(x)`` in Hz, parameterised as a cubic B-spline on a
  regular knot grid whose spacing (``warpres``, mm) is the dominant regulariser.
  For scan ``s`` with phase-encode axis ``pe`` and signed polarity ``sign_s`` and
  total readout time ``r_s`` seconds, the displacement along PE is (in voxels):

      d_s(x) = b(x) * r_s * sign_s

  Each acquired (distorted) scan is resampled at ``x + d_s`` and multiplied by the
  Jacobian ``1 + dd_s/d(pe)`` (intensity modulation: compression brightens). Because
  opposite polarities give ``d_1 = -d_2``, the two resampled+modulated estimates
  agree with each other (and with their running mean) only at the true field.

  Cost = mean over scans of the masked SSD to the mean, plus a bending-energy
  penalty ``lambda * integral (second derivatives of b)^2``. With ``ssqlambda`` the
  penalty weight is scaled by the current SSD so it is intensity-scale invariant.

Optimiser: Gauss-Newton least squares on the spline coefficients. The residual
folds both the data term and ``sqrt(lambda)`` times the bending operator, so a plain
GN normal-equation solve (matrix-free conjugate gradient on ``J^T J`` via
forward/reverse autodiff) with a backtracking, step-rejecting line search does the
Tikhonov-regularised minimisation. This mirrors the Gauss-Newton warp solver in
:mod:`fastfuncstuff.processing.segment` (the pattern that behaves far better here than
a first-order LBFGS, which tends to overshoot). Multi-resolution: a coarse-to-fine
schedule over knot spacing, data smoothing (FWHM) and penalty weight (the FSL
``b02b0`` schedule by default).

Movement estimation is intentionally deferred (see module docstring in the CLI); the
only motion modelled is an optional single global translation along PE
(``-pe_shift``), the tiny scanner-induced shift possible between back-to-back scans.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field as _dc_field

import torch
import torch.nn.functional as F
from torch import Tensor

from .cost import _separable_smooth_3d

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover - tqdm is a hard dep in practice
    _tqdm = None


def _bar(iterable=None, **kw):
    """tqdm wrapper that degrades to a no-op when tqdm is missing or disabled."""
    if _tqdm is None or kw.get("disable"):
        return iterable if iterable is not None else _NullBar()
    return _tqdm(iterable, **kw)


class _NullBar:
    """Minimal stand-in so bar.update/set_postfix/close are always callable."""

    def update(self, *_): ...
    def set_postfix_str(self, *_a, **_k): ...
    def set_description(self, *_a, **_k): ...
    def close(self): ...
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

# NIfTI spatial axis (0=x/i, 1=y/j, 2=z/k) -> tensor dim in an (nz, ny, nx) volume.
_NIFTI_AXIS_TO_TDIM = {0: 2, 1: 1, 2: 0}


# ----------------------------------------------------------------------------
# Configuration / multi-resolution schedule
# ----------------------------------------------------------------------------
@dataclass
class TopupConfig:
    """Multi-resolution schedule and solver settings.

    The list-valued fields define the per-level schedule and must share a length
    (the number of levels). Defaults reproduce FSL's ``b02b0_1.cnf`` (no
    subsampling — the GPU makes it cheap to work full-res throughout).
    """

    warpres: list[float] = _dc_field(default_factory=lambda: [20, 16, 14, 12, 10, 6, 4, 4, 4])
    """Knot spacing (mm) of the B-spline field per level."""

    fwhm: list[float] = _dc_field(default_factory=lambda: [8, 6, 4, 3, 3, 2, 1, 0, 0])
    """FWHM (mm) of Gaussian smoothing applied to the data per level."""

    lam: list[float] = _dc_field(
        default_factory=lambda: [
            5e-4,
            1e-4,
            1e-5,
            1.5e-6,
            5e-7,
            5e-7,
            5e-8,
            5e-10,
            1e-11,
        ]
    )
    """Bending-energy weight per level (before the optional ssq scaling)."""

    miter: list[int] = _dc_field(default_factory=lambda: [5, 5, 5, 5, 5, 10, 10, 20, 20])
    """Max Gauss-Newton iterations per level."""

    subsamp: list[int] = _dc_field(default_factory=lambda: [1, 1, 1, 1, 1, 1, 1, 1, 1])
    """Integer subsampling factor per level (1 = full resolution)."""

    ssqlambda: bool = True
    """If set, scale ``lam`` by the current mean squared difference each iteration."""

    reg_mode: str = "bending"
    """Regularisation model: ``"bending"`` (energy of 2nd derivs) or ``"membrane"``."""

    cg_iters: int = 50
    """Max conjugate-gradient iterations per Gauss-Newton step."""

    cg_tol: float = 1e-4
    """Relative residual tolerance for the CG solve."""

    analytic_gn: bool = True
    """Use the analytic Gauss-Newton matvec (fast). ``False`` = reverse-mode autodiff."""

    def n_levels(self) -> int:
        return len(self.warpres)

    def validate(self) -> None:
        n = self.n_levels()
        for name in ("warpres", "fwhm", "lam", "miter", "subsamp"):
            v = getattr(self, name)
            if len(v) != n:
                raise ValueError(
                    f"TopupConfig.{name} has length {len(v)}, expected {n} "
                    "(all schedule lists must match warpres)"
                )
        if self.reg_mode not in ("bending", "membrane"):
            raise ValueError(f"reg_mode must be 'bending' or 'membrane', got {self.reg_mode}")


# ----------------------------------------------------------------------------
# Cubic B-spline field
# ----------------------------------------------------------------------------
def _bspline3(t: Tensor) -> Tensor:
    """Cubic B-spline basis beta^3(t), nonzero for |t| < 2."""
    a = t.abs()
    out = torch.zeros_like(a)
    m1 = a < 1.0
    m2 = (a >= 1.0) & (a < 2.0)
    out[m1] = 2.0 / 3.0 - a[m1] ** 2 + 0.5 * a[m1] ** 3
    out[m2] = (2.0 - a[m2]) ** 3 / 6.0
    return out


def _bspline_axis_matrix(
    n_vox: int, knot_spacing_vox: float, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """(n_vox, n_knots) cubic B-spline basis for one axis.

    Knots are placed on a regular grid of spacing ``knot_spacing_vox`` voxels with
    one knot before 0 and enough beyond ``n_vox-1`` that the cubic support (+/-2
    knots) covers the whole domain. Each voxel row has at most four nonzeros.
    """
    h = float(knot_spacing_vox)
    # Knot positions p_j = -h + j*h ; need last position >= (n-1) + h.
    n_knots = int(math.ceil((n_vox - 1) / h)) + 3
    j = torch.arange(n_knots, device=device, dtype=dtype)
    p = -h + j * h  # (n_knots,)
    x = torch.arange(n_vox, device=device, dtype=dtype)  # (n_vox,)
    t = (x[:, None] - p[None, :]) / h  # (n_vox, n_knots)
    return _bspline3(t)


@dataclass
class SplineFieldBasis:
    """Separable cubic B-spline basis over a volume, for one knot spacing."""

    Bz: Tensor  # (nz, n_kz)
    By: Tensor  # (ny, n_ky)
    Bx: Tensor  # (nx, n_kx)

    @property
    def coeff_shape(self) -> tuple[int, int, int]:
        return (self.Bz.shape[1], self.By.shape[1], self.Bx.shape[1])

    @property
    def n_coeff(self) -> int:
        nz, ny, nx = self.coeff_shape
        return nz * ny * nx

    def field(self, coeff: Tensor) -> Tensor:
        """Expand coefficient grid (n_kz, n_ky, n_kx) -> dense field (nz, ny, nx)."""
        # Sequential separable contraction.
        t = torch.einsum("Zc,cba->Zba", self.Bz, coeff)  # (nz, n_ky, n_kx)
        t = torch.einsum("Yb,Zba->ZYa", self.By, t)  # (nz, ny, n_kx)
        t = torch.einsum("Xa,ZYa->ZYX", self.Bx, t)  # (nz, ny, nx)
        return t

    def field_adjoint(self, g: Tensor) -> Tensor:
        """Adjoint of :meth:`field`: map a dense field (nz,ny,nx) -> coeff grid.

        Exact transpose of the three separable contractions, applied in reverse
        order. Needed by the analytic Gauss-Newton matvec (``basis.field`` is the
        linear ``B`` operator from coefficients to the field; this is ``B^T``).
        """
        t = torch.einsum("Xa,ZYX->ZYa", self.Bx, g)  # contract nx -> (nz, ny, n_kx)
        t = torch.einsum("Yb,ZYa->Zba", self.By, t)  # contract ny -> (nz, n_ky, n_kx)
        t = torch.einsum("Zc,Zba->cba", self.Bz, t)  # contract nz -> (n_kz, n_ky, n_kx)
        return t


def build_spline_basis(
    shape: tuple[int, int, int],
    voxel_sizes: tuple[float, float, float],
    warpres_mm: float,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> SplineFieldBasis:
    """Build a separable B-spline basis for a given knot spacing (mm).

    ``shape`` is (nz, ny, nx); ``voxel_sizes`` is (vz, vy, vx) in mm.
    """
    nz, ny, nx = shape
    vz, vy, vx = voxel_sizes
    Bz = _bspline_axis_matrix(nz, warpres_mm / vz, device, dtype)
    By = _bspline_axis_matrix(ny, warpres_mm / vy, device, dtype)
    Bx = _bspline_axis_matrix(nx, warpres_mm / vx, device, dtype)
    return SplineFieldBasis(Bz=Bz, By=By, Bx=Bx)


def refit_coeff(field: Tensor, basis: SplineFieldBasis) -> Tensor:
    """Least-squares fit of coefficients so ``basis.field(coeff) ~= field``.

    Separable: solve the small 1-D normal equations per axis. Used to warm-start a
    finer knot grid from the field estimated at the previous (coarser) level.
    """

    def _axis_solve(B: Tensor, x: Tensor, axis: int) -> Tensor:
        # Solve min_C || B C - x ||^2 along one axis via the normal equations.
        BtB = B.T @ B
        # Small ridge keeps the coarse-knot normal equations SPD.
        eye = torch.eye(BtB.shape[0], device=B.device, dtype=B.dtype)
        BtB = BtB + 1e-6 * eye
        xm = x.movedim(axis, -1)
        rhs = torch.einsum("vk,...v->...k", B, xm)
        sol = torch.linalg.solve(BtB, rhs.movedim(-1, -1).unsqueeze(-1)).squeeze(-1)
        return sol.movedim(-1, axis)

    c = _axis_solve(basis.Bz, field, 0)  # (n_kz, ny, nx)
    c = _axis_solve(basis.By, c, 1)  # (n_kz, n_ky, nx)
    c = _axis_solve(basis.Bx, c, 2)  # (n_kz, n_ky, n_kx)
    return c


# ----------------------------------------------------------------------------
# Forward model: field -> resampled + modulated scans, mean, mask
# ----------------------------------------------------------------------------
def _interp_last_axis(vol: Tensor, coord: Tensor) -> Tensor:
    """Linear sample of ``vol`` along its last axis at fractional ``coord`` (border).

    Uses gather (not grid_sample) so the whole forward model is double-
    differentiable — the Gauss-Newton solver needs reverse-over-reverse autodiff,
    which ``grid_sampler_3d`` does not support. Out-of-range samples clamp to edge.
    """
    n = vol.shape[-1]
    c = coord.clamp(0, n - 1)
    lo = torch.floor(c).long()
    hi = (lo + 1).clamp(max=n - 1)
    frac = c - lo.to(c.dtype)
    f_lo = torch.gather(vol, -1, lo)
    f_hi = torch.gather(vol, -1, hi)
    return f_lo * (1 - frac) + f_hi * frac


def _resample_pe(vol: Tensor, disp_vox: Tensor, pe_tdim: int) -> Tensor:
    """Resample ``vol`` (nz,ny,nx) at ``index + disp`` along the PE tensor dim.

    Displacement is purely 1-D along PE, so this is a 1-D linear interpolation on
    the PE axis. ``pe_tdim`` is the tensor dim of the PE axis (0=z, 1=y, 2=x).
    """
    v = vol.movedim(pe_tdim, -1)
    d = disp_vox.movedim(pe_tdim, -1)
    n = v.shape[-1]
    idx = torch.arange(n, device=v.device, dtype=v.dtype).expand_as(v)
    out = _interp_last_axis(v, idx + d)
    return out.movedim(-1, pe_tdim)


def _resample_pe_with_slope(
    vol: Tensor, disp_vox: Tensor, pe_tdim: int
) -> tuple[Tensor, Tensor]:
    """Like :func:`_resample_pe` but also return d(resample)/d(disp) at the current disp.

    For linear interpolation the sample is ``f_lo*(1-frac) + f_hi*frac`` and its
    derivative w.r.t. the (fractional) displacement is ``f_hi - f_lo`` with the
    integer ``lo/hi`` indices held fixed — i.e. the local PE slope of the volume
    sampled at the *displaced* location. This is exactly the Gauss-Newton
    linearisation of the resample (we differentiate the interpolation weight, not
    the index selection), and it matches the autograd derivative of
    :func:`_interp_last_axis`: outside ``[0, n-1]`` the clamp zeroes the gradient, so
    the slope is zeroed there too.
    """
    v = vol.movedim(pe_tdim, -1)
    d = disp_vox.movedim(pe_tdim, -1)
    n = v.shape[-1]
    idx = torch.arange(n, device=v.device, dtype=v.dtype).expand_as(v)
    raw = idx + d
    c = raw.clamp(0, n - 1)
    lo = torch.floor(c).long()
    hi = (lo + 1).clamp(max=n - 1)
    frac = c - lo.to(c.dtype)
    f_lo = torch.gather(v, -1, lo)
    f_hi = torch.gather(v, -1, hi)
    res = f_lo * (1 - frac) + f_hi * frac
    in_bounds = (raw >= 0) & (raw <= n - 1)
    slope = (f_hi - f_lo) * in_bounds.to(v.dtype)
    return res.movedim(-1, pe_tdim), slope.movedim(-1, pe_tdim)


def _central_diff_pe(vol: Tensor, pe_tdim: int) -> Tensor:
    """Central difference along the PE tensor dim (the operator inside the Jacobian)."""
    from .penalty import _central_diff_batched

    return _central_diff_batched(vol, dim=pe_tdim)


def _central_diff_pe_adjoint(g: Tensor, pe_tdim: int) -> Tensor:
    """Adjoint of :func:`_central_diff_pe` (transpose of the boundary-aware stencil).

    Forward stencil (per :func:`penalty._central_diff_batched`): forward diff at the
    first plane, central diff in the interior, backward diff at the last plane. This
    scatters each output row's contribution back onto the input rows it touched.
    """
    x = g.movedim(pe_tdim, -1)
    n = x.shape[-1]
    out = torch.zeros_like(x)
    if n < 2:
        return out.movedim(-1, pe_tdim)
    # Boundary i=0: result[0] = v[1] - v[0]
    out[..., 0] += -x[..., 0]
    out[..., 1] += x[..., 0]
    # Boundary i=n-1: result[n-1] = v[n-1] - v[n-2]
    out[..., n - 1] += x[..., n - 1]
    out[..., n - 2] += -x[..., n - 1]
    # Interior i in 1..n-2: result[i] = 0.5*(v[i+1] - v[i-1])
    if n >= 3:
        gi = 0.5 * x[..., 1 : n - 1]
        out[..., 2:n] += gi
        out[..., 0 : n - 2] += -gi
    return out.movedim(-1, pe_tdim)


def _jacobian_pe(disp_vox: Tensor, pe_tdim: int) -> Tensor:
    """Jacobian 1 + d(disp)/d(index) along the PE tensor dim (central differences)."""
    from .penalty import _central_diff_batched

    # _central_diff_batched maps dim 0/1/2 -> z/y/x tensor dims; PE tensor dim maps
    # directly (0=z,1=y,2=x) to that convention's spatial dim argument.
    dim_arg = {0: 0, 1: 1, 2: 2}[pe_tdim]
    return 1.0 + _central_diff_batched(disp_vox, dim=dim_arg)


@dataclass
class ScanSpec:
    """One input image plus its acquisition geometry."""

    data: Tensor  # (nz, ny, nx) float
    pe_axis: int  # NIfTI spatial axis 0/1/2
    sign: float  # +1 / -1 polarity along that axis
    readout: float  # total readout time (s)


def forward_scans(
    coeff: Tensor,
    basis: SplineFieldBasis,
    scans: list[ScanSpec],
) -> tuple[list[Tensor], Tensor]:
    """Return (modulated resampled scans, mean) for the current coefficients.

    ``field`` (Hz) = basis.field(coeff). Per scan the PE displacement in voxels is
    ``field * readout * sign`` and the modulated estimate is
    ``resample(scan, +disp) * (1 + d disp/d pe)``.
    """
    field_hz = basis.field(coeff)
    modulated: list[Tensor] = []
    for sc in scans:
        pe_tdim = _NIFTI_AXIS_TO_TDIM[sc.pe_axis]
        disp = field_hz * (sc.readout * sc.sign)
        res = _resample_pe(sc.data, disp, pe_tdim)
        jac = _jacobian_pe(disp, pe_tdim)
        modulated.append(res * jac)
    mean = torch.stack(modulated, dim=0).mean(dim=0)
    return modulated, mean


def compute_mask(scans: list[ScanSpec]) -> Tensor:
    """Intersection mask of finite/positive data, with PE-axis edge planes zeroed.

    Mirrors topup's trick of excluding a one-voxel frame in the non-PE and PE
    directions so small edge effects don't dominate the SSD.
    """
    m = torch.ones_like(scans[0].data, dtype=torch.bool)
    for sc in scans:
        m &= torch.isfinite(sc.data) & (sc.data > (sc.data.mean() * 1e-3))
    # Zero the outer plane on every axis (cheap, symmetric version of topup's frame).
    m[0, :, :] = False
    m[-1, :, :] = False
    m[:, 0, :] = False
    m[:, -1, :] = False
    m[:, :, 0] = False
    m[:, :, -1] = False
    return m


def taper_field_to_object(
    field_hz: Tensor,
    scans: list[ScanSpec],
    voxel_sizes: tuple[float, float, float],
    dilate_mm: float = 20.0,
    rolloff_mm: float = 8.0,
) -> Tensor:
    """Roll the field off to zero well outside the imaged object.

    The B-spline field is only constrained where the SSD mask has signal; beyond the
    object it smoothly *extrapolates* the (often large) peri-object field into air.
    That is harmless for the images but makes the saved warp carry a big displacement
    in empty space -- which, when applied through :mod:`ffs_nwarp`, border-clamps into
    an auto-pad margin and replicates real tissue as "mirror" ghosts past the edge, and
    also over-triggers the auto-pad grid growth.

    We keep the field untouched over a generously dilated object mask (real peri-object
    susceptibility distortion lives right at the tissue edge, so the dilation must clear
    it) and taper it smoothly to zero over ``rolloff_mm`` beyond that. Far air -> 0
    displacement -> the warp is identity there and samples out-of-FOV source as clean
    background. Inside/near the object nothing changes.
    """
    mean_mag = torch.stack([sc.data.abs() for sc in scans], dim=0).mean(dim=0)
    pos = mean_mag[mean_mag > 0]
    thr = 0.05 * float(pos.mean()) if pos.numel() else 0.0
    mask = (mean_mag > thr).to(field_hz.dtype)

    mean_vox = sum(voxel_sizes) / 3.0
    dil = int(round(dilate_mm / mean_vox))
    if dil > 0:
        mask = F.max_pool3d(mask[None, None], kernel_size=2 * dil + 1, stride=1, padding=dil)[0, 0]
    sigma_vox = rolloff_mm / mean_vox
    if sigma_vox > 0:
        mask = _separable_smooth_3d(mask.float(), sigma_vox).clamp(0.0, 1.0).to(field_hz.dtype)
    return field_hz * mask


# ----------------------------------------------------------------------------
# Regularisation (bending / membrane energy on the dense field)
# ----------------------------------------------------------------------------
def _second_diff(vol: Tensor, tdim: int) -> Tensor:
    """Second difference [1,-2,1] along a tensor dim, zero at the two boundaries."""
    n = vol.shape[tdim]
    out = torch.zeros_like(vol)
    if n < 3:
        return out
    lo = vol.narrow(tdim, 0, n - 2)
    mid = vol.narrow(tdim, 1, n - 2)
    hi = vol.narrow(tdim, 2, n - 2)
    out.narrow(tdim, 1, n - 2).copy_(hi - 2 * mid + lo)
    return out


def _first_diff(vol: Tensor, tdim: int) -> Tensor:
    n = vol.shape[tdim]
    out = torch.zeros_like(vol)
    if n < 2:
        return out
    out.narrow(tdim, 0, n - 1).copy_(vol.narrow(tdim, 1, n - 1) - vol.narrow(tdim, 0, n - 1))
    return out


def reg_residual(field_hz: Tensor, reg_mode: str) -> Tensor:
    """Flatten the regularisation operator applied to the field into residual rows.

    Bending energy = sum of squared 2nd derivatives (diagonal + sqrt(2) cross
    terms); membrane energy = sum of squared 1st derivatives. Returned as a 1-D
    vector so ``||reg_residual||^2`` equals the (unweighted) energy.
    """
    if reg_mode == "membrane":
        parts = [_first_diff(field_hz, d) for d in (0, 1, 2)]
    else:  # bending
        s2 = math.sqrt(2.0)
        parts = [_second_diff(field_hz, d) for d in (0, 1, 2)]
        # cross terms d^2/dxdy etc, weighted by sqrt(2) so squared -> factor 2.
        parts += [
            s2 * _first_diff(_first_diff(field_hz, 0), 1),
            s2 * _first_diff(_first_diff(field_hz, 0), 2),
            s2 * _first_diff(_first_diff(field_hz, 1), 2),
        ]
    return torch.cat([p.reshape(-1) for p in parts])


def _first_diff_adjoint(g: Tensor, tdim: int) -> Tensor:
    """Adjoint of :func:`_first_diff` (forward diff, last row zero)."""
    x = g.movedim(tdim, -1)
    n = x.shape[-1]
    out = torch.zeros_like(x)
    if n < 2:
        return out.movedim(-1, tdim)
    # result[i] = v[i+1] - v[i] for i in 0..n-2 (result[n-1] = 0, so x[n-1] unused).
    gi = x[..., 0 : n - 1]
    out[..., 0 : n - 1] += -gi
    out[..., 1:n] += gi
    return out.movedim(-1, tdim)


def _second_diff_adjoint(g: Tensor, tdim: int) -> Tensor:
    """Adjoint of :func:`_second_diff` (Laplacian stencil, boundary rows zero)."""
    x = g.movedim(tdim, -1)
    n = x.shape[-1]
    out = torch.zeros_like(x)
    if n < 3:
        return out.movedim(-1, tdim)
    # result[i] = v[i-1] - 2 v[i] + v[i+1] for i in 1..n-2 (rows 0 and n-1 zero).
    gi = x[..., 1 : n - 1]
    out[..., 0 : n - 2] += gi
    out[..., 1 : n - 1] += -2.0 * gi
    out[..., 2:n] += gi
    return out.movedim(-1, tdim)


def reg_residual_adjoint(u: Tensor, shape: tuple[int, int, int], reg_mode: str) -> Tensor:
    """Adjoint of :func:`reg_residual`: map reg-rows back onto a field (nz,ny,nx).

    Splits ``u`` into the same per-operator chunks :func:`reg_residual` produced and
    applies each operator's transpose, summing into the field. For the cross terms
    the composition transposes in reverse order.
    """
    n = int(shape[0] * shape[1] * shape[2])
    g = torch.zeros(shape, device=u.device, dtype=u.dtype)

    def chunk(i: int) -> Tensor:
        return u[i * n : (i + 1) * n].reshape(shape)

    if reg_mode == "membrane":
        for d in (0, 1, 2):
            g = g + _first_diff_adjoint(chunk(d), d)
        return g
    # bending
    s2 = math.sqrt(2.0)
    for d in (0, 1, 2):
        g = g + _second_diff_adjoint(chunk(d), d)
    # cross terms: forward was s2 * D1_{d1}(D1_{d0}(f)); adjoint = s2 * D1_{d0}^T D1_{d1}^T.
    g = g + s2 * _first_diff_adjoint(_first_diff_adjoint(chunk(3), 1), 0)
    g = g + s2 * _first_diff_adjoint(_first_diff_adjoint(chunk(4), 2), 0)
    g = g + s2 * _first_diff_adjoint(_first_diff_adjoint(chunk(5), 2), 1)
    return g


# ----------------------------------------------------------------------------
# Gauss-Newton least-squares solver (one level)
# ----------------------------------------------------------------------------
def _residual_vector(
    coeff: Tensor,
    basis: SplineFieldBasis,
    scans: list[ScanSpec],
    mask_idx: Tensor,
    n_scans_minus_1: int,
    lam_eff: float,
    reg_mode: str,
) -> Tensor:
    """Full residual: masked data rows (scaled) followed by sqrt(lam) reg rows."""
    field_hz = basis.field(coeff)
    modulated: list[Tensor] = []
    for sc in scans:
        pe_tdim = _NIFTI_AXIS_TO_TDIM[sc.pe_axis]
        disp = field_hz * (sc.readout * sc.sign)
        res = _resample_pe(sc.data, disp, pe_tdim)
        jac = _jacobian_pe(disp, pe_tdim)
        modulated.append(res * jac)
    mean = torch.stack(modulated, dim=0).mean(dim=0)
    scale = 1.0 / math.sqrt(max(1, mask_idx.numel()) * max(1, n_scans_minus_1))
    data_rows = [(m - mean).reshape(-1)[mask_idx] * scale for m in modulated]
    rows = data_rows
    if lam_eff > 0:
        rows = rows + [math.sqrt(lam_eff) * reg_residual(field_hz, reg_mode)]
    return torch.cat(rows)


def _dot64(a: Tensor, b: Tensor) -> Tensor:
    """Inner product accumulated in float64 regardless of operand dtype.

    The residual/coefficient vectors run in float32 for GPU speed, but they are large
    (millions of masked voxels), and a float32 reduction loses precision. Accumulating
    the sum in float64 keeps the CG scalars (and the cost) accurate at negligible cost.
    """
    return (a * b).sum(dtype=torch.float64)


def _cg_solve(matvec, b: Tensor, max_iter: int, tol: float) -> Tensor:
    """Conjugate gradient for the SPD system ``matvec(x) = b`` (matrix-free).

    Vectors stay in ``b``'s dtype (float32 hot path); the scalar inner products
    accumulate in float64 (:func:`_dot64`) so the recurrence stays stable.
    """
    dt = b.dtype
    x = torch.zeros_like(b)
    r = b - matvec(x)
    p = r.clone()
    rs = _dot64(r, r)
    rs0 = rs
    if rs0 <= 0:
        return x
    for _ in range(max_iter):
        Ap = matvec(p)
        alpha = rs / _dot64(p, Ap).clamp_min(1e-30)
        x = x + alpha.to(dt) * p
        r = r - alpha.to(dt) * Ap
        rs_new = _dot64(r, r)
        if rs_new <= tol * tol * rs0:
            break
        p = r + (rs_new / rs).to(dt) * p
        rs = rs_new
    return x


def _gn_direction(coeff, residual, lam_eff, cg_iters, cg_tol):
    """One Gauss-Newton step direction (solve J^T J delta = -J^T r) and current cost.

    Matrix-free with reverse-mode autodiff only (grid-free 1-D resampling still has no
    forward-AD rule via the shared gather path): ``J v`` comes from a double-backward
    through ``J^T w``, ``J^T u`` from a plain vjp. Kept as a module-level helper so the
    autodiff closures never capture an outer loop variable.
    """
    c = coeff.detach().requires_grad_(True)
    r0 = residual(c, lam_eff)
    cost = float(_dot64(r0.detach(), r0.detach()))

    # Persistent graphs: r0 -> c (for vjp), and J^T w -> w (for jvp via 2x reverse).
    w = torch.zeros_like(r0, requires_grad=True)
    (jtw,) = torch.autograd.grad(r0, c, grad_outputs=w, create_graph=True, retain_graph=True)

    def vjp(u: Tensor) -> Tensor:  # J^T u, shape of c
        return torch.autograd.grad(r0, c, grad_outputs=u, retain_graph=True)[0]

    def jvp(v: Tensor) -> Tensor:  # J v, shape of r
        return torch.autograd.grad(jtw, w, grad_outputs=v, retain_graph=True)[0]

    def matvec(v: Tensor) -> Tensor:
        jv = jvp(v.reshape(coeff.shape))
        return vjp(jv).reshape(-1) + 1e-8 * v

    g = vjp(r0.detach()).reshape(-1)  # J^T r
    delta = _cg_solve(matvec, -g, cg_iters, cg_tol)
    return delta, cost


@dataclass
class _Linearization:
    """Frozen forward-model linearisation about the current coefficients.

    Holds everything the analytic ``J``/``J^T`` need so each CG matvec is pure
    einsum + elementwise + finite-difference (no autograd, no gather-scatter). The
    per-scan data-row map is ``dm_s = a_s * (P_s * dfield + Q_s * D_pe(dfield))``
    where ``P_s = slope_s * jac_s`` and ``Q_s = res_s`` (see :func:`_linearize`).
    """

    basis: SplineFieldBasis
    shape: tuple[int, int, int]
    mask_idx: Tensor
    scale: float
    lam_eff: float
    reg_mode: str
    a: list[float]
    pe_tdim: list[int]
    P: list[Tensor]
    Q: list[Tensor]


def _linearize(
    coeff: Tensor,
    basis: SplineFieldBasis,
    scans: list[ScanSpec],
    mask_idx: Tensor,
    n_scans_minus_1: int,
    lam_eff: float,
    reg_mode: str,
) -> _Linearization:
    """Precompute the per-scan slope/Jacobian maps for the analytic GN matvec."""
    field0 = basis.field(coeff)
    scale = 1.0 / math.sqrt(max(1, mask_idx.numel()) * max(1, n_scans_minus_1))
    a: list[float] = []
    tdims: list[int] = []
    P: list[Tensor] = []
    Q: list[Tensor] = []
    for sc in scans:
        pe_tdim = _NIFTI_AXIS_TO_TDIM[sc.pe_axis]
        a_s = sc.readout * sc.sign
        disp = field0 * a_s
        res, slope = _resample_pe_with_slope(sc.data, disp, pe_tdim)
        jac = _jacobian_pe(disp, pe_tdim)
        a.append(a_s)
        tdims.append(pe_tdim)
        P.append(slope * jac)
        Q.append(res)
    return _Linearization(
        basis=basis,
        shape=tuple(field0.shape),  # type: ignore[arg-type]
        mask_idx=mask_idx,
        scale=scale,
        lam_eff=lam_eff,
        reg_mode=reg_mode,
        a=a,
        pe_tdim=tdims,
        P=P,
        Q=Q,
    )


def _lin_jv(lin: _Linearization, dc: Tensor) -> Tensor:
    """Analytic ``J v``: coefficient perturbation -> residual perturbation."""
    dfield = lin.basis.field(dc)
    dms = [
        lin.a[s] * (lin.P[s] * dfield + lin.Q[s] * _central_diff_pe(dfield, lin.pe_tdim[s]))
        for s in range(len(lin.P))
    ]
    dmean = torch.stack(dms, dim=0).mean(dim=0)
    rows = [(dm - dmean).reshape(-1)[lin.mask_idx] * lin.scale for dm in dms]
    if lin.lam_eff > 0:
        rows.append(math.sqrt(lin.lam_eff) * reg_residual(dfield, lin.reg_mode))
    return torch.cat(rows)


def _lin_jtu(lin: _Linearization, u: Tensor) -> Tensor:
    """Analytic ``J^T u``: residual perturbation -> coefficient perturbation (flat)."""
    n_scans = len(lin.P)
    nmask = lin.mask_idx.numel()
    numel = int(lin.shape[0] * lin.shape[1] * lin.shape[2])
    # Scatter each masked data chunk back to a full volume (with the row scaling).
    us: list[Tensor] = []
    for s in range(n_scans):
        chunk = u[s * nmask : (s + 1) * nmask]
        full = torch.zeros(numel, device=u.device, dtype=u.dtype)
        full[lin.mask_idx] = chunk * lin.scale
        us.append(full.reshape(lin.shape))
    # Adjoint of the (m_s - mean) centering across scans (symmetric).
    umean = torch.stack(us, dim=0).mean(dim=0)
    gfield = torch.zeros(lin.shape, device=u.device, dtype=u.dtype)
    for s in range(n_scans):
        w = us[s] - umean
        gfield = gfield + lin.a[s] * (
            lin.P[s] * w + _central_diff_pe_adjoint(lin.Q[s] * w, lin.pe_tdim[s])
        )
    if lin.lam_eff > 0:
        ureg = u[n_scans * nmask :]
        gfield = gfield + math.sqrt(lin.lam_eff) * reg_residual_adjoint(
            ureg, lin.shape, lin.reg_mode
        )
    return lin.basis.field_adjoint(gfield).reshape(-1)


def _gn_direction_analytic(
    coeff: Tensor,
    basis: SplineFieldBasis,
    scans: list[ScanSpec],
    mask_idx: Tensor,
    n_sm1: int,
    lam_eff: float,
    reg_mode: str,
    cg_iters: int,
    cg_tol: float,
) -> tuple[Tensor, float]:
    """Analytic Gauss-Newton step: same math as :func:`_gn_direction`, no autograd.

    The forward model is linearised once (:func:`_linearize`) and every CG matvec is
    ``J^T (J v)`` via :func:`_lin_jv` / :func:`_lin_jtu` — einsum + elementwise +
    finite-difference adjoints — eliminating the gather-backward that dominated the
    autograd path. Precision-neutral: it evaluates the identical ``J^T J`` operator.
    """
    r0 = _residual_vector(coeff, basis, scans, mask_idx, n_sm1, lam_eff, reg_mode)
    cost = float(_dot64(r0, r0))
    lin = _linearize(coeff, basis, scans, mask_idx, n_sm1, lam_eff, reg_mode)

    def matvec(v: Tensor) -> Tensor:
        jv = _lin_jv(lin, v.reshape(coeff.shape))
        return _lin_jtu(lin, jv) + 1e-8 * v

    g = _lin_jtu(lin, r0)  # J^T r
    delta = _cg_solve(matvec, -g, cg_iters, cg_tol)
    return delta, cost


def gn_solve_level(
    coeff: Tensor,
    basis: SplineFieldBasis,
    scans: list[ScanSpec],
    mask: Tensor,
    lam: float,
    ssqlambda: bool,
    reg_mode: str,
    max_iter: int,
    cg_iters: int,
    cg_tol: float,
    progress: bool = False,
    desc: str = "GN",
    analytic: bool = True,
) -> tuple[Tensor, float]:
    """Gauss-Newton least-squares minimisation of the topup cost at one level.

    ``J^T J`` is applied matrix-free. With ``analytic`` (default) the forward model is
    linearised once per GN step and ``J``/``J^T`` are explicit einsum + elementwise +
    finite-difference operators (:func:`_gn_direction_analytic`) — no autograd, no
    gather-scatter. Set ``analytic=False`` to use the reverse-mode-autodiff matvec
    (``J v`` via double-backward through ``J^T w``); the two evaluate the identical
    operator and are kept in sync by ``tests/test_topup.py``. A backtracking,
    step-rejecting line search guards every step. Returns ``(coefficients, cost)``.
    """
    mask_idx = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(-1)
    n_sm1 = max(1, len(scans) - 1)
    coeff = coeff.clone()

    def residual(c: Tensor, lam_eff: float) -> Tensor:
        return _residual_vector(c, basis, scans, mask_idx, n_sm1, lam_eff, reg_mode)

    def data_ssd(c: Tensor) -> float:
        r = residual(c, 0.0)
        return float(_dot64(r, r))

    bar = _bar(
        total=max_iter, desc=desc, leave=False, disable=not progress,
        bar_format="  {desc} {bar} {n_fmt}/{total_fmt} [{elapsed}] {postfix}",
    )
    prev_cost = None
    last_cost = float("nan")
    for _it in range(max_iter):
        lam_eff = lam * data_ssd(coeff) if ssqlambda else lam
        if analytic:
            delta, cost = _gn_direction_analytic(
                coeff, basis, scans, mask_idx, n_sm1, lam_eff, reg_mode, cg_iters, cg_tol
            )
        else:
            delta, cost = _gn_direction(coeff, residual, lam_eff, cg_iters, cg_tol)

        # Backtracking line search; reject any step that does not reduce the cost.
        step = 1.0
        accepted = False
        new_cost = cost
        for _ in range(12):
            trial = coeff + (step * delta).reshape(coeff.shape)
            r_new = residual(trial, lam_eff)
            new_cost = float(_dot64(r_new, r_new))
            if new_cost < cost:
                coeff = trial
                accepted = True
                break
            step *= 0.5
        last_cost = new_cost if accepted else cost
        bar.update(1)
        bar.set_postfix_str(f"cost={last_cost:.3e} λ={lam_eff:.1e}{'' if accepted else ' (reject)'}")
        if not accepted:
            break
        if prev_cost is not None and abs(prev_cost - new_cost) < 1e-6 * prev_cost:
            break
        prev_cost = new_cost
    bar.close()
    return coeff, last_cost


# ----------------------------------------------------------------------------
# Optional single global PE-direction translation (scanner shift)
# ----------------------------------------------------------------------------
def estimate_pe_shift(
    scans: list[ScanSpec], max_shift_vox: float = 1.0, smooth_sigma_vox: float = 3.0
) -> float:
    """Estimate one global translation (voxels, along PE) between the first two scans.

    Best-effort pre-alignment for the tiny (sub-voxel) scanner shift that can occur
    between back-to-back blip-up/down scans. Grid-searches the shift ``s`` that best
    matches scan 0 shifted by ``+s/2`` and scan 1 by ``-s/2`` (SSD).

    LIMITATION: a rigid translation is confounded by the (large) susceptibility field
    that legitimately differs between the two polarities, so this cannot fully isolate
    a scanner shift from field distortion the way FSL's joint movement+field estimation
    does. The images are heavily smoothed first (``smooth_sigma_vox``) so the estimate
    reflects bulk translation rather than local field structure, and the search is
    capped small (``max_shift_vox``) so it cannot introduce a large spurious shift.
    Off by default; treat the result as a coarse hint.
    """
    if len(scans) < 2:
        return 0.0
    pe_tdim = _NIFTI_AXIS_TO_TDIM[scans[0].pe_axis]
    a, b = scans[0].data, scans[1].data
    if smooth_sigma_vox > 0:
        a = _separable_smooth_3d(a, smooth_sigma_vox)
        b = _separable_smooth_3d(b, smooth_sigma_vox)
    best_s, best_c = 0.0, None
    steps = torch.linspace(-max_shift_vox, max_shift_vox, 81, device=a.device)
    for s in steps.tolist():
        aw = _resample_pe(a, torch.full_like(a, s * 0.5), pe_tdim)
        bw = _resample_pe(b, torch.full_like(b, -s * 0.5), pe_tdim)
        c = float(((aw - bw) ** 2).mean())
        if best_c is None or c < best_c:
            best_c, best_s = c, s
    return best_s


def apply_pe_shift(scans: list[ScanSpec], shift_vox: float) -> None:
    """Apply +/- shift/2 along PE to scans 0 and 1 in place (symmetric)."""
    if shift_vox == 0.0 or len(scans) < 2:
        return
    pe_tdim = _NIFTI_AXIS_TO_TDIM[scans[0].pe_axis]
    scans[0].data = _resample_pe(
        scans[0].data, torch.full_like(scans[0].data, shift_vox * 0.5), pe_tdim
    )
    scans[1].data = _resample_pe(
        scans[1].data, torch.full_like(scans[1].data, -shift_vox * 0.5), pe_tdim
    )


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------
@dataclass
class TopupResult:
    field_hz: Tensor  # (nz, ny, nx) off-resonance field, Hz
    coeff: Tensor  # final spline coefficients
    basis: SplineFieldBasis  # final (finest) basis
    pe_shift_vox: float
    unwarped: list[Tensor]  # per-scan Jacobian-modulated undistorted images (native intensity)
    mean_unwarped: Tensor  # mean of the undistorted images


def run_topup(
    scans: list[ScanSpec],
    voxel_sizes: tuple[float, float, float],
    config: TopupConfig,
    pe_shift: bool = False,
    progress: bool = True,
    solve_dtype: torch.dtype = torch.float32,
    mask_field: bool = True,
) -> TopupResult:
    """Estimate the off-resonance field from opposing-PE scans.

    ``scans`` share a grid (nz, ny, nx); ``voxel_sizes`` = (vz, vy, vx) mm.

    ``mask_field`` tapers the returned field (and hence the warp) to zero well outside
    the imaged object (see :func:`taper_field_to_object`) so the saved warp has no
    spurious displacement in air -- otherwise ffs_nwarp's border-clamp extrapolation
    replicates tissue into an auto-pad margin. The object and its distortion are
    untouched; only far air changes.

    ``solve_dtype`` is the working precision of the whole solve (field, coefficients,
    resampling, CG). float32 is ~2x faster than float64 on consumer GPUs at no accuracy
    cost here, because the numerically sensitive reductions (cost, CG inner products)
    accumulate in float64 regardless (see :func:`_dot64`); the smooth Hz field itself has
    plenty of headroom in float32. Pass ``torch.float64`` to reproduce the old behaviour.
    """
    config.validate()
    shape = tuple(scans[0].data.shape)  # type: ignore[assignment]
    device = scans[0].data.device
    dtype = scans[0].data.dtype

    # Work on clones so the caller's tensors are never mutated. Individual-scale each
    # scan to a common mean (topup's --scale=1) so the data term is comparable across
    # scans and against the regularisation; the field itself is intensity-independent,
    # so we un-scale the returned unwarped images back to native intensity at the end.
    scales = []
    work: list[ScanSpec] = []
    for sc in scans:
        m = float(sc.data.mean())
        s = (100.0 / m) if m != 0 else 1.0
        scales.append(s)
        work.append(
            ScanSpec(
                data=(sc.data * s).to(solve_dtype),
                pe_axis=sc.pe_axis,
                sign=sc.sign,
                readout=sc.readout,
            )
        )

    shift = 0.0
    if pe_shift:
        shift = estimate_pe_shift(work)
        apply_pe_shift(work, shift)
        if progress:
            print(f"  estimated PE shift: {shift:+.3f} vox")

    n_lev = config.n_levels()
    level_bar = _bar(
        total=n_lev, desc="blipflip", leave=True, disable=not progress,
        bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} levels [{elapsed}<{remaining}] {postfix}",
    )
    coeff = None
    basis = None
    for lvl in range(n_lev):
        wr = config.warpres[lvl]
        fwhm = config.fwhm[lvl]
        lam = config.lam[lvl]
        miter = config.miter[lvl]
        ss = config.subsamp[lvl]

        # Smooth (and optionally subsample) the data for this level.
        level_scans = _prepare_level_scans(work, voxel_sizes, fwhm, ss)
        level_shape = tuple(level_scans[0].data.shape)  # type: ignore[assignment]
        level_vox = tuple(v * ss for v in voxel_sizes)

        new_basis = build_spline_basis(level_shape, level_vox, wr, device, solve_dtype)
        if coeff is None or basis is None:
            coeff = torch.zeros(new_basis.coeff_shape, device=device, dtype=solve_dtype)
        else:
            # Warm-start: refit the previous field onto the new (level) grid + knots.
            prev_field = basis.field(coeff)
            prev_field = _resize_field(prev_field, level_shape)
            coeff = refit_coeff(prev_field, new_basis)
        basis = new_basis

        mask = compute_mask(level_scans)
        level_bar.set_postfix_str(
            f"warpres={wr}mm fwhm={fwhm}mm ss={ss} λ={lam:.1e} "
            f"grid={'x'.join(map(str, level_shape))} knots={'x'.join(map(str, basis.coeff_shape))}"
        )
        coeff, cost = gn_solve_level(
            coeff,
            basis,
            level_scans,
            mask,
            lam,
            config.ssqlambda,
            config.reg_mode,
            miter,
            config.cg_iters,
            config.cg_tol,
            progress,
            desc=f"L{lvl + 1}/{n_lev} {wr}mm",
            analytic=config.analytic_gn,
        )
        level_bar.update(1)
        level_bar.set_postfix_str(
            f"warpres={wr}mm fwhm={fwhm}mm ss={ss} λ={lam:.1e} cost={cost:.3e}"
        )
    level_bar.close()

    # Expand the final field back to the full (un-subsampled) grid.
    assert basis is not None and coeff is not None
    full_basis = build_spline_basis(shape, voxel_sizes, config.warpres[-1], device, solve_dtype)
    full_coeff = refit_coeff(_resize_field(basis.field(coeff), shape), full_basis)
    field_hz = full_basis.field(full_coeff).to(dtype)

    if mask_field:
        # Taper the field to zero in air, then derive the images from the *tapered*
        # field so warp, field map and unwarped stay mutually consistent (the coeff
        # in the result is still the raw fit).
        field_hz = taper_field_to_object(field_hz, work, voxel_sizes)
        field_s = field_hz.to(solve_dtype)
        unwarped = []
        for sc, s in zip(work, scales, strict=True):
            pe_tdim = _NIFTI_AXIS_TO_TDIM[sc.pe_axis]
            disp = field_s * (sc.readout * sc.sign)
            m = _resample_pe(sc.data, disp, pe_tdim) * _jacobian_pe(disp, pe_tdim)
            unwarped.append((m / s).to(dtype))
    else:
        # Jacobian-modulated undistorted images, un-scaled back to native intensity.
        modulated, _ = forward_scans(full_coeff, full_basis, work)
        unwarped = [(m / s).to(dtype) for m, s in zip(modulated, scales, strict=True)]
    mean_unwarped = torch.stack(unwarped, dim=0).mean(dim=0)
    return TopupResult(
        field_hz=field_hz,
        coeff=full_coeff,
        basis=full_basis,
        pe_shift_vox=shift,
        unwarped=unwarped,
        mean_unwarped=mean_unwarped,
    )


def _resize_field(field: Tensor, out_shape: tuple[int, int, int]) -> Tensor:
    if tuple(field.shape) == tuple(out_shape):
        return field
    out = F.interpolate(field[None, None], size=out_shape, mode="trilinear", align_corners=True)[
        0, 0
    ]
    return out


def _prepare_level_scans(
    scans: list[ScanSpec],
    voxel_sizes: tuple[float, float, float],
    fwhm_mm: float,
    subsamp: int,
) -> list[ScanSpec]:
    """Smooth (FWHM mm) and optionally subsample each scan for one level."""
    vz, vy, vx = voxel_sizes
    out: list[ScanSpec] = []
    for sc in scans:
        d = sc.data
        if fwhm_mm > 0:
            # Convert FWHM(mm) to sigma(voxels); use mean voxel size (fields are smooth).
            sigma_vox = (fwhm_mm / (2.0 * math.sqrt(2.0 * math.log(2.0)))) / ((vz + vy + vx) / 3.0)
            if sigma_vox > 0:
                # Smooth in float32 (the shared kernel is float32; smoothing is not the
                # numerically sensitive step) and restore the solve dtype.
                d = _separable_smooth_3d(d.float(), sigma_vox).to(d.dtype)
        if subsamp > 1:
            d = d[::subsamp, ::subsamp, ::subsamp].contiguous()
        out.append(ScanSpec(data=d, pe_axis=sc.pe_axis, sign=sc.sign, readout=sc.readout))
    return out
