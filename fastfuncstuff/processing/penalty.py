"""Warp distortion penalty functions.

Implements AFNI's eight-corner ``hexahedron_energy`` from
``IW3D_load_energy()``, which penalizes excessive bulk distortion and
determinant-normalized shear/vorticity.
The penalty encourages the warp to be a smooth diffeomorphism.

The penalty has two components:
  - je: bulk volume distortion energy (Jacobian determinant deviation from 1)
  - se: shear and vorticity energy

Total penalty = pen_fac * (sum_of_energies)^0.25

Both single-volume and batched versions are provided. The batched version
processes B patches in parallel on GPU without Python loops.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor

try:
    from .penalty_triton import penalty_sums_triton
except Exception:  # pragma: no cover - Triton is optional and CUDA-only
    penalty_sums_triton = None

_penalty_triton_unavailable = False


def _set_penalty_triton_unavailable(message: str) -> None:
    """Latch the fused penalty off for the rest of the process, saying so once."""
    global _penalty_triton_unavailable
    if not _penalty_triton_unavailable:
        _penalty_triton_unavailable = True
        print(f"** {message}")


def _central_diff_batched(vol: Tensor, dim: int) -> Tensor:
    """Central difference with one-sided differences on the volume faces."""
    tdim = dim - 3
    n = vol.shape[tdim]
    if n < 2:
        return torch.zeros_like(vol)
    result = torch.zeros_like(vol)
    result.narrow(tdim, 1, n - 2).copy_(
        0.5 * (vol.narrow(tdim, 2, n - 2) - vol.narrow(tdim, 0, n - 2))
    )
    result.narrow(tdim, 0, 1).copy_(vol.narrow(tdim, 1, 1) - vol.narrow(tdim, 0, 1))
    result.narrow(tdim, n - 1, 1).copy_(vol.narrow(tdim, n - 1, 1) - vol.narrow(tdim, n - 2, 1))
    return result


def compute_jacobian_energy(xd: Tensor, yd: Tensor, zd: Tensor) -> tuple[Tensor, Tensor]:
    """Legacy central-difference energy used by formwarp's sparse fold guard."""
    dxd_di = _central_diff_batched(xd, dim=2)
    dxd_dj = _central_diff_batched(xd, dim=1)
    dxd_dk = _central_diff_batched(xd, dim=0)
    dyd_di = _central_diff_batched(yd, dim=2)
    dyd_dj = _central_diff_batched(yd, dim=1)
    dyd_dk = _central_diff_batched(yd, dim=0)
    dzd_di = _central_diff_batched(zd, dim=2)
    dzd_dj = _central_diff_batched(zd, dim=1)
    dzd_dk = _central_diff_batched(zd, dim=0)

    a11, a12, a13 = 1.0 + dxd_di, dxd_dj, dxd_dk
    a21, a22, a23 = dyd_di, 1.0 + dyd_dj, dyd_dk
    a31, a32, a33 = dzd_di, dzd_dj, 1.0 + dzd_dk
    det = (
        a11 * (a22 * a33 - a23 * a32)
        - a12 * (a21 * a33 - a23 * a31)
        + a13 * (a21 * a32 - a22 * a31)
    )
    je = (det - 1.0).square()
    se = (
        (0.5 * (a12 + a21)).square()
        + (0.5 * (a13 + a31)).square()
        + (0.5 * (a23 + a32)).square()
        + (0.5 * (a12 - a21)).square()
        + (0.5 * (a13 - a31)).square()
        + (0.5 * (a23 - a32)).square()
        + 0.5 * ((a11 - 1.0).square() + (a22 - 1.0).square() + (a33 - 1.0).square())
    )
    return je, se


def compute_hexahedron_energy(xd: Tensor, yd: Tensor, zd: Tensor) -> tuple[Tensor, Tensor]:
    """Compute AFNI's hexahedral bulk and shear energy fields.

    Works for both single volumes (nz, ny, nx) and batched (B, nz, ny, nx).

    Args:
        xd, yd, zd: (..., nz, ny, nx) displacement fields.

    Returns:
        (je, se): Bulk distortion and shear/vorticity energy, same shape.
    """
    if xd.shape != yd.shape or xd.shape != zd.shape or xd.ndim < 3:
        raise ValueError("displacement components must share at least three spatial dimensions")

    shape = xd.shape

    def _corners(vol: Tensor) -> tuple[Tensor, ...]:
        flat = vol.reshape(-1, 1, *shape[-3:])
        p = torch.nn.functional.pad(flat, (0, 1, 0, 1, 0, 1), mode="replicate")[:, 0]
        nz, ny, nx = shape[-3:]
        return (
            p[:, :nz, :ny, :nx],
            p[:, :nz, :ny, 1 : nx + 1],
            p[:, :nz, 1 : ny + 1, :nx],
            p[:, :nz, 1 : ny + 1, 1 : nx + 1],
            p[:, 1 : nz + 1, :ny, :nx],
            p[:, 1 : nz + 1, :ny, 1 : nx + 1],
            p[:, 1 : nz + 1, 1 : ny + 1, :nx],
            p[:, 1 : nz + 1, 1 : ny + 1, 1 : nx + 1],
        )

    xc, yc, zc = _corners(xd), _corners(yd), _corners(zd)

    # Average first differences at the 000 and opposite 111 corners, exactly
    # as AFNI's hexahedron_energy(). Its matrix is transposed relative to the
    # conventional displacement-gradient layout; determinant and energy are
    # invariant to that transpose.
    fxx = 0.5 * ((xc[1] - xc[0]) + (xc[7] - xc[6])) + 1.0
    fxy = 0.5 * ((yc[1] - yc[0]) + (yc[7] - yc[6]))
    fxz = 0.5 * ((zc[1] - zc[0]) + (zc[7] - zc[6]))
    fyx = 0.5 * ((xc[2] - xc[0]) + (xc[7] - xc[5]))
    fyy = 0.5 * ((yc[2] - yc[0]) + (yc[7] - yc[5])) + 1.0
    fyz = 0.5 * ((zc[2] - zc[0]) + (zc[7] - zc[5]))
    fzx = 0.5 * ((xc[4] - xc[0]) + (xc[7] - xc[3]))
    fzy = 0.5 * ((yc[4] - yc[0]) + (yc[7] - yc[3]))
    fzz = 0.5 * ((zc[4] - zc[0]) + (zc[7] - zc[3])) + 1.0

    det = (
        fxx * (fyy * fzz - fyz * fzy)
        - fxy * (fyx * fzz - fyz * fzx)
        + fxz * (fyx * fzy - fyy * fzx)
    ).clamp(0.1, 10.0)

    je = (1.0 / 3.0) * (det - det.reciprocal()).square()
    matrix_norm = sum(f.square() for f in (fxx, fxy, fxz, fyx, fyy, fyz, fzx, fzy, fzz))
    vorticity = 2.0 * ((fyz - fzy).square() + (fxz - fzx).square() + (fxy - fyx).square())
    se = ((matrix_norm + vorticity) / det.pow(2.0 / 3.0) - 3.0).clamp_min(0.0)

    je = je.reshape(shape)
    se = se.reshape(shape)

    return je, se


# Voxel-wise energy below this contributes NOTHING to the penalty (AFNI's
# Hpen_cut, mri_nwarp.c:2241). AFNI's reciprocal-Jacobian bulk energy is steep
# under compression: expansion above ~2.19 or compression below ~0.46 crosses
# this deadband. Ordinary deformation is free.
HPEN_CUT = 1.0


def penalty_energy(je: Tensor, se: Tensor, cut: float = HPEN_CUT) -> Tensor:
    """Per-voxel penalty contribution: the excess over ``cut``, to the 4th power.

    Both halves matter and we had neither, which quietly crippled every warp.

    The **deadband** is what makes the penalty local. AFNI charges nothing for
    benign deformation and only starts counting once a voxel is genuinely extreme;
    summing raw ``je + se`` instead taxes every voxel of a perfectly sound warp, so
    the penalty scales with how much the image deformed *at all* rather than with
    how badly it misbehaved. Measured on a T1->MNI pair, that flat tax shrank the
    warp to 22% of AFNI's displacement (mean 0.98 vs 4.46 voxels) and made every
    level past the second actively worse.

    The **4th power** is what makes it steep once it does bite. Paired with the
    ``^0.25`` applied to the total, a single dominant voxel contributes
    ``(ev^4)^0.25 = ev`` -- so the penalty behaves like a soft maximum over the
    worst excess in the field, not an average over all of it. Penalising the mean
    is what a flat tax does; penalising the worst is what a guard should do.
    """
    ej = (je - cut).clamp(min=0.0)
    es = (se - cut).clamp(min=0.0)
    return ej.pow(4) + es.pow(4)


def compute_penalty(
    xd: Tensor,
    yd: Tensor,
    zd: Tensor,
    pen_fac: float = 0.033333,
    external_sum: float = 0.0,
) -> float:
    """Compute total warp distortion penalty (single volume, serial path)."""
    je, se = compute_hexahedron_energy(xd, yd, zd)
    hsum = external_sum + float(penalty_energy(je, se).sum().item())
    if hsum > 0:
        return pen_fac * (hsum**0.25)
    return 0.0


def compute_penalty_batched(
    xd: Tensor,
    yd: Tensor,
    zd: Tensor,
    pen_fac: float,
    external_sums: Tensor,
) -> Tensor:
    """Compute penalty for B patches in parallel. Returns (B,) tensor, differentiable.

    No Python loops - fully vectorized on GPU.

    Args:
        xd, yd, zd: (B, nz, ny, nx) composed displacement fields per patch.
        pen_fac: Penalty scaling factor.
        external_sums: (B,) pre-computed external penalty per patch.

    Returns:
        (B,) penalty values, differentiable.
    """
    patch_sums = None
    # The fused kernel keeps the nine difference fields and the forty expressions
    # over them in registers, so nothing between the displacements and this sum is
    # ever written down. It is not differentiable, so the Adam path -- which needs
    # a backward pass through exactly these terms -- keeps the tensor version.
    if (
        penalty_sums_triton is not None
        and not _penalty_triton_unavailable
        and xd.device.type == "cuda"
        and xd.dtype == torch.float32
        and xd.ndim == 4
        and not (xd.requires_grad or yd.requires_grad or zd.requires_grad)
        and os.environ.get("FFS_PENALTY_NO_TRITON") != "1"
    ):
        try:
            patch_sums = penalty_sums_triton(xd, yd, zd, HPEN_CUT)
        except AssertionError:
            raise  # a failed assertion is a bug here, not a missing GPU capability
        except Exception as exc:  # pragma: no cover - needs a Triton-hostile GPU
            _set_penalty_triton_unavailable(
                f"fused deformation penalty unavailable ({type(exc).__name__}: {exc}); "
                "falling back to the tensor implementation"
            )

    if patch_sums is None:
        # Batched Jacobian energy: (B, nz, ny, nx)
        je, se = compute_hexahedron_energy(xd, yd, zd)
        # Sum over spatial dims, keep batch: (B,)
        patch_sums = penalty_energy(je, se).sum(dim=(-3, -2, -1))

    hsum = (external_sums + patch_sums).clamp(min=0)

    return pen_fac * hsum.pow(0.25)
