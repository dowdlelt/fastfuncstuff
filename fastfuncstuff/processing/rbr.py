"""Recursive Boundary Registration (RBR) — the nonlinear stage.

Method and design decisions: ``../fmri_wiki/concepts/Recursive Boundary
Registration.md``; source: ``../fmri_wiki/sources/Van Mourik 2019.md``.

Where ``ffs_bbr`` refines the *affine* (single-element BBR), this fixes the
*local* geometric distortion an affine cannot — dominantly along the phase-encode
axis for EPI. We deviate from the paper's octree + fixed-tetrahedra deformation
(which the authors themselves flag as biased) in favour of a **smooth
multi-resolution control-point (free-form-deformation) warp**: continuous by
construction, no self-intersection, and optimized end-to-end by autograd through
the differentiable BBR boundary cost in :mod:`fastfuncstuff.processing.bbr`.

The warp is parametrized by a coarse→fine control grid of PE-axis displacements;
the boundary point cloud (from a WM mask, cast into EPI space) is shifted by the
interpolated field and the BBR contrast is maximized, with a membrane
smoothness penalty standing in for the paper's neighbour smoothing.

Coordinate convention follows :mod:`bbr` / ``interp``: points and fields are
``(x, y, z)`` voxel-index space over a ``(nz, ny, nx)`` volume.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .bbr import boundary_contrast, greve_fischl_cost, ngf_eta, ngf_score_at
from .interp import trilinear_interpolate
from .tissue import synthesis_cost_at_points

# ── Control-grid (free-form deformation) warp ────────────────────────────────


def control_grid_shape(vol_shape: tuple[int, int, int], spacing: float) -> tuple[int, int, int]:
    """(Kz, Ky, Kx) control-point counts spanning the volume at ~``spacing`` vox."""
    kz, ky, kx = (max(2, int(round(d / spacing)) + 1) for d in vol_shape)
    return (kz, ky, kx)


def eval_displacement(ctrl: Tensor, points: Tensor, vol_shape: tuple[int, int, int]) -> Tensor:
    """Interpolate the control-grid displacement at ``points``.

    Args:
        ctrl: (C, Kz, Ky, Kx) control-point displacements for ``C`` free axes.
        points: (P, 3) sample locations in (x, y, z) voxel coordinates.
        vol_shape: (nz, ny, nx) — the grid the control lattice spans.

    Returns:
        (P, C) displacement per point (differentiable in ``ctrl``).
    """
    nz, ny, nx = vol_shape
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    gx = 2.0 * x / (nx - 1) - 1.0 if nx > 1 else x * 0.0
    gy = 2.0 * y / (ny - 1) - 1.0 if ny > 1 else y * 0.0
    gz = 2.0 * z / (nz - 1) - 1.0 if nz > 1 else z * 0.0
    grid = torch.stack([gx, gy, gz], dim=-1)[None, None, None]  # (1,1,1,P,3)
    disp = F.grid_sample(ctrl[None], grid, align_corners=True, padding_mode="border")
    return disp.reshape(ctrl.shape[0], -1).T  # (P, C)


def dense_field(ctrl: Tensor, vol_shape: tuple[int, int, int], axes: list[int]) -> Tensor:
    """Expand the control grid to a dense ``(3, nz, ny, nx)`` displacement field.

    The control lattice maps linearly onto the volume, so the dense field is just
    the control grid trilinearly upsampled to the volume shape, scattered into
    the chosen ``axes`` (x=0, y=1, z=2); other axes stay zero.
    """
    up = F.interpolate(ctrl[None], size=tuple(vol_shape), mode="trilinear", align_corners=True)[0]
    field = torch.zeros((3, *vol_shape), device=ctrl.device, dtype=ctrl.dtype)
    for i, ax in enumerate(axes):
        field[ax] = up[i]
    return field


def membrane_penalty(ctrl: Tensor) -> Tensor:
    """Mean squared first difference across the three grid axes — a membrane
    (first-order) smoothness regularizer standing in for the paper's neighbour
    smoothing. Scale-stable (mean, not sum) across grid resolutions."""
    p = ctrl.new_zeros(())
    for d in (1, 2, 3):
        if ctrl.shape[d] > 1:
            p = p + (torch.diff(ctrl, dim=d) ** 2).mean()
    return p


def upsample_ctrl(ctrl: Tensor, new_shape: tuple[int, int, int]) -> Tensor:
    """Trilinearly resample a control grid to a new (Kz, Ky, Kx) — warm-start a
    finer level from the coarser solution."""
    return F.interpolate(ctrl[None], size=tuple(new_shape), mode="trilinear", align_corners=True)[0]


# ── Coarse→fine optimization ─────────────────────────────────────────────────


def optimize_rbr(
    volume: Tensor,
    points: Tensor,
    normals: Tensor,
    *,
    axes: list[int] | None = None,
    spacings: list[float] | None = None,
    offsets: list[float] | float = 2.0,
    reg_weight: float = 1.0,
    reverse: bool = False,
    weight: Tensor | None = None,
    edge_points: Tensor | None = None,
    edge_normals: Tensor | None = None,
    grad_field: Tensor | None = None,
    edge_weight: Tensor | None = None,
    ngf_weight: float = 1.0,
    tissue_coords: Tensor | None = None,
    tissue_F: Tensor | None = None,
    tissue_Fpinv: Tensor | None = None,
    tissue_weight: float = 1.0,
    iters: int = 200,
    lr: float = 0.3,
    tol: float = 1e-5,
    verbose: bool = False,
) -> dict:
    """Fit a smooth multi-resolution PE-axis warp that seats a boundary on edges.

    Coarse→fine: at each control-grid spacing, Adam minimizes the BBR boundary
    cost (boundary shifted by the interpolated field) plus a membrane smoothness
    penalty; the finer level warm-starts from the upsampled coarser field. Normals
    are held fixed — valid for the small, smooth residual a good affine leaves.

    Args:
        volume: (nz, ny, nx) target (EPI) intensity volume.
        points, normals: boundary point cloud (x, y, z) and unit outward normals.
        axes: free displacement axes (default [1] = the y / phase-encode axis).
        spacings: control-grid spacings in voxels, coarse→fine
            (default [volume/4, /8, /16] capped to a sane range).
        offsets: BBR sampling half-distance per level (scalar or per-level list);
            wider on coarse levels widens the capture range.
        reg_weight: membrane-penalty weight (higher = smoother/stiffer).
        reverse, weight: passed to the BBR cost (polarity, per-point weights).
        edge_points, edge_normals, grad_field: optional anat edge cloud + EPI
            gradient field for an added NGF (gradient-direction) data term, shifted
            by the same warp. If given, they drive the fit alongside (or instead of)
            the WM BBR term. ``points``/``normals`` may be empty to use edges only.
        edge_weight: optional per-edge-point weights for the NGF term.
        ngf_weight: weight of the NGF edge term relative to the WM BBR term.
        tissue_coords, tissue_F, tissue_Fpinv: optional dense partial-volume
            (tissue-synthesis) term — EPI-space sample coords + static design and
            projector (from ``tissue.build_tissue_design``/``tissue_projector``),
            shifted by the same warp. Sharp well, so best combined with wm/edges.
        tissue_weight: weight of the tissue term relative to the others.
        iters, lr, tol: Adam cap, learning rate, relative-cost tol per level.
        verbose: print per-level progress.

    Returns:
        dict with ``field`` (3, nz, ny, nx) dense displacement, ``ctrl`` (finest
        control grid), ``axes``, ``init_cost``, ``final_cost``, ``level_costs``.
    """
    dev, dt = points.device, points.dtype
    vol_shape = tuple(volume.shape)  # (nz, ny, nx)
    if axes is None:
        axes = [1]
    C = len(axes)
    ax_t = torch.tensor(axes, device=dev)

    if spacings is None:
        m = max(vol_shape)
        spacings = [max(6.0, m / 4), max(5.0, m / 8), max(4.0, m / 16)]
    if isinstance(offsets, (int, float)):
        offsets = [float(offsets)] * len(spacings)
    if len(offsets) != len(spacings):
        raise ValueError("offsets must be a scalar or match len(spacings)")

    use_wm = points.numel() > 0
    use_edges = edge_points is not None and grad_field is not None and edge_normals is not None
    use_tissue = tissue_coords is not None and tissue_F is not None and tissue_Fpinv is not None
    if not (use_wm or use_edges or use_tissue):
        raise ValueError("optimize_rbr needs a WM boundary, an edge (NGF), and/or a tissue cloud")
    eta = ngf_eta(grad_field) if use_edges and grad_field is not None else 0.0

    def data_cost(ctrl: Tensor, offset: float) -> Tensor:
        c = ctrl.new_zeros(())
        if use_wm:
            disp = eval_displacement(ctrl, points, vol_shape)  # (P, C)
            shifted = points.index_add(1, ax_t, disp)  # add disp to the chosen axes
            contrast = boundary_contrast(volume, shifted, normals, offset)
            c = c + greve_fischl_cost(contrast, reverse=reverse, weight=weight)
        if use_edges:
            assert edge_points is not None and edge_normals is not None and grad_field is not None
            edisp = eval_displacement(ctrl, edge_points, vol_shape)  # (Pe, C)
            eshift = edge_points.index_add(1, ax_t, edisp)
            c = c + ngf_weight * ngf_score_at(
                grad_field, eshift, edge_normals, eta=eta, weight=edge_weight
            )
        if use_tissue:
            assert tissue_coords is not None and tissue_F is not None and tissue_Fpinv is not None
            tdisp = eval_displacement(ctrl, tissue_coords, vol_shape)  # (Pt, C)
            tshift = tissue_coords.index_add(1, ax_t, tdisp)
            c = c + tissue_weight * synthesis_cost_at_points(
                volume, tshift, tissue_F, tissue_Fpinv
            )
        return c

    # Initial (identity) cost at the finest offset, for reporting.
    init_cost = data_cost(
        torch.zeros((C, 2, 2, 2), device=dev, dtype=dt), float(offsets[-1])
    ).item()

    ctrl: Tensor | None = None
    level_costs: list[float] = []
    for lvl, (spacing, offset) in enumerate(zip(spacings, offsets, strict=True)):
        K = control_grid_shape(vol_shape, spacing)
        if ctrl is None:
            ctrl = torch.zeros((C, *K), device=dev, dtype=dt)
        else:
            ctrl = upsample_ctrl(ctrl, K)
        ctrl = ctrl.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([ctrl], lr=lr)
        prev = float("inf")
        for _ in range(iters):
            opt.zero_grad()
            loss = data_cost(ctrl, float(offset)) + reg_weight * membrane_penalty(ctrl)
            loss.backward()
            opt.step()
            cur = loss.item()
            if abs(prev - cur) < tol * (abs(prev) + 1e-12):
                break
            prev = cur
        with torch.no_grad():
            lc = data_cost(ctrl, float(offset)).item()
        level_costs.append(lc)
        if verbose:
            print(f"  level {lvl} (K={tuple(K)}, offset={offset}): data cost → {lc:.4f}")

    assert ctrl is not None
    field = dense_field(ctrl.detach(), vol_shape, axes)  # type: ignore[arg-type]
    final_cost = data_cost(ctrl.detach(), float(offsets[-1])).item()
    return {
        "field": field,
        "ctrl": ctrl.detach(),
        "axes": axes,
        "init_cost": init_cost,
        "final_cost": final_cost,
        "level_costs": level_costs,
    }


# ── Resampling helpers ───────────────────────────────────────────────────────


def sample_zero_outside(vol: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    """Trilinear sample with **zero** outside the volume (not border-clamp).

    ``trilinear_interpolate`` uses border padding, which *replicates* edge voxels
    — catastrophic when the sample grid extends past a small EPI FoV (edge slices
    smear across the empty region). Anything out of bounds is set to 0 here.
    """
    nz, ny, nx = vol.shape
    oob = (x < -0.5) | (x > nx - 0.5) | (y < -0.5) | (y > ny - 0.5) | (z < -0.5) | (z > nz - 0.5)
    out = trilinear_interpolate(vol, x, y, z)
    return torch.where(oob, torch.zeros_like(out), out)


def epi_grid_coords(
    epi_shape: tuple[int, int, int], upsample: int, device: torch.device | str = "cpu"
) -> tuple[tuple[int, int, int], Tensor, Tensor, Tensor]:
    """Voxel coords of an ``upsample``× finer grid over the *same* EPI FoV.

    Returns the target shape and the native-EPI-voxel (x, y, z) coordinates of
    every target voxel (target voxel ``t`` ↔ EPI voxel ``t/upsample``). The saved
    NIfTI affine for this grid is ``epi_affine @ diag(1/N, 1/N, 1/N, 1)``.
    """
    nz, ny, nx = epi_shape
    n = int(upsample)
    tz, ty, tx = torch.meshgrid(
        torch.arange(n * nz, device=device, dtype=torch.float32),
        torch.arange(n * ny, device=device, dtype=torch.float32),
        torch.arange(n * nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (n * nz, n * ny, n * nx), tx.reshape(-1) / n, ty.reshape(-1) / n, tz.reshape(-1) / n


def invert_displacement_field(field: Tensor, iters: int = 12) -> Tensor:
    """Approximate inverse of a dense pull displacement field.

    Given ``field`` = d with ``undistorted(p) = raw(p + d(p))``, returns d⁻¹ with
    ``raw(p) = undistorted(p + d⁻¹(p))`` — i.e. the warp that *distorts* an
    undistorted volume (e.g. the affine-aligned anat) back into the raw EPI frame,
    so it overlays on the acquired EPI. Fixed-point iteration
    ``d⁻¹ ← -d(p + d⁻¹)``; converges for the small, smooth fields RBR produces
    (|∇d| < 1). Same (3, nz, ny, nx) layout.

    Args:
        field: (3, nz, ny, nx) forward displacement, (x, y, z) components.
        iters: fixed-point iterations (12 is plenty for sub-voxel smooth fields).

    Returns:
        (3, nz, ny, nx) inverse displacement field.
    """
    _, nz, ny, nx = field.shape
    dev, dt = field.device, field.dtype
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev, dtype=dt),
        torch.arange(ny, device=dev, dtype=dt),
        torch.arange(nx, device=dev, dtype=dt),
        indexing="ij",
    )
    xf, yf, zf = xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)
    inv = torch.zeros_like(field)
    for _ in range(iters):
        px = xf + inv[0].reshape(-1)
        py = yf + inv[1].reshape(-1)
        pz = zf + inv[2].reshape(-1)
        inv = -torch.stack(
            [
                trilinear_interpolate(field[0], px, py, pz).reshape(nz, ny, nx),
                trilinear_interpolate(field[1], px, py, pz).reshape(nz, ny, nx),
                trilinear_interpolate(field[2], px, py, pz).reshape(nz, ny, nx),
            ],
            dim=0,
        )
    return inv


def resample_with_affine_field(
    epi: Tensor,
    matrix_anat2epi: Tensor,
    field: Tensor,
    out_shape: tuple[int, int, int],
) -> Tensor:
    """Resample the EPI onto the anat grid through ``affine then RBR field``.

    For each anat voxel ``s`` the EPI is sampled at ``A·s + d(A·s)``, where ``A``
    is the anat→EPI voxel affine and ``d`` the EPI-space displacement field — the
    exact nonlinear analogue of the linear ``epi[A'·s]`` (there ``d`` is constant).
    Anat voxels whose affine image ``A·s`` falls outside the EPI FoV get 0 (no
    data), not a replicated edge slice.

    Args:
        epi: (nz, ny, nx) EPI intensity volume (native grid).
        matrix_anat2epi: (4, 4) anat→EPI voxel affine.
        field: (3, ez, ey, ex) EPI-grid displacement, (x, y, z) components.
        out_shape: (nz, ny, nx) of the anat grid.

    Returns:
        (nz, ny, nx) EPI resampled onto the anat grid, distortion-corrected.
    """
    dev = epi.device
    onz, ony, onx = out_shape
    ez, ey, ex = epi.shape
    A = matrix_anat2epi.to(device=dev, dtype=torch.float32)
    zz, yy, xx = torch.meshgrid(
        torch.arange(onz, device=dev, dtype=torch.float32),
        torch.arange(ony, device=dev, dtype=torch.float32),
        torch.arange(onx, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(xx)
    coords = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1), ones.reshape(-1)], dim=0)
    p = A @ coords  # (4, N) → EPI voxel coords
    px, py, pz = p[0], p[1], p[2]
    # No EPI data outside the FoV — mask on the affine image, not the warped one.
    in_fov = (
        (px >= -0.5)
        & (px <= ex - 0.5)
        & (py >= -0.5)
        & (py <= ey - 0.5)
        & (pz >= -0.5)
        & (pz <= ez - 0.5)
    )
    dx = trilinear_interpolate(field[0], px, py, pz)
    dy = trilinear_interpolate(field[1], px, py, pz)
    dz = trilinear_interpolate(field[2], px, py, pz)
    out = trilinear_interpolate(epi, px + dx, py + dy, pz + dz)
    out = torch.where(in_fov, out, torch.zeros_like(out))
    return out.reshape(onz, ony, onx)
