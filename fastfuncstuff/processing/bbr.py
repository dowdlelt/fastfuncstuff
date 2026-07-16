"""Boundary-Based Registration (BBR) contrast cost — the core primitive for
Recursive Boundary Registration (RBR).

Method and design decisions: ``../fmri_wiki/concepts/Recursive Boundary
Registration.md``; source: ``../fmri_wiki/sources/Van Mourik 2019.md``.

Volume-first (Phase 1): the WM/GM boundary point cloud and its outward unit
normals are derived from a user-supplied *binary WM mask* (assumed FreeSurfer,
but any segmentation works — the mask arrives already in the target grid, e.g.
the anatomical brought into EPI space by inverting the EPI→anat affine). The BBR
contrast cost samples the target volume just inside and just outside the boundary
along the normal and rewards a strong, consistent edge. It is differentiable in
the transform parameters so per-element transforms can be optimized by autograd /
Gauss-Newton in Phase 2.

Coordinate convention matches ``interp.trilinear_interpolate``: points are
``(x, y, z)`` in voxel-index coordinates, where ``x`` indexes the last array
axis (``nx``) and ``z`` the first (``nz``) of a ``(nz, ny, nx)`` volume.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor

from .interp import trilinear_interpolate

# ── Boundary + normal extraction ─────────────────────────────────────────────


def extract_boundary_normals(
    wm_mask: Tensor | np.ndarray,
    *,
    refine: bool = True,
    min_grad: float = 1e-2,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Boundary point cloud + outward unit normals from a binary WM mask.

    The signed distance transform (SDT) of the mask is positive outside WM and
    negative inside, so its spatial gradient points outward (WM→GM) — exactly
    the BBR normal. Boundary points are the WM surface voxels (WM voxels with a
    background neighbour); with ``refine`` they are nudged onto the sub-voxel
    zero level set along the normal.

    Args:
        wm_mask: (nz, ny, nx) binary mask (anything > 0 is WM).
        refine: push voxel-centre points onto the zero level set (sub-voxel).
        min_grad: drop points whose SDT gradient magnitude is below this (flat
            regions where the normal is undefined).
        device, dtype: placement of the returned tensors.

    Returns:
        points: (P, 3) float, (x, y, z) voxel-index coordinates on the boundary.
        normals: (P, 3) float, unit outward normals (WM→GM).
    """
    from scipy import ndimage

    if isinstance(wm_mask, Tensor):
        mask_np = wm_mask.detach().cpu().numpy()
    else:
        mask_np = np.asarray(wm_mask)
    wm = mask_np > 0
    if wm.ndim != 3:
        raise ValueError(f"wm_mask must be 3D (nz, ny, nx); got shape {mask_np.shape}")
    if not wm.any():
        raise ValueError("wm_mask is empty — no boundary to extract")

    # SDT: + outside WM, − inside; gradient points WM→GM.
    dist_out = ndimage.distance_transform_edt(~wm)
    dist_in = ndimage.distance_transform_edt(wm)
    sdt = (dist_out - dist_in).astype(np.float64)

    # Gradient (np.gradient returns [d/axis0, d/axis1, d/axis2] = [z, y, x]).
    gz, gy, gx = np.gradient(sdt)

    # Boundary voxels: WM surface (WM with a background 6-neighbour).
    eroded = ndimage.binary_erosion(wm, iterations=1, border_value=0)
    surf = wm & ~eroded
    kk, jj, ii = np.nonzero(surf)  # array indices (z, y, x)

    nx_c = gx[kk, jj, ii]
    ny_c = gy[kk, jj, ii]
    nz_c = gz[kk, jj, ii]
    gmag = np.sqrt(nx_c**2 + ny_c**2 + nz_c**2)

    keep = gmag > min_grad
    kk, jj, ii = kk[keep], jj[keep], ii[keep]
    nx_c, ny_c, nz_c, gmag = nx_c[keep], ny_c[keep], nz_c[keep], gmag[keep]
    if kk.size == 0:
        raise ValueError("no boundary points survived the min_grad filter")

    normals = np.stack([nx_c / gmag, ny_c / gmag, nz_c / gmag], axis=1)  # (P,3) x,y,z
    points = np.stack([ii, jj, kk], axis=1).astype(np.float64)  # (P,3) x,y,z

    if refine:
        # Move onto the boundary along the normal. The binary-mask surface sits
        # ~half a voxel outside the last foreground centre, but the EDT measures
        # centre-to-centre distance and overshoots by 0.5 — correct for it so a
        # surface voxel (|sdt|≈1) lands at the true half-voxel interface.
        s = sdt[kk, jj, ii]
        s_corr = s - 0.5 * np.sign(s)
        s_corr[np.abs(s) < 0.5] = s[np.abs(s) < 0.5]  # too close to correct safely
        points = points - s_corr[:, None] * normals

    pts = torch.as_tensor(points, dtype=dtype, device=device)
    nrm = torch.as_tensor(normals, dtype=dtype, device=device)
    return pts, nrm


# ── Transform (pivot-centred rotation · scale · translation) ─────────────────


def rst_matrix(params: Tensor, pivot: Tensor) -> Tensor:
    """4×4 pivot-centred transform from 9 RBR parameters.

    ``params = [rx, ry, rz, sx, sy, sz, tx, ty, tz]`` — rotations in degrees,
    scales (1 = identity), translations in voxels. The transform is
    ``p' = R·S·(p − pivot) + pivot + t`` (rotation composed ``Rz@Rx@Ry`` to match
    the rest of the codebase). Differentiable in ``params``.

    Args:
        params: (9,) parameter vector.
        pivot: (3,) rotation/scale centre in (x, y, z) voxel coordinates.

    Returns:
        (4, 4) homogeneous matrix mapping column vectors ``[x, y, z, 1]``.
    """
    dev, dt = params.device, params.dtype
    rx, ry, rz = params[0], params[1], params[2]
    s = params[3:6]
    t = params[6:9]

    deg = math.pi / 180.0
    cx, sx = torch.cos(rx * deg), torch.sin(rx * deg)
    cy, sy = torch.cos(ry * deg), torch.sin(ry * deg)
    cz, sz = torch.cos(rz * deg), torch.sin(rz * deg)
    zero = torch.zeros((), device=dev, dtype=dt)
    one = torch.ones((), device=dev, dtype=dt)

    Rx = torch.stack(
        [
            torch.stack([one, zero, zero]),
            torch.stack([zero, cx, -sx]),
            torch.stack([zero, sx, cx]),
        ]
    )
    Ry = torch.stack(
        [
            torch.stack([cy, zero, sy]),
            torch.stack([zero, one, zero]),
            torch.stack([-sy, zero, cy]),
        ]
    )
    Rz = torch.stack(
        [
            torch.stack([cz, -sz, zero]),
            torch.stack([sz, cz, zero]),
            torch.stack([zero, zero, one]),
        ]
    )
    R = Rz @ Rx @ Ry
    M3 = R @ torch.diag(s)  # scale then rotate

    piv = pivot.to(device=dev, dtype=dt)
    offset = piv + t - M3 @ piv  # p' = M3·p + (pivot + t − M3·pivot)

    M = torch.eye(4, device=dev, dtype=dt)
    M = M.clone()
    M[:3, :3] = M3
    M[:3, 3] = offset
    return M


def apply_transform(points: Tensor, normals: Tensor, mat: Tensor) -> tuple[Tensor, Tensor]:
    """Apply a 4×4 transform to points and (covariantly) to normals.

    Normals transform by the inverse-transpose of the 3×3 linear block and are
    renormalized, so the sampling direction stays perpendicular to the deformed
    boundary under anisotropic scale.

    Args:
        points: (P, 3) (x, y, z).
        normals: (P, 3) unit vectors.
        mat: (4, 4) transform from :func:`rst_matrix`.

    Returns:
        (points', normals') each (P, 3); normals' are unit length.
    """
    M3 = mat[:3, :3]
    pts = points @ M3.T + mat[:3, 3]
    nrm = normals @ torch.linalg.inv(M3)  # n' = (M3^{-T} n) → row-vec: n @ M3^{-1}
    nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return pts, nrm


# ── Contrast + cost ──────────────────────────────────────────────────────────


def boundary_contrast(
    volume: Tensor,
    points: Tensor,
    normals: Tensor,
    offset: float | Tensor = 1.0,
) -> Tensor:
    """Per-point BBR gradient contrast ``(white − grey)/(white + grey)``.

    Samples the target ``volume`` one ``offset`` inside (white/WM side,
    ``p − offset·n̂``) and one outside (grey/GM side, ``p + offset·n̂``) along the
    outward normal. Differentiable in ``points`` and ``normals``.

    Args:
        volume: (nz, ny, nx) target intensity volume.
        points: (P, 3) (x, y, z) voxel coordinates on the boundary.
        normals: (P, 3) unit outward normals.
        offset: sampling half-distance in voxels; scalar or (P,) per-point
            (stand-in for the paper's 0.3·thickness when no thickness map).

    Returns:
        (P,) contrast in [−1, 1]; undefined/degenerate points set to 0.
    """
    if isinstance(offset, Tensor):
        off = offset.to(device=points.device, dtype=points.dtype)[:, None]
    else:
        off = float(offset)
    white_pts = points - off * normals
    grey_pts = points + off * normals

    white = trilinear_interpolate(volume, white_pts[:, 0], white_pts[:, 1], white_pts[:, 2])
    grey = trilinear_interpolate(volume, grey_pts[:, 0], grey_pts[:, 1], grey_pts[:, 2])

    denom = white + grey
    contrast = (white - grey) / denom.clamp_min(1e-6)
    # Divisions by ~0 (both sides dark) or |c|>1 are meaningless → 0 contribution.
    bad = (denom.abs() < 1e-6) | (contrast.abs() > 1.0) | ~torch.isfinite(contrast)
    return torch.where(bad, torch.zeros_like(contrast), contrast)


def greve_fischl_cost(
    contrast: Tensor,
    *,
    reverse: bool = False,
    reduce: str = "mean",
    weight: Tensor | None = None,
) -> Tensor:
    """Greve–Fischl aggregation of per-point contrast into a scalar *cost*.

    ``cost = −reduce( tanh(Mv · contrast) )`` with ``Mv = −1`` when ``reverse``
    (target where GM is brighter than WM, e.g. T2*/EPI). Lower is better. The
    ``mean`` reduction (default) is scale-invariant across elements of different
    point counts — preferable to the paper's ``sum`` when many elements are
    optimized together.

    Args:
        contrast: (P,) per-point contrast from :func:`boundary_contrast`.
        reverse: flip the rewarded contrast sign.
        reduce: ``"mean"`` or ``"sum"``.
        weight: optional (P,) per-point weights (e.g. to down-weight a less
            reliable second boundary such as the pial/CSF edge).

    Returns:
        Scalar cost tensor (differentiable).
    """
    mv = -1.0 if reverse else 1.0
    t = torch.tanh(mv * contrast)
    if weight is not None:
        w = weight.to(dtype=t.dtype, device=t.device)
        agg = (w * t).sum()
        if reduce == "mean":
            agg = agg / w.sum().clamp_min(1e-12)
    else:
        agg = t.mean() if reduce == "mean" else t.sum()
    return -agg


def bbr_cost(
    volume: Tensor,
    points: Tensor,
    normals: Tensor,
    params: Tensor,
    *,
    pivot: Tensor | None = None,
    offset: float | Tensor = 1.0,
    reverse: bool = False,
    reduce: str = "mean",
    weight: Tensor | None = None,
) -> Tensor:
    """BBR cost of placing a boundary, as a function of a transform.

    Transforms ``points``/``normals`` by ``rst_matrix(params, pivot)``, samples
    the boundary contrast, and aggregates. Differentiable in ``params`` — the
    objective the optimizer minimizes.

    Args:
        volume: (nz, ny, nx) target volume.
        points: (P, 3) boundary points (x, y, z).
        normals: (P, 3) unit outward normals.
        params: (9,) transform parameters (see :func:`rst_matrix`).
        pivot: (3,) transform centre; defaults to the point-cloud centroid.
        offset: sampling half-distance in voxels (scalar or (P,)).
        reverse: pass through to :func:`greve_fischl_cost`.
        reduce: pass through to :func:`greve_fischl_cost`.
        weight: optional (P,) per-point weights.

    Returns:
        Scalar cost tensor (lower = better boundary placement).
    """
    if pivot is None:
        pivot = points.mean(dim=0)
    mat = rst_matrix(params, pivot)
    pts, nrm = apply_transform(points, normals, mat)
    contrast = boundary_contrast(volume, pts, nrm, offset)
    return greve_fischl_cost(contrast, reverse=reverse, reduce=reduce, weight=weight)


def identity_params(
    device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32
) -> Tensor:
    """The 9-parameter identity ``[0,0,0, 1,1,1, 0,0,0]``."""
    return torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)


# ── Reliability weighting (don't do BBR where the EPI has no edge) ────────────


def gradient_field(vol: Tensor) -> Tensor:
    """Central-difference gradient as a ``(3, nz, ny, nx)`` field of (x, y, z)
    components (x = derivative along the last axis), matching the point/normal
    coordinate convention."""
    gz = torch.zeros_like(vol)
    gy = torch.zeros_like(vol)
    gx = torch.zeros_like(vol)
    gz[1:-1] = (vol[2:] - vol[:-2]) * 0.5
    gy[:, 1:-1] = (vol[:, 2:] - vol[:, :-2]) * 0.5
    gx[:, :, 1:-1] = (vol[:, :, 2:] - vol[:, :, :-2]) * 0.5
    return torch.stack([gx, gy, gz], dim=0)


def _gradient_magnitude(vol: Tensor) -> Tensor:
    """Central-difference |∇vol| — edge strength, alignment-independent."""
    g = gradient_field(vol)
    return torch.sqrt((g * g).sum(dim=0))


def boundary_reliability(
    volume: Tensor,
    points: Tensor,
    *,
    grad_mag: Tensor | None = None,
    floor: float = 0.05,
) -> Tensor:
    """Per-boundary-point weight in ``[0, 1]`` from local EPI edge strength.

    A filled WM mask has boundary points everywhere on its surface, but the EPI
    only has a grey/white *edge* at some of them — elsewhere (dropout, low tSNR,
    a boundary lying in-plane) BBR would fit noise. Weighting each point by the
    EPI gradient magnitude at its location lets confident edges drive the fit
    while flat/dropout points contribute ~nothing; in the regularized warp their
    displacement is then filled in from reliable neighbours (the desired
    "default to no change where there's no signal" behaviour).

    The weight is ``g / (g + median(g>0))`` (soft, ~0.5 at the median edge, →1
    for strong edges, →0 for flat), lifted by ``floor`` so nothing is fully
    discarded. Alignment-independent (uses |∇EPI|, not the current contrast).

    Args:
        volume: (nz, ny, nx) target (EPI) volume.
        points: (P, 3) boundary points (x, y, z).
        grad_mag: optional precomputed |∇volume| (else computed here).
        floor: minimum weight.

    Returns:
        (P,) weights in [floor, 1].
    """
    if grad_mag is None:
        grad_mag = _gradient_magnitude(volume)
    g = trilinear_interpolate(grad_mag, points[:, 0], points[:, 1], points[:, 2])
    g = g.clamp_min(0)
    pos = g[g > 0]
    ref = pos.median() if pos.numel() > 0 else g.new_tensor(1.0)
    w = g / (g + ref + 1e-12)
    return w.clamp_min(floor)


# ── Edge target + normalized-gradient-field (NGF) cost ───────────────────────
#
# Generalizes the single WM boundary to *every* edge in the anatomy (brain edge,
# ventricles, subcortical, WM/GM). Because the EPI intensity polarity differs
# across edge types, the cost matches gradient *direction* (polarity-agnostic),
# not a signed contrast — the classic multi-modal NGF criterion.


def extract_edge_normals(
    anat: Tensor | np.ndarray,
    *,
    mask: Tensor | np.ndarray | None = None,
    blur: float = 1.0,
    percentile: float = 75.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Edge point cloud + gradient-direction normals from an anatomical image.

    Uses the image's own gradient as the geometry, so *all* boundaries become
    targets — no segmentation required. Blur first to bring fine anatomy toward
    the EPI's scale (edges the EPI can't resolve only add noise), then keep the
    voxels in the top ``100 − percentile`` % of gradient magnitude.

    Args:
        anat: (nz, ny, nx) anatomical image (ideally already in the target/EPI
            grid, so points land in EPI space — same pattern as the WM mask).
        mask: optional (nz, ny, nx) brain mask to exclude background edges.
        blur: Gaussian sigma (voxels) applied before the gradient (0 = none).
        percentile: keep gradient magnitudes at/above this percentile (default 75
            → strongest 25 % of edges).
        device, dtype: placement of the returned tensors.

    Returns:
        points: (P, 3) edge points (x, y, z) voxel coordinates.
        normals: (P, 3) unit gradient-direction normals (∇anat / |∇anat|).
    """
    from scipy import ndimage

    v = anat.detach().cpu().numpy() if isinstance(anat, Tensor) else np.asarray(anat, float)
    if v.ndim != 3:
        raise ValueError(f"anat must be 3D (nz, ny, nx); got {v.shape}")
    if blur > 0:
        v = ndimage.gaussian_filter(v.astype(np.float64), blur)
    gz, gy, gx = np.gradient(v)
    mag = np.sqrt(gx**2 + gy**2 + gz**2)

    sel = np.ones(v.shape, dtype=bool)
    if mask is not None:
        m = mask.detach().cpu().numpy() if isinstance(mask, Tensor) else np.asarray(mask)
        sel &= m > 0
    thr = np.percentile(mag[sel], percentile) if sel.any() else 0.0
    sel &= mag >= max(thr, 1e-12)
    kk, jj, ii = np.nonzero(sel)
    if kk.size == 0:
        raise ValueError("no edge points survived the threshold")

    m = mag[kk, jj, ii]
    normals = np.stack([gx[kk, jj, ii] / m, gy[kk, jj, ii] / m, gz[kk, jj, ii] / m], axis=1)
    points = np.stack([ii, jj, kk], axis=1).astype(np.float64)
    return (
        torch.as_tensor(points, dtype=dtype, device=device),
        torch.as_tensor(normals, dtype=dtype, device=device),
    )


def ngf_eta(grad_field: Tensor) -> float:
    """Default edge-presence scale: the median non-zero EPI |∇| over the volume."""
    mvol = torch.sqrt((grad_field * grad_field).sum(dim=0))
    pos = mvol[mvol > 0]
    return float(pos.median()) if pos.numel() > 0 else 1.0


def ngf_score_at(
    grad_field: Tensor,
    pts: Tensor,
    nrm: Tensor,
    *,
    eta: float,
    weight: Tensor | None = None,
) -> Tensor:
    """NGF cost at *already-positioned* points/normals (no transform applied).

    Shared core of the affine (:func:`ngf_cost`) and warp (RBR) stages: samples the
    EPI gradient at ``pts``, rewards directional alignment ``(n̂·ĝ)²`` weighted by
    EPI edge presence ``mag/(mag+eta)``. Differentiable in ``pts`` (and ``nrm``).
    """
    gx = trilinear_interpolate(grad_field[0], pts[:, 0], pts[:, 1], pts[:, 2])
    gy = trilinear_interpolate(grad_field[1], pts[:, 0], pts[:, 1], pts[:, 2])
    gz = trilinear_interpolate(grad_field[2], pts[:, 0], pts[:, 1], pts[:, 2])
    g = torch.stack([gx, gy, gz], dim=1)  # (P, 3) EPI gradient at the points
    mag = g.norm(dim=1)
    ghat = g / mag.clamp_min(1e-8)[:, None]  # unit EPI gradient direction
    score = (nrm * ghat).sum(dim=1) ** 2  # (P,) direction alignment cos²θ ∈ [0, 1]

    # Weight by EPI edge presence so points on the STRONG part of an edge localize
    # it (a plain w.sum() normalization would cancel that and leave a straight
    # edge un-localizable along its normal). Bounded in [0, 1).
    w = mag / (mag + eta)
    if weight is not None:
        w = w * weight.to(dtype=w.dtype, device=w.device)
    return -(w * score).mean()


def ngf_cost(
    grad_field: Tensor,
    points: Tensor,
    normals: Tensor,
    params: Tensor,
    *,
    pivot: Tensor | None = None,
    eta: float | None = None,
    weight: Tensor | None = None,
) -> Tensor:
    """Normalized-gradient-field cost: align anat edge normals to EPI gradients.

    Transforms the anat edge points/normals, samples the (precomputed) EPI
    gradient there, and rewards *directional* alignment ``(n̂_anat · ĝ_epi)²`` —
    polarity-agnostic, so it works across all edge types at once. Each point is
    weighted by the local EPI gradient magnitude, so anat edges the EPI can't see
    contribute ~nothing (built-in reliability). Differentiable in ``params``.

    Args:
        grad_field: (3, nz, ny, nx) precomputed EPI gradient (:func:`gradient_field`).
        points, normals: anat edge points (x, y, z) and unit gradient normals.
        params: (9,) transform parameters (see :func:`rst_matrix`).
        pivot: transform centre; defaults to the point-cloud centroid.
        eta: reference EPI |∇| scale for the edge-presence weight ``mag/(mag+eta)``;
            defaults to the median non-zero EPI |∇| over the volume. Points where
            the EPI has a strong edge (mag ≫ eta) count ~fully; flat points ~0.
        weight: optional (P,) extra per-point weights.

    Returns:
        Scalar cost tensor (lower = better edge alignment).
    """
    if pivot is None:
        pivot = points.mean(dim=0)
    if eta is None:
        eta = ngf_eta(grad_field)
    mat = rst_matrix(params, pivot)
    pts, nrm = apply_transform(points, normals, mat)
    return ngf_score_at(grad_field, pts, nrm, eta=eta, weight=weight)


# ── Global optimization (single-element BBR = affine refinement) ─────────────

# Free-parameter indices into the 9-vector [rx,ry,rz, sx,sy,sz, tx,ty,tz] per
# named DoF mode. "pe" is the phase-encode-axis distortion mode (translate+scale
# in y) motivating RBR; "rigid" is the classic bbregister refinement.
MODE_FREE_PARAMS: dict[str, list[int]] = {
    "rigid": [0, 1, 2, 6, 7, 8],  # 3 rot + 3 trans
    "shift": [6, 7, 8],  # translation only
    "similarity": [0, 1, 2, 3, 4, 5, 6, 7, 8],  # rot + scale + trans
    "pe": [7, 4],  # PE-axis translate (ty) + scale (sy) — assumes y is PE
    "pe_shift": [7],  # PE-axis translate only
}

# Translation-parameter indices, for the coarse capture-range search.
_TRANS_IDX = {6, 7, 8}


def auto_polarity(
    volume: Tensor,
    points: Tensor,
    normals: Tensor,
    *,
    offset: float | Tensor = 1.0,
    pivot: Tensor | None = None,
    weight: Tensor | None = None,
) -> bool:
    """Pick ``reverse`` by which sign gives the lower cost at the identity.

    Returns True (GM brighter than WM, e.g. T2*/EPI) when reversing the rewarded
    contrast fits the data better — so callers don't have to know the target
    contrast a priori.
    """
    p = identity_params(points.device, points.dtype)
    c_fwd = bbr_cost(volume, points, normals, p, pivot=pivot, offset=offset, weight=weight).item()
    c_rev = bbr_cost(
        volume, points, normals, p, pivot=pivot, offset=offset, reverse=True, weight=weight
    ).item()
    return c_rev < c_fwd


def correct_sign_fraction(
    volume: Tensor,
    points: Tensor,
    normals: Tensor,
    params: Tensor,
    *,
    pivot: Tensor | None = None,
    offset: float | Tensor = 1.0,
    reverse: bool = False,
) -> float:
    """Fraction of boundary points whose contrast has the rewarded sign — a
    scale-free QC number (higher = boundary better seated on the edge)."""
    if pivot is None:
        pivot = points.mean(dim=0)
    with torch.no_grad():
        pts, nrm = apply_transform(points, normals, rst_matrix(params, pivot))
        c = boundary_contrast(volume, pts, nrm, offset)
        mv = -1.0 if reverse else 1.0
        nz = c != 0
        if nz.sum() == 0:
            return 0.0
        return ((mv * c[nz]) > 0).float().mean().item()


def optimize_bbr(
    volume: Tensor,
    points: Tensor,
    normals: Tensor,
    *,
    mode: str = "rigid",
    offset: float | Tensor = 1.0,
    reverse: bool = False,
    pivot: Tensor | None = None,
    weight: Tensor | None = None,
    cost_fn: Callable[[Tensor], Tensor] | None = None,
    coarse_range: float = 4.0,
    coarse_step: float = 1.0,
    iters: int = 300,
    lr: float = 0.2,
    tol: float = 1e-5,
    verbose: bool = False,
) -> dict:
    """Optimize a single global transform to seat a boundary on the target edge.

    This is single-element BBR — the classic ``bbregister``/``flirt -cost bbr``
    affine refinement, and the primitive RBR recurses on. A cheap coordinate-
    descent coarse search over the free translation axes first brings the
    boundary into BBR's (limited) capture range, then Adam refines all free
    parameters of ``mode``.

    Args:
        volume: (nz, ny, nx) target intensity volume (e.g. the EPI).
        points, normals: boundary point cloud (x, y, z) and unit outward normals.
        mode: key of :data:`MODE_FREE_PARAMS` (which parameters are free).
        offset: sampling half-distance in voxels (scalar or (P,)).
        reverse: rewarded-contrast polarity (see :func:`greve_fischl_cost`).
        pivot: transform centre; defaults to the point-cloud centroid.
        weight: optional (P,) per-point weights.
        cost_fn: optional ``params → scalar`` cost overriding the default signed
            BBR contrast (e.g. an NGF edge cost, or a WM+edge blend). When given,
            ``volume``/``offset``/``reverse``/``weight`` are unused by the fit.
        coarse_range, coarse_step: half-width and step (voxels) of the coarse
            translation search; set ``coarse_range=0`` to skip it.
        iters, lr, tol: Adam iteration cap, learning rate, relative-cost tol. The
            LR is a step in raw parameter units — voxels for translations, degrees
            for rotations. A sharp-welled cost (the dense tissue-synthesis term)
            overshoots at the default 0.2; lower it (e.g. 0.01) so it converges.
        verbose: print coarse/refine progress.

    Returns:
        dict with ``params`` (9,), ``matrix`` (4,4), ``init_cost``,
        ``final_cost``, ``n_iter``.
    """
    if mode not in MODE_FREE_PARAMS:
        raise ValueError(f"unknown mode {mode!r}; choose from {sorted(MODE_FREE_PARAMS)}")
    dev, dt = points.device, points.dtype
    if pivot is None:
        pivot = points.mean(dim=0)
    free = MODE_FREE_PARAMS[mode]

    if cost_fn is not None:
        cost_of = cost_fn
    else:

        def cost_of(p: Tensor) -> Tensor:
            return bbr_cost(
                volume,
                points,
                normals,
                p,
                pivot=pivot,
                offset=offset,
                reverse=reverse,
                weight=weight,
            )

    p9 = identity_params(dev, dt)
    init_cost = cost_of(p9).item()

    # Coarse coordinate-descent over free translation axes (capture range).
    free_trans = [i for i in free if i in _TRANS_IDX]
    if coarse_range > 0 and free_trans:
        grid = torch.arange(-coarse_range, coarse_range + 1e-6, coarse_step, device=dev, dtype=dt)
        with torch.no_grad():
            for _ in range(2):  # two sweeps let the axes settle jointly
                for idx in free_trans:
                    best_v, best_c = p9[idx].item(), cost_of(p9).item()
                    for g in grid:
                        trial = p9.clone()
                        trial[idx] = g
                        c = cost_of(trial).item()
                        if c < best_c:
                            best_c, best_v = c, g.item()
                    p9[idx] = best_v
        if verbose:
            print(f"  coarse: cost {init_cost:.4f} → {cost_of(p9).item():.4f}")

    # Adam refine over all free params (mask fixed grads to hold them at identity).
    p9 = p9.clone().requires_grad_(True)
    fixed = [i for i in range(9) if i not in free]
    opt = torch.optim.Adam([p9], lr=lr)
    prev = cost_of(p9).item()
    n_iter = 0
    for step in range(1, iters + 1):
        opt.zero_grad()
        c = cost_of(p9)
        c.backward()
        if p9.grad is not None and fixed:
            p9.grad[fixed] = 0.0
        opt.step()
        n_iter = step
        cur = c.item()
        if abs(prev - cur) < tol * (abs(prev) + 1e-12):
            break
        prev = cur
    final = p9.detach()
    final_cost = cost_of(final).item()
    if verbose:
        print(f"  refine: cost → {final_cost:.4f} in {n_iter} iters")

    return {
        "params": final,
        "matrix": rst_matrix(final, pivot).detach(),
        "init_cost": init_cost,
        "final_cost": final_cost,
        "n_iter": n_iter,
    }
