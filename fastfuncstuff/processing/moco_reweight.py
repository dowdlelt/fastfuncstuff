"""Data-driven weight refinement for ffs_moco (``-reweight``).

A pre-pass, conceptually like ``-twopass``: instead of trusting every bright
voxel in the registration weight, it *looks at the data* to learn which regions
move consistently with the head and drops the rest (bright artifacts, vessels,
ghosts that pull the rigid fit without tracking real motion).

The base is tiled into space-filling **bloks** (the same lattice ffs uses for
LPC/LPA, ``cost_blok.assign_bloks``). The agreement test is built on one fact
about small patches: a ~5-voxel-radius blok **cannot observe rotation** — over
such a small region a rotation's displacement field is nearly constant, so it is
collinear with a translation and the optimizer can't separate them. A small
patch *can* reliably estimate its **local displacement** (a 3-DOF translation:
"my content moved by this much"). So we:

1. Estimate the **global** 6-DOF rigid motion per TR (the whole brain has ample
   leverage to see rotation — this is well posed). This is the consensus.
2. Estimate each patch's **3-DOF local displacement** per TR (well posed).
3. For each patch, predict the displacement the *global* motion implies **at that
   patch's location** (``M_global @ x_patch - x_patch``) — this bakes in the
   location-dependent rotation, so good patches on opposite sides of the head are
   both expected to move as observed, not flagged for moving oppositely.
4. Detrend (Legendre polort) and correlate measured-vs-predicted displacement per
   axis. A patch that agrees on enough axes is kept; one that is uncorrelated or
   anti-correlated (moves against what the head is doing) is zeroed out.

The per-patch rigid parametrisation and coordinate origin match the whole-image
pass, so the global estimate here is directly comparable to the main estimation.
This is ffs_moco-only and rigid; it does not touch ffs_allineate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from fastfuncstuff.design.builder import legendre_polynomials
from fastfuncstuff.utils import linalg_device, warn_mps_float32_precision

from .affine import _build_homo_coords, identity_params, params_to_matrix_batched
from .cost_blok import assign_bloks
from .interp import _separable_resample_3d

# Per-patch resample kernel: cubic is cheap (floor convention, 4 taps/axis) and
# plenty accurate for a diagnostic motion estimate. _separable_resample_3d has no
# "linear", and the per-patch params only need to be good enough to correlate.
_REWEIGHT_INTERP = "cubic"


@dataclass
class ReweightResult:
    """Outcome of the reweight pre-pass."""

    weight: Tensor  # (nz, ny, nx) refined weight (== original if not applied)
    patch_labels: Tensor  # (nz, ny, nx) int: random distinct id per kept patch, 0 else
    n_kept: int
    n_patches: int
    applied: bool  # False if the low-motion guard skipped reweighting


def _auto_polort(nt: int, tr: float) -> int:
    """AFNI-style drift degree: 1 + floor(run_seconds / 150), clamped to fit nt."""
    deg = 1 + int(math.floor(nt * max(tr, 1e-3) / 150.0))
    return max(1, min(deg, nt - 2)) if nt >= 3 else 0


def _detrend_columns(mat: Tensor, poly: Tensor) -> Tensor:
    """Residualise each column of ``mat`` (nt, P) against the polynomial basis.

    ``poly`` is (nt, deg+1). Uses a least-squares projection so the returned
    columns have the drift (and mean) removed.
    """
    # beta = pinv(poly) @ mat ; resid = mat - poly @ beta
    sol = torch.linalg.lstsq(poly, mat).solution
    return mat - poly @ sol


def _estimate_motion(
    base: Tensor,
    timeseries: Tensor,
    weight0: Tensor,
    derivs: Tensor,
    bi: Tensor,
    coords_v: Tensor,
    P: int,
    ndof: int,
    max_iter: int,
    device: torch.device,
    disable_pbar: bool,
    desc: str,
) -> Tensor:
    """Per-group Gauss-Newton WLS for every TR -> params ``(nt, P, ndof)``.

    Generalises the per-blok solver to either:
      - the **global** motion (``bi`` all zeros, ``P=1``, ``ndof=6``): the whole
        valid-voxel set as one group — well posed for rotation, and
      - per-patch **local displacement** (dense ``bi``, ``ndof=3``): translation
        only, the only DOF a small patch can observe.

    ``bi`` (Nv,) is the group id of each valid voxel; ``coords_v`` (4, Nv) their
    homogeneous indices. The normal equations are segment-reductions over groups,
    so all groups solve together each iteration; only valid voxels are resampled.
    ``ndof`` selects the leading Jacobian rows (3 = translations, 6 = full rigid),
    matching the param layout [dx, dy, dz, rz, rx, ry, ...].
    """
    dtype = base.dtype
    nt = timeseries.shape[0]
    vol_shape = tuple(base.shape)

    # base / weight / Jacobian restricted to the valid (in-blok) voxels carried
    # by coords_v; recover their flat indices to gather.
    flat_idx = _flat_index_from_coords(coords_v, vol_shape)
    base_v = base.reshape(-1)[flat_idx]  # (Nv,)
    weight_v = weight0.reshape(-1)[flat_idx]  # (Nv,)
    WJ_v = derivs[:ndof, flat_idx] * weight_v[None]  # (ndof, Nv)

    # Per-group normal matrix JtWJ (P, ndof, ndof), float64 (matches whole-image
    # GN). MPS has no float64, so the normal equations and per-group solve run in
    # float32 there (use -device cpu for full-precision reweighting).
    solve_dtype = torch.float32 if device.type == "mps" else torch.float64
    if device.type == "mps":
        warn_mps_float32_precision("moco reweight GN solve")
    JtWJ = torch.zeros(P, ndof, ndof, dtype=solve_dtype, device=device)
    WJ64 = WJ_v.to(solve_dtype)
    for a in range(ndof):
        for b in range(a, ndof):
            seg = torch.zeros(P, dtype=solve_dtype, device=device)
            seg.index_add_(0, bi, WJ64[a] * WJ64[b])
            JtWJ[:, a, b] = seg
            if a != b:
                JtWJ[:, b, a] = seg
    eps = 1e-6 * JtWJ.diagonal(dim1=1, dim2=2).mean().clamp(min=1e-30)
    JtWJ_reg = JtWJ + eps * torch.eye(ndof, dtype=solve_dtype, device=device)[None]

    out = torch.zeros(nt, P, ndof, dtype=torch.float64)
    ident = identity_params(device=device, dtype=dtype)

    pbar = tqdm(
        range(nt),
        desc=desc,
        disable=disable_pbar,
        unit="vol",
        ncols=80,
        leave=True,
    )
    for t in pbar:
        source = timeseries[t].to(device=device, dtype=dtype)
        params = ident[None].repeat(P, 1).clone()  # (P, 12)
        for _ in range(max_iter):
            mats = params_to_matrix_batched(params)  # (P, 4, 4)
            M_vox = mats[bi]  # (Nv, 4, 4)
            src = torch.einsum("nij,jn->ni", M_vox, coords_v)  # (Nv, 4)
            warped = _separable_resample_3d(
                source, src[:, 0], src[:, 1], src[:, 2], _REWEIGHT_INTERP
            )  # (Nv,)
            residual = weight_v * (base_v - warped)  # (Nv,)
            rhs = torch.zeros(P, ndof, dtype=solve_dtype, device=device)
            rhs.index_add_(0, bi, (WJ_v * residual[None]).t().to(solve_dtype))  # (P, ndof)
            dp = torch.linalg.solve(JtWJ_reg, rhs)  # (P, ndof)
            params[:, :ndof] += dp.to(dtype)
        # .cpu() before .double(): MPS has no float64.
        out[t] = params[:, :ndof].detach().cpu().double()

    return out


def _flat_index_from_coords(coords_v: Tensor, vol_shape: tuple[int, int, int]) -> Tensor:
    """Recover row-major flat indices from homogeneous (x, y, z) coords.

    ``coords_v`` is (4, Nv) with rows [x, y, z, 1] in integer voxel positions
    (the subset of ``_build_homo_coords`` kept for valid voxels). Flat index into
    a (nz, ny, nx).reshape(-1) volume is ``z*ny*nx + y*nx + x``.
    """
    _nz, ny, nx = vol_shape
    x = coords_v[0].round().long()
    y = coords_v[1].round().long()
    z = coords_v[2].round().long()
    return z * (ny * nx) + y * nx + x


def compute_reweight(
    base: Tensor,
    timeseries: Tensor,
    weight0: Tensor,
    derivs: Tensor,
    global_matrices: Tensor,
    *,
    voxdims: tuple[float, float, float],
    tr: float,
    bloktype: str = "rhdd",
    blokrad: float = 0.0,
    minparams: int = 2,
    rmin: float = 0.1,
    polort: int = -1,
    max_iter: int = 6,
    min_motion: float = 0.05,
    device: torch.device | None = None,
    verb: int = 1,
) -> ReweightResult:
    """Refine ``weight0`` by dropping patches that don't move with the head.

    Args:
        base: (nz, ny, nx) reference volume on ``device``.
        timeseries: (nt, nz, ny, nx) input series (CPU or device).
        weight0: (nz, ny, nx) original registration weight on ``device``.
        derivs: (6, N) spatial derivative images of the base (from
            ``ffs_moco.compute_derivative_images``); shared with the main pass.
        global_matrices: (nt, 4, 4) voxel-space rigid transforms from the fast
            whole-image fit (base->source pull, same convention as the per-patch
            solver). This is the consensus motion — reused rather than recomputed.
        voxdims: (dx, dy, dz) voxel sizes in mm (x is the fastest axis).
        tr: repetition time in seconds (for the auto drift degree).
        bloktype: "rhdd" (default), "tohd", or "cube".
        blokrad: blok radius in mm; 0 auto-sizes to ~555 voxels.
        minparams: keep a patch if its measured displacement agrees with the
            global-motion prediction on at least this many of the 3 axes.
        rmin: per-axis correlation threshold for "agrees".
        polort: drift degree for detrending; <0 auto (1 + floor(nt*TR/150)).
        max_iter: GN iterations for the cheap per-patch displacement estimate.
        min_motion: skip reweighting if the global motion (predicted displacement,
            voxels) stays below this — a still subject has no signal to separate
            good patches from bad.
        device: torch device (defaults to base's).
        verb: verbosity.

    Returns:
        :class:`ReweightResult`.
    """
    if device is None:
        device = base.device
    vol_shape = tuple(base.shape)
    nt = timeseries.shape[0]
    dtype = base.dtype

    # Tile into bloks over the weighted (in-mask) region.
    blokset = assign_bloks(
        vol_shape,
        voxdims=voxdims,
        bloktype=bloktype,
        blokrad=None if blokrad <= 0 else blokrad,
        mask=(weight0 > 0).to(device),
        device=device,
    )
    idx = blokset.index.to(device)  # (N,) global blok id, -1 excluded

    valid = idx >= 0
    if int(valid.sum()) == 0 or blokset.n_populated < 2:
        if verb >= 1:
            print("  -reweight: too few populated patches; keeping original weight")
        return ReweightResult(
            weight=weight0,
            patch_labels=torch.zeros(vol_shape, dtype=torch.int32, device=device),
            n_kept=0,
            n_patches=blokset.n_populated,
            applied=False,
        )

    # Remap surviving global blok ids to a dense 0..P-1 space.
    uniq, bi_full = torch.unique(idx[valid], return_inverse=True)
    P = int(uniq.numel())
    homo = _build_homo_coords(vol_shape, device, dtype)  # (4, N)
    coords_v = homo[:, valid]  # (4, Nv)
    bi = bi_full  # (Nv,) dense blok id per valid voxel

    disable_pbar = verb == 0

    # Per-patch 3-DOF local displacement (translation only — the observable DOF
    # of a small patch). The global 6-DOF motion (the consensus) is supplied by
    # the caller's fast whole-image fit, so we don't recompute it here.
    tmeas = _estimate_motion(
        base,
        timeseries,
        weight0,
        derivs,
        bi,
        coords_v,
        P,
        3,
        max_iter,
        device,
        disable_pbar,
        desc="  Reweight patches",
    )  # (nt, P, 3) float64 cpu

    # --- Predicted vs measured patch displacement ---------------------------
    # Patch centroids (x, y, z) in voxel index space, on CPU for the stats below.
    # These centroid stats are float64 and end up on CPU anyway (xp.cpu() below),
    # so compute them on linalg_device — i.e. CPU on MPS, which has no float64.
    ld = linalg_device(device)
    bi_ld = bi.to(ld)
    coords_ld = coords_v.to(ld)
    cnt = torch.zeros(P, dtype=torch.float64, device=ld)
    cnt.index_add_(0, bi_ld, torch.ones(bi.numel(), dtype=torch.float64, device=ld))
    cen = torch.zeros(P, 3, dtype=torch.float64, device=ld)
    for j in range(3):
        s = torch.zeros(P, dtype=torch.float64, device=ld)
        s.index_add_(0, bi_ld, coords_ld[j].double())
        cen[:, j] = s / cnt.clamp(min=1)
    xp = torch.cat([cen, torch.ones(P, 1, dtype=torch.float64, device=ld)], dim=1)  # (P,4)
    xp = xp.cpu()

    # Displacement the GLOBAL motion implies at each patch centre, per TR
    # (Mg maps base->source; displacement = transformed - original, x/y/z).
    Mg = global_matrices.to(dtype=torch.float64, device="cpu")  # (nt, 4, 4)
    dpred = torch.einsum("tij,pj->tpi", Mg, xp)[:, :, :3] - xp[None, :, :3]  # (nt, P, 3)

    # Low-motion guard: if the head barely moved, the predicted displacement has
    # no temporal signal to separate good patches from bad — leave the weight be.
    motion_amp = float(dpred.std(dim=0).max())
    if motion_amp < min_motion:
        if verb >= 1:
            print(
                f"  -reweight: global motion below threshold "
                f"({motion_amp:.3g} vox); keeping original weight"
            )
        return ReweightResult(
            weight=weight0,
            patch_labels=torch.zeros(vol_shape, dtype=torch.int32, device=device),
            n_kept=P,
            n_patches=P,
            applied=False,
        )

    # Detrend both, correlate per axis, count agreeing axes per patch.
    deg = _auto_polort(nt, tr) if polort < 0 else min(polort, nt - 2)
    poly = torch.from_numpy(legendre_polynomials(nt, max(deg, 0)).astype(np.float64))
    keep_count = torch.zeros(P, dtype=torch.int64)
    for j in range(3):
        md = _detrend_columns(tmeas[:, :, j], poly)  # (nt, P)
        pd = _detrend_columns(dpred[:, :, j], poly)
        mc = md - md.mean(dim=0, keepdim=True)
        pc = pd - pd.mean(dim=0, keepdim=True)
        r = (mc * pc).sum(dim=0) / (mc.norm(dim=0) * pc.norm(dim=0) + 1e-20)  # (P,)
        keep_count += (r >= rmin).to(torch.int64)

    # Keep stats are computed on CPU (small); move to the blok device for scatter.
    keep = (keep_count >= minparams).to(device)  # (P,)
    n_kept = int(keep.sum())

    # Refined weight: keep the original weight ONLY inside kept patches; zero
    # everywhere else (rejected patches, unassigned/pruned bloks, and the rim —
    # rim weight where the brain leaves the FOV is exactly what we want gone). So
    # the weight map matches the patch label map one-to-one.
    keep_vox = torch.zeros(idx.shape, dtype=torch.bool, device=device)
    keep_vox[valid] = keep[bi]
    weight1 = weight0 * keep_vox.reshape(vol_shape).to(weight0.dtype)

    # Patch label map: a random distinct positive id per kept patch (0 elsewhere).
    labels_blok = torch.zeros(P, dtype=torch.int32, device=device)
    kept_ids = keep.nonzero(as_tuple=True)[0]
    if kept_ids.numel() > 0:
        perm = torch.randperm(kept_ids.numel(), device=device).to(torch.int32) + 1
        labels_blok[kept_ids] = perm
    labels_flat = torch.zeros(idx.shape, dtype=torch.int32, device=device)
    labels_flat[valid] = labels_blok[bi]
    patch_labels = labels_flat.reshape(vol_shape)

    if verb >= 1:
        print(
            f"  -reweight: kept {n_kept}/{P} patches "
            f"(bloktype={bloktype}, polort={deg}, agree {minparams}/3 axes, rmin={rmin})"
        )

    return ReweightResult(
        weight=weight1,
        patch_labels=patch_labels,
        n_kept=n_kept,
        n_patches=P,
        applied=True,
    )
