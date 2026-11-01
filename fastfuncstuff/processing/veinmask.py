"""Vein candidates from an EPI mean and the anat's tissue fractions.

Where to let spatial phase regression (ffs_phasereg -spr -spr_mask) borrow phase,
without a GRE/SWI. The anat says what each EPI voxel holds (GM/WM/CSF fractions);
the EPI mean, divided by a smooth bias field, should then look like
``mu_gm*c1 + mu_wm*c2 + mu_csf*c3``. A vein in a GM voxel dephases at the EPI's TE
and comes out darker than that prediction; a local-mean comparison alone cannot
tell it from white matter or a tissue boundary, which the tissue model can.

Two models. 'global' (:func:`tissue_darkness`): one GM/WM/CSF level for the brain
times a smooth bias. Its ratio map carries slab-wide structure (dark bottom slices, a
GM ribbon brighter than one level allows), so a fixed cut means different things in
different places. 'local' (:func:`local_tissue_darkness`, 2026-10-14): the three levels
are fitted per voxel over a Gaussian neighbourhood, which flattens the baseline (on
ds003427 the share of cortex reading >25% brighter than predicted fell from 11-13% to
4-5%). Then shape (:func:`dark_shapes`) and a ridge-following mask
(:func:`ridge_follow_mask`): pial veins at the brain edge and in sulci are plates,
vessels running through slices are tubes, punctate veins blobs. See the fmri_wiki note
"Dark vein map from EPI" for the knobs and the ds003427 numbers.
"""

from __future__ import annotations

import torch
from torch import Tensor

from fastfuncstuff.stats.smooth3d import _gaussian_kernel_1d, blur_axis
from fastfuncstuff.unique import sorted_unique
from fastfuncstuff.utils import to_linalg_f64


def masked_blur(x: Tensor, weight: Tensor, sigma_vox: tuple[float, float, float]) -> Tensor:
    """Gaussian average of ``x`` over ``weight`` (normalised convolution).

    Voxels with zero weight neither contribute nor drag the result toward zero, so
    the smooth field is defined up to the mask edge."""
    num, den = x * weight, weight.to(x.dtype)
    for ax, s in enumerate(sigma_vox):
        if s > 0:
            k = _gaussian_kernel_1d(s, x.dtype, x.device)
            num, den = blur_axis(num, k, ax), blur_axis(den, k, ax)
    return num / den.clamp_min(1e-3)


def tissue_darkness(
    mean: Tensor,
    tissue: Tensor,
    fit_mask: Tensor,
    sigma_vox: tuple[float, float, float],
    n_iter: int = 8,
    trim: float = 2.5,
) -> tuple[Tensor, Tensor, Tensor]:
    """``mean / (bias * tissue prediction)``, centred so typical tissue reads 1.

    ``tissue`` is ``(3, *mean.shape)``: GM, WM, CSF fractions on the EPI grid. Alternates
    the three tissue levels (least squares) with a smooth multiplicative bias
    (``sigma_vox``). Voxels further than ``trim`` robust SDs from the median log
    ratio leave the next fit on BOTH sides: trimming only the dark tail (the veins)
    biases the levels up every round, so typical GM would read darker than 1.

    Returns ``(ratio, expected, levels)``: ``expected = bias * prediction`` in EPI
    units (what the voxel should look like), ``levels`` the fitted GM/WM/CSF
    intensities before bias."""
    dt = torch.float64
    m = mean.to(dt)
    c = tissue.to(dt)
    flat_c = c.reshape(3, -1)
    use = fit_mask & (m > 0) & (c.sum(0) > 0.5)
    w = use.clone()
    bias = torch.ones_like(m)
    levels = torch.zeros(3, dtype=dt, device=m.device)
    expected = torch.ones_like(m)
    log_r = torch.zeros_like(m)
    for _ in range(n_iter):
        sel = w.reshape(-1)
        a = flat_c[:, sel].T
        b = (m / bias).reshape(-1)[sel]
        levels = torch.linalg.lstsq(a, b[:, None]).solution[:, 0].clamp_min(0)
        pred = torch.einsum("k,k...->...", levels, c).clamp_min(1e-6)
        wf = w.to(dt)
        bias = masked_blur(m * pred, wf, sigma_vox) / masked_blur(
            pred * pred, wf, sigma_vox
        ).clamp_min(1e-12)
        expected = (bias * pred).clamp_min(1e-6)
        log_r = torch.log(m.clamp_min(1e-6) / expected)
        r = log_r[use]
        med = r.median()
        mad = 1.4826 * (r - med).abs().median()
        w = use & ((log_r - med).abs() <= trim * mad)
    med = log_r[use].median()
    expected = expected * torch.exp(med)
    ratio = torch.where(use, torch.exp(log_r - med), torch.ones_like(m))
    return ratio.to(mean.dtype), expected.to(mean.dtype), levels


def _dilate6(x: Tensor) -> Tensor:
    out = x.clone()
    for ax in range(3):
        for s in (1, -1):
            sh = torch.roll(x, s, ax)
            idx = [slice(None)] * 3
            idx[ax] = slice(0, 1) if s == 1 else slice(-1, None)
            sh[tuple(idx)] = False
            out |= sh
    return out


def vein_region(tissue: Tensor, brain: Tensor, gm_min: float = 0.3, gm_core: float = 0.5) -> Tensor:
    """Where a vein that matters for BOLD can be: GM (fraction >= ``gm_min``), or CSF
    touching cortex (CSF the largest class, next to a voxel with GM >= ``gm_core``):
    the pial veins in the sulci. WM is left out on purpose -- deep WM veins carry no
    cortical response, and dark WM is the confound this model exists to avoid."""
    gm, wm, csf = tissue[0], tissue[1], tissue[2]
    pial = (csf > torch.maximum(gm, wm)) & _dilate6(gm >= gm_core)
    return brain & ((gm >= gm_min) | pial)


def vein_mask(ratio: Tensor, region: Tensor, darker: float, dilate: int = 1) -> Tensor:
    """Region voxels darker than predicted by more than ``darker`` (0.25 = a quarter
    darker), grown ``dilate`` voxels (6-connected) to forgive residual EPI-anat
    misregistration, as the GRE mask it stands in for needed."""
    m = region & (ratio < 1.0 - darker)
    for _ in range(dilate):
        m = _dilate6(m)
    return m


def local_tissue_darkness(
    mean: Tensor,
    tissue: Tensor,
    fit_mask: Tensor,
    sigma_vox: tuple[float, float, float],
    n_iter: int = 6,
    trim: float = 2.5,
    prior: float = 0.05,
) -> tuple[Tensor, Tensor]:
    """``mean / local tissue prediction``, centred so typical tissue reads 1.

    Per voxel, a Gaussian-weighted (``sigma_vox``) least-squares fit of the EPI on the
    neighbours' GM/WM/CSF fractions (``tissue``, ``(3, *mean.shape)``): the levels
    absorb coil shading and any spatial drift of tissue contrast. A class absent nearby
    is tied to the global level by a ridge of weight ``prior`` (relative to the local
    normal matrix's trace), so its level is not arbitrary. Robust trimming on both
    sides (``trim`` MADs of the log ratio) keeps veins out of their own reference.
    A pial vein at the brain edge is compared with the GM beside it (the fit only sees
    ``fit_mask``), and a vein in bright CSF with CSF. Returns ``(ratio, expected)``."""
    dt = torch.float64
    m, c = to_linalg_f64(mean), to_linalg_f64(tissue)
    use = fit_mask & (m > 0) & (c.sum(0) > 0.5)
    sel = use.reshape(-1)
    glob = torch.linalg.lstsq(c.reshape(3, -1)[:, sel].T, m.reshape(-1)[sel][:, None]).solution
    glob = glob[:, 0].clamp_min(0)
    eye = torch.eye(3, dtype=dt, device=m.device)
    w = use.clone()
    lr = torch.zeros_like(m)
    expected = torch.ones_like(m)
    for _ in range(n_iter):
        wf = w.to(dt)
        a = torch.stack(
            [
                torch.stack([masked_blur(c[k] * c[j], wf, sigma_vox) for j in range(3)], -1)
                for k in range(3)
            ],
            -2,
        )
        b = torch.stack([masked_blur(c[k] * m, wf, sigma_vox) for k in range(3)], -1)
        lam = prior * a.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-12) / 3
        lev = torch.linalg.solve(
            a + lam[..., None, None] * eye, (b + lam[..., None] * glob)[..., None]
        )[..., 0]
        expected = (lev.movedim(-1, 0) * c).sum(0).clamp_min(1e-6)
        lr = torch.log(m.clamp_min(1e-6) / expected)
        r = lr[use]
        med = r.median()
        mad = 1.4826 * (r - med).abs().median()
        w = use & ((lr - med).abs() <= trim * mad)
    med = lr[use].median()
    ratio = torch.where(use, torch.exp(lr - med), torch.ones_like(m))
    return ratio.to(mean.dtype), (expected * torch.exp(med)).to(mean.dtype)


def _hessian(d: Tensor, zooms: tuple[float, float, float]) -> Tensor:
    """Central-difference Hessian in mm of a 3-D volume, ``(*d.shape, 3, 3)``."""
    g = [torch.gradient(d, dim=a, spacing=float(zooms[a]))[0] for a in range(3)]
    h = torch.empty(*d.shape, 3, 3, dtype=d.dtype, device=d.device)
    for i in range(3):
        for j in range(i, 3):
            h[..., i, j] = h[..., j, i] = torch.gradient(g[i], dim=j, spacing=float(zooms[j]))[0]
    return h


def dark_shapes(
    ratio: Tensor,
    brain: Tensor,
    zooms: tuple[float, float, float],
    scales_mm: tuple[float, ...] = (0.5, 0.8, 1.2),
    chunk: int = 1 << 18,
) -> tuple[Tensor, Tensor, Tensor]:
    """Dark-ridge strength, shape class and ridge centre line, from a darkness ratio.

    Darkness ``D = max(-log ratio, 0)`` in ``brain``, 0 outside, so the brain edge is a
    dark band against nothing. Hessian of the Gaussian-smoothed D at each scale (mm,
    scale-normalised by sigma^2); eigenvalues by magnitude ``|l1| <= |l2| <= |l3|``.
    A dark structure is a ridge of D: strong negative curvature across its thin
    directions. ``thin`` = max over scales of ``max(-l3, 0)``; ``shape`` 1 = plate
    (only l3 strong: a sulcal sheet, the brain edge), 2 = tube (l2 too: a vessel through
    slices), 3 = blob (all three: punctate), by an eigenvalue ratio of 0.5. ``ridge``:
    the smoothed D is >= its value one voxel either way along the thin direction (the
    l3 eigenvector, and l2 too for tubes and blobs; non-maximum suppression, as edge
    detectors thin edges). ``zooms``
    are the voxel sizes in the volume's axis order."""
    dt = torch.float32
    d = torch.where(
        brain,
        (-torch.log(ratio.to(dt).clamp_min(1e-3))).clamp_min(0),
        torch.zeros((), dtype=dt, device=ratio.device),
    )
    thin = torch.zeros_like(d)
    shape = torch.zeros(d.shape, dtype=torch.uint8, device=d.device)
    ridge = torch.zeros(d.shape, dtype=torch.bool, device=d.device)
    idx = brain.nonzero()
    step = min(zooms)
    for s in scales_mm:
        ds = masked_blur(d, torch.ones_like(d), tuple(s / z for z in zooms))
        h = _hessian(ds, zooms) * (s * s)
        for lo in range(0, len(idx), chunk):
            p = idx[lo : lo + chunk]
            ev, vec = torch.linalg.eigh(h[p[:, 0], p[:, 1], p[:, 2]].double())
            order = ev.abs().argsort(-1)
            ev = ev.gather(-1, order)
            v3 = vec.gather(-1, order[:, None, 2:3].expand(-1, 3, 1))[..., 0]
            v2 = vec.gather(-1, order[:, None, 1:2].expand(-1, 3, 1))[..., 0]
            t = (-ev[:, 2]).clamp_min(0).to(dt)
            a3 = ev[:, 2].abs().clamp_min(1e-12)
            cls = torch.where(
                ev[:, 0].abs() / a3 >= 0.5, 3, torch.where(ev[:, 1].abs() / a3 >= 0.5, 2, 1)
            ).to(torch.uint8)
            here = ds[p[:, 0], p[:, 1], p[:, 2]]
            on = _centre(ds, p, here, v3, zooms, step)
            # A tube or blob is thin in two directions, and for a round vessel the two
            # are interchangeable: the centre line must be a maximum across both, or a
            # halo voxel passes along whichever one the solver happened to return.
            on &= (cls == 1) | _centre(ds, p, here, v2, zooms, step)
            better = t > thin[p[:, 0], p[:, 1], p[:, 2]]
            q = p[better]
            thin[q[:, 0], q[:, 1], q[:, 2]] = t[better]
            shape[q[:, 0], q[:, 1], q[:, 2]] = cls[better]
            ridge[q[:, 0], q[:, 1], q[:, 2]] = on[better]
    return thin, shape, ridge


def _centre(ds: Tensor, p: Tensor, here: Tensor, v: Tensor, zooms, step: float) -> Tensor:
    """``here`` >= ``ds`` one voxel's worth of mm either way along unit vector ``v``."""
    off = (v * step / torch.tensor(zooms, dtype=v.dtype, device=v.device)).to(ds.dtype)
    q = p.to(ds.dtype)
    return (here >= _trilinear(ds, q + off)) & (here >= _trilinear(ds, q - off))


def _trilinear(vol: Tensor, pts: Tensor) -> Tensor:
    """``vol`` sampled at fractional voxel coordinates ``pts`` (N, 3), edge-clamped."""
    hi = torch.tensor(vol.shape, dtype=pts.dtype, device=pts.device) - 1
    pts = torch.minimum(pts.clamp_min(0), hi)
    p0 = pts.floor().long()
    p1 = torch.minimum(p0 + 1, hi.long())
    f = pts - p0.to(pts.dtype)
    out = torch.zeros(len(pts), dtype=vol.dtype, device=vol.device)
    for cz in (0, 1):
        for cy in (0, 1):
            for cx in (0, 1):
                w = (
                    (f[:, 0] if cz else 1 - f[:, 0])
                    * (f[:, 1] if cy else 1 - f[:, 1])
                    * (f[:, 2] if cx else 1 - f[:, 2])
                )
                iz, iy, ix = (
                    (p1 if cz else p0)[:, 0],
                    (p1 if cy else p0)[:, 1],
                    (p1 if cx else p0)[:, 2],
                )
                out += w * vol[iz, iy, ix]
    return out


def ridge_follow_mask(
    ratio: Tensor,
    region: Tensor,
    thin: Tensor,
    ridge: Tensor,
    seed: float = 0.7,
    grow: float = 0.85,
    seed_pct: float = 95.0,
    grow_pct: float = 75.0,
) -> Tensor:
    """Dark veins that FOLLOW continuous dark structure instead of dilating it.

    Seeds: ``region`` voxels with ratio < ``seed``, or a strong ridge (thinness above
    the region's ``seed_pct`` percentile) with ratio < ``grow``. A voxel joins a seed's
    26-connected component only if ratio < ``grow``, its thinness is above the region's
    ``grow_pct`` percentile (a real ridge, not the partial-volume halo around a seed),
    and it is on the ridge centre line. ``grow <= seed`` turns growth off."""
    import numpy as np
    from scipy import ndimage

    tr = thin[region]
    t_seed = torch.quantile(tr.float()[:: max(1, len(tr) // 1_000_000)], seed_pct / 100)
    t_grow = torch.quantile(tr.float()[:: max(1, len(tr) // 1_000_000)], grow_pct / 100)
    seeds = region & ((ratio < seed) | ((thin > t_seed) & (ratio < grow)))
    if grow <= seed:
        return region & (ratio < seed)
    cand = seeds | (region & (ratio < grow) & ridge & (thin > t_grow))
    lab, n = ndimage.label(cand.cpu().numpy(), np.ones((3, 3, 3), bool))
    keep = np.zeros(n + 1, bool)
    keep[sorted_unique(lab[seeds.cpu().numpy()])] = True
    keep[0] = False
    return torch.from_numpy(keep[lab]).to(ratio.device)
