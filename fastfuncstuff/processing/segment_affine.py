"""Affine registration of an image to a tissue-probability map (``spm_maff8`` port).

Unified Segmentation needs the subject roughly in template space before the
generative fit starts. SPM gets there by registering the image *to the TPM itself*:
there is no intensity template. For a candidate affine, each sample voxel looks up its
prior tissue probabilities ``b_k(x)``, an EM loop learns one 256-bin intensity
histogram per class (``P(g|k)``), and the score is the mutual information between
intensity and tissue class. The affine that best explains the intensities with the
tissue the TPM expects at each location wins. Gauss-Newton on 12 polar-decomposition
parameters, with a Gaussian prior on the 6 zoom/shear parameters learned from real
brains (``spm_affine_priors``), keeps the zooms from running away.

:func:`register_to_tpm` is ``spm_maff8``; :func:`affine_to_tpm` is
``spm_preproc_run.m:run_affine`` (two coarse starts — header origin and FoV centre —
then two refinements at ``samp``).

Coordinates are 0-based nibabel voxel indices throughout; SPM's 1-based ``y3>=1``
"above the bottom of the TPM" test becomes ``z >= 0``. The returned matrix maps
subject world mm → TPM world mm, i.e. SPM's ``Affine`` and ``fit_segment``'s
``world_affine``.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.linalg
import torch
from torch import Tensor

from .segment import apply_affine_pts, fudge_factor, sample_tpm_prior

# spm_affine_priors.m: mean and inverse covariance of the 6 zoom/shear parameters
# (log of the symmetric polar factor), by population.
_MNI_MU = [0.0667, 0.0039, 0.0008, 0.0333, 0.0071, 0.1071]
_MNI_ISIG = [
    [0.0902, -0.0345, -0.0106, -0.0025, -0.0005, -0.0163],
    [-0.0345, 0.7901, 0.3883, 0.0041, -0.0103, -0.0116],
    [-0.0106, 0.3883, 2.2599, 0.0113, 0.0396, -0.0060],
    [-0.0025, 0.0041, 0.0113, 0.0925, 0.0471, -0.0440],
    [-0.0005, -0.0103, 0.0396, 0.0471, 0.2964, -0.0062],
    [-0.0163, -0.0116, -0.0060, -0.0440, -0.0062, 0.1144],
]
_EASTERN_MU = [0.0719, -0.0040, -0.0032, 0.1416, 0.0601, 0.2578]
_EASTERN_ISIG = [
    [0.0757, 0.0220, -0.0224, -0.0049, 0.0304, -0.0327],
    [0.0220, 0.3125, -0.1555, 0.0280, -0.0012, -0.0284],
    [-0.0224, -0.1555, 1.9727, 0.0196, -0.0019, 0.0122],
    [-0.0049, 0.0280, 0.0196, 0.0576, -0.0282, -0.0200],
    [0.0304, -0.0012, -0.0019, -0.0282, 0.2128, -0.0275],
    [-0.0327, -0.0284, 0.0122, -0.0200, -0.0275, 0.0511],
]
_SUBJ_ISIG = [
    [0.8876, 0.0784, 0.0784, -0.1749, 0.0784, -0.1749],
    [0.0784, 5.3894, 0.2655, 0.0784, 0.2655, 0.0784],
    [0.0784, 0.2655, 5.3894, 0.0784, 0.2655, 0.0784],
    [-0.1749, 0.0784, 0.0784, 0.8876, 0.0784, -0.1749],
    [0.0784, 0.2655, 0.2655, 0.0784, 5.3894, 0.0784],
    [-0.1749, 0.0784, 0.0784, -0.1749, 0.0784, 0.8876],
]

AFFINE_REG_TYPES = ("mni", "imni", "eastern", "subj", "rigid", "none")


def affine_priors(regtype: str) -> tuple[np.ndarray, np.ndarray]:
    """``spm_affine_priors``: ``(mu, isig)`` for the 6 zoom/shear parameters."""
    regtype = regtype.lower()
    if regtype in ("mni", "imni"):
        mu = np.array(_MNI_MU) * (-1.0 if regtype == "imni" else 1.0)
        return mu, 1e4 * np.array(_MNI_ISIG)
    if regtype == "eastern":
        return np.array(_EASTERN_MU), 1e4 * np.array(_EASTERN_ISIG)
    if regtype == "subj":
        return np.zeros(6), 1e3 * np.array(_SUBJ_ISIG)
    if regtype == "rigid":
        return np.zeros(6), np.eye(6) * 1e8
    if regtype == "none":
        return np.zeros(6), np.zeros((6, 6))
    raise ValueError(f"affine regularisation {regtype!r} not one of {AFFINE_REG_TYPES}")


# Polar-decomposition parametrisation: M = [V·R  t], R = expm(skew), V = expm(sym).
# The index lists are MATLAB column-major positions into a 3x3, kept as (row, col).
_ROT_IDX = ((1, 0), (2, 0), (2, 1))
_SYM_IDX = ((0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (2, 2))


def params_to_affine(p: np.ndarray) -> np.ndarray:
    """``P2M``: 12 polar parameters → 4×4 affine."""
    t = np.zeros((3, 3))
    for (r, c), v in zip(_ROT_IDX, p[3:6], strict=True):
        t[r, c] = -v
    rot = scipy.linalg.expm(t - t.T)
    t = np.zeros((3, 3))
    for (r, c), v in zip(_SYM_IDX, p[6:12], strict=True):
        t[r, c] = v
    sym = scipy.linalg.expm(t + t.T - np.diag(np.diag(t)))
    m = np.eye(4)
    m[:3, :3] = sym @ rot
    m[:3, 3] = p[:3]
    return m


def affine_to_params(m: np.ndarray) -> np.ndarray:
    """``M2P``: 4×4 affine → 12 polar parameters (inverse of :func:`params_to_affine`)."""
    j = m[:3, :3]
    v = np.real(scipy.linalg.sqrtm(j @ j.T))
    r = np.linalg.solve(v, j)
    lv = np.real(scipy.linalg.logm(v))
    lr = -scipy.linalg.logm(r)
    if np.sum(np.imag(lr) ** 2) > 1e-6:
        raise ValueError("affine_to_params: rotations by pi are not representable")
    lr = np.real(lr)
    p = np.zeros(12)
    p[:3] = m[:3, 3]
    p[3:6] = [lr[r_, c_] for r_, c_ in _ROT_IDX]
    p[6:12] = [lv[r_, c_] for r_, c_ in _SYM_IDX]
    return p


def _smooth_kernel(fwhm: float, x: np.ndarray) -> np.ndarray:
    """``spm_smoothkern(fwhm, x, 0)``: a Gaussian integrated over each unit bin."""
    s = (fwhm / math.sqrt(8.0 * math.log(2.0))) ** 2 + np.finfo(float).eps
    w1 = 1.0 / math.sqrt(2.0 * s)
    erf = np.vectorize(math.erf)
    k = 0.5 * (erf(w1 * (x + 0.5)) - erf(w1 * (x - 0.5)))
    return np.clip(k, 0.0, None)


def _round_half_up(x: Tensor) -> Tensor:
    """MATLAB ``round`` for the non-negative values used here (torch rounds half-to-even)."""
    return torch.floor(x + 0.5)


class _Samples:
    """``loadbuf``: the strided, uint8-quantised image samples one registration uses."""

    def __init__(
        self,
        volume: Tensor,
        subj_affine: np.ndarray,
        samp: float,
        *,
        dither: float,
        seed: int,
    ):
        vox = np.sqrt((subj_affine[:3, :3] ** 2).sum(axis=0))  # (vx, vy, vz)
        sk = _stride(vox, samp)

        g = volume[:: sk[2], :: sk[1], :: sk[0]].to(torch.float64)  # (nz, ny, nx) layout
        dev = g.device
        finite = torch.isfinite(g)
        # 4000-bin histogram for the 0.05/99.95% intensity quantiles
        gmin = float(g[finite].min())
        gmax = float(g[finite].max())
        a = 3999.0 / (gmax - gmin) if gmax > gmin else 1.0
        b = 1.0 - a * gmin
        keep = finite & (g != 0) & (g != -3024)  # -3024: CT air, never tissue
        idx = _round_half_up(g[keep] * a + b).long().clamp(1, 4000) - 1
        h = torch.bincount(idx, minlength=4000).to(torch.float64)
        h = torch.cumsum(h, 0) / h.sum()
        lo = (float(torch.nonzero(h > 0.0005)[0]) + 1 - b) / a
        hi = (float(torch.nonzero(h > 0.9995)[0]) + 1 - b) / a
        a2 = 255.0 / (hi - lo) if hi > lo else 1.0
        b2 = -a2 * lo

        msk = finite & (g != 0)
        vals = g[msk]
        if dither:
            gen = torch.Generator(device="cpu").manual_seed(seed)
            noise = torch.rand(vals.shape, generator=gen, dtype=torch.float64).to(dev)
            vals = vals + noise * dither - dither / 2
        self.g = _round_half_up(vals * a2 + b2).clamp(0, 255).long()
        zz, yy, xx = torch.nonzero(msk, as_tuple=True)
        # subject voxel coords (x, y, z) of each sample, in the full-resolution grid
        self.coords = torch.stack([xx * sk[0], yy * sk[1], zz * sk[2]], dim=1).to(torch.float64)


def _jacobian_params(tpm_affine: np.ndarray, p: np.ndarray, subj_affine: np.ndarray) -> np.ndarray:
    """``derivs``: d vec(T[:3,:]) / d params (12×12, row-major vec), numerically."""
    tpm_inv = np.linalg.inv(tpm_affine)
    m0 = (tpm_inv @ params_to_affine(p) @ subj_affine)[:3].reshape(-1)
    out = np.zeros((12, 12))
    dp = 1e-7
    for i in range(12):
        p1 = p.copy()
        p1[i] += dp
        out[:, i] = ((tpm_inv @ params_to_affine(p1) @ subj_affine)[:3].reshape(-1) - m0) / dp
    return out


def register_to_tpm(
    volume: Tensor,
    subj_affine: np.ndarray,
    log_prior: Tensor,
    tpm_affine: np.ndarray,
    bg_low: Tensor,
    bg_high: Tensor,
    *,
    samp: float = 3.0,
    fwhm: float = 0.0,
    init: np.ndarray | None = None,
    regtype: str = "mni",
    max_iter: int = 200,
    dither: float = 0.0,
    seed: int = 1,
    dtype: torch.dtype = torch.float32,
    samples: _Samples | None = None,
) -> tuple[np.ndarray, float]:
    """``spm_maff8``: affine (subject world → TPM world) maximising intensity–tissue MI.

    Args:
        volume: ``(nz, ny, nx)`` image, unmasked (zeros and non-finite are excluded).
        subj_affine, tpm_affine: nibabel voxel→world affines.
        log_prior, bg_low, bg_high: from :func:`segment.load_tpm`.
        samp: sampling stride in mm.
        fwhm: smoothness for the fudge factor only — it scales the prior, it smooths
            nothing. SPM's "closer to rigid" passes use ``(fwhm+1)*16``.
        init: starting affine; ``None`` starts at the prior mean.
        regtype: zoom/shear prior (:data:`AFFINE_REG_TYPES`).
        dither: integer-quantisation step to dither by (SPM's ``scrand``).
        dtype: precision of the TPM sampling; histogram and normal equations are float64.
        samples: precomputed :class:`_Samples` to reuse across calls at the same samp.

    Returns:
        ``(affine, ll)`` — the 4×4 world→world map and its final log-likelihood (bits per
        sample, penalised), comparable between starts at the same ``samp``.
    """
    if samples is None:
        samples = _Samples(volume, subj_affine, samp, dither=dither, seed=seed)
    dev = volume.device
    s = samples
    vox = np.sqrt((subj_affine[:3, :3] ** 2).sum(axis=0))
    ff = fudge_factor(tuple(float(v) for v in vox), tuple(_stride(vox, samp)), fwhm)
    mu6, isig = affine_priors(regtype)
    mu = np.concatenate([np.zeros(6), mu6])
    alpha0 = np.zeros((12, 12))
    alpha0[:6, :6] = np.eye(6) * 1e-5
    alpha0[6:, 6:] = isig
    alpha0 *= ff

    sol = affine_to_params(init) if init is not None else mu.copy()
    n_tissue = log_prior.shape[0]
    tpm_inv = np.linalg.inv(tpm_affine)
    coords = s.coords.to(dtype)
    gidx = s.g
    lp = log_prior.to(dtype)
    krn = torch.as_tensor(
        _smooth_kernel(4.0, np.arange(-256, 257, dtype=np.float64)), dtype=torch.float64, device=dev
    ).flip(0)[None, None]
    eps = float(np.finfo(float).eps)
    h1 = torch.ones((256, n_tissue), dtype=torch.float64, device=dev)

    def tpm_coords(p: np.ndarray) -> tuple[Tensor, Tensor]:
        t = torch.as_tensor(tpm_inv @ params_to_affine(p) @ subj_affine, dtype=dtype, device=dev)
        y = apply_affine_pts(coords, t)
        keep = y[:, 2] >= 0  # drop samples below the bottom of the TPM
        return y[keep], keep

    def evaluate(p: np.ndarray) -> float:
        """EM-fit the class histograms at ``p``; returns the penalised log-likelihood."""
        nonlocal h1
        d = p - mu
        penalty = 0.5 * d @ alpha0 @ d
        y, keep = tpm_coords(p)
        prior = sample_tpm_prior(lp, y, bg_low, bg_high, kernel="bspline2").to(torch.float64)
        g = gidx[keep]
        ll1 = 0.0
        for subit in range(1, 33):
            check = subit % 4 == 0
            if check:
                ll0, ll1 = ll1, 0.0
            q = h1[g] * prior
            sq = q.sum(dim=1, keepdim=True) + eps
            if check:
                ll1 = float(torch.log(sq).sum())
            h0 = torch.full((256, n_tissue), eps, dtype=torch.float64, device=dev)
            h0.index_add_(0, g, q / sq)
            p_joint = (h0 + eps) / (h0 + eps).sum()
            sm = torch.nn.functional.conv1d(p_joint.T[:, None], krn, padding=256)[:, 0].T
            # dividing out both marginals makes sum(h0·log h1) a mutual information
            h1 = sm / (sm.sum(dim=1, keepdim=True) * sm.sum(dim=0, keepdim=True))
            if check and (ll1 - ll0) / float(h0.sum()) < 1e-5:
                break
        ssh = float(h0.sum())
        return (float((h0 * torch.log(h1)).sum()) - penalty) / ssh / math.log(2.0)

    def gauss_newton_step(p: np.ndarray) -> np.ndarray:
        y, keep = tpm_coords(p)
        y = y.detach().requires_grad_(True)
        prior = sample_tpm_prior(lp, y, bg_low, bg_high, kernel="bspline2")
        mi = (h1[gidx[keep]].to(dtype) * prior).sum(dim=1) + eps
        (dy,) = torch.autograd.grad(torch.log(mi).sum(), y)  # (n, 3) = dmi/mi per sample
        xh = torch.cat([s.coords[keep], torch.ones_like(s.coords[keep, :1])], dim=1)
        a = (dy.to(torch.float64)[:, :, None] * xh[:, None, :]).reshape(-1, 12)
        alpha = (a.T @ a).cpu().numpy()
        beta = -a.sum(dim=0).cpu().numpy()
        r = _jacobian_params(tpm_affine, p, subj_affine)
        alpha = r.T @ alpha @ r
        beta = r.T @ beta
        return np.linalg.solve(alpha + alpha0, beta + alpha0 @ (p - mu))

    ll = -math.inf
    dsol = np.zeros(12)
    for it in range(max_iter):
        stepsize = 1.0
        for search in range(12):
            sol1 = sol - stepsize * dsol if it > 0 else sol
            ll1 = evaluate(sol1)
            if it == 0:
                break
            if abs(ll1 - ll) < 1e-4:
                return params_to_affine(sol1), ll
            if ll1 < ll:
                stepsize *= 0.5
                if search == 11:  # SPM returns the last (worse) trial; keep the best instead
                    return params_to_affine(sol), ll
            else:
                break
        ll, sol = ll1, sol1
        dsol = gauss_newton_step(sol)
    return params_to_affine(sol), ll


def _stride(vox: np.ndarray, samp: float) -> list[int]:
    """Voxel stride ``sk = max(1, round(samp/vx))`` (MATLAB rounding)."""
    return [max(1, int(math.floor(samp / float(v) + 0.5))) for v in vox]


def affine_to_tpm(
    volume: Tensor,
    subj_affine: np.ndarray,
    log_prior: Tensor,
    tpm_affine: np.ndarray,
    bg_low: Tensor,
    bg_high: Tensor,
    *,
    samp: float = 3.0,
    fwhm: float = 0.0,
    regtype: str = "mni",
    dither: float = 0.0,
    dtype: torch.dtype = torch.float32,
    verbose: bool = False,
) -> np.ndarray:
    """``run_affine``: SPM Segment's automatic subject→TPM affine.

    Headers often put the origin somewhere arbitrary, so two coarse (8 mm) registrations
    start from identity — one with the header origin, one with the origin moved to the
    centre of the field of view — and the better log-likelihood seeds two refinements at
    ``samp``: first with the prior inflated ``(fwhm+1)*16`` ("closer to rigid"), then at
    the real ``fwhm``.

    Returns:
        4×4 subject world → TPM world (SPM's ``Affine``; ``fit_segment``'s ``world_affine``).
    """
    common = dict(log_prior=log_prior, tpm_affine=tpm_affine, bg_low=bg_low, bg_high=bg_high)
    kw = dict(regtype=regtype, dither=dither, dtype=dtype)
    coarse = _Samples(volume, subj_affine, 8.0, dither=dither, seed=1)

    # The FoV-centre start only changes the origin; the samples (in voxels) are shared.
    dims = np.array(volume.shape[::-1], dtype=np.float64)  # (nx, ny, nz)
    centred = subj_affine.copy()
    centred[:3, 3] = -subj_affine[:3, :3] @ ((dims - 1) / 2)
    a1, ll1 = register_to_tpm(
        volume, centred, **common, samp=8.0, fwhm=(fwhm + 1) * 16, samples=coarse, **kw
    )
    a1 = a1 @ centred @ np.linalg.inv(subj_affine)
    a2, ll2 = register_to_tpm(
        volume, subj_affine, **common, samp=8.0, fwhm=(fwhm + 1) * 16, samples=coarse, **kw
    )
    if verbose:
        print(f"  affine starts: FoV-centre ll={ll1:.4f}, header-origin ll={ll2:.4f}")
    aff = a1 if ll1 > ll2 else a2

    fine = _Samples(volume, subj_affine, samp, dither=dither, seed=1)
    aff, _ = register_to_tpm(
        volume, subj_affine, **common, samp=samp, fwhm=(fwhm + 1) * 16, init=aff, samples=fine, **kw
    )
    aff, ll = register_to_tpm(
        volume, subj_affine, **common, samp=samp, fwhm=fwhm, init=aff, samples=fine, **kw
    )
    if verbose:
        print(f"  affine refined: ll={ll:.4f}")
    return aff
