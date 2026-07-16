"""GPU Unified Segmentation — a port of SPM's ``spm_preproc8`` (New Segment).

One EM loop that jointly estimates a **Gaussian mixture** over tissue classes, a
smooth multiplicative **bias field**, and a **deformation** that warps a tissue-
probability template (TPM) onto the subject. Method + staged plan:
``../fmri_wiki/concepts/Unified Segmentation.md``; reference MATLAB in
``matlab_toolboxes/spm25/spm_preproc8.m``.

The generative model at each voxel ``x`` with intensity ``y`` (``n_chan`` channels):

    resp_k(x) ∝ mix_k · N(bias(x)·y ; means_k, covs_k) · TPM_{tissue_of[k]}(φ(x))

``n_gauss`` Gaussians are grouped into ``n_tissue`` classes by ``tissue_of`` (SPM's
``lkp``). ``n_tissue`` is whatever the TPM provides — 6 for SPM's default ``TPM.nii``
(GM, WM, CSF, bone, soft tissue, air), but fully flexible: a user 4-D TPM with any
number of classes works.

Naming vs the reference (``spm_preproc8``): ``means``≡``mn``, ``covs``≡``vr``,
``mix``≡``mg``, ``tissue_of``≡``lkp``, bias coeffs ``T``≡``Tbias``. Intensities are
carried as ``(n_vox, n_chan)``; the likelihood/covariance reductions run in float64
([[Float32 vs float64]]).

Built bottom-up: Gaussian-mixture core, then the bias field; the TPM/warp sampling
and the EM driver follow.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # tqdm optional
    _tqdm = None

_LOG_2PI = 1.8378770664093453  # log(2π)


def _quantile(t: Tensor, q: float) -> Tensor:
    """``torch.quantile`` that also works past its ~2^24-element input limit.

    Above the limit ``torch.quantile`` raises; fall back to ``kthvalue`` (an exact
    order statistic, no interpolation), which is all the focus-map threshold needs.
    """
    if t.numel() <= (1 << 24):
        return torch.quantile(t, q)
    k = min(max(int(round(q * (t.numel() - 1))) + 1, 1), t.numel())
    return t.kthvalue(k).values


def _gaussian_kernel1d(sigma: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Normalised 1-D Gaussian kernel with radius ``ceil(3σ)``."""
    radius = max(1, int(torch.ceil(torch.tensor(3.0 * sigma)).item()))
    x = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def blur_log_prior(log_prior: Tensor, sigma: float) -> Tensor:
    """Spatially blur a (log-)TPM by ``sigma`` **TPM-voxel** units, in prob space.

    Smoothing happens on the probabilities (``exp`` of the stored log-prior) then the
    log is retaken with the same ``tiny=1e-4`` floor as :func:`load_tpm`, so the result
    is a valid log-prior. A blurred TPM has broad, gently-sloping tissue boundaries: the
    warp's data term then pulls over a much wider capture range, which matters when the
    subject still carries large residual distortion (e.g. an under-corrected EPI) and a
    sharp prior would leave the warp stuck with zero gradient far from the true edge.
    Used for the coarse first pass of :func:`fit_segment` (``blur_tpms``). ``sigma<=0``
    returns the input unchanged.
    """
    import torch.nn.functional as F  # noqa: N812

    if sigma <= 0:
        return log_prior
    kernel = _gaussian_kernel1d(sigma, log_prior.device, torch.float32)
    r = kernel.numel() // 2
    tiny = 1e-4
    prob = torch.exp(log_prior.to(torch.float32))  # (n_tissue, nz, ny, nx)
    x = prob[:, None]  # (n_tissue, 1, nz, ny, nx) — tissues batched, single channel
    for axis in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = kernel.numel()
        w = kernel.reshape(shape)
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - axis)] = r
        pad[2 * (2 - axis) + 1] = r
        x = F.conv3d(F.pad(x, pad, mode="replicate"), w)
    return torch.log(x[:, 0].clamp_min(0.0) + tiny).to(log_prior.dtype)


def _smooth_field(field: Tensor, sigma: float, keep_weight: Tensor | None = None) -> Tensor:
    """Separable Gaussian smoothing of a dense displacement field ``(gz, gy, gx, 3)``.

    Applied to the warp between optimiser steps (a Sobolev / diffeomorphic-style
    update): each grid node's raw data-driven jump is spatially averaged with its
    neighbours, so the dense field stays **smooth** instead of developing the
    speckly per-node / checkerboard pattern a freely-optimised field falls into.
    ``sigma`` is in grid-node units.

    ``keep_weight`` (``(gz, gy, gx)`` in ``[0, 1]``, optional) makes the smoothing
    **spatially adaptive**: the result is ``smoothed + keep_weight·(raw − smoothed)``,
    so a node with ``keep_weight=0`` is fully smoothed (the default everywhere) while a
    node with ``keep_weight=1`` keeps its raw, freely-optimised displacement. This lets
    a localised region — a peninsula of large residual distortion — deform hard without
    relaxing the smoothing that keeps the rest of the field stable (see ``warp_focus``).
    """
    import torch.nn.functional as F  # noqa: N812

    if sigma <= 0:
        return field
    kernel = _gaussian_kernel1d(sigma, field.device, field.dtype)
    r = kernel.numel() // 2
    x = field.permute(3, 0, 1, 2)[None]  # (1, 3, gz, gy, gx) — treat components as channels
    for axis in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = kernel.numel()
        w = kernel.reshape(shape).repeat(3, 1, 1, 1, 1)
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - axis)] = r
        pad[2 * (2 - axis) + 1] = r
        x = F.conv3d(F.pad(x, pad, mode="replicate"), w, groups=3)
    smoothed = x[0].permute(1, 2, 3, 0).contiguous()
    if keep_weight is None:
        return smoothed
    kw = keep_weight.to(smoothed.dtype)[..., None]  # broadcast over the 3 components
    return smoothed + kw * (field - smoothed)


# ---------------------------------------------------------------------------
# Gaussian mixture — closed-form E-step and M-step, tissue-prior weighted.
# ---------------------------------------------------------------------------


def gmm_responsibilities(
    corrected: Tensor,
    prior: Tensor,
    means: Tensor,
    covs: Tensor,
    mix: Tensor,
    tissue_of: Tensor,
) -> tuple[Tensor, Tensor]:
    """E-step: per-Gaussian responsibilities and the per-voxel log-likelihood.

    Args:
        corrected: (n_vox, n_chan) bias-corrected intensities.
        prior: (n_vox, n_tissue) tissue-prior (warped TPM) values at the voxels; rows
            need not sum to 1 (the E-step renormalises).
        means: (n_chan, n_gauss) Gaussian means.
        covs: (n_chan, n_chan, n_gauss) Gaussian covariances (positive definite).
        mix: (n_gauss,) within-tissue mixing proportions.
        tissue_of: (n_gauss,) 0-based tissue index per Gaussian, in ``[0, n_tissue)``.

    Returns:
        resp: (n_vox, n_gauss) responsibilities (rows sum to 1).
        loglik: (n_vox,) data log-likelihood ``log Σ_k unnormalised-density``.
    """
    # Per-voxel work runs in the caller's dtype (float32 fit hot path → fast + cheap
    # autograd on consumer GPUs, where float64 is ~1/32 rate). Only the tiny per-Gaussian
    # covariance factorisation is promoted to float64 for conditioning ([[Float32 vs
    # float64]]).
    wdt = corrected.dtype
    n_chan = corrected.shape[1]
    means = means.to(wdt)
    mix = mix.to(wdt)
    prior = prior.to(wdt)

    # log N(y; mean, cov) = -n_chan/2·log2π - 1/2·log|cov| - 1/2·(y-mean)' cov⁻¹ (y-mean).
    # Batch the tiny per-Gaussian factorisations (covs are n_chan×n_chan, n_chan small)
    # into ONE call each — a Python loop of 1×1 Cholesky/triangular-solves was pure
    # LAPACK-dispatch overhead (dominant on CPU, launch-bound on GPU).
    covs_b = covs.to(torch.float64).permute(2, 0, 1)  # (n_gauss, n_chan, n_chan)
    chol = torch.linalg.cholesky(covs_b)  # M-step keeps covs PD
    logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=1, dim2=2)).sum(dim=1)  # (n_gauss,)
    prec = torch.cholesky_inverse(chol).to(wdt)  # (n_gauss, n_chan, n_chan) = cov⁻¹
    logdet = logdet.to(wdt)
    centred = corrected[:, :, None] - means[None, :, :]  # (n_vox, n_chan, n_gauss)
    tmp = torch.einsum("vck,kcd->vdk", centred, prec)
    maha = (tmp * centred).sum(dim=1)  # (n_vox, n_gauss)

    log_dens = (
        torch.log(mix.clamp_min(1e-40))
        + torch.log(prior[:, tissue_of].clamp_min(1e-40))
        - 0.5 * (n_chan * _LOG_2PI + logdet + maha)
    )
    loglik = torch.logsumexp(log_dens, dim=1)  # (n_vox,)
    resp = torch.exp(log_dens - loglik[:, None])  # (n_vox, n_gauss), rows sum to 1
    return resp, loglik


def gmm_moments(corrected: Tensor, resp: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Accumulate the responsibility-weighted 0th/1st/2nd moments for the M-step.

    Args:
        corrected: (n_vox, n_chan) bias-corrected intensities.
        resp: (n_vox, n_gauss) responsibilities.

    Returns:
        count: (n_gauss,) soft counts ``Σ resp``.
        sum1: (n_chan, n_gauss) weighted sums ``Σ resp·y``.
        sum2: (n_chan, n_chan, n_gauss) weighted outer products ``Σ resp·y·yᵀ``.
    """
    corrected = corrected.to(torch.float64)
    resp = resp.to(torch.float64)
    count = resp.sum(dim=0)
    sum1 = torch.einsum("vn,vk->nk", corrected, resp)
    sum2 = torch.einsum("vn,vm,vk->nmk", corrected, corrected, resp)
    return count, sum1, sum2


def gmm_update(
    count: Tensor,
    sum1: Tensor,
    sum2: Tensor,
    tissue_of: Tensor,
    cov_prior: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """M-step: closed-form mixing/means/covariances from accumulated moments.

    Means are ``sum1/count``. Covariances follow SPM's eq. 25 with a Wishart-style
    prior ``cov_prior`` (``vr0``): ``vr = (scatter + n_chan·cov_prior)/(count + n_chan)``
    where ``scatter = sum2 - count·mean·meanᵀ``. This both regularises toward the prior
    and keeps every covariance positive definite. Mixing proportions are normalised
    *within* each tissue (Gaussians sharing a ``tissue_of`` value).

    Args:
        count, sum1, sum2: moments from :func:`gmm_moments`.
        tissue_of: (n_gauss,) 0-based tissue index per Gaussian.
        cov_prior: (n_chan, n_chan) Wishart covariance prior (SPM ``vr0``).

    Returns:
        means, covs, mix — as in :func:`gmm_responsibilities`.
    """
    n_chan = sum1.shape[0]
    count_safe = count.clamp_min(1e-40)
    means = sum1 / count_safe
    cov_prior = cov_prior.to(sum2.dtype)

    # eq.25 for every Gaussian at once (batched over k) — no Python loop.
    mt = means.T  # (n_gauss, n_chan)
    outer = mt[:, :, None] * mt[:, None, :]  # (n_gauss, n_chan, n_chan)
    scatter = sum2.permute(2, 0, 1) - count_safe[:, None, None] * outer
    cov = (scatter + n_chan * cov_prior[None]) / (count_safe[:, None, None] + n_chan)
    cov = 0.5 * (cov + cov.transpose(1, 2))  # symmetrise
    covs = cov.permute(1, 2, 0).contiguous()  # (n_chan, n_chan, n_gauss)

    # within-tissue mixing proportions: count / (Σ count over the Gaussians sharing a tissue)
    n_tissue = int(tissue_of.max().item()) + 1
    tissue_tot = torch.zeros(n_tissue, dtype=count.dtype, device=count.device)
    tissue_tot.index_add_(0, tissue_of, count)
    mix = count / tissue_tot[tissue_of].clamp_min(1e-40)
    return means, covs, mix


# ---------------------------------------------------------------------------
# Bias field — smooth multiplicative INU, a 3-D DCT (SPM's transf/spm_dctmtx).
# ---------------------------------------------------------------------------


def dct_basis(pos: Tensor, n: int, n_coef: int) -> Tensor:
    """DCT-II basis (``spm_dctmtx``) evaluated at continuous positions.

    Args:
        pos: (n_vox,) sample positions in ``[0, n)`` (voxel index, may be fractional).
        n: dimension length the basis is defined over.
        n_coef: number of basis functions (orders 0..n_coef-1).

    Returns:
        (n_vox, n_coef) basis; column 0 is ``1/√n`` (DC), column ``j`` is
        ``√(2/n)·cos(π(2·pos+1)·j/(2n))``.
    """
    dt = pos.dtype if pos.is_floating_point() else torch.float64
    basis = torch.empty((pos.shape[0], n_coef), dtype=dt, device=pos.device)
    basis[:, 0] = 1.0 / (n**0.5)
    if n_coef > 1:
        order = torch.arange(1, n_coef, dtype=dt, device=pos.device)
        angle = torch.pi * (2.0 * pos[:, None].to(dt) + 1.0) * order[None, :] / (2.0 * n)
        basis[:, 1:] = (2.0 / n) ** 0.5 * torch.cos(angle)
    return basis


def bias_field_shape(
    dim: tuple[int, int, int],
    vox: tuple[float, float, float],
    fwhm: float,
    biasreg: float,
    ff: float = 1.0,
    *,
    device: torch.device | str = "cpu",
) -> tuple[tuple[int, int, int], Tensor]:
    """Number of DCT coefficients per axis and SPM's diagonal bias precision.

    For each axis, ``sd = vx·dim/fwhm`` sets both the coefficient count
    ``n_coef = ceil(2·sd)`` (SPM's ``d3``) and the 1-D smoothness kernel
    ``kern[j] = exp(-j²/sd²)/√vx``. The precision on coefficient ``(i,j,k)`` is

        prec[i,j,k] = (kern_x[i]·kern_y[j]·kern_z[k])^-2 · biasreg · ff

    a Gaussian-shaped diagonal that clamps high frequencies so the field stays smooth
    (penalty ``0.5·Σ prec·T²``). ``ff`` is SPM's sample-density fudge factor.

    Returns:
        n_coef: ``(nbx, nby, nbz)`` number of DCT coefficients per axis.
        prec: ``(nbx, nby, nbz)`` diagonal precision on the coefficients.
    """
    counts: list[int] = []
    kerns: list[Tensor] = []
    for d0, vx in zip(dim, vox, strict=True):
        sd = vx * d0 / fwhm
        nc = max(int(torch.ceil(torch.tensor(2.0 * sd)).item()), 1)
        counts.append(nc)
        order = torch.arange(nc, dtype=torch.float64, device=device)
        kerns.append(torch.exp(-(order**2) / sd**2) / (vx**0.5))
    kx, ky, kz = kerns
    kern3 = kx[:, None, None] * ky[None, :, None] * kz[None, None, :]
    prec = kern3.pow(-2.0) * (biasreg * ff)
    return (counts[0], counts[1], counts[2]), prec


def eval_log_bias(coef: Tensor, basis_x: Tensor, basis_y: Tensor, basis_z: Tensor) -> Tensor:
    """Evaluate the log bias field ``Σ_ijk coef[i,j,k]·bx[v,i]·by[v,j]·bz[v,k]``.

    Args:
        coef: (nbx, nby, nbz) DCT coefficients.
        basis_x, basis_y, basis_z: (n_vox, nb*) per-axis DCT bases at the samples.

    Returns:
        (n_vox,) log-bias; the multiplicative field is ``exp`` of this. Contracted
        axis-by-axis so intermediates stay small even with many coefficients.
    """
    dt = coef.dtype
    contract_z = torch.einsum("ijk,vk->ijv", coef, basis_z.to(dt))
    contract_y = torch.einsum("ijv,vj->iv", contract_z, basis_y.to(dt))
    return torch.einsum("iv,vi->v", contract_y, basis_x.to(dt))


def bias_penalty_value(coef: Tensor, prec: Tensor) -> Tensor:
    """Bias regulariser ``0.5·Σ prec·coef²`` (negative log-prior on the DCT coeffs)."""
    return 0.5 * (prec.to(coef.dtype) * coef * coef).sum()


# ---------------------------------------------------------------------------
# Geometry — subject voxel → template (TPM) voxel, and TPM prior sampling.
# ---------------------------------------------------------------------------


def compose_vox2vox(subj_affine: Tensor, tpm_affine: Tensor, world_affine: Tensor) -> Tensor:
    """Voxel→voxel map subject → TPM (SPM's ``tpm.M \\ Affine * image.mat``).

    All affines are nibabel 0-based ``(x,y,z)``→world; ``world_affine`` (SPM's
    ``Affine`` from ``spm_maff8``/ffs_allineate) is the subject-world→TPM-world map,
    which is convention-free (world-to-world). The composition maps a subject voxel
    index ``(x,y,z)`` to a TPM voxel index.
    """
    dtype = torch.float64
    return torch.linalg.inv(tpm_affine.to(dtype)) @ world_affine.to(dtype) @ subj_affine.to(dtype)


def apply_affine_pts(coords: Tensor, mat: Tensor) -> Tensor:
    """Apply a 4×4 affine to ``(n_vox, 3)`` points ``(x,y,z)`` → ``(n_vox, 3)``."""
    mat = mat.to(coords.dtype)
    return coords @ mat[:3, :3].T + mat[:3, 3]


def load_tpm(
    path: str, *, device: torch.device | str = "cpu"
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Load a 4-D tissue-probability template (``spm_load_priors8`` equivalent).

    The number of tissue classes ``n_tissue`` is taken from the 4th dimension — 6 for
    SPM's default ``TPM.nii`` but whatever the file provides (user TPMs with any class
    count work). Each class is clamped to ``[0,1]`` and the log is stored (with a
    ``tiny=1e-4`` floor) so the prior is sampled in log-space then exponentiated.

    Returns:
        log_prior: ``(n_tissue, nz, ny, nx)`` = ``log(p + tiny)``.
        tpm_affine: ``(4, 4)`` nibabel voxel→world affine.
        bg_low: ``(n_tissue,)`` mean of the first z-plane (below-FoV background prob).
        bg_high: ``(n_tissue,)`` mean of the last z-plane (above-FoV background prob).
    """
    from .io import load_image  # local import: keeps the numeric core import-light

    data, hdr = load_image(path, device=device)  # (n_tissue, nz, ny, nx) or (nz,ny,nx)
    if data.ndim == 3:
        data = data[None]
    tiny = 1e-4
    prob = data.clamp(0.0, 1.0)
    bg_low = prob[:, 0, :, :].mean(dim=(1, 2))  # first z-plane
    bg_high = prob[:, -1, :, :].mean(dim=(1, 2))  # last z-plane
    log_prior = torch.log(prob + tiny)
    affine = torch.as_tensor(hdr["affine"], dtype=torch.float64, device=data.device)
    return log_prior, affine, bg_low, bg_high


def sample_tpm_prior(
    log_prior: Tensor,
    coords: Tensor,
    bg_low: Tensor,
    bg_high: Tensor,
    *,
    kernel: str = "linear",
) -> Tensor:
    """Sample the (log-)TPM at TPM voxel ``coords`` → normalised tissue prior.

    Faithful to ``spm_sample_priors8``: in-bounds voxels take ``exp(interp(logp))``;
    voxels below the volume (``z < 0``) take the below-FoV background ``bg_low``; other
    out-of-bounds voxels take ``bg_high``; then each row is renormalised to sum to 1.

    ``kernel`` picks the interpolation: ``"linear"`` (trilinear — the default, and the
    only differentiable option, used inside the warp fit) or a smooth higher-order
    Lagrange kernel (``"cubic"``/``"quintic"``/``"heptic"``, SPM's degree-2 B-spline
    analogue). The low-resolution TPM sampled trilinearly imprints its own grid on the
    output (blocky priors → blocky posteriors); a smooth kernel removes that, so the
    output pass uses ``"cubic"``.

    Args:
        log_prior: ``(n_tissue, nz, ny, nx)`` from :func:`load_tpm`.
        coords: ``(n_vox, 3)`` TPM voxel coords ``(x, y, z)``.
        bg_low, bg_high: ``(n_tissue,)`` background probabilities.
        kernel: interpolation kernel (see above).

    Returns:
        ``(n_vox, n_tissue)`` tissue prior, rows summing to 1.
    """
    from . import interp as _interp

    if kernel == "linear":
        sampler = _interp.trilinear_interpolate
    else:
        resamplers = {
            "cubic": _interp.cubic_resample_3d,
            "quintic": _interp.quintic_resample_3d,
            "heptic": _interp.heptic_resample_3d,
            "wsinc5": _interp.wsinc5_resample_3d,
        }
        sampler = resamplers[kernel]

    n_tissue, nz, ny, nx = log_prior.shape
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    in_bounds = (x >= 0) & (x <= nx - 1) & (y >= 0) & (y <= ny - 1) & (z >= 0) & (z <= nz - 1)
    below = z < 0
    out_dtype = coords.dtype  # follow the caller's precision (float32 fit / float64 checks)

    if kernel == "linear":
        # one grid_sample for all tissues (shared sample locations) — the hot path.
        # grid_sample uses padding_mode="border", so `sampled` is continuous as coords
        # exit the volume (edge value extends outward); the background substitution below
        # is ramped over one voxel so the whole prior stays continuous & differentiable —
        # essential for the Gauss-Newton warp line search, which globally rejects any step
        # that raises the objective. A hard in/out switch cliffs the objective at voxels
        # sitting exactly on the boundary and stalls the solver.
        sampled = torch.exp(_interp.trilinear_interpolate_multi(log_prior, x, y, z).to(out_dtype))
        bg = torch.where(below[:, None], bg_low.to(out_dtype), bg_high.to(out_dtype))
        # distance (in voxels) outside the [0, N-1] box along each axis, 0 when inside
        relu = torch.nn.functional.relu
        outside = (
            relu(-x)
            + relu(x - (nx - 1))
            + relu(-y)
            + relu(y - (ny - 1))
            + relu(-z)
            + relu(z - (nz - 1))
        )
        w = outside.clamp(0.0, 1.0).to(out_dtype)[:, None]  # 0 inside → 1 a voxel out
        prior = (1.0 - w) * sampled + w * bg
    else:
        # higher-order resamplers size their output to the source dtype (output pass only)
        xs, ys, zs = x.to(log_prior.dtype), y.to(log_prior.dtype), z.to(log_prior.dtype)
        cols = [torch.exp(sampler(log_prior[k], xs, ys, zs).to(out_dtype)) for k in range(n_tissue)]
        sampled = torch.stack(cols, dim=1)
        prior = torch.where(in_bounds[:, None], sampled, bg_high.to(out_dtype))
        prior = torch.where(below[:, None], bg_low.to(out_dtype), prior)

    total = prior.sum(dim=1, keepdim=True).clamp_min(1e-40)
    return prior / total


def weight_prior(prior: Tensor, wp: Tensor) -> Tensor:
    """Apply per-tissue weights ``wp`` and renormalise (SPM ``log_spatial_priors``).

    The tissue weights let the model deviate from the raw TPM mixing proportions
    (e.g. a subject with more CSF than the template) — ``prior_w ∝ wp·prior``.
    """
    weighted = prior * wp.to(prior.dtype)
    return weighted / weighted.sum(dim=1, keepdim=True).clamp_min(1e-40)


# ---------------------------------------------------------------------------
# Deformation regulariser — membrane + bending energy on the dense field.
# ---------------------------------------------------------------------------


def warp_penalty(
    field: Tensor, reg: tuple[float, float, float], vox: tuple[float, float, float]
) -> Tensor:
    """Smoothness penalty on a dense displacement field ``(gz, gy, gx, 3)``.

    ``reg = (absolute, membrane, bending)`` weights three energies (SPM's first
    three ``reg`` terms; linear-elastic ``le1/le2`` are deferred to the GN solver):

    - **absolute** ``Σ u²`` — pulls the field toward zero (keeps it from drifting).
    - **membrane** ``Σ |∇u|²`` — penalises stretch, the dominant smoothness term.
    - **bending** ``Σ |∇²u|²`` — penalises curvature for a smooth, fold-free warp.

    Gradients are finite differences in grid units scaled by ``vox`` (mm) so the
    penalty is physically meaningful regardless of the subsampling.
    """
    w_abs, w_mem, w_bend = reg
    penalty = field.new_zeros(())
    if w_abs:
        penalty = penalty + w_abs * (field**2).sum()
    for axis, step in zip((0, 1, 2), (vox[2], vox[1], vox[0]), strict=True):
        if field.shape[axis] < 3:
            continue
        d1 = torch.diff(field, dim=axis) / step
        if w_mem:
            penalty = penalty + w_mem * (d1**2).sum()
        if w_bend and field.shape[axis] >= 3:
            d2 = torch.diff(field, n=2, dim=axis) / (step * step)
            penalty = penalty + w_bend * (d2**2).sum()
    return penalty


def _cg_solve(beta: Tensor, apply_a, n_iter: int = 24, tol: float = 1e-6) -> Tensor:
    """Conjugate gradient for the SPD system ``A·x = beta`` (``apply_a`` computes ``A·p``).

    Used by the Gauss-Newton warp solver to invert ``(Alpha + L)`` — the per-node GN
    Hessian plus the regularisation operator — the way SPM's ``spm_field`` multigrid does,
    but matrix-free on the GPU (``apply_a`` is a couple of tensor ops + one autograd pass
    for the regulariser). Starts from ``x=0`` so the initial residual is ``beta``.
    """
    x = torch.zeros_like(beta)
    r = beta.clone()
    p = r.clone()
    rs = (r * r).sum()
    rs0 = rs.clone()
    for _ in range(n_iter):
        ap = apply_a(p)
        alpha = rs / (p * ap).sum().clamp_min(1e-30)
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = (r * r).sum()
        if rs_new <= tol * tol * rs0:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


# ---------------------------------------------------------------------------
# EM driver — moment init + interleaved GMM / bias / warp updates.
# ---------------------------------------------------------------------------


def _build_tissue_of(ngaus: list[int]) -> Tensor:
    """Expand a per-tissue Gaussian count into the flat ``tissue_of`` (``lkp``)."""
    idx = [t for t, k in enumerate(ngaus) for _ in range(k)]
    return torch.tensor(idx, dtype=torch.long)


def _moment_init(
    corrected: Tensor,
    prior_w: Tensor,
    ngaus: list[int],
    cov_prior: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Initialise Gaussians from TPM-prior-weighted moments (SPM's moment init).

    One Gaussian per tissue from ``Σ prior·y`` moments (shared, ``vr0``-regularised
    variance), then each tissue split into ``ngaus[t]`` jittered components.
    """
    n_chan = corrected.shape[1]
    n_tissue = prior_w.shape[1]
    device = corrected.device

    count = prior_w.sum(dim=0)  # (n_tissue,)
    sum1 = torch.einsum("vn,vt->nt", corrected, prior_w)
    mean1 = sum1 / count.clamp_min(1e-30)  # (n_chan, n_tissue)
    # shared within-tissue scatter, regularised toward the data covariance
    scatter = corrected.new_zeros((n_chan, n_chan))
    for t in range(n_tissue):
        s2 = torch.einsum("vn,vm->nm", corrected * prior_w[:, t : t + 1], corrected)
        scatter = scatter + (s2 - count[t] * torch.outer(mean1[:, t], mean1[:, t]))
    var1 = (scatter + n_chan * cov_prior) / (count.sum() + n_chan)  # (n_chan, n_chan)

    tissue_of = _build_tissue_of(ngaus)
    n_gauss = tissue_of.numel()
    means = torch.empty((n_chan, n_gauss), dtype=torch.float64, device=device)
    covs = torch.empty((n_chan, n_chan, n_gauss), dtype=torch.float64, device=device)
    mix = torch.empty(n_gauss, dtype=torch.float64, device=device)
    chol = torch.linalg.cholesky(var1)
    gen = torch.Generator(device="cpu").manual_seed(0)
    for t in range(n_tissue):
        cols = (tissue_of == t).nonzero(as_tuple=True)[0]
        kk = cols.numel()
        w = 1.0 / (1.0 + torch.exp(torch.tensor(-(kk - 1) * 0.25))) - 0.5  # 0 when kk==1
        jitter = torch.randn(n_chan, kk, generator=gen).to(device=device, dtype=torch.float64)
        means[:, cols] = (chol @ jitter) * w + mean1[:, t : t + 1]
        covs[:, :, cols] = (var1 * (1.0 - w))[:, :, None]
        mix[cols] = 1.0 / kk
    return means, covs, mix


def fit_segment(
    volume: Tensor | list[Tensor],
    subj_affine: Tensor,
    log_prior: Tensor,
    tpm_affine: Tensor,
    bg_low: Tensor,
    bg_high: Tensor,
    world_affine: Tensor | None = None,
    *,
    vox2vox: Tensor | None = None,
    ngaus: list[int] | None = None,
    biasreg: float = 1e-4,
    biasfwhm: float = 60.0,
    reg: tuple[float, float, float] = (0.0, 0.0, 0.1),
    samp: float = 3.0,
    n_iter: int = 20,
    tol: float = 1e-4,
    fit_warp: bool = True,
    pe_axis: int | None = None,
    blur_tpms: float = 0.0,
    blur_frac: float = 0.4,
    warp_focus: float = 0.0,
    focus_quantile: float = 0.9,
    wp_reg: float = 100.0,
    bias_lr: float = 0.1,
    warp_solver: str = "adam",
    warp_lr: float = 1.0,
    warp_iters: int = 8,
    warp_smooth: float = 0.8,
    fit_chunk: int | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    verbose: bool = True,
) -> dict:
    """Fit the Unified Segmentation model to one or more volumes (Phase-1 EM driver).

    ``volume`` is a single 3-D tensor, a 4-D ``(n_chan, nz, ny, nx)`` stack, or a list of
    3-D tensors — the **multi-spectral** case (e.g. T1 + T2 + PD), which must already be
    aligned on the same grid. Each channel gets its own bias field; the Gaussian mixture
    is joint across channels (full ``n_chan × n_chan`` covariances), exactly as SPM.

    Interleaves the closed-form GMM E/M step with autograd (Adam) updates of the DCT
    bias field and the dense deformation, at a subsampled grid (``samp`` mm). The
    subject→template alignment is fixed here (the deformation refines the residual);
    supply it either as ``world_affine`` (subject→TPM world, from ffs_allineate/
    spm_maff8) or directly as ``vox2vox`` (subject voxel → TPM voxel, e.g. from an
    ``.aff12.1D`` chain via ``load_matrix_chain``). Exactly one is required.

    ``n_iter``/``tol`` cap and early-stop the EM (relative log-likelihood change);
    ``warp_smooth`` (grid-node sigma) Sobolev-smooths the deformation each iteration so
    the dense field stays smooth rather than speckly. ``pe_axis`` (0/1/2 = x/y/z), when
    set, constrains the deformation to that single voxel axis — the phase-encode
    direction — so the warp is a pure 1-D EPI distortion field (PE-mode, like ffs_rbr).

    ``blur_tpms`` (TPM-voxel sigma) runs a **coarse-to-fine** schedule: the first
    ``blur_frac`` of the iterations use a spatially-blurred TPM (:func:`blur_log_prior`)
    so the warp has a wide capture basin, then the remaining iterations switch to the
    sharp TPM to refine. This rescues cases where a subject with large residual
    distortion (an under-corrected EPI) would otherwise leave the warp stuck far from
    the true tissue boundary with no gradient to follow. ``blur_tpms=0`` disables it.

    ``warp_focus`` (0..1) makes the warp **work harder on localised misfit**. Each
    iteration the worst-fitting grid nodes — where the warped TPM prior disagrees most
    with the intensity-driven tissue posterior, i.e. a stretched peninsula of residual
    distortion — have their Sobolev smoothing relaxed by up to ``warp_focus`` (1 = no
    smoothing there, 0 = the uniform default everywhere). Only nodes above
    ``focus_quantile`` of the misfit distribution are relaxed, and the relaxation map is
    itself smoothed into a coherent blob, so the peninsula deforms hard while the rest
    of the field stays stable. ``warp_focus=0`` disables it (default).

    ``wp_reg`` regularises the tissue mixing weights toward uniform (SPM's ``wp_reg``,
    default 100). The weights follow SPM's self-correcting ``observed/expected`` update;
    a small ``wp_reg`` lets a tissue's weight run away over many iterations (positive
    feedback that inflates it past its boundary — "growing brains"), so keep it high.

    ``warp_solver`` selects the deformation optimiser. ``"adam"`` (default) is the
    autograd + Sobolev-smoothing update — fast, robust, the knobs below apply.
    ``"gn"`` is SPM's **Gauss-Newton** update (``spm_preproc8``): per-node gradient +
    rank-1 GN Hessian ``dp·dpᵀ``, the regularisation operator added, solved by conjugate
    gradient (in place of SPM's multigrid), with Armijo backtracking so every step is
    guaranteed to improve the penalised likelihood. GN needs no learning rate, converges
    in far fewer sub-iterations (set ``warp_iters`` to ~2-3), and — being properly
    regularised and monotone — does not inflate at high iteration counts the way a free
    Adam warp can. GN ignores ``warp_lr``/``warp_smooth``/``warp_focus`` (it regularises
    through ``reg`` instead); give ``reg`` a non-zero membrane term for a well-posed solve.

    Global warp **aggressiveness** (how far the deformation moves overall, distinct from
    the localised ``warp_focus``) is set by four knobs: ``warp_lr`` (Adam step size,
    default 1.0 — the most direct dial), ``warp_iters`` (gradient steps per EM iteration,
    default 8), and the two dampers ``reg`` (smoothness penalty; lower the bending term to
    allow larger folds) and ``warp_smooth`` (per-iteration Sobolev sigma; lower lets more
    data-driven displacement survive). Raise ``warp_lr``/``warp_iters`` and/or lower
    ``reg``/``warp_smooth`` to warp more.

    All per-sample work (GMM moments, bias/warp gradients, tissue-weight update) is
    processed in chunks of ``fit_chunk`` samples so peak VRAM is bounded by the chunk,
    not the sample count — a fine ``samp`` (e.g. 1 mm on sub-mm data, millions of
    samples) no longer OOMs. ``fit_chunk=None`` sizes it from free memory via
    [[Memory module]]; when all samples fit in one chunk the loop runs once, so the
    common ``samp=3`` case is unaffected. The chunked accumulation is exact — the sums
    and autograd gradients are identical to the whole-batch result up to fp round-off.

    Returns a dict with the fitted params (``means``/``covs``/``mix``/``tissue_of``/
    ``wp``/bias ``coef``/dense ``twarp``) and the geometry needed to expand tissue
    posteriors to full resolution (:func:`segment_apply`).
    """
    device = torch.device(device)
    # Working dtype for the hot path (interp, bias/warp autograd, likelihood): float32 by
    # default — on consumer GPUs float64 runs at a small fraction of float32 throughput,
    # and the fit's autograd graph dominates. Numerically load-bearing steps (the
    # covariance factorisation and the moment reduction) stay float64 internally.
    wdt = dtype
    # Accept one volume (3-D) or several ALIGNED channels (4-D ``(n_chan,nz,ny,nx)`` or a
    # list) — SPM's multi-spectral segmentation. Every channel shares the grid + affine.
    if isinstance(volume, (list, tuple)):
        vols = torch.stack([v.to(device=device, dtype=wdt) for v in volume], dim=0)
    else:
        volume = volume.to(device=device, dtype=wdt)
        vols = volume if volume.ndim == 4 else volume[None]
    n_chan = vols.shape[0]
    # keep every operand on the fit device (the interpolator needs prior + coords to
    # share a device); callers may hand us CPU tensors even for a cuda fit.
    log_prior = log_prior.to(device=device, dtype=wdt)
    bg_low, bg_high = bg_low.to(device), bg_high.to(device)
    nz, ny, nx = vols.shape[1:]
    vox = tuple(float(v) for v in torch.linalg.norm(subj_affine[:3, :3].to(torch.float64), dim=0))
    n_tissue = log_prior.shape[0]
    if ngaus is None:
        ngaus = [1, 1, 2, 3, 4, 2][:n_tissue] + [1] * max(0, n_tissue - 6)

    # subsampled brain grid (samples that are finite AND non-zero in EVERY channel)
    sk = [max(1, round(samp / v)) for v in vox]
    gz = torch.arange(0, nz, sk[2], device=device)
    gy = torch.arange(0, ny, sk[1], device=device)
    gx = torch.arange(0, nx, sk[0], device=device)
    zz, yy, xx = torch.meshgrid(gz, gy, gx, indexing="ij")
    grid_shape = zz.shape  # (ngz, ngy, ngx)
    fz, fy, fx = zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)
    coords = torch.stack([fx, fy, fz], dim=1).to(wdt)
    intens = vols[:, fz, fy, fx].T.contiguous()  # (n_samp, n_chan)
    keep = torch.isfinite(intens).all(dim=1) & (intens != 0).all(dim=1)
    coords, intens = coords[keep], intens[keep]

    if vox2vox is None:
        if world_affine is None:
            raise ValueError("fit_segment needs either world_affine or vox2vox")
        vox2vox = compose_vox2vox(
            subj_affine.to(device), tpm_affine.to(device), world_affine.to(device)
        )
    else:
        vox2vox = vox2vox.to(device=device, dtype=torch.float64)

    # bias field basis shape (SPM 1-based DCT positions) + precision; the per-sample
    # bases themselves are (re)built per chunk in _basis, not stored full-size — at a
    # fine samp they would be the single largest persistent allocation.
    (nbx, nby, nbz), bias_prec = bias_field_shape(
        (nx, ny, nz), vox, biasfwhm, biasreg, device=device
    )

    # Wishart covariance prior vr0 = diag(per-channel variance)/Kb² (SPM eq. after 25);
    # kept float64 for the M-step. Diagonal even for multi-channel (SPM's vr0).
    data_var = intens.var(dim=0, unbiased=False).to(torch.float64)  # (n_chan,)
    cov_prior = torch.diag(data_var / n_tissue**2)  # (n_chan, n_chan)

    tissue_of = _build_tissue_of(ngaus).to(device)
    # one bias field per channel (SPM's chan(n).T): (n_chan, nbx, nby, nbz)
    coef = torch.zeros((n_chan, nbx, nby, nbz), dtype=wdt, device=device)
    twarp = torch.zeros((*grid_shape, 3), dtype=wdt, device=device)
    keep_grid = keep  # to index the flat grid ↔ warp field
    wp = torch.ones(n_tissue, dtype=wdt, device=device) / n_tissue

    log_prior_blur = blur_log_prior(log_prior, blur_tpms) if blur_tpms > 0 else log_prior
    n_coarse = int(round(blur_frac * n_iter)) if blur_tpms > 0 else 0

    # --- sample chunking: bound working VRAM so a fine samp doesn't OOM ---
    # Size via the shared estimator ([[Memory module]]). Model the per-sample footprint
    # as the working tensors (~n_gauss + n_tissue reductions) plus headroom for the
    # grid_sample autograd graph, and lift the estimator's default 90k GPU cap
    # (max_chunk_size=n_samp) — that cap, not our memory, was needlessly splitting data
    # that fits many times over.
    n_samp = coords.shape[0]
    if fit_chunk is None:
        from ..memory import estimate_chunk_size

        fit_chunk = estimate_chunk_size(
            n_samp,
            1,
            (tissue_of.numel() + n_tissue) * 8,
            device,
            operation="glm",
            use_double=(wdt == torch.float64),
            max_chunk_size=n_samp,
        )
    fit_chunk = max(1, min(int(fit_chunk), n_samp))
    n_chunks = (n_samp + fit_chunk - 1) // fit_chunk
    kept_flat = keep_grid.nonzero(as_tuple=True)[0]  # flat grid index of each kept sample
    if verbose:
        how = "single pass — all samples fit" if n_chunks == 1 else f"{n_chunks} chunks"
        print(
            f"segment fit: {n_samp:,} samples ({samp:g} mm, sk={sk}), {wdt}, "
            f"chunk={fit_chunk:,} → {how}"
        )

    def _chunks():
        for s in range(0, n_samp, fit_chunk):
            yield slice(s, min(s + fit_chunk, n_samp))

    def _basis(sl: slice) -> tuple[Tensor, Tensor, Tensor]:
        return (
            dct_basis(coords[sl, 0] + 1.0, nx, nbx),
            dct_basis(coords[sl, 1] + 1.0, ny, nby),
            dct_basis(coords[sl, 2] + 1.0, nz, nbz),
        )

    def corrected_full(coef_val: Tensor) -> Tensor:
        """Bias-corrected intensities at every sample (constant within a phase)."""
        out = torch.empty((n_samp, n_chan), dtype=wdt, device=device)
        with torch.no_grad():
            for sl in _chunks():
                bx, by, bz = _basis(sl)
                for c in range(n_chan):
                    out[sl, c] = intens[sl, c] * torch.exp(eval_log_bias(coef_val[c], bx, by, bz))
        return out

    def warped_prior_full(active_log_prior: Tensor) -> Tensor:
        """Weighted warped-TPM prior at every sample (built chunked, then reused)."""
        out = torch.empty((n_samp, n_tissue), dtype=wdt, device=device)
        with torch.no_grad():
            for sl in _chunks():
                disp = twarp.reshape(-1, 3)[kept_flat[sl]]
                tpm_coords = apply_affine_pts(coords[sl] + disp, vox2vox)
                out[sl] = weight_prior(
                    sample_tpm_prior(active_log_prior, tpm_coords, bg_low, bg_high), wp
                )
        return out

    # --- Gauss-Newton warp helpers (used only when warp_solver == "gn") ---
    # a small absolute term guarantees the (Alpha + L) system is SPD for the CG solve
    # even when the user's reg has no membrane/absolute component (bending has a null
    # space of affine fields). Consistent across gradient, operator and objective.
    reg_gn = (max(reg[0], 1e-3), reg[1], reg[2])

    def _pe_project(field: Tensor) -> Tensor:
        """Zero the non-phase-encode components in place (PE-mode constraint)."""
        if pe_axis is not None:
            for c in range(3):
                if c != pe_axis:
                    field[..., c] = 0.0
        return field

    def _warp_data_term(tw: Tensor, active_log_prior: Tensor, *, want_grad: bool):
        """Chunked data term ``-Σ log-likelihood(warp)`` (+ its autograd gradient)."""
        tw = tw.detach().requires_grad_(want_grad)
        total = 0.0
        for sl in _chunks():
            disp = tw.reshape(-1, 3)[kept_flat[sl]]
            tpm_coords = apply_affine_pts(coords[sl] + disp, vox2vox)
            pw = weight_prior(sample_tpm_prior(active_log_prior, tpm_coords, bg_low, bg_high), wp)
            _, ll = gmm_responsibilities(corrected[sl], pw, means, covs, mix, tissue_of)
            nll = -ll.sum()
            if want_grad:
                nll.backward()
            total += nll.item()
        if not want_grad:
            return total
        grad = tw.grad if tw.grad is not None else torch.zeros_like(tw)
        return total, grad.detach()

    def _reg_grad(field: Tensor) -> Tensor:
        """Regularisation operator ``L·field`` = gradient of the (quadratic) warp penalty."""
        v = field.detach().requires_grad_(True)
        (g,) = torch.autograd.grad(warp_penalty(v, reg_gn, vox), v)
        return g.detach()

    def _warp_objective(tw: Tensor, active_log_prior: Tensor) -> float:
        """Penalised negative log-likelihood the GN line search must decrease."""
        return (
            _warp_data_term(tw, active_log_prior, want_grad=False)
            + warp_penalty(tw, reg_gn, vox).item()
        )

    means = covs = mix = None
    prev_ll = -float("inf")
    bar = (
        _tqdm(total=n_iter, desc="segment EM", leave=True)
        if (_tqdm and verbose and n_iter >= 5)
        else None
    )
    for it in range(n_iter):
        # coarse-to-fine: blurred TPM for the first n_coarse iters, then sharp
        cur_log_prior = log_prior_blur if it < n_coarse else log_prior
        prior_w = warped_prior_full(cur_log_prior)  # (n_samp, n_tissue), constant this iter
        if means is None:
            means, covs, mix = _moment_init(intens, prior_w, ngaus, cov_prior)

        # --- GMM closed-form EM (a few sub-iterations) ---
        # corrected is constant across the sub-iterations (coef fixed); build it once.
        corrected = corrected_full(coef)
        for _ in range(20):
            count = sum1 = sum2 = None
            for sl in _chunks():
                resp_c, _ = gmm_responsibilities(
                    corrected[sl], prior_w[sl], means, covs, mix, tissue_of
                )
                cc, s1, s2 = gmm_moments(corrected[sl], resp_c)
                count = cc if count is None else count + cc
                sum1 = s1 if sum1 is None else sum1 + s1
                sum2 = s2 if sum2 is None else sum2 + s2
            new_means, new_covs, new_mix = gmm_update(count, sum1, sum2, tissue_of, cov_prior)
            delta = (new_means - means).abs().max()
            means, covs, mix = new_means, new_covs, new_mix
            if delta < 1e-3:
                break

        # --- bias field (autograd: maximise data ll − penalty; grad accumulated) ---
        coef = coef.detach().requires_grad_(True)
        opt = torch.optim.Adam([coef], lr=bias_lr)
        for _ in range(8):
            opt.zero_grad()
            for sl in _chunks():
                bx, by, bz = _basis(sl)
                corrected_c = torch.stack(
                    [
                        intens[sl, c] * torch.exp(eval_log_bias(coef[c], bx, by, bz))
                        for c in range(n_chan)
                    ],
                    dim=1,
                )  # (chunk, n_chan)
                _, ll = gmm_responsibilities(corrected_c, prior_w[sl], means, covs, mix, tissue_of)
                (-ll.sum()).backward()  # sum of chunk grads == whole-batch grad
            # per-channel DCT smoothness prior, summed (SPM sums chan(n).ll)
            torch.stack(
                [bias_penalty_value(coef[c], bias_prec) for c in range(n_chan)]
            ).sum().backward()
            opt.step()
        coef = coef.detach()
        del opt

        # corrected under the updated bias — constant for the warp + wp steps below
        corrected = corrected_full(coef)

        # --- deformation ---
        if fit_warp and warp_solver == "gn":
            # SPM Gauss-Newton: Beta = data-grad + L·twarp; per-node GN Hessian Alpha =
            # dp·dpᵀ (rank-1); Update = (Alpha + L)⁻¹ Beta via CG; Armijo backtracking so
            # every accepted step lowers the penalised negative log-likelihood.
            for _ in range(warp_iters):
                _, g_data = _warp_data_term(twarp, cur_log_prior, want_grad=True)
                g_data = _pe_project(g_data)  # per-node data gradient (= -dp)
                beta = g_data + _reg_grad(twarp)

                def _apply_a(pv: Tensor, _g: Tensor = g_data) -> Tensor:
                    # (Alpha + L)·p: per-node rank-1 GN Hessian dp·dpᵀ, plus L·p
                    ap = _g * (_g * pv).sum(dim=-1, keepdim=True) + _reg_grad(pv)
                    return _pe_project(ap)

                update = _pe_project(_cg_solve(beta, _apply_a))
                base = _warp_objective(twarp, cur_log_prior)
                armijo, improved = 1.0, False
                for _ in range(8):  # backtracking line search (SPM's Armijo)
                    cand = (twarp - armijo * update).detach()
                    if _warp_objective(cand, cur_log_prior) < base:
                        twarp, improved = cand, True
                        break
                    armijo *= 0.75
                if not improved:
                    break  # no downhill step this EM iteration — stop refining the warp
        elif fit_warp:
            twarp = twarp.detach().requires_grad_(True)
            opt = torch.optim.Adam([twarp], lr=warp_lr)
            for _ in range(warp_iters):
                opt.zero_grad()
                for sl in _chunks():
                    disp = twarp.reshape(-1, 3)[kept_flat[sl]]
                    tpm_coords = apply_affine_pts(coords[sl] + disp, vox2vox)
                    pw = weight_prior(
                        sample_tpm_prior(cur_log_prior, tpm_coords, bg_low, bg_high), wp
                    )
                    _, ll = gmm_responsibilities(corrected[sl], pw, means, covs, mix, tissue_of)
                    (-ll.sum()).backward()
                pen = warp_penalty(twarp, reg, vox)  # smoothness prior, once
                if pen.requires_grad:  # all-zero reg → constant, nothing to backprop
                    pen.backward()
                if pe_axis is not None:
                    # PE-mode: only the phase-encode component may move (distortion is
                    # 1-D along the PE axis, like ffs_rbr) — zero the other two grads.
                    for c in range(3):
                        if c != pe_axis and twarp.grad is not None:
                            twarp.grad[..., c] = 0.0
                opt.step()
            del opt
            # focus map: relax smoothing on the worst-fitting nodes so a localised
            # peninsula of residual distortion can deform hard (see warp_focus)
            keep_weight = None
            if warp_focus > 0:
                with torch.no_grad():
                    misfit = torch.empty(n_samp, dtype=wdt, device=device)
                    for sl in _chunks():
                        disp = twarp.detach().reshape(-1, 3)[kept_flat[sl]]
                        tpm_coords = apply_affine_pts(coords[sl] + disp, vox2vox)
                        pw = weight_prior(
                            sample_tpm_prior(cur_log_prior, tpm_coords, bg_low, bg_high), wp
                        )
                        resp_f, _ = gmm_responsibilities(
                            corrected[sl], pw, means, covs, mix, tissue_of
                        )
                        tpost = torch.zeros_like(pw)  # collapse Gaussians → per-tissue posterior
                        tpost.index_add_(1, tissue_of, resp_f)
                        # per-node prior/posterior overlap; low overlap = the prior sits in
                        # the wrong place here (intensity says one tissue, the TPM another)
                        misfit[sl] = (1.0 - (tpost * pw).sum(dim=1)).clamp(0.0, 1.0)
                    thr = _quantile(misfit, focus_quantile)
                    rel = ((misfit - thr) / (misfit.max() - thr).clamp_min(1e-6)).clamp(0.0, 1.0)
                    mf = torch.zeros(keep_grid.numel(), dtype=wdt, device=device)
                    mf[keep_grid] = rel * warp_focus
                    # smooth the relaxation into a coherent blob (no single-node spikes)
                    kw = mf.reshape(grid_shape)[..., None].repeat(1, 1, 1, 3)
                    keep_weight = _smooth_field(kw, warp_smooth)[..., 0].clamp(0.0, 1.0)
            # Sobolev smoothing: keep the dense field smooth (no per-node speckle),
            # except where keep_weight relaxes it
            twarp = _smooth_field(twarp.detach(), warp_smooth, keep_weight)

        # --- tissue weights wp + convergence ll ---
        # SPM spm_preproc8 (eq. wp update): wp_k = (observed_k + wp_reg)/(expected_k +
        # wp_reg·Kb), normalised. observed_k = Σ soft-counts of tissue k; expected_k =
        # mgm_k = Σ_v B(v,k)/(B(v,:)·wp) over the UNWEIGHTED warped TPM B. The ratio is
        # self-correcting (expected grows with wp), unlike observed/total-observed, which
        # positively feeds back and inflates a tissue past its boundary at high iteration
        # counts ("growing brains"). wp_reg=100 biases toward uniform 1/Kb.
        with torch.no_grad():
            tissue_mass = torch.zeros(n_tissue, dtype=torch.float64, device=device)
            mgm = torch.zeros(n_tissue, dtype=torch.float64, device=device)
            total_ll = 0.0
            wp64 = wp.to(torch.float64)
            for sl in _chunks():
                disp = twarp.reshape(-1, 3)[kept_flat[sl]]
                tpm_coords = apply_affine_pts(coords[sl] + disp, vox2vox)
                raw_prior = sample_tpm_prior(cur_log_prior, tpm_coords, bg_low, bg_high)
                pw = weight_prior(raw_prior, wp)
                resp_c, ll_c = gmm_responsibilities(corrected[sl], pw, means, covs, mix, tissue_of)
                tissue_mass.index_add_(0, tissue_of, resp_c.sum(dim=0).to(torch.float64))
                b64 = raw_prior.to(torch.float64)
                s = 1.0 / (b64 @ wp64).clamp_min(1e-40)  # 1 / (B·wp) per voxel
                mgm += (b64 * s[:, None]).sum(dim=0)  # Σ_v B(v,k)/(B·wp) = expected mass
                total_ll += ll_c.sum().item()
            wp = (tissue_mass + wp_reg) / (mgm + wp_reg * n_tissue)
            wp = (wp / wp.sum()).to(wdt)

        ll_per_vox = total_ll / n_samp
        if bar is not None:
            bar.set_postfix(ll_vox=f"{ll_per_vox:+.4f}")
            bar.update(1)
        elif verbose:
            print(f"iter {it + 1:2d}/{n_iter}  ll/vox {ll_per_vox:+.4f}")
        # don't early-stop inside the coarse (blurred-TPM) phase — the ll there is on a
        # different objective; only converge once the sharp TPM is in play
        if abs(total_ll - prev_ll) < tol * abs(total_ll) and it > 2 and it >= n_coarse:
            break
        prev_ll = total_ll
    if bar is not None:
        bar.close()

    return {
        "means": means,
        "covs": covs,
        "mix": mix,
        "tissue_of": tissue_of,
        "wp": wp,
        "ll": ll_per_vox,
        "coef": coef,
        "twarp": twarp,
        "grid_shape": grid_shape,
        "sk": sk,
        "vox": vox,
        "vox2vox": vox2vox,
        "bias_shape": (nbx, nby, nbz),
        "n_tissue": n_tissue,
        "n_chan": n_chan,
        "n_samp": n_samp,
        "fit_chunk": fit_chunk,
        "n_chunks": n_chunks,
    }


def segment_apply(
    volume: Tensor | list[Tensor],
    log_prior: Tensor,
    bg_low: Tensor,
    bg_high: Tensor,
    fit: dict,
    *,
    mrf: float = 1.0,
    cleanup: int = 1,
    debridge: int = 0,
    prior_kernel: str = "cubic",
    save_precleanup: bool = False,
    device: torch.device | str | None = None,
    chunk: int | None = None,
    verbose: bool = True,
) -> dict:
    """Expand a fitted model (:func:`fit_segment`) to full-resolution outputs.

    Evaluates the bias field, upsamples the dense deformation, samples the warped
    TPM, and computes tissue posteriors at **every** voxel — chunked over voxels so a
    large volume ([[Data Variety]]: up to millions of voxels) fits in VRAM. Then the
    optional post-passes, in order: :func:`mrf_cleanup` (``mrf=0`` disables),
    :func:`debridge_gm` thin-dura removal (``debridge=0`` disables), and
    :func:`clean_gwc` morphological brain extraction (``cleanup=0`` disables).

    ``save_precleanup`` additionally returns the posteriors *before* any post-pass under
    ``posteriors_precleanup`` (so the cleanup can be inspected / undone).

    ``volume`` matches :func:`fit_segment` — one volume or the aligned multi-channel
    stack/list. ``corrected``/``bias`` are then per channel (``(n_chan, nz, ny, nx)``),
    squeezed to 3-D for the single-channel case.

    Returns:
        posteriors: ``(n_tissue, nz, ny, nx)`` per-tissue posterior probability.
        corrected: bias-corrected intensity — ``(nz, ny, nx)`` or ``(n_chan, nz, ny, nx)``.
        bias: multiplicative bias field — same shape as ``corrected``.
        posteriors_precleanup: pre-post-pass posteriors (only if ``save_precleanup``).
    """
    from .interp import trilinear_interpolate

    if device is None:
        device = volume[0].device if isinstance(volume, (list, tuple)) else volume.device
    device = torch.device(device)
    # normalise to a channel stack (n_chan, nz, ny, nx) — matches fit_segment's input
    if isinstance(volume, (list, tuple)):
        vols = torch.stack([v.to(device) for v in volume], dim=0)
    else:
        volume = volume.to(device)
        vols = volume if volume.ndim == 4 else volume[None]
    n_chan = vols.shape[0]
    nz, ny, nx = vols.shape[1:]
    n_tissue = fit["n_tissue"]
    sk = fit["sk"]
    nbx, nby, nbz = fit["bias_shape"]
    coef = fit["coef"].to(device)  # (n_chan, nbx, nby, nbz)
    twarp = fit["twarp"].to(device)  # (gz, gy, gx, 3)
    vox2vox = fit["vox2vox"].to(device)
    means, covs, mix, tissue_of = (fit["means"], fit["covs"], fit["mix"], fit["tissue_of"])
    wp = fit["wp"].to(device)

    n_vox = nz * ny * nx
    if chunk is None:
        from ..memory import estimate_chunk_size

        chunk = estimate_chunk_size(
            n_vox, 1, 2 * (tissue_of.numel() + n_tissue), device, operation="glm", use_double=True
        )
    chunk = max(1, min(chunk, n_vox))

    posteriors = torch.empty((n_tissue, n_vox), dtype=torch.float32, device="cpu")
    corrected_out = torch.empty((n_chan, n_vox), dtype=torch.float32, device="cpu")
    bias_out = torch.empty((n_chan, n_vox), dtype=torch.float32, device="cpu")
    vols_flat = vols.reshape(n_chan, -1)

    twarp_c = [twarp[..., c].contiguous() for c in range(3)]  # (gz,gy,gx) each
    flat_idx = torch.arange(n_vox, device=device)
    starts = range(0, n_vox, chunk)
    if _tqdm is not None and verbose and n_vox > chunk:
        starts = _tqdm(starts, desc="segment apply", leave=True)
    for start in starts:
        idx = flat_idx[start : start + chunk]
        z = (idx // (ny * nx)).to(torch.float64)
        y = ((idx // nx) % ny).to(torch.float64)
        x = (idx % nx).to(torch.float64)
        intens = vols_flat[:, idx].to(torch.float64)  # (n_chan, chunk)

        # per-channel bias field (SPM 1-based DCT positions)
        bx = dct_basis(x + 1.0, nx, nbx)
        by = dct_basis(y + 1.0, ny, nby)
        bz = dct_basis(z + 1.0, nz, nbz)
        bias = torch.stack(
            [torch.exp(eval_log_bias(coef[c], bx, by, bz)) for c in range(n_chan)], dim=0
        )  # (n_chan, chunk)
        corrected = (intens * bias).T  # (chunk, n_chan)

        # upsample the deformation to these voxels (grid coords = full / sk)
        gxp, gyp, gzp = x / sk[0], y / sk[1], z / sk[2]
        disp = torch.stack(
            [trilinear_interpolate(twarp_c[c], gxp, gyp, gzp).to(torch.float64) for c in range(3)],
            dim=1,
        )
        coords = torch.stack([x, y, z], dim=1) + disp
        prior_w = weight_prior(
            sample_tpm_prior(
                log_prior, apply_affine_pts(coords, vox2vox), bg_low, bg_high, kernel=prior_kernel
            ),
            wp,
        )

        resp, _ = gmm_responsibilities(corrected, prior_w, means, covs, mix, tissue_of)
        # collapse Gaussians → per-tissue posterior in one scatter (no per-Gaussian loop)
        tpost = torch.zeros((idx.numel(), n_tissue), dtype=torch.float64, device=device)
        tpost.index_add_(1, tissue_of, resp)

        posteriors[:, start : start + idx.numel()] = tpost.T.to(torch.float32).cpu()
        corrected_out[:, start : start + idx.numel()] = corrected.T.to(torch.float32).cpu()
        bias_out[:, start : start + idx.numel()] = bias.to(torch.float32).cpu()

    posteriors = posteriors.reshape(n_tissue, nz, ny, nx)
    precleanup = posteriors.clone() if save_precleanup else None
    if mrf > 0 or cleanup > 0 or debridge > 0:
        posteriors = posteriors.to(device)
        if mrf > 0:
            posteriors = mrf_cleanup(posteriors, mrf, fit["vox"])
        if debridge > 0:  # strip thin dura bridges before the WM-seeded brain extraction
            posteriors = debridge_gm(posteriors, radius=debridge)
        if cleanup > 0:
            posteriors = clean_gwc(posteriors, level=cleanup)
        posteriors = posteriors.cpu()
    corrected_r = corrected_out.reshape(n_chan, nz, ny, nx)
    bias_r = bias_out.reshape(n_chan, nz, ny, nx)
    out = {
        "posteriors": posteriors,
        # squeeze the channel axis for the single-channel case (back-compatible shape)
        "corrected": corrected_r[0] if n_chan == 1 else corrected_r,
        "bias": bias_r[0] if n_chan == 1 else bias_r,
    }
    if precleanup is not None:
        out["posteriors_precleanup"] = precleanup
    return out


def mrf_cleanup(
    posteriors: Tensor,
    mrf: float,
    vox: tuple[float, float, float],
    *,
    n_iter: int = 10,
) -> Tensor:
    """Markov-random-field spatial cleanup of tissue posteriors (SPM ``spm_mrf``).

    A mean-field Potts iteration that pulls each voxel toward the tissue its
    neighbours agree on, cleaning salt-and-pepper misclassification and closing small
    holes. With SPM's diagonal connectivity ``G = mrf·I`` the per-voxel update is

        q_k ∝ p_k · exp(mrf · a_k),   a_k = (1/6) Σ_axis w_axis·(q_k[-] + q_k[+])

    where ``p`` is the fixed data-term posterior (the GMM × warped-TPM result), ``q``
    the iteratively-smoothed field, and ``w = 1/vox²`` handles anisotropy. ``mrf=0``
    returns the input unchanged.

    Args:
        posteriors: ``(n_tissue, nz, ny, nx)`` GMM data-term posteriors.
        mrf: connectivity strength (SPM default 1); 0 disables.
        vox: ``(vx, vy, vz)`` mm voxel sizes.
        n_iter: mean-field iterations (SPM uses 10).

    Returns:
        ``(n_tissue, nz, ny, nx)`` cleaned posteriors (each voxel sums to 1).
    """
    if mrf <= 0:
        return posteriors
    data = posteriors.to(torch.float32)
    field = data.clone()
    # posteriors are (tissue, z, y, x): axes 1,2,3 ↔ z,y,x → weights 1/vox_z², /vox_y², /vox_x²
    w = [1.0 / vox[2] ** 2, 1.0 / vox[1] ** 2, 1.0 / vox[0] ** 2]
    for _ in range(n_iter):
        neigh = torch.zeros_like(field)
        for axis, wa in zip((1, 2, 3), w, strict=True):
            lo = torch.roll(field, shifts=1, dims=axis)
            hi = torch.roll(field, shifts=-1, dims=axis)
            # zero the wrapped-around planes so edges see fewer neighbours (not toroidal)
            idx_lo = [slice(None)] * 4
            idx_lo[axis] = 0
            lo[tuple(idx_lo)] = 0.0
            idx_hi = [slice(None)] * 4
            idx_hi[axis] = field.shape[axis] - 1
            hi[tuple(idx_hi)] = 0.0
            neigh = neigh + wa * (lo + hi)
        neigh = neigh / 6.0
        field = data * torch.exp(mrf * neigh)
        field = field / field.sum(dim=0, keepdim=True).clamp_min(1e-20)
    return field


def _smooth3(vol: Tensor, kernel: Tensor) -> Tensor:
    """Separable 3-D convolution of a ``(nz, ny, nx)`` volume (replicate borders)."""
    import torch.nn.functional as F  # noqa: N812

    x = vol[None, None]  # (1,1,nz,ny,nx)
    for axis in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = kernel.numel()
        k = kernel.reshape(shape)
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - axis)] = 1
        pad[2 * (2 - axis) + 1] = 1
        x = F.conv3d(F.pad(x, pad, mode="replicate"), k)
    return x[0, 0]


def clean_gwc(posteriors: Tensor, level: int = 1) -> Tensor:
    """Ad-hoc morphological brain cleanup of GM/WM/CSF (SPM ``clean_gwc``).

    Grows a brain mask from the WM seed by conditional dilation through connected
    GM+WM (32 iterations of threshold → keep GM+WM mass → smooth), repeats including
    CSF, then zeroes GM/WM outside the GM+WM mask and CSF outside the GM+WM+CSF mask
    and renormalises. Strips dura/skull/eyeball voxels misclassified as brain tissue —
    the classic reason a GM or CSF map is unusable as a mask.

    Assumes the first three classes are GM, WM, CSF (SPM order); needs > 3 classes.
    ``level`` 1 (default) or 2 (stricter dilation threshold 0.2 vs 0.15).

    Args:
        posteriors: ``(n_tissue, nz, ny, nx)`` tissue posteriors (rows sum to 1).
        level: cleanup aggressiveness (1 or 2).

    Returns:
        ``(n_tissue, nz, ny, nx)`` cleaned posteriors (rows sum to 1).
    """
    if posteriors.shape[0] <= 3:
        return posteriors  # SPM: "Cleanup not done" — needs GM/WM/CSF + more
    dtype = torch.float32
    out = posteriors.to(dtype).clone()
    gm, wm, csf = out[0], out[1], out[2]
    kernel = torch.tensor([0.3, 0.4, 0.3], dtype=dtype, device=out.device)  # [.75 1 .75]/2.5
    th1 = 0.2 if level >= 2 else 0.15

    b = wm.clone()
    for j in range(32):
        th = 0.6 if j < 2 else th1  # erode twice, then conditionally dilate
        b = (b > th).to(dtype) * (wm + gm)
        b = _smooth3(b, kernel)
    c = b.clone()
    for _ in range(32):
        c = (c > th1).to(dtype) * (wm + gm + csf)
        c = _smooth3(c, kernel)

    th = 0.05
    brain = ((b > th).to(dtype) * (gm + wm)) > th
    csf_brain = ((c > th).to(dtype) * (gm + wm + csf)) > th
    out[0] = gm * brain
    out[1] = wm * brain
    out[2] = csf * csf_brain
    out = out / out.sum(dim=0, keepdim=True).clamp_min(1e-20)
    return out


def _binary_dilate(mask: Tensor, radius: int) -> Tensor:
    """Grow a boolean mask by ``radius`` voxels (cube structuring element, max-pool)."""
    import torch.nn.functional as F  # noqa: N812

    grown = F.max_pool3d(mask[None, None].float(), 2 * radius + 1, stride=1, padding=radius)
    return grown[0, 0] > 0.5


def _binary_erode(mask: Tensor, radius: int) -> Tensor:
    """Shrink a boolean mask by ``radius`` voxels (erosion = dilation of the complement)."""
    return ~_binary_dilate(~mask, radius)


def debridge_gm(
    posteriors: Tensor, radius: int = 1, *, gm_index: int = 0, thresh: float = 0.5
) -> Tensor:
    """Strip thin GM sheets/bridges (mislabelled dura) via a morphological opening.

    Cortical GM is a **thick** (several-voxel), convoluted sheet; mislabelled dura
    instead appears as **thin** (~1-2 voxel) sheets that bridge adjacent gyri or ring the
    brain periphery — topologically wrong "bridges" and "weird thin things". A binary
    **opening** (erode then dilate by ``radius``) of the GM mask deletes any structure
    thinner than ~``2·radius`` while leaving thick cortex intact, so the bridges vanish
    but the cortical ribbon stays. The removed GM probability is redistributed to the
    other classes (renormalise), so each stripped voxel falls back to whatever else is
    likely there (bone/soft/CSF) — effectively "recategorised" as non-brain / dura.

    This is more aggressive than SPM's ``clean_gwc`` (which only removes GM *disconnected*
    from the WM-seeded brain; a bridge that touches cortex survives it). ``radius=0``
    disables. Larger ``radius`` removes thicker structures — watch tight sulci, where real
    cortex is also thin. Assumes GM is class ``gm_index`` (0 in SPM order).
    """
    if radius <= 0:
        return posteriors
    out = posteriors.to(torch.float32).clone()
    gm = out[gm_index]
    mask = gm > thresh
    opened = _binary_dilate(_binary_erode(mask, radius), radius)  # remove thin structures
    removed = mask & ~opened
    out[gm_index] = torch.where(removed, torch.zeros_like(gm), gm)
    return out / out.sum(dim=0, keepdim=True).clamp_min(1e-20)


def full_resolution_warp(
    fit: dict, shape: tuple[int, int, int], *, device: torch.device | str = "cpu"
) -> Tensor:
    """Upsample the fitted deformation to the full subject grid (voxel-unit disp).

    The dense ``twarp`` is stored on the subsampled grid; this trilinearly upsamples
    it to every subject voxel and returns ``(nz, ny, nx, 3)`` displacements in subject
    voxel units — the ``(x, y, z)`` components ready for
    ``io.save_warp_field(..., units="mm")``, so the subject-space deformation drops
    straight into the composable ffs_nwarp chain (alongside the affine).
    """
    from .interp import trilinear_interpolate

    device = torch.device(device)
    nz, ny, nx = shape
    sk = fit["sk"]
    twarp = fit["twarp"].to(device)
    twarp_c = [twarp[..., c].contiguous() for c in range(3)]

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float64),
        torch.arange(ny, device=device, dtype=torch.float64),
        torch.arange(nx, device=device, dtype=torch.float64),
        indexing="ij",
    )
    gxp, gyp, gzp = xx.reshape(-1) / sk[0], yy.reshape(-1) / sk[1], zz.reshape(-1) / sk[2]
    comps = [trilinear_interpolate(twarp_c[c], gxp, gyp, gzp).reshape(nz, ny, nx) for c in range(3)]
    return torch.stack(comps, dim=-1)


def _fullres_grid(
    shape: tuple[int, int, int], device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    """Flattened full-resolution voxel coordinate grids ``(x, y, z)`` (float64)."""
    nz, ny, nx = shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float64),
        torch.arange(ny, device=device, dtype=torch.float64),
        torch.arange(nx, device=device, dtype=torch.float64),
        indexing="ij",
    )
    return xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)


def undistort_input(
    volume: Tensor, fit: dict, *, device: torch.device | str | None = None
) -> Tensor:
    """Geometry-correct the input by removing the fitted deformation (PE-mode: the
    EPI's distortion), in the input's own grid.

    The fit models ``prior(p) = TPM(affine·(p + disp(p)))``, so ``disp`` is the
    *negative* of the physical distortion; the corrected image is therefore
    ``corrected(p) = volume(p − disp(p))`` (pull sampling). In PE-mode ``disp`` is
    nonzero only on the phase-encode axis, so this is exactly the distortion-corrected
    EPI (analogous to ``ffs_rbr``'s ``epi_undistorted``). Independent of the affine —
    the correction is in the input's own space; compose the affine separately to reach
    template space.
    """
    from .interp import trilinear_interpolate

    device = torch.device(device) if device is not None else volume.device
    volume = volume.to(device)
    disp = full_resolution_warp(fit, tuple(volume.shape), device=device)  # (nz,ny,nx,3)
    x, y, z = _fullres_grid(tuple(volume.shape), device)
    out = trilinear_interpolate(
        volume,
        x - disp[..., 0].reshape(-1),
        y - disp[..., 1].reshape(-1),
        z - disp[..., 2].reshape(-1),
    )
    return out.reshape(volume.shape)


def input_in_template(
    volume: Tensor,
    fit: dict,
    out_shape: tuple[int, int, int],
    *,
    use_warp: bool = True,
    kernel: str = "linear",
    device: torch.device | str | None = None,
) -> Tensor:
    """Resample the input into the TEMPLATE's space (affine — and warp — applied).

    This is the ``epi_in_anat`` counterpart to :func:`undistort_input` (which stays in
    the input's own grid): here the input is placed on the template grid ``out_shape``,
    so it overlays the anat/template. Done in two steps — undistort in the input's own
    space (removing the nonlinear part), then apply the affine ``vox2vox`` to resample
    onto the template grid (after undistortion the two frames differ only by the
    affine). ``use_warp=False`` skips the undistortion → the affine-only "initial"
    cast, a before/after baseline for the deformation.

    Args:
        volume: the input volume (input grid).
        fit: result of :func:`fit_segment`.
        out_shape: template grid shape ``(nz, ny, nx)`` (e.g. the TPM's).
        use_warp: apply the deformation (corrected) or affine only (initial).
        kernel: interpolation for the template-grid resample.

    Returns:
        ``out_shape`` image — the input in template space (save with the TPM affine).
    """
    from . import interp as _interp

    device = torch.device(device) if device is not None else volume.device
    src = undistort_input(volume, fit, device=device) if use_warp else volume.to(device)
    tpm_to_input = torch.linalg.inv(fit["vox2vox"].to(torch.float64)).to(device)
    x, y, z = _fullres_grid(out_shape, device)
    coords = apply_affine_pts(torch.stack([x, y, z], dim=1), tpm_to_input)  # input voxel coords
    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
    if kernel == "linear":
        out = _interp.trilinear_interpolate(src, xs, ys, zs)
    else:
        sampler = {
            "cubic": _interp.cubic_resample_3d,
            "quintic": _interp.quintic_resample_3d,
            "heptic": _interp.heptic_resample_3d,
            "wsinc5": _interp.wsinc5_resample_3d,
        }[kernel]
        out = sampler(src, xs.to(src.dtype), ys.to(src.dtype), zs.to(src.dtype))
    return out.reshape(out_shape)


def cast_template_to_input(
    source: Tensor,
    source_affine: Tensor,
    tpm_affine: Tensor,
    fit: dict,
    out_shape: tuple[int, int, int],
    *,
    use_warp: bool = True,
    kernel: str = "linear",
    device: torch.device | str | None = None,
) -> Tensor:
    """Resample a template-space image onto the input grid via the fitted transform.

    For each input voxel ``p`` the fit gives its template location
    ``t = vox2vox·(p + disp(p))`` (TPM voxel coords); this converts that to the
    ``source`` image's voxel frame (``inv(source_affine)·tpm_affine·t``, so ``source``
    need not share the TPM's grid, only its world space) and samples there. The result
    is ``source`` pulled into the input's space — e.g. the anat or a native tissue map
    cast into EPI space (``tpm_source_in_epi``). Handles 3-D or 4-D ``source``.

    ``use_warp=False`` drops the deformation and uses the affine alone — the "initial"
    cast, a before/after baseline for the nonlinear fit.
    """
    from .interp import trilinear_interpolate

    device = torch.device(device) if device is not None else source.device
    source = source.to(device)
    vol4d = source[None] if source.ndim == 3 else source  # (T, nz, ny, nx)
    x, y, z = _fullres_grid(out_shape, device)
    if use_warp:
        disp = full_resolution_warp(fit, out_shape, device=device)
        coords = torch.stack(
            [
                x + disp[..., 0].reshape(-1),
                y + disp[..., 1].reshape(-1),
                z + disp[..., 2].reshape(-1),
            ],
            dim=1,
        )
    else:
        coords = torch.stack([x, y, z], dim=1)
    tpm_c = apply_affine_pts(coords, fit["vox2vox"].to(device))
    tpm_to_src = torch.linalg.inv(source_affine.to(torch.float64)) @ tpm_affine.to(torch.float64)
    src_c = apply_affine_pts(tpm_c, tpm_to_src.to(device))
    xs, ys, zs = src_c[:, 0], src_c[:, 1], src_c[:, 2]
    if kernel != "linear":
        from . import interp as _interp

        sampler = {
            "cubic": _interp.cubic_resample_3d,
            "quintic": _interp.quintic_resample_3d,
            "heptic": _interp.heptic_resample_3d,
            "wsinc5": _interp.wsinc5_resample_3d,
        }[kernel]
        xs, ys, zs = xs.to(source.dtype), ys.to(source.dtype), zs.to(source.dtype)
    else:
        sampler = trilinear_interpolate
    out = torch.stack(
        [sampler(vol4d[t], xs, ys, zs).reshape(out_shape) for t in range(vol4d.shape[0])], dim=0
    )
    return out[0] if source.ndim == 3 else out


# SPM's default 6-class TPM order (TPM.nii); used to label the diagnostic plot when the
# caller doesn't supply names. Extra/fewer classes fall back to generic "class N".
_SPM_TISSUE_NAMES = ["GM", "WM", "CSF", "bone", "soft tissue", "air/background"]


def plot_intensity_fit(
    fit: dict,
    corrected: Tensor,
    posteriors: Tensor,
    *,
    tissue_names: list[str] | None = None,
    n_bins: int = 200,
    path: str | None = None,
):
    """SPM-style diagnostic: intensity histogram(s) with the fitted GMM overlaid.

    One panel per input channel (a row). Grey bars are the bias-corrected data;
    each coloured line is a tissue's fitted Gaussian-mixture **marginal** in that
    channel (``Σ_{k∈t} w_k · N(x; μ_ck, Σ_cc,k)``, weight ``w_k`` = that tissue's
    posterior mass × the within-tissue mixing proportion); the black line is the
    total model. Where the grey histogram rises above the black line the model
    under-explains that intensity — a quick read on *what got missed*.

    For multi-channel fits each Gaussian is joint over channels, so a panel shows
    only that channel's marginal (the means separate per channel, e.g. GM bright on
    T1 / dark on T2). ``ngaus`` is per **tissue**, not per channel — a single joint
    Gaussian already carries a per-channel mean, so you don't add Gaussians just
    because you added an image.

    Args:
        fit: the dict from :func:`fit_segment` (needs means/covs/mix/tissue_of).
        corrected: bias-corrected data, ``(nz,ny,nx)`` or ``(n_chan,nz,ny,nx)`` —
            i.e. ``segment_apply(...)["corrected"]``.
        posteriors: tissue posteriors ``(n_tissue,nz,ny,nx)`` (the ``cN`` maps).
        tissue_names: labels per tissue; defaults to the SPM 6-class names.
        n_bins: histogram bins.
        path: if given, save the figure there and close it; else return the Figure.

    Returns:
        the matplotlib ``Figure`` when ``path`` is None, else ``None``.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: this is a save-to-disk diagnostic
    import matplotlib.pyplot as plt
    import numpy as np

    means = fit["means"].detach().cpu().double()  # (n_chan, n_gauss)
    covs = fit["covs"].detach().cpu().double()  # (n_chan, n_chan, n_gauss)
    mix = fit["mix"].detach().cpu().double()  # (n_gauss,)
    tissue_of = fit["tissue_of"].detach().cpu()  # (n_gauss,)
    n_chan = means.shape[0]
    n_tissue = int(tissue_of.max().item()) + 1

    corr = corrected.detach().cpu().float()
    if corr.ndim == 3:
        corr = corr[None]
    post = posteriors.detach().cpu().float()

    if tissue_names is None:
        tissue_names = (
            _SPM_TISSUE_NAMES[:n_tissue]
            if n_tissue <= len(_SPM_TISSUE_NAMES)
            else [f"class {t + 1}" for t in range(n_tissue)]
        )
    colours = plt.get_cmap("tab10")(np.linspace(0, 1, 10))

    # shared foreground mask (same voxels for every channel so tissue proportions
    # are consistent): finite and non-zero in at least one channel, excluding the
    # exact-zero out-of-FoV background that would otherwise spike the low bin.
    mask = torch.isfinite(corr).all(dim=0) & (corr.abs() > 0).any(dim=0)  # (nz,ny,nx)
    flat = mask.reshape(-1)
    post_f = post.reshape(n_tissue, -1)[:, flat]  # (n_tissue, n_fg)
    tissue_mass = post_f.sum(dim=1)  # (n_tissue,)
    total_mass = tissue_mass.sum().clamp_min(1e-30)
    # per-Gaussian data weight: tissue proportion × within-tissue mixing fraction
    w = (tissue_mass[tissue_of] / total_mass) * mix  # (n_gauss,), sums ≈ 1

    fig, axes = plt.subplots(1, n_chan, figsize=(6.4 * n_chan, 4.6), squeeze=False)
    for c in range(n_chan):
        ax = axes[0, c]
        vals = corr[c].reshape(-1)[flat].numpy()
        lo, hi = np.percentile(vals, [0.2, 99.8])  # robust range, ignore outliers
        if hi <= lo:
            hi = lo + 1.0
        counts, edges = np.histogram(vals, bins=n_bins, range=(lo, hi))
        centres = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1] - edges[0]
        n_fg = int(flat.sum().item())

        ax.bar(centres, counts, width=width, color="0.8", label="data", zorder=1)
        xs = np.linspace(lo, hi, 512)
        total_curve = np.zeros_like(xs)
        for t in range(n_tissue):
            cols = (tissue_of == t).nonzero(as_tuple=True)[0]
            dens = np.zeros_like(xs)
            for k in cols.tolist():
                mu = float(means[c, k])
                var = float(covs[c, c, k].clamp_min(1e-12))
                dens += float(w[k]) * np.exp(-0.5 * (xs - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)
            curve = dens * n_fg * width  # density → expected counts per bin
            total_curve += curve
            if tissue_mass[t] / total_mass > 1e-4:  # skip tissues with ~no mass
                ax.plot(
                    xs, curve, color=colours[t % 10], lw=2.0,
                    label=f"{tissue_names[t]} ({100 * tissue_mass[t] / total_mass:.0f}%)",
                    zorder=3,
                )  # fmt: skip
        ax.plot(xs, total_curve, color="k", lw=1.4, ls="--", label="total model", zorder=4)
        ax.set_xlabel(
            f"bias-corrected intensity (channel {c + 1})"
            if n_chan > 1
            else "bias-corrected intensity"
        )
        ax.set_ylabel("voxel count")
        ax.set_xlim(lo, hi)
        # a single dominant background/air bin would flatten the tissue peaks; cap the
        # y-axis to the interesting structure (fitted peak + bulk of the histogram) and
        # let the spike run off the top.
        ymax = max(float(total_curve.max()), float(np.percentile(counts, 99))) * 1.25
        ax.set_ylim(0, ymax if ymax > 0 else None)
        ax.legend(fontsize=8, framealpha=0.9)
    fig.suptitle("Segmentation intensity fit — data vs fitted tissue mixture")
    fig.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig
