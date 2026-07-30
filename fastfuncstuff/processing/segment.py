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
    prior = prior.to(corrected.dtype)
    log_gauss = gaussian_log_density(corrected, means, covs, mix)  # (n_vox, n_gauss)
    log_dens = log_gauss + torch.log(prior[:, tissue_of].clamp_min(1e-40))
    loglik = torch.logsumexp(log_dens, dim=1)  # (n_vox,)
    resp = torch.exp(log_dens - loglik[:, None])  # (n_vox, n_gauss), rows sum to 1
    return resp, loglik


def gaussian_log_density(corrected: Tensor, means: Tensor, covs: Tensor, mix: Tensor) -> Tensor:
    """``log(mix_k) + log N(y; mn_k, vr_k)`` per Gaussian — ``(n_vox, n_gauss)``.

    The intensity-only part of the E-step (no spatial prior). Factored out because it is
    **constant while only the warp changes**, so the deformation step evaluates the
    Gaussians once per EM iteration (via :func:`gmm_tissue_likelihood`) instead of on every
    warp sub-iteration / line-search step.

    Per-voxel work runs in the caller's dtype (float32 fit hot path → fast + cheap autograd
    on consumer GPUs, where float64 is ~1/32 rate). Only the tiny per-Gaussian covariance
    factorisation is promoted to float64 for conditioning ([[Float32 vs float64]]).
    """
    wdt = corrected.dtype
    n_chan = corrected.shape[1]
    # log N(y; mean, cov) = -n_chan/2·log2π - 1/2·log|cov| - 1/2·(y-mean)' cov⁻¹ (y-mean).
    # Batch the tiny per-Gaussian factorisations (covs are n_chan×n_chan, n_chan small)
    # into ONE call each — a Python loop of 1×1 Cholesky/triangular-solves was pure
    # LAPACK-dispatch overhead (dominant on CPU, launch-bound on GPU).
    covs_b = covs.to(torch.float64).permute(2, 0, 1)  # (n_gauss, n_chan, n_chan)
    chol = torch.linalg.cholesky(covs_b)  # M-step keeps covs PD
    logdet = (2.0 * torch.log(torch.diagonal(chol, dim1=1, dim2=2)).sum(dim=1)).to(wdt)
    prec = torch.cholesky_inverse(chol).to(wdt)  # (n_gauss, n_chan, n_chan) = cov⁻¹
    centred = corrected[:, :, None] - means.to(wdt)[None, :, :]  # (n_vox, n_chan, n_gauss)
    tmp = torch.einsum("vck,kcd->vdk", centred, prec)
    maha = (tmp * centred).sum(dim=1)  # (n_vox, n_gauss)
    return torch.log(mix.to(wdt).clamp_min(1e-40)) - 0.5 * (n_chan * _LOG_2PI + logdet + maha)


def gmm_tissue_likelihood(
    corrected: Tensor, means: Tensor, covs: Tensor, mix: Tensor, tissue_of: Tensor, n_tissue: int
) -> Tensor:
    """Per-**tissue** relative likelihood ``Σ_{k∈t} mix_k·N(y; mn_k, vr_k)`` — ``(n_vox,
    n_tissue)``, Gaussians collapsed, spatial prior excluded (SPM ``buf.dat``).

    Row-scaled by ``exp(−rowmax)`` for numerical safety; the dropped ``rowmax`` is a
    per-voxel constant that shifts the warp log-likelihood by a fixed offset (irrelevant to
    the gradient and to line-search *comparisons*). The deformation objective is then just
    ``Σ log(Σ_t lik_t · prior_t(φ))`` — one cheap ``grid_sample`` + dot per warp step, with
    the Cholesky/Mahalanobis paid once per EM iteration rather than once per sub-iteration.
    """
    qt = gaussian_log_density(corrected, means, covs, mix)  # (n_vox, n_gauss)
    e = torch.exp(qt - qt.max(dim=1, keepdim=True).values)
    lik = torch.zeros(corrected.shape[0], n_tissue, dtype=e.dtype, device=e.device)
    lik.index_add_(1, tissue_of, e)
    return lik


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
    path: str,
    *,
    add_background: str = "auto",
    background_threshold: float = 0.01,
    device: torch.device | str = "cpu",
    verbose: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor, bool]:
    """Load a 4-D tissue-probability template (``spm_load_priors8`` equivalent).

    The number of tissue classes ``n_tissue`` is taken from the 4th dimension — 6 for
    SPM's default ``TPM.nii`` but whatever the file provides (user TPMs with any class
    count work). Each class is clamped to ``[0,1]``, voxels whose classes sum to more
    than 1 are rescaled to sum to 1 (``spm_load_priors8``'s ``t = s>1`` pass), and the
    log is stored (with a ``tiny=1e-4`` floor) so the prior is sampled in log-space then
    exponentiated.

    **Auto background class.** A generative mixture needs the classes to explain *every*
    voxel: SPM's ``TPM.nii`` does (its six classes sum to 1.0 everywhere — measured, not
    assumed, because its 6th class is air/background). A limited-coverage template — say
    GM/WM/CSF only — does not, and the shortfall is exactly the head-and-air the model
    would otherwise be forced to shoehorn into a tissue class. ``add_background``:

    - ``"auto"`` (default) — append ``1 - Σ_k p_k`` as an extra final class when the mean
      shortfall exceeds ``background_threshold``. No-op for a complete template.
    - ``"yes"`` / ``"no"`` — force.

    The synthesised class is last, so existing class indices (``c1`` = GM, …) are
    unchanged and it is written as one more ``cN``. Give it several Gaussians via
    ``-ngaus``: it is a genuinely multi-modal "everything else" (air, skull, scalp, fat,
    neck), not one population.

    Returns:
        log_prior: ``(n_tissue, nz, ny, nx)`` = ``log(p + tiny)``.
        tpm_affine: ``(4, 4)`` nibabel voxel→world affine.
        bg_low: ``(n_tissue,)`` mean of the first z-plane (below-FoV background prob).
        bg_high: ``(n_tissue,)`` mean of the last z-plane (above-FoV background prob).
        added_background: whether a background class was appended.
    """
    from .io import load_image  # local import: keeps the numeric core import-light

    data, hdr = load_image(path, device=device)  # (n_tissue, nz, ny, nx) or (nz,ny,nx)
    if data.ndim == 3:
        data = data[None]
    tiny = 1e-4
    prob = data.clamp(0.0, 1.0)
    total = prob.sum(dim=0, keepdim=True)
    prob = torch.where(total > 1.0, prob / total.clamp_min(1e-20), prob)

    shortfall = (1.0 - prob.sum(dim=0, keepdim=True)).clamp_min(0.0)
    mean_short = float(shortfall.mean())
    if add_background == "auto":
        added = mean_short > background_threshold
    else:
        added = add_background == "yes"
    if added:
        prob = torch.cat([prob, shortfall], dim=0)
        if verbose:
            print(
                f"  TPM classes explain {1.0 - mean_short:.1%} of the probability mass — "
                f"appended a background class (c{prob.shape[0]}) holding the remainder"
            )

    # bg1/bg2 are taken AFTER the renormalisation, as in spm_load_priors8
    bg_low = prob[:, 0, :, :].mean(dim=(1, 2))  # first z-plane
    bg_high = prob[:, -1, :, :].mean(dim=(1, 2))  # last z-plane
    log_prior = torch.log(prob + tiny)
    affine = torch.as_tensor(hdr["affine"], dtype=torch.float64, device=data.device)
    return log_prior, affine, bg_low, bg_high, added


def bspline2_interpolate_multi(volume: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    """Degree-2 B-spline sampling of a ``(C, nz, ny, nx)`` volume — SPM's TPM kernel.

    ``spm_load_priors8`` stores the TPM as ``spm_bsplinc(log(p+tiny), [1 1 1 ...])``, and
    **degree-1 ``bsplinc`` is a no-op** (the degree-1 B-spline coefficients *are* the
    data). ``spm_sample_priors8`` then samples those raw values with ``tpm.deg = 2``. So
    SPM evaluates a degree-2 B-spline basis against unprefiltered data: an
    **approximating** (mildly low-pass) kernel, not an interpolating one. That is a
    deliberate part of the model — the prior it warps is a slightly blurred log-TPM, and
    the smooth analytic derivatives are what make its Gauss-Newton warp well behaved.
    Trilinear (interpolating, ``C⁰``, piecewise-constant gradient) is a different prior.

    Weights follow ``bsplines.c``: ``i = floor(t - 0.5)`` with ``w[k] = wt2(t - i - k)``,
    i.e. taps at ``c-1, c, c+1`` for ``c = floor(t + 0.5)`` and ``d = t - c ∈ [-0.5, 0.5)``:

        w₋₁ = ½(½ - d)²,   w₀ = ¾ - d²,   w₊₁ = ½(½ + d)²      (sum to 1)

    ``C¹`` continuous across the tap switch, so it is differentiable — the gradient flows
    through the weights (the integer taps are piecewise constant, as they should be).
    Out-of-range taps clamp to the border, matching the ``padding_mode="border"``
    behaviour of the trilinear path so the two share the same background handling.

    The 27 taps are gathered in **one** indexed read rather than 27, and the separable
    weights are outer-producted by broadcasting: a per-tap Python loop cost ~110 kernel
    launches per call in a loop that is already launch-bound. The ``(n_chan, 27, m)``
    gather is the only large transient, so ``m`` is sub-chunked to keep it bounded.

    Args:
        volume: ``(C, nz, ny, nx)`` float tensor.
        x, y, z: ``(N,)`` sample locations in index coordinates.

    Returns:
        ``(N, C)`` sampled values.
    """
    n_chan, nz, ny, nx = volume.shape
    dt = volume.dtype
    flat = volume.reshape(n_chan, -1)
    n = x.shape[0]

    def taps(t: Tensor, size: int) -> tuple[Tensor, Tensor]:
        """``(3, m)`` clamped tap indices and their weights, centred on ``floor(t+½)``."""
        c = torch.floor(t + 0.5)
        d = t - c
        w = torch.stack([0.5 * (0.5 - d) ** 2, 0.75 - d * d, 0.5 * (0.5 + d) ** 2])
        offs = torch.tensor([-1.0, 0.0, 1.0], dtype=c.dtype, device=c.device)[:, None]
        return (c[None] + offs).clamp(0, size - 1).long(), w

    # bound the (n_chan, 27, m) gather; one pass for any ordinary sample count
    per_sample = n_chan * 27 * volume.element_size()
    step = max(1, min(n, (256 * 1024 * 1024) // max(per_sample, 1)))
    out = torch.empty((n, n_chan), dtype=dt, device=volume.device)
    for s in range(0, n, step):
        sl = slice(s, min(s + step, n))
        ix, wx = taps(x[sl].to(dt), nx)
        iy, wy = taps(y[sl].to(dt), ny)
        iz, wz = taps(z[sl].to(dt), nz)
        # separable outer products over the 3x3x3 stencil → (27, m)
        idx = ((iz[:, None, None] * ny + iy[None, :, None]) * nx + ix[None, None, :]).reshape(
            27, -1
        )
        w = (wz[:, None, None] * wy[None, :, None] * wx[None, None, :]).reshape(27, -1)
        vals = flat[:, idx.reshape(-1)].reshape(n_chan, 27, -1)
        out[sl] = torch.einsum("ckm,km->mc", vals, w)
    return out


def bspline3_coefficients(volume: Tensor) -> Tensor:
    """``spm_bsplinc(vol, [3 3 3 0 0 0])`` — cubic B-spline coefficients, mirror boundary.

    Unlike the degree-2 case (where ``spm_bsplinc`` is a no-op and SPM deliberately samples
    raw values), degree 3 runs a real prefilter, so the result is an **interpolating**
    cubic spline: it passes through the nodes and is ``C²`` between them. SPM uses this —
    not trilinear — to expand the fitted deformation from the ``samp`` grid to full
    resolution (``spm_preproc_write8.m:133-136`` + ``defs``).

    Thevenaz/Unser recursive filter with the single pole ``z = √3 - 2`` and gain 6, applied
    along each of the **last three** axes (so a ``(nz, ny, nx)`` volume or a channel-first
    ``(C, nz, ny, nx)`` stack both work). The recursions are sequential in the filtered axis
    but fully batched across the rest, and this runs once per apply on the small ``samp``
    grid, so the loop cost is irrelevant.
    """
    import math

    z = math.sqrt(3.0) - 2.0
    c = volume.to(torch.float64) * (6.0**3)  # gain λ = (1-z)(1-1/z) = 6 per axis
    for axis in (-3, -2, -1):
        n = c.shape[axis]
        if n < 2:
            continue
        c = c.movedim(axis, 0).contiguous()
        # causal initialisation for a mirrored signal (Thevenaz InitialCausalCoefficient)
        zn, iz, z2n = z, 1.0 / z, z ** (n - 1)
        acc = c[0] + z2n * c[n - 1]
        z2n = z2n * z2n
        for k in range(1, n - 1):
            z2n = z2n * iz
            acc = acc + (zn + z2n) * c[k]
            zn = zn * z
        rows = [acc / (1.0 - zn * zn)]
        for k in range(1, n):  # causal recursion
            rows.append(c[k] + z * rows[k - 1])
        # anticausal initialisation + recursion
        out = [torch.empty(0)] * n
        out[n - 1] = (z / (z * z - 1.0)) * (z * rows[n - 2] + rows[n - 1])
        for k in range(n - 2, -1, -1):
            out[k] = z * (out[k + 1] - rows[k])
        c = torch.stack(out, dim=0).movedim(0, axis).contiguous()
    return c.to(volume.dtype)


def _mirror_index(i: Tensor, size: int) -> Tensor:
    """Whole-sample mirror indexing (period ``2N-2``) — the boundary
    :func:`bspline3_coefficients` assumes, so evaluation must match it."""
    if size == 1:
        return torch.zeros_like(i)
    period = 2 * size - 2
    i = i.abs().remainder(period)
    return torch.where(i >= size, period - i, i)


def _bspline_taps(t: Tensor, size: int, degree: int) -> tuple[Tensor, Tensor]:
    """``(deg+1, m)`` tap indices and B-spline weights at continuous ``t``.

    Degree 2 centres on ``floor(t+½)`` with taps ``{-1,0,1}``; degree 3 centres on
    ``floor(t)`` with taps ``{-1,0,1,2}``. Both partition unity.

    Boundaries differ by design: degree 2 (the TPM prior) **clamps** to the border, so it
    matches ``grid_sample(padding_mode="border")`` and the one-voxel background ramp in
    :func:`sample_tpm_prior`; degree 3 (the deformation) **mirrors**, because that is the
    boundary condition its prefilter was built with, and clamping instead leaves a real
    error in the outermost planes.
    """
    if degree == 2:
        c = torch.floor(t + 0.5)
        d = t - c
        w = torch.stack([0.5 * (0.5 - d) ** 2, 0.75 - d * d, 0.5 * (0.5 + d) ** 2])
        offs = (-1.0, 0.0, 1.0)
    else:  # degree 3
        c = torch.floor(t)
        d = t - c
        e = 1.0 - d
        # β³ basis: w₋₁ = e³/6, w₀ = ⅔ - d² + d³/2, w₊₁ = ⅔ - e² + e³/2, w₊₂ = d³/6
        w = torch.stack(
            [
                (1.0 / 6.0) * e**3,
                (2.0 / 3.0) - d * d + 0.5 * d**3,
                (2.0 / 3.0) - e * e + 0.5 * e**3,
                (1.0 / 6.0) * d**3,
            ]
        )
        offs = (-1.0, 0.0, 1.0, 2.0)
    off_t = torch.tensor(offs, dtype=c.dtype, device=c.device)[:, None]
    idx = (c[None] + off_t).long()
    return (idx.clamp(0, size - 1) if degree == 2 else _mirror_index(idx, size)), w


def bspline_interpolate_multi(
    volume: Tensor, x: Tensor, y: Tensor, z: Tensor, *, degree: int = 2
) -> Tensor:
    """Separable B-spline sampling of a ``(C, nz, ny, nx)`` volume — see
    :func:`bspline2_interpolate_multi` for the degree-2 (TPM) case and
    :func:`bspline3_coefficients` for the degree-3 (deformation) case, which expects
    **coefficients**, not raw samples, as ``volume``.

    All ``(deg+1)³`` taps go out in one indexed read with broadcast outer-product weights,
    sub-chunked so the ``(C, taps, m)`` transient stays bounded.
    """
    n_chan, nz, ny, nx = volume.shape
    dt = volume.dtype
    flat = volume.reshape(n_chan, -1)
    n = x.shape[0]
    ntap = (degree + 1) ** 3

    per_sample = n_chan * ntap * volume.element_size()
    step = max(1, min(n, (256 * 1024 * 1024) // max(per_sample, 1)))
    out = torch.empty((n, n_chan), dtype=dt, device=volume.device)
    for s in range(0, n, step):
        sl = slice(s, min(s + step, n))
        ix, wx = _bspline_taps(x[sl].to(dt), nx, degree)
        iy, wy = _bspline_taps(y[sl].to(dt), ny, degree)
        iz, wz = _bspline_taps(z[sl].to(dt), nz, degree)
        idx = ((iz[:, None, None] * ny + iy[None, :, None]) * nx + ix[None, None, :]).reshape(
            ntap, -1
        )
        w = (wz[:, None, None] * wy[None, :, None] * wx[None, None, :]).reshape(ntap, -1)
        vals = flat[:, idx.reshape(-1)].reshape(n_chan, ntap, -1)
        out[sl] = torch.einsum("ckm,km->mc", vals, w)
    return out


# Kernels that are differentiable w.r.t. the sample coordinates, so usable inside the
# warp fit; anything else is an output-pass-only resampler.
_DIFFERENTIABLE_KERNELS = ("linear", "bspline2")


def _image_resample(volume: Tensor, x: Tensor, y: Tensor, z: Tensor, kernel: str) -> Tensor:
    """Resample an *image* (not the TPM) at ``(x, y, z)`` with a named kernel.

    Shares the ``-prior_interp`` name space so one flag covers the whole output pass, but
    ``"bspline2"`` is a TPM kernel (a deliberately approximating, low-pass sampler for a
    low-resolution probability map) and is the wrong choice for anatomy — it would soften
    the image. Map it to the nearest sharp image kernel instead.
    """
    from . import interp as _interp

    if kernel == "linear":
        return _interp.trilinear_interpolate(volume, x, y, z)
    sampler = {
        "bspline2": _interp.cubic_resample_3d,
        "cubic": _interp.cubic_resample_3d,
        "quintic": _interp.quintic_resample_3d,
        "heptic": _interp.heptic_resample_3d,
        "wsinc5": _interp.wsinc5_resample_3d,
    }[kernel]
    dt = volume.dtype
    return sampler(volume, x.to(dt), y.to(dt), z.to(dt))


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

    ``kernel`` picks the interpolation. ``"bspline2"`` is SPM's own
    (:func:`bspline2_interpolate_multi`) and the faithful choice for the fit;
    ``"linear"`` is trilinear (cheaper — one ``grid_sample`` — but a different, sharper
    prior). Both are differentiable, so either can drive the warp. The higher-order
    Lagrange kernels (``"cubic"``/``"quintic"``/``"heptic"``/``"wsinc5"``) are
    output-pass only: the low-resolution TPM sampled trilinearly imprints its own grid on
    the output (blocky priors → blocky posteriors) and a smooth kernel removes that.

    Args:
        log_prior: ``(n_tissue, nz, ny, nx)`` from :func:`load_tpm`.
        coords: ``(n_vox, 3)`` TPM voxel coords ``(x, y, z)``.
        bg_low, bg_high: ``(n_tissue,)`` background probabilities.
        kernel: interpolation kernel (see above).

    Returns:
        ``(n_vox, n_tissue)`` tissue prior, rows summing to 1.
    """
    from . import interp as _interp

    if kernel not in _DIFFERENTIABLE_KERNELS:
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

    if kernel in _DIFFERENTIABLE_KERNELS:
        # all tissues sampled together (shared sample locations) — the hot path. Both
        # kernels clamp out-of-range taps to the border, so `sampled` is continuous as
        # coords exit the volume (edge value extends outward); the background substitution
        # below is ramped over one voxel so the whole prior stays continuous &
        # differentiable — essential for the Gauss-Newton warp line search, which globally
        # rejects any step that raises the objective. A hard in/out switch cliffs the
        # objective at voxels sitting exactly on the boundary and stalls the solver.
        interp_multi = (
            _interp.trilinear_interpolate_multi
            if kernel == "linear"
            else bspline2_interpolate_multi
        )
        sampled = torch.exp(interp_multi(log_prior, x, y, z).to(out_dtype))
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


def tpm_coverage_mask(
    tpm_coords: Tensor,
    tpm_shape: tuple[int, int, int],
    tpm_vox: tuple[float, float, float],
    *,
    bottom_mm: float = 5.0,
    fov_mm: float | None = None,
) -> Tensor:
    """``True`` where a template coordinate lies inside the TPM's usable field of view.

    Args:
        tpm_coords: ``(n, 3)`` TPM voxel coords ``(x, y, z)``, 0-based.
        tpm_shape: ``(nz, ny, nx)`` of the template.
        tpm_vox: template voxel sizes ``(vx, vy, vz)`` in mm.
        bottom_mm: SPM's rule (``spm_preproc8.m:233-237``) — reject anything within this
            many mm of the **bottom** of the template. On a whole-head T1 that is the neck
            and shoulders. ``0`` disables.
        fov_mm: **beyond SPM** — reject anything more than this many mm outside the
            template box on *any* axis, not just below it. ``None`` disables.

    The two rules answer different questions. SPM only guards the inferior direction,
    because it assumes a whole-head template that the subject sits inside; everything
    outside still gets the out-of-FoV background prior (``bg_low``/``bg_high``) and is
    classified anyway. That is fine for the fit and wrong for the *output*: a voxel in the
    neck maps far below the template, picks up whatever the template's edge planes happen
    to contain, and can come back labelled as brain tissue. When the subject→template
    affine is supplied (as it always is here), "outside the template" is a known fact
    about the geometry, not something to infer from intensity — so `fov_mm` lets the caller
    say so. The reverse case already works: a limited-FoV EPI input with a whole-head
    template simply produces maps over the EPI's own extent.
    """
    nz, ny, nx = tpm_shape
    x, y, z = tpm_coords[:, 0], tpm_coords[:, 1], tpm_coords[:, 2]
    keep = torch.ones(tpm_coords.shape[0], dtype=torch.bool, device=tpm_coords.device)
    if bottom_mm > 0:
        # SPM tests in 1-based template indices; ours are 0-based, hence the -1
        keep &= z > (bottom_mm / tpm_vox[2] - 1.0)
    if fov_mm is not None:
        mx, my, mz = (fov_mm / v for v in tpm_vox)
        keep &= (x >= -mx) & (x <= nx - 1 + mx)
        keep &= (y >= -my) & (y <= ny - 1 + my)
        keep &= (z >= -mz) & (z <= nz - 1 + mz)
    return keep


def update_tissue_weights(
    tissue_mass: Tensor, expected_mass: Tensor, wp_reg: float, n_tissue: int
) -> Tensor:
    """SPM's self-correcting tissue-weight (``wp``) update, normalised.

    ``wp_k = (observed_k + wp_reg) / (expected_k + wp_reg·Kb)`` then normalised (SPM
    ``spm_preproc8`` eq. after 27). ``observed_k`` is the soft tissue mass; ``expected_k``
    is ``mgm_k = Σ_v B(v,k)/(B(v,:)·wp)`` over the **unweighted** warped TPM ``B``. Because
    ``expected_k`` grows with ``wp_k``, the ratio self-corrects — unlike an ``observed/
    total-observed`` update, which positively feeds back and inflates one tissue past its
    boundary over many iterations ("growing brains"). ``wp_reg`` (SPM default 100) biases
    the result toward uniform ``1/Kb``: larger ``wp_reg`` → closer to uniform.
    """
    wp = (tissue_mass + wp_reg) / (expected_mass + wp_reg * n_tissue)
    return wp / wp.sum()


def weight_prior(prior: Tensor, wp: Tensor) -> Tensor:
    """Apply per-tissue weights ``wp`` and renormalise (SPM ``log_spatial_priors``).

    The tissue weights let the model deviate from the raw TPM mixing proportions
    (e.g. a subject with more CSF than the template) — ``prior_w ∝ wp·prior``.
    """
    weighted = prior * wp.to(prior.dtype)
    return weighted / weighted.sum(dim=1, keepdim=True).clamp_min(1e-40)


# ---------------------------------------------------------------------------
# Deformation regulariser — SPM's `spm_diffeo('vel2mom')`, transcribed exactly.
# ---------------------------------------------------------------------------
#
# The operator is a fixed 5x5x5 symmetric stencil on the dense field, built from
# `reg = (abs, membrane, bending, mu, lambda)` and `param(1:3)`, the **node spacing**
# of the warp lattice (SPM's `sk.*vx`). Reference: `src/shoot_regularisers.c:vel2mom`.
#
# Two things here are easy to get wrong and were wrong until 2026-07-29:
#
# 1. **`v_i = param_i²`, in the NUMERATOR.** The old implementation evaluated the
#    energy with finite differences divided by the node spacing (`v_i = 1/h²`). SPM
#    squares `param(1:3)` and uses it as a multiplier. At `samp 3` on 1 mm data that is
#    an 81x error per `v`, and bending goes as `v²` — so the deformation prior was
#    ~729x too weak once the per-component `1/v_c` below is accounted for. A grossly
#    under-penalised bending term does NOT give a bigger warp: it gives a *spiky* one,
#    because each node then follows only its own data gradient instead of the smooth
#    prior propagating displacement out of the high-gradient regions. Measured against
#    SPM's own `Twarp` on a reference subject: 41x its bending energy at 0.57x its mean
#    displacement, which inflated CSF (thin sheet, so most sensitive) by 41%.
#
# 2. **`vel2mom` always uses the linear-elastic branch.** Only the `kernel()` helper
#    has a `mu==0 && lam==0` shortcut; `vel2mom` itself unconditionally divides the
#    absolute/membrane/bending part of component `c` by `v_c` (`:743`, `:757`, `:771`).
#    So the isotropic part scales as `v²/v = v`, not `v²`.
#
# The stencil is written in SPM's "difference form" — every term is `w·(neighbour −
# centre)` rather than a plain weighted sum. SPM does that for rounding accuracy; it is
# also what makes the Neumann boundary free, since a mirrored neighbour equals the
# centre at an edge and the term vanishes. Note this means SPM's `w?000` centre weights
# are never used by `vel2mom` (only by the relaxation and kernel helpers), which is why
# they do not appear below — but they *are* the diagonal, so
# :func:`warp_prior_diagonal` reconstructs them for the preconditioner.


def _pad_sym2(x: Tensor, dim: int) -> Tensor:
    """Pad ``dim`` by 2 each side with SPM's Neumann (half-sample mirror) extension.

    ``shoot_boundary.c:neumann_boundary`` maps ``-1→0, -2→1`` and ``n→n-1, n+1→n-2``:
    a mirror that **repeats** the edge sample (numpy's ``symmetric``, not ``reflect``).
    Torch has no such pad mode, so build it by flipping the two outermost planes.
    """
    n = x.shape[dim]
    if n == 1:  # every shift lands back on the single plane
        one = x.narrow(dim, 0, 1)
        return torch.cat([one, one, x, one, one], dim=dim)
    if n == 2:
        lo = x.narrow(dim, 0, 2).flip(dim)  # [x1, x0] ↔ indices -2, -1
        hi = x.narrow(dim, 0, 2).flip(dim)  # [x1, x0] ↔ indices n, n+1
        return torch.cat([lo, x, hi], dim=dim)
    lo = x.narrow(dim, 0, 2).flip(dim)  # [x1, x0]
    hi = x.narrow(dim, n - 2, 2).flip(dim)  # [x_{n-1}, x_{n-2}]
    return torch.cat([lo, x, hi], dim=dim)


# Displacement component index → the field axis it is indexed along.
# `twarp` is (gz, gy, gx, 3) with components (x, y, z), so SPM's i/j/k (x/y/z) offsets
# move along array axes 2/1/0 respectively.
_SPM_AXIS = (3, 2, 1)  # for a (3, gz, gy, gx) stack: x→axis 3, y→axis 2, z→axis 1


def spm_vel2mom_weights(
    reg: tuple[float, ...], param: tuple[float, float, float], dims: tuple[int, int, int]
) -> dict[str, float]:
    """Scalar stencil weights of ``spm_diffeo('vel2mom')`` — ``shoot_regularisers.c:601``.

    Args:
        reg: ``(absolute, membrane, bending, mu, lambda)``; a 3-tuple is padded with zeros.
        param: SPM's ``param(1:3)`` = **node spacing** of the warp lattice (``sk.*vx``, mm).
        dims: lattice shape as SPM sees it, ``(nx, ny, nz)`` — only used for the
            degenerate-dimension corrections (``:631-670``).

    Returns a dict of the weights consumed by :func:`warp_prior_grad`.
    """
    lam0, lam1, lam2, mu, lam = (tuple(reg) + (0.0,) * (5 - len(reg)))[:5]
    v0, v1, v2 = (float(p) * float(p) for p in param)
    tot = v0 + v1 + v2
    w = {
        "w100": lam2 * (-4.0 * v0 * tot) - lam1 * v0,
        "w010": lam2 * (-4.0 * v1 * tot) - lam1 * v1,
        "w001": lam2 * (-4.0 * v2 * tot) - lam1 * v2,
        "w200": lam2 * v0 * v0,
        "w020": lam2 * v1 * v1,
        "w002": lam2 * v2 * v2,
        "w110": lam2 * 2.0 * v0 * v1,
        "w101": lam2 * 2.0 * v0 * v2,
        "w011": lam2 * 2.0 * v1 * v2,
    }
    nx, ny, nz = dims
    # A lattice 2 wide cannot carry a ±2 tap: the mirror folds it onto a *different*
    # sample, so SPM drops the term rather than double-counting it (:631-650).
    if nx <= 2:
        w["w200"] = 0.0
    if ny <= 2:
        w["w020"] = 0.0
    if nz <= 2:
        w["w002"] = 0.0
    if ny == 1 and nz == 1:
        w["w011"] = 0.0
    # per-component first-neighbour weights: the elastic part, plus the isotropic part
    # divided by this component's v (SPM's wx100 = -2mu - lam + w100/v0, etc.)
    w["wx100"] = -2.0 * mu - lam + w["w100"] / v0
    w["wx010"] = -mu * v1 / v0 + w["w010"] / v0
    w["wx001"] = -mu * v2 / v0 + w["w001"] / v0
    w["wy100"] = -mu * v0 / v1 + w["w100"] / v1
    w["wy010"] = -2.0 * mu - lam + w["w010"] / v1
    w["wy001"] = -mu * v2 / v1 + w["w001"] / v1
    w["wz100"] = -mu * v0 / v2 + w["w100"] / v2
    w["wz010"] = -mu * v1 / v2 + w["w010"] / v2
    w["wz001"] = -2.0 * mu - lam + w["w001"] / v2
    if ny == 1:
        w["wx010"] = w["wy010"] = w["wz010"] = 0.0
    if nz == 1:
        w["wx001"] = w["wy001"] = w["wz001"] = 0.0
    w["w2"] = 0.25 * mu + 0.25 * lam  # off-diagonal (shear/divergence) coupling
    w["lam0"] = lam0
    w["v0"], w["v1"], w["v2"] = v0, v1, v2
    return w


def _all_zero(reg: tuple[float, ...]) -> bool:
    return not any(float(r) != 0.0 for r in reg)


def warp_prior_grad(
    field: Tensor, reg: tuple[float, ...], vox: tuple[float, float, float]
) -> Tensor:
    """``L·field`` — SPM's ``spm_diffeo('vel2mom')`` on a dense field ``(gz, gy, gx, 3)``.

    ``vox`` is SPM's ``param(1:3)``: the **node spacing** of the warp lattice (``sk·vx``
    mm), and ``field`` must be in **node units** (SPM regularises ``Twarp./sk``) — see
    :func:`fit_segment`, which divides by ``sk`` before calling.

    Transcribed term-for-term from ``shoot_regularisers.c:vel2mom`` (see the module
    comment above for the two scaling subtleties). Every neighbour enters as
    ``w·(neighbour − centre)``, so the Neumann boundary needs no special case. Pure
    slicing and arithmetic, so it is differentiable and ``warp_penalty``'s autograd
    gradient is exactly this operator.
    """
    if _all_zero(reg):
        return torch.zeros_like(field)
    gz, gy, gx = field.shape[:3]
    w = spm_vel2mom_weights(reg, vox, (gx, gy, gz))
    u = field.permute(3, 0, 1, 2)  # (3, gz, gy, gx)
    # pad all three spatial axes by 2 once; every tap is then a plain (view) slice
    p = u
    for dim in (1, 2, 3):
        p = _pad_sym2(p, dim)

    def tap(c: int, dx: int, dy: int, dz: int) -> Tensor:
        """Component ``c`` shifted by ``(dx, dy, dz)`` in SPM's (i, j, k) = (x, y, z)."""
        return p[c].narrow(0, 2 + dz, gz).narrow(1, 2 + dy, gy).narrow(2, 2 + dx, gx)

    # The shear/divergence coupling below reads a MIXED-derivative stencil, whose kernel is
    # even in the joint offset (k(-d) = k(d)) but not in each axis separately
    # (k(-1,-1) = -k(+1,-1)). A per-axis symmetric extension therefore makes it
    # non-self-adjoint at the lattice boundary — verified: exactly symmetric for interior
    # support, off by ~1% for a field that touches the edge. SPM has the same property
    # (same `bound()` on those taps) and gets away with it because `fmg` multigrid
    # tolerates mild asymmetry; conjugate gradient does not — it silently converges to the
    # wrong thing. So the cross term alone uses ZERO extension, which is a plain truncated
    # convolution and exactly symmetric for an even kernel. Identical to SPM in the
    # interior, and the warp lattice's outer plane sits well outside the head anyway.
    pz_ = u
    for dim in (1, 2, 3):
        pz_ = torch.cat(
            [torch.zeros_like(pz_.narrow(dim, 0, 1)), pz_, torch.zeros_like(pz_.narrow(dim, 0, 1))],
            dim=dim,
        )

    def tap_zero(c: int, dx: int, dy: int, dz: int) -> Tensor:
        """As :func:`tap` but zero outside the lattice (offsets limited to ±1)."""
        return pz_[c].narrow(0, 1 + dz, gz).narrow(1, 1 + dy, gy).narrow(2, 1 + dx, gx)

    out = []
    for c, (k1, k2, k3, vc) in enumerate(
        (
            ("wx100", "wx010", "wx001", "v0"),
            ("wy100", "wy010", "wy001", "v1"),
            ("wz100", "wz010", "wz001", "v2"),
        )
    ):
        ctr = u[c]

        def d(dx: int, dy: int, dz: int, _c: int = c, _ctr: Tensor = ctr) -> Tensor:
            return tap(_c, dx, dy, dz) - _ctr

        g = w[k1] * (d(-1, 0, 0) + d(1, 0, 0))
        g = g + w[k2] * (d(0, -1, 0) + d(0, 1, 0))
        g = g + w[k3] * (d(0, 0, -1) + d(0, 0, 1))
        # isotropic (absolute + membrane + bending) part, divided by this component's v
        iso = w["lam0"] * ctr
        if w["w110"]:
            iso = iso + w["w110"] * (d(-1, -1, 0) + d(1, -1, 0) + d(-1, 1, 0) + d(1, 1, 0))
        if w["w101"]:
            iso = iso + w["w101"] * (d(-1, 0, -1) + d(1, 0, -1) + d(-1, 0, 1) + d(1, 0, 1))
        if w["w011"]:
            iso = iso + w["w011"] * (d(0, -1, -1) + d(0, 1, -1) + d(0, -1, 1) + d(0, 1, 1))
        if w["w200"]:
            iso = iso + w["w200"] * (d(-2, 0, 0) + d(2, 0, 0))
        if w["w020"]:
            iso = iso + w["w020"] * (d(0, -2, 0) + d(0, 2, 0))
        if w["w002"]:
            iso = iso + w["w002"] * (d(0, 0, -2) + d(0, 0, 2))
        g = g + iso / w[vc]
        # off-diagonal shear/divergence coupling to the other two components. SPM's
        # sign pattern (`:727`): +(+1,-1) -(+1,+1) +(-1,+1) -(-1,-1) in the two axes
        # spanned by this component and the partner's.
        if w["w2"]:
            cross = field.new_zeros(())
            for other in (0, 1, 2):
                if other == c:
                    continue

                def t(sc: int, so: int, _o: int = other, _c: int = c) -> Tensor:
                    """Partner component ``_o``, offset ``sc`` along this component's own
                    axis and ``so`` along the partner's."""
                    off = [0, 0, 0]
                    off[_c] = sc
                    off[_o] = so
                    return tap_zero(_o, off[0], off[1], off[2])

                cross = cross + (t(1, -1) - t(1, 1) + t(-1, 1) - t(-1, -1))
            g = g + w["w2"] * cross
        out.append(g)
    return torch.stack(out, dim=-1)


def warp_penalty(field: Tensor, reg: tuple[float, ...], vox: tuple[float, float, float]) -> Tensor:
    """Deformation prior energy ``½·uᵀLu`` — SPM's ``llr = -0.5·Σ u·vel2mom(u)``.

    ``reg = (absolute, membrane, bending, mu, lambda)`` (default ``[0 0 0.1 0.01 0.04]``);
    a 3-tuple is accepted (``mu=lambda=0``). ``vox`` is the lattice **node spacing**
    (``sk·vx``) and ``field`` is in node units — see :func:`warp_prior_grad`, which is
    both the operator and (because ``L`` is symmetric) this function's exact gradient.

    Defined *through* the operator rather than as an independently-written energy, so the
    two can no longer disagree — they did before 2026-07-29, when the energy was a
    hand-rolled sum of squared finite differences and the operator its analytic adjoint,
    both self-consistent but neither matching ``spm_diffeo``.
    """
    if _all_zero(reg):
        return field.new_zeros(())
    return 0.5 * (field * warp_prior_grad(field, reg, vox)).sum()


def warp_prior_diagonal(
    reg: tuple[float, ...], vox: tuple[float, float, float], dims: tuple[int, int, int]
) -> tuple[float, float, float]:
    """Per-component diagonal of ``L`` — SPM's ``wx000/wy000/wz000``.

    In the difference form each ``w·(neighbour − centre)`` tap contributes ``−w`` to the
    centre, so the diagonal is recoverable from the same weights; the identity
    ``diag_x == 2mu(2v0+v1+v2)/v0 + 2lam + w000/v0`` is pinned by a unit test. ``dims``
    is ``(nx, ny, nz)``.
    """
    w = spm_vel2mom_weights(reg, vox, dims)
    diag = []
    for k1, k2, k3, vc in (
        ("wx100", "wx010", "wx001", "v0"),
        ("wy100", "wy010", "wy001", "v1"),
        ("wz100", "wz010", "wz001", "v2"),
    ):
        iso = (
            w["lam0"]
            - 4.0 * (w["w110"] + w["w101"] + w["w011"])
            - 2.0 * (w["w200"] + w["w020"] + w["w002"])
        )
        diag.append(-2.0 * (w[k1] + w[k2] + w[k3]) + iso / w[vc])
    return diag[0], diag[1], diag[2]


def warp_prior_symbol(
    reg: tuple[float, ...],
    vox: tuple[float, float, float],
    shape: tuple[int, int, int],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Eigenvalues of ``L``'s diagonal blocks under the DCT-II basis — ``(3, gz, gy, gx)``.

    A symmetric convolution with half-sample-symmetric (Neumann) boundaries is
    diagonalised by the DCT-II, so each diagonal block of ``L`` has the closed-form
    symbol below at ``θ_k = πk/N``. This is what makes an exact, O(N log N)
    preconditioner possible (:func:`_dct_precond`) — and it is the whole reason the
    Gauss-Newton warp can now run at SPM's regularisation strength instead of a
    weakened one (see :func:`_cg_solve`).

    The off-diagonal shear/divergence coupling (``w2·sin θ_a sin θ_b``) is **not**
    DCT-diagonal — it maps cosine modes to sine modes — so it is omitted. That only
    makes the preconditioner approximate, never wrong: CG converges regardless, this
    just decides how fast. ``mu``/``lambda`` are small next to bending anyway.
    """
    gz, gy, gx = shape
    if _all_zero(reg):
        return torch.zeros((3, gz, gy, gx), device=device, dtype=dtype)
    w = spm_vel2mom_weights(reg, vox, (gx, gy, gz))
    ax = [
        torch.arange(n, device=device, dtype=torch.float64) * (torch.pi / n) for n in (gz, gy, gx)
    ]
    cz, cy, cx = (torch.cos(a) for a in ax)
    c2z, c2y, c2x = (torch.cos(2.0 * a) for a in ax)
    # broadcast to (gz, gy, gx)
    Cx, Cy, Cz = cx[None, None, :], cy[None, :, None], cz[:, None, None]
    C2x, C2y, C2z = c2x[None, None, :], c2y[None, :, None], c2z[:, None, None]
    out = []
    for k1, k2, k3, vc in (
        ("wx100", "wx010", "wx001", "v0"),
        ("wy100", "wy010", "wy001", "v1"),
        ("wz100", "wz010", "wz001", "v2"),
    ):
        lam = 2.0 * w[k1] * (Cx - 1.0) + 2.0 * w[k2] * (Cy - 1.0) + 2.0 * w[k3] * (Cz - 1.0)
        iso = (
            w["lam0"]
            + 4.0 * w["w110"] * (Cx * Cy - 1.0)
            + 4.0 * w["w101"] * (Cx * Cz - 1.0)
            + 4.0 * w["w011"] * (Cy * Cz - 1.0)
            + 2.0 * w["w200"] * (C2x - 1.0)
            + 2.0 * w["w020"] * (C2y - 1.0)
            + 2.0 * w["w002"] * (C2z - 1.0)
        )
        out.append((lam + iso / w[vc]).expand(gz, gy, gx))
    return torch.stack(out, dim=0).to(dtype)


def _dct_matrix(n: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Orthonormal DCT-II matrix ``(n, n)``. Small enough that an explicit matmul beats
    the FFT reindexing tricks, and leaves nothing to get subtly wrong."""
    k = torch.arange(n, device=device, dtype=torch.float64)[:, None]
    i = torch.arange(n, device=device, dtype=torch.float64)[None, :]
    m = torch.cos(torch.pi * k * (2.0 * i + 1.0) / (2.0 * n)) * ((2.0 / n) ** 0.5)
    m[0] = (1.0 / n) ** 0.5
    return m.to(dtype)


def _dct3(x: Tensor, mats: tuple[Tensor, Tensor, Tensor], *, inverse: bool) -> Tensor:
    """Separable orthonormal DCT-II (or its transpose, DCT-III) over the last three axes
    of a ``(3, gz, gy, gx)`` stack."""
    for axis, m in zip((1, 2, 3), mats, strict=True):
        mm = m.T if inverse else m
        x = torch.movedim(torch.matmul(mm, torch.movedim(x, axis, -2)), -2, axis)
    return x


def _dct_precond(
    reg: tuple[float, ...],
    vox: tuple[float, float, float],
    shape: tuple[int, int, int],
    shift: Tensor,
    *,
    sym_scale: Tensor | None = None,
    device: torch.device,
    dtype: torch.dtype,
):
    """Preconditioner ``M⁻¹ ≈ (shift·I + L)⁻¹``, applied exactly in the DCT domain.

    ``shift`` is a per-component positive scalar standing in for the data Hessian
    ``Alpha``'s diagonal (its mean); it also regularises ``L``'s null space — bending
    alone is blind to affine fields, so the zero-frequency symbol is 0. ``sym_scale``
    rescales the symbol per component, which is how the caller accounts for solving in
    image-voxel units while ``L`` is defined in node units (a factor ``1/sk_c²``).

    Why this and not multigrid: the Gauss-Newton system is ``(Alpha + L)u = Beta``, and
    at SPM's regularisation ``L`` is stiff enough that plain CG needs O(1000) iterations
    — it converges fast on high-frequency modes and slowly on exactly the smooth,
    long-wavelength deformation the prior is there to produce. Truncating it (the old
    fixed 10 iterations) therefore returned the *high-frequency half* of the Newton
    step, which is why the warp came out spiky. ``L`` is the stiff part **and** it is
    DCT-diagonalisable, so inverting it exactly removes the ill-conditioning and leaves
    CG only the well-conditioned per-node ``Alpha`` variation to work on.
    """
    sym = warp_prior_symbol(reg, vox, shape, device=device, dtype=dtype)
    if sym_scale is not None:
        sym = sym * sym_scale.reshape(3, 1, 1, 1).to(dtype)
    inv = 1.0 / (sym + shift.reshape(3, 1, 1, 1).to(dtype)).clamp_min(1e-20)
    mats = tuple(_dct_matrix(n, device, dtype) for n in shape)

    def apply(r: Tensor) -> Tensor:
        s = r.permute(3, 0, 1, 2).contiguous()
        s = _dct3(_dct3(s, mats, inverse=False) * inv, mats, inverse=True)
        return s.permute(1, 2, 3, 0).contiguous()

    return apply


def fudge_factor(vox: tuple[float, float, float], sk: tuple[int, int, int], fwhm: float) -> float:
    """SPM's noise-nonindependence fudge factor ``ff`` (``spm_preproc8`` line 123).

    ``ff = prod(4π·(s/vx/sk)² + 1)^½`` with ``s = (fwhm + mean(vx))/√(8ln2)``. It
    inflates both the bias precision and the warp regularisation to (approximately)
    account for the effective number of *independent* samples given the sampling stride
    ``sk`` and image smoothness ``fwhm`` — so the regularisation strength tracks ``samp``
    the way SPM's does instead of being a fixed factor off. ``fwhm=0`` is standard for
    MRI (still ``ff>1`` via the ``mean(vx)`` term); use ~5 mm for PET/SPECT.
    """
    import math

    s = (fwhm + sum(vox) / 3.0) / math.sqrt(8.0 * math.log(2.0))
    prod = 1.0
    for vx, k in zip(vox, sk, strict=True):
        prod *= 4.0 * math.pi * (s / (vx * k)) ** 2 + 1.0
    return prod**0.5


def _cg_solve(beta: Tensor, apply_a, n_iter: int = 10, precond=None) -> Tensor:
    """(Preconditioned) conjugate gradient for the SPD system ``A·x = beta``.

    ``apply_a`` computes ``A·p``; ``precond`` (optional) applies ``M⁻¹`` and turns this
    into PCG. Used by the Gauss-Newton warp solver to invert ``(Alpha + L)`` — the
    per-node GN Hessian plus the regularisation operator — where SPM uses the
    ``spm_diffeo('fmg')`` full multigrid. Starts from ``x=0``, so the initial residual is
    ``beta``.

    **The preconditioner is not an optimisation, it is what makes the solve possible.**
    Unpreconditioned CG on this system converges quickly in the high-frequency modes and
    very slowly in the smooth ones — the opposite of what is wanted, since the smooth,
    long-wavelength deformation is precisely what the prior exists to produce. Measured
    at SPM's regularisation strength on a reference subject, plain CG reached a warp with
    mean |displacement| 0.012 / 0.25 / 0.78 / 1.20 voxels at 10 / 50 / 200 / 800
    iterations against SPM's 1.63, i.e. still climbing after 800. Truncating it — the
    fixed 10 iterations used here until 2026-07-29 — returns the high-frequency part of
    the Newton step and nothing else, which is exactly the spiky warp that was inflating
    CSF. With the DCT preconditioner (:func:`_dct_precond`) the stiff operator is
    inverted exactly and a few tens of iterations suffice.

    Runs a fixed ``n_iter`` with no residual-tolerance check: the check would compare a
    GPU scalar in a Python ``if`` every iteration, forcing a host sync that idles the GPU.
    The Armijo line search validates the resulting direction anyway, so all scalar
    arithmetic stays on-device and CG runs sync-free.
    """
    x = torch.zeros_like(beta)
    r = beta.clone()
    z = precond(r) if precond is not None else r
    p = z.clone()
    rz = (r * z).sum()
    for _ in range(n_iter):
        ap = apply_a(p)
        alpha = rz / (p * ap).sum().clamp_min(1e-30)
        x = x + alpha * p
        r = r - alpha * ap
        z = precond(r) if precond is not None else r
        rz_new = (r * z).sum()
        p = z + (rz_new / rz.clamp_min(1e-30)) * p
        rz = rz_new
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
    cov_prior: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Initialise **one Gaussian per tissue** from TPM-prior-weighted moments.

    SPM's moment init (``spm_preproc8.m:330-364``): means are ``Σ prior·y / Σ prior`` per
    tissue and every tissue starts with the *same* ``vr0``-regularised pooled covariance
    ``vr1 = (Σ_k scatter_k + N·vr0)/(Σ mom0 + N)``, with ``mg = 1``. The per-tissue
    ``ngaus`` split happens later, from the **converged** single-Gaussian fit — see
    :func:`split_gaussians`.

    Returns ``means`` ``(n_chan, n_tissue)``, ``covs`` ``(n_chan, n_chan, n_tissue)``,
    ``mix`` ``(n_tissue,)``.
    """
    n_chan = corrected.shape[1]
    n_tissue = prior_w.shape[1]

    count = prior_w.sum(dim=0)  # (n_tissue,)
    sum1 = torch.einsum("vn,vt->nt", corrected, prior_w)
    mean1 = sum1 / count.clamp_min(1e-30)  # (n_chan, n_tissue)
    # pooled within-tissue scatter, regularised toward the data covariance
    scatter = corrected.new_zeros((n_chan, n_chan))
    for t in range(n_tissue):
        s2 = torch.einsum("vn,vm->nm", corrected * prior_w[:, t : t + 1], corrected)
        scatter = scatter + (s2 - count[t] * torch.outer(mean1[:, t], mean1[:, t]))
    var1 = (scatter + n_chan * cov_prior) / (count.sum() + n_chan)  # (n_chan, n_chan)

    means = mean1.to(torch.float64)
    covs = var1.to(torch.float64)[:, :, None].expand(n_chan, n_chan, n_tissue).contiguous()
    mix = torch.ones(n_tissue, dtype=torch.float64, device=corrected.device)
    return means, covs, mix


def split_gaussians(
    means1: Tensor, covs1: Tensor, ngaus: list[int], *, seed: int = 0
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split a converged one-Gaussian-per-tissue fit into ``ngaus[t]`` components each.

    SPM's crude heuristic (``spm_preproc8.m:646-675``), run **after** the first full GMM
    round (20 sub-iterations) and the first bias update — not at initialisation:

        w  = 1/(1+exp(-(kk-1)·0.25)) - 0.5        (0 when kk == 1)
        mn = sqrtm(vr1_k)·randn(N,kk)·w + mn1_k
        vr = vr1_k·(1-w),        mg = 1/kk

    The jitter and the component covariances are scaled by that tissue's **own converged
    covariance** ``vr1_k``. Seeding the split from the pooled/global variance instead —
    which is what happens if you split at moment-init time, when every tissue still shares
    one covariance — scatters the components of a tight class (WM) as widely as a broad
    one (soft tissue), and lands the extra Gaussians of the multi-component classes
    (CSF ×2, bone ×3, soft ×4) in the wrong places. That was the behaviour here until
    2026-07-22.

    Args:
        means1: ``(n_chan, n_tissue)`` converged per-tissue means.
        covs1: ``(n_chan, n_chan, n_tissue)`` converged per-tissue covariances.
        ngaus: per-tissue Gaussian counts (SPM's default ``[1 1 2 3 4 2]``).
        seed: RNG seed for the jitter (SPM fixes its generator likewise).

    Returns:
        means, covs, mix, tissue_of — as consumed by :func:`gmm_responsibilities`.
    """
    n_chan, n_tissue = means1.shape
    device = means1.device
    tissue_of = _build_tissue_of(ngaus).to(device)
    n_gauss = tissue_of.numel()
    means = torch.empty((n_chan, n_gauss), dtype=torch.float64, device=device)
    covs = torch.empty((n_chan, n_chan, n_gauss), dtype=torch.float64, device=device)
    mix = torch.empty(n_gauss, dtype=torch.float64, device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for t in range(n_tissue):
        cols = (tissue_of == t).nonzero(as_tuple=True)[0]
        kk = cols.numel()
        w = 1.0 / (1.0 + torch.exp(torch.tensor(-(kk - 1) * 0.25))) - 0.5  # 0 when kk==1
        vr_t = covs1[:, :, t].to(torch.float64)
        chol = torch.linalg.cholesky(vr_t)  # sqrtm's role: a factor with chol·cholᵀ = vr_t
        jitter = torch.randn(n_chan, kk, generator=gen).to(device=device, dtype=torch.float64)
        means[:, cols] = (chol @ jitter) * w + means1[:, t : t + 1].to(torch.float64)
        covs[:, :, cols] = (vr_t * (1.0 - w))[:, :, None]
        mix[cols] = 1.0 / kk
    return means, covs, mix, tissue_of


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
    reg: tuple[float, ...] = (0.0, 0.0, 0.1, 0.01, 0.04),
    fwhm: float = 0.0,
    samp: float = 3.0,
    dither: float | list[float] = 0.0,
    n_iter: int = 30,
    n_inner: int = 8,
    split_after: int = 2,
    min_iter: int = 10,
    tol: float = 1e-4,
    tpm_bottom_mm: float = 5.0,
    tpm_fov_mm: float | None = None,
    fit_prior_interp: str = "bspline2",
    fit_warp: bool = True,
    warp_anneal: bool = True,
    pe_axis: int | None = None,
    reverse_volume: Tensor | list[Tensor] | None = None,
    blur_tpms: float = 0.0,
    blur_frac: float = 0.4,
    warp_focus: float = 0.0,
    focus_quantile: float = 0.9,
    wp_reg: float = 100.0,
    bias_solver: str = "gn",
    bias_iters: int = 1,
    bias_lr: float = 0.1,
    warp_solver: str = "gn",
    warp_lr: float = 1.0,
    warp_iters: int | None = None,
    warp_cg_iters: int = 32,
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

    The loop mirrors ``spm_preproc8``'s nesting: ``n_iter`` (SPM 30) outer iterations, each
    re-sampling the warped TPM then running up to ``n_inner`` (SPM 8) rounds of
    [GMM ×20 sub-iterations → bias update], then the deformation. ``tol`` is SPM's
    ``tol1``: every break tests the **absolute** log-likelihood gain against
    ``tol·n_samp`` (``2·tol·n_samp`` for the outer/inner loops), not a relative change.
    ``min_iter`` (SPM 10) is a floor on the outer loop — the heavy-to-light warp anneal
    (``2^max(10-iter,0)`` on the bending term) only reaches its target at iteration 10, so
    stopping earlier leaves the deformation clamped by up to 2¹⁰ and effectively unfitted.

    ``split_after`` is how many ``[GMM, bias]`` rounds run before the per-tissue Gaussians
    are expanded to ``ngaus`` (:func:`split_gaussians`). SPM splits after the **first**
    round; the problem is that the first round's GMM runs on the *uncorrected* image
    (``Tbias`` starts at zero and the bias step comes after), so every tissue's covariance
    is still inflated by the bias field it has not seen removed yet. ``split_gaussians``
    then hands both children that inflated covariance (``vr = vr1·(1-w)``), and a
    too-broad pair is free to bifurcate: one child contracts onto the tissue while the
    other expands onto the intensity tails and becomes an outlier-catcher, spending a
    Gaussian on a fraction of a percent of the data.

    Measured on a reference T1 at ``samp 3``: splitting after 1 round starts from a CSF
    sd of 196 (the converged single-Gaussian value is 83) and the pair ends at means
    284/1009 with mixing 0.97/0.03 — one real CSF Gaussian and one catcher sitting on
    ~180 samples of bright skull-base tissue. Waiting one more round starts from sd 116
    and gives 226/340 at 0.41/0.59, against SPM's 224/349 at 0.43/0.57. The failure is
    deterministic (six different split seeds collapse identically) and is *not* a
    convergence-effort problem (``tol=0``, 2296 M-steps, collapses the same way).

    Default 2 rather than SPM's 1 because it reproduces SPM's *fitted Gaussians* far more
    closely, which is the thing that matters; ``split_after=1`` restores SPM's literal
    schedule. Note this also makes ``ngaus`` behave the way a user expects — with the
    early split, adding a Gaussian to a tissue could simply feed the catcher instead of
    modelling that tissue's structure.

    ``tpm_bottom_mm`` (SPM 5) drops samples the affine places below the bottom of the TPM
    — on a whole-head T1 that is the neck and shoulders, which SPM never fits. Set 0 to
    keep everything (e.g. a template that already covers the sample volume).
    ``tpm_fov_mm`` goes further (beyond SPM): drop samples further than this many mm
    outside the template box on **any** axis. See :func:`tpm_coverage_mask`; it is also
    carried into :func:`segment_apply`, which is where it matters most.

    ``fit_prior_interp`` picks the TPM kernel used *inside* the fit: ``"bspline2"``
    (default) is SPM's own degree-2 B-spline on unprefiltered data — an approximating,
    mildly low-pass kernel — and ``"linear"`` is trilinear, cheaper (one ``grid_sample``
    vs 27 gathers) but a sharper, ``C⁰`` prior with a piecewise-constant gradient.

    ``warp_smooth`` (grid-node sigma) Sobolev-smooths the deformation each iteration so
    the dense field stays smooth rather than speckly. ``pe_axis`` (0/1/2 = x/y/z), when
    set, constrains the deformation to that single voxel axis — the phase-encode
    direction — so the warp is a pure 1-D EPI distortion field (PE-mode, like ffs_rbr).

    ``reverse_volume`` (PE-mode only) enables **dual-echo distortion correction**: a
    reverse-phase-encode EPI (same grid, opposite blip) images the same anatomy with the
    *opposite* distortion, so its correcting warp is exactly ``−u`` where the forward's is
    ``+u``. Passing it makes the single PE warp be driven by **both** echoes at once — the
    forward samples pull the TPM at ``coords + u`` and the reverse samples at ``coords − u``,
    sharing the GMM/bias/TPM. This uses both blips instead of discarding the reverse (à la
    TOPUP), doubling the information constraining the residual distortion field. The output
    ``twarp`` is the forward warp ``+u`` (negate for the reverse's own correction).

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

    ``reg`` is SPM's full deformation regulariser ``(absolute, membrane, bending, le1,
    le2)`` (default ``(0, 0, 0.1, 0.01, 0.04)``); a 3-tuple is accepted (``le1=le2=0``).
    ``fwhm`` (image smoothness, mm) feeds SPM's fudge factor ``ff`` that inflates both the
    bias precision and the warp regulariser to account for spatially-correlated noise —
    0 for MRI, ~5 for PET/SPECT. Both the bias and warp regularisers now carry ``ff`` and
    the warp penalty uses the true node spacing (``samp·vox``), so their strength tracks
    ``samp`` as in SPM. ``warp_anneal`` (default on) applies SPM's heavy-to-light schedule
    — the bending term is stiffened by up to ``2¹⁰`` early and relaxed to target by iter
    10, so the warp can't over-commit before the intensity model settles.

    ``bias_solver`` selects the bias-field optimiser. ``"gn"`` (default) is SPM's analytic
    Gauss-Newton on the DCT coefficients (``spm_preproc8`` eq. 34): a closed-form gradient
    ``Beta`` and GN Hessian ``Alpha`` assembled from the responsibility-weighted per-voxel
    terms, solved as ``(Alpha + C)u = Beta + C·T`` with Armijo backtracking — no autograd
    graph, recovers the field far more accurately than gradient descent, and is faster.
    ``bias_iters`` (default 1) is the number of GN sweeps per **inner** round — SPM's
    ``for subit=1:1`` (``spm_preproc8.m:532``), so with ``n_inner`` restored to 8 the bias
    gets SPM's 8 sweeps per outer iteration. It was 2 while the inner loop was collapsed to
    a single round, which after that fix would have doubled SPM's bias work; each sweep
    costs a full E-step pass per Armijo probe, so this is a real cost, not a rounding
    detail. ``"adam"`` is the
    autograd gradient-descent fallback (``bias_lr``). Both include the ``+Σ log|bias|``
    change-of-variables Jacobian (SPM's ``+1`` in eq. 34 / ``chan.ll += sum(bf)``).

    ``warp_solver`` selects the deformation optimiser. ``"gn"`` (default) is SPM's
    **Gauss-Newton** update (``spm_preproc8``): per-node gradient + rank-1 GN Hessian
    ``dp·dpᵀ``, the regularisation operator added, solved by **DCT-preconditioned**
    conjugate gradient (``warp_cg_iters``, default 32; in place of SPM's multigrid), with
    Armijo backtracking. The preconditioner is load-bearing, not a speed tweak: the
    regularisation operator is stiff and ill-conditioned in exactly the smooth modes the
    prior exists to produce, so unpreconditioned CG needs O(1000) iterations and a
    truncated run yields only the high-frequency part of the Newton step — a spiky warp.
    Because ``L`` is a symmetric convolution with Neumann boundaries it is exactly
    DCT-diagonalisable, so ``(shift·I + L)⁻¹`` is available in closed form
    (:func:`_dct_precond`). It **converges** — the warp reaches a stable
    fixed point and stops growing once the fit settles, so a large ``n_iter`` is safe
    (matches SPM). ``"adam"`` is the autograd + Sobolev-smoothing update (``warp_lr``/
    ``warp_smooth``/``warp_focus`` apply), but it **accumulates** displacement across EM
    iterations rather than converging, so the warp keeps growing with ``n_iter`` and can drag
    the tissue priors out of place (GM/CSF into the skull/neck) — use it only for short runs.

    The warp regulariser follows SPM's ``param``: ``prod(sk·vx)·ff·reg`` (the ``prod`` node-
    volume factor is essential — without it the deformation is ~20× under-regularised and
    runs away). Global warp **aggressiveness** (distinct from the localised ``warp_focus``,
    Adam only) is set by ``reg`` (lower the bending/elastic terms to allow larger folds) and,
    for Adam, ``warp_lr``/``warp_iters``/``warp_smooth``.

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
    # Warp sub-iterations per EM step: GN converges in ~3 (each is a full CG solve + Armijo
    # line search over all samples — expensive), Adam wants more small gradient steps.
    if warp_iters is None:
        warp_iters = 3 if warp_solver == "gn" else 8
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
    coords_full = torch.stack([fx, fy, fz], dim=1).to(wdt)  # (n_grid, 3)
    flat_idx_full = torch.arange(coords_full.shape[0], device=device)  # flat grid index

    if vox2vox is None:
        if world_affine is None:
            raise ValueError("fit_segment needs either world_affine or vox2vox")
        vox2vox = compose_vox2vox(
            subj_affine.to(device), tpm_affine.to(device), world_affine.to(device)
        )
    else:
        vox2vox = vox2vox.to(device=device, dtype=torch.float64)

    # Reject samples the affine places outside the template's usable field: SPM's
    # inferior-only `bottom_mm` rule, plus the optional full-box `fov_mm` rule. Uses the
    # affine alone, as SPM does (the rule is applied before any warp).
    tpm_shape = (log_prior.shape[1], log_prior.shape[2], log_prior.shape[3])
    tpm_vox = tuple(
        float(v) for v in torch.linalg.norm(tpm_affine[:3, :3].to(torch.float64), dim=0)
    )
    in_tpm = tpm_coverage_mask(
        apply_affine_pts(coords_full.to(torch.float64), vox2vox),
        tpm_shape,
        tpm_vox,
        bottom_mm=tpm_bottom_mm,
        fov_mm=tpm_fov_mm,
    )

    def _samples(v: Tensor) -> tuple[Tensor, Tensor]:
        it = v[:, fz, fy, fx].T.contiguous()  # (n_grid, n_chan)
        kp = torch.isfinite(it).all(dim=1) & (it != 0).all(dim=1) & in_tpm
        return it, kp

    intens_f, keep = _samples(vols)
    # PE dual-echo distortion mode: a reverse-phase-encode EPI images the SAME anatomy with
    # the OPPOSITE distortion, so its correcting warp is exactly −u (the forward's is +u).
    # Stack it as extra samples that share the GMM/bias/TPM and carry warp_sign=−1, so the
    # single warp field is driven by BOTH echoes at once (a dual optimisation — both blips,
    # not one). Requires pe_axis (the opposite-sign relation only holds along the PE axis).
    if reverse_volume is not None:
        if pe_axis is None:
            raise ValueError("reverse_volume requires pe_axis (the ± relation is PE-only)")
        if isinstance(reverse_volume, (list, tuple)):
            rvols = torch.stack([v.to(device=device, dtype=wdt) for v in reverse_volume], dim=0)
        else:
            reverse_volume = reverse_volume.to(device=device, dtype=wdt)
            rvols = reverse_volume if reverse_volume.ndim == 4 else reverse_volume[None]
        intens_r, keep_r = _samples(rvols)
        coords = torch.cat([coords_full[keep], coords_full[keep_r]], dim=0)
        intens = torch.cat([intens_f[keep], intens_r[keep_r]], dim=0)
        kept_flat = torch.cat([flat_idx_full[keep], flat_idx_full[keep_r]], dim=0)
        warp_sign = torch.cat(
            [coords.new_ones(int(keep.sum())), -coords.new_ones(int(keep_r.sum()))]
        )
    else:
        coords, intens = coords_full[keep], intens_f[keep]
        kept_flat = flat_idx_full[keep]
        warp_sign = coords.new_ones(coords.shape[0])

    # Integer-data dither (SPM `scrand`, spm_preproc8.m:212-217, 259-266). An image stored
    # as int16 has intensities on a lattice of `scl_slope`, and the GMM will happily put a
    # near-zero-variance Gaussian on one of those spikes — the bias field then has an
    # aliasing pattern to chase. SPM adds one quantisation step of uniform noise,
    # `rand*slope - slope/2`, to break the lattice up. Applied once to the sampled values,
    # as SPM does, so it is fixed for the whole fit; seeded for reproducibility.
    if dither:
        steps = [float(dither)] * n_chan if isinstance(dither, (int, float)) else list(dither)
        if len(steps) != n_chan:
            raise ValueError(f"dither must be a scalar or {n_chan} values, got {len(steps)}")
        gen = torch.Generator(device="cpu").manual_seed(0)
        for c, step in enumerate(steps):
            if step <= 0:
                continue
            noise = torch.rand(intens.shape[0], generator=gen).to(device=device, dtype=wdt)
            intens[:, c] += (noise - 0.5) * step

    # SPM's fudge factor (non-independence of smoothed voxels): scales BOTH the bias
    # precision and the warp regularisation so their strength tracks the sampling stride
    # `sk` the way SPM's does. `node_vox` is the warp grid's node spacing (sk·vx mm),
    # the correct step for the deformation regulariser (the field lives on the subsampled
    # grid, so full-res `vox` would over-penalise by `sk` per derivative).
    ff = fudge_factor(vox, tuple(sk), fwhm)
    node_vox = (sk[0] * vox[0], sk[1] * vox[1], sk[2] * vox[2])
    # SPM scales the warp reg by the node volume `prod(sk·vx)` (spm_preproc8 `param`), NOT
    # just `ff`. Dropping it (an earlier "correctness" simplification) made the deformation
    # ~20× under-regularised → the warp ran away over iterations, dragging the GM/CSF priors
    # into the skull/neck (worse with more iters). Restored: warp reg = prod·ff·reg, matching
    # SPM's shipped magnitude so the fit converges to a stable, modest warp.
    prod_node_vox = node_vox[0] * node_vox[1] * node_vox[2]

    # bias field basis shape (SPM 1-based DCT positions) + precision; the per-sample
    # bases themselves are (re)built per chunk in _basis, not stored full-size — at a
    # fine samp they would be the single largest persistent allocation.
    (nbx, nby, nbz), bias_prec = bias_field_shape(
        (nx, ny, nz), vox, biasfwhm, biasreg, ff, device=device
    )
    d3 = nbx * nby * nbz  # DCT coefficients per channel (bias GN works in this flat space)
    # NB the GN assembly no longer builds a (chunk, d3) design matrix — it contracts the
    # lattice axis by axis (`_assemble_separable`) — so its transients are just the
    # per-sample responsibilities and `bias_chunk` is simply `fit_chunk` (set below).

    # Wishart covariance prior vr0 = diag(per-channel variance)/Kb² (SPM eq. after 25);
    # kept float64 for the M-step. Diagonal even for multi-channel (SPM's vr0).
    data_var = intens.var(dim=0, unbiased=False).to(torch.float64)  # (n_chan,)
    cov_prior = torch.diag(data_var / n_tissue**2)  # (n_chan, n_chan)

    # SPM starts with lkp = 1:Kb — ONE Gaussian per tissue — and only expands to `ngaus`
    # after the first [GMM, bias] round, from the converged per-tissue fit (see
    # `split_gaussians`). `n_gauss_max` is the eventual count, used for chunk sizing so the
    # memory estimate doesn't have to be revised after the split.
    n_gauss_max = sum(ngaus)
    tissue_of = torch.arange(n_tissue, dtype=torch.long, device=device)
    # one bias field per channel (SPM's chan(n).T): (n_chan, nbx, nby, nbz)
    coef = torch.zeros((n_chan, nbx, nby, nbz), dtype=wdt, device=device)
    twarp = torch.zeros((*grid_shape, 3), dtype=wdt, device=device)
    n_grid = coords_full.shape[0]  # flat warp-grid node count (kept_flat indexes into this)
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
            (n_gauss_max + n_tissue) * 8,
            device,
            operation="glm",
            use_double=(wdt == torch.float64),
            max_chunk_size=n_samp,
        )
    fit_chunk = max(1, min(int(fit_chunk), n_samp))
    bias_chunk = fit_chunk
    n_chunks = (n_samp + fit_chunk - 1) // fit_chunk
    if verbose:
        how = "single pass — all samples fit" if n_chunks == 1 else f"{n_chunks} chunks"
        print(
            f"segment fit: {n_samp:,} samples ({samp:g} mm, sk={sk}), {wdt}, "
            f"chunk={fit_chunk:,} → {how}"
        )

    def _chunks():
        for s in range(0, n_samp, fit_chunk):
            yield slice(s, min(s + fit_chunk, n_samp))

    # The DCT bases depend only on the (fixed) sample coordinates, never on a parameter,
    # yet `_basis` is called on every bias objective evaluation — a few hundred times per
    # EM iteration once the Armijo line searches are counted. Cache them whole when they
    # are small, and fall back to rebuilding per chunk when they are not: at a fine `samp`
    # the full bases would be the single largest persistent allocation, which is why they
    # were not cached before.
    _basis_bytes = n_samp * (nbx + nby + nbz) * (8 if wdt == torch.float64 else 4)
    _basis_cache: tuple[Tensor, Tensor, Tensor] | None = None
    if _basis_bytes <= 256 * 1024 * 1024:
        _basis_cache = (
            dct_basis(coords[:, 0] + 1.0, nx, nbx).to(wdt),
            dct_basis(coords[:, 1] + 1.0, ny, nby).to(wdt),
            dct_basis(coords[:, 2] + 1.0, nz, nbz).to(wdt),
        )

    def _basis(sl: slice) -> tuple[Tensor, Tensor, Tensor]:
        if _basis_cache is not None:
            return (_basis_cache[0][sl], _basis_cache[1][sl], _basis_cache[2][sl])
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

    def warped_prior_raw(active_log_prior: Tensor) -> Tensor:
        """**Unweighted** warped-TPM prior at every sample (SPM's ``buf.dat``).

        Held raw, not ``wp``-weighted, because ``wp`` is re-estimated on every GMM
        sub-iteration (SPM applies the weights inside ``latent``/``log_spatial_priors``,
        and needs the unweighted ``B`` for the ``mgm`` expected-mass term).
        """
        out = torch.empty((n_samp, n_tissue), dtype=wdt, device=device)
        with torch.no_grad():
            for sl in _chunks():
                disp = warp_sign[sl][:, None] * twarp.reshape(-1, 3)[kept_flat[sl]]
                tpm_coords = apply_affine_pts(coords[sl] + disp, vox2vox)
                out[sl] = sample_tpm_prior(
                    active_log_prior, tpm_coords, bg_low, bg_high, kernel=fit_prior_interp
                )
        return out

    def _prior_w(sl: slice) -> Tensor:
        """``wp``-weighted prior for a chunk (SPM ``log_spatial_priors``, pre-log)."""
        return weight_prior(prior_raw[sl], wp)

    # Warp regularisation: SPM's full 5-vector (abs, mem, bend, mu, lambda), scaled by the
    # fudge factor `ff`, and re-derived each EM iteration for the heavy-to-light schedule
    # (see the loop). `warp_reg` is the Adam penalty weight; `warp_reg_gn` adds a small
    # absolute floor so the Gauss-Newton (Alpha + L) system is SPD for the CG solve even
    # when the user gives no membrane/absolute term (bending alone has an affine null
    # space). Both are nonlocal so the closures below always see the current iteration's
    # values; initialised here (iter-0) so the references exist before the loop.
    reg5 = tuple(reg) + (0.0,) * (5 - len(reg))

    def _scale_warp_reg(anneal: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
        # SPM's param(4:8) = prod(vx·sk)·ff·reg; the bending term (param(6)) is additionally
        # scaled heavy→light by `anneal`.
        scale = prod_node_vox * ff
        wr = tuple(scale * (r * anneal if j == 2 else r) for j, r in enumerate(reg5))
        wr_gn = (max(wr[0], 1e-3), *wr[1:])
        return wr, wr_gn

    # (re)assigned each EM iteration for the heavy-to-light schedule; the GN closures
    # above and the Adam step below read the current iteration's values (late binding).
    warp_reg, warp_reg_gn = _scale_warp_reg(1.0)  # noqa: F841  (overwritten in the loop)

    # SPM regularises the deformation in NODE units, not image-voxel units: it feeds
    # `Twarp./sk4` to vel2mom and rescales the solved update by `sk4` (spm_preproc8.m:805,
    # 808), and its data gradient is in the same units because MM = M*MT carries the `sk`
    # scaling (line 765). Our `twarp` is in image-voxel units with a data gradient to
    # match, so the penalty must see `twarp/sk` or the prior is `sk_i²` too strong for the
    # same `reg` — 9× at 1 mm/samp 3, 16× at 0.7 mm/samp 3, which is why the warp
    # under-deformed and needed compensating knobs. `sk` is constant, so autograd carries
    # the chain rule through for the GN regularisation operator.
    sk_t = torch.tensor([float(sk[0]), float(sk[1]), float(sk[2])], dtype=wdt, device=device)

    def _warp_prior_energy(tw: Tensor, weights: tuple[float, ...]) -> Tensor:
        """SPM's ``½·uᵀLu`` deformation prior, evaluated in node units."""
        return warp_penalty(tw / sk_t, weights, node_vox)

    def _pe_project(field: Tensor) -> Tensor:
        """Zero the non-phase-encode components in place (PE-mode constraint)."""
        if pe_axis is not None:
            for c in range(3):
                if c != pe_axis:
                    field[..., c] = 0.0
        return field

    def _warp_data_term(tw: Tensor, active_log_prior: Tensor, *, want_grad: bool):
        """Chunked data term ``-Σ log-likelihood(warp)`` (+ its autograd gradient).

        Uses the per-EM-iteration ``tissue_lik`` (Gaussians already collapsed), so only the
        warped prior is recomputed per call — no Cholesky/Mahalanobis in the warp loop.
        """
        tw = tw.detach().requires_grad_(want_grad)
        # Accumulate the objective on the GPU and sync ONCE at the end (never in the gradient
        # path, where the total is discarded) — a per-chunk `.item()` stalled the GPU on a
        # host round-trip every chunk, the main cause of the GN warp's low GPU utilisation.
        tot = torch.zeros((), device=device) if not want_grad else None
        for sl in _chunks():
            disp = warp_sign[sl][:, None] * tw.reshape(-1, 3)[kept_flat[sl]]
            tpm_coords = apply_affine_pts(coords[sl] + disp, vox2vox)
            pw = weight_prior(
                sample_tpm_prior(
                    active_log_prior, tpm_coords, bg_low, bg_high, kernel=fit_prior_interp
                ),
                wp,
            )
            nll = -torch.log((tissue_lik[sl] * pw).sum(dim=1).clamp_min(1e-40)).sum()
            if want_grad:
                nll.backward()
            elif tot is not None:
                tot = tot + nll.detach()
        if not want_grad:
            assert tot is not None
            return tot.item()
        grad = tw.grad if tw.grad is not None else torch.zeros_like(tw)
        return 0.0, grad.detach()

    def _reg_grad(field: Tensor) -> Tensor:
        """Regularisation operator ``L·field``, analytic (see :func:`warp_prior_grad`).

        Node units in and out: the energy is ``P(u/sk)``, so the chain rule contributes a
        ``1/sk`` on the way in and another on the way out.
        """
        return warp_prior_grad(field.detach() / sk_t, warp_reg_gn, node_vox) / sk_t

    def _warp_objective(tw: Tensor, active_log_prior: Tensor) -> float:
        """Penalised negative log-likelihood the GN line search must decrease."""
        return (
            _warp_data_term(tw, active_log_prior, want_grad=False)
            + _warp_prior_energy(tw, warp_reg_gn).item()
        )

    # --- analytic Gauss-Newton bias helpers (used only when bias_solver == "gn") ---
    def _bias_chunks():
        for s in range(0, n_samp, bias_chunk):
            yield slice(s, min(s + bias_chunk, n_samp))

    def _khatri_rao_rows(bx: Tensor, by: Tensor, bz: Tensor) -> Tensor:
        """Row-wise DCT design matrix ``Phi[v, flat(i,j,k)] = bx[v,i]·by[v,j]·bz[v,k]`` —
        ``(n, d3)``, flattened C-order to match ``coef.reshape(-1)`` (so ``Phi @ coef.ravel``
        reproduces :func:`eval_log_bias`). Kept for the reference implementation the
        separable assembly is tested against; the fit itself no longer builds it."""
        n = bx.shape[0]
        xy = (bx[:, :, None] * by[:, None, :]).reshape(n, -1)  # (n, nbx·nby)
        return (xy[:, :, None] * bz[:, None, :]).reshape(n, -1)  # (n, d3)

    # --- separable (Kronecker) bias-GN assembly, SPM's `spm_krutil` structure ---------
    #
    # The GN normal equations are
    #     Beta[i,j,k]            = Σ_v wt1[v]·B1[x,i]·B2[y,j]·B3[z,k]
    #     Alpha[(ijk),(i'j'k')]  = Σ_v wt2[v]·B1[x,i]B1[x,i']·B2[y,j]B2[y,j']·B3[z,k]B3[z,k']
    #
    # Building the dense `(n_samp, d3)` design matrix and forming `Phiᵀ diag(wt2) Phi`
    # costs O(n_samp·d3²) — the single largest FLOP sink in the fit, and it grows
    # QUADRATICALLY as `-biasfwhm` is lowered. SPM never forms it: the samples sit on a
    # regular lattice and the basis is separable, so it contracts one axis at a time
    # (`kron(b3*b3', spm_krutil(wt2,B1,B2,1))`). Same arithmetic, O(ngz·nbx²nby²nbz²)
    # instead. On the reference T1 at `-biasfwhm 30` (d3 = 13·17·14 = 3094) that is
    # 2.4 TMAC/sweep versus 6.4e8 — ~3700x fewer operations.
    #
    # The per-axis bases are evaluated on the LATTICE (a few hundred rows), not per sample;
    # `wt1`/`wt2` are scattered back onto the lattice with zeros outside the mask, exactly
    # as SPM zero-fills `wt1(buf(z).msk)`. `scatter_add_` (not assignment) because
    # dual-echo PE mode maps two samples to one node and their contributions must sum.
    ngz, ngy, ngx = grid_shape
    lat_b1 = dct_basis(gx.to(torch.float64) + 1.0, nx, nbx).to(wdt)  # (ngx, nbx)
    lat_b2 = dct_basis(gy.to(torch.float64) + 1.0, ny, nby).to(wdt)  # (ngy, nby)
    lat_b3 = dct_basis(gz.to(torch.float64) + 1.0, nz, nbz).to(wdt)  # (ngz, nbz)
    # outer products per axis: P[a, i*nb + i'] = B[a,i]·B[a,i']
    lat_p1 = (lat_b1[:, :, None] * lat_b1[:, None, :]).reshape(ngx, -1)
    lat_p2 = (lat_b2[:, :, None] * lat_b2[:, None, :]).reshape(ngy, -1)
    lat_p3 = (lat_b3[:, :, None] * lat_b3[:, None, :]).reshape(ngz, -1)
    # z-chunk so the (nz_c, nbx², nby²) intermediate stays bounded at a fine `samp`
    z_chunk = max(1, int(40_000_000 // max(nbx * nbx * nby * nby, 1)))

    def _assemble_separable(wt1_g: Tensor, wt2_g: Tensor) -> tuple[Tensor, Tensor]:
        """``(Beta, Alpha)`` from lattice-shaped weights ``(ngz, ngy, ngx)``."""
        beta_acc = torch.zeros((nbx, nby, nbz), dtype=torch.float64, device=device)
        alpha_acc = torch.zeros(
            (nbx * nbx * nby * nby, nbz * nbz), dtype=torch.float64, device=device
        )
        for s in range(0, ngz, z_chunk):
            e = min(s + z_chunk, ngz)
            # Beta: contract x, then y, then z
            t1 = wt1_g[s:e].reshape(-1, ngx) @ lat_b1  # (nz_c·ngy, nbx)
            u1 = torch.einsum("zya,yb->zab", t1.reshape(e - s, ngy, nbx), lat_b2)
            beta_acc += torch.einsum("zab,zc->abc", u1, lat_b3[s:e]).double()
            # Alpha: same, on the per-axis outer products
            t2 = wt2_g[s:e].reshape(-1, ngx) @ lat_p1  # (nz_c·ngy, nbx²)
            u2 = torch.einsum("zya,yb->zab", t2.reshape(e - s, ngy, nbx * nbx), lat_p2)
            alpha_acc += (u2.reshape(e - s, -1).T @ lat_p3[s:e]).double()
        # (i,i',j,j',k,k') → (i,j,k, i',j',k'), C-order flattening to match coef.reshape(-1)
        alpha = (
            alpha_acc.reshape(nbx, nbx, nby, nby, nbz, nbz)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(d3, d3)
        )
        return beta_acc.reshape(-1), alpha

    def _bias_objective(coef_val: Tensor) -> float:
        """Bias-relevant objective (SPM's ``ll`` terms that move with the bias): data ll +
        Σ log|bias| (the change-of-variables Jacobian) − Σ 0.5·TᵀCT smoothness prior."""
        total = 0.0
        with torch.no_grad():
            for sl in _chunks():
                bx, by, bz = _basis(sl)
                logb = [eval_log_bias(coef_val[c], bx, by, bz) for c in range(n_chan)]
                corr = torch.stack([intens[sl, c] * torch.exp(logb[c]) for c in range(n_chan)], 1)
                _, ll = gmm_responsibilities(corr, _prior_w(sl), means, covs, mix, tissue_of)
                total += ll.sum().item() + sum(lb.sum().item() for lb in logb)
        total -= sum(bias_penalty_value(coef_val[c], bias_prec).item() for c in range(n_chan))
        return total

    def _bias_gn_sweep(coef_val: Tensor) -> Tensor:
        """One SPM Gauss-Newton sweep over the channels' DCT bias coefficients (eq. 34).

        For each channel a Gauss-Seidel update: assemble the closed-form gradient ``Beta``
        and GN Hessian ``Alpha`` from the responsibility-weighted per-voxel terms, solve
        ``(Alpha + C)·u = Beta + C·T`` (``C`` = the DCT smoothness precision), and Armijo-
        backtrack on :func:`_bias_objective` so every accepted step improves it. The ``+1``
        in ``wt1``/``wt2`` is the multiplicative bias's Jacobian (SPM ``chan.ll += sum(bf)``).
        """
        assert means is not None and covs is not None  # set by the EM loop before any sweep
        coef_val = coef_val.detach().clone()
        # per-Gaussian precisions covⁿ¹ (n_gauss, n_chan, n_chan), float64 for conditioning
        prec = torch.cholesky_inverse(
            torch.linalg.cholesky(covs.to(torch.float64).permute(2, 0, 1)),
        )
        c_diag = bias_prec.reshape(-1).to(torch.float64)  # diagonal DCT precision (d3,)
        # Assemble in the working dtype (float32): the (chunk × d3) DCT design matrix `Phi`
        # and its `Phiᵀ·diag(wt2)·Phi` GEMM dominate the whole fit, and a float64 GEMM runs
        # ~30× slower on consumer GPUs. The GN Hessian is well-conditioned (+C diagonal,
        # Armijo-guarded), so float32 is plenty; only the tiny d3×d3 solve + the cross-chunk
        # accumulator stay float64.
        prec_w = prec.to(wdt)  # (n_gauss, n_chan, n_chan)
        means_w = means.to(wdt)  # (n_chan, n_gauss)
        for c in range(n_chan):
            # accumulate the per-voxel GN weights on the sample lattice, zero outside the
            # mask (SPM's `wt1 = zeros(d(1:2)); wt1(buf(z).msk) = ...`)
            wt1_g = torch.zeros(n_grid, dtype=wdt, device=device)
            wt2_g = torch.zeros(n_grid, dtype=wdt, device=device)
            with torch.no_grad():
                for sl in _bias_chunks():
                    bx, by, bz = _basis(sl)
                    corr = torch.stack(
                        [intens[sl, cc] * torch.exp(eval_log_bias(coef_val[cc], bx, by, bz))
                         for cc in range(n_chan)], dim=1,
                    )  # fmt: skip
                    resp, _ = gmm_responsibilities(corr, _prior_w(sl), means, covs, mix, tissue_of)
                    # w0[v,k] = Σ_n1 prec[k,n1,c]·(mean[n1,k] − corrected[v,n1])  (US eq.34)
                    diff = means_w.T[None, :, :] - corr[:, None, :]  # (chunk, n_gauss, n_chan)
                    w0 = torch.einsum("kn,vkn->vk", prec_w[:, :, c], diff)
                    w1 = (resp * w0).sum(dim=1)  # (chunk,)
                    w2 = resp @ prec_w[:, c, c]  # (chunk,)
                    cr_c = corr[:, c]
                    wt1_g.scatter_add_(0, kept_flat[sl], -(1.0 + cr_c * w1))  # 1 = Jacobian
                    wt2_g.scatter_add_(0, kept_flat[sl], cr_c * cr_c * w2 + 1.0)  # PSD, ≥1
                beta, alpha = _assemble_separable(
                    wt1_g.reshape(grid_shape), wt2_g.reshape(grid_shape)
                )
            t_flat = coef_val[c].reshape(-1).to(torch.float64)
            update = (
                torch.linalg.solve(alpha + torch.diag(c_diag), beta + c_diag * t_flat)
                .reshape(coef_val[c].shape)
                .to(wdt)
            )
            # Armijo backtracking on the true objective (SPM's bias line search)
            base = _bias_objective(coef_val)
            armijo = 1.0
            for _ in range(12):
                cand = coef_val.clone()
                cand[c] = coef_val[c] - armijo * update
                if _bias_objective(cand) >= base:
                    coef_val = cand
                    break
                armijo *= 0.5
        return coef_val

    means = covs = mix = None
    prior_raw = None  # (n_samp, n_tissue) unweighted warped TPM, refreshed per outer iter
    tissue_lik = None  # (n_samp, n_tissue) collapsed Gaussian likelihood, refreshed per iter
    split_done = False
    rounds_done = 0  # completed [GMM, bias] rounds in outer iteration 0, for `split_after`
    # SPM's tol1 is an ABSOLUTE per-voxel log-likelihood gain (`ll - oll < tol1*nm`), not a
    # relative change — every break below uses it, so `tol` means the same thing throughout.
    tol_abs = tol * n_samp
    total_ll = -float("inf")
    ll_per_vox = float("nan")
    bar = (
        _tqdm(total=n_iter, desc="segment EM", leave=True)
        if (_tqdm and verbose and n_iter >= 5)
        else None
    )

    def _bias_ll(corrected_val: Tensor, coef_val: Tensor) -> float:
        """SPM's ``llrb`` = Σ_c (Σ log|bias_c| − ½·T_cᵀ C T_c).

        The Jacobian term is read straight off the corrected data
        (``corrected = exp(bias)·y``) rather than re-evaluating the DCT.
        """
        jac = 0.0
        with torch.no_grad():  # chunked: a full-size log() transient is avoidable here
            for sl in _chunks():
                jac += (
                    (
                        corrected_val[sl].abs().clamp_min(1e-30).log()
                        - intens[sl].abs().clamp_min(1e-30).log()
                    )
                    .sum()
                    .item()
                )
        return jac - sum(bias_penalty_value(coef_val[c], bias_prec).item() for c in range(n_chan))

    for it in range(n_iter):
        # coarse-to-fine: blurred TPM for the first n_coarse iters, then sharp
        cur_log_prior = log_prior_blur if it < n_coarse else log_prior
        prior_raw = warped_prior_raw(cur_log_prior)  # constant this outer iteration
        if means is None:
            # SPM's moment init weights by the RAW warped TPM (`b = buf(z).dat(:,k1)`,
            # spm_preproc8.m:342), not the wp-weighted prior — wp is still uniform here.
            means, covs, mix = _moment_init(intens, prior_raw, cov_prior)
        # warp prior contribution to the objective (constant until the deformation moves)
        llr = -_warp_prior_energy(twarp, warp_reg_gn).item() if fit_warp else 0.0

        # SPM's `for iter1=1:8`: interleave [GMM ×20, bias ×1] until the ll stops improving.
        # Collapsing this to a single round (the behaviour here until 2026-07-22) leaves the
        # intensity model and the bias far from converged at every deformation step, and
        # never re-fits the GMM after the last bias update within an iteration.
        ooll = -float("inf")
        for inner in range(n_inner):
            # corrected is constant across the GMM sub-iterations (coef fixed); build once
            # per inner round, after the bias moved. The float64 view the moment reduction
            # needs is constant too, so cast it here rather than once per chunk per
            # sub-iteration (`gmm_moments`'s own `.to()` is then a no-op).
            corrected = corrected_full(coef)
            corrected64 = corrected.to(torch.float64)
            llrb = _bias_ll(corrected, coef)

            # --- GMM closed-form EM, with wp re-estimated every sub-iteration ---
            oll = -float("inf")
            ll = -float("inf")
            for subit in range(20):
                count = sum1 = sum2 = None
                tissue_mass = torch.zeros(n_tissue, dtype=torch.float64, device=device)
                mgm = torch.zeros(n_tissue, dtype=torch.float64, device=device)
                # accumulate on-device and sync ONCE per sub-iteration (the break needs the
                # value on the host); a per-chunk `.item()` drains the pipeline every chunk.
                data_ll_t = torch.zeros((), dtype=torch.float64, device=device)
                for sl in _chunks():
                    b = prior_raw[sl]
                    resp_c, ll_c = gmm_responsibilities(
                        corrected[sl], weight_prior(b, wp), means, covs, mix, tissue_of
                    )
                    cc, s1, s2 = gmm_moments(corrected64[sl], resp_c)
                    count = cc if count is None else count + cc
                    sum1 = s1 if sum1 is None else sum1 + s1
                    sum2 = s2 if sum2 is None else sum2 + s2
                    tissue_mass.index_add_(0, tissue_of, resp_c.sum(dim=0).to(torch.float64))
                    # mgm_k = Σ_v B(v,k)/(B(v,:)·wp) — the model-EXPECTED tissue mass, over
                    # the unweighted warped TPM. Paired with the observed mass it makes the
                    # wp update self-correcting (see `update_tissue_weights`). Ratios are
                    # O(1) and the reduction is pairwise, so the working dtype is plenty;
                    # only the cross-chunk accumulator is float64.
                    mgm += (b / (b @ wp).clamp_min(1e-30)[:, None]).sum(dim=0).to(torch.float64)
                    data_ll_t += ll_c.sum().to(torch.float64)
                # SPM measures convergence on the full objective (llr + llrb + data), and
                # the ll here is the one BEFORE this sub-iteration's update, as in SPM.
                ll = data_ll_t.item() + llrb + llr
                means, covs, mix = gmm_update(count, sum1, sum2, tissue_of, cov_prior)
                # wp is re-estimated on EVERY sub-iteration (spm_preproc8.m:435). Updating
                # it once per outer iteration leaves the whole GMM/bias/warp running on the
                # previous iteration's weights.
                wp = update_tissue_weights(tissue_mass, mgm, wp_reg, n_tissue).to(wdt)
                if subit > 0 and ll - oll < tol_abs:
                    break
                oll = ll

            if inner > 0 and not (ll - ooll > 2.0 * tol_abs):
                break
            ooll = ll

            # --- bias field ---
            if bias_solver == "gn":
                # SPM's analytic Gauss-Newton on the DCT coefficients (eq. 34): closed-form
                # gradient/Hessian, no autograd graph.
                for _ in range(bias_iters):
                    coef = _bias_gn_sweep(coef)
            else:
                # autograd Adam fallback: maximise data ll + Σ log|bias| Jacobian − penalty.
                coef = coef.detach().requires_grad_(True)
                opt = torch.optim.Adam([coef], lr=bias_lr)
                for _ in range(8):
                    opt.zero_grad()
                    for sl in _chunks():
                        bx, by, bz = _basis(sl)
                        log_bias = [eval_log_bias(coef[c], bx, by, bz) for c in range(n_chan)]
                        corrected_c = torch.stack(
                            [intens[sl, c] * torch.exp(log_bias[c]) for c in range(n_chan)],
                            dim=1,
                        )  # (chunk, n_chan)
                        _, ll_b = gmm_responsibilities(
                            corrected_c, _prior_w(sl), means, covs, mix, tissue_of
                        )
                        # + Σ log|bias|: the multiplicative bias's change-of-variables
                        # Jacobian (corrected = exp(bias)·y). Without it the fit lowers −ll
                        # by shrinking the bias. SPM has it (the +1 in eq.34 / chan.ll).
                        obj = ll_b.sum() + sum(lb.sum() for lb in log_bias)
                        (-obj).backward()  # sum of chunk grads == whole-batch grad
                    torch.stack(
                        [bias_penalty_value(coef[c], bias_prec) for c in range(n_chan)]
                    ).sum().backward()
                    opt.step()
                coef = coef.detach()
                del opt

            # SPM expands lkp to `ngaus` HERE — after the first [GMM, bias] round of the
            # first outer iteration (spm_preproc8.m:646-675), so each tissue is split from
            # its own converged mean/covariance rather than from the pooled moment init.
            # `split_after` lets that wait for more [GMM, bias] rounds: see its docstring
            # entry — the first round runs on the *uncorrected* image, and splitting from
            # the resulting over-broad covariance is what lets a component run away.
            if not split_done and it == 0:
                rounds_done += 1
                if rounds_done >= split_after or inner == n_inner - 1:
                    means, covs, mix, tissue_of = split_gaussians(means, covs, ngaus)
                    split_done = True

        # corrected under the updated bias — constant for the warp + wp steps below
        corrected = corrected_full(coef)

        # collapse the Gaussians into a per-tissue likelihood ONCE (SPM's buf.dat): the
        # deformation + wp steps below only need Σ_t lik_t·prior_t, so the Cholesky/
        # Mahalanobis is paid once per EM iteration, not once per warp sub-iteration.
        if fit_warp or warp_focus > 0:
            with torch.no_grad():
                tissue_lik = torch.empty((n_samp, n_tissue), dtype=wdt, device=device)
                for sl in _chunks():
                    tissue_lik[sl] = gmm_tissue_likelihood(
                        corrected[sl], means, covs, mix, tissue_of, n_tissue
                    ).to(wdt)

        # heavy-to-light regularisation (SPM `scal = 2^max(10-iter,0)` on the bending
        # term): stiffen the warp early — before the GMM/bias have settled — then relax to
        # the target by iter 10, so it can't over-commit to a bad deformation up front.
        anneal = 2.0 ** max(10 - (it + 1), 0) if (warp_anneal and fit_warp) else 1.0
        warp_reg, warp_reg_gn = _scale_warp_reg(anneal)

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

                # Preconditioner (shift·I + L)⁻¹ in the DCT domain. `L` is the stiff,
                # ill-conditioned part of the system and is exactly DCT-diagonalisable, so
                # inverting it leaves CG only the per-node Alpha variation — without this,
                # CG needs O(1000) iterations at SPM's regularisation and a truncated run
                # returns only the high-frequency part of the Newton step. `shift` is the
                # mean of Alpha's diagonal (Alpha_cc = g_c², since Alpha = g·gᵀ), floored
                # so PE-mode's zeroed components stay invertible. `sym_scale = 1/sk²`
                # because we solve in image-voxel units while L lives in node units.
                shift = g_data.pow(2).mean(dim=(0, 1, 2))
                # Floor relative to the largest component, not absolutely: in PE-mode the
                # projected-out components have shift exactly 0, and an absolute floor like
                # 1e-12 then makes 1/(sym+shift) ~1e12 where `sym` also vanishes (bending's
                # affine null space). Round-off in those components blows up and NaNs the
                # whole CG through the shared `rz` reduction.
                shift = shift.clamp_min(shift.max() * 1e-6 + 1e-30)
                _pre = _dct_precond(
                    warp_reg_gn,
                    node_vox,
                    grid_shape,
                    shift,
                    sym_scale=1.0 / (sk_t * sk_t),
                    device=device,
                    dtype=wdt,
                )
                # keep the constrained components identically zero inside the solve too
                precond = (lambda r, _p=_pre: _pe_project(_p(r))) if pe_axis is not None else _pre
                update = _pe_project(_cg_solve(beta, _apply_a, warp_cg_iters, precond))
                base = _warp_objective(twarp, cur_log_prior)
                armijo, improved = 1.0, False
                for _ in range(12):  # backtracking line search (SPM's Armijo: 12 tries)
                    cand = (twarp - armijo * update).detach()
                    if _warp_objective(cand, cur_log_prior) < base:
                        twarp, improved = cand, True
                        break
                    armijo *= 0.75
                if not improved:
                    break  # no downhill step this EM iteration — stop refining the warp
                # SPM breaks out of its `subit=1:3` loop once the deformation stops paying
                # for itself (`spm_preproc8.m:852`); we only had the line-search failure.
                if base - _warp_objective(twarp, cur_log_prior) <= tol_abs:
                    break
        elif fit_warp:
            twarp = twarp.detach().requires_grad_(True)
            opt = torch.optim.Adam([twarp], lr=warp_lr)
            for _ in range(warp_iters):
                opt.zero_grad()
                for sl in _chunks():
                    disp = warp_sign[sl][:, None] * twarp.reshape(-1, 3)[kept_flat[sl]]
                    tpm_coords = apply_affine_pts(coords[sl] + disp, vox2vox)
                    pw = weight_prior(
                        sample_tpm_prior(
                            cur_log_prior, tpm_coords, bg_low, bg_high, kernel=fit_prior_interp
                        ),
                        wp,
                    )
                    # data ll from the precomputed tissue likelihood (no Gaussian eval here)
                    ll_w = torch.log((tissue_lik[sl] * pw).sum(dim=1).clamp_min(1e-40))
                    (-ll_w.sum()).backward()
                pen = _warp_prior_energy(twarp, warp_reg)  # smoothness prior, once
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
                        disp = warp_sign[sl][:, None] * twarp.detach().reshape(-1, 3)[kept_flat[sl]]
                        tpm_coords = apply_affine_pts(coords[sl] + disp, vox2vox)
                        pw = weight_prior(
                            sample_tpm_prior(
                                cur_log_prior, tpm_coords, bg_low, bg_high, kernel=fit_prior_interp
                            ),
                            wp,
                        )
                        # per-tissue posterior from the precomputed likelihood × warped prior
                        tp = tissue_lik[sl] * pw
                        tpost = tp / tp.sum(dim=1, keepdim=True).clamp_min(1e-40)
                        # per-node prior/posterior overlap; low overlap = the prior sits in
                        # the wrong place here (intensity says one tissue, the TPM another)
                        misfit[sl] = (1.0 - (tpost * pw).sum(dim=1)).clamp(0.0, 1.0)
                    thr = _quantile(misfit, focus_quantile)
                    rel = ((misfit - thr) / (misfit.max() - thr).clamp_min(1e-6)).clamp(0.0, 1.0)
                    # scatter the per-sample relaxation onto grid nodes (dual-echo mode maps
                    # two samples to a node; last write wins — fine for this niche Adam knob)
                    mf = torch.zeros(n_grid, dtype=wdt, device=device)
                    mf.scatter_(0, kept_flat, (rel * warp_focus).to(wdt))
                    # smooth the relaxation into a coherent blob (no single-node spikes)
                    kw = mf.reshape(grid_shape)[..., None].repeat(1, 1, 1, 3)
                    keep_weight = _smooth_field(kw, warp_smooth)[..., 0].clamp(0.0, 1.0)
            # Sobolev smoothing: keep the dense field smooth (no per-node speckle),
            # except where keep_weight relaxes it
            twarp = _smooth_field(twarp.detach(), warp_smooth, keep_weight)

        # --- convergence ll ---
        # `wp` is NOT updated here: SPM re-estimates it inside the GMM sub-iterations
        # (above), not after the deformation. This block only re-scores the model under the
        # warp that just moved, so the outer stopping rule sees the current objective.
        with torch.no_grad():
            data_ll = 0.0
            for sl in _chunks():
                disp = warp_sign[sl][:, None] * twarp.reshape(-1, 3)[kept_flat[sl]]
                tpm_coords = apply_affine_pts(coords[sl] + disp, vox2vox)
                pw = weight_prior(
                    sample_tpm_prior(
                        cur_log_prior, tpm_coords, bg_low, bg_high, kernel=fit_prior_interp
                    ),
                    wp,
                )
                _, ll_c = gmm_responsibilities(corrected[sl], pw, means, covs, mix, tissue_of)
                data_ll += ll_c.sum().item()
            # SPM's ll = llr + llrb + data: data ll + bias Jacobian − bias smoothness prior
            # − warp prior. Both priors are already the ½·xᵀCx / ½·uᵀLu forms.
            warp_pen = _warp_prior_energy(twarp, warp_reg_gn).item() if fit_warp else 0.0
            total_ll = data_ll + _bias_ll(corrected, coef) - warp_pen

        ll_per_vox = total_ll / n_samp
        if bar is not None:
            bar.set_postfix(ll_vox=f"{ll_per_vox:+.4f}")
            bar.update(1)
        elif verbose:
            print(f"iter {it + 1:2d}/{n_iter}  ll/vox {ll_per_vox:+.4f}")
        # SPM: `if iter>=10 && ~((ll-ooll)>2*tol1*nm), break` (spm_preproc8.m:859). The
        # reference for the gain is `ooll`, which SPM last wrote *inside* this same outer
        # iteration's `iter1` loop (`:516`) — so the test asks "did the deformation improve
        # on the state before the last bias update", not "did this iteration beat the last
        # one". The previous-iteration form used here until 2026-07-30 is a strictly larger
        # gain, so it kept iterating past SPM's stopping point. The `min_iter` floor is
        # load-bearing, not defensive — the heavy-to-light schedule only relaxes the bending
        # term to its target at iteration 10, so an earlier exit leaves the deformation
        # stiffened by up to 2¹⁰ and effectively unfitted. Also don't stop inside the coarse
        # (blurred-TPM) phase: the ll there is on a different objective.
        if it + 1 >= min_iter and it >= n_coarse and not (total_ll - ooll > 2.0 * tol_abs):
            break
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
        "tpm_shape": tpm_shape,
        "tpm_vox": tpm_vox,
        "tpm_bottom_mm": tpm_bottom_mm,
        "tpm_fov_mm": tpm_fov_mm,
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
    dura_clean: float = 0.0,
    dura_method: str = "geodesic",
    prior_kernel: str = "bspline2",
    warp_kernel: str = "bspline3",
    mask_outside_tpm: bool = True,
    gm_class: tuple[int, ...] = (0,),
    wm_class: tuple[int, ...] = (1,),
    csf_class: tuple[int, ...] = (2,),
    background_class: int = -1,
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

    ``mask_outside_tpm`` (default on) re-applies the fit's own field-of-view rule
    (:func:`tpm_coverage_mask`, from ``tpm_bottom_mm``/``tpm_fov_mm`` recorded in ``fit``)
    to the **output**. Without it the fit correctly refuses to *learn* from the neck while
    the write-out happily *classifies* it: those voxels map outside the template, pick up
    whatever its edge planes contain, and come back carrying brain-tissue probability far
    from any brain. Voxels that fail the test are assigned wholly to ``background_class``
    (default ``-1``, the last class — see ``load_tpm``'s auto background).

    ``gm_class``/``wm_class``/``csf_class`` tell the morphological post-passes which
    channels play which role, so the cleanups are not hardwired to SPM's ``c1/c2/c3``
    ordering. Each takes **several** indices — a template with two GM classes (say
    cortical and subcortical) passes ``gm_class=(0, 6)`` and they are summed for the
    purposes of the cleanup, then the resulting mask is applied to each member channel.
    For CSF, pass only the class that represents the outer/subarachnoid shell; that is the
    one :func:`clean_gwc` and :func:`dura_cleanup` reason about.

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
    # prefilter once, not per chunk (the samp grid is small; the chunks are not)
    warp_coeffs = (
        bspline3_coefficients(twarp.permute(3, 0, 1, 2).contiguous())
        if warp_kernel != "linear"
        else twarp
    )
    tpm_shape = fit.get("tpm_shape")
    bg_idx = background_class % n_tissue
    if mask_outside_tpm and tpm_shape is None and verbose:
        print("  note: fit predates the TPM field-of-view record — output masking skipped")
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
        if warp_kernel == "linear":
            disp = torch.stack(
                [
                    trilinear_interpolate(twarp_c[c], gxp, gyp, gzp).to(torch.float64)
                    for c in range(3)
                ],
                dim=1,
            )
        else:  # SPM: one degree-3 B-spline prefilter, reused across every chunk
            disp = bspline_interpolate_multi(warp_coeffs, gxp, gyp, gzp, degree=3).to(torch.float64)
        coords = torch.stack([x, y, z], dim=1) + disp
        tpm_c = apply_affine_pts(coords, vox2vox)
        prior_w = weight_prior(
            sample_tpm_prior(log_prior, tpm_c, bg_low, bg_high, kernel=prior_kernel), wp
        )

        resp, _ = gmm_responsibilities(corrected, prior_w, means, covs, mix, tissue_of)
        # collapse Gaussians → per-tissue posterior in one scatter (no per-Gaussian loop)
        tpost = torch.zeros((idx.numel(), n_tissue), dtype=torch.float64, device=device)
        tpost.index_add_(1, tissue_of, resp)
        if mask_outside_tpm and tpm_shape is not None:
            # The fit refused to LEARN from these voxels; don't let the write-out CLASSIFY
            # them either. Outside the template the prior is whatever the edge planes
            # happen to hold, so intensity alone decides — which is how brain-tissue
            # probability ends up in the neck.
            inside = tpm_coverage_mask(
                tpm_c,
                tpm_shape,
                fit["tpm_vox"],
                bottom_mm=fit.get("tpm_bottom_mm", 0.0),
                fov_mm=fit.get("tpm_fov_mm"),
            )
            tpost[~inside] = 0.0
            tpost[~inside, bg_idx] = 1.0

        posteriors[:, start : start + idx.numel()] = tpost.T.to(torch.float32).cpu()
        corrected_out[:, start : start + idx.numel()] = corrected.T.to(torch.float32).cpu()
        bias_out[:, start : start + idx.numel()] = bias.to(torch.float32).cpu()

    posteriors = posteriors.reshape(n_tissue, nz, ny, nx)
    precleanup = posteriors.clone() if save_precleanup else None
    if mrf > 0 or cleanup > 0 or debridge > 0 or dura_clean > 0:
        posteriors = posteriors.to(device)
        if mrf > 0:
            posteriors = mrf_cleanup(posteriors, mrf, fit["vox"])
        if debridge > 0:  # strip thin dura bridges before the WM-seeded brain extraction
            posteriors = debridge_gm(posteriors, radius=debridge, gm_index=gm_class[0])
        if dura_clean > 0:  # dura removal (WM-geodesic front, or CSF-sheet-gap fill)
            posteriors = dura_cleanup(
                posteriors,
                fit["vox"],
                max_thick_mm=dura_clean,
                method=dura_method,
                gm_index=gm_class[0],
                wm_index=wm_class[0],
                csf_index=csf_class[0],
            )
        if cleanup > 0:
            posteriors = clean_gwc(
                posteriors,
                level=cleanup,
                gm_class=gm_class,
                wm_class=wm_class,
                csf_class=csf_class,
            )
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

    Updated **red-black in place**, as ``spm_mrf.c`` does — see the comment in the body.
    (One deliberate non-replication: SPM carries ``p``/``q`` as ``uint8`` and re-quantises
    every sweep. That is a storage decision in the reference, not part of the model, so
    this runs in float.)

    Args:
        posteriors: ``(n_tissue, nz, ny, nx)`` GMM data-term posteriors.
        mrf: connectivity strength (SPM default 1); 0 disables.
        vox: ``(vx, vy, vz)`` mm voxel sizes.
        n_iter: mean-field iterations (SPM uses 10, each a full red + black sweep).

    Returns:
        ``(n_tissue, nz, ny, nx)`` cleaned posteriors (each voxel sums to 1).
    """
    if mrf <= 0:
        return posteriors
    data = posteriors.to(torch.float32)
    # SPM starts its state at zero (`P = zeros(...,'uint8')` in `clean_write_tissues`), so
    # the first half-sweep sees no neighbour mass and reduces to the data term.
    field = torch.zeros_like(data)
    # posteriors are (tissue, z, y, x): axes 1,2,3 ↔ z,y,x → weights 1/vox_z², /vox_y², /vox_x²
    w = [1.0 / vox[2] ** 2, 1.0 / vox[1] ** 2, 1.0 / vox[0] ** 2]

    # SPM's `spm_mrf` is a RED-BLACK (checkerboard) Gauss-Seidel sweep updated IN PLACE, not
    # the synchronous Jacobi iteration used here until 2026-07-30. Each call makes two
    # half-sweeps (`for it=0; it<2`), and decoding its `i0start/i1start/i2start` loop bounds
    # shows they are exactly the two parities of `(x+y+z) % 2`. The distinction is not
    # cosmetic: Gauss-Seidel propagates each update to its neighbours within the same
    # iteration, so it converges roughly twice as fast as Jacobi and does not exhibit the
    # two-colour oscillation Jacobi can settle into on a Potts field.
    zz, yy, xx = torch.meshgrid(
        *(torch.arange(n, device=data.device) for n in data.shape[1:]), indexing="ij"
    )
    parity = ((xx + yy + zz) % 2).to(torch.bool)[None]  # (1, nz, ny, nx), broadcast over tissue

    for _ in range(n_iter):
        for colour in (False, True):
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
            upd = data * torch.exp(mrf * neigh)
            upd = upd / upd.sum(dim=0, keepdim=True).clamp_min(1e-20)
            # write back only this colour; the next half-sweep then reads the new values
            field = torch.where(parity == colour, upd, field)
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


def _role_maps(
    out: Tensor, gm_class: tuple[int, ...], wm_class: tuple[int, ...], csf_class: tuple[int, ...]
) -> tuple[Tensor, Tensor, Tensor]:
    """Sum the channels playing each role → ``(gm, wm, csf)`` maps.

    A role may span several classes (two GM classes, say), so the morphology reasons about
    their total and the resulting mask is later applied to each member channel.
    """
    return (
        out[list(gm_class)].sum(dim=0),
        out[list(wm_class)].sum(dim=0),
        out[list(csf_class)].sum(dim=0),
    )


def clean_gwc(
    posteriors: Tensor,
    level: int = 1,
    *,
    gm_class: tuple[int, ...] = (0,),
    wm_class: tuple[int, ...] = (1,),
    csf_class: tuple[int, ...] = (2,),
) -> Tensor:
    """Ad-hoc morphological brain cleanup of GM/WM/CSF (SPM ``clean_gwc``).

    Grows a brain mask from the WM seed by conditional dilation through connected
    GM+WM (32 iterations of threshold → keep GM+WM mass → smooth), repeats including
    CSF, then zeroes GM/WM outside the GM+WM mask and CSF outside the GM+WM+CSF mask
    and renormalises. Strips dura/skull/eyeball voxels misclassified as brain tissue —
    the classic reason a GM or CSF map is unusable as a mask.

    ``gm_class``/``wm_class``/``csf_class`` name the channels playing each role rather
    than hardwiring SPM's ``c1/c2/c3``; each accepts several indices, which are summed for
    the morphology and then masked individually. For CSF give the **outer/subarachnoid**
    shell only — that is the compartment the conditional dilation is reasoning about.
    ``level`` 1 (default) or 2 (stricter dilation threshold 0.2 vs 0.15).

    Args:
        posteriors: ``(n_tissue, nz, ny, nx)`` tissue posteriors (rows sum to 1).
        level: cleanup aggressiveness (1 or 2).
        gm_class, wm_class, csf_class: 0-based channel indices per role.

    Returns:
        ``(n_tissue, nz, ny, nx)`` cleaned posteriors (rows sum to 1).
    """
    roles = (*gm_class, *wm_class, *csf_class)
    if posteriors.shape[0] <= len(roles):
        return posteriors  # SPM: "Cleanup not done" — needs the roles plus something else
    dtype = torch.float32
    out = posteriors.to(dtype).clone()
    gm, wm, csf = _role_maps(out, gm_class, wm_class, csf_class)
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
    # apply each role's mask to every channel playing that role, not to the summed map
    for k in (*gm_class, *wm_class):
        out[k] = out[k] * brain
    for k in csf_class:
        out[k] = out[k] * csf_brain
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


def _flanked_by(mask: Tensor, k: int) -> Tensor:
    """True where ``mask`` is present within ``k`` voxels on **both** sides along at least one
    axis — the "high … gap … high" sheet-continuity test (directional, not isotropic)."""
    flank = torch.zeros_like(mask)
    for axis in (0, 1, 2):
        ahead = torch.zeros_like(mask)
        behind = torch.zeros_like(mask)
        for j in range(1, k + 1):
            ahead = ahead | torch.roll(mask, -j, dims=axis)
            behind = behind | torch.roll(mask, j, dims=axis)
        flank = flank | (ahead & behind)
    return flank


def dura_cleanup(
    posteriors: Tensor,
    vox: tuple[float, float, float],
    *,
    max_thick_mm: float = 6.0,
    method: str = "geodesic",
    csf_barrier: float = 0.5,
    gm_th: float = 0.1,
    wm_seed: float = 0.5,
    flank_k: int = 3,
    csf_present: float = 0.05,
    gm_index: int = 0,
    wm_index: int = 1,
    csf_index: int = 2,
) -> Tensor:
    """Demote dura misclassified as GM. Two methods, both restricted to the **outer shell**
    (far from WM) so legitimate sulcal CSF/cortex near WM is untouched:

    - ``"geodesic"`` (default, SPM-validated) — grow a front from WM through brain tissue
      (``GM+WM > gm_th``) **blocked by CSF** (``CSF < csf_barrier``), out to ``max_thick_mm``;
      demote GM the front can't reach. A gyral crown stays (tissue-connected to WM through the
      gyrus); only GM reachable solely by crossing a CSF moat is removed. Far less destructive
      than :func:`debridge_gm`'s blind opening. On the reference subject this brings GM >6 mm
      from WM to SPM's own 2.8 % (Dice-vs-SPM 0.950→0.951).
    - ``"csf_gap"`` — the **inverted** view: the dura is a *hole* in the outer CSF sheet, so
      ``CSF`` reads "present … gap … present". A GM voxel in the outer shell whose CSF is
      essentially absent but is **flanked on opposite sides within ``flank_k`` voxels by CSF
      present at a light threshold (>csf_present)** is that gap → reassigned to CSF. The sheet
      is thresholded *lightly* on purpose: the subarachnoid CSF wrapping the dura is
      thin/partial-volumed and often only reads ~0.02–0.1, so a confident (>0.4) sheet test
      finds nothing to flank and the dura — itself extremely-low-probability CSF — survives as a
      hole. Directional flanking (not an isotropic morphological closing) spares a one-sided
      concavity. More robust than the wavefront when the inner subarachnoid CSF is
      thin/partial-volumed (a weak barrier the front would leak through, keeping the dura). The
      outer shell is defined geodesically from WM through the *whole* brain (``GM+WM+CSF``) out
      to ``max_thick_mm`` — cortex is nearer than that, dura beyond.

    ``max_thick_mm<=0`` disables. SPM order ``GM, WM, CSF, …`` assumed. Returns
    ``(n_tissue, nz, ny, nx)`` posteriors with dura demoted (rows sum to 1).
    """
    import math

    if max_thick_mm <= 0 or posteriors.shape[0] <= max(gm_index, wm_index, csf_index):
        return posteriors
    gm, wm, csf = posteriors[gm_index], posteriors[wm_index], posteriors[csf_index]
    n_iter = max(1, int(math.ceil(max_thick_mm / min(vox))))
    out = posteriors.to(torch.float32).clone()

    if method == "geodesic":
        allowed = ((gm + wm) > gm_th) & (csf < csf_barrier)  # brain tissue, CSF moat blocks
        reach = wm > wm_seed
        for _ in range(n_iter):
            reach = reach | (_binary_dilate(reach, 1) & allowed)  # advance the front one voxel
        dura = (gm > 0.5) & ~reach
        out[gm_index] = torch.where(dura, torch.zeros_like(gm), gm)  # redistribute on renorm
    elif method == "csf_gap":
        # outer shell = far from WM geodesically through the WHOLE brain (not CSF-blocked)
        brain = (gm + wm + csf) > gm_th
        near = wm > wm_seed
        for _ in range(n_iter):
            near = near | (_binary_dilate(near, 1) & brain)
        # light threshold: the sheet wrapping the dura is thin/partial-volumed (~0.02–0.1), so a
        # confident (>0.4) test would see no sheet to flank and leave the dura hole unfilled.
        present = csf > csf_present
        gap = _flanked_by(present, flank_k) & ~present & (gm > 0.5) & ~near  # a hole in the sheet
        out[csf_index] = csf + torch.where(gap, gm, torch.zeros_like(gm))  # the gap IS CSF
        out[gm_index] = torch.where(gap, torch.zeros_like(gm), gm)
    else:
        raise ValueError(f"dura_cleanup method must be 'geodesic' or 'csf_gap', got {method!r}")

    return out / out.sum(dim=0, keepdim=True).clamp_min(1e-20)


# ---------------------------------------------------------------------------
# Autobox — crop the all-zero margin before the fit, restore full dims on save.
# ---------------------------------------------------------------------------


def autobox_bounds(
    volume: Tensor | list[Tensor], pad: int = 4
) -> tuple[tuple[slice, slice, slice], tuple[int, int, int]]:
    """Bounding box of the finite non-zero region across one or more aligned volumes.

    Trims the all-zero planes from each face of the ``(nz, ny, nx)`` grid (union over
    channels) and keeps ``pad`` voxels of safe margin, clamped to the volume. Returns the
    crop ``slices`` (array axis order ``z, y, x``) and the ``(x0, y0, z0)`` voxel offset
    (affine / world axis order) for the affine shift (:func:`crop_affine`, which keeps world
    alignment exact). All-zero input → the full volume.

    Note MR background noise is **not** zero, so on raw data this trims nothing — mask the
    head first (e.g. ``processing/mask.automask``, zeroing outside a padded mask) so there is
    an all-zero margin to trim. The EM fit already excludes zero voxels, so the crop itself
    doesn't change the estimate; the masking that precedes it does (removes background noise).
    """
    vols = list(volume) if isinstance(volume, (list, tuple)) else [volume]
    nz, ny, nx = vols[0].shape
    mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=vols[0].device)
    for v in vols:
        mask |= torch.isfinite(v) & (v != 0)
    if not bool(mask.any()):
        return (slice(0, nz), slice(0, ny), slice(0, nx)), (0, 0, 0)
    zsel = mask.any(dim=2).any(dim=1)  # (nz,)
    ysel = mask.any(dim=2).any(dim=0)  # (ny,)
    xsel = mask.any(dim=1).any(dim=0)  # (nx,)

    def _range(sel: Tensor, n: int) -> tuple[int, int]:
        idx = torch.nonzero(sel, as_tuple=False).flatten()
        return max(0, int(idx[0]) - pad), min(n, int(idx[-1]) + 1 + pad)

    z0, z1 = _range(zsel, nz)
    y0, y1 = _range(ysel, ny)
    x0, x1 = _range(xsel, nx)
    return (slice(z0, z1), slice(y0, y1), slice(x0, x1)), (x0, y0, z0)


def crop_affine(affine: Tensor, offset_xyz: tuple[int, int, int]) -> Tensor:
    """Shift a voxel-``(x,y,z)``→world affine so the cropped volume's voxel ``(0,0,0)``
    keeps its original world coordinate (``offset_xyz`` = the crop start in x,y,z voxel
    order, from :func:`autobox_bounds`). Rotation/scale unchanged; only the origin moves."""
    new = affine.clone()
    off = torch.tensor(offset_xyz, dtype=affine.dtype, device=affine.device)
    new[:3, 3] = affine[:3, 3] + affine[:3, :3] @ off
    return new


def embed_in_full(
    arr: Tensor,
    full_shape: tuple[int, int, int],
    slices: tuple[slice, slice, slice],
    *,
    fill: float = 0.0,
) -> Tensor:
    """Place a cropped array back into a full-size volume (inverse of the autobox crop).

    ``arr``'s **last three** axes are the spatial ``(nz', ny', nx')`` grid (so ``(nz,ny,nx)``,
    ``(K, nz,ny,nx)``, and single warp components all work); any leading axes are kept. The
    out-of-box margin is set to ``fill`` (0 for tissue/displacement maps, 1 for a bias field).
    """
    lead = tuple(arr.shape[:-3])
    out = torch.full((*lead, *full_shape), fill, dtype=arr.dtype, device=arr.device)
    out[(..., *slices)] = arr
    return out


def full_resolution_warp(
    fit: dict,
    shape: tuple[int, int, int],
    *,
    kernel: str = "bspline3",
    device: torch.device | str = "cpu",
) -> Tensor:
    """Upsample the fitted deformation to the full subject grid (voxel-unit disp).

    The dense ``twarp`` is stored on the subsampled grid; this upsamples it to every
    subject voxel and returns ``(nz, ny, nx, 3)`` displacements in subject voxel units —
    the ``(x, y, z)`` components ready for ``io.save_warp_field(..., units="mm")``, so the
    subject-space deformation drops straight into the composable ffs_nwarp chain
    (alongside the affine).

    ``kernel="bspline3"`` (default) is SPM's: ``spm_preproc_write8`` builds
    ``spm_bsplinc(Twarp, [3 3 3 0 0 0])`` once and its ``defs`` samples with the matching
    degree-3 basis. That matters more than it looks — at ``samp 3`` on 0.7 mm data the
    stride is ``sk=4``, so this is a **4x upsample** of the field that positions every
    tissue prior, and trilinear (``kernel="linear"``, the behaviour here until 2026-07-22)
    is ``C⁰`` with a piecewise-constant gradient: it facets the deformation on the
    coarse-grid lattice and systematically undershoots between nodes. Cheap to evaluate
    either way — the prefilter runs once on the small ``samp`` grid.
    """
    device = torch.device(device)
    nz, ny, nx = shape
    sk = fit["sk"]
    twarp = fit["twarp"].to(device)

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float64),
        torch.arange(ny, device=device, dtype=torch.float64),
        torch.arange(nx, device=device, dtype=torch.float64),
        indexing="ij",
    )
    gxp, gyp, gzp = xx.reshape(-1) / sk[0], yy.reshape(-1) / sk[1], zz.reshape(-1) / sk[2]
    if kernel == "linear":
        from .interp import trilinear_interpolate

        comps = [
            trilinear_interpolate(twarp[..., c].contiguous(), gxp, gyp, gzp).reshape(nz, ny, nx)
            for c in range(3)
        ]
        return torch.stack(comps, dim=-1)
    # (3, gz, gy, gx) coefficient stack — all three components share the sample locations
    coeffs = bspline3_coefficients(twarp.permute(3, 0, 1, 2).contiguous())
    disp = bspline_interpolate_multi(coeffs, gxp, gyp, gzp, degree=3)  # (n_vox, 3)
    return disp.reshape(nz, ny, nx, 3)


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

    device = torch.device(device) if device is not None else volume.device
    src = undistort_input(volume, fit, device=device) if use_warp else volume.to(device)
    tpm_to_input = torch.linalg.inv(fit["vox2vox"].to(torch.float64)).to(device)
    x, y, z = _fullres_grid(out_shape, device)
    coords = apply_affine_pts(torch.stack([x, y, z], dim=1), tpm_to_input)  # input voxel coords
    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
    return _image_resample(src, xs, ys, zs, kernel).reshape(out_shape)


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
    out = torch.stack(
        [
            _image_resample(vol4d[t], xs, ys, zs, kernel).reshape(out_shape)
            for t in range(vol4d.shape[0])
        ],
        dim=0,
    )
    return out[0] if source.ndim == 3 else out


# SPM's default 6-class TPM order (TPM.nii); used to label the diagnostic plot when the
# caller doesn't supply names. Extra/fewer classes fall back to generic "class N".
_SPM_TISSUE_NAMES = ["GM", "WM", "CSF", "bone", "soft tissue", "air/background"]


def _second_peak_height(counts) -> float:
    """Height of the tallest histogram peak that is *not* the dominant (air) spike.

    Walk down from the global-max bin to its local valleys on both sides, mask out
    that whole basin, and return the tallest bin left over. This is the y-scale that
    lets the tissue peaks fill the panel while the air/background spike runs off the
    top, without picking an arbitrary window width.
    """
    import numpy as np

    counts = np.asarray(counts, dtype=float)
    if counts.size == 0:
        return 0.0
    # smooth before walking the basin: raw histogram bins are noisy, so a strict
    # monotonic descent halts at the first tiny uptick and never exits the air spike.
    win = max(3, counts.size // 40)
    kernel = np.ones(win) / win
    smooth = np.convolve(counts, kernel, mode="same")
    i_max = int(np.argmax(smooth))
    lo = i_max
    while lo > 0 and smooth[lo - 1] <= smooth[lo]:
        lo -= 1
    hi = i_max
    while hi < smooth.size - 1 and smooth[hi + 1] <= smooth[hi]:
        hi += 1
    rest = np.concatenate([counts[:lo], counts[hi + 1 :]])  # tallest raw bin outside the basin
    return float(rest.max()) if rest.size else float(counts[i_max])


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

    Two rows of panels, one column per input channel. Grey bars are the
    bias-corrected data; each coloured line is a tissue's fitted Gaussian-mixture
    **marginal** in that channel (``Σ_{k∈t} w_k · N(x; μ_ck, Σ_cc,k)``, weight
    ``w_k`` = that tissue's posterior mass × the within-tissue mixing proportion);
    the black line is the total model. Where the grey histogram rises above the
    black line the model under-explains that intensity — a quick read on *what got
    missed*.

    The **top row** is scaled to the full data maximum (the air/background spike
    fills the panel, the whole model visible in context). The **bottom row** is the
    same plot scaled to the *second* peak height, so the air spike runs off the top
    and the tissue peaks — GM/WM/CSF — fill the panel legibly. One figure, both
    views, all the information.

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

    # two rows: top scaled to the full max, bottom scaled to the 2nd peak (air off-scale)
    fig, axes = plt.subplots(2, n_chan, figsize=(6.4 * n_chan, 8.4), squeeze=False)
    n_fg = int(flat.sum().item())
    xlabel = "bias-corrected intensity"
    row_titles = ("scaled to full max", "scaled to 2nd peak (air off-scale)")

    def draw(ax, c):
        """Draw the histogram + fitted curves for channel ``c``; return (counts, curves)."""
        vals = corr[c].reshape(-1)[flat].numpy()
        lo, hi = np.percentile(vals, [0.2, 99.8])  # robust range, ignore outliers
        if hi <= lo:
            hi = lo + 1.0
        counts, edges = np.histogram(vals, bins=n_bins, range=(lo, hi))
        centres = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1] - edges[0]

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
        ax.set_xlim(lo, hi)
        return counts, total_curve

    for c in range(n_chan):
        counts, total_curve = draw(axes[0, c], c)
        _, _ = draw(axes[1, c], c)  # same plot, different y-scale set below

        # top row: whole picture, air spike included
        full = max(float(counts.max()), float(total_curve.max()))
        axes[0, c].set_ylim(0, full * 1.05 if full > 0 else None)
        # bottom row: clip to the second peak so the tissue structure fills the panel.
        # exclude the air basin from the *model* curve too — the air/background class is
        # itself a fitted Gaussian, so total_curve.max() would re-inflate the limit.
        second = max(_second_peak_height(counts), _second_peak_height(total_curve))
        axes[1, c].set_ylim(0, second * 1.25 if second > 0 else None)

        for r in range(2):
            ax = axes[r, c]
            ax.set_ylabel("voxel count")
            ax.set_xlabel(f"{xlabel} (channel {c + 1})" if n_chan > 1 else xlabel)
            ax.set_title(row_titles[r], fontsize=9)
            ax.legend(fontsize=8, framealpha=0.9)
    fig.suptitle("Segmentation intensity fit — data vs fitted tissue mixture")
    fig.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig
