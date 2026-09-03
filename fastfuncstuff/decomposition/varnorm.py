"""Voxel-wise variance normalisation — dividing by *noise*, not by total variance.

Paper-derived. No implementation is a reference for this module; see
``../fmri_wiki/notes/FSL clean-room policy.md``.

References
----------
- Beckmann, C.F. & Smith, S.M. (2004). *Probabilistic independent component analysis for
  functional magnetic resonance imaging*. IEEE TMI 23(2):137-152, §II-B. The PICA
  generative model is ``x = As + mu + noise`` with **voxel-wise** noise variance; the
  whitening that the model requires is only valid once that variance has been equalised.

Why not just divide by each voxel's standard deviation
------------------------------------------------------
The quantity the model wants equalised is the *noise* standard deviation. A voxel's total
temporal standard deviation is signal plus noise, so dividing by it penalises exactly the
voxels with the most signal -- strong, clean responses get scaled down toward the noise
floor and the components that should be easiest to find are made harder to find.

So estimate the noise instead: project the data onto a low-rank signal subspace, and take
the standard deviation of what is left. Two consequences worth stating, because they are
what makes this defensible rather than arbitrary:

- **The residual has fewer degrees of freedom than the data.** Removing ``r`` components
  from ``T`` timepoints leaves ``T - r - 1`` (the extra one for the mean). Taking a plain
  standard deviation of the residual therefore *underestimates* the noise by
  ``sqrt((T - r - 1) / (T - 1))``, which at a typical ``T=200, r=30`` is a 7% bias --
  small, systematic, and free to correct. :func:`noise_std_map` corrects it.
- **The rank is explicit.** A data-dependent rule (shrink whitened coefficients below some
  threshold, keep what survives) makes the effective rank differ per voxel and per
  dataset, which is untraceable and cannot carry a degrees-of-freedom correction at all.
  A fixed rank with an honest correction is both simpler and better founded.

Assumption: many more voxels than timepoints
--------------------------------------------
The signal subspace is estimated from the data, and the *top* ``r`` principal directions
of a finite sample capture more variance than an arbitrary ``r`` directions would --
Marchenko-Pastur spread. So the residual is slightly smaller than the degrees-of-freedom
count alone predicts, and the noise estimate is correspondingly low. The excess shrinks as
``T/V`` does, because the sample covariance of noise approaches a multiple of the identity
and the "top" directions stop being special:

===========================  ==========================
``T/V``                      estimated / true noise sd
===========================  ==========================
0.5   (V=400,   T=200)       0.88
0.1   (V=2000,  T=200)       0.95
0.02  (V=10000, T=200)       0.98
0.004 (V=50000, T=200)       0.99
===========================  ==========================

fMRI lives at the bottom of that table -- hundreds of timepoints, tens or hundreds of
thousands of voxels -- so the residual bias is under a percent and no further correction is
warranted. The bias is uniform across voxels in any case, so it scales the whole dataset
rather than distorting one voxel relative to another, which is what would actually matter
here. Do not reuse this function in a regime with few voxels without revisiting that.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "DEFAULT_SIGNAL_RANK",
    "apply_noise_std_map",
    "noise_std_map",
    "variance_normalize",
]

DEFAULT_SIGNAL_RANK = 30
"""Rank of the signal subspace removed before measuring the noise.

**This is a real quality/count trade-off, not a nuisance parameter.** An earlier version of
this docstring claimed it "does not need to be right, only generous, since the
degrees-of-freedom correction handles whatever it removes". Both halves of that are wrong,
and in opposite directions:

**Everything below was measured on UNSMOOTHED 3 mm data, and the effect is much smaller
once the data is smoothed** -- see the blur contrast at the end. Do not carry these numbers
over to a conventionally smoothed pipeline.

*It sets the model order.* At fixed effective sample size on ds005165, sweeping the rank
2/5/10/20/30/50/80/120 moves the selected order 62/64/69/73/76/78/81/92 (rest run 1) and
62/63/68/72/76/77/86/93 (localizer run 1). A generous rank absorbs real signal in high-SNR
voxels, so their estimated noise comes out too small, dividing by it over-amplifies exactly
the voxels carrying signal, and more eigenvalues clear the floor. The DoF correction
compensates for the dimensions removed, not for the signal absorbed. The relation is
monotone increasing, so iterating ``rank := selected order`` does not converge.

*But a generous rank is also better.* Judged by cross-run component reproducibility --
5 rest runs, 10 pairs, k held fixed at 62 so the comparison is quality not count --
components reproducing at ``|r| >= 0.25`` go 11.1 / 14.6 / 18.0 / 21.2 / 21.6 at rank
2 / 10 / 30 / 80 / 120, and the mean of the top-20 matches goes .268 / .295 / .322 / .352 /
.357. This is not correlation inflation: the unmatched-pair null barely moves (.0175 ->
.0209) while the matched-minus-null excess grows .251 -> .336. The ratio peaks near rank 80.

So the default of 30 is *conservative* by the reproducibility criterion, not too generous.

Do not replace this with a plain total-stdev divide to make the count match a reference.
That was measured too, and it is much worse: at the same k it reproduces roughly half as
many components (10.1 vs 18.0 at ``|r| >= 0.25``, top-20 .260 vs .322), which is exactly
the failure this module's header predicts -- dividing by total SD penalises the voxels with
the most signal. A rank-free high-frequency spectral estimator sits in between (15.1,
.308) and is more knob-stable than this one but still not knob-free.

*Smoothing washes most of this out.* With ``-do_blur 5`` on the same runs the reproducibility
sweep gives 23.9 / 25.0 / 26.5 / 26.9 at rank 2 / 30 / 80 / 120 -- a 13% span where the
unsmoothed data gave 95% -- and the selected order moves 30->51 instead of 62->92. Blurring
raises per-voxel SNR, so the noise estimate stops being what limits the decomposition. The
same happens to the choice of estimator: total-stdev is 44% worse than the low-rank one
unsmoothed (10.1 vs 18.0) but only 11% worse at 5 mm (22.2 vs 25.0), with HF-spectral tied
or better. **These are low-smoothness phenomena.** On smoothed data leave the default alone.

See ``../fmri_wiki/concepts/ICA noise normalisation.md``.
"""


def _signal_rank(n_time: int, requested: int | None) -> int:
    r = DEFAULT_SIGNAL_RANK if requested is None else int(requested)
    # Leave at least a few degrees of freedom in the residual.
    return max(1, min(r, max(1, n_time - 3)))


@torch.inference_mode()
def noise_std_map(
    data_vox_t: Tensor,
    signal_rank: int | None = None,
    *,
    const_threshold: float = 1e-6,
) -> tuple[Tensor, Tensor, int]:
    """Per-voxel noise standard deviation, from the residual of a low-rank fit.

    Returns ``(noise_std, const_mask, n_constant)``. ``noise_std`` is bias-corrected for
    the degrees of freedom spent on the signal subspace. ``const_mask`` marks voxels with
    no usable variance -- these are excluded rather than normalised, because dividing a
    flat time series by its own near-zero spread manufactures a unit-variance "signal"
    out of nothing and the mixture model downstream will happily fit it. That was a real
    bug; see ``../fmri_wiki/concepts/Constant voxels break the mixture model.md``.

    ``data_vox_t`` is ``(n_vox, n_time)``. Computed in float64 where the device supports
    it, since the covariance eigendecomposition is the precision-sensitive step.
    """
    x_t = data_vox_t.T  # (T, V)
    n_time, n_vox = int(x_t.shape[0]), int(x_t.shape[1])

    total_std = torch.std(x_t, dim=0, unbiased=True)
    if n_time < 4 or n_vox < 2:
        const_mask = total_std < const_threshold
        return total_std, const_mask, int(const_mask.sum().item())

    r = _signal_rank(n_time, signal_rank)

    # Temporal demeaning: the covariance whose eigenvectors we want is of the time series
    # about their own means, and the mean is separately accounted for in the DoF below.
    work_dtype = torch.float32 if x_t.device.type == "mps" else torch.float64
    xc = x_t.to(work_dtype)
    xc = xc - xc.mean(dim=0, keepdim=True)

    cov_t = (xc @ xc.T) / float(n_vox)  # (T, T)
    evals, evecs = torch.linalg.eigh(cov_t)
    del cov_t
    order = torch.argsort(evals, descending=True)
    basis = evecs[:, order][:, :r]  # (T, r), orthonormal
    del evals, evecs

    # Residual after projecting each voxel's time series onto the signal subspace.
    # (I - B Bt) x, computed as x - B (Bt x) so nothing (T, T) is materialised.
    resid = xc - basis @ (basis.T @ xc)
    del xc, basis

    # Plain std would divide by (T - 1); the residual actually carries (T - r - 1)
    # degrees of freedom, so rescale to remove the downward bias.
    dof = max(1, n_time - r - 1)
    noise_var = (resid * resid).sum(dim=0) / float(dof)
    del resid
    noise_std = torch.sqrt(torch.clamp(noise_var, min=0.0)).to(total_std.dtype)

    const_mask = (total_std < const_threshold) | (noise_std < const_threshold)
    return noise_std, const_mask, int(const_mask.sum().item())


@torch.inference_mode()
def apply_noise_std_map(
    data_vox_t: Tensor,
    noise_std: Tensor,
    const_mask: Tensor,
) -> Tensor:
    """Divide each voxel by its noise std; zero the voxels marked constant.

    Split from :func:`noise_std_map` so a map estimated once -- on an across-run mean, say
    -- can be applied to each run on the same voxel grid, which keeps runs on a common
    scale instead of normalising each to its own noise level.
    """
    out = data_vox_t.clone()
    safe = torch.where(const_mask, torch.ones_like(noise_std), noise_std)
    out /= safe.unsqueeze(1)
    out[const_mask, :] = 0.0
    return out


@torch.inference_mode()
def variance_normalize(
    data_vox_t: Tensor,
    signal_rank: int | None = None,
) -> tuple[Tensor, int]:
    """Estimate the noise std map and apply it. Returns ``(normalized, n_constant)``.

    ``data_vox_t`` is ``(n_vox, n_time)``; the result has the same shape.
    """
    noise_std, const_mask, n_const = noise_std_map(data_vox_t, signal_rank)
    return apply_noise_std_map(data_vox_t, noise_std, const_mask), n_const
