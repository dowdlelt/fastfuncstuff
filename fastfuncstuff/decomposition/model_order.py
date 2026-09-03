"""Model-order selection for PCA/ICA — how many components does the data support?

Paper-derived throughout. No implementation is a reference for this module; see
``../fmri_wiki/notes/FSL clean-room policy.md``.

References
----------
- Minka, T.P. (2000). *Automatic Choice of Dimensionality for PCA*. NIPS 13:598-604.
  The Laplace approximation to the PPCA model evidence, eq. 76-79.
- Marchenko, V.A. & Pastur, L.A. (1967). *Distribution of eigenvalues for some sets of
  random matrices*. Mat. Sb. 72(4):507-536. The limiting spectrum of a pure-noise
  sample covariance, which is what tells us where the noise bulk ends.
- Beckmann, C.F. & Smith, S.M. (2004). *Probabilistic independent component analysis for
  functional magnetic resonance imaging*. IEEE TMI 23(2):137-152. Establishes that
  spatial smoothness inflates the apparent sample size and must be corrected for, and
  that the evidence is evaluated on an eigenspectrum *adjusted* for the noise-only
  distribution.
- Smith, S.M., Beckmann, C.F. & DeLuca, M. (2005). Phil Trans R Soc B 360(1457):1001-1013.
  Restates the adjustment step explicitly: "we can adjust the observed eigenspectrum by
  the quantiles of the predicted cumulative distribution of eigenvalues from Gaussian
  noise (Johnstone 2000), prior to estimating the model order".
- Johnstone, I.M. (2001). *On the distribution of the largest eigenvalue in principal
  components analysis*. Ann. Statist. 29(2):295-327. The null distribution being
  divided out.
- Worsley, K.J. et al. (1992). J. Cereb. Blood Flow Metab. 12:900-918. Resels.

The estimator
-------------
Three published facts, composed **in this order**:

0. **Adjust the spectrum first.** Pure Gaussian noise does not produce a flat spectrum --
   its eigenvalues spread across the whole Marchenko-Pastur interval. Dividing the
   observed spectrum by the expected noise-only order statistics removes that slope
   (:func:`adjust_eigenspectrum`). PICA does this before selecting, and it is not
   optional: see the measurement below.
1. **Marchenko-Pastur gives a ceiling.** For a pure-noise matrix, the sample eigenvalues
   concentrate on a known interval; anything above its upper edge is inconsistent with
   noise. The count of such eigenvalues is the most components the spectrum can even
   distinguish from noise, and no probabilistic criterion should be allowed past it.
2. **Minka's Laplace evidence picks within that.** Among ranks that are defensible at
   all, choose the one the PPCA model evidence prefers.

The failure mode this is built to avoid is the Laplace curve running away into the noise
bulk. Correcting ``n_samples`` for spatial smoothness (:func:`effective_sample_size`) is
*necessary but not sufficient*: measured on ds005165 rest run 1, the Laplace argmax sits
at ``T-1`` -- the largest value available -- at every effective sample size from 78,832
down to 1,000, so the evidence never binds and the MP ceiling alone selects. What fixes
it is step 0. At the same ``n_samples``, the unadjusted spectrum gives k=84 and the
adjusted one k=59 (MELODIC on that run: 66).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ModelOrderResult",
    "effective_sample_size",
    "effective_sample_size_from_resels",
    "adjust_eigenspectrum",
    "laplace_evidence_curve",
    "mp_expected_eigenvalues",
    "mp_noise_level",
    "mp_signal_count",
    "select_model_order",
]


@dataclass
class ModelOrderResult:
    """Selected model order plus everything needed to argue with the choice."""

    k: int
    """Selected number of components."""

    k_mp: int
    """Marchenko-Pastur signal count — the ceiling."""

    sigma2: float
    """Estimated noise variance (the MP bulk scale)."""

    lambda_plus: float
    """MP upper edge; eigenvalues above this are not consistent with noise."""

    n_samples: int
    """Effective sample size actually used."""

    log_evidence: np.ndarray = field(default_factory=lambda: np.empty(0))
    """Laplace log-evidence per candidate k (indexed from ``k_min``)."""

    k_min: int = 1
    """First k the evidence curve was evaluated at."""

    at_ceiling: bool = False
    """Evidence was still rising where the search stopped **and** the stop was the
    user's ``k_max``. The MP cap binding is the normal, intended outcome and is not
    flagged here -- read ``k_mp`` for that."""

    ceiling_source: str = "spectrum"
    """What set the top of the search: ``"mp"``, ``"k_max"`` or ``"spectrum"``."""

    def as_dict(self) -> dict:
        return {
            "k": int(self.k),
            "k_mp": int(self.k_mp),
            "sigma2": float(self.sigma2),
            "lambda_plus": float(self.lambda_plus),
            "n_samples": int(self.n_samples),
            "k_min": int(self.k_min),
            "at_ceiling": bool(self.at_ceiling),
            "ceiling_source": str(self.ceiling_source),
            "log_evidence": np.asarray(self.log_evidence, dtype=float).tolist(),
        }


def effective_sample_size(n_voxels: int, fwhm_voxels: tuple[float, float, float]) -> int:
    """Independent spatial samples in ``n_voxels``, given a smoothness estimate.

    A smoothed image has fewer independent observations than it has voxels, and Minka's
    evidence is proportional to that count -- feeding it the raw voxel count is what makes
    the curve rise without bound. One resel is ``FWHM_x * FWHM_y * FWHM_z`` voxels
    (Worsley et al. 1992), so the resel count is the natural sample size.

    ``fwhm_voxels`` is the estimated smoothness **in voxel units** along each axis; get it
    from :mod:`fastfuncstuff.stats.fwhmx`. Values below 1 voxel mean no smoothing along
    that axis and contribute a factor of 1.
    """
    resel = 1.0
    for f in fwhm_voxels:
        resel *= max(1.0, float(f))
    return effective_sample_size_from_resels(n_voxels, resel)


def effective_sample_size_from_resels(n_voxels: int, resels: float, floor: int = 1) -> int:
    """As :func:`effective_sample_size`, given the resel *product* directly.

    One resel is the smoothness volume, so the count of independent observations is
    ``n_voxels / resels`` -- the definition (Nichols, "FWHM/RESEL details for SPM and
    FSL"), with no additional scaling. ``floor`` lets a caller keep the result at or
    above some minimum (typically the number of timepoints, below which a temporal
    covariance estimate stops being meaningful).

    ``resels`` is clamped at 1: a resel smaller than a voxel would report *more*
    independent samples than there are voxels, which is impossible. Estimators do return
    sub-voxel FWHMs on barely-smoothed data (a per-axis FWHM near or below the voxel
    width), and :func:`effective_sample_size` already clamps per axis for the same
    reason -- this is the same guard for callers that pass the product directly.
    """
    if n_voxels <= 0:
        raise ValueError(f"n_voxels must be positive, got {n_voxels}")
    return max(int(floor), 1, int(n_voxels / max(float(resels), 1.0)))


def mp_noise_level(
    evals: np.ndarray,
    n_samples: int,
    *,
    max_iter: int = 32,
    tol: float = 1e-6,
) -> tuple[float, float]:
    """Estimate the noise variance and MP upper edge from a spectrum.

    Returns ``(sigma2, lambda_plus)``.

    The naive estimate -- take the median of the lower half of the spectrum -- is biased
    whenever signal reaches into that half, and the bias is upward, which pushes the edge
    out and makes the criterion *under*-count. So instead fit the bulk iteratively:
    estimate ``sigma2`` from the eigenvalues currently believed to be noise, recompute the
    edge, drop whatever now sits above it, and repeat to a fixed point. This is the
    standard way to fit an MP bulk in the presence of spikes and it converges in a handful
    of passes.

    ``evals`` must be sorted descending. ``n_samples`` is the number of (effective)
    observations; the aspect ratio is ``len(evals) / n_samples``.
    """
    ev = np.asarray(evals, dtype=np.float64)
    if ev.ndim != 1 or ev.size == 0:
        raise ValueError("evals must be a non-empty 1-D array")
    n_features = ev.size
    n_samples = max(1, int(n_samples))
    gamma = n_features / n_samples

    # Start from the whole spectrum; the first pass is deliberately over-inclusive so a
    # heavily-spiked spectrum still gets a finite starting scale.
    noise = ev
    sigma2 = float(np.mean(noise)) if noise.size else 0.0
    lambda_plus = sigma2 * (1.0 + np.sqrt(gamma)) ** 2

    for _ in range(max_iter):
        keep = ev[ev <= lambda_plus]
        if keep.size < 2:
            # Everything looks like signal; fall back to the smallest few eigenvalues so
            # the edge stays finite and the caller gets a usable ceiling.
            keep = ev[-max(2, n_features // 10) :]
        new_sigma2 = float(np.mean(keep))
        new_edge = new_sigma2 * (1.0 + np.sqrt(gamma)) ** 2
        converged = abs(new_edge - lambda_plus) <= tol * max(1.0, abs(lambda_plus))
        sigma2, lambda_plus = new_sigma2, new_edge
        if converged:
            break

    return sigma2, lambda_plus


def mp_signal_count(evals: np.ndarray, n_samples: int) -> tuple[int, float, float]:
    """Number of eigenvalues above the Marchenko-Pastur upper edge.

    Returns ``(count, sigma2, lambda_plus)``. This is the RMT statement of "how many
    directions in this data are not explainable as noise", and it is used here as a
    ceiling rather than as the answer.
    """
    ev = np.asarray(evals, dtype=np.float64)
    sigma2, lambda_plus = mp_noise_level(ev, n_samples)
    return int((ev > lambda_plus).sum()), sigma2, lambda_plus


def laplace_evidence_curve(
    evals: np.ndarray,
    n_samples: int,
    k_min: int,
    k_max: int,
) -> np.ndarray:
    """Minka (2000) Laplace log-evidence for PPCA, for each rank in ``[k_min, k_max]``.

    Direct implementation of eq. 76-79 of the paper. Returns an array of length
    ``k_max - k_min + 1``; entries for ranks that are not admissible (a zero eigenvalue
    inside the retained block, or a rank reaching the feature count) are ``-inf``.

    The evidence has four parts -- a prior volume term over the Stiefel manifold, the
    likelihood of the retained eigenvalues, the likelihood of the noise block at its
    pooled variance, and the Occam factor from the Laplace approximation, whose ``pa``
    term is the sum over eigenvalue pairs and is what penalises ranks that split a
    degenerate (i.e. noise) block.
    """
    ev = np.asarray(evals, dtype=np.float64)
    if ev.ndim != 1:
        raise ValueError("evals must be 1-D")
    d = ev.size
    n = float(max(1, int(n_samples)))
    eps = 1e-15

    k_min = max(1, int(k_min))
    k_max = min(int(k_max), d - 1)
    if k_max < k_min:
        return np.full(0, -np.inf)

    out = np.full(k_max - k_min + 1, -np.inf, dtype=np.float64)

    # pu, the Stiefel prior volume, is a running sum over i and so is shared across k.
    idx = np.arange(1, d + 1, dtype=np.float64)
    from scipy.special import gammaln  # local: keeps module import cheap

    pu_terms = gammaln((d - idx + 1.0) / 2.0) - np.log(np.pi) * (d - idx + 1.0) / 2.0
    pu_cum = np.cumsum(pu_terms)

    log_ev = np.log(np.maximum(ev, eps))
    log_ev_cum = np.cumsum(log_ev)
    # Suffix sums of the eigenvalues give the noise-block variance for every k at once.
    suffix_sum = np.concatenate([np.cumsum(ev[::-1])[::-1], [0.0]])

    for k in range(k_min, k_max + 1):
        if ev[k - 1] < eps:
            continue

        pu = -k * np.log(2.0) + pu_cum[k - 1]
        pl = -log_ev_cum[k - 1] * n / 2.0

        v = max(eps, suffix_sum[k] / (d - k))
        pv = -np.log(v) * n * (d - k) / 2.0

        m = d * k - k * (k + 1.0) / 2.0
        pp = np.log(2.0 * np.pi) * (m + k) / 2.0

        # pa = sum_{i<k} sum_{j>i} log[(l_i - l_j)(1/lhat_j - 1/lhat_i) N], with the noise
        # block held at v. Vectorised over j for each i.
        ev_hat = ev.copy()
        ev_hat[k:] = v
        inv_hat = 1.0 / np.maximum(ev_hat, eps)
        pa = 0.0
        for i in range(k):
            diff_eig = ev[i] - ev[i + 1 :]
            diff_inv = inv_hat[i + 1 :] - inv_hat[i]
            prod = diff_eig * diff_inv * n
            pa += float(np.sum(np.log(np.maximum(prod, eps))))

        out[k - k_min] = pu + pl + pv + pp - pa / 2.0 - k * np.log(n) / 2.0

    return out


def mp_expected_eigenvalues(
    n_features: int,
    n_samples: int,
    sigma2: float = 1.0,
    *,
    grid: int = 20000,
) -> np.ndarray:
    """Expected eigenvalues of pure Gaussian noise, descending (Marchenko-Pastur quantiles).

    Under a Gaussian-noise null the sample eigenvalues are Wishart distributed, and for
    ``n_features/n_samples = gamma`` their limiting density is Marchenko-Pastur (1967) on
    ``[lambda_-, lambda_+] = sigma2 * (1 -+ sqrt(gamma))^2``. The ``i``-th largest expected
    eigenvalue is the ``1 - (i - 0.5)/p`` quantile of that law.

    Even pure noise produces a *sloping* spectrum -- the largest noise eigenvalue is
    ``(1+sqrt(gamma))^2`` and the smallest ``(1-sqrt(gamma))^2``, a real spread that has
    nothing to do with signal. This function is that expected slope, so it can be divided
    out; see :func:`adjust_eigenspectrum`.
    """
    p = int(n_features)
    if p < 1:
        raise ValueError(f"n_features must be positive, got {n_features}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    g = p / float(n_samples)
    s2 = max(float(sigma2), 1e-30)
    lo, hi = s2 * (1.0 - np.sqrt(g)) ** 2, s2 * (1.0 + np.sqrt(g)) ** 2
    if not np.isfinite(hi) or hi <= lo:
        return np.full(p, s2, dtype=np.float64)
    x = np.linspace(lo, hi, int(grid))
    dens = np.sqrt(np.clip((hi - x) * (x - lo), 0.0, None)) / (
        2.0 * np.pi * g * np.maximum(x, 1e-30) * s2
    )
    cdf = np.cumsum(dens)
    total = float(cdf[-1])
    if total <= 0:
        return np.full(p, s2, dtype=np.float64)
    cdf = cdf / total
    q = 1.0 - (np.arange(1, p + 1) - 0.5) / p  # descending order statistics
    return np.interp(q, cdf, x)


def adjust_eigenspectrum(evals: np.ndarray, n_samples: int) -> tuple[np.ndarray, float]:
    """Divide out the eigenspectrum slope that Gaussian noise alone would produce.

    Returns ``(adjusted, sigma2)``. This is the step Beckmann & Smith's PICA applies
    *before* model-order selection, and skipping it is the single biggest source of
    over-counting in this module:

        "the eigenvalues have a Wishart distribution and we can adjust the observed
        eigenspectrum by the quantiles of the predicted cumulative distribution of
        eigenvalues from Gaussian noise (Johnstone 2000), prior to estimating the model
        order [...] we use the Laplace approximation to the posterior distribution of the
        model evidence that can be calculated efficiently from the adjusted eigenspectrum"
        -- Smith, Beckmann & DeLuca (2005), Phil Trans R Soc B 360:1001, reviewing
        Beckmann & Smith (2004), IEEE TMI 23:137.

    Without it both criteria see a noise tail that still slopes, so one more component
    always pays: measured on ds005165 rest run 1, the raw spectrum gives k=84 and the
    adjusted one k=59 (MELODIC: 66) at the same effective sample size.
    """
    ev = np.asarray(evals, dtype=np.float64)
    if ev.ndim != 1 or ev.size < 3:
        raise ValueError(f"need at least 3 eigenvalues, got {ev.size}")
    if not np.all(np.diff(ev) <= 1e-9 * max(1.0, float(ev[0]))):
        ev = np.sort(ev)[::-1]
    ev = np.clip(ev, 0.0, None)
    sigma2, _ = mp_noise_level(ev, n_samples)
    expected = mp_expected_eigenvalues(ev.size, n_samples, sigma2=sigma2)
    return ev / np.maximum(expected, 1e-12), float(sigma2)


def select_model_order(
    evals: np.ndarray,
    n_samples: int,
    *,
    k_min: int = 1,
    k_max: int | None = None,
    use_mp_ceiling: bool = True,
    mp_slack: float = 1.0,
    adjust_spectrum: bool = True,
) -> ModelOrderResult:
    """Choose a model order: Laplace evidence, capped by Marchenko-Pastur.

    ``evals`` is the eigenvalue spectrum, sorted descending. ``n_samples`` should be the
    *effective* sample size (see :func:`effective_sample_size`) -- passing a raw voxel
    count is the single most common way to get an absurd answer out of this.

    ``mp_slack`` widens the ceiling to ``mp_slack * k_mp``. The default of 1.0 is a hard
    cap: never return more components than the spectrum can distinguish from noise.
    Raising it lets the evidence argue for a few more, which is occasionally right when
    the noise is not iid and the MP null is therefore slightly misspecified; it is a knob,
    not a default.

    ``adjust_spectrum`` divides out the slope Gaussian noise alone would produce before
    either criterion looks at the spectrum (:func:`adjust_eigenspectrum`). This is part of
    the published PICA recipe, not a tweak, and turning it off is what makes the count run
    away -- keep it on unless you are deliberately reproducing the unadjusted behaviour.

    Selection is the **global** maximum of the evidence over the admissible range, which
    is what Minka's paper prescribes. There is no first-peak rule: with a corrected
    ``n_samples`` and an MP ceiling the curve is unimodal in practice, and a first-peak
    rule mostly encodes a fix for a curve that was misspecified to begin with.
    """
    ev = np.asarray(evals, dtype=np.float64)
    if ev.ndim != 1 or ev.size < 3:
        raise ValueError(f"need at least 3 eigenvalues, got {ev.size}")
    if not np.all(np.diff(ev) <= 1e-9 * max(1.0, float(ev[0]))):
        ev = np.sort(ev)[::-1]
    ev = np.clip(ev, 0.0, None)
    d = ev.size

    if adjust_spectrum:
        ev, _ = adjust_eigenspectrum(ev, n_samples)

    k_mp, sigma2, lambda_plus = mp_signal_count(ev, n_samples)

    hi_spectrum = d - 1
    hi_user = hi_spectrum if k_max is None else min(int(k_max), hi_spectrum)
    # k_mp == 0 means MP found nothing above the noise edge. That is the ceiling doing its
    # job at its strongest, so it must still bind -- the old `k_mp >= 1` guard fell through
    # to an unconstrained search in exactly the case where the data supports no components
    # at all, and the Laplace curve then ran to its argmax on pure noise.
    hi_mp = max(1, int(round(mp_slack * k_mp))) if use_mp_ceiling else hi_user
    hi = min(hi_user, hi_mp)
    # Which constraint actually bound decides whether a still-rising curve is worth
    # warning about: the MP cap binding is the design working, a k_max cap binding means
    # the user asked for fewer components than the data supports.
    if hi == hi_mp and hi_mp < hi_user:
        ceiling_source = "mp"
    elif k_max is not None and hi == hi_user < hi_spectrum:
        ceiling_source = "k_max"
    else:
        ceiling_source = "spectrum"
    lo = max(1, min(int(k_min), hi))

    curve = laplace_evidence_curve(ev, n_samples, lo, hi)
    if curve.size == 0 or not np.isfinite(curve).any():
        # Degenerate spectrum: fall back on the RMT count, which needs no likelihood.
        k = int(max(1, min(k_mp if k_mp > 0 else 1, d - 1)))
        return ModelOrderResult(
            k=k,
            k_mp=k_mp,
            sigma2=sigma2,
            lambda_plus=lambda_plus,
            n_samples=int(n_samples),
            log_evidence=curve,
            k_min=lo,
            ceiling_source=ceiling_source,
        )

    k = int(lo + int(np.argmax(np.where(np.isfinite(curve), curve, -np.inf))))
    finite = curve[np.isfinite(curve)]
    still_rising = bool(k == hi and finite.size >= 2 and finite[-1] > finite[-2])
    at_ceiling = still_rising and ceiling_source == "k_max"

    return ModelOrderResult(
        k=k,
        k_mp=k_mp,
        sigma2=sigma2,
        lambda_plus=lambda_plus,
        n_samples=int(n_samples),
        log_evidence=curve,
        k_min=lo,
        at_ceiling=at_ceiling,
        ceiling_source=ceiling_source,
    )
