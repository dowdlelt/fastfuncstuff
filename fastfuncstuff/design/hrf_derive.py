"""
Data-derived HRF library construction (NSD-style).

This module implements the pipeline used by the Natural Scenes Dataset (NSD)
team to derive a custom HRF library from a subject's own FIR/TENT-fit BOLD
timecourses (see `hrf_derivecanonicalpcs.m` / `hrf_constructmanifold.m` in
https://github.com/cvnlab/nsddatapaper).

Overview of the pipeline (matches NSD paper supplement):

1. **Per-voxel FIR/TENT fit** (done by the caller, e.g. ``ffs_librarian``).
   All trials in the group are pooled into a single "condition" to make the
   estimate feasible. The result is a per-voxel impulse response of shape
   ``(n_voxels, n_lags)`` at TR resolution, plus an R² map.

2. **Voxel selection** — :func:`select_voxels`.  Keep only voxels above an
   R² threshold (NSD used 10%); if more than ``max_voxels`` survive, take
   a random sample (with replacement in NSD; without here, deterministic).

3. **SVD on the selected impulse responses** — :func:`svd_decompose`.
   Treating each voxel's FIR estimate as a vector in lag-space, the top-K
   right singular vectors form a temporal basis (default K=3, matching
   NSD).  The per-voxel weights on this basis are ``U · diag(S)``.

4. **Unit-sphere projection** — :func:`project_unit_sphere`.  Normalize
   each voxel's K-D weight vector to unit length, following the Temporal
   Decomposition Method of Chen et al. 2021 used in the NSD paper to
   visualize HRF-shape variation independent of amplitude.

5. **Manifold sampling** — pick ``n_points`` directions out of the
   density on the unit sphere.  Two families, and the choice is a claim
   about the data:

   * **1-D curve** — :func:`trace_manifold_auto` (default) walks the
     density *ridge*, so ``n_points`` samples spaced ``angular_step_deg``
     apart trace the principal path through the cloud and adjacent
     library indices hold similar HRFs.  This is NSD's model of HRF
     variation: one family running early-to-late.
     :func:`trace_manifold_grid` is a density-blind 1-D fallback.
   * **Filled region** — :func:`trace_manifold_blob` covers the cloud's
     support evenly and :func:`trace_manifold_kmeans` covers it in
     proportion to voxel count.  Use these when the cloud has genuine
     width the curve misses; :func:`manifold_coverage` is the referee.

   Every sampler here works at **any K**.  The ridge walk used to be
   capped at K=3 by a Fibonacci-sphere grid, which is also what made it
   trace a straight line instead of the density (see :func:`_mean_shift`).
   :func:`trace_manifold_from_points` overrides with clicked points.

6. **Reconstruction & re-sampling** — :func:`reconstruct_timecourses`.
   Each manifold point ``w`` reconstructs a TR-resolution HRF as
   ``V.T @ w``; we then cubic-interpolate to a 0.1 s grid and normalize
   to peak = 1, matching the format of
   :data:`fastfuncstuff.design.hrf._HRF_LIBRARY_FILE`.

7. **(Optional) parametric fit** — :func:`fit_double_gamma` snaps each
   reconstructed curve to an SPM-style double-gamma, smoothing out
   high-frequency noise from the FIR fit while preserving timing.  The
   ``raw`` reconstructions are always returned alongside the fits so the
   user can inspect the difference and decide whether a richer parametric
   family (multi-gamma, FLOBS, …) is needed.

Design notes
------------
- **Pure numpy/scipy.**  No torch dependency — this module is fed by the
  CLI after GLM fitting and operates on small arrays (~20k voxels × ~30
  lags).
- **No condition awareness.**  Conditions are pooled / grouped by the
  caller before the FIR fit; this module sees only the resulting impulse
  responses.  That keeps the algorithm a "library builder" rather than
  a condition-aware HRF model.
- **K=3 by default, but not a limit.**  K=3 is what NSD used and is the
  well-tested path; K=2 leaves a 1-D arc that is barely a "manifold".
  Higher K is fully supported by every sampler — the ridge walk is
  mean-shift-based and the region fillers are data-candidate-based, so
  none of them needs a grid on the sphere.  Raising K is worth doing when
  the eigenvalue spectrum has no elbow at 3 and ``manifold_coverage``
  reports a poorly-served tail; the QC heatmap stays a (PC2, PC3)
  projection and no longer shows the whole story.

Duration handling
-----------------
The FIR/TENT fit estimates the **response to whatever event shape was in
the data** — including the stimulus duration's boxcar.  Downstream tools
(``ffs_hrfopt``, ``ffs_denoise``, …) re-convolve the library HRF with the
event boxcar at modelling time, so the library must hold the *impulse
response*; otherwise the design is doubly convolved.  For brief events
(duration ≪ HRF width) the difference is negligible; for block designs
it is not, and two conditions sharing an HRF but differing in duration
would otherwise yield different-looking library entries.

Two corrections are available (``deconv_method`` in
:func:`derive_library`):

- ``"fit"`` (default, NSD-faithful) — put the boxcar inside the
  double-gamma forward model and fit, so the recovered parameters
  describe the impulse response directly.  This is what
  ``hrf_fitspmhrftomanifold.m`` does; no numerical inverse is involved.
  See :func:`fit_double_gamma_through_boxcar`.
- ``"wiener"`` — explicitly invert the boxcar convolution
  (:func:`deconvolve_event_duration`).  Required when ``fit_gamma=False``
  since there is then no parametric family to carry the correction, but
  it has to regularize away the boxcar's spectral zeros at multiples of
  ``1/duration``, which the fit approach never encounters.

Multi-subject libraries
-----------------------
:func:`derive_library` is subject-agnostic: it sees a ``(n_rows, n_lags)``
beta matrix and an R² vector.  To build a study-wide library, row-stack
each subject's *selected* FIR betas and pass the stack.  NSD did exactly
this across its 8 subjects, sampling a fixed count per subject so each
contributes equally regardless of how many voxels passed its R² gate.
``ffs_librarian -combine`` implements that; see
:func:`stack_subject_betas`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.interpolate import BSpline, PchipInterpolator
from scipy.optimize import curve_fit
from scipy.stats import gamma as _gamma

# ----------------------------------------------------------------------------
# Voxel selection
# ----------------------------------------------------------------------------


def select_voxels(
    r2: np.ndarray,
    threshold: float = 0.10,
    max_voxels: int = 20_000,
    seed: int = 42,
) -> np.ndarray:
    """Select high-R² voxels for HRF library derivation.

    Mirrors NSD's selection step: keep voxels with model R² above
    ``threshold``; if more than ``max_voxels`` survive, draw a random
    sample.  NSD drew with replacement (to weight high-R² regions); we
    draw *without* replacement and seed the RNG so results are
    reproducible — for the typical case of >20k high-R² voxels the
    difference is negligible.

    Parameters
    ----------
    r2 : np.ndarray, shape (n_voxels,)
        Per-voxel coefficient of determination from the pooled FIR/TENT
        fit.  Values outside [0, 1] are allowed; only ``> threshold`` is
        checked.
    threshold : float, default 0.10
        Minimum R² to keep.  NSD used 0.10 (10%).
    max_voxels : int, default 20000
        Maximum number of voxels to return.  If fewer survive, all are
        returned.
    seed : int, default 42
        Seed for the random subsample.

    Returns
    -------
    indices : np.ndarray, shape (n_selected,)
        Indices into the original ``r2`` array.  ``n_selected <=
        min(max_voxels, (r2 > threshold).sum())``.
    """
    high = np.flatnonzero(r2 > threshold)
    if high.size <= max_voxels:
        return high
    rng = np.random.default_rng(seed)
    return rng.choice(high, size=max_voxels, replace=False)


def select_library_voxels(
    betas: np.ndarray,
    r2: np.ndarray,
    threshold: float = 0.10,
    max_voxels: int = 20_000,
    seed: int = 42,
) -> np.ndarray:
    """R² gate plus the dead-voxel filter, as one reusable step.

    :func:`select_voxels` applies only the R² threshold.  This wrapper
    then drops voxels whose FIR beta vector is numerically zero.

    Why the second filter exists: without a brain mask, air voxels reach
    the GLM as (post-scaling) constants.  Anything in the nuisance block
    fits a constant perfectly, so their residual is ~0 and R² ~1 — they
    look like the *best* voxels to an R² gate while carrying all-zero
    FIR betas, and they then dominate the SVD and turn the PCs into
    noise.  NSD sidestepped this with a BET mask up front; we also
    filter post-hoc so the failure cannot happen silently.

    Callers that need the selection *before* calling
    :func:`derive_library` (to derive PCs for the NSD refit) should use
    this and pass the result back in as ``precomputed_selection``, so
    both stages provably see the same voxels.

    Returns indices into the original voxel axis.
    """
    sel = select_voxels(r2, threshold=threshold, max_voxels=max_voxels, seed=seed)
    norms = np.linalg.norm(betas[sel], axis=1)
    floor = 1e-8 * max(float(np.median(norms[norms > 0])) if (norms > 0).any() else 1.0, 1.0)
    alive = norms > floor
    n_dropped = int((~alive).sum())
    if n_dropped > 0.5 * sel.size and sel.size > 0:
        import warnings

        warnings.warn(
            f"select_library_voxels: {n_dropped} of {sel.size} voxels passing the "
            f"R² gate had zero-norm FIR betas (constant/air signal scoring R²≈1). "
            f"Strongly recommend passing -mask <brain_mask.nii.gz>.",
            RuntimeWarning,
            stacklevel=2,
        )
    return sel[alive]


# ----------------------------------------------------------------------------
# SVD decomposition
# ----------------------------------------------------------------------------


@dataclass
class SVDResult:
    """Container for the SVD step of the HRF library derivation.

    Attributes
    ----------
    weights : np.ndarray, shape (n_voxels, n_pcs)
        Per-voxel coefficients on the principal-component basis,
        equal to ``U[:, :K] @ diag(S[:K])``.  These represent each
        voxel's HRF shape in PC space; their sign and magnitude both
        carry information.
    pcs : np.ndarray, shape (n_pcs, n_lags)
        The temporal PCs (right singular vectors).  Each row is a basis
        function over the FIR/TENT lags.  These are unit-norm and
        orthogonal.
    eigvals : np.ndarray, shape (min(n_voxels, n_lags),)
        Full singular value spectrum — kept so the caller can decide
        whether ``n_pcs`` was a reasonable choice (look for an elbow).
    variance_explained : np.ndarray, shape (n_pcs,)
        Fraction of total variance captured by each of the kept PCs.
    """

    weights: np.ndarray
    pcs: np.ndarray
    eigvals: np.ndarray
    variance_explained: np.ndarray


def svd_decompose(
    betas: np.ndarray,
    n_pcs: int = 3,
    *,
    unit_normalize: bool = True,
    sign_align: bool = True,
) -> SVDResult:
    """Decompose per-voxel FIR/TENT betas into a low-rank temporal basis.

    Computes the truncated SVD of the selected ``(n_voxels, n_lags)``
    matrix, **after per-row unit-length normalization** (NSD-style).

    Why row normalization is critical: BOLD HRFs across cortex differ
    in *shape* (time-to-peak, width, undershoot depth) and in
    *amplitude*.  We want a library of shape variants.  Raw FIR betas
    have both, and amplitude dominates the SVD — PC1 ends up reflecting
    "where the signal is strong" rather than HRF shape, and the
    resulting PCs look noise-like rather than HRF-shaped.  Dividing each
    voxel's FIR vector by its L2 norm projects every voxel onto the
    unit sphere in lag-space, after which SVD recovers the shape
    manifold cleanly.  This matches NSD's ``hrf_derivecanonicalpcs.m``::

        vlen = vectorlength(firs(vxselect(p,:),2,:,p),3);
        firsSELECT(:,:,:,p) = bsxfun(@rdivide, firs(...), vlen);
        [u,s,v] = svd(firsSELECT_squished, 0);

    Why this works: HRFs across cortex span a low-dimensional manifold
    of shape variation.  After unit-norming, three PCs typically capture
    >90% of the variance and are easy to visualize on a sphere (see
    :func:`project_unit_sphere`).

    Parameters
    ----------
    betas : np.ndarray, shape (n_voxels, n_lags)
        Per-voxel FIR/TENT betas at TR resolution, after voxel selection.
        Each row is one voxel's estimated HRF shape over the FIR window.
    n_pcs : int, default 3
        Number of singular components to keep.  3 is the well-validated
        NSD choice and keeps the whole cloud visible in the QC heatmap;
        2 collapses the manifold to a 1-D arc.  4+ is supported by every
        sampler (:func:`trace_manifold_auto` included), at the cost of the
        heatmap becoming a projection — check the eigenvalue spectrum for
        an elbow and :func:`manifold_coverage` for a poorly-served tail
        before spending the extra dimensions.
    unit_normalize : bool, default True
        Divide each voxel's FIR vector by its L2 norm before SVD.
        Almost always what you want for HRF-shape extraction.  Set to
        False only for diagnostic / amplitude-aware analyses.
    sign_align : bool, default True
        Ensure each PC has a positive peak (flip sign if the largest
        absolute value is negative).  Cosmetic — makes PC1 look like a
        canonical positive HRF rather than its mirror image — and is
        downstream-relevant because the per-voxel sign-flip step uses
        the sign of the PC1 loading.

    Returns
    -------
    SVDResult
        ``weights`` are the per-voxel coordinates on the PCs of the
        **unit-normalized** input.  When ``unit_normalize=True`` these
        approximately live on the unit sphere already (small departures
        from it come from truncating at K=3).

    Raises
    ------
    ValueError
        If ``n_pcs`` exceeds ``min(n_voxels, n_lags)``.
    """
    if betas.ndim != 2:
        raise ValueError(f"betas must be 2D (n_voxels, n_lags); got shape {betas.shape}")
    n_voxels, n_lags = betas.shape
    if n_pcs > min(n_voxels, n_lags):
        raise ValueError(f"n_pcs={n_pcs} exceeds min(n_voxels={n_voxels}, n_lags={n_lags})")

    if unit_normalize:
        norms = np.linalg.norm(betas, axis=1, keepdims=True)
        safe = np.where(norms > 1e-12, norms, 1.0)
        work = betas / safe
        # Drop zero-norm rows: they're either masked-out voxels or
        # numerical sinks and would otherwise contribute a 0-vector
        # row that biases SVD's right-singular vectors by nothing
        # but also wastes a left-singular slot.
        keep = norms.squeeze() > 1e-12
        work = work[keep]
    else:
        work = betas
        keep = np.ones(n_voxels, dtype=bool)

    # full_matrices=False keeps U as (n_voxels, k) with k=min(n_voxels,n_lags);
    # we do not need the full (n_voxels, n_voxels) left-singular basis.
    U, S, Vt = np.linalg.svd(work, full_matrices=False)
    pcs = Vt[:n_pcs]  # (n_pcs, n_lags)

    if sign_align:
        # Flip each PC so the largest-magnitude sample is positive.
        # This is purely cosmetic for PC1 (canonical-looking) and
        # downstream-relevant: the per-voxel sign-flip uses sign(PC1
        # loading), so we want PC1 to have an unambiguous "positive
        # HRF" orientation across runs.
        peak_signs = np.sign(pcs[np.arange(n_pcs), np.argmax(np.abs(pcs), axis=1)])
        peak_signs = np.where(peak_signs == 0, 1.0, peak_signs)
        pcs = pcs * peak_signs[:, None]
        # Mirror sign on U/loadings to preserve `U·S·Vt = work`.
        U_k = U[:, :n_pcs] * peak_signs[None, :]
    else:
        U_k = U[:, :n_pcs]

    # Per-voxel loadings on the K-PC basis (for rows that were kept).
    # Equivalent to `work @ pcs.T` (after sign alignment).
    loadings_kept = U_k * S[:n_pcs]
    # Re-expand to full n_voxels with zeros for the dropped rows so
    # downstream code can index by the original voxel order.
    weights = np.zeros((n_voxels, n_pcs), dtype=loadings_kept.dtype)
    weights[keep] = loadings_kept

    var = (S**2) / max((S**2).sum(), 1e-30)
    return SVDResult(
        weights=weights,
        pcs=pcs,
        eigvals=S,
        variance_explained=var[:n_pcs],
    )


def crossval_n_pcs(
    betas_train: np.ndarray,
    betas_test: np.ndarray,
    max_pcs: int | None = None,
) -> np.ndarray:
    """Cross-validated variance explained as a function of PC count.

    This is the step that told NSD to use K=3.  ``hrf_derivecanonicalpcs.m``
    fits the FIR on odd runs and even runs separately, derives PCs from
    the odd-run betas, projects the odd-run betas onto the top-K PCs, and
    scores that reconstruction against the **even-run** betas::

        recon0 = firsSELECT(:,2,:,q) * (v(:,1:p)*v(:,1:p)');
        metricR2(q,p) = calccod(flatten(recon0), flatten(firsSELECT(:,3,:,q)));

    Because the projector is rank-K, adding PCs can only help on the
    training split; the held-out split is what makes the curve turn over
    (or flatten), which is what identifies the useful dimensionality.

    Both inputs are row-unit-normalized here, matching the normalization
    the real PCA step applies.

    Parameters
    ----------
    betas_train, betas_test : np.ndarray, shape (n_voxels, n_lags)
        FIR betas from two independent halves of the data, **same voxels
        in the same order**.
    max_pcs : int, optional
        Score K = 1 … ``max_pcs``.  Defaults to ``n_lags``.

    Returns
    -------
    r2 : np.ndarray, shape (max_pcs,)
        Held-out coefficient of determination for each K, computed
        relative to zero (not to the mean) to match MATLAB ``calccod``
        defaults, so a useless reconstruction scores ≤ 0.
    """
    if betas_train.shape != betas_test.shape:
        raise ValueError(
            f"betas_train {betas_train.shape} and betas_test {betas_test.shape} must match"
        )

    def _unit_rows(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(norms > 1e-12, norms, 1.0)

    train = _unit_rows(betas_train)
    test = _unit_rows(betas_test)
    n_lags = train.shape[1]
    max_pcs = n_lags if max_pcs is None else min(max_pcs, n_lags)

    _, _, vt = np.linalg.svd(train, full_matrices=False)
    ss_tot = float((test**2).sum())
    out = np.zeros(max_pcs, dtype=np.float64)
    for k in range(1, max_pcs + 1):
        v = vt[:k]  # (k, n_lags), orthonormal rows
        recon = (train @ v.T) @ v
        ss_res = float(((test - recon) ** 2).sum())
        out[k - 1] = 1.0 - ss_res / max(ss_tot, 1e-30)
    return out


# ----------------------------------------------------------------------------
# Unit-sphere projection
# ----------------------------------------------------------------------------


def project_unit_sphere(
    weights: np.ndarray,
    eps: float = 1e-12,
    *,
    sign_flip_by_first: bool = True,
) -> np.ndarray:
    """Normalize per-voxel PC weights to unit length, optionally sign-flipped.

    Following the Temporal Decomposition Method (Chen et al. 2021) and
    NSD's ``hrf_constructmanifold.m``, we look at HRF *shape* by
    discarding amplitude.  Each voxel's K-D weight vector is divided by
    its norm so it lies on the unit K-1-sphere.

    Per-voxel sign flip (NSD-style, default on)
    -------------------------------------------
    NSD does::

        m0 = bsxfun(@times, m0, sign(m0(:,1)));   % flip so PC1 loading > 0
        m0 = unitlength(m0,2);

    Why: a unit-normed vector and its negation reconstruct *the same*
    HRF up to a sign, but they live on opposite hemispheres of the
    sphere and would split the density estimate in two.  Flipping every
    voxel so PC1 loading > 0 keeps all voxels on one hemisphere, so the
    2-D heatmap of (PC2, PC3) loadings (the "unit-circle heatmap" QC
    plot) shows one coherent density manifold rather than two
    mirror-image lobes.

    Parameters
    ----------
    weights : np.ndarray, shape (n_voxels, K)
        Weights from :func:`svd_decompose` (or any other K-D embedding).
    eps : float, default 1e-12
        Voxels with norm below ``eps`` are returned as zeros (they
        contribute nothing to the density).
    sign_flip_by_first : bool, default True
        Multiply each row by ``sign(weights[:, 0])`` before normalizing.

    Returns
    -------
    unit : np.ndarray, shape (n_voxels, K)
        Unit-norm rows (with PC1 loading non-negative when
        ``sign_flip_by_first=True``).  Zero rows correspond to near-zero
        voxels that should be filtered out before density estimation.
    """
    if sign_flip_by_first and weights.shape[1] >= 1:
        signs = np.sign(weights[:, 0:1])
        signs = np.where(signs == 0, 1.0, signs)
        weights = weights * signs
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    safe = np.where(norms > eps, norms, 1.0)
    unit = weights / safe
    # Suppress tiny-norm rows so they don't bias the density estimate.
    unit = np.where(norms > eps, unit, 0.0)
    return unit


# ----------------------------------------------------------------------------
# Manifold tracing on the unit sphere
# ----------------------------------------------------------------------------


def _spherical_kde(
    query: np.ndarray,
    data: np.ndarray,
    bandwidth_rad: float,
) -> np.ndarray:
    """Spherical kernel density at ``query`` points (von-Mises-Fisher-like).

    Uses ``exp((q·d - 1) / sigma^2)`` summed over data, which is the
    log-linearized vMF kernel up to a normalization constant — fine for
    *ranking* densities, which is all we need.
    """
    # cos_sim: (n_query, n_data), entries in [-1, 1]
    cos_sim = query @ data.T
    sigma2 = bandwidth_rad**2
    # (q·d - 1) ∈ [-2, 0]; exp gives a smooth peak at q=d.
    return np.exp((cos_sim - 1.0) / sigma2).sum(axis=1)


def _kde_sources(data: np.ndarray, max_n: int = 20000) -> np.ndarray:
    """Thin ``data`` to at most ``max_n`` rows for use as KDE kernel centres.

    A KDE is an *average* over its sources, so its shape is essentially
    unchanged by evaluating it on a fraction of them, while the cost of
    every density evaluation falls in proportion.  The walk below queries
    the density a few hundred times, so this is the difference between a
    fraction of a second and a minute on a whole-brain library.

    The subsample is a fixed stride rather than an RNG draw: it needs no
    seed, is reproducible across runs, and — because row order here is
    voxel order — spreads the sources over the whole brain instead of
    concentrating them in one slab.
    """
    if data.shape[0] <= max_n:
        return data
    return data[:: int(np.ceil(data.shape[0] / max_n))]


def _tangent_basis(p: np.ndarray) -> np.ndarray:
    """Orthonormal basis ``(K, K-1)`` of the tangent space at unit vector ``p``.

    The K=3 version of this used ``np.cross`` twice, which is the reason
    ridge tracing was capped at three PCs.  A QR of ``[p | I]`` puts ``p``
    (up to sign) in the first column and an orthonormal basis of its
    complement in the rest, for any K, at negligible cost — K is 3-6 here.
    """
    k = p.shape[0]
    q, _ = np.linalg.qr(np.column_stack([p, np.eye(k)]))
    return q[:, 1:k]


def _kernel_weights(point: np.ndarray, data: np.ndarray, sigma2: float) -> np.ndarray:
    """Per-source vMF-like kernel weights of ``data`` around ``point``."""
    return np.exp((data @ point - 1.0) / sigma2)


def _density_at(point: np.ndarray, data: np.ndarray, sigma2: float) -> float:
    """Spherical KDE evaluated at a single ``point``."""
    return float(_kernel_weights(point, data, sigma2).sum())


def _mean_shift(
    point: np.ndarray,
    data: np.ndarray,
    sigma2: float,
    *,
    forbid_direction: np.ndarray | None = None,
    max_step_rad: float,
    max_total_rad: float,
    n_iter: int = 12,
    tol: float = 1e-9,
) -> np.ndarray:
    """Move ``point`` uphill to the local KDE mode, on the sphere.

    Each iteration takes the kernel-weighted mean of the data around
    ``point``, converts it to a tangential direction, and makes a geodesic
    hop of at most ``max_step_rad`` that way.  This is mean shift, so it
    converges on the density mode rather than on whichever precomputed
    grid point happened to be nearest.

    ``max_step_rad`` caps one hop and ``max_total_rad`` caps the whole
    excursion from ``point``.  The total budget is the one that matters:
    without it, twelve iterations of a 6 degree cap let a "local" snap
    travel 72 degrees, which showed up as library entries 38 degrees apart
    where 6 was asked for.

    ``forbid_direction`` (a unit tangent) is projected out of every hop.
    That is what makes this a *ridge* correction instead of a mode
    collapse: when walking along a ridge, the point must be free to slide
    sideways onto the crest but must not be allowed to slide backwards
    along the crest into the peak it came from.  Subspace-constrained mean
    shift, in the usual principal-curve sense.

    Why this replaces a grid snap.  The old correction chose the densest
    point of a 4096-point Fibonacci sphere within half a step.  That grid
    has ~3.1 degree spacing, and the cone it searched had a 3 degree
    half-angle, so it held a MEDIAN OF ONE CANDIDATE: the correction was a
    no-op, the walk was an uncorrected geodesic extrapolation — a great
    circle, i.e. a straight line — and it drifted off a clean synthetic
    ridge by up to 8 degrees over 20 points while emitting samples spaced
    3.9-6.2 degrees apart instead of the 6 requested.  Raising the grid
    resolution is not a fix: matching 3 degrees on S^3 needs ~260k points
    and on S^4 ~16M, so the grid is also exactly what capped the whole
    method at K=3.  Mean shift is continuous (no resolution floor) and
    dimension-agnostic (no grid), which is why one change fixes both.
    """
    cur = point
    for _ in range(n_iter):
        w = _kernel_weights(cur, data, sigma2)
        total = float(w.sum())
        if total <= 0.0:
            break
        mean = (w @ data) / total
        # Tangential component of the weighted mean: the uphill direction.
        v = mean - cur * float(cur @ mean)
        if forbid_direction is not None:
            v = v - forbid_direction * float(forbid_direction @ v)
        nv = float(np.linalg.norm(v))
        if nv < tol:
            break
        along = float(cur @ mean)
        # arctan2 gives the geodesic angle to the shifted mean; when the mean
        # sits behind us (along <= 0) the kernel has caught a far-off lobe, so
        # take the capped step rather than trusting the angle.
        angle = np.arctan2(nv, along) if along > 0.0 else max_step_rad
        angle = min(angle, max_step_rad)
        nxt = cur * np.cos(angle) + (v / nv) * np.sin(angle)
        nxt /= np.linalg.norm(nxt)
        travelled = float(np.arccos(np.clip(nxt @ point, -1.0, 1.0)))
        if travelled > max_total_rad:
            # Spend exactly the remaining budget along the same geodesic and
            # stop: past here we are no longer correcting, we are wandering.
            u = nxt - point * float(point @ nxt)
            nu = float(np.linalg.norm(u))
            if nu > tol:
                cur = point * np.cos(max_total_rad) + (u / nu) * np.sin(max_total_rad)
                cur /= np.linalg.norm(cur)
            break
        converged = float(nxt @ cur) > 1.0 - tol
        cur = nxt
        if converged:
            break
    return cur


def _local_ridge_direction(
    point: np.ndarray,
    data: np.ndarray,
    sigma2: float,
) -> np.ndarray:
    """Direction the density ridge runs at ``point``: local tangent-space PCA.

    Expresses the kernel-weighted neighbourhood in the tangent space at
    ``point`` and returns the top eigenvector of its weighted covariance,
    mapped back to the ambient space.  For K=3 the tangent space is a
    plane and this is the old behaviour; for K>3 it is a (K-1)-space and
    the same eigenvector is still "the way the ridge goes".
    """
    basis = _tangent_basis(point)  # (K, K-1)
    w = _kernel_weights(point, data, sigma2)
    total = float(w.sum())
    if total <= 0.0:
        return basis[:, 0]
    # Sources more than a few bandwidths away contribute nothing but cost.
    near = w > w.max() * 1e-6
    w = w[near]
    coords = data[near] @ basis  # (n_near, K-1)
    if coords.shape[0] < 3:
        return basis[:, 0]
    total = float(w.sum())
    mu = (w @ coords) / total
    centred = coords - mu
    cov = (centred * w[:, None]).T @ centred / total
    _, evecs = np.linalg.eigh(cov)
    direction = basis @ evecs[:, -1]  # largest eigenvalue last from eigh
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > 1e-12 else basis[:, 0]


def _walk_ridge_one_way(
    start: np.ndarray,
    tangent: np.ndarray,
    data: np.ndarray,
    sigma2: float,
    n_steps: int,
    step_rad: float,
    floor: float,
) -> list[np.ndarray]:
    """Predict-and-correct walk along a density ridge in ONE direction.

    Each step hops ``step_rad`` along the current tangent (the *predict*),
    then mean-shifts onto the ridge crest with motion along the direction
    of travel forbidden (the *correct*).  The tangent is then recomputed
    as the geodesic direction actually travelled, so it carries forward.

    Carrying the tangent is what makes this a directed walk.  An earlier
    implementation chose purely by density within an annulus and forbade
    only near-exact repeats, which let it fold back and re-traverse the
    same arc a few degrees off.  The turn guard below closes the same door
    for the mean-shift correction: a hairpin means the ridge ended and the
    walk is about to double back, so stop.

    Returns the points visited *excluding* ``start``.
    """
    out: list[np.ndarray] = []
    cur = start
    tan = tangent
    for _ in range(n_steps):
        # Predict a hop of `hop`, correct onto the crest, then re-predict once
        # with the hop rescaled by how far we actually got.  The correction is
        # perpendicular to the direction of travel, so on a curving ridge it
        # pulls the point back inside the geodesic and the achieved spacing
        # comes up short -- 3.8 degrees for a requested 6 on a tightly curved
        # arc.  One rescale is enough to land on the requested spacing, which
        # is what `angular_step_deg` promises the caller.
        hop = step_rad
        nxt = None
        for _attempt in range(2):
            target = cur * np.cos(hop) + tan * np.sin(hop)
            target /= np.linalg.norm(target)
            # Parallel-transport the tangent along the geodesic we just hopped
            # down, so the correction is forbidden from moving along the ridge
            # AT THE POINT IT ACTS ON rather than one step back.
            tan_at_target = -cur * np.sin(hop) + tan * np.cos(hop)
            nxt = _mean_shift(
                target,
                data,
                sigma2,
                forbid_direction=tan_at_target,
                max_step_rad=step_rad,
                max_total_rad=step_rad,
            )
            achieved = float(np.arccos(np.clip(cur @ nxt, -1.0, 1.0)))
            if achieved < 1e-9:
                break
            scale = float(np.clip(step_rad / achieved, 0.5, 2.0))
            if abs(scale - 1.0) < 0.02:
                break
            hop = hop * scale
        if nxt is None:
            break
        if _density_at(nxt, data, sigma2) < floor:
            break
        new_tan = nxt - cur * float(cur @ nxt)
        norm = float(np.linalg.norm(new_tan))
        if norm < 1e-12:
            break
        new_tan /= norm
        if float(new_tan @ tan) <= 0.0:
            break  # hairpin: the ridge ended and we are turning back
        tan = new_tan
        cur = nxt
        out.append(cur)
    return out


def trace_manifold_auto(
    unit_vectors: np.ndarray,
    n_points: int = 20,
    angular_step_deg: float = 6.0,
    bandwidth_deg: float = 8.0,
    density_floor_frac: float = 0.05,
) -> np.ndarray:
    """Trace a 1-D density ridge across the unit sphere, in any dimension.

    Two-sided predict-and-correct walk:

    1. Evaluate the spherical KDE (bandwidth ``bandwidth_deg``) at the
       voxel directions themselves and mean-shift the densest one to
       convergence, giving the density mode.
    2. Estimate the ridge's local tangent direction there
       (:func:`_local_ridge_direction`).
    3. Walk ``+tangent`` and ``-tangent`` away from the mode, splitting
       the ``n_points`` budget between the two arms; if one arm
       terminates early its remaining budget goes to the other.  Each
       step is a geodesic hop of ``angular_step_deg`` followed by a
       mean-shift back onto the crest (:func:`_walk_ridge_one_way`).
    4. Stop an arm when density drops below
       ``density_floor_frac × peak_density``.

    This automates what NSD did by hand — ``hrf_constructmanifold.m``
    has a human click 12 points on the (PC2, PC3) density heatmap and
    then great-circle-interpolates between them at 6° spacing.  The
    walk here is deliberately simple; for pathological densities use
    :func:`trace_manifold_from_points` to supply clicked points
    directly, exactly as NSD did.

    Any K >= 2.  The correction step used to snap to the nearest point of
    a Fibonacci 2-sphere, which both capped the method at K=3 and — with
    only ~1 grid point inside the snap cone — left the walk tracing an
    uncorrected great circle instead of the data's curve.  See
    :func:`_mean_shift` for the measurements.

    Parameters
    ----------
    unit_vectors : np.ndarray, shape (n_voxels, K)
        Unit-norm voxel directions from :func:`project_unit_sphere`.
    n_points : int, default 20
        Maximum number of manifold samples to emit.
    angular_step_deg : float, default 6.0
        Target angular spacing between successive samples, matching
        NSD's 6° parameterization.
    bandwidth_deg : float, default 8.0
        Width of the spherical KDE kernel.  Larger = smoother density
        but blurs across the manifold.
    density_floor_frac : float, default 0.05
        Stop walking when density drops below this fraction of the
        starting peak.

    Returns
    -------
    manifold : np.ndarray, shape (n_actual, K)
        Ordered manifold points on the unit sphere.
        ``n_actual <= n_points`` (early termination is allowed).

    Raises
    ------
    ValueError
        If ``unit_vectors`` has fewer than 2 columns.
    """
    if unit_vectors.shape[1] < 2:
        raise ValueError(
            f"trace_manifold_auto needs K>=2 (got K={unit_vectors.shape[1]}); "
            "a 1-D embedding has no manifold to trace."
        )

    data = _nonzero_rows(unit_vectors)
    sources = _kde_sources(data)
    sigma2 = np.deg2rad(bandwidth_deg) ** 2
    step_rad = np.deg2rad(angular_step_deg)

    # Seed at the densest voxel direction, then mean-shift it to the actual
    # mode: the densest *sample* is only within a nearest-neighbour spacing
    # of the mode, and the whole point of this rewrite is not to inherit a
    # discretization error from the thing we happen to evaluate on.
    seed_pool = _kde_sources(data, max_n=4096)
    seed_density = _spherical_kde(seed_pool, sources, np.deg2rad(bandwidth_deg))
    start = _mean_shift(
        seed_pool[int(seed_density.argmax())],
        sources,
        sigma2,
        max_step_rad=step_rad,
        max_total_rad=5.0 * np.deg2rad(bandwidth_deg),
    )
    floor = _density_at(start, sources, sigma2) * density_floor_frac

    tangent = _local_ridge_direction(start, sources, sigma2)

    # Split the budget either side of the peak.  Walk both arms with the
    # full remaining budget available to each, then trim: an arm that
    # dies early (ridge ends, density floor) hands its slots to the other.
    budget = n_points - 1
    n_fwd_target = budget // 2 + budget % 2
    forward = _walk_ridge_one_way(start, tangent, sources, sigma2, budget, step_rad, floor)
    backward = _walk_ridge_one_way(start, -tangent, sources, sigma2, budget, step_rad, floor)

    n_fwd = min(len(forward), n_fwd_target)
    n_bwd = min(len(backward), budget - n_fwd)
    n_fwd = min(len(forward), budget - n_bwd)  # reclaim slots the back arm left

    points = list(reversed(backward[:n_bwd])) + [start] + forward[:n_fwd]
    return np.stack(points)


def _nonzero_rows(unit_vectors: np.ndarray) -> np.ndarray:
    """Drop the zero rows :func:`project_unit_sphere` leaves for dead voxels."""
    data = unit_vectors[np.linalg.norm(unit_vectors, axis=1) > 0.5]
    if data.size == 0:
        raise ValueError("No non-zero unit vectors to sample.")
    return data


def _farthest_point_sample(candidates: np.ndarray, n: int, start: int) -> np.ndarray:
    """Greedy farthest-point (maximin) subset of ``candidates`` on the sphere.

    Deterministic given ``start``.  Each pick is the candidate whose angular
    distance to the already-picked set is largest, which spreads points
    evenly over whatever region the candidates occupy — no assumption that
    the region is 1-D.
    """
    n = min(n, candidates.shape[0])
    picked = [start]
    # Track each candidate's distance (1 - cos) to the nearest picked point.
    dist = 1.0 - candidates @ candidates[start]
    for _ in range(n - 1):
        nxt = int(dist.argmax())
        picked.append(nxt)
        dist = np.minimum(dist, 1.0 - candidates @ candidates[nxt])
    return candidates[picked]


def trace_manifold_blob(
    unit_vectors: np.ndarray,
    n_points: int = 20,
    bandwidth_deg: float = 8.0,
    density_floor_frac: float = 0.05,
    n_candidates: int = 4096,
) -> np.ndarray:
    """Cover the density blob's **support** evenly, instead of tracing a 1-D arc.

    A ridge walk assumes HRF-shape variation is essentially
    one-dimensional — a single family running from early to late.  That is
    a good description of NSD's data and of most datasets, but it is an
    assumption, and when the blob has genuine width the arc leaves the
    off-arc voxels poorly represented (see :func:`manifold_coverage`).

    This sampler makes no such assumption:

    1. Spherical-KDE density evaluated at the voxel directions themselves.
    2. Keep the directions whose density is at least
       ``density_floor_frac × peak`` — that set *is* the blob.
    3. Farthest-point-sample ``n_points`` of them, starting at the density
       peak, giving near-uniform angular coverage of the support.

    Candidates are voxel directions rather than points of a Fibonacci
    sphere.  That makes the sampler work at any K — a grid fine enough to
    resolve S^3 needs ~260k points and S^4 ~16M, so the grid was what
    pinned this to K=3 — and it also guarantees every candidate sits where
    voxels actually are, which a grid does not.

    The result is **not** an ordered curve — with a filled region there is
    no meaningful "next" entry — so neighbouring library indices need not
    be similar.  :func:`derive_library` still sorts the output by
    time-to-peak for a deterministic, interpretable order, but adjacency
    stops implying similarity the way it does for a ridge.

    Use this when the sphere-density QC plot shows a genuinely round or
    forked blob rather than a clean arc, or when
    :func:`manifold_coverage` reports a large tail of poorly-covered
    voxels under ``auto``.  Compare with :func:`trace_manifold_kmeans`,
    which fills the same region in proportion to voxel count rather than
    evenly.

    Parameters
    ----------
    unit_vectors : np.ndarray, shape (n_voxels, K)
        Unit-norm voxel directions from :func:`project_unit_sphere`.
    n_points : int, default 20
        Exact number of library entries to emit (unlike ``auto``, which
        may stop early when the ridge ends).
    bandwidth_deg : float, default 8.0
        Spherical KDE bandwidth.
    density_floor_frac : float, default 0.05
        Directions below this fraction of peak density are outside the
        blob.  Raise it to sample only the dense core; lower it to chase
        the tails.
    n_candidates : int, default 4096
        Cap on how many voxel directions are considered as candidates.
        Farthest-point sampling is O(n_candidates × n_points), so this
        bounds the cost on a whole-brain library.

    Returns
    -------
    manifold : np.ndarray, shape (n_actual, K)
        ``n_actual == min(n_points, #candidates above the floor)``.
    """
    if unit_vectors.shape[1] < 2:
        raise ValueError(f"trace_manifold_blob needs K>=2 (got K={unit_vectors.shape[1]}).")
    data = _nonzero_rows(unit_vectors)
    sources = _kde_sources(data)
    candidates = _kde_sources(data, max_n=n_candidates)

    density = _spherical_kde(candidates, sources, np.deg2rad(bandwidth_deg))
    peak_idx = int(density.argmax())
    inside = density >= density[peak_idx] * density_floor_frac
    kept = candidates[inside]
    if kept.shape[0] == 0:
        raise ValueError("No candidate directions above the density floor.")
    # Index of the density peak within the masked subset.
    start = int(density[inside].argmax())
    return _farthest_point_sample(kept, n_points, start)


def trace_manifold_kmeans(
    unit_vectors: np.ndarray,
    n_points: int = 20,
    n_iter: int = 100,
    seed: int = 42,
    tol: float = 1e-7,
) -> np.ndarray:
    """Spherical k-means on the voxel directions — density-**proportional** 2-D sampling.

    Where :func:`trace_manifold_blob` spreads entries evenly over the
    blob's *support*, this spreads them according to how many voxels
    actually live in each part of it: dense regions get more library
    entries, sparse fringes get fewer.  That is the right objective if
    you want to minimize the expected mismatch between a random voxel and
    its best library entry, since Lloyd's algorithm on cosine distance is
    directly minimizing exactly that quantity.

    Works for any K (not just 3), so it also covers the ``n_pcs != 3``
    case that ``auto`` cannot handle.

    Centroids are re-normalized to the sphere each iteration (spherical
    k-means); empty clusters are re-seeded to the worst-fit data point so
    the requested count is always returned.

    Parameters
    ----------
    unit_vectors : np.ndarray, shape (n_voxels, K)
        Unit-norm voxel directions.
    n_points : int, default 20
        Number of clusters, hence library entries.
    n_iter : int, default 100
        Maximum Lloyd iterations.
    seed : int, default 42
        Only used if the deterministic farthest-point init degenerates.
    tol : float, default 1e-7
        Stop when no centroid moves by more than this.

    Returns
    -------
    manifold : np.ndarray, shape (n_points, K)
        Unit-norm cluster centroids.
    """
    data = _nonzero_rows(unit_vectors)
    n_points = min(n_points, data.shape[0])

    # Deterministic init: farthest-point from the medoid-ish direction.
    mean_dir = data.mean(axis=0)
    norm = np.linalg.norm(mean_dir)
    if norm < 1e-12:
        rng = np.random.default_rng(seed)
        start = int(rng.integers(data.shape[0]))
    else:
        start = int(np.argmax(data @ (mean_dir / norm)))
    centroids = _farthest_point_sample(data, n_points, start).copy()

    for _ in range(n_iter):
        assign = np.argmax(data @ centroids.T, axis=1)
        new = np.zeros_like(centroids)
        for k in range(n_points):
            members = data[assign == k]
            if members.shape[0] == 0:
                # Empty cluster: re-seed to the point currently worst served.
                worst = int(np.argmin(np.max(data @ centroids.T, axis=1)))
                new[k] = data[worst]
                continue
            s = members.sum(axis=0)
            n_s = np.linalg.norm(s)
            new[k] = s / n_s if n_s > 1e-12 else centroids[k]
        shift = float(np.abs(new - centroids).max())
        centroids = new
        if shift < tol:
            break
    return centroids


def manifold_coverage(unit_vectors: np.ndarray, manifold: np.ndarray) -> dict:
    """How well does this library represent the voxels it came from?

    For every voxel, the angle to its nearest library entry.  Because the
    PCs are orthonormal and both vectors are unit-norm, the **cosine of
    that angle is exactly the (uncentered) correlation between the two
    reconstructed HRF waveforms** — ``<w1·PCs, w2·PCs> = w1·w2``.  So this
    is a direct statement about HRF shape mismatch, not a proxy for one.

    This is the number to look at when deciding between a 1-D ridge
    (``-manifold auto``) and 2-D coverage (``blob`` / ``kmeans``): a large
    p90 or max means a substantial population of voxels whose shape no
    library entry comes close to matching.

    Returns a dict with ``angles_deg`` (per voxel) and the summary keys
    ``median_deg``, ``p90_deg``, ``max_deg``, ``median_shape_r``,
    ``p10_shape_r`` (the 10th percentile of shape correlation — the
    poorly-served tail).
    """
    data = _nonzero_rows(unit_vectors)
    best_cos = np.clip((data @ manifold.T).max(axis=1), -1.0, 1.0)
    angles = np.degrees(np.arccos(best_cos))
    return {
        "angles_deg": angles,
        "median_deg": float(np.median(angles)),
        "p90_deg": float(np.percentile(angles, 90)),
        "max_deg": float(angles.max()),
        "median_shape_r": float(np.median(best_cos)),
        "p10_shape_r": float(np.percentile(best_cos, 10)),
    }


def trace_manifold_grid(unit_vectors: np.ndarray, n_points: int = 20) -> np.ndarray:
    """Order voxels along the first principal axis, sample evenly.

    Despite the name this is **not** a grid — it is a second 1-D path,
    kept because it is deterministic and density-agnostic and works for
    any K.  Projects ``unit_vectors`` onto their first PCA direction,
    sorts by that coordinate, and picks ``n_points`` evenly spaced
    percentiles.  The picked vectors are re-normalized to unit length.

    For actual 2-D coverage of the blob use :func:`trace_manifold_blob`
    or :func:`trace_manifold_kmeans`.

    Parameters
    ----------
    unit_vectors : np.ndarray, shape (n_voxels, K)
    n_points : int, default 20

    Returns
    -------
    manifold : np.ndarray, shape (n_points, K)
    """
    nz = np.linalg.norm(unit_vectors, axis=1) > 0.5
    data = unit_vectors[nz]
    if data.shape[0] < n_points:
        raise ValueError(f"Need at least n_points={n_points} unit vectors; got {data.shape[0]}")
    mean = data.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(data - mean, full_matrices=False)
    axis = vt[0]  # primary direction (K,)
    proj = data @ axis
    order = np.argsort(proj)
    pct = np.linspace(0, len(order) - 1, n_points).astype(int)
    picked = data[order[pct]]
    picked = picked / np.linalg.norm(picked, axis=1, keepdims=True)
    return picked


def trace_manifold_from_points(
    points: np.ndarray,
) -> np.ndarray:
    """Pass-through for user-supplied manifold points.

    Re-normalizes each row to unit length and returns as-is.  Use this
    when the user supplies a JSON of clicked points (the
    ``-manifold-points`` CLI flag).

    Parameters
    ----------
    points : np.ndarray, shape (n_points, K)
        User-supplied K-D vectors.  Need not be unit-norm.

    Returns
    -------
    np.ndarray, shape (n_points, K)
        Unit-norm rows.
    """
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"points must be 2D (n_points, K); got shape {arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if (norms < 1e-12).any():
        raise ValueError("Supplied manifold points contain zero-norm rows.")
    return arr / norms


# ----------------------------------------------------------------------------
# NSD refinement: build PC-basis design from onsets & PCs
# ----------------------------------------------------------------------------


def build_pc_basis_design_per_run(
    onsets_per_run: list[np.ndarray],
    pcs: np.ndarray,
    lag_times: np.ndarray,
    tr: float,
    n_timepoints_per_run: list[int],
    basis: str = "FIR",
) -> list[np.ndarray]:
    """Build per-run PC-basis design (for NSD's refit step).

    Given the K temporal PCs derived from the FIR/TENT betas, treat
    each PC as a candidate HRF and convolve the onset train with each
    PC.  The resulting per-run design has K task columns (one per PC),
    so a subsequent GLM fit produces per-voxel coefficients that
    directly measure how much each PC contributes to that voxel's
    response — exactly NSD's ``modelmd`` array used by
    ``hrf_constructmanifold.m`` to place each voxel on the unit sphere.

    For TR-locked onsets (``basis="FIR"``) we sample each PC at integer
    TR offsets and convolve the binary onset vector with the PC.  For
    sub-TR onsets (``basis="TENT"``/``"TENTzero"``) we evaluate each PC
    at the *exact* (TR_time − onset_time) for every (TR, onset) pair —
    same logic as ``make_tent_design`` — and sum across onsets.  Both
    paths take linear interpolation of the PC between knots so that
    fractional-TR lag values are handled.

    Parameters
    ----------
    onsets_per_run : list[np.ndarray]
        One array per run, each containing onset times in seconds
        relative to the run start (the pooled-condition onsets for
        the group).
    pcs : np.ndarray, shape (K, n_lags)
        Temporal PCs at lag_times resolution.
    lag_times : np.ndarray, shape (n_lags,)
        Lag-time grid for the PCs (seconds).
    tr : float
        Repetition time in seconds.
    n_timepoints_per_run : list[int]
        Number of TR timepoints in each run.
    basis : {"FIR", "TENT", "TENTzero"}
        Whether onsets should be quantized to TR (FIR) or evaluated at
        the exact onset time (TENT family).

    Returns
    -------
    designs : list[np.ndarray]
        One ``(n_run_tp, K)`` matrix per run.

    Notes
    -----
    This is numpy-only (no torch) — the matrices are tiny (n_tp × 3
    columns) so torch's batch primitives buy nothing.  The CLI converts
    to torch when handing the result to ``fit_glm``.
    """
    K, n_lags = pcs.shape
    if lag_times.shape[0] != n_lags:
        raise ValueError(
            f"lag_times length ({lag_times.shape[0]}) must equal pcs.shape[1] ({n_lags})"
        )
    if len(onsets_per_run) != len(n_timepoints_per_run):
        raise ValueError(
            f"onsets_per_run ({len(onsets_per_run)} runs) and "
            f"n_timepoints_per_run ({len(n_timepoints_per_run)} runs) "
            "must match in length"
        )

    designs: list[np.ndarray] = []
    src_lo, src_hi = float(lag_times[0]), float(lag_times[-1])

    for onset_times, n_run_tp in zip(onsets_per_run, n_timepoints_per_run, strict=False):
        block = np.zeros((n_run_tp, K), dtype=np.float64)
        if onset_times.size == 0:
            designs.append(block)
            continue

        tr_times = np.arange(n_run_tp, dtype=np.float64) * tr

        if basis == "FIR":
            # TR-locked path: round onsets to nearest TR, then convolve
            # the binary onset vector with each PC sampled at TR.  We
            # interpolate the PCs onto integer-TR lags (lag_times are
            # already TR-spaced when basis="FIR", but they may not start
            # at exactly 0 or be precisely tr-spaced; interpolate to be
            # safe).
            idx = np.round(onset_times / tr).astype(int)
            idx = idx[(idx >= 0) & (idx < n_run_tp)]
            onset_vec = np.zeros(n_run_tp, dtype=np.float64)
            onset_vec[idx] = 1.0
            # Resample each PC to integer-TR lags
            tr_lags = np.arange(n_lags) * tr
            for k in range(K):
                pc_at_tr = np.interp(tr_lags, lag_times, pcs[k])
                col = np.convolve(onset_vec, pc_at_tr)[:n_run_tp]
                block[:, k] = col
        else:
            # Sub-TR onsets: for every (TR_time, onset_time) pair,
            # evaluate PC at the relative lag (TR_time - onset_time)
            # via linear interpolation, summed across onsets.
            rel = tr_times[:, None] - onset_times[None, :]  # (n_run_tp, n_events)
            # Mask: only pairs where rel ∈ [src_lo, src_hi] contribute
            inside = (rel >= src_lo) & (rel <= src_hi)
            for k in range(K):
                # np.interp is 1-D; vectorize via flat-then-reshape.
                vals = np.zeros_like(rel)
                flat_rel = rel[inside]
                if flat_rel.size > 0:
                    vals[inside] = np.interp(flat_rel, lag_times, pcs[k])
                block[:, k] = vals.sum(axis=1)
        designs.append(block)

    return designs


# ----------------------------------------------------------------------------
# Duration deconvolution (placeholder — see module docstring)
# ----------------------------------------------------------------------------


def deconvolve_event_duration(
    library: np.ndarray,
    duration: float,
    dt: float,
    *,
    method: str = "wiener",
    snr: float = 100.0,
    normalize_peak: bool = True,
) -> np.ndarray:
    """Recover the impulse-response HRF from a duration-convolved library.

    The FIR/TENT betas (and any library reconstructed from them via
    SVD + manifold + interpolation) represent::

        h_obs(t) = h_imp(t) ⊛ box_D(t)

    where ``h_imp`` is the underlying impulse response and ``box_D``
    is the event-duration boxcar.  Downstream consumers (``ffs_hrfopt``
    et al.) re-convolve the library entry with the event boxcar at
    modelling time, so to avoid *double*-convolution the library entry
    must be the impulse response ``h_imp``.  This function inverts the
    duration convolution.

    Wiener deconvolution (default ``method="wiener"``):

    .. math::

        \\hat{H}_{imp}(\\omega) = H_{obs}(\\omega)
            \\frac{\\overline{BOX_D(\\omega)}}{|BOX_D(\\omega)|^2 + 1/\\mathrm{SNR}}

    The ``1/SNR`` regularizer prevents naive division-blow-up at
    frequencies where ``|BOX_D(ω)|`` is small (boxcars have spectral
    zeros at integer multiples of ``1/D``).  Higher SNR = sharper
    result but noisier; lower SNR = smoother but more biased.  100
    has worked well in practice for canonical-library smoothness.

    Limit cases:

    - ``duration <= dt``: impulse-like; we just return a copy of the
      library — deconvolution is a no-op (and would be numerically
      unstable since the boxcar collapses to a single sample).
    - HRF length ≫ duration: deconvolution sharpens the rising edge
      and shifts the peak slightly earlier; effect is mild.
    - duration ~ HRF length: deconvolution can substantially reshape
      the curve.  This is exactly the case we care about for block
      designs.

    Parameters
    ----------
    library : np.ndarray, shape (n_hrfs, n_t)
        Library of HRF waveforms at uniform ``dt`` spacing.  These
        are assumed to be ``h_imp ⊛ box_D``.  Caller passes the cubic
        reconstruction or the double-gamma fit; either works.
    duration : float
        Event duration ``D`` in seconds.
    dt : float
        Sample spacing of ``library`` in seconds (e.g. 0.1 for the
        canonical library TSV grid).
    method : {"wiener"}, default "wiener"
        Deconvolution algorithm.  Only Wiener is implemented; the
        argument is kept so future Toeplitz-ridge / per-event-fit
        variants can slot in without an API break.
    snr : float, default 100.0
        Signal-to-noise prior used by the Wiener filter.  Higher =
        more aggressive deconvolution (sharper but admits more noise).
        100 corresponds to a 1% noise floor — appropriate for the
        clean, already-smoothed library waveforms emitted by the
        SVD+manifold+pchip pipeline.
    normalize_peak : bool, default True
        If True, peak-normalize each output row to +1 (matching the
        canonical-library convention) and flip sign if the largest
        absolute value is negative.

    Returns
    -------
    impulse : np.ndarray, shape (n_hrfs, n_t)
        Deconvolved library entries.  Same shape and sampling as input.

    Raises
    ------
    NotImplementedError
        If ``method`` is anything other than ``"wiener"``.
    """
    if method != "wiener":
        raise NotImplementedError(f"Only method='wiener' is implemented; got {method!r}")
    if duration <= dt:
        # Impulse-like — convolving with a 1-sample boxcar is a no-op,
        # and dividing by its spectrum (≈ flat) would just amplify noise.
        return library.copy()

    n_hrfs, n_t = library.shape
    n_box = max(1, int(round(duration / dt)))

    # FFT length: at least n_t + n_box - 1 to make the *linear*
    # convolution we're inverting circular-safe; round up to a power
    # of two for FFT speed.
    n_lin = n_t + n_box - 1
    n_fft = 1 << (int(np.ceil(np.log2(n_lin))))

    # Boxcar at unit height (the FIR/TENT estimate is the response to
    # a height-1 boxcar by construction of how onset_vec was built and
    # how the data was scaled per-run; the only ambiguity is overall
    # amplitude, which we re-normalize at the end).
    box = np.zeros(n_fft, dtype=np.float64)
    box[:n_box] = 1.0
    BOX = np.fft.rfft(box)
    BOX_mag_sq = BOX.real**2 + BOX.imag**2
    # Wiener filter kernel in frequency domain.  Computed once, reused
    # for every library row.
    wiener_kernel = np.conj(BOX) / (BOX_mag_sq + 1.0 / snr)

    out = np.zeros_like(library, dtype=np.float64)
    for i in range(n_hrfs):
        h_padded = np.zeros(n_fft, dtype=np.float64)
        h_padded[:n_t] = library[i]
        H = np.fft.rfft(h_padded)
        h_imp = np.fft.irfft(H * wiener_kernel, n=n_fft)[:n_t]
        if normalize_peak:
            # Divide by the SIGNED max (NSD convention).  Never negate —
            # an upside-down curve here means the input was not HRF-like,
            # and flipping it hides that instead of surfacing it.
            pos_peak = float(np.max(h_imp))
            if pos_peak > 0:
                h_imp = h_imp / pos_peak
        out[i] = h_imp
    return out


# ----------------------------------------------------------------------------
# Reconstruction
# ----------------------------------------------------------------------------


def reconstruct_timecourses(
    manifold: np.ndarray,
    pcs: np.ndarray,
    lag_times: np.ndarray,
    target_dt: float = 0.1,
    target_duration: float | None = None,
    normalize: Literal["peak", "none"] = "peak",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct manifold-point HRFs at a fine time grid.

    Each manifold point ``w_i`` yields a TR-resolution waveform
    ``c_i = w_i @ pcs``.  We then **cubic-spline interpolate** that
    waveform from the FIR/TENT lag grid (``lag_times``) onto a uniform
    grid at ``target_dt``, and (optionally) normalize each to peak = 1.

    The output format — 0.1 s resolution, peak=1 — matches
    :data:`fastfuncstuff.design.hrf._HRF_LIBRARY_FILE` so the result is
    a drop-in replacement.

    Parameters
    ----------
    manifold : np.ndarray, shape (n_points, K)
        Manifold points (unit-norm or not — magnitude only affects sign
        of the reconstructed waveform).
    pcs : np.ndarray, shape (K, n_lags)
        Temporal PC basis from :func:`svd_decompose`.
    lag_times : np.ndarray, shape (n_lags,)
        Times in seconds corresponding to each FIR/TENT lag.  E.g. for
        a TR=1 s FIR with 30 lags this is ``[0, 1, 2, ..., 29]``.
    target_dt : float, default 0.1
        Output sampling interval in seconds.  Matches the native
        canonical-library resolution.
    target_duration : float, optional
        Output duration in seconds.  Defaults to ``lag_times[-1]``.
    normalize : {"peak", "none"}, default "peak"
        If ``"peak"``, divide each reconstructed waveform by its
        **signed** maximum so the positive peak is +1 — exactly NSD's
        ``tc0/max(tc0)`` and what ``getcanonicalhrflibrary.tsv`` stores.

    Returns
    -------
    waveforms : np.ndarray, shape (n_points, n_target)
        Reconstructed HRFs at ``target_dt`` resolution.
    target_times : np.ndarray, shape (n_target,)
        The time grid used (in seconds).
    valid : np.ndarray of bool, shape (n_points,)
        False for entries whose positive peak is not the dominant
        excursion (``max(h) <= max|h| × 0.5``) or is non-positive.  Such
        a curve is not an HRF — it is an inverted or undershoot-dominated
        edge of the manifold — and the caller should drop it rather than
        put it in a library.  Earlier versions normalized by the
        *absolute* peak and negated the curve when that peak was
        negative, which silently entered upside-down HRFs into the
        library; NSD never flips.

    Notes
    -----
    PCHIP (Piecewise Cubic Hermite Interpolating Polynomial,
    ``scipy.interpolate.PchipInterpolator``) is used because NSD uses
    it and because it is **monotonic** and does not overshoot the input
    samples.  Plain cubic splines with natural boundary conditions
    overshoot wildly past the last FIR lag, producing the ugly
    oscillating tails that early versions of ``ffs_librarian`` emitted.
    """
    if pcs.shape[0] != manifold.shape[1]:
        raise ValueError(f"manifold K ({manifold.shape[1]}) does not match pcs K ({pcs.shape[0]})")
    if pcs.shape[1] != lag_times.shape[0]:
        raise ValueError(
            f"pcs has {pcs.shape[1]} lags but lag_times has {lag_times.shape[0]} entries"
        )

    # Reconstruct at the original FIR/TENT grid first.
    waveforms_coarse = manifold @ pcs  # (n_points, n_lags)

    if target_duration is None:
        target_duration = float(lag_times[-1])
    n_target = int(np.floor(target_duration / target_dt)) + 1
    target_times = np.arange(n_target) * target_dt

    # Clip the target grid to the source domain to avoid spline
    # extrapolation, which can produce wild excursions past the last
    # FIR lag (the HRF tail).
    src_lo, src_hi = float(lag_times[0]), float(lag_times[-1])
    target_clipped = np.clip(target_times, src_lo, src_hi)

    out = np.zeros((manifold.shape[0], n_target), dtype=np.float64)
    valid = np.ones(manifold.shape[0], dtype=bool)
    for i, coarse in enumerate(waveforms_coarse):
        # PCHIP is monotonic and won't overshoot — important for HRFs
        # where the tail decays to zero and a CubicSpline natural BC
        # would oscillate wildly past the last lag.
        spline = PchipInterpolator(lag_times, coarse, extrapolate=False)
        fine = spline(target_clipped)
        # Replace NaNs from extrapolation (target outside [lag_times[0],
        # lag_times[-1]]) with the nearest endpoint value, then clamp
        # the post-window tail to 0 since the HRF should have decayed.
        if np.isnan(fine).any():
            fine = np.nan_to_num(fine, nan=0.0)
        # Force a zero tail past the last FIR lag — the FIR fit has
        # no information out there and the library convention is that
        # the HRF has returned to baseline.
        fine = np.where(target_times > src_hi, 0.0, fine)
        if normalize == "peak":
            pos_peak = float(np.max(fine))
            abs_peak = float(np.max(np.abs(fine)))
            # An HRF's positive lobe must dominate.  If it does not, this
            # manifold point is off the end of anything HRF-like; flag it
            # so the caller drops it (NSD's clicked path never strayed
            # this far, an automated ridge walk can).
            if abs_peak <= 0 or pos_peak <= 0.5 * abs_peak:
                valid[i] = False
            if pos_peak > 0:
                fine = fine / pos_peak
        out[i] = fine

    return out, target_times, valid


# ----------------------------------------------------------------------------
# Optional double-gamma parametric fit
# ----------------------------------------------------------------------------


def _double_gamma(
    t: np.ndarray,
    a1: float,
    b1: float,
    a2: float,
    b2: float,
    c: float,
) -> np.ndarray:
    """SPM-style double-gamma HRF, **un-normalized**.

    ``h(t) = gamma_pdf(t; a1, scale=b1) - c · gamma_pdf(t; a2, scale=b2)``

    Same parameterization as :func:`fastfuncstuff.design.hrf.get_canonical_hrf`
    so the fitted parameters are directly interpretable.
    """
    main = _gamma.pdf(t, a1, scale=b1)
    under = _gamma.pdf(t, a2, scale=b2)
    return main - c * under


def _double_gamma_boxcar(
    t: np.ndarray,
    a1: float,
    b1: float,
    a2: float,
    b2: float,
    c: float,
    amp: float,
    *,
    n_box: int,
) -> np.ndarray:
    """Double-gamma convolved with an ``n_box``-sample unit boxcar, scaled.

    The forward model NSD fits in ``hrf_fitspmhrftomanifold.m``::

        fun = @(pp) pp(7)*subscript(conv(spm_hrf(0.1,[pp(1:6) 50]), ...
                                          ones(30,1)),{...});

    i.e. the stimulus-duration boxcar sits *inside* the model, so the
    free parameters describe the impulse response even though the data
    being fit is the duration-convolved response.
    """
    imp = _double_gamma(t, a1, b1, a2, b2, c)
    conv = np.convolve(imp, np.ones(n_box, dtype=np.float64))[: t.size]
    return amp * conv


def fit_double_gamma_through_boxcar(
    timecourse: np.ndarray,
    duration: float,
    dt: float = 0.1,
    p0: tuple[float, float, float, float, float] = (6.0, 1.0, 16.0, 1.0, 1.0 / 6.0),
    bounds: tuple[tuple, tuple] | None = None,
    maxfev: int = 10000,
) -> tuple[np.ndarray, dict]:
    """Recover the impulse-response double-gamma from a duration-convolved curve.

    This is the NSD-faithful duration correction and the preferred one.
    Instead of inverting the boxcar convolution numerically (see
    :func:`deconvolve_event_duration`, which has to regularize away the
    boxcar's spectral zeros at multiples of ``1/duration``), we put the
    boxcar in the *forward* model and fit::

        timecourse ≈ amp · (double_gamma(θ) ⊛ box_D)

    The returned waveform is ``double_gamma(θ)`` — the impulse response —
    peak-normalized.  No ill-conditioned inverse is involved, and the
    parametric family does the regularizing.  ``hrf_fitspmhrftomanifold.m``
    does exactly this with ``lsqnonlin``.

    Parameters
    ----------
    timecourse : np.ndarray, shape (n,)
        The duration-convolved manifold curve (peak-normalized).
    duration : float
        Event duration ``D`` in seconds.  ``<= dt`` makes this equivalent
        to :func:`fit_double_gamma`.
    dt : float, default 0.1
        Sample spacing in seconds.
    p0, bounds, maxfev
        As :func:`fit_double_gamma`; ``p0``/``bounds`` cover the five
        shape parameters and the amplitude is appended automatically.

    Returns
    -------
    impulse : np.ndarray, shape (n,)
        Peak-normalized impulse-response HRF.  Falls back to the input
        ``timecourse`` if the fit fails.
    params : dict
        ``a1``, ``b1``, ``a2``, ``b2``, ``c``, ``amp``, ``fit_ok``,
        ``residual_rms`` (residual is against the *convolved* model, so
        it is comparable to the input curve).
    """
    n_box = max(1, int(round(duration / dt)))
    if n_box <= 1:
        return fit_double_gamma(timecourse, dt=dt, p0=p0, bounds=bounds, maxfev=maxfev)

    if bounds is None:
        bounds = (
            (2.0, 0.3, 6.0, 0.3, 0.0),
            (12.0, 5.0, 30.0, 5.0, 1.0),
        )
    # Append the free amplitude that absorbs the boxcar's gain (≈ n_box).
    lo = (*tuple(bounds[0]), 0.0)
    hi = (*tuple(bounds[1]), np.inf)
    seed = (*tuple(p0), 1.0 / n_box)
    t = np.arange(timecourse.size) * dt

    def model(tt, a1, b1, a2, b2, c, amp):
        return _double_gamma_boxcar(tt, a1, b1, a2, b2, c, amp, n_box=n_box)

    try:
        popt, _ = curve_fit(model, t, timecourse, p0=seed, bounds=(lo, hi), maxfev=maxfev)
        residual = float(np.sqrt(np.mean((model(t, *popt) - timecourse) ** 2)))
        impulse = _double_gamma(t, *popt[:5])
        pos_peak = float(np.max(impulse))
        if pos_peak <= 0 or not np.isfinite(pos_peak):
            raise ValueError("fitted impulse response has no positive peak")
        impulse = impulse / pos_peak
        return impulse, {
            "a1": float(popt[0]),
            "b1": float(popt[1]),
            "a2": float(popt[2]),
            "b2": float(popt[3]),
            "c": float(popt[4]),
            "amp": float(popt[5]),
            "duration_s": float(duration),
            "fit_ok": True,
            "residual_rms": residual,
        }
    except Exception as exc:  # noqa: BLE001 — silent fallback, caller decides
        return timecourse.copy(), {
            "a1": float("nan"),
            "b1": float("nan"),
            "a2": float("nan"),
            "b2": float("nan"),
            "c": float("nan"),
            "amp": float("nan"),
            "duration_s": float(duration),
            "fit_ok": False,
            "residual_rms": float("nan"),
            "error": repr(exc),
        }


def _bspline_basis(
    n_samples: int,
    dt: float,
    n_knots: int,
    degree: int = 3,
) -> np.ndarray:
    """Cubic B-spline basis on the lag grid: ``(n_samples, n_knots + degree - 1)``.

    Evenly spaced interior knots with the usual clamped end repetition, so
    the basis is a partition of unity across the whole window.
    """
    x = np.arange(n_samples, dtype=np.float64) * dt
    interior = np.linspace(x[0], x[-1], n_knots)
    knots = np.r_[[interior[0]] * degree, interior, [interior[-1]] * degree]
    return np.asarray(BSpline.design_matrix(x, knots, degree, extrapolate=True).todense())


def _penalized_ls_gcv(
    design: np.ndarray,
    y: np.ndarray,
    penalty: np.ndarray,
    lambdas: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Penalized least squares with the smoothing weight chosen by GCV.

    Returns ``(beta, lambda, effective_dof)``.  The basis is small (a dozen
    columns), so every candidate lambda is solved outright rather than by a
    rank-revealing decomposition -- the whole sweep costs less than one
    ``curve_fit`` call.
    """
    ata = design.T @ design
    aty = design.T @ y
    dtd = penalty.T @ penalty
    n = y.size

    best: tuple[float, np.ndarray, float, float] | None = None
    for lam in lambdas:
        mat = ata + lam * dtd
        try:
            beta = np.linalg.solve(mat, aty)
            # tr(H) with H = A (A'A + lam D'D)^-1 A'; trace is cyclic, and the
            # basis-sized form avoids ever forming the n x n hat matrix.
            edof = float(np.trace(np.linalg.solve(mat, ata)))
        except np.linalg.LinAlgError:
            continue
        resid = y - design @ beta
        rss = float(resid @ resid)
        denom = n - edof
        if denom <= 1e-6:
            continue
        gcv = n * rss / (denom**2)
        if not np.isfinite(gcv):
            continue
        if best is None or gcv < best[0]:
            best = (gcv, beta, float(lam), edof)

    if best is None:
        raise np.linalg.LinAlgError("no usable lambda in the GCV sweep")
    return best[1], best[2], best[3]


def smooth_with_penalized_spline(
    curve: np.ndarray,
    dt: float = 0.1,
    n_knots: int = 12,
    degree: int = 3,
    lambdas: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Penalized-spline smooth of an arbitrary curve, weight chosen by GCV.

    The bare smoother behind :func:`fit_spline_through_boxcar`, without any of
    that function's HRF-specific machinery -- no boxcar in the model, no peak
    normalization, and no "the positive lobe must dominate" check.  All three
    are wrong for a PC.  Peak normalization is meaningless for an orthonormal
    basis vector, and a PC's SIGN is arbitrary, so a validity test keyed on
    the positive lobe rejects a curve on a coin flip: on a real 3-PC set, PC3
    passes as written and fails negated.

    Returns ``(smoothed, params)`` with ``lambda`` and ``edof`` in the dict.
    Falls back to the input curve unchanged if the solve fails.
    """
    from fastfuncstuff.design.matrices import make_penalty_matrix

    n = curve.size
    if lambdas is None:
        lambdas = np.logspace(-6, 4, 40)
    try:
        basis = _bspline_basis(n, dt, n_knots, degree)
        penalty = make_penalty_matrix(basis.shape[1], order=2)
        beta, lam, edof = _penalized_ls_gcv(basis, curve, penalty, lambdas)
        return basis @ beta, {"lambda": float(lam), "edof": float(edof), "fit_ok": True}
    except Exception as exc:  # noqa: BLE001 — caller keeps the raw curve
        return curve.copy(), {
            "lambda": float("nan"),
            "edof": float("nan"),
            "fit_ok": False,
            "error": repr(exc),
        }


def fit_spline_through_boxcar(
    timecourse: np.ndarray,
    duration: float,
    dt: float = 0.1,
    n_knots: int = 12,
    lambdas: np.ndarray | None = None,
    degree: int = 3,
    min_column_weight: float = 0.15,
) -> tuple[np.ndarray, dict]:
    """Recover a **smooth** impulse response from a duration-convolved curve.

    The penalized-spline counterpart of
    :func:`fit_double_gamma_through_boxcar`, and the reason to reach for it
    is what it does *not* impose.  Both keep the stimulus boxcar inside the
    forward model::

        timecourse ≈ (B·β) ⊛ box_D

    so neither runs the ill-conditioned explicit inverse that
    :func:`deconvolve_event_duration` has to regularize (a boxcar has
    spectral zeros at every multiple of ``1/D``).  The difference is the
    family on the left.  A double-gamma regularizes by *shape*: roughly two
    or three effective degrees of freedom, and anything outside that family
    is discarded.  Measured against an impulse response carrying a late
    secondary bump, the double-gamma recovers it at r=0.89 and this at
    r=0.99.  Against a true double-gamma it costs almost nothing: r=0.993
    to 0.9999.  This regularizes by *smoothness* instead -- any curve is
    reachable, high-frequency wiggle is what gets penalized, and the
    strength is picked per curve by GCV rather than assumed.

    The returned waveform is the impulse response, peak-normalized -- a
    drop-in library entry for consumers that re-convolve with the event
    boxcar at modelling time.

    Identifiability, and why late basis functions are dropped
    ---------------------------------------------------------
    Convolution with a ``D``-second boxcar means the curve only ever
    reports ``h(τ) - h(τ-D)``, so ``h`` is determined only up to a
    ``D``-periodic component -- the boxcar's spectral zeros, seen in the
    time domain.  Over a window ``T`` the late impulse response is
    therefore barely constrained: with ``T=36`` and ``D=20`` the final
    convolved design column carries 4% of the weight of the strongest.
    A double-gamma survives this because its family forbids the ambiguous
    component outright; a free basis does not, and the second-difference
    penalty is no help because its null space is exactly
    {constant, linear} -- a linear ramp off to 40% of peak in the tail
    costs zero penalty.  Adding a ridge component was measured and does not
    fix it either; the information is simply absent.

    So the unidentifiable directions are dropped rather than regularized:
    any basis function whose *convolved* design column falls below
    ``min_column_weight`` of the strongest is removed, and ``h`` decays to
    zero across the ones that remain.  Measured over noise levels 0.01 to
    0.08 on the curve, this takes the worst-case tail excursion from 0.54
    of peak to exactly 0, at equal or better recovery of the true impulse.
    Set ``min_column_weight=0`` to keep every basis function and see the
    unconstrained fit.

    Parameters
    ----------
    timecourse : np.ndarray, shape (n,)
        The duration-convolved manifold curve (peak-normalized).
    duration : float
        Event duration ``D`` in seconds.  ``<= dt`` fits the curve
        directly, with no boxcar in the model and nothing to drop.
    dt : float, default 0.1
        Sample spacing in seconds.
    n_knots : int, default 12
        Evenly spaced knots across the window.  More knots buy resolution
        and do not by themselves cause overfitting -- the penalty, not the
        knot count, controls smoothness -- but they do cost conditioning.
    lambdas : np.ndarray, optional
        Smoothing weights to search.  Defaults to 40 points log-spaced
        over ``1e-6 .. 1e4``.
    degree : int, default 3
        Spline degree; 3 is cubic.
    min_column_weight : float, default 0.15
        Drop basis functions whose convolved design column norm is below
        this fraction of the largest.  See above.

    Returns
    -------
    impulse : np.ndarray, shape (n,)
        Peak-normalized impulse-response HRF.  Falls back to the input
        ``timecourse`` if the fit fails or produces no dominant positive
        peak.
    params : dict
        ``lambda``, ``edof`` (effective degrees of freedom actually used),
        ``n_knots``, ``n_basis_kept`` / ``n_basis_total`` (how many basis
        functions survived the identifiability drop), ``duration_s``,
        ``fit_ok``, ``residual_rms`` (against the *convolved* model, so it
        is comparable to the input curve).
    """
    n = timecourse.size
    if lambdas is None:
        lambdas = np.logspace(-6, 4, 40)

    try:
        # Local import: design.matrices pulls torch at module scope, and this
        # module is documented torch-free so the CLIs can reach it cheaply.
        # By the time a spline fit runs, the caller has torch loaded anyway.
        from fastfuncstuff.design.matrices import make_penalty_matrix

        basis = _bspline_basis(n, dt, n_knots, degree)
        # h(0) = 0: the clamped basis has exactly one function equal to 1 at
        # the left edge, so dropping it removes the response-before-stimulus.
        basis = basis[:, 1:]
        n_basis_total = basis.shape[1]

        n_box = max(1, int(round(duration / dt)))
        if n_box <= 1:
            design = basis
        else:
            design = np.stack(
                [np.convolve(basis[:, j], np.ones(n_box))[:n] for j in range(basis.shape[1])],
                axis=1,
            )

        if min_column_weight > 0:
            col = np.linalg.norm(design, axis=0)
            keep = col >= min_column_weight * col.max()
            if keep.sum() >= degree + 2:
                basis = basis[:, keep]
                design = design[:, keep]

        penalty = make_penalty_matrix(basis.shape[1], order=2)
        beta, lam, edof = _penalized_ls_gcv(design, timecourse, penalty, lambdas)

        impulse = basis @ beta
        peak = float(np.max(impulse))
        if peak <= 0 or not np.isfinite(peak):
            raise ValueError("fitted impulse response has no positive peak")
        if peak <= 0.5 * float(np.max(np.abs(impulse))):
            raise ValueError("fitted impulse response is not peak-dominated")
        residual = float(np.sqrt(np.mean((design @ beta - timecourse) ** 2)))
        return impulse / peak, {
            "lambda": float(lam),
            "edof": float(edof),
            "n_knots": int(n_knots),
            "n_basis_kept": int(basis.shape[1]),
            "n_basis_total": int(n_basis_total),
            "duration_s": float(duration),
            "fit_ok": True,
            "residual_rms": residual,
        }
    except Exception as exc:  # noqa: BLE001 — silent fallback, caller decides
        return timecourse.copy(), {
            "lambda": float("nan"),
            "edof": float("nan"),
            "n_knots": int(n_knots),
            "n_basis_kept": 0,
            "n_basis_total": 0,
            "duration_s": float(duration),
            "fit_ok": False,
            "residual_rms": float("nan"),
            "error": repr(exc),
        }


def spline_prediction_through_boxcar(
    impulse: np.ndarray,
    duration: float,
    dt: float = 0.1,
) -> np.ndarray:
    """Push a fitted impulse response back through the boxcar, peak-normalized.

    The duration-convolved QC counterpart of a
    :func:`fit_spline_through_boxcar` result, so the "gamma fit --
    duration-convolved" panel shows the same fit the library entry came
    from rather than an independently fitted curve.
    """
    n_box = max(1, int(round(duration / dt)))
    if n_box <= 1:
        pred = impulse.copy()
    else:
        pred = np.convolve(impulse, np.ones(n_box))[: impulse.size]
    peak = float(np.max(pred))
    return pred / peak if peak > 0 else pred


def fit_double_gamma(
    timecourse: np.ndarray,
    dt: float = 0.1,
    normalize_peak: bool = True,
    p0: tuple[float, float, float, float, float] = (6.0, 1.0, 16.0, 1.0, 1.0 / 6.0),
    bounds: tuple[tuple, tuple] | None = None,
    maxfev: int = 5000,
) -> tuple[np.ndarray, dict]:
    """Fit a single double-gamma to a reconstructed HRF curve.

    Wraps :func:`scipy.optimize.curve_fit` over the SPM double-gamma
    family with sensible bounds.  Used by ``ffs_librarian`` to smooth
    the raw reconstructions into a parametric library that matches
    ``getcanonicalhrflibrary.tsv``.

    For a duration-convolved input, prefer
    :func:`fit_double_gamma_through_boxcar`, which recovers the impulse
    response in one step.

    If the fit fails (no convergence, NaNs), returns the input
    ``timecourse`` unchanged and ``params["fit_ok"] = False`` so the
    caller can fall back to the raw reconstruction.  This is
    deliberately silent — the caller decides whether to warn.

    Parameters
    ----------
    timecourse : np.ndarray, shape (n,)
        HRF samples to fit.  Should already be peak-normalized.
    dt : float, default 0.1
        Sampling interval in seconds; only used to build the time axis.
    normalize_peak : bool, default True
        If True, re-normalize the fitted curve to peak = 1 (the input
        was already peak-1, but the fit may shift the peak slightly).
    p0 : tuple, default ``(6, 1, 16, 1, 1/6)``
        Initial guess for ``(a1, b1, a2, b2, c)`` — SPM defaults.
    bounds : ((lo,…), (hi,…)), optional
        Override the default parameter bounds.  Defaults clamp the gamma
        *shape* parameters ``a1 ∈ [2, 12]`` and ``a2 ∈ [6, 30]``, the
        scales ``b1, b2 ∈ [0.3, 5]``, and the undershoot ratio
        ``c ∈ [0, 1]``.  Note these bound the shape parameter, not the
        time-to-peak — TTP is ``(a1 - 1) · b1``, so the admissible peak
        range is roughly 0.3–55 s.
    maxfev : int, default 5000
        Max function evaluations for curve_fit.

    Returns
    -------
    fitted : np.ndarray, shape (n,)
        The fitted (or fallback) waveform.
    params : dict
        Keys: ``a1``, ``b1``, ``a2``, ``b2``, ``c``, ``fit_ok``,
        ``residual_rms``.
    """
    if bounds is None:
        bounds = (
            (2.0, 0.3, 6.0, 0.3, 0.0),  # lower
            (12.0, 5.0, 30.0, 5.0, 1.0),  # upper
        )
    t = np.arange(timecourse.size) * dt

    try:
        popt, _ = curve_fit(_double_gamma, t, timecourse, p0=p0, bounds=bounds, maxfev=maxfev)
        fitted = _double_gamma(t, *popt)
        if normalize_peak:
            peak = float(np.max(fitted))
            if peak > 0:
                fitted = fitted / peak
        residual = float(np.sqrt(np.mean((fitted - timecourse) ** 2)))
        return fitted, {
            "a1": float(popt[0]),
            "b1": float(popt[1]),
            "a2": float(popt[2]),
            "b2": float(popt[3]),
            "c": float(popt[4]),
            "fit_ok": True,
            "residual_rms": residual,
        }
    except Exception as exc:  # noqa: BLE001 — silent fallback
        return timecourse.copy(), {
            "a1": float("nan"),
            "b1": float("nan"),
            "a2": float("nan"),
            "b2": float("nan"),
            "c": float("nan"),
            "fit_ok": False,
            "residual_rms": float("nan"),
            "error": repr(exc),
        }


# ----------------------------------------------------------------------------
# Multi-subject aggregation
# ----------------------------------------------------------------------------


def stack_subject_betas(
    per_subject_betas: list[np.ndarray],
    per_subject_r2: list[np.ndarray],
    *,
    r2_threshold: float = 0.10,
    per_subject_voxels: int = 20_000,
    seed: int = 42,
    equalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-stack several subjects' FIR betas into one matrix for the SVD.

    NSD's ``hrf_derivecanonicalpcs.m`` samples a **fixed count per
    subject** — "choose a random set of 20000 from each subject [subjects
    are contributing equally]" — rather than pooling all supra-threshold
    voxels and sampling globally.  That matters: their per-subject
    supra-threshold counts ranged from 2834 to 17387, so a global sample
    would have let the highest-SNR subject supply six times more rows
    than the lowest and dominate the PCs.  NSD sampled *with* replacement
    to hit 20000 even for subjects that had fewer; we upsample the same
    way when ``equalize=True``.

    Parameters
    ----------
    per_subject_betas : list of (n_voxels_i, n_lags)
        One FIR beta matrix per subject.  ``n_lags`` must match across
        subjects — i.e. same TR and same FIR window.
    per_subject_r2 : list of (n_voxels_i,)
        Matching R² vectors.
    r2_threshold : float, default 0.10
        Per-subject R² gate, applied before sampling.
    per_subject_voxels : int, default 20000
        Rows to draw from each subject.
    seed : int, default 42
        Base RNG seed; subject *i* uses ``seed + i`` so subjects are
        independent but the whole thing stays reproducible.
    equalize : bool, default True
        Draw exactly ``per_subject_voxels`` rows per subject, sampling
        with replacement where a subject has fewer (NSD behaviour).  When
        False, take everything that passes the gate, capped at
        ``per_subject_voxels`` — subjects then contribute unequally.

    Returns
    -------
    betas : np.ndarray, (n_total, n_lags)
        Row-stacked selected betas, subject-major.
    r2 : np.ndarray, (n_total,)
        Matching R² values, so the stack can flow through the ordinary
        single-subject path unchanged.
    subject_ids : np.ndarray, (n_total,)
        Which subject each row came from — kept for per-subject QC
        (e.g. colouring the unit-sphere histogram by subject to check
        that no one subject owns a lobe of the manifold).
    """
    if len(per_subject_betas) != len(per_subject_r2):
        raise ValueError(
            f"{len(per_subject_betas)} beta matrices but {len(per_subject_r2)} R² vectors"
        )
    if not per_subject_betas:
        raise ValueError("No subjects supplied.")
    n_lags = per_subject_betas[0].shape[1]
    for i, b in enumerate(per_subject_betas):
        if b.shape[1] != n_lags:
            raise ValueError(
                f"Subject {i} has {b.shape[1]} FIR lags but subject 0 has {n_lags}. "
                "All subjects must share a TR and FIR window to share a basis."
            )
        if b.shape[0] != per_subject_r2[i].shape[0]:
            raise ValueError(
                f"Subject {i}: {b.shape[0]} beta rows but {per_subject_r2[i].shape[0]} R² values."
            )

    betas_out: list[np.ndarray] = []
    r2_out: list[np.ndarray] = []
    ids_out: list[np.ndarray] = []
    for i, (b, r) in enumerate(zip(per_subject_betas, per_subject_r2, strict=True)):
        sel = select_library_voxels(
            b, r, threshold=r2_threshold, max_voxels=per_subject_voxels, seed=seed + i
        )
        if sel.size == 0:
            raise ValueError(
                f"Subject {i}: no voxels passed R² > {r2_threshold} with non-trivial "
                "FIR betas.  Exclude this subject or lower the threshold."
            )
        if equalize and sel.size < per_subject_voxels:
            rng = np.random.default_rng(seed + i)
            sel = rng.choice(sel, size=per_subject_voxels, replace=True)
        betas_out.append(b[sel])
        r2_out.append(r[sel])
        ids_out.append(np.full(sel.size, i, dtype=np.int32))

    return (
        np.concatenate(betas_out, axis=0),
        np.concatenate(r2_out, axis=0),
        np.concatenate(ids_out, axis=0),
    )


# ----------------------------------------------------------------------------
# High-level orchestration
# ----------------------------------------------------------------------------


@dataclass
class LibraryResult:
    """All artifacts of one HRF library derivation run.

    Returned by :func:`derive_library`.  The CLI writes the relevant
    pieces to TSV/JSON.
    """

    raw: np.ndarray  # (n_hrfs, n_target) — pchip recon BEFORE deconv
    target_times: np.ndarray  # (n_target,)
    manifold: np.ndarray  # (n_hrfs, n_pcs) — points on the unit sphere
    svd: SVDResult
    selected_voxels: np.ndarray  # indices into the original FIR betas
    fitted: np.ndarray | None = None  # (n_hrfs, n_target) — double-gamma fits, or None
    raw_deconvolved: np.ndarray | None = (
        None  # (n_hrfs, n_target) — pchip recon AFTER duration deconv (impulse)
    )
    fitted_deconvolved: np.ndarray | None = (
        None  # (n_hrfs, n_target) — double-gamma fit OF the impulse curves
    )
    gamma_params: list[dict] = field(default_factory=list)
    gamma_params_deconvolved: list[dict] = field(default_factory=list)
    # Manifold points whose reconstruction had no dominant positive peak
    # and were therefore dropped rather than entered into the library.
    n_dropped_invalid: int = 0
    # manifold_coverage() of the final entries against the selected voxels:
    # angle (== shape correlation) from each voxel to its best library entry.
    coverage: dict | None = None
    reconvolution_r: np.ndarray | None = None  # per-entry, library ⊛ boxcar vs `raw`
    reconvolution_leakage: np.ndarray | None = None  # per-entry, |conv| energy past the window
    # Provenance for the future deconvolution step (see module docstring).
    # event_durations is the per-group event-duration metadata (seconds);
    # duration_convolved=True means the library entries are the duration-
    # convolved response, NOT the impulse response.  A future
    # ``deconvolve_event_duration`` pass would flip this to False.
    event_durations: np.ndarray | None = None
    duration_convolved: bool = True
    shape_model: str = "none"  # which family produced `fitted` / `fitted_deconvolved`
    # QC artifacts.  These exist so the caller can write inspection
    # plots and TSVs without re-running the SVD or the projection.
    mean_fir_hrf: np.ndarray | None = None  # (n_lags,) — pooled task HRF
    unit_sphere_points: np.ndarray | None = None  # (n_selected, n_pcs)
    sphere_hist2d: np.ndarray | None = None  # (n_bins, n_bins) if n_pcs=3
    sphere_hist_edges: np.ndarray | None = None  # (n_bins+1,) for both axes


def derive_library(
    betas: np.ndarray,
    r2: np.ndarray,
    lag_times: np.ndarray,
    *,
    n_pcs: int = 3,
    n_hrfs: int = 20,
    r2_threshold: float = 0.10,
    max_voxels: int = 20_000,
    angular_step_deg: float = 6.0,
    bandwidth_deg: float = 8.0,
    target_dt: float = 0.1,
    target_duration: float | None = None,
    manifold_mode: Literal["auto", "blob", "kmeans", "grid", "points"] = "auto",
    manifold_points: np.ndarray | None = None,
    density_floor_frac: float = 0.05,
    fit_gamma: bool = True,
    shape_model: Literal["double", "spline"] = "double",
    spline_knots: int = 12,
    seed: int = 42,
    event_durations: np.ndarray | None = None,
    refit_weights: np.ndarray | None = None,
    deconvolve_duration: float | None = None,
    deconv_method: Literal["fit", "wiener"] = "fit",
    deconv_snr: float = 100.0,
    precomputed_svd: SVDResult | None = None,
    precomputed_selection: np.ndarray | None = None,
) -> LibraryResult:
    """End-to-end NSD-style HRF library derivation.

    This is the convenience wrapper used by ``ffs_librarian`` for the
    common case.  For finer control call the individual steps directly.

    Parameters
    ----------
    betas : np.ndarray, shape (n_voxels, n_lags)
        FIR/TENT per-voxel impulse response.  TR-resolution.
    r2 : np.ndarray, shape (n_voxels,)
        Per-voxel R² from the FIR/TENT fit.  Used for voxel selection.
    lag_times : np.ndarray, shape (n_lags,)
        Times (s) for each beta column.
    n_pcs : int, default 3
        Temporal PCs to retain.  3 = NSD default.
    n_hrfs : int, default 20
        Target library size (manifold sample count).
    r2_threshold, max_voxels, seed
        Forwarded to :func:`select_voxels`.
    angular_step_deg, bandwidth_deg
        Forwarded to :func:`trace_manifold_auto`.
    target_dt, target_duration
        Forwarded to :func:`reconstruct_timecourses`.  ``target_dt=0.1``
        matches the canonical library TSV.
    manifold_mode : {"auto", "blob", "kmeans", "grid", "points"}
        How to pick library entries out of the sphere density.

        - ``"auto"`` — 1-D density ridge (:func:`trace_manifold_auto`),
          NSD's model of HRF variation.  Ordered, adjacent entries are
          similar.  Any K.
        - ``"blob"`` — even coverage of the density region's support
          (:func:`trace_manifold_blob`).  Any K.
        - ``"kmeans"`` — coverage weighted by voxel density
          (:func:`trace_manifold_kmeans`); minimizes expected shape
          mismatch.  Any K.
        - ``"grid"`` — 1-D ordering along the first PCA axis (a legacy
          name; not actually a grid).
        - ``"points"`` — user-supplied, requires ``manifold_points``.

        The region-filling modes drop the "neighbouring index ⇒ similar
        HRF" property; check ``LibraryResult.coverage`` to see what they
        buy.
    density_floor_frac : float, default 0.05
        ``manifold_mode="blob"`` only — fraction of peak KDE density that
        still counts as inside the blob.
    manifold_points : np.ndarray, optional
        Required for ``manifold_mode="points"``.  Shape (n_hrfs, n_pcs);
        re-normalized to unit length by
        :func:`trace_manifold_from_points`.
    fit_gamma : bool, default True
        If True, fit a parametric shape family to every reconstructed
        waveform (which family is ``shape_model``).  Output ``fitted`` is
        filled; otherwise it is ``None`` (caller saves only ``raw``).
    shape_model : {"double", "spline"}, default "double"
        Which family, when ``fit_gamma`` is True.

        - ``"double"`` — SPM-style double-gamma.  NSD-faithful, maximally
          smooth, and the most noise-robust option; it also projects the
          library onto two or three effective shape degrees of freedom,
          which discards roughly a third of the library's angular spread
          when the data carries shape structure the family cannot express.
        - ``"spline"`` — penalized cubic B-spline
          (:func:`fit_spline_through_boxcar`), regularized by smoothness
          rather than by shape, with the strength chosen per curve by GCV.
          Reach for it when the point of raising ``n_pcs`` is to capture
          shape variation the double-gamma would flatten back out.  It
          costs noise robustness: measured against a true double-gamma
          under increasing noise on the curve, recovery falls to r=0.93
          where the double-gamma holds r=0.99.
    spline_knots : int, default 12
        ``shape_model="spline"`` only — knot count for the B-spline basis.
    deconvolve_duration : float, optional
        Event duration (s).  If given (and ``> target_dt``), correct the
        library entries so they represent the *impulse response* rather
        than the duration-convolved response.  This matters because
        downstream consumers re-convolve with the event boxcar at
        modelling time — without this step the design is doubly
        convolved.
    deconv_method : {"fit", "wiener"}, default "fit"
        How to do that correction.  ``"fit"`` is NSD-faithful: put the
        boxcar inside the double-gamma forward model and fit
        (:func:`fit_double_gamma_through_boxcar`), which needs no
        numerical inverse.  ``"wiener"`` runs the explicit deconvolution
        (:func:`deconvolve_event_duration`) and then gamma-fits the
        result — the only option when ``fit_gamma=False``, since there
        is no parametric family to carry the correction.
    deconv_snr : float, default 100.0
        Wiener filter SNR; ``deconv_method="wiener"`` only.
    precomputed_svd, precomputed_selection : optional
        Reuse an SVD and voxel selection the caller already computed
        (e.g. to derive the PCs for the NSD refit) instead of repeating
        them here.  Must be supplied together, and ``precomputed_svd``
        must have been computed on ``betas[precomputed_selection]``.
    refit_weights : np.ndarray, optional, shape (n_voxels, n_pcs)
        **NSD refinement.**  When provided, use these per-voxel
        coefficients (computed by the caller via a second GLM fit with
        the K PCs as design basis, see
        :func:`build_pc_basis_design_per_run`) **instead of** the SVD's
        own U·diag(S) loadings.  This matches NSD's
        ``hrf_constructmanifold.m`` flow: re-fit the data with the PCs
        as the design and use those direct-from-data loadings to place
        each voxel on the unit sphere.  The refit suppresses noise
        much better than projecting noisy FIR betas onto the PCs.
        Must be aligned with the FULL ``betas`` row order (not the
        post-selection order).

    Returns
    -------
    LibraryResult
        See dataclass.  ``result.raw`` is the always-emitted reconstruction;
        ``result.fitted`` is the double-gamma fit (or ``None``).
    """
    if (precomputed_svd is None) != (precomputed_selection is None):
        raise ValueError("precomputed_svd and precomputed_selection must be supplied together")

    if precomputed_selection is not None:
        sel = np.asarray(precomputed_selection)
    else:
        sel = select_library_voxels(
            betas, r2, threshold=r2_threshold, max_voxels=max_voxels, seed=seed
        )
    selected_betas = betas[sel]
    if sel.size < n_hrfs:
        raise ValueError(
            f"Only {sel.size} usable voxels survived selection "
            f"(R² > {r2_threshold}, FIR-betas non-trivial); need at least "
            f"n_hrfs={n_hrfs}.  Likely causes: the R² threshold is too high for "
            f"this data, or no brain mask was supplied and air voxels dominated "
            f"the selection.  Re-run with -mask <brain_mask.nii.gz> or lower "
            f"-r2-threshold."
        )

    # QC: pooled task HRF — mean of FIR betas across (post-filter) voxels.
    # This is the "average HRF in active cortex" sanity-check the user
    # can plot to confirm the FIR fit picked up something HRF-shaped
    # before any of the SVD machinery runs.
    mean_fir = selected_betas.mean(axis=0)

    if precomputed_svd is not None:
        svd = precomputed_svd
    else:
        svd = svd_decompose(selected_betas, n_pcs=n_pcs, unit_normalize=True, sign_align=True)

    if refit_weights is not None:
        # NSD refinement: replace the SVD's loadings (which are the
        # projection of NOISY FIR betas onto the PCs) with per-voxel
        # coefficients from a separate GLM fit that uses the PCs
        # themselves as the design basis.  ``hrf_constructmanifold.m``
        # uses these — they're cleaner because they're fit directly
        # against the data, not against the FIR estimates.  We expect
        # the array in the SAME ROW ORDER as the original ``betas``;
        # subselect to ``sel`` here.
        if refit_weights.shape != (betas.shape[0], n_pcs):
            raise ValueError(
                f"refit_weights shape {refit_weights.shape} does not match "
                f"(n_voxels={betas.shape[0]}, n_pcs={n_pcs})"
            )
        weights_for_sphere = refit_weights[sel]
    else:
        weights_for_sphere = svd.weights

    unit = project_unit_sphere(weights_for_sphere, sign_flip_by_first=True)

    # QC: 2D histogram on the (PC2, PC3) plane — this is NSD's
    # "unit-circle heatmap" used to visually pick the density ridge.  At
    # K>3 it is a projection of the cloud rather than the whole of it, but
    # a partial view beats the no view that gating this on K==3 gave.
    sphere_hist2d = None
    sphere_hist_edges = None
    if n_pcs >= 3:
        # Match NSD's bin grid (loadings ∈ [-1.5, 1.5], step 0.02).
        edges = np.arange(-1.5, 1.5 + 0.02, 0.02)
        sphere_hist2d, _, _ = np.histogram2d(unit[:, 1], unit[:, 2], bins=(edges, edges))
        sphere_hist_edges = edges

    if manifold_mode == "auto":
        manifold = trace_manifold_auto(
            unit,
            n_points=n_hrfs,
            angular_step_deg=angular_step_deg,
            bandwidth_deg=bandwidth_deg,
            density_floor_frac=density_floor_frac,
        )
    elif manifold_mode == "blob":
        manifold = trace_manifold_blob(
            unit,
            n_points=n_hrfs,
            bandwidth_deg=bandwidth_deg,
            density_floor_frac=density_floor_frac,
        )
    elif manifold_mode == "kmeans":
        manifold = trace_manifold_kmeans(unit, n_points=n_hrfs, seed=seed)
    elif manifold_mode == "grid":
        manifold = trace_manifold_grid(unit, n_points=n_hrfs)
    elif manifold_mode == "points":
        if manifold_points is None:
            raise ValueError("manifold_mode='points' requires manifold_points")
        manifold = trace_manifold_from_points(manifold_points)
    else:
        raise ValueError(f"Unknown manifold_mode: {manifold_mode}")

    raw, target_times, raw_valid = reconstruct_timecourses(
        manifold,
        svd.pcs,
        lag_times,
        target_dt=target_dt,
        target_duration=target_duration,
        normalize="peak",
    )

    # Drop manifold points that did not reconstruct to something HRF-like
    # (undershoot-dominated or wholly negative — see reconstruct_timecourses).
    n_invalid = int((~raw_valid).sum())
    if n_invalid:
        if raw_valid.sum() < 2:
            raise ValueError(
                f"All but {int(raw_valid.sum())} manifold points reconstructed to "
                "non-HRF-like curves (no dominant positive peak).  The PCs are "
                "probably not capturing HRF shape — check the FIR fit R² map and "
                "the mean-FIR QC curve."
            )
        raw = raw[raw_valid]
        manifold = manifold[raw_valid]

    # Order the library by time-to-peak so the index is monotonic in HRF
    # timing.  The ridge walk starts at the density peak and proceeds in
    # whichever direction the local tangent pointed, so without this the
    # ordering (and its direction) is arbitrary between runs — awkward for
    # anything downstream that reads the chosen index as "early vs late".
    order = np.argsort(np.argmax(raw, axis=1), kind="stable")
    raw = raw[order]
    manifold = manifold[order]

    # How well the final entries represent the voxels they came from.  This
    # is what tells you whether a 1-D ridge was the right model for this
    # dataset or whether the blob has real 2-D extent worth sampling.
    coverage = manifold_coverage(unit, manifold)

    # Duration correction — recover the impulse response so downstream
    # consumers, which re-convolve with the event boxcar at modelling
    # time, do not end up doubly convolved.  The pre-correction curves
    # stay on the result for QC.
    do_deconv = deconvolve_duration is not None and deconvolve_duration > target_dt
    # "fit" carries the correction inside the parametric family, so it
    # needs one; without a gamma fit the explicit inverse is all there is.
    method = deconv_method if fit_gamma else "wiener"

    fitted = None
    gamma_params: list[dict] = []
    raw_deconvolved: np.ndarray | None = None
    fitted_deconvolved: np.ndarray | None = None
    gamma_params_deconvolved: list[dict] = []
    duration_convolved_flag = not do_deconv

    boxcar_fit_path = do_deconv and method == "fit"

    if fit_gamma and not boxcar_fit_path:
        # Smooth `raw` itself, with no boxcar in the model: either there is no
        # duration to correct, or the wiener branch below strips it separately.
        # Either way this is the duration-convolved QC curve, not the library.
        fitted = np.zeros_like(raw)
        for i in range(raw.shape[0]):
            if shape_model == "spline":
                fit, p = fit_spline_through_boxcar(
                    raw[i], duration=0.0, dt=target_dt, n_knots=spline_knots
                )
            else:
                fit, p = fit_double_gamma(raw[i], dt=target_dt)
            fitted[i] = fit
            gamma_params.append(p)

    if boxcar_fit_path:
        # NSD-faithful: the boxcar lives in the forward model, so the
        # fitted parameters already describe the impulse response.  No
        # numerical inverse, hence no raw_deconvolved counterpart.
        #
        # `fitted` -- the duration-convolved QC curve -- is this same fit
        # pushed back THROUGH the boxcar, not an independent bare-double-gamma
        # fit of `raw`.  It used to be the latter, which asked an impulse
        # response family to reproduce a 20 s block response: it cannot, so
        # every fit saturated (a1 pinned at its upper bound 12.0 in every case
        # measured, c slammed to 0 or 1), correlated only r=0.75-0.89 with the
        # curve it was drawn against, and manufactured excursions to -0.48
        # where the raw cubic reached -0.13 -- the fitter cancelling a fast
        # rise the block response does not have. Pushing the honest fit back
        # through the boxcar reproduces the raw curve at r=1.0000.
        fitted_deconvolved = np.zeros_like(raw)
        fitted = np.zeros_like(raw)
        t_grid = np.arange(raw.shape[1], dtype=np.float64) * target_dt
        n_box = max(1, int(round(float(deconvolve_duration) / target_dt)))
        for i in range(raw.shape[0]):
            if shape_model == "spline":
                fit, p = fit_spline_through_boxcar(
                    raw[i],
                    duration=float(deconvolve_duration),
                    dt=target_dt,
                    n_knots=spline_knots,
                )
            else:
                fit, p = fit_double_gamma_through_boxcar(
                    raw[i], duration=float(deconvolve_duration), dt=target_dt
                )
            fitted_deconvolved[i] = fit
            gamma_params_deconvolved.append(p)
            if not p["fit_ok"]:
                fitted[i] = raw[i]
            elif shape_model == "spline":
                fitted[i] = spline_prediction_through_boxcar(
                    fit, float(deconvolve_duration), dt=target_dt
                )
            else:
                pred = _double_gamma_boxcar(
                    t_grid, p["a1"], p["b1"], p["a2"], p["b2"], p["c"], p["amp"], n_box=n_box
                )
                peak = float(np.max(pred))
                fitted[i] = pred / peak if peak > 0 else pred
            gamma_params.append(p)
    elif do_deconv:
        raw_deconvolved = deconvolve_event_duration(
            raw,
            duration=float(deconvolve_duration),
            dt=target_dt,
            snr=deconv_snr,
            normalize_peak=True,
        )
        if fit_gamma:
            # Fit in impulse-response space, not a fit of the convolved curve.
            # The boxcar is already gone, so there is none in the model here.
            fitted_deconvolved = np.zeros_like(raw_deconvolved)
            for i in range(raw_deconvolved.shape[0]):
                if shape_model == "spline":
                    fit, p = fit_spline_through_boxcar(
                        raw_deconvolved[i], duration=0.0, dt=target_dt, n_knots=spline_knots
                    )
                else:
                    fit, p = fit_double_gamma(raw_deconvolved[i], dt=target_dt)
                fitted_deconvolved[i] = fit
                gamma_params_deconvolved.append(p)

    # Closing the loop: a library entry is only correct if putting it back
    # through the event boxcar reproduces the curve it was derived from --
    # which is precisely what every downstream consumer does at modelling
    # time.  Cheap, and the only end-to-end statement about the duration
    # correction, so it is always computed.  It matters most on the wiener
    # path, where the explicit inverse rings at multiples of 1/D and a smooth
    # fit through that ringing can look plausible while no longer describing
    # the data.
    reconvolution_r = None
    reconvolution_leakage = None
    if fitted_deconvolved is not None and do_deconv:
        n_box_chk = max(1, int(round(float(deconvolve_duration) / target_dt)))
        n_t = raw.shape[1]
        rs = np.zeros(fitted_deconvolved.shape[0])
        leak = np.zeros(fitted_deconvolved.shape[0])
        for i in range(fitted_deconvolved.shape[0]):
            full = np.convolve(fitted_deconvolved[i], np.ones(n_box_chk))
            pred = full[:n_t]
            # How much of the response this entry predicts lands BEYOND the
            # window we can compare against.  Without this, the r above is a
            # passing grade on the part of the prediction we happen to look at:
            # a late impulse lobe convolves to D seconds past itself, falls off
            # the end, and never enters the correlation.  Measured at 13-38% on
            # a real 6-PC library whose r read 0.998 while its impulse
            # responses had lobes at 26 s.
            total = float(np.abs(full).sum())
            leak[i] = float(np.abs(full[n_t:]).sum() / total) if total > 0 else np.nan
            peak = float(np.max(pred))
            if peak <= 0:
                rs[i] = np.nan
                continue
            rs[i] = float(np.corrcoef(raw[i], pred / peak)[0, 1])
        reconvolution_r = rs
        reconvolution_leakage = leak

    return LibraryResult(
        raw=raw,
        target_times=target_times,
        manifold=manifold,
        svd=svd,
        selected_voxels=sel,
        fitted=fitted,
        raw_deconvolved=raw_deconvolved,
        fitted_deconvolved=fitted_deconvolved,
        gamma_params=gamma_params,
        gamma_params_deconvolved=gamma_params_deconvolved,
        n_dropped_invalid=n_invalid,
        coverage=coverage,
        reconvolution_r=reconvolution_r,
        reconvolution_leakage=reconvolution_leakage,
        event_durations=(
            np.asarray(event_durations, dtype=float) if event_durations is not None else None
        ),
        duration_convolved=duration_convolved_flag,
        shape_model=(shape_model if fit_gamma else "none"),
        mean_fir_hrf=mean_fir,
        unit_sphere_points=unit,
        sphere_hist2d=sphere_hist2d,
        sphere_hist_edges=sphere_hist_edges,
    )
