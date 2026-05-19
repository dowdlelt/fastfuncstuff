"""
FLOBS — FMRIB's Linear Optimal Basis Set, plus filmbabe-style constrained fit.

This module implements the basis-set + Bayesian-constraint construction from
Woolrich, Behrens, Smith (2004), "Constrained Linear Basis Sets for HRF
Modelling Using Variational Bayes" (FMRIB Technical Report TR04MW2).

Two pieces:

1. :func:`generate_flobs_basis` — sample N HRFs from a half-cosine
   parameterization, SVD to get the top K eigenHRFs (the basis), then
   fit a multivariate-normal constraint MVN(m, C) on the basis
   coefficients by regressing the sample HRFs back onto the basis.  The
   first three eigenHRFs come out looking remarkably like canonical +
   temporal-derivative + dispersion-derivative — see TR04MW2 fig 3.

2. :func:`fit_flobs_constrained` — penalised-least-squares "filmbabe
   lite": uses the MVN(m, C) as a Bayesian prior on the basis-set
   coefficients so the GLM cannot pick nonsense HRF shapes (the
   pathology of unconstrained basis-set fits, TR04MW2 fig 4(a)).  This
   is a deliberate simplification of the full Variational Bayes
   inference in TR04MW2 §3 — it skips AR(P) noise, MRF spatial prior,
   and joint inference over the amplitude D and shape β.  It captures
   the main practical benefit (shape constraint) at much lower
   complexity and validates trivially against unconstrained OLS.

References
----------
- Woolrich, Behrens, Smith (2004) TR04MW2, https://www.fmrib.ox.ac.uk/...
  Mirrored in this repo at raw_sources/FLOBS_filmbabe_tr04mw2.{pdf,txt}.
- FSL FLOBS web tool: https://fsl.fmrib.ox.ac.uk/fsl/docs/task_fmri/flobs.html
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from tqdm.auto import tqdm

from fastfuncstuff.design.hrf import get_spm_hrf_with_derivatives, pighs_halfcos
from fastfuncstuff.utils import get_device


# ----------------------------------------------------------------------------
# Basis generation
# ----------------------------------------------------------------------------


@dataclass
class FLOBSBasis:
    """Output of :func:`generate_flobs_basis`.

    Attributes
    ----------
    basis_functions : np.ndarray, shape (n_basis, n_t)
        The top-K eigenHRFs from the SVD of the half-cosine HRF sample
        matrix.  Sampled at ``dt`` resolution.  TR04MW2 §2.5 (matrix G
        of equation 15) — these are the columns of G after the SVD.
        First three typically look like canonical + temporal-derivative
        + dispersion-derivative (TR04MW2 fig 3).
    eigenvalues : np.ndarray, shape (min(n_t, n_samples),)
        Full singular-value spectrum from the SVD on the sample matrix.
        Inspect the elbow to decide ``n_basis``; TR04MW2 used 3.
    m : np.ndarray, shape (n_basis,)
        Mean of the multivariate-normal constraint on the basis-set
        coefficients (TR04MW2 equation 17, parameter "m" of
        MVN(m, C)).  Computed by regressing the half-cosine HRF
        samples onto the basis: R = (G'G)^-1 G' W, m = mean(R, axis=1).
    C : np.ndarray, shape (n_basis, n_basis)
        Covariance of the same MVN constraint (TR04MW2 equation 17,
        parameter "C").  ``C = cov(R, rowvar=True)``.  The prior on
        any condition's basis coefficients is N(D·m, D²·C) where D is
        the per-voxel amplitude; for our ridge-prior fit we collapse
        to the simpler N(m, C) form.
    dt : float
        Sample spacing of the basis functions in seconds.
    duration : float
        Total duration in seconds (= n_t × dt).
    n_samples : int
        Number of half-cosine HRF samples used to derive the basis.
    parametrization : dict
        Sampling ranges used.  Defaults match TR04MW2 equation 14:
        m1 ∈ [0, 2] s, m2 ∈ [2, 6] s, m3 ∈ [2, 6] s, m4 ∈ [2, 8] s,
        c2 ∈ [0, 0.5].
    sample_hrfs : np.ndarray, optional
        The raw sampled HRFs (n_t, n_samples), kept for QC plots /
        reconstruction-error checks.  Suppressed when ``keep_samples``
        is False to save memory.
    """

    basis_functions: np.ndarray
    eigenvalues: np.ndarray
    m: np.ndarray
    C: np.ndarray
    dt: float
    duration: float
    n_samples: int
    parametrization: dict
    sample_hrfs: np.ndarray | None = None


_DEFAULT_PARAMETRIZATION = {
    # TR04MW2 equation 14.  m1..m4 in seconds; c2 unitless (undershoot
    # depth as fraction of peak).  These are uniform-distribution
    # ranges, sampled independently.
    "m1": (0.0, 2.0),
    "m2": (2.0, 6.0),
    "m3": (2.0, 6.0),
    "m4": (2.0, 8.0),
    "c2": (0.0, 0.5),
}


def generate_flobs_basis(
    n_basis: int = 3,
    n_samples: int = 1000,
    duration: float = 32.0,
    dt: float = 0.1,
    parametrization: dict | None = None,
    seed: int = 42,
    keep_samples: bool = False,
) -> FLOBSBasis:
    """Generate the FLOBS basis + MVN(m, C) shape constraint.

    Procedure (TR04MW2 §2.5–§2.6):

    1. Sample ``n_samples`` HRFs from a half-cosine parameterization,
       drawing each shape parameter uniformly from its range in
       ``parametrization``.  Defaults match TR04MW2 equation 14.
    2. Stack samples into ``W`` (``n_t × n_samples``).  SVD: W ≈ G S Vt.
       Take the top ``n_basis`` left-singular vectors as G — these are
       the FLOBS eigenHRFs.
    3. Regress the original samples back onto the basis:
       ``R = (G'G)^-1 G' W`` (TR04MW2 equation 16; in practice
       ``R = G' W`` since G is orthonormal post-SVD).
    4. Fit a multivariate-normal distribution to the columns of R:
       ``m = mean(R, axis=1)``, ``C = cov(R, rowvar=True)``.  These
       are the prior parameters in TR04MW2 equation 13: a Gaussian on
       the basis coefficients that excludes nonsense HRF shapes.

    Parameters
    ----------
    n_basis : int, default 3
        Number of basis functions to keep.  TR04MW2 uses 3 — the
        eigenvalue spectrum from the half-cosine parameterization has
        a sharp elbow at K=3 (first three explain >95 % of variance).
    n_samples : int, default 1000
        Number of HRF samples drawn before SVD.  TR04MW2 used 1000.
    duration : float, default 32.0
        Total HRF window in seconds.  TR04MW2 used 51.2 s (length
        Nt = 512 at dt = 0.1 s).  32 s matches our canonical-library
        TSV convention and is plenty for typical event-related fMRI.
    dt : float, default 0.1
        Sample spacing in seconds.  TR04MW2 used 0.1 s (resolution
        Nt = 512 over 51.2 s).
    parametrization : dict, optional
        Override the sampling ranges.  Keys ``m1, m2, m3, m4, c2``,
        each ``(lo, hi)``.  Defaults match TR04MW2 equation 14.
    seed : int, default 42
        Seed for the (deterministic) Latin-Hypercube sampler inside
        :func:`create_pighs_library`.
    keep_samples : bool, default False
        If True, also return the raw sample HRF matrix on the result
        (useful for diagnostic plots; ~``n_t × n_samples × 8`` bytes
        of memory).

    Returns
    -------
    FLOBSBasis
        See dataclass for fields.

    Notes
    -----
    Unlike NSD-style library derivation
    (:mod:`fastfuncstuff.design.hrf_derive`), FLOBS does NOT look at
    the user's data — the basis + constraint are derived purely from a
    parametric HRF prior.  The "fit to data" step happens later, in
    :func:`fit_flobs_constrained`, where the MVN(m, C) acts as a
    prior on the per-voxel basis coefficients.
    """
    params = dict(_DEFAULT_PARAMETRIZATION)
    if parametrization is not None:
        params.update(parametrization)

    # Pure-FLOBS sampling: 5 independent uniform draws per HRF sample,
    # matching TR04MW2 equation 14 exactly.  We do NOT route through
    # ``create_pighs_library`` (which grid-samples peak-time + LHS on
    # the rest "to guarantee coverage") — that's a different sampling
    # strategy that the FLOBS paper does not use, and conflating them
    # would muddy the parameter (m, C) the constraint inherits.  Same
    # *parametric family* (half-cosines), different sampler.
    rng = np.random.default_rng(seed)
    m1_lo, m1_hi = params["m1"]
    m2_lo, m2_hi = params["m2"]
    m3_lo, m3_hi = params["m3"]
    m4_lo, m4_hi = params["m4"]
    c2_lo, c2_hi = params["c2"]
    m1_vals = rng.uniform(m1_lo, m1_hi, size=n_samples)
    m2_vals = rng.uniform(m2_lo, m2_hi, size=n_samples)
    m3_vals = rng.uniform(m3_lo, m3_hi, size=n_samples)
    m4_vals = rng.uniform(m4_lo, m4_hi, size=n_samples)
    c2_vals = rng.uniform(c2_lo, c2_hi, size=n_samples)

    # Generate each HRF via the existing half-cosine builder.  The
    # parametrization (m1=delay, m2=rise, m3=fall, m4=recovery, c2=
    # undershoot fraction) is identical to FLOBS's (h1, h2, h3, h4,
    # f2) so we can reuse pighs_halfcos directly per-sample.  HRFs are
    # generated at the requested ``dt`` and peak-normalized to +1.
    n_t = int(np.ceil(duration / dt))
    W = np.zeros((n_t, n_samples), dtype=np.float64)
    for i in range(n_samples):
        h = pighs_halfcos(
            m1=float(m1_vals[i]),
            m2=float(m2_vals[i]),
            m3=float(m3_vals[i]),
            m4=float(m4_vals[i]),
            c2=float(c2_vals[i]),
            duration=duration,
            sample_rate=dt,
            device=torch.device("cpu"),
        ).cpu().numpy()
        # Truncate / pad to exactly n_t (pighs_halfcos uses arange
        # which can produce one fewer or more sample depending on
        # rounding).
        if h.size >= n_t:
            W[:, i] = h[:n_t]
        else:
            W[:h.size, i] = h

    # SVD on the sample matrix.  full_matrices=False keeps U as
    # (n_t, min(n_t, n_s)); we just need the first n_basis columns.
    U, S, _ = np.linalg.svd(W, full_matrices=False)
    if n_basis > U.shape[1]:
        raise ValueError(
            f"n_basis={n_basis} exceeds available {U.shape[1]} singular vectors."
        )

    # Sign-align: each basis function should have a positive dominant
    # extremum.  Cosmetic but makes downstream plots / interpretation
    # consistent (PC1 looks like canonical positive HRF, not its
    # mirror image).  We mirror sign on the regression-coefficient
    # computation below so the (m, C) constraint is in the same frame.
    # ``U`` from numpy's SVD already has L2-unit columns, which is the
    # convention TR04MW2 fig 3 uses (and which gives the (m, C)
    # numerics in equation 17).  Without this normalization, m and C
    # come out scaled by σ_i (the singular value), and the prior
    # weight in :func:`fit_flobs_constrained` would need to absorb
    # that — much cleaner to fix it here.
    G = U[:, :n_basis].copy()                # (n_t, n_basis), L2-unit cols
    sign_flip = np.sign(G[np.argmax(np.abs(G), axis=0), np.arange(n_basis)])
    sign_flip = np.where(sign_flip == 0, 1.0, sign_flip)
    G = G * sign_flip[np.newaxis, :]

    # R = (G' G)^-1 G' W.  Since U is orthonormal, (G'G)^-1 = I and
    # R = G' W (cheaper, numerically cleaner).
    R = G.T @ W                              # (n_basis, n_samples)

    # MVN(m, C) on the columns of R (each column is one sample's
    # basis-coefficient vector).
    m_vec = R.mean(axis=1)
    C_mat = np.cov(R, rowvar=True)
    # Guard against C being exactly singular if n_samples ≤ n_basis.
    # Add a tiny diagonal regularizer; the prior remains effectively
    # the same.
    C_mat = C_mat + 1e-12 * np.eye(n_basis)

    return FLOBSBasis(
        basis_functions=G.T,                 # (n_basis, n_t) — row per basis
        eigenvalues=S,
        m=m_vec,
        C=C_mat,
        dt=float(dt),
        duration=float(duration),
        n_samples=int(n_samples),
        parametrization=params,
        sample_hrfs=W if keep_samples else None,
    )


# ----------------------------------------------------------------------------
# Constrained GLM fit (filmbabe lite)
# ----------------------------------------------------------------------------


def generate_spmg_basis(
    n_basis: int = 2,
    duration: float = 32.0,
    dt: float = 0.1,
) -> FLOBSBasis:
    """Build an SPMG1/SPMG2/SPMG3 basis as a :class:`FLOBSBasis`.

    Same container as :func:`generate_flobs_basis` so the constrained-
    fit primitive doesn't care which basis was used.  The default
    ``m`` and ``C`` here are placeholders (zero mean, identity
    covariance) — for SPMG fits you'll typically construct the prior
    via :func:`spmg_prior` instead and pass it explicitly to
    :func:`fit_basis_constrained_ridge`.

    Parameters
    ----------
    n_basis : int, default 2
        1 → SPMG1 (canonical only), 2 → SPMG2 (+ time derivative),
        3 → SPMG3 (+ dispersion derivative).
    duration : float, default 32.0
        Total HRF duration in seconds.
    dt : float, default 0.1
        Sample spacing in seconds.

    Returns
    -------
    FLOBSBasis
        Container with ``basis_functions`` (n_basis, n_t).
        ``m`` defaults to zeros and ``C`` to identity — callers
        should supply real priors via :func:`spmg_prior`.
    """
    if n_basis not in (1, 2, 3):
        raise ValueError(f"SPMG basis n_basis must be 1, 2, or 3; got {n_basis}")
    basis_set = get_spm_hrf_with_derivatives(
        microtime_dt=dt,
        hrf_duration=duration,
        n_basis=n_basis,
        device=torch.device("cpu"),
    )                                               # (n_basis, n_t)
    G = basis_set.cpu().numpy().astype(np.float64)

    # L2-normalize each row so the (m, C) numerics interoperate with
    # the FLOBS path's conventions (and with :func:`spmg_prior`).
    norms = np.linalg.norm(G, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    G = G / norms

    # Placeholder zero-mean identity prior — real callers use spmg_prior.
    m = np.zeros(n_basis, dtype=np.float64)
    C = np.eye(n_basis, dtype=np.float64)
    return FLOBSBasis(
        basis_functions=G,
        eigenvalues=np.array([1.0] * n_basis, dtype=np.float64),
        m=m,
        C=C,
        dt=float(dt),
        duration=float(duration),
        n_samples=0,
        parametrization={"family": f"SPMG{n_basis}"},
    )


@dataclass
class FLOBSFitResult:
    """Output of :func:`fit_flobs_constrained`.

    Attributes
    ----------
    betas : np.ndarray, shape (n_voxels, n_total_cols)
        Per-voxel coefficients after the constrained fit.  Layout
        matches the input design's column order: task block
        (per-condition blocks of ``n_basis`` columns) first, then any
        nuisance columns.
    hrfs : np.ndarray, shape (n_voxels, n_conditions, n_t)
        Reconstructed per-(voxel, condition) HRF after the constrained
        fit: basis_functions weighted by the corresponding task betas.
        Same time axis as the FLOBS basis.
    r2 : np.ndarray, shape (n_voxels,)
        Coefficient of determination of the constrained-fit predicted
        timecourse versus the data.
    betas_ols : np.ndarray, shape (n_voxels, n_total_cols)
        Per-voxel **unconstrained** OLS coefficients (the pre-pass
        used to estimate σ² for the auto prior weight).  Saved so the
        caller can show how much the constraint shifted the fit —
        critical for validating "the prior is doing the right thing".
        Same column layout as ``betas``.
    hrfs_ols : np.ndarray, shape (n_voxels, n_conditions, n_t)
        Reconstructed per-(voxel, condition) HRF from the
        unconstrained OLS pre-pass.  Comparing this to ``hrfs`` shows
        exactly where the FLOBS prior reshapes the response.
    r2_ols : np.ndarray, shape (n_voxels,)
        Unconstrained-fit R² for comparison with ``r2``.
    sigma2_mean : float
        Mean per-voxel residual variance from the OLS pre-pass —
        the natural Bayesian prior weight (``prior_weight="auto"``
        uses this directly).
    effective_prior_weight : float
        The actual scalar λ used to weight the FLOBS prior in the
        constrained solve.  Equals ``sigma2_mean`` when
        ``prior_weight="auto"``, otherwise ``user_multiplier ×
        sigma2_mean``.
    n_iter : int
        Number of iterations the alternating fit ran (1 for the
        non-iterative ridge-prior path).
    """

    betas: np.ndarray
    hrfs: np.ndarray
    r2: np.ndarray
    betas_ols: np.ndarray
    hrfs_ols: np.ndarray
    r2_ols: np.ndarray
    sigma2_mean: float
    effective_prior_weight: float
    n_iter: int = 1
    # Optional VB diagnostics — populated when the caller asks for them
    # via ``return_vb_diagnostics`` in :func:`fit_basis_constrained_ridge`.
    sigma2_per_voxel: np.ndarray | None = None      # (n_vox,)
    lambda_per_voxel: np.ndarray | None = None      # (n_vox,) effective λ used


def fit_basis_constrained_ridge(
    data: np.ndarray | torch.Tensor,
    design_task: np.ndarray | torch.Tensor,
    basis_functions: np.ndarray,
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    n_blocks: int,
    *,
    nuisance: np.ndarray | torch.Tensor | None = None,
    prior_weight: float | str = "auto",
    prior_weight_per_voxel: np.ndarray | None = None,
    device: torch.device | None = None,
    reconstruct_hrfs: bool = True,
    chunk_size: int | None = None,
    lambda_mode: str = "global",
    lambda_n_bins: int = 20,
    return_vb_diagnostics: bool = False,
) -> FLOBSFitResult:
    """Penalised-least-squares fit with an arbitrary MVN(m, C) shape prior.

    This is the **single primitive** powering FLOBS and SPMG2/SPMG3
    constrained fits (and any other parametric basis with a Gaussian
    shape prior).  The math is identical across model choices — only
    ``basis_functions``, ``prior_mean``, ``prior_cov`` differ.

    The previous name was ``fit_flobs_constrained`` and tied the prior
    to a :class:`FLOBSBasis` object; this version decouples the prior
    from the basis so the caller can swap any (m, C) onto any basis.
    See :func:`flobs_prior`, :func:`spmg_prior`, :func:`ridge_prior`
    for convenient constructors.

    For one condition with basis-coefficient vector β_c (size
    ``n_basis``), the penalised objective is:

    .. math::

        \\hat{\\beta} = \\arg\\min_{\\beta} \\| y - X \\beta \\|^2 +
                       \\sum_c \\lambda \\, (\\beta_c - m)^T C^{-1} (\\beta_c - m)

    where ``λ = prior_weight``.  This is a standard ridge with a
    block-diagonal precision matrix; the closed-form solution is

    .. math::

        \\hat{\\beta} = (X^T X + \\lambda P)^{-1} (X^T y + \\lambda P\\bar{m})

    where ``P`` is block-diagonal of ``C^{-1}`` repeated ``n_conditions``
    times (followed by zeros for any nuisance columns), and
    ``P\\bar{m}`` is the corresponding stack of ``C^{-1} m`` vectors.

    Relation to plain ridge regression
    ----------------------------------
    This IS ridge regression — generalized ridge, with a problem-
    informed prior::

        plain ridge       :  β ~ N(0, σ²/λ · I)              [shrink to zero]
        FLOBS constrained :  β ~ N(m, C/λ)                   [shrink to canonical HRF]

    Plain ridge with λI would shrink every basis-coefficient toward
    zero, which is the wrong target (HRF = 0 is the *least* sensible
    shape).  FLOBS uses the empirical mean ``m`` and covariance ``C``
    of the basis coefficients obtained from regressing 1000 sensible
    HRF samples onto the basis — so the prior pushes toward an
    average sensible HRF, with full (not diagonal) precision so
    *combinations* of basis functions that produce nonsense shapes
    are penalised even when each individual coefficient looks fine.
    Off-diagonal terms of ``C^{-1}`` are what make this stronger than
    independent per-coefficient shrinkage.

    Differences from full filmbabe (TR04MW2 §3):

    - No AR(P) noise model (we assume white residuals).
    - No MRF spatial prior on AR coefficients.
    - No joint inference of amplitude D and shape β — we collapse to
      a single coefficient vector.  In practice the prior MVN(m, C)
      already constrains the *direction* of β to sensible HRF shapes;
      the missing D × β decomposition is mostly a re-parameterization
      that helps VB tractability rather than changing what the fit
      can express.
    - No spatial mixture-modelling probability map.

    Parameters
    ----------
    data : array, shape (n_voxels, n_timepoints)
        Preprocessed BOLD signal.
    design_task : array, shape (n_timepoints, n_conditions × n_basis)
        Task design: each condition convolved with each FLOBS basis
        function.  Columns must be in condition-major order
        (cond0×basis0..N, cond1×basis0..N, …) — same convention as
        :func:`fastfuncstuff.design.builder.build_per_run_task_designs`.
    basis : FLOBSBasis
        Output of :func:`generate_flobs_basis`.  ``basis.m`` and
        ``basis.C`` define the Gaussian prior on each condition's
        basis coefficients.
    n_conditions : int
        Number of conditions in the task design (so we know how to
        block ``design_task``).
    nuisance : array, shape (n_timepoints, n_nuisance), optional
        Additional regressors (polynomial drift, motion, etc.) that
        get fit with NO prior — these are "free" parameters.  If
        provided, they are appended to the task design before the
        solve and their coefficients are returned attached to the end
        of ``betas`` (callers can slice them off).
    prior_weight : float or "auto", default "auto"
        Strength of the FLOBS prior.  The Bayesian-optimal weight is
        ``σ²`` (the noise variance) — at high SNR the prior should
        fade out (σ²→0 ⇒ OLS), at low SNR it should dominate.
        ``"auto"`` runs a quick OLS pass, estimates the mean residual
        variance, and uses ``effective_weight = σ²_mean``.  Passing a
        float overrides as a *multiplier* on σ²_mean — e.g. 2.0 makes
        the prior twice as strong as the Bayesian-optimal weight.
        0 → unconstrained OLS.
    device : torch.device, optional
        Compute device.  Defaults to :func:`get_device`.
    reconstruct_hrfs : bool, default True
        If True, eagerly compute the full reconstructed HRF per
        ``(voxel, block)`` and return on ``hrfs`` / ``hrfs_ols``.
        Memory cost: ``n_voxels × n_blocks × n_t_basis × 8 bytes``.
        For single-trial mode this can blow up — set False to skip the
        reconstruction; ``hrfs`` and ``hrfs_ols`` will be ``None``,
        and the caller can reconstruct per-chunk on demand.
    lambda_mode : {"global", "voxelwise"}, default "global"
        Choice of prior weight λ:

        - ``"global"`` (current default): one scalar λ used for every
          voxel.  ``"auto"`` resolves to ``λ = σ²_mean`` across voxels.
        - ``"voxelwise"``: per-voxel λ_v = ``prior_weight × σ²_v``
          using each voxel's own OLS residual variance.  Honest
          Bayesian behaviour — high-SNR voxels get less shrinkage
          (their σ² is low), low-SNR voxels get more.  Implementation
          bins voxels by σ² quantile (``lambda_n_bins`` bins, default
          20) and Cholesky-factors one constrained-system matrix per
          bin, then solves each voxel against its bin's factor.
          ~20× extra factorisations of a tiny (p × p) matrix; the
          marginal cost is negligible compared to the per-voxel
          right-hand-side product.

        When ``prior_weight`` is a scalar (e.g. ``2.0``), it
        *multiplies* the per-voxel σ²_v in voxelwise mode.  The
        CV-grid search continues to work in both modes — the grid
        sweeps the multiplier, not σ² itself.
    lambda_n_bins : int, default 20
        Number of σ² quantile bins for ``lambda_mode="voxelwise"``.
        20 is enough resolution given that fMRI noise variance has
        a smooth distribution across voxels; higher buys a little
        per-voxel precision at the cost of more Cholesky factorisations.

    Returns
    -------
    FLOBSFitResult
        See dataclass.  When ``reconstruct_hrfs=False``, ``hrfs`` and
        ``hrfs_ols`` are ``None`` (saves up to tens of GB for
        single-trial fits).
    """
    if device is None:
        device = get_device()

    # Respect the caller's device choice.  Data on cuda stays on
    # cuda; data on CPU stays on CPU.  The chunked solver below
    # streams slices to the compute device — when source and target
    # are the same device that's a no-op view, so passing pre-loaded
    # cuda data avoids unnecessary host↔device transfers per chunk.
    # (Old code forced CPU "for safety" and re-uploaded chunks every
    # call — wasteful for cuda users with enough memory.)
    # Keep ``y`` in its original dtype/device — upcasting the entire
    # (n_voxels × n_t) tensor to float64 up front doubles peak memory
    # and OOMs on big data even before the chunked passes start
    # (e.g. 381k × 2380 × 8B = 7.2 GB on top of the existing float32
    # copy already on GPU).  The chunk loops below cast per-chunk
    # with ``.to(device=device, dtype=torch.float64)``, so the math
    # remains float64 where it matters; only the storage stays in
    # its original precision.
    y = (
        torch.as_tensor(data)
        if not isinstance(data, torch.Tensor)
        else data
    )
    X = (
        torch.as_tensor(design_task, dtype=torch.float64, device=device)
        if not isinstance(design_task, torch.Tensor)
        else design_task.to(device=device, dtype=torch.float64)
    )
    if y.ndim != 2:
        raise ValueError(f"data must be 2-D (n_voxels, n_t); got {y.shape}")
    if X.ndim != 2:
        raise ValueError(f"design_task must be 2-D (n_t, n_cols); got {X.shape}")
    n_voxels, n_t = y.shape
    if X.shape[0] != n_t:
        raise ValueError(
            f"design_task has {X.shape[0]} rows but data has {n_t} timepoints"
        )

    n_basis = basis_functions.shape[0]
    if X.shape[1] != n_blocks * n_basis:
        raise ValueError(
            f"design_task has {X.shape[1]} columns but expected "
            f"n_blocks ({n_blocks}) × n_basis ({n_basis}) = "
            f"{n_blocks * n_basis}"
        )
    if prior_mean.shape != (n_basis,):
        raise ValueError(
            f"prior_mean must have shape ({n_basis},); got {prior_mean.shape}"
        )
    if prior_cov.shape != (n_basis, n_basis):
        raise ValueError(
            f"prior_cov must have shape ({n_basis}, {n_basis}); got {prior_cov.shape}"
        )

    if nuisance is not None:
        Z = (
            torch.as_tensor(nuisance, dtype=torch.float64, device=device)
            if not isinstance(nuisance, torch.Tensor)
            else nuisance.to(device=device, dtype=torch.float64)
        )
        if Z.ndim != 2 or Z.shape[0] != n_t:
            raise ValueError(
                f"nuisance must be (n_t, n_nuisance) with n_t={n_t}; "
                f"got {Z.shape}"
            )
        X_full = torch.cat([X, Z], dim=1)
    else:
        X_full = X

    n_cols = X_full.shape[1]
    n_task_cols = n_blocks * n_basis

    # Build the block-diagonal precision matrix P (size n_cols x n_cols).
    # First n_task_cols rows/cols: block-diagonal of C^-1 repeated
    # ``n_blocks`` times (one block per condition OR per trial,
    # depending on caller's design).  Remaining n_nuisance rows/cols:
    # zeros (nuisance is unpenalised — no prior on polynomial drift).
    C_inv = torch.from_numpy(np.linalg.inv(prior_cov)).to(device=device, dtype=torch.float64)
    P = torch.zeros((n_cols, n_cols), dtype=torch.float64, device=device)
    for c in range(n_blocks):
        s = c * n_basis
        P[s:s + n_basis, s:s + n_basis] = C_inv

    # Prior mean vector m̄ (size n_cols): stacked ``prior_mean`` for
    # each block (condition/trial), zeros for nuisance.
    m_torch = torch.from_numpy(prior_mean).to(device=device, dtype=torch.float64)
    m_bar = torch.zeros(n_cols, dtype=torch.float64, device=device)
    for c in range(n_blocks):
        m_bar[c * n_basis:(c + 1) * n_basis] = m_torch

    # ---------------------------------------------------------------
    # Voxel-chunked two-pass solver.  We never materialise the full
    # ``(n_voxels, n_t)`` predicted / residual tensors — for
    # 306k voxels × 1152 TR × float64 that's ~2.8 GB per tensor, which
    # OOMs a 16 GB GPU once PyTorch caching and other intermediates
    # are factored in.  Per-chunk peak memory is
    # ``chunk × n_t × 16 bytes`` (data + predicted) + tiny constants.
    #
    # Pass 1 (OLS)         → ss_res_ols, beta_ols, ss_tot
    # Pass 2 (constrained) → ss_res, betas
    # Cholesky factors of X'X and (X'X + λ P) computed once, reused
    # across chunks.
    # ---------------------------------------------------------------
    XtX = X_full.T @ X_full

    # Pre-factor for the OLS path (shared across chunks).
    try:
        L0 = torch.linalg.cholesky(XtX)
        use_chol_ols = True
    except torch.linalg.LinAlgError:
        L0 = None
        use_chol_ols = False

    if chunk_size is None:
        # Memory model: per-chunk peak ≈ chunk × n_t × 16 (data +
        # predicted, float64) + n_cols × chunk × 8 (per-voxel betas).
        # ``estimate_chunk_size`` handles the per-device defaults.
        from fastfuncstuff.memory import estimate_chunk_size
        chunk_size = estimate_chunk_size(
            n_voxels=n_voxels,
            n_timepoints=n_t,
            n_regressors=n_cols,
            device=device,
            operation="glm",
            use_double=True,
        )

    # CPU-resident output accumulators.
    beta_ols_full = torch.empty((n_voxels, n_cols), dtype=torch.float64, device="cpu")
    betas_full = torch.empty((n_voxels, n_cols), dtype=torch.float64, device="cpu")
    ss_res_ols_full = torch.empty(n_voxels, dtype=torch.float64, device="cpu")
    ss_res_constrained_full = torch.empty(n_voxels, dtype=torch.float64, device="cpu")
    ss_tot_full = torch.empty(n_voxels, dtype=torch.float64, device="cpu")

    # ----------- Pass 1: OLS + sufficient stats ---------------------
    n_chunks_p1 = (n_voxels + chunk_size - 1) // chunk_size
    for start in tqdm(
        range(0, n_voxels, chunk_size),
        total=n_chunks_p1, desc="  Pass 1 (OLS)", unit="chunk",
        leave=False, disable=n_chunks_p1 <= 1,
    ):
        end = min(start + chunk_size, n_voxels)
        y_chunk = y[start:end].to(
            device=device, dtype=torch.float64, non_blocking=True,
        )                                                            # (chunk, n_t)
        Xty_chunk = X_full.T @ y_chunk.T                          # (n_cols, chunk)
        if use_chol_ols:
            beta_chunk = torch.cholesky_solve(Xty_chunk, L0)
        else:
            beta_chunk = torch.linalg.lstsq(X_full, y_chunk.T).solution
        pred = (X_full @ beta_chunk).T                            # (chunk, n_t)
        resid = y_chunk - pred
        ss_res_chunk = (resid ** 2).sum(dim=1)
        y_mean = y_chunk.mean(dim=1, keepdim=True)
        ss_tot_chunk = ((y_chunk - y_mean) ** 2).sum(dim=1)

        beta_ols_full[start:end] = beta_chunk.T.cpu()
        ss_res_ols_full[start:end] = ss_res_chunk.cpu()
        ss_tot_full[start:end] = ss_tot_chunk.cpu()
        del y_chunk, Xty_chunk, beta_chunk, pred, resid, ss_res_chunk, y_mean, ss_tot_chunk

    # σ² mean from accumulated OLS residuals.
    dof = max(1, n_t - n_cols)
    sigma2_mean = float(ss_res_ols_full.mean().item() / dof)

    if isinstance(prior_weight, str):
        if prior_weight != "auto":
            raise ValueError(f"prior_weight: expected float or 'auto'; got {prior_weight!r}")
        effective_weight = sigma2_mean
    else:
        effective_weight = float(prior_weight) * sigma2_mean

    # ----------- Pass 2: constrained solve --------------------------
    # Two modes:
    #
    #   global      : one λ for every voxel.  Single A = X'X + λP,
    #                 one Cholesky factor, share across all chunks.
    #                 (Cheap; old default.)
    #
    #   voxelwise   : per-voxel λ_v ∝ σ²_v.  Bin voxels by σ² quantile,
    #                 factor one A_b = X'X + λ_b P per bin (where
    #                 λ_b is the bin centre), solve voxels in that
    #                 bin against that factor.  Honest-Bayesian per-
    #                 voxel weighting at ~20× tiny p×p Cholesky cost.
    #
    # In voxelwise mode, ``effective_weight`` (computed above) acts as
    # a global MULTIPLIER applied on top of σ²_v.  The CV grid still
    # works — it sweeps that multiplier.
    if lambda_mode not in ("global", "voxelwise"):
        raise ValueError(
            f"lambda_mode must be 'global' or 'voxelwise'; got {lambda_mode!r}"
        )

    # prior_weight_per_voxel forces voxelwise mode regardless of caller
    # preference — the override is intrinsically per-voxel.
    if prior_weight_per_voxel is not None:
        lambda_mode = "voxelwise"

    # ── Fast path: prior_weight effectively zero ────────────────────
    # When the prior contributes nothing (``-reg none``, or
    # ``prior_weight_per_voxel`` is all zeros, or scalar 0.0), Pass 2
    # would solve A = X'X + 0·P → identical to the OLS pass 1.  Skip
    # it: betas = beta_ols, ss_res = ss_res_ols.  Halves the per-call
    # work — which matters for LSS where we make N_trials such calls.
    # Also avoids the voxelwise σ²-binning machinery in that case.
    _prior_is_zero = (
        (not isinstance(prior_weight, str) and float(prior_weight) == 0.0)
        or (prior_weight_per_voxel is not None
            and not np.any(np.asarray(prior_weight_per_voxel)))
    )
    if _prior_is_zero:
        betas_full.copy_(beta_ols_full)
        ss_res_constrained_full.copy_(ss_res_ols_full)
        effective_weight = 0.0
        # Set voxel_bin / bin_lambda placeholders so the VB-diagnostic
        # path below doesn't reference unbound names.
        voxel_bin = torch.zeros(n_voxels, dtype=torch.long)
        bin_lambda = torch.zeros(1, dtype=torch.float64)
        # Jump straight to R² assembly + return below.
        _skip_pass2 = True
    else:
        _skip_pass2 = False

    Pm_unit = P @ m_bar                                            # (n_cols,)

    if _skip_pass2:
        # Fast path took it; betas / ss_res already populated above.
        pass
    elif lambda_mode == "global":
        # One A, one factor.
        A = XtX + effective_weight * P
        Pm = effective_weight * Pm_unit
        try:
            L_A = torch.linalg.cholesky(A)
            use_chol_A = True
        except torch.linalg.LinAlgError:
            L_A = None
            use_chol_A = False

        n_chunks_p2 = (n_voxels + chunk_size - 1) // chunk_size
        for start in tqdm(
            range(0, n_voxels, chunk_size),
            total=n_chunks_p2, desc="  Pass 2 (constrained)", unit="chunk",
            leave=False, disable=n_chunks_p2 <= 1,
        ):
            end = min(start + chunk_size, n_voxels)
            y_chunk = y[start:end].to(
                device=device, dtype=torch.float64, non_blocking=True,
            )
            Xty_chunk = X_full.T @ y_chunk.T
            rhs_chunk = Xty_chunk + Pm[:, None]
            if use_chol_A:
                beta_chunk = torch.cholesky_solve(rhs_chunk, L_A)
            else:
                beta_chunk = torch.linalg.solve(A, rhs_chunk)
            pred = (X_full @ beta_chunk).T
            resid = y_chunk - pred
            ss_res_chunk = (resid ** 2).sum(dim=1)

            betas_full[start:end] = beta_chunk.T.cpu()
            ss_res_constrained_full[start:end] = ss_res_chunk.cpu()
            del y_chunk, Xty_chunk, rhs_chunk, beta_chunk, pred, resid, ss_res_chunk

    else:  # voxelwise
        # Per-voxel σ²_v from pass 1 OLS residuals; user multiplier
        # rescales the whole grid (so prior_weight=auto → unit multiplier,
        # prior_weight=2.0 → 2× σ²_v per voxel, etc.).
        #
        # VB override: when ``prior_weight_per_voxel`` is provided
        # (filmbabe β_size update), use those values *directly* as the
        # per-voxel λ_v.  This bypasses the σ² × user_multiplier
        # computation — caller has already done that math and knows the
        # exact effective weight it wants per voxel.
        sigma2_per_voxel = (ss_res_ols_full / dof).clamp_min(1e-30)  # CPU tensor
        if prior_weight_per_voxel is not None:
            if prior_weight_per_voxel.shape != (n_voxels,):
                raise ValueError(
                    f"prior_weight_per_voxel must have shape ({n_voxels},); "
                    f"got {prior_weight_per_voxel.shape}"
                )
            lambda_per_voxel_cpu = torch.from_numpy(
                prior_weight_per_voxel.astype(np.float64)
            ).clamp_min(0.0)
        else:
            if isinstance(prior_weight, str):
                user_mult = 1.0
            else:
                user_mult = float(prior_weight)
            lambda_per_voxel_cpu = user_mult * sigma2_per_voxel        # (n_voxels,)

        # Quantile-bin: assign each voxel to one of ``lambda_n_bins``
        # bins by σ² quantile.  Within a bin use the bin's median λ
        # as the representative λ_b.
        n_bins_eff = max(1, min(int(lambda_n_bins), n_voxels))
        # torch.quantile + bucketize: edges at i/n_bins quantiles,
        # for i = 1..n_bins-1.
        if n_bins_eff > 1:
            qs = torch.linspace(0.0, 1.0, n_bins_eff + 1)[1:-1].to(lambda_per_voxel_cpu.dtype)
            edges = torch.quantile(lambda_per_voxel_cpu, qs)
            voxel_bin = torch.bucketize(lambda_per_voxel_cpu, edges)
        else:
            voxel_bin = torch.zeros(n_voxels, dtype=torch.long)

        # Per-bin representative λ = median of in-bin λ_v.
        bin_lambda = torch.empty(n_bins_eff, dtype=torch.float64)
        for b in range(n_bins_eff):
            mask_b = voxel_bin == b
            if mask_b.any():
                bin_lambda[b] = lambda_per_voxel_cpu[mask_b].median()
            else:
                bin_lambda[b] = torch.tensor(0.0, dtype=torch.float64)

        # Pre-factor one A per bin (small p×p matrices; ~20 of them).
        chol_factors: dict[int, torch.Tensor | None] = {}
        for b in range(n_bins_eff):
            lam = float(bin_lambda[b].item())
            A_b = XtX + lam * P
            try:
                chol_factors[b] = torch.linalg.cholesky(A_b)
            except torch.linalg.LinAlgError:
                chol_factors[b] = None
        # Pm shared computation per bin too — Pm_b = λ_b · P · m̄.
        # Lay out as a (n_bins, n_cols) tensor for fast per-voxel
        # right-hand-side assembly.
        Pm_per_bin = bin_lambda.to(Pm_unit.dtype).unsqueeze(1) * Pm_unit.cpu().unsqueeze(0)
        Pm_per_bin = Pm_per_bin.to(device)

        n_chunks_p2v = (n_voxels + chunk_size - 1) // chunk_size
        for start in tqdm(
            range(0, n_voxels, chunk_size),
            total=n_chunks_p2v, desc="  Pass 2 (voxelwise λ)", unit="chunk",
            leave=False, disable=n_chunks_p2v <= 1,
        ):
            end = min(start + chunk_size, n_voxels)
            y_chunk = y[start:end].to(
                device=device, dtype=torch.float64, non_blocking=True,
            )
            Xty_chunk = X_full.T @ y_chunk.T                       # (n_cols, chunk)
            chunk_bins = voxel_bin[start:end].to(device)
            # Per-voxel Pm: lookup from Pm_per_bin via the bin index.
            # Shape: (n_cols, chunk).
            Pm_chunk = Pm_per_bin[chunk_bins].T
            rhs_chunk = Xty_chunk + Pm_chunk
            # Solve per-bin within the chunk.  Group voxels by bin so
            # we make one cholesky_solve call per (bin, chunk).
            beta_chunk = torch.empty_like(rhs_chunk)
            for b in range(n_bins_eff):
                idx_in_chunk = (chunk_bins == b).nonzero(as_tuple=True)[0]
                if idx_in_chunk.numel() == 0:
                    continue
                rhs_b = rhs_chunk[:, idx_in_chunk]
                L_b = chol_factors[b]
                if L_b is not None:
                    beta_b = torch.cholesky_solve(rhs_b, L_b)
                else:
                    lam = float(bin_lambda[b].item())
                    beta_b = torch.linalg.solve(XtX + lam * P, rhs_b)
                beta_chunk[:, idx_in_chunk] = beta_b
            pred = (X_full @ beta_chunk).T
            resid = y_chunk - pred
            ss_res_chunk = (resid ** 2).sum(dim=1)

            betas_full[start:end] = beta_chunk.T.cpu()
            ss_res_constrained_full[start:end] = ss_res_chunk.cpu()
            del y_chunk, Xty_chunk, Pm_chunk, rhs_chunk, beta_chunk, pred, resid, ss_res_chunk

    # ----------- R² assembly + return -------------------------------
    # Some voxels have ss_tot ≈ 0 (e.g. zeroed by PSC scaling
    # violations, dead voxels, or rare constant timeseries).  The naive
    # ``1 − ss_res/ss_tot`` with a tiny clamp produces enormous negative
    # R² values for those voxels (the OLS pass *fits* a constant via
    # the polynomial nuisance → ss_res = 0 → R² = 1, but the
    # constrained pass shrinks β toward m → ss_res > 0 → R² = 1 − big
    # / tiny → −1e28).  A single bad voxel like that destroys the
    # mean R² printed downstream.  Detect them and report R² = 0
    # (uninformative); a downstream user-visible mask should drop
    # these voxels anyway.
    _SS_TOT_MIN = 1e-6                     # well above float32 floor; well below real-data ss_tot
    _valid_mask = ss_tot_full > _SS_TOT_MIN
    r2 = torch.where(
        _valid_mask,
        1.0 - ss_res_constrained_full / torch.clamp(ss_tot_full, min=_SS_TOT_MIN),
        torch.zeros_like(ss_tot_full),
    )
    r2_ols = torch.where(
        _valid_mask,
        1.0 - ss_res_ols_full / torch.clamp(ss_tot_full, min=_SS_TOT_MIN),
        torch.zeros_like(ss_tot_full),
    )

    betas_np = betas_full.numpy()
    betas_ols_np = beta_ols_full.numpy()

    # Reconstruct per-(voxel, block) HRFs if requested.  Cost is
    # n_voxels × n_blocks × n_t × 8 bytes; for single-trial fits the
    # caller should pass reconstruct_hrfs=False and reconstruct per
    # chunk on demand (as ffs_fitbasis does in its CLI).
    if reconstruct_hrfs:
        task_betas = betas_np[:, :n_task_cols].reshape(n_voxels, n_blocks, n_basis)
        task_betas_ols = betas_ols_np[:, :n_task_cols].reshape(n_voxels, n_blocks, n_basis)
        hrfs = task_betas @ basis_functions
        hrfs_ols = task_betas_ols @ basis_functions
    else:
        hrfs = None
        hrfs_ols = None

    # ── Optional VB diagnostics ──────────────────────────────────────
    # When the caller asks for them (filmbabe loop), surface the per-voxel
    # σ² and λ that were actually used.  In ``global`` lambda_mode each
    # voxel sees the same λ = ``effective_weight``.  In ``voxelwise``
    # mode each voxel sees its bin's median λ (or, with the
    # ``prior_weight_per_voxel`` override, the exact value passed in).
    if return_vb_diagnostics:
        sigma2_pv_np = (
            (ss_res_ols_full / dof).clamp_min(1e-30).numpy().astype(np.float32)
        )
        if lambda_mode == "voxelwise":
            # ``voxel_bin`` and ``bin_lambda`` are set in the voxelwise
            # branch above.  Map back: λ_v = bin_lambda[voxel_bin[v]].
            lambda_pv_np = bin_lambda[voxel_bin].numpy().astype(np.float32)
        else:
            lambda_pv_np = np.full(
                (n_voxels,), float(effective_weight), dtype=np.float32,
            )
    else:
        sigma2_pv_np = None
        lambda_pv_np = None

    return FLOBSFitResult(
        betas=betas_np,
        hrfs=hrfs,
        r2=r2.numpy(),
        betas_ols=betas_ols_np,
        hrfs_ols=hrfs_ols,
        r2_ols=r2_ols.numpy(),
        sigma2_mean=sigma2_mean,
        effective_prior_weight=effective_weight,
        n_iter=1,
        sigma2_per_voxel=sigma2_pv_np,
        lambda_per_voxel=lambda_pv_np,
    )


@dataclass
class FLOBSCVResult:
    """Output of :func:`cv_basis_constrained_ridge`.

    Attributes
    ----------
    weights : list[float | str]
        The grid of prior-weight multipliers evaluated, in order.
        ``"OLS"`` is the unconstrained baseline (no shape prior).
    r2_per_weight : np.ndarray, shape (n_voxels, n_weights)
        Held-out R² for each (voxel, weight) — single R² per voxel
        per weight, computed from predictions concatenated across CV
        folds (the GLMdenoise / GLMsingle convention).
    argmax_weight_idx : np.ndarray, shape (n_voxels,)
        For each voxel, the index into ``weights`` whose held-out R²
        is largest.  Save this as a 3-D NIfTI to see *where* each
        regularization strength wins.
    n_splits : int
        Number of CV folds run.
    """

    weights: list
    r2_per_weight: np.ndarray
    argmax_weight_idx: np.ndarray
    n_splits: int


def cv_basis_constrained_ridge(
    per_run_data: list[torch.Tensor],
    per_run_task_designs: list[torch.Tensor],
    basis_functions: np.ndarray,
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    n_blocks: int,
    polort: int,
    *,
    weight_grid: list[float] | None = None,
    include_ols: bool = True,
    leave_n_out: int = 1,
    n_perms: int = 50,
    lambda_mode: str = "global",
    lambda_n_bins: int = 20,
    device: torch.device | None = None,
    verbose: bool = True,
) -> FLOBSCVResult:
    """Run-based LORO cross-validation of the constrained-ridge fit.

    Sweeps a grid of prior-weight multipliers (plus an OLS baseline),
    fits on training runs, predicts held-out runs, and aggregates a
    single per-voxel R² per weight from the concatenated
    cross-fold predictions (GLMdenoise convention — see
    :func:`fastfuncstuff.glm.xval.compute_xval_r2`).

    This is the framework that lets us A/B every later regularization
    change empirically.  Without it, every choice of prior_weight is
    a just-so story.

    Strategy:

    1. Project per-run polynomial nuisance out of data AND task
       design (via :func:`fastfuncstuff.glm.xval.project_out_nuisance_per_run`)
       once, up front.  After projection the task design is
       polynomial-free within each run, which makes per-fold fits
       trivially comparable.
    2. For each prior_weight in the grid (plus ``"OLS"``):
       a. For each (train_runs, test_runs) split:
          - Slice cleaned data + design by train run indices.
          - Fit :func:`fit_basis_constrained_ridge` (with that weight)
            on the train slice, ``polort=-1`` so the function
            doesn't try to add polys again.
          - Predict the cleaned test data using the task betas:
            ``y_test_pred = X_test_clean @ β_task``.
          - Place the predictions into a concatenated-by-time buffer.
       b. After all splits, compute per-voxel R² from the full
          concatenated predictions vs the full cleaned data.
    3. Return :class:`FLOBSCVResult` with per-weight R² maps and a
       per-voxel argmax.

    Parameters
    ----------
    per_run_data : list[torch.Tensor], each (n_voxels, n_tp_run)
        Voxel data per run.
    per_run_task_designs : list[torch.Tensor], each (n_tp_run, n_task)
        Task design per run (output of
        :func:`fastfuncstuff.design.builder.build_per_run_task_designs`).
    basis_functions : np.ndarray, shape (n_basis, n_t_basis)
        Basis used to convolve onsets.  Only needed here for the
        ``n_basis`` count; reconstruction happens upstream.
    prior_mean : np.ndarray, shape (n_basis,)
        Prior mean.
    prior_cov : np.ndarray, shape (n_basis, n_basis)
        Prior covariance.
    n_blocks : int
        Number of condition / trial blocks in the task design
        (block_count × n_basis = n_task_cols).
    polort : int
        Polynomial detrend order per run.  ``-1`` disables.
    weight_grid : list[float], optional
        Multipliers on σ² to evaluate.  Defaults to
        ``[0.1, 0.3, 1.0, 3.0, 10.0]``.
    include_ols : bool, default True
        Also evaluate the unconstrained OLS path as a baseline.
        Held-out R² of OLS is the right floor: a constrained fit
        must beat it to earn its keep.
    leave_n_out : int, default 1
        How many runs to leave out per fold.  ``1`` = LORO.
    n_perms : int, default 50
        Max number of CV splits.  For LORO with ≤50 runs all splits
        are evaluated.
    device, verbose
        Forwarded to the per-fold solver.

    Returns
    -------
    FLOBSCVResult
    """
    from fastfuncstuff.glm.xval import (
        generate_cv_splits,
        project_out_nuisance_per_run,
    )
    from fastfuncstuff.design.builder import legendre_polynomials

    if device is None:
        device = get_device()
    if weight_grid is None:
        weight_grid = [0.1, 0.3, 1.0, 3.0, 10.0]

    n_runs = len(per_run_data)
    if n_runs < 2:
        raise ValueError(f"CV needs at least 2 runs; got {n_runs}")
    n_basis = basis_functions.shape[0]
    n_task_cols = n_blocks * n_basis

    n_tp_per_run = [d.shape[1] for d in per_run_data]
    run_starts = [0]
    for n in n_tp_per_run[:-1]:
        run_starts.append(run_starts[-1] + n)

    # Concatenate data + task design row-wise (no polys yet).
    # design_full: (total_tp, n_task_cols)
    # data_full:   (n_voxels, total_tp)
    data_full = torch.cat([d.to(device).float() for d in per_run_data], dim=1)
    n_voxels = data_full.shape[0]
    design_full = torch.cat(
        [t.to(device).float() for t in per_run_task_designs], dim=0
    )

    # Per-run nuisance (polynomial detrend).  Project out from BOTH
    # data and task design once, up front — after this the design is
    # polynomial-free and per-fold fits collapse to a clean ridge
    # solve on the task block.
    if polort >= 0:
        nuisance_per_run = [
            torch.from_numpy(legendre_polynomials(n_tp_per_run[r], polort))
            .to(device).float()
            for r in range(n_runs)
        ]
        if verbose:
            print(
                f"  CV: projecting out per-run polynomials (polort={polort}, "
                f"{polort + 1} cols × {n_runs} runs)…"
            )
        data_clean, design_clean = project_out_nuisance_per_run(
            data=data_full,
            design=design_full,
            nuisance_per_run=nuisance_per_run,
            run_starts=run_starts,
            device=device,
            verbose=False,
        )
        # See fit_basis_fracridge for rationale — project_out forces
        # CPU storage for large datasets; LORO slicing needs them on
        # the compute device.
        if data_clean.device != device:
            data_clean = data_clean.to(device)
        if design_clean.device != device:
            design_clean = design_clean.to(device)
    else:
        data_clean, design_clean = data_full, design_full

    del data_full, design_full

    # Splits
    splits = generate_cv_splits(n_runs, strategy=leave_n_out, n_perms=n_perms)
    n_splits = len(splits)
    if verbose:
        print(
            f"  CV: {n_splits} splits (leave-{leave_n_out}-out), "
            f"{len(weight_grid)} prior weights"
            + (" + OLS baseline" if include_ols else "") + "."
        )

    # Per-run boundary tensor for slicing.
    run_ends = [run_starts[r] + n_tp_per_run[r] for r in range(n_runs)]

    # Cleaned data needs ss_tot computed AFTER nuisance projection
    # (because the projection removes per-run means; the relevant
    # SS_tot is the variance of the projected data).
    y_clean_mean = data_clean.mean(dim=1, keepdim=True)
    ss_tot_full = ((data_clean - y_clean_mean) ** 2).sum(dim=1)  # (n_voxels,)

    weights_to_run: list[float | str] = list(weight_grid)
    if include_ols:
        weights_to_run = ["OLS"] + weights_to_run

    n_weights = len(weights_to_run)
    r2_per_weight = np.zeros((n_voxels, n_weights), dtype=np.float32)

    # Loop weights × splits.  For each weight, accumulate ss_res across
    # folds in a single (n_voxels,) tensor — no full-prediction buffer
    # needed when LORO (each timepoint appears in exactly one fold).
    weights_iter = tqdm(
        list(enumerate(weights_to_run)), total=n_weights,
        desc="  CV weights", unit="weight",
        leave=False, disable=(not verbose) or n_weights <= 1,
    )
    for wi, w in weights_iter:
        ss_res_accum = torch.zeros(n_voxels, dtype=torch.float64, device=device)
        weights_iter.set_postfix_str("OLS" if w == "OLS" else f"λ×σ²={w}")
        for train_runs, test_runs in tqdm(
            splits, total=n_splits, desc="    folds", unit="fold",
            leave=False, disable=(not verbose) or n_splits <= 1,
        ):
            # Build train / test row-index lists from run boundaries.
            train_rows = torch.cat([
                torch.arange(run_starts[r], run_ends[r], device=device)
                for r in train_runs
            ])
            test_rows = torch.cat([
                torch.arange(run_starts[r], run_ends[r], device=device)
                for r in test_runs
            ])
            X_train = design_clean[train_rows]                 # (n_tp_train, n_task)
            X_test = design_clean[test_rows]
            y_train = data_clean[:, train_rows]                # (n_voxels, n_tp_train)
            y_test = data_clean[:, test_rows]

            # Fit on train.  We treat the cleaned design as a single
            # task block (no nuisance, polort=-1) and pass through
            # fit_basis_constrained_ridge with reconstruct_hrfs=False
            # (we only need task betas).
            if w == "OLS":
                # OLS: closed-form via normal equations.  In CV mode
                # everything stays in design.dtype (float32) for speed
                # and memory; the constrained primitive uses float64
                # internally but we cast back here.
                XtX = X_train.T @ X_train
                Xty = X_train.T @ y_train.T
                try:
                    L0 = torch.linalg.cholesky(XtX)
                    beta_chunk = torch.cholesky_solve(Xty, L0)
                except torch.linalg.LinAlgError:
                    beta_chunk = torch.linalg.lstsq(X_train, y_train.T).solution
                # beta_chunk: (n_task_cols, n_voxels)
                task_betas = beta_chunk.T
            else:
                fit = fit_basis_constrained_ridge(
                    data=y_train,
                    design_task=X_train,
                    basis_functions=basis_functions,
                    prior_mean=prior_mean,
                    prior_cov=prior_cov,
                    n_blocks=n_blocks,
                    nuisance=None,                         # already projected
                    prior_weight=float(w),
                    device=device,
                    reconstruct_hrfs=False,
                    lambda_mode=lambda_mode,
                    lambda_n_bins=lambda_n_bins,
                )
                # fit.betas is numpy float64 — cast to match X_test
                # dtype so the matmul below works.
                task_betas = torch.from_numpy(
                    fit.betas[:, :n_task_cols]
                ).to(device=device, dtype=X_test.dtype)

            # Predict cleaned test data using TASK betas only.
            y_test_pred = task_betas @ X_test.T              # (n_voxels, n_tp_test)
            ss_res_split = ((y_test - y_test_pred) ** 2).sum(dim=1)
            ss_res_accum += ss_res_split.double()
            del X_train, X_test, y_train, y_test, task_betas, y_test_pred, ss_res_split

        # Compute R² for this weight against the full ss_tot.
        r2_w = 1.0 - ss_res_accum / torch.clamp(ss_tot_full.double(), min=1e-30)
        r2_per_weight[:, wi] = r2_w.cpu().numpy()
        if verbose:
            print(
                f"    median held-out R² = {float(np.median(r2_per_weight[:, wi])):.4f}, "
                f"mean = {float(np.mean(r2_per_weight[:, wi])):.4f}"
            )

    argmax = np.argmax(r2_per_weight, axis=1).astype(np.int32)
    return FLOBSCVResult(
        weights=weights_to_run,
        r2_per_weight=r2_per_weight,
        argmax_weight_idx=argmax,
        n_splits=n_splits,
    )


@dataclass
class FracRidgeFitResult:
    """Output of :func:`fit_basis_fracridge`.

    Mirrors the fields of :class:`FLOBSFitResult` so the CLI can
    consume both interchangeably, plus fracridge-specific fields.

    Attributes
    ----------
    betas : np.ndarray, shape (n_voxels, n_task_cols)
        Per-voxel task coefficients at each voxel's CV-optimal frac.
        Nuisance was projected out up front, so there are no nuisance
        columns — ``betas[:, :n_task_cols]`` is the full task block.
    r2 : np.ndarray, shape (n_voxels,)
        Held-out R² at the optimal frac.
    betas_ols : np.ndarray, shape (n_voxels, n_task_cols)
        OLS task betas (frac=1.0).
    r2_ols : np.ndarray, shape (n_voxels,)
        Held-out R² at frac=1.0 (the OLS baseline).
    optimal_fracs : np.ndarray, shape (n_voxels,)
        Per-voxel argmax over the frac grid.
    r2_by_frac : np.ndarray, shape (n_voxels, n_fracs)
        Held-out R² for every frac on the grid.
    fracs : np.ndarray, shape (n_fracs,)
        The frac grid evaluated (highest-to-lowest by convention).
    n_iter : int
        Always 1 for fracridge (single SVD pass).
    sigma2_mean : float
        Mean per-voxel residual variance from OLS pre-pass.  Kept for
        :class:`FLOBSFitResult` API parity.
    effective_prior_weight : float
        Always 0 for fracridge — there is no MVN prior.  Kept for API
        parity.
    """

    betas: np.ndarray
    r2: np.ndarray
    betas_ols: np.ndarray
    r2_ols: np.ndarray
    optimal_fracs: np.ndarray
    r2_by_frac: np.ndarray
    fracs: np.ndarray
    n_iter: int = 1
    sigma2_mean: float = 0.0
    effective_prior_weight: float = 0.0


def fit_basis_lss(
    per_run_data: list[torch.Tensor],
    per_run_designs: list[torch.Tensor],
    block_labels: list[str],
    condition_labels: list[str],
    basis_functions: np.ndarray,
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    polort: int,
    *,
    prior_weight: float | str = "auto",
    lambda_mode: str = "global",
    lambda_n_bins: int = 20,
    lss_exclude: list[str] | None = None,
    lsa_fit: "FLOBSFitResult | None" = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> "FLOBSFitResult":
    """Least-Squares-Separate single-trial fit (LSS).

    Per-trial estimator commonly used in single-trial fMRI work
    (Mumford 2012, also GLMsingle / 3dLSS).  For each non-excluded
    trial, fit a small design with:

    - ``K`` cols for the current trial's onset × basis
    - ``K`` cols for the sum of all *other* trials in the trial's
      home condition
    - ``K`` cols per *other condition* (sum of all trials in that
      cond)

    The shape prior (m, C) is applied only to the **current trial's**
    K cols; the rest are nuisance.  Reduces the trial-to-trial
    coefficient collinearity that the all-at-once "LSA" path suffers
    from when trials are tightly packed.

    Excluded conditions (``lss_exclude``) contribute their summed
    K-col regressor to every LSS design but are not iterated over;
    their per-cond β comes from a parallel LSA fit (which the caller
    can also supply via ``lsa_fit`` to avoid redundant work).

    ``prior_weight_per_voxel`` is computed once from the LSA fit's
    per-voxel residual variance (``lsa_fit.sigma2_per_voxel``) and
    passed to every per-trial fit; this avoids each LSS solve
    estimating its own (partially-conditioned) σ² from too little
    data.

    Parameters mirror :func:`fit_basis_constrained_ridge` for the
    reg / prior knobs; ``polort`` is the per-run polynomial degree
    (projected out once, before the per-trial loop).

    Returns a :class:`FLOBSFitResult` whose ``betas`` and
    ``betas_ols`` match the standard single-trial layout
    ``(n_voxels, n_total_cols)`` — first ``n_blocks × n_basis``
    entries are the per-trial task betas (LSS for non-excluded
    trials, LSA for excluded), then per-run polynomial nuisance.
    """
    from fastfuncstuff.glm.xval import project_out_nuisance_per_run
    from fastfuncstuff.design.builder import legendre_polynomials

    if device is None:
        device = get_device()
    if lss_exclude is None:
        lss_exclude = []
    excluded_set = set(lss_exclude)

    n_basis = basis_functions.shape[0]
    n_runs = len(per_run_data)
    n_blocks = len(block_labels)
    n_voxels = per_run_data[0].shape[0]
    n_tp_per_run = [d.shape[1] for d in per_run_data]
    total_tp = int(sum(n_tp_per_run))
    run_starts = [0]
    for n in n_tp_per_run[:-1]:
        run_starts.append(run_starts[-1] + n)
    n_task_cols_full = n_blocks * n_basis

    # Map block_label → condition + trial-index-within-cond.
    cond_to_blocks: dict[str, list[int]] = {c: [] for c in condition_labels}
    block_to_cond: list[str] = []
    for b, label in enumerate(block_labels):
        cond, _, _ = str(label).partition("_trial")
        if cond not in cond_to_blocks:
            raise ValueError(
                f"block_label {label!r} has condition {cond!r} not in "
                f"condition_labels {condition_labels}"
            )
        cond_to_blocks[cond].append(b)
        block_to_cond.append(cond)

    # We need a full LSA single-trial fit for: (a) the unconstrained
    # baseline in betas_ols, (b) σ²_v for the LSS prior weight, (c) the
    # excluded conditions' main betas.  Caller can pre-supply ``lsa_fit``
    # to avoid redundant work; otherwise we run it here.
    if lsa_fit is None:
        if verbose:
            print("  LSS: running LSA pre-fit (provides σ²_v, excluded "
                  "conditions' main betas, and unconstrained baseline)…")
        from fastfuncstuff.design.builder import pack_for_shared_task_glm
        packed_lsa = pack_for_shared_task_glm(
            per_run_data=per_run_data,
            per_run_task_designs=per_run_designs,
            polort=polort, device=device,
        )
        lsa_fit = fit_basis_constrained_ridge(
            data=packed_lsa.data_concat,
            design_task=packed_lsa.design_concat[:, :n_task_cols_full],
            basis_functions=basis_functions,
            prior_mean=prior_mean, prior_cov=prior_cov,
            n_blocks=n_blocks,
            nuisance=(
                packed_lsa.design_concat[:, n_task_cols_full:]
                if packed_lsa.design_concat.shape[1] > n_task_cols_full
                else None
            ),
            prior_weight=prior_weight, device=device,
            reconstruct_hrfs=False,
            lambda_mode=lambda_mode, lambda_n_bins=lambda_n_bins,
            return_vb_diagnostics=True,
        )
        del packed_lsa

    user_mult = 1.0 if isinstance(prior_weight, str) else float(prior_weight)
    # When the user picked -reg none → prior_weight = 0 → no σ²·user_mult
    # work is needed at all (the per-trial fits hit the fast path).
    # Save the per-voxel σ² gathering for the cases that actually use it.
    if user_mult == 0.0:
        prior_pw_per_voxel = None
    else:
        sigma2_per_voxel = lsa_fit.sigma2_per_voxel
        if sigma2_per_voxel is None:
            # Caller passed an lsa_fit without diagnostics — fall back to a
            # uniform σ²_mean.  Less Bayesian-honest but won't crash.
            sigma2_per_voxel = np.full(
                n_voxels, float(lsa_fit.sigma2_mean), dtype=np.float32,
            )
        prior_pw_per_voxel = (sigma2_per_voxel * user_mult).astype(np.float32)

    # ── Project per-run polys out of data + per-trial designs ───────
    # After projection the LSS design is purely task (block 1..3 per
    # trial), no polynomial cols.  Each LSS fit then has just the
    # per-trial task block as ``task`` and the rest of the trial's
    # design as ``nuisance`` (un-penalised).
    data_full = torch.cat(
        [d.to(device).float() for d in per_run_data], dim=1,
    )                                                            # (n_vox, total_tp)
    design_full = torch.cat(
        [d.to(device).float() for d in per_run_designs], dim=0,
    )                                                            # (total_tp, n_blocks*K)
    if polort >= 0:
        nuisance_per_run = [
            torch.from_numpy(
                legendre_polynomials(n_tp_per_run[r], polort)
            ).to(device=device, dtype=torch.float32)
            for r in range(n_runs)
        ]
        if verbose:
            print(
                f"  LSS: projecting out per-run polynomials "
                f"(polort={polort}, {polort + 1} cols × {n_runs} runs)…"
            )
        data_clean, design_clean = project_out_nuisance_per_run(
            data=data_full, design=design_full,
            nuisance_per_run=nuisance_per_run,
            run_starts=run_starts, device=device, verbose=False,
        )
        if data_clean.device != device:
            data_clean = data_clean.to(device)
        if design_clean.device != device:
            design_clean = design_clean.to(device)
    else:
        data_clean, design_clean = data_full, design_full
    del data_full, design_full

    # Per-condition aggregated designs (sum of all that cond's trial
    # designs across runs).  Shape: (n_cond, total_tp, K).
    cond_full_designs: dict[str, torch.Tensor] = {}
    for c in condition_labels:
        block_ids = cond_to_blocks[c]
        if not block_ids:
            cond_full_designs[c] = torch.zeros(
                total_tp, n_basis, device=device, dtype=torch.float32,
            )
            continue
        # Slice design_clean columns for this cond's blocks and sum.
        accum = torch.zeros(
            total_tp, n_basis, device=device, dtype=torch.float32,
        )
        for b in block_ids:
            accum = accum + design_clean[:, b * n_basis:(b + 1) * n_basis]
        cond_full_designs[c] = accum

    # Initialize result betas from LSA — excluded conds keep their LSA
    # values; non-excluded conds will be overwritten by LSS below.
    betas_full = lsa_fit.betas.copy()
    betas_ols_full = lsa_fit.betas_ols.copy()

    # ── LSS per (non-excluded cond, trial in cond) ──────────────────
    non_excluded_blocks: list[int] = [
        b for b, c in enumerate(block_to_cond) if c not in excluded_set
    ]
    if not non_excluded_blocks:
        if verbose:
            print(
                "  LSS: all conditions are -lss-exclude'd; nothing to "
                "fit per-trial.  Returning LSA result."
            )
        return lsa_fit

    if verbose:
        n_excluded_blocks = n_blocks - len(non_excluded_blocks)
        excluded_str = (
            f"; {n_excluded_blocks} excluded-cond blocks keep LSA betas"
            if excluded_set else ""
        )
        print(
            f"  LSS: fitting {len(non_excluded_blocks)} trials "
            f"({len(condition_labels) - len(excluded_set)} active "
            f"conditions × ~{len(non_excluded_blocks) // max(1, len(condition_labels) - len(excluded_set))} "
            f"trials each){excluded_str}…"
        )

    # Per-trial loop.  Each fit is small (K_total ≈ 4-12 task cols × K)
    # so we delegate to fit_basis_constrained_ridge directly with
    # n_blocks=1 (only the current trial's K cols are penalised).
    trial_iter = tqdm(
        non_excluded_blocks, total=len(non_excluded_blocks),
        desc="  LSS trials", unit="trial",
        leave=False, disable=(not verbose) or len(non_excluded_blocks) <= 1,
    )
    for b in trial_iter:
        cond = block_to_cond[b]
        # Block 1: this trial's K cols.
        X_trial = design_clean[:, b * n_basis:(b + 1) * n_basis]    # (T, K)
        # Block 2: rest of this cond (cond_full − trial).
        X_rest_of_cond = cond_full_designs[cond] - X_trial            # (T, K)
        # Block 3: other conds' summed designs (one K-col block each).
        other_blocks: list[torch.Tensor] = []
        for other_c in condition_labels:
            if other_c == cond:
                continue
            other_blocks.append(cond_full_designs[other_c])
        if other_blocks:
            X_other_conds = torch.cat(other_blocks, dim=1)            # (T, K * (n_cond - 1))
            nuisance_trial = torch.cat(
                [X_rest_of_cond, X_other_conds], dim=1,
            )                                                          # (T, K + K*(n_cond - 1))
        else:
            nuisance_trial = X_rest_of_cond

        # Solve the LSS system: task = trial's K cols (penalised by
        # the user's chosen (m, C) prior); nuisance = everything else
        # (un-penalised).  σ²_v passed via prior_weight_per_voxel so
        # the partial fit doesn't re-estimate it.
        trial_fit = fit_basis_constrained_ridge(
            data=data_clean,
            design_task=X_trial,
            basis_functions=basis_functions,
            prior_mean=prior_mean, prior_cov=prior_cov,
            n_blocks=1,
            nuisance=nuisance_trial,
            prior_weight=prior_weight,
            prior_weight_per_voxel=prior_pw_per_voxel,
            device=device,
            reconstruct_hrfs=False,
            lambda_mode=lambda_mode, lambda_n_bins=lambda_n_bins,
        )
        # Overwrite this trial's K betas in the result with the LSS
        # estimate.  The LSA fit's OLS betas stay (they're the
        # baseline comparator).
        betas_full[:, b * n_basis:(b + 1) * n_basis] = (
            trial_fit.betas[:, :n_basis]
        )

    if verbose:
        print(f"  ✓ LSS fit complete ({len(non_excluded_blocks)} trials).")

    # Build the result.  R² of the LSS fit isn't trivially
    # comparable to the LSA fit (every trial sees a different model),
    # so we report the LSA R² unchanged — both fits explain the same
    # data, just split per-trial variance differently.  For real held-
    # out comparison the caller should use -xval-r2.
    return FLOBSFitResult(
        betas=betas_full,
        hrfs=None,                                  # type: ignore[arg-type]
        r2=lsa_fit.r2,                               # see above
        betas_ols=betas_ols_full,
        hrfs_ols=None,                              # type: ignore[arg-type]
        r2_ols=lsa_fit.r2_ols,
        sigma2_mean=lsa_fit.sigma2_mean,
        effective_prior_weight=lsa_fit.effective_prior_weight,
        n_iter=1,
        sigma2_per_voxel=lsa_fit.sigma2_per_voxel,
        lambda_per_voxel=lsa_fit.lambda_per_voxel,
    )


def fit_basis_fracridge(
    per_run_data: list[torch.Tensor],
    per_run_task_designs: list[torch.Tensor],
    n_blocks: int,
    n_basis: int,
    polort: int,
    *,
    fracs: np.ndarray | None = None,
    leave_n_out: int = 1,
    n_perms: int = 50,
    device: torch.device | None = None,
    verbose: bool = True,
) -> FracRidgeFitResult:
    """Constrained basis-set fit via fractional ridge regression.

    Alternative to the MVN-prior path: instead of constraining HRF
    *shape* via (m, C), fracridge constrains the OLS coefficient
    *norm* — keeping a fraction ``f ∈ [0, 1]`` of ``||β_OLS||`` per
    voxel.  No HRF prior; the only knob is the fraction.  CV across
    runs picks the per-voxel optimal fraction (GLMsingle convention).

    Strategy:

    1. Project per-run polynomials out of data + task design once.
    2. LORO over the cleaned data; for each fold, fit all fracs in a
       single SVD pass (``fastfuncstuff.glm.ridge._fit_ridge_multiple_fracs``),
       predict the held-out fold, accumulate per-frac SS_res.
    3. Per voxel, pick the frac with the highest held-out R².
    4. Re-fit all fracs on the **full** cleaned design (one more SVD
       pass) and gather per voxel at the optimal frac.

    Returns
    -------
    FracRidgeFitResult
    """
    from fastfuncstuff.glm.ridge import _fit_ridge_multiple_fracs
    from fastfuncstuff.glm.xval import (
        generate_cv_splits,
        project_out_nuisance_per_run,
    )
    from fastfuncstuff.design.builder import legendre_polynomials

    if device is None:
        device = get_device()
    if fracs is None:
        # ffs_ridge defaults; same grid GLMsingle uses by default.
        fracs = np.linspace(0.1, 1.0, 10).astype(np.float64)
    fracs = np.asarray(fracs, dtype=np.float64)
    n_fracs = len(fracs)

    n_runs = len(per_run_data)
    if n_runs < 2:
        raise ValueError(f"fracridge CV needs at least 2 runs; got {n_runs}")
    n_task_cols = n_blocks * n_basis

    n_tp_per_run = [d.shape[1] for d in per_run_data]
    run_starts = [0]
    for n in n_tp_per_run[:-1]:
        run_starts.append(run_starts[-1] + n)

    data_full = torch.cat([d.to(device).float() for d in per_run_data], dim=1)
    n_voxels = data_full.shape[0]
    design_full = torch.cat(
        [t.to(device).float() for t in per_run_task_designs], dim=0
    )

    if polort >= 0:
        nuisance_per_run = [
            torch.from_numpy(legendre_polynomials(n_tp_per_run[r], polort))
            .to(device).float()
            for r in range(n_runs)
        ]
        if verbose:
            print(
                f"  fracridge: projecting out per-run polynomials "
                f"(polort={polort}, {polort + 1} cols × {n_runs} runs)…"
            )
        data_clean, design_clean = project_out_nuisance_per_run(
            data=data_full,
            design=design_full,
            nuisance_per_run=nuisance_per_run,
            run_starts=run_starts,
            device=device,
            verbose=False,
        )
        # project_out_nuisance_per_run allocates the result on CPU
        # when the dataset is large (>~1 GB) — its design is built
        # for streaming, not for an in-loop LORO slice.  Push back
        # to the compute device so train_rows / test_rows (on cuda)
        # can index data_clean directly.  For 9.4T scale (~1.4 GB
        # data) this is ~1.4 GB of VRAM, easy on any modern card.
        if data_clean.device != device:
            data_clean = data_clean.to(device)
        if design_clean.device != device:
            design_clean = design_clean.to(device)
    else:
        data_clean, design_clean = data_full, design_full
    del data_full, design_full

    run_ends = [run_starts[r] + n_tp_per_run[r] for r in range(n_runs)]
    y_clean_mean = data_clean.mean(dim=1, keepdim=True)
    ss_tot_full = ((data_clean - y_clean_mean) ** 2).sum(dim=1).double()

    splits = generate_cv_splits(n_runs, strategy=leave_n_out, n_perms=n_perms)
    n_splits = len(splits)
    if verbose:
        print(
            f"  fracridge: {n_splits} splits (leave-{leave_n_out}-out), "
            f"{n_fracs} fracs in [{fracs.min():.2f}, {fracs.max():.2f}]"
        )

    # SS_res accumulator: (n_fracs, n_voxels), float64 for numeric stability.
    ss_res_accum = torch.zeros(n_fracs, n_voxels, dtype=torch.float64, device=device)

    # Voxel chunk size, used for BOTH:
    #  (a) the SVD fit itself — _fit_ridge_multiple_fracs materialises
    #      ``Vt.T @ ridge_flat`` of size ``n_features × (n_fracs · V)``,
    #      which at 9.4T single-trial scale (n_features=432, V=306k,
    #      n_fracs=10) is ~5 GB and OOMs on a 16 GB card.  Passing
    #      ``chunk_size`` switches the function to streaming-over-y
    #      mode (y stays on CPU, chunks of ``chunk_size`` voxels are
    #      moved to device per iteration, result returned on CPU).
    #  (b) the prediction step — y_pred + resid + (resid**2) is
    #      ``3 × n_tp_test × n_fracs × 4`` bytes per voxel, ~3.5 GB
    #      at 9.4T scale.
    #
    # The per-voxel cost is dominated by whichever phase has more
    # intermediates.  SVD phase per voxel: roughly
    # ``2 × n_features × n_fracs × 4`` bytes (ridge_coef_rotated_all
    # + coefs_flat slice).  Prediction phase per voxel: roughly
    # ``3 × n_tp_test_max × n_fracs × 4``.  Take the max, target 25 %
    # of free VRAM.
    n_features = design_clean.shape[1]
    n_tp_test_max = max(n_tp_per_run)
    if device.type == "cuda":
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
        except Exception:
            free_bytes = 4 * 1024 ** 3
        bytes_per_vox_svd = 2 * n_features * n_fracs * 4
        bytes_per_vox_pred = 3 * n_tp_test_max * n_fracs * 4
        bytes_per_vox = max(bytes_per_vox_svd, bytes_per_vox_pred)
        v_chunk = max(1, int(free_bytes * 0.25 / max(bytes_per_vox, 1)))
        v_chunk = min(v_chunk, n_voxels)
    else:
        v_chunk = n_voxels                                  # CPU: no chunking needed
    if verbose:
        n_v_chunks = (n_voxels + v_chunk - 1) // v_chunk
        print(
            f"  fracridge: SVD fit + prediction in {n_v_chunks} voxel "
            f"chunk{'s' if n_v_chunks > 1 else ''} of {v_chunk:,}"
        )

    splits_iter = tqdm(
        splits, total=n_splits,
        desc="  fracridge LORO", unit="fold",
        leave=False, disable=(not verbose) or n_splits <= 1,
    )
    for train_runs, test_runs in splits_iter:
        train_rows = torch.cat([
            torch.arange(run_starts[r], run_ends[r], device=device)
            for r in train_runs
        ])
        test_rows = torch.cat([
            torch.arange(run_starts[r], run_ends[r], device=device)
            for r in test_runs
        ])
        X_train = design_clean[train_rows]                  # (n_tp_tr, n_task)
        X_test = design_clean[test_rows]                    # (n_tp_te, n_task)
        y_train = data_clean[:, train_rows]                 # (n_voxels, n_tp_tr)
        y_test = data_clean[:, test_rows]                   # (n_voxels, n_tp_te)

        # _fit_ridge_multiple_fracs expects y as (n_samples, n_targets).
        # On cuda we pass chunk_size to keep ``Vt.T @ ridge_flat``
        # bounded — the non-chunked path materialises a 5+ GB tensor
        # at 9.4T single-trial scale.  The chunked path returns coefs
        # on CPU; we move slices to device during prediction.
        use_chunked = device.type == "cuda" and v_chunk < n_voxels
        coefs = _fit_ridge_multiple_fracs(
            X=X_train, y=y_train.T, fracs=fracs, device=device,
            chunk_size=(v_chunk if use_chunked else None),
        )                                                   # (n_task, n_fracs, n_voxels)
        coefs_on_cpu = use_chunked                          # tracks where coefs live

        # Chunked prediction: einsum materialises a (T, F, V) tensor
        # which is ~3.5 GB for 9.4T-scale data and 10 fracs.  Loop
        # voxel-chunks instead so peak VRAM is bounded.  If coefs are
        # on CPU from the chunked SVD path, move each slice to device
        # before the einsum.
        for v0 in range(0, n_voxels, v_chunk):
            v1 = min(v0 + v_chunk, n_voxels)
            coefs_chunk = coefs[:, :, v0:v1]                # (n_task, n_fracs, V_c)
            if coefs_on_cpu:
                coefs_chunk = coefs_chunk.to(device, non_blocking=True)
            y_pred = torch.einsum(
                "tf,fkv->tkv", X_test, coefs_chunk,
            )                                               # (T, n_fracs, V_c)
            resid = y_test[v0:v1].T.unsqueeze(1) - y_pred
            ss_res_accum[:, v0:v1] += (resid ** 2).sum(dim=0).double()
            del coefs_chunk, y_pred, resid

        del X_train, X_test, y_train, y_test, coefs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Held-out R² per frac per voxel.  ss_tot_full is (V,); broadcast to (F, V).
    r2_by_frac_t = 1.0 - ss_res_accum / torch.clamp(ss_tot_full.unsqueeze(0), min=1e-30)
    r2_by_frac = r2_by_frac_t.cpu().numpy().T.astype(np.float32)  # (V, n_fracs)

    optimal_fracs_idx = np.argmax(r2_by_frac, axis=1)               # (V,)
    optimal_fracs = fracs[optimal_fracs_idx].astype(np.float32)
    r2_optimal = r2_by_frac[np.arange(n_voxels), optimal_fracs_idx]

    # ── Final fit on full cleaned data, all fracs ────────────────────
    if verbose:
        print(
            f"  fracridge: final SVD fit on full data ({n_voxels:,} "
            f"voxels × {n_fracs} fracs)…"
        )
    # Same VRAM constraint as the LORO fit — use the chunked path on
    # cuda.  Result is on CPU, so the gather + numpy-cast at the end
    # happens on host (no extra D→H transfer beyond the final betas).
    use_chunked_final = device.type == "cuda" and v_chunk < n_voxels
    final_coefs = _fit_ridge_multiple_fracs(
        X=design_clean, y=data_clean.T, fracs=fracs, device=device,
        chunk_size=(v_chunk if use_chunked_final else None),
    )                                                       # (n_task, n_fracs, n_voxels)

    # Gather per-voxel optimal frac.  When the SVD path was chunked,
    # final_coefs lives on CPU; do gather on CPU too.
    if use_chunked_final:
        opt_idx_t = torch.from_numpy(optimal_fracs_idx).long()
    else:
        opt_idx_t = torch.from_numpy(optimal_fracs_idx).to(
            device=device, dtype=torch.long,
        )
    gather_idx = opt_idx_t.view(1, 1, -1).expand(n_task_cols, 1, -1)
    betas_opt = final_coefs.gather(1, gather_idx).squeeze(1)  # (n_task, n_voxels)
    betas = betas_opt.T.cpu().numpy().astype(np.float64) if betas_opt.is_cuda \
        else betas_opt.T.numpy().astype(np.float64)           # (n_voxels, n_task)

    # OLS baseline = frac=1.0 coefficients (last entry if grid ends at 1.0).
    if np.isclose(fracs[-1], 1.0):
        ols_idx = n_fracs - 1
    else:
        ols_idx = int(np.argmin(np.abs(fracs - 1.0)))
    betas_ols_slice = final_coefs[:, ols_idx, :]
    betas_ols = (
        betas_ols_slice.T.cpu().numpy().astype(np.float64)
        if betas_ols_slice.is_cuda
        else betas_ols_slice.T.numpy().astype(np.float64)
    )
    r2_ols = r2_by_frac[:, ols_idx]

    del final_coefs, betas_opt, betas_ols_slice
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # σ²_mean from OLS residuals on full cleaned data — preserved for
    # API parity and for downstream Bayesian-weight comparison with the
    # MVN path.  Chunked over voxels: materialising y_pred_ols + resid
    # full-size is 2 × (n_voxels × n_t) × 4 bytes, ~2.8 GB at 9.4T
    # scale, which combined with data_clean (1.4 GB) exhausts VRAM
    # right after the SVD fit.
    n_total_tp = data_clean.shape[1]
    dof = max(1, n_total_tp - n_task_cols)
    sigma2_sum = 0.0
    betas_ols_t = torch.from_numpy(betas_ols).to(
        device=device, dtype=design_clean.dtype,
    )
    Xt = design_clean.T                                     # (n_task, n_t)
    for v0 in range(0, n_voxels, v_chunk):
        v1 = min(v0 + v_chunk, n_voxels)
        y_pred_chunk = betas_ols_t[v0:v1] @ Xt              # (V_c, n_t)
        resid_chunk = data_clean[v0:v1] - y_pred_chunk
        sigma2_sum += float((resid_chunk ** 2).sum().item()) / dof
        del y_pred_chunk, resid_chunk
    sigma2_mean = sigma2_sum / max(1, n_voxels)
    del betas_ols_t, Xt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if verbose:
        print(
            f"  fracridge: median held-out R² = {float(np.median(r2_optimal)):.4f}, "
            f"mean = {float(np.mean(r2_optimal)):.4f}; "
            f"median optimal frac = {float(np.median(optimal_fracs)):.2f}"
        )

    return FracRidgeFitResult(
        betas=betas,
        r2=r2_optimal.astype(np.float32),
        betas_ols=betas_ols,
        r2_ols=r2_ols.astype(np.float32),
        optimal_fracs=optimal_fracs,
        r2_by_frac=r2_by_frac,
        fracs=fracs.astype(np.float32),
        n_iter=1,
        sigma2_mean=sigma2_mean,
        effective_prior_weight=0.0,
    )


@dataclass
class ARMAWhitenCell:
    """One ARMA(1,1) grid cell after binning + whitening.

    The voxels listed in :attr:`voxel_indices` all share the same
    ``(a, b)`` and therefore the same per-run Cholesky factor
    ``L_r``.  The whitened task design (and whitened polynomial
    nuisance) are stored *per run* and are shared across all voxels
    in this cell — the only per-voxel object is the whitened data.

    Attributes
    ----------
    a, b : float
        The grid-cell ARMA(1,1) parameters used to whiten this cell.
    voxel_indices : np.ndarray, shape (n_vox_cell,)
        Indices into the original full-voxel ordering — gather
        results back with this when stitching cells together.
    per_run_data : list[torch.Tensor], each (n_vox_cell, n_tp_run)
        Per-run data after applying L_r⁻¹ to each voxel timeseries.
    per_run_task_designs : list[torch.Tensor], each (n_tp_run, n_task)
        Per-run task design after applying L_r⁻¹.  Shared by every
        voxel in this cell.
    per_run_polys : list[torch.Tensor] | None
        Per-run polynomial nuisance after applying L_r⁻¹.  ``None``
        when ``polort < 0`` at the source call site.  Feed these to
        ``pack_for_shared_task_glm`` via ``extra_regressors_per_run``
        with ``polort=-1`` so the helper doesn't re-build *un-whitened*
        polys on top.
    """

    a: float
    b: float
    voxel_indices: np.ndarray
    per_run_data: list[torch.Tensor]
    per_run_task_designs: list[torch.Tensor]
    per_run_polys: list[torch.Tensor] | None


def estimate_arma11_per_voxel(
    per_run_data: list[torch.Tensor],
    per_run_task_designs: list[torch.Tensor],
    polort: int,
    *,
    device: torch.device | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """REML grid search for per-voxel ARMA(1,1) ``(a, b)``.

    Wraps ``fastfuncstuff.glm.arma`` primitives (the same code that
    powers ``ffs_reml``): builds the packed task + block-diagonal
    polynomial design, precomputes ``L⁻¹`` / Q / log-det matrices
    over the default AFNI 3dREMLfit grid (9 × 7 = 63 cells), and
    evaluates each voxel's REML likelihood against every grid point.

    Returns ``(n_voxels, 2)`` of grid-snapped ``(a, b)`` per voxel —
    suitable as input to :func:`bin_and_whiten_arma11`.
    """
    from fastfuncstuff.glm.arma import (
        get_default_arma_grids,
        precompute_reml_grid,
        search_voxels_precomputed_grid,
    )
    from fastfuncstuff.design.builder import legendre_polynomials

    if device is None:
        device = get_device()

    n_runs = len(per_run_data)
    n_tp_per_run = [d.shape[1] for d in per_run_data]
    run_starts: list[int] = [0]
    for n in n_tp_per_run[:-1]:
        run_starts.append(run_starts[-1] + n)
    total_tp = int(sum(n_tp_per_run))

    # Concat task design row-wise (shared task across runs).
    design_task = torch.cat(
        [t.to(device).float() for t in per_run_task_designs], dim=0
    )                                                          # (total_tp, n_task)

    # Build block-diagonal polynomial nuisance and concatenate.
    if polort >= 0:
        n_poly = polort + 1
        Z_full = torch.zeros(
            total_tp, n_runs * n_poly, device=device, dtype=torch.float32,
        )
        for r in range(n_runs):
            Z_r = torch.from_numpy(
                legendre_polynomials(n_tp_per_run[r], polort)
            ).to(device=device, dtype=torch.float32)
            Z_full[
                run_starts[r]:run_starts[r] + n_tp_per_run[r],
                r * n_poly:(r + 1) * n_poly,
            ] = Z_r
        X_full = torch.cat([design_task, Z_full], dim=1)
    else:
        X_full = design_task

    Y_full = torch.cat(
        [d.to(device).float() for d in per_run_data], dim=1
    )                                                          # (n_vox, total_tp)
    n_voxels = Y_full.shape[0]

    a_grid, b_grid = get_default_arma_grids(device)
    if verbose:
        print(
            f"  per-voxel ARMA: precomputing REML grid "
            f"({len(a_grid)} a × {len(b_grid)} b) over {total_tp} TR…"
        )

    # cholesky_on_cpu default = True in ffs_reml's typical large-grid
    # use case (saves VRAM at the cost of CPU work).  Here we're
    # specifically inside a GPU-targeted CLI: if the user asked for
    # cuda we precompute on cuda end-to-end so the BLAS-3 GEMMs in
    # the search loop don't fall to slow CPU paths.  MPS still gets
    # CPU Cholesky (float64 / triangular-solve flake on MPS).
    grid_on_gpu = device.type == "cuda"
    try:
        grid = precompute_reml_grid(
            X=X_full,
            n_timepoints=total_tp,
            a_grid=a_grid,
            b_grid=b_grid,
            device=device,
            verbose=verbose,
            cholesky_on_cpu=not grid_on_gpu,
            run_starts=run_starts,
        )
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        # OOM on GPU — fall back to CPU precompute, then move per-cell
        # entries to device inside the search loop.
        if grid_on_gpu and ("out of memory" in str(e).lower() or
                            isinstance(e, torch.cuda.OutOfMemoryError)):
            if verbose:
                print(
                    "  per-voxel ARMA: GPU precompute OOM; falling back "
                    "to cholesky_on_cpu=True (search loop will stream "
                    "grid entries device-side)."
                )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            grid = precompute_reml_grid(
                X=X_full,
                n_timepoints=total_tp,
                a_grid=a_grid,
                b_grid=b_grid,
                device=device,
                verbose=verbose,
                cholesky_on_cpu=True,
                run_starts=run_starts,
            )
        else:
            raise

    if not grid:
        raise RuntimeError(
            "precompute_reml_grid returned no valid (a, b) pairs — "
            "check that the design is well-conditioned."
        )

    # CRITICAL: precompute_reml_grid stores everything on CPU after the
    # Cholesky (designed to be loaded on-demand by ffs_reml's batching
    # path).  For a single big-batch search like ours, leaving the grid
    # on CPU means the L_inv @ Y.T BLAS-3 inside the search loop runs on
    # CPU — 117 grid points × multi-second CPU GEMM is ~25 minutes.
    # Move the entire grid to the compute device once before searching,
    # exactly like fit_glm_arma11 does.
    if device.type == "cuda":
        if verbose:
            print("  per-voxel ARMA: loading grid to GPU (one-time cost)…")
        for key in grid:
            for field in ("L_inv", "X_w", "Q", "logdet_Rcorr", "logdet_XwTXw"):
                if field in grid[key]:
                    grid[key][field] = grid[key][field].to(device)
        torch.cuda.empty_cache()

    # The actual device the grid sits on — informs where Y must live.
    grid_device = next(iter(grid.values()))["L_inv"].device
    Y_search = Y_full.to(grid_device) if Y_full.device != grid_device else Y_full
    X_search = X_full.to(grid_device) if X_full.device != grid_device else X_full

    # Voxel batching for the search.  Each grid iter materialises a
    # (n_tp, n_vox_batch) prewhitened-Y tensor and the search loop is
    # over (n_grid_points × n_batches).  Keep batches comfortably under
    # available memory; fall back to ~50k voxels when introspection
    # fails.
    if grid_device.type == "cuda":
        try:
            free_bytes, _ = torch.cuda.mem_get_info(grid_device)
        except Exception:
            free_bytes = 4 * 1024 ** 3
        # Per voxel cost across the inner loop: Y row (n_tp), Y_w row
        # (n_tp), one rss + qt_yw scratch.  ~4× n_tp float32 = safe.
        bytes_per_vox = 4 * total_tp * 4
        # Use ~25 % of free VRAM for the search workspace (the grid
        # tensors already occupy a chunk of the remaining 75 %).
        n_vox_batch = max(1, int(free_bytes * 0.25 / max(bytes_per_vox, 1)))
        n_vox_batch = min(n_vox_batch, n_voxels)
    else:
        # CPU path: keep batches modest to avoid blowing host memory.
        n_vox_batch = min(n_voxels, 50_000)

    n_batches = (n_voxels + n_vox_batch - 1) // n_vox_batch
    if verbose:
        print(
            f"  per-voxel ARMA: REML search over {n_voxels:,} voxels "
            f"on {grid_device} "
            f"({n_batches} batch{'es' if n_batches > 1 else ''} × "
            f"{n_vox_batch:,} vox)…"
        )

    best_params = torch.empty(n_voxels, 2, dtype=Y_search.dtype)
    batch_iter = tqdm(
        range(n_batches), total=n_batches,
        desc="  REML search batches", unit="batch",
        leave=False, disable=(not verbose) or n_batches <= 1,
    )
    for batch_idx in batch_iter:
        b0 = batch_idx * n_vox_batch
        b1 = min(b0 + n_vox_batch, n_voxels)
        bp, _ = search_voxels_precomputed_grid(
            X=X_search,
            Y_batch=Y_search[b0:b1],
            precomputed_grid=grid,
            device=grid_device,
            verbose=(verbose and batch_idx == 0),
        )
        best_params[b0:b1] = bp.detach().cpu()

    return best_params.numpy().astype(np.float32)


def bin_and_whiten_arma11(
    per_run_data: list[torch.Tensor],
    per_run_task_designs: list[torch.Tensor],
    arma_per_voxel: np.ndarray,
    polort: int,
    *,
    device: torch.device | None = None,
    verbose: bool = True,
) -> list[ARMAWhitenCell]:
    """Group voxels by ARMA grid cell, whiten data + design per cell.

    For each unique ``(a, b)`` in ``arma_per_voxel``: build the per-run
    Cholesky factor ``L_r``, apply ``L_r⁻¹`` via triangular solve to
    the (shared) task design, the per-run polynomial nuisance, and
    that cell's voxels' data.  Whitened polynomials come back as an
    explicit list — feed them to ``pack_for_shared_task_glm`` via
    ``extra_regressors_per_run`` (with ``polort=-1``) so the
    canonical packer doesn't add un-whitened polys back in.

    Cholesky factors are computed in **one batched call per unique
    run length** on the compute device — for the typical equal-runs
    case that's a single ``torch.linalg.cholesky((n_cells, n_t,
    n_t))`` launch on GPU instead of ``n_cells × n_runs`` separate
    CPU calls.
    """
    from fastfuncstuff.glm.arma import (
        build_arma11_covariance,
        build_arma11_covariance_batch,
    )
    from fastfuncstuff.design.builder import legendre_polynomials

    if device is None:
        device = get_device()

    n_runs = len(per_run_data)
    n_tp_per_run = [d.shape[1] for d in per_run_data]
    if arma_per_voxel.shape[1] != 2:
        raise ValueError(
            f"arma_per_voxel must be (n_voxels, 2); got {arma_per_voxel.shape}"
        )

    # Pre-build per-run polynomials (un-whitened, on device) — shared
    # across cells; only the L_r⁻¹ application differs per cell.
    if polort >= 0:
        polys_raw_per_run = [
            torch.from_numpy(
                legendre_polynomials(n_tp_per_run[r], polort)
            ).to(device=device, dtype=torch.float32)
            for r in range(n_runs)
        ]
    else:
        polys_raw_per_run = None

    design_task_per_run_dev = [
        t.to(device).float() for t in per_run_task_designs
    ]

    # Unique (a, b) cells (rounded to 6 decimals to avoid float noise).
    rounded = np.round(arma_per_voxel.astype(np.float64), 6)
    keys = [tuple(r.tolist()) for r in rounded]
    unique_keys = sorted(set(keys))

    if verbose:
        print(
            f"  per-voxel ARMA: {len(unique_keys)} unique (a, b) cells "
            f"covering {arma_per_voxel.shape[0]:,} voxels"
        )

    # ── Batched Cholesky per unique run length ──────────────────────
    # For the typical equal-runs case this is ONE batched call on the
    # compute device instead of ``len(unique_keys) × n_runs`` small
    # CPU Choleskys (which was the visible CPU usage in the cell loop).
    # We index L_per_run[run_idx][cell_idx] inside the whitening loop.
    unique_run_lengths = sorted(set(n_tp_per_run))
    # Build R_batch for each (a, b) at each unique length once.
    R_by_length: dict[int, tuple[torch.Tensor, list[tuple[float, float]]]] = {}
    a_tensor = torch.tensor([k[0] for k in unique_keys],
                            dtype=torch.float32, device=device)
    b_tensor = torch.tensor([k[1] for k in unique_keys],
                            dtype=torch.float32, device=device)
    if verbose:
        print(
            f"  ARMA whiten: batched Cholesky for {len(unique_keys)} cells "
            f"× {len(unique_run_lengths)} unique run length(s) on {device}…"
        )
    for n_t in unique_run_lengths:
        # build_arma11_covariance_batch takes (a_grid, b_grid) and
        # returns all valid pairs.  We want the diagonal — a[i] paired
        # with b[i] — so call it per cell length and pull out the
        # subset we asked for.  In practice the valid set drops some
        # pairs (λ ≤ 0), so we fall back to per-cell on misses.
        R_one_grid, _, param_list_one = build_arma11_covariance_batch(
            a_tensor, b_tensor, n_t, device, dtype=torch.float32,
        )
        # param_list_one is the Cartesian product, not the diagonal.
        # Map (a, b) → index into the returned batch.
        lookup = {(float(p[0]), float(p[1])): i for i, p in enumerate(param_list_one)}
        R_select_idx: list[int] = []
        missing: list[int] = []
        for ci, (a, b) in enumerate(unique_keys):
            j = lookup.get((float(a), float(b)))
            if j is None:
                missing.append(ci)
                R_select_idx.append(-1)
            else:
                R_select_idx.append(j)
        R_select_idx_t = torch.tensor(R_select_idx, dtype=torch.long, device=device)
        # Build the per-cell stack via index_select where valid, else
        # fall back to the scalar call (rare; happens at λ <= 0 corners).
        R_stack = torch.empty(
            (len(unique_keys), n_t, n_t), dtype=torch.float32, device=device,
        )
        valid_mask = R_select_idx_t >= 0
        if valid_mask.any():
            R_stack[valid_mask] = R_one_grid.index_select(
                0, R_select_idx_t[valid_mask],
            )
        for ci in missing:
            a, b = unique_keys[ci]
            R_r = build_arma11_covariance(
                float(a), float(b), n_t, device, dtype=torch.float32,
            )
            if R_r is None:
                raise ValueError(
                    f"build_arma11_covariance failed for (a={a}, b={b}, n={n_t}); "
                    "(a, b) was selected per-voxel but the grid rejects it."
                )
            R_stack[ci] = R_r
        # Batched Cholesky on device — one kernel launch.
        L_stack = torch.linalg.cholesky(R_stack)
        del R_stack, R_one_grid, R_select_idx_t, valid_mask
        # Cache: keyed by run length, then (a, b) → L tensor.
        R_by_length[n_t] = (L_stack, [k for k in unique_keys])

    # Per-run L lookup: run r uses the L_stack for length n_tp_per_run[r].
    L_per_run = []
    for r in range(n_runs):
        L_stack, keys_in_stack = R_by_length[n_tp_per_run[r]]
        L_per_run.append({k: L_stack[i] for i, k in enumerate(keys_in_stack)})

    cells: list[ARMAWhitenCell] = []
    cell_iter = tqdm(
        unique_keys, total=len(unique_keys),
        desc="  ARMA whiten cells", unit="cell",
        leave=False, disable=(not verbose) or len(unique_keys) <= 1,
    )
    for a, b in cell_iter:
        cell_vox = np.where(
            (rounded[:, 0] == a) & (rounded[:, 1] == b)
        )[0].astype(np.int64)
        if cell_vox.size == 0:
            continue

        task_w_runs: list[torch.Tensor] = []
        poly_w_runs: list[torch.Tensor] = [] if polys_raw_per_run is not None else None  # type: ignore[assignment]
        data_w_runs: list[torch.Tensor] = []
        for r in range(n_runs):
            L_r = L_per_run[r][(a, b)]

            # Whiten task design (shared across all voxels in cell).
            X_task_r = design_task_per_run_dev[r]
            task_w_runs.append(
                torch.linalg.solve_triangular(L_r, X_task_r, upper=False)
            )

            # Whiten polynomial nuisance (shared too).
            if polys_raw_per_run is not None:
                Z_w = torch.linalg.solve_triangular(
                    L_r, polys_raw_per_run[r], upper=False,
                )
                poly_w_runs.append(Z_w)

            # Whiten this cell's voxels' data.
            y_cell_r = (
                per_run_data[r][cell_vox].to(device).float()
            )                                                  # (n_vox_cell, n_tp_r)
            y_cell_w = torch.linalg.solve_triangular(
                L_r, y_cell_r.T, upper=False,
            ).T
            data_w_runs.append(y_cell_w)
            del y_cell_r, y_cell_w

        cells.append(ARMAWhitenCell(
            a=float(a), b=float(b),
            voxel_indices=cell_vox,
            per_run_data=data_w_runs,
            per_run_task_designs=task_w_runs,
            per_run_polys=poly_w_runs,
        ))

    return cells


def compute_vb_block_trace(
    design: torch.Tensor,
    prior_cov: np.ndarray,
    n_blocks: int,
    n_basis: int,
    lambda_per_voxel: np.ndarray,
    sigma2_per_voxel: np.ndarray,
    *,
    device: torch.device | None = None,
    n_bins: int = 20,
    verbose: bool = False,
) -> np.ndarray:
    """Per-voxel block-marginal posterior trace for the VB β_size update.

    Computes ``σ²_v · Σ_b tr(C⁻¹ · (A⁻¹)_b)`` per voxel, where:

    - ``A = X'X + λ_v · P`` is the joint precision used by the
      constrained solver (P is block-diag(C⁻¹) on the task cols, zero
      on nuisance).
    - ``(A⁻¹)_b`` is the K × K block-marginal posterior covariance
      for task block ``b`` (single-trial: per trial; per-condition:
      per condition).
    - ``σ²_v`` rescales because the actual posterior covariance is
      ``σ²_v · A⁻¹``.

    Voxels are binned by ``λ_v`` quantile (default 20 bins); within a
    bin all voxels share ``A_bin``, so we invert it *once* per bin and
    extract the block diagonals.  The output is ``σ²_v · trace_bin``
    for the voxel's bin, applied per-voxel.

    Returns
    -------
    np.ndarray, shape (n_voxels,)
        Total per-voxel ``σ²_v · Σ_b tr(C⁻¹ · (A⁻¹)_b)``.  Plug into
        :func:`vb_update_beta_size` as ``block_trace_summed``.
    """
    if device is None:
        device = get_device()

    X = design.to(device=device, dtype=torch.float64)
    n_cols = X.shape[1]
    n_task_cols = n_blocks * n_basis
    if n_task_cols > n_cols:
        raise ValueError(
            f"design has {n_cols} cols but n_blocks × n_basis = "
            f"{n_task_cols} task cols expected"
        )

    n_voxels = lambda_per_voxel.shape[0]
    if sigma2_per_voxel.shape[0] != n_voxels:
        raise ValueError(
            f"sigma2_per_voxel has {sigma2_per_voxel.shape[0]} entries "
            f"but lambda_per_voxel has {n_voxels}"
        )

    # Build P matching the solver: block-diag(C⁻¹) on task cols.
    C_inv = torch.from_numpy(np.linalg.inv(prior_cov)).to(
        device=device, dtype=torch.float64,
    )
    P = torch.zeros((n_cols, n_cols), dtype=torch.float64, device=device)
    for c in range(n_blocks):
        s = c * n_basis
        P[s:s + n_basis, s:s + n_basis] = C_inv
    XtX = X.T @ X

    # Quantile-bin λ_v.
    lambda_t = torch.from_numpy(lambda_per_voxel.astype(np.float64))
    n_bins_eff = max(1, min(int(n_bins), n_voxels))
    if n_bins_eff > 1:
        qs = torch.linspace(0.0, 1.0, n_bins_eff + 1)[1:-1]
        edges = torch.quantile(lambda_t, qs.to(lambda_t.dtype))
        voxel_bin = torch.bucketize(lambda_t, edges)
    else:
        voxel_bin = torch.zeros(n_voxels, dtype=torch.long)

    # Per-bin λ and per-bin trace (sum over task blocks).
    bin_trace_summed = np.zeros(n_bins_eff, dtype=np.float64)
    bins_iter = range(n_bins_eff)
    if verbose:
        bins_iter = tqdm(
            bins_iter, total=n_bins_eff,
            desc="  VB block-trace bins", unit="bin", leave=False,
            disable=n_bins_eff <= 1,
        )
    for b in bins_iter:
        mask_b = voxel_bin == b
        if not mask_b.any():
            continue
        lam_b = float(lambda_t[mask_b].median().item())
        A_b = XtX + lam_b * P                              # (n_cols, n_cols)
        # A⁻¹ via Cholesky; fall back to direct inverse on failure
        # (only happens at λ=0 + rank-deficient X — vanishingly rare
        # in the VB path because λ_v > 0 after the first iter).
        try:
            L_b = torch.linalg.cholesky(A_b)
            eye_n = torch.eye(n_cols, dtype=A_b.dtype, device=device)
            A_inv_b = torch.cholesky_solve(eye_n, L_b)
        except torch.linalg.LinAlgError:
            A_inv_b = torch.linalg.pinv(A_b)
        # Sum trace(C⁻¹ · (A⁻¹)_block) over task blocks.
        t_sum = 0.0
        for c in range(n_blocks):
            s = c * n_basis
            block = A_inv_b[s:s + n_basis, s:s + n_basis]
            t_sum += float((C_inv * block).sum().item())   # tr(C⁻¹ · block)
        bin_trace_summed[b] = t_sum
        del A_b, A_inv_b

    # Broadcast per voxel: σ²_v · bin_trace_summed[bin_of_v].
    voxel_bin_np = voxel_bin.numpy()
    return (
        sigma2_per_voxel.astype(np.float64) * bin_trace_summed[voxel_bin_np]
    ).astype(np.float32)


def vb_update_beta_size(
    task_betas: np.ndarray,
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    block_trace_summed: np.ndarray,
    *,
    c_prior: float = 1.0,
    d_prior: float = 1.0,
    floor: float = 1e-12,
) -> np.ndarray:
    """Filmbabe-style VB update for the per-voxel β_size (prior precision).

    Following FMRIB TR04MW2 §3 (gamma-conjugate posterior on β_size):

    .. math::

        \\hat\\beta_{size,v} = \\frac{n_b K / 2 + c_0}
                                    {\\tfrac12 \\sum_b ||\\beta_{v,b} - m||^2_{C^{-1}}
                                     + \\tfrac12 \\, \\mathrm{tr}_{v} + d_0}

    where ``tr_v = σ²_v · Σ_b tr(C⁻¹ · Σ_β,b,v)`` is the block-marginal
    posterior-covariance trace computed by
    :func:`compute_vb_block_trace`.  ``c_prior``, ``d_prior`` are the
    gamma-prior hyperparameters (default ``0`` → non-informative).

    Parameters
    ----------
    task_betas : np.ndarray, shape (n_voxels, n_blocks, K)
        Current posterior mean of the task betas per voxel × block.
    prior_mean : np.ndarray, shape (K,)
    prior_cov : np.ndarray, shape (K, K)
    block_trace_summed : np.ndarray, shape (n_voxels,)
        Output of :func:`compute_vb_block_trace`.
    c_prior, d_prior : float, default 1.0 (weakly informative)
        Gamma-prior hyperparameters on β_size.  ``c=d=0`` reproduces
        the non-informative MAP estimate but is **unstable**: when the
        shape prior tightens, ``||β-m||²_{C⁻¹}`` → 0 and the trace
        term shrinks too, so β_size diverges.  Filmbabe uses weakly
        informative ``c=d=1`` to bound the update; that's the default
        here.  Larger ``c, d`` pull the update toward
        ``β_size ≈ c/d = 1`` (the prior mean) and damp adaptation.
    floor : float
        Numerical clamp on the denominator.

    Returns
    -------
    np.ndarray, shape (n_voxels,)
        Updated β_size_v.  Plug into the next constrained-fit call as
        the per-voxel multiplier on σ²:
        ``prior_weight_per_voxel = beta_size_v * sigma2_per_voxel``.
    """
    K = prior_mean.size
    n_voxels, n_blocks, K_check = task_betas.shape
    if K_check != K:
        raise ValueError(
            f"task_betas last dim is {K_check} but prior_mean has {K} elements"
        )
    if block_trace_summed.shape != (n_voxels,):
        raise ValueError(
            f"block_trace_summed must have shape ({n_voxels},); "
            f"got {block_trace_summed.shape}"
        )

    C_inv = np.linalg.inv(prior_cov).astype(np.float64)
    diffs = task_betas.astype(np.float64) - prior_mean[None, None, :]
    # ||β_b - m||²_{C⁻¹} per (vox, block), summed across blocks.
    quad = np.einsum("vbi,ij,vbj->vb", diffs, C_inv, diffs).sum(axis=1)

    numer = 0.5 * n_blocks * K + c_prior
    denom = 0.5 * quad + 0.5 * block_trace_summed.astype(np.float64) + d_prior
    return (numer / np.maximum(denom, floor)).astype(np.float32)


def compute_xval_r2_per_voxel(
    per_run_data: list[torch.Tensor],
    all_onsets: list[list[np.ndarray]],
    condition_labels: list[str],
    basis_functions: np.ndarray,
    basis_lag_times: np.ndarray,
    basis_mode: str,
    tr: float,
    n_tp_per_run: list[int],
    polort: int,
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    prior_weight: float | str,
    *,
    single_trials: bool,
    single_trial_betas: np.ndarray | None = None,
    block_labels: list[str] | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """LORO cross-validated R² per voxel.

    Per-condition mode (``single_trials=False``):
        Standard LORO — train on N−1 runs with the user's
        ``-reg`` / prior config, predict the held-out run with the
        per-condition task betas, aggregate ``Σ ss_res / Σ ss_tot``
        across folds.

    Single-trial mode (``single_trials=True``):
        **No re-fitting** — the single-trial fit's task design has
        one column per (cond, run, trial), and those columns are
        time-disjoint across runs (each trial only has non-zero
        support inside its home run).  ``X'X`` is therefore
        block-diagonal across runs at the task level, so each
        trial's β depends *only* on its home run's data.  Subsetting
        the existing single-trial betas to "trials in train runs",
        averaging within condition, and predicting the held-out run
        with the per-condition design gives a faithful held-out R²
        for free — no per-fold re-fit needed.

        Requires ``single_trial_betas`` (shape
        ``(n_vox, n_blocks, K)``, from the main fit) and the parallel
        ``block_labels`` list so we can parse each block's home run
        from its label (format ``"{cond}_trial{NNN}_run{R}"``).

        Useful diagnostic: if the held-out R² is low while the
        in-sample R² is high, the single-trial estimates are
        oscillating around the mean rather than tracking the
        cross-run condition response.

    Returns ``(n_voxels,)`` of held-out R² values.
    """
    from fastfuncstuff.glm.xval import generate_cv_splits
    from fastfuncstuff.design.builder import (
        legendre_polynomials, pack_for_shared_task_glm,
    )
    from fastfuncstuff.design.hrf_derive import build_pc_basis_design_per_run

    if device is None:
        device = get_device()

    n_runs = len(per_run_data)
    if n_runs < 2:
        raise ValueError(f"xval needs ≥2 runs; got {n_runs}")
    n_voxels = per_run_data[0].shape[0]
    n_basis = basis_functions.shape[0]
    n_cond = len(condition_labels)

    # Per-condition designs for ALL runs — used for prediction in
    # both single-trial and per-condition modes.
    cond_designs_per_run: list[torch.Tensor] = []
    for r in range(n_runs):
        cond_blocks: list[np.ndarray] = []
        for c in range(n_cond):
            bd = build_pc_basis_design_per_run(
                onsets_per_run=[all_onsets[c][r]],
                pcs=basis_functions, lag_times=basis_lag_times,
                tr=tr, n_timepoints_per_run=[n_tp_per_run[r]],
                basis=basis_mode,
            )
            cond_blocks.append(bd[0])
        cond_designs_per_run.append(
            torch.from_numpy(
                np.concatenate(cond_blocks, axis=1).astype(np.float32)
            ).to(device)
        )

    splits = generate_cv_splits(n_runs, strategy=1)
    n_splits = len(splits)
    if verbose:
        print(
            f"  xval R²: {n_splits} folds (LORO), "
            f"mode={'single-trial → cond-average' if single_trials else 'per-condition'}"
        )

    ss_res_accum = torch.zeros(n_voxels, dtype=torch.float64, device=device)
    ss_tot_accum = torch.zeros(n_voxels, dtype=torch.float64, device=device)

    fold_iter = tqdm(
        splits, total=n_splits, desc="  xval R² folds", unit="fold",
        leave=False, disable=(not verbose) or n_splits <= 1,
    )
    for train_runs, test_runs in fold_iter:
        # ── Train fit ────────────────────────────────────────────────
        per_run_data_train = [per_run_data[r] for r in train_runs]
        if single_trials:
            # No re-fitting: take the already-computed single-trial
            # betas from the main fit, restrict to "trials whose
            # home run is in train_runs", and average per condition.
            # Block layout is shared with the main fit's block_labels
            # parameter — each label is "{cond}_trial{NNN}_run{R}",
            # where R is 1-indexed.
            cond_betas_train = np.zeros(
                (n_voxels, n_cond, n_basis), dtype=np.float64,
            )
            counts = np.zeros(n_cond, dtype=np.int64)
            test_run_set = set(test_runs)
            cond_label_to_idx = {c: i for i, c in enumerate(condition_labels)}
            assert single_trial_betas is not None and block_labels is not None
            for b_idx, label in enumerate(block_labels):
                # Parse "{cond}_trial{NNN}_run{R}" → cond name, run idx (0-based).
                cond_name, _, tail = str(label).partition("_trial")
                # tail is "{NNN}_run{R}"
                run_part = tail.split("_run", 1)[1]
                home_run = int(run_part) - 1
                if home_run in test_run_set:
                    continue                                # exclude test trials
                c = cond_label_to_idx[cond_name]
                cond_betas_train[:, c, :] += single_trial_betas[:, b_idx, :]
                counts[c] += 1
            # Voxels with zero train trials of some condition → leave
            # at zero (no estimate available; will predict zero for
            # that condition in this fold).  Avoid div-by-zero.
            cond_betas_train /= np.maximum(counts[None, :, None], 1)
            cond_betas_train_t = torch.from_numpy(cond_betas_train).to(
                device=device, dtype=torch.float32,
            )
        else:
            # Per-condition train fit.
            train_designs = [cond_designs_per_run[r] for r in train_runs]
            packed_train = pack_for_shared_task_glm(
                per_run_data=per_run_data_train,
                per_run_task_designs=train_designs,
                polort=polort, device=device,
            )
            n_task_cols_train = packed_train.n_task_cols
            train_fit = fit_basis_constrained_ridge(
                data=packed_train.data_concat,
                design_task=packed_train.design_concat[:, :n_task_cols_train],
                basis_functions=basis_functions,
                prior_mean=prior_mean, prior_cov=prior_cov,
                n_blocks=n_cond,
                nuisance=(
                    packed_train.design_concat[:, n_task_cols_train:]
                    if packed_train.design_concat.shape[1] > n_task_cols_train
                    else None
                ),
                prior_weight=prior_weight, device=device,
                reconstruct_hrfs=False, lambda_mode="global",
            )
            cond_betas_train_t = torch.from_numpy(
                train_fit.betas[:, :n_task_cols_train].reshape(
                    n_voxels, n_cond, n_basis,
                )
            ).to(device=device, dtype=torch.float32)

        # ── Predict test runs ────────────────────────────────────────
        cond_betas_flat = cond_betas_train_t.reshape(n_voxels, n_cond * n_basis)
        for r in test_runs:
            X_test_cond = cond_designs_per_run[r]                    # (n_tp_r, n_cond*K)
            test_data = per_run_data[r].to(device).float()           # (n_vox, n_tp_r)
            # Project out polys per-run from BOTH data and design
            # so the prediction is comparable on the same residual
            # subspace the fit operates in.
            if polort >= 0:
                Z_r = torch.from_numpy(
                    legendre_polynomials(n_tp_per_run[r], polort)
                ).to(device=device, dtype=torch.float32)
                Q_z, _ = torch.linalg.qr(Z_r)
                test_data_proj = test_data - (test_data @ Q_z) @ Q_z.T
                X_test_proj = X_test_cond - Q_z @ (Q_z.T @ X_test_cond)
            else:
                test_data_proj = test_data
                X_test_proj = X_test_cond
            y_pred = cond_betas_flat @ X_test_proj.T                 # (n_vox, n_tp_r)
            resid = test_data_proj - y_pred
            mean_test = test_data_proj.mean(dim=1, keepdim=True)
            ss_res_accum += (resid ** 2).sum(dim=1).double()
            ss_tot_accum += ((test_data_proj - mean_test) ** 2).sum(dim=1).double()
            del test_data, test_data_proj, X_test_proj, y_pred, resid

    r2 = 1.0 - ss_res_accum / torch.clamp(ss_tot_accum, min=1e-30)
    return r2.cpu().numpy().astype(np.float32)


def compute_per_voxel_residuals(
    per_run_data: list[torch.Tensor],
    per_run_task_designs: list[torch.Tensor],
    polort: int,
    *,
    task_betas: np.ndarray,
    nuisance_betas: np.ndarray | None,
    device: torch.device | None = None,
) -> list[torch.Tensor]:
    """Compute residuals ``e = y − (X_task β_task + Z β_nuis)`` per run.

    Works in the **original (un-whitened) space** even when the betas
    were fit in a whitened domain.  Under prewhitening, the constrained
    GLS β estimates are identical for whitened and un-whitened X
    (the whitening just transforms the noise covariance, not the
    parameter); so the per-voxel betas plug into the un-whitened
    design without further transformation.

    Used by the VB-style iterative loop to feed residuals back into a
    fresh per-voxel ARMA REML estimate.

    Parameters
    ----------
    per_run_data : list[Tensor], each (n_vox, n_tp_run)
        Original (un-whitened) data per run.
    per_run_task_designs : list[Tensor], each (n_tp_run, n_task)
        Original task design per run.
    polort : int
        Polynomial nuisance degree (``-1`` disables; matches the
        original fit's choice).
    task_betas : np.ndarray, shape (n_vox, n_task)
        Per-voxel task coefficients.
    nuisance_betas : np.ndarray | None, shape (n_vox, n_runs × (polort+1))
        Per-voxel nuisance coefficients in block-diagonal layout
        (run-major: cols ``r·k:(r+1)·k`` belong to run ``r``).
        ``None`` skips the nuisance contribution.

    Returns
    -------
    list[Tensor], each (n_vox, n_tp_run)
        Residuals per run (on ``device``).
    """
    from fastfuncstuff.design.builder import legendre_polynomials

    if device is None:
        device = get_device()

    n_runs = len(per_run_data)
    n_poly = polort + 1 if polort >= 0 else 0
    task_betas_t = torch.from_numpy(task_betas).to(device=device, dtype=torch.float32)
    nuis_betas_t = (
        torch.from_numpy(nuisance_betas).to(device=device, dtype=torch.float32)
        if (nuisance_betas is not None and n_poly > 0)
        else None
    )

    residuals: list[torch.Tensor] = []
    for r in range(n_runs):
        n_tp_r = per_run_data[r].shape[1]
        X_task_r = per_run_task_designs[r].to(device).float()    # (n_tp_r, n_task)
        y_r = per_run_data[r].to(device).float()                 # (n_vox, n_tp_r)

        # Task prediction: (n_vox, n_task) @ (n_task, n_tp_r) → (n_vox, n_tp_r)
        y_pred = task_betas_t @ X_task_r.T

        # Nuisance prediction (run r's slice of the block-diag betas).
        if nuis_betas_t is not None and n_poly > 0:
            Z_r = torch.from_numpy(
                legendre_polynomials(n_tp_r, polort)
            ).to(device=device, dtype=torch.float32)             # (n_tp_r, n_poly)
            nuis_beta_r = nuis_betas_t[:, r * n_poly:(r + 1) * n_poly]
            y_pred = y_pred + nuis_beta_r @ Z_r.T

        residuals.append(y_r - y_pred)

    return residuals


def estimate_and_apply_arma11_prewhitening(
    per_run_data: list[torch.Tensor],
    per_run_task_designs: list[torch.Tensor],
    polort: int,
    *,
    device: torch.device | None = None,
    verbose: bool = True,
) -> tuple[list[torch.Tensor], list[torch.Tensor], float, float]:
    """Global ARMA(1,1) prewhitening foundation for the VB basis-set fit.

    First step toward the iterative Variational Bayes loop (cf. FMRIB
    TR04MW2 §3): replace the i.i.d. noise assumption with a temporal
    autocorrelation model, prewhiten, then run the existing
    constrained-basis solver on the prewhitened design / data.

    This is the "single-shot, global (a,b)" variant — the simplest
    useful step beyond OLS noise.  Phase B (per-voxel ARMA) and
    Phase C (alternating β / noise updates) build on top of this
    scaffold but are not yet implemented.

    Strategy:

    1. OLS pre-pass on each per-run (task design + polynomial
       nuisance) → residuals.
    2. Average residuals across voxels per run; concatenate.
    3. Fit a single global ``(a, b)`` via
       :func:`fastfuncstuff.glm.arma.reml_grid_search` on the mean
       residual timeseries vs an all-zeros design (we only want the
       noise-covariance estimate).
    4. Per run: build ARMA(1,1) Cholesky factor and apply
       ``L⁻¹`` to both ``y`` and ``X`` via triangular solve.

    The returned lists feed the existing
    :func:`fit_basis_constrained_ridge` /
    :func:`fit_basis_fracridge` paths unchanged.

    Parameters
    ----------
    per_run_data, per_run_task_designs
        Same shapes as the rest of the fit primitives consume.
    polort
        Polynomial-nuisance order for the OLS pre-pass.  ``-1``
        disables.  Used only to estimate residuals for ARMA fitting;
        the returned prewhitened designs do *not* have polys
        embedded — callers add them downstream as usual.
    """
    from fastfuncstuff.glm.arma import (
        build_arma11_covariance,
        reml_grid_search,
    )
    from fastfuncstuff.design.builder import legendre_polynomials

    if device is None:
        device = get_device()

    n_runs = len(per_run_data)
    if n_runs < 1:
        raise ValueError("Need at least one run.")
    if len(per_run_task_designs) != n_runs:
        raise ValueError(
            f"per_run_task_designs has {len(per_run_task_designs)} entries "
            f"but per_run_data has {n_runs}."
        )

    # ── 1+2. OLS pre-pass per run; pool residuals across voxels. ─────
    if verbose:
        print(f"  prewhiten: OLS pre-pass on {n_runs} runs to estimate ARMA(1,1)…")

    residual_mean_per_run: list[np.ndarray] = []
    for y_r, X_task_r in zip(per_run_data, per_run_task_designs):
        y_dev = y_r.to(device).float()              # (n_voxels, n_tp_r)
        X_task_dev = X_task_r.to(device).float()    # (n_tp_r, n_task)
        n_tp_r = X_task_dev.shape[0]

        if polort >= 0:
            Z_r = torch.from_numpy(
                legendre_polynomials(n_tp_r, polort)
            ).to(device=device, dtype=torch.float32)
            X_full = torch.cat([X_task_dev, Z_r], dim=1)
        else:
            X_full = X_task_dev

        # OLS via lstsq on (X_full, y_dev.T) — (n_tp_r, n_voxels)
        # Use cholesky_solve for speed since X_full has many fewer
        # columns than timepoints and is well-conditioned post-polys.
        XtX = X_full.T @ X_full
        Xty = X_full.T @ y_dev.T
        try:
            L0 = torch.linalg.cholesky(XtX)
            beta = torch.cholesky_solve(Xty, L0)
        except torch.linalg.LinAlgError:
            beta = torch.linalg.lstsq(X_full, y_dev.T).solution

        resid = y_dev - (X_full @ beta).T            # (n_voxels, n_tp_r)
        residual_mean_per_run.append(resid.mean(dim=0).detach().cpu().numpy())
        del y_dev, X_task_dev, X_full, XtX, Xty, beta, resid

    residual_concat = np.concatenate(residual_mean_per_run)
    run_starts: list[int] = [0]
    for res_run in residual_mean_per_run[:-1]:
        run_starts.append(run_starts[-1] + res_run.size)

    # ── 3. REML grid search on the mean-residual timeseries. ─────────
    # Use an empty design (just a constant) — we're estimating noise
    # covariance after the task+poly fit has already been removed.
    n_total = residual_concat.size
    X_const = torch.ones(n_total, 1, device=device, dtype=torch.float32)
    Y_resid = torch.from_numpy(residual_concat).to(device=device, dtype=torch.float32)

    a_opt, b_opt, _ = reml_grid_search(
        X=X_const, Y=Y_resid, run_starts=run_starts, device=device,
    )
    if verbose:
        print(f"  prewhiten: global ARMA(1,1) → a={a_opt:.3f}, b={b_opt:.3f}")

    # ── 4. Per-run Cholesky + triangular solve to whiten. ────────────
    per_run_data_white: list[torch.Tensor] = []
    per_run_design_white: list[torch.Tensor] = []
    for r in range(n_runs):
        n_tp_r = per_run_task_designs[r].shape[0]
        R_r = build_arma11_covariance(
            a_opt, b_opt, n_tp_r, torch.device("cpu"),
            dtype=torch.float32, run_starts=None,
        )
        if R_r is None:
            raise ValueError(
                f"build_arma11_covariance returned None for run {r} "
                f"with a={a_opt}, b={b_opt}"
            )
        L_r = torch.linalg.cholesky(R_r).to(device)

        # Whiten design: L⁻¹ X (apply along time axis = dim 0).
        X_r = per_run_task_designs[r].to(device).float()
        X_r_white = torch.linalg.solve_triangular(L_r, X_r, upper=False)

        # Whiten data: per voxel, treat as (n_tp_r, n_voxels) = data.T.
        y_r = per_run_data[r].to(device).float()
        y_r_white = torch.linalg.solve_triangular(L_r, y_r.T, upper=False).T

        per_run_data_white.append(y_r_white)
        per_run_design_white.append(X_r_white)
        del R_r, L_r, X_r, X_r_white, y_r, y_r_white

    return per_run_data_white, per_run_design_white, float(a_opt), float(b_opt)


def fit_flobs_constrained(
    data: np.ndarray | torch.Tensor,
    design_task: np.ndarray | torch.Tensor,
    basis: FLOBSBasis,
    n_conditions: int,
    *,
    nuisance: np.ndarray | torch.Tensor | None = None,
    prior_weight: float | str = "auto",
    device: torch.device | None = None,
) -> FLOBSFitResult:
    """Deprecated alias for :func:`fit_basis_constrained_ridge`.

    Kept for backwards compatibility — extracts (m, C, basis) from a
    :class:`FLOBSBasis` and forwards to the renamed primitive.  New
    callers should use :func:`fit_basis_constrained_ridge` directly so
    they can pass arbitrary priors (e.g. SPMG2/3, plain ridge).
    """
    return fit_basis_constrained_ridge(
        data=data,
        design_task=design_task,
        basis_functions=basis.basis_functions,
        prior_mean=basis.m,
        prior_cov=basis.C,
        n_blocks=n_conditions,
        nuisance=nuisance,
        prior_weight=prior_weight,
        device=device,
    )


# ----------------------------------------------------------------------------
# Prior constructors — return (m, C) for use with fit_basis_constrained_ridge.
# ----------------------------------------------------------------------------


def flobs_prior(basis: FLOBSBasis) -> tuple[np.ndarray, np.ndarray]:
    """Return the (m, C) MVN prior derived from a FLOBS basis.

    Trivial extractor — exists for symmetry with :func:`spmg_prior`
    and :func:`ridge_prior` so the caller pattern is uniform:

    .. code-block:: python

        basis = generate_flobs_basis(n_basis=3)
        prior_m, prior_C = flobs_prior(basis)
        fit = fit_basis_constrained_ridge(
            data, design, basis.basis_functions,
            prior_m, prior_C, n_blocks=n_conditions, ...
        )
    """
    return basis.m.copy(), basis.C.copy()


def spmg_prior(
    canonical_std: float = 5.0,
    derivative_std: float = 0.5,
    dispersion_std: float | None = None,
    canonical_mean: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Hand-specified MVN prior for SPMG1 / SPMG2 / SPMG3 basis fits.

    Returns a **diagonal** (m, C) suitable for use with
    :func:`fit_basis_constrained_ridge`.  The diagonal structure
    encodes the physical intuition that:

    - The canonical-amplitude coefficient should be largely
      unconstrained (large ``canonical_std`` → weak prior).
    - The temporal-derivative coefficient should be much smaller in
      magnitude than the amplitude (small ``derivative_std`` → tight
      prior toward zero).
    - The dispersion-derivative coefficient (SPMG3) likewise tight.

    With ``canonical_mean=0`` and zero means for the derivatives, the
    prior expresses "you can have any amplitude in either direction,
    but derivative coefficients should be small relative to it" — the
    typical anti-overfitting structure for SPMG2/SPMG3 single-trial
    fits at short ISIs.

    Parameters
    ----------
    canonical_std : float, default 5.0
        Standard deviation on the canonical (first) basis coefficient.
        Large by default — we don't want to shrink amplitude.
    derivative_std : float, default 0.5
        Standard deviation on the temporal-derivative coefficient.
        TR04MW2-style choice would be ~10 % of canonical_std; the
        default produces SPMG2/3 fits without insane delay shifts.
    dispersion_std : float, optional
        If given, returns a 3-element prior for SPMG3.  Else 2-element
        SPMG2.  Pass ``None`` (default) for SPMG2; pass a float for
        SPMG3.
    canonical_mean : float, default 0.0
        Mean on the canonical coefficient.  Use 0 to leave amplitude
        sign-free; use a positive value to bias the fit toward
        positive responses (e.g., when fitting only an "active"
        contrast).

    Returns
    -------
    m : np.ndarray, shape (K,)
    C : np.ndarray, shape (K, K)
        Diagonal covariance ``diag(canonical_std², derivative_std², …)``.
    """
    if dispersion_std is None:
        m = np.array([canonical_mean, 0.0], dtype=np.float64)
        C = np.diag([canonical_std**2, derivative_std**2]).astype(np.float64)
    else:
        m = np.array([canonical_mean, 0.0, 0.0], dtype=np.float64)
        C = np.diag(
            [canonical_std**2, derivative_std**2, dispersion_std**2]
        ).astype(np.float64)
    return m, C


def decouple_amplitude_prior(
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a full MVN(m, C) prior into the **amplitude-decoupled** form.

    TR04MW2 §2.4 reparameterises ``β = D · ŝ`` — amplitude scalar
    times unit-norm shape vector — with the prior on ``ŝ`` only.
    Amplitude is never shrunk; only the shape direction is
    constrained.  This directly fixes the "amplitudes too small"
    pathology that CV exposed on high-SNR voxels (TR04MW2 §3 fig 4
    discusses the same trade-off).

    Implementation as a closed-form generalised ridge:

    1. Pick the amplitude direction ``u = m / ||m||``.  This is the
       direction in K-D coefficient space along which the prior
       *expects* an HRF to live.
    2. Build a Householder-style orthonormal rotation ``R`` whose
       first column is ``u``.  In this rotated frame, the first
       coordinate is "amplitude", the remaining K-1 coordinates are
       "shape".
    3. The rotated covariance is ``C_r = Rᵀ C R``; the shape sub-
       covariance is ``C_r[1:, 1:]``.  Set the amplitude row/column
       to **zero precision** (no penalty in that direction), leaving
       only the shape precision ``inv(C_r[1:, 1:])``.
    4. Rotate the resulting precision back to the original frame:
       ``P_decoupled = R · diag(0, inv(C_r[1:, 1:])) · Rᵀ``.

    The decoupled "mean" is zero — since amplitude is unconstrained
    we don't pull toward any specific magnitude; the *direction* of
    the prior was encoded by the rotation.

    Pass the returned ``(m_decoupled, C_decoupled)`` directly to
    :func:`fit_basis_constrained_ridge` and the rest of the solver
    works unchanged.

    Parameters
    ----------
    prior_mean : np.ndarray, shape (K,)
        Original prior mean.  Must be non-zero — there's no
        amplitude direction to decouple from a zero mean.
    prior_cov : np.ndarray, shape (K, K)
        Original prior covariance.  Symmetric positive-definite.

    Returns
    -------
    m_decoupled : np.ndarray, shape (K,)
        Zero vector (the prior is centred at the origin in the
        rotated frame).
    C_decoupled : np.ndarray, shape (K, K)
        Pseudo-covariance.  Inverting gives the rank-(K-1) precision
        matrix that's zero along the amplitude direction.  We return
        a covariance form so the existing solver can call
        ``np.linalg.inv(prior_cov)`` and get the right precision —
        we add a tiny regularizer along the amplitude direction so
        the inversion is numerically stable but the effective
        precision in that direction is ~zero.
    """
    K = prior_mean.shape[0]
    if K == 1:
        # Degenerate: no shape direction to decouple from.  Return a
        # very weak prior (effectively unconstrained amplitude).
        return np.zeros(1), np.array([[1e12]])
    norm_m = float(np.linalg.norm(prior_mean))
    if norm_m < 1e-12:
        raise ValueError(
            "decouple_amplitude_prior: prior_mean is ~zero — no amplitude "
            "direction to decouple.  Use the plain prior or set a non-zero "
            "mean (e.g. spmg_prior(canonical_mean=2.0))."
        )
    u = prior_mean / norm_m

    # Build R whose first column is u (Gram–Schmidt of [u | I]).
    R = np.zeros((K, K), dtype=np.float64)
    R[:, 0] = u
    fill_idx = 1
    for k in range(K):
        if fill_idx == K:
            break
        e_k = np.zeros(K); e_k[k] = 1.0
        # Project off all previously-collected R columns.
        v = e_k - R[:, :fill_idx] @ (R[:, :fill_idx].T @ e_k)
        nv = np.linalg.norm(v)
        if nv > 1e-10:
            R[:, fill_idx] = v / nv
            fill_idx += 1
    # Numerical safety: re-orthonormalise via QR.
    Q, _ = np.linalg.qr(R)
    # Ensure first column is still u (QR may flip signs).
    sign = np.sign(Q[:, 0] @ u)
    if sign == 0:
        sign = 1.0
    Q[:, 0] = sign * Q[:, 0]
    R = Q

    # Shape block in rotated frame.
    C_r = R.T @ prior_cov @ R
    C_shape = C_r[1:, 1:]
    P_shape = np.linalg.inv(C_shape)                   # (K-1, K-1)

    # Build rotated precision: amplitude precision = 0, shape = P_shape.
    P_r = np.zeros((K, K), dtype=np.float64)
    P_r[1:, 1:] = P_shape

    # Rotate precision back: P_decoupled = R · P_r · Rᵀ.
    P_decoupled = R @ P_r @ R.T

    # Return as a covariance for API uniformity.  Invert with a tiny
    # ridge along the amplitude direction so the matrix is stably
    # invertible (the solver inverts C internally to get P).  The
    # effective precision in the amplitude direction is then
    # 1/(1e12) = ~0, exactly what we want.
    amp_proj = np.outer(u, u)                          # rank-1 amplitude
    C_decoupled = P_decoupled + 1e-12 * amp_proj       # ensure full rank
    C_decoupled = np.linalg.inv(C_decoupled)
    # Symmetrize.
    C_decoupled = 0.5 * (C_decoupled + C_decoupled.T)

    m_decoupled = np.zeros(K, dtype=np.float64)
    return m_decoupled, C_decoupled


def ridge_prior(
    n_basis: int,
    coefficient_std: float = 1.0,
    mean: float | np.ndarray = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Plain isotropic ridge: prior ``β ~ N(mean, coefficient_std² · I)``.

    The simplest possible prior — shrink each coefficient toward
    ``mean`` by the same amount.  This is the "generic ridge" that
    most reviewers will recognise, but it has a known weakness for
    HRF basis sets: it shrinks the canonical-amplitude coefficient the
    same as the derivative coefficient, so all responses get tamped
    down.  Use :func:`spmg_prior` or :func:`flobs_prior` instead when
    the basis is HRF-like.  Provided here for completeness and as a
    baseline that ``-reg ridge`` can dispatch to.

    Parameters
    ----------
    n_basis : int
        Dimensionality of the prior.
    coefficient_std : float, default 1.0
        Per-coefficient standard deviation.  The Bayesian-optimal
        weight in :func:`fit_basis_constrained_ridge` (``λ = σ²``)
        means this just sets the *relative* strength of shrinkage.
    mean : float or array, default 0.0
        Scalar (broadcast to all coefficients) or per-coefficient
        vector.

    Returns
    -------
    m : np.ndarray, shape (n_basis,)
    C : np.ndarray, shape (n_basis, n_basis)
        Isotropic ``coefficient_std² · I``.
    """
    if np.isscalar(mean):
        m = np.full(n_basis, float(mean), dtype=np.float64)
    else:
        m = np.asarray(mean, dtype=np.float64)
        if m.shape != (n_basis,):
            raise ValueError(f"mean shape {m.shape} ≠ (n_basis={n_basis},)")
    C = np.eye(n_basis, dtype=np.float64) * (coefficient_std**2)
    return m, C
