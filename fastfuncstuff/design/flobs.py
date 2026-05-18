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
    device: torch.device | None = None,
    reconstruct_hrfs: bool = True,
    chunk_size: int | None = None,
    lambda_mode: str = "global",
    lambda_n_bins: int = 20,
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
    y = (
        torch.as_tensor(data, dtype=torch.float64)
        if not isinstance(data, torch.Tensor)
        else data.to(dtype=torch.float64)
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
    for start in range(0, n_voxels, chunk_size):
        end = min(start + chunk_size, n_voxels)
        y_chunk = y[start:end].to(device, non_blocking=True)      # (chunk, n_t)
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

    Pm_unit = P @ m_bar                                            # (n_cols,)

    if lambda_mode == "global":
        # One A, one factor.
        A = XtX + effective_weight * P
        Pm = effective_weight * Pm_unit
        try:
            L_A = torch.linalg.cholesky(A)
            use_chol_A = True
        except torch.linalg.LinAlgError:
            L_A = None
            use_chol_A = False

        for start in range(0, n_voxels, chunk_size):
            end = min(start + chunk_size, n_voxels)
            y_chunk = y[start:end].to(device, non_blocking=True)
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
        if isinstance(prior_weight, str):
            user_mult = 1.0
        else:
            user_mult = float(prior_weight)
        sigma2_per_voxel = (ss_res_ols_full / dof).clamp_min(1e-30)  # CPU tensor
        lambda_per_voxel_cpu = user_mult * sigma2_per_voxel             # (n_voxels,)

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

        for start in range(0, n_voxels, chunk_size):
            end = min(start + chunk_size, n_voxels)
            y_chunk = y[start:end].to(device, non_blocking=True)
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
    r2 = 1.0 - ss_res_constrained_full / torch.clamp(ss_tot_full, min=1e-30)
    r2_ols = 1.0 - ss_res_ols_full / torch.clamp(ss_tot_full, min=1e-30)

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
    for wi, w in enumerate(weights_to_run):
        ss_res_accum = torch.zeros(n_voxels, dtype=torch.float64, device=device)
        if verbose:
            label = "OLS" if w == "OLS" else f"weight×σ²={w}"
            print(f"  CV: {label} …")
        for train_runs, test_runs in splits:
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
