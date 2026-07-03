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

5. **Manifold tracing** — :func:`trace_manifold_auto` (default) or
   :func:`trace_manifold_grid` (fallback) or
   :func:`trace_manifold_from_points` (override).  For K=3 the unit
   vectors live on the 2-sphere; we trace a 1-D path that follows the
   density ridge of the histogram in spherical coordinates.  ``n_points``
   evenly spaced (in angle) samples along this path give the HRF
   "library" — same shape diversity as NSD's 20-HRF canonical library.

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
- **K=3 by default.**  Lower K leaves you with a 1-D arc that's barely
  a "manifold"; higher K loses the planar density-tracing trick.  K=3 is
  what NSD used and is the well-tested path; other values are allowed
  but auto-manifold tracing falls back to PCA-based 1-D ordering.

Duration / deconvolution caveat (TODO)
--------------------------------------
The FIR/TENT fit estimates the **response to whatever event shape was in
the data** — including the stimulus duration's boxcar.  Downstream tools
(``ffs_hrfopt``, ``ffs_denoise``, etc.) re-convolve this library HRF with
the event boxcar at modelling time.  Strictly correct usage therefore
requires that the library represent the *impulse response* — i.e. the
HRF you'd see for a duration-0 stimulus — so that the downstream
convolution recovers the actual measured response.

For brief events (duration << HRF width) the FIR estimate ≈ impulse
response, so the difference is negligible.  For block designs (e.g. 10 s
blocks) the FIR estimate is already wider than the true HRF and using it
directly as a "library HRF" will produce a *doubly-convolved* design.
Two conditions with the same underlying HRF but different durations will
also yield different-looking library entries, which a per-condition
deconvolution step would harmonize.

For the MVP we punt this to a later "group harmonization" stage and
return the FIR estimate as-is, but mark the deconvolution slot in the
pipeline with :func:`deconvolve_event_duration` (raises
``NotImplementedError`` today) so callers can see where the missing step
lives.  The library sidecar JSON should record the per-group event
durations so a future stage can apply the deconvolution after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.interpolate import PchipInterpolator
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
        NSD choice; 2 collapses the manifold to a 1-D arc, 4+ loses the
        planar-density-tracing trick used by :func:`trace_manifold_auto`.
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


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Generate ``n`` near-uniformly spaced points on the unit 2-sphere.

    Uses the golden-angle spiral.  Returns shape ``(n, 3)``.
    """
    indices = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1 - 2 * indices / n)  # polar angle in [0, π]
    theta = np.pi * (1 + 5**0.5) * indices  # golden-angle azimuth
    return np.column_stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])


def trace_manifold_auto(
    unit_vectors: np.ndarray,
    n_points: int = 20,
    angular_step_deg: float = 6.0,
    bandwidth_deg: float = 8.0,
    n_grid: int = 4096,
    density_floor_frac: float = 0.05,
) -> np.ndarray:
    """Trace a 1-D density ridge across the unit sphere (K=3 only).

    Greedy walk:

    1. Build a near-uniform Fibonacci grid of ``n_grid`` candidate
       directions on the 2-sphere.
    2. Compute the spherical-KDE density of ``unit_vectors`` at each
       grid point (bandwidth = ``bandwidth_deg``).
    3. Start at the global density peak.
    4. From the current point, propose moves to grid points whose
       angular distance from the current point is within ±20 % of
       ``angular_step_deg``.  Pick the one with highest density that is
       also at least ``0.7 × angular_step`` away from every previously
       visited point (prevents backtracking).
    5. Stop early if density drops below
       ``density_floor_frac × peak_density`` or no valid forward move
       exists.

    This is a deliberately simple heuristic — it is **not** principal
    curves or graph-based ridge extraction.  For NSD-like data the
    density manifold is a clean 1-D arc, and a greedy walk on a
    Fibonacci grid recovers it; for pathological cases use
    :func:`trace_manifold_from_points` to override.

    Parameters
    ----------
    unit_vectors : np.ndarray, shape (n_voxels, 3)
        Unit-norm voxel directions from :func:`project_unit_sphere`.
        Pass only ``K == 3``; for other K use :func:`trace_manifold_grid`
        or supply explicit points.
    n_points : int, default 20
        Maximum number of manifold samples to emit.
    angular_step_deg : float, default 6.0
        Target angular spacing between successive samples, matching
        NSD's 6° parameterization.
    bandwidth_deg : float, default 8.0
        Width of the spherical KDE kernel.  Larger = smoother density
        but blurs across the manifold.
    n_grid : int, default 4096
        Number of Fibonacci-sphere candidate directions.  Higher gives
        a smoother walk at O(n_grid · n_data) cost; 4096 has been
        adequate in testing.
    density_floor_frac : float, default 0.05
        Stop walking when density drops below this fraction of the
        starting peak.

    Returns
    -------
    manifold : np.ndarray, shape (n_actual, 3)
        Ordered manifold points on the unit sphere.
        ``n_actual <= n_points`` (early termination is allowed).

    Raises
    ------
    ValueError
        If ``unit_vectors.shape[1] != 3``.
    """
    if unit_vectors.shape[1] != 3:
        raise ValueError(
            "trace_manifold_auto requires K=3 (got "
            f"K={unit_vectors.shape[1]}); use trace_manifold_grid or supply points."
        )

    # Drop zero rows (filtered-out voxels) so they don't tug the density to 0.
    nz_mask = np.linalg.norm(unit_vectors, axis=1) > 0.5
    data = unit_vectors[nz_mask]
    if data.size == 0:
        raise ValueError("No non-zero unit vectors to trace.")

    grid = _fibonacci_sphere(n_grid)  # (n_grid, 3)
    bw = np.deg2rad(bandwidth_deg)
    density = _spherical_kde(grid, data, bw)  # (n_grid,)

    # Cosines for angular-distance comparisons.
    step_rad = np.deg2rad(angular_step_deg)
    step_lo = np.cos(step_rad * 1.2)  # widest accepted distance
    step_hi = np.cos(step_rad * 0.8)  # narrowest accepted distance
    back_cos = np.cos(step_rad * 0.7)  # min distance from prior points

    peak_idx = int(density.argmax())
    peak_dens = float(density[peak_idx])
    floor = peak_dens * density_floor_frac

    visited_idx = [peak_idx]
    visited = [grid[peak_idx]]

    while len(visited) < n_points:
        cur = visited[-1]
        cos_to_cur = grid @ cur
        # Forward annulus: grid points at angular distance ≈ step_rad from cur.
        annulus = (cos_to_cur <= step_hi) & (cos_to_cur >= step_lo)
        if not annulus.any():
            break

        # Forbid points too close to *any* prior visited point (no backtrack).
        visited_arr = np.stack(visited)  # (k, 3)
        cos_to_prior = grid @ visited_arr.T  # (n_grid, k)
        not_too_close = (cos_to_prior <= back_cos).all(axis=1)

        valid = annulus & not_too_close
        if not valid.any():
            break

        # Choose the highest-density forward candidate.
        cand_density = np.where(valid, density, -np.inf)
        nxt_idx = int(cand_density.argmax())
        if density[nxt_idx] < floor:
            break

        visited_idx.append(nxt_idx)
        visited.append(grid[nxt_idx])

    # Try to extend backward from the original peak too — gives a symmetric
    # ridge instead of a one-sided walk.  Same logic in reverse direction.
    while len(visited) < n_points:
        cur = visited[0]
        cos_to_cur = grid @ cur
        annulus = (cos_to_cur <= step_hi) & (cos_to_cur >= step_lo)
        if not annulus.any():
            break
        visited_arr = np.stack(visited)
        cos_to_prior = grid @ visited_arr.T
        not_too_close = (cos_to_prior <= back_cos).all(axis=1)
        valid = annulus & not_too_close
        if not valid.any():
            break
        cand_density = np.where(valid, density, -np.inf)
        prev_idx = int(cand_density.argmax())
        if density[prev_idx] < floor:
            break
        visited.insert(0, grid[prev_idx])
        visited_idx.insert(0, prev_idx)

    return np.stack(visited)


def trace_manifold_grid(unit_vectors: np.ndarray, n_points: int = 20) -> np.ndarray:
    """Fallback: order voxels along the first principal axis, sample evenly.

    For K != 3 or when the user wants a deterministic non-density-aware
    sampler.  Projects ``unit_vectors`` onto their first PCA direction,
    sorts by that coordinate, and picks ``n_points`` evenly spaced
    percentiles.  The picked vectors are re-normalized to unit length.

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
            abs_peak = float(np.max(np.abs(h_imp)))
            if abs_peak > 0:
                # Flip sign so the largest *signed* extremum is positive,
                # then divide by the new max to put the peak at +1.
                signed_peak = float(h_imp[int(np.argmax(np.abs(h_imp)))])
                if signed_peak < 0:
                    h_imp = -h_imp
                h_imp = h_imp / float(np.max(h_imp))
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
) -> tuple[np.ndarray, np.ndarray]:
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
        If ``"peak"``, divide each reconstructed waveform by its peak
        absolute value so the maximum is +1.  If the raw peak is
        negative the *sign* is flipped first (the library convention is
        positive peaks).  This is exactly what
        ``getcanonicalhrflibrary.tsv`` stores.

    Returns
    -------
    waveforms : np.ndarray, shape (n_points, n_target)
        Reconstructed HRFs at ``target_dt`` resolution.
    target_times : np.ndarray, shape (n_target,)
        The time grid used (in seconds).

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
            # Match the canonical-library convention: positive peak = 1.
            abs_peak = float(np.max(np.abs(fine)))
            if abs_peak > 0:
                # Flip sign so the largest *signed* extremum is positive.
                signed_peak = float(fine[np.argmax(np.abs(fine))])
                if signed_peak < 0:
                    fine = -fine
                fine = fine / float(np.max(fine))
        out[i] = fine

    return out, target_times


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
        Override the default parameter bounds.  Defaults clamp time-to-
        peak in [2, 12] s, dispersions in [0.3, 5], and undershoot
        ratio in [0, 1].
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
    # Provenance for the future deconvolution step (see module docstring).
    # event_durations is the per-group event-duration metadata (seconds);
    # duration_convolved=True means the library entries are the duration-
    # convolved response, NOT the impulse response.  A future
    # ``deconvolve_event_duration`` pass would flip this to False.
    event_durations: np.ndarray | None = None
    duration_convolved: bool = True
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
    manifold_mode: Literal["auto", "grid", "points"] = "auto",
    manifold_points: np.ndarray | None = None,
    fit_gamma: bool = True,
    seed: int = 42,
    event_durations: np.ndarray | None = None,
    refit_weights: np.ndarray | None = None,
    deconvolve_duration: float | None = None,
    deconv_snr: float = 100.0,
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
    manifold_mode : {"auto", "grid", "points"}
        Selects the manifold-tracing strategy.  ``"points"`` requires
        ``manifold_points`` to be passed.
    manifold_points : np.ndarray, optional
        Required for ``manifold_mode="points"``.  Shape (n_hrfs, n_pcs);
        re-normalized to unit length by
        :func:`trace_manifold_from_points`.
    fit_gamma : bool, default True
        If True, also compute parametric double-gamma fits of every
        reconstructed waveform.  Output ``fitted`` is filled; otherwise
        it is ``None`` (caller saves only ``raw``).
    deconvolve_duration : float, optional
        If given (and ``> target_dt``), apply Wiener duration
        deconvolution (see :func:`deconvolve_event_duration`) to the
        reconstructed library entries so the output represents the
        *impulse response* HRF rather than the duration-convolved
        response.  This is the right thing to do for any non-impulse
        event duration: downstream consumers re-convolve with the
        event boxcar at modelling time, so without this step the
        library produces a doubly-convolved design.  Both the raw
        cubic recon and (if ``fit_gamma=True``) the gamma fit are
        deconvolved; the pre-deconv versions are also kept in
        ``LibraryResult`` for QC inspection.
    deconv_snr : float, default 100.0
        Wiener filter SNR.  See :func:`deconvolve_event_duration`.
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
    sel = select_voxels(r2, threshold=r2_threshold, max_voxels=max_voxels, seed=seed)

    # Air / background voxels are a treacherous failure mode here.  When
    # no brain mask is supplied, voxels outside the head pass through
    # the loader as constant (post-scaling) zeros.  The GLM's polynomial
    # detrending fits a constant perfectly → SS_residual = 0 → R² = 1,
    # so these voxels look like the "best" voxels by the R² gate and
    # end up dominating the SVD with their *all-zero* FIR betas.  The
    # PCs then come out as pure noise.  NSD avoided this with a BET
    # brain mask up front; we filter post-hoc: require the FIR beta
    # vector to have non-negligible L2 norm.  This is also robust to
    # any other "constant signal, high R²" failure mode.
    selected_betas = betas[sel]
    fir_norms = np.linalg.norm(selected_betas, axis=1)
    norm_floor = 1e-8 * max(
        float(np.median(fir_norms[fir_norms > 0])) if (fir_norms > 0).any() else 1.0, 1.0
    )
    alive = fir_norms > norm_floor
    n_dropped = int((~alive).sum())
    if n_dropped > 0:
        sel = sel[alive]
        selected_betas = selected_betas[alive]
    if sel.size < n_hrfs:
        raise ValueError(
            f"Only {sel.size} usable voxels survived selection "
            f"(R² > {r2_threshold}, FIR-betas non-trivial); need at least "
            f"n_hrfs={n_hrfs}.  Likely cause: no brain mask supplied and air "
            f"voxels dominated the R² selection ({n_dropped} dropped as zero-norm). "
            f"Re-run with -mask <brain_mask.nii.gz> or lower -r2-threshold."
        )
    if n_dropped > 0.5 * (sel.size + n_dropped):
        # Massive air-voxel contamination — the FIR fit may not be
        # capturing real signal anywhere either.  Tell the user.
        import warnings

        warnings.warn(
            f"derive_library: {n_dropped} of {sel.size + n_dropped} selected "
            f"voxels had zero-norm FIR betas (constant/air signal with R²=1). "
            f"Strongly recommend passing -mask <brain_mask.nii.gz>.",
            RuntimeWarning,
            stacklevel=2,
        )

    # QC: pooled task HRF — mean of FIR betas across (post-filter) voxels.
    # This is the "average HRF in active cortex" sanity-check the user
    # can plot to confirm the FIR fit picked up something HRF-shaped
    # before any of the SVD machinery runs.
    mean_fir = selected_betas.mean(axis=0)

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
    # "unit-circle heatmap" used to visually pick the density ridge.
    sphere_hist2d = None
    sphere_hist_edges = None
    if n_pcs == 3:
        # Match NSD's bin grid (loadings ∈ [-1.5, 1.5], step 0.02).
        edges = np.arange(-1.5, 1.5 + 0.02, 0.02)
        sphere_hist2d, _, _ = np.histogram2d(unit[:, 1], unit[:, 2], bins=(edges, edges))
        sphere_hist_edges = edges

    if manifold_mode == "auto":
        if n_pcs != 3:
            # Auto requires sphere; degrade gracefully.
            manifold = trace_manifold_grid(unit, n_points=n_hrfs)
        else:
            manifold = trace_manifold_auto(
                unit,
                n_points=n_hrfs,
                angular_step_deg=angular_step_deg,
                bandwidth_deg=bandwidth_deg,
            )
    elif manifold_mode == "grid":
        manifold = trace_manifold_grid(unit, n_points=n_hrfs)
    elif manifold_mode == "points":
        if manifold_points is None:
            raise ValueError("manifold_mode='points' requires manifold_points")
        manifold = trace_manifold_from_points(manifold_points)
    else:
        raise ValueError(f"Unknown manifold_mode: {manifold_mode}")

    raw, target_times = reconstruct_timecourses(
        manifold,
        svd.pcs,
        lag_times,
        target_dt=target_dt,
        target_duration=target_duration,
        normalize="peak",
    )

    fitted = None
    gamma_params: list[dict] = []
    if fit_gamma:
        fitted = np.zeros_like(raw)
        for i in range(raw.shape[0]):
            fit, p = fit_double_gamma(raw[i], dt=target_dt)
            fitted[i] = fit
            gamma_params.append(p)

    # Duration deconvolution — recover the impulse response from the
    # duration-convolved library so downstream consumers don't double-
    # convolve.  We apply to BOTH the raw cubic recon and the gamma
    # fit (if computed); the pre-deconv versions are kept on the
    # result for diagnostics.
    raw_deconvolved: np.ndarray | None = None
    fitted_deconvolved: np.ndarray | None = None
    gamma_params_deconvolved: list[dict] = []
    duration_convolved_flag = True
    if deconvolve_duration is not None and deconvolve_duration > target_dt:
        raw_deconvolved = deconvolve_event_duration(
            raw,
            duration=float(deconvolve_duration),
            dt=target_dt,
            snr=deconv_snr,
            normalize_peak=True,
        )
        if fit_gamma:
            # Re-fit the gamma family AGAINST the deconvolved curve, so
            # the parametric library is in impulse-response space (not
            # a gamma-fit of the convolved curve).
            fitted_deconvolved = np.zeros_like(raw_deconvolved)
            for i in range(raw_deconvolved.shape[0]):
                fit, p = fit_double_gamma(raw_deconvolved[i], dt=target_dt)
                fitted_deconvolved[i] = fit
                gamma_params_deconvolved.append(p)
        duration_convolved_flag = False

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
        event_durations=(
            np.asarray(event_durations, dtype=float) if event_durations is not None else None
        ),
        duration_convolved=duration_convolved_flag,
        mean_fir_hrf=mean_fir,
        unit_sphere_points=unit,
        sphere_hist2d=sphere_hist2d,
        sphere_hist_edges=sphere_hist_edges,
    )
