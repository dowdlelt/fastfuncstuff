# MELODIC_NOTES

## Why this exists

This note records the practical, non-obvious choices FSL MELODIC makes for model-order selection and ICA scaling, especially where behavior differs from a "plain" PCA/ICA workflow. The goal is parity first, experimentation second.

Melodic code location: https://git.fmrib.ox.ac.uk/fsl/melodic

---

## High-level takeaway

MELODIC is **not** just "run PCA, then run ICA, choose k by Minka".
It is a tightly coupled stack of preprocessing, eigenspectrum shaping, heuristic clamps, and first-peak criteria logic.

Small implementation differences can shift model order substantially.

---

## MELODIC path vs "normal" expectation

### 1) Voxel variance normalization

- **Common expectation**:
  - divide each voxel time series by its own temporal standard deviation.
- **MELODIC behavior**:
  - uses a **residual-noise-based varnorm** path:
    1. PCA on demeaned data,
    2. build white/dewhite transforms,
    3. threshold small coefficients in whitened space (`vn_level`, default 2.3),
    4. reconstruct denoised signal,
    5. estimate residual voxel noise std,
    6. divide each voxel by residual std,
    7. clamp/zero constant voxels (`std < 0.01`).
- **Why it matters**:
  - changes eigenspectrum shape before PPCA; this alone can move selected k.

### 2) Effective sample size for PPCA

- **Common expectation**:
  - use voxel count directly as sample size.
- **MELODIC behavior**:
  - computes smoothness `resels = FWHM_x * FWHM_y * FWHM_z` (in voxel units), then
  - `N_eff = floor(n_vox / (2.5 * resels))`.
- **Important**:
  - using mm units does not change the formula if units are converted consistently (volume factors cancel).

### 3) PCA dimensionality basis

- **Common expectation**:
  - use a reduced rank (often `T-1`) after demeaning.
- **MELODIC behavior**:
  - runs PPCA path from full temporal covariance eigen-structure, then applies custom truncation in `adj_eigspec`.
- **Practical consequence**:
  - off-by-one in available eigenvalues can propagate into PPCA row count and peak location.

### 4) Eigenspectrum adjustment (`adj_eigspec`)

- **Common expectation**:
  - run PPCA directly on raw eigenvalues.
- **MELODIC behavior**:
  - reverse/drop handling (effectively removing 2 smallest modes in this stage),
  - divide by expected Marchenko-Pastur spectrum (`Feta`),
  - sort adjusted eigenvalues,
  - choose truncation at first crossing of 98% cumulative variance of original spectrum,
  - if too small, fallback to half-spectrum.
- **Extra quirks**:
  - hard floors such as `CircleLaw >= 5e-9`.

### 5) PPCA criteria are multi-track

- **Common expectation**:
  - use one criterion (e.g., max Laplace evidence).
- **MELODIC behavior**:
  - computes 5 criteria curves: lap, bic, mdl, rrn, aic,
  - normalizes each to [0, 1],
  - picks first local rise-stop (first peak) for each,
  - applies `aut` logic: prefer bic when `bic < lap` and cumulative variance at bic index > 0.8, else lap.

### 6) First-peak selection, not global max

- **Common expectation**:
  - choose argmax of criterion curve.
- **MELODIC behavior**:
  - starts at low k and walks upward while curve increases; stops at first peak.
- **Why it matters**:
  - avoids late-k numerical artifacts but is sensitive to tiny curve-shape shifts.

### 7) Additional PPCA internals/clamps

MELODIC `ppca_est` includes hard-coded behaviors that materially affect curves:

- `tmp1(1) = 0.95 * tmp1(2)` tail handling,
- floor clamps: `tmp1/tmp3/tmp4 >= 0.01`,
- non-positive Hessian-related terms forced to 1 before log,
- multiple cumulative-sum transforms.

These are not "optional niceties"; they are part of the selection behavior.

---

## `melodic_PPCA` file column interpretation (practical)

For standard MELODIC outputs, `melodic_PPCA` commonly has 7 columns:

1. selected criterion curve (after `ppca_select`, depends on estimator mode),
2. normalized adjusted-eigenvalue curve,
3. lap,
4. bic,
5. mdl,
6. rrn,
7. aic.

When using `aut`, column 1 may reflect lap or bic depending on the rule above.

---

## What this means for parity work

If model order differs, check in this order:

1. varnorm parity (residual-based vs simple std scaling),
2. smoothness/resels and `N_eff`,
3. eigenvalue count entering `adj_eigspec`,
4. MP/Feta adjustment and truncation crossing index,
5. first-peak implementation details (`<` vs `<=`, max-k ceiling),
6. criterion normalization and `aut` rule.

---

## Current project status

- A MELODIC-specific residual varnorm path is implemented for melodic/auto selection mode.
- PPCA trace dumping is available via `-ppca_debug_dump` for run-by-run forensic comparison.
- On the reference dataset used during debugging, this change moved selection to the MELODIC-matching component count.

---

## Suggested future hardening

- Add a dedicated regression test fixture comparing trace checkpoints against known MELODIC outputs (within tolerances).
- Keep melodic-path logic isolated from non-melodic paths to avoid accidental behavioral drift.
- Preserve trace JSON schema stability so cross-version debugging stays easy.
