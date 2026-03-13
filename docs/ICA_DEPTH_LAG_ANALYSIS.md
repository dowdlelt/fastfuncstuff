# ICA Depth-Dependent Lag Analysis

## Overview
`bin/ffs_ica.py` now includes a post-ICA depth-lag analysis stage designed to identify components with depth-dependent temporal delays (BOLD-like vs non-BOLD-like behavior).

This stage runs after ICA decomposition, z-map generation, and component sorting.

## Problem Being Solved
A component can look spatially plausible but still be physiologically implausible. In cortical depth data, plausible BOLD components are expected to show structured lag behavior across depth bins.

The new pipeline estimates, for each component:
- depth-specific weighted timeseries,
- lag (seconds) and peak correlation (`r`) per depth vs a reference depth,
- Spearman rank correlation between depth index and lag.

## Inputs and Flags
Depth-lag analysis is controlled via these CLI flags:

- `-depth_mask <file>`
  - 3D integer-labeled depth map (e.g., 1..5, or more).
  - Must match run space and is intersected with the final run mask.
- `-depth_lag` / `-no_depth_lag`
  - Enable/disable depth-lag post-step.
- `-depth_lag_reference_depth <int>`
  - Reference depth label for lag comparisons (default: `3`).
- `-depth_lag_z_thresh <float>`
  - Component z-threshold used for weighting (default: `2.3`).
- `-depth_lag_min_voxels <int>`
  - Minimum weighted voxels required in each depth bin (default: `20`).
- `-depth_lag_max_lag_s <float>`
  - Max absolute lag search window in seconds (default: `6.0`).
- `-depth_lag_match_preproc` / `-no_depth_lag_match_preproc`
  - Apply same temporal preprocessing (polort/high-pass) to source timeseries before lag estimation.
- `-depth_lag_use_unsmoothed` / `-no_depth_lag_use_unsmoothed`
  - Prefer unsmoothed run data as source for depth timeseries (default: on).

## Core Logic
For each component `k`:

1. Build voxel weights from component z-map:
   - `w(v) = z_k(v)` if `z_k(v) > z_thresh`, else `0`.
2. For each depth label `d` in the depth mask:
   - select voxels in depth `d` ∩ final ICA mask,
   - compute weighted average timeseries using `w(v)`.
3. Compare each depth timeseries to the reference depth timeseries:
   - estimate lag (seconds) and peak `r`.
4. Compute Spearman correlation:
   - depth labels (`1..N` or provided integer labels) vs lag values,
   - report `rho`, `pvalue`, and number of usable depths.

## Lag Estimation Backend
`_best_lag_and_r` implements a two-stage backend:

1. Try Rapidtide (`rapidtide.correlate.quickcorr`) if available.
2. Fallback to normalized NumPy cross-correlation within a bounded lag window.

The chosen backend is recorded in metadata (`lag_method`).

## Output Structure
Depth-lag results are stored under:

- `spatial_guidance.depth_lag` in per-run metadata JSON

Includes:
- run-level settings (`reference_depth`, thresholds, limits),
- `component_results` list with per-component status and per-depth lag/r,
- `lag_seconds_matrix` (components × depth labels),
- `peak_r_matrix` (components × depth labels),
- backend identifier (`lag_method`).

Also writes:
- `*_depth_lag_seconds.png`
  - heatmap of lag (seconds) per component × depth.

## Current Scope
Implemented now:
- robust post-step computation,
- preprocessing-aware source handling,
- weighted depth timeseries extraction,
- lag + peak-r + Spearman scoring,
- metadata and figure outputs.

Deferred intentionally:
- hard classification policy (BOLD-like/non-BOLD-like) from lag features,
- confidence intervals/bootstrap on lag estimates,
- subject/run aggregation and group modeling,
- deeper Rapidtide API integration beyond opportunistic backend usage.

## Known Practical Considerations
- If the reference depth has insufficient weighted voxels, component status is `missing_reference_depth`.
- Sparse depth bins can reduce Spearman robustness (`n_depths < 3` yields null Spearman output).
- Very high z-threshold or restrictive masks may suppress usable depth timeseries.

## Suggested Next Steps
1. Add explicit component ranking fields from depth-lag metrics.
2. Add optional robust lag estimators (e.g., phase-based / coherence-informed).
3. Add QC plots per component:
   - depth timeseries overlays,
   - lag-vs-depth scatter with fitted trend.
4. Introduce a dedicated module in `fastfuncstuff/` for this logic (to remove CLI bloat and enable unit tests).
5. Add targeted tests for:
   - depth selector mapping,
   - weighted timeseries behavior,
   - lag estimator consistency (synthetic shifts),
   - Spearman output validity.
