# Outstanding Issues

---

## `ffs_moco` slower than single-core AFNI `3dvolreg`

### Symptom
On current benchmark datasets `ffs_moco` wall-clock time is consistently longer
than `3dvolreg` run on a single CPU core. The gap persists on GPU runs, not just
CPU — so parallelism isn't closing it.

### Likely contributors (hypotheses, not confirmed)
- **Per-volume Python / PyTorch overhead.** 3dvolreg is a tight C loop with
  hand-tuned shear-warp kernels (`thd_shear3d.c`). Our Gauss–Newton iteration in
  `fastfuncstuff/processing/ffs_moco.py` pays per-volume tensor-construction,
  autograd, and dispatch costs that dwarf the actual arithmetic on small
  EPI volumes.
- **Heptic interpolation cost.** We use a separable-kernel heptic resampler in
  `affine.py:apply_affine_interp`; AFNI's shear-warp composition avoids the full
  7-tap separable evaluation by decomposing the rotation into 1D shears. We have
  not implemented a shear-warp path.
- **Iteration count.** We may simply be converging more slowly than 3dvolreg
  for sub-voxel motion regimes. Worth profiling the Gauss–Newton step count per
  volume vs. 3dvolreg's reported iterations.

### Why this may not be fixable
Matching single-core AFNI speed in Python + PyTorch for this kind of small-volume
iterative alignment is a known losing proposition — the constant overheads favor
C. GPU helps once volumes are large or many runs are batched, but our current
structure aligns one volume at a time. Batched per-volume Gauss–Newton across the
whole timeseries on GPU could plausibly win, but would be a substantial rewrite.

### What would help confirm the diagnosis
1. Profile `ffs_moco` with `torch.profiler` on a single run: fraction of time in
   Python vs. kernel launches vs. actual compute.
2. Record per-volume iteration counts and compare to what AFNI prints with
   `-verbose`.
3. Time the resampling-only path (a known matrix + heptic apply) vs. 3dvolreg
   with the matrix applied via `3dAllineate -1Dmatrix_apply`.

Until then: accept the speed gap as the cost of a portable PyTorch implementation
and focus on correctness + GPU batching for multi-run workflows.

## ICA Temporal Concat differences (closer, not matched)

After re-aligning `_run_concat_ica` to MELODIC's `setup_migp` path and adding
the `n_eff = floor(V / (2.5 * resels))` correction for the MP noise floor
(was previously raw voxel count), the over-count on ds005165 dropped but is
not resolved:

- Before fix: ffs=258 vs melodic=147
- After fix: ffs=194 (rest) / 192 (localizer) vs melodic=147
- Matched-component mean |r| = 0.40, coverage@0.5 = 19/147

Remaining suspects (unconfirmed):
- **`cov_r` centering semantics.** MELODIC's `std_pca` calls
  `cov_r(Mat, false, econ)` with no explicit `remmean` on the main
  `perf_pca` path, and with `remmean(in,2)` inside `varnorm`. FFS
  row-centers explicitly in both. If `cov_r` actually centers by column
  means (or doesn't center), the noise eigenvalue distribution shifts and
  PPCA will keep a different number of components. Confirm by reading
  `miscmaths::cov_r`.
- **`update_mask` after varnorm.** MELODIC calls `update_mask(Mask, Data)`
  on the full concat (`meldata.cc:568-572`) to drop constant/degenerate
  voxels; all downstream stages (varnorm, PCA, whitening, ICA) then run
  on the reduced matrix. FFS applies an analogous filter only inside
  `estimate_ica_component_count` (`filter_voxels_for_melodic_model_order`)
  and not to the main PCA/whitening input. Whether the reduced voxel count
  also feeds back into `resels` needs checking.
- **Constant-voxel handling in varnorm.** MELODIC sets `noise_std = 1` and
  zeroes the column for `noise_std < 0.01` (`melhlprfns.cc:167-171`). FFS
  matches this but the `const_mask` threshold includes `input_std < 1e-6`
  as an extra condition — possibly zeroing more columns than MELODIC and
  skewing the tail of the eigenspectrum.
- **PCA input for the main whitening.** MELODIC `perf_pca` runs
  `std_pca(in, ...)` with no remmean. FFS row-centers before the eigen
  decomposition. Suspected to be equivalent (since per-voxel demean
  enforces zero column-means before concat and the row-mean drift should
  be small), but not verified head-to-head.

Next debugging step: dump the raw eigenvalue spectrum (pre- and
post-MP-adjustment) from both implementations on the same input, so we can
see whether the discrepancy is in the PCA spectrum itself or only in the
PPCA/Minka stopping criterion.

## ICA Tensor-ICA (MELODIC `--approach=tica`) not implemented

FFS ships a `-tensor` mode (`_run_tensorial_ica`) that does spatial
concatenation `(T, n_runs*V)` with a shared temporal mixing and per-run
spatial maps — adequate for multi-echo or identical-design datasets where
the user just wants shared temporal structure.

This is **not** the same as MELODIC's tensor ICA (`--approach=tica`):

- MELODIC tICA runs `setup_classic` with per-run PCA+whitening stacked
  block-diagonally into `(n_runs*order, V)`, then after ICA factors the
  temporal mixing via `krfact` (Khatri-Rao / Kronecker factorisation,
  `melodic_src/melodic/melhlprfns.cc:krfact`, `meldata.cc:218-243`) into
  three modes: spatial × temporal × subject-loadings.
- Our `-tensor` mode produces two modes only (shared temporal + per-run
  spatial) and does no Kronecker factorisation.

Implementing true tICA would require:
1. Per-run PCA+whitening stacked block-diagonally to `(n_runs*order, V)`.
2. ICA on the stacked matrix.
3. `krfact` port in `decomposition/` to decompose the `(n_runs*order, k)`
   mixing into `(n_runs, k)` subject-loadings and `(order, k)` shared
   temporal modes.
4. Output `_Smodes.txt` / `_Tmodes.txt`-compatible files.

Not blocking for current benchmark goals; note as a future alignment item
when group-ICA becomes a priority.

## Parametric Duration Modulation (future feature)

Per-event HRF amplitude scaling by event duration — i.e., each event's HRF is scaled
by that event's individual duration rather than treating all events of a condition as
identical.

### What this would look like
BIDS events TSV already provides per-event `duration` values. The feature would use
these to scale the HRF amplitude (or boxcar height) for each individual event before
convolution with the canonical HRF. This is distinct from the current approach where
all events of a condition share one duration (derived from the unique values in the TSV).

### Why it is not implemented yet
The current pipeline works at the condition level: `all_onsets[cond][run]` is an array
of onset times, and `durations[cond]` is a single scalar. Supporting per-event
modulation requires threading per-event duration values through:
- `create_onset_matrix_microtime` (currently takes a scalar duration per condition)
- `build_glm_design` / `build_single_trial_design` (same assumption)
- The HRF convolution step

This is a moderate refactor — the data structures need to change from scalar durations
to arrays-of-durations — and should be done carefully to avoid breaking the existing
API surface.

### Code locations to modify
- `fastfuncstuff/design/builder.py` — `create_onset_matrix_microtime`: accept per-event
  duration arrays in addition to scalar duration
- `fastfuncstuff/design/bids_events.py` — `parse_bids_events`: return per-event durations
  alongside the per-condition median (already stores them internally in `cond_dur_sets`)
- All CLIs that accept `-events` — thread per-event durations through to the design builder

## ffs_phasereg: broken, producing nonsense output

Phase regression tool runs without errors but produces garbage results on real data.

### Symptoms on real data
- Correlation map is sparse and does not match expected macrovascular structures
- Cleaned magnitude data has lost spatial structure
- R² is always 0 in synthetic tests — the correlation threshold (`|r| > 2/sqrt(n)`) zeros
  out slopes even for genuinely correlated data (too aggressive, or polort=3 on short
  timeseries eats the correlation)
- Slopes are huge when non-zero (±50) — Deming phi=100 clamp inflates slopes on noisy data

### Likely issues
1. **Correlation threshold too aggressive** — needs tuning or replacement with F-test /
   permutation approach
2. **R² formula** — currently `1 - SS_corrected / SS_detrended` with clamp(0,1) hiding
   negative R² (overcorrection). May need to compute from residual (task-removed) signals
   instead of detrended signals
3. **Deming phi clamp** — upper bound of 100 may be too high for real data where magnitude
   variance >> phase variance but not 100:1

### Standalone components that work
- `phasereg/deming.py` — Deming/OLS regression confirmed working (0.712 vs 0.833 error)
- `phasereg/noise.py` — FFT-based variance ratio returns correct phi (~16 for true 16)

### Next steps
1. Test with real magnitude + phase NIfTI data (the real validation)
2. Tune correlation threshold or replace with proper statistical test
3. Verify TENT task removal path end-to-end
4. Allow negative R² as overcorrection diagnostic
5. Test CLI end-to-end with actual files

## ffs_nwarp -phase: working but introduces magnitude artifacts

The `-source` / `-phase` complex warping pipeline is functional and produces correct
phase output. However, the warped magnitude shows artifacts that appear to be introduced
by the phase component.

### What works
- Phase scaling to radians (auto-scales any input range to [-π, π])
- Complex decomposition: mag+phase → real+imag, warp each, recombine
- Phase output is correct

### What needs investigation
- Warped magnitude has artifacts not present when warping magnitude alone (without -phase)
- Root cause unclear — could be interpolation of real/imag components introducing
  Gibbs-like ringing, or phase unwrapping issues before decomposition, or numerical
  precision in the recombination step

## Open TODOs in source (scan 2026-04-20)

Collected from inline `TODO`/`FIXME` markers.

### Memory / chunking — not using `memory.py`
- `fastfuncstuff/glm/xval.py:986` — hybrid-mode accumulator branch uses hardcoded
  batch sizes instead of `compute_chunk_size()`; likely over-thrashing on systems
  where the memory module could allocate a larger batch. Check whether xval has
  its own memory helper that already covers this path before adding a new one.
- `fastfuncstuff/glm/xval.py:1260` — float64 cast is done per-batch just for the
  residual subtraction (`test_data.double() - pred.double()`). Question whether
  promoting the matmul upstream to float64 once would be cleaner at acceptable
  VRAM cost, vs. the current duplication.
- `fastfuncstuff/cli/deconvolve.py:1679` — no chunk-size estimation before
  `fit_glm`; should route through `compute_chunk_size(operation="glm")`.

### Algorithmic gaps
- `fastfuncstuff/decomposition/pca.py:353` — Minka's MLE dimensionality selection
  is not implemented; falls back to 95% variance cumulative. Stub has been there
  a while; either implement or remove the method.
- `fastfuncstuff/cli/hrfopt.py:1018-1019` — `final_results` / `canonical_results`
  are written as `None` in the output bundle; should refit with optimal HRFs and
  with canonical HRF so downstream code has both for comparison.
- `fastfuncstuff/design/hrf_selection.py:388` — split-half xval can produce very
  negative R² for specific HRFs (observed with 9 runs / strategy=0.5 on HRFs 4
  and 17-19) due to sign-flipped OLS betas in low-SNR folds; LORO is immune.
  Needs characterisation across datasets to decide whether to warn or switch
  defaults.

### Minor / CLI surface
- `fastfuncstuff/cli/reml.py:45` — `parse_prefix` imported but not yet applied
  to individual output flags; would unify prefix handling across `-Rbuck`,
  `-Rvar`, etc.
- `fastfuncstuff/cli/tps.py:443` — run boundaries inferred by equal division;
  variable-length runs need a `-num_stimts`-style flag.
- `fastfuncstuff/visualization.py:1289` — PC plotting restricted to noise pool;
  fitting PCs to the whole-brain mask (and saving per-run 4D NIfTIs of PC
  spatial maps) would give a fuller picture of where components live.