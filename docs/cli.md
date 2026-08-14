# CLI reference

This is a working reference for the command-line tools installed by `fastfuncstuff`. Every tool exposes a full argparse `-help` listing — this document covers what each tool is for, the minimum command to make it run, and the flags most users will actually care about. For the exhaustive option list, run `<tool> -help`.

Conventions used below:

- Required arguments are listed in the minimum-example commands.
- Flags use the AFNI-style single-dash convention (e.g. `-prefix`), matching the tools themselves.
- All tools accept `-device {cuda,mps,cpu}` and `-verb {0,1,2}` (or the aliases `-quiet` / `-verbose`). These are omitted from the per-tool notes unless behaviour differs.
- On Apple Silicon, `ffs_formwarp` and `ffs_qwarp` resolve `-device auto` to CPU because
  PyTorch's 3-D grid-sample backward currently falls back from MPS to CPU. Explicit
  `-device mps` remains available. `ffs_optiwarp` is forward-only and can benefit from
  MPS, particularly with its LK and Horn–Schunck force models.
- `ffs_nwarp` is also a strong MPS path for full-size volumes. Static nonlinear
  4-D applies batch frames with shared coordinates; time-varying motion and joint
  slice timing retain their per-frame or sliding-window algorithms.
- `ffs_nordic`, `ffs_blipflip`, and `ffs_segment` also resolve auto to CPU on Mac.
  NORDIC's usual complex SVD falls back to CPU; blipflip and segment require operations
  MPS cannot perform. Whole-brain `ffs_pyrf` can benefit from MPS, while `ffs_ica`
  generally remains faster on the Mac CPU.
- `-prefix` (or `-output`) is the output path; many tools write multiple files derived from this prefix.

---

## Analysis

### `ffs_build_design` — build a design matrix

Assemble a `.xmat.1D` design matrix from stimulus timing files, similar to `3dDeconvolve -x1D_stop` but with a cleaner syntax. Pairs with `ffs_reml` and `ffs_xval_r2`.

```
ffs_build_design \
    -input run01.nii.gz run02.nii.gz \
    -polort 3 \
    -stim times.task.txt 'SPMG1(5)' task \
    -ortvec motion.1D motion \
    -xmat X.xmat.1D
```

Key flags:

- `-stim FILE HRF LABEL` / `-stim_IM FILE HRF LABEL` — stimulus regressor; `_IM` produces one column per event (single-trial).
- `-ortvec FILE LABEL` for nuisance regressors covering the full concatenated length; `-padortvec FILE LABEL RUN` for per-run regressors that should be zero-padded automatically.
- `-gltsym 'SYM: +1*A -1*B' LABEL` for symbolic contrasts.
- `-TR` overrides the TR read from `-input`.

### `ffs_reml` — ARMA(1,1) GLM fit

Whitened GLM with per-voxel ARMA(1,1) noise modelling. Functionally analogous to `3dREMLfit`. 

```
ffs_reml \
    -input run*.nii.gz \
    -matrix X.xmat.1D \
    -Rbuck stats_reml.nii.gz \
    -Rvar arma_params.nii.gz \
    -mask mask.nii.gz
```

Key flags:

- Pass in a BIDS events file with `-events` and skip the headache
- or pass `-onsets` and `durations and let `ffs_reml` build the design internally.
- `-matrix` for an existing design, from 3dDeconvolve (for copmarison, for example)
- Output controls: `-Rbuck` (REML betas + stats), `-Rbeta` (betas only), `-Rerrts` / `-Rfitts`, `-Obuck` for OLS comparison.
- `-load_Rvar` to reuse a previous `-Rvar` and skip the (a, b) grid search on re-runs.
- `-add_fdr` writes AFNI `FDRCURVE` attributes for in-GUI thresholding (and faster!)
- `-single_trials LABEL` reorders single-trial betas by onset time.

### `ffs_deconvolve` — FIR / TENT / cubic-spline deconvolution

Model-free HRF estimation with FIR or TENT bases. Auto-detects TR-locked timing; cross-validates the window length when asked.

```
ffs_deconvolve \
    -input run*.nii.gz \
    -events sub-01_task-foo_run-*_events.tsv \
    -prefix decon \
    -mask mask.nii.gz
```

Key flags:

- Provide timing as either `-onsets` (AFNI) or `-events` (BIDS TSVs). With `-events`, durations come from the TSV; with `-onsets`, give `-durations`.
- `-model {AUTO,FIR,TENT,TENTzero,CSPLIN,CSPLINzero}` — `AUTO` picks based on TR-locking.
- `-window '0 15'` for a shared HRF window, or `-window '0,15 0,20 0,25'` for per-condition windows.
- `-xval_tr_range N` cross-validates the upper bound of the TENT window over ±N TRs. (untested WIP)
- `-event_ignore fixation null` to skip uninteresting trial types.

### `ffs_xval_r2` — cross-validated R² for an existing design

Drop a fitted design matrix in, rapidly get out cross-validated R² maps. Useful for model thinking about data, and designs.

```
ffs_xval_r2 \
    -input run*.nii.gz \
    -matrix X.xmat.1D \
    -prefix xvalR2 \
    -cv_strategy 1
```

Key flags:

- `-cv_strategy` — float (0-1) for train fraction, integer for leave-N-runs-out. `1` = LORO.
- `-metric {cod,corr,corr2}` — coefficient of determination or Pearson.
- `-R2method fast` uses ~3 MB streaming stats (LORO only).

### `ffs_ridge` — ridge regression / GLMsingle-style single-trial

Single-trial or condition betas with cross-validated ridge fraction selection.

```
ffs_ridge \
    -input run*.nii.gz \
    -events sub-01_task-foo_run-*_events.tsv \
    -prefix ridge \
    -single_trials
```

Key flags:

- `-ridge_fracs START END STEP` — fraction of OLS norm to retain (1 = OLS, 0 = max regularization).
- `-single_trials` switches from timeseries CV to beta-space CV (GLMsingle convention).
- `-autoscale` / `-no_autoscale` for GLMsingle-style post-hoc shrinkage correction (on by default).
- `-hrf_opt PREFIX` to use per-voxel HRFs estimated by `ffs_hrfopt`.
- also supports nuisance regressors, for full GLMsingle similarity

### `ffs_hrfopt` — per-voxel HRF library selection

Searches a library of HRFs (double-gamma variants or PIGHS half-cosine basis) and picks the best one per voxel by cross-validation.

```
ffs_hrfopt \
    -input run*.nii.gz \
    -events sub-01_task-foo_run-*_events.tsv \
    -prefix hrfopt
```

Key flags:

- `-hrf_mode {library,pighs}` — `library` is double-gamma variants (default); `pighs` is a parametric half-cosine basis with configurable ranges.
- `-n_hrfs N` — library size (PIGHS only).
- `-select {xval,full}` — pick by CV or by full-data fit.
- Outputs `{prefix}_hrf_index.nii.gz`, consumable by `ffs_ridge -hrf_opt`, `ffs_denoise -hrf_opt`, and `ffs_denoisatorial -hrf_opt`.

### `ffs_denoise` — cross-validated PC/IC denoising

GLMdenoise/GLMsingle-style noise pool extraction with cross-validated component count.
Supports non-TR locked events AND conditions can have variable durations!

```
ffs_denoise \
    -input run*.nii.gz \
    -events sub-01_task-foo_run-*_events.tsv \
    -prefix denoise \
    -mask mask.nii.gz
```

Key flags:

- `-noise {pca,ica}` — type of components to extract.
- `-r2_threshold 0.05` defines the noise pool (voxels with CV R² below threshold).
- `-max_comps` upper bound; `-pcstop 1.05` stopping rule (within 5% of max CV R²).
- `-brainthresh PERCENTILE FRACTION` for unmasked data to exclude background before the noise pool selection.
- `-save_pcs timecourse` to keep the chosen noise regressors as a `.1D` file.

### `ffs_denoisatorial` — combinatorial PC denoising

Same idea as `ffs_denoise` but evaluates all 2^k subsets of the top PCs (or just singletons with `-singleton_only`). 
_Slow_, idiotically thorough and likely a bad approach.
But some testing shows that `-singleton` might just be a good idea (and it does xval).
Here we only keep in PCs that improved xval R2 - so it will skip over bad ones that hurt the model AND each run can have a variable number of noise regressors!

```
ffs_denoisatorial \
    -input run*.nii.gz \
    -onsets cond1.1D cond2.1D \
    -durations 3 \
    -prefix denoisatorial \
    -max_pcs 7
```

Key flags:

- `-max_pcs 7` → 128 combinations (default). 8 → 256, 10 → 1024.
- `-singleton_only` evaluates k+1 combinations instead of 2^k.
- `-selection_strategy parsimonious` prefers the smallest combination within 1% of the max CoD.

### `ffs_pathfinder` — joint HRF + denoising

Runs HRF library search and PC denoising in a single CV loop, returning a per-voxel HRF index and selected noise regressors together. WIP, not tested, likely broken.

```
ffs_pathfinder \
    -input run*.nii.gz \
    -onsets cond1.1D cond2.1D \
    -durations 3 \
    -prefix path
```

Useful when you don't want to commit to an HRF before picking the noise regressors. Roughly the union of `ffs_hrfopt` and `ffs_denoise` options; see `-help` for the full grid.

### `ffs_tps` — thin-plate-spline HRF estimation

Penalized splines for smooth HRF estimation with cross-validated smoothness λ. Based on Chen et al. (2023, NeuroImage). WIP, not tested, likely broken.

```
ffs_tps \
    -input func.nii.gz \
    -stim_times face.txt house.txt \
    -stim_labels face house \
    -tps_window '0 15' \
    -output_prefix tps
```

Key flags:

- `-optimize_level {global,per_voxel}` — single λ vs adaptive per-voxel.
- `-n_knots N` (auto if omitted) and `-force_zero_edges` for HRFs pinned to 0 at the edges of the window.
- `-save_lambda_map` to inspect where the spline wanted to smooth more.

### `ffs_ica` — whole-brain ICA

Run-wise FastICA / Infomax with optional ICASSO clustering, MELODIC-style component-number selection, MIGP for long timeseries, and a depth-lag classifier.
At its core it is designed to reproduce MELDDIC outputs on a given dataset - that is, select the same number of components and reproduce similar maps (bt 50x faster on my CPU).

```
ffs_ica \
    -input run*.nii.gz \
    -prefix ica \
    -num_comps auto \
    -mask mask.nii.gz
```

Key flags:

- `-num_comps` — integer, float in (0, 1) for variance fraction, or `auto`/`melodic`/`hybrid`/`mp`.
- `-icasso -icasso_runs 100` for stability-based component selection.
- `-migp` for very long timeseries (MIGP-style incremental PCA).
- `-depth_lag` enables the cortical-depth-vs-lag classifier (writes a "good/bad" label per component).
- `-trace DIR` dumps per-iteration diagnostics (slow, useful for debugging).

### `ffs_decompose` — PCA and/or ICA

Lower-level PCA/ICA than `ffs_ica`. Use this when you want PCA output, or a simple ICA without the run-wise / MELODIC-style scaffolding.

```
ffs_decompose \
    -input func.nii.gz \
    -mask mask.nii.gz \
    -pca 0.85 \
    -ica 25 \
    -output decomp
```

Key flags:

- `-pca N` — int (count), float (variance fraction), or `all`.
- `-ica N` / `-icasso N` — fixed count or `auto` with `-ica_range START STOP STEP`.
- `-stability N` — repeat ICA N times to assess component reliability.

### `ffs_phasereg` — phase regression for macrovascular suppression

Removes the macrovascular BOLD component by regressing magnitude on phase using Deming (errors-in-variables) or OLS regression. Designed for high-resolution / high-field acquisitions where phase data is available.

```
ffs_phasereg \
    -magnitude run*_magn.nii.gz \
    -phase run*_phase.nii.gz \
    -prefix phasereg \
    -task_removal tent \
    -events events*.tsv
```

Key flags:

- `-task_removal {none,tent,canonical}` — model and remove task before estimating the magnitude–phase slope. `tent` recommended for task data.
- `-regression {deming,ols}` — Deming is the closed-form errors-in-variables solution; OLS matches Chang & Giovanello (2026) post-NORDIC.
- `-phi_method {fft,residual}` — how to estimate the variance ratio. `fft` is safe for both task and rest; `residual` only after task removal.
- `-freq_range HZ` — frequency band used by `-phi_method fft`.

### `ffs_nordic` — NORDIC denoising

NORDIC / MP-PCA patch-based denoising for magnitude or magnitude+phase 4D data.

```
ffs_nordic \
    -input-magn func_magn.nii.gz \
    -input-phase func_phase.nii.gz \
    -prefix denoised
```

Key flags:

- `-magnitude-only` if you don't have phase.
- `-mp {0,1,2}` to switch to MP-PCA thresholding instead of NORDIC.
- `-kernel-size-pca KX KY KZ` and `-patch-overlap` for the LLR patch geometry.
- `-noise-volume-last N` to use trailing noise-only volumes (preferred when available).
- `-save-gfactor-map` / `-save-residual-map` for QC.

### `ffs_sauna` — noise-volume g-factor + Gavish-Donoho shrinkage

Newer denoiser that tries to cleverly use trailing noise-only volumes to estimate the g-factor map directly, then applies optimal singular-value shrinkage. Requires `-noise-volume-last >= 2`. Similar results, testing required. Perhaps reasonable?

```
ffs_sauna \
    -input-magn func_magn.nii.gz \
    -input-phase func_phase.nii.gz \
    -prefix sauna \
    -noise-volume-last 3
```

Key flags:

- `-gfactor-method {gaussian,polynomial,auto}` — how to smooth/fit the noise-volume std map.
- `-shrinkage {optimal,hard}` — Gavish-Donoho optimal shrinkage or MP-PCA hard threshold.
- `-magnitude-only` for magnitude-only acquisitions.

---

## Image processing

### `ffs_moco` — motion correction

GPU port of `3dvolreg`. Outputs the corrected timeseries plus optional motion parameter files. Slower than `3dvolreg`, lets call it a proof of some principle. 

```
ffs_moco \
    -input epi.nii.gz \
    -prefix epi_mc.nii.gz \
    -1Dfile motion.1D \
    -1Dmatrix_save mat.aff12.1D
```

Key flags:

- `-base N` (volume index) or `-base path/to/3d.nii.gz` (external reference).
- `-cost {wls,lpa,quad}` — `wls` default; `lpa` for cross-contrast.
- `-twopass` for coarse-blur + fine-pass when motion is large.
- `-1Dfile` / `-1Dmatrix_save` / `-dfile` for motion parameter outputs in AFNI format.
- `-save_mean` writes the mean of the corrected output (useful as a reference for `ffs_motsim` or downstream alignment).

### `ffs_slicetime` — slice-timing correction

`3dTshift` equivalent. Reads either a text file or BIDS JSON for slice timings.
The key feature I like is that you can change timesteps here - say, go to a 1s TR when you didn't have that in your data. 

```
ffs_slicetime \
    -input epi.nii.gz \
    -prefix epi_st.nii.gz \
    -tpattern bold.json
```

Key flags:

- `-tzero T` — target time within TR to align to (default: mean of slice times).
- `-resample TR_NEW` — resample to a new TR grid after correction; useful for TR-locking onsets for GLMsingle-style analysis.
- Interpolation: `-Fourier` (default), `-linear`, `-cubic`, `-quintic`, `-heptic`, `-wsinc5`, `-wsinc9`.

### `ffs_allineate` — affine alignment

GPU port of `3dAllineate`. Rigid, affine, or EPI-constrained alignment between two 3D volumes.

```
ffs_allineate \
    -base T1.nii.gz \
    -source EPI_mean.nii.gz \
    -prefix EPI_in_T1.nii.gz \
    -1Dmatrix_save EPI_to_T1.aff12.1D \
    -cost lpc
```

Key flags:

- Mode: `-rigid` (6 DoF), `-affine` (12 DoF, default), `-EPI` (9 DoF).
- `-cost {ls,lpa,lpc}` — `lpa` for similar contrast, `lpc` for cross-contrast (e.g. EPI ↔ anat).
- `-1Dmatrix_apply` to apply a previously saved matrix without searching.
- `-fast` / `-superfast` to cut iterations when you only need rough alignment.

### `ffs_qwarp` — nonlinear warping

GPU port of `3dQwarp` with two modes:

```
# Standard 3D → 3D nonlinear alignment
ffs_qwarp -base T1.nii.gz -source T2.nii.gz -prefix warped

# Timeseries mode: per-volume warping of a 4D file to its own base
ffs_qwarp -base epi_4d.nii.gz -prefix warped
```

Key flags:

- `-base_method {first,mean,median}` and `-base_navg N` to build a robust base in timeseries mode.
- `-chainwarp` initialises each volume's warp from the previous result — faster for fMRI-like timeseries.
- `-affine MAT.aff12.1D` to combine with a pre-computed affine.
- `-noXdis` / `-noYdis` / `-noZdis` to freeze axes (useful for distortion-only warps).
- `-tsmooth SIGMA` for temporal smoothing of the per-volume warp fields.

### `ffs_nwarp` — apply / compose warps

Apply a chain of warps and affine matrices to a magnitude (and optionally phase) volume.

```
ffs_nwarp \
    -source epi.nii.gz \
    -nwarp 'epi_to_T1.aff12.1D T1_to_MNI_warp.nii.gz' \
    -prefix epi_in_MNI.nii.gz \
    -master MNI152.nii.gz
```

Key flags:

- `-nwarp 'A B C'` composes left-to-right (output = `C(B(A(x)))`); these can be `.aff12.1D` matrices or `_WARP.nii.gz` displacement fields.
- `-phase` plus one of `-phase_warp {complex,split,direct,circular}` for warping phase data correctly.
- `-master` defines the output grid; `-dxyz` forces an isotropic voxel size.

### `ffs_motsim` — motion-simulation regressors

Implements Patriat et al. (2017): apply motion parameters to a reference EPI, simulate motion-induced signal changes, then PCA them into regressors of no interest.

```
ffs_motsim \
    -base mean_epi.nii.gz \
    -aff12 epi.aff12.1D \
    -prefix motsim
```

Key flags:

- Input motion: `-aff12 .aff12.1D` (preferred), `-1Dfile motion.1D`, or `-dfile diag.1D` — all from `ffs_moco`.
- `-n_pcs N` (default 12) — components to retain.
- `-variant {forward,backward,both}` — Patriat's `both` (concatenated forward+backward sims) is the default and recommended.

### `ffs_util_automask` — brain mask from a 3D volume

GPU port of `3dAutomask`. Outputs a binary brain mask suitable for the analysis tools.

```
ffs_util_automask -input mean_epi.nii.gz -prefix mask.nii.gz
```

Key flags: `-clfrac` (clip-level fraction, default 0.5, matches AFNI), `-dilate`, `-peelcount`, `-peelthr`.

### `ffs_util_pcwarp` — temporal PCs from per-volume warps

Extracts the dominant temporal PCs from a directory of per-volume warp displacement files written by `ffs_qwarp`. The output `.1D` file can be passed as a nuisance regressor. Think of this like motsim for time-varying nonlinear warps.

```
ffs_util_pcwarp -warp_dir sub01_warps -n_pcs 5 -prefix warpPCs.1D
```

Use `-axes Y` (or any subset of `XYZ`) to restrict the PCA to specific displacement axes — useful when only PE-direction motion matters.

---

## Stats / utilities

### `ffs_spatial_xcorr` — spatial cross-correlation between 4D volumes

Pairwise correlation between sub-bricks of two 4D NIfTI files. Useful for comparing ICA decompositions, beta maps, etc. across pipelines.

```
ffs_spatial_xcorr \
    -a pipeline_A.nii.gz \
    -b pipeline_B.nii.gz \
    -method spearman \
    -abs \
    -plot xcorr.png
```

Key flags:

- `-method {pearson,spearman,kendall}` — Spearman is the right default for cross-pipeline comparisons; Kendall is CPU-only and slow.
- `-abs` for sign-ambiguous data (ICA components, eigenvectors).
- `-mask` for a shared mask, or `-mask_a` / `-mask_b` for per-dataset masks (intersected).
- `-save_matrix` writes the full correlation matrix; `-plot` writes a heatmap with the optimal Hungarian matching highlighted.

---

## Benchmark

### `ffs_benchmark` — accuracy + timing against AFNI / MELODIC / GLMsingle

Runs the full FFS pipeline and the corresponding reference tools on a BIDS dataset (default: OpenNeuro `ds005165`), validates outputs against each other, and records timing. Used to produce the figures in the README.

```
ffs_benchmark -download                            # fetch the default datasets
ffs_benchmark -stages moco,slicetime,glm           # run a subset of stages
ffs_benchmark -validate-only                       # only re-validate existing outputs
ffs_benchmark -plot plots/                         # render timing / speedup plots
```

Key flags:

- `-config YAML` for a custom dataset; `-data-dir` to point at an existing BIDS tree.
- `-force-ffs` / `-force-ref` / `-force-all` to re-run despite cached outputs.
- `-ref-only` runs the AFNI / MELODIC / MATLAB references only — handy on machines without a usable GPU.
- `-list-cache` / `-remove-cache` / `-import-cache` to manage `benchmark_cache.json`.

Stages can be invoked in any combination but **never run them in parallel** — timing data will be corrupted if multiple stages share GPU/CPU resources.
