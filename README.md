# FastFuncSim

GPU-accelerated fMRI analysis toolkit: GLM fitting, denoising, HRF optimization,
ridge regression, ICA decomposition, and design optimization. Built on PyTorch
for automatic MPS/CUDA/CPU device selection.

Implements the GLMsingle/GLMdenoise pipeline (Prince et al., 2022), AFNI-style
ARMA(1,1) prewhitening, fractional ridge regression, and MELODIC-style ICA --
all with GPU acceleration and cross-validated model selection.


## Installation

```bash
git clone <repo-url>
cd fastfuncstuff
pip install -e .
```

With test/dev dependencies:

```bash
pip install -e ".[dev]"
```

Requires Python >= 3.12 and PyTorch >= 2.0. For GPU acceleration, install
PyTorch with CUDA or MPS support.


## Package Structure

```
fastfuncstuff/
  glm/             Core GLM engine, cross-validation, ARMA prewhitening
  design/          Design matrices, HRF generation, onset parsing
  denoise/         Cross-validated noise PC denoising
  decomposition/   PCA, FastICA, ICASSO stability analysis
  simulation/      Noise generation, fMRI simulation, design metrics
  io/              AFNI format support, NIfTI I/O
  processing/      Motion correction, alignment, warping
  cli/             Command-line tools (installed as console scripts)
```


## Command-Line Tools

All tools are installed as commands via `pip install -e .` and accept
`-help` for full usage. Input files can be `.nii`, `.nii.gz`, or `.nii.zst`
(zstandard-compressed NIfTI).


### ffs_denoise -- Cross-validated noise PC denoising

Implements the GLMdenoise algorithm (Kay et al., 2013). Identifies a noise
pool (voxels with low task R-squared), extracts principal components from that
pool, and uses leave-one-run-out cross-validation to select the optimal number
of PCs to include as nuisance regressors. The anti-overfitting strategy:
training data is denoised but test predictions are evaluated against raw data.

Supports PCA or ICA-based noise extraction, automatic component caps via
Marchenko-Pastur thresholding, brainstem/CSF noise pool targeting, and
single-trial design modes. Outputs denoised data, noise PC timecourses,
R-squared maps, and diagnostic plots.

```
ffs_denoise -input run*.nii.gz -onsets face.txt house.txt \
            -durations 2.0 5.0 -tr 1.5 -prefix sub01_denoise
```


### ffs_hrfopt -- Per-voxel HRF optimization

Cross-validated HRF library selection. Tests each HRF shape from a library
(default: 20 canonical variants spanning peak times ~3-8s and with/without
undershoot) against every voxel via LORO cross-validation. Selects the
best-fitting HRF per voxel. Supports both canonical parameter-variation
libraries and PIGHS (half-cosine basis) libraries.

Two-pass GPU-efficient approach: pass 1 evaluates all HRFs in cross-validation,
pass 2 recomputes betas using the selected HRF per voxel.

```
ffs_hrfopt -input run*.nii.gz -onsets face.txt house.txt \
           -durations 2.0 5.0 -tr 1.5 -prefix sub01_hrfopt
```


### ffs_reml -- ARMA(1,1) prewhitened GLM

GPU-accelerated equivalent of AFNI's 3dREMLfit. Fits a GLM with ARMA(1,1)
noise modeling via REML grid search over (a, b) parameter space. Produces
correctly-weighted t-statistics that account for temporal autocorrelation.
Accepts AFNI-style design matrices (`-matrix`) or builds designs from onset
files. Outputs AFNI-compatible bucket files with beta/t-stat pairs.

```
ffs_reml -input run*.nii.gz -matrix design.1D -prefix sub01_reml
```


### ffs_ridge -- Fractional ridge regression

Single-trial beta estimation with per-voxel regularization (GLMsingle Type D).
Uses fractional ridge where the regularization parameter is expressed as a
fraction of lambda_max, giving a bounded [0, 1] parameter space. Cross-validates
to select the optimal fraction per voxel. Can incorporate denoising results
(`-denoise`) and HRF optimization results (`-hrf_opt`) from upstream steps.

```
ffs_ridge -input run*.nii.gz -onsets face.txt house.txt \
          -durations 2.0 5.0 -tr 1.5 -prefix sub01_ridge \
          -denoise sub01_denoise -hrf_opt sub01_hrfopt
```


### ffs_xval_r2 -- Cross-validated R-squared

Computes cross-validated R-squared maps using LORO or split-half CV with
nuisance projection. Useful for comparing model quality across different
preprocessing choices or design specifications without overfitting.

```
ffs_xval_r2 -input run*.nii.gz -onsets face.txt house.txt \
            -durations 2.0 -tr 1.5 -prefix sub01_xval
```


### ffs_build_design -- Design matrix construction

Builds AFNI-compatible design matrices from onset timing files and HRF
specifications. Supports microtime resolution convolution, multiple HRF
models (spmg1, spmg2, spmg3, dmBLOCK, etc.), polynomial drift regressors,
and extra nuisance regressors from motion files. Outputs a 1D matrix file
readable by ffs_reml or AFNI programs.

```
ffs_build_design -onsets face.txt house.txt -durations 2.0 5.0 \
                 -tr 1.5 -n_timepoints 200 -prefix design
```


### ffs_deconvolve -- Event-related deconvolution

FIR-style deconvolution without assuming an HRF shape. Estimates the
hemodynamic response at each lag timepoint using regularized least squares.
Useful for validating HRF assumptions or exploring response dynamics.

```
ffs_deconvolve -input run*.nii.gz -onsets stim.txt -tr 1.5 \
               -n_lags 20 -prefix sub01_fir
```


### ffs_ica -- MELODIC-style ICA

Whole-brain ICA with automatic component estimation. Supports Bayesian
dimensionality estimation (MELODIC-style), ICASSO stability analysis for
robust component selection, and depth-dependent lag analysis for identifying
BOLD vs. non-BOLD components. Preprocessing includes optional spatial
smoothing, polynomial detrending, Fourier high-pass filtering, and
percent-signal scaling. Processes runs independently.

```
ffs_ica -input run1.nii.gz -n_components auto -icasso 25 \
        -prefix sub01_ica
```


### ffs_decompose -- ICA decomposition with stability

Similar to ffs_ica but focused on component stability analysis via ICASSO
clustering. Runs ICA multiple times with different initializations, clusters
the resulting components by similarity, and extracts stable centroids.

```
ffs_decompose -input func.nii.gz -n_components 30 -n_runs 25 \
              -prefix sub01_decomp
```


### ffs_denoisatorial -- Combinatorial denoising

Exhaustive evaluation of all 2^k PC subsets (for moderate k) to find the
optimal non-contiguous combination of noise PCs. Unlike sequential denoising
(which tests prefixes 1..k), this tests every possible subset. Uses LORO
cross-validation with an inner CV loop for criteria voxel selection. Supports
"argmax" and "parsimonious" (fewest PCs within 1% of max) selection strategies.

```
ffs_denoisatorial -input run*.nii.gz -onsets face.txt house.txt \
                  -durations 2.0 -tr 1.5 -max_pcs 10 -prefix sub01_combo
```


### ffs_pathfinder -- Joint HRF + denoising optimization

Jointly optimizes HRF selection and noise PC denoising. For each candidate
HRF, evaluates denoised cross-validated R-squared to find the HRF that
works best with the selected denoising level per voxel.

```
ffs_pathfinder -input run*.nii.gz -onsets face.txt house.txt \
               -durations 2.0 -tr 1.5 -prefix sub01_pathfinder
```


### ffs_tps -- Thin-plate spline HRF estimation

Estimates the HRF using penalized cubic splines with automatic
cross-validated smoothness selection. Adapts to local SNR: high-SNR voxels
get less smoothing, low-SNR voxels get more. Supports global optimization
(one smoothness for all voxels) or per-voxel optimization.

```
ffs_tps -input func.nii.gz -stim_times onsets.txt -tps_window 0,20 \
        -n_knots 15 -optimize_level global -output_prefix sub01_tps
```


### Image Processing

| Command | Description |
|---------|-------------|
| `ffs_moco` | Rigid-body motion correction with GPU-accelerated cost functions |
| `ffs_allineate` | Affine (6/9/12-parameter) alignment between volumes |
| `ffs_nwarp` | Non-linear warping with regularized displacement fields |
| `ffs_qwarp` | Qwarp-style iterative non-linear registration |
| `ffs_automask` | Automatic brain mask generation from EPI data |
| `ffs_motsim` | Simulate motion artifacts for testing correction pipelines |
| `ffs_util_pcwarp` | PC-based warp field analysis and manipulation |


### I/O Notes

All tools read `.nii`, `.nii.gz`, and `.nii.zst` (zstandard-compressed NIfTI)
transparently. Zstandard offers ~30% better compression than gzip at much
higher decompression speed, useful for large datasets. Install zstandard
support with `pip install zstandard`.


## Typical Workflow

A standard GLMsingle-style analysis pipeline:

```
1. ffs_denoise      Identify and remove structured noise (noise PC selection)
2. ffs_hrfopt       Select best HRF per voxel from a library
3. ffs_ridge        Fit single-trial betas with per-voxel regularization
```

Or for a simpler GLM:

```
1. ffs_build_design   Construct design matrix from onset times
2. ffs_reml           Fit GLM with ARMA(1,1) prewhitening
```


## Python API

```python
import fastfuncstuff as ffs

device = ffs.get_device()  # auto-detect MPS / CUDA / CPU

# Build design from onset times
hrf = ffs.get_canonical_hrf(stim_duration=2.0, tr=1.5, device=device)
design = ffs.build_glm_design(onsets, hrf, n_timepoints=200,
                               mode='assumed', device=device)

# Fit GLM
results = ffs.fit_glm(data, design, tr=1.5, device=device)
# results.betas, results.r2, results.tstats, results.fstats

# Write output
ffs.write_glm_results_nifti(results, output_dir="./out", prefix="sub01",
                             condition_names=["face", "house"])
```


### GLM Fitting

`fit_glm()` is the core solver. It handles polynomial detrending, extra
nuisance regressors, multi-run concatenation, and automatic GPU chunking.

```python
results = ffs.fit_glm(
    data,                     # (n_voxels, n_timepoints) or list of runs
    design,                   # design matrix or list per run
    tr=1.5,
    max_poly_degree=3,        # Legendre polynomial drift removal (default: auto)
    extra_regressors=None,    # motion params, noise PCs, etc.
    want_residuals=False,     # return residuals (memory intensive)
    device=device,
)
```

Returns a `GLMResults` object with `betas`, `r2`, `r2_run`, `tstats`,
`fstats`, `sigma2`, `meanvol`, and optionally `residuals` and `predicted`.


### HRF Library

Try multiple HRF shapes, pick the best per voxel:

```python
library = ffs.get_hrf_library('canonical', stim_duration=2.0,
                               tr=1.5, n_hrfs=20, device=device)
results, hrf_index, r2_all = ffs.fit_glm_hrf_library(
    data, onsets, library, tr=1.5, device=device)
```


### ARMA(1,1) Prewhitening

Correct for temporal autocorrelation (equivalent to AFNI 3dREMLfit):

```python
arma_results = ffs.fit_glm_arma11(
    data, design, tr=1.5,
    max_poly_degree=3,
    device=device,
)
```


### Noise Generation and Simulation

Generate realistic fMRI noise for power analysis:

```python
noise = ffs.generate_fmri_noise(
    tr=1.5, duration_s=300, matrix_size=(64, 64, 30),
    pink_exp=1.0, resp_freq=0.3, cardiac_freq=1.0,
    device=device,
)

data = ffs.simulate_fmri_run(
    onsets, betas=[3.0, 2.0], hrf=hrf, tr=1.5,
    n_timepoints=200, matrix_size=(64, 64, 30),
    device=device,
)
```


### Design Optimization

Optimize experimental designs using Liu & Frank efficiency metrics:

```python
from fastfuncstuff.design.optimization import find_optimal_designs, ISIConstraints

constraints = ISIConstraints(min_isi=2.0, max_isi=12.0, mean_isi=5.0)
designs = find_optimal_designs(
    n_conditions=3, n_trials=60, tr=1.5,
    constraints=constraints, n_candidates=1000,
)
```


## Key Design Decisions

**Legendre polynomials for drift modeling.** Raw monomials are numerically
unstable. All polynomial regressors use orthogonal Legendre polynomials,
zero-padded per run during cross-validation.

**Fractional ridge regularization.** Ridge fractions are expressed as fractions
of lambda_max (range [0, 1]), making the parameter space bounded and
interpretable regardless of data scale.

**Microtime onset alignment.** Onsets are placed on a sub-TR grid using
`bins_per_tr = round(tr/dt)` to avoid cumulative drift when tr/dt is
not an integer.

**GPU memory management.** Large arrays (voxel timeseries) are processed in
chunks; small matrices (design, polynomials) stay on GPU. The `memory` module
computes safe chunk sizes for the current device.


## License

MIT License. See LICENSE for details.


## References

- Prince JS, Charest I, Kurzawski JW, et al. (2022). Improving the accuracy
  of single-trial fMRI response estimates using GLMsingle. *eLife*, 11:e77599.
- Kay K, Rokem A, Winawer J, et al. (2013). GLMdenoise: a fast, automated
  technique for denoising task-based fMRI data. *Frontiers in Neuroscience*, 7:247.
