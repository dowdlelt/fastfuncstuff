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
cd fastfuncsim
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
fastfuncsim/
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

After installation, these are available as commands:

### Analysis

| Command | Description |
|---------|-------------|
| `ffs_denoise` | Cross-validated noise PC denoising (GLMdenoise) |
| `ffs_hrfopt` | Per-voxel HRF optimization via cross-validation |
| `ffs_reml` | ARMA(1,1) prewhitened GLM (like AFNI 3dREMLfit) |
| `ffs_ridge` | Fractional ridge regression, per-voxel regularization |
| `ffs_xval_r2` | Cross-validated R-squared for model comparison |
| `ffs_build_design` | Build design matrices from AFNI-style onset files |
| `ffs_deconvolve` | Event-related deconvolution (FIR estimation) |
| `ffs_ica` | MELODIC-style ICA with auto component selection |
| `ffs_decompose` | ICA decomposition with stability analysis |
| `ffs_denoisatorial` | Combinatorial denoising (exhaustive PC subset search) |
| `ffs_pathfinder` | Joint HRF + denoising optimization |
| `ffs_tps` | Thin-plate spline HRF estimation |

### Image Processing

| Command | Description |
|---------|-------------|
| `ffs_moco` | Motion correction |
| `ffs_allineate` | Affine alignment |
| `ffs_nwarp` | Non-linear warping |
| `ffs_qwarp` | Qwarp-style non-linear registration |
| `ffs_automask` | Automatic brain masking |
| `ffs_motsim` | Motion artifact simulation |

Each tool accepts `-help` for full usage and options.


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
import fastfuncsim as ffs

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
from fastfuncsim.design.optimization import find_optimal_designs, ISIConstraints

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
