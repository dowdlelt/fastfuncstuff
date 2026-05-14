# fastfuncstuff (ffs)

A small collection of fMRI analysis tools reimplemented in PyTorch, with a GPU-first bias.
TODO: This agent created readme is garbage - sorry, but I wanted to get something to start. its th first thing to fix.

## Why this exists

I have a slow CPU and a fast graphics card. The tools I like to use, namely AFNI
are great, but they are really built to take advantage of server CPU setups.  
This project is an going experiment - how well can current models port good code
to a pytorch like set up. Python chosen not because of speed, of course, but
because I can at least review the code for sanity.  

These are offered as-is, no guarantees. The CLIs have been tested
extensively in day-to-day use, and an in-tree benchmark (`ffs_benchmark`)
compares outputs and timing against reference implementations on a public
dataset.

## Install

`pyproject.toml` is a plain PEP 621 spec (setuptools backend), so any of pip,
uv, or a conda env + pip will work. There is no conda-forge package; the
"conda" path means "make the env with conda, then `pip install -e .`".

Python: `>=3.11` is required, **3.13 advised** (latest with broad PyTorch
wheel coverage). 3.14 may work but is gated on whether your chosen PyTorch
build has wheels for it.

```bash
git clone <repo-url>
cd fastfuncstuff
```

**pip** (simplest):

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e .
```

**uv** (fastest):

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e .
```

**conda** (recommended if you want CUDA without thinking about it):

```bash
conda create -n ffs python=3.13
conda activate ffs
pip install -e .
```

Add `".[dev]"` instead of `.` for tests + linters.

### PyTorch and the GPU

The default `pip install torch` gives you whatever wheel pip resolves for
your platform — usually CPU on Linux, MPS on Apple Silicon. If you want
CUDA, install torch *first* from the official index for your CUDA version,
then `pip install -e .`:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # or cu121, cu126, …
pip install -e .
```

Apple Silicon: the default wheel includes MPS. Some ops still fall back to
CPU; this is a known weak spot — see the cross-cutting CPU-paths note in the
wiki. CPU-only works everywhere but is the slow path; that is the whole
reason this project exists.

### File formats

Inputs read `.nii`, `.nii.gz`, and `.nii.zst` (zstandard) transparently.
Default outputs are `.nii.gz` written with `pigz` when available (parallel
gzip — much faster than the stdlib path on large 4D files).

## Command-line tools

Every CLI is registered as a console script and accepts `-help`. Flag style
follows AFNI conventions (single dash) where possible.

### GLM and design

| command | description |
|---|---|
| `ffs_build_design` | Build a 1D design matrix from onsets, durations, an HRF model, polynomials, and motion. Output is readable by `ffs_reml` and AFNI tools. |
| `ffs_reml` | OLS / ARMA(1,1) prewhitened GLM with REML grid search over (a, b). AFNI `3dREMLfit`-style bucket output, FDR curves attached. |
| `ffs_ridge` | Fractional ridge regression for single-trial betas (GLMsingle Type D). Per-voxel optimal fraction by cross-validation. |
| `ffs_deconvolve` | FIR / event-related deconvolution without an assumed HRF shape. |
| `ffs_tps` | Thin-plate-spline HRF estimation with cross-validated smoothness. Global or per-voxel λ. |
| `ffs_xval_r2` | Cross-validated R² maps (LORO or split-half) with proper nuisance projection. |

### HRF and denoising

| command | description |
|---|---|
| `ffs_hrfopt` | Per-voxel HRF library selection. Tests each HRF in a library by LORO CV; refits with the winner. Canonical or PIGHS libraries. |
| `ffs_denoise` | GLMdenoise-style noise-PC denoising (Kay et al. 2013). Identifies a noise pool, extracts PCs, picks the count by LORO CV. |
| `ffs_denoisatorial` | Exhaustive 2^k subset evaluation of noise PCs, when you want the best non-contiguous combination rather than a prefix. |
| `ffs_pathfinder` | Joint HRF + denoising optimisation. Picks the HRF that works best with the denoising level chosen for that voxel. |
| `ffs_phasereg` | Magnitude-on-phase Deming regression for macrovascular BOLD suppression (Menon 2002, Curtis 2014, Stanley 2021; phaseprep parity). |
| `ffs_nordic` | NORDIC-style patch-SVD denoising. Magnitude-only or complex (mag + phase), with optional g-factor map. |
| `ffs_sauna` | NORDIC-adjacent denoiser. g-factor from trailing noise volumes + Gavish–Donoho optimal singular-value shrinkage. |

### Decomposition

| command | description |
|---|---|
| `ffs_ica` | MELODIC-style probabilistic ICA. Auto component count, GGM mixture-model thresholding, MIGP for temporal concat, optional ICASSO and depth-lag classification. |
| `ffs_decompose` | ICA with an emphasis on stability via ICASSO clustering. |

### Image processing and registration

| command | description |
|---|---|
| `ffs_moco` | Rigid-body motion correction (Gauss–Newton, heptic resampling). Writes AFNI-compatible motion files. |
| `ffs_allineate` | 6/9/12-parameter affine alignment. |
| `ffs_qwarp` | Iterative nonlinear warp estimation (`3dQwarp`-style). |
| `ffs_nwarp` | Apply a warp to a volume or 4D timeseries. Supports complex (mag + phase) warping. |
| `ffs_slicetime` | Slice-timing correction (`3dTshift`-style), Fourier or sinc. |
| `ffs_motsim` | Motion-simulation nuisance regressors (Patriat, Reynolds & Birn 2017). |
| `ffs_util_automask` | Automatic brain mask from EPI. |
| `ffs_util_pcwarp` | PC-based warp-field utilities. |
| `ffs_spatial_xcorr` | Spatial cross-correlation matrix between two 4D volumes within a mask, with optimal matching and consistency metrics. |

### Benchmarking

| command | description |
|---|---|
| `ffs_benchmark` | Run AFNI and `ffs_*` tools side by side on a BIDS dataset (default: OpenNeuro ds005165) and compare outputs and timing. `-validate-only` skips re-running and just compares. |

## A typical pipeline

GLMsingle-style single-trial analysis:

```
ffs_denoise   -> noise PCs + count
ffs_hrfopt    -> per-voxel HRF
ffs_ridge     -> single-trial betas with per-voxel ridge
```

Or a more conventional GLM:

```
ffs_build_design  -> design.1D
ffs_reml          -> betas, t-stats, F-stats, FDR curves
```

CLIs are Python-callable — every tool exposes `def main(argv: list[str] | None = None)`,
so you can drive a pipeline from a script without going through subprocess:

```python
from fastfuncstuff.cli.reml import main as ffs_reml
ffs_reml(["-input", "run*.nii.gz", "-matrix", "design.1D", "-prefix", "sub01"])
```

## Python API

The library is usable directly. A skeleton:

```python
import fastfuncstuff as ffs

device = ffs.get_device()  # CUDA / MPS / CPU
hrf    = ffs.get_canonical_hrf(stim_duration=2.0, tr=1.5, device=device)
design = ffs.build_glm_design(onsets, hrf, n_timepoints=200, device=device)

results = ffs.fit_glm(data, design, tr=1.5, device=device)
# .betas, .r2, .tstats, .fstats, .sigma2, .meanvol

ffs.write_glm_results_nifti(results, output_dir="./out", prefix="sub01",
                            condition_names=["face", "house"])
```

Other entry points worth knowing about: `fit_glm_arma11`, `fit_glm_hrf_library`,
`generate_fmri_noise`, `simulate_fmri_run`, and `find_optimal_designs` (Liu &
Frank efficiency). See `docs/` for details.

## Status

Active, single-author, very much a research codebase. 

## License

MIT. See `LICENSE`.

## References

Primary inspirations and direct method references — most are linked from
docstrings in the relevant module:

- AFNI: Cox 1996; `3dREMLfit`, `3dDeconvolve`, `3dQwarp`, `3dTshift`,
  `3dLocalstat` tech notes.
- GLMdenoise: Kay, Rokem, Winawer, Dougherty & Wandell (2013), *Front Neurosci*.
- GLMsingle: Prince, Charest, Kurzawski et al. (2022), *eLife*.
- Fractional ridge: Rokem & Kay (2020), *GigaScience*.
- MELODIC / probabilistic ICA: Beckmann & Smith (2004), *IEEE TMI*.
- MIGP: Smith, Hyvärinen, Varoquaux, Miller & Beckmann (2014), *NeuroImage*.
- NORDIC: Moeller et al. (2021), *NeuroImage*.
- Phase regression: Menon (2002); Curtis et al. (2014); Stanley et al. (2021);
  Liem (phaseprep, 2023).
- Optimal SVHT / shrinkage: Gavish & Donoho (2014, 2017).
- Motion simulation: Patriat, Reynolds & Birn (2017).
- Design efficiency: Liu & Frank (2004); Buracas & Boynton (2002); Das et al. (2023).
