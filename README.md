# fastfuncstuff (ffs)

A small collection of fMRI analysis tools reimplemented in PyTorch, with a GPU-first bias.
That said - there has been work to speed up the cpu branches - and in some cases they might be faster. 
For Mac, use of cpu is recommended. CUDA is preferred, where possible. 

## Why this exists

I have a slow CPU but I was able to get my hands on a reasonble GPU. 
The tools I like to use, namely AFNI are great, but they are really built to take advantage of server CPU setups.
This project is an ongoing experiment - how well can current models port good code to a pytorch like set up.
Python chosen not because of speed, of course, but because I can at least review the code for sanity.
It also makes installion much easier - sure, pure CUDA/C++ would be faster but what a nightmare...
These are offered as-is, no guarantees.
The CLIs have been tested extensively in day-to-day use, and an CLI tool to benchmark (`ffs_benchmark`) that compares outputs and timing against reference implementations on public datasets.

The current validation/benchmark data is in `notebooks/ffs_benchmark_showcase.ipynb` - they you can see how well ffs is able to match the reference implementations. 

## Install

`pyproject.toml` is a plain PEP 621 spec (setuptools backend), so any of pip,
uv, or a conda env + pip will work. There is no conda-forge package; the
"conda" path means "make the env with conda, then `pip install -e .`".

Python: `>=3.11` is required, **3.13 advised** (latest with broad PyTorch
wheel coverage). 3.14 may work but is gated on whether your chosen PyTorch
build has wheels for it.

```bash
git clone https://github.com/dowdlelt/fastfuncstuff.git
cd fastfuncstuff
```

**pip**:

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e .
```

**uv**:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e .
```

**conda** (What I used, because I'm used to it):

```bash
conda create -n ffs python=3.13
conda activate ffs
pip install -e .
```

Add `".[dev]"` instead of `.` for tests + linters.

### PyTorch and the GPU
*note* Mac should use -device cpu where possible, MPS is flakey. 
The CPU paths for most tools are also quite fast, and get attention.

The default `pip install torch` gives you whatever wheel pip resolves for
your platform — usually CPU on Linux, MPS on Apple Silicon. If you want
CUDA, install torch *first* from the official index for your CUDA version,
then `pip install -e .`:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # or cu121, cu126, …
pip install -e .
```

Apple Silicon: the default wheel includes MPS. Many ops still fall back to
CPU and some break entirely (float64 support, for example) this is a known weak spot CPU-only works everywhere but is the slow path. 
I want to improve this, so consider that is on the TODO list. 

### File formats

Inputs read `.nii`, `.nii.gz`, and `.nii.zst` (zstandard) transparently.
ZSTD support also offers another speedup, I tend to work with large files, and compressing and decompressing is a headache. This helps (but reduces compatability).
Default outputs are `.nii.gz` written with `pigz` when available.

## What I'm very happy with

WIP.

### ffs_nwarp
One of the things I am most happy with is `ffs_nwarp` which is a near drop in replacment for AFNI's `3dNwarpApply` - produces near identical output, with wsinc5 interp (though permits negative values) but about 50x faster. 
This is particularly noticeable on my slow CPU, even when I use 10 cores. 
This for example (as of latest test), "just works" 
```shell
ffs_nwarp -interp wsinc5 \
 -master ./sswarper_output/anatQQ.sub-01.nii \
 -nwarp './sswarper_output/anatQQ.sub-01_WARP.nii ./sswarper_output/anatQQ.sub-01.aff12.1D anat_al_keep_e2a_only_mat.aff12.1D moco_sub-01_ses-01_task-localizer_run-1_bold_mat.aff12.1D' \
 -source ../sub-01/ses-01/func/sub-01_ses-01_task-localizer_run-1_bold.nii \
 -prefix ./ffs_mni_run-01.nii.gz -dxyz 3
```
What is also very fun is that this supports phase input and you can choose to warp directly (assume phase is unwrapped/interpolatable) or internally convert to real/imag, apply warps, and save phase out from that. This is useful for `ffs_phasereg`, for example. In other words, once you have magnitude processed its very easy to have the phase tag along. 

### ffs_reml with per-voxel HRFs
`ffs_reml` can take the output of `ffs_hrfopt` and fit ARMA(1,1) REML using each voxel's best-fitting HRF. Pass `-hrfopt_prefix PREFIX` (the same prefix you gave to `ffs_hrfopt`) and the design is rebuilt per HRF group. Pairs naturally with `-single_trials LABEL`, which rebuilds the design with one regressor per event (GLMsingle-style) and writes chronologically ordered betas.

### ffs_perm (beta)
Non-parametric permutation testing — generates a null by sign-flipping/permuting labels (FSL/`randomise`-style) and gives you voxelwise + cluster-corrected null distributions. The whole thing is GPU-batched so 5,000 permutations across a full brain takes about 2 minutes here, ~50x faster than the conventional `randomise` workflow (on 8 cores, my machine). Produces AFNI-compatible cluster tables. Still beta — see `-help` for current options.

### ffs_design_spec and ffs_util_concalc
A specification approach to create design matricies - read in BIDS events and data lengths to produce a TOML file.
Here you can specify models (SPMG1 or others), durations, etc.
Comments tell you about your data. 
This can be edited and compiled to a design matrix (fully AFNI compatable).
Contrast can easily be added and - this part I really do love - you can run `ffs_util_concalc` to add those contrasts to an existing OLS or REML stats dataset.
No need to rerun the GLM! 
Also permits correcting a contrast mistake and doing an inplace update of the stats. 


## Command-line tool listing

I work in the command line, and I tend to use bash scripts - so, everything is more or less built with that in mind. 
Every CLI is registered as a console script and accepts `-help`. Flag style follows AFNI conventions (single dash).

### GLM and design

| command | description |
|---|---|
| `ffs_reml` | OLS / ARMA(1,1) prewhitened GLM with REML grid search over (a, b). AFNI `3dREMLfit`-style bucket output, FDR curves attached. Accepts per-voxel HRF assignments via `-hrfopt_prefix`, and `-single_trials LABEL` rebuilds the design with one regressor per event. Can take `-spec design.spec` to compile the whole design itself. |
| `ffs_design_spec` | Compile a single `design.spec` TOML (runs, events, per-task HRF, nuisance, contrasts with wildcards/F-tests) into an AFNI `.xmat.1D`. `stub` writes a skeleton from `events.tsv` + NIfTI headers. |
| `ffs_util_concalc` | Recompute or add GLM contrasts (t and F) on an existing REML bucket without rerunning the GLM — reuses the per-(a,b) ARMA inverse so it's near-free. |
| `ffs_ridge` | Fractional ridge regression for single-trial betas (GLMsingle Type D; Prince et al. 2022). Per-voxel optimal fraction by cross-validation. |
| `ffs_deconvolve` | FIR / event-related deconvolution without an assumed HRF shape (Glover 1999). |
| `ffs_perm` *(beta)* | GPU-batched non-parametric permutation testing (ex. for single trials or group statistics; Nichols & Holmes 2002). ~5,000 permutations / 2 minutes for a full brain. Writes AFNI-compatible corrected t-stats and attacheds cluster tables (for max cluster size correction). |


### HRF and denoising

| command | description |
|---|---|
| `ffs_hrfopt` | Per-voxel HRF library selection (as in GLMsingle; Prince et al. 2022). Tests each HRF in a library by LORO CV; refits with the winner. Canonical or PIGHS libraries. |
| `ffs_librarian` | Derive a custom HRF library from a subject's own data, for use with `ffs_hrfopt` (cf. the NSD library; Allen et al. 2022). |
| `ffs_fitbasis` | Constrained basis-set HRF fits (SPMG1/2/3, FLOBS; Woolrich, Behrens & Smith 2004). |
| `ffs_denoise` | GLMdenoise-style noise-PC denoising (Kay et al. 2013). Identifies a noise pool, extracts PCs, picks the count by LORO CV. |
| `ffs_denoisatorial` | Exhaustive 2^k subset evaluation of noise PCs, when you want the best non-contiguous combination rather than a prefix. |
| `ffs_phasereg` | Magnitude-on-phase Deming regression for macrovascular BOLD suppression (Menon 2002, Curtis 2014, Stanley 2021; phaseprep parity). |
| `ffs_nordic` | NORDIC-style patch-SVD denoising (Moeller et al. 2021). Magnitude-only or complex (mag + phase), with optional g-factor map. |
| `ffs_sauna` *(beta)*  | NORDIC-adjacent denoiser. g-factor from trailing noise volumes + Gavish–Donoho optimal singular-value shrinkage. VERY experimental, was an exploration, not vetted (but produces very similar timeseries) |

### Decomposition

| command | description |
|---|---|
| `ffs_ica` | MELODIC-style probabilistic ICA (Beckmann & Smith 2004). Auto component count, GGM mixture-model thresholding, MIGP for temporal concat, optional ICASSO and *(beta)* depth-lag/mask/nuisance component classification (on TODO list for testing). |
| `ffs_decompose` | ICA with an emphasis on stability via ICASSO clustering (Himberg et al. 2004). |
| `ffs_bsds` | Bayesian switching dynamical systems on ROI time series (dynamic brain states; Taghia 2018 / Cai 2024). |

### Image processing and registration

| command | description |
|---|---|
| `ffs_moco` | Rigid-body motion correction (Gauss–Newton, wsinc5/heptic resampling). Writes AFNI-compatible motion files. Estimation and resampling for ~300 volume 2D data is ~3 seconds, time dominated by I/O. |
| `ffs_locomoco` | Residual non-linear motion correction for single-echo EPI via optical flow (mostly the phase-encode axis, after rigid moco). Writes a per-frame warp for `ffs_nwarp`. |
| `ffs_allineate` | ~100x faster 6/9/12-parameter affine alignment with `3dAllineate`-style cost functions: `lpa`/`lpc` local Pearson (same- and cross-modal; Saad et al. 2009), `ls`, `mi`/`nmi`, Hellinger, and correlation ratio. Use `lpa` for same-contrast (e.g. anat→MNI) and `lpc` for cross-modal (EPI→anat).|
| `ffs_qwarp` | Iterative nonlinear warp estimation (`3dQwarp`-style). |
| `ffs_formwarp` | SyN nonlinear registration (ANTs-style symmetric normalization; Avants et al. 2008); an alternative backend to `ffs_qwarp`. Single-pair or per-volume timeseries; writes `ffs_nwarp`-compatible warps. |
| `ffs_nwarp` | Apply a warp to a volume or 4D timeseries. Supports complex (mag + phase) warping. Optional joint slice-timing correction (`-tpattern`, Roche 2011 space-time) folds slice timing into the same resample; tissue-following by default (`-frozen` for the slow-motion-assumption path). |
| `ffs_medic` | Multi-echo distortion correction (MEDIC; Van et al. 2026): frame-wise B0 field maps from phase, for dynamic distortion under motion. |
| `ffs_slicetime` | Slice-timing correction (`3dTshift`-style), Fourier or sinc. |
| `ffs_t2smap` | Multi-echo T2*/S0 estimation and optimal echo combination (Posse et al. 1999). |
| `ffs_motsim` | Motion-simulation nuisance regressors (Patriat, Reynolds & Birn 2017). |
| `ffs_util_automask` | Automatic brain mask from EPI. |
| `ffs_util_pcwarp` | Extract PCs from a warp-field timeseries (folder of 4D frames or a 5D file) as nuisance regressors. |
| `ffs_spatial_xcorr` | Spatial cross-correlation matrix between two 4D volumes within a mask, with optimal matching and consistency metrics. |

### Benchmarking

| command | description |
|---|---|
| `ffs_benchmark` | Run AFNI and `ffs_*` tools side by side on a BIDS dataset (default: OpenNeuro ds005165) and compare outputs and timing. `-validate-only` skips re-running and just compares. |

### Less used tools

These are either WIP CLI tools, things that I am tinkering with or ones that I just don't think are super important.

| command | description |
|---|---|
| `ffs_build_design` | Build a 1D design matrix from onsets, durations, an HRF model, polynomials, and motion. Output is readable by `ffs_reml` and AFNI tools. |
| `ffs_tps` *(beta)*  | Thin-plate-spline HRF estimation with cross-validated smoothness. Global or per-voxel λ. Further work required here |
| `ffs_xval_r2` | Cross-validated R² maps (LORO or split-half) with proper nuisance projection. |
| `ffs_pathfinder` | Joint HRF + denoising optimisation. Picks the HRF that works best with the denoising level chosen for that voxel. |
| `ffs_util_fwhm` | Whole-volume spatial smoothness of a residual dataset (`3dFWHMx`-style: classic FWHM + mixed ACF a/b/c; Cox et al. 2017). |
| `ffs_util_localstat` | Local spatial statistics over a neighborhood (`3dLocalstat` / `3dLocalACF`-style). |
| `ffs_util_3dmath` | Voxelwise math over one or more datasets (`3dcalc` / `3dMean`-style). |
| `ffs_util_updatedof` | Adjust the degrees of freedom of a stat bucket (e.g. after NORDIC) and convert t/F to z. |
| `ffs_util_convert_medic` | Convert warpkit MEDIC output into `ffs_nwarp` warps. |

## A typical pipeline

GLMsingle-style single-trial analysis:

```
ffs_hrfopt    -> per-voxel HRF
ffs_denoise   -> noise PCs + count (with voxelwise HRFs)
ffs_ridge     -> single-trial betas with per-voxel ridge (with HRFs + nuisance PCs)
```

Or a more conventional GLM:

```
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
