# fastfuncstuff (ffs)

A small collection of fMRI analysis tools reimplemented in PyTorch, with a GPU-first bias.
CPU paths also receive performance work and can be faster for smaller linear-algebra workloads.
CUDA is preferred where available; see the Apple Silicon guidance below for CPU versus MPS.

## Why this exists

I have a slow CPU but I was able to get my hands on a good GPU. 
The tools I like to use, namely AFNI, are great, but they are really built to take advantage of server CPU setups.
This project started as an experiment - how well can current models port good code to a pytorch like set up? 
It has since evolved dramatically.


Python chosen not because of speed, of course, but because I can at least review the code for sanity.
It also makes installion much easier - sure, pure CUDA/C++/Rust would be faster but what a nightmare...
These are offered as-is, no guarantees.
I've tested the CLI tools below extensively on real data (my own and some from openneuro), and there is also an (in-progress) CLI tool to benchmark (`ffs_benchmark`) that compares outputs and timing against reference implementations on public datasets.

The current validation/benchmark data is in `notebooks/ffs_benchmark_showcase.ipynb` - they you can see how well ffs is able to match the reference implementations. 

## Install

```bash
git clone https://github.com/dowdlelt/fastfuncstuff.git
cd fastfuncstuff
pip install -e .          # Python >= 3.11; add ".[dev]" for tests + linters
```

See **[INSTALL.md](INSTALL.md)** for details about conda / uv,  `zstd` needed for `.nii.zst`, Bash requirements for `ffs_autoproc` scripts, FreeSurfer / AFNI / ROMEO / SPM-template expectations, and a list of the environment variable the tools read.
If something fails at install or launch time, see **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**.

### File formats

Inputs read `.nii`, `.nii.gz`, and `.nii.zst` (zstd) transparently. ZSTD is a fast option for
large files that are read many times. AFNI `BRIK/HEAD` support  also included, but not as rigourously  tested. `MGZ` support is limited, lets say minimally functional.
It needs the `zstd` / `pzstd` (preferred - gets you parallel  deflate) programs installed (see
[INSTALL.md](INSTALL.md#zstd-niizst-support)) and AFNI cannot read it, so default outputs are
`.nii.gz`, written with ISA-L when available, otherwise `pigz`.

## What I'm very happy with

### ffs_autoproc
Its like afni_proc.py - but a slightly different focus. Here, I wanted a tool that you can point at a BIDS or BIDS-like (BIDS-lite?) dataset, and it will write out the entire processing script. Sure, there are still many many options, but this gets you an entire, editable script very quick. And if you were worried about slow python, I attempt to reduce that by batching - so if 100 runs are going to go through motion correction, we loop, in python (rather than looping over ffs_moco calls). 

Still a WIP, but supports xrun and xses linear and nonlinear alignment, multiple fieldmaps per session, nonlinear motion correction, automatic event file reading, and several other features. 

Intermeidates can use nii.zst for speed, but the GLM can write out nii.gz for compatability. 

### ffs_nwarp
One of the things I am most happy with is `ffs_nwarp` which is a near drop in replacment for AFNI's `3dNwarpApply` - produces near identical output, with wsinc5 interp (though permits negative values, `-no_neg` to avoid) but about 50x faster. As the world is sometimes kind - it will work with AFNI derived `_WARP` files as well as ANTs warps. (Or calculate your own with `ffs_qwarp` or `ffs_formwarp`).


The gains particularly noticeable on my slow CPU, even when I use 10 cores. 
This for example (as of latest test), "just works" 
```shell
ffs_nwarp -interp wsinc5 \
 -master ./sswarper_output/anatQQ.sub-01.nii \
 -nwarp './sswarper_output/anatQQ.sub-01_WARP.nii ./sswarper_output/anatQQ.sub-01.aff12.1D anat_al_keep_e2a_only_mat.aff12.1D moco_sub-01_ses-01_task-localizer_run-1_bold_mat.aff12.1D' \
 -source ../sub-01/ses-01/func/sub-01_ses-01_task-localizer_run-1_bold.nii \
 -prefix ./ffs_mni_run-01.nii.gz -dxyz 3
```
What is also very fun is that this supports phase input and you can choose to warp directly (assume phase is unwrapped/interpolatable) or internally convert to real/imag, apply warps, and save phase out from that. This is useful for `ffs_phasereg`, for example. In other words, once you have magnitude processed its very easy to have the phase tag along. 

Also *fun* is that it supports slice-timing correction at warp time - that is, a full 4D interpolation - apply slice timing as part of the warp process - one single interpolation for both space and time (as in Roche 2011).

Additional **fun** is `-jac` jacobian modulation at warp time support. This is primarily to support `topup` like fieldmaps - the jacobian modulation only applies to the distortion correction fieldmap - not intermediate, time-varying, in-plane warps. Fortunately, those are small - the fieldmap is where you need it the most. 

### ffs_blipflip - GPU topup-like
`ffs_blipflip` can produce topup-like fieldmaps and warps that were taking ~15 minutes on 10 cores in about 30 seconds. Writes out field, etc. Motion estimate is off (but avaialble)- optimistically, the person didn't move between forward and reverse? A guard prevents negative jacobian (that is, a warp that would "fold" over). 

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

### ffs_locomoco
After motion correction there can still be nonlinear displacements due to changes in the B0 field from changes in head position. 
This tool can *very* quickly estimate these nonlinear shifts, either via optical flow methods, or local crosscorrelation focused on shifts in a provided phase encoding direction.
On a 5070Ti, this can be done for hundreds of volumes, at 1.6mm resoluation, in about 30 seconds. 
Writes out a 4D warp file (a 5D nii, with warps along the phase encoding direction, and the partition direction for 3-D acquisitions), that can be applied with `ffs_nwarp`.
The residual effects of this can also be reduced by using PCs from the 4D warp, extracted via `ffs_util_pcwarp`, conceptually similar to using motion parameters in the GLM. 


## Command-line tool listing

I work in the command line, and I tend to use bash scripts - so, everything is more or less built with that in mind.
Every CLI is registered as a console script and accepts `-help`. Flag style follows AFNI conventions (single dash).
Tools marked *(beta)* are working but less tested; *(experimental)* means exploratory, and may or may not be a good idea.

### Pipelines

| command | description |
|---|---|
| `ffs_autoproc` | Write a readable, resumable preprocessing script from a BIDS (or BIDS-like) dataset: moco, distortion correction (fieldmaps, blip-up/down, MEDIC), residual nonlinear motion (`ffs_locomoco`), NORDIC, alignment across runs and sessions, single-echo and multi-echo, volume and surface outputs, GLM. Runs of the same step are batched in one Python process. |

### GLM and design

| command | description |
|---|---|
| `ffs_reml` | OLS / ARMA(1,1) prewhitened GLM with REML grid search over (a, b). AFNI `3dREMLfit`-style bucket output, FDR curves attached. Accepts per-voxel HRF assignments via `-hrfopt_prefix`, and `-single_trials LABEL` rebuilds the design with one regressor per event. Can take `-spec design.spec` to compile the whole design itself. Volume or surface (`.func.gii`) input. |
| `ffs_design_spec` | Compile a single `design.spec` TOML (runs, events, per-task HRF, nuisance, contrasts with wildcards/F-tests) into an AFNI `.xmat.1D`. `stub` writes a skeleton from `events.tsv` + NIfTI headers. |
| `ffs_build_design` | Build a 1D design matrix from onsets, durations, an HRF model, polynomials, and motion. Output is readable by `ffs_reml` and AFNI tools. This is primarily a helper tool at this point. |
| `ffs_util_concalc` | Recompute (i.e. correct mistakes) or add GLM contrasts (t and F) on an existing OLS/REML stats "bucket" without rerunning the GLM — for REML, it requires and reuses the per-(a,b) ARMA inverse. |
| `ffs_ridge` | Fractional ridge regression for single-trial betas (GLMsingle Type D; Prince et al. 2022). Per-voxel optimal fraction by cross-validation. |
| `ffs_deconvolve` | FIR / event-related deconvolution without an assumed HRF shape (Glover 1999). |
| `ffs_util_eventcheck` | Check what an event timing lets a FIR/TENT estimate resolve: problems, fixes, and the finest usable knot spacing. |
| `ffs_simulate` | Single-subject design power by Monte Carlo: from a described experiment or its timing files and a noise level, the amplitude each contrast needs to be detected per tSNR, and whether autocorrelation would mislead an OLS analysis. |
| `ffs_xval_r2` | Cross-validated R² maps (LORO or split-half) with nuisance projection done per fold. Mostly legacy, many tools have an internal loop for this now |
| `ffs_pyrf` *(beta)* | Incredibly fast population receptive field (pRF) mapping with the compressive spatial summation model, following Kendrick Kay's `analyzePRF` — same super-grid seeding, staged fit, and parameter conventions. Batched Gauss-Newton with per-voxel HRF selection; reports `x`/`y`/`angle`/`ecc`/`rfsize` plus leave-one-run-out R², in degrees of visual angle when given `-screen_extent`. Includes denoising and HRF fitting. Apertures as `.npy`/NIfTI movies or PNG frame directories. **Cite [Kay et al. 2013](https://doi.org/10.1152/jn.00105.2013).** |
| `ffs_varpart` | Variance partitioning of single-trial (or sufficient multi-run) betas for fully crossed factorial designs. |

### Statistics and inference

| command | description |
|---|---|
| `ffs_perm` *(beta)* | GPU-batched non-parametric permutation testing (sign-flip / label permutation; Nichols & Holmes 2002). Writes AFNI-compatible corrected t-stats and cluster tables (max cluster size correction). |
| `ffs_clustsim` | Really really fast Monte-Carlo cluster-size thresholds (`3dClustSim`), from ACF parameters, a residual dataset, or a FWHM. AFNI or `ffs_viewer` compatible outputs |
| `ffs_util_fwhm` | Whole-volume spatial smoothness of a residual dataset (`3dFWHMx`-style: classic FWHM + mixed ACF a/b/c; Cox et al. 2017). |
| `ffs_util_localstat` | Local spatial statistics over a neighborhood (`3dLocalstat` / `3dLocalACF`-style) - but exceedingly fast. |
| `ffs_util_updatedof` | Adjust the degrees of freedom of a stat bucket (e.g. after NORDIC) and convert t/F to z (for viewing in AFNI/`ffs_viewer`). |
| `ffs_util_gmm` | Gaussian/Gamma mixture model of a statistic image (MELODIC-like mixture-model stage): fit the null, rescale to z, report P(signal). |
| `ffs_util_noiseceiling` | Per-voxel noise ceilings from repeated runs (think retinotopy or otherwise identical designs) or single-trial betas. |

### HRF and denoising

| command | description |
|---|---|
| `ffs_hrfopt` | Per-voxel HRF library selection (as in GLMsingle; Prince et al. 2022). Tests each HRF in a library by LORO CV; refits with the winner. Canonical or PIGHS libraries. No requirement for TR-locking or identical event durations. |
| `ffs_librarian` | Derive a custom HRF library from a subject's own data, for use with `ffs_hrfopt` (cf. the NSD library; Allen et al. 2022). |
| `ffs_fitbasis` | Constrained basis-set HRF fits (SPMG1/2/3, FLOBS; Woolrich, Behrens & Smith 2004). |
| `ffs_tps` *(experimental)* | Thin-plate-spline HRF estimation with cross-validated smoothness. Global or per-voxel λ. |
| `ffs_pathfinder` *(experimental)* | Joint HRF + denoising optimisation: picks the HRF that works best with the denoising level chosen for that voxel. |
| `ffs_denoise` | GLMdenoise-style noise-PC denoising (Kay et al. 2013). Identifies a noise pool, extracts PCs, picks the count by LORO CV. No requirement of TR locking or identical durations. |
| `ffs_nordic` | Fast NORDIC-style patch-SVD denoising (Moeller et al. 2021). Magnitude-only or complex (mag + phase), with optional g-factor map. Writes the number of components removed per voxel, for `ffs_util_updatedof` to give you corrected statistics, per-voxel. |
| `ffs_motsim` | Motion-simulation nuisance regressors (Patriat, Reynolds & Birn 2017). |
| `ffs_denoisatorial` *experimental* | Exhaustive 2^k subset evaluation of noise PCs, for the best non-contiguous combination rather than a prefix. |
| `ffs_sauna` *(experimental)* | NORDIC-adjacent denoiser: g-factor from trailing noise volumes + Gavish–Donoho optimal singular-value shrinkage. Not vetted, not trusted. |
| `ffs_pcpatch` *(failed experiment)* | Patch-based residual-PC projection after NORDIC. Over-removes signal; kept for the record, do not use. |

### Phase

| command | description |
|---|---|
| `ffs_phase_unwrap` | GPU phase unwrapping of a whole time series against one ROMEO-unwrapped reference frame, single- or multi-echo. Builds on ROMEO start for even more speed. Writes the field map and, for 3-D multi-echo, the partition shift/stretch tables `ffs_moco` uses. |
| `ffs_phasereg` | Magnitude-on-phase regression for macrovascular BOLD suppression (Menon 2002, Curtis 2014, Stanley 2021; phaseprep parity). Per echo, or one field regressor combined across echoes. |
| `ffs_phasedrift` | Remove per-slice (or per-volume) dynamic field fluctuation from a native-space phase series, ahead of phase regression. |
| `ffs_util_complex` | Convert magnitude/phase to real/imaginary and back. |
| `ffs_veinmask` *beta* | Vein map from the GE-EPI mean and the anatomical tissue fractions, for `ffs_phasereg -spr_mask` and QC, when there is no GRE/SWI. |
| `ffs_megre` | Multi-echo GRE anatomy: T2*, R2*, S0, the T2*-weighted echo combination (an EPI-like alignment target), and R2*·S0 vein contrast (Stanley et al. 2021). |

### Motion, distortion, and multi-echo

| command | description |
|---|---|
| `ffs_moco` | Rigid-body motion correction (Gauss–Newton, wsinc5/heptic resampling). Writes AFNI-compatible motion files. Multi-echo: echo 1 drives the estimate; for 3-D multi-echo EPI it applies the partition shift/stretch tables from the phase. |
| `ffs_locomoco` | Exceedingly quick nonlinear motion correction: the time-varying displacement left after rigid moco, from B0 changes with head position. Optical flow (default, best), local cross-correlation (*experiemntal*), or a qwarp backend. Single- or multi-echo; 3-D acquisitions solve the in-plane PE and partition axes together, with the partition shift scaled by echo time. Writes a per-frame warp for `ffs_nwarp`; multi-echo can write one shared warp (reduce ram/storage) applied per echo with `VSCALE:`. Phase input is an experimental option. |
| `ffs_medic` | Multi-echo distortion correction (uses MEDIC in part; Van et al. 2026): frame-wise B0 field maps from phase, for dynamic distortion under motion. Field estimate from warpkit, the GPU unwrapper, or a given field map; undistortion on the GPU. Writes a 5-D warp usable with `ffs_nwarp`. |
| `ffs_util_convert_medic` | Convert warpkit MEDIC output into `ffs_nwarp` warps. |
| `ffs_blipflip` | Blip-up/blip-down susceptibility distortion correction (topup-like field model; Andersson et al. 2003). Writes the field, warps, and unwarped images; a guard prevents folding (negative Jacobian). |
| `ffs_util_b0fmap` | Dual-echo GRE fieldmap → off-resonance field → EPI undistortion warp (SPM FieldMap-style; ROMEO does the unwrapping). |
| `ffs_slicetime` | Slice-timing correction (`3dTshift`-style), Fourier or sinc. |
| `ffs_t2smap` | Multi-echo T2*/S0 estimation and optimal echo combination (Posse et al. 1999). With phase, also a combined field map in Hz. |
| `ffs_util_shiftcorr` | Estimate and correct the TE-dependent partition-axis shift of multi-echo 3-D EPI/GRE by whole-volume inter-echo cross-correlation. |
| `ffs_util_outcount` | Per-TR fraction of outlier voxels (`3dToutcount`), with an optional censor file. |
| `ffs_util_pcwarp` | Extract PCs from a warp-field timeseries (folder of 4D frames or a 5D file) as nuisance regressors. |

### Registration and warps

| command | description |
|---|---|
| `ffs_allineate` | 6/9/12-parameter affine alignment with `3dAllineate`-style cost functions: `lpa`/`lpc` local Pearson (Saad et al. 2009), `ls`, `mi`/`nmi`, Hellinger, correlation ratio. `lpa` for same-contrast (e.g. anat→MNI), `lpc` for cross-modal (EPI→anat). |
| `ffs_optiwarp` | Absurdly quick optical-flow nonlinear registration: demons (Thirion 1998 / Vercauteren 2009), Lucas–Kanade, or Horn–Schunck on a dense 3D field. Diffeomorphic stepping, per-axis limits, optional `-final_qwarp` polish. |
| `ffs_qwarp` | Iterative nonlinear warp estimation (`3dQwarp`-style). |
| `ffs_formwarp` | SyN nonlinear registration (ANTs-style symmetric normalization; Avants et al. 2008). Single-pair or per-volume timeseries; writes `ffs_nwarp`-compatible warps. |
| `ffs_tunewarp` | Search nonlinear registration settings across backends and subjects, scored by measures the backend did not optimise plus a deformation-regularity gate. |
| `ffs_nwarp` | Apply a chain of warps (single or time-varying) and affines to a volume or 4D timeseries in one resample (`3dNwarpApply`-style; reads AFNI and ANTs warps). Complex (mag + phase) warping, joint slice-timing correction (`-tpattern`, Roche 2011), Jacobian modulation (`-jac`), `VSCALE:` per-axis warp scaling, and projection onto surfaces (`-surf`). |
| `ffs_bbr` | Boundary-based refinement of an EPI→anat affine: seats the anatomical WM/GM boundary on the EPI's grey/white edge (the bbregister idea). |
| `ffs_rbr` | Nonlinear boundary-based distortion correction: a smooth control-point warp that seats the WM boundary on the EPI edge everywhere. Run after `ffs_bbr`. |
| `ffs_util_warpqc` | Deformation-regularity QC for a warp (folding, Jacobian range, displacement, bending energy). |
| `ffs_util_cost` | Report every alignment cost for an image pair (`3dAllineate -allcostX`). |
| `ffs_util_resample` | Reorient and/or resample onto a new grid (`3dresample`). |
| `ffs_util_autobox` | Crop to the box around non-zero data, keeping header geometry (`3dAutobox`). |
| `ffs_util_automask` | Automatic brain mask from EPI (`3dAutomask`). |

### Anatomy and surfaces

| command | description |
|---|---|
| `ffs_segment` | Unified segmentation: joint bias-field correction, tissue classification, and TPM-driven normalisation in one EM fit, following SPM's New Segment (`spm_preproc8`). |
| `ffs_util_presurf` | Prepare an MP2RAGE UNI/INV2 pair for FreeSurfer `recon-all` (the presurfer workflow of Kashyap, on `ffs_segment`, without MATLAB/SPM). |
| `ffs_util_mp2rage_denoise` | Denoise an MP2RAGE INV1/INV2 pair jointly and rebuild UNI from the denoised pair. |
| `ffs_util_surf2layers` | Cortical depth and equivolume layers (LN2_LAYERS outputs) computed directly from FreeSurfer white/pial surfaces, on any grid. |
| `ffs_util_blursurf` | Heat-kernel smoothing along the cortical surface (SurfSmooth HEAT_07): by a FWHM, to a target FWHM, or to a tSNR. |
| `ffs_util_surfmask` | Vertex mask across runs for surface GLMs (intersection of per-run footprints, minus signal dropout). |

### Decomposition and dynamics

| command | description |
|---|---|
| `ffs_ica` | Probabilistic ICA (Beckmann & Smith 2004). Auto component count, GGM mixture-model thresholding, MIGP for temporal concat, optional ICASSO and *(beta)* component classification. |
| `ffs_decompose` | ICA with an emphasis on stability via ICASSO clustering (Himberg et al. 2004). |
| `ffs_bsds` | Bayesian switching dynamical systems on ROI time series (dynamic brain states; Taghia 2018 / Cai 2024). |
| `ffs_spatial_xcorr` | Spatial cross-correlation matrix between two 4D volumes within a mask, with optimal matching and consistency metrics. |
| `ffs_util_tcorr` | Voxel-to-voxel temporal correlation between datasets on one grid (`3dTcorrelate` for N inputs, one r-map per pair). |

### Viewing and inspection

| command | description |
|---|---|
| `ffs_viewer` | Interactive GPU viewer for volumes and FreeSurfer/GIFTI surfaces, with overlays and layered datasets. View stats, test effects of smoothing, see clusters and much much more. Some modes, inspired by Instacorr: InstaGLM, InstaICA, InstaPCA. Can edit FreeSurfer surfaces directly, view voxel data on varying surfaces, at various depths.|
| `ffs_info` | Dataset header information (`3dinfo`-style) for `.nii`, `.nii.gz`, `.nii.zst`, and AFNI HEAD/BRIK, without decompressing the image data; can draw the volume in the terminal. |
| `ffs_util_3dmath` | Voxelwise math over one or more datasets (`3dcalc` / `3dMean`-style). |

### Benchmarking and development

| command | description |
|---|---|
| `ffs_benchmark` | Run AFNI and `ffs_*` tools side by side on a BIDS dataset (default: OpenNeuro ds005165) and compare outputs and timing. `-validate-only` skips re-running and just compares. |
| `ffs_profile` | Profile one ffs command in-process (Python call stack and PyTorch operations). |
| `ffs_completion` | Generate static bash/fish/zsh command line tab completions for the `ffs_*` tools. |

## A typical pipeline for denoising

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
hrf = ffs.get_canonical_hrf(stim_duration=2.0, tr=1.5, device=device)
design = ffs.build_glm_design(onsets, hrf, n_timepoints=200, device=device)

results = ffs.fit_glm(data, design, tr=1.5, device=device)
# .betas, .r2, .tstats, .fstats, .sigma2, .meanvol

ffs.write_glm_results_nifti(
    results, output_dir="./out", prefix="sub01", condition_names=["face", "house"]
)
```

Other entry points worth knowing about: `fit_glm_arma11`, `fit_glm_hrf_library`,
`generate_fmri_noise`, `simulate_fmri_run`, and `find_optimal_designs` (Liu &
Frank efficiency). See `docs/` for details.


### PyTorch, CPU and Mac MPS

The CPU paths are first-class and honour `-device cpu,N`, `FFS_NUM_THREADS`, and
`OMP_NUM_THREADS`. On Apple Silicon you no longer have to choose: **`-device auto`
(the default) picks MPS**, and the toolbox routes the individual operations that
Metal handles badly back to the CPU for you. Two mechanisms do that:

- Ops PyTorch has not implemented for Metal fall back to the CPU automatically
  (`PYTORCH_ENABLE_MPS_FALLBACK`, set before torch is imported).
- Ops that *are* implemented but are broken or slower than Apple's CPU are listed
  in `fastfuncstuff/utils.py:_MPS_CPU_OPS` and routed off Metal explicitly.
  Regenerate that table after a torch upgrade with
  `python scripts/bench_mps_policy.py`.

Measured on an M4 Max, torch 2.14 (CPU-float32 time ÷ MPS time; >1 means MPS wins):

| Stays on Metal | | Routed to the CPU | |
|---|---|---|---|
| grid_sample 3-D | 17.5× (2.7× fwd+bwd) | QR (1000,60) | 12000× *slower* |
| rfft | 13× | cholesky (500,500) | 10× slower |
| elementwise | 3.8× | solve (500,500) | 4.3× slower |
| matmul (large) | 3.6× | SVD, pinv | ~1.5× slower |
| eigh (batched) | 3.2× | eigh (unbatched) | 1.15× slower |
| conv3d | 1.7× | batched SVD, lstsq | error in the Metal compiler |

Apple Silicon requires torch ≥ 2.14, which is where native Metal linear algebra
and native 3-D `grid_sample` (border padding and backward) landed. Auto-detect
declines MPS on an older torch, and on a build where the CPU fallback did not
arm, rather than hand you a device that dies partway through a fit.

Device choice still has a size dimension, because the CPU-float64 islands cost a
round-trip. Measured on a single-run OLS fit (10 regressors), MPS against CPU:

| voxels x TRs | winner |
|---|---|
| 20k x 300 | CPU by 17x (251 ms vs 15 ms) |
| 100k x 300 | CPU by 1.6x |
| 300k x 300 | **MPS by 1.4x** |
| 300k x 900 | **MPS by 1.5x** |

The crossover sits near ~200k voxels, which is why MPS is the default: real fMRI
volumes live above it, and when the CPU does win below it the whole fit was a few
hundred milliseconds anyway. For a small ROI or a quick interactive fit, `-device
cpu` is measurably faster.

float64 is the one hard limit: Metal has none. Numerically sensitive steps run
CPU-float64 automatically and reduction accumulators stay float32 on-device, so
results are correct either way — but `-device cpu` remains fully supported and is
the right call when you want guaranteed full float64 precision.

Nonlinear registration has its own device profile:

| Tool | Apple Silicon recommendation | Why |
|---|---|---|
| `ffs_optiwarp` | MPS for LK/HS; benchmark demons | Forward-only flow kernels work natively; LK/HS usually win on full-size volumes. |
| `ffs_formwarp` | MPS | SyN leans on 3-D `grid_sample` backward, which torch 2.14 runs natively on Metal (~2.7× the CPU fwd+bwd) instead of falling back every iteration. |
| `ffs_qwarp` | MPS | Its autograd patch optimizer uses the same backward; CUDA remains the fastest target overall. |
| `ffs_nwarp` | MPS for full-size volumes | Forward-only application, three-component composition, static frames, phase channels, and frozen temporal taps share memory-planned interpolation work. |

For `ffs_formwarp` and `ffs_qwarp`, `-device auto` therefore selects CPU on a
Mac unless CUDA is available. An explicit `-device mps` is still honored (but falls back to cpu automatically as needed).

Other compute-heavy tools have a mixed Apple Silicon profile:

| Tool | Apple Silicon recommendation | Why |
|---|---|---|
| `ffs_pyrf` | MPS for whole-brain fits; CPU for small masks | Its float32 candidate GEMMs cross over in MPS's favour on large voxel sets. |
| `ffs_nordic` | CPU | Typical complex SVD patches fall back from MPS to CPU; explicit MPS remains available for Gram-eigh-shaped workloads. |
| `ffs_ica` | CPU | Mac CPU linear algebra wins at ordinary fMRI dimensions; CUDA remains valuable on large decompositions. |
| `ffs_blipflip` | CPU | Stable Gauss–Newton CG uses float64 scalar reductions, which MPS cannot represent. |
| `ffs_segment` | CPU | Required float64 geometry and 3-D interpolation backward are not supported natively on MPS. |

`ffs_nordic`, `ffs_blipflip`, and `ffs_segment` therefore choose CPU for
`-device auto` on a Mac. `ffs_nordic -device mps` is still honored; the latter two reject
explicit MPS early with an actionable message instead of failing deep in a fit.

Apple Silicon wheels include MPS support. MPS has no float64 support, so ffs keeps
large accelerator work in float32 and performs only small precision-sensitive
operations on CPU where needed.


## Status

Active, single (agent assisted, more honestly just...agent with human sanity checks) author, very much a research codebase. 

## License

**GPL-2.0-or-later.** See `LICENSE` for the full text and `PROVENANCE.md` for attribution.

The FSL-adjacent tools — `ffs_ica`, `ffs_blipflip`, `ffs_util_gmm`, `ffs_fitbasis` — are
implemented **from the published papers only**, with no FSL source used as a reference.

## References

Primary inspirations and direct method references — most are linked from
docstrings in the relevant module:

- AFNI: Cox 1996; `3dREMLfit`, `3dDeconvolve`, `3dQwarp`, `3dTshift`,
  `3dLocalstat` tech notes.
- GLMdenoise: Kay, Rokem, Winawer, Dougherty & Wandell (2013), *Front Neurosci*.
- GLMsingle: Prince, Charest, Kurzawski et al. (2022), *eLife*.
- Fractional ridge: Rokem & Kay (2020), *GigaScience*.
- pRF / compressive spatial summation: Kay, Winawer, Mezer & Wandell (2013),
  *J Neurophysiol* 110(2):481-494 — and the `analyzePRF` MATLAB toolbox by
  Kendrick Kay (http://kendrickkay.net/analyzePRF/, © 2014, CC BY 3.0), which
  `ffs_pyrf` follows for its model, seeding, and parameter conventions. Earlier
  pRF formulation: Dumoulin & Wandell (2008), *NeuroImage*.
- Probabilistic ICA, variance normalisation, and Gaussian-Gamma mixture modelling:
  Beckmann & Smith (2004), *IEEE TMI* 23(2):137-152.
- PCA model order: Minka (2000), NIPS 13:598-604; Marchenko & Pastur (1967).
- MIGP: Smith, Hyvärinen, Varoquaux, Miller & Beckmann (2014), *NeuroImage*.
- FastICA: Hyvärinen (1999), *IEEE TNN*. ICASSO: Himberg et al. (2004), *NeuroImage*.
- Blip-up/blip-down distortion correction: Andersson, Skare & Ashburner (2003),
  *NeuroImage* 20:870-888.
- NORDIC: Moeller et al. (2021), *NeuroImage*.
- Phase regression: Menon (2002); Curtis et al. (2014); Stanley et al. (2021);
  Liem (phaseprep, 2023).
- Optimal SVHT / shrinkage: Gavish & Donoho (2014, 2017).
- Motion simulation: Patriat, Reynolds & Birn (2017).
- Design efficiency: Liu & Frank (2004); Buracas & Boynton (2002); Das et al. (2023).
