# Installing fastfuncstuff

Details on the python install, the non-Python software needed, and the environment variables.
Head back to the [README](README.md) for what the code actually does.

## Python environment

`pyproject.toml` is a plain PEP 621 spec (setuptools backend), so any of pip,
uv, or a conda env + pip will work. There is no conda-forge package; the
"conda" path means "make the env with conda, then `pip install -e .`".

Python: `>=3.11` is required, **3.13 is where it started** (latest with broad PyTorch
wheel coverage). 3.14  now preferred. 

```bash
git clone https://github.com/dowdlelt/fastfuncstuff.git
cd fastfuncstuff
```

**conda** (What I used, because I'm used to it):

```bash
conda create -n ffs python=3.14
conda activate ffs
pip install -e .
```

**pip** (untested):

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e .
```

**uv** (untested):

```bash
uv venv --python 3.14
source .venv/bin/activate
uv pip install -e .
```


`-e` editable preferred because this updates frequently.
Add `".[dev]"` instead of `.` for tests + linters. A machine that only *runs* the tools doesn't
need `[dev]` (pytest, xdist, scikit-learn, ruff). The viewer (PySide6) is installed either way.

### CPU-only Linux: skip the CUDA wheels

On Linux, torch from PyPI is the CUDA build and pulls several GB of `nvidia-*` / `cuda-*`
packages. pip can't detect a GPU (dependency markers see OS, CPU architecture and Python version,
not hardware), and the CPU build lives on PyTorch's own index, which `pyproject.toml` can't name.
On a machine without a GPU, install CPU torch **first**. The package's `torch` requirement is then
already met, and pip leaves it alone:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

uv can make this choice itself: `uv pip install -e . --torch-backend=auto` looks for an NVIDIA
driver and picks the CUDA or CPU wheel to match. macOS needs neither; its PyPI wheels have no CUDA.

## zstd (`.nii.zst` support)

`.nii.zst` is what I use by default - much faster on write and read. It is not a Python dependency:
reading and writing it shells out to the **`zstd` command-line program**, which pip does not install.

```bash
brew install zstd              # macOS
sudo apt install zstd          # Debian / Ubuntu
conda install -c conda-forge zstd
command -v zstd pzstd          # both should print a path
```

If `zstd` is missing, every `.nii.zst` read or write stops with "zstd is not installed or not
available in PATH". Good to know:

- **It must be on `PATH` where the job runs**, not only in your interactive shell: batch/SLURM
  jobs, `ffs_autoproc` scripts run from `cron` or a different login shell, and a conda env that
  you did not activate.
- **`pzstd` provides for parallel decompression, and only applies for files written by `pzstd`.** `pzstd` is
  normally installed alongside `zstd`, but not always, so check with `command -v pzstd`.
  ffs uses it automatically (threads follow `FFS_NUM_THREADS` / `-device cpu,N`). Without it,
  ffs falls back to `zstd`: files still read and write, just slower reading.
- **Files compressed by plain `zstd` (even `zstd -T8`) are single-stream and decode on one core.** This is what motivated the pzst approach - parallel compression and decompression.
- **AFNI cannot read `.nii.zst`** (it reports `NO-DSET`); convert to `.nii.gz` first, or point at
  an AFNI build that supports it.
- Python 3.14's standard-library `compression.zstd` is used only to peek at headers (for
  `ffs_info` and other header reads). Bulk reads and all writes still use the program above.
- Default outputs are `.nii.gz`, written with `pigz` when available.

## Requirements and environment

The Python tools (`ffs_reml`, `ffs_nwarp`, `ffs_nordic`, ...) need only the Python environment above.
The items below matter once you use `ffs_autoproc`, surfaces, multi-echo/phase, the viewer, or the benchmark. 
If something fails with "not found", look here first. _Nothing is bundled._

**Bash >= 4 (for `ffs_autoproc` scripts).** The generated pipelines use associative arrays.
macOS ships Bash 3.2, so `brew install bash`. The script then hands itself over to a newer Bash
if it finds one (`bash` on `PATH`, `/opt/homebrew/bin`, `/usr/local/bin`, `/opt/local/bin`, or
`$CONDA_PREFIX/bin`), and otherwise stops with an error. Each script also runs a preflight that
checks its input files and every `ffs_*` / `mri_synthstrip` tool it will call before doing any work.

**Installed with pip, nothing to do:** `warpkit` (MEDIC, with ROMEO built in), `rapidgzip`,
`numba`, `connected-components-3d`.

**External software.** 
We expect to find FreeSurfer, AFNI, and SPM (for TPMs only). All of the FreeSurfer, SUMA and AFNI rows below are conditional on the option named in the first column. 
With no `-suma` and no `-fs_subj`, a pipeline needs FreeSurfer only for `mri_synthstrip` (and only with `-anat_skull yes`), and AFNI only for the default MNI template (and not at all if you pass `-mni_template`).

| Needed for | Software | Notes |
|---|---|---|
| `-anat_skull yes` (and the `-presurf` strip) | **FreeSurfer >= 8**: `mri_synthstrip` on `PATH` | Skull-stripping a raw T1w in-script, with no `recon-all`. |
| `-suma DIR` | **A FreeSurfer subject processed with SUMA** (`@SUMA_Make_Spec_FS`) | Needs `brain.nii.gz`, `aseg.auto.nii.gz` and `*_SurfVol.nii` in the directory. We read `recon-all` output; we do not run it (`ffs_util_presurf` prepares MP2RAGE for it). |
| `-suma` subject TPM | **AFNI** on `PATH` | The one place a generated pipeline calls AFNI directly: `3dcalc`, `3dMean`, `3dAutomask` and `3dTcat` build the subject tissue template. Everything else in `ffs_autoproc` is ffs tools. |
| `-do_mni` | AFNI's `MNI152_2009_template_SSW.nii.gz` | Looked up *next to the `afni` binary on `PATH`*. Without AFNI, pass `-mni_template FILE` (sub-brick 0 must be the skull-stripped volume). |
| `-fs_subj`, `ffs_viewer -surfaces`, laminar tools | FreeSurfer subject directory | A path, or a bare name resolved in `$SUBJECTS_DIR`, then `$FREESURFER_HOME/subjects` (where `fsaverage` lives). Label names come from `$FREESURFER_HOME/FreeSurferColorLUT.txt`. |
| `ffs_megre`, `ffs_util_b0fmap` | **ROMEO** binary (from [MRItools](https://github.com/korbinian90/MRItools/releases)) | On `PATH`, or `-romeo PATH` on `ffs_megre`. `ffs_medic` does not need it: it uses `warpkit`, a pip dependency with ROMEO built in. |
| `.nii.zst` read/write | `zstd` (and `pzstd`) | **Required for `.nii.zst`**; see [zstd](#zstd-niizst-support) above. `.nii.gz` writes use `pigz` when found, else zlib. |
| `ffs_dataviz` movies | `ffmpeg` | GIFs fall back to `imageio` (extra `viz`). |
| `ffs_viewer` | PySide6 (installed by default) | `fd` is optional and makes browsing large result trees faster. |
| Benchmark (`ffs_benchmark`) | `aws` CLI; AFNI; MATLAB; FSL `melodic` | `aws` downloads the OpenNeuro datasets (`ds005165`, `ds003427`) from `s3://openneuro.org` with `--no-sign-request` into `$FFS_BENCHMARK_DATA_DIR` (default `test_data/`). AFNI (`3dvolreg`, `3dDeconvolve`, `3dREMLfit`, ...), MATLAB (NORDIC, GLMsingle) and `melodic` are the *reference* implementations. A stage whose reference is missing is skipped. |

**Templates**
- **SPM tissue-probability map** (`tpm/TPM.nii` from your SPM install). It is the group prior for
  `-anat_nonlin`, `-presurf` and `ffs_segment` when there is no `-suma`. Give it as `-spm_tpm FILE`
  / `-tpm FILE`, or `export FFS_TPM=/path/to/spm/tpm/TPM.nii`. A script that needs one and has none
  stops with "set FFS_TPM".
- **MNI152_2009 SSW template**, as in the table above.

**Hardware.** CUDA, Apple MPS (torch >= 2.14, chosen automatically on Apple Silicon) or CPU
(`-device cpu[,N]`, `-device cuda,N`). VRAM is read with `torch.cuda.mem_get_info` and RAM with
`psutil`; chunk sizes are derived from what is free, so nothing has to be set. On CPU,
`torch.compile`d kernels need a C++ compiler; if compilation fails the tool drops to eager mode
with one warning.

**Environment variables.** None are required, apart from `FFS_TPM` when a script needs a TPM.

| Variable | Effect |
|---|---|
| `FFS_NUM_THREADS` | Caps CPU threads for ffs only. Precedence: `-device cpu,N`, then `FFS_NUM_THREADS`, `OMP_NUM_THREADS`, `SLURM_CPUS_PER_TASK`, then the CPU affinity mask (physical cores). |
| `FFS_GPU_SAFETY_FACTOR`, `FFS_CPU_SAFETY_FACTOR` | Fraction of free VRAM / RAM chunking may plan for (defaults 0.5 / 0.75). If you are sharing VRAM, you should lower this! |
| `FFS_MIN_CHUNK_SIZE`, `FFS_MAX_CHUNK_SIZE_GPU`, `FFS_MAX_CHUNK_SIZE_CPU`, `FFS_DATA_THRESHOLD_GB`, `FFS_ARMA_SAFETY_FACTOR` | Hard overrides for chunk planning. |
| `FFS_DEBUG_VRAM=1` | Reports predicted vs actual peak GPU memory. |
| `FFS_LOAD_THREADS`, `FFS_GZIP_LEVEL` | Decode threads and gzip level for image I/O. |
| `FFS_TPM` | Default SPM TPM path. |
| `FREESURFER_HOME`, `SUBJECTS_DIR` | Standard FreeSurfer variables; see the table above. |
| `FFS_BENCHMARK_DATA_DIR`, `NORDIC_MATLAB_PATH`, `ROMEO_PATH` | Benchmark data location, the NORDIC MATLAB toolbox, and a ROMEO binary that is not on `PATH` (benchmark only). |
| `XDG_CACHE_HOME`, `XDG_CONFIG_HOME` | Compiled-kernel and surface caches go to `$XDG_CACHE_HOME/fastfuncstuff` (default `~/.cache`); viewer favourites to `$XDG_CONFIG_HOME/fastfuncstuff` (default `~/.config`). |
| `NO_COLOR` | Plain terminal output. |

Escape hatches for debugging (`FFS_*_NO_TRITON`, `FFS_NWARP_NO_COMPILE=1`, `FFS_*_LEGACY`) are
documented in the source and are not meant for normal use.
