# FastFuncStuff Design Principles

Core engineering philosophy and non-negotiable patterns for the fastfuncstuff codebase.

---

## GLM and Nuisance Modeling

**The block-diagonal principle.** When modeling multiple runs together, nuisance regressors (drift polynomials, motion, noise PCs) must be run-specific. They are concatenated as block-diagonal columns — each block covers exactly one run's timepoints, with zeros elsewhere. This guarantees that run-specific nuisance only absorbs variance within that run, not across runs.

```
Full design (2 runs, 3 poly terms each):
[  task  | poly_run0 |     0     ]
[  task  |     0     | poly_run1 ]
          ^---block-diagonal---^
```

**Legendre polynomials, not monomials.** Drift regressors must use orthogonal Legendre polynomials P_k(t) over each run's time axis. Raw monomials (1, t, t², ...) are numerically ill-conditioned: they become collinear at higher degrees and cause near-singular design matrices. Legendre polynomials are orthogonal by construction, so adding higher degrees does not destabilize lower-degree estimates.

**`project_out_nuisance_per_run()` is the canonical nuisance removal function.** It projects nuisance from both data and design consistently, building the correct block-diagonal structure per run from local `run_starts` indices. Use it for any nuisance type: drift polynomials, motion regressors, noise PCs. Never roll your own nuisance projection.

**Single-run extraction: strip zero-padded columns.** When you subset a single run's data out of a concatenated design, remove the zero columns belonging to other runs — do not pass a design with trailing blocks of zeros into a GLM. Use local `run_starts` computed for the subset, not global indices.

**`max_poly_degree=0` is only valid without nuisance columns.** If you have a block-diagonal nuisance design and set `max_poly_degree=0`, you add no polynomial columns at all. This is fine for pre-cleaned data. But if you pass a block-diagonal design that already encodes run structure and then add zero-degree polynomials, you risk rank deficiency (columns of all ones per run collapse into the intercept). See CLAUDE.md for the exact constraint.

---

## Cross-Validation

**The LORO pattern.** Leave-one-run-out (LORO) cross-validation is the standard CV scheme for fMRI: train on all runs except one, predict the held-out run, aggregate R² across folds. This is preferred over split-half because it uses more data per fold and gives unbiased out-of-sample estimates.

**Polynomial projection must be fold-local.** The single most common CV implementation error is projecting polynomials from the full dataset and then splitting. Polynomials from all runs together do not form a valid block-diagonal — they bleed across run boundaries. The correct pattern:

1. Subset train runs → build local `run_starts` for train subset → project nuisance from train data AND train design
2. Subset test run(s) → build local `run_starts` for test subset → project nuisance from test data AND test design
3. Fit betas on clean train; predict on clean test; compare to clean test data

**Treat train and test identically.** Whatever nuisance is projected from train data must also be projected from test data (using test-local projectors). Asymmetric treatment inflates or deflates CV R² because the signal space differs between prediction and ground truth.

**Polynomials are not signal.** The task variance that correlates with low-frequency drift is negligible. Projecting polynomials from both sides loses a tiny amount of true signal in exchange for a numerically clean, unbiased estimate. This is the correct trade-off.

---

## Ridge Regression

**Fractional ridge over arbitrary alpha.** Rather than searching over an unbounded alpha grid, FFS uses fractions of lambda_max (the smallest regularization that shrinks all betas to zero). `frac ∈ [0, 1]` is a bounded, interpretable parameter:
- `frac=0`: pure OLS
- `frac=1`: maximum sensible regularization

This makes hyperparameter search stable and results comparable across datasets and voxels.

**Per-voxel optimization via SVD.** Fit all ridge fractions simultaneously using the SVD of the design matrix (compute once, evaluate all fractions via singular value scaling). Select the optimal fraction per voxel based on CV R². Save the per-voxel optimal fraction map — it is a diagnostic of where regularization is needed.

**Two R² values to save.** Initial R² is the CV performance at minimal ridge (frac ≈ 0.05), representing the OLS-like generalization baseline. Final R² is the in-sample performance at the voxel-optimal ridge fraction. Both together tell you whether regularization helped and where.

---

## PyTorch Device Management

**All tensors in an operation must share a device.** The most common silent bug is a CPU design matrix multiplied against a GPU data tensor — PyTorch raises a device mismatch error. Always place all inputs on the same device before any computation.

**Create tensors on-device, not on CPU then moved.** Prefer `torch.zeros(n, m, device=device)` over `torch.zeros(n, m).to(device)`. The latter allocates on CPU first, then copies — wasteful for large arrays.

**CPU as storage, GPU as compute.** For datasets too large to fit in GPU memory, keep the full array on CPU and stream voxel chunks to GPU. Pre-compute small matrices (design, polynomials, projection matrices) on GPU once; stream large arrays (voxel timeseries) in chunks. Accumulate final results on CPU.

**GPU accumulators for streaming statistics.** Running sums, sum-of-squares, and LORO partial sums have negligible memory footprint. Accumulate on GPU to avoid repeated CPU↔GPU transfers.

---

## Benchmark

The benchmark (`ffs_benchmark`) is the ground truth for correctness and performance. See `docs/BENCHMARK.md` for full architecture and CLI usage. Key principles:

**Purpose**: Every major FFS tool must be verifiable against an established reference (AFNI/MELODIC/GLMsingle, etc). We want to go beyond these tools, eventually, but must conform to the best in class resources, at minimum. 

**Stage interface**: Every stage implements `name`, `check_prerequisites()`, `run_ref()`, `run_ffs()`, `validate()`. The `validate()` function returns `{"passed": bool, "summary": str, ...}` and must define explicit numeric thresholds. See `docs/BENCHMARK.md` for the minimal template.

**Threshold rationale**: Thresholds encode the expected agreement between algorithms:
- OLS/linear operations: ≥ 0.99 spatial r (same math, only float precision differences)
- REML/iterative solvers: ≥ 0.90–0.95 (ARMA estimation paths differ slightly)
- Deconvolution (TENT): ≥ 0.95 temporal median r + ≥ 0.95 spatial r at midpoint
- ICA: ≥ 0.70 mean matched |r| (stochastic, ordering differs)

Always document in the `THRESHOLDS` dict why a threshold is set where it is. If a stage fails, investigate whether it is a bug before lowering the threshold.

**Timing is architecture-specific**: The timing cache keys on GPU model + CPU arch + OS + OMP_NUM_THREADS. Never compare timing across different machines without that context.

**Never run stages in parallel**: Benchmark stages share file system state and produce timing data. Parallelizing stages corrupts both.

**CLI → Python callable**: Every CLI tool exposes `main(argv: list[str] | None = None)` that accepts a list of argument strings. This enables calling any CLI from Python (notebooks, pipelines, benchmark stages) without subprocess:

```python
from fastfuncstuff.cli.moco import main as ffs_moco
ffs_moco(["-input", "epi.nii.gz", "-prefix", "epi_mc.nii.gz"])
```

Not all CLIs have this signature yet — those that do accept `argv` can be used this way. For those that call `sys.argv` directly, use `run_timed()` in benchmark stages (subprocess). The pattern to adopt when adding new CLIs is `def main(argv: list[str] | None = None)`.

---

## Precision (float32 / float64)

**Default: float32.** It is 2× faster on GPU, uses half the VRAM, and is sufficient for many neuroimaging operations. 

**Numerical tricks can improve float32 precision.** While it often is associated with low accuracy, we use numerical tricks (scaling, etc) in order to achieve near parity with float64 operations in some cases.

**float64 only when algorithmically necessary.** The canonical example is `ffs_reml`: ARMA(1,1) parameter estimation involves iterative log-likelihood maximization where float32 accumulation errors cause the optimizer to converge to wrong (a, b) pairs. The solution is a hybrid approach:

1. Load data as float32 (I/O speed, memory)
2. Cast to float64 only for the ARMA likelihood surface computation
3. Cast back to float32 for storage and downstream GLM (betas, t-stats)

The user controls this with `-use_double` when they need exact AFNI parity. The default should always be float32 with precision applied surgically where it matters.

**Memory accounting**: float64 doubles VRAM requirements. The `memory.py` module has a `double_precision_multiplier = 2.0` that must be factored into chunk size calculations when float64 is in use. See `MemoryConfig.double_precision_multiplier`.

**Never silently downcast.** If a user requests float64 mode, respect it throughout. If a computation genuinely requires float64 for numerical stability, document it in a comment — do not let it silently fail with float32 and produce wrong answers.

---

## Data Formats

**NIfTI with pigz for speed.** Standard `.nii.gz` files are the default format, but gzip compression is single-threaded and slow for large 4D volumes. Use `pigz` (parallel gzip) when available:

```python
# processing/io.py: save_image(..., use_pigz=True)
# Uses pigz if installed, falls back to gzip silently
```

All FFS save functions that write `.nii.gz` accept `use_pigz=True`. This should be the default for any function that writes intermediate files.

**Zstandard (`.nii.zst`) as alternative, faster compression.** The IO layer (`io/afni.py`) supports `.nii.zst` via the `zstd` command-line tool. Zstd is 5–10× faster to decompress than gzip at similar compression ratios, making it better for data that will be read many times. The load path transparently decompresses to a temp file, so callers don't need to handle it explicitly.

**Format support hierarchy**:
- `.nii.gz` — default for outputs (widest compatibility)
- `.nii.zst` — preferred for large intermediate files (fast I/O)
- `.nii` — uncompressed, only for very small files or debugging
- AFNI `.BRIK.gz` / `.HEAD` — input only (reading AFNI outputs for benchmark comparison)


---

## Code Architecture: Reuse

**The central rule: one flexible module, used everywhere.** Before writing a new function, ask whether an existing function can be parameterized to cover the new case. Neuroimaging computations have very few unique primitives:

- Linear algebra: `glm/core.py` — `fit_glm`, `fit_glm_chunk`, `_loro_r2_per_voxel`
- Design matrices: `design/matrices.py` — `build_design_matrix`, `construct_polynomial_matrix`
- Nuisance projection: `glm/core.py` — `project_out_nuisance_per_run`
- Spatial stats: `stats/spatial.py` — `spatial_correlation`, `consistency_report`
- Memory/chunking: `memory.py` — `compute_chunk_size`, `MemoryConfig`
- I/O: `io/afni.py`, `processing/io.py` — `load_nifti`, `save_nifti`, `load_image`, `save_image`

**Don't copy-paste GPU chunks.** The chunking pattern (`for start in range(0, n_vox, chunk_size)`) should call `fit_glm_chunk` or an equivalent shared function — never re-implement the inner loop. The pre-factored Cholesky optimization (`cholesky_L` parameter in `fit_glm_chunk`) is already there; use it.

**CLIs are thin wrappers.** CLI files in `cli/` parse arguments and call library functions. They should contain minimal logic. Business logic belongs in `glm/`, `design/`, `processing/`, etc. A CLI function should be: parse → validate → call library → save outputs.

**Benchmark stages are also thin wrappers.** They call CLI tools via subprocess (for reference tools like AFNI) or `run_timed()`. They do not reimplement validation logic — they call `validation.py` functions.

**Shared validation helpers.** All output comparison logic lives in `benchmark/validation.py`. If two stages compare similar things (e.g., both compare bucket files), they call the same `compare_im_bucket()` or `compare_bucket_volumes()` function. Do not write per-stage comparison code.

---

## Memory Module

`fastfuncstuff/memory.py` is the single source of truth for memory estimation and chunking. All GPU-accelerated modules must use it — never hardcode chunk sizes or GB estimates.

**Core API**:
```python
from fastfuncstuff.memory import compute_chunk_size, MemoryConfig, get_memory_config

# Get chunk size for a specific operation
chunk_size = compute_chunk_size(
    n_voxels=n_voxels,
    n_timepoints=n_timepoints,
    n_regressors=n_regressors,
    device=device,
    operation="glm",         # "glm", "arma", "ica", "ridge", ...
    dtype=torch.float32,
)
```

**Device-aware behavior**:
- GPU: queries `torch.cuda.mem_get_info()` for available VRAM, applies `gpu_safety_factor` (default 0.5)
- CPU: queries `psutil.virtual_memory()` for available RAM, applies `cpu_safety_factor` (default 0.75)
- MPS (Apple Silicon): treated like GPU with conservative factor

**Safety factors exist for a reason.** PyTorch caches allocations; `mem_get_info()` reports free memory but PyTorch's allocator may not release until forced. The 0.5 safety factor accounts for this. Do not increase it speculatively.

**Per-operation memory models.** Different operations have different intermediate array profiles. ARMA has higher peak memory (Cholesky decomposition of (n_tp × n_tp) matrices); ICA needs (n_comp × n_vox) intermediates. Each operation should have a registered model in `memory.py` rather than ad-hoc estimates.

**Double precision multiplier.** When float64 is in use, multiply all memory estimates by `MemoryConfig.double_precision_multiplier` (default 2.0). This is done automatically if you pass `dtype=torch.float64` to `compute_chunk_size`.

**The memory module is how FFS stays portable.** A chunk size that works on an RTX 5070 Ti (16GB VRAM) will not work on a laptop with 4GB VRAM. By querying available memory at runtime and applying conservative factors, the same code runs correctly on any hardware — it just goes faster on bigger GPUs.

**Never hardcode chunk sizes.** A fixed `chunk_size=60000` ignores both the timepoint dimension and available hardware. 60,000 voxels with 800 timepoints is ~10× the memory of 60,000 voxels with 80 timepoints. Always call `estimate_chunk_size()` and pass `max_chunk_size=n_voxels` so the formula runs uncapped — the safety factor provides the headroom.

**No ad-hoc reductions on top of the model.** A pattern like `chunk_size = estimate_chunk_size(...) // 3` double-counts what the per-voxel formula already includes. `bytes_per_voxel_glm` uses `5 × n_timepoints` to cover data + betas + residuals + predictions + intermediates. Adding a further divisor makes the estimate wrong in the opposite direction (too conservative), wastes VRAM, and creates unnecessary chunks.

---

## VRAM Debugging

**`VRAMDebugger` and `make_vram_debugger()` in `memory.py`.** A background sampling thread that calls `torch.cuda.memory_allocated()` at 25ms intervals during a chunk loop, then reports predicted vs. actual peak VRAM at the end.

**Why `memory_allocated()` and not `mem_get_info()`.** `memory_allocated()` reads PyTorch's internal C++ allocator counter — a single atomic read, no CUDA driver call, no GPU synchronization. Overhead is microseconds. `mem_get_info()` calls `cudaMemGetInfo()` which may synchronize the GPU; never use it inside a hot loop.

**The ratio is the signal.** The report prints `ratio = actual_peak / predicted`. Interpreting it:
- `< 0.5`: model over-predicts by more than 2× — chunk size could be doubled or more
- `0.5–0.8`: conservative — safely under-utilized, reasonable headroom
- `0.8–1.05`: accurate — model matched reality well
- `> 1.25`: model under-predicted — OOM risk on tighter hardware; investigate the memory model

**Activate without code changes.** Set `FFS_DEBUG_VRAM=1` in the environment to enable for any CLI. Or pass `-debug_memory` to: `ffs_reml`, `ffs_deconvolve`, `ffs_moco`.

**Where it is wired in.** The debugger wraps the outermost chunk loops:
- `glm/core.py`: OLS voxel chunk loop in `fit_glm()` (operation `ols_fit`)
- `glm/arma.py`: ARMA grid search loop (operation `arma_grid_search`) and GLS sub-batch loop for first (a,b) group (operation `gls_fitting`)
- `processing/ffs_moco.py`: registration loop pass 1 (operation `moco_registration`) and resampling loop pass 2 (operation `moco_resample`)

**Adding it to a new loop.** The pattern is always:
```python
_dbg = make_vram_debugger(
    device,
    chunk_size * bytes_per_voxel_OPERATION(n_timepoints, n_regressors),
    operation="my_operation",
    chunk_size=chunk_size,
    enabled=debug_memory,
)
_dbg.__enter__()
for chunk in ...:
    ...
_dbg.__exit__(None, None, None)
```
When `debug_memory=False` (or device is not CUDA), `make_vram_debugger` returns `contextlib.nullcontext()` — zero overhead.
