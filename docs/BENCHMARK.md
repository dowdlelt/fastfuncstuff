# Benchmark: AFNI vs FFS Validation and Timing

## Overview

The benchmark framework (`ffs_benchmark`) compares FFS tools against AFNI/FSL equivalents on real neuroimaging data (OpenNeuro ds005165, subject sub-01). It serves two purposes:

1. **Accuracy validation** — spatial/temporal correlations between AFNI and FFS outputs confirm algorithmic correctness
2. **Performance timing** — wall-clock comparisons per stage, cached per hardware architecture

This is the ultimate integration test: end-to-end processing through every major FFS tool, validated against established ground truth.

## Data

- **Dataset**: [OpenNeuro ds005165](https://openneuro.org/datasets/ds005165), sub-01/ses-01
- **Tasks**: `localizer` (5 runs), `rest` (5 runs)
- **Anatomical**: T1w for skull stripping and MNI normalization
- **Location**: `test_data/ds005165-download/` (gitignored)
- **Download**: `aws s3 sync --no-sign-request s3://openneuro.org/ds005165 ds005165-download/ --exclude "*" --include "sub-01/ses-01/*"`

All derived outputs live in `test_data/ds005165-download/processing/`.

## Architecture

```
fastfuncstuff/benchmark/
    __init__.py
    arch.py            # Hardware fingerprinting (GPU, CPU, OS, OMP_NUM_THREADS)
    runner.py          # BenchmarkContext, run_timed(), stage orchestration
    validation.py      # compare_volumes, compare_1d, compare_timeseries_4d, etc.
    timing_cache.py    # JSON cache keyed by architecture fingerprint
    reporting.py       # Terminal tables, JSON output
    stages/
        __init__.py    # Stage registry
        moco.py        # Motion correction: 3dvolreg vs ffs_moco
        slicetime.py   # Slice timing: 3dTshift vs ffs_slicetime
        align.py       # Alignment: sswarper2 vs ffs_allineate + ffs_qwarp
        warp.py        # Warp apply: 3dNwarpApply vs ffs_nwarp
        glm.py         # GLM: 3dDeconvolve + 3dREMLfit vs ffs_reml
        ica.py         # ICA: melodic vs ffs_ica -temp_concat

fastfuncstuff/cli/benchmark.py   # CLI entry point (ffs_benchmark)
tests/test_benchmark.py          # pytest integration wrapper
```

## Stage Protocol

Every stage module exposes:

```python
name = "stage_name"                              # used in CLI --stages and reports
description = "Human-readable description"

def check_prerequisites(ctx) -> list[str]:       # missing files (empty = ready)
def run_afni(ctx) -> float:                      # run AFNI tools, return seconds
def run_ffs(ctx) -> float:                       # run FFS tools, return seconds
def validate(ctx) -> dict:                       # compare outputs, return metrics + pass/fail
```

- `check_prerequisites` checks for input data (execution mode) or output files (validate-only mode)
- `run_afni` and `run_ffs` call CLI tools via `subprocess` with `time.monotonic()` timing
- `validate` compares outputs and returns `{"passed": bool, "summary": str, ...}`
- All commands run via `run_timed()` which captures wall-clock time and raises on non-zero exit

### Adding a New Stage

1. Create `fastfuncstuff/benchmark/stages/your_stage.py`
2. Implement `name`, `check_prerequisites`, `run_afni`, `run_ffs`, `validate`
3. Add `from . import your_stage` to `stages/__init__.py` and append to `ALL_STAGES`
4. Choose thresholds based on what the algorithms should agree on (see Threshold Logic below)

Minimal template:

```python
"""Your stage benchmark: AFNI_tool vs ffs_tool."""

from __future__ import annotations
from pathlib import Path
from ..runner import BenchmarkContext, run_timed
from ..validation import compare_volumes  # or whichever comparison

name = "your_stage"
description = "Your stage (AFNI_tool vs ffs_tool)"

THRESHOLDS = {
    "metric_name": 0.95,
}

def check_prerequisites(ctx: BenchmarkContext) -> list[str]:
    missing = []
    if ctx.validate_only:
        # Check that both AFNI and FFS outputs exist
        ...
    else:
        # Check that raw inputs exist
        ...
    return missing

def run_afni(ctx: BenchmarkContext) -> float:
    total = 0.0
    # For each output:
    out = ctx.processing_dir / "afni_output.nii.gz"
    if out.exists() and not ctx.force_afni:
        return 0.0
    elapsed, _ = run_timed("afni_command ...", label="afni_tool", cwd=ctx.processing_dir)
    total += elapsed
    return total

def run_ffs(ctx: BenchmarkContext) -> float:
    total = 0.0
    out = ctx.processing_dir / "ffs_output.nii.gz"
    if out.exists() and not ctx.force_ffs:
        return 0.0
    elapsed, _ = run_timed("ffs_tool ...", label="ffs_tool", cwd=ctx.processing_dir)
    total += elapsed
    return total

def validate(ctx: BenchmarkContext) -> dict:
    result = compare_volumes(
        ctx.processing_dir / "afni_output.nii.gz",
        ctx.processing_dir / "ffs_output.nii.gz",
    )
    passed = result["r"] >= THRESHOLDS["metric_name"]
    return {"passed": passed, "summary": f"r={result['r']:.4f}", **result}
```

## Validation Functions

All in `benchmark/validation.py`, wrapping `fastfuncstuff.stats.spatial`:

| Function | What it compares | Returns |
|----------|-----------------|---------|
| `compare_volumes(a, b)` | Two 3D volumes (spatial Pearson r) | `{"r", "n_voxels"}` |
| `compare_timeseries_4d(a, b)` | Two 4D volumes (voxelwise temporal r) | `{"median_r", "frac_above_0.95", ...}` |
| `compare_1d_params(a, b)` | Two `.1D` parameter files (per-column r) | `{"per_column_r", "mean_r", "min_r"}` |
| `compare_moco_ssd(a, b)` | Two 4D motion-corrected volumes (MSD) | `{"mean_msd", "nrmsd", ...}` |
| `compare_ica_components(a, b)` | Two 4D IC maps (optimal matching, abs corr) | `{"mean_matched_r", "coverage_0.5", ...}` |
| `compare_bucket_volumes(a, b)` | Two GLM bucket files (per-sub-brick r) | `{"per_brick", "mean_r", "min_r"}` |

### NIfTI Axis Conventions

nibabel loads 4D NIfTI as `(x, y, z, t)` but FFS spatial functions expect `(t, x, y, z)`. The validation functions handle this with `permute(3, 0, 1, 2)`. AFNI bucket files sometimes have shape `(x, y, z, 1, n)` — `np.squeeze()` removes the singleton.

### Masking

When no explicit mask is provided, `_automask` thresholds at 10% of the 98th percentile. This is conservative and handles both functional and anatomical volumes. For ICA comparisons, the melodic mask is used explicitly.

## Threshold Logic

Thresholds reflect expected algorithmic agreement, not arbitrary numbers:

| Stage | Metric | Threshold | Rationale |
|-------|--------|-----------|-----------|
| moco | mean image r | 0.98 | Different optimizers converge to same alignment |
| moco | motion param r | 0.85 (diagnostic) | Sub-voxel paths differ, alignment quality matters |
| slicetime | voxelwise temporal r | 0.98 | Interpolation kernels differ slightly |
| align | warped anat r | 0.80 | Entirely different nonlinear warping algorithms |
| warp | warped func r | 0.95 | Same warps applied by different interpolators |
| glm (OLS) | beta map r | 0.99 | Same math, minor float precision differences |
| glm (REML) | beta map r | 0.95 | Different ARMA implementations, more divergence expected |
| ica | mean matched \|r\| | 0.70 | ICA is stochastic; masking and component count differences |

**When to adjust thresholds**: If a stage fails, first check whether the FFS implementation has a bug. If the outputs are genuinely correct but algorithms legitimately disagree (e.g., different warping algorithms), lower the threshold. Document the reason in the stage's `THRESHOLDS` dict comment.

**Diagnostic vs pass/fail metrics**: Some metrics (like per-column motion parameters) are reported but don't affect pass/fail. Motion parameters can differ in sub-voxel regimes while producing identical aligned images. Use pass/fail for output quality metrics, diagnostic for intermediate quantities.

## Timing Cache

Results are cached in `{data_dir}/benchmark_cache.json`, keyed by architecture fingerprint:

```json
{
  "schema_version": 1,
  "runs": [{
    "arch_id": "linux-x86_64-cuda-NVIDIA_GeForce_RTX_5070_Ti",
    "timestamp": "2026-03-26T20:08:09Z",
    "architecture": {"gpu": "...", "torch": "2.8.0", "omp_num_threads": "10", ...},
    "stages": {
      "moco": {"afni_seconds": 193.1, "ffs_seconds": 212.7},
      "slicetime": {"afni_seconds": 5.9, "ffs_seconds": 5.2}
    }
  }]
}
```

- Skipped stages (output already existed) are not cached (0.0 timing is ignored)
- `--report` pulls from cache to show timing even in validate-only mode
- Each architecture gets one entry, updated on re-run

## CLI Usage

```bash
# Validate existing outputs (fast, no re-computation)
ffs_benchmark --validate-only

# Full run: execute AFNI + FFS, then validate
ffs_benchmark --force-all

# Single stage, re-run FFS only
ffs_benchmark --stages moco --force-ffs

# Show timing table (uses cached data if available)
ffs_benchmark --validate-only --report

# Save JSON results
ffs_benchmark --validate-only --json results.json

# Multiple stages
ffs_benchmark --stages moco,slicetime,glm --force-ffs --report

# Generate plots from current run
ffs_benchmark --force-all --report --plot benchmark_plots/

# Generate plots from committed cache files (multiple architectures)
ffs_benchmark --plot-from-cache \
    results/rtx5070ti_cache.json \
    results/m4pro_cache.json \
    --plot benchmark_plots/
```

## pytest Integration

`tests/test_benchmark.py` provides a thin pytest wrapper:

```bash
# Run validation tests (requires existing outputs, ~60s)
pytest -m benchmark_validation tests/test_benchmark.py -v

# Run full execution tests (downloads data, runs tools, ~30min)
pytest -m benchmark_full tests/test_benchmark.py -v
```

The validation tests exercise all comparison utilities and by extension the core `stats.spatial` module, contributing to code coverage without requiring AFNI/FSL to be installed.

## Processing Pipeline

The stages form a dependency chain reflecting a standard fMRI pipeline:

```
raw data
  ├── moco (motion correction)
  │     └── align (anatomical registration)
  │           └── warp (MNI normalization)
  │                 ├── glm (statistical modeling)
  │                 └── ica (decomposition)
  └── slicetime (independent, single run)
```

All stages use **shared AFNI inputs** where possible: e.g., both AFNI and FFS warping use the same AFNI-generated warp fields and alignment matrices. This isolates each stage to test only its specific algorithm.

## OMP_NUM_THREADS

AFNI parallelism is controlled by `OMP_NUM_THREADS`. This is logged in the architecture fingerprint. For fair comparisons:

- Set `OMP_NUM_THREADS` explicitly before benchmarking
- AFNI's `3dDeconvolve` uses `-jobs N` (defaults to 10 in the benchmark)
- FFS tools use PyTorch threading for CPU ops and CUDA for GPU ops

## Multiple Datasets

The framework supports multiple OpenNeuro datasets. Each dataset gets its own `benchmark_cache.json` and the `dataset_id` is auto-detected from the directory name (e.g., `ds005165-download` -> `ds005165`).

To add a new dataset:

1. Download it into `test_data/<dataset_id>-download/`
2. Create the `processing/` directory and run AFNI + FFS tools (or add `run_afni`/`run_ffs` to stages)
3. Stages that need dataset-specific paths (e.g., different subjects, tasks, runs) should parameterize via `ctx.data_dir`
4. The plotting system automatically labels bars with `(dataset_id)` when multiple datasets are present

When plotting from multiple cache files, the merge logic combines all architecture+dataset combinations into a single chart. This lets you compare:
- Same dataset, different hardware (RTX 5070 Ti vs M4 Pro on ds005165)
- Same hardware, different datasets (ds005165 vs ds003456 on RTX 5070 Ti)
- Or any combination

## Key Design Decisions

1. **Not pytest by default**: Benchmark downloads GB of data and takes 30+ min. It's a CLI tool, not a unit test. The pytest wrapper is for CI integration with pre-existing outputs.

2. **Validate-only mode**: Most useful day-to-day. Run once to generate outputs, then validate repeatedly as FFS code changes.

3. **Architecture fingerprinting**: Timing results are meaningless without hardware context. The cache key includes GPU model, CPU arch, OS, and OMP_NUM_THREADS.

4. **Shared inputs**: AFNI preprocessing outputs feed both AFNI and FFS tools. This means a stage comparison is purely algorithmic — no upstream differences propagate.

5. **Graceful degradation**: If AFNI or melodic aren't installed, execution skips those tools. Validation still works if outputs exist from a previous run.

## Plots

Two charts are generated:

1. **Timing bars** (`benchmark_timing.png`): Grouped vertical bars — AFNI vs FFS per stage, color-coded by architecture. Shows absolute wall-clock time. Useful for seeing which stages dominate total runtime.

2. **Speedup chart** (`benchmark_speedup.png`): Horizontal bars — FFS speedup factor per stage, one bar per architecture. Red dashed line at 1.0x marks parity. Bars to the right = FFS is faster.

When multiple architectures or datasets are present, bars are grouped and labeled automatically (e.g., "RTX 5070 Ti (ds005165)", "M4 Pro (ds005165)").

### Committing Benchmark Results

The intended workflow for tracking performance across machines:

```
benchmark_results/
    rtx5070ti_ds005165_cache.json   # committed
    m4pro_ds005165_cache.json       # committed
    benchmark_timing.png            # generated from caches
    benchmark_speedup.png           # generated from caches
```

```bash
# After running on a new machine, copy the cache
cp test_data/ds005165-download/benchmark_cache.json \
   benchmark_results/<machine>_ds005165_cache.json

# Regenerate combined plots
ffs_benchmark --plot-from-cache benchmark_results/*_cache.json \
    --plot benchmark_results/
```
