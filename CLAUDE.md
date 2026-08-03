# CLAUDE.md — fastfuncstuff working agreement

Project-wide guidance for Claude working in this repo. Personal preferences live in auto-memory; this file is what every contributor should expect to apply.

## What this codebase is

GPU-accelerated fMRI processing tools. Each `ffs_*` CLI is a thin wrapper over reusable primitives in `fastfuncstuff/{glm,design,processing,stats,io}`. The goal is to **match or beat established references** ([[AFNI]] / [[FSL]] / [[GLMsingle]] / [[GLMdenoise]] / [[NORDIC]]) for correctness while being substantially faster — thested with `ffs_benchmark`, and offering more features.

## GPU-first, CPU-respectable

CUDA is the primary target. Design every new algorithm so the hot path lives on the GPU, uses [[Memory module]] chunking, and avoids host↔device round-trips inside loops. **But CPU is a first-class fallback**: most users start there and many fMRI workflows live on CPU permanently. The expectation is **rough parity with the equivalent reference tool on CPU**, not "CPU is the slow path we don't care about". MPS is best-effort — float64 ops still fall back to CPU and some break entirely, so prefer `-device cpu` on Mac unless an op is known to work.

Practical rules:
- Honour the user's `-device` flag end-to-end. Never silently force CPU mid-pipeline.
- Write the CPU path even when slower, so users without a GPU aren't blocked.
- Long-running loops (voxel chunks, CV folds, per-trial saves) should show a `tqdm` bar with `leave=True` and `disable=` for trivial cases — silence on tiny workloads, visible progress on real ones, and a persistent final line that doubles as timing info.

The companion wiki at `../fmri_wiki/` (Obsidian-compatible) is the long-form knowledge base: papers, design rationale, principles, debugging stories. **Read `../fmri_wiki/index.md` first** when a question touches methodology or "why does it do X this way".

## Principles to apply

All of these are spelled out under `../fmri_wiki/principles/`. Don't re-derive them — link to the principle note. Highlights:
- [[Data Variety]] - fMRI data can span an enormous conceptual space. There could be no task (resting-state) or 1 to 100 to N tasks. The number of repeats per stimulus can vary from once to 1000s. TRs can span subsecond (0.1 even) to 3 or even 6 seconds. The number of voxels could be in the 100,000s or even millions. tSNR can go from 6 to 100s. 
- [[Code reuse]] — one flexible module everywhere; CLIs are thin (parse → validate → library → save). Before writing a new function, find the existing primitive and parametrise it.
- [[Block-diagonal nuisance]] — per-run nuisance lives in zero-padded block-diag columns. Use `glm/core.py:project_out_nuisance_per_run`; never roll your own. **Never combine `max_poly_degree=0` with block-diagonal nuisance** — rank-deficient, betas blow up.
- [[LORO cross-validation]] — held-out R² is the empirical referee. The most common bug is projecting polynomials from the *full* dataset before splitting; projection must be fold-local.
- [[Legendre polynomials]] — drift uses orthogonal polynomials via `design/matrices.py:construct_polynomial_matrix`, never raw monomials.
- [[Per-voxel optimization]] — when a hyperparameter is voxel-wise (ridge fraction, HRF index, ARMA(a,b)), fit it per voxel and save the parameter map as a diagnostic. A global value is a bad compromise everywhere.
- [[Fractional ridge]] — parametrise ridge by `frac ∈ [0, 1]` (fraction of OLS norm kept), evaluated via a single SVD pass across the grid. See `fastfuncstuff/glm/ridge.py:_fit_ridge_multiple_fracs`.
- [[Memory module]] — `fastfuncstuff/memory.py` is the single source of truth for chunk sizing. Every GPU-accelerated path must call `compute_chunk_size`; never hardcode. The 0.5 GPU safety factor compensates for PyTorch's caching allocator.
- [[Device management]] — all tensors in an operation share a device; create on-device (`torch.zeros(..., device=device)`), don't `.to()` after. For data too large for GPU, CPU stores and GPU computes via chunked streaming.
- [[Float32 vs float64]] — default float32 (One consumer GPUs several times faster, half VRAM). Promote to float64 only for the numerically sensitive step (REML likelihood, Cholesky on near-singular matrices); cast back for storage.
- [[Data formats]] — `.nii.gz` with `use_pigz=True` for outputs (parallel gzip); `.nii.zst` for big intermediates read many times.
- [[Benchmark validation]] — `ffs_benchmark` is the correctness gate. Thresholds are documented per stage. **If a stage fails, investigate before lowering the threshold.**
- **Reference AFNI** — the AFNI on `PATH` cannot read `.nii.zst` (it reports `NO-DSET`). Use the locally built tree at `~/afni_binaries/afni/src/` by full path (`~/afni_binaries/afni/src/3dinfo dset.nii.zst`); the same directory holds the AFNI **C sources** (`3dAutobox.c`, `thd_automask.c`, …) — read them when checking a port instead of guessing.
- [[VRAM debugging]] — enable with `FFS_DEBUG_VRAM=1` (or `-debug_memory` on tools that wire it). The actual-vs-predicted peak ratio tells you whether the memory model is right.

## Testing

```bash
/home/logan/miniconda3/envs/py312_movie_tasks/bin/python -m pytest tests/ -q
```

Per-module tests are colocated under `tests/`. Add a test for any new primitive — synthetic data is fine; we're verifying correctness, not benchmarking. The bar is "the test would have caught a real bug we hit", not coverage for its own sake.

## Type checking and linting

`ty` (Astral) for type checks, `ruff` for lint/format. Both fast enough to run on every change:

```bash
ty check fastfuncstuff/
ruff check fastfuncstuff/
ruff format fastfuncstuff/
```

Pre-existing warnings that aren't yours: leave them. Don't expand the diff just to silence unrelated diagnostics.

## What lives where

- `fastfuncstuff/cli/` — argparse + dispatch only. Business logic does not live here.
- `fastfuncstuff/glm/` — core linear-algebra primitives, ARMA(1,1), ridge, REML, xval.
- `fastfuncstuff/design/` — design matrices, basis sets (FLOBS/SPMG), HRF derivation, builders.
- `fastfuncstuff/processing/` — multi-step pipelines that compose primitives (denoising, motion, etc.).
- `fastfuncstuff/stats/` — spatial / temporal statistics, second-level comparisons.
- `fastfuncstuff/io/` — AFNI HEAD/BRIK + NIfTI I/O. Always go through `load_image` / `save_image`.
- `fastfuncstuff/benchmark/` — reference-comparison stages and validation report.
- `tests/` — unit tests, one file per primitive module.

## When to update the wiki vs the code

- Bug fix or feature → code change + test; the wiki only if the *rationale* needs documenting (a non-obvious design choice or a story future-you needs to remember).
- New algorithm / paper / method → a `concepts/` or `sources/` note in the wiki, then code that references it.
- A repeated debugging story or "we tried X and it failed because Y" → wiki note, not a comment in the code.

Update `../fmri_wiki/log.md` chronologically when you ingest a new source or finish a significant feature. Update `../fmri_wiki/index.md` when you add a new note.

## Conventions

- **CLI flags**: AFNI-style single-dash long flags (`-hrf-library`, `-lambda-mode`). Accept both `-foo-bar` and `-foo_bar` via argparse aliases when you can.
- **Names**: prefer descriptive (`vb_basis`) over branded (don't put external project names like "filmbabe" in identifiers; reference them in comments only).
- **Comments**: write *why*, not *what*. Identifiers tell what; comments are for hidden constraints, surprising trade-offs, and bug-of-record references. Default to no comment.
- **Commits**: imperative subject, one logical change per commit, body explains the why and any non-obvious trade-off.

## What "done" looks like

A change is done when:
1. The code change exists and is minimal.
2. A test exists if the primitive is new or the bug was preventable.
3. The benchmark still passes (or you've documented the threshold delta in the wiki).
4. The wiki has a note if the *reasoning* is non-obvious.
5. The commit message tells future-you why.
