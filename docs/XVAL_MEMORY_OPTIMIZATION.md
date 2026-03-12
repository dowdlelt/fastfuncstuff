# Cross-Validation Memory Optimization

## Problem

Loading all voxels to GPU at once causes OOM. A typical dataset
(900k voxels x 3000 TRs x 4 bytes = ~10.8 GB) plus intermediate
copies during CV easily exceeds GPU memory.


## Solution: Two-Level Batching

### Strategy 1: GPU-Resident Data (default)

When the full dataset fits on GPU:
- Load data to GPU once (~11 GB)
- Slice by runs using views (zero-copy on GPU)
- Precompute projection matrices once per CV split (~16 MB each)
- Project and fit in voxel batches (default 5000 voxels)

Peak GPU memory: data + ~200 MB working memory.

### Strategy 2: Chunked Data Loading

When data does not fit on GPU (use `-data_chunk_size`):
- Process voxels in chunks (e.g., 100k at a time)
- Each chunk is loaded to GPU, runs all CV splits, then freed
- Within each chunk, same batched projection as Strategy 1

Peak GPU memory: ~1.5 GB regardless of dataset size.


## Key Optimizations

**GPU slicing is free.** `data[:, run_indices]` creates a view, not a copy.
Exploit this by keeping data on GPU and slicing per CV split.

**Precompute projection matrices.** The nuisance projection matrix
P = I - Q @ Q^T (from QR decomposition) is tiny (n_timepoints x n_timepoints)
and reused across all voxel batches within a CV split.

**Batch only the large operation.** The projection `batch - (P @ batch.T).T`
is the memory bottleneck. Batching this at 5000 voxels keeps temporaries
around 40 MB.


## Device Assignment

| Operation | Device | Rationale |
|-----------|--------|-----------|
| Data storage | CPU or GPU | depends on dataset size |
| Run slicing | same as data | views are free |
| Projection matrix | GPU | small, reused across batches |
| Projection (batched) | GPU | fast matmul |
| OLS fitting | GPU | matrix operations |
| Result accumulation | CPU | grows with n_voxels |


## Tuning

- `batch_size`: default 5000 voxels. Reduce if GPU memory is tight.
- `data_chunk_size`: set to enable Strategy 2 for very large datasets.
- `float16` data: halves memory footprint if precision is acceptable.
