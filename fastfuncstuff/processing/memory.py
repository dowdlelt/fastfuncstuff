"""GPU memory estimation and 4D chunking utilities.

Estimates GPU memory usage for qwarp operations and determines optimal
chunking strategies for 4D datasets.
"""

from __future__ import annotations


def estimate_gpu_memory_bytes(
    nx: int,
    ny: int,
    nz: int,
    n_basis_max: int = 30,
) -> dict[str, int]:
    """Estimate GPU memory usage for a qwarp run on a single 3D volume.

    Returns a breakdown of memory usage by category. All values in bytes.

    Args:
        nx, ny, nz: Volume dimensions.
        n_basis_max: Maximum number of basis functions (quintic_lite=10, quintic=27).

    Returns:
        Dict with memory breakdown and total.
    """
    n_voxels = nx * ny * nz
    float32 = 4  # bytes per float32

    usage = {}

    # Input images: base, source
    usage["base_image"] = n_voxels * float32
    usage["source_image"] = n_voxels * float32

    # Weight and mask
    usage["weight"] = n_voxels * float32
    usage["mask"] = n_voxels  # byte tensor

    # Global warp: xd, yd, zd
    usage["warp_xd"] = n_voxels * float32
    usage["warp_yd"] = n_voxels * float32
    usage["warp_zd"] = n_voxels * float32

    # Warped source (kept in memory for incremental updates)
    usage["warped_source"] = n_voxels * float32

    # Basis functions: (n_basis, n_patch_voxels) - worst case is full volume
    # In practice patches are smaller, but at lev=0 the patch is the full volume
    usage["basis_matrix"] = n_basis_max * n_voxels * float32

    # Cost function temporaries: coordinate grids (3x), interpolation results,
    # Jacobian computation (9 gradient fields), etc.
    # Coordinate grids for patch (worst case = full volume)
    usage["coord_grids"] = 3 * n_voxels * float32
    # Patch warp evaluation + composition
    usage["patch_warp_temps"] = 6 * n_voxels * float32
    # Jacobian energy computation (9 partial derivatives + det + energies)
    usage["jacobian_temps"] = 12 * n_voxels * float32
    # Incremental correlation: patch copies of base, source, weight
    usage["incor_temps"] = 3 * n_voxels * float32

    # PyTorch overhead (CUDA context, allocator fragmentation, etc.)
    usage["pytorch_overhead"] = 300 * 1024 * 1024  # ~300 MB

    usage["total"] = sum(usage.values())
    return usage


def estimate_gpu_memory_gb(nx: int, ny: int, nz: int) -> float:
    """Quick estimate of GPU memory needed in GB."""
    usage = estimate_gpu_memory_bytes(nx, ny, nz)
    return usage["total"] / (1024**3)


def compute_chunk_plan(
    nx: int,
    ny: int,
    nz: int,
    nt: int,
    gpu_memory_gb: float = 15.0,
    safety_factor: float = 0.85,
) -> list[tuple[int, int]]:
    """Plan chunking strategy for a 4D dataset.

    Determines how many timepoints can be processed simultaneously given
    GPU memory constraints.

    Args:
        nx, ny, nz: Spatial dimensions.
        nt: Number of timepoints.
        gpu_memory_gb: Available GPU memory in GB.
        safety_factor: Fraction of GPU memory to use (accounts for fragmentation).

    Returns:
        List of (start_t, end_t) tuples for each chunk. end_t is exclusive.
    """
    available_bytes = int(gpu_memory_gb * (1024**3) * safety_factor)
    per_volume = estimate_gpu_memory_bytes(nx, ny, nz)

    # Memory for one qwarp run: the full working set
    mem_per_volume = per_volume["total"]

    if mem_per_volume > available_bytes:
        print(
            f"WARNING: Single volume requires {mem_per_volume / 1e9:.2f} GB, "
            f"but only {available_bytes / 1e9:.2f} GB available. "
            f"Processing may fail or use system RAM."
        )

    # For 4D processing, each timepoint needs its own qwarp run.
    # We process one volume at a time since each needs independent optimization.
    # The overhead is the source 4D data sitting in CPU RAM.
    chunks = [(t, t + 1) for t in range(nt)]
    return chunks


def print_memory_report(nx: int, ny: int, nz: int, nt: int = 1) -> None:
    """Print a human-readable GPU memory usage report."""
    usage = estimate_gpu_memory_bytes(nx, ny, nz)

    print(f"GPU Memory Estimate for {nx}x{ny}x{nz} volume:")
    print(f"{'Category':<25} {'Size (MB)':>10}")
    print("-" * 37)

    for key, val in usage.items():
        if key == "total":
            continue
        print(f"  {key:<23} {val / 1e6:>10.1f}")

    print("-" * 37)
    print(f"  {'TOTAL':<23} {usage['total'] / 1e6:>10.1f}")
    print(f"  {'TOTAL (GB)':<23} {usage['total'] / 1e9:>10.2f}")

    if nt > 1:
        print(f"\n4D dataset: {nt} timepoints")
        print("  Each processed independently (serial chunking)")
        print(f"  CPU RAM for source 4D: {nx * ny * nz * nt * 4 / 1e9:.2f} GB")
