"""NORDIC-style denoising for complex or magnitude-only fMRI data.

Faithful PyTorch port of NIFTI_NORDIC.m (SteenMoeller/NORDIC_Raw).
The data-flow matches MATLAB line-by-line:

1. Load magnitude (+phase), form complex II, normalize by ABSOLUTE_SCALE
2. Phase stabilization pass 1: optionally remove meanphase (only when
   phase_slice_average=True, matching MATLAB phase_slice_average_for_
   kspace_centering=1); compute DD_phase via low-pass k-space windowing;
   apply temporal_phase==2 correction; multiply by exp(-j*angle(DD_phase))
3. G-factor estimation: extract first N volumes, run MP-PCA LLR,
   compute sqrt(noise_map) as g-factor proxy
4. Undo DD_phase, divide by gfactor, extract noise volume (SINGLE
   volume) BEFORE DD_phase application, apply temporal_phase==3
   correction (if mode==3), apply DD_phase
5. Measure noise from the extracted single noise volume
6. Compute NORDIC threshold (with sqrt(2) for complex data)
7. Main LLR patch-SVD denoising
8. Undo: multiply by gfactor, reapply DD_phase, reapply meanphase,
   rescale by ABSOLUTE_SCALE
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.signal.windows import tukey
from tqdm.auto import tqdm

from fastfuncstuff.io.afni import load_nifti, save_nifti
from fastfuncstuff.memory import (
    estimate_chunk_size,
    estimate_nordic_llr_memory,
    get_available_memory,
)
from fastfuncstuff.utils import get_device, to_tensor


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class NordicConfig:
    """Configuration for NORDIC-style denoising."""

    temporal_phase: int = 1
    phase_filter_width: float = 10.0
    noise_volume_last: int = 0
    factor_error: float = 1.0
    nordic: bool = True
    mp_mode: int = 0
    magnitude_only: bool = False
    kernel_size_pca: tuple[int, int, int] | None = None
    kernel_size_gfactor: tuple[int, int, int] = (14, 14, 1)
    gfactor_nvols: int = 90
    patch_overlap: int = 2
    gfactor_patch_overlap: int = 2
    use_magn_for_gfactor: bool = False
    phase_slice_average: bool = False
    save_gfactor_map: bool = False
    make_complex_nii: bool = False
    full_dynamic_range: bool = False
    write_gzipped_niftis: bool = True
    svd_batch_size: int = 512
    decomp_method: str = "auto"
    verbose: bool = True


@dataclass
class NordicOutputs:
    """Output paths from a NORDIC run."""

    magnitude_file: Path
    phase_file: Path | None
    gfactor_file: Path | None
    metadata_file: Path


@dataclass
class _PhaseArtifacts:
    """Stored phase fields for inverse reapplication."""

    mean_phase: torch.Tensor | None  # (nx, ny, nz) complex mean over time
    dd_phase: torch.Tensor | None  # (nx, ny, nz, nt) low-pass phase


@dataclass
class _LLRStats:
    """Diagnostic maps from LLR denoising."""

    weight: torch.Tensor
    noise_map: torch.Tensor
    threshold_map: torch.Tensor
    energy_removed: torch.Tensor
    snr_weight: torch.Tensor


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _default_kernel_size_pca(
    n_timepoints: int,
    n_slices: int | None = None,
) -> tuple[int, int, int]:
    """MATLAB: round((size(KSP2,4)*11)^(1/3)), replicated 3x.

    If n_slices < k, MATLAB falls back to:
        k2d = round((nt*11/nz)^(1/2)); kernel = [k2d, k2d, nz]
    """
    k = int(round((max(1, n_timepoints) * 11.0) ** (1.0 / 3.0)))
    k = max(3, k)
    if n_slices is not None and n_slices < k:
        k2d = int(round((max(1, n_timepoints) * 11.0 / n_slices) ** 0.5))
        k2d = max(3, k2d)
        return (k2d, k2d, n_slices)
    return (k, k, k)


def _build_patch_starts(dim: int, window: int, step: int) -> list[int]:
    if dim <= window:
        return [0]
    starts = list(range(0, dim - window + 1, max(1, step)))
    if starts[-1] != dim - window:
        starts.append(dim - window)
    return starts


def _phase_to_radians(phase_raw: np.ndarray) -> np.ndarray:
    """Match MATLAB: (I_P./range_norm - range_center)*2*pi."""
    phase_max = float(np.max(phase_raw))
    phase_min = float(np.min(phase_raw))
    range_norm = phase_max - phase_min
    if range_norm <= 0:
        return np.zeros_like(phase_raw, dtype=np.float32)
    range_center = (phase_max + phase_min) / range_norm * 0.5
    return (phase_raw.astype(np.float32) / range_norm - range_center) * (2.0 * np.pi)


# ---------------------------------------------------------------------------
# Phase stabilization  (matches MATLAB lines ~200-320)
# ---------------------------------------------------------------------------


def _compute_meanphase(
    data: torch.Tensor,
    noise_volume_last: int,
) -> torch.Tensor:
    """MATLAB: meanphase=mean(KSP2(:,:,:,[1:end-noise_volume_last]),4)."""
    nt = data.shape[-1]
    if noise_volume_last > 0 and nt > noise_volume_last:
        src = data[..., : nt - noise_volume_last]
    else:
        src = data
    return torch.mean(src, dim=3)  # (nx, ny, nz) complex


def _meanphase_unit(meanphase: torch.Tensor) -> torch.Tensor:
    """Precompute exp(-j·angle(meanphase)) = conj(meanphase)/|meanphase|.

    Returns (nx, ny, nz) complex64 unit-phasor.  Conjugate to restore.
    Uses abs+divide instead of atan2+sin+cos.
    """
    mag = torch.abs(meanphase)
    safe = torch.where(mag < 1e-12, torch.ones_like(mag), mag)
    return meanphase.conj() / safe


def _remove_meanphase(data: torch.Tensor, mp_unit: torch.Tensor) -> torch.Tensor:
    """Vectorized: KSP2(:,:,z,:) *= mp_unit(:,:,z).  mp_unit is pre-computed."""
    data *= mp_unit.unsqueeze(-1)
    return data


def _restore_meanphase(data: torch.Tensor, mp_unit: torch.Tensor) -> torch.Tensor:
    """Conjugate of mp_unit restores the original phase."""
    data *= mp_unit.conj().unsqueeze(-1)
    return data


def _compute_dd_phase(
    data: torch.Tensor,
    phase_filter_width: float,
    verbose: bool,
    z_batch: int = 8,
) -> torch.Tensor:
    """Low-pass temporal phase via DC-centered k-space Tukey windowing.

    Processes slabs of ``z_batch`` z-slices at a time for good GPU saturation
    without the peak memory of operating on the full 4-D array.
    Each slab: (nx, ny, z_batch * nt) → single batched fft2.
    """
    nx, ny, nz, nt = data.shape
    win_x = torch.tensor(
        tukey(nx, alpha=1.0, sym=True),
        dtype=torch.float32,
        device=data.device,
    ) ** float(phase_filter_width)
    win_y = torch.tensor(
        tukey(ny, alpha=1.0, sym=True),
        dtype=torch.float32,
        device=data.device,
    ) ** float(phase_filter_width)
    # win2d shape: (nx, ny, 1) — broadcasts over batch dim
    win2d = (win_x[:, None] * win_y[None, :]).unsqueeze(-1)

    dd = torch.empty_like(data)
    for z0 in range(0, nz, z_batch):
        z1 = min(z0 + z_batch, nz)
        n_slices = z1 - z0
        # View as (nx, ny, n_slices * nt) — no copy
        sl = data[:, :, z0:z1, :].reshape(nx, ny, n_slices * nt)
        k = torch.fft.fftshift(torch.fft.fft2(sl, dim=(0, 1)), dim=(0, 1))
        k *= win2d
        dd[:, :, z0:z1, :] = torch.fft.ifft2(
            torch.fft.ifftshift(k, dim=(0, 1)), dim=(0, 1)
        ).reshape(nx, ny, n_slices, nt)
        del k
    return dd


def _apply_temporal_phase_correction(
    data: torch.Tensor,
    dd_phase: torch.Tensor,
    mode: int,
) -> torch.Tensor:
    """Apply temporal_phase==2 or ==3 spike correction to dd_phase.

    Processes per z-slice to limit peak GPU memory (~4 slice-sized temps
    instead of 4 full 4-D arrays).

    Mode 2 (MATLAB lines ~300): mask = abs(phase_diff)>1, replace dd with data.
    Mode 3 (MATLAB lines ~490): same but additionally requires abs(data)>sqrt(2).

    *dd_phase* may reside on CPU; corrected slices are written back to it.
    """
    if mode < 2:
        return dd_phase
    dev = data.device
    nz = data.shape[2]
    sqrt2 = math.sqrt(2.0)
    for z in range(nz):
        dd_z = dd_phase[:, :, z, :].to(dev, non_blocking=True)
        data_z = data[:, :, z, :]  # view on GPU
        safe_dd = torch.where(torch.abs(dd_z) < 1e-12, torch.ones_like(dd_z), dd_z)
        phase_diff = torch.angle(data_z / safe_dd)
        mask = torch.abs(phase_diff) > 1.0
        if mode == 3:
            mask = mask & (torch.abs(data_z) > sqrt2)
        result_z = torch.where(mask, data_z, dd_z)
        # Write back — handles both GPU and CPU dd_phase
        dd_phase[:, :, z, :] = result_z.to(dd_phase.device, non_blocking=True)
        del dd_z, safe_dd, phase_diff, mask, result_z
    return dd_phase


def _unit_phasor_neg(z: torch.Tensor) -> torch.Tensor:
    """Compute exp(-j·angle(z)) = conj(z)/|z| without trig (atan2+sin+cos).

    Falls back to zero where |z| < 1e-12 to avoid divide-by-zero.
    """
    mag = torch.abs(z)
    safe = torch.where(mag < 1e-12, torch.ones_like(mag), mag)
    return z.conj() / safe


def _dd_phase_multiply_inplace(
    data: torch.Tensor,
    dd_phase: torch.Tensor,
    *,
    conjugate: bool = False,
) -> None:
    """Apply or undo dd_phase in-place, one z-slice at a time.

    Uses ``conj(z)/|z|`` instead of ``exp(-j·angle(z))`` to avoid
    atan2+sin+cos — just abs+divide+conj which is ~3× faster per element.

    Pass ``conjugate=True`` to undo/restore (multiply by ``exp(+j·angle)``
    instead of ``exp(-j·angle)``).

    If *dd_phase* lives on CPU it is transferred slice-by-slice via
    ``non_blocking`` copies so the GPU never holds the entire array.
    """
    dev = data.device
    nz = data.shape[2]
    for z in range(nz):
        dd_z = dd_phase[:, :, z, :].to(dev, non_blocking=True)
        u_z = _unit_phasor_neg(dd_z)
        if conjugate:
            data[:, :, z, :] *= u_z.conj()
        else:
            data[:, :, z, :] *= u_z
        del dd_z, u_z


# ---------------------------------------------------------------------------
# LLR patch processing
# ---------------------------------------------------------------------------


def _mp_hard_index_batch(
    s_batch: torch.Tensor,
    m: int,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized MP-PCA threshold for a batch of singular value vectors.

    Parameters
    ----------
    s_batch : (B, R) real singular values
    m, n    : patch matrix dimensions

    Returns
    -------
    cuts : (B,) int64 tensor — number of components to KEEP per patch
    noise_est : (B,) float — MATLAB-compatible sigmasq_2(t) per patch
    """
    B, R = s_batch.shape
    if R <= 1:
        return (
            torch.ones(B, dtype=torch.long, device=s_batch.device),
            torch.zeros(B, dtype=s_batch.dtype, device=s_batch.device),
        )

    nn_mat = max(m, n)
    mm_mat = min(m, n)
    R_eff = min(m, n)
    s_use = s_batch[:, :R_eff]  # (B, R_eff)

    vals = s_use**2 / float(mm_mat)  # (B, R_eff)

    # cumulative mean from tail
    csum = torch.flip(torch.cumsum(torch.flip(vals, [1]), dim=1), [1])
    denom = torch.arange(R_eff, 0, -1, device=s_batch.device, dtype=s_batch.dtype)
    cmean = csum / denom.unsqueeze(0)

    idx = torch.arange(R_eff, device=s_batch.device, dtype=s_batch.dtype)
    scaling = (float(nn_mat) - idx) / float(mm_mat)
    scaling = torch.clamp(scaling, min=1e-8)
    sigmasq_1 = cmean / scaling.unsqueeze(0)

    gamma = (float(m) - idx) / float(n)
    gamma = torch.clamp(gamma, min=1e-8)
    range_mp = 4.0 * torch.sqrt(gamma)
    range_data = vals - vals[:, -1:]
    sigmasq_2 = range_data / torch.clamp(range_mp.unsqueeze(0), min=1e-8)

    # First index where sigmasq_2 < sigmasq_1
    below = sigmasq_2 < sigmasq_1  # (B, R_eff)
    # argmax on bool gives first True; if no True, gives 0 (we use 'any' to handle)
    has_hit = below.any(dim=1)
    first_hit = below.to(torch.int64).argmax(dim=1)  # (B,)
    # If no hit, keep all (cut = R_eff)
    cuts = torch.where(has_hit, first_hit, torch.tensor(R_eff, device=s_batch.device))
    cuts = torch.clamp(cuts, min=1, max=R_eff)

    # MATLAB: NOISE += sigmasq_2(t) — gather sigmasq_2 at the cut index
    cut_clamped = torch.clamp(cuts, max=R_eff - 1)
    noise_est = sigmasq_2.gather(1, cut_clamped.unsqueeze(1)).squeeze(1)  # (B,)
    return cuts, noise_est


def _optimal_shrinkage_weights(
    s: torch.Tensor,
    sigma: float,
    m: int,
    n: int,
) -> torch.Tensor:
    """Gavish-Donoho optimal singular value shrinkage weights (Frobenius loss).

    For an m×n patch matrix with i.i.d. noise entries of variance σ², the
    optimal denoised singular value is::

        η(y) = √(max(0, (y² - σ²(m+n))² - 4σ⁴mn)) / y

    for y > σ(√m + √n) (the Marchenko-Pastur bulk edge), and 0 otherwise.

    Returns weights w_i = η(y_i)/y_i in [0, 1] so the shrunk singular
    values are simply ``s * w``.

    Parameters
    ----------
    s : (B, K) real singular values (descending)
    sigma : noise std per matrix entry
    m, n : patch matrix dimensions (rows, cols)

    Returns
    -------
    weights : (B, K) float tensor in [0, 1]
    """
    s2 = s**2
    sigma2 = sigma**2
    sigma4 = sigma2**2

    # Marchenko-Pastur bulk edge for singular values
    bulk_edge = sigma * (math.sqrt(m) + math.sqrt(n))

    # (y² - σ²(m+n))² - 4σ⁴mn
    inner = (s2 - sigma2 * (m + n)) ** 2 - 4.0 * sigma4 * m * n
    inner = torch.clamp(inner, min=0.0)

    # w = η(y)/y = √inner / y²
    s2_safe = torch.clamp(s2, min=1e-30)
    weights = torch.sqrt(inner) / s2_safe

    # Zero out everything at or below bulk edge
    weights = torch.where(s > bulk_edge, weights, torch.zeros_like(weights))
    weights = torch.clamp(weights, min=0.0, max=1.0)
    return weights


def _llr_denoise(
    data: torch.Tensor,
    kernel_size: tuple[int, int, int],
    patch_overlap: int,
    threshold_mode: str,
    threshold_value: float,
    verbose: bool,
    return_recon: bool = True,
    svd_batch_size: int = 512,
    decomp_method: str = "auto",
    device: torch.device | None = None,
    noise_sigma: float = 0.0,
) -> tuple[torch.Tensor, _LLRStats]:
    """Patch-based local low-rank denoising with batched decomposition.

    Supports two decomposition backends:
    - **svd**: ``torch.linalg.svd`` (or ``svdvals``).
    - **eigh**: Gram-matrix eigendecomposition ``X^H X`` → ``eigh``.
      For tall-skinny patches (M >> N), the Gram matrix (N×N) is much
      smaller than the patch matrix (M×N), and cuSOLVER's batched syevd
      is faster than batched gesvd.  Reconstruction via subspace
      projection ``X @ V_k @ V_k^H`` avoids ever computing U.
    - **auto** (default): picks *eigh* when M/N ≥ 2, else *svd*.

    Parameters
    ----------
    device : torch.device, optional
        Compute device for decomposition and accumulators.  When different
        from ``data.device`` (e.g. data on CPU, device='cuda'), patches are
        extracted on the data device then transferred.  This lets large
        volumes live on CPU while computation stays on GPU.
        Default: ``data.device``.
    """
    if device is None:
        device = data.device
    data_dev = data.device  # where the raw data lives
    cross_device = device != data_dev

    nx, ny, nz, nt = data.shape
    wx = min(kernel_size[0], nx)
    wy = min(kernel_size[1], ny)
    wz = min(kernel_size[2], nz)
    n_patch_voxels = wx * wy * wz

    sx = max(1, wx // max(1, patch_overlap))
    sy = max(1, wy // max(1, patch_overlap))
    sz = max(1, wz // max(1, patch_overlap))

    xs = _build_patch_starts(nx, wx, sx)
    ys = _build_patch_starts(ny, wy, sy)
    zs = _build_patch_starts(nz, wz, sz)

    # Pre-compute all patch corner coordinates
    corners = []
    for x0 in xs:
        for y0 in ys:
            for z0 in zs:
                corners.append((x0, y0, z0))
    total = len(corners)

    recon_acc = (
        torch.zeros(nx, ny, nz, nt, dtype=data.dtype, device=device) if return_recon else None
    )
    weight = torch.zeros((nx, ny, nz), dtype=torch.float32, device=device)
    noise_map = torch.zeros_like(weight)
    threshold_map = torch.zeros_like(weight)
    energy_removed = torch.zeros_like(weight)
    snr_weight = torch.zeros_like(weight)

    pbar = tqdm(total=total, desc="LLR patches", unit="patch") if verbose else None

    K = min(n_patch_voxels, nt)
    M, N = n_patch_voxels, nt

    # Choose decomposition method
    if decomp_method == "auto":
        use_eigh = M >= 2 * N
    elif decomp_method == "eigh":
        use_eigh = True
    else:
        use_eigh = False

    if verbose:
        method_name = "eigh (Gram)" if use_eigh else "svd"
        print(f"  Decomp method: {method_name}  (patch {M}×{N}, ratio {M / N:.1f})")

    # --- Vectorized patch I/O setup ---
    # 3D voxel offsets within a patch (M,) — for extraction via advanced indexing
    # This avoids data.contiguous() which would copy the entire volume.
    _ox = torch.arange(wx)
    _oy = torch.arange(wy)
    _oz = torch.arange(wz)
    gx, gy, gz = torch.meshgrid(_ox, _oy, _oz, indexing="ij")
    dx = gx.ravel()  # (M,)
    dy = gy.ravel()
    dz = gz.ravel()

    # 1D flat offsets (for scatter into contiguous accumulators)
    local_offsets_flat = dx * (ny * nz) + dy * nz + dz  # (M,) CPU

    # Decompose corners into per-axis arrays (total,) — CPU
    corner_xs = torch.tensor([c[0] for c in corners], dtype=torch.long)
    corner_ys = torch.tensor([c[1] for c in corners], dtype=torch.long)
    corner_zs = torch.tensor([c[2] for c in corners], dtype=torch.long)
    corners_flat = corner_xs * (ny * nz) + corner_ys * nz + corner_zs

    # Flatten accumulators for scatter (all allocated contiguous, so reshape = view)
    if return_recon:
        recon_flat = recon_acc.reshape(-1, nt)
    weight_flat = weight.reshape(-1)
    noise_flat = noise_map.reshape(-1)
    threshold_flat = threshold_map.reshape(-1)
    energy_flat = energy_removed.reshape(-1)
    snr_flat = snr_weight.reshape(-1)
    ones_BM = torch.ones(svd_batch_size * M, device=device)

    for batch_start in range(0, total, svd_batch_size):
        B = min(svd_batch_size, total - batch_start)

        # --- Extract patches via 3D advanced indexing (no contiguous copy) ---
        bx = corner_xs[batch_start : batch_start + B]  # (B,) CPU
        by = corner_ys[batch_start : batch_start + B]
        bz = corner_zs[batch_start : batch_start + B]
        # Build (B, M) index arrays per axis — on data's device for extraction
        xi = (bx[:, None] + dx[None, :]).to(data_dev)  # (B, M)
        yi = (by[:, None] + dy[None, :]).to(data_dev)
        zi = (bz[:, None] + dz[None, :]).to(data_dev)
        mats = data[xi, yi, zi, :]  # (B, M, nt) — on data_dev
        if cross_device:
            mats = mats.to(device)

        # Flat indices for scatter (accumulators are contiguous, on `device`)
        b_corners_flat = corners_flat[batch_start : batch_start + B]
        flat_b = (b_corners_flat[:, None] + local_offsets_flat[None, :]).reshape(-1).to(device)

        # ---- Decomposition ----
        if use_eigh:
            # Gram-matrix eigendecomposition: eigh(X^H X) gives
            # eigenvalues = σ² (ascending) and eigenvectors = right
            # singular vectors V.  Cheaper than full SVD when M >> N
            # because the Gram is only (N×N) vs the (M×N) patch matrix.
            G = mats.mH @ mats  # (B, N, N) — cuBLAS batched matmul
            if return_recon:
                eigvals, V = torch.linalg.eigh(G)  # ascending
                del G
                # Keep mats alive for projection-based reconstruction
            else:
                eigvals = torch.linalg.eigvalsh(G)  # ascending, no eigvecs
                del G, mats
            # Flip to descending order (SVD convention)
            s0 = torch.sqrt(torch.clamp(eigvals.flip(1), min=0))  # (B, K)
            del eigvals
            if return_recon:
                V = V.flip(2)  # columns now match descending eigenvalue order
        else:
            if return_recon:
                # Full SVD needed for reconstruction
                u, s, vh = torch.linalg.svd(mats, full_matrices=False)
                del mats
                s0 = s.abs()  # (B, K)
            else:
                # G-factor pass: only need singular values → ~2× faster
                s0 = torch.linalg.svdvals(mats).abs()  # (B, K)
                del mats

        # ---- Threshold / weight computation (fully vectorized) ----
        # All modes produce float `weights` (B, K) in [0, 1]:
        #   binary (mp/nordic): 0 or 1
        #   optimal shrinkage:  continuous Gavish-Donoho weights
        if threshold_mode == "optimal":
            weights = _optimal_shrinkage_weights(s0, noise_sigma, M, N)
            idx_removed_vec = (weights == 0).sum(dim=1).float()
            local_noise_vec = torch.full((B,), noise_sigma**2, device=device, dtype=torch.float32)
        elif threshold_mode == "mp":
            cuts, mp_noise = _mp_hard_index_batch(s0, m=n_patch_voxels, n=nt)
            idx_range = torch.arange(K, device=s0.device).unsqueeze(0)  # (1, K)
            weights = (idx_range < cuts.unsqueeze(1)).float()  # (B, K)
            idx_removed_vec = (K - cuts).float()  # (B,)
            # MATLAB: NOISE += sigmasq_2(t) — the MP-estimated noise variance
            local_noise_vec = mp_noise
        else:
            # NORDIC hard threshold
            weights = (s0 >= threshold_value).float()  # (B, K)
            idx_removed_vec = (weights == 0).sum(dim=1).float()  # (B,)
            local_noise_vec = torch.full(
                (B,), threshold_value**2 / max(1, nt), device=device, dtype=torch.float32
            )

        n_kept = (weights > 0).sum(dim=1)  # (B,) — number of non-zero weight components

        if return_recon:
            max_k = max(1, int(n_kept.max().item()))

            if use_eigh:
                # Subspace projection: X_denoised = X @ V_k @ diag(mask) @ V_k^H
                # This is exact and doesn't require U.  Truncation to max_k
                # is always valid: any component beyond max_k is not kept by
                # ANY patch in the batch, so V columns beyond max_k contribute
                # nothing.  The per-patch mask handles variable k correctly.
                V_k = V[:, :, :max_k]  # (B, N, max_k)
                del V
                coeff = mats @ V_k  # (B, M, max_k) — project into eigenspace
                del mats
                mask_k = weights[:, :max_k].unsqueeze(1)  # (B, 1, max_k)
                coeff = coeff * mask_k  # scale by shrinkage weights
                recon_batch = coeff @ V_k.mH  # (B, M, N)
                del coeff, V_k
            else:
                # SVD truncated reconstruction with prefix safety check.
                # Singular values from svd are guaranteed descending, so the
                # keep mask is always a contiguous prefix (0..k-1).  Verify
                # at runtime and fall back to full matmul if violated.
                prefix_sums = (weights[:, :max_k] > 0).sum(dim=1)
                is_prefix = (prefix_sums == n_kept).all()

                if is_prefix:
                    s_trunc = s[:, :max_k] * weights[:, :max_k]  # (B, max_k)
                    recon_batch = (u[:, :, :max_k] * s_trunc.unsqueeze(1)) @ vh[:, :max_k, :]
                    del u, s, s_trunc, vh
                else:
                    # Non-prefix (should never happen). Full reconstruction.
                    s_masked = s * weights  # (B, K)
                    recon_batch = (u * s_masked.unsqueeze(1)) @ vh
                    del u, s, s_masked, vh

        # ---- Vectorized diagnostics (no per-patch .item() syncs) ----
        # Energy removed: Frobenius norm of removed signal per patch.
        # For binary weights: sqrt(sum(s_discarded²))
        # For continuous shrinkage: sqrt(sum((s*(1-w))²))
        s0_removed = s0 * (1.0 - weights)  # amount removed per component
        e_rem_vec = torch.sqrt((s0_removed**2).sum(dim=1))  # (B,)

        # SNR weight: s0[:,0] / s0 at boundary of kept region
        boundary_idx = torch.clamp(n_kept - 1, min=0).unsqueeze(1)  # (B, 1)
        s0_boundary = s0.gather(1, boundary_idx).squeeze(1)  # (B,)
        s0_boundary = torch.clamp(s0_boundary, min=1e-8)
        snr_w_vec = s0[:, 0] / s0_boundary  # (B,)

        del s0, s0_removed, weights

        # ---- Scatter results back (vectorized index_add_) ----
        if return_recon:
            recon_flat.index_add_(0, flat_b, recon_batch.reshape(-1, nt))
            del recon_batch
        weight_flat.index_add_(0, flat_b, ones_BM[: B * M])

        # Per-patch scalar diagnostics → expand to per-voxel → scatter
        for vals, target in zip(
            [idx_removed_vec.float(), local_noise_vec, e_rem_vec, snr_w_vec],
            [threshold_flat, noise_flat, energy_flat, snr_flat],
        ):
            target.index_add_(0, flat_b, vals.repeat_interleave(M))
        del idx_removed_vec, local_noise_vec, e_rem_vec, snr_w_vec, flat_b

        if pbar is not None:
            pbar.update(B)

    if pbar is not None:
        pbar.close()

    w = torch.clamp(weight, min=1.0)
    if return_recon:
        recon_out = recon_acc / w[..., None]
    else:
        recon_out = torch.empty((0,), dtype=data.dtype, device=device)

    stats = _LLRStats(
        weight=w,
        noise_map=noise_map / w,
        threshold_map=threshold_map / w,
        energy_removed=energy_removed / w,
        snr_weight=snr_weight / w,
    )
    return recon_out, stats


# ---------------------------------------------------------------------------
# NORDIC threshold estimation  (MATLAB lines ~580-600)
# ---------------------------------------------------------------------------


def _estimate_nordic_lambda(
    kernel_size: tuple[int, int, int],
    n_timepoints: int,
    measured_noise: float,
    factor_error: float,
    is_complex: bool,
    device: torch.device,
) -> float:
    """MATLAB: NVR_threshold/10 * [sqrt(2)*] measured_noise * factor_error.

    Uses a single batched SVD (10 random matrices at once) instead of 10
    sequential calls.
    """
    n_patch = int(np.prod(kernel_size))
    # Batched: (10, n_patch, n_timepoints)
    rnd = torch.randn((10, n_patch, n_timepoints), dtype=torch.float32, device=device)
    svals = torch.linalg.svdvals(rnd)  # (10, min(n_patch, n_timepoints))
    base = float(svals[:, 0].mean().item())
    if is_complex:
        return base * math.sqrt(2.0) * measured_noise * factor_error
    else:
        return base * measured_noise * factor_error


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_nordic(
    magnitude_file: str,
    phase_file: str | None,
    output_prefix: str,
    config: NordicConfig | None = None,
    device: torch.device | None = None,
) -> NordicOutputs:
    """Run NORDIC-style denoising matching NIFTI_NORDIC.m data flow."""
    cfg = config or NordicConfig()
    dev = (
        device if device is not None else get_device("cuda" if torch.cuda.is_available() else None)
    )

    # ------------------------------------------------------------------
    # 1. Load data, form complex, apply ABSOLUTE_SCALE
    # ------------------------------------------------------------------
    mag_img = load_nifti(magnitude_file)
    mag_np = np.abs(mag_img.get_fdata(dtype=np.float32)).astype(np.float32)

    phase_np: np.ndarray | None = None
    phase_in_used = phase_file is not None and not cfg.magnitude_only
    if phase_in_used:
        phase_img = load_nifti(phase_file)
        phase_raw = phase_img.get_fdata(dtype=np.float32).astype(np.float32)
        phase_np = _phase_to_radians(phase_raw)

    if cfg.magnitude_only:
        # MATLAB: II=single(I_M); temporal_phase=0
        ii_np = mag_np.astype(np.float32)
    elif phase_np is not None:
        ii_np = (mag_np * np.exp(1j * phase_np)).astype(np.complex64)
    else:
        ii_np = mag_np.astype(np.complex64)

    is_complex = not cfg.magnitude_only and phase_np is not None

    # MATLAB: ABSOLUTE_SCALE = min(TEMPVOL(TEMPVOL~=0)); II=II./ABSOLUTE_SCALE
    tempvol = np.abs(ii_np[..., 0])
    nonzero = tempvol[tempvol != 0]
    if nonzero.size > 0:
        absolute_scale = float(np.min(nonzero))
    else:
        absolute_scale = 1.0
    absolute_scale = max(absolute_scale, 1e-30)
    ii_np = ii_np / absolute_scale

    # Move to device as complex64 (even magnitude-only uses complex in MATLAB
    # for the SVD pathway; for magnitude_only the imaginary part is zero).
    dtype = torch.complex64
    II = to_tensor(ii_np.astype(np.complex64), dtype=dtype, device=dev)
    del ii_np
    nx, ny, nz, nt = II.shape

    # Kernel sizes
    if cfg.kernel_size_pca is None:
        kernel_pca = _default_kernel_size_pca(nt, n_slices=nz)
    else:
        kernel_pca = cfg.kernel_size_pca

    if cfg.verbose:
        print("\nNORDIC denoising")
        print(f"  Input shape: {tuple(II.shape)}")
        print(f"  Device: {dev}")
        print(f"  Kernel PCA: {kernel_pca}")
        print(f"  Temporal phase mode: {cfg.temporal_phase}")
        print(f"  ABSOLUTE_SCALE: {absolute_scale:.6g}")

    # Memory diagnostic (informational only)
    _ = estimate_chunk_size(
        n_voxels=nx * ny * nz,
        n_timepoints=nt,
        n_regressors=int(np.prod(kernel_pca)),
        device=dev,
        operation="denoise",
        verbose=cfg.verbose,
    )

    # ------------------------------------------------------------------
    # 2. Phase stabilization pass 1  (for g-factor estimation)
    #    MATLAB: KSP2=II; remove meanphase; compute DD_phase; apply
    #    temporal_phase==2; apply DD_phase
    #
    #    Memory optimization: instead of II.clone(), we operate in-place
    #    and undo dd_phase after g-factor to restore to meanphase-only
    #    state for step 4 (which needs meanphase-corrected data without
    #    dd_phase).  Saves one full 4D complex copy (~2-5 GB).
    # ------------------------------------------------------------------
    effective_tp = cfg.temporal_phase if (phase_in_used and not cfg.magnitude_only) else 0

    # MATLAB: meanphase *= phase_slice_average_for_kspace_centering (default 0)
    # When this is False (default), meanphase correction is a no-op — matching
    # MATLAB's default behaviour where meanphase is zeroed.
    mp_unit: torch.Tensor | None = None
    if cfg.phase_slice_average:
        meanphase = _compute_meanphase(II, cfg.noise_volume_last)
        mp_unit = _meanphase_unit(meanphase)
        del meanphase
        _remove_meanphase(II, mp_unit)

    dd_phase: torch.Tensor | None = None
    if effective_tp > 0:
        dd_phase = _compute_dd_phase(II, cfg.phase_filter_width, cfg.verbose)
        if effective_tp == 2:
            dd_phase = _apply_temporal_phase_correction(II, dd_phase, mode=2)
        # Apply dd_phase in-place for g-factor pass (per z-slice)
        _dd_phase_multiply_inplace(II, dd_phase)

        # Move dd_phase to CPU — frees ~2.5 GiB GPU during g-factor LLR.
        # It will be transferred back per z-slice when needed.
        dd_phase = dd_phase.cpu()

    # ------------------------------------------------------------------
    # 3. G-factor estimation  (MP-PCA pass on first N volumes)
    # ------------------------------------------------------------------
    gfactor = torch.ones((nx, ny, nz), dtype=torch.float32, device=dev)
    gfactor_file: Path | None = None

    if cfg.save_gfactor_map or cfg.nordic or cfg.mp_mode == 1:
        g_nvol = min(max(1, cfg.gfactor_nvols), nt)
        if cfg.use_magn_for_gfactor:
            g_data = torch.abs(II[..., :g_nvol]).to(dtype)
        else:
            g_data = II[..., :g_nvol]

        # Clean nan/inf
        g_data = torch.where(torch.isfinite(g_data), g_data, torch.zeros_like(g_data))

        _, g_stats = _llr_denoise(
            g_data,
            kernel_size=cfg.kernel_size_gfactor,
            patch_overlap=max(1, cfg.gfactor_patch_overlap),
            threshold_mode="mp",
            threshold_value=0.0,
            verbose=cfg.verbose,
            return_recon=False,
            svd_batch_size=cfg.svd_batch_size,
            decomp_method=cfg.decomp_method,
            device=dev,
        )
        del g_data

        # MATLAB: gfactor = sqrt(NOISE./KSP_weight)  (already averaged in stats)
        gfactor = torch.sqrt(torch.clamp(g_stats.noise_map, min=1e-8))
        gvals = gfactor[torch.isfinite(gfactor) & (gfactor > 0)]
        if gvals.numel() > 0:
            med = torch.median(gvals)
            gfactor = torch.where(gfactor <= 0, med, gfactor)
            gfactor = torch.where(torch.isfinite(gfactor), gfactor, med)
        else:
            gfactor = torch.ones_like(gfactor)

        # MATLAB: gfactor(gfactor<1) = median(gfactor(gfactor~=0)) when zero elements
        bad = gfactor < 1.0
        if bad.any():
            gvals2 = gfactor[gfactor > 0]
            if gvals2.numel() > 0:
                gfactor[bad] = torch.median(gvals2)

    if cfg.mp_mode == 2:
        gfactor = torch.ones_like(gfactor)

    if cfg.verbose:
        print(f"  G-factor range: [{float(gfactor.min()):.4f}, {float(gfactor.max()):.4f}]")

    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 4. Undo dd_phase from g-factor pass (restore to meanphase-only),
    #    divide by gfactor, extract noise vol, apply tp==3, apply dd_phase
    #    This avoids a full 4D clone by undoing+redoing dd_phase in-place.
    # ------------------------------------------------------------------
    if dd_phase is not None:
        _dd_phase_multiply_inplace(II, dd_phase, conjugate=True)
    # II is now back to meanphase-corrected state

    # Divide by gfactor in-place
    II /= gfactor[..., None]

    # Extract noise volume BEFORE DD_phase (MATLAB line ~470)
    noise_vol: torch.Tensor | None = None
    if cfg.noise_volume_last > 0 and nt > cfg.noise_volume_last:
        noise_vol = II[..., nt - cfg.noise_volume_last].clone()

    # temporal_phase==3 correction (applied AFTER gfactor division, BEFORE dd_phase)
    if effective_tp == 3 and dd_phase is not None:
        dd_phase = _apply_temporal_phase_correction(II, dd_phase, mode=3)

    # Apply DD_phase in-place (per z-slice)
    if dd_phase is not None:
        _dd_phase_multiply_inplace(II, dd_phase)

    # Clean nan/inf in-place
    nan_mask = ~torch.isfinite(II)
    if nan_mask.any():
        II[nan_mask] = 0

    # II is now ready for main LLR denoising (rename for clarity)
    KSP2 = II

    # ------------------------------------------------------------------
    # 5. Measure noise from extracted noise volume
    # ------------------------------------------------------------------
    # MATLAB: measured_noise = std(KSP2_NOISE(KSP2_NOISE~=0))
    if noise_vol is not None:
        nv = noise_vol.reshape(-1)
        nv = nv[nv != 0]
        if nv.numel() == 0:
            measured_noise = 1.0
        else:
            measured_noise = float(torch.std(nv).real.item())
        del noise_vol
    else:
        measured_noise = 1.0

    # MATLAB: if ~use_magn_for_gfactor & ~magnitude_only
    #             measured_noise = measured_noise / sqrt(2)
    if not cfg.use_magn_for_gfactor and not cfg.magnitude_only:
        measured_noise /= math.sqrt(2.0)
    measured_noise = max(measured_noise, 1e-8)

    if cfg.verbose:
        print(f"  Measured noise sigma: {measured_noise:.6g}")

    # ------------------------------------------------------------------
    # 6. Compute NORDIC threshold
    # ------------------------------------------------------------------
    if cfg.mp_mode > 0:
        threshold_mode = "mp"
        threshold_value = 0.0
    else:
        threshold_mode = "nordic"
        threshold_value = _estimate_nordic_lambda(
            kernel_size=kernel_pca,
            n_timepoints=nt,
            measured_noise=measured_noise,
            factor_error=cfg.factor_error,
            is_complex=is_complex,
            device=dev,
        )
        if cfg.verbose:
            print(f"  NORDIC lambda: {threshold_value:.6g}")

    # ------------------------------------------------------------------
    # 7. Main LLR denoising
    # ------------------------------------------------------------------
    # Memory check: if data + accumulators + working set exceeds available
    # GPU memory, offload data to CPU and stream patches to GPU per batch.
    if dev.type == "cuda":
        mem_est = estimate_nordic_llr_memory(
            shape=KSP2.shape,
            kernel_size=kernel_pca,
            svd_batch_size=cfg.svd_batch_size,
            dtype_bytes=KSP2.element_size(),
            return_recon=True,
        )
        avail = get_available_memory(dev)
        # Need accumulators + working set on GPU (data can go to CPU)
        gpu_without_data = mem_est["total"] - mem_est["data"]
        if mem_est["total"] > avail:
            if cfg.verbose:
                print(
                    f"  Memory guard: LLR needs ~{mem_est['total'] / 1024**3:.2f} GiB "
                    f"but only {avail / 1024**3:.2f} GiB available."
                )
                print(
                    f"  Offloading input ({mem_est['data'] / 1024**3:.2f} GiB) to CPU; "
                    f"accumulators + working set = {gpu_without_data / 1024**3:.2f} GiB stay on GPU."
                )
            KSP2 = KSP2.cpu()
            torch.cuda.empty_cache()

    denoised, llr_stats = _llr_denoise(
        KSP2,
        kernel_size=kernel_pca,
        patch_overlap=max(1, cfg.patch_overlap),
        threshold_mode=threshold_mode,
        threshold_value=threshold_value,
        verbose=cfg.verbose,
        svd_batch_size=cfg.svd_batch_size,
        decomp_method=cfg.decomp_method,
        device=dev,
    )

    # Free the input array — denoised is a separate allocation.
    # KSP2 is an alias for II; delete both to drop the refcount.
    del KSP2
    II = None  # noqa: F841
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    mean_removed = float(torch.mean(llr_stats.threshold_map).item())
    if cfg.verbose:
        print(f"  Mean components removed per patch: {mean_removed:.3f}")
        if mean_removed < 0.1:
            print(
                "  WARNING: Near no-op denoising detected "
                f"(mean singular values removed per patch={mean_removed:.4f})."
            )

    # ------------------------------------------------------------------
    # 8. Undo transformations: gfactor, meanphase, DD_phase, scale
    # ------------------------------------------------------------------
    # All operations are in-place to avoid allocating a second 4-D copy.
    # MATLAB: IMG2 *= gfactor; IMG2 *= exp(i*angle(DD_phase)); IMG2 *= exp(i*angle(meanphase))
    denoised *= gfactor[..., None]

    if dd_phase is not None:
        _dd_phase_multiply_inplace(denoised, dd_phase, conjugate=True)
        del dd_phase
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    if mp_unit is not None:
        _restore_meanphase(denoised, mp_unit)
    del mp_unit

    # MATLAB: IMG2 *= ABSOLUTE_SCALE
    denoised *= absolute_scale

    # Clean nan in-place
    nan_mask = ~torch.isfinite(denoised)
    if nan_mask.any():
        denoised[nan_mask] = 0
    del nan_mask

    # ------------------------------------------------------------------
    # 9. Write outputs
    # ------------------------------------------------------------------
    out_prefix = Path(output_prefix)
    out_dir = out_prefix.parent if out_prefix.parent != Path("") else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = ".nii.gz" if cfg.write_gzipped_niftis else ".nii"

    if cfg.make_complex_nii:
        magn_path = out_dir / f"{out_prefix.name}_magn{ext}"
        phase_path = out_dir / f"{out_prefix.name}_phase{ext}"
    else:
        magn_path = out_dir / f"{out_prefix.name}{ext}"
        phase_path = out_dir / f"{out_prefix.name}_phase{ext}" if phase_in_used else None

    # Compute magnitude (and phase) on GPU, move to CPU, then free GPU
    # to avoid holding denoised + abs + angle all at once.
    magn_np_out = torch.abs(denoised).cpu().numpy().astype(np.float32)
    phase_out_np: np.ndarray | None = None
    if phase_path is not None:
        phase_out_np = torch.angle(denoised).cpu().numpy().astype(np.float32)
    del denoised
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    save_nifti(magn_np_out, output_path=magn_path, reference_img=magnitude_file)
    del magn_np_out

    if phase_path is not None and phase_out_np is not None:
        save_nifti(
            phase_out_np,
            output_path=phase_path,
            reference_img=phase_file if phase_file is not None else magnitude_file,
        )
        del phase_out_np

    if cfg.save_gfactor_map:
        gfactor_file = out_dir / f"{out_prefix.name}_gfactor{ext}"
        save_nifti(
            gfactor.detach().cpu().numpy().astype(np.float32),
            output_path=gfactor_file,
            reference_img=magnitude_file,
        )

    meta = {
        "magnitude_file": str(magnitude_file),
        "phase_file": str(phase_file) if phase_file is not None else None,
        "output_prefix": str(output_prefix),
        "shape": [int(nx), int(ny), int(nz), int(nt)],
        "device": str(dev),
        "absolute_scale": absolute_scale,
        "config": {
            "temporal_phase": cfg.temporal_phase,
            "phase_filter_width": cfg.phase_filter_width,
            "noise_volume_last": cfg.noise_volume_last,
            "factor_error": cfg.factor_error,
            "nordic": cfg.nordic,
            "mp_mode": cfg.mp_mode,
            "magnitude_only": cfg.magnitude_only,
            "kernel_size_pca": list(kernel_pca),
            "kernel_size_gfactor": list(cfg.kernel_size_gfactor),
            "patch_overlap": cfg.patch_overlap,
            "gfactor_patch_overlap": cfg.gfactor_patch_overlap,
            "phase_slice_average": cfg.phase_slice_average,
        },
        "threshold": {
            "mode": threshold_mode,
            "value": float(threshold_value),
            "measured_noise": float(measured_noise),
        },
        "diagnostics": {
            "mean_threshold_removed": mean_removed,
            "mean_energy_removed": float(torch.mean(llr_stats.energy_removed).item()),
            "mean_snr_weight": float(torch.mean(llr_stats.snr_weight).item()),
            "near_noop": bool(mean_removed < 0.1),
        },
        "outputs": {
            "magnitude": str(magn_path),
            "phase": str(phase_path) if phase_path is not None else None,
            "gfactor": str(gfactor_file) if gfactor_file is not None else None,
        },
    }

    meta_file = out_dir / f"{out_prefix.name}_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return NordicOutputs(
        magnitude_file=magn_path,
        phase_file=phase_path,
        gfactor_file=gfactor_file,
        metadata_file=meta_file,
    )
