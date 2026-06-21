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
from functools import lru_cache
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
    save_residual_map: bool = False
    save_num_comps: bool = False
    make_complex_nii: bool = False
    full_dynamic_range: bool = False
    write_gzipped_niftis: bool = True
    svd_batch_size: int = 512
    decomp_method: str = "auto"
    verbose: bool = True

    # Multi-echo signal-rescue (only active on the multi-echo path; see
    # run_nordic_multiecho). A component NORDIC would remove from one echo is
    # protected if it has a correlated partner in another echo — thermal noise
    # is the only thing independent across echoes, so cross-echo correlation
    # marks signal. These have no effect on the single-echo path.
    rescue: bool = True
    rescue_band: float = 0.25  # top fraction of each echo's kill set tested
    rescue_alpha: float = 0.05  # false-rescue rate = (1 - alpha) null quantile
    # Multi-echo g-factor: default shares echo 1's map (TE-invariant geometry,
    # cleanest where signal is strongest). When thermal sigma is not actually
    # TE-invariant (per-echo bandwidth/partial-Fourier/scaling differences), set
    # this to estimate each echo's own g-factor (and hence its own thermal sigma)
    # via a per-echo MP pass — equivalent to NORDIC-normalizing each echo
    # independently, with the joint rescue still applied. Costs E g-factor passes.
    per_echo_gfactor: bool = False
    # Multi-echo QC: voxel-wise cross-echo residual correlation map (flags where
    # shared signal was removed) + an FDR (1-q) map and AFNI-header FDR curve.
    # Cheap (reuses the residuals); on by default for the multi-echo path.
    resid_qc: bool = True


@dataclass
class NordicOutputs:
    """Output paths from a NORDIC run."""

    magnitude_file: Path
    phase_file: Path | None
    gfactor_file: Path | None
    residual_file: Path | None
    num_comps_file: Path | None
    recfactor_file: Path | None
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
    # Per-voxel count of components rescued from removal by the cross-echo
    # guard. Zero everywhere on the single-echo path. Diagnostic only.
    rescued_map: torch.Tensor | None = None
    # Per-voxel recommended threshold-scale ratio in (0, 1]: the smallest factor
    # multiplier that would have kept (without rescue) the deepest cross-echo-
    # aligned component covering this voxel. 1.0 = no decrease suggested. Multi-
    # echo + nordic threshold only; None otherwise. Decrease-only by construction.
    recfactor_map: torch.Tensor | None = None


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
# Multi-echo cross-echo signal rescue
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4096)
def _rescue_null_threshold(
    n_time: int,
    k_cand: int,
    d_target: int,
    n_other: int,
    alpha: float,
    n_trials: int = 4000,
) -> float:
    """Per-patch family-wise rescue threshold under H0 (all thermal noise).

    Returns the ``(1 - alpha)`` quantile of the *max* projection norm

        max over candidate i, other echo e'   || P_{S_{e'}} v_i ||

    where each candidate ``v_i`` is a Haar-random complex unit temporal vector
    in ``C^n_time`` (the SVD right vectors of a pure-noise patch are Haar) and
    each target ``S_{e'}`` is a Haar-random ``d_target``-dim complex subspace
    (independent across the ``n_other`` other echoes). Controls the probability
    of *any* false rescue in a patch at ``alpha``.

    Content-independent — depends only on the dimensions — so it is cached and
    reused across every patch of the run. Computed on CPU (one-off, tiny) and
    seeded for reproducibility. The quantile reduction is in float64.
    """
    if k_cand <= 0 or d_target <= 0 or n_other <= 0:
        return 1.0  # nothing to compare → never rescue
    k = min(k_cand, n_time)
    d = min(d_target, n_time)
    gen = torch.Generator(device="cpu").manual_seed(0)
    stats = []
    done = 0
    chunk = 512
    while done < n_trials:
        b = min(chunk, n_trials - done)
        # Haar-orthonormal candidate columns: QR of complex Gaussian.
        cg = torch.randn(b, n_time, k, dtype=torch.complex64, generator=gen)
        cq, _ = torch.linalg.qr(cg)  # (b, n_time, k)
        best = torch.zeros(b, k)
        for _ in range(n_other):
            qg = torch.randn(b, n_time, d, dtype=torch.complex64, generator=gen)
            qq, _ = torch.linalg.qr(qg)  # (b, n_time, d)
            proj = qq.mH @ cq  # (b, d, k)
            pn = (proj.abs() ** 2).sum(dim=1).sqrt()  # (b, k) per-candidate proj norm
            best = torch.maximum(best, pn)
        stats.append(best.amax(dim=1))  # (b,) max over candidates
        done += b
    stat = torch.cat(stats).double()
    return float(torch.quantile(stat, 1.0 - alpha).item())


def _residual_xcorr_qc(
    residuals: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Voxel-wise cross-echo residual correlation QC for the multi-echo path.

    The residual of each echo is the reconstruction of the *removed* components.
    Thermal noise is independent across echoes, so under correct denoising the
    residuals are uncorrelated voxel-by-voxel. A voxel where two echoes' removed
    signals **correlate** is a voxel where shared (non-thermal) signal was taken
    out — over-removal. Because a removed component's spatial loading can be sharp
    (a vein, an edge), this shows up as focal single-voxel peaks, not whole
    patches, which is why the per-voxel view matters.

    Correlation is on the **magnitude** of each residual time course, not the
    complex residual. The shared signal NORDIC over-removes lives in the
    magnitude ("same time course, different per-echo weight"); the complex
    Hermitian correlation also fires on shared *phase* (B0/off-resonance evolves
    coherently across echoes and survives dd-phase), which inflates the
    correlation brain-wide even for independent thermal magnitude noise. Magnitude
    targets the over-removal we care about and drops the phase confound.

    Computed in the g-factor-normalized algorithm space (cleanest null).
    Returns ``(max_r, tstat, dof)``:

    - ``max_r`` (nx, ny, nz): max over echo pairs of ``|Pearson r|`` of the
      magnitude residuals — for eyeballing peaks.
    - ``tstat`` (nx, ny, nz): ``r`` mapped to a Student-t, ``t = r·√(dof/(1-r²))``
      with ``dof = T - 2``, so it can be tagged ``fitt`` and FDR-thresholded.
    - ``dof``: ``T - 2``.

    Voxels with ~zero residual energy in any echo (outside the brain / fully
    kept) get r = 0.
    """
    E = len(residuals)
    nx, ny, nz, nt = residuals[0].shape
    dev = residuals[0].device
    # (E, V, T) magnitude residuals, demeaned over time and unit-normalized.
    flat = []
    for r in residuals:
        a = r.reshape(-1, nt).abs()  # magnitude → real
        a = a - a.mean(dim=1, keepdim=True)
        norm = a.pow(2).sum(dim=1, keepdim=True).sqrt()
        a = a / norm.clamp(min=1e-12)
        flat.append((a, norm.squeeze(1)))
    max_r = torch.zeros(nx * ny * nz, device=dev)
    for e in range(E):
        ae, ne = flat[e]
        for ep in range(e + 1, E):
            aep, nep = flat[ep]
            # |Pearson r| of the magnitude residuals (scale-invariant).
            r = (ae * aep).sum(dim=1).abs()
            valid = (ne > 1e-12) & (nep > 1e-12)
            r = torch.where(valid, r, torch.zeros_like(r))
            max_r = torch.maximum(max_r, r)
    max_r = max_r.clamp(0.0, 0.999999).reshape(nx, ny, nz)
    dof = max(1, nt - 2)
    tstat = max_r * torch.sqrt(dof / (1.0 - max_r.pow(2)).clamp(min=1e-12))
    return max_r, tstat, dof


def _save_residual_qc(
    max_r: torch.Tensor,
    tstat: torch.Tensor,
    dof: int,
    output_prefix: str,
    reference_img: str,
    cfg: NordicConfig,
    brain_mask: torch.Tensor | None = None,
) -> dict:
    """Write the cross-echo residual QC as one 4-sub-brick file and return a summary.

    Single dataset-level 4D map ``_resid_xcorr`` with sub-bricks:
      0 ``xcorr_pearson_r``  max-over-pairs Pearson r (eyeball the focal peaks),
      1 ``xcorr_tstat``      r→t (tagged ``Ttest(dof)`` so AFNI thresholds it),
      2 ``xcorr_1minus_q``   ``1 - q`` (BH-FDR), high = significant.
    The t-stat sub-brick carries an injected AFNI FDRCURVE, so AFNI's GUI /
    ``fdrval`` read q from it directly; the ``1-q`` brick covers other viewers.

    ``brain_mask`` (if given) restricts the FDR multiple-comparison set and the
    summary to in-brain voxels — out-of-head residual structure is artifact in
    noise and would otherwise dominate the q-map and the reported fraction.
    """
    from fastfuncstuff.stats.fdr import (
        add_fdrcurves_to_nifti,
        compute_fdr_curve,
        fdr_qvalues,
    )

    out_dir = Path(output_prefix).parent
    name = Path(output_prefix).name
    ext = ".nii.gz" if cfg.write_gzipped_niftis else ".nii"
    # FDR runs through scipy on CPU; the QC maps are tiny volumes, so move them
    # off-device once here (avoids MPS/CUDA round-trip surprises downstream).
    max_r = max_r.cpu()
    tstat = tstat.cpu()
    # FDR set: in-brain when available (the correct comparison set), else nonzero.
    mask = tstat > 0
    if brain_mask is not None:
        mask = mask & brain_mask.cpu()
    q = fdr_qvalues(tstat, stat_code="fitt", dof=float(dof), mask=mask)
    one_minus_q = torch.where(torch.isfinite(q), 1.0 - q, torch.zeros_like(q))

    # Stack into (X, Y, Z, 3); label + stat-tag the t sub-brick (AFNI code 3 =
    # Ttest, one param = dof) so AFNI can threshold and FDR it.
    vol = np.stack(
        [
            max_r.numpy().astype(np.float32),
            tstat.numpy().astype(np.float32),
            one_minus_q.numpy().astype(np.float32),
        ],
        axis=-1,
    )
    xcorr_path = out_dir / f"{name}_resid_xcorr{ext}"
    save_nifti(
        vol,
        output_path=xcorr_path,
        reference_img=reference_img,
        brick_labels=["xcorr_pearson_r", "xcorr_tstat", "xcorr_1minus_q"],
        brick_stataux={1: (3, (float(dof),))},
    )

    # AFNI-native q on the t sub-brick (index 1). save_nifti created the AFNI
    # extension via the labels/stataux, so injection works even for plain-NIfTI
    # inputs; the 1-q brick is the fallback, so failure is non-fatal.
    fdr_in_header = False
    try:
        curve = compute_fdr_curve(tstat, "fitt", float(dof), mask=mask)
        add_fdrcurves_to_nifti(xcorr_path, {1: curve})
        fdr_in_header = True
    except Exception as exc:  # noqa: BLE001 — QC is best-effort
        if cfg.verbose:
            print(f"  Residual QC: FDR curve not injected into header ({exc}); 1-q brick written.")

    qf = q.flatten()
    qf = qf[torch.isfinite(qf)]
    n_valid = int(qf.numel())
    frac_q05 = float((qf < 0.05).float().mean().item()) if n_valid else 0.0
    # max_r reported over the same set as the FDR (in-brain when masked).
    rsel = max_r[mask] if brain_mask is not None else max_r
    max_r_report = float(rsel.max().item()) if rsel.numel() else 0.0
    return {
        "max_r": max_r_report,
        "max_r_whole_volume": float(max_r.max().item()),
        "dof": int(dof),
        "n_valid_voxels": n_valid,
        "frac_q_lt_0.05": frac_q05,
        "in_brain": brain_mask is not None,
        "fdr_in_header": fdr_in_header,
        "map": str(xcorr_path),
        "sub_bricks": ["xcorr_pearson_r", "xcorr_tstat", "xcorr_1minus_q"],
    }


def _llr_denoise_multiecho(
    data_echoes: list[torch.Tensor],
    kernel_size: tuple[int, int, int],
    patch_overlap: int,
    threshold_mode: str,
    threshold_values: list[float],
    *,
    rescue: bool,
    rescue_band: float,
    rescue_alpha: float,
    verbose: bool,
    svd_batch_size: int = 512,
    device: torch.device | None = None,
) -> list[tuple[torch.Tensor, _LLRStats]]:
    """Per-echo NORDIC LLR with a cross-echo signal-rescue guard.

    Each echo is denoised exactly as the single-echo path (same threshold, same
    reconstruction), but before a below-threshold component is removed it is
    tested against the *other* echoes. A component in echo e's **marginal kill
    band** (the top ``rescue_band`` fraction of its kill set, by singular value)
    that aligns — above the null threshold — with another echo's signal subspace
    (that echo's keep set ∪ its marginal band) is **promoted back to keep**.
    Thermal noise is independent across echoes, so cross-echo alignment marks
    signal. The guard only ever moves components kill→keep; with ``rescue=False``
    (or no alignment found) the result is identical to E independent runs.

    Returns one ``(recon, stats)`` per echo, in input order. Patches share
    corners across echoes (same grid), so the candidate→target test is a small
    batched matmul on the temporal singular vectors.
    """
    E = len(data_echoes)
    if device is None:
        device = data_echoes[0].device
    data_dev = data_echoes[0].device
    cross_device = device != data_dev

    nx, ny, nz, nt = data_echoes[0].shape
    wx = min(kernel_size[0], nx)
    wy = min(kernel_size[1], ny)
    wz = min(kernel_size[2], nz)
    M = wx * wy * wz
    N = nt
    K = min(M, N)

    sx = max(1, wx // max(1, patch_overlap))
    sy = max(1, wy // max(1, patch_overlap))
    sz = max(1, wz // max(1, patch_overlap))
    xs = _build_patch_starts(nx, wx, sx)
    ys = _build_patch_starts(ny, wy, sy)
    zs = _build_patch_starts(nz, wz, sz)
    corners = [(x0, y0, z0) for x0 in xs for y0 in ys for z0 in zs]
    total = len(corners)

    # Accumulators (one recon per echo; geometry weight shared).
    recon_acc = [
        torch.zeros(nx, ny, nz, nt, dtype=data_echoes[e].dtype, device=device) for e in range(E)
    ]
    weight = torch.zeros((nx, ny, nz), dtype=torch.float32, device=device)
    removed_map = [torch.zeros_like(weight) for _ in range(E)]
    rescued_map = [torch.zeros_like(weight) for _ in range(E)]
    # Recommended-factor map: per voxel, the patch-mean threshold-scale ratio
    # (lowest cross-echo-aligned sigma / lambda). <1 suggests lowering the factor,
    # >1 raising it, 1 no change. Accumulated as a sum here, divided by the patch
    # weight at the end. Only meaningful for the nordic threshold + rescue.
    want_recfactor = threshold_mode == "nordic" and rescue and E >= 2
    recfactor_map = [torch.zeros_like(weight) for _ in range(E)] if want_recfactor else None

    # Per-patch voxel offsets (same machinery as _llr_denoise).
    _ox, _oy, _oz = torch.arange(wx), torch.arange(wy), torch.arange(wz)
    gx, gy, gz = torch.meshgrid(_ox, _oy, _oz, indexing="ij")
    dx, dy, dz = gx.ravel(), gy.ravel(), gz.ravel()
    local_offsets_flat = dx * (ny * nz) + dy * nz + dz  # (M,)
    corner_xs = torch.tensor([c[0] for c in corners], dtype=torch.long)
    corner_ys = torch.tensor([c[1] for c in corners], dtype=torch.long)
    corner_zs = torch.tensor([c[2] for c in corners], dtype=torch.long)
    corners_flat = corner_xs * (ny * nz) + corner_ys * nz + corner_zs

    recon_flat = [r.reshape(-1, nt) for r in recon_acc]
    weight_flat = weight.reshape(-1)
    removed_flat = [m.reshape(-1) for m in removed_map]
    rescued_flat = [m.reshape(-1) for m in rescued_map]
    recfactor_flat = [m.reshape(-1) for m in recfactor_map] if recfactor_map is not None else None
    ones_BM = torch.ones(svd_batch_size * M, device=device)

    idx_arange = torch.arange(K, device=device)

    pbar = tqdm(total=total, desc="ME-LLR patches", unit="patch") if verbose else None

    for batch_start in range(0, total, svd_batch_size):
        B = min(svd_batch_size, total - batch_start)
        bx = corner_xs[batch_start : batch_start + B]
        by = corner_ys[batch_start : batch_start + B]
        bz = corner_zs[batch_start : batch_start + B]
        xi = (bx[:, None] + dx[None, :]).to(data_dev)  # (B, M)
        yi = (by[:, None] + dy[None, :]).to(data_dev)
        zi = (bz[:, None] + dz[None, :]).to(data_dev)
        b_corners_flat = corners_flat[batch_start : batch_start + B]
        flat_b = (b_corners_flat[:, None] + local_offsets_flat[None, :]).reshape(-1).to(device)

        # ---- Pass 1: per-echo SVD → temporal vectors + base keep count ----
        vh_list: list[torch.Tensor] = []  # (B, K, N) rows = right singular vectors
        n_keep_list: list[torch.Tensor] = []  # (B,) base keep count (prefix)
        s0_list: list[torch.Tensor] = []  # (B, K) singular values (recfactor only)
        for e in range(E):
            mats = data_echoes[e][xi, yi, zi, :]
            if cross_device:
                mats = mats.to(device)
            # Right singular vectors (rows of vh) carry the temporal structure;
            # U is not needed — reconstruction uses the projection X V V^H.
            _, s, vh = torch.linalg.svd(mats, full_matrices=False)
            del mats
            s0 = s.abs()
            if threshold_mode == "mp":
                cuts, _ = _mp_hard_index_batch(s0, m=M, n=N)
                n_keep = cuts.to(device)
            else:
                n_keep = (s0 >= threshold_values[e]).sum(dim=1).to(device)
            vh_list.append(vh)
            n_keep_list.append(n_keep.clamp(min=0, max=K))
            s0_list.append(s0.to(device) if want_recfactor else None)
            del s

        # ---- Base keep / marginal-band masks (B, K) ----
        keep_masks = [idx_arange[None, :] < nk[:, None] for nk in n_keep_list]
        n_kill = [K - nk for nk in n_keep_list]
        band_counts = [torch.ceil(rescue_band * nkl.float()).long() for nkl in n_kill]
        band_masks = [
            (idx_arange[None, :] >= n_keep_list[e][:, None])
            & (idx_arange[None, :] < (n_keep_list[e] + band_counts[e])[:, None])
            for e in range(E)
        ]
        target_masks = [keep_masks[e] | band_masks[e] for e in range(E)]
        aug_masks = [keep_masks[e].clone() for e in range(E)]
        rescued_counts = [torch.zeros(B, device=device) for _ in range(E)]
        patch_ratios = [torch.ones(B, device=device) for _ in range(E)]

        # ---- Pass 2: cross-echo rescue ----
        if rescue and E >= 2:
            for e in range(E):
                cand_e = band_masks[e]  # (B, K)
                if not bool(cand_e.any()):
                    continue
                proj_best = torch.zeros(B, K, device=device)  # per-candidate max proj norm
                d_target_rep = 0
                for ep in range(E):
                    if ep == e:
                        continue
                    corr = vh_list[e] @ vh_list[ep].mH  # (B, K, K) complex inner products
                    tgt = target_masks[ep].float()  # (B, K) cols to keep
                    p2 = (corr.abs() ** 2) * tgt[:, None, :]
                    pn = p2.sum(dim=2).sqrt()  # (B, K) proj norm of each comp onto e' target
                    proj_best = torch.maximum(proj_best, pn)
                    d_target_rep = max(d_target_rep, int(target_masks[ep].sum(dim=1).max().item()))
                    del corr, p2, pn
                # Conservative per-batch null dims (batch-max candidate / target sizes).
                k_cand_rep = int(band_counts[e].max().item())
                thr = _rescue_null_threshold(
                    n_time=N,
                    k_cand=k_cand_rep,
                    d_target=d_target_rep,
                    n_other=E - 1,
                    alpha=rescue_alpha,
                )
                resc = cand_e & (proj_best > thr)  # (B, K)
                aug_masks[e] = aug_masks[e] | resc
                rescued_counts[e] = resc.sum(dim=1).float()
                if want_recfactor:
                    # Bidirectional suggested-threshold ratio per patch, BOUNDED to a
                    # +/-band window around the current cut so it can't run away:
                    # ideal cut = sigma of the lowest cross-echo-aligned component in
                    # [n_keep - band, n_keep + band). An aligned band comp (below
                    # lambda) -> ratio<1 (suggest lower); only the lowest *kept* comps
                    # non-aligned (noise-like) -> ratio>1 (suggest raise), capped at
                    # the window edge; 1.0 = no change. The window cap matters: without
                    # it a near-rank-1 patch jumps ideal to the top sigma (huge ratio).
                    lam = float(threshold_values[e])
                    lo = (n_keep_list[e] - band_counts[e]).clamp(min=0)  # (B,)
                    hi = (n_keep_list[e] + band_counts[e]).clamp(max=K)  # (B,)
                    window = (idx_arange[None, :] >= lo[:, None]) & (
                        idx_arange[None, :] < hi[:, None]
                    )
                    aligned = window & (proj_best > thr)
                    big = torch.full_like(s0_list[e], float("inf"))
                    min_aligned = torch.where(aligned, s0_list[e], big).amin(dim=1)  # (B,)
                    ideal = torch.where(
                        torch.isfinite(min_aligned),
                        min_aligned,
                        torch.full_like(min_aligned, lam),
                    )
                    patch_ratios[e] = ideal / max(lam, 1e-12)

        # ---- Pass 3: reconstruct each echo with its augmented mask ----
        weight_flat.index_add_(0, flat_b, ones_BM[: B * M])
        for e in range(E):
            mats = data_echoes[e][xi, yi, zi, :]
            if cross_device:
                mats = mats.to(device)
            # X_denoised = X V_k V_k^H, with V_k = kept right singular vectors.
            # Zero non-kept rows of vh, then P = vh_k^H vh_k is the projector.
            vh_k = vh_list[e] * aug_masks[e][:, :, None]  # (B, K, N)
            proj = vh_k.mH @ vh_k  # (B, N, N)
            del vh_k
            recon = mats @ proj  # (B, M, N)
            del mats, proj
            recon_flat[e].index_add_(0, flat_b, recon.reshape(-1, nt))
            del recon
            n_removed = (K - aug_masks[e].sum(dim=1)).float()  # (B,)
            removed_flat[e].index_add_(0, flat_b, n_removed.repeat_interleave(M))
            rescued_flat[e].index_add_(0, flat_b, rescued_counts[e].repeat_interleave(M))
            if recfactor_flat is not None:
                # Weighted mean over covering patches (bidirectional: decreases and
                # increases both contribute; 1.0 = no suggestion).
                recfactor_flat[e].index_add_(0, flat_b, patch_ratios[e].repeat_interleave(M))

        del vh_list, s0_list
        if pbar is not None:
            pbar.update(B)

    if pbar is not None:
        pbar.close()

    w = torch.clamp(weight, min=1.0)
    uncovered = weight == 0
    results: list[tuple[torch.Tensor, _LLRStats]] = []
    zeros_w = torch.zeros_like(w)
    for e in range(E):
        recon_out = recon_acc[e] / w[..., None]
        rec_map = None
        if recfactor_map is not None:
            rec_map = recfactor_map[e] / w
            rec_map[uncovered] = 1.0  # no patch covered → no suggestion
        stats = _LLRStats(
            weight=w,
            noise_map=zeros_w,
            threshold_map=removed_map[e] / w,
            energy_removed=zeros_w,
            snr_weight=zeros_w,
            rescued_map=rescued_map[e] / w,
            recfactor_map=rec_map,
        )
        results.append((recon_out, stats))
    return results


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


@dataclass
class _PreppedEcho:
    """Output of NORDIC preprocessing (steps 1-6) for a single echo.

    Shared by the single-echo (run_nordic) and multi-echo
    (run_nordic_multiecho) paths so prep/finalize have one source of truth.
    """

    ksp2: torch.Tensor | None  # prepped complex data, length nt_keep
    gfactor: torch.Tensor
    mp_unit: torch.Tensor | None
    dd_phase: torch.Tensor | None
    threshold_mode: str
    threshold_value: float
    measured_noise: float
    absolute_scale: float
    kernel_pca: tuple[int, int, int]
    is_complex: bool
    phase_in_used: bool
    n_noise_vols: int
    orig_shape: tuple[int, int, int, int]  # nx, ny, nz, nt (pre-trim)
    nt_keep: int


def _prepare_echo(
    magnitude_file: str,
    phase_file: str | None,
    cfg: NordicConfig,
    dev: torch.device,
    *,
    gfactor_override: torch.Tensor | None = None,
    echo_label: str = "",
) -> _PreppedEcho:
    """NORDIC preprocessing for one echo: load -> phase -> g-factor -> threshold.

    ``gfactor_override`` lets later echoes reuse echo 1's g-factor map (estimated
    where signal is strongest) instead of re-running the MP g-factor pass —
    g-factor is coil/acceleration geometry and TE-invariant, so this is both a
    speedup and the default for the multi-echo path. Thermal sigma is always
    measured per echo (own noise volume, else 1.0): it is not shared, because
    each echo's threshold is its own keep/kill decision and the cross-echo rescue
    compares unit-norm temporal vectors (scale-invariant).
    """
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

    # Trailing noise-only volumes calibrate the threshold but must not enter
    # the patch/SVD math — their statistics differ from the signal volumes and
    # would corrupt the low-rank estimate. Size the kernel, the threshold, and
    # the saved output to the signal-only length; the noise volumes are dropped
    # once measured_noise has been computed (step 5b).
    n_noise_vols = (
        cfg.noise_volume_last if (cfg.noise_volume_last > 0 and nt > cfg.noise_volume_last) else 0
    )
    nt_keep = nt - n_noise_vols

    # Kernel sizes
    if cfg.kernel_size_pca is None:
        kernel_pca = _default_kernel_size_pca(nt_keep, n_slices=nz)
    else:
        kernel_pca = cfg.kernel_size_pca

    if cfg.verbose:
        print(f"\nNORDIC denoising{echo_label}")
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

    if gfactor_override is not None:
        # Multi-echo: reuse echo 1's g-factor map (estimated where signal is
        # strongest). g-factor is TE-invariant, so this is both correct and a
        # large speedup (skips the per-echo MP pass).
        gfactor = gfactor_override.to(dev)
    elif cfg.save_gfactor_map or cfg.nordic or cfg.mp_mode == 1:
        g_nvol = min(max(1, cfg.gfactor_nvols), nt_keep)
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
    # 5b. Drop the trailing noise volumes now that the threshold is
    #     calibrated, so they never enter the patch/SVD. dd_phase is
    #     trimmed in lockstep so the inverse phase reapplication in step 8
    #     lines up with the denoised length.
    # ------------------------------------------------------------------
    if n_noise_vols > 0:
        KSP2 = KSP2[..., :nt_keep]
        if dd_phase is not None:
            dd_phase = dd_phase[..., :nt_keep]
        if cfg.verbose:
            print(f"  Trimmed {n_noise_vols} noise volume(s) before denoising: {nt} -> {nt_keep}")

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
            n_timepoints=nt_keep,
            measured_noise=measured_noise,
            factor_error=cfg.factor_error,
            is_complex=is_complex,
            device=dev,
        )
        if cfg.verbose:
            print(f"  NORDIC lambda: {threshold_value:.6g}")

    return _PreppedEcho(
        ksp2=KSP2,
        gfactor=gfactor,
        mp_unit=mp_unit,
        dd_phase=dd_phase,
        threshold_mode=threshold_mode,
        threshold_value=threshold_value,
        measured_noise=measured_noise,
        absolute_scale=absolute_scale,
        kernel_pca=kernel_pca,
        is_complex=is_complex,
        phase_in_used=phase_in_used,
        n_noise_vols=n_noise_vols,
        orig_shape=(nx, ny, nz, nt),
        nt_keep=nt_keep,
    )


def _finalize_echo(
    prepped: _PreppedEcho,
    denoised: torch.Tensor,
    llr_stats: _LLRStats,
    residual: torch.Tensor | None,
    mean_removed: float,
    output_prefix: str,
    cfg: NordicConfig,
    dev: torch.device,
    magnitude_file: str,
    phase_file: str | None,
    *,
    extra_meta: dict | None = None,
) -> NordicOutputs:
    """Undo NORDIC transforms (step 8) and write outputs (step 9) for one echo.

    ``extra_meta`` is merged into the metadata JSON (multi-echo uses it to record
    echo index and rescue counts).
    """
    gfactor = prepped.gfactor
    dd_phase = prepped.dd_phase
    mp_unit = prepped.mp_unit
    absolute_scale = prepped.absolute_scale
    phase_in_used = prepped.phase_in_used
    nx, ny, nz, nt = prepped.orig_shape
    nt_keep = prepped.nt_keep
    n_noise_vols = prepped.n_noise_vols
    kernel_pca = prepped.kernel_pca
    threshold_mode = prepped.threshold_mode
    threshold_value = prepped.threshold_value
    measured_noise = prepped.measured_noise

    # ------------------------------------------------------------------
    # 8. Undo transformations: gfactor, meanphase, DD_phase, scale
    # ------------------------------------------------------------------
    # All operations are in-place to avoid allocating a second 4-D copy.
    # MATLAB: IMG2 *= gfactor; IMG2 *= exp(i*angle(DD_phase)); IMG2 *= exp(i*angle(meanphase))
    denoised *= gfactor[..., None]
    if residual is not None:
        residual *= gfactor.to(residual.device)[..., None]

    if dd_phase is not None:
        _dd_phase_multiply_inplace(denoised, dd_phase, conjugate=True)
        if residual is not None:
            _dd_phase_multiply_inplace(residual, dd_phase.to(residual.device), conjugate=True)
        del dd_phase
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    if mp_unit is not None:
        _restore_meanphase(denoised, mp_unit)
        if residual is not None:
            _restore_meanphase(residual, mp_unit.to(residual.device))
    del mp_unit

    # MATLAB: IMG2 *= ABSOLUTE_SCALE
    denoised *= absolute_scale
    if residual is not None:
        residual *= absolute_scale

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

    gfactor_file: Path | None = None
    if cfg.save_gfactor_map:
        gfactor_file = out_dir / f"{out_prefix.name}_gfactor{ext}"
        save_nifti(
            gfactor.detach().cpu().numpy().astype(np.float32),
            output_path=gfactor_file,
            reference_img=magnitude_file,
        )

    residual_file: Path | None = None
    if residual is not None:
        residual_file = out_dir / f"{out_prefix.name}_residual{ext}"
        save_nifti(
            torch.abs(residual).cpu().numpy().astype(np.float32),
            output_path=residual_file,
            reference_img=magnitude_file,
        )
        del residual

    rescued_file: Path | None = None
    if llr_stats.rescued_map is not None and cfg.rescue:
        rescued_file = out_dir / f"{out_prefix.name}_rescued{ext}"
        save_nifti(
            llr_stats.rescued_map.detach().cpu().numpy().astype(np.float32),
            output_path=rescued_file,
            reference_img=magnitude_file,
        )

    # Per-voxel count of components removed (patch-averaged, so fractional):
    # shows where energy is being taken from. This is the same threshold_map
    # that feeds mean_threshold_removed.
    num_comps_file: Path | None = None
    if cfg.save_num_comps:
        num_comps_file = out_dir / f"{out_prefix.name}_numcomps{ext}"
        save_nifti(
            llr_stats.threshold_map.detach().cpu().numpy().astype(np.float32),
            output_path=num_comps_file,
            reference_img=magnitude_file,
        )

    # Per-voxel suggested factor_error: the threshold-scale ratio (the lowest
    # cross-echo-aligned component's sigma / lambda) scaled by the factor used.
    # < factor_error where signal sits below threshold (decrease); > where the
    # lowest kept components are noise-like (increase); = factor_error for no change.
    recfactor_file: Path | None = None
    if llr_stats.recfactor_map is not None and cfg.rescue:
        recfactor_file = out_dir / f"{out_prefix.name}_recfactor{ext}"
        save_nifti(
            (cfg.factor_error * llr_stats.recfactor_map).detach().cpu().numpy().astype(np.float32),
            output_path=recfactor_file,
            reference_img=magnitude_file,
        )

    meta = {
        "magnitude_file": str(magnitude_file),
        "phase_file": str(phase_file) if phase_file is not None else None,
        "output_prefix": str(output_prefix),
        "shape": [int(nx), int(ny), int(nz), int(nt)],
        "output_shape": [int(nx), int(ny), int(nz), int(nt_keep)],
        "noise_volumes_trimmed": int(n_noise_vols),
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
            "residual": str(residual_file) if residual_file is not None else None,
            "rescued": str(rescued_file) if rescued_file is not None else None,
            "num_comps": str(num_comps_file) if num_comps_file is not None else None,
            "recfactor": str(recfactor_file) if recfactor_file is not None else None,
        },
    }
    if extra_meta is not None:
        meta.update(extra_meta)

    meta_file = out_dir / f"{out_prefix.name}_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return NordicOutputs(
        magnitude_file=magn_path,
        phase_file=phase_path,
        gfactor_file=gfactor_file,
        residual_file=residual_file,
        num_comps_file=num_comps_file,
        recfactor_file=recfactor_file,
        metadata_file=meta_file,
    )


def _llr_main_pass(
    prepped: _PreppedEcho,
    cfg: NordicConfig,
    dev: torch.device,
) -> tuple[torch.Tensor, _LLRStats, torch.Tensor | None, float]:
    """Step 7: memory guard + main single-echo LLR. Returns
    (denoised, stats, residual, mean_removed). Consumes prepped.ksp2."""
    KSP2 = prepped.ksp2
    assert KSP2 is not None
    kernel_pca = prepped.kernel_pca

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
        threshold_mode=prepped.threshold_mode,
        threshold_value=prepped.threshold_value,
        verbose=cfg.verbose,
        svd_batch_size=cfg.svd_batch_size,
        decomp_method=cfg.decomp_method,
        device=dev,
    )

    residual: torch.Tensor | None = None
    if cfg.save_residual_map:
        residual = KSP2.to(denoised.device) - denoised
    del KSP2
    prepped.ksp2 = None
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
    return denoised, llr_stats, residual, mean_removed


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

    prepped = _prepare_echo(magnitude_file, phase_file, cfg, dev)
    denoised, llr_stats, residual, mean_removed = _llr_main_pass(prepped, cfg, dev)
    return _finalize_echo(
        prepped,
        denoised,
        llr_stats,
        residual,
        mean_removed,
        output_prefix,
        cfg,
        dev,
        magnitude_file,
        phase_file,
    )


def run_nordic_multiecho(
    magnitude_files: list[str],
    phase_files: list[str | None] | None,
    output_prefix: str,
    config: NordicConfig | None = None,
    device: torch.device | None = None,
) -> list[NordicOutputs]:
    """Multi-echo NORDIC: denoise each echo independently, guarded by a
    cross-echo signal-rescue step (see ``_llr_denoise_multiecho``).

    By default the g-factor map is estimated once on echo 1 and shared (it is
    TE-invariant coil/acceleration geometry, cleanest where signal is strongest);
    ``cfg.per_echo_gfactor`` instead estimates each echo's own g-factor for the
    case where thermal sigma is not TE-invariant. Thermal sigma is measured per
    echo either way (own noise volume, else 1.0) — sharing it is unnecessary
    because the cross-echo rescue compares scale-invariant unit temporal vectors.
    Outputs one denoised file per echo (``{prefix}_echo-NN``), each with its own
    metadata + rescue-count map. Returns one ``NordicOutputs`` per echo, in input
    order.
    """
    cfg = config or NordicConfig()
    dev = (
        device if device is not None else get_device("cuda" if torch.cuda.is_available() else None)
    )
    E = len(magnitude_files)
    if E < 2:
        raise ValueError("run_nordic_multiecho requires >= 2 echoes")
    if phase_files is None:
        phase_files = [None] * E
    if len(phase_files) != E:
        raise ValueError(f"got {E} magnitude files but {len(phase_files)} phase files")

    # 1. Prep echo 1 fully. Its g-factor map is the default shared map; sigma is
    #    always per echo (echo 1's own here).
    p0 = _prepare_echo(magnitude_files[0], phase_files[0], cfg, dev, echo_label=" (echo 1)")
    prepped = [p0]
    shared_gfactor = None if cfg.per_echo_gfactor else p0.gfactor

    # 2. Prep echoes 2..E. Default: reuse echo 1's g-factor map (TE-invariant,
    #    fast). With per_echo_gfactor: estimate each echo's own g-factor (and so
    #    its own thermal sigma) for non-TE-invariant thermal noise. Sigma is
    #    measured per echo regardless (own noise volume, else 1.0).
    for e in range(1, E):
        prepped.append(
            _prepare_echo(
                magnitude_files[e],
                phase_files[e],
                cfg,
                dev,
                gfactor_override=shared_gfactor,
                echo_label=f" (echo {e + 1})",
            )
        )

    # 3. Memory guard: the joint pass holds E echoes' data + E recon
    #    accumulators. Offload echo data to CPU and stream patches if the
    #    estimated GPU footprint exceeds what's available (accumulators stay on
    #    GPU; the LLR's cross-device path transfers patches per batch).
    data_echoes: list[torch.Tensor] = [p.ksp2 for p in prepped]  # type: ignore[misc]
    if dev.type == "cuda":
        mem_est = estimate_nordic_llr_memory(
            shape=data_echoes[0].shape,
            kernel_size=prepped[0].kernel_pca,
            svd_batch_size=cfg.svd_batch_size,
            dtype_bytes=data_echoes[0].element_size(),
            return_recon=True,
            n_echoes=E,
        )
        avail = get_available_memory(dev)
        if mem_est["total"] > avail:
            if cfg.verbose:
                print(
                    f"  Memory guard: {E}-echo LLR estimate ~"
                    f"{mem_est['total'] / 1024**3:.2f} GiB > {avail / 1024**3:.2f} GiB; "
                    "offloading echo data to CPU."
                )
            data_echoes = [d.cpu() for d in data_echoes]
            for p in prepped:
                p.ksp2 = None
            torch.cuda.empty_cache()

    # 4. Joint multi-echo LLR with cross-echo rescue.
    results = _llr_denoise_multiecho(
        data_echoes,
        kernel_size=prepped[0].kernel_pca,
        patch_overlap=max(1, cfg.patch_overlap),
        threshold_mode=prepped[0].threshold_mode,
        threshold_values=[p.threshold_value for p in prepped],
        rescue=cfg.rescue,
        rescue_band=cfg.rescue_band,
        rescue_alpha=cfg.rescue_alpha,
        verbose=cfg.verbose,
        svd_batch_size=cfg.svd_batch_size,
        device=dev,
    )
    # Per-echo residual = original − denoised, computed before the echo data is
    # freed. Each residual goes through the same inverse transforms as the
    # denoised output in _finalize_echo, so the saved map is in native units and
    # directly comparable across echoes (a voxel correlated with its later-echo
    # counterpart in the residual flags over-removed shared signal).
    residuals: list[torch.Tensor | None] = [None] * E
    if cfg.save_residual_map or cfg.resid_qc:
        for e in range(E):
            denoised_e = results[e][0]
            residuals[e] = data_echoes[e].to(denoised_e.device) - denoised_e
    del data_echoes
    for p in prepped:
        p.ksp2 = None
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # Brain mask (dilated automask on echo 1) so QC and the factor suggestion are
    # reported over the head, not out-of-head voxels where residual structure is
    # artifact-in-noise (a real effect, but not signal we removed and misleading
    # if it dominates the summary). Best-effort: fall back to whole-volume.
    brain_mask: torch.Tensor | None = None
    if cfg.resid_qc and E >= 2:
        try:
            from fastfuncstuff.processing.mask import automask

            mag0 = load_nifti(magnitude_files[0]).get_fdata(dtype=np.float32)
            mag0 = np.abs(mag0)
            mag0 = mag0.mean(-1) if mag0.ndim == 4 else mag0  # (nx, ny, nz)
            bm = automask(torch.from_numpy(mag0).float(), dilate_extra=3, verbose=False)
            brain_mask = bm.to(torch.bool).cpu()
            if not bool(brain_mask.any()):
                brain_mask = None
        except Exception as exc:  # noqa: BLE001 — QC is best-effort
            if cfg.verbose:
                print(f"  Residual QC: automask failed ({exc}); reporting whole-volume.")
            brain_mask = None

    # Voxel-wise cross-echo residual correlation QC (algorithm space). Done
    # before _finalize_echo applies per-echo inverse transforms to the residuals.
    qc_summary: dict | None = None
    if cfg.resid_qc and E >= 2:
        max_r, tstat, dof = _residual_xcorr_qc([r for r in residuals if r is not None])
        qc_summary = _save_residual_qc(
            max_r, tstat, dof, output_prefix, magnitude_files[0], cfg, brain_mask=brain_mask
        )
        if cfg.verbose and qc_summary is not None:
            where = "in-brain" if brain_mask is not None else "whole-volume"
            print(
                f"  Residual QC ({where}): {qc_summary['frac_q_lt_0.05'] * 100:.2f}% of voxels "
                f"show cross-echo residual correlation at q<0.05 (max r {qc_summary['max_r']:.3f})"
            )

    # Suggested factor_error from the per-voxel ratio maps, summarized in-brain.
    # Bidirectional but a suggestion only (never auto-applied). A global factor
    # can't be raised to satisfy the most-noise patch, so we report the two sides
    # separately: a representative decrease among the over-removed voxels and a
    # representative increase among the noise-like-kept voxels, with the fraction
    # of voxels on each side. The per-voxel _recfactor map carries the spatial
    # detail. A small deadband around 1.0 ignores no-change patches.
    rec_factor: dict | None = None
    rf_vals = [s.recfactor_map for _, s in results if s.recfactor_map is not None]
    if rf_vals:
        if brain_mask is not None:
            bm = brain_mask.to(rf_vals[0].device)
            vals = torch.cat([m[bm].flatten() for m in rf_vals])
        else:
            vals = torch.cat([m.flatten() for m in rf_vals])
        vals = vals[torch.isfinite(vals)].float()
        if vals.numel() > 0:
            fe = cfg.factor_error
            dec = vals[vals < 0.98]
            inc = vals[vals > 1.02]
            rec_factor = {
                "current": float(fe),
                "in_brain": brain_mask is not None,
                "frac_suggest_decrease": float((vals < 0.98).float().mean().item()),
                "frac_suggest_increase": float((vals > 1.02).float().mean().item()),
                "decrease_to": (
                    float(fe * torch.quantile(dec, 0.25).item()) if dec.numel() else None
                ),
                "increase_to": (
                    float(fe * torch.quantile(inc, 0.75).item()) if inc.numel() else None
                ),
            }
            if cfg.verbose:
                d, i = rec_factor["decrease_to"], rec_factor["increase_to"]
                msg = f"  Suggested factor_error (current {fe:.3f}):"
                if d is not None:
                    msg += (
                        f" lower to ~{d:.3f} over "
                        f"{rec_factor['frac_suggest_decrease'] * 100:.0f}% (over-removed)"
                    )
                if i is not None:
                    msg += (
                        f"{';' if d else ''} raise to ~{i:.3f} over "
                        f"{rec_factor['frac_suggest_increase'] * 100:.0f}% (noise-like kept)"
                    )
                if d is None and i is None:
                    msg += " no change suggested"
                print(msg)

    # 5. Finalize each echo (undo transforms + write outputs).
    outputs: list[NordicOutputs] = []
    for e in range(E):
        denoised, stats = results[e]
        mean_removed = float(torch.mean(stats.threshold_map).item())
        mean_rescued = (
            float(torch.mean(stats.rescued_map).item()) if stats.rescued_map is not None else 0.0
        )
        if cfg.verbose:
            print(
                f"  Echo {e + 1}: mean removed/patch {mean_removed:.3f}, "
                f"mean rescued/patch {mean_rescued:.4f}"
            )
        extra_meta = {
            "multiecho": {
                "echo_index": e + 1,
                "n_echoes": E,
                "rescue_enabled": bool(cfg.rescue),
                "rescue_band": cfg.rescue_band,
                "rescue_alpha": cfg.rescue_alpha,
                "mean_rescued_per_patch": mean_rescued,
                "gfactor_mode": "per-echo" if cfg.per_echo_gfactor else "shared-echo1",
                "measured_noise": float(prepped[e].measured_noise),
                "recommended_factor_error": rec_factor,
            }
        }
        if qc_summary is not None:
            extra_meta["residual_qc"] = qc_summary
        outputs.append(
            _finalize_echo(
                prepped[e],
                denoised,
                stats,
                residuals[e] if cfg.save_residual_map else None,
                mean_removed,
                f"{output_prefix}_echo-{e + 1:02d}",
                cfg,
                dev,
                magnitude_files[e],
                phase_files[e],
                extra_meta=extra_meta,
            )
        )
    return outputs
