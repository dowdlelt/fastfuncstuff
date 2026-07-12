"""Patch-based residual-PC projection (``ffs_pcpatch``).

FAILED EXPERIMENT — kept for the record, not for use. The premise (below) does
not hold: NORDIC's residual is the *noise floor*, which is temporally
near-full-rank, so a variance fraction selects ~f·T directions almost uniformly
everywhere (Marchenko–Pastur is self-averaging) and projecting them out guts the
signal. There is no low-DoF re-removal of thermal noise to be had here — use
``ffs_reml -adjust_dof`` / ``ffs_util_updatedof`` to *account* for NORDIC's DoF
cost, or ``ffs_nordic -retain_dof`` to *cap* it at estimation time. See
``fmri_wiki/concepts/Residual PC projection.md`` for the full write-up.

Original idea (did not work):

Second-stage thermal-noise removal that spends far fewer degrees of freedom than
NORDIC. The idea:

* NORDIC runs on the *raw* series and, to pull thermal noise out of un-smoothed
  data, has to remove many components per patch (often ~100). That DoF cost can
  invalidate the downstream GLM's statistics (see [[DoF adjustment after NORDIC]]).
* Instead, carry NORDIC's *residual* (the removed noise, ``ffs_nordic
  -save_residual_map -add_mean``) through the **same** preprocessing as the data
  (motion, blur, interpolation, warp). Blurring/interpolation correlate
  neighbouring noise, so the residual becomes **lower rank** than it was at
  NORDIC time — the same thermal noise now lives in a smaller temporal subspace.
* ``ffs_pcpatch`` re-estimates that subspace patch-by-patch from the transformed
  residual and projects it out of the **non-denoised** (but identically
  preprocessed) data. Fewer components removed → fewer DoF lost.

Per patch (residual ``R`` and data ``D``, both ``M voxels × T``):

1. **Temporal-demean** ``R`` and, if requested, project the *same* GLM nuisance
   (``-polort`` Legendre drift + ``-ort`` motion regressors) out of ``R``. This
   is the anti-double-dip step: the extracted subspace is then ⊥ the GLM
   nuisance, so removing it from the data cannot re-inject motion/drift variance
   the GLM already handles.
2. **Eigendecompose** the cleaned residual patch (Gram ``RᵀR``, ``T×T``) →
   temporal singular vectors ``V`` and variances ``σ²`` (descending).
3. **Skip the first ``skip_first`` PCs** — preprocessing (interp edges, motion)
   injects structured, high-variance signal that piles up at the top and is *not*
   thermal noise. Drop it from both the selection and the variance normalisation.
4. **Select ``k``** from the post-skip spectrum: a fixed count, or the smallest
   ``k`` whose cumulative post-skip variance reaches ``var_frac``. The near-zero
   tail (the components that were never noise) contributes ~nothing, so the
   fraction lands on the real thermal band.
5. **Project PCs ``[skip_first : skip_first+k)`` out of the data patch** ``D``
   (temporal mean preserved), overlap-average like NORDIC.

Outputs the cleaned series and a per-voxel ``_numcomps`` map (patch-averaged
``k``) that feeds ``ffs_util_updatedof`` / ``ffs_reml -adjust_dof``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from fastfuncstuff.denoise.nordic import _build_patch_starts, _default_kernel_size_pca
from fastfuncstuff.io.afni import load_nifti, save_nifti
from fastfuncstuff.utils import get_device, to_tensor

# ---------------------------------------------------------------------------
# Config / outputs
# ---------------------------------------------------------------------------


@dataclass
class PCPatchConfig:
    """Configuration for patch-based residual-PC projection.

    Exactly one of ``n_comps`` / ``var_frac`` selects how many components per
    patch to remove: ``n_comps`` (int) = a fixed count, ``var_frac`` (float in
    (0, 1]) = the smallest count reaching that cumulative post-skip variance.
    """

    n_comps: int | None = None
    var_frac: float | None = 0.95
    skip_first: int = 0
    kernel_size: tuple[int, int, int] | None = None  # None -> NORDIC default from T
    patch_overlap: int = 2
    polort: int = -1  # <0 = off; >=0 = Legendre degree projected out of residual
    ort: np.ndarray | None = None  # (T, p) extra nuisance regressors for the residual
    write_gzipped: bool = True
    svd_batch_size: int = 512
    verbose: bool = True


@dataclass
class PCPatchOutputs:
    data_file: Path
    num_comps_file: Path
    metadata_file: Path


# ---------------------------------------------------------------------------
# Nuisance basis
# ---------------------------------------------------------------------------


def build_nuisance_basis(
    n_t: int,
    polort: int,
    ort: np.ndarray | None,
    device: torch.device,
) -> torch.Tensor | None:
    """Orthonormal temporal nuisance basis ``Q`` (T, p) to project out of the
    residual, or ``None`` when nothing is requested.

    Columns are the Legendre drift polynomials (degree 0..polort) and any
    ``-ort`` regressors, then QR-orthonormalised. Near-dependent columns are
    dropped by a rank tolerance so the projector is well-conditioned.
    """
    cols: list[torch.Tensor] = []
    if polort >= 0:
        # Reuse the project's Legendre builder (orthogonal, numerically stable).
        from fastfuncstuff.glm.core import construct_polynomial_matrix

        cols.append(construct_polynomial_matrix(n_t, polort, device, dtype=torch.float32))
    if ort is not None:
        o = np.asarray(ort, dtype=np.float32)
        if o.ndim == 1:
            o = o[:, None]
        if o.shape[0] != n_t:
            raise ValueError(f"-ort has {o.shape[0]} rows but data has {n_t} timepoints")
        cols.append(to_tensor(o, dtype=torch.float32, device=device))
    if not cols:
        return None
    z = torch.cat(cols, dim=1)  # (T, p)
    q, r = torch.linalg.qr(z)
    # Drop columns whose pivot is tiny (rank-deficient / duplicated nuisance).
    keep = torch.abs(torch.diagonal(r)) > 1e-6 * float(torch.abs(r).max().clamp(min=1e-30))
    q = q[:, keep]
    return q if q.shape[1] > 0 else None


# ---------------------------------------------------------------------------
# Core patch projection
# ---------------------------------------------------------------------------


def _pcpatch_project(
    data: torch.Tensor,
    residual: torch.Tensor,
    valid: torch.Tensor,
    kernel_size: tuple[int, int, int],
    patch_overlap: int,
    *,
    n_comps: int | None,
    var_frac: float | None,
    skip_first: int,
    nuisance_q: torch.Tensor | None,
    svd_batch_size: int,
    device: torch.device,
    verbose: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Project the residual's temporal noise subspace out of ``data``.

    ``data`` / ``residual`` are (nx, ny, nz, T) real; ``valid`` is (nx, ny, nz)
    bool marking where there is data. Returns ``(cleaned, num_comps, summary)``
    with ``cleaned`` on ``device``, ``num_comps`` (nx, ny, nz) patch-averaged
    count removed, and a small summary dict.

    Patches are extracted on the data's device and moved to ``device`` for the
    decomposition (so the volumes can live on CPU while compute stays on GPU).
    """
    data_dev = data.device
    cross_device = device != data_dev

    nx, ny, nz, nt = data.shape
    wx = min(kernel_size[0], nx)
    wy = min(kernel_size[1], ny)
    wz = min(kernel_size[2], nz)

    sx = max(1, wx // max(1, patch_overlap))
    sy = max(1, wy // max(1, patch_overlap))
    sz = max(1, wz // max(1, patch_overlap))
    xs = _build_patch_starts(nx, wx, sx)
    ys = _build_patch_starts(ny, wy, sy)
    zs = _build_patch_starts(nz, wz, sz)
    corners = [(x0, y0, z0) for x0 in xs for y0 in ys for z0 in zs]
    total = len(corners)

    # Accumulators on the compute device.
    recon_acc = torch.zeros(nx, ny, nz, nt, dtype=data.dtype, device=device)
    weight = torch.zeros((nx, ny, nz), dtype=torch.float32, device=device)
    ncomp_acc = torch.zeros_like(weight)

    # Per-patch voxel offsets (same machinery as nordic._llr_denoise).
    _ox, _oy, _oz = torch.arange(wx), torch.arange(wy), torch.arange(wz)
    gx, gy, gz = torch.meshgrid(_ox, _oy, _oz, indexing="ij")
    dx, dy, dz = gx.ravel(), gy.ravel(), gz.ravel()
    local_offsets_flat = dx * (ny * nz) + dy * nz + dz  # (M,)
    corner_xs = torch.tensor([c[0] for c in corners], dtype=torch.long)
    corner_ys = torch.tensor([c[1] for c in corners], dtype=torch.long)
    corner_zs = torch.tensor([c[2] for c in corners], dtype=torch.long)
    corners_flat = corner_xs * (ny * nz) + corner_ys * nz + corner_zs

    recon_flat = recon_acc.reshape(-1, nt)
    weight_flat = weight.reshape(-1)
    ncomp_flat = ncomp_acc.reshape(-1)

    idx_arange = torch.arange(nt, device=device)
    skip = min(max(0, skip_first), nt)

    total_skipvar = 0.0
    total_skipvar_w = 0.0

    pbar = tqdm(total=total, desc="pcpatch", unit="patch") if verbose else None
    for batch_start in range(0, total, svd_batch_size):
        B = min(svd_batch_size, total - batch_start)
        bx = corner_xs[batch_start : batch_start + B]
        by = corner_ys[batch_start : batch_start + B]
        bz = corner_zs[batch_start : batch_start + B]
        xi = (bx[:, None] + dx[None, :]).to(data_dev)  # (B, M)
        yi = (by[:, None] + dy[None, :]).to(data_dev)
        zi = (bz[:, None] + dz[None, :]).to(data_dev)

        R = residual[xi, yi, zi, :]  # (B, M, T)
        vpatch = valid[xi, yi, zi].to(device=device, dtype=data.dtype)  # (B, M)
        if cross_device:
            R = R.to(device)
        R = R.to(torch.float32)

        # --- (1) demean + nuisance-project the residual (temporal) ---
        R = R - R.mean(dim=2, keepdim=True)
        e0 = (R * R).sum(dim=(1, 2))  # (B,) demeaned residual energy (pre-nuisance)
        if nuisance_q is not None:
            # R <- R - (R Q) Qᵀ ; Q (T, p) orthonormal
            R = R - (R @ nuisance_q) @ nuisance_q.mT

        # --- (2) temporal Gram eigendecomposition ---
        G = R.mT @ R  # (B, T, T) real symmetric
        del R
        eigvals, V = torch.linalg.eigh(G)  # ascending
        del G
        s2 = eigvals.flip(1).clamp(min=0)  # (B, T) descending variances
        V = V.flip(2)  # columns match descending order

        # --- (3) skip leading PCs; (4) select k over the remainder ---
        total_var = s2.sum(dim=1)  # (B,)
        skip_var = s2[:, :skip].sum(dim=1) if skip > 0 else torch.zeros(B, device=device)
        avail = s2[:, skip:]  # (B, T-skip)
        if n_comps is not None:
            k = torch.full((B,), min(int(n_comps), avail.shape[1]), device=device, dtype=torch.long)
        else:
            denom = avail.sum(dim=1, keepdim=True).clamp(min=1e-30)
            cumfrac = torch.cumsum(avail, dim=1) / denom  # (B, T-skip)
            # smallest index reaching var_frac (count = index + 1)
            reached = cumfrac >= float(var_frac)
            k = reached.to(torch.int64).argmax(dim=1) + 1
            k = torch.where(reached.any(dim=1), k, torch.tensor(avail.shape[1], device=device))
        k = k.clamp(min=0, max=avail.shape[1])

        # Guard: a residual whose energy collapses after nuisance projection (fully
        # explained by drift/motion, or an empty patch) has no thermal noise to
        # remove — selecting var_frac of pure numerical noise would remove junk.
        negligible = total_var <= torch.clamp(1e-4 * e0, min=1e-12)
        k = torch.where(negligible, torch.zeros_like(k), k)

        # Removal band = columns [skip, skip+k). Zero the rest of V, build the
        # projector P = V_sel V_selᵀ onto that subspace.
        removal = (idx_arange[None, :] >= skip) & (idx_arange[None, :] < (skip + k)[:, None])
        V_sel = V * removal[:, None, :].to(V.dtype)  # (B, T, T), non-removed cols zeroed
        del V
        P = V_sel @ V_sel.mT  # (B, T, T)
        del V_sel

        # --- (5) project the band out of the data patch, preserve temporal mean ---
        D = data[xi, yi, zi, :]  # (B, M, T)
        if cross_device:
            D = D.to(device)
        D = D.to(torch.float32)
        Dmean = D.mean(dim=2, keepdim=True)
        Dc = D - Dmean
        D_clean = Dmean + (Dc - Dc @ P)  # remove only the noise-subspace fluctuation
        del D, Dc, P

        # Only write valid voxels (invalid ones keep their original data via the
        # weight==0 fill at the end). Weight/recon scaled by per-voxel validity.
        vscale = vpatch  # (B, M) in {0, 1}
        D_clean = D_clean * vscale[:, :, None]

        b_corners_flat = corners_flat[batch_start : batch_start + B]
        flat_b = (b_corners_flat[:, None] + local_offsets_flat[None, :]).reshape(-1).to(device)
        recon_flat.index_add_(0, flat_b, D_clean.reshape(-1, nt))
        weight_flat.index_add_(0, flat_b, vscale.reshape(-1))
        ncomp_flat.index_add_(0, flat_b, (k.to(data.dtype)[:, None] * vscale).reshape(-1))
        del D_clean, flat_b

        # Skipped-variance fraction, patch-weighted by valid-voxel count.
        pw = vscale.sum(dim=1)  # (B,)
        frac = (skip_var / total_var.clamp(min=1e-30)).to(torch.float64)
        total_skipvar += float((frac * pw).sum().item())
        total_skipvar_w += float(pw.sum().item())

        if pbar is not None:
            pbar.update(B)
    if pbar is not None:
        pbar.close()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    covered = weight > 0
    w = weight.clamp(min=1.0)
    num_comps = ncomp_acc / w  # (nx, ny, nz), cheap; stays on the compute device

    # Normalise in place (recon_acc is private) then finalise on CPU: the
    # accumulator already fills most of VRAM when inputs were streamed, so
    # building the uncovered-voxel fill on-device would need a second full-volume
    # allocation and OOM. Move off-device, free the accumulator, then fill.
    recon_acc /= w[..., None]
    cleaned = recon_acc.to("cpu")
    del recon_acc
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Uncovered / invalid voxels: restore the original (non-denoised) data.
    if bool((~covered).any()):
        covered_cpu = covered.to("cpu")
        orig_cpu = data if data.device.type == "cpu" else data.to("cpu")
        cleaned = torch.where(covered_cpu[..., None], cleaned, orig_cpu.to(cleaned.dtype))
        num_comps = torch.where(covered, num_comps, torch.zeros_like(num_comps))

    mean_skipvar = total_skipvar / total_skipvar_w if total_skipvar_w > 0 else 0.0
    nc_valid = num_comps[covered]
    summary = {
        "mean_num_comps": float(nc_valid.mean().item()) if nc_valid.numel() else 0.0,
        "median_num_comps": float(nc_valid.median().item()) if nc_valid.numel() else 0.0,
        "max_num_comps": float(nc_valid.max().item()) if nc_valid.numel() else 0.0,
        "mean_skipped_variance_frac": mean_skipvar,
        "n_patches": total,
    }
    return cleaned, num_comps, summary


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_pcpatch(
    data_file: str,
    residual_file: str,
    output_prefix: str,
    config: PCPatchConfig | None = None,
    device: torch.device | None = None,
    mask_file: str | None = None,
    orig_numcomps_file: str | None = None,
) -> PCPatchOutputs:
    """Estimate the residual's temporal noise subspace per patch and project it
    out of the (non-denoised, identically preprocessed) data."""
    cfg = config or PCPatchConfig()
    dev = (
        device if device is not None else get_device("cuda" if torch.cuda.is_available() else None)
    )

    data_np = load_nifti(data_file).get_fdata(dtype=np.float32).astype(np.float32)
    resid_np = load_nifti(residual_file).get_fdata(dtype=np.float32).astype(np.float32)
    if data_np.shape != resid_np.shape:
        raise ValueError(
            f"data {data_np.shape} and residual {resid_np.shape} must have identical shape "
            "(the residual must be carried through the SAME preprocessing as the data)"
        )
    if data_np.ndim != 4:
        raise ValueError(f"expected 4D data, got shape {data_np.shape}")
    nx, ny, nz, nt = data_np.shape

    # Validity: where the residual actually has data (nonzero temporal energy),
    # intersected with an explicit mask when provided.
    valid_np = np.abs(resid_np).sum(axis=-1) > 0
    if mask_file is not None:
        m = load_nifti(mask_file).get_fdata(dtype=np.float32)
        if m.shape[:3] != (nx, ny, nz):
            raise ValueError(f"mask {m.shape[:3]} does not match data grid {(nx, ny, nz)}")
        valid_np &= m > 0

    kernel = cfg.kernel_size or _default_kernel_size_pca(nt, n_slices=nz)

    # Data volumes live on the data device; when CUDA, keep them on CPU and stream
    # patches to the GPU only if they would not comfortably fit. Two 4-D float32
    # volumes + a full 4-D accumulator is the resident cost.
    data_dev = dev
    if dev.type == "cuda":
        vol_bytes = data_np.size * 4
        free_b, _ = torch.cuda.mem_get_info(dev)
        # data + residual + recon accumulator, with headroom for the batch working set.
        if 3.2 * vol_bytes > 0.5 * free_b:
            data_dev = torch.device("cpu")
            if cfg.verbose:
                print(
                    f"  Memory guard: {3.2 * vol_bytes / 1024**3:.2f} GiB resident vs "
                    f"{0.5 * free_b / 1024**3:.2f} GiB budget — streaming inputs from CPU."
                )

    data = to_tensor(data_np, dtype=torch.float32, device=data_dev)
    residual = to_tensor(resid_np, dtype=torch.float32, device=data_dev)
    valid = torch.from_numpy(valid_np).to(data_dev)
    del data_np, resid_np

    nuisance_q = build_nuisance_basis(nt, cfg.polort, cfg.ort, dev)

    if cfg.verbose:
        sel = f"{cfg.n_comps} comps" if cfg.n_comps is not None else f"{cfg.var_frac:g} var-frac"
        nuis = 0 if nuisance_q is None else nuisance_q.shape[1]
        print(
            f"  Patch {kernel}, overlap {cfg.patch_overlap}, select={sel}, "
            f"skip_first={cfg.skip_first}, nuisance dims={nuis}"
        )

    cleaned, num_comps, summary = _pcpatch_project(
        data,
        residual,
        valid,
        kernel_size=kernel,
        patch_overlap=cfg.patch_overlap,
        n_comps=cfg.n_comps,
        var_frac=cfg.var_frac,
        skip_first=cfg.skip_first,
        nuisance_q=nuisance_q,
        svd_batch_size=cfg.svd_batch_size,
        device=dev,
        verbose=cfg.verbose,
    )

    # --- Save ---
    out_prefix = Path(output_prefix)
    out_dir = out_prefix.parent if out_prefix.parent != Path("") else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ".nii.gz" if cfg.write_gzipped else ".nii"

    data_out = out_dir / f"{out_prefix.name}{ext}"
    save_nifti(
        cleaned.cpu().numpy().astype(np.float32), output_path=data_out, reference_img=data_file
    )
    num_comps_np = num_comps.cpu().numpy().astype(np.float32)
    del cleaned, num_comps

    numcomps_out = out_dir / f"{out_prefix.name}_numcomps{ext}"
    save_nifti(num_comps_np, output_path=numcomps_out, reference_img=data_file)

    # Optional comparison to NORDIC's original (carried-through) numcomps.
    orig_summary = None
    if orig_numcomps_file is not None:
        orig = load_nifti(orig_numcomps_file).get_fdata(dtype=np.float32)
        orig_v = orig[..., 0] if orig.ndim == 4 else orig
        vmask = valid_np & (num_comps_np > 0)
        if orig_v.shape[:3] != (nx, ny, nz):
            # The orig numcomps must be on the SAME grid — i.e. carried through the
            # SAME preprocessing as the data. A raw-grid NORDIC map won't match.
            print(
                f"  WARNING: -orig_numcomps grid {tuple(orig_v.shape[:3])} != data grid "
                f"{(nx, ny, nz)}; skipping comparison. Pass the numcomps map carried "
                "through the SAME preprocessing as the data (not the raw NORDIC output)."
            )
        elif not vmask.any():
            print("  WARNING: no valid voxels for -orig_numcomps comparison; skipping.")
        else:
            orig_summary = {
                "mean_orig_num_comps": float(orig_v[vmask].mean()),
                "mean_new_num_comps": float(num_comps_np[vmask].mean()),
                "mean_dof_saved": float((orig_v[vmask] - num_comps_np[vmask]).mean()),
            }
            if cfg.verbose:
                print(
                    f"  Original rank ~{orig_summary['mean_orig_num_comps']:.1f} -> "
                    f"new ~{orig_summary['mean_new_num_comps']:.1f} comps/voxel "
                    f"(~{orig_summary['mean_dof_saved']:.1f} DoF/voxel saved)"
                )

    meta = {
        "data_file": str(data_file),
        "residual_file": str(residual_file),
        "output_prefix": str(output_prefix),
        "shape": [int(nx), int(ny), int(nz), int(nt)],
        "device": str(dev),
        "config": {
            "n_comps": cfg.n_comps,
            "var_frac": cfg.var_frac,
            "skip_first": cfg.skip_first,
            "kernel_size": list(kernel),
            "patch_overlap": cfg.patch_overlap,
            "polort": cfg.polort,
            "n_ort": 0 if cfg.ort is None else int(np.asarray(cfg.ort).reshape(nt, -1).shape[1]),
            "mask_file": mask_file,
        },
        "diagnostics": summary,
        "comparison": orig_summary,
        "outputs": {"data": str(data_out), "num_comps": str(numcomps_out)},
    }
    if cfg.verbose:
        frac_removed = summary["mean_num_comps"] / max(1, nt)
        print(
            f"  Mean components removed: {summary['mean_num_comps']:.2f} of {nt} "
            f"({100 * frac_removed:.0f}%)  "
            f"(median {summary['median_num_comps']:.1f}, max {summary['max_num_comps']:.0f})"
        )
        if frac_removed > 0.5:
            # Thermal noise is temporally near-full-rank, so a variance fraction on
            # a noise residual selects ~frac*T directions and guts the data (those
            # directions overlap signal). This spends MORE DoF than intended.
            how = (
                f"var-frac {cfg.var_frac:g}" if cfg.var_frac is not None else f"{cfg.n_comps} comps"
            )
            print(
                "  WARNING: removing >50% of the temporal directions — the residual is "
                "not low-rank (thermal noise is near-full-rank in time), so "
                f"{how} is over-removing and gutting signal (those directions overlap it). "
                "For the intended tradeoff use a small fixed count (e.g. -ncomps 20-40) "
                "and -skip-first to target only the concentrated (structured) residual modes."
            )
    meta_file = out_dir / f"{out_prefix.name}_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return PCPatchOutputs(
        data_file=data_out,
        num_comps_file=numcomps_out,
        metadata_file=meta_file,
    )
