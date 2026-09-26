"""Tissue-map noise components: per-run PCs of the voxels inside an (eroded) tissue map.

The map (a probability map or a mask) is resampled onto the data grid when the grids
differ, thresholded, and then eroded in *data-grid* voxels — erosion is meant to peel the
partial-volume shell at the EPI's resolution, not the map's.

Each run's tissue voxels have that run's other nuisance (polynomials plus any -ortvec
blocks, e.g. motion) projected out before the PCA, so the components describe noise those
regressors do not already explain instead of rediscovering them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.denoise.sequential import extract_noise_pcs_per_run
from fastfuncstuff.io.afni import load_and_concatenate_runs
from fastfuncstuff.io.headers import read_nifti_header
from fastfuncstuff.processing.grid import resample_to_grid
from fastfuncstuff.processing.io import load_image
from fastfuncstuff.processing.mask import _erode_6conn


def tissue_mask_on_grid(
    map_path: str | Path,
    ref_shape: tuple[int, int, int],
    ref_affine: np.ndarray,
    erode: int = 0,
    threshold: float = 0.5,
    device: torch.device | None = None,
) -> tuple[np.ndarray, dict]:
    """Boolean ``(x, y, z)`` mask on the reference grid: resample, threshold, erode.

    ``ref_shape`` is the data grid in NIfTI ``(x, y, z)`` order. Returns the mask and a
    dict of voxel counts at each step (for reporting).
    """
    vol, header = load_image(map_path, device=device)
    if vol.ndim != 3:
        raise ValueError(
            f"{map_path}: noise-component map must be a single 3D volume, got shape {tuple(vol.shape)}"
        )
    src_affine = np.asarray(header["affine"], dtype=np.float64)
    out_shape = (int(ref_shape[2]), int(ref_shape[1]), int(ref_shape[0]))
    resampled = tuple(vol.shape) != out_shape or not np.allclose(src_affine, ref_affine, atol=1e-4)
    if resampled:
        vol = resample_to_grid(
            vol.float(),
            src_affine,
            out_shape,
            np.asarray(ref_affine, dtype=np.float64),
            interp="linear",
        )
    mask = vol > threshold
    counts = {"resampled": resampled, "above_threshold": int(mask.sum())}
    mask = _erode_6conn(mask, iterations=int(erode)) > 0
    counts["after_erode"] = int(mask.sum())
    return mask.cpu().numpy().transpose(2, 1, 0), counts


def tissue_noise_components(
    input_files: list[str | Path],
    mask_xyz: np.ndarray,
    n_components: int,
    run_starts: list[int],
    nuisance_per_run: list[torch.Tensor],
    drop_first: int = 0,
    drop_last: int = 0,
    device: torch.device | None = None,
) -> list[np.ndarray]:
    """Per-run top-``n_components`` PCs of the voxels in ``mask_xyz``.

    ``nuisance_per_run[i]`` (T_i, q_i) is projected out of run ``i``'s tissue voxels before
    its PCA. Returns one ``(T_i, n_components)`` array per run.
    """
    mask_flat = np.ascontiguousarray(mask_xyz).reshape(-1)
    data, _ = load_and_concatenate_runs(
        input_files,
        keep_on_cpu=True,
        mask_flat=mask_flat,
        drop_first=drop_first,
        drop_last=drop_last,
    )
    data = data.float()
    # A voxel that is constant in any run (outside the FOV in one run, say) would enter that
    # run's PCA as zeros and the others' as signal; drop it everywhere.
    bounds = list(run_starts) + [data.shape[1]]
    live = torch.ones(data.shape[0], dtype=torch.bool)
    for s, e in zip(bounds[:-1], bounds[1:], strict=True):
        live &= data[:, s:e].std(dim=1) > 0
    n_live = int(live.sum())
    if n_live < max(n_components, 10):
        raise ValueError(
            f"only {n_live} usable voxels in the noise-component map; need at least "
            f"{max(n_components, 10)} (erode less or use a larger map)"
        )
    dev = device if device is not None else torch.device("cpu")
    pcs = extract_noise_pcs_per_run(
        data[live].to(dev),
        run_starts,
        torch.ones(n_live, dtype=torch.bool, device=dev),
        max_components=n_components,
        variance_threshold=1.0,
        nuisance_per_run=[n.to(dev) for n in nuisance_per_run],
        device=dev,
    )
    assert isinstance(pcs, list)
    return [p.detach().cpu().numpy().astype(np.float32) for p in pcs]


def reference_grid(input_file: str | Path) -> tuple[tuple[int, int, int], np.ndarray]:
    """``(x, y, z)`` shape and affine of a data file, from its header only."""
    try:
        hdr = read_nifti_header(input_file)
        shape, affine = hdr.get_data_shape(), hdr.get_best_affine()
    except Exception:  # AFNI HEAD/BRIK and anything else the header reader cannot parse
        from fastfuncstuff.io.afni import load_nifti

        img = load_nifti(input_file)
        assert img.shape is not None
        shape, affine = img.shape, img.affine
    nx, ny, nz = (int(s) for s in shape[:3])
    return (nx, ny, nz), np.asarray(affine, dtype=np.float64)
