"""Slicewise baseline regressors for ARMA(1,1) REML (AFNI 3dREMLfit -slibase).

``-slibase`` / ``-slibase_sm`` supply extra nuisance regressors (typically
physiological noise) that differ per imaging slice. A single ``.1D`` file holds
``n_slices * m`` columns; this module de-interleaves them into per-slice blocks of
``m`` regressors each. The two flags differ only in column ordering:

- ``-slibase``    — *slice-minor* (cyclic): column ``r*n_slices + s`` → slice ``s``,
  regressor ``r``. For 3 slices, 6 cols: [s0,s1,s2, s0,s1,s2].
- ``-slibase_sm`` — *slice-major* (blocked): column ``s*m + r`` → slice ``s``,
  regressor ``r``. For 3 slices, 6 cols: [s0r0,s0r1, s1r0,s1r1, s2r0,s2r1].

Slices are the 3rd spatial (z) axis, matching AFNI's storage-order convention.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_1d_matrix(path: str | Path) -> torch.Tensor:
    """Load an AFNI ``.1D`` numeric matrix as a ``(n_rows, n_cols)`` float32 tensor.

    ``#``-prefixed comment lines (AFNI header decorations) are ignored.
    """
    arr = np.loadtxt(str(path), comments="#", ndmin=2).astype(np.float32)
    return torch.from_numpy(arr)


def deinterleave_slibase(
    mat: torch.Tensor, n_slices: int, slice_major: bool
) -> torch.Tensor:
    """De-interleave a ``(n_time, n_slices*m)`` matrix into ``(n_slices, n_time, m)``.

    Parameters
    ----------
    mat : (n_time, n_slices*m) tensor
        Raw columns from one ``.1D`` file.
    n_slices : int
        Number of slices (z extent).
    slice_major : bool
        False for ``-slibase`` (slice-minor / cyclic), True for ``-slibase_sm``
        (slice-major / blocked).
    """
    n_time, n_cols = mat.shape
    if n_cols % n_slices != 0:
        raise ValueError(
            f"slibase file has {n_cols} columns, not an integer multiple of "
            f"n_slices={n_slices}"
        )
    m = n_cols // n_slices
    if slice_major:
        # column = s*m + r  ->  reshape (n_time, n_slices, m), slices on axis 1
        return mat.reshape(n_time, n_slices, m).permute(1, 0, 2).contiguous()
    # column = r*n_slices + s  ->  reshape (n_time, m, n_slices), slices on axis 2
    return mat.reshape(n_time, m, n_slices).permute(2, 0, 1).contiguous()


def build_slice_blocks(
    files: list[str] | None,
    files_sm: list[str] | None,
    n_slices: int,
    n_timepoints: int,
) -> tuple[torch.Tensor, list[str]]:
    """Load and concatenate all slibase files into one ``(n_slices, n_time, M)`` block.

    ``files`` use slice-minor ordering (``-slibase``); ``files_sm`` use slice-major
    (``-slibase_sm``). Regressors from every file are concatenated along the last
    axis (``M`` = total regressors per slice). Returns the block plus one nuisance
    label per regressor (``slibase#k``).
    """
    blocks: list[torch.Tensor] = []
    for path in files or []:
        blocks.append(_one_file_block(path, n_slices, n_timepoints, slice_major=False))
    for path in files_sm or []:
        blocks.append(_one_file_block(path, n_slices, n_timepoints, slice_major=True))
    if not blocks:
        raise ValueError("build_slice_blocks called with no slibase files")
    slice_blocks = torch.cat(blocks, dim=2)  # (n_slices, n_time, M)
    labels = [f"slibase#{k}" for k in range(slice_blocks.shape[2])]
    return slice_blocks, labels


def _one_file_block(
    path: str, n_slices: int, n_timepoints: int, slice_major: bool
) -> torch.Tensor:
    mat = load_1d_matrix(path)
    if mat.shape[0] != n_timepoints:
        raise ValueError(
            f"slibase file '{path}' has {mat.shape[0]} rows but data has "
            f"{n_timepoints} timepoints"
        )
    return deinterleave_slibase(mat, n_slices, slice_major)


def voxel_slice_indices(
    n_voxels_total: int,
    n_slices: int,
    mask_tensor: torch.Tensor | None,
) -> torch.Tensor:
    """Slice (z) index per analysis voxel.

    With C-order flattening of ``(nx, ny, nz, nt)``, z is the fastest-varying spatial
    axis, so the slice of flat index ``i`` is ``i % nz``. When a mask is applied, only
    the kept voxels' flat indices are used (aligned to the masked data's voxel axis).
    """
    if mask_tensor is not None:
        flat = mask_tensor.nonzero(as_tuple=True)[0]
    else:
        flat = torch.arange(n_voxels_total)
    return (flat % n_slices).long()
