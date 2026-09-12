"""Derived layers: a dataset the viewer computed, that stays in the stack.

A mode's overlay is transient -- it displaces overlay-prime and is taken back
when the mode is left. A *derived* layer is the opposite: it is a new dataset
that happens to have been made here rather than read off disk, and it behaves
like one. It can be graphed beside its source, promoted to underlay, kept while
you switch modes, and stepped through with the rest of the stack.

The first derivation is the one that motivated the mechanism: **project the
design's nuisance regressors out of a functional run** and look at what is
left, next to what went in. That comparison is the whole reason to want it --
a denoised series shown alone tells you much less than the same series shown
against its own before.

Nothing here imports Qt, and nothing here touches session state: it takes
arrays and returns arrays, so it runs on a worker thread.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

#: ``progress(fraction, message)`` -- called from a worker thread.
ProgressFn = Callable[[float, str], None]

#: AFNI's ColumnGroups convention, which ffs writes too: -1 is polynomial
#: drift, 0 is baseline / motion / generic nuisance, and 1..N are the stimulus
#: groups. So "the nuisance matrix of this design" is exactly the columns whose
#: group is not positive -- no guessing from labels, and it round-trips through
#: any xmat 3dDeconvolve or ffs_design_spec wrote.
NUISANCE_GROUP_MAX = 0

#: Singular values below this fraction of the largest are treated as rank
#: deficiency rather than as directions.
RANK_TOL = 1e-8


@dataclass(frozen=True)
class Nuisance:
    """The columns to be projected out, and where they came from."""

    columns: np.ndarray  # (T, k)
    labels: tuple[str, ...]
    description: str

    @property
    def n_columns(self) -> int:
        return int(self.columns.shape[1])


def legendre_columns(n_time: int, order: int) -> np.ndarray:
    """Orthogonal Legendre drift columns -- never raw monomials.

    See [[Legendre polynomials]]: the monomial basis is badly conditioned by
    degree 3 and the coefficients stop meaning anything.
    """
    from fastfuncstuff.glm.core import construct_polynomial_matrix

    poly = construct_polynomial_matrix(n_time, order, torch.device("cpu"), torch.float64)
    return np.asarray(poly.numpy(), dtype=np.float64)


def read_nuisance(
    path: str | Path | None,
    *,
    n_time: int,
    polort: int = -1,
) -> Nuisance:
    """Assemble the nuisance matrix from a design matrix, a 1D file, or both.

    An ``.xmat.1D`` carries ColumnGroups, so the design's *own* idea of which
    columns are nuisance is used -- the alternative, matching labels like
    "Run#1Pol#0", is guessing at something the file already states. A plain 1D
    file has no such header, so every column in it is nuisance, which is what a
    motion file is.

    ``polort`` adds drift columns on top. It defaults to off because an xmat
    already contains its own; asking for both is harmless (the duplicated
    directions collapse in :func:`orthonormal_basis`) but it is not the
    default, because silently adding a second baseline to someone's design is
    not a thing to do quietly.
    """
    blocks: list[np.ndarray] = []
    labels: list[str] = []
    described: list[str] = []

    if path is not None and str(path) not in ("", "-"):
        columns, names, kind = _read_matrix(Path(path))
        if columns.shape[0] != n_time:
            raise ValueError(
                f"{path} has {columns.shape[0]} rows but the dataset has {n_time} volumes"
            )
        if columns.shape[1] == 0:
            raise ValueError(f"{path} contributes no nuisance columns")
        blocks.append(columns)
        labels.extend(names)
        described.append(f"{kind} {Path(path).name} ({columns.shape[1]})")

    if polort >= 0:
        poly = legendre_columns(n_time, polort)
        blocks.append(poly)
        labels.extend(f"Pol#{i}" for i in range(poly.shape[1]))
        described.append(f"polort {polort}")

    if not blocks:
        raise ValueError("nothing to project out: give a matrix, a polort, or both")

    return Nuisance(
        columns=np.concatenate(blocks, axis=1),
        labels=tuple(labels),
        description=" + ".join(described),
    )


def _read_matrix(path: Path) -> tuple[np.ndarray, list[str], str]:
    """Columns, labels and a word for where they came from."""
    from fastfuncstuff.io.afni import read_afni_design_matrix

    try:
        design = read_afni_design_matrix(path)
        groups = design.get("column_groups")
    except (ValueError, KeyError, IndexError, UnicodeDecodeError):
        groups = None
        design = None

    if design is not None and groups:
        matrix = np.asarray(design["matrix"], dtype=np.float64)
        names = list(design.get("column_labels") or [])
        keep = [i for i, g in enumerate(groups) if int(g) <= NUISANCE_GROUP_MAX]
        labels = [names[i] if i < len(names) else f"col{i}" for i in keep]
        return matrix[:, keep], labels, "design"

    # No ColumnGroups: a plain regressor file, every column of which is
    # nuisance by virtue of having been handed to a denoiser.
    from fastfuncstuff.design.hrf_selection import load_nuisance_file

    matrix = np.asarray(load_nuisance_file(path), dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    return matrix, [f"{path.stem}#{i}" for i in range(matrix.shape[1])], "1D"


def orthonormal_basis(columns: np.ndarray, *, tol: float = RANK_TOL) -> np.ndarray:
    """An orthonormal basis for the nuisance column space, rank-safe.

    SVD rather than QR. Collinear nuisance columns are not an edge case here --
    an xmat already carries its own drift, so a requested ``polort`` duplicates
    it exactly, and per-run block-diagonal blocks can be rank deficient on
    their own. QR would hand back a basis containing directions that are
    numerically noise, and projecting those out removes real signal. Dropping
    the small singular values says "this design spans r dimensions", which is
    the true statement.

    Computed in float64 whatever the data is: this is the numerically sensitive
    step and the matrix is (T, k), so the cost is nothing.
    """
    matrix = np.asarray(columns, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("nuisance columns must be a non-empty (T, k) matrix")
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    if s.size == 0 or s[0] <= 0:
        raise ValueError("nuisance matrix is all zeros")
    rank = int(np.sum(s > s[0] * tol))
    return np.ascontiguousarray(u[:, :rank])


def project_out(
    flat: torch.Tensor,
    basis: torch.Tensor,
    *,
    keep_mean: bool = True,
) -> torch.Tensor:
    """Residualise ``(V, T)`` against an orthonormal ``(T, r)`` basis.

    The one-block form, so the carpet plot and the derived layer residualise
    through the same three lines rather than two that drift.
    """
    mean = flat.mean(-1, keepdim=True)
    out = flat - (flat @ basis) @ basis.T
    if keep_mean:
        out = out - out.mean(-1, keepdim=True) + mean
    return out


def denoise(
    data: np.ndarray,
    nuisance: Nuisance,
    *,
    device: torch.device | None = None,
    keep_mean: bool = True,
    progress: ProgressFn | None = None,
) -> np.ndarray:
    """Residualise a 4-D series against a nuisance matrix.

    ``keep_mean`` restores each voxel's original temporal mean. Without it the
    result is mean-zero wherever the design carried a baseline, which makes a
    "denoised BOLD" that shares no axis with the series it came from -- and
    putting the two on one graph is the entire point. Restoring the mean rather
    than skipping the constant column keeps that true whether or not the design
    happened to contain one.
    """
    if data.ndim != 4:
        raise ValueError(f"expected a 4-D series, got shape {data.shape}")
    nx, ny, nz, nt = data.shape
    if nuisance.columns.shape[0] != nt:
        raise ValueError(f"nuisance has {nuisance.columns.shape[0]} rows, dataset has {nt} volumes")

    device = device or torch.device("cpu")
    basis = torch.as_tensor(orthonormal_basis(nuisance.columns), dtype=torch.float32)
    basis = basis.to(device)  # (T, r)

    # The store keeps arrays C-contiguous, so (V, T) is a free view and time --
    # the axis every operation below runs along -- stays last.
    flat = np.asarray(data, dtype=np.float32).reshape(-1, nt)
    out = np.empty_like(flat)

    from fastfuncstuff.memory import estimate_chunk_size

    chunk = estimate_chunk_size(
        n_voxels=flat.shape[0],
        n_timepoints=nt,
        n_regressors=int(basis.shape[1]),
        device=device,
        operation="denoise",
    )

    for start in range(0, flat.shape[0], chunk):
        stop = min(start + chunk, flat.shape[0])
        block = torch.as_tensor(flat[start:stop]).to(device)
        out[start:stop] = project_out(block, basis, keep_mean=keep_mean).cpu().numpy()
        if progress is not None:
            progress(stop / flat.shape[0], f"projecting {nuisance.n_columns} columns")

    return out.reshape(nx, ny, nz, nt)


__all__ = [
    "NUISANCE_GROUP_MAX",
    "Nuisance",
    "denoise",
    "legendre_columns",
    "orthonormal_basis",
    "project_out",
    "read_nuisance",
]
