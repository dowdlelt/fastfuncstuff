"""The InstaPCA engine: one run, decomposed the way ffs_denoise extracts noise PCs.

Each voxel's time series has the Legendre drift projected out and is then scaled
to unit length before the decomposition, exactly as
``denoise/sequential.py:extract_noise_pcs_per_run`` prepares its noise pool.
Without the scaling the leading components are a picture of which voxels are
loud -- vessels and the brain edge -- rather than of shared temporal structure.

That preparation is also what makes the maps readable. A voxel's prepared series
is unit length and mean-free, and so is each component's time course, so their
dot product *is* the Pearson correlation between the voxel's detrended series and
the component: a map in ``[-1, 1]`` that means the same thing in every voxel and
every component. This is the amplitude-free map ``ffs_denoise`` writes by
default (``compute_full_brain_pc_loadings(match_extraction=True)``). The
amplitude map -- how many percent of the voxel's mean the component moves it --
is offered beside it, because "where does it live" and "how much does it cost
here" are different questions.

The decomposition goes through the ``(T, T)`` Gram rather than an SVD of the
wide ``(V, T)`` matrix; see the wiki's *Wide-matrix decompositions* principle.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ProgressFn = Callable[[float, str], None]


@dataclass
class Decomposition:
    """A run's leading components, sign-fixed and ready to draw."""

    #: ``(T, K)``, zero mean, unit standard deviation -- the ortvec columns.
    timecourses: np.ndarray
    #: ``(V, K)`` correlation of each masked voxel's detrended series with each
    #: component.
    correlation: np.ndarray
    #: ``(V, K)`` the standard deviation of the component's share of each voxel,
    #: in percent of that voxel's mean. Signed like ``correlation``.
    amplitude: np.ndarray
    #: ``(K,)`` fraction of the prepared (detrended, unit-norm) variance.
    explained: np.ndarray
    mask: np.ndarray
    affine: np.ndarray
    tr: float
    polort: int
    #: ``"automask"`` or the mask file's name, for the ortvec header.
    mask_source: str

    @property
    def n_components(self) -> int:
        return int(self.timecourses.shape[1])

    def volume(self, k: int, kind: str = "correlation") -> np.ndarray:
        """Component ``k``'s map on the run's grid, NaN outside the mask.

        NaN rather than zero because zero is a legal correlation, and a map that
        draws "outside the brain" and "uncorrelated" alike is the bug the
        ffs_denoise PC figures had.
        """
        values = self.amplitude if kind == "amplitude" else self.correlation
        out = np.full(self.mask.shape, np.nan, dtype=np.float32)
        out[self.mask] = values[:, k]
        return out


@contextlib.contextmanager
def _no_tf32() -> Iterator[None]:
    """A Gram is a sum of ~10^6 products; TF32's 10-bit mantissa is not enough."""
    saved = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved


def max_components(n_time: int, polort: int) -> int:
    """How many components a run of ``n_time`` can have after ``polort`` drift."""
    return max(1, n_time - (polort + 1))


def decompose(
    data: np.ndarray,
    *,
    affine: np.ndarray,
    tr: float,
    mask: np.ndarray | None = None,
    mask_source: str = "automask",
    polort: int = 2,
    n_components: int = 20,
    device: torch.device | None = None,
    progress: ProgressFn | None = None,
) -> Decomposition:
    """Decompose one 4-D run into its ``n_components`` leading components."""
    from fastfuncstuff.memory import estimate_chunk_size
    from fastfuncstuff.utils import factor_device
    from fastfuncstuff.viewer.derive import orthonormal_basis, read_nuisance

    if data.ndim != 4:
        raise ValueError(f"InstaPCA needs one 4-D run; got shape {tuple(data.shape)}")
    n_time = int(data.shape[3])
    # The constant is not optional: a component of uncentred data is the mean.
    polort = max(0, int(polort))
    if n_time < polort + 3:
        raise ValueError(f"{n_time} volumes is too short for polort {polort}")
    device = device or torch.device("cpu")

    if progress is not None:
        progress(0.02, "masking")
    if mask is None:
        from fastfuncstuff.viewer.series import automask_from_series

        mask = automask_from_series(data, device=device)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != data.shape[:3]:
        raise ValueError(f"mask shape {mask.shape} does not match the run {data.shape[:3]}")
    n_vox = int(mask.sum())
    if n_vox < 2:
        raise ValueError("the mask has fewer than two voxels; nothing to decompose")

    drift = orthonormal_basis(read_nuisance(None, n_time=n_time, polort=polort).columns)
    q = torch.as_tensor(drift, dtype=torch.float32, device=device)
    rows = np.asarray(data, dtype=np.float32).reshape(-1, n_time)[mask.reshape(-1)]
    k = min(int(n_components), max_components(n_time, polort))
    step = estimate_chunk_size(n_vox, n_time, k, device, operation="warp_gram")

    def prepared(i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ``rows`` is a fresh fancy-indexed copy, so nothing here can write
        # through to the store's array -- but the subtraction is still not in
        # place, so that stays true if the gather above ever becomes a view.
        staged = torch.as_tensor(rows[i : i + step]).to(device, torch.float32)
        mean = staged.mean(dim=1)
        c = staged - (staged @ q) @ q.T
        norm = torch.linalg.vector_norm(c, dim=1)
        return c / norm.clamp(min=1e-10)[:, None], norm, mean

    gram = torch.zeros(n_time, n_time, dtype=torch.float64, device=device)
    norms = np.empty(n_vox, dtype=np.float64)
    means = np.empty(n_vox, dtype=np.float64)
    with _no_tf32():
        for i in range(0, n_vox, step):
            if progress is not None:
                progress(0.05 + 0.45 * i / n_vox, "gram")
            c, norm, mean = prepared(i)
            gram += (c.T @ c).double()
            norms[i : i + step] = norm.double().cpu().numpy()
            means[i : i + step] = mean.double().cpu().numpy()
            del c

        if progress is not None:
            progress(0.5, "eigh")
        gram_cpu = gram.to(factor_device(device))
        evals, evecs = torch.linalg.eigh(gram_cpu)
        order = torch.argsort(evals, descending=True)[:k]
        u = evecs[:, order]
        explained = (evals[order] / torch.trace(gram_cpu).clamp(min=1e-30)).cpu().numpy()

        u_dev = u.to(device, torch.float32)
        corr = np.empty((n_vox, k), dtype=np.float32)
        for i in range(0, n_vox, step):
            if progress is not None:
                progress(0.55 + 0.4 * i / n_vox, "maps")
            c, _norm, _mean = prepared(i)
            corr[i : i + step] = (c @ u_dev).cpu().numpy()
            del c

    # eigh's sign is arbitrary. Pointing each map's heavy tail positive is the
    # convention MELODIC uses, and it is what makes a re-run of the same data
    # give the same picture -- and the same sign on the saved regressor.
    flip = np.where(np.sum(corr.astype(np.float64) ** 3, axis=0) < 0, -1.0, 1.0)
    corr *= flip.astype(np.float32)
    timecourses = u.cpu().numpy() * flip[None, :] * np.sqrt(n_time)

    # The component's share of a voxel is ``corr * norm * u``; ``u`` is unit
    # length and mean-free, so its standard deviation is ``corr * norm / sqrt(T)``.
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(np.abs(means) > 0, 100.0 * norms / (np.sqrt(n_time) * means), 0.0)
    amplitude = (corr * scale[:, None]).astype(np.float32)
    if progress is not None:
        progress(1.0, "decomposed")
    return Decomposition(
        timecourses=timecourses,
        correlation=corr,
        amplitude=amplitude,
        explained=explained,
        mask=mask,
        affine=np.asarray(affine, dtype=float),
        tr=float(tr),
        polort=polort,
        mask_source=mask_source,
    )


def read_mask(path: str | Path, shape: tuple[int, int, int]) -> np.ndarray:
    """A mask file on the run's own grid; nonzero is inside."""
    from fastfuncstuff.io.afni import load_nifti

    img = load_nifti(Path(path).expanduser())
    values = np.asanyarray(img.dataobj)
    if values.ndim == 4 and values.shape[3] == 1:
        values = values[..., 0]
    if values.shape != tuple(shape):
        raise ValueError(
            f"mask {Path(path).name} is {values.shape}, the run is {tuple(shape)}; "
            "resample it onto the run's grid first"
        )
    return np.asarray(values != 0)


def write_ortvec(
    path: str | Path, decomposition: Decomposition, components: Sequence[int], *, source: str = ""
) -> Path:
    """The chosen components' time courses as an ortvec 1D file, one column each.

    The header is ``#`` comments, which every 1D reader in the toolbox and AFNI
    skips, so the file says what it is without being anything but numbers.
    """
    chosen = sorted(set(int(c) for c in components))
    if not chosen:
        raise ValueError("no components chosen")
    if chosen[-1] >= decomposition.n_components:
        raise ValueError(f"component {chosen[-1]} is past the last one")
    path = Path(path).expanduser()
    header = [
        f"InstaPCA noise components of {source or 'run'}: {' '.join(map(str, chosen))}",
        f"polort {decomposition.polort}, mask {decomposition.mask_source}, "
        f"{decomposition.n_components} components, unit-variance columns",
    ]
    np.savetxt(
        path,
        decomposition.timecourses[:, chosen],
        fmt="%.6f",
        header="\n".join(header),
        comments="# ",
    )
    return path


__all__ = ["Decomposition", "decompose", "max_components", "read_mask", "write_ortvec"]
