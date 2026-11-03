"""Pair volume and surface runs in time while retaining independent geometries."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.cli_utils import LoadResult, load_and_preprocess_runs
from fastfuncstuff.io.afni import load_nifti, save_nifti
from fastfuncstuff.io.gifti import is_gifti, surface_meta

_VOLUME_EXTS = (".nii.gz", ".nii.zst", ".nii")


@dataclass(frozen=True)
class PairedHeader:
    """Header for a location axis that is a volume grid followed by a surface mesh.

    Paired data is laid out on a pseudo-grid of shape ``(n_grid + n_vertices, 1, 1)``
    so mask, restrict and scatter code written for one domain works unchanged.
    ``save_nifti`` recognises this header and writes ``<stem>_volume<ext>`` and
    ``<stem>_surface<surface_ext>``, each with its own geometry.
    """

    volume_shape: tuple
    volume_affine: np.ndarray
    volume_header: object
    surface_shape: tuple
    surface_header: object
    surface_ext: str = ".shape.gii"

    @property
    def n_volume(self) -> int:
        return int(np.prod(self.volume_shape))

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.n_volume + int(np.prod(self.surface_shape)), 1, 1)

    def copy(self) -> PairedHeader:
        return self

    def paths(self, output_path: str | Path) -> tuple[str, str]:
        path = str(output_path)
        ext = next((e for e in _VOLUME_EXTS if path.endswith(e)), None)
        if ext is None:
            raise ValueError(f"Paired outputs need a NIfTI path to split, got {path}")
        stem = path[: -len(ext)]
        return f"{stem}_volume{ext}", f"{stem}_surface{self.surface_ext}"

    def save(self, data: np.ndarray, output_path: str | Path, **kwargs) -> None:
        from fastfuncstuff.io.afni import save_nifti

        data = np.asarray(data)
        if data.shape[:3] != self.shape:
            raise ValueError(f"Paired data must be shaped {self.shape}, got {data.shape}")
        rest = data.shape[3:]
        flat = data.reshape(self.shape[0], *rest)
        volume_path, surface_path = self.paths(output_path)
        save_nifti(
            flat[: self.n_volume].reshape(*self.volume_shape, *rest),
            volume_path,
            affine=self.volume_affine,
            header=self.volume_header,
            **kwargs,
        )
        save_nifti(
            flat[self.n_volume :].reshape(*self.surface_shape, *rest),
            surface_path,
            header=self.surface_header,
            **kwargs,
        )


@dataclass
class PairedRuns:
    data: torch.Tensor  # (volume locations + surface locations, total time)
    run_starts: list[int]
    tr: float
    volume: LoadResult  # geometry only; its data is released once concatenated
    surface: LoadResult
    header: PairedHeader
    keep_on_cpu: bool

    @property
    def mask_flat(self) -> np.ndarray:
        """In-analysis locations on the pseudo-grid (see PairedHeader)."""
        return np.concatenate(
            [
                d.mask_flat
                if d.mask_flat is not None
                else np.ones(int(np.prod(d.volume_shape)), bool)
                for d in (self.volume, self.surface)
            ]
        )

    @property
    def mask(self) -> np.ndarray:
        return self.mask_flat.reshape(self.header.shape)

    @property
    def as_load_result(self) -> LoadResult:
        """The paired data as the ``LoadResult`` a single-domain tool already takes.

        The pseudo-grid is the whole trick: locations are a volume grid followed by
        a mesh, so a tool's masking, screening, chunking and saving code is
        unchanged -- a vector of locations is a vector of locations, whether the
        entry is a voxel or a vertex. ``nifti_header`` is the PairedHeader, which
        ``save_nifti`` recognises and splits into ``<stem>_volume`` and
        ``<stem>_surface`` with each domain's own geometry. So a tool gains paired
        input by choosing a loader, not by growing a second output path.
        """
        mask_flat = self.mask_flat
        return replace(
            self.volume,
            data=self.data,
            run_starts=self.run_starts,
            tr=self.tr,
            volume_shape=self.header.shape,
            mask=mask_flat.reshape(self.header.shape),
            mask_flat=mask_flat,
            n_voxels=int(self.data.shape[0]),
            n_timepoints=int(self.data.shape[1]),
            n_runs=len(self.run_starts),
            keep_on_cpu=self.keep_on_cpu,
            nifti_header=self.header,
        )

    def split_map(
        self, values: torch.Tensor | np.ndarray
    ) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
        """Split location-first results without expanding either domain's mask."""
        if values.shape[0] != self.data.shape[0]:
            raise ValueError(f"Expected {self.data.shape[0]} locations, got {values.shape[0]}")
        return values[: self.volume.n_voxels], values[self.volume.n_voxels :]

    def save_map(
        self, values: torch.Tensor | np.ndarray, volume_path: str | Path, surface_path: str | Path
    ) -> None:
        """Write a location-first map to two files with each input's own geometry."""
        for part, domain, path, surface in zip(
            self.split_map(values),
            (self.volume, self.surface),
            (volume_path, surface_path),
            (False, True),
            strict=True,
        ):
            if is_gifti(str(path)) != surface:
                raise ValueError(
                    f"Wrong output format for {'surface' if surface else 'volume'}: {path}"
                )
            arr = (
                part.detach().cpu().numpy() if isinstance(part, torch.Tensor) else np.asarray(part)
            )
            shape = tuple(domain.volume_shape)
            if domain.mask_flat is not None:
                full = np.zeros((int(np.prod(shape)), *arr.shape[1:]), dtype=arr.dtype)
                full[domain.mask_flat] = arr
                arr = full
            arr = arr.reshape(*shape, *arr.shape[1:])
            save_nifti(arr, output_path=path, affine=domain.affine, header=domain.nifti_header)


def _declared_tr(path: str) -> float | None:
    """Read an actual time-unit declaration, never NIfTI's default pixdim=1."""
    img = load_nifti(path)
    if is_gifti(path):
        value = (surface_meta(img.header) or {}).get("TR_seconds")
        return float(value) if value else None
    if len(img.shape) != 4:
        raise ValueError(f"Expected 4-D run: {path}")
    from fastfuncstuff.io.headers import header_tr

    return header_tr(img, require_units=True)


def load_paired_runs(
    volume_files: list[str],
    surface_files: list[str],
    *,
    tr: float | None = None,
    volume_mask: str | None = None,
    surface_mask: str | None = None,
    blur_fwhm: float | None = None,
    device: torch.device = torch.device("cpu"),
    force_cpu: bool = False,
    load_threads: int | None = None,
    drop_first: int = 0,
    drop_last: int = 0,
    surface_ext: str = ".shape.gii",
) -> PairedRuns:
    """Match two spatial domains by run index, then concatenate their locations.

    Inputs must each form a stable geometry across runs. Temporal pairing is
    positional: one volume and one surface file represent the same physical run.
    """
    if not volume_files or len(volume_files) != len(surface_files):
        raise ValueError("Pass one volume and one surface file per paired run")
    if any(is_gifti(f) for f in volume_files) or any(not is_gifti(f) for f in surface_files):
        raise ValueError("Volume files must be NIfTI/AFNI and surface files must be GIfTI")
    if tr is not None and (not np.isfinite(tr) or tr <= 0):
        raise ValueError("TR must be a positive finite number of seconds")
    if drop_first < 0 or drop_last < 0:
        raise ValueError("Dropped frame counts must be nonnegative")
    declared = [_declared_tr(f) for f in [*volume_files, *surface_files]]
    known = [v for v in declared if v is not None]
    effective_tr = tr if tr is not None else (known[0] if known else None)
    if effective_tr is None:
        raise ValueError("Neither input declares a TR; pass -tr explicitly")
    if any(
        not np.isfinite(v) or v <= 0 or not np.isclose(v, effective_tr, rtol=1e-5) for v in known
    ):
        raise ValueError("Paired runs declare inconsistent TR values")
    kwargs = dict(
        tr=effective_tr,
        blur_fwhm=blur_fwhm,
        device=device,
        force_cpu=force_cpu,
        load_threads=load_threads,
        drop_first=drop_first,
        drop_last=drop_last,
        verbose=False,
    )
    volume = load_and_preprocess_runs(volume_files, mask_file=volume_mask, **kwargs)
    surface = load_and_preprocess_runs(surface_files, mask_file=surface_mask, **kwargs)
    if volume.run_starts != surface.run_starts or volume.n_timepoints != surface.n_timepoints:
        raise ValueError("Paired runs must have equal time lengths after trimming")
    if not np.isclose(volume.tr, surface.tr):
        raise ValueError("Paired runs have inconsistent TR")
    # Each domain sized its own device strategy; if either had to stay on the
    # host the combined matrix must too.
    keep_on_cpu = volume.keep_on_cpu or surface.keep_on_cpu
    target = torch.device("cpu") if keep_on_cpu else volume.data.device
    data = torch.cat((volume.data.to(target), surface.data.to(target)), dim=0)
    # The per-domain copies would otherwise sit beside the concatenation for the
    # whole fit, doubling the resident data.
    empty = torch.empty(0)
    volume, surface = replace(volume, data=empty), replace(surface, data=empty)
    header = PairedHeader(
        tuple(volume.volume_shape),
        volume.affine,
        volume.nifti_header,
        tuple(surface.volume_shape),
        surface.nifti_header,
        surface_ext,
    )
    return PairedRuns(data, volume.run_starts, effective_tr, volume, surface, header, keep_on_cpu)
