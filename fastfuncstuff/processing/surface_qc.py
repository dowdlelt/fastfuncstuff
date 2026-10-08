"""What the surface projection actually read, and how much it blurred: QC maps.

Three questions [[Wang 2022]] says a fine-scale surface analysis has to answer, each
measured through the run's own chain (motion frame 0, distortion, registration) with
the same points the projection read:

* **Which acquired voxels fed the surface, and how often** -- ``samples``: every
  footprint read assigned to its nearest native EPI voxel, counted on the native grid.
  Zeros inside the ribbon are voxels a coarse mesh missed; a smooth band means none
  were.
* **How big an EPI voxel is in the anatomy** -- ``voxel_volume``: ``1 / |det J|`` of
  the composed anatomy-mm -> source-voxel map at each vertex (mm^3). Distortion
  correction moves signal; it cannot make the acquisition finer where the field
  compressed it, or coarser where it stretched it. Rigid motion does not change it.
* **How much the projection smoothed** -- ``blur_fwhm``: white noise pushed through the
  same chain, kernel, footprint fold and depth weights; the noise SD left at each
  vertex is mapped to the FWHM of the Gaussian on the native grid that leaves the same
  SD (the paper's white-noise meter). ``noise_ratio`` is that SD itself:
  ``1 / noise_ratio^2`` is how many independent voxels a vertex averages.

The coordinate maps are exact: linear interpolation reads a linear function exactly,
and a constant volume says which reads were wholly inside the grid. The blur map is
a Monte Carlo over ``n_frames`` noise volumes, so its precision is about
``1 / sqrt(2 n_frames)`` in ``noise_ratio``. Slice timing and ``-jac`` are left out on
purpose: one interpolates in time (which also lowers a white-noise SD), the other
scales intensity; neither is spatial smoothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.surface.projection import depth_surfaces

__all__ = ["gaussian_noise_ratio", "noise_ratio_to_fwhm", "surface_qc"]

_FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def gaussian_noise_ratio(fwhm: np.ndarray, zooms) -> np.ndarray:
    """White-noise SD left by a unit-sum Gaussian of ``fwhm`` mm sampled on a grid of
    voxel size ``zooms``: the product over axes of ``sqrt(sum g_k^2)``."""
    f = np.atleast_1d(np.asarray(fwhm, np.float64))
    out = np.ones_like(f)
    for d in np.asarray(zooms, np.float64)[:3]:
        for i, fw in enumerate(f):
            if fw <= 0:
                continue
            sigma = fw / _FWHM_PER_SIGMA / d
            k = np.arange(-int(np.ceil(6 * sigma)) - 1, int(np.ceil(6 * sigma)) + 2)
            g = np.exp(-0.5 * (k / sigma) ** 2)
            g /= g.sum()
            out[i] *= np.sqrt((g**2).sum())
    return out


def noise_ratio_to_fwhm(ratio: np.ndarray, zooms, max_fwhm: float = 40.0) -> np.ndarray:
    """Invert :func:`gaussian_noise_ratio`: the Gaussian FWHM (mm) that leaves ``ratio``.

    Below about half a voxel a sampled Gaussian barely changes the noise (the lookup
    plateaus, as in the paper), so ratios near 1 read as 0 rather than a small number
    the grid cannot tell apart. Ratios below the ``max_fwhm`` level read as it.
    """
    grid = np.linspace(0.0, max_fwhm, 801)
    table = gaussian_noise_ratio(grid, zooms)  # decreasing
    r = np.asarray(ratio, np.float64)
    out = np.interp(-r, -table, grid)
    return np.where(r >= 1.0, 0.0, out).astype(np.float32)


def _forge(source_image, points, nwarp_specs, master_path, source_path, **kw):
    from .nwarpforge import nwarpforge

    out = nwarpforge(
        source_path=source_path,
        nwarp_specs=nwarp_specs,
        prefix="",
        master_path=master_path,
        points=points,
        source_image=source_image,
        **kw,
    )
    assert out is not None
    return out


def _header_3d(header: dict, shape_zyx) -> dict:
    hdr = header["header"].copy()
    hdr.set_data_shape(tuple(int(n) for n in shape_zyx[::-1]))
    return {"affine": header["affine"], "header": hdr}


def surface_qc(
    source_path: str,
    nwarp_specs: list[str],
    master_path: str,
    targets,
    prefix: str,
    weights: np.ndarray,
    source_image=None,
    n_frames: int = 64,
    interp: str = "wsinc5",
    ainterp: str = "cubic",
    device: torch.device | None = None,
    step_mm: float = 0.25,
    seed: int = 0,
    verb: int = 1,
) -> list[Path]:
    """Write the QC maps for ``targets`` (:func:`surface_targets` output) at ``prefix``.

    Per target ``{prefix}.{space}.{hemi}``: ``.voxel_volume``, ``.blur_fwhm`` and
    ``.noise_ratio`` as ``.shape.gii``; per space ``{prefix}.{space}.samples.nii.gz`` on
    the native EPI grid. ``weights`` are the depth weights the data were combined with.
    """
    from fastfuncstuff.io.gifti import save_gifti_data

    from .io import load_image, save_image
    from .surface_projection import SurfaceFold

    if source_image is None:
        source_image = load_image(source_path, device=None)
    src, header = source_image
    shape = tuple(src.shape[-3:])  # (nz, ny, nx)
    zooms = np.asarray(header["header"].get_zooms()[:3], np.float64)
    nominal = float(np.prod(zooms))
    h3 = _header_3d(header, shape)
    kw = dict(interp="linear", ainterp=ainterp, device=device, verb=0)

    # --- exact coordinates: where each point reads the native grid -------------------
    centres = [
        depth_surfaces(t.white, t.pial, t.faces, t.sampling.fractions).reshape(-1, 3)
        for t in targets
    ]
    c0 = np.concatenate(centres)
    steps = [s * step_mm * np.eye(3)[a] for a in range(3) for s in (1.0, -1.0)]
    reads = np.concatenate([t.sampling.points for t in targets])
    pts = np.concatenate([c0, *(c0 + d for d in steps), reads])
    zz, yy, xx = torch.meshgrid(
        *(torch.arange(n, dtype=torch.float32) for n in shape), indexing="ij"
    )
    coord = []
    for vol in (xx, yy, zz, torch.ones(shape)):
        coord.append(_forge((vol, h3), pts, nwarp_specs, master_path, source_path, **kw))
    xyz = torch.stack(coord[:3], 1).cpu().numpy().astype(np.float64)  # (N, 3) source ijk
    inside = np.abs(coord[3].cpu().numpy() - 1.0) < 1e-3
    n_c = c0.shape[0]

    written: list[Path] = []
    # samples: footprint reads per native voxel, per space (all hemispheres and depths)
    r_ijk, r_in = xyz[7 * n_c :], inside[7 * n_c :]
    edges = np.cumsum([t.sampling.points.shape[0] for t in targets])[:-1]
    per_target = list(zip(np.split(r_ijk, edges), np.split(r_in, edges), strict=True))
    for space in dict.fromkeys(t.mesh for t in targets):
        count = np.zeros(shape, np.float32)
        for t, (ijk, ok) in zip(targets, per_target, strict=True):
            if t.mesh != space:
                continue
            idx = np.rint(ijk[ok]).astype(np.int64)
            good = ((idx >= 0) & (idx < np.asarray(shape[::-1]))).all(axis=1)
            x, y, z = idx[good].T
            np.add.at(count, (z, y, x), 1.0)
        path = Path(f"{prefix}.{space}.samples.nii.gz")
        save_image(torch.from_numpy(count), path, h3)
        written.append(path)

    # voxel_volume: 1/|det d(ijk)/d(mm)| at each vertex and depth, depth-weighted
    blocks = [xyz[i * n_c : (i + 1) * n_c] for i in range(7)]
    ok = np.all([inside[i * n_c : (i + 1) * n_c] for i in range(7)], axis=0)
    jac = np.stack([(blocks[1 + 2 * a] - blocks[2 + 2 * a]) / (2 * step_mm) for a in range(3)], 2)
    det = np.abs(np.linalg.det(jac))
    vvol = np.divide(1.0, det, out=np.zeros_like(det), where=ok & (det > 1e-9))

    # blur: white noise through the same chain, kernel, fold and depth weights
    # The noise rides the run's own per-frame chain, so one pass cannot outrun the
    # series: passes of at most T frames until n_frames. Its mean is 0 by construction
    # (the fold renormalises, never offsets), so the SD is the RMS.
    n_frames = int(max(2, n_frames))
    per_pass = min(n_frames, src.shape[0]) if src.ndim == 4 else n_frames
    gen = torch.Generator().manual_seed(seed)
    fold = SurfaceFold(targets, weights)
    sumsq = [np.zeros(t.sampling.n_vertices) for t in targets]
    cover = [np.ones(t.sampling.n_vertices) for t in targets]
    done = 0
    while done < n_frames:
        m = min(per_pass, n_frames - done)
        noise = torch.randn((m, *shape), generator=gen, dtype=torch.float32)
        hdr = header if m > 1 else h3  # one frame is a 3-D source
        out = _forge(
            (noise if m > 1 else noise[0], hdr), reads, nwarp_specs, master_path, source_path,
            interp=interp, ainterp=ainterp, device=device, verb=0, point_reducer=fold,
            time_range=(0, m) if m > 1 else None,
        ).cpu().numpy()  # fmt: skip
        for i, (_, combined, _, cov) in enumerate(fold.split(out)):
            sumsq[i] += (combined.astype(np.float64) ** 2).sum(1)
            cover[i] = np.minimum(cover[i], cov.min(axis=0))
        done += m

    starts = np.cumsum([0] + [c.shape[0] for c in centres])
    for i, t in enumerate(targets):
        stem = f"{prefix}.{t.mesh}.{t.hemi}"
        k, v = t.sampling.n_depths, t.sampling.n_vertices
        vv = vvol[starts[i] : starts[i + 1]].reshape(k, v)
        w = weights[:, None] * (vv > 0)
        vox = np.divide((w * vv).sum(0), w.sum(0), out=np.zeros(v), where=w.sum(0) > 0)
        ratio = np.sqrt(sumsq[i] / n_frames).astype(np.float32)
        fwhm = noise_ratio_to_fwhm(ratio, zooms)
        fwhm[cover[i] == 0] = 0.0
        meta = {
            **t.meta,
            "nominal_voxel_mm3": f"{nominal:.4g}",
            "voxel_mm": " ".join(f"{z:.4g}" for z in zooms),
            "noise_frames": str(n_frames),
            "geometry": Path(f"{stem}.midthickness.surf.gii").name,
        }
        for name, vals in (("voxel_volume", vox), ("blur_fwhm", fwhm), ("noise_ratio", ratio)):
            path = Path(f"{stem}.{name}.shape.gii")
            save_gifti_data(path, np.asarray(vals, np.float32), meta, time_series=False)
            written.append(path)
        if verb >= 1:
            m = cover[i] > 0.99
            if m.any():
                print(
                    f"  QC {t.mesh} {t.hemi}: blur FWHM median {np.median(fwhm[m]):.2f} mm "
                    f"(5-95% {np.percentile(fwhm[m], 5):.2f}-{np.percentile(fwhm[m], 95):.2f}); "
                    f"EPI voxel {np.median(vox[m]):.3g} mm^3 (nominal {nominal:.3g})"
                )
    return written
