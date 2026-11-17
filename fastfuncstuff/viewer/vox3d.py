"""The 3-D voxel window's arithmetic: what the ray marcher draws, without the GPU.

vox3d renders volumes, not meshes. The domain -- the overlay's grid, or the
anatomy's when there is no overlay -- is drawn as its bounding box with front
faces culled, so every pixel the box covers runs the fragment shader once, from
inside the box or out. The shader marches the ray from the eye through that
pixel between where it enters and leaves the box, sampling the overlay and the
anatomy through their own mm-to-texture transforms (so the two need not share a
grid, any more than they do in the slices), and reduces the samples by the
render mode:

* ``mip`` -- the sample with the largest thresholded statistic, coloured by the
  overlay there, over a mean ("X-ray") of the anatomy. The view for a vessel
  tree: vessels wind through slices, and only a projection draws them whole.
* ``minip`` -- the smallest value (the overlay's, or the anatomy's when there
  is none: veins in an SWI). Exact zeros are outside the data and skipped.
* ``mean`` -- the average along the ray: an X-ray.
* ``first hit`` -- the first sample over threshold, lit by the overlay's
  gradient: an isosurface of the cut, which reads as a solid shape when turned.
* ``composite`` -- front-to-back alpha: the anatomy as faint fog, the
  thresholded overlay as coloured glass.

Picking mirrors the shader on the CPU (:func:`pick`): a click goes to the voxel
the pixel is showing -- the max a MIP drew, the surface a first hit drew -- so
clicking a vessel in a projection puts the crosshair *on* it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastfuncstuff.viewer.surface3d import _mat, _vec, texture_from_mm

RENDER_MODES = ("mip", "minip", "mean", "first hit", "composite")
#: Uniform block size; must match vox3d.frag / vox3d.vert.
UNIFORM_BYTES = 4 * 64 + 6 * 16
#: The shader's loop bound; more samples than this on one ray are not taken.
MAX_STEPS = 1024
#: Sign modes as the layer stack names them, in the shader's numbering.
SIGN_BOTH, SIGN_POS, SIGN_NEG = 0, 1, 2


def box_corners(shape: tuple[int, int, int], affine: np.ndarray) -> np.ndarray:
    """``(8, 3)`` scanner-mm corners of a grid's outer voxel faces.

    Corner ``c`` has bit 0 for i, 1 for j, 2 for k: ``(c >> axis) & 1`` says
    whether it is at the low (``-0.5``) or high (``n - 0.5``) face.
    """
    aff = np.asarray(affine, np.float64)
    out = np.empty((8, 3))
    for c in range(8):
        ijk = np.array([(shape[a] - 0.5) if (c >> a) & 1 else -0.5 for a in range(3)])
        out[c] = aff[:3, :3] @ ijk + aff[:3, 3]
    return out


#: The box's twelve triangles, as corner indices. Winding does not matter: the
#: pipeline culls nothing, and the shader only uses the fragment's position.
BOX_TRIANGLES = np.array(
    [
        [0, 1, 3], [0, 3, 2],  # k low
        [4, 6, 7], [4, 7, 5],  # k high
        [0, 4, 5], [0, 5, 1],  # j low
        [2, 3, 7], [2, 7, 6],  # j high
        [0, 2, 6], [0, 6, 4],  # i low
        [1, 5, 7], [1, 7, 3],  # i high
    ],
    dtype=np.uint32,
)  # fmt: skip
#: The box's twelve edges, for the wireframe drawn over the render.
BOX_EDGES = np.array(
    [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
)


def box_vertices(shape: tuple[int, int, int], affine: np.ndarray) -> np.ndarray:
    """``(36, 3)`` float32 triangle-list positions of the box, ready to upload."""
    return box_corners(shape, affine)[BOX_TRIANGLES.reshape(-1)].astype(np.float32)


@dataclass
class VolumeSlot:
    """One volume as the canvas needs it: voxels, where they are, how to colour them.

    ``key`` identifies the voxels; the texture re-uploads only when it changes,
    so a threshold or colormap change touches the uniforms and the LUT alone.
    """

    key: tuple
    #: ``(nz, ny, nx)`` float32, NaN already zeroed (surface3d.texture_data).
    data: np.ndarray
    #: mm -> normalised texture coordinates.
    frame: np.ndarray
    shape: tuple[int, int, int]
    affine: np.ndarray
    lo: float = 0.0
    hi: float = 1.0
    threshold: float = 0.0
    opacity: float = 1.0
    sign_mode: int = SIGN_BOTH
    #: ``(256, 4)`` uint8 colour table; the anatomy's is grey.
    lut: np.ndarray | None = None
    #: A second volume on the same grid to cut on (a GLM's t under its beta).
    stat: np.ndarray | None = None


def slot_frame(shape: tuple[int, int, int], affine: np.ndarray) -> np.ndarray:
    return texture_from_mm(shape, affine)


def pack_uniforms(
    mvp: np.ndarray,
    eye: np.ndarray,
    domain: VolumeSlot,
    *,
    overlay: VolumeSlot | None,
    anatomy: VolumeSlot | None,
    mode: str,
    samples_per_voxel: float = 1.5,
    anatomy_opacity: float = 0.5,
    clip: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bytes:
    """The uniform block as bytes, matrices column-major as GLSL reads them.

    ``clip`` is a world plane ``(nx, ny, nz, d)``: points with ``n . p > d`` are
    cut away. A zero normal is no cut.
    """
    eye = np.asarray(eye, np.float64)
    density = float(samples_per_voxel) * float(max(domain.shape))
    ov = overlay
    an = anatomy
    identity = np.eye(4)
    parts = [
        _mat(mvp),
        _mat(domain.frame),
        _mat(ov.frame if ov is not None else identity),
        _mat(an.frame if an is not None else identity),
        _vec(eye[0], eye[1], eye[2], density),
        _vec(
            *((ov.lo, ov.hi, ov.threshold, ov.opacity) if ov is not None else (0.0, 1.0, 0.0, 0.0))
        ),
        _vec(*((an.lo, an.hi) if an is not None else (0.0, 1.0)), float(anatomy_opacity), 0.0),
        _vec(
            float(RENDER_MODES.index(mode)),
            float(ov.sign_mode if ov is not None else SIGN_BOTH),
            1.0 if ov is not None else 0.0,
            1.0 if an is not None else 0.0,
        ),
        _vec(*clip),
        _vec(*background, 1.0 if ov is not None and ov.stat is not None else 0.0),
    ]
    out = b"".join(parts)
    assert len(out) == UNIFORM_BYTES
    return out


# -- the CPU side of the ray --------------------------------------------------


def _box_span(
    origin: np.ndarray, direction: np.ndarray, frame: np.ndarray
) -> tuple[float, float] | None:
    """Ray parameters (mm along ``direction``) where it is inside a grid's box."""
    a = frame[:3, :3] @ origin + frame[:3, 3]
    d = frame[:3, :3] @ direction
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(np.abs(d) > 1e-12, 1.0 / d, np.inf)
        t0 = (0.0 - a) * inv
        t1 = (1.0 - a) * inv
    t0 = np.where(np.isnan(t0), -np.inf, t0)
    t1 = np.where(np.isnan(t1), np.inf, t1)
    lo = float(np.max(np.minimum(t0, t1)))
    hi = float(np.min(np.maximum(t0, t1)))
    lo = max(lo, 0.0)
    return (lo, hi) if hi > lo else None


def _sample(volume: np.ndarray, frame: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Nearest-voxel values of an ``(nx, ny, nz)`` volume at mm points; NaN outside."""
    shape = np.asarray(volume.shape[:3])
    tc = points @ frame[:3, :3].T + frame[:3, 3]
    ijk = np.floor(tc * shape).astype(int)
    inside = np.all((ijk >= 0) & (ijk < shape), axis=1)
    out = np.full(len(points), np.nan)
    i = ijk[inside]
    out[inside] = volume[i[:, 0], i[:, 1], i[:, 2]]
    return out


def _score(values: np.ndarray, sign_mode: int) -> np.ndarray:
    if sign_mode == SIGN_POS:
        return values
    if sign_mode == SIGN_NEG:
        return -values
    return np.abs(values)


def pick(
    origin: np.ndarray,
    direction: np.ndarray,
    *,
    volume: np.ndarray,
    affine: np.ndarray,
    mode: str,
    threshold: float,
    sign_mode: int = SIGN_BOTH,
    stat: np.ndarray | None = None,
    clip: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    step_mm: float | None = None,
) -> np.ndarray | None:
    """The mm point a pixel's ray is showing in ``volume``, or ``None``.

    ``mip``: where the thresholded statistic is largest along the ray, if it
    passes the threshold. ``minip``: the smallest nonzero value. Every other
    mode: the first sample over threshold -- what a first hit draws, and the
    nearest thing a composite shows. ``volume`` is ``(nx, ny, nz)``.
    """
    volume = np.nan_to_num(np.asarray(volume, np.float32), nan=0.0)
    origin = np.asarray(origin, np.float64)
    direction = np.asarray(direction, np.float64)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    frame = texture_from_mm(volume.shape[:3], affine)
    span = _box_span(origin, direction, frame)
    if span is None:
        return None
    lo, hi = span
    normal = np.asarray(clip[:3], np.float64)
    if np.any(normal):
        f0 = float(normal @ origin) - clip[3]
        f1 = float(normal @ direction)
        if abs(f1) < 1e-12:
            if f0 > 0:
                return None
        elif f1 > 0:
            hi = min(hi, -f0 / f1)
        else:
            lo = max(lo, -f0 / f1)
        if hi <= lo:
            return None
    if step_mm is None:
        step_mm = 0.5 * float(np.min(np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)))
    n = int(min(max(np.ceil((hi - lo) / step_mm), 1), 4 * MAX_STEPS))
    t = lo + (np.arange(n) + 0.5) * (hi - lo) / n
    pts = origin[None, :] + t[:, None] * direction[None, :]
    values = _sample(volume, frame, pts)
    cut = values if stat is None else _sample(np.nan_to_num(stat, nan=0.0), frame, pts)
    score = _score(np.nan_to_num(cut, nan=0.0), sign_mode)
    if mode == "mip":
        k = int(np.argmax(score))
        return pts[k] if score[k] >= threshold and score[k] > 0 else None
    if mode == "minip":
        valid = np.isfinite(values) & (values != 0)
        if not valid.any():
            return None
        k = int(np.flatnonzero(valid)[np.argmin(values[valid])])
        return pts[k]
    over = np.flatnonzero((score >= threshold) & (score > 0))
    return pts[over[0]] if over.size else None


def crosshair_segments(
    mm: np.ndarray, shape: tuple[int, int, int], affine: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """The three grid axes through a point, clipped to the box: what the window draws."""
    aff = np.asarray(affine, np.float64)
    ijk = np.linalg.solve(aff[:3, :3], np.asarray(mm, np.float64) - aff[:3, 3])
    out = []
    for axis in range(3):
        a, b = ijk.copy(), ijk.copy()
        a[axis], b[axis] = -0.5, shape[axis] - 0.5
        out.append((aff[:3, :3] @ a + aff[:3, 3], aff[:3, :3] @ b + aff[:3, 3]))
    return out


__all__ = [
    "BOX_EDGES",
    "BOX_TRIANGLES",
    "MAX_STEPS",
    "RENDER_MODES",
    "UNIFORM_BYTES",
    "VolumeSlot",
    "box_corners",
    "box_vertices",
    "crosshair_segments",
    "pack_uniforms",
    "pick",
]
