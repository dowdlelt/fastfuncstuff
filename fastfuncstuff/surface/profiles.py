"""Intensity profiles across the cortical ribbon: a whole-brain QC column.

For every vertex, the anatomy is read along the line from its white point to
its matching pial point -- extended a little past both ends -- with each depth
averaged over a small tube around that line to quiet the noise. Grey matter is
roughly uniform along that line; where a surface is wrong the profile says so:
a pial surface that ran out through CSF shows a dark dip *inside* the ribbon
(and dura beyond it), one that stopped short shows grey-matter brightness
continuing past pial.

These are flags for the eye, not corrections. Profile evidence alone cannot
say which anatomical structure owns an edge -- see the wiki's "T1 cortical
surface refinement experiments" -- which is why the column is clickable: it
takes the person to the place, and the editor does the rest.

Rows are ordered back to front in thin coronal slabs, and within a slab around
that slab's outline of the cortex, so neighbouring rows are neighbouring
cortex: reading down the column is walking the ribbon slice by slice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fastfuncstuff.memory import get_available_memory
from fastfuncstuff.processing.interp import trilinear_interpolate


@dataclass(frozen=True)
class ProfileSpec:
    #: ``fraction``: white and pial land on the same columns for every vertex
    #: (thickness stretched to uniform). ``mm``: one column per ``step_mm``
    #: from white, so pial lands wherever the cortex is thick (bumpy).
    mode: str = "fraction"
    #: Samples across the ribbon in fraction mode, white (0) to pial (1).
    n_inner: int = 21
    #: How far past each end to look, mm -- into white matter, out past pial.
    inside_mm: float = 1.5
    outside_mm: float = 2.5
    #: Sample spacing for the margins (and for the whole line in mm mode).
    step_mm: float = 0.25
    #: Deepest point of the line in mm mode, measured from white.
    mm_max: float = 6.0
    #: Tube around each line: radius (mm) and points on its ring (plus the
    #: centre). Zero radius reads the line alone.
    tube_radius: float = 0.5
    tube_points: int = 6


@dataclass
class Profiles:
    values: np.ndarray  # (V, S) float32
    #: Column coordinates. Fraction mode: ribbon columns are fractions of the
    #: local thickness (0 white, 1 pial) and margin columns are mm from the
    #: nearer end (negative into WM, positive past pial). mm mode: mm from white.
    columns: np.ndarray  # (S,)
    #: Which part of the line each column is: -1 inside white, 0 ribbon (in mm
    #: mode: anything from white outward), +1 beyond pial.
    kind: np.ndarray  # (S,) int8
    mode: str
    thickness: np.ndarray  # (V,) mm


def _columns(spec: ProfileSpec) -> tuple[np.ndarray, np.ndarray]:
    """Column coordinates and kinds.

    Fraction mode: margins in mm (negative inside white, positive beyond
    pial), ribbon in fractions. mm mode: mm from white throughout.
    """
    step = spec.step_mm
    inside = -np.arange(int(round(spec.inside_mm / step)), 0, -1) * step
    if spec.mode == "mm":
        cols = np.concatenate([inside, np.arange(0.0, spec.mm_max + 1e-9, step)])
        kind = np.where(cols < 0, -1, 0).astype(np.int8)
        return cols, kind
    if spec.mode != "fraction":
        raise ValueError(f"profile mode must be 'fraction' or 'mm', got {spec.mode!r}")
    ribbon = np.linspace(0.0, 1.0, spec.n_inner)
    outside = np.arange(1, int(round(spec.outside_mm / step)) + 1) * step
    cols = np.concatenate([inside, ribbon, outside])
    kind = np.concatenate(
        [np.full(inside.size, -1), np.zeros(ribbon.size), np.ones(outside.size)]
    ).astype(np.int8)
    return cols, kind


def _perpendicular(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Two unit vectors perpendicular to each row of ``u`` and to each other."""
    helper = torch.zeros_like(u)
    # Use the axis least aligned with u, so the cross product never vanishes.
    helper[torch.arange(u.shape[0]), torch.argmin(u.abs(), dim=1)] = 1.0
    a = torch.linalg.cross(u, helper)
    a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-12)
    b = torch.linalg.cross(u, a)
    return a, b


def sample_profiles(
    white: np.ndarray,
    pial: np.ndarray,
    volume: np.ndarray | torch.Tensor,
    affine: np.ndarray,
    spec: ProfileSpec = ProfileSpec(),
    *,
    device: torch.device | None = None,
) -> Profiles:
    """Profiles for every vertex, chunked to the device's free memory."""
    device = device or torch.device("cpu")
    cols, kind = _columns(spec)
    vol = torch.as_tensor(np.asarray(volume, np.float32) if not torch.is_tensor(volume) else volume)
    if vol.ndim == 4:
        vol = vol[..., 0]
    vol = vol.to(device=device, dtype=torch.float32)
    inv = torch.as_tensor(np.linalg.inv(np.asarray(affine, np.float64)), dtype=torch.float32)
    inv = inv.to(device)
    w_all = torch.as_tensor(np.asarray(white, np.float32))
    p_all = torch.as_tensor(np.asarray(pial, np.float32))
    n_v, n_s = w_all.shape[0], cols.size
    t_pts = 1 + (spec.tube_points if spec.tube_radius > 0 else 0)
    # Per vertex: S*T points x (3 coords + 3 index + value), float32, with
    # grid_sample's own temporaries -- a generous 12 floats per point.
    per_vertex = n_s * t_pts * 12 * 4
    chunk = int(max(1024, min(n_v, get_available_memory(device, empty_cache=False) // per_vertex)))
    col_t = torch.as_tensor(cols, dtype=torch.float32, device=device)
    kind_t = torch.as_tensor(kind, device=device)
    angles = torch.arange(spec.tube_points, device=device, dtype=torch.float32)
    angles = angles * (2 * np.pi / max(spec.tube_points, 1))
    out = np.empty((n_v, n_s), np.float32)
    thick = np.empty(n_v, np.float32)
    for start in range(0, n_v, chunk):
        w = w_all[start : start + chunk].to(device)
        p = p_all[start : start + chunk].to(device)
        seg = p - w
        length = seg.norm(dim=1, keepdim=True)
        u = seg / length.clamp_min(1e-6)
        if spec.mode == "mm":
            offset = col_t[None, :] * torch.ones_like(length)  # mm from white
        else:
            # Ribbon columns are fractions of the local thickness; margins are
            # mm from the nearer end.
            ribbon = col_t[None, :] * length
            below = col_t[None, :] * torch.ones_like(length)
            beyond = length + col_t[None, :]
            offset = torch.where(
                kind_t[None, :] < 0, below, torch.where(kind_t[None, :] > 0, beyond, ribbon)
            )
        centre = w[:, None, :] + offset[..., None] * u[:, None, :]  # (n, S, 3)
        if t_pts > 1:
            a, b = _perpendicular(u)
            ring = spec.tube_radius * (
                torch.cos(angles)[None, :, None] * a[:, None, :]
                + torch.sin(angles)[None, :, None] * b[:, None, :]
            )  # (n, T-1, 3)
            ring = torch.cat([torch.zeros_like(ring[:, :1]), ring], dim=1)  # centre too
            pts = centre[:, :, None, :] + ring[:, None, :, :]  # (n, S, T, 3)
        else:
            pts = centre[:, :, None, :]
        flat = pts.reshape(-1, 3)
        ijk = flat @ inv[:3, :3].T + inv[:3, 3]
        # trilinear_interpolate reads (nz, ny, nx) with x fastest: the array's
        # last axis is its "x", so k -> x, j -> y, i -> z.
        vals = trilinear_interpolate(vol, ijk[:, 2], ijk[:, 1], ijk[:, 0])
        vals = vals.reshape(pts.shape[:3]).mean(dim=2)
        out[start : start + w.shape[0]] = vals.cpu().numpy()
        thick[start : start + w.shape[0]] = length[:, 0].cpu().numpy()
    return Profiles(out, cols, kind, spec.mode, thick)


def equivolume_fraction(alpha, white_area, pial_area):
    """Depth fraction (white 0 .. pial 1) enclosing volume fraction ``alpha``.

    Equivolume layering (Waehnert et al. 2014), in the per-vertex form
    pycortex uses: cortical area varies linearly with depth from the white
    area to the pial area, so the volume between white and depth rho is a
    quadratic in rho, solved here for rho. In a gyral crown (pial area >
    white) the outer layers are the thin ones, so the mid-volume surface
    sits nearer pial; in a fundus, nearer white. Equal areas give
    rho = alpha. Works elementwise on scalars or arrays; the
    fragment shader does the same per pixel.
    """
    a = np.asarray(alpha, np.float64)
    aw = np.asarray(white_area, np.float64)
    ap = np.asarray(pial_area, np.float64)
    delta = ap - aw
    root = np.sqrt(np.maximum((1.0 - a) * aw * aw + a * ap * ap, 0.0))
    flat = np.abs(delta) <= 1e-4 * np.maximum(aw + ap, 1e-12)
    return np.where(flat, a, (root - aw) / np.where(flat, 1.0, delta))


def sample_depths(
    white: np.ndarray,
    pial: np.ndarray,
    volume: np.ndarray,
    affine: np.ndarray,
    fractions: np.ndarray,
    *,
    white_area: np.ndarray | None = None,
    pial_area: np.ndarray | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    """A volume at given cortical depths for each vertex: ``(V, K)`` or ``(V, K, T)``.

    ``fractions`` run white (0) to pial (1); outside that range the line is
    extended linearly into white matter and past pial, so a profile can show
    where the ribbon starts and stops. Inside it, with both areas given, the
    fractions are **equivolume** (:func:`equivolume_fraction`), as the 3-D
    view samples. A 4-D volume ``(X, Y, Z, T)`` is read for every time point
    in one pass, time as channels -- the depth timecourses laminar models eat.
    """
    device = device or torch.device("cpu")
    frac = np.asarray(fractions, np.float64)
    w = np.asarray(white, np.float64)
    p = np.asarray(pial, np.float64)
    if white_area is not None and pial_area is not None:
        inside = (frac >= 0) & (frac <= 1)
        rho = np.broadcast_to(frac, (w.shape[0], frac.size)).copy()
        rho[:, inside] = equivolume_fraction(
            frac[None, inside], np.asarray(white_area)[:, None], np.asarray(pial_area)[:, None]
        )
    else:
        rho = np.broadcast_to(frac, (w.shape[0], frac.size))
    pts = w[:, None, :] + rho[..., None] * (p - w)[:, None, :]  # (V, K, 3)
    inv = np.linalg.inv(np.asarray(affine, np.float64))
    ijk = pts @ inv[:3, :3].T + inv[:3, 3]
    vol = np.asarray(volume, np.float32)
    four_d = vol.ndim == 4
    if not four_d:
        vol = vol[..., None]
    shape = np.array(vol.shape[:3], np.float64)
    # grid_sample's (x, y, z) index (W, H, D); the array is (X, Y, Z), taken
    # as (D, H, W) -- so the grid is (k, j, i), normalised to [-1, 1] at the
    # first and last voxel centres (align_corners=True).
    grid = 2.0 * ijk[..., ::-1] / np.maximum(shape[::-1] - 1, 1) - 1.0
    g = torch.as_tensor(grid.reshape(1, 1, 1, -1, 3), dtype=torch.float32, device=device)
    n_t = vol.shape[3]
    out = np.empty((pts.shape[0] * pts.shape[1], n_t), np.float32)
    per_channel = int(np.prod(vol.shape[:3])) * 4 + g.numel() * 4
    step = int(
        max(1, min(n_t, get_available_memory(device, empty_cache=False) // max(per_channel, 1)))
    )
    for t0 in range(0, n_t, step):
        block = np.ascontiguousarray(np.moveaxis(vol[..., t0 : t0 + step], 3, 0))
        x = torch.as_tensor(block, device=device)[None]  # (1, C, X, Y, Z)
        s = torch.nn.functional.grid_sample(
            x, g, mode="bilinear", padding_mode="zeros", align_corners=True
        )  # (1, C, 1, 1, N)
        out[:, t0 : t0 + step] = s[0, :, 0, 0, :].T.cpu().numpy()
    out = out.reshape(pts.shape[0], pts.shape[1], n_t)
    return out if four_d else out[..., 0]


@dataclass
class TissueLevels:
    wm: float
    gm: float
    csf: float


def tissue_levels(prof: Profiles) -> TissueLevels:
    """Whole-brain WM / GM / CSF levels read off the profiles themselves.

    GM is the mid-ribbon median; WM the median ~1 mm inside white; CSF the
    median of each profile's darkest point beyond pial (the darkest, because
    a narrow sulcus puts the next gyrus's grey matter there too).
    """
    v, cols, kind = prof.values, prof.columns, prof.kind
    if prof.mode == "fraction":
        mid = (kind == 0) & (cols >= 0.3) & (cols <= 0.7)
        beyond = kind > 0
    else:
        mid = (kind == 0) & (cols >= 0.3) & (cols <= 1.2)
        beyond = (kind == 0) & (cols >= 3.0)
    inside = (kind < 0) & (cols <= -0.75)
    gm = float(np.median(v[:, mid])) if mid.any() else float(np.median(v))
    wm = float(np.median(v[:, inside])) if inside.any() else gm
    csf = float(np.median(v[:, beyond].min(axis=1))) if beyond.any() else 0.0
    return TissueLevels(wm=wm, gm=gm, csf=csf)


#: Score names, in the order the window offers them.
SCORES = ("worst", "pial_out", "pial_short", "white_deep", "white_shallow", "nonlinear")

#: Robust deviations (MADs from the brain-wide typical profile) that map to a
#: score of 1. Four MADs is about 2.7 SD.
FULL_SCALE_MADS = 4.0


def _robust_unit(raw: np.ndarray, floor: float) -> np.ndarray:
    """How unusual each vertex's statistic is across the brain, mapped to [0, 1].

    Robust z against the statistic's own brain-wide median and MAD, so a
    statistic with a built-in bias -- the minimum of ten noisy columns always
    looks like a dip -- is judged against how it usually comes out, not
    against zero. ``floor`` keeps a near-noiseless image's tiny MAD from
    turning grid-level differences into huge z.
    """
    med = float(np.median(raw))
    mad = max(float(np.median(np.abs(raw - med))) * 1.4826, floor)
    return np.clip((raw - med) / mad / FULL_SCALE_MADS, 0.0, 1.0).astype(np.float32)


def profile_scores(prof: Profiles) -> dict[str, np.ndarray]:
    """Per-vertex flags in [0, 1]; higher is more suspicious.

    Each is a raw statistic per vertex, then judged by how unusual it is
    across the brain (:func:`_robust_unit`). The first version compared with
    global tissue means and flagged 21% of the brain as "pial ran out": the
    outermost ribbon columns are partial volume on every vertex. Judging each
    statistic against its own brain-wide distribution carries that, the
    image's contrast and blur, and the bias of taking a minimum.

    * ``pial_out`` -- darkest point of the outer ribbon (0.5-0.95, smoothed
      over 3 columns): something CSF-like inside; pial ran out, often to dura.
    * ``pial_short`` -- brightness 0.25-0.75 mm past pial: grey matter
      continuing outside. A tight sulcus does this legitimately.
    * ``white_deep`` -- brightness in the inner ribbon (0.05-0.25): white
      matter there; white sits too deep.
    * ``white_shallow`` -- darkness 0.25-0.75 mm inside white: grey matter
      there; white sits out in GM.
    * ``nonlinear`` -- RMS of the ribbon's residual from a straight line:
      the "should be roughly flat" check.
    * ``worst`` -- the maximum of the four placement flags.

    Fraction-mode profiles only; mm-mode columns do not line up with pial.
    """
    if prof.mode != "fraction":
        raise ValueError("scores need fraction-mode profiles (pial on a fixed column)")
    v, cols, kind = prof.values, prof.columns, prof.kind
    typical = np.median(v, axis=0)
    floor = 0.02 * max(float(np.ptp(typical)), 1e-6)
    ribbon = kind == 0
    outer = ribbon & (cols >= 0.5) & (cols <= 0.95)
    inner = ribbon & (cols >= 0.05) & (cols <= 0.25)
    past = (kind > 0) & (cols >= 0.25) & (cols <= 0.75)
    under = (kind < 0) & (cols <= -0.25) & (cols >= -0.75)
    smooth = v.copy()
    smooth[:, 1:-1] = (v[:, :-2] + v[:, 1:-1] + v[:, 2:]) / 3.0
    x = cols[ribbon]
    design = np.stack([np.ones_like(x), x], 1)
    coef, *_ = np.linalg.lstsq(design, v[:, ribbon].T, rcond=None)
    rms = np.sqrt(((v[:, ribbon] - (design @ coef).T) ** 2).mean(axis=1))
    out = {
        "pial_out": _robust_unit(-smooth[:, outer].min(axis=1), floor),
        "pial_short": _robust_unit(v[:, past].mean(axis=1), floor),
        "white_deep": _robust_unit(v[:, inner].mean(axis=1), floor),
        "white_shallow": _robust_unit(-v[:, under].mean(axis=1), floor),
        "nonlinear": _robust_unit(rms, floor / 4),
    }
    out["worst"] = np.max(
        np.stack([out[k] for k in ("pial_out", "pial_short", "white_deep", "white_shallow")]),
        axis=0,
    )
    return out


def slab_contour_order(
    anatomy: np.ndarray,
    around: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    slab_mm: float = 1.0,
) -> np.ndarray:
    """Row order: back to front by coronal slab, then around each slab's ring.

    Slabs are cut in ``anatomy`` (mid-thickness points, scanner mm). Within a
    slab and group (hemisphere), vertices go by angle about the slab centroid
    in the x-z plane of ``around`` -- the sphere or inflated positions, where
    the ring is nearly convex. Angle on the folded surface itself jumps across
    every sulcus: consecutive rows were a median 4 mm apart that way, 1.1 mm on
    the sphere. Some jumps remain, because a slab cuts the folded surface into
    several separate pieces. Returns indices into ``anatomy``.
    """
    pts = np.asarray(anatomy, np.float64)
    ring = pts if around is None else np.asarray(around, np.float64)
    groups = np.zeros(pts.shape[0], np.int64) if groups is None else np.asarray(groups)
    slab = np.floor((pts[:, 1] - pts[:, 1].min()) / slab_mm).astype(np.int64)
    angle = np.zeros(pts.shape[0])
    key = slab * (int(groups.max()) + 1) + groups
    order = np.argsort(key, kind="stable")
    bounds = np.flatnonzero(np.diff(key[order])) + 1
    for idx in np.split(order, bounds):
        c = ring[idx].mean(axis=0)
        # From straight down (-z), so each ring starts at the base.
        angle[idx] = np.arctan2(ring[idx, 0] - c[0], -(ring[idx, 2] - c[2]))
    return np.lexsort((angle, groups, slab))


__all__ = [
    "SCORES",
    "equivolume_fraction",
    "sample_depths",
    "ProfileSpec",
    "Profiles",
    "TissueLevels",
    "profile_scores",
    "sample_profiles",
    "slab_contour_order",
    "tissue_levels",
]
