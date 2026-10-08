"""The 3-D surface view, minus Qt: shapes, camera, texture frames, picking.

How data reaches the surface is pycortex's idea, and it is the reason for
building this rather than colouring vertices. Every vertex carries two
positions it is *drawn* at (the shapes being morphed between: folded,
inflated, flat...) and two positions it is *sampled* at -- its white and pial
points in scanner millimetres. The rasteriser interpolates the sampling
positions across each triangle, and the fragment shader reads the volume at
``mix(white, pial, depth)`` for every pixel. So a 0.35 mm dataset shows its own
voxels on an inflated mesh built at 1 mm, and an edit to white or pial changes
what the inflated view shows the moment it lands.

Kept free of Qt so the geometry, the camera and the shading arithmetic are
testable without a GPU; :mod:`viewer.ui.surfacewindow` only uploads what this
module computes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fastfuncstuff.io.freesurfer import Hemisphere
from fastfuncstuff.surface.profiles import equivolume_fraction

#: Shapes a hemisphere can be drawn as, in the order the window offers them.
#: ``mid`` is half-way between white and pial (pycortex's "fiducial"); ``flat``
#: is the first patch whose name says flat.
SHAPES = ("mid", "white", "pial", "inflated", "sphere", "flat")
#: Shapes that are the anatomy itself, drawn where the head is. The rest are
#: laid out side by side, because inflated hemispheres share a centroid and
#: would otherwise be drawn through each other.
ANATOMICAL_SHAPES = ("mid", "white", "pial", "smoothwm")
#: Gap between laid-out hemispheres, mm.
HEMI_GAP = 8.0


def flat_patch(hemi: Hemisphere):
    """The hemisphere's flat patch, preferring a name that says flat."""
    if not hemi.patches:
        return None
    for name, patch in hemi.patches.items():
        if "flat" in name:
            return patch
    return next(iter(hemi.patches.values()))


def flat_layout(hemi: Hemisphere, patch) -> np.ndarray:
    """``(V, 2)`` patch coordinates turned to a lateral view: superior up, and
    posterior toward the other hemisphere (lh's to the right, rh's to the left), so
    side by side the two flat maps meet occipital pole to occipital pole, as
    pycortex lays them out.

    A patch file's own x/y are whatever the flattening left -- two hemispheres come
    out at unrelated angles, and either can be mirrored. The turn is the orthogonal
    map (a reflection allowed) that best carries the patch onto the anatomy's own
    anterior-posterior / superior-inferior plane, fitted over every patch vertex.
    """
    xy = patch.coords[:, :2].astype(np.float64)
    inside = patch.in_patch
    out = np.zeros_like(xy)
    if inside.sum() < 3:
        return out.astype(np.float32)
    f = xy[inside] - xy[inside].mean(axis=0)
    anat = hemi.states["white"][inside][:, 1:3].astype(np.float64)
    target = anat - anat.mean(axis=0)
    if hemi.name.startswith("lh"):
        target[:, 0] *= -1.0  # lh: posterior (-y) points right, toward rh
    u, sv, vt = np.linalg.svd(f.T @ target)
    if sv[1] <= 1e-9 * sv[0]:
        return xy.astype(np.float32)  # no 2-D anatomy to turn by (a flat test sheet)
    out[inside] = f @ (u @ vt)
    return out.astype(np.float32)


def shape_positions(hemi: Hemisphere, shape: str) -> np.ndarray | None:
    """``(V, 3)`` float32 positions of ``shape``, before layout; ``None`` if absent."""
    if shape == "mid":
        return (0.5 * (hemi.states["white"] + hemi.states["pial"])).astype(np.float32)
    if shape == "flat":
        patch = flat_patch(hemi)
        if patch is None:
            return None
        # A patch is 2-D in its own x/y; lay it in the axial plane so the
        # default (superior) camera looks straight at it.
        out = np.zeros((patch.coords.shape[0], 3), np.float32)
        out[:, :2] = flat_layout(hemi, patch)
        return out
    found = hemi.states.get(shape)
    return None if found is None else found.astype(np.float32)


def shape_faces(hemi: Hemisphere, shape: str) -> np.ndarray:
    """Faces to draw for ``shape``: the patch's own for flat, every face otherwise."""
    if shape == "flat":
        patch = flat_patch(hemi)
        if patch is not None:
            return patch.faces(hemi.faces).astype(np.uint32)
    return hemi.faces.astype(np.uint32)


def layout_offsets(
    positions: dict[str, np.ndarray], shape: str, faces: dict[str, np.ndarray] | None = None
) -> dict[str, np.ndarray]:
    """Per-hemisphere translation so laid-out shapes sit side by side.

    Anatomical shapes are left where the head is (offset zero). Others are
    centred on a common origin with left to the left and right to the right,
    ``HEMI_GAP`` apart. Only vertices that are drawn (those in ``faces``)
    count toward the extent -- a flat patch leaves the rest at zero.
    """
    zero = {h: np.zeros(3, np.float32) for h in positions}
    if shape in ANATOMICAL_SHAPES:
        return zero
    used: dict[str, np.ndarray] = {}
    for h, p in positions.items():
        if faces is not None and h in faces and faces[h].size:
            used[h] = p[np.unique(faces[h])]
        else:
            used[h] = p
    out: dict[str, np.ndarray] = {}
    for h, p in used.items():
        lo, hi = p.min(axis=0), p.max(axis=0)
        centre = 0.5 * (lo + hi)
        off = -centre
        half = 0.5 * float(hi[0] - lo[0])
        if h == "lh":
            off[0] -= half + HEMI_GAP / 2
        elif h == "rh":
            off[0] += half + HEMI_GAP / 2
        out[h] = off.astype(np.float32)
    return out


def hemisphere_models(
    positions: dict[str, np.ndarray], split: float = 0.0, hinge: float = 0.0
) -> dict[str, np.ndarray]:
    """Per-hemisphere 4x4 placing ``positions`` (as drawn, before this) in the window.

    ``split`` pushes the hemispheres apart along x, mm in total. ``hinge``
    swings them open about a vertical axis, total degrees split evenly between
    the two: positive pivots each on its front medial edge (nose to nose),
    negative on its back medial edge (occipital to occipital). Each pivots on
    its *own* edge, so the hinge is where the two hemispheres touch, the way
    a hot-dog bun opens; the split then moves them apart along x, which at
    180 degrees is straight away from each other.
    """
    out: dict[str, np.ndarray] = {}
    a = np.radians(float(hinge))
    for h, pos in positions.items():
        side = -1.0 if h == "lh" else 1.0
        m = np.eye(4)
        if a != 0.0 and pos.size:
            lo, hi = pos.min(axis=0), pos.max(axis=0)
            # The medial edge faces the other hemisphere: lh's largest x.
            px = float(hi[0] if side < 0 else lo[0])
            py = float(hi[1] if a > 0 else lo[1])
            # The end away from the hinge swings out to the hemisphere's own
            # side: lh's back end to -x for a front hinge (clockwise from
            # above), its front end to -x for a back hinge (counter-clockwise).
            r = _rotation([0, 0, 1], side * a / 2.0)
            pivot = np.array([px, py, 0.0])
            m[:3, :3] = r
            m[:3, 3] = pivot - r @ pivot
        m[0, 3] += side * split / 2.0
        out[h] = m
    return out


def texture_from_mm(shape: tuple[int, int, int], affine: np.ndarray) -> np.ndarray:
    """4x4 from scanner mm to normalised 3-D texture coordinates.

    Voxel ``i`` is centred at ``(i + 0.5) / n``. The texture's x/y/z are the
    array's i/j/k, which is why the upload transposes to k-slowest.
    """
    scale = np.diag([1.0 / shape[0], 1.0 / shape[1], 1.0 / shape[2], 1.0])
    shift = np.eye(4)
    shift[:3, 3] = 0.5
    return scale @ shift @ np.linalg.inv(np.asarray(affine, np.float64))


def texture_data(volume: np.ndarray) -> np.ndarray:
    """A ``(nx, ny, nz)`` volume as the ``(nz, ny, nx)`` float32 a 3-D texture uploads."""
    vol = np.asarray(volume, dtype=np.float32)
    if vol.ndim == 4:
        vol = vol[..., 0]
    return np.ascontiguousarray(np.nan_to_num(vol, nan=0.0).transpose(2, 1, 0))


# -- camera -------------------------------------------------------------------


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def _basis(toward_camera, up) -> np.ndarray:
    """World->camera rotation from where the camera sits and which way is up."""
    z = np.asarray(toward_camera, np.float64)
    y = np.asarray(up, np.float64)
    return np.stack([np.cross(y, z), y, z])


#: Named camera orientations, in the order ``v`` cycles them. Rows are the
#: camera's right, up and toward-camera axes in scanner RAS. Lateral views put
#: anterior toward the face, so left lateral has anterior on the left.
VIEWS: dict[str, np.ndarray] = {
    "top": _basis([0, 0, 1], [0, 1, 0]),
    "left lateral": _basis([-1, 0, 0], [0, 0, 1]),
    "left medial": _basis([1, 0, 0], [0, 0, 1]),
    "right lateral": _basis([1, 0, 0], [0, 0, 1]),
    "right medial": _basis([-1, 0, 0], [0, 0, 1]),
    "front": _basis([0, 1, 0], [0, 0, 1]),
    "back": _basis([0, -1, 0], [0, 0, 1]),
    "bottom": _basis([0, 0, -1], [0, 1, 0]),
}


@dataclass
class Camera:
    """Trackball camera: a rotation of the world, a distance, a target.

    The default looks down on the brain from above with anterior at the top
    of the screen -- both hemispheres in view, and a flat patch laid in the
    axial plane faces it directly.
    """

    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    distance: float = 350.0
    target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    fov_deg: float = 30.0

    def view(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = self.rotation
        m[:3, 3] = -self.rotation @ self.target
        m[2, 3] -= self.distance
        return m

    def projection(self, aspect: float) -> np.ndarray:
        """OpenGL-convention perspective; the renderer applies the backend's correction."""
        f = 1.0 / np.tan(np.radians(self.fov_deg) / 2)
        near, far = max(self.distance * 0.05, 1.0), self.distance * 4 + 500.0
        p = np.zeros((4, 4))
        p[0, 0] = f / max(aspect, 1e-6)
        p[1, 1] = f
        p[2, 2] = (far + near) / (near - far)
        p[2, 3] = 2 * far * near / (near - far)
        p[3, 2] = -1.0
        return p

    def orbit(self, dx: float, dy: float) -> None:
        """Rotate by a screen drag, in radians about screen y (dx) and x (dy)."""
        r = _rotation([0, 1, 0], dx) @ _rotation([1, 0, 0], dy)
        self.rotation = r @ self.rotation

    def roll(self, angle: float) -> None:
        self.rotation = _rotation([0, 0, 1], angle) @ self.rotation

    def zoom(self, factor: float) -> None:
        self.distance = float(np.clip(self.distance * factor, 20.0, 3000.0))

    def pan(self, dx: float, dy: float) -> None:
        """Move the target by a screen drag in units of the view height."""
        h = 2 * self.distance * np.tan(np.radians(self.fov_deg) / 2)
        right, up = self.rotation[0], self.rotation[1]
        self.target = self.target - (dx * h) * right + (dy * h) * up

    def frame(self, points: np.ndarray) -> None:
        """Aim at and back off to fit ``points``."""
        lo, hi = points.min(axis=0), points.max(axis=0)
        self.target = 0.5 * (lo + hi)
        radius = 0.5 * float(np.linalg.norm(hi - lo))
        self.distance = radius / np.sin(np.radians(self.fov_deg) / 2) * 1.05

    def ray(self, x_ndc: float, y_ndc: float, aspect: float) -> tuple[np.ndarray, np.ndarray]:
        """World-space origin and unit direction through a point in [-1, 1]^2."""
        inv = np.linalg.inv(self.projection(aspect) @ self.view())
        near = inv @ np.array([x_ndc, y_ndc, -1.0, 1.0])
        far = inv @ np.array([x_ndc, y_ndc, 1.0, 1.0])
        near, far = near[:3] / near[3], far[:3] / far[3]
        d = far - near
        return near, d / np.linalg.norm(d)


# -- picking -------------------------------------------------------------------


def pick(
    origin: np.ndarray, direction: np.ndarray, positions: np.ndarray, faces: np.ndarray
) -> tuple[int, np.ndarray, float] | None:
    """Nearest triangle hit by a ray: (face index, barycentric (3,), distance).

    Moller-Trumbore over every face at once; ~20-40 ms for a hemisphere, which
    a click can afford and which avoids keeping a BVH in step with edits.
    """
    v0 = positions[faces[:, 0]].astype(np.float64)
    e1 = positions[faces[:, 1]] - v0
    e2 = positions[faces[:, 2]] - v0
    p = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = origin - v0
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    v = (q @ direction) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    if not hit.any():
        return None
    idx = np.flatnonzero(hit)
    k = idx[np.argmin(t[idx])]
    return int(k), np.array([1 - u[k] - v[k], u[k], v[k]]), float(t[k])


# -- uniforms -------------------------------------------------------------------

#: Overlay layers the shader composites, bottom to top.
MAX_LAYERS = 4
#: Bytes of one ``Layer`` struct and of the whole block (std140); must match
#: surface.vert/.frag.
LAYER_BYTES = 2 * 64 + 3 * 16
UNIFORM_BYTES = 2 * 64 + 5 * 16 + MAX_LAYERS * LAYER_BYTES

#: Outline modes, from the layer's own flags so 2-D and 3-D agree: ``boxed``
#: draws the fill and its border, ``edges`` the border alone -- for an ROI or
#: atlas, the lines between regions; for a stat map, its clusters' contours.
OUTLINE_NONE, OUTLINE_BOXED, OUTLINE_ONLY = 0, 1, 2


@dataclass
class ShadeParams:
    """Everything the fragment shader needs to colour a layer as the slices do."""

    lo: float = 0.0
    hi: float = 1.0
    threshold: float = 0.0
    opacity: float = 1.0
    sign_mode: int = 0  # 0 both, 1 pos, 2 neg
    alpha_mode: int = 0  # 0 off, 1 linear, 2 quadratic
    n_panes: int = 0
    has_data: bool = False
    #: A label layer: colour by value from the palette, sampled nearest at
    #: mid-depth only.
    labels: bool = False
    outline: int = OUTLINE_NONE


@dataclass
class LayerUniforms:
    """One overlay's slot in the uniform block."""

    tex_from_mm: np.ndarray
    stat_from_mm: np.ndarray
    shade: ShadeParams
    lut_row: int = 0
    palette_row: int = 0


def _mat(m: np.ndarray) -> bytes:
    return np.asarray(m, np.float32).T.tobytes()


def _vec(*xs: float) -> bytes:
    return np.array(xs, np.float32).tobytes()


def pack_uniforms(
    mvp: np.ndarray,
    view_model: np.ndarray,
    *,
    morph: float,
    depth: tuple[float, float],
    samples: int,
    fold_contrast: float,
    layers: list[LayerUniforms],
    cross: tuple[float, float, float, float],
    cross_rgb: tuple[float, float, float],
    equivolume: bool = False,
    map_opacity: float = 1.0,
    depth_stat: str = "mean",
    cubic: bool = False,
) -> bytes:
    """The uniform block as bytes. Matrices go column-major, as GLSL reads them.

    ``layers`` beyond :data:`MAX_LAYERS` are dropped from the bottom -- the
    top of the stack is what is looked at; empty slots are kind 0 (off).
    """
    parts = [
        _mat(mvp),
        _mat(view_model),
        _vec(morph, 0.0, 0.0, 0.0),
        _vec(depth[0], depth[1], float(samples), fold_contrast),
        _vec(*cross),
        _vec(*cross_rgb, 0.0),
        _vec(
            1.0 if equivolume else 0.0,
            map_opacity,
            float(DEPTH_STATS.index(depth_stat)),
            1.0 if cubic else 0.0,
        ),
    ]
    shown = layers[-MAX_LAYERS:]
    for k in range(MAX_LAYERS):
        if k < len(shown):
            L = shown[k]
            sh = L.shade
            kind = (2.0 if sh.labels else 1.0) if sh.has_data else 0.0
            parts += [
                _mat(L.tex_from_mm),
                _mat(L.stat_from_mm),
                _vec(sh.lo, sh.hi, sh.threshold, sh.opacity),
                _vec(sh.sign_mode, sh.alpha_mode, sh.n_panes, kind),
                _vec(L.lut_row, L.palette_row, sh.outline, 0.0),
            ]
        else:
            parts.append(bytes(LAYER_BYTES))
    out = b"".join(parts)
    assert len(out) == UNIFORM_BYTES
    return out


#: Folding shades for the base, in the order ``k`` cycles them.
FOLDING = ("curv", "sulc", "binary", "off")


def folding_values(hemi: Hemisphere, mode: str) -> np.ndarray:
    """Per-vertex folding shade in [-1, 1] (positive = sulcal = darker).

    ``curv`` is the fine folding; ``sulc`` the broad sulcal depth, which
    reads best on an inflated surface; ``binary`` FreeSurfer's two-tone
    gyri/sulci. Scaled to the 98th percentile so subjects look alike.
    """
    n = hemi.n_vertices
    if mode == "off":
        return np.zeros(n, np.float32)
    key = "curv" if mode == "binary" else mode
    values = hemi.morph.get(key)
    if values is None:
        values = hemi.morph.get("curv")
    if values is None:
        return np.zeros(n, np.float32)
    if mode == "binary":
        return np.sign(values).astype(np.float32)
    top = float(np.percentile(np.abs(values), 98)) or 1.0
    return np.clip(values / top, -1.0, 1.0).astype(np.float32)


#: Per-vertex maps a surface window can paint, in the order offered.
VERTEX_MAPS = ("", "thickness", "sulc", "curv", "annot", "flags")


def over(top: np.ndarray, under: np.ndarray) -> np.ndarray:
    """``top`` RGBA (uint8) alpha-blended over ``under``: a surface layer over a map."""
    a = top[:, 3:4].astype(np.float32) / 255.0
    b = under[:, 3:4].astype(np.float32) / 255.0
    out_a = a + b * (1.0 - a)
    rgb = top[:, :3] * a + under[:, :3] * b * (1.0 - a)
    rgb = np.divide(rgb, out_a, out=np.zeros_like(rgb), where=out_a > 0)
    out = np.empty_like(top)
    out[:, :3] = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    out[:, 3] = np.clip(np.round(out_a[:, 0] * 255), 0, 255).astype(np.uint8)
    return out


def flag_ramp() -> np.ndarray:
    """256 RGB entries, quiet grey (fine) to saturated red (flagged).

    Shared by the profile column and the surface map so a flag is the same
    colour in both. Not ``hot``: on a light palette its worst end is the
    colour of the background.
    """
    t = np.linspace(0.0, 1.0, 256)[:, None]
    quiet = np.array([0.80, 0.80, 0.78])
    mid = np.array([0.96, 0.62, 0.18])
    alarm = np.array([0.86, 0.10, 0.08])
    lo = quiet + (mid - quiet) * np.clip(t / 0.6, 0, 1)
    rgb = np.where(t < 0.6, lo, mid + (alarm - mid) * np.clip((t - 0.6) / 0.4, 0, 1))
    return (rgb * 255).astype(np.uint8)


def vertex_map_scale(hemi: Hemisphere, kind: str) -> tuple[float, float, str] | None:
    """(value at the LUT's start, value at its end, LUT name) of a continuous map.

    ``None`` for maps that are not a scale (parcellation, flags) or absent.
    Thickness on a fixed 1-4.5 mm viridis scale, so two subjects read the
    same; sulc and curv red-blue, symmetric at their 98th percentile, with
    FreeSurfer's sign (positive is sulcal, deep) toward blue.
    """
    if kind not in ("thickness", "sulc", "curv"):
        return None
    values = hemi.morph.get(kind)
    if values is None:
        return None
    if kind == "thickness":
        return 1.0, 4.5, "viridis"
    cortex = hemi.cortex if hemi.cortex is not None else np.ones(hemi.n_vertices, bool)
    top = float(np.percentile(np.abs(values[cortex]), 98)) or 1.0
    return top, -top, "RdBu"


def map_lut(name: str) -> np.ndarray:
    """``(256, 4)`` uint8 entries of a named colormap."""
    import torch

    from fastfuncstuff.viewer.colormap import build_lut

    return (np.clip(build_lut(name, 256, device=torch.device("cpu")).numpy(), 0, 1) * 255).astype(
        np.uint8
    )


def vertex_colors(
    hemi: Hemisphere, kind: str, annotation=None, flags: np.ndarray | None = None
) -> np.ndarray | None:
    """``(V, 4)`` uint8 colours of a per-vertex map, or ``None`` for no map.

    Scales as :func:`vertex_map_scale`; ``annot`` in the parcellation's own
    colours. The medial wall is left transparent: it has no thickness and no
    region.
    """
    if not kind:
        return None
    n = hemi.n_vertices
    cortex = hemi.cortex if hemi.cortex is not None else np.ones(n, bool)
    if kind == "flags":
        # Profile-column flags: transparent where fine, so the anatomy shows
        # through and only the trouble is coloured.
        if flags is None:
            return None
        f = np.nan_to_num(flags, nan=0.0)
        out = np.zeros((n, 4), np.uint8)
        out[:, :3] = flag_ramp()[np.round(np.clip(f, 0, 1) * 255).astype(int)]
        out[:, 3] = (np.clip((f - 0.3) / 0.5, 0, 1) * 255).astype(np.uint8)
        out[~cortex, 3] = 0
        return out
    if kind == "annot":
        if annotation is None:
            return None
        out = annotation.vertex_rgba()
        out[~cortex, 3] = 0
        return out
    values = hemi.morph.get(kind)
    scale = vertex_map_scale(hemi, kind)
    if values is None or scale is None:
        return None
    lo, hi, name = scale
    lut = map_lut(name)
    unit = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    out = np.zeros((n, 4), np.uint8)
    out[:, :3] = lut[np.round(unit * 255).astype(int)][:, :3]
    out[:, 3] = np.where(cortex, 255, 0)
    return out


#: Palette entries per texture row; must match PALETTE_W in surface.frag.
PALETTE_W = 4096


def palette_texture(rgb: np.ndarray) -> np.ndarray:
    """``(N, 3)`` label colours in [0, 1] as an ``(rows, PALETTE_W, 4)`` uint8 image."""
    n = rgb.shape[0]
    rows = max(1, -(-n // PALETTE_W))
    out = np.zeros((rows * PALETTE_W, 4), np.uint8)
    out[:n, :3] = np.clip(np.round(np.asarray(rgb) * 255), 0, 255)
    out[:n, 3] = 255
    return out.reshape(rows, PALETTE_W, 4)


def depth_fractions(depth: tuple[float, float], samples: int) -> np.ndarray:
    """Where the shader samples between white (0) and pial (1)."""
    if samples <= 1:
        return np.array([depth[0]])
    return np.linspace(depth[0], depth[1], samples)


#: How the depth samples between white and pial become one value, in the
#: order ``d`` cycles them; the index is what the shader receives. Names are
#: 3dVol2Surf's map functions. Selections (median, max, min, max_abs) take
#: the threshold statistic from the *same* depth as the value, so a colour
#: and its threshold always describe one point; means average both over the
#: same samples. No ``mode``: of continuous samples every value is unique,
#: and label layers already vote across depth.
DEPTH_STATS = ("mean", "median", "max", "min", "max_abs", "nzmean")


def reduce_depth(values: np.ndarray, stats: np.ndarray, how: str) -> tuple[np.ndarray, np.ndarray]:
    """Reduce ``(..., S)`` depth samples to one value and threshold statistic.

    CPU twin of the fragment shader's ``reduceDepth``. For an even sample
    count ``median`` is the upper middle, as in the shader -- a sample that
    exists, so its statistic is that sample's too.
    """
    v = np.asarray(values, np.float64)
    s = np.asarray(stats, np.float64)
    if how == "mean":
        return v.mean(axis=-1), s.mean(axis=-1)
    if how == "nzmean":
        nz = v != 0
        n = nz.sum(axis=-1)
        safe = np.maximum(n, 1)
        return (
            np.where(n > 0, (v * nz).sum(axis=-1) / safe, 0.0),
            np.where(n > 0, (s * nz).sum(axis=-1) / safe, 0.0),
        )
    if how == "median":
        order = np.argsort(v, axis=-1, kind="stable")
        pick = np.take(order, [v.shape[-1] // 2], axis=-1)
    elif how == "max":
        pick = np.argmax(v, axis=-1)[..., None]
    elif how == "min":
        pick = np.argmin(v, axis=-1)[..., None]
    elif how == "max_abs":
        pick = np.argmax(np.abs(v), axis=-1)[..., None]
    else:
        raise ValueError(f"unknown depth statistic {how!r}; one of {', '.join(DEPTH_STATS)}")
    return (
        np.take_along_axis(v, pick, axis=-1)[..., 0],
        np.take_along_axis(s, pick, axis=-1)[..., 0],
    )


def shade_reference(values: np.ndarray, stat: np.ndarray, shade: ShadeParams, lut: np.ndarray):
    """CPU twin of the fragment shader's colouring, through :mod:`viewer.colormap`.

    ``values``/``stat`` are already depth-averaged. Returns ``(rgb, alpha)``.
    The shader's arithmetic is checked against this, so 3-D and 2-D agree on
    every colour because both are this.
    """
    import torch

    from fastfuncstuff.viewer.colormap import apply_colormap, threshold_alpha
    from fastfuncstuff.viewer.layers import AlphaMode, SignMode

    signs = [SignMode.BOTH, SignMode.POS, SignMode.NEG]
    alphas = [AlphaMode.OFF, AlphaMode.LINEAR, AlphaMode.QUADRATIC]
    v = torch.as_tensor(np.asarray(values, np.float32))
    s = torch.as_tensor(np.asarray(stat, np.float32))
    rgb = apply_colormap(
        v,
        lut=torch.as_tensor(lut[:, :3]),
        lo=shade.lo,
        hi=shade.hi,
        sign_mode=signs[shade.sign_mode],
        n_panes=shade.n_panes,
    )
    alpha = threshold_alpha(
        s, shade.threshold, mode=alphas[shade.alpha_mode], sign_mode=signs[shade.sign_mode]
    )
    return rgb.numpy(), (alpha * shade.opacity).numpy()


__all__ = [
    "ANATOMICAL_SHAPES",
    "SHAPES",
    "FOLDING",
    "MAX_LAYERS",
    "VIEWS",
    "Camera",
    "LayerUniforms",
    "folding_values",
    "ShadeParams",
    "PALETTE_W",
    "VERTEX_MAPS",
    "palette_texture",
    "depth_fractions",
    "DEPTH_STATS",
    "reduce_depth",
    "equivolume_fraction",
    "vertex_colors",
    "vertex_map_scale",
    "map_lut",
    "flag_ramp",
    "flat_patch",
    "layout_offsets",
    "hemisphere_models",
    "pack_uniforms",
    "pick",
    "shade_reference",
    "shape_faces",
    "shape_positions",
    "texture_data",
    "texture_from_mm",
]
