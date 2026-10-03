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
        xy = patch.coords[:, :2].astype(np.float32)
        out = np.zeros((xy.shape[0], 3), np.float32)
        out[:, :2] = xy
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

#: std140 block shared by both shaders; field order must match surface.vert/.frag.
UNIFORM_BYTES = 4 * 64 + 7 * 16


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


def pack_uniforms(
    mvp: np.ndarray,
    view_model: np.ndarray,
    tex_from_mm: np.ndarray,
    stat_from_mm: np.ndarray,
    *,
    morph: float,
    depth: tuple[float, float],
    samples: int,
    curv_contrast: float,
    shade: ShadeParams,
    cross: tuple[float, float, float, float],
    cross_rgb: tuple[float, float, float],
    equivolume: bool = False,
    map_opacity: float = 1.0,
) -> bytes:
    """The uniform block as bytes. Matrices go column-major, as GLSL reads them."""

    def mat(m: np.ndarray) -> bytes:
        return np.asarray(m, np.float32).T.tobytes()

    def vec(*xs: float) -> bytes:
        return np.array(xs, np.float32).tobytes()

    out = b"".join(
        [
            mat(mvp),
            mat(view_model),
            mat(tex_from_mm),
            mat(stat_from_mm),
            vec(morph, 0.0, 0.0, 0.0),
            vec(depth[0], depth[1], float(samples), curv_contrast),
            vec(shade.lo, shade.hi, shade.threshold, shade.opacity),
            vec(
                shade.sign_mode,
                shade.alpha_mode,
                shade.n_panes,
                (2.0 if shade.labels else 1.0) if shade.has_data else 0.0,
            ),
            vec(*cross),
            vec(*cross_rgb, 0.0),
            vec(1.0 if equivolume else 0.0, map_opacity, 0.0, 0.0),
        ]
    )
    assert len(out) == UNIFORM_BYTES
    return out


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


#: Per-vertex maps a surface window can paint, in the order offered.
VERTEX_MAPS = ("", "thickness", "sulc", "curv", "annot", "flags")


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


def vertex_colors(
    hemi: Hemisphere, kind: str, annotation=None, flags: np.ndarray | None = None
) -> np.ndarray | None:
    """``(V, 4)`` uint8 colours of a per-vertex map, or ``None`` for no map.

    Thickness on a fixed 1-4.5 mm viridis scale, so two subjects read the
    same; sulc and curv on a symmetric red-blue scale at their 98th
    percentile; ``annot`` in the parcellation's own colours. The medial wall
    is left transparent: it has no thickness and no region.
    """
    import torch

    from fastfuncstuff.viewer.colormap import build_lut

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
    if values is None:
        return None
    if kind == "thickness":
        lo, hi, name = 1.0, 4.5, "viridis"
    else:
        top = float(np.percentile(np.abs(values[cortex]), 98)) or 1.0
        # FreeSurfer's sign: positive sulc/curv is sulcal (deep); show it blue.
        lo, hi, name = top, -top, "RdBu"
    lut = (np.clip(build_lut(name, 256, device=torch.device("cpu")).numpy(), 0, 1) * 255).astype(
        np.uint8
    )
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
    "VIEWS",
    "Camera",
    "ShadeParams",
    "PALETTE_W",
    "VERTEX_MAPS",
    "palette_texture",
    "depth_fractions",
    "equivolume_fraction",
    "vertex_colors",
    "flag_ramp",
    "flat_patch",
    "layout_offsets",
    "pack_uniforms",
    "pick",
    "shade_reference",
    "shape_faces",
    "shape_positions",
    "texture_data",
    "texture_from_mm",
]
