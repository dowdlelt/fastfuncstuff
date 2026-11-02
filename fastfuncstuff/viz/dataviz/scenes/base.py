"""What every scene is: a way of drawing the shared drives into pixels."""

from __future__ import annotations

import math

import torch

from ..drives import Drives, Param


class Scene:
    name = ""
    blurb = ""
    PARAMS: tuple[Param, ...] = ()

    def __init__(self, drives: Drives, width: int, height: int, seed: int = 0):
        self.dr = drives
        self.f = drives.f
        self.p = drives.p
        self.device = drives.device
        self.width, self.height = width, height
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.accum = torch.zeros(height * width, 3, device=self.device)
        # Where each voxel was last drawn, in scene pixels, for picking:
        # (sx, sy, voxel index or None for 0..N-1). None when not pickable.
        self._pick: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None = None
        self._pick_weight: torch.Tensor | None = None

    def remember(
        self,
        sx: torch.Tensor,
        sy: torch.Tensor,
        vox: torch.Tensor | None = None,
        weight: torch.Tensor | None = None,
    ) -> None:
        """Record where voxels were drawn. ``weight`` breaks ties in a click:
        projections stack many voxels in one cell, and the brightest is the
        one the eye was looking at."""
        self._pick = (sx, sy, vox)
        self._pick_weight = weight

    def pick(self, x: float, y: float, radius: float = 6.0) -> int | None:
        """Voxel drawn nearest to scene pixel (x, y), within ``radius``."""
        if self._pick is None:
            return None
        sx, sy, vox = self._pick
        d2 = (sx - x) ** 2 + (sy - y) ** 2
        i = int(torch.argmin(d2))
        if float(d2[i]) > radius * radius:
            return None
        if self._pick_weight is not None:
            near = d2 <= max(float(d2[i]), 1.0) * 2.0
            score = torch.where(near, self._pick_weight, torch.tensor(-1.0, device=d2.device))
            i = int(torch.argmax(score))
        return int(vox[i]) if vox is not None else i

    def locate(self, v: int) -> tuple[float, float] | None:
        """Scene pixel where voxel v was last drawn, if it was."""
        if self._pick is None:
            return None
        sx, sy, vox = self._pick
        if vox is None:
            return float(sx[v]), float(sy[v])
        hit = torch.nonzero(vox == v)
        if hit.numel() == 0:
            return None
        i = int(hit[0, 0])
        return float(sx[i]), float(sy[i])

    def rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=self.gen).to(self.device)

    def randn(self, *shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=self.gen).to(self.device)

    def relative(self, x: torch.Tensor, key: str = "act") -> torch.Tensor:
        """``x`` over its own recent peak (98th percentile, ~5 s memory), in [0, 1].

        Absolute activity is small almost everywhere (most of the brain sits
        below threshold), so thresholds on it never trip; relative to what this
        run has recently reached, the active places read as active.
        """
        flat = x.reshape(-1)
        sample = flat if flat.numel() <= 200_000 else flat[:: flat.numel() // 200_000 + 1]
        peak = float(torch.quantile(sample, 0.98).clamp_min(1e-6))
        ref = self.__dict__.setdefault("_peaks", {}).get(key, peak)
        ref += (max(peak, ref * 0.5) - ref) * 0.01
        self._peaks[key] = ref
        return (x / ref).clamp(0, 1)

    def resize(self, width: int, height: int) -> None:
        if (width, height) != (self.width, self.height):
            self.width, self.height = int(width), int(height)
            self.accum = torch.zeros(self.height * self.width, 3, device=self.device)
            self.on_resize()

    def on_resize(self) -> None:
        pass

    def on_param(self, name: str, value: float) -> None:
        pass

    def step(self, dt: float) -> None:
        raise NotImplementedError

    def render(self) -> torch.Tensor:
        """(H, W, 3) uint8 on the scene's device."""
        raise NotImplementedError

    @property
    def density(self) -> float:
        """Per-point brightness that exposes alike whatever the voxel count."""
        return (self.width * self.height * 0.08) / max(1, self.f.n_voxels)

    def fade_in(self, frame: torch.Tensor) -> None:
        """Mix a new frame into the trails buffer."""
        tr = self.p["trails"]
        self.accum.mul_(tr).add_(frame * (1 - tr) * 4.0)


# ----------------------------------------------------------------- helpers


def orbit_project(
    pos: torch.Tensor, yaw: float, pitch: float, zoom: float, width: int, height: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perspective projection of RAS points seen from behind the head, z up.

    Returns screen x, screen y and depth (distance from the camera).
    """
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    rot_yaw = torch.tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], device=pos.device)
    rot_pitch = torch.tensor([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], device=pos.device)
    q = pos @ (rot_pitch @ rot_yaw).T
    depth = q[:, 1] + 3.2
    scale = zoom * height * 1.05 / depth
    return width * 0.5 + q[:, 0] * scale, height * 0.5 - q[:, 2] * scale, depth


def axial_cells(
    anat: torch.Tensor, gh: int, gw: int, spacing: float, fill: float = 0.92
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Each voxel's cell on a (gh, gw) grid seeing the head from above.

    Anterior up, right on the right, aspect kept. Returns (flat cell index,
    (gh, gw) bool silhouette of the head, blur passes that close the gaps).

    A grid finer than the voxels leaves whole empty rows and columns between
    voxel centres -- 1.8 cells per voxel leaves gaps of one and two cells -- so
    the silhouette is dilated, and fields blurred, by the voxel footprint.
    """
    xy = anat[:, :2]
    lo = xy.amin(0)
    hi = xy.amax(0)
    span = (hi - lo).clamp_min(1e-6)
    s = fill * min(gw / float(span[0]), gh / float(span[1]))
    cx = (gw - float(span[0]) * s) / 2
    cy = (gh - float(span[1]) * s) / 2
    col = ((xy[:, 0] - lo[0]) * s + cx).long().clamp(0, gw - 1)
    row = (gh - 1 - ((xy[:, 1] - lo[1]) * s + cy)).long().clamp(0, gh - 1)
    cell = row * gw + col
    passes = max(1, math.ceil(spacing * s))
    hit = torch.zeros(gh * gw, device=anat.device)
    hit[cell] = 1.0
    # Morphological closing by the voxel footprint: dilating closes the gaps
    # between voxel centres, eroding back keeps the edge where it was. (A blur
    # and threshold dilutes isolated hits to nothing once there are many cells
    # per voxel.)
    import torch.nn.functional as F

    r = max(1, math.ceil(spacing * s / 2))
    img = hit.reshape(1, 1, gh, gw)
    grown = F.max_pool2d(F.pad(img, (r, r, r, r)), 2 * r + 1, stride=1)
    shrunk = 1 - F.max_pool2d(F.pad(1 - grown, (r, r, r, r), value=1.0), 2 * r + 1, stride=1)
    return cell, shrunk[0, 0] > 0.5, passes


def field(
    values: torch.Tensor, cell: torch.Tensor, n_cells: int, how: str = "mean"
) -> torch.Tensor:
    """Per-cell mean (or max) of per-voxel values, (n_cells,) or (n_cells, C)."""
    if how == "max":
        out = torch.zeros(n_cells, *values.shape[1:], device=values.device)
        idx = cell if values.ndim == 1 else cell[:, None].expand_as(values)
        return out.scatter_reduce_(0, idx, values, reduce="amax", include_self=True)
    tot = torch.zeros(n_cells, *values.shape[1:], device=values.device).index_add_(0, cell, values)
    cnt = torch.zeros(n_cells, device=values.device).index_add_(
        0, cell, torch.ones_like(cell, dtype=torch.float32)
    )
    return tot / cnt.clamp_min(1).reshape(-1, *([1] * (values.ndim - 1)))


def filled_field(
    values: torch.Tensor, cell: torch.Tensor, gh: int, gw: int, passes: int = 2
) -> torch.Tensor:
    """Per-cell mean of per-voxel values with the gaps between voxel columns filled.

    A grid finer than the voxels leaves empty cells in a regular lattice; a
    plain mean draws that lattice as dark lines (and walls growth in). Blur the
    sum and the count separately and divide. Returns (C, gh, gw) or (gh, gw).
    """
    n = gh * gw
    vals = values if values.ndim == 2 else values[:, None]
    tot = torch.zeros(n, vals.shape[1], device=vals.device).index_add_(0, cell, vals)
    cnt = torch.zeros(n, device=vals.device).index_add_(
        0, cell, torch.ones_like(cell, dtype=torch.float32)
    )
    tot = blur(tot.T.reshape(-1, gh, gw), passes)
    cnt = blur(cnt.reshape(gh, gw), passes)
    out = tot / cnt.clamp_min(1e-3)
    return out if values.ndim == 2 else out[0]


def disks(spots: torch.Tensor, r: int) -> torch.Tensor:
    """Grow each True cell of an (H, W) mask into a disk of radius r (round, not square)."""
    import torch.nn.functional as F

    y = torch.arange(-r, r + 1, device=spots.device, dtype=torch.float32)
    k = ((y[:, None] ** 2 + y[None, :] ** 2) <= r * r + 0.5).float()[None, None]
    return F.conv2d(F.pad(spots.float()[None, None], (r, r, r, r)), k)[0, 0] > 0


def blur(img: torch.Tensor, passes: int = 1) -> torch.Tensor:
    """Cheap separable 3-tap box blur of an (H, W) or (C, H, W) image."""
    import torch.nn.functional as F

    x = img[None] if img.ndim == 3 else img[None, None]
    c = x.shape[1]
    k = torch.ones(c, 1, 3, 3, device=img.device) / 9.0
    for _ in range(passes):
        x = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), k, groups=c)
    return x[0] if img.ndim == 3 else x[0, 0]


def splat_lines(
    frame: torch.Tensor,
    x0: torch.Tensor,
    y0: torch.Tensor,
    x1: torch.Tensor,
    y1: torch.Tensor,
    rgb: torch.Tensor,
    w: int,
    h: int,
    samples: int = 16,
) -> None:
    """Additive anti-aliased segments: ``samples`` bilinear dots per segment.

    Brightness per segment is spread over its samples and scaled by its length,
    so a long line is as bright per pixel as a short one.
    """
    from ..drives import splat

    t = torch.linspace(0, 1, samples, device=frame.device)
    xs = x0[:, None] + (x1 - x0)[:, None] * t
    ys = y0[:, None] + (y1 - y0)[:, None] * t
    length = ((x1 - x0) ** 2 + (y1 - y0) ** 2).sqrt().clamp_min(1.0)
    per = rgb * (length / samples).clamp(max=1.0)[:, None]
    splat(frame, xs.reshape(-1), ys.reshape(-1), per.repeat_interleave(samples, 0), w, h)
