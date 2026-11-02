"""Milkdrop: a feedback buffer warped every frame, the Winamp-visualiser way.

Every frame the previous image is resampled through a warp -- zoom toward the
viewer, rotation that varies with radius, a wobble -- then faded and recoloured
a little, and this frame's points are drawn on top. Points therefore leave
streaks that spiral outward into tunnels.

The warp is the data's: the zoom pumps with the global signal, the rotation is
a sum of radial harmonics each driven by one family's time course (so every
network twists its own band of the tunnel), DVARS spikes jolt the warp, and
the colour drifts faster when the families are busy. The injected points are
the voxels laid out in correlation space (or as an axial projection of the
head), lit by their live values.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..drives import Param, splat, uniform_disc
from .base import Scene

S = "milkdrop"
INJECT = ("scope", "function", "axial", "ring")


class MilkdropScene(Scene):
    name = S
    blurb = "feedback tunnel: zoom, swirl and colour drift from the data"
    PARAMS = (
        Param(
            "decay", 0.94, 0.8, 0.999, S, "fraction of the warped image kept each frame", scene=S
        ),
        Param("pump", 1.0, 0.0, 4.0, S, "tunnel zoom with the global signal", scene=S),
        Param("drift", 0.3, -1.0, 1.0, S, "steady outward zoom (negative: inward)", scene=S),
        Param("swirl", 1.0, 0.0, 4.0, S, "rotation per unit of family activity", scene=S),
        Param("wobble", 1.0, 0.0, 4.0, S, "warp jolt on DVARS spikes", scene=S),
        Param("hue", 0.3, 0.0, 2.0, S, "colour drift of the feedback", scene=S),
        Param("inject", 0, 0, len(INJECT) - 1, S, "layout of the injected points", INJECT, scene=S),
        Param("isize", 0.45, 0.1, 1.5, S, "size of the injected figure", scene=S),
        Param("ink_gain", 0.15, 0.0, 1.0, S, "brightness of the injected points", scene=S),
        Param("scope_amp", 0.25, 0.0, 1.0, S, "scope: radial swing per unit z", scene=S),
    )

    def __init__(self, drives, width, height, seed=0):
        super().__init__(drives, width, height, seed)
        self.buf = torch.zeros(1, 3, height, width, device=self.device)
        self._grid_base()
        f = self.f
        self.pts = {
            "function": uniform_disc(f.func[:, :2]),
            "axial": f.anat[:, :2] * torch.tensor([1.0, -1.0], device=self.device),
        }
        order = torch.argsort(torch.argsort(f.family_center[:, 2]))
        az = torch.atan2(f.anat[:, 1], f.anat[:, 0])
        r = 0.35 + 0.6 * (order.float()[f.family] + 0.5) / drives.n_fam
        self.pts["ring"] = torch.stack([r * torch.cos(az), r * torch.sin(az)], 1)
        # The oscilloscope: the most salient voxels around one circle by
        # anatomical azimuth. Its radius is pushed per voxel by the live value
        # at render time, so the trace is the brain's waveform.
        keep = f.salience >= torch.quantile(f.salience, 0.75)
        self.scope_idx = torch.nonzero(keep).squeeze(1)
        self.scope_cos = torch.cos(az[keep])
        self.scope_sin = torch.sin(az[keep])
        self.hue_angle = 0.0

    def on_resize(self):
        self.buf = torch.zeros(1, 3, self.height, self.width, device=self.device)
        self._grid_base()

    def _grid_base(self) -> None:
        h, w = self.height, self.width
        ys = torch.linspace(-1, 1, h, device=self.device)
        xs = torch.linspace(-1, 1, w, device=self.device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        self.aspect = w / h
        self.px = gx * self.aspect  # isotropic coordinates
        self.py = gy
        self.pr = (self.px**2 + self.py**2).sqrt()

    def step(self, dt: float) -> None:
        p, dr = self.p, self.dr
        r = self.pr
        # Zoom: < 1 samples from closer to the centre, so content flows outward.
        zoom = 1.0 - dt * (0.6 * p["drift"] + 0.8 * p["pump"] * math.tanh(dr.g * 0.6))
        # Rotation: one radial harmonic per family, weighted by its activity.
        k = torch.arange(1, dr.n_fam + 1, device=self.device, dtype=torch.float32)
        amp = dr.fam.clamp(-3, 3) * dr.family_salience / dr.family_salience.max().clamp_min(1e-6)
        rot = (amp[:, None, None] * torch.sin(k[:, None, None] * math.pi * r[None] * 0.7)).sum(0)
        rot = rot * p["swirl"] * dt * 0.35
        jolt = max(0.0, dr.d - 2.5) * p["wobble"] * 0.004
        c, s = torch.cos(rot), torch.sin(rot)
        sx = (self.px * c - self.py * s) * zoom
        sy = (self.px * s + self.py * c) * zoom
        if jolt > 0:
            sx = sx + jolt * torch.sin(self.py * 9 + dr.wall * 13)
            sy = sy + jolt * torch.sin(self.px * 9 - dr.wall * 11)
        grid = torch.stack([sx / self.aspect, sy], -1)[None]
        warped = F.grid_sample(
            self.buf, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        # Hue drift: rotate the kept colour about the grey axis, faster when the
        # families are busy.
        busy = dr.fam.abs().mean()  # stays on the device
        self.hue_angle = p["hue"] * dt * (0.3 + busy)
        warped = _hue_rotate(warped, self.hue_angle)
        self.buf = warped * (p["decay"] ** (dt * 60))

    def render(self) -> torch.Tensor:
        p, dr = self.p, self.dr
        h, w = self.height, self.width
        scale = 0.5 * h * p["isize"]
        frame = torch.zeros(h * w, 3, device=self.device)
        mode = INJECT[int(p["inject"])]
        if mode == "scope":
            idx = self.scope_idx
            lv = dr.live[idx]
            r = (1.0 + p["scope_amp"] * torch.tanh(lv * 0.7)) * scale
            # Every point on the trace glows a little, the rising ones more, so
            # the line is always there and the data decides where it flares.
            glow = (0.15 + dr.brightness()[idx]) * self.density * p["ink_gain"] * 2.0
            rgb = dr.colors()[idx] * glow[:, None]
            sx, sy = w * 0.5 + r * self.scope_cos, h * 0.5 + r * self.scope_sin
            self.remember(sx, sy, idx)
            splat(frame, sx, sy, rgb, w, h)
        else:
            pts = self.pts[mode]
            glow = dr.brightness() * self.density * p["ink_gain"]
            rgb = dr.colors() * glow[:, None]
            sx, sy = w * 0.5 + pts[:, 0] * scale, h * 0.5 + pts[:, 1] * scale
            self.remember(sx, sy)
            splat(frame, sx, sy, rgb, w, h)
        self.buf = self.buf + frame.T.reshape(1, 3, h, w)
        return dr.finish(self.buf[0].reshape(3, -1).T, h, w)


def _hue_rotate(img: torch.Tensor, angle) -> torch.Tensor:
    """Rotate colours about the grey axis by ``angle`` radians (Rodrigues).

    ``angle`` may be a 0-d device tensor: the matrix is assembled on the
    device, so the frame does not wait to read it back.
    """
    angle = torch.as_tensor(angle, device=img.device, dtype=img.dtype)
    c, s = torch.cos(angle), torch.sin(angle)
    k = 1 / math.sqrt(3)
    a = c + (1 - c) / 3
    b = (1 - c) / 3 - k * s
    d = (1 - c) / 3 + k * s
    m = torch.stack([torch.stack([a, b, d]), torch.stack([d, a, b]), torch.stack([b, d, a])])
    return torch.einsum("ij,bjhw->bihw", m, img).clamp_min(0)
