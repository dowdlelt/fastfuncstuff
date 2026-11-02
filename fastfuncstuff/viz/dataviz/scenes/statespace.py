"""State space: the whole brain's path through its top components, drawn as light.

At each moment the run is a point in the space of its leading principal
components; over time that point traces a path. This scene draws it as a
glowing ribbon in 3-D (PC1-3 as position, PC4-6 as colour), with a comet at
the present. Behind it is the cloud of voxels placed by their loadings in the
same space -- a living biplot -- lit by their live values. Since each voxel's
value is the dot product of its loading with the state, the voxels that light
up are the ones in the direction the state is heading.

Recurrence arcs join the present to past moments when the brain was in nearly
the same state (more than 15 TRs ago). A block design draws as a loop the
state keeps returning to, with arcs fanning back to every earlier lap; a slow
drift draws as a path that never closes.
"""

from __future__ import annotations

import math

import torch

from ..drives import Param, interp_rows, splat
from .base import Scene, orbit_project, splat_lines

S = "statespace"


class StatespaceScene(Scene):
    name = S
    blurb = "the brain's path through PC space as a ribbon, through the cloud of its voxels"
    PARAMS = (
        Param("tail", 40.0, 5.0, 300.0, S, "seconds of path drawn", scene=S),
        Param(
            "recur",
            0.25,
            0.0,
            1.0,
            S,
            "recurrence distance (fraction of the state's spread)",
            scene=S,
        ),
        Param("cloud", 0.4, 0.0, 2.0, S, "brightness of the voxel cloud", scene=S),
        Param("sorbit", 0.08, -0.5, 0.5, S, "camera turn rate", scene=S),
        Param("szoom", 1.4, 0.3, 4.0, S, "zoom", scene=S),
    )

    def __init__(self, drives, width, height, seed=0):
        super().__init__(drives, width, height, seed)
        f = self.f
        k = f.tcs.shape[1]
        tcs = f.tcs.float()
        # Smooth over ~2 TRs so the ribbon is a path, not a scribble.
        ker = torch.tensor([0.25, 0.5, 0.25], device=self.device)
        pad = torch.cat([tcs[:1], tcs, tcs[-1:]])
        tcs = ker[0] * pad[:-2] + ker[1] * pad[1:-1] + ker[2] * pad[2:]
        state = tcs[:, : min(3, k)]
        self.scale = float(state.norm(dim=1).quantile(0.95).clamp_min(1e-6))
        self.state = torch.zeros(f.n_time, 3, device=self.device)
        self.state[:, : state.shape[1]] = state / self.scale
        extra = tcs[:, 3:6] if k >= 6 else torch.zeros(f.n_time, 3, device=self.device)
        self.hue = 0.5 + 0.5 * torch.tanh(extra * 0.7)  # PC4-6 as colour
        # The path sampled every quarter TR, once, so drawing it is a slice.
        self.fine_state = _catmull_rom(self.state, 4)
        self.fine_hue = _catmull_rom(self.hue, 4)
        load = f.loadings[:, : min(3, k)]
        self.cloud_pos = torch.zeros(f.n_voxels, 3, device=self.device)
        self.cloud_pos[:, : load.shape[1]] = load / load.norm(dim=1).quantile(0.98).clamp_min(1e-6)

    def step(self, dt: float) -> None:
        pass

    def _proj(self, pts: torch.Tensor):
        p = self.p
        # PC space has no anatomical up: the camera just turns slowly.
        return orbit_project(
            pts, self.dr.wall * p["sorbit"], 0.35, p["szoom"], self.width, self.height
        )

    def render(self) -> torch.Tensor:
        p, dr, f = self.p, self.dr, self.f
        h, w = self.height, self.width
        frame = torch.zeros(h * w, 3, device=self.device)
        # The voxel cloud: dim by salience, lit by the live value.
        cx, cy, cd = self._proj(self.cloud_pos)
        fog = torch.exp(-(cd - 3.2).clamp_min(-1.5) * 0.6)
        glow = (dr.brightness() * 0.6 + 0.08 * f.salience) * self.density * p["cloud"] * fog
        splat(frame, cx, cy, dr.colors() * glow[:, None], w, h)
        self.remember(cx, cy, weight=dr.brightness())
        # The path: the last `tail` seconds, sampled every 0.25 TR.
        t = dr.t
        span = p["tail"] / max(f.tr, 1e-6)
        a, b = int(max(0.0, t - span) * 4), int(t * 4)
        if b - a > 2:
            pts, col = self.fine_state[a:b], self.fine_hue[a:b]
            ts = torch.arange(a, b, device=self.device, dtype=torch.float32) / 4
            age = ((t - ts) / span).clamp(0, 1)
            sx, sy, _ = self._proj(pts)
            bright = ((1 - age) ** 1.5 * 0.6)[:-1, None]
            splat_lines(frame, sx[:-1], sy[:-1], sx[1:], sy[1:], col[:-1] * bright, w, h, 8)
        now = interp_rows(self.state, t)
        nx, ny, _ = self._proj(now[None])
        # Recurrence: earlier moments (over 15 TRs back) in nearly the same state.
        past_t = torch.arange(0, max(0, int(t) - 15), device=self.device)
        if past_t.numel():
            d = (self.state[past_t] - now).norm(dim=1)
            near = d < p["recur"]
            if bool(near.any()):
                pp = self.state[past_t[near]]
                px, py, _ = self._proj(pp)
                strength = (1 - d[near] / p["recur"]).clamp(0, 1)[:, None] * 0.35
                arc = torch.tensor([0.75, 0.8, 1.0], device=self.device) * strength
                splat_lines(frame, nx.expand_as(px), ny.expand_as(py), px, py, arc, w, h, 24)
        # The comet head.
        head = torch.tensor([[1.0, 0.95, 0.85]], device=self.device) * (1.5 + dr.sync)
        for r in (0.0, 1.0):
            for a in range(0, 360, 90 if r else 360):
                ox, oy = r * math.cos(math.radians(a)), r * math.sin(math.radians(a))
                splat(frame, nx + ox, ny + oy, head * (0.5 if r else 1.0), w, h)
        self.fade_in(frame)
        return dr.finish(self.accum, h, w)


def _catmull_rom(x: torch.Tensor, per: int) -> torch.Tensor:
    """(T, D) rows resampled ``per`` times per row on a Catmull-Rom spline.

    A straight-line resample turned the path into a polygon with a corner at
    every TR.
    """
    n = x.shape[0]
    t = torch.arange(0, n - 1, 1.0 / per, device=x.device)
    i = t.floor().long()
    u = (t - i)[:, None]
    p0, p1 = x[(i - 1).clamp(0, n - 1)], x[i]
    p2, p3 = x[(i + 1).clamp(max=n - 1)], x[(i + 2).clamp(max=n - 1)]
    u2, u3 = u * u, u * u * u
    return 0.5 * (
        2 * p1
        + (-p0 + p2) * u
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * u3
    )
