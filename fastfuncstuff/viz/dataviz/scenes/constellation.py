"""Constellation: the salient voxels as stars, wired to their correlation partners.

The most salient voxels become stars at their anatomical places in a slowly
orbiting head. Each is linked to the few voxels whose correlation profile is
most like its own (or most opposite) *anywhere else in the head* -- partners
closer than ``reach_mm`` are skipped, so the web is made of the long-range
links that anatomy alone hides. This is how distance gets encoded: two
voxels across the brain that move together are joined by a line.

A link lights up when both its ends rise together (or, for an anticorrelated
link, when one rises as the other falls), and pulses run along the lit links.
At rest the web is a faint lattice; the data draws its live subnetworks in
light over it.
"""

from __future__ import annotations

import torch

from ..drives import Param, splat
from .base import Scene, orbit_project, splat_lines

S = "constellation"


class ConstellationScene(Scene):
    name = S
    blurb = "salient voxels as stars, linked to their long-range correlation partners"
    PARAMS = (
        Param("stars", 1500, 300, 8000, S, "how many salient voxels become stars", scene=S),
        Param("links", 2, 1, 8, S, "correlation partners per star", scene=S),
        Param(
            "reach_mm",
            0.25,
            0.0,
            0.8,
            S,
            "skip partners nearer than this (fraction of head radius)",
            scene=S,
        ),
        Param("web", 0.15, 0.0, 1.0, S, "faint glow of every link at rest", scene=S),
        Param(
            "spark", 1.2, 0.0, 4.0, S, "co-activation (z^2) a link needs before it lights", scene=S
        ),
        Param("pulse", 0.6, 0.0, 3.0, S, "speed of the pulses along lit links", scene=S),
        Param("corbit", 0.05, -0.5, 0.5, S, "camera turn rate", scene=S),
        Param("cpitch", 0.5, -1.5, 1.5, S, "camera pitch", scene=S),
        Param("czoom", 1.7, 0.3, 4.0, S, "zoom", scene=S),
    )

    def __init__(self, drives, width, height, seed=0):
        super().__init__(drives, width, height, seed)
        self._build()

    def on_param(self, name, value):
        if name in ("stars", "links", "reach_mm"):
            self._build()

    def _build(self) -> None:
        f = self.f
        m = min(int(self.p["stars"]), f.n_voxels)
        star = torch.argsort(f.salience, descending=True)[:m]
        rows = f.loadings[star]
        rows = rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-9)
        sim = rows @ rows.T  # cosine of correlation profiles
        pos = f.anat[star]
        near = torch.cdist(pos, pos) < self.p["reach_mm"]
        sim = sim.masked_fill(near, 0.0)  # includes the diagonal
        k = int(self.p["links"])
        top = sim.abs().topk(k, dim=1)
        a = torch.arange(m, device=self.device)[:, None].expand(-1, k).reshape(-1)
        b = top.indices.reshape(-1)
        # Each pair once, whichever side chose it; sim is symmetric.
        pairs = torch.unique(torch.stack([torch.minimum(a, b), torch.maximum(a, b)], 1), dim=0)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        strength = sim[pairs[:, 0], pairs[:, 1]]
        self.star = star
        self._cold = torch.tensor([[0.55, 0.45, 1.0]], device=self.device)
        self.ea, self.eb = pairs[:, 0], pairs[:, 1]
        self.esign = torch.sign(strength)
        self.estrength = strength.abs()
        self.phase = self.rand(len(pairs))

    def step(self, dt: float) -> None:
        pass

    def render(self) -> torch.Tensor:
        p, dr, f = self.p, self.dr, self.f
        h, w = self.height, self.width
        star = self.star
        sx, sy, depth = orbit_project(
            f.anat[star], dr.wall * p["corbit"], p["cpitch"], p["czoom"], w, h
        )
        self.remember(sx, sy, star)
        fog = torch.exp(-(depth - 3.2).clamp_min(-1.5) * 0.7)
        lv = dr.live[star]
        gate = dr.gate()[star] if p["gate"] > 0 else torch.ones_like(lv)
        ea, eb = self.ea, self.eb
        # Co-activation: both ends up (or, signed, one up as the other falls).
        co = (self.esign * lv[ea] * lv[eb] - p["spark"]).clamp_min(0).sqrt()
        co = co * (gate[ea] * gate[eb]).sqrt() * p["gain"] * 0.03
        rest = p["web"] * self.estrength * 0.05
        col = dr.colors()[star]
        warm = 0.5 * (col[ea] + col[eb])
        cold = self._cold.expand_as(warm)
        ecol = torch.where((self.esign > 0)[:, None], warm, cold)
        efog = (fog[ea] * fog[eb]).sqrt()[:, None]
        frame = torch.zeros(h * w, 3, device=self.device)
        splat_lines(
            frame, sx[ea], sy[ea], sx[eb], sy[eb], ecol * ((co + rest)[:, None]) * efog, w, h, 20
        )
        # Pulses: a bright bead running along every lit link.
        # Every link carries a bead weighted by its co-activation, so unlit
        # links add nothing: fixed shapes, no mask, no sync.
        u = (self.phase + dr.wall * p["pulse"]) % 1.0
        px = sx[ea] + (sx[eb] - sx[ea]) * u
        py = sy[ea] + (sy[eb] - sy[ea]) * u
        splat(frame, px, py, ecol * (co * 3.0)[:, None] * efog, w, h)
        glow = (dr.brightness()[star] * 0.05 + 0.04) * fog
        splat(frame, sx, sy, col * glow[:, None], w, h)
        self.fade_in(frame)
        return dr.finish(self.accum, h, w)
