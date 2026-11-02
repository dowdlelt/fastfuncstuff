"""Turing: Gray-Scott reaction-diffusion growing inside the head, fed by the data.

Two chemicals, U and V, react and diffuse on a grid that sees the head from
above. Where the feed rate is low the pattern is sparse worms and mazes; as it
rises it passes through spots, labyrinths and coral. Here the feed rate at
each cell is the live activity under it (pooled, salience-gated), so active
cortex grows different patterns from quiet cortex, and the boundaries between
pattern types move as the data does. The global signal shifts the kill rate
for the whole field, so whole-brain swings morph every pattern at once, and a
DVARS spike seeds fresh growth at random. Outside the head nothing is fed and
the chemistry dies away.

The pattern is coloured by which family lives under each cell and lit with a
relief from V's gradient -- procedural structure, nothing drawn by hand.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..drives import Param
from .base import Scene, axial_cells, blur, filled_field

S = "turing"


class TuringScene(Scene):
    name = S
    blurb = "reaction-diffusion inside the head: activity sets the feed, so patterns follow it"
    PARAMS = (
        Param("feed_lo", 0.036, 0.010, 0.080, S, "feed rate where quiet", scene=S),
        Param("feed_hi", 0.058, 0.010, 0.100, S, "feed rate where active", scene=S),
        Param("kill", 0.062, 0.040, 0.075, S, "kill rate (the global signal nudges it)", scene=S),
        Param("kill_swing", 0.003, 0.0, 0.01, S, "kill-rate swing with the global signal", scene=S),
        Param("iters", 20, 1, 60, S, "reaction steps per frame (growth speed)", scene=S),
        Param("cell", 2, 1, 6, S, "pixels per cell", scene=S),
        Param("relief", 1.0, 0.0, 3.0, S, "embossed lighting from the pattern's gradient", scene=S),
        Param("seed", 1.0, 0.0, 4.0, S, "fresh growth seeded on DVARS spikes", scene=S),
    )

    def __init__(self, drives, width, height, seed=0):
        super().__init__(drives, width, height, seed)
        self._alloc()

    def on_resize(self):
        self._alloc()

    def on_param(self, name, value):
        if name == "cell":
            self._alloc()

    def _alloc(self) -> None:
        f = self.f
        c = int(self.p["cell"])
        self.gh, self.gw = max(16, self.height // c), max(16, self.width // c)
        self.cell_of, self.inside, self.passes = axial_cells(
            f.anat, self.gh, self.gw, self.dr.spacing
        )
        self.U = torch.ones(self.gh, self.gw, device=self.device)
        self.V = torch.zeros(self.gh, self.gw, device=self.device)
        # Sparse, isolated blobs: a large uniform patch of V collapses instead of patterning.
        self._seed(0.004)
        # The family that dominates each cell, as a colour: averaging the
        # colours instead turns every cell the same pastel grey.
        onehot = torch.zeros(f.n_voxels, self.dr.n_fam, device=self.device)
        onehot[torch.arange(f.n_voxels, device=self.device), f.family] = f.family_weight.clamp_min(
            0.05
        )
        terr = filled_field(onehot, self.cell_of, self.gh, self.gw, self.passes + 2)  # (F, gh, gw)
        self.color = blur(self.dr.family_rgb()[terr.argmax(0)].permute(2, 0, 1), 1)
        self.lap = torch.tensor(
            [[0.05, 0.2, 0.05], [0.2, -1.0, 0.2], [0.05, 0.2, 0.05]], device=self.device
        )[None, None]
        # Warm up: Gray-Scott needs hundreds of steps before a pattern exists.
        self._react(torch.full_like(self.U, self.p["feed_lo"]), self.p["kill"], 400)

    def _seed(self, frac: float) -> None:
        drop = (self.rand(self.gh, self.gw) < frac) & self.inside
        # Seeds are small blobs, not single cells, so they survive diffusion.
        drop = blur(drop.float(), 2) > 0.02
        self.V = torch.where(drop, torch.full_like(self.V, 0.5), self.V)
        self.U = torch.where(drop, torch.full_like(self.U, 0.5), self.U)

    def _react(self, feed: torch.Tensor, kill: float, iters: int) -> None:
        feed = torch.where(self.inside, feed, torch.full_like(feed, 0.01))
        kill_f = torch.where(self.inside, torch.full_like(feed, kill), torch.full_like(feed, 0.08))
        U, V = self.U[None, None], self.V[None, None]
        for _ in range(iters):
            lu = F.conv2d(F.pad(U, (1, 1, 1, 1), mode="replicate"), self.lap)
            lv = F.conv2d(F.pad(V, (1, 1, 1, 1), mode="replicate"), self.lap)
            uvv = U * V * V
            U = U + (1.0 * lu - uvv + feed * (1 - U))
            V = V + (0.5 * lv + uvv - (feed + kill_f) * V)
        self.U, self.V = U[0, 0].clamp(0, 1), V[0, 0].clamp(0, 1)

    def step(self, dt: float) -> None:
        p, dr = self.p, self.dr
        act = filled_field(dr.brightness(), self.cell_of, self.gh, self.gw, self.passes + 1)
        act = 1 - torch.exp(-act / max(p["gain"], 1e-3) * 1.5)
        feed = p["feed_lo"] + (p["feed_hi"] - p["feed_lo"]) * act
        kill = p["kill"] + p["kill_swing"] * math.tanh(dr.g * 0.6)
        self._react(feed, kill, int(p["iters"]))
        if p["seed"] > 0 and dr.d > 2.5:
            self._seed(min(0.004, (dr.d - 2.5) * 0.0008 * p["seed"]))

    def render(self) -> torch.Tensor:
        p, dr = self.p, self.dr
        V = self.V
        tone = (V * 3.0).clamp(0, 1) ** 0.8
        # Relief: light from the upper left on V's gradient.
        gx = torch.zeros_like(V)
        gy = torch.zeros_like(V)
        gx[:, 1:-1] = V[:, 2:] - V[:, :-2]
        gy[1:-1] = V[2:] - V[:-2]
        shade = (1.0 + p["relief"] * 4.0 * (-gx - gy)).clamp(0.3, 2.0)
        img = self.color * tone * shade + 0.04 * self.color * self.inside
        img = F.interpolate(img[None], size=(self.height, self.width), mode="nearest")[0]
        c = self.width / self.gw
        cell = self.cell_of
        self.remember(
            ((cell % self.gw).float() + 0.5) * c,
            ((cell // self.gw).float() + 0.5) * (self.height / self.gh),
            weight=dr.brightness(),
        )
        return dr.finish(img.reshape(3, -1).T * 1.4, self.height, self.width)
