"""Sand: each family pours grains from its own lane, and the pile keeps the history.

The top edge is split into one lane per ICA family, ordered left to right by
the family's left/right centroid (so lateralised networks pour from their own
side) with widths following family size. A family pours while it is active,
at a rate set by its activity and its salience; each grain carries the
family's colour and glows with the heat it was born with, cooling as it
settles. A falling-sand automaton does the rest.

So the landscape is a record of the run: pile heights are each family's
cumulative activity, strata are time. The global signal blows a wind that
biases which way grains slide, a DVARS spike shakes the pile into avalanches,
and when the pile fills up it sinks, scrolling the strata down.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..drives import Param
from .base import Scene

S = "sand"


class SandScene(Scene):
    name = S
    blurb = "falling sand: families pour from their lanes; the pile is the history"
    PARAMS = (
        Param("pour", 1.0, 0.0, 5.0, S, "grains per unit of family activity", scene=S),
        Param("trickle", 0.05, 0.0, 1.0, S, "baseline pour of quiet families", scene=S),
        Param("onset", 0.3, -1.0, 2.0, S, "family activity (z) above which it pours", scene=S),
        Param("grain", 2, 1, 6, S, "pixels per grain", scene=S),
        Param("fall", 3, 1, 8, S, "automaton passes per frame (fall speed)", scene=S),
        Param("wind", 1.0, 0.0, 3.0, S, "sideways slide bias from the global signal", scene=S),
        Param("quake", 1.0, 0.0, 4.0, S, "avalanche on DVARS spikes", scene=S),
        Param("sink", 0.75, 0.3, 0.98, S, "fill fraction at which the pile sinks a row", scene=S),
        Param("heat", 1.0, 0.0, 3.0, S, "glow of fresh grains", scene=S),
        Param("nozzle", 1.0, 0.0, 20.0, S, "spread of each lane's pour, in grains", scene=S),
    )

    def __init__(self, drives, width, height, seed=0):
        super().__init__(drives, width, height, seed)
        f = self.f
        order = torch.argsort(f.family_center[:, 0])  # left to right
        widths = drives.family_size[order].sqrt()
        edges = torch.cat([torch.zeros(1, device=self.device), widths.cumsum(0) / widths.sum()])
        self.lane_order = order
        self.lane_edges = edges
        self._alloc()
        self.parity = 0
        self.pending = torch.zeros(drives.n_fam, device=self.device)

    def _alloc(self) -> None:
        g = int(self.p["grain"])
        self.gh, self.gw = max(8, self.height // g), max(8, self.width // g)
        self.occ = torch.zeros(self.gh, self.gw, dtype=torch.bool, device=self.device)
        self.col = torch.zeros(4, self.gh, self.gw, device=self.device)  # rgb + heat

    def on_resize(self):
        self._alloc()

    def on_param(self, name, value):
        if name == "grain":
            self._alloc()

    # ------------------------------------------------------------ automaton

    def _move(self, mover: torch.Tensor, dy: int, dx: int) -> None:
        """Move grains flagged in ``mover`` by (dy, dx); targets are known empty."""
        dst = torch.roll(mover, shifts=(dy, dx), dims=(0, 1))
        moved_col = torch.roll(self.col * mover, shifts=(dy, dx), dims=(1, 2))
        self.col = torch.where(mover, 0.0, self.col)
        self.col = torch.where(dst, moved_col, self.col)
        self.occ = (self.occ & ~mover) | dst

    def _pass(self, slide_dir: int) -> None:
        occ = self.occ
        h, w = occ.shape
        below = torch.zeros_like(occ)
        below[:-1] = occ[1:]
        floor = torch.zeros_like(occ)
        floor[-1] = True
        down = occ & ~below & ~floor
        self._move(down, 1, 0)
        occ = self.occ
        below = torch.zeros_like(occ)
        below[:-1] = occ[1:]
        diag = torch.ones_like(occ)  # occupied-or-wall at (y+1, x+dir)
        if slide_dir > 0:
            diag[:-1, :-1] = occ[1:, 1:]
        else:
            diag[:-1, 1:] = occ[1:, :-1]
        side = torch.ones_like(occ)  # the cell beside must be free too, or grains tunnel
        if slide_dir > 0:
            side[:, :-1] = occ[:, 1:]
        else:
            side[:, 1:] = occ[:, :-1]
        slide = occ & below & ~diag & ~side & ~floor
        self._move(slide, 1, slide_dir)

    # ---------------------------------------------------------------- frame

    def step(self, dt: float) -> None:
        p, dr = self.p, self.dr
        gw = self.gw
        # Pour: whole grains per lane this frame, carrying fractional remainders.
        sal = dr.family_salience / dr.family_salience.max().clamp_min(1e-6)
        act = (dr.fam - p["onset"]).clamp_min(0)
        rate = (act * p["pour"] + p["trickle"]) * sal * dr.family_size.sqrt() * gw * 2.5
        self.pending = self.pending + rate * dt
        n_new = self.pending.floor()
        self.pending = self.pending - n_new
        fam_rgb = dr.family_rgb()
        top = 0
        for lane, fam in enumerate(self.lane_order.tolist()):
            k = int(n_new[fam])
            if k <= 0:
                continue
            a = int(self.lane_edges[lane] * gw)
            b = max(a + 1, int(self.lane_edges[lane + 1] * gw))
            # A narrow nozzle at the lane's centre: streams and cones, like an
            # hourglass, rather than confetti across the whole lane.
            centre = 0.5 * (a + b)
            xs = (
                (centre + self.randn(k) * p["nozzle"])
                .round()
                .long()
                .clamp(a, b - 1)
                .clamp(0, gw - 1)
            )
            ys = torch.full_like(xs, top) + (self.rand(k) * 2).long()
            free = ~self.occ[ys, xs]
            xs, ys = xs[free], ys[free]
            if xs.numel() == 0:
                continue
            tone = 0.75 + 0.25 * self.rand(xs.numel())  # a little grain-to-grain texture
            self.occ[ys, xs] = True
            self.col[:3, ys, xs] = fam_rgb[fam][:, None] * tone[None]
            self.col[3, ys, xs] = float(act[fam]) * p["heat"]

        wind = math.tanh(dr.g * 0.7) * p["wind"]
        quake = dr.d > 2.5 and p["quake"] > 0
        passes = int(p["fall"]) + (int((dr.d - 2.5) * p["quake"] * 2) if quake else 0)
        for _ in range(passes):
            self.parity ^= 1
            # Wind biases the slide direction: with no wind they alternate.
            prob_right = 0.5 + 0.45 * max(-1.0, min(1.0, wind))
            slide_dir = 1 if float(torch.rand(1, generator=self.gen)) < prob_right else -1
            self._pass(slide_dir)
        if quake:
            # Shake: lift a scatter of surface grains a row so they re-settle.
            surface = self.occ & ~torch.roll(self.occ, 1, 0)
            lift = surface & (
                self.rand(*self.occ.shape) < min(0.5, (dr.d - 2.5) * 0.1 * p["quake"])
            )
            lift[0] = False
            above_free = ~torch.roll(self.occ, 1, 0)
            self._move(lift & above_free, -1, 0)

        self.col[3] = self.col[3] * math.exp(-1.5 * dt)
        # Sink when full: drop the bottom row, the strata scroll down.
        if float(self.occ.float().mean()) > p["sink"]:
            self.occ = torch.roll(self.occ, 1, 0)
            self.occ[0] = False
            self.col = torch.roll(self.col, 1, 1)
            self.col[:, 0] = 0

    def render(self) -> torch.Tensor:
        dr = self.dr
        heat = self.col[3:4]
        img = self.col[:3] * (0.55 + heat) * self.occ
        img = F.interpolate(img[None], size=(self.height, self.width), mode="nearest")[0]
        return dr.finish(img.reshape(3, -1).T * 1.6, self.height, self.width)
