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
    CANDIDATES = 16  # grains each lane may drop per frame
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

    def _alloc(self) -> None:
        g = int(self.p["grain"])
        self.gh, self.gw = max(8, self.height // g), max(8, self.width // g)
        self.occ = torch.zeros(self.gh, self.gw, dtype=torch.bool, device=self.device)
        # Lane geometry per family, in cells: the nozzle's centre and its
        # bounds (lane_order maps lane position to family).
        lo = torch.zeros(self.dr.n_fam, dtype=torch.long, device=self.device)
        hi = torch.zeros_like(lo)
        for lane, fam in enumerate(self.lane_order.tolist()):
            a = int(self.lane_edges[lane] * self.gw)
            b = max(a + 1, int(self.lane_edges[lane + 1] * self.gw))
            lo[fam], hi[fam] = min(a, self.gw - 1), min(b - 1, self.gw - 1)
        self.lane_lo, self.lane_hi = lo, hi
        self.lane_mid = (lo + hi).float() * 0.5
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
        # Pour: a fixed number of candidate grains per lane per frame, each kept
        # with the probability that gives the lane its rate. Fixed shapes and
        # no host reads -- the per-lane loop read each lane's count back off the
        # GPU every frame.
        sal = dr.family_salience / dr.family_salience.max().clamp_min(1e-6)
        act = (dr.fam - p["onset"]).clamp_min(0)
        rate = (act * p["pour"] + p["trickle"]) * sal * dr.family_size.sqrt() * gw * 2.5
        n_f, c = dr.n_fam, self.CANDIDATES
        keep_p = (rate * dt / c).clamp(0, 1)[:, None]
        xs = (self.lane_mid[:, None] + self.randn(n_f, c) * p["nozzle"]).round().long()
        xs = torch.maximum(torch.minimum(xs, self.lane_hi[:, None]), self.lane_lo[:, None]).reshape(
            -1
        )
        ys = (self.rand(n_f, c) * 2).long().reshape(-1)
        idx = ys * gw + xs
        occ_flat = self.occ.reshape(-1)
        drop = ((self.rand(n_f, c) < keep_p).reshape(-1)) & ~occ_flat[idx]
        tone = 0.75 + 0.25 * self.rand(n_f * c)
        fam_rgb = dr.family_rgb()[:, :, None].expand(n_f, 3, c).permute(1, 0, 2).reshape(3, -1)
        heat = (act * p["heat"])[:, None].expand(n_f, c).reshape(-1)
        new = torch.cat([fam_rgb * tone[None], heat[None]], 0)  # (4, n_f*c)
        col_flat = self.col.reshape(4, -1)
        col_flat[:, idx] = torch.where(drop[None], new, col_flat[:, idx])
        # Counted, so a candidate that lands on an already-claimed cell cannot
        # clear it when two candidates share a cell (an accumulating
        # index_put, which every backend has; scatter_reduce is patchier).
        hits = torch.zeros(occ_flat.numel(), dtype=torch.int32, device=self.device)
        hits.index_put_((idx,), drop.to(torch.int32), accumulate=True)
        self.occ = (occ_flat | (hits > 0)).reshape(self.occ.shape)

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
        # The fill check reads the GPU, so only every 15 frames.
        self._tick = getattr(self, "_tick", 0) + 1
        if self._tick % 15 == 0 and float(self.occ.float().mean()) > p["sink"]:
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
