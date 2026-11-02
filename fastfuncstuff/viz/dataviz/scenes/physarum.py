"""Physarum: a slime mould per family, foraging on the data inside the head.

The Jones (2010) agent model of Physarum polycephalum: each agent senses the
trail field ahead, ahead-left and ahead-right, turns toward the strongest,
steps, and deposits; the field diffuses and decays. From these rules emerge
transport networks -- veins that merge, prune and reroute.

Each agent is a salient voxel and belongs to its family's species. A species
follows its own trail and is put off by the others', so families carve out
interleaved networks. Food is the live activity under each cell, so the
networks reach toward active cortex and abandon quiet tissue; an agent also
deposits more while its own voxel is active. The global signal widens the
sensor angle (calm networks vs. restless ones), and a DVARS spike scrambles
headings. Agents stay inside the head's silhouette.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..drives import Param
from .base import Scene, blur, filled_field

S = "physarum"


class PhysarumScene(Scene):
    name = S
    blurb = "slime moulds, one species per family, foraging on activity"
    PARAMS = (
        Param("agents", 40000, 5000, 300000, S, "agents (the most salient voxels)", scene=S),
        Param("sense", 0.5, 0.1, 1.5, S, "sensor angle, radians", scene=S),
        Param("reach_s", 6.0, 1.0, 20.0, S, "sensor distance, cells", scene=S),
        Param("turn", 0.4, 0.05, 1.5, S, "turn per step, radians", scene=S),
        Param("stride", 1.0, 0.2, 3.0, S, "step length, cells", scene=S),
        Param("shun", 0.6, 0.0, 2.0, S, "how much a species avoids the others' trails", scene=S),
        Param("food", 1.0, 0.0, 5.0, S, "pull of active cortex", scene=S),
        Param("evaporate", 0.9, 0.5, 0.99, S, "trail kept per step", scene=S),
        Param("pcell", 1, 1, 4, S, "pixels per cell", scene=S),
    )

    def __init__(self, drives, width, height, seed=0):
        super().__init__(drives, width, height, seed)
        self._alloc()

    def on_resize(self):
        self._alloc()

    def on_param(self, name, value):
        if name in ("canvas", "pcell", "agents"):
            self._alloc()

    def _alloc(self) -> None:
        f, dr = self.f, self.dr
        c = int(self.p["pcell"])
        self.gh, self.gw = max(32, self.height // c), max(32, self.width // c)
        self.cell_of, self.inside, self.passes = self.grid_cells(self.gh, self.gw)
        m = min(int(self.p["agents"]), f.n_voxels)
        self.vox = torch.argsort(f.salience, descending=True)[:m]
        self.species = f.family[self.vox]
        cell = self.cell_of[self.vox]
        self.pos = torch.stack(
            [(cell % self.gw).float(), (cell // self.gw).float()], 1
        ) + self.rand(m, 2)
        self.heading = self.rand(m) * 2 * math.pi
        self.trail = torch.zeros(dr.n_fam, self.gh, self.gw, device=self.device)

    def _sample(
        self, field_flat: torch.Tensor, xy: torch.Tensor, offset: torch.Tensor | int = 0
    ) -> torch.Tensor:
        x = xy[:, 0].round().long().clamp(0, self.gw - 1)
        y = xy[:, 1].round().long().clamp(0, self.gh - 1)
        return field_flat[offset + y * self.gw + x]

    def step(self, dt: float) -> None:
        p, dr = self.p, self.dr
        n_cells = self.gh * self.gw
        act = filled_field(dr.brightness(), self.cell_of, self.gh, self.gw, self.passes + 1)
        food = (1 - torch.exp(-act / max(p["gain"], 1e-3))) * self.inside * p["food"]
        sa = p["sense"] * (1 + 0.5 * math.tanh(dr.g * 0.6))
        dep = 0.5 + dr.brightness()[self.vox] / max(p["gain"], 1e-3)
        sp_off = self.species * n_cells
        for _ in range(2):
            total = self.trail.sum(0)
            own_flat = self.trail.reshape(-1)
            score = []
            for ang in (-sa, 0.0, sa):
                h = self.heading + ang
                probe = self.pos + p["reach_s"] * torch.stack([torch.cos(h), torch.sin(h)], 1)
                own = self._sample(own_flat, probe, sp_off)
                rest = self._sample(total.reshape(-1), probe) - own
                score.append(own - p["shun"] * rest + self._sample(food.reshape(-1), probe))
            left, mid, right = score
            jitter = (self.rand(len(mid)) - 0.5) * 0.2
            turn = torch.where(
                (mid >= left) & (mid >= right), jitter, torch.where(left > right, -1.0, 1.0)
            )
            self.heading = self.heading + turn * p["turn"]
            step = p["stride"] * torch.stack([torch.cos(self.heading), torch.sin(self.heading)], 1)
            nxt = self.pos + step
            ok = self._sample(self.inside.reshape(-1).float(), nxt) > 0
            ok &= (
                (nxt[:, 0] >= 0) & (nxt[:, 0] < self.gw) & (nxt[:, 1] >= 0) & (nxt[:, 1] < self.gh)
            )
            self.pos = torch.where(ok[:, None], nxt, self.pos)
            # Blocked agents turn around, a little at random.
            self.heading = torch.where(
                ok, self.heading, self.heading + math.pi + (self.rand(len(ok)) - 0.5)
            )
            x = self.pos[:, 0].long().clamp(0, self.gw - 1)
            y = self.pos[:, 1].long().clamp(0, self.gh - 1)
            flat = self.trail.reshape(-1)
            flat.index_add_(0, sp_off + y * self.gw + x, dep)
            self.trail = blur(self.trail, 1) * p["evaporate"]
        if dr.d > 2.5:
            scramble = self.rand(len(self.heading)) < min(0.5, (dr.d - 2.5) * 0.1)
            self.heading = torch.where(
                scramble, self.rand(len(self.heading)) * 2 * math.pi, self.heading
            )

    def render(self) -> torch.Tensor:
        dr = self.dr
        # Each species against its own bright end, so a sparse network and a
        # dense one both read; veins, not a uniform glow.
        sample = self.trail.reshape(self.dr.n_fam, -1)[:, ::7]
        ref = (
            torch.topk(sample, max(1, sample.shape[1] // 200), dim=1).values[:, -1].clamp_min(1e-3)
        )
        tr = (self.trail / ref[:, None, None]).clamp(0, 1.5) ** 1.5
        img = torch.einsum("fc,fhw->chw", dr.family_rgb(), tr)
        img = F.interpolate(img[None], size=(self.height, self.width), mode="nearest")[0]
        cw, ch = self.width / self.gw, self.height / self.gh
        self.remember(
            self.pos[:, 0] * cw, self.pos[:, 1] * ch, self.vox, weight=dr.brightness()[self.vox]
        )
        return dr.finish(img.reshape(3, -1).T * 1.6, self.height, self.width)
