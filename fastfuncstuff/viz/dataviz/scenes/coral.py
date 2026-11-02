"""Coral: each family grows a living reef in its own territory while it is active.

The grid sees the head from above. Every family owns a territory -- the cells
where its member voxels are dense -- and a few immortal seed cells at the
heart of it. While a family is active it grows: empty cells next to its reef
are claimed with a probability set by the family's activity and how much of
its territory the cell is, strongly preferring tips (a cell with exactly one
reef neighbour), so growth branches like coral or lichen instead of filling
blobs. When the family goes quiet its reef dies back, oldest first.

Each cell remembers when it was born, and the colour carries faint bands by
birth time, so the reef's growth rings are the run's history. Fresh tips glow.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..drives import Param
from .base import Scene, filled_field

S = "coral"


class CoralScene(Scene):
    name = S
    blurb = "families grow branching reefs in their own territory while active; quiet reefs recede"
    PARAMS = (
        Param("growth", 1.0, 0.0, 5.0, S, "growth rate per unit of family activity", scene=S),
        Param("bud", 0.0, -1.0, 2.0, S, "family activity (z) above which it grows", scene=S),
        Param(
            "fractal",
            2.0,
            0.5,
            4.0,
            S,
            "nutrient exponent: higher grows thinner, more branched fingers",
            scene=S,
        ),
        Param("nutrient", 1.0, 0.0, 4.0, S, "nutrient released by active cortex", scene=S),
        Param("lifespan", 20.0, 2.0, 120.0, S, "seconds an untended cell lives", scene=S),
        Param("rings", 8.0, 1.0, 40.0, S, "TRs per growth-ring band", scene=S),
        Param("bloom_c", 1.0, 0.0, 3.0, S, "glow of fresh growth", scene=S),
        Param("ccell", 2, 1, 6, S, "pixels per cell", scene=S),
    )

    def __init__(self, drives, width, height, seed=0):
        super().__init__(drives, width, height, seed)
        self._alloc()

    def on_resize(self):
        self._alloc()

    def on_param(self, name, value):
        if name in ("ccell", "canvas"):
            self._alloc()

    def _alloc(self) -> None:
        f, dr = self.f, self.dr
        c = int(self.p["ccell"])
        self.gh, self.gw = max(16, self.height // c), max(16, self.width // c)
        n = self.gh * self.gw
        self.cell_of, self.inside, self.passes = self.grid_cells(self.gh, self.gw)
        self.inside = self.inside.reshape(-1)
        onehot = torch.zeros(f.n_voxels, dr.n_fam, device=self.device)
        onehot[torch.arange(f.n_voxels, device=self.device), f.family] = f.family_weight.clamp_min(
            0.05
        )
        terr = filled_field(onehot, self.cell_of, self.gh, self.gw, self.passes + 2).reshape(
            dr.n_fam, -1
        )
        self.territory = terr / terr.amax(1, keepdim=True).clamp_min(1e-6)  # (F, n)
        # Three immortal seeds per family at the heart of its territory.
        self.seeds = self.territory.topk(3, dim=1).indices  # (F, 3)
        self.label = torch.zeros(n, dtype=torch.long, device=self.device)  # 0 empty, f+1 family
        self.born = torch.zeros(n, device=self.device)  # data time (TRs) of birth
        self.age = torch.zeros(n, device=self.device)  # seconds
        self.food = torch.zeros(self.gh, self.gw, device=self.device)
        self._plant()
        self.kernel = torch.ones(1, 1, 3, 3, device=self.device)
        self.cross = torch.tensor(
            [[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]], device=self.device
        )[None, None]

    def _plant(self) -> None:
        for k in range(self.dr.n_fam):
            self.label[self.seeds[k]] = k + 1

    def step(self, dt: float) -> None:
        p, dr = self.p, self.dr
        sal = dr.family_salience / dr.family_salience.max().clamp_min(1e-6)
        act = ((dr.fam - p["bud"]).clamp(0, 3) * sal).tolist()
        self.age = self.age + dt
        # Nutrient: released by active cortex, diffusing, eaten by the reef.
        # Growth goes as nutrient**fractal at the frontier: tips that reach out
        # into fresh nutrient starve the fjords behind them, which is the
        # dielectric-breakdown model of DLA -- branching fingers, not blobs.
        occ2 = (self.label > 0).reshape(self.gh, self.gw)
        act_field = filled_field(dr.brightness(), self.cell_of, self.gh, self.gw, self.passes + 1)
        src = (1 - torch.exp(-act_field / max(p["gain"], 1e-3))) * self.inside.reshape(
            self.gh, self.gw
        )
        food = self.food[None, None]
        for _ in range(6):
            food = F.conv2d(F.pad(food, (1, 1, 1, 1), mode="replicate"), self.cross)
            food = food + (src * p["nutrient"] * 0.05)[None, None]
            food = torch.where(occ2[None, None], torch.zeros_like(food), food).clamp(0, 1)
        self.food = food[0, 0]
        hunger = self.food.reshape(-1) ** p["fractal"]
        empty = (self.label == 0) & self.inside
        for k in range(dr.n_fam):
            if act[k] <= 0:
                continue
            mine = (self.label == k + 1).float().reshape(1, 1, self.gh, self.gw)
            nb = F.conv2d(F.pad(mine, (1, 1, 1, 1)), self.kernel).reshape(-1) - mine.reshape(-1)
            frontier = empty & (nb > 0)
            prob = (p["growth"] * act[k] * dt * 40.0) * hunger * (0.25 + self.territory[k])
            claim = frontier & (self.rand(prob.numel()) < prob)
            self.label[claim] = k + 1
            self.born[claim] = dr.t
            self.age[claim] = 0.0
            empty = empty & ~claim
        # Die-back: untended cells of quiet families age out, oldest first.
        fam = (self.label - 1).clamp_min(0)
        a = torch.tensor(act, device=self.device)[fam].clamp(0, 1)
        risk = (self.age / p["lifespan"]).clamp(0, 3) * (1 - a) * dt / p["lifespan"] * 3.0
        dead = (self.label > 0) & (self.rand(risk.numel()) < risk)
        self.label[dead] = 0
        self._plant()
        # Tended cells stay young.
        self.age = torch.where((self.label > 0) & (a > 0.3), self.age * (1 - 0.5 * dt), self.age)

    def render(self) -> torch.Tensor:
        p, dr = self.p, self.dr
        occ = self.label > 0
        fam = (self.label - 1).clamp_min(0)
        col = dr.family_rgb()[fam]  # (n, 3)
        band = 0.82 + 0.18 * torch.cos(2 * math.pi * self.born / p["rings"])
        fresh = torch.exp(-self.age / 1.5) * p["bloom_c"]
        img = col * (band * 0.75 + fresh)[:, None] + fresh[:, None] * 0.4
        img = img * occ[:, None] + 0.012 * self.inside[:, None].float()
        img = img.T.reshape(3, self.gh, self.gw)
        img = F.interpolate(img[None], size=(self.height, self.width), mode="nearest")[0]
        cw, ch = self.width / self.gw, self.height / self.gh
        cell = self.cell_of
        self.remember(
            ((cell % self.gw).float() + 0.5) * cw,
            ((cell // self.gw).float() + 0.5) * ch,
            weight=dr.brightness(),
        )
        return dr.finish(img.reshape(3, -1).T * 1.3, self.height, self.width)
