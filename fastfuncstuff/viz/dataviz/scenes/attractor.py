"""Attractor: a strange attractor rendered as a flame, its shape steered by the data.

Every voxel is a point iterated through a two-dimensional chaotic map -- Peter
de Jong's, x' = sin(a y) - cos(b x), y' = sin(c x) - cos(d y), or Clifford
Pickover's. Points fall onto the attractor and paint it as a density, drawn
on a log scale the way fractal flames are, so the fine filaments show next to
the bright folds.

The four map parameters are four principal-component time courses, smoothed
over a couple of seconds, so the attractor morphs continuously as the run's
dominant modes rise and fall: a block design makes it breathe between two
shapes, a drift lets it slide through a family of them. Each point carries
its family's colour and burns brighter while its voxel is active.

Every family runs its own slightly offset copy of the map, the offset growing
with the family's activity. Quiet families overlap on one shape and their
colours sum to white; an active family peels away into its own coloured
filaments. (With one shared map every colour averaged to white.)
"""

from __future__ import annotations

import math

import torch

from ..drives import Param, interp_rows, splat
from .base import Scene

S = "attractor"
MAPS = ("dejong", "clifford")
_BASE = {"dejong": (1.4, -2.3, 2.4, -2.1), "clifford": (-1.4, 1.6, 1.0, 0.7)}


class AttractorScene(Scene):
    name = S
    blurb = "a strange attractor as a flame; four PC time courses steer its parameters"
    PARAMS = (
        Param(
            "map",
            1,
            0,
            len(MAPS) - 1,
            S,
            "chaotic map (de Jong is structurally stable: families barely peel)",
            MAPS,
            scene=S,
        ),
        Param("steer", 0.4, 0.0, 1.0, S, "parameter swing per unit of PC activity", scene=S),
        Param(
            "peel", 0.3, 0.0, 0.6, S, "how far an active family's own attractor peels away", scene=S
        ),
        Param("ease", 2.0, 0.1, 10.0, S, "seconds of smoothing on the steering", scene=S),
        Param("hops", 4, 1, 12, S, "map iterations drawn per frame", scene=S),
        Param("afterglow", 0.9, 0.0, 0.99, S, "density persistence", scene=S),
        Param("burn", 1.0, 0.0, 4.0, S, "extra brightness of active voxels' points", scene=S),
        Param("azoom", 1.0, 0.3, 3.0, S, "zoom", scene=S),
        Param("flame", 1.2, 0.2, 4.0, S, "flame brightness", scene=S),
    )

    def __init__(self, drives, width, height, seed=0):
        super().__init__(drives, width, height, seed)
        self.pts = (self.rand(self.f.n_voxels, 2) - 0.5) * 4
        self.steer = torch.zeros(4, device=self.device)
        self.offset = torch.nn.functional.normalize(self.randn(self.dr.n_fam, 4), dim=1)

    def step(self, dt: float) -> None:
        p, dr = self.p, self.dr
        k = min(4, self.f.tcs.shape[1])
        target = torch.zeros(4, device=self.device)
        target[:k] = torch.tanh(interp_rows(self.f.tcs, dr.t)[:k] * 0.6)
        self.steer = self.steer + (target - self.steer) * min(1.0, dt / max(p["ease"], 1e-3))

    def _params(self) -> torch.Tensor:
        """(4,) map parameters on the device: base plus steering (no read-back)."""
        base = self.dr._const(MAPS[int(self.p["map"])], _BASE[MAPS[int(self.p["map"])]])
        return base + self.p["steer"] * self.steer

    def render(self) -> torch.Tensor:
        p, dr, f = self.p, self.dr, self.f
        h, w = self.height, self.width
        par = self._params()
        # Per-point parameters: the shared map plus the family's peel.
        fam = f.family
        if self.offset.shape[0] != dr.n_fam:
            self.offset = torch.nn.functional.normalize(self.randn(dr.n_fam, 4), dim=1)
        peel = self.offset * (p["peel"] * (0.15 + dr.fam.clamp(0, 3)))[:, None]  # (F, 4)
        pa, pb, pc, pd = (par[i] + peel[fam, i] for i in range(4))
        mode = MAPS[int(p["map"])]
        # Framing from the base parameters: steering moves them by at most
        # `steer`, so the host can fit the view without reading the device.
        base = _BASE[mode]
        reach = 2.0 if mode == "dejong" else 1.0 + max(abs(base[2]), abs(base[3])) + self.p["steer"]
        scale = 0.48 * min(w, h) * p["azoom"] / reach
        hops = int(p["hops"])
        glow = (0.2 + p["burn"] * dr.brightness() / max(p["gain"], 1e-3)) * self.density / hops
        rgb = dr.family_rgb()[f.family] * glow[:, None]
        frame = torch.zeros(h * w, 3, device=self.device)
        x, y = self.pts[:, 0], self.pts[:, 1]
        for _ in range(hops):
            if mode == "dejong":
                x, y = torch.sin(pa * y) - torch.cos(pb * x), torch.sin(pc * x) - torch.cos(pd * y)
            else:
                x, y = (
                    torch.sin(pa * y) + pc * torch.cos(pa * x),
                    torch.sin(pb * x) + pd * torch.cos(pb * y),
                )
            splat(frame, w * 0.5 + x * scale, h * 0.5 + y * scale, rgb, w, h)
        self.pts = torch.stack([x, y], 1)
        self.remember(w * 0.5 + x * scale, h * 0.5 + y * scale)
        k = p["afterglow"]
        self.accum.mul_(k).add_(frame * (1 - k))
        # Flame tone: log density, so filaments show beside the bright folds.
        flame = torch.log1p(self.accum * 400.0) / math.log1p(400.0)
        return dr.finish(flame * p["flame"], h, w)
