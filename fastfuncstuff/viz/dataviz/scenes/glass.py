"""Stained glass: the head as a Voronoi window, each pane lit by its voxels.

Pane centres are spread over the head from above by farthest-point sampling,
so the panes tile it evenly; every voxel belongs to the pane it falls in. A
pane takes the colour of the family that dominates it and glows with the mean
live activity of its voxels, so the window lights up region by region as the
data plays -- a block design is a set of panes that flare together, a
travelling wave walks across the glass. The lead between panes is drawn from
the gap between the nearest and second-nearest centre; it catches the light
when a DVARS spike flashes. Panes drift a little with their own activity, so
the leading flexes as the window breathes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..drives import Param
from .base import Scene, axial_cells

S = "glass"


class GlassScene(Scene):
    name = S
    blurb = "a Voronoi window over the head; each pane glows with its voxels' activity"
    PARAMS = (
        Param("panes", 180, 20, 800, S, "how many panes", scene=S),
        Param("lead", 1.2, 0.0, 5.0, S, "lead width, cells", scene=S),
        Param("sway", 0.6, 0.0, 3.0, S, "how far panes drift with their activity, cells", scene=S),
        Param("ember", 0.06, 0.0, 1.0, S, "glow of a pane at rest", scene=S),
        Param("gcell", 2, 1, 4, S, "pixels per cell", scene=S),
    )

    def __init__(self, drives, width, height, seed=0):
        super().__init__(drives, width, height, seed)
        self._alloc()

    def on_resize(self):
        self._alloc()

    def on_param(self, name, value):
        if name in ("gcell", "panes"):
            self._alloc()

    def _alloc(self) -> None:
        f, dr = self.f, self.dr
        c = int(self.p["gcell"])
        self.gh, self.gw = max(32, self.height // c), max(32, self.width // c)
        self.cell_of, self.inside, _ = axial_cells(f.anat, self.gh, self.gw, dr.spacing)
        vox_xy = torch.stack(
            [(self.cell_of % self.gw).float(), (self.cell_of // self.gw).float()], 1
        )
        # Farthest-point sampling over the voxels: even panes, no clumps.
        m = min(int(self.p["panes"]), f.n_voxels)
        pick = [int(torch.argmax(f.salience))]
        dist = (vox_xy - vox_xy[pick[0]]).square().sum(1)
        for _ in range(m - 1):
            i = int(torch.argmax(dist))
            pick.append(i)
            dist = torch.minimum(dist, (vox_xy - vox_xy[i]).square().sum(1))
        self.home = vox_xy[pick]  # (M, 2) pane centres in cells
        self.member = torch.cdist(vox_xy, self.home).argmin(1)  # voxel -> pane
        cnt = torch.bincount(self.member, minlength=m).float()
        self.cnt = cnt.clamp_min(1)
        onehot = torch.zeros(m, dr.n_fam, device=self.device)
        onehot.index_put_((self.member, f.family), f.family_weight.clamp_min(0.05), accumulate=True)
        self.pane_rgb = dr.family_rgb()[onehot.argmax(1)]
        self.drift_dir = F.normalize(self.randn(m, 2), dim=1)
        ys, xs = torch.meshgrid(
            torch.arange(self.gh, device=self.device, dtype=torch.float32),
            torch.arange(self.gw, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        self.grid = torch.stack([xs.reshape(-1), ys.reshape(-1)], 1)
        self.inside_flat = self.inside.reshape(-1)
        self.texture = 0.85 + 0.15 * self.rand(self.gh * self.gw)

    def step(self, dt: float) -> None:
        pass

    def render(self) -> torch.Tensor:
        p, dr = self.p, self.dr
        act = (
            torch.zeros(len(self.home), device=self.device).index_add_(
                0, self.member, dr.brightness()
            )
            / self.cnt
        )
        act = self.relative(act)
        centres = self.home + self.drift_dir * (act * p["sway"])[:, None]
        pts = self.grid[self.inside_flat]
        d, idx = torch.cdist(pts, centres).topk(2, dim=1, largest=False)
        pane = idx[:, 0]
        # Brighter toward the pane's centre, like light through thick glass.
        depth = 1 - 0.35 * d[:, 0] / (d[:, 0] + d[:, 1]).clamp_min(1e-3)
        light = (p["ember"] + act[pane] ** 1.5 * 2.2) * depth * self.texture[self.inside_flat]
        rgb = self.pane_rgb[pane] * light[:, None]
        lead = (d[:, 1] - d[:, 0]) < p["lead"]
        flash = max(0.0, dr.d - 2.5) * 0.3
        rgb = torch.where(
            lead[:, None], torch.tensor([0.9, 0.75, 0.45], device=self.device) * (0.04 + flash), rgb
        )
        img = torch.zeros(self.gh * self.gw, 3, device=self.device)
        img[self.inside_flat] = rgb
        img = img.T.reshape(3, self.gh, self.gw)
        img = F.interpolate(img[None], size=(self.height, self.width), mode="nearest")[0]
        cw, ch = self.width / self.gw, self.height / self.gh
        cell = self.cell_of
        self.remember(
            ((cell % self.gw).float() + 0.5) * cw,
            ((cell // self.gw).float() + 0.5) * ch,
            weight=dr.brightness(),
        )
        return dr.finish(img.reshape(3, -1).T * 1.4, self.height, self.width)
