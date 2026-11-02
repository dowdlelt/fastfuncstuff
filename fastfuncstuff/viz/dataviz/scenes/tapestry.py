"""Tapestry: the weave itself, every voxel a stitch, rippling like cloth.

All the voxels tiled over the full field by correlation (the weave canvas),
each stitch coloured by its family and glowing with its live value. The live
field is also a height map: the cloth is lit from the upper left by its
gradient and rippled by it, so rising patches swell and catch the light. Since
neighbouring stitches are correlated voxels, activity shows as coherent
patches that bloom and fade rather than as noise, and a block design is a
pattern woven in and out of the fabric.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..drives import Param
from .base import Scene, blur

S = "tapestry"


class TapestryScene(Scene):
    name = S
    blurb = "every voxel a stitch on the weave; the live field ripples the cloth"
    PARAMS = (
        Param(
            "swell", 1.0, 0.0, 4.0, S, "how far the cloth ripples with activity, pixels", scene=S
        ),
        Param("sheen", 1.0, 0.0, 3.0, S, "lighting from the activity height map", scene=S),
        Param("weft", 0.25, 0.0, 1.0, S, "brightness of a stitch at rest", scene=S),
    )

    def __init__(self, drives, width, height, seed=0):
        super().__init__(drives, width, height, seed)
        self._alloc()

    def on_resize(self):
        self._alloc()

    def _alloc(self) -> None:
        row, col, self.rows, self.cols = self.dr.weave(self.width / self.height)
        self.idx = (row * self.rows).long().clamp(0, self.rows - 1) * self.cols + (
            col * self.cols
        ).long().clamp(0, self.cols - 1)
        self.row, self.col = row, col
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, self.height, device=self.device),
            torch.linspace(-1, 1, self.width, device=self.device),
            indexing="ij",
        )
        self.base_grid = torch.stack([xs, ys], -1)[None]

    def step(self, dt: float) -> None:
        pass

    def render(self) -> torch.Tensor:
        p, dr, f = self.p, self.dr, self.f
        n = self.rows * self.cols
        glow = dr.brightness() / max(p["gain"], 1e-3)
        rgb = dr.colors() * glow[:, None] * 0.6 + dr.family_rgb()[f.family] * p["weft"] * (
            0.3 + f.salience[:, None]
        )
        cloth = (
            torch.zeros(n, 3, device=self.device)
            .index_add_(0, self.idx, rgb)
            .T.reshape(3, self.rows, self.cols)
        )
        height = torch.zeros(n, device=self.device).index_add_(0, self.idx, dr.live.clamp(-2, 3))
        height = blur(height.reshape(self.rows, self.cols), 2)
        gy, gx = torch.gradient(height)
        shade = (1.0 + p["sheen"] * 0.6 * (-gx - gy)).clamp(0.4, 2.0)
        # Pool on the weave itself -- neighbours here are correlated voxels, so
        # this is correlation pooling -- and the cloth shows patches, not grain.
        cloth = blur(cloth, 1) * shade[None]
        # Ripple: sample the cloth displaced along the height gradient.
        disp = (
            torch.stack([gx, gy], -1)[None]
            * (p["swell"] * 2.0 / max(self.width, self.height))
            * 4.0
        )
        disp = F.interpolate(
            disp.permute(0, 3, 1, 2), size=(self.height, self.width), mode="bilinear"
        ).permute(0, 2, 3, 1)
        img = F.grid_sample(
            cloth[None],
            self.base_grid + disp,
            mode="nearest",
            padding_mode="border",
            align_corners=True,
        )[0]
        self.remember(self.col * self.width, self.row * self.height, weight=dr.brightness())
        return dr.finish(img.reshape(3, -1).T * 1.4, self.height, self.width)
