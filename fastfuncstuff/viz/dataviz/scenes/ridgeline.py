"""Ridgeline: the brain as a scrolling stack of profiles, Unknown Pleasures style.

Voxels are laid along one line -- grouped by family (ordered by the family's
height in the head), and within a family by anatomical azimuth -- and binned
into columns. Every few tenths of a TR the live activity of each column
becomes one ridge; ridges recede into the distance in perspective, nearer
ridges hiding farther ones. So the picture is space across and time into
depth: a block design is a regular procession of mountain ranges in some
families, a travelling wave is a ridge that drifts sideways as it recedes.
"""

from __future__ import annotations

import torch

from ..drives import Param
from .base import Scene, splat_lines

S = "ridgeline"


class RidgelineScene(Scene):
    name = S
    blurb = "space across, time into depth: one ridge per moment, receding"
    PARAMS = (
        Param("columns", 160, 32, 480, S, "bins along the voxel line", scene=S),
        Param("rows", 56, 8, 160, S, "ridges kept", scene=S),
        Param("every", 0.5, 0.1, 4.0, S, "TRs between ridges", scene=S),
        Param("height", 2.0, 0.1, 6.0, S, "ridge height per unit of activity", scene=S),
        Param("depth", 0.006, 0.0, 0.05, S, "perspective shrink per ridge", scene=S),
        Param("mono", 0, 0, 1, S, "white ridges (1) or family colours (0)", scene=S),
    )

    def __init__(self, drives, width, height, seed=0):
        super().__init__(drives, width, height, seed)
        self._build()

    def on_param(self, name, value):
        if name in ("columns", "rows"):
            self._build()

    def _build(self) -> None:
        f, dr = self.f, self.dr
        height_rank = torch.argsort(torch.argsort(f.family_center[:, 2]))
        az = torch.atan2(f.anat[:, 1], f.anat[:, 0])
        key = height_rank[f.family].float() * 10.0 + (az + torch.pi) / (2 * torch.pi)
        order = torch.argsort(key)
        b = int(self.p["columns"])
        self.bin_of = torch.empty(f.n_voxels, dtype=torch.long, device=self.device)
        self.bin_of[order] = torch.arange(f.n_voxels, device=self.device) * b // f.n_voxels
        self.n_bins = b
        cnt = torch.bincount(self.bin_of, minlength=b).float()
        self.cnt = cnt.clamp_min(1)
        famc = torch.zeros(b, 3, device=self.device).index_add_(
            0, self.bin_of, dr.family_rgb()[f.family]
        )
        self.bin_rgb = famc / self.cnt[:, None]
        # Most salient voxel per bin, for clicking a ridge.
        best = torch.full((b,), -1.0, device=self.device).scatter_reduce_(
            0, self.bin_of, f.salience, reduce="amax", include_self=True
        )
        hit = f.salience >= best[self.bin_of] - 1e-9
        self.rep = torch.zeros(b, dtype=torch.long, device=self.device)
        self.rep[self.bin_of[hit]] = torch.nonzero(hit).squeeze(1)
        self.hist = torch.zeros(int(self.p["rows"]), b, device=self.device)
        self.next_t = None

    def _profile(self) -> torch.Tensor:
        v = torch.zeros(self.n_bins, device=self.device).index_add_(
            0, self.bin_of, self.dr.brightness()
        )
        return (v / self.cnt / max(self.p["gain"], 1e-3)).sqrt()

    def step(self, dt: float) -> None:
        t = self.dr.t
        every = self.p["every"]
        if self.next_t is None or t < self.next_t - 2 * every or t > self.next_t + 10 * every:
            self.next_t = t  # first frame, a seek or a wrap
        while t >= self.next_t:
            self.hist = torch.roll(self.hist, 1, 0)
            self.hist[0] = self._profile()
            self.next_t += every

    def render(self) -> torch.Tensor:
        p, dr = self.p, self.dr
        h, w = self.height, self.width
        rows, b = self.hist.shape
        frac = 0.0
        if self.next_t is not None:
            frac = min(max(float((dr.t - (self.next_t - p["every"])) / p["every"]), 0.0), 1.0)
        # Every ridge at once: (rows, w) screen heights on a shared x grid.
        depth = torch.arange(rows, device=self.device, dtype=torch.float32) + frac
        s = 1.0 / (1.0 + depth * p["depth"])  # perspective scale per ridge
        y0 = h * 0.93 - depth * (0.62 * h / rows)
        half = 0.39 * w * s
        xs = torch.arange(w, device=self.device, dtype=torch.float32)
        u = (xs[None] - (w * 0.5 - half[:, None])) / (2 * half[:, None]) * (b - 1)
        span = (u >= 0) & (u <= b - 1)
        i0 = u.floor().long().clamp(0, b - 2)
        fu = u - i0
        val = self.hist.gather(1, i0) * (1 - fu) + self.hist.gather(1, i0 + 1) * fu
        y = y0[:, None] - val * p["height"] * 0.18 * h * s[:, None]
        y = torch.where(span, y, torch.full_like(y, float(h)))
        # Occlusion: a ridge shows only above every nearer ridge -- the running
        # minimum of the screen height over the ridges in front of it.
        nearer = torch.cummin(y, dim=0).values
        horizon = torch.cat([torch.full((1, w), float(h), device=self.device), nearer[:-1]], 0)
        vis = span & (y < horizon) & (y0[:, None] > 0)
        seg = vis[:, 1:] & vis[:, :-1]
        fade = ((1.0 - depth / rows).clamp_min(0) ** 1.3)[:, None]
        if int(p["mono"]) == 1:
            col = torch.tensor([0.85, 0.88, 1.0], device=self.device).expand(rows, w, 3)
        else:
            col = self.bin_rgb[i0]
        rgb = col * ((0.35 + val.clamp(max=2.0)) * fade * 0.5)[..., None]
        r_idx, x_idx = torch.nonzero(seg, as_tuple=True)
        frame = torch.zeros(h * w, 3, device=self.device)
        splat_lines(
            frame,
            x_idx.float(),
            y[r_idx, x_idx],
            x_idx.float() + 1,
            y[r_idx, x_idx + 1],
            rgb[r_idx, x_idx],
            w,
            h,
            3,
        )
        centres = (torch.arange(b, device=self.device).float() / (b - 1)) * 2 * float(half[0]) + (
            w * 0.5 - float(half[0])
        )
        self.remember(centres, torch.full_like(centres, float(y0[0])), self.rep)
        return dr.finish(frame * 3.0, h, w)
