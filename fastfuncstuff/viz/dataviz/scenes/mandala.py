"""Mandala: families as concentric rings, mirrored into N-fold symmetry.

Each ICA family is a ring (inner to outer by the family's height in the head),
and within a ring a voxel sits at its anatomical azimuth around the head's
vertical axis -- so neighbours stay neighbours and a response that moves
around the brain moves around its ring. That wedge of data is mirrored into
``fold`` sectors, the kaleidoscope.

The data moves it: a ring spins while its family is active (the integral of
its time course), sprouts ``fold``-fold petals in proportion to its activity,
voxels push outward as they rise, the whole figure breathes with the global
signal, and a DVARS spike scatters it.
"""

from __future__ import annotations

import math

import torch

from ..drives import Param, splat
from .base import Scene

S = "mandala"


class MandalaScene(Scene):
    name = S
    blurb = "families as rings in N-fold symmetry; spin, petals and breath from the data"
    PARAMS = (
        Param("fold", 6, 2, 12, S, "symmetry order (mirrored sectors)", scene=S),
        Param("spin", 0.6, 0.0, 3.0, S, "ring rotation per unit of family activity", scene=S),
        Param("petal", 0.25, 0.0, 1.0, S, "petal depth per unit of family activity", scene=S),
        Param("push", 0.06, 0.0, 0.3, S, "outward push of a voxel as it rises", scene=S),
        Param("breath", 0.6, 0.0, 2.0, S, "whole-figure breathing with the global signal", scene=S),
        Param("scatter", 0.5, 0.0, 3.0, S, "scatter on DVARS spikes", scene=S),
        Param(
            "rest",
            0.25,
            0.0,
            1.5,
            S,
            "faint glow of every voxel, so the rings show at rest",
            scene=S,
        ),
        Param("ring_width", 0.5, 0.05, 1.0, S, "how much of its band each ring fills", scene=S),
        Param("mzoom", 1.0, 0.3, 3.0, S, "zoom", scene=S),
    )

    def __init__(self, drives, width, height, seed=0):
        super().__init__(drives, width, height, seed)
        f = self.f
        n_fam = drives.n_fam
        order = torch.argsort(torch.argsort(f.family_center[:, 2]))  # rank by height
        self.ring = (order.float() + 0.5) / n_fam  # (F,) in (0, 1)
        a = f.anat
        azim = torch.atan2(a[:, 1], a[:, 0])
        self.u = (azim + math.pi) / (2 * math.pi)  # (N,) position along the ring in [0, 1)
        # Within the band: distance from the head's axis, so deep voxels sit
        # on the inner edge of their ring and cortex on the outer.
        rad = a[:, :2].norm(dim=1)
        self.depth = (rad / torch.quantile(rad, 0.98).clamp_min(1e-6)).clamp(0, 1)
        self.phi = torch.zeros(n_fam, device=self.device)
        self.kick = torch.zeros(f.n_voxels, device=self.device)

    def step(self, dt: float) -> None:
        p, dr = self.p, self.dr
        sign = torch.where(torch.arange(dr.n_fam, device=self.device) % 2 == 0, 1.0, -1.0)
        self.phi = self.phi + dr.fam * sign * p["spin"] * dt
        if p["scatter"] > 0 and dr.d > 2.5:
            self.kick = (
                self.kick + self.randn(self.f.n_voxels).abs() * (dr.d - 2.5) * p["scatter"] * 0.03
            )
        self.kick = self.kick * math.exp(-3.0 * dt)

    def render(self) -> torch.Tensor:
        p, dr, f = self.p, self.dr, self.f
        h, w = self.height, self.width
        n_fold = int(p["fold"])
        fam = f.family
        lv = dr.live
        band = 1.0 / dr.n_fam
        act = dr.fam.clamp(-2, 3)[fam]

        # Angle inside one mirrored half-sector: the full ring of azimuths is
        # squeezed into pi/fold, then reflected and repeated 2*fold times.
        half = math.pi / n_fold
        theta0 = self.u * half + self.phi[fam] * half
        theta0 = torch.remainder(theta0, 2 * half)
        theta0 = torch.where(theta0 > half, 2 * half - theta0, theta0)

        r = self.ring[fam] + (self.depth - 0.5) * band * p["ring_width"]
        r = r + lv.clamp_min(0) * p["push"] * band * 4 + self.kick
        breathe = 1.0 + math.tanh(dr.g * 0.5) * p["breath"] * 0.06
        scale = 0.47 * min(w, h) * p["mzoom"] * breathe

        glow = dr.brightness() * self.density
        rest = p["rest"] * self.density * 0.6 * (0.3 + f.salience)
        rgb = dr.colors() * glow[:, None] + dr.family_rgb()[fam] * rest[:, None]
        rgb = rgb / (2 * n_fold) * 6

        # Petals: the radius swells fold-fold around the circle in proportion
        # to the family's activity. cos(fold * theta) is the same in every
        # mirrored copy, which is what lets the mirroring happen in image space.
        petal = p["petal"] * act.clamp_min(0) * 0.25
        rr = r * (1 + petal * torch.cos(n_fold * theta0) ** 2) * scale  # pixels

        # Splat one half-sector into a polar buffer (rows: radius in pixels,
        # columns: angle within the half-sector), keep the trails there, then
        # unfold to the full kaleidoscope with one grid_sample. The cost scales
        # with pixels instead of points x 2*fold copies -- 12x less on the CPU.
        n_r, n_th = self._polar_shape(half)
        dth = half / n_th
        # A polar bin covers r * dtheta * dr Cartesian pixels: weight by its
        # inverse so brightness matches drawing the points directly.
        wt = 1.0 / (rr.clamp_min(1.0) * dth)
        frame = torch.zeros(n_r * n_th, 3, device=self.device)
        splat(frame, theta0 / dth, rr, rgb * wt[:, None], n_th, n_r)
        if self.accum.shape[0] != n_r * n_th:
            self.accum = torch.zeros(n_r * n_th, 3, device=self.device)
        self.fade_in(frame)
        return dr.finish(self._unfold(half, n_r, n_th), h, w)

    def _polar_shape(self, half: float) -> tuple[int, int]:
        n_r = int(0.75 * max(self.width, self.height)) + 2
        n_th = max(32, int(n_r * half))
        return n_r, n_th

    def _unfold(self, half: float, n_r: int, n_th: int) -> torch.Tensor:
        """(h*w, 3) Cartesian image from the polar half-sector buffer."""
        import torch.nn.functional as F

        h, w = self.height, self.width
        key = (h, w, half, n_r, n_th)
        if getattr(self, "_grid_key", None) != key:
            ys = torch.arange(h, device=self.device, dtype=torch.float32) - h * 0.5 + 0.5
            xs = torch.arange(w, device=self.device, dtype=torch.float32) - w * 0.5 + 0.5
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            rad = (gx**2 + gy**2).sqrt()
            th = torch.remainder(torch.atan2(gy, gx), 2 * half)
            th = torch.where(th > half, 2 * half - th, th)  # the mirror
            # grid_sample wants [-1, 1] across each axis of the source.
            u = th / half * 2 - 1
            v = rad / (n_r - 1) * 2 - 1
            self._grid = torch.stack([u, v], -1)[None]
            self._grid_key = key
        src = self.accum.T.reshape(1, 3, n_r, n_th)
        out = F.grid_sample(
            src, self._grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        return out[0].reshape(3, -1).T
