"""Lenia: a continuous cellular automaton whose creatures live where the data does.

Lenia (Chan 2019) is the Game of Life made smooth: a field A in [0, 1], a ring
kernel K of radius R, and a growth curve G(u) = 2 exp(-(u - mu)^2 / 2 sigma^2)
- 1 applied to K * A each step. In a narrow band of (mu, sigma) it grows
gliding, dividing, breathing organisms that look alive.

Here the band is opened locally by the data: sigma rises with the live
activity under each cell. Quiet tissue sits below the band, where any pattern
starves, and active cortex rises into it, so living patterns appear only
where the data is active, grow while it stays active and die back as the
activity moves on. (Measured: at sigma 0.014 a random soup dies; at 0.024 it
fills with labyrinth.)
The global signal nudges mu for the whole field; a DVARS spike sows noise.
The field is coloured by the family that dominates each cell and lit with a
relief from its gradient. Convolution is by FFT, so the kernel radius costs
nothing.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..drives import Param
from .base import Scene, axial_cells, blur, disks, filled_field

S = "lenia"


class LeniaScene(Scene):
    name = S
    blurb = "continuous automaton: organisms grow and divide where cortex is active"
    PARAMS = (
        Param("mu", 0.15, 0.05, 0.35, S, "growth centre", scene=S),
        Param(
            "sigma",
            0.014,
            0.005,
            0.05,
            S,
            "growth width where quiet (below ~0.015 patterns starve)",
            scene=S,
        ),
        Param("fertile", 0.012, 0.0, 0.03, S, "extra growth width where active", scene=S),
        Param("radius", 10, 4, 24, S, "kernel radius, cells", scene=S),
        Param("lstep", 0.1, 0.02, 0.5, S, "time step", scene=S),
        Param("ncell", 2, 1, 6, S, "pixels per cell", scene=S),
        Param(
            "sow_l",
            1.0,
            0.0,
            4.0,
            S,
            "noise sown on DVARS spikes and into empty active tissue",
            scene=S,
        ),
    )

    def __init__(self, drives, width, height, seed=0):
        super().__init__(drives, width, height, seed)
        self._alloc()

    def on_resize(self):
        self._alloc()

    def on_param(self, name, value):
        if name in ("ncell", "radius"):
            self._alloc()

    def _alloc(self) -> None:
        f, dr = self.f, self.dr
        c = int(self.p["ncell"])
        self.gh, self.gw = max(32, self.height // c), max(32, self.width // c)
        self.cell_of, self.inside, self.passes = axial_cells(f.anat, self.gh, self.gw, dr.spacing)
        onehot = torch.zeros(f.n_voxels, dr.n_fam, device=self.device)
        onehot[torch.arange(f.n_voxels, device=self.device), f.family] = f.family_weight.clamp_min(
            0.05
        )
        terr = filled_field(onehot, self.cell_of, self.gh, self.gw, self.passes + 2)
        self.color = blur(dr.family_rgb()[terr.argmax(0)].permute(2, 0, 1), 1)
        # Ring kernel exp(4 - 1/(r(1-r))) on r in (0, 1), centred at (0, 0) for the FFT.
        R = float(self.p["radius"])
        ys = torch.fft.fftfreq(self.gh, 1.0 / self.gh, device=self.device)
        xs = torch.fft.fftfreq(self.gw, 1.0 / self.gw, device=self.device)
        r = (ys[:, None] ** 2 + xs[None, :] ** 2).sqrt() / R
        ring = torch.where(
            (r > 0) & (r < 1), torch.exp(4 - 1 / (r * (1 - r)).clamp_min(1e-6)), torch.zeros_like(r)
        )
        self.kf = torch.fft.rfft2(ring / ring.sum())
        self.A = torch.zeros(self.gh, self.gw, device=self.device)
        self._sow(torch.full_like(self.A, 0.02) * self.inside)

    def _sow(self, prob: torch.Tensor) -> None:
        """Drop soft noise blobs (radius ~ R/2) with the given per-cell probability."""
        spots = self.rand(self.gh, self.gw) < prob * 0.02
        blobs = disks(spots, max(1, int(self.p["radius"]) // 2))
        noise = self.rand(self.gh, self.gw)
        self.A = torch.where(blobs, torch.maximum(self.A, noise), self.A)

    def step(self, dt: float) -> None:
        p, dr = self.p, self.dr
        act = filled_field(dr.brightness(), self.cell_of, self.gh, self.gw, self.passes + 1)
        act = self.relative(act * self.inside)
        # Cubed: the smooth field is "somewhat active" over most of the head,
        # and only the genuinely active places should reach the fertile band.
        sigma = p["sigma"] + p["fertile"] * act**3
        mu = p["mu"] + 0.01 * math.tanh(dr.g * 0.5)
        for _ in range(2):
            u = torch.fft.irfft2(torch.fft.rfft2(self.A) * self.kf, s=self.A.shape)
            growth = 2 * torch.exp(-((u - mu) ** 2) / (2 * sigma**2)) - 1
            self.A = (self.A + p["lstep"] * growth).clamp(0, 1) * self.inside
        # Keep active tissue from going barren: sow where active and empty.
        empty_active = act * (self.A < 0.05).float() * self.inside
        sow = empty_active * 0.3 * p["sow_l"]
        if dr.d > 2.5:
            sow = sow + (dr.d - 2.5) * 0.05 * p["sow_l"] * self.inside
        self._sow(sow)

    def render(self) -> torch.Tensor:
        dr = self.dr
        A = self.A
        gx = torch.zeros_like(A)
        gy = torch.zeros_like(A)
        gx[:, 1:-1] = A[:, 2:] - A[:, :-2]
        gy[1:-1] = A[2:] - A[:-2]
        shade = (1.0 + 2.5 * (-gx - gy)).clamp(0.4, 1.8)
        img = self.color * (A**0.8) * shade + 0.02 * self.color * self.inside
        img = F.interpolate(img[None], size=(self.height, self.width), mode="nearest")[0]
        cw, ch = self.width / self.gw, self.height / self.gh
        cell = self.cell_of
        self.remember(
            ((cell % self.gw).float() + 0.5) * cw,
            ((cell // self.gw).float() + 0.5) * ch,
            weight=dr.brightness(),
        )
        return dr.finish(img.reshape(3, -1).T * 1.5, self.height, self.width)
