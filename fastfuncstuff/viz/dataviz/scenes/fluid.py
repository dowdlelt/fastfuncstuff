"""Fluid: stable-fluids ink filling the screen, one jet per family.

Jos Stam's stable fluids on a grid that fills the frame: semi-Lagrangian
advection, a Jacobi pressure projection that keeps the flow incompressible,
and vorticity confinement to keep the curls alive. Every ICA family is a jet
placed where the family sits on the weave. The jet points along the
family's Hilbert phase, so it turns as the family's oscillation cycles; it
pushes with the family's envelope and pours ink in its colour while the family
is active. When families fall into step their jets turn together and the ink
organises; out of step, it churns. The global signal lifts the whole fluid, and
a DVARS spike is a burst from the centre.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..drives import Param
from .base import Scene, family_ring_sites, representative

S = "fluid"


class FluidScene(Scene):
    name = S
    blurb = "full-screen stable fluids: one ink jet per family, turning with its phase"
    PARAMS = (
        Param("thrust", 1.0, 0.0, 5.0, S, "jet force per unit of family envelope", scene=S),
        Param("pour_f", 1.0, 0.0, 4.0, S, "ink poured per unit of family activity", scene=S),
        Param("curl", 0.6, 0.0, 3.0, S, "vorticity confinement (keeps swirls alive)", scene=S),
        Param("fade_f", 0.993, 0.9, 0.999, S, "ink kept per frame", scene=S),
        Param("jet", 6.0, 2.0, 20.0, S, "jet radius, cells", scene=S),
        Param("fcell", 2, 1, 4, S, "pixels per cell", scene=S),
    )

    def __init__(self, drives, width, height, seed=0):
        super().__init__(drives, width, height, seed)
        self._alloc()

    def on_resize(self):
        self._alloc()

    def on_param(self, name, value):
        if name in ("fcell", "families"):
            self._alloc()

    def _alloc(self) -> None:
        c = int(self.p["fcell"])
        self.gh, self.gw = max(32, self.height // c), max(32, self.width // c)
        self.vel = torch.zeros(2, self.gh, self.gw, device=self.device)
        self.dye = torch.zeros(3, self.gh, self.gw, device=self.device)
        ys, xs = torch.meshgrid(
            torch.arange(self.gh, device=self.device, dtype=torch.float32),
            torch.arange(self.gw, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        self.xs, self.ys = xs, ys
        site = family_ring_sites(self)
        # Keep jets off the very edge so their ink has room.
        self.site = torch.stack([site[:, 0] * self.gw, site[:, 1] * self.gh], 1)
        self.rep = representative(self)
        self._jets_r = None
        self.k_lap = torch.tensor(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]], device=self.device, dtype=torch.float32
        )[None, None]

    def _advect(self, field: torch.Tensor, dt: float) -> torch.Tensor:
        x = self.xs - dt * self.vel[0]
        y = self.ys - dt * self.vel[1]
        grid = torch.stack([x / (self.gw - 1) * 2 - 1, y / (self.gh - 1) * 2 - 1], -1)[None]
        return F.grid_sample(
            field[None], grid, mode="bilinear", padding_mode="border", align_corners=True
        )[0]

    def _ddx(self, a: torch.Tensor) -> torch.Tensor:
        return (torch.roll(a, -1, -1) - torch.roll(a, 1, -1)) * 0.5

    def _ddy(self, a: torch.Tensor) -> torch.Tensor:
        return (torch.roll(a, -1, -2) - torch.roll(a, 1, -2)) * 0.5

    def step(self, dt: float) -> None:
        p, dr = self.p, self.dr
        sal = dr.family_salience / dr.family_salience.max().clamp_min(1e-6)
        act = dr.fam.clamp(0, 3)
        push = dr.fam_amp * (0.3 + act) * sal * p["thrust"] * 25.0
        pour = (0.15 + act) * sal * p["pour_f"] * 6.0
        if self._jets_r != p["jet"]:
            # (F, gh, gw) jet footprints, built once per radius: the per-family
            # loop read three floats per family off the GPU every frame.
            r2 = p["jet"] ** 2
            dx = self.xs[None] - self.site[:, 0, None, None]
            dy = self.ys[None] - self.site[:, 1, None, None]
            self.jets = torch.exp(-(dx**2 + dy**2) / r2)
            self._jets_r = p["jet"]
        fx = push * torch.cos(dr.fam_phase)
        fy = push * torch.sin(dr.fam_phase)
        self.vel[0] += dt * torch.einsum("f,fhw->hw", fx, self.jets)
        self.vel[1] += dt * torch.einsum("f,fhw->hw", fy, self.jets)
        self.dye += dt * torch.einsum("f,fc,fhw->chw", pour, dr.family_rgb(), self.jets)
        self.vel[1] -= dt * 4.0 * math.tanh(dr.g * 0.5)  # the global signal lifts
        if dr.d > 2.5:
            cx, cy = self.gw / 2, self.gh / 2
            rx, ry = self.xs - cx, self.ys - cy
            rr = (rx**2 + ry**2).sqrt().clamp_min(1.0)
            burst = (dr.d - 2.5) * 30.0 * torch.exp(-rr / (0.2 * self.gw))
            self.vel[0] += burst * rx / rr
            self.vel[1] += burst * ry / rr
            self.dye += 0.05 * (dr.d - 2.5) * torch.exp(-rr / 6.0)[None]
        if p["curl"] > 0:
            w = self._ddx(self.vel[1]) - self._ddy(self.vel[0])
            gx, gy = self._ddx(w.abs()), self._ddy(w.abs())
            n = (gx**2 + gy**2).sqrt().clamp_min(1e-5)
            self.vel[0] += dt * p["curl"] * 8.0 * (gy / n) * w
            self.vel[1] -= dt * p["curl"] * 8.0 * (gx / n) * w
        self.vel = self._advect(self.vel, dt)
        self.dye = self._advect(self.dye, dt)
        # Incompressibility: solve lap(pressure) = div with Jacobi, subtract grad.
        div = self._ddx(self.vel[0]) + self._ddy(self.vel[1])
        pr = torch.zeros_like(div)[None, None]
        d4 = div[None, None]
        for _ in range(20):
            pr = (F.conv2d(F.pad(pr, (1, 1, 1, 1), mode="replicate"), self.k_lap) - d4) * 0.25
        pr = pr[0, 0]
        self.vel[0] -= self._ddx(pr)
        self.vel[1] -= self._ddy(pr)
        self.vel *= 0.995
        self.dye *= p["fade_f"] ** (dt * 60)

    def render(self) -> torch.Tensor:
        img = F.interpolate(
            self.dye[None], size=(self.height, self.width), mode="bilinear", align_corners=False
        )[0]
        cw, ch = self.width / self.gw, self.height / self.gh
        self.remember(self.site[:, 0] * cw, self.site[:, 1] * ch, self.rep)
        return self.dr.finish(img.reshape(3, -1).T * 2.0, self.height, self.width)
