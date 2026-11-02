"""Interference: every family a source of ripples; the screen is their sum.

Each ICA family emits circular waves from where it sits on the weave. A
wave's amplitude is the family's envelope (weighted by how salient its
members are), its phase is the family's Hilbert phase, and each family has its
own wavelength. The screen shows the superposition. When families are out of
step the pattern is a shifting moire; when they synchronise, the fringes lock
into a standing figure, so synchrony across the brain is something you can
see. In op-art mode the sum is cut at zero into black and white bands, like
Bridget Riley; otherwise each family's ripples carry its colour.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..drives import Param
from .base import Scene, family_ring_sites, representative

S = "interference"


class InterferenceScene(Scene):
    name = S
    blurb = "ripples from every family; synchrony locks the moire into a standing figure"
    PARAMS = (
        Param("wavelength", 0.08, 0.02, 0.4, S, "base wavelength, fraction of the screen", scene=S),
        Param("ripple", 0.4, 0.0, 3.0, S, "outward ripple speed", scene=S),
        Param("op", 1, 0, 1, S, "op-art: cut the sum into black and white bands", scene=S),
        Param("bands_i", 1.0, 0.2, 4.0, S, "contrast of the fringes", scene=S),
        Param("icell", 2, 1, 4, S, "pixels per cell", scene=S),
    )

    def __init__(self, drives, width, height, seed=0):
        super().__init__(drives, width, height, seed)
        self._alloc()

    def on_resize(self):
        self._alloc()

    def on_param(self, name, value):
        if name in ("icell", "families"):
            self._alloc()

    def _alloc(self) -> None:
        c = int(self.p["icell"])
        self.gh, self.gw = max(32, self.height // c), max(32, self.width // c)
        site = family_ring_sites(self)
        ys, xs = torch.meshgrid(
            torch.linspace(0, 1, self.gh, device=self.device),
            torch.linspace(0, 1, self.gw, device=self.device) * (self.gw / self.gh),
            indexing="ij",
        )
        sx = site[:, 0] * (self.gw / self.gh)
        sy = site[:, 1]
        self.dist = (
            (xs[None] - sx[:, None, None]) ** 2 + (ys[None] - sy[:, None, None]) ** 2
        ).sqrt()
        # Each family its own wavelength, spread over an octave.
        n = self.dr.n_fam
        self.lam = 2 ** (torch.arange(n, device=self.device, dtype=torch.float32) / max(n - 1, 1))
        self.site = site
        self.rep = representative(self)

    def step(self, dt: float) -> None:
        pass

    def render(self) -> torch.Tensor:
        p, dr = self.p, self.dr
        sal = dr.family_salience / dr.family_salience.max().clamp_min(1e-6)
        amp = dr.fam_amp * (0.3 + dr.fam.clamp(0, 3)) * sal
        amp = amp / amp.sum().clamp_min(1e-6)
        k = 2 * math.pi / (p["wavelength"] * self.lam)
        arg = (
            k[:, None, None] * self.dist
            - p["ripple"] * dr.wall * 2 * math.pi
            + dr.fam_phase[:, None, None]
        )
        waves = torch.cos(arg) * amp[:, None, None]  # (F, gh, gw)
        total = waves.sum(0)
        if int(p["op"]) == 1:
            # Black and white bands, tinted by whichever family dominates there.
            band = (total > 0).float()
            tint = torch.einsum("fc,fhw->chw", dr.family_rgb(), waves.abs())
            tint = tint / tint.amax(0, keepdim=True).clamp_min(1e-6)
            img = band[None] * (0.55 + 0.45 * tint)
        else:
            lit = (0.5 + 0.5 * torch.cos(arg)) ** (2 * p["bands_i"])
            img = torch.einsum("fc,fhw->chw", dr.family_rgb(), lit * amp[:, None, None]) * 1.1
        img = F.interpolate(img[None], size=(self.height, self.width), mode="nearest")[0]
        self.remember(self.site[:, 0] * self.width, self.site[:, 1] * self.height, self.rep)
        return dr.finish(img.reshape(3, -1).T * (1.0 + dr.sync), self.height, self.width)
