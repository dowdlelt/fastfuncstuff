"""Flow: ink in water. Voxels in correlation space, stirred by their families.

Each voxel starts at its place in the 2-D correlation-space layout, where
correlated voxels sit together. Each ICA family owns a vortex at its members'
centroid; the vortex winds up with the family's time course (signed, so a
falling network turns back) and its reach scales with the family's spread.

The motion is a twist map, not forces: each vortex rotates points about its
centre by an angle that depends only on distance from it, and the angles relax
back to zero, so ink untwists home when its family goes quiet. A twist
preserves area exactly. The force version (vortex plus a spring home) did not:
its equilibria all shrink toward the vortex centre by tether/sqrt(tether^2 +
spin^2), which drew as a white core swallowing the picture. A divergence-free
curl offset scaled by frame-to-frame unpredictability adds eddies. Long trails
turn the paths into ink.
"""

from __future__ import annotations

import math

import torch

from ..drives import Param, splat, uniform_disc
from .base import Scene

S = "flow"
HOMES = ("axial", "function")


class FlowScene(Scene):
    name = S
    blurb = "ink: correlation space stirred by one vortex per family"
    PARAMS = (
        Param(
            "stir",
            1.5,
            0.0,
            6.0,
            S,
            "twist (radians per second) per unit of family activity",
            scene=S,
        ),
        Param(
            "tether", 0.6, 0.0, 5.0, S, "how fast a twist relaxes back home (per second)", scene=S
        ),
        Param("eddies", 0.3, 0.0, 3.0, S, "curl noise scaled by unpredictability", scene=S),
        Param(
            "ink", 0.12, 0.0, 1.5, S, "faint glow of every voxel, so the ink shows at rest", scene=S
        ),
        Param("ink_trails", 0.95, 0.5, 0.995, S, "trail persistence (ink length)", scene=S),
        Param("fzoom", 1.0, 0.3, 3.0, S, "zoom", scene=S),
        Param(
            "home",
            0,
            0,
            len(HOMES) - 1,
            S,
            "where ink settles: the head from above, or correlation space",
            HOMES,
            scene=S,
        ),
    )

    def __init__(self, drives, width, height, seed=0):
        super().__init__(drives, width, height, seed)
        f = self.f
        self.homes = {
            # Anterior up; the head seen from above.
            "axial": f.anat[:, :2] * torch.tensor([1.0, -1.0], device=self.device),
            "function": uniform_disc(f.func[:, :2]),
        }
        self._set_home()
        self.pos = self.home.clone()

    def on_param(self, name, value):
        if name == "home":
            self._set_home()

    def _set_home(self) -> None:
        f = self.f
        self.home = self.homes[HOMES[int(self.p["home"])]]
        n_fam = self.dr.n_fam
        w = f.family_weight.clamp_min(0.05)
        cnt = torch.zeros(n_fam, device=self.device).index_add_(0, f.family, w)
        c = torch.zeros(n_fam, 2, device=self.device).index_add_(
            0, f.family, self.home * w[:, None]
        )
        self.centre = c / cnt.clamp_min(1e-6)[:, None]
        d2 = (self.home - self.centre[f.family]).square().sum(1)
        spread = torch.zeros(n_fam, device=self.device).index_add_(0, f.family, d2 * w)
        self.sigma2 = (spread / cnt.clamp_min(1e-6)).clamp(0.005, 0.3) * 0.5

    def step(self, dt: float) -> None:
        p, dr = self.p, self.dr
        if not hasattr(self, "phi") or self.phi.numel() != dr.n_fam:
            self.phi = torch.zeros(dr.n_fam, device=self.device)
        sal = dr.family_salience / dr.family_salience.max().clamp_min(1e-6)
        self.phi = (self.phi + dr.fam.clamp(-3, 3) * sal * p["stir"] * dt) * math.exp(
            -p["tether"] * dt
        )
        x = self.home.clone()
        for k in range(dr.n_fam):
            rel = x - self.centre[k]
            ang = self.phi[k] * torch.exp(-rel.square().sum(1) / self.sigma2[k])
            c, s = torch.cos(ang), torch.sin(ang)
            x = self.centre[k] + torch.stack(
                [rel[:, 0] * c - rel[:, 1] * s, rel[:, 0] * s + rel[:, 1] * c], 1
            )
        if p["eddies"] > 0:
            # Each component depends only on the other coordinate: divergence-free.
            q = x * 6.0
            wt = dr.wall * 0.5
            curl = torch.stack(
                [
                    torch.sin(q[:, 1] + wt) + torch.cos(q[:, 1] * 0.7 - wt),
                    torch.sin(q[:, 0] - wt) + torch.cos(q[:, 0] * 1.3 + wt),
                ],
                1,
            )
            x = x + curl * (dr.innovation * p["eddies"] * 0.02)[:, None]
        self.pos = x

    def render(self) -> torch.Tensor:
        p, dr, f = self.p, self.dr, self.f
        h, w = self.height, self.width
        scale = 0.45 * min(w, h) * p["fzoom"]
        glow = dr.brightness() * self.density
        rest = p["ink"] * self.density * 0.6 * (0.2 + f.salience)
        rgb = dr.colors() * glow[:, None] + dr.family_rgb()[f.family] * rest[:, None]
        frame = torch.zeros(h * w, 3, device=self.device)
        sx, sy = w * 0.5 + self.pos[:, 0] * scale, h * 0.5 + self.pos[:, 1] * scale
        self.remember(sx, sy)
        splat(frame, sx, sy, rgb, w, h)
        tr = p["ink_trails"]
        self.accum.mul_(tr).add_(frame * (1 - tr) * 4.0)
        return dr.finish(self.accum, h, w)
