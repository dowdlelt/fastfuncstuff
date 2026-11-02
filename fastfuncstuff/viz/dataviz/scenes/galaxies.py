"""Galaxies: every ICA family is a spiral galaxy, and they collide as the data says.

Each family is a disc of its member voxels: the strongest members in the core,
weaker ones out in the arms, arranged by anatomical azimuth and wound into
spiral arms. Stars orbit their own galaxy's centre under Plummer gravity.

The galaxies feel each other with a coupling read off the data. Over a sliding
window, two families whose time courses correlate attract and two that
anticorrelate repel. A family's mass swells while it is active, so a network
that switches on pulls its partners in. The collisions are therefore the data's
doing, and tidal tails and bridges between galaxies are real gravitational
dynamics (each star feels every galaxy centre: the restricted N-body problem
that makes the Antennae).

Stripped stars slowly relax back toward circular orbits in their own galaxy,
so the system keeps colliding instead of dissolving. A DVARS spike is a
supernova -- every star kicked at random. The global signal breathes G.
"""

from __future__ import annotations

import math

import torch

from ..drives import Param, splat
from .base import Scene

S = "galaxies"


def windowed_corr(tcs: torch.Tensor, window: int) -> torch.Tensor:
    """(T, F, F) correlation of the family time courses in a window centred on each TR."""
    n_time, n_f = tcs.shape
    out = torch.empty(n_time, n_f, n_f, device=tcs.device)
    half = max(2, window // 2)
    for t in range(n_time):
        seg = tcs[max(0, t - half) : min(n_time, t + half + 1)]
        seg = seg - seg.mean(0)
        seg = seg / seg.norm(dim=0).clamp_min(1e-6)
        out[t] = seg.T @ seg
    return out


class GalaxiesScene(Scene):
    name = S
    blurb = "families as spiral galaxies; correlation is gravity, activity is mass"
    PARAMS = (
        Param("gravity", 1.0, 0.1, 4.0, S, "overall gravitational constant", scene=S),
        Param(
            "coupling",
            1.5,
            0.0,
            4.0,
            S,
            "attraction per unit of windowed correlation (negative repels)",
            scene=S,
        ),
        Param("window", 30, 6, 120, S, "correlation window, TRs", scene=S),
        Param("heft", 1.5, 0.0, 5.0, S, "mass gained per unit of family activity", scene=S),
        Param(
            "tide", 0.35, 0.0, 2.0, S, "how fast stripped stars relax back into their disc", scene=S
        ),
        Param("winding", 2.5, 0.0, 6.0, S, "spiral-arm winding", scene=S),
        Param("nova", 1.0, 0.0, 4.0, S, "random kick to every star on DVARS spikes", scene=S),
        Param("tempo", 1.0, 0.1, 4.0, S, "simulation speed", scene=S),
        Param(
            "glow",
            0.35,
            0.0,
            1.5,
            S,
            "faint glow of every star, so the galaxies show at rest",
            scene=S,
        ),
        Param("gzoom", 1.0, 0.3, 3.0, S, "zoom", scene=S),
    )

    def __init__(self, drives, width, height, seed=0):
        super().__init__(drives, width, height, seed)
        f = self.f
        n_f = drives.n_fam
        dev = self.device
        size = drives.family_size
        self.base_mass = (size / size.max()).sqrt().clamp_min(0.15)
        self.radius = 0.14 + 0.18 * self.base_mass
        self.spin = torch.where(torch.arange(n_f, device=dev) % 2 == 0, 1.0, -1.0)
        ang = torch.arange(n_f, device=dev, dtype=torch.float32) / n_f * 2 * math.pi
        self.c = torch.stack([torch.cos(ang), torch.sin(ang)], 1) * 0.75
        # Start the galaxies circling the centre so they fall past each other.
        self.u = torch.stack([-torch.sin(ang), torch.cos(ang)], 1) * 0.35
        # Membership rank inside each family: 1 = strongest member -> the core.
        q = torch.zeros(f.n_voxels, device=dev)
        for k in range(n_f):
            m = torch.nonzero(f.family == k).squeeze(1)
            if m.numel():
                order = torch.argsort(torch.argsort(f.family_weight[m]))
                q[m] = order.float() / max(1, m.numel() - 1)
        self.q = q
        self.az = torch.atan2(f.anat[:, 1], f.anat[:, 0])
        self._corr_key = None
        self._spawn(torch.ones(f.n_voxels, dtype=torch.bool, device=dev))

    def _disc(self, which: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Disc positions and orbital velocities for the stars selected by ``which``."""
        f = self.f
        fam = f.family[which]
        rad = self.radius[fam] * ((1 - self.q[which]) ** 0.8 + 0.03)
        th = self.az[which] + self.p["winding"] * torch.log(rad / self.radius[fam] + 0.05)
        rel = torch.stack([torch.cos(th), torch.sin(th)], 1) * rad[:, None]
        return self.c[fam] + rel, self.u[fam] + self._circular(rel, fam)

    def _spawn(self, which: torch.Tensor) -> None:
        if not hasattr(self, "pos"):
            self.pos = torch.zeros(self.f.n_voxels, 2, device=self.device)
            self.vel = torch.zeros(self.f.n_voxels, 2, device=self.device)
        self.pos[which], self.vel[which] = self._disc(which)

    def _circular(self, rel: torch.Tensor, fam: torch.Tensor) -> torch.Tensor:
        """Circular-orbit velocity about the own galaxy's centre (Plummer)."""
        r2 = rel[:, 0] ** 2 + rel[:, 1] ** 2
        eps2 = (0.4 * self.radius[fam]) ** 2
        g = (
            self.p["gravity"] * self.mass[fam]
            if hasattr(self, "mass")
            else self.p["gravity"] * self.base_mass[fam]
        )
        speed = (g * r2 / ((r2 + eps2) * (r2 + eps2).sqrt())).sqrt()
        perp = torch.stack([-rel[:, 1], rel[:, 0]], 1) / r2.sqrt().clamp_min(1e-6)[:, None]
        return perp * (speed * self.spin[fam])[:, None]

    def _corr(self) -> torch.Tensor:
        key = int(self.p["window"])
        if key != self._corr_key:
            self._corr_all = windowed_corr(self.f.family_tcs.float(), key)
            self._corr_key = key
        return self._corr_all[int(self.dr.t) % self.f.n_time]

    def step(self, dt: float) -> None:
        p, dr, f = self.p, self.dr, self.f
        h = dt * p["tempo"] * 0.5
        breath = 1.0 + 0.15 * math.tanh(dr.g * 0.5)
        G = p["gravity"] * breath
        self.mass = self.base_mass * (1 + p["heft"] * dr.fam.clamp(0, 3))
        # Baseline slight repulsion: only correlated families seek each other.
        k = -0.15 + p["coupling"] * self._corr()  # (F, F)
        k.fill_diagonal_(0)
        fam = f.family
        for _ in range(2):
            # Galaxies: pairwise softened gravity scaled by the data's coupling,
            # a gentle pull to the middle so they keep coming back, a little drag.
            d = self.c[None, :, :] - self.c[:, None, :]  # (F, F, 2): a -> b
            # Softened by the two galaxies' sizes: extended bodies, not points.
            soft = self.radius[:, None] + self.radius[None, :]
            r2 = d.square().sum(-1) + soft**2
            acc_c = (
                G * self.mass[None, :, None] * k[:, :, None] * d / (r2 * r2.sqrt())[:, :, None]
            ).sum(1)
            # A soft core: galaxies pass through and tear at each other but do
            # not merge into one blob for good.
            core = (self.radius[:, None] + self.radius[None, :]) * 0.8
            dist = d.norm(dim=-1).clamp_min(1e-4)
            push = ((core - dist) / core).clamp_min(0) ** 2 * 25.0
            push.fill_diagonal_(0)
            acc_c = acc_c - (push[:, :, None] * d / dist[:, :, None]).sum(1)
            acc_c = acc_c - self.c * 0.6 - self.u * 0.15
            self.u = self.u + acc_c * h
            speed = self.u.norm(dim=1, keepdim=True)
            self.u = self.u * (1.5 / speed.clamp_min(1.5))
            self.c = self.c + self.u * h

            # Stars: every galaxy's pull (restricted N-body: tides for free).
            acc = torch.zeros_like(self.pos)
            for g in range(dr.n_fam):
                # Per component: a reduction over a size-2 axis is slow on the CPU.
                rx, ry = self.c[g, 0] - self.pos[:, 0], self.c[g, 1] - self.pos[:, 1]
                r2s = rx * rx + ry * ry + (0.4 * self.radius[g]) ** 2
                pull = G * self.mass[g] / (r2s * r2s.sqrt())
                acc = acc + torch.stack([rx * pull, ry * pull], 1)
            self.vel = self.vel + acc * h
            # Near a core the softened pull is huge; cap speed so a close pass
            # tears a tail instead of firing stars off the screen.
            sp = self.vel.norm(dim=1, keepdim=True)
            self.vel = self.vel * (2.5 / sp.clamp_min(2.5))
            if p["tide"] > 0:
                target = self.u[fam] + self._circular(self.pos - self.c[fam], fam)
                self.vel = self.vel + (target - self.vel) * min(1.0, p["tide"] * h)
            self.pos = self.pos + self.vel * h

        if p["nova"] > 0 and dr.d > 2.5:
            self.vel = self.vel + self.randn(f.n_voxels, 2) * (dr.d - 2.5) * p["nova"] * 0.03
        # Strays that wander off for good are reborn in their disc.
        # Strays reborn in their disc, by where over every star (no mask, no
        # sync) -- every 15 frames, since it recomputes every star's disc place.
        self._tick = getattr(self, "_tick", 0) + 1
        if self._tick % 15 == 0:
            lost = ((self.pos - self.c[fam]).norm(dim=1) > 1.2)[:, None]
            home_pos, home_vel = self._disc(slice(None))
            self.pos = torch.where(lost, home_pos, self.pos)
            self.vel = torch.where(lost, home_vel, self.vel)

    def render(self) -> torch.Tensor:
        p, dr, f = self.p, self.dr, self.f
        h, w = self.height, self.width
        wts = self.mass / self.mass.sum()
        centre = (self.c * wts[:, None]).sum(0)
        # The camera follows the system and frames its spread, smoothly.
        # Camera fit kept on the device: reading the spread back cost a sync.
        reach = ((self.c - centre).norm(dim=1).max() + self.radius.max() * 1.5).clamp_min(0.6)
        if not hasattr(self, "_cam"):
            self._cam, self._reach = centre.clone(), reach
        self._cam = self._cam + (centre - self._cam) * 0.04
        self._reach = self._reach + (reach - self._reach) * 0.02
        scale = 0.48 * min(w, h) * p["gzoom"] / self._reach
        sx = w * 0.5 + (self.pos[:, 0] - self._cam[0]) * scale
        sy = h * 0.5 + (self.pos[:, 1] - self._cam[1]) * scale
        self.remember(sx, sy)
        glow = dr.brightness() * self.density
        rest = p["glow"] * self.density * (0.25 + self.q**2 * 1.5)
        rgb = dr.colors() * glow[:, None] + dr.family_rgb()[f.family] * rest[:, None]
        frame = torch.zeros(h * w, 3, device=self.device)
        splat(frame, sx, sy, rgb, w, h)
        self.fade_in(frame)
        return dr.finish(self.accum, h, w)
