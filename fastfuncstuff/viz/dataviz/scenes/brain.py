"""The anatomical swarm: the brain as a rotating 3-D cloud.

Two kinds of particle share one state array:

* **dust** -- one per voxel, on its anatomical home, lit by the mean image's
  local contrast: the scaffold the eye uses to know where things are.
* **sparks** -- lit by the voxel's live value and pushed around by the data:
  they bloom outward as the voxel rises, drift with their ICA family as the
  family swings (correlated voxels visibly move together -- what flocking
  means here), align with their neighbours through a coarse velocity grid, and
  shimmer with how unpredictable the voxel is from frame to frame.
"""

from __future__ import annotations

import math

import torch

from ..drives import Param, splat
from .base import Scene, orbit_project


def golden_directions(n: int, device: torch.device) -> torch.Tensor:
    """n roughly even unit vectors (Fibonacci sphere)."""
    i = torch.arange(n, device=device, dtype=torch.float32) + 0.5
    phi = torch.acos(1 - 2 * i / n)
    theta = math.pi * (1 + 5**0.5) * i
    return torch.stack(
        [torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi)], 1
    )


S = "brain"


class BrainScene(Scene):
    name = S
    blurb = "the brain as a 3-D swarm: anatomy as dust, live signal as sparks"
    PARAMS = (
        Param(
            "morph",
            0.0,
            0.0,
            1.0,
            S,
            "anatomical position (0) to correlation-space position (1)",
            scene=S,
        ),
        Param("spring", 6.0, 0.0, 40.0, S, "pull of each particle toward its home", scene=S),
        Param("drag", 3.0, 0.0, 20.0, S, "velocity damping", scene=S),
        Param("bloom", 0.05, 0.0, 0.3, S, "outward push of sparks when their voxel rises", scene=S),
        Param(
            "flock",
            0.06,
            0.0,
            0.4,
            S,
            "drift of each family along its own direction as it swings",
            scene=S,
        ),
        Param(
            "align",
            2.0,
            0.0,
            10.0,
            S,
            "match the velocity of nearby particles (coarse grid)",
            scene=S,
        ),
        Param(
            "shimmer",
            0.25,
            0.0,
            4.0,
            S,
            "turbulence scaled by frame-to-frame unpredictability",
            scene=S,
        ),
        Param("shudder", 0.5, 0.0, 3.0, S, "whole-swarm radial jolt on DVARS spikes", scene=S),
        Param("dust", 0.3, 0.0, 2.0, S, "brightness of the anatomical dust", scene=S),
        Param("fog", 0.6, 0.0, 1.0, S, "depth dimming", scene=S),
        Param("orbit", 0.04, -0.5, 0.5, S, "camera turn rate (radians per second)", scene=S),
        Param("yaw", 0.0, -math.pi, math.pi, S, "camera yaw", scene=S),
        Param("pitch", 0.35, -1.5, 1.5, S, "camera pitch", scene=S),
        Param("zoom", 1.3, 0.3, 4.0, S, "camera zoom", scene=S),
        Param("sparks", 1, 0, 4, S, "spark particles per voxel", scene=S),
    )

    def __init__(self, drives, width, height, seed=0):
        super().__init__(drives, width, height, seed)
        f = self.f
        self.fam_dir = golden_directions(drives.n_fam, self.device)
        self.radial = f.anat / f.anat.norm(dim=1, keepdim=True).clamp_min(1e-3)
        # Dust brightness: the mean image's local contrast, so the scaffold
        # draws tissue boundaries (cortex, ventricles, the brain's edge) rather
        # than a uniformly lit ball.
        nb = f.extras["neighbours"]
        edge = (f.mean[:, None] - f.mean[nb]).abs().mean(1)
        self.dust_w = (edge / torch.quantile(edge, 0.95).clamp_min(1e-6)).clamp(0, 1) ** 1.5
        self.spacing = float((f.anat - f.anat[nb[:, 0]]).norm(dim=1).median())
        self._build(int(self.p["sparks"]))

    def _build(self, sparks_per_voxel: int) -> None:
        self.n_sparks = sparks_per_voxel  # _rep reads it while building
        n = self.f.n_voxels
        ar = torch.arange(n, device=self.device)
        self.vox = ar.repeat(1 + sparks_per_voxel)
        self.is_spark = torch.arange(self.vox.numel(), device=self.device) >= n
        m = self.vox.numel()
        # Each spark copy gets a small fixed offset so several per voxel read as
        # a soft cluster; dust gets up to half a voxel, because a perfect
        # lattice through a rotating camera moires.
        jitter = self.randn(m, 3) * 0.004
        jitter[:n] = (self.rand(n, 3) - 0.5) * self.spacing
        self.jitter = jitter
        self.pos = self._home() + jitter
        self.vel = torch.zeros(m, 3, device=self.device)
        self.n_sparks = sparks_per_voxel

    def _rep(self, x: torch.Tensor) -> torch.Tensor:
        """Per-voxel values onto the particles (dust, then each spark copy).

        The particles are the voxels repeated, so a repeat does what a gather
        by ``vox`` did, at a fraction of the cost on the CPU.
        """
        return x.repeat(1 + self.n_sparks, *([1] * (x.ndim - 1)))

    def on_param(self, name, value):
        if name == "sparks" and int(value) != self.n_sparks:
            self._build(int(value))

    def _home(self) -> torch.Tensor:
        f = self.f
        m = self.p["morph"]
        base = f.anat if m <= 0 else f.anat * (1 - m) + f.func * m
        return self._rep(base)

    def step(self, dt: float) -> None:
        p, f, dr = self.p, self.f, self.dr
        spark = self.is_spark
        lv = self._rep(dr.live)

        target = self._home() + self.jitter
        lift = (lv.clamp_min(0) * p["bloom"])[:, None] * self._rep(self.radial)
        drift_v = (dr.fam[:, None] * self.fam_dir) * p["flock"]  # (F, 3)
        drift = self._rep(drift_v[f.family] * f.family_weight[:, None])
        target = target + (lift + drift) * spark[:, None]
        force = (target - self.pos) * p["spring"]

        if p["shimmer"] > 0:
            # A smooth analytic flow field (curl of sines), scaled by how
            # unpredictable each voxel is frame to frame.
            q = self.pos * 7.0
            w = dr.wall * 0.7
            flow = torch.stack(
                [
                    torch.sin(q[:, 1] + w) - torch.cos(q[:, 2] * 1.3 - w),
                    torch.sin(q[:, 2] * 1.1 + w * 1.2) - torch.cos(q[:, 0] + w * 0.8),
                    torch.sin(q[:, 0] * 0.9 - w) - torch.cos(q[:, 1] * 1.2 + w * 0.6),
                ],
                1,
            )
            amp = self._rep(dr.innovation) * (0.3 + lv.abs().clamp(max=3)) * spark
            force = force + flow * (amp * p["shimmer"] * 0.6)[:, None]

        # Everything gets a radial kick on a DVARS spike: motion and other
        # whole-brain events become a visible jolt.
        if p["shudder"] > 0 and dr.d > 2.5:
            force = force + self._rep(self.radial) * ((dr.d - 2.5) * p["shudder"] * 0.8)

        self.vel = self.vel + force * dt
        if p["align"] > 0:
            self.vel = self.vel + (self._grid_velocity() - self.vel) * min(1.0, p["align"] * dt)
        self.vel = self.vel * math.exp(-p["drag"] * dt)
        self.pos = self.pos + self.vel * dt

    def _grid_velocity(self, g: int = 32) -> torch.Tensor:
        """Mean velocity of the particles sharing each cell of a coarse grid."""
        cell = ((self.pos.clamp(-1.5, 1.5) + 1.5) / 3.0 * (g - 1)).round().long()
        key = (cell[:, 0] * g + cell[:, 1]) * g + cell[:, 2]
        acc = torch.zeros(g**3, 3, device=self.device).index_add_(0, key, self.vel)
        cnt = torch.zeros(g**3, device=self.device).index_add_(
            0, key, torch.ones_like(key, dtype=torch.float32)
        )
        return acc[key] / cnt[key, None]

    def _project(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.p
        return orbit_project(
            self.pos,
            p["yaw"] + self.dr.wall * p["orbit"],
            p["pitch"],
            p["zoom"],
            self.width,
            self.height,
        )

    def render(self) -> torch.Tensor:
        p, dr = self.p, self.dr
        h, w = self.height, self.width
        sx, sy, depth = self._project()
        n = self.f.n_voxels
        self.remember(sx[:n], sy[:n])  # the dust: one particle per voxel, on its home
        spark = self.is_spark
        dens = self.density
        spark_i = self._rep(dr.brightness()) * dens / max(1, self.n_sparks)
        dust_i = p["dust"] * (0.05 + self._rep(self.dust_w)) * dens * 1.5
        inten = torch.where(spark, spark_i, dust_i)
        inten = inten * torch.exp(-(depth - 3.2).clamp_min(-1.5) * p["fog"] * 1.2)
        dust_rgb = dr._const("dust", (0.55, 0.65, 0.9))
        rgb = torch.where(spark[:, None], self._rep(dr.colors()), dust_rgb) * inten[:, None]
        frame = torch.zeros(h * w, 3, device=self.device)
        splat(frame, sx, sy, rgb, w, h)
        self.fade_in(frame)
        return dr.finish(self.accum, h, w)
