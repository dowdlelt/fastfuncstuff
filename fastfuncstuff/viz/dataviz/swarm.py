"""The particle swarm and its pixel renderer, both plain torch.

Two kinds of particle share one state array:

* **dust** -- one per voxel, held on its anatomical home and lit by the voxel's
  mean brightness. It is the scaffold the eye uses to know where things are.
* **sparks** -- one (or more) per voxel, lit by the voxel's *live* value and
  pushed around by the data: they bloom outward when the voxel rises, drift
  with their ICA family as the family's time course swings (so correlated
  voxels visibly move together, which is what flocking means here), align with
  their neighbours through a coarse velocity grid, and shimmer in proportion to
  how unpredictable the voxel is from one frame to the next.

Nothing is fitted. The swarm only replays the series, so whatever structure the
run carries -- a block design, a sweeping wedge, a motion spike, a slow drift --
appears because it is there, and the eye does the finding.

Rendering is additive splatting into a low-resolution float buffer with a
decaying history (trails), then a soft tone curve. Many faint points summed is
where the "pixel art from a lot of pixels" look comes from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .features import Features


@dataclass(frozen=True)
class Param:
    name: str
    default: float
    lo: float
    hi: float
    group: str
    help: str
    choices: tuple[str, ...] = ()


COLOR_SOURCES = ("family", "wave", "sign", "tsnr", "coherence", "mean")
LAYOUTS = ("anatomy", "function")

PARAMS: tuple[Param, ...] = (
    # time
    Param("speed", 2.0, 0.1, 20.0, "time", "TRs played per second"),
    Param(
        "clarity",
        0.35,
        0.0,
        1.0,
        "pool",
        "pool across correlation: mix toward the low-rank (PCA) reconstruction",
    ),
    Param(
        "pool", 0.5, 0.0, 1.0, "pool", "pool across space: mix each voxel with its neighbourhood"
    ),
    Param("reach", 2, 1, 6, "pool", "neighbourhood size (passes of face-neighbour averaging)"),
    Param("gate", 2.0, 0.0, 6.0, "pool", "dim voxels whose variance is not shared (salience)"),
    Param(
        "smooth", 0.25, 0.0, 0.95, "time", "temporal smoothing of the live value (EMA per frame)"
    ),
    # layout
    Param(
        "morph",
        0.0,
        0.0,
        1.0,
        "layout",
        "anatomical position (0) to correlation-space position (1)",
    ),
    Param("spring", 6.0, 0.0, 40.0, "layout", "pull of each particle toward its home"),
    Param("drag", 3.0, 0.0, 20.0, "layout", "velocity damping"),
    # motion
    Param("bloom", 0.05, 0.0, 0.3, "motion", "outward push of sparks when their voxel rises"),
    Param(
        "flock",
        0.06,
        0.0,
        0.4,
        "motion",
        "drift of each ICA family along its own direction as it swings",
    ),
    Param(
        "align", 2.0, 0.0, 10.0, "motion", "match the velocity of nearby particles (coarse grid)"
    ),
    Param(
        "shimmer", 0.25, 0.0, 4.0, "motion", "turbulence scaled by frame-to-frame unpredictability"
    ),
    Param("shudder", 0.5, 0.0, 3.0, "motion", "whole-swarm radial jolt on DVARS spikes"),
    # colour
    Param("color", 0, 0, len(COLOR_SOURCES) - 1, "color", "spark colour source", COLOR_SOURCES),
    Param("wave", 0, 0, 3, "color", "which complex-PCA wave the 'wave' colour reads"),
    Param("gain", 6.0, 0.0, 30.0, "color", "spark brightness per unit z"),
    Param("threshold", 0.6, 0.0, 3.0, "color", "z below which sparks stay dark"),
    Param("negative", 0.35, 0.0, 1.0, "color", "how much falling voxels glow (cool) vs. stay dark"),
    Param("dust", 0.3, 0.0, 2.0, "color", "brightness of the anatomical dust"),
    Param("tint", 0.5, 0.0, 2.0, "color", "global-signal colour temperature shift"),
    Param("flash", 0.4, 0.0, 2.0, "color", "global brightening on DVARS spikes"),
    # render
    Param("trails", 0.82, 0.0, 0.99, "render", "fraction of the previous frame kept"),
    Param("exposure", 1.0, 0.1, 8.0, "render", "tone curve exposure"),
    Param("fog", 0.6, 0.0, 1.0, "render", "depth dimming"),
    Param("orbit", 0.04, -0.5, 0.5, "render", "camera turns per minute... per second x10"),
    Param("yaw", 0.0, -math.pi, math.pi, "render", "camera yaw"),
    Param("pitch", 0.35, -1.5, 1.5, "render", "camera pitch"),
    Param("zoom", 1.3, 0.3, 4.0, "render", "camera zoom"),
    Param("sparks", 1, 0, 4, "render", "spark particles per voxel"),
    Param("layout", 0, 0, 1, "layout", "home layout for the dust", LAYOUTS),
)
PARAM_BY_NAME = {p.name: p for p in PARAMS}


def default_params() -> dict[str, float]:
    return {p.name: float(p.default) for p in PARAMS}


def golden_directions(n: int, device: torch.device) -> torch.Tensor:
    """n roughly even unit vectors (Fibonacci sphere)."""
    i = torch.arange(n, device=device, dtype=torch.float32) + 0.5
    phi = torch.acos(1 - 2 * i / n)
    theta = math.pi * (1 + 5**0.5) * i
    return torch.stack(
        [torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi)], 1
    )


def hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    h = (h % 1.0) * 6.0
    k = torch.stack([(5 + h) % 6, (3 + h) % 6, (1 + h) % 6], -1)
    return v[..., None] - v[..., None] * s[..., None] * (torch.minimum(k, 4 - k).clamp(0, 1))


def ramp(x: torch.Tensor, stops: list[tuple[float, float, float]]) -> torch.Tensor:
    """Piecewise-linear colour ramp over [0, 1]."""
    c = torch.tensor(stops, device=x.device, dtype=torch.float32)
    pos = x.clamp(0, 1) * (len(stops) - 1)
    i = pos.floor().long().clamp(max=len(stops) - 2)
    f = (pos - i)[:, None]
    return c[i] * (1 - f) + c[i + 1] * f


EMBER = [
    (0.10, 0.05, 0.25),
    (0.55, 0.10, 0.45),
    (0.95, 0.40, 0.25),
    (1.0, 0.85, 0.45),
    (1.0, 1.0, 0.9),
]
GLACIER = [(0.05, 0.10, 0.25), (0.10, 0.35, 0.60), (0.30, 0.75, 0.85), (0.85, 0.98, 1.0)]


class Swarm:
    """Particle state, one simulation step, one render.

    ``step(dt)`` advances data time by ``speed * dt`` TRs and integrates the
    particles; ``render()`` returns an (H, W, 3) uint8 frame on the host.
    """

    def __init__(self, feats: Features, width: int = 480, height: int = 270, seed: int = 0):
        self.f = feats
        self.device = feats.z.device
        self.p = default_params()
        self.width, self.height = width, height
        self.t = 0.0  # data time in TRs
        self.wall = 0.0
        self._gen = torch.Generator(device="cpu").manual_seed(seed)
        n = feats.n_voxels
        self.n_fam = int(feats.family_tcs.shape[1])
        self.fam_dir = golden_directions(self.n_fam, self.device)
        # Radial direction from the anatomical centre: "outward" for bloom.
        a = feats.anat
        self.radial = a / a.norm(dim=1, keepdim=True).clamp_min(1e-3)
        # Per-voxel innovation scale: how much of a voxel's variance is
        # frame-to-frame change. Smooth signal glides, noise fizzes.
        dz = (feats.z[1:].float() - feats.z[:-1].float()).square().mean(0).sqrt()
        self.innovation = (dz / 2**0.5).clamp(0, 1.5)
        self.live = torch.zeros(n, device=self.device)
        # Dust brightness: the mean image's local contrast, so the scaffold
        # draws tissue boundaries (cortex, ventricles, the brain's edge) rather
        # than a uniformly lit ball.
        nb = feats.extras["neighbours"]
        edge = (feats.mean[:, None] - feats.mean[nb]).abs().mean(1)
        self.dust_w = (edge / torch.quantile(edge, 0.95).clamp_min(1e-6)).clamp(0, 1) ** 1.5
        self._build_particles(int(self.p["sparks"]))
        self.accum = torch.zeros(height * width, 3, device=self.device)
        self._static_color_key: tuple | None = None

    # ------------------------------------------------------------------ setup

    def _build_particles(self, sparks_per_voxel: int) -> None:
        n = self.f.n_voxels
        vox = [torch.arange(n, device=self.device)]
        kind = [torch.zeros(n, dtype=torch.bool, device=self.device)]
        for _ in range(sparks_per_voxel):
            vox.append(torch.arange(n, device=self.device))
            kind.append(torch.ones(n, dtype=torch.bool, device=self.device))
        self.vox = torch.cat(vox)
        self.is_spark = torch.cat(kind)
        m = self.vox.numel()
        # Each copy of a voxel gets a fixed small offset, so several sparks per
        # voxel read as a soft cluster rather than one point drawn twice.
        # Dust gets up to half a voxel of fixed jitter too: a perfect lattice
        # projected through a rotating camera moires.
        jitter = torch.randn(m, 3, generator=self._gen).to(self.device) * 0.004
        nb = self.f.extras["neighbours"][:, 0]
        spacing = float((self.f.anat - self.f.anat[nb]).norm(dim=1).median())
        n_dust = int((~self.is_spark).sum())
        jitter[~self.is_spark] = (
            torch.rand(n_dust, 3, generator=self._gen).to(self.device) - 0.5
        ) * spacing
        self.jitter = jitter
        self.pos = self._home(torch.zeros(self.f.n_voxels, device=self.device)) + jitter
        self.vel = torch.zeros(m, 3, device=self.device)
        self.n_sparks = sparks_per_voxel
        self._static_color_key = None

    def resize(self, width: int, height: int) -> None:
        if (width, height) != (self.width, self.height):
            self.width, self.height = int(width), int(height)
            self.accum = torch.zeros(self.height * self.width, 3, device=self.device)

    def seek(self, t: float) -> None:
        self.t = float(t) % (self.n_time - 1)

    def set_param(self, name: str, value: float) -> None:
        self.p[name] = float(value)
        if name == "sparks" and int(value) != self.n_sparks:
            self._build_particles(int(value))

    # --------------------------------------------------------------- data time

    @property
    def n_time(self) -> int:
        return self.f.n_time

    def _interp_rows(self, mat: torch.Tensor, t: float) -> torch.Tensor:
        """Catmull-Rom interpolation of rows of a (T, ...) tensor at fractional t."""
        n = mat.shape[0]
        i = math.floor(t)
        u = t - i
        idx = [min(max(i + k, 0), n - 1) for k in (-1, 0, 1, 2)]
        p0, p1, p2, p3 = (mat[j].float() for j in idx)
        u2, u3 = u * u, u * u * u
        return 0.5 * (
            2 * p1
            + (-p0 + p2) * u
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2
            + (-p0 + 3 * p1 - 3 * p2 + p3) * u3
        )

    def activity(self, t: float) -> torch.Tensor:
        """Each voxel's value at data time t: raw z blended toward its low-rank part."""
        raw = self._interp_rows(self.f.z, t)
        c = self.p["clarity"]
        if c <= 0:
            return raw
        # Unscaled for the same reason as ``pooled``: a voxel's low-rank part
        # keeps the variance its correlated partners vouch for, and loses the rest.
        recon = self.f.loadings @ self._interp_rows(self.f.tcs, t)
        return raw * (1 - c) + recon * c

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        """Blend each voxel's value with its neighbourhood's (``reach`` passes)."""
        mix = self.p["pool"]
        if mix <= 0:
            return x
        nb = self.f.extras["neighbours"]
        local = x
        for _ in range(int(self.p["reach"])):
            local = (local + local[nb].sum(1)) / 7.0
        # No rescale on purpose: averaging shrinks independent noise and leaves
        # a spatially coherent response at full size, and that contrast is the
        # point -- most of the brain carries no task signal.
        return x * (1 - mix) + local * mix

    def global_at(self, t: float) -> tuple[float, float]:
        g = float(self._interp_rows(self.f.global_signal[:, None], t)[0])
        d = float(self._interp_rows(self.f.dvars[:, None], t)[0])
        return g, d

    # ---------------------------------------------------------------- dynamics

    def _home(self, live: torch.Tensor) -> torch.Tensor:
        f = self.f
        m = self.p["morph"]
        base = f.anat if m <= 0 else f.anat * (1 - m) + f.func * m
        home_v = base  # (N, 3)
        out = home_v[self.vox] + (self.jitter if hasattr(self, "jitter") else 0)
        return out

    def step(self, dt: float) -> None:
        p = self.p
        f = self.f
        dt = float(min(dt, 0.05))
        self.wall += dt
        self.t = (self.t + p["speed"] * dt) % (self.n_time - 1)
        a = self.pooled(self.activity(self.t))
        k = p["smooth"]
        self.live = self.live * k + a * (1 - k)
        live = self.live

        g, d = self.global_at(self.t)
        self._g, self._d = g, d

        spark = self.is_spark
        vox = self.vox
        lv = live[vox]

        target = self._home(live)
        # Bloom: rising sparks lift off along their outward normal.
        lift = (lv.clamp_min(0) * p["bloom"])[:, None] * self.radial[vox]
        # Flock: each family drifts along its own direction as its time course
        # swings, weighted by membership. Same family, same motion.
        ftc = self._interp_rows(f.family_tcs, self.t)  # (F,)
        drift_v = (ftc[:, None] * self.fam_dir) * p["flock"]  # (F, 3)
        drift = drift_v[f.family[vox]] * f.family_weight[vox][:, None]
        offset = (lift + drift) * spark[:, None]
        target = target + offset

        force = (target - self.pos) * p["spring"]

        # Shimmer: a smooth analytic flow field (curl of sines), scaled by how
        # unpredictable each voxel is frame to frame.
        if p["shimmer"] > 0:
            q = self.pos * 7.0
            w = self.wall * 0.7
            flow = torch.stack(
                [
                    torch.sin(q[:, 1] + w) - torch.cos(q[:, 2] * 1.3 - w),
                    torch.sin(q[:, 2] * 1.1 + w * 1.2) - torch.cos(q[:, 0] + w * 0.8),
                    torch.sin(q[:, 0] * 0.9 - w) - torch.cos(q[:, 1] * 1.2 + w * 0.6),
                ],
                1,
            )
            amp = self.innovation[vox] * (0.3 + lv.abs().clamp(max=3)) * spark
            force = force + flow * (amp * p["shimmer"] * 0.6)[:, None]

        # Shudder: everything gets a radial kick on a DVARS spike -- motion and
        # other global events become a visible jolt.
        if p["shudder"] > 0 and d > 2.5:
            force = force + self.radial[vox] * ((d - 2.5) * p["shudder"] * 0.8)

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

    # ------------------------------------------------------------------ colour

    def _static_colors(self) -> torch.Tensor:
        """(N, 3) per-voxel spark hue from the chosen colour source."""
        p = self.p
        src = COLOR_SOURCES[int(p["color"])]
        key = (src, int(p["wave"]))
        if key == self._static_color_key:
            return self._static_rgb
        f = self.f
        n = f.n_voxels
        one = torch.ones(n, device=self.device)
        if src == "family":
            hue = (f.family.float() * 0.618034 + 0.08) % 1.0
            rgb = hsv_to_rgb(hue, 0.35 + 0.5 * f.family_weight, one)
        elif src == "wave":
            w = min(int(p["wave"]), f.phase.shape[1] - 1)
            hue = (f.phase[:, w] + math.pi) / (2 * math.pi)
            rgb = hsv_to_rgb(hue, 0.15 + 0.8 * f.phase_power[:, w], one)
        elif src == "tsnr":
            rgb = ramp(f.tsnr, GLACIER)
        elif src == "coherence":
            rgb = ramp((f.coherence.clamp(0, 0.6) / 0.6), EMBER)
        elif src == "mean":
            rgb = ramp(f.mean, GLACIER)
        else:  # sign: decided per frame
            rgb = torch.ones(n, 3, device=self.device)
        # Normalise luminance so colour choice doesn't change brightness.
        lum = (rgb * torch.tensor([0.30, 0.59, 0.11], device=self.device)).sum(1, keepdim=True)
        self._static_rgb = rgb / lum.clamp_min(0.15) * 0.7
        self._static_color_key = key
        return self._static_rgb

    def _gate(self) -> torch.Tensor:
        """Per-voxel spark weight from salience, in [0, 1]."""
        key = self.p["gate"]
        if getattr(self, "_gate_key", None) != key:
            self._gate_w = self.f.salience**key
            self._gate_key = key
        return self._gate_w

    # ------------------------------------------------------------------ render

    def _project(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.p
        yaw = p["yaw"] + self.wall * p["orbit"]
        pitch = p["pitch"]
        cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
        # World is RAS: x right, y anterior, z superior. Camera looks along +y
        # (from behind the head) with z up, then yaw about z and pitch about x.
        rot_yaw = torch.tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], device=self.device)
        rot_pitch = torch.tensor([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], device=self.device)
        r = rot_pitch @ rot_yaw
        q = self.pos @ r.T
        depth = q[:, 1] + 3.2  # distance from camera
        scale = p["zoom"] * self.height * 1.05 / depth
        sx = self.width * 0.5 + q[:, 0] * scale
        sy_ = self.height * 0.5 - q[:, 2] * scale
        return sx, sy_, depth

    def render(self) -> np.ndarray:
        return self.render_tensor().cpu().numpy()

    def render_tensor(self) -> torch.Tensor:
        p = self.p
        h, w = self.height, self.width
        sx, sy, depth = self._project()
        vox = self.vox
        spark = self.is_spark
        lv = self.live[vox]
        g, d = getattr(self, "_g", 0.0), getattr(self, "_d", 0.0)

        # Brightness: dust by mean image, sparks by live value above threshold.
        thr = p["threshold"]
        up = (lv - thr).clamp_min(0)
        down = (-lv - thr).clamp_min(0) * p["negative"]
        # Brightness per particle is normalised by how many particles share the
        # screen, so a 50k-voxel slab and a 1M-voxel brain expose alike.
        density = (h * w * 0.08) / max(1, self.f.n_voxels)
        spark_i = (up + down) * p["gain"] * density / max(1, self.n_sparks)
        if p["gate"] > 0:
            spark_i = spark_i * self._gate()[vox]
        dust_i = p["dust"] * (0.05 + self.dust_w[vox]) * density * 1.5
        inten = torch.where(spark, spark_i, dust_i)
        fog = torch.exp(-(depth - 3.2).clamp_min(-1.5) * p["fog"] * 1.2)
        inten = inten * fog

        base = self._static_colors()[vox]
        if COLOR_SOURCES[int(p["color"])] == "sign":
            warm = torch.tensor([1.0, 0.55, 0.2], device=self.device)
            cool = torch.tensor([0.25, 0.55, 1.0], device=self.device)
            base = torch.where((lv >= 0)[:, None], warm, cool).expand(lv.numel(), 3)
        else:
            # Falling voxels glow in a cool version of their own colour.
            cool = base * torch.tensor([0.45, 0.7, 1.3], device=self.device)
            base = torch.where((lv >= 0)[:, None], base, cool)
        dust_rgb = torch.tensor([0.55, 0.65, 0.9], device=self.device)
        rgb = torch.where(spark[:, None], base, dust_rgb) * inten[:, None]

        # Bilinear splat: four index_adds with sub-pixel weights, so motion is
        # continuous even though the buffer is coarse.
        x0 = sx.floor()
        y0 = sy.floor()
        fx = (sx - x0)[:, None]
        fy = (sy - y0)[:, None]
        x0 = x0.long()
        y0 = y0.long()
        frame = torch.zeros(h * w, 3, device=self.device)
        for dx, dy, wt in (
            (0, 0, (1 - fx) * (1 - fy)),
            (1, 0, fx * (1 - fy)),
            (0, 1, (1 - fx) * fy),
            (1, 1, fx * fy),
        ):
            xi = x0 + dx
            yi = y0 + dy
            ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            frame.index_add_(0, (yi * w + xi)[ok], (rgb * wt)[ok])

        self.accum.mul_(p["trails"]).add_(frame * (1 - p["trails"]) * 4.0)

        # Global signal shifts colour temperature; DVARS spikes flash.
        tint = math.tanh(g * 0.5) * p["tint"] * 0.25
        tint_v = torch.tensor([1 + tint, 1.0, 1 - tint], device=self.device).clamp_min(0)
        flash = 1.0 + max(0.0, d - 2.0) * p["flash"] * 0.25
        x = self.accum * tint_v * (p["exposure"] * flash)
        out = 1.0 - torch.exp(-x)
        bg = torch.tensor([0.0, 0.0, 0.004], device=self.device)
        out = (out + bg * (1 - out)).clamp(0, 1) ** (1 / 1.8)
        return (out.reshape(h, w, 3) * 255).to(torch.uint8)
