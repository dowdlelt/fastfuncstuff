"""The shared data clock every scene reads, and the pixel helpers they share.

A scene is only a way of *drawing* the drives. The drives themselves -- each
voxel's pooled live value, each family's time course, the global signal and
DVARS -- are computed once per frame here, so switching scene never changes
what the data is saying, only how it is shown.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from fastfuncstuff import policy_linalg as plinalg

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
    scene: str | None = None  # None: shared by every scene

    @property
    def integer(self) -> bool:
        return bool(self.choices) or (
            float(self.default).is_integer()
            and float(self.lo).is_integer()
            and float(self.hi).is_integer()
            and self.hi - self.lo <= 12
        )


CANVASES = ("head", "weave")
COLOR_SOURCES = ("family", "phase", "spectrum", "wave", "sign", "tsnr", "coherence", "mean")

SHARED_PARAMS: tuple[Param, ...] = (
    Param("speed", 2.0, 0.1, 20.0, "time", "TRs played per second"),
    Param(
        "smooth", 0.25, 0.0, 0.95, "time", "temporal smoothing of the live value (EMA per frame)"
    ),
    Param(
        "clarity",
        0.35,
        0.0,
        1.0,
        "pool",
        "pool across correlation: mix toward the low-rank reconstruction",
    ),
    Param(
        "pool", 0.5, 0.0, 1.0, "pool", "pool across space: mix each voxel with its neighbourhood"
    ),
    Param("reach", 2, 1, 6, "pool", "neighbourhood size (passes of face-neighbour averaging)"),
    Param("gate", 2.0, 0.0, 6.0, "pool", "dim voxels whose variance is not shared (salience)"),
    Param("color", 0, 0, len(COLOR_SOURCES) - 1, "color", "colour source", COLOR_SOURCES),
    Param("wave", 0, 0, 3, "color", "which complex-PCA wave the 'wave' colour reads"),
    Param("gain", 6.0, 0.0, 30.0, "color", "brightness per unit z"),
    Param("threshold", 0.6, 0.0, 3.0, "color", "z below which voxels stay dark"),
    Param("negative", 0.35, 0.0, 1.0, "color", "how much falling voxels glow (cool) vs. stay dark"),
    Param("tint", 0.5, 0.0, 2.0, "color", "global-signal colour temperature shift"),
    Param("flash", 0.4, 0.0, 2.0, "color", "global brightening on DVARS spikes"),
    Param(
        "canvas",
        0,
        0,
        1,
        "render",
        "grid scenes: the head from above, or every voxel woven across the full field by correlation",
        CANVASES,
    ),
    Param("trails", 0.82, 0.0, 0.995, "render", "fraction of the previous frame kept"),
    Param("exposure", 1.0, 0.1, 8.0, "render", "tone curve exposure"),
    Param(
        "saturation",
        1.5,
        0.0,
        3.0,
        "render",
        "colour saturation after mixing (many hues summed go grey)",
    ),
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


def family_hues(n: int) -> torch.Tensor:
    """Golden-ratio hue per family: neighbours in index never share a colour."""
    return (torch.arange(n, dtype=torch.float32) * 0.618034 + 0.08) % 1.0


def _interp_np(arr, t: float) -> float:
    """Catmull-Rom of a host 1-D array at fractional t (matches ``interp_rows``)."""
    n = len(arr)
    i = math.floor(t)
    u = t - i
    p0, p1, p2, p3 = (float(arr[min(max(i + k, 0), n - 1)]) for k in (-1, 0, 1, 2))
    return 0.5 * (
        2 * p1
        + (-p0 + p2) * u
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
        + (-p0 + 3 * p1 - 3 * p2 + p3) * u**3
    )


def interp_many(mat: torch.Tensor, ts) -> torch.Tensor:
    """``interp_rows`` at many fractional times at once: (len(ts), ...).

    One gather and one weighted sum, instead of a handful of small kernels per
    time point (a 32-sample window was ~150 launches a frame).
    """
    import numpy as np

    t = np.asarray(ts, dtype=np.float64)
    n = mat.shape[0]
    i = np.floor(t).astype(np.int64)
    u = t - i
    idx = np.clip(i[:, None] + np.arange(-1, 3)[None], 0, n - 1)  # (m, 4)
    u2, u3 = u * u, u * u * u
    w = 0.5 * np.stack([-u + 2 * u2 - u3, 2 - 5 * u2 + 3 * u3, u + 4 * u2 - 3 * u3, -u2 + u3], 1)
    rows = mat[torch.as_tensor(idx.reshape(-1), device=mat.device)].float()
    rows = rows.reshape(len(t), 4, *mat.shape[1:])
    wt = torch.as_tensor(w, dtype=torch.float32, device=mat.device).reshape(
        len(t), 4, *([1] * (mat.ndim - 1))
    )
    return (rows * wt).sum(1)


def interp_cols(mat: torch.Tensor, t: float, cols: torch.Tensor) -> torch.Tensor:
    """``interp_rows(mat, t)[cols]`` reading only four rows' worth of ``cols``."""
    n = mat.shape[0]
    i = math.floor(t)
    rows = torch.tensor([min(max(i + k, 0), n - 1) for k in (-1, 0, 1, 2)], device=mat.device)
    return interp_rows(mat[rows][:, cols], 1.0 + (t - i))


def interp_rows(mat: torch.Tensor, t: float) -> torch.Tensor:
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


def splat(
    frame: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor, rgb: torch.Tensor, w: int, h: int
) -> None:
    """Bilinear additive splat of points into an (h*w, 3) buffer.

    Sub-pixel weights keep motion continuous even on a coarse buffer, which is
    what keeps chunky pixels from crawling. Off-screen taps are clamped and
    zero-weighted rather than masked out: a boolean mask forces a host sync on
    CUDA and a new tensor shape every frame on Metal (a kernel recompile). On
    the CPU a compiled loop replaces the torch ops, whose temporaries cost more
    than the accumulation (52 -> 12 ms for 320k points).
    """
    if frame.device.type == "cpu":
        _splat_cpu(
            frame.numpy(),
            sx.float().contiguous().numpy(),
            sy.float().contiguous().numpy(),
            rgb.float().contiguous().numpy(),
            w,
            h,
        )
        return
    x0 = sx.floor()
    y0 = sy.floor()
    fx = (sx - x0)[:, None]
    fy = (sy - y0)[:, None]
    x0 = x0.long()
    y0 = y0.long()
    xs = torch.cat([x0, x0 + 1, x0, x0 + 1])
    ys = torch.cat([y0, y0, y0 + 1, y0 + 1])
    wt = torch.cat([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy])
    ok = ((xs >= 0) & (xs < w) & (ys >= 0) & (ys < h))[:, None]
    idx = ys.clamp(0, h - 1) * w + xs.clamp(0, w - 1)
    frame.index_add_(0, idx, torch.where(ok, rgb.repeat(4, 1) * wt, 0.0))


def _splat_cpu(frame, sx, sy, rgb, w: int, h: int) -> None:
    global _SPLAT_NB
    if _SPLAT_NB is None:
        _SPLAT_NB = _compile_splat()
    _SPLAT_NB(frame, sx, sy, rgb, w, h)


_SPLAT_NB = None


def _compile_splat():
    import numba
    import numpy as np

    @numba.njit(cache=True, fastmath=True)
    def splat_nb(frame, sx, sy, rgb, w, h):  # pragma: no cover - compiled
        for i in range(sx.shape[0]):
            x = sx[i]
            y = sy[i]
            if not (x > -1.0 and x < w and y > -1.0 and y < h):
                continue  # also drops NaN and inf
            x0 = int(np.floor(x))
            y0 = int(np.floor(y))
            fx = x - x0
            fy = y - y0
            for dy in range(2):
                yi = y0 + dy
                if yi < 0 or yi >= h:
                    continue
                wy = fy if dy else 1.0 - fy
                for dx in range(2):
                    xi = x0 + dx
                    if xi < 0 or xi >= w:
                        continue
                    wgt = wy * (fx if dx else 1.0 - fx)
                    k = yi * w + xi
                    frame[k, 0] += rgb[i, 0] * wgt
                    frame[k, 1] += rgb[i, 1] * wgt
                    frame[k, 2] += rgb[i, 2] * wgt

    return splat_nb


def uniform_disc(xy: torch.Tensor) -> torch.Tensor:
    """Keep each point's angle, replace its radius by sqrt(rank) -- a uniform disc.

    The correlation-space embedding piles noise voxels (tiny loadings) at the
    origin, so drawn as-is it is one bright blob. Ranking the radius spreads
    the cloud evenly while keeping who sits near whom by angle and by order.
    """
    r = xy.norm(dim=1)
    rank = torch.empty_like(r)
    rank[torch.argsort(r)] = torch.linspace(0, 1, r.numel(), device=r.device)
    return xy / r.clamp_min(1e-9)[:, None] * rank.sqrt()[:, None]


_LUMA = (0.30, 0.59, 0.11)


def luma(rgb: torch.Tensor) -> torch.Tensor:
    """(..., 1) perceptual luminance, from scalars (no per-call device tensor)."""
    return rgb[..., :1] * _LUMA[0] + rgb[..., 1:2] * _LUMA[1] + rgb[..., 2:3] * _LUMA[2]


class Drives:
    """Data time, the live per-voxel value, and everything derived per frame."""

    def __init__(self, feats: Features, params: dict[str, float]):
        self.f = feats
        self.p = params
        self.device = feats.z.device
        self.t = 0.0  # data time, TRs
        self.wall = 0.0  # seconds since start, for motion that runs while paused
        self.live = torch.zeros(feats.n_voxels, device=self.device)
        self.g = 0.0
        self.d = 0.0
        dz = (feats.z[1:].float() - feats.z[:-1].float()).square().mean(0).sqrt()
        # Per-voxel innovation: how much of the variance is frame-to-frame
        # change. Smooth signal glides, noise fizzes.
        self.innovation = (dz / 2**0.5).clamp(0, 1.5)
        self._color_key: tuple | None = None
        self._gate_key: tuple | None = None
        self.refresh_families()
        nb = feats.extras["neighbours"][:, 0]
        #: Voxel spacing in the normalised anatomical frame.
        self.spacing = float((feats.anat - feats.anat[nb]).norm(dim=1).median().clamp_min(1e-4))

    def refresh_families(self) -> None:
        """Per-family statistics; rerun after ``Features.refit_families``."""
        f = self.f
        self.n_fam = int(f.family_tcs.shape[1])
        self.fam = (
            interp_rows(f.family_tcs, self.t)
            if hasattr(self, "t")
            else torch.zeros(self.n_fam, device=self.device)
        )
        counts = torch.bincount(f.family, minlength=self.n_fam).float()
        sal = torch.zeros(self.n_fam, device=self.device).index_add_(0, f.family, f.salience)
        self.family_salience = sal / counts.clamp_min(1)
        self.family_size = counts / counts.sum()
        self._color_key = None
        self.mute = torch.zeros(self.n_fam, dtype=torch.bool, device=self.device)
        self.mute_host = [False] * self.n_fam
        self.any_muted = False
        self._mute_version = getattr(self, "_mute_version", 0) + 1
        self._family_rgb_n = None
        self._gs_np = f.global_signal.float().cpu().numpy()
        self._dv_np = f.dvars.float().cpu().numpy()
        # Each family as an oscillator: the analytic signal of its time course
        # (periods under 6 s removed), for an instantaneous phase and envelope.
        from .features import analytic_band

        an = analytic_band(f.family_tcs, f.tr)
        self._fam_re = an.real.contiguous()
        self._fam_im = an.imag.contiguous()
        self.fam_phase = torch.zeros(self.n_fam, device=self.device)
        self.fam_amp = torch.zeros(self.n_fam, device=self.device)
        self.sync = 0.0

    def auto_mute(self, on: bool) -> None:
        """Mute (or unmute) every family flagged as noise-like."""
        flags = self.f.extras.get("family_flags", [[]] * self.n_fam)
        self.set_mutes([bool(on and fl) for fl in flags])

    def set_mutes(self, muted: list[bool]) -> None:
        """Replace the mute set; ``any_muted`` is a host flag so frames need no sync."""
        self.mute = torch.tensor(muted, dtype=torch.bool, device=self.device)
        self.mute_host = list(muted)
        self.any_muted = any(muted)
        self._mute_version += 1

    def set_mute(self, family: int, muted: bool) -> None:
        flags = list(self.mute_host)
        flags[family] = bool(muted)
        self.set_mutes(flags)

    def voxel_mute(self) -> torch.Tensor:
        """(N,) brightness factor: muted families' voxels sink to a faint ember."""
        return torch.where(self.mute[self.f.family], 0.08, 1.0)

    @property
    def n_time(self) -> int:
        return self.f.n_time

    def seek(self, t: float) -> None:
        self.t = float(t) % (self.n_time - 1)

    def activity(self, t: float) -> torch.Tensor:
        """Raw z blended toward the low-rank part (correlation pooling)."""
        raw = interp_rows(self.f.z, t)
        c = self.p["clarity"]
        if c <= 0:
            return raw
        # Unscaled on purpose: a voxel's low-rank part keeps the variance its
        # correlated partners vouch for and loses the rest.
        recon = self.f.loadings @ interp_rows(self.f.tcs, t)
        return raw * (1 - c) + recon * c

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        """Blend each voxel's value with its neighbourhood's (``reach`` passes)."""
        mix = self.p["pool"]
        if mix <= 0:
            return x
        local = x
        if x.device.type == "cpu":
            # A sparse (I + adjacency) / 7 product: 0.9 ms a pass against 1.7
            # for the gather on a 6-core CPU. Elsewhere the gather is as fast
            # (CUDA) or the sparse product is routed off the device (Metal).
            a = self._pool_matrix()
            for _ in range(int(self.p["reach"])):
                local = (a @ local[:, None])[:, 0]
        else:
            nb = self.f.extras["neighbours"]
            for _ in range(int(self.p["reach"])):
                local = (local + local[nb].sum(1)) / 7.0
        # No rescale: averaging shrinks independent noise and leaves a spatially
        # coherent response at full size, and that contrast is the point -- most
        # of the brain carries no task signal.
        return x * (1 - mix) + local * mix

    def weave(self, aspect: float) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Every voxel on a full rectangular grid, correlated voxels adjacent.

        A batch self-organising map (Kohonen) is trained on each voxel's
        correlation profile (its row of the PCA loadings, unit length) over a
        small grid of nodes with the screen's aspect; each voxel takes its
        best-matching node plus a sub-node offset toward its second best, and a
        rank grid over those coordinates tiles the screen exactly. A SOM keeps
        topology locally, which a 2-D linear projection cannot: ranking the
        first two PCA axes of the profiles put uncorrelated voxels side by side
        and drew the weave as salt-and-pepper noise. Returns (row, col)
        fractions in [0, 1) and the grid's (rows, cols).
        """
        key = round(aspect, 3)
        if getattr(self, "_weave_key", None) != key:
            f = self.f
            n = f.n_voxels
            xy = self._som_coords(aspect)
            cols = max(1, round(math.sqrt(n * aspect)))
            rows = math.ceil(n / cols)
            order = torch.argsort(xy[:, 0])
            col = torch.empty(n, dtype=torch.long, device=self.device)
            col[order] = torch.arange(n, device=self.device) // rows
            # Within each column, rank by the second axis: sort by it, then a
            # stable sort by column (a float64 composite key raises on Metal).
            by_y = torch.argsort(xy[:, 1])
            o2 = by_y[torch.sort(col[by_y], stable=True).indices]
            row = torch.empty(n, dtype=torch.long, device=self.device)
            row[o2] = torch.arange(n, device=self.device) % rows
            self._weave = ((row.float() + 0.5) / rows, (col.float() + 0.5) / cols, rows, cols)
            self._weave_key = key
        return self._weave

    def _som_coords(self, aspect: float, nodes: int = 2400, epochs: int = 12) -> torch.Tensor:
        """(N, 2) continuous SOM coordinates (x across, y down) of every voxel."""
        f = self.f
        data = f.loadings / f.loadings.norm(dim=1, keepdim=True).clamp_min(1e-9)
        gw = max(2, round(math.sqrt(nodes * aspect)))
        gh = max(2, round(nodes / gw))
        gy, gx = torch.meshgrid(
            torch.arange(gh, device=self.device, dtype=torch.float32),
            torch.arange(gw, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        pos = torch.stack([gx.reshape(-1), gy.reshape(-1)], 1)  # (M, 2)
        # Initialise on the profiles' top two axes, so the map starts unfolded.
        sub = data[torch.randperm(len(data), device=self.device)[: min(len(data), 60000)]]
        mean = sub.mean(0)
        _, _, vh = plinalg.svd(sub - mean, full_matrices=False)
        score = (sub - mean) @ vh[:2].T
        sd = score.std(0)
        w = (
            mean
            + ((pos[:, :1] / (gw - 1) - 0.5) * 4 * sd[0]) * vh[0]
            + ((pos[:, 1:] / (gh - 1) - 0.5) * 4 * sd[1]) * vh[1]
        )
        # Batch SOM: neighbourhood shrinks geometrically from a third of the map
        # to under a node.
        s0, s1 = max(gw, gh) / 3, 0.7
        dist2 = torch.cdist(pos, pos).square()
        for e in range(epochs):
            sigma = s0 * (s1 / s0) ** (e / max(1, epochs - 1))
            bmu = (sub @ w.T).argmax(1)  # unit vectors: max cosine = nearest
            # Exactly the batch update, through per-node sums: (M, M) @ (M, K)
            # instead of an (M, n_samples) neighbourhood matrix (~0.6 GB).
            sums = torch.zeros_like(w).index_add_(0, bmu, sub)
            counts = torch.bincount(bmu, minlength=len(w)).float()
            h = torch.exp(-dist2 / (2 * sigma**2))
            w = (h @ sums) / (h @ counts).clamp_min(1e-9)[:, None]
            w = w / w.norm(dim=1, keepdim=True).clamp_min(1e-9)
        out = torch.empty(len(data), 2, device=self.device)
        for a in range(0, len(data), 65536):
            sim = data[a : a + 65536] @ w.T
            top = sim.topk(2, dim=1)
            p1, p2 = pos[top.indices[:, 0]], pos[top.indices[:, 1]]
            # A third of the way toward the runner-up: order inside a node.
            out[a : a + 65536] = p1 + (p2 - p1) * 0.33
        return out

    def _pool_matrix(self) -> torch.Tensor:
        """(N, N) CSR of (identity + face adjacency) / 7, built once."""
        if getattr(self, "_pool_csr", None) is None:
            nb = self.f.extras["neighbours"]
            n = nb.shape[0]
            rows = torch.arange(n, device=nb.device)[:, None].expand(-1, 7).reshape(-1)
            cols = torch.cat([torch.arange(n, device=nb.device)[:, None], nb], 1).reshape(-1)
            vals = torch.full((7 * n,), 1.0 / 7.0, device=nb.device)
            coo = torch.sparse_coo_tensor(
                torch.stack([rows, cols]), vals, (n, n), check_invariants=False
            )
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
                self._pool_csr = coo.coalesce().to_sparse_csr()
        return self._pool_csr

    def shown_series_voxel(self, v: int) -> torch.Tensor:
        """(T,) what the scenes draw for voxel ``v`` at each TR, exactly.

        Pooling ``reach`` passes over face neighbours only ever reaches voxels
        within ``reach`` steps, so this pools that small ball (a few hundred
        voxels) instead of the whole brain at every TR -- which took 1.6 s on
        the CPU each time the inspector opened or a pooling setting moved. The
        ball's outer shell borrows its own value for neighbours outside it; that
        only corrupts the shell after the first pass, which ``v`` never reads.
        """
        p, f = self.p, self.f
        # The ball is a few hundred indices: built on the CPU (unique and
        # searchsorted need no Metal question), then the data is read on-device.
        nb = f.extras["neighbours"]
        nb_cpu = getattr(self, "_nb_cpu", None)
        if nb_cpu is None or nb_cpu.shape != nb.shape:
            nb_cpu = self._nb_cpu = nb.cpu()
        reach = int(p["reach"]) if p["pool"] > 0 else 0
        ball = torch.tensor([int(v)])
        for _ in range(reach):
            ball = torch.unique(torch.cat([ball, nb_cpu[ball].reshape(-1)]))
        centre = int((ball == v).nonzero()[0, 0])
        ball_dev = ball.to(self.device)
        act = f.z[:, ball_dev].float()  # (T, B)
        c = p["clarity"]
        if c > 0:
            act = act * (1 - c) + (f.tcs @ f.loadings[ball_dev].T) * c
        if reach == 0:
            return act[:, centre]
        # Neighbour table in ball-local indices; outside the ball -> itself.
        nbb = nb_cpu[ball]
        pos = torch.searchsorted(ball, nbb).clamp(max=len(ball) - 1)
        inside = ball[pos] == nbb
        local_nb = torch.where(inside, pos, torch.arange(len(ball))[:, None]).to(self.device)
        loc = act
        for _ in range(reach):
            loc = (loc + loc[:, local_nb].sum(-1)) / 7.0
        out = act * (1 - p["pool"]) + loc * p["pool"]
        return out[:, centre]

    def advance(self, dt: float, playing: bool = True) -> None:
        self.wall += dt
        self._frame = getattr(self, "_frame", 0) + 1
        if playing:
            self.t = (self.t + self.p["speed"] * dt) % (self.n_time - 1)
        a = self.pooled(self.activity(self.t))
        k = self.p["smooth"]
        self.live = self.live * k + a * (1 - k)
        # Host-side: two floats a frame read off the GPU were two syncs a frame.
        self.g = float(_interp_np(self._gs_np, self.t))
        self.d = float(_interp_np(self._dv_np, self.t))
        self.fam = interp_rows(self.f.family_tcs, self.t)
        if self.any_muted:
            self.fam = torch.where(self.mute, torch.zeros_like(self.fam), self.fam)
        re, im = interp_rows(self._fam_re, self.t), interp_rows(self._fam_im, self.t)
        self.fam_phase = torch.atan2(im, re)
        self.fam_amp = (re**2 + im**2).sqrt()
        if self.any_muted:
            self.fam_amp = torch.where(self.mute, torch.zeros_like(self.fam_amp), self.fam_amp)
        # Kuramoto order parameter, envelope-weighted: 1 when the families
        # swing in step, near 0 when their phases are scattered.
        w = self.fam_amp / self.fam_amp.sum().clamp_min(1e-6)
        # Stays on the device: scenes only multiply by it.
        # Real arithmetic, not exp(1j*phase): no complex tensor every frame.
        self.sync = (
            (w * torch.cos(self.fam_phase)).sum() ** 2 + (w * torch.sin(self.fam_phase)).sum() ** 2
        ).sqrt()
        # A slow shadow of the live value: live minus shadow is the trend, and
        # (live, trend) is a voxel's own instantaneous phase.
        if not hasattr(self, "shadow"):
            self.shadow = self.live.clone()
        self.shadow = self.shadow + (self.live - self.shadow) * min(1.0, dt * 1.5)

    # --------------------------------------------------------------- shading

    def gate(self) -> torch.Tensor:
        # Keyed on the salience tensor too: the components slider replaces it.
        key = (self.p["gate"], id(self.f.salience))
        if key != self._gate_key:
            self._gate_w = self.f.salience ** key[0]
            self._gate_key = key
        return self._gate_w

    def _frame_key(self, *names: str) -> tuple:
        return (getattr(self, "_frame", 0), self._mute_version, *(self.p[n] for n in names))

    def brightness(self, lv: torch.Tensor | None = None) -> torch.Tensor:
        """Per-voxel glow from the live value: threshold, gain, salience gate.

        Cached per frame: most scenes ask several times (picking weights,
        fields, splats), and each ask was a handful of full-length kernels.
        """
        if lv is None:
            key = self._frame_key("threshold", "negative", "gain", "gate")
            if getattr(self, "_bright_key", None) != key:
                self._bright = self._brightness(self.live)
                self._bright_key = key
            return self._bright
        return self._brightness(lv)

    def _brightness(self, lv: torch.Tensor) -> torch.Tensor:
        p = self.p
        thr = p["threshold"]
        up = (lv - thr).clamp_min(0)
        down = (-lv - thr).clamp_min(0) * p["negative"]
        out = (up + down) * p["gain"]
        if p["gate"] > 0:
            out = out * self.gate()
        if self.any_muted:
            out = out * self.voxel_mute()
        return out

    def base_colors(self) -> torch.Tensor:
        """(N, 3) per-voxel colour from the chosen source, luminance-normalised."""
        p = self.p
        src = COLOR_SOURCES[int(p["color"])]
        key = (src, int(p["wave"]))
        if key != self._color_key:
            f = self.f
            n = f.n_voxels
            one = torch.ones(n, device=self.device)
            if src == "family":
                hue = family_hues(self.n_fam).to(self.device)[f.family]
                rgb = hsv_to_rgb(hue, 0.35 + 0.5 * f.family_weight, one)
            elif src == "spectrum":
                b = f.extras["bands"]
                rgb = (b / b.amax(1, keepdim=True).clamp_min(1e-6)) ** 1.5
                rgb = rgb[:, [2, 1, 0]] * 0.8 + 0.1  # fast -> red, mid -> green, slow -> blue
            elif src == "wave":
                w = min(int(p["wave"]), f.phase.shape[1] - 1)
                hue = (f.phase[:, w] + math.pi) / (2 * math.pi)
                rgb = hsv_to_rgb(hue, 0.15 + 0.8 * f.phase_power[:, w], one)
            elif src == "tsnr":
                rgb = ramp(f.tsnr, GLACIER)
            elif src == "coherence":
                rgb = ramp(f.coherence.clamp(0, 0.6) / 0.6, EMBER)
            elif src == "mean":
                rgb = ramp(f.mean, GLACIER)
            else:
                rgb = torch.ones(n, 3, device=self.device)
            self._color = rgb / luma(rgb).clamp_min(0.15) * 0.7
            self._color_key = key
        return self._color

    def colors(self, lv: torch.Tensor | None = None) -> torch.Tensor:
        """Per-voxel colour for this frame: rising voxels warm, falling cool (cached per frame)."""
        if lv is None:
            key = self._frame_key("color", "wave")
            if getattr(self, "_colors_key", None) != key:
                self._colors = self._colors_for(self.live)
                self._colors_key = key
            return self._colors
        return self._colors_for(lv)

    def _colors_for(self, lv: torch.Tensor) -> torch.Tensor:
        src = COLOR_SOURCES[int(self.p["color"])]
        if src == "phase":
            # Hue = angle of (live, trend): rising, peak, falling, trough go
            # round the wheel, so a travelling wave sweeps across as a rainbow.
            trend = (self.live - getattr(self, "shadow", self.live)) * 3.0
            hue = (torch.atan2(trend, self.live) / (2 * math.pi)) % 1.0
            rgb = hsv_to_rgb(hue, torch.full_like(hue, 0.75), torch.ones_like(hue))
            return rgb / luma(rgb).clamp_min(0.15) * 0.7
        base = self.base_colors()
        if src == "sign":
            warm, cool = (
                self._const("warm", (1.0, 0.55, 0.2)),
                self._const("cool", (0.25, 0.55, 1.0)),
            )
            return torch.where((lv >= 0)[:, None], warm, cool)
        cool = base * self._const("chill", (0.45, 0.7, 1.3))
        return torch.where((lv >= 0)[:, None], base, cool)

    def _const(self, name: str, values: tuple[float, ...]) -> torch.Tensor:
        """A small constant on the device, made once (each fresh one was a blocking copy)."""
        cache = self.__dict__.setdefault("_consts", {})
        if name not in cache:
            cache[name] = torch.tensor(values, device=self.device)
        return cache[name]

    def family_rgb(self) -> torch.Tensor:
        """(F, 3) one colour per family, matching ``colors`` under 'family' (cached)."""
        if getattr(self, "_family_rgb_n", None) != self.n_fam:
            hue = family_hues(self.n_fam).to(self.device)
            rgb = hsv_to_rgb(hue, torch.full_like(hue, 0.7), torch.ones_like(hue))
            self._family_rgb = rgb / luma(rgb).clamp_min(0.15) * 0.7
            self._family_rgb_n = self.n_fam
        return self._family_rgb

    def finish(self, accum: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Tone-map an (h*w, 3) linear buffer to (h, w, 3) uint8 with the global effects."""
        p = self.p
        # Global signal shifts colour temperature; DVARS spikes flash.
        tint = math.tanh(self.g * 0.5) * p["tint"] * 0.25
        flash = 1.0 + max(0.0, self.d - 2.0) * p["flash"] * 0.25
        x = accum * (p["exposure"] * flash)
        # Colour temperature by scalar per channel: no device tensor a frame.
        x = torch.stack([x[:, 0] * max(0.0, 1 + tint), x[:, 1], x[:, 2] * max(0.0, 1 - tint)], 1)
        out = 1.0 - torch.exp(-x)
        if p["saturation"] != 1.0:
            lum = luma(out)
            out = lum + (out - lum) * p["saturation"]
        # Gamma and quantisation by one table lookup: a pow over every pixel
        # channel was the slowest step of the tone map on the CPU.
        if getattr(self, "_gamma_lut", None) is None:
            ramp = torch.linspace(0, 1, 4096, device=self.device)
            self._gamma_lut = (ramp ** (1 / 1.8) * 255 + 0.5).to(torch.uint8)
        idx = (out.clamp(0, 1) * 4095).to(torch.int32)
        return self._gamma_lut[idx].reshape(h, w, 3)
