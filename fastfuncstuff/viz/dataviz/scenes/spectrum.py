"""Spectrum: a Winamp spectrum analyser for the brain's networks.

Each ICA family gets a group of bars: the power spectrum of its time course
over a sliding window ending now (Hann-tapered), slowest frequencies on the
left of each group. Bars are drawn as stacked LED segments with peak caps that
fall under gravity and a dim reflection below, and the same spectra scroll up
the background as a waterfall. A block design lights the same bar in the
responding families at every block; physiology aliased into the fast band
sits at the right of every group; a quiet family's bars stay low.
"""

from __future__ import annotations

import torch

from ..drives import Param, interp_many
from .base import Scene, representative

S = "spectrum"


class SpectrumScene(Scene):
    name = S
    blurb = "a Winamp spectrum analyser: one group of bars per family"
    PARAMS = (
        Param("span", 32, 8, 128, S, "spectral window, TRs", scene=S),
        Param("bins", 12, 4, 32, S, "bars per family", scene=S),
        Param("drop", 1.5, 0.1, 6.0, S, "peak-cap fall speed", scene=S),
        Param("waterfall", 0.5, 0.0, 1.5, S, "brightness of the scrolling spectrogram", scene=S),
        Param("leds", 1, 0, 1, S, "draw bars as LED segments", scene=S),
    )

    def __init__(self, drives, width, height, seed=0):
        super().__init__(drives, width, height, seed)
        self.caps = None
        self.ref = None
        self.fall = torch.zeros(1, self.height, self.width, device=self.device)
        self.rep = representative(self)

    def on_resize(self):
        self.fall = torch.zeros(1, self.height, self.width, device=self.device)

    def _spectra(self) -> torch.Tensor:
        p, dr = self.p, self.dr
        n = int(p["span"])
        b = min(int(p["bins"]), n // 2)
        ts = [dr.t - n + 1 + i for i in range(n)]
        seg = interp_many(self.f.family_tcs, [max(0.0, u) for u in ts])  # (n, F)
        seg = (seg - seg.mean(0)) * torch.hann_window(n, device=self.device)[:, None]
        pw = torch.fft.rfft(seg, dim=0).abs()[1 : b + 1].T  # (F, b)
        # Each family against its own recent loudest bar: the spectrum keeps its
        # shape. (Per-bar normalising filled every bar to the top.)
        loud = pw.amax(1)
        if self.ref is None or self.ref.shape != loud.shape:
            self.ref = loud.clone().clamp_min(1e-3)
        self.ref = torch.maximum(self.ref * 0.997, loud)
        return (pw / self.ref[:, None].clamp_min(1e-3)).clamp(0, 1) ** 0.7

    def step(self, dt: float) -> None:
        spec = self._spectra()
        if self.caps is None or self.caps.shape != spec.shape:
            self.caps = spec.clone()
        self.caps = torch.maximum(self.caps - dt * self.p["drop"] * 0.5, spec)
        self.spec = spec

    def render(self) -> torch.Tensor:
        p, dr = self.p, self.dr
        h, w = self.height, self.width
        spec = getattr(self, "spec", None)
        if spec is None:
            self.step(0.0)
            spec = self.spec
        n_f, b = spec.shape
        bars = n_f * b
        gap = 2  # columns between groups
        bar_w = max(1, (int(w * 0.9) - gap * (n_f - 1)) // bars)
        total = bars * bar_w + gap * (n_f - 1)
        x0 = (w - total) // 2
        xs = torch.arange(w, device=self.device)
        rel = xs - x0
        group = (rel // (b * bar_w + gap)).clamp(0, n_f - 1)
        within = rel - group * (b * bar_w + gap)
        bar = (within // bar_w).clamp(0, b - 1)
        on_bar = (
            (rel >= 0)
            & (rel < total)
            & (within < b * bar_w)
            & ((within % bar_w) < max(1, bar_w - 1))
        )
        base = int(h * 0.68)
        top = int(h * 0.62)
        height = (spec[group, bar] * top).round()
        caps = (self.caps[group, bar] * top).round()
        ys = torch.arange(h, device=self.device)[:, None]
        up = base - ys  # pixels above the baseline
        lit = (up >= 0) & (up < height[None]) & on_bar[None]
        if int(p["leds"]) == 1:
            lit &= (up % 4) != 3
        cap = (up == caps[None]) & on_bar[None]
        refl = (up < 0) & (-up < height[None] * 0.4) & on_bar[None] & ((-up % 3) != 2)
        col = dr.family_rgb()[group]  # (w, 3)
        # Hotter toward the top of each bar, like the classic gradient.
        heat = (up.float() / top).clamp(0, 1)[..., None]
        bar_rgb = col[None] * (0.5 + 0.8 * heat) + heat**3 * 0.6
        img = torch.zeros(h, w, 3, device=self.device)
        img = torch.where(lit[..., None], bar_rgb, img)
        img = torch.where(refl[..., None], col[None] * 0.18, img)
        img = torch.where(cap[..., None], torch.ones(3, device=self.device) * 0.9, img)
        # Waterfall: the spectra scroll up behind everything.
        row = (spec[group, bar] * on_bar)[None, None]  # (1, 1, w)
        self.fall = torch.roll(self.fall, -1, 1)
        self.fall[:, -1] = row[0, 0]
        wf = self.fall[0, :, :, None] * col[None] * p["waterfall"] * 0.35
        img = torch.where(lit[..., None] | cap[..., None] | refl[..., None], img, wf)
        xc = (
            x0
            + (torch.arange(n_f, device=self.device) * (b * bar_w + gap) + b * bar_w // 2).float()
        )
        self.remember(xc, torch.full_like(xc, base - top * 0.3), self.rep)
        return dr.finish(img.reshape(-1, 3) * 1.6, h, w)
