"""InstaCorr: the overlay is a seed correlation, recomputed as you click.

The measurement this is built around, on an M4 Max: a seed against 900k voxels
by 1000 time points is 9.3 ms in float32 and 5.4 ms in float16. That is memory
bandwidth, not arithmetic -- the correlation reads the whole prepared array once
and does one multiply-add per element. It is already faster than the display
refreshes, so there is no reason to subsample or to restrict the map to a slab.

The cost that matters is therefore *preparation*, not correlation: detrending,
bandpassing and blurring a whole 4-D dataset takes seconds. So preparation is
cached and only redone when a parameter that affects it changes, while moving
the seed re-runs nothing but the one matrix-vector product.
"""

from __future__ import annotations

import numpy as np
import torch

from fastfuncstuff.viewer.commands import Aspect, Command
from fastfuncstuff.viewer.modes.base import (
    ComputedOverlay,
    Control,
    FloatControl,
    IntControl,
    Mode,
    OverlayKind,
    Trace,
    mode,
)
from fastfuncstuff.viewer.vocab import SetSeed


@mode
class InstaCorrMode(Mode):
    name = "instacorr"
    label = "InstaCorr"
    overlay_kind = OverlayKind.CORRELATION

    def controls(self) -> tuple[Control, ...]:
        return (
            IntControl(
                name="polort",
                label="polort",
                lo=-1,
                hi=6,
                default=2,
                help="Legendre detrend order; -1 disables.",
            ),
            FloatControl(
                name="fbot",
                label="highpass",
                lo=0.0,
                hi=0.2,
                default=0.01,
                step=0.005,
                unit="Hz",
            ),
            FloatControl(
                name="ftop",
                label="lowpass",
                lo=0.0,
                hi=0.5,
                default=0.10,
                step=0.005,
                unit="Hz",
                help="0 disables the low-pass edge.",
            ),
            FloatControl(
                name="blur",
                label="blur",
                lo=0.0,
                hi=12.0,
                default=4.0,
                step=0.5,
                unit="mm",
            ),
            FloatControl(
                name="seed_radius",
                label="seed r",
                lo=0.0,
                hi=14.0,
                default=6.0,
                step=1.0,
                unit="mm",
                help="Average the seed over a sphere, as AFNI does. 0 uses one voxel.",
            ),
        )

    # -- source --------------------------------------------------------
    def _source_layer(self):
        """The 4-D layer to correlate. The first time-linked layer in the stack."""
        if self.session is None:
            return None
        for layer in self.session.state.layers:
            if layer.time_linked and not layer.is_computed:
                return layer
        return None

    def input_layer_key(self) -> str | None:
        layer = self._source_layer()
        return None if layer is None else layer.key

    # -- preparation ---------------------------------------------------
    def _prepare(self) -> bool:
        """Detrend, bandpass, blur and normalize once. Returns readiness."""
        if not self._dirty and getattr(self, "_prepared", None) is not None:
            return True
        session = self.session
        layer = self._source_layer()
        if session is None or layer is None:
            return False

        res = session.store.get(layer.key)
        if res.array is None:
            # Still inflating. Returning False rather than blocking keeps the
            # click that triggered this from freezing the window.
            return False

        device = session.store.device
        arr = torch.as_tensor(np.ascontiguousarray(res.array), dtype=torch.float32)
        nx, ny, nz, nt = arr.shape

        if float(self.params.get("blur") or 0.0) > 0:
            from fastfuncstuff.stats.smooth3d import fwhm_mm_to_sigma_vox, gaussian3d_batched

            zooms = tuple(float(abs(layer.affine[i, i])) or 1.0 for i in range(3))
            sigma = fwhm_mm_to_sigma_vox(float(self.params["blur"]), zooms)
            # (T, X, Y, Z): the batched smoother wants time on the leading axis.
            vol = arr.permute(3, 0, 1, 2).to(device)
            vol = gaussian3d_batched(vol, sigma)
            data = vol.reshape(nt, -1)
        else:
            data = arr.reshape(-1, nt).T.contiguous().to(device)

        data = self._detrend(data, layer)
        data = self._bandpass(data, layer)

        data = data - data.mean(0, keepdim=True)
        norm = data.norm(dim=0, keepdim=True)
        # Constant voxels (outside the brain, mostly) would divide by zero and
        # then correlate perfectly with everything.
        self._valid = (norm > 1e-9).squeeze(0)
        data = data / norm.clamp(min=1e-9)
        data = torch.where(self._valid.unsqueeze(0), data, torch.zeros_like(data))

        self._prepared = data
        self._shape = (nx, ny, nz)
        self._affine = layer.affine
        self._layer_key = layer.key
        self._dirty = False
        return True

    def _detrend(self, data: torch.Tensor, layer) -> torch.Tensor:
        """Project out Legendre polynomials -- never raw monomials."""
        order = int(self.params.get("polort", 2))
        if order < 0:
            return data
        from fastfuncstuff.glm.core import construct_polynomial_matrix

        nt = data.shape[0]
        poly = construct_polynomial_matrix(nt, order, data.device, data.dtype)
        q, _ = torch.linalg.qr(poly)
        return data - q @ (q.T @ data)

    def _bandpass(self, data: torch.Tensor, layer) -> torch.Tensor:
        """Zero the rFFT bins outside the band.

        Needs a TR. A dataset without one cannot be bandpassed in Hz at all, so
        the filter is skipped rather than applied against an assumed 1 s.
        """
        fbot = float(self.params.get("fbot") or 0.0)
        ftop = float(self.params.get("ftop") or 0.0)
        if fbot <= 0.0 and ftop <= 0.0:
            return data
        tr = self._tr(layer)
        if tr <= 0.0:
            return data
        nt = data.shape[0]
        freqs = torch.fft.rfftfreq(nt, d=tr).to(data.device)
        keep = torch.ones_like(freqs, dtype=torch.bool)
        if fbot > 0:
            keep &= freqs >= fbot
        if ftop > 0:
            keep &= freqs <= ftop
        keep[0] = False  # the mean is handled by centring, not by the filter
        spec = torch.fft.rfft(data, dim=0)
        spec = spec * keep.unsqueeze(1)
        return torch.fft.irfft(spec, n=nt, dim=0)

    def _tr(self, layer) -> float:
        if self.session is None:
            return 0.0
        try:
            return float(self.session.store.get(layer.key).info.tr)
        except KeyError:
            return 0.0

    # -- seed ----------------------------------------------------------
    def _seed_timecourse(self, data: torch.Tensor) -> torch.Tensor | None:
        """Seed signal, averaged over a sphere the way AFNI does.

        A single voxel is noisy enough that the map changes character as you
        move one step; the sphere is why AFNI's InstaCorr looks stable.
        """
        session = self.session
        if session is None or session.state.seed is None:
            return None
        nx, ny, nz = self._shape
        si, sj, sk = session.state.seed
        radius = float(self.params.get("seed_radius") or 0.0)
        zooms = [float(abs(self._affine[i, i])) or 1.0 for i in range(3)]

        if radius <= 0.0:
            if not (0 <= si < nx and 0 <= sj < ny and 0 <= sk < nz):
                return None
            flat = (si * ny + sj) * nz + sk
            return data[:, flat] if bool(self._valid[flat]) else None

        rad_vox = [max(0, int(radius / z)) for z in zooms]
        ii = torch.arange(max(0, si - rad_vox[0]), min(nx, si + rad_vox[0] + 1))
        jj = torch.arange(max(0, sj - rad_vox[1]), min(ny, sj + rad_vox[1] + 1))
        kk = torch.arange(max(0, sk - rad_vox[2]), min(nz, sk + rad_vox[2] + 1))
        if not (ii.numel() and jj.numel() and kk.numel()):
            return None
        gi, gj, gk = torch.meshgrid(ii, jj, kk, indexing="ij")
        dist = torch.sqrt(
            ((gi - si) * zooms[0]) ** 2 + ((gj - sj) * zooms[1]) ** 2 + ((gk - sk) * zooms[2]) ** 2
        )
        inside = dist <= radius
        flat = ((gi * ny + gj) * nz + gk)[inside].reshape(-1).to(data.device)
        flat = flat[self._valid[flat]]
        if flat.numel() == 0:
            return None
        seed = data[:, flat].mean(1)
        seed = seed - seed.mean()
        n = seed.norm()
        return seed / n if float(n) > 1e-9 else None

    # -- production ----------------------------------------------------
    def compute(self) -> ComputedOverlay | None:
        if not self._prepare():
            return None
        data = self._prepared
        seed = self._seed_timecourse(data)
        if seed is None:
            return None
        # The whole correlation: one mat-vec over normalized columns.
        r = seed @ data
        vol = r.reshape(self._shape).cpu().numpy()
        return ComputedOverlay(
            values=np.nan_to_num(vol, nan=0.0),
            affine=self._affine,
            name="instacorr",
            kind=OverlayKind.CORRELATION,
            colormap="redblue",
            display_range=(-1.0, 1.0),
            threshold=0.3,
        )

    # -- reaction ------------------------------------------------------
    def on_command(self, cmd: Command, dirty: Aspect) -> Aspect:
        if isinstance(cmd, SetSeed):
            return self.refresh()
        return Aspect.NOTHING

    def series(self, ijk: tuple[int, int, int]) -> list[Trace]:
        """The prepared (filtered) signal, which is what was actually correlated."""
        if getattr(self, "_prepared", None) is None:
            return []
        nx, ny, nz = self._shape
        i, j, k = ijk
        if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
            return []
        flat = (i * ny + j) * nz + k
        vals = self._prepared[:, flat].cpu().numpy()
        return [Trace(label="instacorr (filtered)", values=vals, x_label="TR")]

    def status(self) -> str:
        if self.session is None or self.session.state.seed is None:
            return "instacorr: ctrl-click to set a seed"
        if getattr(self, "_prepared", None) is None:
            return "instacorr: preparing…"
        return f"instacorr: seed {self.session.state.seed}"
