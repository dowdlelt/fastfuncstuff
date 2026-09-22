"""InstaCorr: the overlay is a seed correlation, recomputed as you click.

The measurement this is built around, on an M4 Max: a seed against 900k voxels
by 1000 time points is 9.3 ms in float32. That is memory bandwidth, not
arithmetic -- the correlation reads the prepared array once and does one
multiply-add per element. It is already faster than the display refreshes.

The cost that matters is therefore preparation, not correlation. Detrending,
bandpassing and blurring a real dataset takes seconds, so it is split out into
:meth:`prepare`, which the UI runs on a worker behind a progress bar. Moving the
seed afterwards runs nothing but one matrix-vector product.

Preparation captures what it needs from the source dataset rather than holding
the layer. A mode's overlay replaces the primary overlay, which is often the
layer the mode was computed from; keeping a reference to the array means that
displacement cannot leave the mode unable to re-prepare.
"""

from __future__ import annotations

import numpy as np
import torch

from fastfuncstuff.viewer.commands import Aspect, Command
from fastfuncstuff.viewer.modes.base import (
    ComputedOverlay,
    Control,
    IntControl,
    Mode,
    OptionalFloatControl,
    OverlayKind,
    ProgressFn,
    Trace,
    mode,
)
from fastfuncstuff.viewer.vocab import SetSeed


@mode
class InstaCorrMode(Mode):
    name = "instacorr"
    label = "InstaCorr"
    tag = "ICORR"
    overlay_kind = OverlayKind.CORRELATION

    def controls(self) -> tuple[Control, ...]:
        # Defaults are deliberately minimal: detrend only. Bandpass and blur
        # change what the correlation means, so they are opt-in and visibly off
        # rather than quietly applied.
        return (
            IntControl(
                name="polort",
                label="detrend",
                lo=-1,
                hi=6,
                default=2,
                help="Legendre polynomial order; -1 disables detrending.",
            ),
            OptionalFloatControl(
                name="fbot",
                on_value=0.01,
                label="highpass",
                lo=0.0,
                hi=0.2,
                default=0.0,
                step=0.005,
                unit=" Hz",
                help="Off by default. Needs a TR to mean anything.",
            ),
            OptionalFloatControl(
                name="ftop",
                on_value=0.1,
                label="lowpass",
                lo=0.0,
                hi=0.5,
                default=0.0,
                step=0.005,
                unit=" Hz",
                help="Off by default. Needs a TR to mean anything.",
            ),
            OptionalFloatControl(
                name="blur",
                on_value=4.0,
                label="blur",
                lo=0.0,
                hi=12.0,
                default=0.0,
                step=0.5,
                unit=" mm",
                help="Spatial smoothing applied before correlating.",
            ),
            OptionalFloatControl(
                name="seed_radius",
                on_value=6.0,
                label="seed radius",
                lo=0.0,
                hi=14.0,
                default=0.0,
                step=1.0,
                unit=" mm",
                help="Average the seed over a sphere. Off means a single voxel.",
            ),
        )

    def preparation_params(self) -> frozenset[str]:
        # seed_radius only affects which columns are averaged at compute time,
        # so changing it must not trigger a multi-second re-preparation.
        return frozenset({"polort", "fbot", "ftop", "blur"})

    def __init__(self) -> None:
        self._prepared: torch.Tensor | None = None
        self._valid: torch.Tensor | None = None
        self._shape: tuple[int, int, int] = (0, 0, 0)
        self._affine: np.ndarray | None = None
        self._source: np.ndarray | None = None
        self._source_key: str | None = None
        self._tr = 0.0
        self._zooms = (1.0, 1.0, 1.0)
        super().__init__()

    # -- source --------------------------------------------------------
    def _capture_source(self) -> bool:
        """Take a reference to the chosen run and everything describing it.

        Held by reference rather than by layer key so that installing the
        correlation on top of the run it came from -- and hiding that run --
        cannot strand the mode.
        """
        layer = self.source_layer()
        if layer is None or self.session is None:
            return self._source is not None and self._source_key is not None
        if self._source is not None and self._source_key == layer.key:
            return True
        try:
            res = self.session.store.get(layer.key)
        except KeyError:
            return False
        if res.array is None:
            return False  # still inflating
        self._source = res.array
        self._source_key = layer.key
        self._affine = np.asarray(layer.affine, dtype=float)
        self._tr = float(res.info.tr)
        self._zooms = tuple(float(abs(layer.affine[i, i])) or 1.0 for i in range(3))
        self._dirty = True
        return True

    def input_layer_key(self) -> str | None:
        """The run the prepared array came from; see InstaGLM for why."""
        return self._source_key

    def attach(self, session) -> None:
        super().attach(session)
        self._capture_source()

    def detach(self) -> None:
        """Free the prepared array; it can be gigabytes of device memory.

        The instance is kept for its parameters, and the map stays in the
        stack, so coming back costs one re-preparation and nothing else.
        """
        self._prepared = None
        self._valid = None
        self._source = None
        self._source_key = None
        self._dirty = True
        super().detach()

    # -- preparation ---------------------------------------------------
    def prepare(self, progress: ProgressFn | None = None) -> bool:
        """Detrend, bandpass, blur and normalize. Seconds on a real dataset."""
        if not self._dirty and self._prepared is not None:
            return True
        if not self._capture_source() or self._source is None:
            return False

        def step(fraction: float, message: str) -> None:
            if progress is not None:
                progress(fraction, message)

        device = self.session.store.device if self.session is not None else torch.device("cpu")
        nx, ny, nz, nt = self._source.shape

        blur = float(self.params.get("blur") or 0.0)
        if blur > 0:
            step(0.1, f"blur {blur:g} mm")
            from fastfuncstuff.stats.smooth3d import fwhm_mm_to_sigma_vox, gaussian3d_batched

            sigma = fwhm_mm_to_sigma_vox(blur, self._zooms)
            vol = torch.as_tensor(self._source, dtype=torch.float32)
            vol = gaussian3d_batched(vol.permute(3, 0, 1, 2).to(device), sigma)
            # (T,X,Y,Z) -> (V,T). The one transpose in the pipeline, and only
            # on the opt-in path.
            data = vol.reshape(nt, -1).T.contiguous()
        else:
            # The store keeps the array C-contiguous, so this is a free view of
            # data that is already resident -- no copy, no upload on CPU.
            step(0.1, "arranging")
            data = torch.as_tensor(self._source, dtype=torch.float32).reshape(-1, nt)
            if device.type != "cpu":
                step(0.2, "to device")
                data = data.to(device)

        # Time is the last axis from here on, which is also the axis every
        # operation below runs along -- so each one reads contiguously.
        step(0.45, "detrend")
        data = self._detrend(data)
        step(0.65, "filter")
        data = self._bandpass(data)

        step(0.85, "normalize")
        data = data - data.mean(-1, keepdim=True)
        norm = data.norm(dim=-1, keepdim=True)
        # Constant voxels -- outside the brain, mostly -- would divide by zero
        # and then correlate perfectly with everything.
        self._valid = (norm > 1e-9).squeeze(-1)
        data = data / norm.clamp(min=1e-9)
        self._prepared = torch.where(self._valid.unsqueeze(-1), data, torch.zeros_like(data))
        self._shape = (nx, ny, nz)
        self._dirty = False
        step(1.0, "ready")
        return True

    def _detrend(self, data: torch.Tensor) -> torch.Tensor:
        """Project out Legendre polynomials -- never raw monomials."""
        order = int(self.params.get("polort", 2))
        if order < 0:
            return data
        from fastfuncstuff.glm.core import construct_polynomial_matrix

        poly = construct_polynomial_matrix(data.shape[-1], order, data.device, data.dtype)
        q, _ = torch.linalg.qr(poly)  # (T, k)
        return data - (data @ q) @ q.T

    def _bandpass(self, data: torch.Tensor) -> torch.Tensor:
        """Zero the rFFT bins outside the band.

        Skipped without a TR: a filter specified in Hz against an assumed 1 s
        sampling interval is not the filter anyone asked for.
        """
        fbot = float(self.params.get("fbot") or 0.0)
        ftop = float(self.params.get("ftop") or 0.0)
        if (fbot <= 0.0 and ftop <= 0.0) or self._tr <= 0.0:
            return data
        nt = data.shape[-1]
        freqs = torch.fft.rfftfreq(nt, d=self._tr).to(data.device)
        keep = torch.ones_like(freqs, dtype=torch.bool)
        if fbot > 0:
            keep &= freqs >= fbot
        if ftop > 0:
            keep &= freqs <= ftop
        keep[0] = False  # the mean is handled by centring, not by the filter
        spec = torch.fft.rfft(data, dim=-1) * keep
        return torch.fft.irfft(spec, n=nt, dim=-1)

    # -- seed ----------------------------------------------------------
    def _to_source(self, ijk: tuple[int, int, int]) -> tuple[int, int, int]:
        """A display-grid voxel as a voxel of the run being correlated.

        ``session.layer_voxel`` does the arithmetic; this exists because the
        run is held by reference rather than as a layer, so there is no layer
        to ask for the affine.
        """
        if self.session is None or self._affine is None:
            return ijk
        return self.session.layer_voxel(self._affine, ijk)

    def _seed_timecourse(self) -> torch.Tensor | None:
        data, valid = self._prepared, self._valid
        if data is None or valid is None or self.session is None:
            return None
        if self.session.state.seed is None:
            return None
        seed_ijk = self._to_source(self.session.state.seed)
        nx, ny, nz = self._shape
        si, sj, sk = seed_ijk
        radius = float(self.params.get("seed_radius") or 0.0)

        if radius <= 0.0:
            if not (0 <= si < nx and 0 <= sj < ny and 0 <= sk < nz):
                return None
            flat = (si * ny + sj) * nz + sk
            return data[flat] if bool(valid[flat]) else None

        rad = [max(0, int(radius / z)) for z in self._zooms]
        ii = torch.arange(max(0, si - rad[0]), min(nx, si + rad[0] + 1))
        jj = torch.arange(max(0, sj - rad[1]), min(ny, sj + rad[1] + 1))
        kk = torch.arange(max(0, sk - rad[2]), min(nz, sk + rad[2] + 1))
        if not (ii.numel() and jj.numel() and kk.numel()):
            return None
        gi, gj, gk = torch.meshgrid(ii, jj, kk, indexing="ij")
        dist = torch.sqrt(
            ((gi - si) * self._zooms[0]) ** 2
            + ((gj - sj) * self._zooms[1]) ** 2
            + ((gk - sk) * self._zooms[2]) ** 2
        )
        flat = ((gi * ny + gj) * nz + gk)[dist <= radius].reshape(-1).to(data.device)
        flat = flat[valid[flat]]
        if flat.numel() == 0:
            return None
        seed = data[flat].mean(0)
        seed = seed - seed.mean()
        n = seed.norm()
        return seed / n if float(n) > 1e-9 else None

    # -- production ----------------------------------------------------
    def compute(self) -> ComputedOverlay | None:
        if self._prepared is None or self._affine is None:
            return None
        seed = self._seed_timecourse()
        if seed is None:
            return None
        r = self._prepared @ seed  # the whole correlation, one mat-vec
        vol = r.reshape(self._shape).cpu().numpy()
        return ComputedOverlay(
            values=np.nan_to_num(vol, nan=0.0),
            affine=self._affine,
            name=self.output_name(),
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
        """The source signal and the prepared one that was actually correlated.

        The source is contributed here because the map displaces its own input
        from the layer stack, and the graph draws from the stack -- without
        this, setting a seed would make the very time course the correlation
        came from disappear from view.
        """
        i, j, k = self._to_source(ijk)
        if self._source is not None:
            nx, ny, nz, _ = self._source.shape
            if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
                return []
        out: list[Trace] = []
        if self._source is not None:
            out.append(
                Trace(
                    label="source",
                    key="source",
                    short="icorr source",
                    values=np.asarray(self._source[i, j, k, :], dtype=np.float32),
                    x_label="TR",
                )
            )
        if self._prepared is not None:
            nx, ny, nz = self._shape
            if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                flat = (i * ny + j) * nz + k
                out.append(
                    Trace(
                        label="prepared",
                        key="prepared",
                        short="icorr prepared",
                        values=self._prepared[flat].cpu().numpy(),
                        x_label="TR",
                    )
                )
        return out

    def residency(self) -> str:
        """Where the prepared data lives and how much of it there is.

        Worth surfacing: the whole design rests on the prepared array staying
        resident between clicks, and a status line that says so is the
        difference between trusting that and guessing.
        """
        if self._prepared is None:
            return ""
        gb = self._prepared.numel() * self._prepared.element_size() / 1e9
        return f"{gb:.2f} GB on {self._prepared.device.type}"

    def status(self) -> str:
        if self._source is None:
            return "instacorr: needs a 4-D dataset"
        if self._preparing:
            return "instacorr: preparing…"
        if self._prepared is None:
            return "instacorr: ready to prepare"
        where = self.residency()
        if self.session is None or self.session.state.seed is None:
            return f"instacorr: ctrl-click to set a seed  ·  {where}"
        return f"instacorr: seed {self.session.state.seed}  ·  {where}"
