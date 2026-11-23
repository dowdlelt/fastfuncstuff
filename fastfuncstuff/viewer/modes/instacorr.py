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

from fastfuncstuff.viewer import ortvec
from fastfuncstuff.viewer.commands import Aspect, Command
from fastfuncstuff.viewer.modes.base import (
    BoolControl,
    ChoiceControl,
    ComputedOverlay,
    Control,
    IntControl,
    Mode,
    OptionalFloatControl,
    OverlayKind,
    PathListControl,
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
    gated = True

    def controls(self) -> tuple[Control, ...]:
        # Defaults are deliberately minimal: detrend only. Bandpass and blur
        # change what the correlation means, so they are opt-in and visibly off
        # rather than quietly applied.
        return (
            *self.time_cut_controls(),
            IntControl(
                name="polort",
                label="detrend",
                newline=True,
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
            PathListControl(
                name="ortvec",
                label="ortvec",
                filter="1D / xmat (*.1D *.txt);;All (*)",
                newline=True,
                transforms=True,
                sample_interval=self.source_tr() or self._tr,
                help="Time courses projected out of every voxel before correlating -- a "
                "motion file, a respiration trace, a white-matter or CSF mean -- together "
                "with the detrend. Untick one to see what it was removing from the map. "
                "Under a bandpass they are filtered by the same band before they are "
                "projected out. d/dt and BANDS work as in InstaGLM.",
            ),
            BoolControl(
                name="automask",
                label="brain only",
                default=True,
                help="Correlate inside 3dAutomask's brain only. A third of the voxels, so a "
                "seed is a third of the time and memory; off for a phantom or a cropped run.",
            ),
            *self.gate_controls(),
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
            ChoiceControl(
                name="seed_roi",
                label="seed ROI",
                choices=("crosshair", *self._roi_choices()),
                default="crosshair",
                help="A painted (or loaded ROI) layer as the seed: the mean time course of "
                "its voxels, updated as you paint. Replaces the crosshair seed and radius.",
            ),
        )

    def _roi_choices(self) -> list[str]:
        if self.session is None:
            return []
        return [
            f"{ly.key} {ly.name}"
            for ly in self.session.state.layers
            if ly.roi or self.session.painter.owns(ly.key)
        ]

    def _roi_key(self) -> str | None:
        text = str(self.params.get("seed_roi") or "crosshair").split(" ", 1)[0]
        if text == "crosshair" or self.session is None:
            return None
        return text if self.session.state.layers.find(text) is not None else None

    def preparation_params(self) -> frozenset[str]:
        # seed_radius only affects which columns are averaged at compute time,
        # so changing it must not trigger a multi-second re-preparation.
        return frozenset(
            {"polort", "fbot", "ftop", "blur", "ortvec", "automask", "drop_first", "drop_last"}
        )

    def __init__(self) -> None:
        #: Prepared rows live in the session's store under this key (RAM, plus a
        #: device mirror while the budget allows) -- see :meth:`_rows`.
        self._hold_key: str | None = None
        #: ``(nx, ny, nz)`` int32: each voxel's row, -1 where nothing was kept.
        self._lookup: np.ndarray | None = None
        self._n_time = 0
        self._shape: tuple[int, int, int] = (0, 0, 0)
        self._affine: np.ndarray | None = None
        self._source: np.ndarray | None = None
        self._source_key: str | None = None
        self._tr = 0.0
        self._window = slice(0, None)
        #: Ortvec columns projected out, for the p's degrees of freedom.
        #: (samples, dimensions removed) of the last preparation, for the map's p.
        self._stat: tuple[float, float] = (0.0, 1.0)
        self._roi_message = ""
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
        from fastfuncstuff.viewer.tools.smooth import voxel_sizes

        # Column norms, not the diagonal: an oblique run's diagonal understates
        # each spacing, and a run stored with permuted axes has zeros there --
        # both of which mis-sized the blur and the seed sphere.
        self._zooms = voxel_sizes(layer.affine)
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
        self._release()
        self._source = None
        self._source_key = None
        self._dirty = True
        super().detach()

    # -- preparation ---------------------------------------------------
    def _rows(self) -> torch.Tensor | None:
        """The prepared ``(rows, T)``: the device mirror if it has one, else RAM.

        Not pinned to either. A seed is one mat-vec -- 6-17 ms on six cores
        for a real run's brain -- so when the budget or another process has
        taken the mirror back, correlating from RAM is still interactive, and
        cheaper than re-uploading for a click.
        """
        if self._hold_key is None or self.session is None:
            return None
        try:
            return self.session.store.working(self._hold_key)
        except KeyError:
            return None

    def _release(self) -> None:
        if self._hold_key is not None and self.session is not None:
            self.session.store.drop_working(self._hold_key)
        self._hold_key = None
        self._lookup = None

    def prepare(self, progress: ProgressFn | None = None) -> bool:
        """Detrend, bandpass, blur and normalize. Seconds on a real dataset.

        Runs on the compute device, where it is 50x faster than on the CPU
        (24 ms against 1.3 s for 550k voxels x 400), and keeps only the result:
        the store holds it in RAM and mirrors it while the budget allows. A
        card too full to prepare on -- a fit running -- falls back to the CPU
        rather than failing.
        """
        if not self._dirty and self._rows() is not None:
            return True
        if not self._capture_source() or self._source is None:
            return False
        nx, ny, nz, nt_full = self._source.shape
        try:
            self._window = self.time_window(nt_full)
        except ValueError as exc:
            self._roi_message = str(exc)
            self._dirty = False
            return False
        try:
            nuisance = self._nuisance(nt_full)
        except (OSError, ValueError, IndexError) as exc:
            # Dropped rather than left standing: the map on screen would be the
            # old model's, and the status line only speaks when there are none.
            self._release()
            self._roi_message = str(exc)
            self._dirty = False
            return False

        if bool(self.params.get("automask", True)):
            from fastfuncstuff.viewer.series import automask_from_series

            # The whole run, not the window: it is the array the carpet and the
            # other modes ask about, so the mask comes back from their cache.
            keep = automask_from_series(self._source, device=self._device())
        else:
            keep = np.ones((nx, ny, nz), dtype=bool)
        if not keep.any():
            self._roi_message = "the brain mask is empty; turn 'brain only' off"
            self._dirty = False
            return False

        self._release()
        device = self._device()
        try:
            rows = self._prepare_rows(keep, nuisance, device, progress)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            rows = self._prepare_rows(keep, nuisance, torch.device("cpu"), progress)

        assert self.session is not None
        self._hold_key = f"instacorr:{id(self)}"
        self.session.store.hold(self._hold_key, rows)
        del rows
        lookup = np.full((nx, ny, nz), -1, dtype=np.int32)
        lookup[keep] = np.arange(int(keep.sum()), dtype=np.int32)
        self._lookup = lookup
        self._shape = (nx, ny, nz)
        self._n_time = len(range(*self._window.indices(nt_full)))
        self._dirty = False
        if progress is not None:
            progress(1.0, "ready")
        return True

    def _device(self) -> torch.device:
        return self.session.store.device if self.session is not None else torch.device("cpu")

    def _prepare_rows(
        self,
        keep: np.ndarray,
        nuisance: np.ndarray | None,
        device: torch.device,
        progress: ProgressFn | None,
    ) -> torch.Tensor:
        """``(kept voxels, T)`` detrended, filtered, unit-norm rows on ``device``."""

        def step(fraction: float, message: str) -> None:
            if progress is not None:
                progress(fraction, message)

        assert self._source is not None
        source = self._source[..., self._window]
        nt = source.shape[-1]
        flat = torch.as_tensor(np.flatnonzero(keep.reshape(-1)), device=device)

        blur = float(self.params.get("blur") or 0.0)
        if blur > 0:
            step(0.1, f"blur {blur:g} mm")
            from fastfuncstuff.stats.smooth3d import fwhm_mm_to_sigma_vox, gaussian3d_batched

            sigma = fwhm_mm_to_sigma_vox(blur, self._zooms)
            vol = torch.as_tensor(source, dtype=torch.float32)
            vol = gaussian3d_batched(vol.permute(3, 0, 1, 2).to(device), sigma)
            # The blur keeps its input's strides, so this is still the store's
            # (X,Y,Z,T) layout under a (T,X,Y,Z) view: permuting back to
            # (V,T) is free, where reshaping as (T,V) would copy it twice.
            data = vol.permute(1, 2, 3, 0).reshape(-1, nt)[flat]
            del vol
        else:
            # The store keeps the array C-contiguous, so (V, T) is a free view
            # and the row gather reads whole time courses. Gathered before the
            # upload, so only the brain crosses to the device.
            step(0.1, "arranging")
            data = torch.as_tensor(source, dtype=torch.float32).reshape(-1, nt)
            data = data[flat.cpu()].to(device)

        # The cleaning ffs_fc does, by the same code: drift, ortvecs and the band in
        # one projection (the FFT algorithm here -- a viewer run is never censored).
        step(0.45, "filter + detrend")
        from fastfuncstuff.connectivity.matrix import unit_rows
        from fastfuncstuff.connectivity.prep import build_cleaner

        cleaner = build_cleaner(
            [nuisance], [0], nt, tr=self._tr if self._tr > 0 else None, passband=self._band()
        )
        self._stat = (float(cleaner.n_kept), float(cleaner.n_removed))
        data = cleaner.apply(data, device=device)

        step(0.85, "normalize")
        # Constant voxels come back as zero rows, so they correlate 0 with everything.
        data, _valid = unit_rows(data)
        return data

    def _nuisance(self, nt_full: int) -> np.ndarray | None:
        """``(T, k)`` float64: the Legendre drift, then every ticked ortvec column.

        Raises what reading an ortvec raises -- a missing file, a row count
        that is not the run's -- for :meth:`prepare` to report.
        """
        n_time = len(range(*self._window.indices(nt_full)))
        blocks = []
        order = self._polort()
        if order >= 0:
            from fastfuncstuff.glm.core import construct_polynomial_matrix

            poly = construct_polynomial_matrix(n_time, order, torch.device("cpu"), torch.float64)
            blocks.append(poly.numpy())
        ort, _labels = ortvec.regressor_block(
            PathListControl.enabled(self.params.get("ortvec")),
            tr=self._tr,
            keep=self._window,
            n_full=nt_full,
        )
        if ort is not None:
            blocks.append(ort)
        return np.concatenate(blocks, axis=1) if blocks else None

    def _band(self) -> tuple[float, float] | None:
        """``(fbot, ftop)`` for the cleaner, or None.

        None without a TR: a filter specified in Hz against an assumed 1 s
        sampling interval is not the filter anyone asked for.
        """
        fbot = float(self.params.get("fbot") or 0.0)
        ftop = float(self.params.get("ftop") or 0.0)
        if (fbot <= 0.0 and ftop <= 0.0) or self._tr <= 0.0:
            return None
        return (fbot, ftop)

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

    def _seed_timecourse(self, data: torch.Tensor) -> torch.Tensor | None:
        if self._lookup is None or self.session is None:
            return None
        roi_key = self._roi_key()
        if roi_key is not None:
            return self._roi_timecourse(data, roi_key)
        if self.session.state.seed is None:
            return None
        seed_ijk = self._to_source(self.session.state.seed)
        nx, ny, nz = self._shape
        si, sj, sk = seed_ijk
        radius = float(self.params.get("seed_radius") or 0.0)

        if radius <= 0.0:
            if not (0 <= si < nx and 0 <= sj < ny and 0 <= sk < nz):
                return None
            row = int(self._lookup[si, sj, sk])
            if row < 0 or float(data[row].norm()) < 0.5:
                return None  # outside the brain, or a constant voxel
            return data[row]

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
        return self._mean_of(data, ((gi * ny + gj) * nz + gk)[dist <= radius].numpy())

    def _mean_of(self, data: torch.Tensor, flat: np.ndarray) -> torch.Tensor | None:
        """The unit-length mean prepared course of box voxels ``flat``; None if none count."""
        assert self._lookup is not None
        rows = self._lookup.reshape(-1)[np.asarray(flat, dtype=np.int64).reshape(-1)]
        rows = torch.as_tensor(rows[rows >= 0], device=data.device, dtype=torch.long)
        if rows.numel():
            rows = rows[data[rows].norm(dim=-1) > 0.5]
        if rows.numel() == 0:
            return None
        seed = data[rows].mean(0)
        seed = seed - seed.mean()
        n = seed.norm()
        return seed / n if float(n) > 1e-9 else None

    def _roi_timecourse(self, data: torch.Tensor, key: str) -> torch.Tensor | None:
        """The mean prepared time course of a painted ROI's voxels, unit length."""
        if self.session is None or self._lookup is None:
            return None
        like = self.session.state.layers.find(self._source_key or "")
        if like is None:
            return None
        flat = np.flatnonzero(self.session.mask_from_layer(key, like).reshape(-1))
        n_in = int((self._lookup.reshape(-1)[flat] >= 0).sum())
        seed = self._mean_of(data, flat)
        if seed is None:
            self._roi_message = "the seed ROI has no voxels inside the brain yet"
            return None
        self._roi_message = f"seed ROI {key}: {n_in} voxels"
        return seed

    # -- production ----------------------------------------------------
    def compute(self) -> ComputedOverlay | None:
        data = self._rows()
        if data is None or self._affine is None or self._lookup is None:
            return None
        seed = self._seed_timecourse(data)
        if seed is None:
            return None
        r = (data @ seed).cpu().numpy()  # the whole correlation, one mat-vec
        vol = np.zeros(self._shape, dtype=np.float32)
        inside = self._lookup >= 0
        vol[inside] = r[self._lookup[inside]]
        return ComputedOverlay(
            values=np.nan_to_num(vol, nan=0.0),
            affine=self._affine,
            name=self.output_name(),
            kind=OverlayKind.CORRELATION,
            colormap="redblue",
            display_range=(-1.0, 1.0),
            threshold=0.3,
            # AFNI fico(samples, 1, removed): the band's dimensions are counted
            # exactly, the way ffs_fc writes its maps.
            threshold_stat=("fico", (self._stat[0], 1.0, self._stat[1])),
        )

    def _polort(self) -> int:
        return int(self.params.get("polort", 2))

    # -- reaction ------------------------------------------------------
    def on_command(self, cmd: Command, dirty: Aspect) -> Aspect:
        if isinstance(cmd, SetSeed):
            return self.refresh()
        # Only the cheap half: a stroke changes the seed, not the prepared run.
        if self.roi_stroke(cmd, self._roi_key()):
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
                    values=np.asarray(self._source[i, j, k, self._window], dtype=np.float32),
                    x_label="TR",
                )
            )
        data = self._rows()
        if data is not None and self._lookup is not None:
            nx, ny, nz = self._shape
            row = (
                int(self._lookup[i, j, k]) if (0 <= i < nx and 0 <= j < ny and 0 <= k < nz) else -1
            )
            if row >= 0:
                out.append(
                    Trace(
                        label="prepared",
                        key="prepared",
                        short="icorr prepared",
                        values=data[row].cpu().numpy(),
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
        data = self._rows()
        if data is None:
            return ""
        gb = data.numel() * data.element_size() / 1e9
        return f"{gb:.2f} GB on {data.device.type}"

    def status(self) -> str:
        if self._source is None:
            return "instacorr: needs a 4-D dataset"
        if self._preparing:
            return "instacorr: preparing…"
        if self._held:
            return f"instacorr: {self.pending_note().lstrip(' ·')}"
        if self._rows() is None:
            return f"instacorr: {self._roi_message or 'ready to prepare'}"
        where = self.residency()
        if self._roi_key() is not None:
            return f"instacorr: {self._roi_message or 'paint the seed ROI'}  ·  {where}"
        if self.session is None or self.session.state.seed is None:
            return f"instacorr: ctrl-click to set a seed  ·  {where}"
        return f"instacorr: seed {self.session.state.seed}  ·  {where}"
