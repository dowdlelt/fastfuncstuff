"""InstaSpectra: every voxel's spectrum, as one map per frequency bin.

The run goes in, and out comes a layer with one map per frequency bin, stepped
by its own sub-brick index -- ``◀ bin`` / ``bin ▶``, left / right in the
spectrum window, or `,` / `.` with the layer selected -- and never by the time
index: stepping frequency must not move the run underneath to another volume.
(It was laid along the time axis at first; that moved every time series in the
stack with it.) A spectrum window follows the crosshair -- the voxel's density
and the brain's, the bin being shown marked -- and clicking or dragging along
it moves the map to that bin.

The use it was built for: find the band a heartbeat (or breathing, or a
scanner artefact) occupies, then look at where it lives. ``→ voxel peak`` jumps
to the strongest bin of the voxel under the crosshair (click a vessel, jump to
its heartbeat); ``HR bpm`` folds a heart rate below Nyquist for this TR and
``→ HR alias`` jumps there. The robust z scale makes one threshold mean the
same thing in every frame, so a vessel tree that stands out at the cardiac bin
stays drawn at the same cut in a MIP (the 3-D voxel window).

The spectra are the slow half (gated by GO); the map kind, band width, floor
and scale are views of them, remade on every change. See
``viewer/instaspectra.py`` for the arithmetic and the choice of defaults.
"""

from __future__ import annotations

import numpy as np

from fastfuncstuff.stats import spectra as spectral
from fastfuncstuff.viewer import instaspectra as engine
from fastfuncstuff.viewer.commands import Aspect
from fastfuncstuff.viewer.modes.base import (
    HALF,
    THIRD,
    ActionControl,
    ChoiceControl,
    ComputedOverlay,
    Control,
    FloatControl,
    IntControl,
    Mode,
    OptionalFloatControl,
    OverlayKind,
    ProgressFn,
    Trace,
    mode,
)
from fastfuncstuff.viewer.modes.masking import RunMaskMixin
from fastfuncstuff.viewer.vocab import SetVolume

#: The cut a robust z map starts at: well out of the brain's own distribution.
Z_THRESHOLD = 3.0


@mode
class InstaSpectraMode(RunMaskMixin, Mode):
    name = "instaspectra"
    label = "InstaSpectra"
    tag = "ISPEC"
    overlay_kind = OverlayKind.VALUE
    gated = True
    #: The run is the anatomy the maps are read against.
    hides_input = False

    def __init__(self) -> None:
        self._result: engine.Spectra | None = None
        self._identity: tuple | None = None
        self._source_key: str | None = None
        self._source_name = ""
        self._message = ""
        #: ``(params) -> (V, F)`` maps, so a refresh that changed nothing
        #: hands the store the same provider and nothing re-uploads.
        self._maps_key: tuple | None = None
        self._maps: np.ndarray | None = None
        self._shown: tuple[str, str] | None = None
        super().__init__()

    def controls(self) -> tuple[Control, ...]:
        return (
            *self.mask_controls(),
            *self.time_cut_controls(),
            IntControl(
                name="polort",
                label="polort",
                lo=0,
                hi=9,
                default=2,
                span=HALF,
                newline=True,
                help="Legendre drift projected out of every voxel first. Quadratic (2) is "
                "the default the cardiac work settled on; zero still removes the mean.",
            ),
            ChoiceControl(
                name="method",
                label="estimator",
                choices=spectral.METHODS,
                default="periodogram",
                span=HALF,
                help="periodogram: one Hann-tapered FFT, full resolution, noisy bins. "
                "multitaper: the average over DPSS tapers -- smoother at a resolution of "
                "2*NW/duration. welch: averaged half-overlapping segments -- smoother, "
                "coarser.",
            ),
            FloatControl(
                name="nw",
                label="NW",
                lo=1.0,
                hi=8.0,
                default=2.0,
                step=0.5,
                span=HALF,
                newline=True,
                visible_when=("method", ("multitaper",)),
                help="Time-bandwidth product: 2*NW-1 tapers, bins smoothed over 2*NW/duration Hz.",
            ),
            IntControl(
                name="segments",
                label="segments",
                lo=2,
                hi=32,
                default=4,
                span=HALF,
                visible_when=("method", ("welch",)),
                help="Half-overlapping segments; the frequency step grows by about as much.",
            ),
            *self.gate_controls(),
            ChoiceControl(
                name="map",
                label="map",
                choices=engine.MAP_KINDS,
                default="share",
                style="radio",
                newline=True,
                help="share: the band's percent of the voxel's power above the floor -- "
                "loudness-free. peak: the band's density over its neighbours' (1 = no "
                "peak) -- a narrow line on a slope. amplitude: the band's sd in percent "
                "of the voxel's mean -- what it costs there.",
            ),
            ChoiceControl(
                name="scale",
                label="scale",
                choices=engine.SCALES,
                default="robust z",
                style="radio",
                span=HALF,
                newline=True,
                help="robust z: each frame against the median and MAD of its own masked "
                "voxels, floored at 0, so one threshold means the same in every frame. "
                "value: the map's own units.",
            ),
            FloatControl(
                name="width",
                label="band Hz",
                lo=0.0,
                hi=1.0,
                default=0.0,
                step=0.005,
                unit="Hz",
                span=HALF,
                help="Width of the band around each bin. 0 is one bin; widen it to gather "
                "a heartbeat that wanders over the run.",
            ),
            FloatControl(
                name="floor",
                label="floor Hz",
                lo=0.0,
                hi=0.5,
                default=0.04,
                step=0.005,
                unit="Hz",
                span=HALF,
                newline=True,
                visible_when=("map", ("share",)),
                help="Power below this is left out of the share's denominator: drift "
                "residue would otherwise set every other bin's share.",
            ),
            OptionalFloatControl(
                name="heart_bpm",
                label="HR bpm",
                lo=30.0,
                hi=200.0,
                default=0.0,
                off_value=0.0,
                on_value=60.0,
                step=1.0,
                span=HALF,
                newline=True,
                help="A heart rate, to fold below Nyquist at this TR: the status line "
                "says where it lands, and → HR alias jumps there.",
            ),
        )

    def actions(self) -> tuple[ActionControl, ...]:
        return (
            ActionControl(name="prev", label="◀ bin", span=THIRD, help="Previous frequency bin"),
            ActionControl(name="next", label="bin ▶", span=THIRD, help="Next frequency bin"),
            ActionControl(
                name="voxel_peak",
                label="→ voxel peak",
                span=THIRD,
                help="Jump to the strongest bin above the floor of the voxel under the crosshair.",
            ),
            ActionControl(
                name="alias",
                label="→ HR alias",
                span=THIRD,
                help="Jump to the bin the heart rate aliases to at this TR.",
            ),
            *super().actions(),
        )

    def preparation_params(self) -> frozenset[str]:
        return frozenset(
            {"roi", "mask", "polort", "method", "nw", "segments", "drop_first", "drop_last"}
        )

    def panel_names(self) -> tuple[str, ...]:
        return ("spectrum",)

    def input_layer_key(self) -> str | None:
        return self._source_key

    # -- the slow half -------------------------------------------------
    def _spectral_params(self) -> dict:
        return {
            "polort": int(self.params.get("polort", 2)),
            "method": str(self.params.get("method") or "periodogram"),
            "nw": float(self.params.get("nw") or 2.0),
            "segments": int(self.params.get("segments") or 4),
        }

    def prepare(self, progress: ProgressFn | None = None) -> bool:
        if self.session is None:
            return False
        layer = self.source_layer()
        if layer is None:
            self._message = "needs a 4-D run"
            self._dirty = False
            return False
        sp = self._spectral_params()
        cut = (int(self.params.get("drop_first") or 0), int(self.params.get("drop_last") or 0))
        identity = (layer.key, self.mask_identity(), cut, tuple(sorted(sp.items())))
        if self._result is not None and identity == self._identity:
            self._dirty = False
            return True
        try:
            data = self.session.store.ensure_ram(layer.key)
            data = data[..., self.time_window(data.shape[-1])]
            result = engine.compute(
                data,
                affine=np.asarray(layer.affine, dtype=float),
                tr=self.source_tr(),
                mask=self.resolve_mask(layer, data.shape[:3]),
                device=self.session.store.device,
                progress=progress,
                **sp,
            )
        except (OSError, ValueError) as exc:
            self._message = str(exc)
            self._dirty = False
            return False
        self._result = result
        self._identity = identity
        self._maps_key = None
        self._source_key = layer.key
        self._source_name = layer.name
        self._message = (
            f"{result.n_frames} bins of {layer.name} to {result.freqs[-1]:.3g} {result.unit}, "
            f"step {result.df:.3g}, {int(result.mask.sum())} voxels ({self.mask_source()})"
            + (f", volumes {cut[0]}..-{cut[1]} cut" if any(cut) else "")
            + ("" if result.tr_known else "  ·  ⚠ no TR in the header")
        )
        self._dirty = False
        return True

    # -- production ----------------------------------------------------
    def _layer(self):
        """The map layer this mode owns, once it exists."""
        if self.session is None:
            return None
        return self.session.state.layers.find_by_source(self.layer_source)

    @property
    def _frame(self) -> int:
        """The bin shown: the map layer's own sub-brick."""
        n = self._result.n_frames if self._result is not None else 1
        layer = self._layer()
        k = layer.volume_index if layer is not None else 0
        return max(0, min(int(k), n - 1))

    def _view(self) -> tuple[str, str, float, float]:
        return (
            str(self.params.get("map") or "share"),
            str(self.params.get("scale") or "robust z"),
            float(self.params.get("width") or 0.0),
            float(self.params.get("floor") if self.params.get("floor") is not None else 0.04),
        )

    def compute(self) -> ComputedOverlay | None:
        if self._result is None:
            return None
        result = self._result
        kind, scale, width, floor = self._view()
        key = (id(result), kind, scale, width, floor)
        if key != self._maps_key or self._maps is None:
            self._maps = result.maps(kind, width=width, floor=floor, scale=scale)
            self._maps_key = key
        maps = self._maps
        n = result.n_frames

        def frame(i: int, maps: np.ndarray = maps) -> np.ndarray:
            return result.volume(maps[:, max(0, min(int(i), n - 1))])

        z = scale == "robust z"
        finite = maps[np.isfinite(maps)]
        top = float(np.percentile(finite, 99.9)) if finite.size else 1.0
        if z:
            top = max(top, 2 * Z_THRESHOLD)
        top = top if top > 0 else 1.0
        rescale = self._shown is not None and self._shown != (kind, scale)
        self._shown = (kind, scale)
        unit = "z" if z else engine.MAP_UNITS[kind]
        labels = tuple(f"{kind} {unit} @ {f:.4g} {result.unit}" for f in result.freqs)
        return ComputedOverlay(
            values=frame(0),
            affine=result.affine,
            name=self.output_name(kind),
            kind=OverlayKind.VALUE,
            colormap="hot",
            display_range=(0.0, top),
            threshold=Z_THRESHOLD if z else 0.0,
            volume_labels=labels,
            rescale=rescale,
            frames=frame,
            n_frames=n,
            frame=self._frame,
        )

    # -- actions -------------------------------------------------------
    def action(self, name: str, progress: ProgressFn | None = None) -> Aspect:
        if name in ("prev", "next"):
            return self._goto(self._frame + (-1 if name == "prev" else 1))
        if name == "voxel_peak":
            psd = self._voxel_psd()
            if psd is None or self._result is None:
                return Aspect.NOTHING
            floor = self._view()[3]
            above = self._result.freqs >= floor
            if not above.any():
                above[:] = True
            k = int(np.flatnonzero(above)[np.argmax(psd[above])])
            return self._goto(k)
        if name == "alias":
            f = self._heart_alias()
            if f is None or self._result is None:
                return Aspect.NOTHING
            return self._goto(self._result.frame_of(f))
        return super().action(name, progress)

    def _goto(self, k: int) -> Aspect:
        """Show bin ``k``: the map layer's sub-brick, leaving the time index alone."""
        layer = self._layer()
        if layer is None or self._result is None:
            return Aspect.NOTHING
        target = max(0, min(int(k), self._result.n_frames - 1))
        if target == layer.volume_index:
            return Aspect.NOTHING
        assert self.session is not None
        self.session.state.layers.update(layer.key, volume_index=target)
        return SetVolume.aspects | Aspect.GRAPH

    def _heart_alias(self) -> float | None:
        bpm = float(self.params.get("heart_bpm") or 0.0)
        if bpm <= 0 or self._result is None or not self._result.tr_known:
            return None
        return float(spectral.alias(bpm / 60.0, self._result.tr))

    # -- graph ---------------------------------------------------------
    def _voxel_psd(self) -> np.ndarray | None:
        """The crosshair voxel's PSD, prepared exactly as the maps' voxels were."""
        if self.session is None or self._result is None or self._source_key is None:
            return None
        series = self.session.timeseries(self._source_key)
        if series.size == 0:
            return None
        series = series[self.time_window(series.size)]
        if series.size != self._result.n_time:
            return None
        sp = self._spectral_params()
        return engine.voxel_psd(series, tr=self._result.tr, **sp)

    def panels(self) -> dict[str, list[Trace]]:
        result = self._result
        if result is None:
            return {}
        k = self._frame
        layer = self._layer()
        scrub_key = layer.key if layer is not None else ""
        lines: list[Trace] = []
        psd = self._voxel_psd()
        x_label = f"{result.unit}"
        if psd is not None:
            lines.append(
                Trace(
                    label=f"voxel log10 PSD  ·  {result.freqs[k]:.4g} {result.unit}",
                    values=np.log10(np.maximum(psd, 1e-30)).astype(np.float32),
                    x=result.freqs.astype(np.float32),
                    x_label=x_label,
                    key="voxel",
                    short="voxel",
                    marker=k,
                    scrub=True,
                    scrub_key=scrub_key,
                )
            )
        lines.append(
            Trace(
                label="mask mean log10 PSD",
                values=np.log10(np.maximum(result.mask_mean, 1e-30)).astype(np.float32),
                x=result.freqs.astype(np.float32),
                x_label=x_label,
                key="mask",
                short="mask mean",
                marker=None if psd is not None else k,
                scrub=True,
                scrub_key=scrub_key,
            )
        )
        return {"spectrum": lines}

    def status(self) -> str:
        if self._preparing:
            return f"{self.name}: spectra…"
        if self._held:
            return f"{self.name}: {self.pending_note().lstrip(' ·')}"
        if self._result is None:
            return f"{self.name}: {self._message or 'pick a 4-D run'}"
        r = self._result
        k = self._frame
        heart = self._heart_alias()
        alias = (
            f"  ·  HR {float(self.params.get('heart_bpm')):.0f} bpm → {heart:.3g} Hz "
            f"(bin {r.frame_of(heart)})"
            if heart is not None
            else ""
        )
        return (
            f"{self.name}: bin {k}/{r.n_frames - 1} = {r.freqs[k]:.4g} {r.unit}{alias}  ·  "
            f"{self._message}"
        )


__all__ = ["InstaSpectraMode"]
