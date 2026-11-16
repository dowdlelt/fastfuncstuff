"""InstaPCA: decompose the run you are looking at, and keep the noise.

ICA mode reviews a decomposition someone already made. This one makes it, from
the input run, in seconds: drift out, every voxel scaled to unit length, the
leading components of what is left. The preparation is ffs_denoise's (see
``viewer/instapca.py``), so the map is a correlation and reads the same in
every component -- the trick that makes its PC figures interpretable.

The loop is ICA's: look, decide, next. A component labelled noise is a column
of the ortvec SAVE writes, and that file goes straight into InstaGLM's ortvec
list or a ``-ortvec`` on the command line.

Labels belong to one decomposition. A different run, mask or polort is a
different set of components, so the labels go with it; more or fewer components
is not -- the leading ones do not change -- so only labels past the end go.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fastfuncstuff.viewer import instapca as engine
from fastfuncstuff.viewer.commands import Aspect, Command
from fastfuncstuff.viewer.modes.base import (
    HALF,
    THIRD,
    ActionControl,
    ChoiceControl,
    ComputedOverlay,
    Control,
    IntControl,
    Mode,
    OverlayKind,
    PathControl,
    ProgressFn,
    Trace,
    correlation_stat,
    mode,
)

LABELS = ("signal", "noise")
#: How each map's units read in a volume label.
_MAP_UNITS = {"amplitude": "amplitude (% signal, sd)", "z": "z"}
#: Upper bound on the component spinner, whatever the run could support. Past
#: this a review is not a review.
MAX_COMPONENTS = 200
#: Suffixes stripped from a run's filename to name its ortvec.
_IMAGE_SUFFIXES = (".nii.gz", ".nii.zst", ".nii", "+orig.HEAD", "+tlrc.HEAD", ".HEAD")


def default_ortvec_path(run_path: str, tail: str = "pca_noise") -> Path:
    """``sub-01_bold.nii.gz`` to ``sub-01_bold_pca_noise.1D`` beside it."""
    path = Path(run_path)
    name = path.name
    for suffix in _IMAGE_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return path.with_name(f"{name}_{tail}.1D")


@mode
class InstaPCAMode(Mode):
    name = "instapca"
    label = "InstaPCA"
    tag = "IPCA"
    overlay_kind = OverlayKind.CORRELATION
    gated = True
    #: What one component is called on screen, and the ortvec's suffix --
    #: the hooks InstaICA (a rotation of this decomposition) overrides.
    component_word = "PC"
    ortvec_suffix = "pca_noise"
    map_choices: tuple[str, ...] = ("correlation", "amplitude")
    #: The run is the anatomy each component's map is judged against.
    hides_input = False

    def __init__(self) -> None:
        self._result: engine.Decomposition | None = None
        #: ``(run key, mask, polort)`` the labels and the result belong to.
        self._identity: tuple | None = None
        #: The decomposition before a subclass's step (InstaICA's rotation),
        #: and what it was made from: a new seed reuses it.
        self._raw: engine.Decomposition | None = None
        self._raw_key: tuple | None = None
        self._n_requested = 0
        self._source_key: str | None = None
        self._source_name = ""
        self._source_path = ""
        self._message = ""
        #: ``(map, cut)`` last drawn: a change of either is a new question.
        self._shown_map: tuple[str, str] | None = None
        #: component index -> "signal" | "noise". Absent means unlabelled.
        self.labels: dict[int, str] = {}
        self._saved_to: Path | None = None
        super().__init__()

    def controls(self) -> tuple[Control, ...]:
        n = self._result.n_components if self._result is not None else 0
        return (
            ChoiceControl(
                name="roi",
                label="mask ROI",
                choices=("none", *self._roi_choices()),
                default="none",
                help="A painted (or loaded ROI) layer as the mask, followed live as you "
                "paint it. Overrides the mask below.",
            ),
            PathControl(
                name="mask",
                label="mask",
                filter="NIfTI (*.nii *.nii.gz *.nii.zst);;All (*)",
                help="A mask file on the run's grid, or the key of a loaded layer (a "
                "painted ROI, an atlas: nonzero is inside; any grid). Empty means AFNI's "
                "automask of the run.",
            ),
            *self.time_cut_controls(),
            IntControl(
                name="polort",
                label="polort",
                lo=0,
                hi=9,
                default=2,
                span=HALF,
                newline=True,
                help="Legendre drift projected out of every voxel before the decomposition. "
                "Zero still removes the mean.",
            ),
            IntControl(
                name="n_components",
                label="# comps",
                lo=1,
                hi=MAX_COMPONENTS,
                default=20,
                span=HALF,
                help="How many leading components to keep. Changing it keeps the labels: "
                "the leading components are the same either way.",
            ),
            *self.gate_controls(),
            IntControl(
                name="component",
                label="component",
                lo=0,
                hi=max(n - 1, 0),
                default=0,
                span=HALF,
                newline=True,
                help="Which component to show.",
            ),
            ChoiceControl(
                name="map",
                label="map",
                choices=self.map_choices,
                default="correlation",
                style="radio",
                span=HALF,
                help="correlation: each voxel's detrended series against the component, "
                "amplitude-free -- where it lives. amplitude: the standard deviation of its "
                "share of the voxel, in percent signal change -- what removing it costs.",
            ),
            ChoiceControl(
                name="cut",
                label="thr on",
                choices=("correlation", "map"),
                default="correlation",
                style="radio",
                span=HALF,
                help="What the threshold cuts. correlation: colour by the map, keep only "
                "the voxels where the component is really there (the p spinner reads the "
                "correlation's p) -- amplitude coloured and r cut answers 'where, and how "
                "big'. map: cut the map shown on its own values.",
            ),
            PathControl(
                name="ortvec",
                label="save to",
                filter="1D (*.1D);;All (*)",
                newline=True,
                help=f"Where SAVE NOISE writes. Empty means <run>_{self.ortvec_suffix}.1D "
                "beside the run.",
            ),
        )

    def actions(self) -> tuple[ActionControl, ...]:
        return (
            ActionControl(name="prev", label="◀ prev", span=THIRD, help="Previous component"),
            ActionControl(name="next", label="next ▶", span=THIRD, help="Next component"),
            ActionControl(
                name="unlabel", label="clear", span=THIRD, help="Remove this component's label"
            ),
            ActionControl(name="signal", label="signal", span=THIRD, help="Label signal, step on"),
            ActionControl(name="noise", label="noise", span=THIRD, help="Label noise, step on"),
            ActionControl(
                name="save",
                label="save noise",
                span=THIRD,
                help="Write the noise components' time courses as an ortvec 1D file, "
                "one unit-variance column each.",
            ),
            *super().actions(),
        )

    def preparation_params(self) -> frozenset[str]:
        return frozenset({"roi", "mask", "polort", "n_components", "drop_first", "drop_last"})

    def _roi_choices(self) -> list[str]:
        if self.session is None:
            return []
        return [
            f"{ly.key} {ly.name}"
            for ly in self.session.state.layers
            if ly.roi or self.session.painter.owns(ly.key)
        ]

    def _roi_key(self) -> str | None:
        """The layer chosen as the mask ROI, if it is still loaded."""
        text = str(self.params.get("roi") or "none").split(" ", 1)[0]
        if text == "none" or self.session is None:
            return None
        return text if self.session.state.layers.find(text) is not None else None

    def on_command(self, cmd: Command, dirty: Aspect) -> Aspect:
        # A stroke on the ROI that is the mask changes the voxels decomposed.
        if self.roi_stroke(cmd, self._roi_key()):
            return self.stale()
        return Aspect.NOTHING

    def panel_names(self) -> tuple[str, ...]:
        return ("timecourse", "spectrum", "scree")

    def input_layer_key(self) -> str | None:
        return self._source_key

    # -- the slow half -------------------------------------------------
    def prepare(self, progress: ProgressFn | None = None) -> bool:
        if self.session is None:
            return False
        layer = self.source_layer()
        if layer is None:
            self._message = "needs a 4-D run"
            self._dirty = False
            return False
        mask_text = str(self.params.get("mask") or "").strip()
        roi_key = self._roi_key()
        if roi_key is not None:
            mask_text = roi_key
        mask_layer = (
            self.session.state.layers.find(mask_text.split(" ", 1)[0]) if mask_text else None
        )
        polort = int(self.params.get("polort", 2))
        n_comp = int(self.params.get("n_components") or 20)
        # A layer used as the mask can be painted after the fact: its data
        # version is part of what the decomposition was made from.
        mask_version = self.session.store.get(mask_layer.key).generation if mask_layer else 0
        cut = (int(self.params.get("drop_first") or 0), int(self.params.get("drop_last") or 0))
        identity = (layer.key, f"{mask_text}@{mask_version}", polort, cut, *self._identity_extra())
        # Coming back into the mode re-marks it dirty; the decomposition it
        # left is still the right one unless what it was made from changed.
        if self._result is not None and identity == self._identity and n_comp == self._n_requested:
            self._dirty = False
            return True
        # The decomposition itself, before a subclass's step: a new ICA seed
        # rotates the same PCA again (milliseconds) rather than redoing it.
        raw_key = (*identity[: len(identity) - len(self._identity_extra())], n_comp)
        raw = self._raw if self._raw_key == raw_key else None
        try:
            if raw is None:
                data = self.session.store.ensure_ram(layer.key)
                data = data[..., self.time_window(data.shape[-1])]
                tr = float(self.session.store.get(layer.key).info.tr or 0.0)
                if mask_layer is not None:
                    mask = self.session.mask_from_layer(mask_layer.key, layer)
                else:
                    mask = engine.read_mask(mask_text, data.shape[:3]) if mask_text else None
                result = engine.decompose(
                    data,
                    affine=np.asarray(layer.affine, dtype=float),
                    tr=tr,
                    mask=mask,
                    mask_source=(
                        mask_layer.name
                        if mask_layer
                        else Path(mask_text).name
                        if mask_text
                        else "automask"
                    ),
                    polort=polort,
                    n_components=n_comp,
                    device=self.session.store.device,
                    progress=progress,
                )
                self._raw, self._raw_key = result, raw_key
            else:
                result = raw
            result = self._post(result, progress)
        except (OSError, ValueError) as exc:
            self._message = str(exc)
            self._dirty = False
            return False
        if identity != self._identity:
            self.labels = {}
            self._saved_to = None
        else:
            self.labels = {k: v for k, v in self.labels.items() if k < result.n_components}
        self._result = result
        self._identity = identity
        self._n_requested = n_comp
        self._source_key = layer.key
        self._source_name = layer.name
        self._source_path = str(layer.path or "")
        total = float(np.sum(result.explained)) * 100
        self._message = (
            f"{result.n_components} components of {layer.name}, {total:.0f}% of the variance, "
            f"{int(result.mask.sum())} voxels"
            + (f", volumes {cut[0]}..-{cut[1]} cut" if any(cut) else "")
            + (f"  ·  \u26a0 {result.note}" if result.note else "")
        )
        self._dirty = False
        return True

    def _identity_extra(self) -> tuple:
        """What else the decomposition was made from, for a subclass."""
        return ()

    def _post(
        self, result: engine.Decomposition, progress: ProgressFn | None
    ) -> engine.Decomposition:
        """What a subclass does to the decomposition. PCA: nothing."""
        return result

    def mixing_matrix(self) -> np.ndarray | None:
        """``(T, K)`` component time courses -- what Denoise regresses."""
        return None if self._result is None else np.asarray(self._result.timecourses)

    # -- production ----------------------------------------------------
    @property
    def _index(self) -> int:
        n = self._result.n_components if self._result is not None else 0
        return max(0, min(int(self.params.get("component") or 0), n - 1))

    def _label_text(self, k: int) -> str:
        return f"[{self.labels[k]}]" if k in self.labels else ""

    def compute(self) -> ComputedOverlay | None:
        if self._result is None:
            return None
        k = self._index
        kind = str(self.params.get("map") or "correlation")
        cut = (
            "correlation" if str(self.params.get("cut") or "correlation") == "correlation" else kind
        )
        values = self._result.volume(k, kind)
        top = self._robust_top(values, kind)
        cut_values = None if cut == kind else self._result.volume(k, cut)
        cut_top = top if cut_values is None else self._robust_top(cut_values, cut)
        # The threshold slider is named after the map it cuts, not the one drawn.
        self.overlay_kind = (  # ty: ignore[invalid-attribute-access]  (as InstaGLM)
            OverlayKind.CORRELATION if cut == "correlation" else OverlayKind.VALUE
        )
        rescale = self._shown_map is not None and self._shown_map != (kind, cut)
        self._shown_map = (kind, cut)
        pct = self._result.explained[k] * 100
        word = f"{self.component_word} {k}"
        shown = f"{word} {_MAP_UNITS.get(kind, kind)} ({pct:.1f}%)"
        return ComputedOverlay(
            values=values,
            affine=self._result.affine,
            name=self.output_name(f"{word} {self._label_text(k)}".strip()),
            kind=self.overlay_kind,
            colormap="redblue",
            display_range=(-top, top) if top > 0 else None,
            threshold=cut_top * 0.4 if cut_top > 0 else 0.0,
            threshold_values=cut_values,
            volume_labels=(shown,) if cut_values is None else (shown, f"{word} {cut}"),
            rescale=rescale,
            threshold_stat=(
                correlation_stat(self._result.timecourses.shape[0], self._result.polort)
                if cut == "correlation"
                else None
            ),
        )

    @staticmethod
    def _robust_top(values: np.ndarray, kind: str) -> float:
        finite = values[np.isfinite(values)]
        top = float(np.percentile(np.abs(finite), 99.0)) if finite.size else 1.0
        return min(top, 1.0) if kind == "correlation" else top

    # -- actions -------------------------------------------------------
    def action(self, name: str, progress: ProgressFn | None = None) -> Aspect:
        if name in ("prev", "next"):
            return self._step(-1 if name == "prev" else 1)
        if name in LABELS or name == "unlabel":
            if self._result is None:
                return Aspect.NOTHING
            k = self._index
            if name == "unlabel":
                self.labels.pop(k, None)
                return self.refresh() | Aspect.GRAPH
            self.labels[k] = name
            # Stays on the last component rather than wrapping, so the end of
            # a review is visible as the end.
            if k < self._result.n_components - 1:
                return self._step(1)
            return self.refresh() | Aspect.GRAPH
        if name == "save":
            self._save()
            return Aspect.GRAPH
        return super().action(name, progress)

    def _step(self, delta: int) -> Aspect:
        if self._result is None:
            return Aspect.NOTHING
        target = max(0, min(self._index + delta, self._result.n_components - 1))
        return self.set_param("component", target) | Aspect.GRAPH

    def noise_components(self) -> list[int]:
        return sorted(k for k, v in self.labels.items() if v == "noise")

    def ortvec_path(self) -> Path:
        chosen = str(self.params.get("ortvec") or "").strip()
        if chosen:
            return Path(chosen).expanduser()
        if not self._source_path:
            raise ValueError("the run has no file to save beside; choose where to save")
        return default_ortvec_path(self._source_path, self.ortvec_suffix)

    def _save(self) -> Path:
        if self._result is None:
            raise ValueError("nothing decomposed yet")
        noise = self.noise_components()
        if not noise:
            raise ValueError("no component is labelled noise yet")
        path = engine.write_ortvec(
            self.ortvec_path(), self._result, noise, source=self._source_name
        )
        self._saved_to = path
        self._message = f"saved {len(noise)} noise columns to {path}"
        return path

    # -- graph ---------------------------------------------------------
    def _timecourse(self, k: int) -> Trace | None:
        if self._result is None:
            return None
        return Trace(
            label=f"{self.component_word} {k} time course {self._label_text(k)}".strip(),
            values=np.asarray(self._result.timecourses[:, k], dtype=np.float32),
            x_label="TR",
            key="timecourse",
            short=f"{self.component_word} time course",
        )

    def _spectrum(self, k: int) -> Trace | None:
        if self._result is None:
            return None
        ts = self._result.timecourses[:, k]
        values = np.abs(np.fft.rfft(ts))[1:] ** 2
        if not values.size:
            return None
        x, x_label = None, "bin"
        if self._result.tr > 0:
            nyquist = 0.5 / self._result.tr
            x = np.linspace(nyquist / values.size, nyquist, values.size)
            x_label = "Hz"
        return Trace(
            label=f"{self.component_word} {k} power spectrum",
            values=values,
            x=x,
            x_label=x_label,
            key="spectrum",
            short=f"{self.component_word} spectrum",
        )

    def _scree(self) -> Trace | None:
        """Variance per component, the one being reviewed marked.

        Two readings of one line: percent of the run's total variance (left),
        and percent of what the kept components explain (right) -- the second
        is the one that says how much of *this decomposition* a component is.
        """
        if self._result is None:
            return None
        explained = np.asarray(self._result.explained, dtype=np.float64)
        kept = float(explained.sum())
        k = self._index
        share = explained[k] / kept * 100 if kept > 0 else float("nan")
        return Trace(
            label=f"{self.component_word} {k}: {explained[k] * 100:.2f}% of total, "
            f"{share:.1f}% of the {kept * 100:.0f}% explained  (left: total, right: explained)",
            values=np.asarray(explained * 100, dtype=np.float32),
            x=np.arange(self._result.n_components, dtype=np.float32),
            x_label="component",
            key="scree",
            short="% variance",
            marker=k,
            right_scale=1.0 / kept if kept > 0 else None,
        )

    def panels(self) -> dict[str, list[Trace]]:
        if self._result is None:
            return {}
        k = self._index
        lines = {"timecourse": self._timecourse(k), "spectrum": self._spectrum(k)}
        lines["scree"] = self._scree()
        return {name: [t] for name, t in lines.items() if t is not None}

    def series(self, ijk: tuple[int, int, int]) -> list[Trace]:
        """The component's time course and spectrum, the same in every voxel."""
        if self._result is None:
            return []
        k = self._index
        return [t for t in (self._timecourse(k), self._spectrum(k)) if t is not None]

    def status(self) -> str:
        if self._preparing:
            return f"{self.name}: decomposing…"
        if self._held:
            return f"{self.name}: {self.pending_note().lstrip(' ·')}"
        if self._result is None:
            return f"{self.name}: {self._message or 'pick a 4-D run'}"
        n = self._result.n_components
        n_noise = len(self.noise_components())
        n_sig = sum(1 for v in self.labels.values() if v == "signal")
        here = self.labels.get(self._index, "unlabelled")
        pct = self._result.explained[self._index] * 100
        return (
            f"{self.name}: {self.component_word} {self._index}/{n - 1} {here} ({pct:.1f}%)  ·  "
            f"{n_noise} noise, {n_sig} signal, {n - n_noise - n_sig} left  ·  {self._message}"
        )


__all__ = ["InstaPCAMode", "default_ortvec_path"]
