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
from fastfuncstuff.viewer.commands import Aspect
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
    mode,
)

LABELS = ("signal", "noise")
#: Upper bound on the component spinner, whatever the run could support. Past
#: this a review is not a review.
MAX_COMPONENTS = 200
#: Suffixes stripped from a run's filename to name its ortvec.
_IMAGE_SUFFIXES = (".nii.gz", ".nii.zst", ".nii", "+orig.HEAD", "+tlrc.HEAD", ".HEAD")


def default_ortvec_path(run_path: str) -> Path:
    """``sub-01_bold.nii.gz`` to ``sub-01_bold_pca_noise.1D`` beside it."""
    path = Path(run_path)
    name = path.name
    for suffix in _IMAGE_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return path.with_name(f"{name}_pca_noise.1D")


@mode
class InstaPCAMode(Mode):
    name = "instapca"
    label = "InstaPCA"
    tag = "IPCA"
    overlay_kind = OverlayKind.CORRELATION

    def __init__(self) -> None:
        self._result: engine.Decomposition | None = None
        #: ``(run key, mask, polort)`` the labels and the result belong to.
        self._identity: tuple[str, str, int] | None = None
        self._n_requested = 0
        self._source_key: str | None = None
        self._source_name = ""
        self._source_path = ""
        self._message = ""
        self._shown_map: str | None = None
        #: component index -> "signal" | "noise". Absent means unlabelled.
        self.labels: dict[int, str] = {}
        self._saved_to: Path | None = None
        super().__init__()

    def controls(self) -> tuple[Control, ...]:
        n = self._result.n_components if self._result is not None else 0
        return (
            PathControl(
                name="mask",
                label="mask",
                filter="NIfTI (*.nii *.nii.gz *.nii.zst);;All (*)",
                help="A mask on the run's grid. Empty means AFNI's automask of the run.",
            ),
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
                choices=("correlation", "amplitude"),
                default="correlation",
                style="radio",
                span=HALF,
                help="correlation: each voxel's detrended series against the component, "
                "amplitude-free -- where it lives. amplitude: the standard deviation of its "
                "share of the voxel, in percent of the voxel's mean -- what removing it costs.",
            ),
            PathControl(
                name="ortvec",
                label="save to",
                filter="1D (*.1D);;All (*)",
                newline=True,
                help="Where SAVE NOISE writes. Empty means <run>_pca_noise.1D beside the run.",
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
        return frozenset({"mask", "polort", "n_components"})

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
        polort = int(self.params.get("polort", 2))
        n_comp = int(self.params.get("n_components") or 20)
        identity = (layer.key, mask_text, polort)
        # Coming back into the mode re-marks it dirty; the decomposition it
        # left is still the right one unless what it was made from changed.
        if self._result is not None and identity == self._identity and n_comp == self._n_requested:
            self._dirty = False
            return True
        data = self.session.store.ensure_ram(layer.key)
        tr = float(self.session.store.get(layer.key).info.tr or 0.0)
        try:
            mask = engine.read_mask(mask_text, data.shape[:3]) if mask_text else None
            result = engine.decompose(
                data,
                affine=np.asarray(layer.affine, dtype=float),
                tr=tr,
                mask=mask,
                mask_source=Path(mask_text).name if mask_text else "automask",
                polort=polort,
                n_components=n_comp,
                device=self.session.store.device,
                progress=progress,
            )
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
        )
        self._dirty = False
        return True

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
        values = self._result.volume(k, kind)
        finite = values[np.isfinite(values)]
        top = float(np.percentile(np.abs(finite), 99.0)) if finite.size else 1.0
        if kind == "correlation":
            top = min(top, 1.0)
        rescale = self._shown_map is not None and self._shown_map != kind
        self._shown_map = kind
        pct = self._result.explained[k] * 100
        return ComputedOverlay(
            values=values,
            affine=self._result.affine,
            name=self.output_name(f"PC {k} {self._label_text(k)}".strip()),
            kind=OverlayKind.CORRELATION if kind == "correlation" else OverlayKind.VALUE,
            colormap="redblue",
            display_range=(-top, top) if top > 0 else None,
            threshold=top * 0.4 if top > 0 else 0.0,
            volume_labels=(f"PC {k} {kind} ({pct:.1f}%)",),
            rescale=rescale,
        )

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
        return default_ortvec_path(self._source_path)

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
            label=f"PC {k} time course {self._label_text(k)}".strip(),
            values=np.asarray(self._result.timecourses[:, k], dtype=np.float32),
            x_label="TR",
            key="timecourse",
            short="PC time course",
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
            label=f"PC {k} power spectrum",
            values=values,
            x=x,
            x_label=x_label,
            key="spectrum",
            short="PC spectrum",
        )

    def _scree(self) -> Trace | None:
        if self._result is None:
            return None
        return Trace(
            label="variance explained (%)",
            values=np.asarray(self._result.explained * 100, dtype=np.float32),
            x=np.arange(self._result.n_components, dtype=np.float32),
            x_label="component",
            key="scree",
            short="% variance",
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
            return "instapca: decomposing…"
        if self._result is None:
            return f"instapca: {self._message or 'pick a 4-D run'}"
        n = self._result.n_components
        n_noise = len(self.noise_components())
        n_sig = sum(1 for v in self.labels.values() if v == "signal")
        here = self.labels.get(self._index, "unlabelled")
        pct = self._result.explained[self._index] * 100
        return (
            f"instapca: PC {self._index}/{n - 1} {here} ({pct:.1f}%)  ·  "
            f"{n_noise} noise, {n_sig} signal, {n - n_noise - n_sig} left  ·  {self._message}"
        )


__all__ = ["InstaPCAMode", "default_ortvec_path"]
