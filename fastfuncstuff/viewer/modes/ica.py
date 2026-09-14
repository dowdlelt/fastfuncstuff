"""ICA: the overlay is one component of a decomposition, picked by index.

Reads a decomposition directory -- the MELODIC-compatible layout ffs already
writes -- and turns it into something you step through: the component map as the
overlay, its time course and single-sided spectrum as graph traces.

``melodic_FTmix`` is the spectrum MELODIC itself computed, so it is used when
present rather than recomputed; a spectrum derived from the same mixing matrix
by a slightly different convention would disagree with every other tool looking
at the same directory. Without it, the rFFT of the time course is computed here.

Reviewing a decomposition is a loop of look, decide, next. The component map is
the mode's overlay (``A_ICA IC 3``); its time course and spectrum open in trace
windows of their own; and a label -- signal or noise -- is one key in either of
those windows, which also steps to the next component. Labels are kept in
``ic_labels.tsv`` beside the decomposition, written when SAVE LABELS is pressed
and read back when the folder is loaded, so a review can stop and resume.

This file is the whole mode. That is the point of the framework: no window code
changed to add it, and the component slider appears because it is declared here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fastfuncstuff.viewer.commands import Aspect, Command
from fastfuncstuff.viewer.modes.base import (
    ActionControl,
    Control,
    IntControl,
    Mode,
    OverlayKind,
    PathControl,
    ProgressFn,
    Trace,
    mode,
)

#: Filenames in a MELODIC-compatible decomposition directory.
IC_MAPS = ("melodic_IC.nii.gz", "melodic_IC.nii", "ica_maps.nii.gz")
IC_MIX = ("melodic_mix", "ica_timeseries.1D")
IC_FTMIX = ("melodic_FTmix",)
#: Where a review's labels are kept, beside the decomposition they describe.
LABELS_FILE = "ic_labels.tsv"
LABELS = ("signal", "noise")


def find_decomposition(directory: Path) -> Path | None:
    """The directory holding a decomposition, looking one level down too.

    ffs writes its MELODIC-compatible files into a subdirectory, so pointing the
    viewer at a results folder should still find them.
    """
    for candidate in (directory, *sorted(p for p in directory.iterdir() if p.is_dir())):
        if any((candidate / n).exists() for n in IC_MAPS):
            return candidate
    return None


def _first(directory: Path, names: tuple[str, ...]) -> Path | None:
    for n in names:
        p = directory / n
        if p.exists():
            return p
    return None


@mode
class ICAMode(Mode):
    name = "ica"
    label = "ICA"
    tag = "ICA"
    overlay_kind = OverlayKind.COMPONENT

    def controls(self) -> tuple[Control, ...]:
        # The range is discovered, so this is built per call rather than fixed:
        # the panel rebuilds when layers change and picks up the real count.
        return (
            PathControl(
                name="folder",
                label="folder",
                directory=True,
                help="A decomposition folder (melodic_IC + melodic_mix), or a results "
                "folder with one a level down. Empty means the directory READ last.",
            ),
            IntControl(
                name="component",
                label="component",
                lo=0,
                hi=max(self._n_components - 1, 0),
                default=0,
                help="Which independent component to show.",
            ),
        )

    def actions(self) -> tuple[ActionControl, ...]:
        return (
            ActionControl(
                name="prev", label="◀ prev", help="Previous component (Left in a trace window)"
            ),
            ActionControl(
                name="next", label="next ▶", help="Next component (Right in a trace window)"
            ),
            ActionControl(name="signal", label="signal", help="Label signal and step on (s)"),
            ActionControl(name="noise", label="noise", help="Label noise and step on (n)"),
            ActionControl(name="unlabel", label="clear", help="Remove this component's label (u)"),
            ActionControl(
                name="save_labels",
                label="save labels",
                help=f"Write {LABELS_FILE} beside the decomposition",
            ),
            *super().actions(),
        )

    def preparation_params(self) -> frozenset[str]:
        return frozenset({"folder"})

    def panel_names(self) -> tuple[str, ...]:
        return ("timecourse", "spectrum")

    def __init__(self) -> None:
        self._n_components = 0
        self._maps: np.ndarray | None = None
        self._affine: np.ndarray | None = None
        self._mix: np.ndarray | None = None
        self._ftmix: np.ndarray | None = None
        self._tr = 0.0
        self._dir: Path | None = None
        #: component index -> "signal" | "noise". Absent means unlabelled.
        self.labels: dict[int, str] = {}
        self._labels_saved = True
        super().__init__()

    # -- loading -------------------------------------------------------
    def _folder(self) -> Path | None:
        chosen = str(self.params.get("folder") or "").strip()
        if chosen:
            return Path(chosen).expanduser()
        if self.session is not None and self.session.catalog_dir is not None:
            return Path(self.session.catalog_dir)
        return None

    def prepare(self, progress: ProgressFn | None = None) -> bool:
        """Read the decomposition. On the worker: a component stack is a big file."""
        return self._load()

    def _load(self) -> bool:
        """Read the decomposition from the chosen folder, or the READ directory."""
        if self._maps is not None and not self._dirty:
            return True
        folder = self._folder()
        found = find_decomposition(folder) if folder is not None and folder.is_dir() else None
        maps_path = _first(found, IC_MAPS) if found is not None else None
        if found is None or maps_path is None:
            self._maps, self._n_components, self._dir = None, 0, None
            self._dirty = False
            return False
        from fastfuncstuff.io.afni import load_nifti

        img = load_nifti(maps_path)
        arr = np.asanyarray(img.dataobj, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., None]
        self._maps = arr
        self._affine = np.asarray(img.affine, dtype=float)
        self._n_components = int(arr.shape[3])
        if found != self._dir:
            # Only for a different folder: reloading the same one on the way
            # back into the mode must not throw away labels not yet saved.
            self.labels = read_labels(found / LABELS_FILE)
            self._labels_saved = True
        self._dir = found

        mix = _first(found, IC_MIX)
        self._mix = np.loadtxt(mix) if mix is not None else None
        if self._mix is not None and self._mix.ndim == 1:
            self._mix = self._mix[:, None]

        ft = _first(found, IC_FTMIX)
        self._ftmix = np.loadtxt(ft) if ft is not None else None
        if self._ftmix is not None and self._ftmix.ndim == 1:
            self._ftmix = self._ftmix[:, None]

        self._tr = self._source_tr()
        self._dirty = False
        return True

    def _source_tr(self) -> float:
        """TR of any 4-D layer, so the spectrum can be labelled in Hz."""
        if self.session is None:
            return 0.0
        for layer in self.session.state.layers:
            if layer.n_volumes > 1:
                try:
                    tr = float(self.session.store.get(layer.key).info.tr)
                except KeyError:
                    continue
                if tr > 0:
                    return tr
        return 0.0

    @property
    def _index(self) -> int:
        return max(0, min(int(self.params.get("component") or 0), self._n_components - 1))

    def _label_text(self, k: int) -> str:
        return f"[{self.labels[k]}]" if k in self.labels else ""

    # -- production ----------------------------------------------------
    def compute(self):
        from fastfuncstuff.viewer.modes.base import ComputedOverlay

        if not self._load() or self._maps is None or self._affine is None:
            return None
        k = self._index
        values = np.ascontiguousarray(self._maps[..., k])
        finite = values[np.isfinite(values)]
        # Components are conventionally z-scaled, so a symmetric range about
        # zero is what makes two of them comparable at a glance.
        top = float(np.percentile(np.abs(finite), 99.0)) if finite.size else 1.0
        detail = f"IC {k} {self._label_text(k)}".strip()
        return ComputedOverlay(
            values=values,
            affine=self._affine,
            name=self.output_name(detail),
            kind=OverlayKind.COMPONENT,
            colormap="redblue",
            display_range=(-top, top) if top > 0 else None,
            threshold=top * 0.4 if top > 0 else 0.0,
        )

    # -- actions -------------------------------------------------------
    def action(self, name: str, progress: ProgressFn | None = None) -> Aspect:
        if name in ("prev", "next"):
            return self._step(-1 if name == "prev" else 1)
        if name in LABELS or name == "unlabel":
            if self._maps is None:
                return Aspect.NOTHING
            k = self._index
            if name == "unlabel":
                self.labels.pop(k, None)
                self._labels_saved = False
                return self.refresh()
            self.labels[k] = name
            self._labels_saved = False
            # Labelling is a decision about this one; the next is what you want
            # to see. Stays put on the last component rather than wrapping, so
            # the end of a review is visible as the end.
            if k < self._n_components - 1:
                return self._step(1)
            return self.refresh()
        if name == "save_labels":
            if self._dir is None:
                raise ValueError("no decomposition loaded, so nowhere to save labels")
            write_labels(self._dir / LABELS_FILE, self.labels, self._n_components)
            self._labels_saved = True
            return Aspect.GRAPH
        return super().action(name, progress)

    def _step(self, delta: int) -> Aspect:
        if self._n_components == 0:
            return Aspect.NOTHING
        target = max(0, min(self._index + delta, self._n_components - 1))
        return self.set_param("component", target) | Aspect.GRAPH

    def noise_components(self) -> list[int]:
        """Components labelled noise -- what a denoise would regress out."""
        return sorted(k for k, v in self.labels.items() if v == "noise")

    def mixing_matrix(self) -> np.ndarray | None:
        """``(T, K)`` component time courses, or ``None`` before a load."""
        return None if self._mix is None else np.asarray(self._mix, dtype=np.float64)

    # -- graph ---------------------------------------------------------
    def _timecourse(self, k: int) -> Trace | None:
        if self._mix is None or k >= self._mix.shape[1]:
            return None
        return Trace(
            label=f"IC {k} time course {self._label_text(k)}".strip(),
            values=np.asarray(self._mix[:, k]),
            x_label="TR",
            key="timecourse",
            short="IC time course",
        )

    def _spectrum(self, k: int) -> Trace | None:
        """MELODIC's own FTmix when present, else the power spectrum of the mix.

        FTmix first because a spectrum derived here by a slightly different
        convention would disagree with every other tool reading this folder.
        """
        values, what = None, ""
        if self._ftmix is not None and k < self._ftmix.shape[1]:
            values, what = np.asarray(self._ftmix[:, k]), "spectrum (melodic_FTmix)"
        elif self._mix is not None and k < self._mix.shape[1]:
            ts = self._mix[:, k]
            values, what = np.abs(np.fft.rfft(ts - ts.mean()))[1:] ** 2, "power spectrum"
        if values is None or not values.size:
            return None
        x = None
        x_label = "bin"
        if self._tr > 0:
            nyquist = 0.5 / self._tr
            x = np.linspace(nyquist / values.size, nyquist, values.size)
            x_label = "Hz"
        return Trace(
            label=f"IC {k} {what}",
            values=values,
            x=x,
            x_label=x_label,
            key="spectrum",
            short="IC spectrum",
        )

    def panels(self) -> dict[str, Trace]:
        if self._maps is None:
            return {}
        k = self._index
        out = {}
        for name, trace in (("timecourse", self._timecourse(k)), ("spectrum", self._spectrum(k))):
            if trace is not None:
                out[name] = trace
        return out

    def series(self, ijk: tuple[int, int, int]) -> list[Trace]:
        """The component's time course, and its spectrum, for grid graphs.

        Not voxel-wise: a component has one time course for the whole brain, so
        these are the same in every cell of a grid graph. The trace windows are
        the readable place for them; here they sit beside the voxel traces they
        are supposed to explain.
        """
        if self._maps is None:
            return []
        k = self._index
        # Both, always: whether the spectrum is drawn is a tick box on each
        # graph window, not a mode-wide switch that hides it from all of them.
        return [t for t in (self._timecourse(k), self._spectrum(k)) if t is not None]

    # -- reaction ------------------------------------------------------
    def on_command(self, cmd: Command, dirty: Aspect) -> Aspect:
        # A new directory means a different decomposition -- unless a folder
        # was chosen, which READ does not change.
        if cmd.name == "READ" and not str(self.params.get("folder") or "").strip():
            self._maps = None
            self.invalidate()
            return self.refresh()
        return Aspect.NOTHING

    def status(self) -> str:
        if self._preparing:
            return "ica: loading…"
        if self._n_components == 0:
            return "ica: no decomposition found — choose a folder containing melodic_IC"
        where = self._dir.name if self._dir else "?"
        n_sig = sum(1 for v in self.labels.values() if v == "signal")
        n_noise = sum(1 for v in self.labels.values() if v == "noise")
        todo = self._n_components - n_sig - n_noise
        here = self.labels.get(self._index, "unlabelled")
        unsaved = "  · unsaved" if not self._labels_saved else ""
        return (
            f"ica: IC {self._index}/{self._n_components - 1} {here}  ·  "
            f"{n_sig} signal, {n_noise} noise, {todo} left ({where}){unsaved}"
        )


def read_labels(path: Path) -> dict[int, str]:
    """``component<TAB>label`` lines; anything unreadable is simply unlabelled."""
    out: dict[int, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1] in LABELS:
            out[int(parts[0])] = parts[1]
    return out


def write_labels(path: Path, labels: dict[int, str], n_components: int) -> None:
    """Every component, labelled or not, so the file says how far the review got."""
    rows = ["component\tlabel"]
    rows += [f"{k}\t{labels.get(k, 'unlabelled')}" for k in range(n_components)]
    path.write_text("\n".join(rows) + "\n")
