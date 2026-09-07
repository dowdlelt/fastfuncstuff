"""ICA: the overlay is one component of a decomposition, picked by index.

Reads a decomposition directory -- the MELODIC-compatible layout ffs already
writes -- and turns it into something you step through: the component map as the
overlay, its time course and single-sided spectrum as graph traces.

``melodic_FTmix`` is the spectrum MELODIC itself computed, so it is used when
present rather than recomputed; a spectrum derived from the same mixing matrix
by a slightly different convention would disagree with every other tool looking
at the same directory. Without it, the rFFT of the time course is computed here.

This file is the whole mode. That is the point of the framework: no window code
changed to add it, and the component slider appears because it is declared here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fastfuncstuff.viewer.commands import Aspect, Command
from fastfuncstuff.viewer.modes.base import (
    BoolControl,
    Control,
    IntControl,
    Mode,
    OverlayKind,
    Trace,
    mode,
)

#: Filenames in a MELODIC-compatible decomposition directory.
IC_MAPS = ("melodic_IC.nii.gz", "melodic_IC.nii", "ica_maps.nii.gz")
IC_MIX = ("melodic_mix", "ica_timeseries.1D")
IC_FTMIX = ("melodic_FTmix",)


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
    overlay_kind = OverlayKind.COMPONENT

    def controls(self) -> tuple[Control, ...]:
        # The range is discovered, so this is built per call rather than fixed:
        # the panel rebuilds when layers change and picks up the real count.
        return (
            IntControl(
                name="component",
                label="component",
                lo=0,
                hi=max(self._n_components - 1, 0),
                default=0,
                help="Which independent component to show.",
            ),
            BoolControl(
                name="spectrum",
                label="spectrum",
                default=True,
                help="Add the single-sided spectrum as a second graph trace.",
            ),
        )

    def __init__(self) -> None:
        self._n_components = 0
        self._maps: np.ndarray | None = None
        self._affine: np.ndarray | None = None
        self._mix: np.ndarray | None = None
        self._ftmix: np.ndarray | None = None
        self._tr = 0.0
        self._dir: Path | None = None
        super().__init__()

    # -- loading -------------------------------------------------------
    def _load(self) -> bool:
        """Read the decomposition from the session's current directory."""
        if self._maps is not None and not self._dirty:
            return True
        session = self.session
        if session is None or session.catalog_dir is None:
            return False
        found = find_decomposition(Path(session.catalog_dir))
        if found is None:
            return False

        maps_path = _first(found, IC_MAPS)
        if maps_path is None:
            return False
        from fastfuncstuff.io.afni import load_nifti

        img = load_nifti(maps_path)
        arr = np.asanyarray(img.dataobj, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., None]
        self._maps = arr
        self._affine = np.asarray(img.affine, dtype=float)
        self._n_components = int(arr.shape[3])
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
        return ComputedOverlay(
            values=values,
            affine=self._affine,
            name=f"IC {k}",
            kind=OverlayKind.COMPONENT,
            colormap="redblue",
            display_range=(-top, top) if top > 0 else None,
            threshold=top * 0.4 if top > 0 else 0.0,
        )

    # -- graph ---------------------------------------------------------
    def series(self, ijk: tuple[int, int, int]) -> list[Trace]:
        """The component's time course, and its single-sided spectrum.

        Not voxel-wise: a component has one time course for the whole brain, so
        these are the same in every cell of a grid graph. That is correct and
        useful -- it puts the component's own signal beside the voxel traces it
        is supposed to explain.
        """
        if not self._load():
            return []
        k = self._index
        out: list[Trace] = []
        if self._mix is not None and k < self._mix.shape[1]:
            out.append(Trace(label=f"IC {k}", values=self._mix[:, k], x_label="TR"))

        if not self.params.get("spectrum", True):
            return out

        spectrum = None
        if self._ftmix is not None and k < self._ftmix.shape[1]:
            spectrum = self._ftmix[:, k]
        elif self._mix is not None and k < self._mix.shape[1]:
            ts = self._mix[:, k]
            spectrum = np.abs(np.fft.rfft(ts - ts.mean()))[1:]

        if spectrum is not None and spectrum.size:
            nyquist = 0.5 / self._tr if self._tr > 0 else 0.0
            label = f"IC {k} spectrum" + (f" (0-{nyquist:.3g} Hz)" if nyquist else "")
            out.append(Trace(label=label, values=np.asarray(spectrum), x_label="Hz"))
        return out

    # -- reaction ------------------------------------------------------
    def on_command(self, cmd: Command, dirty: Aspect) -> Aspect:
        # A new directory means a different decomposition.
        if cmd.name == "READ":
            self._maps = None
            self.invalidate()
            return self.refresh()
        return Aspect.NOTHING

    def status(self) -> str:
        if self._n_components == 0:
            return "ica: no decomposition found — READ a directory containing melodic_IC"
        where = self._dir.name if self._dir else "?"
        return f"ica: component {self._index} of {self._n_components} ({where})"
