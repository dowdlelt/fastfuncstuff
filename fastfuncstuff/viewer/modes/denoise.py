"""Denoise: project nuisance out of a run, and show what it took.

DERIVE used to be a one-shot button in the controller. It is a mode because
denoising is a loop, not a click: pick what counts as nuisance, apply, look at
what came out, change your mind. The mode's controls are the nuisance spec; its
APPLY button does the projection; and it leaves two layers behind:

* ``A_DENOISE`` -- the cleaned run, just above its source, graphable beside it
  and ready to be the input of another tab's InstaCorr or carpet.
* ``A_DENOISE_VR`` -- the fraction of each voxel's variance that was removed,
  as the mode's map. The first question after any projection is *where did it
  bite*, and a map of that answers it before a carpet has been built.

Nothing runs on a parameter change. A projection is seconds over a whole 4-D
array, and a spin box that re-ran it per click would queue work nobody asked
for; the parameters are read when APPLY is pressed.
"""

from __future__ import annotations

import numpy as np

from fastfuncstuff.viewer.commands import Aspect
from fastfuncstuff.viewer.modes.base import (
    ActionControl,
    BoolControl,
    ComputedOverlay,
    Control,
    IntControl,
    Mode,
    OverlayKind,
    PathControl,
    ProgressFn,
    mode,
)


@mode
class DenoiseMode(Mode):
    name = "denoise"
    label = "Denoise"
    tag = "DENOISE"
    overlay_kind = OverlayKind.VALUE

    def __init__(self) -> None:
        #: Captured on the GUI thread when APPLY is pressed, so preparation
        #: reads only these and never the stack it would be racing.
        self._job: dict | None = None
        self._result: tuple[np.ndarray, np.ndarray, str] | None = None
        self._installed = True
        self._message = ""
        super().__init__()
        # Nothing is pending until APPLY; the base class starts dirty.
        self._dirty = False

    def attach(self, session) -> None:
        super().attach(session)
        # The base marks a fresh mode dirty so it prepares on the way in. There
        # is nothing to prepare before APPLY, and an empty preparation on the
        # worker is exactly what made the first APPLY find the runner busy.
        self._dirty = self._job is not None and self._result is None

    def controls(self) -> tuple[Control, ...]:
        return (
            PathControl(
                name="matrix",
                label="nuisance",
                filter="1D / xmat (*.1D);;All (*)",
                help="An .xmat.1D, whose ColumnGroups say which columns are nuisance, "
                "or a plain 1D file, all of whose columns are -- a motion file, say.",
            ),
            IntControl(
                name="polort",
                label="polort",
                lo=-1,
                hi=9,
                default=2,
                help="Legendre drift columns added on top; -1 for none. Harmless "
                "beside an xmat's own: duplicated directions collapse.",
            ),
            BoolControl(
                name="keep_mean",
                label="keep mean",
                default=True,
                help="Restore each voxel's mean, so the cleaned run shares an axis with the raw one.",
            ),
        )

    def actions(self) -> tuple[ActionControl, ...]:
        return (
            ActionControl(
                name="apply",
                label="apply",
                help="Project the nuisance out of the selected run (or the topmost run).",
            ),
            ActionControl(
                name="carpets",
                label="carpets",
                help="Open carpets of the raw and the denoised run, side by side.",
            ),
        )

    def preparation_params(self) -> frozenset[str]:
        return frozenset()

    # -- which run -----------------------------------------------------
    def source_layer(self):
        """The selected layer if it is a run, else the topmost run that was
        not itself denoised -- denoising a denoise is almost never meant."""
        if self.session is None:
            return None
        state = self.session.state
        chosen = state.layers.find(state.selected) if state.selected else None
        if chosen is not None and chosen.time_linked and chosen.n_volumes > 1:
            if chosen.source.startswith("derived:denoise:") and chosen.derived_from:
                return state.layers.find(chosen.derived_from) or chosen
            return chosen
        for layer in reversed(list(state.layers)):
            if layer.time_linked and layer.n_volumes > 1 and not layer.is_derived:
                return layer
        return None

    # -- actions -------------------------------------------------------
    def action(self, name: str, progress: ProgressFn | None = None) -> Aspect:
        if self.session is None:
            return Aspect.NOTHING
        if name == "apply":
            return self._apply()
        if name == "carpets":
            return self._carpets()
        return super().action(name, progress)

    def _apply(self) -> Aspect:
        assert self.session is not None
        layer = self.source_layer()
        if layer is None:
            self._message = "no run to denoise"
            return Aspect.NOTHING
        self._job = {
            "key": layer.key,
            "name": layer.name,
            "data": self.session.store.ensure_ram(layer.key),
            "matrix": str(self.params.get("matrix") or ""),
            "polort": int(self.params.get("polort", 2)),
            "keep_mean": bool(self.params.get("keep_mean", True)),
            "affine": np.asarray(layer.affine, dtype=float),
            "device": self.session.store.device,
        }
        self._message = f"denoising {layer.name}…"
        self.invalidate()
        return self.refresh()

    def _carpets(self) -> Aspect:
        from fastfuncstuff.viewer.state import Plane
        from fastfuncstuff.viewer.viewports import ViewKind
        from fastfuncstuff.viewer.vocab import SetViewTraces

        assert self.session is not None
        raw = self.source_layer()
        if raw is None:
            return Aspect.NOTHING
        clean = self.session.state.layers.find_by_source(f"derived:denoise:{raw.key}")
        dirty = Aspect.NOTHING
        for layer in [raw] + ([clean] if clean is not None else []):
            vid = self.session.open_view(ViewKind.CARPET, Plane.AXIAL)
            dirty |= self.session.do(SetViewTraces(vid, layer.key))
        return dirty | Aspect.VIEWPORTS

    # -- the slow half -------------------------------------------------
    def prepare(self, progress: ProgressFn | None = None) -> bool:
        from fastfuncstuff.viewer import derive

        if not self._dirty:
            return self._result is not None
        job = self._job
        if job is None:
            self._dirty = False
            return False
        data = job["data"]
        nuisance = derive.read_nuisance(
            job["matrix"] or None, n_time=int(data.shape[-1]), polort=job["polort"]
        )
        clean = derive.denoise(
            data, nuisance, device=job["device"], keep_mean=job["keep_mean"], progress=progress
        )
        if progress is not None:
            progress(1.0, "variance removed")
        removed = derive.variance_removed(data, clean)
        self._result = (clean, removed, nuisance.description)
        self._installed = False
        self._dirty = False
        return True

    def refresh(self, progress: ProgressFn | None = None) -> Aspect:
        """Install the cleaned run once per APPLY, then the map on top of it."""
        if self.session is None or self._preparing:
            return Aspect.NOTHING
        if self.needs_prepare and self.defer_preparation:
            return Aspect.NOTHING
        if not self.prepare(progress) or self._result is None or self._job is None:
            return Aspect.NOTHING
        dirty = Aspect.NOTHING
        if not self._installed:
            clean, _removed, detail = self._result
            dirty |= self.session.install_derived(
                self._job["key"], clean, op="denoise", detail=detail, name=self.output_name()
            )
            self._installed = True
            self._message = f"{self.output_name()}: {detail}"
        return dirty | super().refresh(progress)

    def compute(self) -> ComputedOverlay | None:
        if self._result is None or self._job is None:
            return None
        _clean, removed, _detail = self._result
        return ComputedOverlay(
            values=removed,
            affine=self._job["affine"],
            name=f"{self.output_name()}_VR",
            kind=OverlayKind.VALUE,
            colormap="hot",
            display_range=(0.0, 1.0),
            threshold=0.1,
        )

    def input_layer_key(self) -> str | None:
        return None if self._job is None else self._job["key"]

    def status(self) -> str:
        if self._preparing:
            return "denoise: projecting…"
        return (
            f"denoise: {self._message}" if self._message else "denoise: set nuisance, press APPLY"
        )
