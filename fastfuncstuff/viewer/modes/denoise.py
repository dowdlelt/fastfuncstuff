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
    ChoiceControl,
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
                name="ica_noise",
                label="ICA noise",
                default=False,
                help="Also regress out the components this tab's ICA review labelled noise.",
            ),
            ChoiceControl(
                name="ica_style",
                label="ICA style",
                choices=("non-aggressive", "aggressive"),
                default="non-aggressive",
                help="Non-aggressive (ICA-AROMA's default) fits every component and removes only "
                "the noise ones' share; aggressive projects the noise time courses out entirely.",
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
    def default_input(self, candidates):
        """The topmost run this mode did not itself produce.

        Denoising a denoise is almost never what was meant, and the output is
        selected when it lands -- so without this, pressing APPLY twice cleans
        the cleaned run. It is still in the input picker, so chaining is one
        choice away; it just is not what the second press does.
        """
        return next(
            (ly for ly in candidates if not ly.source.startswith("derived:denoise:")),
            super().default_input(candidates),
        )

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
        ica = None
        if self.params.get("ica_noise"):
            ica = self._ica_columns(int(layer.n_volumes))
        self._job = {
            "ica": ica,
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

    def _ica_columns(self, n_time: int) -> tuple[np.ndarray, list[int], str]:
        """This tab's ICA mixing matrix and its noise components, checked now.

        Checked on the GUI thread at APPLY rather than discovered on the worker,
        so "nothing is labelled noise" is a message at the button, not a failed
        job in the status bar a second later.
        """
        assert self.session is not None
        from fastfuncstuff.viewer.modes.ica import ICAMode

        found = self.session.mode_named("ica")
        mix = found.mixing_matrix() if isinstance(found, ICAMode) else None
        noise = found.noise_components() if isinstance(found, ICAMode) else []
        if mix is None:
            raise ValueError(
                "ICA noise is on, but this tab has no decomposition loaded in ICA mode"
            )
        if not noise:
            raise ValueError("ICA noise is on, but no component is labelled noise yet")
        if mix.shape[0] != n_time:
            raise ValueError(
                f"the decomposition has {mix.shape[0]} time points and the run has {n_time}"
            )
        return mix, noise, str(self.params.get("ica_style") or "non-aggressive")

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
        n_time = int(data.shape[-1])
        nuisance = None
        if job["matrix"] or job["polort"] >= 0 or job["ica"] is None:
            nuisance = derive.read_nuisance(
                job["matrix"] or None, n_time=n_time, polort=job["polort"]
            )
        if job["ica"] is None:
            assert nuisance is not None
            clean = derive.denoise(
                data, nuisance, device=job["device"], keep_mean=job["keep_mean"], progress=progress
            )
            description = nuisance.description
        else:
            mix, noise, style = job["ica"]
            is_noise = np.isin(np.arange(mix.shape[1]), noise)
            # Aggressive: only the noise time courses enter the fit, so all of
            # their variance goes. Non-aggressive: every component is fitted and
            # the signal ones stay, taking their share of what they overlap.
            ica_cols = mix if style == "non-aggressive" else mix[:, is_noise]
            ica_remove = is_noise if style == "non-aggressive" else np.ones(len(noise), bool)
            blocks = [ica_cols]
            remove = [ica_remove]
            if nuisance is not None:
                blocks.insert(0, nuisance.columns)
                remove.insert(0, np.ones(nuisance.n_columns, bool))
            clean = derive.denoise_partial(
                data,
                np.concatenate(blocks, axis=1),
                np.concatenate(remove),
                device=job["device"],
                keep_mean=job["keep_mean"],
                progress=progress,
            )
            ica_text = f"ICA noise {len(noise)}/{mix.shape[1]} {style}"
            description = (
                f"{nuisance.description} + {ica_text}" if nuisance is not None else ica_text
            )
        if progress is not None:
            progress(1.0, "variance removed")
        removed = derive.variance_removed(data, clean)
        self._result = (clean, removed, description)
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
