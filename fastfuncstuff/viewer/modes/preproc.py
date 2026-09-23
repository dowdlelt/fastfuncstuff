"""Preproc: run a processing step and look at what it did.

Not a GUI for the pipeline -- that is what scripts are for, and a button per
CLI flag would be a worse script. This is a teaching surface. It takes the
handful of steps whose effect is *visible* and makes the before and after two
neighbours in the layer stack, because "the volumes were not aligned and now
they are" is a claim you can show in two keypresses and cannot show in a log.

The mode itself holds almost nothing. Each button is a
:class:`~fastfuncstuff.viewer.tools.base.Tool`, each tool opens a
:class:`DialogSpec`, and the dialog runs it on a worker and hands the result
back here to install. What that leaves in this file is the one thing common to
every tool: choosing the input, and putting the output somewhere sensible.

The output goes in RAM under the tool's own name (``A_MOCO``), immediately above
the run it came from, and a second run **replaces** it. That is the difference
between this and the CLI: trying heptic, then wsinc5, then a different reference
is three looks at one layer, not three files. Comparing two of them is what a
second controller tab is for, and keeping one is an explicit save.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fastfuncstuff.viewer.commands import Aspect
from fastfuncstuff.viewer.modes.base import (
    ActionControl,
    ChoiceControl,
    ComputedOverlay,
    DialogSpec,
    Mode,
    ProgressFn,
    Trace,
    mode,
)
from fastfuncstuff.viewer.tools import registry as tools
from fastfuncstuff.viewer.tools.base import AuxVolume, Tool, ToolOutcome


def symmetric_range(*volumes: np.ndarray) -> tuple[float, float] | None:
    """A display window centred on zero, spanning every volume given.

    The 99th percentile of the magnitude rather than the maximum: one hot voxel
    at the edge of the field of view would otherwise set the scale and leave the
    map that matters looking empty.

    Several volumes at once because a scale is only comparable if it is shared.
    A before-and-after pair scaled separately draws corrected noise exactly as
    boldly as uncorrected motion, which is the opposite of what the pair is for.
    """
    finite = [v[np.isfinite(v)] for v in volumes]
    pooled = np.concatenate([f for f in finite if f.size]) if finite else np.array([])
    if not pooled.size:
        return None
    top = float(np.percentile(np.abs(pooled), 99.0))
    return (-top, top) if top > 0 else None


def shared_ranges(aux: list[AuxVolume]) -> dict[str, tuple[float, float] | None]:
    """One display range per named range group."""
    groups: dict[str, list[np.ndarray]] = {}
    for volume in aux:
        if volume.range_group:
            groups.setdefault(volume.range_group, []).append(volume.values)
    return {name: symmetric_range(*values) for name, values in groups.items()}


@mode
class PreprocMode(Mode):
    name = "preproc"
    label = "Preproc"
    # It makes datasets, not overlays. Without this the framework would mint an
    # empty A_PREPROC layer and offer a KEEP button for it.
    produces_overlay = False
    # Each tool picks its own input in its own dialog, so the mode as a whole
    # has none: one INPUT box at the top of the panel would be a second, always
    # wrong, answer to a question the dialogs already ask.
    input_kind = "none"

    def __init__(self) -> None:
        #: Panels from the last tool that ran, by window title. Kept on the mode
        #: rather than the tool because a tool is a stateless description of how
        #: to do something, and this is the result of having done it.
        self._panels: dict[str, list[Trace]] = {}
        super().__init__()

    def compute(self) -> ComputedOverlay | None:
        return None

    def panel_names(self) -> tuple[str, ...]:
        """Nothing until something has run. Preproc has no standing plots."""
        return tuple(self._panels)

    def panels(self) -> dict[str, list[Trace]]:
        return self._panels

    def actions(self) -> tuple[ActionControl, ...]:
        """One button per registered tool; the registry is the only list."""
        return tuple(ActionControl(name=t.name, label=t.label, help=t.blurb) for t in tools.all())

    # -- input ---------------------------------------------------------
    def inputs_for(self, tool: Tool) -> dict[str, str]:
        """Offerable layers for a tool, as ``display name -> layer key``.

        Filtered by what the tool can actually accept, so a 3-D anatomical is
        never listed for motion correction. Names are what a person recognises,
        but two layers can share one, so a collision falls back to showing the
        key as well rather than silently picking whichever came first.
        """
        if self.session is None:
            return {}
        offered = [
            layer
            for layer in self.session.state.layers
            if not layer.roi and (tool.input_kind != "4d" or layer.n_volumes > 1)
        ]
        seen: dict[str, int] = {}
        for layer in offered:
            seen[layer.name] = seen.get(layer.name, 0) + 1
        return {
            (layer.name if seen[layer.name] == 1 else f"{layer.name} [{layer.key}]"): layer.key
            for layer in offered
        }

    def _default_input(self, tool: Tool, choices: dict[str, str]) -> str:
        """The selected layer when it qualifies, else the first thing offered.

        With one redirection: installing an output selects it, so re-opening the
        dialog would otherwise default to the tool's own previous result and a
        second RUN would correct the corrected run. Coming back to MOCO means
        "try that again differently", so the default walks back to the layer the
        output was made from. Chaining is still available -- the output is in
        the dropdown like anything else -- it just is not what you get by
        pressing the same button twice.
        """
        if self.session is None:
            return next(iter(choices), "")
        selected = self.session.state.selected
        layer = self.session.state.layers.find(selected) if selected else None
        if layer is not None:
            made_by = layer.source.split(":", 2)
            if made_by[:2] == ["derived", tool.op or tool.name]:
                selected = made_by[2]
        for label, key in choices.items():
            if key == selected:
                return label
        return next(iter(choices), "")

    # -- dialogs -------------------------------------------------------
    def dialog_for(self, action: str) -> DialogSpec | None:
        tool = tools.find(action)
        if tool is None or self.session is None:
            return None

        choices = self.inputs_for(tool)
        wanted = "a 4-D time series" if tool.input_kind == "4d" else "a dataset"
        input_control = ChoiceControl(
            name="input",
            label="input",
            choices=tuple(choices),
            default=self._default_input(tool, choices),
            help=f"Which loaded layer to run {tool.label} on.",
        )
        params: dict[str, Any] = {"input": input_control.default}
        params.update({c.name: getattr(c, "default", None) for c in tool.controls()})

        return DialogSpec(
            name=tool.name,
            title=tool.label,
            blurb=tool.blurb,
            controls=(input_control, *tool.controls()),
            params=params,
            run=lambda p, progress, t=tool: self._run(t, p, progress),
            install=lambda outcome, t=tool: self._install(t, outcome),
            run_label="run",
            blocked="" if choices else f"nothing loaded that {tool.label} can use — needs {wanted}",
        )

    def _run(self, tool: Tool, params: dict[str, Any], progress: ProgressFn | None) -> ToolOutcome:
        """On the worker. Reads the session, mutates none of it."""
        if self.session is None:
            raise RuntimeError("no session")
        choices = self.inputs_for(tool)
        chosen = str(params.get("input") or "")
        key = choices.get(chosen)
        if key is None:
            raise ValueError(f"{chosen or 'no input'} is not a layer {tool.label} can run on")
        outcome = tool.run(self.session, {**params, "input": key}, progress)
        outcome.source_key = key
        return outcome

    def _install(self, tool: Tool, outcome: ToolOutcome) -> Aspect:
        """On the GUI thread. Lands above the source, replacing a previous run.

        QC volumes go in first and the result last, because each lands directly
        above the source: installing in that order leaves the result adjacent to
        the run it corrected, which is what `[` and `]` flip between, with the
        QC volumes stacked above it in the order the tool listed them.
        """
        if self.session is None:
            return Aspect.NOTHING
        # Replaced, not merged: these describe this run. Leaving the previous
        # tool's plots up beside a new tool's result is how you end up reading
        # the wrong motion parameters.
        self._panels = dict(outcome.panels)
        stem = self.output_name_for(tool)
        op = tool.op or tool.name
        dirty = Aspect.NOTHING
        pooled = shared_ranges(outcome.aux)

        for volume in reversed(outcome.aux):
            dirty |= self.session.install_derived(
                outcome.source_key,
                volume.values,
                op=f"{op}.{volume.slot}",
                detail=outcome.detail,
                name=f"{stem} {volume.name}",
                labels=volume.labels or None,
                colormap=volume.colormap or None,
                display_range=pooled.get(volume.range_group)
                if volume.range_group
                else (symmetric_range(volume.values) if volume.symmetric else None),
                # Four QC volumes switched on at once would bury the anatomy.
                # They are here to be flipped to, not to be drawn over.
                visible=False,
                select=False,
            )

        return dirty | self.session.install_derived(
            outcome.source_key,
            outcome.values,
            op=op,
            detail=outcome.detail,
            name=stem,
        )

    def output_name_for(self, tool: Tool) -> str:
        """``A_MOCO``: the controller that made it, and what made it."""
        label = self.session.label if self.session is not None else ""
        stem = tool.tag or tool.name.upper()
        return f"{label}_{stem}" if label else stem

    def status(self) -> str:
        names = ", ".join(t.label for t in tools.all())
        return f"preproc: {names}" if names else "preproc: no tools registered"
