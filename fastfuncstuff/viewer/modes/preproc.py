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

from fastfuncstuff.viewer.commands import Aspect
from fastfuncstuff.viewer.modes.base import (
    ActionControl,
    ChoiceControl,
    ComputedOverlay,
    DialogSpec,
    Mode,
    ProgressFn,
    mode,
)
from fastfuncstuff.viewer.tools import registry as tools
from fastfuncstuff.viewer.tools.base import Tool, ToolOutcome


@mode
class PreprocMode(Mode):
    name = "preproc"
    label = "Preproc"
    # It makes datasets, not overlays. Without this the framework would mint an
    # empty A_PREPROC layer and offer a KEEP button for it.
    produces_overlay = False

    def compute(self) -> ComputedOverlay | None:
        return None

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
        """On the GUI thread. Lands above the source, replacing a previous run."""
        if self.session is None:
            return Aspect.NOTHING
        return self.session.install_derived(
            outcome.source_key,
            outcome.values,
            op=tool.op or tool.name,
            detail=outcome.detail,
            name=self.output_name_for(tool),
        )

    def output_name_for(self, tool: Tool) -> str:
        """``A_MOCO``: the controller that made it, and what made it."""
        label = self.session.label if self.session is not None else ""
        stem = tool.tag or tool.name.upper()
        return f"{label}_{stem}" if label else stem

    def status(self) -> str:
        names = ", ".join(t.label for t in tools.all())
        return f"preproc: {names}" if names else "preproc: no tools registered"
