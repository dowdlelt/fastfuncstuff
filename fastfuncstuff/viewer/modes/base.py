"""Modes: where a computed layer comes from.

The viewer's core is the data selector: load a dataset, and it is a layer. A
mode changes one thing about that -- instead of a layer being a file you
picked, it is something computed from one that is loaded. InstaCorr computes a
correlation, calc evaluates an expression, GLM fits a model, ICA reads a
decomposition.

Which loaded layer a mode reads is :attr:`ViewerState.input_key`, set by the
INPUT picker and resolved by :meth:`ViewerSession.input_layer`. It is a
separate question from what is drawn and from what is selected, so a run can be
switched off in the stack -- or buried under the anatomy -- and still be the
data the fit is made of.

Everything else a mode needs is declared, not coded:

* :attr:`Mode.overlay_kind` tells the threshold control whether it is looking at
  data values or a statistic, which is the only thing that differs about the
  slider between modes.
* :meth:`Mode.controls` returns a list of parameter specs. The UI renders
  whatever the active mode declares and calls back with the new value, so
  adding a mode never means adding widget code.
* :meth:`Mode.series` contributes traces to the graph windows -- an ICA
  component's time course, a GLM's fitted response.
* :meth:`Mode.actions` declares buttons -- KEEP, APPLY -- the same way controls
  are declared, and :meth:`Mode.action` runs one.

**What a mode makes is a layer, and it outlives the mode.** The live output is
named for the controller and the mode (``A_ICORR``) and stays in the stack when
you switch away; coming back picks it up again. KEEP freezes a numbered copy
(``A_ICORR_1``) so two seeds, or two components, can be flipped between. That is
what lets tabs compose modes: B denoises a run, then B's InstaCorr correlates
the result, and ``B_ICORR`` sits beside ``A_ICORR``.

That is the whole contract. A new mode is one file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from fastfuncstuff.viewer.commands import Aspect, Command

if TYPE_CHECKING:
    from fastfuncstuff.viewer.session import ViewerSession


#: ``progress(fraction, message)`` -- called from a worker thread, so an
#: implementation must marshal to the GUI thread itself.
ProgressFn = Callable[[float, str], None]


class OverlayKind(StrEnum):
    """What the overlay's numbers mean, which sets how the slider is labelled."""

    VALUE = "value"  # raw data units
    STATISTIC = "statistic"  # a test statistic; p/q conversion is meaningful
    CORRELATION = "correlation"  # bounded [-1, 1]
    COMPONENT = "component"  # a decomposition weight, z-scaled by convention


# ---------------------------------------------------------------------------
# declarative controls
# ---------------------------------------------------------------------------


#: Width units in one row of a control panel. Six so that halves and thirds
#: both divide it, which is the whole vocabulary a parameter form needs: a
#: path wants the row, three pickers share it, two spinners split it.
ROW_UNITS = 6
FULL, HALF, THIRD = 6, 3, 2


@dataclass(frozen=True)
class Control:
    """Base for a mode parameter the UI should offer."""

    name: str
    label: str
    help: str = ""
    #: How much of a row this control asks for, out of :data:`ROW_UNITS`.
    #: Controls are packed left to right and wrap when the budget runs out.
    #: The default is the whole row, because that is what every control did
    #: before spans existed and a mode that has not thought about layout
    #: should not get a surprising one.
    span: int = FULL
    #: Start a new row even if this control would have fitted on the last.
    #: Grouping is meaning, not just width: ``show`` / ``thresh`` / ``column``
    #: are one thought and belong together on a line of their own.
    newline: bool = False
    #: ``(other parameter, values that reveal this one)``. Hidden otherwise --
    #: but the space is kept, so revealing a control does not shove the rest of
    #: the panel down. A library index beside an HRF set to 'custom' is a knob
    #: that does nothing, and a knob that does nothing is worse than no knob.
    visible_when: tuple[str, tuple[str, ...]] | None = None


@dataclass(frozen=True)
class FloatControl(Control):
    lo: float = 0.0
    hi: float = 1.0
    default: float = 0.0
    step: float = 0.01
    unit: str = ""


@dataclass(frozen=True)
class OptionalFloatControl(FloatControl):
    """A float that can be switched off entirely.

    Rendered as a checkbox beside a slider, with the slider faded when off, so
    a disabled filter reads as disabled instead of as "set to zero, probably" --
    the distinction matters when zero is also a legal value.

    ``off_value`` is what the parameter takes when unchecked; ``on_value`` is
    where it lands when first switched on, since the default is "off" and
    enabling a control that then does nothing is a dead end.
    """

    off_value: float = 0.0
    on_value: float = 0.0


@dataclass(frozen=True)
class IntControl(Control):
    lo: int = 0
    hi: int = 10
    default: int = 0


@dataclass(frozen=True)
class ChoiceControl(Control):
    choices: tuple[str, ...] = ()
    default: str = ""
    #: ``"combo"`` for a drop-down, ``"radio"`` for the choices side by side.
    #: Radio for a short, stable set whose options are worth reading without a
    #: click -- two units, three colour rules. A drop-down hides how many
    #: choices there are and costs a click to find out, which is the wrong
    #: trade for a set of two.
    style: str = "combo"


@dataclass(frozen=True)
class BoolControl(Control):
    default: bool = False


@dataclass(frozen=True)
class PathControl(Control):
    """A file path, typed or browsed. Committed on enter, never per keystroke."""

    default: str = ""
    #: Qt file-dialog filter, e.g. ``"1D / xmat (*.1D);;All (*)"``.
    filter: str = "All (*)"
    #: Browse for a folder rather than a file.
    directory: bool = False


@dataclass(frozen=True)
class TextControl(Control):
    """Free text in a small grammar of the caller's -- a channel list, a few numbers.

    Committed on enter or focus-out, never per keystroke, for the same reason
    a path is: a half-typed value is not a parameter.
    """

    default: str = ""
    #: Shown greyed while the field is empty: what leaving it empty means.
    placeholder: str = ""


#: Separates entries in a :class:`PathListControl` value. A parameter has to
#: survive being written down as one string -- that is what makes it
#: replayable through ``SET_MODE_PARAM`` -- so the list is encoded rather than
#: held as a list. Each entry is ``+path`` or ``-path``, ticked or not.
PATH_LIST_SEP = "|"


@dataclass(frozen=True)
class PathListControl(Control):
    """Several files, each of which can be ticked off without being removed.

    The point is the comparison. One motion file, one respiration trace, one
    set of physio regressors: the question is never "which of these do I want"
    but "what does each of them do to the map", and answering it by deleting a
    path and typing it back in is answering it badly. Unticking keeps the entry
    where it is, so the next fit is one click away and the click is reversible.
    """

    default: str = ""
    filter: str = "All (*)"
    #: Entries are regressor files that can be transformed in place: the list
    #: offers a derivative button and a band splitter, each of which adds
    #: entries in :mod:`fastfuncstuff.viewer.ortvec`'s grammar.
    transforms: bool = False
    #: Seconds per row of those files, for the band splitter's Hz axis. Zero
    #: when the run has no TR, and the splitter says so rather than guessing.
    sample_interval: float = 0.0
    #: Files already open, as ``(label, path)``. When there are any, ``+``
    #: offers them first and the file picker last: what is wanted in the list
    #: is usually what is already on screen.
    offer: tuple[tuple[str, str], ...] = ()

    @staticmethod
    def parse(value: object) -> list[tuple[str, bool]]:
        """``"+a.1D|-b.1D"`` to ``[("a.1D", True), ("b.1D", False)]``."""
        out: list[tuple[str, bool]] = []
        for entry in str(value or "").split(PATH_LIST_SEP):
            entry = entry.strip()
            if not entry:
                continue
            if entry[0] in "+-":
                out.append((entry[1:], entry[0] == "+"))
            else:
                # An unprefixed path is one a person typed or a script wrote
                # by hand. Taking it as enabled is the reading that does what
                # they meant.
                out.append((entry, True))
        return out

    @staticmethod
    def encode(entries: Sequence[tuple[str, bool]]) -> str:
        return PATH_LIST_SEP.join(f"{'+' if on else '-'}{p}" for p, on in entries if p)

    @staticmethod
    def enabled(value: object) -> list[str]:
        """Just the ticked paths, in order -- what a mode actually reads."""
        return [p for p, on in PathListControl.parse(value) if on]


@dataclass(frozen=True)
class ActionControl(Control):
    """A button. Runs :meth:`Mode.action` with this control's name."""


@dataclass(frozen=True)
class DialogSpec:
    """A window one of a mode's buttons opens, described without naming Qt.

    Some actions are not a button press but a small form: pick an input, set two
    parameters, run something slow, keep the result. Declaring that here rather
    than building it in the window is what keeps the rule that a new mode -- or a
    new preproc tool -- is one file and no window code.

    The two callables are deliberately split by thread, the same split
    :meth:`Mode.prepare` and :meth:`Mode.compute` make and for the same reason:
    ``run`` is slow and goes on a worker, ``install`` touches the session and so
    must land back on the thread that paints.
    """

    #: Identifies the dialog, so pressing the same button twice reuses the window
    #: rather than stacking a second one over the first.
    name: str
    title: str
    #: One line under the title saying what the step does.
    blurb: str = ""
    controls: tuple[Control, ...] = ()
    #: Starting values, by control name. The dialog owns them from then on --
    #: a tool's settings are per-run, not mode state that outlives the window.
    params: dict[str, Any] = field(default_factory=dict)
    #: ``run(params, progress) -> result``, on a worker thread.
    run: Callable[[dict[str, Any], ProgressFn | None], Any] | None = None
    #: ``install(result) -> Aspect``, on the GUI thread.
    install: Callable[[Any], Aspect] | None = None
    run_label: str = "run"
    #: Why the dialog cannot run right now, or empty when it can.
    blocked: str = ""
    #: What finishing did, for the status line. Empty says a layer was made,
    #: which is what a preproc tool does; a dialog that writes a file or moves
    #: an existing layer says so instead. A callable is asked after the run,
    #: for an outcome only the result knows (what was written, what to check).
    done: str | Callable[[], str] = ""


# ---------------------------------------------------------------------------
# what a mode produces
# ---------------------------------------------------------------------------


@dataclass
class ComputedOverlay:
    """A volume a mode made, to be installed as the mode's own layer."""

    values: np.ndarray  # (nx, ny, nz)
    affine: np.ndarray
    name: str
    kind: OverlayKind = OverlayKind.VALUE
    colormap: str = "redblue"
    display_range: tuple[float, float] | None = None
    threshold: float | None = None
    #: A second volume to cut on, when what you want to *see* and what you
    #: want to *believe* are different numbers. That is the ordinary case for
    #: a GLM -- colour by effect size, threshold on the t -- and the layer
    #: stack already knows how to do it for a stats bucket read off disk, so a
    #: mode says which volume is which and everything downstream follows:
    #: the colour bar scales the shown one, the p spinner reads the cut one.
    threshold_values: np.ndarray | None = None
    #: What the two volumes are called, once there are two of them to tell
    #: apart -- ``("Faces beta", "Faces t")``.
    volume_labels: tuple[str, ...] = ()
    #: ``("fitt", dof)`` for the threshold volume, in the same shape
    #: :meth:`Layer.stat_spec` returns, so a computed t map earns the p
    #: spinner a loaded one gets. ``None`` when the cut is on a plain value.
    threshold_stat: tuple[str, float | tuple[float, ...] | None] | None = None
    #: Whether this output means something different from the last one, so the
    #: colour scale must follow it. Off by default, because the usual update is
    #: the *same* quantity recomputed -- the next seed, the next component --
    #: and resetting a threshold on every step would undo the gesture a review
    #: is made of. A mode whose output can change *kind* under one layer sets
    #: it: InstaGLM switching a beta map in percent signal change for a t map
    #: is not a rescale of the same picture, it is a different question, and
    #: keeping the old range renders the new one flat.
    rescale: bool = False
    #: A colour view (viewer.chromalayer.ColourSource) instead of values: the
    #: RGB mode's output. ``values`` is unused when this is set.
    colour: Any = None
    #: A stack of volumes instead of one: ``frames(i)`` makes volume ``i`` of
    #: ``n_frames`` on request. The layer steps through them by its own
    #: sub-brick index (SET_VOLUME, `,` / `.` when selected), not the global
    #: time index -- InstaSpectra's one map per frequency bin, where stepping
    #: frequency must not move the run underneath. ``values`` is just a frame,
    #: for the shape and a fallback range; ``frame`` is the one shown when the
    #: layer is first made. The callable must not change after it is handed
    #: over -- KEEP shares it -- so a new look is a new callable.
    frames: Callable[[int], np.ndarray] | None = None
    n_frames: int = 0
    frame: int = 0


@dataclass(frozen=True)
class Mark:
    """A place a mode points at on the slices: a centre, a radius, a colour.

    Drawn by every image window as a dot and the circle where a sphere of
    ``radius_mm`` cuts the slice shown -- the RGB mode's regions A and B.
    """

    mm: tuple[float, float, float]
    radius_mm: float
    rgb: tuple[float, float, float]
    label: str = ""


@dataclass
class Trace:
    """One line for a graph window."""

    label: str
    values: np.ndarray
    x: np.ndarray | None = None
    #: Free-text axis hint, e.g. "TR" or "Hz". The graph shows it; nothing
    #: parses it.
    x_label: str = ""
    #: Stable identity across recomputes -- ``timecourse``, not "IC 3 time
    #: course" -- so a graph's tick box and colour for this line survive
    #: stepping to IC 4. Falls back to the label.
    key: str = ""
    #: Short name for a legend; falls back to the label.
    short: str = ""
    #: Index of the one point that is "where you are" -- the component a
    #: scree is showing -- drawn as a bar across the plot and a dot on the line.
    marker: int | None = None
    #: A second reading of the same line on a right-hand axis: right = left *
    #: ``right_scale``. One line, two units (% of the total variance on the
    #: left, % of the explained variance on the right), not a second line.
    right_scale: float | None = None
    #: Clicking or dragging along the line picks the nearest sample: it sets
    #: the global time index, or, with ``scrub_key``, that layer's sub-brick --
    #: InstaSpectra's spectrum, one bin per frame of its map layer.
    scrub: bool = False
    scrub_key: str = ""
    #: ``(lo, hi)`` in x units, shaded across the plot: the range a marker
    #: stands for, when that is wider than one sample (a summed band).
    band: tuple[float, float] | None = None
    #: Text written at the marker, on the plot: what "where you are" is --
    #: ``0.81 Hz``. Drawn whatever the number of lines, unlike ``label``,
    #: which gives way to a legend once a plot has two.
    marker_label: str = ""

    @property
    def ident(self) -> str:
        return self.key or self.label

    @property
    def legend(self) -> str:
        return self.short or self.label


def correlation_stat(
    n_time: float, polort: int, *, n_ort: int = 0
) -> tuple[str, tuple[float, float, float]]:
    """The ``threshold_stat`` of a map of correlations with one time course.

    AFNI's ``Correl(n, nfit, nort)``: one fitted regressor, and ``polort + 1``
    drift columns plus ``n_ort`` ortvec columns projected out first (just the
    mean when detrending is off), so the p comes from ``n - 2 - polort - n_ort``
    degrees of freedom. Nominal: it assumes independent samples, which fMRI is
    not, and says nothing about a regressor chosen because it fits the data.
    """
    return ("fico", (float(n_time), 1.0, float(max(polort, 0) + 1 + n_ort)))


# ---------------------------------------------------------------------------
# the mode itself
# ---------------------------------------------------------------------------


class Mode(ABC):
    """Base class. Subclasses live one per file under ``viewer/modes/``."""

    name: ClassVar[str] = ""
    label: ClassVar[str] = ""
    #: Short upper-case stem for what this mode makes: ``A_ICORR``, ``B_ICA``.
    tag: ClassVar[str] = ""
    overlay_kind: ClassVar[OverlayKind] = OverlayKind.VALUE
    #: Whether the MODE dropdown lists it. A mode reached by a button instead
    #: -- the ICA folder reviewer, opened by choosing a folder -- is still a
    #: mode, with its panel and its overlay, just not one you pick from a list.
    in_selector: ClassVar[bool] = True
    #: Whether this mode owns an overlay layer at all. Plain mode does not --
    #: its overlay is whatever the user picked.
    produces_overlay: ClassVar[bool] = True
    #: Whether the run the mode read is switched off when its output first
    #: appears. Right for a 4-D run under its own correlation map; wrong for a
    #: decomposition reviewer, where the run is the anatomy you judge each
    #: component's map against.
    hides_input: ClassVar[bool] = True
    #: What this mode can read, as a declaration rather than a search, in the
    #: same words :attr:`~fastfuncstuff.viewer.tools.base.Tool.input_kind`
    #: uses: ``"4d"`` for a time series, ``"any"`` for any loaded layer,
    #: ``"none"`` for a mode that reads nothing from the stack. Each mode used
    #: to find its own input with its own rule -- "the selected layer if it is
    #: a run, else the topmost run" in one, "the first time-linked layer" in
    #: another -- so two modes could silently disagree about what they were
    #: looking at, and neither could be told otherwise.
    input_kind: ClassVar[str] = "4d"
    #: Whether a change to a :meth:`preparation_params` entry waits for GO
    #: unless ``live`` is ticked. For modes whose preparation is seconds of work:
    #: a spinner tapped three times should be one decomposition, not three.
    gated: ClassVar[bool] = False

    #: Set by a UI that runs preparation on a worker. When true, a refresh
    #: that would need the slow path does nothing instead, and the UI is
    #: responsible for preparing and then calling refresh again. Without this
    #: a seed click would run preparation inline and freeze the window --
    #: which is the whole failure the split exists to prevent.
    defer_preparation: bool = False

    def __init__(self) -> None:
        self.session: ViewerSession | None = None
        self._preparing = False
        self.params: dict[str, Any] = {c.name: getattr(c, "default", None) for c in self.controls()}
        self._dirty = True
        #: Preparation parameters changed since the last preparation, waiting for GO.
        self._held = False

    # -- lifecycle -----------------------------------------------------
    def attach(self, session: ViewerSession) -> None:
        self.session = session
        self._dirty = True
        self._held = False

    def detach(self) -> None:
        """Let go of the session. The output layer stays where it is.

        It used to be removed, on the theory that switching modes should leave
        no residue. But the output is the result: an InstaCorr map you switch
        away from to go and denoise is the map you wanted to compare against.
        """
        self.session = None

    def output_name(self, detail: str = "") -> str:
        """``A_ICORR``, or ``A_ICA IC 3`` when the output has an identity of its own."""
        label = self.session.label if self.session is not None else ""
        stem = f"{label}_{self.tag or self.name.upper()}" if label else (self.tag or self.name)
        return f"{stem} {detail}" if detail else stem

    @property
    def layer_source(self) -> str:
        return f"mode:{self.name}"

    # -- declaration ---------------------------------------------------
    def controls(self) -> Sequence[Control]:
        """Parameters the UI should offer. Static per mode."""
        return ()

    def actions(self) -> Sequence[ActionControl]:
        """Buttons the UI should offer. KEEP, for any mode with an output."""
        if not self.produces_overlay:
            return ()
        return (
            ActionControl(
                name="keep",
                label="keep",
                help="Freeze a numbered copy of the current output (A_ICORR_1, _2, ...) "
                "to compare the next one against.",
            ),
        )

    def panel_controls(self) -> Sequence[Control]:
        """The controls the panel shows now: all of them, unless a mode has views.

        :meth:`controls` stays the whole declaration -- what a script may set and
        what the defaults are seeded from -- so a parameter hidden by a view is
        still settable, and takes effect when its view comes back.
        """
        return self.controls()

    def panel_actions(self) -> Sequence[ActionControl]:
        """The buttons the panel shows now; see :meth:`panel_controls`."""
        return self.actions()

    def dialog_for(self, action: str) -> DialogSpec | None:
        """The form this action opens, or ``None`` if it acts immediately.

        Checked before :meth:`action`, so a mode declares a button once and
        decides here whether pressing it does something or asks something first.
        """
        return None

    def action(self, name: str, progress: ProgressFn | None = None) -> Aspect:
        """Run one declared action on the GUI thread; return what it dirtied."""
        if name == "go" and self.gated:
            return self._go()
        if name == "keep" and self.produces_overlay:
            if self.session is None:
                return Aspect.NOTHING
            return self.session.keep_output(self)
        raise KeyError(f"mode {self.name!r} has no action {name!r}")

    def set_param(self, name: str, value: Any) -> Aspect:
        """Update a parameter and recompute if it changed."""
        if self.params.get(name) == value:
            return Aspect.NOTHING
        self.params[name] = value
        if name == "live" and self.gated:
            return self._go() if value and self._held else Aspect.GRAPH
        if name in self.preparation_params():
            return self.stale()
        return self.refresh()

    def stale(self) -> Aspect:
        """What a parameter (or painted ROI) that the preparation reads just changed.

        Gated and not live: remembered for GO, and nothing runs. Otherwise
        re-prepared now.
        """
        if self.gated and not self.params.get("live"):
            self._held = True
            return Aspect.GRAPH  # the status line says changes are waiting
        self.invalidate()
        return self.refresh()

    def _go(self) -> Aspect:
        self.invalidate()
        return self.refresh() | Aspect.GRAPH

    @property
    def pending(self) -> bool:
        """Whether parameters are waiting for GO."""
        return self._held

    def gate_controls(self) -> tuple[Control, ...]:
        """The ``live`` box and GO button of a gated mode, to place in its panel."""
        return (
            BoolControl(
                name="live",
                label="live",
                default=False,
                span=THIRD,
                newline=True,
                help="Re-run as soon as a setting changes. Off: changes wait for GO, so "
                "several can be made without a run after each.",
            ),
            ActionControl(
                name="go",
                label="go",
                span=THIRD,
                help="Apply the settings changed since the last run.",
            ),
        )

    def time_cut_controls(self) -> tuple[Control, ...]:
        """Volumes to leave out at either end of the run: ``drop first`` / ``drop last``."""
        return (
            IntControl(
                name="drop_first",
                label="drop first",
                lo=0,
                hi=999,
                default=0,
                span=HALF,
                newline=True,
                help="Volumes cut from the start of the run before anything is computed "
                "(non-steady-state volumes, a scanner artefact).",
            ),
            IntControl(
                name="drop_last",
                label="drop last",
                lo=0,
                hi=999,
                default=0,
                span=HALF,
                help="Volumes cut from the end of the run.",
            ),
        )

    def time_window(self, n_time: int) -> slice:
        """The volumes kept after ``drop first`` / ``drop last``.

        Refuses a cut that leaves fewer than three volumes: a correlation or a
        decomposition of two points is arithmetic, not a map.
        """
        first = max(0, int(self.params.get("drop_first") or 0))
        last = max(0, int(self.params.get("drop_last") or 0))
        if n_time - first - last < 3:
            raise ValueError(
                f"drop first {first} + drop last {last} leaves under 3 of the run's "
                f"{n_time} volumes"
            )
        return slice(first, n_time - last)

    def preparation_params(self) -> frozenset[str]:
        """Parameters whose change forces the slow path.

        Everything else is assumed cheap, so a control that only affects
        ``compute`` does not pay for a re-preparation.
        """
        return frozenset(c.name for c in self.controls())

    def invalidate(self) -> None:
        """Mark cached preparation stale, so the next refresh redoes it."""
        self._dirty = True
        # Whatever was waiting for GO is in force from the next preparation on.
        self._held = False

    def pending_note(self) -> str:
        return "  ·  changes pending: press GO (or tick live)" if self._held else ""

    def roi_stroke(self, cmd: Command, key: str | None) -> bool:
        """Whether ``cmd`` edited the painted layer ``key``: a finished stroke or an undo."""
        from fastfuncstuff.viewer.vocab import PaintStroke, PaintUndo

        return bool(key) and isinstance(cmd, PaintStroke | PaintUndo) and cmd.key == key

    def accepts(self, layer: Any) -> bool:
        """Whether this mode could read that layer. Drives the input picker.

        Visibility and selection are deliberately not consulted: whether a run
        is ticked is a statement about the picture, not about what the numbers
        are good for, and the whole point of naming an input is to be able to
        fit one thing while looking at another.
        """
        if self.input_kind == "none" or layer.is_computed:
            # A mode's own live output is not an input. A ``derived:`` layer --
            # a denoised copy of a run -- is, which is what lets one mode's
            # result be the next one's data.
            return False
        if self.input_kind == "4d":
            return bool(layer.time_linked and layer.n_volumes > 1)
        return True

    def default_input(self, candidates: Sequence[Any]) -> Any | None:
        """Which candidate to read when nobody has named one. Topmost first.

        A hook rather than a fixed rule because "the obvious one" is
        mode-specific: Denoise walks past a run it denoised itself, since
        denoising a denoise is almost never what was meant -- while still
        offering it, because chaining is not forbidden, just not the default.
        """
        return candidates[0] if candidates else None

    def source_layer(self) -> Any | None:
        """The layer this mode reads: the named input, or the best candidate."""
        return None if self.session is None else self.session.input_layer(self)

    def source_tr(self) -> float:
        """The TR of the run this mode reads, from its header; 0 when unknown.

        Before any preparation, so an ortvec list's band splitter has its Hz
        axis before the first fit -- choosing the bands is part of deciding
        what to fit.
        """
        layer = self.source_layer()
        if layer is None or self.session is None:
            return 0.0
        try:
            return float(self.session.store.get(layer.key).info.tr or 0.0)
        except (KeyError, AttributeError):
            return 0.0

    def input_layer_key(self) -> str | None:
        """The layer this mode consumes, if any.

        What :meth:`~ViewerSession.forget` will not free. Modes that cache a
        prepared array override this to report the key they prepared *from*,
        which can lag the current answer by one re-preparation.
        """
        layer = self.source_layer()
        return None if layer is None else layer.key

    # -- reaction ------------------------------------------------------
    def on_command(self, cmd: Command, dirty: Aspect) -> Aspect:
        """React to a dispatched command. Default: nothing."""
        return Aspect.NOTHING

    def series(self, ijk: tuple[int, int, int]) -> list[Trace]:
        """Extra graph traces at a voxel."""
        return []

    def panel_names(self) -> tuple[str, ...]:
        """Named lines this mode shows in windows of their own, opened on entry."""
        return ()

    def marks(self) -> list[Mark]:
        """Points with a radius to draw on every slice. Default: none."""
        return []

    def panels(self) -> dict[str, list[Trace]]:
        """The current lines for each of :meth:`panel_names`.

        A list because the natural unit of a panel is sometimes a set -- six
        motion parameters mean nothing plotted apart. Lines in one panel share
        a y-axis, so they must share units: that is what makes translation and
        rotation two panels rather than one crowded one.
        """
        return {}

    # -- production ----------------------------------------------------
    #
    # Split in two because the halves have wildly different costs. InstaCorr
    # preparation is ~600 ms on a small dataset and seconds on a real one,
    # while the correlation itself is ~1 ms. Keeping them separate is what lets
    # the UI run the slow half on a worker with a progress bar and the fast
    # half per interaction, instead of freezing on every click.

    def prepare(self, progress: ProgressFn | None = None) -> bool:
        """Do the expensive, cacheable work. Must not touch session state.

        Called off the GUI thread, so it may only read the session -- anything
        it mutates would be a data race with the paint it is about to trigger.
        Returns whether the mode is ready to compute.
        """
        return True

    @abstractmethod
    def compute(self) -> ComputedOverlay | None:
        """Produce the overlay from prepared data. Must be fast."""

    @property
    def needs_prepare(self) -> bool:
        """Whether the next refresh would do slow work."""
        return self._dirty

    @property
    def preparing(self) -> bool:
        """Whether a worker is currently inside :meth:`prepare`."""
        return self._preparing

    def refresh(self, progress: ProgressFn | None = None) -> Aspect:
        """Prepare if needed, then compute and install.

        Blocking, unless a worker already owns preparation or the UI has asked
        to run it itself.
        """
        if self.session is None or not self.produces_overlay:
            return Aspect.NOTHING
        if self._preparing:
            return Aspect.NOTHING
        if self.needs_prepare and self.defer_preparation:
            return Aspect.NOTHING
        if not self.prepare(progress):
            return Aspect.NOTHING
        overlay = self.compute()
        if overlay is None:
            return Aspect.NOTHING
        self.session.install_computed_overlay(self.layer_source, overlay)
        return Aspect.LAYERS | Aspect.SLICES

    # -- status --------------------------------------------------------
    def status(self) -> str:
        """One line for the status bar; what the mode is currently doing."""
        return ""


class ModeRegistry:
    """Name to mode class. Modules register themselves on import."""

    def __init__(self) -> None:
        self._modes: dict[str, type[Mode]] = {}

    def register[M: Mode](self, cls: type[M]) -> type[M]:
        if not cls.name:
            raise ValueError(f"{cls.__name__} must set a name")
        self._modes[cls.name] = cls
        return cls

    def get(self, name: str) -> type[Mode]:
        try:
            return self._modes[name]
        except KeyError:
            raise KeyError(
                f"unknown mode {name!r}; have {', '.join(sorted(self._modes))}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._modes)

    def labels(self) -> dict[str, str]:
        return {n: self._modes[n].label or n for n in self.names()}


registry = ModeRegistry()
mode = registry.register


__all__ = [
    "BoolControl",
    "ChoiceControl",
    "ComputedOverlay",
    "Control",
    "correlation_stat",
    "DialogSpec",
    "FloatControl",
    "IntControl",
    "Mode",
    "OptionalFloatControl",
    "PathControl",
    "PathListControl",
    "PATH_LIST_SEP",
    "ROW_UNITS",
    "FULL",
    "HALF",
    "THIRD",
    "ActionControl",
    "ProgressFn",
    "ModeRegistry",
    "OverlayKind",
    "Trace",
    "mode",
    "registry",
]
