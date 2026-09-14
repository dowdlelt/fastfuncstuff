"""Modes: where the overlay comes from.

The viewer's core is the data selector -- underlay, overlay, +1. A mode changes
only one thing about that core: instead of the overlay being a file you picked,
it is something computed from what is loaded. InstaCorr computes a correlation,
calc evaluates an expression, GLM fits a model, ICA reads a decomposition.

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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Control:
    """Base for a mode parameter the UI should offer."""

    name: str
    label: str
    help: str = ""


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
class ActionControl(Control):
    """A button. Runs :meth:`Mode.action` with this control's name."""


@dataclass(frozen=True)
class DatasetControl(Control):
    """Pick a dataset from the catalog -- a mode's input, not its overlay."""

    kinds: tuple[str, ...] = ()


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

    @property
    def ident(self) -> str:
        return self.key or self.label

    @property
    def legend(self) -> str:
        return self.short or self.label


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
    #: Whether this mode owns an overlay layer at all. Plain mode does not --
    #: its overlay is whatever the user picked.
    produces_overlay: ClassVar[bool] = True

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

    # -- lifecycle -----------------------------------------------------
    def attach(self, session: ViewerSession) -> None:
        self.session = session
        self._dirty = True

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

    def action(self, name: str, progress: ProgressFn | None = None) -> Aspect:
        """Run one declared action on the GUI thread; return what it dirtied."""
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
        if name in self.preparation_params():
            self.invalidate()
        return self.refresh()

    def preparation_params(self) -> frozenset[str]:
        """Parameters whose change forces the slow path.

        Everything else is assumed cheap, so a control that only affects
        ``compute`` does not pay for a re-preparation.
        """
        return frozenset(c.name for c in self.controls())

    def invalidate(self) -> None:
        """Mark cached preparation stale, so the next refresh redoes it."""
        self._dirty = True

    def input_layer_key(self) -> str | None:
        """The layer this mode consumes, if any.

        An input is not a display layer: once InstaCorr is showing a
        correlation, drawing the 4-D series it was computed from on top of the
        anatomy is just noise. The session hides the input while the mode is
        active and restores it on the way out.
        """
        return None

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

    def panels(self) -> dict[str, Trace]:
        """The current line for each of :meth:`panel_names`."""
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

    def register(self, cls: type[Mode]) -> type[Mode]:
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
    "DatasetControl",
    "FloatControl",
    "IntControl",
    "Mode",
    "OptionalFloatControl",
    "PathControl",
    "ActionControl",
    "ProgressFn",
    "ModeRegistry",
    "OverlayKind",
    "Trace",
    "mode",
    "registry",
]
