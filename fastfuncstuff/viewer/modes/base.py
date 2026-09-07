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

That is the whole contract. A new mode is one file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from fastfuncstuff.viewer.commands import Aspect, Command

if TYPE_CHECKING:
    from fastfuncstuff.viewer.session import ViewerSession


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


# ---------------------------------------------------------------------------
# the mode itself
# ---------------------------------------------------------------------------


class Mode(ABC):
    """Base class. Subclasses live one per file under ``viewer/modes/``."""

    name: ClassVar[str] = ""
    label: ClassVar[str] = ""
    overlay_kind: ClassVar[OverlayKind] = OverlayKind.VALUE
    #: Whether this mode owns an overlay layer at all. Plain mode does not --
    #: its overlay is whatever the user picked.
    produces_overlay: ClassVar[bool] = True

    def __init__(self) -> None:
        self.session: ViewerSession | None = None
        self.params: dict[str, Any] = {c.name: getattr(c, "default", None) for c in self.controls()}
        self._dirty = True

    # -- lifecycle -----------------------------------------------------
    def attach(self, session: ViewerSession) -> None:
        self.session = session
        self._dirty = True

    def detach(self) -> None:
        """Drop the mode's overlay so switching modes leaves no residue."""
        if self.session is not None and self.produces_overlay:
            self.session.remove_computed_overlay(self.layer_source)
        self.session = None

    @property
    def layer_source(self) -> str:
        return f"mode:{self.name}"

    # -- declaration ---------------------------------------------------
    def controls(self) -> Sequence[Control]:
        """Parameters the UI should offer. Static per mode."""
        return ()

    def set_param(self, name: str, value: Any) -> Aspect:
        """Update a parameter and recompute if it changed."""
        if self.params.get(name) == value:
            return Aspect.NOTHING
        self.params[name] = value
        self.invalidate()
        return self.refresh()

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

    # -- production ----------------------------------------------------
    @abstractmethod
    def compute(self) -> ComputedOverlay | None:
        """Produce the overlay, or ``None`` if inputs are not ready."""

    def refresh(self) -> Aspect:
        """Recompute and install the overlay."""
        if self.session is None or not self.produces_overlay:
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
    "ModeRegistry",
    "OverlayKind",
    "Trace",
    "mode",
    "registry",
]
