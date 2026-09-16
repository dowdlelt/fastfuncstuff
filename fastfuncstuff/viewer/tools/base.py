"""Preproc tools: where a new *dataset* comes from.

A mode answers "where does the overlay come from". A tool answers a different
question -- "what dataset do I make from the one I have" -- and the two must not
be confused. Motion correction does not produce an overlay; it produces another
run, which you scrub against its own before. So tools are not modes: they are
buttons inside one Preproc mode, and adding a tenth does not put a tenth entry
in the mode selector.

The contract mirrors :class:`~fastfuncstuff.viewer.modes.base.Mode` deliberately,
because the declarative half is what made modes cheap to add:

* :attr:`Tool.input_kind` says what the input dropdown may offer. Motion
  correction wants a time series, so a 3-D anatomical is never listed --
  choosing one is not a mistake worth a later error message.
* :meth:`Tool.controls` returns the rest of the parameters. The dialog renders
  whatever is declared, through the same :class:`ControlPanel` the mode panel
  uses, so "expand it to cover interpolation" is a line of declaration rather
  than widget code.
* :meth:`Tool.run` does the work on a worker thread and returns a
  :class:`ToolOutcome`. It may *read* the session -- that is how it gets the
  voxels -- and must mutate nothing, exactly like ``Mode.prepare``.

**What a tool makes stays in RAM.** Not written to disk, because the point is to
try one, look at it, and try another: running motion correction ten times to
eyeball the interpolation should leave ten files nowhere. The output replaces
the previous one under the same name (``A_MOCO``), so re-running refines rather
than accumulates; comparing two means opening a second controller tab, which is
already how two of anything are compared here. Saving is a deliberate act, and
:meth:`ViewerSession.save_layer` is what does it.

That the result survives a low-RAM machine is :mod:`residency`'s problem, not
this module's: a made dataset spills to a temp directory when the budget bites
and comes back when it is asked for.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from fastfuncstuff.viewer.modes.base import Control, ProgressFn, Trace

if TYPE_CHECKING:
    from fastfuncstuff.viewer.session import ViewerSession


@dataclass
class AuxVolume:
    """A small volume a tool made on the side, to look at rather than to use.

    The teaching half of a preproc step. Motion correction's real output is a
    corrected run, but what *shows* that it worked is the difference between the
    first and last volumes, before against after: structured edges around the
    brain beforehand, noise afterwards.

    These stay in RAM like everything a tool makes, arrive hidden so four of
    them do not bury the anatomy, and can be saved one at a time if one turns
    out to be worth keeping.
    """

    #: Stable within a tool, so re-running replaces this volume rather than
    #: adding a fifth: ``qc_diff_before``, not "difference map 3".
    slot: str
    #: What the layer is called, after the tool's stem: ``A_MOCO diff (before)``.
    name: str
    values: np.ndarray
    labels: tuple[str, ...] = ()
    colormap: str = ""
    #: Centre the display range on zero. What makes a signed difference map
    #: readable, and wrong for anything measured in intensity units.
    symmetric: bool = False


@dataclass
class ToolOutcome:
    """What a tool made, in the form the session installs it from.

    ``values`` is ``(nx, ny, nz, nt)`` -- the viewer's order, not the
    time-first order the processing library works in. Converting at the tool
    boundary keeps the rest of the viewer from having to know that two
    conventions exist.
    """

    values: np.ndarray
    #: Provenance, in words. Lands in the layer's ``path`` field, which is what
    #: the picker shows for a dataset that has no file: ``<moco: heptic, base 0>``.
    detail: str = ""
    #: Sub-brick labels for the output, when it has meaningful ones.
    labels: tuple[str, ...] = field(default_factory=tuple)
    #: The layer this was made from. Filled in by the mode after the tool
    #: returns, rather than by each tool: it is the one piece of bookkeeping
    #: every tool would have to repeat, and forgetting it puts the result in
    #: the wrong place in the stack.
    source_key: str = ""
    #: QC volumes made alongside the result, in the order they should read in
    #: the stack.
    aux: list[AuxVolume] = field(default_factory=list)
    #: Named plots the tool wants opened beside the images, each a set of lines
    #: sharing one y-axis. The key is what the window is titled.
    panels: dict[str, list[Trace]] = field(default_factory=dict)


class Tool(ABC):
    """Base for one preproc tool. Subclasses live one per file under ``tools/``."""

    name: ClassVar[str] = ""
    label: ClassVar[str] = ""
    #: Short upper-case stem for what it makes: ``A_MOCO``.
    tag: ClassVar[str] = ""
    #: One line, shown at the top of the dialog. This is a teaching tool, so it
    #: says what the step *does*, not what the button does.
    blurb: ClassVar[str] = ""
    #: ``"4d"`` restricts the input dropdown to time series; ``"any"`` does not.
    input_kind: ClassVar[str] = "4d"
    #: Short verb for the operation, used to name and to replace the output.
    op: ClassVar[str] = ""

    def controls(self) -> Sequence[Control]:
        """Parameters beyond the input, which the dialog adds for every tool."""
        return ()

    @abstractmethod
    def run(
        self,
        session: ViewerSession,
        params: dict[str, Any],
        progress: ProgressFn | None = None,
    ) -> ToolOutcome:
        """Do the work on a worker thread. Reads the session; mutates nothing."""


class ToolRegistry:
    """Name to tool instance. Modules register themselves on import."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register[T: Tool](self, cls: type[T]) -> type[T]:
        if not cls.name:
            raise ValueError(f"{cls.__name__} must set a name")
        self._tools[cls.name] = cls()
        return cls

    def find(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return [self._tools[n] for n in sorted(self._tools)]


registry = ToolRegistry()
tool = registry.register


__all__ = ["AuxVolume", "Tool", "ToolOutcome", "ToolRegistry", "registry", "tool"]
