"""Viewports: every image and every graph is an addressable object.

The viewer used to keep three image panes in a dict keyed by plane and three
graph windows in another, which quietly made *plane* the identity of a window.
Everything that wants two windows of the same plane -- one soloed on the EPI
and one on the anat, to flip between them -- or a per-window zoom, or a layout
you can save, was blocked on that single assumption.

So a window is a :class:`Viewport`: a record with an id, held in
:class:`ViewerState`, created and configured through commands. Two consequences
are worth stating because they are the reason for the change:

* **Zoom, pan and locks finally have an owner.** They were global state with
  nothing drawing from them, because "the zoom of the viewer" is not a thing
  that exists once there is more than one image on screen.
* **Layout save/restore is not a feature.** The viewports are state, state is
  built by commands, and commands are what a recorded script is made of -- so a
  replayed script rebuilds the windows it was recorded with.

Nothing here imports Qt. The window manager in :mod:`viewer.ui.manager`
reconciles real windows against this list and owns no layout truth of its own.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from enum import StrEnum


class Plane(StrEnum):
    """The three cardinal display planes.

    Oblique views are a display grid with a rotated affine rather than a fourth
    plane, so this stays closed.

    It lives here rather than in :mod:`viewer.state` only because a viewport
    takes one as a field default and state holds the viewports; state
    re-exports it, which is still where the rest of the viewer imports it from.
    """

    AXIAL = "axial"
    SAGITTAL = "sagittal"
    CORONAL = "coronal"


#: Cells per side in a graph viewport. The grid is stepped with + and - rather
#: than chosen from a menu of 1/4/9: it is a square that grows, and the useful
#: size depends on voxel size and on what you are chasing, not on three blessed
#: values. The cap is where a cell stops being readable at a sane window size,
#: not where it stops being fast -- traces decimate to the cell width, so 256
#: cells paint in about the time 16 used to.
MIN_GRID = 1
MAX_GRID = 16


class ViewKind(StrEnum):
    IMAGE = "image"
    GRAPH = "graph"
    CARPET = "carpet"
    MATRIX = "matrix"
    CLUSTERS = "clusters"
    TRACE = "trace"


@dataclass(frozen=True)
class Viewport:
    """One window: what it shows and how, but never any voxels.

    Frozen and scalar-only for the same reason :class:`~viewer.layers.Layer`
    is -- it has to be cheap to copy and safe to write into a script.
    """

    id: str
    kind: ViewKind
    plane: Plane = Plane.AXIAL

    #: Follow the shared crosshair. An unlocked viewport keeps the slice it
    #: was parked on, which is how you leave a reference view up while
    #: navigating somewhere else.
    locked: bool = True
    #: The parked slice, set when a viewport is unlocked. ``None`` means it is
    #: following the crosshair and has no slice of its own -- which is why it
    #: is a separate field rather than a number that is sometimes ignored.
    position: int | None = None

    # -- image ---------------------------------------------------------
    zoom: float = 1.0
    pan: tuple[float, float] = (0.0, 0.0)
    #: Draw only the selected layer instead of compositing the stack. The
    #: point is flipping: with an EPI and an anat aligned in one stack, `[`
    #: and `]` alternate between them in place, which is the way to see what
    #: registration actually did. Composite (the default) is the ordinary
    #: underlay/overlay behaviour.
    solo: bool = False

    # -- graph ---------------------------------------------------------
    grid_n: int = 2
    #: Which layers to plot, as layer keys. Empty means "every time-linked
    #: layer", which is the right default and also the only one that keeps
    #: working as the stack grows. A 3-D layer is never plotted and never
    #: offered -- an anatomy has no time course to draw.
    traces: tuple[str, ...] = ()
    shared_scale: bool = True
    #: Lines a graph window has ticked off, by identity: a layer key, or
    #: ``mode:<trace key>`` for a line a mode adds. The set to *hide* rather
    #: than to show, for the same reason ``traces`` defaults to everything: a
    #: line that appears later -- a mode's spectrum -- starts drawn.
    hidden: tuple[str, ...] = ()
    #: Design columns kept in a graph, as :class:`viewer.design.Pin` specs
    #: (``run:col:path``). Separate from ``traces`` because a regressor is not
    #: a layer: it has no voxels, and it is the same line in every cell.
    regressors: tuple[str, ...] = ()

    # -- carpet and matrix ---------------------------------------------
    #: Row order. See :mod:`viewer.carpet` for what each one groups.
    order: str = "pc1"
    #: Node order in a matrix window. A separate field from ``order`` because
    #: the two vocabularies have nothing in common -- "corr with the seed
    #: voxel" is not a thing you can do to a connectivity matrix, and sharing
    #: one field would mean a window that changed meaning when you switched it.
    matrix_order: str = "hierarchical"
    #: Which ROI layer supplies a matrix's nodes, as a layer key. Empty means
    #: the topmost ROI layer, and no ROI layer at all means the matrix falls
    #: back to bins of voxels.
    rois: str = ""
    #: Legendre drift projected out before drawing. A carpet is unreadable
    #: through a linear ramp and a correlation between two undetrended runs is
    #: mostly a correlation between two drifts; anything richer is a job for
    #: DERIVE, and then this window is pointed at the result.
    detrend: int = 1
    #: ``z`` or ``psc``.
    scaling: str = "z"

    # -- trace -----------------------------------------------------------
    #: Which of the active mode's named panels a trace window shows --
    #: ``timecourse``, ``spectrum``. A name rather than an index, so a script
    #: that opened the spectrum window still opens it when a mode adds a panel.
    panel: str = ""

    #: Last known on-screen rectangle, so a saved session comes back where it
    #: was. The window manager writes it; nothing else reads it.
    geometry: tuple[int, int, int, int] | None = None

    def with_(self, **changes: object) -> Viewport:
        return replace(self, **changes)  # type: ignore[arg-type]

    @property
    def is_image(self) -> bool:
        return self.kind is ViewKind.IMAGE

    @property
    def is_graph(self) -> bool:
        return self.kind is ViewKind.GRAPH

    @property
    def is_carpet(self) -> bool:
        return self.kind is ViewKind.CARPET

    @property
    def is_matrix(self) -> bool:
        return self.kind is ViewKind.MATRIX

    @property
    def is_clusters(self) -> bool:
        return self.kind is ViewKind.CLUSTERS

    @property
    def is_trace(self) -> bool:
        return self.kind is ViewKind.TRACE

    @property
    def cells(self) -> int:
        return self.grid_n * self.grid_n

    @property
    def title(self) -> str:
        """What the window's title bar says.

        Includes the id because with several windows of the same plane open,
        "axial" alone stops identifying anything -- and the id is what a
        script addresses, so seeing it is what makes a recording readable.
        """
        if self.is_carpet:
            return f"carpet · {self.order}  [{self.id}]"
        if self.is_matrix:
            return f"matrix · {self.matrix_order}  [{self.id}]"
        if self.is_clusters:
            return f"clusters  [{self.id}]"
        if self.is_trace:
            return f"{self.panel or 'trace'}  [{self.id}]"
        what = self.plane.value if self.is_image else f"graph · {self.plane.value}"
        extra = " · solo" if self.solo else ""
        if self.is_graph:
            extra = f" · {self.cells}"
        return f"{what}{extra}  [{self.id}]"


@dataclass
class ViewportSet:
    """The open windows, in the order they were opened."""

    viewports: list[Viewport] = field(default_factory=list)
    _seq: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.viewports)

    def __iter__(self) -> Iterator[Viewport]:
        return iter(self.viewports)

    @property
    def ids(self) -> list[str]:
        return [v.id for v in self.viewports]

    def mint_id(self, kind: ViewKind) -> str:
        """A fresh id: ``V1`` image, ``G1`` graph, ``C1`` carpet, ``M1`` matrix, ``K1`` clusters."""
        stem = {
            ViewKind.IMAGE: "V",
            ViewKind.GRAPH: "G",
            ViewKind.CARPET: "C",
            ViewKind.MATRIX: "M",
            ViewKind.CLUSTERS: "K",
            ViewKind.TRACE: "T",
        }[kind]
        n = self._seq.get(stem, 0)
        while True:
            n += 1
            candidate = f"{stem}{n}"
            if not any(v.id == candidate for v in self.viewports):
                self._seq[stem] = n
                return candidate

    def find(self, vid: str) -> Viewport | None:
        for v in self.viewports:
            if v.id == vid:
                return v
        return None

    def get(self, vid: str) -> Viewport:
        found = self.find(vid)
        if found is None:
            raise KeyError(f"no viewport {vid!r}")
        return found

    def add(self, viewport: Viewport) -> Viewport:
        if self.find(viewport.id) is not None:
            raise ValueError(f"viewport {viewport.id!r} already open")
        self.viewports.append(viewport)
        return viewport

    def open(self, kind: ViewKind, plane: Plane, *, vid: str | None = None) -> Viewport:
        # A carpet defaults to removing the linear trend because it is
        # unreadable through a ramp. A graph defaults to removing nothing,
        # because a time course drawn in its own units, at its own level, is
        # what a graph is *for* -- detrending it is a question you ask, not the
        # state you start in.
        detrend = -1 if kind is ViewKind.GRAPH else Viewport.detrend
        return self.add(
            Viewport(id=vid or self.mint_id(kind), kind=kind, plane=plane, detrend=detrend)
        )

    def close(self, vid: str) -> Viewport:
        v = self.get(vid)
        self.viewports.remove(v)
        return v

    def update(self, vid: str, **changes: object) -> Viewport:
        i = self.viewports.index(self.get(vid))
        self.viewports[i] = self.viewports[i].with_(**changes)
        return self.viewports[i]

    def of_kind(self, kind: ViewKind) -> list[Viewport]:
        return [v for v in self.viewports if v.kind is kind]

    @property
    def images(self) -> list[Viewport]:
        return self.of_kind(ViewKind.IMAGE)

    @property
    def graphs(self) -> list[Viewport]:
        return self.of_kind(ViewKind.GRAPH)

    @property
    def carpets(self) -> list[Viewport]:
        return self.of_kind(ViewKind.CARPET)

    @property
    def matrices(self) -> list[Viewport]:
        return self.of_kind(ViewKind.MATRIX)

    @property
    def cluster_views(self) -> list[Viewport]:
        return self.of_kind(ViewKind.CLUSTERS)


def clamp_grid(n: int) -> int:
    return max(MIN_GRID, min(int(n), MAX_GRID))


__all__ = [
    "MAX_GRID",
    "MIN_GRID",
    "Plane",
    "ViewKind",
    "Viewport",
    "ViewportSet",
    "clamp_grid",
]
