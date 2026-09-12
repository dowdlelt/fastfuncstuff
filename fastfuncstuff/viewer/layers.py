"""The layer stack: an ordered set of volumes composited bottom-up.

AFNI fixes three roles -- one underlay, one overlay, one threshold sub-brick --
and every later feature that wanted to show a fourth thing had to invent its own
mechanism. Here a layer is just a layer, and "underlay" is a position rather
than a type. That is what lets a second contrast, an atlas, a warp field and a
set of echoes coexist without special cases.

Display parameters live on the layer; the voxels do not. Residency is
:mod:`fastfuncstuff.viewer.residency`'s problem, so a layer stays cheap to copy
and safe to hold in undo history.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from enum import StrEnum

import numpy as np


class SignMode(StrEnum):
    """Which side of zero a layer shows."""

    BOTH = "both"
    POS = "pos"
    NEG = "neg"


class AlphaMode(StrEnum):
    """How sub-threshold voxels fade.

    ``LINEAR`` and ``QUADRATIC`` reproduce AFNI's alpha ramp, which is the
    honest way to draw a statistic map: a voxel just under threshold stops being
    invisible and starts being faint.
    """

    OFF = "off"
    LINEAR = "linear"
    QUADRATIC = "quadratic"


@dataclass(frozen=True)
class Layer:
    """One volume in the stack, plus how it is drawn.

    ``key`` is stable for the life of the layer and is what commands address, so
    a recorded script keeps working when the stack is reordered.
    """

    key: str
    name: str
    path: str
    shape: tuple[int, int, int]
    n_volumes: int
    affine: np.ndarray
    #: Sub-brick labels from the header, when the file carries them --
    #: ``("Full_Fstat", "Faces#0_Coef", "Faces#0_Tstat", ...)``. ffs and
    #: 3dDeconvolve write these; a raw time series has none. Empty means the
    #: sub-bricks have no names, not that they were not read.
    labels: tuple[str, ...] = ()

    visible: bool = True
    opacity: float = 1.0
    colormap: str = "gray"
    n_panes: int = 0  # 0 = continuous; >0 = AFNI-style discrete colour bar
    sign_mode: SignMode = SignMode.BOTH
    volume_index: int = 0
    #: Whether the global time index drives this layer. True for a time series
    #: (a dataset with a TR), False for a stats dataset, whose sub-bricks are
    #: unrelated contrasts rather than time points. Both are 4-D on disk, so
    #: dimensionality cannot tell them apart -- scrubbing a stats dataset
    #: through its contrasts would be nonsense.
    time_linked: bool = False
    #: Where the voxels came from. ``"file"`` for anything loaded off disk;
    #: ``"mode:<name>"`` for an overlay a mode computes and owns, which the
    #: mode replaces in place and the picker must not offer to reload.
    source: str = "file"

    #: Display range. ``None`` means "derive from the data" -- resolved once the
    #: volume is resident, then cached here.
    range_lo: float | None = None
    range_hi: float | None = None

    threshold: float = 0.0
    #: Sub-brick supplying the threshold statistic. ``None`` thresholds on the
    #: displayed volume itself, which is what a plain intensity map wants.
    threshold_index: int | None = None
    alpha_mode: AlphaMode = AlphaMode.OFF
    boxed: bool = False

    def with_(self, **changes: object) -> Layer:
        """Return a copy with fields replaced."""
        return replace(self, **changes)  # type: ignore[arg-type]

    @property
    def is_4d(self) -> bool:
        return self.n_volumes > 1

    @property
    def is_computed(self) -> bool:
        return self.source.startswith("mode:")

    def sub_brick(self, index: int | None = None) -> str:
        """How one sub-brick should be named on screen.

        The index is always shown, even when there is a label: the label is
        what you recognise and the index is what a command takes, and a readout
        that gives only one of them makes you go and look up the other.
        """
        i = self.volume_index if index is None else index
        if 0 <= i < len(self.labels) and self.labels[i]:
            return f"#{i} {self.labels[i]}"
        return f"#{i}"


@dataclass
class LayerStack:
    """Ordered layers, bottom first.

    Index 0 is drawn first and everything else composites over it, so "the
    underlay" is simply ``stack[0]``.
    """

    layers: list[Layer] = field(default_factory=list)
    _seq: int = 0

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self) -> Iterator[Layer]:
        return iter(self.layers)

    def __getitem__(self, index: int) -> Layer:
        return self.layers[index]

    @property
    def keys(self) -> list[str]:
        return [ly.key for ly in self.layers]

    def mint_key(self, stem: str = "L") -> str:
        """A key no layer in this stack currently uses."""
        self._seq += 1
        key = f"{stem}{self._seq}"
        while any(ly.key == key for ly in self.layers):
            self._seq += 1
            key = f"{stem}{self._seq}"
        return key

    def index_of(self, key: str) -> int:
        for i, ly in enumerate(self.layers):
            if ly.key == key:
                return i
        raise KeyError(f"no layer {key!r}")

    def get(self, key: str) -> Layer:
        return self.layers[self.index_of(key)]

    def find(self, key: str) -> Layer | None:
        try:
            return self.get(key)
        except KeyError:
            return None

    def add(self, layer: Layer, *, at: int | None = None) -> Layer:
        """Insert a layer, on top by default."""
        if self.find(layer.key) is not None:
            raise ValueError(f"layer {layer.key!r} already in the stack")
        if at is None:
            self.layers.append(layer)
        else:
            self.layers.insert(max(0, min(at, len(self.layers))), layer)
        return layer

    def remove(self, key: str) -> Layer:
        return self.layers.pop(self.index_of(key))

    def update(self, key: str, **changes: object) -> Layer:
        """Replace fields on one layer in place."""
        i = self.index_of(key)
        self.layers[i] = self.layers[i].with_(**changes)
        return self.layers[i]

    def move(self, key: str, to: int) -> int:
        """Move a layer to an absolute position; returns where it landed."""
        i = self.index_of(key)
        layer = self.layers.pop(i)
        to = max(0, min(to, len(self.layers)))
        self.layers.insert(to, layer)
        return to

    # -- underlay / overlay roles --------------------------------------
    #
    # Position is the truth: index 0 is drawn first, so "the underlay" is
    # simply the bottom of the stack and "the overlay" the one above it. These
    # helpers exist because that is how people think and how the buttons are
    # labelled, not because the stack has a second notion of identity.

    def set_underlay(self, layer: Layer) -> Layer:
        """Replace the bottom layer, keeping everything stacked above it."""
        if self.layers:
            old = self.layers[0]
            self.layers[0] = layer
            if old.key != layer.key:
                # Two layers must never share a key; the replaced one is gone.
                self.layers = [layer] + [ly for ly in self.layers[1:] if ly.key != layer.key]
        else:
            self.layers.append(layer)
        return layer

    def set_overlay(self, layer: Layer) -> Layer:
        """Replace the primary overlay -- the layer just above the underlay.

        Additional overlays pushed with :meth:`add_overlay` are left alone, so
        swapping the stats map you are looking at does not drop the atlas you
        put on top of it.
        """
        self.layers = [ly for ly in self.layers if ly.key != layer.key]
        if len(self.layers) >= 2:
            self.layers[1] = layer
        else:
            self.layers.append(layer)
        return layer

    def add_overlay(self, layer: Layer) -> Layer:
        """Push another overlay on top of the stack."""
        return self.add(layer)

    @property
    def overlay(self) -> Layer | None:
        """The primary overlay, if there is one."""
        return self.layers[1] if len(self.layers) > 1 else None

    def find_by_source(self, source: str) -> Layer | None:
        """The layer a given mode owns, if it has produced one."""
        for ly in self.layers:
            if ly.source == source:
                return ly
        return None

    def visible_layers(self) -> list[Layer]:
        return [ly for ly in self.layers if ly.visible and ly.opacity > 0.0]

    @property
    def base(self) -> Layer | None:
        """The bottom layer -- what AFNI would call the underlay."""
        return self.layers[0] if self.layers else None
