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

    visible: bool = True
    opacity: float = 1.0
    colormap: str = "gray"
    n_panes: int = 0  # 0 = continuous; >0 = AFNI-style discrete colour bar
    sign_mode: SignMode = SignMode.BOTH
    volume_index: int = 0

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

    def visible_layers(self) -> list[Layer]:
        return [ly for ly in self.layers if ly.visible and ly.opacity > 0.0]

    @property
    def base(self) -> Layer | None:
        """The bottom layer -- what AFNI would call the underlay."""
        return self.layers[0] if self.layers else None
