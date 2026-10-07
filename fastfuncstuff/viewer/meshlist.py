"""The mesh list: every white/pial mesh loaded, in order, the top-most of each in use.

A subject loads one white and one pial per hemisphere, and those are what
editing, depth sampling and depth maps act on. Editing creates a second
version of a surface, and comparing against an alternative recon needs a
third. So the meshes are kept as an ordered list, and the rule is short: per
hemisphere and type, the **top-most** row is the one in use. Its vertices stay
where they always have (``Hemisphere.states``), so nothing that reads them
changes. Every other row holds its own copy and is only drawn, as a
comparison outline.

Nothing here imports Qt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fastfuncstuff.io.freesurfer import infer_label

#: The two types a row can be. smoothwm is a white matter surface for this
#: purpose: what matters is which side of the cortex it bounds.
KINDS = ("white", "pial")

#: Colours handed to rows that are not in use, in turn. Away from the yellow
#: and red the in-use pair keeps, so a comparison never looks like the real one.
COMPARE_RGB: tuple[tuple[float, float, float], ...] = (
    (0.25, 0.95, 0.35),
    (1.0, 0.35, 1.0),
    (1.0, 0.6, 0.1),
    (0.3, 0.75, 1.0),
    (0.75, 0.55, 1.0),
    (0.6, 1.0, 0.85),
)


@dataclass
class MeshEntry:
    """One row of the mesh list."""

    #: Stable id (``m1``, ``m2``...) handed out in order, so a replayed script
    #: names the same rows.
    key: str
    hemi: str
    kind: str
    name: str
    #: The file it came from; ``None`` for an in-memory snapshot.
    path: Path | None
    #: Scanner-RAS vertices, or ``None`` while this row is in use -- the
    #: hemisphere's own state array is then the data, and is the only copy.
    positions: np.ndarray | None
    faces: np.ndarray
    rgb: tuple[float, float, float]
    shown: bool = True
    #: Moved since it was loaded (or since it last matched its file).
    edited: bool = False

    @property
    def label(self) -> str:
        return f"{self.name}{' *' if self.edited else ''}"


__all__ = ["COMPARE_RGB", "KINDS", "MeshEntry", "infer_label"]
