"""Cortical surfaces in the viewer: meshes held beside the layer stack.

Surfaces are not layers. A layer is voxels on a grid with an affine; a surface
is a mesh whose vertices are already in scanner millimetres, and drawing it on
a slice is a geometric intersection rather than a resample. So the meshes live
here, on the session, and the state carries only the scalars a script needs
to rebuild them (which subject, which surfaces are drawn) -- the same split as
layers, whose voxels live in the store while the state holds the record.

Nothing here imports Qt. Outlines come out in the drawn image's fractional
(row, col), through :meth:`PlaneView.points_to_image`, so the flip and crop
arithmetic stays in the one place every other overlay already uses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fastfuncstuff.io.freesurfer import (
    Annotation,
    Hemisphere,
    load_subject,
    read_annotation,
    read_color_lut,
)
from fastfuncstuff.surface.edit import EditResult, SnapParams, SurfaceEdit
from fastfuncstuff.surface.geometry import SliceIndex, apply_affine
from fastfuncstuff.surface.mesh import MeshTopology
from fastfuncstuff.surface.sampling import VolumeSampler
from fastfuncstuff.viewer.slicing import PlaneView

#: Outline colours (RGB, 0-1). Yellow white and red pial follow freeview, which
#: is what anyone checking a recon has been looking at for years.
OUTLINE_RGB: dict[str, tuple[float, float, float]] = {
    "white": (1.0, 0.92, 0.0),
    "pial": (1.0, 0.15, 0.15),
    "smoothwm": (0.2, 0.85, 1.0),
}

#: The surfaces that are anatomically placed, and so can be drawn on a slice.
#: Inflated and sphere positions have no meaning in the scanner.
ANATOMICAL = ("white", "pial", "smoothwm")


@dataclass(frozen=True)
class Outline:
    hemi: str
    surface: str
    rgb: tuple[float, float, float]
    #: ``(N, 2, 2)`` segment endpoints as fractional (row, col) image pixels,
    #: pixel ``r`` centred at ``r + 0.5``.
    segments: np.ndarray


#: The surfaces an edit can grab, and the one each keeps outside/inside.
PARTNER = {"white": "pial", "pial": "white"}


@dataclass(frozen=True)
class Grab:
    """What a press near an outline took hold of."""

    hemi: str
    surface: str
    vertex: int
    #: Where the press landed, scanner mm; a drag is measured from here.
    at_mm: tuple[float, float, float]


@dataclass
class _Undo:
    hemi: str
    moves: list[tuple[str, np.ndarray, np.ndarray]]  # (surface, ids, old positions)


@dataclass
class EditLog:
    """Every committed edit, for the JSON written next to saved surfaces."""

    entries: list[dict] = field(default_factory=list)


class SurfaceStore:
    """The loaded hemispheres, slice indices cached per display grid, and edits."""

    def __init__(self) -> None:
        self.subject: Path | None = None
        self.hemis: dict[str, Hemisphere] = {}
        self._grid_key: bytes | None = None
        self._grid_inverse: np.ndarray | None = None
        self._index: dict[tuple[str, str], SliceIndex] = {}
        self._topo: dict[str, MeshTopology] = {}
        self._active: tuple[Grab, SurfaceEdit, np.ndarray] | None = None
        #: The last previewed edit, by the parameters that produced it, so the
        #: committing command does not recompute what is already on screen.
        self._pending: tuple[tuple, EditResult] | None = None
        self._undo: list[_Undo] = []
        self.edited: set[tuple[str, str]] = set()
        self.log = EditLog()
        #: Bumped whenever any vertex moves or surfaces are (re)loaded, per
        #: hemisphere, so a 3-D window knows when to re-upload its buffers.
        self.version: dict[str, int] = {}
        self._annots: dict[tuple[str, str], Annotation | None] = {}
        self._atlases: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
        self._lut: dict[int, tuple[str, tuple[int, int, int]]] | None = None
        self._trees: dict[str, tuple[int, object, np.ndarray]] = {}
        #: The latest profile-column flags, ``{hemi: per-vertex score}`` (NaN
        #: off cortex), published by a profile window so a surface window can
        #: paint them. Bumped ``flags_version`` tells it they changed.
        self.flags: dict[str, np.ndarray] = {}
        self.flags_version = 0

    def load(self, subject_dir: str | Path, hemis: tuple[str, ...] = ("lh", "rh")) -> None:
        loaded = load_subject(subject_dir, hemis)
        if not loaded:
            raise FileNotFoundError(f"no ?h.white surfaces under {subject_dir}/surf")
        self.subject = Path(subject_dir)
        self.hemis = loaded
        self._reset_edits()
        for h in loaded:
            self.version[h] = self.version.get(h, 0) + 1

    def clear(self) -> None:
        self.subject = None
        self.hemis = {}
        self._reset_edits()

    def publish_flags(self, flags: dict[str, np.ndarray]) -> None:
        self.flags = flags
        self.flags_version += 1

    def _reset_edits(self) -> None:
        self.flags = {}
        self._annots.clear()
        self._atlases.clear()
        self._trees.clear()
        self._index.clear()
        self._topo.clear()
        self._active = None
        self._pending = None
        self._undo.clear()
        self.edited.clear()
        self.log = EditLog()

    def topology(self, hemi: str) -> MeshTopology:
        found = self._topo.get(hemi)
        if found is None:
            h = self.hemis[hemi]
            found = self._topo[hemi] = MeshTopology.from_faces(h.faces, h.n_vertices)
        return found

    def _slice_index(self, hemi: str, surface: str, grid_affine: np.ndarray) -> SliceIndex | None:
        key = np.asarray(grid_affine, np.float64).tobytes()
        if key != self._grid_key:
            # A new display grid re-expresses every vertex; indices built in
            # the old one would cut the wrong slice.
            self._index.clear()
            self._grid_key = key
            self._grid_inverse = np.linalg.inv(grid_affine)
        found = self._index.get((hemi, surface))
        if found is None:
            h = self.hemis[hemi]
            if surface not in h.states:
                return None
            ijk = apply_affine(np.linalg.inv(grid_affine), h.states[surface])
            found = self._index[(hemi, surface)] = SliceIndex(ijk, h.faces)
        return found

    def _move(
        self, hemi: str, surface: str, ids: np.ndarray, positions: np.ndarray, faces: np.ndarray
    ) -> None:
        """Put vertices somewhere, keeping any built slice index in step."""
        if ids.size == 0:
            return
        self.hemis[hemi].states[surface][ids] = positions
        self.version[hemi] = self.version.get(hemi, 0) + 1
        index = self._index.get((hemi, surface))
        if index is not None and self._grid_inverse is not None:
            index.move(ids, apply_affine(self._grid_inverse, positions), faces)

    def outlines(
        self,
        grid_affine: np.ndarray,
        view: PlaneView,
        position: int,
        shown: tuple[str, ...],
        only: set[tuple[str, str]] | None = None,
    ) -> list[Outline]:
        """Where each shown surface crosses the slice at ``position``.

        ``only`` limits it to some (hemi, surface) pairs -- what a drag moved.
        """
        out: list[Outline] = []
        axis = view.layout.fixed
        for hemi in self.hemis:
            for surface in shown:
                if surface not in ANATOMICAL:
                    continue
                if only is not None and (hemi, surface) not in only:
                    continue
                index = self._slice_index(hemi, surface, grid_affine)
                if index is None:
                    continue
                seg = index.segments(axis, float(position))
                if not len(seg):
                    continue
                rgb = OUTLINE_RGB.get(surface, (1.0, 1.0, 1.0))
                out.append(Outline(hemi, surface, rgb, view.points_to_image(seg)))
        return out

    # -- atlases ---------------------------------------------------------------
    def annotation(self, hemi: str, name: str) -> Annotation | None:
        """``label/<hemi>.<name>.annot``, cached; ``None`` if absent or unreadable."""
        key = (hemi, name)
        if key not in self._annots:
            found = None
            if self.subject is not None and hemi in self.hemis:
                path = self.subject / "label" / f"{hemi}.{name}.annot"
                if path.exists():
                    try:
                        found = read_annotation(path, self.hemis[hemi].n_vertices)
                    except (OSError, ValueError):
                        found = None
            self._annots[key] = found
        return self._annots[key]

    def volume_atlas(self, name: str) -> tuple[np.ndarray, np.ndarray] | None:
        """``mri/<name>.mgz`` as (integer labels, affine), cached."""
        if name not in self._atlases:
            found = None
            if self.subject is not None:
                path = self.subject / "mri" / f"{name}.mgz"
                if path.exists():
                    import nibabel as nib

                    img = nib.load(str(path))
                    found = (np.asarray(img.dataobj).astype(np.int32), np.asarray(img.affine))
            self._atlases[name] = found
        return self._atlases[name]

    def color_lut(self) -> dict[int, tuple[str, tuple[int, int, int]]]:
        if self._lut is None:
            self._lut = read_color_lut()
        return self._lut

    def nearest_vertex(
        self, mm: tuple[float, float, float], max_mm: float = 3.0
    ) -> tuple[str, int, float] | None:
        """The cortex vertex whose mid-thickness point is nearest ``mm``, if within ``max_mm``.

        Mid-thickness, so a point anywhere in the ribbon finds the column it
        belongs to; cortex only, because the medial wall names nothing.
        """
        from scipy.spatial import cKDTree

        best: tuple[str, int, float] | None = None
        for hemi, h in self.hemis.items():
            version = self.version.get(hemi, 0)
            cached = self._trees.get(hemi)
            if cached is None or cached[0] != version:
                ids = np.flatnonzero(h.cortex) if h.cortex is not None else np.arange(h.n_vertices)
                mid = 0.5 * (h.states["white"][ids] + h.states["pial"][ids])
                cached = (version, cKDTree(mid), ids)
                self._trees[hemi] = cached
            _, tree, ids = cached
            d, k = tree.query(np.asarray(mm, np.float64))  # type: ignore[attr-defined]
            if d <= max_mm and (best is None or d < best[2]):
                best = (hemi, int(ids[k]), float(d))
        return best

    def region_lines(
        self, mm: tuple[float, float, float] | None, annot: str, atlas: str
    ) -> list[str]:
        """What the selected parcellation and volume atlas call the point ``mm``.

        The surface name comes from the nearest cortex vertex (within 3 mm),
        so it answers in the ribbon and on the surface; the volume atlas from
        the voxel itself, so it also names white matter and subcortex.
        """
        if mm is None or not self.hemis:
            return []
        lines: list[str] = []
        if annot:
            near = self.nearest_vertex(mm)
            if near is not None:
                ann = self.annotation(near[0], annot)
                name = ann.name_at(near[1]) if ann is not None else None
                if name:
                    lines.append(f"{near[0]} {name}  ({annot})")
        if atlas:
            vol = self.volume_atlas(atlas)
            if vol is not None:
                data, aff = vol
                ijk = np.linalg.inv(aff) @ np.array([*mm, 1.0])
                i, j, k = np.floor(ijk[:3] + 0.5).astype(int)
                if all(0 <= a < n for a, n in zip((i, j, k), data.shape, strict=True)):
                    label = int(data[i, j, k])
                    if label:
                        name = self.color_lut().get(label, (str(label), (0, 0, 0)))[0]
                        lines.append(f"{name}  ({atlas})")
        return lines

    # -- editing -----------------------------------------------------------
    def grab(
        self,
        grid_affine: np.ndarray,
        view: PlaneView,
        position: int,
        shown: tuple[str, ...],
        row: float,
        col: float,
        tolerance: float,
    ) -> Grab | None:
        """The outline nearest a press, if one is within ``tolerance`` image pixels.

        The vertex taken is the nearest corner of the cut face -- in 3-D, so
        it is one the slice actually passes beside, not one a sulcus away.
        """
        click = view.image_to_points(row, col, position)
        axis = view.layout.fixed
        best: tuple[float, str, str, int] | None = None
        for hemi in self.hemis:
            for surface in shown:
                if surface not in PARTNER:
                    continue
                index = self._slice_index(hemi, surface, grid_affine)
                if index is None:
                    continue
                seg, faces = index.segments_with_faces(axis, float(position))
                if not len(seg):
                    continue
                img = view.points_to_image(seg)  # distances in image pixels
                d = _point_segment_distance(np.array([row, col]), img[:, 0], img[:, 1])
                k = int(np.argmin(d))
                if d[k] <= tolerance and (best is None or d[k] < best[0]):
                    corners = index.faces[faces[k]]
                    near = corners[
                        np.argmin(np.linalg.norm(index.vertices[corners] - click, axis=1))
                    ]
                    best = (float(d[k]), hemi, surface, int(near))
        if best is None:
            return None
        mm = apply_affine(grid_affine, click[None])[0]
        return Grab(best[1], best[2], best[3], (float(mm[0]), float(mm[1]), float(mm[2])))

    def begin(self, grab: Grab, sampler: VolumeSampler, params: SnapParams) -> None:
        """Start a drag: the brush is fixed here, every preview starts from here."""
        h = self.hemis[grab.hemi]
        topo = self.topology(grab.hemi)
        partner = h.states.get(PARTNER[grab.surface])
        edit = SurfaceEdit(
            h.states[grab.surface],
            topo,
            grab.vertex,
            sampler,
            params,
            role=grab.surface,
            partner=partner,
        )
        self._active = (grab, edit, topo.faces_of(edit.ids))
        self._pending = None

    @property
    def editing(self) -> Grab | None:
        return None if self._active is None else self._active[0]

    @property
    def editing_keys(self) -> set[tuple[str, str]] | None:
        """The (hemi, surface) pairs the active drag can move, or ``None``."""
        if self._active is None:
            return None
        hemi = self._active[0].hemi
        return {(hemi, "white"), (hemi, "pial")}

    def preview(self, drag_mm: np.ndarray) -> EditResult | None:
        """Show the active drag at ``drag_mm`` from the press, without committing."""
        if self._active is None:
            return None
        grab, edit, faces = self._active
        res = edit.update(np.asarray(drag_mm, np.float64))
        self._show(grab.hemi, grab.surface, edit, faces, res)
        self._pending = (self._key(grab, drag_mm, edit.params), res)
        return res

    def _show(
        self, hemi: str, surface: str, edit: SurfaceEdit, faces: np.ndarray, res: EditResult
    ) -> None:
        self._move(hemi, surface, res.ids, res.positions, faces)
        if edit.partner_start is not None:
            partner = PARTNER[surface]
            # Back to where the press found it, then pushed wherever this
            # update pushes it -- the pushed set changes as the drag does.
            self._move(hemi, partner, edit.ids, edit.partner_start, faces)
            self._move(hemi, partner, res.partner_ids, res.partner_positions, faces)

    def cancel(self) -> None:
        """Abandon the active drag and put everything back."""
        if self._active is None:
            return
        grab, edit, faces = self._active
        self._move(grab.hemi, grab.surface, edit.ids, edit.start, faces)
        if edit.partner_start is not None:
            self._move(grab.hemi, PARTNER[grab.surface], edit.ids, edit.partner_start, faces)
        self._active = None
        self._pending = None

    @staticmethod
    def _key(grab: Grab, drag_mm, params: SnapParams) -> tuple:
        drag = tuple(round(float(x), 6) for x in drag_mm)
        return (grab.hemi, grab.surface, grab.vertex, drag, params)

    def apply(
        self,
        grab: Grab,
        drag_mm: tuple[float, float, float],
        sampler: VolumeSampler | None,
        params: SnapParams,
    ) -> EditResult:
        """Commit an edit -- the preview already on screen, or computed afresh.

        Afresh is the replay path: the same press, drag and brush on the same
        image reproduce the same displacement, so a recorded EDIT_SURFACE
        rebuilds the surface rather than storing it.
        """
        key = self._key(grab, drag_mm, params)
        active = self._active
        if active is not None and self._pending is not None and self._pending[0] == key:
            _, edit, faces = active
            res = self._pending[1]
        else:
            self.cancel()
            if sampler is None:
                raise ValueError("an edit needs an image to snap to")
            self.begin(grab, sampler, params)
            assert self._active is not None
            _, edit, faces = self._active
            res = edit.update(np.asarray(drag_mm, np.float64))
            self._show(grab.hemi, grab.surface, edit, faces, res)
        moves = [(grab.surface, edit.ids.copy(), edit.start.astype(np.float32))]
        if edit.partner_start is not None:
            moves.append(
                (PARTNER[grab.surface], edit.ids.copy(), edit.partner_start.astype(np.float32))
            )
        self._undo.append(_Undo(grab.hemi, moves))
        self.edited.add((grab.hemi, grab.surface))
        if res.partner_ids.size:
            self.edited.add((grab.hemi, PARTNER[grab.surface]))
        self.log.entries.append(
            {
                "hemi": grab.hemi,
                "surface": grab.surface,
                "vertex": grab.vertex,
                "drag_mm": [float(x) for x in drag_mm],
                "params": params.__dict__,
                "moved": int(np.count_nonzero(res.displacement)),
                "max_displacement_mm": float(np.abs(res.displacement).max(initial=0.0)),
                "pushed_partner": int(res.partner_ids.size),
            }
        )
        self._active = None
        self._pending = None
        return res

    def undo(self) -> bool:
        """Put the last committed edit back. False if there is none."""
        if not self._undo:
            return False
        entry = self._undo.pop()
        topo = self.topology(entry.hemi)
        for surface, ids, old in entry.moves:
            self._move(entry.hemi, surface, ids, old, topo.faces_of(ids))
        self.log.entries.append({"undo": True})
        return True

    def _installed_states(self) -> list[tuple[str, str]]:
        """(hemi, surface) pairs an install writes: the edited, plus smoothwm with white."""
        out = sorted(self.edited)
        for hemi, surface in list(out):
            if surface == "white" and "smoothwm" in self.hemis[hemi].paths:
                out.append((hemi, "smoothwm"))
        return out

    def install_plan(self, stamp: str | None = None) -> InstallPlan:
        """The originals an install would replace, and where each is backed up."""
        import time

        stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
        files = []
        for hemi, surface in self._installed_states():
            original = self.hemis[hemi].paths[surface]
            backup = original.with_name(f"{original.name}.pre-ffsedit-{stamp}")
            n = 1
            while backup.exists():
                n += 1
                backup = original.with_name(f"{original.name}.pre-ffsedit-{stamp}-{n}")
            files.append((original, backup))
        return InstallPlan(files, stamp)

    def install(self, plan: InstallPlan | None = None) -> InstallPlan:
        """Replace the original surface files with the edits, keeping backups.

        Deliberately **not** a recorded command: a replayed script must never
        overwrite a subject's surfaces. Each original is copied to its backup
        first; the edited surface is written beside it and renamed over the
        original, so an interruption leaves either the old file or the new
        one, never half of one. Afterwards the installed surfaces are the new
        baseline -- nothing is "edited" any more and undo starts afresh.
        """
        import shutil

        if not self.edited:
            raise ValueError("no edited surfaces to install")
        plan = plan or self.install_plan()
        # Work out every position before touching any file: smoothwm's
        # displacement is white's current position against white's *file*.
        positions: dict[tuple[str, str], np.ndarray] = {}
        for hemi, surface in self._installed_states():
            h = self.hemis[hemi]
            if surface == "smoothwm":
                positions[(hemi, surface)] = h.original("smoothwm") + (
                    h.states["white"] - h.original("white")
                )
            else:
                positions[(hemi, surface)] = h.states[surface].copy()
        for (hemi, surface), (original, backup) in zip(
            self._installed_states(), plan.files, strict=True
        ):
            shutil.copy2(original, backup)
            tmp = original.with_name(f".{original.name}.ffsedit-tmp")
            self.hemis[hemi].save_positions(surface, tmp, positions[(hemi, surface)])
            tmp.replace(original)
            if surface in self.hemis[hemi].states:
                self.hemis[hemi].states[surface] = positions[(hemi, surface)].astype(np.float32)
        if plan.files:
            log = plan.files[0][0].with_name(f"surface_edits.installed-{plan.stamp}.json")
            installed = [[str(o), str(b)] for o, b in plan.files]
            log.write_text(
                json.dumps({"installed": installed, "edits": self.log.entries}, indent=1)
            )
        self.edited.clear()
        self._undo.clear()
        self.log = EditLog()
        return plan

    def save(self, suffix: str = "ffsedit") -> list[Path]:
        """Write every edited surface as ``?h.<surface>.<suffix>`` beside its original.

        An edited white also writes ``?h.smoothwm.<suffix>`` with white's
        displacement. Copies only -- :meth:`Hemisphere.save_state` refuses the original's own
        path -- plus a JSON log of the edits that produced them.
        """
        if not suffix or "/" in suffix:
            raise ValueError(f"bad suffix {suffix!r}")
        written: list[Path] = []
        for hemi, surface in sorted(self.edited):
            h = self.hemis[hemi]
            out = h.paths[surface].with_name(f"{hemi}.{surface}.{suffix}")
            h.save_state(surface, out)
            written.append(out)
            if surface == "white" and "smoothwm" in h.paths:
                # smoothwm is a smoothed white; give it white's displacement so
                # a rerun from these surfaces starts from a consistent pair.
                # Untouched vertices have zero displacement and stay
                # bit-identical, as in every other saved copy.
                moved = h.states["white"] - h.original("white")
                smooth = h.original("smoothwm") + moved
                out = h.paths["smoothwm"].with_name(f"{hemi}.smoothwm.{suffix}")
                h.save_positions("smoothwm", out, smooth)
                written.append(out)
        if written:
            log = written[0].with_name(f"surface_edits.{suffix}.json")
            log.write_text(json.dumps(self.log.entries, indent=1))
            written.append(log)
        return written


@dataclass(frozen=True)
class InstallPlan:
    """What :meth:`SurfaceStore.install` would do: (original, backup) per file."""

    files: list[tuple[Path, Path]]
    stamp: str


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from ``p`` (2,) to each segment ``a[i]-b[i]`` (N, 2)."""
    ab = b - a
    t = np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12)
    closest = a + np.clip(t, 0.0, 1.0)[:, None] * ab
    return np.linalg.norm(closest - p, axis=1)


__all__ = ["ANATOMICAL", "OUTLINE_RGB", "PARTNER", "Grab", "InstallPlan", "Outline", "SurfaceStore"]
