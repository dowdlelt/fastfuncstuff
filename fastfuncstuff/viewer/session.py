"""A viewer session: state, residency, and the command bus wired together.

This is what a UI, a CLI or a test drives. Nothing above this layer touches
:class:`ViewerState` directly -- everything goes through
:meth:`ViewerSession.do`, which is what keeps the recording honest.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.io.dsetinfo import DatasetInfo
from fastfuncstuff.viewer import catalog as catalog_mod
from fastfuncstuff.viewer.catalog import CatalogEntry
from fastfuncstuff.viewer.commands import Aspect, Command, CommandBus
from fastfuncstuff.viewer.design import Design, Provenance, RegressorTrace, run_of
from fastfuncstuff.viewer.layers import Layer, SignMode
from fastfuncstuff.viewer.modes import Mode, registry
from fastfuncstuff.viewer.modes.base import ComputedOverlay, Trace
from fastfuncstuff.viewer.paintlayer import Painter
from fastfuncstuff.viewer.residency import Resident, VolumeStore, _c_order
from fastfuncstuff.viewer.rois import RoiSet, looks_like_labels, rois_from_frames, rois_from_labels
from fastfuncstuff.viewer.state import Plane, ViewerState
from fastfuncstuff.viewer.surfaces import SurfaceStore
from fastfuncstuff.viewer.viewports import ViewKind
from fastfuncstuff.viewer.vocab import (
    AddLayer,
    CloseView,
    Load,
    OpenView,
    ReloadLayer,
    RemoveLayer,
    SetIJK,
    SetIndex,
    SetVolume,
    SetXYZ,
    install,
)

#: Percentiles used to auto-range a layer. AFNI's autorange takes the maximum,
#: which one bright voxel is enough to ruin; percentiles are what make a map
#: readable without anyone reaching for a slider.
AUTORANGE_PERCENTILES = (2.0, 98.0)

#: Where a statistic's threshold starts, as a p (two-sided for t/z, upper tail for F).
OVERLAY_START_P = 1e-3
#: A signed map is mirrored about zero only when each sign holds at least this
#: share of its non-zero voxels -- real mass on both sides, not a few ringing
#: voxels, before half the bar goes to one of them ...
MIRROR_MIN_SHARE = 0.2
#: ... and the two tails (the autorange percentile within each sign) are
#: within this ratio of each other.
MIRROR_BALANCE = 0.5
#: Values read when judging a map's shape. Percentiles of a strided sample this
#: size are indistinguishable on screen from the full volume's, at a fraction
#: of the sort.
LOOK_SAMPLE = 200_000

#: How long the crosshair must sit still before the place it leaves becomes the
#: `j` target, seconds. A drag or a run of clicks is one jump, from where it
#: started; without this, jumping back lands a few pixels along the drag.
JUMP_REST_S = 0.5


def derive_range(
    values: np.ndarray, percentiles: tuple[float, float] = AUTORANGE_PERCENTILES
) -> tuple[float, float]:
    """Display range from a volume, ignoring non-finite voxels.

    Falls back to the finite min/max when the percentile span collapses, which
    happens on masks and other near-constant volumes.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo, hi = (float(v) for v in np.percentile(finite, percentiles))
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        hi = lo + 1.0
    return (lo, hi)


def is_coefficient(layer: Layer, index: int) -> bool:
    """Whether a sub-brick is a regression coefficient (``_Coef``, betas): by
    its label, or by the filename when the dataset carries no labels.

    A coefficient is signed by construction, so it is drawn diverging and
    mirrored even when this particular map happens to lean one way.
    """
    if layer.labels:
        name = layer.labels[index] if 0 <= index < len(layer.labels) else ""
    else:
        name = layer.name
    lowered = name.lower()
    return "coef" in lowered or "beta" in lowered


def _look_sample(values: np.ndarray) -> np.ndarray:
    """Finite, non-zero values to read a colour scale from, strided down to
    :data:`LOOK_SAMPLE`. Zero is excluded because a bucket is zero outside its
    mask, and counting that puts every percentile near nothing."""
    flat = values.reshape(-1)
    flat = flat[:: max(1, flat.size // LOOK_SAMPLE)]
    return flat[np.isfinite(flat) & (flat != 0)]


def infer_time_linked(info: DatasetInfo) -> bool:
    """Whether the global time index should drive this dataset.

    A time series and a stats dataset are both 4-D on disk, and NIfTI cannot
    reliably tell them apart: ``pixdim[4]`` defaults to 1.0, so almost every
    file claims a TR. Sub-brick labels are the usable signal — 3dDeconvolve and
    ffs write them on stats output, raw time series carry none.

    So: 4-D defaults to time-linked because that is the common case, and labels
    turn it off. Both are guesses about a file that does not say, which is why
    SET_TIME_LINKED exists to correct it.
    """
    if info.n_volumes <= 1:
        return False
    return not info.labels


def layer_from_info(info: DatasetInfo, key: str, path: Path) -> Layer:
    """Build a display layer from a header-only read."""
    nx, ny, nz, nv = info.shape
    return Layer(
        key=key,
        name=path.name,
        path=str(path),
        shape=(int(nx), int(ny), int(nz)),
        n_volumes=max(int(nv), 1),
        affine=np.asarray(info.affine, dtype=float),
        labels=tuple(info.labels),
        stataux=dict(info.stataux),
        time_linked=infer_time_linked(info),
    )


def _canonical(path) -> Path:
    try:
        return Path(path).resolve()
    except OSError:
        return Path(path).absolute()


def _fmt(value: float | None) -> str:
    return "--" if value is None else f"{value:.4g}"


def _brick_name(layer: Layer, index: int) -> str:
    """``A#0_Coef `` for a labelled sub-brick; nothing for a plain volume."""
    if 0 <= index < len(layer.labels) and layer.labels[index]:
        return f"{layer.labels[index]} "
    return f"#{index} " if layer.n_volumes > 1 and not layer.time_linked else ""


def _overlay_array(overlay: ComputedOverlay) -> tuple[np.ndarray, tuple[int, int, int]]:
    """A mode's output as the ``(nx, ny, nz, nv)`` block the store adopts.

    Two volumes when the mode wants to colour one thing and cut on another,
    stacked shown-first so that ``volume_index`` 0 and ``threshold_index`` 1
    are the same convention a stats bucket off disk already uses.
    """
    values = np.asarray(overlay.values, dtype=np.float32)
    nx, ny, nz = (int(v) for v in values.shape[:3])
    shape = (nx, ny, nz)
    if overlay.threshold_values is None:
        return values[..., None], shape
    cut = np.asarray(overlay.threshold_values, dtype=np.float32)
    if cut.shape[:3] != shape:
        raise ValueError(f"threshold volume {cut.shape[:3]} does not match the overlay {shape}")
    return np.stack([values, cut], axis=3), shape


def _overlay_cut(overlay: ComputedOverlay) -> dict[str, object]:
    """Layer fields that say where this overlay's threshold reads from."""
    one_volume = overlay.threshold_values is None
    stataux: dict[int, tuple[int, tuple[float, ...]]] = {}
    if overlay.threshold_stat is not None:
        from fastfuncstuff.io.afni import stat_type_to_stataux

        name, params = overlay.threshold_stat
        flat = () if params is None else (params if isinstance(params, tuple) else (params,))
        try:
            stataux[0 if one_volume else 1] = stat_type_to_stataux(name, flat)
        except ValueError:
            stataux = {}
    if one_volume:
        # Back to cutting on what is shown. Said explicitly because a layer
        # that was two volumes a moment ago still points at sub-brick 1.
        return {"threshold_index": None, "threshold_follow": "same", "stataux": stataux}
    # ``fixed``, not ``next``: the threshold brick is where the mode put it,
    # and the shown brick never moves off 0, so following would be a rule
    # describing something that cannot happen.
    return {"threshold_index": 1, "threshold_follow": "fixed", "stataux": stataux}


def _short(name: str, limit: int = 32) -> str:
    for ext in (".nii.gz", ".nii.zst", ".nii", ".mgz", ".mgh", ".HEAD", ".func.gii", ".shape.gii"):
        name = name.removesuffix(ext)
    if len(name) <= limit:
        return name
    # Both ends: results sharing a subject prefix differ at the tail, and ones
    # sharing a suffix (``_stats``) differ at the head.
    head = (limit - 1) // 2
    return name[:head] + "…" + name[-(limit - 1 - head) :]


class ValueLine(str):
    """A readout line whose first ``name_end`` characters name what the rest measures.

    Still a plain ``str`` to everything that compares or prints it; a pane too
    narrow for the whole line shortens the name and keeps the numbers.
    """

    name_end: int

    def __new__(cls, name: str, values: str) -> ValueLine:
        line = super().__new__(cls, name + values)
        line.name_end = len(name)
        return line


@dataclass(frozen=True)
class LayerReading:
    """A layer's value(s) at the crosshair, said two ways.

    ``short`` is numbers only (``1.23 / 4.5``: shown, then threshold), for a
    row of the layer list where the layer's name is already beside it;
    ``long`` names the sub-bricks, for a line that has to stand on its own.
    """

    short: str
    long: str


def resample_mask_nearest(
    mask: np.ndarray,
    affine: np.ndarray,
    shape: tuple[int, int, int],
    target_affine: np.ndarray,
) -> np.ndarray:
    """A boolean volume on another grid, nearest neighbour, False outside it.

    One z-slab at a time so a fine target grid never materialises every
    voxel's source index at once.
    """
    mask = np.asarray(mask, dtype=bool)
    to_source = np.linalg.inv(np.asarray(affine, dtype=float)) @ np.asarray(
        target_affine, dtype=float
    )
    rot, shift = to_source[:3, :3], to_source[:3, 3]
    nx, ny, nz = (int(n) for n in shape)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    plane = np.stack([ii.ravel(), jj.ravel(), np.zeros(ii.size)]).astype(float)
    out = np.zeros((nx, ny, nz), dtype=bool)
    bounds = np.asarray(mask.shape)[:, None]
    for k in range(nz):
        plane[2] = k
        src = np.rint(rot @ plane + shift[:, None]).astype(np.int64)
        inside = np.all((src >= 0) & (src < bounds), axis=0)
        slab = np.zeros(plane.shape[1], dtype=bool)
        slab[inside] = mask[src[0, inside], src[1, inside], src[2, inside]]
        out[:, :, k] = slab.reshape(nx, ny)
    return out


def _voxel_size(affine: np.ndarray) -> float:
    """One number for how big this grid's voxels are: the cube root of one
    voxel's volume.

    A single scalar rather than three, because the question it answers --
    is this layer coarser than the grid -- has one answer. ``abs(det)`` rather
    than the diagonal so an oblique or anisotropic grid is measured by the
    volume it actually covers.
    """
    return float(abs(np.linalg.det(np.asarray(affine, dtype=float)[:3, :3])) ** (1.0 / 3.0))


class ViewerSession:
    """Owns the state, the residency store and the bus."""

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        record: bool = True,
        store: VolumeStore | None = None,
    ) -> None:
        self.state = ViewerState()
        #: Seconds, monotonic; replaceable so a test can step time.
        self.clock: Callable[[], float] = time.monotonic
        self._last_move = float("-inf")
        self.store = store or VolumeStore(device=device)
        #: Painted ROI layers: label tables, strokes, undo (viewer.paintlayer).
        self.painter = Painter(self)
        #: What each colour layer draws from, by layer key (viewer.chromalayer).
        #: Kept copies share their source with the live layer they froze.
        self.colour_sources: dict = {}
        self.bus = install(
            CommandBus(self.state, record=record), open_layer=self._open, session=self
        )
        #: FreeSurfer meshes. Beside the layer stack, not in it: see
        #: :mod:`viewer.surfaces` for why a surface is not a layer.
        self.surfaces = SurfaceStore(device=self.display_device)
        #: A picker's preview layer: (key, path, the selection before it).
        self._preview: tuple[str, str, str | None] | None = None
        #: Set while a preview opens its layer, so no full inflate starts.
        self._previewing = False
        self.catalog: list[CatalogEntry] = []
        self.catalog_dir: Path | None = None
        #: The files last listed by name. The catalog is this list while
        #: ``catalog_dir`` is None; kept after a READ so the list can be gone back to.
        self.file_list: tuple[Path, ...] = ()
        #: Catalogs read earlier, by (directory, recursive) or by the file list.
        self._catalogs: dict[tuple, list[CatalogEntry]] = {}
        self.catalog_recursive = False
        #: Catalog files that appeared or changed since the directory was
        #: read, as ``{path: "new" | "updated"}``. Cleared per file when it is
        #: picked, and wholesale by a fresh READ.
        self.catalog_fresh: dict[Path, str] = {}
        #: Every directory read or loaded from, in first-use order, each with
        #: whether it was read recursively. A directory's number is its place
        #: here plus one -- what the layer list tags a file with once files
        #: come from more than one place -- so it only ever grows.
        self.directories: dict[Path, bool] = {}
        self._directory_numbers: dict[str, int | None] = {}
        self.mode: Mode = registry.get("plain")()
        #: Every mode this session has been in, by name. Switching back resumes
        #: the same instance -- its folder, its component, its unsaved labels --
        #: because a mode left to go and look at something else is a mode you
        #: are coming back to.
        self._modes: dict[str, Mode] = {self.mode.name: self.mode}
        self.mode.attach(self)
        self._volume_cache: dict[tuple, torch.Tensor] = {}
        #: Display transforms of time series, per (layer, hemi, view, generation).
        self._time_views: dict[tuple, object] = {}
        #: Built on demand from a layer's voxels and kept until the flag or the
        #: layer changes. Describing an atlas is one pass over the volume, and
        #: the crosshair readout asks which region it is on every move.
        self._roi_sets: dict[str, RoiSet] = {}
        self._roi_palettes: dict[tuple[str, str], torch.Tensor] = {}
        self._clustsim: dict[str, dict] = {}
        #: Surface layers' vertex values and ribbon maps, by layer key (see
        #: viewer.surfacelayers). The layer itself is an ordinary Layer.
        self.surface_layers: dict = {}
        #: Display-only masks, ``{owner: {layer key: keep}}`` on each layer's own
        #: grid. Keyed by owner so two windows masking one layer compose (AND)
        #: and closing one takes away only its own.
        self._display_masks: dict[str, dict[str, np.ndarray]] = {}
        self._display_mask_tensors: dict[str, torch.Tensor] = {}
        self._source_masks: dict[tuple[str, int], np.ndarray] = {}
        self._threshold_scales: dict[tuple[str, int], float] = {}
        #: Designs offered in graph windows, by resolved path.
        self.designs: dict[str, Design] = {}
        self._design_errors: dict[str, str] = {}
        self._design_paths: dict[str, str] = {}
        self._provenance_cache: dict[str, Provenance | None] = {}
        self._run_cache: dict[tuple[Provenance, str], int | None] = {}
        self._mode_dirty: Aspect = Aspect.NOTHING
        #: The controller letter this session is, set by a UI that has several.
        #: It is the stem every mode output is named with -- ``A_ICORR`` -- so a
        #: layer compared across tabs says which tab made it.
        self.label = "A"
        #: Set once by a UI that runs mode preparation on a worker. Applied to
        #: every mode as it is attached -- setting it on the mode afterwards
        #: would be too late, since set_mode refreshes on the way in and that
        #: first refresh is exactly the one that would freeze the window.
        self.defer_mode_preparation = False
        # A time-index or sub-brick change invalidates the cached device volume;
        # doing it here rather than in each handler means a command added later
        # cannot forget to.
        self.bus.subscribe(self._on_command)

    # -- loading -------------------------------------------------------
    def _open(self, path: str, key: str) -> Layer:
        """Header peek plus first volume; the full inflate starts in background.

        Returning after only the preview is deliberate. Volume 0 costs about
        2.5 ms while a full dataset takes seconds, so the layer becomes visible
        immediately and becomes scrubbable when the worker finishes.
        """
        from fastfuncstuff.viewer.surfacelayers import is_surface_path

        if self.directory_number(path) is None:
            self._note_directory(Path(path).parent)
        if is_surface_path(path):
            return self._open_surface(path, key)
        res = self.store.open(path, key=key)
        layer = layer_from_info(res.info, key, res.path)
        preview = self.store.preview(key)
        lo, hi = derive_range(preview)
        layer = layer.with_(range_lo=lo, range_hi=hi)
        # A 3-D volume of small non-negative integers is an atlas or a
        # segmentation far more often than it is a picture, and guessing here
        # is what lets one land already coloured and already named. A 4-D stack
        # of masks cannot be told from a stats bucket, so that one waits to be
        # asked.
        if looks_like_labels(preview):
            layer = layer.with_(integer_labels=True, roi=layer.n_volumes == 1)
        if layer.n_volumes > 1 and not self._previewing:
            self.store.load_async(key, on_done=self._on_loaded)
        return layer

    def _open_surface(self, path: str, key: str) -> Layer:
        """A per-vertex result as a layer: painted into the ribbon on the display grid.

        Needs the meshes it was made on (matched by fingerprint); the other
        hemisphere's file joins the same layer when it sits beside it. The grid is
        the display grid when there is one, else a 1 mm box around the meshes.
        """
        from fastfuncstuff.viewer.surfacelayers import SurfaceLayerData, open_surface_parts

        parts = open_surface_parts(path, self.surfaces.place_data)
        if self.state.grid is not None:
            shape, affine = self.state.grid.shape, np.asarray(self.state.grid.affine)
        else:
            shape, affine = self.surfaces.mesh_grid(list(parts))
        maps = self.surfaces.ribbon_maps(list(parts), shape, affine)
        sld = SurfaceLayerData(parts, maps)
        self.surface_layers[key] = sld
        first = next(iter(parts.values()))
        name = Path(path).name + (" +" + "+".join(list(parts)[1:]) if len(parts) > 1 else "")
        self.store.adopt_lazy(key, shape, sld.n_volumes, sld.paint, name=name)
        stats = dict(first.stat)
        values0 = np.concatenate([d.values[:, 0] for d in parts.values()])
        lo, hi = derive_range(values0[values0 != 0] if np.any(values0 != 0) else values0)
        return Layer(
            key=key,
            name=name,
            path=str(path),
            shape=tuple(int(s) for s in shape),
            n_volumes=sld.n_volumes,
            affine=np.asarray(affine, float),
            labels=() if first.is_series else tuple(first.labels),
            stataux=stats,
            time_linked=first.is_series,
            source="surface:" + ",".join(parts),
            range_lo=lo,
            range_hi=hi,
        )

    def layer_sampler(self, key: str, index: int, mode: str = "linear"):
        """A point sampler of one sub-brick of a layer (scanner mm in, values out).

        :meth:`surface_sampler` follows a layer's *displayed* sub-brick; an overlay
        also needs the one its threshold reads. Cached per (layer, sub-brick, grid),
        a few at a time.
        """
        from fastfuncstuff.surface.sampling import VolumeSampler

        layer = self.state.layers.get(key)
        # The store record's generation too: a mode re-adopts new values under
        # the same key, and a sampler of the old ones would draw them forever.
        generation = self.store.get(key).generation
        cache_key = (key, int(index), generation, np.asarray(layer.affine).tobytes())
        cache = getattr(self, "_layer_samplers", None)
        if cache is None:
            cache = self._layer_samplers = {}
        hit = cache.get(cache_key)
        if hit is None:
            hit = VolumeSampler(self.volume(key, int(index)), layer.affine)
            if len(cache) > 8:
                cache.pop(next(iter(cache)))
            cache[cache_key] = hit
        return hit.with_mode(mode)

    def surface_display_token(self, hemi: str) -> tuple:
        """What :meth:`surface_vertex_rgba` depends on, hashable: a 3-D window
        recolours only when it changes."""
        layer = self._surface_choice(hemi)
        if layer is None:
            return ()
        sld = self.surface_layers[layer.key]
        masked = any(layer.key in m for m in self._display_masks.values())
        cut = sld.clusters.get(hemi)
        return (
            layer.key, layer.volume_index, layer.threshold_index, layer.threshold,
            layer.colormap, layer.colormap_reversed, layer.range_lo, layer.range_hi,
            str(layer.sign_mode), str(layer.alpha_mode), layer.opacity, layer.n_panes,
            self.state.time_index if layer.time_linked else -1,
            masked, None if cut is None else int(cut[1].sum()),
        )  # fmt: skip

    def _surface_choice(self, hemi: str) -> Layer | None:
        """The surface layer the 3-D window shows on ``hemi``: the selected one when it
        is a visible surface layer there, else the top-most visible one. Unticking
        the selected one hands the surface to the next, as on the slices."""
        selected = self.state.selected_layer()
        on_selected = self.surface_layers.get(selected.key) if selected is not None else None
        if (
            selected is not None
            and selected.visible
            and on_selected is not None
            and hemi in on_selected.parts
        ):
            return selected
        for layer in reversed(list(self.state.layers)):
            sld = self.surface_layers.get(layer.key)
            if layer.visible and sld is not None and hemi in sld.parts:
                return layer
        return None

    def surface_vertex_rgba(self, hemi: str):
        """``(rgba (V, 4), layer)`` for ``hemi`` in the 3-D window, or None.

        The selected layer when it is a surface layer on this hemisphere, else the
        top-most visible one: the controls drive whichever surface result is in
        front of you. Coloured by the layer's own display parameters (see
        viewer.surfacelayers.vertex_rgba); a cluster table's cut applies when a
        cluster window is masking this layer.
        """
        from fastfuncstuff.viewer.surfacelayers import vertex_rgba

        chosen = self._surface_choice(hemi)
        if chosen is None:
            return None
        sld = self.surface_layers[chosen.key]
        data = sld.parts[hemi]
        idx = self.state.time_index if chosen.time_linked else chosen.volume_index
        idx = max(0, min(int(idx), data.values.shape[1] - 1))
        values = self.vertex_frame(chosen, hemi, idx)
        stat = (
            data.values[:, chosen.threshold_brick] if chosen.threshold_index is not None else values
        )
        keep = None
        if any(chosen.key in m for m in self._display_masks.values()) and hemi in sld.clusters:
            keep = ~sld.clusters[hemi][1]
        return vertex_rgba(chosen, values, stat, keep), chosen

    def _on_loaded(self, key: str) -> None:
        self._notify_loaded(key)

    #: Replaced by the UI to marshal completion back onto its own thread. The
    #: worker calls this from a load thread, so the default must stay trivial.
    _notify_loaded: Callable[[str], None] = staticmethod(lambda key: None)

    def on_loaded(self, callback: Callable[[str], None]) -> None:
        """Install the completion hook for background loads."""
        self._notify_loaded = callback  # type: ignore[assignment]

    def load(self, path: str | Path, *, key: str | None = None) -> str:
        """Add a dataset as a new top layer; returns its key."""
        chosen = key or self.state.layers.mint_key()
        self.do(AddLayer(str(path), chosen))
        return chosen

    # -- stepping ------------------------------------------------------
    def frame_step(self, delta: int) -> Command | None:
        """What `,` and `.` move: time, or the selected layer's own sub-bricks.

        A selected 4-D layer that does not follow time -- a stats bucket, a QC
        stack -- is what is being looked at, so the keys step it rather than
        scrub every other layer while it sits still. With a threshold that
        follows at an offset, one step is one group (Coef to Coef, the t
        following). Anything else steps the shared time index. ``None`` when
        there is nothing to move.
        """
        from fastfuncstuff.viewer.vocab import brick_stops

        st = self.state
        layer = st.selected_layer()
        if layer is not None and layer.n_volumes > 1 and not layer.time_linked:
            stops = brick_stops(layer)
            if not stops:
                return None
            at = layer.volume_index
            if delta > 0:
                nxt = next((i for i in stops if i > at), stops[0])
            else:
                nxt = next((i for i in reversed(stops) if i < at), stops[-1])
            return None if nxt == at else SetVolume(layer.key, nxt)
        hi = st.max_time_index()
        if hi <= 0:
            return None
        return SetIndex((st.time_index + delta) % (hi + 1))

    def brick_caption(self, key: str) -> str:
        """``name: label`` of the sub-brick a layer shows, and what it is cut on."""
        layer = self.state.layers.get(key)

        def label(i: int) -> str:
            return layer.labels[i] if i < len(layer.labels) else f"#{i}"

        text = f"{layer.name}: {label(layer.volume_index)}"
        if layer.threshold_index is not None and layer.threshold_brick != layer.volume_index:
            text += f"  (cut on {label(layer.threshold_brick)})"
        return text

    # -- previewing ----------------------------------------------------
    @property
    def preview_path(self) -> str | None:
        """The file on show as a preview layer, if one is."""
        return self._preview[1] if self._preview else None

    def preview(self, path: str | Path | None) -> Aspect:
        """Show a file's first volume on top of the stack, without committing to it.

        For hunting through a long directory: each look costs the header and
        volume 0 (milliseconds even for a 250 MB run), never the background
        inflate a real LOAD starts, and nothing is recorded -- a script of a
        session should not replay forty files glanced at on the way to one.
        ``None`` withdraws the preview. :meth:`commit_preview` keeps it.
        """
        dirty = Aspect.NOTHING
        if self._preview is not None:
            key, _, selected = self._preview
            self._preview = None
            if self.state.layers.find(key) is not None:
                dirty |= self._unrecorded(RemoveLayer(key))
            if selected is None or self.state.layers.find(selected) is not None:
                self.state.selected = selected
        if path is None:
            return dirty
        selected = self.state.selected
        key = self.state.layers.mint_key("L")
        self._previewing = True
        try:
            dirty |= self._unrecorded(Load(str(path), key))
        finally:
            self._previewing = False
        self._preview = (key, str(path), selected)
        return dirty

    def commit_preview(self) -> str | None:
        """Keep the preview layer as an ordinary LOAD; returns its key.

        Recorded with its key: the previews before it consumed keys, so a
        replay minting afresh would name it differently from every command
        after it that refers to it.
        """
        if self._preview is None:
            return None
        key, path, _ = self._preview
        self._preview = None
        layer = self.state.layers.find(key)
        if layer is None:
            return None
        self.bus.record(Load(path, key))
        if layer.n_volumes > 1:
            self.store.load_async(key, on_done=self._on_loaded)
        return key

    def _unrecorded(self, cmd: Command) -> Aspect:
        was = self.bus.recording
        self.bus.recording = False
        try:
            return self.do(cmd)
        finally:
            self.bus.recording = was

    # -- dispatch ------------------------------------------------------
    def _on_command(self, cmd: Command, dirty: Aspect) -> None:
        # Not on TIME: a time-linked layer's cache key already carries the
        # index, and dropping everything re-uploaded the static underlay on
        # every volume step -- 45 ms of a 110 ms step on a 1 mm anatomy.
        # Anything that rewrites a layer's voxels invalidates that key itself.
        if dirty & Aspect.LAYERS:
            self.invalidate()
        elif isinstance(cmd, SetVolume):
            self.invalidate(cmd.key)
        # The mode reacts after the state has settled, so an InstaCorr seed sees
        # the seed already moved. Its own dirty aspects are folded into what the
        # dispatch reports, which is how a recomputed overlay reaches the panes.
        self._mode_dirty = self.mode.on_command(cmd, dirty)

    def do(self, cmd: Command) -> Aspect:
        self._mode_dirty = Aspect.NOTHING
        before, before_mm = self.state.crosshair, self.state.crosshair_mm
        dirty = self.bus.dispatch(cmd)
        if isinstance(cmd, SetIJK | SetXYZ) and self.state.crosshair != before:
            self._note_move(before, before_mm)
        return dirty | self._mode_dirty

    def _note_move(self, before: tuple[int, int, int], before_mm) -> None:
        """Remember where the crosshair left from, if that move was a jump.

        A jump is a move of more than one voxel away from a place it rested
        at. Arrow nudges and slice scrolling step one voxel, so they leave the
        target alone: jumping somewhere, nudging to look around, then `j`
        goes back to where you were before the jump.
        """
        now = self.clock()
        far = max(abs(a - b) for a, b in zip(self.state.crosshair, before, strict=True)) > 1
        if far and before_mm is not None and now - self._last_move >= JUMP_REST_S:
            self.state.crosshair_back = before_mm
        self._last_move = now

    # -- catalog -------------------------------------------------------
    def read_directory(self, directory: str | Path, *, recursive: bool = False) -> list:
        """Populate the catalog the pickers draw from.

        A directory read before is rescanned against what it held then, so
        going back to it stats its files rather than re-reading every header
        (0.7 s for 677 datasets, against milliseconds).
        """
        self._keep_catalog()
        known = self._catalogs.get((Path(directory), bool(recursive)))
        if known is None:
            self.catalog = catalog_mod.scan(directory, recursive=recursive)
        else:
            self.catalog = catalog_mod.rescan(directory, known, recursive=recursive).entries
        self.catalog_dir = Path(directory)
        self.catalog_recursive = bool(recursive)
        self.catalog_fresh = {}
        self._note_directory(directory, recursive=recursive)
        return self.catalog

    def read_files(self, paths: Iterable[str | Path]) -> list:
        """Populate the catalog from these files alone, in the order given.

        No directory is listed, now or on a rescan. On a synced folder (Dropbox)
        a READ's header pass can pull down every file in it; this reads the
        headers of the files named and only stats them afterwards.
        """
        self._keep_catalog()
        listed = tuple(dict.fromkeys(Path(p).expanduser() for p in paths))
        known = self._catalogs.get(listed)
        if known is None:
            self.catalog = list(catalog_mod.describe(listed))
        else:
            self.catalog = catalog_mod.rescan_files(listed, known).entries
        self.file_list = listed
        self.catalog_dir = None
        self.catalog_recursive = False
        self.catalog_fresh = {}
        return self.catalog

    @property
    def reading_files(self) -> bool:
        """Whether the catalog is the file list rather than a directory."""
        return self.catalog_dir is None and bool(self.file_list)

    @property
    def work_dir(self) -> Path | None:
        """Where this session's data is, for starting a dialog or placing a file.

        The directory read, else the first listed file's. Naming it lists nothing.
        """
        if self.catalog_dir is not None:
            return Path(self.catalog_dir)
        return self.file_list[0].parent if self.file_list else None

    def _keep_catalog(self) -> None:
        if self.catalog_dir is not None:
            self._catalogs[(self.catalog_dir, self.catalog_recursive)] = self.catalog
        elif self.file_list:
            self._catalogs[self.file_list] = self.catalog

    def offer_directories(self, directories: Iterable[str | Path]) -> None:
        """Put directories on the numbered list without reading them.

        Read when picked: pointing at several must not scan them all.
        """
        for directory in directories:
            self._note_directory(directory)

    def _note_directory(self, directory: str | Path, *, recursive: bool = False) -> None:
        d = Path(directory).expanduser().resolve()
        self.directories[d] = self.directories.get(d, False) or bool(recursive)
        self._directory_numbers.clear()

    def directory_number(self, path: str | Path) -> int | None:
        """The number of the directory a file came from, or ``None`` if from none of them.

        Its own directory first; failing that, the deepest one read
        recursively that holds it.
        """
        cached = self._directory_numbers.get(str(path), -1)
        if cached != -1:
            return cached
        parent = Path(path).expanduser().resolve().parent
        found = None
        for n, (d, recursive) in enumerate(self.directories.items(), start=1):
            if d == parent:
                found = n
                break
            if recursive and parent.is_relative_to(d):
                if found is None or len(d.parts) > len(list(self.directories)[found - 1].parts):
                    found = n
        self._directory_numbers[str(path)] = found
        return found

    def apply_rescan(self, result: catalog_mod.Rescan) -> list[str]:
        """Take a rescan's catalog, mark what is new, reload what was overwritten.

        Returns the keys of the layers reloaded. Runs on the GUI thread: the
        header reads happened on a worker, and this only swaps lists and
        dispatches.
        """
        self.catalog = result.entries
        for path in result.removed:
            self.catalog_fresh.pop(path, None)
        for path in result.added:
            self.catalog_fresh[path] = "new"
        for path in result.changed:
            # Something new that was then rewritten is still new to the person.
            self.catalog_fresh.setdefault(path, "updated")
        reloaded = self.layers_from(result.changed)
        for key in reloaded:
            self.do(ReloadLayer(key))
        return reloaded

    def layers_from(self, paths) -> list[str]:
        """Keys of loaded file layers whose file is one of ``paths``."""
        wanted = {_canonical(p) for p in paths}
        return [
            ly.key
            for ly in self.state.layers
            if ly.source == "file" and _canonical(ly.path) in wanted
        ]

    def drop_layer_data(self, key: str) -> None:
        """Forget everything read from a layer's file, ahead of reading it again.

        Unlike :meth:`forget` this does not spare the mode's input: a file
        overwritten under the mode is exactly the case where its cached voxels
        are wrong.
        """
        layer = self.state.layers.find(key)
        self.invalidate(key)
        self.forget_rois(key)
        for stale in [k for k in self._threshold_scales if k[0] == key]:
            del self._threshold_scales[stale]
        if layer is not None:
            self._provenance_cache.pop(layer.path, None)
            self._run_cache.clear()
        self.store.close(key)

    def apply_overlay_defaults(self, key: str) -> None:
        """Give a freshly-picked overlay the colour scale its contents call for.

        Only statistics get a threshold, at p 0.001 -- the cut a stats map is
        usually read at. Anything else (a mean, a tSNR map, a second
        anatomical) is a picture, and a percentile cut on a picture just hides
        part of it. Likewise the colormap: gray for a picture, hot for stats,
        which :meth:`overlay_look` turns diverging for a coefficient or a
        balanced signed map. "Statistics" is the signal the file picker uses:
        stataux, sub-brick labels, or a stats-like filename.
        """
        layer = self.state.layers.find(key)
        if layer is None or layer.is_computed:
            return
        stem = layer.name.lower()
        stats = bool(layer.stataux or layer.labels) or any(
            t in stem for t in catalog_mod.STATS_NAME_TOKENS
        )
        label = (layer.labels[0] if layer.labels else "").lower()
        coef = is_coefficient(layer, 0)
        # A bucket remembers its p even when it opens on a coefficient, so its
        # first t gets it.
        from fastfuncstuff.viewer.vocab import _threshold_at_p

        threshold_p = OVERLAY_START_P if layer.stataux else None
        at_p = _threshold_at_p(layer, threshold_p)
        if (
            at_p is None
            and not layer.stataux
            and any(t in (label or stem) for t in ("tstat", "zstat", "zscore", "zscr"))
        ):
            # A t or z with no dof on record: the normal cut is the large-dof
            # limit of the t one, slightly lenient on a short run.
            from scipy.stats import norm

            at_p = float(norm.isf(OVERLAY_START_P / 2))
        # Alpha stays off: a hard threshold is what a stats map is read at, and
        # a fade makes "which voxels survive" something you have to squint at.
        self.state.layers.update(
            key,
            threshold=0.0 if at_p is None else at_p,
            threshold_p=threshold_p,
            **self.overlay_look(key, 0, colormap="hot" if stats or coef else "gray"),
        )

    def _look_values(self, key: str, index: int) -> np.ndarray:
        """The values a colour scale is read from: a surface layer's own vertices
        (its painted ribbon counts each vertex once per voxel it owns), else the
        volume."""
        sld = self.surface_layers.get(key)
        layer = self.state.layers.find(key)
        if sld is not None and layer is not None:
            return np.concatenate([self.vertex_frame(layer, h, index) for h in sld.parts])
        return self.shown_volume(key, index)

    def overlay_look(self, key: str, index: int, *, colormap: str) -> dict[str, object]:
        """Colour scale for one sub-brick of an overlay: range, and hot or red-blue.

        Mirrored about zero, with a diverging map, only when the map is signed
        *and* its two tails are of comparable size (:data:`MIRROR_BALANCE`), or
        when it is a coefficient. A map with a sliver of negative noise is
        one-signed for display purposes: it runs from zero to the top
        percentile, and mirroring it would spend half the bar on nothing. The
        colormap is only swapped between the two defaults -- a map someone
        chose, gray included, stays.
        """
        try:
            values = self._look_values(key, index)
        except (KeyError, FileNotFoundError, ValueError):
            return {}
        sample = _look_sample(values)
        if sample.size == 0:
            return {}
        layer = self.state.layers.find(key)
        coef = layer is not None and is_coefficient(layer, index)
        pct = AUTORANGE_PERCENTILES[1]
        negative, positive = -sample[sample < 0], sample[sample > 0]
        neg_top = float(np.percentile(negative, pct)) if negative.size else 0.0
        pos_top = float(np.percentile(positive, pct)) if positive.size else 0.0
        share = negative.size / sample.size
        balanced = MIRROR_MIN_SHARE <= share <= 1.0 - MIRROR_MIN_SHARE and min(
            neg_top, pos_top
        ) >= MIRROR_BALANCE * max(neg_top, pos_top)
        out: dict[str, object] = {}
        if coef or balanced:
            top = float(np.percentile(np.abs(sample), pct)) or 1.0
            out.update(range_lo=-top, range_hi=top)
        elif share > 0.5:
            out.update(range_lo=-(neg_top or 1.0), range_hi=0.0)
        else:
            out.update(range_lo=0.0, range_hi=pos_top or 1.0)
        if colormap in ("hot", "redblue"):
            out["colormap"] = "redblue" if out["range_lo"] < 0 else "hot"  # type: ignore[operator]
        return out

    def negative_tail(self, key: str, index: int) -> float | None:
        """The value at the same percentile into the negative tail that max uses
        on the positive one, for setting min in one click. ``None`` when the
        sub-brick has no negative values."""
        try:
            sample = _look_sample(self._look_values(key, index))
        except (KeyError, FileNotFoundError, ValueError):
            return None
        negative = sample[sample < 0]
        if negative.size == 0:
            return None
        return float(np.percentile(negative, 100.0 - AUTORANGE_PERCENTILES[1]))

    def threshold_scale(self, key: str) -> float:
        """How far a threshold slider on this layer should reach.

        Read from the sub-brick the threshold *applies to*, not the displayed
        one: colouring by a beta of 0.3 and thresholding on its t of 12 is the
        ordinary case, and a slider spanning the beta can never reach the t.
        """
        layer = self.state.layers.get(key)
        brick = layer.threshold_brick
        # Only a file's sub-bricks hold still; a mode rewrites its overlay in
        # place under the same key.
        cacheable = layer.source == "file" and not layer.time_linked
        cached = self._threshold_scales.get((key, brick)) if cacheable else None
        if cached is not None:
            return cached
        try:
            values = self.volume(key, None if layer.time_linked else brick)
        except (KeyError, FileNotFoundError, ValueError):
            return 1.0
        finite = np.abs(values[np.isfinite(values)])
        scale = (float(finite.max()) if finite.size else 0.0) or 1.0
        if cacheable:
            self._threshold_scales[(key, brick)] = scale
        return scale

    def suggested_underlay(self) -> CatalogEntry | None:
        return catalog_mod.suggest_underlay(self.catalog)

    # -- modes ---------------------------------------------------------
    # -- viewports -----------------------------------------------------
    def open_view(self, kind: ViewKind, plane: Plane) -> str:
        """Open a window and return its id.

        The id is minted here and passed into the command rather than being
        returned by it, so the recorded line names the window it opened and
        every later line that addresses that window still resolves on replay.
        """
        vid = self.state.viewports.mint_id(kind)
        self.do(OpenView(vid, str(kind), str(plane)))
        return vid

    def default_layout(self) -> Aspect:
        """Open what a viewer with nothing configured should show.

        Three images, no graph. Goal zero is an underlay and an overlay
        together; a graph is something you ask for.

        Dispatched rather than built directly, so the recording starts with the
        windows it started with -- a script that replays into a viewer with no
        windows in it is a script that does not restore the session.
        """
        if len(self.state.viewports):
            return Aspect.NOTHING
        dirty = Aspect.NOTHING
        for plane in (Plane.AXIAL, Plane.SAGITTAL, Plane.CORONAL):
            dirty |= self.do(OpenView(self.state.viewports.mint_id(ViewKind.IMAGE), "image", plane))
        return dirty

    def open_mode_panels(self) -> Aspect:
        """Open a trace window for each panel the active mode names, once.

        Dispatched, so the windows are in the recording like any other. A panel
        already on screen is left alone -- re-entering a mode must not stack a
        second spectrum window over the first.
        """
        from fastfuncstuff.viewer.vocab import SetViewPanel

        shown = {v.panel for v in self.state.viewports.of_kind(ViewKind.TRACE)}
        dirty = Aspect.NOTHING
        for name in self.mode.panel_names():
            if name in shown:
                continue
            vid = self.open_view(ViewKind.TRACE, Plane.AXIAL)
            dirty |= Aspect.VIEWPORTS | self.do(SetViewPanel(vid, name))
        return dirty

    def close_view(self, vid: str) -> Aspect:
        return self.do(CloseView(vid))

    def graph_layers(self) -> list[Layer]:
        """Layers a graph can plot: the time-linked ones, bottom-up.

        A 3-D anatomy is never offered. It has no time course, and listing it
        with an empty checkbox invites the reading that the trace is hidden
        rather than that it does not exist.
        """
        return [ly for ly in self.state.layers if ly.time_linked and ly.n_volumes > 1]

    def traces_for(self, viewport) -> list[Layer]:
        """The layers one graph viewport should plot.

        An empty selection means all of them: a new layer starts plotted, which
        is what someone who just loaded it expects. Keys that no longer name a
        layer are dropped rather than erroring, because a viewport outlives the
        layers it was pointed at.
        """
        available = self.graph_layers()
        if not viewport.traces:
            return available
        wanted = set(viewport.traces)
        return [ly for ly in available if ly.key in wanted]

    def series_source(self, viewport) -> Layer | None:
        """The one run a carpet or a matrix window is built from.

        Reuses a graph's trace selection rather than inventing a second way to
        say "this layer": a carpet is a graph of every voxel and a matrix is a
        graph of every pair, and the picker that chooses what a graph plots is
        the same question in all three.
        """
        available = self.graph_layers()
        if not available:
            return None
        for key in viewport.traces:
            found = next((ly for ly in available if ly.key == key), None)
            if found is not None:
                return found
        return available[-1]

    def top_visible(self, *, besides: str | None = None) -> Layer | None:
        """The topmost drawn layer, optionally ignoring one.

        "What is on top" rather than "overlay-prime": in a stack of anat, run
        and stats the thing worth reading a colour off is the stats map, and
        index 1 is the run. Asked by everything that has to colour something
        outside the panes -- a carpet's sidebar, a graph cell's wash -- so that
        all of them agree with the picture and with each other.
        """
        for layer in reversed(list(self.state.layers)):
            if layer.visible and layer.key != besides:
                return layer
        return None

    def carpet_overlay(self, source: Layer) -> Layer | None:
        """Which layer labels a carpet's rows: whatever is on top of the run."""
        return self.top_visible(besides=source.key)

    def build_carpet(self, viewport, *, progress=None):
        """Render one carpet window's picture. Slow; runs on the worker.

        Reads arrays and returns one, like a mode's prepare() -- no session
        state is touched, because the thread that paints owns all of that.
        """
        from fastfuncstuff.viewer import carpet as carpet_mod

        layer = self.series_source(viewport)
        if layer is None:
            raise ValueError("no time series loaded to draw a carpet of")
        data = self.store.ensure_ram(layer.key)

        overlay = self.carpet_overlay(layer)
        order_volume = None
        if viewport.order in ("overlay", "roi"):
            if overlay is None or overlay.key == layer.key:
                raise ValueError(f"ordering by {viewport.order!r} needs an overlay above the run")
            order_volume = self._aligned_volume(overlay, layer)

        seed = None
        if viewport.order == "seed":
            ijk = self.state.seed or self.state.crosshair
            seed = self.timeseries(layer.key, ijk)

        sidebar = None
        if overlay is not None and overlay.key != layer.key:
            sidebar = self._aligned_volume(overlay, layer)

        return layer, carpet_mod.build_carpet(
            data,
            order=viewport.order,
            seed_series=seed,
            order_volume=order_volume,
            sidebar_volume=sidebar,
            polort=int(viewport.detrend),
            normalize=viewport.scaling,
            device=self.store.device,
            progress=progress,
        )

    def matrix_rois(self, viewport):
        """The ROI set a matrix window uses, or ``None`` for voxel bins.

        A named layer wins; otherwise the topmost ROI layer, so dropping an
        atlas on the stack is enough to turn every open matrix into a
        connectivity matrix without visiting a picker.
        """
        if viewport.rois:
            return self.roi_set(viewport.rois)
        for layer in reversed(self.roi_layers()):
            found = self.roi_set(layer.key)
            if found is not None and len(found):
                return found
        return None

    def build_matrix(self, viewport, *, progress=None):
        """Build one matrix window's picture. Slow; runs on the worker.

        Reads arrays and returns one, like a mode's prepare() -- no session
        state is touched, because the thread that paints owns all of that.
        """
        from fastfuncstuff.viewer import matrix as matrix_mod

        layer = self.series_source(viewport)
        if layer is None:
            raise ValueError("no time series loaded to correlate")
        data = self.store.ensure_ram(layer.key)

        rois = self.matrix_rois(viewport)
        if rois is not None and rois.shape != layer.shape:
            # Saying so beats resampling silently: an atlas averaged on the
            # wrong grid gives a full matrix of plausible numbers.
            raise ValueError(
                f"{rois.name} is on a {rois.shape} grid and {layer.name} is {layer.shape}; "
                "resample one to the other to correlate them"
            )
        return layer, matrix_mod.build_matrix(
            data,
            rois=rois,
            order=viewport.matrix_order,
            polort=int(viewport.detrend),
            normalize=viewport.scaling,
            device=self.store.device,
            progress=progress,
        )

    def _aligned_volume(self, layer: Layer, like: Layer) -> np.ndarray | None:
        """One layer's values on another's voxel grid, or None if they differ.

        A carpet's rows are the *series*' voxels, so an overlay can only label
        them if it sits on the same grid. Resampling it here would work, but a
        silent resample is how a stat map ends up labelling the wrong rows --
        better to say the overlay does not apply.
        """
        if layer.shape != like.shape or not np.allclose(layer.affine, like.affine, atol=1e-4):
            return None
        volume = self.volume(layer.key)
        return np.asarray(volume, dtype=np.float32)

    # -- how a layer is drawn into the grid --------------------------------

    #: Beyond this ratio of layer voxel to display voxel, ``auto`` calls it
    #: upsampling. Not 1.0, because two grids that differ by a rounding error
    #: are the same grid, and flipping interpolation on that is noise.
    UPSAMPLE_RATIO = 1.2

    @staticmethod
    def is_label_layer(layer: Layer) -> bool:
        """Whether a layer's values are identities, so only nearest may sample it.

        An ROI, atlas, mask, painted or cluster layer, or data that loaded as
        integer labels. Between label 0 and label 3, linear and cubic paint a
        1.5 that is no region at all -- at the edge of a depth-layer file on a
        finer grid, a rim of "low depth" that is really outside the cortex.
        """
        return layer.roi or layer.integer_labels

    def resample_mode(self, layer: Layer) -> str:
        """``"nearest"``, ``"linear"`` or ``"cubic"`` for one layer. Display only.

        This changes how the layer is *drawn* and nothing else. The voxels in
        the store are untouched, every readout, graph, carpet, cluster table
        and mode input still reads the layer's own array on the layer's own
        grid, and switching it does not invalidate a fit.

        ``auto`` decides by direction; see :attr:`Layer.resample` for why that
        is the axis that matters. The exception is the underlay on a resized
        grid (:attr:`ViewerState.grid_mm`): a finer grid exists so that the
        anatomy is the layer interpolated, and nearest would only redraw its
        own voxels as blocks. Set its DRAW to nearest to see them anyway.

        A label layer (:meth:`is_label_layer`) is nearest whatever is set,
        underlay included: there is no in-between of two labels to draw.
        """
        if self.is_label_layer(layer):
            return "nearest"
        if layer.resample in ("nearest", "linear", "cubic"):
            return layer.resample
        grid = self.state.grid
        if grid is None:
            return "linear"
        base = self.state.layers.base
        if base is not None and base.key == layer.key:
            return "linear"
        return (
            "nearest"
            if _voxel_size(layer.affine) > _voxel_size(grid.affine) * self.UPSAMPLE_RATIO
            else "linear"
        )

    # -- what a mode reads -----------------------------------------------

    def input_candidates(self, mode: Mode | None = None) -> list[Layer]:
        """Every layer the mode could read, topmost first.

        Every *loaded* layer, not every visible one. Unticking a run is how you
        get it out of the picture once its stats map is drawn over the anatomy,
        and it would be a strange viewer in which that also stopped you fitting
        the run.
        """
        mode = self.mode if mode is None else mode
        return [ly for ly in reversed(list(self.state.layers)) if mode.accepts(ly)]

    def input_layer(self, mode: Mode | None = None) -> Layer | None:
        """The layer the mode reads: the named one, else the best candidate.

        The fallback is recomputed rather than written back, unlike
        :meth:`ViewerState.selected_layer`, and the difference is deliberate.
        A selection is a gesture and has to stay where it was put; an input
        nobody has named is a default, and a default that froze onto the first
        run loaded would mean the second run could never be fitted without
        first being found in a menu.
        """
        mode = self.mode if mode is None else mode
        named = self.state.layers.find(self.state.input_key) if self.state.input_key else None
        if named is not None and mode.accepts(named):
            return named
        return mode.default_input(self.input_candidates(mode))

    def set_input(self, key: str | None) -> Aspect:
        """Name the layer the mode reads, or clear it back to the default."""
        key = key or None
        if key is not None and self.state.layers.find(key) is None:
            raise KeyError(f"no layer {key!r} to use as input")
        before = self.input_layer()
        self.state.input_key = key
        after = self.input_layer()
        if before is not None and after is not None and before.key == after.key:
            return Aspect.NOTHING
        # The prepared array belongs to the old input, so it is not merely
        # stale -- recomputing without re-preparing would draw the last run's
        # map and label it with this one's name.
        self.mode.invalidate()
        return Aspect.LAYERS | Aspect.GRAPH | self.mode.refresh()

    def set_mode(self, name: str) -> Aspect:
        """Switch modes, tearing down the old one's overlay."""
        if self.mode.name == name:
            return Aspect.NOTHING
        cls = registry.get(name)
        self.mode.detach()
        cached = self._modes.get(name)
        self.mode = cached if isinstance(cached, cls) else cls()
        self._modes[name] = self.mode
        self.mode.defer_preparation = self.defer_mode_preparation
        self.mode.attach(self)
        return (Aspect.LAYERS | Aspect.SLICES | Aspect.GRAPH) | self.mode.refresh()

    def mode_named(self, name: str) -> Mode | None:
        """This session's instance of a mode it has been in, active or not.

        How one mode reads another's state without the stack in between --
        Denoise taking the components ICA's review labelled noise.
        """
        return self._modes.get(name)

    def refresh_mode(self) -> Aspect:
        """Recompute and install the active mode's overlay."""
        return self.mode.refresh()

    def set_mode_param(self, param: str, value: str) -> Aspect:
        """Coerce a text parameter to its control's type and apply it."""
        spec = next((c for c in self.mode.controls() if c.name == param), None)
        if spec is None:
            raise KeyError(f"mode {self.mode.name!r} has no parameter {param!r}")
        coerced: object = value
        default = getattr(spec, "default", None)
        if isinstance(default, bool):
            coerced = value not in ("0", "false", "False", "")
        elif isinstance(default, int):
            coerced = int(float(value))
        elif isinstance(default, float):
            coerced = float(value)
        return self.mode.set_param(param, coerced)

    def mode_series(self, ijk: tuple[int, int, int] | None = None) -> list[Trace]:
        return self.mode.series(ijk or self.state.crosshair)

    # -- overlay readout -------------------------------------------------

    def overlay_readout(self) -> list[str]:
        """One line per visible overlay: its value(s) under the crosshair, top layer first.

        A stats overlay cut on another sub-brick shows both numbers -- the beta
        you are colouring by and the t deciding whether it is drawn -- since
        either alone leaves you asking what the other one is. A region layer
        names the region. The layer's name leads only when there are several
        overlays, because with one it is already on screen in the picker.
        """
        st = self.state
        if st.grid is None:
            return []
        # A colour layer is read out even at the bottom of the stack: its
        # numbers are the channels, which no other line shows.
        overlays = [ly for i, ly in enumerate(st.layers) if ly.visible and (i > 0 or ly.colour)]
        lines: list[str] = []
        for layer in reversed(overlays):
            reading = self.layer_reading(layer)
            if reading is None:
                continue
            prefix = f"{_short(layer.name)}  " if len(overlays) > 1 else ""
            lines.append(
                ValueLine(prefix, reading.long) if not layer.roi else prefix + reading.long
            )
        # The region the crosshair is in, from the subject's own parcellation
        # and label volume -- whatever else is loaded.
        lines.extend(self.surfaces.region_lines(st.crosshair_mm, st.surface_annot, st.volume_atlas))
        return lines

    def layer_reading(self, layer: Layer) -> LayerReading | None:
        """One layer's value(s) under the crosshair, or ``None`` with no data to read.

        The single lookup behind every readout -- the layer list's value column,
        the status bar, the optional box on the image -- so they cannot disagree
        and a crosshair move pays for each layer once. A stats overlay cut on
        another sub-brick reads both: the beta you colour by and the t that
        decides whether it is drawn.
        """
        from fastfuncstuff.viewer.slicing import voxel_value

        st = self.state
        if st.grid is None:
            return None
        if layer.colour:
            # Before display_volume: for a colour layer that is a whole volume
            # of the scalar stand-in, built and copied to the host on every
            # frame just to be ignored here.
            source = self.colour_sources.get(layer.key)
            if source is None:
                return None
            ijk = self.layer_voxel(layer.affine, st.crosshair)
            text = source.readout(ijk, self.layer_time(layer))
            return LayerReading(short=text.split("  |  ")[0], long=text)
        volume = self.display_volume(layer.key)
        if volume is None:
            return None
        value = voxel_value(volume, st.grid, layer.affine, st.crosshair)
        if layer.roi:
            name = self.region_name(layer, value)
            return LayerReading(short=name, long=name)
        short = [_fmt(value)]
        long = [f"{_brick_name(layer, layer.volume_index)}{_fmt(value)}"]
        if layer.threshold_index is not None and layer.threshold_index != layer.volume_index:
            stat_vol = self.display_volume(layer.key, index=layer.threshold_index)
            if stat_vol is not None:
                stat = voxel_value(stat_vol, st.grid, layer.affine, st.crosshair)
                short.append(_fmt(stat))
                long.append(f"{_brick_name(layer, layer.threshold_index)}{_fmt(stat)}")
        return LayerReading(short=" / ".join(short), long="   ".join(long))

    # -- overlay colour per voxel ---------------------------------------

    def overlay_colors(
        self, cells: list[tuple[int, int, int]]
    ) -> list[tuple[float, float, float] | None] | None:
        """The topmost drawn layer's colour at each display voxel, or ``None`` where cut.

        Drawn exactly as the slices draw it -- the same LUT, range, sign mode
        and threshold sub-brick -- so a cell's colour and the voxel under the
        crosshair cannot disagree. The threshold is applied hard whatever the
        alpha mode: a cell either shows a surviving voxel or it does not, and a
        faded wash would be unreadable behind the curves.

        ``None`` for the whole answer when there is no overlay to read.
        """
        from fastfuncstuff.viewer.colormap import (
            apply_colormap,
            apply_label_colors,
            threshold_alpha,
        )
        from fastfuncstuff.viewer.compose import cached_lut

        st = self.state
        layer = self.top_visible()
        base = st.layers.base
        if layer is not None and base is not None and layer.key == base.key:
            # Only the base image is drawn, and washing a graph in the anatomy
            # it is already sitting on says nothing.
            layer = None
        if layer is None or st.grid is None or not cells:
            return None
        volume = self.display_volume(layer.key)
        if volume is None:
            return None

        # Display voxels to this layer's voxels: an overlay need not share the
        # underlay's grid, and the slices resample it for the same reason.
        ijk = np.asarray(cells, dtype=float)
        mm = np.c_[ijk, np.ones(len(cells))] @ np.asarray(st.grid.affine, float).T
        own = np.rint(mm @ np.linalg.inv(np.asarray(layer.affine, float)).T)[:, :3].astype(int)
        shape = np.asarray(volume.shape[:3])
        inside = np.all((own >= 0) & (own < shape), axis=1)
        clipped = np.clip(own, 0, shape - 1)
        if layer.colour:
            source = self.colour_sources.get(layer.key)
            if source is None:
                return None
            rgb_c, drawn = source.rgb_at(clipped, self.layer_time(layer))
            return [
                (float(r), float(g), float(b)) if d and k else None
                for (r, g, b), d, k in zip(rgb_c, drawn, inside, strict=True)
            ]
        index = torch.as_tensor(clipped.T, device=volume.device)
        values = volume[index[0], index[1], index[2]].float()

        palette = self.roi_palette(layer.key, values.device) if layer.roi else None
        if palette is not None:
            rgb, alpha = apply_label_colors(values, palette)
        else:
            stat = values
            if layer.threshold_index is not None:
                stat_vol = self.display_volume(layer.key, index=layer.threshold_index)
                if stat_vol is not None:
                    stat = stat_vol[index[0], index[1], index[2]].float()
            lo = layer.range_lo if layer.range_lo is not None else 0.0
            hi = layer.range_hi if layer.range_hi is not None else 1.0
            rgb = apply_colormap(
                values,
                lut=cached_lut(layer.colormap, values.device, reverse=layer.colormap_reversed),
                lo=float(lo),
                hi=float(hi),
                sign_mode=layer.sign_mode,
                n_panes=layer.n_panes,
            )
            alpha = threshold_alpha(stat, layer.threshold, sign_mode=layer.sign_mode)
        rgb_np = rgb.float().cpu().numpy()
        keep = (alpha.float().cpu().numpy() > 0) & inside
        return [
            (float(r), float(g), float(b)) if k else None
            for (r, g, b), k in zip(rgb_np, keep, strict=True)
        ]

    # -- design regressors ----------------------------------------------
    #
    # Voxel-independent lines in a graph: a design column is the same curve in
    # every cell. Read from disk on demand and cached by path, so a crosshair
    # drag never re-parses a history or an xmat.

    def load_design(self, path: str | Path) -> Design:
        """Read a design and offer it in every graph window's DESIGN menu."""
        from fastfuncstuff.viewer.design import load_design

        design = load_design(path)
        self.designs[design.path] = design
        self._design_errors.pop(str(path), None)
        return design

    def design(self, path: str) -> Design | None:
        """A design by path as some history or pin wrote it, or ``None`` if unreadable.

        Both outcomes are remembered per spelling of the path: this is asked on
        every repaint, and resolving a path or failing to read one costs a trip
        to the filesystem each time -- except one stat of a known file, so a
        design rewritten by a rerun fit is read again rather than drawn stale.
        """
        if path in self._design_errors:
            return None
        resolved = self._design_paths.get(path)
        if resolved is not None and resolved in self.designs:
            cached = self.designs[resolved]
            try:
                current = os.stat(resolved).st_mtime_ns
            except OSError:
                return cached
            # One stat per repaint; re-reading the matrix only when a rerun
            # GLM has rewritten it.
            if current == cached.mtime_ns:
                return cached
        try:
            design = self.load_design(path)
        except (OSError, ValueError, IndexError) as exc:
            self._design_errors[path] = str(exc)
            return None
        self._design_paths[path] = design.path
        return design

    def _provenance(self, layer: Layer) -> Provenance | None:
        from fastfuncstuff.io.dsetinfo import read_info
        from fastfuncstuff.viewer.design import parse_provenance

        if layer.path not in self._provenance_cache:
            try:
                history = read_info(layer.path).history
            except (OSError, ValueError):
                history = ""
            path = Path(layer.path)
            self._provenance_cache[layer.path] = parse_provenance(
                history, stats_name=path.name, base=path.parent
            )
        return self._provenance_cache[layer.path]

    def auto_regressor(self) -> tuple[Design, int, int] | None:
        """The design column the shown stats sub-brick is about, for the graphed run.

        Only a ``_Coef`` or ``_Tstat`` of one column qualifies: a contrast or an
        F spans several, and picking one to draw would misdescribe it. The run
        is the first graphed layer that the fit lists as an input, and the
        column is drawn only if that run's length matches the layer's -- a
        trimmed or re-cut run would put every event in the wrong place.
        """
        st = self.state
        stats = [
            ly
            for ly in reversed(list(st.layers))
            if ly.visible and not ly.time_linked and ly.labels and ly.source == "file"
        ]
        for layer in stats:
            provenance = self._provenance(layer)
            if provenance is None:
                continue
            design = self.design(provenance.design)
            if design is None:
                continue
            index = layer.volume_index
            label = layer.labels[index] if 0 <= index < len(layer.labels) else ""
            col = design.column_for_brick(label)
            if col is None:
                return None
            for run_layer in self.graph_layers():
                run = self._run_of(provenance, run_layer.path)
                if run is None:
                    continue
                if run > design.n_runs or design.run_length(run) != run_layer.n_volumes:
                    return None
                return design, run, col
            return None
        return None

    def _run_of(self, provenance: Provenance, path: str) -> int | None:
        # Resolving paths touches the filesystem, and this is asked per repaint.
        key = (provenance, path)
        if key not in self._run_cache:
            self._run_cache[key] = run_of(provenance, path)
        return self._run_cache[key]

    def regressor_series(self, viewport) -> list[RegressorTrace]:
        """The automatic column, then the pinned ones, for one graph window."""
        from fastfuncstuff.viewer.design import AUTO_IDENT, Pin, pin_label

        out: list[RegressorTrace] = []
        auto = self.auto_regressor()
        if auto is not None:
            design, run, col = auto
            out.append(
                RegressorTrace(
                    AUTO_IDENT,
                    f"{pin_label(design, run, col)} (auto)",
                    f"{Path(design.path).name}: run {run}, column {col} "
                    "-- follows the stats sub-brick shown",
                    design.column(run, col),
                    Pin(design.path, run, col),
                )
            )
        for spec in viewport.regressors:
            pin = Pin.decode(spec)
            design = self.design(pin.path)
            if design is None:
                continue
            try:
                values = design.column(pin.run, pin.col)
            except IndexError:
                continue
            out.append(
                RegressorTrace(
                    pin.ident,
                    pin_label(design, pin.run, pin.col),
                    f"{Path(design.path).name}: run {pin.run}, column {pin.col}",
                    values,
                    pin,
                )
            )
        return out

    # -- derived layers -------------------------------------------------
    #
    # Same split as a mode's prepare()/compute(): the arithmetic is seconds
    # over a whole 4-D array and runs on a worker, so it must not touch session
    # state; installing the result is instant and happens on the GUI thread.

    def compute_denoise(
        self,
        key: str,
        *,
        matrix: str = "",
        polort: int = -1,
        keep_mean: bool = True,
        progress=None,
    ) -> tuple[np.ndarray, str]:
        """The slow half: residualise a run, and say what was projected out.

        Touches nothing but the arrays it reads and the one it builds, so it is
        safe on a worker thread.
        """
        from fastfuncstuff.viewer import derive

        layer = self.state.layers.get(key)
        if layer.n_volumes <= 1:
            raise ValueError(f"{layer.name} is not a time series; nothing to denoise")
        nuisance = derive.read_nuisance(matrix or None, n_time=layer.n_volumes, polort=polort)
        data = self.store.ensure_ram(key)
        values = derive.denoise(
            data,
            nuisance,
            device=self.store.device,
            keep_mean=keep_mean,
            progress=progress,
        )
        return values, nuisance.description

    def denoise(
        self,
        key: str,
        *,
        matrix: str = "",
        polort: int = -1,
        keep_mean: bool = True,
        progress=None,
    ) -> Aspect:
        """Both halves, in order. What the DENOISE command runs on replay."""
        values, detail = self.compute_denoise(
            key, matrix=matrix, polort=polort, keep_mean=keep_mean, progress=progress
        )
        return self.install_derived(key, values, op="denoise", detail=detail)

    def install_derived(
        self,
        source_key: str,
        values: np.ndarray,
        *,
        op: str,
        detail: str = "",
        name: str | None = None,
        labels: tuple[str, ...] | None = None,
        colormap: str | None = None,
        display_range: tuple[float, float] | None = None,
        visible: bool | None = None,
        select: bool = True,
    ) -> Aspect:
        """Put a computed dataset into the stack, right above what made it.

        Two decisions, both about making the comparison the easy one:

        * It lands **immediately above its source** and inherits how the source
          is drawn. Neighbours in the stack are what `[`, `]` and a soloed
          window flip between, so raw against denoised is one keypress -- the
          same gesture that checks an EPI against an anat.
        * Re-deriving from the same source with the same operation **replaces**
          the layer rather than pushing another. Clicking twice must not grow
          the stack without bound, and the parameters that produced it are in
          the recorded command either way.

        Inheriting the source's drawing is right for a dataset of the same kind
        -- a denoised or motion-corrected run -- and wrong for a QC volume made
        alongside it. A difference map is signed and centred on zero, so it
        wants its own colour map and its own symmetric range; carrying over the
        intensity window of the run it came from would draw it as a black
        rectangle. Hence the overrides: given, they win; omitted, the source
        still decides.
        """
        source = self.state.layers.get(source_key)
        tag = f"derived:{op}:{source_key}"
        existing = self.state.layers.find_by_source(tag)
        key = existing.key if existing is not None else self.state.layers.mint_key("D")
        name = name or f"{source.name} ·{op}d"
        lo, hi = display_range if display_range is not None else (source.range_lo, source.range_hi)

        self.store.adopt(key, values, name=name)
        self.invalidate(key)
        if existing is not None:
            self.state.layers.update(
                key,
                name=name,
                path=f"<{op}: {detail}>",
                n_volumes=int(values.shape[3]) if values.ndim == 4 else 1,
                labels=labels if labels is not None else source.labels,
                range_lo=lo,
                range_hi=hi,
            )
            return Aspect.LAYERS | Aspect.SLICES | Aspect.GRAPH

        self.state.layers.add(
            Layer(
                key=key,
                name=name,
                # No file backs it, so the field that would hold one carries
                # the provenance instead: what was projected out, in words.
                path=f"<{op}: {detail}>",
                shape=source.shape,
                n_volumes=int(values.shape[3]) if values.ndim == 4 else 1,
                affine=source.affine,
                labels=labels if labels is not None else source.labels,
                visible=source.visible if visible is None else visible,
                opacity=source.opacity,
                colormap=colormap or source.colormap,
                colormap_reversed=source.colormap_reversed if colormap is None else False,
                # The same display range as its source, so the two are
                # comparable at a glance rather than each auto-scaled to
                # itself -- which would hide exactly the difference you made
                # the layer to see.
                range_lo=lo,
                range_hi=hi,
                # A QC volume is not the run's time base: its sub-bricks are
                # "first" and "last", not time points, so scrubbing must not
                # drag it along.
                time_linked=source.time_linked and labels is None,
                source=tag,
            ),
            at=self.state.layers.index_of(source_key) + 1,
        )
        if select:
            self.state.selected = key
        return Aspect.LAYERS | Aspect.SLICES | Aspect.GRAPH

    # -- computed overlays ---------------------------------------------
    def _add_on_top(self, layer: Layer, *, hide: Iterable[str] = ()) -> None:
        """Push a made layer on top, selected, hiding the layers named.

        Only on creation, and only the ones named: a result is worth looking
        at, and a 4-D run left drawn under its own correlation map is noise
        over the anatomy, but those are the two facts -- everything else in the
        stack was put there on purpose. This used to hide every visible layer
        above the underlay, which made a second map impossible to keep beside
        the first without switching it back on after every recompute.
        """
        for key in hide:
            found = self.state.layers.find(key)
            if found is not None and found.visible and found.key != layer.key:
                self.state.layers.update(key, visible=False)
        self.state.layers.add(layer)
        self.state.selected = layer.key
        if self.state.grid is None:
            self.state.adopt_grid(layer.shape, layer.affine)

    # -- computed overlays ---------------------------------------------
    def install_computed_overlay(self, source: str, overlay: ComputedOverlay) -> str:
        """Install (or update in place) the layer a mode owns.

        Updating in place matters: a mode recomputes on every seed click, and
        pushing a new layer each time would grow the stack without bound and
        reset the threshold the user just set.

        On top of the stack, not in the overlay slot. It used to displace the
        primary overlay and hand it back when the mode was left, which made
        sense while an output died with its mode; now that it stays, a layer
        taken out of the stack would never come back.
        """
        if overlay.colour is not None:
            return self._install_colour(source, overlay)
        existing = self.state.layers.find_by_source(source)
        key = existing.key if existing is not None else self.state.layers.mint_key("M")
        array, shape_of = _overlay_array(overlay)
        self.store.adopt(key, array, name=overlay.name)
        self.invalidate(key)

        lo, hi = overlay.display_range or derive_range(overlay.values)
        cut = _overlay_cut(overlay)
        if existing is not None:
            # The name is identity -- showing "IC 0" while displaying IC 4 is a
            # lie. Range and threshold deliberately do NOT follow: stepping
            # through components at a threshold you set is how they get
            # reviewed, and resetting it on every step would undo the gesture.
            # Unless the mode says this output is a different quantity from the
            # last one, in which case the old scale describes nothing.
            changes: dict[str, object] = {}
            if existing.name != overlay.name:
                changes["name"] = overlay.name
            # The shape of the output can change under one layer: ticking a
            # threshold map on turns a one-volume overlay into two. Restated
            # every time rather than only on creation, because a layer left
            # claiming one volume while the store holds two thresholds on a
            # brick that is no longer there.
            changes.update(
                n_volumes=int(array.shape[3]),
                shape=shape_of,
                labels=overlay.volume_labels,
                volume_index=0,
                **cut,
            )
            if overlay.rescale:
                changes["threshold"] = overlay.threshold or 0.0
                # FIXED is the user saying the scale is theirs; the threshold
                # still follows, since a t cut left on an R2 map hides it all.
                if not existing.range_fixed:
                    changes.update(colormap=overlay.colormap, range_lo=lo, range_hi=hi)
            if changes:
                self.state.layers.update(key, **changes)
        else:
            consumed = self.mode.input_layer_key() if self.mode.layer_source == source else None
            self._add_on_top(
                hide=[consumed] if consumed else [],
                layer=Layer(
                    key=key,
                    name=overlay.name,
                    path=f"<{overlay.name}>",
                    shape=shape_of,
                    n_volumes=int(array.shape[3]),
                    labels=overlay.volume_labels,
                    affine=np.asarray(overlay.affine, dtype=float),
                    colormap=overlay.colormap,
                    range_lo=lo,
                    range_hi=hi,
                    threshold=overlay.threshold or 0.0,
                    source=source,
                    **cut,
                ),
            )
        return key

    def layer_time(self, layer: Layer) -> int:
        """The frame a layer shows: the global time index if it follows time."""
        return self.state.time_index if layer.time_linked else layer.volume_index

    def _install_colour(self, source: str, overlay: ComputedOverlay) -> str:
        """Install (or update in place) a mode's colour layer.

        Its store entry is lazy -- the first coordinate, made per frame on
        request -- so the generic paths that ask the store for a volume (AUTO,
        a montage, the 3-D window) see a sensible scalar and nothing holds a
        4-D copy of colours.
        """
        colour = overlay.colour
        n = int(colour.n_frames)
        shape = tuple(int(v) for v in colour.chans.shape)
        existing = self.state.layers.find_by_source(source)
        if existing is not None and self.colour_sources.get(existing.key) is colour:
            # The same source again (a refresh that changed nothing): re-adopting
            # would mint a new generation and make every window re-upload.
            return existing.key
        key = existing.key if existing is not None else self.state.layers.mint_key("M")
        self.colour_sources[key] = colour
        self.store.adopt_lazy(key, shape, n, colour.lightness_volume, name=overlay.name)
        self.invalidate(key)
        fields = dict(
            name=overlay.name,
            shape=shape,
            n_volumes=n,
            affine=np.asarray(overlay.affine, dtype=float),
            time_linked=n > 1,
            labels=overlay.volume_labels,
        )
        if existing is not None:
            self.state.layers.update(key, **fields)
            return key
        consumed = self.mode.input_layer_key() if self.mode.layer_source == source else None
        self._add_on_top(
            hide=[consumed] if consumed else [],
            layer=Layer(
                key=key,
                path=f"<{overlay.name}>",
                range_lo=0.0,
                range_hi=1.0,
                colour=True,
                source=source,
                **fields,  # type: ignore[arg-type]
            ),
        )
        return key

    def remove_computed_overlay(self, source: str) -> None:
        """Drop a mode's output layer, if it has one."""
        existing = self.state.layers.find_by_source(source)
        if existing is None:
            return
        self.state.layers.remove(existing.key)
        self.forget(existing.key)

    def keep_output(self, mode: Mode) -> Aspect:
        """Freeze the mode's live output as ``A_ICORR_1``, ``A_ICORR_2``, ...

        The copy goes directly under the live layer and starts hidden: it is
        identical to what is on top until the next seed moves the live one, and
        two identical maps stacked would only double the alpha. ``[`` and ``]``
        step to it, and solo flips between the two.
        """
        live = self.state.layers.find_by_source(mode.layer_source)
        if live is None:
            return Aspect.NOTHING
        stem = mode.output_name()
        taken = [ly for ly in self.state.layers if ly.source == f"kept:{mode.layer_source}"]
        number = 1 + max(
            (int(n) for ly in taken if (n := ly.name[len(stem) + 1 :].split(" ")[0]).isdigit()),
            default=0,
        )
        detail = live.name[len(stem) :].strip()
        name = f"{stem}_{number}" + (f" {detail}" if detail else "")
        key = self.state.layers.mint_key("K")
        if live.colour:
            # A source is never changed after it is made -- a new look is a
            # new source -- so the copy can share the live one's.
            frozen = self.colour_sources[live.key]
            self.colour_sources[key] = frozen
            self.store.adopt_lazy(
                key, live.shape, live.n_volumes, frozen.lightness_volume, name=name
            )
        else:
            values = np.array(self.store.ensure_ram(live.key), dtype=np.float32, copy=True)
            self.store.adopt(key, values, name=name)
        self.state.layers.add(
            live.with_(
                key=key,
                name=name,
                path=f"<{name}>",
                visible=False,
                source=f"kept:{mode.layer_source}",
            ),
            at=self.state.layers.index_of(live.key),
        )
        return Aspect.LAYERS | Aspect.SLICES

    def mode_action(self, name: str) -> Aspect:
        return self.mode.action(name)

    def forget(self, key: str) -> None:
        """Drop a layer's cached and resident data.

        Skips anything the current mode is using or has displaced: dropping a
        mode's source mid-session is what turns a re-prepare into a silent
        stale map.
        """
        held = {self.mode.input_layer_key()}
        if key in held:
            return
        self.invalidate(key)
        self.colour_sources.pop(key, None)
        self.painter.tables.pop(key, None)
        for stale in [k for k in self._threshold_scales if k[0] == key]:
            del self._threshold_scales[stale]
        for stale in [k for k in self._source_masks if k[0] == key]:
            del self._source_masks[stale]
        self.surface_layers.pop(key, None)
        self.store.close(key)

    def run_script(self, text: str) -> Aspect:
        return self.bus.run_script(text)

    def to_script(self, *, header: str | None = None) -> str:
        return self.bus.to_script(header=header)

    def launch_command(self) -> str:
        """The ``ffs_viewer`` command line that opens what is loaded now.

        What is on disk, by absolute path: the directories read, the stack's
        files bottom first, the subject's surfaces (and any
        meshes added beside them), and surface results. Not edits, display
        settings or windows -- that is the session script -- and not layers
        computed here (a mode's overlay, a DERIVE), which have no file to name;
        those are listed after a ``#``.

        The stack's files lead, because ``-read`` takes every path after it.
        Listed files nobody loaded are left out while anything is loaded: naming
        one would load it.
        """
        import shlex

        def full(p) -> str:
            return str(Path(p).expanduser().resolve())

        args = []
        if self.catalog_dir is not None:
            current = self.catalog_dir.expanduser().resolve()
            # A loaded file's directory rejoins the list when the file loads.
            loaded = {
                Path(ly.path).expanduser().resolve().parent for ly in self.state.layers if ly.path
            }
            others = [
                str(d) for d in self.directories if d != current and d not in loaded and d.is_dir()
            ]
            args += ["-read", str(current), *others]
        surfaces = self.surfaces
        surf_dir = None
        if surfaces.subject is not None:
            args += ["-surfaces", full(surfaces.subject)]
            if surfaces.space != "native":
                args += ["-surf_space", surfaces.space]
            surf_dir = Path(surfaces.subject).resolve() / "surf"
        for m in surfaces.meshes:
            if m.path is not None and Path(m.path).resolve().parent != surf_dir:
                args += ["-mesh", full(m.path)]
        files, surf_data, skipped = [], [], []
        for layer in self.state.layers:
            if layer.source == "file" and Path(layer.path).exists():
                files.append(full(layer.path))
            elif layer.source.startswith("surface:") and Path(layer.path).exists():
                surf_data += ["-surf_data", full(layer.path)]
            else:
                skipped.append(layer.name)
        # Naming a file loads it, so a list with nothing loaded reopens as that list.
        if self.reading_files and not files:
            files = [full(p) for p in self.file_list]
        line = shlex.join(["ffs_viewer", *files, *args, *surf_data])
        return line + (f"  # not reloaded: {', '.join(skipped)}" if skipped else "")

    def save_script(self, path: str | Path, *, header: str | None = None) -> Path:
        out = Path(path)
        out.write_text(self.to_script(header=header))
        return out

    # -- data access ---------------------------------------------------
    def resident(self, key: str) -> Resident:
        return self.store.get(key)

    @property
    def display_device(self) -> torch.device:
        """Where compositing happens, which is not always where computing does.

        The GUI thread repaints while a worker is inside a mode's preparation or
        a preproc tool, so these are two threads using one device. CUDA is fine
        with that -- work serialises on its stream -- and keeps the GPU here. MPS
        is not: two threads in Metal abort the process on an internal assertion
        that nothing can catch, which showed up as the viewer vanishing when you
        clicked the brain while motion correction was running.

        So on Metal the repaint moves to the CPU and the worker keeps the GPU to
        itself. The policy lives in the measured table rather than in an
        ``if device.type == "mps"`` here; see ``utils.py:_MPS_CPU_OPS``.
        """
        from fastfuncstuff.utils import cpu_if_mps

        return cpu_if_mps(self.store.device, "viewer_compose")

    def volume(self, key: str, index: int | None = None) -> np.ndarray:
        """One 3-D volume for display, from RAM when resident, disk when not."""
        res = self.store.get(key)
        idx = self.state.time_index if index is None else index
        idx = max(0, min(idx, res.info.n_volumes - 1))
        if res.array is not None:
            return res.array[..., idx]
        return self.store.preview(key, idx)

    def time_view(self, layer: Layer, hemi: str | None = None):
        """The layer's display transform (viewer.timeview), or None when it is shown raw.

        Built once per (layer, view, data generation) -- the drift fit is the
        expensive part, every frame after it is cheap. Per hemisphere for a
        surface result, whose series are vertices rather than voxels. None
        while a volume is not yet resident: it shows raw until it is.
        """
        from fastfuncstuff.viewer.timeview import build_time_view

        if (layer.time_detrend < 0 and not layer.time_psc) or not layer.time_linked:
            return None
        res = self.store.get(layer.key)
        key = (layer.key, hemi, layer.time_detrend, layer.time_psc, res.generation)
        cache = self._time_views
        hit = cache.get(key)
        if hit is not None:
            return hit
        sld = self.surface_layers.get(layer.key)
        if sld is not None:
            if hemi is None or hemi not in sld.parts:
                return None
            series = sld.parts[hemi].values
        else:
            if res.array is None:
                return None
            series = res.array
        for stale in [k for k in cache if k[0] == layer.key and k[1] == hemi]:
            del cache[stale]
        cache[key] = view = build_time_view(series, layer.time_detrend, layer.time_psc)
        return view

    def vertex_frame(self, layer: Layer, hemi: str, index: int) -> np.ndarray:
        """A surface result's values at ``index`` on ``hemi``, as viewed."""
        data = self.surface_layers[layer.key].parts[hemi]
        idx = max(0, min(int(index), data.values.shape[1] - 1))
        values = data.values[:, idx]
        view = self.time_view(layer, hemi)
        return values if view is None else view.frame(values, idx)

    def shown_volume(self, key: str, index: int | None = None) -> np.ndarray:
        """:meth:`volume` through the layer's time view: what is drawn, not what is stored.

        The display paths (slices, the 3-D window, the readout, AUTO's range)
        read this; analysis (DERIVE, edit snapping) keeps reading :meth:`volume`.
        """
        layer = self.state.layers.find(key)
        raw = self.volume(key, index)
        if layer is None or (layer.time_detrend < 0 and not layer.time_psc):
            return raw
        idx = self.state.time_index if index is None else index
        idx = max(0, min(int(idx), layer.n_volumes - 1))
        sld = self.surface_layers.get(key)
        if sld is not None:
            vol = np.zeros(raw.shape, np.float32)
            for hemi in sld.parts:
                sld.maps[hemi].paint(self.vertex_frame(layer, hemi, idx), out=vol)
            return vol
        view = self.time_view(layer)
        return raw if view is None else view.frame(raw, idx)

    def surface_sampler(self, key: str | None = None, mode: str = "linear"):
        """The image a surface edit snaps to: ``key``'s current volume, or the bottom layer's.

        ``mode`` (nearest / linear / cubic) is how a view reads it; edits snap
        with linear. Every mode shares the one cached copy of the voxels.

        Built on the CPU from the layer's own voxels and affine -- an edit
        reads a few thousand points, which a device round trip would only
        slow down -- and cached until the layer or its sub-brick changes.
        """
        from fastfuncstuff.surface.sampling import VolumeSampler

        layer = self.state.layers.find(key) if key else self.state.layers.base
        if layer is None:
            raise ValueError("no layer to snap surface edits to; load the anatomy first")
        idx = self.state.time_index if layer.time_linked else layer.volume_index
        cache = (layer.key, int(idx), np.asarray(layer.affine).tobytes())
        hit = getattr(self, "_surface_sampler", None)
        if hit is not None and hit[0] == cache:
            return hit[1].with_mode(mode)
        sampler = VolumeSampler(self.volume(layer.key, int(idx)), layer.affine)
        self._surface_sampler = (cache, sampler)
        return sampler.with_mode(mode)

    def display_volume(self, key: str, index: int | None = None) -> torch.Tensor | None:
        """The currently displayed sub-brick as a device tensor, cached.

        Cached per ``(layer, sub-brick)`` because a redraw asks for the same
        volume three times -- once per plane -- and the host-to-device copy
        measures 10 GB/s even on unified memory. Returns ``None`` when the
        layer's data has gone, so a repaint mid-eviction draws nothing rather
        than raising into the paint handler.
        """
        try:
            res = self.store.get(key)
        except KeyError:
            return None
        layer = self.state.layers.find(key)
        if layer is None:
            return None

        if index is not None:
            idx = index
        elif layer.time_linked:
            idx = self.state.time_index
        else:
            idx = layer.volume_index
        idx = max(0, min(int(idx), res.info.n_volumes - 1))

        cache_key = (key, idx, layer.time_detrend, layer.time_psc)
        hit = self._volume_cache.get(cache_key)
        if hit is not None:
            return hit

        tensor = self._device_frame(res, layer, idx)
        if tensor is None:
            try:
                arr = self.shown_volume(key, idx)
            except (KeyError, FileNotFoundError, ValueError):
                return None
            # A frame of the store's (X, Y, Z, T) array is a strided view;
            # _c_order gathers it with torch's threaded copy, 4x NumPy's.
            tensor = torch.from_numpy(_c_order(np.asarray(arr, dtype=np.float32))).to(
                self.display_device
            )
        # One sub-brick per layer is enough: panes share it, and holding more
        # would quietly duplicate what the residency store already owns.
        for stale in [k for k in self._volume_cache if k[0] == key]:
            del self._volume_cache[stale]
        self._volume_cache[cache_key] = tensor
        return tensor

    def _device_frame(self, res: Resident, layer: Layer, idx: int) -> torch.Tensor | None:
        """Frame ``idx`` sliced from a device copy something else already made; else None.

        Never promotes. Scrubbing used to promote the whole run so a frame was
        a device slice, which parked every 4-D layer you had looked at on the
        card -- about a third of free VRAM for just viewing. The host gather it
        avoided is 1.4 ms on a 100x100x72x300 run through torch's threaded
        copy (:meth:`display_volume`), so the trade was not worth it.
        """
        series = res.tensor
        if (
            series is None
            or res.info.n_volumes < 2
            or layer.time_detrend >= 0
            or layer.time_psc
            or series.device.type != self.display_device.type
        ):
            return None
        return series[..., idx].contiguous()

    def invalidate(self, key: str | None = None) -> None:
        """Drop cached device volumes for one layer, or all of them."""
        if key is None:
            self._volume_cache.clear()
            return
        for stale in [k for k in self._volume_cache if k[0] == key]:
            del self._volume_cache[stale]

    # -- ROIs ----------------------------------------------------------
    def roi_set(self, key: str) -> RoiSet | None:
        """The groups one ROI layer defines, or ``None`` if it defines none.

        Never blocks. A 4-D set of masks needs every frame, and the readout
        asks this on every crosshair move -- so an unloaded one answers
        ``None`` and answers properly once the inflate lands, rather than
        freezing the window for the seconds it takes.
        """
        layer = self.state.layers.find(key)
        if layer is None or not layer.roi:
            return None
        hit = self._roi_sets.get(key)
        if hit is not None:
            return hit
        try:
            res = self.store.get(key)
        except KeyError:
            return None
        if layer.n_volumes > 1:
            if res.array is None:
                return None
            built = rois_from_frames(
                res.array,
                name=layer.name,
                frame_names=layer.labels,
                source=f"file:{key}",
            )
        else:
            try:
                volume = self.volume(key, 0)
            except (KeyError, FileNotFoundError, ValueError):
                return None
            # A painted layer's own table, so a region's swatch in a list is
            # the colour it is painted in, and its name the one it was given.
            names = dict(self.painter.tables[key]) if self.painter.owns(key) else None
            built = rois_from_labels(
                volume,
                name=layer.name,
                names=names if names is not None else self._label_table(res),
                source=f"file:{key}",
            )
        self._roi_sets[key] = built
        return built

    def _label_table(self, res: Resident) -> dict:
        """Region names for a label volume: the header first, then a sidecar.

        The sidecar lookup happens here rather than in the header read because
        it is a guess by filename, and it is only a *safe* guess once something
        has established that this volume really is labels. ``run1.txt`` beside
        ``run1.nii.gz`` is a stimulus timing file far more often than a LUT.
        """
        from fastfuncstuff.io.labels import label_table

        if res.info.value_labels:
            return dict(res.info.value_labels)
        try:
            return label_table(res.path)
        except OSError:
            return {}

    def region_name(self, layer: Layer, value: float | None) -> str:
        """The region an ROI layer has at the crosshair, ``--`` outside every one.

        One answer for every readout. A painted layer names it from its label
        table and the value under the crosshair; a loaded one from its region
        set. Not the region set for a painted layer: every dab invalidates
        it, and rebuilding it is a scan of the whole volume per brush move.
        """
        if self.painter.owns(layer.key):
            return "--" if value is None else self.painter.name_at(layer.key, value)
        rois = self.roi_set(layer.key)
        found = rois.at(self.state.crosshair) if rois is not None else None
        return found.name if found else "--"

    def roi_palette(self, key: str, device: torch.device | None = None) -> torch.Tensor | None:
        """``(max label + 1, 3)`` colours in ``[0, 1]``, for the renderer.

        Indexed by label value, so drawing is one gather and the colour on
        screen is the same one the ROI list shows beside the region's name --
        there is no second palette to fall out of step.
        """
        where = device or self.display_device
        cache_key = (key, str(where))
        hit = self._roi_palettes.get(cache_key)
        if hit is not None:
            return hit
        if self.painter.owns(key):
            # By value from the layer's table: no scan of the volume, so a
            # newly painted label has its colour on the first dab.
            built = self.painter.palette(key, where)
            self._roi_palettes[cache_key] = built
            return built
        rois = self.roi_set(key)
        if rois is None or not len(rois):
            return None
        built = torch.as_tensor(rois.palette(), dtype=torch.float32, device=where) / 255.0
        self._roi_palettes[cache_key] = built
        return built

    # -- clusters ------------------------------------------------------
    def clustsim_table(self, key: str, nn: int, sidedness: str):
        """This dataset's own ClustSim table for one NN and sidedness.

        Read from the file the layer came from, and cached, because it is a
        header parse and the window asks on every threshold drag. ``None`` when
        the dataset carries none, which the table then reports rather than
        papering over.
        """
        cache = self._clustsim.get(key)
        if cache is None:
            from fastfuncstuff.stats.clustsim import read_clustsim_tables_from

            try:
                cache = read_clustsim_tables_from(self.store.get(key).path)
            except Exception:  # a header that will not parse simply has no tables
                cache = {}
            self._clustsim[key] = cache
        return cache.get((int(nn), sidedness))

    def clusterize(
        self,
        key: str | None = None,
        *,
        nn: int = 1,
        min_voxels: int = 1,
        mask: np.ndarray | None = None,
        alpha: float | None = None,
    ):
        """Cluster one layer at the threshold it is currently drawn with.

        ``alpha`` sets the minimum size from the dataset's ClustSim table (voxels,
        or mm^2 per hemisphere on a surface) in place of ``min_voxels``.

        The layer's own threshold, not one passed in: a table computed at a
        different cut does not describe the picture beside it, and two things
        on screen disagreeing is worse than either alone.
        """
        from fastfuncstuff.stats.fdr import stat_value_to_pvalue
        from fastfuncstuff.viewer.clusters import SIDEDNESS, clusterize

        layer = self.state.layers.find(key) if key else self.state.selected_layer()
        if layer is None:
            raise ValueError("no layer to clusterize")
        if layer.threshold <= 0:
            raise ValueError(f"{layer.name} has no threshold set; nothing to cluster")
        sld = self.surface_layers.get(layer.key)
        if sld is not None:
            # On the mesh, not the painted voxels: areas, mesh adjacency and the
            # bucket's own SurfClustSim table; painted back for every consumer.
            from fastfuncstuff.stats.fdr import stat_value_to_pvalue
            from fastfuncstuff.viewer.surfacelayers import surface_clusterize

            spec = layer.stat_spec()
            pthr = (
                None
                if spec is None
                else stat_value_to_pvalue(float(layer.threshold), spec[0], spec[1])
            )
            geometry = {
                h: (
                    self.surfaces.hemis[h].states["white"],
                    self.surfaces.hemis[h].states.get(
                        "pial", self.surfaces.hemis[h].states["white"]
                    ),
                    self.surfaces.hemis[h].faces,
                )
                for h in sld.parts
            }
            return layer, surface_clusterize(
                layer, sld, geometry, min_voxels=min_voxels, pthr=pthr, alpha=alpha
            )
        values = self.volume(layer.key, layer.volume_index)
        stat = self.volume(layer.key, layer.threshold_brick)

        # The p the threshold corresponds to is what indexes a ClustSim row, so
        # a bucket that does not say what test it is gets no corrected alpha --
        # rather than one read off whichever row happened to be first.
        pthr = None
        spec = layer.stat_spec()
        if spec is not None:
            pthr = stat_value_to_pvalue(float(layer.threshold), spec[0], spec[1])
        table = self.clustsim_table(layer.key, nn, SIDEDNESS[layer.sign_mode])

        # The determinant of the affine's rotation block, not the product of
        # three zooms: on an oblique dataset those differ, and cluster volumes
        # in mm3 are a number people put in papers.
        affine = np.asarray(layer.affine, dtype=float)
        return layer, clusterize(
            values,
            stat=stat,
            threshold=float(layer.threshold),
            sign_mode=layer.sign_mode,
            nn=nn,
            min_voxels=min_voxels,
            affine=affine,
            voxel_mm3=float(abs(np.linalg.det(affine[:3, :3]))),
            table=table,
            pthr=pthr,
            mask=mask,
            alpha=alpha,
        )

    # -- display masks -----------------------------------------------------
    def mask_from_layer(self, source_key: str, like: Layer) -> np.ndarray:
        """Where ``source_key`` has brain, as a boolean on ``like``'s grid.

        An ROI layer already is a mask, so it is used as drawn (nonzero); any
        other layer goes through ``processing/mask.py:automask`` on the
        sub-brick it is showing. Nearest-neighbour onto ``like``'s grid, since
        a mask interpolated is a mask with a fuzzy, made-up edge -- the anat's
        1 mm brain boundary on a 3 mm map lands where the map's voxels are.
        """
        source = self.state.layers.get(source_key)
        index = self.state.time_index if source.time_linked else source.volume_index
        cache_key = (source_key, int(index))
        native = self._source_masks.get(cache_key)
        if native is None:
            values = np.nan_to_num(np.asarray(self.volume(source_key, index), dtype=np.float32))
            if source.roi:
                native = values != 0
            else:
                from fastfuncstuff.processing.mask import automask

                native = (
                    automask(torch.as_tensor(values), dilate_extra=1).cpu().numpy().astype(bool)
                )
            for stale in [k for k in self._source_masks if k[0] == source_key]:
                del self._source_masks[stale]
            self._source_masks[cache_key] = native
        return resample_mask_nearest(native, source.affine, like.shape, like.affine)

    def set_display_mask(self, owner: str, key: str | None, keep: np.ndarray | None) -> None:
        """Hide one layer's voxels outside ``keep``, on behalf of ``owner``.

        Display only, like :meth:`resample_mode`: readouts, graphs and cluster
        tables still read every voxel. ``keep=None`` (or ``key=None``) drops
        whatever ``owner`` was masking, so a window points its mask at a new
        layer by setting it, and takes it away by closing.
        """
        before = self._display_masks.pop(owner, {})
        if key is not None and keep is not None:
            self._display_masks[owner] = {key: np.asarray(keep, dtype=bool)}
        for stale in set(before) | ({key} if key else set()):
            self._display_mask_tensors.pop(stale, None)

    def display_mask(self, key: str) -> torch.Tensor | None:
        """Every owner's mask on one layer, ANDed, as a float display tensor."""
        hit = self._display_mask_tensors.get(key)
        if hit is not None:
            return hit
        masks = [m[key] for m in self._display_masks.values() if key in m]
        if not masks:
            return None
        keep = np.logical_and.reduce(masks) if len(masks) > 1 else masks[0]
        tensor = torch.as_tensor(keep.astype(np.float32)).to(self.display_device)
        self._display_mask_tensors[key] = tensor
        return tensor

    def cluster_series(self, table, index: int) -> tuple[np.ndarray | None, str]:
        """Mean time course of one cluster, from a run on the same grid.

        The cluster's average, not its peak voxel's. The peak is by definition
        the most extreme voxel in the blob, so its time course is the one most
        selected for -- plotting it is the classic way to make an effect look
        larger than it is.

        ``(None, why)`` rather than blocking when no run is resident: the window
        asks on every row click, and says ``why`` in place of the trace -- a
        stat bucket is not a run, so a map loaded alone never has one.
        """
        if table is None:
            return None, "no cluster table"
        picked = np.asarray(table.labels) == int(index)
        if not picked.any():
            return None, "cluster has no voxels"
        same_grid = [ly for ly in self.graph_layers() if ly.shape == picked.shape]
        for layer in reversed(same_grid):
            resident = self.store.get(layer.key)
            if resident.array is None:
                continue
            return np.asarray(resident.array[picked].mean(0), dtype=np.float32), ""
        if same_grid:
            return None, "the run on this grid is still loading"
        return None, "no 4-D run on this grid to average -- load the time series"

    def install_rois(self, rois, *, name: str, source: str) -> str:
        """Adopt an ROI set as a layer, so it can be used like any other.

        The clusters a threshold just produced become an atlas the moment they
        are in the stack: the matrix can use them as nodes, the readout names
        them, and a seed can come from one. Nothing downstream has to learn
        what a cluster is.
        """
        key = self.state.layers.mint_key("R")
        self.store.adopt(key, rois.labels.astype(np.float32), name=name)
        shape = rois.shape
        grid_affine = self.state.grid.affine if self.state.grid is not None else np.eye(4)
        self.state.layers.add(
            Layer(
                key=key,
                name=name,
                path=name,
                shape=shape,
                n_volumes=1,
                affine=np.asarray(grid_affine, dtype=float),
                roi=True,
                source=source,
                range_lo=0.0,
                range_hi=float(max(rois.indices, default=1)),
            )
        )
        self._roi_sets[key] = rois
        return key

    def install_selection(
        self, source: str, mask: np.ndarray, *, like: Layer, name: str
    ) -> tuple[str, Aspect]:
        """Put a voxel selection in the stack as a mask layer, or update it.

        One layer per selecting window, keyed by ``source``, and replaced in
        place on every new selection -- a selection is a question being
        refined, and a stack that grows a layer per drag answers a different
        one.

        On creation it goes on top, is selected, and every other visible layer
        above the bottom of the stack is hidden: the selected voxels may be scattered
        across the brain, and a stat map drawn over them hides exactly where
        they went. Only on creation, so an overlay turned back on by hand stays
        on while the selection is refined.
        """
        values = np.asarray(mask, dtype=np.float32)
        existing = self.state.layers.find_by_source(source)
        key = existing.key if existing is not None else self.state.layers.mint_key("S")
        self.store.adopt(key, values, name=name)
        self.invalidate(key)
        if existing is not None:
            self.state.layers.update(key, name=name, path=f"<{name}>")
            return key, Aspect.LAYERS | Aspect.SLICES

        base = self.state.layers.base
        self._add_on_top(
            hide=[ly.key for ly in self.state.layers if base is None or ly.key != base.key],
            layer=Layer(
                key=key,
                name=name,
                path=f"<{name}>",
                shape=tuple(int(v) for v in values.shape[:3]),
                n_volumes=1,
                affine=np.asarray(like.affine, dtype=float),
                colormap="red",
                sign_mode=SignMode.POS,
                range_lo=0.0,
                range_hi=1.0,
                threshold=0.5,
                source=source,
            ),
        )
        return key, Aspect.LAYERS | Aspect.SLICES | Aspect.GRID

    def save_layer(self, key: str, path: str | Path) -> Path:
        """Write one layer's voxels to NIfTI, on its own grid.

        What makes a selection, a derived run or an adopted cluster map a
        result rather than something that only existed while the viewer was
        open. Sub-brick labels go with it, so a bucket saved back out still
        names its contrasts.
        """
        from fastfuncstuff.io.afni import save_nifti

        layer = self.state.layers.get(key)
        if layer.colour:
            return self.colour_sources[key].save(path)
        if self.painter.owns(key):
            return self.painter.save(key, path)
        data = np.asarray(self.store.ensure_ram(key), dtype=np.float32)
        if data.ndim == 4 and data.shape[3] == 1:
            data = data[..., 0]
        out = Path(path)
        save_nifti(
            data,
            output_path=out,
            affine=np.asarray(layer.affine, dtype=float),
            brick_labels=list(layer.labels) or None,
        )
        return out

    def forget_rois(self, key: str | None = None) -> None:
        """Drop cached ROI descriptions, for one layer or all of them."""
        if key is None:
            self._roi_sets.clear()
            self._roi_palettes.clear()
            return
        self._roi_sets.pop(key, None)
        for stale in [k for k in self._roi_palettes if k[0] == key]:
            del self._roi_palettes[stale]

    def roi_layers(self) -> list[Layer]:
        """Every layer that defines ROIs, bottom-up."""
        return [ly for ly in self.state.layers if ly.roi]

    def roi_at(self, ijk: tuple[int, int, int] | None = None):
        """``(layer, roi)`` for the topmost ROI layer covering a voxel."""
        where = ijk if ijk is not None else self.state.crosshair
        for layer in reversed(self.roi_layers()):
            rois = self.roi_set(layer.key)
            if rois is None:
                continue
            found = rois.at(where)
            if found is not None:
                return layer, found
        return None

    def layer_voxel(self, affine: np.ndarray, ijk: tuple[int, int, int]) -> tuple[int, int, int]:
        """A display-grid voxel as the nearest voxel of a layer on ``affine``.

        Through millimetres, because the display grid belongs to the bottom of
        the stack and nothing else is obliged to share it. **Anything that
        reads a layer's array at "where the crosshair is" has to come through
        here.** Indexing a 3 mm run with an index into a 1 mm anatomy is out of
        bounds near the far edge -- which reads as "the graph stopped working"
        -- and, worse, *in* bounds near the origin, where it silently returns a
        voxel three times too close to the corner.

        Nearest rather than interpolated: a plotted time course should be one
        the scanner measured.
        """
        grid = self.state.grid
        if grid is None:
            return ijk
        mm = grid.ijk_to_mm(ijk)
        v = np.linalg.inv(np.asarray(affine, dtype=float)) @ np.array([*mm, 1.0])
        return (int(round(float(v[0]))), int(round(float(v[1]))), int(round(float(v[2]))))

    def timeseries(self, key: str, ijk: tuple[int, int, int] | None = None) -> np.ndarray:
        """The time course at a display-grid voxel, empty if not yet resident.

        Returns empty rather than blocking: the graph pane asks on every
        crosshair move, and waiting seconds for an inflate would be exactly the
        lock-up this design exists to avoid.
        """
        res = self.store.get(key)
        if res.array is None:
            return np.empty(0, dtype=np.float32)
        layer = self.state.layers.find(key)
        ijk = self.state.crosshair if ijk is None else ijk
        i, j, k = ijk if layer is None else self.layer_voxel(layer.affine, ijk)
        nx, ny, nz = res.array.shape[:3]
        if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
            return np.empty(0, dtype=np.float32)
        return np.asarray(res.array[i, j, k, :], dtype=np.float32)

    def vertex_voxels(self, hemi: str, vertices: np.ndarray, depth: float) -> list:
        """Display-grid voxels under vertices, at a cortical depth (0 white .. 1 pial)."""
        grid = self.state.grid
        h = self.surfaces.hemis.get(hemi)
        if grid is None or h is None:
            return []
        white = np.asarray(h.states["white"], np.float64)[vertices]
        pial = np.asarray(h.states.get("pial", h.states["white"]), np.float64)[vertices]
        mm = white + float(depth) * (pial - white)
        inv = np.linalg.inv(np.asarray(grid.affine, np.float64))
        ijk = np.rint(mm @ inv[:3, :3].T + inv[:3, 3]).astype(int)
        return [grid.clamp(tuple(v)) for v in ijk]

    def vertex_timeseries(
        self, layer: Layer, hemi: str, vertices: np.ndarray, depth: float
    ) -> np.ndarray:
        """A time-linked layer's courses at vertices, ``(len(vertices), T)``.

        A surface result is read off its own vertices. A volume is read where
        the 3-D window samples it: at ``depth`` between white and pial, at the
        display voxel there -- so a vertex's line and its colour agree. Rows
        are empty-width (T = 0) while the volume is not resident.
        """
        sld = self.surface_layers.get(layer.key)
        if sld is not None:
            data = sld.parts.get(hemi)
            if data is None:
                return np.empty((len(vertices), 0), np.float32)
            return np.asarray(data.values[vertices], np.float32)
        rows = [
            self.timeseries(layer.key, ijk) for ijk in self.vertex_voxels(hemi, vertices, depth)
        ]
        if not rows or any(r.size == 0 for r in rows):
            return np.empty((len(vertices), 0), np.float32)
        return np.stack(rows)

    def vertex_tints(
        self, hemi: str, vertices: np.ndarray, depth: float
    ) -> list[tuple[float, float, float] | None] | None:
        """The overlay's colour at each vertex, ``None`` where cut; None without one.

        The surface result the 3-D window shows when there is one, else the
        volume overlay at the vertex's voxel, as the graph window tints.
        """
        shown = self.surface_vertex_rgba(hemi)
        if shown is not None:
            rgba = np.asarray(shown[0])[vertices]
            return [
                (px[0] / 255.0, px[1] / 255.0, px[2] / 255.0) if px[3] > 0 else None
                for px in rgba.astype(float)
            ]
        return self.overlay_colors(self.vertex_voxels(hemi, vertices, depth))

    def close(self) -> None:
        self.store.shutdown()
