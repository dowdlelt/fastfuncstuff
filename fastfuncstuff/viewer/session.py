"""A viewer session: state, residency, and the command bus wired together.

This is what a UI, a CLI or a test drives. Nothing above this layer touches
:class:`ViewerState` directly -- everything goes through
:meth:`ViewerSession.do`, which is what keeps the recording honest.
"""

from __future__ import annotations

from collections.abc import Callable
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
from fastfuncstuff.viewer.residency import Resident, VolumeStore
from fastfuncstuff.viewer.rois import RoiSet, looks_like_labels, rois_from_frames, rois_from_labels
from fastfuncstuff.viewer.state import Plane, ViewerState
from fastfuncstuff.viewer.viewports import ViewKind
from fastfuncstuff.viewer.vocab import AddLayer, CloseView, OpenView, SetVolume, install

#: Percentiles used to auto-range a layer. AFNI's autorange takes the maximum,
#: which one bright voxel is enough to ruin; percentiles are what make a map
#: readable without anyone reaching for a slider.
AUTORANGE_PERCENTILES = (2.0, 98.0)

#: Where a freshly-picked overlay starts its threshold. High enough that the
#: underlay reads through the noise -- a quarter of the brain tinted is not a
#: view of anything -- and low enough that real structure is already on screen
#: before anyone touches the slider.
OVERLAY_START_PERCENTILE = 90.0


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
        self.store = store or VolumeStore(device=device)
        self.bus = install(
            CommandBus(self.state, record=record), open_layer=self._open, session=self
        )
        self.catalog: list[CatalogEntry] = []
        self.catalog_dir: Path | None = None
        self.mode: Mode = registry.get("plain")()
        #: Every mode this session has been in, by name. Switching back resumes
        #: the same instance -- its folder, its component, its unsaved labels --
        #: because a mode left to go and look at something else is a mode you
        #: are coming back to.
        self._modes: dict[str, Mode] = {self.mode.name: self.mode}
        self.mode.attach(self)
        self._volume_cache: dict[tuple[str, int], torch.Tensor] = {}
        #: Built on demand from a layer's voxels and kept until the flag or the
        #: layer changes. Describing an atlas is one pass over the volume, and
        #: the crosshair readout asks which region it is on every move.
        self._roi_sets: dict[str, RoiSet] = {}
        self._roi_palettes: dict[tuple[str, str], torch.Tensor] = {}
        self._clustsim: dict[str, dict] = {}
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
        if layer.n_volumes == 1 and looks_like_labels(preview):
            layer = layer.with_(roi=True)
        if layer.n_volumes > 1:
            self.store.load_async(key, on_done=self._on_loaded)
        return layer

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

    # -- dispatch ------------------------------------------------------
    def _on_command(self, cmd: Command, dirty: Aspect) -> None:
        if dirty & (Aspect.TIME | Aspect.LAYERS):
            self.invalidate()
        elif isinstance(cmd, SetVolume):
            self.invalidate(cmd.key)
        # The mode reacts after the state has settled, so an InstaCorr seed sees
        # the seed already moved. Its own dirty aspects are folded into what the
        # dispatch reports, which is how a recomputed overlay reaches the panes.
        self._mode_dirty = self.mode.on_command(cmd, dirty)

    def do(self, cmd: Command) -> Aspect:
        self._mode_dirty = Aspect.NOTHING
        dirty = self.bus.dispatch(cmd)
        return dirty | self._mode_dirty

    # -- catalog -------------------------------------------------------
    def read_directory(self, directory: str | Path, *, recursive: bool = False) -> list:
        """Populate the catalog the pickers draw from."""
        self.catalog = catalog_mod.scan(directory, recursive=recursive)
        self.catalog_dir = Path(directory)
        return self.catalog

    def apply_overlay_defaults(self, key: str) -> None:
        """Give a freshly-picked overlay a state you can actually see through.

        A layer loaded with threshold 0 and no alpha is opaque everywhere, so
        dropping one on an anatomical hides it completely -- which defeats the
        first thing anyone does, checking that the two line up. Starting at a
        high percentile means the overlay reads as structure over anatomy from
        the moment it lands, and the slider takes it from there.
        """
        layer = self.state.layers.find(key)
        if layer is None or layer.is_computed:
            return
        try:
            values = self.volume(key, 0)
        except (KeyError, FileNotFoundError, ValueError):
            return
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return
        threshold = float(np.percentile(np.abs(finite), OVERLAY_START_PERCENTILE))
        # Alpha stays off: a hard threshold is what a stats map is read at, and
        # a fade makes "which voxels survive" something you have to squint at.
        self.state.layers.update(
            key, threshold=threshold, **self.overlay_look(key, 0, colormap="hot")
        )

    def overlay_look(self, key: str, index: int, *, colormap: str) -> dict[str, object]:
        """Colour scale for one sub-brick of an overlay: range, and hot or red-blue.

        A signed sub-brick gets a symmetric range and a diverging map, so zero
        sits in the middle of the bar and a negative effect is as visible as a
        positive one. A one-signed one runs from zero. The colormap is only
        swapped between the two defaults -- a map someone chose stays.

        Percentiles over the non-zero voxels: a bucket is zero outside the
        mask, and counting that zero puts the 98th percentile near nothing.
        """
        try:
            values = self.volume(key, index)
        except (KeyError, FileNotFoundError, ValueError):
            return {}
        finite = values[np.isfinite(values) & (values != 0)]
        if finite.size == 0:
            return {}
        top = float(np.percentile(np.abs(finite), AUTORANGE_PERCENTILES[1])) or 1.0
        negative, positive = bool((finite < 0).any()), bool((finite > 0).any())
        out: dict[str, object] = {}
        if negative and positive:
            out.update(range_lo=-top, range_hi=top)
            signed = True
        elif negative:
            out.update(range_lo=-top, range_hi=0.0)
            signed = True
        else:
            out.update(range_lo=0.0, range_hi=top)
            signed = False
        if colormap in ("hot", "redblue"):
            out["colormap"] = "redblue" if signed else "hot"
        return out

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

    def carpet_overlay(self, source: Layer) -> Layer | None:
        """Which layer labels a carpet's rows.

        The topmost visible layer that is not the run being drawn -- not
        "overlay-prime". In a stack of anat, run and stats the thing worth
        drawing beside the rows is the stats map on top, and overlay-prime is
        the run itself.
        """
        for layer in reversed(list(self.state.layers)):
            if layer.key != source.key and layer.visible:
                return layer
        return None

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
        to the filesystem each time. Loading through :meth:`load_design` is
        what re-reads a file that changed.
        """
        if path in self._design_errors:
            return None
        resolved = self._design_paths.get(path)
        if resolved is not None and resolved in self.designs:
            return self.designs[resolved]
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
    def _add_on_top(self, layer: Layer) -> None:
        """Push a made layer on top, selected, with other overlays hidden.

        Hidden only here, on creation: whatever it was computed from -- a run,
        usually -- drawn under a correlation map is noise over the anatomy. A
        layer switched back on by hand stays on while the output is refined.
        """
        base = self.state.layers.base
        for other in list(self.state.layers):
            if other.visible and (base is None or other.key != base.key):
                self.state.layers.update(other.key, visible=False)
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
        existing = self.state.layers.find_by_source(source)
        key = existing.key if existing is not None else self.state.layers.mint_key("M")
        self.store.adopt(key, overlay.values, name=overlay.name)
        self.invalidate(key)

        lo, hi = overlay.display_range or derive_range(overlay.values)
        if existing is not None:
            # The name is identity -- showing "IC 0" while displaying IC 4 is a
            # lie. Range and threshold deliberately do NOT follow: stepping
            # through components at a threshold you set is how they get
            # reviewed, and resetting it on every step would undo the gesture.
            if existing.name != overlay.name:
                self.state.layers.update(key, name=overlay.name)
        else:
            self._add_on_top(
                Layer(
                    key=key,
                    name=overlay.name,
                    path=f"<{overlay.name}>",
                    shape=tuple(int(v) for v in overlay.values.shape[:3]),
                    n_volumes=1,
                    affine=np.asarray(overlay.affine, dtype=float),
                    colormap=overlay.colormap,
                    range_lo=lo,
                    range_hi=hi,
                    threshold=overlay.threshold or 0.0,
                    source=source,
                )
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
        values = np.array(self.store.ensure_ram(live.key), dtype=np.float32, copy=True)
        stem = mode.output_name()
        taken = [ly for ly in self.state.layers if ly.source == f"kept:{mode.layer_source}"]
        number = 1 + max(
            (int(n) for ly in taken if (n := ly.name[len(stem) + 1 :].split(" ")[0]).isdigit()),
            default=0,
        )
        detail = live.name[len(stem) :].strip()
        name = f"{stem}_{number}" + (f" {detail}" if detail else "")
        key = self.state.layers.mint_key("K")
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
        for stale in [k for k in self._threshold_scales if k[0] == key]:
            del self._threshold_scales[stale]
        self.store.close(key)

    def run_script(self, text: str) -> Aspect:
        return self.bus.run_script(text)

    def to_script(self, *, header: str | None = None) -> str:
        return self.bus.to_script(header=header)

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

        cache_key = (key, idx)
        hit = self._volume_cache.get(cache_key)
        if hit is not None:
            return hit

        try:
            arr = self.volume(key, idx)
        except (KeyError, FileNotFoundError, ValueError):
            return None
        tensor = torch.as_tensor(np.ascontiguousarray(arr), dtype=torch.float32).to(
            self.display_device
        )
        # One sub-brick per layer is enough: panes share it, and holding more
        # would quietly duplicate what the residency store already owns.
        for stale in [k for k in self._volume_cache if k[0] == key]:
            del self._volume_cache[stale]
        self._volume_cache[cache_key] = tensor
        return tensor

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
            built = rois_from_labels(
                volume,
                name=layer.name,
                names=self._label_table(res),
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

    def roi_palette(self, key: str, device: torch.device | None = None) -> torch.Tensor | None:
        """``(max label + 1, 3)`` colours in ``[0, 1]``, for the renderer.

        Indexed by label value, so drawing is one gather and the colour on
        screen is the same one the ROI list shows beside the region's name --
        there is no second palette to fall out of step.
        """
        rois = self.roi_set(key)
        if rois is None or not len(rois):
            return None
        where = device or self.display_device
        cache_key = (key, str(where))
        hit = self._roi_palettes.get(cache_key)
        if hit is not None:
            return hit
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
            from fastfuncstuff.stats.clustsim import read_clustsim_tables

            try:
                from fastfuncstuff.io.headers import read_nifti_header

                cache = read_clustsim_tables(read_nifti_header(self.store.get(key).path))
            except Exception:  # a header that will not parse simply has no tables
                cache = {}
            self._clustsim[key] = cache
        return cache.get((int(nn), sidedness))

    def clusterize(self, key: str | None = None, *, nn: int = 1, min_voxels: int = 1):
        """Cluster one layer at the threshold it is currently drawn with.

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
        )

    def cluster_series(self, table, index: int) -> np.ndarray | None:
        """Mean time course of one cluster, from a run on the same grid.

        The cluster's average, not its peak voxel's. The peak is by definition
        the most extreme voxel in the blob, so its time course is the one most
        selected for -- plotting it is the classic way to make an effect look
        larger than it is.

        ``None`` rather than blocking when no run is resident: the window asks
        on every row click.
        """
        if table is None:
            return None
        picked = np.asarray(table.labels) == int(index)
        if not picked.any():
            return None
        for layer in reversed(self.graph_layers()):
            if layer.shape != picked.shape:
                continue
            resident = self.store.get(layer.key)
            if resident.array is None:
                continue
            return np.asarray(resident.array[picked].mean(0), dtype=np.float32)
        return None

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
        above the underlay is hidden: the selected voxels may be scattered
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

        self._add_on_top(
            Layer(
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
            )
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

    def timeseries(self, key: str, ijk: tuple[int, int, int] | None = None) -> np.ndarray:
        """The time course at a voxel, or an empty array if not yet resident.

        Returns empty rather than blocking: the graph pane asks on every
        crosshair move, and waiting seconds for an inflate would be exactly the
        lock-up this design exists to avoid.
        """
        res = self.store.get(key)
        if res.array is None:
            return np.empty(0, dtype=np.float32)
        i, j, k = ijk if ijk is not None else self.state.crosshair
        nx, ny, nz = res.array.shape[:3]
        if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
            return np.empty(0, dtype=np.float32)
        return np.asarray(res.array[i, j, k, :], dtype=np.float32)

    def close(self) -> None:
        self.store.shutdown()
