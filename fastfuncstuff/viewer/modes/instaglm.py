"""InstaGLM: the model as something you argue with.

One run, one events file, and then every knob that a GLM actually turns on --
the HRF's shape, the drift order, a motion file, its derivatives, principal
components off the noise pool. Each one refits and redraws, and the point is
never the final map. It is the *difference* between two of them: step polort up
and watch the drift walk out of the residual, tick the motion file on and watch
the spikes go, drag the HRF peak a second later and watch a beta map brighten
or dim. That derivative with respect to the model is the thing a batch GLM can
never show, and the thing everyone has to learn.

The split between the halves is the framework's, and it lands neatly here. The
slow, cacheable work -- masking the run and gathering it into voxels-by-time,
then solving the design against all of it -- is :meth:`prepare`, on a worker
with a progress bar. The fast half, :meth:`compute`, only *picks* which of the
maps already computed to colour the brain with. So changing the design is a
refit and changing the view is instant, which is exactly the distinction a
learner should feel in their fingers.

Every column is pickable, not just the conditions. That is deliberate: "how much
percent signal change does roll carry, and where" is as good a question as any
about faces, and the answer is a map. See ``viewer/instaglm.py`` for why that
required its own fit rather than ``glm/core.py:fit_glm``.
"""

from __future__ import annotations

import numpy as np
import torch

from fastfuncstuff.viewer import instaglm as engine
from fastfuncstuff.viewer.commands import Aspect
from fastfuncstuff.viewer.modes.base import (
    ActionControl,
    BoolControl,
    ChoiceControl,
    ComputedOverlay,
    Control,
    FloatControl,
    IntControl,
    Mode,
    OverlayKind,
    PathControl,
    ProgressFn,
    Trace,
    mode,
)

#: How each map should be drawn, since they are not all signed and not all
#: bounded. ``(colormap, kind, range, threshold)``; a ``None`` range is derived
#: from the data the way any other computed overlay's is.
PRESENTATION: dict[str, tuple[str, OverlayKind, tuple[float, float] | None, float]] = {
    "beta": ("redblue", OverlayKind.VALUE, None, 0.0),
    "t": ("redblue", OverlayKind.STATISTIC, None, 2.5),
    "R2": ("hot", OverlayKind.VALUE, (0.0, 1.0), 0.05),
    "unique R2": ("hot", OverlayKind.VALUE, (0.0, 0.5), 0.02),
    "task R2": ("hot", OverlayKind.VALUE, (0.0, 0.5), 0.02),
    "task F": ("hot", OverlayKind.STATISTIC, None, 3.0),
    "resid sd": ("hot", OverlayKind.VALUE, None, 0.0),
}

#: Lines the mode contributes at every graphed voxel, and what each is for.
LINE_HELP = {
    "data": "the measurement, as it came off disk",
    "signal": "the measurement with the nuisance fit taken out of it",
    "fit": "what the model says the signal should be",
    "resid": "what the model did not account for",
    "column": "the selected column's own contribution",
}


@mode
class InstaGLMMode(Mode):
    name = "instaglm"
    label = "InstaGLM"
    tag = "IGLM"
    overlay_kind = OverlayKind.VALUE

    def __init__(self) -> None:
        self._prepared: engine.Prepared | None = None
        self._fit: engine.Fit | None = None
        self._source_key: str | None = None
        self._source_name = ""
        self._tr = 0.0
        self._hrf: np.ndarray | None = None
        self._hrf_dt = 0.1
        self._message = ""
        super().__init__()

    # -- declaration ---------------------------------------------------
    def controls(self) -> tuple[Control, ...]:
        # Built per call rather than fixed, so the column picker offers the
        # model that is actually loaded. The panel is rebuilt on LAYERS, which
        # is exactly what a refit dirties.
        return (
            PathControl(
                name="events",
                label="events",
                filter="events (*.tsv *.1D *.txt);;All (*)",
                help="A BIDS *_events.tsv, or an AFNI timing file as a single condition. "
                "Without one the model is drift and nuisance only, which is still worth "
                "looking at.",
            ),
            ChoiceControl(
                name="basis",
                label="HRF",
                choices=("spmg1", "spmg2", "spmg3", "library", "custom"),
                default="spmg1",
                help="spmg1/2/3 add the time and dispersion derivatives; library steps the "
                "20 canonical curves ffs fits with; custom is the three sliders below.",
            ),
            IntControl(
                name="hrf_index",
                label="library #",
                lo=0,
                hi=max(engine.library_size() - 1, 0),
                default=0,
                help="Which curve of the canonical library. Only read when HRF is 'library'.",
            ),
            FloatControl(
                name="peak",
                label="peak",
                lo=2.0,
                hi=12.0,
                default=6.0,
                step=0.25,
                unit=" s",
                help="Time to peak of the custom double gamma. Only read when HRF is 'custom'.",
            ),
            FloatControl(
                name="width",
                label="width",
                lo=0.4,
                hi=3.0,
                default=1.0,
                step=0.05,
                help="Dispersion of the custom double gamma. Only read when HRF is 'custom'.",
            ),
            FloatControl(
                name="undershoot",
                label="undershoot",
                lo=0.0,
                hi=0.6,
                default=0.167,
                step=0.01,
                help="Depth of the custom double gamma's undershoot, as a fraction of the "
                "peak. Only read when HRF is 'custom'.",
            ),
            IntControl(
                name="polort",
                label="polort",
                lo=-1,
                hi=9,
                default=2,
                help="Legendre drift order; -1 removes the baseline entirely, which is worth "
                "doing once to see what it was holding up.",
            ),
            PathControl(
                name="ortvec",
                label="ortvec",
                filter="1D / xmat (*.1D *.txt);;All (*)",
                help="Extra regressors, one column each -- a motion file, a respiration "
                "trace. Every column is pickable as a map of its own.",
            ),
            BoolControl(
                name="ort_deriv",
                label="+ deriv",
                default=False,
                help="Also fit each ortvec column's backward difference. A motion column "
                "removes signal that tracks where the head is; its derivative removes "
                "signal that tracks the head moving, which is usually the bigger one.",
            ),
            IntControl(
                name="pcs",
                label="noise PCs",
                lo=0,
                hi=10,
                default=0,
                help="Principal components of the noise pool -- bright voxels the task "
                "explains no better than chance -- added as regressors, GLMdenoise style.",
            ),
            ChoiceControl(
                name="show",
                label="show",
                choices=engine.MAPS,
                default="beta",
                help="What to colour the brain with. Changing this does not refit.",
            ),
            ChoiceControl(
                name="column",
                label="column",
                choices=self._column_choices(),
                default=self._column_choices()[0],
                help="Which regressor the beta, t and unique-R2 maps are of.",
            ),
            ChoiceControl(
                name="psc",
                label="units",
                choices=engine.PSC_MODES,
                default="swing",
                help="'swing' scales a beta by its own regressor's excursion, so a "
                "condition and a motion column are comparable on one colour bar. "
                "'per unit' is the plain beta/mean reading, in the regressor's own units.",
            ),
        )

    def _column_choices(self) -> tuple[str, ...]:
        if self._fit is None:
            return ("--",)
        return self._fit.model.labels or ("--",)

    def actions(self) -> tuple[ActionControl, ...]:
        return (
            ActionControl(
                name="fit",
                label="fit",
                help="Refit from scratch, rereading the events and ortvec files from disk.",
            ),
            *super().actions(),
        )

    def preparation_params(self) -> frozenset[str]:
        """Everything that changes the model. The three that only change the
        *view* -- which map, which column, which units -- are deliberately not
        here, so picking a different beta to look at costs a reduction and not
        a refit."""
        return frozenset(
            {
                "events",
                "basis",
                "hrf_index",
                "peak",
                "width",
                "undershoot",
                "polort",
                "ortvec",
                "ort_deriv",
                "pcs",
            }
        )

    def panel_names(self) -> tuple[str, ...]:
        """The HRF in a window of its own.

        This is what closes the loop. Dragging the peak slider moves a curve you
        can see at the same time as the map it is redrawing, which is the whole
        difference between "the HRF matters" as a sentence and as an experience.
        """
        return ("hrf",)

    # -- the run -------------------------------------------------------
    def source_layer(self):
        """The selected layer if it is a run, else the topmost run."""
        if self.session is None:
            return None
        state = self.session.state
        chosen = state.layers.find(state.selected) if state.selected else None
        if chosen is not None and chosen.time_linked and chosen.n_volumes > 1:
            return chosen
        for layer in reversed(list(state.layers)):
            if layer.time_linked and layer.n_volumes > 1:
                return layer
        return None

    def input_layer_key(self) -> str | None:
        return self._source_key

    def attach(self, session) -> None:
        super().attach(session)
        self._dirty = True

    def detach(self) -> None:
        """Free the gathered array; it is the same gigabyte InstaCorr holds.

        The fit goes with it -- it is voxel sized too -- and the parameters
        stay, so coming back costs one gather and one solve and nothing else.
        """
        self._prepared = None
        self._fit = None
        self._source_key = None
        self._dirty = True
        super().detach()

    def action(self, name: str, progress: ProgressFn | None = None) -> Aspect:
        if name == "fit":
            # A full rebuild, source included: the events file may have been
            # edited on disk since it was read, and "press fit again" is the
            # gesture people expect to pick that up.
            self._prepared = None
            self._fit = None
            self.invalidate()
            return self.refresh(progress)
        return super().action(name, progress)

    # -- the slow half -------------------------------------------------
    def prepare(self, progress: ProgressFn | None = None) -> bool:
        """Gather the run once, then solve the current design against it."""
        if not self._dirty and self._fit is not None:
            return True
        if self.session is None:
            return False
        device = self.session.store.device

        if self._prepared is None and not self._gather(device, progress):
            return False
        assert self._prepared is not None

        try:
            model = self._build_model(device, progress)
        except (OSError, ValueError, IndexError) as exc:
            self._message = str(exc)
            self._fit = None
            self._dirty = False
            return False

        if progress is not None:
            progress(0.6, f"fitting {model.n_columns} regressors")
        self._fit = engine.fit_model(self._prepared, model, device=device, progress=progress)
        self._message = f"{model.note}, {model.n_columns} columns, {self._fit.dof} dof"
        if model.n_columns > self._fit.rank:
            # Not an error -- polort 0 beside a constant ortvec is one click
            # away and the minimum-norm answer is still drawn -- but it is the
            # explanation for two columns whose maps suddenly look halved.
            self._message += f"  ·  rank {self._fit.rank}, collinear"
        self._dirty = False
        if progress is not None:
            progress(1.0, "fitted")
        return True

    def _gather(self, device: torch.device, progress: ProgressFn | None) -> bool:
        assert self.session is not None
        layer = self.source_layer()
        if layer is None:
            self._message = "needs a 4-D run"
            return False
        # A missing TR is not refused here. A drift-only model does not need
        # one, and refusing to fit anything because a header is thin would
        # block the very first useful thing this mode does. Onsets are in
        # seconds, so ``read_events`` asks for the TR when it is actually
        # needed, and says so in those terms.
        tr = float(self.session.store.get(layer.key).info.tr)
        data = self.session.store.ensure_ram(layer.key)
        self._prepared = engine.prepare(
            data,
            affine=np.asarray(layer.affine, dtype=float),
            tr=tr,
            device=device,
            progress=progress,
        )
        self._source_key = layer.key
        self._source_name = layer.name
        self._tr = tr
        return True

    def _build_model(self, device: torch.device, progress: ProgressFn | None) -> engine.Model:
        assert self._prepared is not None
        n_time, tr = self._prepared.n_time, self._prepared.tr

        task, task_labels = None, []
        events_path = str(self.params.get("events") or "").strip()
        if events_path:
            if progress is not None:
                progress(0.15, "events")
            events = engine.read_events(events_path, n_time=n_time, tr=tr)
            curves, suffixes = self._hrf_curves(tr, device)
            task, task_labels = engine.task_columns(
                events,
                n_time=n_time,
                tr=tr,
                curves=curves,
                suffixes=suffixes,
                device=device,
            )

        ort, ort_labels = None, []
        ortvec_path = str(self.params.get("ortvec") or "").strip()
        if ortvec_path:
            from fastfuncstuff.viewer.derive import read_nuisance

            # Through derive's reader, so an xmat contributes its ColumnGroups
            # nuisance and a plain 1D file contributes all of itself -- the same
            # rule DERIVE applies, rather than a second opinion about what
            # counts as a regressor file.
            read = read_nuisance(ortvec_path, n_time=n_time, polort=-1)
            ort, ort_labels = read.columns, list(read.labels)

        polort = int(self.params.get("polort", 2))
        deriv = bool(self.params.get("ort_deriv", False))
        n_pcs = int(self.params.get("pcs", 0))

        pcs = None
        if n_pcs > 0:
            if progress is not None:
                progress(0.35, f"{n_pcs} noise PCs")
            # The pool is chosen against the model *without* the PCs, which is
            # the only order that makes sense: a component picked using itself
            # as a regressor would be selected for explaining what it is about
            # to be asked to explain.
            base = engine.build_model(
                n_time=n_time,
                tr=tr,
                task=task,
                task_labels=task_labels,
                polort=polort,
                ort=ort,
                ort_labels=ort_labels,
                ort_derivatives=deriv,
            )
            first = engine.fit_model(self._prepared, base, device=device)
            pool = engine.noise_pool(first)
            pcs, _ratios = engine.noise_pcs(self._prepared, pool, base.matrix, n_pcs, device=device)

        return engine.build_model(
            n_time=n_time,
            tr=tr,
            task=task,
            task_labels=task_labels,
            polort=polort,
            ort=ort,
            ort_labels=ort_labels,
            ort_derivatives=deriv,
            pcs=pcs,
        )

    def _hrf_curves(self, tr: float, device: torch.device) -> tuple[torch.Tensor, tuple[str, ...]]:
        from fastfuncstuff.design.matrices import commensurate_microtime_dt

        dt = commensurate_microtime_dt(tr)
        curves, suffixes = engine.hrf_bases(
            str(self.params.get("basis") or "spmg1"),
            microtime_dt=dt,
            index=int(self.params.get("hrf_index", 0)),
            delay=float(self.params.get("peak", 6.0)),
            dispersion=float(self.params.get("width", 1.0)),
            ratio=float(self.params.get("undershoot", 0.167)),
            device=device,
        )
        # Kept for the HRF panel, which draws the curve that was actually used
        # rather than rebuilding one and hoping it matches. Scaled by the
        # anchor's peak, never per curve: a derivative basis is not an
        # independent regressor, and dividing each curve by its own peak would
        # draw the time derivative 2.56x too tall relative to the shape it is
        # the derivative of. The design builder anchors for the same reason,
        # and a panel that disagreed with it would be worse than none.
        curves_np = curves.detach().cpu().numpy().astype(np.float32)
        anchor = float(np.abs(curves_np[0]).max())
        self._hrf = curves_np / anchor if anchor > 0 else curves_np
        self._hrf_dt = dt
        return curves, suffixes

    # -- the fast half -------------------------------------------------
    def _column_index(self) -> int:
        if self._fit is None:
            return 0
        chosen = str(self.params.get("column") or "")
        found = self._fit.model.index_of(chosen)
        return found if found is not None else 0

    def compute(self) -> ComputedOverlay | None:
        if self._fit is None:
            return None
        kind = str(self.params.get("show") or "beta")
        column = self._column_index()
        colormap, overlay_kind, span, threshold = PRESENTATION.get(
            kind, ("redblue", OverlayKind.VALUE, None, 0.0)
        )
        # An instance attribute, not the class one: what the threshold slider
        # should call itself depends on which map is up, and a t map and an R2
        # map are not the same question.
        self.overlay_kind = overlay_kind

        values = self._fit.volume(kind, column=column, psc=str(self.params.get("psc") or "swing"))
        label = self._fit.model.labels[column] if self._fit.model.n_columns else ""
        detail = f"{label} {kind}" if kind in ("beta", "t", "unique R2") else kind
        return ComputedOverlay(
            values=values,
            affine=self._fit.prepared.affine,
            name=self.output_name(detail),
            kind=overlay_kind,
            colormap=colormap,
            display_range=span,
            threshold=threshold,
        )

    # -- graph lines ---------------------------------------------------
    def series(self, ijk: tuple[int, int, int]) -> list[Trace]:
        """The model pulled apart at this voxel.

        The raw run is contributed here as well as decomposed, because the map
        displaces its own input from the layer stack and the graph draws from
        the stack: without it, fitting would make the very time course the fit
        came from disappear from view.

        All five lines, always. Which of them a given graph window draws is a
        tick box on that window -- the whole reason they are crowded is that
        seeing the fit against the residual against the raw data is the
        picture, and a mode-wide switch would hide it from every window at once.
        """
        if self._fit is None:
            return []
        lines = self._fit.decompose(ijk, column=self._column_index())
        if not lines:
            return []
        label = self._fit.model.labels[self._column_index()] if self._fit.model.n_columns else ""
        short = {"column": f"iglm {label}" if label else "iglm column"}
        return [
            Trace(
                label=LINE_HELP.get(name, name),
                key=name,
                short=short.get(name, f"iglm {name}"),
                values=values,
                x_label="TR",
            )
            for name, values in lines.items()
        ]

    def panels(self) -> dict[str, list[Trace]]:
        if self._hrf is None:
            return {}
        seconds = np.arange(self._hrf.shape[1], dtype=np.float32) * self._hrf_dt
        suffixes = ("", "'", "''")
        return {
            "hrf": [
                Trace(
                    label=f"HRF{suffixes[i] if i < len(suffixes) else i}",
                    key=f"hrf{i}",
                    values=curve,
                    x=seconds,
                    x_label="s",
                )
                for i, curve in enumerate(self._hrf)
            ]
        }

    # -- status --------------------------------------------------------
    def residency(self) -> str:
        if self._prepared is None:
            return ""
        gb = self._prepared.bytes / 1e9
        return f"{self._prepared.n_voxels} voxels, {gb:.2f} GB on {self._prepared.y.device.type}"

    def status(self) -> str:
        if self._preparing:
            return "instaglm: fitting…"
        if self._prepared is None:
            return f"instaglm: {self._message or 'pick a run and an events file, then FIT'}"
        return f"instaglm: {self._message}  ·  {self.residency()}"
