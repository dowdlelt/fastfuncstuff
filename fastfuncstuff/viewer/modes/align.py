"""Align: move one image onto another by hand, or hand it to allineate.

The fixed image is the underlay -- ``u`` makes anything the underlay, so there is
no second way to say it. The moving image is the mode's INPUT. Anything that
should ride along -- the stat map computed from the moving run -- is attached
with ATTACH and follows on its own grid, which is the useful case when the
grey/white contrast of the moving image is too low to judge: activation that
lands in grey matter is evidence the run is in the right place.

Three ways to move it, all of them the same transform on the same layer:

* **Drag** in any image window, the way ITK-SNAP's manual registration works:
  grab the ring to turn the image about the plane's normal, grab the centre
  handle (or shift-drag anywhere) to slide it. Live, because a transform is an
  affine and every layer is resampled per frame anyway.
* **Sliders**: three shifts and three turns, about a pivot that travels with the
  image. What the sliders show is read back from the transform, so a drag moves
  them too, and a fitted 12-parameter result keeps its scale and shear while
  the sliders turn it.
* **ALLINEATE**, starting from wherever the image is now. The hand alignment
  does not have to be good, only good enough for the search to find the basin
  -- which is the whole reason to do it on data the search fails on.

SAVE writes the transform as an ``.aff12.1D`` (base = the underlay, source = the
moving image), which ``ffs_allineate -1Dmatrix_init`` refines and
``-1Dmatrix_apply`` applies as is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fastfuncstuff.viewer import align
from fastfuncstuff.viewer.commands import Aspect, Command
from fastfuncstuff.viewer.modes.base import (
    ActionControl,
    BoolControl,
    ChoiceControl,
    ComputedOverlay,
    Control,
    DialogSpec,
    FloatControl,
    Mode,
    PathControl,
    ProgressFn,
    mode,
)
from fastfuncstuff.viewer.vocab import SetLayerFollows, SetLayerXform

#: What a transform dirties, for a mode method that moves a layer itself.
MOVED = SetLayerXform.aspects

#: Slider ranges. Wide enough for a header that is a head-width off; the typed
#: box beside each slider goes past them.
SHIFT_MM = 120.0
TURN_DEG = 180.0

COSTS = ("lpc", "lpa", "ls", "nmi", "hel")


@mode
class AlignMode(Mode):
    name = "align"
    label = "Align"
    tag = "ALIGN"
    produces_overlay = False
    input_kind = "any"

    def __init__(self) -> None:
        #: The point the image turns about, in the moving layer's *native* world
        #: coordinates -- a point of the image, so it travels with it. Reset to
        #: the image's centre whenever the moving layer changes.
        self._pivot: np.ndarray | None = None
        self._pivot_for: str | None = None
        #: The moving image chosen when nobody named one. Pinned, because the
        #: automatic answer follows the stack: load or select a stat map to
        #: attach, and "the topmost candidate" would quietly become the map.
        self._pinned: str | None = None
        super().__init__()

    def compute(self) -> ComputedOverlay | None:
        return None

    # -- declaration ---------------------------------------------------
    def controls(self) -> tuple[Control, ...]:
        # One per row: three to a row does not fit the controller's width, and
        # a long slider is what fine hand adjustment wants anyway.
        shift = "mm the image centre has moved along +{}".format
        turn = "degrees about the {} axis, through the pivot".format
        return (
            *(
                FloatControl(
                    name=n,
                    label=label,
                    lo=-SHIFT_MM,
                    hi=SHIFT_MM,
                    default=0.0,
                    step=0.1,
                    unit="mm",
                    help=shift(axis),
                )
                for n, label, axis in (("tx", "R", "R"), ("ty", "A", "A"), ("tz", "S", "S"))
            ),
            *(
                FloatControl(
                    name=n,
                    label=label,
                    lo=-TURN_DEG,
                    hi=TURN_DEG,
                    default=0.0,
                    step=0.1,
                    unit="°",
                    help=turn(axis),
                )
                for n, label, axis in (
                    ("rx", "pitch", "R-L"),
                    ("ry", "roll", "A-P"),
                    ("rz", "yaw", "I-S"),
                )
            ),
        )

    def actions(self) -> tuple[ActionControl, ...]:
        return (
            ActionControl(
                name="attach",
                label="attach",
                help="Make the selected layer ride with the moving image -- the stat map "
                "computed from it, say.",
            ),
            ActionControl(name="detach", label="detach", help="Stop the selected layer following."),
            ActionControl(
                name="pivot",
                label="pivot here",
                help="Turn about the crosshair rather than the image centre.",
            ),
            ActionControl(
                name="cmass",
                label="cmass",
                help="Slide the moving image so its centre of mass sits on the underlay's "
                "(the same centroid as ffs_allineate -cmass). Keeps any rotation.",
            ),
            ActionControl(
                name="reset", label="reset", help="Put the moving image back where its header says."
            ),
            ActionControl(
                name="save",
                label="save",
                help="Write the transform as an .aff12.1D for ffs_allineate -1Dmatrix_init "
                "or -1Dmatrix_apply.",
            ),
            ActionControl(name="load", label="load", help="Start from an .aff12.1D."),
            ActionControl(
                name="allineate",
                label="allineate",
                help="Refine from where the image is now, with ffs_allineate.",
            ),
        )

    def accepts(self, layer: Any) -> bool:
        """Anything but the fixed image, a mode's output, or a follower.

        A follower is moved by its parent; offering it as the moving image would
        be a second, fighting, hand on the same layer.
        """
        if layer.is_computed or layer.follows is not None:
            return False
        base = None if self.session is None else self.session.state.layers.base
        return base is None or layer.key != base.key

    def default_input(self, candidates):
        for layer in candidates:
            if layer.key == self._pinned:
                return layer
        chosen = candidates[0] if candidates else None
        self._pinned = None if chosen is None else chosen.key
        return chosen

    # -- reading the state -----------------------------------------------
    def moving(self):
        return self.source_layer()

    def fixed(self):
        return None if self.session is None else self.session.state.layers.base

    def pivot(self) -> np.ndarray | None:
        """The pivot in native coordinates of the moving layer."""
        layer = self.moving()
        if layer is None:
            return None
        if self._pivot_for != layer.key or self._pivot is None:
            self._pivot = align.centre_mm(layer)
            self._pivot_for = layer.key
        return self._pivot

    def pivot_mm(self) -> np.ndarray | None:
        """Where the pivot is drawn now, in display millimetres."""
        layer, pivot = self.moving(), self.pivot()
        if layer is None or pivot is None:
            return None
        return (align.layer_xform(layer) @ np.append(pivot, 1.0))[:3]

    def xform(self) -> np.ndarray:
        layer = self.moving()
        return align.IDENTITY.copy() if layer is None else align.layer_xform(layer)

    def _sync_params(self) -> None:
        pivot = self.pivot()
        values = (0.0,) * 6 if pivot is None else align.decompose(self.xform(), pivot)
        self.params.update({n: round(v, 6) for n, v in zip(align.PARAMS, values, strict=True)})

    # -- changing it ---------------------------------------------------
    def move_to(self, xform: np.ndarray) -> Command | None:
        """The command that draws the moving image through ``xform``.

        Returned rather than applied, so a gesture goes through the bus like
        every other change and lands in the recording.
        """
        layer = self.moving()
        return None if layer is None else SetLayerXform.of(layer.key, xform)

    def turned(self, axis, degrees: float) -> np.ndarray:
        """The current transform, turned about the drawn pivot."""
        centre = self.pivot_mm()
        if centre is None:
            return self.xform()
        return align.about(centre, align.axis_rotation(axis, degrees)) @ self.xform()

    def shifted(self, mm) -> np.ndarray:
        return align.translation(mm) @ self.xform()

    def _apply(self, xform: np.ndarray) -> Aspect:
        """Move the layer from inside a SET_MODE_PARAM or MODE_ACTION.

        Direct rather than dispatched: the command already being handled is the
        one the recording holds, and replaying it redoes this.
        """
        layer = self.moving()
        if self.session is None or layer is None:
            return Aspect.NOTHING
        align.set_xform(self.session.state.layers, layer.key, xform)
        self._sync_params()
        return MOVED

    def set_param(self, name: str, value: Any) -> Aspect:
        if name not in align.PARAMS:
            return super().set_param(name, value)
        pivot = self.pivot()
        if pivot is None:
            return Aspect.NOTHING
        self._sync_params()
        self.params[name] = float(value)
        _, stretch = align.polar(self.xform()[:3, :3])
        values = [self.params[n] for n in align.PARAMS]
        return self._apply(align.compose(values, pivot, stretch))

    def attach(self, session) -> None:
        super().attach(session)
        self._sync_params()

    def on_command(self, cmd: Command, dirty: Aspect) -> Aspect:
        # Whatever moved the layer -- a drag, a replayed line, another tab's
        # script -- the sliders are a reading of it.
        if dirty & (Aspect.SLICES | Aspect.LAYERS):
            self._sync_params()
        return Aspect.NOTHING

    def action(self, name: str, progress: ProgressFn | None = None) -> Aspect:
        if self.session is None:
            return Aspect.NOTHING
        stack = self.session.state.layers
        layer = self.moving()
        if name == "reset":
            return self._apply(align.IDENTITY)
        if name == "cmass":
            if layer is None or self.fixed() is None:
                return Aspect.NOTHING
            moving, fixed, base, source = self._volumes()
            target = align.centre_of_mass_mm(base, fixed.affine)
            drawn = self.xform() @ np.append(
                align.centre_of_mass_mm(source, align.native_affine(moving)), 1.0
            )
            return self._apply(self.shifted(target - drawn[:3]))
        if name == "pivot":
            here = self.session.state.crosshair_mm
            if layer is None or here is None:
                return Aspect.NOTHING
            # Stored in native coordinates, so it stays on the same bit of anatomy.
            self._pivot = (np.linalg.inv(self.xform()) @ np.append(here, 1.0))[:3]
            self._pivot_for = layer.key
            self._sync_params()
            return Aspect.SLICES
        if name in ("attach", "detach"):
            chosen = self.session.state.selected_layer()
            if chosen is None or layer is None:
                return Aspect.NOTHING
            if name == "detach":
                align.set_follows(stack, chosen.key, None)
            elif chosen.key == layer.key:
                raise ValueError(f"{chosen.name} is the moving image; select the layer to attach")
            else:
                align.set_follows(stack, chosen.key, layer.key)
            return SetLayerFollows.aspects
        return super().action(name, progress)

    # -- dialogs -------------------------------------------------------
    def _blocked(self) -> str:
        if self.fixed() is None:
            return "load a fixed image first; the underlay is what the others are aligned to"
        if self.moving() is None:
            return "load a second image to move onto the underlay"
        return ""

    def default_matrix_path(self) -> str:
        layer, base = self.moving(), self.fixed()
        if layer is None or base is None:
            return ""
        stem = Path(layer.path).name.split(".")[0] if layer.path else layer.name
        base_stem = Path(base.path).name.split(".")[0] if base.path else base.name
        folder = Path(layer.path).parent if layer.path else Path.cwd()
        return str(folder / f"{stem}_to_{base_stem}.aff12.1D")

    def dialog_for(self, action: str) -> DialogSpec | None:
        if self.session is None:
            return None
        blocked = self._blocked()
        moving, fixed = self.moving(), self.fixed()
        pair = "" if blocked else f"{moving.name} onto {fixed.name}"
        if action in ("save", "load"):
            saving = action == "save"
            return DialogSpec(
                name=f"align-{action}",
                title="save matrix" if saving else "load matrix",
                blurb=(
                    f"{pair}. Base->source in DICOM mm, the .aff12.1D ffs_allineate and "
                    "3dAllineate read: -base the underlay, -source the moving image."
                    if saving
                    else f"Draw {moving.name if moving else 'the moving image'} where an "
                    ".aff12.1D puts it, relative to the underlay."
                ),
                controls=(
                    PathControl(
                        name="path",
                        label="file",
                        default=self.default_matrix_path(),
                        filter="AFNI matrix (*.aff12.1D *.1D);;All (*)",
                    ),
                ),
                params={"path": self.default_matrix_path()},
                run=self._save if saving else self._load,
                install=self._install,
                run_label=action,
                blocked=blocked,
                done="written" if saving else "moved",
            )
        if action == "allineate":
            controls = (
                ChoiceControl(
                    name="cost",
                    label="cost",
                    choices=COSTS,
                    default="lpc",
                    help="lpc for EPI onto a T1 (opposite contrast), ls or lpa for like "
                    "onto like, nmi when unsure.",
                ),
                ChoiceControl(
                    name="dof",
                    label="warp",
                    choices=("rigid", "affine"),
                    default="rigid",
                    style="radio",
                    help="rigid: 6 parameters, the same head. affine: 12, for scanner "
                    "scaling or a template.",
                ),
                BoolControl(
                    name="small",
                    label="small range",
                    default=True,
                    help="Search near where the image is now. On when the hand alignment "
                    "is close; off to let the search roam.",
                ),
            )
            return DialogSpec(
                name="align-allineate",
                title="allineate",
                blurb=f"Refine {pair}, starting from where it is drawn now."
                if pair
                else "Refine the moving image onto the underlay.",
                controls=controls,
                params={c.name: getattr(c, "default", None) for c in controls},
                run=self._allineate,
                install=self._install,
                run_label="allineate",
                blocked=blocked,
                done="moved",
            )
        return None

    def _pair(self) -> tuple[Any, Any, np.ndarray, np.ndarray]:
        moving, fixed = self.moving(), self.fixed()
        if moving is None or fixed is None:
            raise ValueError(self._blocked())
        return moving, fixed, np.asarray(fixed.affine, float), align.native_affine(moving)

    def _volumes(self) -> tuple[Any, Any, np.ndarray, np.ndarray]:
        """``(moving, fixed, fixed voxels, moving voxels)``, each ``(nx, ny, nz)``.

        The displayed sub-brick of each: for a run, the volume on screen, which
        is the one that was just lined up by eye.
        """
        moving, fixed, _, _ = self._pair()
        assert self.session is not None
        base = self.session.volume(fixed.key, fixed.volume_index)
        source = self.session.volume(
            moving.key, None if moving.time_linked else moving.volume_index
        )
        return moving, fixed, np.asarray(base, np.float32), np.asarray(source, np.float32)

    def _save(self, params: dict[str, Any], progress: ProgressFn | None) -> None:
        moving, fixed, base_aff, native = self._pair()
        path = str(params.get("path") or "").strip()
        if not path:
            raise ValueError("no file named")
        align.save_aff12(
            path,
            self.xform(),
            base_aff,
            native,
            header=f"ffs_viewer align: base {fixed.path or fixed.name}, "
            f"source {moving.path or moving.name}",
        )
        return None

    def _load(self, params: dict[str, Any], progress: ProgressFn | None) -> Command:
        _, _, base_aff, native = self._pair()
        path = str(params.get("path") or "").strip()
        command = self.move_to(align.load_aff12(path, base_aff, native))
        if command is None:
            raise ValueError("nothing to move")
        return command

    def _allineate(self, params: dict[str, Any], progress: ProgressFn | None) -> Command:
        """On the worker: read, fit, and return the command that installs it."""
        import torch

        from fastfuncstuff.processing.allineate import AffineAlignConfig, allineate

        if self.session is None:
            raise RuntimeError("no session")
        moving, fixed, base_aff, native = self._pair()
        if progress is not None:
            progress(0.0, f"aligning {moving.name} to {fixed.name}…")
        _, _, base, source = self._volumes()
        to_zyx = lambda v: torch.from_numpy(np.ascontiguousarray(v.transpose(2, 1, 0)))  # noqa: E731
        config = AffineAlignConfig(
            dof=str(params.get("dof") or "rigid"),
            cost=str(params.get("cost") or "lpc"),
            range_scale=0.5 if params.get("small", True) else 1.0,
            init_matrix=align.to_aff12(self.xform(), base_aff, native),
            device=str(self.session.store.device),
            verb=1,
        )
        matrix, _ = allineate(
            to_zyx(np.asarray(base, dtype=np.float32)),
            to_zyx(np.asarray(source, dtype=np.float32)),
            config,
            base_header={"affine": base_aff},
            source_header={"affine": native},
        )
        voxel = matrix.detach().to("cpu", torch.float64).numpy()
        xform = base_aff @ np.linalg.inv(voxel) @ np.linalg.inv(native)
        if progress is not None:
            progress(1.0, "done")
        return SetLayerXform.of(moving.key, xform)

    def _install(self, result: Command | None) -> Aspect:
        """On the GUI thread: apply through the bus so the move is recorded."""
        if self.session is None or result is None:
            return Aspect.NOTHING
        return self.session.do(result)

    # -- status --------------------------------------------------------
    def status(self) -> str:
        moving, fixed = self.moving(), self.fixed()
        if self.session is None or moving is None or fixed is None:
            return "align: " + self._blocked()
        riders = [ly.name for ly in align.followers(self.session.state.layers, moving.key)]
        extra = f" (+ {', '.join(riders)})" if riders else ""
        return f"align: {moving.name}{extra} onto {fixed.name}"


__all__ = ["AlignMode"]
