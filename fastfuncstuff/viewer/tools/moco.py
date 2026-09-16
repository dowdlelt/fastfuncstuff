"""Motion correction, as something you can watch rather than run.

The teaching claim this exists to make is simple and hard to believe from a log
file: the volumes in a run are not aligned, and after this step they are. One
input, one button, and a corrected run that lands directly above its source in
the stack so ``[`` and ``]`` flip between them at the same crosshair.

Everything here defers to :func:`fastfuncstuff.processing.ffs_moco.moco`; the
tool is a translation layer and nothing else. The one real piece of work it does
is the axis order: the viewer holds ``(nx, ny, nz, nt)`` and the processing
library works time-first in ``(nt, nz, ny, nx)``, so the transpose happens here,
at the boundary, rather than leaking either convention into the other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from fastfuncstuff.viewer.modes.base import ChoiceControl, Control, ProgressFn
from fastfuncstuff.viewer.tools.base import AuxVolume, Tool, ToolOutcome, tool

if TYPE_CHECKING:
    from fastfuncstuff.viewer.session import ViewerSession

#: Interpolation kernels ffs_moco accepts, coarsest first. The estimation
#: default is heptic and the final-resample default is wsinc5, matching the CLI
#: -- this dialog is meant to teach what the CLI does, so it must not quietly
#: disagree with it.
KERNELS = ("linear", "cubic", "quintic", "heptic", "wsinc5")

#: Which volume everything is aligned to, in the words a person uses. The middle
#: volume is the defensible default for a long run -- it halves the worst-case
#: distance anything has to move -- but ffs_moco's own default is the first, so
#: that stays the default here too.
BASES = ("first", "middle", "last")


def base_index(which: str, n_volumes: int) -> int:
    """Turn a named reference into a volume index."""
    if which == "middle":
        return n_volumes // 2
    if which == "last":
        return n_volumes - 1
    return 0


def qc_volumes(series: np.ndarray, when: str) -> list[AuxVolume]:
    """The first and last volumes, and their difference, for one run.

    This is the picture that shows whether motion correction did anything. The
    difference between the first and last volume of an uncorrected run has the
    shape of the brain in it -- bright rims where an edge moved across a voxel
    boundary. Corrected, the same difference is noise. Two maps, side by side,
    and the step explains itself.

    Split into an intensity pair and a signed difference rather than one
    three-volume stack, because they cannot share a display range: first and
    last are in intensity units and the difference is centred on zero, so one
    window that suits either draws the other as a flat rectangle.
    """
    first, last = series[..., 0], series[..., -1]
    return [
        AuxVolume(
            slot=f"qc_pair_{when}",
            name=f"first/last ({when})",
            values=np.stack([first, last], axis=-1),
            labels=("first", "last"),
        ),
        AuxVolume(
            slot=f"qc_diff_{when}",
            name=f"diff ({when})",
            # Signed and unclamped: where the signal went matters as much as
            # how far, and a diverging map shows both directions at once.
            values=(last - first)[..., None],
            labels=("diff(last-first)",),
            colormap="redblue",
            symmetric=True,
        ),
    ]


@tool
class MocoTool(Tool):
    name = "moco"
    label = "MOCO"
    tag = "MOCO"
    op = "moco"
    input_kind = "4d"
    blurb = (
        "Rigid-body motion correction. Estimates six parameters per volume — three "
        "translations, three rotations — and resamples every volume onto the reference."
    )

    def controls(self) -> tuple[Control, ...]:
        return (
            ChoiceControl(
                name="base",
                label="reference",
                choices=BASES,
                default="first",
                help="The volume every other volume is aligned to.",
            ),
            ChoiceControl(
                name="interp",
                label="interp",
                choices=KERNELS,
                default="heptic",
                help="Kernel used while estimating the motion parameters. Coarser is "
                "faster and slightly less precise.",
            ),
            ChoiceControl(
                name="final_interp",
                label="final interp",
                choices=KERNELS,
                default="wsinc5",
                help="Kernel used for the one resample that produces the output. This is "
                "the one that costs you blurring, so it is finer than the estimation kernel.",
            ),
        )

    def run(
        self,
        session: ViewerSession,
        params: dict[str, Any],
        progress: ProgressFn | None = None,
    ) -> ToolOutcome:
        from fastfuncstuff.processing.ffs_moco import MocoConfig, moco

        key = str(params["input"])
        layer = session.state.layers.get(key)
        if progress is not None:
            progress(0.0, f"reading {layer.name}…")
        values = session.store.ensure_ram(key)  # (nx, ny, nz, nt)
        if values.ndim != 4 or values.shape[3] < 2:
            raise ValueError(f"{layer.name} is not a time series; nothing to align")

        series = torch.from_numpy(np.ascontiguousarray(values.transpose(3, 2, 1, 0)))
        which = str(params.get("base") or "first")
        config = MocoConfig(
            base_index=base_index(which, int(series.shape[0])),
            interp=str(params.get("interp") or "heptic"),
            final_interp=str(params.get("final_interp") or "wsinc5"),
            device=str(session.store.device),
            # Its running report is the dialog's details pane, which is where
            # the interesting part of a preproc step actually is: the device it
            # picked, the cost function, and what the estimation cost per volume.
            verb=1,
        )

        if progress is not None:
            # moco() reports nothing back, so this is a start marker rather than
            # a fraction. The dialog shows an indeterminate bar until it returns.
            progress(0.0, f"aligning {series.shape[0]} volumes to {which}…")
        result = moco(series, config, header_info={"affine": np.asarray(layer.affine, dtype=float)})

        aligned = np.ascontiguousarray(
            result.aligned.detach().to("cpu").numpy().transpose(3, 2, 1, 0), dtype=np.float32
        )
        if progress is not None:
            progress(1.0, "done")
        return ToolOutcome(
            values=aligned,
            detail=f"base {which}, {config.interp}/{config.final_interp}",
            aux=[*qc_volumes(values, "before"), *qc_volumes(aligned, "after")],
        )


__all__ = ["MocoTool", "BASES", "KERNELS", "base_index", "qc_volumes"]
