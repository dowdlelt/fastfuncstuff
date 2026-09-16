"""Slice-timing correction: the step whose effect is real and nearly invisible.

Smoothing shows itself at a glance. This one does not, and that is the lesson.
The slices of a volume were not acquired at the same instant -- in a 2 s TR with
40 slices, the top of the brain is sampled almost two seconds after the bottom
-- and correcting it shifts each slice's time course by a fraction of a TR.
Nothing about any single volume looks different afterwards. What changes is the
*timing* of everything downstream, which is why the step exists and why it is so
easy to skip.

So the thing to look at is a voxel's time course in a superior slice against the
same voxel before correction, in the graph window, with the crosshair parked
where the two differ. No QC volume would help; a picture of a corrected volume
looks exactly like a picture of an uncorrected one.

The timing table comes from the BIDS sidecar, found beside the input by its own
name. That is deliberate: the alternative is asking someone to type 40 numbers,
or worse, to pick a named pattern (alt+z, seq+z) that may not be what the
scanner actually did. The scanner wrote the answer down; read it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from fastfuncstuff.viewer.catalog import SUFFIXES
from fastfuncstuff.viewer.modes.base import ChoiceControl, Control, PathControl, ProgressFn
from fastfuncstuff.viewer.tools.base import Tool, ToolOutcome, tool

if TYPE_CHECKING:
    from fastfuncstuff.viewer.session import ViewerSession

#: Temporal kernels ffs_slicetime accepts. Fourier is its default and AFNI's,
#: and is the right one for a pure fractional shift of a band-limited signal.
KERNELS = ("fourier", "linear", "cubic", "quintic", "heptic", "wsinc5", "wsinc9")

#: What all slices are aligned *to*. The names are what a person would say; the
#: arithmetic is in :func:`reference_time`.
REFERENCES = ("mean", "start of TR", "first slice", "middle slice")


def dataset_stem(path: Path) -> str:
    """``sub-01_bold`` from ``sub-01_bold.nii.gz``.

    ``Path.stem`` only strips one suffix, which leaves ``.nii`` on every
    gzipped NIfTI and makes the sidecar lookup miss on the common case.
    """
    name = path.name
    for suffix in SUFFIXES:
        if name.lower().endswith(suffix.lower()):
            return name[: -len(suffix)]
    return path.stem


def sidecar_for(path: Path) -> Path | None:
    """The BIDS JSON sitting beside a dataset under the same name."""
    candidate = path.with_name(f"{dataset_stem(path)}.json")
    return candidate if candidate.is_file() else None


def source_file(session: ViewerSession, key: str) -> Path | None:
    """The file a layer came from, following derived layers back to one.

    A run that has already been motion-corrected has no file of its own, but
    its slice timing is whatever the scanner did and no processing since has
    changed it -- so the original's sidecar is still the right answer.
    """
    seen: set[str] = set()
    while key and key not in seen:
        seen.add(key)
        layer = session.state.layers.find(key)
        if layer is None:
            return None
        path = Path(layer.path)
        if path.is_file():
            return path
        made_by = layer.source.split(":", 2)
        key = made_by[2] if made_by[:1] == ["derived"] and len(made_by) == 3 else ""
    return None


def reference_time(which: str, timing: list[float]) -> float | None:
    """Turn a named reference into the time all slices are shifted to.

    ``None`` means "let the library decide", which is the mean -- the same
    default 3dTshift uses, and the one that moves every slice the least.
    """
    if which == "start of TR":
        return 0.0
    if which == "first slice":
        return float(timing[0])
    if which == "middle slice":
        return float(timing[len(timing) // 2])
    return None


def repetition_time(sidecar: Path | None) -> float:
    """``RepetitionTime`` from a sidecar, or 0.0 if it does not say."""
    if sidecar is None or sidecar.suffix.lower() != ".json":
        return 0.0
    try:
        return float(json.loads(sidecar.read_text()).get("RepetitionTime", 0.0))
    except (OSError, ValueError, TypeError):
        return 0.0


@tool
class SliceTimeTool(Tool):
    name = "slicetime"
    label = "SLICETIME"
    tag = "SLICETIME"
    op = "slicetime"
    input_kind = "4d"
    blurb = (
        "Shift every slice to a common time. The slices of a volume were acquired "
        "seconds apart; nothing about a single volume looks different afterwards, but "
        "every time course does — graph a superior voxel before and after."
    )

    def controls(self) -> tuple[Control, ...]:
        return (
            PathControl(
                name="timing",
                label="timing",
                filter="BIDS sidecar (*.json);;Timing file (*.1D *.txt);;All (*)",
                help="A BIDS sidecar with SliceTiming, or a text file of one offset per "
                "slice in seconds. Empty means the .json beside the input, under the "
                "same name — which is where the scanner already wrote it.",
            ),
            ChoiceControl(
                name="reference",
                label="reference",
                choices=REFERENCES,
                default="mean",
                help="The time every slice is shifted to. The mean moves each slice the "
                "least and is what 3dTshift does; start of TR aligns to the volume's "
                "nominal onset, which is what a model built on TR boundaries assumes.",
            ),
            ChoiceControl(
                name="interp",
                label="interp",
                choices=KERNELS,
                default="fourier",
                help="Temporal kernel for the sub-TR shift. Fourier is exact for a "
                "band-limited signal and is the default here and in AFNI.",
            ),
        )

    def run(
        self,
        session: ViewerSession,
        params: dict[str, Any],
        progress: ProgressFn | None = None,
    ) -> ToolOutcome:
        from fastfuncstuff.processing.slicetime import load_slice_timing, slicetime_correct

        key = str(params["input"])
        layer = session.state.layers.get(key)

        chosen = str(params.get("timing") or "").strip()
        origin = Path(chosen).expanduser() if chosen else None
        if origin is None:
            source = source_file(session, key)
            origin = sidecar_for(source) if source is not None else None
            if origin is None:
                where = source.parent if source is not None else "the input's directory"
                raise ValueError(
                    f"no slice timing found for {layer.name}: expected "
                    f"{dataset_stem(source) if source else layer.name}.json in {where}. "
                    "Choose a timing file, or point the viewer at the raw data."
                )
        if not origin.is_file():
            raise ValueError(f"no such timing file: {origin}")

        timing = load_slice_timing(origin)
        if progress is not None:
            progress(0.0, f"reading {layer.name}…")
        values = session.store.ensure_ram(key)  # (nx, ny, nz, nt)
        if values.ndim != 4 or values.shape[3] < 2:
            raise ValueError(f"{layer.name} is not a time series; nothing to shift")
        if len(timing) != values.shape[2]:
            raise ValueError(
                f"{origin.name} lists {len(timing)} slice times but {layer.name} has "
                f"{values.shape[2]} slices"
            )

        tr = float(session.store.get(key).info.tr) or repetition_time(origin)
        if tr <= 0:
            raise ValueError(
                f"{layer.name} has no TR in its header and {origin.name} gives no "
                "RepetitionTime; slice timing is meaningless without one"
            )

        which = str(params.get("reference") or "mean")
        method = str(params.get("interp") or "fourier")
        if progress is not None:
            progress(0.0, f"shifting {len(timing)} slices to {which}…")

        series = torch.from_numpy(np.ascontiguousarray(values.transpose(3, 2, 1, 0)))
        shifted = slicetime_correct(
            series,
            timing,
            tr=tr,
            tzero=reference_time(which, timing),
            method=method,
            device=session.store.device,
            verbose=True,
        )
        out = np.ascontiguousarray(
            shifted.detach().to("cpu").numpy().transpose(3, 2, 1, 0), dtype=np.float32
        )
        if progress is not None:
            progress(1.0, "done")
        span = max(timing) - min(timing)
        return ToolOutcome(
            values=out,
            detail=f"{which}, {method}, TR {tr:g}s, {span:g}s across slices ({origin.name})",
        )


__all__ = [
    "KERNELS",
    "REFERENCES",
    "SliceTimeTool",
    "dataset_stem",
    "reference_time",
    "repetition_time",
    "sidecar_for",
    "source_file",
]
