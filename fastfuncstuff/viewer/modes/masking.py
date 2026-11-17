"""Which voxels of the run a mode reads: a painted ROI, a mask file, or the automask.

Shared by the modes that compute something per masked voxel (InstaPCA and its
ICA rotation, InstaSpectra), so "mask" means one thing in all of them: a ROI
layer chosen in the picker is followed live as it is painted, a path (or a
loaded layer's key) is read on any grid, and nothing at all is AFNI's automask
of the run.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fastfuncstuff.viewer.commands import Aspect, Command
from fastfuncstuff.viewer.modes.base import ChoiceControl, PathControl

if TYPE_CHECKING:
    from fastfuncstuff.viewer.session import ViewerSession


class RunMaskMixin:
    """Mask controls and their resolution, for a :class:`Mode`."""

    session: ViewerSession | None
    params: dict[str, Any]

    def mask_controls(self) -> tuple[ChoiceControl, PathControl]:
        return (
            ChoiceControl(
                name="roi",
                label="mask ROI",
                choices=("none", *self._roi_choices()),
                default="none",
                help="A painted (or loaded ROI) layer as the mask, followed live as you "
                "paint it. Overrides the mask below.",
            ),
            PathControl(
                name="mask",
                label="mask",
                filter="NIfTI (*.nii *.nii.gz *.nii.zst);;All (*)",
                help="A mask file on the run's grid, or the key of a loaded layer (a "
                "painted ROI, an atlas: nonzero is inside; any grid). Empty means AFNI's "
                "automask of the run.",
            ),
        )

    def _roi_choices(self) -> list[str]:
        if self.session is None:
            return []
        return [
            f"{ly.key} {ly.name}"
            for ly in self.session.state.layers
            if ly.roi or self.session.painter.owns(ly.key)
        ]

    def _roi_key(self) -> str | None:
        """The layer chosen as the mask ROI, if it is still loaded."""
        text = str(self.params.get("roi") or "none").split(" ", 1)[0]
        if text == "none" or self.session is None:
            return None
        return text if self.session.state.layers.find(text) is not None else None

    def mask_spec(self) -> tuple[str, Any]:
        """``(text, layer or None)``: what the mask is, before reading any voxels."""
        assert self.session is not None
        mask_text = str(self.params.get("mask") or "").strip()
        roi_key = self._roi_key()
        if roi_key is not None:
            mask_text = roi_key
        mask_layer = (
            self.session.state.layers.find(mask_text.split(" ", 1)[0]) if mask_text else None
        )
        return mask_text, mask_layer

    def mask_identity(self) -> str:
        """The mask as part of a cache key: a painted layer's data version counts."""
        assert self.session is not None
        text, layer = self.mask_spec()
        version = self.session.store.get(layer.key).generation if layer is not None else 0
        return f"{text}@{version}"

    def mask_source(self) -> str:
        """The mask's name for a header or a status line."""
        text, layer = self.mask_spec()
        return layer.name if layer is not None else Path(text).name if text else "automask"

    def resolve_mask(self, run_layer: Any, shape: tuple[int, int, int]) -> np.ndarray | None:
        """The mask on the run's grid, or ``None`` for the automask."""
        from fastfuncstuff.viewer.instapca import read_mask

        assert self.session is not None
        text, layer = self.mask_spec()
        if layer is not None:
            return self.session.mask_from_layer(layer.key, run_layer)
        return read_mask(text, shape) if text else None

    def mask_stroke(self, cmd: Command) -> bool:
        """Whether ``cmd`` painted the ROI that is the mask."""
        return self.roi_stroke(cmd, self._roi_key())  # ty: ignore[unresolved-attribute]

    def on_command(self, cmd: Command, dirty: Aspect) -> Aspect:
        # A stroke on the ROI that is the mask changes the voxels read.
        if self.mask_stroke(cmd):
            return self.stale()  # ty: ignore[unresolved-attribute]
        return Aspect.NOTHING


__all__ = ["RunMaskMixin"]
