"""Spatial smoothing, which is the easiest processing step to see and to misread.

Blur a run and the picture gets prettier: noise falls away, blobs firm up. What
it also does is destroy the thing people usually came for -- the boundary
between two structures a few millimetres apart -- and that is equally visible if
you look at the right place. Stepping between the run and its smoothed copy at a
crosshair on a sulcus is the whole lesson, and it needs no QC volume: the data
is the QC.

No aux volumes and no plots here for exactly that reason. A step whose effect is
the image itself should not be explained with a second image.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from fastfuncstuff.viewer.modes.base import Control, FloatControl, ProgressFn
from fastfuncstuff.viewer.tools.base import Tool, ToolOutcome, tool

if TYPE_CHECKING:
    from fastfuncstuff.viewer.session import ViewerSession


def voxel_sizes(affine: np.ndarray) -> tuple[float, float, float]:
    """Millimetres per voxel along each axis, from the affine's rotation block.

    The column norms rather than the diagonal, so an oblique dataset -- where
    the diagonal understates every spacing -- is blurred by the width asked for
    rather than by rather less than that.
    """
    dims = np.sqrt((np.asarray(affine, dtype=float)[:3, :3] ** 2).sum(axis=0))
    return (float(dims[0]), float(dims[1]), float(dims[2]))


@tool
class SmoothTool(Tool):
    name = "smooth"
    label = "SMOOTH"
    tag = "SMOOTH"
    op = "smooth"
    input_kind = "4d"
    blurb = (
        "Gaussian spatial smoothing. Raises sensitivity to blobs larger than the kernel "
        "and erases detail smaller than it — step between the two to see both halves."
    )

    def controls(self) -> tuple[Control, ...]:
        return (
            FloatControl(
                name="fwhm",
                label="FWHM",
                lo=0.0,
                hi=16.0,
                default=4.0,
                step=0.5,
                unit="mm",
                help="Width of the Gaussian at half its height, in millimetres — not in "
                "voxels, so the same number means the same blur on any grid. A common "
                "choice is about twice the voxel size.",
            ),
        )

    def run(
        self,
        session: ViewerSession,
        params: dict[str, Any],
        progress: ProgressFn | None = None,
    ) -> ToolOutcome:
        from fastfuncstuff.utils import gaussian_blur_3d

        key = str(params["input"])
        layer = session.state.layers.get(key)
        fwhm = float(params.get("fwhm") or 0.0)
        if fwhm <= 0:
            raise ValueError("FWHM must be greater than zero; a 0 mm blur is the input")

        if progress is not None:
            progress(0.0, f"reading {layer.name}…")
        values = session.store.ensure_ram(key)  # (nx, ny, nz, nt)
        if values.ndim != 4:
            raise ValueError(f"{layer.name} is not a time series")

        sizes = voxel_sizes(layer.affine)
        if progress is not None:
            progress(0.0, f"blurring {values.shape[3]} volumes at {fwhm:g} mm…")
        # Already (x, y, z, t), which is what gaussian_blur_3d wants -- this is
        # the one tool that needs no axis juggling at the boundary.
        blurred = np.ascontiguousarray(
            gaussian_blur_3d(
                np.asarray(values, dtype=np.float32),
                fwhm_mm=fwhm,
                voxel_sizes=sizes,
                device=session.store.device,
                verbose=True,
            ),
            dtype=np.float32,
        )
        if progress is not None:
            progress(1.0, "done")
        voxel = "x".join(f"{s:g}" for s in sizes)
        return ToolOutcome(values=blurred, detail=f"FWHM {fwhm:g} mm on {voxel} mm voxels")


__all__ = ["SmoothTool", "voxel_sizes"]
