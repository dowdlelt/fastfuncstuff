"""Record a nonlinear warp as the optimizer builds it, then render it as a movie.

The recorder never warps a volume. A frame only shows a few display planes, so it
pulls the source image through the *current* field at just those planes' pixels --
a few hundred thousand trilinear samples, small next to one optimizer iteration.
The samples stay on the device as float16 until :meth:`WarpMovieRecorder.render`,
so capturing costs no host sync; composition and encoding happen once, on the CPU,
after the fit.

Frame budget. Tools stop levels early, so how many iterations a run will take is
unknown up front and a fixed "every N" cadence gives anywhere from a handful of
frames to thousands. With ``max_frames`` the recorder instead starts at every
iteration and, whenever the buffer overflows, drops every other unpinned frame and
doubles its stride -- so the kept frames always stay evenly spaced in iteration
count and never exceed the budget. Pinned frames (a level's returned field) are
never dropped.

Calling convention for a tool::

    if recorder is not None and recorder.tick():
        recorder.capture_displacement(field, label=f"L{lev}/{n} it {it} cost {c:.4f}")
    ...
    recorder.capture_displacement(best, label="L1/3 best", pinned=True)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .compose import compose_frame, display_edges, edge_range, intensity_window, panel_sizes
from .encode import movie_path, write_movie
from .slices import SlicePlanes

Field = tuple[Tensor, Tensor, Tensor]


@dataclass
class _Frame:
    values: Tensor
    label: str
    pinned: bool


class WarpMovieRecorder:
    """Accumulates sampled display planes of a source image seen through a warp.

    Args:
        moving: (nz, ny, nx) image being warped, on the grid the planes index.
        planes: Display planes, from :func:`fastfuncstuff.viz.slices.build_slice_planes`.
        every: Fixed capture cadence in :meth:`tick` calls. Mutually exclusive with
            ``max_frames``.
        max_frames: Frame budget for unpinned frames (see module docstring).
        reference: Optional (nz, ny, nx) fixed image (the base) on the same grid. Its
            planes are kept so ``render(overlay="edges")`` can outline it.
        tool: Name shown at the start of every caption.
        device: Where sampling happens. Defaults to ``moving``'s device.
    """

    def __init__(
        self,
        moving: Tensor,
        planes: SlicePlanes,
        *,
        every: int | None = None,
        max_frames: int | None = 150,
        reference: Tensor | None = None,
        tool: str = "",
        device: torch.device | None = None,
    ) -> None:
        if every is not None and every < 1:
            raise ValueError("every must be >= 1")
        if every is None and (max_frames is None or max_frames < 2):
            raise ValueError("max_frames must be >= 2 when no fixed cadence is given")
        if tuple(moving.shape) != planes.grid_shape:
            raise ValueError(
                f"moving {tuple(moving.shape)} is not on the planes' grid {planes.grid_shape}"
            )
        self.device = device if device is not None else moving.device
        self.planes = planes
        self.tool = tool
        self._every = every
        self._max_frames = None if every is not None else max_frames
        self._stride = every or 1
        self._ticks = 0
        self._frames: list[_Frame] = []

        nz, ny, nx = planes.grid_shape
        self._moving = moving.detach().float().to(self.device)[None, None]
        self._points = torch.as_tensor(planes.points, device=self.device)
        self._flat = torch.as_tensor(planes.flat_indices(), device=self.device)
        # grid_sample's normalized frame with align_corners=True is the same for a
        # coarse pyramid level and the full grid when the level was made by an
        # align_corners resize -- which every FFS pyramid is -- so one grid serves all.
        denom = torch.tensor([max(nz - 1, 1), max(ny - 1, 1), max(nx - 1, 1)], device=self.device)
        self._denom = denom.float()
        self._plane_grid = self._normalize(self._points)
        self._reference = None
        if reference is not None:
            if tuple(reference.shape) != planes.grid_shape:
                raise ValueError("reference must be on the planes' grid")
            flat = self._flat.to(reference.device)
            self._reference = reference.detach().float().reshape(-1)[flat].cpu().numpy()

    # -- capture -------------------------------------------------------------------

    def _normalize(self, zyx: Tensor) -> Tensor:
        """(N, 3) z,y,x voxel coordinates -> (1, 1, 1, N, 3) x,y,z grid_sample grid."""
        g = 2.0 * zyx / self._denom - 1.0
        return g.flip(-1)[None, None, None]

    def tick(self) -> bool:
        """Advance one iteration; True when this iteration should be captured."""
        due = self._ticks % self._stride == 0
        self._ticks += 1
        return due

    @property
    def n_frames(self) -> int:
        return len(self._frames)

    def capture_values(self, values: Tensor, label: str = "", pinned: bool = False) -> None:
        """Store an already-sampled (N,) plane vector as a frame."""
        self._frames.append(_Frame(values.detach().to(torch.float16), label, pinned))
        if self._max_frames is not None:
            n_free = sum(not f.pinned for f in self._frames)
            if n_free > self._max_frames:
                self._decimate()

    def _decimate(self) -> None:
        kept, ordinal = [], 0
        for f in self._frames:
            if f.pinned:
                kept.append(f)
                continue
            # Unpinned frame k was captured at tick k * stride; keeping the even k
            # leaves exactly the multiples of the doubled stride.
            if ordinal % 2 == 0:
                kept.append(f)
            ordinal += 1
        self._frames = kept
        self._stride *= 2

    def capture_displacement(self, field: Field, label: str = "", pinned: bool = False) -> None:
        """Capture the source pulled through a displacement field.

        ``field`` is ``(xd, yd, zd)``, each (gz, gy, gx) in voxel units *of that grid*
        -- the convention of :func:`fastfuncstuff.processing.interp.warp_image_linear`,
        where output voxel (k, j, i) samples the source at (i + xd, j + yd, k + zd).
        The grid may be a coarse pyramid level (an align_corners resize of the full
        grid); its displacements are rescaled to full-grid voxels here.
        """
        with torch.no_grad():
            xd, yd, zd = field
            full = self.planes.grid_shape
            if tuple(xd.shape) == full:
                disp = torch.stack([c.reshape(-1)[self._flat] for c in (zd, yd, xd)], dim=1)
            else:
                gz, gy, gx = xd.shape
                stacked = torch.stack((xd, yd, zd))[None].float()
                sampled = F.grid_sample(
                    stacked,
                    self._plane_grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )[0, :, 0, 0]  # (3, N) as x, y, z
                ratio = torch.tensor(
                    [
                        (full[2] - 1) / max(gx - 1, 1),
                        (full[1] - 1) / max(gy - 1, 1),
                        (full[0] - 1) / max(gz - 1, 1),
                    ],
                    device=self.device,
                )
                disp = (sampled * ratio[:, None]).flip(0).T  # (N, 3) z, y, x
            src = self._points + disp.to(self._points.dtype)
            values = F.grid_sample(
                self._moving,
                self._normalize(src),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).reshape(-1)
        self.capture_values(values, label, pinned)

    def capture_identity(self, label: str = "", pinned: bool = True) -> None:
        """Capture the unwarped source (the movie's starting frame)."""
        with torch.no_grad():
            values = self._moving.reshape(-1)[self._flat]
        self.capture_values(values, label, pinned)

    # -- render --------------------------------------------------------------------

    def render(
        self,
        path: str,
        *,
        fps: int = 10,
        size: int = 256,
        fmt: str = "mp4",
        hold: float = 0.5,
        overlay: str = "none",
        edge_opacity: float = 1.0,
    ) -> str | None:
        """Compose and encode every captured frame; return the path written.

        Args:
            path: Output path or prefix (the container extension is added if absent).
            fps: Frames per second.
            size: Pixel height of the physically tallest panel.
            fmt: ``mp4`` or ``gif``.
            hold: Seconds each pinned frame (a level's result) stays on screen; the
                final frame is held at least a second.
            overlay: ``none`` or ``edges`` -- thin edges of ``reference``, found in each
                displayed plane at display resolution (see :func:`.compose.display_edges`).
            edge_opacity: 0..1 blend of the edge colour.
        """
        if not self._frames:
            return None
        if overlay not in ("none", "edges"):
            raise ValueError(f"unknown overlay {overlay!r}")
        if overlay == "edges" and self._reference is None:
            raise ValueError("overlay='edges' needs a reference image passed to the recorder")

        values = torch.stack([f.values for f in self._frames]).float().cpu().numpy()
        views = self.planes.views
        window = intensity_window(values[0])
        edge_panels: Sequence[np.ndarray] | None = None
        edge_vmax = 1.0
        if overlay == "edges" and self._reference is not None:
            sizes = panel_sizes(views, size)
            edge_panels = [
                display_edges(plane, hw)
                for plane, hw in zip(self.planes.split(self._reference), sizes, strict=True)
            ]
            edge_vmax = edge_range(np.concatenate([e.ravel() for e in edge_panels]))

        n = len(self._frames)
        hold_n = max(1, int(round(hold * fps)))
        out: list[np.ndarray] = []
        for i, (f, row) in enumerate(zip(self._frames, values, strict=True)):
            label = f"{self.tool}  {f.label}".strip() if self.tool else f.label
            frame = compose_frame(
                self.planes.split(row),
                views,
                size,
                window,
                edges=edge_panels,
                edge_vmax=edge_vmax,
                edge_opacity=edge_opacity,
                label=label,
                progress=i / max(n - 1, 1),
            )
            repeats = hold_n if f.pinned else 1
            if i == n - 1:
                repeats = max(repeats, fps)
            out.extend([frame] * repeats)
        return write_movie(np.stack(out), movie_path(path, fmt), fps, fmt)
