"""Record a nonlinear warp as the optimizer builds it, then render it as a movie.

The recorder never warps a volume. A frame only shows a few display planes, so it
follows each plane pixel through the *current* field(s) and samples the source there
-- a few hundred thousand trilinear samples, small next to one optimizer iteration.
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

Tools differ in how their working grid relates to the image. :class:`FieldFrame`
says so once: a padding offset (qwarp works on a padded grid), and how a coarse level
was made from the full grid (an align-corners resize, an align-centres resize, or a
stride). A capture then accepts fields on any level of that pyramid.

Calling convention for a tool::

    if recorder is not None and recorder.tick():
        recorder.capture_displacement(field, label=f"L{lev}/{n} it {it} cost {c:.4f}")
    ...
    recorder.capture_displacement(best, label="L1/3 best", pinned=True)
"""

from __future__ import annotations

import math
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
"""``(xd, yd, zd)`` voxel displacements, each (nz, ny, nx) on one grid: output voxel
(k, j, i) samples the source at (i + xd, j + yd, k + zd) -- the convention of
:func:`fastfuncstuff.processing.interp.warp_image_linear`."""

_PYRAMID_MAPPINGS = ("corners", "centres", "stride")


@dataclass(frozen=True)
class FieldFrame:
    """Where a tool's fields live relative to the display planes' grid.

    Attributes:
        offset: (z, y, x) index, in the field's full-resolution grid, of the planes
            grid's voxel 0. Nonzero when the tool pads (qwarp).
        full_shape: The field's full-resolution grid. ``None`` means the planes grid.
        mapping: How a coarse level of shape ``g`` was made from ``full_shape`` ``N``:
            ``corners`` (``F.interpolate(align_corners=True)``, the optiwarp/formwarp
            pyramid), ``centres`` (``align_corners=False``, qwarp's octaves) or
            ``stride`` (``vol[::s]``, blipflip's subsampling).
    """

    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    full_shape: tuple[int, int, int] | None = None
    mapping: str = "corners"

    def __post_init__(self) -> None:
        if self.mapping not in _PYRAMID_MAPPINGS:
            raise ValueError(f"mapping must be one of {_PYRAMID_MAPPINGS}, got {self.mapping!r}")


def _level_affine(n: int, g: int, mapping: str) -> tuple[float, float]:
    """(a, b) with level coordinate = a * full coordinate + b, for one axis."""
    if g == n:
        return 1.0, 0.0
    if mapping == "corners":
        return ((g - 1) / (n - 1) if n > 1 else 1.0), 0.0
    if mapping == "centres":
        a = g / n
        return a, 0.5 * a - 0.5
    return 1.0 / math.ceil(n / g), 0.0  # stride: g = ceil(n / s)


@dataclass
class _Frame:
    values: Tensor  # (n_rows, N) float16
    label: str
    pinned: bool


class WarpMovieRecorder:
    """Accumulates sampled display planes of one or more images seen through warps.

    Args:
        moving: (nz, ny, nx) image being warped, or a sequence of them -- one movie row
            each (blipflip's blip-up and blip-down) -- all on the planes' grid.
        planes: Display planes, from :func:`fastfuncstuff.viz.slices.build_slice_planes`.
        every: Fixed capture cadence in :meth:`tick` calls. Mutually exclusive with
            ``max_frames``.
        max_frames: Frame budget for unpinned frames (see module docstring).
        reference: Optional (nz, ny, nx) fixed image (the base) on the same grid. Its
            planes are kept so ``render(overlay="edges")`` can outline it.
        tool: Name shown at the start of every caption.
        row_labels: Optional name per row, drawn on the row.
        device: Where sampling happens. Defaults to the first image's device.
    """

    def __init__(
        self,
        moving: Tensor | Sequence[Tensor],
        planes: SlicePlanes,
        *,
        every: int | None = None,
        max_frames: int | None = 150,
        reference: Tensor | None = None,
        tool: str = "",
        row_labels: Sequence[str] | None = None,
        device: torch.device | None = None,
    ) -> None:
        images = [moving] if isinstance(moving, Tensor) else list(moving)
        if not images:
            raise ValueError("at least one moving image is required")
        if every is not None and every < 1:
            raise ValueError("every must be >= 1")
        if every is None and (max_frames is None or max_frames < 2):
            raise ValueError("max_frames must be >= 2 when no fixed cadence is given")
        for img in images:
            if tuple(img.shape) != planes.grid_shape:
                raise ValueError(
                    f"moving {tuple(img.shape)} is not on the planes' grid {planes.grid_shape}"
                )
        if row_labels is not None and len(row_labels) != len(images):
            raise ValueError("need one row label per moving image")
        self.device = device if device is not None else images[0].device
        self.planes = planes
        self.tool = tool
        self.row_labels = list(row_labels) if row_labels is not None else None
        self.frame = FieldFrame()
        self.context = ""
        self._max_frames = None if every is not None else max_frames
        self._stride = every or 1
        self._ticks = 0
        self._frames: list[_Frame] = []

        self._moving = [img.detach().float().to(self.device)[None, None] for img in images]
        self._points = torch.as_tensor(planes.points, device=self.device)
        self._flat = torch.as_tensor(planes.flat_indices(), device=self.device)
        nz, ny, nx = planes.grid_shape
        self._denom = torch.tensor(
            [max(nz - 1, 1), max(ny - 1, 1), max(nx - 1, 1)], device=self.device
        ).float()
        self._image_offset = torch.zeros(3, device=self.device)
        self._reference = None
        if reference is not None:
            if tuple(reference.shape) != planes.grid_shape:
                raise ValueError("reference must be on the planes' grid")
            flat = self._flat.to(reference.device)
            self._reference = reference.detach().float().reshape(-1)[flat].cpu().numpy()

    @property
    def n_rows(self) -> int:
        return len(self._moving)

    @property
    def n_frames(self) -> int:
        return len(self._frames)

    def set_frame(self, frame: FieldFrame) -> None:
        """Declare how subsequent captures' fields map onto the planes' grid."""
        self.frame = frame

    def set_images(
        self, images: Sequence[Tensor], offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> None:
        """Replace the rows' images -- for a tool that rescales, shifts, pads or
        motion-corrects its working copies after the recorder was built.

        ``offset`` is the (z, y, x) index in the new images of the planes grid's voxel 0,
        so a padded working copy can be shown with its padding reachable: tissue a warp
        pulls in from beyond the original grid then appears instead of black.
        """
        if len(images) != self.n_rows:
            raise ValueError(f"need {self.n_rows} images, got {len(images)}")
        shape = tuple(images[0].shape)
        if any(tuple(img.shape) != shape for img in images):
            raise ValueError("replacement images must share one grid")
        self._moving = [img.detach().float().to(self.device)[None, None] for img in images]
        self._image_offset = torch.tensor(offset, device=self.device).float()
        self._denom = torch.tensor([max(n - 1, 1) for n in shape], device=self.device).float()

    def set_context(self, text: str) -> None:
        """Text prepended to every later caption (a pyramid octave, a pass)."""
        self.context = text

    # -- capture -------------------------------------------------------------------

    def tick(self) -> bool:
        """Advance one iteration; True when this iteration should be captured."""
        due = self._ticks % self._stride == 0
        self._ticks += 1
        return due

    def _sample_field(self, field: Field, x: Tensor) -> Tensor:
        """Displacement (N, 3) z,y,x in planes-grid voxels at planes-grid points ``x``."""
        xd, yd, zd = field
        g = tuple(xd.shape)
        full = self.frame.full_shape or self.planes.grid_shape
        q = x + torch.tensor(self.frame.offset, device=x.device, dtype=x.dtype)
        coords, scales = [], []
        for axis in range(3):
            a, b = _level_affine(full[axis], g[axis], self.frame.mapping)
            c = q[:, axis] * a + b
            coords.append(2.0 * c / max(g[axis] - 1, 1) - 1.0 if g[axis] > 1 else c * 0.0)
            scales.append(1.0 / a)
        grid = torch.stack(coords[::-1], dim=-1)[None, None, None]
        stacked = torch.stack((xd, yd, zd))[None].to(device=x.device, dtype=x.dtype)
        sampled = F.grid_sample(
            stacked, grid, mode="bilinear", padding_mode="border", align_corners=True
        )[0, :, 0, 0]  # (3, N) as x, y, z
        return torch.stack(
            (sampled[2] * scales[0], sampled[1] * scales[1], sampled[0] * scales[2]), dim=1
        )

    def _follow(self, chain: Sequence[Field], x: Tensor) -> Tensor:
        """Push planes-grid points through ``chain`` in order: x <- x + u(x) per field."""
        for field in chain:
            x = x + self._sample_field(field, x)
        return x

    def _sample_moving(self, row: int, x: Tensor) -> Tensor:
        grid = (2.0 * (x + self._image_offset) / self._denom - 1.0).flip(-1)[None, None, None]
        return F.grid_sample(
            self._moving[row], grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).reshape(-1)

    def _jacobian(self, chain: Sequence[Field]) -> Tensor:
        """det of d(mapped point)/d(point) by central differences at the plane pixels."""
        cols = []
        for axis in range(3):
            step = torch.zeros(3, device=self.device)
            step[axis] = 1.0
            cols.append(
                (
                    self._follow(chain, self._points + step)
                    - self._follow(chain, self._points - step)
                )
                / 2.0
            )
        return torch.linalg.det(torch.stack(cols, dim=-1))

    def capture(
        self,
        chains: Sequence[Sequence[Field]],
        label: str = "",
        pinned: bool = False,
        modulate: bool = False,
    ) -> None:
        """Capture every row through its own chain of fields.

        Args:
            chains: One sequence of fields per row, applied in order (a composition:
                the point moves by the first field, then by the second at the moved
                point). Fields may sit on any level of the grid :attr:`frame` describes.
            label: Caption for the frame (after :attr:`context`).
            pinned: Never dropped by the frame budget, and held on screen.
            modulate: Scale intensity by the Jacobian determinant of the chain, as a
                distortion correction that conserves signal does (blipflip).
        """
        if len(chains) != self.n_rows:
            raise ValueError(f"need one chain per row ({self.n_rows}), got {len(chains)}")
        with torch.no_grad():
            rows = []
            for r, chain in enumerate(chains):
                vals = self._sample_moving(r, self._follow(chain, self._points))
                if modulate:
                    vals = vals * self._jacobian(chain)
                rows.append(vals)
            values = torch.stack(rows)
        text = f"{self.context}  {label}".strip() if self.context else label
        self._store(values, text, pinned)

    def capture_displacement(self, field: Field, label: str = "", pinned: bool = False) -> None:
        """Single-row shorthand: the source seen through one displacement field."""
        self.capture([[field]], label, pinned)

    def capture_identity(self, label: str = "", pinned: bool = True) -> None:
        """Capture every row unwarped (a movie's starting frame)."""
        with torch.no_grad():
            values = torch.stack([self._sample_moving(r, self._points) for r in range(self.n_rows)])
        text = f"{self.context}  {label}".strip() if self.context else label
        self._store(values, text, pinned)

    def _store(self, values: Tensor, label: str, pinned: bool) -> None:
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
        windows = [intensity_window(values[0, r]) for r in range(self.n_rows)]
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
        for i, (f, frame_values) in enumerate(zip(self._frames, values, strict=True)):
            label = f"{self.tool}  {f.label}".strip() if self.tool else f.label
            frame = compose_frame(
                [self.planes.split(row) for row in frame_values],
                views,
                size,
                windows,
                edges=edge_panels,
                edge_vmax=edge_vmax,
                edge_opacity=edge_opacity,
                label=label,
                row_labels=self.row_labels,
                progress=i / max(n - 1, 1),
            )
            repeats = hold_n if f.pinned else 1
            if i == n - 1:
                repeats = max(repeats, fps)
            out.extend([frame] * repeats)
        return write_movie(np.stack(out), movie_path(path, fmt), fps, fmt)
