"""Equal-area flat views of a spherical surface, and per-vertex maps drawn on them.

A retinotopic map is read on the sphere (FreeSurfer's ``?h.sphere``): every vertex
keeps its neighbours, nothing is cut, and a Lambert azimuthal equal-area projection
about one point flattens a cap of it without distorting area -- a cap of angular
radius ``r`` lands on a disc of radius ``2 sin(r / 2)``, whatever the centre. The
view is oriented as seen from outside the brain, dorsal up, so both hemispheres and
every subject come out the same way round without visual-area labels; given labels
(or any vertex set), it can instead be rotated so that set's long axis runs
left-right, the way V1's calcarine axis is usually drawn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CENTER_PRESETS", "FlatView", "center_direction", "draw_flat_map", "sphere_view"]

#: Named view centres, as directions on a FreeSurfer sphere (x right, y anterior,
#: z superior); ``lateral`` and ``medial`` depend on the hemisphere.
CENTER_PRESETS = ("posterior", "anterior", "dorsal", "ventral", "lateral", "medial")


@dataclass(frozen=True)
class FlatView:
    xy: np.ndarray  # (V, 2) projected coordinates of every vertex
    faces: np.ndarray  # the faces wholly inside the view
    keep: np.ndarray  # (V,) bool: inside the view
    edge: float  # projected radius of the view's rim, 2 sin(radius / 2)


def center_direction(name: str, hemi: str | None = None) -> np.ndarray:
    """Unit direction of a :data:`CENTER_PRESETS` view on a FreeSurfer sphere."""
    if name in ("lateral", "medial"):
        if hemi not in ("lh", "rh"):
            raise ValueError(f"a {name} view needs the hemisphere (lh or rh)")
        outward = -1.0 if hemi == "lh" else 1.0
        x = outward if name == "lateral" else -outward
        return np.array([x, 0.0, 0.0])
    table = {
        "posterior": (0.0, -1.0, 0.0),
        "anterior": (0.0, 1.0, 0.0),
        "dorsal": (0.0, 0.0, 1.0),
        "ventral": (0.0, 0.0, -1.0),
    }
    if name not in table:
        raise ValueError(f"unknown view centre {name!r}; one of {', '.join(CENTER_PRESETS)}")
    return np.array(table[name])


def sphere_view(
    vertices: np.ndarray,
    faces: np.ndarray,
    center: np.ndarray,
    radius_deg: float = 42.0,
    rotate_deg: float = 0.0,
    align: np.ndarray | None = None,
    up: np.ndarray | None = None,
) -> FlatView:
    """Project the sphere about ``center`` and keep the cap within ``radius_deg`` of it.

    ``center`` is any direction from the sphere's centre. The view looks at it from
    outside, with ``up`` (default superior, +z; anterior when looking straight down
    or up) pointing up the page and the subject's right to the right when seen from
    behind. ``align`` (a boolean vertex mask) rotates the view so that set's
    principal axis is horizontal, by the smaller of the two possible turns, so up
    stays roughly up. ``rotate_deg`` then turns it counter-clockwise.
    """
    if not 0 < radius_deg <= 180:
        raise ValueError(f"radius_deg must be in (0, 180], got {radius_deg}")
    u = np.asarray(vertices, np.float64)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    c = np.asarray(center, np.float64)
    c = c / np.linalg.norm(c)
    up_dir = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, np.float64)
    tangent_up = up_dir - (up_dir @ c) * c
    if np.linalg.norm(tangent_up) < 1e-6:  # looking along up: anterior is up instead
        anterior = np.array([0.0, 1.0, 0.0])
        tangent_up = anterior - (anterior @ c) * c
    e_up = tangent_up / np.linalg.norm(tangent_up)
    e_right = np.cross(-c, e_up)
    theta = np.arccos(np.clip(u @ c, -1.0, 1.0))
    a, b = u @ e_right, u @ e_up
    norm = np.hypot(a, b)
    rho = 2.0 * np.sin(theta / 2.0)
    scale = np.divide(rho, norm, out=np.zeros_like(rho), where=norm > 0)
    xy = np.stack([a * scale, b * scale], axis=1)
    turn = np.deg2rad(rotate_deg)
    keep = theta <= np.deg2rad(radius_deg)
    if align is not None:
        chosen = np.asarray(align, bool) & keep
        if chosen.sum() >= 3:
            points = xy[chosen] - xy[chosen].mean(axis=0)
            values, vectors = np.linalg.eigh(np.cov(points.T))
            axis = vectors[:, np.argmax(values)]
            # The axis has no sign; take the turn under 90 degrees.
            turn -= np.arctan(axis[1] / axis[0]) if axis[0] != 0 else np.pi / 2
    if turn:
        cos, sin = np.cos(turn), np.sin(turn)
        xy = xy @ np.array([[cos, sin], [-sin, cos]])
    faces = np.asarray(faces)
    return FlatView(
        xy=xy,
        faces=faces[keep[faces].all(axis=1)],
        keep=keep,
        edge=float(2.0 * np.sin(np.deg2rad(radius_deg) / 2.0)),
    )


def _circular_face_values(values: np.ndarray, faces: np.ndarray, period: float) -> np.ndarray:
    """Each face's circular mean of its vertices' values, in ``[0, period)``."""
    phase = 2.0 * np.pi * values[faces] / period
    mean = np.arctan2(np.sin(phase).mean(axis=1), np.cos(phase).mean(axis=1))
    return (mean % (2.0 * np.pi)) * period / (2.0 * np.pi)


def draw_flat_map(
    ax,
    view: FlatView,
    values: np.ndarray,
    *,
    cmap="viridis",
    norm=None,
    mask: np.ndarray | None = None,
    underlay: np.ndarray | None = None,
    borders: np.ndarray | None = None,
    period: float | None = None,
    title: str = "",
    border_names: dict[int, str] | None = None,
) -> object | None:
    """Draw ``values`` on the view: a sulcal underlay, the data, label borders, the rim.

    ``underlay`` is a FreeSurfer-style morph value (sulc or curv, positive in
    sulci), drawn as two greys. ``mask`` limits which vertices are coloured; a
    face is drawn only when all three of its vertices are. ``borders`` is an
    integer label per vertex; each nonzero label gets an outline, and its name from
    ``border_names`` at its in-view vertex furthest from any other label. ``period``
    marks the values as cyclic (polar angle: 360): faces are then coloured flat by
    their circular mean, because smooth shading would sweep through the whole
    colour wheel between 350 and 10. Returns the data's mappable (for a colorbar),
    or None when nothing was drawn.
    """
    from matplotlib.tri import Triangulation

    xy = view.xy
    triangulation = Triangulation(xy[:, 0], xy[:, 1], view.faces)
    if underlay is not None:
        grey = np.where(np.asarray(underlay) > 0, 0.55, 0.8)
        ax.tripcolor(triangulation, grey, cmap="gray", vmin=0, vmax=1, shading="gouraud")
    data = np.asarray(values, np.float64).copy()
    show = np.isfinite(data) if mask is None else (np.asarray(mask, bool) & np.isfinite(data))
    drawn = view.faces[show[view.faces].all(axis=1)]
    mappable = None
    if len(drawn):
        shown = Triangulation(xy[:, 0], xy[:, 1], drawn)
        data[~show] = 0.0
        if period is not None:
            mappable = ax.tripcolor(
                shown,
                facecolors=_circular_face_values(data, drawn, period),
                cmap=cmap,
                norm=norm,
                shading="flat",
            )
        else:
            mappable = ax.tripcolor(shown, data, cmap=cmap, norm=norm, shading="gouraud")
    if borders is not None:
        labels = np.asarray(borders)
        for label in np.unique(labels[view.keep]):
            if label == 0:
                continue
            ax.tricontour(
                triangulation,
                (labels == label).astype(float),
                levels=[0.5],
                colors="k",
                linewidths=0.6,
            )
            name = (border_names or {}).get(int(label))
            inside = np.flatnonzero((labels == label) & view.keep)
            if name and inside.size >= 3:
                import matplotlib.patheffects as effects
                from scipy.spatial import KDTree

                # The point deepest inside the area: a ring's centre (V2 around
                # V1) lies in its neighbour, so the centroid would mislabel.
                outside = np.flatnonzero((labels != label) & view.keep)
                pts = xy[inside]
                depth = KDTree(xy[outside]).query(pts)[0] if outside.size else np.zeros(len(pts))
                at = pts[np.argmax(depth)]
                ax.text(*at, name, fontsize=6, ha="center", va="center",
                        path_effects=[effects.withStroke(linewidth=1.5, foreground="w")])  # fmt: skip
    rim = np.linspace(0, 2 * np.pi, 361)
    ax.plot(view.edge * np.cos(rim), view.edge * np.sin(rim), color="0.4", linewidth=0.6)
    limit = view.edge * 1.02
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9)
    return mappable
