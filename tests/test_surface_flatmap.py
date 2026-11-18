"""Equal-area flat views of a sphere: geometry, orientation and drawing."""

from __future__ import annotations

import numpy as np
import pytest

from fastfuncstuff.surface.flatmap import (
    _circular_face_values,
    center_direction,
    draw_flat_map,
    sphere_view,
)


def _sphere(n: int = 4000, radius: float = 100.0):
    """A triangulated sphere (Fibonacci points, convex hull), FreeSurfer-sized."""
    from scipy.spatial import ConvexHull

    k = np.arange(n) + 0.5
    polar = np.arccos(1 - 2 * k / n)
    azimuth = np.pi * (1 + 5**0.5) * k
    points = np.stack(
        [np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)], 1
    )
    return radius * points, ConvexHull(points).simplices


def _area(xy, faces):
    a, b, c = xy[faces[:, 0]], xy[faces[:, 1]], xy[faces[:, 2]]
    return 0.5 * np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0]).sum()


def test_the_projection_is_equal_area_and_keeps_the_cap():
    vertices, faces = _sphere()
    radius = 50.0
    view = sphere_view(vertices, faces, center=[0.3, -1.0, 0.2], radius_deg=radius)
    # Lambert keeps area: the kept faces cover as much of the page as of the unit
    # sphere (the faces are flat, so to the mesh's resolution).
    unit = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    a, b, c = (unit[view.faces[:, k]] for k in range(3))
    on_sphere = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    assert _area(view.xy, view.faces) == pytest.approx(on_sphere, rel=0.01)
    # And the rim is the disc holding the whole cap, 2 pi (1 - cos r) of area.
    cap = 2 * np.pi * (1 - np.cos(np.deg2rad(radius)))
    assert view.edge == pytest.approx(np.sqrt(cap / np.pi))
    assert np.all(np.linalg.norm(view.xy[view.keep], axis=1) <= view.edge + 1e-9)


def test_seen_from_behind_dorsal_is_up_and_the_subjects_right_is_right():
    vertices, faces = _sphere()
    view = sphere_view(vertices, faces, center=center_direction("posterior"), radius_deg=60)
    unit = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    near = view.keep
    # Superior points sit above the centre, the subject's right to its right.
    assert np.corrcoef(unit[near, 2], view.xy[near, 1])[0, 1] > 0.95
    assert np.corrcoef(unit[near, 0], view.xy[near, 0])[0, 1] > 0.95
    # Looking straight down there is no "superior": anterior goes up instead.
    top = sphere_view(vertices, faces, center=center_direction("dorsal"), radius_deg=60)
    assert np.corrcoef(unit[top.keep, 1], top.xy[top.keep, 1])[0, 1] > 0.95


def test_medial_and_lateral_depend_on_the_hemisphere():
    assert center_direction("medial", "lh")[0] == 1.0
    assert center_direction("lateral", "lh")[0] == -1.0
    assert center_direction("medial", "rh")[0] == -1.0
    # V1 is medial and below the pole on the registered sphere, far from posterior
    for hemi, medial in (("lh", 1), ("rh", -1)):
        v = center_direction("visual", hemi)
        assert np.isclose(np.linalg.norm(v), 1) and v[1] < 0 and v[2] < 0 and v[0] * medial > 0
        assert np.degrees(np.arccos(v @ center_direction("posterior"))) > 35
    with pytest.raises(ValueError, match="hemisphere"):
        center_direction("medial")
    with pytest.raises(ValueError, match="unknown"):
        center_direction("sideways")


def test_align_turns_a_set_horizontal_and_rotate_turns_the_view():
    vertices, faces = _sphere()
    unit = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    center = center_direction("posterior")
    # A band tilted 30 degrees in the view: its long axis must end up horizontal.
    plain = sphere_view(vertices, faces, center=center, radius_deg=60)
    tilt = np.deg2rad(30)
    along = plain.xy @ np.array([np.cos(tilt), np.sin(tilt)])
    across = plain.xy @ np.array([-np.sin(tilt), np.cos(tilt)])
    band = plain.keep & (np.abs(across) < 0.05) & (np.abs(along) < 0.6)
    aligned = sphere_view(vertices, faces, center=center, radius_deg=60, align=band)
    spread = np.cov(aligned.xy[band].T)
    assert spread[0, 0] > 20 * spread[1, 1]
    assert abs(spread[0, 1]) < 0.05 * spread[0, 0]
    # The turn is the smaller one, so up stays up.
    assert np.corrcoef(unit[aligned.keep, 2], aligned.xy[aligned.keep, 1])[0, 1] > 0.8
    turned = sphere_view(vertices, faces, center=center, radius_deg=60, rotate_deg=90)
    np.testing.assert_allclose(turned.xy[:, 0], -plain.xy[:, 1], atol=1e-12)
    np.testing.assert_allclose(turned.xy[:, 1], plain.xy[:, 0], atol=1e-12)


def test_cyclic_values_average_round_the_wrap():
    faces = np.array([[0, 1, 2]])
    # 350, 10 and 0 degrees average to 0, not to the arithmetic 120.
    got = _circular_face_values(np.array([350.0, 10.0, 0.0]), faces, 360.0)
    assert min(got[0], 360 - got[0]) < 1e-9


def test_draw_flat_map_draws_only_shown_faces(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vertices, faces = _sphere(1500)
    view = sphere_view(vertices, faces, center=center_direction("posterior"), radius_deg=45)
    values = np.degrees(np.arctan2(vertices[:, 2], vertices[:, 0])) % 360
    fig, ax = plt.subplots()
    mappable = draw_flat_map(
        ax,
        view,
        values,
        cmap="hsv",
        mask=vertices[:, 2] > 0,
        underlay=vertices[:, 0],
        borders=(vertices[:, 0] > 0).astype(int),
        period=360.0,
    )
    assert mappable is not None
    assert len(mappable.get_array()) < len(view.faces)  # only the dorsal half
    nothing = draw_flat_map(ax, view, np.full(len(vertices), np.nan))
    assert nothing is None
    fig.savefig(tmp_path / "map.png")
    plt.close(fig)
