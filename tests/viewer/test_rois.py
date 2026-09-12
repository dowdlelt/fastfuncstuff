"""ROI sets: the properties that decide whether a parcellation means anything.

An ROI set is the thing a correlation matrix's rows *are*, so a mistake here is
not a wrong pixel -- it is a matrix row attributed to the wrong region, which
looks exactly like a result. These pin the parts that could go wrong silently:
that 4-D frames collapsing to labels reports what it lost, that the label table
routes agree, and that averaging by scatter matches averaging by mask.
"""

from __future__ import annotations

import numpy as np
import torch

from fastfuncstuff.io.labels import parse_atlas_points, parse_dtable, read_label_file
from fastfuncstuff.viewer.rois import (
    looks_like_labels,
    roi_color,
    roi_means,
    rois_from_frames,
    rois_from_labels,
)


def test_labels_are_measured_once_and_correctly():
    labels = np.zeros((6, 6, 6), dtype=np.int32)
    labels[0:2, 0:2, 0:2] = 3  # 8 voxels, centre (0.5, 0.5, 0.5)
    labels[4, 4, 4] = 7
    rois = rois_from_labels(labels)
    assert rois.indices == (3, 7)
    assert rois.find(3).n_voxels == 8
    assert rois.find(3).center == (0.5, 0.5, 0.5)
    assert rois.find(7).center_ijk == (4, 4, 4)
    assert rois.at((4, 4, 4)).index == 7
    assert rois.at((5, 5, 5)) is None


def test_frames_report_the_overlap_they_had_to_resolve():
    """A voxel in two frames lands in the first, and the count says so.

    Silently resolving overlap is how a parcellation with a shared border
    produces a correlation matrix nobody can reproduce.
    """
    data = np.zeros((4, 4, 4, 2), dtype=np.float32)
    data[0:2, :, :, 0] = 1.0
    data[1:3, :, :, 1] = 1.0  # the x==1 plane is in both
    rois = rois_from_frames(data, frame_names=("front", "back"))
    assert rois.overlaps == 16
    assert rois.find(1).n_voxels == 32  # kept the whole of frame 0
    assert rois.find(2).n_voxels == 16  # frame 1 lost the contested plane
    assert [r.name for r in rois] == ["front", "back"]


def test_an_anatomy_does_not_read_as_an_atlas():
    rng = np.random.default_rng(0)
    assert not looks_like_labels(rng.normal(size=(8, 8, 8)))  # continuous
    assert not looks_like_labels(rng.integers(0, 20000, size=(40, 40, 40)))  # too many
    assert looks_like_labels(rng.integers(0, 12, size=(8, 8, 8)))


def test_every_roi_gets_its_own_colour():
    colors = {roi_color(i) for i in range(64)}
    assert len(colors) == 64


def test_scatter_means_match_masked_means():
    """The fast path and the obvious path must agree, or the matrix is fiction."""
    rng = np.random.default_rng(1)
    rows = torch.as_tensor(rng.normal(size=(200, 30)).astype(np.float32))
    labels = torch.as_tensor(rng.integers(0, 4, size=200))  # 0 = outside
    means = roi_means(rows, labels, (1, 2, 3))
    for position, value in enumerate((1, 2, 3)):
        expected = rows[labels == value].mean(0)
        assert torch.allclose(means[position], expected, atol=1e-5)


def test_the_label_table_formats_agree(tmp_path):
    """Four spellings of the same table, one answer."""
    want = {1: "Left-Thalamus", 2: "Right-Thalamus"}

    dtable = parse_dtable('"1" "Left-Thalamus"  "2" "Right-Thalamus"')
    assert {k: v.name for k, v in dtable.items()} == want

    escaped = parse_dtable(
        "&quot;1&quot; &quot;Left-Thalamus&quot; &quot;2&quot; &quot;Right-Thalamus&quot;"
    )
    assert {k: v.name for k, v in escaped.items()} == want

    points = parse_atlas_points(
        '<ATLAS_POINT STRUCT="Left-Thalamus" VAL="1" />'
        '<ATLAS_POINT STRUCT="Right-Thalamus" VAL="2" />'
    )
    assert {k: v.name for k, v in points.items()} == want

    tsv = tmp_path / "atlas_dseg.tsv"
    tsv.write_text("index\tname\tcolor\n1\tLeft-Thalamus\t#ff0000\n2\tRight-Thalamus\t0 255 0\n")
    read = read_label_file(tsv)
    assert {k: v.name for k, v in read.items()} == want
    assert read[1].color == (255, 0, 0)
    assert read[2].color == (0, 255, 0)

    lut = tmp_path / "atlas.txt"
    lut.write_text(
        "# a FreeSurfer LUT\n0 Unknown 0 0 0 0\n1 Left-Thalamus 255 0 0 0\n"
        "2 Right-Thalamus 0 255 0 0\n"
    )
    assert {k: v.name for k, v in read_label_file(lut).items()} == want


def test_background_is_never_a_region():
    assert 0 not in parse_dtable('"0" "Unknown" "1" "Thing"')


# ---------------------------------------------------------------------------
# through a session: the guess, the names, and the pixels
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from fastfuncstuff.viewer.compose import render_plane  # noqa: E402
from fastfuncstuff.viewer.session import ViewerSession  # noqa: E402
from fastfuncstuff.viewer.state import Plane  # noqa: E402
from fastfuncstuff.viewer.vocab import SetIJK, SetLayerRoi  # noqa: E402

nib = pytest.importorskip("nibabel")


def _atlas(tmp_path, name="atlas.nii.gz"):
    labels = np.zeros((10, 10, 6), dtype=np.float32)
    labels[1:4, 1:4, 1:4] = 1
    labels[6:9, 6:9, 1:4] = 2
    img = nib.Nifti1Image(labels, np.diag([3.0, 3.0, 3.0, 1.0]))
    path = tmp_path / name
    nib.save(img, str(path))
    return path


def test_an_atlas_lands_already_recognised_and_already_named(tmp_path):
    sidecar = tmp_path / "atlas.tsv"
    sidecar.write_text("index\tname\n1\tputamen\n2\tcaudate\n")
    session = ViewerSession(device=torch.device("cpu"), record=False)
    key = session.load(_atlas(tmp_path))

    assert session.state.layers.get(key).roi is True
    rois = session.roi_set(key)
    assert [r.name for r in rois] == ["putamen", "caudate"]
    assert rois.find(1).n_voxels == 27

    session.do(SetIJK(2, 2, 2))
    layer, roi = session.roi_at()
    assert (layer.key, roi.name) == (key, "putamen")
    assert session.roi_at((0, 0, 0)) is None
    session.close()


def test_turning_the_guess_off_puts_the_layer_back_on_a_colour_scale(tmp_path):
    """The flag is what routes rendering, and the cached description follows it."""
    session = ViewerSession(device=torch.device("cpu"), record=False)
    key = session.load(_atlas(tmp_path))
    assert session.roi_palette(key) is not None

    session.do(SetLayerRoi(key, False))
    assert session.roi_set(key) is None
    assert session.roi_palette(key) is None

    session.do(SetLayerRoi(key, True))
    assert session.roi_palette(key) is not None
    session.close()


def test_two_regions_draw_in_two_colours(tmp_path):
    """An atlas coloured by a continuous LUT is a picture of its label numbers.

    Region 2 would be twice as bright as region 1 and nothing would look wrong,
    which is exactly why this is pinned in pixels rather than in the palette.
    """
    session = ViewerSession(device=torch.device("cpu"), record=False)
    key = session.load(_atlas(tmp_path))
    session.do(SetIJK(2, 2, 2))
    pane = render_plane(session, Plane.AXIAL)
    assert pane is not None
    rgba = pane.rgba.numpy()

    rois = session.roi_set(key)
    drawn = {tuple(int(c) for c in px[:3]) for px in rgba.reshape(-1, 4)}
    # Two regions on this slice plus the ground they sit on: two colours, and
    # no ramp between them.
    assert drawn == {(0, 0, 0), *(roi.color for roi in rois)}
    session.close()
