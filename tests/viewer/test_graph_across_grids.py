"""Reading a layer at the crosshair when it is not on the display grid.

The display grid belongs to the bottom of the stack. Everything that reads a
layer's array at "where the crosshair is" therefore has to convert, and for a
long time most of it did not -- it indexed the layer with the display's indices.

That has two failure modes and the quiet one is worse. Near the far edge the
index is out of bounds and the trace vanishes, which reads as "the graph broke".
Near the origin it is *in* bounds, and you get a confident plot of a voxel three
times too close to the corner.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.viewer.session import ViewerSession
from fastfuncstuff.viewer.vocab import Load, SetIJK, SetLayerVisible, SetMode

nib = pytest.importorskip("nibabel")
CPU = torch.device("cpu")

#: Both datasets share this corner, so a display voxel's millimetres are
#: exactly 3x its index in the run -- which makes the expected answer arithmetic
#: rather than something the test has to be told.
ORIGIN = (-24.0, -27.0, -21.0)


def _save(path, data, step):
    aff = np.diag([step, step, step, 1.0])
    aff[:3, 3] = ORIGIN
    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), aff)
    if data.ndim == 4:
        img.header["pixdim"][4] = 2.0
        img.header.set_xyzt_units("mm", "sec")
    nib.save(img, str(path))
    return path


@pytest.fixture
def run_values():
    """A run whose value names its own voxel, so a trace identifies where it came from."""
    v = np.zeros((16, 18, 14, 8), dtype=np.float32)
    idx = np.indices((16, 18, 14))
    v[..., :] = (idx[0] * 10000 + idx[1] * 100 + idx[2])[..., None]
    return v


@pytest.fixture
def session(tmp_path, run_values):
    """1 mm anatomical at the bottom, 3 mm run above it."""
    rng = np.random.default_rng(2)
    _save(tmp_path / "anat.nii.gz", rng.random((48, 54, 42)) * 100, 1.0)
    _save(tmp_path / "bold.nii.gz", run_values, 3.0)
    s = ViewerSession(device=CPU)
    s.do(Load(str(tmp_path / "anat.nii.gz")))
    s.do(Load(str(tmp_path / "bold.nii.gz")))
    s.store.ensure_ram(s.state.layers.layers[-1].key)
    yield s
    s.close()


def _run_key(session):
    return next(ly.key for ly in session.state.layers if ly.n_volumes > 1)


def _encoded(ijk):
    return float(ijk[0] * 10000 + ijk[1] * 100 + ijk[2])


# ---------------------------------------------------------------------------
# the loud failure
# ---------------------------------------------------------------------------


def test_the_trace_survives_a_crosshair_past_the_run_s_index_range(session):
    """Display voxel (40, 45, 35) is far outside a (16, 18, 14) run."""
    session.do(SetIJK(40, 45, 35))
    values = session.timeseries(_run_key(session))
    assert values.size == 8, "this is the 'my graph disappeared' report"


def test_the_trace_is_there_everywhere_the_run_covers(session):
    """The 3 mm run spans 45 mm; the 1 mm anatomy's first 46 voxels are inside it."""
    key = _run_key(session)
    for ijk in [(0, 0, 0), (45, 51, 39), (45, 0, 39), (24, 27, 21)]:
        session.do(SetIJK(*ijk))
        assert session.timeseries(key).size == 8, ijk


def test_a_crosshair_outside_the_run_s_field_of_view_still_gives_nothing(session):
    """Converting must not turn "not measured here" into an edge-clamped lie.

    The anatomy is 48 mm across and the run only 45, so the last anatomical
    slices are over nothing. Empty is the honest answer and it is a different
    thing from the bug above, where the voxel existed and was not found.
    """
    session.do(SetIJK(47, 53, 41))
    assert session.timeseries(_run_key(session)).size == 0


# ---------------------------------------------------------------------------
# the quiet one
# ---------------------------------------------------------------------------


def test_the_trace_comes_from_the_voxel_the_crosshair_is_actually_on(session):
    """In bounds and wrong is worse than out of bounds and empty."""
    session.do(SetIJK(9, 12, 6))  # 1 mm anat index -> 3 mm run index (3, 4, 2)
    values = session.timeseries(_run_key(session))
    assert values[0] == pytest.approx(_encoded((3, 4, 2)))
    assert values[0] != pytest.approx(_encoded((9, 12, 6))), "the old, plausible answer"


def test_it_agrees_with_the_readout(session):
    """The number under the crosshair and the trace it plots are one voxel."""
    from fastfuncstuff.viewer.slicing import voxel_value

    key = _run_key(session)
    session.do(SetIJK(30, 15, 24))
    layer = session.state.layers.get(key)
    shown = voxel_value(
        session.display_volume(key), session.state.grid, layer.affine, session.state.crosshair
    )
    assert session.timeseries(key)[0] == pytest.approx(shown)


def test_a_layer_on_the_display_grid_is_unaffected(session):
    """The conversion must be identity when there is nothing to convert."""
    anat = session.state.layers.base
    session.do(SetIJK(9, 12, 6))
    assert session.layer_voxel(anat.affine, (9, 12, 6)) == (9, 12, 6)


# ---------------------------------------------------------------------------
# it does not depend on the run being drawn
# ---------------------------------------------------------------------------


def test_an_unticked_run_still_graphs(session):
    """The gesture: fit on the run, untick it, keep the time course."""
    key = _run_key(session)
    session.do(SetLayerVisible(key, on=False))
    session.do(SetIJK(9, 12, 6))
    assert key in [ly.key for ly in session.graph_layers()]
    assert session.timeseries(key)[0] == pytest.approx(_encoded((3, 4, 2)))


# ---------------------------------------------------------------------------
# and the modes' own lines
# ---------------------------------------------------------------------------


def test_instacorr_contributes_its_source_line_across_grids(session):
    session.do(SetMode("instacorr"))
    session.refresh_mode()
    session.do(SetIJK(36, 42, 30))
    labels = {t.ident for t in session.mode_series()}
    assert "source" in labels, "the run the correlation came from must stay plottable"
    source = next(t for t in session.mode_series() if t.ident == "source")
    assert source.values[0] == pytest.approx(_encoded((12, 14, 10)))


def test_instaglm_decomposes_at_the_right_voxel_across_grids(session):
    """The user's report: the run went into the model, the graph showed nothing.

    (36, 42, 30) in the anatomy is (12, 14, 10) in the run -- inside its brain
    mask, which the low-index corner of this synthetic ramp is not.
    """
    session.do(SetMode("instaglm"))
    session.refresh_mode()
    assert session.mode._fit is not None, "the fit itself was never the problem"
    session.do(SetIJK(36, 42, 30))
    traces = session.mode_series()
    assert traces, "a fit with no lines reads as the model having failed"
    data = next((t for t in traces if t.ident == "data"), None)
    assert data is not None and data.values.size == 8
    assert data.values[0] == pytest.approx(_encoded((12, 14, 10)))


def test_instaglm_at_the_old_indices_would_have_found_nothing(session):
    """Pins the failure, not just the fix: the display index is out of the mask."""
    session.do(SetMode("instaglm"))
    session.refresh_mode()
    prepared = session.mode._fit.prepared
    assert prepared.row((36, 42, 30)) < 0, "the display-grid index, used directly"
    assert prepared.row(session.mode._to_run((36, 42, 30))) >= 0
