"""Viewports: windows as addressable state.

The bugs guarded against here are the ones that made viewports necessary in the
first place -- a window's settings leaking into a global, a script that cannot
rebuild the layout it recorded, and a run of per-window commands collapsing into
one because the recorder only compared command *types*.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastfuncstuff.viewer import vocab
from fastfuncstuff.viewer.commands import Aspect, CommandBus, parse_script
from fastfuncstuff.viewer.layers import Layer
from fastfuncstuff.viewer.state import Plane, ViewerState
from fastfuncstuff.viewer.viewports import MAX_GRID, ViewKind, ViewportSet, clamp_grid
from fastfuncstuff.viewer.vocab import (
    CloseView,
    OpenView,
    SelectLayer,
    SetViewGeometry,
    SetViewGrid,
    SetViewPlane,
    SetViewSolo,
    SetViewTraces,
    SetZoom,
)


def _affine(step: float = 3.0) -> np.ndarray:
    a = np.eye(4)
    a[0, 0] = a[1, 1] = a[2, 2] = step
    a[:3, 3] = [-30.0, -40.0, -20.0]
    return a


def _layer(key: str, *, n_volumes: int = 1, time_linked: bool = False) -> Layer:
    return Layer(
        key=key,
        name=key,
        path=f"/tmp/{key}.nii.gz",
        shape=(20, 24, 16),
        n_volumes=n_volumes,
        affine=_affine(),
        time_linked=time_linked,
    )


def _bus(*layers: Layer) -> CommandBus:
    st = ViewerState()
    bus = vocab.install(CommandBus(st))
    for ly in layers:
        st.layers.add(ly)
        if st.grid is None:
            st.adopt_grid(ly.shape, ly.affine)
    return bus


# ---------------------------------------------------------------------------
# the set
# ---------------------------------------------------------------------------


def test_ids_are_per_kind_and_never_reused() -> None:
    s = ViewportSet()
    assert s.open(ViewKind.IMAGE, Plane.AXIAL).id == "V1"
    assert s.open(ViewKind.GRAPH, Plane.AXIAL).id == "G1"
    assert s.open(ViewKind.IMAGE, Plane.CORONAL).id == "V2"
    s.close("V1")
    # V1 is gone but the counter does not walk backwards onto it: a script that
    # still says SET_ZOOM V1 must not silently land on a different window.
    assert s.open(ViewKind.IMAGE, Plane.SAGITTAL).id == "V3"


def test_two_windows_of_the_same_plane_coexist() -> None:
    """The assumption the whole change exists to remove."""
    s = ViewportSet()
    a = s.open(ViewKind.IMAGE, Plane.AXIAL)
    b = s.open(ViewKind.IMAGE, Plane.AXIAL)
    assert a.id != b.id
    s.update(b.id, solo=True)
    assert not s.get(a.id).solo and s.get(b.id).solo


def test_grid_clamps_rather_than_raising() -> None:
    assert clamp_grid(0) == 1
    assert clamp_grid(MAX_GRID + 5) == MAX_GRID


def test_default_layout_is_three_images_and_no_graph() -> None:
    from fastfuncstuff.viewer.session import ViewerSession

    session = ViewerSession()
    try:
        session.default_layout()
        st = session.state
        assert [v.plane for v in st.viewports.images] == [
            Plane.AXIAL,
            Plane.SAGITTAL,
            Plane.CORONAL,
        ]
        assert st.viewports.graphs == []
        session.default_layout()  # idempotent: must not double the windows
        assert len(st.viewports) == 3
        # Through the bus, so a replayed script comes back with its windows.
        assert session.to_script().count("OPEN_VIEW") == 3
    finally:
        session.close()


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def test_open_and_close_round_trip_through_script_text() -> None:
    bus = _bus()
    bus.dispatch(OpenView("V1", "image", "coronal"))
    bus.dispatch(OpenView("G1", "graph", "axial"))
    bus.dispatch(SetViewGrid("G1", 3))
    text = bus.to_script()

    replayed = ViewerState()
    vocab.install(CommandBus(replayed)).dispatch_all(parse_script(text))
    assert replayed.viewports.ids == ["V1", "G1"]
    assert replayed.viewports.get("V1").plane is Plane.CORONAL
    assert replayed.viewports.get("G1").grid_n == 3


def test_reopening_an_existing_id_is_a_no_op() -> None:
    """Replaying a script over a live session must not raise."""
    bus = _bus()
    bus.dispatch(OpenView("V1", "image", "axial"))
    assert bus.dispatch(OpenView("V1", "image", "axial")) is Aspect.NOTHING
    assert len(bus.state.viewports) == 1


def test_setting_the_value_a_viewport_already_has_dirties_nothing() -> None:
    bus = _bus()
    bus.dispatch(OpenView("V1", "image", "axial"))
    assert bus.dispatch(SetZoom("V1", 2.0)) & Aspect.SLICES
    assert bus.dispatch(SetZoom("V1", 2.0)) is Aspect.NOTHING


def test_zoom_is_per_window() -> None:
    bus = _bus()
    bus.dispatch(OpenView("V1", "image", "axial"))
    bus.dispatch(OpenView("V2", "image", "axial"))
    bus.dispatch(SetZoom("V1", 3.0))
    assert bus.state.viewports.get("V1").zoom == 3.0
    assert bus.state.viewports.get("V2").zoom == 1.0


def test_closing_an_absent_view_is_quiet() -> None:
    bus = _bus()
    assert bus.dispatch(CloseView("V9")) is Aspect.NOTHING


def test_a_command_for_an_open_view_that_is_gone_raises() -> None:
    bus = _bus()
    with pytest.raises(KeyError):
        bus.dispatch(SetViewPlane("V9", "axial"))


# ---------------------------------------------------------------------------
# the collapse bug this change would otherwise have introduced
# ---------------------------------------------------------------------------


def test_geometry_for_several_windows_survives_recording() -> None:
    """Tiling emits one line per window and every one of them matters.

    Collapsing on command *type* alone would record only the last window's
    rectangle, and replay would stack every window on top of one another.
    """
    bus = _bus()
    for vid in ("V1", "V2", "V3"):
        bus.dispatch(OpenView(vid, "image", "axial"))
    for i, vid in enumerate(("V1", "V2", "V3")):
        bus.dispatch(SetViewGeometry(vid, i * 400, 0, 400, 400))
    lines = [ln for ln in bus.to_script().splitlines() if ln.startswith("SET_VIEW_GEOMETRY")]
    assert len(lines) == 3


def test_a_drag_on_one_window_still_collapses() -> None:
    bus = _bus()
    bus.dispatch(OpenView("V1", "image", "axial"))
    for z in (1.2, 1.4, 1.6, 1.8):
        bus.dispatch(SetZoom("V1", z))
    lines = [ln for ln in bus.to_script().splitlines() if ln.startswith("SET_ZOOM")]
    assert lines == ["SET_ZOOM V1 1.8"]


# ---------------------------------------------------------------------------
# selection and solo
# ---------------------------------------------------------------------------


def test_selection_is_state_so_solo_has_something_to_draw() -> None:
    bus = _bus(_layer("L1"), _layer("L2"))
    assert bus.state.selected_layer().key == "L2"  # top of the stack by default
    dirty = bus.dispatch(SelectLayer("L1"))
    assert dirty & Aspect.SLICES  # a soloed window redraws on a selection change
    assert bus.state.selected_layer().key == "L1"


def test_selecting_a_layer_that_is_not_there_raises() -> None:
    bus = _bus(_layer("L1"))
    with pytest.raises(KeyError):
        bus.dispatch(SelectLayer("L9"))


def test_selection_falls_back_when_the_selected_layer_is_removed() -> None:
    bus = _bus(_layer("L1"), _layer("L2"))
    bus.dispatch(SelectLayer("L1"))
    bus.state.layers.remove("L1")
    assert bus.state.selected_layer().key == "L2"


def test_solo_is_per_window() -> None:
    bus = _bus(_layer("L1"))
    bus.dispatch(OpenView("V1", "image", "axial"))
    bus.dispatch(OpenView("V2", "image", "axial"))
    bus.dispatch(SetViewSolo("V2", True))
    assert not bus.state.viewports.get("V1").solo
    assert bus.state.viewports.get("V2").solo


# ---------------------------------------------------------------------------
# graph trace selection
# ---------------------------------------------------------------------------


def test_traces_parse_from_a_comma_list_and_empty_means_all() -> None:
    bus = _bus()
    bus.dispatch(OpenView("G1", "graph", "axial"))
    bus.dispatch(SetViewTraces("G1", "L2,L3"))
    assert bus.state.viewports.get("G1").traces == ("L2", "L3")
    bus.dispatch(SetViewTraces("G1", ""))
    assert bus.state.viewports.get("G1").traces == ()


def test_a_graph_never_offers_a_three_d_layer() -> None:
    """An anatomy has no time course; listing it invites 'why is it hidden'."""
    from fastfuncstuff.viewer.session import ViewerSession

    session = ViewerSession()
    try:
        session.state.layers.add(_layer("anat"))
        session.state.layers.add(_layer("bold", n_volumes=120, time_linked=True))
        assert [ly.key for ly in session.graph_layers()] == ["bold"]
    finally:
        session.close()


def test_a_graph_keeps_only_the_traces_it_names() -> None:
    from fastfuncstuff.viewer.session import ViewerSession

    session = ViewerSession()
    try:
        st = session.state
        st.layers.add(_layer("anat"))
        st.layers.add(_layer("bold", n_volumes=120, time_linked=True))
        st.layers.add(_layer("clean", n_volumes=120, time_linked=True))
        vid = session.open_view(ViewKind.GRAPH, Plane.AXIAL)
        assert [ly.key for ly in session.traces_for(st.viewports.get(vid))] == ["bold", "clean"]
        session.do(SetViewTraces(vid, "bold"))
        assert [ly.key for ly in session.traces_for(st.viewports.get(vid))] == ["bold"]
        # A key that no longer names a layer is dropped, not an error: a
        # viewport outlives the layers it was pointed at.
        st.layers.remove("bold")
        assert session.traces_for(st.viewports.get(vid)) == []
    finally:
        session.close()
