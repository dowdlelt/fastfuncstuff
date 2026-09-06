"""Command bus, layer stack, and vocabulary.

The bug these guard against is a recorder that lies: if a command does not
round-trip through script text, or a handler mutates state without reporting
what it dirtied, then a recorded session replays into a different view than the
one the person actually built.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastfuncstuff.viewer import vocab
from fastfuncstuff.viewer.commands import (
    Aspect,
    CommandBus,
    parse_script,
    registered_names,
    resolve,
)
from fastfuncstuff.viewer.layers import AlphaMode, Layer, LayerStack, SignMode
from fastfuncstuff.viewer.state import ViewerState
from fastfuncstuff.viewer.vocab import (
    AddLayer,
    MoveLayer,
    SetAlpha,
    SetIJK,
    SetIndex,
    SetLayerOpacity,
    SetRange,
    SetSeed,
    SetSign,
    SetThreshold,
    SetXYZ,
)


def _affine(step: float = 3.0) -> np.ndarray:
    a = np.eye(4)
    a[0, 0] = a[1, 1] = a[2, 2] = step
    a[:3, 3] = [-30.0, -40.0, -20.0]
    return a


def _layer(key: str = "L1", *, shape=(20, 24, 16), n_volumes=1) -> Layer:
    return Layer(
        key=key,
        name=key,
        path=f"/tmp/{key}.nii.gz",
        shape=shape,
        n_volumes=n_volumes,
        affine=_affine(),
    )


def _session(*, layers: list[Layer] | None = None) -> CommandBus:
    st = ViewerState()
    opened: list[str] = []

    def open_layer(path: str, key: str) -> Layer:
        opened.append(path)
        return Layer(
            key=key,
            name=path.rsplit("/", 1)[-1],
            path=path,
            shape=(20, 24, 16),
            n_volumes=50,
            affine=_affine(),
        )

    bus = vocab.install(CommandBus(st), open_layer=open_layer)
    for ly in layers or []:
        st.layers.add(ly)
        if st.grid is None:
            st.adopt_grid(ly.shape, ly.affine)
    return bus


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------


def test_every_registered_command_round_trips_through_script_text():
    """A command that cannot round-trip is invisible to replay."""
    samples = [
        SetIJK(1, 2, 3),
        SetXYZ(-4.5, 0.0, 12.25),
        SetIndex(7),
        SetThreshold("L1", 2.5),
        SetRange("L1", -3.0, 3.0),
        SetSign("L1", "pos"),
        SetAlpha("L1", "quadratic"),
        AddLayer("/data/stats.nii.gz", "L2"),
        MoveLayer("L2", 0),
        SetSeed(5, 6, 7),
    ]
    for cmd in samples:
        line = cmd.to_line()
        back = parse_script(line)[0]
        assert back == cmd, f"{cmd.name} did not round-trip: {line!r}"


def test_paths_with_spaces_survive_the_script():
    cmd = AddLayer("/data/my study/sub 01.nii.gz", "L9")
    assert parse_script(cmd.to_line())[0] == cmd


def test_resolve_is_case_insensitive_and_rejects_unknown():
    assert resolve("set_ijk") is SetIJK
    with pytest.raises(KeyError):
        resolve("NOT_A_COMMAND")


def test_registry_covers_the_vocabulary():
    names = registered_names()
    for expected in ("SET_IJK", "ADD_LAYER", "SET_THRESHOLD", "SET_SEED"):
        assert expected in names


def test_script_ignores_blanks_and_comments():
    text = "# a recorded session\n\nSET_IJK 1 2 3\n\n# trailing note\n"
    assert parse_script(text) == [SetIJK(1, 2, 3)]


def test_bad_script_line_names_its_line_number():
    with pytest.raises(ValueError, match="line 2"):
        parse_script("SET_IJK 1 2 3\nSET_IJK 1 2\n")


# ---------------------------------------------------------------------------
# dispatch and dirty-tracking
# ---------------------------------------------------------------------------


def test_no_op_command_dirties_nothing():
    """Dragging back onto the same voxel must not trigger a repaint."""
    bus = _session(layers=[_layer()])
    bus.dispatch(SetIJK(5, 5, 5))
    assert bus.dispatch(SetIJK(5, 5, 5)) is Aspect.NOTHING


def test_crosshair_is_clamped_to_the_display_grid():
    bus = _session(layers=[_layer(shape=(20, 24, 16))])
    bus.dispatch(SetIJK(999, -4, 8))
    assert bus.state.crosshair == (19, 0, 8)


def test_mm_and_ijk_addressing_agree():
    bus = _session(layers=[_layer()])
    bus.dispatch(SetIJK(6, 7, 8))
    mm = bus.state.crosshair_mm
    assert mm is not None
    bus.dispatch(SetIJK(0, 0, 0))
    bus.dispatch(SetXYZ(*mm))
    assert bus.state.crosshair == (6, 7, 8)


def test_time_index_is_clamped_to_the_longest_layer():
    bus = _session(layers=[_layer(n_volumes=10)])
    bus.dispatch(SetIndex(999))
    assert bus.state.time_index == 9


def test_listeners_see_what_actually_changed():
    bus = _session(layers=[_layer()])
    seen: list[Aspect] = []
    bus.subscribe(lambda cmd, dirty: seen.append(dirty))
    bus.dispatch(SetIJK(3, 3, 3))
    bus.dispatch(SetIJK(3, 3, 3))
    assert seen[0] & Aspect.CROSSHAIR
    assert seen[1] is Aspect.NOTHING


def test_unsubscribe_stops_delivery():
    bus = _session(layers=[_layer()])
    seen: list[Aspect] = []
    off = bus.subscribe(lambda cmd, dirty: seen.append(dirty))
    bus.dispatch(SetIJK(1, 1, 1))
    off()
    bus.dispatch(SetIJK(2, 2, 2))
    assert len(seen) == 1


def test_opacity_is_clamped():
    bus = _session(layers=[_layer()])
    bus.dispatch(SetLayerOpacity("L1", 5.0))
    assert bus.state.layers.get("L1").opacity == 1.0


def test_range_accepts_reversed_bounds():
    bus = _session(layers=[_layer()])
    bus.dispatch(SetRange("L1", 3.0, -3.0))
    layer = bus.state.layers.get("L1")
    assert (layer.range_lo, layer.range_hi) == (-3.0, 3.0)


def test_enum_valued_commands_reject_nonsense():
    bus = _session(layers=[_layer()])
    with pytest.raises(ValueError):
        bus.dispatch(SetSign("L1", "sideways"))


def test_alpha_and_sign_apply():
    bus = _session(layers=[_layer()])
    bus.dispatch(SetAlpha("L1", "linear"))
    bus.dispatch(SetSign("L1", "neg"))
    layer = bus.state.layers.get("L1")
    assert layer.alpha_mode is AlphaMode.LINEAR
    assert layer.sign_mode is SignMode.NEG


# ---------------------------------------------------------------------------
# loading and the display grid
# ---------------------------------------------------------------------------


def test_first_layer_adopts_the_display_grid_and_centres_the_crosshair():
    bus = _session()
    dirty = bus.dispatch(AddLayer("/data/anat.nii.gz", "L1"))
    assert dirty & Aspect.GRID
    assert bus.state.grid is not None
    assert bus.state.crosshair == (10, 12, 8)


def test_second_layer_does_not_replace_the_grid():
    """Loading a functional dataset must not throw away anatomical resolution."""
    bus = _session()
    bus.dispatch(AddLayer("/data/anat.nii.gz", "L1"))
    grid = bus.state.grid
    dirty = bus.dispatch(AddLayer("/data/stats.nii.gz", "L2"))
    assert bus.state.grid is grid
    assert not (dirty & Aspect.GRID)


def test_add_layer_without_a_loader_is_an_error_not_a_silent_noop():
    bus = vocab.install(CommandBus(ViewerState()))
    with pytest.raises(RuntimeError, match="no loader"):
        bus.dispatch(AddLayer("/data/x.nii.gz", "L1"))


def test_layer_keys_are_minted_uniquely():
    bus = _session()
    bus.dispatch(AddLayer("/data/a.nii.gz"))
    bus.dispatch(AddLayer("/data/b.nii.gz"))
    assert len(set(bus.state.layers.keys)) == 2


# ---------------------------------------------------------------------------
# recording
# ---------------------------------------------------------------------------


def test_recorded_session_replays_to_the_same_state():
    """The whole point of the bus: replay must reproduce the view."""
    bus = _session()
    bus.dispatch(AddLayer("/data/anat.nii.gz", "L1"))
    bus.dispatch(AddLayer("/data/stats.nii.gz", "L2"))
    bus.dispatch(SetIJK(4, 5, 6))
    bus.dispatch(SetThreshold("L2", 3.1))
    bus.dispatch(SetAlpha("L2", "linear"))
    bus.dispatch(SetIndex(3))
    script = bus.to_script(header="recorded")

    replay = _session()
    replay.run_script(script)

    assert replay.state.crosshair == bus.state.crosshair
    assert replay.state.time_index == bus.state.time_index
    assert replay.state.layers.keys == bus.state.layers.keys
    assert replay.state.layers.get("L2").threshold == 3.1
    assert replay.state.layers.get("L2").alpha_mode is AlphaMode.LINEAR


def test_slider_drags_collapse_to_their_final_value():
    """A drag emits a command per pixel; the script should not."""
    bus = _session(layers=[_layer()])
    for v in range(20):
        bus.dispatch(SetThreshold("L1", float(v)))
    lines = [ln for ln in bus.to_script().splitlines() if ln.startswith("SET_THRESHOLD")]
    assert lines == ["SET_THRESHOLD L1 19.0"]


def test_major_events_are_not_collapsed_away():
    bus = _session()
    bus.dispatch(AddLayer("/data/a.nii.gz", "L1"))
    bus.dispatch(AddLayer("/data/b.nii.gz", "L2"))
    script = bus.to_script()
    assert script.count("ADD_LAYER") == 2


def test_major_events_are_reported_for_replay_pauses():
    bus = _session()
    bus.dispatch(AddLayer("/data/a.nii.gz", "L1"))
    bus.dispatch(SetIJK(1, 2, 3))
    bus.dispatch(SetSeed(4, 5, 6))
    assert [c.name for c in bus.major_events()] == ["ADD_LAYER", "SET_SEED"]


def test_recording_can_be_disabled():
    st = ViewerState()
    bus = vocab.install(CommandBus(st, record=False))
    st.layers.add(_layer())
    st.adopt_grid((20, 24, 16), _affine())
    bus.dispatch(SetIJK(1, 1, 1))
    assert bus.log == ()


# ---------------------------------------------------------------------------
# layer stack
# ---------------------------------------------------------------------------


def test_stack_order_puts_new_layers_on_top():
    stack = LayerStack()
    stack.add(_layer("a"))
    stack.add(_layer("b"))
    assert stack.keys == ["a", "b"]
    assert stack.base is not None and stack.base.key == "a"


def test_move_reorders_and_reports_where_it_landed():
    stack = LayerStack()
    for k in "abc":
        stack.add(_layer(k))
    assert stack.move("c", 0) == 0
    assert stack.keys == ["c", "a", "b"]


def test_move_clamps_out_of_range_targets():
    stack = LayerStack()
    for k in "ab":
        stack.add(_layer(k))
    assert stack.move("a", 99) == 1
    assert stack.keys == ["b", "a"]


def test_duplicate_keys_are_rejected():
    stack = LayerStack()
    stack.add(_layer("a"))
    with pytest.raises(ValueError):
        stack.add(_layer("a"))


def test_layers_are_immutable_and_update_replaces():
    stack = LayerStack()
    original = stack.add(_layer("a"))
    updated = stack.update("a", threshold=2.0)
    assert original.threshold == 0.0
    assert updated.threshold == 2.0


def test_visible_layers_skips_hidden_and_transparent():
    stack = LayerStack()
    stack.add(_layer("a"))
    stack.add(_layer("b"))
    stack.add(_layer("c"))
    stack.update("b", visible=False)
    stack.update("c", opacity=0.0)
    assert [ly.key for ly in stack.visible_layers()] == ["a"]
