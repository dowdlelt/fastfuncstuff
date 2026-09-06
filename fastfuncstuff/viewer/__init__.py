"""Interactive GPU-first data explorer.

The viewer is split into a headless core (state, commands, loading, residency,
colourmapping) and a thin UI shell. Everything in the core is testable without a
display, which is also what makes session recording and replay possible: the UI
never mutates state directly, it dispatches commands.
"""

from fastfuncstuff.viewer.colormap import (
    apply_colormap,
    available_colormaps,
    build_lut,
    composite,
    suprathreshold_edges,
    threshold_alpha,
    to_rgba8,
)
from fastfuncstuff.viewer.commands import (
    Aspect,
    Command,
    CommandBus,
    command,
    parse_script,
    registered_names,
    resolve,
)
from fastfuncstuff.viewer.layers import AlphaMode, Layer, LayerStack, SignMode
from fastfuncstuff.viewer.residency import Resident, Tier, VolumeStore
from fastfuncstuff.viewer.session import ViewerSession, derive_range
from fastfuncstuff.viewer.slicing import extract_plane, plane_shape, voxel_value
from fastfuncstuff.viewer.state import DisplayGrid, Locks, Plane, ViewerState

__all__ = [
    "AlphaMode",
    "Aspect",
    "Command",
    "CommandBus",
    "DisplayGrid",
    "Layer",
    "LayerStack",
    "Locks",
    "Plane",
    "Resident",
    "SignMode",
    "Tier",
    "ViewerSession",
    "ViewerState",
    "VolumeStore",
    "apply_colormap",
    "available_colormaps",
    "build_lut",
    "command",
    "composite",
    "derive_range",
    "extract_plane",
    "parse_script",
    "plane_shape",
    "registered_names",
    "resolve",
    "suprathreshold_edges",
    "threshold_alpha",
    "to_rgba8",
    "voxel_value",
]
