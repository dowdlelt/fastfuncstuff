"""Interactive GPU-first data explorer.

The viewer is split into a headless core (state, commands, loading, residency,
colourmapping) and a thin UI shell. Everything in the core is testable without a
display, which is also what makes session recording and replay possible: the UI
never mutates state directly, it dispatches commands.

The names below resolve on first use: every ``fastfuncstuff.viewer.<module>`` import
runs this file first, and these pull in torch -- which ffs_viewer needs to be able
to show a window without (see ``ui/boot.py``).
"""

from importlib import import_module
from typing import Any

_HOMES = {
    "apply_colormap": ".colormap",
    "available_colormaps": ".colormap",
    "build_lut": ".colormap",
    "composite": ".colormap",
    "suprathreshold_edges": ".colormap",
    "threshold_alpha": ".colormap",
    "to_rgba8": ".colormap",
    "Aspect": ".commands",
    "Command": ".commands",
    "CommandBus": ".commands",
    "command": ".commands",
    "parse_script": ".commands",
    "registered_names": ".commands",
    "resolve": ".commands",
    "AlphaMode": ".layers",
    "Layer": ".layers",
    "LayerStack": ".layers",
    "SignMode": ".layers",
    "Resident": ".residency",
    "Tier": ".residency",
    "VolumeStore": ".residency",
    "ViewerSession": ".session",
    "derive_range": ".session",
    "extract_plane": ".slicing",
    "plane_shape": ".slicing",
    "voxel_value": ".slicing",
    "DisplayGrid": ".state",
    "Plane": ".state",
    "ViewerState": ".state",
    "ViewKind": ".viewports",
    "Viewport": ".viewports",
    "ViewportSet": ".viewports",
}

__all__ = sorted(_HOMES)


def __getattr__(name: str) -> Any:
    home = _HOMES.get(name)
    if home is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(home, __name__), name)
