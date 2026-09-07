"""Viewer modes. Importing the package registers every built-in mode."""

from fastfuncstuff.viewer.modes.base import (
    BoolControl,
    ChoiceControl,
    ComputedOverlay,
    Control,
    DatasetControl,
    FloatControl,
    IntControl,
    Mode,
    OverlayKind,
    Trace,
    mode,
    registry,
)
from fastfuncstuff.viewer.modes.ica import ICAMode  # noqa: E402
from fastfuncstuff.viewer.modes.instacorr import InstaCorrMode  # noqa: E402

# Registration is an import side effect, so every built-in mode must be
# imported here or it will not appear in the mode selector.
from fastfuncstuff.viewer.modes.plain import PlainMode  # noqa: E402

__all__ = [
    "BoolControl",
    "ChoiceControl",
    "ComputedOverlay",
    "Control",
    "DatasetControl",
    "FloatControl",
    "ICAMode",
    "InstaCorrMode",
    "IntControl",
    "Mode",
    "OverlayKind",
    "PlainMode",
    "Trace",
    "mode",
    "registry",
]
