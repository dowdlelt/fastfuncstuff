"""Viewer modes. Importing the package registers every built-in mode."""

from fastfuncstuff.viewer.modes.base import (
    BoolControl,
    ChoiceControl,
    ComputedOverlay,
    Control,
    DialogSpec,
    FloatControl,
    IntControl,
    Mode,
    OverlayKind,
    Trace,
    mode,
    registry,
)
from fastfuncstuff.viewer.modes.denoise import DenoiseMode  # noqa: E402
from fastfuncstuff.viewer.modes.ica import ICAMode  # noqa: E402
from fastfuncstuff.viewer.modes.instacorr import InstaCorrMode  # noqa: E402
from fastfuncstuff.viewer.modes.instaglm import InstaGLMMode  # noqa: E402

# Registration is an import side effect, so every built-in mode must be
# imported here or it will not appear in the mode selector.
from fastfuncstuff.viewer.modes.plain import PlainMode  # noqa: E402
from fastfuncstuff.viewer.modes.preproc import PreprocMode  # noqa: E402

__all__ = [
    "BoolControl",
    "ChoiceControl",
    "ComputedOverlay",
    "Control",
    "DenoiseMode",
    "DialogSpec",
    "FloatControl",
    "ICAMode",
    "InstaCorrMode",
    "InstaGLMMode",
    "IntControl",
    "Mode",
    "OverlayKind",
    "PlainMode",
    "PreprocMode",
    "Trace",
    "mode",
    "registry",
]
