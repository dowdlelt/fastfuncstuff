"""Viewer modes. Importing the package registers every built-in mode."""

from fastfuncstuff.viewer.modes.align import AlignMode  # noqa: E402
from fastfuncstuff.viewer.modes.base import (
    BoolControl,
    ChoiceControl,
    ComputedOverlay,
    Control,
    DialogSpec,
    FloatControl,
    IntControl,
    Mark,
    Mode,
    OverlayKind,
    TextControl,
    Trace,
    mode,
    registry,
)
from fastfuncstuff.viewer.modes.denoise import DenoiseMode  # noqa: E402
from fastfuncstuff.viewer.modes.ica import ICAMode  # noqa: E402
from fastfuncstuff.viewer.modes.instacorr import InstaCorrMode  # noqa: E402
from fastfuncstuff.viewer.modes.instaglm import InstaGLMMode  # noqa: E402
from fastfuncstuff.viewer.modes.instaica import InstaICAMode  # noqa: E402
from fastfuncstuff.viewer.modes.instapca import InstaPCAMode  # noqa: E402
from fastfuncstuff.viewer.modes.instaspectra import InstaSpectraMode  # noqa: E402
from fastfuncstuff.viewer.modes.paint import PaintMode  # noqa: E402

# Registration is an import side effect, so every built-in mode must be
# imported here or it will not appear in the mode selector.
from fastfuncstuff.viewer.modes.plain import PlainMode  # noqa: E402
from fastfuncstuff.viewer.modes.preproc import PreprocMode  # noqa: E402
from fastfuncstuff.viewer.modes.rgb import RGBMode  # noqa: E402
from fastfuncstuff.viewer.modes.tcorr import TCorrMode  # noqa: E402

__all__ = [
    "AlignMode",
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
    "InstaICAMode",
    "InstaPCAMode",
    "InstaSpectraMode",
    "IntControl",
    "Mark",
    "Mode",
    "OverlayKind",
    "PaintMode",
    "PlainMode",
    "PreprocMode",
    "RGBMode",
    "TCorrMode",
    "TextControl",
    "Trace",
    "mode",
    "registry",
]
