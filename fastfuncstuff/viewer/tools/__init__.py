"""Preproc tools. Importing the package registers every built-in tool."""

from fastfuncstuff.viewer.tools.base import (
    AuxVolume,
    Tool,
    ToolOutcome,
    ToolRegistry,
    registry,
    tool,
)

# Registration is an import side effect, so every built-in tool must be
# imported here or its button will not appear in Preproc.
from fastfuncstuff.viewer.tools.moco import MocoTool  # noqa: E402
from fastfuncstuff.viewer.tools.slicetime import SliceTimeTool  # noqa: E402
from fastfuncstuff.viewer.tools.smooth import SmoothTool  # noqa: E402

__all__ = [
    "AuxVolume",
    "MocoTool",
    "SliceTimeTool",
    "SmoothTool",
    "Tool",
    "ToolOutcome",
    "ToolRegistry",
    "registry",
    "tool",
]
