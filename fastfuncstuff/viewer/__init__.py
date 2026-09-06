"""Interactive GPU-first data explorer.

The viewer is split into a headless core (state, commands, loading, residency,
colormapping, slicing) and a thin UI shell. Everything in the core is testable
without a display, which is also what makes session recording and replay
possible: the UI never mutates state directly, it dispatches commands.
"""

from fastfuncstuff.viewer.commands import (
    Aspect,
    Command,
    CommandBus,
    command,
    parse_script,
    resolve,
)

__all__ = [
    "Aspect",
    "Command",
    "CommandBus",
    "command",
    "parse_script",
    "resolve",
]
