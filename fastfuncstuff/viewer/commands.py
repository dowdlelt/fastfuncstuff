"""The command bus: every viewer state mutation is a typed, recorded command.

Nothing in the UI mutates state directly. A click, a keystroke and a line in a
replayed script all arrive here as the same :class:`Command`, which is what lets
one mechanism serve session recording, replay, external driving and undo.

AFNI reached the same design from the other direction -- ``afni_driver.c`` grew
a 101-entry string dispatch table over twenty years. The lesson worth taking is
that the table has to come first: a recorder bolted on afterwards records the
subset of actions someone remembered to instrument, which is a recorder that
lies about what the session did.

Script syntax is deliberately AFNI-flavoured (``SET_IJK 32 40 25``) so recorded
sessions stay human-editable and the vocabulary reads as familiar.
"""

from __future__ import annotations

import shlex
import typing
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import MISSING, dataclass, fields, is_dataclass
from enum import IntFlag, auto
from typing import Any, ClassVar


class Aspect(IntFlag):
    """What a command invalidates, so redraws stay proportionate.

    Moving the crosshair must not re-upload a texture, and changing a colormap
    must not re-slice the volume. Commands declare what they dirty and the UI
    repaints only that -- this is the whole defence against a viewer that
    stutters when you drag something.
    """

    NOTHING = 0
    CROSSHAIR = auto()
    SLICES = auto()
    COLORMAP = auto()
    THRESHOLD = auto()
    LAYERS = auto()
    TIME = auto()
    GRAPH = auto()
    GRID = auto()
    #: A window opened, closed, or changed what it shows. The window manager
    #: reconciles against the viewport list when it sees this.
    VIEWPORTS = auto()
    #: The interface's palette changed. Every window restyles; nothing
    #: re-slices, because a palette says nothing about the data.
    THEME = auto()
    ALL = (
        CROSSHAIR | SLICES | COLORMAP | THRESHOLD | LAYERS | TIME | GRAPH | GRID | VIEWPORTS | THEME
    )


@dataclass(frozen=True)
class Command:
    """One state mutation, serializable to a script line.

    Subclasses are frozen dataclasses registered with :func:`command`. They
    carry only plain scalars so that a recorded script round-trips through text
    without needing the objects that produced it.
    """

    name: ClassVar[str] = ""
    aspects: ClassVar[Aspect] = Aspect.NOTHING
    #: Whether a replay should offer to pause here. Loading data, running a fit
    #: and setting a seed are the points a person actually wants to watch; the
    #: crosshair moves between them are not.
    major: ClassVar[bool] = False

    def to_args(self) -> list[str]:
        """Positional arguments for this command's script line."""
        out: list[str] = []
        for f in fields(self):
            out.extend(_encode(getattr(self, f.name)))
        return out

    def to_line(self) -> str:
        args = self.to_args()
        return " ".join([self.name, *(shlex.quote(a) for a in args)]).rstrip()


_REGISTRY: dict[str, type[Command]] = {}


def command[C: type[Command]](cls: C) -> C:
    """Register a command class under its ``name``.

    Registration is what makes a command reachable from a script or from an
    external driver, so an unregistered command is invisible to replay -- the
    decorator is not optional bookkeeping.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls.__name__} must be a dataclass")
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set a class-level name")
    existing = _REGISTRY.get(cls.name)
    if existing is not None and existing is not cls:
        raise ValueError(f"command name {cls.name!r} already registered to {existing.__name__}")
    _REGISTRY[cls.name] = cls
    return cls


def resolve(name: str) -> type[Command]:
    """Look up a registered command class by name."""
    try:
        return _REGISTRY[name.upper()]
    except KeyError:
        raise KeyError(f"unknown command {name!r}") from None


def registered_names() -> list[str]:
    """Every command name the bus understands, sorted."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# scalar encoding
#
# Commands hold only scalars and homogeneous tuples of scalars. That keeps the
# script text readable and means replay never has to reconstruct a tensor or a
# path-dependent object from a string.
# ---------------------------------------------------------------------------

_SCALARS: tuple[type, ...] = (int, float, str, bool)


def _encode(value: Any) -> list[str]:
    if isinstance(value, bool):
        return ["1" if value else "0"]
    if isinstance(value, (int, float, str)):
        return [str(value)]
    if isinstance(value, tuple):
        out: list[str] = []
        for item in value:
            out.extend(_encode(item))
        return out
    if value is None:
        return ["-"]
    raise TypeError(f"cannot encode {type(value).__name__} in a command")


def _decode_scalar(text: str, kind: Any) -> Any:
    if kind is bool:
        return text not in ("0", "false", "False", "")
    if kind is int:
        return int(text)
    if kind is float:
        return float(text)
    return text


def _build(cls: type[Command], args: Sequence[str]) -> Command:
    """Construct a command from its positional script arguments."""
    hints = typing.get_type_hints(cls)
    values: dict[str, Any] = {}
    pos = 0
    for f in fields(cls):
        kind = hints.get(f.name, str)
        origin = typing.get_origin(kind)
        if origin is tuple:
            members = typing.get_args(kind)
            if len(members) == 2 and members[1] is Ellipsis:
                raise TypeError(f"{cls.__name__}.{f.name}: variadic tuples are not supported")
            take = args[pos : pos + len(members)]
            if len(take) != len(members):
                raise ValueError(f"{cls.name} expects {len(members)} values for {f.name}")
            values[f.name] = tuple(_decode_scalar(t, m) for t, m in zip(take, members, strict=True))
            pos += len(members)
            continue
        # Optional[X] shows up as a union; take the first non-None member and
        # let "-" stand for absence, matching how AFNI's driver spells a
        # skipped argument.
        if origin is typing.Union or str(origin) == "typing.Union":
            members = [m for m in typing.get_args(kind) if m is not type(None)]
            kind = members[0] if members else str
        if pos >= len(args):
            if f.default is not MISSING:
                continue
            raise ValueError(f"{cls.name} is missing a value for {f.name}")
        text = args[pos]
        pos += 1
        values[f.name] = None if text == "-" else _decode_scalar(text, kind)
    return cls(**values)


def parse_line(line: str) -> Command | None:
    """Parse one script line into a command, or ``None`` for blank/comment."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    parts = shlex.split(stripped)
    return _build(resolve(parts[0]), parts[1:])


def parse_script(text: str) -> list[Command]:
    """Parse a whole recorded script."""
    out: list[Command] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        try:
            cmd = parse_line(line)
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"line {lineno}: {exc}") from exc
        if cmd is not None:
            out.append(cmd)
    return out


# ---------------------------------------------------------------------------
# the bus
# ---------------------------------------------------------------------------

Handler = Callable[[Command, Any], Aspect]
Listener = Callable[[Command, Aspect], None]


class CommandBus:
    """Dispatches commands against a state object and records what it applied.

    The bus owns no state of its own beyond the recording; ``state`` is whatever
    object the handlers mutate. Handlers return the aspects they actually
    dirtied, which may be narrower than the command's declared aspects -- a
    crosshair move that lands on the same voxel dirties nothing.
    """

    def __init__(self, state: Any, *, record: bool = True) -> None:
        self.state = state
        self._handlers: dict[str, Handler] = {}
        self._listeners: list[Listener] = []
        self._log: list[Command] = []
        self.recording = record

    # -- registration --------------------------------------------------
    def handle(self, name: str) -> Callable[[Handler], Handler]:
        """Register the handler for one command name."""

        def deco(fn: Handler) -> Handler:
            if name in self._handlers:
                raise ValueError(f"handler for {name!r} already registered")
            self._handlers[name] = fn
            return fn

        return deco

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        """Observe applied commands; returns an unsubscribe callable.

        Listeners are how the UI learns what to repaint. They run after the
        state has changed and must not dispatch, so a repaint cannot cascade
        into another mutation.
        """
        self._listeners.append(listener)

        def off() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return off

    # -- dispatch ------------------------------------------------------
    def dispatch(self, cmd: Command) -> Aspect:
        """Apply one command, record it, and notify listeners."""
        handler = self._handlers.get(cmd.name)
        if handler is None:
            raise KeyError(f"no handler registered for {cmd.name!r}")
        dirty = handler(cmd, self.state)
        if dirty is None:  # a handler that forgot to report is treated as total
            dirty = cmd.aspects
        if self.recording:
            self._log.append(cmd)
        for listener in tuple(self._listeners):
            listener(cmd, dirty)
        return dirty

    def record(self, cmd: Command) -> None:
        """Log a command whose effect has already been applied.

        The one case this exists for: work slow enough to run on a worker. The
        UI computes the result off-thread and installs it on the GUI thread, so
        by the time the command could be dispatched its effect is already
        there, and dispatching would redo seconds of arithmetic to reach the
        state the session is in.

        Replay is unaffected -- it dispatches the command for real, which is
        right, because a headless replay has no interface to keep responsive.
        Use this only when the effect is genuinely equivalent to dispatching;
        anything else puts a lie in the recording, which is the failure the
        whole bus exists to prevent.
        """
        if self.recording:
            self._log.append(cmd)

    def dispatch_all(self, cmds: Iterable[Command]) -> Aspect:
        """Apply a sequence, returning the union of what it dirtied."""
        total = Aspect.NOTHING
        for cmd in cmds:
            total |= self.dispatch(cmd)
        return total

    def run_script(self, text: str) -> Aspect:
        """Parse and apply a recorded script."""
        return self.dispatch_all(parse_script(text))

    # -- recording -----------------------------------------------------
    @property
    def log(self) -> tuple[Command, ...]:
        return tuple(self._log)

    def clear_log(self) -> None:
        self._log.clear()

    def to_script(self, *, header: str | None = None) -> str:
        """Render the recorded session as a replayable script.

        Consecutive commands that supersede one another collapse to the last of
        the run: dragging a slider emits one line per pixel, and replaying every
        intermediate value would be both slow and misleading about what the
        person meant to do.
        """
        lines: list[str] = []
        if header:
            lines.extend(f"# {ln}" for ln in header.splitlines())
        for cmd in _collapse(self._log):
            if cmd.major and lines and not lines[-1].startswith("#"):
                lines.append("")
            lines.append(cmd.to_line())
        return "\n".join(lines) + "\n" if lines else ""

    def major_events(self) -> list[Command]:
        """The recorded commands a replay should be able to pause on."""
        return [c for c in self._log if c.major]


def _target(cmd: Command) -> tuple[Any, ...]:
    """What a command is aimed at, for deciding whether two supersede.

    Dragging one layer's opacity slider emits a line per pixel and only the
    last matters. Setting the geometry of four windows in a row also emits four
    lines of one type -- and every one of them matters, because each names a
    different window. So the fields that identify a target take part in the
    comparison: same type *and* same target is a supersede, same type and
    different target is two separate actions.
    """
    return tuple(
        getattr(cmd, f.name) for f in fields(cmd) if f.name in ("key", "view", "which", "name")
    )


def _collapse(log: Sequence[Command]) -> Iterator[Command]:
    """Drop superseded runs of the same non-major command, per target."""
    for i, cmd in enumerate(log):
        if cmd.major:
            yield cmd
            continue
        nxt = log[i + 1] if i + 1 < len(log) else None
        if (
            nxt is not None
            and type(nxt) is type(cmd)
            and not nxt.major
            and _target(nxt) == _target(cmd)
        ):
            continue
        yield cmd
