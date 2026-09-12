"""One palette, one set of font sizes, one way to write a hotkey.

The look is a terminal roguelike's: a near-black ground, warm grey-green text,
and **the key you press written into the label itself** -- ``[R]EAD``,
``[U]NDERLAY``, ``AXIAL [1]``. A control that names its own key does not need
to be discovered twice, once by hunting the help and once by hovering for a
tooltip, and the `h` panel becomes a reminder rather than the only source.

Three stylesheets used to live inline in ``window.py``, ``gridgraph.py`` and
``shortcuts.py``, which is how the graph window ended up a slightly different
grey from the main one. Everything reads from here now -- which is also what
makes a light palette a switch rather than a rewrite.

**What flips and what does not.** Every colour the interface owns flips:
chrome, text, crosshair, edge labels, trace colours, the colour bar's own
frame. The *data* does not. A greyscale anatomy stays greyscale and a hot
overlay stays hot, because a colour scale is a claim about values -- inverting
it to suit the furniture would make two screenshots of the same map disagree.
Invert a layer deliberately with its colormap picker if that is what you want.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

Rgb = tuple[float, float, float]


@dataclass(frozen=True)
class Palette:
    """Every colour the interface owns, in one object.

    Two of them carry meaning on their own and nothing else may use them:
    ``key`` is *always* a key you can press, ``accent`` is *always* the thing
    currently active. That only holds if a palette is a closed set, which is
    why these are fields rather than constants scattered across the widgets.
    """

    name: str

    bg: str  # the ground
    panel: str  # anything inset: lists, boxes, fields
    edge: str  # borders at rest
    edge_lit: str  # borders on a focused or hovered thing
    select: str  # selected row / checked button fill

    text: str  # body text
    dim: str  # labels, units, anything secondary
    faint: str  # disabled

    key: str  # a key you can press. Only ever that.
    accent: str  # active, checked, the current thing. Only ever that.
    head: str  # section headings
    warn: str  # the threshold marker, and anything that wants the eye

    #: Graph traces, ordered so the first two are the furthest apart -- a cell
    #: with two lines on it is the common case.
    series: tuple[Rgb, ...]
    crosshair: Rgb
    #: The pane's slice caption.
    label: Rgb
    #: Anatomical edge labels (L/R/A/P/S/I). Distinct from `label` because it
    #: has to stay legible over the brain as well as over the letterbox.
    edge_label: str


DARK = Palette(
    name="dark",
    bg="#0B0E0C",
    panel="#12181A",
    edge="#23302F",
    edge_lit="#3D5A54",
    select="#17342F",
    text="#C9D6C8",
    dim="#7A8C86",
    faint="#41524F",
    key="#E8C56A",
    accent="#7DE3C3",
    head="#6FB2A0",
    warn="#D9A441",
    series=(
        (0.49, 0.89, 0.76),  # mint
        (0.91, 0.77, 0.42),  # amber
        (0.45, 0.70, 0.95),  # blue
        (0.85, 0.55, 0.85),  # magenta
        (0.60, 0.85, 0.45),  # green
        (0.95, 0.60, 0.40),  # orange
    ),
    crosshair=(0.35, 0.95, 0.85),
    label=(0.55, 0.65, 0.63),
    edge_label="#5C8EA0",
)

#: Not the dark values inverted -- an inverted mid-grey is another mid-grey.
#: Hue and role are kept, lightness is re-chosen against a pale ground, so
#: amber stays the key colour and teal stays the active colour while both gain
#: enough contrast to read on white.
LIGHT = Palette(
    name="light",
    bg="#F5F4EF",
    panel="#FFFFFF",
    edge="#D3D0C6",
    edge_lit="#7FA096",
    select="#DCEBE4",
    text="#171C1A",
    dim="#55635E",
    faint="#A4AEA9",
    key="#8A5B06",
    accent="#0B6E58",
    head="#2B6759",
    warn="#A8600B",
    series=(
        (0.04, 0.48, 0.39),  # teal
        (0.62, 0.42, 0.04),  # amber
        (0.13, 0.35, 0.68),  # blue
        (0.55, 0.20, 0.58),  # magenta
        (0.25, 0.50, 0.13),  # green
        (0.72, 0.34, 0.10),  # orange
    ),
    crosshair=(0.02, 0.42, 0.34),
    label=(0.33, 0.40, 0.38),
    edge_label="#2E5E72",
)

PALETTES = {p.name: p for p in (DARK, LIGHT)}
_active = DARK


def palette() -> Palette:
    return _active


def theme_names() -> list[str]:
    return list(PALETTES)


def set_theme(name: str) -> bool:
    """Switch palettes; returns whether anything changed.

    Module state rather than a value threaded through every widget, because a
    palette is genuinely global -- there is no coherent viewer in which one
    window is light and another dark. What is *not* global is the decision:
    ``ViewerState.theme`` holds it, SET_THEME changes it, and this is the
    mirror the painting code reads.
    """
    global _active
    if name not in PALETTES:
        raise KeyError(f"unknown theme {name!r}; have {sorted(PALETTES)}")
    if _active.name == name:
        return False
    _active = PALETTES[name]
    return True


# -- type ------------------------------------------------------------------
#
# Bigger than the Qt defaults on purpose: this is read across a desk while
# something else has your hands, not leaned into.

MONO = "Menlo" if sys.platform == "darwin" else "monospace"
FONT_BODY = 13
FONT_HEAD = 12
FONT_SMALL = 11
FONT_READOUT = 14


def key_label(text: str, key: str | None) -> str:
    """``("READ", "r") -> "[R]EAD"``; the key bracketed where it occurs.

    Falls back to a suffix when the key is not a letter of the word, which is
    what makes ``AXIAL [1]`` and ``[C]OLOR`` come out of the same call. The
    point is that every label carries its key in the same visual shape, so the
    eye learns one pattern rather than two.
    """
    if not key:
        return text
    if len(key) == 1:
        lowered = text.lower()
        at = lowered.find(key.lower())
        if at >= 0:
            return f"{text[:at]}[{text[at]}]{text[at + 1 :]}"
    return f"{text} [{key}]"


def stylesheet() -> str:
    c = palette()
    return f"""
QMainWindow, QWidget {{ background: {c.bg}; color: {c.text};
    font-family: {MONO}; font-size: {FONT_BODY}px; }}
QDockWidget::title {{ background: {c.panel}; padding: 7px 9px;
    font-size: {FONT_HEAD}px; letter-spacing: 2px; color: {c.head}; }}
QListWidget {{ background: {c.panel}; border: 1px solid {c.edge}; outline: none;
    font-family: {MONO}; font-size: {FONT_BODY}px; }}
QListWidget::item {{ padding: 5px 8px; }}
QListWidget::item:selected {{ background: {c.select}; color: {c.accent}; }}
QLabel {{ color: {c.dim}; font-size: {FONT_HEAD}px; letter-spacing: 1px; }}
QLabel#value {{ color: {c.text}; font-family: {MONO}; font-size: {FONT_READOUT}px;
    letter-spacing: 0px; }}
QLabel#head {{ color: {c.head}; font-size: {FONT_HEAD}px; letter-spacing: 2px; }}
QLabel#key {{ color: {c.key}; font-family: {MONO}; font-size: {FONT_BODY}px;
    letter-spacing: 0px; }}
QLabel#group {{ color: {c.head}; font-size: {FONT_HEAD}px; letter-spacing: 2px; }}
QSlider::groove:horizontal {{ height: 2px; background: {c.edge}; }}
QSlider::handle:horizontal {{ background: {c.accent}; width: 9px; margin: -6px 0; }}
QComboBox, QSpinBox, QDoubleSpinBox {{ background: {c.panel}; border: 1px solid {c.edge};
    padding: 4px 7px; font-family: {MONO}; font-size: {FONT_BODY}px; color: {c.text}; }}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {c.edge_lit}; }}
QComboBox QAbstractItemView {{ background: {c.panel}; color: {c.text};
    selection-background-color: {c.select}; selection-color: {c.accent}; }}
QPushButton {{ background: {c.panel}; border: 1px solid {c.edge}; padding: 5px 11px;
    font-family: {MONO}; font-size: {FONT_BODY}px; letter-spacing: 1px; color: {c.text}; }}
QPushButton:hover {{ background: {c.select}; border-color: {c.edge_lit}; }}
QPushButton:checked {{ background: {c.select}; border-color: {c.edge_lit}; color: {c.accent}; }}
QPushButton:disabled {{ color: {c.faint}; border-color: {c.edge}; }}
QCheckBox {{ color: {c.dim}; font-size: {FONT_HEAD}px; letter-spacing: 1px; }}
QCheckBox::indicator {{ width: 13px; height: 13px;
    border: 1px solid {c.edge_lit}; background: {c.panel}; }}
QCheckBox::indicator:checked {{ background: {c.accent}; border-color: {c.accent}; }}
QCheckBox::indicator:disabled {{ border-color: {c.edge}; }}
QStatusBar {{ background: {c.panel}; color: {c.dim};
    font-family: {MONO}; font-size: {FONT_BODY}px; }}
QStatusBar::item {{ border: 0; }}
QProgressBar {{ background: {c.panel}; border: 1px solid {c.edge}; height: 14px;
    text-align: center; font-size: {FONT_SMALL}px; color: {c.dim}; }}
QProgressBar::chunk {{ background: {c.head}; }}
QToolBar {{ background: {c.panel}; border: 0; spacing: 6px; padding: 6px 8px; }}
QScrollArea {{ background: {c.bg}; }}
QToolTip {{ background: {c.panel}; color: {c.text}; border: 1px solid {c.edge_lit};
    font-family: {MONO}; font-size: {FONT_SMALL}px; padding: 4px; }}
"""


__all__ = [
    "DARK",
    "FONT_BODY",
    "FONT_HEAD",
    "FONT_READOUT",
    "FONT_SMALL",
    "LIGHT",
    "MONO",
    "PALETTES",
    "Palette",
    "key_label",
    "palette",
    "set_theme",
    "stylesheet",
    "theme_names",
]
