"""One palette, one set of font sizes, one way to write a hotkey.

The look is a terminal roguelike's: a near-black ground, warm grey-green text,
and **the key you press written into the label itself** -- ``[R]EAD``,
``[U]NDERLAY``, ``AXIAL [1]``. A control that names its own key does not need
to be discovered twice, once by hunting the help and once by hovering for a
tooltip, and the `h` panel becomes a reminder rather than the only source.

Three stylesheets used to live inline in ``window.py``, ``gridgraph.py`` and
``shortcuts.py``, which is how the graph window ended up a slightly different
grey from the main one. Everything reads from here now.
"""

from __future__ import annotations

import sys

# -- palette ---------------------------------------------------------------
#
# Warm-cold split with a job for each: amber is *always* a key you can press,
# mint is *always* something currently active. Nothing else may use them, or
# they stop carrying information.

BG = "#0B0E0C"  # the ground
PANEL = "#12181A"  # anything inset: lists, boxes, fields
EDGE = "#23302F"  # borders at rest
EDGE_LIT = "#3D5A54"  # borders on a focused or hovered thing
SELECT = "#17342F"  # selected row / checked button fill

TEXT = "#C9D6C8"  # body text
DIM = "#7A8C86"  # labels, units, anything secondary
FAINT = "#41524F"  # disabled

KEY = "#E8C56A"  # amber: a key you can press. Only ever that.
ACCENT = "#7DE3C3"  # mint: active, checked, the current thing. Only ever that.
HEAD = "#6FB2A0"  # section headings
WARN = "#D9814A"

# Graph trace colours, ordered so the first two are the ones you get on a
# two-trace cell and are the furthest apart.
SERIES_RGB = (
    (0.49, 0.89, 0.76),  # mint
    (0.91, 0.77, 0.42),  # amber
    (0.45, 0.70, 0.95),  # blue
    (0.85, 0.55, 0.85),  # magenta
    (0.60, 0.85, 0.45),  # green
    (0.95, 0.60, 0.40),  # orange
)

CROSSHAIR_RGB = (0.35, 0.95, 0.85)
LABEL_RGB = (0.55, 0.65, 0.63)

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
    return f"""
QMainWindow, QWidget {{ background: {BG}; color: {TEXT};
    font-family: {MONO}; font-size: {FONT_BODY}px; }}
QDockWidget::title {{ background: {PANEL}; padding: 7px 9px;
    font-size: {FONT_HEAD}px; letter-spacing: 2px; color: {HEAD}; }}
QListWidget {{ background: {PANEL}; border: 1px solid {EDGE}; outline: none;
    font-family: {MONO}; font-size: {FONT_BODY}px; }}
QListWidget::item {{ padding: 5px 8px; }}
QListWidget::item:selected {{ background: {SELECT}; color: {ACCENT}; }}
QLabel {{ color: {DIM}; font-size: {FONT_HEAD}px; letter-spacing: 1px; }}
QLabel#value {{ color: {TEXT}; font-family: {MONO}; font-size: {FONT_READOUT}px;
    letter-spacing: 0px; }}
QLabel#head {{ color: {HEAD}; font-size: {FONT_HEAD}px; letter-spacing: 2px; }}
QLabel#key {{ color: {KEY}; font-family: {MONO}; font-size: {FONT_BODY}px;
    letter-spacing: 0px; }}
QLabel#group {{ color: {HEAD}; font-size: {FONT_HEAD}px; letter-spacing: 2px; }}
QSlider::groove:horizontal {{ height: 2px; background: {EDGE}; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 9px; margin: -6px 0; }}
QComboBox, QSpinBox, QDoubleSpinBox {{ background: {PANEL}; border: 1px solid {EDGE};
    padding: 4px 7px; font-family: {MONO}; font-size: {FONT_BODY}px; color: {TEXT}; }}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {EDGE_LIT}; }}
QComboBox QAbstractItemView {{ background: {PANEL}; color: {TEXT};
    selection-background-color: {SELECT}; selection-color: {ACCENT}; }}
QPushButton {{ background: {PANEL}; border: 1px solid {EDGE}; padding: 5px 11px;
    font-family: {MONO}; font-size: {FONT_BODY}px; letter-spacing: 1px; color: {TEXT}; }}
QPushButton:hover {{ background: {SELECT}; border-color: {EDGE_LIT}; }}
QPushButton:checked {{ background: {SELECT}; border-color: {EDGE_LIT}; color: {ACCENT}; }}
QPushButton:disabled {{ color: {FAINT}; border-color: {EDGE}; }}
QCheckBox {{ color: {DIM}; font-size: {FONT_HEAD}px; letter-spacing: 1px; }}
QCheckBox::indicator {{ width: 13px; height: 13px;
    border: 1px solid {EDGE_LIT}; background: {PANEL}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QCheckBox::indicator:disabled {{ border-color: {EDGE}; }}
QStatusBar {{ background: {PANEL}; color: {DIM};
    font-family: {MONO}; font-size: {FONT_BODY}px; }}
QStatusBar::item {{ border: 0; }}
QProgressBar {{ background: {PANEL}; border: 1px solid {EDGE}; height: 14px;
    text-align: center; font-size: {FONT_SMALL}px; color: {DIM}; }}
QProgressBar::chunk {{ background: {HEAD}; }}
QToolBar {{ background: {PANEL}; border: 0; spacing: 6px; padding: 6px 8px; }}
QToolTip {{ background: {PANEL}; color: {TEXT}; border: 1px solid {EDGE_LIT};
    font-family: {MONO}; font-size: {FONT_SMALL}px; padding: 4px; }}
"""


__all__ = [
    "ACCENT",
    "BG",
    "CROSSHAIR_RGB",
    "DIM",
    "EDGE",
    "EDGE_LIT",
    "FAINT",
    "FONT_BODY",
    "FONT_HEAD",
    "FONT_READOUT",
    "FONT_SMALL",
    "HEAD",
    "KEY",
    "LABEL_RGB",
    "MONO",
    "PANEL",
    "SELECT",
    "SERIES_RGB",
    "TEXT",
    "WARN",
    "key_label",
]
