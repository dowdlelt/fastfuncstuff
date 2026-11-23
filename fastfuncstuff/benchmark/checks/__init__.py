"""Benchmark checks: one tool, compared on identical frozen inputs, judged one or more ways.

A check declares what a dataset must offer (``needs``), how it draws units
(``sampling``), and how to run and score one unit. The worklist and runner do the rest.
"""

from __future__ import annotations

from .base import Check, UnitResult
from .blipflip import BlipflipCheck
from .moco import MocoCheck
from .physio import PhysioCheck
from .qwarp import QwarpCheck
from .reml import RemlCheck
from .slicetime import SlicetimeCheck

CHECKS: dict[str, type[Check]] = {
    c.name: c
    for c in (MocoCheck, SlicetimeCheck, BlipflipCheck, RemlCheck, PhysioCheck, QwarpCheck)
}

__all__ = ["CHECKS", "Check", "UnitResult"]
