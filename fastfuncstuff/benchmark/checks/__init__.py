"""Benchmark checks: one tool, compared on identical frozen inputs, judged one or more ways.

A check declares what a dataset must offer (``needs``), how it draws units
(``sampling``), and how to run and score one unit. The worklist and runner do the rest.
"""

from __future__ import annotations

from .base import Check, UnitResult
from .moco import MocoCheck
from .slicetime import SlicetimeCheck

CHECKS: dict[str, type[Check]] = {c.name: c for c in (MocoCheck, SlicetimeCheck)}

__all__ = ["CHECKS", "Check", "UnitResult"]
