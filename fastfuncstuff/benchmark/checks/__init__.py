"""Benchmark checks: one tool, compared on identical frozen inputs, judged one or more ways.

A check declares what a dataset must offer (``needs``), how it draws units
(``sampling``), and how to run and score one unit. The worklist and runner do the rest.
"""

from __future__ import annotations

from .allineate import AllineateCheck
from .base import Check, UnitResult
from .blipflip import BlipflipCheck
from .fc import FcCheck
from .locomoco import LocomocoCheck
from .moco import MocoCheck
from .nordic import NordicCheck
from .physio import PhysioCheck
from .qwarp import QwarpCheck
from .reml import RemlCheck
from .segment import SegmentCheck
from .slicetime import SlicetimeCheck
from .warp_mni import WarpMniCheck
from .warp_pair import WarpPairCheck

CHECKS: dict[str, type[Check]] = {
    c.name: c
    for c in (
        MocoCheck,
        SlicetimeCheck,
        BlipflipCheck,
        RemlCheck,
        PhysioCheck,
        QwarpCheck,
        AllineateCheck,
        NordicCheck,
        FcCheck,
        SegmentCheck,
        WarpPairCheck,
        WarpMniCheck,
        LocomocoCheck,
    )
}

__all__ = ["CHECKS", "Check", "UnitResult"]
