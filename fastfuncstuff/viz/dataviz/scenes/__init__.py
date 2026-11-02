"""Scenes: different ways of drawing the same drives."""

from __future__ import annotations

from .attractor import AttractorScene
from .base import Scene
from .brain import BrainScene
from .constellation import ConstellationScene
from .coral import CoralScene
from .flow import FlowScene
from .fluid import FluidScene
from .galaxies import GalaxiesScene
from .glass import GlassScene
from .interference import InterferenceScene
from .lenia import LeniaScene
from .life import LifeScene
from .mandala import MandalaScene
from .milkdrop import MilkdropScene
from .physarum import PhysarumScene
from .ridgeline import RidgelineScene
from .sand import SandScene
from .spectrum import SpectrumScene
from .statespace import StatespaceScene
from .tapestry import TapestryScene
from .turing import TuringScene

SCENES: dict[str, type[Scene]] = {
    cls.name: cls
    for cls in (
        MandalaScene,
        MilkdropScene,
        GalaxiesScene,
        ConstellationScene,
        TuringScene,
        CoralScene,
        FlowScene,
        SandScene,
        RidgelineScene,
        BrainScene,
        LifeScene,
        LeniaScene,
        PhysarumScene,
        GlassScene,
        AttractorScene,
        FluidScene,
        InterferenceScene,
        StatespaceScene,
        SpectrumScene,
        TapestryScene,
    )
}
SCENE_NAMES = tuple(SCENES)

#: Number-key sets (g cycles them; 1-9 then 0 pick within the current set).
SETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "particles",
        ("mandala", "milkdrop", "galaxies", "constellation", "flow", "sand", "ridgeline", "brain"),
    ),
    ("growth", ("life", "lenia", "turing", "coral", "physarum", "glass", "attractor")),
    ("full field", ("fluid", "interference", "statespace", "spectrum", "tapestry")),
)
assert all(name in SCENES for _, names in SETS for name in names)
assert sorted(n for _, names in SETS for n in names) == sorted(SCENES), (
    "every scene in exactly one set"
)

__all__ = ["SCENES", "SCENE_NAMES", "SETS", "Scene"]
