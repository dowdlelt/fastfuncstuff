"""Scenes: different ways of drawing the same drives."""

from __future__ import annotations

from .base import Scene
from .brain import BrainScene
from .constellation import ConstellationScene
from .coral import CoralScene
from .flow import FlowScene
from .galaxies import GalaxiesScene
from .mandala import MandalaScene
from .milkdrop import MilkdropScene
from .ridgeline import RidgelineScene
from .sand import SandScene
from .turing import TuringScene

# Order is the number keys: 1-9, then 0.
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
    )
}
SCENE_NAMES = tuple(SCENES)

__all__ = ["SCENES", "SCENE_NAMES", "Scene"]
