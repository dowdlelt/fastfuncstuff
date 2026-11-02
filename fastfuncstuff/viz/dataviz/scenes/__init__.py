"""Scenes: different ways of drawing the same drives."""

from __future__ import annotations

from .base import Scene
from .brain import BrainScene
from .flow import FlowScene
from .mandala import MandalaScene
from .milkdrop import MilkdropScene
from .sand import SandScene

SCENES: dict[str, type[Scene]] = {
    cls.name: cls for cls in (MandalaScene, MilkdropScene, FlowScene, SandScene, BrainScene)
}
SCENE_NAMES = tuple(SCENES)

__all__ = ["SCENES", "SCENE_NAMES", "Scene"]
