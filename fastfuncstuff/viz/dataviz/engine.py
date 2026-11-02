"""One object the window and the movie writer drive: the drives plus a scene.

Scenes are built lazily and kept, so flipping between them is instant and each
resumes where it was; all of them share the one parameter dict and the one
data clock.
"""

from __future__ import annotations

import numpy as np
import torch

from .drives import SHARED_PARAMS, Drives, Param
from .features import Features
from .scenes import SCENE_NAMES, SCENES, Scene

SCENE_PARAM = Param(
    "scene", 0, 0, len(SCENE_NAMES) - 1, "scene", "how the data is drawn", SCENE_NAMES
)
PARAMS: tuple[Param, ...] = (
    SCENE_PARAM,
    *SHARED_PARAMS,
    *(prm for name in SCENE_NAMES for prm in SCENES[name].PARAMS),
)
PARAM_BY_NAME = {p.name: p for p in PARAMS}
assert len(PARAM_BY_NAME) == len(PARAMS), "parameter names must be unique across scenes"


def default_params() -> dict[str, float]:
    return {p.name: float(p.default) for p in PARAMS}


class Visualizer:
    def __init__(self, feats: Features, width: int = 640, height: int = 360, seed: int = 0):
        self.f = feats
        self.p = default_params()
        self.p["families"] = float(feats.family_tcs.shape[1])
        self.drives = Drives(feats, self.p)
        self.width, self.height = width, height
        self.seed = seed
        self._scenes: dict[str, Scene] = {}

    @property
    def scene_name(self) -> str:
        return SCENE_NAMES[int(self.p["scene"])]

    @property
    def scene(self) -> Scene:
        name = self.scene_name
        if name not in self._scenes:
            self._scenes[name] = SCENES[name](self.drives, self.width, self.height, self.seed)
        sc = self._scenes[name]
        sc.resize(self.width, self.height)
        return sc

    @property
    def t(self) -> float:
        return self.drives.t

    @property
    def n_time(self) -> int:
        return self.f.n_time

    def seek(self, t: float) -> None:
        self.drives.seek(t)

    def resize(self, width: int, height: int) -> None:
        self.width, self.height = int(width), int(height)

    def set_param(self, name: str, value: float) -> None:
        if name not in self.p:
            raise KeyError(name)
        if name == "families":
            n = max(2, min(int(value), int(self.f.loadings.shape[1])))
            if n != self.drives.n_fam:
                self.f.refit_families(n, self.seed)
                self.drives.refresh_families()
                # Every scene caches family-shaped state; rebuilding is cheap.
                self._scenes.clear()
            self.p[name] = float(n)
            return
        self.p[name] = float(value)
        for sc in self._scenes.values():
            if any(prm.name == name for prm in sc.PARAMS):
                sc.on_param(name, float(value))

    def step(self, dt: float, playing: bool = True) -> None:
        dt = float(min(dt, 0.05))
        self.drives.advance(dt, playing)
        self.scene.step(dt)

    def render_tensor(self) -> torch.Tensor:
        return self.scene.render()

    def render(self) -> np.ndarray:
        return self.render_tensor().cpu().numpy()
