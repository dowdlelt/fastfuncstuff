"""One object the window and the movie writer drive: the drives plus a scene.

Scenes are built lazily and kept, so flipping between them is instant and each
resumes where it was; all of them share the one parameter dict and the one
data clock.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .drives import SHARED_PARAMS, Drives, Param
from .features import Features
from .scenes import SCENE_NAMES, SCENES, Scene

SCENE_PARAM = Param(
    "scene", 0, 0, len(SCENE_NAMES) - 1, "scene", "how the data is drawn", SCENE_NAMES
)
COMPONENT_PARAM = Param(
    "components",
    16,
    2,
    64,
    "families",
    "PCA components in use (default from the Marchenko-Pastur edge, at most 32): the "
    "low-rank reconstruction, salience and the families' subspace all follow",
)
FAMILY_PARAM = Param(
    "families",
    10,
    2,
    16,
    "families",
    "ICA families: fewer moves as one blob, more moves as noise (reruns the ICA, ~0.5 s)",
)
MUTE_PARAM = Param("mute_noise", 0, 0, 1, "families", "mute every family flagged as noise-like")
PARAMS: tuple[Param, ...] = (
    SCENE_PARAM,
    COMPONENT_PARAM,
    FAMILY_PARAM,
    MUTE_PARAM,
    *SHARED_PARAMS,
    *(prm for name in SCENE_NAMES for prm in SCENES[name].PARAMS),
)
PARAM_BY_NAME = {p.name: p for p in PARAMS}
assert len(PARAM_BY_NAME) == len(PARAMS), "parameter names must be unique across scenes"


#: Never remixed: data pacing, the decomposition itself, and the size or count
#: settings that would rebuild a scene from scratch or move the camera away.
REMIX_SKIP = frozenset(
    {
        "scene", "speed", "families", "components", "mute_noise",
        "cell", "ccell", "lcell", "ncell", "pcell", "gcell", "icell", "fcell", "grain",
        "sparks", "stars", "agents", "panes", "columns", "rows", "radius", "links", "span", "bins", "window",
        "yaw", "pitch", "zoom", "mzoom", "fzoom", "gzoom", "czoom", "szoom", "azoom", "isize",
    }
)  # fmt: skip


def default_params() -> dict[str, float]:
    return {p.name: float(p.default) for p in PARAMS}


class Visualizer:
    def __init__(self, feats: Features, width: int = 640, height: int = 360, seed: int = 0):
        self.f = feats
        self.p = default_params()
        self.p["families"] = float(feats.family_tcs.shape[1])
        self.p["components"] = float(feats.k_active)
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
            n = max(2, min(int(value), int(self.f.k_active)))
            if n != self.drives.n_fam:
                self.f.refit_families(n, self.seed)
                self._families_changed()
            self.p[name] = float(n)
            return
        if name == "components":
            k = max(2, min(int(value), int(self.f.loadings_all.shape[1])))
            if k != self.f.k_active:
                self.f.set_components(k, int(self.p["families"]), self.seed)
                self.p["families"] = float(self.f.family_tcs.shape[1])
                self._families_changed()
            self.p[name] = float(k)
            return
        if name == "mute_noise":
            self.p[name] = float(value)
            self.drives.auto_mute(bool(int(value)))
            return
        self.p[name] = float(value)
        # Shared parameters (canvas) can matter to any scene, so every built
        # scene hears every change; each ignores what it does not use.
        for sc in self._scenes.values():
            sc.on_param(name, float(value))

    def _families_changed(self) -> None:
        self.drives.refresh_families()
        self.drives._weave_key = None  # the weave reads the active components
        self.drives.auto_mute(bool(int(self.p.get("mute_noise", 0))))
        # Every scene caches family-shaped state; rebuilding is cheap.
        self._scenes.clear()

    def set_mute(self, family: int, muted: bool) -> None:
        self.drives.mute[family] = muted

    # ------------------------------------------------------------- remix, config

    def remix(self, seed: int | None = None) -> dict[str, float]:
        """Randomise the shared look and the current scene's settings, reasonably.

        Each setting is kept with probability 0.3; otherwise it is drawn near its
        default -- log-uniform within 4x for settings spanning a decade or more,
        uniform within +-40% of the range otherwise -- and choices are drawn
        uniformly. Returns the settings before the remix, for undo.
        """
        import random

        rng = random.Random(seed)
        before = dict(self.p)
        scene = self.scene_name
        for prm in PARAMS:
            if prm.name in REMIX_SKIP or (prm.scene is not None and prm.scene != scene):
                continue
            if rng.random() < 0.3:
                continue
            if prm.choices:
                value = float(rng.randrange(len(prm.choices)))
            elif prm.lo > 0 and prm.hi / prm.lo >= 10:
                lo, hi = max(prm.lo, prm.default / 4), min(prm.hi, prm.default * 4)
                value = math.exp(rng.uniform(math.log(lo), math.log(hi)))
            else:
                span = 0.4 * (prm.hi - prm.lo)
                value = min(prm.hi, max(prm.lo, prm.default + rng.uniform(-span, span)))
            if prm.integer:
                value = float(round(value))
            self.set_param(prm.name, value)
        return before

    def restore(self, params: dict[str, float]) -> None:
        for name in ("scene", "components", "families"):
            if name in params:
                self.set_param(name, params[name])
        for name, value in params.items():
            if name not in ("scene", "components", "families") and self.p.get(name) != value:
                self.set_param(name, value)

    def to_config(self) -> dict:
        """Everything needed to come back to this look, as plain JSON."""
        out = {}
        for prm in PARAMS:
            v = self.p[prm.name]
            out[prm.name] = (
                prm.choices[int(v)] if prm.choices else (int(v) if prm.integer else round(v, 6))
            )
        return {
            "ffs_dataviz_config": 1,
            "scene": self.scene_name,
            "params": out,
            "muted": [int(i) for i in torch.nonzero(self.drives.mute).flatten().tolist()],
        }

    def from_config(self, cfg: dict) -> None:
        params = {}
        for name, v in cfg.get("params", {}).items():
            prm = PARAM_BY_NAME.get(name)
            if prm is None:
                continue  # a setting from another version: skip, don't fail
            params[name] = (
                float(prm.choices.index(v)) if prm.choices and isinstance(v, str) else float(v)
            )
        if "scene" in cfg and cfg["scene"] in SCENE_NAMES:
            params["scene"] = float(SCENE_NAMES.index(cfg["scene"]))
        self.restore(params)
        for i in cfg.get("muted", []):
            if 0 <= int(i) < self.drives.n_fam:
                self.drives.mute[int(i)] = True

    def step(self, dt: float, playing: bool = True) -> None:
        dt = float(min(dt, 0.05))
        self.drives.advance(dt, playing)
        self.scene.step(dt)

    def render_tensor(self) -> torch.Tensor:
        return self.scene.render()

    def render(self) -> np.ndarray:
        return self.render_tensor().cpu().numpy()
