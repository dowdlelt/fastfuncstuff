"""What every scene is: a way of drawing the shared drives into pixels."""

from __future__ import annotations

import torch

from ..drives import Drives, Param


class Scene:
    name = ""
    blurb = ""
    PARAMS: tuple[Param, ...] = ()

    def __init__(self, drives: Drives, width: int, height: int, seed: int = 0):
        self.dr = drives
        self.f = drives.f
        self.p = drives.p
        self.device = drives.device
        self.width, self.height = width, height
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.accum = torch.zeros(height * width, 3, device=self.device)

    def rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=self.gen).to(self.device)

    def randn(self, *shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=self.gen).to(self.device)

    def resize(self, width: int, height: int) -> None:
        if (width, height) != (self.width, self.height):
            self.width, self.height = int(width), int(height)
            self.accum = torch.zeros(self.height * self.width, 3, device=self.device)
            self.on_resize()

    def on_resize(self) -> None:
        pass

    def on_param(self, name: str, value: float) -> None:
        pass

    def step(self, dt: float) -> None:
        raise NotImplementedError

    def render(self) -> torch.Tensor:
        """(H, W, 3) uint8 on the scene's device."""
        raise NotImplementedError

    @property
    def density(self) -> float:
        """Per-point brightness that exposes alike whatever the voxel count."""
        return (self.width * self.height * 0.08) / max(1, self.f.n_voxels)

    def fade_in(self, frame: torch.Tensor) -> None:
        """Mix a new frame into the trails buffer."""
        tr = self.p["trails"]
        self.accum.mul_(tr).add_(frame * (1 - tr) * 4.0)
