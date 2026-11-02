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
        # Where each voxel was last drawn, in scene pixels, for picking:
        # (sx, sy, voxel index or None for 0..N-1). None when not pickable.
        self._pick: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None = None

    def remember(self, sx: torch.Tensor, sy: torch.Tensor, vox: torch.Tensor | None = None) -> None:
        self._pick = (sx, sy, vox)

    def pick(self, x: float, y: float, radius: float = 6.0) -> int | None:
        """Voxel drawn nearest to scene pixel (x, y), within ``radius``."""
        if self._pick is None:
            return None
        sx, sy, vox = self._pick
        d2 = (sx - x) ** 2 + (sy - y) ** 2
        i = int(torch.argmin(d2))
        if float(d2[i]) > radius * radius:
            return None
        return int(vox[i]) if vox is not None else i

    def locate(self, v: int) -> tuple[float, float] | None:
        """Scene pixel where voxel v was last drawn, if it was."""
        if self._pick is None:
            return None
        sx, sy, vox = self._pick
        if vox is None:
            return float(sx[v]), float(sy[v])
        hit = torch.nonzero(vox == v)
        if hit.numel() == 0:
            return None
        i = int(hit[0, 0])
        return float(sx[i]), float(sy[i])

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
