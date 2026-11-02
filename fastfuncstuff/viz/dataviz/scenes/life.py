"""Life: cellular automata zoomed far out, seeded by the data, inside the head.

A grid as fine as the screen sees the head from above. Each frame, cells are
born at random wherever the live activity under them is high, carrying the
colour of the family that lives there; then the automaton runs. A newborn
takes the mean colour of the neighbours that made it, so lineages mix where
families meet. Zoomed this far out the gliders, puffers and oscillators blur
into smoke, flame and weather, which is the point: tiny rules, seen from
far enough away, become texture -- and here the data decides where the
texture lives.

Rules: Conway's Life (B3/S23), HighLife (B36/S23, replicators), Day & Night
(B3678/S34678, symmetric blobs), Seeds (B2/S, explosive) and Brian's Brain
(B2, on -> dying -> off; endless travelling sparks). A DVARS spike drops a
burst of random life.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..drives import Param
from .base import Scene, blur, disks, filled_field

S = "life"
RULES = ("life", "highlife", "daynight", "seeds", "brain")
_BIRTH = {"life": (3,), "highlife": (3, 6), "daynight": (3, 6, 7, 8), "seeds": (2,), "brain": (2,)}
_SURVIVE = {
    "life": (2, 3),
    "highlife": (2, 3),
    "daynight": (3, 4, 6, 7, 8),
    "seeds": (),
    "brain": (),
}


class LifeScene(Scene):
    name = S
    blurb = "cellular automata seen from far away, seeded where the data is active"
    PARAMS = (
        Param("rule", 0, 0, len(RULES) - 1, S, "automaton rule", RULES, scene=S),
        Param("sow", 1.0, 0.0, 5.0, S, "births per frame per unit of local activity", scene=S),
        Param("gens", 2, 1, 6, S, "generations per frame", scene=S),
        Param("lcell", 1, 1, 4, S, "pixels per cell", scene=S),
        Param("smoke", 0.85, 0.0, 0.99, S, "trail persistence (smoke)", scene=S),
        Param("burst", 1.0, 0.0, 4.0, S, "random life dropped on DVARS spikes", scene=S),
    )

    def __init__(self, drives, width, height, seed=0):
        super().__init__(drives, width, height, seed)
        self._alloc()

    def on_resize(self):
        self._alloc()

    def on_param(self, name, value):
        if name in ("lcell", "canvas"):
            self._alloc()

    def _alloc(self) -> None:
        f, dr = self.f, self.dr
        c = int(self.p["lcell"])
        self.gh, self.gw = max(16, self.height // c), max(16, self.width // c)
        self.cell_of, self.inside, self.passes = self.grid_cells(self.gh, self.gw)
        onehot = torch.zeros(f.n_voxels, dr.n_fam, device=self.device)
        onehot[torch.arange(f.n_voxels, device=self.device), f.family] = f.family_weight.clamp_min(
            0.05
        )
        terr = filled_field(onehot, self.cell_of, self.gh, self.gw, self.passes + 2)
        self.home_rgb = blur(dr.family_rgb()[terr.argmax(0)].permute(2, 0, 1), 1)  # (3, gh, gw)
        self.state = torch.zeros(
            self.gh, self.gw, dtype=torch.uint8, device=self.device
        )  # 0 off 1 on 2 dying
        self.rgb = torch.zeros(3, self.gh, self.gw, device=self.device)
        self.glow = torch.zeros(3, self.gh, self.gw, device=self.device)
        self.k = torch.ones(1, 1, 3, 3, device=self.device)
        self.k[0, 0, 1, 1] = 0

    def _count(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.pad(x[None, None], (1, 1, 1, 1)), self.k)[0, 0]

    def _generation(self, rule: str) -> None:
        on = (self.state == 1).float()
        n = self._count(on).round().long()
        birth = torch.zeros_like(on, dtype=torch.bool)
        for b in _BIRTH[rule]:
            birth |= n == b
        birth &= (self.state == 0) & self.inside
        # Newborns inherit the mean colour of the live neighbours that made them.
        mix = F.conv2d(
            F.pad((self.rgb * on)[None], (1, 1, 1, 1)), self.k.expand(3, 1, 3, 3), groups=3
        )[0]
        mix = mix / n.clamp_min(1).float()
        if rule == "brain":
            new = torch.zeros_like(self.state)
            new[self.state == 1] = 2  # on -> dying
            new[birth] = 1  # off -> on with two live neighbours
        else:
            keep = torch.zeros_like(birth)
            for s in _SURVIVE[rule]:
                keep |= n == s
            keep &= self.state == 1
            new = (keep | birth).to(torch.uint8)
        self.rgb = torch.where(birth[None], mix, self.rgb)
        self.state = new

    def step(self, dt: float) -> None:
        p, dr = self.p, self.dr
        rule = RULES[int(p["rule"])]
        act = filled_field(dr.brightness(), self.cell_of, self.gh, self.gw, self.passes + 1)
        act = 1 - torch.exp(-act / max(p["gain"], 1e-3))
        prob = act * p["sow"] * 0.02
        if dr.d > 2.5 and p["burst"] > 0:
            prob = prob + (dr.d - 2.5) * 0.01 * p["burst"]
        # Soup patches, not single cells: an isolated birth dies at once under
        # every rule but Seeds, so sparse sowing alone left the grid empty.
        spots = self.rand(self.gh, self.gw) < prob * 0.08
        patch = disks(spots, 3) & (self.rand(self.gh, self.gw) < 0.45)
        sown = patch & self.inside & (self.state == 0)
        self.state[sown] = 1
        self.rgb = torch.where(sown[None], self.home_rgb, self.rgb)
        for _ in range(int(p["gens"])):
            self._generation(rule)

    def render(self) -> torch.Tensor:
        p, dr = self.p, self.dr
        lit = (self.state == 1).float() + 0.35 * (self.state == 2).float()
        k = p["smoke"]
        self.glow = self.glow * k + self.rgb * lit * (1 - k) * 3.0
        img = self.glow + 0.006 * self.inside * self.home_rgb
        img = F.interpolate(img[None], size=(self.height, self.width), mode="nearest")[0]
        cw, ch = self.width / self.gw, self.height / self.gh
        cell = self.cell_of
        self.remember(
            ((cell % self.gw).float() + 0.5) * cw,
            ((cell // self.gw).float() + 0.5) * ch,
            weight=dr.brightness(),
        )
        return dr.finish(img.reshape(3, -1).T * 1.5, self.height, self.width)
