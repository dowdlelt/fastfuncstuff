"""Several values per voxel, drawn as one colour.

Echoes, inversions, betas, component maps: anything with N >= 2 values at a
voxel. A colour view is a linear map from those N values to three colour
coordinates, fitted once and applied per plane, so it costs a matmul at draw
time and plays as a movie when its channels are time series. Colour and time
never compete for one display axis, which is the problem a 4-D RGB file runs
into in every viewer that stores the colour.

Why not just put three channels into R, G and B: channels that measure the same
tissue are highly correlated, so voxels crowd the grey diagonal and colour
carries one dimension -- "how much did it decay" -- as a single brown hue. Each
basis is a different answer to that:

``pick``      three channels, each standardised. The baseline.
``dstretch``  decorrelation stretch (the Karhunen-Loeve whitening JPL used on
              planetary imagery and DStretch on rock art): whiten in the
              eigenbasis, rotate back, so what *differs* between channels gets
              the colour and hues still lean towards their channels.
``pca``       the whitened principal components themselves: most contrast,
              no fixed meaning of a hue.
``te``        a fixed basis rather than a fitted one: Legendre polynomials in
              echo time. On log signal, P0 is the level, P1 the decay rate
              (R2*) and P2 the departure from a single exponential; on percent
              change, P0 is the TE-independent (S0) part and P1 the
              TE-proportional (BOLD) part. Hues mean the same in every subject.
``custom``    weights typed by hand, one row per colour coordinate, on the
              standardised channels -- for when you know the contrast you want.

The fit is split from the :class:`Look`. A fit (which basis, from which
statistics) is tens of milliseconds; a look -- each coordinate's shift and
stretch, a hue rotation, a lightness gamma -- is applied per draw and costs
nothing, which is what makes it a live contrast manipulator rather than a
render setting.

Two colour spaces. ``rgb`` maps the three coordinates to the three primaries.
``oklab`` puts the first coordinate in lightness and the other two on the
opponent axes (b: blue-yellow, a: green-red), so a change of hue never reads
as a change of brightness and anatomy stays legible under the tint.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import Tensor

from fastfuncstuff.utils import quantiles

BASES = ("pick", "dstretch", "pca", "te", "custom")
SPACES = ("auto", "rgb", "oklab")
SIGNALS = ("raw", "log", "psc")
LIGHTNESS = ("basis", "anatomy")

#: Lightness range in Oklab. The floor keeps the darkest tissue distinguishable
#: from the masked background; the ceiling leaves headroom for chroma, since a
#: saturated colour at L=1 is out of the sRGB gamut and clips to white.
L_FLOOR, L_CEIL = 0.2, 0.8

#: How hard the anatomy can be nudged by the basis's first coordinate when the
#: lightness comes from anatomy -- enough to see S0 flicker, not enough to
#: repaint the anatomy.
ANATOMY_NUDGE = 0.15


@dataclass(frozen=True)
class ChromaSpec:
    """How the 3 x N map is fitted. Changing any of it refits (tens of ms)."""

    basis: str = "dstretch"
    signal: str = "log"
    #: Eigenvalues below ``floor * largest`` are not whitened further. Without
    #: it the weakest component -- usually noise -- is amplified the most,
    #: which on three echoes shows up as speckle.
    floor: float = 0.05
    #: Channel indices shown as the three coordinates by ``pick`` and
    #: ``dstretch``; -1 is first / middle / last.
    pick: tuple[int, int, int] = (-1, -1, -1)
    #: Echo times in ms for the ``te`` basis; empty reads sidecars.
    te: str = ""
    #: ``custom`` weights: three rows separated by ``;``, N numbers each.
    weights: str = ""

    def __post_init__(self) -> None:
        if self.basis not in BASES:
            raise ValueError(f"basis must be one of {BASES}, not {self.basis!r}")
        if self.signal not in SIGNALS:
            raise ValueError(f"signal must be one of {SIGNALS}, not {self.signal!r}")

    def picked(self, n: int) -> tuple[int, int, int]:
        auto = (0, 1, 1) if n == 2 else ((0, n // 2, n - 1) if n > 3 else (0, 1, 2))
        r, g, b = (a if p < 0 else p for p, a in zip(self.pick, auto, strict=True))
        if max(r, g, b) >= n:
            raise ValueError(
                f"R/G/B channel {max(r, g, b)} does not exist; there are {n} (0..{n - 1})"
            )
        return (r, g, b)

    def te_ms(self, n: int) -> list[float] | None:
        """Typed echo times, or ``None`` when they are to be read from sidecars."""
        if not self.te.strip():
            return None
        try:
            tes = [float(v) for v in self.te.replace(";", ",").split(",") if v.strip()]
        except ValueError:
            raise ValueError(f"TE must be comma-separated numbers, got {self.te!r}") from None
        if len(tes) != n:
            raise ValueError(f"TE lists {len(tes)} echo times for {n} channels")
        if len(set(tes)) < 2:
            raise ValueError("echo times must differ")
        return tes

    def custom_weights(self, n: int) -> np.ndarray:
        rows = [r for r in self.weights.split(";") if r.strip()]
        if len(rows) != 3:
            raise ValueError("custom weights need three rows separated by ';'")
        try:
            W = np.array([[float(v) for v in r.split(",")] for r in rows])
        except ValueError:
            raise ValueError(f"custom weights must be numbers, got {self.weights!r}") from None
        if W.shape != (3, n):
            raise ValueError(f"custom weights need {n} numbers per row, one per channel")
        return W


@dataclass(frozen=True)
class Look:
    """How the fitted coordinates become colour. Applied per draw; no refit.

    ``shift`` and ``stretch`` are per coordinate, in window units: the window
    is the stats region's median +- ``spread`` standard deviations, so a shift
    of 0.5 moves the centre half a window and a stretch of 2 doubles the
    contrast. A negative stretch flips that coordinate's direction.
    """

    space: str = "oklab"
    lightness: str = "basis"
    #: Window half-width in standard deviations of the stats region. Larger
    #: is calmer; smaller saturates and clips sooner.
    spread: float = 2.5
    chroma: float = 0.25
    shift: tuple[float, float, float] = (0.0, 0.0, 0.0)
    stretch: tuple[float, float, float] = (1.0, 1.0, 1.0)
    #: Rotation of the two chroma coordinates, degrees: oklab hue, or the G-B
    #: plane in rgb. Moves which tissue lands on which colour.
    hue: float = 0.0
    #: Lightness gamma: above 1 darkens the dim end, below 1 lifts it.
    gamma: float = 1.0

    def __post_init__(self) -> None:
        if self.space not in ("rgb", "oklab"):
            raise ValueError(f"space must be rgb or oklab, not {self.space!r}")
        if self.lightness not in LIGHTNESS:
            raise ValueError(f"lightness must be one of {LIGHTNESS}, not {self.lightness!r}")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if self.spread <= 0:
            raise ValueError("spread must be positive")


def resolve_space(space: str, basis: str) -> str:
    """``auto``: Oklab where the first coordinate is a level (te, pca), RGB otherwise."""
    if space != "auto":
        return space
    return "oklab" if basis in ("te", "pca") else "rgb"


# ---------------------------------------------------------------------------
# signal transforms
# ---------------------------------------------------------------------------


def transform(x: Tensor, signal: str, *, eps: float, mean: Tensor | None = None) -> Tensor:
    """Channels (N, ...) into the space the basis is fitted in.

    ``log`` because MR signal decays multiplicatively -- log S is linear in R2*,
    so the te basis is a straight-line fit and a covariance is not dominated by
    the brightest tissue. ``psc`` reads change from each voxel's own mean,
    which is what a breath-hold or a task is.
    """
    if signal == "raw":
        return x
    if signal == "log":
        return torch.log(x.clamp(min=eps))
    if mean is None:
        raise ValueError("percent change needs each channel's mean")
    ok = mean > eps
    return torch.where(ok, 100.0 * (x / torch.where(ok, mean, 1.0) - 1.0), 0.0)


def echo_basis(te_ms: Sequence[float]) -> np.ndarray:
    """(3, N): least-squares Legendre coefficients in TE, decay rate signed positive.

    Rows are the level, the decay rate and the curvature. P1 is negated so that
    faster decay -- shorter T2*, iron, deoxygenated blood -- is the positive
    direction and lands warm, slower decay (CSF) cool. With two echoes there
    is no curvature and its row is zero.
    """
    te = np.asarray(te_ms, dtype=np.float64)
    span = te.max() - te.min()
    if span <= 0:
        raise ValueError("echo times must differ")
    t = 2.0 * (te - te.min()) / span - 1.0
    order = min(3, len(te))
    P = np.stack([np.ones_like(t), t, 0.5 * (3.0 * t**2 - 1.0)][:order], 1)
    rows = np.linalg.pinv(P)
    out = np.zeros((3, len(te)))
    out[:order] = rows
    out[1] *= -1.0
    return out


# ---------------------------------------------------------------------------
# the fitted map
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChromaMap:
    """Transformed channels in, coordinates out. Fitted by :func:`fit_map`."""

    #: (3, N), applied to ``(x - mu) / sd``.
    W: Tensor
    mu: Tensor
    sd: Tensor
    #: Per coordinate: median and standard deviation over the stats region.
    #: The window is ``center +- look.spread * width``.
    center: Tensor
    width: Tensor
    #: Lightness window on the first coordinate (robust percentiles, because
    #: it is a level and its distribution is skewed, not a deviation).
    l_lo: float
    l_hi: float

    def to(self, device: torch.device) -> ChromaMap:
        return replace(
            self,
            W=self.W.to(device),
            mu=self.mu.to(device),
            sd=self.sd.to(device),
            center=self.center.to(device),
            width=self.width.to(device),
        )

    def coords(self, x: Tensor) -> Tensor:
        """(N, ...) -> (3, ...): the basis coordinates, unwindowed."""
        n = x.shape[0]
        z = (x.reshape(n, -1) - self.mu[:, None]) / self.sd[:, None]
        return (self.W @ z).reshape(3, *x.shape[1:])

    def unit(self, y: Tensor, look: Look) -> Tensor:
        """Coordinates in window units, the look's shift and stretch applied."""
        shift, stretch = look_tensors(look, y)
        return unit_coords(y, self.center, self.width, look.spread, shift, stretch)

    def colors(self, x: Tensor, look: Look, lightness: Tensor | None = None) -> Tensor:
        """(N, ...) transformed channels -> (..., 3) sRGB in [0, 1].

        ``lightness`` (in [0, 1], same spatial shape) replaces the first
        coordinate's lightness with anatomy, which the first coordinate then
        only nudges.
        """
        y = self.coords(x)
        shift, stretch = look_tensors(look, y)
        return paint(
            y, self.center, self.width, self.l_lo, self.l_hi, look, shift, stretch, lightness
        )


def look_tensors(look: Look, like: Tensor) -> tuple[Tensor, Tensor]:
    """The look's shift and stretch as tensors on ``like``'s device."""
    shift = torch.tensor(look.shift, dtype=like.dtype, device=like.device)
    stretch = torch.tensor(look.stretch, dtype=like.dtype, device=like.device)
    return shift, stretch


def unit_coords(
    y: Tensor, center: Tensor, width: Tensor, spread: float, shift: Tensor, stretch: Tensor
) -> Tensor:
    """(3, ...) coordinates into window units: ``stretch * ((y - c) / (spread * w) - shift)``."""
    shape = (3,) + (1,) * (y.ndim - 1)
    u = (y - center.reshape(shape)) / (spread * width.reshape(shape))
    return (u - shift.reshape(shape)) * stretch.reshape(shape)


def paint(
    y: Tensor,
    center: Tensor,
    width: Tensor,
    l_lo: float | Tensor,
    l_hi: float | Tensor,
    look: Look,
    shift: Tensor,
    stretch: Tensor,
    lightness: Tensor | None = None,
) -> Tensor:
    """(3, ...) coordinates -> (..., 3) sRGB in [0, 1]: the one colour pipeline.

    Shift and stretch come in as tensors so the contrast optimiser can
    differentiate through exactly what the screen draws, rather than a copy
    of it that could drift.
    """
    u = unit_coords(y, center, width, look.spread, shift, stretch)
    c, s = math.cos(math.radians(look.hue)), math.sin(math.radians(look.hue))
    u1, u2 = c * u[1] - s * u[2], s * u[1] + c * u[2]
    if look.space == "rgb":
        rgb = (0.5 + 0.5 * torch.stack([u[0], u1, u2], -1)).clamp(0.0, 1.0)
        if look.gamma != 1.0:
            rgb = rgb.clamp(min=1e-6) ** (1.0 / look.gamma)
        if lightness is not None:
            rgb = rgb * lightness.unsqueeze(-1) ** look.gamma
        return rgb
    if lightness is None:
        mid, half = 0.5 * (l_lo + l_hi), 0.5 * (l_hi - l_lo)
        half = half.clamp(min=1e-12) if isinstance(half, Tensor) else max(half, 1e-12)
        level = (0.5 + 0.5 * ((y[0] - mid) / half - shift[0]) * stretch[0]).clamp(0.0, 1.0)
        L = L_FLOOR + (L_CEIL - L_FLOOR) * level.clamp(min=1e-6) ** look.gamma
    else:
        L = L_FLOOR + (L_CEIL - L_FLOOR) * lightness.clamp(min=1e-6) ** look.gamma
        L = L + ANATOMY_NUDGE * torch.tanh(u[0])
    # tanh rather than a clip: saturation rolls off instead of flattening, so
    # the strongest voxels are still distinguishable from strong ones.
    b = look.chroma * torch.tanh(1.25 * u1)
    a = look.chroma * torch.tanh(1.25 * u2)
    return oklab_to_srgb(L, a, b)


def srgb_to_oklab(rgb: Tensor) -> Tensor:
    """(..., 3) gamma-encoded sRGB in [0, 1] to Oklab, where distance is perceived difference."""
    lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb.clamp(min=0.0) + 0.055) / 1.055) ** 2.4)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = (v.clamp(min=1e-9) ** (1.0 / 3.0) for v in (l, m, s))
    return torch.stack(
        [
            0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
        ],
        -1,
    )


def oklab_to_srgb(L: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Oklab (Ottosson 2020) to gamma-encoded sRGB, clipped to the gamut. (..., 3)."""
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_**3, m_**3, s_**3
    rgb = torch.stack(
        [
            4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960771 * l - 0.7034186147 * m + 1.7076147010 * s,
        ],
        -1,
    ).clamp(0.0, 1.0)
    return torch.where(
        rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb.clamp(min=1e-12) ** (1 / 2.4) - 0.055
    )


def _eig(cov: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decomposition of an N x N covariance, largest first, floored."""
    lam, E = np.linalg.eigh(np.atleast_2d(cov))
    lam, E = lam[::-1], E[:, ::-1]
    # Sign is arbitrary out of eigh; fixing it makes the same data always
    # draw in the same colours, and puts a component that loads every channel
    # the same way on its bright side.
    E = E * np.where(E.sum(0) < 0, -1.0, 1.0)
    lam = np.maximum(lam, max(floor, 1e-6) * max(lam[0], 1e-12))
    return lam, E


def fit_map(
    samples: Tensor | np.ndarray, spec: ChromaSpec, te_ms: Sequence[float] | None = None
) -> ChromaMap:
    """Fit a colour map to transformed channel samples ``(M, N)`` from the stats region.

    The samples stay where they are -- on the GPU for a real dataset -- and
    only the N x N covariance comes to the host, in float64, for the eigen
    decomposition; that is a few channels squared, not a million voxels.
    """
    x = torch.as_tensor(samples, dtype=torch.float32)
    x = x[torch.isfinite(x).all(dim=1)]
    if x.shape[0] < 8:
        raise ValueError("too few voxels in the statistics region to fit colours")
    n = x.shape[1]
    pick = np.zeros((3, n))
    for row, c in enumerate(spec.picked(n)):
        pick[row, c] = 1.0

    if spec.basis == "te":
        # A fixed basis on the signal itself: standardising each channel
        # would reweight the echoes and bend the straight-line fit.
        mu, sd = np.zeros(n), np.ones(n)
        te = np.arange(n, dtype=float) if te_ms is None or len(te_ms) != n else np.asarray(te_ms)
        W = echo_basis(te)
    else:
        mu_t = x.mean(0)
        sd_t = x.std(0)
        sd_t = torch.where(sd_t > 0, sd_t, torch.ones_like(sd_t))
        mu, sd = mu_t.cpu().double().numpy(), sd_t.cpu().double().numpy()
        if spec.basis == "pick":
            W = pick
        elif spec.basis == "custom":
            W = spec.custom_weights(n)
        else:
            z = (x - mu_t) / sd_t
            cov = (z.T @ z / max(z.shape[0] - 1, 1)).cpu().double().numpy()
            lam, E = _eig(cov, spec.floor)
            if spec.basis == "dstretch":
                W = pick @ (E @ np.diag(lam**-0.5) @ E.T)
            else:  # pca
                k = min(3, n)
                W = np.zeros((3, n))
                W[:k] = (E[:, :k] / np.sqrt(lam[:k])).T

    def t(v: np.ndarray) -> Tensor:
        return torch.as_tensor(np.asarray(v), dtype=torch.float32)

    W_t, mu_t, sd_t = t(W).to(x.device), t(mu).to(x.device), t(sd).to(x.device)
    y = ((x - mu_t) / sd_t) @ W_t.T
    center = y.median(dim=0).values
    width = y.std(0)
    width = torch.where(width > 0, width, torch.ones_like(width))
    l_lo, l_hi = quantiles(y[:, 0], (0.02, 0.995))

    return ChromaMap(
        W=W_t,
        mu=mu_t,
        sd=sd_t,
        center=center,
        width=width,
        l_lo=float(l_lo),
        l_hi=float(l_hi),
    )


# ---------------------------------------------------------------------------
# finding channels
# ---------------------------------------------------------------------------

_DIGITS = re.compile(r"\d+")


def sibling_order(names: Sequence[str], seed: int) -> list[int]:
    """Indices of names that differ from ``names[seed]`` only in one number, by that number.

    ``..._SBRef_e1.nii.gz``, ``_e2``, ``_e3`` are one acquisition's echoes; the
    run number, TR and voxel size elsewhere in the name are the same, and a
    different run of the same sequence differs in a different number. Returns
    just ``[seed]`` when nothing matches.
    """
    skeleton = _DIGITS.split(names[seed])
    numbers = _DIGITS.findall(names[seed])
    best: list[int] = [seed]
    for position in range(len(numbers)):
        found: list[tuple[int, int]] = []
        for i, name in enumerate(names):
            if _DIGITS.split(name) != skeleton:
                continue
            other = _DIGITS.findall(name)
            if all(
                o == s for j, (o, s) in enumerate(zip(other, numbers, strict=True)) if j != position
            ):
                found.append((int(other[position]), i))
        if len({v for v, _ in found}) > len(best):
            best = [i for _, i in sorted(found)]
    return best
