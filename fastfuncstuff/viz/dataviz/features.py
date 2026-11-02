"""Everything the swarm reads off a 4-D run, computed once before playback.

The swarm needs per-voxel *traits* (where it sits, which family it belongs to,
how coherent it is with its neighbours, its phase at the run's dominant period)
and per-time *drives* (each voxel's live value, each component's time course,
the global signal and DVARS). All of it comes from one detrended, z-scored
matrix and one low-rank decomposition of it, so the picture and the sound agree
about what the data is doing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from fastfuncstuff import policy_linalg as plinalg


@dataclass
class Features:
    """Per-voxel traits and per-time drives, all on ``device``.

    Shapes: N masked voxels, T time points, K components.
    """

    tr: float
    run_starts: list[int]
    ijk: np.ndarray  # (N, 3) int voxel indices, for mapping back to the grid
    anat: torch.Tensor  # (N, 3) world position, centred, ~unit radius
    func: torch.Tensor  # (N, 3) functional embedding, ~unit radius
    z: torch.Tensor  # (T, N) float16 detrended z-scores
    loadings: torch.Tensor  # (N, K) PCA spatial loadings (z = loadings @ tcs.T)
    tcs: torch.Tensor  # (T, K) PCA time courses, unit variance
    pc_var: torch.Tensor  # (K,) fraction of variance each component explains
    family: torch.Tensor  # (N,) int ICA family index
    family_weight: torch.Tensor  # (N,) membership strength in [0, 1]
    family_tcs: torch.Tensor  # (T, F) ICA family time courses, unit variance
    family_center: torch.Tensor  # (F, 3) anatomical centroid of each family
    mean: torch.Tensor  # (N,) temporal mean, normalised to [0, 1]
    tsnr: torch.Tensor  # (N,) normalised to [0, 1]
    coherence: torch.Tensor  # (N,) mean correlation with face neighbours
    communality: torch.Tensor  # (N,) fraction of variance in the low-rank part
    salience: torch.Tensor  # (N,) [0, 1]: shared with partners and with neighbours
    phase: torch.Tensor  # (N, W) radians, relative phase on each leading wave
    phase_power: torch.Tensor  # (N, W) fraction of band power each wave explains, [0, 1]
    global_signal: torch.Tensor  # (T,) mean z, unit variance
    dvars: torch.Tensor  # (T,) robust z of the frame-to-frame RMS change
    extras: dict = field(default_factory=dict)

    @property
    def n_voxels(self) -> int:
        return int(self.z.shape[1])

    @property
    def n_time(self) -> int:
        return int(self.z.shape[0])


def drift_degree(n_time: int, tr: float) -> int:
    """AFNI's ``-polort A`` rule: 1 + duration / 150 s."""
    return 1 + int(n_time * tr / 150.0)


def _detrend(
    flat: torch.Tensor, run_starts: list[int], tr: float, polort: int | None
) -> torch.Tensor:
    """Remove per-run Legendre drift (block-diagonal, one basis per run)."""
    from fastfuncstuff.glm.core import construct_polynomial_matrix
    from fastfuncstuff.glm.xval import project_out_nuisance

    n_time = flat.shape[1]
    bounds = [*run_starts, n_time]
    blocks = []
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        deg = polort if polort is not None else drift_degree(b - a, tr)
        blocks.append(construct_polynomial_matrix(b - a, deg, flat.device))
    design = torch.block_diag(*blocks)
    clean, _ = project_out_nuisance(flat, design, list(range(design.shape[1])))
    return clean


def _robust_unit(x: torch.Tensor) -> torch.Tensor:
    """Map to [0, 1] between the 2nd and 98th percentiles."""
    xs = x.float()
    sample = (
        xs
        if xs.numel() <= 1_000_000
        else xs[torch.randperm(xs.numel(), device=xs.device)[:1_000_000]]
    )
    lo, hi = torch.quantile(sample, torch.tensor([0.02, 0.98], device=xs.device))
    return ((xs - lo) / (hi - lo).clamp_min(1e-12)).clamp(0, 1)


def _rank_unit(x: torch.Tensor) -> torch.Tensor:
    r = torch.empty_like(x)
    r[torch.argsort(x)] = torch.arange(x.numel(), device=x.device, dtype=x.dtype)
    return r / max(1, x.numel() - 1)


def _centre_scale(xyz: torch.Tensor) -> torch.Tensor:
    xyz = xyz - xyz.mean(0, keepdim=True)
    r = xyz.norm(dim=1)
    return xyz / torch.quantile(r, 0.98).clamp_min(1e-12)


def neighbour_coherence(
    z: torch.Tensor, ijk: np.ndarray, shape: tuple[int, int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean correlation of each voxel with its in-mask face neighbours.

    Also returns the (N, 6) neighbour table, with a voxel standing in for any
    out-of-mask neighbour, so the swarm can pool live values spatially.

    ``z`` is (T, N) with unit variance per voxel, so the correlation is just
    the time-mean of products. Neighbours are found by a dense index volume,
    which costs one int32 per grid voxel instead of a (T, X, Y, Z) copy.
    """
    device = z.device
    n_time, n = z.shape
    idx = torch.full(shape, -1, dtype=torch.long, device=device)
    ijk_t = torch.as_tensor(ijk, device=device, dtype=torch.long)
    idx[ijk_t[:, 0], ijk_t[:, 1], ijk_t[:, 2]] = torch.arange(n, device=device)
    total = torch.zeros(n, device=device)
    count = torch.zeros(n, device=device)
    table = torch.arange(n, device=device)[:, None].repeat(1, 6)
    col = 0
    for axis in range(3):
        for step in (-1, 1):
            nb = ijk_t.clone()
            nb[:, axis] += step
            ok = (nb[:, axis] >= 0) & (nb[:, axis] < shape[axis])
            j = torch.full((n,), -1, dtype=torch.long, device=device)
            j[ok] = idx[nb[ok, 0], nb[ok, 1], nb[ok, 2]]
            has = j >= 0
            src = torch.nonzero(has).squeeze(1)
            dst = j[has]
            acc = torch.zeros(src.numel(), device=device)
            for t0 in range(0, n_time, 64):
                zt = z[t0 : t0 + 64].float()
                acc += (zt[:, src] * zt[:, dst]).sum(0)
            total[src] += acc / n_time
            count[src] += 1
            table[src, col] = dst
            col += 1
    return total / count.clamp_min(1), table


def analytic_band(z: torch.Tensor, tr: float, min_period_s: float = 6.0) -> torch.Tensor:
    """(T, N) complex analytic signal of ``z`` with periods under ``min_period_s`` removed."""
    n_time = z.shape[0]
    f = torch.fft.fft(z.float(), dim=0)
    freqs = torch.fft.fftfreq(n_time, d=tr).to(z.device)
    keep = (freqs > 0) & (freqs <= 1.0 / min_period_s)
    f = f * (2.0 * keep.to(f.dtype))[:, None]
    return torch.fft.ifft(f, dim=0)


def wave_phase(
    z: torch.Tensor, tr: float, period_s: float | None, n_waves: int = 4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-voxel relative phase of the run's leading travelling waves.

    Complex (Hilbert) PCA: the analytic signal turns a delay into a phase, so a
    response that sweeps across cortex -- a wedge, a bar, a ring -- is one
    complex component whose per-voxel phase is the sweep's arrival order, and a
    block design is one component with the same phase in every responsive
    voxel. No periodicity is assumed, which matters because real localisers and
    pRF sweeps are not periodic. With ``period_s`` the classic phase-encoded
    answer is returned instead: the Fourier phase at that period.

    Returns (phase (N, W), power (N, W), wave time courses (T, W) complex),
    where power is the fraction of each voxel's band power the wave explains.
    """
    n_time, n = z.shape
    chunk = max(1, int(2**26 // max(n_time, 1)))
    if period_s is not None:
        freqs = torch.fft.rfftfreq(n_time, d=tr).to(z.device)
        k = max(1, int((freqs - 1.0 / period_s).abs().argmin()))
        phase = torch.empty(n, 1, device=z.device)
        power = torch.empty(n, 1, device=z.device)
        for a in range(0, n, chunk):
            f = torch.fft.rfft(z[:, a : a + chunk].float(), dim=0)
            p = f.abs().square()
            phase[a : a + chunk, 0] = torch.angle(f[k])
            power[a : a + chunk, 0] = p[k] / p[1:].sum(0).clamp_min(1e-12)
        t = torch.arange(n_time, device=z.device, dtype=torch.float32) * tr
        tc = torch.exp(2j * math.pi * t / period_s)[:, None]
        return phase, power, tc
    gram = torch.zeros(n_time, n_time, dtype=torch.complex64, device=z.device)
    for a in range(0, n, chunk):
        za = analytic_band(z[:, a : a + chunk], tr)
        gram += za @ za.conj().T
    evals, evecs = torch.linalg.eigh(gram.cpu().to(torch.complex128))
    order = torch.argsort(evals, descending=True)[:n_waves]
    u = evecs[:, order].to(device=z.device, dtype=torch.complex64)  # (T, W)
    phase = torch.empty(n, u.shape[1], device=z.device)
    power = torch.empty(n, u.shape[1], device=z.device)
    for a in range(0, n, chunk):
        za = analytic_band(z[:, a : a + chunk], tr)
        load = za.T @ u.conj()  # (n, W)
        phase[a : a + chunk] = torch.angle(load)
        power[a : a + chunk] = load.abs().square() / za.abs().square().sum(0)[:, None].clamp_min(
            1e-12
        )
    return phase, power, u * math.sqrt(n_time)


def _family_ica(loadings: torch.Tensor, n_families: int, seed: int) -> torch.Tensor:
    """Spatial ICA on the PCA loadings: (N, F) independent spatial maps.

    PCA axes are variance-ordered blends of networks; the ICA rotation of the
    same subspace finds the sparse, spatially distinct groups -- the families a
    viewer recognises as "visual cortex", "the ventricles", "that vein".
    """
    from fastfuncstuff.decomposition.ica import fastica_batch

    x = loadings[:, :n_families].T.float()  # (F, N), PCA loadings are already white
    x = x / x.std(dim=1, keepdim=True).clamp_min(1e-12)
    w, _, _ = fastica_batch(x, 1, n_families, base_seed=seed)
    maps = (w[0] @ x).T  # (N, F)
    # Sign each map so its heavy tail is positive.
    skew = (maps**3).mean(0)
    return maps * torch.sign(skew + 1e-12)


def compute_features(
    data: torch.Tensor,
    mask: np.ndarray,
    affine: np.ndarray,
    tr: float,
    run_starts: list[int],
    *,
    n_components: int = 16,
    n_families: int = 10,
    period_s: float | None = None,
    polort: int | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> Features:
    """Traits and drives from ``data`` (T, X, Y, Z) restricted to ``mask``.

    ``data`` may live on the CPU; the masked matrix goes to ``mask``'s compute
    device in one copy and everything after is on-device.
    """
    device = data.device
    steps = tqdm(total=7, desc="dataviz features", leave=True, disable=not verbose)
    shape = tuple(int(s) for s in data.shape[1:])
    ijk = np.argwhere(mask)
    flat_idx = torch.as_tensor(np.ravel_multi_index(ijk.T, shape), device=device)
    flat = data.reshape(data.shape[0], -1)[:, flat_idx].T.contiguous().float()  # (N, T)
    n, n_time = flat.shape

    raw_mean = flat.mean(1)
    resid = _detrend(flat, run_starts, tr, polort)
    del flat
    sd = resid.std(1).clamp_min(1e-6)
    tsnr = raw_mean.abs() / sd
    zf = resid / sd[:, None]  # (N, T)
    del resid
    steps.update()

    # Low rank through the (T, T) Gram: T is hundreds, N is hundreds of thousands.
    gram = zf.T @ zf
    evals, evecs = plinalg.eigh(gram.double())
    order = torch.argsort(evals, descending=True)
    k = min(n_components, n_time - 1)
    evals = evals[order[:k]].float().clamp_min(1e-12)
    tcs = evecs[:, order[:k]].float()  # (T, K), orthonormal columns
    loadings = zf @ tcs  # (N, K): z ≈ loadings @ tcs.T
    pc_var = evals / gram.diagonal().sum()
    tcs_unit = tcs * math.sqrt(n_time)
    loadings_unit = loadings / math.sqrt(n_time)
    steps.update()

    n_fam = min(n_families, k)
    ica_maps = _family_ica(loadings, n_fam, seed)  # (N, F)
    zmaps = ica_maps / ica_maps.std(0, keepdim=True).clamp_min(1e-12)
    best = zmaps.max(1)
    family = best.indices
    family_weight = ((best.values - 1.0) / 3.0).clamp(0, 1)
    # Family time course: membership-weighted mean of the members' z.
    member_w = zmaps.clamp_min(0) ** 2
    family_tcs = (zf.T @ member_w) / member_w.sum(0).clamp_min(1e-12)
    family_tcs = (family_tcs - family_tcs.mean(0)) / family_tcs.std(0).clamp_min(1e-12)
    steps.update()

    world = torch.as_tensor(
        affine[:3, :3] @ ijk.T + affine[:3, 3:4], device=device, dtype=torch.float32
    ).T
    anat = _centre_scale(world)
    family_center = torch.zeros(n_fam, 3, device=device).index_add_(0, family, anat)
    family_center /= torch.bincount(family, minlength=n_fam).clamp_min(1)[:, None]

    # Functional layout: each voxel's direction in correlation space. Two voxels
    # whose rows of the row-normalised loading matrix agree are correlated, so
    # projecting those rows onto their own top three axes puts correlated voxels
    # side by side regardless of where they sit in the head.
    rows = loadings_unit / loadings_unit.norm(dim=1, keepdim=True).clamp_min(1e-12)
    rows = rows - rows.mean(0, keepdim=True)
    _, _, vh = plinalg.svd(rows[:: max(1, n // 200_000)].double(), full_matrices=False)
    func = _centre_scale(rows @ vh[:3].T.float())
    steps.update()

    zt = zf.T.contiguous()  # (T, N)
    del zf
    coherence, neighbours = neighbour_coherence(zt, ijk, shape)
    # Salience: what fraction of a voxel's variance is *shared* -- with its
    # correlated partners anywhere (communality) and with its neighbours
    # (coherence). Task cortex and networks score high; white matter, whose
    # fluctuation is mostly independent thermal noise however high its tSNR,
    # scores low. Ranks, so neither measure's scale dominates.
    communality = (loadings_unit.square().sum(1)).clamp(0, 1)
    salience = (_rank_unit(communality) * _rank_unit(coherence)).sqrt()
    steps.update()

    phase, phase_power, wave_tcs = wave_phase(zt, tr, period_s)
    steps.update()

    g = zt.float().mean(1)
    g = (g - g.mean()) / g.std().clamp_min(1e-12)
    dv = torch.zeros(n_time, device=device)
    dv[1:] = (zt[1:].float() - zt[:-1].float()).square().mean(1).sqrt()
    dv[0] = dv[1]
    med = dv.median()
    mad = (dv - med).abs().median().clamp_min(1e-6) * 1.4826
    dv = (dv - med) / mad
    steps.update()
    steps.close()

    return Features(
        tr=tr,
        run_starts=list(run_starts),
        ijk=ijk,
        anat=anat,
        func=func,
        z=zt.half(),
        loadings=loadings_unit,
        tcs=tcs_unit,
        pc_var=pc_var,
        family=family,
        family_weight=family_weight,
        family_tcs=family_tcs,
        family_center=family_center,
        mean=_robust_unit(raw_mean),
        tsnr=_robust_unit(tsnr),
        coherence=coherence,
        communality=communality,
        salience=salience,
        phase=phase,
        phase_power=torch.stack([_robust_unit(p) for p in phase_power.T], 1),
        global_signal=g,
        dvars=dv,
        extras={"wave_tcs": wave_tcs, "neighbours": neighbours},
    )


def load_features(
    paths: list[str | Path],
    device: torch.device,
    *,
    mask_path: str | Path | None = None,
    tr: float | None = None,
    max_voxels: int | None = None,
    seed: int = 0,
    verbose: bool = True,
    **kwargs,
) -> tuple[Features, dict]:
    """Load one or more runs on one grid and compute their features.

    Runs are concatenated in time with per-run drift removal. Returns the
    features and a small info dict (grid shape, affine, TR, source names).
    """
    from fastfuncstuff.processing.io import load_image
    from fastfuncstuff.processing.mask import automask

    vols = []
    affine = None
    header_tr = None
    for p in paths:
        data, info = load_image(p)  # (T, Z, Y, X) per load_image; we keep it as is
        if data.ndim != 4:
            raise ValueError(f"{p}: expected a 4-D time series, got shape {tuple(data.shape)}")
        if affine is None:
            affine = info["affine"]
            zooms = info["header"].get_zooms()
            header_tr = float(zooms[3]) if len(zooms) > 3 else None
        elif data.shape[1:] != vols[0].shape[1:]:
            raise ValueError(f"{p}: grid {tuple(data.shape[1:])} differs from the first run's")
        vols.append(data)
    assert affine is not None
    run_tr = tr if tr is not None else header_tr
    if not run_tr or run_tr <= 0:
        raise ValueError("no usable TR in the header; pass -TR")
    run_starts = list(np.cumsum([0] + [v.shape[0] for v in vols[:-1]]))
    data = torch.cat(vols, 0) if len(vols) > 1 else vols[0]
    del vols

    # load_image returns (T, Z, Y, X) -- the affine indexes (X, Y, Z), so
    # transpose the grid axes back to the affine's order once, here.
    data = data.permute(0, 3, 2, 1)
    if mask_path is not None:
        m, _ = load_image(mask_path)
        mask = m.permute(2, 1, 0).numpy() > 0
    else:
        mask = (
            automask(data[: run_starts[1] if len(run_starts) > 1 else None].mean(0), dilate_extra=0)
            .cpu()
            .numpy()
            > 0
        )
    if max_voxels is not None and mask.sum() > max_voxels:
        rng = np.random.default_rng(seed)
        keep = np.argwhere(mask)
        drop = rng.choice(len(keep), len(keep) - max_voxels, replace=False)
        mask[tuple(keep[drop].T)] = False

    data = data.to(device)
    feats = compute_features(
        data, mask, affine, run_tr, run_starts, seed=seed, verbose=verbose, **kwargs
    )
    info = {
        "shape": tuple(data.shape[1:]),
        "affine": affine,
        "tr": run_tr,
        "names": [Path(p).name for p in paths],
    }
    return feats, info
