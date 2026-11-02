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
from fastfuncstuff.memory import spectral_voxel_chunk


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
    loadings_all: torch.Tensor  # (N, Kmax) PCA spatial loadings (z = loadings @ tcs.T)
    tcs_all: torch.Tensor  # (T, Kmax) PCA time courses, unit variance
    pc_var_all: torch.Tensor  # (Kmax,) fraction of variance each component explains
    k_active: int  # components in use (the rest are kept for the slider)
    k_suggested: int  # Marchenko-Pastur count, the default's ceiling
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
    def loadings(self) -> torch.Tensor:
        return self.loadings_all[:, : self.k_active]

    @property
    def tcs(self) -> torch.Tensor:
        return self.tcs_all[:, : self.k_active]

    @property
    def pc_var(self) -> torch.Tensor:
        return self.pc_var_all[: self.k_active]

    def refit_families(self, n_families: int, seed: int = 0) -> None:
        """Recompute the families in place for a new family count."""
        for k, v in assign_families(self.loadings, self.z, self.anat, n_families, seed).items():
            setattr(self, k, v)
        self.flag_families()

    def set_components(self, k: int, n_families: int, seed: int = 0) -> None:
        """Use the top ``k`` components: communality, salience and families follow."""
        self.k_active = max(2, min(int(k), self.loadings_all.shape[1]))
        self.communality = self.loadings.square().sum(1).clamp(0, 1)
        self.salience = (_rank_unit(self.communality) * _rank_unit(self.coherence)).sqrt()
        self.refit_families(min(n_families, self.k_active), seed)

    def flag_families(self) -> None:
        """Mark families that look like noise rather than signal, without a model.

        Hints, not verdicts, so they are shown and offered for muting, never
        acted on silently. ``extras["family_noise"]`` keeps the numbers:

        * ``fast``   fraction of the family's power above 0.1 Hz (aliased
          cardiac/respiratory, thermal) -- flagged over 0.45;
        * ``global`` |r| with the global signal -- flagged over 0.75;
        * ``motion`` |r| between the family's frame-to-frame change and
          DVARS -- flagged over 0.45;
        * ``edge``   members' share of brain-boundary voxels relative to the
          brain's -- flagged over 2x (edge and ghosting components).
        """
        tcs = self.family_tcs.float()
        n_time, n_fam = tcs.shape
        freqs = torch.fft.rfftfreq(n_time, d=self.tr).to(tcs.device)[1:]
        pw = torch.fft.rfft(tcs - tcs.mean(0), dim=0).abs().square()[1:]
        fast = (
            pw[freqs > 0.1].sum(0) / pw.sum(0).clamp_min(1e-12)
            if bool((freqs > 0.1).any())
            else torch.zeros(n_fam, device=tcs.device)
        )

        def corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            a = a - a.mean(0)
            b = (b - b.mean())[:, None]
            return (a * b).sum(0) / (a.norm(dim=0) * b.norm()).clamp_min(1e-12)

        r_gs = corr(tcs, self.global_signal.float()).abs()
        change = torch.zeros_like(tcs)
        change[1:] = (tcs[1:] - tcs[:-1]).abs()
        r_dv = corr(change, self.dvars.float()).abs()
        nb = self.extras["neighbours"]
        boundary = (nb == torch.arange(len(nb), device=nb.device)[:, None]).any(1).float()
        w = self.family_weight.clamp_min(0.05)
        share = torch.zeros(n_fam, device=w.device).index_add_(0, self.family, w * boundary)
        share = share / torch.zeros(n_fam, device=w.device).index_add_(0, self.family, w).clamp_min(
            1e-9
        )
        edge = share / boundary.mean().clamp_min(1e-6)
        self.extras["family_noise"] = {"fast": fast, "global": r_gs, "motion": r_dv, "edge": edge}
        flags = []
        for i in range(n_fam):
            f = []
            if fast[i] > 0.45:
                f.append("fast")
            if r_gs[i] > 0.75:
                f.append("global")
            if r_dv[i] > 0.45:
                f.append("motion")
            if edge[i] > 2.0:
                f.append("edge")
            flags.append(f)
        self.extras["family_flags"] = flags

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
    chunk = spectral_voxel_chunk(n, n_time, z.device)
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
    evals, evecs = plinalg.eigh(gram.cpu().to(torch.complex128))
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


#: Band edges in Hz: slow (drift, slow networks), mid (BOLD task and resting
#: fluctuations), fast (aliased physiology, thermal noise).
BAND_EDGES_HZ = (0.03, 0.1)


def band_powers(z: torch.Tensor, tr: float) -> torch.Tensor:
    """(N, 3) fraction of each voxel's power in slow / mid / fast bands.

    A spectral fingerprint: drawn as RGB it separates BOLD-like voxels from
    drifting ones and from physiology aliased into the fast band, with no model.
    """
    n_time, n = z.shape
    freqs = torch.fft.rfftfreq(n_time, d=tr).to(z.device)[1:]
    lo, hi = BAND_EDGES_HZ
    sel = [freqs < lo, (freqs >= lo) & (freqs < hi), freqs >= hi]
    out = torch.empty(n, 3, device=z.device)
    chunk = spectral_voxel_chunk(n, n_time, z.device)
    for a in range(0, n, chunk):
        pw = torch.fft.rfft(z[:, a : a + chunk].float(), dim=0).abs().square()[1:]
        tot = pw.sum(0).clamp_min(1e-12)
        for b, m in enumerate(sel):
            out[a : a + chunk, b] = pw[m].sum(0) / tot if bool(m.any()) else 0.0
    return out


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


def assign_families(
    loadings: torch.Tensor, z_tn: torch.Tensor, anat: torch.Tensor, n_families: int, seed: int
) -> dict[str, torch.Tensor]:
    """Spatial ICA of the loadings into families, and what each family does.

    Returns the Features fields ``family`` (argmax of the z-scored ICA maps),
    ``family_weight`` ((max z - 1) / 3 in [0, 1]), ``family_tcs`` (membership-
    weighted mean of the members' z, unit variance) and ``family_center``.
    Cheap enough (~0.5 s) to rerun live when the family count changes.
    """
    n_fam = max(2, min(n_families, loadings.shape[1]))
    ica_maps = _family_ica(loadings, n_fam, seed)  # (N, F)
    zmaps = ica_maps / ica_maps.std(0, keepdim=True).clamp_min(1e-12)
    best = zmaps.max(1)
    family = best.indices
    member_w = zmaps.clamp_min(0) ** 2
    tcs = (z_tn.float() @ member_w) / member_w.sum(0).clamp_min(1e-12)
    tcs = (tcs - tcs.mean(0)) / tcs.std(0).clamp_min(1e-12)
    centre = torch.zeros(n_fam, 3, device=anat.device).index_add_(0, family, anat)
    centre /= torch.bincount(family, minlength=n_fam).clamp_min(1)[:, None]
    return {
        "family": family,
        "family_weight": ((best.values - 1.0) / 3.0).clamp(0, 1),
        "family_tcs": tcs,
        "family_center": centre,
    }


def suggest_components(evals_desc: np.ndarray, n_voxels: int) -> int:
    """How many components stand above the Marchenko-Pastur noise edge.

    Each voxel is an observation of a T-dimensional series; the number of
    independent observations is fewer than the voxel count by however much the
    data is spatially smooth, but this count is used as a ceiling for a
    default, so the voxel count is fine -- it errs toward more components,
    which the pooling and the salience gate are there to tolerate.
    """
    from fastfuncstuff.decomposition.model_order import mp_signal_count

    ev = np.asarray(evals_desc, dtype=np.float64)
    ev = ev[ev > 0]
    count, _, _ = mp_signal_count(ev, max(1, int(n_voxels)))
    return max(2, int(count))


def compute_features(
    data: torch.Tensor,
    mask: np.ndarray,
    affine: np.ndarray,
    tr: float,
    run_starts: list[int],
    *,
    n_components: int | None = None,
    n_families: int = 10,
    max_components: int = 64,
    fwhm_mm: float = 6.0,
    remove_global: bool = True,
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

    raw_mean = flat.mean(1)  # the anatomy (dust, slices) stays sharp
    coherence = neighbours = None
    if fwhm_mm > 0:
        # Neighbour coherence from the unsmoothed series: after a 6 mm blur every
        # neighbour pair correlates at ~0.9 (dynaloc median 0.93) and the measure
        # only reports the kernel, which would flatten salience.
        raw = _detrend(flat, run_starts, tr, polort)
        raw = (raw / raw.std(1, keepdim=True).clamp_min(1e-6)).T.contiguous()
        coherence, neighbours = neighbour_coherence(raw, ijk, shape)
        del raw
        # Smooth inside the mask before anything is estimated: on raw data the
        # leading components and families are largely thermal and physiological
        # noise, and a few mm of shared signal is what ICA needs to find
        # networks. Normalised by the blurred mask, so edge voxels are not pulled
        # toward zero.
        from fastfuncstuff.cli_utils import blur_masked_data

        zooms = tuple(float(v) for v in np.linalg.norm(np.asarray(affine)[:3, :3], axis=0))
        flat = blur_masked_data(
            flat,
            fwhm_mm=fwhm_mm,
            volume_shape=shape,
            voxel_sizes=zooms,
            mask_flat=mask.reshape(-1),
            run_starts=run_starts,
            device=device,
            verbose=False,
        )
    resid = _detrend(flat, run_starts, tr, polort)
    del flat
    sd = resid.std(1).clamp_min(1e-6)
    tsnr = raw_mean.abs() / sd
    zf = resid / sd[:, None]  # (N, T)
    del resid
    # The whole-brain signals, taken before anything is removed: they are what
    # tints, flashes and jolts every scene.
    g = zf.mean(0)
    dv = torch.zeros(n_time, device=device)
    dv[1:] = (zf[:, 1:] - zf[:, :-1]).square().mean(0).sqrt()
    dv[0] = dv[1]
    if remove_global:
        # Families and live values come from what the global signal leaves.
        # On unprocessed data -- smoothed especially, once thermal noise is
        # gone -- every family otherwise rides the global signal (all ten
        # flagged on dynaloc); "things that affect all voxels" are drawn as
        # colour and motion instead, from g and DVARS above.
        gn = g - g.mean()
        beta = (zf @ gn) / (gn @ gn).clamp_min(1e-12)
        zf = zf - beta[:, None] * gn[None]
        zf = zf / zf.std(1, keepdim=True).clamp_min(1e-6)
    steps.update()

    # Low rank through the (T, T) Gram: T is hundreds, N is hundreds of thousands.
    # The eigendecomposition gives every component for free, so keep up to
    # ``max_components`` and let a slider choose how many are in use.
    gram = zf.T @ zf
    # Device first, then float64 (on Metal a combined cast returns zeros and
    # .double() raises); a (T, T) eigh is cheapest on the CPU everywhere.
    evals_all, evecs = plinalg.eigh(gram.to("cpu").double())
    evecs = evecs.float().to(device)
    order = torch.argsort(evals_all, descending=True)
    k_max = min(max_components, n_time - 1)
    evals = evals_all[order[:k_max]].float().clamp_min(1e-12)  # on the CPU
    tcs = evecs[:, order[:k_max]].float()  # (T, Kmax), orthonormal columns
    loadings = zf @ tcs  # (N, Kmax): z ≈ loadings @ tcs.T
    pc_var = evals.to(device) / gram.diagonal().sum()
    tcs_unit = tcs * math.sqrt(n_time)
    loadings_unit = loadings / math.sqrt(n_time)
    k_suggested = suggest_components(evals_all[order].cpu().numpy(), n)
    k_active = min(k_max, n_components if n_components is not None else min(k_suggested, 32))
    steps.update()

    world = torch.as_tensor(
        affine[:3, :3] @ ijk.T + affine[:3, 3:4], device=device, dtype=torch.float32
    ).T
    anat = _centre_scale(world)
    fams = assign_families(loadings[:, :k_active], zf.T, anat, min(n_families, k_active), seed)
    steps.update()

    # Functional layout: each voxel's direction in correlation space. Two voxels
    # whose rows of the row-normalised loading matrix agree are correlated, so
    # projecting those rows onto their own top three axes puts correlated voxels
    # side by side regardless of where they sit in the head.
    rows = loadings_unit[:, :k_active]
    rows = rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
    rows = rows - rows.mean(0, keepdim=True)
    _, _, vh = plinalg.svd(rows[:: max(1, n // 200_000)].to("cpu").double(), full_matrices=False)
    func = _centre_scale(rows @ vh[:3].T.float().to(rows.device))
    steps.update()

    zt = zf.T.contiguous()  # (T, N)
    del zf
    if coherence is None:
        coherence, neighbours = neighbour_coherence(zt, ijk, shape)
    # Salience: what fraction of a voxel's variance is *shared* -- with its
    # correlated partners anywhere (communality) and with its neighbours
    # (coherence). Task cortex and networks score high; white matter, whose
    # fluctuation is mostly independent thermal noise however high its tSNR,
    # scores low. Ranks, so neither measure's scale dominates.
    communality = (loadings_unit[:, :k_active].square().sum(1)).clamp(0, 1)
    salience = (_rank_unit(communality) * _rank_unit(coherence)).sqrt()
    steps.update()

    phase, phase_power, wave_tcs = wave_phase(zt, tr, period_s)
    bands = band_powers(zt, tr)
    steps.update()

    g = (g - g.mean()) / g.std().clamp_min(1e-12)
    med = dv.median()
    mad = (dv - med).abs().median().clamp_min(1e-6) * 1.4826
    dv = (dv - med) / mad
    steps.update()
    steps.close()

    feats = Features(
        tr=tr,
        run_starts=list(run_starts),
        ijk=ijk,
        anat=anat,
        func=func,
        z=zt.half(),
        loadings_all=loadings_unit,
        tcs_all=tcs_unit,
        pc_var_all=pc_var,
        k_active=k_active,
        k_suggested=k_suggested,
        **fams,
        mean=_robust_unit(raw_mean),
        tsnr=_robust_unit(tsnr),
        coherence=coherence,
        communality=communality,
        salience=salience,
        phase=phase,
        phase_power=torch.stack([_robust_unit(p) for p in phase_power.T], 1),
        global_signal=g,
        dvars=dv,
        extras={
            "wave_tcs": wave_tcs,
            "neighbours": neighbours,
            # Percent signal change is z * sd / mean * 100; keep both so the
            # inspector can show either unit.
            "raw_mean": raw_mean,
            "sd": sd,
            "shape": shape,
            "affine": np.asarray(affine, dtype=np.float64),
            "bands": bands,
        },
    )
    feats.flag_families()
    return feats


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
