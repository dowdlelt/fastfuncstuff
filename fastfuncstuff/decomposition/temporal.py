"""Temporal ICA via spatial-ICA dimensionality reduction.

fMRI has V≈10^5-10^6 voxels but only T timepoints, so ICA over the raw voxel
axis to find *temporally* independent sources is hopelessly ill-conditioned.
The established fix (Glasser et al. 2018, the HCP tICA pipeline) is a two-stage
decomposition:

  1. Reduce the spatial dimension with a group **spatial** ICA (or PCA) →
     ``K_sica`` spatial components shared across all runs/subjects.
  2. Back-project (dual-regression stage 1) the group maps onto each run to
     recover per-run component timecourses at the native time resolution.
  3. Concatenate those timecourses along time across all runs → (T_total,
     K_sica) with T_total ≫ K_sica, and run **temporal** ICA on it.

The key enabling fact is that :class:`~fastfuncstuff.decomposition.ica.FastICA`
seeks independence over the *feature* (column) axis: spatial ICA feeds ``(T, V)``
and gets spatial maps; temporal ICA is the transpose — feed ``(K_sica, T_total)``
and independence is maximised over time, yielding temporal sources
``components_ = (K_tica, T_total)`` and a component-space mixing
``mixing_ = (K_sica, K_tica)``. Projecting that mixing through the group spatial
maps gives the tICA spatial maps.

See ``../fmri_wiki/concepts/Temporal ICA.md`` for the rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from tqdm.auto import tqdm

from fastfuncstuff.decomposition.ica import FastICA, create_ica
from fastfuncstuff.decomposition.icasso import (
    cluster_components,
    compute_cluster_quality,
    compute_similarity_matrix,
    select_centrotypes,
)
from fastfuncstuff.utils import to_linalg_f64, to_tensor


@dataclass
class TemporalICAResult:
    """Outputs of a two-stage temporal ICA.

    Shapes use K_sica = number of stage-1 spatial components, K_tica = number
    of temporal components, V = voxels, T_total = concatenated timepoints.
    """

    temporal_sources: np.ndarray  # (K_tica, T_total) — temporally independent sources
    spatial_maps: np.ndarray  # (K_tica, V) — tICA spatial maps (mixing.T @ group_maps)
    mixing: np.ndarray  # (K_sica, K_tica) — sICA-space -> tICA-source mixing
    group_spatial_maps: np.ndarray  # (K_sica, V) — the stage-1 reduction basis
    per_run_sources: list[np.ndarray]  # each (K_tica, T_i) — sources split per run
    run_lengths: list[int]
    explained_share: np.ndarray  # (K_tica,) — per-source variance share (sorted desc)
    n_iter: int
    method: str
    reducer: str
    variance_normalized: bool
    stability: np.ndarray | None = None  # (K_tica,) ICASSO Iq per source, or None
    diagnostics: dict = field(default_factory=dict)

    def subset(self, keep: np.ndarray) -> TemporalICAResult:
        """Return a copy keeping only the components selected by boolean `keep`.

        Used to drop temporal components that failed the ICASSO reproducibility
        threshold (Iq) when `K_tica` is chosen automatically.
        """
        keep = np.asarray(keep, dtype=bool)
        diag = dict(self.diagnostics)
        diag["k_tica"] = int(keep.sum())
        return TemporalICAResult(
            temporal_sources=self.temporal_sources[keep],
            spatial_maps=self.spatial_maps[keep],
            mixing=self.mixing[:, keep],
            group_spatial_maps=self.group_spatial_maps,
            per_run_sources=[b[keep] for b in self.per_run_sources],
            run_lengths=self.run_lengths,
            explained_share=self.explained_share[keep],
            n_iter=self.n_iter,
            method=self.method,
            reducer=self.reducer,
            variance_normalized=self.variance_normalized,
            stability=None if self.stability is None else self.stability[keep],
            diagnostics=diag,
        )


@torch.inference_mode()
def spatial_regression(
    group_maps: torch.Tensor,
    run_data: torch.Tensor,
    ridge: float = 1e-6,
) -> torch.Tensor:
    """Dual-regression stage 1: back-project group maps onto one run.

    Solves, per timepoint ``t``, the OLS problem ``d_t ≈ group_maps.T @ tc_t``
    for the component loadings ``tc_t``. This is the spatial-regression half of
    FSL's dual regression (``fsl_glm -i data -d maps --demean``): it recovers
    each run's component timecourses at native time resolution, which group
    reduction (MIGP) otherwise destroys.

    Parameters
    ----------
    group_maps : (K, V) Tensor
        Group spatial components (the stage-1 reduction basis).
    run_data : (V, T) Tensor
        One run's voxels-by-time data on the same device/grid as ``group_maps``.
    ridge : float
        Tiny Tikhonov term added to the KxK normal matrix for stability when
        the maps are near-collinear. Negligible at K≈100.

    Returns
    -------
    timecourses : (T, K) Tensor
        Per-run component timecourses.
    """
    if group_maps.shape[1] != run_data.shape[0]:
        raise ValueError(
            f"spatial_regression: voxel dim mismatch — group_maps {tuple(group_maps.shape)} "
            f"vs run_data {tuple(run_data.shape)}"
        )
    device = group_maps.device
    run_data = run_data.to(device)
    # Demean each voxel over time (FSL dual_regression --demean). Preprocessing
    # usually already does this, but make the regression self-contained.
    run_data = run_data - run_data.mean(dim=1, keepdim=True)
    # Normal equations on the KxK gram (K is small, so this is exact and cheap
    # and avoids the lstsq driver differences across CPU/CUDA/MPS).
    gram = group_maps @ group_maps.T  # (K, K)
    k = gram.shape[0]
    gram = gram + ridge * torch.eye(k, device=device, dtype=gram.dtype)
    rhs = group_maps @ run_data  # (K, T)
    tc = torch.linalg.solve(gram, rhs)  # (K, T)
    return tc.T.contiguous()  # (T, K)


@torch.inference_mode()
def group_spatial_ica(
    data_tv: torch.Tensor,
    n_components: int,
    method: str = "fastica",
    max_iter: int = 500,
    tol: float = 5e-5,
    fun: str = "pow3",
    seed: int | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, int]:
    """Stage-1 spatial ICA over the concatenated group data.

    Thin wrapper over :func:`create_ica` that returns just the group spatial
    maps (independence sought over the voxel axis).

    Parameters
    ----------
    data_tv : (T_total, V) Tensor
        Temporally concatenated (optionally MIGP-reduced) group data.
    n_components : int
        Number of spatial components ``K_sica`` to keep.

    Returns
    -------
    group_maps : (K_sica, V) Tensor
    n_iter : int
    """
    device = device if device is not None else data_tv.device
    ica = create_ica(
        method=method,
        n_components=n_components,
        pca_components=n_components,
        max_iter=max_iter,
        tol=tol,
        fun=fun,
        random_state=seed,
        device=device,
    )
    ica.fit(data_tv)
    assert ica.components_ is not None  # set by fit()
    return ica.components_.to(device), int(ica.n_iter_)


def _solve_temporal(
    whitened: torch.Tensor,
    n_components: int,
    method: str,
    fun: str,
    max_iter: int,
    tol: float,
    seed: int | None,
    icasso_runs: int,
    device: torch.device,
    verbose: bool = False,
) -> tuple[torch.Tensor, np.ndarray | None, int]:
    """Run the ICA solver on correctly-whitened temporal data.

    ``whitened`` is (k, T_total) with unit-variance rows. Independence is sought
    over the T_total columns, so ``sources = W @ whitened`` are the temporally
    independent components.

    With ``icasso_runs <= 1`` this is a single FastICA/InfoMax solve. With
    ``icasso_runs > 1`` it runs that many random-init solves and clusters the
    source estimates ([[ICASSO]]), returning the centrotype of each cluster and
    its stability index Iq — the empirical referee for how many temporal
    components are reproducible. We cannot reuse ``icasso.icasso`` directly
    because it calls ``FastICA.fit`` (spatial-orientation centering); here the
    data is already correctly whitened, so we drive the core solver.

    Returns ``(sources (k, T_total), iq (k,) or None, n_iter)``.
    """
    ica = create_ica(
        method=method,
        n_components=n_components,
        pca_components=n_components,
        max_iter=max_iter,
        tol=tol,
        fun=fun,
        random_state=seed,
        device=device,
    )
    ica.verbose = False

    def _one_solve(seed_i: int | None) -> tuple[torch.Tensor, int]:
        ica.random_state = seed_i
        if isinstance(ica, FastICA):
            unmix, n_iter = ica._fastica(whitened, n_components)
        else:
            _res = ica._infomax(whitened, n_components)
            unmix, n_iter = _res[0], _res[1]
        return unmix @ whitened, int(n_iter)  # (k, T_total)

    if icasso_runs <= 1:
        sources, n_iter = _one_solve(seed)
        return sources, None, n_iter

    # ICASSO: repeat, cluster the source estimates, take centrotypes + Iq.
    base = seed if seed is not None else 0
    estimates: list[np.ndarray] = []
    it = range(icasso_runs)
    if verbose:
        it = tqdm(it, desc="tICA ICASSO", leave=True)
    for i in it:
        s_i, _ = _one_solve(base + i)
        estimates.append(s_i.detach().cpu().numpy().astype(np.float32))
    similarity = compute_similarity_matrix(estimates)
    labels, _ = cluster_components(similarity, n_clusters=n_components, method="average")
    iq = compute_cluster_quality(labels, similarity)["iq"]
    centro_idx = select_centrotypes(labels, similarity)
    all_est = np.concatenate(estimates, axis=0)  # (icasso_runs * k, T_total)
    sources = torch.as_tensor(all_est[centro_idx], device=device)  # (k, T_total)
    return sources, np.asarray(iq, dtype=np.float32), int(icasso_runs)


@torch.inference_mode()
def temporal_ica(
    sica_timecourses: torch.Tensor,
    group_maps: torch.Tensor,
    n_components: int,
    run_lengths: list[int],
    method: str = "fastica",
    max_iter: int = 500,
    tol: float = 5e-5,
    fun: str = "logcosh",
    seed: int | None = None,
    variance_normalize: bool = True,
    icasso_runs: int = 0,
    device: torch.device | None = None,
    verbose: bool = False,
) -> TemporalICAResult:
    """Stage-2 temporal ICA on concatenated sICA timecourses.

    Parameters
    ----------
    sica_timecourses : (T_total, K_sica) Tensor
        Per-run stage-1 component timecourses concatenated along time, in the
        run order implied by ``run_lengths``.
    group_maps : (K_sica, V) Tensor
        The stage-1 spatial reduction basis (used to build tICA spatial maps).
    n_components : int
        Number of temporal components ``K_tica`` to extract.
    run_lengths : list[int]
        Length of each run's block along the concatenated time axis; must sum
        to ``T_total``. Used to split the recovered sources back per run.
    fun : str
        ICA nonlinearity. Defaults to ``"logcosh"`` (a general contrast): unlike
        the spatial default ``"pow3"`` (a *skewness* contrast, MELODIC's), it can
        separate symmetric non-Gaussian temporal sources, which are common.
    variance_normalize : bool
        If True (default), apply HCP's per-run variance normalization: scale each
        run's block of each sICA timecourse to unit variance, then restore the
        component's global std. Makes every run contribute equally (for a single
        subject, equalizes its runs) without flattening between-component
        amplitude. Matches ``ComputeGroupTICA.m``.
    icasso_runs : int
        If > 1, stabilize the decomposition with that many random-init ICASSO
        repetitions and populate ``result.stability`` with per-source Iq. HCP
        uses 100; 25 is a good cheap default for this small matrix. If <= 1, a
        single FastICA solve (no stability estimate).

    Returns
    -------
    TemporalICAResult
    """
    device = device if device is not None else group_maps.device
    tcs = to_tensor(sica_timecourses, device=device)  # (T_total, K_sica)
    group_maps = to_tensor(group_maps, device=device)  # (K_sica, V)
    t_total, k_sica = tcs.shape
    if sum(run_lengths) != t_total:
        raise ValueError(
            f"temporal_ica: run_lengths sum to {sum(run_lengths)} but "
            f"sica_timecourses has {t_total} timepoints"
        )
    if group_maps.shape[0] != k_sica:
        raise ValueError(
            f"temporal_ica: group_maps has {group_maps.shape[0]} components but "
            f"sica_timecourses has {k_sica}"
        )
    n_components = min(int(n_components), k_sica)

    # Orient as channels(K_sica) × time and center each channel over time. This
    # is the correct temporal-ICA centering (remove each component's temporal
    # mean). We deliberately do NOT call FastICA.fit here: its MELODIC-style
    # whitening centers per-feature-over-samples, which in this orientation
    # subtracts a per-timepoint mean *across components* — destroying a temporal
    # signal direction. Instead we whiten explicitly (per-channel centering) and
    # reuse only the core solver.
    x = tcs.T.contiguous()  # (K_sica, T_total)
    x = x - x.mean(dim=1, keepdim=True)
    if variance_normalize:
        # Per-run variance normalization (HCP ComputeGroupTICA.m:87): divide each
        # run's block by its own per-component std, then restore the global
        # per-component std. This makes every run contribute equally — for a
        # single subject that means equalizing its runs — while preserving the
        # relative amplitude *between* components. A plain global z-score would
        # instead let long/high-variance runs dominate the decomposition.
        global_std = x.std(dim=1, keepdim=True)  # (K_sica, 1)
        off = 0
        for length in run_lengths:
            sl = slice(off, off + length)
            block_std = torch.clamp(x[:, sl].std(dim=1, keepdim=True), min=1e-8)
            x[:, sl] = x[:, sl] / block_std * global_std
            off += length

    # PCA whitening of the channel covariance; keep the top n_components dims.
    cov = (x @ x.T) / float(t_total)  # (K_sica, K_sica)
    evals, evecs = torch.linalg.eigh(to_linalg_f64(cov))
    evals = evals.flip(0).to(x.dtype)
    evecs = evecs.flip(1).to(x.dtype)
    evals = torch.clamp(evals[:n_components], min=1e-12)
    evecs = evecs[:, :n_components]  # (K_sica, k)
    sqrt_ev = torch.sqrt(evals)
    white = (evecs / sqrt_ev.unsqueeze(0)).T  # (k, K_sica)
    whitened = white @ x  # (k, T_total) — unit-variance rows

    # Solve (single FastICA or ICASSO-stabilized), then recover the channel-space
    # mixing by OLS: x ≈ mixing @ sources ⇒ mixing = x @ pinv(sources). This is
    # uniform across the single/ICASSO paths (ICASSO centrotypes have no unmixing
    # matrix of their own) and captures the full projection when k < K_sica.
    sources, stability, n_iter = _solve_temporal(
        whitened, n_components, method, fun, max_iter, tol, seed, icasso_runs, device, verbose
    )
    mixing = x @ torch.linalg.pinv(sources)  # (K_sica, K_tica)

    # tICA spatial maps: each temporal source's spatial footprint is the
    # mixing-weighted sum of the group sICA maps.
    spatial_maps = mixing.T @ group_maps  # (K_tica, V)

    # Sign convention (FSL): make each spatial map's largest-magnitude voxel
    # positive, applying the same flip to the source and the mixing column.
    max_abs = spatial_maps.abs().max(dim=1).values
    max_pos = spatial_maps.max(dim=1).values
    flip = (max_abs > max_pos).to(spatial_maps.dtype) * -2.0 + 1.0  # ±1
    spatial_maps = spatial_maps * flip.unsqueeze(1)
    sources = sources * flip.unsqueeze(1)
    mixing = mixing * flip.unsqueeze(0)

    # Order components by temporal-source variance share (largest first).
    src_var = sources.var(dim=1, unbiased=False)
    explained = (src_var / torch.clamp(src_var.sum(), min=1e-15)).cpu().numpy()
    order = np.argsort(-explained)
    explained = explained[order].astype(np.float32)
    order_t = torch.as_tensor(order, device=device)
    sources = sources.index_select(0, order_t)
    spatial_maps = spatial_maps.index_select(0, order_t)
    mixing = mixing.index_select(1, order_t)
    if stability is not None:
        stability = stability[order].astype(np.float32)

    sources_np = sources.detach().cpu().numpy().astype(np.float32)

    # Split sources back into per-run blocks along the time axis.
    per_run: list[np.ndarray] = []
    off = 0
    for length in run_lengths:
        per_run.append(sources_np[:, off : off + length].copy())
        off += length

    return TemporalICAResult(
        temporal_sources=sources_np,
        spatial_maps=spatial_maps.detach().cpu().numpy().astype(np.float32),
        mixing=mixing.detach().cpu().numpy().astype(np.float32),
        group_spatial_maps=group_maps.detach().cpu().numpy().astype(np.float32),
        per_run_sources=per_run,
        run_lengths=list(run_lengths),
        explained_share=explained,
        n_iter=int(n_iter),
        method=method,
        reducer="",  # filled in by the caller if desired
        variance_normalized=bool(variance_normalize),
        stability=stability,
        diagnostics={
            "k_sica": int(k_sica),
            "k_tica": int(n_components),
            "icasso_runs": int(icasso_runs),
            "n_stable_iq0.5": (int((stability > 0.5).sum()) if stability is not None else None),
        },
    )
