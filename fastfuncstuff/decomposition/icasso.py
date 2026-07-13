"""
ICASSO: ICA with component clustering for stability-based selection.

Implements the ICASSO algorithm (Himberg et al., 2004) following the
GIFT/icasso122 reference implementation.

Algorithm:
1. Run ICA many times with different random initializations (and/or
   bootstrap resampling of the data).
2. Compute pairwise cosine similarity between all demixing vectors
   projected through the dewhitening matrix, matching GIFT's ``corrw``.
3. Agglomerative hierarchical clustering (average linkage by default)
   on the dissimilarity matrix (1 - |cosine similarity|).
4. For each cluster, compute stability index Iq = mean(intra-cluster
   similarity) - mean(extra-cluster similarity).
5. Select the centrotype (real ICA estimate most similar to all others
   in its cluster) as the representative component.

References:
    Himberg, J., Hyvärinen, A., & Esposito, F. (2004). Validating the
    independent components of neuroimaging time series via clustering and
    visualization. NeuroImage, 22(3), 1214-1222.

    Himberg, J. & Hyvärinen, A. (2003). ICASSO: software for investigating
    the reliability of ICA estimates by clustering and visualization.
    IEEE Workshop on Neural Networks for Signal Processing.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from tqdm.auto import tqdm

from fastfuncstuff.memory import get_available_memory
from fastfuncstuff.utils import get_device, to_tensor

from .ica import create_ica

# ---------------------------------------------------------------------------
# Similarity matrix — GIFT corrw (default)
# ---------------------------------------------------------------------------


def _recover_W_and_D(
    ica,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover unmixing W and dewhitening D from a fitted ICA object.

    W = components_ @ pinv(pca_components[:n_ica])   — unmixing in whitened space
    D = pca_components[:n_ica]                        — dewhitening (whitened → data)

    This avoids storing extra attributes on the ICA classes.
    """
    comp = ica.components_  # (k, V)
    pca_comp = ica.pca_.components_[: comp.shape[0]]  # (k, V)
    W = comp @ torch.linalg.pinv(pca_comp)  # (k, k)
    return W, pca_comp


@torch.inference_mode()
def compute_similarity_corrw(
    wd_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device | None = None,
) -> np.ndarray:
    """GIFT-style cosine similarity via ``corrw(W, D)``.

    For each run i, ``B_i = rownorm(W_i @ D_i)`` where W_i is the
    (k, k) unmixing matrix in whitened space and D_i is the (k, V)
    dewhitening matrix.  All B_i are stacked into a (N, V) matrix and
    the similarity is ``R = |B @ B'|``, clipped to [0, 1].

    Matches ``icassoCluster.m:215`` → ``abs(corrw(W, dewhitemat))``.

    Parameters
    ----------
    wd_pairs : list of (W, D) tuples
        W: (k, k) unmixing, D: (k, V) dewhitening, per run.
    device : torch.device, optional
        Device for the matmul. Falls back to CPU to avoid competing with
        the ICA loop for GPU memory.

    Returns
    -------
    similarity : (N, N) float32 numpy array
    """
    if device is None:
        device = torch.device("cpu")

    rows = []
    for W, D in wd_pairs:
        W = W.to(device=device, dtype=torch.float32)
        D = D.to(device=device, dtype=torch.float32)
        B = W @ D  # (k, V)
        norms = B.norm(dim=1, keepdim=True).clamp(min=1e-12)
        B = B / norms  # row-normalise
        rows.append(B)

    all_B = torch.cat(rows, dim=0)  # (N, V)
    N = all_B.shape[0]
    del rows

    avail = get_available_memory(device, safety_factor=0.4)
    chunk_bytes_full = N * all_B.shape[1] * 4  # (N_chunk, V) @ (V, N)
    max_chunk_rows = max(1, int(avail // chunk_bytes_full)) if chunk_bytes_full > 0 else N
    max_chunk_rows = min(max_chunk_rows, N)

    if max_chunk_rows >= N:
        S = torch.abs(all_B @ all_B.T)
    else:
        S = torch.empty((N, N), dtype=torch.float32, device=device)
        for i in range(0, N, max_chunk_rows):
            j = min(i + max_chunk_rows, N)
            S[i:j, :] = torch.abs(all_B[i:j] @ all_B.T)

    S.clamp_(0.0, 1.0)
    return S.cpu().numpy()


# ---------------------------------------------------------------------------
# Similarity matrix — spatial-map Pearson correlation (fallback)
# ---------------------------------------------------------------------------


def compute_similarity_matrix(
    components_list: list[np.ndarray],
    batch_size: int | None = None,
) -> np.ndarray:
    """Pairwise absolute Pearson correlation of spatial maps.

    Uses standardized (zero-mean, unit-norm) row vectors so that the
    dot product equals the Pearson correlation.  Takes absolute value
    to handle ICA sign ambiguity (GIFT convention).

    .. note::
        For GIFT parity prefer :func:`compute_similarity_corrw` which
        uses demixing-space cosine similarity matching ``corrw.m``.
        This function is retained as a fallback.

    Parameters
    ----------
    components_list : list of (n_components, n_features) arrays
    batch_size : int, optional

    Returns
    -------
    similarity : (N, N) float32 array
    """
    all_components = np.concatenate(components_list, axis=0)
    n_total = all_components.shape[0]

    row_means = all_components.mean(axis=1, keepdims=True)
    centered = all_components - row_means
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    standardized = centered / norms

    if batch_size is None or n_total <= batch_size:
        similarity = np.abs(standardized @ standardized.T)
    else:
        similarity = np.empty((n_total, n_total), dtype=np.float32)
        for i in range(0, n_total, batch_size):
            end_i = min(i + batch_size, n_total)
            similarity[i:end_i, :] = np.abs(standardized[i:end_i] @ standardized.T)

    np.clip(similarity, 0.0, 1.0, out=similarity)
    return similarity


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def cluster_components(
    similarity: np.ndarray,
    n_clusters: int,
    method: str = "average",
) -> tuple[np.ndarray, np.ndarray]:
    """Agglomerative hierarchical clustering on dissimilarity = 1 - |r|.

    Parameters
    ----------
    similarity : (N, N) absolute-correlation similarity matrix.
    n_clusters : int
        Number of clusters to cut.
    method : str
        Linkage method ('average', 'single', 'complete').

    Returns
    -------
    labels : (N,) cluster labels (1-based, matching scipy convention).
    linkage_matrix : scipy linkage matrix for dendrogram plotting.
    """
    distance = 1.0 - similarity
    distance = np.maximum(distance, 0.0)
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)

    condensed = squareform(distance, checks=False)
    Z = linkage(condensed, method=method)
    labels = fcluster(Z, n_clusters, criterion="maxclust")
    return labels, Z


# ---------------------------------------------------------------------------
# Quality index  (Iq)
# ---------------------------------------------------------------------------


def compute_cluster_quality(
    labels: np.ndarray,
    similarity: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute ICASSO stability index Iq for each cluster.

    Iq = mean(intra-cluster similarity) - mean(extra-cluster similarity)

    Matches the GIFT/Himberg ``clusterquality`` with ``'mean'`` scoring.

    Parameters
    ----------
    labels : (N,) cluster labels (1-based).
    similarity : (N, N) similarity matrix.

    Returns
    -------
    dict with:
        iq : (n_clusters,) stability index per cluster.
        intra : (n_clusters,) mean intra-cluster similarity.
        extra : (n_clusters,) mean extra-cluster similarity.
        size : (n_clusters,) int, cluster size.
    """
    cluster_ids = np.unique(labels)
    n_clusters = len(cluster_ids)
    iq = np.full(n_clusters, np.nan)
    intra = np.full(n_clusters, np.nan)
    extra = np.full(n_clusters, np.nan)
    size = np.zeros(n_clusters, dtype=int)

    for idx, cid in enumerate(cluster_ids):
        mask = labels == cid
        n_in = mask.sum()
        size[idx] = n_in

        if n_in < 2:
            # Singleton — Iq undefined (GIFT returns NaN)
            intra[idx] = 1.0
            extra[idx] = 0.0
            iq[idx] = np.nan
            continue

        # Intra-cluster: mean of off-diagonal similarities within cluster
        sim_in = similarity[np.ix_(mask, mask)]
        intra_vals = sim_in[~np.eye(n_in, dtype=bool)]
        intra[idx] = intra_vals.mean()

        # Extra-cluster: mean similarity to non-members
        sim_out = similarity[np.ix_(mask, ~mask)]
        extra[idx] = sim_out.mean() if sim_out.size > 0 else 0.0

        iq[idx] = intra[idx] - extra[idx]

    return {
        "iq": iq,
        "intra": intra,
        "extra": extra,
        "size": size,
    }


# ---------------------------------------------------------------------------
# R-index  (Davies-Bouldin validity, GIFT rindex.m)
# ---------------------------------------------------------------------------


def _clusterstat(
    mat: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Compute intra/inter/between cluster statistics on a square matrix.

    Port of GIFT ``clusterstat.m``.  Works with either similarity or
    distance matrices — the caller decides which to pass.

    Parameters
    ----------
    mat : (N, N) symmetric matrix (similarity or distance).
    labels : (N,) 1-based cluster labels.

    Returns
    -------
    dict with keys:
        N : (K,) cluster sizes.
        internal.avg : (K,) mean off-diagonal intra-cluster value.
        external.avg : (K,) mean extra-cluster value.
        between.avg : (K, K) mean between-cluster value (diagonal = Inf).
    """
    cluster_ids = np.unique(labels)
    K = len(cluster_ids)

    internal_avg = np.full(K, np.nan)
    external_avg = np.full(K, np.nan)
    between_avg = np.full((K, K), np.nan)
    sizes = np.zeros(K, dtype=int)

    for i, ci in enumerate(cluster_ids):
        mi = labels == ci
        ni = mi.sum()
        sizes[i] = ni

        # Intra: off-diagonal entries of sub-matrix
        S_in = mat[np.ix_(mi, mi)]
        if ni > 1:
            mask_offdiag = ~np.eye(ni, dtype=bool)
            internal_avg[i] = S_in[mask_offdiag].mean()

        # External: members of ci vs non-members
        if K > 1:
            S_out = mat[np.ix_(mi, ~mi)]
            external_avg[i] = S_out.mean() if S_out.size > 0 else np.nan

    # Between-cluster averages (upper triangle, then symmetrise)
    between_avg[:] = 0.0
    for i in range(K):
        mi = labels == cluster_ids[i]
        for j in range(i + 1, K):
            mj = labels == cluster_ids[j]
            vals = mat[np.ix_(mi, mj)]
            between_avg[i, j] = vals.mean() if vals.size > 0 else 0.0
    between_avg = between_avg + between_avg.T
    np.fill_diagonal(between_avg, np.inf)

    return {
        "N": sizes,
        "internal": {"avg": internal_avg},
        "external": {"avg": external_avg},
        "between": {"avg": between_avg},
    }


def compute_rindex(
    distance: np.ndarray,
    linkage_matrix: np.ndarray,
    max_clusters: int | None = None,
) -> np.ndarray:
    """Davies-Bouldin R-index for cluster count selection.

    For each candidate cluster count k, the R-index is:

        R(k) = mean_i( avg_intra_dist(i) / min_j!=i( avg_inter_dist(i,j) ) )

    Lower values indicate better clustering (compact, well-separated).
    Matches GIFT ``rindex.m`` / ``icassoCluster.m:236``.

    Parameters
    ----------
    distance : (N, N) dissimilarity matrix (e.g. ``1 - similarity``).
    linkage_matrix : scipy linkage matrix from hierarchical clustering.
    max_clusters : int, optional
        Maximum number of clusters to evaluate.  Default: N (all).

    Returns
    -------
    rindex : (max_clusters,) array, 1-based indexing.
        ``rindex[0]`` corresponds to k=1 (always NaN).
        ``rindex[k-1]`` is the R-index for k clusters (NaN if undefined).
    """
    N = distance.shape[0]
    if max_clusters is None:
        max_clusters = N
    max_clusters = min(max_clusters, N)

    ri = np.full(max_clusters, np.nan)

    for k in range(2, max_clusters + 1):
        labels = fcluster(linkage_matrix, k, criterion="maxclust")

        # Skip if any cluster has only 1 member (R-index undefined)
        _, counts = np.unique(labels, return_counts=True)
        if (counts == 1).any() or k == 1:
            continue

        stat = _clusterstat(distance, labels)
        # R-index = mean( internal.avg / min(between.avg, axis=1) )
        min_between = stat["between"]["avg"].min(axis=1)
        ratios = stat["internal"]["avg"] / min_between
        ri[k - 1] = ratios.mean()

    return ri


# ---------------------------------------------------------------------------
# Centrotype selection  (GIFT convention)
# ---------------------------------------------------------------------------


def select_centrotypes(
    labels: np.ndarray,
    similarity: np.ndarray,
) -> np.ndarray:
    """Select the centrotype index for each cluster.

    The centrotype is the *real* ICA estimate whose sum of similarities
    to all other cluster members is maximal — i.e. the point most
    representative of the cluster.  Unlike averaging, this guarantees
    that the result is an actual ICA estimate.

    Parameters
    ----------
    labels : (N,) cluster labels (1-based).
    similarity : (N, N) similarity matrix.

    Returns
    -------
    centrotype_indices : (n_clusters,) global indices into the
        concatenated component array.
    """
    cluster_ids = np.unique(labels)
    centrotype_indices = np.empty(len(cluster_ids), dtype=int)

    for idx, cid in enumerate(cluster_ids):
        members = np.where(labels == cid)[0]
        if len(members) == 1:
            centrotype_indices[idx] = members[0]
        else:
            sub_sim = similarity[np.ix_(members, members)]
            # Centrotype = argmax of column/row sum (matrix is symmetric)
            best_local = int(sub_sim.sum(axis=1).argmax())
            centrotype_indices[idx] = members[best_local]

    return centrotype_indices


# ---------------------------------------------------------------------------
# Mixing matrix extraction
# ---------------------------------------------------------------------------


def _extract_mixing_for_centrotypes(
    centrotype_indices: np.ndarray,
    n_components_per_run: int,
    mixing_list: list[np.ndarray],
    components_list: list[np.ndarray],
    all_components: np.ndarray,
    X: np.ndarray | torch.Tensor | None = None,
    reproject: bool = False,
) -> np.ndarray:
    """Build a mixing matrix for the centrotype components.

    In randinit mode (``reproject=False``), each centrotype's mixing
    column is taken from the ICA run that produced it.  This is valid
    because all runs operate on the same data.

    In bootstrap/both mode (``reproject=True``), each run saw different
    (resampled) data so its mixing column is wrong for the original
    data.  Instead we least-squares project the original data X onto
    the centrotype spatial maps S:

        A = X @ S^T @ (S @ S^T)^{-1}

    Parameters
    ----------
    centrotype_indices : (n_clusters,) global indices.
    n_components_per_run : int
    mixing_list : list of (T, n_comp) arrays.
    components_list : list of (n_comp, V) arrays.
    all_components : (N, V) concatenated component array.
    X : (T, V) original data, required when ``reproject=True``.
    reproject : bool
        If True, reproject X onto centrotype components instead of
        pulling mixing columns from individual runs.

    Returns
    -------
    mixing : (T, n_clusters) mixing matrix.
    """
    n_clusters = len(centrotype_indices)

    if reproject:
        if X is None:
            raise ValueError("X required when reproject=True")
        S = all_components[centrotype_indices]  # (k, V)
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = np.asarray(X)
        StS = S @ S.T  # (k, k)
        mixing = X_np @ S.T @ np.linalg.inv(StS)  # (T, k)
        return mixing.astype(np.float64)

    n_samples = mixing_list[0].shape[0]
    mixing = np.empty((n_samples, n_clusters), dtype=np.float64)

    for ci, gidx in enumerate(centrotype_indices):
        run_idx = gidx // n_components_per_run
        comp_idx = gidx % n_components_per_run
        mixing[:, ci] = mixing_list[run_idx][:, comp_idx]

    return mixing


# ---------------------------------------------------------------------------
# Main ICASSO function
# ---------------------------------------------------------------------------


def icasso(
    X: np.ndarray | torch.Tensor,
    n_components: int,
    n_runs: int = 100,
    pca_components: int | float | str | None = 0.85,
    min_stability: float = 0.7,
    device: torch.device | None = None,
    verbose: bool = True,
    batch_size: int | None = None,
    ica_method: str = "fastica",
    base_seed: int = 0,
    mode: str = "randinit",
    linkage_method: str = "average",
    similarity_method: str = "corrw",
) -> dict:
    """Run ICASSO: ICA with component clustering for reliability.

    Runs ICA ``n_runs`` times, clusters all component estimates, and
    returns centrotype representatives ranked by stability (Iq).

    Parameters
    ----------
    X : (n_samples, n_features) data matrix.
    n_components : int
        Number of ICA components per run.
    n_runs : int
        Number of ICA repetitions.
    pca_components : int, float, str, or None
        PCA dimensionality reduction passed to ICA.
    min_stability : float
        Minimum Iq for a component to be considered "stable".
    device : torch.device, optional
    verbose : bool
    batch_size : int, optional
        Batch size for spatial-map similarity (only used when
        ``similarity_method='spatial'``).
    ica_method : str
        ICA algorithm ('fastica' or 'infomax').
    base_seed : int
        Base random seed; each run uses ``base_seed + i``.
    mode : str
        Resampling mode:
        - 'randinit': fixed data, random ICA seeds (default, fast).
        - 'bootstrap': bootstrap data columns, fixed ICA seed.
        - 'both': bootstrap data AND random ICA seeds.
    linkage_method : str
        Hierarchical clustering linkage ('average', 'single', 'complete').
    similarity_method : str
        'corrw' (default): GIFT-style cosine similarity of demixing
        vectors projected through dewhitening, matching ``corrw.m``.
        'spatial': absolute Pearson correlation of spatial maps (legacy).

    Returns
    -------
    dict with keys:

        Stable subset (filtered by min_stability):
            components, mixing, stability, n_stable

        Full decomposition (all n_components clusters):
            all_centroids, all_mixing, all_stability, all_iq

        Diagnostics:
            cluster_quality, cluster_labels, similarity, linkage_matrix,
            centrotype_indices, n_components, pca_eigenvalues,
            pca_components, pca_variance_explained, pca_variance_cumsum
    """
    device = device if device is not None else get_device()
    X = to_tensor(X, device=device)

    n_total = n_runs * n_components
    use_corrw = similarity_method == "corrw"

    # Auto batch size for spatial-map path
    if batch_size is None and not use_corrw:
        matrix_bytes = n_total**2 * 4
        if matrix_bytes > 1024**3:
            batch_size = max(100, int(np.sqrt(500 * 1024**3 / 4)))
            if verbose:
                print(
                    f"  Auto batch_size={batch_size} "
                    f"(similarity matrix ~{matrix_bytes / 1024**3:.1f} GB)"
                )

    if verbose:
        print(f"ICASSO: {n_runs} runs × {n_components} components = {n_total} estimates")
        print(
            f"  mode={mode}, linkage={linkage_method}, "
            f"ica={ica_method}, seed={base_seed}, "
            f"similarity={similarity_method}"
        )

    # ------------------------------------------------------------------
    # Step 1: Run ICA multiple times
    # ------------------------------------------------------------------
    components_list: list[np.ndarray] = []
    mixing_list: list[np.ndarray] = []
    wd_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    pca_eigenvalues = None
    pca_components_arr = None
    pca_variance_explained = None

    iterator = range(n_runs)
    if verbose:
        iterator = tqdm(iterator, desc="ICASSO ICA runs")

    for i in iterator:
        if mode == "randinit":
            seed_i = base_seed + i
            X_i = X
        elif mode == "bootstrap":
            seed_i = base_seed
            rng = np.random.RandomState(base_seed + i)
            n_cols = X.shape[1] if isinstance(X, torch.Tensor) else X.shape[1]
            boot_idx = rng.randint(0, n_cols, size=n_cols)
            if isinstance(X, torch.Tensor):
                X_i = X[:, boot_idx]
            else:
                X_i = X[:, boot_idx]
        else:  # 'both'
            seed_i = base_seed + i
            rng = np.random.RandomState(base_seed + n_runs + i)
            n_cols = X.shape[1] if isinstance(X, torch.Tensor) else X.shape[1]
            boot_idx = rng.randint(0, n_cols, size=n_cols)
            if isinstance(X, torch.Tensor):
                X_i = X[:, boot_idx]
            else:
                X_i = X[:, boot_idx]

        ica = create_ica(
            method=ica_method,
            n_components=n_components,
            pca_components=pca_components,
            random_state=seed_i,
            device=device,
        )
        ica.verbose = False
        ica.fit(X_i)

        if i == 0:
            evar = ica.pca_.explained_variance_.cpu().numpy()
            pca_variance_explained = evar / evar.sum()
            pca_eigenvalues = evar.copy()
            pca_components_arr = ica.pca_.components_.cpu().numpy()

        if use_corrw:
            W_i, D_i = _recover_W_and_D(ica)
            wd_pairs.append((W_i.cpu(), D_i.cpu()))

        components_list.append(ica.components_.cpu().numpy())
        mixing_list.append(ica.mixing_.cpu().numpy())

        if device.type == "cuda" and i % 10 == 9:
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 2: Similarity matrix
    # ------------------------------------------------------------------
    if verbose:
        print(f"Computing {n_total}×{n_total} similarity matrix (method={similarity_method}) ...")

    if use_corrw:
        similarity = compute_similarity_corrw(wd_pairs)
        del wd_pairs
    else:
        similarity = compute_similarity_matrix(components_list, batch_size=batch_size)

    # ------------------------------------------------------------------
    # Step 3: Hierarchical clustering
    # ------------------------------------------------------------------
    if verbose:
        print(f"Clustering ({linkage_method} linkage) ...")

    labels, linkage_matrix = cluster_components(
        similarity,
        n_clusters=n_components,
        method=linkage_method,
    )

    # ------------------------------------------------------------------
    # Step 4: Cluster quality (Iq)
    # ------------------------------------------------------------------
    quality = compute_cluster_quality(labels, similarity)
    iq = quality["iq"]

    # ------------------------------------------------------------------
    # Step 4b: R-index (Davies-Bouldin validity)
    # ------------------------------------------------------------------
    distance_matrix = 1.0 - similarity
    r_index = compute_rindex(distance_matrix, linkage_matrix, max_clusters=n_total)
    r_max = np.nanmax(r_index) if np.any(np.isfinite(r_index)) else np.nan
    if np.isfinite(r_max) and r_max > 0:
        r_index_normalized = r_index / r_max
    else:
        r_index_normalized = r_index.copy()

    # ------------------------------------------------------------------
    # Step 5: Centrotype selection
    # ------------------------------------------------------------------
    centrotype_indices = select_centrotypes(labels, similarity)
    all_components = np.concatenate(components_list, axis=0)
    centroids = all_components[centrotype_indices]

    # Build mixing matrix from centrotype runs.
    # Bootstrap mode: each run saw resampled data, so reproject original X.
    needs_reproject = mode in ("bootstrap", "both")
    all_mixing = _extract_mixing_for_centrotypes(
        centrotype_indices,
        n_components,
        mixing_list,
        components_list,
        all_components,
        X=X if needs_reproject else None,
        reproject=needs_reproject,
    )

    # ------------------------------------------------------------------
    # Step 6: Sort by Iq (most stable first)
    # ------------------------------------------------------------------
    # Replace NaN Iq (singletons) with -1 for sorting
    iq_sort = np.where(np.isnan(iq), -1.0, iq)
    sort_order = np.argsort(-iq_sort)

    centroids = centroids[sort_order]
    all_mixing = all_mixing[:, sort_order]
    iq = iq[sort_order]
    quality = {k: v[sort_order] for k, v in quality.items()}

    # Remap centrotype indices
    centrotype_indices = centrotype_indices[sort_order]

    # ------------------------------------------------------------------
    # Step 7: Stable subset
    # ------------------------------------------------------------------
    stable_mask = iq >= min_stability
    # NaN Iq (singletons) are never stable
    stable_mask = stable_mask & ~np.isnan(iq)
    n_stable = int(stable_mask.sum())

    if verbose:
        iq_valid = iq[~np.isnan(iq)]
        print(
            f"Stability (Iq): mean={np.mean(iq_valid):.3f}, "
            f"median={np.median(iq_valid):.3f}, "
            f"range=[{np.min(iq_valid):.3f}, {np.max(iq_valid):.3f}]"
        )
        print(f"Stable components (Iq >= {min_stability:.2f}): {n_stable}/{n_components}")

    return {
        # Stable subset
        "components": centroids[stable_mask],
        "mixing": all_mixing[:, stable_mask],
        "stability": iq[stable_mask],
        "n_stable": n_stable,
        # Full decomposition (all clusters, sorted by Iq)
        "all_centroids": centroids,
        "all_mixing": all_mixing,
        "all_stability": iq,
        "all_iq": iq,
        # Diagnostics
        "cluster_quality": quality,
        "cluster_labels": labels,
        "similarity": similarity,
        "linkage_matrix": linkage_matrix,
        "centrotype_indices": centrotype_indices,
        "r_index": r_index,
        "r_index_normalized": r_index_normalized,
        "n_components": n_components,
        "n_runs": n_runs,
        "mode": mode,
        # PCA info
        "pca_eigenvalues": pca_eigenvalues,
        "pca_components": pca_components_arr,
        "pca_variance_explained": pca_variance_explained,
        "pca_variance_cumsum": (
            pca_variance_explained.cumsum() if pca_variance_explained is not None else None
        ),
        # For compatibility
        "all_components": components_list,
        "all_mixing_list": mixing_list,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def icasso_plot(
    results: dict,
    output_path: str | None = None,
    dpi: int = 150,
    show: bool = False,
) -> None:
    """Generate ICASSO diagnostic plots.

    Produces a figure with four panels:
    1. Stability index (Iq) bar chart per cluster
    2. Dendrogram from hierarchical clustering
    3. Similarity matrix heatmap (sorted by cluster)
    4. Cluster size distribution

    Parameters
    ----------
    results : dict
        Output from :func:`icasso`.
    output_path : str, optional
        Save figure to this path (PNG/PDF/SVG).
    dpi : int
        Figure resolution.
    show : bool
        Call plt.show() (for interactive use).
    """
    import matplotlib

    if output_path and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    iq = results["all_iq"]
    quality = results["cluster_quality"]
    linkage_mat = results["linkage_matrix"]
    similarity = results["similarity"]
    labels = results["cluster_labels"]
    n_components = results["n_components"]
    n_runs = results["n_runs"]
    n_stable = results.get("n_stable", 0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"ICASSO: {n_runs} runs × {n_components} components ({results.get('mode', 'randinit')})",
        fontsize=13,
        fontweight="bold",
    )

    # --- Panel 1: Iq bar chart ---
    ax = axes[0, 0]
    n_c = len(iq)
    colors = []
    for val in iq:
        if np.isnan(val):
            colors.append("#cccccc")
        elif val >= 0.9:
            colors.append("#2ecc71")
        elif val >= 0.7:
            colors.append("#f39c12")
        elif val >= 0.5:
            colors.append("#e67e22")
        else:
            colors.append("#e74c3c")
    ax.bar(
        range(n_c), np.where(np.isnan(iq), 0, iq), color=colors, edgecolor="black", linewidth=0.3
    )
    ax.set_xlabel("Component (sorted by Iq)")
    ax.set_ylabel("Stability index (Iq)")
    ax.set_title("Cluster stability")
    ax.set_xlim(-0.5, n_c - 0.5)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.9, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axhline(0.7, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
    if n_stable > 0 and n_stable < n_c:
        ax.axvline(
            n_stable - 0.5,
            color="blue",
            linestyle="-",
            alpha=0.5,
            linewidth=1.2,
            label=f"stable cutoff ({n_stable})",
        )
        ax.legend(fontsize=8)

    # --- Panel 2: Dendrogram ---
    ax = axes[0, 1]
    dendrogram(
        linkage_mat,
        ax=ax,
        truncate_mode="lastp",
        p=min(n_components, 50),
        color_threshold=0,
        above_threshold_color="steelblue",
        no_labels=True,
    )
    ax.set_title("Hierarchical clustering dendrogram")
    ax.set_xlabel("Component estimates")
    ax.set_ylabel("Dissimilarity (1 - |r|)")

    # --- Panel 3: Similarity matrix (sorted by cluster) ---
    ax = axes[1, 0]
    # Sort by cluster label for visual grouping
    sort_idx = np.argsort(labels)
    sorted_sim = similarity[np.ix_(sort_idx, sort_idx)]
    im = ax.imshow(sorted_sim, cmap="hot", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_title("Similarity matrix (sorted by cluster)")
    ax.set_xlabel("Component estimate")
    ax.set_ylabel("Component estimate")
    fig.colorbar(im, ax=ax, shrink=0.8, label="|r|")

    # Draw cluster boundaries
    boundaries = []
    sorted_labels = labels[sort_idx]
    for i in range(1, len(sorted_labels)):
        if sorted_labels[i] != sorted_labels[i - 1]:
            boundaries.append(i - 0.5)
    for b in boundaries:
        ax.axhline(b, color="cyan", linewidth=0.5, alpha=0.7)
        ax.axvline(b, color="cyan", linewidth=0.5, alpha=0.7)

    # --- Panel 4: Cluster size + Iq scatter ---
    ax = axes[1, 1]
    sizes = quality["size"]
    iq_plot = np.where(np.isnan(iq), 0, iq)
    scatter = ax.scatter(
        sizes,
        iq_plot,
        c=iq_plot,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        edgecolors="black",
        linewidth=0.5,
        s=60,
    )
    ax.set_xlabel("Cluster size (n estimates)")
    ax.set_ylabel("Stability index (Iq)")
    ax.set_title("Size vs stability")
    ax.axhline(0.7, color="gray", linestyle=":", alpha=0.5)
    expected_size = n_runs
    ax.axvline(
        expected_size, color="blue", linestyle="--", alpha=0.3, label=f"expected={expected_size}"
    )
    ax.legend(fontsize=8)
    fig.colorbar(scatter, ax=ax, shrink=0.8, label="Iq")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        if not show:
            plt.close(fig)
    if show:
        plt.show()


# ---------------------------------------------------------------------------
# Auto component selection
# ---------------------------------------------------------------------------


def icasso_auto_select(
    X: np.ndarray | torch.Tensor,
    n_components_range: range | list[int],
    n_runs: int = 50,
    pca_components: int | float | str | None = 0.85,
    min_stability: float = 0.7,
    device: torch.device | None = None,
    verbose: bool = True,
    batch_size: int | None = None,
    ica_method: str = "fastica",
    base_seed: int = 0,
) -> dict:
    """Automatically select optimal number of ICA components using ICASSO.

    Runs ICASSO for different numbers of components and selects the
    number that produces the most stable decomposition.

    Parameters
    ----------
    X : (n_samples, n_features) data.
    n_components_range : iterable of int
        Component counts to test (e.g., range(15, 35, 5)).
    n_runs : int
        ICA runs per component count.
    pca_components, min_stability, device, verbose, batch_size
        Passed through to :func:`icasso`.
    ica_method : str
    base_seed : int

    Returns
    -------
    dict with optimal_n_components, optimal_results,
    n_stable_by_n_components, stability_ratios, all_results.
    """
    from .ica import FastICA

    all_results = {}
    n_stable_by_n = {}

    if verbose:
        print(
            f"Testing {len(list(n_components_range))} component counts: {list(n_components_range)}"
        )

    # PCA once for variance curve
    if verbose:
        print("Running PCA for variance analysis ...")
    temp_ica = FastICA(n_components=1, pca_components=pca_components, device=device)
    temp_ica.fit(X)
    evar = temp_ica.pca_.explained_variance_.cpu().numpy()
    pca_variance_curve = evar / evar.sum()
    pca_cumsum_curve = pca_variance_curve.cumsum()

    if verbose:
        print(
            f"  PCA: {len(pca_variance_curve)} components, "
            f"total variance: {pca_cumsum_curve[-1]:.1%}"
        )
        print()

    for n_comp in n_components_range:
        if verbose:
            print(f"{'=' * 60}")
            print(f"n_components = {n_comp}")
            if n_comp <= len(pca_cumsum_curve):
                print(f"PCA variance: {pca_cumsum_curve[n_comp - 1]:.1%}")
            print(f"{'=' * 60}")

        icasso_results = icasso(
            X,
            n_components=n_comp,
            n_runs=n_runs,
            pca_components=pca_components,
            min_stability=min_stability,
            device=device,
            verbose=verbose,
            batch_size=batch_size,
            ica_method=ica_method,
            base_seed=base_seed,
        )

        all_results[n_comp] = icasso_results
        n_stable_by_n[n_comp] = icasso_results["n_stable"]

        if verbose:
            print(f"  → {icasso_results['n_stable']}/{n_comp} stable")
            print()

    # Optimal = max ratio of stable/requested
    stability_ratios = {n: n_stable_by_n[n] / n for n in n_components_range}
    optimal_n = max(stability_ratios, key=stability_ratios.get)

    if verbose:
        print(f"\n{'=' * 60}")
        print("ICASSO Auto-Selection Results")
        print(f"{'=' * 60}")
        print(f"\n{'n_comp':>6} | {'stable':>6} | {'ratio':>5} | {'pca_var':>7}")
        print("-" * 35)
        for n in sorted(n_stable_by_n.keys()):
            ratio = stability_ratios[n]
            marker = " ←" if n == optimal_n else ""
            pca_var_str = f"{pca_cumsum_curve[n - 1]:.1%}" if n <= len(pca_cumsum_curve) else "N/A"
            print(f"{n:6d} | {n_stable_by_n[n]:6d} | {ratio:5.2f} | {pca_var_str:>7}{marker}")
        print(f"\nSelected n_components = {optimal_n} ({n_stable_by_n[optimal_n]} stable)")

    return {
        "optimal_n_components": optimal_n,
        "optimal_results": all_results[optimal_n],
        "n_stable_by_n_components": n_stable_by_n,
        "stability_ratios": stability_ratios,
        "all_results": all_results,
    }
