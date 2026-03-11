"""
ICASSO: ICA with component clustering for stability-based selection

Implements the ICASSO algorithm (Himberg et al., 2004) for finding reliable
ICA components through:
1. Running ICA many times with different random initializations
2. Clustering components across runs to find stable patterns
3. Selecting components based on cluster quality (compactness)

Reference:
Himberg, J., Hyvärinen, A., & Esposito, F. (2004). Validating the independent
components of neuroimaging time series via clustering and visualization.
NeuroImage, 22(3), 1214-1222.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from tqdm.auto import tqdm

from .ica import FastICA
from .utils import get_device, to_tensor


def compute_component_similarity(
    comp1: np.ndarray,
    comp2: np.ndarray,
) -> float:
    """
    Compute absolute correlation between two components

    ICA components are determined up to sign, so we use absolute correlation.

    Parameters
    ----------
    comp1, comp2 : np.ndarray
        Component spatial maps or timeseries

    Returns
    -------
    similarity : float
        Absolute correlation (0-1, higher = more similar)
    """
    # Flatten and compute correlation
    corr = np.corrcoef(comp1.flatten(), comp2.flatten())[0, 1]
    return np.abs(corr)


def compute_similarity_matrix(
    components_list: list[np.ndarray],
    batch_size: int | None = None,
) -> np.ndarray:
    """
    Compute pairwise similarity matrix across all components from all runs

    Parameters
    ----------
    components_list : list of np.ndarray
        List of component matrices from different ICA runs
        Each array has shape (n_components, n_features)
    batch_size : int, optional
        If provided, compute similarity in batches to save memory.
        Recommended for large datasets (e.g., batch_size=500)

    Returns
    -------
    similarity : np.ndarray, shape (n_total_components, n_total_components)
        Pairwise similarity matrix (absolute correlation)
    """
    # Stack all components
    n_runs = len(components_list)
    n_components_per_run = components_list[0].shape[0]
    n_features = components_list[0].shape[1]

    # Concatenate all components
    all_components = np.concatenate(components_list, axis=0)  # (n_runs * n_components, n_features)
    n_total = all_components.shape[0]

    # Normalize each component (row)
    norms = np.linalg.norm(all_components, axis=1, keepdims=True)
    all_components_norm = all_components / (norms + 1e-10)

    # Compute correlation matrix
    if batch_size is None or n_total <= batch_size:
        # Compute all at once (fast but memory-intensive)
        similarity = np.abs(all_components_norm @ all_components_norm.T)
    else:
        # Compute in batches (slower but memory-efficient)
        similarity = np.zeros((n_total, n_total), dtype=np.float32)
        n_batches = (n_total + batch_size - 1) // batch_size

        for i in range(n_batches):
            start_i = i * batch_size
            end_i = min((i + 1) * batch_size, n_total)
            batch_i = all_components_norm[start_i:end_i]

            # Compute this batch against all components
            similarity[start_i:end_i, :] = np.abs(batch_i @ all_components_norm.T)

    return similarity


def cluster_components(
    components_list: list[np.ndarray],
    method: str = "average",
    criterion: str = "maxclust",
    n_clusters: int | None = None,
    batch_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster components across ICA runs using hierarchical clustering

    Parameters
    ----------
    components_list : list of np.ndarray
        List of component matrices from different ICA runs
    method : str, default='average'
        Linkage method for hierarchical clustering
        Options: 'single', 'average', 'complete', 'ward'
    criterion : str, default='maxclust'
        Criterion for forming clusters ('maxclust' or 'distance')
    n_clusters : int, optional
        Number of clusters (if None, uses n_components from runs)
    batch_size : int, optional
        Batch size for memory-efficient similarity computation.
        Use for large datasets (e.g., batch_size=500)

    Returns
    -------
    cluster_labels : np.ndarray, shape (n_total_components,)
        Cluster assignment for each component
    similarity : np.ndarray, shape (n_total_components, n_total_components)
        Similarity matrix used for clustering
    """
    # Compute similarity matrix (with optional batching)
    similarity = compute_similarity_matrix(components_list, batch_size=batch_size)

    # Convert similarity to distance (dissimilarity)
    # Clip similarity to [0, 1] to avoid numerical issues
    similarity = np.clip(similarity, 0.0, 1.0)
    distance = 1 - similarity

    # Ensure distance matrix is valid (symmetric, non-negative)
    distance = np.maximum(distance, 0.0)  # Ensure non-negative
    distance = (distance + distance.T) / 2  # Ensure symmetric

    # Hierarchical clustering
    # Convert distance matrix to condensed form for scipy
    condensed_distance = squareform(distance, checks=False)
    linkage_matrix = linkage(condensed_distance, method=method)

    # Form clusters
    if n_clusters is None:
        n_clusters = components_list[0].shape[0]  # Same as n_components per run

    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion=criterion)

    return cluster_labels, similarity


def compute_cluster_quality(
    components_list: list[np.ndarray],
    cluster_labels: np.ndarray,
    similarity: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Compute quality metrics for each cluster

    Metrics:
    - Compactness (Iq): Average intra-cluster similarity (higher = more stable)
    - Isolation: Ratio of intra-cluster to inter-cluster similarity
    - Size: Number of components in cluster

    Parameters
    ----------
    components_list : list of np.ndarray
        Component matrices from ICA runs
    cluster_labels : np.ndarray
        Cluster assignment for each component
    similarity : np.ndarray
        Pairwise similarity matrix

    Returns
    -------
    quality : dict
        Dictionary with quality metrics:
        - 'compactness': Average intra-cluster similarity per cluster
        - 'isolation': Isolation index per cluster
        - 'size': Number of components per cluster
        - 'stability': Combined quality score (0-1)
    """
    n_clusters = cluster_labels.max()
    n_components_per_run = components_list[0].shape[0]

    compactness = np.zeros(n_clusters)
    isolation = np.zeros(n_clusters)
    size = np.zeros(n_clusters, dtype=int)

    for cluster_id in range(1, n_clusters + 1):
        # Get components in this cluster
        cluster_mask = cluster_labels == cluster_id
        size[cluster_id - 1] = cluster_mask.sum()

        if size[cluster_id - 1] < 2:
            # Need at least 2 components to compute metrics
            compactness[cluster_id - 1] = 0.0
            isolation[cluster_id - 1] = 0.0
            continue

        # Intra-cluster similarity (similarity within cluster)
        intra_sim = similarity[cluster_mask][:, cluster_mask]
        # Exclude diagonal (self-similarity)
        intra_sim_no_diag = intra_sim[~np.eye(intra_sim.shape[0], dtype=bool)]
        compactness[cluster_id - 1] = (
            intra_sim_no_diag.mean() if len(intra_sim_no_diag) > 0 else 0.0
        )

        # Inter-cluster similarity (similarity to other clusters)
        inter_sim = similarity[cluster_mask][:, ~cluster_mask]
        inter_sim_mean = inter_sim.mean() if inter_sim.size > 0 else 1.0

        # Isolation: ratio of intra to inter similarity
        if inter_sim_mean > 0:
            isolation[cluster_id - 1] = compactness[cluster_id - 1] / inter_sim_mean
        else:
            isolation[cluster_id - 1] = compactness[cluster_id - 1]

    # Combined stability score: weighted average of compactness and isolation
    # Normalize isolation to [0, 1] range
    isolation_norm = isolation / (isolation.max() + 1e-10)
    stability = 0.7 * compactness + 0.3 * isolation_norm

    return {
        "compactness": compactness,
        "isolation": isolation,
        "size": size,
        "stability": stability,
    }


def extract_cluster_centroids(
    components_list: list[np.ndarray],
    cluster_labels: np.ndarray,
) -> np.ndarray:
    """
    Extract representative component for each cluster (centroid)

    For each cluster, computes the average of all components in that cluster.

    Parameters
    ----------
    components_list : list of np.ndarray
        Component matrices from ICA runs
    cluster_labels : np.ndarray
        Cluster assignment for each component

    Returns
    -------
    centroids : np.ndarray, shape (n_clusters, n_features)
        Representative component for each cluster
    """
    # Stack all components
    all_components = np.concatenate(components_list, axis=0)

    n_clusters = cluster_labels.max()
    n_features = all_components.shape[1]

    centroids = np.zeros((n_clusters, n_features))

    for cluster_id in range(1, n_clusters + 1):
        cluster_mask = cluster_labels == cluster_id
        cluster_comps = all_components[cluster_mask]

        if cluster_comps.shape[0] > 0:
            # Average components in cluster (accounting for sign flips)
            # Use first component as reference
            ref_comp = cluster_comps[0]
            aligned_comps = []

            for comp in cluster_comps:
                # Flip sign if negative correlation with reference
                corr = np.corrcoef(ref_comp, comp)[0, 1]
                if corr < 0:
                    comp = -comp
                aligned_comps.append(comp)

            centroids[cluster_id - 1] = np.mean(aligned_comps, axis=0)

    return centroids


def icasso(
    X: np.ndarray | torch.Tensor,
    n_components: int,
    n_runs: int = 100,
    pca_components: int | float | str | None = 0.85,
    min_stability: float = 0.7,
    device: torch.device | None = None,
    verbose: bool = True,
    batch_size: int | None = None,
) -> dict:
    """
    Run ICASSO: ICA with component clustering for reliability assessment

    Runs ICA multiple times, clusters components across runs, and returns
    only stable, reliable components.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        fMRI data
    n_components : int
        Number of ICA components to extract per run
    n_runs : int, default=100
        Number of ICA runs with different random seeds
    pca_components : int, float, or str, optional
        PCA dimensionality reduction (default: 0.85 = 85% variance)
    min_stability : float, default=0.7
        Minimum stability threshold for keeping components
    device : torch.device, optional
        Computation device
    verbose : bool, default=True
        Show progress
    batch_size : int, optional
        Batch size for memory-efficient similarity computation.
        If None, auto-selects based on total components:
        - < 1000: no batching
        - >= 1000: batch_size = 500
        Use smaller values if running out of memory.

    Returns
    -------
    results : dict
        Dictionary with:
        - 'components': Stable component spatial maps
        - 'mixing': Corresponding timeseries (from best run)
        - 'stability': Stability score per component
        - 'cluster_quality': Full quality metrics per cluster
        - 'n_stable': Number of stable components found
        - 'cluster_labels': Cluster assignments
        - 'best_run_idx': Index of run closest to cluster centroids

    Examples
    --------
    >>> results = icasso(fmri_data, n_components=25, n_runs=100)
    >>> print(f"Found {results['n_stable']} stable components")
    >>> stable_maps = results['components']
    >>> stable_timeseries = results['mixing']
    """
    device = device if device is not None else get_device()
    X = to_tensor(X, device=device)

    # Auto-select batch size based on memory estimate
    n_total_components = n_runs * n_components
    if batch_size is None:
        # Estimate memory needed for similarity matrix (float64)
        # Matrix size: n_total × n_total × 8 bytes
        matrix_gb = (n_total_components**2 * 8) / (1024**3)

        # Use batching if matrix > 1 GB (conservative threshold for CPU RAM)
        if matrix_gb > 1.0:
            # Aim for ~500 MB chunks
            batch_size = max(100, int(np.sqrt(500 * 1024**3 / 8)))
            if verbose:
                print(f"Estimated similarity matrix: {matrix_gb:.2f} GB")
                print(f"Auto-selected batch_size={batch_size} for memory efficiency")

    if verbose:
        print(
            f"Running ICASSO: {n_runs} ICA runs × {n_components} components = {n_total_components} total"
        )
        print(f"Memory mode: {'batched' if batch_size else 'standard'}")

    # Step 1: Run ICA multiple times
    components_list = []
    mixing_list = []
    pca_variance_explained = None  # Track PCA variance from first run

    pca_eigenvalues = None
    pca_components_arr = None

    iterator = range(n_runs)
    if verbose:
        iterator = tqdm(iterator, desc=f"ICA runs ({n_components} components each)")

    for i in iterator:
        ica = FastICA(
            n_components=n_components,
            pca_components=pca_components,
            random_state=i,
            device=device,
        )
        ica.fit(X)

        # Save PCA variance from first run (same for all runs with same pca_components)
        if i == 0:
            pca_variance_explained = ica.pca_.explained_variance_ratio_.cpu().numpy()
            pca_eigenvalues = ica.pca_.explained_variance_.cpu().numpy()
            pca_components_arr = ica.pca_.components_.cpu().numpy()

        # Move to CPU immediately and convert to numpy to free GPU memory
        components_list.append(ica.components_.cpu().numpy())
        mixing_list.append(ica.mixing_.cpu().numpy())

        # Clear GPU cache periodically to prevent memory accumulation
        if device.type == "cuda" and i % 10 == 9:
            torch.cuda.empty_cache()

    # Step 2: Cluster components
    if verbose:
        print(f"Clustering {n_total_components} components across runs...")
        if batch_size:
            print(f"  Using batched computation (batch_size={batch_size}) to save memory")

    cluster_labels, similarity = cluster_components(
        components_list,
        method="average",
        n_clusters=n_components,
        batch_size=batch_size,
    )

    # Step 3: Compute cluster quality
    quality = compute_cluster_quality(components_list, cluster_labels, similarity)

    # Step 4: Select stable clusters
    stable_mask = quality["stability"] >= min_stability
    n_stable = stable_mask.sum()

    if verbose:
        print(
            f"Found {n_stable}/{n_components} stable components (stability >= {min_stability:.2f})"
        )
        print(f"  Mean stability: {quality['stability'].mean():.3f}")
        print(f"  Mean compactness: {quality['compactness'].mean():.3f}")

    # Step 5: Extract stable component centroids
    centroids = extract_cluster_centroids(components_list, cluster_labels)
    stable_components = centroids[stable_mask]

    # Step 6: Find best run (most similar to centroids)
    # Compute similarity between each run and centroids
    best_run_idx = find_best_run(components_list, centroids[stable_mask])

    # Get mixing matrix from best run (aligned to stable components)
    best_components = components_list[best_run_idx]
    best_mixing = mixing_list[best_run_idx]

    # Match stable centroids to best run components
    stable_mixing = match_components_to_centroids(best_components, best_mixing, stable_components)

    # Match ALL centroids to best run components (for saving all components)
    all_mixing = match_components_to_centroids(
        best_components,
        best_mixing,
        centroids,  # ALL centroids, not just stable
    )

    return {
        "components": stable_components,
        "mixing": stable_mixing,
        "stability": quality["stability"][stable_mask],
        "all_centroids": centroids,  # ALL component centroids (not just stable)
        "all_mixing": all_mixing,  # Mixing for ALL centroids
        "all_stability": quality["stability"],  # Stability for ALL components
        "cluster_quality": quality,
        "n_stable": n_stable,
        "n_components": n_components,
        "cluster_labels": cluster_labels,
        "best_run_idx": best_run_idx,
        "all_components": components_list,
        "all_mixing_list": mixing_list,  # All mixing matrices from all runs
        "similarity": similarity,
        "pca_variance_explained": pca_variance_explained,  # Variance explained by PCA
        "pca_variance_cumsum": pca_variance_explained.cumsum()
        if pca_variance_explained is not None
        else None,
        "pca_eigenvalues": pca_eigenvalues,  # Raw PCA eigenvalues (explained_variance_)
        "pca_components": pca_components_arr,  # PCA spatial components (k, V)
    }


def find_best_run(
    components_list: list[np.ndarray],
    target_components: np.ndarray,
) -> int:
    """
    Find ICA run with components most similar to target

    Parameters
    ----------
    components_list : list of np.ndarray
        Component matrices from different runs
    target_components : np.ndarray
        Target components to match

    Returns
    -------
    best_idx : int
        Index of best matching run
    """
    best_similarity = -1
    best_idx = 0

    for i, components in enumerate(components_list):
        # Compute average max similarity between target and run components
        similarities = []
        for target in target_components:
            # Find best match in this run
            max_sim = 0
            for comp in components:
                sim = compute_component_similarity(target, comp)
                max_sim = max(max_sim, sim)
            similarities.append(max_sim)

        avg_similarity = np.mean(similarities)
        if avg_similarity > best_similarity:
            best_similarity = avg_similarity
            best_idx = i

    return best_idx


def match_components_to_centroids(
    components: np.ndarray,
    mixing: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """
    Match components from a run to cluster centroids and extract timeseries

    Parameters
    ----------
    components : np.ndarray, shape (n_components, n_features)
        Components from single ICA run
    mixing : np.ndarray, shape (n_samples, n_components)
        Mixing matrix (timeseries) from same run
    centroids : np.ndarray, shape (n_centroids, n_features)
        Target cluster centroids

    Returns
    -------
    matched_mixing : np.ndarray, shape (n_samples, n_centroids)
        Timeseries matching the centroids
    """
    n_samples = mixing.shape[0]
    n_centroids = centroids.shape[0]
    matched_mixing = np.zeros((n_samples, n_centroids))

    for i, centroid in enumerate(centroids):
        # Find best matching component
        best_similarity = -1
        best_idx = 0
        best_sign = 1

        for j, comp in enumerate(components):
            corr = np.corrcoef(centroid.flatten(), comp.flatten())[0, 1]
            sim = np.abs(corr)

            if sim > best_similarity:
                best_similarity = sim
                best_idx = j
                best_sign = 1 if corr > 0 else -1

        # Extract corresponding timeseries with correct sign
        matched_mixing[:, i] = best_sign * mixing[:, best_idx]

    return matched_mixing


def icasso_auto_select(
    X: np.ndarray | torch.Tensor,
    n_components_range: range | list[int],
    n_runs: int = 50,
    pca_components: int | float | str | None = 0.85,
    min_stability: float = 0.7,
    device: torch.device | None = None,
    verbose: bool = True,
    batch_size: int | None = None,
) -> dict:
    """
    Automatically select optimal number of ICA components using ICASSO

    Runs ICASSO for different numbers of components and selects the number
    that produces the most stable decomposition.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        fMRI data
    n_components_range : range or list of int
        Range of component numbers to test (e.g., range(15, 35, 5))
    n_runs : int, default=50
        Number of ICA runs per component number
    pca_components : int, float, or str, optional
        PCA dimensionality reduction
    min_stability : float, default=0.7
        Minimum stability threshold
    device : torch.device, optional
        Computation device
    verbose : bool, default=True
        Show progress
    batch_size : int, optional
        Batch size for memory-efficient similarity computation

    Returns
    -------
    results : dict
        Dictionary with:
        - 'optimal_n_components': Recommended number of components
        - 'optimal_results': Full ICASSO results for optimal n_components
        - 'n_stable_by_n_components': Number of stable components found for each n
        - 'all_results': Dict of ICASSO results for all n_components tested

    Examples
    --------
    >>> results = icasso_auto_select(
    ...     fmri_data,
    ...     n_components_range=range(15, 35, 5),
    ...     n_runs=50
    ... )
    >>> print(f"Optimal: {results['optimal_n_components']} components")
    >>> stable_maps = results['optimal_results']['components']
    """
    all_results = {}
    n_stable_by_n = {}

    if verbose:
        print(f"Testing {len(list(n_components_range))} different component numbers...")
        print(f"Range: {list(n_components_range)}")
        print()

    # Run PCA once to get variance explained curve
    # This allows us to show variance BEFORE each ICA test
    if verbose:
        print("\nRunning PCA for variance analysis...")
    from .ica import FastICA

    temp_ica = FastICA(n_components=1, pca_components=pca_components, device=device)
    temp_ica.fit(X)
    pca_variance_curve = temp_ica.pca_.explained_variance_ratio_.cpu().numpy()
    pca_cumsum_curve = pca_variance_curve.cumsum()

    if verbose:
        n_pca_total = len(pca_variance_curve)
        print(f"  PCA extracted {n_pca_total} components")
        print(f"  Total variance explained: {pca_cumsum_curve[-1]:.1%}")
        print()

    for n_comp in n_components_range:
        if verbose:
            print(f"{'=' * 60}")
            print(f"Testing n_components = {n_comp}")
            print(f"{'=' * 60}")

            # Show PCA variance BEFORE running ICA
            if n_comp <= len(pca_cumsum_curve):
                pca_var = pca_cumsum_curve[n_comp - 1]
                print(f"PCA: First {n_comp} components explain {pca_var:.1%} of variance")

        # Run ICASSO
        icasso_results = icasso(
            X,
            n_components=n_comp,
            n_runs=n_runs,
            pca_components=pca_components,
            min_stability=min_stability,
            device=device,
            verbose=verbose,
            batch_size=batch_size,
        )

        all_results[n_comp] = icasso_results
        n_stable_by_n[n_comp] = icasso_results["n_stable"]

        if verbose:
            print(f"Stable components: {icasso_results['n_stable']}/{n_comp}")
            print()

    # Select optimal: maximize ratio of stable components to requested
    stability_ratios = {n: n_stable_by_n[n] / n for n in n_components_range}
    optimal_n = max(stability_ratios, key=stability_ratios.get)

    if verbose:
        print(f"\n{'=' * 60}")
        print("ICASSO Automatic Selection Results")
        print(f"{'=' * 60}")
        print("\nn_components | n_stable | ratio | pca_var")
        print("-" * 47)
        for n in sorted(n_stable_by_n.keys()):
            ratio = stability_ratios[n]
            marker = " <-- OPTIMAL" if n == optimal_n else ""

            # Get PCA variance for this n_components
            pca_var_cumsum = all_results[n].get("pca_variance_cumsum")
            if pca_var_cumsum is not None and len(pca_var_cumsum) >= n:
                pca_var = pca_var_cumsum[n - 1]
                print(f"{n:12d} | {n_stable_by_n[n]:8d} | {ratio:5.2f} | {pca_var:6.1%}{marker}")
            else:
                print(f"{n:12d} | {n_stable_by_n[n]:8d} | {ratio:5.2f} | N/A{marker}")
        print()
        print(f"Selected n_components = {optimal_n}")
        print(f"Found {n_stable_by_n[optimal_n]} stable components")

    return {
        "optimal_n_components": optimal_n,
        "optimal_results": all_results[optimal_n],
        "n_stable_by_n_components": n_stable_by_n,
        "stability_ratios": stability_ratios,
        "all_results": all_results,
    }
