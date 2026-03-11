
import numpy as np
import pytest
import torch
from scipy.stats import pearsonr
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.decomposition import FastICA as SklearnFastICA

from fastfuncsim.ica import FastICA
from fastfuncsim.pca import PCA


def check_sign_flip_correlation(a, b):
    """
    Check if columns/rows of a and b are correlated, allowing for sign flip.
    Returns mean absolute correlation.
    Assumes a and b are (n_components, n_features).
    """
    n_comps = a.shape[0]
    corrs = []
    
    # For each component in a, find best match in b
    # This is a greedy matching, but sufficient for verification if correct
    used_indices = set()
    
    for i in range(n_comps):
        vec_a = a[i]
        best_corr = 0.0
        best_idx = -1
        
        for j in range(n_comps):
            if j in used_indices:
                continue
            
            # Compute correlation
            if isinstance(vec_a, torch.Tensor):
                vec_a = vec_a.cpu().numpy()
            if isinstance(b, torch.Tensor):
                b_np = b.cpu().numpy()
            else:
                b_np = b
                
            vec_b = b_np[j]
            
            # Simple pearson correlation
            r, _ = pearsonr(vec_a.flatten(), vec_b.flatten())
            if abs(r) > abs(best_corr):
                best_corr = r
                best_idx = j
        
        if best_idx != -1:
            used_indices.add(best_idx)
            corrs.append(abs(best_corr))
            
    return np.mean(corrs) if corrs else 0.0

@pytest.mark.parametrize("n_samples, n_features, n_components", [
    (100, 50, 10),
    (50, 100, 10), # n_features > n_samples
    (200, 20, 0.95), # fraction
])
def test_pca_vs_sklearn(n_samples, n_features, n_components):
    """
    Verify fastfuncsim.PCA matches sklearn.PCA
    """
    # Generate random data
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, n_features)
    
    # Sklearn PCA
    sk_pca = SklearnPCA(n_components=n_components, svd_solver='full', random_state=42)
    X_sk = sk_pca.fit_transform(X)
    
    # Check if number of components matches (for float n_components)
    n_comp_actual = sk_pca.n_components_
    
    # FastFuncSim PCA
    ffs_pca = PCA(n_components=n_components, device=torch.device('cpu')) # Test CPU first for strict numerical match
    X_ffs = ffs_pca.fit_transform(X)
    
    # 1. Verify n_components matches
    assert ffs_pca.n_components_ == n_comp_actual
    
    # 2. Verify explained variance ratio
    # Allow small tolerance
    np.testing.assert_allclose(
        ffs_pca.explained_variance_ratio_.numpy(), 
        sk_pca.explained_variance_ratio_, 
        atol=1e-5
    )
    
    # 3. Verify components (eigenvectors)
    # Signs might be flipped
    mean_corr = check_sign_flip_correlation(ffs_pca.components_, sk_pca.components_)
    assert mean_corr > 0.99, f"PCA Components do not match sklearn (mean corr={mean_corr})"
    
    # 4. Verify transformed data (scores)
    # Signs might be flipped per component
    # We can check correlation of scores column by column
    X_ffs_np = X_ffs.numpy()
    score_corrs = []
    for i in range(n_comp_actual):
        r, _ = pearsonr(X_ffs_np[:, i], X_sk[:, i])
        score_corrs.append(abs(r))
    
    assert np.mean(score_corrs) > 0.99, f"PCA Scores do not match sklearn (mean corr={np.mean(score_corrs)})"


def test_pca_gpu_execution():
    """Verify PCA runs on GPU if available"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    X = np.random.randn(100, 50).astype(np.float32)
    
    pca = PCA(n_components=10, device=torch.device('cuda'))
    X_trans = pca.fit_transform(X)
    
    assert X_trans.device.type == 'cuda'
    assert pca.components_.device.type == 'cuda'
    assert pca.explained_variance_.device.type == 'cuda'


def test_ica_spatial_recovery():
    """
    Test that FastICA recovers known spatial sources.
    
    We simulate: X = Mixing @ Sources
    Mixing: (Time, n_components)
    Sources: (n_components, Voxels) -> Independently generated
    X: (Time, Voxels)
    
    Our FastICA implementation is Spatial ICA.
    It should recover Sources (up to sign/permutation) in .components_
    """
    n_time = 200
    n_voxels = 1000
    n_comps = 3
    
    rng = np.random.RandomState(42)
    
    # 1. Generate independent spatial sources (super-gaussian)
    # e.g. potentially overlapping blobs or random sparse noise
    S_true = rng.laplace(size=(n_comps, n_voxels))
    
    # 2. Generate mixing matrix (timecourses)
    M_true = rng.randn(n_time, n_comps)
    
    # 3. Create data
    X = M_true @ S_true # (200, 1000)
    
    # Add some noise
    X += 0.01 * rng.randn(n_time, n_voxels)
    
    # Fit FastICA
    # Note: we need to set n_components explicitly
    # pca_components must be >= n_components
    ica = FastICA(n_components=n_comps, pca_components=n_comps + 2, random_state=42, device=torch.device('cpu'))
    S_est = ica.fit_transform(X) # Returns mixing (Time x Comps) for Spatial ICA? 
    # Wait, docstring says:
    # transform -> S (n_samples, n_components). 
    # But for Spatial ICA, we usually want the SPATIAL MAPS.
    # FastICA.components_ are the spatial maps (n_comps, n_voxels).
    
    # Check spatial maps recovery
    mean_corr_spatial = check_sign_flip_correlation(ica.components_, S_true)
    assert mean_corr_spatial > 0.90, f"Failed to recover spatial sources (corr={mean_corr_spatial})"
    
    # Check timecourse recovery (mixing matrix)
    # ica.mixing_ is (Time, Comps)
    # We need to match columns of mixing_ to columns of M_true
    # But order might be permuted same as spatial maps.
    # Actually check_sign_flip_correlation expects (n_features, n_samples) rows?
    # No, it iterates over rows (dim 0).
    # M_true is (Time, Comps). We want to check columns. Transpose.
    mean_corr_temporal = check_sign_flip_correlation(ica.mixing_.T, M_true.T)
    assert mean_corr_temporal > 0.90, f"Failed to recover mixing timecourses (corr={mean_corr_temporal})"

def test_ica_vs_sklearn_spatial_logic():
    """
    Compare fastfuncsim.FastICA (Spatial) with sklearn.FastICA on Transposed data.
    
    fastfuncsim: input (Time, Voxels) -> finds independent rows of components_ (Voxels)
    sklearn: input (Voxels, Time) -> finds independent rows of components_ (Time) ???
    
    No.
    Sklearn FastICA(X): X = S @ A.T. Assumes S are independent. S has same n_samples as X.
    
    If we feed X_sk = X_ffs.T (Voxels, Time).
    Sklearn finds S (Voxels, Comps) that are independent.
    So S (from sklearn transform) should match ica.components_.T (from ffs).
    
    Let's verify this specific equivalence.
    """
    n_time = 100
    n_voxels = 500
    n_comps = 5
    
    rng = np.random.RandomState(42)
    S_true = rng.laplace(size=(n_comps, n_voxels))
    M_true = rng.randn(n_time, n_comps)
    X = M_true @ S_true # (100, 500)
    
    # 1. fastfuncsim FastICA (Spatial)
    # It does PCA first internally.
    # To match sklearn exactly, sklearn must also do PCA.
    # But sklearn PCA is on columns.
    # ffs does PCA on X (Time x Voxels) -> X_pca (Time x n_comps).
    # Then ICA on PCA components.
    
    ica_ffs = FastICA(
        n_components=n_comps, 
        pca_components=n_comps, # Whiten/Reduce exactly to n_comps
        whiten=True,
        random_state=42,
        device=torch.device('cpu') 
    )
    ica_ffs.fit(X)
    spatial_maps_ffs = ica_ffs.components_ # (n_comps, n_voxels)
    
    # 2. Sklearn FastICA
    # We want Spatial ICA. So samples = Voxels.
    # Input to sklearn: (Voxels, Time) = X.T
    X_sk = X.T
    
    ica_sk = SklearnFastICA(
        n_components=n_comps,
        whiten='unit-variance', # Sklearn default whitening
        random_state=42
    )
    # transform returns S (n_samples, n_components) = (Voxels, Comps)
    spatial_maps_sk = ica_sk.fit_transform(X_sk).T # (n_comps, n_voxels)
    
    # Note: The exact algorithms differ slightly (solver, whitening details).
    # But for clean data, they should find similar subspaces.
    # With PCA preprocessing in FFS, it might differ from Sklearn if Sklearn does whitening differently.
    # Sklearn whiten='unit-variance' does PCA whitening.
    
    mean_corr = check_sign_flip_correlation(spatial_maps_ffs, spatial_maps_sk)
    
    # Warning: FFS uses symmetric decorrelation vs Sklearn might use deflation or parallel.
    # FFS default fun='logcosh', same as sklearn.
    # Expect moderate to high correlation if implementations are sound.
    # Relax threshold slightly due to implementation diffs (svd solver, etc)
    print(f"Mean spatial correlation FFS vs Sklearn: {mean_corr}")
    assert mean_corr > 0.8
