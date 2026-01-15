
import numpy as np
import torch
import pytest
import nibabel as nib
from pathlib import Path
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.decomposition import FastICA as SklearnFastICA
from scipy.stats import pearsonr

from fastfuncsim.pca import PCA
from fastfuncsim.ica import FastICA
from fastfuncsim.utils import get_device

DATA_PATH = Path("/home/logan/Dropbox/Resources/code/fastfuncsim/test_data/small_validation_afni_data/small_test_r01.nii.gz")

def check_sign_flip_correlation(a, b):
    """
    Check if columns/rows of a and b are correlated, allowing for sign flip.
    Returns mean absolute correlation.
    Assumes a and b are (n_components, n_features).
    """
    n_comps = a.shape[0]
    corrs = []
    
    used_indices = set()
    
    for i in range(n_comps):
        vec_a = a[i]
        best_corr = 0.0
        best_idx = -1
        
        for j in range(n_comps):
            if j in used_indices:
                continue
            
            if isinstance(vec_a, torch.Tensor):
                vec_a = vec_a.detach().cpu().numpy()
            if isinstance(b, torch.Tensor):
                b_np = b.detach().cpu().numpy()
            else:
                b_np = b
                
            vec_b = b_np[j]
            
            r, _ = pearsonr(vec_a.flatten(), vec_b.flatten())
            if abs(r) > abs(best_corr):
                best_corr = r
                best_idx = j
        
        if best_idx != -1:
            used_indices.add(best_idx)
            corrs.append(abs(best_corr))
            
    return np.mean(corrs) if corrs else 0.0

def load_real_data(masked=False):
    if not DATA_PATH.exists():
        pytest.skip(f"Data not found at {DATA_PATH}")
        
    img = nib.load(DATA_PATH)
    data = img.get_fdata() # (x, y, z, t)
    
    # Flatten: (t, voxels)
    nx, ny, nz, nt = data.shape
    X = data.reshape(-1, nt).T # (nt, n_voxels)
    
    # Simple masking: remove constant voxels
    if masked:
        var = X.var(axis=0)
        mask = var > 1e-6
        X = X[:, mask]
        print(f"Masked data shape: {X.shape} (from {nx*ny*nz} voxels)")
    else:
        print(f"Data shape: {X.shape}")
        
    return X, nt

def test_real_data_pca_vs_sklearn():
    """Verify PCA on real fMRI data"""
    X, n_samples = load_real_data(masked=True) # Mask to speed up and reduce noise
    
    n_components = 20
    
    # Sklearn
    print("Running Sklearn PCA...")
    sk_pca = SklearnPCA(n_components=n_components, svd_solver='full', random_state=42)
    X_sk = sk_pca.fit_transform(X)
    
    # FastFuncSim (GPU)
    print("Running FastFuncSim PCA...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ffs_pca = PCA(n_components=n_components, device=device)
    X_ffs = ffs_pca.fit_transform(X)
    
    # Compare Explained Variance Ratio
    print("Explained Variance Ratio Diff:", 
          np.abs(ffs_pca.explained_variance_ratio_.cpu().numpy() - sk_pca.explained_variance_ratio_).max())
    np.testing.assert_allclose(
        ffs_pca.explained_variance_ratio_.cpu().numpy(), 
        sk_pca.explained_variance_ratio_, 
        atol=1e-4
    )
    
    # Compare Components (Spatial Maps)
    # Signs might be flipped
    mean_corr = check_sign_flip_correlation(ffs_pca.components_, sk_pca.components_)
    print(f"PCA Components Mean Correlation: {mean_corr}")
    assert mean_corr > 0.98

def test_real_data_ica_vs_sklearn():
    """Verify Spatial ICA on real fMRI data"""
    X, n_samples = load_real_data(masked=True)
    
    # Reduce dimensions first for speed/stability
    n_components = 10
    
    # FastFuncSim (Spatial ICA)
    print("Running FastFuncSim ICA...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ica_ffs = FastICA(
        n_components=n_components, 
        pca_components=n_components, # Exact match
        whiten=True,
        random_state=42, 
        device=device,
        tol=1e-3
    )
    # We can't easily inject X used by sklearn unless we manually do it.
    
    # Better approach: 
    # 1. Run FFS ICA up to PCA stage
    ica_ffs.pca_ = PCA(n_components=n_components, whiten=True, device=device)
    X_pca_ffs = ica_ffs.pca_.fit_transform(X) # (Samp, Comp)
    
    # Get the whitening matrix / components
    # ica_ffs components_: (Comp, Voxels)
    # This is "whitened" spatial data (Rows are orthonormal)
    pca_comp = ica_ffs.pca_.components_ # (Comp, Voxels)
    n_voxels = pca_comp.shape[1]
    
    # The input to FFS _fastica is:
    X_white_ffs = pca_comp * np.sqrt(n_voxels) # (n_comp, n_voxels)
    
    # Shared Initialization
    torch.manual_seed(42)
    w_init_torch = torch.randn(n_components, n_components, device=device, dtype=X_white_ffs.dtype)
    w_init_np = w_init_torch.cpu().numpy()
    
    # Run FFS Core
    W_ffs, _ = ica_ffs._fastica(X_white_ffs, n_components, w_init=w_init_torch)
    # FFS Components = W @ pca_comp
    spatial_maps_ffs = W_ffs @ pca_comp
    
    # Run Sklearn on SAME input
    # Sklearn input: (n_samples, n_features) -> (n_voxels, n_comp)
    # So we input X_white_ffs.T
    X_input_sk = X_white_ffs.cpu().numpy().T
    
    print("Running Sklearn ICA (Core)...")
    ica_sk = SklearnFastICA(
        n_components=n_components,
        whiten=False, # ALREADY WHITENED
        algorithm='parallel', # Default, matches FFS symmetric
        fun='logcosh', # Default
        random_state=42, # Ignored if w_init is provided?
        w_init=w_init_np, # START FROM SAME POINT
        max_iter=1000,
        tol=1e-3
    )
    # Sklearn finds sources S. 
    # fit_transform returns S (n_samples, n_components) -> (Voxels, Comp)
    S_sk = ica_sk.fit_transform(X_input_sk)
    
    # The columns of S_sk should match rows of spatial_maps_ffs?
    # Wait. 
    # FFS: W is unmixing of components. 
    # maps = W @ pca_comps.
    # X_white = pca_comps * sqrt(N).
    # so maps = W @ (X_white / sqrt(N)).
    #
    # Sklearn: Input X_in = X_white.T.
    # Model: X_in = S @ A.T. -> S are independent. Use W to get S = X_in @ W.T
    # So S (Voxels, Comp) are the independent components found from X_white.
    # These should match the independent components found by FFS from X_white.
    # FFS W finds independent rows.
    # Sklearn finds independent columns.
    # Basically S_sk.T (Comp, Voxels) should match spatial_maps_ffs?
    # NO.
    # spatial_maps_ffs are result of applying W to original PCA maps.
    # S_sk are result of applying Sklearn W to X_input_sk (which is scaled PCA maps).
    # So S_sk * sqrt(N) ???
    # Or just check correlation. Scale doesn't affect correlation.
    
    spatial_maps_sk = S_sk.T # (Comp, Voxels)
    
    mean_corr = check_sign_flip_correlation(spatial_maps_ffs, spatial_maps_sk)
    print(f"ICA Core Spatial Correlation: {mean_corr}")
    
    assert mean_corr > 0.90 # Should be very high if algorithms are equiv
