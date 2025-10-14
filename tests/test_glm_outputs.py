"""Tests for GLM NIfTI export utilities."""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from fastfuncsim.glm_core import fit_glm
from fastfuncsim.glm_outputs import write_glm_results_nifti


def test_write_glm_results_nifti_roundtrip():
    device = torch.device("cpu")
    torch.manual_seed(0)
    
    # Simulate a tiny dataset
    n_timepoints = 60
    n_voxels = 8
    n_regressors = 2
    data = torch.randn(n_voxels, n_timepoints, device=device)
    design = torch.randn(n_timepoints, n_regressors, device=device)
    
    results = fit_glm(data, design, tr=1.5, verbose=False, device=device)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out = write_glm_results_nifti(
            results,
            tmpdir,
            prefix="unit",
            condition_names=["stim_a", "stim_b"],
            include_beta=True,
            include_tstat=True,
            include_fstat=True,
            include_r2=True,
            include_mean=True,
            include_sigma=True,
            write_residuals=False,
            write_predictions=False,
            volume_shape=(2, 2, 2),
        )
        
        stats_path = Path(out['stats'])
        assert stats_path.exists(), "Stats NIfTI missing"
        
        # There should be 4 volumes: beta/tstat pair per condition
        import nibabel as nib
        stats_img = nib.load(str(stats_path))
        assert stats_img.shape == (2, 2, 2, 4), "Unexpected stats volume shape"
        assert np.isclose(stats_img.header['pixdim'][4], 1.5), "TR not encoded"
        
        meta = json.loads(Path(out['stats_meta']).read_text())
        assert len(meta['volumes']) == 4, "Metadata length mismatch"
        assert meta['volumes'][0]['condition'] == 'stim_a'
        assert meta['volumes'][0]['metric'] == 'beta'
        assert meta['volumes'][1]['condition'] == 'stim_b'
        assert meta['volumes'][1]['metric'] == 'beta'
        assert meta['volumes'][2]['condition'] == 'stim_a'
        assert meta['volumes'][2]['metric'] == 'tstat'
        assert meta['volumes'][3]['condition'] == 'stim_b'
        assert meta['volumes'][3]['metric'] == 'tstat'

        assert Path(out['fstat']).exists(), "F-stat map missing"
        assert Path(out['r2']).exists(), "R2 map missing"
        assert Path(out['mean']).exists(), "Mean volume missing"
        assert Path(out['sigma']).exists(), "Sigma volume missing"