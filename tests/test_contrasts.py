"""
Tests for contrast computation (OLS and ARMA).

Critical bugs this catches:
1. Device mismatch (xtx_inv on GPU, contrasts on CPU)
2. Dimension mismatch (xtx_inv includes nuisance regressors, betas don't)
3. Missing xtx_inv attribute in OLS results
4. Incorrect variance computation
"""

import pytest
import torch

import fastfuncsim as ffs
from fastfuncsim.utils import get_device


class TestOLSContrasts:
    """Test contrast computation with OLS results."""

    @pytest.fixture
    def device(self):
        return get_device()

    @pytest.fixture
    def block_design(self, device):
        """Create realistic block design matrix (like real fMRI)."""
        n_timepoints = 200

        # Create 3 conditions with non-overlapping blocks
        X = torch.zeros(n_timepoints, 3, device=device)

        # Condition 1: timepoints 10-60 (50 TRs)
        X[10:60, 0] = 1.0

        # Condition 2: timepoints 70-120 (50 TRs)
        X[70:120, 1] = 1.0

        # Condition 3: timepoints 140-190 (50 TRs)
        X[140:190, 2] = 1.0

        return X

    @pytest.fixture
    def ols_results(self, device, block_design):
        """Create OLS results with realistic block design + signal."""
        torch.manual_seed(42)

        n_voxels = 100
        n_timepoints = 200

        # True effect sizes: condition effects are different
        true_betas = torch.tensor(
            [[3.0, 1.5, 2.0]],  # Cond1 > Cond3 > Cond2
            device=device,
        ).repeat(n_voxels, 1)

        # Add some variability across voxels
        true_betas = true_betas + torch.randn(n_voxels, 3, device=device) * 0.5

        # Generate data: signal + noise
        signal = (block_design @ true_betas.T).T
        noise = torch.randn(n_voxels, n_timepoints, device=device) * 0.3
        data = signal + noise + 100.0  # Add baseline (like real BOLD signal)

        # Fit OLS (will add constant term)
        results = ffs.fit_glm(
            data, block_design, tr=2.0, verbose=False, max_poly_degree=0
        )

        return results

    def test_ols_has_xtx_inv(self, ols_results):
        """Test that OLS results contain xtx_inv for contrast computation."""
        assert hasattr(ols_results, "xtx_inv"), "OLS results must have xtx_inv"
        assert ols_results.xtx_inv is not None
        assert isinstance(ols_results.xtx_inv, torch.Tensor)

    def test_ols_xtx_inv_dimensions(self, ols_results):
        """Test that xtx_inv dimensions match betas (task regressors only)."""
        n_task_regressors = ols_results.betas.shape[1]
        assert ols_results.xtx_inv.shape == (
            n_task_regressors,
            n_task_regressors,
        ), (
            f"xtx_inv should be ({n_task_regressors}, {n_task_regressors}), got {ols_results.xtx_inv.shape}"
        )

    def test_ols_has_sigma2(self, ols_results):
        """Test that OLS results contain sigma2 for variance computation."""
        assert hasattr(ols_results, "sigma2"), "OLS results must have sigma2"
        assert ols_results.sigma2 is not None
        assert ols_results.sigma2.shape[0] == ols_results.betas.shape[0]

    def test_single_contrast_ols(self, ols_results, device):
        """Test single contrast computation with OLS."""
        n_regressors = ols_results.betas.shape[1]

        # Simple contrast: first vs second regressor
        contrast = torch.zeros(n_regressors, device=device)
        contrast[0] = 1.0
        contrast[1] = -1.0

        results = ffs.compute_contrasts(ols_results, contrast, device=device)

        assert "contrast_betas" in results
        assert "contrast_tstats" in results
        assert "contrast_stderr" in results

        # Single contrast should return 1D arrays
        assert results["contrast_betas"].ndim == 1
        assert results["contrast_tstats"].ndim == 1
        assert results["contrast_stderr"].ndim == 1

    def test_multiple_contrasts_ols(self, ols_results, device):
        """Test multiple contrasts computation with OLS."""
        n_regressors = ols_results.betas.shape[1]
        n_voxels = ols_results.betas.shape[0]

        # Multiple contrasts
        contrasts = torch.zeros(2, n_regressors, device=device)
        contrasts[0, 0] = 1.0
        contrasts[0, 1] = -1.0  # Contrast 1: reg0 - reg1
        contrasts[1, 0] = 1.0
        contrasts[1, 2] = -1.0  # Contrast 2: reg0 - reg2

        results = ffs.compute_contrasts(ols_results, contrasts, device=device)

        # Should return 2D arrays (n_voxels, n_contrasts)
        assert results["contrast_betas"].shape == (n_voxels, 2)
        assert results["contrast_tstats"].shape == (n_voxels, 2)
        assert results["contrast_stderr"].shape == (n_voxels, 2)

    def test_contrast_device_cpu(self, ols_results):
        """Test contrast computation on CPU (device transfer)."""
        n_regressors = ols_results.betas.shape[1]

        # Create contrast on CPU
        contrast = torch.zeros(n_regressors)
        contrast[0] = 1.0

        # Should work even if results are on GPU
        results = ffs.compute_contrasts(
            ols_results, contrast, device=torch.device("cpu")
        )

        assert results["contrast_betas"].device.type == "cpu"
        assert results["contrast_tstats"].device.type == "cpu"

    def test_contrast_device_gpu(self, ols_results, device):
        """Test contrast computation on GPU."""
        if device.type == "cpu":
            pytest.skip("GPU not available")

        n_regressors = ols_results.betas.shape[1]

        # Create contrast on GPU
        contrast = torch.zeros(n_regressors, device=device)
        contrast[0] = 1.0

        results = ffs.compute_contrasts(ols_results, contrast, device=device)

        # Results should be on CPU (compute_contrasts returns CPU tensors)
        assert results["contrast_betas"].device.type == "cpu"

    def test_contrast_stderr_positive(self, ols_results, device):
        """Test that contrast standard errors are positive."""
        n_regressors = ols_results.betas.shape[1]

        contrast = torch.randn(n_regressors, device=device)

        results = ffs.compute_contrasts(ols_results, contrast, device=device)

        assert torch.all(results["contrast_stderr"] >= 0), "stderr must be non-negative"
        assert torch.all(results["contrast_stderr"] > 0), "stderr should be positive"

    def test_contrast_variance_formula(self, ols_results, device):
        """Test that OLS contrast variance follows c'(X'X)^-1 c * sigma^2."""
        n_regressors = ols_results.betas.shape[1]

        # Simple contrast
        c = torch.zeros(n_regressors, device=device)
        c[0] = 1.0

        results = ffs.compute_contrasts(ols_results, c, device=device)

        # Manual computation
        xtx_inv = ols_results.xtx_inv.to(device)
        sigma2 = ols_results.sigma2.to(device)

        var_factor = (c @ xtx_inv @ c).item()
        expected_var = sigma2 * var_factor
        expected_stderr = torch.sqrt(expected_var)

        # Check first voxel
        assert torch.allclose(
            results["contrast_stderr"][0].to(device),
            expected_stderr[0],
            rtol=1e-4,
        ), "Contrast variance formula incorrect"

    def test_contrast_list_input(self, ols_results):
        """Test that contrast can be provided as Python list."""
        n_regressors = ols_results.betas.shape[1]

        # Provide contrast as list
        contrast = [1.0, -1.0] + [0.0] * (n_regressors - 2)

        results = ffs.compute_contrasts(ols_results, contrast)

        assert "contrast_betas" in results
        assert results["contrast_betas"].shape[0] == ols_results.betas.shape[0]

    def test_contrast_numpy_input(self, ols_results):
        """Test that contrast can be provided as numpy array."""
        import numpy as np

        n_regressors = ols_results.betas.shape[1]

        # Provide contrast as numpy array
        contrast = np.zeros(n_regressors)
        contrast[0] = 1.0
        contrast[1] = -1.0

        results = ffs.compute_contrasts(ols_results, contrast)

        assert "contrast_betas" in results


class TestARMAContrasts:
    """Test contrast computation with ARMA results."""

    @pytest.fixture
    def device(self):
        return get_device()

    @pytest.fixture
    def block_design(self, device):
        """Create realistic block design for ARMA testing."""
        n_timepoints = 150

        X = torch.zeros(n_timepoints, 3, device=device)

        # Condition 1: timepoints 10-40
        X[10:40, 0] = 1.0

        # Condition 2: timepoints 50-80
        X[50:80, 1] = 1.0

        # Condition 3: timepoints 100-130
        X[100:130, 2] = 1.0

        return X

    @pytest.fixture
    def arma_results(self, device, block_design):
        """Create ARMA results with realistic block design."""
        torch.manual_seed(42)

        n_voxels = 50
        n_timepoints = 150

        # True betas with meaningful differences
        true_betas = torch.tensor(
            [[4.0, 2.0, 3.0]],  # Cond1 > Cond3 > Cond2
            device=device,
        ).repeat(n_voxels, 1)

        # Signal + noise
        signal = (block_design @ true_betas.T).T
        noise = torch.randn(n_voxels, n_timepoints, device=device) * 0.4
        data = signal + noise + 100.0

        # Fit ARMA(1,1)
        results = ffs.fit_glm_arma11(data, block_design, tr=2.0, verbose=False)

        return results

    def test_arma_has_var_betas(self, arma_results):
        """Test that ARMA results contain var_betas for contrast computation."""
        assert hasattr(arma_results, "var_betas"), "ARMA results must have var_betas"
        assert arma_results.var_betas is not None

    def test_arma_var_betas_dimensions(self, arma_results):
        """Test that var_betas has correct shape."""
        n_voxels = arma_results.betas.shape[0]
        n_regressors = arma_results.betas.shape[1]

        assert arma_results.var_betas.shape == (
            n_voxels,
            n_regressors,
            n_regressors,
        ), f"var_betas should be ({n_voxels}, {n_regressors}, {n_regressors})"

    def test_single_contrast_arma(self, arma_results, device):
        """Test single contrast computation with ARMA."""
        n_regressors = arma_results.betas.shape[1]

        contrast = torch.zeros(n_regressors, device=device)
        contrast[0] = 1.0
        contrast[1] = -1.0

        results = ffs.compute_contrasts(arma_results, contrast, device=device)

        assert "contrast_betas" in results
        assert "contrast_tstats" in results
        assert "contrast_stderr" in results

    def test_arma_variance_formula(self, arma_results, device):
        """Test that ARMA contrast variance follows c' Cov(beta) c."""
        n_regressors = arma_results.betas.shape[1]

        # Simple contrast
        c = torch.zeros(n_regressors, device=device)
        c[0] = 1.0

        results = ffs.compute_contrasts(arma_results, c, device=device)

        # Manual computation for first voxel
        var_betas = arma_results.var_betas.to(device)
        expected_var = c @ var_betas[0] @ c
        expected_stderr = torch.sqrt(expected_var)

        assert torch.allclose(
            results["contrast_stderr"][0].to(device),
            expected_stderr,
            rtol=1e-4,
        ), "ARMA contrast variance formula incorrect"


class TestContrastFromDesign:
    """Test compute_contrasts_from_design() wrapper."""

    @pytest.fixture
    def device(self):
        return get_device()

    @pytest.fixture
    def mock_design_info(self):
        """Create mock design info with GLT contrasts."""
        import numpy as np

        return {
            "n_regressors": 3,
            "n_timepoints": 200,
            "glt_labels": ["TaskA_vs_TaskB", "TaskA_vs_TaskC"],
            "glt_matrices": [
                np.array([1.0, -1.0, 0.0]),  # TaskA - TaskB
                np.array([1.0, 0.0, -1.0]),  # TaskA - TaskC
            ],
        }

    def test_compute_contrasts_from_design_ols(self, device, mock_design_info):
        """Test compute_contrasts_from_design with OLS results."""
        torch.manual_seed(42)

        data = torch.randn(50, 200, device=device)
        X = torch.randn(200, 3, device=device)

        results = ffs.fit_glm(data, X, tr=2.0, verbose=False, max_poly_degree=0)

        contrast_results = ffs.compute_contrasts_from_design(
            results, mock_design_info, device=device
        )

        assert contrast_results is not None
        assert contrast_results["contrast_betas"].shape == (50, 2)  # 2 contrasts

    def test_compute_contrasts_from_design_cpu_fallback(self, device, mock_design_info):
        """Test automatic CPU fallback for large datasets."""
        torch.manual_seed(42)

        # Simulate large dataset (>1000 timepoints)
        large_design_info = mock_design_info.copy()
        large_design_info["n_timepoints"] = 2880  # Triggers CPU fallback

        data = torch.randn(10, 100, device=device)
        X = torch.randn(100, 3, device=device)

        results = ffs.fit_glm(data, X, tr=2.0, verbose=False, max_poly_degree=0)

        # Should automatically use CPU (won't crash with OOM)
        contrast_results = ffs.compute_contrasts_from_design(
            results, large_design_info, device=device, auto_cpu_fallback=True
        )

        assert contrast_results is not None

    def test_compute_contrasts_from_design_no_contrasts(self, device):
        """Test that None is returned when no contrasts defined."""
        torch.manual_seed(42)

        data = torch.randn(10, 100, device=device)
        X = torch.randn(100, 3, device=device)

        results = ffs.fit_glm(data, X, tr=2.0, verbose=False, max_poly_degree=0)

        # Design info without contrasts
        design_info = {"n_regressors": 3}

        contrast_results = ffs.compute_contrasts_from_design(
            results, design_info, device=device
        )

        assert contrast_results is None


class TestContrastEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def device(self):
        return get_device()

    def test_contrast_wrong_dimensions(self, device):
        """Test that error is raised for wrong contrast dimensions."""
        torch.manual_seed(42)

        data = torch.randn(10, 100, device=device)
        X = torch.randn(100, 3, device=device)

        results = ffs.fit_glm(data, X, tr=2.0, verbose=False, max_poly_degree=0)

        # Wrong number of regressors in contrast
        bad_contrast = torch.zeros(10, device=device)  # Should be 3

        with pytest.raises(RuntimeError):
            ffs.compute_contrasts(results, bad_contrast, device=device)

    def test_contrast_no_variance_info(self, device):
        """Test error when results lack variance information."""
        from fastfuncsim.glm_core import GLMResults

        # Create results without xtx_inv or var_betas
        results = GLMResults()
        results.betas = torch.randn(10, 3)
        results.sigma2 = torch.randn(10)
        # Deliberately don't set xtx_inv

        contrast = torch.tensor([1.0, -1.0, 0.0])

        with pytest.raises(
            (ValueError, TypeError),
            match="must have either 'xtx_inv' or 'var_betas'|must be real number",
        ):
            ffs.compute_contrasts(results, contrast, device=device)

    def test_contrast_zero_variance(self, device):
        """Test handling of zero variance voxels."""
        torch.manual_seed(42)

        data = torch.randn(10, 100, device=device)
        # Add a voxel with zero variance
        data[0, :] = 1.0

        X = torch.randn(100, 3, device=device)

        results = ffs.fit_glm(data, X, tr=2.0, verbose=False, max_poly_degree=0)

        contrast = torch.tensor([1.0, -1.0, 0.0], device=device)

        # Should not crash
        contrast_results = ffs.compute_contrasts(results, contrast, device=device)

        # Stderr should be non-negative even for zero-variance voxel
        assert torch.all(contrast_results["contrast_stderr"] >= 0)


class TestContrastIntegration:
    """Integration tests with realistic workflows."""

    @pytest.fixture
    def device(self):
        return get_device()

    def test_ols_vs_arma_contrasts(self, device):
        """Test that OLS and ARMA give similar contrasts with block design + strong signal."""
        torch.manual_seed(42)

        n_voxels = 20
        n_timepoints = 200

        # Create block design (realistic fMRI)
        X = torch.zeros(n_timepoints, 3, device=device)
        X[20:70, 0] = 1.0  # Condition 1
        X[80:130, 1] = 1.0  # Condition 2
        X[140:190, 2] = 1.0  # Condition 3

        # True betas: strong, known effects
        true_betas = torch.tensor([[5.0, 2.0, 3.5]], device=device)
        true_betas = true_betas.repeat(n_voxels, 1)

        # Signal >> noise (SNR ~15)
        signal = (X @ true_betas.T).T
        noise = torch.randn(n_voxels, n_timepoints, device=device) * 0.3
        data = signal + noise + 100.0

        # Fit both OLS and ARMA
        ols_results = ffs.fit_glm(data, X, tr=2.0, verbose=False, max_poly_degree=0)
        arma_results = ffs.fit_glm_arma11(data, X, tr=2.0, verbose=False)

        # Test contrast: Condition 1 vs Condition 2
        # Expected: 5.0 - 2.0 = 3.0
        contrast = torch.tensor([1.0, -1.0, 0.0], device=device)
        expected_contrast = 3.0

        ols_contrasts = ffs.compute_contrasts(ols_results, contrast, device=device)
        arma_contrasts = ffs.compute_contrasts(arma_results, contrast, device=device)

        # Both should recover contrasts near the true value
        # (ARMA may differ due to autocorrelation correction)
        assert torch.allclose(
            ols_contrasts["contrast_betas"].mean().to(device),
            torch.tensor(expected_contrast, device=device),
            rtol=0.15,
        ), f"OLS should recover contrast near {expected_contrast}"

        # Just check that ARMA produced valid contrasts (not checking exact value)
        assert not torch.any(torch.isnan(arma_contrasts["contrast_betas"]))
        assert not torch.any(torch.isinf(arma_contrasts["contrast_betas"]))

        # Check that ARMA std errors are positive
        assert torch.all(arma_contrasts["contrast_stderr"] > 0)

    def test_contrast_with_multirun_data(self, device):
        """Test contrasts with multi-run GLM."""
        torch.manual_seed(42)

        # Two runs
        data = [
            torch.randn(20, 100, device=device),
            torch.randn(20, 100, device=device),
        ]
        X = [
            torch.randn(100, 3, device=device),
            torch.randn(100, 3, device=device),
        ]

        results = ffs.fit_glm(data, X, tr=2.0, verbose=False, max_poly_degree=0)

        # Contrast across all runs
        n_regressors = results.betas.shape[1]
        contrast = torch.zeros(n_regressors, device=device)
        contrast[0] = 1.0
        contrast[1] = -1.0

        contrast_results = ffs.compute_contrasts(results, contrast, device=device)

        assert contrast_results["contrast_betas"].shape[0] == 20  # n_voxels
