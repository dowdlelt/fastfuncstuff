"""Tests for processing/motsim.py — motion simulation regressors."""

import numpy as np
import pytest
import torch

from fastfuncstuff.processing.motsim import (
    automask_dilate,
    expand_mask_both,
    extract_pcs,
    load_dfile,
    load_motion_1d,
    params_to_voxel_matrices,
    run_forward_sim,
    save_1d,
)

DEV = torch.device("cpu")


# ── load_motion_1d ──


class TestLoadMotion1D:
    def test_basic_load(self, tmp_path):
        path = tmp_path / "motion.1D"
        path.write_text("# comment line\n0.1 0.2 0.3 0.4 0.5 0.6\n0.0 0.0 0.0 0.0 0.0 0.0\n")
        params = load_motion_1d(str(path))
        assert params.shape == (2, 6)
        assert params.dtype == np.float64

    def test_skips_comments_and_blanks(self, tmp_path):
        path = tmp_path / "motion.1D"
        path.write_text(
            "# header\n\n0.1 0.2 0.3 0.4 0.5 0.6\n# another comment\n0.0 0.0 0.0 0.0 0.0 0.0\n"
        )
        params = load_motion_1d(str(path))
        assert params.shape == (2, 6)

    def test_bad_columns_raises(self, tmp_path):
        path = tmp_path / "bad.1D"
        path.write_text("0.1 0.2 0.3\n")
        with pytest.raises(ValueError, match="Expected 6 columns"):
            load_motion_1d(str(path))

    def test_mapping_order(self, tmp_path):
        """Verify AFNI→DICOM parameter mapping: roll→-rz, pitch→rx, yaw→ry,
        dS→-dz, dL→dx, dP→dy."""
        path = tmp_path / "motion.1D"
        # roll pitch yaw dS dL dP
        path.write_text("1.0 2.0 3.0 4.0 5.0 6.0\n")
        params = load_motion_1d(str(path))
        # Expected DICOM: [dL, dP, -dS, -roll, pitch, yaw]
        #               = [5.0, 6.0, -4.0, -1.0, 2.0, 3.0]
        np.testing.assert_allclose(params[0], [5.0, 6.0, -4.0, -1.0, 2.0, 3.0])


# ── load_dfile ──


class TestLoadDfile:
    def test_basic_load(self, tmp_path):
        path = tmp_path / "dfile.1D"
        path.write_text("0 0.1 0.2 0.3 0.4 0.5 0.6 1.0 0.9\n1 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.4\n")
        params = load_dfile(str(path))
        assert params.shape == (2, 6)

    def test_bad_columns_raises(self, tmp_path):
        path = tmp_path / "bad_dfile.1D"
        path.write_text("0 0.1 0.2\n")
        with pytest.raises(ValueError, match="Expected >= 7 columns"):
            load_dfile(str(path))

    def test_mapping_matches_motion_1d(self, tmp_path):
        """Same motion params should produce identical DICOM output."""
        mot_path = tmp_path / "motion.1D"
        mot_path.write_text("1.0 2.0 3.0 4.0 5.0 6.0\n")
        df_path = tmp_path / "dfile.1D"
        df_path.write_text("0 1.0 2.0 3.0 4.0 5.0 6.0 0.0 0.0\n")

        from_1d = load_motion_1d(str(mot_path))
        from_df = load_dfile(str(df_path))
        np.testing.assert_allclose(from_1d, from_df)


# ── params_to_voxel_matrices ──


class TestParamsToVoxelMatrices:
    def test_identity_for_zero_params(self):
        params = np.zeros((2, 6), dtype=np.float64)
        affine = np.eye(4)
        matrices = params_to_voxel_matrices(params, affine)
        assert matrices.shape == (2, 4, 4)
        # Zero params → identity matrix
        for t in range(2):
            torch.testing.assert_close(
                matrices[t],
                torch.eye(4),
                atol=1e-5,
                rtol=1e-5,
            )

    def test_output_shape(self):
        nt = 5
        params = np.zeros((nt, 6), dtype=np.float64)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        matrices = params_to_voxel_matrices(params, affine)
        assert matrices.shape == (nt, 4, 4)


# ── automask_dilate ──


class TestAutomaskDilate:
    def test_basic_mask(self):
        vol = torch.zeros(10, 12, 14, device=DEV)
        vol[3:7, 4:8, 5:9] = 100.0
        mask = automask_dilate(vol, dilate_voxels=0)
        assert mask.dtype == torch.bool
        # Interior should be masked in
        assert mask[5, 6, 7]

    def test_dilation_expands(self):
        vol = torch.zeros(12, 12, 12, device=DEV)
        vol[5:7, 5:7, 5:7] = 100.0
        mask_no_dilate = automask_dilate(vol, dilate_voxels=0)
        mask_dilated = automask_dilate(vol, dilate_voxels=2)
        assert mask_dilated.sum() >= mask_no_dilate.sum()

    def test_all_zeros_returns_all_true(self):
        vol = torch.zeros(6, 6, 6, device=DEV)
        mask = automask_dilate(vol, dilate_voxels=0)
        assert mask.all()

    def test_output_shape(self):
        vol = torch.randn(8, 10, 12, device=DEV).abs()
        mask = automask_dilate(vol)
        assert mask.shape == vol.shape


# ── expand_mask_both ──


class TestExpandMaskBoth:
    def test_doubles_z_dimension(self):
        mask = torch.ones(4, 6, 8, dtype=torch.bool, device=DEV)
        expanded = expand_mask_both(mask)
        assert expanded.shape == (8, 6, 8)

    def test_content_repeated(self):
        torch.manual_seed(0)
        mask = torch.randint(0, 2, (4, 6, 8), dtype=torch.bool, device=DEV)
        expanded = expand_mask_both(mask)
        assert (expanded[:4] == mask).all()
        assert (expanded[4:] == mask).all()


# ── run_forward_sim ──


class TestRunForwardSim:
    def test_identity_matrices_returns_reference(self):
        torch.manual_seed(1)
        ref = torch.randn(8, 10, 12, device=DEV) + 5.0
        nt = 3
        matrices = torch.eye(4).unsqueeze(0).expand(nt, -1, -1)
        sim = run_forward_sim(ref, matrices, DEV, interp="linear", verb=0)
        assert sim.shape == (nt, 8, 10, 12)
        # Identity inverse is identity, so sim should ≈ reference
        for t in range(nt):
            torch.testing.assert_close(sim[t], ref, atol=1e-4, rtol=1e-4)

    def test_output_shape(self):
        ref = torch.randn(6, 8, 10, device=DEV)
        nt = 4
        matrices = torch.eye(4).unsqueeze(0).expand(nt, -1, -1)
        sim = run_forward_sim(ref, matrices, DEV, verb=0)
        assert sim.shape == (nt, 6, 8, 10)


# ── extract_pcs ──


class TestExtractPCs:
    def test_basic_extraction(self):
        torch.manual_seed(2)
        nt, nz, ny, nx = 10, 6, 8, 10
        data = torch.randn(nt, nz, ny, nx, device=DEV)
        mask = torch.ones(nz, ny, nx, dtype=torch.bool, device=DEV)
        n_pcs = 3
        pcs, var_explained = extract_pcs(data, mask, n_pcs, verb=0)
        assert pcs.shape == (nt, n_pcs)
        assert var_explained.shape == (n_pcs,)

    def test_var_explained_sums_to_leq_1(self):
        torch.manual_seed(3)
        nt, nz, ny, nx = 15, 6, 8, 10
        data = torch.randn(nt, nz, ny, nx, device=DEV)
        mask = torch.ones(nz, ny, nx, dtype=torch.bool, device=DEV)
        _, var_explained = extract_pcs(data, mask, 5, verb=0)
        assert var_explained.sum().item() <= 1.0 + 1e-5

    def test_pcs_are_unit_variance(self):
        torch.manual_seed(4)
        nt, nz, ny, nx = 20, 6, 8, 10
        data = torch.randn(nt, nz, ny, nx, device=DEV)
        mask = torch.ones(nz, ny, nx, dtype=torch.bool, device=DEV)
        pcs, _ = extract_pcs(data, mask, 4, verb=0)
        for i in range(4):
            std = pcs[:, i].std().item()
            assert abs(std - 1.0) < 0.15  # approximately unit variance

    def test_n_pcs_clamped(self):
        """n_pcs should be clamped to nt-1."""
        torch.manual_seed(5)
        nt = 5
        data = torch.randn(nt, 4, 4, 4, device=DEV)
        mask = torch.ones(4, 4, 4, dtype=torch.bool, device=DEV)
        pcs, var_explained = extract_pcs(data, mask, 10, verb=0)
        assert pcs.shape[1] == nt - 1  # clamped to 4


# ── save_1d ──


class TestSave1D:
    def test_writes_file(self, tmp_path):
        pcs = torch.randn(10, 3)
        var_explained = torch.tensor([0.3, 0.2, 0.1])
        path = str(tmp_path / "test.1D")
        save_1d(pcs, var_explained, path, variant="both", n_vols=10)

        with open(path) as f:
            lines = f.readlines()
        # Should have 3 comment lines + 10 data lines
        comment_lines = [l for l in lines if l.startswith("#")]
        data_lines = [l for l in lines if not l.startswith("#")]
        assert len(comment_lines) == 3
        assert len(data_lines) == 10

    def test_header_content(self, tmp_path):
        pcs = torch.randn(5, 2)
        var_explained = torch.tensor([0.5, 0.3])
        path = str(tmp_path / "test.1D")
        save_1d(pcs, var_explained, path, variant="forward", n_vols=5)

        with open(path) as f:
            text = f.read()
        assert "MotSim" in text
        assert "forward" in text
        assert "5 volumes" in text
        assert "2 PCs" in text
