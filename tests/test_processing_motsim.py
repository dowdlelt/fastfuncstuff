"""Tests for processing/motsim.py — motion simulation regressors."""

import numpy as np
import pytest
import torch

from fastfuncstuff.processing.motsim import (
    MotSimSpec,
    build_motsim_mask,
    expand_mask_both,
    extract_pcs,
    load_dfile,
    load_motion_1d,
    motsim_regressors,
    params_to_voxel_matrices,
    parse_motsim_spec,
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


# ── build_motsim_mask ──


class TestBuildMotsimMask:
    def _phantom(self):
        vol = torch.zeros(16, 20, 20, device=DEV)
        vol[5:11, 6:14, 6:14] = 100.0
        return vol

    def test_basic_mask(self):
        mask = build_motsim_mask(self._phantom(), dilate=0)
        assert mask.dtype == torch.bool
        assert mask[8, 10, 10]

    def test_dilation_expands(self):
        vol = self._phantom()
        tight = build_motsim_mask(vol, dilate=0)
        loose = build_motsim_mask(vol, dilate=2)
        assert loose.sum() > tight.sum()

    def test_supplied_mask_is_dilated_too(self):
        """The paper's mask is brain dilated 2 voxels; a caller-supplied mask is
        not exempt, or -mask silently discards the edge the method lives on."""
        supplied = torch.zeros(12, 12, 12, dtype=torch.bool, device=DEV)
        supplied[5:7, 5:7, 5:7] = True
        out = build_motsim_mask(torch.zeros(12, 12, 12), mask=supplied, dilate=2)
        assert out.sum() > supplied.sum()
        assert out[supplied].all()

    def test_supplied_mask_undilated_is_passthrough(self):
        supplied = torch.zeros(10, 10, 10, dtype=torch.bool, device=DEV)
        supplied[3:7, 3:7, 3:7] = True
        out = build_motsim_mask(torch.zeros(10, 10, 10), mask=supplied, dilate=0)
        assert (out == supplied).all()

    def test_output_shape(self):
        vol = self._phantom()
        assert build_motsim_mask(vol).shape == vol.shape


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


# ── parse_motsim_spec ──


class TestParseMotsimSpec:
    def test_default_is_the_papers_headline_model(self):
        assert parse_motsim_spec(None) == MotSimSpec("both", 12)
        assert parse_motsim_spec("") == MotSimSpec("both", 12)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("both,12", MotSimSpec("both", 12)),
            ("both,24", MotSimSpec("both", 24)),
            ("forward,12", MotSimSpec("forward", 12)),
            ("backward,12", MotSimSpec("backward", 12)),
            ("forward", MotSimSpec("forward", 12)),
            ("BOTH , 6 ", MotSimSpec("both", 6)),
        ],
    )
    def test_counts(self, text, expected):
        assert parse_motsim_spec(text) == expected

    def test_fraction_is_kept_as_float(self):
        """An int means 'this many'; a float in (0,1) means 'this much variance'.
        The two must not collapse, or -motsim both,0.95 silently asks for 0 PCs."""
        spec = parse_motsim_spec("both,0.95")
        assert spec.n_pcs == 0.95
        assert isinstance(spec.n_pcs, float)
        assert isinstance(parse_motsim_spec("both,12").n_pcs, int)

    @pytest.mark.parametrize(
        "text", ["sideways,12", "both,12,3", "both,0", "both,-4", "both,1.5", "both,nope"]
    )
    def test_rejects_nonsense(self, text):
        with pytest.raises(ValueError):
            parse_motsim_spec(text)

    def test_roundtrips_through_str(self):
        for text in ("both,12", "forward,24", "backward,0.95"):
            assert str(parse_motsim_spec(text)) == text

    def test_needs_backward(self):
        assert not parse_motsim_spec("forward").needs_backward
        assert parse_motsim_spec("backward").needs_backward
        assert parse_motsim_spec("both").needs_backward


# ── the construction itself ──


def _phantom(nz=24, ny=32, nx=32):
    """A head-ish blob with off-centre structure.

    The asymmetric inclusions are not decoration: a smooth ellipsoid is nearly
    rotationally degenerate, so a registration run against one recovers
    translations fine and rotations not at all. Any test that asserts on
    recovered rotation needs something for the cost to grip.
    """
    zz, yy, xx = torch.meshgrid(torch.arange(nz), torch.arange(ny), torch.arange(nx), indexing="ij")

    def blob(cz, cy, cx, sz, sy, sx):
        return torch.exp(-(((zz - cz) / sz) ** 2 + ((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))

    vol = 100.0 * blob(nz / 2, ny / 2, nx / 2, 8, 11, 11)
    vol += 60.0 * blob(nz / 2 + 3, ny / 2 - 6, nx / 2 + 4, 3, 3, 3)
    vol += 45.0 * blob(nz / 2 - 4, ny / 2 + 7, nx / 2 + 2, 2.5, 4, 2)
    vol -= 35.0 * blob(nz / 2 + 1, ny / 2 + 2, nx / 2 - 7, 3, 2, 3)
    return vol.float()


def _affine():
    a = np.diag([3.0, 3.0, 3.0, 1.0])
    a[:3, 3] = [-45.0, -48.0, -36.0]
    return a


class TestForwardSimDirection:
    def test_forward_sim_roundtrip(self):
        """The load-bearing claim: the simulated series must carry the motion we
        fed in. Motion-correcting it has to recover the same parameters — an
        inverted sign here produces plausible-looking regressors of the wrong
        motion, which no shape assertion would catch."""
        from fastfuncstuff.processing.ffs_moco import MocoConfig, moco

        affine = _affine()
        params = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -2.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 2.0, 0.0],
            ]
        )
        matrices = params_to_voxel_matrices(params, affine)
        sim = run_forward_sim(_phantom(), matrices, DEV, interp="wsinc5", verb=0)

        cfg = MocoConfig(
            base_index=0, max_iter=23, compile=False, device="cpu", verb=0, use_shear=True
        )
        recovered = moco(sim, cfg, header_info={"affine": affine}).params

        np.testing.assert_allclose(recovered, params, atol=0.1)

    def test_translation_moves_the_right_way(self):
        """A pure +x DICOM translation must shift the simulated volume, not the
        opposite direction and not nothing."""
        affine = _affine()
        params = np.zeros((2, 6))
        params[1, 0] = 6.0  # 2 voxels of dx
        matrices = params_to_voxel_matrices(params, affine)
        ref = _phantom()
        sim = run_forward_sim(ref, matrices, DEV, interp="linear", verb=0)

        torch.testing.assert_close(sim[0], ref, atol=1e-3, rtol=1e-3)
        # Shifted by 2 voxels along the fastest axis, away from the FoV edges.
        torch.testing.assert_close(
            sim[1][4:20, 4:28, 4:26], ref[4:20, 4:28, 6:28], atol=0.5, rtol=0
        )


class TestMotsimRegressors:
    def _run(self, spec_text, **kw):
        affine = _affine()
        rng = np.random.default_rng(0)
        nt = 14
        params = np.cumsum(rng.normal(0, 0.3, (nt, 6)), axis=0)
        params[0] = 0.0
        matrices = params_to_voxel_matrices(params, affine)
        return (
            motsim_regressors(
                _phantom(),
                matrices,
                parse_motsim_spec(spec_text),
                DEV,
                interp="cubic",
                header_info={"affine": affine},
                verb=0,
                **kw,
            ),
            nt,
        )

    @pytest.mark.parametrize("variant", ["forward", "backward", "both"])
    def test_every_variant_produces_regressors(self, variant):
        result, nt = self._run(f"{variant},4")
        assert result.pcs.shape == (nt, 4)
        assert result.var_explained.shape == (4,)
        assert result.spec.variant == variant

    def test_pcs_are_demeaned_and_unit_variance(self):
        """A nuisance regressor with a DC offset is collinear with the polynomial
        baseline it sits next to."""
        result, _ = self._run("forward,4")
        assert result.pcs.mean(0).abs().max() < 1e-5
        torch.testing.assert_close(result.pcs.std(0), torch.ones(4), atol=1e-4, rtol=1e-4)

    def test_fraction_picks_fewer_components_than_a_tighter_one(self):
        loose, nt = self._run("forward,0.5")
        tight, _ = self._run("forward,0.99")
        assert 0 < loose.pcs.shape[1] <= tight.pcs.shape[1] < nt
        assert float(loose.var_explained.sum()) >= 0.5

    def test_both_concatenates_spatially(self):
        result, _ = self._run("both,3", keep_sims=True)
        assert result.forward is not None and result.backward is not None
        assert result.forward.shape == result.backward.shape
        # The mask the PCA ran in covers one volume; 'both' doubles it along z.
        assert result.mask.shape == result.forward.shape[1:]

    def test_forward_variant_skips_the_second_registration(self):
        result, _ = self._run("forward,3", keep_sims=True)
        assert result.backward is None

    def test_sims_are_dropped_unless_asked_for(self):
        result, _ = self._run("both,3")
        assert result.forward is None and result.backward is None

    def test_backward_differs_from_forward(self):
        """MotSimReg is the residual of the correction, not a copy of MotSim. If
        the backward pass returned the input unchanged the variants would be
        indistinguishable and 'both' would just duplicate itself."""
        result, _ = self._run("both,3", keep_sims=True)
        assert (result.forward - result.backward).abs().mean() > 1e-3
