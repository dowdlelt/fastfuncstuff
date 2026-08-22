import numpy as np
import pytest
import torch

from fastfuncstuff.processing.affine import (
    _build_homo_coords,
    identity_params,
    save_matrix_1D,
)
from fastfuncstuff.processing.ffs_moco import (
    MocoConfig,
    MocoResult,
    compute_derivative_images,
    compute_max_displacement,
    gauss_newton_rigid,
    save_maxdisp_1D,
    save_moco_1D,
    save_moco_dfile,
)
from fastfuncstuff.processing.slicetime import (
    load_slice_timing,
    shift_timeseries,
    slicetime_correct,
    temporal_resample,
)
from fastfuncstuff.processing.weight import compute_weight_image

DEV = torch.device("cpu")


class TestMocoConfig:
    def test_moco_config_defaults(self):
        cfg = MocoConfig()
        assert cfg.base_index == 0
        assert cfg.cost == "wls"
        # defaults match 3dvolreg (-maxite 23, -x_thresh 0.01, -rot_thresh 0.02)
        assert cfg.max_iter == 23
        assert cfg.dxy_thresh == pytest.approx(0.01)
        assert cfg.dph_thresh == pytest.approx(0.02)
        assert cfg.chain_init is False  # independent per-volume estimation by default
        assert isinstance(cfg.chain_init, bool)
        assert isinstance(cfg.twopass, bool)
        assert cfg.reweight is False
        assert cfg.reweight_tolerance == pytest.approx(1.1)
        assert cfg.device is None


class TestMocoResult:
    def test_moco_result_dataclass(self):
        nt = 3
        nz, ny, nx = 4, 4, 4
        result = MocoResult(
            aligned=torch.zeros(nt, nz, ny, nx),
            params=np.zeros((nt, 6)),
            matrices_vox=torch.zeros(nt, 4, 4),
            matrices_dicom=np.zeros((nt, 4, 4)),
            max_displacement=np.zeros(nt),
            rms_before=np.zeros(nt),
            rms_after=np.zeros(nt),
            n_iters=np.zeros(nt, dtype=np.int32),
        )
        assert result.aligned.shape == (nt, nz, ny, nx)
        assert result.params.shape == (nt, 6)
        assert result.matrices_vox.shape == (nt, 4, 4)
        assert result.max_displacement.shape == (nt,)


class TestDerivativeImages:
    def _make_gaussian_blob(self, shape=(8, 8, 8)):
        nz, ny, nx = shape
        zz, yy, xx = torch.meshgrid(
            torch.arange(nz, dtype=torch.float32),
            torch.arange(ny, dtype=torch.float32),
            torch.arange(nx, dtype=torch.float32),
            indexing="ij",
        )
        cx, cy, cz = (nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
        return torch.exp(-r2 / (2 * 2.0**2))

    def test_compute_derivative_images_shape(self):
        vol = self._make_gaussian_blob((8, 8, 8))
        derivs = compute_derivative_images(vol, DEV)
        assert derivs.shape == (6, 8 * 8 * 8)

    def test_compute_derivative_images_constant(self):
        vol = torch.ones(8, 8, 8, device=DEV)
        derivs = compute_derivative_images(vol, DEV)
        assert derivs.abs().max().item() < 0.15


class TestMaxDisplacement:
    def test_compute_max_displacement_identity(self):
        mat = torch.eye(4, device=DEV)
        shape = (8, 8, 8)
        voxel_sizes = np.array([1.0, 1.0, 1.0])
        disp = compute_max_displacement(mat, shape, voxel_sizes)
        assert disp == pytest.approx(0.0, abs=1e-6)

    def test_compute_max_displacement_translation(self):
        dx, dy, dz = 2.0, 0.0, 0.0
        mat = torch.eye(4, device=DEV)
        mat[0, 3] = dx
        mat[1, 3] = dy
        mat[2, 3] = dz
        shape = (8, 8, 8)
        voxel_sizes = np.array([2.0, 2.0, 2.0])
        disp = compute_max_displacement(mat, shape, voxel_sizes)
        expected = np.sqrt((dx * 2.0) ** 2 + (dy * 2.0) ** 2 + (dz * 2.0) ** 2)
        assert disp == pytest.approx(expected, abs=1e-4)


class TestSaveFunctions:
    def test_save_moco_1D_roundtrip(self, tmp_path):
        # Input params are DICOM correction-transform [dx, dy, dz, rz, rx, ry].
        # AFNI .1D reports subject motion = negation of correction, so every
        # column in the output file is the negation of the DICOM component.
        params = np.array([[1.0, 2.0, 3.0, 0.5, 0.1, 0.2], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        fpath = str(tmp_path / "moco.1D")
        save_moco_1D(params, fpath)
        data = np.loadtxt(fpath)
        # Column order: roll=-rz, pitch=-rx, yaw=-ry, dS=-dz, dL=-dx, dP=-dy
        assert data.shape == (2, 6)
        assert np.allclose(data[0], [-0.5, -0.1, -0.2, -3.0, -1.0, -2.0], atol=1e-3)
        assert np.allclose(data[1], 0.0, atol=1e-3)

    def test_save_matrix_1D_multivolume_roundtrip(self, tmp_path):
        nt = 3
        mats = np.stack([np.eye(4) for _ in range(nt)])
        fpath = str(tmp_path / "moco.aff12.1D")
        save_matrix_1D(mats, fpath, header="test header")
        data = np.loadtxt(fpath, comments="#")
        assert data.shape == (nt, 12)

    def test_save_moco_dfile(self, tmp_path):
        nt = 4
        params = np.zeros((nt, 6))
        rms_before = np.random.rand(nt)
        rms_after = np.random.rand(nt)
        fpath = str(tmp_path / "dfile")
        save_moco_dfile(params, rms_before, rms_after, fpath)
        data = np.loadtxt(fpath)
        assert data.shape[1] == 9
        assert data.shape[0] == nt

    def test_save_maxdisp_1D(self, tmp_path):
        nt = 5
        maxdisp = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        fpath = str(tmp_path / "maxdisp.1D")
        save_maxdisp_1D(maxdisp, fpath)
        data = np.loadtxt(fpath)
        assert data.shape == (nt,)
        assert np.allclose(data, maxdisp, atol=1e-5)


class TestGaussNewtonRigid:
    def _make_gaussian_blob(self, shape=(8, 8, 8)):
        nz, ny, nx = shape
        zz, yy, xx = torch.meshgrid(
            torch.arange(nz, dtype=torch.float32),
            torch.arange(ny, dtype=torch.float32),
            torch.arange(nx, dtype=torch.float32),
            indexing="ij",
        )
        cx, cy, cz = (nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
        return torch.exp(-r2 / (2 * 2.0**2))

    def test_gauss_newton_rigid_identical(self):
        base = self._make_gaussian_blob((8, 8, 8)).to(DEV)
        source = base.clone()

        weight = compute_weight_image(base).to(DEV)
        derivs = compute_derivative_images(base, DEV)

        base_flat = base.reshape(-1)
        weight_flat = weight.reshape(-1)

        WJ = weight_flat.unsqueeze(0) * derivs
        JtWJ = WJ @ WJ.t()

        init_params = identity_params(device=DEV, dtype=torch.float32)
        coords = _build_homo_coords(base.shape, DEV, torch.float32)
        config = MocoConfig(max_iter=3, device="cpu", verb=0, compile=False)

        params, n_iter = gauss_newton_rigid(
            base_flat,
            source,
            weight_flat,
            WJ,
            JtWJ,
            init_params,
            config,
            coords=coords,
        )

        assert n_iter >= 1
        assert torch.allclose(params[:3], torch.zeros(3, device=DEV), atol=0.1)
        assert torch.allclose(params[3:6], torch.zeros(3, device=DEV), atol=0.5)


class TestLoadSliceTiming:
    def test_load_slice_timing_text_file(self, tmp_path):
        timings = [0.0, 0.5, 1.0, 1.5]
        fpath = tmp_path / "slice_timing.txt"
        fpath.write_text("\n".join(str(t) for t in timings))
        loaded = load_slice_timing(fpath)
        assert len(loaded) == len(timings)
        for a, b in zip(loaded, timings, strict=False):
            assert a == pytest.approx(b)

    def test_load_slice_timing_json(self, tmp_path):
        import json

        timings = [0.0, 0.33, 0.66, 1.0]
        fpath = tmp_path / "slice_timing.json"
        fpath.write_text(json.dumps({"SliceTiming": timings}))
        loaded = load_slice_timing(fpath)
        assert len(loaded) == len(timings)


class TestShiftTimeseries:
    def test_shift_timeseries_zero_shift(self):
        ts = torch.randn(10, 20)
        out = shift_timeseries(ts, 0.0)
        assert torch.allclose(out, ts, atol=1e-6)

    def test_shift_timeseries_shape(self):
        ts = torch.randn(10, 20)
        out = shift_timeseries(ts, 0.3, method="fourier")
        assert out.shape == ts.shape
        out2 = shift_timeseries(ts, 0.3, method="cubic")
        assert out2.shape == ts.shape


class TestSlicetimeCorrect:
    def test_slicetime_correct_uniform_timing(self):
        nt, nz, ny, nx = 4, 5, 4, 4
        vol = torch.randn(nt, nz, ny, nx)
        timing = [0.0] * nz
        out = slicetime_correct(vol, timing, tr=2.0)
        assert torch.allclose(out, vol, atol=1e-4)

    def test_slicetime_correct_shape(self):
        nt, nz, ny, nx = 4, 5, 4, 4
        vol = torch.randn(nt, nz, ny, nx)
        timing = [float(i) * 0.1 for i in range(nz)]
        out = slicetime_correct(vol, timing, tr=2.0)
        assert out.shape == vol.shape


class TestTemporalResample:
    def test_temporal_resample_shape(self):
        nt, nz, ny, nx = 10, 3, 4, 4
        vol = torch.randn(nt, nz, ny, nx)
        out = temporal_resample(vol, tr_old=2.0, tr_new=2.0)
        assert out.shape == vol.shape
