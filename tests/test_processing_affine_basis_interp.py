import numpy as np
import torch

from fastfuncstuff.processing.affine import (
    apply_affine,
    apply_affine_batched,
    dicom_matrix_to_voxel,
    identity_params,
    load_matrix_1D,
    matrix_to_params,
    params_to_matrix,
    params_to_matrix_batched,
    save_matrix_1D,
    voxel_matrix_to_dicom,
)
from fastfuncstuff.processing.basis import (
    HermiteCubic,
    HermiteQuintic,
    build_3d_basis_cubic,
    build_3d_basis_quintic,
    compute_basis_coords,
    compute_half_widths_cubic,
    evaluate_patch_warp,
    evaluate_patch_warp_batched,
)
from fastfuncstuff.processing.interp import (
    cubic_resample_3d,
    quintic_resample_3d,
    trilinear_interpolate,
    warp_image_linear,
)

DEV = torch.device("cpu")


class TestAffine:
    def test_identity_params(self):
        p = identity_params(device=DEV)
        assert p.shape == (12,)
        assert torch.allclose(p[:3], torch.zeros(3))
        assert torch.allclose(p[3:6], torch.zeros(3))
        assert torch.allclose(p[6:9], torch.ones(3))
        assert torch.allclose(p[9:12], torch.zeros(3))

    def test_params_to_matrix_roundtrip(self):
        p = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        M = params_to_matrix(p)
        p2 = matrix_to_params(M)
        assert torch.allclose(p, p2, atol=1e-4)

    def test_decompose_affine_sdu_roundtrip(self):
        """decompose_affine_sdu is the exact inverse of params_to_matrix."""
        from fastfuncstuff.processing.affine import decompose_affine_sdu

        # translation + rotation (Z,X,Y) + anisotropic scale + shear
        p = torch.tensor(
            [-0.5, -22.5, -58.0, 3.88, 0.32, -0.50, 1.105, 1.076, 1.060, -0.002, -0.0035, -0.023]
        )
        M = params_to_matrix(p)
        p2 = decompose_affine_sdu(M)
        torch.testing.assert_close(p, p2, atol=2e-3, rtol=0.0)

    def test_format_final_fit_params_block(self):
        from fastfuncstuff.processing.affine import format_final_fit_params

        p = torch.tensor(
            [
                -0.479,
                -22.484,
                -57.972,
                3.8835,
                0.3227,
                -0.5023,
                1.1047,
                1.0760,
                1.0602,
                -0.0019,
                -0.0035,
                -0.0233,
            ]
        )
        s = format_final_fit_params(p)
        assert "x-shift=  -0.4790" in s
        assert "enorm=" in s and "62.1" in s  # sqrt(0.48^2+22.48^2+57.97^2)
        assert "z-angle=   3.8835" in s
        assert "base smaller than source" in s  # vol3D > 1

    def test_params_to_matrix_batched(self):
        B = 4
        params = torch.zeros(B, 12)
        params[:, :3] = torch.randn(B, 3)
        params[:, 3:6] = torch.randn(B, 3) * 5.0
        # Anisotropic scale + nonzero shear: must match the single path here too
        # (a regression guard — scaling D@U by column instead of row diverged only
        # for non-identity scale, which the old scale==1 test never exercised).
        params[:, 6:9] = 1.0 + torch.randn(B, 3) * 0.05
        params[:, 9:12] = torch.randn(B, 3) * 0.03
        M_batch = params_to_matrix_batched(params)
        for i in range(B):
            M_single = params_to_matrix(params[i])
            assert torch.allclose(M_batch[i], M_single, atol=1e-5)

    def test_apply_affine_identity(self):
        vol = torch.randn(5, 5, 5, device=DEV)
        M = torch.eye(4, device=DEV)
        out = apply_affine(vol, M)
        assert torch.allclose(out, vol, atol=1e-5)

    def test_apply_affine_translation(self):
        vol = torch.zeros(5, 5, 5, device=DEV)
        vol[2, 2, 2] = 1.0
        M = torch.eye(4, device=DEV)
        M[0, 3] = 1.0
        out = apply_affine(vol, M)
        assert out[2, 2, 1].item() > 0.5

    def test_apply_affine_batched_consistency(self):
        vol = torch.randn(5, 5, 5, device=DEV)
        M = torch.eye(4, device=DEV)
        M[0, 3] = 0.5
        M[1, 3] = -0.3
        out_single = apply_affine(vol, M)
        out_batch = apply_affine_batched(vol, M.unsqueeze(0))
        assert torch.allclose(out_single, out_batch[0], atol=1e-5)

    def test_voxel_dicom_roundtrip(self):
        M_ijk = torch.eye(4)
        base_aff = np.eye(4)
        base_aff[0, 0] = 2.0
        src_aff = np.eye(4)
        src_aff[1, 1] = 3.0
        M_dicom = voxel_matrix_to_dicom(M_ijk, base_aff, src_aff)
        M_back = dicom_matrix_to_voxel(M_dicom, base_aff, src_aff)
        assert torch.allclose(M_back, M_ijk, atol=1e-4)

    def test_dicom_uses_cardinal_frame_for_oblique(self):
        # AFNI's ijk_to_dicom is the CARDINAL (axis-snapped) affine. The DICOM
        # conversion must therefore ignore a source/base's obliquity: an oblique
        # affine and its deobliqued twin must produce the *same* DICOM matrix.
        # (Regression for the cross-modal .aff12.1D ↔ ffs_nwarp frame mismatch.)
        from fastfuncstuff.processing.nwarpforge import compute_cardinal_affine

        theta = np.deg2rad(9.3)
        c, s = np.cos(theta), np.sin(theta)
        oblique = np.eye(4)
        oblique[:3, :3] = np.array([[3.0, 0, 0], [0, 3.0 * c, -3.0 * s], [0, 3.0 * s, 3.0 * c]])
        oblique[:3, 3] = [10.0, -5.0, 2.0]
        cardinal = compute_cardinal_affine(oblique)

        M_ijk = torch.eye(4)
        M_ijk[0, 3] = 1.5
        d_obl = voxel_matrix_to_dicom(M_ijk, oblique, oblique)
        d_car = voxel_matrix_to_dicom(M_ijk, cardinal, cardinal)
        assert torch.allclose(d_obl, d_car, atol=1e-4)
        # and the inverse recovers M_ijk from the oblique-tagged DICOM matrix
        M_back = dicom_matrix_to_voxel(d_obl, oblique, oblique)
        assert torch.allclose(M_back, M_ijk, atol=1e-4)

    def test_save_load_matrix_1D(self, tmp_path):
        M_ijk = torch.eye(4)
        M_ijk[0, 3] = 2.5
        base_aff = np.eye(4)
        src_aff = np.eye(4)
        p = tmp_path / "test.aff12.1D"
        save_matrix_1D(M_ijk, p, base_aff, src_aff)
        M_loaded = load_matrix_1D(p, base_aff, src_aff)
        assert torch.allclose(M_loaded, M_ijk, atol=1e-4)


class TestBasis:
    def test_hermite_cubic_at_zero(self):
        x = torch.tensor([0.0])
        b0, b1 = HermiteCubic.eval_1d(x)
        assert torch.allclose(b0, torch.tensor([1.0]), atol=1e-6)
        assert torch.allclose(b1, torch.tensor([0.0]), atol=1e-6)

    def test_hermite_cubic_sum_at_boundary(self):
        x = torch.tensor([1.0, -1.0])
        b0, b1 = HermiteCubic.eval_1d(x)
        assert torch.allclose(b0, torch.zeros(2), atol=1e-6)
        assert torch.allclose(b1, torch.zeros(2), atol=1e-6)

    def test_hermite_quintic_at_zero(self):
        x = torch.tensor([0.0])
        b0, b1, b2 = HermiteQuintic.eval_1d(x)
        assert torch.allclose(b0, torch.tensor([1.0]), atol=1e-6)
        assert torch.allclose(b1, torch.tensor([0.0]), atol=1e-6)
        assert torch.allclose(b2, torch.tensor([0.0]), atol=1e-6)

    def test_compute_basis_coords_range(self):
        coords = compute_basis_coords(8, DEV)
        assert coords.min() >= -1.0 - 1e-5
        assert coords.max() <= 1.0 + 1e-5

    def test_build_3d_basis_cubic_lite(self):
        nx, ny, nz = 4, 4, 4
        basis = build_3d_basis_cubic(nx, ny, nz, DEV, lite=True)
        assert basis.shape == (4, nx * ny * nz)

    def test_build_3d_basis_cubic_full(self):
        nx, ny, nz = 4, 4, 4
        basis = build_3d_basis_cubic(nx, ny, nz, DEV, lite=False)
        assert basis.shape == (8, nx * ny * nz)

    def test_build_3d_basis_quintic_lite(self):
        nx, ny, nz = 4, 4, 4
        basis = build_3d_basis_quintic(nx, ny, nz, DEV, lite=True)
        assert basis.shape == (10, nx * ny * nz)

    def test_evaluate_patch_warp_zero_params(self):
        nx, ny, nz = 4, 4, 4
        basis = build_3d_basis_cubic(nx, ny, nz, DEV, lite=True)
        n_basis = basis.shape[0]
        params = torch.zeros(3 * n_basis, device=DEV)
        hw = compute_half_widths_cubic(nx, ny, nz)
        xd, yd, zd = evaluate_patch_warp(basis, params, hw)
        assert torch.allclose(xd, torch.zeros(nx * ny * nz))
        assert torch.allclose(yd, torch.zeros(nx * ny * nz))
        assert torch.allclose(zd, torch.zeros(nx * ny * nz))

    def test_evaluate_patch_warp_batched_zero(self):
        nx, ny, nz = 4, 4, 4
        B = 3
        basis = build_3d_basis_cubic(nx, ny, nz, DEV, lite=True)
        n_basis = basis.shape[0]
        params = torch.zeros(B, 3 * n_basis, device=DEV)
        hw = compute_half_widths_cubic(nx, ny, nz)
        xd, yd, zd = evaluate_patch_warp_batched(basis, params, hw)
        V = nx * ny * nz
        assert xd.shape == (B, V)
        assert torch.allclose(xd, torch.zeros(B, V))
        assert torch.allclose(yd, torch.zeros(B, V))
        assert torch.allclose(zd, torch.zeros(B, V))


class TestInterp:
    def test_trilinear_at_integer_coords(self):
        vol = torch.randn(5, 5, 5, device=DEV)
        x = torch.tensor([0.0, 2.0, 4.0], device=DEV)
        y = torch.tensor([1.0, 3.0, 0.0], device=DEV)
        z = torch.tensor([0.0, 1.0, 2.0], device=DEV)
        out = trilinear_interpolate(vol, x, y, z)
        expected = torch.tensor([vol[0, 1, 0], vol[1, 3, 2], vol[2, 0, 4]], device=DEV)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_trilinear_midpoint(self):
        vol = torch.zeros(2, 2, 2, device=DEV)
        vol[0, 0, 0] = 1.0
        vol[1, 1, 1] = 1.0
        vol[0, 1, 1] = 1.0
        vol[1, 0, 0] = 1.0
        x = torch.tensor([0.5], device=DEV)
        y = torch.tensor([0.5], device=DEV)
        z = torch.tensor([0.5], device=DEV)
        out = trilinear_interpolate(vol, x, y, z)
        assert torch.allclose(out, torch.tensor([0.5]), atol=1e-5)

    def test_warp_image_linear_zero_displacement(self):
        vol = torch.randn(5, 5, 5, device=DEV)
        zd = torch.zeros(5, 5, 5, device=DEV)
        yd = torch.zeros(5, 5, 5, device=DEV)
        xd = torch.zeros(5, 5, 5, device=DEV)
        out = warp_image_linear(vol, xd, yd, zd)
        assert torch.allclose(out, vol, atol=1e-5)

    def test_cubic_resample_at_integers(self):
        vol = torch.randn(8, 8, 8, device=DEV)
        c = torch.arange(2, 6, dtype=torch.float32, device=DEV)
        z_c = c[:, None, None].expand(4, 4, 4)
        y_c = c[None, :, None].expand(4, 4, 4)
        x_c = c[None, None, :].expand(4, 4, 4)
        out = cubic_resample_3d(vol, x_c, y_c, z_c)
        expected = vol[2:6, 2:6, 2:6]
        assert torch.allclose(out, expected, atol=1e-4)

    def test_quintic_resample_at_integers(self):
        vol = torch.randn(8, 8, 8, device=DEV)
        c = torch.arange(2, 6, dtype=torch.float32, device=DEV)
        z_c = c[:, None, None].expand(4, 4, 4)
        y_c = c[None, :, None].expand(4, 4, 4)
        x_c = c[None, None, :].expand(4, 4, 4)
        out = quintic_resample_3d(vol, x_c, y_c, z_c)
        expected = vol[2:6, 2:6, 2:6]
        assert torch.allclose(out, expected, atol=1e-4)


class TestSourceBatchedCompose:
    """The source-batched (multi-volume) compose primitive must equal N
    sequential single-volume calls to the byte — the Stage-0 gate for batching
    qwarp over many volumes that share one base. See [[Outstanding issues]]."""

    def test_multi_equals_sequential(self):
        from fastfuncstuff.processing.interp import (
            batched_compose_and_interpolate,
            batched_compose_and_interpolate_multi,
        )

        torch.manual_seed(0)
        N, nz, ny, nx = 5, 16, 18, 20
        ph = pw = pd = 5
        kk, jj, ii = torch.meshgrid(
            torch.arange(pd), torch.arange(pw), torch.arange(ph), indexing="ij"
        )
        ii_p, jj_p, kk_p = (t.reshape(-1).float() for t in (ii, jj, kk))
        V = ii_p.numel()
        P = 6
        ibots = torch.randint(0, nx - ph, (P,)).float()
        jbots = torch.randint(0, ny - pw, (P,)).float()
        kbots = torch.randint(0, nz - pd, (P,)).float()
        base_i = ibots[:, None] + ii_p[None, :]
        base_j = jbots[:, None] + jj_p[None, :]
        base_k = kbots[:, None] + kk_p[None, :]
        source = torch.randn(N, nz, ny, nx)
        gw = torch.randn(N, 3, nz, ny, nx) * 0.5
        pxd, pyd, pzd = (torch.randn(N, P, V) * 0.3 for _ in range(3))

        seq = [[] for _ in range(4)]
        for n in range(N):
            outs = batched_compose_and_interpolate(
                source[n], gw[n, 0], gw[n, 1], gw[n, 2], pxd[n], pyd[n], pzd[n],
                ii_p, jj_p, kk_p, ibots, jbots, kbots, nx, ny, nz,
                global_warp_3ch=gw[n], base_i=base_i, base_j=base_j, base_k=base_k,
            )
            for s, o in zip(seq, outs, strict=True):
                s.append(o)
        seq = [torch.stack(s) for s in seq]

        multi = batched_compose_and_interpolate_multi(
            source, gw, pxd, pyd, pzd, base_i, base_j, base_k, nx, ny, nz
        )
        for a, b in zip(seq, multi, strict=True):
            assert b.shape == (N, P, V)
            assert torch.allclose(a, b, atol=1e-5)
