"""Mixing voxel timeseries into column x depth cells and fitting there must give
the same curves as fitting voxels and mixing the curves -- the identity the
two-step route (ffs_deconvolve -> ffs_laminascope -iresp_prefix) relies on.

The fit is a time-axis operator and the depth mixing is a voxel-axis operator, so
they commute exactly -- but only while every unit shares one lambda. The second
test is the tripwire for that: per-voxel lambda breaks the identity, which is why
the single-step path offers a global lambda and never a per-cell one.
"""

import numpy as np
import pytest
import torch

from fastfuncstuff.design.builder import build_per_run_task_designs
from fastfuncstuff.glm.core import construct_polynomial_matrix
from fastfuncstuff.glm.smooth_basis import fit_smooth_basis, penalty_matrix
from fastfuncstuff.laminar import columns as col
from fastfuncstuff.processing.roi_depth import DepthOptions
from fastfuncstuff.surface.volume_depth import resolve_ribbon_surfaces

TR = 2.0
N_KNOTS = 8
K = 6


def _ribbon(tmp_path):
    """An icosphere shell written as white/pial meshes, with its voxel grid."""
    from fastfuncstuff.io.gifti import save_gifti_surface
    from tests.test_surface_voxelize import icosphere

    v, f = icosphere(4)
    r_w, r_p = 15.0, 21.0
    save_gifti_surface(tmp_path / "lh.white.surf.gii", v * r_w, f)
    save_gifti_surface(tmp_path / "lh.pial.surf.gii", v * r_p, f)
    n = 26
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    aff[:3, 3] = -(n - 1)
    ijk = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1)
    xyz = ijk @ aff[:3, :3].T + aff[:3, 3]
    surfaces = resolve_ribbon_surfaces(
        None, None, None, [str(tmp_path / "lh.white.surf.gii")],
        [str(tmp_path / "lh.pial.surf.gii")], None, hemi="lh", log=None,
    )  # fmt: skip
    return surfaces, aff, (n, n, n), xyz, np.linalg.norm(xyz, axis=-1), (r_w, r_p)


def _setup(tmp_path, n_runs=2, n_tp=40, seed=0, depth_noise=0):
    """Columns, depth reads, a two-condition TENT design with per-run drift, and
    synthetic voxel timeseries on the voxels the reads touch."""
    surfaces, aff, shape3, xyz, r, (r_w, r_p) = _ribbon(tmp_path)
    resp = [np.asarray((surfaces[0].white[:, 2] > 0.0 * r_w), bool)]
    columns = col.make_columns(surfaces, resp, radius_mm=8.0, spacing_mm=6.0, seed=0)
    reads = col.vertex_depth_reads(
        surfaces, columns, aff, shape3, [K], DepthOptions(),
        mask=None, psf=False, device=torch.device("cpu"),
    )  # fmt: skip

    rng = np.random.default_rng(seed)
    onsets = [[np.array([6.0, 30.0, 54.0]) for _ in range(n_runs)],
              [np.array([18.0, 42.0, 66.0]) for _ in range(n_runs)]]  # fmt: skip
    task = build_per_run_task_designs(
        onsets, [n_tp] * n_runs, TR, basis="TENT",
        fir_window_s=[(0.0, (N_KNOTS - 1) * TR)] * 2, device=torch.device("cpu"),
    )  # fmt: skip
    n_task = task.per_run[0].shape[1]

    # Block-diagonal per-run polynomial drift ([[Block-diagonal nuisance]]).
    polys = [
        construct_polynomial_matrix(n_tp, 2, device=torch.device("cpu")) for _ in range(n_runs)
    ]
    n_nuis = polys[0].shape[1]
    total = n_tp * n_runs
    design = torch.zeros((total, n_task + n_nuis * n_runs), dtype=torch.float32)
    for i in range(n_runs):
        sl = slice(i * n_tp, (i + 1) * n_tp)
        design[sl, :n_task] = task.per_run[i]
        design[sl, n_task + i * n_nuis : n_task + (i + 1) * n_nuis] = polys[i]

    # Voxel signal: a depth-graded HRF-ish response, drift, and noise.
    depth = np.clip((r.reshape(-1)[reads.voxels] - r_w) / (r_p - r_w), 0, 1)
    t = np.arange(N_KNOTS) * TR
    hrf = (t / 5.0) ** 2 * np.exp(-(t - 5.0) / 1.5) * (t > 0)
    hrf = hrf / hrf.max()
    beta = np.concatenate([
        np.outer(0.5 + depth, hrf), np.outer((0.5 + depth) * 1.3, hrf),
    ], axis=1)  # fmt: skip
    n_v = reads.voxels.size
    coef = np.zeros((n_v, design.shape[1]), np.float64)
    coef[:, :n_task] = beta
    coef[:, n_task:] = 0.3 * rng.standard_normal((n_v, n_nuis * n_runs))
    # The noise level is a depth gradient, which is what makes a per-unit lambda a
    # depth gradient too: depth_noise=-1 puts the noise at the white end (the real
    # EPI case, deep grey darker), +1 at the pial end.
    grad = depth if depth_noise > 0 else (1.0 - depth)
    sd = (0.05 + 0.45 * grad)[:, None] if depth_noise else 0.05
    data = torch.as_tensor(
        coef @ design.numpy().T.astype(np.float64) + sd * rng.standard_normal((n_v, total)),
        dtype=torch.float32,
    )
    penalty = penalty_matrix("diff2", [N_KNOTS, N_KNOTS], [TR, TR])
    return columns, reads, design, n_task, penalty, data


def _curves_from_betas(betas: np.ndarray) -> np.ndarray:
    """(units, 2 * N_KNOTS) -> (units, N_KNOTS, 2), the trailing shape column_curves
    mixes over (time x condition), as the iresp route hands it."""
    return betas.reshape(betas.shape[0], 2, N_KNOTS).transpose(0, 2, 1)


def _fit(data, design, n_task, penalty, **kw):
    return (
        fit_smooth_basis(data, design, n_task, penalty, device=torch.device("cpu"), **kw)
        .betas.cpu()
        .numpy()
        .astype(np.float64)
    )


def test_mix_then_fit_equals_fit_then_mix_at_one_lambda(tmp_path):
    columns, reads, design, n_task, penalty, data = _setup(tmp_path)
    lam = 3.0

    # Route A, today: fit every voxel, then mix the curves into column x depth.
    vox_betas = _fit(data, design, n_task, penalty, method="fixed", lam=lam)
    route_a = col.column_curves(reads, columns, _curves_from_betas(vox_betas), K).values

    # Route B, single step: mix the timeseries into cells, then fit each cell.
    cells = col.column_curves(reads, columns, data.numpy().astype(np.float64), K)
    n_c = cells.values.shape[0]
    flat = cells.values.reshape(n_c * K, -1)
    ok = np.isfinite(flat).all(1)
    cell_betas = np.full((n_c * K, n_task), np.nan)
    cell_betas[ok] = _fit(
        torch.as_tensor(flat[ok], dtype=torch.float32), design, n_task, penalty,
        method="fixed", lam=lam,
    )  # fmt: skip
    route_b = _curves_from_betas(cell_betas).reshape(n_c, K, N_KNOTS, 2)

    good = np.isfinite(route_a) & np.isfinite(route_b)
    assert good.sum() > 100, "the synthetic ribbon gave almost no complete cells"
    # One ULP on a compiled/chunked path, as the no-bit-equality rule asks.
    np.testing.assert_allclose(route_b[good], route_a[good], rtol=1e-5, atol=1e-5)


def _depth_profile(curves: np.ndarray) -> np.ndarray:
    """Peak of the baseline condition per depth, averaged over columns."""
    peak = np.nanmax(curves[..., 0], axis=2)  # (C, K)
    return np.nanmean(peak, axis=0)


def test_per_voxel_lambda_lets_the_noise_gradient_set_the_depth_profile(tmp_path):
    """Why the single-step path takes ONE lambda for every cell.

    The signal is identical in both halves of this test; only which END of the
    ribbon is noisy flips. One lambda is a single linear operator, so it returns
    the same (depth-blurred) profile either way. A per-unit lambda shrinks the
    noisy end harder, so the measured pial/deep ratio moves in OPPOSITE directions
    with the noise gradient -- the profile is reporting the noise, not the
    physiology. Note the blurred ratio is the honest target: the depth PSF, not the
    smoother, is what accounts for mixing across depth bins.
    """
    ratios = {}
    for sign in (-1, 1):
        columns, reads, design, n_task, penalty, data = _setup(tmp_path, depth_noise=sign, seed=1)

        def route_a(dat=data, des=design, nt=n_task, pen=penalty, cols=columns, rd=reads, **kw):
            betas = _fit(dat, des, nt, pen, **kw)
            return col.column_curves(rd, cols, _curves_from_betas(betas), K).values

        a_fixed = route_a(method="fixed", lam=3.0)
        a_reml = route_a(method="reml")

        # One lambda still commutes with the mixing, noise gradient or not.
        cells = col.column_curves(reads, columns, data.numpy().astype(np.float64), K)
        n_c = cells.values.shape[0]
        flat = cells.values.reshape(n_c * K, -1)
        ok = np.isfinite(flat).all(1)
        out = np.full((n_c * K, n_task), np.nan)
        out[ok] = _fit(torch.as_tensor(flat[ok], dtype=torch.float32), design, n_task,
                       penalty, method="fixed", lam=3.0)  # fmt: skip
        b_fixed = _curves_from_betas(out).reshape(n_c, K, N_KNOTS, 2)
        good = np.isfinite(a_fixed) & np.isfinite(b_fixed)
        assert good.sum() > 100
        assert np.abs(b_fixed[good] - a_fixed[good]).max() < 1e-5 * max(
            np.abs(a_fixed[good]).max(), 1.0
        )

        p_fixed, p_reml = _depth_profile(a_fixed), _depth_profile(a_reml)
        assert np.all(np.isfinite(p_fixed)) and np.all(np.isfinite(p_reml))
        ratios[sign] = (p_fixed[-1] / p_fixed[0], p_reml[-1] / p_reml[0])

    (fix_lo, reml_lo), (fix_hi, reml_hi) = ratios[-1], ratios[1]
    # One lambda: the same signal gives the same profile, whichever end is noisy.
    assert fix_lo == pytest.approx(fix_hi, rel=0.05), (
        f"one lambda should not care where the noise is: {fix_lo:.3f} vs {fix_hi:.3f}"
    )
    # Per-voxel lambda: the two noise gradients straddle it, noisy-deep above and
    # noisy-pial below, and the spread dwarfs one lambda's. The move is not
    # symmetric -- shrinking the small deep amplitude has more leverage on the
    # ratio than shrinking the large pial one (measured 3.05 / 2.33 vs 2.42).
    assert reml_lo > fix_lo and reml_hi < fix_hi, (
        f"per-voxel lambda should track the noise gradient; pial/deep came out "
        f"noisy-deep {reml_lo:.3f}, noisy-pial {reml_hi:.3f}, one-lambda ~{fix_lo:.3f}"
    )
    spread_fixed = abs(fix_lo - fix_hi) / fix_lo
    spread_reml = abs(reml_lo - reml_hi) / reml_lo
    assert spread_reml > 10 * spread_fixed and spread_reml > 0.15, (
        f"the noise gradient moved the per-voxel-lambda profile by {spread_reml:.1%} "
        f"and the one-lambda profile by {spread_fixed:.1%}"
    )
