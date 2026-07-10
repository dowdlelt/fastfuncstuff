"""Tests for voxel-wise (-dsort / ANATICOR) regressors in the ARMA(1,1) GLM.

The -dsort feature gives every voxel its own extra baseline regressor(s). AFNI
estimates the ARMA (a,b) WITHOUT them, then redoes the final GLS per voxel WITH
them. These tests verify, on synthetic data:
- the known per-voxel regressor coefficient and task betas are recovered,
- degrees of freedom drop by exactly n_dsort,
- the constant-timeseries guard fires and is safe,
- the -dsort_nods snapshot matches a plain (no-dsort) fit.
"""

import warnings

import numpy as np
import pytest
import torch

from fastfuncstuff.glm.arma import _dsort_constant_guard, fit_glm_arma11
from fastfuncstuff.glm.core import fit_glm


def test_precomputed_zero_ab_equals_ols():
    """(a, b) pinned to (0, 0) is white noise: the GLS must reduce to exact OLS.

    This also pins the non-batched precomputed-params GLS path, which used to read
    a non-existent ``L_inv`` cache key (KeyError). It is how OLS + -dsort runs.
    """
    device = torch.device("cpu")
    rng = np.random.default_rng(3)
    n_time, n_vox = 120, 40
    X = _make_design(n_time)
    beta = rng.standard_normal((n_vox, X.shape[1])).astype(np.float32) * 2.0
    data = beta @ X.T + 0.1 * rng.standard_normal((n_vox, n_time)).astype(np.float32)

    zero_ab = torch.zeros(n_vox, 2, dtype=torch.float64)
    arma = fit_glm_arma11(
        torch.from_numpy(data),
        torch.from_numpy(X),
        tr=2.0,
        device=device,
        verbose=False,
        use_double=True,
        precomputed_arma_params=zero_ab,
    )
    ols = fit_glm(
        torch.from_numpy(data),
        torch.from_numpy(X),
        tr=2.0,
        device=device,
        verbose=False,
        use_double=True,
        max_poly_degree=-1,
    )
    assert np.allclose(arma.betas.numpy(), ols.betas.numpy(), atol=1e-6, rtol=1e-4)
    assert arma.dof == n_time - X.shape[1]


def _make_design(n_time):
    t = np.arange(n_time)
    X = np.stack(
        [
            np.ones(n_time),
            np.sin(2 * np.pi * t / 40.0),
            np.cos(2 * np.pi * t / 27.0),
            (t / n_time) - 0.5,  # linear drift
        ],
        axis=1,
    ).astype(np.float32)
    return X


@pytest.fixture
def synthetic(rng=np.random.default_rng(0)):
    device = torch.device("cpu")
    n_time, n_vox = 200, 250
    X = _make_design(n_time)
    p = X.shape[1]
    q = 2
    dsort = rng.standard_normal((n_vox, q, n_time)).astype(np.float32)
    beta = rng.standard_normal((n_vox, p)).astype(np.float32) * 2.0
    gamma = rng.standard_normal((n_vox, q)).astype(np.float32) * 3.0
    noise = 0.05 * rng.standard_normal((n_vox, n_time)).astype(np.float32)
    data = beta @ X.T + np.einsum("vq,vqt->vt", gamma, dsort) + noise
    return dict(
        device=device,
        X=X,
        p=p,
        q=q,
        n_time=n_time,
        n_vox=n_vox,
        dsort=dsort,
        beta=beta,
        gamma=gamma,
        data=data,
    )


def test_recovers_task_and_voxelwise_coefficients(synthetic):
    s = synthetic
    res = fit_glm_arma11(
        torch.from_numpy(s["data"]),
        torch.from_numpy(s["X"]),
        tr=2.0,
        device=s["device"],
        verbose=False,
        use_double=True,
        dsort=torch.from_numpy(s["dsort"]),
    )
    # Task betas recovered (all base columns, no task_indices filter).
    assert np.abs(res.betas.numpy() - s["beta"]).max() < 0.05
    # Per-voxel dsort coefficients recovered.
    assert res.dsort_betas is not None
    assert res.dsort_betas.shape == (s["n_vox"], s["q"])
    assert np.abs(res.dsort_betas.numpy() - s["gamma"]).max() < 0.05
    assert res.dsort_labels == ["dsort#0", "dsort#1"]


def test_degrees_of_freedom_drop_by_n_dsort(synthetic):
    s = synthetic
    res = fit_glm_arma11(
        torch.from_numpy(s["data"]),
        torch.from_numpy(s["X"]),
        tr=2.0,
        device=s["device"],
        verbose=False,
        dsort=torch.from_numpy(s["dsort"]),
    )
    assert res.dof == s["n_time"] - s["p"] - s["q"]


def test_task_indices_filter_shapes(synthetic):
    s = synthetic
    res = fit_glm_arma11(
        torch.from_numpy(s["data"]),
        torch.from_numpy(s["X"]),
        tr=2.0,
        device=s["device"],
        verbose=False,
        use_double=True,
        dsort=torch.from_numpy(s["dsort"]),
        task_indices=[1, 2],
    )
    # Only the two task columns are stored, dsort kept separate.
    assert res.betas.shape == (s["n_vox"], 2)
    assert res.dsort_betas.shape == (s["n_vox"], s["q"])
    assert np.abs(res.betas.numpy() - s["beta"][:, [1, 2]]).max() < 0.05


def test_constant_guard_fires_and_is_safe(synthetic):
    s = synthetic
    dsort = s["dsort"].copy()
    dsort[:4, 0, :] = 5.0  # force first dsort dataset constant for 4 voxels
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = fit_glm_arma11(
            torch.from_numpy(s["data"]),
            torch.from_numpy(s["X"]),
            tr=2.0,
            device=s["device"],
            verbose=False,
            use_double=True,
            dsort=torch.from_numpy(dsort),
        )
    assert any("constant" in str(x.message) for x in w)
    # The 4 guarded voxels had col-0 replaced by the dataset mean, so their fit is
    # intentionally mis-specified; recovery is only guaranteed for the rest.
    assert np.abs(res.dsort_betas.numpy()[4:, 1] - s["gamma"][4:, 1]).max() < 0.05
    assert np.isfinite(res.dsort_betas.numpy()).all()


def test_dsort_constant_guard_replaces_with_mean():
    dsort = torch.randn(10, 1, 50)
    dsort[3, 0, :] = 2.0  # constant
    out = _dsort_constant_guard(dsort.clone())
    # Replaced row must no longer be constant (it is the dataset mean over voxels).
    assert out[3, 0].std() > 1e-6
    # Non-degenerate rows untouched.
    assert torch.allclose(out[0], dsort[0])


def test_nods_snapshot_matches_plain_fit(synthetic):
    s = synthetic
    common = dict(
        tr=2.0,
        device=s["device"],
        verbose=False,
        use_double=True,
    )
    res = fit_glm_arma11(
        torch.from_numpy(s["data"]),
        torch.from_numpy(s["X"]),
        dsort=torch.from_numpy(s["dsort"]),
        want_dsort_nods=True,
        **common,
    )
    plain = fit_glm_arma11(
        torch.from_numpy(s["data"]),
        torch.from_numpy(s["X"]),
        **common,
    )
    assert res.nods_results is not None
    assert res.nods_results.dof == s["n_time"] - s["p"]
    # The no-dsort snapshot is the base-design fit: it must match a plain run.
    assert np.allclose(res.nods_results.betas.numpy(), plain.betas.numpy(), atol=1e-4, rtol=1e-3)
    # And it must differ from the dsort fit (which models the extra signal).
    assert np.abs(res.betas.numpy() - res.nods_results.betas.numpy()).max() > 0.1
