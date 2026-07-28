"""Ljung-Box whiteness statistic — AFNI ``3dREMLfit -Rvar[5]`` parity.

The golden values below were produced by compiling AFNI's own
``thd_ljungbox.c:ljung_box_uneven`` and feeding it the arrays these tests
rebuild from a fixed seed. They cover both the plain-index and the ``tau``
(censoring / run-gap) branches, the auto-``h`` fallback, and the two degenerate
returns.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.glm.arma import (
    ARMA11Results,
    _ljung_box_batched,
    build_ljung_box_tau,
    compute_ljung_box_statistic,
    fit_glm_arma11,
    ljung_box_max_lag,
    save_arma_rvar,
)

# Reference output of AFNI ljung_box_uneven() on the arrays built by _series().
GOLDEN = {
    "white_h20": 11.787321878531632,
    "white_auto": 10.844804374321278,
    "ar1_h20": 289.69836269029236,
    "offset_h20": 4263.635175204613,
    "censored_h18": 17.07203899833587,
    "tworun_h20": 17.01169408355821,
    "tworun_notau_h20": 13.931389818506556,
}


def _series():
    """Rebuild the exact arrays the golden values were computed from."""
    rng = np.random.default_rng(1234)
    white = rng.standard_normal(240)
    n = 240
    eps = rng.standard_normal(n)
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.7 * ar[t - 1] + eps[t]
    offset = rng.standard_normal(240) + 5.0
    keep = np.sort(rng.choice(240, 200, replace=False))
    censored = rng.standard_normal(200)
    tworun_tau = np.concatenate([np.arange(120), 66666 + np.arange(120)])
    tworun = rng.standard_normal(240)
    return {
        "white": white,
        "ar1": ar,
        "offset": offset,
        "censored": (censored, keep),
        "tworun": (tworun, tworun_tau),
    }


def _lb(val, max_lag=None, tau=None):
    t = torch.from_numpy(np.asarray(val, dtype=np.float64)).reshape(1, -1)
    tt = torch.from_numpy(np.asarray(tau, dtype=np.int64)) if tau is not None else None
    return _ljung_box_batched(t, max_lag, tt).item()


class TestAfniParity:
    """Each case must match AFNI to double-precision rounding."""

    def test_plain_index_path(self):
        s = _series()
        assert _lb(s["white"], 20) == pytest.approx(GOLDEN["white_h20"], rel=1e-12)
        assert _lb(s["ar1"], 20) == pytest.approx(GOLDEN["ar1_h20"], rel=1e-12)

    def test_auto_max_lag_matches_afni_fallback(self):
        """max_lag=None hits ljung_box_uneven's own 2+min(n/8, 3·ln n) branch."""
        assert _lb(_series()["white"], None) == pytest.approx(GOLDEN["white_auto"], rel=1e-12)

    def test_out_of_range_lag_is_re_derived(self):
        """h > n/2 is replaced by the same fallback, exactly as in the C."""
        white = _series()["white"]
        assert _lb(white, 10_000) == pytest.approx(GOLDEN["white_auto"], rel=1e-12)

    def test_residuals_are_not_centred(self):
        """c_k is a raw lagged product over Σr², not a Pearson correlation.

        Bug of record: the previous implementation z-scored each voxel and used
        np.corrcoef, which silently removed the mean. A non-zero-mean residual
        makes the two disagree by two orders of magnitude — this case is the
        regression guard.
        """
        s = _series()
        assert _lb(s["offset"], 20) == pytest.approx(GOLDEN["offset_h20"], rel=1e-12)
        # And it is emphatically not the centred answer.
        assert _lb(s["offset"], 20) > 100 * _lb(s["offset"] - s["offset"].mean(), 20)

    def test_tau_path_censored(self):
        val, keep = _series()["censored"]
        assert _lb(val, 18, keep) == pytest.approx(GOLDEN["censored_h18"], rel=1e-12)

    def test_tau_path_run_separation(self):
        """Cross-run pairs must drop out, so tau changes the answer."""
        val, tau = _series()["tworun"]
        assert _lb(val, 20, tau) == pytest.approx(GOLDEN["tworun_h20"], rel=1e-12)
        assert _lb(val, 20) == pytest.approx(GOLDEN["tworun_notau_h20"], rel=1e-12)
        assert _lb(val, 20, tau) != pytest.approx(_lb(val, 20), rel=1e-6)

    def test_degenerate_returns_zero(self):
        """AFNI's two "could not compute" exits: n < 10 and an all-zero series."""
        assert _lb(np.random.default_rng(0).standard_normal(9), 4) == 0.0
        assert _lb(np.zeros(100), 10) == 0.0


class TestMaxLag:
    def test_matches_afni_formula(self):
        """h = nrega + 2 + min(min_run/8, round(3·ln min_run)), capped at min_run/2."""
        # min_run=200: h1=25, h2=round(3·ln200)=round(15.89)=16 → 10+2+16 = 28
        assert ljung_box_max_lag(200, n_regressors=10, min_run=200) == 28
        # The min_run/2 cap bites when the design is large relative to the run.
        assert ljung_box_max_lag(60, n_regressors=40, min_run=60) == 30
        # Single-run default: min_run falls back to n_time.
        assert ljung_box_max_lag(200, n_regressors=10) == 28

    def test_dof_is_h_minus_two(self):
        h = ljung_box_max_lag(300, n_regressors=5, min_run=300)
        assert h - 2 > 0


class TestBuildTau:
    def test_none_for_uncensored_single_run(self):
        """Lag == index there, so the cheaper no-tau path is exact."""
        assert build_ljung_box_tau(100) is None
        assert build_ljung_box_tau(100, run_starts=[0]) is None

    def test_runs_separated_far_enough_to_never_bin(self):
        tau = build_ljung_box_tau(200, run_starts=[0, 100])
        assert tau is not None
        # Within-run indices restart; the gap across the boundary is enormous.
        assert tau[0].item() == 0 and tau[100].item() == 66666
        assert (tau[100] - tau[99]).item() > 60_000

    def test_censoring_stretches_within_run_lag(self):
        """Survivors flanking a censored TR sit at lag 2, not lag 1."""
        within = torch.tensor([0, 1, 3, 4])  # TR 2 censored
        tau = build_ljung_box_tau(4, run_starts=[0], tau=within)
        assert tau is not None
        assert (tau[2] - tau[1]).item() == 2


class TestBatching:
    def test_batched_matches_per_voxel(self):
        rng = np.random.default_rng(3)
        resid = rng.standard_normal((16, 220))
        batched = compute_ljung_box_statistic(resid, max_lag=20)
        one_at_a_time = np.array([_lb(resid[v], 20) for v in range(16)])
        np.testing.assert_allclose(batched, one_at_a_time, rtol=1e-5)

    def test_scale_invariant(self):
        """LB is a ratio, so the per-voxel float32 conditioning applied to Y in
        fit_glm_arma11 cannot shift it — the in-loop reduction is safe to run
        before the unscale block."""
        rng = np.random.default_rng(4)
        resid = rng.standard_normal((8, 200))
        scales = rng.uniform(0.01, 100.0, size=(8, 1))
        np.testing.assert_allclose(
            compute_ljung_box_statistic(resid, max_lag=15),
            compute_ljung_box_statistic(resid * scales, max_lag=15),
            rtol=1e-5,
        )

    def test_accepts_torch_and_numpy(self):
        rng = np.random.default_rng(5)
        resid = rng.standard_normal((4, 120)).astype(np.float32)
        np.testing.assert_allclose(
            compute_ljung_box_statistic(resid, max_lag=12),
            compute_ljung_box_statistic(torch.from_numpy(resid), max_lag=12),
            rtol=1e-5,
        )


class TestFitIntegration:
    """The statistic must come out of the fit itself, not from retained residuals."""

    @staticmethod
    def _fit(want_residuals: bool):
        rng = np.random.default_rng(11)
        n_time, n_vox = 160, 40
        design = np.column_stack(
            [np.ones(n_time), np.sin(np.arange(n_time) * 0.15), np.cos(np.arange(n_time) * 0.07)]
        )
        # Half the voxels get strongly autocorrelated noise, half get white.
        noise = rng.standard_normal((n_vox, n_time))
        for v in range(n_vox // 2):
            for t in range(1, n_time):
                noise[v, t] = 0.85 * noise[v, t - 1] + 0.5 * noise[v, t]
        data = (design @ np.array([100.0, 2.0, 1.5])) + noise
        return fit_glm_arma11(
            data.astype(np.float32),
            design.astype(np.float32),
            tr=2.0,
            want_ljung_box=True,
            want_residuals=want_residuals,
            verbose=False,
        )

    def test_populated_without_want_residuals(self):
        """The old code zero-filled unless -Rwherr had kept the residuals; AFNI
        computes it for every -Rvar."""
        res = self._fit(want_residuals=False)
        assert res.residuals_whitened is None
        assert res.ljung_box is not None
        assert res.ljung_box.shape == (40,)
        assert (res.ljung_box > 0).all()
        assert res.ljung_box_dof is not None and res.ljung_box_dof > 0

    def test_agrees_with_retained_residual_recompute(self):
        res = self._fit(want_residuals=True)
        direct = compute_ljung_box_statistic(
            res.residuals_whitened,
            max_lag=ljung_box_max_lag(160, n_regressors=3, min_run=160),
        )
        np.testing.assert_allclose(res.ljung_box.numpy(), direct, rtol=1e-4, atol=1e-4)

    def test_not_requested_leaves_it_none(self):
        res = fit_glm_arma11(
            np.random.default_rng(0).standard_normal((5, 120)).astype(np.float32),
            np.ones((120, 1), dtype=np.float32),
            tr=2.0,
            verbose=False,
        )
        assert res.ljung_box is None


class TestRvarOutput:
    def test_writes_six_bricks_with_chisq_tag(self, tmp_path):
        import nibabel as nib

        from fastfuncstuff.io.afni import read_brick_labels, read_brick_stataux

        res = ARMA11Results()
        n_vox = 27
        res.arma_params = torch.rand(n_vox, 2) * 0.5
        res.arma_lambda = torch.rand(n_vox)
        res.sigma2 = torch.rand(n_vox) + 0.5
        res.reml_likelihood = torch.rand(n_vox)
        res.ljung_box = torch.rand(n_vox) * 30.0
        res.ljung_box_dof = 18

        out = save_arma_rvar(res, tmp_path / "rvar.nii.gz", volume_shape=(3, 3, 3))
        img = nib.load(str(out))
        assert img.shape[-1] == 6
        assert read_brick_labels(img)[5] == "LjungBox"
        # AFNI code 6 = fict (chi-squared), one parameter: the DOF.
        assert read_brick_stataux(img)[5] == (6, (18.0,))

    def test_warns_when_nothing_computed(self, tmp_path):
        """A zero brick 5 is a gap in the output, not a default — say so."""
        res = ARMA11Results()
        res.arma_params = torch.rand(8, 2)
        res.arma_lambda = torch.rand(8)
        res.sigma2 = torch.rand(8) + 0.5
        res.reml_likelihood = torch.rand(8)
        with pytest.warns(UserWarning, match="LjungBox"):
            save_arma_rvar(res, tmp_path / "rvar.nii.gz", volume_shape=(2, 2, 2))
