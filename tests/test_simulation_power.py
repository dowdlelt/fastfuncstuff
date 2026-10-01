"""Monte-Carlo design power: calibrated false positives, and power that matches theory."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.simulation.core import simulate_bold
from fastfuncstuff.simulation.power import (
    amplitude_for_power,
    design_quality,
    effect_needed,
    simulate_design_power,
)

CPU = torch.device("cpu")
TR = 1.25
RUNS = [240, 240]


def _design(shift_s: float = 0.0) -> torch.Tensor:
    rng = np.random.default_rng(0)
    on_a = [np.sort(rng.uniform(10, 280, 15)) + shift_s for _ in RUNS]
    on_b = [np.sort(rng.uniform(10, 280, 15)) + shift_s for _ in RUNS]
    return simulate_bold(
        [on_a, on_b], [2.0, 2.0], TR, RUNS, [0, 0], tsnr=50, n_voxels=1, device=CPU
    )["design"]


def _run(noise, amplitudes=(0.5, 1.0), n_reps=3000, **kw):
    return simulate_design_power(
        _design(),
        RUNS,
        TR,
        {"A": [1, 0], "A-B": [1, -1]},
        list(amplitudes),
        noise,
        beta_pattern=[1, 0],
        n_reps=n_reps,
        alpha=0.05,
        device=CPU,
        **kw,
    )


def _rows(res, noise, contrast):
    return {
        r["amplitude"]: r for r in res["table"] if r["noise"] == noise and r["contrast"] == contrast
    }


def test_white_noise_matches_theory():
    res = _run([{"label": "w", "tsnr": 40.0, "phys_fraction": 0.0}])
    for c in ("A", "A-B"):
        rows = _rows(res, "w", c)
        assert rows[0.0]["power"] == pytest.approx(0.05, abs=0.015)  # false-positive rate
        for amp in (0.5, 1.0):
            r = rows[amp]
            assert r["mean_est"] == pytest.approx(r["true_effect"], abs=0.03)
            assert r["sd_est"] == pytest.approx(r["sd_predicted"], rel=0.06)
            assert r["power"] == pytest.approx(r["power_predicted"], abs=0.04)


def test_autocorrelation_inflates_naive_ols_and_the_correction_restores_alpha():
    """The whole reason to carry the ARMA: naive OLS t is far too liberal."""
    res = _run([{"label": "ou", "tsnr": 40.0, "phys_fraction": 0.7, "tau": 6.0}])
    for c in ("A", "A-B"):
        null = _rows(res, "ou", c)[0.0]
        assert null["power_naive"] > 0.15
        assert null["power"] == pytest.approx(0.05, abs=0.025)
        one = _rows(res, "ou", c)[1.0]
        assert one["sd_est"] == pytest.approx(one["sd_predicted"], rel=0.06)
        assert one["power"] == pytest.approx(one["power_predicted"], abs=0.04)


def test_measured_arma_is_accepted_directly():
    res = _run([{"label": "ma", "tsnr": 40.0, "arma": (0.0, 0.4)}], amplitudes=(1.0,))
    null = _rows(res, "ma", "A")[0.0]
    assert null["power"] == pytest.approx(0.05, abs=0.025)


def test_hrf_mismatch_shows_as_bias():
    """Signal generated 2 s late, fitted with nominal onsets: the amplitude is underestimated."""
    res = _run(
        [{"label": "w", "tsnr": 40.0, "phys_fraction": 0.0}],
        amplitudes=(1.0,),
        true_design=_design(shift_s=2.0),
    )
    assert _rows(res, "w", "A")[1.0]["mean_est"] < 0.9


def test_noisier_bins_need_bigger_effects():
    res = _run(
        [
            {"label": "worst", "tsnr": 20.0, "phys_fraction": 0.3, "tau": 4.0},
            {"label": "best", "tsnr": 100.0, "phys_fraction": 0.7, "tau": 6.0},
        ],
        amplitudes=np.linspace(0.1, 3.0, 12),
        n_reps=200,
    )
    need = amplitude_for_power(res, target=0.8)
    assert need[("best", "A")] < need[("worst", "A")]
    assert np.isfinite(need[("best", "A")])


def test_collinear_conditions_are_refused():
    d = _design()
    with pytest.raises(ValueError, match="rank-deficient"):
        simulate_design_power(
            torch.cat([d[:, :1], d[:, :1]], 1),
            RUNS,
            TR,
            {"A": [1, 0]},
            [1.0],
            [{"tsnr": 40.0}],
            device=CPU,
            n_reps=10,
        )


def test_large_effects_do_not_produce_nan_power():
    """scipy's noncentral-t cdf is nan deep in the tail."""
    from fastfuncstuff.simulation.power import _two_tailed_power

    assert _two_tailed_power(3.3, 150, 16.0) == pytest.approx(1.0)
    assert _two_tailed_power(3.3, 150, -40.0) == pytest.approx(1.0)
    assert _two_tailed_power(3.3, 150, 0.0) == pytest.approx(0.0012, abs=2e-4)


@pytest.mark.parametrize("dof", [2, 10, 150])
@pytest.mark.parametrize("alpha", [0.05, 0.001])
@pytest.mark.parametrize("target", [0.5, 0.8, 0.99])
def test_fast_power_root_matches_separate_tail_reference(dof, alpha, target):
    from scipy import stats
    from scipy.optimize import brentq

    from fastfuncstuff.simulation.power import _nc_for_power, _two_tailed_power

    crit = stats.t.ppf(1 - alpha / 2, dof)

    def reference(nc):
        far = stats.nct.cdf(-crit, dof, nc)
        return stats.nct.sf(crit, dof, nc) + (0.0 if np.isnan(far) else far)

    expected = brentq(lambda nc: reference(nc) - target, 0.0, 200.0)
    actual = _nc_for_power(target, crit, dof)
    assert actual == pytest.approx(expected, abs=1e-9)
    values = np.array([-actual, 0.0, actual, 40.0])
    np.testing.assert_allclose(
        _two_tailed_power(crit, dof, values), [reference(abs(n)) for n in values], atol=1e-14
    )


def test_cached_ou_parameters_follow_changed_noise_settings():
    from fastfuncstuff.simulation.noise import ou_to_arma11
    from fastfuncstuff.simulation.power import _noise_arma, _ou_arma_parameters

    _ou_arma_parameters.cache_clear()
    noise = {"tau": 6.0, "phys_fraction": 0.5}
    for tr, tau, fraction in [(1.0, 6.0, 0.5), (1.0, 6.0, 0.5), (0.5, 6.0, 0.5), (0.5, 8.0, 0.8)]:
        noise.update(tau=tau, phys_fraction=fraction)
        assert _noise_arma(noise, tr) == tuple(float(v) for v in ou_to_arma11(tr, tau, fraction))
    assert _ou_arma_parameters.cache_info().hits == 1


def test_batched_ridge_reliability_matches_full_matrix_grid():
    from fastfuncstuff.simulation.core import default_microtime_dt, hrfs_from_spec
    from fastfuncstuff.simulation.experiment import ExperimentSpec, Interval, Unit, realize
    from fastfuncstuff.simulation.power import (
        _noise_correlation,
        _nuisance,
        _trial_reliability,
        single_trial_quality,
        trial_regressors,
    )

    spec = ExperimentSpec(
        tr=1.0,
        units=[Unit.parse("A", "A:0.25", 12)],
        isi=Interval.parse("uniform:2,8"),
        post_fix=16,
    )
    real = realize(spec, 4)
    noise = [{"label": "n", "tsnr": 60.0, "phys_fraction": 0.5, "tau": 6.0}]
    dt = default_microtime_dt(1.0)
    bases = hrfs_from_spec("spmg1", dt, CPU)[0][1]
    X, _ = trial_regressors(real, 1.0, bases, dt)
    Q, _ = torch.linalg.qr(_nuisance(real.run_lengths, 1))
    X = X - Q @ (Q.T @ X)
    U, s, Vh = torch.linalg.svd(X, full_matrices=False)
    n = X.shape[1]
    C = torch.eye(n, dtype=torch.float64) - torch.ones(n, n, dtype=torch.float64) / n
    ones = torch.ones(n, dtype=torch.float64)
    R = _noise_correlation(noise[0], 1.0, real.run_lengths)
    assert R is not None
    sigma2 = (100 / 60) ** 2
    lams = [0.0, *(float(torch.median(s**2)) * np.logspace(-3, 2, 26))]
    grid = []
    for lam in lams:
        P = (Vh.T * (s / (s**2 + lam))) @ U.T
        Z, N = P @ X, sigma2 * P @ R @ P.T
        r = _trial_reliability(Z, N, C, ones, 1.0, 0.25)
        norm = float((Z @ ones).square().sum() + 0.25 * Z.square().sum() + torch.trace(N))
        grid.append((lam, r, norm))
    top = max(r for _, r, _ in grid)
    lam, expected, norm = next(row for row in grid if row[1] >= top - 1e-3)
    actual = single_trial_quality(real, 1.0, noise, poly_degree=1)["reliability"]["n"]
    assert actual["ridge_lambda"] == pytest.approx(lam)
    assert actual["ridge"] == pytest.approx(expected, abs=1e-12)
    assert actual["ridge_frac"] == pytest.approx(np.sqrt(norm / grid[0][2]), abs=1e-12)


def test_analytic_power_accounts_for_hrf_mismatch():
    """With a late true response the analytic curve must follow the biased estimate."""
    res = _run(
        [{"label": "w", "tsnr": 40.0, "phys_fraction": 0.0}],
        amplitudes=(1.0, 2.0),
        true_design=_design(shift_s=3.0),
    )
    for amp in (1.0, 2.0):
        r = _rows(res, "w", "A")[amp]
        assert r["expected_est"] < 0.85 * r["true_effect"]
        assert r["mean_est"] == pytest.approx(r["expected_est"], abs=0.03)
        assert r["power"] == pytest.approx(r["power_predicted"], abs=0.04)


def test_power_table_round_trips_through_tsv(tmp_path):
    from fastfuncstuff.cli.simulate import main
    from fastfuncstuff.simulation.power import compare_designs, load_power_table, scan_seconds

    assert (
        main(
            [
                "-tr",
                "2",
                "-trial",
                "A",
                "2",
                "8",
                "-isi",
                "4",
                "-tsnr",
                "50",
                "-amplitudes",
                "1",
                "3",
                "-ndesigns",
                "2",
                "-nreps",
                "20",
                "-device",
                "cpu",
                "-no_plots",
                "-prefix",
                str(tmp_path / "x"),
            ]
        )
        == 0
    )
    res = load_power_table(tmp_path / "x_power.tsv")
    assert res["name"] == "x" and isinstance(res["table"][0]["power"], float)
    assert {r["design"] for r in res["table"]} == {0, 1}
    assert scan_seconds(res) is not None
    rows = compare_designs({"x": res})
    assert {r["contrast"] for r in rows} == {"A"} and rows[0]["n_realizations"] == 2


class TestDifferenceSweep:
    """A zero-sum contrast sweeps the difference; -shared sets the level underneath."""

    def _reals(self):
        from fastfuncstuff.simulation.experiment import ExperimentSpec, Interval, Unit, realize

        spec = ExperimentSpec(
            tr=2,
            units=[Unit.parse(c, f"{c}:2", 1) for c in "ABC"],
            isi=Interval.parse("exp:4,2,12"),
            scan_time=300,
            n_runs=2,
            post_fix=16,
        )
        return [realize(spec, s) for s in range(2)]

    def _run(self, shared=0.0, true_hrf="same", contrasts=None):
        from fastfuncstuff.simulation.power import simulate_realizations_power

        return simulate_realizations_power(
            self._reals(),
            2,
            contrasts or {"A": [1, 0, 0], "A-B": [1, -1, 0]},
            [0.5, 1.0],
            [{"label": "t", "tsnr": 60.0, "phys_fraction": 0.5, "tau": 6.0}],
            n_reps=100,
            device=CPU,
            progress=False,
            shared=shared,
            true_hrf=true_hrf,
        )

    def test_the_swept_value_is_the_difference(self):
        res = self._run(shared=4.0)
        for r in res["table"]:
            if r["contrast"] == "A-B":
                assert r["swept"] == "difference" and r["shared"] == 4.0
                assert r["true_effect"] == pytest.approx(r["amplitude"])
            else:
                assert r["swept"] == "amplitude" and r["shared"] == 0.0

    def test_a_general_zero_sum_contrast_gets_exactly_the_difference(self):
        res = self._run(shared=2.0, contrasts={"A+B-2C": [1, 1, -2]})
        for r in res["table"]:
            assert r["true_effect"] == pytest.approx(r["amplitude"])

    def test_shared_response_cancels_under_the_right_hrf(self):
        flat, high = self._run(0.0), self._run(4.0)
        for a, b in zip(flat["table"], high["table"], strict=True):
            if a["contrast"] == "A-B":
                assert a["power_predicted"] == pytest.approx(b["power_predicted"], abs=1e-9)
                assert a["expected_est"] == pytest.approx(b["expected_est"], abs=1e-9)

    def test_but_not_under_a_mismatch(self):
        flat, high = self._run(0.0, "lib:3"), self._run(4.0, "lib:3")
        diff = [
            abs(a["expected_est"] - b["expected_est"])
            for a, b in zip(flat["table"], high["table"], strict=True)
            if a["contrast"] == "A-B"
        ]
        assert max(diff) > 0.01


def test_analytic_power_includes_residual_misfit():
    """A mismatched HRF leaves signal in the residuals: t shrinks as the effect grows.

    Without the misfit term the analytic curve promised 80% power where Monte
    Carlo measured ~0 (A-B on a large shared response, wrong HRF).
    """
    res = _run(
        [{"label": "w", "tsnr": 150.0, "phys_fraction": 0.0}],
        amplitudes=(2.0, 4.0),
        true_design=_design(shift_s=3.0),
        n_reps=1500,
    )
    for amp in (2.0, 4.0):
        r = _rows(res, "w", "A")[amp]
        assert r["power"] == pytest.approx(r["power_predicted"], abs=0.06)


def _row(design, noise, amp, power, contrast="A"):
    return {
        "design": design,
        "true_hrf": "",
        "noise": noise,
        "contrast": contrast,
        "amplitude": amp,
        "true_effect": amp,
        "expected_est": amp,
        "power": power,
        "power_predicted": power,
    }


def test_effect_needed_keeps_one_value_per_realization():
    # design 0 reaches 80% between 1 and 2; design 1 never does.
    rows = [_row(0, "lo", a, p) for a, p in ((0, 0.0), (1, 0.6), (2, 1.0))]
    rows += [_row(1, "lo", a, p) for a, p in ((0, 0.0), (1, 0.1), (2, 0.3))]
    need = effect_needed({"table": rows})
    assert list(need) == [("lo", "A")]
    np.testing.assert_allclose(need[("lo", "A")], [1.5, np.nan])


def _two_condition_design(offset_b: float, n=200, tr=2.0):
    from fastfuncstuff.simulation.core import (
        build_task_design,
        default_microtime_dt,
        hrfs_from_spec,
    )

    dt = default_microtime_dt(tr)
    bases = hrfs_from_spec("spmg1", dt, torch.device("cpu"))[0][1]
    a = np.arange(10.0, n * tr - 30, 23.0)
    onsets = [[a], [a + offset_b]]
    return build_task_design(onsets, [2.0, 2.0], tr, [n], bases, dt, device=torch.device("cpu"))


NOISE = [{"label": "t50", "tsnr": 50.0, "phys_fraction": 0.5, "tau": 6.0}]


def test_design_quality_names_the_combination_that_is_not_estimable():
    q = design_quality(_two_condition_design(0.0), [200], 2.0, NOISE)
    assert q["deficient"] and q["rank"] == q["n_columns"] - 1
    w = q["null_weights"]
    np.testing.assert_allclose(np.abs(w), [1.0, 1.0], atol=1e-6)
    assert np.sign(w[0]) != np.sign(w[1])  # A - B, the identical-timing pair


def test_design_quality_matches_the_simulated_effect_needed():
    # The separability matrix is the engine's analytic power solved for 80%:
    # it must agree with what the simulation reports for the same contrasts.
    X = _two_condition_design(4.0)
    q = design_quality(X, [200], 2.0, NOISE)
    assert not q["deficient"] and q["vif"].min() >= 1.0
    res = simulate_design_power(
        X,
        [200],
        2.0,
        {"A": [1, 0], "A-B": [1, -1]},
        np.linspace(0.2, 12, 120),
        NOISE,
        beta_pattern=[1, 0],
        n_reps=10,
        device=torch.device("cpu"),
    )
    need = amplitude_for_power(res, 0.8, column="power_predicted")
    m = q["needed"]["t50"]
    assert m[0, 0] == pytest.approx(need[("t50", "A")], rel=0.01)
    assert m[1, 0] == pytest.approx(need[("t50", "A-B")], rel=0.01)


def test_correlation_sign_decides_whether_a_difference_is_cheap():
    # 4 s apart, A and B co-occur (r > 0): the fit sees that something happened
    # but not which, so A-B costs far more than A. Alternating evenly (r < 0)
    # the difference is nearly as cheap as each condition.
    near = design_quality(_two_condition_design(4.0), [200], 2.0, NOISE)
    apart = design_quality(_two_condition_design(11.5), [200], 2.0, NOISE)
    assert near["corr"][1, 0] > 0 > apart["corr"][1, 0]
    ratio = [q["needed"]["t50"][1, 0] / q["needed"]["t50"][0, 0] for q in (near, apart)]
    assert ratio[0] > 1.4 and ratio[1] < 1.1


def test_corrected_terms_match_the_dense_formula():
    # The trace expansion avoids forming M = I - X P (n_t^3); it must equal it.
    from fastfuncstuff.simulation.power import _corrected_terms, _noise_correlation

    X = torch.cat([_two_condition_design(4.0), torch.ones(200, 1, dtype=torch.float64)], 1)
    X = X.double()
    P = torch.linalg.inv(X.T @ X) @ X.T
    R = _noise_correlation({"tsnr": 50.0, "phys_fraction": 0.6, "tau": 6.0}, 2.0, [120, 80])
    PRPt, tr_MR, dof = _corrected_terms(X, P, R)
    MR = (torch.eye(200, dtype=torch.float64) - X @ P) @ R
    torch.testing.assert_close(PRPt, P @ R @ P.T)
    assert tr_MR == pytest.approx(float(torch.trace(MR)), rel=1e-10)
    assert dof == pytest.approx(float(torch.trace(MR)) ** 2 / float((MR * MR.T).sum()), rel=1e-10)


def test_scan_time_sweep_scales_as_one_over_sqrt_time_and_reports_trimmed_runs():
    from fastfuncstuff.simulation.experiment import ExperimentSpec, Interval, Unit
    from fastfuncstuff.simulation.power import scan_time_sweep

    spec = ExperimentSpec(
        tr=1.0,
        units=[Unit.parse(c, f"{c}:30", 1, "block") for c in ("E1", "E2")],
        isi=Interval.parse(10),
        initial_fix=10,
        post_fix=15,
        order="permuted_block",
        n_runs=2,
    )
    # 15 + 40 n s holds n blocks exactly: 8 and 32, balanced, 4x the blocks
    out = scan_time_sweep(
        spec, [335, 1295], 3, {"E1": [1, 0], "E1-E2": [1, -1]}, NOISE, progress=False
    )
    rows = out["rows"]
    assert not out["skipped"]

    def med(st, c, key):
        return float(
            np.median([r[key] for r in rows if r["scan_time"] == st and r["contrast"] == c])
        )

    # 4x the scan: the effect needed halves, so effect x sqrt(minutes) is ~flat
    for c in ("E1", "E1-E2"):
        assert med(1295, c, "needed") / med(335, c, "needed") == pytest.approx(0.5, abs=0.06)
        assert med(1295, c, "per_minute") / med(335, c, "per_minute") == pytest.approx(1, abs=0.1)
    # minutes are the runs actually scanned (after any trim), not the request
    assert all(r["minutes"] == pytest.approx(2 * r["run_s"] / 60) for r in rows)


def test_estimation_trades_off_against_detection():
    # Liu et al. 2001: rapid jitter recovers the response shape and detects
    # poorly; sparse events the reverse. The design-quality pass must show both.
    from fastfuncstuff.simulation.experiment import ExperimentSpec, Unit, realize
    from fastfuncstuff.simulation.power import realizations_design_quality

    def spec(gap):
        return ExperimentSpec(
            tr=1.0,
            units=[Unit.parse("c", f"E1:0.25:uniform:{gap}", 1, "block")],
            n_runs=2,
            initial_fix=10,
            post_fix=15,
            scan_time=330,
        )

    def score(gap):
        q = realizations_design_quality([realize(spec(gap), s) for s in range(4)], 1.0, NOISE)
        det = np.median([x["needed"]["t50"][0, 0] for x in q])
        return det, np.median([x["shape_sd"]["t50"][0] for x in q]), np.median([x["xi"] for x in q])

    packed, sparse = score("2,5"), score("10,14")
    assert packed[0] > 1.3 * sparse[0]  # detects worse
    assert packed[1] < 0.7 * sparse[1] and packed[2] > 2 * sparse[2]  # estimates better


def test_a_fixed_soa_leaves_the_response_shape_barely_estimable():
    from fastfuncstuff.simulation.experiment import ExperimentSpec, Interval, Unit, realize
    from fastfuncstuff.simulation.power import estimation_quality

    fixed = ExperimentSpec(
        tr=1.0, units=[Unit.parse("A", "A:0.25", 60)], isi=Interval.parse(2.75), post_fix=16
    )
    q = estimation_quality(realize(fixed, 0), 1.0, NOISE)
    assert q["xi"] < 0.05


def test_single_trial_lss_matches_a_brute_force_fit_and_sees_leakage():
    # Vectorized LSS (FWL onto a 2-column solve per trial) against fitting each
    # trial's own model directly; then the design effect it exists to show.
    from fastfuncstuff.simulation.core import default_microtime_dt, hrfs_from_spec
    from fastfuncstuff.simulation.experiment import ExperimentSpec, Interval, Unit, realize
    from fastfuncstuff.simulation.power import (
        _noise_correlation,
        _nuisance,
        single_trial_quality,
        trial_regressors,
    )

    spec = ExperimentSpec(
        tr=1.0,
        units=[Unit.parse("A", "A:1", 12), Unit.parse("B", "B:1", 12)],
        isi=Interval.parse("exp:4,2,10"),
        post_fix=15,
    )
    r = realize(spec, 0)
    dt = default_microtime_dt(1.0)
    bases = hrfs_from_spec("spmg1", dt, torch.device("cpu"))[0][1]
    Xt, cond = trial_regressors(r, 1.0, bases, dt)
    D = _nuisance(r.run_lengths, 1)
    R = _noise_correlation(NOISE[0], 1.0, r.run_lengths)
    S = torch.stack([Xt[:, cond == q].sum(1) for q in range(2)], 1)
    brute = []
    for i, q in enumerate(cond):
        Z = torch.cat([Xt[:, [i]], (S[:, q] - Xt[:, i])[:, None], S[:, [1 - q]], D], 1)
        a = torch.linalg.pinv(Z)[0]
        brute.append(float(torch.sqrt(a @ R @ a)) * 100 / 50)
    q = single_trial_quality(r, 1.0, NOISE, poly_degree=1)
    for k in range(2):
        assert q["lss_sd"]["t50"][k] == pytest.approx(np.median(np.array(brute)[cond == k]))

    def leak(isi):
        s = ExperimentSpec(tr=1.0, units=[Unit.parse("A", "A:0.25", 40)],
                           isi=Interval.parse(isi), post_fix=15)  # fmt: skip
        return single_trial_quality(realize(s, 0), 1.0, NOISE)["leakage"][0]

    assert leak("uniform:2,5") > 3 * leak("uniform:10,14")  # overlap mixes neighbours in


def test_a_conditions_only_trial_is_estimable():
    from fastfuncstuff.simulation.experiment import ExperimentSpec, Interval, Unit, realize
    from fastfuncstuff.simulation.power import single_trial_quality

    spec = ExperimentSpec(
        tr=1.0,
        units=[Unit.parse("A", "A:1", 10), Unit.parse("B", "B:1", 1)],
        isi=Interval.parse(8),
        post_fix=15,
    )
    q = single_trial_quality(realize(spec, 0), 1.0, NOISE)
    assert np.isfinite(q["lss_sd"]["t50"]).all() and np.isfinite(q["leakage"]).all()


def test_trial_pattern_reliability_matches_a_monte_carlo_and_ridge_rescues_lsa():
    # The analytic expected correlation of estimated with true trial deviations,
    # against fitting simulated trials (amplitudes 1 +- 0.5 %, ARMA noise).
    from fastfuncstuff.simulation.core import default_microtime_dt, hrfs_from_spec
    from fastfuncstuff.simulation.experiment import ExperimentSpec, Interval, Unit, realize
    from fastfuncstuff.simulation.power import (
        _noise_correlation,
        _nuisance,
        single_trial_quality,
        trial_regressors,
    )

    spec = ExperimentSpec(
        tr=1.0,
        units=[Unit.parse("A", "A:0.25", 40)],
        isi=Interval.parse("exp:2,1,6"),
        post_fix=15,
    )
    r = realize(spec, 0)
    rel = single_trial_quality(r, 1.0, NOISE, poly_degree=2)["reliability"]["t50"]
    assert rel["ridge"] >= rel["lsa"] and 0 < rel["ridge_frac"] <= 1
    assert rel["ridge"] > 1.5 * rel["lsa"]  # packed trials: LSA's variance explodes

    dt = default_microtime_dt(1.0)
    bases = hrfs_from_spec("spmg1", dt, torch.device("cpu"))[0][1]
    Xt, _ = trial_regressors(r, 1.0, bases, dt)
    Q, _ = torch.linalg.qr(_nuisance(r.run_lengths, 2))
    Xp = Xt - Q @ (Q.T @ Xt)
    L = torch.linalg.cholesky(_noise_correlation(NOISE[0], 1.0, r.run_lengths))
    n = Xt.shape[1]
    H = torch.linalg.solve(
        Xp.T @ Xp + rel["ridge_lambda"] * torch.eye(n, dtype=torch.float64), Xp.T
    )
    gen = torch.Generator().manual_seed(0)
    num = den_e = den_t = 0.0
    for _ in range(300):
        beta = 1.0 + 0.5 * torch.randn(n, generator=gen, dtype=torch.float64)
        y = Xt @ beta + 2.0 * (L @ torch.randn(Xt.shape[0], generator=gen, dtype=torch.float64))
        est = H @ (y - Q @ (Q.T @ y))
        de, dt_ = (est - est.mean()).numpy(), (beta - beta.mean()).numpy()
        num, den_e, den_t = num + de @ dt_, den_e + de @ de, den_t + dt_ @ dt_
    assert rel["ridge"] == pytest.approx(num / np.sqrt(den_e * den_t), abs=0.03)


def _curve(powers, scale=1.0):
    return {
        "table": [
            {"noise": "n", "contrast": "E1", "amplitude": a, "true_effect": scale * a,
             "power": p, "power_predicted": p, "expected_est": scale * a}
            for a, p in zip([0.0, 1.0, 2.0, 3.0, 4.0], powers, strict=True)
        ]
    }  # fmt: skip


def test_effect_needed_is_the_true_effect_at_the_last_crossing():
    # -pattern E1=3: the sweep is a third of E1's response; report E1's response.
    assert amplitude_for_power(_curve([0.0, 0.4, 0.8, 1.0, 1.0], scale=3.0))[("n", "E1")] == (
        pytest.approx(6.0)
    )
    # Non-monotone (a misfit shared response): high at zero, a dip, then a rise.
    # The first crossing read 0; the answer is where power stays above target.
    assert amplitude_for_power(_curve([0.9, 0.3, 0.6, 0.9, 1.0]))[("n", "E1")] == pytest.approx(
        2 + (0.8 - 0.6) / (0.9 - 0.6)
    )
    assert np.isnan(amplitude_for_power(_curve([0.0, 0.1, 0.2, 0.3, 0.4]))[("n", "E1")])


def test_shape_steps_short_events_resolve_shapes_blocks_do_not():
    # The ordered library (peak 2.7 -> 5.7 s): brief jittered events tell nearby
    # shapes apart; a 30 s boxcar smooths every library shape into the same
    # regressor. Shape resolution is what HRF selection (hrfopt, GLMsingle) needs.
    from fastfuncstuff.simulation.experiment import ExperimentSpec, Interval, Unit, realize
    from fastfuncstuff.simulation.power import shape_steps

    noise = [{"label": "t100", "tsnr": 100.0, "phys_fraction": 0.5, "tau": 6.0}]

    def steps(unit, isi):
        spec = ExperimentSpec(tr=1.0, units=[unit], isi=Interval.parse(isi), n_runs=2,
                              initial_fix=10, post_fix=15, scan_time=330)  # fmt: skip
        out = shape_steps(realize(spec, 0), 1.0, noise, [1.0])
        assert out["power"]["t100"].shape == (1, out["max_step"])
        return out["steps"]["t100"][0], out["power"]["t100"][0]

    ev_steps, ev_power = steps(Unit.parse("E", "E:0.25", 1), "exp:3,1,10")
    bl_steps, _ = steps(Unit.parse("E", "E:30", 1, "block"), 30)
    assert np.isfinite(ev_steps) and ev_steps < 5 and np.isnan(bl_steps)
    assert np.all(np.diff(ev_power) >= -1e-9)  # further apart: easier


def test_hrf_robustness_ceiling_for_events_and_blocks_hold_up():
    # Fitting SPMG1 while the truth is the fastest library HRF: brief events keep
    # ~20% of the response and the rest inflates the residuals, so power levels
    # off below 80% at any amplitude. A 20 s boxcar smooths shapes alike.
    from fastfuncstuff.simulation.experiment import ExperimentSpec, Interval, Unit, realize
    from fastfuncstuff.simulation.power import hrf_robustness

    noise = {"label": "t60", "tsnr": 60.0, "phys_fraction": 0.5, "tau": 6.0}

    def run(units, isi):
        spec = ExperimentSpec(tr=1.0, units=units, isi=Interval.parse(isi), n_runs=2,
                              initial_fix=10, post_fix=15, scan_time=330)  # fmt: skip
        return hrf_robustness(realize(spec, 0), 1.0, {"A": [1.0]}, noise)["contrasts"]["A"]

    ev = run([Unit.parse("A", "A:0.25", 1)], "exp:4,1,12")
    bl = run([Unit.parse("A", "A:20", 1, "block")], 20)
    assert np.isinf(ev["needed"][0]) and ev["ceiling"][0] < 0.8 and ev["recovered"][0] < 0.3
    assert np.all(np.isfinite(bl["needed"])) and min(bl["recovered"]) > 0.4
    # at the library shape nearest the fitted one, the cost is close to a right HRF
    assert min(ev["needed"]) == pytest.approx(ev["fitted"], rel=0.15)
    # Fitting the true shape itself: always detectable, never worse than the
    # mismatched fit, and a slow response costs a rapid design more than a fast one
    m, n = np.asarray(ev["matched"]), np.asarray(ev["needed"])
    assert np.all(np.isfinite(m)) and np.all(m <= n * 1.001)
    assert m[-1] > m[0]


def test_sweep_grid_stops_at_the_plateau_and_reaches_past_a_fixed_edge():
    from fastfuncstuff.simulation.power import AutoSweep, sweep_grid

    fine = np.geomspace(1e-3, 100, 400)
    auto = AutoSweep(n_points=10)
    easy = 1 - np.exp(-((fine / 0.3) ** 3))  # saturated by ~0.6
    hard = 1 - np.exp(-((fine / 8.0) ** 3))  # 80% near 9.3, past the old 3% edge
    for curve, lo, hi in ((easy, 0.05, 0.6), (hard, 1.0, 16.0)):
        grid = sweep_grid(fine, curve[:, None], auto)
        assert len(grid) == 10 and lo < grid[-1] < hi
        assert np.interp(grid[-1], fine, curve) >= 0.98
        assert np.interp(grid[0], fine, curve) < 0.15  # a point on the toe
    # A curve that plateaus below the ceiling stops where it stops moving.
    capped = 0.6 * easy
    assert sweep_grid(fine, capped[:, None], auto)[-1] < 1.0


def test_auto_sweep_gives_each_noise_level_its_own_grid():
    from fastfuncstuff.simulation.experiment import ExperimentSpec, Interval, Unit, realize
    from fastfuncstuff.simulation.power import AutoSweep, simulate_realizations_power

    spec = ExperimentSpec(
        tr=1.5, units=[Unit.parse("A", "A:1", 12)], isi=Interval.parse("exp:4,2,10"), post_fix=12
    )
    reals = [realize(spec, s) for s in range(2)]
    noise = [{"label": "lo", "tsnr": 30.0}, {"label": "hi", "tsnr": 120.0}]
    res = simulate_realizations_power(
        reals, 1.5, {"A": [1]}, AutoSweep(8, extra=(1.0,)), noise, n_reps=200,
        device=CPU, progress=False,
    )  # fmt: skip
    grid = {
        n: sorted({r["amplitude"] for r in res["table"] if r["noise"] == n}) for n in ("lo", "hi")
    }
    assert 0.0 in grid["lo"] and 1.0 in grid["lo"] and 1.0 in grid["hi"]
    assert grid["hi"][-1] < grid["lo"][-1]  # cleaner data saturates sooner
    for n in grid:
        top = [r for r in res["table"] if r["noise"] == n and r["amplitude"] == grid[n][-1]]
        assert np.mean([r["power_predicted"] for r in top]) > 0.95


def test_tabulated_noncentrality_matches_the_root_finder():
    from scipy import stats

    from fastfuncstuff.simulation.power import _nc_for_power, _nc_needed

    for alpha, target in ((0.001, 0.8), (0.05, 0.95)):
        for dof in (1.5, 2.0, 7.3, 150.4, 1e5):
            ref = _nc_for_power(target, float(stats.t.ppf(1 - alpha / 2, dof)), dof)
            assert _nc_needed(target, alpha, dof) == pytest.approx(ref, rel=1e-7)


def test_neural_activity_outlasting_the_stimulus_is_a_mismatch():
    from fastfuncstuff.simulation.experiment import ExperimentSpec, Interval, Unit, realize
    from fastfuncstuff.simulation.power import neural_durations, simulate_realizations_power

    assert neural_durations([0.5, 2.0], None) == [0.5, 2.0]
    assert neural_durations([0.5, 2.0], "+3") == [3.5, 5.0]
    assert neural_durations([0.5, 2.0], 4) == [4.0, 4.0]
    with pytest.raises(ValueError):
        neural_durations([1.0], "-2")
    spec = ExperimentSpec(
        tr=1.0, units=[Unit.parse("A", "A:0.5", 40)], isi=Interval.parse("exp:3,1,8"), post_fix=12
    )
    reals = [realize(spec, 0)]
    noise = [{"label": "t60", "tsnr": 60.0}]
    kw = dict(n_reps=200, device=CPU, progress=False)
    same = simulate_realizations_power(reals, 1.0, {"A": [1]}, [1.0], noise, **kw)
    longer = simulate_realizations_power(
        reals, 1.0, {"A": [1]}, [1.0], noise, true_duration="+4", **kw
    )
    pick = lambda res: next(r for r in res["table"] if r["amplitude"] == 1.0)  # noqa: E731
    assert pick(same)["expected_est"] == pytest.approx(1.0, rel=1e-6)
    assert abs(pick(longer)["expected_est"] - 1.0) > 0.02  # the model no longer fits the truth
