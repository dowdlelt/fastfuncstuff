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
