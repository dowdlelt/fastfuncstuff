"""ffs_laminar: the notebook's analysis as a tool, refereed by the MATLAB reference.

The inversion oracle holds the reference's two-condition responses, its shipped PSF
kernel and the free energy MATLAB reached for two models. Written out the way the
toolbox writes curves (ffs_deconvolve's tidy tables, depth 1 deepest) and run through
the CLI end to end, the same F must come back -- which pins the depth flip, the block
layout, the input, the reference baselines, the priors and the kernel all at once.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fastfuncstuff.cli import laminar
from tests.test_laminar_io import _write_table

ORACLE = Path(__file__).parent / "laminar_oracle" / "laminar_inversion_oracle.json"


@pytest.fixture
def reference(tmp_path):
    if not ORACLE.exists():  # pragma: no cover
        pytest.skip(f"no oracle at {ORACLE}")
    o = json.loads(ORACLE.read_text())
    y = np.asarray(o["y"], float).reshape(o["ns"], o["K"])
    times = np.arange(7) * float(o["TR"])
    a = _write_table(tmp_path / "k9_iresp_c1.tsv", times, y[1:8][:, ::-1])
    b = _write_table(tmp_path / "k9_iresp_c3.tsv", times, y[29:36][:, ::-1])
    kern = tmp_path / "kernel.1D"
    np.savetxt(kern, np.asarray(o["kernel"], float).reshape(-1))
    return o, a, b, kern


def _table(path):
    rows = [ln.split("\t") for ln in open(path).read().splitlines()]
    return rows[0], rows[1:]


@pytest.mark.slow  # 55s
def test_the_cli_writes_the_five_tables(reference, tmp_path):
    o, a, b, kern = reference
    prefix = str(tmp_path / "quick")
    rc = laminar.main(
        ["-cond1", str(a), "-cond2", str(b), "-duration", "1.6", "-psf_kernel", str(kern),
         "-reference_delay", "-modulation", "reference", "-models", "null", "-max_iter", "2",
         "-prefix", prefix,
         "-workers", "1", "-verb", "0"]
    )  # fmt: skip
    assert rc == 0
    h, rows = _table(f"{prefix}_laminar_models.tsv")
    assert (
        h[:8] == ["roi", "name", "hemi", "model", "F", "p_model", "best", "r2"] and h[8] == "F_K9"
    )
    assert [r[3] for r in rows] == ["null"] and float(rows[0][5]) == pytest.approx(1.0)
    h, rows = _table(f"{prefix}_laminar_depths.tsv")
    assert [r[3] for r in rows] == ["superficial", "middle", "deep"]
    h, rows = _table(f"{prefix}_laminar_devein.tsv")
    assert len(rows) == 3 * 9 and {r[4] for r in rows} == {str(d) for d in range(1, 10)}
    h, rows = _table(f"{prefix}_laminar_neuronal.tsv")
    assert len(rows) == 2 * 7 * 3 and {r[3] for r in rows} == {"1", "2"}
    # The profile table is per condition per VASCULAR depth (K=9), not per neuronal
    # depth: it is the measured curve and the part of it each depth made itself, both
    # in percent, which is what the profile panel draws.
    h, rows = _table(f"{prefix}_laminar_profiles.tsv")
    assert h == ["roi", "name", "hemi", "condition", "depth", "depth_frac",
                 "measured_pct", "deveined_pct", "drained_pct", "fit_pct", "model_local_pct"]  # fmt: skip
    assert len(rows) == 2 * 9 and {r[3] for r in rows} == {"1", "2"}
    assert {r[4] for r in rows} == {str(d) for d in range(1, 10)}
    # depth_frac is 1 at the pial end and 0 at white matter, one row per condition.
    # The NUMERICS live in test_laminar_devein.py: this fit is -max_iter 2 on purpose,
    # so its W is not converged and the deep end keeps only 63% of its signal (97% and
    # 91% on a converged real fit). Asserting the physics here would be asserting that
    # a two-iteration fit is a good one.
    deep = [r for r in rows if r[5] == "0"]
    assert len(deep) == 2 and {r[3] for r in deep} == {"1", "2"}
    # measured = deveined + drained, to the table's own 6 significant figures
    assert all(float(r[6]) == pytest.approx(float(r[7]) + float(r[8]), rel=1e-5) for r in rows)
    doc = json.load(open(f"{prefix}_laminar.json"))
    assert doc["psf"] == "kernel" and doc["K"] == [9] and doc["delay_seconds"] == "TR/2"


def test_bad_inputs_are_reported(reference, tmp_path, capsys):
    o, a, b, kern = reference
    base = ["-cond1", str(a), "-cond2", str(b), "-duration", "1.6", "-prefix", str(tmp_path / "x")]
    assert laminar.main([*base, "-models", "nonsense"]) == 1
    assert "unknown model" in capsys.readouterr().err
    assert laminar.main([*base, "-roi", "nowhere"]) == 1
    assert "no ROI" in capsys.readouterr().err


@pytest.mark.slow
def test_the_pipeline_reaches_matlabs_free_energy_with_its_padding(reference):
    """~1 minute: two full inversions, compared with the reference to 1e-6.

    The reference sets the noise hyperprior from the variance of the WHOLE data vector,
    masked padding included, so its F depends on the padding's random draw. With the
    oracle's own padding rows the pipeline must reproduce MATLAB exactly."""
    import torch

    from fastfuncstuff.laminar.io import pick_roi, read_iresp_table, two_condition_data
    from fastfuncstuff.laminar.pipeline import fit_roi, reference_p0

    o, a, b, kern = reference
    y = np.asarray(o["y"], float).reshape(o["ns"], o["K"])
    data = two_condition_data(
        pick_roi(read_iresp_table(a), "V1"), pick_roi(read_iresp_table(b), "V1"),
        duration=1.6, delay_seconds=None, p0=reference_p0(), modulation="reference",
    )  # fmt: skip
    data.y[8:28] = torch.tensor(y[8:28])
    k = torch.tensor(np.loadtxt(kern))
    fit = fit_roi({9: data}, {9: k}, targets=[(1, 0, 0), (0, 0, 0)], names=["superficial", "null"])
    got = dict(zip(fit.models, fit.F, strict=True))
    for m in o["models"]:
        assert abs(got[m["name"]] - float(m["F"])) / abs(float(m["F"])) < 1e-6


@pytest.mark.slow
def test_the_cli_reaches_matlabs_model_comparison(reference, tmp_path):
    """Through the CLI, with its own padding draw: the absolute F moves by ~1 nat with
    the draw (measured), the comparison by ~0.2 -- so the comparison must match."""
    o, a, b, kern = reference
    prefix = str(tmp_path / "parity")
    rc = laminar.main(
        ["-cond1", str(a), "-cond2", str(b), "-duration", "1.6", "-psf_kernel", str(kern),
         "-reference_delay", "-modulation", "reference", "-models", "superficial", "null",
         "-prefix", prefix,
         "-workers", "1", "-verb", "0"]
    )  # fmt: skip
    assert rc == 0
    _, rows = _table(f"{prefix}_laminar_models.tsv")
    got = {r[3]: float(r[4]) for r in rows}
    want = {m["name"]: float(m["F"]) for m in o["models"]}
    assert abs((got["superficial"] - got["null"]) - (want["superficial"] - want["null"])) < 0.5
    assert all(abs(got[n] - want[n]) < 3.0 for n in want)


def test_a_measured_psf_matrix_is_matched_by_roi_and_k(reference, tmp_path, capsys):
    from fastfuncstuff.processing.roi_depth import write_depth_psf

    o, a, b, kern = reference
    m = np.eye(9) * 0.7 + np.eye(9, k=1) * 0.15 + np.eye(9, k=-1) * 0.15
    psf = write_depth_psf(str(tmp_path / "k9_depth_psf.tsv"), [("V1", "lh", m)], [1])
    prefix = str(tmp_path / "mat")
    base = ["-cond1", str(a), "-cond2", str(b), "-duration", "1.6", "-models", "null",
            "-max_iter", "2", "-prefix", prefix, "-workers", "1", "-verb", "0"]  # fmt: skip
    assert laminar.main([*base, "-psf_matrix", psf]) == 0
    assert json.load(open(f"{prefix}_laminar.json"))["psf"] == "matrix"
    other = write_depth_psf(str(tmp_path / "k7_depth_psf.tsv"), [("V1", "lh", np.eye(7))], [1])
    assert laminar.main([*base, "-psf_matrix", other]) == 1
    assert "K = [9]" in capsys.readouterr().err


def test_padding_is_explicit_and_only_the_reference_prior_feels_it(reference, tmp_path):
    """The padding rows are masked out of the fit, but the reference centres the noise
    prior on the variance of the whole padded series -- so the default F moves with the
    padding's draw and amplitude, and -hyperprior rows removes the dependence entirely."""
    o, a, b, kern = reference

    def run(tag, *extra):
        prefix = str(tmp_path / tag)
        base = ["-cond1", str(a), "-cond2", str(b), "-duration", "1.6", "-models", "null",
                "-max_iter", "3", "-workers", "1", "-verb", "0", "-prefix", prefix]  # fmt: skip
        assert laminar.main([*base, *extra]) == 0
        _, rows = _table(f"{prefix}_laminar_models.tsv")
        return float(rows[0][4]), json.load(open(f"{prefix}_laminar.json"))

    f1, doc = run("s1", "-pad_seed", "1")
    f2, _ = run("s2", "-pad_seed", "2")
    f3, _ = run("s3", "-pad_sd", "0.3")
    assert f1 != f2 and f1 != f3  # the reference prior sees the padding
    assert doc["padding"] == {"rows": 20, "sd": 0.1, "seed": 1, "hyperprior": "reference"}
    r1, doc = run("r1", "-pad_seed", "1", "-hyperprior", "rows")
    r2, _ = run("r2", "-pad_seed", "2", "-hyperprior", "rows")
    r3, _ = run("r3", "-pad_rows", "30", "-pad_sd", "0.3", "-hyperprior", "rows")
    assert r1 == pytest.approx(r2, rel=1e-9) == pytest.approx(r3, rel=1e-9)
    assert doc["padding"]["hyperprior"] == "rows"


def test_fit_prefix_finds_tables_and_psf_and_reads_a_k_column(reference, tmp_path):
    """The output of `ffs_deconvolve -depths 9 ...` (a K column, one PSF file) drives
    ffs_laminar from just the prefix and the two condition names."""
    from fastfuncstuff.processing.roi_depth import write_depth_psf

    o, a, b, kern = reference
    for cond, src in (("c1", a), ("c3", b)):  # re-write the per-condition tables with a K column
        rows = [ln.split("\t") for ln in open(src).read().splitlines()]
        with open(tmp_path / f"fit_iresp_{cond}.tsv", "w") as fh:
            fh.write("\t".join(rows[0][:3] + ["K"] + rows[0][3:]) + "\n")
            for r in rows[1:]:
                fh.write("\t".join(r[:3] + ["9"] + r[3:]) + "\n")
    m = np.eye(9) * 0.8 + np.eye(9, k=1) * 0.1 + np.eye(9, k=-1) * 0.1
    write_depth_psf(str(tmp_path / "fit_depth_psf.tsv"), [("V1", "lh", m)], [1], ks=[9])
    prefix = str(tmp_path / "viaprefix")
    rc = laminar.main(
        ["-fit_prefix", str(tmp_path / "fit"), "-conds", "c1", "c3", "-duration", "1.6",
         "-models", "null", "-max_iter", "2", "-prefix", prefix, "-workers", "1", "-verb", "0"]
    )  # fmt: skip
    assert rc == 0
    doc = json.load(open(f"{prefix}_laminar.json"))
    assert doc["psf"] == "matrix" and doc["K"] == [9]
    assert (
        laminar.main(["-fit_prefix", str(tmp_path / "fit"), "-duration", "1", "-prefix", prefix])
        == 1
    )


@pytest.mark.slow
def test_a_stalled_fit_is_flagged_warned_about_and_rescued_on_request(reference):
    """Bug of record: with the CLI's default padding draw the noise prior's centre lands
    0.005 below the oracle's, the damping collapses after one rejected step, and
    superficial+deep "converges" at F = -624 (MATLAB, on its own padding: -125.7; the model
    it contains, 'deep': -140). The default must still return the reference's answer -- but
    say it stalled and that the nested models are inconsistent; -rescue must recover it."""
    import torch

    from fastfuncstuff.laminar.io import pick_roi, read_iresp_table, two_condition_data
    from fastfuncstuff.laminar.pipeline import fit_roi, reference_p0

    o, a, b, kern = reference
    data = two_condition_data(
        pick_roi(read_iresp_table(a), "V1"), pick_roi(read_iresp_table(b), "V1"),
        duration=1.6, delay_seconds=None, p0=reference_p0(), modulation="reference",
    )  # fmt: skip
    k = torch.tensor(np.loadtxt(kern))
    kw = dict(targets=[(0, 0, 1), (1, 0, 1)], names=["deep", "superficial+deep"])
    plain = fit_roi({9: data}, {9: k}, **kw)
    assert plain.stalled[1, 0] and not plain.stalled[0, 0]
    assert plain.F[1] < -400  # the reference's own answer, unchanged
    assert any("below the model it contains" in w for w in plain.warnings)
    assert any("-rescue" in w for w in plain.warnings)

    fixed = fit_roi({9: data}, {9: k}, rescue=2, **kw)
    assert not fixed.stalled.any() and not fixed.warnings
    assert fixed.F[1] > fixed.F[0] - 5  # a superset is no worse than what it contains
    assert abs(fixed.F[1] - (-125.7)) < 6  # MATLAB's value, on a different padding draw


@pytest.mark.slow  # 56 s: the stalled fit, serial and batched, each with rescue
def test_batched_rescue_matches_serial_rescue(reference):
    """The stall case above, through the batched engine: the same fit stalls, and
    rescue restarts it to the serial rescue's answer."""
    import torch

    from fastfuncstuff.laminar.batched import variational_laplace_batched
    from fastfuncstuff.laminar.experiment import faes_priors
    from fastfuncstuff.laminar.inversion import variational_laplace
    from fastfuncstuff.laminar.io import pick_roi, read_iresp_table, two_condition_data
    from fastfuncstuff.laminar.pipeline import reference_p0

    o, a, b, kern = reference
    data = two_condition_data(
        pick_roi(read_iresp_table(a), "V1"), pick_roi(read_iresp_table(b), "V1"),
        duration=1.6, delay_seconds=None, p0=reference_p0(), modulation="reference",
    )  # fmt: skip
    k = torch.tensor(np.loadtxt(kern))
    pr = faes_priors(data.spec, (1, 0, 1))
    Y = torch.stack([data.y, data.y])  # one stalls; the copy checks rows stay apart
    kw = dict(rows=data.rows, kernel=k, **data.integrate_kwargs)
    plain = variational_laplace_batched(data.spec, pr, data.u, Y, device="cpu", **kw)
    assert bool(plain.stalled.all())
    fixed = variational_laplace_batched(data.spec, pr, data.u, Y, device="cpu", rescue=2, **kw)
    serial = variational_laplace(data.spec, pr, data.u, data.y, rescue=2, **kw)
    assert not bool(fixed.stalled.any()) and not serial.stalled
    for i in range(2):
        assert float(fixed.F[i]) == pytest.approx(float(serial.F), rel=1e-8, abs=1e-6)


@pytest.mark.slow  # 41s: the real fit plus three permutation replicates
def test_a_permutation_null_is_fitted_and_turned_into_p_perm(reference, tmp_path):
    """The null replicates must be refitted through the identical model space, and the
    p-value must be the permutation one -- floored at 1/(1+n), never 0."""
    o, a, b, kern = reference
    y = np.asarray(o["y"], float).reshape(o["ns"], o["K"])
    times = np.arange(7) * float(o["TR"])
    rng = np.random.default_rng(0)
    nulls = []
    for i in range(3):
        # A replicate's "permutation" here is only a relabelling of the same curves plus
        # noise: the point under test is the plumbing and the arithmetic, not the physics.
        pair = y[1:8][:, ::-1], y[29:36][:, ::-1]
        if i % 2:
            pair = pair[1], pair[0]
        p1 = _write_table(tmp_path / f"n{i}_iresp_c1.tsv", times,
                          pair[0] + 0.01 * rng.standard_normal(pair[0].shape))  # fmt: skip
        p2 = _write_table(tmp_path / f"n{i}_iresp_c3.tsv", times,
                          pair[1] + 0.01 * rng.standard_normal(pair[1].shape))  # fmt: skip
        nulls.append((str(p1), str(p2)))
    prefix = str(tmp_path / "withnull")
    rc = laminar.main(
        ["-cond1", str(a), "-cond2", str(b), "-duration", "1.6", "-psf_kernel", str(kern),
         "-reference_delay", "-modulation", "reference",
         "-models", "null", "superficial", "-max_iter", "2",
         "-null_cond1", *[n[0] for n in nulls], "-null_cond2", *[n[1] for n in nulls],
         "-prefix", prefix, "-workers", "1", "-verb", "0"]
    )  # fmt: skip
    assert rc == 0
    h, rows = _table(f"{prefix}_laminar_null.tsv")
    assert h[:7] == ["roi", "name", "hemi", "statistic", "real", "p_perm", "n_perm"]
    by = {r[3]: r for r in rows}
    assert "logBF_vs_null" in by and {
        f"p_modulated_{d}" for d in ("superficial", "middle", "deep")
    } <= set(by)
    for r in rows:
        assert int(r[6]) == 3, "every statistic must see every replicate"
        # (1 + #{null >= real}) / (1 + n) with n = 3: nothing below 0.25 is expressible.
        assert 0.25 - 1e-9 <= float(r[5]) <= 1.0, f"{r[3]}: p_perm {r[5]}"
    h, draws = _table(f"{prefix}_laminar_null_draws.tsv")
    assert len({r[3] for r in draws}) == 3 and len(draws) == 3 * len(rows)
    doc = json.load(open(f"{prefix}_laminar.json"))
    assert doc["null_replicates"] == 3
    # The real tables must be unchanged by the presence of a null: one line per model.
    _h, models = _table(f"{prefix}_laminar_models.tsv")
    assert [r[3] for r in models] == ["null", "superficial"]


def _fake_invert(task):
    """Finishes out of task order, which is what the pool is free to do."""
    import time

    delay, value = task
    time.sleep(delay)
    if value is None:
        raise TypeError("a failed inversion")
    return value


def _fake_summarise(job, runs):
    return job, runs


@pytest.mark.parametrize("workers", [1, 2])
def test_run_jobs_returns_each_roi_its_own_inversions_in_order(workers, monkeypatch):
    """The averaging across K groups results by POSITION, so a worker finishing early must
    not reorder them, and the permutation nulls are sliced off the output by position too."""
    bars = []
    real = laminar._inversion_bar

    def spy(total, verb):
        bars.append(real(total, verb))
        return bars[-1]

    monkeypatch.setattr(laminar, "_inversion_bar", spy)
    tasks = [[(0.6, "a0"), (0.0, "a1")], [(0.3, "b0"), (0.0, "b1"), (0.0, "b2")]]
    out = laminar._run_jobs(
        ["a", "b"], tasks, workers, 1, invert=_fake_invert, summarise=_fake_summarise
    )
    assert out == [("a", ["a0", "a1"]), ("b", ["b0", "b1", "b2"])]
    assert bars[0].n == 5


def test_run_jobs_raises_a_worker_failure():
    with pytest.raises(TypeError, match="failed inversion"):
        laminar._run_jobs(
            ["a"],
            [[(0.0, None), (0.0, "ok")]],
            2,
            0,
            invert=_fake_invert,
            summarise=_fake_summarise,
        )


def test_forward_ad_warning_is_silenced():
    """A torch upgrade rewording the message would bring the warning back in every worker."""
    import subprocess
    import sys

    code = (
        "import fastfuncstuff.laminar.integrate, torch\n"
        "torch.autograd.forward_ad._maybe_load_decompositions()\n"
    )
    run = subprocess.run(
        [sys.executable, "-W", "default", "-c", code], capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr
    assert "torch.jit.script" not in run.stderr
