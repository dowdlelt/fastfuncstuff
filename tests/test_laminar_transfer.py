"""-transfer: fit the first condition, carry its physiology, fit the second's drive.

Synthetic two-condition data with known drives per neuronal depth, generated as two
separate single-condition runs so that neither is "the modulated one".
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.cli import laminar
from fastfuncstuff.laminar.experiment import faes_priors
from fastfuncstuff.laminar.integrate import integrate
from fastfuncstuff.laminar.io import two_condition_batch
from fastfuncstuff.laminar.pipeline import reference_p0
from fastfuncstuff.laminar.transfer import fit_transfer, stage_rows, target_priors
from tests.test_laminar_io import _write_table

K, NT, DUR = 7, 16, 2.0
TIMES = np.arange(NT) * 1.0


def _single(drive, s_d=0.3):
    """One condition from rest, deep depth first as the toolbox writes curves."""
    z = np.zeros((1, NT, K))
    d = two_condition_batch(z, z, TIMES, duration=DUR, p0=reference_p0())
    P = {k: v.clone() for k, v in faes_priors(d.spec, (0, 0, 0)).pE.items()}
    P["C"][:, 1] = torch.tensor(drive, dtype=torch.float64)
    P["s_d"] = torch.tensor(s_d, dtype=torch.float64)
    y = integrate(d.u, P, d.spec, d.y.shape[1], **d.integrate_kwargs)
    return y[1 : d.meta["block"]].numpy()[:, ::-1].copy()


def _pair(drive_a, drive_b, seed=3):
    rng = np.random.default_rng(seed)
    a = _single(drive_a) + 0.03 * rng.standard_normal((NT, K))
    b = _single(drive_b) + 0.03 * rng.standard_normal((NT, K))
    return a, b


def test_the_second_stage_frees_only_the_drive():
    z = np.zeros((1, NT, K))  # no integration here: this test stays in the fast tier
    d = two_condition_batch(z, z, TIMES, duration=DUR, p0=reference_p0())
    rows_a, rows_b = stage_rows(d.rows)
    assert int(rows_a.sum()) == NT and int(rows_b.sum()) == NT and not (rows_a & rows_b).any()
    source = {k: v + 0.123 for k, v in faes_priors(d.spec, (1, 1, 1)).pE.items()}
    pr = target_priors(d.spec, source)
    free = {k for k, v in pr.pC.items() if float(v.abs().sum()) > 0}
    assert free == {"C"}
    # carried from the first fit, except the drive, which restarts from the neutral prior
    assert float(pr.pE["s_d"]) == pytest.approx(0.123)
    assert pr.pE["C"][:, 1].tolist() == [1.0, 1.0, 1.0]


@pytest.mark.slow  # 15s
def test_a_moved_drive_profile_is_recovered_as_drive():
    """Input moved from deep to superficial at the same overall size. The published
    model space can only call this a gain change (it answered B = +2.7, -0.9, -13 on
    the same data); here each condition's drive comes back within its interval."""
    true_a, true_b = [0.1, 0.3, 0.8], [0.8, 0.3, 0.1]
    a, b = _pair(true_a, true_b)
    d = two_condition_batch(a[None], b[None], TIMES, duration=DUR, p0=reference_p0())
    d.y = d.y[0]
    fit = fit_transfer({K: d}, rescue=2)
    np.testing.assert_allclose(fit.drive[0], true_a, atol=0.1)
    np.testing.assert_allclose(fit.drive[1], true_b, atol=0.1)
    np.testing.assert_allclose(fit.difference, np.subtract(true_b, true_a), atol=0.12)
    assert (fit.r2 > 0.98).all()
    # the second condition's interval is conditional on the first's physiology
    assert (fit.drive_sd[1] < fit.drive_sd[0] + 1e-9).all()


@pytest.mark.slow  # 9s after the file's ~13s compile warm-up, which it pays if run alone
def test_the_cli_writes_the_transfer_table_and_refuses_what_does_not_apply(tmp_path, capsys):
    a, b = _pair([0.4, 0.4, 0.4], [0.8, 0.4, 0.4])
    ta = _write_table(tmp_path / "k7_iresp_a.tsv", TIMES, a)
    tb = _write_table(tmp_path / "k7_iresp_b.tsv", TIMES, b)
    base = ["-cond1", str(ta), "-cond2", str(tb), "-duration", str(DUR), "-workers", "1",
            "-verb", "0", "-transfer"]  # fmt: skip
    assert laminar.main([*base, "-models", "null", "-prefix", str(tmp_path / "no")]) == 1
    assert "-models" in capsys.readouterr().err
    prefix = str(tmp_path / "tr")
    assert laminar.main([*base, "-rescue", "2", "-prefix", prefix]) == 0
    lines = [ln.split("\t") for ln in open(f"{prefix}_laminar_transfer.tsv").read().splitlines()]
    head, rows = lines[0], lines[1:]
    assert [r[head.index("neuronal_depth")] for r in rows] == ["superficial", "middle", "deep"]
    diff = [float(r[head.index("difference")]) for r in rows]
    assert diff[0] == pytest.approx(0.4, abs=0.12)
    assert abs(diff[1]) < 0.12 and abs(diff[2]) < 0.12
    assert not (tmp_path / "tr_laminar_models.tsv").exists()  # no model comparison is run
