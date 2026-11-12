"""ffs_group end to end on tiny synthetic files (CPU)."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
from scipy import stats

from fastfuncstuff.cli.group import main, parse_set, short_label
from fastfuncstuff.io.afni import save_nifti
from fastfuncstuff.io.headers import read_brick_labels, read_brick_stataux


def _write_sets(tmp_path, na=6, nb=5, shape=(6, 5, 4)):
    rng = np.random.default_rng(0)
    a, b = [], []
    for grp, n, lst in (("a", na, a), ("b", nb, b)):
        for i in range(n):
            p = tmp_path / f"{grp}{i}.nii.gz"
            # Two sub-bricks so selectors are exercised; brick 1 is the data.
            vol = np.stack([np.zeros(shape), rng.standard_normal(shape) + (grp == "a")], -1)
            save_nifti(vol.astype(np.float32), p, affine=np.eye(4), brick_labels=["junk", "beta"])
            lst.append(p)
    return a, b


def test_two_sample_with_label_selector_and_covariates(tmp_path):
    a, b = _write_sets(tmp_path)
    cov = tmp_path / "cov.txt"
    rows = [f"s{k} {k * 1.5:.2f}" for k in range(11)]
    cov.write_text("subj age\n" + "\n".join(rows) + "\nextra 9\n")
    seta = ["A"] + [x for k, p in enumerate(a) for x in (f"s{k}", f"{p}[beta]")]
    setb = ["B"] + [x for k, p in enumerate(b) for x in (f"s{k + 6}", f"{p}[1]")]
    out = tmp_path / "tt.nii.gz"
    main(["-setA", *seta, "-setB", *setb, "-covariates", str(cov), "-prefix", str(out),
          "-device", "cpu", "-verb", "0"])  # fmt: skip
    img = nib.load(out)
    labs = read_brick_labels(img)
    assert labs[:4] == ["A-B_mean", "A-B_Tstat", "A-B_age", "A-B_age_Tstat"]
    assert read_brick_stataux(img)[1] == (3, (7.0,))  # 11 - 2 - 2*1


def test_short_form_one_sample_matches_scipy(tmp_path):
    a, _ = _write_sets(tmp_path)
    out = tmp_path / "one.nii.gz"
    main(["-setA", *[f"{p}[1]" for p in a], "-labelA", "vis", "-prefix", str(out),
          "-device", "cpu", "-verb", "0"])  # fmt: skip
    img = nib.load(out)
    assert read_brick_labels(img) == ["vis_mean", "vis_Tstat"]
    data = np.stack([np.asarray(nib.load(p).dataobj)[..., 1] for p in a])
    np.testing.assert_allclose(
        np.asarray(img.dataobj)[..., 1], stats.ttest_1samp(data, 0).statistic, rtol=1e-4
    )


def test_set_parsing_rules(tmp_path):
    a, _ = _write_sets(tmp_path, na=2, nb=0)
    assert short_label("/x/Elvis_Presley_long.nii.gz[3]") == "Elvis_Presle"
    lab, subj, ds = parse_set([str(p) for p in a], "SetA", None)
    assert (lab, subj) == ("SetA", ["a0", "a1"])
    with pytest.raises(ValueError, match="long form"):
        parse_set(["G", "s0", str(a[0]), "s1"], "SetA", None)
    with pytest.raises(ValueError, match="drop -labelA"):
        parse_set(["G", "s0", str(a[0])], "SetA", "X")


def test_missing_covariate_row_is_an_error(tmp_path, capsys):
    a, _ = _write_sets(tmp_path, nb=0)
    cov = tmp_path / "cov.txt"
    cov.write_text("subj age\na0 1\n")
    with pytest.raises(SystemExit):
        main(["-setA", *[f"{p}[1]" for p in a], "-covariates", str(cov), "-prefix",
              str(tmp_path / "x"), "-device", "cpu"])  # fmt: skip
    assert "no row for subject" in capsys.readouterr().err


def test_mema_short_form_beta_t_pairs(tmp_path):
    rng = np.random.default_rng(1)
    shape = (5, 4, 3)
    specs = []
    for i in range(7):
        b = rng.normal(0.5, 1.0, shape)
        t = b / rng.uniform(0.3, 1.0, shape)
        p = tmp_path / f"s{i}.nii.gz"
        save_nifti(np.stack([b, t], -1).astype(np.float32), p, affine=np.eye(4),
                   brick_labels=["c#0_Coef", "c#0_Tstat"])  # fmt: skip
        specs += [f"{p}[c#0_Coef]", f"{p}[c#0_Tstat]"]
    out = tmp_path / "mema.nii.gz"
    main(["-model", "mema", "-setA", *specs, "-labelA", "c", "-prefix", str(out),
          "-device", "cpu", "-verb", "0"])  # fmt: skip
    img = nib.load(out)
    assert read_brick_labels(img) == ["c:b", "c:t", "I2", "tau", "QE:Chisq"]
    assert read_brick_stataux(img)[1] == (3, (6.0,))


def test_mema_only_flags_are_refused_for_ols(tmp_path, capsys):
    a, _ = _write_sets(tmp_path, nb=0)
    with pytest.raises(SystemExit):
        main(["-setA", *[f"{p}[1]" for p in a], "-no_HKtest", "-prefix", str(tmp_path / "x"),
              "-device", "cpu"])  # fmt: skip
    assert "-model mema only" in capsys.readouterr().err
